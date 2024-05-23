import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from bitnet.bit_ffn import BitFeedForward
from bitnet.bit_attention import BitAttention
from bitnet.bitlinear import BitLinear


def l2norm(t, dim=-1):
    return F.normalize(t, dim=dim)


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) module.

    Args:
        dim (int): The input dimension.
        affine (bool, optional): If True, apply an affine transformation to the normalized output.
            Default is True.

    Attributes:
        scale (float): The scaling factor for the normalized output.
        gamma (torch.Tensor or float): The learnable parameter for the affine transformation.

    """

    def __init__(self, dim, affine=True):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.0

    def forward(self, x):
        return l2norm(x) * self.gamma * self.scale


class Transformer(nn.Module):
    """
    Transformer module that applies multi-head attention and feed-forward layers.

    Args:
        dim (int): The dimension of the input and output tensors.
        heads (int): The number of attention heads.
        depth (int): The number of transformer layers.
        ff_mult (int, optional): The multiplier for the hidden dimension in the feed-forward layers.
            Defaults to 2.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        layers (nn.ModuleList): List of multi-head attention layers.
        ffn_layers (nn.ModuleList): List of feed-forward layers.

    """

    def __init__(
        self, dim: int, heads: int, depth: int = 1, ff_mult: int = 4, mode="binary", *args, **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(BitAttention(dim, heads,mode=mode, *args, **kwargs))

            self.ffn_layers.append(
                BitFeedForward(
                    dim,
                    dim,
                    ff_mult,
                    mode=mode,
                ),
            )
        self.cache = {}  # Cache for attention and feed-forward layers


    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        skip = x
        for attn, ffn in zip(self.layers, self.ffn_layers):
            x = attn(x,*args, **kwargs)
            x = x + skip
            x = ffn(x) + x

            # Update the cache with the caches from the attention and feed-forward layers
            self.cache.update(attn.attn_cache)
            self.cache.update(ffn.activation_cache)

        return x


# [MAIN MODEL] BitNetTransformer
class BitNetTransformer(nn.Module):
    """
    BitNetTransformer is a transformer-based model for BitNet.

    Args:
        dim (int): The dimension of the token embeddings.
        depth (int): The number of transformer layers.
        num_tokens (int): The number of tokens in the vocabulary.
        heads (int, optional): The number of attention heads in the transformer. Defaults to 8.
        ff_mult (int, optional): The multiplier for the feed-forward layer dimension. Defaults to 4.

    Examples:
    >>> import torch
    >>> from bitnet import BitNetTransformer
    >>> x = torch.randint(0, 20000, (1, 1024))
    >>> bitnet = BitNetTransformer(
    ...     num_tokens=20000,
    ...     dim=1024,
    ...     depth=6,
    ...     heads=8,
    ...     ff_mult=4,
    ... )
    >>> logits = bitnet(x)
    >>> print(logits)
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        in_features: int,
        out_features: int,
        heads=4,
        ff_mult=4,
        random_seed=None,
        mode="binary",
    ):
        super().__init__()
        # Set the random seed for PyTorch if a seed is provided
        if random_seed is not None:
            torch.cuda.manual_seed_all(random_seed)

        self.emb = nn.Embedding(in_features, dim)

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, ff_mult=ff_mult, mode=mode
        )

        self.to_logits = nn.Sequential(RMSNorm(dim), nn.Linear(dim, out_features, bias=False))
        

        self.cache = {}  # Cache for transformer layer


    def forward(self, x):
        x = self.emb(x)
        x = self.transformer(x)
        # Update the cache with the cache from the transformer layer
        self.cache.update(self.transformer.cache)

        return self.to_logits(x)
