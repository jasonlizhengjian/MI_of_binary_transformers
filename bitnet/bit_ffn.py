from typing import Callable, Optional

import torch
from torch import nn, Tensor

from bitnet.bitlinear import BitLinear


def default(val, d):
    return val if val is not None else d


def init_zero_(tensor):
    nn.init.constant_(tensor, 0.0)


# [GLU]
class GLU(nn.Module):
    """
    Gated Linear Unit (GLU) module.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        activation (Callable): Activation function to be applied to the gate.
        mult_bias (bool, optional): Whether to multiply the bias term. Defaults to False.
        linear (Callable, optional): Linear function to be used for projection. Defaults to False.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: Callable,
        mult_bias: bool = False,
        linear: Callable = False,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation
        self.mult_bias = mult_bias

        if linear:
            self.proj = linear(dim_in, dim_out * 2)
        else:
            self.proj = BitLinear(dim_in, dim_out * 4, *args, **kwargs)

        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.0

    def forward(self, x: Tensor):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.activation(gate) * self.mult_bias


# [FEATURE] Add type hints to the forward method
class BitFeedForward(nn.Module):
    """
    BitFeedForward module performs feed-forward operations on the input tensor.

    Args:
        dim (int): The input dimension.
        dim_out (int, optional): The output dimension. If not provided, it is set to the input dimension.
        mult (int, optional): The multiplier for the inner dimension. Default is 4.
        glu (bool, optional): Whether to use Gated Linear Unit (GLU) activation. Default is False.
        glu_mult_bias (bool, optional): Whether to apply bias to the GLU activation. Default is False.
        swish (bool, optional): Whether to use Swish activation. Default is False.
        relu_squared (bool, optional): Whether to use squared ReLU activation. Default is False.
        post_act_ln (bool, optional): Whether to apply Layer Normalization after activation. Default is False.
        dropout (float, optional): The dropout probability. Default is 0.0.
        no_bias (bool, optional): Whether to exclude bias in linear layers. Default is False.
        zero_init_output (bool, optional): Whether to initialize the last linear layer to 0. Default is False.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        glu: bool = False,
        glu_mult_bias: bool = False,
        swish: bool = False,
        post_act_ln: bool = False,
        dropout: float = 0.0,
        no_bias: bool = True,
        zero_init_output: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        activation = nn.ReLU()

        self.layer1 = BitLinear(dim, inner_dim, bias=False, *args, **kwargs)
        self.activation = activation
        self.layer2 = BitLinear(inner_dim, dim_out, bias=False, *args, **kwargs)

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.layer2)

        self.activation_cache = {}  # Cache for pre and post activation tensors


    def forward(self, x):
        """
        Forward pass of the BitFeedForward module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Apply the first linear layer to the input tensor 'x'
        linear_output = self.layer1(x)

        # Detach the output tensor from the computation graph and store it in the cache
        # This tensor will not have a gradient during backpropagation
        # Include the layer name in the key used to store the tensor in the cache
        self.activation_cache[f'pre_activation_{self.layer1.__class__.__name__}'] = linear_output.detach()

        # Apply the activation function to the output of the first linear layer
        # This tensor is not detached, so it will have a gradient during backpropagation
        activated_output = self.activation(linear_output)

        # Detach the activated output tensor from the computation graph and store it in the cache
        # This tensor will not have a gradient during backpropagation
        # Include the layer name in the key used to store the tensor in the cache
        self.activation_cache[f'post_activation_{self.layer1.__class__.__name__}'] = activated_output.detach()

        # Apply the second linear layer to the activated output tensor
        # The result of this operation will be returned by the forward method
        output = self.layer2(activated_output)

        # Return the output tensor
        return output
