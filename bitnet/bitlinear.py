import torch
from torch import Tensor, nn

# binary quantization
class BitLinearBinary(nn.Linear):
    """
    BitLinear is a custom linear layer that performs binarization of weights and quantization of activations
    in a group-wise manner.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        num_groups (int, optional): Number of groups to divide the weights and activations into. Default is 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        num_groups: int = 1,
        b: int = 8,
    ):
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.b = b
        self.num_groups = num_groups
        self.eps = 1e-5
        # self.norm = nn.LayerNorm(in_features)

    def ste(self, x):
        """
        Applies the sign function for binarization and uses Straight-Through Estimator (STE) during backward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Binarized tensor.
        """
        binarized_x = torch.sign(x)
        binarized_x = (binarized_x - x).detach() + x
        return binarized_x

    def binarize_weights_groupwise(self):
        """
        Binarizes the weights of the layer in a group-wise manner using STE.

        Returns:
            Tensor: Binarized weights tensor.
        """
        group_size = self.weight.shape[0] // self.num_groups
        binarized_weights = torch.zeros_like(self.weight)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = self.weight[start_idx:end_idx]

            alpha_g = weight_group.mean()
            binarized_weights[start_idx:end_idx] = self.ste(weight_group - alpha_g)

        return binarized_weights

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # # Normalize input
        # x = self.norm(x)

        # Binarize weights and quantize activations
        binarized_weights = self.binarize_weights_groupwise()

        # Perform linear transformation
        output = torch.nn.functional.linear(x, binarized_weights, self.bias)

        # # Quantize activations
        # output = self.quantize_activations_groupwise(output)

        # # Dequantize activations
        # output = self.dequantize_activations_groupwise(output)

        # Return output
        return output


def absmean_quantize_weights(weights):
    """
    Quantizes the weights to -1, 0, or +1 using an absmean quantization function.

    Parameters:
    - weights (Tensor): The weights of a neural network layer.

    Returns:
    - Tensor: The quantized weights.
    """
    # Calculate the average absolute value (γ) of the weights
    gamma = torch.mean(torch.abs(weights))

    # Scale weights by γ and round to the nearest integer among {-1, 0, +1}
    quantized_weights = torch.clamp(torch.round(weights / gamma), min=-1, max=1)
    # quantized_weights = torch.clamp(torch.round(weights / (gamma + 1e-5)), min=-1, max=1)

    return quantized_weights

def weight_quant(w):
        """ Per−tensor quantization to 1.58 bits. No grouping is needed for quantization.
        Args:
        w: a weight tensor with shape [d, k]
        Returns:
        u: a quantized weight with shape [d, k]
        """
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        u = (w * scale).round().clamp_(-1, 1) / scale
        return u

# ternary quantization
class BitLinearTernary(nn.Linear):
    """
    BitLinear is a custom linear layer that performs binarization of weights and quantization of activations
    in a group-wise manner.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        num_groups (int, optional): Number of groups to divide the weights and activations into. Default is 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        num_groups: int = 1,
        b: int = 8,
    ):
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.b = b
        self.num_groups = num_groups
        self.eps = 1e-5
        self.norm = nn.LayerNorm(in_features)

    def ste(self, x):
        """
        Applies the sign function for binarization and uses Straight-Through Estimator (STE) during backward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Binarized tensor.
        """
        ternarized_x = weight_quant(x)
        ternarized_x = (ternarized_x - x).detach() + x
        return ternarized_x

    def binarize_weights_groupwise(self):
        """
        Binarizes the weights of the layer in a group-wise manner using STE.

        Returns:
            Tensor: Binarized weights tensor.
        """
        group_size = self.weight.shape[0] // self.num_groups
        binarized_weights = torch.zeros_like(self.weight)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = self.weight[start_idx:end_idx]

            alpha_g = weight_group.mean()
            binarized_weights[start_idx:end_idx] = self.ste(weight_group - alpha_g)

        return binarized_weights

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        
        # Binarize weights and quantize activations
        binarized_weights = self.binarize_weights_groupwise()

        # Perform linear transformation
        output = torch.nn.functional.linear(x, binarized_weights, self.bias)

        

        # Dequantize activations
        # output = self.dequantize_activations_groupwise(output)

        # Return output
        return output
    

class BitLinear(nn.Linear):
    """
    BitLinear is a custom linear layer that performs binarization of weights and quantization of activations
    in a group-wise manner. It can operate in either binary or ternary mode.

    Args:
        mode (str): The mode of operation. Either 'binary' or 'ternary'.
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        num_groups (int, optional): Number of groups to divide the weights and activations into. Default is 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        num_groups: int = 1,
        b: int = 8,
        mode: str = 'ternary',
    ):
        super().__init__(in_features, out_features, bias)
        self.num_groups = num_groups
        self.b = b
        if mode == 'binary':
            self.layer = BitLinearBinary(in_features, out_features, bias, num_groups, b)
        elif mode == 'ternary':
            self.layer = BitLinearTernary(in_features, out_features, bias, num_groups, b)
        else:
            raise ValueError("Mode must be either 'binary' or 'ternary'.")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.layer.forward(x)