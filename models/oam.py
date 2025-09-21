"""
Defines the OptimizationAlgorithmModule (OAM)
A transformer-based module for optimization reasoning.
"""

from torch import nn, Tensor
from models.transformer import TransformerBlock

class OptimizationAlgorithmModule(nn.Module):
    """
    OptimizationAlgorithmModule applies a TransformerEncoder to perform optimization reasoning.

    Args:
        hidden_size (int): Dimensionality of input embeddings.
        num_layers (int): Number of Transformer encoder layers.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, hidden_size: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        self.transformer = TransformerBlock(hidden_size, num_heads, num_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the transformer.

        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, hidden_size].

        Returns:
            Tensor: Output tensor of same shape as input.
        """
        return self.transformer(x)
