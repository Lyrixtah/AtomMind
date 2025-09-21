"""Defines a reusable Transformer block for sequence processing."""

from torch import nn
from torch import Tensor

class TransformerBlock(nn.Module):
    """
    TransformerBlock wraps a stack of TransformerEncoder layers.

    Args:
        hidden_size (int): Dimensionality of input embeddings.
        num_heads (int): Number of attention heads per layer.
        num_layers (int): Number of TransformerEncoder layers.
    """

    def __init__(self, hidden_size: int, num_heads: int, num_layers: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Transformer block.

        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, hidden_size].
        
        Returns:
            Tensor: Output tensor of same shape as input.
        """
        return self.transformer(x)
