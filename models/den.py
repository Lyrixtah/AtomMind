"""
Defines the DomainExpertNetwork (DEN) using a Transformer encoder for domain-specific embeddings.
"""

from torch import nn, Tensor
from models.transformer import TransformerBlock

class DomainExpertNetwork(nn.Module):
    """
    DomainExpertNetwork applies a TransformerEncoder to domain-specific input embeddings.

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

        Args: x (Tensor):
            Input tensor of shape [batch, seq_len, hidden_size].
        Returns:
            Tensor: Output tensor of shape [batch, seq_len, hidden_size].
        """
        return self.transformer(x)
