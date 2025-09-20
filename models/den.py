"""
Defines the DomainExpertNetwork (DEN) using a Transformer encoder for domain-specific embeddings.
"""

from torch import nn, Tensor

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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True  # input/output: [batch, seq_len, hidden_size]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the transformer.

        Args: x (Tensor):
            Input tensor of shape [batch, seq_len, hidden_size].
        Returns:
            Tensor: Output tensor of shape [batch, seq_len, hidden_size].
        """
        return self.transformer(x)
