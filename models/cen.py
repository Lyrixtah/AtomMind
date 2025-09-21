"""
Defines the ChatExpertNetwork (CEN)
A transformer-based expert network for processing chat sequences.
"""

from torch import nn
from torch import Tensor
from config import HIDDEN_SIZE, CEN_LAYERS, NUM_ATTENTION_HEADS

class ChatExpertNetwork(nn.Module):
    """
    ChatExpertNetwork applies a stack of 
    TransformerEncoder layers to input chat embeddings.

    Attributes:
        transformer (nn.TransformerEncoder): Transformer encoder stack.
    """

    def __init__(self) -> None:
        """
        Initialize the ChatExpertNetwork with a multi-layer TransformerEncoder.
        """
        super().__init__()  # Python 3 style super()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_SIZE,
            nhead=NUM_ATTENTION_HEADS,
            dim_feedforward=HIDDEN_SIZE * 4,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=CEN_LAYERS)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the transformer.

        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, hidden_size].
        Returns:
            Tensor: Output tensor of shape [batch, seq_len, hidden_size].
        """
        # Transformer expects [seq_len, batch, hidden_size]
        x = x.transpose(0, 1)
        out = self.transformer(x)
        return out.transpose(0, 1)
