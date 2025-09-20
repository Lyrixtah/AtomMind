"""
Defines the SymbolicReasoningModule (SRM) for reasoning over
knowledge representations in the SmallScientificLLM.
"""

from torch import nn
from torch import Tensor


class SymbolicReasoningModule(nn.Module):
    """
    SymbolicReasoningModule applies transformer-based symbolic reasoning
    over input feature representations.

    Args:
        hidden_size (int): Hidden size of input and transformer layers.
        num_layers (int): Number of transformer encoder layers.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, hidden_size: int, num_layers: int, num_heads: int) -> None:
        """
        Initialize the SymbolicReasoningModule.
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the SymbolicReasoningModule.

        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, hidden_size].

        Returns:
            Tensor: Output tensor after symbolic reasoning, same shape as input.
        """
        return self.transformer(x)
