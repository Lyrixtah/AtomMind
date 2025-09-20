"""
Defines the GeneralKnowledgeBackbone (GKB), a transformer-based module for general knowledge representation.
"""

from torch import nn, Tensor
import torch
from typing import List

class GeneralKnowledgeBackbone(nn.Module):
    """
    GeneralKnowledgeBackbone applies a TransformerEncoder to aggregate multiple domain outputs.

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

    def forward(self, domain_outputs: List[Tensor]) -> Tensor:
        """
        Forward pass through the transformer.

        Args:
            domain_outputs (List[Tensor]): List of tensors, each of shape [batch, seq_len, hidden_size].
        Returns:
            Tensor: Aggregated tensor of shape [batch, total_seq_len, hidden_size].
        """
        x = torch.cat(domain_outputs, dim=1)
        return self.transformer(x)
