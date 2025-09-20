"""
Defines the GeneralKnowledgeBackbone (GKB), a transformer-based module for general knowledge representation.
"""

from typing import List
from torch import nn, Tensor
import torch
from models.transformer import TransformerBlock

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
        self.transformer = TransformerBlock(hidden_size, num_heads, num_layers)

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
