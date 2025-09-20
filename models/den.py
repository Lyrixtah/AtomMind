import torch
import torch.nn as nn

class DomainExpertNetwork(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)
