import torch
import torch.nn as nn

class GeneralKnowledgeBackbone(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, domain_outputs):
        x = torch.cat(domain_outputs, dim=1)
        return self.transformer(x)
