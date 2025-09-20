import torch
import torch.nn as nn
from config import HIDDEN_SIZE, CEN_LAYERS, NUM_ATTENTION_HEADS

class ChatExpertNetwork(nn.Module):
    def __init__(self):
        super(ChatExpertNetwork, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_SIZE,
            nhead=NUM_ATTENTION_HEADS,
            dim_feedforward=HIDDEN_SIZE * 4,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=CEN_LAYERS)
    
    def forward(self, x):
        # x: [batch, seq_len, hidden_size] -> transformer expects [seq_len, batch, hidden_size]
        x = x.transpose(0, 1)
        out = self.transformer(x)
        return out.transpose(0, 1)
