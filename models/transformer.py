from torch import nn, Tensor

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int):
        """
        Initializes the TransformerBlock.

        Args:
            hidden_size (int): The number of expected features in the input.
            num_heads (int): The number of heads in the multiheadattention models.
            num_layers (int): The number of sub-encoder-layers in the encoder.
        """
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
        return self.transformer(x)
