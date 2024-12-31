import torch
import torch.nn as nn


class MaskedTransformer(nn.Module):
    def __init__(self, input_dim=5, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.input_embed = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 5, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.ModuleList(
            [
                nn.Linear(embed_dim, 1),
                nn.Linear(embed_dim, 1),
                nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid()),
                nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid()),
                nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid()),
            ]
        )

    def forward(self, inputs):
        seq_len = inputs.shape[1]
        x = self.input_embed(inputs) + self.pos_encoding[:, :seq_len, :]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=inputs.device) * float("-inf"),
            diagonal=1,
        )
        encoded = self.transformer_encoder(x, mask=mask)
        return [head(encoded[:, i]) for i, head in enumerate(self.fc_out)]
