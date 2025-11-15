# src/model.py
import torch.nn as nn
import torch

class TimeSeriesTransformer(nn.Module):
    """
    Small Transformer that outputs logits (no final sigmoid).
    Use BCEWithLogitsLoss in training.
    """
    def __init__(self, n_features, seq_len=48, d_model=128, n_heads=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_proj(x)
        if x.size(1) <= self.pos_emb.size(1):
            x = x + self.pos_emb[:, :x.size(1), :]
        else:
            x = x + self.pos_emb[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # pooled representation
        out = self.classifier(x)  # logits
        return out.squeeze(-1)
