# src/model.py
import math
import logging
import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)


def _sinusoidal_pos_enc(seq_len: int, d_model: int) -> torch.Tensor:
    """Return (1, seq_len, d_model) sinusoidal positional encoding."""
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even for sinusoidal encoding, got {d_model}")
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, T, d_model)


class TimeSeriesTransformer(nn.Module):
    """
    Transformer encoder for clinical time-series classification.

    Improvements over the original:
    - CLS token pooling instead of mean pooling — the CLS token attends to all
      timesteps and produces a global representation without diluting signal from
      padded positions.
    - Sinusoidal positional embedding initialisation — better gradient flow early
      in training than all-zero init.
    - src_key_padding_mask support — front-padded positions are masked out so the
      attention mechanism ignores them. Pass pad_mask from PatientDataset.
    """

    def __init__(
        self,
        n_features,
        seq_len=48,
        d_model=128,
        n_heads=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable CLS token prepended to the sequence before the encoder.
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional embedding: seq_len + 1 to account for the CLS position.
        # Initialised with sinusoidal values; remains trainable.
        pos_enc = _sinusoidal_pos_enc(seq_len + 1, d_model)
        self.pos_emb = nn.Parameter(pos_enc)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, src_key_padding_mask=None):
        """
        x:                    (B, T, F)
        src_key_padding_mask: (B, T) bool — True at positions to ignore (front-padding).
        Returns logits: (B,)
        """
        B, T, F = x.shape
        x = self.input_proj(x)  # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)           # (B, T+1, d_model)

        # Positional embedding — slice to actual length in case seq_len differs
        seq_total = x.size(1)
        P = self.pos_emb.size(1)
        if seq_total <= P:
            x = x + self.pos_emb[:, :seq_total, :]
        else:
            _logger.warning(
                "Input sequence length %d exceeds trained pos_emb length %d; "
                "tokens beyond position %d receive no positional encoding.",
                seq_total, P, P,
            )
            x = torch.cat(
                [x[:, :P, :] + self.pos_emb, x[:, P:, :]], dim=1
            )

        # Extend padding mask to cover the CLS position (always attend to it).
        if src_key_padding_mask is not None:
            cls_col = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_col, src_key_padding_mask.to(x.device)], dim=1)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Use CLS token output for classification
        out = self.classifier(x[:, 0, :])
        return out.squeeze(-1)  # (B,)
