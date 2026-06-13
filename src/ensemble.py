# src/ensemble.py
"""
Ensemble inference combining Transformer and GRU-D predictions.

Loads both checkpoints and blends probabilities at inference time.
GRU-D is stronger on sparse/missing-heavy patients; Transformer on dense data.
The default 50/50 blend is a free performance gain over either model alone.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from model import TimeSeriesTransformer
from model_grud import GRUD


class EnsembleModel(nn.Module):
    """
    Combines a Transformer and a GRU-D by averaging their output probabilities.
    alpha controls the Transformer weight (1-alpha for GRU-D).
    """

    def __init__(
        self,
        transformer: TimeSeriesTransformer,
        grud: GRUD,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.transformer = transformer
        self.grud = grud
        self.alpha = alpha

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        deltas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns blended probability (B,) — already passed through sigmoid.
        Caller must pass mask and deltas for GRU-D.
        """
        logit_t = self.transformer(x, src_key_padding_mask=pad_mask)
        prob_t = torch.sigmoid(logit_t)

        if mask is None:
            mask = torch.ones_like(x)
        if deltas is None:
            deltas = torch.zeros_like(x)
        logit_g = self.grud(x, mask, deltas)
        prob_g = torch.sigmoid(logit_g)

        return self.alpha * prob_t + (1.0 - self.alpha) * prob_g


def load_ensemble(
    transformer_ckpt: str,
    grud_ckpt: str,
    n_features: int = 40,
    seq_len: int = 48,
    alpha: float = 0.5,
    device: str = "cpu",
) -> EnsembleModel:
    transformer = TimeSeriesTransformer(n_features=n_features, seq_len=seq_len)
    _load_state(transformer, transformer_ckpt)
    transformer.eval()

    grud = GRUD(n_features=n_features)
    _load_state(grud, grud_ckpt)
    grud.eval()

    model = EnsembleModel(transformer, grud, alpha=alpha).to(device)
    model.eval()
    return model


def _load_state(model: nn.Module, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state" in sd:
        sd = sd["model_state"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
