"""Tests for src/ensemble.py (EnsembleModel forward pass)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import TimeSeriesTransformer
from model_grud import GRUD
from ensemble import EnsembleModel

N_FEATURES = 10
SEQ_LEN = 12
BATCH = 4


@pytest.fixture
def models():
    transformer = TimeSeriesTransformer(n_features=N_FEATURES, seq_len=SEQ_LEN)
    grud = GRUD(n_features=N_FEATURES)
    transformer.eval()
    grud.eval()
    return transformer, grud


@pytest.fixture
def ensemble(models):
    transformer, grud = models
    return EnsembleModel(transformer, grud, alpha=0.5)


class TestEnsembleModelForward:
    def test_output_shape(self, ensemble):
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        mask = torch.ones(BATCH, SEQ_LEN, N_FEATURES)
        deltas = torch.zeros(BATCH, SEQ_LEN, N_FEATURES)
        with torch.no_grad():
            out = ensemble(x, mask=mask, deltas=deltas)
        assert out.shape == (BATCH,)

    def test_output_is_probability(self, ensemble):
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        with torch.no_grad():
            out = ensemble(x)
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0)

    def test_output_finite(self, ensemble):
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        with torch.no_grad():
            out = ensemble(x)
        assert torch.isfinite(out).all()

    def test_alpha_zero_uses_only_grud(self, models):
        transformer, grud = models
        ens_alpha0 = EnsembleModel(transformer, grud, alpha=0.0)
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        mask = torch.ones(BATCH, SEQ_LEN, N_FEATURES)
        deltas = torch.zeros(BATCH, SEQ_LEN, N_FEATURES)
        with torch.no_grad():
            out = ens_alpha0(x, mask=mask, deltas=deltas)
            grud_logit = grud(x, mask, deltas)
            grud_prob = torch.sigmoid(grud_logit)
        assert torch.allclose(out, grud_prob, atol=1e-5)

    def test_alpha_one_uses_only_transformer(self, models):
        transformer, grud = models
        ens_alpha1 = EnsembleModel(transformer, grud, alpha=1.0)
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        pad_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
        with torch.no_grad():
            out = ens_alpha1(x, pad_mask=pad_mask)
            trans_logit = transformer(x, src_key_padding_mask=pad_mask)
            trans_prob = torch.sigmoid(trans_logit)
        assert torch.allclose(out, trans_prob, atol=1e-5)

    def test_none_mask_defaults_to_ones(self, ensemble):
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        with torch.no_grad():
            out_none = ensemble(x, mask=None, deltas=None)
        assert out_none.shape == (BATCH,)

    def test_gradient_flows(self, ensemble):
        ensemble.train()
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES, requires_grad=True)
        out = ensemble(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_alpha_blending(self, models):
        transformer, grud = models
        ens = EnsembleModel(transformer, grud, alpha=0.7)
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        mask = torch.ones(BATCH, SEQ_LEN, N_FEATURES)
        deltas = torch.zeros(BATCH, SEQ_LEN, N_FEATURES)
        with torch.no_grad():
            out = ens(x, mask=mask, deltas=deltas)
            trans_prob = torch.sigmoid(transformer(x))
            grud_prob = torch.sigmoid(grud(x, mask, deltas))
            expected = 0.7 * trans_prob + 0.3 * grud_prob
        assert torch.allclose(out, expected, atol=1e-5)
