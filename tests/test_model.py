"""
Unit tests for model architectures.
Run from project root: python -m pytest tests/
"""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import TimeSeriesTransformer
from model_grud import GRUD


class TestTimeSeriesTransformer:
    def test_output_shape_standard(self):
        model = TimeSeriesTransformer(n_features=10, seq_len=16)
        x = torch.randn(4, 16, 10)
        out = model(x)
        assert out.shape == (4,), f"Expected (4,), got {out.shape}"

    def test_output_shape_single_sample(self):
        model = TimeSeriesTransformer(n_features=40, seq_len=48)
        x = torch.randn(1, 48, 40)
        out = model(x)
        assert out.shape == (1,)

    def test_sequence_longer_than_pos_emb(self):
        """Model must handle inputs longer than the trained positional embedding."""
        model = TimeSeriesTransformer(n_features=10, seq_len=16)
        x = torch.randn(2, 32, 10)  # 2× longer than trained seq_len
        out = model(x)
        assert out.shape == (2,)

    def test_sequence_shorter_than_pos_emb(self):
        model = TimeSeriesTransformer(n_features=10, seq_len=48)
        x = torch.randn(2, 10, 10)
        out = model(x)
        assert out.shape == (2,)

    def test_output_is_finite(self):
        model = TimeSeriesTransformer(n_features=40, seq_len=48)
        x = torch.randn(8, 48, 40)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self):
        model = TimeSeriesTransformer(n_features=10, seq_len=16)
        x = torch.randn(2, 16, 10)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestGRUD:
    def test_output_shape(self):
        model = GRUD(n_features=10)
        B, T, F = 3, 16, 10
        x = torch.randn(B, T, F)
        mask = torch.ones(B, T, F)
        deltas = torch.zeros(B, T, F)
        out = model(x, mask, deltas)
        assert out.shape == (B,), f"Expected ({B},), got {out.shape}"

    def test_output_shape_full_features(self):
        model = GRUD(n_features=40)
        x = torch.randn(2, 48, 40)
        mask = torch.ones(2, 48, 40)
        deltas = torch.zeros(2, 48, 40)
        out = model(x, mask, deltas)
        assert out.shape == (2,)

    def test_with_missing_data(self):
        """Masked (missing) features should not cause NaN outputs."""
        model = GRUD(n_features=10)
        x = torch.randn(2, 16, 10)
        mask = torch.zeros(2, 16, 10)  # all features missing
        mask[:, :, :5] = 1.0            # first 5 observed
        deltas = torch.rand(2, 16, 10) * 5
        out = model(x, mask, deltas)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self):
        model = GRUD(n_features=10)
        x = torch.randn(2, 16, 10)
        mask = torch.ones(2, 16, 10)
        deltas = torch.zeros(2, 16, 10)
        out = model(x, mask, deltas)
        out.sum().backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
