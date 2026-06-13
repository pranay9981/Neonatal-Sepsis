"""
Tests for the training loop (src/train_local.py).
Uses a tiny synthetic dataset to keep tests fast.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from train_local import train, EarlyStopping, safe_metrics, seed_everything


def _write_synthetic_index(folder: Path, n: int = 30, seq_len: int = 48, n_features: int = 40) -> str:
    """Write n synthetic patient .pt files and an index, balanced labels."""
    folder.mkdir(parents=True, exist_ok=True)
    paths, labels = [], []
    for i in range(n):
        X = torch.randn(seq_len, n_features)
        y = i % 2
        p = folder / f"p{i:03d}.pt"
        torch.save({"X": X, "y": y, "meta": {}}, p)
        paths.append(str(p))
        labels.append(y)
    idx_path = folder / "index.pt"
    torch.save({"x_paths": paths, "y": labels}, idx_path)
    return str(idx_path)


class TestEarlyStopping:
    def test_no_stop_improving(self):
        es = EarlyStopping(patience=3)
        for v in [0.5, 0.6, 0.7, 0.8]:
            assert es.step(v, epoch=1) is False

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3)
        es.step(0.8, epoch=1)   # sets best
        assert es.step(0.7, epoch=2) is False
        assert es.step(0.7, epoch=3) is False
        assert es.step(0.7, epoch=4) is True  # triggered

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=2)
        es.step(0.8, epoch=1)
        es.step(0.7, epoch=2)  # wait=1
        es.step(0.9, epoch=3)  # improvement -> wait resets
        assert es.wait == 0
        assert es.step(0.8, epoch=4) is False  # wait=1, not triggered yet

    def test_patience_one(self):
        es = EarlyStopping(patience=1)
        es.step(0.5, epoch=1)
        assert es.step(0.4, epoch=2) is True


class TestSafeMetrics:
    def test_normal_case(self):
        y = [0, 1, 0, 1, 0, 1]
        logits = [-2.0, 2.0, -1.0, 1.5, -0.5, 0.5]
        auc, ap = safe_metrics(y, logits)
        assert 0.5 <= auc <= 1.0
        assert 0.0 <= ap <= 1.0

    def test_single_class(self):
        y = [0, 0, 0]
        logits = [-1.0, 0.0, 1.0]
        auc, ap = safe_metrics(y, logits)
        assert auc == 0.0
        assert ap == 0.0

    def test_empty(self):
        auc, ap = safe_metrics([], [])
        assert auc == 0.0
        assert ap == 0.0


class TestTrainFunction:
    def test_creates_checkpoint(self, tmp_path):
        idx = _write_synthetic_index(tmp_path / "data", n=20)
        train(
            index_path=idx,
            model_name="transformer",
            epochs=2,
            batch_size=8,
            lr=1e-3,
            seed=42,
            run_name="test_run",
            checkpoint_root=str(tmp_path / "runs"),
            patience=10,
        )
        checkpoints = list((tmp_path / "runs").glob("*/checkpoints/model_best.pt"))
        assert len(checkpoints) == 1, "Expected one model_best.pt checkpoint"

    def test_creates_metrics_csv(self, tmp_path):
        idx = _write_synthetic_index(tmp_path / "data", n=20)
        train(
            index_path=idx,
            model_name="transformer",
            epochs=2,
            batch_size=8,
            lr=1e-3,
            seed=42,
            run_name="test_metrics",
            checkpoint_root=str(tmp_path / "runs"),
            patience=10,
        )
        csvs = list((tmp_path / "runs").glob("*/metrics.csv"))
        assert len(csvs) == 1
        import csv
        with open(csvs[0]) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2  # 2 epochs
        assert "val_auc" in rows[0]

    def test_early_stopping_fires(self, tmp_path):
        """Training with patience=1 on a static loss should stop early."""
        idx = _write_synthetic_index(tmp_path / "data", n=20)
        train(
            index_path=idx,
            model_name="transformer",
            epochs=10,       # many epochs planned
            batch_size=8,
            lr=0.0,          # zero LR → model never improves → early stop fires
            seed=42,
            run_name="test_early",
            checkpoint_root=str(tmp_path / "runs"),
            patience=1,
        )
        csvs = list((tmp_path / "runs").glob("*/metrics.csv"))
        import csv
        with open(csvs[0]) as f:
            rows = list(csv.DictReader(f))
        # Should have stopped well before 10 epochs (at most 2: epoch 1 sets best, epoch 2 triggers)
        assert len(rows) < 10, f"Expected early stopping before 10 epochs, ran {len(rows)}"

    def test_grud_model(self, tmp_path):
        idx = _write_synthetic_index(tmp_path / "data", n=20)
        train(
            index_path=idx,
            model_name="grud",
            epochs=2,
            batch_size=8,
            lr=1e-3,
            seed=42,
            run_name="test_grud",
            checkpoint_root=str(tmp_path / "runs"),
            patience=10,
        )
        checkpoints = list((tmp_path / "runs").glob("*/checkpoints/model_best.pt"))
        assert len(checkpoints) == 1

    def test_seed_reproducibility(self, tmp_path):
        seed_everything(7)
        x1 = torch.randn(5)
        seed_everything(7)
        x2 = torch.randn(5)
        assert torch.allclose(x1, x2)
