"""
Tests for the evaluation pipeline (src/evaluate.py).
Verifies metric computation, checkpoint loading, and JSON output.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from model import TimeSeriesTransformer
from model_grud import GRUD
from evaluate import evaluate_single_ckpt, build_model_for_eval


def _write_synthetic_data(folder: Path, n: int = 40, seq_len: int = 48, n_features: int = 40) -> str:
    """Write n synthetic patient tensors and an index file."""
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


def _save_model_ckpt(folder: Path, name: str = "model.pt", n_features: int = 40, seq_len: int = 48) -> str:
    """Save a fresh (untrained) Transformer checkpoint."""
    model = TimeSeriesTransformer(n_features=n_features, seq_len=seq_len)
    ckpt_path = folder / name
    torch.save(model.state_dict(), ckpt_path)
    return str(ckpt_path)


class TestBuildModelForEval:
    def test_transformer(self):
        model = build_model_for_eval("transformer", n_features=40, seq_len=48)
        assert isinstance(model, TimeSeriesTransformer)

    def test_grud(self):
        model = build_model_for_eval("grud", n_features=40, seq_len=48)
        assert isinstance(model, GRUD)

    def test_unknown_model(self):
        with pytest.raises((ValueError, AssertionError)):
            build_model_for_eval("unknown_model", n_features=40, seq_len=48)


class TestEvaluateSingleCkpt:
    def test_returns_dict_with_keys(self, tmp_path):
        idx = _write_synthetic_data(tmp_path / "data")
        ckpt = _save_model_ckpt(tmp_path)
        result = evaluate_single_ckpt(
            index_path=idx,
            ckpt_path=ckpt,
            model_name="transformer",
            n_features=40,
            seq_len=48,
        )
        for key in ("auroc", "auprc", "n", "y_true", "y_prob"):
            assert key in result, f"Missing key: {key}"

    def test_auroc_in_range(self, tmp_path):
        idx = _write_synthetic_data(tmp_path / "data")
        ckpt = _save_model_ckpt(tmp_path)
        result = evaluate_single_ckpt(
            index_path=idx,
            ckpt_path=ckpt,
            model_name="transformer",
            n_features=40,
            seq_len=48,
        )
        assert isinstance(result["auroc"], float), f"Expected float AUROC, got {type(result['auroc'])}"
        assert 0.0 <= result["auroc"] <= 1.0, f"AUROC out of range: {result['auroc']}"
        assert isinstance(result["auprc"], float), f"Expected float AUPRC, got {type(result['auprc'])}"
        assert 0.0 <= result["auprc"] <= 1.0, f"AUPRC out of range: {result['auprc']}"

    def test_sample_count(self, tmp_path):
        n = 30
        idx = _write_synthetic_data(tmp_path / "data", n=n)
        ckpt = _save_model_ckpt(tmp_path)
        result = evaluate_single_ckpt(
            index_path=idx,
            ckpt_path=ckpt,
            model_name="transformer",
            n_features=40,
            seq_len=48,
        )
        assert result["n"] == n
        assert len(result["y_true"]) == n
        assert len(result["y_prob"]) == n

    def test_probs_in_range(self, tmp_path):
        idx = _write_synthetic_data(tmp_path / "data")
        ckpt = _save_model_ckpt(tmp_path)
        result = evaluate_single_ckpt(
            index_path=idx,
            ckpt_path=ckpt,
            model_name="transformer",
            n_features=40,
            seq_len=48,
        )
        probs = np.array(result["y_prob"])
        assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities out of [0, 1] range"

    def test_saves_json(self, tmp_path):
        idx = _write_synthetic_data(tmp_path / "data")
        ckpt = _save_model_ckpt(tmp_path)
        out_json = tmp_path / "results.json"
        evaluate_single_ckpt(
            index_path=idx,
            ckpt_path=ckpt,
            model_name="transformer",
            n_features=40,
            seq_len=48,
            out_file=str(out_json),
        )
        assert out_json.exists(), "out_file was not created"
        with open(out_json) as f:
            data = json.load(f)
        assert "auroc" in data
        assert "y_true" in data

    def test_train_local_ckpt_format(self, tmp_path):
        """evaluate.py must handle the {'model_state': ...} format from train_local.py."""
        model = TimeSeriesTransformer(n_features=40, seq_len=48)
        ckpt_path = tmp_path / "model_best.pt"
        torch.save({"model_state": model.state_dict(), "epoch": 5, "auc": 0.72}, ckpt_path)
        idx = _write_synthetic_data(tmp_path / "data")
        result = evaluate_single_ckpt(
            index_path=idx,
            ckpt_path=str(ckpt_path),
            model_name="transformer",
            n_features=40,
            seq_len=48,
        )
        assert result["n"] > 0

    def test_missing_checkpoint(self, tmp_path):
        idx = _write_synthetic_data(tmp_path / "data")
        with pytest.raises(FileNotFoundError):
            evaluate_single_ckpt(
                index_path=idx,
                ckpt_path=str(tmp_path / "nonexistent.pt"),
                model_name="transformer",
                n_features=40,
                seq_len=48,
            )
