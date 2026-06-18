"""
Unit tests for the preprocessing pipeline.
Run from project root: python -m pytest tests/
"""
import os
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parallel_preprocess import (
    detect_sep_from_file,
    safe_read,
    make_datetime_index,
    process_file,
)

FEATURE_COLS = [
    "HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2",
    "BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN",
    "Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct",
    "Glucose","Lactate","Magnesium","Phosphate","Potassium","Bilirubin_total",
    "TroponinI","Hct","Hgb","PTT","WBC","Fibrinogen","Platelets","Age","Gender","Unit1","Unit2",
    "HospAdmTime","ICULOS",
]


def _make_psv(n_rows: int = 10, sep: str = "|", sepsis_label: int = 0) -> str:
    rng = np.random.default_rng(42)
    cols = FEATURE_COLS + ["SepsisLabel"]
    df = pd.DataFrame(rng.random((n_rows, len(cols))), columns=cols)
    df["ICULOS"] = range(1, n_rows + 1)
    df["SepsisLabel"] = sepsis_label
    with tempfile.NamedTemporaryFile(mode="w", suffix=".psv", delete=False) as f:
        df.to_csv(f, sep=sep, index=False)
        return f.name


class TestDetectSep:
    def test_pipe(self):
        path = _make_psv(sep="|")
        assert detect_sep_from_file(path) == "|"
        os.unlink(path)

    def test_comma(self):
        path = _make_psv(sep=",")
        assert detect_sep_from_file(path) == ","
        os.unlink(path)


class TestSafeRead:
    def test_reads_all_columns(self):
        path = _make_psv()
        df = safe_read(path)
        assert df.shape[1] >= 40
        os.unlink(path)

    def test_strips_whitespace_from_column_names(self):
        path = _make_psv()
        df = safe_read(path)
        assert all(c == c.strip() for c in df.columns)
        os.unlink(path)


class TestProcessFile:
    def test_success_full_length(self):
        path = _make_psv(n_rows=60)
        with tempfile.TemporaryDirectory() as out_dir:
            result = process_file(path, out_dir, seq_len=48)
            fp, ok = result[0], result[1]
            info = result[2]
            assert ok, f"Expected success, got: {info}"
            d = torch.load(info, weights_only=True)
            assert d["X"].shape == (48, 40)
        os.unlink(path)

    def test_success_short_sequence_padded(self):
        path = _make_psv(n_rows=10)
        with tempfile.TemporaryDirectory() as out_dir:
            result = process_file(path, out_dir, seq_len=48)
            fp, ok = result[0], result[1]
            info = result[2]
            assert ok, f"Expected success with padding, got: {info}"
            d = torch.load(info, weights_only=True)
            assert d["X"].shape == (48, 40)
        os.unlink(path)

    def test_label_extracted_positive(self):
        path = _make_psv(n_rows=20, sepsis_label=1)
        with tempfile.TemporaryDirectory() as out_dir:
            result = process_file(path, out_dir, seq_len=48)
            fp, ok = result[0], result[1]
            info = result[2]
            assert ok
            d = torch.load(info, weights_only=True)
            assert d["y"] == 1
        os.unlink(path)

    def test_label_extracted_negative(self):
        path = _make_psv(n_rows=20, sepsis_label=0)
        with tempfile.TemporaryDirectory() as out_dir:
            result = process_file(path, out_dir, seq_len=48)
            fp, ok = result[0], result[1]
            info = result[2]
            assert ok
            d = torch.load(info, weights_only=True)
            assert d["y"] == 0
        os.unlink(path)

    def test_output_tensor_is_finite(self):
        path = _make_psv(n_rows=50)
        with tempfile.TemporaryDirectory() as out_dir:
            result = process_file(path, out_dir, seq_len=48)
            fp, ok = result[0], result[1]
            info = result[2]
            assert ok
            d = torch.load(info, weights_only=True)
            assert torch.isfinite(d["X"]).all()
        os.unlink(path)

    def test_zero_row_file_returns_failure(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".psv", delete=False) as f:
            f.write("HR|O2Sat\n")  # header only, no data rows
            path = f.name
        with tempfile.TemporaryDirectory() as out_dir:
            fp, ok, info = process_file(path, out_dir, seq_len=48)
            assert not ok
        os.unlink(path)

    def test_missing_columns_returns_failure(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".psv", delete=False) as f:
            # File has no SepsisLabel and no ICULOS — process_file cannot form a valid sequence
            f.write("NotAFeature|AnotherFake\n1.0|2.0\n")
            path = f.name
        with tempfile.TemporaryDirectory() as out_dir:
            fp, ok, info = process_file(path, out_dir, seq_len=48)
            assert not ok
        os.unlink(path)
