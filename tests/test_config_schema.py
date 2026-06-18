"""Tests for src/config_schema.py (Pydantic config models)."""
import os
import sys
import tempfile

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config_schema import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    FederatedConfig,
    EvaluationConfig,
    MLflowConfig,
    OptunaConfig,
    ProjectConfig,
    load_config,
)


class TestDataConfig:
    def test_defaults(self):
        dc = DataConfig()
        assert dc.seq_len == 48
        assert dc.n_features == 40
        assert dc.train_ratio == pytest.approx(0.70)

    def test_ratio_validation_passes(self):
        dc = DataConfig(train_ratio=0.70, val_ratio=0.15)
        assert dc.train_ratio + dc.val_ratio <= 1.0

    def test_ratio_validation_fails(self):
        with pytest.raises(Exception):
            DataConfig(train_ratio=0.80, val_ratio=0.30)

    def test_custom_values(self):
        dc = DataConfig(seq_len=24, n_features=20)
        assert dc.seq_len == 24
        assert dc.n_features == 20


class TestModelConfig:
    def test_defaults(self):
        mc = ModelConfig()
        assert mc.d_model == 128
        assert mc.n_heads == 4
        assert mc.d_model % mc.n_heads == 0

    def test_head_divisibility_fails(self):
        with pytest.raises(Exception):
            ModelConfig(d_model=128, n_heads=3)

    def test_head_divisibility_passes(self):
        mc = ModelConfig(d_model=64, n_heads=8)
        assert mc.d_model % mc.n_heads == 0

    def test_custom_model_name(self):
        mc = ModelConfig(name="grud")
        assert mc.name == "grud"


class TestTrainingConfig:
    def test_defaults(self):
        tc = TrainingConfig()
        assert tc.epochs == 50
        assert tc.seed == 42
        assert tc.use_focal is False

    def test_custom_values(self):
        tc = TrainingConfig(epochs=10, lr=1e-3, patience=5)
        assert tc.epochs == 10
        assert tc.lr == pytest.approx(1e-3)


class TestFederatedConfig:
    def test_defaults(self):
        fc = FederatedConfig()
        assert fc.rounds == 20
        assert fc.n_clients == 5
        assert fc.strategy == "fedavg"


class TestProjectConfig:
    def test_defaults(self):
        pc = ProjectConfig()
        assert isinstance(pc.data, DataConfig)
        assert isinstance(pc.model, ModelConfig)
        assert isinstance(pc.training, TrainingConfig)
        assert isinstance(pc.federated, FederatedConfig)
        assert isinstance(pc.evaluation, EvaluationConfig)
        assert isinstance(pc.mlflow, MLflowConfig)
        assert isinstance(pc.optuna, OptunaConfig)

    def test_extra_keys_forbidden(self):
        with pytest.raises(Exception):
            ProjectConfig(**{"unknown_key": "bad_value"})

    def test_nested_override(self):
        pc = ProjectConfig(training={"epochs": 5, "seed": 123})
        assert pc.training.epochs == 5
        assert pc.training.seed == 123


class TestLoadConfig:
    def test_no_path_returns_defaults(self):
        cfg = load_config(path=None)
        assert isinstance(cfg, ProjectConfig)
        assert cfg.data.seq_len == 48

    def test_load_from_yaml(self):
        config_dict = {
            "training": {"epochs": 7, "seed": 99},
            "data": {"seq_len": 24},
        }
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w"
        ) as f:
            yaml.dump(config_dict, f)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg.training.epochs == 7
            assert cfg.training.seed == 99
            assert cfg.data.seq_len == 24
        finally:
            os.remove(path)

    def test_empty_yaml_returns_defaults(self):
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w"
        ) as f:
            f.write("")
            path = f.name
        try:
            cfg = load_config(path)
            assert isinstance(cfg, ProjectConfig)
        finally:
            os.remove(path)
