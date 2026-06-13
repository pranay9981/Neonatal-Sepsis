# src/config_schema.py
"""Pydantic config schema. Load from configs/base.yaml via load_config()."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
try:
    from pydantic import BaseModel, field_validator
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseModel, validator as field_validator  # type: ignore

import yaml


class DataConfig(BaseModel):
    raw_folder: str = "data/raw"
    processed_folder: str = "data/processed/patients"
    splits_dir: str = "data/splits"
    seq_len: int = 48
    n_features: int = 40
    train_ratio: float = 0.70
    val_ratio: float = 0.15


class ModelConfig(BaseModel):
    name: str = "transformer"
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    hidden_size: int = 128


class TrainingConfig(BaseModel):
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-4
    patience: int = 10
    warmup_epochs: int = 3
    clip_grad: float = 1.0
    use_focal: bool = False
    focal_gamma: float = 2.0
    augment: bool = True
    use_temperature_scaling: bool = True
    seed: int = 42


class FederatedConfig(BaseModel):
    rounds: int = 20
    n_clients: int = 5
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 1e-3
    mu: float = 0.01
    strategy: str = "fedavg"
    heterogeneous: bool = False


class EvaluationConfig(BaseModel):
    bootstrap_n: int = 1000
    bootstrap_ci: float = 0.95
    horizon: int = 6


class MLflowConfig(BaseModel):
    enabled: bool = False
    experiment: str = "neonatal_sepsis"
    tracking_uri: str = "mlruns"


class OptunaConfig(BaseModel):
    n_trials: int = 50
    study_name: str = "sepsis_hpo"


class ProjectConfig(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    federated: FederatedConfig = FederatedConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    mlflow: MLflowConfig = MLflowConfig()
    optuna: OptunaConfig = OptunaConfig()


def load_config(path: str | None = None) -> ProjectConfig:
    """Load config from YAML; missing keys fall back to Pydantic defaults."""
    if path is None:
        default = Path(__file__).parent.parent / "configs" / "base.yaml"
        path = str(default) if default.exists() else None
    if path is None:
        return ProjectConfig()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return ProjectConfig(**{k: v for k, v in raw.items() if v is not None})
