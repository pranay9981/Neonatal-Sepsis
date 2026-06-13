"""
FastAPI inference endpoint for the Neonatal Sepsis model.

Usage:
  uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

POST /predict
  Body: { "data": [[f1, f2, ..., f40], ...] }   # SEQ_LEN x N_FEATURES matrix
  Returns: { "probability": 0.73, "risk_level": "HIGH", "model": "global_best" }
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MODEL_PATH, SCALER_PATH, N_FEATURES, SEQ_LEN
from model import TimeSeriesTransformer

_model = None
_scaler = None


def _load_artifacts():
    global _model, _scaler
    p = Path(MODEL_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    raw = torch.load(str(p), map_location="cpu")
    if isinstance(raw, dict) and "model_state" in raw:
        raw = raw["model_state"]
    m = TimeSeriesTransformer(n_features=N_FEATURES, seq_len=SEQ_LEN)
    try:
        m.load_state_dict(raw)
    except Exception:
        m.load_state_dict(raw, strict=False)
    m.eval()
    _model = m

    sp = Path(SCALER_PATH)
    if sp.exists():
        with open(sp) as f:
            _scaler = json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(
    title="Neonatal Sepsis Detection API",
    description="REST endpoint for the federated sepsis detection model.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    data: List[List[float]]

    @field_validator("data")
    @classmethod
    def check_shape(cls, v):
        if len(v) == 0:
            raise ValueError("data must not be empty")
        n_cols = len(v[0])
        if n_cols != N_FEATURES:
            raise ValueError(f"Each row must have {N_FEATURES} features, got {n_cols}")
        for row in v:
            if len(row) != N_FEATURES:
                raise ValueError(f"Inconsistent row length: expected {N_FEATURES}")
        return v


class PredictResponse(BaseModel):
    probability: float
    risk_level: str
    model: str
    n_timesteps: int


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    arr = np.array(req.data, dtype=np.float32)
    n_rows = arr.shape[0]

    # Pad or truncate to SEQ_LEN
    if n_rows < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - n_rows, N_FEATURES), dtype=np.float32)
        arr = np.concatenate([pad, arr], axis=0)
    elif n_rows > SEQ_LEN:
        arr = arr[-SEQ_LEN:]

    # Apply scaler if available
    if _scaler is not None:
        mean = np.array(_scaler["mean"], dtype=np.float32)
        std = np.array(_scaler["std"], dtype=np.float32)
        arr = (arr - mean) / (std + 1e-8)

    tensor = torch.tensor(arr).unsqueeze(0)  # (1, SEQ_LEN, N_FEATURES)

    with torch.no_grad():
        logit = _model(tensor).squeeze()
        prob = float(torch.sigmoid(logit).item())

    if prob >= 0.5:
        risk = "HIGH"
    elif prob >= 0.25:
        risk = "MODERATE"
    else:
        risk = "LOW"

    return PredictResponse(
        probability=prob,
        risk_level=risk,
        model=Path(MODEL_PATH).stem,
        n_timesteps=len(req.data),
    )
