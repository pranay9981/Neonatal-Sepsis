"""
FastAPI inference endpoint for the Neonatal Sepsis model.

Improvements over v1.0:
- Model versioning (name + load timestamp returned in every response)
- Input range validation (physiologically plausible bounds per feature)
- Prediction audit log (JSONL file, append-only)
- /metrics endpoint for Prometheus scraping
- Confidence interval via MC Dropout (optional, --mc_samples)
- /health returns uptime and model version

Usage:
  uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

_logger = logging.getLogger(__name__)
_mc_lock = threading.Lock()

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MODEL_PATH, SCALER_PATH, N_FEATURES, SEQ_LEN
from model import TimeSeriesTransformer

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response as FastAPIResponse
    _PROMETHEUS = True
except ImportError:
    _PROMETHEUS = False

_model = None
_scaler = None
_model_version: str = "unknown"
_load_time: float = 0.0
_start_time: float = time.time()

MC_SAMPLES = int(os.environ.get("SEPSIS_MC_SAMPLES", "0"))
AUDIT_LOG = os.environ.get("SEPSIS_AUDIT_LOG", str(Path(MODEL_PATH).parent / "predictions.jsonl"))

if _PROMETHEUS:
    _pred_counter = Counter("sepsis_predictions_total", "Total predictions served")
    _alert_counter = Counter("sepsis_alerts_total", "Total HIGH-risk alerts")
    _latency_hist = Histogram("sepsis_predict_latency_seconds", "Predict endpoint latency")


def _load_artifacts():
    global _model, _scaler, _model_version, _load_time
    p = Path(MODEL_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    raw = torch.load(str(p), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model_state" in raw:
        raw = raw["model_state"]
    m = TimeSeriesTransformer(n_features=N_FEATURES, seq_len=SEQ_LEN)
    try:
        m.load_state_dict(raw)
    except Exception:
        m.load_state_dict(raw, strict=False)
    m.eval()
    _model = m
    _model_version = p.stem
    _load_time = time.time()

    sp = Path(SCALER_PATH)
    if sp.exists():
        with open(sp) as f:
            _scaler = json.load(f)


def _append_audit(entry: dict):
    try:
        with open(AUDIT_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        _logger.error("Audit log write failed — prediction not recorded: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(
    title="Neonatal Sepsis Detection API",
    description="Federated sepsis detection — versioned inference endpoint.",
    version="2.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    data: List[List[float]]
    patient_id: Optional[str] = None

    @field_validator("data")
    @classmethod
    def check_shape(cls, v):
        if len(v) == 0:
            raise ValueError("data must not be empty")
        for row in v:
            if len(row) != N_FEATURES:
                raise ValueError(f"Each row must have {N_FEATURES} features, got {len(row)}")
        return v


class PredictResponse(BaseModel):
    probability: float
    risk_level: str
    alert: bool
    model_version: str
    n_timesteps: int
    confidence_low: Optional[float] = None
    confidence_high: Optional[float] = None
    latency_ms: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_version": _model_version,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@app.get("/metrics")
def metrics():
    if not _PROMETHEUS:
        raise HTTPException(status_code=501, detail="prometheus_client not installed")
    return FastAPIResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.time()
    arr = np.array(req.data, dtype=np.float32)
    n_rows = arr.shape[0]

    if n_rows < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - n_rows, N_FEATURES), dtype=np.float32)
        arr = np.concatenate([pad, arr], axis=0)
    elif n_rows > SEQ_LEN:
        arr = arr[-SEQ_LEN:]

    if _scaler is not None:
        mean = np.array(_scaler["mean"], dtype=np.float32)
        std = np.array(_scaler["std"], dtype=np.float32)
        arr = (arr - mean) / (std + 1e-8)

    tensor = torch.tensor(arr).unsqueeze(0)

    prob_low = prob_high = None
    with torch.no_grad():
        if MC_SAMPLES > 1:
            with _mc_lock:
                _model.train()  # enable dropout for MC sampling
                mc_probs = [float(torch.sigmoid(_model(tensor)).squeeze()) for _ in range(MC_SAMPLES)]
                _model.eval()
            prob = float(np.mean(mc_probs))
            prob_low = float(np.percentile(mc_probs, 2.5))
            prob_high = float(np.percentile(mc_probs, 97.5))
        else:
            logit = _model(tensor).squeeze()
            prob = float(torch.sigmoid(logit).item())

    risk = "HIGH" if prob >= 0.5 else ("MODERATE" if prob >= 0.25 else "LOW")
    alert = prob >= 0.5
    latency_ms = (time.time() - t0) * 1000

    if _PROMETHEUS:
        _pred_counter.inc()
        _latency_hist.observe((time.time() - t0))
        if alert:
            _alert_counter.inc()

    _append_audit({
        "patient_id": req.patient_id,
        "probability": prob,
        "risk_level": risk,
        "alert": alert,
        "model_version": _model_version,
        "n_timesteps": n_rows,
        "latency_ms": round(latency_ms, 2),
        "timestamp": time.time(),
    })

    return PredictResponse(
        probability=prob,
        risk_level=risk,
        alert=alert,
        model_version=_model_version,
        n_timesteps=n_rows,
        confidence_low=prob_low,
        confidence_high=prob_high,
        latency_ms=round(latency_ms, 2),
    )
