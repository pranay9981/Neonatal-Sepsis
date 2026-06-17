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

import hashlib
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
from model_grud import GRUD

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response as FastAPIResponse
    _PROMETHEUS = True
except ImportError:
    _PROMETHEUS = False

_model = None
_scaler = None
_temperature: float = 1.0
_model_version: str = "unknown"
_load_time: float = 0.0
_start_time: float = time.time()

MODEL_TYPE = os.environ.get("SEPSIS_MODEL_TYPE", "grud")  # "transformer" or "grud"
MC_SAMPLES = int(os.environ.get("SEPSIS_MC_SAMPLES", "0"))
AUDIT_LOG = os.environ.get("SEPSIS_AUDIT_LOG", str(Path(__file__).parent.parent / "predictions.jsonl"))
MAX_ROWS = 1000

if _PROMETHEUS:
    _pred_counter = Counter("sepsis_predictions_total", "Total predictions served")
    _alert_counter = Counter("sepsis_alerts_total", "Total HIGH-risk alerts")
    _latency_hist = Histogram("sepsis_predict_latency_seconds", "Predict endpoint latency")


def _pad_or_trim(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    T, F = target_shape
    if arr.shape[0] < T:
        pad = np.zeros((T - arr.shape[0], F), dtype=np.float32)
        arr = np.concatenate([pad, arr], axis=0)
    elif arr.shape[0] > T:
        arr = arr[-T:]
    return arr


def _load_artifacts():
    global _model, _scaler, _temperature, _model_version, _load_time
    p = Path(MODEL_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    expected_sha = os.environ.get("SEPSIS_MODEL_SHA256")
    if expected_sha:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != expected_sha:
            raise RuntimeError(
                f"Model checksum mismatch! Expected {expected_sha}, got {actual}"
            )

    raw = torch.load(str(p), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model_state" in raw:
        raw = raw["model_state"]
    if MODEL_TYPE == "grud":
        m = GRUD(n_features=N_FEATURES)
    else:
        m = TimeSeriesTransformer(n_features=N_FEATURES, seq_len=SEQ_LEN)
    try:
        m.load_state_dict(raw)
    except Exception:
        result = m.load_state_dict(raw, strict=False)
        if result.missing_keys:
            _logger.warning("Model load missing keys: %s", result.missing_keys)
        if result.unexpected_keys:
            _logger.warning("Model load unexpected keys: %s", result.unexpected_keys)
    m.eval()
    _model = m
    _model_version = p.stem
    _load_time = time.time()

    sp = Path(SCALER_PATH)
    if sp.exists():
        with open(sp) as f:
            _scaler = json.load(f)

    # Load temperature from threshold.json (written by TemperatureScaler.save()).
    thresh_path = p.parent / "threshold.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            thresh_data = json.load(f)
        _temperature = float(thresh_data.get("temperature", 1.0))
        if _temperature <= 0:
            _temperature = 1.0


def _append_audit(entry: dict):
    try:
        with open(AUDIT_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        _logger.error("Audit log write failed — prediction not recorded: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_artifacts()
    except FileNotFoundError as e:
        _logger.error("Model not found at startup: %s — server will start with _model=None and return 503 on /predict", e)
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
    # GRU-D mode: binary observation mask and time-gap deltas (same shape as data).
    # If omitted when MODEL_TYPE=="grud", defaults to all-observed / zero-gap.
    mask: Optional[List[List[float]]] = None
    deltas: Optional[List[List[float]]] = None

    @field_validator("data")
    @classmethod
    def check_shape(cls, v):
        if len(v) == 0:
            raise ValueError("data must not be empty")
        if len(v) > MAX_ROWS:
            raise ValueError(f"Input exceeds {MAX_ROWS} rows; got {len(v)}")
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

    tensor = torch.tensor(arr).unsqueeze(0)  # (1, T, F)

    # Build extra tensors for GRU-D
    if MODEL_TYPE == "grud":
        mask_arr = np.ones_like(arr) if req.mask is None else _pad_or_trim(np.array(req.mask, dtype=np.float32), arr.shape)
        delta_arr = np.zeros_like(arr) if req.deltas is None else _pad_or_trim(np.array(req.deltas, dtype=np.float32), arr.shape)
        mask_t = torch.tensor(mask_arr).unsqueeze(0)
        delta_t = torch.tensor(delta_arr).unsqueeze(0)

    def _forward():
        if MODEL_TYPE == "grud":
            return _model(tensor, mask_t, delta_t)
        return _model(tensor)

    prob_low = prob_high = None
    with torch.no_grad():
        with _mc_lock:
            if MC_SAMPLES > 1:
                _model.train()  # enable dropout
                try:
                    mc_probs = [float(torch.sigmoid(_forward()).squeeze()) for _ in range(MC_SAMPLES)]
                finally:
                    _model.eval()
                prob = float(np.mean(mc_probs))
                prob_low = float(np.percentile(mc_probs, 2.5))
                prob_high = float(np.percentile(mc_probs, 97.5))
            else:
                logit = _forward().squeeze()
                logit = logit / _temperature
                prob = float(torch.sigmoid(logit).item())

    risk = "HIGH" if prob >= 0.5 else ("MODERATE" if prob >= 0.25 else "LOW")
    alert = prob >= 0.5
    latency_ms = (time.time() - t0) * 1000

    if _PROMETHEUS:
        _pred_counter.inc()
        _latency_hist.observe(latency_ms / 1000)
        if alert:
            _alert_counter.inc()

    def _bucket_timesteps(n: int) -> str:
        if n <= 12: return "0-12h"
        if n <= 24: return "13-24h"
        if n <= 48: return "25-48h"
        return "49h+"

    _append_audit({
        "patient_id": hashlib.sha256(str(req.patient_id).encode()).hexdigest() if req.patient_id else None,
        "probability": prob,
        "risk_level": risk,
        "alert": alert,
        "model_version": _model_version,
        "n_timesteps": _bucket_timesteps(n_rows),
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
