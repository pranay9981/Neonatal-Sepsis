"""
Central configuration for the Neonatal Sepsis project.
All paths and key constants are resolved here and can be overridden
via environment variables without touching any source file.

Usage:
  export SEPSIS_MODEL_PATH=/path/to/my_model.pt
  streamlit run app.py
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# --- Model & artifact paths ---
MODEL_PATH = os.environ.get(
    "SEPSIS_MODEL_PATH",
    str(PROJECT_ROOT / "server_out" / "global_best.pt"),
)
SCALER_PATH = os.environ.get(
    "SEPSIS_SCALER_PATH",
    str(PROJECT_ROOT / "data" / "processed" / "patients" / "scaler.json"),
)
EVAL_FEDERATED_JSON = os.environ.get(
    "SEPSIS_EVAL_FED",
    str(PROJECT_ROOT / "eval_results_federated.json"),
)
EVAL_LOCAL_JSON = os.environ.get(
    "SEPSIS_EVAL_LOCAL",
    str(PROJECT_ROOT / "eval_results_local.json"),
)
PLOT_ROC_PATH = os.environ.get(
    "SEPSIS_PLOT_ROC",
    str(PROJECT_ROOT / "model_comparison_plot.png"),
)
PLOT_PRC_PATH = os.environ.get(
    "SEPSIS_PLOT_PRC",
    str(PROJECT_ROOT / "model_comparison_plot_prc.png"),
)

# --- Model architecture constants ---
N_FEATURES: int = int(os.environ.get("SEPSIS_N_FEATURES", "40"))
SEQ_LEN: int = int(os.environ.get("SEPSIS_SEQ_LEN", "48"))
