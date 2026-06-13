# app.py
"""
Launcher for the multi-page Streamlit dashboard — Neonatal Sepsis Detection.
Dynamically loads page modules from app_pages/ and calls their render() method.
"""

import streamlit as st
from pathlib import Path
import importlib.util
import sys
import json

ROOT = Path(__file__).parent
PAGES_DIR = ROOT / "app_pages"

MODEL_PATH = ROOT / "server_out" / "global_best.pt"
EVAL_FED_PATH = ROOT / "eval_results_federated.json"
EVAL_LOC_PATH = ROOT / "eval_results_local.json"

# Map: (label, filename, class_name)
PAGES = [
    ("📘 Overview",         "1_00_📘_Project_Summary.py",  "ProjectSummaryPage"),
    ("📈 Predict",          "1_03_📈_Predict.py",          "PredictPage"),
    ("🧪 Model Metrics",    "1_04_🧪_Model_Metrics.py",    "MetricsPage"),
    ("📂 Training Runs",    "1_05_📂_Training_Runs.py",    "TrainingRunsPage"),
    ("🏥 Clinical Metrics", "1_06_🏥_Clinical_Metrics.py", "ClinicalMetricsPage"),
]

st.set_page_config(
    page_title="Neonatal Sepsis Dashboard",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base & fonts ─────────────────────────────────────────────── */
  html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
  }
  /* ── Sidebar ──────────────────────────────────────────────────── */
  section[data-testid="stSidebar"] {
    background: #1E3A5F;
    color: #E8F0FE;
  }
  section[data-testid="stSidebar"] .stRadio label {
    color: #C8D8F0 !important;
    font-size: 0.95rem;
    padding: 4px 0;
  }
  section[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div {
    color: #FFFFFF !important;
    font-weight: 600;
  }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] div {
    color: #E8F0FE;
  }
  /* ── Main area top bar ────────────────────────────────────────── */
  .main-header {
    background: linear-gradient(90deg, #1E3A5F 0%, #2563EB 100%);
    color: white;
    padding: 18px 28px 14px 28px;
    border-radius: 10px;
    margin-bottom: 24px;
  }
  .main-header h1 {
    margin: 0;
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: white !important;
  }
  .main-header p {
    margin: 4px 0 0 0;
    font-size: 0.9rem;
    opacity: 0.85;
    color: #dde8ff !important;
  }
  /* ── Status badge ─────────────────────────────────────────────── */
  .status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px 0;
  }
  .badge-ok   { background: #D4EDDA; color: #1E7E34; }
  .badge-miss { background: #FDECEA; color: #C62828; }
  /* ── Metric card ──────────────────────────────────────────────── */
  .metric-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .metric-card .metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #1E3A5F;
    line-height: 1;
  }
  .metric-card .metric-label {
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  /* ── Section card ─────────────────────────────────────────────── */
  .section-card {
    background: #F8FAFC !important;
    border-left: 4px solid #2196F3;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    color: #1E293B !important;
  }
  .section-card b, .section-card strong {
    color: #1E3A5F !important;
  }
  /* ── Callout box ──────────────────────────────────────────────── */
  .callout-success {
    background: #E8F5E9 !important;
    border-left: 4px solid #4CAF50;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    color: #1B5E20 !important;
    font-weight: 500;
    margin: 12px 0;
  }
  .callout-info {
    background: #E3F2FD !important;
    border-left: 4px solid #2196F3;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    color: #0D47A1 !important;
    margin: 12px 0;
  }
  .callout-info b, .callout-info strong,
  .callout-success b, .callout-success strong {
    color: inherit !important;
  }
  /* ── Feature category chips ───────────────────────────────────── */
  .chip {
    display: inline-block;
    background: #EFF6FF !important;
    color: #1D4ED8 !important;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
  }
  /* ── Step number circles ──────────────────────────────────────── */
  .step-num {
    background: #1E3A5F !important;
    color: white !important;
  }
  .step-desc {
    color: #475569 !important;
  }
  /* hide default streamlit title padding */
  .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 18px 0;">
      <div style="font-size:2.2rem;">👶</div>
      <div style="font-size:1rem; font-weight:700; color:#FFFFFF; letter-spacing:-0.2px;">
        Neonatal Sepsis
      </div>
      <div style="font-size:0.75rem; color:#9DB8D9; margin-top:2px;">
        FL Detection Dashboard
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    choice = st.radio("Navigation", [p[0] for p in PAGES], label_visibility="collapsed")
    st.markdown("---")

    # Status section
    model_ok = MODEL_PATH.exists()
    fed_ok   = EVAL_FED_PATH.exists()
    loc_ok   = EVAL_LOC_PATH.exists()

    st.markdown("<div style='font-size:0.78rem; color:#9DB8D9; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:6px;'>SYSTEM STATUS</div>", unsafe_allow_html=True)

    def _badge(label, ok):
        cls = "badge-ok" if ok else "badge-miss"
        icon = "✓" if ok else "✗"
        st.markdown(f"<span class='status-badge {cls}'>{icon} {label}</span>", unsafe_allow_html=True)

    _badge("Model loaded", model_ok)
    _badge("Fed eval results", fed_ok)
    _badge("Local eval results", loc_ok)

    st.markdown("<div style='margin-top:24px; font-size:0.7rem; color:#6B8CAE;'>v2 · improvements/v2 branch</div>", unsafe_allow_html=True)

# ── Resolve page ───────────────────────────────────────────────────────────────
selected_label, selected_file, selected_class = None, None, None
for label, fname, cls in PAGES:
    if label == choice:
        selected_label, selected_file, selected_class = label, fname, cls
        break

if selected_file is None:
    st.error("Selected page not found.")
    st.stop()

module_path = PAGES_DIR / selected_file
if not module_path.exists():
    st.error(f"Page file not found: {module_path}")
    st.stop()

spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
module = importlib.util.module_from_spec(spec)
sys.modules[module_path.stem] = module

try:
    spec.loader.exec_module(module)
except Exception as e:
    st.exception(e)
    st.stop()

# Call the page's render method
page_class = getattr(module, selected_class, None)
if page_class is not None and hasattr(page_class, "render"):
    try:
        page_class.render()
    except Exception as e:
        st.exception(e)
elif hasattr(module, "render"):
    try:
        module.render()
    except Exception as e:
        st.exception(e)
else:
    st.error(f"Page '{selected_label}' does not expose a render() method.")
