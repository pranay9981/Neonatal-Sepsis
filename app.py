# app.py
"""
Launcher for the multi-page Streamlit dashboard — Neonatal Sepsis Detection.
Dynamically loads page modules from app_pages/ and calls their render() method.
"""

import logging as _logging
import streamlit as st
from pathlib import Path
import importlib.util
import sys
import json

ROOT = Path(__file__).parent
PAGES_DIR = ROOT / "app_pages"

MODEL_PATH = ROOT / "server_out" / "global_best.pt"
EVAL_FED_PATH = ROOT / "eval_results_federated.json"
EVAL_LOC_PATHS = [ROOT / "eval_results_grud.json", ROOT / "eval_results_transformer.json"]
EVAL_WINDOWED_PATH = ROOT / "eval_results_windowed_grud.json"

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
  /* ── Main background ─────────────────────────────────────────── */
  .stApp {
    background: #0F1117;
  }
  /* ── Sidebar ──────────────────────────────────────────────────── */
  section[data-testid="stSidebar"] {
    background: #0D1321 !important;
    border-right: 1px solid #1E2A45;
  }
  section[data-testid="stSidebar"] > div {
    background: #0D1321 !important;
  }
  /* ── Sidebar nav inactive buttons ────────────────────────────── */
  section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    color: #64748B !important;
    text-align: left !important;
    padding: 9px 14px !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
    width: 100% !important;
    margin: 1px 0 !important;
    transition: background 0.15s, color 0.15s !important;
  }
  section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(59,130,246,0.08) !important;
    color: #94A3B8 !important;
    border: none !important;
  }
  section[data-testid="stSidebar"] .stButton > button:focus {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
  }
  /* ── Block container padding ──────────────────────────────────── */
  .block-container { padding-top: 1rem; }
  /* ── Dataframe / table ────────────────────────────────────────── */
  .stDataFrame { border: 1px solid #1E2A45 !important; }
  /* ── Metric widget ────────────────────────────────────────────── */
  [data-testid="stMetric"] {
    background: #141827;
    border: 1px solid #1E2A45;
    border-radius: 10px;
    padding: 14px 16px;
  }
  [data-testid="stMetricLabel"] { color: #94A3B8 !important; }
  [data-testid="stMetricValue"] { color: #F1F5F9 !important; }
  /* ── Expander ─────────────────────────────────────────────────── */
  details summary {
    color: #94A3B8 !important;
  }
  /* ── Tabs ─────────────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab"] {
    color: #64748B !important;
  }
  .stTabs [aria-selected="true"] {
    color: #60A5FA !important;
    border-bottom-color: #3B82F6 !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
if "selected_page" not in st.session_state:
    st.session_state.selected_page = PAGES[0][0]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo block
    st.markdown("""
    <div style="padding:12px 4px 20px 4px; border-bottom:1px solid #1E2A45; margin-bottom:12px;">
      <div style="display:flex; align-items:center; gap:10px;">
        <div style="background:rgba(59,130,246,0.15); border-radius:10px; padding:8px;
             font-size:1.4rem; line-height:1;">👶</div>
        <div>
          <div style="font-size:0.95rem; font-weight:700; color:#F1F5F9;">Neonatal Sepsis</div>
          <div style="font-size:0.72rem; color:#475569; margin-top:1px;">FL Detection · v2</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    for label, fname, cls in PAGES:
        is_active = st.session_state.selected_page == label
        if is_active:
            st.markdown(
                f'<div style="background:rgba(59,130,246,0.13); border:1px solid rgba(59,130,246,0.25);'
                f'border-radius:8px; padding:9px 14px; color:#60A5FA; font-size:0.9rem;'
                f'font-weight:600; margin:1px 0; pointer-events:none; box-sizing:border-box;">'
                f'{label}</div>',
                unsafe_allow_html=True,
            )
        else:
            if st.button(label, key=f"nav_{label}", use_container_width=True):
                st.session_state.selected_page = label
                st.rerun()

    # Status section
    model_ok    = MODEL_PATH.exists()
    fed_ok      = EVAL_FED_PATH.exists()
    loc_ok      = any(p.exists() for p in EVAL_LOC_PATHS)
    windowed_ok = EVAL_WINDOWED_PATH.exists()

    st.markdown("""
    <div style="margin-top:20px; padding-top:16px; border-top:1px solid #1E2A45;">
      <div style="font-size:0.68rem; color:#475569; text-transform:uppercase;
           letter-spacing:1px; margin-bottom:10px;">System Status</div>
    </div>
    """, unsafe_allow_html=True)

    def _status_dot(label, ok):
        dot_color = "#10B981" if ok else "#EF4444"
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:8px; margin:6px 0;">'
            f'<div style="width:7px; height:7px; border-radius:50%; background:{dot_color};'
            f' flex-shrink:0; box-shadow:0 0 6px {dot_color};"></div>'
            f'<span style="font-size:0.8rem; color:#64748B;">{label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _status_dot("Model checkpoint", model_ok)
    _status_dot("Federated eval JSON", fed_ok)
    _status_dot("Local eval JSON", loc_ok)
    _status_dot("Windowed eval JSON", windowed_ok)

    st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
    if st.button("Clear cache & reload", key="clear_cache_btn", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.pop("page_modules", None)
        for _key in list(sys.modules.keys()):
            if 'app_pages' in _key:
                del sys.modules[_key]
        st.rerun()

    st.markdown(
        '<div style="margin-top:16px; font-size:0.7rem; color:#334155;">'
        'v3 &middot; improvements/v3</div>',
        unsafe_allow_html=True,
    )

# ── Resolve page ───────────────────────────────────────────────────────────────
choice = st.session_state.selected_page

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

_page_logger = _logging.getLogger("app.pages")

if "page_modules" not in st.session_state:
    st.session_state["page_modules"] = {}
mod_key = str(module_path)
if mod_key not in st.session_state["page_modules"]:
    spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"app_pages.{module_path.stem}"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        _page_logger.exception("Failed to load page module %s", module_path.stem)
        st.error("An unexpected error occurred loading this page — check server logs.")
        st.stop()
    st.session_state["page_modules"][mod_key] = module
module = st.session_state["page_modules"][mod_key]

# Call the page's render method
page_class = getattr(module, selected_class, None)
if page_class is not None and hasattr(page_class, "render"):
    try:
        page_class.render()
    except Exception as e:
        _page_logger.exception("Error in %s.render()", selected_class)
        st.error("An unexpected error occurred — check server logs.")
elif hasattr(module, "render"):
    try:
        module.render()
    except Exception as e:
        _page_logger.exception("Error in module.render() for %s", module_path.stem)
        st.error("An unexpected error occurred — check server logs.")
else:
    st.error(f"Page '{selected_label}' does not expose a render() method.")
