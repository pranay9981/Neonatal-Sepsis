# app.py
"""
Launcher for the multi-page Streamlit dashboard.
It dynamically loads page modules from the app_pages directory and calls their render() method.
"""

import streamlit as st
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).parent
PAGES_DIR = ROOT / "app_pages"

# Map: label -> filename
PAGES = [
    ("📘 Project Summary", "1_00_📘_Project_Summary.py"),
    ("📈 Predict", "1_03_📈_Predict.py"),
    ("🧪 Model Metrics", "1_04_🧪_Model_Metrics.py")
]

st.set_page_config(page_title="Neonatal Sepsis Dashboard", page_icon="👶", layout="wide")
st.title("👶 Neonatal Sepsis Dashboard")

choice = st.sidebar.radio("Select Page", [p[0] for p in PAGES])

# find selected filename
selected_file = None
for label, fname in PAGES:
    if label == choice:
        selected_file = fname
        break

if selected_file is None:
    st.error("Selected page not found")
else:
    module_path = PAGES_DIR / selected_file
    if not module_path.exists():
        st.error(f"Page file not found: {module_path}")
    else:
        # Dynamically load module from file and call render()
        spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_path.stem] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            st.exception(e)
        else:
            # Try calling class-based renderers first
            try:
                if choice.startswith("📘") and hasattr(module, "ProjectSummaryPage") and hasattr(module.ProjectSummaryPage, "render"):
                    module.ProjectSummaryPage.render()
                elif choice.startswith("📈") and hasattr(module, "PredictPage") and hasattr(module.PredictPage, "render"):
                    module.PredictPage.render()
                elif choice.startswith("🧪") and hasattr(module, "MetricsPage") and hasattr(module.MetricsPage, "render"):
                    module.MetricsPage.render()
                elif hasattr(module, "render"):
                    module.render()
                else:
                    st.error("The page module does not expose a render() function or expected class.")
            except Exception as e:
                st.exception(e)
