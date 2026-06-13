"""Clinical metrics dashboard page."""
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import EVAL_FEDERATED_JSON, EVAL_LOCAL_JSON
from clinical_metrics import compute_all, sensitivity_at_specificity, specificity_at_sensitivity


def render():
    st.title("🏥 Clinical Metrics")
    st.markdown("Clinical performance beyond AUROC — what matters at the bedside.")

    result_files = {
        "Federated Model": EVAL_FEDERATED_JSON,
        "Local Model": EVAL_LOCAL_JSON,
    }

    loaded = {}
    for name, path in result_files.items():
        if Path(path).exists():
            with open(path) as f:
                loaded[name] = json.load(f)

    if not loaded:
        st.warning("No evaluation results found. Run the pipeline first.")
        st.code("python scripts/run_pipeline.py")
        return

    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

    for model_name, data in loaded.items():
        y_true = np.array(data.get("y_true", []))
        y_prob = np.array(data.get("y_prob", []))
        if len(y_true) == 0:
            continue

        st.subheader(f"📊 {model_name}")
        metrics = compute_all(y_true, y_prob, threshold=threshold)
        if "error" in metrics:
            st.warning(f"Cannot compute metrics: {metrics['error']}")
            continue

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sensitivity @ 95% Spec", f"{metrics['sensitivity_at_95spec']:.3f}")
        col2.metric("Specificity @ 90% Sens", f"{metrics['specificity_at_90sens']:.3f}")
        col3.metric("Alert Fatigue (FP/day)", f"{metrics['alert_fatigue_rate_per_day']:.2f}")
        col4.metric("NNAlert", f"{metrics['nn_alert']:.1f}")

        col5, col6, col7 = st.columns(3)
        col5.metric("Precision", f"{metrics['precision']:.3f}")
        col6.metric("Recall", f"{metrics['recall']:.3f}")
        col7.metric("F1", f"{metrics['f1']:.3f}")

        with st.expander("Confusion Matrix"):
            st.table({
                "": ["Predicted 0", "Predicted 1"],
                "Actual 0": [metrics["tn"], metrics["fp"]],
                "Actual 1": [metrics["fn"], metrics["tp"]],
            })
        st.divider()
