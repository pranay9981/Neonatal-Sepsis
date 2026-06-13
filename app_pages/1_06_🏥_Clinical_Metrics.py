"""Clinical metrics dashboard page."""
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from config import EVAL_FEDERATED_JSON as _FED_PATH, EVAL_LOCAL_JSON as _LOC_PATH
    EVAL_FEDERATED_JSON = Path(_FED_PATH)
    EVAL_LOCAL_JSON = Path(_LOC_PATH)
except Exception:
    EVAL_FEDERATED_JSON = Path("eval_results_federated.json")
    EVAL_LOCAL_JSON = Path("eval_results_local.json")

try:
    from clinical_metrics import compute_all
    _CLINICAL_METRICS_AVAILABLE = True
except Exception:
    _CLINICAL_METRICS_AVAILABLE = False


class ClinicalMetricsPage:
    @staticmethod
    def render():
        st.markdown("""
        <div class="main-header">
          <h1>🏥 Clinical Metrics</h1>
          <p>Bedside-relevant performance — beyond AUROC</p>
        </div>
        """, unsafe_allow_html=True)

        if not _CLINICAL_METRICS_AVAILABLE:
            st.error("clinical_metrics module could not be imported. Check that `src/clinical_metrics.py` exists.")
            return

        result_files = {
            "Federated Model": EVAL_FEDERATED_JSON,
            "Local Model": EVAL_LOCAL_JSON,
        }

        loaded = {}
        for name, path in result_files.items():
            if Path(path).exists():
                try:
                    with open(path) as f:
                        loaded[name] = json.load(f)
                except Exception:
                    pass

        if not loaded:
            st.warning("No evaluation results found. Run the pipeline first.")
            st.code("python scripts/run_pipeline.py")
            st.markdown("""
            <div class="callout-info">
            Or run evaluation manually:<br>
            <code>python src/evaluate.py --index data/splits/test_index.pt --ckpt server_out/global_best.pt --model transformer --out_file eval_results_federated.json</code>
            </div>
            """, unsafe_allow_html=True)
            return

        # Metric explanations
        with st.expander("What do these metrics mean?"):
            st.markdown("""
            | Metric | Meaning |
            |---|---|
            | **Sensitivity @ 95% Specificity** | Recall when we allow only 5% false-alarm rate — how many true sepsis cases we catch |
            | **Specificity @ 90% Sensitivity** | Specificity when recall is forced to 90% — false-alarm rate at high recall |
            | **Alert Fatigue (FP/day)** | False positive alerts per day (assumes 1 alert per patient per shift ≈ 8 h) |
            | **NNAlert** | Number Needed to Alert — how many alerts to find one true sepsis case |
            """)

        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01,
                              help="Adjusts the classification cutoff across all models below")

        for model_name, data in loaded.items():
            y_true = np.array(data.get("y_true", []))
            y_prob = np.array(data.get("y_prob", []))
            if len(y_true) == 0:
                st.warning(f"{model_name}: no y_true data found.")
                continue

            st.markdown(f"### {model_name}")
            metrics = compute_all(y_true, y_prob, threshold=threshold)

            if "error" in metrics:
                st.warning(f"Cannot compute metrics: {metrics['error']}")
                st.divider()
                continue

            # Primary clinical metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{metrics.get('sensitivity_at_95spec', float('nan')):.3f}</div>
              <div class="metric-label">Sensitivity @ 95% Spec</div>
            </div>
            """, unsafe_allow_html=True)
            c2.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{metrics.get('specificity_at_90sens', float('nan')):.3f}</div>
              <div class="metric-label">Specificity @ 90% Sens</div>
            </div>
            """, unsafe_allow_html=True)
            c3.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{metrics.get('alert_fatigue_rate_per_day', float('nan')):.2f}</div>
              <div class="metric-label">Alert Fatigue (FP/day)</div>
            </div>
            """, unsafe_allow_html=True)
            c4.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{metrics.get('nn_alert', float('nan')):.1f}</div>
              <div class="metric-label">NNAlert</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Standard classification metrics
            c5, c6, c7 = st.columns(3)
            c5.metric("Precision", f"{metrics.get('precision', float('nan')):.3f}")
            c6.metric("Recall", f"{metrics.get('recall', float('nan')):.3f}")
            c7.metric("F1-Score", f"{metrics.get('f1', float('nan')):.3f}")

            # Confusion matrix
            with st.expander("Confusion Matrix"):
                tn = metrics.get("tn", 0)
                fp = metrics.get("fp", 0)
                fn = metrics.get("fn", 0)
                tp = metrics.get("tp", 0)
                st.table({
                    "": ["Predicted Negative", "Predicted Positive"],
                    "Actual Negative": [f"TN = {tn}", f"FP = {fp}"],
                    "Actual Positive": [f"FN = {fn}", f"TP = {tp}"],
                })

            st.divider()
