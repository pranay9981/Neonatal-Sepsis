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
    _CLINICAL_AVAILABLE = True
except Exception:
    _CLINICAL_AVAILABLE = False


def _info_card(title, what_it_measures, why_it_matters, good_value, color="#1E3A5F"):
    return f"""
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
         padding:16px 18px;margin:8px 0;border-top:3px solid {color};">
      <div style="font-size:1rem;font-weight:700;color:{color};margin-bottom:8px;">{title}</div>
      <div style="margin-bottom:6px;">
        <span style="font-size:0.78rem;font-weight:600;color:#64748B;text-transform:uppercase;
              letter-spacing:0.4px;">What it measures</span><br>
        <span style="font-size:0.86rem;color:#334155;">{what_it_measures}</span>
      </div>
      <div style="margin-bottom:6px;">
        <span style="font-size:0.78rem;font-weight:600;color:#64748B;text-transform:uppercase;
              letter-spacing:0.4px;">Why it matters clinically</span><br>
        <span style="font-size:0.86rem;color:#334155;">{why_it_matters}</span>
      </div>
      <div>
        <span style="font-size:0.78rem;font-weight:600;color:#64748B;text-transform:uppercase;
              letter-spacing:0.4px;">What "good" looks like</span><br>
        <span style="font-size:0.86rem;font-weight:600;color:{color};">{good_value}</span>
      </div>
    </div>"""


def _metric_display(value, label, sub=None, color="#1E3A5F"):
    sub_html = f'<div style="font-size:0.78rem;color:#64748B;margin-top:4px;">{sub}</div>' if sub else ""
    return f"""
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
         padding:18px 16px;text-align:center;border-top:3px solid {color};">
      <div style="font-size:1.8rem;font-weight:700;color:{color};line-height:1;">{value}</div>
      <div style="font-size:0.78rem;color:#64748B;margin-top:6px;text-transform:uppercase;
           letter-spacing:0.5px;">{label}</div>
      {sub_html}
    </div>"""


class ClinicalMetricsPage:
    @staticmethod
    def render():
        # ── Header ────────────────────────────────────────────────────────
        st.markdown("""
        <div style="background:linear-gradient(90deg,#1B5E20 0%,#2E7D32 100%);
             color:white;padding:20px 28px 16px 28px;border-radius:10px;margin-bottom:24px;
             box-sizing:border-box;width:100%;">
          <div style="font-size:1.7rem;font-weight:700;">🏥 Clinical Metrics</div>
          <div style="font-size:0.92rem;opacity:0.85;margin-top:4px;">
            What matters at the bedside — beyond AUROC and accuracy
          </div>
          <div style="font-size:0.82rem;opacity:0.7;margin-top:8px;">
            Standard ML metrics don't tell you if a model is safe to use in a hospital.
            These metrics translate model performance into clinical impact.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Why standard ML metrics aren't enough ────────────────────────
        st.markdown("### Why AUROC alone isn't enough")
        st.markdown("""
        AUROC = 0.85 sounds good. But it doesn't answer the questions that matter to a clinician:

        - **"If I use this model, how many real sepsis cases will I miss?"**
        - **"How many false alarms will it generate per shift?"**
        - **"On average, how many alerts do I have to act on to find one true case?"**

        The four clinical metrics below answer these questions directly.
        """)

        # ── Metric explanations ───────────────────────────────────────────
        st.markdown("### What each metric measures")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                _info_card(
                    "Sensitivity @ 95% Specificity",
                    "At the threshold where only 5% of healthy patients trigger an alert "
                    "(95% specificity = very few false alarms), what fraction of true sepsis "
                    "cases does the model still catch?",
                    "In a NICU, false alarms cause alarm fatigue — nurses start ignoring alerts. "
                    "Setting specificity to 95% keeps false alarms low. Sensitivity at that point "
                    "tells you how many real sepsis cases you're still catching.",
                    "≥ 0.70 — catching 70%+ of sepsis at only 5% false-alarm rate",
                    "#1565C0",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                _info_card(
                    "Alert Fatigue Rate (FP/day)",
                    "How many false-positive alerts does the model generate per day "
                    "(based on the test set positive rate and 8-hour shifts)?",
                    "Alarm fatigue is a documented patient safety problem. If a NICU of 20 beds "
                    "gets 40 false sepsis alerts per shift, staff start ignoring them — and miss "
                    "real ones. This metric tells you the operational burden of the model.",
                    "< 5 per day — fewer than 1 false alarm per 4-hour block",
                    "#E65100",
                ),
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                _info_card(
                    "Specificity @ 90% Sensitivity",
                    "When the threshold is set so the model catches 90% of all sepsis cases "
                    "(very high recall), what fraction of healthy patients does it correctly "
                    "identify as healthy (not alert on)?",
                    "High sensitivity is critical — missing sepsis is dangerous. But if catching "
                    "90% of cases means alerting on every patient, the model is useless. This "
                    "metric shows the false-alarm cost of achieving high recall.",
                    "≥ 0.70 — 70%+ of healthy patients correctly cleared at 90% recall",
                    "#1B5E20",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                _info_card(
                    "NNAlert (Number Needed to Alert)",
                    "On average, how many patients does a clinician need to respond to "
                    "before finding one true sepsis case? Inverse of precision: 1 / precision.",
                    "If NNAlert = 10, a clinician must investigate 10 alerts to find 1 real "
                    "sepsis patient. NNAlert = 2 means every other alert is real. Lower is better. "
                    "It directly measures how much clinician time the model wastes.",
                    "< 5 — at most 5 investigations per confirmed sepsis case",
                    "#6A1B9A",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Threshold explanation ─────────────────────────────────────────
        st.markdown("""
        <div style="background:#E3F2FD;border-left:4px solid #2196F3;border-radius:0 8px 8px 0;
             padding:14px 18px;color:#0D47A1;margin-bottom:20px;">
        <b>About the threshold slider below:</b> The model outputs a probability (0–1).
        Calling a patient "high risk" requires choosing a cutoff. A lower threshold catches
        more sepsis cases (higher recall) but triggers more false alarms. The slider lets you
        explore this trade-off. The four metrics above are computed from the full ROC curve
        and don't depend on this threshold — but precision, recall, and F1 do.
        </div>
        """, unsafe_allow_html=True)

        # ── Check dependencies ────────────────────────────────────────────
        if not _CLINICAL_AVAILABLE:
            st.error(
                "`clinical_metrics` module could not be imported. "
                "Check that `src/clinical_metrics.py` exists and `src/` is in the Python path."
            )
            return

        # ── Load eval files ───────────────────────────────────────────────
        loaded = {}
        for name, path in [("Federated Model", EVAL_FEDERATED_JSON),
                            ("Local Model", EVAL_LOCAL_JSON)]:
            if Path(path).exists():
                try:
                    with open(path) as f:
                        loaded[name] = json.load(f)
                except Exception:
                    pass

        if not loaded:
            st.warning("No evaluation results found. Run the pipeline first.")
            st.code("python scripts/run_pipeline.py")
            return

        threshold = st.slider(
            "Decision threshold",
            0.0, 1.0, 0.5, 0.01,
            help="Adjusts precision/recall/F1 below. Sensitivity@spec and Specificity@sens are threshold-independent.",
        )

        for model_name, data in loaded.items():
            y_true = np.array(data.get("y_true", []))
            y_prob = np.array(data.get("y_prob", []))
            if len(y_true) == 0:
                continue

            st.markdown(f"### {model_name}")
            metrics = compute_all(y_true, y_prob, threshold=threshold)

            if "error" in metrics:
                st.warning(f"Cannot compute metrics: {metrics['error']}")
                st.divider()
                continue

            # Primary clinical metrics row
            c1, c2, c3, c4 = st.columns(4)
            sens95 = metrics.get("sensitivity_at_95spec", float("nan"))
            spec90 = metrics.get("specificity_at_90sens", float("nan"))
            af     = metrics.get("alert_fatigue_rate_per_day", float("nan"))
            nna    = metrics.get("nn_alert", float("nan"))

            c1.markdown(_metric_display(
                f"{sens95:.3f}", "Sensitivity @ 95% Spec",
                "Catch rate at low false-alarm setting", "#1565C0",
            ), unsafe_allow_html=True)
            c2.markdown(_metric_display(
                f"{spec90:.3f}", "Specificity @ 90% Sens",
                "False-alarm rate at high-recall setting", "#1B5E20",
            ), unsafe_allow_html=True)
            c3.markdown(_metric_display(
                f"{af:.2f}", "Alert Fatigue (FP/day)",
                "False alerts per day operational burden", "#E65100",
            ), unsafe_allow_html=True)
            c4.markdown(_metric_display(
                f"{nna:.1f}", "NNAlert",
                "Investigations per confirmed case", "#6A1B9A",
            ), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Threshold-dependent metrics
            st.markdown(f"**At threshold = {threshold:.2f}:**")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Precision", f"{metrics.get('precision', float('nan')):.3f}",
                      help="Of all patients flagged, what fraction truly have sepsis?")
            c6.metric("Recall", f"{metrics.get('recall', float('nan')):.3f}",
                      help="Of all true sepsis patients, what fraction were flagged?")
            c7.metric("F1-Score", f"{metrics.get('f1', float('nan')):.3f}",
                      help="Harmonic mean of precision and recall")
            c8.metric("Specificity", f"{1 - metrics.get('fp', 0) / max(metrics.get('tn', 0) + metrics.get('fp', 0), 1):.3f}",
                      help="Of all healthy patients, what fraction were correctly cleared?")

            # Confusion matrix
            with st.expander("Confusion matrix at this threshold"):
                tn = metrics.get("tn", 0)
                fp = metrics.get("fp", 0)
                fn = metrics.get("fn", 0)
                tp = metrics.get("tp", 0)
                total = tn + fp + fn + tp
                st.markdown(f"""
                | | Predicted: No Sepsis | Predicted: Sepsis |
                |---|---|---|
                | **Actual: No Sepsis** | ✅ TN = {tn} ({tn/max(total,1):.1%}) | ❌ FP = {fp} ({fp/max(total,1):.1%}) |
                | **Actual: Sepsis** | ❌ FN = {fn} ({fn/max(total,1):.1%}) | ✅ TP = {tp} ({tp/max(total,1):.1%}) |

                - **TN** (True Negative): healthy patient correctly cleared — no alert, no wasted time
                - **FP** (False Positive): healthy patient incorrectly flagged — wasted clinical time (alarm fatigue)
                - **FN** (False Negative): sepsis patient missed — most dangerous outcome
                - **TP** (True Positive): sepsis patient correctly caught — the whole point of the model
                """)

            st.divider()
