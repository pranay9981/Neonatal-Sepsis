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
    from clinical_metrics import (
        sensitivity_at_specificity,
        specificity_at_sensitivity,
        alert_fatigue_rate,
        nn_alert,
        compute_all,
    )
    _CLINICAL_AVAILABLE = True
except Exception:
    _CLINICAL_AVAILABLE = False

ROOT = Path(__file__).parent.parent

# (filename, display_name, color, calibrated_threshold)
KNOWN_EVAL_FILES = [
    ("eval_results_federated.json",      "Federated GRU-D (FedAvg, IID)",      "#3B82F6", 0.35),
    ("eval_results_noniid.json",         "Federated GRU-D (FedAvg, non-IID)",  "#F59E0B", 0.35),
    ("eval_results_fedbn.json",          "Federated GRU-D (FedBN)",            "#06B6D4", 0.35),
    ("eval_results_transformer_fl.json", "Federated Transformer (FedAvg)",     "#10B981", 0.35),
    ("eval_results_grud.json",           "GRU-D (Local)",                      "#EF4444", 0.3539),
    ("eval_results_transformer.json",    "Transformer (Local)",                "#8B5CF6", 0.3397),
    ("eval_results_ensemble.json",       "Ensemble",                           "#EC4899", 0.35),
    ("eval_results_windowed_grud.json",  "GRU-D Windowed (Early Warning, 6h)", "#A78BFA", 0.2294),
]


@st.cache_data
def _load(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _info_card(title, what_it_measures, why_it_matters, good_value, accent="#3B82F6"):
    return f"""
    <div style="background:#141827; border:1px solid #1E2A45; border-radius:10px;
         padding:16px 18px; margin:8px 0; border-top:3px solid {accent};
         box-sizing:border-box;">
      <div style="font-size:1rem; font-weight:700; color:#F1F5F9; margin-bottom:8px;">{title}</div>
      <div style="margin-bottom:6px;">
        <span style="font-size:0.78rem; font-weight:600; color:#475569; text-transform:uppercase;
              letter-spacing:0.4px;">What it measures</span><br>
        <span style="font-size:0.86rem; color:#94A3B8;">{what_it_measures}</span>
      </div>
      <div style="margin-bottom:6px;">
        <span style="font-size:0.78rem; font-weight:600; color:#475569; text-transform:uppercase;
              letter-spacing:0.4px;">Why it matters clinically</span><br>
        <span style="font-size:0.86rem; color:#94A3B8;">{why_it_matters}</span>
      </div>
      <div>
        <span style="font-size:0.78rem; font-weight:600; color:#475569; text-transform:uppercase;
              letter-spacing:0.4px;">What "good" looks like</span><br>
        <span style="font-size:0.86rem; font-weight:600; color:{accent};">{good_value}</span>
      </div>
    </div>"""


def _metric_display(value, label, sub=None, accent="#3B82F6"):
    sub_html = (
        f'<div style="font-size:0.78rem; color:#475569; margin-top:4px;">{sub}</div>'
        if sub else ""
    )
    return f"""
    <div style="background:#141827; border:1px solid #1E2A45; border-radius:10px;
         padding:18px 16px; text-align:center; border-top:3px solid {accent};
         box-shadow:0 0 20px rgba(59,130,246,0.07); box-sizing:border-box;">
      <div style="font-size:1.8rem; font-weight:700; color:#F1F5F9; line-height:1;">{value}</div>
      <div style="font-size:0.78rem; color:#475569; margin-top:6px; text-transform:uppercase;
           letter-spacing:0.5px;">{label}</div>
      {sub_html}
    </div>"""


class ClinicalMetricsPage:
    @staticmethod
    def render():
        # ── Page header ───────────────────────────────────────────────────────
        st.markdown("""
        <div style="margin-bottom:28px; padding-bottom:16px; border-bottom:1px solid #1E2A45;">
          <div style="display:inline-block; background:rgba(16,185,129,0.10); color:#34D399;
               padding:3px 10px; border-radius:20px; font-size:0.7rem; font-weight:700;
               letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;">CLINICAL</div>
          <div style="font-size:1.6rem; font-weight:700; color:#F1F5F9; letter-spacing:-0.3px;">
            🏥 Clinical Metrics
          </div>
          <div style="font-size:0.88rem; color:#64748B; margin-top:6px; max-width:640px;">
            What matters at the bedside — beyond AUROC and accuracy.
            Standard ML metrics don't tell you if a model is safe to use in a hospital.
            These metrics translate model performance into clinical impact.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Why standard ML metrics aren't enough ─────────────────────────────
        st.markdown("### Why AUROC alone isn't enough")
        st.markdown("""
        AUROC = 0.92 sounds good. But it doesn't answer the questions that matter to a clinician:

        - **"If I use this model, how many real sepsis cases will I miss?"**
        - **"How many false alarms will it generate per shift?"**
        - **"On average, how many alerts do I have to act on to find one true case?"**

        The four clinical metrics below answer these questions directly.
        """)

        # ── Metric explanations ───────────────────────────────────────────────
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
                    "#3B82F6",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                _info_card(
                    "Alert Fatigue Rate (FP/day)",
                    "How many false-positive alerts does the model generate per day "
                    "(based on the test set positive rate and patient-hours in the dataset)?",
                    "Alarm fatigue is a documented patient safety problem. If a NICU of 20 beds "
                    "gets 40 false sepsis alerts per shift, staff start ignoring them — and miss "
                    "real ones. This metric tells you the operational burden of the model.",
                    "< 5 per day — fewer than 1 false alarm per 4-hour block",
                    "#F59E0B",
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
                    "#10B981",
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
                    "#8B5CF6",
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div style="border-top:1px solid #1E2A45; margin:24px 0;"></div>',
            unsafe_allow_html=True,
        )

        if not _CLINICAL_AVAILABLE:
            st.error(
                "`clinical_metrics` module could not be imported. "
                "Check that `src/clinical_metrics.py` exists and `src/` is in the Python path."
            )
            return

        # ── Load all available eval files ─────────────────────────────────────
        available = {}   # display_name -> (data, color, calibrated_threshold)
        for fname, display_name, color, cal_threshold in KNOWN_EVAL_FILES:
            data = _load(str(ROOT / fname))
            if data is not None:
                available[display_name] = (data, color, cal_threshold)

        if not available:
            st.warning("No evaluation results found. Run the pipeline first.")
            st.code("python src/evaluate.py --index data/splits/test_index.pt "
                    "--ckpt server_out/global_best.pt --model grud "
                    "--out_file eval_results_federated.json")
            return

        # ── Summary table (all models, calibrated thresholds) ─────────────────
        st.markdown("### Clinical metrics summary — all models")
        st.markdown(
            '<span style="font-size:0.85rem; color:#64748B;">Computed at each model\'s '
            'calibrated threshold. Threshold-independent metrics (Sens@Spec, Spec@Sens) '
            'reflect the full ROC curve.</span>',
            unsafe_allow_html=True,
        )

        # Separate patient-level from windowed
        patient_models  = {k: v for k, v in available.items() if "windowed" not in k.lower()}
        windowed_models = {k: v for k, v in available.items() if "windowed" in k.lower()}

        def _build_summary_rows(models_dict):
            rows = []
            for name, (data, color, cal_thr) in models_dict.items():
                y_true = np.array(data["y_true"])
                y_prob = np.array(data["y_prob"])
                if len(np.unique(y_true)) < 2:
                    continue
                sens95, _ = sensitivity_at_specificity(y_true, y_prob, 0.95)
                spec90, _ = specificity_at_sensitivity(y_true, y_prob, 0.90)
                y_pred = (y_prob >= cal_thr).astype(int)
                n = len(y_true)
                afr = alert_fatigue_rate(y_true, y_pred, patient_hours=float(n * 48))
                metrics = compute_all(y_true, y_pred)
                sens = metrics.get("sensitivity", float("nan"))
                spec = metrics.get("specificity", float("nan"))
                ppv = metrics.get("precision", float("nan"))
                nna = metrics.get("nn_alert", float("nan"))
                rows.append({
                    "Model": name,
                    "Threshold": f"{cal_thr:.4f}",
                    "AUROC": f"{data.get('auroc', float('nan')):.4f}",
                    "Sens@95Spec": f"{sens95:.3f}",
                    "Spec@90Sens": f"{spec90:.3f}",
                    "Sensitivity": f"{sens:.3f}" if not np.isnan(sens) else "N/A",
                    "Specificity": f"{spec:.3f}" if not np.isnan(spec) else "N/A",
                    "PPV": f"{ppv:.3f}" if not np.isnan(ppv) else "N/A",
                    "NNAlert": f"{nna:.1f}" if not np.isnan(nna) else "N/A",
                    "FA/day": f"{afr:.2f}",
                })
            return rows

        if patient_models:
            import pandas as pd
            rows = _build_summary_rows(patient_models)
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
                st.caption("FA/day: per patient-day (assuming 48h stay per test patient)")

        if windowed_models:
            st.markdown(
                '<div style="background:rgba(245,158,11,0.08); border-left:4px solid #F59E0B;'
                'border-radius:0 8px 8px 0; padding:10px 16px; color:#FCD34D; '
                'margin:12px 0; font-size:0.86rem;">'
                '<b>Early Warning model below (different task)</b> — predicts sepsis onset in next 6h '
                'from a 12h window. Not directly comparable to the patient-level models above.</div>',
                unsafe_allow_html=True,
            )
            rows = _build_summary_rows(windowed_models)
            if rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        st.markdown(
            '<div style="border-top:1px solid #1E2A45; margin:24px 0;"></div>',
            unsafe_allow_html=True,
        )

        # ── Interactive threshold explorer ─────────────────────────────────────
        st.markdown("### Interactive threshold explorer")
        st.markdown(
            '<div style="background:rgba(59,130,246,0.08); border-left:4px solid #3B82F6;'
            'border-radius:0 8px 8px 0; padding:14px 18px; color:#93C5FD; margin-bottom:20px;">'
            '<b>About the threshold slider:</b> The model outputs a probability (0–1). '
            'Calling a patient "high risk" requires choosing a cutoff. A lower threshold catches '
            'more sepsis cases (higher recall) but triggers more false alarms. The slider lets you '
            'explore this trade-off. The Sens@Spec and Spec@Sens metrics above are threshold-independent '
            '— they reflect the full ROC curve, not a single cutoff.</div>',
            unsafe_allow_html=True,
        )

        model_name = st.selectbox("Select model:", list(available.keys()))
        data, color, cal_thr = available[model_name]

        threshold = st.slider(
            "Decision threshold",
            0.0, 1.0, float(cal_thr), 0.01,
            help="Adjusts precision/recall/F1 below. Sens@Spec and Spec@Sens are threshold-independent.",
        )

        y_true = np.array(data.get("y_true", []))
        y_prob = np.array(data.get("y_prob", []))

        if len(y_true) == 0:
            st.warning("No y_true/y_prob data in this eval file.")
            return

        metrics = compute_all(y_true, y_prob, threshold=threshold,
                              patient_hours=float(len(y_true) * 48))

        if "error" in metrics:
            st.warning(f"Cannot compute metrics: {metrics['error']}")
            return

        # Primary clinical metrics row
        c1, c2, c3, c4 = st.columns(4)
        sens95, _ = sensitivity_at_specificity(y_true, y_prob, 0.95)
        spec90, _ = specificity_at_sensitivity(y_true, y_prob, 0.90)
        af  = metrics.get("alert_fatigue_rate_per_day", float("nan"))
        nna = metrics.get("nn_alert", float("nan"))

        c1.markdown(_metric_display(
            f"{sens95:.3f}", "Sensitivity @ 95% Spec",
            "Catch rate at low false-alarm setting", "#3B82F6",
        ), unsafe_allow_html=True)
        c2.markdown(_metric_display(
            f"{spec90:.3f}", "Specificity @ 90% Sens",
            "False-alarm rate at high-recall setting", "#10B981",
        ), unsafe_allow_html=True)
        c3.markdown(_metric_display(
            f"{af:.2f}", "Alert Fatigue (FP/day)",
            "False alerts per day operational burden", "#F59E0B",
        ), unsafe_allow_html=True)
        c4.markdown(_metric_display(
            f"{nna:.1f}", "NNAlert",
            "Investigations per confirmed case", "#8B5CF6",
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
        tn = metrics.get("tn", 0)
        fp = metrics.get("fp", 0)
        c8.metric("Specificity", f"{tn / max(tn + fp, 1):.3f}",
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
