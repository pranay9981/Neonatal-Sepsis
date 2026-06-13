# 1_00_📘_Project_Summary.py
import streamlit as st


# Inline-styled card helpers — these work regardless of Streamlit theme
def _metric_card(value, label):
    return f"""
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
         padding:20px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.07);">
      <div style="font-size:2rem;font-weight:700;color:#1E3A5F;line-height:1;">{value}</div>
      <div style="font-size:0.75rem;color:#64748B;margin-top:6px;text-transform:uppercase;
           letter-spacing:0.6px;">{label}</div>
    </div>"""


def _arch_card(title, subtitle, body):
    return f"""
    <div style="background:#F8FAFC;border-left:4px solid #2196F3;border-radius:0 8px 8px 0;
         padding:14px 18px;margin:10px 0;color:#1E293B;">
      <div style="font-size:1rem;font-weight:700;color:#1E3A5F;margin-bottom:4px;">{title}</div>
      <div style="font-size:0.82rem;color:#2563EB;margin-bottom:6px;">{subtitle}</div>
      <div style="font-size:0.84rem;color:#475569;line-height:1.5;">{body}</div>
    </div>"""


def _step(num, title, desc):
    return f"""
    <div style="display:flex;align-items:flex-start;margin:10px 0;">
      <div style="background:#1E3A5F;color:white;border-radius:50%;width:28px;height:28px;
           display:flex;align-items:center;justify-content:center;font-size:0.8rem;
           font-weight:700;flex-shrink:0;margin-top:2px;">{num}</div>
      <div style="margin-left:12px;">
        <div style="font-weight:600;color:#1E293B;">{title}</div>
        <div style="font-size:0.83rem;color:#64748B;margin-top:2px;">{desc}</div>
      </div>
    </div>"""


def _callout(color_bg, color_border, color_text, text):
    return f"""
    <div style="background:{color_bg};border-left:4px solid {color_border};
         border-radius:0 8px 8px 0;padding:12px 16px;color:{color_text};margin:10px 0;">
      {text}
    </div>"""


class ProjectSummaryPage:
    @staticmethod
    def render():
        # ── Hero ──────────────────────────────────────────────────────────
        st.markdown("""
        <div style="background:linear-gradient(90deg,#1E3A5F 0%,#2563EB 100%);
             color:white;padding:20px 28px 16px 28px;border-radius:10px;margin-bottom:24px;
             box-sizing:border-box;width:100%;">
          <div style="font-size:1.7rem;font-weight:700;letter-spacing:-0.3px;">
            👶 Neonatal Sepsis Detection
          </div>
          <div style="font-size:0.92rem;opacity:0.85;margin-top:4px;">
            Federated Learning pipeline — privacy-first early warning from 48-hour clinical time-series
          </div>
          <div style="font-size:0.82rem;opacity:0.7;margin-top:8px;">
            Patient data never leaves the hospital. Only model weights are shared across sites.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── What is this project? ─────────────────────────────────────────
        st.markdown("### What is this project?")
        st.markdown("""
        This system detects **neonatal sepsis** early — before it becomes life-threatening — using
        machine learning on routine ICU measurements. Sepsis in newborns kills within hours if missed,
        but the early signs (subtle changes in heart rate, oxygen, labs) are buried in noisy clinical data.

        The core challenge: **hospitals can't share patient data** (privacy laws). So we use
        **Federated Learning** — each hospital trains locally, only the model improvements are shared.
        The result is a stronger model than any single hospital could build alone.
        """)

        # ── Stat cards ─────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        for col, (val, label) in zip(
            [c1, c2, c3, c4],
            [("40,323", "Patient Files"), ("40", "Clinical Features"),
             ("48", "Timesteps / Window"), ("3", "FL Clients")],
        ):
            col.markdown(_metric_card(val, label), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Architecture | Pipeline ────────────────────────────────────────
        left, right = st.columns(2)

        with left:
            st.markdown("### Model Architecture")
            st.markdown(
                _arch_card(
                    "TimeSeriesTransformer",
                    "Primary model — handles temporal context",
                    "Transformer encoder with a CLS token that attends over all 48 timesteps "
                    "simultaneously. Sinusoidal positional embeddings encode ICU hour. Pad masking "
                    "ignores zero-padded missing hours. Good at capturing long-range correlations "
                    "between vitals and lab values across an entire ICU stay."
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                _arch_card(
                    "GRU-D",
                    "Secondary model — handles irregular sampling",
                    "Gated Recurrent Unit with learned decay. In real ICUs, lab tests aren't drawn "
                    "every hour — values go missing for 4–12 hours. GRU-D models the <em>time since "
                    "last observation</em> per feature, gracefully handling sparse measurements by "
                    "decaying toward the population mean when data is missing."
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                _arch_card(
                    "Ensemble + Temperature Scaling",
                    "Blends both models, calibrates probabilities",
                    "Final score = α·Transformer + (1-α)·GRU-D (α=0.5). Then Temperature Scaling "
                    "(LBFGS on validation set) adjusts the output so that <em>predicted probability "
                    "≈ true frequency</em> — a score of 0.8 should mean ~80% of patients at that "
                    "score actually have sepsis."
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                _arch_card(
                    "Federated Learning (Flower)",
                    "FedAvg · FedProx · FedBN — no data leaves each site",
                    "Server coordinates training rounds. Each client trains on local patients for "
                    "1 epoch, sends gradients to server. Server averages them (FedAvg) or applies "
                    "proximal regularisation (FedProx μ=0.01) to prevent client drift. FedBN keeps "
                    "BatchNorm statistics local to each hospital's data distribution."
                ),
                unsafe_allow_html=True,
            )

        with right:
            st.markdown("### Pipeline Steps")
            steps = [
                ("1", "Preprocess raw PSV files",
                 "Reads 40,323 pipe-separated patient files. Computes X (features), mask "
                 "(which values were actually measured), delta (hours since last measurement), "
                 "and per-timestep sepsis labels. Output: one .pt tensor per patient."),
                ("2", "Freeze train/val/test splits",
                 "Stratified 70/15/15 patient-level split created ONCE and locked. The test "
                 "set is never touched during training or model selection — prevents any data "
                 "leakage into evaluation."),
                ("3", "Local baseline training",
                 "Trains Transformer or GRU-D on the 70% train split using focal loss "
                 "(handles 7–9% sepsis rate), warmup+cosine LR schedule, gradient clipping, "
                 "and early stopping. Checkpoint saved to runs/."),
                ("4", "Federated learning simulation",
                 "Flower server coordinates N clients. Each client holds a private partition "
                 "of train patients. Server aggregates updates for 20 rounds. Best global "
                 "model saved to server_out/global_best.pt."),
                ("5", "Evaluation on frozen test set",
                 "Loads the frozen test_index.pt (never seen during training). Computes "
                 "AUROC, AUPRC, F1, calibration error, and 95% bootstrap CIs. Outputs "
                 "eval_results_federated.json and eval_results_local.json."),
                ("6", "Comparison plots",
                 "ROC and Precision-Recall curves with confidence bands for both models. "
                 "Visual confirmation that federated model generalises better."),
                ("7", "Serve via FastAPI",
                 "FastAPI v2 endpoint at /v2/predict. MC Dropout for uncertainty intervals. "
                 "Prometheus /metrics for monitoring. Every prediction logged to audit JSONL."),
            ]
            for num, title, desc in steps:
                st.markdown(_step(num, title, desc), unsafe_allow_html=True)

        st.markdown("---")

        # ── Dataset: what the data looks like ─────────────────────────────
        st.markdown("### Dataset: What Each Patient File Contains")
        st.markdown("""
        Each patient is one `.psv` (pipe-separated) file with one row per ICU hour.
        The model sees the **last 48 hours** of measurements before a potential sepsis event.
        """)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("**Feature categories (40 total)**")
            cats = {
                "Vital Signs (8)": "HR · O2Sat · Temp · SBP · MAP · DBP · Resp · EtCO2",
                "Blood Gas (10)": "pH · PaCO2 · HCO3 · BaseExcess · SaO2 · Chloride · K · Ca · Mg · PO4",
                "Metabolic / Organ (10)": "Glucose · Lactate · BUN · Creatinine · AST · ALP · Bili (direct+total) · TroponinI · FiO2",
                "Hematology (6)": "Hct · Hgb · WBC · Platelets · Fibrinogen · PTT",
                "Metadata (6)": "Age (days) · Gender · Unit1 · Unit2 · HospAdmTime · ICULOS",
            }
            for cat, feats in cats.items():
                st.markdown(
                    f'<div style="margin:6px 0;">'
                    f'<span style="font-weight:600;color:#1E3A5F;">{cat}</span><br>'
                    f'<span style="font-size:0.83rem;color:#475569;">{feats}</span></div>',
                    unsafe_allow_html=True,
                )

        with col_b:
            st.markdown("**Key data challenges this pipeline handles**")
            challenges = [
                ("Sparse labs", "Lab tests drawn every 4–12 h, not every hour. GRU-D models the gap with learned decay."),
                ("Class imbalance", "Only 7–9% of windows are positive. Focal loss focuses training on hard examples."),
                ("Variable length", "Patients with <48 ICU hours are zero-padded. Pad masking prevents those zeros from misleading attention."),
                ("Site heterogeneity", "Different hospitals measure different labs at different frequencies. FedBN keeps BatchNorm stats local per site."),
                ("Calibration", "Raw model outputs are overconfident. Temperature Scaling fixes this so probabilities are trustworthy at the bedside."),
            ]
            for title, desc in challenges:
                st.markdown(
                    f'<div style="background:#F1F5F9;border-radius:6px;padding:8px 12px;margin:5px 0;">'
                    f'<span style="font-weight:600;color:#1E3A5F;">{title}:</span> '
                    f'<span style="font-size:0.84rem;color:#334155;">{desc}</span></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Key design decisions ───────────────────────────────────────────
        st.markdown("### Key Design Decisions")
        d1, d2, d3 = st.columns(3)
        decisions = [
            ("#E3F2FD", "#2196F3", "#0D47A1",
             "<b>Frozen test set</b>",
             "Split is created once and locked before any training begins. "
             "No hyperparameter or model selection decision ever sees test data."),
            ("#E8F5E9", "#4CAF50", "#1B5E20",
             "<b>Patient-level split</b>",
             "Each patient is entirely in one split (never split across train/val/test). "
             "Prevents time-series leakage where early hours train and late hours evaluate."),
            ("#FFF3E0", "#FF9800", "#E65100",
             "<b>FL test exclusion</b>",
             "Patients in the frozen test set are excluded from all FL client partitions. "
             "Federated clients only see train/val patients — zero leakage guaranteed."),
        ]
        for col, (bg, border, text, title, body) in zip([d1, d2, d3], decisions):
            col.markdown(
                f'<div style="background:{bg};border-left:4px solid {border};'
                f'border-radius:0 8px 8px 0;padding:14px 16px;color:{text};height:100%;">'
                f'<div style="font-size:0.95rem;margin-bottom:6px;">{title}</div>'
                f'<div style="font-size:0.83rem;font-weight:400;opacity:0.9;">{body}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            _callout(
                "#EFF6FF", "#2563EB", "#1E3A5F",
                "<b>Navigation guide:</b> Use <b>Predict</b> to run inference on a patient file. "
                "Use <b>Model Metrics</b> to compare Federated vs Local AUROC/AUPRC. "
                "Use <b>Clinical Metrics</b> to see bedside-relevant stats (sensitivity at specificity, "
                "alert fatigue, NNAlert). Use <b>Training Runs</b> to browse saved checkpoints."
            ),
            unsafe_allow_html=True,
        )
