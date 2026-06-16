# 1_00_📘_Project_Summary.py
import streamlit as st


# ── Dark-theme inline card helpers ────────────────────────────────────────────
def _metric_card(value, label, accent="#3B82F6"):
    return f"""
    <div style="background:#141827; border:1px solid #1E2A45; border-top:3px solid {accent};
         border-radius:12px; padding:22px 16px; text-align:center;
         box-shadow:0 0 20px rgba(59,130,246,0.07); box-sizing:border-box;">
      <div style="font-size:2rem; font-weight:700; color:#F1F5F9; line-height:1;">{value}</div>
      <div style="font-size:0.72rem; color:#475569; margin-top:8px; text-transform:uppercase;
           letter-spacing:0.8px;">{label}</div>
    </div>"""


def _arch_card(title, subtitle, body, accent="#3B82F6", accent_light="#60A5FA"):
    return f"""
    <div style="background:#141827; border:1px solid #1E2A45; border-left:4px solid {accent};
         border-radius:0 10px 10px 0; padding:16px 18px; margin:10px 0;
         box-sizing:border-box;">
      <div style="font-size:1rem; font-weight:700; color:#F1F5F9; margin-bottom:4px;">{title}</div>
      <div style="font-size:0.82rem; color:{accent_light}; margin-bottom:6px;">{subtitle}</div>
      <div style="font-size:0.84rem; color:#94A3B8; line-height:1.6;">{body}</div>
    </div>"""


def _step(num, title, desc):
    return f"""
    <div style="display:flex; align-items:flex-start; margin:10px 0;">
      <div style="background:rgba(59,130,246,0.15); color:#60A5FA; border:1px solid rgba(59,130,246,0.3);
           border-radius:50%; width:28px; height:28px; display:flex; align-items:center;
           justify-content:center; font-size:0.8rem; font-weight:700; flex-shrink:0;
           margin-top:2px;">{num}</div>
      <div style="margin-left:12px;">
        <div style="font-weight:600; color:#F1F5F9;">{title}</div>
        <div style="font-size:0.83rem; color:#64748B; margin-top:2px;">{desc}</div>
      </div>
    </div>"""


def _callout_info(text):
    return f"""
    <div style="background:rgba(59,130,246,0.08); border-left:4px solid #3B82F6;
         border-radius:0 8px 8px 0; padding:12px 16px; color:#93C5FD; margin:12px 0;
         box-sizing:border-box;">
      {text}
    </div>"""


class ProjectSummaryPage:
    @staticmethod
    def render():
        # ── Page header ───────────────────────────────────────────────────────
        st.markdown("""
        <div style="margin-bottom:28px; padding-bottom:16px; border-bottom:1px solid #1E2A45;">
          <div style="display:inline-block; background:rgba(59,130,246,0.12); color:#60A5FA;
               padding:3px 10px; border-radius:20px; font-size:0.7rem; font-weight:700;
               letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;">OVERVIEW</div>
          <div style="font-size:1.6rem; font-weight:700; color:#F1F5F9; letter-spacing:-0.3px;">
            👶 Neonatal Sepsis Detection
          </div>
          <div style="font-size:0.88rem; color:#64748B; margin-top:6px; max-width:640px;">
            Federated Learning pipeline — privacy-first early warning from 48-hour clinical time-series.
            Patient data never leaves the hospital. Only model weights are shared across sites.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── What is this project? ─────────────────────────────────────────────
        st.markdown("### What is this project?")
        st.markdown("""
        This system detects **neonatal sepsis** early — before it becomes life-threatening — using
        machine learning on routine ICU measurements. Sepsis in newborns kills within hours if missed,
        but the early signs (subtle changes in heart rate, oxygen, labs) are buried in noisy clinical data.

        The core challenge: **hospitals can't share patient data** (privacy laws). So we use
        **Federated Learning** — each hospital trains locally, only the model improvements are shared.
        The result is a stronger model than any single hospital could build alone.
        """)

        # ── Stat cards ────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        accents = ["#3B82F6", "#06B6D4", "#8B5CF6", "#10B981"]
        for col, (val, label), accent in zip(
            [c1, c2, c3, c4],
            [("40,323", "Patient Files"), ("40", "Clinical Features"),
             ("48", "Timesteps / Window"), ("5", "FL Clients")],
            accents,
        ):
            col.markdown(_metric_card(val, label, accent), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Architecture | Pipeline ───────────────────────────────────────────
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
                    "between vitals and lab values across an entire ICU stay.",
                    "#3B82F6", "#60A5FA",
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
                    "decaying toward the population mean when data is missing.",
                    "#06B6D4", "#22D3EE",
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
                    "score actually have sepsis.",
                    "#8B5CF6", "#A78BFA",
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
                    "BatchNorm statistics local to each hospital's data distribution.",
                    "#10B981", "#6EE7B7",
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
                 "Flower server coordinates 5 clients. Each client holds a private partition "
                 "of train patients. Server aggregates updates for 10 rounds. Best global "
                 "model saved to server_out/global_best.pt."),
                ("5", "Evaluation on frozen test set",
                 "Loads the frozen test_index.pt (never seen during training). Computes "
                 "AUROC, AUPRC, F1, calibration error, and 95% bootstrap CIs. Outputs "
                 "eval_results_*.json per model."),
                ("6", "Comparison plots",
                 "ROC and Precision-Recall curves with confidence bands for both models. "
                 "Visual confirmation that federated model generalises better."),
                ("7", "Serve via FastAPI",
                 "FastAPI v2 endpoint at /v2/predict. MC Dropout for uncertainty intervals. "
                 "Prometheus /metrics for monitoring. Every prediction logged to audit JSONL."),
            ]
            for num, title, desc in steps:
                st.markdown(_step(num, title, desc), unsafe_allow_html=True)

        st.markdown(
            '<div style="border-top:1px solid #1E2A45; margin:24px 0;"></div>',
            unsafe_allow_html=True,
        )

        # ── Dataset: what the data looks like ─────────────────────────────────
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
                    f'<div style="background:#141827; border:1px solid #1E2A45; border-radius:8px;'
                    f'padding:10px 14px; margin:6px 0;">'
                    f'<span style="font-weight:600; color:#60A5FA; font-size:0.88rem;">{cat}</span><br>'
                    f'<span style="font-size:0.83rem; color:#64748B;">{feats}</span></div>',
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
                    f'<div style="background:#141827; border:1px solid #1E2A45; border-radius:8px;'
                    f'padding:10px 14px; margin:6px 0;">'
                    f'<span style="font-weight:600; color:#F1F5F9; font-size:0.88rem;">{title}:</span> '
                    f'<span style="font-size:0.84rem; color:#64748B;">{desc}</span></div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            '<div style="border-top:1px solid #1E2A45; margin:24px 0;"></div>',
            unsafe_allow_html=True,
        )

        # ── Key design decisions ───────────────────────────────────────────────
        st.markdown("### Key Design Decisions")
        d1, d2, d3 = st.columns(3)
        decisions = [
            ("#3B82F6",
             "<b>Frozen test set</b>",
             "Split is created once and locked before any training begins. "
             "No hyperparameter or model selection decision ever sees test data."),
            ("#10B981",
             "<b>Patient-level split</b>",
             "Each patient is entirely in one split (never split across train/val/test). "
             "Prevents time-series leakage where early hours train and late hours evaluate."),
            ("#F59E0B",
             "<b>FL test exclusion</b>",
             "Patients in the frozen test set are excluded from all FL client partitions. "
             "Federated clients only see train/val patients — zero leakage guaranteed."),
        ]
        for col, (accent, title, body) in zip([d1, d2, d3], decisions):
            col.markdown(
                f'<div style="background:#141827; border:1px solid #1E2A45; border-left:4px solid {accent};'
                f'border-radius:0 10px 10px 0; padding:16px 18px; height:100%;'
                f'box-sizing:border-box;">'
                f'<div style="font-size:0.95rem; font-weight:700; color:#F1F5F9; margin-bottom:6px;">{title}</div>'
                f'<div style="font-size:0.83rem; color:#64748B; line-height:1.6;">{body}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            _callout_info(
                "<b>Navigation guide:</b> Use <b>Predict</b> to run inference on a patient file. "
                "Use <b>Model Metrics</b> to compare Federated vs Local AUROC/AUPRC. "
                "Use <b>Clinical Metrics</b> to see bedside-relevant stats (sensitivity at specificity, "
                "alert fatigue, NNAlert). Use <b>Training Runs</b> to browse saved checkpoints."
            ),
            unsafe_allow_html=True,
        )
