# 1_00_📘_Project_Summary.py
import streamlit as st


class ProjectSummaryPage:
    @staticmethod
    def render():
        # Hero
        st.markdown("""
        <div class="main-header">
          <h1>👶 Neonatal Sepsis Detection</h1>
          <p>Federated Learning pipeline — privacy-first early warning from clinical time-series</p>
        </div>
        """, unsafe_allow_html=True)

        # Stat cards
        c1, c2, c3, c4 = st.columns(4)
        cards = [
            ("40,323", "PSV Patient Files"),
            ("40", "Clinical Features"),
            ("48", "Timesteps per Window"),
            ("3", "FL Clients"),
        ]
        for col, (val, label) in zip([c1, c2, c3, c4], cards):
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Two-column: architecture | pipeline
        left, right = st.columns(2)
        with left:
            st.markdown("### Architecture")
            st.markdown("""
            <div class="section-card">
            <b>TimeSeriesTransformer</b><br>
            CLS token · sinusoidal pos-emb · pad mask · focal loss
            </div>
            <div class="section-card">
            <b>GRU-D</b><br>
            Per-feature decay · empirical mean imputation · hidden decay
            </div>
            <div class="section-card">
            <b>Ensemble + Calibration</b><br>
            Blended probabilities · Temperature Scaling (LBFGS)
            </div>
            <div class="section-card">
            <b>Federated Learning</b><br>
            Flower (flwr) · FedAvg / FedProx (μ=0.01) / FedBN
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown("### Pipeline Steps")
            steps = [
                ("1", "Preprocess", "PSV → per-patient .pt tensors (X, mask, deltas)"),
                ("2", "Freeze splits", "70/15/15 stratified patient-level split"),
                ("3", "Local baseline", "Train Transformer or GRU-D on train split"),
                ("4", "FL simulation", "Flower server + N clients — FedAvg/FedBN"),
                ("5", "Evaluate", "AUROC, AUPRC, bootstrap CIs on frozen test set"),
                ("6", "Plots", "ROC / PRC comparison with confidence bands"),
                ("7", "Serve", "FastAPI v2 · MC Dropout CIs · Prometheus /metrics"),
            ]
            for num, title, desc in steps:
                st.markdown(f"""
                <div style="display:flex;align-items:flex-start;margin:8px 0;">
                  <div style="background:#1E3A5F;color:white;border-radius:50%;
                       width:26px;height:26px;display:flex;align-items:center;
                       justify-content:center;font-size:0.8rem;font-weight:700;
                       flex-shrink:0;margin-top:2px;">{num}</div>
                  <div style="margin-left:10px;">
                    <b>{title}</b><br>
                    <span style="color:#64748B;font-size:0.85rem;">{desc}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Feature categories
        st.markdown("### 40 Clinical Features")
        categories = {
            "Vital Signs": ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"],
            "Blood Gas / Electrolytes": ["pH", "PaCO2", "HCO3", "BaseExcess", "SaO2",
                                         "Chloride", "Potassium", "Calcium", "Magnesium", "Phosphate"],
            "Metabolic / Organ": ["Glucose", "Lactate", "BUN", "Creatinine", "AST",
                                  "Alkalinephos", "Bilirubin_direct", "Bilirubin_total", "TroponinI", "FiO2"],
            "Hematology": ["Hct", "Hgb", "WBC", "Platelets", "Fibrinogen", "PTT"],
            "Metadata": ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"],
        }
        col_a, col_b = st.columns(2)
        items = list(categories.items())
        for i, (cat, feats) in enumerate(items):
            target = col_a if i % 2 == 0 else col_b
            with target:
                st.markdown(f"**{cat}**")
                chips = "".join(f"<span class='chip'>{f}</span>" for f in feats)
                st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
                st.markdown("")

        st.markdown("---")

        # Key design decisions
        st.markdown("### Key Design Decisions")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown("""
            <div class="callout-info">
            <b>Frozen test set</b><br>
            Test split created once and never touched during training or model selection.
            </div>
            """, unsafe_allow_html=True)
        with d2:
            st.markdown("""
            <div class="callout-info">
            <b>Focal loss</b><br>
            Handles ~7–9% positive rate without resampling; AUPRC is primary metric.
            </div>
            """, unsafe_allow_html=True)
        with d3:
            st.markdown("""
            <div class="callout-info">
            <b>FL test exclusion</b><br>
            Test-split patients are excluded from FL client partitions — no data leakage.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="callout-success" style="margin-top:16px;">
        Navigate to <b>Predict</b> to run live inference, or <b>Model Metrics</b> to audit
        Federated vs Local performance on the frozen test set.
        </div>
        """, unsafe_allow_html=True)
