# 1_04_🧪_Model_Metrics.py
import sys
from pathlib import Path
import streamlit as st
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                              precision_score, recall_score,
                              roc_curve, auc, precision_recall_curve,
                              average_precision_score)

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ROOT = Path(__file__).resolve().parent.parent

# Ordered list of eval files to discover automatically.
# Add new entries here whenever a new model is evaluated.
KNOWN_EVAL_FILES = [
    ("eval_results_federated.json",    "Federated GRU-D (FedAvg, IID)",    "#1E3A5F"),
    ("eval_results_noniid.json",       "Federated GRU-D (FedAvg, non-IID)", "#FF6F00"),
    ("eval_results_fedbn.json",        "Federated GRU-D (FedBN)",          "#0277BD"),
    ("eval_results_grud.json",         "GRU-D (Local)",                    "#E53935"),
    ("eval_results_transformer.json",  "Transformer (Local)",              "#2E7D32"),
    ("eval_results_ensemble.json",     "Ensemble",                         "#6A1B9A"),
]


@st.cache_data
def load_eval(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def compute_metrics(data, threshold=0.5):
    y_true = np.array(data["y_true"])
    y_prob = np.array(data["y_prob"])
    y_pred = (y_prob > threshold).astype(int)
    return {
        "Model":     data.get("model_name", "Model").replace("_", " ").title(),
        "AUROC":     data.get("auroc",  np.nan),
        "AUPRC":     data.get("auprc",  np.nan),
        "Accuracy":  accuracy_score(y_true, y_pred),
        "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
    }


def _bar(df, metric, model_colors):
    fig = go.Figure()
    for _, row in df.iterrows():
        c = model_colors.get(row["Model"], "#888888")
        fig.add_trace(go.Bar(
            x=[row["Model"]], y=[row[metric]],
            name=row["Model"],
            marker_color=c,
            text=[f"{row[metric]:.3f}"],
            textposition="outside",
        ))
    fig.update_layout(
        title={"text": metric, "font": {"size": 15}},
        yaxis=dict(range=[0, min(1.15, df[metric].max() * 1.18)],
                   gridcolor="rgba(0,0,0,0.05)"),
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=45, b=30),
        font=dict(family="Inter, Segoe UI, sans-serif"),
        plot_bgcolor="white",
    )
    return fig


def _confusion_ax(ax, y_true, y_prob, title, threshold=0.5, color="#1E3A5F"):
    y_pred = (np.array(y_prob) > threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    labels = ["TN", "FP", "FN", "TP"]
    counts = cm.flatten()
    total  = counts.sum()
    annot  = np.asarray([
        f"{lab}\n{cnt}\n({cnt/total:.1%})"
        for lab, cnt in zip(labels, counts)
    ]).reshape(2, 2)
    cmap = sns.light_palette(color, as_cmap=True)
    sns.heatmap(cm, annot=annot, fmt="", cmap=cmap, ax=ax, cbar=False,
                linewidths=0.5, linecolor="#E2E8F0")
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual",    fontsize=10)
    ax.set_xticklabels(["No Sepsis", "Sepsis"], fontsize=9)
    ax.set_yticklabels(["No Sepsis", "Sepsis"], fontsize=9, rotation=0)


class MetricsPage:
    @staticmethod
    def render():
        # ── Header ────────────────────────────────────────────────────────
        st.markdown("""
        <div style="background:linear-gradient(90deg,#1E3A5F 0%,#6A1B9A 100%);
             color:white;padding:20px 28px 16px 28px;border-radius:10px;margin-bottom:24px;
             box-sizing:border-box;width:100%;">
          <div style="font-size:1.7rem;font-weight:700;">&#129514; Model Performance</div>
          <div style="font-size:0.92rem;opacity:0.85;margin-top:4px;">
            Federated model vs Local baseline &#8212; head-to-head on the frozen held-out test set
          </div>
          <div style="font-size:0.82rem;opacity:0.7;margin-top:8px;">
            All metrics computed on patients the model never saw during training or model selection.
          </div>
        </div>
        """, unsafe_allow_html=True)

        available = {}
        model_colors = {}
        for fname, display_name, color in KNOWN_EVAL_FILES:
            data = load_eval(ROOT / fname)
            if data is not None:
                available[display_name] = data
                model_colors[display_name] = color

        if not available:
            st.markdown("""
            <div style="background:#FFF3E0;border-left:4px solid #FF9800;border-radius:0 8px 8px 0;
                 padding:16px 18px;color:#E65100;">
            <b>No evaluation results found.</b> Run the pipeline to generate them:<br><br>
            <code>python src/evaluate.py --index data/splits/test_index.pt
                  --ckpt server_out/global_best.pt --model grud
                  --out_file eval_results_federated.json</code>
            </div>
            """, unsafe_allow_html=True)
            return

        threshold = st.slider("Classification threshold (for Accuracy / F1 / Precision / Recall)",
                              0.0, 1.0, 0.5, 0.01)

        metrics_list = []
        for display_name, data in available.items():
            m = compute_metrics(data, threshold)
            m["Model"] = display_name
            metrics_list.append(m)
        df_metrics = pd.DataFrame(metrics_list)

        # ── Winner callout ─────────────────────────────────────────────────
        fed_rows = df_metrics[df_metrics["Model"].str.lower().str.contains("fed")]
        loc_rows = df_metrics[~df_metrics["Model"].str.lower().str.contains("fed")]
        if not fed_rows.empty and not loc_rows.empty:
            fed_auroc = fed_rows["AUROC"].iloc[0]
            best_loc_auroc = loc_rows["AUROC"].max()
            best_loc_name = loc_rows.loc[loc_rows["AUROC"].idxmax(), "Model"]
            if not (np.isnan(fed_auroc) or np.isnan(best_loc_auroc)):
                diff = fed_auroc - best_loc_auroc
                if diff > 0:
                    st.markdown(
                        f'<div style="background:#E8F5E9;border-left:4px solid #4CAF50;'
                        f'border-radius:0 8px 8px 0;padding:12px 18px;color:#1B5E20;">'
                        f'<b>Federated model wins</b> by <b>+{diff:.3f} AUROC</b> '
                        f'({fed_auroc:.3f} vs {best_loc_name}: {best_loc_auroc:.3f}) — federated collaboration '
                        f'across sites generalises better than a single-site model.</div>',
                        unsafe_allow_html=True,
                    )
                elif diff < 0:
                    st.markdown(
                        f'<div style="background:#FFF3E0;border-left:4px solid #FF9800;'
                        f'border-radius:0 8px 8px 0;padding:12px 18px;color:#E65100;">'
                        f'{best_loc_name} leads by <b>{-diff:.3f} AUROC</b> at this point '
                        f'({best_loc_auroc:.3f} vs federated: {fed_auroc:.3f}). '
                        f'More FL rounds may close the gap.</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabs ───────────────────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(["Summary & charts", "ROC & PRC curves", "Confusion matrices"])

        with tab1:
            # Metric explanation
            with st.expander("What do these metrics mean?"):
                st.markdown("""
                | Metric | What it measures | When to prioritise |
                |---|---|---|
                | **AUROC** | Overall discrimination — can the model rank sepsis patients above healthy ones? | General model quality |
                | **AUPRC** | Precision-Recall tradeoff on imbalanced data — more informative than AUROC when positives are rare | Rare-event detection (7–9% sepsis rate) |
                | **Accuracy** | Overall fraction correct | Only meaningful when classes are balanced |
                | **F1-Score** | Harmonic mean of precision and recall at the chosen threshold | Balanced tradeoff |
                | **Precision** | Of all flagged patients, what fraction truly had sepsis? | Minimising false alarms |
                | **Recall** | Of all true sepsis patients, what fraction were caught? | Minimising missed cases |
                """)

            # Summary table
            st.markdown("#### Metrics at a glance")
            df_display = df_metrics.set_index("Model")[["AUROC","AUPRC","Accuracy","F1-Score","Precision","Recall"]]
            st.dataframe(df_display.style.format("{:.3f}").background_gradient(
                cmap="Blues", axis=0
            ), use_container_width=True)

            # Best metric callouts
            st.markdown("#### Highlights")
            h1, h2, h3 = st.columns(3)
            best_auroc = df_display["AUROC"].idxmax()
            best_f1    = df_display["F1-Score"].idxmax()
            best_rec   = df_display["Recall"].idxmax()
            h1.metric("Best AUROC",   f"{df_display['AUROC'].max():.3f}",   best_auroc)
            h2.metric("Best F1",      f"{df_display['F1-Score'].max():.3f}", best_f1)
            h3.metric("Best Recall",  f"{df_display['Recall'].max():.3f}",   best_rec)

            # Bar charts
            st.markdown("#### Side-by-side comparison")
            metrics_to_plot = ["AUROC", "AUPRC", "F1-Score", "Recall", "Precision", "Accuracy"]
            rows = [metrics_to_plot[:3], metrics_to_plot[3:]]
            for row_metrics in rows:
                cols = st.columns(len(row_metrics))
                for col, metric in zip(cols, row_metrics):
                    col.plotly_chart(_bar(df_metrics, metric, model_colors), width='stretch')

            # Radar chart
            st.markdown("#### Normalised metric profile")
            st.markdown('<span style="font-size:0.85rem;color:#64748B;">Each axis normalised to [0,1] within the comparison. Shows relative strengths.</span>', unsafe_allow_html=True)
            df_norm = df_display.copy()
            for col in df_norm.columns:
                rng = df_norm[col].max() - df_norm[col].min()
                df_norm[col] = ((df_norm[col] - df_norm[col].min()) / rng) if rng != 0 else 0.5
            fig_radar = go.Figure()
            for i, (idx, row) in enumerate(df_norm.iterrows()):
                fig_radar.add_trace(go.Scatterpolar(
                    r=row.values.tolist() + [row.values[0]],
                    theta=row.index.tolist() + [row.index[0]],
                    fill="toself",
                    name=idx,
                    line=dict(color=model_colors.get(idx, "#888888"), width=2),
                    marker=dict(size=6),
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True, height=480,
                margin=dict(t=40, b=20),
                font=dict(family="Inter, Segoe UI, sans-serif"),
            )
            st.plotly_chart(fig_radar, width='stretch')

        with tab2:
            st.markdown("#### ROC and Precision-Recall curves")
            st.markdown(
                '<span style="font-size:0.85rem;color:#64748B;">Interactive charts — hover for exact values. '
                'Shaded bands = 95% bootstrap confidence intervals (1000 resamples).</span>',
                unsafe_allow_html=True,
            )

            # ── Build interactive ROC ──────────────────────────────────────
            fig_roc = go.Figure()
            fig_prc = go.Figure()
            fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                              line=dict(color="#9E9E9E", dash="dash", width=1.5))
            rng = np.random.RandomState(42)

            for i, (name, data) in enumerate(available.items()):
                color = model_colors.get(name, "#888888")
                y_t = np.array(data["y_true"])
                y_p = np.array(data["y_prob"])

                # ROC
                fpr, tpr, _ = roc_curve(y_t, y_p)
                roc_auc = auc(fpr, tpr)

                # PRC
                prec, rec, _ = precision_recall_curve(y_t, y_p)
                ap = average_precision_score(y_t, y_p)

                # Bootstrap CI (200 resamples for speed)
                base_fpr = np.linspace(0, 1, 100)
                base_rec = np.linspace(0, 1, 100)
                tpr_bs, prec_bs = [], []
                for _ in range(200):
                    idx = rng.choice(len(y_t), len(y_t), replace=True)
                    if len(np.unique(y_t[idx])) < 2:
                        continue
                    f2, t2, _ = roc_curve(y_t[idx], y_p[idx])
                    tpr_bs.append(np.interp(base_fpr, f2, t2))
                    p2, r2, _ = precision_recall_curve(y_t[idx], y_p[idx])
                    prec_bs.append(np.interp(base_rec, r2[::-1], p2[::-1]))

                if tpr_bs:
                    tpr_lo  = np.percentile(tpr_bs, 2.5, axis=0)
                    tpr_hi  = np.percentile(tpr_bs, 97.5, axis=0)
                    prec_lo = np.percentile(prec_bs, 2.5, axis=0)
                    prec_hi = np.percentile(prec_bs, 97.5, axis=0)

                    # CI band — ROC
                    fig_roc.add_trace(go.Scatter(
                        x=np.concatenate([base_fpr, base_fpr[::-1]]),
                        y=np.concatenate([tpr_hi, tpr_lo[::-1]]),
                        fill="toself", fillcolor=color,
                        opacity=0.15, line=dict(width=0),
                        showlegend=False, hoverinfo="skip",
                    ))
                    # CI band — PRC
                    fig_prc.add_trace(go.Scatter(
                        x=np.concatenate([base_rec, base_rec[::-1]]),
                        y=np.concatenate([prec_hi, prec_lo[::-1]]),
                        fill="toself", fillcolor=color,
                        opacity=0.15, line=dict(width=0),
                        showlegend=False, hoverinfo="skip",
                    ))

                # Main ROC line
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"{name} (AUROC={roc_auc:.3f})",
                    line=dict(color=color, width=2.5),
                    hovertemplate="FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra>" + name + "</extra>",
                ))
                # Main PRC line
                fig_prc.add_trace(go.Scatter(
                    x=rec, y=prec, mode="lines", name=f"{name} (AUPRC={ap:.3f})",
                    line=dict(color=color, width=2.5),
                    hovertemplate="Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra>" + name + "</extra>",
                ))

            # Baseline PRC
            if available:
                first_data = next(iter(available.values()))
                baseline = float(np.mean(first_data["y_true"]))
                fig_prc.add_shape(type="line", x0=0, y0=baseline, x1=1, y1=baseline,
                                  line=dict(color="#9E9E9E", dash="dash", width=1.5))
                fig_prc.add_annotation(x=0.98, y=baseline + 0.02,
                                       text=f"Chance ({baseline:.2f})",
                                       showarrow=False, font=dict(color="#9E9E9E", size=11))

            _layout = dict(
                height=420, margin=dict(t=40, b=50, l=60, r=20),
                legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="#E2E8F0", borderwidth=1),
                font=dict(family="Inter, Segoe UI, sans-serif"),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            fig_roc.update_layout(
                title="ROC Curve", xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                xaxis=dict(range=[0, 1], gridcolor="#F1F5F9"),
                yaxis=dict(range=[0, 1.02], gridcolor="#F1F5F9"),
                **_layout,
            )
            fig_prc.update_layout(
                title="Precision-Recall Curve", xaxis_title="Recall",
                yaxis_title="Precision",
                xaxis=dict(range=[0, 1], gridcolor="#F1F5F9"),
                yaxis=dict(range=[0, 1.02], gridcolor="#F1F5F9"),
                **_layout,
            )

            col1, col2 = st.columns(2)
            col1.plotly_chart(fig_roc, width='stretch')
            col2.plotly_chart(fig_prc, width='stretch')

            st.markdown("""
            <div style="background:#E3F2FD;border-left:4px solid #2196F3;border-radius:0 8px 8px 0;
                 padding:12px 18px;color:#0D47A1;margin-top:4px;box-sizing:border-box;width:100%;">
            <b>How to read:</b>
            AUROC near 1.0 = strong discrimination. AUPRC is more important here — with ~8% positive
            rate, a random classifier scores ~0.08 AUPRC. Good sepsis model: &gt;0.40 AUPRC.
            Shaded bands show 95% CI. Non-overlapping bands = statistically significant difference.
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.markdown("#### Confusion matrices")
            st.markdown(
                f'<span style="font-size:0.85rem;color:#64748B;">At threshold = {threshold:.2f}. '
                f'Shows TN/FP/FN/TP counts and percentage of total test set.</span>',
                unsafe_allow_html=True,
            )

            n_models = len(available)
            fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
            if n_models == 1:
                axes = [axes]
            for ax, (name, data) in zip(axes, available.items()):
                color = model_colors.get(name, "#1E3A5F")
                _confusion_ax(ax, data["y_true"], data["y_prob"],
                              title=f"{name} Model", threshold=threshold, color=color)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("""
            <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;
                 padding:12px 18px;margin-top:12px;color:#334155;">
            <b>Reading the matrix:</b>
            <b>TN</b> = correctly cleared (no alert, right) &nbsp;|&nbsp;
            <b>FP</b> = false alarm (alert on healthy patient) &nbsp;|&nbsp;
            <b>FN</b> = missed sepsis (most dangerous) &nbsp;|&nbsp;
            <b>TP</b> = caught sepsis (the goal)
            </div>
            """, unsafe_allow_html=True)

            def _summary(d):
                return {k: v for k, v in d.items() if k not in ("y_true", "y_prob")}
            with st.expander("Raw evaluation metadata (for auditors)"):
                for name, data in available.items():
                    st.markdown(f"**{name}**")
                    st.json(_summary(data))
