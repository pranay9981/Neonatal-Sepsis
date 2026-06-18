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
    ("eval_results_federated.json",       "Federated GRU-D (FedAvg, IID)",    "#3B82F6"),
    ("eval_results_noniid.json",          "Federated GRU-D (FedAvg, non-IID)", "#F59E0B"),
    ("eval_results_fedbn.json",           "Federated GRU-D (FedBN)",          "#06B6D4"),
    ("eval_results_transformer_fl.json",  "Federated Transformer (FedAvg)",   "#10B981"),
    ("eval_results_grud.json",            "GRU-D (Local)",                    "#EF4444"),
    ("eval_results_transformer.json",     "Transformer (Local)",              "#8B5CF6"),
    ("eval_results_ensemble.json",        "Ensemble",                         "#EC4899"),
    # Early-warning task (different from patient-level above): 12h window → predict sepsis in next 6h
    ("eval_results_windowed_grud.json",   "GRU-D Windowed (Early Warning, 6h)", "#A78BFA"),
]

# Dark chart layout shared across plotly figures
_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#141827",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#94A3B8"),
    legend=dict(
        bgcolor="rgba(20,24,39,0.9)", bordercolor="#1E2A45", borderwidth=1,
        font=dict(color="#94A3B8"),
    ),
    xaxis=dict(gridcolor="#1E2A45", tickfont=dict(color="#64748B"), linecolor="#1E2A45"),
    yaxis=dict(gridcolor="#1E2A45", tickfont=dict(color="#64748B"), linecolor="#1E2A45"),
)


@st.cache_data
def load_eval(path: str):
    """Load evaluation JSON; cache key is the resolved absolute path (I-17)."""
    resolved = str(Path(path).resolve())
    p = Path(resolved)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def compute_metrics(data, threshold=0.5):
    y_true = data.get('y_true')
    y_prob = data.get('y_prob')
    if y_true is None or y_prob is None:
        st.error("Evaluation file is missing 'y_true' or 'y_prob' keys.")
        return {}
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > threshold).astype(int)
    stored_auroc = data.get("auroc", np.nan)
    stored_auprc = data.get("auprc", np.nan)
    if len(np.unique(y_true)) >= 2:
        from sklearn.metrics import roc_auc_score, average_precision_score
        recomputed_auroc = float(roc_auc_score(y_true, y_prob))
        recomputed_auprc = float(average_precision_score(y_true, y_prob))
        if not np.isnan(stored_auroc) and abs(recomputed_auroc - stored_auroc) > 0.01:
            import logging
            logging.getLogger(__name__).warning(
                "Stored AUROC %.4f differs from recomputed %.4f for %s",
                stored_auroc, recomputed_auroc, data.get("model_name", "?"),
            )
        auroc = recomputed_auroc
        auprc = recomputed_auprc
    else:
        auroc = stored_auroc
        auprc = stored_auprc
    return {
        "Model":     data.get("model_name", "Model").replace("_", " ").title(),
        "AUROC":     auroc,
        "AUPRC":     auprc,
        "Accuracy":  accuracy_score(y_true, y_pred),
        "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
    }


def _bar(df, metric, model_colors):
    fig = go.Figure()
    for _, row in df.iterrows():
        c = model_colors.get(row["Model"], "#475569")
        fig.add_trace(go.Bar(
            x=[row["Model"]], y=[row[metric]],
            name=row["Model"],
            marker_color=c,
            text=[f"{row[metric]:.3f}"],
            textposition="outside",
            textfont=dict(color="#94A3B8"),
        ))
    metric_values = [v for v in df[metric] if not (isinstance(v, float) and np.isnan(v))]
    upper = max(metric_values) * 1.18 if metric_values else 1.0
    upper = max(upper, 0.01)
    fig.update_layout(
        title=dict(text=metric, font=dict(size=15, color="#94A3B8")),
        yaxis=dict(range=[0, min(1.15, upper)],
                   gridcolor="#1E2A45", tickfont=dict(color="#64748B"), linecolor="#1E2A45"),
        xaxis=dict(gridcolor="#1E2A45", tickfont=dict(color="#64748B"), linecolor="#1E2A45"),
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=45, b=30),
        font=dict(family="Inter, Segoe UI, sans-serif", color="#94A3B8"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#141827",
    )
    return fig


def _confusion_ax(ax, y_true, y_prob, title, threshold=0.5, color="#3B82F6"):
    y_pred = (np.array(y_prob) > threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    labels = ["TN", "FP", "FN", "TP"]
    counts = cm.flatten()
    total  = counts.sum()
    annot  = np.asarray([
        f"{lab}\n{cnt}\n({cnt/total:.1%})"
        for lab, cnt in zip(labels, counts)
    ]).reshape(2, 2)
    cmap = sns.dark_palette(color, as_cmap=True)
    sns.heatmap(cm, annot=annot, fmt="", cmap=cmap, ax=ax, cbar=False,
                linewidths=0.5, linecolor="#0F1117")
    ax.set_title(title, fontsize=12, pad=10, color="#94A3B8")
    ax.set_xlabel("Predicted", fontsize=10, color="#64748B")
    ax.set_ylabel("Actual",    fontsize=10, color="#64748B")
    ax.set_xticklabels(["No Sepsis", "Sepsis"], fontsize=9, color="#64748B")
    ax.set_yticklabels(["No Sepsis", "Sepsis"], fontsize=9, rotation=0, color="#64748B")
    ax.tick_params(colors="#64748B")


class MetricsPage:
    @staticmethod
    def render():
        # ── Page header ───────────────────────────────────────────────────────
        st.markdown("""
        <div style="margin-bottom:28px; padding-bottom:16px; border-bottom:1px solid #1E2A45;">
          <div style="display:inline-block; background:rgba(139,92,246,0.12); color:#A78BFA;
               padding:3px 10px; border-radius:20px; font-size:0.7rem; font-weight:700;
               letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;">EVALUATION</div>
          <div style="font-size:1.6rem; font-weight:700; color:#F1F5F9; letter-spacing:-0.3px;">
            🧪 Model Performance
          </div>
          <div style="font-size:0.88rem; color:#64748B; margin-top:6px; max-width:640px;">
            Federated model vs Local baseline — head-to-head on the frozen held-out test set.
            All metrics computed on patients the model never saw during training or model selection.
          </div>
        </div>
        """, unsafe_allow_html=True)

        available = {}
        model_colors = {}
        for fname, display_name, color in KNOWN_EVAL_FILES:
            data = load_eval(str(ROOT / fname))
            if data is not None:
                available[display_name] = data
                model_colors[display_name] = color

        if not available:
            st.markdown("""
            <div style="background:rgba(245,158,11,0.08); border-left:4px solid #F59E0B;
                 border-radius:0 8px 8px 0; padding:16px 18px; color:#FCD34D;">
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

        # ── Winner callout (patient-level models only, exclude windowed) ──────
        windowed_marker = "windowed"
        pl_rows  = df_metrics[~df_metrics["Model"].str.lower().str.contains(windowed_marker)]
        fed_rows = pl_rows[pl_rows["Model"].str.lower().str.contains("fed")]
        loc_rows = pl_rows[~pl_rows["Model"].str.lower().str.contains("fed")]
        if not fed_rows.empty and not loc_rows.empty:
            best_fed_auroc = fed_rows["AUROC"].max()
            best_fed_name  = fed_rows.loc[fed_rows["AUROC"].idxmax(), "Model"]
            best_loc_auroc = loc_rows["AUROC"].max()
            best_loc_name  = loc_rows.loc[loc_rows["AUROC"].idxmax(), "Model"]
            if not (np.isnan(best_fed_auroc) or np.isnan(best_loc_auroc)):
                diff = best_fed_auroc - best_loc_auroc
                if diff > 0:
                    st.markdown(
                        f'<div style="background:rgba(16,185,129,0.08); border-left:4px solid #10B981;'
                        f'border-radius:0 8px 8px 0; padding:12px 18px; color:#6EE7B7;">'
                        f'<b>Best federated model wins</b> by <b>+{diff:.3f} AUROC</b> '
                        f'({best_fed_name}: {best_fed_auroc:.3f} vs {best_loc_name}: {best_loc_auroc:.3f}) — '
                        f'federated collaboration generalises better than a single-site model.</div>',
                        unsafe_allow_html=True,
                    )
                elif diff < 0:
                    st.markdown(
                        f'<div style="background:rgba(245,158,11,0.08); border-left:4px solid #F59E0B;'
                        f'border-radius:0 8px 8px 0; padding:12px 18px; color:#FCD34D;">'
                        f'{best_loc_name} leads by <b>{-diff:.3f} AUROC</b> '
                        f'({best_loc_auroc:.3f} vs best federated {best_fed_name}: {best_fed_auroc:.3f}). '
                        f'More FL rounds may close the gap.</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabs ───────────────────────────────────────────────────────────────
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

            # Summary table — split patient-level vs windowed
            st.markdown("#### Metrics at a glance")
            df_display = df_metrics.set_index("Model")[["AUROC","AUPRC","Accuracy","F1-Score","Precision","Recall"]]
            windowed_mask = df_display.index.str.lower().str.contains("windowed")
            df_patient  = df_display[~windowed_mask]
            df_windowed = df_display[windowed_mask]
            if not df_patient.empty:
                st.caption("Patient-level models (full ICU stay → sepsis risk)")
                st.dataframe(df_patient.style.format("{:.3f}", na_rep="-").background_gradient(
                    cmap="Blues", axis=0
                ), use_container_width=True)
            if not df_windowed.empty:
                st.caption("Early warning model (12h window → sepsis in next 6h) — different task, not directly comparable")
                st.dataframe(df_windowed.style.format("{:.3f}", na_rep="-"), use_container_width=True)

            # Best metric callouts (patient-level only)
            st.markdown("#### Highlights (patient-level models)")
            h1, h2, h3 = st.columns(3)
            _hl = df_patient if not df_patient.empty else df_display
            best_auroc = _hl["AUROC"].idxmax() if not _hl["AUROC"].isna().all() else "N/A"
            best_f1    = _hl["F1-Score"].idxmax() if not _hl["F1-Score"].isna().all() else "N/A"
            best_rec   = _hl["Recall"].idxmax() if not _hl["Recall"].isna().all() else "N/A"
            h1.metric("Best AUROC",   f"{_hl['AUROC'].max():.3f}",   best_auroc)
            h2.metric("Best F1",      f"{_hl['F1-Score'].max():.3f}", best_f1)
            h3.metric("Best Recall",  f"{_hl['Recall'].max():.3f}",   best_rec)

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
            if len(df_display) < 2:
                st.info("Radar chart requires at least 2 models to compare.")
            else:
                st.markdown(
                    '<span style="font-size:0.85rem; color:#64748B;">Each axis normalised to [0,1] within the comparison. Shows relative strengths.</span>',
                    unsafe_allow_html=True,
                )
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
                        line=dict(color=model_colors.get(idx, "#475569"), width=2),
                        marker=dict(size=6),
                    ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="#141827",
                        radialaxis=dict(visible=True, range=[0, 1],
                                        gridcolor="#1E2A45", tickfont=dict(color="#64748B"),
                                        linecolor="#1E2A45"),
                        angularaxis=dict(gridcolor="#1E2A45", tickfont=dict(color="#94A3B8"),
                                         linecolor="#1E2A45"),
                    ),
                    showlegend=True, height=480,
                    margin=dict(t=40, b=20),
                    font=dict(family="Inter, Segoe UI, sans-serif", color="#94A3B8"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(bgcolor="rgba(20,24,39,0.9)", bordercolor="#1E2A45", borderwidth=1,
                                font=dict(color="#94A3B8")),
                )
                st.plotly_chart(fig_radar, width='stretch')

        with tab2:
            st.markdown("#### ROC and Precision-Recall curves")
            st.markdown(
                '<span style="font-size:0.85rem; color:#64748B;">Interactive charts — hover for exact values. '
                'Shaded bands = 95% bootstrap confidence intervals (1000 resamples).</span>',
                unsafe_allow_html=True,
            )

            # ── Build interactive ROC ──────────────────────────────────────────
            fig_roc = go.Figure()
            fig_prc = go.Figure()
            fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                              line=dict(color="#475569", dash="dash", width=1.5))
            # rng is intentionally defined outside the loop to maintain reproducible sequence across bootstrap iterations.
            rng = np.random.RandomState(42)

            for i, (name, data) in enumerate(available.items()):
                color = model_colors.get(name, "#475569")
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
                        opacity=0.12, line=dict(width=0),
                        showlegend=False, hoverinfo="skip",
                    ))
                    # CI band — PRC
                    fig_prc.add_trace(go.Scatter(
                        x=np.concatenate([base_rec, base_rec[::-1]]),
                        y=np.concatenate([prec_hi, prec_lo[::-1]]),
                        fill="toself", fillcolor=color,
                        opacity=0.12, line=dict(width=0),
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
                                  line=dict(color="#475569", dash="dash", width=1.5))
                fig_prc.add_annotation(x=0.98, y=baseline + 0.02,
                                       text=f"Chance ({baseline:.2f})",
                                       showarrow=False,
                                       font=dict(color="#475569", size=11))

            _layout = dict(
                height=420, margin=dict(t=40, b=50, l=60, r=20),
                legend=dict(x=0.01, y=0.01,
                            bgcolor="rgba(20,24,39,0.9)", bordercolor="#1E2A45", borderwidth=1,
                            font=dict(color="#94A3B8")),
                font=dict(family="Inter, Segoe UI, sans-serif", color="#94A3B8"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#141827",
            )
            fig_roc.update_layout(
                title=dict(text="ROC Curve", font=dict(color="#94A3B8")),
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                xaxis=dict(range=[0, 1], gridcolor="#1E2A45", tickfont=dict(color="#64748B"),
                           linecolor="#1E2A45"),
                yaxis=dict(range=[0, 1.02], gridcolor="#1E2A45", tickfont=dict(color="#64748B"),
                           linecolor="#1E2A45"),
                **_layout,
            )
            fig_prc.update_layout(
                title=dict(text="Precision-Recall Curve", font=dict(color="#94A3B8")),
                xaxis_title="Recall",
                yaxis_title="Precision",
                xaxis=dict(range=[0, 1], gridcolor="#1E2A45", tickfont=dict(color="#64748B"),
                           linecolor="#1E2A45"),
                yaxis=dict(range=[0, 1.02], gridcolor="#1E2A45", tickfont=dict(color="#64748B"),
                           linecolor="#1E2A45"),
                **_layout,
            )

            col1, col2 = st.columns(2)
            col1.plotly_chart(fig_roc, width='stretch')
            col2.plotly_chart(fig_prc, width='stretch')

            st.markdown("""
            <div style="background:rgba(59,130,246,0.08); border-left:4px solid #3B82F6;
                 border-radius:0 8px 8px 0; padding:12px 18px; color:#93C5FD;
                 margin-top:4px; box-sizing:border-box; width:100%;">
            <b>How to read:</b>
            AUROC near 1.0 = strong discrimination. AUPRC is more important here — with ~8% positive
            rate, a random classifier scores ~0.08 AUPRC. Good sepsis model: &gt;0.40 AUPRC.
            Shaded bands show 95% CI. Non-overlapping bands = statistically significant difference.
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.markdown("#### Confusion matrices")
            st.markdown(
                f'<span style="font-size:0.85rem; color:#64748B;">At threshold = {threshold:.2f}. '
                f'Shows TN/FP/FN/TP counts and percentage of total test set.</span>',
                unsafe_allow_html=True,
            )

            st.markdown("""
            <div style="background:#141827; border:1px solid #1E2A45; border-radius:8px;
                 padding:12px 18px; margin-bottom:12px; color:#94A3B8;">
            <b style="color:#F1F5F9;">Reading the matrix:</b>
            <b>TN</b> = correctly cleared (no alert, right) &nbsp;|&nbsp;
            <b>FP</b> = false alarm (alert on healthy patient) &nbsp;|&nbsp;
            <b>FN</b> = missed sepsis (most dangerous) &nbsp;|&nbsp;
            <b>TP</b> = caught sepsis (the goal)
            </div>
            """, unsafe_allow_html=True)

            plt.rcParams.update({
                'figure.facecolor': '#141827',
                'axes.facecolor': '#0F1117',
                'axes.edgecolor': '#1E2A45',
                'axes.labelcolor': '#94A3B8',
                'text.color': '#94A3B8',
                'xtick.color': '#64748B',
                'ytick.color': '#64748B',
                'grid.color': '#1E2A45',
            })

            # 2-column wrapping grid — one figure per model
            model_items = list(available.items())
            for row_start in range(0, len(model_items), 2):
                pair = model_items[row_start: row_start + 2]
                cols = st.columns(len(pair))
                for col, (name, data) in zip(cols, pair):
                    with col:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        fig.patch.set_facecolor('#141827')
                        ax.set_facecolor('#0F1117')
                        color = model_colors.get(name, "#3B82F6")
                        _confusion_ax(ax, data["y_true"], data["y_prob"],
                                      title=name, threshold=threshold, color=color)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

            def _summary(d):
                return {k: v for k, v in d.items() if k not in ("y_true", "y_prob")}
            with st.expander("Raw evaluation metadata (for auditors)"):
                for name, data in available.items():
                    st.markdown(f"**{name}**")
                    st.json(_summary(data))
