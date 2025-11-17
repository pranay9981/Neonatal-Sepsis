# 1_04_🧪_Model_Metrics.py
import streamlit as st
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

EVAL_FEDERATED_JSON = Path("eval_results_federated.json")
EVAL_LOCAL_JSON = Path("eval_results_local.json")
PLOT_ROC_PATH = Path("model_comparison_plot.png")
PLOT_PRC_PATH = PLOT_ROC_PATH.with_name(PLOT_ROC_PATH.stem + "_prc" + PLOT_ROC_PATH.suffix)

# Professional palette: blue for federated, coral for local (fallback if names differ)
PALETTE = ["#0052CC", "#FF6F61"]  # deep blue, coral
FONT_FAMILY = "Arial"

@st.cache_data
def load_eval_results(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def calculate_all_metrics(eval_data, threshold=0.5):
    y_true = np.array(eval_data['y_true'])
    y_prob = np.array(eval_data['y_prob'])
    y_pred = (y_prob > threshold).astype(int)
    return {
        "Model": eval_data.get('model_name', 'Model').replace('_', ' ').title(),
        "AUROC": eval_data.get('auroc', np.nan),
        "AUPRC": eval_data.get('auprc', np.nan),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0)
    }

def plot_confusion_ax(ax, y_true, y_prob, title, threshold=0.5):
    y_pred = (np.array(y_prob) > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    labels = ['TN', 'FP', 'FN', 'TP']
    counts = cm.flatten()
    pct = [f"{v:.2%}" for v in counts / np.sum(counts)]
    annot = np.asarray([f"{lab}\n{cnt}\n{p}" for lab, cnt, p in zip(labels, counts, pct)]).reshape(2, 2)
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

def build_color_map(model_names):
    """
    Build a color mapping for the model names using PALETTE.
    If there are more than two models, palette will cycle.
    """
    colors = {}
    for i, name in enumerate(model_names):
        colors[name] = PALETTE[i % len(PALETTE)]
    return colors

def styled_bar_chart(df, metric, color_map):
    """
    Returns a Plotly bar chart styled professionally for the given metric.
    df: dataframe with columns ['Model', metric]
    color_map: dict {model_name: color}
    """
    # Assign color by model
    df_plot = df.copy()
    df_plot['color'] = df_plot['Model'].map(color_map)

    fig = go.Figure()
    for _, row in df_plot.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Model']],
            y=[row[metric]],
            name=row['Model'],
            marker_color=row['color'],
            text=[f"{row[metric]:.3f}"],
            textposition='outside',
            hovertemplate=f"<b>{row['Model']}</b><br>{metric}: {{y:.4f}}<extra></extra>"
        ))

    fig.update_layout(
        title={'text': f"{metric} Comparison", 'x':0.01, 'xanchor':'left', 'font': {'size':18, 'family': FONT_FAMILY}},
        xaxis=dict(title='', tickfont=dict(size=12)),
        yaxis=dict(title=metric, tickfont=dict(size=12), gridcolor='rgba(0,0,0,0.05)'),
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
        font=dict(family=FONT_FAMILY)
    )
    # Ensure y-axis starts at 0 for interpretability for metrics like Accuracy/F1/Recall
    if metric in ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUPRC', 'AUROC']:
        fig.update_yaxes(range=[0, max(1.0, df[metric].max() * 1.05)])
    return fig

class MetricsPage:
    @staticmethod
    def render():
        st.title("🧪 Model Performance (Federated vs Local)")
        st.markdown("Compare the Federated model against a Local model on the hold-out test set. Charts are styled for professional presentation.")

        fed = load_eval_results(EVAL_FEDERATED_JSON)
        loc = load_eval_results(EVAL_LOCAL_JSON)

        if fed is None or loc is None:
            st.error("Evaluation JSON files not found. Expected files: 'eval_results_federated.json' and 'eval_results_local.json'.")
            st.info("Run evaluation scripts to create these files (e.g., src/evaluate.py).")
            return

        metrics_fed = calculate_all_metrics(fed)
        metrics_loc = calculate_all_metrics(loc)
        df_metrics = pd.DataFrame([metrics_fed, metrics_loc])

        # Normalize display names, order
        df_metrics_plot = df_metrics.set_index('Model')
        df_metrics_plot = df_metrics_plot[['AUROC','AUPRC','Accuracy','F1-Score','Precision','Recall']]

        # Build color mapping using model names
        model_names = df_metrics['Model'].tolist()
        color_map = build_color_map(model_names)

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Metric Summary", "📈 ROC & PRC", "🔢 Confusion Matrices"])

        with tab1:
            st.subheader("Metric Summary Table")
            st.dataframe(df_metrics_plot.style.format("{:.3f}"), use_container_width=True)

            st.markdown("### Key Metric Highlights")
            c1, c2, c3 = st.columns(3)
            best_auroc = df_metrics_plot['AUROC'].idxmax()
            best_f1 = df_metrics_plot['F1-Score'].idxmax()
            best_recall = df_metrics_plot['Recall'].idxmax()
            c1.metric("Best AUROC", f"{df_metrics_plot['AUROC'].max():.3f}", best_auroc)
            c2.metric("Best F1-Score", f"{df_metrics_plot['F1-Score'].max():.3f}", best_f1)
            c3.metric("Best Recall", f"{df_metrics_plot['Recall'].max():.3f}", best_recall)

            st.markdown("### Performance Bar Charts")
            # Create a two-column layout with stacked charts on each side
            left_col, right_col = st.columns(2)
            with left_col:
                st.plotly_chart(styled_bar_chart(df_metrics, "AUROC", color_map), use_container_width=True)
                st.plotly_chart(styled_bar_chart(df_metrics, "F1-Score", color_map), use_container_width=True)
                st.plotly_chart(styled_bar_chart(df_metrics, "Recall", color_map), use_container_width=True)
            with right_col:
                st.plotly_chart(styled_bar_chart(df_metrics, "AUPRC", color_map), use_container_width=True)
                st.plotly_chart(styled_bar_chart(df_metrics, "Precision", color_map), use_container_width=True)
                st.plotly_chart(styled_bar_chart(df_metrics, "Accuracy", color_map), use_container_width=True)

            st.markdown("### Normalized Metric Profile (Radar)")
            # Prepare normalized dataframe for radar
            df_norm = df_metrics_plot.copy()
            for col in df_norm.columns:
                rng = df_norm[col].max() - df_norm[col].min()
                if rng != 0:
                    df_norm[col] = (df_norm[col] - df_norm[col].min()) / rng
                else:
                    df_norm[col] = 0.5
            fig_radar = go.Figure()
            for idx, row in df_norm.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=row.values.tolist() + [row.values[0]],
                    theta=row.index.tolist() + [row.index[0]],
                    fill='toself',
                    name=idx,
                    line=dict(color=color_map.get(idx, "#888888"), width=2),
                    marker=dict(size=6)
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10))),
                showlegend=True,
                height=520,
                margin=dict(t=50, b=20, l=20, r=20),
                font=dict(family=FONT_FAMILY)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with tab2:
            st.subheader("ROC & Precision-Recall (PRC)")
            col1, col2 = st.columns(2)
            with col1:
                if PLOT_ROC_PATH.exists():
                    st.image(str(PLOT_ROC_PATH), caption="ROC Curve", use_container_width=True)
                else:
                    st.warning(f"ROC plot not found at '{PLOT_ROC_PATH}'.")
            with col2:
                if PLOT_PRC_PATH.exists():
                    st.image(str(PLOT_PRC_PATH), caption="PRC Curve", use_container_width=True)
                else:
                    st.warning(f"PRC plot not found at '{PLOT_PRC_PATH}'.")

            st.markdown("""
            **Interpretation tips (for reviewers):**
            - **AUROC** shows discrimination ability (1.0 is perfect).  
            - **AUPRC** is more informative on imbalanced data — higher is better.  
            - Compare where the Federated model curve lies relative to the Local model curve; separation with non-overlapping CI bands suggests statistical significance.
            """)

        with tab3:
            st.subheader("Confusion Matrices (threshold = 0.5)")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            plot_confusion_ax(ax1, fed['y_true'], fed['y_prob'], title=f"Federated ({fed.get('model_name','Federated')})")
            plot_confusion_ax(ax2, loc['y_true'], loc['y_prob'], title=f"Local ({loc.get('model_name','Local')})")
            plt.tight_layout()
            st.pyplot(fig)

            with st.expander("Raw Evaluation JSONs (for auditing)"):
                st.subheader("Federated Eval JSON")
                st.json(fed)
                st.subheader("Local Eval JSON")
                st.json(loc)
