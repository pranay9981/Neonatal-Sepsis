# 1_05_📂_Training_Runs.py
"""
Training Run Browser — explore all local training runs stored in the runs/ directory.
Shows run metadata, best metrics, and training curves; helps you pick the best checkpoint.
"""
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def _load_run(run_dir: Path) -> Optional[dict]:
    info_path = run_dir / "run_info.json"
    metrics_path = run_dir / "metrics.csv"
    best_ckpt = run_dir / "checkpoints" / "model_best.pt"
    if not info_path.exists():
        return None
    try:
        with open(info_path) as f:
            info = json.load(f)
    except Exception:
        return None

    metrics_df = None
    best_auc = best_ap = None
    if metrics_path.exists():
        try:
            metrics_df = pd.read_csv(metrics_path)
            if "val_auc" in metrics_df.columns:
                best_auc = float(metrics_df["val_auc"].max())
            if "val_ap" in metrics_df.columns:
                best_ap = float(metrics_df["val_ap"].max())
        except Exception:
            pass

    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "model": info.get("model", "?"),
        "epochs_planned": info.get("epochs", "?"),
        "batch_size": info.get("batch_size", "?"),
        "lr": info.get("lr", "?"),
        "timestamp": info.get("timestamp_utc", "?"),
        "device": info.get("device", "?"),
        "best_ckpt": str(best_ckpt) if best_ckpt.exists() else None,
        "best_auc": best_auc,
        "best_ap": best_ap,
        "metrics_df": metrics_df,
    }


class TrainingRunsPage:
    @staticmethod
    def render():
        st.markdown("""
        <div style="background:linear-gradient(90deg,#4A148C 0%,#7B1FA2 100%);
             color:white;padding:20px 28px 16px 28px;border-radius:10px;margin-bottom:24px;
             box-sizing:border-box;width:100%;">
          <div style="font-size:1.7rem;font-weight:700;">&#128194; Training Run Browser</div>
          <div style="font-size:0.92rem;opacity:0.85;margin-top:4px;">
            Browse all local training runs &#8212; compare checkpoints, metrics, and learning curves
          </div>
          <div style="font-size:0.82rem;opacity:0.7;margin-top:8px;">
            Runs are saved under <code>runs/</code> each time you call
            <code>python src/train_local.py ...</code>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if not RUNS_DIR.exists():
            st.info("No `runs/` directory found. Run local training first (`make train-local` or `python src/train_local.py`).")
            return

        run_dirs = sorted(
            [d for d in RUNS_DIR.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not run_dirs:
            st.info("No training runs found in `runs/`. Run training first.")
            return

        runs = [_load_run(d) for d in run_dirs]
        runs = [r for r in runs if r is not None]

        if not runs:
            st.warning("Found run directories but could not read any run_info.json files.")
            return

        # ── Summary Table ────────────────────────────────────────────────────────
        st.subheader(f"Found {len(runs)} training run(s)")
        table_data = []
        for r in runs:
            table_data.append({
                "Run Name": r["name"],
                "Model": r["model"],
                "Epochs": r["epochs_planned"],
                "Batch Size": r["batch_size"],
                "LR": r["lr"],
                "Best AUROC": f"{r['best_auc']:.4f}" if r["best_auc"] is not None else "—",
                "Best AUPRC": f"{r['best_ap']:.4f}" if r["best_ap"] is not None else "—",
                "Checkpoint": "✅" if r["best_ckpt"] else "❌",
                "Timestamp": r["timestamp"],
            })
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True)

        # ── Run Selector ─────────────────────────────────────────────────────────
        st.markdown("---")
        run_names = [r["name"] for r in runs]
        selected_name = st.selectbox("Select a run to inspect:", run_names)
        selected = next((r for r in runs if r["name"] == selected_name), None)

        if selected is None:
            return

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Run Info**")
            st.json({
                "model": selected["model"],
                "epochs_planned": selected["epochs_planned"],
                "batch_size": selected["batch_size"],
                "lr": selected["lr"],
                "device": selected["device"],
                "timestamp_utc": selected["timestamp"],
            })
        with col2:
            st.markdown("**Best Metrics**")
            c1, c2 = st.columns(2)
            c1.metric("Best AUROC", f"{selected['best_auc']:.4f}" if selected["best_auc"] else "N/A")
            c2.metric("Best AUPRC", f"{selected['best_ap']:.4f}" if selected["best_ap"] else "N/A")

            if selected["best_ckpt"]:
                st.success("Checkpoint available")
                st.code(selected["best_ckpt"], language="text")
                st.caption("Set `SEPSIS_MODEL_PATH` to this path to load it in the Predict page.")
            else:
                st.error("No best checkpoint found.")

        # ── Training Curve ───────────────────────────────────────────────────────
        mdf = selected.get("metrics_df")
        if mdf is not None and not mdf.empty:
            st.markdown("### Training Curve")
            fig = go.Figure()
            if "train_loss" in mdf.columns:
                fig.add_trace(go.Scatter(x=mdf["epoch"], y=mdf["train_loss"], name="Train Loss",
                                         line=dict(color="#FF6F61"), yaxis="y2"))
            if "val_auc" in mdf.columns:
                fig.add_trace(go.Scatter(x=mdf["epoch"], y=mdf["val_auc"], name="Val AUROC",
                                          line=dict(color="#0052CC")))
            if "val_ap" in mdf.columns:
                fig.add_trace(go.Scatter(x=mdf["epoch"], y=mdf["val_ap"], name="Val AUPRC",
                                          line=dict(color="#2CA02C", dash="dash")))
            fig.update_layout(
                xaxis_title="Epoch",
                yaxis=dict(title="Metric (AUROC / AUPRC)", range=[0, 1.05]),
                yaxis2=dict(title="Train Loss", overlaying="y", side="right", showgrid=False),
                height=400,
                legend=dict(x=0.01, y=0.99),
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig, width='stretch')

            with st.expander("Raw metrics CSV"):
                st.dataframe(mdf, use_container_width=True)
        else:
            st.info("No metrics.csv found for this run.")
