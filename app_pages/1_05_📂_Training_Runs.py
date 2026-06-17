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
        # ── Page header ───────────────────────────────────────────────────────
        st.markdown("""
        <div style="margin-bottom:28px; padding-bottom:16px; border-bottom:1px solid #1E2A45;">
          <div style="display:inline-block; background:rgba(245,158,11,0.10); color:#FCD34D;
               padding:3px 10px; border-radius:20px; font-size:0.7rem; font-weight:700;
               letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;">RUNS</div>
          <div style="font-size:1.6rem; font-weight:700; color:#F1F5F9; letter-spacing:-0.3px;">
            📂 Training Run Browser
          </div>
          <div style="font-size:0.88rem; color:#64748B; margin-top:6px; max-width:640px;">
            Browse all local training runs — compare checkpoints, metrics, and learning curves.
            Runs are saved under <code style="background:#141827; border:1px solid #1E2A45;
            border-radius:4px; padding:1px 6px; color:#60A5FA;">runs/</code> each time you call
            <code style="background:#141827; border:1px solid #1E2A45;
            border-radius:4px; padding:1px 6px; color:#60A5FA;">python src/train_local.py ...</code>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if not RUNS_DIR.exists():
            st.info("No `runs/` directory found. Run local training first (`make train-local` or `python src/train_local.py`).")
            return

        run_dirs = sorted(
            [d for d in RUNS_DIR.iterdir() if d.is_dir() and not d.is_symlink()],
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

        # ── Summary Table ─────────────────────────────────────────────────────
        st.markdown(
            f'<div style="font-size:1.05rem; font-weight:700; color:#F1F5F9; margin-bottom:16px;">'
            f'Found {len(runs)} training run(s)</div>',
            unsafe_allow_html=True,
        )
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

        # ── Run Selector ──────────────────────────────────────────────────────
        st.markdown(
            '<div style="border-top:1px solid #1E2A45; margin:24px 0;"></div>',
            unsafe_allow_html=True,
        )
        run_names = [r["name"] for r in runs]
        selected_name = st.selectbox("Select a run to inspect:", run_names)
        selected = next((r for r in runs if r["name"] == selected_name), None)

        if selected is None:
            return

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '<div style="font-size:0.88rem; font-weight:600; color:#94A3B8; margin-bottom:8px;">Run Info</div>',
                unsafe_allow_html=True,
            )
            st.json({
                "model": selected["model"],
                "epochs_planned": selected["epochs_planned"],
                "batch_size": selected["batch_size"],
                "lr": selected["lr"],
                "device": selected["device"],
                "timestamp_utc": selected["timestamp"],
            })
        with col2:
            st.markdown(
                '<div style="font-size:0.88rem; font-weight:600; color:#94A3B8; margin-bottom:8px;">Best Metrics</div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            c1.metric("Best AUROC", f"{selected['best_auc']:.4f}" if selected["best_auc"] is not None else "N/A")
            c2.metric("Best AUPRC", f"{selected['best_ap']:.4f}" if selected["best_ap"] is not None else "N/A")

            if selected["best_ckpt"]:
                st.markdown(
                    '<div style="background:rgba(16,185,129,0.08); border-left:4px solid #10B981;'
                    'border-radius:0 8px 8px 0; padding:10px 14px; color:#6EE7B7; '
                    'font-size:0.85rem; margin-top:8px;">Checkpoint available</div>',
                    unsafe_allow_html=True,
                )
                st.code(selected["best_ckpt"], language="text")
                st.caption("Set `SEPSIS_MODEL_PATH` to this path to load it in the Predict page.")
            else:
                st.markdown(
                    '<div style="background:rgba(239,68,68,0.08); border-left:4px solid #EF4444;'
                    'border-radius:0 8px 8px 0; padding:10px 14px; color:#FCA5A5; '
                    'font-size:0.85rem; margin-top:8px;">No best checkpoint found.</div>',
                    unsafe_allow_html=True,
                )

        # ── Training Curve ────────────────────────────────────────────────────
        mdf = selected.get("metrics_df")
        if mdf is not None and not mdf.empty:
            st.markdown(
                '<div style="font-size:1.1rem; font-weight:700; color:#F1F5F9; margin:24px 0 12px 0;">'
                'Training Curve</div>',
                unsafe_allow_html=True,
            )
            fig = go.Figure()
            if "train_loss" in mdf.columns:
                fig.add_trace(go.Scatter(
                    x=mdf["epoch"], y=mdf["train_loss"], name="Train Loss",
                    line=dict(color="#EF4444"), yaxis="y2",
                ))
            if "val_auc" in mdf.columns:
                fig.add_trace(go.Scatter(
                    x=mdf["epoch"], y=mdf["val_auc"], name="Val AUROC",
                    line=dict(color="#3B82F6"),
                ))
            if "val_ap" in mdf.columns:
                fig.add_trace(go.Scatter(
                    x=mdf["epoch"], y=mdf["val_ap"], name="Val AUPRC",
                    line=dict(color="#10B981", dash="dash"),
                ))
            fig.update_layout(
                xaxis_title="Epoch",
                yaxis=dict(
                    title="Metric (AUROC / AUPRC)", range=[0, 1.05],
                    gridcolor="#1E2A45", tickfont=dict(color="#64748B"), linecolor="#1E2A45",
                ),
                yaxis2=dict(
                    title="Train Loss", overlaying="y", side="right", showgrid=False,
                    tickfont=dict(color="#64748B"), linecolor="#1E2A45",
                ),
                xaxis=dict(gridcolor="#1E2A45", tickfont=dict(color="#64748B"), linecolor="#1E2A45"),
                height=400,
                legend=dict(
                    x=0.01, y=0.99,
                    bgcolor="rgba(20,24,39,0.9)", bordercolor="#1E2A45", borderwidth=1,
                    font=dict(color="#94A3B8"),
                ),
                margin=dict(t=20, b=40),
                font=dict(family="Inter, Segoe UI, sans-serif", color="#94A3B8"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#141827",
            )
            st.plotly_chart(fig, width='stretch')

            with st.expander("Raw metrics CSV"):
                st.dataframe(mdf, use_container_width=True)
        else:
            st.info("No metrics.csv found for this run.")
