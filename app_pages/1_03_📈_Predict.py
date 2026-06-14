# 1_03_📈_Predict.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
import sys
from typing import Optional
import plotly.graph_objects as go
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os

try:
    from model import TimeSeriesTransformer
    from model_grud import GRUD
    from config import (
        MODEL_PATH as _MODEL_PATH,
        EVAL_FEDERATED_JSON as _EVAL_FED,
        SCALER_PATH as _SCALER_PATH,
        N_FEATURES,
        SEQ_LEN,
    )
except Exception:
    TimeSeriesTransformer = None
    GRUD = None
    _MODEL_PATH = "server_out/global_best.pt"
    _EVAL_FED   = "eval_results_federated.json"
    _SCALER_PATH = "data/processed/patients/scaler.json"
    N_FEATURES = 40
    SEQ_LEN = 48

MODEL_TYPE = os.environ.get("SEPSIS_MODEL_TYPE", "grud")

MODEL_PATH          = Path(_MODEL_PATH)
EVAL_FEDERATED_JSON = Path(_EVAL_FED)
SCALER_PATH         = Path(_SCALER_PATH)

FEATURE_NAMES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
]
assert len(FEATURE_NAMES) == N_FEATURES

FEATURE_SPEC = {
    "HR": (140.0, 30.0, 240.0, 1.0),
    "O2Sat": (98.0, 50.0, 100.0, 0.1),
    "Temp": (36.8, 30.0, 40.0, 0.1),
    "SBP": (65.0, 20.0, 200.0, 1.0),
    "MAP": (45.0, 10.0, 150.0, 1.0),
    "DBP": (40.0, 10.0, 150.0, 1.0),
    "Resp": (40.0, 5.0, 120.0, 1.0),
    "EtCO2": (35.0, 0.0, 100.0, 0.1),
    "BaseExcess": (0.0, -30.0, 30.0, 0.1),
    "HCO3": (22.0, 0.0, 60.0, 0.1),
    "FiO2": (21.0, 21.0, 100.0, 1.0),
    "pH": (7.35, 6.5, 7.8, 0.01),
    "PaCO2": (40.0, 5.0, 150.0, 0.1),
    "SaO2": (98.0, 50.0, 100.0, 0.1),
    "AST": (30.0, 0.0, 1000.0, 1.0),
    "BUN": (8.0, 0.0, 200.0, 0.1),
    "Alkalinephos": (120.0, 0.0, 2000.0, 1.0),
    "Calcium": (9.0, 0.0, 20.0, 0.1),
    "Chloride": (100.0, 50.0, 140.0, 0.1),
    "Creatinine": (0.5, 0.0, 10.0, 0.01),
    "Bilirubin_direct": (0.1, 0.0, 20.0, 0.01),
    "Glucose": (80.0, 10.0, 1000.0, 1.0),
    "Lactate": (1.5, 0.0, 30.0, 0.1),
    "Magnesium": (2.0, 0.5, 5.0, 0.01),
    "Phosphate": (4.0, 0.5, 10.0, 0.1),
    "Potassium": (4.0, 1.0, 10.0, 0.01),
    "Bilirubin_total": (1.0, 0.0, 30.0, 0.01),
    "TroponinI": (0.01, 0.0, 50.0, 0.01),
    "Hct": (40.0, 10.0, 70.0, 0.1),
    "Hgb": (14.0, 5.0, 25.0, 0.1),
    "PTT": (35.0, 10.0, 200.0, 0.1),
    "WBC": (10.0, 0.1, 100.0, 0.1),
    "Fibrinogen": (250.0, 50.0, 1000.0, 1.0),
    "Platelets": (250.0, 10.0, 1000.0, 1.0),
    "Age": (0.1, 0.0, 3650.0, 0.1),
    "Gender": (0.0, 0.0, 1.0, 1.0),
    "Unit1": (0.0, 0.0, 1.0, 1.0),
    "Unit2": (0.0, 0.0, 1.0, 1.0),
    "HospAdmTime": (0.0, 0.0, 10000.0, 0.1),
    "ICULOS": (0.0, 0.0, 10000.0, 0.1),
}
for fname in FEATURE_NAMES:
    if fname not in FEATURE_SPEC:
        FEATURE_SPEC[fname] = (0.0, -1e6, 1e6, 0.1)


# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str, n_features: int = N_FEATURES, seq_len: int = SEQ_LEN):
    p = Path(model_path)
    if not p.exists():
        return None, f"Model file not found: {model_path}"
    try:
        if MODEL_TYPE == "grud":
            if GRUD is None:
                return None, "Could not import GRUD"
            m = GRUD(n_features=n_features)
        else:
            if TimeSeriesTransformer is None:
                return None, "Could not import TimeSeriesTransformer"
            m = TimeSeriesTransformer(n_features=n_features, seq_len=seq_len)
        state = torch.load(str(p), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        m.load_state_dict(state)
        m.eval()
        return m, None
    except Exception as e:
        return None, f"Error loading model: {e}"


@st.cache_data
def load_eval_data(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data
def load_scaler(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


# ── Helpers ────────────────────────────────────────────────────────────────────
def _apply_scaler(arr: np.ndarray, scaler: dict) -> np.ndarray:
    mean = np.array(scaler["mean"], dtype=np.float32)
    std  = np.array(scaler["std"],  dtype=np.float32)
    return (arr - mean) / (std + 1e-8)


def validate_input(df: pd.DataFrame) -> list:
    warnings = []
    for fname in FEATURE_NAMES:
        if fname not in df.columns:
            continue
        _, vmin, vmax, _ = FEATURE_SPEC.get(fname, (0, -1e9, 1e9, 0.1))
        col = pd.to_numeric(df[fname], errors="coerce")
        n_out = int(((col < vmin) | (col > vmax)).sum())
        if n_out > 0:
            warnings.append(f"**{fname}**: {n_out} timestep(s) outside expected range [{vmin}, {vmax}]")
    return warnings


def preprocess_dataframe(df: pd.DataFrame, seq_len: int = SEQ_LEN,
                          n_features: int = N_FEATURES, scaler=None):
    messages = []
    try:
        df_numeric = df.apply(pd.to_numeric, errors="coerce")
    except Exception as e:
        return None, [f"Error converting to numeric: {e}"]

    if df_numeric.isnull().values.any():
        n_missing = int(df_numeric.isnull().sum().sum())
        messages.append(f"Found {n_missing} missing/non-numeric values — filling with 0.")
        df_numeric = df_numeric.fillna(0)

    if df_numeric.shape[1] != n_features:
        return None, [f"Expected {n_features} columns, got {df_numeric.shape[1]}"]

    rows = df_numeric.shape[0]
    if rows > seq_len:
        messages.append(f"Input has {rows} rows — truncating to last {seq_len} (most recent hours).")
        df_proc = df_numeric.tail(seq_len)
    elif rows < seq_len:
        pad = seq_len - rows
        messages.append(f"Input has {rows} rows — zero-padding {pad} rows at the start.")
        pad_df = pd.DataFrame(np.zeros((pad, n_features)), columns=df_numeric.columns)
        df_proc = pd.concat([pad_df, df_numeric], ignore_index=True)
    else:
        df_proc = df_numeric

    data_np = df_proc.to_numpy(dtype=np.float32)
    if scaler is not None:
        data_np = _apply_scaler(data_np, scaler)
    return torch.tensor(data_np).unsqueeze(0), messages


def safe_predict(model, tensor):
    try:
        model.eval()
        with torch.no_grad():
            if MODEL_TYPE == "grud":
                mask = torch.ones_like(tensor)
                deltas = torch.zeros_like(tensor)
                out = model(tensor, mask, deltas)
            else:
                out = model(tensor)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out   = out.detach().cpu().squeeze()
            logit = out.reshape(-1)[0] if out.numel() > 1 else out
            return float(torch.sigmoid(logit).item()), None
    except Exception as e:
        return None, str(e)


def _model_forward(model, x):
    if MODEL_TYPE == "grud":
        mask = torch.ones_like(x)
        deltas = torch.zeros_like(x)
        return model(x, mask, deltas)
    return model(x)


def compute_feature_importance(model, tensor: torch.Tensor, scaler=None) -> Optional[np.ndarray]:
    try:
        x = tensor.clone().float().requires_grad_(True)
        model.eval()
        with torch.enable_grad():
            out = _model_forward(model, x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out.sum().backward()
        return x.grad.abs().mean(dim=1).squeeze(0).detach().numpy()
    except Exception:
        return None


def plot_prediction_histogram(probs_neg, probs_pos, score):
    fig, ax = plt.subplots(figsize=(10, 4))
    if len(probs_neg) > 0:
        ax.hist(probs_neg, bins=50, alpha=0.7, color="#2196F3", label="No Sepsis (test set)", density=True)
    if len(probs_pos) > 0:
        ax.hist(probs_pos, bins=50, alpha=0.7, color="#F44336", label="Sepsis (test set)", density=True)
    ax.axvline(score, color="#FF6F00", linestyle="--", lw=3, label=f"This patient ({score:.2f})")
    ax.set_title("Where this patient sits in the test-set distribution", fontsize=13)
    ax.set_xlabel("Predicted Sepsis Probability")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_threshold_sensitivity(y_true, y_prob, current_threshold):
    from sklearn.metrics import precision_score, recall_score, f1_score
    thresholds   = np.linspace(0.0, 1.0, 101)
    y_true_arr   = np.array(y_true)
    y_prob_arr   = np.array(y_prob)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        y_pred = (y_prob_arr >= t).astype(int)
        precs.append(precision_score(y_true_arr, y_pred, zero_division=0))
        recs.append(recall_score(y_true_arr, y_pred, zero_division=0))
        f1s.append(f1_score(y_true_arr, y_pred, zero_division=0))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precs, name="Precision", line=dict(color="#1E3A5F", width=2)))
    fig.add_trace(go.Scatter(x=thresholds, y=recs,  name="Recall",    line=dict(color="#F44336", width=2)))
    fig.add_trace(go.Scatter(x=thresholds, y=f1s,   name="F1-Score",  line=dict(color="#2E7D32", width=2, dash="dash")))
    fig.add_vline(x=current_threshold, line_color="#FF6F00", line_dash="dot",
                  annotation_text=f"Current threshold = {current_threshold:.2f}",
                  annotation_position="top right")
    fig.update_layout(
        title="Precision / Recall / F1 across all thresholds",
        xaxis_title="Decision Threshold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        height=360,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Inter, Segoe UI, sans-serif"),
    )
    return fig


def parse_text_to_df(text: str):
    try:
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if not lines:
            return None, "No data found."
        rows = [[p.strip() for p in ln.split(",") if p.strip()] for ln in lines]
        col_counts = {len(r) for r in rows}
        if len(col_counts) > 1:
            return None, f"Inconsistent column counts: {sorted(col_counts)}"
        return pd.DataFrame(rows), None
    except Exception as e:
        return None, f"Failed to parse: {e}"


def template_csv_random():
    rng = np.random.default_rng()
    rows = []
    for _ in range(SEQ_LEN):
        row = []
        for fname in FEATURE_NAMES:
            default, vmin, vmax, step = FEATURE_SPEC.get(fname, (0.0, -1e6, 1e6, 0.1))
            if fname in ("Gender", "Unit1", "Unit2"):
                row.append(int(rng.integers(0, 2)))
            elif step >= 1.0:
                row.append(int(rng.integers(int(np.ceil(vmin)), int(np.floor(vmax)) + 1)))
            else:
                precision = max(0, int(-np.floor(np.log10(step)))) if step != 0 else 3
                row.append(round(float(rng.uniform(vmin, vmax)), precision))
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_NAMES).to_csv(index=False).encode("utf-8")


# ── Page ───────────────────────────────────────────────────────────────────────
class PredictPage:
    @staticmethod
    def render():
        # ── Header ────────────────────────────────────────────────────────
        st.markdown("""
        <div style="background:linear-gradient(90deg,#1E3A5F 0%,#2563EB 100%);
             color:white;padding:20px 28px 16px 28px;border-radius:10px;margin-bottom:24px;
             box-sizing:border-box;width:100%;">
          <div style="font-size:1.7rem;font-weight:700;">&#128202; Live Patient Prediction</div>
          <div style="font-size:0.92rem;opacity:0.85;margin-top:4px;">
            Provide 40 clinical features across up to 48 ICU hours &#8594; get a sepsis probability score
          </div>
          <div style="font-size:0.82rem;opacity:0.7;margin-top:8px;">
            The model looks at trends over time, not just the most recent values.
            More hours of data = better accuracy.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Model status ───────────────────────────────────────────────────
        model, model_err = load_model(str(MODEL_PATH))
        eval_data = load_eval_data(str(EVAL_FEDERATED_JSON))
        scaler    = load_scaler(str(SCALER_PATH))

        stat_col1, stat_col2, stat_col3 = st.columns(3)
        _stat = lambda ok, yes, no: (
            f'<div style="background:{"#E8F5E9" if ok else "#FDECEA"};border-radius:8px;'
            f'padding:10px 14px;color:{"#1B5E20" if ok else "#B71C1C"};font-size:0.85rem;font-weight:600;">'
            f'{"&#10003;" if ok else "&#10007;"} {yes if ok else no}</div>'
        )
        stat_col1.markdown(_stat(model is not None,    "Model loaded",           f"Model missing — {model_err}"), unsafe_allow_html=True)
        stat_col2.markdown(_stat(scaler is not None,   "Scaler loaded",          "Scaler not found — predictions may be less accurate"), unsafe_allow_html=True)
        stat_col3.markdown(_stat(eval_data is not None,"Eval data loaded",       "No eval data — some analysis tabs will be empty"), unsafe_allow_html=True)

        if model is None:
            st.markdown("""
            <div style="background:#FFF3E0;border-left:4px solid #FF9800;border-radius:0 8px 8px 0;
                 padding:14px 18px;color:#E65100;margin-top:12px;">
            <b>Model not available.</b> Train a model first, then run FL simulation:<br>
            <code>python src/train_local.py --index data/splits/train_index.pt --model transformer</code><br>
            <code>python scripts/run_fl_sim.py --client_indexes ...</code>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── How to use this page ───────────────────────────────────────────
        with st.expander("How to use this page — input format guide"):
            st.markdown(f"""
            This page accepts patient data in **four ways** (choose below):

            | Method | Best for |
            |---|---|
            | **Upload CSV** | You have a real patient file with {N_FEATURES} feature columns |
            | **Paste CSV text** | Quick testing — paste comma-separated values directly |
            | **Single row** | You only have the latest hour of measurements |
            | **Fill manually** | Testing specific clinical scenarios — all 40 sliders |

            **Data format:**
            - Each row = one ICU hour
            - Each column = one of the 40 features (HR, O2Sat, Temp, SBP, ...)
            - Missing values: leave blank or use `NaN` — will be filled with 0
            - The model uses the **last {SEQ_LEN} hours**; shorter data is zero-padded at the start
            - Column order must match: `{", ".join(FEATURE_NAMES[:8])}, ...`

            **Template:** Download a randomly-generated valid CSV below to see the exact format.
            """)

        col_dl, _ = st.columns([1, 3])
        col_dl.download_button(
            "Download example CSV template",
            data=template_csv_random(),
            file_name="patient_template_example.csv",
            mime="text/csv",
        )

        # ── Input mode ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Step 1 — Provide patient data")

        mode = st.radio(
            "Input method:",
            ("Upload CSV file", "Paste CSV text", "Enter single row (40 values)", "Fill features manually"),
            horizontal=True,
        )

        if "df_input_temp" not in st.session_state:
            st.session_state.df_input_temp = None

        # ── Upload CSV ──────────────────────────────────────────────────────
        if mode == "Upload CSV file":
            uploaded = st.file_uploader(
                f"Upload patient CSV ({N_FEATURES} feature columns, one row per ICU hour)",
                type=["csv"],
            )
            if uploaded is not None:
                try:
                    uploaded.seek(0)
                    df_try = pd.read_csv(uploaded)
                    if df_try.shape[1] == N_FEATURES:
                        df_input = df_try.copy()
                    else:
                        uploaded.seek(0)
                        df_input = pd.read_csv(uploaded, header=None)
                    if df_input.shape[1] == N_FEATURES and list(df_input.columns) != FEATURE_NAMES:
                        df_input.columns = FEATURE_NAMES
                    st.session_state.df_input_temp = df_input
                    st.success(f"CSV loaded: {df_input.shape[0]} rows × {df_input.shape[1]} columns")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")

        # ── Paste CSV ───────────────────────────────────────────────────────
        elif mode == "Paste CSV text":
            st.markdown('<span style="font-size:0.85rem;color:#64748B;">One ICU hour per line, 40 comma-separated values per line</span>', unsafe_allow_html=True)
            text = st.text_area("Paste CSV text", height=180, placeholder="140,98.5,36.8,65,45,40,40,35,...")
            if st.button("Parse", type="primary"):
                if not text.strip():
                    st.error("No text entered.")
                else:
                    df_parsed, msg = parse_text_to_df(text)
                    if df_parsed is None:
                        st.error(msg)
                    else:
                        if df_parsed.shape[1] == N_FEATURES:
                            df_parsed.columns = FEATURE_NAMES
                        st.session_state.df_input_temp = df_parsed
                        st.success(f"Parsed: {df_parsed.shape[0]} rows")

        # ── Single row ──────────────────────────────────────────────────────
        elif mode == "Enter single row (40 values)":
            st.markdown('<span style="font-size:0.85rem;color:#64748B;">Enter the most recent hour\'s measurements as 40 comma-separated numbers</span>', unsafe_allow_html=True)
            single_text = st.text_area(f"40 comma-separated values", height=80,
                                       placeholder="140, 98.5, 36.8, 65, 45, 40, 40, 35, ...")
            tile_option = st.radio(
                "How to expand to 48 timesteps:",
                ("Tile this row across all 48 hours", "Use as latest hour, zero-pad the earlier hours"),
                help="The model needs 48 timesteps. Tiling repeats your single row — zero-pad is more realistic.",
            )
            if st.button("Use this row", type="primary"):
                parts = [p.strip() for p in single_text.strip().split(",") if p.strip()]
                if len(parts) != N_FEATURES:
                    st.error(f"Expected {N_FEATURES} values, got {len(parts)}.")
                else:
                    try:
                        row_vals = [float(x) for x in parts]
                        if tile_option.startswith("Tile"):
                            arr = np.tile(np.array(row_vals)[None, :], (SEQ_LEN, 1))
                        else:
                            arr = np.zeros((SEQ_LEN, N_FEATURES))
                            arr[-1, :] = row_vals
                        st.session_state.df_input_temp = pd.DataFrame(arr, columns=FEATURE_NAMES)
                        st.success("Stored — scroll down to run prediction.")
                    except Exception as e:
                        st.error(f"Parse error: {e}")

        # ── Manual fill ─────────────────────────────────────────────────────
        else:
            st.markdown('<span style="font-size:0.85rem;color:#64748B;">Set each feature value below (pre-filled with typical neonatal ICU defaults)</span>', unsafe_allow_html=True)
            cols = st.columns(4)
            manual_vals = {}
            per_col = int(np.ceil(len(FEATURE_NAMES) / 4))
            for col_idx, col in enumerate(cols):
                for fname in FEATURE_NAMES[col_idx * per_col: (col_idx + 1) * per_col]:
                    default, vmin, vmax, step = FEATURE_SPEC.get(fname, (0.0, -1e6, 1e6, 0.1))
                    manual_vals[fname] = col.number_input(
                        label=fname, value=float(default),
                        min_value=float(vmin), max_value=float(vmax), step=float(step),
                        format="%.3f" if step < 1 else "%.1f",
                        key=f"manual_{fname}",
                    )
            manual_tile = st.radio(
                "Expand to 48 timesteps:",
                ("Tile across all 48 hours", "Use as latest hour, zero-pad earlier hours"),
                key="manual_tile",
            )
            if st.button("Use these values", type="primary"):
                parsed = [manual_vals[f] for f in FEATURE_NAMES]
                arr = (np.tile(np.array(parsed)[None, :], (SEQ_LEN, 1))
                       if manual_tile.startswith("Tile")
                       else np.vstack([np.zeros((SEQ_LEN - 1, N_FEATURES)),
                                       np.array(parsed)[None, :]]))
                st.session_state.df_input_temp = pd.DataFrame(arr, columns=FEATURE_NAMES)
                st.success("Stored — scroll down to run prediction.")

        # ── Preview + run ───────────────────────────────────────────────────
        if st.session_state.df_input_temp is not None:
            st.markdown("---")
            st.markdown("### Step 2 — Review and run")

            df_preview = st.session_state.df_input_temp.copy()
            if df_preview.shape[1] == N_FEATURES and list(df_preview.columns) != FEATURE_NAMES:
                df_preview.columns = FEATURE_NAMES
            df_preview = df_preview.apply(pd.to_numeric, errors="coerce")

            with st.expander(f"Data preview ({df_preview.shape[0]} rows × {df_preview.shape[1]} columns)"):
                st.dataframe(df_preview, use_container_width=True)

            range_warnings = validate_input(df_preview)
            if range_warnings:
                with st.expander(f"⚠️ {len(range_warnings)} out-of-range value(s) — click to review"):
                    for w in range_warnings:
                        st.warning(w)

            tensor, messages = preprocess_dataframe(df_preview, SEQ_LEN, N_FEATURES, scaler=scaler)
            for m in messages:
                st.info(m)

            if tensor is None:
                st.error("Preprocessing failed. Fix the input and try again.")
                return

            c_thresh, c_run = st.columns([3, 1])
            threshold = c_thresh.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01,
                                         help="Probability above this → HIGH RISK alert")
            run_clicked = c_run.button("Run prediction", type="primary", use_container_width=True)

            if run_clicked:
                if model is None:
                    st.error("Model not available — train or run FL first.")
                    return
                with st.spinner("Running model inference..."):
                    prob, err = safe_predict(model, tensor)
                if err:
                    st.error(f"Prediction error: {err}")
                    return

                # ── Result banner ───────────────────────────────────────────
                st.markdown("---")
                st.markdown("### Result")

                risk_pct = prob * 100
                if prob > threshold:
                    banner_bg, banner_border, banner_color = "#FDECEA", "#F44336", "#B71C1C"
                    risk_label = "HIGH RISK"
                    risk_icon  = "&#9888;"
                    advice = (
                        "Immediate clinical evaluation recommended. "
                        "Consider blood cultures, CBC, CRP/PCT, and early antibiotic coverage "
                        "per local protocol. Increase monitoring frequency."
                    )
                elif prob > threshold * 0.6:
                    banner_bg, banner_border, banner_color = "#FFF8E1", "#FF9800", "#E65100"
                    risk_label = "MODERATE RISK"
                    risk_icon  = "&#8505;"
                    advice = (
                        "Elevated vigilance advised. "
                        "Consider repeat assessment in 2–4 hours. Low threshold for labs "
                        "or escalation if clinical picture deteriorates."
                    )
                else:
                    banner_bg, banner_border, banner_color = "#E8F5E9", "#4CAF50", "#1B5E20"
                    risk_label = "LOW RISK"
                    risk_icon  = "&#10003;"
                    advice = "Continue standard monitoring. Re-assess if clinical condition changes."

                st.markdown(
                    f'<div style="background:{banner_bg};border-left:6px solid {banner_border};'
                    f'border-radius:0 10px 10px 0;padding:16px 20px;margin:12px 0;">'
                    f'<span style="font-size:1.3rem;font-weight:700;color:{banner_color};">'
                    f'{risk_icon} {risk_label} — {risk_pct:.1f}%</span><br>'
                    f'<span style="font-size:0.9rem;color:{banner_color};opacity:0.9;margin-top:4px;display:block;">'
                    f'{advice}</span></div>',
                    unsafe_allow_html=True,
                )

                # ── Gauge + score breakdown ─────────────────────────────────
                col_gauge, col_info = st.columns([1, 1])
                with col_gauge:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_pct,
                        number={"suffix": "%", "font": {"size": 32, "color": "#1E3A5F"}},
                        title={"text": "Sepsis Risk Score", "font": {"size": 16, "color": "#64748B"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickfont": {"size": 11}},
                            "bar":  {"color": banner_border, "thickness": 0.25},
                            "steps": [
                                {"range": [0,  30], "color": "#E8F5E9"},
                                {"range": [30, 60], "color": "#FFF8E1"},
                                {"range": [60, 100], "color": "#FDECEA"},
                            ],
                            "threshold": {
                                "line": {"color": "#1E3A5F", "width": 3},
                                "value": threshold * 100,
                            },
                        },
                    ))
                    fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col_info:
                    st.markdown(
                        '<div style="font-size:0.8rem;font-weight:600;color:#64748B;'
                        'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px;">'
                        'Score breakdown</div>',
                        unsafe_allow_html=True,
                    )
                    for label, val in [
                        ("Raw probability", f"{prob:.4f}"),
                        ("As percentage",   f"{risk_pct:.1f}%"),
                        ("Threshold",        f"{threshold:.2f}"),
                        ("Decision",         risk_label),
                    ]:
                        st.markdown(
                            f'<div style="background:#F8FAFC;border:1px solid #E2E8F0;'
                            f'border-radius:6px;padding:8px 12px;margin:4px 0;'
                            f'display:flex;justify-content:space-between;">'
                            f'<span style="color:#64748B;font-size:0.85rem;">{label}</span>'
                            f'<span style="font-weight:600;color:#1E3A5F;font-size:0.85rem;">{val}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # ── Analysis tabs ───────────────────────────────────────────
                st.markdown("---")
                tab1, tab2, tab3 = st.tabs([
                    "Distribution comparison",
                    "Feature importance",
                    "Threshold sensitivity",
                ])

                with tab1:
                    st.markdown("**How this patient compares to the test set**")
                    st.markdown(
                        '<span style="font-size:0.85rem;color:#64748B;">Blue = no-sepsis patients, '
                        'Red = sepsis patients, Orange line = this patient\'s score</span>',
                        unsafe_allow_html=True,
                    )
                    if eval_data:
                        arr_probs = np.array(eval_data["y_prob"])
                        arr_true  = np.array(eval_data["y_true"])
                        st.pyplot(plot_prediction_histogram(
                            arr_probs[arr_true == 0], arr_probs[arr_true == 1], prob
                        ))
                    else:
                        st.info("Run evaluation first to enable this chart: `python src/evaluate.py ...`")

                with tab2:
                    st.markdown("**Which features most influenced this prediction**")
                    st.markdown(
                        '<span style="font-size:0.85rem;color:#64748B;">Computed via gradient saliency '
                        '(|d_output/d_input| averaged over timesteps). Higher = more influence.</span>',
                        unsafe_allow_html=True,
                    )
                    with st.spinner("Computing feature importance..."):
                        importance = compute_feature_importance(model, tensor, scaler)
                    if importance is not None:
                        top_n   = 15
                        top_idx = np.argsort(importance)[::-1][:top_n]
                        top_names = [FEATURE_NAMES[i] for i in top_idx]
                        top_vals  = importance[top_idx] / (importance[top_idx].max() + 1e-8)
                        fig_imp = go.Figure(go.Bar(
                            x=top_vals[::-1], y=top_names[::-1],
                            orientation="h",
                            marker_color=[
                                f"rgba(30,58,95,{0.4 + 0.6 * v})" for v in top_vals[::-1]
                            ],
                        ))
                        fig_imp.update_layout(
                            title=f"Top {top_n} most influential features",
                            xaxis_title="Normalised importance",
                            height=420, margin=dict(l=20, r=20, t=50, b=40),
                            font=dict(family="Inter, Segoe UI, sans-serif"),
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)
                    else:
                        st.warning("Feature importance could not be computed for this input.")

                with tab3:
                    st.markdown("**How precision and recall change across thresholds**")
                    st.markdown(
                        '<span style="font-size:0.85rem;color:#64748B;">Based on the frozen test set. '
                        'Use this to choose a threshold that balances catching sepsis vs false alarms.</span>',
                        unsafe_allow_html=True,
                    )
                    if eval_data:
                        st.plotly_chart(
                            plot_threshold_sensitivity(eval_data["y_true"], eval_data["y_prob"], threshold),
                            use_container_width=True,
                        )
                    else:
                        st.info("Run evaluation first to enable this chart.")
