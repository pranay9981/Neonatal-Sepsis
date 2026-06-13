# 1_03_📈_Predict.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
import sys
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from model import TimeSeriesTransformer
    from config import (
        MODEL_PATH as _MODEL_PATH,
        EVAL_FEDERATED_JSON as _EVAL_FED,
        SCALER_PATH as _SCALER_PATH,
        N_FEATURES,
        SEQ_LEN,
    )
except Exception:
    TimeSeriesTransformer = None
    _MODEL_PATH = "server_out/global_best.pt"
    _EVAL_FED = "eval_results_federated.json"
    _SCALER_PATH = "data/processed/patients/scaler.json"
    N_FEATURES = 40
    SEQ_LEN = 48

MODEL_PATH = Path(_MODEL_PATH)
EVAL_FEDERATED_JSON = Path(_EVAL_FED)
SCALER_PATH = Path(_SCALER_PATH)

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
    if TimeSeriesTransformer is None:
        return None, "Could not import TimeSeriesTransformer"
    p = Path(model_path)
    if not p.exists():
        return None, f"Model file not found: {model_path}"
    try:
        m = TimeSeriesTransformer(n_features=n_features, seq_len=seq_len)
        state = torch.load(str(p), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        try:
            m.load_state_dict(state)
        except Exception:
            m.load_state_dict(state, strict=False)
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
    std = np.array(scaler["std"], dtype=np.float32)
    return (arr - mean) / (std + 1e-8)


def validate_input(df: pd.DataFrame) -> list:
    """Return list of warning strings for out-of-range or suspicious values."""
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


def preprocess_dataframe(df: pd.DataFrame, seq_len: int = SEQ_LEN, n_features: int = N_FEATURES, scaler=None):
    messages = []
    try:
        df_numeric = df.apply(pd.to_numeric, errors="coerce")
    except Exception as e:
        return None, [f"Error converting to numeric: {e}"]

    if df_numeric.isnull().values.any():
        n_missing = int(df_numeric.isnull().sum().sum())
        messages.append(f"Found {n_missing} non-numeric/missing values — filling with 0.")
        df_numeric = df_numeric.fillna(0)

    if df_numeric.shape[1] != n_features:
        return None, [f"Invalid input: expected {n_features} columns, got {df_numeric.shape[1]}"]

    rows = df_numeric.shape[0]
    if rows > seq_len:
        messages.append(f"Data has {rows} rows; truncating to last {seq_len}.")
        df_proc = df_numeric.tail(seq_len)
    elif rows < seq_len:
        pad = seq_len - rows
        messages.append(f"Data has {rows} rows; padding {pad} rows of zeros at the start.")
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
            out = model(tensor)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out.detach().cpu().squeeze()
            logit = out.reshape(-1)[0] if out.numel() > 1 else out
            return float(torch.sigmoid(logit).item()), None
    except Exception as e:
        return None, str(e)


def compute_feature_importance(model, tensor: torch.Tensor, scaler=None):
    """
    Compute per-feature importance using gradient saliency (|d_output/d_input|),
    averaged across the time dimension. Falls back to SHAP GradientExplainer when available.
    Returns array of shape (N_FEATURES,) or None on failure.
    """
    # Try SHAP first
    try:
        import shap

        if scaler:
            mean = np.array(scaler["mean"], dtype=np.float32)
            std = np.array(scaler["std"], dtype=np.float32)
            # Background: 20 samples drawn from standard normal (already normalised space)
            bg_np = np.random.default_rng(42).standard_normal((20, SEQ_LEN, N_FEATURES)).astype(np.float32)
        else:
            bg_np = np.zeros((20, SEQ_LEN, N_FEATURES), dtype=np.float32)

        bg_tensor = torch.from_numpy(bg_np)
        model.train()  # GradientExplainer requires gradient mode
        e = shap.GradientExplainer(model, bg_tensor)
        shap_values = e.shap_values(tensor)  # (1, SEQ_LEN, N_FEATURES)
        model.eval()
        # Average absolute SHAP values over the time dimension
        importance = np.abs(np.array(shap_values)).mean(axis=1).squeeze()
        return importance
    except Exception:
        pass

    # Fallback: gradient saliency
    try:
        x = tensor.clone().float().requires_grad_(True)
        model.eval()
        with torch.enable_grad():
            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out.sum().backward()
        saliency = x.grad.abs().mean(dim=1).squeeze(0)  # (N_FEATURES,)
        return saliency.detach().numpy()
    except Exception:
        return None


def plot_prediction_histogram(probs_neg, probs_pos, prediction_score):
    fig, ax = plt.subplots(figsize=(10, 4))
    if len(probs_neg) > 0:
        ax.hist(probs_neg, bins=50, alpha=0.7, label="Actual Low-Risk (Test Set)", density=True)
    if len(probs_pos) > 0:
        ax.hist(probs_pos, bins=50, alpha=0.7, label="Actual High-Risk (Test Set)", density=True)
    ax.axvline(prediction_score, color="red", linestyle="--", lw=3, label=f"Your Score ({prediction_score:.2f})")
    ax.set_title("Prediction Score vs. Test Set Outcomes")
    ax.set_xlabel("Predicted Risk Score (Probability)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig


def plot_threshold_sensitivity(y_true, y_prob, current_threshold):
    """Plot precision, recall, and F1 across thresholds."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    thresholds = np.linspace(0.0, 1.0, 101)
    precs, recs, f1s = [], [], []
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    for t in thresholds:
        y_pred = (y_prob_arr >= t).astype(int)
        precs.append(precision_score(y_true_arr, y_pred, zero_division=0))
        recs.append(recall_score(y_true_arr, y_pred, zero_division=0))
        f1s.append(f1_score(y_true_arr, y_pred, zero_division=0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precs, name="Precision", line=dict(color="#0052CC")))
    fig.add_trace(go.Scatter(x=thresholds, y=recs, name="Recall", line=dict(color="#FF6F61")))
    fig.add_trace(go.Scatter(x=thresholds, y=f1s, name="F1-Score", line=dict(color="#2CA02C", dash="dash")))
    fig.add_vline(x=current_threshold, line_color="red", line_dash="dot",
                  annotation_text=f"Threshold = {current_threshold:.2f}", annotation_position="top right")
    fig.update_layout(
        title="Threshold Sensitivity",
        xaxis_title="Decision Threshold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        height=350,
        margin=dict(t=50, b=40),
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
        st.title("📈 Live Prediction (Neonatal Sepsis)")
        st.info(
            f"Provide {N_FEATURES} features per timestep. Data will be padded/truncated to {SEQ_LEN} timesteps."
        )

        st.download_button(
            "Download Template CSV (random plausible values)",
            data=template_csv_random(),
            file_name="patient_template_random.csv",
            mime="text/csv",
        )

        model, model_err = load_model(str(MODEL_PATH))
        eval_data = load_eval_data(str(EVAL_FEDERATED_JSON))
        scaler = load_scaler(str(SCALER_PATH))
        if scaler is None:
            st.info("Feature scaler not found — run preprocessing first for best accuracy.")
        if model is None:
            st.warning(f"Model not available: {model_err or 'Model file missing.'}")
            st.info("Generate 'server_out/global_best.pt' (train or run FL) to enable predictions.")

        if "df_input_temp" not in st.session_state:
            st.session_state.df_input_temp = None

        st.subheader("Input Data")
        mode = st.radio("Input method:", (
            "Upload CSV file",
            "Paste CSV text",
            "Enter single row (40 values)",
            "Fill features manually",
        ))

        # ── Upload CSV ──────────────────────────────────────────────────────────
        if mode == "Upload CSV file":
            uploaded = st.file_uploader("Upload patient CSV (header optional)", type=["csv"])
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
                    st.success("CSV loaded.")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")

        # ── Paste CSV ───────────────────────────────────────────────────────────
        elif mode == "Paste CSV text":
            text = st.text_area("Paste CSV text (one timestep per line)", height=200,
                                placeholder="val1,val2,... (40 values per line)")
            if st.button("Parse pasted CSV"):
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
                        st.success("Parsed and stored.")

        # ── Single row ─────────────────────────────────────────────────────────
        elif mode == "Enter single row (40 values)":
            single_text = st.text_area(f"Enter {N_FEATURES} comma-separated values", height=100,
                                       placeholder="v1, v2, ..., v40")
            tile_option = st.radio("Expand to 48 timesteps by:", (
                "Tile the same row across all timesteps",
                "Use this as the most recent timestep, pad previous with zeros",
            ))
            if st.button("Parse single row"):
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
                        st.success("Stored.")
                    except Exception as e:
                        st.error(f"Parse error: {e}")

        # ── Manual fill ────────────────────────────────────────────────────────
        else:
            st.markdown("Fill each feature below (prefilled with plausible defaults).")
            cols = st.columns(4)
            manual_vals = {}
            per_col = int(np.ceil(len(FEATURE_NAMES) / 4))
            for col_idx, col in enumerate(cols):
                start = col_idx * per_col
                for fname in FEATURE_NAMES[start: start + per_col]:
                    default, vmin, vmax, step = FEATURE_SPEC.get(fname, (0.0, -1e6, 1e6, 0.1))
                    manual_vals[fname] = col.number_input(
                        label=fname, value=float(default),
                        min_value=float(vmin), max_value=float(vmax),
                        step=float(step),
                        format="%.3f" if step < 1 else "%.1f",
                        key=f"manual_{fname}",
                    )
            manual_tile = st.radio("Expand to 48 timesteps by:", (
                "Tile the same row across all timesteps",
                "Use this as the most recent timestep, pad previous with zeros",
            ), key="manual_tile")
            if st.button("Use manual inputs"):
                parsed = [manual_vals[f] for f in FEATURE_NAMES]
                if manual_tile.startswith("Tile"):
                    arr = np.tile(np.array(parsed)[None, :], (SEQ_LEN, 1))
                else:
                    arr = np.zeros((SEQ_LEN, N_FEATURES))
                    arr[-1, :] = parsed
                st.session_state.df_input_temp = pd.DataFrame(arr, columns=FEATURE_NAMES)
                st.success("Stored.")

        # ── If we have input ────────────────────────────────────────────────────
        if st.session_state.df_input_temp is not None:
            st.markdown("---")
            st.subheader("Preview (first 5 rows)")
            try:
                df_preview = st.session_state.df_input_temp.copy()
                if df_preview.shape[1] == N_FEATURES and list(df_preview.columns) != FEATURE_NAMES:
                    df_preview.columns = FEATURE_NAMES
                df_preview = df_preview.apply(pd.to_numeric, errors="coerce")
            except Exception:
                df_preview = st.session_state.df_input_temp
            st.dataframe(df_preview.head())

            # Input validation warnings
            range_warnings = validate_input(df_preview)
            if range_warnings:
                with st.expander(f"⚠️ {len(range_warnings)} out-of-range value(s) detected — expand to review"):
                    for w in range_warnings:
                        st.warning(w)

            tensor, messages = preprocess_dataframe(df_preview, SEQ_LEN, N_FEATURES, scaler=scaler)
            for m in messages:
                st.info(m)

            if tensor is None:
                st.error("Preprocessing failed. Fix input and try again.")
            else:
                threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
                if st.button("▶️ Run Prediction"):
                    if model is None:
                        st.error("Model not available.")
                        return
                    with st.spinner("Running model ..."):
                        prob, err = safe_predict(model, tensor)
                    if err:
                        st.error(f"Prediction error: {err}")
                        return

                    # ── Result ──────────────────────────────────────────────────
                    st.subheader("Prediction Result")
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob * 100,
                            title={"text": "Sepsis Risk (%)", "font": {"size": 20}},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "darkblue"},
                                "steps": [
                                    {"range": [0, 25], "color": "rgba(44,160,44,0.7)"},
                                    {"range": [25, 50], "color": "rgba(255,127,14,0.7)"},
                                    {"range": [50, 100], "color": "rgba(214,39,40,0.7)"},
                                ],
                                "threshold": {"line": {"color": "red", "width": 4}, "value": threshold * 100},
                            },
                        ))
                        fig_gauge.update_layout(height=300, margin=dict(t=30, b=10))
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    with col2:
                        if prob > threshold:
                            st.error("HIGH RISK: Sepsis likely.", icon="⚠️")
                            st.markdown("""
                            **Clinical Recommendations:**
                            - Immediate clinical evaluation required.
                            - Consider blood cultures and early antibiotics per local protocol.
                            - Monitor vitals and inflammatory markers (CRP, PCT).
                            """)
                        elif prob > threshold * 0.5:
                            st.warning("Moderate Risk: Increased vigilance suggested.", icon="ℹ️")
                            st.markdown("""
                            **Recommendations:**
                            - Increase monitoring frequency.
                            - Low threshold for labs or escalation.
                            """)
                        else:
                            st.success("Low Risk: Continue routine monitoring.", icon="✅")
                            st.markdown("Continue standard monitoring and follow hospital protocols.")

                    # ── Analysis tabs ───────────────────────────────────────────
                    tab1, tab2, tab3 = st.tabs(["📊 vs Test Set", "🔬 Feature Importance (SHAP)", "📉 Threshold Sensitivity"])

                    with tab1:
                        if eval_data:
                            arr_probs = np.array(eval_data["y_prob"])
                            arr_true = np.array(eval_data["y_true"])
                            st.pyplot(plot_prediction_histogram(
                                arr_probs[arr_true == 0], arr_probs[arr_true == 1], prob
                            ))
                        else:
                            st.info("Evaluation file not found — run evaluation to display histogram.")

                    with tab2:
                        with st.spinner("Computing feature importance ..."):
                            importance = compute_feature_importance(model, tensor, scaler)
                        if importance is not None:
                            top_n = 15
                            top_idx = np.argsort(importance)[::-1][:top_n]
                            top_names = [FEATURE_NAMES[i] for i in top_idx]
                            top_vals = importance[top_idx]
                            top_vals_norm = top_vals / (top_vals.max() + 1e-8)
                            fig_imp = go.Figure(go.Bar(
                                x=top_vals_norm[::-1],
                                y=top_names[::-1],
                                orientation="h",
                                marker_color="steelblue",
                            ))
                            fig_imp.update_layout(
                                title=f"Top {top_n} Most Influential Features",
                                xaxis_title="Normalised Importance",
                                height=420,
                                margin=dict(l=20, r=20, t=50, b=40),
                            )
                            st.plotly_chart(fig_imp, use_container_width=True)
                            st.caption("Importance = |gradient| averaged over timesteps. Higher = more influence on prediction.")
                        else:
                            st.warning("Feature importance could not be computed for this input.")

                    with tab3:
                        if eval_data:
                            st.plotly_chart(
                                plot_threshold_sensitivity(eval_data["y_true"], eval_data["y_prob"], threshold),
                                use_container_width=True,
                            )
                            st.caption("Based on the held-out test set. Adjust the threshold slider above to see how it affects precision and recall.")
                        else:
                            st.info("Evaluation file not found — run evaluation to display threshold sensitivity.")
