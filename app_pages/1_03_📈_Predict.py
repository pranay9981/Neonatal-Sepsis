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

# ensure src is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
try:
    from model import TimeSeriesTransformer
except Exception:
    TimeSeriesTransformer = None

# ---------- Constants ----------
MODEL_PATH = Path("server_out/global_best.pt")
N_FEATURES = 40
SEQ_LEN = 48
EVAL_FEDERATED_JSON = Path("eval_results_federated.json")

FEATURE_NAMES = [
    "HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2",
    "BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN",
    "Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct",
    "Glucose","Lactate","Magnesium","Phosphate","Potassium","Bilirubin_total",
    "TroponinI","Hct","Hgb","PTT","WBC","Fibrinogen","Platelets","Age","Gender","Unit1","Unit2",
    "HospAdmTime","ICULOS"
]
assert len(FEATURE_NAMES) == N_FEATURES

# Clinically plausible defaults and ranges
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
    "ICULOS": (0.0, 0.0, 10000.0, 0.1)
}
for fname in FEATURE_NAMES:
    if fname not in FEATURE_SPEC:
        FEATURE_SPEC[fname] = (0.0, -1e6, 1e6, 0.1)

# ---------- Helpers ----------
@st.cache_resource
def load_model(model_path: str, n_features: int = N_FEATURES, seq_len: int = SEQ_LEN):
    if TimeSeriesTransformer is None:
        return None, "Could not import TimeSeriesTransformer from src/model.py"
    p = Path(model_path)
    if not p.exists():
        return None, f"Model file not found at '{model_path}'"
    try:
        model = TimeSeriesTransformer(n_features=n_features, seq_len=seq_len)
        state = torch.load(str(p), map_location=torch.device("cpu"))
        try:
            model.load_state_dict(state)
        except Exception:
            model.load_state_dict(state, strict=False)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Error loading model: {e}"

@st.cache_data
def load_eval_data(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None

def preprocess_dataframe(df: pd.DataFrame, seq_len: int = SEQ_LEN, n_features: int = N_FEATURES):
    messages = []
    try:
        df_numeric = df.apply(pd.to_numeric, errors="coerce")
    except Exception as e:
        return None, [f"Error converting CSV to numeric: {e}"]

    if df_numeric.isnull().values.any():
        n_missing = int(df_numeric.isnull().sum().sum())
        messages.append(f"Found {n_missing} non-numeric/missing values — filling with 0.")
        df_numeric = df_numeric.fillna(0)

    if df_numeric.shape[1] != n_features:
        return None, [f"Invalid CSV: expected {n_features} columns but got {df_numeric.shape[1]}"]

    rows = df_numeric.shape[0]
    if rows > seq_len:
        messages.append(f"Data has {rows} rows; truncating to last {seq_len} rows.")
        df_proc = df_numeric.tail(seq_len)
    elif rows < seq_len:
        pad = seq_len - rows
        messages.append(f"Data has {rows} rows; padding {pad} rows of zeros at the beginning.")
        pad_df = pd.DataFrame(np.zeros((pad, n_features)), columns=df_numeric.columns)
        df_proc = pd.concat([pad_df, df_numeric], ignore_index=True)
    else:
        df_proc = df_numeric

    tensor = torch.tensor(df_proc.to_numpy(dtype=np.float32)).unsqueeze(0)
    return tensor, messages

def safe_predict(model, tensor):
    try:
        model.eval()
        with torch.no_grad():
            out = model(tensor)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out.detach().cpu().squeeze()
            if hasattr(out, "numel") and out.numel() > 1:
                try:
                    logit = out.reshape(-1)[0]
                except Exception:
                    logit = out.reshape(-1).mean()
            else:
                logit = out
            prob = float(torch.sigmoid(logit).item())
            return prob, None
    except Exception as e:
        return None, str(e)

def plot_prediction_histogram(probs_neg, probs_pos, prediction_score):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if len(probs_neg) > 0:
        ax.hist(probs_neg, bins=50, alpha=0.7, label="Actual Low-Risk (Test Set)", density=True)
    if len(probs_pos) > 0:
        ax.hist(probs_pos, bins=50, alpha=0.7, label="Actual High-Risk (Test Set)", density=True)
    ax.axvline(prediction_score, color="red", linestyle="--", lw=3, label=f"Your Prediction ({prediction_score:.2f})")
    ax.set_title("Prediction Score vs. Test Set Outcomes")
    ax.set_xlabel("Predicted Risk Score (Probability)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig

def parse_text_to_df(text: str):
    try:
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip() != ""]
        if len(lines) == 0:
            return None, "No data found in text."
        rows = []
        for ln in lines:
            parts = [p.strip() for p in ln.split(",") if p.strip() != ""]
            rows.append(parts)
        col_counts = set(len(r) for r in rows)
        if len(col_counts) > 1:
            return None, f"Inconsistent number of columns across lines: found counts {sorted(col_counts)}"
        df = pd.DataFrame(rows)
        return df, None
    except Exception as e:
        return None, f"Failed to parse text: {e}"

# ---------- Template generator (NOT cached so it changes each time) ----------
def template_csv_random():
    """
    Generate a SEQ_LEN x N_FEATURES CSV with random plausible values based on FEATURE_SPEC.
    Each call uses a fresh RNG so it produces different values every time.
    """
    rng = np.random.default_rng()  # new random generator (non-deterministic seed)
    rows = []
    for _ in range(SEQ_LEN):
        row = []
        for fname in FEATURE_NAMES:
            default, vmin, vmax, step = FEATURE_SPEC.get(fname, (0.0, -1e6, 1e6, 0.1))
            # integer-like features (step >= 1) or small discrete ranges -> integer draw
            if step >= 1.0 or (vmax - vmin) <= 2 and (step >= 1.0 or fname in ("Gender", "Unit1", "Unit2")):
                lo = int(np.ceil(vmin))
                hi = int(np.floor(max(lo, vmax)))
                if hi < lo:
                    hi = lo
                if fname in ("Gender", "Unit1", "Unit2"):
                    val = int(rng.integers(0, 2))
                else:
                    val = int(rng.integers(lo, hi + 1))
                row.append(val)
            else:
                val = float(rng.uniform(vmin, vmax))
                if step < 1:
                    precision = max(0, int(-np.floor(np.log10(step)))) if step != 0 else 3
                    val = float(round(val, precision))
                else:
                    val = float(round(val, 3))
                row.append(val)
        rows.append(row)
    df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    return df.to_csv(index=False).encode("utf-8")

# ---------- Page ----------
class PredictPage:
    @staticmethod
    def render():
        st.title("📈 Live Prediction (Neonatal Sepsis)")
        st.info(
            f"Provide data via CSV upload, paste CSV text, enter a single row, or fill features manually.\n"
            f"All inputs must represent {N_FEATURES} features. Data will be padded/truncated to {SEQ_LEN} timesteps."
        )

        st.download_button(
            "Download Template CSV (random plausible values)",
            data=template_csv_random(),
            file_name="patient_template_random.csv",
            mime="text/csv"
        )

        model, model_err = load_model(str(MODEL_PATH))
        eval_data = load_eval_data(str(EVAL_FEDERATED_JSON))

        if model is None:
            st.warning(f"Model not available: {model_err or 'Model file missing.'}")
            st.info("Please generate 'server_out/global_best.pt' (train or run federated simulation) to enable predictions.")

        # initialize session state storage for the input dataframe
        if "df_input_temp" not in st.session_state:
            st.session_state.df_input_temp = None

        st.subheader("Input Data Options")
        mode = st.radio("Choose input method:", ("Upload CSV file", "Paste CSV text", "Enter single row (40 values)", "Fill features manually"))

        # ---------- Upload CSV ----------
        if mode == "Upload CSV file":
            uploaded = st.file_uploader("Upload patient CSV (header optional). If header present, it will be used if columns match.", type=["csv"], key="uploader")
            if uploaded is not None:
                try:
                    uploaded.seek(0)
                    df_try = pd.read_csv(uploaded)
                    if df_try.shape[1] == N_FEATURES:
                        df_input = df_try.copy()
                    else:
                        uploaded.seek(0)
                        df_input = pd.read_csv(uploaded, header=None)
                    # ensure columns names if headerless
                    if df_input.shape[1] == N_FEATURES and list(df_input.columns) != FEATURE_NAMES:
                        df_input.columns = FEATURE_NAMES
                    st.session_state.df_input_temp = df_input
                    st.success("CSV loaded and stored for prediction.")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")

        # ---------- Paste CSV Text ----------
        elif mode == "Paste CSV text":
            st.markdown("Paste CSV text below. Each line = one timestep. Values separated by commas.")
            text = st.text_area("Paste CSV text here", height=200, placeholder="val1,val2,... (40 values)\nval1,val2,...\n...")
            if st.button("Parse pasted CSV", key="parse_text"):
                if not text.strip():
                    st.error("No text entered.")
                else:
                    df_parsed, msg = parse_text_to_df(text)
                    if df_parsed is None:
                        st.error(msg)
                    else:
                        # set header if dimensions match
                        if df_parsed.shape[1] == N_FEATURES:
                            df_parsed.columns = FEATURE_NAMES
                        st.session_state.df_input_temp = df_parsed
                        st.success("Pasted CSV parsed and stored for prediction.")

        # ---------- Single row entry ----------
        elif mode == "Enter single row (40 values)":
            st.markdown(f"Enter **{N_FEATURES}** comma-separated feature values for a single timestep; you can tile or pad to {SEQ_LEN}.")
            single_text = st.text_area("Enter single row values (comma-separated)", height=120, placeholder="v1, v2, ..., v40")
            tile_option = st.radio("Expand single row to 48 timesteps by:", ("Tile the same row across all timesteps", "Use this row as the most recent timestep and pad previous rows with zeros"), key="single_tile")
            if st.button("Parse single row", key="parse_single"):
                if not single_text.strip():
                    st.error("No values entered.")
                else:
                    parts = [p.strip() for p in single_text.strip().split(",") if p.strip() != ""]
                    if len(parts) != N_FEATURES:
                        st.error(f"Expected {N_FEATURES} values, but got {len(parts)}.")
                    else:
                        try:
                            row_vals = [float(x) for x in parts]
                        except Exception as e:
                            st.error(f"Failed to parse numbers: {e}")
                        else:
                            if tile_option.startswith("Tile"):
                                rows = np.tile(np.array(row_vals)[None, :], (SEQ_LEN, 1))
                                df_input = pd.DataFrame(rows, columns=FEATURE_NAMES)
                            else:
                                rows = np.zeros((SEQ_LEN, N_FEATURES))
                                rows[-1, :] = np.array(row_vals)
                                df_input = pd.DataFrame(rows, columns=FEATURE_NAMES)
                            st.session_state.df_input_temp = df_input
                            st.success("Single-row input processed and stored for prediction.")

        # ---------- Manual per-feature fill ----------
        else:
            st.markdown("Fill each feature below. Fields are prefilled with plausible defaults and validated by range.")
            cols = st.columns(4)
            manual_vals = {}
            per_col = int(np.ceil(len(FEATURE_NAMES) / 4))
            for col_idx, col in enumerate(cols):
                start = col_idx * per_col
                end = min(start + per_col, len(FEATURE_NAMES))
                for fname in FEATURE_NAMES[start:end]:
                    default, vmin, vmax, step = FEATURE_SPEC.get(fname, (0.0, -1e6, 1e6, 0.1))
                    val = col.number_input(
                        label=fname,
                        value=float(default),
                        min_value=float(vmin),
                        max_value=float(vmax),
                        step=float(step),
                        format="%.3f" if step < 1 else "%.1f",
                        key=f"manual_{fname}"
                    )
                    manual_vals[fname] = val
            manual_tile = st.radio("Expand manual row to 48 timesteps by:", ("Tile the same row across all timesteps", "Use this row as the most recent timestep and pad previous timesteps with zeros"), key="manual_tile")
            if st.button("Use manual inputs", key="use_manual"):
                parsed = [manual_vals[f] for f in FEATURE_NAMES]
                if manual_tile.startswith("Tile"):
                    rows = np.tile(np.array(parsed)[None, :], (SEQ_LEN, 1))
                    df_input = pd.DataFrame(rows, columns=FEATURE_NAMES)
                else:
                    rows = np.zeros((SEQ_LEN, N_FEATURES))
                    rows[-1, :] = np.array(parsed)
                    df_input = pd.DataFrame(rows, columns=FEATURE_NAMES)
                st.session_state.df_input_temp = df_input
                st.success("Manual inputs stored for prediction.")

        # ---------- If we have stored input, preview and allow prediction ----------
        if st.session_state.df_input_temp is not None:
            st.markdown("---")
            st.subheader("Preview of Input Data (first 5 rows)")
            try:
                df_preview = st.session_state.df_input_temp.copy()
                if df_preview.shape[1] == N_FEATURES and list(df_preview.columns) != FEATURE_NAMES:
                    df_preview.columns = FEATURE_NAMES
                df_preview = df_preview.apply(pd.to_numeric, errors="coerce")
            except Exception:
                df_preview = st.session_state.df_input_temp
            st.dataframe(df_preview.head())

            tensor, messages = preprocess_dataframe(df_preview, SEQ_LEN, N_FEATURES)
            for m in messages:
                st.info(m)

            if tensor is None:
                st.error("Preprocessing failed. Fix input and try again.")
            else:
                threshold = st.slider("Decision threshold (affects recommendation & threshold line)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
                if st.button("▶️ Run Prediction", key="run_prediction"):
                    if model is None:
                        st.error("Model not available — cannot run prediction.")
                        return
                    with st.spinner("Running model..."):
                        prob, err = safe_predict(model, tensor)
                        if err:
                            st.error(f"Prediction error: {err}")
                            return
                        st.subheader("Prediction Result")
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=prob * 100,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Sepsis Risk (%)", 'font': {'size': 20}},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 25], 'color': 'rgba(44,160,44,0.7)'},
                                        {'range': [25, 50], 'color': 'rgba(255,127,14,0.7)'},
                                        {'range': [50, 100], 'color': 'rgba(214,39,40,0.7)'}
                                    ],
                                    'threshold': {'line': {'color': "red", 'width': 4}, 'value': threshold * 100}
                                }
                            ))
                            fig_gauge.update_layout(height=300, margin=dict(t=30, b=10))
                            st.plotly_chart(fig_gauge, use_container_width=True)
                        with col2:
                            if prob > threshold:
                                st.error("HIGH RISK: Sepsis likely.", icon="⚠️")
                                st.markdown("""
                                **Clinical Recommendations:**
                                - Immediate evaluation by clinician.
                                - Consider blood cultures and early antibiotics per local protocol.
                                - Monitor vitals and inflammatory markers (CRP, PCT).
                                """)
                            elif prob > (threshold * 0.5):
                                st.warning("Moderate Risk: Increased vigilance suggested.", icon="ℹ️")
                                st.markdown("""
                                **Recommendations:**
                                - Increase monitoring frequency.
                                - Low threshold for labs or escalation.
                                """)
                            else:
                                st.success("Low Risk: Continue routine monitoring.", icon="✅")
                                st.markdown("**Recommendations:** Continue standard monitoring and follow hospital protocols.")
                        st.subheader("Prediction Analysis vs Test Set")
                        if eval_data:
                            arr_probs = np.array(eval_data['y_prob'])
                            arr_true = np.array(eval_data['y_true'])
                            probs_neg = arr_probs[arr_true == 0]
                            probs_pos = arr_probs[arr_true == 1]
                            hist_fig = plot_prediction_histogram(probs_neg, probs_pos, prob)
                            st.pyplot(hist_fig)
                        else:
                            st.info(f"Evaluation file '{EVAL_FEDERATED_JSON}' not found — cannot display analysis histogram.")
