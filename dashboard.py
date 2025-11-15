import streamlit as st
import torch
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import json

# Add the 'src' directory to the Python path
# This is crucial so we can import 'model.py'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    # Try to import the model class
    from model import TimeSeriesTransformer
except ImportError:
    st.error("FATAL ERROR: Could not find 'src/model.py'. Make sure this dashboard is run from the project's root directory.")
    st.stop()
except ModuleNotFoundError:
    st.error("FATAL ERROR: A core library (like 'torch' or 'matplotlib') is not installed in this environment.")
    st.info("Please activate your project's virtual environment (`venv\\Scripts\\activate`) and then run `streamlit run dashboard.py`")
    st.stop()


# --- Model & App Constants ---

# These must match the model you trained
MODEL_PATH = "server_out/global_best.pt"
N_FEATURES = 40
SEQ_LEN = 48

# --- NEW: Path to the evaluation results ---
# This file MUST exist for the plot to work.
EVAL_JSON_PATH = "eval_results_federated.json"

# Paths to your comparison plots
PLOT_ROC_PATH = "model_comparison_plot.png"
PLOT_PRC_PATH = "model_comparison_plot_prc.png"

# --- Page Configuration ---
st.set_page_config(
    page_title="Neonatal Sepsis Prediction",
    page_icon="👶",
    layout="wide"
)

# --- Model Loading ---
@st.cache_resource
def load_model(model_path, n_features, seq_len):
    """
    Loads the trained PyTorch model.
    @st.cache_resource ensures this only runs once.
    """
    if not os.path.exists(model_path):
        st.error(f"FATAL ERROR: Model file not found at '{model_path}'.")
        st.info("Please run the federated learning simulation first (`python src/fl_server.py ...`) to generate this file.")
        st.stop()
        
    # Initialize model architecture
    model = TimeSeriesTransformer(n_features=n_features, seq_len=seq_len)
    
    # Load the saved state dictionary
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except RuntimeError as e:
        st.error(f"Error loading model weights: {e}")
        st.info("This usually means the model_best.pt file does not match the model architecture (e.g., wrong N_FEATURES or SEQ_LEN).")
        st.stop()
        
    model.eval()  # Set model to evaluation mode
    return model

# --- NEW: Load Test Set Predictions ---
@st.cache_data
def load_eval_data(json_path):
    """
    Loads the y_true and y_prob from the federated evaluation JSON file.
    """
    if not os.path.exists(json_path):
        st.warning(f"Evaluation file not found: '{json_path}'. Cannot display prediction histogram.")
        st.info("Please run `python src/evaluate.py ... --out_file eval_results_federated.json` to generate this file.")
        return None, None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        y_true = np.array(data['y_true'])
        y_prob = np.array(data['y_prob'])
        
        # Separate probabilities based on the true label
        probs_negative = y_prob[y_true == 0]
        probs_positive = y_prob[y_true == 1]
        
        return probs_negative, probs_positive
    except Exception as e:
        st.error(f"Error loading evaluation data: {e}")
        return None, None

# --- NEW: Plotting Function ---
def plot_prediction_histogram(probs_neg, probs_pos, prediction_score):
    """
    Plots the prediction score on a histogram of the test set scores.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot the two histograms
    ax.hist(probs_neg, bins=50, alpha=0.7, label='Actual Low-Risk (Test Set)', color='blue', density=True)
    ax.hist(probs_pos, bins=50, alpha=0.7, label='Actual High-Risk (Test Set)', color='orange', density=True)
    
    # Plot the prediction line
    ax.axvline(x=prediction_score, color='red', linestyle='--', lw=3, 
                label=f'Your Prediction ({prediction_score:.2f})')
    
    # Style the plot
    ax.set_title('Prediction Score vs. Test Set Outcomes', fontsize=14)
    ax.set_xlabel('Predicted Risk Score (Probability)', fontsize=12)
    ax.set_ylabel('Density of Patients', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig)

# --- Data Processing Function ---
@st.cache_data
def preprocess_data(df, seq_len, n_features):
    """
    Validates and preprocesses the uploaded DataFrame.
    Returns a 3D tensor for the model.
    """
    try:
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        st.error(f"Error converting data to numeric: {e}")
        return None, "Failed to convert columns to numbers."

    if df_numeric.isnull().values.any():
        st.warning("Non-numeric values detected in CSV. They will be treated as 0.")
        df_numeric = df_numeric.fillna(0)

    if df_numeric.shape[1] != n_features:
        st.error(f"Invalid CSV: Incorrect number of columns. Expected {n_features}, but got {df_numeric.shape[1]}.")
        return None, f"Expected {n_features} features (columns)."

    current_rows = df_numeric.shape[0]
    
    if current_rows > seq_len:
        st.warning(f"Data has {current_rows} rows. Truncating to the **last {seq_len} rows** (most recent timesteps).")
        df_processed = df_numeric.tail(seq_len)
    elif current_rows < seq_len:
        st.warning(f"Data has {current_rows} rows. Padding with {seq_len - current_rows} rows of zeros at the **beginning**.")
        num_missing_rows = seq_len - current_rows
        padding = pd.DataFrame(np.zeros((num_missing_rows, n_features)), columns=df_numeric.columns)
        df_processed = pd.concat([padding, df_numeric], ignore_index=True)
    else:
        df_processed = df_numeric
    
    df_processed = df_processed.fillna(0)
    data_np = df_processed.to_numpy(dtype=np.float32)
    data_tensor = torch.tensor(data_np).unsqueeze(0)  # Add batch dimension -> (1, 48, 40)
    
    return data_tensor, "Data processed successfully."

# --- Prediction Function ---
def make_prediction(model, tensor_data):
    """
    Runs the model and returns the sepsis risk probability.
    """
    with torch.no_grad():
        logits = model(tensor_data)
        probability = torch.sigmoid(logits).item()
    return probability

# --- Main App UI ---
st.title("👶 Neonatal Sepsis Prediction Dashboard")
st.write(f"Using a federated `Transformer` model trained to predict sepsis risk from time-series data ({SEQ_LEN} timesteps, {N_FEATURES} features).")

# Load the global federated model
model = load_model(MODEL_PATH, N_FEATURES, SEQ_LEN)
# --- NEW: Load eval data for plotting ---
probs_neg, probs_pos = load_eval_data(EVAL_JSON_PATH)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Live Prediction", "Model Performance"])

if page == "Live Prediction":
    st.header("Sepsis Risk Prediction")
    
    st.info(f"Upload a patient's time-series data as a CSV file. The file must have **{N_FEATURES} columns** (features) with no header. The dashboard will automatically handle any number of rows by padding or truncating to the required **{SEQ_LEN} timesteps**.")
    
    @st.cache_data
    def get_template_csv():
        template_df = pd.DataFrame(np.round(np.random.rand(SEQ_LEN, N_FEATURES), 2))
        return template_df.to_csv(header=False, index=False).encode('utf-8')

    st.download_button(
        label=f"Download Template CSV ({SEQ_LEN} rows x {N_FEATURES} cols)",
        data=get_template_csv(),
        file_name="patient_template.csv",
        mime="text/csv",
    )
    
    uploaded_file = st.file_uploader("Upload patient CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            with st.expander("Show Uploaded Data (first 5 rows)"):
                st.dataframe(df.head())
            
            tensor_data, message = preprocess_data(df, SEQ_LEN, N_FEATURES)
            
            if tensor_data is not None:
                if st.button("Calculate Sepsis Risk"):
                    with st.spinner("Running model..."):
                        probability = make_prediction(model, tensor_data)
                        
                        st.subheader("Prediction Result")
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.metric("Sepsis Risk Score", f"{probability * 100:.2f}%")
                            st.progress(probability)
                            if probability > 0.5:
                                st.error("HIGH RISK: Sepsis likely. Immediate clinical review recommended.", icon="⚠️")
                            elif probability > 0.25:
                                st.warning("Moderate Risk: Continue close monitoring.", icon="ℹ️")
                            else:
                                st.success("Low Risk: Sepsis unlikely. Continue standard monitoring.", icon="✅")
                        with col2:
                            st.write("") 
                        
                        # --- NEW: Show the plot ---
                        st.subheader("Prediction Analysis")
                        st.markdown("""
                        This plot shows where your new prediction (red line) falls in relation to all the predictions from our held-out test set.
                        * **Blue:** Distribution of scores for *actual* low-risk patients.
                        * **Orange:** Distribution of scores for *actual* high-risk (septic) patients.
                        """)
                        
                        if probs_neg is not None and probs_pos is not None:
                            plot_prediction_histogram(probs_neg, probs_pos, probability)
                        else:
                            st.info("Cannot display analysis plot because the test set evaluation file is missing.")
            
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

else: # page == "Model Performance"
    st.header("Federated Model Performance")
    
    st.markdown("""
    This page shows the performance of our **Federated Model** (trained on data from multiple clients)
    compared to a **Local Model** (trained on data from only one client).
    
    The models were evaluated on an unseen test set (`client3`) to measure their **generalizability**.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROC Curve (AUROC)")
        if os.path.exists(PLOT_ROC_PATH):
            st.image(PLOT_ROC_PATH)
            st.markdown("""
            **Interpretation:**
            * **AUROC** measures the model's ability to distinguish between "septic" and "non-septic" patients. A score of 1.0 is perfect, and 0.5 is random chance.
            * Our **Federated Model** is significantly better, proving it learned more robust patterns from the diverse data.
            * The shaded areas represent the 95% confidence interval. The lack of overlap shows our federated model's superiority is statistically significant.
            """)
        else:
            st.warning(f"Plot not found: '{PLOT_ROC_PATH}'. Please run `python src/plot_results.py ...` first.")

    with col2:
        st.subheader("Precision-Recall Curve (AUPRC)")
        prc_plot_path = PLOT_ROC_PATH.replace('.png', '_prc.png').replace('.jpg', '_prc.jpg')
        if os.path.exists(prc_plot_path):
            st.image(prc_plot_path)
            st.markdown("""
            **Interpretation:**
            * **AUPRC** is a better metric for imbalanced data (like sepsis). It answers: "Of all the high-risk alerts, what percentage are correct?"
            * The **"Chance" line (dotted black)** shows the baseline (the percentage of positive sepsis cases in the test set).
            * Our **Federated Model** curve is much higher than both the Local Model and the Chance line, proving it provides real, actionable clinical value.
            """)
        else:
            st.warning(f"Plot not found: '{prc_plot_path}'. Please run `python src/plot_results.py ...` first.")