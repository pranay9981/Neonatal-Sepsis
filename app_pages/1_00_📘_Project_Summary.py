# 1_00_📘_Project_Summary.py
import streamlit as st

class ProjectSummaryPage:
    @staticmethod
    def render():
        st.set_page_config(page_title="Project Summary — Neonatal Sepsis Detection", layout="wide")
        st.title("📘 Neonatal Sepsis Detection Using Federated Learning")
        st.markdown("---")

        st.markdown(
            """
            ### Executive summary
            Neonatal sepsis is an acute, high-risk condition where early recognition materially improves outcomes. Clinical
            data are noisy, sparsely observed and, importantly, siloed across hospitals due to privacy and regulatory
            constraints. This project demonstrates a privacy-first, production-minded pipeline that uses **Federated Learning (FL)**
            to collaboratively train a robust model for neonatal sepsis detection while keeping raw patient data local to each site.
            """
        )

        st.markdown("### Dataset at a glance")
        st.markdown(
            """
            - **Scale:** Training and evaluation use a large collection of patient-window files (your dataset includes **>40,000** such files).  
            - **Format:** Individual patient windows are stored as **pipe-separated values (PSV)** with a header row describing the columns. Each file represents one time-series window for a single patient.  
            - **Window & features:** Each instance represents a fixed-length time window with **48 timesteps** and **40 clinical features** per timestep. The model ingests this as a tensor of shape **(1, 48, 40)** for inference.  
            - **Target label:** Each patient-window file includes a binary target `SepsisLabel` (0 = no sepsis, 1 = sepsis) used for supervised training.
            """
        )

        st.markdown("### Feature set (high level)")
        st.markdown(
            """
            The 40 features combine routinely measured **vital signs**, **laboratory measurements**, **hematology/coagulation markers**, and **clinical metadata**.  
            Categories include:
            - **Vital signs:** HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2  
            - **Blood gas / electrolytes:** pH, PaCO2, HCO3, BaseExcess, SaO2, Chloride, Potassium, Calcium, Magnesium, Phosphate  
            - **Metabolic / organ markers:** Glucose, Lactate, BUN, Creatinine, AST, Alkalinephos, Bilirubin (direct/total), TroponinI  
            - **Hematology & coagulation:** Hct, Hgb, WBC, Platelets, Fibrinogen, PTT  
            - **Metadata & provenance:** Age (days), Gender (binary), Unit flags (Unit1/Unit2), HospAdmTime (hours since admission), ICULOS (hours in ICU)
            """
        )

        st.markdown("### Observed dataset characteristics")
        st.markdown(
            """
            Based on the dataset structure and representative files:
            - **Measurement density is heterogeneous.** Vital signs (HR, O2Sat, Resp, BP) are observed frequently, while many laboratory tests are sparse or intermittently recorded.  
            - **High missingness for certain labs.** Laboratory and specialty markers often contain large blocks of missing values which inform our imputation strategy.  
            - **Temporal anchors present.** Fields such as `ICULOS` and `HospAdmTime` provide useful timing context (time since ICU admission or hospital admission) that supplement the raw vitals/labs.  
            - **Label imbalance.** Sepsis events are relatively rare compared to non-sepsis windows — this motivates focusing on metrics like AUPRC and recall for evaluation and tuning.
            """
        )

        st.markdown("### Preprocessing pipeline (concise & reproducible)")
        st.markdown(
            """
            1. **Parsing & shape validation** — read PSV with `sep='|'`, require 40 feature columns. Windows shorter than 48 timesteps are left-padded with zeros to preserve the most recent observations; windows longer than 48 are truncated to the most recent 48 timesteps.  
            2. **Type coercion & missing values** — coerce to numeric; during inference the dashboard fills missing or non-numeric cells with `0` (training uses a consistent imputation policy or learned imputers). Warnings are surfaced when non-numeric values are present.  
            3. **Normalization** — fit a scaler (feature-wise standardization or robust scaling) on the training distribution and persist it; apply the same transform at inference to ensure consistent input ranges for the model.  
            4. **Tensor formatting** — convert the processed window to a float32 `torch.tensor` and add a batch dimension: `tensor.unsqueeze(0)` → shape `(1, 48, 40)`. The model outputs a single logit which is passed through `sigmoid` to yield a probability score.
            """
        )

        st.markdown("### Modeling & federated setup")
        st.markdown(
            """
            - **Architecture:** a *TimeSeriesTransformer* implemented in PyTorch. Transformers are chosen because they (a) capture temporal dependencies over medium-length windows, (b) attend across features to model cross-signal interactions, and (c) scale to multivariate clinical time series without heavy feature engineering.  
            - **Output:** single scalar logit → `sigmoid(logit)` gives the sepsis probability (0–1).  
            - **Federation:** training is simulated via **Flower (flwr)**. The server coordinates local training on each client and aggregates model updates to produce a global model — no patient-level data are shared. In our evaluation we simulate **2 training clients** and retain a held-out client for final testing to measure cross-site generalization.
            """
        )

        st.markdown("### Evaluation methodology")
        st.markdown(
            """
            - **Metrics reported:** AUROC, AUPRC, Accuracy, Precision, Recall, and F1. Because sepsis is a relatively rare outcome, **AUPRC** and **recall/F1 at clinically-relevant thresholds** are emphasized.  
            - **Audit artifacts:** confusion matrices, ROC/PRC curves and raw JSON evaluation files are available in the dashboard for transparent auditing.  
            - **Generalization test:** performance is measured on a held-out client to estimate out-of-sample generalization across hospital practices and measurement patterns.
            """
        )

        st.markdown("### Clinical integration & dashboard features")
        st.markdown(
            """
            - **Live Prediction:** the Predict page accepts PSV/CSV uploads, pasted CSV text, a one-row snapshot, or manual per-feature entry (40 fields). The UI pads/truncates to 48 timesteps and runs inference with the saved global model.  
            - **Decision support:** predictions are presented as a probability gauge; thresholded guidance (low / moderate / high) and a visualization showing how a patient's score compares to the test-set distribution are provided to support clinical interpretation.  
            - **Auditability & reproducibility:** evaluation JSONs, confusion matrices, and ROC/PRC plots are exposed for reviewers and auditors in the Model Metrics page.
            """
        )

        st.markdown("### Data governance, safety & deployment considerations")
        st.markdown(
            """
            - **Privacy-first design:** Federated Learning keeps patient data on-site; only model updates are exchanged. If required, secure aggregation and differential privacy mechanisms may be integrated to strengthen privacy guarantees.  
            - **Clinical safety:** the system is a *decision support* tool and does not replace clinician judgment — any high-risk prediction should trigger clinical evaluation. Prospective clinical validation is required prior to any live clinical deployment.  
            - **Production readiness:** for deployment add monitoring for data drift, model calibration checks, and human-in-the-loop workflows for flagged cases.
            """
        )

        st.markdown("### What reviewers and mentors should look for")
        st.markdown(
            """
            1. **Reproduce predictions**: use the Predict page (manual-entry mode is useful for edge-case testing).  
            2. **Audit evaluation**: inspect ROC/PRC curves, confusion matrices, and raw evaluation JSON files. Focus on recall and AUPRC for rare-event detection.  
            3. **Assess generalization**: review held-out client performance to verify that the federated model improves cross-site robustness versus a single-site model.  
            4. **Request additional artifacts**: if needed, reviewers can ask for de-identified sample windows, the fitted scaler, or training logs for deeper verification.
            """
        )

        st.markdown("---")
        st.info("Tip: After reading this summary, try the Predict page to run a live inference and then visit Model Metrics to inspect evaluation artifacts and model behavior.")
