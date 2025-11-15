# src/plot_results.py
"""
Generates comparison ROC and PRC (AUPRC) curve plots from one or more
evaluation JSON files. Includes 95% confidence intervals (CI) for scores
and plots shaded CI bands for the curves themselves.

This script implements the "stepwise" (non-smoothed) curves as recommended
for academic publication.

Usage:
  python src/plot_results.py --results eval_results_federated.json eval_results_local.json --out_file model_comparison.png
"""
import argparse
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.utils import resample
import numpy as np
import os
try:
    from scipy.interpolate import interp1d
except ImportError:
    print("[ERROR] Scipy not found. This script requires Scipy for CI band interpolation.")
    print("Please add 'scipy' to requirements.txt and run 'pip install -r requirements.txt'")
    interp1d = None

def get_bootstrapped_ci(y_true, y_prob, n_bootstraps=1000):
    """
    Calculates 95% CI for AUROC and AUPRC scores, and the CI bands for the curves.
    """
    if interp1d is None:
        raise ImportError("Scipy.interpolate.interp1d not found.")
        
    rng = np.random.RandomState(42)  # for reproducibility
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    aurocs = []
    auprcs = []
    
    # Base axes for interpolation
    base_fpr = np.linspace(0, 1, 101)
    base_recall = np.linspace(0, 1, 101)
    
    tpr_bootstraps = []
    precision_bootstraps = []

    for _ in range(n_bootstraps):
        try:
            indices = rng.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[indices])) < 2:
                continue

            # Calculate scores
            aurocs.append(roc_auc_score(y_true[indices], y_prob[indices]))
            auprcs.append(average_precision_score(y_true[indices], y_prob[indices]))

            # Calculate curves and interpolate for CI bands
            # ROC
            fpr, tpr, _ = roc_curve(y_true[indices], y_prob[indices])
            interp_tpr = interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))(base_fpr)
            tpr_bootstraps.append(interp_tpr)

            # PRC
            precision, recall, _ = precision_recall_curve(y_true[indices], y_prob[indices])
            # Sort by recall (x-axis)
            idx = np.argsort(recall)
            interp_precision = interp1d(recall[idx], precision[idx], kind='linear', bounds_error=False, fill_value='extrapolate')(base_recall)
            precision_bootstraps.append(interp_precision)

        except Exception:
            continue

    # --- Score CI ---
    if not aurocs or not auprcs:
        return None # Not enough valid bootstrap samples

    auc_low, auc_high = np.percentile(aurocs, [2.5, 97.5])
    prc_low, prc_high = np.percentile(auprcs, [2.5, 97.5])

    # --- Curve CI Bands ---
    tpr_bootstraps = np.array(tpr_bootstraps)
    tpr_mean = np.mean(tpr_bootstraps, axis=0)
    tpr_low = np.percentile(tpr_bootstraps, 2.5, axis=0)
    tpr_high = np.percentile(tpr_bootstraps, 97.5, axis=0)

    precision_bootstraps = np.array(precision_bootstraps)
    precision_mean = np.mean(precision_bootstraps, axis=0)
    precision_low = np.percentile(precision_bootstraps, 2.5, axis=0)
    precision_high = np.percentile(precision_bootstraps, 97.5, axis=0)

    return {
        "roc_score_ci": (auc_low, auc_high),
        "prc_score_ci": (prc_low, prc_high),
        "roc_curve_ci": (base_fpr, tpr_low, tpr_high),
        "prc_curve_ci": (base_recall, precision_low, precision_high)
    }


def plot_curves(results_files, out_file_base, plot_type='roc'):
    """
    Generates either an ROC or PRC plot based on the plot_type.
    """
    plt.figure(figsize=(10, 8))
    
    for filepath in results_files:
        if not os.path.exists(filepath):
            print(f"[WARN] File not found, skipping: {filepath}")
            continue
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        y_true = np.array(data.get('y_true'))
        y_prob = np.array(data.get('y_prob'))
        model_name = data.get('model_name', os.path.basename(filepath)).replace('_', ' ').title()
        
        if y_true is None or y_prob is None:
            print(f"[WARN] File {filepath} is missing 'y_true' or 'y_prob' data. Skipping.")
            continue
            
        # Get bootstrapped CIs for scores and bands
        ci_data = get_bootstrapped_ci(y_true, y_prob)
        if ci_data is None:
            print(f"[WARN] Could not generate CI for {model_name}, skipping CI plot.")
            continue
            
        if plot_type == 'roc':
            # Plot the actual, non-smoothed curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            score = roc_auc_score(y_true, y_prob)
            ci_low, ci_high = ci_data['roc_score_ci']
            score_name = "AUROC"
            
            # Plot the mean curve (stepwise)
            plt.plot(fpr, tpr, drawstyle='steps-post', lw=2, 
                     label=f'{model_name}\n({score_name} = {score:.3f} [95% CI: {ci_low:.3f} - {ci_high:.3f}])')
            
            # Plot the shaded CI band
            base_fpr, tpr_low, tpr_high = ci_data['roc_curve_ci']
            plt.fill_between(base_fpr, tpr_low, tpr_high, alpha=0.2)

        elif plot_type == 'prc':
            # Plot the actual, non-smoothed curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            score = average_precision_score(y_true, y_prob)
            ci_low, ci_high = ci_data['prc_score_ci']
            score_name = "AUPRC"

            # Plot the mean curve (stepwise)
            plt.plot(recall, precision, drawstyle='steps-post', lw=2,
                     label=f'{model_name}\n({score_name} = {score:.3f} [95% CI: {ci_low:.3f} - {ci_high:.3f}])')
            
            # Plot the shaded CI band
            base_recall, precision_low, precision_high = ci_data['prc_curve_ci']
            plt.fill_between(base_recall, precision_low, precision_high, alpha=0.2)

    if plot_type == 'roc':
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance (AUROC = 0.500)')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Model Comparison - ROC Curve', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
    
    elif plot_type == 'prc':
        baseline = np.sum(y_true) / len(y_true) if len(y_true) > 0 else 0
        plt.axhline(baseline, ls='--', color='k', lw=2, label=f'Chance (AUPRC = {baseline:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Model Comparison - Precision-Recall Curve', fontsize=14)
        plt.legend(loc='lower left', fontsize=10) # Moved legend

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    try:
        filename = out_file_base
        if plot_type == 'prc':
            base, ext = os.path.splitext(out_file_base)
            filename = f"{base}_prc{ext}"
        
        plt.savefig(filename)
        print(f"[PLOT] Saved {plot_type.upper()} plot to: {filename}")
    except Exception as e:
        print(f"[PLOT][ERROR] Failed to save plot: {e}")
    plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs='+', required=True, help="List of one or more evaluation JSON files (e.g., eval1.json eval2.json)")
    ap.add_argument("--out_file", type=str, default="model_comparison_plot.png", help="Path to save the output plot. '_prc' will be added for the PR curve.")
    args = ap.parse_args()
    
    if interp1d is None:
        print("[FATAL] 'scipy' is required but not found. Please install it with 'pip install scipy' or add it to requirements.txt.")
        exit(1)
    
    # Generate both plots
    plot_curves(args.results, args.out_file, plot_type='roc')
    plot_curves(args.results, args.out_file, plot_type='prc')