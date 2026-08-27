"""Reusable Baseline Evaluation & Diagnostics Script for AI Risk Manager.

Loads the trained baseline model from models/baseline_model.joblib,
evaluates on the Validation set (data/processed/val.csv),
and generates the 7 required diagnostic figures in reports/figures/:
1. Confusion Matrix Heatmap
2. Precision-Recall Curve
3. ROC Curve
4. Threshold vs. Precision Curve
5. Threshold vs. Recall Curve
6. Threshold vs. F1-Score Curve
7. Threshold vs. Estimated Financial Risk Cost Curve
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
)
import joblib

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.train_baseline import calculate_financial_cost

FIGURES_DIR = ROOT_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_baseline_model(
    val_data_path: Path = PROCESSED_DATA_DIR / "val.csv",
    model_path: Path = MODELS_DIR / "baseline_model.joblib",
    preprocessor_path: Path = MODELS_DIR / "preprocessing_pipeline.pkl",
):
    print("=" * 75)
    print("[EVALUATION] EXECUTING BASELINE MODEL VALIDATION EVALUATION")
    print(f"Validation Data: {val_data_path}")
    print(f"Model Artifact:  {model_path}")
    print("=" * 75)

    if not model_path.exists():
        raise FileNotFoundError(f"Baseline model artifact not found at {model_path}. Run src/train_baseline.py first.")

    model_payload = joblib.load(model_path)
    model = model_payload["model"]
    model_name = model_payload.get("model_type", "BaselineModel")
    requires_scaling = model_payload.get("requires_scaling", False)
    scaler = model_payload.get("scaler", None)
    optimal_thresh = model_payload.get("optimal_threshold", 0.50)

    # Load preprocessor
    extractor = joblib.load(preprocessor_path)

    # Load validation data
    val_df = pd.read_csv(val_data_path)
    y_val = val_df["is_fraud"].values
    amounts_val = val_df["amt"].values

    # Feature extraction
    X_val = extractor.transform(val_df)
    if requires_scaling and scaler is not None:
        X_val_input = scaler.transform(X_val)
    else:
        X_val_input = X_val

    # Predict probabilities
    p_val = model.predict_proba(X_val_input)[:, 1]

    # Metrics at optimal threshold
    y_pred_opt = (p_val >= optimal_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_opt).ravel()

    prec = precision_score(y_val, y_pred_opt, zero_division=0)
    rec = recall_score(y_val, y_pred_opt, zero_division=0)
    f1 = f1_score(y_val, y_pred_opt, zero_division=0)

    # Curves and AUC
    prec_pts, rec_pts, _ = precision_recall_curve(y_val, p_val)
    pr_auc = auc(rec_pts, prec_pts)
    fpr_pts, tpr_pts, _ = roc_curve(y_val, p_val)
    roc_auc = roc_auc_score(y_val, p_val)

    costs = calculate_financial_cost(y_val, y_pred_opt, amounts_val)

    print("\n[1] Validation Metrics Summary:")
    print(f"    * Model:                  {model_name}")
    print(f"    * Optimal Threshold:      {optimal_thresh:.2f}")
    print(f"    * Precision:              {prec:.4f}")
    print(f"    * Recall (Coverage):      {rec:.4f}")
    print(f"    * F1-Score:               {f1:.4f}")
    print(f"    * PR-AUC:                 {pr_auc:.4f}")
    print(f"    * ROC-AUC:                {roc_auc:.4f}")
    print(f"    * Confusion Matrix:       TP={tp} | FP={fp} | FN={fn} | TN={tn}")
    print(f"    * Estimated Total Cost:   ${costs['total_cost']:,.2f}")
    print(f"    * Net Protected Capital:  ${costs['net_savings']:,.2f}")

    # Threshold Sweep
    thresholds = np.linspace(0.05, 0.95, 37)
    prec_list, rec_list, f1_list, cost_list = [], [], [], []

    for t in thresholds:
        yp = (p_val >= t).astype(int)
        prec_list.append(precision_score(y_val, yp, zero_division=0))
        rec_list.append(recall_score(y_val, yp, zero_division=0))
        f1_list.append(f1_score(y_val, yp, zero_division=0))
        c = calculate_financial_cost(y_val, yp, amounts_val)
        cost_list.append(c["total_cost"])

    # -------------------------------------------------------------
    # GENERATE 7 DIAGNOSTIC FIGURES
    # -------------------------------------------------------------
    print("\n[2] Generating 7 Diagnostic Charts in reports/figures/...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {"navy": "#0C2340", "blue": "#3395FF", "green": "#00C48C", "red": "#FF4D4F", "amber": "#FAAD14"}

    # 1. Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_matrix = [[tn, fp], [fn, tp]]
    cax = ax.imshow(cm_matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Legit (0)", "Predicted Fraud (1)"], fontweight="bold")
    ax.set_yticklabels(["Actual Legit (0)", "Actual Fraud (1)"], fontweight="bold")
    for i in range(2):
        for j in range(2):
            val = cm_matrix[i][j]
            ax.text(j, i, f"{val:,}", ha="center", va="center", color="white" if val > (tn/2) else "black", fontsize=12, fontweight="bold")
    ax.set_title(f"Baseline Confusion Matrix (Threshold = {optimal_thresh:.2f})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p1 = FIGURES_DIR / "baseline_1_confusion_matrix.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p1.name}")

    # 2. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec_pts, prec_pts, color=colors["blue"], linewidth=2.5, label=f"{model_name} (PR-AUC = {pr_auc:.4f})")
    ax.axhline(y=np.mean(y_val), color="gray", linestyle="--", label=f"No-Skill Baseline ({np.mean(y_val)*100:.1f}%)")
    ax.set_title("Precision-Recall Curve (Validation Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall (Detection Rate)")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    p2 = FIGURES_DIR / "baseline_2_precision_recall_curve.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p2.name}")

    # 3. ROC Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_pts, tpr_pts, color=colors["green"], linewidth=2.5, label=f"{model_name} (ROC-AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Chance (AUC = 0.50)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    p3 = FIGURES_DIR / "baseline_3_roc_curve.png"
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p3.name}")

    # 4. Threshold vs. Precision
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, prec_list, color=colors["blue"], linewidth=2.5, marker="s")
    ax.axvline(x=optimal_thresh, color=colors["amber"], linestyle="--", label=f"Selected Threshold ({optimal_thresh:.2f})")
    ax.set_title("Decision Threshold vs. Precision", fontsize=12, fontweight="bold")
    ax.set_xlabel("Decision Threshold (Probability Cutoff)")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    p4 = FIGURES_DIR / "baseline_4_threshold_vs_precision.png"
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p4.name}")

    # 5. Threshold vs. Recall
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, rec_list, color=colors["green"], linewidth=2.5, marker="o")
    ax.axvline(x=optimal_thresh, color=colors["amber"], linestyle="--", label=f"Selected Threshold ({optimal_thresh:.2f})")
    ax.set_title("Decision Threshold vs. Recall (Fraud Catch Rate)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Decision Threshold (Probability Cutoff)")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    p5 = FIGURES_DIR / "baseline_5_threshold_vs_recall.png"
    fig.savefig(p5, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p5.name}")

    # 6. Threshold vs. F1-Score
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, f1_list, color=colors["navy"], linewidth=2.5, marker="^")
    ax.axvline(x=optimal_thresh, color=colors["amber"], linestyle="--", label=f"Selected Threshold ({optimal_thresh:.2f})")
    ax.set_title("Decision Threshold vs. F1-Score", fontsize=12, fontweight="bold")
    ax.set_xlabel("Decision Threshold (Probability Cutoff)")
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    p6 = FIGURES_DIR / "baseline_6_threshold_vs_f1.png"
    fig.savefig(p6, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p6.name}")

    # 7. Threshold vs. Estimated Financial Risk Cost
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(thresholds, cost_list, color=colors["red"], linewidth=2.5, marker="d")
    ax.axvline(x=optimal_thresh, color=colors["green"], linestyle="--", label=f"Cost-Optimal Threshold ({optimal_thresh:.2f})")
    ax.set_title("Estimated Financial Risk Cost vs. Decision Threshold", fontsize=12, fontweight="bold")
    ax.set_xlabel("Decision Threshold (Probability Cutoff)")
    ax.set_ylabel("Total Estimated Cost ($)")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    p7 = FIGURES_DIR / "baseline_7_threshold_vs_cost.png"
    fig.savefig(p7, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p7.name}")

    print("\n" + "=" * 75)
    print("[COMPLETE] BASELINE EVALUATION & VISUALIZATIONS FINISHED")
    print("=" * 75)


if __name__ == "__main__":
    evaluate_baseline_model()

