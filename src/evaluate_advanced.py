"""Advanced Model Evaluation & Diagnostic Visualization Script.
Razorpay AI Buildathon — AI Risk Manager Track.

Loads the trained advanced model from models/advanced_model.joblib,
evaluates on the Validation partition (data/processed/val.csv),
and generates the 7 required diagnostic figures in reports/figures/:
1. Confusion Matrix Heatmap
2. Precision-Recall Curve
3. ROC Curve
4. Multi-Metric Threshold Curve (Precision, Recall, F1)
5. Financial Risk Cost vs. Decision Threshold Curve
6. Model-Based Feature Importance Ranking Chart
7. Train vs. Validation Overfitting Bar Chart
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
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
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_advanced_model(
    val_data_path: Path = PROCESSED_DATA_DIR / "val.csv",
    model_path: Path = MODELS_DIR / "advanced_model.joblib",
    preprocessor_path: Path = MODELS_DIR / "preprocessor.joblib",
):
    print("=" * 75)
    print("[EVALUATION] EXECUTING ADVANCED MODEL VALIDATION EVALUATION")
    print(f"Validation Data: {val_data_path}")
    print(f"Model Artifact:  {model_path}")
    print("=" * 75)

    if not model_path.exists():
        raise FileNotFoundError(f"Advanced model artifact not found at {model_path}. Run src/train_advanced.py first.")

    payload = joblib.load(model_path)
    model = payload["model"]
    model_type = payload.get("model_type", "AdvancedModel")
    best_params = payload.get("best_params", {})
    optimal_thresh = payload.get("optimal_threshold", 0.50)
    train_metrics = payload.get("train_metrics", {})
    feature_names = payload.get("feature_names", [])

    # Load preprocessor
    extractor = joblib.load(preprocessor_path)

    # Load validation data
    val_df = pd.read_csv(val_data_path)
    y_val = val_df["is_fraud"].values
    amounts_val = val_df["amt"].values

    # Extract features
    X_val = extractor.transform(val_df)

    # Predict probabilities
    p_val = model.predict_proba(X_val)[:, 1]

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
    print(f"    * Model Architecture:     {model_type} (Tuned)")
    print(f"    * Best Hyperparameters:   {best_params}")
    print(f"    * Cost-Optimal Threshold: {optimal_thresh:.2f}")
    print(f"    * Precision:              {prec:.4f}")
    print(f"    * Recall (Coverage):      {rec:.4f}")
    print(f"    * F1-Score:               {f1:.4f}")
    print(f"    * PR-AUC:                 {pr_auc:.4f}")
    print(f"    * ROC-AUC:                {roc_auc:.4f}")
    print(f"    * Confusion Matrix:       TP={tp} | FP={fp} | FN={fn} | TN={tn}")
    print(f"    * Estimated Total Cost:   ${costs['total_cost']:,.2f}")
    print(f"    * Net Capital Saved:      ${costs['net_savings']:,.2f}")

    # Threshold sweeps for charts
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
    colors = {"navy": "#0C2340", "blue": "#3395FF", "green": "#00C48C", "red": "#FF4D4F", "amber": "#FAAD14", "purple": "#8B5CF6"}

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
    ax.set_title(f"Advanced Model Confusion Matrix (Threshold = {optimal_thresh:.2f})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p1 = FIGURES_DIR / "advanced_1_confusion_matrix.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p1.name}")

    # 2. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec_pts, prec_pts, color=colors["navy"], linewidth=2.5, label=f"{model_type} (PR-AUC = {pr_auc:.4f})")
    ax.axhline(y=np.mean(y_val), color="gray", linestyle="--", label=f"No-Skill Baseline ({np.mean(y_val)*100:.1f}%)")
    ax.set_title("Precision-Recall Curve (Validation Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall (Fraud Catch Rate)")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    p2 = FIGURES_DIR / "advanced_2_precision_recall_curve.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p2.name}")

    # 3. ROC Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_pts, tpr_pts, color=colors["blue"], linewidth=2.5, label=f"{model_type} (ROC-AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Chance (AUC = 0.50)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    p3 = FIGURES_DIR / "advanced_3_roc_curve.png"
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p3.name}")

    # 4. Multi-Metric Threshold Curve (Precision, Recall, F1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, prec_list, color=colors["blue"], linewidth=2, label="Precision")
    ax.plot(thresholds, rec_list, color=colors["green"], linewidth=2, label="Recall")
    ax.plot(thresholds, f1_list, color=colors["navy"], linewidth=2.5, label="F1-Score")
    ax.axvline(x=optimal_thresh, color=colors["amber"], linestyle="--", label=f"Cost-Optimal Cutoff ({optimal_thresh:.2f})")
    ax.set_title("Threshold Trade-Off: Precision vs. Recall vs. F1-Score", fontsize=12, fontweight="bold")
    ax.set_xlabel("Decision Threshold (Probability Cutoff)")
    ax.set_ylabel("Metric Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    p4 = FIGURES_DIR / "advanced_4_threshold_vs_f1_precision_recall.png"
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p4.name}")

    # 5. Financial Risk Cost Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, cost_list, color=colors["red"], linewidth=2.5, marker="o")
    ax.axvline(x=optimal_thresh, color=colors["green"], linestyle="--", label=f"Optimal Cutoff ({optimal_thresh:.2f}) - Min Cost ${costs['total_cost']:,.2f}")
    ax.set_title("Total Financial Risk Cost vs. Decision Threshold", fontsize=12, fontweight="bold")
    ax.set_xlabel("Decision Threshold (Probability Cutoff)")
    ax.set_ylabel("Total Financial Risk Cost ($)")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    p5 = FIGURES_DIR / "advanced_5_threshold_vs_financial_cost.png"
    fig.savefig(p5, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p5.name}")

    # 6. Model-Based Feature Importance
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    else:
        from sklearn.inspection import permutation_importance
        res_p = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42)
        imps = res_p.importances_mean

    feat_imp = pd.Series(imps, index=feature_names).sort_values(ascending=True)
    y_pos = np.arange(len(feat_imp))
    bars = ax.barh(y_pos, feat_imp.values, color=colors["blue"], height=0.6)
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2.0, f"{feat_imp.values[i]:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_imp.index, fontsize=10)
    ax.set_xlabel("Model Importance (Gini / Gain / Permutation)")
    ax.set_title(f"Feature Importance Ranking ({model_type})", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(feat_imp.values) * 1.15)
    fig.tight_layout()
    p6 = FIGURES_DIR / "advanced_6_feature_importance.png"
    fig.savefig(p6, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p6.name}")

    # 7. Train vs. Validation Overfitting Comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    metrics_comp = ["PR-AUC", "F1-Score"]
    train_scores = [train_metrics.get("pr_auc", pr_auc), train_metrics.get("f1_score", f1)]
    val_scores = [pr_auc, f1]

    x = np.arange(len(metrics_comp))
    width = 0.35
    ax.bar(x - width/2, train_scores, width, label="Training Set", color=colors["blue"])
    ax.bar(x + width/2, val_scores, width, label="Validation Set", color=colors["green"])

    ax.set_ylabel("Score")
    ax.set_title("Overfitting Check: Train vs. Validation Performance", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_comp, fontweight="bold")
    ax.set_ylim(0, 1.15)
    for i in range(len(metrics_comp)):
        ax.text(i - width/2, train_scores[i] + 0.02, f"{train_scores[i]:.4f}", ha="center", fontweight="bold", fontsize=9)
        ax.text(i + width/2, val_scores[i] + 0.02, f"{val_scores[i]:.4f}", ha="center", fontweight="bold", fontsize=9)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    p7 = FIGURES_DIR / "advanced_7_train_vs_val_overfitting.png"
    fig.savefig(p7, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p7.name}")

    print("\n" + "=" * 75)
    print("[COMPLETE] ADVANCED EVALUATION & VISUALIZATIONS FINISHED")
    print("=" * 75)


if __name__ == "__main__":
    evaluate_advanced_model()

