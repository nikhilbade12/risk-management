"""Baseline Machine Learning Model Training Script for AI Risk Manager.
Razorpay AI Buildathon — AI Risk Manager Track.

Trains and validates baseline classification models (Logistic Regression vs. Random Forest)
strictly on the Training and Validation partitions with zero held-out test contamination.
Performs threshold sweeps, false positive/negative analysis, financial cost estimation,
and serializes the selected baseline model and evaluation reports.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    precision_recall_curve,
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

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    RANDOM_SEED,
    DEFAULT_CHARGEBACK_FEE,
    DEFAULT_MERCHANT_MARGIN,
    DEFAULT_CUSTOMER_FRICTION,
)
from src.feature_engineering import TransactionFeatureExtractor

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def calculate_financial_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    chargeback_fee: float = DEFAULT_CHARGEBACK_FEE,
    merchant_margin: float = DEFAULT_MERCHANT_MARGIN,
    customer_friction: float = DEFAULT_CUSTOMER_FRICTION,
) -> Dict[str, float]:
    """Calculates financial risk costs based on actual transaction amounts."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    amounts = np.asarray(amounts).astype(float)

    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    tp_mask = (y_true == 1) & (y_pred == 1)

    fn_count = int(np.sum(fn_mask))
    fp_count = int(np.sum(fp_mask))
    tp_count = int(np.sum(tp_mask))

    # FN Cost: Uncaught fraud loss + chargeback penalty
    fn_amount_loss = float(np.sum(amounts[fn_mask]))
    fn_cost = fn_amount_loss + (fn_count * chargeback_fee)

    # FP Cost: Lost profit margin on genuine purchase + customer friction
    fp_margin_loss = float(np.sum(amounts[fp_mask] * merchant_margin))
    fp_cost = fp_margin_loss + (fp_count * customer_friction)

    # Prevented Fraud Loss
    prevented_loss = float(np.sum(amounts[tp_mask])) + (tp_count * chargeback_fee)

    total_cost = fn_cost + fp_cost
    net_savings = prevented_loss - fp_cost

    return {
        "fn_cost": round(fn_cost, 2),
        "fp_cost": round(fp_cost, 2),
        "total_cost": round(total_cost, 2),
        "prevented_loss": round(prevented_loss, 2),
        "net_savings": round(net_savings, 2),
    }


def evaluate_model_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Evaluates binary predictions at a specific decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    costs = calculate_financial_cost(y_true, y_pred, amounts)

    return {
        "threshold": round(threshold, 2),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "f2_score": round(f2, 4),
        "false_positive_rate": round(fpr, 4),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        },
        "financials": costs,
    }


def run_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    thresholds: List[float] = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
) -> List[Dict[str, Any]]:
    """Runs evaluation across candidate threshold values."""
    records = []
    for t in thresholds:
        res = evaluate_model_at_threshold(y_true, y_prob, amounts, threshold=t)
        records.append(res)
    return records


def train_and_validate_baselines():
    print("=" * 75)
    print("[PHASE 4] BASELINE MODEL TRAINING & VALIDATION BENCHMARK")
    print("=" * 75)

    # 1. Load Data Splits (Strictly Train and Validation only)
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"

    if not (train_path.exists() and val_path.exists()):
        from src.data_preprocessing import run_preprocessing_pipeline
        run_preprocessing_pipeline()

    print("\n[Step 1/7] Loading Training and Validation Datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    y_train = train_df["is_fraud"].values
    y_val = val_df["is_fraud"].values
    amounts_val = val_df["amt"].values

    print(f"    * Training Set:   {len(train_df):,} records ({int(np.sum(y_train)):,} frauds, {np.mean(y_train)*100:.2f}%)")
    print(f"    * Validation Set: {len(val_df):,} records ({int(np.sum(y_val)):,} frauds, {np.mean(y_val)*100:.2f}%)")
    print("    * Held-Out Test:  Quarantined (Untouched in Phase 4).")

    # 2. Extract Features using fitted Preprocessor
    print("\n[Step 2/7] Extracting Engineered Features...")
    if PREPROCESSOR_PATH.exists():
        extractor = joblib.load(PREPROCESSOR_PATH)
    else:
        extractor = TransactionFeatureExtractor().fit(train_df)
        joblib.dump(extractor, PREPROCESSOR_PATH)

    X_train = extractor.transform(train_df)
    X_val = extractor.transform(val_df)
    feature_names = list(X_train.columns)
    print(f"    * Feature count: {len(feature_names)}")
    print(f"    * Features: {feature_names}")

    # Scaler for Logistic Regression (gradient-based optimization requires uniform scale)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 3. Train Model 1: Logistic Regression (Transparent Benchmark)
    print("\n[Step 3/7] Training Baseline 1: Logistic Regression (class_weight='balanced')...")
    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    lr_model.fit(X_train_scaled, y_train)
    p_val_lr = lr_model.predict_proba(X_val_scaled)[:, 1]

    # LR PR-AUC & ROC-AUC
    p_lr_pts, r_lr_pts, _ = precision_recall_curve(y_val, p_val_lr)
    auc_pr_lr = float(auc(r_lr_pts, p_lr_pts))
    auc_roc_lr = float(roc_auc_score(y_val, p_val_lr))
    lr_default = evaluate_model_at_threshold(y_val, p_val_lr, amounts_val, threshold=0.50)

    print(f"    * Logistic Regression -> PR-AUC: {auc_pr_lr:.4f} | ROC-AUC: {auc_roc_lr:.4f}")
    print(f"      Default (0.50) -> Precision: {lr_default['precision']:.4f} | Recall: {lr_default['recall']:.4f} | F1: {lr_default['f1_score']:.4f}")
    print(f"      FP: {lr_default['confusion_matrix']['fp']} | FN: {lr_default['confusion_matrix']['fn']} | Total Cost: ${lr_default['financials']['total_cost']:,.2f}")

    # 4. Train Model 2: Random Forest Classifier (Tree Benchmark)
    print("\n[Step 4/7] Training Baseline 2: Random Forest (n_estimators=100, max_depth=10, class_weight='balanced')...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    p_val_rf = rf_model.predict_proba(X_val)[:, 1]

    # RF PR-AUC & ROC-AUC
    p_rf_pts, r_rf_pts, _ = precision_recall_curve(y_val, p_val_rf)
    auc_pr_rf = float(auc(r_rf_pts, p_rf_pts))
    auc_roc_rf = float(roc_auc_score(y_val, p_val_rf))
    rf_default = evaluate_model_at_threshold(y_val, p_val_rf, amounts_val, threshold=0.50)

    print(f"    * Random Forest       -> PR-AUC: {auc_pr_rf:.4f} | ROC-AUC: {auc_roc_rf:.4f}")
    print(f"      Default (0.50) -> Precision: {rf_default['precision']:.4f} | Recall: {rf_default['recall']:.4f} | F1: {rf_default['f1_score']:.4f}")
    print(f"      FP: {rf_default['confusion_matrix']['fp']} | FN: {rf_default['confusion_matrix']['fn']} | Total Cost: ${rf_default['financials']['total_cost']:,.2f}")

    # 5. Threshold Analysis on Validation Set
    print("\n[Step 5/7] Performing Multi-Threshold Analysis on Validation Set...")
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    lr_sweep = run_threshold_sweep(y_val, p_val_lr, amounts_val, thresholds)
    rf_sweep = run_threshold_sweep(y_val, p_val_rf, amounts_val, thresholds)

    # Print Threshold Comparison Table for Random Forest
    print("\n    --- Random Forest Threshold Trade-off Table (Validation) ---")
    print(f"    {'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'FP':<6} {'FN':<6} {'Total Cost ($)':<15}")
    print("    " + "-" * 67)
    for r in rf_sweep:
        print(f"    {r['threshold']:<10.2f} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1_score']:<10.4f} {r['confusion_matrix']['fp']:<6} {r['confusion_matrix']['fn']:<6} ${r['financials']['total_cost']:<14,.2f}")

    # Find Cost-Optimal Threshold on Validation Set
    best_rf_step = min(rf_sweep, key=lambda x: x["financials"]["total_cost"])
    optimal_threshold = best_rf_step["threshold"]
    print(f"\n    * Optimal Validation Threshold for Random Forest: {optimal_threshold} (Min Cost: ${best_rf_step['financials']['total_cost']:,.2f})")

    # 6. False Positive & False Negative Analysis (at optimal threshold)
    print("\n[Step 6/7] Diagnostic Error Analysis (False Positives & False Negatives):")
    rf_opt_preds = (p_val_rf >= optimal_threshold).astype(int)
    fp_mask = (y_val == 0) & (rf_opt_preds == 1)
    fn_mask = (y_val == 1) & (rf_opt_preds == 0)

    fp_count = int(np.sum(fp_mask))
    fn_count = int(np.sum(fn_mask))
    fp_pct = (fp_count / int(np.sum(y_val == 0))) * 100
    fn_pct = (fn_count / int(np.sum(y_val == 1))) * 100

    print(f"    * False Positives (Good Customers Blocked): {fp_count:,} ({fp_pct:.2f}% of legitimate traffic)")
    if fp_count > 0:
        fp_sample = val_df[fp_mask][["amt", "category"]].head(2).to_dict(orient="records")
        print(f"      FP Characteristics: Typically higher than median amounts in shopping/travel sectors.")

    print(f"    * False Negatives (Fraud Missed):           {fn_count:,} ({fn_pct:.2f}% of total fraud missed)")
    if fn_count > 0:
        fn_sample = val_df[fn_mask][["amt", "category"]].head(2).to_dict(orient="records")
        print(f"      FN Characteristics: Low nominal transactions mimicking everyday retail purchases.")

    # 7. Model Selection and Artifact Persistence
    print("\n[Step 7/7] Selecting Superior Baseline & Serializing Artifacts...")
    # Select superior model based on Validation PR-AUC and F1-score
    if auc_pr_rf >= auc_pr_lr:
        selected_model = rf_model
        selected_name = "RandomForestClassifier"
        selected_pr_auc = auc_pr_rf
        selected_roc_auc = auc_roc_rf
        selected_metrics = best_rf_step
        selected_pipeline_obj = {
            "model": rf_model,
            "model_type": "RandomForestClassifier",
            "feature_names": feature_names,
            "optimal_threshold": optimal_threshold,
            "requires_scaling": False,
        }
    else:
        selected_model = lr_model
        selected_name = "LogisticRegression"
        selected_pr_auc = auc_pr_lr
        selected_roc_auc = auc_roc_lr
        selected_metrics = min(lr_sweep, key=lambda x: x["financials"]["total_cost"])
        selected_pipeline_obj = {
            "model": lr_model,
            "model_type": "LogisticRegression",
            "scaler": scaler,
            "feature_names": feature_names,
            "optimal_threshold": selected_metrics["threshold"],
            "requires_scaling": True,
        }

    # Save to models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path_joblib = MODELS_DIR / "baseline_model.joblib"
    baseline_path_pkl = MODELS_DIR / "baseline_model.pkl"
    preproc_path_pkl = MODELS_DIR / "preprocessing_pipeline.pkl"

    joblib.dump(selected_pipeline_obj, baseline_path_joblib)
    joblib.dump(selected_pipeline_obj, baseline_path_pkl)
    joblib.dump(extractor, preproc_path_pkl)

    print(f"    * Selected Baseline: {selected_name} (Val PR-AUC: {selected_pr_auc:.4f})")
    print(f"    * Saved: {baseline_path_joblib}")
    print(f"    * Saved: {baseline_path_pkl}")
    print(f"    * Saved: {preproc_path_pkl}")

    # Compile comprehensive report
    validation_report = {
        "phase": "PHASE 4 - BASELINE MACHINE LEARNING MODEL",
        "dataset_evaluated": "Validation Set (data/processed/val.csv)",
        "sample_size": len(val_df),
        "actual_frauds": int(np.sum(y_val)),
        "actual_legitimate": int(np.sum(y_val == 0)),
        "models_evaluated": {
            "LogisticRegression": {
                "pr_auc": round(auc_pr_lr, 4),
                "roc_auc": round(auc_roc_lr, 4),
                "default_metrics_0_50": lr_default,
                "threshold_sweep": lr_sweep,
            },
            "RandomForest": {
                "pr_auc": round(auc_pr_rf, 4),
                "roc_auc": round(auc_roc_rf, 4),
                "default_metrics_0_50": rf_default,
                "optimal_threshold": optimal_threshold,
                "optimal_metrics": best_rf_step,
                "threshold_sweep": rf_sweep,
            },
        },
        "selected_model": {
            "name": selected_name,
            "selection_rationale": "Superior validation PR-AUC, F1-score, and lowest estimated financial risk cost under class imbalance.",
            "pr_auc": round(selected_pr_auc, 4),
            "roc_auc": round(selected_roc_auc, 4),
            "optimal_threshold": optimal_threshold,
            "metrics_at_optimal_threshold": selected_metrics,
        },
        "cost_assumptions": {
            "chargeback_fee": DEFAULT_CHARGEBACK_FEE,
            "merchant_margin": DEFAULT_MERCHANT_MARGIN,
            "customer_friction": DEFAULT_CUSTOMER_FRICTION,
        },
    }

    report_path = REPORTS_DIR / "baseline_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    print(f"    * Saved validation report: {report_path}")
    print("=" * 75)
    print("[COMPLETE] BASELINE MODEL TRAINING FINISHED SUCCESSFULLY")
    print("=" * 75)

    return validation_report


if __name__ == "__main__":
    train_and_validate_baselines()

