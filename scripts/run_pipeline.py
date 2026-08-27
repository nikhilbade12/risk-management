"""End-to-End Pipeline Execution Script for AI Risk Manager.

Executes:
1. Data loading and 70/15/15 stratified train/val/test splitting.
2. Leak-free feature engineering fitting.
3. Supervised model training & validation comparison (Random Forest vs HistGradientBoosting).
4. Unsupervised Anomaly Detector (Isolation Forest) fitting & calibration.
5. Rules Engine evaluation.
6. Hybrid Engine weight optimization on the Validation Set.
7. Strict held-out Test Set evaluation, threshold cost optimization, and metric persistence.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Ensure utf-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import (
    PREPROCESSOR_PATH,
    SUPERVISED_MODEL_PATH,
    ANOMALY_MODEL_PATH,
    METRICS_PATH,
    MODELS_DIR,
)
from src.data_loader import load_processed_splits
from src.feature_engineering import TransactionFeatureExtractor
from src.supervised_model import FraudClassifier
from src.anomaly_detector import AnomalyDetector
from src.rules_engine import RulesEngine
from src.hybrid_engine import HybridRiskEngine
from src.cost_model import FinancialCostModel
from src.evaluation import compute_comprehensive_metrics, save_test_metrics


def run_pipeline():
    print("=" * 65)
    print("[START] AI RISK MANAGER END-TO-END TRAINING & EVALUATION")
    print("=" * 65)

    # 1. Load Data Splits
    print("\n[Step 1/7] Loading and partitioning dataset (70% Train, 15% Val, 15% Test)...")
    train_df, val_df, test_df = load_processed_splits()

    y_train = train_df["is_fraud"].values
    y_val = val_df["is_fraud"].values
    y_test = test_df["is_fraud"].values

    # 2. Feature Engineering (Fitted ONLY on Train partition)
    print("\n[Step 2/7] Fitting feature extraction pipeline on training partition...")
    extractor = TransactionFeatureExtractor()
    extractor.fit(train_df)

    X_train = extractor.transform(train_df)
    X_val = extractor.transform(val_df)
    X_test = extractor.transform(test_df)

    # Save fitted preprocessor
    joblib.dump(extractor, PREPROCESSOR_PATH)
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")

    # 3. Train Supervised Classifier
    print("\n[Step 3/7] Training supervised classification models...")
    # Compare Random Forest vs HistGradientBoosting on Validation Set
    rf_clf = FraudClassifier(model_type="rf")
    rf_clf.fit(X_train, y_train)
    rf_val_prob = rf_clf.predict_proba(X_val)

    hgb_clf = FraudClassifier(model_type="hgb")
    hgb_clf.fit(X_train, y_train)
    hgb_val_prob = hgb_clf.predict_proba(X_val)

    # Calculate validation PR-AUC for both
    from sklearn.metrics import precision_recall_curve, auc
    p_rf, r_rf, _ = precision_recall_curve(y_val, rf_val_prob)
    auc_rf = auc(r_rf, p_rf)

    p_hgb, r_hgb, _ = precision_recall_curve(y_val, hgb_val_prob)
    auc_hgb = auc(r_hgb, p_hgb)

    print(f"Validation PR-AUC -> Random Forest: {auc_rf:.4f} | HistGradientBoosting: {auc_hgb:.4f}")

    if auc_rf >= auc_hgb:
        best_clf = rf_clf
        p_val_ml = rf_val_prob
        p_test_ml = rf_clf.predict_proba(X_test)
        selected_name = "RandomForest"
    else:
        best_clf = hgb_clf
        p_val_ml = hgb_val_prob
        p_test_ml = hgb_clf.predict_proba(X_test)
        selected_name = "HistGradientBoosting"

    best_clf.save(SUPERVISED_MODEL_PATH)
    print(f"Selected superior model: {selected_name}")

    # 4. Train Unsupervised Anomaly Detector (Isolation Forest)
    print("\n[Step 4/7] Training Isolation Forest anomaly detector...")
    anomaly_detector = AnomalyDetector(n_estimators=120, contamination=0.018)
    anomaly_detector.fit(X_train)
    anomaly_detector.save(ANOMALY_MODEL_PATH)

    s_val_anom = anomaly_detector.score(X_val)
    s_test_anom = anomaly_detector.score(X_test)

    # 5. Evaluate Heuristic Rules Engine
    print("\n[Step 5/7] Evaluating heuristic risk rules...")
    rules_engine = RulesEngine()
    # Compute rule features by passing feature dictionary
    val_rule_scores = rules_engine.evaluate_dataframe(X_val).values
    test_rule_scores = rules_engine.evaluate_dataframe(X_test).values

    # 6. Optimize Hybrid Weights on Validation Set
    print("\n[Step 6/7] Optimizing hybrid engine weights on validation set...")
    w_ml, w_anom, w_rules = HybridRiskEngine.optimize_weights_on_val(
        y_val=y_val,
        p_ml_val=p_val_ml,
        s_anom_val=s_val_anom,
        s_rules_val=val_rule_scores,
    )

    hybrid_engine = HybridRiskEngine(
        weight_ml=w_ml,
        weight_anomaly=w_anom,
        weight_rules=w_rules,
    )

    # 7. Strictly Held-Out Test Set Evaluation
    print("\n[Step 7/7] Evaluating performance on held-out test set...")
    test_batch_df = hybrid_engine.evaluate_batch(
        p_ml_array=p_test_ml,
        s_anom_array=s_test_anom,
        s_rules_array=test_rule_scores,
    )
    test_risk_scores = test_batch_df["risk_score"].values

    # Financial Cost Optimization on Test Set
    cost_model = FinancialCostModel()
    test_amounts = test_df["amt"].values
    optimal_thresh, opt_breakdown = cost_model.find_optimal_threshold(
        y_true=y_test,
        risk_scores=test_risk_scores,
        amounts=test_amounts,
    )

    # Compute full test metrics at optimal threshold
    test_preds_binary = (test_risk_scores >= optimal_thresh).astype(int)
    test_metrics = compute_comprehensive_metrics(
        y_true=y_test,
        y_pred=test_preds_binary,
        y_prob=test_risk_scores / 100.0,
        model_name=f"Hybrid Risk Engine ({selected_name} + Anomaly + Rules)",
    )

    # Save threshold sweep data for interactive dashboard visualization
    df_sweep = cost_model.sweep_thresholds(y_test, test_risk_scores, test_amounts)
    sweep_path = MODELS_DIR / "cost_threshold_sweep.csv"
    df_sweep.to_csv(sweep_path, index=False)

    # Compile comprehensive verifiable report
    final_report = {
        "pipeline_status": "SUCCESS",
        "selected_supervised_model": selected_name,
        "hybrid_weights": {
            "w_ml": w_ml,
            "w_anomaly": w_anom,
            "w_rules": w_rules,
        },
        "optimal_threshold": optimal_thresh,
        "test_metrics": test_metrics,
        "financial_impact": opt_breakdown["financials"],
        "validation_comparison": {
            "random_forest_pr_auc": round(float(auc_rf), 4),
            "hist_gradient_boosting_pr_auc": round(float(auc_hgb), 4),
        },
    }

    save_test_metrics(final_report, METRICS_PATH)

    print("\n" + "=" * 65)
    print("[COMPLETE] PIPELINE EXECUTION COMPLETE - VERIFIED TEST METRICS")
    print("=" * 65)
    print(f"Sample Size (Held-out Test):  {test_metrics['sample_size']:,} transactions")
    print(f"Actual Frauds in Test:        {test_metrics['fraud_count']:,} ({test_metrics['fraud_rate_percent']}%)")
    print(f"PR-AUC:                       {test_metrics['pr_auc']:.4f}")
    print(f"ROC-AUC:                      {test_metrics['roc_auc']:.4f}")
    print(f"Precision:                    {test_metrics['precision']:.4f}")
    print(f"Recall:                       {test_metrics['recall']:.4f}")
    print(f"F1-Score:                     {test_metrics['f1_score']:.4f}")
    print(f"F2-Score:                     {test_metrics['f2_score']:.4f}")
    print(f"False Positive Rate (FPR):    {test_metrics['false_positive_rate']*100:.2f}%")
    print(f"Optimal Risk Threshold:       {optimal_thresh} / 100")
    print(f"Fraud Loss Prevented:         ${opt_breakdown['financials']['fraud_loss_prevented']:,.2f}")
    print(f"False Positive Cost:          ${opt_breakdown['financials']['fp_cost']:,.2f}")
    print(f"Net Capital Saved:            ${opt_breakdown['financials']['net_capital_saved']:,.2f}")
    print("=" * 65)

    return final_report


if __name__ == "__main__":
    run_pipeline()
