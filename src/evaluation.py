"""Comprehensive Evaluation Module for AI Risk Manager.

Computes precision, recall, F1, F2, PR-AUC, ROC-AUC, FPR, and full confusion matrix.
Persists authentic test-set metrics to models/test_metrics.json for zero-hallucination reporting.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
)

from src.config import METRICS_PATH


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Hybrid Risk Engine",
) -> Dict[str, Any]:
    """
    Computes all standard fraud risk metrics on real test data.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Rates
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Curves and AUCs
    roc_auc = roc_auc_score(y_true, y_prob)
    prec_pts, rec_pts, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec_pts, prec_pts)

    metrics = {
        "model_name": model_name,
        "sample_size": len(y_true),
        "fraud_count": int(np.sum(y_true)),
        "legitimate_count": int(len(y_true) - np.sum(y_true)),
        "fraud_rate_percent": round(float(np.mean(y_true) * 100), 2),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "f2_score": round(float(f2), 4),
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "false_positive_rate": round(float(fpr), 4),
        "confusion_matrix": {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        },
    }
    return metrics


def save_test_metrics(metrics_dict: Dict[str, Any], filepath: Path = METRICS_PATH) -> None:
    """Serializes test metrics to JSON for application dashboard use."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Verified test metrics saved to {filepath}")


def load_test_metrics(filepath: Path = METRICS_PATH) -> Optional[Dict[str, Any]]:
    """Loads saved test metrics."""
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

