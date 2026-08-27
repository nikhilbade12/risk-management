"""Financial Cost Model & False-Positive Optimization Engine.

Models the exact business trade-offs of payment gateway fraud decisions:
- False Negatives: Direct loss of goods + chargeback fees ($20) + operational dispute overhead.
- False Positives: Lost merchant profit margin (e.g. 20%) + customer churn & support friction ($10).
- Sweeps risk thresholds (1 to 99) to locate the cost-minimizing operating threshold.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from src.config import (
    DEFAULT_CHARGEBACK_FEE,
    DEFAULT_MERCHANT_MARGIN,
    DEFAULT_CUSTOMER_FRICTION,
    DEFAULT_OPERATIONAL_COST,
)


class FinancialCostModel:
    """Calculates financial risk costs and optimizes decision thresholds."""

    def __init__(
        self,
        chargeback_fee: float = DEFAULT_CHARGEBACK_FEE,
        merchant_margin: float = DEFAULT_MERCHANT_MARGIN,
        customer_friction: float = DEFAULT_CUSTOMER_FRICTION,
        operational_cost: float = DEFAULT_OPERATIONAL_COST,
    ):
        self.chargeback_fee = chargeback_fee
        self.merchant_margin = merchant_margin
        self.customer_friction = customer_friction
        self.operational_cost = operational_cost

    def compute_cost_breakdown(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        amounts: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Computes detailed financial loss and savings breakdown for given predictions.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        amounts = np.asarray(amounts)

        tp_mask = (y_true == 1) & (y_pred == 1)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)
        tn_mask = (y_true == 0) & (y_pred == 0)

        tp_count = int(np.sum(tp_mask))
        fp_count = int(np.sum(fp_mask))
        fn_count = int(np.sum(fn_mask))
        tn_count = int(np.sum(tn_mask))

        # Financial Calculations
        # 1. False Negative Cost (Uncaught fraud): Stolen transaction amount + chargeback fee + ops cost
        fn_amounts = amounts[fn_mask]
        fn_fraud_loss = float(np.sum(fn_amounts))
        fn_chargeback_fees = fn_count * self.chargeback_fee
        total_fn_cost = fn_fraud_loss + fn_chargeback_fees + (fn_count * self.operational_cost)

        # 2. False Positive Cost (Good customer rejected): Lost gross profit margin on order + customer churn friction
        fp_amounts = amounts[fp_mask]
        fp_margin_loss = float(np.sum(fp_amounts * self.merchant_margin))
        fp_friction_loss = fp_count * self.customer_friction
        total_fp_cost = fp_margin_loss + fp_friction_loss

        # 3. Fraud Prevented (Value preserved by blocking fraudulent transactions)
        tp_amounts = amounts[tp_mask]
        fraud_prevented = float(np.sum(tp_amounts)) + (tp_count * self.chargeback_fee)

        # 4. Total Financial Risk Cost
        total_cost = total_fn_cost + total_fp_cost
        net_saved = fraud_prevented - total_fp_cost

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        return {
            "confusion_matrix": {
                "tp": tp_count,
                "fp": fp_count,
                "fn": fn_count,
                "tn": tn_count,
            },
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            },
            "financials": {
                "fn_cost": round(total_fn_cost, 2),
                "fp_cost": round(total_fp_cost, 2),
                "total_risk_cost": round(total_cost, 2),
                "fraud_loss_prevented": round(fraud_prevented, 2),
                "net_capital_saved": round(net_saved, 2),
            },
        }

    def sweep_thresholds(
        self,
        y_true: np.ndarray,
        risk_scores: np.ndarray,
        amounts: np.ndarray,
    ) -> pd.DataFrame:
        """
        Sweeps risk score thresholds from 1 to 99 and calculates performance
        and financial metrics at each step.
        """
        records = []
        for thresh in range(1, 100):
            y_pred = (risk_scores >= thresh).astype(int)
            res = self.compute_cost_breakdown(y_true, y_pred, amounts)

            records.append({
                "threshold": thresh,
                "precision": res["metrics"]["precision"],
                "recall": res["metrics"]["recall"],
                "f1": res["metrics"]["f1"],
                "tp": res["confusion_matrix"]["tp"],
                "fp": res["confusion_matrix"]["fp"],
                "fn": res["confusion_matrix"]["fn"],
                "tn": res["confusion_matrix"]["tn"],
                "fn_cost": res["financials"]["fn_cost"],
                "fp_cost": res["financials"]["fp_cost"],
                "total_cost": res["financials"]["total_risk_cost"],
                "fraud_loss_prevented": res["financials"]["fraud_loss_prevented"],
                "net_capital_saved": res["financials"]["net_capital_saved"],
            })

        df_sweep = pd.DataFrame(records)
        return df_sweep

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        risk_scores: np.ndarray,
        amounts: np.ndarray,
    ) -> Tuple[int, Dict[str, Any]]:
        """Finds the operating threshold that mathematically minimizes total cost."""
        df_sweep = self.sweep_thresholds(y_true, risk_scores, amounts)
        min_idx = df_sweep["total_cost"].idxmin()
        best_row = df_sweep.loc[min_idx]
        optimal_threshold = int(best_row["threshold"])

        y_pred_opt = (risk_scores >= optimal_threshold).astype(int)
        opt_breakdown = self.compute_cost_breakdown(y_true, y_pred_opt, amounts)
        opt_breakdown["optimal_threshold"] = optimal_threshold

        return optimal_threshold, opt_breakdown

