"""Hybrid Risk Engine combining Supervised ML, Anomaly Detection, and Risk Rules.

Produces a unified, explainable Risk Score (0-100) and maps to actionable Risk Tiers:
LOW RISK (0-35), MEDIUM RISK (36-70), and HIGH RISK (71-100).
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_WEIGHT_ML,
    DEFAULT_WEIGHT_ANOMALY,
    DEFAULT_WEIGHT_RULES,
    TIER_LOW_MAX,
    TIER_MEDIUM_MAX,
)


class HybridRiskEngine:
    """Combines multi-modal fraud detection signals into a calibrated 0-100 score."""

    def __init__(
        self,
        weight_ml: float = DEFAULT_WEIGHT_ML,
        weight_anomaly: float = DEFAULT_WEIGHT_ANOMALY,
        weight_rules: float = DEFAULT_WEIGHT_RULES,
        tier_low_max: float = TIER_LOW_MAX,
        tier_med_max: float = TIER_MEDIUM_MAX,
    ):
        total_w = weight_ml + weight_anomaly + weight_rules
        self.w_ml = weight_ml / total_w
        self.w_anom = weight_anomaly / total_w
        self.w_rules = weight_rules / total_w
        self.tier_low_max = tier_low_max
        self.tier_med_max = tier_med_max

    def calculate_score(
        self,
        p_ml: float,
        s_anom: float,
        s_rules: float,
    ) -> int:
        """Fuses three signals into a 0-100 integer score."""
        raw_score = (self.w_ml * p_ml) + (self.w_anom * s_anom) + (self.w_rules * s_rules)
        scaled_score = int(np.clip(np.round(raw_score * 100), 0, 100))
        return scaled_score

    def classify_tier(self, score: int) -> Tuple[str, str]:
        """
        Maps a 0-100 score into a Risk Tier and recommended Merchant Action.
        Returns (risk_level, recommended_action).
        """
        if score <= self.tier_low_max:
            return "LOW RISK", "Approve Transaction (Frictionless Checkout)"
        elif score <= self.tier_med_max:
            return "MEDIUM RISK", "Challenge (Request 3D-Secure / OTP Verification)"
        else:
            return "HIGH RISK", "Decline / Block (Prevent Potential Chargeback)"

    def evaluate_transaction(
        self,
        p_ml: float,
        s_anom: float,
        s_rules: float,
        rule_details: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Evaluates a single transaction comprehensively."""
        score = self.calculate_score(p_ml, s_anom, s_rules)
        tier, action = self.classify_tier(score)

        return {
            "risk_score": score,
            "risk_level": tier,
            "recommended_action": action,
            "signals": {
                "supervised_ml_probability": round(float(p_ml), 4),
                "anomaly_score": round(float(s_anom), 4),
                "rule_risk_score": round(float(s_rules), 4),
            },
            "weights": {
                "w_ml": round(self.w_ml, 3),
                "w_anom": round(self.w_anom, 3),
                "w_rules": round(self.w_rules, 3),
            },
            "triggered_rules": rule_details or [],
        }

    def evaluate_batch(
        self,
        p_ml_array: np.ndarray,
        s_anom_array: np.ndarray,
        s_rules_array: np.ndarray,
    ) -> pd.DataFrame:
        """Batch evaluation of multi-signal vectors."""
        raw = (self.w_ml * p_ml_array) + (self.w_anom * s_anom_array) + (self.w_rules * s_rules_array)
        scores = np.clip(np.round(raw * 100), 0, 100).astype(int)

        tiers = []
        actions = []
        for s in scores:
            t, a = self.classify_tier(s)
            tiers.append(t)
            actions.append(a)

        return pd.DataFrame({
            "risk_score": scores,
            "risk_level": tiers,
            "recommended_action": actions,
            "p_ml": np.round(p_ml_array, 4),
            "s_anom": np.round(s_anom_array, 4),
            "s_rules": np.round(s_rules_array, 4),
        })

    @staticmethod
    def optimize_weights_on_val(
        y_val: np.ndarray,
        p_ml_val: np.ndarray,
        s_anom_val: np.ndarray,
        s_rules_val: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Grid search on validation set to find weights (w_ml, w_anom, w_rules)
        that maximize PR-AUC.
        """
        from sklearn.metrics import precision_recall_curve, auc

        best_pr_auc = -1.0
        best_weights = (0.60, 0.15, 0.25)

        # Grid search over simplex
        for w1 in np.linspace(0.4, 0.8, 9):
            for w2 in np.linspace(0.05, 0.35, 7):
                w3 = 1.0 - (w1 + w2)
                if w3 < 0.05:
                    continue
                combined = (w1 * p_ml_val) + (w2 * s_anom_val) + (w3 * s_rules_val)
                prec, rec, _ = precision_recall_curve(y_val, combined)
                score_auc = auc(rec, prec)

                if score_auc > best_pr_auc:
                    best_pr_auc = score_auc
                    best_weights = (round(w1, 3), round(w2, 3), round(w3, 3))

        print(f"Optimized Hybrid Weights: w_ml={best_weights[0]}, w_anom={best_weights[1]}, w_rules={best_weights[2]} (Val PR-AUC={best_pr_auc:.4f})")
        return best_weights

