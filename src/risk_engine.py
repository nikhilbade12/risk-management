"""Hybrid Risk Engine for AI Risk Manager.
Razorpay AI Buildathon — AI Risk Manager Track.

Fuses three independent intelligence signals:
1. Supervised Machine Learning Fraud Probability (XGBoost / Gradient Boosting)
2. Unsupervised Anomaly Risk Score (Isolation Forest)
3. Explainable Behavioral Risk Signals (Deterministic Heuristic Checks)

Computes a unified, calibrated Hybrid Risk Score: 0–100, assigns actionable Risk Tiers
(LOW, MEDIUM, HIGH), and produces structured decision outputs.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import joblib

from src.config import (
    DEFAULT_WEIGHT_ML,
    DEFAULT_WEIGHT_ANOMALY,
    DEFAULT_WEIGHT_RULES,
    TIER_LOW_MAX,
    TIER_MEDIUM_MAX,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    ANOMALY_MODEL_PATH,
)
from src.risk_signals import BehavioralRiskEvaluator


class HybridRiskEngine:
    """Multi-signal decision engine combining Supervised ML, Anomaly Detection, and Behavioral Rules."""

    def __init__(
        self,
        weight_ml: float = 0.65,
        weight_anomaly: float = 0.20,
        weight_behavioral: float = 0.15,
        tier_low_max: float = 35.0,
        tier_med_max: float = 69.0,
        advanced_model_payload: Optional[Dict[str, Any]] = None,
        anomaly_detector: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
    ):
        total_w = weight_ml + weight_anomaly + weight_behavioral
        self.w_ml = weight_ml / total_w
        self.w_anom = weight_anomaly / total_w
        self.w_behav = weight_behavioral / total_w

        self.tier_low_max = tier_low_max
        self.tier_med_max = tier_med_max

        self.advanced_model_payload = advanced_model_payload
        self.anomaly_detector = anomaly_detector
        self.preprocessor = preprocessor
        self.signal_evaluator = BehavioralRiskEvaluator()

    @classmethod
    def load_default(cls, config_path: Optional[Path] = None) -> "HybridRiskEngine":
        """Factory method loading serialized artifacts from models/ directory."""
        adv_path = MODELS_DIR / "advanced_model.joblib"
        if not adv_path.exists():
            adv_path = MODELS_DIR / "advanced_model.pkl"

        anom_path = MODELS_DIR / "anomaly_model.joblib"
        if not anom_path.exists():
            anom_path = MODELS_DIR / "anomaly_model.pkl"

        preproc_path = PREPROCESSOR_PATH

        adv_payload = joblib.load(adv_path) if adv_path.exists() else None
        anom_model = joblib.load(anom_path) if anom_path.exists() else None
        preproc = joblib.load(preproc_path) if preproc_path.exists() else None

        # Load weights from config if present
        w_ml, w_anom, w_behav = 0.65, 0.20, 0.15
        tier_low, tier_med = 35.0, 69.0
        cfg_file = config_path or (MODELS_DIR / "hybrid_config.json")
        if cfg_file.exists():
            import json
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                weights = cfg.get("component_weights", {})
                w_ml = weights.get("ml_weight", cfg.get("ml_weight", w_ml))
                w_anom = weights.get("anomaly_weight", cfg.get("anomaly_weight", w_anom))
                w_behav = weights.get("behavioral_weight", cfg.get("behavioral_weight", w_behav))

                thresholds = cfg.get("risk_thresholds", {})
                tier_low = thresholds.get("low_risk_max", cfg.get("low_risk_threshold", tier_low))
                tier_med = thresholds.get("medium_risk_max", cfg.get("medium_risk_threshold", tier_med))

        return cls(
            weight_ml=w_ml,
            weight_anomaly=w_anom,
            weight_behavioral=w_behav,
            tier_low_max=tier_low,
            tier_med_max=tier_med,
            advanced_model_payload=adv_payload,
            anomaly_detector=anom_model,
            preprocessor=preproc,
        )

    def calculate_hybrid_score(
        self,
        p_ml: float,
        s_anom_100: float,
        s_behav_100: float,
    ) -> int:
        """
        Fuses three normalized signals into an integer Hybrid Risk Score: 0–100.
        All input components are on a 0–100 scale:
        - ML Score: p_ml * 100
        - Anomaly Score: s_anom_100
        - Behavioral Score: s_behav_100
        """
        ml_score_100 = float(p_ml * 100.0)
        raw = (self.w_ml * ml_score_100) + (self.w_anom * s_anom_100) + (self.w_behav * s_behav_100)
        return int(np.clip(np.round(raw), 0, 100))

    def classify_tier(self, score: int) -> Tuple[str, str]:
        """
        Maps a 0–100 score into a Risk Tier and recommended Merchant Action.
        Returns: (risk_level, recommended_action)
        """
        if score <= self.tier_low_max:
            return "LOW RISK", "Approve Transaction (Frictionless Checkout)"
        elif score <= self.tier_med_max:
            return "MEDIUM RISK", "Step-Up Verification (3D-Secure OTP / Verification Required)"
        else:
            return "HIGH RISK", "Hold for Review / Decline (High Chargeback Probability)"

    def predict_risk(self, txn_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Public API for single transaction evaluation.
        Compatible with Streamlit application and backend API services.
        """
        txn_df = pd.DataFrame([txn_dict])
        txn_id = str(txn_dict.get("trans_num", txn_dict.get("transaction_id", "TXN_UNKNOWN")))

        # 1. Feature extraction
        if self.preprocessor is not None:
            X_feat = self.preprocessor.transform(txn_df)
        else:
            X_feat = txn_df

        # 2. Supervised ML Fraud Probability
        if self.advanced_model_payload is not None:
            model = self.advanced_model_payload["model"]
            feat_names = self.advanced_model_payload.get("feature_names", list(X_feat.columns))
            X_input = X_feat[feat_names] if all(c in X_feat.columns for c in feat_names) else X_feat
            p_ml = float(model.predict_proba(X_input)[0, 1])
        else:
            p_ml = 0.0

        # 3. Unsupervised Anomaly Score (0–100)
        if self.anomaly_detector is not None:
            s_anom = float(self.anomaly_detector.score_100(X_feat)[0])
        else:
            s_anom = 0.0

        # 4. Behavioral Risk Signals
        eval_row = {**txn_dict, **X_feat.iloc[0].to_dict()}
        behav_res = self.signal_evaluator.evaluate_transaction(eval_row)
        s_behav = behav_res["behavioral_score"]
        signals = behav_res["triggered_signals"]

        # 5. Hybrid Risk Score Fusion
        hybrid_score = self.calculate_hybrid_score(p_ml, s_anom, s_behav)
        risk_level, action = self.classify_tier(hybrid_score)

        # 6. Plain-language explanation summary
        if len(signals) > 0:
            explanation = f"Flagged by {len(signals)} risk signal(s): " + "; ".join([s["signal"] for s in signals])
        elif hybrid_score >= 50:
            explanation = "Elevated statistical anomaly detected by unsupervised models."
        else:
            explanation = "Transaction behavior aligns with routine customer spending patterns."

        return {
            "transaction_id": txn_id,
            "fraud_probability": round(p_ml, 4),
            "anomaly_score": round(s_anom, 2),
            "behavioral_risk": round(s_behav, 2),
            "hybrid_score": hybrid_score,
            "risk_level": risk_level,
            "recommended_action": action,
            "risk_signals": signals,
            "explanation": explanation,
        }

    def predict_batch(
        self,
        p_ml_array: np.ndarray,
        s_anom_array: np.ndarray,
        s_behav_array: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Vectorized evaluation of multiple transactions."""
        p_ml_array = np.asarray(p_ml_array, dtype=float)
        s_anom_array = np.asarray(s_anom_array, dtype=float)
        s_behav_array = np.asarray(s_behav_array, dtype=float)

        raw = (self.w_ml * (p_ml_array * 100.0)) + (self.w_anom * s_anom_array) + (self.w_behav * s_behav_array)
        scores = np.clip(np.round(raw), 0, 100).astype(int)

        levels = np.full(scores.shape, "LOW RISK", dtype=object)
        levels[(scores > self.tier_low_max) & (scores <= self.tier_med_max)] = "MEDIUM RISK"
        levels[scores > self.tier_med_max] = "HIGH RISK"

        return {
            "hybrid_scores": scores,
            "risk_levels": levels,
        }

