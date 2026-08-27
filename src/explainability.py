"""Explainable AI (XAI) Module for AI Risk Manager.
Razorpay AI Buildathon — AI Risk Manager Track.

Provides local and global explainability via SHAP (SHapley Additive exPlanations)
fused with unsupervised anomaly rationale and deterministic behavioral risk factors.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import shap
import joblib

from src.config import (
    MODELS_DIR,
    PREPROCESSOR_PATH,
    ANOMALY_MODEL_PATH,
)
from src.anomaly_detector import AnomalyDetector
from src.risk_signals import BehavioralRiskEvaluator
from src.risk_engine import HybridRiskEngine
from src.explanation_formatter import (
    FEATURE_DISPLAY_NAMES,
    get_feature_display_name,
    format_feature_context,
    format_anomaly_reason,
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class XAIExplainer:
    """Provides SHAP-based local/global explanations and multi-signal risk decomposition."""

    def __init__(
        self,
        ml_model: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        anomaly_detector: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        hybrid_engine: Optional[HybridRiskEngine] = None,
    ):
        self.ml_model = ml_model
        self.feature_names = feature_names or []
        self.anomaly_detector = anomaly_detector
        self.preprocessor = preprocessor
        self.hybrid_engine = hybrid_engine or HybridRiskEngine.load_default()
        self.signal_evaluator = BehavioralRiskEvaluator()

        # Initialize SHAP TreeExplainer if tree model available
        if self.ml_model is not None:
            try:
                self.shap_explainer = shap.TreeExplainer(self.ml_model)
            except Exception as e:
                print(f"[XAI] Warning: TreeExplainer fallback: {e}")
                self.shap_explainer = shap.Explainer(self.ml_model)
        else:
            self.shap_explainer = None

    @classmethod
    def load_default(cls) -> "XAIExplainer":
        """Factory method loading serialized artifacts from models/ directory."""
        adv_path = MODELS_DIR / "advanced_model.joblib"
        if not adv_path.exists():
            adv_path = MODELS_DIR / "advanced_model.pkl"

        anom_path = MODELS_DIR / "anomaly_model.joblib"
        if not anom_path.exists():
            anom_path = MODELS_DIR / "anomaly_model.pkl"

        preproc_path = PREPROCESSOR_PATH

        adv_payload = joblib.load(adv_path) if adv_path.exists() else None
        ml_model = adv_payload["model"] if adv_payload else None
        feat_names = adv_payload.get("feature_names", []) if adv_payload else []
        anom_detector = AnomalyDetector.load(anom_path) if anom_path.exists() else None
        preproc = joblib.load(preproc_path) if preproc_path.exists() else None
        hybrid_engine = HybridRiskEngine.load_default()

        return cls(
            ml_model=ml_model,
            feature_names=feat_names,
            anomaly_detector=anom_detector,
            preprocessor=preproc,
            hybrid_engine=hybrid_engine,
        )

    def explain_local_shap(self, X_input: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        """Calculates local SHAP values and identifies risk-increasing and protective factors."""
        if self.shap_explainer is None:
            return {"top_risk_factors": [], "protective_factors": [], "shap_values": []}

        # Calculate SHAP values for single instance
        shap_values_raw = self.shap_explainer(X_input)
        vals = shap_values_raw.values[0]
        # For binary classifier, handle 1D or 2D SHAP output
        if len(vals.shape) == 2:
            vals = vals[:, 1]  # positive fraud class

        base_val = float(shap_values_raw.base_values[0]) if hasattr(shap_values_raw, "base_values") else 0.0
        if isinstance(base_val, (np.ndarray, list)):
            base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])

        cols = list(X_input.columns)
        records = []
        for col, s_val, f_val in zip(cols, vals, X_input.iloc[0].values):
            records.append({
                "feature": col,
                "display_name": get_feature_display_name(col),
                "feature_value": float(f_val),
                "shap_value": float(s_val),
                "impact": "increases_risk" if s_val > 0 else "decreases_risk",
                "explanation": format_feature_context(col, float(f_val), float(s_val)),
            })

        # Sort risk-increasing factors (positive SHAP, largest first)
        risk_increasing = sorted([r for r in records if r["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)
        # Sort protective factors (negative SHAP, most negative first)
        protective = sorted([r for r in records if r["shap_value"] < 0], key=lambda x: x["shap_value"])

        return {
            "base_value": base_val,
            "all_features": records,
            "top_risk_factors": risk_increasing[:top_k],
            "protective_factors": protective[:top_k],
            "shap_values_array": vals,
        }

    def explain_transaction(self, txn_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive Multi-Signal Transaction Explanation API.
        Combines:
        1. Supervised ML Probability & Local SHAP contributions.
        2. Unsupervised Isolation Forest Anomaly Score & behavioral drivers.
        3. Deterministic Behavioral Risk Signals & Reason Codes.
        4. Hybrid Risk Score Breakdown (exact weights).
        """
        txn_df = pd.DataFrame([txn_dict])
        txn_id = str(txn_dict.get("trans_num", txn_dict.get("transaction_id", "TXN_SAMPLE")))

        # 1. Feature Extraction
        if self.preprocessor is not None:
            X_feat = self.preprocessor.transform(txn_df)
        else:
            X_feat = txn_df

        # 2. Supervised ML Probability & Local SHAP
        X_input = X_feat[self.feature_names] if all(c in X_feat.columns for c in self.feature_names) else X_feat
        p_ml = float(self.ml_model.predict_proba(X_input)[0, 1]) if self.ml_model is not None else 0.0
        shap_res = self.explain_local_shap(X_input)

        # 3. Unsupervised Anomaly Score
        s_anom = float(self.anomaly_detector.score_100(X_feat)[0]) if self.anomaly_detector is not None else 0.0
        anom_explanation = format_anomaly_reason(s_anom, X_feat.iloc[0].to_dict())

        # 4. Behavioral Heuristic Rules
        eval_row = {**txn_dict, **X_feat.iloc[0].to_dict()}
        behav_res = self.signal_evaluator.evaluate_transaction(eval_row)
        s_behav = behav_res["behavioral_score"]
        triggered_rules = behav_res["triggered_signals"]

        # 5. Hybrid Risk Score Fusion & Breakdown
        w_ml = self.hybrid_engine.w_ml
        w_anom = self.hybrid_engine.w_anom
        w_behav = self.hybrid_engine.w_behav

        ml_component = float(p_ml * 100.0)
        ml_contrib = float(w_ml * ml_component)
        anom_contrib = float(w_anom * s_anom)
        behav_contrib = float(w_behav * s_behav)

        hybrid_score = int(np.clip(np.round(ml_contrib + anom_contrib + behav_contrib), 0, 100))
        risk_level, action = self.hybrid_engine.classify_tier(hybrid_score)

        # 6. Plain-Language Synthesis Summary
        summary_reasons = []
        if p_ml >= 0.70:
            summary_reasons.append(f"High fraud probability ({p_ml*100:.1f}%) from supervised ML model.")
        if s_anom >= 65.0:
            summary_reasons.append(f"Transaction behavior is statistically unusual (Anomaly Score: {s_anom:.0f}/100).")
        if len(triggered_rules) > 0:
            rule_names = ", ".join([r["signal"] for r in triggered_rules[:2]])
            summary_reasons.append(f"Triggered behavioral risk rules ({rule_names}).")

        if not summary_reasons:
            if hybrid_score <= 35:
                summary_reasons.append("Transaction conforms to typical historical patterns across all dimensions.")
            else:
                summary_reasons.append("Elevated cumulative variance across multiple moderate risk signals.")

        return {
            "transaction_id": txn_id,
            "fraud_probability": round(p_ml, 4),
            "anomaly_score": round(s_anom, 2),
            "behavioral_risk": round(s_behav, 2),
            "hybrid_score": hybrid_score,
            "risk_level": risk_level,
            "recommended_action": action,
            "top_risk_factors": shap_res["top_risk_factors"],
            "protective_factors": shap_res["protective_factors"],
            "anomaly_explanation": anom_explanation,
            "behavioral_signals": triggered_rules,
            "score_breakdown": {
                "ml_weight_pct": round(w_ml * 100, 1),
                "anomaly_weight_pct": round(w_anom * 100, 1),
                "behavioral_weight_pct": round(w_behav * 100, 1),
                "ml_contribution": round(ml_contrib, 2),
                "anomaly_contribution": round(anom_contrib, 2),
                "behavioral_contribution": round(behav_contrib, 2),
                "hybrid_score": hybrid_score,
            },
            "explanation_summary": summary_reasons,
        }

    def generate_global_shap_visualizations(self, X_sample: pd.DataFrame) -> None:
        """Generates global SHAP summary plot, feature importance bar chart, and dependence plots."""
        if self.shap_explainer is None:
            return

        print("[XAI] Computing Global SHAP values across background sample...")
        shap_values_raw = self.shap_explainer(X_sample)
        vals = shap_values_raw.values
        if len(vals.shape) == 3:
            vals = vals[:, :, 1]  # binary class positive

        # 1. SHAP Feature Importance Bar Chart
        fig, ax = plt.subplots(figsize=(8, 5))
        mean_abs = np.mean(np.abs(vals), axis=0)
        sort_idx = np.argsort(mean_abs)
        display_names = [get_feature_display_name(c) for c in X_sample.columns]

        ax.barh(np.arange(len(sort_idx)), mean_abs[sort_idx], color="#0C2340", height=0.55)
        ax.set_yticks(np.arange(len(sort_idx)))
        ax.set_yticklabels([display_names[i] for i in sort_idx], fontsize=9)
        ax.set_xlabel("Mean |SHAP Value| (Average Model Impact)", fontweight="bold")
        ax.set_title("Global SHAP Feature Importance (Supervised XGBoost)", fontsize=12, fontweight="bold")
        for i, idx in enumerate(sort_idx):
            val = mean_abs[idx]
            ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=9, fontweight="bold")
        ax.set_xlim(0, max(mean_abs) * 1.18)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "shap_feature_importance.png", dpi=300)
        plt.close(fig)

        # 2. SHAP Beeswarm Summary Plot
        fig = plt.figure(figsize=(9, 5.5))
        # Use shap's summary plot with friendly column names
        X_sample_renamed = X_sample.rename(columns=FEATURE_DISPLAY_NAMES)
        shap.summary_plot(vals, X_sample_renamed, show=False, max_display=10)
        plt.title("SHAP Beeswarm Distribution (Feature Value vs. Risk Impact)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=300)
        plt.close()

        # 3. SHAP Dependence Plot for amt_to_user_avg_ratio
        if "amt_to_user_avg_ratio" in X_sample.columns:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ratio_idx = list(X_sample.columns).index("amt_to_user_avg_ratio")
            ax.scatter(X_sample["amt_to_user_avg_ratio"], vals[:, ratio_idx], color="#3395FF", alpha=0.6, s=25)
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("Cardholder Spending Ratio (Amount / User Average Spend)", fontweight="bold")
            ax.set_ylabel("SHAP Value (Impact on Log-Odds of Fraud)", fontweight="bold")
            ax.set_title("SHAP Dependence: Spending Ratio vs. Fraud Risk Impact", fontsize=12, fontweight="bold")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "shap_dependence_amount.png", dpi=300)
            plt.close(fig)

        print("    * Saved shap_feature_importance.png, shap_summary.png, shap_dependence_amount.png.")

    def plot_local_waterfall(self, txn_dict: Dict[str, Any], save_name: str = "shap_local_waterfall.png") -> None:
        """Creates a transaction-specific horizontal force/waterfall bar plot showing risk drivers."""
        txn_df = pd.DataFrame([txn_dict])
        X_feat = self.preprocessor.transform(txn_df) if self.preprocessor is not None else txn_df
        X_input = X_feat[self.feature_names] if all(c in X_feat.columns for c in self.feature_names) else X_feat

        shap_res = self.explain_local_shap(X_input, top_k=8)
        all_feats = shap_res["all_features"]
        # Sort by absolute SHAP value
        sorted_feats = sorted(all_feats, key=lambda x: abs(x["shap_value"]), reverse=True)[:8]
        sorted_feats.reverse()  # lowest first for horizontal bar

        names = [f["display_name"] for f in sorted_feats]
        shap_vals = [f["shap_value"] for f in sorted_feats]
        bar_colors = ["#FF4D4F" if v > 0 else "#3395FF" for v in shap_vals]

        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, shap_vals, color=bar_colors, height=0.55)
        ax.axvline(0, color="black", linestyle="-", linewidth=1.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9, fontweight="bold")
        ax.set_xlabel("SHAP Impact on Fraud Risk (Positive = Increases Risk, Negative = Protective)", fontweight="bold")
        ax.set_title(f"Local Transaction Risk Drivers: {txn_dict.get('trans_num', 'TXN')}", fontsize=12, fontweight="bold")

        for b, v in zip(bars, shap_vals):
            offset = 0.05 if v >= 0 else -0.05
            ha = "left" if v >= 0 else "right"
            ax.text(v + offset, b.get_y() + b.get_height()/2.0, f"{v:+.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")

        limit = max(abs(min(shap_vals)), abs(max(shap_vals))) * 1.3
        ax.set_xlim(-limit, limit)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / save_name, dpi=300)
        plt.close(fig)
        print(f"    * Saved local waterfall plot: {FIGURES_DIR / save_name}")
