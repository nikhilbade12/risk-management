"""Behavioral Risk Signals Module for AI Risk Manager.
Razorpay AI Buildathon — AI Risk Manager Track.

Evaluates explainable, deterministic heuristic signals from transaction data:
1. Extreme Amount Outlier: Order amount >= $500 and exceeds 3.5x category median.
2. Spending Spike: Current amount exceeds 4.0x customer historical average.
3. Impossible Travel / Distance: Distance between billing home and merchant terminal >= 300 km.
4. Velocity Spike: Rapid authorization bursts (>= 2 txns in 1 hour or >= 4 txns in 24 hours).
5. Off-Hours Authorization: Authorizations executed between midnight and 5:00 AM.
6. High-Risk Merchant Category: Online checkouts in dispute-prone sectors (misc_net, travel, etc.).

Outputs individual signal triggers with human-readable explanations and computes a normalized
Behavioral Risk Score: 0–100.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.config import (
    RULE_HIGH_AMOUNT_THRESHOLD,
    RULE_HIGH_AMOUNT_RATIO,
    RULE_HIGH_VELOCITY_1H,
    RULE_HIGH_DISTANCE_KM,
    RULE_HIGH_RISK_CATEGORIES,
)


class BehavioralRiskEvaluator:
    """Evaluates rule-based behavioral risk signals and outputs explainable reason codes."""

    def __init__(self):
        self.signal_definitions = [
            {
                "id": "SIG_EXTREME_AMOUNT",
                "name": "Extreme Amount Outlier",
                "weight": 25.0,
                "description": "Transaction amount exceeds $500 and is significantly higher than category median.",
            },
            {
                "id": "SIG_SPEND_SPIKE",
                "name": "Cardholder Spend Spike",
                "weight": 30.0,
                "description": "Order amount is more than 4.0x the customer's historical average spend.",
            },
            {
                "id": "SIG_DISTANCE_JUMP",
                "name": "Geographical Jump",
                "weight": 20.0,
                "description": "Merchant terminal is located over 300 km from cardholder billing residence.",
            },
            {
                "id": "SIG_VELOCITY_BURST",
                "name": "High Velocity Burst",
                "weight": 20.0,
                "description": "Multiple authorizations detected on this card within a short time window.",
            },
            {
                "id": "SIG_OFF_HOURS",
                "name": "Late Night / Off-Hours",
                "weight": 15.0,
                "description": "Authorization occurred during high-risk night hours (12:00 AM - 05:00 AM).",
            },
            {
                "id": "SIG_HIGH_RISK_CATEGORY",
                "name": "High-Dispute Sector",
                "weight": 10.0,
                "description": "Merchant operates in a sector with historically elevated chargeback rates.",
            },
        ]

    def evaluate_transaction(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates a single transaction dictionary and returns structured risk signals."""
        amt = float(txn.get("amt", 0.0))
        user_ratio = float(txn.get("amt_to_user_avg_ratio", 1.0))
        cat_ratio = float(txn.get("amt_to_cat_median_ratio", 1.0))
        dist_km = float(txn.get("haversine_distance_km", 0.0))
        v1h = float(txn.get("trans_velocity_1h", 0))
        v24h = float(txn.get("trans_velocity_24h", 0))
        is_night = int(txn.get("is_night_transaction", 0))
        category = str(txn.get("category", "")).lower()

        signals = []
        accumulated_score = 0.0

        # Signal 1: Extreme Amount
        sig1_triggered = (amt >= 500.0) and (cat_ratio >= 3.0)
        if sig1_triggered:
            severity = min(1.0, (amt / 1000.0) * 0.8)
            accumulated_score += 25.0 * severity
            signals.append({
                "signal": "Extreme Amount Outlier",
                "triggered": True,
                "severity": round(severity, 2),
                "explanation": f"Transaction amount (${amt:.2f}) exceeds $500 and is {cat_ratio:.1f}x the category median.",
            })

        # Signal 2: Spending Spike
        sig2_triggered = user_ratio >= 4.0
        if sig2_triggered:
            severity = min(1.0, (user_ratio / 8.0))
            accumulated_score += 30.0 * severity
            signals.append({
                "signal": "Cardholder Spend Spike",
                "triggered": True,
                "severity": round(severity, 2),
                "explanation": f"Order value is {user_ratio:.1f}x higher than this cardholder's baseline average spend.",
            })

        # Signal 3: Geographical Jump
        sig3_triggered = dist_km >= 300.0
        if sig3_triggered:
            severity = min(1.0, (dist_km / 1200.0))
            accumulated_score += 20.0 * severity
            signals.append({
                "signal": "Geographical Jump",
                "triggered": True,
                "severity": round(severity, 2),
                "explanation": f"Terminal location is {dist_km:.1f} km away from registered billing address.",
            })

        # Signal 4: Velocity Burst
        sig4_triggered = (v1h >= 2) or (v24h >= 4)
        if sig4_triggered:
            severity = min(1.0, (v1h / 3.0) if v1h >= 2 else (v24h / 6.0))
            accumulated_score += 20.0 * severity
            signals.append({
                "signal": "High Velocity Burst",
                "triggered": True,
                "severity": round(severity, 2),
                "explanation": f"Card velocity spike: {int(v1h)} txns in last hour, {int(v24h)} txns in 24 hours.",
            })

        # Signal 5: Off-Hours Night Authorization
        sig5_triggered = is_night == 1
        if sig5_triggered:
            severity = 0.7 if amt < 200 else 1.0
            accumulated_score += 15.0 * severity
            signals.append({
                "signal": "Late Night / Off-Hours",
                "triggered": True,
                "severity": round(severity, 2),
                "explanation": "Transaction authorized during vulnerable midnight-to-dawn window (12 AM - 5 AM).",
            })

        # Signal 6: High Risk Category
        sig6_triggered = (category in RULE_HIGH_RISK_CATEGORIES) and (amt >= 250.0)
        if sig6_triggered:
            severity = 0.8
            accumulated_score += 10.0 * severity
            signals.append({
                "signal": "High-Dispute Sector",
                "triggered": True,
                "severity": round(severity, 2),
                "explanation": f"High nominal order (${amt:.2f}) placed in elevated chargeback sector '{category}'.",
            })

        # Normalize score to 0–100 scale
        behavioral_score = float(np.clip(np.round(accumulated_score, 1), 0.0, 100.0))

        return {
            "behavioral_score": behavioral_score,
            "triggered_signals": signals,
            "signal_count": len(signals),
        }

    def evaluate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch evaluates a DataFrame and returns behavioral scores and trigger counts."""
        scores = []
        counts = []
        for _, row in df.iterrows():
            res = self.evaluate_transaction(row.to_dict())
            scores.append(res["behavioral_score"])
            counts.append(res["signal_count"])

        return pd.DataFrame({
            "behavioral_score": scores,
            "signal_count": counts,
        }, index=df.index)

