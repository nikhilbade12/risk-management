"""Deterministic Risk Rules Engine for payment fraud detection.

Implements transparent, explainable heuristic checks based on domain risk patterns.
Outputs bounded risk scores [0.0, 1.0] and detailed human-readable reason codes.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd

from src.config import (
    RULE_HIGH_AMOUNT_THRESHOLD,
    RULE_HIGH_AMOUNT_RATIO,
    RULE_HIGH_VELOCITY_1H,
    RULE_HIGH_DISTANCE_KM,
    RULE_NIGHT_START_HOUR,
    RULE_NIGHT_END_HOUR,
    RULE_HIGH_RISK_CATEGORIES,
)


class RiskRule:
    """Represents an individual heuristic risk rule."""

    def __init__(self, rule_id: str, name: str, weight: float, description: str):
        self.rule_id = rule_id
        self.name = name
        self.weight = weight
        self.description = description

    def evaluate(self, txn: Dict[str, Any]) -> bool:
        """Returns True if the transaction violates this risk rule."""
        raise NotImplementedError


class RuleExtremeAmount(RiskRule):
    def __init__(self):
        super().__init__(
            rule_id="R1_EXTREME_AMOUNT",
            name="Extreme Spending Outlier",
            weight=0.30,
            description="Transaction amount is exceptionally high and exceeds 4x customer historical average.",
        )

    def evaluate(self, txn: Dict[str, Any]) -> bool:
        amt = float(txn.get("amt", 0.0))
        ratio = float(txn.get("amt_to_user_avg_ratio", 1.0))
        return amt >= RULE_HIGH_AMOUNT_THRESHOLD and ratio >= RULE_HIGH_AMOUNT_RATIO


class RuleVelocitySpike(RiskRule):
    def __init__(self):
        super().__init__(
            rule_id="R2_VELOCITY_SPIKE",
            name="Rapid Transaction Velocity Burst",
            weight=0.35,
            description="Rapid burst of 3 or more transactions on this card within a 60-minute window.",
        )

    def evaluate(self, txn: Dict[str, Any]) -> bool:
        v1h = float(txn.get("trans_velocity_1h", 0))
        return v1h >= RULE_HIGH_VELOCITY_1H


class RuleImpossibleTravel(RiskRule):
    def __init__(self):
        super().__init__(
            rule_id="R3_IMPOSSIBLE_TRAVEL",
            name="Geographical Distance Discrepancy",
            weight=0.30,
            description="Transaction initiated over 800 km away from cardholder's billing home location.",
        )

    def evaluate(self, txn: Dict[str, Any]) -> bool:
        dist = float(txn.get("haversine_distance_km", 0.0))
        return dist >= RULE_HIGH_DISTANCE_KM


class RuleLateNightHighValue(RiskRule):
    def __init__(self):
        super().__init__(
            rule_id="R4_LATE_NIGHT_HIGH_VALUE",
            name="High-Value Off-Hours Transaction",
            weight=0.25,
            description="Significant purchase executed during late night / early morning hours (12 AM - 5 AM).",
        )

    def evaluate(self, txn: Dict[str, Any]) -> bool:
        is_night = int(txn.get("is_night_transaction", 0))
        cat_ratio = float(txn.get("amt_to_cat_median_ratio", 1.0))
        return is_night == 1 and cat_ratio >= 3.0


class RuleHighRiskOnlineCategory(RiskRule):
    def __init__(self):
        super().__init__(
            rule_id="R5_HIGH_RISK_CATEGORY",
            name="Elevated Risk Merchant Category Spike",
            weight=0.20,
            description="High-value checkout in historically high-fraud categories (digital goods, online retail).",
        )

    def evaluate(self, txn: Dict[str, Any]) -> bool:
        cat = str(txn.get("category", "")).lower()
        amt = float(txn.get("amt", 0.0))
        return cat in RULE_HIGH_RISK_CATEGORIES and amt >= 500.0


class RulesEngine:
    """Composite rules engine evaluating all registered heuristic checks."""

    def __init__(self):
        self.rules: List[RiskRule] = [
            RuleExtremeAmount(),
            RuleVelocitySpike(),
            RuleImpossibleTravel(),
            RuleLateNightHighValue(),
            RuleHighRiskOnlineCategory(),
        ]

    def evaluate_single(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates all rules for a single transaction dictionary or Series."""
        triggered = []
        accumulated_score = 0.0

        for rule in self.rules:
            if rule.evaluate(txn):
                triggered.append({
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "weight": rule.weight,
                    "description": rule.description,
                })
                accumulated_score += rule.weight

        # Bounded rule risk score [0.0, 1.0]
        rule_score = min(1.0, accumulated_score)

        return {
            "rule_score": round(rule_score, 4),
            "triggered_rules": triggered,
            "rule_count": len(triggered),
        }

    def evaluate_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """Batch evaluates a DataFrame and returns a Series of rule scores [0.0, 1.0]."""
        scores = []
        for _, row in df.iterrows():
            res = self.evaluate_single(row.to_dict())
            scores.append(res["rule_score"])
        return pd.Series(scores, index=df.index, name="rule_score")

