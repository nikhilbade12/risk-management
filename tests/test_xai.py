"""Unit tests for Explainable AI (XAI) module and explanation formatter."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.explainability import XAIExplainer
from src.explanation_formatter import (
    get_feature_display_name,
    format_feature_context,
    format_anomaly_reason,
)


class TestXAIModule(unittest.TestCase):
    """Verifies XAIExplainer and Explanation Formatter functionality."""

    @classmethod
    def setUpClass(cls):
        cls.explainer = XAIExplainer.load_default()
        cls.sample_txn = {
            "trans_num": "TXN_TEST_1001",
            "cc_num": "4532000011112222",
            "amt": 750.0,
            "category": "misc_net",
            "city": "Dallas",
            "state": "TX",
            "lat": 32.7767,
            "long": -96.7970,
            "merch_lat": 34.0522,
            "merch_long": -118.2437,
            "trans_date_trans_time": "2020-05-15 02:30:00",
            "dob": "1980-01-01",
            "amt_to_user_avg_ratio": 6.5,
            "amt_to_cat_median_ratio": 5.0,
            "haversine_distance_km": 1900.0,
            "trans_velocity_1h": 2,
            "trans_velocity_24h": 4,
            "is_night_transaction": 1,
            "hour": 2,
            "day_of_week": 4,
            "customer_age": 40,
        }

    def test_explainer_initialization(self):
        """Checks that model, preprocessor, and TreeExplainer are loaded."""
        self.assertIsNotNone(self.explainer.ml_model)
        self.assertIsNotNone(self.explainer.shap_explainer)
        self.assertGreater(len(self.explainer.feature_names), 0)

    def test_explain_transaction_schema(self):
        """Checks that explain_transaction returns all required keys."""
        res = self.explainer.explain_transaction(self.sample_txn)
        required_keys = [
            "transaction_id",
            "fraud_probability",
            "anomaly_score",
            "behavioral_risk",
            "hybrid_score",
            "risk_level",
            "recommended_action",
            "top_risk_factors",
            "protective_factors",
            "anomaly_explanation",
            "score_breakdown",
        ]
        for k in required_keys:
            self.assertIn(k, res, f"Missing key '{k}' in explanation result")

    def test_score_breakdown_consistency(self):
        """Checks that hybrid score decomposition mathematically aligns."""
        res = self.explainer.explain_transaction(self.sample_txn)
        bd = res["score_breakdown"]
        self.assertEqual(bd["ml_weight_pct"], 50.0)
        self.assertEqual(bd["anomaly_weight_pct"], 25.0)
        self.assertEqual(bd["behavioral_weight_pct"], 25.0)
        expected_score = int(np.clip(np.round(bd["ml_contribution"] + bd["anomaly_contribution"] + bd["behavioral_contribution"]), 0, 100))
        self.assertEqual(res["hybrid_score"], expected_score)

    def test_formatter_names(self):
        """Checks that feature display names are mapped to clean titles."""
        self.assertEqual(get_feature_display_name("amt"), "Transaction Amount")
        self.assertEqual(get_feature_display_name("amt_to_user_avg_ratio"), "Cardholder Spending Ratio")
        self.assertEqual(get_feature_display_name("haversine_distance_km"), "Terminal Geolocation Distance")

    def test_formatter_context_strings(self):
        """Checks contextual explanations for positive and negative impacts."""
        ctx_pos = format_feature_context("amt", 850.0, 0.45)
        self.assertIn("elevated fraud risk", ctx_pos)
        ctx_neg = format_feature_context("amt", 25.0, -0.30)
        self.assertIn("reduced risk", ctx_neg)


if __name__ == "__main__":
    unittest.main()
