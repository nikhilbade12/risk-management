"""Unit tests for Streamlit application backend components.
Verifies loading, preprocessing, hybrid risk evaluation, and XAI formatting without requiring a live browser.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.risk_engine import HybridRiskEngine
from src.explainability import XAIExplainer
from src.config import MODELS_DIR, PROCESSED_DATA_DIR


class TestAppBackendComponents(unittest.TestCase):
    """Tests all backend modules and components utilized by app.py."""

    @classmethod
    def setUpClass(cls):
        cls.engine = HybridRiskEngine.load_default()
        cls.explainer = XAIExplainer.load_default()
        cls.val_path = PROCESSED_DATA_DIR / "val.csv"
        if cls.val_path.exists():
            cls.val_sample = pd.read_csv(cls.val_path).iloc[0].to_dict()
        else:
            cls.val_sample = {
                "trans_num": "TXN_TEST_APP",
                "cc_num": "4532000011112222",
                "amt": 85.0,
                "category": "grocery_pos",
                "hour": 14,
                "day_of_week": 2,
                "customer_age": 35,
                "haversine_distance_km": 5.0,
                "amt_to_user_avg_ratio": 1.0,
                "amt_to_cat_median_ratio": 1.0,
                "trans_velocity_1h": 0,
                "trans_velocity_24h": 1,
                "is_night_transaction": 0,
                "lat": 32.7767,
                "long": -96.7970,
                "merch_lat": 32.7800,
                "merch_long": -96.7900,
                "trans_date_trans_time": "2020-05-15 14:00:00",
            }

    def test_engine_load(self):
        """Verifies that HybridRiskEngine loads properly."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.w_ml, 0.50)
        self.assertEqual(self.engine.w_anom, 0.25)
        self.assertEqual(self.engine.w_behav, 0.25)

    def test_explainer_load(self):
        """Verifies that XAIExplainer loads properly."""
        self.assertIsNotNone(self.explainer)
        self.assertIsNotNone(self.explainer.shap_explainer)

    def test_risk_prediction_output(self):
        """Checks risk prediction range and tier classification."""
        res = self.engine.predict_risk(self.val_sample)
        self.assertIn("hybrid_score", res)
        self.assertIn("risk_level", res)
        self.assertIn("recommended_action", res)

        # Check bounds
        self.assertGreaterEqual(res["hybrid_score"], 0)
        self.assertLessEqual(res["hybrid_score"], 100)
        self.assertIn(res["risk_level"], ["LOW RISK", "MEDIUM RISK", "HIGH RISK"])

    def test_xai_explanation_output(self):
        """Checks full transaction explanation schema."""
        exp = self.explainer.explain_transaction(self.val_sample)
        self.assertIn("top_risk_factors", exp)
        self.assertIn("protective_factors", exp)
        self.assertIn("score_breakdown", exp)
        self.assertIn("anomaly_explanation", exp)

        # Check score decomposition
        bd = exp["score_breakdown"]
        self.assertIn("ml_contribution", bd)
        self.assertIn("anomaly_contribution", bd)
        self.assertIn("behavioral_contribution", bd)

    def test_invalid_input_handling(self):
        """Verifies that unusual or extreme values don't crash the risk engine."""
        extreme_sample = {
            "trans_num": "TXN_EXTREME",
            "cc_num": "9999999999999999",
            "amt": 99999.0,
            "category": "unknown_cat",
            "hour": 23,
            "day_of_week": 0,
            "customer_age": 100,
            "haversine_distance_km": 10000.0,
            "amt_to_user_avg_ratio": 50.0,
            "amt_to_cat_median_ratio": 50.0,
            "trans_velocity_1h": 20,
            "trans_velocity_24h": 50,
            "is_night_transaction": 1,
            "lat": 0.0,
            "long": 0.0,
            "merch_lat": 90.0,
            "merch_long": 180.0,
            "trans_date_trans_time": "2020-01-01 23:59:59",
        }
        res = self.engine.predict_risk(extreme_sample)
        self.assertGreaterEqual(res["hybrid_score"], 70)
        self.assertEqual(res["risk_level"], "HIGH RISK")


if __name__ == "__main__":
    unittest.main()

