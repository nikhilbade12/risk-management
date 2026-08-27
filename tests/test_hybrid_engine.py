"""Unit tests for the Hybrid Risk Engine."""

import unittest
from src.hybrid_engine import HybridRiskEngine


class TestHybridRiskEngine(unittest.TestCase):

    def setUp(self):
        self.engine = HybridRiskEngine(
            weight_ml=0.60,
            weight_anomaly=0.15,
            weight_rules=0.25,
            tier_low_max=35.0,
            tier_med_max=70.0,
        )

    def test_zero_risk_score(self):
        score = self.engine.calculate_score(0.0, 0.0, 0.0)
        self.assertEqual(score, 0)
        tier, action = self.engine.classify_tier(score)
        self.assertEqual(tier, "LOW RISK")

    def test_max_risk_score(self):
        score = self.engine.calculate_score(1.0, 1.0, 1.0)
        self.assertEqual(score, 100)
        tier, action = self.engine.classify_tier(score)
        self.assertEqual(tier, "HIGH RISK")

    def test_medium_tier_mapping(self):
        score = self.engine.calculate_score(0.5, 0.5, 0.5)
        self.assertEqual(score, 50)
        tier, action = self.engine.classify_tier(score)
        self.assertEqual(tier, "MEDIUM RISK")

    def test_weight_normalization(self):
        # Weights not summing to 1 initially should be normalized
        engine = HybridRiskEngine(weight_ml=6, weight_anomaly=2, weight_rules=2)
        self.assertAlmostEqual(engine.w_ml, 0.60)
        self.assertAlmostEqual(engine.w_anom, 0.20)
        self.assertAlmostEqual(engine.w_rules, 0.20)


if __name__ == "__main__":
    unittest.main()

