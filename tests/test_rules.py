"""Unit tests for deterministic risk rules."""

import unittest
from src.rules_engine import (
    RulesEngine,
    RuleExtremeAmount,
    RuleVelocitySpike,
    RuleImpossibleTravel,
    RuleLateNightHighValue,
)


class TestRulesEngine(unittest.TestCase):

    def setUp(self):
        self.engine = RulesEngine()

    def test_extreme_amount_rule(self):
        rule = RuleExtremeAmount()
        # Normal transaction
        self.assertFalse(rule.evaluate({"amt": 80.0, "amt_to_user_avg_ratio": 1.2}))
        # Extreme transaction
        self.assertTrue(rule.evaluate({"amt": 2500.0, "amt_to_user_avg_ratio": 5.5}))

    def test_velocity_spike_rule(self):
        rule = RuleVelocitySpike()
        self.assertFalse(rule.evaluate({"trans_velocity_1h": 1}))
        self.assertTrue(rule.evaluate({"trans_velocity_1h": 4}))

    def test_impossible_travel_rule(self):
        rule = RuleImpossibleTravel()
        self.assertFalse(rule.evaluate({"haversine_distance_km": 25.0}))
        self.assertTrue(rule.evaluate({"haversine_distance_km": 1200.0}))

    def test_late_night_rule(self):
        rule = RuleLateNightHighValue()
        self.assertFalse(rule.evaluate({"is_night_transaction": 0, "amt_to_cat_median_ratio": 4.0}))
        self.assertTrue(rule.evaluate({"is_night_transaction": 1, "amt_to_cat_median_ratio": 3.5}))

    def test_engine_score_bounds(self):
        # Transaction violating multiple rules
        txn = {
            "amt": 3000.0,
            "amt_to_user_avg_ratio": 8.0,
            "trans_velocity_1h": 5,
            "haversine_distance_km": 1500.0,
            "is_night_transaction": 1,
            "amt_to_cat_median_ratio": 5.0,
            "category": "shopping_net",
        }
        res = self.engine.evaluate_single(txn)
        self.assertGreaterEqual(res["rule_score"], 0.0)
        self.assertLessEqual(res["rule_score"], 1.0)
        self.assertGreaterEqual(len(res["triggered_rules"]), 3)


if __name__ == "__main__":
    unittest.main()

