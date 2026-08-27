"""Unit tests for feature engineering."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime

from src.feature_engineering import haversine_distance, TransactionFeatureExtractor


class TestFeatureEngineering(unittest.TestCase):

    def test_haversine_zero_distance(self):
        # Distance between identical coordinates should be 0.0 km
        dist = haversine_distance(
            np.array([12.9716]),
            np.array([77.5946]),
            np.array([12.9716]),
            np.array([77.5946]),
        )
        self.assertAlmostEqual(dist[0], 0.0, places=3)

    def test_haversine_known_distance(self):
        # Bangalore (12.9716, 77.5946) to Mumbai (19.0760, 72.8777) ~ 840-850 km
        dist = haversine_distance(
            np.array([12.9716]),
            np.array([77.5946]),
            np.array([19.0760]),
            np.array([72.8777]),
        )
        self.assertTrue(800.0 < dist[0] < 900.0)

    def test_feature_extractor(self):
        records = [
            {
                "trans_date_trans_time": "2025-01-01 14:30:00",
                "cc_num": "4111222233334444",
                "merchant": "RupayMart",
                "category": "grocery_pos",
                "amt": 50.0,
                "lat": 12.97,
                "long": 77.59,
                "merch_lat": 12.98,
                "merch_long": 77.60,
                "dob": "1990-05-15",
                "is_fraud": 0,
            },
            {
                "trans_date_trans_time": "2025-01-01 14:45:00",
                "cc_num": "4111222233334444",
                "merchant": "FlipkartMerch",
                "category": "shopping_net",
                "amt": 250.0,
                "lat": 12.97,
                "long": 77.59,
                "merch_lat": 19.07,
                "merch_long": 72.87,
                "dob": "1990-05-15",
                "is_fraud": 1,
            },
        ]
        df = pd.DataFrame(records)
        extractor = TransactionFeatureExtractor()
        X = extractor.fit_transform(df)

        self.assertEqual(len(X), 2)
        self.assertIn("haversine_distance_km", X.columns)
        self.assertIn("amt_to_user_avg_ratio", X.columns)
        self.assertIn("customer_age", X.columns)
        self.assertIn("is_night_transaction", X.columns)
        self.assertTrue((X["customer_age"] >= 18).all())


if __name__ == "__main__":
    unittest.main()

