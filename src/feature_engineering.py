"""Feature Engineering pipeline for transaction risk detection.

Computes behavioral, geographical, velocity, and temporal signals
with zero data leakage between train and test partitions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Computes the great-circle distance between two points on the Earth (in km)
    using the Haversine formula.
    """
    R = 6371.0  # Earth radius in kilometers

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


class TransactionFeatureExtractor:
    """
    Stateful feature extractor that fits baseline customer & category distributions
    strictly on the training dataset, and safely transforms train, val, and test splits.
    """

    def __init__(self):
        self.user_historical_mean: Dict[str, float] = {}
        self.category_median_amt: Dict[str, float] = {}
        self.category_fraud_rate: Dict[str, float] = {}
        self.global_mean_amt: float = 50.0
        self.global_median_amt: float = 40.0
        self.global_fraud_rate: float = 0.015
        self.feature_columns = [
            "amt",
            "haversine_distance_km",
            "amt_to_user_avg_ratio",
            "amt_to_cat_median_ratio",
            "trans_velocity_1h",
            "trans_velocity_24h",
            "is_night_transaction",
            "hour",
            "day_of_week",
            "customer_age",
            "category_fraud_rate",
        ]

    def fit(self, df: pd.DataFrame) -> "TransactionFeatureExtractor":
        """Learns reference distributions from training data only."""
        # Calculate global baselines
        self.global_mean_amt = float(df["amt"].mean())
        self.global_median_amt = float(df["amt"].median())
        if "is_fraud" in df.columns:
            self.global_fraud_rate = float(df["is_fraud"].mean())

        # Customer average amounts
        user_grp = df.groupby("cc_num")["amt"].mean()
        self.user_historical_mean = user_grp.to_dict()

        # Category median amounts
        cat_amt_grp = df.groupby("category")["amt"].median()
        self.category_median_amt = cat_amt_grp.to_dict()

        # Category fraud rate (target encoding)
        if "is_fraud" in df.columns:
            cat_fraud_grp = df.groupby("category")["is_fraud"].mean()
            self.category_fraud_rate = cat_fraud_grp.to_dict()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw transactions into risk feature dataframe."""
        df_out = df.copy()

        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(df_out["trans_date_trans_time"]):
            df_out["trans_date_trans_time"] = pd.to_datetime(df_out["trans_date_trans_time"])

        # 1. Temporal signals
        df_out["hour"] = df_out["trans_date_trans_time"].dt.hour
        df_out["day_of_week"] = df_out["trans_date_trans_time"].dt.dayofweek
        df_out["is_night_transaction"] = df_out["hour"].apply(lambda h: 1 if 0 <= h <= 5 else 0)

        # 2. Customer demographics
        if "dob" in df_out.columns:
            dob_dt = pd.to_datetime(df_out["dob"], errors="coerce")
            df_out["customer_age"] = df_out["trans_date_trans_time"].dt.year - dob_dt.dt.year
            df_out["customer_age"] = df_out["customer_age"].fillna(40).clip(18, 100)
        else:
            df_out["customer_age"] = 40.0

        # 3. Geographical distance (Haversine)
        df_out["haversine_distance_km"] = haversine_distance(
            df_out["lat"].values,
            df_out["long"].values,
            df_out["merch_lat"].values,
            df_out["merch_long"].values,
        )

        # 4. Spending Deviation Ratios
        if "cc_num" in df_out.columns:
            user_means = df_out["cc_num"].map(self.user_historical_mean).fillna(self.global_mean_amt)
        else:
            user_means = pd.Series(self.global_mean_amt, index=df_out.index)
        if "amt_to_user_avg_ratio" not in df_out.columns:
            df_out["amt_to_user_avg_ratio"] = df_out["amt"] / (user_means + 1e-5)

        if "amt_to_cat_median_ratio" not in df_out.columns:
            cat_medians = df_out["category"].map(self.category_median_amt).fillna(self.global_median_amt)
            df_out["amt_to_cat_median_ratio"] = df_out["amt"] / (cat_medians + 1e-5)

        # 5. Category historical fraud rate
        if "category_fraud_rate" not in df_out.columns:
            df_out["category_fraud_rate"] = df_out["category"].map(self.category_fraud_rate).fillna(self.global_fraud_rate)

        # 6. Transaction velocity (Rolling 1-hour and 24-hour transaction count)
        if "trans_velocity_1h" not in df_out.columns or "trans_velocity_24h" not in df_out.columns:
            if "cc_num" not in df_out.columns:
                df_out["cc_num"] = "UNKNOWN_CARD"
            df_out["__orig_idx"] = np.arange(len(df_out))
            # Compute velocity chronologically by cardholder
            df_sorted = df_out.sort_values(by=["cc_num", "trans_date_trans_time"]).copy()

            # Groupby rolling requires setting trans_date_trans_time as index
            df_sorted = df_sorted.set_index("trans_date_trans_time")
            
            # Velocity in 1 hour (exclusive of current transaction, so -1)
            v1h = (
                df_sorted.groupby("cc_num")["amt"]
                .rolling("1h")
                .count()
                .reset_index(level=0, drop=True)
                - 1
            ).clip(lower=0)

            v24h = (
                df_sorted.groupby("cc_num")["amt"]
                .rolling("24h")
                .count()
                .reset_index(level=0, drop=True)
                - 1
            ).clip(lower=0)

            df_sorted["trans_velocity_1h"] = v1h.values
            df_sorted["trans_velocity_24h"] = v24h.values

            # Restore original row order
            df_out = df_sorted.sort_values(by="__orig_idx").drop(columns=["__orig_idx"])

        # Select only the model feature columns
        return df_out[self.feature_columns]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits extractor and transforms training data."""
        return self.fit(df).transform(df)
