"""Unsupervised Anomaly Detection using Isolation Forest.
Razorpay AI Buildathon — AI Risk Manager Track.

Detects novel, unusual, and out-of-distribution transaction patterns without using past fraud labels.
Maps raw Isolation Forest path length decision scores into a calibrated Anomaly Score: 0–100.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

from src.config import ANOMALY_MODEL_PATH, RANDOM_SEED

# Default behavioral features strictly excluding fraud label and target-encoded features
ANOMALY_FEATURE_COLS = [
    "amt",
    "haversine_distance_km",
    "amt_to_user_avg_ratio",
    "amt_to_cat_median_ratio",
    "trans_velocity_1h",
    "trans_velocity_24h",
    "is_night_transaction",
    "hour",
    "customer_age",
]


class AnomalyDetector:
    """Isolation Forest anomaly detection wrapper with calibrated 0–100 score mapping."""

    def __init__(
        self,
        n_estimators: int = 120,
        contamination: float = 0.03,
        random_state: int = RANDOM_SEED,
        feature_cols: Optional[List[str]] = None,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.feature_cols = feature_cols or ANOMALY_FEATURE_COLS
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.min_score: float = -0.5
        self.max_score: float = 0.5
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "AnomalyDetector":
        """Fits the Isolation Forest strictly on unsupervised behavioral feature matrix."""
        # Ensure only unsupervised features are used
        X_sub = X[self.feature_cols] if all(col in X.columns for col in self.feature_cols) else X
        self.model.fit(X_sub)

        # Calibrate decision scores using percentiles on training distribution
        raw_scores = self.model.decision_function(X_sub)
        self.min_score = float(np.percentile(raw_scores, 0.5))
        self.max_score = float(np.percentile(raw_scores, 99.5))
        if self.max_score <= self.min_score:
            self.max_score = self.min_score + 1.0
        self.is_fitted = True
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculates normalized anomaly scores between 0.0 (normal) and 1.0 (highly anomalous).
        Lower decision_function values correspond to higher anomaly scores.
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be fitted before calling score().")

        X_sub = X[self.feature_cols] if all(col in X.columns for col in self.feature_cols) else X
        raw_scores = self.model.decision_function(X_sub)
        # Invert: lower raw score -> higher anomaly score
        normalized = (self.max_score - raw_scores) / (self.max_score - self.min_score + 1e-6)
        return np.clip(normalized, 0.0, 1.0)

    def score_100(self, X: pd.DataFrame) -> np.ndarray:
        """Returns calibrated Anomaly Score on a 0 to 100 scale."""
        return np.round(self.score(X) * 100.0, 2)

    def predict_anomaly(self, X: pd.DataFrame, threshold_score: float = 65.0) -> np.ndarray:
        """Predicts binary anomaly flag (1 = Anomalous, 0 = Normal) based on calibrated threshold."""
        scores = self.score_100(X)
        return (scores >= threshold_score).astype(int)

    def save(self, filepath: Path = ANOMALY_MODEL_PATH) -> None:
        """Saves model artifact."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        pkl_path = filepath.with_suffix(".pkl")
        joblib.dump(self, pkl_path)
        print(f"Anomaly detector saved to {filepath} and {pkl_path}")

    @classmethod
    def load(cls, filepath: Path = ANOMALY_MODEL_PATH) -> "AnomalyDetector":
        """Loads model artifact."""
        if not filepath.exists():
            pkl_path = filepath.with_suffix(".pkl")
            if pkl_path.exists():
                filepath = pkl_path
            else:
                raise FileNotFoundError(f"Anomaly model artifact not found at {filepath}")
        return joblib.load(filepath)
