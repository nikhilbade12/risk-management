"""Supervised Classification Model for Payment Fraud Detection.

Supports Random Forest, HistGradientBoosting, and XGBoost with cost-sensitive
class-imbalance handling. Outputs calibrated fraud probabilities P(Fraud).
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib

from src.config import SUPERVISED_MODEL_PATH, RANDOM_SEED


class FraudClassifier:
    """Supervised fraud classifier wrapper supporting multiple algorithms."""

    def __init__(self, model_type: str = "rf"):
        self.model_type = model_type.lower()
        self.model = None
        self.feature_names: Optional[list] = None
        self.is_fitted: bool = False
        self._init_model()

    def _init_model(self):
        if self.model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
        elif self.model_type == "hgb":
            self.model = HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=10,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )
        elif self.model_type == "lr":
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )
        elif self.model_type == "xgb":
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.08,
                    scale_pos_weight=20.0,
                    random_state=RANDOM_SEED,
                    eval_metric="logloss",
                )
            except ImportError:
                print("XGBoost not installed, falling back to HistGradientBoostingClassifier.")
                self.model_type = "hgb"
                self.model = HistGradientBoostingClassifier(
                    max_iter=150,
                    max_depth=10,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FraudClassifier":
        """Fits model on training data."""
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns 1D array of fraud probabilities P(Fraud=1)."""
        if not self.is_fitted:
            raise RuntimeError("FraudClassifier is not fitted yet.")
        probs = self.model.predict_proba(X)
        # Class 1 is fraud
        return probs[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary fraud class given decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, filepath: Path = SUPERVISED_MODEL_PATH) -> None:
        """Serializes fitted classifier and metadata."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, filepath)
        print(f"Supervised model ({self.model_type}) saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path = SUPERVISED_MODEL_PATH) -> "FraudClassifier":
        """Loads serialized classifier."""
        if not filepath.exists():
            raise FileNotFoundError(f"Supervised model artifact not found at {filepath}")
        payload = joblib.load(filepath)
        instance = cls(model_type=payload["model_type"])
        instance.model = payload["model"]
        instance.feature_names = payload["feature_names"]
        instance.is_fitted = payload["is_fitted"]
        return instance

