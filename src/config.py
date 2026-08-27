"""Configuration settings and constants for AI Risk Manager."""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Ensure essential directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Artifact file paths
SUPERVISED_MODEL_PATH = MODELS_DIR / "supervised_model.joblib"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METRICS_PATH = MODELS_DIR / "test_metrics.json"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"

# Data splitting & reproducibility
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Default Hybrid Risk Engine Weights (Subject to validation tuning)
DEFAULT_WEIGHT_ML = 0.60
DEFAULT_WEIGHT_ANOMALY = 0.15
DEFAULT_WEIGHT_RULES = 0.25

# Risk Tier Cutoffs (0 to 100)
TIER_LOW_MAX = 35.0
TIER_MEDIUM_MAX = 70.0

# Financial Cost Defaults (for false-positive vs false-negative optimization)
DEFAULT_CHARGEBACK_FEE = 20.0     # Merchant penalty fee per chargeback ($ / standard units)
DEFAULT_MERCHANT_MARGIN = 0.20    # 20% margin on lost legitimate sales
DEFAULT_CUSTOMER_FRICTION = 10.0  # Support ticket & churn penalty per false decline
DEFAULT_OPERATIONAL_COST = 5.0    # Investigation cost per flagged transaction

# Behavioral Rule Thresholds
RULE_HIGH_AMOUNT_THRESHOLD = 1500.0
RULE_HIGH_AMOUNT_RATIO = 4.0
RULE_HIGH_VELOCITY_1H = 3
RULE_HIGH_DISTANCE_KM = 800.0
RULE_NIGHT_START_HOUR = 0
RULE_NIGHT_END_HOUR = 5
RULE_HIGH_RISK_CATEGORIES = ["shopping_net", "misc_net", "shopping_pos"]

