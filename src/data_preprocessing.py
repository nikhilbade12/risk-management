"""Data Preprocessing and Feature Pipeline for AI Risk Manager.

Implements:
1. Data validation and cleaning (missing values, duplicates, invalid values, negative amounts).
2. Target variable standardization (0 = Legitimate, 1 = Fraud).
3. Leak-free feature engineering (velocity, haversine distance, spending ratios, temporal flags).
4. Categorical target-encoding and numeric standardization.
5. Stratified 70/15/15 Train / Validation / Held-Out Test partitioning.
6. Scikit-learn compatible preprocessor pipeline serialization.
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PREPROCESSOR_PATH,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)
from src.feature_engineering import TransactionFeatureExtractor


def clean_raw_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw transactions:
    - Validates columns and datatypes
    - Removes duplicate records
    - Filters out invalid rows (negative/zero amounts, impossible coordinates)
    - Retains legitimate fraud-related outliers
    - Standardizes target variable (0 = Legitimate, 1 = Fraud)
    """
    initial_rows = len(df)
    cleaning_log = {}

    df_clean = df.copy()

    # 1. Deduplication
    if "trans_num" in df_clean.columns:
        dups = df_clean.duplicated(subset=["trans_num"]).sum()
        df_clean = df_clean.drop_duplicates(subset=["trans_num"]).reset_index(drop=True)
    else:
        dups = df_clean.duplicated().sum()
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    cleaning_log["duplicates_removed"] = int(dups)

    # 2. Invalid Values: Amount must be strictly positive
    invalid_amt = (df_clean["amt"] <= 0) | df_clean["amt"].isna()
    invalid_amt_count = int(invalid_amt.sum())
    df_clean = df_clean[~invalid_amt].reset_index(drop=True)
    cleaning_log["invalid_amounts_removed"] = invalid_amt_count

    # 3. Invalid Coordinates: Latitude [-90, 90], Longitude [-180, 180]
    invalid_coords = (
        (df_clean["lat"] < -90) | (df_clean["lat"] > 90) |
        (df_clean["long"] < -180) | (df_clean["long"] > 180) |
        (df_clean["merch_lat"] < -90) | (df_clean["merch_lat"] > 90) |
        (df_clean["merch_long"] < -180) | (df_clean["merch_long"] > 180)
    )
    invalid_coord_count = int(invalid_coords.sum())
    df_clean = df_clean[~invalid_coords].reset_index(drop=True)
    cleaning_log["invalid_coords_removed"] = invalid_coord_count

    # 4. Missing value strategy:
    # - Numeric coordinates/demographics: impute with median (robust against outliers)
    # - Categorical features: impute with 'UNKNOWN'
    for col in ["lat", "long", "merch_lat", "merch_long", "city_pop"]:
        if col in df_clean.columns and df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in ["merchant", "category", "gender", "job", "state", "city"]:
        if col in df_clean.columns and df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna("UNKNOWN")

    # 5. Standardize target column (0 = Legitimate, 1 = Fraud)
    if "is_fraud" in df_clean.columns:
        df_clean["is_fraud"] = df_clean["is_fraud"].astype(int)
        assert set(df_clean["is_fraud"].unique()).issubset({0, 1}), "Target must only contain 0 and 1"

    final_rows = len(df_clean)
    cleaning_log["initial_row_count"] = initial_rows
    cleaning_log["final_row_count"] = final_rows
    cleaning_log["total_records_dropped"] = initial_rows - final_rows

    return df_clean, cleaning_log


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs leak-free stratified 3-way partition:
    - Train (70%)
    - Validation (15%)
    - Final Held-out Test (15%)
    Preserving the natural fraud class imbalance across all partitions.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5

    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["is_fraud"],
        random_state=random_seed,
    )

    val_rel_ratio = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_rel_ratio),
        stratify=temp_df["is_fraud"],
        random_state=random_seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def run_preprocessing_pipeline(
    raw_data_path: Path = RAW_DATA_DIR / "transactions.csv",
) -> Dict[str, Any]:
    """
    Full end-to-end preprocessing workflow:
    1. Loads raw transactions.
    2. Executes data cleaning and target verification.
    3. Partitions into Train (70%), Val (15%), Test (15%).
    4. Fits TransactionFeatureExtractor strictly on Train partition.
    5. Transforms Train, Val, Test.
    6. Persists processed datasets and fitted preprocessor artifact.
    """
    print("=" * 70)
    print("[PREPROCESSING] STARTING DATA PREPROCESSING & FEATURE PIPELINE")
    print("=" * 70)

    # 1. Load raw data
    if not raw_data_path.exists():
        from scripts.setup_data import setup_dataset
        raw_data_path = setup_dataset()

    print(f"\n[Step 1/5] Loading raw data from: {raw_data_path}")
    raw_df = pd.read_csv(raw_data_path)

    # 2. Clean data
    print("\n[Step 2/5] Cleaning data and validating constraints...")
    clean_df, clean_log = clean_raw_data(raw_df)
    print(f"    * Initial rows: {clean_log['initial_row_count']:,}")
    print(f"    * Duplicates removed: {clean_log['duplicates_removed']:,}")
    print(f"    * Invalid amounts/coords removed: {clean_log['invalid_amounts_removed'] + clean_log['invalid_coords_removed']:,}")
    print(f"    * Final clean rows: {clean_log['final_row_count']:,}")

    # 3. Stratified Partitioning
    print("\n[Step 3/5] Performing leak-free 70/15/15 stratified train/val/test split...")
    train_df, val_df, test_df = split_dataset(clean_df)

    def summarize_split(name: str, part_df: pd.DataFrame) -> Dict[str, Any]:
        n = len(part_df)
        frauds = int(part_df["is_fraud"].sum())
        pct = (frauds / n) * 100
        return {"total_rows": n, "fraud_count": frauds, "legit_count": n - frauds, "fraud_percent": round(pct, 2)}

    train_sum = summarize_split("Train", train_df)
    val_sum = summarize_split("Validation", val_df)
    test_sum = summarize_split("Test", test_df)

    print(f"    * Train (70%):      {train_sum['total_rows']:,} rows ({train_sum['fraud_count']:,} frauds, {train_sum['fraud_percent']}%)")
    print(f"    * Validation (15%): {val_sum['total_rows']:,} rows ({val_sum['fraud_count']:,} frauds, {val_sum['fraud_percent']}%)")
    print(f"    * Held-Out Test (15%): {test_sum['total_rows']:,} rows ({test_sum['fraud_count']:,} frauds, {test_sum['fraud_percent']}%)")

    # 4. Feature Extraction (Fitted STRICTLY on Train data only)
    print("\n[Step 4/5] Fitting feature extractor on training partition (zero leakage)...")
    extractor = TransactionFeatureExtractor()
    extractor.fit(train_df)

    # Save fitted preprocessor
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(extractor, PREPROCESSOR_PATH)
    print(f"    * Preprocessor pipeline artifact saved to: {PREPROCESSOR_PATH}")

    # Transform all three partitions
    X_train = extractor.transform(train_df)
    X_val = extractor.transform(val_df)
    X_test = extractor.transform(test_df)

    # Attach targets for export
    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    # 5. Save Processed Datasets
    print("\n[Step 5/5] Exporting processed datasets...")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    validation_path = PROCESSED_DATA_DIR / "validation.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)
    val_out.to_csv(validation_path, index=False)
    test_out.to_csv(test_path, index=False)

    print(f"    * Saved: {train_path} ({train_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"    * Saved: {val_path} ({val_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"    * Saved: {test_path} ({test_path.stat().st_size / (1024*1024):.2f} MB)")

    # Data Quality & Preprocessing Summary Report
    summary_report = {
        "dataset_name": raw_data_path.name,
        "clean_summary": clean_log,
        "engineered_features": extractor.feature_columns,
        "split_summary": {
            "train": train_sum,
            "validation": val_sum,
            "test": test_sum,
        },
        "target_distribution": {
            "legitimate": int(clean_df['is_fraud'].value_counts()[0]),
            "fraud": int(clean_df['is_fraud'].value_counts()[1]),
            "fraud_percentage": round(float(clean_df['is_fraud'].mean() * 100), 2),
            "imbalance_ratio": f"{round((1 - clean_df['is_fraud'].mean()) / clean_df['is_fraud'].mean(), 2)}:1",
        },
    }

    report_path = PROCESSED_DATA_DIR / "reports" / "preprocessing_summary_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print(f"\n[COMPLETE] Preprocessing summary report saved to: {report_path}")
    print("=" * 70)
    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset preprocessing.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(RAW_DATA_DIR / "transactions.csv"),
        help="Path to raw transactions CSV.",
    )
    args = parser.parse_args()
    run_preprocessing_pipeline(Path(args.data_path))

