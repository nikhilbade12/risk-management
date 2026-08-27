"""Data Loader module for AI Risk Manager.

Loads transaction data, validates schema, and performs leak-free stratified
splitting into Train (70%), Validation (15%), and Held-out Test (15%).
"""

import sys
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


REQUIRED_COLUMNS = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "is_fraud",
]


def load_raw_data() -> pd.DataFrame:
    """Loads raw transactions data, generating it if not yet present."""
    data_path = RAW_DATA_DIR / "transactions.csv"
    if not data_path.exists():
        from scripts.setup_data import setup_dataset
        setup_dataset()

    df = pd.read_csv(data_path)
    # Check for required schema columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Transaction data missing essential columns: {missing}")

    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified 3-way split:
    - Train (70%)
    - Validation (15%)
    - Held-out Test (15%)
    Ensuring identical class balance across all splits.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    # Split train vs temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["is_fraud"],
        random_state=random_seed,
    )

    # Split temp into validation and test (50/50 of temp if val=15%, test=15%)
    val_rel_ratio = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_rel_ratio),
        stratify=temp_df["is_fraud"],
        random_state=random_seed,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_and_save_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepares and saves Train, Val, and Test splits into data/processed/."""
    df = load_raw_data()
    train_df, val_df, test_df = split_data(df)

    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Data successfully split and saved to {PROCESSED_DATA_DIR}:")
    print(f"  * Train:      {len(train_df):,} rows ({train_df['is_fraud'].sum():,} frauds, {train_df['is_fraud'].mean()*100:.2f}%)")
    print(f"  * Validation: {len(val_df):,} rows ({val_df['is_fraud'].sum():,} frauds, {val_df['is_fraud'].mean()*100:.2f}%)")
    print(f"  * Test:       {len(test_df):,} rows ({test_df['is_fraud'].sum():,} frauds, {test_df['is_fraud'].mean()*100:.2f}%)")

    return train_df, val_df, test_df


def load_processed_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads pre-split processed datasets if they exist, or splits them."""
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    if train_path.exists() and val_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        return train_df, val_df, test_df

    return prepare_and_save_splits()


if __name__ == "__main__":
    prepare_and_save_splits()
