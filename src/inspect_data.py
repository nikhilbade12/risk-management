"""Dataset Inspection Script for AI Risk Manager.

Loads raw or processed transaction data and analyzes:
- Shape, columns, and data types
- Missing values and duplicate records
- Unique value cardinality
- Numerical and categorical feature classification
- Fraud target class distribution and class imbalance ratio
Saves structured inspection metrics to JSON.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def inspect_dataset(file_path: Path, target_col: str = "is_fraud") -> Dict[str, Any]:
    """Inspects a transaction dataset and returns a comprehensive metadata dictionary."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    print("=" * 70)
    print(f"[INSPECT] DATASET INSPECTION: {file_path.name}")
    print(f"Path: {file_path}")
    print("=" * 70)

    df = pd.read_csv(file_path)
    n_rows, n_cols = df.shape
    print(f"\n[1] Shape: {n_rows:,} rows | {n_cols} columns")

    # Column datatypes
    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if target_col in num_cols:
        num_features = [c for c in num_cols if c != target_col]
    else:
        num_features = num_cols

    print(f"\n[2] Feature Classification:")
    print(f"    * Numerical columns ({len(num_features)}): {num_features}")
    print(f"    * Categorical columns ({len(cat_cols)}): {cat_cols}")

    # Missing values
    missing_series = df.isnull().sum()
    missing_dict = {}
    total_missing = int(missing_series.sum())
    print(f"\n[3] Missing Values Summary (Total Missing Cells: {total_missing:,}):")
    for col, count in missing_series.items():
        pct = (count / n_rows) * 100
        missing_dict[col] = {"missing_count": int(count), "missing_percentage": round(pct, 3)}
        if count > 0:
            print(f"    ! {col}: {count:,} missing ({pct:.2f}%)")
    if total_missing == 0:
        print("    * No missing values found across all columns.")

    # Duplicate records
    dup_count = int(df.duplicated().sum())
    dup_pct = (dup_count / n_rows) * 100
    print(f"\n[4] Duplicate Rows:")
    print(f"    * Duplicates: {dup_count:,} ({dup_pct:.2f}%)")

    # Cardinality
    cardinality = {col: int(df[col].nunique()) for col in df.columns}

    # Target distribution
    target_summary = {}
    if target_col in df.columns:
        counts = df[target_col].value_counts().to_dict()
        total = len(df)
        legit_count = int(counts.get(0, 0))
        fraud_count = int(counts.get(1, 0))
        legit_pct = (legit_count / total) * 100
        fraud_pct = (fraud_count / total) * 100
        imbalance_ratio = round(legit_count / max(1, fraud_count), 2)

        target_summary = {
            "target_column": target_col,
            "legitimate_count": legit_count,
            "legitimate_percent": round(legit_pct, 2),
            "fraud_count": fraud_count,
            "fraud_percent": round(fraud_pct, 2),
            "imbalance_ratio": f"{imbalance_ratio}:1",
        }

        print(f"\n[5] Target Class Distribution ('{target_col}'):")
        print(f"    * Legitimate (0): {legit_count:,} ({legit_pct:.2f}%)")
        print(f"    * Fraud (1):      {fraud_count:,} ({fraud_pct:.2f}%)")
        print(f"    * Class Imbalance Ratio: {imbalance_ratio} legitimate transactions per 1 fraud")
    else:
        print(f"\n[5] Warning: Target column '{target_col}' not present in dataset.")

    # Numeric distribution highlights for 'amt'
    amt_stats = {}
    if "amt" in df.columns:
        amt_stats = {
            "min": round(float(df["amt"].min()), 2),
            "p25": round(float(df["amt"].quantile(0.25)), 2),
            "median": round(float(df["amt"].median()), 2),
            "mean": round(float(df["amt"].mean()), 2),
            "p75": round(float(df["amt"].quantile(0.75)), 2),
            "p99": round(float(df["amt"].quantile(0.99)), 2),
            "max": round(float(df["amt"].max()), 2),
        }
        print(f"\n[6] Transaction Amount ('amt') Distribution:")
        print(f"    * Min: ${amt_stats['min']:.2f} | Median: ${amt_stats['median']:.2f} | Mean: ${amt_stats['mean']:.2f} | P99: ${amt_stats['p99']:.2f} | Max: ${amt_stats['max']:.2f}")

    # Compile inspection report
    report = {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "row_count": n_rows,
        "column_count": n_cols,
        "columns": list(df.columns),
        "data_types": dtypes_dict,
        "numerical_features": num_features,
        "categorical_features": cat_cols,
        "missing_values": missing_dict,
        "total_missing_cells": total_missing,
        "duplicate_rows": dup_count,
        "cardinality": cardinality,
        "target_distribution": target_summary,
        "amount_statistics": amt_stats,
        "head_sample": df.head(3).to_dict(orient="records"),
    }

    # Save inspection report
    reports_dir = PROCESSED_DATA_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"{file_path.stem}_inspection_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[COMPLETE] Inspection report saved to: {report_file}")
    print("=" * 70)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect transaction dataset.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(RAW_DATA_DIR / "transactions.csv"),
        help="Path to dataset CSV to inspect.",
    )
    args = parser.parse_args()
    inspect_dataset(Path(args.data_path))
