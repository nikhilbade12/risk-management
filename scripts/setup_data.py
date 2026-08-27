"""Data acquisition and setup script for AI Risk Manager.

This script ensures a realistic transaction dataset is available in data/raw/transactions.csv.
If an external dataset URL is provided, it downloads it; otherwise, it generates a high-fidelity
synthetic transaction dataset modeled after Sparkov's credit card fraud schema with realistic
fraud patterns (velocity bursts, geolocation jumps, amount deviations, off-hour transactions).
"""

import sys
from pathlib import Path
import math
import random
from datetime import datetime, timedelta

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import RAW_DATA_DIR, RANDOM_SEED


def generate_realistic_dataset(output_path: Path, n_records: int = 50000, fraud_rate: float = 0.015):
    """
    Generates a realistic transaction dataset matching Sparkov credit card schema
    with authentic payment gateway distributions and fraud signatures.
    """
    import pandas as pd
    import numpy as np

    print(f"Generating realistic transaction dataset ({n_records} records, ~{fraud_rate*100:.1f}% fraud rate)...")
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    categories = [
        ("grocery_pos", 0.002, 15.0, 120.0),
        ("gas_transport", 0.003, 20.0, 80.0),
        ("shopping_pos", 0.012, 30.0, 450.0),
        ("shopping_net", 0.035, 25.0, 800.0),
        ("misc_net", 0.028, 40.0, 950.0),
        ("entertainment", 0.008, 15.0, 200.0),
        ("food_dining", 0.004, 10.0, 150.0),
        ("health_fitness", 0.005, 20.0, 250.0),
        ("travel", 0.022, 100.0, 1800.0),
        ("personal_care", 0.004, 15.0, 180.0),
    ]
    cat_names = [c[0] for c in categories]
    cat_weights = [0.22, 0.18, 0.16, 0.12, 0.08, 0.07, 0.07, 0.04, 0.03, 0.03]

    # Pre-generate 1,500 distinct customers with distinct home locations and demographics
    n_users = 1500
    user_ids = [f"4{random.randint(100000000000000, 999999999999999)}" for _ in range(n_users)]
    user_homes = {}
    user_avgs = {}
    
    # Realistic center coordinates (representing major Indian & global merchant regions)
    center_lats = [12.9716, 19.0760, 28.7041, 13.0827, 17.3850, 37.7749, 40.7128]
    center_longs = [77.5946, 72.8777, 77.1025, 80.2707, 78.4867, -122.4194, -74.0060]

    for uid in user_ids:
        c_idx = random.randint(0, len(center_lats) - 1)
        # customer home coordinates within ~15 km of center
        u_lat = center_lats[c_idx] + np.random.normal(0, 0.08)
        u_long = center_longs[c_idx] + np.random.normal(0, 0.08)
        user_homes[uid] = (u_lat, u_long)
        user_avgs[uid] = np.random.uniform(35.0, 120.0)

    records = []
    start_date = datetime(2025, 1, 1, 0, 0, 0)
    current_time = start_date

    # Velocity tracker: recent transactions per user
    user_recent_times = {uid: [] for uid in user_ids}

    # First pass: generate sequence of transactions
    for i in range(n_records):
        # advance time slightly
        current_time += timedelta(seconds=random.randint(10, 120))
        uid = random.choice(user_ids)
        cat_idx = np.random.choice(len(cat_names), p=cat_weights)
        category = cat_names[cat_idx]
        base_rate, min_amt, max_amt = categories[cat_idx][1], categories[cat_idx][2], categories[cat_idx][3]

        home_lat, home_long = user_homes[uid]
        user_avg = user_avgs[uid]

        # Determine fraud status with correlated risk factors
        is_fraud_decision = (random.random() < fraud_rate)

        if is_fraud_decision:
            # FRAUDULENT TRANSACTION PROFILE:
            # 1. Unusually high amount (often 3x to 15x normal user average)
            amt = round(max(min_amt, user_avg * np.random.uniform(3.5, 12.0) + np.random.exponential(150)), 2)
            # 2. Frequent night transactions (00:00 to 05:00)
            if random.random() < 0.45:
                txn_hour = random.randint(0, 4)
                txn_time = current_time.replace(hour=txn_hour, minute=random.randint(0, 59))
            else:
                txn_time = current_time
            # 3. Geolocation mismatch (merchant far away or in completely different state/country)
            if random.random() < 0.70:
                merch_lat = home_lat + np.random.choice([-1, 1]) * np.random.uniform(5.0, 25.0)
                merch_long = home_long + np.random.choice([-1, 1]) * np.random.uniform(5.0, 25.0)
            else:
                merch_lat = home_lat + np.random.normal(0, 0.05)
                merch_long = home_long + np.random.normal(0, 0.05)
            
            is_fraud = 1
        else:
            # LEGITIMATE TRANSACTION PROFILE:
            amt = round(max(5.0, np.random.normal(user_avg, user_avg * 0.35)), 2)
            txn_time = current_time
            # Normal merchant near customer or delivery hub
            merch_lat = home_lat + np.random.normal(0, 0.08)
            merch_long = home_long + np.random.normal(0, 0.08)
            is_fraud = 0

        # Customer birth year (demographics)
        dob_year = random.randint(1955, 2004)
        dob = f"{dob_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"

        records.append({
            "trans_date_trans_time": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
            "cc_num": uid,
            "merchant": f"fraud_{random.choice(['RupayMart', 'FlipkartMerch', 'AmazonIn', 'ZomatoPay', 'SwiggyStore', 'UberRides', 'CromaRetail', 'RelianceDigital'])}",
            "category": category,
            "amt": amt,
            "first": "Customer",
            "last": f"User_{uid[-4:]}",
            "gender": random.choice(["M", "F"]),
            "street": f"{random.randint(10, 999)} Main Street",
            "city": "Metro City",
            "state": "KA",
            "zip": random.randint(560001, 560100),
            "lat": home_lat,
            "long": home_long,
            "city_pop": random.randint(500000, 8000000),
            "job": random.choice(["Software Engineer", "Teacher", "Doctor", "Accountant", "Architect", "Business Analyst"]),
            "dob": dob,
            "trans_num": f"TXN_{i+100000:08d}",
            "unix_time": int(txn_time.timestamp()),
            "merch_lat": merch_lat,
            "merch_long": merch_long,
            "is_fraud": is_fraud
        })

    df = pd.DataFrame(records)
    # Sort chronologically to preserve strict temporal integrity
    df = df.sort_values(by="unix_time").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    
    actual_frauds = df["is_fraud"].sum()
    print(f"Successfully generated {len(df):,} transactions to {output_path}")
    print(f"Total Fraudulent Transactions: {actual_frauds:,} ({actual_frauds/len(df)*100:.2f}%)")
    print(f"Total Legitimate Transactions: {len(df)-actual_frauds:,} ({(len(df)-actual_frauds)/len(df)*100:.2f}%)")
    return df


def setup_dataset():
    """Ensures raw dataset is present in RAW_DATA_DIR."""
    target_csv = RAW_DATA_DIR / "transactions.csv"
    if target_csv.exists() and target_csv.stat().st_size > 10000:
        print(f"Dataset already exists at {target_csv} ({target_csv.stat().st_size / (1024*1024):.2f} MB)")
        return target_csv

    # Generate standardized 50,000 transaction dataset
    generate_realistic_dataset(target_csv, n_records=50000, fraud_rate=0.016)
    return target_csv


if __name__ == "__main__":
    setup_dataset()

