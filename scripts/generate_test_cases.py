"""Preset Transaction Scenarios Generator for Demo & Interactive Simulator.

Creates diverse, realistic test cases representing various merchant fraud challenges:
- Low Risk (Normal everyday purchases)
- Medium Risk (Marginal deviations warranting OTP / Step-up auth)
- High Risk (Stolen credentials, rapid bot attacks, impossible travel, extreme amounts)
"""

import json
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_CASES_PATH = ROOT_DIR / "data" / "demo_scenarios.json"


DEMO_SCENARIOS = [
    {
        "id": "SCENARIO_1_LEGIT_GROCERY",
        "title": "🟢 Everyday In-Store Grocery (Normal)",
        "description": "Customer purchasing routine groceries near their home during daytime. Typical spending amount.",
        "transaction": {
            "trans_num": "TXN_LEGIT_001",
            "amt": 42.50,
            "category": "grocery_pos",
            "hour": 14,
            "day_of_week": 2,
            "is_night_transaction": 0,
            "haversine_distance_km": 3.8,
            "amt_to_user_avg_ratio": 0.85,
            "amt_to_cat_median_ratio": 0.95,
            "trans_velocity_1h": 0,
            "trans_velocity_24h": 1,
            "customer_age": 36,
            "category_fraud_rate": 0.002,
        },
        "expected_tier": "LOW RISK",
    },
    {
        "id": "SCENARIO_2_VELOCITY_BURST",
        "title": "🔴 Rapid Bot Attack / Card Testing (High Velocity)",
        "description": "Card used 4 times in the past 45 minutes on e-commerce sites. Classic automated card-testing burst.",
        "transaction": {
            "trans_num": "TXN_BURST_002",
            "amt": 280.00,
            "category": "shopping_net",
            "hour": 22,
            "day_of_week": 4,
            "is_night_transaction": 0,
            "haversine_distance_km": 15.0,
            "amt_to_user_avg_ratio": 4.20,
            "amt_to_cat_median_ratio": 3.10,
            "trans_velocity_1h": 4,
            "trans_velocity_24h": 7,
            "customer_age": 29,
            "category_fraud_rate": 0.035,
        },
        "expected_tier": "HIGH RISK",
    },
    {
        "id": "SCENARIO_3_IMPOSSIBLE_TRAVEL",
        "title": "🔴 Physical POS Impossible Travel (Stolen Card/Cloning)",
        "description": "Physical terminal in-person charge initiated 1,450 km away from customer's home coordinates.",
        "transaction": {
            "trans_num": "TXN_TRAVEL_003",
            "amt": 650.00,
            "category": "shopping_pos",
            "hour": 18,
            "day_of_week": 5,
            "is_night_transaction": 0,
            "haversine_distance_km": 1450.0,
            "amt_to_user_avg_ratio": 7.50,
            "amt_to_cat_median_ratio": 4.80,
            "trans_velocity_1h": 1,
            "trans_velocity_24h": 3,
            "customer_age": 45,
            "category_fraud_rate": 0.015,
        },
        "expected_tier": "HIGH RISK",
    },
    {
        "id": "SCENARIO_4_LATE_NIGHT_SPIKE",
        "title": "🔴 Late-Night High-Value Electronics Outlier",
        "description": "Transaction at 03:15 AM for $1,750 on digital luxury goods, 12x higher than typical user spend.",
        "transaction": {
            "trans_num": "TXN_NIGHT_004",
            "amt": 1750.00,
            "category": "misc_net",
            "hour": 3,
            "day_of_week": 1,
            "is_night_transaction": 1,
            "haversine_distance_km": 820.0,
            "amt_to_user_avg_ratio": 12.40,
            "amt_to_cat_median_ratio": 6.50,
            "trans_velocity_1h": 2,
            "trans_velocity_24h": 4,
            "customer_age": 52,
            "category_fraud_rate": 0.028,
        },
        "expected_tier": "HIGH RISK",
    },
    {
        "id": "SCENARIO_5_MEDIUM_TRAVEL_DINING",
        "title": "🟡 Business Travel Dining (Borderline / Step-up Auth)",
        "description": "Slightly elevated restaurant bill while traveling for work. Merits 3D Secure / OTP without outright blocking.",
        "transaction": {
            "trans_num": "TXN_DINING_005",
            "amt": 165.00,
            "category": "food_dining",
            "hour": 20,
            "day_of_week": 3,
            "is_night_transaction": 0,
            "haversine_distance_km": 280.0,
            "amt_to_user_avg_ratio": 2.10,
            "amt_to_cat_median_ratio": 2.40,
            "trans_velocity_1h": 0,
            "trans_velocity_24h": 2,
            "customer_age": 38,
            "category_fraud_rate": 0.004,
        },
        "expected_tier": "MEDIUM RISK",
    },
]


def save_demo_scenarios():
    TEST_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_CASES_PATH, "w", encoding="utf-8") as f:
        json.dump(DEMO_SCENARIOS, f, indent=2)
    print(f"Generated {len(DEMO_SCENARIOS)} demo scenarios to {TEST_CASES_PATH}")


if __name__ == "__main__":
    save_demo_scenarios()

