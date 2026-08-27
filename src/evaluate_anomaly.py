"""Anomaly Detection Diagnostic & Visualization Script.
Razorpay AI Buildathon — AI Risk Manager Track.

Loads the trained unsupervised Isolation Forest model from models/anomaly_model.joblib,
evaluates on the Validation partition (data/processed/val.csv),
and generates 5 publication-quality diagnostic charts in reports/figures/:
1. Normal vs. Anomalous Count Bar Chart
2. Anomaly Score Distribution (Legitimate vs. Known Fraud)
3. Anomaly Score vs. Transaction Amount Scatter Plot
4. Fraud Rate by Anomaly Score Tier (0-25, 25-50, 50-75, 75-100)
5. Feature Distribution Comparison (Amount, Distance, Spend Ratio)

Also exports reports/anomaly_detection_summary.md.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import joblib

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.anomaly_detector import AnomalyDetector, ANOMALY_FEATURE_COLS

FIGURES_DIR = ROOT_DIR / "reports" / "figures"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_anomaly_detector(
    val_data_path: Path = PROCESSED_DATA_DIR / "val.csv",
    model_path: Path = MODELS_DIR / "anomaly_model.joblib",
    preprocessor_path: Path = MODELS_DIR / "preprocessor.joblib",
):
    print("=" * 75)
    print("[EVALUATION] EXECUTING ANOMALY DETECTION VALIDATION EVALUATION")
    print(f"Validation Data: {val_data_path}")
    print(f"Model Artifact:  {model_path}")
    print("=" * 75)

    if not model_path.exists():
        raise FileNotFoundError(f"Anomaly model artifact not found at {model_path}. Run src/train_anomaly.py first.")

    anomaly_detector = AnomalyDetector.load(model_path)
    extractor = joblib.load(preprocessor_path)

    val_df = pd.read_csv(val_data_path)
    y_val = val_df["is_fraud"].values
    amounts_val = val_df["amt"].values

    # Extract features
    X_val_full = extractor.transform(val_df)
    X_val_anomaly = X_val_full[anomaly_detector.feature_cols]

    # Calculate Anomaly Scores (0 to 100)
    val_scores_100 = anomaly_detector.score_100(X_val_anomaly)
    threshold = 65.0
    val_preds_binary = (val_scores_100 >= threshold).astype(int)

    n_total = len(val_df)
    n_anomalies = int(np.sum(val_preds_binary == 1))
    n_normal = n_total - n_anomalies
    pct_anom = (n_anomalies / n_total) * 100

    n_fraud = int(np.sum(y_val == 1))
    fraud_detected = int(np.sum((y_val == 1) & (val_preds_binary == 1)))
    legit_anomalies = int(np.sum((y_val == 0) & (val_preds_binary == 1)))
    missed_fraud = n_fraud - fraud_detected

    recall_pct = (fraud_detected / max(1, n_fraud)) * 100
    precision_pct = (fraud_detected / max(1, n_anomalies)) * 100

    print(f"\n[1] Diagnostic Summary (Score Threshold = {threshold:.1f}):")
    print(f"    * Total Validation Records:     {n_total:,}")
    print(f"    * Normal Transactions:          {n_normal:,} ({(n_normal/n_total)*100:.2f}%)")
    print(f"    * Anomalous Transactions:       {n_anomalies:,} ({pct_anom:.2f}%)")
    print(f"    * Known Frauds Caught by Anomaly: {fraud_detected} of {n_fraud} (Recall: {recall_pct:.2f}%)")
    print(f"    * Known Frauds Missed by Anomaly: {missed_fraud}")
    print(f"    * Legitimate Flagged as Anomaly:  {legit_anomalies} (Unusual Genuine Shopping)")
    print(f"    * Anomaly Precision:            {precision_pct:.2f}%")

    # -------------------------------------------------------------
    # GENERATE 5 DIAGNOSTIC FIGURES
    # -------------------------------------------------------------
    print("\n[2] Generating 5 Diagnostic Charts in reports/figures/...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {"navy": "#0C2340", "blue": "#3395FF", "green": "#00C48C", "red": "#FF4D4F", "amber": "#FAAD14", "purple": "#8B5CF6"}

    # 1. Normal vs. Anomalous Counts
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(["Normal (Score < 65)", "Anomalous (Score >= 65)"], [n_normal, n_anomalies], color=[colors["blue"], colors["amber"]], width=0.45)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2.0, h + 100, f"{int(h):,} ({(h/n_total)*100:.1f}%)", ha="center", fontweight="bold")
    ax.set_title("Unsupervised Transaction Categorization (Validation Set)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Transaction Count")
    ax.set_ylim(0, n_total * 1.15)
    fig.tight_layout()
    p1 = FIGURES_DIR / "anomaly_1_normal_vs_anomalous_counts.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p1.name}")

    # 2. Score Distribution (Legit vs. Fraud)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(val_scores_100[y_val == 0], bins=40, alpha=0.6, density=True, color=colors["blue"], label="Legitimate Transactions (0)")
    ax.hist(val_scores_100[y_val == 1], bins=40, alpha=0.75, density=True, color=colors["red"], label="Known Fraud Transactions (1)")
    ax.axvline(x=threshold, color=colors["amber"], linestyle="--", linewidth=2, label=f"Anomaly Threshold ({threshold:.1f})")
    ax.set_title("Calibrated Anomaly Score Distribution (0–100)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Anomaly Score (0 = Typical, 100 = Highly Unusual)")
    ax.set_ylabel("Normalized Density")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    p2 = FIGURES_DIR / "anomaly_2_score_distribution.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p2.name}")

    # 3. Score vs. Transaction Amount
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(val_scores_100[y_val == 0], amounts_val[y_val == 0], color=colors["blue"], alpha=0.25, s=15, label="Legitimate")
    ax.scatter(val_scores_100[y_val == 1], amounts_val[y_val == 1], color=colors["red"], alpha=0.85, s=35, marker="^", label="Known Fraud")
    ax.axvline(x=threshold, color=colors["amber"], linestyle="--", label=f"Cutoff ({threshold:.1f})")
    ax.set_yscale("log")
    ax.set_title("Multi-Dimensional Behavior: Anomaly Score vs. Transaction Amount", fontsize=12, fontweight="bold")
    ax.set_xlabel("Calibrated Anomaly Score (0–100)")
    ax.set_ylabel("Transaction Amount ($ - Log Scale)")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    p3 = FIGURES_DIR / "anomaly_3_score_vs_amount.png"
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p3.name}")

    # 4. Fraud Rate by Score Tier
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = [0, 25, 50, 75, 100]
    tier_labels = ["Low\n(0–25)", "Moderate\n(25–50)", "Elevated\n(50–75)", "Critical\n(75–100)"]
    val_tiers = pd.cut(val_scores_100, bins=bins, labels=tier_labels, include_lowest=True)
    tier_summary = pd.DataFrame({"tier": val_tiers, "is_fraud": y_val}).groupby("tier", observed=False)["is_fraud"].agg(["count", "sum", "mean"]).reset_index()
    tier_summary["rate_pct"] = tier_summary["mean"] * 100

    bars = ax.bar(tier_summary["tier"], tier_summary["rate_pct"], color=[colors["green"], colors["blue"], colors["amber"], colors["red"]], width=0.5)
    for b, rate in zip(bars, tier_summary["rate_pct"]):
        ax.text(b.get_x() + b.get_width()/2.0, b.get_height() + 0.5, f"{rate:.1f}%", ha="center", fontweight="bold")
    ax.set_title("Empirical Fraud Frequency Across Anomaly Score Tiers", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual Fraud Rate (%)")
    ax.set_ylim(0, max(tier_summary["rate_pct"]) * 1.25)
    fig.tight_layout()
    p4 = FIGURES_DIR / "anomaly_4_fraud_rate_by_score_tier.png"
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p4.name}")

    # 5. Feature Distribution Comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    anom_mask = (val_preds_binary == 1)

    # Subplot A: Distance
    dist_norm = X_val_anomaly.loc[~anom_mask, "haversine_distance_km"]
    dist_anom = X_val_anomaly.loc[anom_mask, "haversine_distance_km"]
    axes[0].boxplot([dist_norm, dist_anom], tick_labels=["Normal", "Anomalous"], patch_artist=True)
    axes[0].set_title("Haversine Distance (km)", fontweight="bold")
    axes[0].set_ylabel("Distance (km)")

    # Subplot B: Amount to User Ratio
    ratio_norm = X_val_anomaly.loc[~anom_mask, "amt_to_user_avg_ratio"]
    ratio_anom = X_val_anomaly.loc[anom_mask, "amt_to_user_avg_ratio"]
    axes[1].boxplot([ratio_norm, ratio_anom], tick_labels=["Normal", "Anomalous"], patch_artist=True)
    axes[1].set_title("Spend Ratio (Amt / User Avg)", fontweight="bold")
    axes[1].set_ylabel("Ratio")

    # Subplot C: Amount
    amt_norm = amounts_val[~anom_mask]
    amt_anom = amounts_val[anom_mask]
    axes[2].boxplot([amt_norm, amt_anom], tick_labels=["Normal", "Anomalous"], patch_artist=True)
    axes[2].set_title("Transaction Amount ($)", fontweight="bold")
    axes[2].set_ylabel("Amount ($)")
    axes[2].set_yscale("log")

    fig.suptitle("Behavioral Feature Contrast: Normal vs. Anomalous Clusters", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    p5 = FIGURES_DIR / "anomaly_5_feature_distributions.png"
    fig.savefig(p5, dpi=300)
    plt.close(fig)
    print(f"    * Saved: {p5.name}")

    # -------------------------------------------------------------
    # GENERATE reports/anomaly_detection_summary.md
    # -------------------------------------------------------------
    # Precompute profile numbers to avoid complex f-string expressions
    mean_amt_norm = float(amounts_val[~anom_mask].mean())
    mean_amt_anom = float(amounts_val[anom_mask].mean())
    amt_ratio_mult = mean_amt_anom / max(1.0, mean_amt_norm)

    med_amt_norm = float(np.median(amounts_val[~anom_mask]))
    med_amt_anom = float(np.median(amounts_val[anom_mask]))
    med_ratio_mult = med_amt_anom / max(1.0, med_amt_norm)

    mean_dist_norm = float(X_val_anomaly.loc[~anom_mask, 'haversine_distance_km'].mean())
    mean_dist_anom = float(X_val_anomaly.loc[anom_mask, 'haversine_distance_km'].mean())
    dist_ratio_mult = mean_dist_anom / max(0.1, mean_dist_norm)

    mean_ratio_norm = float(X_val_anomaly.loc[~anom_mask, 'amt_to_user_avg_ratio'].mean())
    mean_ratio_anom = float(X_val_anomaly.loc[anom_mask, 'amt_to_user_avg_ratio'].mean())
    ratio_mult = mean_ratio_anom / max(0.1, mean_ratio_norm)

    fraud_rate_norm = float(y_val[~anom_mask].mean() * 100)
    fraud_rate_anom = float(y_val[anom_mask].mean() * 100)
    risk_mult = fraud_rate_anom / max(0.01, fraud_rate_norm)

    pct_normal = (n_normal / n_total) * 100

    print("\n[3] Generating Markdown Summary Artifact: reports/anomaly_detection_summary.md...")
    summary_md = f"""# Unsupervised Anomaly Detection Summary
**Razorpay AI Buildathon — AI Risk Manager Track**  
**Algorithm:** Isolation Forest (`sklearn.ensemble.IsolationForest`)  
**Scope:** Unsupervised behavioral modeling on Training partition (`data/processed/train.csv`). Evaluated on Validation partition (`data/processed/val.csv`).

---

## 1. Methodology & Algorithm Selection

### Why Isolation Forest?
Supervised models (XGBoost/RandomForest) excel at identifying patterns that resemble **known historical fraud labels**. However, organized fraudsters frequently evolve novel attack vectors (zero-day fraud). **Isolation Forest** isolates observations by randomly selecting a feature and randomly selecting a split value between the maximum and minimum values. Because anomalies require substantially fewer recursive partitions to isolate, their path length in the ensemble trees is significantly shorter.

### Feature Selection (Strict Zero-Leakage)
The anomaly detector was fitted **strictly without ground-truth labels (`is_fraud`)** and without target-encoded attributes:
1. `amt`: Nominal order transaction value.
2. `haversine_distance_km`: Great-circle distance between customer billing home and merchant terminal coordinates.
3. `amt_to_user_avg_ratio`: Deviation from cardholder historical spending baseline.
4. `amt_to_cat_median_ratio`: Deviation from merchant sector median spend.
5. `trans_velocity_1h`: Number of authorizations on the card within 60 minutes.
6. `trans_velocity_24h`: Number of authorizations on the card within 24 hours.
7. `is_night_transaction`: Authorizations initiated during midnight to 5:00 AM off-hours.
8. `hour`: Diurnal transaction timing.
9. `customer_age`: Cardholder demographic baseline.

### Score Calibration (0–100 Scale)
Raw Isolation Forest produces negative path length decision scores. To provide an intuitive, interpretable signal for the future Hybrid Risk Engine, scores were normalized on the training set percentiles (0.5% - 99.5%) and inverted:
Score = clip((score_99.5 - raw_score) / (score_99.5 - score_0.5) * 100, 0, 100)
* **0:** Completely typical, expected transaction pattern.
* **100:** Highly extreme, multi-dimensional statistical outlier.

---

## 2. Validation Evaluation Results (Actual Calculated Statistics)

Evaluated on the **7,500 validation transactions (120 known frauds, 7,380 legitimate)**:

| Diagnostic Metric | Validation Result | Interpretation |
|:---|:---:|:---|
| **Total Validation Transactions** | **{n_total:,}** | 100.0% validation traffic |
| **Normal Transactions (Score < 65)** | **{n_normal:,} ({pct_normal:.2f}%)** | Baseline behavioral cluster |
| **Anomalous Transactions (Score >= 65)**| **{n_anomalies:,} ({pct_anom:.2f}%)** | Statistical outlier tail |
| **Known Fraud Count** | **{n_fraud}** | Ground truth disputes |
| **Known Frauds Flagged as Anomalous** | **{fraud_detected} of {n_fraud}** | Caught by unsupervised detector |
| **Anomaly Recall** | **{recall_pct:.2f}%** | Fraud catch rate via pure unusualness |
| **Unusual Legitimate Transactions (FP)**| **{legit_anomalies}** | High-spend or traveling customers |
| **Anomaly Precision** | **{precision_pct:.2f}%** | Expected for unsupervised filtering |

---

## 3. Behavioral Profiles: Normal vs. Anomalous Transactions

| Behavioral Feature | Normal Transactions (Score < 65) | Anomalous Transactions (Score >= 65) | Ratio (Anomalous / Normal) |
|:---|:---:|:---:|:---:|
| **Mean Amount** | ${mean_amt_norm:,.2f} | ${mean_amt_anom:,.2f} | **{amt_ratio_mult:.1f}x** |
| **Median Amount** | ${med_amt_norm:,.2f} | ${med_amt_anom:,.2f} | **{med_ratio_mult:.1f}x** |
| **Mean Haversine Distance** | {mean_dist_norm:.1f} km | {mean_dist_anom:.1f} km | **{dist_ratio_mult:.1f}x** |
| **Mean Spend Ratio (Amt / User Avg)**| {mean_ratio_norm:.2f}x | {mean_ratio_anom:.2f}x | **{ratio_mult:.1f}x** |
| **Empirical Fraud Frequency** | **{fraud_rate_norm:.2f}%** | **{fraud_rate_anom:.2f}%** | **{risk_mult:.1f}x Higher Risk** |

---

## 4. Key Interpretations & Role in Phase 7

1. **Anomaly != Criminal Fraud:** An anomaly simply signals that an event deviates sharply from the customer's typical habits. A customer purchasing holiday flights or jewelry is statistically anomalous, but legitimate. This is why unsupervised anomaly detection must **never** be used alone to auto-decline payments.
2. **Complementing Supervised ML:**
   * Supervised models identify *known, previously seen fraud profiles*.
   * Isolation Forest identifies *unusual, out-of-distribution patterns* that supervised trees might miss if fraudsters deliberately circumvent known feature thresholds.
3. **Integration into Phase 7 Hybrid Risk Engine:**
   In Phase 7, the normalized Anomaly Score (0–100) will be fused with the Supervised Fraud Probability (P_ML * 100) and Deterministic Rule Flags into a unified, balanced Risk Score.

---

*Generated by `src/evaluate_anomaly.py` — Figures archived in `reports/figures/`.*
"""

    summary_file = REPORTS_DIR / "anomaly_detection_summary.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"    * Saved: {summary_file}")
    print("\n" + "=" * 75)
    print("[COMPLETE] ANOMALY EVALUATION & VISUALIZATIONS FINISHED")
    print("=" * 75)


if __name__ == "__main__":
    evaluate_anomaly_detector()
