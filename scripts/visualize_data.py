"""Data Visualization Script for AI Risk Manager.

Generates the 4 required exploratory data analysis figures:
1. Fraud vs. Legitimate Class Distribution (showing severe class imbalance)
2. Transaction Amount Distribution (comparing legitimate vs fraud spending)
3. Fraud Transactions by Hour of Day (showing temporal off-hour spikes)
4. Fraud Distribution by Merchant Category (showing category vulnerability)
Saves static figures to reports/figures/.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

FIGURES_DIR = ROOT_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_visualizations(data_path: Path = RAW_DATA_DIR / "transactions.csv"):
    print("=" * 70)
    print("[VISUALIZATION] GENERATING EXPLORATORY DATA ANALYSIS CHARTS")
    print(f"Output Directory: {FIGURES_DIR}")
    print("=" * 70)

    df = pd.read_csv(data_path)
    if not pd.api.types.is_datetime64_any_dtype(df["trans_date_trans_time"]):
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    df["hour"] = df["trans_date_trans_time"].dt.hour

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {"legit": "#00C48C", "fraud": "#FF4D4F", "navy": "#0C2340", "blue": "#3395FF"}

    # -------------------------------------------------------------
    # 1. Fraud vs. Legitimate Class Distribution
    # -------------------------------------------------------------
    print("  [1/4] Generating Class Distribution Chart...")
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["is_fraud"].value_counts()
    labels = ["Legitimate (0)", "Fraud (1)"]
    bar_colors = [colors["legit"], colors["fraud"]]
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=bar_colors, width=0.45)

    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 500,
            f"{int(height):,} ({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_title("Transaction Class Distribution (Severe Class Imbalance)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Transactions", fontsize=11)
    ax.set_ylim(0, len(df) * 1.12)
    fig.tight_layout()
    p1 = FIGURES_DIR / "1_class_distribution.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    print(f"       Saved: {p1.name}")

    # -------------------------------------------------------------
    # 2. Transaction Amount Distribution (Log Scale)
    # -------------------------------------------------------------
    print("  [2/4] Generating Transaction Amount Distribution Chart...")
    fig, ax = plt.subplots(figsize=(8, 5))
    legit_amt = df[df["is_fraud"] == 0]["amt"]
    fraud_amt = df[df["is_fraud"] == 1]["amt"]

    bins = np.logspace(np.log10(max(1, df["amt"].min())), np.log10(df["amt"].max()), 40)
    ax.hist(legit_amt, bins=bins, alpha=0.6, label=f"Legitimate (Mean: ${legit_amt.mean():.1f})", color=colors["blue"], density=True)
    ax.hist(fraud_amt, bins=bins, alpha=0.7, label=f"Fraudulent (Mean: ${fraud_amt.mean():.1f})", color=colors["fraud"], density=True)

    ax.set_xscale("log")
    ax.set_title("Transaction Amount Distribution: Legitimate vs. Fraud (Log Scale)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Transaction Amount ($ - Logarithmic)", fontsize=11)
    ax.set_ylabel("Normalized Density", fontsize=11)
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    fig.tight_layout()
    p2 = FIGURES_DIR / "2_amount_distribution.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    print(f"       Saved: {p2.name}")

    # -------------------------------------------------------------
    # 3. Fraud Transactions by Hour of Day
    # -------------------------------------------------------------
    print("  [3/4] Generating Fraud Temporal Spike by Hour Chart...")
    fig, ax1 = plt.subplots(figsize=(9, 5))
    hour_grp = df.groupby("hour").agg(
        total_txns=("amt", "count"),
        fraud_txns=("is_fraud", "sum"),
        fraud_rate=("is_fraud", "mean"),
    ).reset_index()

    ax1.bar(hour_grp["hour"], hour_grp["fraud_txns"], color=colors["navy"], alpha=0.85, width=0.6, label="Fraud Count")
    ax1.set_xlabel("Hour of Day (0 to 23)", fontsize=11)
    ax1.set_ylabel("Fraud Transaction Count", color=colors["navy"], fontsize=11, fontweight="bold")
    ax1.set_xticks(range(0, 24))

    # Highlight off-hours risk zone (0 to 5 AM)
    ax1.axvspan(-0.5, 5.5, color="#FEE2E2", alpha=0.5, label="High-Risk Off-Hours (12 AM - 5 AM)")

    ax2 = ax1.twinx()
    ax2.plot(hour_grp["hour"], hour_grp["fraud_rate"] * 100, color=colors["fraud"], marker="o", linewidth=2.5, label="Fraud Rate %")
    ax2.set_ylabel("Fraud Rate (%)", color=colors["fraud"], fontsize=11, fontweight="bold")
    ax2.grid(False)

    ax1.set_title("Fraud Activity by Hour of Day (Off-Hours Spike Detection)", fontsize=13, fontweight="bold", pad=15)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    p3 = FIGURES_DIR / "3_fraud_by_hour.png"
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    print(f"       Saved: {p3.name}")

    # -------------------------------------------------------------
    # 4. Merchant Category Fraud Distribution
    # -------------------------------------------------------------
    print("  [4/4] Generating Merchant Category Risk Chart...")
    fig, ax = plt.subplots(figsize=(9, 5))
    cat_grp = df.groupby("category").agg(
        fraud_rate=("is_fraud", "mean"),
        total=("amt", "count"),
    ).sort_values(by="fraud_rate", ascending=True)

    y_pos = np.arange(len(cat_grp))
    rates = cat_grp["fraud_rate"] * 100
    bars = ax.barh(y_pos, rates, color=colors["blue"], height=0.6)

    # Highlight top categories
    for i, bar in enumerate(bars):
        if rates.iloc[i] > 2.0:
            bar.set_color(colors["fraud"])
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2.0,
            f"{rates.iloc[i]:.2f}%",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat_grp.index, fontsize=10)
    ax.set_xlabel("Historical Fraud Rate (%)", fontsize=11)
    ax.set_title("Merchant Category Vulnerability & Dispute Frequency", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(0, max(rates) * 1.20)
    fig.tight_layout()
    p4 = FIGURES_DIR / "4_category_distribution.png"
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    print(f"       Saved: {p4.name}")

    print("\n[COMPLETE] All 4 exploratory data visualizations generated successfully.")
    print("=" * 70)


if __name__ == "__main__":
    generate_visualizations()

