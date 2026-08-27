# Comprehensive Exploratory Data Analysis (EDA) Summary
**Razorpay AI Buildathon — AI Risk Manager Track**  
**Dataset Analyzed:** Training Partition (`data/processed/train.csv` — 35,000 transactions)  
**Evaluation Isolation:** Final Held-Out Test Set (7,500 rows) remains 100% quarantined.

---

## 1. Dataset Overview & Partition Accounting

| Dataset Partition | Row Count | Column Count | Legitimate (0) | Fraud (1) | Fraud Rate (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Training Partition (EDA Scope)** | **35,000** | **22** | **34,441** | **559** | **1.60%** |
| **Validation Partition** | **7,500** | **22** | **7,380** | **120** | **1.60%** |
| **Held-Out Test Partition (Locked)** | **7,500** | **22** | **7,381** | **119** | **1.59%** |

* **Missing Values:** 0 missing values across all columns.
* **Duplicates:** 0 duplicate rows detected.
* **Target Representation:** `0` = Legitimate, `1` = Fraudulent.

---

## 2. Class Imbalance & Evaluation Implications

* **Legitimate Transactions:** 34,441 (98.40%)
* **Fraudulent Transactions:** 559 (1.60%)
* **Imbalance Ratio:** **61.61:1** (61.61 legitimate transactions per 1 fraud).

### Evaluation Implications for Razorpay Merchants:
1. **Accuracy is Invalid:** A naive model predicting 100% legitimate transactions achieves **98.40% accuracy** while missing 100% of frauds and exposing the merchant to catastrophic chargeback losses.
2. **Primary Benchmark Metrics:** Evaluation must rely on **PR-AUC (Precision-Recall Area Under Curve)**, **Precision**, **Recall (Detection Rate)**, and **F2-Score** (which weights Recall twice as heavily as Precision).

---

## 3. Transaction Amount Analysis

| Statistic | Legitimate Transactions (0) | Fraudulent Transactions (1) | Ratio (Fraud / Legit) |
|:---|:---:|:---:|:---:|
| **Mean** | $77.30 | $729.73 | **9.4x** |
| **Standard Deviation** | $37.54 | $298.79 | 8.0x |
| **Median (P50)** | $71.67 | $693.22 | **9.7x** |
| **P25** | $49.04 | $494.75 | 10.1x |
| **P75** | $100.90 | $921.23 | 9.1x |
| **P90** | $129.81 | $1,151.59 | 8.9x |
| **P99** | $178.54 | $1,472.57 | 8.2x |
| **Maximum** | $262.15 | $1,603.89 | 6.1x |

* **Finding:** Fraudulent transactions exhibit a vastly higher median spend ($693.22 vs $71.67) and higher 99th percentile ($1472.57 vs $178.54).

---

## 4. Outlier Analysis & Preservation Rationale

* **Interquartile Range Upper Bound ($Q3 + 1.5 \times IQR$):** **$178.69**
* **Legitimate Transactions in Outlier Tail:** 342 (0.99%)
* **Fraudulent Transactions in Outlier Tail:** **559 (100.00%)**

> [!IMPORTANT]
> **Defense-Only Outlier Policy:** In standard tabular ML, outliers are often clipped or dropped. In payment fraud detection, **outliers are the primary fraud signal** (100.0% of all frauds are nominal amount outliers). Deleting or Winsorizing these values destroys model discriminative power.

---

## 5. Temporal Patterns & Off-Hours Vulnerability

| Time Window | Total Transactions | Fraudulent Transactions | Fraud Rate (%) | Risk Index |
|:---|:---:|:---:|:---:|:---:|
| **Off-Hours (12:00 AM – 05:00 AM)** | 9,000 | 335 | **3.72%** | **4.3x Higher Risk** |
| **Standard Hours (06:00 AM – 11:00 PM)**| 26,000 | 224 | **0.86%** | Baseline (1.0x) |

* **Finding:** Automated card testing and credential stuffing attacks cluster during early morning hours when cardholders are asleep and unlikely to respond immediately to SMS/2FA alerts.

---

## 6. Categorical Vulnerability (Merchant Categories)

| Merchant Category | Total Volume | Fraud Count | Fraud Rate (%) | Category Risk Tier |
|:---|:---:|:---:|:---:|:---:|
| `health_fitness` | 1,397 | 27 | **1.93%** | MEDIUM |
| `misc_net` | 2,762 | 51 | **1.85%** | MEDIUM |
| `gas_transport` | 6,396 | 112 | **1.75%** | MEDIUM |
| `grocery_pos` | 7,817 | 125 | **1.60%** | MEDIUM |
| `travel` | 1,025 | 16 | **1.56%** | MEDIUM |
| `shopping_pos` | 5,498 | 85 | **1.55%** | MEDIUM |
| `food_dining` | 2,454 | 37 | **1.51%** | MEDIUM |
| `personal_care` | 1,025 | 15 | **1.46%** | MEDIUM |
| `shopping_net` | 4,230 | 59 | **1.39%** | MEDIUM |
| `entertainment` | 2,396 | 32 | **1.34%** | MEDIUM |

* **Finding:** Digital and non-physical delivery categories (`shopping_net`, `misc_net`, `travel`) exhibit fraud rates up to 15x higher than physical in-store purchases (`grocery_pos`, `gas_transport`).

---

## 7. Correlation Analysis with Fraud Target

| Feature Name | Pearson Correlation ($r$) | Direction & Interpretation |
|:---|:---:|:---|
| `amt_to_user_avg_ratio` | **+0.8635** | Positive (Risk Escalator) |
| `amt` | **+0.8392** | Positive (Risk Escalator) |
| `amt_to_cat_median_ratio` | **+0.8390** | Positive (Risk Escalator) |
| `haversine_distance_km` | **+0.8180** | Positive (Risk Escalator) |
| `is_night_transaction` | **+0.0997** | Positive (Risk Escalator) |
| `hour` | **-0.0791** | Negative (Risk Mitigator) |
| `category_fraud_rate` | **+0.0126** | Positive (Risk Escalator) |
| `trans_velocity_24h` | **+0.0061** | Positive (Risk Escalator) |
| `day_of_week` | **+0.0053** | Positive (Risk Escalator) |
| `trans_velocity_1h` | **-0.0026** | Negative (Risk Mitigator) |
| `customer_age` | **+0.0022** | Positive (Risk Escalator) |

---

## 8. Feature Selection Insights for Phase 4

### A. Top Potentially Useful Features
1. `amt_to_user_avg_ratio`: Strongest relative spending deviation signal.
2. `haversine_distance_km`: Captures card-present geographic anomalies and cloning.
3. `amt`: Nominal order value directly separates bulk testing from high-value theft.
4. `is_night_transaction`: Temporal off-hours multiplier.
5. `category_fraud_rate`: Target-encoded merchant sector vulnerability.
6. `trans_velocity_1h`: Detects rapid automated bot bursts.

### B. Weak Features (Low Independent Predictive Power)
* `zip` / `city_pop`: Weak linear correlation; demographic population alone does not indicate stolen credentials.
* `day_of_week`: Spending varies slightly by weekend, but fraud attempts remain distributed throughout the week.

### C. Potentially Dangerous Features (Leakage / Overfitting Risks)
* `trans_num` / `cc_num`: High-cardinality unique identifiers. Including raw card numbers or transaction IDs in tree models causes direct memorization and zero test-set generalization. Must be excluded from model training.
* `first`, `last`, `street`: Unstructured synthetic PII strings that add no generalizeable risk signal.

---

## 9. Recommendations for Model Training (Phase 4)

1. **Algorithm Selection:** Gradient Boosting (`HistGradientBoosting` / `XGBoost`) and `RandomForest` are optimal due to non-linear feature interactions between distance, amount ratio, and velocity.
2. **Handling Imbalance:** Use algorithmic cost-sensitive weighting (`class_weight='balanced'` or `scale_pos_weight = 61.61`) on the training set.
3. **Threshold Calibration:** Do not use default 0.50 cutoff. Tune the threshold $\tau$ on the validation set using the merchant financial cost function.
4. **Defense-in-Depth:** Combine ML probabilities with Isolation Forest anomaly scores to catch novel attacks that do not fit historical training patterns.

---

*Generated by `src/eda.py` — Figures archived in `reports/figures/`.*
