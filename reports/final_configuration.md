# Frozen Final Model Configuration
**Razorpay AI Buildathon — AI Risk Manager Track**  
**Status:** FROZEN PRIOR TO FINAL TEST EVALUATION  
**Audit Rule:** This configuration was determined strictly using the Training partition (35,000 records) and Validation partition (7,500 records). No parameters were chosen or adjusted using the held-out test partition.

---

## 1. Pipeline Artifacts & Models
* **Supervised Primary Classifier:** Tuned XGBoost (`xgboost.XGBClassifier`)
  * Artifact: `models/advanced_model.joblib`
  * Hyperparameters:
    * `n_estimators`: 120
    * `max_depth`: 7
    * `learning_rate`: 0.03
    * `scale_pos_weight`: 61.61
    * `tree_method`: `'hist'`
    * `random_state`: 42
* **Unsupervised Anomaly Detector:** Isolation Forest (`sklearn.ensemble.IsolationForest`)
  * Artifact: `models/anomaly_model.joblib`
  * Hyperparameters:
    * `n_estimators`: 120
    * `contamination`: 0.03
    * `random_state`: 42
  * Normalization Range: Training 0.5%–99.5% percentile min-max inverted to 0–100 scale.
* **Feature Preprocessor:** `TransactionFeatureExtractor`
  * Artifact: `models/preprocessor.joblib`
  * Reference Baselines: Fitted exclusively on `train.csv` (Train Global Mean Amount: $87.72).

---

## 2. Input Features
* **Supervised Feature Matrix (11 attributes):**
  `['amt', 'haversine_distance_km', 'amt_to_user_avg_ratio', 'amt_to_cat_median_ratio', 'trans_velocity_1h', 'trans_velocity_24h', 'is_night_transaction', 'hour', 'day_of_week', 'customer_age', 'category_fraud_rate']`
* **Unsupervised Feature Matrix (9 behavioral attributes, zero target leakage):**
  `['amt', 'haversine_distance_km', 'amt_to_user_avg_ratio', 'amt_to_cat_median_ratio', 'trans_velocity_1h', 'trans_velocity_24h', 'is_night_transaction', 'hour', 'customer_age']`
* **Excluded Identifiers:** `trans_num`, `cc_num`, `first`, `last`, `street`, `zip` (zero ID memorization).

---

## 3. Hybrid Risk Engine Configuration
* **Fusion Formula:**
  $$\text{Hybrid Risk Score} = \text{clip}\left( \text{round}\left( 0.50 \cdot S_{\text{ML}} + 0.25 \cdot S_{\text{anomaly}} + 0.25 \cdot S_{\text{behavioral}} \right), 0, 100 \right)$$
* **Signal Weights:**
  * Supervised ML Weight ($w_{\text{ML}}$): **50%** (0.50)
  * Unsupervised Anomaly Weight ($w_{\text{anomaly}}$): **25%** (0.25)
  * Behavioral Risk Rules Weight ($w_{\text{behavioral}}$): **25%** (0.25)
* **Risk Tier Thresholds:**
  * **LOW RISK:** `0 – 35` (Recommended Action: **APPROVE**)
  * **MEDIUM RISK:** `36 – 69` (Recommended Action: **STEP-UP VERIFICATION / OTP**)
  * **HIGH RISK:** `70 – 100` (Recommended Action: **HOLD / DECLINE**)
* **Frozen Operational Decision Cutoff:** **$\tau^* = 25$** (Determined by validation cost optimization).

---

## 4. Cost Model Assumptions
* **Chargeback Penalty Fee ($C_{\text{chargeback}}$):** **$20.00** per missed fraud incident.
* **Merchant Profit Margin ($M_{\text{margin}}$):** **20%** (0.20) of order value.
* **Customer Friction Support Cost ($C_{\text{friction}}$):** **$10.00** per false decline.
* **False Negative Cost Equation:** $\text{Transaction Amount} + \$20.00$
* **False Positive Cost Equation:** $(\text{Transaction Amount} \times 0.20) + \$10.00$

---

*Frozen Date: 2026-08-27 — AI Risk Manager Pipeline v1.0.0*

