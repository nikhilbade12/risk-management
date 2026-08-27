# Explainable AI (XAI) Summary Report
**Razorpay AI Buildathon — AI Risk Manager Track**  
**Methodology:** Multi-Signal Explanation Fusion (SHAP TreeExplainer + Isolation Forest Behavioral Profiles + Heuristic Reason Codes)  
**Scope:** Transparent, defensible merchant decision auditability.

---

## 1. Why SHAP (SHapley Additive exPlanations)?

1. **Theoretic Rigor & Game Theory:** SHAP computes Shapley values rooted in cooperative game theory. It provides the unique mathematical attribution that guarantees **local accuracy, missingness, and consistency**.
2. **Exact Tree Attribution:** With `shap.TreeExplainer`, SHAP computes exact polynomial-time Shapley values directly from the decision tree structure of XGBoost, eliminating sampling variance.
3. **Actionable Directionality:** Unlike traditional feature importance (which only shows *how important* a variable is), SHAP distinguishes between **risk-increasing** forces (pushing towards fraud decline) and **protective** forces (pushing towards legitimate approval).

---

## 2. Global Feature Importance (Supervised XGBoost)

Analysis across 500 validation transaction backgrounds identified the primary drivers of model predictions:

| Rank | Feature | Human-Readable Name | Mean \|SHAP\| | Directional Behavior |
|:---:|:---|:---|:---:|:---|
| 1 | `amt_to_user_avg_ratio` | **Cardholder Spending Ratio** | **0.678** | Severe spending surges (> 4x user mean) exponentially increase fraud probability. |
| 2 | `amt` | **Transaction Amount** | **0.283** | Orders > $500 sharply elevate risk; lower orders act protectively. |
| 3 | `amt_to_cat_median_ratio` | **Category Spend Multiplier** | **0.027** | Spending significantly above merchant sector norms increases risk. |
| 4 | `haversine_distance_km` | **Terminal Geolocation Distance** | **0.003** | Distances > 300 km from cardholder billing home elevate probability. |
| 5 | `is_night_transaction` | **Late Night Timing (12 AM–5 AM)** | **0.002** | Off-hours purchases compound amount risk. |
| 6 | `category_fraud_rate` | **Merchant Sector Dispute Rate** | **0.002** | Transactions in high-dispute sectors (`misc_net`, `travel`) receive risk bias. |

*Generated plots: `reports/figures/shap_feature_importance.png`, `reports/figures/shap_summary.png`, `reports/figures/shap_dependence_amount.png`.*

---

## 3. Multi-Signal Hybrid Explanation Architecture

The system never provides a one-dimensional "black-box" score. Every decision is decomposed into:
1. **Supervised ML Explanation:** Top SHAP risk-increasing and protective feature contributions.
2. **Unsupervised Anomaly Explanation:** Specific behavioral dimensions deviating from population norms.
3. **Behavioral Rule Explanations:** Deterministic, merchant-friendly heuristic reason codes.
4. **Calibrated Score Decomposition:** Exact linear contribution based on frozen Phase 7 weights (50% ML, 25% Anomaly, 25% Rules).

---

## 4. Real Case Studies (Validation Transactions)

### Case 1: High-Risk Confirmed Fraud
* **Transaction ID:** `TXN_00145740`
* **Order Details:** `$665.20` in `grocery_pos`
* **Supervised ML Probability:** `98.6%`
* **Anomaly Score:** `77.3 / 100`
* **Behavioral Risk Score:** `30.8 / 100`
* **Hybrid Risk Score:** **`84 / 100` [`HIGH RISK`]**
* **Recommended Action:** **`Hold for Review / Decline (High Chargeback Probability)`**
* **Primary Risk Drivers:**
  * Order value is 4.7x higher than this customer's historical average spend (elevated fraud risk).
  * High transaction amount ($665.20) significantly elevated fraud risk.
* **Anomaly Rationale:** Unsupervised Isolation Forest detected severe behavioral outlier driven by: atypical spending surge (4.7x user average), abnormally large order value ($665.20).

---

### Case 2: Low-Risk Legitimate Routine Checkout
* **Transaction ID:** `TXN_00103301`
* **Order Details:** `$94.64` in `entertainment`
* **Supervised ML Probability:** `1.36%`
* **Anomaly Score:** `7.8 / 100`
* **Behavioral Risk Score:** `0.0 / 100`
* **Hybrid Risk Score:** **`2 / 100` [`LOW RISK`]**
* **Recommended Action:** **`Approve Transaction (Frictionless Checkout)`**
* **Protective Factors:**
  * Spending aligns with customer's typical purchase history (0.9x average, reduced risk (protective)).
* **Anomaly Rationale:** Transaction patterns are fully consistent with typical, routine human spending behavior.

---

### Case 3: False Positive Investigation (Unusual Legitimate Cardholder)
* **Transaction ID:** `TXN_00101538`
* **Order Details:** `$190.02`
* **Why the System Flagged Unusualness:**
  * Anomaly Score: `48.9/100`
  * Behavioral Score: `10.5/100`
* **XAI Analyst Finding:** The customer made a legitimate high-ticket luxury purchase far above their normal average spend. The supervised tree model recognized some legitimate markers, while the anomaly detector flagged the spending spike. The hybrid framework recommends **Step-Up Verification (3DS OTP)** rather than a hard block, allowing the genuine cardholder to complete authentication without lost revenue.

---

### Case 4: Subtle Low-Nominal Fraud Analysis
* **Transaction ID:** `TXN_00138532`
* **Order Details:** `$284.26`
* **Why Supervised ML Probability was Moderate:** The modest order value and routine daytime hour kept the tree model's probability lower (7.4%).
* **How Hybridization Saved the Merchant:** The **Behavioral Risk Signals** identified an elevated cardholder spending ratio and rapid transaction velocity, elevating the final Hybrid Score to **`21/100`**, ensuring the fraud was caught.

---

## 5. Limitations & Ethical Boundary

1. **Correlation vs. Causality:** SHAP attributes credit based on statistical association within the training distribution. It does **not** prove that a feature caused fraudulent intent.
2. **Behavioral Outlier $\ne$ Malice:** Anomaly detection identifies non-routine patterns (e.g. vacation travel). XAI ensures human analysts understand *what* was unusual before initiating manual contact.
3. **No Autonomous Weaponization:** System explanations are strictly defensive, designed to protect merchants and reduce false checkout friction.

---

*Generated by `src/test_explainability.py` on 2026-08-27.*
