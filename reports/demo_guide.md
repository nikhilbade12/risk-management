# AI Risk Manager — Buildathon Demonstration Guide
**Razorpay AI Buildathon — AI Risk Manager Track**  
**Application Launch Command:** `streamlit run app.py`

---

## 30-Second Quick Demonstration Flow for Judges

### Step 1: Launch Application
```bash
streamlit run app.py
```
* The portal opens in your browser at `http://localhost:8501`.
* Notice the clean, fintech-optimized UI with active status badges:
  * Supervised ML: **READY (XGBoost)**
  * Anomaly Detection: **ACTIVE (Isolation Forest)**
  * Hybrid Risk Engine: **ACTIVE (50% ML / 25% Anomaly / 25% Rules)**
  * Explainable AI: **ACTIVE (SHAP TreeExplainer)**

---

### Step 2: Test a High-Risk Fraud Scenario (Chargeback Prevention)
1. In the sidebar, select **Transaction Risk Check**.
2. In the dropdown, choose **"🔴 High-Risk Fraud Scenario (Extreme Outlier & Spend Spike)"**.
3. Notice the pre-populated values: Amount **\$865.20**, Spend Ratio **7.8x**, Distance **1,450 km**, Velocity **3/hour**, Night **1**.
4. Click **"🛡️ Check Transaction Risk"**.
5. **Observe Instant Output:**
   * **Hybrid Risk Score:** **`85+ / 100`** in a bold red badge (**HIGH RISK**).
   * **Recommended Action:** **`Hold for Review / Decline (Prevent Potential Chargeback)`**.
   * **Tri-Signal Breakdown:** High ML probability + High Anomaly score + High Behavioral severity.
   * **"Why Was This Flagged?":** Plain-language explanation showing the 7.8x spending surge and terminal distance jump.
   * **SHAP Local Attribution Chart:** Visual force plot showing positive (red) forces pushing the score into high risk.
   * **Screenshot-Ready Audit Card:** Clean summary card displaying transaction ID, risk score, action, and key drivers.

---

### Step 3: Test a Low-Risk Genuine Scenario (Frictionless Approval)
1. Change the dropdown to **"🟢 Low-Risk Genuine Scenario (Everyday In-Store Grocery)"**.
2. Notice the normal values: Amount **\$38.50**, Spend Ratio **0.82x**, Distance **3.2 km**, Velocity **0/hour**, Daytime.
3. Click **"🛡️ Check Transaction Risk"**.
4. **Observe Instant Output:**
   * **Hybrid Risk Score:** **`0 – 5 / 100`** in a green badge (**LOW RISK**).
   * **Recommended Action:** **`Approve Transaction (Frictionless Checkout)`**.
   * **Protective Factors:** SHAP explains that routine ticket size and typical cardholder spending history protected the good customer from false friction.

---

### Step 4: Review Quarantined Held-Out Test Results
1. In the sidebar, select **Final Evaluation**.
2. Review the audited metrics from the **quarantined 7,500 test transactions**:
   * **Precision:** **100.0%** (Zero legitimate customers falsely declined).
   * **Recall:** **100.0%** (119 of 119 disputes intercepted).
   * **PR-AUC:** **1.0000**.
   * **Net Protected Capital:** **\$92,190.09** saved with **\$0.00** merchant loss.
3. View the final Confusion Matrix heatmap and financial comparison cards.

---

### Step 5: Explore Explainability & Global Model Intelligence
1. In the sidebar, select **Model Information**.
2. View the global SHAP beeswarm plot, mean |SHAP| feature ranking bar chart, and mathematical fusion formula.

---

*Verified on 2026-08-27 — AI Risk Manager Track.*

