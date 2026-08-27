# Fraud Detection Error Analysis: False Positives vs. False Negatives
**Razorpay AI Buildathon — AI Risk Manager Track**  
**Methodology:** Investigation of Trade-Off Boundaries, Cost Asymmetry, and Edge Cases.

---

## 1. The Inherent Trade-Off in Fraud Detection

In payment processing, fraud detection is an asymmetric cost minimization problem under extreme class imbalance (~1.6% fraud). Setting a decision boundary involves an unavoidable mathematical tension:

$$\text{F1-Score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

* **Lowering Decision Threshold ($\tau \downarrow$):** Catches more marginal fraud attempts ($\text{Recall} \uparrow$), but flags legitimate consumers with atypical habits ($\text{Precision} \downarrow, \text{False Positives} \uparrow$).
* **Raising Decision Threshold ($\tau \uparrow$):** Ensures approved transactions face zero friction ($\text{Precision} \uparrow$), but allows subtle fraud to slip through ($\text{Recall} \downarrow, \text{False Negatives} \uparrow$).

---

## 2. False Positive Analysis (Type I Error)

### Definition & Anatomy
An authentic transaction initiated by a legitimate cardholder that the system incorrectly classifies as high risk or suspicious.

### Root Causes Observed in Validation Experiments:
1. **Legitimate Luxury / Emergency Outlier Spend:** A cardholder whose average spend is $40 purchases $800 airline tickets or jewelry. The spending ratio (20x) mimics credential theft.
2. **Vacation Travel Displacement:** A customer traveling cross-country initiates a late-night purchase 1,500 km away from home, triggering geographic anomaly rules.
3. **Black Friday / Holiday Velocity Bursts:** Multiple rapid purchases during sales events that resemble automated card-testing scripts.

### Business Consequences:
* **Customer Friction & Abandonment:** Insulting a good customer by blocking their checkout causes up to 33% customer churn.
* **Lost Commercial Margin:** $M_{\text{margin}} \times \text{Amount}$ in lost gross revenue.
* **Operational Customer Support Cost:** $C_{\text{friction}} = \$10.00$ per support ticket/escalation.

### How Our System Mitigates False Positives:
Instead of binary automated blocking, the AI Risk Manager routes intermediate scores (36–69) to **Medium Risk (Step-Up 3DS OTP)**. This allows genuine consumers to self-authenticate in 5 seconds without merchant revenue loss.

---

## 3. False Negative Analysis (Type II Error)

### Definition & Anatomy
An unauthorized, fraudulent transaction that bypasses detection and is approved for settlement.

### Root Causes Observed in Validation Experiments:
1. **Low-Nominal Testing Transactions:** Fraudsters testing stolen card numbers on low-ticket ($15–$50) items like digital music or fast food during standard daytime hours.
2. **Category Camouflage:** Fraud executed in routine consumer categories (e.g. grocery stores) with order sizes that mimic normal family shopping ($180–$280).

### Business Consequences:
* **Direct Financial Loss:** 100% of order value lost upon chargeback.
* **Chargeback Network Fines:** Card networks impose fees of $20.00 per dispute.
* **Excess Dispute Penalty:** Exceeding a 0.9% dispute ratio risks gateway processing suspension.

### How Our System Mitigates False Negatives:
The **Behavioral Risk Signal Layer** specifically monitors customer baseline deviation (`amt_to_user_avg_ratio`) and transaction velocity. Even when nominal order values are modest, abnormal relative deviations trigger the hybrid threshold and catch the dispute.

---

*Verified on 2026-08-27 — AI Risk Manager Track.*

