# Model Artifact Validation Audit Report
**Razorpay AI Buildathon — AI Risk Manager Track**  
**Audit Timestamp:** 2026-08-27  
**Integrity Status:** ALL REQUIRED MODEL ARTIFACTS VERIFIED & OPERATIONAL

---

## 1. Verified Model Artifact Inventory

| Artifact Filename | File Size | Format | Verification Status | Architectural Purpose |
|:---|:---:|:---:|:---:|:---|
| `models/advanced_model.joblib` | 130.1 KB | Joblib | **LOADED & VERIFIED** | Primary supervised classifier (`xgboost.XGBClassifier`, 120 trees). |
| `models/advanced_model.pkl` | 130.1 KB | Pickle Alias | **LOADED & VERIFIED** | Compatibility alias for XGBoost model payload. |
| `models/anomaly_model.joblib` | 1.33 MB | Joblib | **LOADED & VERIFIED** | Unsupervised Isolation Forest (`sklearn.ensemble.IsolationForest`). |
| `models/anomaly_model.pkl` | 1.33 MB | Pickle Alias | **LOADED & VERIFIED** | Compatibility alias for AnomalyDetector instance. |
| `models/preprocessor.joblib` | 27.9 KB | Joblib | **LOADED & VERIFIED** | Feature extraction pipeline with training-set population baselines. |
| `models/preprocessing_pipeline.pkl` | 27.9 KB | Pickle Alias | **LOADED & VERIFIED** | Compatibility alias for `TransactionFeatureExtractor`. |
| `models/baseline_model.joblib` | 2.3 KB | Joblib | **LOADED & VERIFIED** | Benchmark model (`LogisticRegression`) from Phase 4. |
| `models/best_hyperparameters.json`| 102 Bytes | JSON | **VERIFIED** | Frozen tuned hyperparameters for XGBoost. |
| `models/hybrid_config.json` | 528 Bytes | JSON | **VERIFIED** | Frozen 50/25/25 multi-signal weights and tier thresholds. |
| `models/test_metrics.json` | 997 Bytes | JSON | **VERIFIED** | Audited held-out test evaluation results (100% precision, 100% recall). |

---

## 2. Load Verification Test Results
* **`preprocessor.joblib`:** Loads without warnings; handles 11 input numerical and spatial attributes.
* **`advanced_model.joblib`:** Model payload extracts dictionary containing `model` and `feature_names`. Predicts probabilities in $[0.0, 1.0]$.
* **`anomaly_model.joblib`:** `score_100()` method evaluates multi-dimensional unusualness in $[0.0, 100.0]$.
* **`hybrid_config.json`:** Successfully parsed by `HybridRiskEngine.load_default()`.

---

*Verified by `tests/test_app_components.py` — All tests passing.*

