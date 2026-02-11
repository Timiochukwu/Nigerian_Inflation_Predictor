# 🟢 Week 2: Production ML Pipeline

**Build production-grade machine learning code (not notebooks!)**

---

## Overview

This week, you'll build a **production ML training pipeline** using Python scripts, not Jupyter notebooks. This is how real ML engineers work in companies.

### What You'll Build

- ✅ `src/ml/train_baseline.py` - Baseline Logistic Regression
- ✅ `src/ml/train_model.py` - Production XGBoost model
- ✅ `src/ml/evaluate.py` - Model evaluation utilities
- ✅ Model artifacts (saved models, scalers, metadata)
- ✅ Feature importance analysis

### Daily Breakdown

| Day | Focus | Time | Deliverable |
|-----|-------|------|-------------|
| **Day 6** | Baseline Model | 1h | Logistic Regression (84% ROC-AUC) |
| **Day 7** | XGBoost Training | 1h | XGBoost model (97%+ accuracy) |
| **Day 8** | Model Artifacts | 30m | Saved models & metadata |
| **Day 9** | Feature Importance | 45m | Feature analysis report |
| **Day 10** | Model Selection | 45m | Final model choice |

**Total Time:** ~4-5 hours

---

## Learning Outcomes

By the end of this week, you'll be able to:

1. **Build Production ML Pipelines**
   - Not notebooks → Python scripts
   - Reusable code
   - Version controllable

2. **Train Advanced Models**
   - Logistic Regression baseline
   - XGBoost for tabular data
   - Handle class imbalance

3. **Evaluate Models Properly**
   - Accuracy, Precision, Recall, F1
   - ROC-AUC (most important!)
   - Confusion matrix analysis

4. **Manage ML Artifacts**
   - Save/load models
   - Version tracking
   - Metadata storage

---

## Prerequisites

- ✅ Week 1 complete (feature engineering done)
- ✅ Processed data in `data/processed/master_dataset.csv`
- ✅ Python environment with scikit-learn, xgboost installed

---

## Quick Start (Using Pre-Built Code)

**Option A:** Run the production code I already built:

```bash
cd credit-risk-api

# Generate data (if needed)
python src/data/generate_data.py

# Train model
python src/ml/train_model.py
```

**Result:** Trained XGBoost model with 97.9% accuracy in ~2 minutes

**Option B:** Build it yourself following the daily guides

---

## Day-by-Day Guide

### [Day 6: Baseline Model](day6.md)
Build a simple Logistic Regression baseline to establish performance benchmarks.

**Key Concepts:**
- Stratified train/test splits
- StandardScaler for feature scaling
- class_weight='balanced' for imbalanced data
- Comprehensive metrics

**Expected Output:**
- ROC-AUC: ~84%
- Saved model: `models/trained/baseline_logistic.pkl`

---

### Day 7: XGBoost Production Model
Build an XGBoost model with proper handling of imbalanced classes.

**What You'll Learn:**
- XGBoost hyperparameters
- scale_pos_weight for imbalance
- Early stopping
- Evaluation on held-out test set

**Code Location:** `src/ml/train_model.py` (already built)

**Expected Output:**
- Test Accuracy: 97.9%
- ROC-AUC: 99.87%
- False Negative Rate: 4.33%

**To Run:**
```bash
python src/ml/train_model.py
```

---

### Day 8: Model Artifacts & Versioning
Learn how to properly save and version ML models.

**What Gets Saved:**
- Trained model (.pkl)
- Scaler (.pkl)
- Feature names (.txt)
- Performance metrics (.txt)
- Feature importance (.csv)

**Location:** `models/artifacts/`

**Versioning Pattern:**
```
credit_risk_xgboost_v1.pkl
scaler_v1.pkl
metrics_v1.txt
```

---

### Day 9: Feature Importance Analysis
Understand which features drive predictions.

**Top 5 Features (from trained model):**
1. `days_since_last_loan` - 46.8% importance
2. `hist_num_open` - 14.3%
3. `hist_closure_rate` - 10.7%
4. `hist_num_closed` - 6.1%
5. `hist_ontime_rate` - 5.3%

**Business Insight:** Recent borrowing behavior is the strongest default predictor!

---

### Day 10: Model Selection & Cross-Validation
Compare models and select the best one for production.

**Models Compared:**
- Logistic Regression: 84.0% ROC-AUC
- XGBoost: 99.87% ROC-AUC ✅ **WINNER**

**Decision:** XGBoost selected for deployment

---

## Code Structure After Week 2

```
src/ml/
├── __init__.py
├── train_baseline.py        # Logistic Regression
├── train_model.py            # XGBoost (production)
├── feature_engineering.py    # From Week 1
└── evaluate.py               # Evaluation utilities

models/
├── trained/
│   ├── baseline_logistic.pkl
│   └── (XGBoost included in artifacts/)
└── artifacts/
    ├── credit_risk_xgboost_v1.pkl
    ├── scaler_v1.pkl
    ├── feature_names_v1.txt
    ├── metrics_v1.txt
    └── feature_importance_v1.csv
```

---

## Success Criteria

By end of Week 2, you should have:

- [ ] Baseline model trained (>80% ROC-AUC)
- [ ] XGBoost model trained (>95% accuracy)
- [ ] Models saved to `models/` directory
- [ ] Feature importance analyzed
- [ ] Best model selected for production

---

## Troubleshooting

**"Data file not found"**
- Run Week 1 data processing first
- Or run `python src/data/generate_data.py`

**"Poor model performance"**
- Check data quality
- Verify feature engineering
- Try different hyperparameters

**"Import errors"**
- Install: `pip install -r requirements.txt`

---

## Next Week Preview

**Week 3: FastAPI Development**
- Build REST API around your trained model
- Create prediction endpoints
- Add request validation
- Deploy to production

---

**Ready?** Start with [Day 6: Baseline Model →](day6.md)
