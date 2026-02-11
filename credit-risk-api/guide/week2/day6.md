# 📊 Day 6: Baseline Model & Production ML Pipeline

**Goal:** Build your first production-grade ML model with proper evaluation metrics

**Time:** 2-3 hours

---

## 🎯 What You'll Build Today

Instead of using notebooks, we'll build **production Python scripts**:
- `src/ml/train_model.py` - Training pipeline
- `src/ml/evaluate.py` - Model evaluation
- Proper train/test splits with stratification
- Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)

---

## 📋 Step 1: Create Baseline Model Training Script

Create `src/ml/train_baseline.py`:

```python
"""
Baseline model for credit risk prediction.
Uses Logistic Regression as a simple starting point.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import joblib
from pathlib import Path

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_PATH = Path('models/trained/baseline_logistic.pkl')
SCALER_PATH = Path('models/artifacts/scaler_baseline.pkl')

def load_processed_data():
    """Load the processed master dataset from Week 1."""
    df = pd.read_csv('data/processed/master_dataset.csv')
    print(f"✅ Loaded data: {df.shape}")
    return df

def prepare_train_test(df):
    """Split data into train and test sets with stratification."""
    # Drop non-feature columns
    drop_cols = ['customerid', 'systemloanid', 'good_bad_flag', 'target']
    X = df.drop(columns=drop_cols)
    y = df['target']

    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Important for imbalanced data!
    )

    print(f"📊 Train set: {X_train.shape[0]} samples ({y_train.sum()/len(y_train)*100:.1f}% default)")
    print(f"📊 Test set:  {X_test.shape[0]} samples ({y_test.sum()/len(y_test)*100:.1f}% default)")

    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print("✅ Features scaled")
    return X_train_scaled, X_test_scaled, scaler

def train_baseline_model(X_train, y_train):
    """Train simple Logistic Regression baseline."""
    print("\n🤖 Training Logistic Regression baseline...")

    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight='balanced'  # Handle class imbalance
    )

    model.fit(X_train, y_train)
    print("✅ Model trained")

    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Comprehensive model evaluation."""
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION")
    print("="*60)

    # Train predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    # Test predictions
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    print("\n🎯 TRAIN METRICS:")
    print(f"   Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"   Precision: {precision_score(y_train, y_train_pred):.4f}")
    print(f"   Recall:    {recall_score(y_train, y_train_pred):.4f}")
    print(f"   F1 Score:  {f1_score(y_train, y_train_pred):.4f}")
    print(f"   ROC-AUC:   {roc_auc_score(y_train, y_train_proba):.4f}")

    print("\n🎯 TEST METRICS:")
    print(f"   Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"   Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_test_pred):.4f}")
    print(f"   F1 Score:  {f1_score(y_test, y_test_pred):.4f}")
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_test_proba):.4f}")

    print("\n📋 CLASSIFICATION REPORT (Test Set):")
    print(classification_report(y_test, y_test_pred,
                               target_names=['Good', 'Bad']))

    print("="*60 + "\n")

    return {
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_roc_auc': roc_auc_score(y_test, y_test_proba)
    }

def save_model_and_scaler(model, scaler):
    """Save trained model and scaler."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"💾 Model saved to: {MODEL_PATH}")
    print(f"💾 Scaler saved to: {SCALER_PATH}")

def main():
    """Main training pipeline."""
    print("="*60)
    print("🚀 BASELINE MODEL TRAINING PIPELINE")
    print("="*60)

    # Load data
    df = load_processed_data()

    # Split data
    X_train, X_test, y_train, y_test = prepare_train_test(df)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Train model
    model = train_baseline_model(X_train_scaled, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled,
                            y_train, y_test)

    # Save
    save_model_and_scaler(model, scaler)

    print("\n✨ Baseline model training complete!")
    print(f"   Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    print("\n📝 This is your baseline. All future models must beat this!")

if __name__ == "__main__":
    main()
```

---

## 📋 Step 2: Run Baseline Training

```bash
# Navigate to project root
cd credit-risk-api

# Run baseline training
python src/ml/train_baseline.py
```

**Expected Output:**
```
============================================================
🚀 BASELINE MODEL TRAINING PIPELINE
============================================================
✅ Loaded data: (5000, 42)
📊 Train set: 4000 samples (30.0% default)
📊 Test set:  1000 samples (30.0% default)
✅ Features scaled

🤖 Training Logistic Regression baseline...
✅ Model trained

============================================================
📊 MODEL EVALUATION
============================================================

🎯 TRAIN METRICS:
   Accuracy:  0.7825
   Precision: 0.6543
   Recall:    0.5892
   F1 Score:  0.6200
   ROC-AUC:   0.8421

🎯 TEST METRICS:
   Accuracy:  0.7790
   Precision: 0.6512
   Recall:    0.5800
   F1 Score:  0.6135
   ROC-AUC:   0.8405

📋 CLASSIFICATION REPORT (Test Set):
              precision    recall  f1-score   support

        Good       0.83      0.88      0.85       700
         Bad       0.65      0.58      0.61       300

    accuracy                           0.78      1000
   macro avg       0.74      0.73      0.73      1000
weighted avg       0.77      0.78      0.77      1000

============================================================

💾 Model saved to: models/trained/baseline_logistic.pkl
💾 Scaler saved to: models/artifacts/scaler_baseline.pkl

✨ Baseline model training complete!
   Test ROC-AUC: 0.8405

📝 This is your baseline. All future models must beat this!
```

---

## 📊 Step 3: Understanding Your Metrics

### What do these metrics mean?

1. **Accuracy (77.9%)**:
   - Correct predictions / Total predictions
   - ⚠️ Can be misleading with imbalanced data!

2. **Precision (65.1%)**:
   - When you predict "Bad", you're right 65% of the time
   - **Business meaning:** Of loans you reject, 65% would actually default

3. **Recall (58.0%)**:
   - Of all actual "Bad" loans, you catch 58%
   - **Business meaning:** You're catching 58% of defaults, missing 42%

4. **F1 Score (61.4%)**:
   - Harmonic mean of Precision and Recall
   - Balances both metrics

5. **ROC-AUC (84.0%)**:
   - **Most important metric for this problem!**
   - Measures model's ability to distinguish Good vs Bad
   - 84% is decent for a baseline!

---

## 🎯 Step 4: Business Interpretation

### Confusion Matrix Analysis

From the classification report:

| Actual | Predicted Good | Predicted Bad |
|--------|----------------|---------------|
| **Good (700)** | 616 | 84 |
| **Bad (300)** | 126 | 174 |

**Business Impact:**
- ✅ **True Positives (174):** Correctly rejected bad loans → **Saved money!**
- ✅ **True Negatives (616):** Correctly approved good loans → **Made revenue!**
- ❌ **False Positives (84):** Rejected good customers → **Lost business**
- ❌ **False Negatives (126):** Approved bad loans → **Lost money!** (42% miss rate)

**🚨 Problem:** Missing 42% of bad loans is too high!

**Goal for next models:** Reduce False Negatives (improve Recall)

---

## 📝 Step 5: Save Baseline Results

Create `models/evaluation/baseline_results.txt`:

```bash
mkdir -p models/evaluation
cat > models/evaluation/baseline_results.txt << 'EOF'
BASELINE MODEL: Logistic Regression
====================================

Model: LogisticRegression(class_weight='balanced', max_iter=1000)
Features: 38
Training Date: 2026-02-11

PERFORMANCE METRICS:
-------------------
Accuracy:  77.90%
Precision: 65.12%
Recall:    58.00%
F1 Score:  61.35%
ROC-AUC:   84.05%

BUSINESS METRICS:
----------------
False Negative Rate: 42.00% ⚠️ (Too high - missing defaults!)
False Positive Rate: 12.00%

NEXT STEPS:
-----------
- Try Random Forest (Day 11)
- Try XGBoost (Day 12)
- Goal: Improve Recall to >70%
- Goal: Improve ROC-AUC to >90%
EOF
```

---

## ✅ Day 6 Checklist

- [ ] Created `src/ml/train_baseline.py`
- [ ] Ran baseline training successfully
- [ ] Achieved ROC-AUC > 0.80
- [ ] Saved model and scaler to `models/` directory
- [ ] Documented baseline results
- [ ] Understood all evaluation metrics
- [ ] Identified area for improvement (Recall)

---

## 🚀 What's Next?

**Tomorrow (Day 7):** We'll improve the Logistic Regression with:
- Feature selection
- Hyperparameter tuning
- Threshold optimization
- Better handling of class imbalance

**Key Insight:** This baseline (84% ROC-AUC) is your benchmark. Every model you build must beat this, or it's not worth deploying!

---

## 💡 Key Takeaways

1. **Production code > Notebooks**
   - Reusable scripts
   - Version control friendly
   - Easier to deploy

2. **Always use stratified splits**
   - Maintains class balance in train/test
   - Prevents evaluation bias

3. **ROC-AUC is king for this problem**
   - Better than accuracy for imbalanced data
   - Measures discrimination ability

4. **Baseline first, then optimize**
   - Simple model establishes benchmark
   - Iterate and improve from there

---

**🎉 Congratulations!** You've built your first production ML model!

Tomorrow, we'll make it better 🚀
