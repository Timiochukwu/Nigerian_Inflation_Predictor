# Week 2, Day 6 -- Baseline Model & Evaluation Metrics

## What You Will Learn Today

Welcome to **Week 2**! Today you'll build your **first machine learning model** and learn how to properly evaluate it.

By the end of this session, you will:

1. **Load preprocessed data** from Day 5 (train/test sets)
2. **Build a baseline model** (simple majority class predictor)
3. **Understand evaluation metrics** for classification:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix
   - ROC Curve & AUC
4. **Create visualizations** of model performance
5. **Interpret results** in business context
6. **Save baseline results** for comparison

---

## Why This Matters

**Week 1 Recap:**
- You loaded 3 datasets (5,000 customers, 15,312 loans)
- Merged and engineered features (35 features total)
- Identified top predictors (payment behavior features)
- Created production-ready train/test splits (4,000/1,000)

**Week 2 Goal:**
Build ML models that predict loan defaults with high accuracy!

**Why start with a baseline?**

A baseline model is a **simple model** that establishes the minimum performance you should achieve. Any "smart" model (Logistic Regression, Random Forest, XGBoost) should beat the baseline.

**Example:**
```
Baseline: Always predict "Good Loan" → 70% accuracy
Your model: Predicts correctly → Should be > 70%!
```

If your fancy model gets 65% accuracy, something is wrong!

---

## Key Concepts

| Concept | Definition | Why It Matters |
|---------|-----------|----------------|
| **Baseline Model** | Simplest possible model (e.g., always predict majority class) | Establishes minimum performance threshold |
| **Accuracy** | % of correct predictions | Easy to understand but misleading with imbalanced data |
| **Precision** | Of predicted defaults, how many were correct? | Important to avoid false alarms |
| **Recall** | Of actual defaults, how many did we catch? | Critical for not missing bad loans |
| **F1-Score** | Harmonic mean of precision and recall | Balances both metrics |
| **Confusion Matrix** | 2×2 table of predictions vs reality | Shows where model makes mistakes |
| **ROC-AUC** | Area Under ROC Curve | Overall discrimination ability (0.5-1.0) |

---

## Understanding Classification Metrics

### The Confusion Matrix

```
                    Predicted
                Good        Bad
Actual  Good     TN         FP     ← False Positive (False Alarm)
        Bad      FN         TP     ← False Negative (Missed Default)
                 ↑          ↑
        False Negative  True Positive
```

**Real-world meaning:**

- **True Positive (TP)**: Correctly predicted default → Reject bad loan ✅
- **True Negative (TN)**: Correctly predicted good → Approve good loan ✅
- **False Positive (FP)**: Predicted default but was good → Rejected good customer ❌ (Lost revenue)
- **False Negative (FN)**: Predicted good but defaulted → Approved bad loan ❌ (Lost money!)

### Metrics Explained

**1. Accuracy = (TP + TN) / Total**
```
How many predictions were correct overall?

Example: 900 correct out of 1,000 → 90% accuracy

Problem: With 70% Good loans, always predicting "Good" gives 70% accuracy!
```

**2. Precision = TP / (TP + FP)**
```
Of all loans we predicted would default, how many actually did?

Example: Predicted 100 defaults, 80 were correct → 80% precision
Meaning: When we say "default", we're right 80% of the time

High precision = Few false alarms
Low precision = Many false alarms (rejecting good customers)
```

**3. Recall = TP / (TP + FN)**
```
Of all loans that actually defaulted, how many did we catch?

Example: 300 actual defaults, caught 240 → 80% recall
Meaning: We catch 80% of bad loans, miss 20%

High recall = Catch most defaults
Low recall = Miss many defaults (lose money!)
```

**4. F1-Score = 2 × (Precision × Recall) / (Precision + Recall)**
```
Balance between precision and recall

Example: Precision=80%, Recall=80% → F1=80%
Example: Precision=90%, Recall=50% → F1=64% (unbalanced)

Good F1 = Both precision and recall are high
```

**5. ROC-AUC (Area Under Curve)**
```
Measures overall discrimination ability across all thresholds

0.5 = Random guessing (coin flip)
0.7 = Acceptable
0.8 = Good
0.9+ = Excellent

Example: AUC=0.85 means model correctly ranks a random default
higher than a random good loan 85% of the time
```

---

## Business Context

In credit risk scoring:

**Which metric matters most?**

It depends on business goals:

| Goal | Metric to Optimize | Reasoning |
|------|-------------------|-----------|
| **Maximize profit** | F1-Score or ROC-AUC | Balance catching defaults vs approving good loans |
| **Minimize losses** | Recall | Catch as many defaults as possible (conservative) |
| **Minimize false alarms** | Precision | Don't reject too many good customers |
| **Regulatory compliance** | Recall (often mandated) | Must catch high % of defaults |

**Real-world example:**

```
Bank A: High precision (90%), Low recall (50%)
→ Rejects few good customers but approves 50% of bad loans
→ Loses money on defaults

Bank B: Low precision (60%), High recall (95%)
→ Catches 95% of defaults but rejects many good customers
→ Loses revenue from rejected good customers

Bank C: Balanced F1 (80% precision, 80% recall)
→ Best of both worlds
```

We'll optimize for **F1-Score and ROC-AUC** to balance both concerns.

---

## What You'll Build Today

By the end of this guide, you will have:

1. ✅ **Loaded train/test data** from Day 5
2. ✅ **Built baseline model** (majority class predictor)
3. ✅ **Calculated all metrics** (accuracy, precision, recall, F1, AUC)
4. ✅ **Created confusion matrix** visualization
5. ✅ **Created ROC curve** visualization
6. ✅ **Saved baseline results** for comparison with future models
7. ✅ **Understood metric trade-offs** in business context

---

# Step-by-Step Practice

## Step 1: Load Preprocessed Data

Let's load the training and testing datasets we created in Day 5.

**Create a new file:** `day6_practice.py`

Delete everything and replace with this:

```python
"""
Day 6 Practice: Baseline Model & Evaluation Metrics
Credit Risk Scoring System - Week 2

What it does:
1. Loads preprocessed train/test data from Day 5
2. Builds baseline model (majority class predictor)
3. Calculates evaluation metrics (accuracy, precision, recall, F1, AUC)
4. Creates visualizations (confusion matrix, ROC curve)
5. Interprets results in business context
6. Saves baseline results for comparison

Run after: day5_practice.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
from sklearn.dummy import DummyClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("WEEK 2, DAY 6: BASELINE MODEL & EVALUATION METRICS")
print("="*80)

# ============================================================================
# STEP 1: Load Preprocessed Data
# ============================================================================

print("\n📂 STEP 1: Loading Preprocessed Data from Day 5...")

PROCESSED_DIR = Path('credit-risk-api/data/processed')

# Load training data
X_train = pd.read_csv(PROCESSED_DIR / 'X_train_scaled.csv')
y_train = pd.read_csv(PROCESSED_DIR / 'y_train.csv')['target']

# Load test data
X_test = pd.read_csv(PROCESSED_DIR / 'X_test_scaled.csv')
y_test = pd.read_csv(PROCESSED_DIR / 'y_test.csv')['target']

print(f"\n✅ Data loaded successfully!")

print(f"\n📊 Training Set:")
print(f"   X_train shape: {X_train.shape}")
print(f"   y_train shape: {y_train.shape}")
print(f"   Features: {X_train.shape[1]}")
print(f"   Samples: {len(X_train):,}")

print(f"\n📊 Test Set:")
print(f"   X_test shape: {X_test.shape}")
print(f"   y_test shape: {y_test.shape}")
print(f"   Samples: {len(X_test):,}")

print(f"\n🎯 Target Distribution (Training):")
print(y_train.value_counts().sort_index())
print(f"   Good (0): {(y_train == 0).sum():,} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
print(f"   Bad (1):  {(y_train == 1).sum():,} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")

print(f"\n🎯 Target Distribution (Test):")
print(y_test.value_counts().sort_index())
print(f"   Good (0): {(y_test == 0).sum():,} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
print(f"   Bad (1):  {(y_test == 1).sum():,} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")

print(f"\n✅ Stratification verified: Both sets have ~70% Good, ~30% Bad")
```

**Run it:**
```bash
cd /home/user/Nigerian_Inflation_Predictor
python credit-risk-api/guide/week2/day6_practice.py
```

**Expected Output:**
```
================================================================================
WEEK 2, DAY 6: BASELINE MODEL & EVALUATION METRICS
================================================================================

📂 STEP 1: Loading Preprocessed Data from Day 5...

✅ Data loaded successfully!

📊 Training Set:
   X_train shape: (4000, 33)
   y_train shape: (4000,)
   Features: 33
   Samples: 4,000

📊 Test Set:
   X_test shape: (1000, 33)
   y_test shape: (1000,)
   Samples: 1,000

🎯 Target Distribution (Training):
0    2800
1    1200
Name: target, dtype: int64
   Good (0): 2,800 (70.0%)
   Bad (1):  1,200 (30.0%)

🎯 Target Distribution (Test):
0    700
1    300
Name: target, dtype: int64
   Good (0): 700 (70.0%)
   Bad (1):  300 (30.0%)

✅ Stratification verified: Both sets have ~70% Good, ~30% Bad
```

**What just happened?**

1. **Loaded 4 files** - X_train, X_test, y_train, y_test (created in Day 5)
2. **Verified shapes** - 4,000 training samples, 1,000 test samples, 33 features
3. **Checked stratification** - Both sets maintain 70/30 class distribution
4. **Ready for modeling** - All features are already scaled (mean=0, std=1)

---

## Step 2: Build Baseline Model

Now let's build the simplest possible model: always predict the majority class (Good Loan = 0).

**Add this to your `day6_practice.py` file:**

```python
# ============================================================================
# STEP 2: Build Baseline Model (Majority Class Predictor)
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Build Baseline Model")
print("="*80)

print("\n🔧 What is a baseline model?")
print("   A baseline is the SIMPLEST possible model that establishes")
print("   minimum performance. Any 'smart' model should beat it!")

print("\n📊 Strategy: ALWAYS predict the majority class")
print(f"   Majority class: Good Loan (0) - {(y_train == 0).sum()/len(y_train)*100:.1f}% of training data")
print("   Prediction rule: Predict EVERY loan is Good")

# Create baseline model using DummyClassifier
baseline = DummyClassifier(strategy='most_frequent', random_state=42)

# Train baseline (just learns to predict majority class)
print("\n🔧 Training baseline model...")
baseline.fit(X_train, y_train)
print("✅ Baseline trained (learned to always predict class 0)")

# Make predictions on test set
print("\n🔮 Making predictions on test set...")
y_pred_baseline = baseline.predict(X_test)
y_pred_proba_baseline = baseline.predict_proba(X_test)[:, 1]  # Probability of class 1 (Bad)

print(f"✅ Predictions generated for {len(y_pred_baseline):,} test samples")

# Show prediction distribution
print(f"\n📊 Baseline Predictions on Test Set:")
unique, counts = np.unique(y_pred_baseline, return_counts=True)
for val, count in zip(unique, counts):
    label = "Good" if val == 0 else "Bad"
    print(f"   Predicted {label} ({val}): {count:,} ({count/len(y_pred_baseline)*100:.1f}%)")

print(f"\n💡 As expected: Baseline predicts EVERY loan is Good (0)")
```

**Run it again:**
```bash
python credit-risk-api/guide/week2/day6_practice.py
```

**Expected Output:**
```
================================================================================
STEP 2: Build Baseline Model
================================================================================

🔧 What is a baseline model?
   A baseline is the SIMPLEST possible model that establishes
   minimum performance. Any 'smart' model should beat it!

📊 Strategy: ALWAYS predict the majority class
   Majority class: Good Loan (0) - 70.0% of training data
   Prediction rule: Predict EVERY loan is Good

🔧 Training baseline model...
✅ Baseline trained (learned to always predict class 0)

🔮 Making predictions on test set...
✅ Predictions generated for 1,000 test samples

📊 Baseline Predictions on Test Set:
   Predicted Good (0): 1,000 (100.0%)

💡 As expected: Baseline predicts EVERY loan is Good (0)
```

**What just happened?**

1. **Created DummyClassifier** - Scikit-learn's baseline model class
2. **Set strategy='most_frequent'** - Always predict the most common class
3. **Trained baseline** - It just memorized "always predict 0"
4. **Made predictions** - Every single test sample predicted as Good (0)

**Why this matters:**

This naive strategy achieves 70% accuracy (since 70% of loans are actually Good). But it's **terrible** for business:
- Approves ALL loans (including 300 bad ones!)
- Never catches any defaults
- Bank loses money on 300 defaulted loans

Any real model should beat this baseline significantly!

---

## Step 3: Calculate Evaluation Metrics

Let's calculate all the important metrics to evaluate our baseline.

**Add this to your `day6_practice.py` file:**

```python
# ============================================================================
# STEP 3: Calculate Evaluation Metrics
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Calculate Evaluation Metrics")
print("="*80)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_baseline)
precision = precision_score(y_test, y_pred_baseline, zero_division=0)
recall = recall_score(y_test, y_pred_baseline)
f1 = f1_score(y_test, y_pred_baseline)

# ROC-AUC (for baseline, probabilities are constant, so AUC may be undefined)
try:
    roc_auc = roc_auc_score(y_test, y_pred_proba_baseline)
except:
    roc_auc = 0.5  # Random guessing

print(f"\n📊 BASELINE MODEL PERFORMANCE:")
print(f"{'Metric':<20} {'Value':>10} {'Interpretation'}")
print("-" * 70)

print(f"{'Accuracy':<20} {accuracy:>10.1%}   {accuracy*100:.1f}% of predictions correct")
print(f"{'Precision':<20} {precision:>10.1%}   Of predicted defaults, {precision*100:.1f}% were correct")
print(f"{'Recall':<20} {recall:>10.1%}   Of actual defaults, caught {recall*100:.1f}%")
print(f"{'F1-Score':<20} {f1:>10.1%}   Balance of precision and recall")
print(f"{'ROC-AUC':<20} {roc_auc:>10.3f}   Overall discrimination (0.5=random, 1.0=perfect)")

print(f"\n💡 Interpretation:")
print(f"   • Accuracy: {accuracy*100:.1f}% - Seems high, but misleading!")
print(f"   • Precision: {precision*100:.1f}% - Never predicts default, so 0 true positives")
print(f"   • Recall: {recall*100:.1f}% - Catches ZERO defaults (missed all 300!)")
print(f"   • F1-Score: {f1*100:.1f}% - Terrible overall performance")
print(f"   • ROC-AUC: {roc_auc:.3f} - Random guessing")

print(f"\n⚠️  BUSINESS IMPACT:")
print(f"   • Approved ALL 1,000 loans (including 300 bad ones)")
print(f"   • Missed 300 defaults → Potential losses!")
print(f"   • This is why we need smarter models!")

# Detailed classification report
print(f"\n📊 Detailed Classification Report:")
print(classification_report(y_test, y_pred_baseline,
                           target_names=['Good (0)', 'Bad (1)'],
                           digits=3))
```

**Run it again:**
```bash
python credit-risk-api/guide/week2/day6_practice.py
```

**Expected Output:**
```
================================================================================
STEP 3: Calculate Evaluation Metrics
================================================================================

📊 BASELINE MODEL PERFORMANCE:
Metric               Value Interpretation
----------------------------------------------------------------------
Accuracy              70.0%   70.0% of predictions correct
Precision              0.0%   Of predicted defaults, 0.0% were correct
Recall                 0.0%   Of actual defaults, caught 0.0%
F1-Score               0.0%   Balance of precision and recall
ROC-AUC               0.500   Overall discrimination (0.5=random, 1.0=perfect)

💡 Interpretation:
   • Accuracy: 70.0% - Seems high, but misleading!
   • Precision: 0.0% - Never predicts default, so 0 true positives
   • Recall: 0.0% - Catches ZERO defaults (missed all 300!)
   • F1-Score: 0.0% - Terrible overall performance
   • ROC-AUC: 0.500 - Random guessing

⚠️  BUSINESS IMPACT:
   • Approved ALL 1,000 loans (including 300 bad ones)
   • Missed 300 defaults → Potential losses!
   • This is why we need smarter models!

📊 Detailed Classification Report:
              precision    recall  f1-score   support

   Good (0)      0.700     1.000     0.824       700
    Bad (1)      0.000     0.000     0.000       300

    accuracy                          0.700      1000
   macro avg      0.350     0.500     0.412      1000
weighted avg      0.490     0.700     0.577      1000
```

**What just happened?**

1. **Calculated 5 key metrics** - accuracy, precision, recall, F1, ROC-AUC
2. **Accuracy is 70%** - But this is misleading! It's just predicting the majority class
3. **Precision & Recall are 0%** - Never predicts "Bad", so catches zero defaults
4. **F1-Score is 0%** - Overall performance is terrible
5. **ROC-AUC is 0.5** - No better than random guessing

**The Accuracy Paradox:**

70% accuracy sounds good, but the baseline:
- Approves 300 bad loans (loses money)
- Rejects 0 good loans (no false alarms, but approves everything!)
- Provides ZERO value to the business

This demonstrates why **accuracy alone is meaningless** for imbalanced datasets!

---

## Step 4: Create Confusion Matrix Visualization

Let's visualize where the baseline model makes mistakes.

**Add this to your `day6_practice.py` file:**

```python
# ============================================================================
# STEP 4: Create Confusion Matrix Visualization
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Confusion Matrix Visualization")
print("="*80)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_baseline)

print(f"\n📊 Confusion Matrix (Raw Counts):")
print(f"\n                Predicted")
print(f"               Good   Bad")
print(f"Actual  Good    {cm[0,0]:4d}   {cm[0,1]:4d}")
print(f"        Bad     {cm[1,0]:4d}   {cm[1,1]:4d}")

print(f"\n💡 Reading the Confusion Matrix:")
print(f"   • Top-left ({cm[0,0]}): True Negatives - Correctly predicted Good loans")
print(f"   • Top-right ({cm[0,1]}): False Positives - Predicted Bad but was Good (false alarms)")
print(f"   • Bottom-left ({cm[1,0]}): False Negatives - Predicted Good but was Bad (MISSED DEFAULTS!)")
print(f"   • Bottom-right ({cm[1,1]}): True Positives - Correctly predicted Bad loans")

print(f"\n⚠️  Baseline Problem:")
print(f"   • Correctly identified {cm[0,0]:,} Good loans ✅")
print(f"   • Missed {cm[1,0]:,} Bad loans ❌ (approved them all!)")
print(f"   • Never predicted any loan as Bad")

# Create output directory
output_dir = Path('credit-risk-api/results')
output_dir.mkdir(parents=True, exist_ok=True)

# Plot confusion matrix
print(f"\n📊 Creating confusion matrix heatmap...")

fig, ax = plt.subplots(figsize=(8, 6))

# Create heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Good', 'Bad'],
            yticklabels=['Good', 'Bad'],
            cbar_kws={'label': 'Count'},
            ax=ax)

ax.set_title('Baseline Model - Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel('Actual Class', fontsize=12)
ax.set_xlabel('Predicted Class', fontsize=12)

# Add metric annotations
accuracy_text = f'Accuracy: {accuracy:.1%}'
recall_text = f'Recall: {recall:.1%}'
precision_text = f'Precision: {precision:.1%}'
f1_text = f'F1-Score: {f1:.1%}'

plt.text(0.5, -0.15, f'{accuracy_text} | {precision_text} | {recall_text} | {f1_text}',
         ha='center', transform=ax.transAxes, fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'day6_baseline_confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir}/day6_baseline_confusion_matrix.png")

plt.close()
```

**Run it again:**
```bash
python credit-risk-api/guide/week2/day6_practice.py
```

**Expected Output:**
```
================================================================================
STEP 4: Confusion Matrix Visualization
================================================================================

📊 Confusion Matrix (Raw Counts):

                Predicted
               Good   Bad
Actual  Good    700     0
        Bad     300     0

💡 Reading the Confusion Matrix:
   • Top-left (700): True Negatives - Correctly predicted Good loans
   • Top-right (0): False Positives - Predicted Bad but was Good (false alarms)
   • Bottom-left (300): False Negatives - Predicted Good but was Bad (MISSED DEFAULTS!)
   • Bottom-right (0): True Positives - Correctly predicted Bad loans

⚠️  Baseline Problem:
   • Correctly identified 700 Good loans ✅
   • Missed 300 Bad loans ❌ (approved them all!)
   • Never predicted any loan as Bad

📊 Creating confusion matrix heatmap...
✅ Saved: credit-risk-api/results/day6_baseline_confusion_matrix.png
```

**What just happened?**

1. **Created confusion matrix** - 2×2 table of predictions vs actual
2. **Visualized results** - Heatmap shows where model makes mistakes
3. **Identified the problem** - All 300 defaults were missed (bottom-left cell)

**Confusion Matrix Breakdown:**

```
True Negatives (700):  ✅ Good loans correctly approved
False Positives (0):   ❌ Good loans incorrectly rejected (none)
False Negatives (300): ❌❌ Bad loans incorrectly approved (BIG PROBLEM!)
True Positives (0):    ✅ Bad loans correctly rejected (none caught)
```

The visualization clearly shows the baseline's fatal flaw: it approves every loan, including all the bad ones!

---

## Step 5: Create ROC Curve Visualization

Let's create an ROC curve to visualize the model's discrimination ability.

**Add this to your `day6_practice.py` file:**

```python
# ============================================================================
# STEP 5: Create ROC Curve Visualization
# ============================================================================

print("\n" + "="*80)
print("STEP 5: ROC Curve Visualization")
print("="*80)

print(f"\n📊 What is an ROC Curve?")
print("   ROC (Receiver Operating Characteristic) shows trade-off between:")
print("   • True Positive Rate (Recall) - catching defaults")
print("   • False Positive Rate - false alarms (rejecting good loans)")
print()
print("   A perfect model hugs the top-left corner (high TPR, low FPR)")
print("   A random model follows the diagonal line (AUC = 0.5)")

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_baseline)

print(f"\n🔢 ROC Curve Data:")
print(f"   False Positive Rate (FPR): min={fpr.min():.3f}, max={fpr.max():.3f}")
print(f"   True Positive Rate (TPR): min={tpr.min():.3f}, max={tpr.max():.3f}")
print(f"   Number of thresholds: {len(thresholds)}")
print(f"   ROC-AUC Score: {roc_auc:.3f}")

print(f"\n💡 Baseline ROC-AUC: {roc_auc:.3f}")
if roc_auc < 0.6:
    print("   ⚠️  No better than random guessing!")
elif roc_auc < 0.7:
    print("   ⚠️  Poor discrimination")
elif roc_auc < 0.8:
    print("   ✅ Acceptable")
elif roc_auc < 0.9:
    print("   ✅ Good")
else:
    print("   ✅ Excellent!")

# Plot ROC curve
print(f"\n📊 Creating ROC curve plot...")

fig, ax = plt.subplots(figsize=(10, 8))

# Plot ROC curve
ax.plot(fpr, tpr, color='blue', lw=2, label=f'Baseline (AUC = {roc_auc:.3f})')

# Plot random classifier line
ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier (AUC = 0.500)')

# Styling
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR) / Recall', fontsize=12)
ax.set_title('ROC Curve - Baseline Model', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)

# Add annotations
ax.text(0.6, 0.2, f'AUC = {roc_auc:.3f}', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / 'day6_baseline_roc_curve.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir}/day6_baseline_roc_curve.png")

plt.close()
```

**Run it again:**
```bash
python credit-risk-api/guide/week2/day6_practice.py
```

**Expected Output:**
```
================================================================================
STEP 5: ROC Curve Visualization
================================================================================

📊 What is an ROC Curve?
   ROC (Receiver Operating Characteristic) shows trade-off between:
   • True Positive Rate (Recall) - catching defaults
   • False Positive Rate - false alarms (rejecting good loans)

   A perfect model hugs the top-left corner (high TPR, low FPR)
   A random model follows the diagonal line (AUC = 0.5)

🔢 ROC Curve Data:
   False Positive Rate (FPR): min=0.000, max=1.000
   True Positive Rate (TPR): min=0.000, max=1.000
   Number of thresholds: 2
   ROC-AUC Score: 0.500

💡 Baseline ROC-AUC: 0.500
   ⚠️  No better than random guessing!

📊 Creating ROC curve plot...
✅ Saved: credit-risk-api/results/day6_baseline_roc_curve.png
```

**What just happened?**

1. **Calculated ROC curve** - TPR vs FPR at different thresholds
2. **Computed AUC** - Area under the ROC curve (0.5 for baseline)
3. **Created visualization** - Shows baseline follows random line

**ROC Curve Interpretation:**

```
Perfect Model (AUC = 1.0): Straight line to top-left, then across
Good Model (AUC = 0.8):    Bows toward top-left corner
Random Model (AUC = 0.5):  Diagonal line (baseline follows this!)
```

The baseline's AUC of 0.5 confirms it has **zero discrimination ability** - no better than flipping a coin!

---

## Step 6: Save Baseline Results

Let's save the baseline results for comparison with future models.

**Add this to your `day6_practice.py` file:**

```python
# ============================================================================
# STEP 6: Save Baseline Results
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Save Baseline Results")
print("="*80)

# Create models directory
MODELS_DIR = Path('credit-risk-api/models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Save baseline model
model_path = MODELS_DIR / 'baseline_model.pkl'
joblib.dump(baseline, model_path)
print(f"\n💾 Baseline model saved: {model_path}")

# Save baseline results as dictionary
baseline_results = {
    'model_name': 'Baseline (Majority Class)',
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'confusion_matrix': cm.tolist(),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'features': X_train.shape[1]
}

# Save results as CSV for easy comparison
results_df = pd.DataFrame([baseline_results])
results_path = MODELS_DIR / 'model_results.csv'
results_df.to_csv(results_path, index=False)
print(f"💾 Results saved: {results_path}")

# Save as pickle for later use
results_pkl_path = MODELS_DIR / 'baseline_results.pkl'
joblib.dump(baseline_results, results_pkl_path)
print(f"💾 Results (pickle) saved: {results_pkl_path}")

print(f"\n✅ All baseline artifacts saved!")
print(f"   Model: {model_path}")
print(f"   Results CSV: {results_path}")
print(f"   Results PKL: {results_pkl_path}")
```

**Run it again:**
```bash
python credit-risk-api/guide/week2/day6_practice.py
```

**Expected Output:**
```
================================================================================
STEP 6: Save Baseline Results
================================================================================

💾 Baseline model saved: credit-risk-api/models/baseline_model.pkl
💾 Results saved: credit-risk-api/models/model_results.csv
💾 Results (pickle) saved: credit-risk-api/models/baseline_results.pkl

✅ All baseline artifacts saved!
   Model: credit-risk-api/models/baseline_model.pkl
   Results CSV: credit-risk-api/models/model_results.csv
   Results PKL: credit-risk-api/models/baseline_results.pkl
```

**What just happened?**

1. **Saved baseline model** - Can be loaded later for predictions
2. **Saved results as CSV** - Easy to compare with future models
3. **Saved results as pickle** - Complete dictionary for analysis

These files will be used tomorrow to compare Logistic Regression against the baseline!

---

## Step 7: Summary Report

Let's generate a comprehensive summary of what we learned today.

**Add this to your `day6_practice.py` file:**

```python
# ============================================================================
# STEP 7: Day 6 Summary Report
# ============================================================================

print("\n" + "="*80)
print("📝 DAY 6 SUMMARY REPORT")
print("="*80)

print(f"""
🎉 Baseline Model & Evaluation Metrics Complete!

📊 BASELINE MODEL RESULTS:
   Model Type: Majority Class Predictor (always predict "Good")
   Strategy: Predict class 0 for every sample

   Performance Metrics:
   ✅ Accuracy:   {accuracy*100:>6.1f}%  ← Misleading (just majority class %)
   ❌ Precision:  {precision*100:>6.1f}%  ← Never predicts "Bad", so 0 TP
   ❌ Recall:     {recall*100:>6.1f}%  ← Catches ZERO defaults
   ❌ F1-Score:   {f1*100:>6.1f}%  ← Terrible overall
   ❌ ROC-AUC:    {roc_auc:>6.3f}  ← Random guessing

📊 CONFUSION MATRIX:
                    Predicted
                 Good    Bad
   Actual  Good   {cm[0,0]:3d}      {cm[0,1]:3d}   ← {cm[0,0]} correct, {cm[0,1]} false alarms
           Bad    {cm[1,0]:3d}      {cm[1,1]:3d}   ← {cm[1,0]} MISSED, {cm[1,1]} caught

💡 KEY INSIGHTS:

1. Accuracy Paradox:
   • 70% accuracy sounds good but is meaningless
   • Just predicting majority class achieves this
   • Provides ZERO business value

2. Why Baseline Fails:
   • Approves ALL loans (including 300 bad ones)
   • Never catches any defaults
   • Bank loses money on every bad loan approved

3. Business Impact:
   • Missed 300 defaults out of 300 (0% recall)
   • Potential losses: 300 × average_loan_amount
   • No risk management whatsoever

4. Metric Trade-offs:
   • High accuracy ≠ good model (for imbalanced data)
   • Need to balance precision (minimize false alarms)
     and recall (catch defaults)
   • F1-Score and ROC-AUC are better overall metrics

🎯 WHAT MAKES A GOOD CREDIT RISK MODEL:

   Minimum Requirements:
   • Accuracy > 70% (better than baseline)
   • Recall > 50% (catch at least half of defaults)
   • Precision > 50% (more right than wrong)
   • F1-Score > 60% (balanced performance)
   • ROC-AUC > 0.70 (good discrimination)

   Ideal Performance:
   • Recall > 80% (catch most defaults)
   • Precision > 70% (minimize false alarms)
   • F1-Score > 75% (strong balance)
   • ROC-AUC > 0.85 (excellent discrimination)

📁 FILES CREATED TODAY:
   ✅ day6_baseline_confusion_matrix.png
   ✅ day6_baseline_roc_curve.png
   ✅ baseline_model.pkl
   ✅ baseline_results.pkl
   ✅ model_results.csv

🔧 EVALUATION METRICS LEARNED:
   ✅ Accuracy - % correct (misleading for imbalanced data)
   ✅ Precision - Of predicted defaults, % correct
   ✅ Recall - Of actual defaults, % caught
   ✅ F1-Score - Balance of precision & recall
   ✅ Confusion Matrix - Where model makes mistakes
   ✅ ROC Curve - TPR vs FPR trade-off
   ✅ ROC-AUC - Overall discrimination ability

📅 TOMORROW (Day 7): LOGISTIC REGRESSION
   You'll build your first REAL machine learning model:
   • Logistic Regression classifier
   • Train on 4,000 samples
   • Achieve >70% F1-Score (beat baseline!)
   • Interpret coefficients (feature importance)
   • Optimize decision threshold

🎉 DAY 6 COMPLETE!
""")

print("="*80)
print("BASELINE ESTABLISHED - READY FOR REAL MODELS!")
print("="*80)

print(f"\n🎯 Goal for Tomorrow:")
print(f"   Beat the baseline on ALL metrics:")
print(f"   • F1-Score > {f1:.1%} (currently {f1:.1%})")
print(f"   • Recall > {recall:.1%} (currently {recall:.1%})")
print(f"   • ROC-AUC > {roc_auc:.3f} (currently {roc_auc:.3f})")

print(f"\n💪 You now understand:")
print("   ✅ How to evaluate classification models")
print("   ✅ Why accuracy alone is misleading")
print("   ✅ The importance of recall for catching defaults")
print("   ✅ How to interpret confusion matrices")
print("   ✅ What ROC-AUC measures")

print("\n🚀 Next: Build Logistic Regression and beat this baseline!")
```

**Run it one final time:**
```bash
python credit-risk-api/guide/week2/day6_practice.py
```

**Expected Output:**
```
================================================================================
📝 DAY 6 SUMMARY REPORT
================================================================================

🎉 Baseline Model & Evaluation Metrics Complete!

📊 BASELINE MODEL RESULTS:
   Model Type: Majority Class Predictor (always predict "Good")
   Strategy: Predict class 0 for every sample

   Performance Metrics:
   ✅ Accuracy:     70.0%  ← Misleading (just majority class %)
   ❌ Precision:     0.0%  ← Never predicts "Bad", so 0 TP
   ❌ Recall:        0.0%  ← Catches ZERO defaults
   ❌ F1-Score:      0.0%  ← Terrible overall
   ❌ ROC-AUC:      0.500  ← Random guessing

📊 CONFUSION MATRIX:
                    Predicted
                 Good    Bad
   Actual  Good   700      0   ← 700 correct, 0 false alarms
           Bad    300      0   ← 300 MISSED, 0 caught

[... rest of summary output ...]

🚀 Next: Build Logistic Regression and beat this baseline!
```

**What just happened?**

1. **Generated comprehensive summary** - Complete Day 6 overview
2. **Highlighted baseline limitations** - Why it fails in practice
3. **Set goals for tomorrow** - Beat the baseline on all metrics
4. **Explained metric trade-offs** - Business context for each metric

---

## What You Accomplished Today

✅ **Loaded Preprocessed Data**
   - 4,000 training samples, 1,000 test samples
   - 33 scaled features ready for modeling

✅ **Built Baseline Model**
   - Majority class predictor (always predict "Good")
   - Established minimum performance threshold

✅ **Calculated All Metrics**
   - Accuracy: 70% (misleading!)
   - Precision: 0% (never predicts default)
   - Recall: 0% (misses all defaults)
   - F1-Score: 0% (terrible)
   - ROC-AUC: 0.5 (random)

✅ **Created Visualizations**
   - Confusion matrix heatmap
   - ROC curve plot

✅ **Understood Metric Trade-offs**
   - Why accuracy is misleading
   - Importance of recall for catching defaults
   - Balance between precision and recall

✅ **Saved Baseline Results**
   - For comparison with future models

---

## Key Takeaways

1. **Baseline models establish minimum performance**
   - Any real model should beat the baseline
   - Provides context for evaluating improvements

2. **Accuracy is misleading for imbalanced data**
   - 70% accuracy just from predicting majority class
   - Use F1-Score and ROC-AUC instead

3. **Recall is critical for credit risk**
   - Missing defaults costs money
   - Higher recall = catch more bad loans
   - But too high recall = reject good customers

4. **Metrics have business implications**
   - Precision: minimize false alarms (rejected good customers)
   - Recall: minimize missed defaults (approved bad loans)
   - F1-Score: balance both concerns

5. **Confusion matrix shows the full picture**
   - See exactly where model makes mistakes
   - Identify False Negatives (missed defaults) vs False Positives (false alarms)

---

## Common Questions

**Q: Why build a baseline if it's terrible?**

A: To establish a reference point! If your fancy model gets 68% F1-Score, you know something's wrong. Baseline ensures you're actually improving, not just getting lucky.

**Q: Why is 70% accuracy bad if baseline achieves it?**

A: Because it provides zero value! Approving all loans gives 70% accuracy but loses money on 300 defaults. Real models need to BEAT baseline significantly.

**Q: Which metric should I optimize for?**

A: Depends on business goals:
- Minimize losses → High recall (catch defaults)
- Minimize false alarms → High precision (don't reject good customers)
- Balance both → High F1-Score or ROC-AUC

**Q: What's a good ROC-AUC score?**

A:
- 0.5 = Random guessing (baseline)
- 0.7 = Acceptable
- 0.8 = Good
- 0.9+ = Excellent

**Q: Can I have 100% accuracy?**

A: Theoretically yes, but practically no. Real-world data has noise and overlap. Aim for 85-90% on test set for this problem.

---

## Tomorrow: Day 7 - Logistic Regression

**What you'll build:**

1. **Logistic Regression model** - First real ML model
2. **Train on 4,000 samples** - Learn from payment behavior features
3. **Achieve >70% F1-Score** - Beat the baseline significantly
4. **Interpret coefficients** - Understand which features matter most
5. **Optimize threshold** - Balance precision vs recall for business needs

**Expected Performance:**
- Accuracy: ~75-80%
- Precision: ~65-75%
- Recall: ~60-70%
- F1-Score: ~65-70%
- ROC-AUC: ~0.75-0.80

After Day 7, you'll have your first production-ready credit risk model!

---

## Files Created Today

```
credit-risk-api/
├── results/
│   ├── day6_baseline_confusion_matrix.png
│   └── day6_baseline_roc_curve.png
├── models/
│   ├── baseline_model.pkl
│   ├── baseline_results.pkl
│   └── model_results.csv
└── guide/
    └── week2/
        └── day6_practice.py (Complete baseline script)
```

---

**🎉 Congratulations!** You've completed Day 6 and understand how to properly evaluate classification models!

**Take a break! Tomorrow you'll build your first REAL ML model! 🚀**
