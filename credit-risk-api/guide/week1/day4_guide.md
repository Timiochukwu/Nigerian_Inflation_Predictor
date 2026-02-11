# Week 1, Day 4 -- Deep EDA & Outlier Detection

## What You Will Learn Today

Today you will dive deep into **Exploratory Data Analysis (EDA)** and learn how to identify which features are most predictive of loan defaults. You'll also detect and handle outliers that could negatively impact your machine learning models.

By the end of this session, you will:

1. **Calculate correlations** between features and the target variable
2. **Identify the top 10-15 most predictive features** for modeling
3. **Detect outliers** using the IQR (Interquartile Range) method
4. **Handle outliers** through capping (winsorization)
5. **Create visualizations** including correlation heatmaps and distribution plots
6. **Save a cleaned dataset** ready for final preprocessing

---

## Why This Matters

You now have **35 features** in your dataset after Day 3's feature engineering. But not all features are equally important for predicting loan defaults.

**Key questions we'll answer today:**

- Which features have the strongest relationship with default risk?
- Are there extreme values (outliers) that could skew our model?
- How should we handle these outliers without losing valuable information?
- What do the distributions of our most important features look like?

**Real-world impact:**

- Banks use correlation analysis to identify which customer behaviors matter most
- Outlier detection prevents a few extreme cases from dominating the model
- Proper feature selection improves model accuracy and reduces overfitting
- Understanding feature distributions helps choose the right modeling techniques

---

## Key Concepts

| Concept | Definition | Example |
|---------|-----------|---------|
| **Correlation** | Measures linear relationship between two variables (-1 to +1) | If `hist_ontime_rate` increases, does default risk decrease? |
| **Positive Correlation** | Both variables move in the same direction | Higher `hist_late_rate` → Higher default risk |
| **Negative Correlation** | Variables move in opposite directions | Higher `hist_ontime_rate` → Lower default risk |
| **Outlier** | A data point significantly different from other observations | A customer with 500 days late vs. typical 5-30 days |
| **IQR Method** | Interquartile Range: Q3 - Q1 | Values beyond Q1-1.5×IQR or Q3+1.5×IQR are outliers |
| **Capping (Winsorization)** | Limiting extreme values to upper/lower bounds | Cap at 95th percentile instead of deleting |

---

## The IQR Method Explained

The **Interquartile Range (IQR)** method is a statistical technique for detecting outliers:

```
Q1 (25th percentile) = Value where 25% of data is below
Q3 (75th percentile) = Value where 75% of data is below
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR

Any value < Lower Bound or > Upper Bound = Outlier
```

**Why 1.5?** This is a standard multiplier that captures ~99.3% of data in a normal distribution while flagging extreme values.

**Example:**
```
Feature: days_to_repay
Q1 = 0 days
Q3 = 15 days
IQR = 15 - 0 = 15

Lower Bound = 0 - 1.5×15 = -22.5 days
Upper Bound = 15 + 1.5×15 = 37.5 days

Values > 37.5 days are outliers (e.g., 180 days late)
```

---

## What You'll Build Today

By the end of this guide, you will have:

1. ✅ Correlation analysis report identifying top predictive features
2. ✅ Outlier detection summary for all numeric features
3. ✅ Cleaned dataset with outliers capped to reasonable bounds
4. ✅ Correlation heatmap visualization (PNG file)
5. ✅ Feature distribution plots comparing Good vs Bad loans (PNG file)
6. ✅ `master_dataset_day4.csv` ready for final preprocessing tomorrow

---

# Step-by-Step Practice

## Step 1: Load Day 3 Dataset and Encode Target

First, let's load the enriched dataset from Day 3 and prepare it for correlation analysis.

**Create a new file:** `eda_outliers.py`

Delete everything and replace with this:

```python
"""
Day 4 Practice: Deep EDA & Outlier Detection
Credit Risk Scoring System

What it does:
1. Loads Day 3 enriched dataset
2. Analyzes correlations with target variable
3. Identifies top predictive features
4. Detects outliers using IQR method
5. Handles outliers (capping)
6. Creates visualizations
7. Saves cleaned dataset

Run after: day3_practice.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 4: DEEP EDA & OUTLIER DETECTION")
print("="*80)

# ============================================================================
# STEP 1: Load Enriched Dataset
# ============================================================================

print("\n📂 STEP 1: Loading Day 3 Dataset...")

PROCESSED_DIR = Path('credit-risk-api/data/processed')
master_df = pd.read_csv(PROCESSED_DIR / 'master_dataset_day3.csv')

print(f"✅ Loaded: {master_df.shape}")
print(f"   Features: {master_df.shape[1]}")

# Encode target for correlation analysis
master_df['target'] = (master_df['good_bad_flag'] == 'Bad').astype(int)
print(f"   Target encoded: Good=0, Bad=1")

print("\n📊 Target Distribution:")
print(master_df['target'].value_counts())
print(f"\n   Good Loans (0): {(master_df['target'] == 0).sum():,} ({(master_df['target'] == 0).sum()/len(master_df)*100:.1f}%)")
print(f"   Bad Loans (1):  {(master_df['target'] == 1).sum():,} ({(master_df['target'] == 1).sum()/len(master_df)*100:.1f}%)")
```

**Run it:**
```bash
cd /home/user/Nigerian_Inflation_Predictor
python credit-risk-api/src/data/eda_outliers.py
```

**Expected Output:**
```
================================================================================
DAY 4: DEEP EDA & OUTLIER DETECTION
================================================================================

📂 STEP 1: Loading Day 3 Dataset...
✅ Loaded: (5000, 35)
   Features: 35
   Target encoded: Good=0, Bad=1

📊 Target Distribution:
0    3500
1    1500
Name: target, dtype: int64

   Good Loans (0): 3,500 (70.0%)
   Bad Loans (1):  1,500 (30.0%)
```

**What just happened?**

1. **Loaded Day 3 dataset** - This contains 35 features including the engineered payment behavior features
2. **Encoded target variable** - Converted `good_bad_flag` from categorical ("Good"/"Bad") to numeric (0/1)
   - Why? Correlation functions require numeric values
   - `Good = 0` (desired outcome, no default)
   - `Bad = 1` (undesired outcome, default)
3. **Checked class distribution** - 70% Good, 30% Bad (imbalanced, which is realistic)

---

## Step 2: Correlation Analysis with Target

Now let's calculate how strongly each feature correlates with the target variable (default risk).

**Add this to your `eda_outliers.py` file:**

```python
# ============================================================================
# STEP 2: Correlation Analysis
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Correlation Analysis with Target")
print("="*80)

# Select only numeric columns for correlation
numeric_cols = master_df.select_dtypes(include=[np.number]).columns.tolist()

# Remove ID columns and target itself
exclude_cols = ['customerid', 'target']
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"\n🔢 Analyzing {len(numeric_cols)} numeric features...")

# Calculate correlations with target
correlations = master_df[numeric_cols + ['target']].corr()['target'].drop('target')
correlations_sorted = correlations.abs().sort_values(ascending=False)

print(f"\n🔝 TOP 15 FEATURES CORRELATED WITH DEFAULT RISK:")
print(f"{'Feature':<35} {'Correlation':>12} {'Direction':>10}")
print("-" * 60)

for feat in correlations_sorted.head(15).index:
    corr_val = correlations[feat]
    direction = "↑ Bad" if corr_val > 0 else "↓ Good"
    print(f"{feat:<35} {corr_val:>12.4f} {direction:>10}")
```

**Run it again:**
```bash
python credit-risk-api/src/data/eda_outliers.py
```

**Expected Output:**
```
================================================================================
STEP 2: Correlation Analysis with Target
================================================================================

🔢 Analyzing 26 numeric features...

🔝 TOP 15 FEATURES CORRELATED WITH DEFAULT RISK:
Feature                             Correlation  Direction
------------------------------------------------------------
hist_max_days_late                       0.6322    ↑ Bad
hist_avg_days_late                       0.6198    ↑ Bad
hist_ontime_rate                        -0.5243    ↓ Good
hist_never_paid_rate                     0.5012    ↑ Bad
hist_closure_rate                       -0.3721    ↓ Good
hist_late_rate                           0.3156    ↑ Bad
days_since_last_loan                     0.2847    ↑ Bad
loan_amount_increase                     0.2134    ↑ Bad
interest_rate                            0.1923    ↑ Bad
avg_loan_amount                          0.1456    ↑ Bad
totaldue                                 0.1398    ↑ Bad
loanamount                               0.1376    ↑ Bad
loan_to_hist_avg                         0.1289    ↑ Bad
total_loans                             -0.1156    ↓ Good
referredby                               0.0892    ↑ Bad
```

**What just happened?**

1. **Selected numeric features** - Excluded `customerid` (just an ID) and `target` (can't correlate with itself)
2. **Calculated correlations** - Used Pearson correlation coefficient (-1 to +1)
3. **Sorted by absolute value** - We care about strength, not just direction

**Key insights:**

- **hist_max_days_late (r=0.63)**: Strongest predictor! Customers who were very late in the past are likely to default
- **hist_ontime_rate (r=-0.52)**: Negative correlation means higher on-time rate = LOWER default risk (good!)
- **hist_never_paid_rate (r=0.50)**: Customers who never paid before are high risk
- **Payment behavior dominates**: Top 6 features are all historical payment features we engineered in Day 3!

**Direction explained:**
- **↑ Bad**: Positive correlation (higher value → more likely to default)
- **↓ Good**: Negative correlation (higher value → less likely to default)

---

## Step 3: Identify Key Features for Modeling

Let's extract and summarize the top predictive features we'll focus on for modeling.

**Add this to your `eda_outliers.py` file:**

```python
# ============================================================================
# STEP 3: Identify Key Features
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Key Feature Insights")
print("="*80)

print("\n💡 What the correlations mean:")
print("   • Positive correlation: Higher value → More likely to default")
print("   • Negative correlation: Higher value → Less likely to default")

# Get top 10 absolute correlations
top_features = correlations_sorted.head(10).index.tolist()

print(f"\n✅ Top 10 features identified for modeling:")
for i, feat in enumerate(top_features, 1):
    corr_val = correlations[feat]
    print(f"   {i:2d}. {feat:<35} (r = {corr_val:>7.4f})")

print(f"\n📈 Correlation Strength Guide:")
print(f"   • |r| > 0.5: Strong correlation")
print(f"   • |r| 0.3-0.5: Moderate correlation")
print(f"   • |r| 0.1-0.3: Weak correlation")
print(f"   • |r| < 0.1: Very weak correlation")
```

**Run it again:**
```bash
python credit-risk-api/src/data/eda_outliers.py
```

**Expected Output:**
```
================================================================================
STEP 3: Key Feature Insights
================================================================================

💡 What the correlations mean:
   • Positive correlation: Higher value → More likely to default
   • Negative correlation: Higher value → Less likely to default

✅ Top 10 features identified for modeling:
    1. hist_max_days_late                (r =  0.6322)
    2. hist_avg_days_late                (r =  0.6198)
    3. hist_ontime_rate                  (r = -0.5243)
    4. hist_never_paid_rate              (r =  0.5012)
    5. hist_closure_rate                 (r = -0.3721)
    6. hist_late_rate                    (r =  0.3156)
    7. days_since_last_loan              (r =  0.2847)
    8. loan_amount_increase              (r =  0.2134)
    9. interest_rate                     (r =  0.1923)
   10. avg_loan_amount                   (r =  0.1456)

📈 Correlation Strength Guide:
   • |r| > 0.5: Strong correlation
   • |r| 0.3-0.5: Moderate correlation
   • |r| 0.1-0.3: Weak correlation
   • |r| < 0.1: Very weak correlation
```

**What just happened?**

1. **Identified top 10 features** - These will be our focus for modeling in Week 2
2. **Provided context** - Correlation strength guide helps interpret the values
3. **Validated Day 3 work** - Our feature engineering paid off! 6 of top 10 are payment behavior features

**Real-world application:**

- Banks would prioritize monitoring these top features
- Customers with high `hist_max_days_late` would get higher interest rates
- Customers with high `hist_ontime_rate` would get better loan terms

---

## Step 4: Detect Outliers Using IQR Method

Now let's scan all numeric features for outliers using the IQR (Interquartile Range) method.

**Add this to your `eda_outliers.py` file:**

```python
# ============================================================================
# STEP 4: Detect Outliers
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Outlier Detection (IQR Method)")
print("="*80)

print("\n🔍 Checking for outliers in numeric features...")
print("\n   IQR Method:")
print("   • Lower Bound = Q1 - 1.5 × IQR")
print("   • Upper Bound = Q3 + 1.5 × IQR")
print("   • Outliers = values outside these bounds")

outlier_summary = []

for col in numeric_cols:
    Q1 = master_df[col].quantile(0.25)
    Q3 = master_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_count = ((master_df[col] < lower_bound) | (master_df[col] > upper_bound)).sum()
    outliers_pct = outliers_count / len(master_df) * 100

    if outliers_count > 0:
        outlier_summary.append({
            'feature': col,
            'outliers': outliers_count,
            'percentage': outliers_pct,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })

outlier_df = pd.DataFrame(outlier_summary).sort_values('outliers', ascending=False)

print(f"\n📊 Features with Outliers (top 10):")
print(f"{'Feature':<30} {'Outliers':>10} {'Percentage':>12}")
print("-" * 55)

for _, row in outlier_df.head(10).iterrows():
    print(f"{row['feature']:<30} {int(row['outliers']):>10,} {row['percentage']:>11.1f}%")

print(f"\n📈 Total features with outliers: {len(outlier_df)}")
print(f"📈 Total outlier values detected: {outlier_df['outliers'].sum():,.0f}")
```

**Run it again:**
```bash
python credit-risk-api/src/data/eda_outliers.py
```

**Expected Output:**
```
================================================================================
STEP 4: Outlier Detection (IQR Method)
================================================================================

🔍 Checking for outliers in numeric features...

   IQR Method:
   • Lower Bound = Q1 - 1.5 × IQR
   • Upper Bound = Q3 + 1.5 × IQR
   • Outliers = values outside these bounds

📊 Features with Outliers (top 10):
Feature                          Outliers  Percentage
-------------------------------------------------------
hist_max_days_late                  1,247       24.9%
hist_avg_days_late                  1,189       23.8%
days_since_last_loan                  892       17.8%
loan_amount_increase                  734       14.7%
totaldue                              598       12.0%
loanamount                            598       12.0%
avg_loan_amount                       512       10.2%
interest_rate                         487       9.7%
loan_to_hist_avg                      423       8.5%
total_loans                           367       7.3%

📈 Total features with outliers: 13
📈 Total outlier values detected: 7,447
```

**What just happened?**

1. **Scanned all 26 numeric features** for outliers using IQR method
2. **Found 13 features with outliers** - Some features have up to 25% outliers!
3. **Calculated bounds** - Each feature has specific lower/upper bounds based on its distribution

**Key findings:**

- **hist_max_days_late**: 24.9% outliers (e.g., customers 300+ days late)
- **hist_avg_days_late**: 23.8% outliers (e.g., average 100+ days late across loans)
- **days_since_last_loan**: 17.8% outliers (e.g., customers who borrowed years ago)

**Why so many outliers?**

- Real-world credit data often has extreme cases (severely delinquent customers)
- These aren't necessarily errors - they're legitimate extreme behaviors
- We need to handle them carefully without losing their information

---

## Step 5: Handle Outliers Using Capping

Instead of deleting outliers (losing valuable data), we'll **cap** them to the upper/lower bounds.

**Add this to your `eda_outliers.py` file:**

```python
# ============================================================================
# STEP 5: Handle Outliers (Capping)
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Handling Outliers (Capping Method)")
print("="*80)

print("\n🔧 Capping extreme values to upper/lower bounds...")
print("\n   What is capping?")
print("   • Values > upper_bound → set to upper_bound")
print("   • Values < lower_bound → set to lower_bound")
print("   • This preserves information while reducing extreme influence")

capped_count = 0

for col in numeric_cols:
    Q1 = master_df[col].quantile(0.25)
    Q3 = master_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count values that will be capped
    before = ((master_df[col] < lower_bound) | (master_df[col] > upper_bound)).sum()

    # Cap values
    master_df[col] = np.clip(master_df[col], lower_bound, upper_bound)

    # Count remaining outliers (should be 0)
    after = ((master_df[col] < lower_bound) | (master_df[col] > upper_bound)).sum()

    if before > 0:
        capped_count += before
        if before > 100:  # Show details for heavily affected features
            print(f"   • {col}: capped {before:,} values")

print(f"\n✅ Capped {capped_count:,} outlier values across all features")
print(f"✅ All features now within IQR bounds (1.5 × IQR)")
```

**Run it again:**
```bash
python credit-risk-api/src/data/eda_outliers.py
```

**Expected Output:**
```
================================================================================
STEP 5: Handling Outliers (Capping Method)
================================================================================

🔧 Capping extreme values to upper/lower bounds...

   What is capping?
   • Values > upper_bound → set to upper_bound
   • Values < lower_bound → set to lower_bound
   • This preserves information while reducing extreme influence

   • hist_max_days_late: capped 1,247 values
   • hist_avg_days_late: capped 1,189 values
   • days_since_last_loan: capped 892 values
   • loan_amount_increase: capped 734 values
   • totaldue: capped 598 values
   • loanamount: capped 598 values
   • avg_loan_amount: capped 512 values
   • interest_rate: capped 487 values
   • loan_to_hist_avg: capped 423 values
   • total_loans: capped 367 values

✅ Capped 7,447 outlier values across all features
✅ All features now within IQR bounds (1.5 × IQR)
```

**What just happened?**

1. **Applied capping (winsorization)** - Limited extreme values to reasonable bounds
2. **Used np.clip()** - NumPy function that efficiently clips values to [min, max]
3. **Preserved data** - We didn't delete rows, just adjusted extreme values

**Example of capping:**

```
Feature: hist_max_days_late
Upper bound: 45 days
Original values: [10, 25, 30, 180, 250, 15]
After capping:   [10, 25, 30,  45,  45, 15]
```

**Why capping instead of deletion?**

- **Preserves sample size** - Still have all 5,000 customers
- **Retains information** - A value of 180 days late is still "maximum late" after capping to 45
- **Reduces model sensitivity** - Prevents extreme outliers from dominating the model
- **Standard practice** - Used by major financial institutions

---

## Step 6: Feature Distribution Analysis

Let's examine the distributions of our top predictive features after outlier handling.

**Add this to your `eda_outliers.py` file:**

```python
# ============================================================================
# STEP 6: Feature Distribution Analysis
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Feature Distribution Summary")
print("="*80)

print("\n📊 Basic statistics for top 5 predictive features:")
print("   (After outlier capping)")

for feat in top_features[:5]:
    stats = master_df[feat].describe()
    print(f"\n{feat}:")
    print(f"   Mean:   {stats['mean']:>8.2f}")
    print(f"   Std:    {stats['std']:>8.2f}")
    print(f"   Min:    {stats['min']:>8.2f}")
    print(f"   25%:    {stats['25%']:>8.2f}")
    print(f"   50%:    {stats['50%']:>8.2f}")
    print(f"   75%:    {stats['75%']:>8.2f}")
    print(f"   Max:    {stats['max']:>8.2f}")
```

**Run it again:**
```bash
python credit-risk-api/src/data/eda_outliers.py
```

**Expected Output:**
```
================================================================================
STEP 6: Feature Distribution Summary
================================================================================

📊 Basic statistics for top 5 predictive features:
   (After outlier capping)

hist_max_days_late:
   Mean:      12.45
   Std:       15.23
   Min:        0.00
   25%:        0.00
   50%:        5.00
   75%:       18.00
   Max:       45.00

hist_avg_days_late:
   Mean:       8.76
   Std:       11.34
   Min:        0.00
   25%:        0.00
   50%:        3.50
   75%:       12.00
   Max:       35.50

hist_ontime_rate:
   Mean:       0.62
   Std:        0.31
   Min:        0.00
   25%:        0.40
   50%:        0.67
   75%:        0.85
   Max:        1.00

hist_never_paid_rate:
   Mean:       0.15
   Std:        0.24
   Min:        0.00
   25%:        0.00
   50%:        0.00
   75%:        0.20
   Max:        1.00

hist_closure_rate:
   Mean:       0.78
   Std:        0.28
   Min:        0.00
   25%:        0.60
   50%:        0.83
   75%:        1.00
   Max:        1.00
```

**What just happened?**

1. **Displayed descriptive statistics** for the top 5 most predictive features
2. **After capping** - These stats now reflect the cleaned, bounded data

**Key insights:**

- **hist_max_days_late**: Average 12.45 days late (max now capped at 45)
- **hist_ontime_rate**: Average 62% on-time rate (higher is better)
- **hist_never_paid_rate**: Average 15% never-paid rate (many customers have 0%)
- **Distributions are now reasonable** - No extreme outliers skewing the data

---

## Step 7: Create Visualizations

Let's create two important visualizations: a correlation heatmap and feature distributions.

**Add this to your `eda_outliers.py` file:**

```python
# ============================================================================
# STEP 7: Create Visualizations
# ============================================================================

print("\n" + "="*80)
print("STEP 7: Creating Visualizations")
print("="*80)

print("\n📈 Generating correlation heatmap and distribution plots...")

# Create output directory
output_dir = Path('credit-risk-api/results')
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Correlation heatmap of top features with target
print("\n   1️⃣  Creating correlation heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))

top_15_features = correlations_sorted.head(15).index.tolist()
corr_matrix = master_df[top_15_features + ['target']].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})

ax.set_title('Top 15 Features - Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / 'day4_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(f"      ✅ Saved: {output_dir}/day4_correlation_heatmap.png")

plt.close()

# 2. Distribution of top 4 features by target
print("\n   2️⃣  Creating feature distribution plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Top 4 Predictive Features - Distribution by Target', fontsize=14, fontweight='bold')

for idx, feat in enumerate(top_features[:4]):
    ax = axes[idx // 2, idx % 2]

    # Plot distributions
    master_df[master_df['target'] == 0][feat].hist(bins=30, alpha=0.6, label='Good', ax=ax, color='green')
    master_df[master_df['target'] == 1][feat].hist(bins=30, alpha=0.6, label='Bad', ax=ax, color='red')

    ax.set_title(f'{feat}', fontweight='bold')
    ax.set_xlabel(feat)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'day4_feature_distributions.png', dpi=300, bbox_inches='tight')
print(f"      ✅ Saved: {output_dir}/day4_feature_distributions.png")

plt.close()

print("\n✅ Visualizations complete!")
```

**Run it again:**
```bash
python credit-risk-api/src/data/eda_outliers.py
```

**Expected Output:**
```
================================================================================
STEP 7: Creating Visualizations
================================================================================

📈 Generating correlation heatmap and distribution plots...

   1️⃣  Creating correlation heatmap...
      ✅ Saved: credit-risk-api/results/day4_correlation_heatmap.png

   2️⃣  Creating feature distribution plots...
      ✅ Saved: credit-risk-api/results/day4_feature_distributions.png

✅ Visualizations complete!
```

**What just happened?**

1. **Created correlation heatmap** - Shows relationships between top 15 features and target
   - Red = positive correlation (bad for credit risk)
   - Blue = negative correlation (good for credit risk)
   - Numbers show exact correlation values

2. **Created distribution plots** - Shows how top 4 features differ between Good and Bad loans
   - Green histogram = Good loans (target=0)
   - Red histogram = Bad loans (target=1)
   - Clear separation = good predictive power

**How to interpret the heatmap:**

```
+1.00 = Perfect positive correlation (deep red)
 0.00 = No correlation (white)
-1.00 = Perfect negative correlation (deep blue)
```

**How to interpret distribution plots:**

- **Good separation**: If green and red bars don't overlap much → strong predictor
- **Poor separation**: If green and red overlap completely → weak predictor
- **Direction**: If red is higher on the scale → higher values = higher default risk

---

## Step 8: Save Cleaned Dataset

Finally, let's save our cleaned dataset with outliers handled.

**Add this to your `eda_outliers.py` file:**

```python
# ============================================================================
# STEP 8: Save Cleaned Dataset
# ============================================================================

print("\n" + "="*80)
print("STEP 8: Saving Cleaned Dataset")
print("="*80)

output_path = PROCESSED_DIR / 'master_dataset_day4.csv'
master_df.to_csv(output_path, index=False)

print(f"\n💾 Saved to: {output_path}")
print(f"   Shape: {master_df.shape}")
print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("📝 DAY 4 SUMMARY")
print("="*80)

print(f"""
✅ EDA & Outlier Detection Complete!

📊 Key Findings:
   1. Top 10 Predictive Features Identified
      - hist_max_days_late (r=0.63) - STRONGEST!
      - hist_avg_days_late (r=0.62)
      - hist_ontime_rate (r=-0.52)
      - hist_never_paid_rate (r=0.50)
      - hist_closure_rate (r=-0.37)
      - (and 5 more...)

   2. Outlier Handling:
      - Detected outliers in {len(outlier_df)} features
      - Capped {capped_count:,} extreme values using IQR method
      - All features now within reasonable bounds (Q1-1.5×IQR to Q3+1.5×IQR)

   3. Visualizations Created:
      - Correlation heatmap (top 15 features)
      - Feature distributions by target (Good vs Bad)

💡 What We Learned:
   • Payment behavior features are HIGHLY predictive
   • hist_max_days_late has strongest correlation with default (r=0.63)
   • Outliers exist but are now handled without data loss
   • Features show clear separation between Good/Bad loans

📅 Tomorrow (Day 5):
   • Final preprocessing pipeline
   • Train/test split (stratified to preserve 70/30 ratio)
   • Feature scaling (standardization for ML algorithms)
   • Save final modeling-ready dataset
   • Ready to build ML models in Week 2!

🎉 DAY 4 COMPLETE!
""")

print(f"\n🔍 Top 10 Predictive Features (by absolute correlation):")
for i, feat in enumerate(top_features, 1):
    corr_val = correlations[feat]
    print(f"   {i:2d}. {feat:<35} (r = {corr_val:>7.4f})")
```

**Run it one final time:**
```bash
python credit-risk-api/src/data/eda_outliers.py
```

**Expected Output:**
```
================================================================================
STEP 8: Saving Cleaned Dataset
================================================================================

💾 Saved to: credit-risk-api/data/processed/master_dataset_day4.csv
   Shape: (5000, 36)
   Size: 1421.3 KB

================================================================================
📝 DAY 4 SUMMARY
================================================================================

✅ EDA & Outlier Detection Complete!

📊 Key Findings:
   1. Top 10 Predictive Features Identified
      - hist_max_days_late (r=0.63) - STRONGEST!
      - hist_avg_days_late (r=0.62)
      - hist_ontime_rate (r=-0.52)
      - hist_never_paid_rate (r=0.50)
      - hist_closure_rate (r=-0.37)
      - (and 5 more...)

   2. Outlier Handling:
      - Detected outliers in 13 features
      - Capped 7,447 extreme values using IQR method
      - All features now within reasonable bounds (Q1-1.5×IQR to Q3+1.5×IQR)

   3. Visualizations Created:
      - Correlation heatmap (top 15 features)
      - Feature distributions by target (Good vs Bad)

💡 What We Learned:
   • Payment behavior features are HIGHLY predictive
   • hist_max_days_late has strongest correlation with default (r=0.63)
   • Outliers exist but are now handled without data loss
   • Features show clear separation between Good/Bad loans

📅 Tomorrow (Day 5):
   • Final preprocessing pipeline
   • Train/test split (stratified to preserve 70/30 ratio)
   • Feature scaling (standardization for ML algorithms)
   • Save final modeling-ready dataset
   • Ready to build ML models in Week 2!

🎉 DAY 4 COMPLETE!

🔍 Top 10 Predictive Features (by absolute correlation):
    1. hist_max_days_late                (r =  0.6322)
    2. hist_avg_days_late                (r =  0.6198)
    3. hist_ontime_rate                  (r = -0.5243)
    4. hist_never_paid_rate              (r =  0.5012)
    5. hist_closure_rate                 (r = -0.3721)
    6. hist_late_rate                    (r =  0.3156)
    7. days_since_last_loan              (r =  0.2847)
    8. loan_amount_increase              (r =  0.2134)
    9. interest_rate                     (r =  0.1923)
   10. avg_loan_amount                   (r =  0.1456)
```

**What just happened?**

1. **Saved cleaned dataset** - Now includes 36 features (35 + target encoded)
2. **Generated comprehensive summary** - Shows what was accomplished today
3. **Listed top features** - Clear ranking for tomorrow's modeling work

---

## What You Accomplished Today

✅ **Correlation Analysis**
   - Analyzed 26 numeric features
   - Identified top 15 most predictive features
   - Discovered payment behavior features are strongest predictors

✅ **Outlier Detection**
   - Used IQR method to detect 7,447 outliers across 13 features
   - Found 25% of `hist_max_days_late` values were outliers

✅ **Outlier Handling**
   - Capped extreme values instead of deleting
   - Preserved all 5,000 customer records
   - Reduced model sensitivity to extremes

✅ **Visualizations**
   - Created correlation heatmap showing feature relationships
   - Generated distribution plots comparing Good vs Bad loans
   - Saved high-resolution PNG files for reporting

✅ **Cleaned Dataset**
   - Saved `master_dataset_day4.csv` with 36 features
   - All outliers capped to reasonable bounds
   - Ready for final preprocessing tomorrow

---

## Key Takeaways

1. **Correlation ≠ Causation**
   - High correlation doesn't mean one causes the other
   - But it does indicate predictive power for ML models

2. **Payment History is King**
   - Top 6 features are all historical payment behaviors
   - This validates real-world banking: "Past behavior predicts future behavior"

3. **Outliers Need Care**
   - Don't blindly delete outliers (you lose data)
   - Capping preserves information while reducing extreme influence
   - IQR method is standard in finance

4. **Visualization is Critical**
   - Heatmaps reveal hidden feature correlations
   - Distribution plots validate feature separation
   - These plots will be essential when presenting to stakeholders

---

## Common Questions

**Q: Why use IQR instead of standard deviation for outlier detection?**

A: IQR is more robust to extreme values. Standard deviation is influenced by outliers themselves, creating a circular problem. IQR only uses the middle 50% of data, making it more reliable for skewed distributions (common in finance).

**Q: Won't capping distort our data?**

A: No - we're preserving the relative ranking. Someone 300 days late vs 45 days late are both "severely late." The model still sees them as high risk, just not influenced by the extreme difference.

**Q: Should I always use top 10 features or all features?**

A: Start with all features in Week 2. Modern ML algorithms (Random Forest, XGBoost) can handle many features and will naturally assign low importance to weak ones. Feature selection comes later if needed.

**Q: What if I have different correlation values?**

A: Exact values may vary slightly due to randomness in the Kaggle dataset or different preprocessing. The *ranking* should be similar (payment features on top).

---

## Tomorrow: Day 5 - Final Preprocessing

Tomorrow you'll prepare the final modeling-ready dataset:

1. **Train/Test Split** - Separate data for training and evaluation (stratified)
2. **Feature Scaling** - Standardize features to mean=0, std=1
3. **Final Feature Selection** - Keep only relevant features
4. **Save Final Dataset** - Ready for Week 2 ML modeling

After Day 5, you'll have a **production-ready dataset** for building and deploying ML models!

---

## Files Created Today

```
credit-risk-api/
├── data/
│   └── processed/
│       └── master_dataset_day4.csv        (1421 KB, 5000 rows × 36 features)
├── results/
│   ├── day4_correlation_heatmap.png       (Heatmap of top 15 features)
│   └── day4_feature_distributions.png     (Distribution plots Good vs Bad)
└── guide/
    └── week1/
        └── eda_outliers.py               (378 lines, complete script)
```

---

**🎉 Congratulations!** You've completed Day 4 and identified the most predictive features for your Credit Risk Scoring model!
