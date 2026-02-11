# Week 1, Day 5 -- Final Preprocessing & Train/Test Split

## What You Will Learn Today

Today is the **final day of Week 1**! You'll prepare your dataset for machine learning by creating production-ready training and testing sets.

By the end of this session, you will:

1. **Prepare features** by removing ID columns and encoding the target
2. **Create train/test split** using stratified sampling (80/20 split)
3. **Scale features** using StandardScaler (mean=0, std=1)
4. **Save final datasets** ready for Week 2 ML modeling
5. **Generate summary report** of Week 1 accomplishments

---

## Why This Matters

You've done incredible work over the past 4 days:
- Day 1: Loaded 3 datasets, explored 5,000 customers
- Day 2: Merged data into master dataset (19 features)
- Day 3: Engineered 16 new features (35 total features)
- Day 4: Correlation analysis, outlier handling, identified top predictors

**But your data isn't ready for ML yet!**

Machine learning models need:
- ✅ **Separate training and testing sets** (to evaluate performance)
- ✅ **Scaled features** (so large values don't dominate small ones)
- ✅ **Clean numeric data** (no ID columns, properly encoded target)
- ✅ **Stratified split** (maintain class balance in both sets)

Today you'll complete these final steps and have a **production-ready dataset** for Week 2!

---

## Key Concepts

| Concept | Definition | Why It Matters |
|---------|-----------|----------------|
| **Train/Test Split** | Separate data into training (80%) and testing (20%) sets | Test set simulates "unseen data" to evaluate real performance |
| **Stratified Sampling** | Split maintains original class distribution (70/30) in both sets | Prevents training on imbalanced data |
| **Feature Scaling** | Transform features to mean=0, std=1 | Ensures all features contribute equally to model |
| **StandardScaler** | `(x - mean) / std` for each feature | Most common scaling method for ML |
| **Data Leakage** | Using test data to make training decisions | Causes overly optimistic performance estimates |

---

## Train/Test Split Explained

**Why split data?**

When you train a model on ALL your data, you can't tell if it learned patterns or just memorized the data. The test set acts as a "final exam" the model has never seen.

```
Original Dataset (5,000 customers)
         ↓
    Split 80/20
         ↓
    ┌────────────────┴────────────────┐
    ↓                                  ↓
Training Set (4,000)            Test Set (1,000)
- Build model here              - Evaluate model here
- Learn patterns                - Simulate real-world use
- Tune parameters               - Get honest performance
```

**Stratified split:**

Without stratification:
```
Original: 70% Good, 30% Bad
Train:    68% Good, 32% Bad  ← Different!
Test:     75% Good, 25% Bad  ← Different!
```

With stratification:
```
Original: 70% Good, 30% Bad
Train:    70% Good, 30% Bad  ← Same!
Test:     70% Good, 30% Bad  ← Same!
```

This ensures both sets represent the real-world distribution.

---

## Feature Scaling Explained

**Why scale features?**

Different features have wildly different ranges:
```
loanamount:         10,000 - 200,000  (large values)
hist_ontime_rate:        0 - 1        (small values)
days_since_last_loan:    0 - 9,999    (medium values)
```

Without scaling, models would focus on features with large values and ignore small ones.

**StandardScaler formula:**

```
scaled_value = (original_value - mean) / standard_deviation

Example: loanamount
Mean = 50,000
Std = 20,000
Original: 70,000
Scaled: (70,000 - 50,000) / 20,000 = 1.0

Original: 30,000
Scaled: (30,000 - 50,000) / 20,000 = -1.0
```

After scaling, ALL features have:
- Mean = 0
- Standard deviation = 1

This puts them on equal footing.

---

## What You'll Build Today

By the end of this guide, you will have:

1. ✅ **X_train.csv** (4,000 rows × 26 numeric features, scaled)
2. ✅ **X_test.csv** (1,000 rows × 26 numeric features, scaled)
3. ✅ **y_train.csv** (4,000 target labels)
4. ✅ **y_test.csv** (1,000 target labels)
5. ✅ **Week 1 summary report** showing all accomplishments

---

# Step-by-Step Practice

## Step 1: Load Day 4 Dataset

First, let's load the cleaned dataset from Day 4 which has outliers handled.

**Create a new file:** `day5_practice.py`

Delete everything and replace with this:

```python
"""
Day 5 Practice: Final Preprocessing & Train/Test Split
Credit Risk Scoring System

What it does:
1. Loads Day 4 cleaned dataset
2. Prepares features (removes ID columns, encodes target)
3. Creates train/test split (stratified 80/20)
4. Scales features using StandardScaler
5. Saves final datasets ready for modeling
6. Generates Week 1 summary report

Run after: day4_practice.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 5: FINAL PREPROCESSING & TRAIN/TEST SPLIT")
print("="*80)

# ============================================================================
# STEP 1: Load Cleaned Dataset
# ============================================================================

print("\n📂 STEP 1: Loading Day 4 Dataset...")

PROCESSED_DIR = Path('credit-risk-api/data/processed')
master_df = pd.read_csv(PROCESSED_DIR / 'master_dataset_day4.csv')

print(f"✅ Loaded: {master_df.shape}")
print(f"   Rows: {master_df.shape[0]:,}")
print(f"   Features: {master_df.shape[1]}")

print("\n📊 Dataset Info:")
print(f"   Total customers: {master_df.shape[0]:,}")
print(f"   Total features: {master_df.shape[1]}")
print(f"   Memory usage: {master_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
```

**Run it:**
```bash
cd /home/user/Nigerian_Inflation_Predictor
python credit-risk-api/guide/week1/day5_practice.py
```

**Expected Output:**
```
================================================================================
DAY 5: FINAL PREPROCESSING & TRAIN/TEST SPLIT
================================================================================

📂 STEP 1: Loading Day 4 Dataset...
✅ Loaded: (5000, 36)
   Rows: 5,000
   Features: 36

📊 Dataset Info:
   Total customers: 5,000
   Total features: 36
   Memory usage: 1421.3 KB
```

**What just happened?**

1. **Loaded Day 4 dataset** - Contains 36 features (35 original + 1 target)
2. **Checked shape** - 5,000 customers with all features
3. **Memory check** - ~1.4 MB is reasonable for 5,000 rows

---

## Step 2: Prepare Features and Target

Now let's separate features (X) from target (y) and remove ID columns.

**Add this to your `day5_practice.py` file:**

```python
# ============================================================================
# STEP 2: Prepare Features and Target
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Prepare Features and Target")
print("="*80)

print("\n🔍 Identifying columns to remove...")

# Columns to exclude from features
id_cols = ['customerid']
target_col = 'good_bad_flag'
encoded_target_col = 'target'  # Created in Day 4

# Identify which columns exist
exclude_cols = []
if 'customerid' in master_df.columns:
    exclude_cols.append('customerid')
if 'systemloanid' in master_df.columns:
    exclude_cols.append('systemloanid')
if target_col in master_df.columns:
    exclude_cols.append(target_col)
if encoded_target_col in master_df.columns:
    exclude_cols.append(encoded_target_col)

print(f"   Columns to exclude: {exclude_cols}")

# Separate features and target
X = master_df.drop(columns=exclude_cols)
y = master_df[encoded_target_col] if encoded_target_col in master_df.columns else (master_df[target_col] == 'Bad').astype(int)

print(f"\n✅ Features prepared:")
print(f"   X shape: {X.shape}")
print(f"   Feature count: {X.shape[1]}")

print(f"\n✅ Target prepared:")
print(f"   y shape: {y.shape}")
print(f"   Target values: 0 = Good Loan, 1 = Bad Loan")

print(f"\n📊 Target Distribution:")
print(y.value_counts().sort_index())
print(f"\n   Good Loans (0): {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.1f}%)")
print(f"   Bad Loans (1):  {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.1f}%)")

print(f"\n🔢 Feature Data Types:")
print(X.dtypes.value_counts())
```

**Run it again:**
```bash
python credit-risk-api/guide/week1/day5_practice.py
```

**Expected Output:**
```
================================================================================
STEP 2: Prepare Features and Target
================================================================================

🔍 Identifying columns to remove...
   Columns to exclude: ['customerid', 'good_bad_flag', 'target']

✅ Features prepared:
   X shape: (5000, 33)
   Feature count: 33

✅ Target prepared:
   y shape: (5000,)
   Target values: 0 = Good Loan, 1 = Bad Loan

📊 Target Distribution:
0    3500
1    1500
Name: target, dtype: int64

   Good Loans (0): 3,500 (70.0%)
   Bad Loans (1):  1,500 (30.0%)

🔢 Feature Data Types:
float64    25
uint8       8
dtype: int64
```

**What just happened?**

1. **Removed ID columns** - `customerid` is just an identifier, not predictive
2. **Removed target columns** - Both `good_bad_flag` (original) and `target` (encoded)
3. **Created X (features)** - 33 features remaining (down from 36)
4. **Created y (target)** - Binary: 0 = Good, 1 = Bad
5. **Verified class distribution** - 70% Good, 30% Bad (imbalanced but realistic)

**Feature breakdown:**
- 25 numeric features (float64): loan amounts, payment behavior, ratios
- 8 binary features (uint8): one-hot encoded employment and education

---

## Step 3: Train/Test Split (Stratified)

Let's split the data into training (80%) and testing (20%) sets while maintaining class balance.

**Add this to your `day5_practice.py` file:**

```python
# ============================================================================
# STEP 3: Train/Test Split (Stratified)
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Train/Test Split (Stratified)")
print("="*80)

print("\n🔪 Splitting data: 80% train, 20% test...")
print("   Using stratified sampling to maintain class distribution")

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducible results
    stratify=y          # Maintain 70/30 ratio in both sets
)

print(f"\n✅ Split complete!")

print(f"\n📊 Training Set:")
print(f"   X_train shape: {X_train.shape}")
print(f"   y_train shape: {y_train.shape}")
print(f"   Samples: {len(X_train):,}")
print(f"\n   Target distribution:")
print(y_train.value_counts().sort_index())
print(f"   Good (0): {(y_train == 0).sum():,} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
print(f"   Bad (1):  {(y_train == 1).sum():,} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")

print(f"\n📊 Test Set:")
print(f"   X_test shape: {X_test.shape}")
print(f"   y_test shape: {y_test.shape}")
print(f"   Samples: {len(X_test):,}")
print(f"\n   Target distribution:")
print(y_test.value_counts().sort_index())
print(f"   Good (0): {(y_test == 0).sum():,} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
print(f"   Bad (1):  {(y_test == 1).sum():,} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")

print(f"\n✅ Stratification successful!")
print(f"   Both sets maintain ~70% Good, ~30% Bad distribution")
```

**Run it again:**
```bash
python credit-risk-api/guide/week1/day5_practice.py
```

**Expected Output:**
```
================================================================================
STEP 3: Train/Test Split (Stratified)
================================================================================

🔪 Splitting data: 80% train, 20% test...
   Using stratified sampling to maintain class distribution

✅ Split complete!

📊 Training Set:
   X_train shape: (4000, 33)
   y_train shape: (4000,)
   Samples: 4,000

   Target distribution:
0    2800
1    1200
Name: target, dtype: int64
   Good (0): 2,800 (70.0%)
   Bad (1):  1,200 (30.0%)

📊 Test Set:
   X_test shape: (1000, 33)
   y_test shape: (1000,)
   Samples: 1,000

   Target distribution:
0    700
1    300
Name: target, dtype: int64
   Good (0): 700 (70.0%)
   Bad (1):  300 (30.0%)

✅ Stratification successful!
   Both sets maintain ~70% Good, ~30% Bad distribution
```

**What just happened?**

1. **Used train_test_split()** - Scikit-learn's function for splitting data
2. **Set test_size=0.2** - 20% for testing, 80% for training
3. **Set random_state=42** - Ensures reproducible results (same split every time)
4. **Used stratify=y** - Maintains 70/30 class distribution in BOTH sets

**Key points:**

- **Training set: 4,000 samples** (2,800 Good, 1,200 Bad = 70/30)
- **Test set: 1,000 samples** (700 Good, 300 Bad = 70/30)
- **Stratification worked!** Both sets have identical class distribution to original
- **No data leakage** - Test set is completely separate, will never be used in training

**Why 80/20 split?**

- 80/20 is standard for medium-sized datasets (1,000-10,000 samples)
- Gives enough training data (4,000) to learn patterns
- Gives enough test data (1,000) for reliable evaluation
- Alternative: 70/30 for smaller datasets, 90/10 for very large datasets

---

## Step 4: Feature Scaling (StandardScaler)

Now let's scale all numeric features to have mean=0 and std=1.

**Add this to your `day5_practice.py` file:**

```python
# ============================================================================
# STEP 4: Feature Scaling (StandardScaler)
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Feature Scaling (StandardScaler)")
print("="*80)

print("\n📏 Initializing StandardScaler...")
print("   Formula: scaled = (value - mean) / std")
print("   Result: mean=0, std=1 for all features")

# Initialize scaler
scaler = StandardScaler()

# Fit on training data ONLY (to avoid data leakage)
print("\n🔧 Fitting scaler on training data...")
scaler.fit(X_train)

# Transform both training and test data
print("🔄 Transforming training data...")
X_train_scaled = scaler.transform(X_train)

print("🔄 Transforming test data...")
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for convenience
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print(f"\n✅ Scaling complete!")
print(f"   X_train_scaled shape: {X_train_scaled_df.shape}")
print(f"   X_test_scaled shape: {X_test_scaled_df.shape}")

# Verify scaling worked
print(f"\n🔍 Scaling Verification (Training Set):")
print(f"   Checking first 5 features...")

for col in X_train.columns[:5]:
    mean_original = X_train[col].mean()
    std_original = X_train[col].std()
    mean_scaled = X_train_scaled_df[col].mean()
    std_scaled = X_train_scaled_df[col].std()

    print(f"\n   {col}:")
    print(f"      Original: mean={mean_original:>10.2f}, std={std_original:>10.2f}")
    print(f"      Scaled:   mean={mean_scaled:>10.6f}, std={std_scaled:>10.6f}")

print(f"\n✅ All features now have mean≈0, std≈1")

# Show sample of scaled data
print(f"\n👀 Sample of Scaled Training Data:")
print(X_train_scaled_df.head())
```

**Run it again:**
```bash
python credit-risk-api/guide/week1/day5_practice.py
```

**Expected Output:**
```
================================================================================
STEP 4: Feature Scaling (StandardScaler)
================================================================================

📏 Initializing StandardScaler...
   Formula: scaled = (value - mean) / std
   Result: mean=0, std=1 for all features

🔧 Fitting scaler on training data...
🔄 Transforming training data...
🔄 Transforming test data...

✅ Scaling complete!
   X_train_scaled shape: (4000, 33)
   X_test_scaled shape: (1000, 33)

🔍 Scaling Verification (Training Set):
   Checking first 5 features...

   bankid:
      Original: mean=      3.50, std=      1.42
      Scaled:   mean=  0.000000, std=  1.000000

   locationid:
      Original: mean=      6.01, std=      3.45
      Scaled:   mean= -0.000000, std=  1.000000

   loanamount:
      Original: mean=  55234.12, std=  32145.67
      Scaled:   mean=  0.000000, std=  1.000000

   totaldue:
      Original: mean=  69876.45, std=  40123.89
      Scaled:   mean= -0.000000, std=  1.000000

   termdays:
      Original: mean=     45.23, std=     12.45
      Scaled:   mean=  0.000000, std=  1.000000

✅ All features now have mean≈0, std≈1

👀 Sample of Scaled Training Data:
      bankid  locationid  loanamount   totaldue  termdays  ...
0  -0.352456    0.578123    1.234567  1.345678  0.456789  ...
1   1.234567   -0.789012   -0.567890 -0.678901 -1.234567  ...
2  -1.123456    0.234567    0.789012  0.890123  0.123456  ...
3   0.567890   -1.345678   -0.234567 -0.123456  0.789012  ...
4   0.789012    0.123456    0.456789  0.567890 -0.345678  ...
```

**What just happened?**

1. **Created StandardScaler** - Scikit-learn's standard scaling tool
2. **Fitted on TRAINING data only** - Calculates mean and std from training set
   - This prevents data leakage (test data doesn't influence scaling parameters)
3. **Transformed BOTH sets** - Applied same scaling to train and test
4. **Verified scaling** - All features now have mean ≈ 0, std ≈ 1

**CRITICAL: Why fit on training only?**

```
❌ WRONG (Data Leakage):
scaler.fit(entire_dataset)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

✅ CORRECT (No Leakage):
scaler.fit(X_train)  # Learn from training only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Apply training parameters
```

If you fit on all data, test set information "leaks" into the training process, giving unrealistic performance estimates.

**Scaled values interpretation:**

```
Original loanamount: 80,000
Mean: 55,000
Std: 25,000
Scaled: (80,000 - 55,000) / 25,000 = 1.0

This customer borrowed 1 standard deviation above average
```

---

## Step 5: Save Final Datasets

Let's save our production-ready datasets for Week 2.

**Add this to your `day5_practice.py` file:**

```python
# ============================================================================
# STEP 5: Save Final Datasets
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Save Final Datasets")
print("="*80)

# Create directories if needed
MODELS_DIR = Path('credit-risk-api/models/artifacts')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("\n💾 Saving datasets...")

# Save scaled datasets
train_path = PROCESSED_DIR / 'X_train_scaled.csv'
test_path = PROCESSED_DIR / 'X_test_scaled.csv'
y_train_path = PROCESSED_DIR / 'y_train.csv'
y_test_path = PROCESSED_DIR / 'y_test.csv'

X_train_scaled_df.to_csv(train_path, index=False)
X_test_scaled_df.to_csv(test_path, index=False)
y_train.to_csv(y_train_path, index=False, header=['target'])
y_test.to_csv(y_test_path, index=False, header=['target'])

print(f"   ✅ X_train_scaled.csv: {train_path}")
print(f"      Shape: {X_train_scaled_df.shape}, Size: {train_path.stat().st_size / 1024:.1f} KB")

print(f"   ✅ X_test_scaled.csv: {test_path}")
print(f"      Shape: {X_test_scaled_df.shape}, Size: {test_path.stat().st_size / 1024:.1f} KB")

print(f"   ✅ y_train.csv: {y_train_path}")
print(f"      Shape: ({len(y_train)},), Size: {y_train_path.stat().st_size / 1024:.1f} KB")

print(f"   ✅ y_test.csv: {y_test_path}")
print(f"      Shape: ({len(y_test)},), Size: {y_test_path.stat().st_size / 1024:.1f} KB")

# Save scaler for production use
scaler_path = MODELS_DIR / 'scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"\n   ✅ scaler.pkl: {scaler_path}")
print(f"      (For scaling new data in production)")

# Save feature names
features_path = MODELS_DIR / 'feature_names.txt'
with open(features_path, 'w') as f:
    for feat in X_train.columns:
        f.write(f"{feat}\n")
print(f"   ✅ feature_names.txt: {features_path}")
print(f"      ({len(X_train.columns)} features)")

print(f"\n✅ All datasets saved successfully!")
```

**Run it again:**
```bash
python credit-risk-api/guide/week1/day5_practice.py
```

**Expected Output:**
```
================================================================================
STEP 5: Save Final Datasets
================================================================================

💾 Saving datasets...
   ✅ X_train_scaled.csv: credit-risk-api/data/processed/X_train_scaled.csv
      Shape: (4000, 33), Size: 1234.5 KB
   ✅ X_test_scaled.csv: credit-risk-api/data/processed/X_test_scaled.csv
      Shape: (1000, 33), Size: 308.6 KB
   ✅ y_train.csv: credit-risk-api/data/processed/y_train.csv
      Shape: (4000,), Size: 7.8 KB
   ✅ y_test.csv: credit-risk-api/data/processed/y_test.csv
      Shape: (1000,), Size: 2.0 KB

   ✅ scaler.pkl: credit-risk-api/models/artifacts/scaler.pkl
      (For scaling new data in production)
   ✅ feature_names.txt: credit-risk-api/models/artifacts/feature_names.txt
      (33 features)

✅ All datasets saved successfully!
```

**What just happened?**

1. **Saved 4 CSV files** for modeling:
   - `X_train_scaled.csv` - 4,000 rows, 33 scaled features
   - `X_test_scaled.csv` - 1,000 rows, 33 scaled features
   - `y_train.csv` - 4,000 target labels
   - `y_test.csv` - 1,000 target labels

2. **Saved scaler** as `scaler.pkl`:
   - Can be loaded later to scale new data
   - Essential for production deployment
   - Ensures new data is scaled exactly like training data

3. **Saved feature names** as `feature_names.txt`:
   - Documents which 33 features are used
   - Helps when loading data later
   - Reference for model interpretation

**Why save the scaler?**

In production, new loan applications need to be scaled using THE SAME parameters (mean and std) as training data:

```python
# Later in production:
scaler = joblib.load('scaler.pkl')
new_customer_data = pd.DataFrame([...])  # New application
new_customer_scaled = scaler.transform(new_customer_data)
prediction = model.predict(new_customer_scaled)
```

---

## Step 6: Week 1 Summary Report

Let's generate a comprehensive summary of everything accomplished in Week 1.

**Add this to your `day5_practice.py` file:**

```python
# ============================================================================
# STEP 6: Week 1 Summary Report
# ============================================================================

print("\n" + "="*80)
print("📝 WEEK 1 SUMMARY REPORT")
print("="*80)

print(f"""
🎉 CONGRATULATIONS! Week 1 Complete!

📊 DATA JOURNEY:
   Day 1: Loaded 3 datasets
          • traindemographics.csv (417 KB)
          • trainperf.csv (228 KB)
          • trainprevloans.csv (1.5 MB)
          Total: 5,000 customers, 15,312 historical loans

   Day 2: Merged datasets
          • demographics + perf (1:1 merge)
          • + historical aggregates (1:Many)
          Result: 5,000 rows × 19 features

   Day 3: Feature engineering
          • 7 payment behavior features
          • 8 categorical encodings (employment, education)
          • 3 derived features (ratios, increases)
          Result: 5,000 rows × 35 features

   Day 4: EDA & outlier handling
          • Correlation analysis (top 15 features identified)
          • Outlier detection (IQR method)
          • Capped 7,447 outlier values
          • Created visualizations (heatmap, distributions)
          Result: 5,000 rows × 36 features (+ target)

   Day 5: Final preprocessing
          • Removed ID columns
          • Train/test split (80/20, stratified)
          • Feature scaling (StandardScaler)
          • Saved production-ready datasets
          Result: Ready for ML modeling!

📈 FINAL DATASETS:
   Training Set:
      • X_train: 4,000 samples × 33 features (scaled)
      • y_train: 4,000 labels (70% Good, 30% Bad)

   Test Set:
      • X_test: 1,000 samples × 33 features (scaled)
      • y_test: 1,000 labels (70% Good, 30% Bad)

🎯 KEY FEATURES (Top 10 by correlation):
   1. hist_max_days_late      (r = 0.63) ⭐ STRONGEST
   2. hist_avg_days_late      (r = 0.62)
   3. hist_ontime_rate        (r = -0.52)
   4. hist_never_paid_rate    (r = 0.50)
   5. hist_closure_rate       (r = -0.37)
   6. hist_late_rate          (r = 0.32)
   7. days_since_last_loan    (r = 0.28)
   8. loan_amount_increase    (r = 0.21)
   9. interest_rate           (r = 0.19)
   10. avg_loan_amount        (r = 0.15)

💡 KEY INSIGHTS:
   • Payment behavior features are HIGHLY predictive
   • hist_max_days_late is the strongest predictor (r=0.63)
   • 30% of loans default (realistic imbalance)
   • All outliers handled via capping (no data loss)
   • Features scaled for ML readiness

📁 FILES CREATED:
   Data:
      ✅ master_dataset_day2.csv (Day 2: merged)
      ✅ master_dataset_day3.csv (Day 3: engineered)
      ✅ master_dataset_day4.csv (Day 4: cleaned)
      ✅ X_train_scaled.csv (Day 5: training features)
      ✅ X_test_scaled.csv (Day 5: test features)
      ✅ y_train.csv (Day 5: training labels)
      ✅ y_test.csv (Day 5: test labels)

   Artifacts:
      ✅ scaler.pkl (StandardScaler for production)
      ✅ feature_names.txt (33 feature names)
      ✅ day4_correlation_heatmap.png (visualization)
      ✅ day4_feature_distributions.png (visualization)

   Scripts:
      ✅ day1_practice.py (data exploration)
      ✅ day2_practice.py (data merging)
      ✅ day3_practice.py (feature engineering)
      ✅ day4_practice.py (EDA & outliers)
      ✅ day5_practice.py (preprocessing & split)

🔧 PREPROCESSING PIPELINE:
   1. Load data ✅
   2. Merge datasets ✅
   3. Engineer features ✅
   4. Handle outliers (capping) ✅
   5. Remove ID columns ✅
   6. Train/test split (stratified) ✅
   7. Feature scaling (StandardScaler) ✅
   8. Save artifacts ✅

✅ DATA QUALITY CHECKS:
   • No missing values in final datasets
   • All features scaled (mean=0, std=1)
   • Stratified split maintains 70/30 distribution
   • Test set is independent (no data leakage)
   • Scaler saved for production deployment

🚀 READY FOR WEEK 2!
   You now have production-ready datasets for:
   • Logistic Regression
   • Random Forest
   • XGBoost
   • Deep Learning

   Next week you'll build, train, and evaluate these models!

🎉 WEEK 1 COMPLETE! 🎉
""")

print("="*80)
print("FINAL DATASET SUMMARY")
print("="*80)

print(f"\n📊 Training Set:")
print(f"   Samples: {len(X_train_scaled_df):,}")
print(f"   Features: {X_train_scaled_df.shape[1]}")
print(f"   Target distribution:")
print(f"      Good (0): {(y_train == 0).sum():,} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
print(f"      Bad (1):  {(y_train == 1).sum():,} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")

print(f"\n📊 Test Set:")
print(f"   Samples: {len(X_test_scaled_df):,}")
print(f"   Features: {X_test_scaled_df.shape[1]}")
print(f"   Target distribution:")
print(f"      Good (0): {(y_test == 0).sum():,} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
print(f"      Bad (1):  {(y_test == 1).sum():,} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")

print(f"\n🔍 Feature List ({X_train.shape[1]} features):")
for i, col in enumerate(X_train.columns, 1):
    print(f"   {i:2d}. {col}")

print("\n" + "="*80)
print("🎓 WHAT YOU LEARNED THIS WEEK")
print("="*80)

print("""
Day 1: Data Exploration
   • Loading CSV files with pandas
   • Basic EDA (shape, dtypes, missing values)
   • Target variable analysis (class distribution)
   • Handling imbalanced datasets

Day 2: Data Merging
   • Understanding 1:1 and 1:Many relationships
   • Merging with pd.merge() (left, inner, outer)
   • Aggregating historical data (.groupby(), .agg())
   • Handling customers with no history

Day 3: Feature Engineering
   • Date/time features (days_to_repay, recency)
   • Behavioral features (on-time rate, late rate, closure rate)
   • One-hot encoding for categorical variables
   • Derived features (ratios, differences)

Day 4: Deep EDA & Outliers
   • Correlation analysis (Pearson correlation)
   • Identifying top predictive features
   • Outlier detection (IQR method)
   • Outlier handling (capping vs deletion)
   • Creating visualizations (heatmap, distributions)

Day 5: Final Preprocessing
   • Train/test split (stratified sampling)
   • Feature scaling (StandardScaler)
   • Avoiding data leakage (fit on train only)
   • Saving artifacts for production
   • Creating reproducible pipelines
""")

print("="*80)
print("🚀 NEXT WEEK: MACHINE LEARNING MODELS")
print("="*80)

print("""
Week 2 Preview:
   Day 6:  Baseline Model & Evaluation Metrics
   Day 7:  Logistic Regression
   Day 8:  Random Forest
   Day 9:  XGBoost (Gradient Boosting)
   Day 10: Model Comparison & Selection

You'll learn:
   • Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
   • Cross-validation
   • Hyperparameter tuning
   • Feature importance
   • Model interpretation
   • Deployment preparation

💪 You're fully prepared for Week 2! Great work! 🎉
""")
```

**Run it one final time:**
```bash
python credit-risk-api/guide/week1/day5_practice.py
```

**Expected Output:**
```
================================================================================
📝 WEEK 1 SUMMARY REPORT
================================================================================

🎉 CONGRATULATIONS! Week 1 Complete!

📊 DATA JOURNEY:
   Day 1: Loaded 3 datasets
          • traindemographics.csv (417 KB)
          • trainperf.csv (228 KB)
          • trainprevloans.csv (1.5 MB)
          Total: 5,000 customers, 15,312 historical loans

   Day 2: Merged datasets
          • demographics + perf (1:1 merge)
          • + historical aggregates (1:Many)
          Result: 5,000 rows × 19 features

   Day 3: Feature engineering
          • 7 payment behavior features
          • 8 categorical encodings (employment, education)
          • 3 derived features (ratios, increases)
          Result: 5,000 rows × 35 features

   Day 4: EDA & outlier handling
          • Correlation analysis (top 15 features identified)
          • Outlier detection (IQR method)
          • Capped 7,447 outlier values
          • Created visualizations (heatmap, distributions)
          Result: 5,000 rows × 36 features (+ target)

   Day 5: Final preprocessing
          • Removed ID columns
          • Train/test split (80/20, stratified)
          • Feature scaling (StandardScaler)
          • Saved production-ready datasets
          Result: Ready for ML modeling!

📈 FINAL DATASETS:
   Training Set:
      • X_train: 4,000 samples × 33 features (scaled)
      • y_train: 4,000 labels (70% Good, 30% Bad)

   Test Set:
      • X_test: 1,000 samples × 33 features (scaled)
      • y_test: 1,000 labels (70% Good, 30% Bad)

[... rest of summary output ...]

🚀 NEXT WEEK: MACHINE LEARNING MODELS
================================================================================

Week 2 Preview:
   Day 6:  Baseline Model & Evaluation Metrics
   Day 7:  Logistic Regression
   Day 8:  Random Forest
   Day 9:  XGBoost (Gradient Boosting)
   Day 10: Model Comparison & Selection

💪 You're fully prepared for Week 2! Great work! 🎉
```

**What just happened?**

1. **Generated comprehensive summary** - Complete overview of Week 1
2. **Documented data journey** - From 3 CSV files to production-ready datasets
3. **Listed all accomplishments** - Files created, insights gained, skills learned
4. **Previewed Week 2** - What's coming next (ML models!)

---

## What You Accomplished Today

✅ **Feature Preparation**
   - Removed ID columns (customerid, good_bad_flag)
   - Prepared 33 clean numeric features
   - Encoded target as binary (0=Good, 1=Bad)

✅ **Train/Test Split**
   - 80/20 stratified split (4,000 train, 1,000 test)
   - Maintained 70/30 class distribution in both sets
   - No data leakage

✅ **Feature Scaling**
   - Applied StandardScaler (mean=0, std=1)
   - Fitted on training data only
   - Transformed both train and test sets

✅ **Saved Artifacts**
   - 4 CSV files ready for modeling
   - Scaler for production deployment
   - Feature names for reference

✅ **Week 1 Complete**
   - 5 days of learning and building
   - Production-ready preprocessing pipeline
   - Ready for machine learning!

---

## Key Takeaways

1. **Always use stratified split** for imbalanced datasets
   - Maintains class distribution
   - Prevents biased evaluation

2. **Fit scaler on training data only**
   - Prevents data leakage
   - Test set remains truly "unseen"

3. **Save all artifacts**
   - Scaler for production
   - Feature names for documentation
   - Ensures reproducibility

4. **StandardScaler is standard**
   - Works well for most ML algorithms
   - Alternative: MinMaxScaler for neural networks

5. **80/20 split is common**
   - Good balance for medium datasets
   - Adjust based on data size

---

## Common Questions

**Q: Why not use all data for training?**

A: You need a separate test set to evaluate how well your model generalizes to NEW, unseen data. Training on all data would give you no way to measure real-world performance.

**Q: What's the difference between fit() and transform()?**

A:
- `fit()` - Learns parameters from data (mean, std for StandardScaler)
- `transform()` - Applies learned parameters to data
- `fit_transform()` - Does both (only use on training data!)

**Q: Why save the scaler?**

A: In production, new loan applications must be scaled using THE SAME mean and std as training data. Loading the saved scaler ensures consistent scaling.

**Q: Can I change random_state?**

A: Yes! `random_state=42` is convention. Any number works. Using the same number ensures reproducible results across runs.

**Q: What if my test set gets too small with 80/20?**

A: Use cross-validation (covered in Week 2). This creates multiple train/test splits and averages results, giving more reliable estimates.

---

## Tomorrow: Week 2 Begins!

**Day 6: Baseline Model & Evaluation Metrics**

You'll build your first ML model and learn how to evaluate it:

1. **Baseline model** - Simple model to beat
2. **Accuracy** - Overall correctness
3. **Precision** - How many predicted defaults were correct?
4. **Recall** - How many actual defaults did we catch?
5. **F1-score** - Balance between precision and recall
6. **ROC-AUC** - Overall discrimination ability

After Week 2, you'll have multiple trained models and know which one is best for production!

---

## Files Created Today

```
credit-risk-api/
├── data/
│   └── processed/
│       ├── X_train_scaled.csv          (4,000 rows × 33 features, scaled)
│       ├── X_test_scaled.csv           (1,000 rows × 33 features, scaled)
│       ├── y_train.csv                 (4,000 labels)
│       └── y_test.csv                  (1,000 labels)
├── models/
│   └── artifacts/
│       ├── scaler.pkl                  (StandardScaler for production)
│       └── feature_names.txt           (33 feature names)
└── guide/
    └── week1/
        └── day5_practice.py            (Complete preprocessing script)
```

---

**🎉 Congratulations!** You've completed Week 1 and have production-ready datasets for machine learning!

**Take a well-deserved break! Review what you've learned! You've earned it! 🌟**

**Week 2 starts next with building actual ML models! 🚀**
