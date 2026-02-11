# 📅 Day 2: Data Merging & Understanding Relationships

**Goal:** Merge the 3 datasets correctly and create a unified master dataset for modeling.

---

## 🎯 Learning Objectives
- Understand 1:1 vs 1:Many data relationships
- Merge demographics and performance (1:1)
- Aggregate previous loans data (Many:1)
- Validate merged data integrity
- Handle duplicate customerids

---

## 📚 Recap from Day 1

We have 3 datasets:
- **performance.csv** (4k rows) - TARGET: `good_bad_flag` ⭐
- **demographics.csv** (4k rows) - Customer info
- **prev_loans.csv** (18k rows) - Historical loans (multiple per customer)

**Key Insight:** We need ONE row per customer for machine learning!

---

## 💻 Step 1: Load Data (Quick Refresh)

Create a new notebook: `notebooks/day2_merging.ipynb`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
demographics = pd.read_csv('../data/raw/traindemographics.csv')
performance = pd.read_csv('../data/raw/trainperf.csv')
prev_loans = pd.read_csv('../data/raw/trainprevloans.csv')

print("✅ Data loaded!")
print(f"Performance: {performance.shape}")
print(f"Demographics: {demographics.shape}")
print(f"Previous Loans: {prev_loans.shape}")
```

---

## 🔍 Step 2: Understand Data Relationships

### Check for Duplicate CustomerIDs

```python
print("=" * 60)
print("CHECKING FOR DUPLICATES")
print("=" * 60)

# Performance dataset
print("\n📊 Performance dataset:")
print(f"Total rows: {len(performance)}")
print(f"Unique customerids: {performance['customerid'].nunique()}")
print(f"Duplicates? {len(performance) != performance['customerid'].nunique()}")

if len(performance) != performance['customerid'].nunique():
    print("⚠️ WARNING: Duplicate customerids found!")
    dupes = performance[performance.duplicated('customerid', keep=False)]
    print(f"Number of duplicate customers: {dupes['customerid'].nunique()}")
    print("\nExample duplicates:")
    print(dupes.head())

# Demographics dataset
print("\n📊 Demographics dataset:")
print(f"Total rows: {len(demographics)}")
print(f"Unique customerids: {demographics['customerid'].nunique()}")
print(f"Duplicates? {len(demographics) != demographics['customerid'].nunique()}")

# Previous loans dataset
print("\n📊 Previous Loans dataset:")
print(f"Total rows: {len(prev_loans)}")
print(f"Unique customerids: {prev_loans['customerid'].nunique()}")
loans_per_customer = prev_loans.groupby('customerid').size()
print(f"Average loans per customer: {loans_per_customer.mean():.2f}")
print(f"This is EXPECTED - one customer can have many previous loans!")
```

**Expected Output:**
- Performance: No duplicates (or very few)
- Demographics: No duplicates
- Previous Loans: Many duplicates (EXPECTED!)

---

### Visualize the Relationships

```python
# Create a diagram showing relationships
print("\n" + "=" * 60)
print("DATA RELATIONSHIPS")
print("=" * 60)

print("""
┌─────────────────────┐
│  performance.csv    │  ← Contains TARGET (good_bad_flag)
│  (~4k rows)         │
│  1 row per customer │
└─────────┬───────────┘
          │
          │ customerid (1:1)
          │
┌─────────▼────────────┐
│ demographics.csv     │
│ (~4k rows)           │
│ 1 row per customer   │
└──────────────────────┘

┌──────────────────────┐
│  prev_loans.csv      │
│  (~18k rows)         │
│  MANY rows per       │ ◄── Need to AGGREGATE this!
│  customer            │
└─────────┬────────────┘
          │
          │ customerid (Many:1)
          │
          └──────────────► Join to performance
                           after aggregation
""")
```

---

## 🔗 Step 3: Merge Performance + Demographics (1:1)

This is straightforward because both have one row per customer.

```python
print("=" * 60)
print("MERGING PERFORMANCE + DEMOGRAPHICS")
print("=" * 60)

# Check if all customerids match
perf_ids = set(performance['customerid'])
demo_ids = set(demographics['customerid'])

print(f"\nCustomers in performance: {len(perf_ids)}")
print(f"Customers in demographics: {len(demo_ids)}")
print(f"Customers in BOTH: {len(perf_ids & demo_ids)}")
print(f"Only in performance: {len(perf_ids - demo_ids)}")
print(f"Only in demographics: {len(demo_ids - perf_ids)}")

# Merge (use left join to keep all customers from performance)
df_master = performance.merge(
    demographics,
    on='customerid',
    how='left',
    indicator=True  # Track merge source
)

print(f"\n✅ Merged dataset shape: {df_master.shape}")
print("\nMerge indicator:")
print(df_master['_merge'].value_counts())

# Drop the merge indicator
df_master = df_master.drop('_merge', axis=1)

print("\n📋 Columns after merge:")
print(df_master.columns.tolist())
```

**What to expect:**
- Most/all customers should be in both datasets
- Shape should be ~4k rows, ~19 columns (10 from perf + 9 from demo)

---

## 📊 Step 4: Aggregate Previous Loans Data

This is the **MOST IMPORTANT** step! We need to convert 18k rows into 4k rows.

### 4.1 Count Number of Previous Loans

```python
print("=" * 60)
print("AGGREGATING PREVIOUS LOANS - PART 1: COUNTS")
print("=" * 60)

# Count loans per customer
loan_counts = prev_loans.groupby('customerid').agg({
    'systemloanid': 'count'  # Count of loans
}).rename(columns={'systemloanid': 'prev_loan_count'})

print(f"✅ Loan counts aggregated: {loan_counts.shape}")
print("\nPreview:")
print(loan_counts.head(10))

# Distribution of loan counts
print("\n📊 Distribution of previous loan counts:")
print(loan_counts['prev_loan_count'].value_counts().sort_index().head(15))
```

---

### 4.2 Aggregate Loan Amounts

```python
print("\n" + "=" * 60)
print("AGGREGATING PREVIOUS LOANS - PART 2: AMOUNTS")
print("=" * 60)

# Aggregate loan amounts
loan_amounts = prev_loans.groupby('customerid').agg({
    'loanamount': ['sum', 'mean', 'min', 'max', 'std'],
    'totaldue': ['sum', 'mean']
}).reset_index()

# Flatten column names
loan_amounts.columns = ['customerid',
                        'prev_loan_sum', 'prev_loan_mean', 'prev_loan_min', 'prev_loan_max', 'prev_loan_std',
                        'prev_totaldue_sum', 'prev_totaldue_mean']

print(f"✅ Loan amounts aggregated: {loan_amounts.shape}")
print("\nPreview:")
print(loan_amounts.head())

# Fill NaN in std (happens when customer has only 1 loan)
loan_amounts['prev_loan_std'] = loan_amounts['prev_loan_std'].fillna(0)
```

---

### 4.3 Aggregate Loan Term Days

```python
print("\n" + "=" * 60)
print("AGGREGATING PREVIOUS LOANS - PART 3: TERM DAYS")
print("=" * 60)

# Aggregate term days
loan_terms = prev_loans.groupby('customerid').agg({
    'termdays': ['mean', 'max', 'min']
}).reset_index()

loan_terms.columns = ['customerid', 'prev_termdays_mean', 'prev_termdays_max', 'prev_termdays_min']

print(f"✅ Loan terms aggregated: {loan_terms.shape}")
print("\nPreview:")
print(loan_terms.head())
```

---

### 4.4 Calculate Repayment Features (IMPORTANT!)

These features indicate if customers paid on time!

```python
print("\n" + "=" * 60)
print("AGGREGATING PREVIOUS LOANS - PART 4: REPAYMENT BEHAVIOR")
print("=" * 60)

# Convert date columns to datetime
date_cols = ['approveddate', 'creationdate', 'closeddate', 'firstduedate', 'firstrepaiddate']
for col in date_cols:
    prev_loans[col] = pd.to_datetime(prev_loans[col], errors='coerce')

# Calculate days between first due and first repaid
prev_loans['days_to_repay'] = (prev_loans['firstrepaiddate'] - prev_loans['firstduedate']).dt.days

# Flag: Did customer pay BEFORE due date? (negative days_to_repay)
prev_loans['paid_early'] = (prev_loans['days_to_repay'] < 0).astype(int)

# Flag: Did customer pay LATE? (positive days_to_repay)
prev_loans['paid_late'] = (prev_loans['days_to_repay'] > 0).astype(int)

# Flag: Loan was closed
prev_loans['loan_closed'] = prev_loans['closeddate'].notna().astype(int)

print("✅ Repayment features created!")
print("\nSample of repayment data:")
print(prev_loans[['customerid', 'loanamount', 'firstduedate', 'firstrepaiddate',
                   'days_to_repay', 'paid_early', 'paid_late', 'loan_closed']].head(10))

# Aggregate repayment behavior
repayment_agg = prev_loans.groupby('customerid').agg({
    'days_to_repay': ['mean', 'max', 'min', 'std'],
    'paid_early': 'sum',  # Count of early payments
    'paid_late': 'sum',   # Count of late payments
    'loan_closed': 'sum'  # Count of closed loans
}).reset_index()

repayment_agg.columns = ['customerid',
                         'days_to_repay_mean', 'days_to_repay_max', 'days_to_repay_min', 'days_to_repay_std',
                         'count_paid_early', 'count_paid_late', 'count_loans_closed']

# Fill NaN
repayment_agg = repayment_agg.fillna(0)

print(f"\n✅ Repayment behavior aggregated: {repayment_agg.shape}")
print("\nPreview:")
print(repayment_agg.head())
```

---

### 4.5 Calculate Loan Utilization Rate

```python
print("\n" + "=" * 60)
print("AGGREGATING PREVIOUS LOANS - PART 5: UTILIZATION RATE")
print("=" * 60)

# Calculate average utilization: totaldue / loanamount
# (How much interest/fees relative to loan amount)
prev_loans['utilization_rate'] = prev_loans['totaldue'] / prev_loans['loanamount']

utilization_agg = prev_loans.groupby('customerid').agg({
    'utilization_rate': ['mean', 'max']
}).reset_index()

utilization_agg.columns = ['customerid', 'utilization_rate_mean', 'utilization_rate_max']

print(f"✅ Utilization rate aggregated: {utilization_agg.shape}")
print("\nPreview:")
print(utilization_agg.head())
```

---

### 4.6 Combine All Previous Loan Features

```python
print("\n" + "=" * 60)
print("COMBINING ALL PREVIOUS LOAN FEATURES")
print("=" * 60)

# Merge all aggregated features
prev_loans_features = loan_counts
prev_loans_features = prev_loans_features.merge(loan_amounts, on='customerid', how='left')
prev_loans_features = prev_loans_features.merge(loan_terms, on='customerid', how='left')
prev_loans_features = prev_loans_features.merge(repayment_agg, on='customerid', how='left')
prev_loans_features = prev_loans_features.merge(utilization_agg, on='customerid', how='left')

# Reset index to make customerid a column
prev_loans_features = prev_loans_features.reset_index()

print(f"✅ All previous loan features combined: {prev_loans_features.shape}")
print(f"   Columns: {prev_loans_features.shape[1]}")
print("\nColumn names:")
print(prev_loans_features.columns.tolist())

print("\nPreview of aggregated features:")
print(prev_loans_features.head())
```

---

## 🔗 Step 5: Final Merge - Everything Together!

```python
print("=" * 60)
print("FINAL MERGE: MASTER DATASET")
print("=" * 60)

# Merge master (performance + demographics) with previous loans features
df_final = df_master.merge(
    prev_loans_features,
    on='customerid',
    how='left',  # Keep all customers even if no previous loans
    indicator=True
)

print(f"\n✅ Final dataset shape: {df_final.shape}")
print("\nMerge indicator:")
print(df_final['_merge'].value_counts())

# Check how many customers have NO previous loans
no_prev_loans = df_final['_merge'] == 'left_only'
print(f"\n⚠️ Customers with NO previous loan data: {no_prev_loans.sum()}")

# Drop merge indicator
df_final = df_final.drop('_merge', axis=1)

print("\n📋 Final columns:")
print(df_final.columns.tolist())
print(f"\nTotal columns: {df_final.shape[1]}")
```

---

## ✅ Step 6: Data Validation

Always validate your merge!

```python
print("=" * 60)
print("DATA VALIDATION")
print("=" * 60)

# 1. Check shape
print("\n1️⃣ Shape check:")
print(f"   Final dataset: {df_final.shape}")
print(f"   Expected rows: {len(performance)} (from performance dataset)")
print(f"   ✅ Match!" if df_final.shape[0] == len(performance) else "❌ Mismatch!")

# 2. Check target variable is preserved
print("\n2️⃣ Target variable check:")
print(f"   Original Good/Bad counts:")
print(f"   {performance['good_bad_flag'].value_counts()}")
print(f"\n   Final Good/Bad counts:")
print(f"   {df_final['good_bad_flag'].value_counts()}")
print(f"   ✅ Match!" if (df_final['good_bad_flag'].value_counts() == performance['good_bad_flag'].value_counts()).all() else "❌ Mismatch!")

# 3. Check for duplicate customerids
print("\n3️⃣ Duplicate check:")
print(f"   Unique customers: {df_final['customerid'].nunique()}")
print(f"   Total rows: {len(df_final)}")
print(f"   ✅ No duplicates!" if df_final['customerid'].nunique() == len(df_final) else "❌ Duplicates found!")

# 4. Missing values in key columns
print("\n4️⃣ Missing values in key columns:")
key_cols = ['customerid', 'good_bad_flag', 'loanamount', 'birthdate', 'prev_loan_count']
missing = df_final[key_cols].isnull().sum()
print(missing)

# 5. Preview final dataset
print("\n5️⃣ Final dataset preview:")
print(df_final.head())
print("\nTarget distribution:")
print(df_final['good_bad_flag'].value_counts())
```

---

## 💾 Step 7: Save the Merged Dataset

```python
print("=" * 60)
print("SAVING MERGED DATASET")
print("=" * 60)

# Save to processed folder
output_path = '../data/processed/merged_data.csv'
df_final.to_csv(output_path, index=False)

print(f"✅ Dataset saved to: {output_path}")
print(f"   Rows: {df_final.shape[0]}")
print(f"   Columns: {df_final.shape[1]}")

# Also save as a backup with timestamp
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_path = f'../data/processed/merged_data_backup_{timestamp}.csv'
df_final.to_csv(backup_path, index=False)
print(f"✅ Backup saved to: {backup_path}")
```

---

## 📊 Step 8: Quick EDA on Merged Data

```python
print("=" * 60)
print("QUICK EDA ON MERGED DATASET")
print("=" * 60)

# Summary statistics
print("\n📊 Summary Statistics:")
print(df_final.describe())

# Check correlation of previous loan features with target
print("\n🔗 Correlation Analysis:")

# Convert target to numeric (Good=1, Bad=0)
df_final['target_numeric'] = (df_final['good_bad_flag'] == 'Good').astype(int)

# Select numeric columns
numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()

# Correlation with target
target_corr = df_final[numeric_cols].corr()['target_numeric'].sort_values(ascending=False)
print("\nTop 15 features correlated with Good loans:")
print(target_corr.head(15))

print("\nBottom 15 features (negatively correlated with Good loans):")
print(target_corr.tail(15))

# Visualize top correlations
top_10_features = target_corr.drop('target_numeric').abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
top_10_features.plot(kind='barh', color='steelblue')
plt.title('Top 10 Features by Absolute Correlation with Target', fontsize=14, fontweight='bold')
plt.xlabel('Absolute Correlation')
plt.tight_layout()
plt.show()
```

---

## 🔑 Key Findings Summary

```python
print("\n" + "=" * 60)
print("📝 DAY 2 KEY FINDINGS")
print("=" * 60)

print("\n1️⃣ FINAL DATASET:")
print(f"   - Rows: {df_final.shape[0]}")
print(f"   - Columns: {df_final.shape[1]}")
print(f"   - Target preserved: {(df_final['good_bad_flag'].value_counts() == performance['good_bad_flag'].value_counts()).all()}")

print("\n2️⃣ PREVIOUS LOAN FEATURES CREATED:")
prev_loan_cols = [col for col in df_final.columns if col.startswith('prev_')]
print(f"   - Total: {len(prev_loan_cols)}")
print(f"   - Features: {prev_loan_cols}")

print("\n3️⃣ CUSTOMERS WITH NO PREVIOUS LOANS:")
no_prev = df_final['prev_loan_count'].isnull().sum()
print(f"   - Count: {no_prev}")
print(f"   - Percentage: {(no_prev / len(df_final)) * 100:.1f}%")

print("\n4️⃣ TOP PREDICTIVE FEATURES (by correlation):")
print(target_corr.drop('target_numeric').abs().sort_values(ascending=False).head(5))

print("\n5️⃣ NEXT STEPS (Day 3):")
print("   - Handle missing values")
print("   - Create more advanced features")
print("   - Engineer date-based features")
print("   - Encode categorical variables")
```

---

## ✅ Day 2 Checklist

- [ ] Loaded all 3 datasets
- [ ] Checked for duplicate customerids
- [ ] Merged performance + demographics (1:1)
- [ ] Aggregated previous loans data (Many:1)
- [ ] Created 20+ features from previous loans
- [ ] Merged everything into master dataset
- [ ] Validated merge (shape, target, duplicates)
- [ ] Saved merged dataset to `data/processed/`
- [ ] Performed quick correlation analysis
- [ ] Notebook saved as `notebooks/day2_merging.ipynb`

---

## 🎓 What You Learned Today

1. **Data Relationships:** Understanding 1:1 vs 1:Many joins
2. **Aggregation:** Converting many rows to one row per customer
3. **Feature Engineering:** Creating features from historical data
4. **Merge Validation:** Checking data integrity after merging
5. **Pandas Skills:** `.merge()`, `.groupby()`, `.agg()`
6. **Domain Knowledge:** Repayment behavior as a predictor

---

## 🚀 Tomorrow (Day 3)

**Topic:** Advanced Feature Engineering

You'll learn:
- Creating time-based features (age, days since last loan)
- Encoding categorical variables
- Handling missing values strategically
- Feature scaling and normalization
- Creating interaction features

---

## 💡 Pro Tips

1. **Always use `indicator=True` in merge** - Helps track where data came from
2. **Validate after every merge** - Check shapes and key columns
3. **Document your aggregations** - You'll forget what each feature means!
4. **Save intermediate datasets** - Easier to debug issues
5. **Use descriptive column names** - `prev_loan_mean` > `mean_amount`

---

## 📚 Resources

- [Pandas Merge Documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html)
- [GroupBy User Guide](https://pandas.pydata.org/docs/user_guide/groupby.html)
- [Feature Engineering for Machine Learning](https://www.kaggle.com/learn/feature-engineering)

---

**🎉 Congratulations on completing Day 2!**

Your master dataset is ready! Tomorrow we'll make it even better with advanced feature engineering.
