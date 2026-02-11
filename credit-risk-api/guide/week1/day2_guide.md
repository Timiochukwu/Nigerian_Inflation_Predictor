# DAY 2: Data Merging - Complete Walkthrough

**Goal**: Combine loans, customers, and payments into one unified dataset

**Time**: 30 minutes

---

## 📋 What You'll Build Today

By the end of Day 2, you'll have:
- ✅ Merged loans + customers data (one row per loan with customer info)
- ✅ Aggregated payment history (payment patterns per customer)
- ✅ Final merged dataset with all features
- ✅ Saved to `data/processed/merged_day2.csv`

---

## Prerequisites

**Before starting Day 2, make sure you completed Day 1:**

```bash
# Check that Day 1 files exist
ls data/processed/loans_day1.csv
ls data/processed/customers_day1.csv
ls data/processed/payments_day1.csv
```

If files are missing, go back and complete Day 1:
```bash
python src/day1_load_explore.py
```

---

## PART 1: Understanding the Merge Strategy

**We have 3 datasets:**

1. **loans.csv** (5,000 rows) - One row per loan
   - `loan_id`, `customer_id`, `loan_amount`, `loan_term`, etc.

2. **customers.csv** (5,000 rows) - One row per customer
   - `customer_id`, `age`, `income`, `credit_score`, etc.

3. **payments.csv** (50,000 rows) - 10 rows per customer
   - `customer_id`, `payment_status`, `days_late`, etc.

**Merge Strategy:**
```
Step 1: loans + customers → One row per loan with customer info
Step 2: Aggregate payments (50,000 rows → 5,000 summary rows)
Step 3: Merge payment summaries → Final merged dataset
```

**Result**: One row per loan with loan info + customer info + payment history summary

---

## PART 2: Build Data Merging Script

### Step 1: Create Day 2 Merging Script

Create a new file: `src/day2_merge.py`

Delete everything in `src/day2_merge.py` and replace it with this:

```python
"""
DAY 2: Data Merging
Combine loans, customers, and payment history into one dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# Setup Paths
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

print("=" * 80)
print("DAY 2: DATA MERGING")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Load Day 1 Processed Data
# ============================================================================

print("📂 STEP 1: Loading Day 1 processed data...")
print("-" * 80)

loans_df = pd.read_csv(PROCESSED_DIR / 'loans_day1.csv')
customers_df = pd.read_csv(PROCESSED_DIR / 'customers_day1.csv')
payments_df = pd.read_csv(PROCESSED_DIR / 'payments_day1.csv')

print(f"✅ Loans:     {loans_df.shape[0]:,} rows × {loans_df.shape[1]} columns")
print(f"✅ Customers: {customers_df.shape[0]:,} rows × {customers_df.shape[1]} columns")
print(f"✅ Payments:  {payments_df.shape[0]:,} rows × {payments_df.shape[1]} columns")
print()

# ============================================================================
# STEP 2: Merge Loans with Customers
# ============================================================================

print("🔗 STEP 2: Merging loans with customers...")
print("-" * 80)

print("\n📋 Before merge:")
print(f"   Loans columns: {list(loans_df.columns)}")
print(f"   Customers columns: {list(customers_df.columns)}")

# Left join: Keep all loans, add customer info
merged_df = loans_df.merge(
    customers_df,
    on='customer_id',
    how='left'
)

print(f"\n✅ After loans + customers merge: {merged_df.shape}")
print(f"   Columns: {list(merged_df.columns)}")

# Check for missing values after merge
missing_after_merge = merged_df.isnull().sum().sum()
print(f"\n📊 Missing values after merge: {missing_after_merge}")

if missing_after_merge > 0:
    print("⚠️  Some loans have no matching customer!")
    print(merged_df[merged_df.isnull().any(axis=1)])
else:
    print("✅ All loans successfully matched with customers")

print()

# ============================================================================
# STEP 3: Aggregate Payment History
# ============================================================================

print("🔗 STEP 3: Aggregating payment history...")
print("-" * 80)

print("\n📋 Payment data structure:")
print(payments_df.head(15))  # Show payments for first 2 customers

print("\n🔄 Aggregating 50,000 payment records → 5,000 customer summaries...")

# For each customer, calculate payment behavior metrics
payment_agg = payments_df.groupby('customer_id').agg({
    'payment_status': lambda x: (x == 'paid').mean(),  # % of on-time payments
    'days_late': ['mean', 'max', 'min']  # Average, max, min late days
}).reset_index()

# Flatten column names
payment_agg.columns = [
    'customer_id',
    'ontime_rate',
    'avg_days_late',
    'max_days_late',
    'min_days_late'
]

print(f"✅ Payment aggregation complete: {payment_agg.shape}")
print("\n📊 Payment summary preview:")
print(payment_agg.head(10))

print("\n📈 Payment behavior statistics:")
print(payment_agg.describe())

print()

# ============================================================================
# STEP 4: Merge Payment Summaries
# ============================================================================

print("🔗 STEP 4: Merging payment summaries with main dataset...")
print("-" * 80)

print(f"\n📋 Before payment merge: {merged_df.shape}")

# Left join: Add payment summaries to main dataset
merged_df = merged_df.merge(
    payment_agg,
    on='customer_id',
    how='left'
)

print(f"✅ After payment merge: {merged_df.shape}")
print(f"   Total columns: {merged_df.shape[1]}")

# Check for customers with no payment history
customers_no_payments = merged_df['ontime_rate'].isnull().sum()
print(f"\n📊 Customers with no payment history: {customers_no_payments}")

if customers_no_payments > 0:
    print("⚠️  Filling missing payment history with safe defaults...")
    merged_df['ontime_rate'] = merged_df['ontime_rate'].fillna(1.0)  # Assume good
    merged_df['avg_days_late'] = merged_df['avg_days_late'].fillna(0)
    merged_df['max_days_late'] = merged_df['max_days_late'].fillna(0)
    merged_df['min_days_late'] = merged_df['min_days_late'].fillna(0)
    print("✅ Missing values filled")

print()

# ============================================================================
# STEP 5: Verify Final Merged Dataset
# ============================================================================

print("🔍 STEP 5: Verifying final merged dataset...")
print("-" * 80)

print("\n📋 Final dataset info:")
print(f"   Shape: {merged_df.shape}")
print(f"   Columns: {list(merged_df.columns)}")

print("\n📊 Final dataset preview:")
print(merged_df.head())

print("\n📈 Final dataset statistics:")
print(merged_df.describe())

print("\n⚠️  Missing values check:")
missing_summary = merged_df.isnull().sum()
print(missing_summary[missing_summary > 0])

if merged_df.isnull().sum().sum() == 0:
    print("✅ No missing values - dataset is complete!")

print()

# ============================================================================
# STEP 6: Save Merged Dataset
# ============================================================================

print("💾 STEP 6: Saving merged dataset...")
print("-" * 80)

output_file = PROCESSED_DIR / 'merged_day2.csv'
merged_df.to_csv(output_file, index=False)

print(f"✅ Saved: {output_file}")
print(f"   Size: {merged_df.shape[0]:,} rows × {merged_df.shape[1]} columns")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("🎉 DAY 2 COMPLETE!")
print("=" * 80)

print("\n📊 Summary:")
print(f"   • Started with 3 separate datasets")
print(f"   • Merged loans + customers → {merged_df.shape}")
print(f"   • Aggregated {payments_df.shape[0]:,} payment records → {payment_agg.shape[0]:,} summaries")
print(f"   • Final merged dataset: {merged_df.shape[0]:,} rows × {merged_df.shape[1]} columns")
print(f"   • Saved to: {output_file}")

print("\n📋 Available features:")
print("   From loans:", list(loans_df.columns))
print("   From customers:", list(customers_df.columns))
print("   From payments:", list(payment_agg.columns))

print("\n✅ Dataset is ready for feature engineering!")
print("\n✅ Next: Run Day 3 to create predictive features")
print("   python src/day3_feature_engineering.py")
```

---

## PART 3: Run Day 2 Script

### Step 1: Run the merging script

```bash
python src/day2_merge.py
```

**Expected output:**

```
================================================================================
DAY 2: DATA MERGING
================================================================================

📂 STEP 1: Loading Day 1 processed data...
--------------------------------------------------------------------------------
✅ Loans:     5,000 rows × 7 columns
✅ Customers: 5,000 rows × 7 columns
✅ Payments:  50,000 rows × 4 columns

🔗 STEP 2: Merging loans with customers...
--------------------------------------------------------------------------------

📋 Before merge:
   Loans columns: ['loan_id', 'customer_id', 'loan_amount', 'loan_term', 'interest_rate', 'purpose', 'loan_status']
   Customers columns: ['customer_id', 'age', 'income', 'employment_years', 'credit_score', 'total_debt', 'total_credit_limit']

✅ After loans + customers merge: (5000, 13)
   Columns: ['loan_id', 'customer_id', 'loan_amount', 'loan_term', 'interest_rate', 'purpose', 'loan_status', 'age', 'income', 'employment_years', 'credit_score', 'total_debt', 'total_credit_limit']

📊 Missing values after merge: 0
✅ All loans successfully matched with customers

🔗 STEP 3: Aggregating payment history...
--------------------------------------------------------------------------------

📋 Payment data structure:
  customer_id  payment_number payment_status  days_late
0      C00001               1           paid          0
1      C00001               2           paid          0
2      C00001               3           late         12
...

🔄 Aggregating 50,000 payment records → 5,000 customer summaries...
✅ Payment aggregation complete: (5000, 5)

📊 Payment summary preview:
  customer_id  ontime_rate  avg_days_late  max_days_late  min_days_late
0      C00001         0.90            2.5           12.0              0
1      C00002         0.30           35.7           78.0              0
...

🔗 STEP 4: Merging payment summaries with main dataset...
--------------------------------------------------------------------------------

📋 Before payment merge: (5000, 13)
✅ After payment merge: (5000, 17)
   Total columns: 17

📊 Customers with no payment history: 0

🔍 STEP 5: Verifying final merged dataset...
--------------------------------------------------------------------------------

📋 Final dataset info:
   Shape: (5000, 17)
   Columns: ['loan_id', 'customer_id', 'loan_amount', 'loan_term', 'interest_rate',
             'purpose', 'loan_status', 'age', 'income', 'employment_years', 'credit_score',
             'total_debt', 'total_credit_limit', 'ontime_rate', 'avg_days_late',
             'max_days_late', 'min_days_late']

✅ No missing values - dataset is complete!

💾 STEP 6: Saving merged dataset...
--------------------------------------------------------------------------------
✅ Saved: data/processed/merged_day2.csv
   Size: 5,000 rows × 17 columns

================================================================================
🎉 DAY 2 COMPLETE!
================================================================================

📊 Summary:
   • Started with 3 separate datasets
   • Merged loans + customers → (5000, 17)
   • Aggregated 50,000 payment records → 5,000 summaries
   • Final merged dataset: 5,000 rows × 17 columns
   • Saved to: data/processed/merged_day2.csv

✅ Dataset is ready for feature engineering!

✅ Next: Run Day 3 to create predictive features
   python src/day3_feature_engineering.py
```

### Step 2: Verify the merged file was created

```bash
ls -lh data/processed/merged_day2.csv
```

You should see:
```
-rw-r--r-- 1 user user 1.1M Feb 11 10:30 merged_day2.csv
```

### Step 3: Inspect the merged data

```bash
# View first few rows
head -n 5 data/processed/merged_day2.csv

# Count columns
head -n 1 data/processed/merged_day2.csv | tr ',' '\n' | wc -l
```

Expected: **17 columns**

---

## 🎓 What You Just Built

**You now have:**
1. ✅ **Merged dataset** combining loans, customers, and payment history
2. ✅ **17 features** ready for machine learning
3. ✅ **Payment behavior summaries** (on-time rate, average days late, max late days)
4. ✅ **No missing values** - clean dataset ready for feature engineering

**Key Features Added:**
- **From customers**: age, income, credit_score, total_debt, credit_limit
- **From payments**: ontime_rate, avg_days_late, max_days_late, min_days_late

---

## 🔍 Understanding the Code

**What is a LEFT JOIN?**
```python
merged_df = loans_df.merge(customers_df, on='customer_id', how='left')
```
- Keeps ALL loans (5,000 rows)
- Adds customer info where `customer_id` matches
- If no match, fills with NaN (shouldn't happen with our data)

**What is .groupby().agg()?**
```python
payment_agg = payments_df.groupby('customer_id').agg({
    'payment_status': lambda x: (x == 'paid').mean(),
    'days_late': ['mean', 'max', 'min']
})
```
- Groups 50,000 payment records by `customer_id`
- For each customer (5,000 groups):
  - Calculate % of on-time payments: `(x == 'paid').mean()`
  - Calculate average, max, min late days
- Result: 50,000 rows → 5,000 summary rows

**Example:**
```
Customer C00001 has 10 payment records:
  [paid, paid, late, paid, paid, paid, late, paid, paid, paid]

After aggregation:
  ontime_rate = 8/10 = 0.8 (80% on-time)
  avg_days_late = average of all days_late values
  max_days_late = highest days_late value
```

---

## ❓ Troubleshooting

**Error: "FileNotFoundError: loans_day1.csv"**
```bash
# You need to complete Day 1 first:
python src/day1_load_explore.py
```

**Error: "KeyError: 'customer_id'"**
```bash
# Check your data has the correct column names:
python -c "import pandas as pd; print(pd.read_csv('data/loans.csv').columns)"
```

**Warning: "Some loans have no matching customer"**
- This means your loans.csv has `customer_id` values not in customers.csv
- For production data, investigate these orphaned records
- For our generated data, this shouldn't happen

---

## ✅ Day 2 Checklist

Before moving to Day 3, verify:

- [ ] day2_merge.py script created
- [ ] Script ran successfully without errors
- [ ] merged_day2.csv exists in data/processed/
- [ ] File has 5,000 rows and 17 columns
- [ ] No missing values reported
- [ ] Understand left join and groupby aggregation

---

**🎉 Congratulations! Day 2 Complete!**

**You merged:**
- 5,000 loan records
- 5,000 customer records
- 50,000 payment records

**Into:** One unified dataset with 17 features!

**Next**: Day 3 - Feature Engineering (Create predictive features from raw data)

**Ready?** Run:
```bash
# Go to Day 3 guide
cat guide/week1/day3_guide.md
```
