"""
Day 2 Practice: Data Merging & Relationships
Credit Risk Scoring System

Run this script after reading day2_guide.md

What it does:
1. Loads three CSV files (demographics, performance, previous loans)
2. Performs 1:1 merge (demographics + performance)
3. Aggregates historical loans (1:Many → 1:1)
4. Merges historical summary
5. Handles missing values for first-time borrowers
6. Saves master dataset

Prerequisites:
- Kaggle data files in credit-risk-api/data/raw/:
  - traindemographics.csv
  - trainperf.csv
  - trainprevloans.csv
"""

import pandas as pd
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 2: DATA MERGING & RELATIONSHIPS")
print("="*80)

# Load data
RAW_DATA_DIR = Path('credit-risk-api/data/raw')
demographics = pd.read_csv(RAW_DATA_DIR / 'traindemographics.csv')
performance = pd.read_csv(RAW_DATA_DIR / 'trainperf.csv')
prev_loans = pd.read_csv(RAW_DATA_DIR / 'trainprevloans.csv')

print(f"\n📂 Loaded {performance.shape[0]:,} customers")

# Step 1: 1:1 merge (performance + demographics)
base_df = performance.merge(demographics, on='customerid', how='left')
print(f"✅ Step 1: Merged demographics → {base_df.shape}")

# Step 2: Aggregate historical loans
hist_summary = prev_loans.groupby('customerid').agg({
    'loanamount': ['count', 'sum', 'mean', 'max', 'min'],
    'totaldue': 'mean',
    'termdays': 'mean'
}).reset_index()

hist_summary.columns = ['customerid', 'num_prev_loans', 'total_borrowed',
                        'avg_loan_amount', 'max_loan_amount', 'min_loan_amount',
                        'avg_total_due', 'avg_term_days']

print(f"✅ Step 2: Aggregated history → {hist_summary.shape}")

# Step 3: Merge historical summary
master_df = base_df.merge(hist_summary, on='customerid', how='left')
print(f"✅ Step 3: Merged history → {master_df.shape}")

# Step 4: Fill missing values
hist_cols = ['num_prev_loans', 'total_borrowed', 'avg_loan_amount',
             'max_loan_amount', 'min_loan_amount', 'avg_total_due', 'avg_term_days']
master_df[hist_cols] = master_df[hist_cols].fillna(0)
print(f"✅ Step 4: Filled {len(hist_cols)} historical columns with 0 for new customers")

# Step 5: Save master dataset
print("\n" + "="*80)
print("SAVING MASTER DATASET")
print("="*80)

PROCESSED_DATA_DIR = Path('credit-risk-api/data/processed')
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

output_path = PROCESSED_DATA_DIR / 'master_dataset_day2.csv'
master_df.to_csv(output_path, index=False)

print(f"\n💾 Saved master dataset to:")
print(f"   {output_path}")
print(f"   Shape: {master_df.shape}")
print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")

print("\n" + "="*80)
print("📝 DAY 2 SUMMARY")
print("="*80)

print(f"""
✅ Successfully merged three datasets:
   1. traindemographics.csv  (5,000 rows) → Customer info
   2. trainperf.csv          (5,000 rows) → Target variable
   3. trainprevloans.csv     (15,312 rows) → Historical behavior

📊 Final Master Dataset:
   • Rows: {master_df.shape[0]:,} (one per customer)
   • Columns: {master_df.shape[1]} features
   • Missing values: 0

🎯 Columns breakdown:
   • 6 from performance (customerid, systemloanid, loanamount, totaldue, termdays, good_bad_flag)
   • 6 from demographics (bank, location, employment, education)
   • 7 from historical summary (aggregated loan statistics)

💡 Key Learnings:
   • 1:1 merge: demographics + performance (same # rows)
   • 1:Many problem: customers have multiple historical loans
   • Solution: Aggregate first, then merge
   • Handle missing history: 485 first-time borrowers → fill with 0

📅 Tomorrow (Day 3):
   • Engineer advanced features from date columns
   • Calculate on-time payment rate, days late, closure rate
   • Create 20+ new features for better predictions
   • Encode categorical variables (employment, education)

🎉 DAY 2 COMPLETE!
""")
