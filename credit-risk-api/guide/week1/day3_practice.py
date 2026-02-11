"""
Day 3 Practice: Feature Engineering from Historical Loans
Credit Risk Scoring System

What it does:
1. Loads Day 2 master dataset
2. Engineers payment behavior features from date columns
3. Calculates on-time rate, late rate, closure rate
4. Creates recency features
5. Encodes categorical variables
6. Merges all new features into master dataset
7. Saves enriched dataset

Run after: day2_practice.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 3: FEATURE ENGINEERING FROM HISTORICAL LOANS")
print("="*80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================

print("\n📂 STEP 1: Loading Data...")

PROCESSED_DIR = Path('credit-risk-api/data/processed')
RAW_DIR = Path('credit-risk-api/data/raw')

master_df = pd.read_csv(PROCESSED_DIR / 'master_dataset_day2.csv')
prev_loans = pd.read_csv(RAW_DIR / 'trainprevloans.csv')

print(f"✅ Master dataset: {master_df.shape}")
print(f"✅ Previous loans: {prev_loans.shape}")
print(f"   Starting features: {master_df.shape[1]}")

# ============================================================================
# STEP 2: Convert Date Columns
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Converting Date Columns")
print("="*80)

date_cols = ['creationdate', 'approveddate', 'firstduedate', 'firstrepaiddate', 'closeddate']
for col in date_cols:
    prev_loans[col] = pd.to_datetime(prev_loans[col], errors='coerce')

print("✅ Converted 5 date columns to datetime")

# ============================================================================
# STEP 3: Calculate Days to Repayment
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Calculating Payment Timing")
print("="*80)

prev_loans['days_to_repay'] = (prev_loans['firstrepaiddate'] - prev_loans['firstduedate']).dt.days

on_time = (prev_loans['days_to_repay'] <= 0).sum()
late = (prev_loans['days_to_repay'] > 0).sum()
never_paid = prev_loans['days_to_repay'].isnull().sum()
total = len(prev_loans)

print(f"✅ Days to repayment calculated:")
print(f"   On-time (≤0 days):  {on_time:,} ({on_time/total*100:.1f}%)")
print(f"   Late (>0 days):     {late:,} ({late/total*100:.1f}%)")
print(f"   Never paid (NaN):   {never_paid:,} ({never_paid/total*100:.1f}%)")

# ============================================================================
# STEP 4: Aggregate Payment Behavior per Customer
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Aggregating Payment Behavior Features")
print("="*80)

payment_features = []

for customer_id, group in prev_loans.groupby('customerid'):
    features = {'customerid': customer_id}

    # On-time rate
    on_time_count = (group['days_to_repay'] <= 0).sum()
    features['hist_ontime_rate'] = on_time_count / len(group)

    # Late rate
    late_count = (group['days_to_repay'] > 0).sum()
    features['hist_late_rate'] = late_count / len(group)

    # Never paid rate
    never_paid_count = group['days_to_repay'].isnull().sum()
    features['hist_never_paid_rate'] = never_paid_count / len(group)

    # Average days late (for late payments only)
    late_payments = group[group['days_to_repay'] > 0]['days_to_repay']
    features['hist_avg_days_late'] = late_payments.mean() if len(late_payments) > 0 else 0

    # Max days late
    features['hist_max_days_late'] = group['days_to_repay'].max() if group['days_to_repay'].notna().any() else 0

    # Closure rate
    closed_count = group['closeddate'].notna().sum()
    features['hist_closure_rate'] = closed_count / len(group)

    # Days since last loan (recency)
    most_recent = group['approveddate'].max()
    if pd.notna(most_recent):
        reference_date = pd.Timestamp('2024-07-01')  # Assume current date
        features['days_since_last_loan'] = (reference_date - most_recent).days
    else:
        features['days_since_last_loan'] = 9999

    payment_features.append(features)

payment_behavior_df = pd.DataFrame(payment_features)

print(f"✅ Payment behavior features created: {payment_behavior_df.shape}")
print(f"   New features: {list(payment_behavior_df.columns[1:])}")

# ============================================================================
# STEP 5: Merge Payment Behavior Features
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Merging Payment Features into Master Dataset")
print("="*80)

print(f"Before merge: {master_df.shape}")

# Merge payment features
master_df = master_df.merge(payment_behavior_df, on='customerid', how='left')

print(f"After merge: {master_df.shape}")

# Fill NaN for customers with no history
payment_cols = ['hist_ontime_rate', 'hist_late_rate', 'hist_never_paid_rate',
                'hist_avg_days_late', 'hist_max_days_late', 'hist_closure_rate', 'days_since_last_loan']

for col in payment_cols:
    if col == 'days_since_last_loan':
        master_df[col] = master_df[col].fillna(9999)  # Very old/no loans
    else:
        master_df[col] = master_df[col].fillna(0)

print("✅ Filled missing values for first-time borrowers")

# ============================================================================
# STEP 6: Encode Categorical Variables
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Encoding Categorical Variables")
print("="*80)

print("🔧 One-hot encoding employment and education...")

# One-hot encode employment status
master_df = pd.get_dummies(master_df, columns=['employment_status_clients'], prefix='emp', drop_first=False)

# One-hot encode education level
master_df = pd.get_dummies(master_df, columns=['level_of_education_clients'], prefix='edu', drop_first=False)

print(f"✅ Categorical encoding complete")
print(f"   Employment categories: {[col for col in master_df.columns if col.startswith('emp_')]}")
print(f"   Education categories: {[col for col in master_df.columns if col.startswith('edu_')]}")

# ============================================================================
# STEP 7: Create Additional Features
# ============================================================================

print("\n" + "="*80)
print("STEP 7: Creating Additional Features")
print("="*80)

# Loan-to-income proxy (loan amount / avg previous loan)
master_df['loan_to_hist_avg'] = master_df['loanamount'] / (master_df['avg_loan_amount'] + 1)  # +1 to avoid division by zero

# Interest rate (implied)
master_df['interest_rate'] = (master_df['totaldue'] / master_df['loanamount']) - 1

# Loan amount increase (current vs historical average)
master_df['loan_amount_increase'] = master_df['loanamount'] - master_df['avg_loan_amount']

print("✅ Created 3 additional derived features")

# ============================================================================
# STEP 8: Save Enriched Dataset
# ============================================================================

print("\n" + "="*80)
print("STEP 8: Saving Enriched Dataset")
print("="*80)

output_path = PROCESSED_DIR / 'master_dataset_day3.csv'
master_df.to_csv(output_path, index=False)

print(f"💾 Saved to: {output_path}")
print(f"   Shape: {master_df.shape}")
print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("📝 DAY 3 SUMMARY")
print("="*80)

print(f"""
✅ Feature Engineering Complete!

📊 Dataset Growth:
   • Day 2 features: 19
   • Day 3 features: {master_df.shape[1]}
   • New features added: {master_df.shape[1] - 19}

🎯 New Feature Categories:
   1. Payment Behavior (7 features):
      - hist_ontime_rate, hist_late_rate, hist_never_paid_rate
      - hist_avg_days_late, hist_max_days_late
      - hist_closure_rate, days_since_last_loan

   2. Categorical Encoding (~9 features):
      - Employment: Contract, Permanent, Self-Employed, Temporary
      - Education: HND/BSc, Masters, PhD, Secondary, None

   3. Derived Features (3 features):
      - loan_to_hist_avg, interest_rate, loan_amount_increase

💡 Why These Features Matter:
   • hist_ontime_rate: Past payment behavior predicts future behavior
   • hist_closure_rate: Completing loans shows reliability
   • days_since_last_loan: Recent borrowers may be financially stressed
   • Categorical encoding: Employment/education affect default risk

📅 Tomorrow (Day 4):
   • Deep EDA: Correlations, distributions, outliers
   • Identify which features are most predictive
   • Handle outliers and extreme values
   • Visualize feature relationships

🎉 DAY 3 COMPLETE! You now have {master_df.shape[1]} features ready for modeling!
""")

print(f"\n🔍 Feature List:")
for i, col in enumerate(master_df.columns, 1):
    print(f"   {i:2d}. {col}")
