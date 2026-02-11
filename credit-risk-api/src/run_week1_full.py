"""
Week 1 Complete Pipeline: Data Preprocessing (Days 1-5)
Run this to execute the entire Week 1 workflow
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

print("="*80)
print("WEEK 1: DATA PREPROCESSING PIPELINE")
print("="*80)
print()

# ============================================================================
# DAY 1: Data Exploration & Loading
# ============================================================================

print("\n" + "="*80)
print("DAY 1: Data Exploration & Loading")
print("="*80)

# Check if data exists, if not generate it
from src.data.generate_data import generate_sample_data

if not (DATA_DIR / 'loans.csv').exists():
    print("\n⚠️  Data not found. Generating sample data...")
    generate_sample_data()
    print("✅ Sample data generated!")

# Load data
print("\n📂 Loading datasets...")
loans_df = pd.read_csv(DATA_DIR / 'loans.csv')
customers_df = pd.read_csv(DATA_DIR / 'customers.csv')
payments_df = pd.read_csv(DATA_DIR / 'payments.csv')

print(f"✅ Loans: {loans_df.shape}")
print(f"✅ Customers: {customers_df.shape}")
print(f"✅ Payments: {payments_df.shape}")

# Save to processed folder
print(f"\n💾 Saving to processed folder...")
loans_df.to_csv(DATA_DIR / 'processed' / 'loans_day1.csv', index=False)
customers_df.to_csv(DATA_DIR / 'processed' / 'customers_day1.csv', index=False)
payments_df.to_csv(DATA_DIR / 'processed' / 'payments_day1.csv', index=False)
print("✅ Day 1 complete!")

# ============================================================================
# DAY 2: Data Merging
# ============================================================================

print("\n" + "="*80)
print("DAY 2: Data Merging")
print("="*80)

print("\n🔗 Merging datasets...")
# Merge loans with customers
merged_df = loans_df.merge(customers_df, on='customer_id', how='left')
print(f"✅ After customer merge: {merged_df.shape}")

# Aggregate payment history
payment_agg = payments_df.groupby('customer_id').agg({
    'payment_status': lambda x: (x == 'paid').mean(),
    'days_late': ['mean', 'max', 'min']
}).reset_index()
payment_agg.columns = ['customer_id', 'ontime_rate', 'avg_days_late', 'max_days_late', 'min_days_late']

# Merge payment history
merged_df = merged_df.merge(payment_agg, on='customer_id', how='left')
print(f"✅ After payment merge: {merged_df.shape}")

# Save merged data
merged_df.to_csv(DATA_DIR / 'processed' / 'merged_day2.csv', index=False)
print("✅ Day 2 complete!")

# ============================================================================
# DAY 3: Feature Engineering
# ============================================================================

print("\n" + "="*80)
print("DAY 3: Feature Engineering")
print("="*80)

print("\n🛠️  Engineering features...")

# Payment behavior features
merged_df['hist_ontime_rate'] = merged_df['ontime_rate'].fillna(1.0)
merged_df['hist_avg_days_late'] = merged_df['avg_days_late'].fillna(0)
merged_df['hist_max_days_late'] = merged_df['max_days_late'].fillna(0)
merged_df['hist_never_paid_rate'] = 1 - merged_df['hist_ontime_rate']

# Financial ratios
merged_df['loan_to_income'] = merged_df['loan_amount'] / (merged_df['income'] + 1)
merged_df['credit_util_ratio'] = merged_df['total_debt'] / (merged_df['total_credit_limit'] + 1)
merged_df['debt_to_income'] = merged_df['total_debt'] / (merged_df['income'] + 1)

# Demographic features
merged_df['age_group'] = pd.cut(merged_df['age'], bins=[0, 25, 35, 45, 55, 100],
                                 labels=['<25', '25-35', '35-45', '45-55', '55+'])
merged_df['income_group'] = pd.qcut(merged_df['income'], q=4,
                                      labels=['Q1', 'Q2', 'Q3', 'Q4'])

print(f"✅ Created {merged_df.shape[1]} total features")

# Save engineered data
merged_df.to_csv(DATA_DIR / 'processed' / 'engineered_day3.csv', index=False)
print("✅ Day 3 complete!")

# ============================================================================
# DAY 4: EDA & Outlier Detection
# ============================================================================

print("\n" + "="*80)
print("DAY 4: EDA & Outlier Detection")
print("="*80)

print("\n🔍 Detecting outliers...")

# IQR-based outlier detection
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
outlier_counts = {}

for col in numeric_cols:
    Q1 = merged_df[col].quantile(0.25)
    Q3 = merged_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    outliers = ((merged_df[col] < lower) | (merged_df[col] > upper)).sum()
    if outliers > 0:
        outlier_counts[col] = outliers

print(f"✅ Found outliers in {len(outlier_counts)} features")
for col, count in list(outlier_counts.items())[:5]:
    print(f"   • {col}: {count} outliers")

# Cap outliers (instead of removing)
for col in outlier_counts.keys():
    Q1 = merged_df[col].quantile(0.25)
    Q3 = merged_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    merged_df[col] = merged_df[col].clip(lower=lower, upper=upper)

# Save cleaned data
merged_df.to_csv(DATA_DIR / 'processed' / 'cleaned_day4.csv', index=False)
print("✅ Day 4 complete!")

# ============================================================================
# DAY 5: Preprocessing & Train/Test Split
# ============================================================================

print("\n" + "="*80)
print("DAY 5: Preprocessing & Train/Test Split")
print("="*80)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("\n🎯 Preparing for modeling...")

# Select features
feature_cols = [col for col in merged_df.columns if col not in
                ['customer_id', 'loan_id', 'loan_status', 'age_group', 'income_group']]

X = merged_df[feature_cols].copy()
y = (merged_df['loan_status'] == 'defaulted').astype(int)

print(f"✅ Features: {X.shape[1]}")
print(f"✅ Target distribution:")
print(f"   Good loans (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"   Bad loans (1):  {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Train set: {X_train.shape}")
print(f"✅ Test set: {X_test.shape}")

# Scale features
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

# Save final datasets
print("\n💾 Saving final datasets...")
X_train_scaled.to_csv(DATA_DIR / 'final' / 'X_train_scaled.csv', index=False)
X_test_scaled.to_csv(DATA_DIR / 'final' / 'X_test_scaled.csv', index=False)
y_train.to_csv(DATA_DIR / 'final' / 'y_train.csv', index=False, header=['loan_status'])
y_test.to_csv(DATA_DIR / 'final' / 'y_test.csv', index=False, header=['loan_status'])

# Save scaler
import joblib
joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')

print("✅ Day 5 complete!")

print("\n" + "="*80)
print("🎉 WEEK 1 COMPLETE!")
print("="*80)
print(f"\n📁 Files created:")
print(f"   • {DATA_DIR}/final/X_train_scaled.csv ({X_train.shape[0]:,} rows)")
print(f"   • {DATA_DIR}/final/X_test_scaled.csv ({X_test.shape[0]:,} rows)")
print(f"   • {DATA_DIR}/final/y_train.csv")
print(f"   • {DATA_DIR}/final/y_test.csv")
print(f"   • {MODELS_DIR}/scaler.pkl")
print(f"\n✅ Data pipeline ready for Week 2 (Model Training)!")
print(f"\nNext: Run 'python src/run_week2_full.py' to train models")
