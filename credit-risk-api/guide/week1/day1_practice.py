"""
Day 1 Practice: Project Setup & Data Exploration
Credit Risk Scoring System

Run this script to:
1. Load the three CSV files from Kaggle
2. Explore data shapes, types, and basic statistics
3. Understand the target variable distribution
4. Visualize key patterns

Prerequisites:
- Kaggle data files in data/raw/:
  - traindemographics.csv
  - trainperf.csv
  - trainprevloans.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Configure plotting
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("DAY 1: PROJECT SETUP & DATA EXPLORATION")
print("Credit Risk Scoring - Nigerian Inflation Predictor Project")
print("="*80)

# ============================================================================
# STEP 1: Load Data Files
# ============================================================================

print("\n📂 STEP 1: Loading Kaggle Data Files...")
print("-" * 80)

# Define paths
RAW_DATA_DIR = Path('data/raw')

# Load datasets
try:
    demographics = pd.read_csv(RAW_DATA_DIR / 'traindemographics.csv')
    performance = pd.read_csv(RAW_DATA_DIR / 'trainperf.csv')
    prev_loans = pd.read_csv(RAW_DATA_DIR / 'trainprevloans.csv')
    print("✅ All data files loaded successfully!\n")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("\n⚠️  Please ensure Kaggle CSV files are in data/raw/")
    exit(1)

# Display shapes
print(f"📊 Data Shapes:")
print(f"   Demographics:   {demographics.shape[0]:,} rows × {demographics.shape[1]} columns")
print(f"   Performance:    {performance.shape[0]:,} rows × {performance.shape[1]} columns")
print(f"   Previous Loans: {prev_loans.shape[0]:,} rows × {prev_loans.shape[1]} columns")

# ============================================================================
# STEP 2: Explore Performance Dataset (CONTAINS TARGET!)
# ============================================================================

print("\n" + "="*80)
print("📊 PERFORMANCE DATASET (Contains TARGET: good_bad_flag)")
print("="*80)

print("\n📋 Columns:")
for i, col in enumerate(performance.columns, 1):
    print(f"   {i}. {col}")

print("\n👀 First 5 Rows:")
print(performance.head())

print("\n🔤 Data Types:")
print(performance.dtypes)

print("\n❓ Missing Values:")
missing = performance.isnull().sum()
missing_pct = (missing / len(performance) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

print("\n🎯 TARGET VARIABLE DISTRIBUTION:")
target_dist = performance['good_bad_flag'].value_counts()
print(target_dist)
print("\nPercentages:")
print((target_dist / len(performance) * 100).round(2))

# KEY INSIGHT
bad_pct = (target_dist['Bad'] / len(performance) * 100)
print(f"\n💡 KEY INSIGHT: {bad_pct:.1f}% of loans are 'Bad' (default)")
if bad_pct < 40:
    print("   ⚠️  This is an IMBALANCED dataset - we'll need to handle this in modeling!")

# ============================================================================
# STEP 3: Explore Demographics Dataset
# ============================================================================

print("\n" + "="*80)
print("👥 DEMOGRAPHICS DATASET (Customer Information)")
print("="*80)

print("\n📋 Columns:")
for i, col in enumerate(demographics.columns, 1):
    print(f"   {i}. {col}")

print("\n👀 First 5 Rows:")
print(demographics.head())

print("\n🏦 Unique Banks:")
print(f"   {demographics['bank_name_clients'].nunique()} unique banks")
print(demographics['bank_name_clients'].value_counts().head(10))

print("\n📍 Unique Locations:")
print(f"   {demographics['bank_branch_clients'].nunique()} unique locations")
print(demographics['bank_branch_clients'].value_counts().head(10))

print("\n💼 Employment Status Distribution:")
print(demographics['employment_status_clients'].value_counts())

print("\n🎓 Education Level Distribution:")
print(demographics['level_of_education_clients'].value_counts())

# ============================================================================
# STEP 4: Explore Previous Loans Dataset
# ============================================================================

print("\n" + "="*80)
print("📚 PREVIOUS LOANS DATASET (Historical Loan Behavior)")
print("="*80)

print("\n📋 Columns:")
for i, col in enumerate(prev_loans.columns, 1):
    print(f"   {i}. {col}")

print("\n👀 First 5 Rows:")
print(prev_loans.head())

print("\n🔢 Key Statistics:")
print(f"   Total historical loans: {len(prev_loans):,}")
print(f"   Unique customers with history: {prev_loans['customerid'].nunique():,}")
print(f"   Average loans per customer: {len(prev_loans) / prev_loans['customerid'].nunique():.2f}")

# Loan amounts
print("\n💰 Loan Amount Statistics:")
print(prev_loans['loanamount'].describe())

# Check for missing dates
print("\n📅 Date Column Completeness:")
date_cols = ['creationdate', 'approveddate', 'firstduedate', 'firstrepaiddate', 'closeddate']
for col in date_cols:
    if col in prev_loans.columns:
        missing = prev_loans[col].isnull().sum()
        pct = (missing / len(prev_loans) * 100)
        print(f"   {col:20s}: {missing:6,} missing ({pct:5.1f}%)")

# ============================================================================
# STEP 5: Check Data Relationships
# ============================================================================

print("\n" + "="*80)
print("🔗 DATA RELATIONSHIPS & MERGE READINESS")
print("="*80)

# Check customerid overlap
demo_customers = set(demographics['customerid'].unique())
perf_customers = set(performance['customerid'].unique())
hist_customers = set(prev_loans['customerid'].unique())

print(f"\n📊 Customer ID Overlap:")
print(f"   Demographics customers:   {len(demo_customers):,}")
print(f"   Performance customers:    {len(perf_customers):,}")
print(f"   Previous loans customers: {len(hist_customers):,}")

# Check if all performance customers are in demographics
if perf_customers.issubset(demo_customers):
    print("   ✅ All performance customers exist in demographics")
else:
    missing = len(perf_customers - demo_customers)
    print(f"   ⚠️  {missing} performance customers NOT in demographics")

# Check how many customers have NO loan history
customers_no_history = perf_customers - hist_customers
print(f"\n👤 Customers with NO previous loan history: {len(customers_no_history):,}")
print(f"   ({len(customers_no_history)/len(perf_customers)*100:.1f}% of total)")

# ============================================================================
# STEP 6: Basic Visualizations
# ============================================================================

print("\n" + "="*80)
print("📈 GENERATING VISUALIZATIONS...")
print("="*80)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Day 1: Credit Risk Data Exploration', fontsize=16, fontweight='bold')

# 1. Target variable distribution
ax1 = axes[0, 0]
target_counts = performance['good_bad_flag'].value_counts()
colors = ['#2ecc71', '#e74c3c']  # Green for Good, Red for Bad
target_counts.plot(kind='bar', ax=ax1, color=colors, edgecolor='black')
ax1.set_title('Target Variable Distribution\n(Imbalanced Classes!)', fontweight='bold')
ax1.set_xlabel('Loan Status')
ax1.set_ylabel('Count')
ax1.set_xticklabels(target_counts.index, rotation=0)
# Add percentages on bars
for i, v in enumerate(target_counts):
    pct = v / len(performance) * 100
    ax1.text(i, v + 50, f'{v:,}\n({pct:.1f}%)', ha='center', fontweight='bold')

# 2. Loan amount distribution
ax2 = axes[0, 1]
performance['loanamount'].hist(bins=50, ax=ax2, color='skyblue', edgecolor='black')
ax2.set_title('Current Loan Amount Distribution', fontweight='bold')
ax2.set_xlabel('Loan Amount (₦)')
ax2.set_ylabel('Frequency')
ax2.axvline(performance['loanamount'].median(), color='red', linestyle='--',
            label=f'Median: ₦{performance["loanamount"].median():,.0f}')
ax2.legend()

# 3. Employment status distribution
ax3 = axes[1, 0]
employment_counts = demographics['employment_status_clients'].value_counts()
employment_counts.plot(kind='barh', ax=ax3, color='coral', edgecolor='black')
ax3.set_title('Employment Status Distribution', fontweight='bold')
ax3.set_xlabel('Count')
ax3.set_ylabel('Employment Type')

# 4. Education level distribution
ax4 = axes[1, 1]
education_counts = demographics['level_of_education_clients'].value_counts()
education_counts.plot(kind='pie', ax=ax4, autopct='%1.1f%%', startangle=90)
ax4.set_title('Education Level Distribution', fontweight='bold')
ax4.set_ylabel('')

plt.tight_layout()

# Save figure
output_dir = Path('results')
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / 'day1_data_exploration.png', dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved to: {output_dir}/day1_data_exploration.png")

# Show plot
plt.show()

# ============================================================================
# SUMMARY & NEXT STEPS
# ============================================================================

print("\n" + "="*80)
print("📝 DAY 1 SUMMARY")
print("="*80)

print(f"""
✅ Data Loaded Successfully:
   • Demographics: {demographics.shape[0]:,} customers
   • Performance: {performance.shape[0]:,} current applications
   • Previous Loans: {prev_loans.shape[0]:,} historical records

🎯 Key Findings:
   1. Target Distribution:
      - Good loans: {target_dist['Good']:,} ({target_dist['Good']/len(performance)*100:.1f}%)
      - Bad loans: {target_dist['Bad']:,} ({target_dist['Bad']/len(performance)*100:.1f}%)
      ⚠️  IMBALANCED DATASET - Need SMOTE/class weights

   2. Customer Coverage:
      - {len(customers_no_history):,} customers have NO loan history
      - Will need to handle missing historical features

   3. Data Quality:
      - Check missing values in key columns
      - Some historical dates are missing

   4. Feature Opportunities:
      - Can engineer features from loan history
      - Demographics provide rich context
      - Employment & education are categorical features

📅 Tomorrow (Day 2):
   • Merge the three datasets
   • Handle 1:Many relationships (customers with multiple prev loans)
   • Create aggregated features from loan history
   • Prepare master dataset for modeling

💡 Remember:
   • This is a CLASSIFICATION problem (Good vs Bad)
   • Imbalanced classes require special handling
   • Feature engineering from loan history will be crucial
""")

print("="*80)
print("🎉 DAY 1 COMPLETE! Great start!")
print("="*80)
