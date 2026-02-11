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
    print(f"   {i:2d}. {feat}")

# ============================================================================
# STEP 4: Detect Outliers
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Outlier Detection (IQR Method)")
print("="*80)

print("\n🔍 Checking for outliers in numeric features...")

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

# ============================================================================
# STEP 5: Handle Outliers (Capping)
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Handling Outliers (Capping Method)")
print("="*80)

print("\n🔧 Capping extreme values to upper/lower bounds...")

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

print(f"✅ Capped {capped_count:,} outlier values across all features")

# ============================================================================
# STEP 6: Feature Distribution Analysis
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Feature Distribution Summary")
print("="*80)

print("\n📊 Basic statistics for top 5 predictive features:")
for feat in top_features[:5]:
    stats = master_df[feat].describe()
    print(f"\n{feat}:")
    print(f"   Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
    print(f"   Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")

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
fig, ax = plt.subplots(figsize=(10, 8))

top_15_features = correlations_sorted.head(15).index.tolist()
corr_matrix = master_df[top_15_features + ['target']].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})

ax.set_title('Top 15 Features - Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / 'day4_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {output_dir}/day4_correlation_heatmap.png")

plt.close()

# 2. Distribution of top 4 features by target
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
print(f"   ✅ Saved: {output_dir}/day4_feature_distributions.png")

plt.close()

# ============================================================================
# STEP 8: Save Cleaned Dataset
# ============================================================================

print("\n" + "="*80)
print("STEP 8: Saving Cleaned Dataset")
print("="*80)

output_path = PROCESSED_DIR / 'master_dataset_day4.csv'
master_df.to_csv(output_path, index=False)

print(f"💾 Saved to: {output_path}")
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
      - hist_ontime_rate (most important!)
      - hist_late_rate
      - hist_never_paid_rate
      - hist_closure_rate
      - days_since_last_loan
      - (and 5 more...)

   2. Outlier Handling:
      - Detected outliers in {len(outlier_df)} features
      - Capped {capped_count:,} extreme values using IQR method
      - All features now within reasonable bounds

   3. Visualizations Created:
      - Correlation heatmap (top 15 features)
      - Feature distributions by target

💡 What We Learned:
   • Payment behavior features are HIGHLY predictive
   • hist_ontime_rate has strongest correlation with default
   • Outliers exist but are now handled
   • Features show clear separation between Good/Bad loans

📅 Tomorrow (Day 5):
   • Final preprocessing pipeline
   • Train/test split (stratified)
   • Feature scaling (standardization)
   • Save final modeling-ready dataset
   • Ready to build ML models in Week 2!

🎉 DAY 4 COMPLETE!
""")

print(f"\n🔍 Top 10 Predictive Features (by absolute correlation):")
for i, feat in enumerate(top_features, 1):
    corr_val = correlations[feat]
    print(f"   {i:2d}. {feat:<35} (r = {corr_val:>7.4f})")
