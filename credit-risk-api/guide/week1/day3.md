# 📅 Day 3: Advanced Feature Engineering

**Goal:** Create powerful features that will boost your model's predictive performance.

---

## 🎯 Learning Objectives
- Create time-based features (age, recency, tenure)
- Encode categorical variables (one-hot, label encoding)
- Handle missing values strategically
- Create ratio and interaction features
- Feature transformation (log, sqrt)

---

## 📚 Recap from Day 2

We created a master dataset with:
- Demographics
- Current loan info + TARGET
- Aggregated previous loan features (20+ features!)

Today we'll make it **even better** 🚀

---

## 💻 Step 1: Load Merged Data

Create a new notebook: `notebooks/day3_feature_engineering.ipynb`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load merged data from Day 2
df = pd.read_csv('../data/processed/merged_data.csv')

print("✅ Data loaded!")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Convert date columns back to datetime
date_cols = ['approveddate', 'creationdate', 'birthdate']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

print("\n✅ Date columns converted!")
```

---

## 🕒 Step 2: Time-Based Features

### 2.1 Customer Age

```python
print("=" * 60)
print("TIME-BASED FEATURES: AGE")
print("=" * 60)

# Calculate age at loan application
# Use the loan approved date as reference
reference_date = pd.to_datetime('2017-12-31')  # End of training data period

df['age'] = (reference_date - df['birthdate']).dt.days // 365

print(f"✅ Age calculated!")
print(f"\nAge statistics:")
print(df['age'].describe())

# Age groups (binning)
df['age_group'] = pd.cut(df['age'],
                         bins=[0, 25, 35, 45, 55, 100],
                         labels=['18-25', '26-35', '36-45', '46-55', '55+'])

print(f"\nAge group distribution:")
print(df['age_group'].value_counts().sort_index())

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Age distribution
axes[0].hist(df['age'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_title('Customer Age Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Age (years)')
axes[0].set_ylabel('Count')
axes[0].axvline(df['age'].mean(), color='red', linestyle='--', label=f'Mean: {df["age"].mean():.1f}')
axes[0].legend()

# Age vs Target
age_target = df.groupby('age_group')['good_bad_flag'].apply(lambda x: (x == 'Good').mean() * 100)
age_target.plot(kind='bar', ax=axes[1], color='green', alpha=0.7)
axes[1].set_title('Loan Performance by Age Group', fontsize=14, fontweight='bold')
axes[1].set_ylabel('% Good Loans')
axes[1].set_xlabel('Age Group')
axes[1].tick_params(rotation=45)
axes[1].axhline(50, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
```

---

### 2.2 Loan Recency & Frequency Features

```python
print("\n" + "=" * 60)
print("TIME-BASED FEATURES: RECENCY")
print("=" * 60)

# Calculate days since loan was approved
df['days_since_approval'] = (reference_date - df['approveddate']).dt.days

# Days between creation and approval (processing time)
df['approval_processing_days'] = (df['approveddate'] - df['creationdate']).dt.days

print(f"✅ Recency features created!")
print(f"\nDays since approval:")
print(df['days_since_approval'].describe())

print(f"\nApproval processing time:")
print(df['approval_processing_days'].describe())

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['days_since_approval'].dropna(), bins=50, edgecolor='black', alpha=0.7)
plt.title('Days Since Loan Approval', fontsize=14, fontweight='bold')
plt.xlabel('Days')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(df['approval_processing_days'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='orange')
plt.title('Loan Approval Processing Time', fontsize=14, fontweight='bold')
plt.xlabel('Days')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
```

---

### 2.3 Loan-to-Age Ratio

```python
print("\n" + "=" * 60)
print("INTERACTION FEATURE: LOAN-TO-AGE RATIO")
print("=" * 60)

# Higher loan amount relative to age might indicate higher risk
df['loan_per_year_age'] = df['loanamount'] / df['age']

print(f"✅ Loan-per-year-age created!")
print(df['loan_per_year_age'].describe())

# Visualize by target
plt.figure(figsize=(10, 5))
df.boxplot(column='loan_per_year_age', by='good_bad_flag', figsize=(8, 5))
plt.suptitle('')
plt.title('Loan per Year of Age by Loan Performance', fontsize=14, fontweight='bold')
plt.ylabel('Loan Amount / Age')
plt.xlabel('Loan Status')
plt.tight_layout()
plt.show()
```

---

## 🔤 Step 3: Categorical Variable Encoding

### 3.1 Identify Categorical Variables

```python
print("=" * 60)
print("CATEGORICAL VARIABLES")
print("=" * 60)

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove customerid and target
categorical_cols = [col for col in categorical_cols if col not in ['customerid', 'good_bad_flag', 'systemloanid']]

print(f"Categorical columns: {categorical_cols}")

# Check unique values in each
for col in categorical_cols:
    n_unique = df[col].nunique()
    print(f"\n{col}:")
    print(f"  Unique values: {n_unique}")
    print(f"  Missing: {df[col].isnull().sum()} ({df[col].isnull().mean() * 100:.1f}%)")
    if n_unique <= 10:
        print(f"  Values: {df[col].value_counts().to_dict()}")
```

---

### 3.2 Handle Bank Name (High Cardinality)

```python
print("\n" + "=" * 60)
print("ENCODING: BANK NAME")
print("=" * 60)

# For high cardinality (many unique values), keep top N and group others
top_banks = df['bank_name_clients'].value_counts().head(10).index.tolist()

df['bank_name_grouped'] = df['bank_name_clients'].apply(
    lambda x: x if x in top_banks else 'Other'
)

print(f"✅ Bank name grouped into top 10 + Other")
print(f"\nDistribution:")
print(df['bank_name_grouped'].value_counts())

# Also create a binary feature: Is it a top bank?
df['is_top_bank'] = df['bank_name_clients'].isin(top_banks).astype(int)

print(f"\nTop bank indicator created!")
print(f"Top bank customers: {df['is_top_bank'].sum()}")
```

---

### 3.3 One-Hot Encoding

```python
print("\n" + "=" * 60)
print("ONE-HOT ENCODING")
print("=" * 60)

# Columns to one-hot encode
cols_to_encode = ['bank_account_type', 'employment_status_clients',
                  'level_of_education_clients', 'bank_name_grouped']

# Before encoding
print(f"Shape before encoding: {df.shape}")

# One-hot encode
df_encoded = pd.get_dummies(df, columns=cols_to_encode, prefix=cols_to_encode, drop_first=True)

print(f"✅ One-hot encoding complete!")
print(f"Shape after encoding: {df_encoded.shape}")
print(f"New columns added: {df_encoded.shape[1] - df.shape[1]}")

# Show new columns
new_cols = [col for col in df_encoded.columns if col not in df.columns]
print(f"\nNew columns ({len(new_cols)}):")
for col in new_cols[:15]:  # Show first 15
    print(f"  - {col}")
if len(new_cols) > 15:
    print(f"  ... and {len(new_cols) - 15} more")

# Update df
df = df_encoded
```

---

## 🔢 Step 4: Missing Value Handling

### 4.1 Analyze Missing Data

```python
print("=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

# Calculate missing percentage
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Missing_Percentage': missing_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)

print(f"Columns with missing values: {len(missing_df)}")
print("\n" + missing_df.to_string(index=False))

# Visualize
if len(missing_df) > 0:
    plt.figure(figsize=(10, 6))
    plt.barh(missing_df['Column'][:20], missing_df['Missing_Percentage'][:20], color='coral')
    plt.xlabel('Missing Percentage (%)')
    plt.title('Top 20 Columns with Missing Data', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
```

---

### 4.2 Strategic Missing Value Imputation

```python
print("\n" + "=" * 60)
print("MISSING VALUE IMPUTATION")
print("=" * 60)

# Strategy 1: Previous loan features - Fill with 0 (means no previous loan history)
prev_loan_cols = [col for col in df.columns if col.startswith('prev_')]

print(f"📊 Previous loan features: {len(prev_loan_cols)}")
for col in prev_loan_cols:
    missing_before = df[col].isnull().sum()
    df[col] = df[col].fillna(0)
    print(f"  {col}: {missing_before} → 0")

# Strategy 2: Geographic features - Fill with median
geo_cols = ['longitude_gps', 'latitude_gps']
for col in geo_cols:
    if col in df.columns:
        missing_before = df[col].isnull().sum()
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  {col}: {missing_before} filled with median ({median_val:.4f})")

# Strategy 3: Age - Fill with median
if df['age'].isnull().sum() > 0:
    median_age = df['age'].median()
    df['age'] = df['age'].fillna(median_age)
    print(f"  age: filled with median ({median_age:.1f})")

# Strategy 4: Create "missing" indicator for important features
# (Sometimes missingness itself is informative!)
important_cols_with_missing = ['bank_branch_clients', 'referredby']

for col in important_cols_with_missing:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
            print(f"  Created indicator: {col}_missing ({missing_count} cases)")

# Strategy 5: Drop columns with >50% missing (if any)
cols_to_drop = []
for col in df.columns:
    if df[col].isnull().sum() / len(df) > 0.5:
        cols_to_drop.append(col)

if cols_to_drop:
    print(f"\n🗑️ Dropping columns with >50% missing: {cols_to_drop}")
    df = df.drop(cols_to_drop, axis=1)

print(f"\n✅ Missing value handling complete!")
print(f"Remaining missing values: {df.isnull().sum().sum()}")
```

---

## 📐 Step 5: Ratio & Interaction Features

### 5.1 Loan-Related Ratios

```python
print("=" * 60)
print("RATIO FEATURES")
print("=" * 60)

# 1. Total due to loan amount ratio (interest rate proxy)
df['totaldue_to_loan_ratio'] = df['totaldue'] / df['loanamount']

# 2. Current loan vs average previous loan
df['current_vs_avg_prev_loan'] = df['loanamount'] / (df['prev_loan_mean'] + 1)  # +1 to avoid division by zero

# 3. Current loan vs max previous loan
df['current_vs_max_prev_loan'] = df['loanamount'] / (df['prev_loan_max'] + 1)

# 4. Loan to term days ratio (loan intensity)
df['loan_per_day'] = df['loanamount'] / df['termdays']

# 5. Previous loan frequency (loans per year)
# Assuming training period is ~2 years (2016-2017 from the data sample)
df['prev_loan_frequency_per_year'] = df['prev_loan_count'] / 2

print(f"✅ Ratio features created!")
print(f"\nNew features:")
ratio_features = ['totaldue_to_loan_ratio', 'current_vs_avg_prev_loan',
                 'current_vs_max_prev_loan', 'loan_per_day',
                 'prev_loan_frequency_per_year']
print(df[ratio_features].describe())
```

---

### 5.2 Behavioral Features

```python
print("\n" + "=" * 60)
print("BEHAVIORAL FEATURES")
print("=" * 60)

# 1. Percentage of previous loans paid early
df['pct_loans_paid_early'] = df['count_paid_early'] / (df['prev_loan_count'] + 1)

# 2. Percentage of previous loans paid late
df['pct_loans_paid_late'] = df['count_paid_late'] / (df['prev_loan_count'] + 1)

# 3. Percentage of previous loans closed
df['pct_loans_closed'] = df['count_loans_closed'] / (df['prev_loan_count'] + 1)

# 4. Average repayment behavior (negative = early, positive = late)
# Already have days_to_repay_mean from Day 2

# 5. Repayment consistency (lower std = more consistent)
df['repayment_consistency_score'] = 1 / (df['days_to_repay_std'] + 1)

# 6. Overall good borrower score (custom metric)
df['borrower_score'] = (
    df['pct_loans_paid_early'] * 2 +          # Double weight for early payment
    df['pct_loans_closed'] * 1.5 +            # Weight for closed loans
    -df['pct_loans_paid_late'] * 2 +          # Penalty for late payment
    df['repayment_consistency_score'] * 0.5   # Bonus for consistency
)

print(f"✅ Behavioral features created!")
print(f"\nBehavioral features:")
behavioral_features = ['pct_loans_paid_early', 'pct_loans_paid_late',
                       'pct_loans_closed', 'repayment_consistency_score',
                       'borrower_score']
print(df[behavioral_features].describe())

# Visualize borrower score vs target
plt.figure(figsize=(10, 5))
df.boxplot(column='borrower_score', by='good_bad_flag', figsize=(8, 5))
plt.suptitle('')
plt.title('Borrower Score by Loan Performance', fontsize=14, fontweight='bold')
plt.ylabel('Borrower Score')
plt.xlabel('Loan Status')
plt.tight_layout()
plt.show()
```

---

## 📊 Step 6: Feature Transformations

### 6.1 Log Transformations for Skewed Features

```python
print("=" * 60)
print("FEATURE TRANSFORMATIONS")
print("=" * 60)

# Identify skewed features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Calculate skewness
skewness = df[numeric_cols].skew().sort_values(ascending=False)

print("Top 10 most skewed features:")
print(skewness.head(10))

# Apply log transformation to highly skewed features (skew > 2)
highly_skewed = skewness[abs(skewness) > 2].index.tolist()

print(f"\n📊 Applying log transformation to {len(highly_skewed)} features")

for col in highly_skewed:
    # Only transform if values are positive
    if (df[col] >= 0).all():
        df[f'{col}_log'] = np.log1p(df[col])  # log1p = log(1 + x), handles zeros
        print(f"  ✅ {col}_log created")

# Visualize effect of transformation (example)
if 'loanamount_log' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df['loanamount'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title('Original: Loan Amount (Skewed)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Loan Amount')

    axes[1].hist(df['loanamount_log'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_title('Transformed: Log(Loan Amount) (More Normal)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Log(Loan Amount)')

    plt.tight_layout()
    plt.show()
```

---

## 🎯 Step 7: Feature Selection Prep

### 7.1 Check Final Dataset

```python
print("=" * 60)
print("FINAL FEATURE SET")
print("=" * 60)

print(f"✅ Final dataset shape: {df.shape}")
print(f"   Rows: {df.shape[0]}")
print(f"   Columns: {df.shape[1]}")

# Check for any remaining missing values
remaining_missing = df.isnull().sum().sum()
print(f"\n📊 Remaining missing values: {remaining_missing}")

if remaining_missing > 0:
    print("\n⚠️ Columns still with missing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0].sort_values(ascending=False))

# Data types
print(f"\n📋 Data types:")
print(df.dtypes.value_counts())

# Check for infinite values
inf_cols = []
for col in df.select_dtypes(include=[np.number]).columns:
    if np.isinf(df[col]).any():
        inf_cols.append(col)

if inf_cols:
    print(f"\n⚠️ Columns with infinite values: {inf_cols}")
    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # Fill with median
    for col in inf_cols:
        df[col] = df[col].fillna(df[col].median())
    print(f"✅ Infinite values replaced with median")
```

---

### 7.2 Feature Importance Preview (Correlation)

```python
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE PREVIEW")
print("=" * 60)

# Convert target to numeric
df['target_numeric'] = (df['good_bad_flag'] == 'Good').astype(int)

# Select numeric features
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [col for col in numeric_features if col not in ['target_numeric', 'customerid']]

# Calculate correlation with target
correlations = df[numeric_features + ['target_numeric']].corr()['target_numeric'].drop('target_numeric')
correlations = correlations.sort_values(ascending=False)

print(f"📊 Total numeric features: {len(numeric_features)}")
print(f"\n🔝 Top 20 positively correlated features:")
print(correlations.head(20))

print(f"\n🔻 Top 20 negatively correlated features:")
print(correlations.tail(20))

# Visualize top 15
top_15 = pd.concat([correlations.head(10), correlations.tail(5)]).sort_values()

plt.figure(figsize=(10, 8))
top_15.plot(kind='barh', color=['red' if x < 0 else 'green' for x in top_15])
plt.title('Top 15 Features by Correlation with Target', fontsize=14, fontweight='bold')
plt.xlabel('Correlation with Good Loans')
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()
```

---

## 💾 Step 8: Save Engineered Dataset

```python
print("=" * 60)
print("SAVING ENGINEERED DATASET")
print("=" * 60)

# Drop target_numeric (temporary)
df_to_save = df.drop('target_numeric', axis=1)

# Save
output_path = '../data/processed/featured_data.csv'
df_to_save.to_csv(output_path, index=False)

print(f"✅ Dataset saved to: {output_path}")
print(f"   Rows: {df_to_save.shape[0]}")
print(f"   Columns: {df_to_save.shape[1]}")

# Feature list for reference
feature_cols = [col for col in df_to_save.columns if col not in ['customerid', 'systemloanid', 'good_bad_flag']]

# Save feature names
with open('../data/processed/feature_names.txt', 'w') as f:
    for col in feature_cols:
        f.write(f"{col}\n")

print(f"\n✅ Feature names saved to: ../data/processed/feature_names.txt")
print(f"   Total features: {len(feature_cols)}")
```

---

## 📊 Step 9: Feature Engineering Summary Report

```python
print("=" * 60)
print("📝 FEATURE ENGINEERING SUMMARY REPORT")
print("=" * 60)

print("\n1️⃣ DATASET EVOLUTION:")
print(f"   Day 2 (after merge): {df_encoded.shape}")
print(f"   Day 3 (after engineering): {df.shape}")
print(f"   Features added: {df.shape[1] - df_encoded.shape[1]}")

print("\n2️⃣ FEATURE CATEGORIES CREATED:")
categories = {
    'Time-based': ['age', 'age_group', 'days_since_approval', 'approval_processing_days'],
    'Ratio': ['totaldue_to_loan_ratio', 'current_vs_avg_prev_loan', 'loan_per_day'],
    'Behavioral': ['pct_loans_paid_early', 'pct_loans_paid_late', 'borrower_score'],
    'Interaction': ['loan_per_year_age'],
    'Transformed': [col for col in df.columns if '_log' in col],
    'One-hot encoded': [col for col in df.columns if any(x in col for x in ['bank_account_type_', 'employment_status_', 'level_of_education_'])]
}

for category, features in categories.items():
    actual_features = [f for f in features if f in df.columns]
    print(f"   {category}: {len(actual_features)}")

print("\n3️⃣ MISSING VALUES:")
print(f"   Before: ~{missing_df['Missing_Count'].sum()} values")
print(f"   After: {df.isnull().sum().sum()} values")
print(f"   Reduction: {((missing_df['Missing_Count'].sum() - df.isnull().sum().sum()) / missing_df['Missing_Count'].sum() * 100):.1f}%")

print("\n4️⃣ TOP 10 PREDICTIVE FEATURES (by correlation):")
for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
    print(f"   {i}. {feature}: {corr:.4f}")

print("\n5️⃣ DATA QUALITY:")
print(f"   ✅ No missing values in target: {df['good_bad_flag'].isnull().sum() == 0}")
print(f"   ✅ No duplicate customerids: {df['customerid'].nunique() == len(df)}")
print(f"   ✅ No infinite values: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum() == 0}")

print("\n6️⃣ NEXT STEPS (Day 4):")
print("   - Deep EDA with new features")
print("   - Outlier detection and handling")
print("   - Final feature selection")
print("   - Train/test split preparation")
```

---

## ✅ Day 3 Checklist

- [ ] Created time-based features (age, recency, tenure)
- [ ] Encoded categorical variables (one-hot encoding)
- [ ] Handled missing values strategically
- [ ] Created ratio and interaction features
- [ ] Created behavioral features (repayment patterns)
- [ ] Applied transformations to skewed features
- [ ] Analyzed feature correlations with target
- [ ] Saved engineered dataset
- [ ] Generated feature engineering report
- [ ] Notebook saved as `notebooks/day3_feature_engineering.ipynb`

---

## 🎓 What You Learned Today

1. **Time-based Features:** Age, recency, processing time
2. **Categorical Encoding:** One-hot encoding, grouping high cardinality
3. **Missing Value Strategies:** Different strategies for different feature types
4. **Ratio Features:** Creating informative ratios from existing features
5. **Behavioral Features:** Quantifying customer behavior patterns
6. **Feature Transformations:** Log transformations for skewed data
7. **Feature Selection Prep:** Correlation analysis

---

## 🚀 Tomorrow (Day 4)

**Topic:** Deep EDA & Outlier Handling

You'll learn:
- Multivariate analysis with new features
- Outlier detection techniques
- Feature distributions by target class
- Feature selection methods
- Correlation matrix and multicollinearity

---

## 💡 Pro Tips

1. **Domain Knowledge Matters:** Best features often come from understanding the business
2. **Create Feature Families:** Group related features (ratios, time-based, behavioral)
3. **Document Feature Logic:** Future-you will thank present-you!
4. **Keep Original Features:** Sometimes transformed features aren't better
5. **Check for Data Leakage:** Ensure features don't contain future information

---

## 📚 Resources

- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Handling Missing Data](https://scikit-learn.org/stable/modules/impute.html)
- [Feature Engineering Techniques](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)

---

**🎉 Congratulations on completing Day 3!**

Your feature set is now powerful and ready for modeling! Tomorrow we'll dive deeper with EDA.
