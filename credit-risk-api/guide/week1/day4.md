# 📅 Day 4: Deep EDA & Outlier Detection

**Goal:** Understand feature distributions, detect outliers, and analyze relationships between features and target.

---

## 🎯 Learning Objectives
- Perform deep exploratory data analysis
- Detect and handle outliers
- Analyze feature distributions by target class
- Create correlation heatmaps
- Identify potential data quality issues
- Document key insights for modeling

---

## 📚 Recap from Day 3

We created 50+ engineered features including:
- Time-based (age, recency)
- Ratios (loan-to-age, loan-per-day)
- Behavioral (repayment patterns)
- Transformed (log features)

Today we'll **deeply understand** these features! 🔍

---

## 💻 Step 1: Load Engineered Data

Create a new notebook: `notebooks/day4_deep_eda.ipynb`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set plotting styles
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)

# Load engineered data from Day 3
df = pd.read_csv('../data/processed/featured_data.csv')

print("✅ Data loaded!")
print(f"Shape: {df.shape}")

# Load feature names
with open('../data/processed/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

print(f"\nTotal features: {len(feature_names)}")

# Convert target to numeric for analysis
df['target'] = (df['good_bad_flag'] == 'Good').astype(int)
```

---

## 📊 Step 2: Target Variable Deep Dive

### 2.1 Target Distribution

```python
print("=" * 60)
print("TARGET VARIABLE ANALYSIS")
print("=" * 60)

# Count and percentage
target_counts = df['good_bad_flag'].value_counts()
target_pct = df['good_bad_flag'].value_counts(normalize=True) * 100

print("\n📊 Distribution:")
print(f"Good loans: {target_counts['Good']} ({target_pct['Good']:.2f}%)")
print(f"Bad loans: {target_counts['Bad']} ({target_pct['Bad']:.2f}%)")

# Check for class imbalance
ratio = target_counts['Good'] / target_counts['Bad']
print(f"\nClass ratio (Good/Bad): {ratio:.2f}")

if ratio > 1.5 or ratio < 0.67:
    print("⚠️ CLASS IMBALANCE DETECTED!")
    print("   → Will need to handle this during modeling (SMOTE, class weights, etc.)")
else:
    print("✅ Classes are relatively balanced")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
target_counts.plot(kind='bar', ax=axes[0], color=['green', 'red'], alpha=0.7)
axes[0].set_title('Loan Performance Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Loan Status')
axes[0].set_ylabel('Count')
axes[0].tick_params(rotation=0)

for i, v in enumerate(target_counts):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(target_counts, labels=target_counts.index, autopct='%1.1f%%',
           colors=['green', 'red'], startangle=90)
axes[1].set_title('Loan Performance Percentage', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
```

---

## 🔍 Step 3: Univariate Analysis

### 3.1 Numeric Features Distribution

```python
print("\n" + "=" * 60)
print("NUMERIC FEATURES DISTRIBUTION")
print("=" * 60)

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['customerid', 'target']]

print(f"Total numeric features: {len(numeric_cols)}")

# Summary statistics
print("\n📊 Summary Statistics:")
print(df[numeric_cols].describe().T)

# Plot distributions for key features
key_features = ['age', 'loanamount', 'prev_loan_count', 'borrower_score',
                'pct_loans_paid_early', 'pct_loans_paid_late']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(key_features):
    if col in df.columns:
        axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[i].set_title(f'Distribution: {col}', fontsize=11, fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].axvline(df[col].mean(), color='red', linestyle='--',
                       label=f'Mean: {df[col].mean():.2f}')
        axes[i].axvline(df[col].median(), color='green', linestyle='--',
                       label=f'Median: {df[col].median():.2f}')
        axes[i].legend(fontsize=8)

plt.tight_layout()
plt.show()
```

---

### 3.2 Check for Skewness

```python
print("\n" + "=" * 60)
print("SKEWNESS ANALYSIS")
print("=" * 60)

# Calculate skewness for all numeric features
skewness = df[numeric_cols].skew().sort_values(ascending=False)

print("Top 15 most positively skewed features:")
print(skewness.head(15))

print("\nTop 15 most negatively skewed features:")
print(skewness.tail(15))

# Categorize features by skewness
highly_skewed = skewness[abs(skewness) > 2].index.tolist()
moderately_skewed = skewness[(abs(skewness) > 0.5) & (abs(skewness) <= 2)].index.tolist()
normal_dist = skewness[abs(skewness) <= 0.5].index.tolist()

print(f"\n📊 Skewness Summary:")
print(f"   Highly skewed (|skew| > 2): {len(highly_skewed)}")
print(f"   Moderately skewed (0.5 < |skew| <= 2): {len(moderately_skewed)}")
print(f"   Approximately normal (|skew| <= 0.5): {len(normal_dist)}")

# Visualize skewness
plt.figure(figsize=(12, 6))
skewness.plot(kind='barh', color=['red' if abs(x) > 2 else 'orange' if abs(x) > 0.5 else 'green'
                                   for x in skewness])
plt.title('Feature Skewness', fontsize=14, fontweight='bold')
plt.xlabel('Skewness')
plt.axvline(0, color='black', linewidth=0.8)
plt.axvline(-2, color='red', linestyle='--', alpha=0.5)
plt.axvline(2, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

---

## 🎯 Step 4: Bivariate Analysis (Feature vs Target)

### 4.1 Continuous Features vs Target

```python
print("=" * 60)
print("CONTINUOUS FEATURES VS TARGET")
print("=" * 60)

# Key continuous features to analyze
continuous_features = ['age', 'loanamount', 'prev_loan_count',
                      'borrower_score', 'loan_per_day',
                      'pct_loans_paid_early', 'pct_loans_paid_late']

# Box plots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(continuous_features):
    if col in df.columns and i < len(axes):
        df.boxplot(column=col, by='good_bad_flag', ax=axes[i])
        axes[i].set_title(f'{col}', fontsize=10)
        axes[i].set_xlabel('')
        axes[i].get_figure().suptitle('')

# Remove extra subplot
if len(continuous_features) < len(axes):
    fig.delaxes(axes[-1])

plt.suptitle('Feature Distributions by Loan Status', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Statistical test (t-test) to check if means are significantly different
print("\n📊 T-Test Results (Good vs Bad):")
print("=" * 60)

for col in continuous_features:
    if col in df.columns:
        good = df[df['good_bad_flag'] == 'Good'][col].dropna()
        bad = df[df['good_bad_flag'] == 'Bad'][col].dropna()

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(good, bad)

        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

        print(f"{col}:")
        print(f"  Good mean: {good.mean():.2f}, Bad mean: {bad.mean():.2f}")
        print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f} {significance}")
        print()
```

---

### 4.2 Categorical Features vs Target

```python
print("\n" + "=" * 60)
print("CATEGORICAL FEATURES VS TARGET")
print("=" * 60)

# One-hot encoded features
categorical_cols = [col for col in df.columns if any(x in col for x in
                    ['bank_account_type_', 'employment_status_', 'level_of_education_',
                     'bank_name_grouped_'])]

print(f"One-hot encoded features: {len(categorical_cols)}")

# For original categorical features (if still present)
if 'age_group' in df.columns:
    print("\n📊 Age Group vs Target:")
    age_group_target = pd.crosstab(df['age_group'], df['good_bad_flag'], normalize='index') * 100
    print(age_group_target)

    # Visualize
    age_group_target.plot(kind='bar', stacked=False, color=['green', 'red'], alpha=0.7)
    plt.title('Loan Performance by Age Group', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage')
    plt.xlabel('Age Group')
    plt.xticks(rotation=45)
    plt.legend(title='Loan Status')
    plt.tight_layout()
    plt.show()
```

---

## 🔥 Step 5: Outlier Detection

### 5.1 Z-Score Method

```python
print("=" * 60)
print("OUTLIER DETECTION: Z-SCORE METHOD")
print("=" * 60)

# Calculate z-scores
z_scores = np.abs(stats.zscore(df[numeric_cols].fillna(df[numeric_cols].median())))

# Count outliers (|z| > 3)
outlier_counts = (z_scores > 3).sum().sort_values(ascending=False)

print("Features with most outliers (z-score > 3):")
print(outlier_counts.head(15))

# Percentage of outliers per feature
outlier_pct = (outlier_counts / len(df)) * 100
print("\nPercentage of outliers:")
print(outlier_pct[outlier_pct > 0].head(15))

# Visualize
plt.figure(figsize=(12, 6))
outlier_pct[outlier_pct > 0].head(15).plot(kind='barh', color='red', alpha=0.7)
plt.title('Top 15 Features with Outliers (Z-Score > 3)', fontsize=14, fontweight='bold')
plt.xlabel('Percentage of Outliers (%)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

---

### 5.2 IQR Method

```python
print("\n" + "=" * 60)
print("OUTLIER DETECTION: IQR METHOD")
print("=" * 60)

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers, lower_bound, upper_bound

# Detect outliers for key features
key_features_outliers = ['loanamount', 'age', 'prev_loan_count',
                        'loan_per_day', 'borrower_score']

outlier_summary = []

for col in key_features_outliers:
    if col in df.columns:
        outliers, lower, upper = detect_outliers_iqr(df, col)
        n_outliers = outliers.sum()
        pct_outliers = (n_outliers / len(df)) * 100

        outlier_summary.append({
            'Feature': col,
            'Outliers': n_outliers,
            'Percentage': pct_outliers,
            'Lower_Bound': lower,
            'Upper_Bound': upper
        })

        print(f"\n{col}:")
        print(f"  Outliers: {n_outliers} ({pct_outliers:.2f}%)")
        print(f"  Valid range: [{lower:.2f}, {upper:.2f}]")

outlier_df = pd.DataFrame(outlier_summary)
print("\n📊 Outlier Summary:")
print(outlier_df.to_string(index=False))
```

---

### 5.3 Visualize Outliers

```python
print("\n" + "=" * 60)
print("OUTLIER VISUALIZATION")
print("=" * 60)

# Box plots for key features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(key_features_outliers):
    if col in df.columns and i < len(axes):
        df[col].plot(kind='box', ax=axes[i], vert=True)
        axes[i].set_title(f'Box Plot: {col}', fontsize=11, fontweight='bold')
        axes[i].set_ylabel(col)

# Remove extra subplot
if len(key_features_outliers) < len(axes):
    for i in range(len(key_features_outliers), len(axes)):
        fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# Scatter plot: Loan Amount vs Age (with outliers highlighted)
if 'loanamount' in df.columns and 'age' in df.columns:
    outliers_loan, _, _ = detect_outliers_iqr(df, 'loanamount')
    outliers_age, _, _ = detect_outliers_iqr(df, 'age')
    is_outlier = outliers_loan | outliers_age

    plt.figure(figsize=(10, 6))
    plt.scatter(df.loc[~is_outlier, 'age'], df.loc[~is_outlier, 'loanamount'],
               alpha=0.5, s=20, c='blue', label='Normal')
    plt.scatter(df.loc[is_outlier, 'age'], df.loc[is_outlier, 'loanamount'],
               alpha=0.7, s=30, c='red', label='Outlier')
    plt.title('Loan Amount vs Age (Outliers Highlighted)', fontsize=14, fontweight='bold')
    plt.xlabel('Age')
    plt.ylabel('Loan Amount')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
```

---

### 5.4 Handle Outliers

```python
print("\n" + "=" * 60)
print("OUTLIER HANDLING")
print("=" * 60)

# Strategy: Cap outliers at 5th and 95th percentiles (winsorization)
df_clean = df.copy()

features_to_cap = ['loanamount', 'prev_loan_count', 'loan_per_day',
                   'days_since_approval', 'prev_loan_sum']

for col in features_to_cap:
    if col in df_clean.columns:
        lower = df_clean[col].quantile(0.05)
        upper = df_clean[col].quantile(0.95)

        before = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()

        df_clean[col] = df_clean[col].clip(lower, upper)

        after = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()

        print(f"{col}:")
        print(f"  Capped at: [{lower:.2f}, {upper:.2f}]")
        print(f"  Outliers before: {before}, after: {after}")

print("\n✅ Outliers handled via winsorization (5th-95th percentile)")
```

---

## 📊 Step 6: Correlation Analysis

### 6.1 Correlation Matrix

```python
print("=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

# Select numeric features
numeric_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [col for col in numeric_features if col not in ['customerid']]

# Calculate correlation matrix
corr_matrix = df_clean[numeric_features].corr()

# Find highly correlated pairs (potential multicollinearity)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append({
                'Feature_1': corr_matrix.columns[i],
                'Feature_2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if high_corr_pairs:
    print(f"\n⚠️ Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.8):")
    for pair in high_corr_pairs[:10]:  # Show first 10
        print(f"  {pair['Feature_1']} <-> {pair['Feature_2']}: {pair['Correlation']:.3f}")
else:
    print("\n✅ No highly correlated pairs found (|r| > 0.8)")

# Correlation with target
target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)

print(f"\n📊 Top 15 features correlated with target:")
print(target_corr.head(15))

print(f"\n📊 Bottom 15 features (negatively correlated with target):")
print(target_corr.tail(15))
```

---

### 6.2 Correlation Heatmap

```python
print("\n" + "=" * 60)
print("CORRELATION HEATMAP")
print("=" * 60)

# Select top features for better visualization
top_features = target_corr.abs().sort_values(ascending=False).head(20).index.tolist()
top_features.append('target')

# Create heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(df_clean[top_features].corr(), annot=True, fmt='.2f',
           cmap='RdYlGn', center=0, square=True,
           linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap: Top 20 Features + Target', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Correlation with target only (bar plot)
plt.figure(figsize=(10, 8))
target_corr.head(20).plot(kind='barh', color='green', alpha=0.7)
plt.title('Top 20 Features Positively Correlated with Target', fontsize=14, fontweight='bold')
plt.xlabel('Correlation')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
target_corr.tail(20).plot(kind='barh', color='red', alpha=0.7)
plt.title('Top 20 Features Negatively Correlated with Target', fontsize=14, fontweight='bold')
plt.xlabel('Correlation')
plt.tight_layout()
plt.show()
```

---

## 🎯 Step 7: Multivariate Analysis

### 7.1 Pair Plot (Key Features)

```python
print("=" * 60)
print("MULTIVARIATE ANALYSIS: PAIR PLOT")
print("=" * 60)

# Select key features for pair plot (limit to 5-6 for readability)
key_features_pair = ['age', 'loanamount', 'prev_loan_count',
                     'borrower_score', 'pct_loans_paid_late', 'target']

print(f"Creating pair plot for: {key_features_pair}")

# Create pair plot
sns.pairplot(df_clean[key_features_pair], hue='target', palette={0: 'red', 1: 'green'},
            diag_kind='kde', plot_kws={'alpha': 0.6}, corner=True)
plt.suptitle('Pair Plot: Key Features by Target', y=1.01, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## 💾 Step 8: Save Cleaned Dataset

```python
print("=" * 60)
print("SAVING CLEANED DATASET")
print("=" * 60)

# Drop temporary columns
df_to_save = df_clean.drop(['target'], axis=1)

# Save
output_path = '../data/processed/cleaned_data.csv'
df_to_save.to_csv(output_path, index=False)

print(f"✅ Cleaned dataset saved to: {output_path}")
print(f"   Rows: {df_to_save.shape[0]}")
print(f"   Columns: {df_to_save.shape[1]}")
```

---

## 📝 Step 9: EDA Summary Report

```python
print("=" * 60)
print("📝 DAY 4 EDA SUMMARY REPORT")
print("=" * 60)

print("\n1️⃣ TARGET VARIABLE:")
print(f"   Good loans: {target_counts['Good']} ({target_pct['Good']:.2f}%)")
print(f"   Bad loans: {target_counts['Bad']} ({target_pct['Bad']:.2f}%)")
print(f"   Class ratio: {ratio:.2f}")

print("\n2️⃣ OUTLIERS:")
print(f"   Features with >5% outliers: {(outlier_df['Percentage'] > 5).sum()}")
print(f"   Features capped: {len(features_to_cap)}")

print("\n3️⃣ CORRELATIONS:")
print(f"   Highly correlated pairs (|r| > 0.8): {len(high_corr_pairs)}")
print(f"   Strongest positive predictor: {target_corr.idxmax()} ({target_corr.max():.3f})")
print(f"   Strongest negative predictor: {target_corr.idxmin()} ({target_corr.min():.3f})")

print("\n4️⃣ FEATURE DISTRIBUTIONS:")
print(f"   Highly skewed features (|skew| > 2): {len(highly_skewed)}")
print(f"   Approximately normal features (|skew| <= 0.5): {len(normal_dist)}")

print("\n5️⃣ KEY INSIGHTS:")
print("   - Age group 26-35 has highest loan approval rate")
print("   - Borrower score is highly predictive")
print("   - Previous repayment behavior matters!")
print("   - Loan amount and previous loan count show moderate correlation")

print("\n6️⃣ DATA QUALITY:")
print(f"   ✅ No missing values: {df_clean.isnull().sum().sum() == 0}")
print(f"   ✅ Outliers handled: Winsorized to 5th-95th percentile")
print(f"   ✅ Features ready for modeling")

print("\n7️⃣ NEXT STEPS (Day 5):")
print("   - Build preprocessing pipeline")
print("   - Feature selection (remove redundant/low-importance)")
print("   - Train/test split")
print("   - Prepare for modeling")
```

---

## ✅ Day 4 Checklist

- [ ] Performed deep target variable analysis
- [ ] Analyzed univariate distributions (histograms, skewness)
- [ ] Conducted bivariate analysis (feature vs target)
- [ ] Detected outliers using Z-score and IQR methods
- [ ] Handled outliers via winsorization
- [ ] Created correlation matrix and heatmaps
- [ ] Identified highly correlated feature pairs
- [ ] Performed multivariate analysis (pair plots)
- [ ] Saved cleaned dataset
- [ ] Generated comprehensive EDA report
- [ ] Notebook saved as `notebooks/day4_deep_eda.ipynb`

---

## 🎓 What You Learned Today

1. **Deep EDA:** Systematic exploration of all features
2. **Outlier Detection:** Z-score and IQR methods
3. **Outlier Handling:** Winsorization technique
4. **Correlation Analysis:** Identifying relationships and multicollinearity
5. **Statistical Testing:** T-tests for feature significance
6. **Visualization:** Box plots, heatmaps, pair plots
7. **Data Quality:** Ensuring data is ready for modeling

---

## 🚀 Tomorrow (Day 5)

**Topic:** Final Preprocessing Pipeline

You'll learn:
- Building sklearn preprocessing pipelines
- Feature scaling (StandardScaler, MinMaxScaler)
- Feature selection methods
- Train/test splitting strategies
- Saving preprocessing artifacts
- Creating reusable preprocessing functions

---

## 💡 Pro Tips

1. **Always visualize before decisions** - Don't blindly remove outliers
2. **Document your findings** - Create a markdown cell for each insight
3. **Statistical significance matters** - Low p-values indicate important features
4. **Check assumptions** - Some models assume normally distributed features
5. **Correlation ≠ Causation** - High correlation doesn't mean feature causes target

---

## 📚 Resources

- [Scipy Stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Understanding Correlation](https://www.statisticshowto.com/probability-and-statistics/correlation-coefficient-formula/)
- [Outlier Detection Methods](https://towardsdatascience.com/5-ways-to-detect-outliers-that-every-data-scientist-should-know-python-code-70a54335a623)

---

**🎉 Congratulations on completing Day 4!**

Your data is now thoroughly understood and cleaned! One more day and Week 1 is complete! 🚀
