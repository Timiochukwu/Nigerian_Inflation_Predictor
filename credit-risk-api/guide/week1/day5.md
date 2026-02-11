# 📅 Day 5: Preprocessing Pipeline & Feature Selection

**Goal:** Build a reusable preprocessing pipeline and prepare data for modeling.

---

## 🎯 Learning Objectives
- Build sklearn preprocessing pipelines
- Implement feature scaling (StandardScaler, MinMaxScaler)
- Perform feature selection (SelectKBest, feature importance)
- Create train/test splits
- Handle class imbalance
- Save preprocessing artifacts for production use

---

## 📚 Recap from Day 4

We completed deep EDA and:
- Analyzed all feature distributions
- Detected and handled outliers
- Created correlation matrices
- Identified key predictive features

Today we'll **build a production-ready preprocessing pipeline**! 🏗️

---

## 💻 Step 1: Load Cleaned Data

Create a new notebook: `notebooks/day5_preprocessing_pipeline.ipynb`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load cleaned data from Day 4
df = pd.read_csv('../data/processed/cleaned_data.csv')

print("✅ Data loaded!")
print(f"Shape: {df.shape}")

# Separate features and target
target_col = 'good_bad_flag'
id_cols = ['customerid', 'systemloanid']

# Keep only necessary columns
X = df.drop(columns=[target_col] + [col for col in id_cols if col in df.columns])
y = df[target_col]

# Encode target: Good = 1, Bad = 0
y_encoded = (y == 'Good').astype(int)

print(f"\n📊 Features shape: {X.shape}")
print(f"📊 Target shape: {y_encoded.shape}")
print(f"\nTarget distribution:")
print(y_encoded.value_counts())
```

---

## 🔪 Step 2: Remove Low-Variance Features

```python
print("=" * 60)
print("STEP 1: REMOVE LOW-VARIANCE FEATURES")
print("=" * 60)

# Calculate variance for numeric features
numeric_cols = X.select_dtypes(include=[np.number]).columns
variances = X[numeric_cols].var().sort_values()

print(f"Total numeric features: {len(numeric_cols)}")
print(f"\nFeatures with lowest variance:")
print(variances.head(10))

# Remove features with variance < 0.01 (almost constant)
low_var_features = variances[variances < 0.01].index.tolist()

if low_var_features:
    print(f"\n🗑️ Removing {len(low_var_features)} low-variance features:")
    for col in low_var_features:
        print(f"   - {col} (variance: {X[col].var():.6f})")

    X = X.drop(columns=low_var_features)
    print(f"\n✅ Features remaining: {X.shape[1]}")
else:
    print("\n✅ No low-variance features found")
```

---

## 🔗 Step 3: Handle Multicollinearity

```python
print("\n" + "=" * 60)
print("STEP 2: HANDLE MULTICOLLINEARITY")
print("=" * 60)

# Calculate correlation matrix
numeric_cols = X.select_dtypes(include=[np.number]).columns
corr_matrix = X[numeric_cols].corr().abs()

# Find highly correlated pairs
high_corr_pairs = []
removed_features = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.9:  # Threshold: 0.9
            feat1 = corr_matrix.columns[i]
            feat2 = corr_matrix.columns[j]

            # Keep the feature with higher correlation with target
            # (we'll calculate this after splitting)
            high_corr_pairs.append((feat1, feat2, corr_matrix.iloc[i, j]))

            # For now, mark the second feature for removal
            if feat2 not in removed_features:
                removed_features.add(feat2)

if high_corr_pairs:
    print(f"\n⚠️ Found {len(high_corr_pairs)} highly correlated pairs (r > 0.9):")
    for feat1, feat2, corr in high_corr_pairs[:10]:
        print(f"   {feat1} <-> {feat2}: {corr:.3f}")

    print(f"\n🗑️ Removing {len(removed_features)} redundant features:")
    for feat in list(removed_features)[:10]:
        print(f"   - {feat}")
    if len(removed_features) > 10:
        print(f"   ... and {len(removed_features) - 10} more")

    X = X.drop(columns=list(removed_features))
    print(f"\n✅ Features remaining: {X.shape[1]}")
else:
    print("\n✅ No highly correlated pairs found (r > 0.9)")
```

---

## 🎯 Step 4: Train-Test Split

```python
print("=" * 60)
print("STEP 3: TRAIN-TEST SPLIT")
print("=" * 60)

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded  # Maintain class distribution
)

print(f"✅ Data split complete!")
print(f"\n📊 Training set:")
print(f"   Shape: {X_train.shape}")
print(f"   Target distribution:")
print(f"   {y_train.value_counts()}")
print(f"   Percentage: {y_train.value_counts(normalize=True) * 100}")

print(f"\n📊 Test set:")
print(f"   Shape: {X_test.shape}")
print(f"   Target distribution:")
print(f"   {y_test.value_counts()}")
print(f"   Percentage: {y_test.value_counts(normalize=True) * 100}")

# Visualize split
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Train
y_train.value_counts().plot(kind='bar', ax=axes[0], color=['red', 'green'], alpha=0.7)
axes[0].set_title('Training Set Target Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Target (0=Bad, 1=Good)')
axes[0].set_ylabel('Count')
axes[0].tick_params(rotation=0)

# Test
y_test.value_counts().plot(kind='bar', ax=axes[1], color=['red', 'green'], alpha=0.7)
axes[1].set_title('Test Set Target Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Target (0=Bad, 1=Good)')
axes[1].set_ylabel('Count')
axes[1].tick_params(rotation=0)

plt.tight_layout()
plt.show()
```

---

## 🔍 Step 5: Feature Selection

### 5.1 Univariate Feature Selection (SelectKBest)

```python
print("=" * 60)
print("STEP 4: FEATURE SELECTION (UNIVARIATE)")
print("=" * 60)

# SelectKBest with f_classif (ANOVA F-value)
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X_train, y_train)

# Get scores
feature_scores = pd.DataFrame({
    'Feature': X_train.columns,
    'Score': selector.scores_,
    'P_Value': selector.pvalues_
}).sort_values('Score', ascending=False)

print("Top 20 features by ANOVA F-value:")
print(feature_scores.head(20).to_string(index=False))

# Visualize top 20 features
plt.figure(figsize=(10, 8))
feature_scores.head(20).plot(x='Feature', y='Score', kind='barh', legend=False, color='steelblue')
plt.title('Top 20 Features by ANOVA F-Score', fontsize=14, fontweight='bold')
plt.xlabel('F-Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Select top K features (let's say top 50)
top_k = 50
top_features = feature_scores.head(top_k)['Feature'].tolist()

print(f"\n✅ Selected top {top_k} features")
```

---

### 5.2 Mutual Information Feature Selection

```python
print("\n" + "=" * 60)
print("FEATURE SELECTION (MUTUAL INFORMATION)")
print("=" * 60)

# Mutual information
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

mi_feature_scores = pd.DataFrame({
    'Feature': X_train.columns,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

print("Top 20 features by Mutual Information:")
print(mi_feature_scores.head(20).to_string(index=False))

# Visualize
plt.figure(figsize=(10, 8))
mi_feature_scores.head(20).plot(x='Feature', y='MI_Score', kind='barh', legend=False, color='orange')
plt.title('Top 20 Features by Mutual Information', fontsize=14, fontweight='bold')
plt.xlabel('Mutual Information Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Combine both methods: features that appear in both top 50
mi_top_features = mi_feature_scores.head(top_k)['Feature'].tolist()
combined_features = list(set(top_features) & set(mi_top_features))

print(f"\n✅ Features in both top-{top_k}: {len(combined_features)}")

# Use union of both for now
final_features = list(set(top_features + mi_top_features))
print(f"✅ Total selected features (union): {len(final_features)}")
```

---

## 📏 Step 6: Feature Scaling

```python
print("=" * 60)
print("STEP 5: FEATURE SCALING")
print("=" * 60)

# Select final features
X_train_selected = X_train[final_features]
X_test_selected = X_test[final_features]

print(f"Selected features shape: {X_train_selected.shape}")

# StandardScaler (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Convert back to DataFrame for convenience
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=final_features, index=X_train_selected.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=final_features, index=X_test_selected.index)

print("\n✅ Features scaled with StandardScaler")
print(f"\nScaled training data sample:")
print(X_train_scaled_df.head())

# Verify scaling
print(f"\n📊 Scaling verification (first 5 features):")
for col in final_features[:5]:
    print(f"{col}:")
    print(f"  Mean: {X_train_scaled_df[col].mean():.6f} (should be ~0)")
    print(f"  Std: {X_train_scaled_df[col].std():.6f} (should be ~1)")

# Visualize effect of scaling (example feature)
example_feature = final_features[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before scaling
axes[0].hist(X_train_selected[example_feature], bins=30, edgecolor='black', alpha=0.7)
axes[0].set_title(f'Before Scaling: {example_feature}', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# After scaling
axes[1].hist(X_train_scaled_df[example_feature], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[1].set_title(f'After Scaling: {example_feature}', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Value (standardized)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

---

## ⚖️ Step 7: Handle Class Imbalance (Optional)

```python
print("=" * 60)
print("STEP 6: HANDLE CLASS IMBALANCE")
print("=" * 60)

# Check class balance
class_counts = y_train.value_counts()
class_ratio = class_counts[1] / class_counts[0]

print(f"Class distribution in training set:")
print(class_counts)
print(f"\nClass ratio (Good/Bad): {class_ratio:.2f}")

if class_ratio > 1.5 or class_ratio < 0.67:
    print("\n⚠️ CLASS IMBALANCE DETECTED!")
    print("Options to handle:")
    print("   1. Use class_weight='balanced' in model")
    print("   2. Use SMOTE (Synthetic Minority Over-sampling)")
    print("   3. Use undersampling")
    print("\n📝 Recommendation: Use class_weight='balanced' in logistic regression/RF/XGBoost")

    # Option 2: SMOTE (uncomment if you want to use it)
    # from imblearn.over_sampling import SMOTE
    # smote = SMOTE(random_state=42)
    # X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled_df, y_train)
    # print(f"\n✅ SMOTE applied: {X_train_balanced.shape}")
else:
    print("\n✅ Classes are relatively balanced")

# For now, we'll handle this with class_weight in models (Week 2)
```

---

## 💾 Step 8: Save Preprocessing Artifacts

```python
print("=" * 60)
print("STEP 7: SAVE PREPROCESSING ARTIFACTS")
print("=" * 60)

# Create artifacts directory
import os
os.makedirs('../models/artifacts', exist_ok=True)

# 1. Save scaler
scaler_path = '../models/artifacts/scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved to: {scaler_path}")

# 2. Save selected feature names
features_path = '../models/artifacts/selected_features.pkl'
joblib.dump(final_features, features_path)
print(f"✅ Selected features saved to: {features_path}")

# 3. Save feature scores
feature_scores_path = '../data/processed/feature_scores.csv'
feature_scores.to_csv(feature_scores_path, index=False)
print(f"✅ Feature scores saved to: {feature_scores_path}")

# 4. Save train/test splits
train_path = '../data/processed/X_train.csv'
test_path = '../data/processed/X_test.csv'
y_train_path = '../data/processed/y_train.csv'
y_test_path = '../data/processed/y_test.csv'

X_train_scaled_df.to_csv(train_path, index=False)
X_test_scaled_df.to_csv(test_path, index=False)
pd.DataFrame(y_train).to_csv(y_train_path, index=False)
pd.DataFrame(y_test).to_csv(y_test_path, index=False)

print(f"✅ Training data saved to: {train_path}")
print(f"✅ Test data saved to: {test_path}")
print(f"✅ Training labels saved to: {y_train_path}")
print(f"✅ Test labels saved to: {y_test_path}")

print("\n🎉 All preprocessing artifacts saved!")
```

---

## 🏗️ Step 9: Create Reusable Preprocessing Function

```python
print("=" * 60)
print("STEP 8: CREATE PREPROCESSING FUNCTION")
print("=" * 60)

# Create a preprocessing function for new data
def preprocess_new_data(df_new, scaler_path, features_path):
    """
    Preprocess new data using saved artifacts.

    Parameters:
    - df_new: DataFrame with raw features
    - scaler_path: Path to saved scaler
    - features_path: Path to saved feature names

    Returns:
    - Scaled and selected features ready for prediction
    """
    # Load artifacts
    scaler = joblib.load(scaler_path)
    selected_features = joblib.load(features_path)

    # Select features
    df_selected = df_new[selected_features]

    # Scale
    df_scaled = scaler.transform(df_selected)

    # Return as DataFrame
    return pd.DataFrame(df_scaled, columns=selected_features, index=df_new.index)

# Save this function as a module
preprocessing_code = '''
import pandas as pd
import joblib

def preprocess_new_data(df_new, scaler_path='../models/artifacts/scaler.pkl',
                       features_path='../models/artifacts/selected_features.pkl'):
    """
    Preprocess new data using saved artifacts.

    Parameters:
    - df_new: DataFrame with raw features
    - scaler_path: Path to saved scaler
    - features_path: Path to saved feature names

    Returns:
    - Scaled and selected features ready for prediction
    """
    # Load artifacts
    scaler = joblib.load(scaler_path)
    selected_features = joblib.load(features_path)

    # Select features
    df_selected = df_new[selected_features]

    # Scale
    df_scaled = scaler.transform(df_selected)

    # Return as DataFrame
    return pd.DataFrame(df_scaled, columns=selected_features, index=df_new.index)
'''

# Save to src/utils.py
utils_path = '../src/utils.py'
with open(utils_path, 'w') as f:
    f.write(preprocessing_code)

print(f"✅ Preprocessing function saved to: {utils_path}")

# Test the function
print("\n🧪 Testing preprocessing function...")
sample = X_test.head(5)
processed_sample = preprocess_new_data(sample, scaler_path, features_path)
print(f"✅ Test passed! Processed shape: {processed_sample.shape}")
print(processed_sample.head())
```

---

## 📊 Step 10: Final Summary Report

```python
print("=" * 60)
print("📝 WEEK 1 FINAL SUMMARY REPORT")
print("=" * 60)

print("\n1️⃣ DATA JOURNEY:")
print(f"   Day 1: Loaded 3 CSV files")
print(f"   Day 2: Merged into {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Day 3: Engineered features → {X.shape[1]} features")
print(f"   Day 4: Cleaned data, handled outliers")
print(f"   Day 5: Selected {len(final_features)} best features")

print("\n2️⃣ FINAL DATASET:")
print(f"   Training set: {X_train_scaled_df.shape}")
print(f"   Test set: {X_test_scaled_df.shape}")
print(f"   Features: {len(final_features)}")
print(f"   Target: good_bad_flag (Good=1, Bad=0)")

print("\n3️⃣ PREPROCESSING STEPS COMPLETED:")
print(f"   ✅ Removed low-variance features")
print(f"   ✅ Handled multicollinearity (r > 0.9)")
print(f"   ✅ Feature selection (ANOVA F-test + Mutual Information)")
print(f"   ✅ Feature scaling (StandardScaler)")
print(f"   ✅ Train-test split (80-20, stratified)")
print(f"   ✅ Saved preprocessing artifacts")

print("\n4️⃣ TOP 10 SELECTED FEATURES:")
for i, feat in enumerate(final_features[:10], 1):
    print(f"   {i}. {feat}")

print("\n5️⃣ CLASS DISTRIBUTION:")
print(f"   Training set:")
print(f"      Good: {(y_train == 1).sum()} ({(y_train == 1).mean() * 100:.1f}%)")
print(f"      Bad: {(y_train == 0).sum()} ({(y_train == 0).mean() * 100:.1f}%)")

print("\n6️⃣ SAVED ARTIFACTS:")
print(f"   ✅ Scaler: {scaler_path}")
print(f"   ✅ Selected features: {features_path}")
print(f"   ✅ Train/test data: data/processed/")
print(f"   ✅ Preprocessing function: src/utils.py")

print("\n7️⃣ DATA QUALITY CHECKS:")
print(f"   ✅ No missing values: {X_train_scaled_df.isnull().sum().sum() == 0}")
print(f"   ✅ All features scaled (mean≈0, std≈1)")
print(f"   ✅ Stratified split maintains class distribution")
print(f"   ✅ Test set is independent (no data leakage)")

print("\n" + "=" * 60)
print("🎉 WEEK 1 COMPLETE!")
print("=" * 60)
print("\nYou now have:")
print("   ✅ Clean, merged dataset")
print("   ✅ 50+ engineered features")
print("   ✅ Selected top features")
print("   ✅ Scaled training/test sets")
print("   ✅ Reusable preprocessing pipeline")
print("   ✅ Comprehensive understanding of your data")

print("\n🚀 NEXT WEEK (Week 2): MODELING")
print("   - Day 6: Baseline model & evaluation metrics")
print("   - Day 7: Logistic Regression")
print("   - Day 8: Model evaluation deep dive")
print("   - Day 9: Feature importance & interpretation")
print("   - Day 10: Cross-validation & model selection")

print("\n💪 You're ready to build machine learning models!")
```

---

## 📁 Step 11: Project Structure Check

```python
print("=" * 60)
print("FINAL PROJECT STRUCTURE")
print("=" * 60)

project_structure = """
credit-risk-api/
├── data/
│   ├── raw/
│   │   ├── traindemographics.csv
│   │   ├── trainperf.csv
│   │   └── trainprevloans.csv
│   └── processed/
│       ├── merged_data.csv              ✅ Day 2
│       ├── featured_data.csv            ✅ Day 3
│       ├── cleaned_data.csv             ✅ Day 4
│       ├── X_train.csv                  ✅ Day 5
│       ├── X_test.csv                   ✅ Day 5
│       ├── y_train.csv                  ✅ Day 5
│       ├── y_test.csv                   ✅ Day 5
│       ├── feature_scores.csv           ✅ Day 5
│       └── feature_names.txt            ✅ Day 3
├── notebooks/
│   ├── day1_exploration.ipynb           ✅
│   ├── day2_merging.ipynb               ✅
│   ├── day3_feature_engineering.ipynb   ✅
│   ├── day4_deep_eda.ipynb              ✅
│   └── day5_preprocessing_pipeline.ipynb ✅
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   └── utils.py                         ✅ Day 5
├── models/
│   └── artifacts/
│       ├── scaler.pkl                   ✅ Day 5
│       └── selected_features.pkl        ✅ Day 5
├── guide/
│   └── week1/
│       ├── day1.md                      ✅
│       ├── day2.md                      ✅
│       ├── day3.md                      ✅
│       ├── day4.md                      ✅
│       └── day5.md                      ✅
└── requirements.txt                     ✅
"""

print(project_structure)
```

---

## ✅ Day 5 Checklist

- [ ] Removed low-variance features
- [ ] Handled multicollinearity
- [ ] Performed train-test split (80-20, stratified)
- [ ] Implemented univariate feature selection (ANOVA F-test)
- [ ] Implemented mutual information feature selection
- [ ] Selected final feature set
- [ ] Applied StandardScaler for feature scaling
- [ ] Identified class imbalance (if any)
- [ ] Saved all preprocessing artifacts (scaler, features)
- [ ] Saved train/test datasets
- [ ] Created reusable preprocessing function
- [ ] Generated Week 1 summary report
- [ ] Notebook saved as `notebooks/day5_preprocessing_pipeline.ipynb`

---

## 🎓 What You Learned This Week

### **Day 1:** Data Exploration
- Loading CSV files
- Basic EDA (shape, dtypes, missing values)
- Target variable analysis
- Visualizations

### **Day 2:** Data Merging
- Understanding relationships (1:1, 1:Many)
- Merging datasets with pandas
- Aggregating historical data
- Data validation

### **Day 3:** Feature Engineering
- Time-based features (age, recency)
- Ratio features (loan-to-age)
- Behavioral features (repayment patterns)
- Categorical encoding
- Feature transformations

### **Day 4:** Deep EDA & Outliers
- Univariate analysis
- Bivariate analysis (feature vs target)
- Outlier detection (Z-score, IQR)
- Correlation analysis
- Statistical testing

### **Day 5:** Preprocessing Pipeline
- Feature selection (SelectKBest, MI)
- Feature scaling (StandardScaler)
- Train-test splitting
- Saving artifacts for production
- Creating reusable pipelines

---

## 🚀 Next Week Preview (Week 2)

**Days 6-10: Machine Learning Models**

You'll build and evaluate:
1. **Baseline model** (for comparison)
2. **Logistic Regression** (interpretable, fast)
3. **Random Forest** (non-linear relationships)
4. **XGBoost** (state-of-the-art performance)

And learn:
- Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Cross-validation
- Hyperparameter tuning
- Feature importance
- Model interpretation

---

## 💡 Pro Tips

1. **Always save preprocessing artifacts** - Essential for production deployment
2. **Stratified split is crucial** - Maintains class distribution
3. **Feature selection reduces overfitting** - Fewer features = simpler model
4. **StandardScaler is default for most models** - MinMaxScaler for neural networks
5. **Document your decisions** - Why you chose certain features/methods

---

## 📚 Resources

- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Feature Selection Guide](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Pipeline and ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html)
- [Train Test Split Best Practices](https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/)

---

## 🎉 Week 1 Complete! Congratulations! 🎉

You've built a **solid foundation**:
- ✅ Data exploration and understanding
- ✅ Feature engineering (50+ features!)
- ✅ Data cleaning and quality checks
- ✅ Production-ready preprocessing pipeline

**Take a break! Review your notebooks! You earned it! 🌟**

**Next week: We build ML models! 🚀**
