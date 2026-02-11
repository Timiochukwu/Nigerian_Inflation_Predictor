# 📅 Day 1: Project Setup & Data Exploration

**Goal:** Set up your environment, load data, and understand what you're working with.

---

## 🎯 Learning Objectives
- Set up Python environment with required libraries
- Load CSV files into pandas DataFrames
- Perform initial data exploration
- Understand data shapes, types, and basic statistics

---

## 📦 Step 1: Environment Setup

### Install Required Libraries

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Create a requirements.txt

```txt
pandas==2.1.0
numpy==1.25.0
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
xgboost==2.0.0
fastapi==0.103.0
uvicorn==0.23.0
pydantic==2.3.0
```

Save this as `credit-risk-api/requirements.txt` and install:

```bash
pip install -r requirements.txt
```

---

## 📂 Step 2: Project Structure

Create this folder structure:

```
credit-risk-api/
├── data/
│   ├── raw/                    # Original CSV files
│   │   ├── traindemographics.csv
│   │   ├── trainperf.csv
│   │   └── trainprevloans.csv
│   └── processed/              # Cleaned, merged data
├── notebooks/
│   └── day1_exploration.ipynb  # Today's work
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading functions
│   └── utils.py                # Helper functions
├── models/                      # Saved models (later)
├── guide/                       # This guide!
└── requirements.txt
```

---

## 💻 Step 3: Load the Data

Create a new Jupyter notebook: `notebooks/day1_exploration.ipynb`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("✅ Libraries loaded successfully!")
```

### Load CSV Files

```python
# Load the three datasets
demographics = pd.read_csv('../data/raw/traindemographics.csv')
performance = pd.read_csv('../data/raw/trainperf.csv')
prev_loans = pd.read_csv('../data/raw/trainprevloans.csv')

print("✅ Data loaded successfully!")
print(f"Demographics shape: {demographics.shape}")
print(f"Performance shape: {performance.shape}")
print(f"Previous Loans shape: {prev_loans.shape}")
```

**Expected Output:**
```
✅ Data loaded successfully!
Demographics shape: (4000+, 9)
Performance shape: (4000+, 10)
Previous Loans shape: (18000+, 12)
```

---

## 🔍 Step 4: Explore Each Dataset

### 4.1 Performance Dataset (TARGET VARIABLE HERE!)

```python
print("=" * 60)
print("PERFORMANCE DATASET (Contains TARGET: good_bad_flag)")
print("=" * 60)

# Basic info
print("\n📊 Shape:", performance.shape)
print("\n📋 Columns:")
print(performance.columns.tolist())

# First few rows
print("\n👀 First 5 rows:")
print(performance.head())

# Data types
print("\n🔤 Data types:")
print(performance.dtypes)

# Missing values
print("\n❓ Missing values:")
print(performance.isnull().sum())

# Target variable distribution
print("\n🎯 TARGET VARIABLE DISTRIBUTION:")
print(performance['good_bad_flag'].value_counts())
print("\nPercentages:")
print(performance['good_bad_flag'].value_counts(normalize=True) * 100)
```

**Key Questions to Answer:**
- How many "Good" vs "Bad" loans? (Check for class imbalance!)
- Any missing values?
- What columns are available?

---

### 4.2 Demographics Dataset

```python
print("=" * 60)
print("DEMOGRAPHICS DATASET")
print("=" * 60)

print("\n📊 Shape:", demographics.shape)
print("\n📋 Columns:")
print(demographics.columns.tolist())

print("\n👀 First 5 rows:")
print(demographics.head())

print("\n❓ Missing values:")
missing = demographics.isnull().sum()
print(missing[missing > 0])

print("\n📈 Summary statistics:")
print(demographics.describe(include='all'))

# Check unique values for categorical columns
categorical_cols = ['bank_account_type', 'bank_name_clients',
                   'employment_status_clients', 'level_of_education_clients']

for col in categorical_cols:
    print(f"\n🔹 {col}:")
    print(demographics[col].value_counts().head(10))
```

**Key Insights to Look For:**
- How much missing data in employment_status, education, bank_branch?
- What are the most common banks?
- Age distribution (calculate from birthdate)
- Geographic distribution (longitude/latitude)

---

### 4.3 Previous Loans Dataset

```python
print("=" * 60)
print("PREVIOUS LOANS DATASET")
print("=" * 60)

print("\n📊 Shape:", prev_loans.shape)
print("\n📋 Columns:")
print(prev_loans.columns.tolist())

print("\n👀 First 5 rows:")
print(prev_loans.head())

print("\n❓ Missing values:")
missing = prev_loans.isnull().sum()
print(missing[missing > 0])

# Check how many loans per customer
print("\n🔢 Loans per customer:")
loans_per_customer = prev_loans.groupby('customerid').size()
print(f"Min loans: {loans_per_customer.min()}")
print(f"Max loans: {loans_per_customer.max()}")
print(f"Average loans: {loans_per_customer.mean():.2f}")
print(f"Median loans: {loans_per_customer.median():.0f}")

# Distribution of loan counts
print("\n📊 Distribution of loan counts:")
print(loans_per_customer.value_counts().sort_index().head(15))

# Loan amounts
print("\n💰 Loan amount statistics:")
print(prev_loans['loanamount'].describe())
```

**Key Insights:**
- How many previous loans does each customer have?
- Range of loan amounts
- Missing data in closeddate, referredby, firstrepaiddate

---

## 📊 Step 5: Visualizations

### 5.1 Target Variable Distribution

```python
# Plot target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
performance['good_bad_flag'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Loan Performance Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Loan Status')
axes[0].set_ylabel('Count')
axes[0].tick_params(rotation=0)

# Pie chart
performance['good_bad_flag'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                                                  colors=['green', 'red'])
axes[1].set_title('Loan Performance Percentage', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()
```

---

### 5.2 Age Distribution

```python
# Calculate age from birthdate
demographics['birthdate'] = pd.to_datetime(demographics['birthdate'])
demographics['age'] = (pd.Timestamp('2017-12-31') - demographics['birthdate']).dt.days // 365

# Plot age distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(demographics['age'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Age Distribution of Customers', fontsize=14, fontweight='bold')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')
plt.axvline(demographics['age'].median(), color='red', linestyle='--', label=f'Median: {demographics["age"].median():.0f}')
plt.legend()

plt.subplot(1, 2, 2)
demographics['age'].plot(kind='box', vert=True)
plt.title('Age Box Plot', fontsize=14, fontweight='bold')
plt.ylabel('Age (years)')

plt.tight_layout()
plt.show()

print(f"Age statistics:")
print(demographics['age'].describe())
```

---

### 5.3 Loan Amount Distribution

```python
# Previous loans amount distribution
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(prev_loans['loanamount'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Distribution of Previous Loan Amounts', fontsize=14, fontweight='bold')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.axvline(prev_loans['loanamount'].median(), color='red', linestyle='--',
            label=f'Median: ₦{prev_loans["loanamount"].median():,.0f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(performance['loanamount'], bins=50, edgecolor='black', alpha=0.7, color='orange')
plt.title('Distribution of Current Loan Amounts', fontsize=14, fontweight='bold')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.axvline(performance['loanamount'].median(), color='red', linestyle='--',
            label=f'Median: ₦{performance["loanamount"].median():,.0f}')
plt.legend()

plt.tight_layout()
plt.show()
```

---

### 5.4 Geographic Distribution (Map of Nigeria)

```python
# Plot customer locations
plt.figure(figsize=(10, 8))
plt.scatter(demographics['longitude_gps'], demographics['latitude_gps'],
           alpha=0.5, s=10, c='blue')
plt.title('Customer Geographic Distribution (Nigeria)', fontsize=14, fontweight='bold')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, alpha=0.3)
plt.show()

# Top cities (approximate clustering)
print("\n🗺️ Location statistics:")
print(f"Longitude range: {demographics['longitude_gps'].min():.2f} to {demographics['longitude_gps'].max():.2f}")
print(f"Latitude range: {demographics['latitude_gps'].min():.2f} to {demographics['latitude_gps'].max():.2f}")
```

---

## 🔑 Step 6: Key Findings Summary

At the end of your notebook, create a summary:

```python
print("=" * 60)
print("📝 DAY 1 KEY FINDINGS")
print("=" * 60)

print("\n1️⃣ DATASET SIZES:")
print(f"   - Customers: {performance.shape[0]}")
print(f"   - Previous loans: {prev_loans.shape[0]}")
print(f"   - Demographics: {demographics.shape[0]}")

print("\n2️⃣ TARGET VARIABLE:")
good_pct = (performance['good_bad_flag'] == 'Good').mean() * 100
bad_pct = (performance['good_bad_flag'] == 'Bad').mean() * 100
print(f"   - Good loans: {good_pct:.1f}%")
print(f"   - Bad loans: {bad_pct:.1f}%")
if abs(good_pct - bad_pct) > 20:
    print("   ⚠️ CLASS IMBALANCE DETECTED! Will need to handle this.")
else:
    print("   ✅ Relatively balanced classes")

print("\n3️⃣ MISSING DATA:")
print("   Demographics:")
missing_demo = demographics.isnull().sum()
for col in missing_demo[missing_demo > 0].index:
    pct = (missing_demo[col] / len(demographics)) * 100
    print(f"      - {col}: {pct:.1f}%")

print("\n4️⃣ PREVIOUS LOANS INSIGHTS:")
loans_per_cust = prev_loans.groupby('customerid').size()
print(f"   - Average previous loans per customer: {loans_per_cust.mean():.1f}")
print(f"   - Customers with only 1 previous loan: {(loans_per_cust == 1).sum()}")
print(f"   - Customers with 10+ previous loans: {(loans_per_cust >= 10).sum()}")

print("\n5️⃣ NEXT STEPS (Day 2):")
print("   - Merge the three datasets")
print("   - Create features from previous loan history")
print("   - Handle missing values")
print("   - Feature engineering")
```

---

## ✅ Day 1 Checklist

- [ ] Python environment set up with all libraries
- [ ] All 3 CSV files loaded successfully
- [ ] Explored each dataset (shape, columns, dtypes)
- [ ] Checked for missing values
- [ ] Analyzed target variable distribution
- [ ] Created visualizations (age, loan amounts, geography)
- [ ] Documented key findings
- [ ] Notebook saved as `notebooks/day1_exploration.ipynb`

---

## 🎓 What You Learned Today

1. **Data Loading:** How to load CSV files with pandas
2. **Data Exploration:** Using `.head()`, `.info()`, `.describe()`
3. **Missing Data:** Identifying and quantifying missing values
4. **Target Variable:** Understanding the prediction target (good_bad_flag)
5. **Relationships:** Recognizing 1:1 and 1:Many relationships in data
6. **Visualization:** Creating histograms, bar plots, scatter plots

---

## 🚀 Tomorrow (Day 2)

**Topic:** Data Merging & Understanding Relationships

You'll learn:
- How to merge the 3 datasets correctly
- Handling 1:Many relationships (previous loans)
- Creating a master dataset for modeling
- Validating merge operations

---

## 💡 Pro Tips

1. **Always check shapes after loading data** - Quick sanity check
2. **Look for missing data early** - Impacts feature engineering decisions
3. **Understand your target variable** - Everything revolves around predicting this
4. **Save your work frequently** - Ctrl+S / Cmd+S in Jupyter
5. **Document your findings** - Add markdown cells with observations

---

## 📚 Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Jupyter Keyboard Shortcuts](https://towardsdatascience.com/jypyter-notebook-shortcuts-bf0101a98330)

---

**🎉 Congratulations on completing Day 1!**

Take a break, review your notebook, and come back tomorrow for Day 2!
