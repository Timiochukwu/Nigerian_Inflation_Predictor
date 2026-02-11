# DAY 1: Data Exploration & Loading - Complete Walkthrough

**Goal**: Set up project structure, load loan data, and understand the dataset

**Time**: 30 minutes

---

## 📋 What You'll Build Today

By the end of Day 1, you'll have:
- ✅ Complete project folder structure
- ✅ Sample loan data generated (or loaded your own)
- ✅ Data exploration script that shows dataset statistics
- ✅ First processed CSV files saved

---

## PART 1: Create Project Structure

### Step 1: Create All Required Folders

Run these commands in your terminal from the `credit-risk-api/` directory:

```bash
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/final
mkdir -p models
mkdir -p src/data
mkdir -p results
```

Verify your structure:
```bash
ls -R
```

You should see:
```
credit-risk-api/
├── data/
│   ├── raw/
│   ├── processed/
│   └── final/
├── models/
├── src/
│   └── data/
└── results/
```

---

## PART 2: Prepare Your Data

### Option A: You Have Your Own Data

**You should have 3 CSV files:**
1. `loans.csv` - Loan application details
2. `customers.csv` - Customer information
3. `payments.csv` - Payment history

**Required columns for loans.csv:**
- `loan_id`, `customer_id`, `loan_amount`, `loan_status`

**Required columns for customers.csv:**
- `customer_id`, `age`, `income`, `total_debt`, `total_credit_limit`

**Required columns for payments.csv:**
- `customer_id`, `payment_status`, `days_late`

**Copy your 3 CSV files into the `data/` folder:**

```bash
# Copy your files
cp /path/to/your/loans.csv data/loans.csv
cp /path/to/your/customers.csv data/customers.csv
cp /path/to/your/payments.csv data/payments.csv
```

### Option B: Generate Sample Data

**If you don't have data yet, we'll generate realistic sample data.**

**Create the data generator:**

Create a new file: `src/data/generate_data.py`

Delete everything in `src/data/generate_data.py` and replace it with this:

```python
"""
Generate realistic sample credit risk data
"""
import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_data():
    """Generate sample loans, customers, and payments data"""

    np.random.seed(42)
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / 'data'
    DATA_DIR.mkdir(exist_ok=True)

    print("🔄 Generating sample data...")

    n_customers = 5000

    # ============================================================================
    # 1. Generate Customers Data
    # ============================================================================

    customers = pd.DataFrame({
        'customer_id': [f'C{str(i).zfill(5)}' for i in range(1, n_customers + 1)],
        'age': np.random.randint(22, 65, n_customers),
        'income': np.random.lognormal(mean=13, sigma=0.5, size=n_customers).astype(int),
        'employment_years': np.random.randint(0, 30, n_customers),
        'credit_score': np.random.randint(300, 850, n_customers),
        'total_debt': np.random.lognormal(mean=10, sigma=1.2, size=n_customers).astype(int),
        'total_credit_limit': np.random.lognormal(mean=11, sigma=0.8, size=n_customers).astype(int)
    })

    # ============================================================================
    # 2. Generate Loans Data
    # ============================================================================

    loans = pd.DataFrame({
        'loan_id': [f'L{str(i).zfill(5)}' for i in range(1, n_customers + 1)],
        'customer_id': customers['customer_id'],
        'loan_amount': np.random.lognormal(mean=10, sigma=1, size=n_customers).astype(int),
        'loan_term': np.random.choice([12, 24, 36, 48, 60], n_customers),
        'interest_rate': np.random.uniform(5, 25, n_customers).round(2),
        'purpose': np.random.choice(['business', 'personal', 'education', 'medical'], n_customers)
    })

    # Create realistic default pattern (30% default rate)
    # Higher default probability if:
    # - Low credit score
    # - High debt-to-income ratio
    # - High loan amount relative to income

    default_prob = (
        0.3 * (850 - customers['credit_score']) / 550 +
        0.3 * (loans['loan_amount'] / customers['income']) +
        0.2 * (customers['total_debt'] / customers['income']) +
        0.2 * np.random.random(n_customers)
    )

    loans['loan_status'] = np.where(
        default_prob > default_prob.quantile(0.7),
        'defaulted',
        'paid_off'
    )

    # ============================================================================
    # 3. Generate Payment History Data (10 payments per customer)
    # ============================================================================

    payment_records = []

    for customer_id in customers['customer_id']:
        loan_status = loans[loans['customer_id'] == customer_id]['loan_status'].values[0]

        for payment_num in range(1, 11):
            if loan_status == 'defaulted':
                # Defaulted customers have poor payment history
                status = np.random.choice(['paid', 'late', 'missed'], p=[0.3, 0.4, 0.3])
                days_late = np.random.randint(0, 90) if status in ['late', 'missed'] else 0
            else:
                # Good customers have good payment history
                status = np.random.choice(['paid', 'late', 'missed'], p=[0.85, 0.12, 0.03])
                days_late = np.random.randint(0, 30) if status in ['late', 'missed'] else 0

            payment_records.append({
                'customer_id': customer_id,
                'payment_number': payment_num,
                'payment_status': status,
                'days_late': days_late
            })

    payments = pd.DataFrame(payment_records)

    # ============================================================================
    # 4. Save All Data
    # ============================================================================

    customers.to_csv(DATA_DIR / 'customers.csv', index=False)
    loans.to_csv(DATA_DIR / 'loans.csv', index=False)
    payments.to_csv(DATA_DIR / 'payments.csv', index=False)

    print(f"✅ Generated customers.csv: {customers.shape}")
    print(f"✅ Generated loans.csv: {loans.shape}")
    print(f"✅ Generated payments.csv: {payments.shape}")
    print(f"\n📁 Files saved to: {DATA_DIR}/")
    print(f"   • customers.csv ({n_customers:,} customers)")
    print(f"   • loans.csv ({n_customers:,} loans)")
    print(f"   • payments.csv ({len(payments):,} payment records)")

    return customers, loans, payments


if __name__ == "__main__":
    generate_sample_data()
```

**Run the data generator:**

```bash
cd /home/user/Nigerian_Inflation_Predictor/credit-risk-api
python src/data/generate_data.py
```

**Expected output:**
```
🔄 Generating sample data...
✅ Generated customers.csv: (5000, 7)
✅ Generated loans.csv: (5000, 6)
✅ Generated payments.csv: (50000, 4)

📁 Files saved to: data/
   • customers.csv (5,000 customers)
   • loans.csv (5,000 loans)
   • payments.csv (50,000 payment records)
```

**Verify your data was created:**

```bash
ls -lh data/*.csv
```

You should see:
```
-rw-r--r-- 1 user user 450K Feb 11 10:00 data/customers.csv
-rw-r--r-- 1 user user 380K Feb 11 10:00 data/loans.csv
-rw-r--r-- 1 user user 1.2M Feb 11 10:00 data/payments.csv
```

---

## PART 3: Build Data Exploration Script

### Step 1: Create Day 1 Processing Script

Create a new file: `src/day1_load_explore.py`

Delete everything in `src/day1_load_explore.py` and replace it with this:

```python
"""
DAY 1: Data Exploration & Loading
Load datasets and perform initial exploration
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
PROCESSED_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("DAY 1: DATA EXPLORATION & LOADING")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Load Datasets
# ============================================================================

print("📂 STEP 1: Loading datasets...")
print("-" * 80)

# Check if data exists, if not generate it
if not (DATA_DIR / 'loans.csv').exists():
    print("⚠️  Data files not found. Please run:")
    print("   python src/data/generate_data.py")
    print("   OR copy your own data files to data/ folder")
    exit(1)

# Load all 3 datasets
loans_df = pd.read_csv(DATA_DIR / 'loans.csv')
customers_df = pd.read_csv(DATA_DIR / 'customers.csv')
payments_df = pd.read_csv(DATA_DIR / 'payments.csv')

print(f"✅ Loans data:     {loans_df.shape[0]:,} rows × {loans_df.shape[1]} columns")
print(f"✅ Customers data: {customers_df.shape[0]:,} rows × {customers_df.shape[1]} columns")
print(f"✅ Payments data:  {payments_df.shape[0]:,} rows × {payments_df.shape[1]} columns")
print()

# ============================================================================
# STEP 2: Explore Loans Data
# ============================================================================

print("📊 STEP 2: Exploring Loans Dataset")
print("-" * 80)

print("\n🔍 Loans Data Preview:")
print(loans_df.head())

print("\n📋 Loans Data Types:")
print(loans_df.dtypes)

print("\n📈 Loans Data Statistics:")
print(loans_df.describe())

print("\n⚠️  Missing Values in Loans:")
print(loans_df.isnull().sum())

print("\n🎯 Target Variable Distribution (loan_status):")
print(loans_df['loan_status'].value_counts())
print("\nPercentages:")
print(loans_df['loan_status'].value_counts(normalize=True) * 100)

# ============================================================================
# STEP 3: Explore Customers Data
# ============================================================================

print("\n" + "=" * 80)
print("📊 STEP 3: Exploring Customers Dataset")
print("-" * 80)

print("\n🔍 Customers Data Preview:")
print(customers_df.head())

print("\n📋 Customers Data Types:")
print(customers_df.dtypes)

print("\n📈 Customers Data Statistics:")
print(customers_df.describe())

print("\n⚠️  Missing Values in Customers:")
print(customers_df.isnull().sum())

# ============================================================================
# STEP 4: Explore Payments Data
# ============================================================================

print("\n" + "=" * 80)
print("📊 STEP 4: Exploring Payments Dataset")
print("-" * 80)

print("\n🔍 Payments Data Preview:")
print(payments_df.head())

print("\n📋 Payments Data Types:")
print(payments_df.dtypes)

print("\n📈 Payments Data Statistics:")
print(payments_df.describe())

print("\n⚠️  Missing Values in Payments:")
print(payments_df.isnull().sum())

print("\n🎯 Payment Status Distribution:")
print(payments_df['payment_status'].value_counts())
print("\nPercentages:")
print(payments_df['payment_status'].value_counts(normalize=True) * 100)

# ============================================================================
# STEP 5: Save Processed Data
# ============================================================================

print("\n" + "=" * 80)
print("💾 STEP 5: Saving processed data")
print("-" * 80)

loans_df.to_csv(PROCESSED_DIR / 'loans_day1.csv', index=False)
customers_df.to_csv(PROCESSED_DIR / 'customers_day1.csv', index=False)
payments_df.to_csv(PROCESSED_DIR / 'payments_day1.csv', index=False)

print(f"✅ Saved: {PROCESSED_DIR}/loans_day1.csv")
print(f"✅ Saved: {PROCESSED_DIR}/customers_day1.csv")
print(f"✅ Saved: {PROCESSED_DIR}/payments_day1.csv")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("🎉 DAY 1 COMPLETE!")
print("=" * 80)

print("\n📊 Summary:")
print(f"   • Loaded {loans_df.shape[0]:,} loan records")
print(f"   • Loaded {customers_df.shape[0]:,} customer records")
print(f"   • Loaded {payments_df.shape[0]:,} payment records")
print(f"   • Default rate: {(loans_df['loan_status'] == 'defaulted').mean() * 100:.1f}%")
print(f"   • Files saved to: {PROCESSED_DIR}/")

print("\n✅ Next: Run Day 2 to merge these datasets")
print("   python src/day2_merge.py")
```

---

## PART 4: Run Day 1 Script

### Step 1: Run the exploration script

```bash
python src/day1_load_explore.py
```

**Expected output:**

```
================================================================================
DAY 1: DATA EXPLORATION & LOADING
================================================================================

📂 STEP 1: Loading datasets...
--------------------------------------------------------------------------------
✅ Loans data:     5,000 rows × 6 columns
✅ Customers data: 5,000 rows × 7 columns
✅ Payments data:  50,000 rows × 4 columns

📊 STEP 2: Exploring Loans Dataset
--------------------------------------------------------------------------------

🔍 Loans Data Preview:
   loan_id customer_id  loan_amount  loan_term  interest_rate    purpose loan_status
0   L00001      C00001        15234         36          14.23   business    paid_off
1   L00002      C00002        45621         60          18.45   personal  defaulted
...

📋 Loans Data Types:
loan_id           object
customer_id       object
loan_amount        int64
loan_term          int64
interest_rate    float64
purpose           object
loan_status       object

🎯 Target Variable Distribution (loan_status):
paid_off     3500
defaulted    1500

Percentages:
paid_off     70.0
defaulted    30.0

...

💾 STEP 5: Saving processed data
--------------------------------------------------------------------------------
✅ Saved: data/processed/loans_day1.csv
✅ Saved: data/processed/customers_day1.csv
✅ Saved: data/processed/payments_day1.csv

================================================================================
🎉 DAY 1 COMPLETE!
================================================================================

📊 Summary:
   • Loaded 5,000 loan records
   • Loaded 5,000 customer records
   • Loaded 50,000 payment records
   • Default rate: 30.0%
   • Files saved to: data/processed/

✅ Next: Run Day 2 to merge these datasets
   python src/day2_merge.py
```

### Step 2: Verify the processed files were created

```bash
ls -lh data/processed/
```

You should see:
```
-rw-r--r-- 1 user user 380K Feb 11 10:15 loans_day1.csv
-rw-r--r-- 1 user user 450K Feb 11 10:15 customers_day1.csv
-rw-r--r-- 1 user user 1.2M Feb 11 10:15 payments_day1.csv
```

---

## 🎓 What You Just Built

**You now have:**
1. ✅ Complete project folder structure (data/, src/, models/, results/)
2. ✅ 3 CSV files with loan, customer, and payment data
3. ✅ Data exploration script that loads and analyzes all datasets
4. ✅ Processed data saved to `data/processed/`

**Key Insights from the data:**
- **70% good loans**, 30% defaulted loans (imbalanced dataset)
- **5,000 customers** with varying credit profiles
- **50,000 payment records** showing payment behavior
- Data is ready for merging (Day 2)

---

## 🔍 Understanding the Code

**What did generate_data.py do?**
- Created 5,000 synthetic customers with realistic attributes (age, income, credit score)
- Generated 5,000 loans with realistic default patterns
- Created 50,000 payment records (10 per customer) with payment behavior
- Used correlation: bad credit → more likely to default

**What did day1_load_explore.py do?**
- Loaded all 3 CSV files using pandas
- Displayed shape, data types, statistics for each dataset
- Showed missing values (should be zero for generated data)
- Analyzed target variable distribution (70% paid off, 30% defaulted)
- Saved copies to `processed/` folder for next steps

---

## ❓ Troubleshooting

**Error: "No module named 'pandas'"**
```bash
pip install pandas numpy
```

**Error: "File not found: data/loans.csv"**
```bash
# Make sure you ran the data generator:
python src/data/generate_data.py
```

**Error: "Permission denied"**
```bash
# Make sure folders exist:
mkdir -p data/processed
chmod -R 755 data/
```

---

## ✅ Day 1 Checklist

Before moving to Day 2, verify:

- [ ] Folder structure created (data/, src/, models/, results/)
- [ ] Data files exist: loans.csv, customers.csv, payments.csv
- [ ] day1_load_explore.py script created
- [ ] Script ran successfully without errors
- [ ] Processed files saved to data/processed/
- [ ] Understand the 70/30 split (70% good, 30% bad loans)

---

**🎉 Congratulations! Day 1 Complete!**

**Next**: Day 2 - Data Merging (Combine all 3 datasets into one)

**Ready?** Run:
```bash
# Go to Day 2 guide
cat guide/week1/day2_guide.md
```
