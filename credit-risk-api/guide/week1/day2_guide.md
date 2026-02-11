# Week 1, Day 2 -- Data Merging & Relationships: Joining Multiple Tables

## What You Will Learn Today

- How to merge multiple CSV files using `pandas`
- What a 1:1 vs 1:Many relationship means
- How to aggregate historical loan data (many loans per customer)
- How to handle customers with NO previous loan history
- How to create a master dataset ready for feature engineering

## Why This Matters

The Kaggle credit risk dataset comes as **three separate files**:

| File | Rows | What It Contains |
|------|------|------------------|
| `traindemographics.csv` | 5,000 | One row per customer (bank, location, employment, education) |
| `trainperf.csv` | 5,000 | One row per customer with TARGET variable (`good_bad_flag`) |
| `trainprevloans.csv` | 15,312 | Multiple rows per customer (historical loan repayment records) |

You cannot build a credit risk model from any single file. You need:
- Demographics (who is this person?)
- Current application (what loan are they applying for?)
- Historical behavior (how did they repay past loans?)

Today you will merge all three files into **one master DataFrame** where each row represents one customer's current loan application, enriched with their demographics and summarized loan history.

The three core join keys you will use:

| Key | Purpose |
|-----|---------|
| `customerid` | Links all three files (appears in all datasets) |
| `systemloanid` | Identifies each specific loan (current or historical) |

---

## Part 1: Verify Data Files Exist

Before writing code, confirm the Kaggle CSV files are in the correct location.

**Run this in your terminal:**

```bash
ls -lh credit-risk-api/data/raw/
```

**You should see:**

```
-rw-r--r--  traindemographics.csv  (~417K)
-rw-r--r--  trainperf.csv          (~228K)
-rw-r--r--  trainprevloans.csv     (~1.5M)
```

If any file is missing, download them from Kaggle:
👉 https://www.kaggle.com/competitions/data-science-nigeria-credit-risk-prediction/data

Place all three files in `credit-risk-api/data/raw/`.

---

## Part 2: Building the Practice Script (7 Steps)

We will create `guide/week1/day2_practice.py`. This file will grow step-by-step. Each step shows **the complete file from first line to last**. The instruction is always: "Delete everything in the file and replace it with this."

Make sure your virtual environment is activated:
```bash
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows
```

---

### Step 1: Load the Three Files

Create a new file: `guide/week1/day2_practice.py`

Delete everything in `guide/week1/day2_practice.py` and replace it with this:

```python
"""
Day 2 Practice: Data Merging & Relationships
Credit Risk Scoring System

Learning Goals:
- Load three related CSV files
- Understand 1:1 vs 1:Many relationships
- Prepare for merging datasets
"""

import pandas as pd
from pathlib import Path

# Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 2: DATA MERGING & RELATIONSHIPS")
print("="*80)

# Define paths
RAW_DATA_DIR = Path('credit-risk-api/data/raw')

# Load the three datasets
print("\n📂 STEP 1: Loading Three CSV Files...")
print("-" * 80)

demographics = pd.read_csv(RAW_DATA_DIR / 'traindemographics.csv')
performance = pd.read_csv(RAW_DATA_DIR / 'trainperf.csv')
prev_loans = pd.read_csv(RAW_DATA_DIR / 'trainprevloans.csv')

print(f"✅ Demographics loaded:   {demographics.shape[0]:,} rows × {demographics.shape[1]} columns")
print(f"✅ Performance loaded:    {performance.shape[0]:,} rows × {performance.shape[1]} columns")
print(f"✅ Previous Loans loaded: {prev_loans.shape[0]:,} rows × {prev_loans.shape[1]} columns")

print("\n📊 Demographics columns:")
print(demographics.columns.tolist())

print("\n📊 Performance columns:")
print(performance.columns.tolist())

print("\n📊 Previous Loans columns:")
print(prev_loans.columns.tolist())
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see:**

```
================================================================================
DAY 2: DATA MERGING & RELATIONSHIPS
================================================================================

📂 STEP 1: Loading Three CSV Files...
--------------------------------------------------------------------------------
✅ Demographics loaded:   5,000 rows × 7 columns
✅ Performance loaded:    5,000 rows × 6 columns
✅ Previous Loans loaded: 15,312 rows × 10 columns

📊 Demographics columns:
['customerid', 'bank_name_clients', 'bank_branch_clients', 'latitude_gps', 'longitude_gps', 'employment_status_clients', 'level_of_education_clients']

📊 Performance columns:
['customerid', 'systemloanid', 'loanamount', 'totaldue', 'termdays', 'good_bad_flag']

📊 Previous Loans columns:
['customerid', 'systemloanid', 'loanamount', 'totaldue', 'termdays', 'creationdate', 'approveddate', 'firstduedate', 'firstrepaiddate', 'closeddate']
```

**What just happened:**

- `from pathlib import Path` imports Python's modern path handling library. `Path` objects work correctly on Windows, Mac, and Linux without you having to worry about forward vs backward slashes.
- `RAW_DATA_DIR = Path('credit-risk-api/data/raw')` creates a path object pointing to the data folder. You can then use `/` to append filenames: `RAW_DATA_DIR / 'traindemographics.csv'` becomes `credit-risk-api/data/raw/traindemographics.csv`.
- `pd.read_csv()` loads each CSV file into a separate DataFrame. We now have three independent tables in memory.
- `.columns.tolist()` extracts the column names as a Python list. Notice that `customerid` appears in all three files -- this is the **join key** that links them together.
- Demographics and Performance both have exactly 5,000 rows (one per customer). Previous Loans has 15,312 rows because some customers have multiple historical loans.

---

### Step 2: Simple 1:1 Merge (Performance + Demographics)

Delete everything in `guide/week1/day2_practice.py` and replace it with this:

```python
"""
Day 2 Practice: Data Merging & Relationships
Credit Risk Scoring System
"""

import pandas as pd
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 2: DATA MERGING & RELATIONSHIPS")
print("="*80)

RAW_DATA_DIR = Path('credit-risk-api/data/raw')

print("\n📂 STEP 1: Loading Three CSV Files...")
print("-" * 80)

demographics = pd.read_csv(RAW_DATA_DIR / 'traindemographics.csv')
performance = pd.read_csv(RAW_DATA_DIR / 'trainperf.csv')
prev_loans = pd.read_csv(RAW_DATA_DIR / 'trainprevloans.csv')

print(f"✅ Demographics loaded:   {demographics.shape[0]:,} rows × {demographics.shape[1]} columns")
print(f"✅ Performance loaded:    {performance.shape[0]:,} rows × {performance.shape[1]} columns")
print(f"✅ Previous Loans loaded: {prev_loans.shape[0]:,} rows × {prev_loans.shape[1]} columns")

# --- NEW CODE BELOW ---

print("\n" + "="*80)
print("STEP 2: 1:1 Merge (Performance + Demographics)")
print("="*80)

print("\nBefore merge:")
print(f"  Performance:  {performance.shape}")
print(f"  Demographics: {demographics.shape}")

# Merge performance and demographics on customerid
base_df = performance.merge(demographics, on='customerid', how='left')

print(f"\nAfter merge:")
print(f"  Base DataFrame: {base_df.shape}")

print("\n👀 First 3 rows of merged data:")
print(base_df.head(3))

print("\n💡 This is a 1:1 merge because:")
print("   • Each customer appears exactly once in performance")
print("   • Each customer appears exactly once in demographics")
print("   • Result: Same number of rows (5,000), more columns")
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see** (new output at the bottom):

```
================================================================================
STEP 2: 1:1 Merge (Performance + Demographics)
================================================================================

Before merge:
  Performance:  (5000, 6)
  Demographics: (5000, 7)

After merge:
  Base DataFrame: (5000, 12)

👀 First 3 rows of merged data:
     customerid   systemloanid  loanamount  totaldue  termdays good_bad_flag bank_name_clients bank_branch_clients  latitude_gps  longitude_gps employment_status_clients level_of_education_clients
0  CUST_000001  LOAN_00000001       73564     98974       180           Bad     Fidelity Bank       Port Harcourt      5.175709       9.564338                  Contract                    HND/BSc
1  CUST_000002  LOAN_00000002       28946     36276        30          Good       Zenith Bank                Kano      7.346531       3.640082                 Permanent                  Secondary
2  CUST_000003  LOAN_00000003       44231     55623       180          Good               UBA              Kaduna      4.232551       7.783196                 Permanent                        PhD

💡 This is a 1:1 merge because:
   • Each customer appears exactly once in performance
   • Each customer appears exactly once in demographics
   • Result: Same number of rows (5,000), more columns
```

**What just happened:**

- `.merge(demographics, on='customerid', how='left')` joins two DataFrames using the `customerid` column as the key.
- `on='customerid'` tells pandas which column to match on. It finds rows where `customerid` is the same in both DataFrames and combines them into one row.
- `how='left'` means "keep all rows from the left DataFrame (performance) and add matching columns from the right DataFrame (demographics)." This is called a **left join**. If a customer appeared in performance but NOT in demographics (which doesn't happen here), their demographic columns would be filled with `NaN`.
- The result has **same number of rows** (5,000) but **more columns** (12 total: 6 from performance + 7 from demographics, minus 1 because `customerid` appears in both and is not duplicated).
- This is a **1:1 relationship** because each customer appears exactly once in each file. One customer → one current application → one set of demographics.

---

### Step 3: Understand the 1:Many Relationship

Delete everything in `guide/week1/day2_practice.py` and replace it with this:

```python
"""
Day 2 Practice: Data Merging & Relationships
Credit Risk Scoring System
"""

import pandas as pd
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 2: DATA MERGING & RELATIONSHIPS")
print("="*80)

RAW_DATA_DIR = Path('credit-risk-api/data/raw')

print("\n📂 STEP 1: Loading Three CSV Files...")
print("-" * 80)

demographics = pd.read_csv(RAW_DATA_DIR / 'traindemographics.csv')
performance = pd.read_csv(RAW_DATA_DIR / 'trainperf.csv')
prev_loans = pd.read_csv(RAW_DATA_DIR / 'trainprevloans.csv')

print(f"✅ Demographics loaded:   {demographics.shape[0]:,} rows × {demographics.shape[1]} columns")
print(f"✅ Performance loaded:    {performance.shape[0]:,} rows × {performance.shape[1]} columns")
print(f"✅ Previous Loans loaded: {prev_loans.shape[0]:,} rows × {prev_loans.shape[1]} columns")

print("\n" + "="*80)
print("STEP 2: 1:1 Merge (Performance + Demographics)")
print("="*80)

base_df = performance.merge(demographics, on='customerid', how='left')
print(f"Base DataFrame: {base_df.shape}")

# --- NEW CODE BELOW ---

print("\n" + "="*80)
print("STEP 3: Understanding the 1:Many Relationship")
print("="*80)

print("\n🔍 How many loans does each customer have?")

# Count loans per customer
loans_per_customer = prev_loans.groupby('customerid').size()
print(loans_per_customer.head(10))

print(f"\n📊 Statistics:")
print(f"  Total historical loans: {len(prev_loans):,}")
print(f"  Unique customers with history: {prev_loans['customerid'].nunique():,}")
print(f"  Average loans per customer: {len(prev_loans) / prev_loans['customerid'].nunique():.2f}")
print(f"  Max loans by one customer: {loans_per_customer.max()}")
print(f"  Min loans by one customer: {loans_per_customer.min()}")

print("\n💡 This is a 1:Many relationship because:")
print("   • One customer can have MANY historical loans")
print("   • We cannot merge prev_loans directly (would duplicate rows)")
print("   • Solution: AGGREGATE first, then merge")
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see** (new output at the bottom):

```
================================================================================
STEP 3: Understanding the 1:Many Relationship
================================================================================

🔍 How many loans does each customer have?
customerid
CUST_000001     8
CUST_000002     5
CUST_000003     1
CUST_000004     2
CUST_000005     3
CUST_000006     5
CUST_000007     1
CUST_000008     5
CUST_000009     1
CUST_000010     3
dtype: int64

📊 Statistics:
  Total historical loans: 15,312
  Unique customers with history: 4,515
  Average loans per customer: 3.39
  Max loans by one customer: 10
  Min loans by one customer: 1

💡 This is a 1:Many relationship because:
   • One customer can have MANY historical loans
   • We cannot merge prev_loans directly (would duplicate rows)
   • Solution: AGGREGATE first, then merge
```

**What just happened:**

- `.groupby('customerid').size()` groups all rows by `customerid` and counts how many rows are in each group. The result is a Series showing how many historical loans each customer has.
- Customer `CUST_000001` has 8 previous loans. Customer `CUST_000003` has only 1. Some customers are repeat borrowers, others are new.
- **1:Many relationship** means one customer (in performance/demographics) corresponds to multiple rows in prev_loans.
- If you tried to merge prev_loans directly onto base_df, you would get 15,312 rows (one per historical loan). But base_df has 5,000 rows (one per customer). The merge would **duplicate** customer data for each of their past loans, which is wrong. You need **one row per customer**, not one row per historical loan.
- **Solution:** Aggregate (summarize) the historical loans FIRST, then merge the summary. Each customer gets one row with aggregated stats like "total number of loans," "average loan amount," "on-time payment rate," etc.

---

### Step 4: Aggregate Historical Loans

Delete everything in `guide/week1/day2_practice.py` and replace it with this:

```python
"""
Day 2 Practice: Data Merging & Relationships
Credit Risk Scoring System
"""

import pandas as pd
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 2: DATA MERGING & RELATIONSHIPS")
print("="*80)

RAW_DATA_DIR = Path('credit-risk-api/data/raw')

print("\n📂 STEP 1: Loading Three CSV Files...")
demographics = pd.read_csv(RAW_DATA_DIR / 'traindemographics.csv')
performance = pd.read_csv(RAW_DATA_DIR / 'trainperf.csv')
prev_loans = pd.read_csv(RAW_DATA_DIR / 'trainprevloans.csv')

print(f"✅ Loaded {demographics.shape[0]:,} customers, {prev_loans.shape[0]:,} historical loans")

print("\n" + "="*80)
print("STEP 2: 1:1 Merge (Performance + Demographics)")
print("="*80)
base_df = performance.merge(demographics, on='customerid', how='left')
print(f"Base DataFrame: {base_df.shape}")

print("\n" + "="*80)
print("STEP 3: Understanding 1:Many Relationship")
print("="*80)
loans_per_customer = prev_loans.groupby('customerid').size()
print(f"Average loans per customer: {loans_per_customer.mean():.2f}")

# --- NEW CODE BELOW ---

print("\n" + "="*80)
print("STEP 4: Aggregate Historical Loans (Summarize per Customer)")
print("="*80)

print("\n🔧 Creating summary statistics for each customer's loan history...")

# Group by customer and aggregate
hist_summary = prev_loans.groupby('customerid').agg({
    'loanamount': ['count', 'sum', 'mean', 'max', 'min'],
    'totaldue': 'mean',
    'termdays': 'mean'
}).reset_index()

# Flatten multi-level column names
hist_summary.columns = ['customerid', 'num_prev_loans', 'total_borrowed',
                        'avg_loan_amount', 'max_loan_amount', 'min_loan_amount',
                        'avg_total_due', 'avg_term_days']

print(f"\n✅ Historical summary created: {hist_summary.shape}")
print("\n👀 First 5 customers:")
print(hist_summary.head())

print("\n📊 What these columns mean:")
print("  • num_prev_loans:    How many loans this customer had before")
print("  • total_borrowed:    Sum of all previous loan amounts")
print("  • avg_loan_amount:   Average size of their past loans")
print("  • max_loan_amount:   Their largest loan ever")
print("  • avg_total_due:     Average total due (loan + interest)")
print("  • avg_term_days:     Average loan duration in days")
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see** (new output at the bottom):

```
================================================================================
STEP 4: Aggregate Historical Loans (Summarize per Customer)
================================================================================

🔧 Creating summary statistics for each customer's loan history...

✅ Historical summary created: (4515, 8)

👀 First 5 customers:
     customerid  num_prev_loans  total_borrowed  avg_loan_amount  max_loan_amount  min_loan_amount  avg_total_due  avg_term_days
0  CUST_000001               8          395177         49397.12            61315            24015       58827.50          77.50
1  CUST_000002               5          181773         36354.60            73558            12788       49599.00          96.00
2  CUST_000003               1           25530         25530.00            25530            25530       34522.00          30.00
3  CUST_000004               2           89936         44968.00            55403            34533       65050.50          75.00
4  CUST_000005               3          172699         57566.33            97085            27806       75959.67         120.00

📊 What these columns mean:
  • num_prev_loans:    How many loans this customer had before
  • total_borrowed:    Sum of all previous loan amounts
  • avg_loan_amount:   Average size of their past loans
  • max_loan_amount:   Their largest loan ever
  • avg_total_due:     Average total due (loan + interest)
  • avg_term_days:     Average loan duration in days
```

**What just happened:**

- `.groupby('customerid').agg({...})` groups all rows by customer and applies multiple aggregation functions.
- `'loanamount': ['count', 'sum', 'mean', 'max', 'min']` tells pandas to compute 5 different statistics for the `loanamount` column: count (how many loans), sum (total borrowed), mean (average), max (largest loan), min (smallest loan).
- `.agg()` with a dictionary creates **multi-level column names** like `('loanamount', 'count')` and `('loanamount', 'sum')`. These are hard to work with.
- `.reset_index()` converts the `customerid` from being the index back to being a regular column. This makes it easier to merge later.
- `hist_summary.columns = [...]` **flattens** the multi-level columns into simple names: `num_prev_loans`, `total_borrowed`, etc. This makes the DataFrame much easier to read and work with.
- Now we have **one row per customer** (4,515 rows) with summary statistics. Customer `CUST_000001` had 8 previous loans totaling ₦395,177. Their average loan was ₦49,397.

---

### Step 5: Merge Historical Summary onto Base

Delete everything in `guide/week1/day2_practice.py` and replace it with this:

```python
"""
Day 2 Practice: Data Merging & Relationships
Credit Risk Scoring System
"""

import pandas as pd
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 2: DATA MERGING & RELATIONSHIPS")
print("="*80)

RAW_DATA_DIR = Path('credit-risk-api/data/raw')

demographics = pd.read_csv(RAW_DATA_DIR / 'traindemographics.csv')
performance = pd.read_csv(RAW_DATA_DIR / 'trainperf.csv')
prev_loans = pd.read_csv(RAW_DATA_DIR / 'trainprevloans.csv')

print(f"✅ Loaded {demographics.shape[0]:,} customers, {prev_loans.shape[0]:,} historical loans\n")

# Step 2: 1:1 merge
base_df = performance.merge(demographics, on='customerid', how='left')

# Step 4: Aggregate
hist_summary = prev_loans.groupby('customerid').agg({
    'loanamount': ['count', 'sum', 'mean', 'max', 'min'],
    'totaldue': 'mean',
    'termdays': 'mean'
}).reset_index()

hist_summary.columns = ['customerid', 'num_prev_loans', 'total_borrowed',
                        'avg_loan_amount', 'max_loan_amount', 'min_loan_amount',
                        'avg_total_due', 'avg_term_days']

print(f"Base DataFrame: {base_df.shape}")
print(f"Historical Summary: {hist_summary.shape}")

# --- NEW CODE BELOW ---

print("\n" + "="*80)
print("STEP 5: Merge Historical Summary onto Base")
print("="*80)

print("\nBefore merge:")
print(f"  Base:      {base_df.shape[0]:,} rows, {base_df.shape[1]} columns")
print(f"  History:   {hist_summary.shape[0]:,} rows, {hist_summary.shape[1]} columns")

# Merge historical summary onto base
master_df = base_df.merge(hist_summary, on='customerid', how='left')

print(f"\nAfter merge:")
print(f"  Master DF: {master_df.shape[0]:,} rows, {master_df.shape[1]} columns")

print("\n✅ Master DataFrame created!")
print(f"   • {master_df.shape[0]:,} customers (one row per customer)")
print(f"   • {master_df.shape[1]} total features")

print("\n👀 Sample row with all data:")
print(master_df.iloc[0].to_frame().T)

print("\n⚠️  IMPORTANT: Check for missing values...")
missing_hist = master_df['num_prev_loans'].isnull().sum()
print(f"  Customers with NO loan history: {missing_hist:,}")

if missing_hist > 0:
    print(f"  ({missing_hist / len(master_df) * 100:.1f}% of total)")
    print("  → These customers' historical features are NaN")
    print("  → Tomorrow (Day 3) you'll handle this with imputation")
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see** (new output at the bottom):

```
================================================================================
STEP 5: Merge Historical Summary onto Base
================================================================================

Before merge:
  Base:      5,000 rows, 12 columns
  History:   4,515 rows, 8 columns

After merge:
  Master DF: 5,000 rows, 19 columns

✅ Master DataFrame created!
   • 5,000 customers (one row per customer)
   • 19 total features

👀 Sample row with all data:
     customerid   systemloanid  loanamount  totaldue  termdays good_bad_flag bank_name_clients bank_branch_clients  latitude_gps  longitude_gps employment_status_clients level_of_education_clients  num_prev_loans  total_borrowed  avg_loan_amount  max_loan_amount  min_loan_amount  avg_total_due  avg_term_days
0  CUST_000001  LOAN_00000001       73564     98974       180           Bad     Fidelity Bank       Port Harcourt      5.175709       9.564338                  Contract                    HND/BSc             8.0          395177         49397.12            61315            24015       58827.50          77.50

⚠️  IMPORTANT: Check for missing values...
  Customers with NO loan history: 485
  (9.7% of total)
  → These customers' historical features are NaN
  → Tomorrow (Day 3) you'll handle this with imputation
```

**What just happened:**

- `.merge(hist_summary, on='customerid', how='left')` joins the historical summary onto the base DataFrame.
- `how='left'` means "keep all 5,000 customers from base_df, even if some don't appear in hist_summary."
- 485 customers appear in base_df but NOT in hist_summary. Why? Because they have NO previous loan history in the `prev_loans` file. These are first-time borrowers.
- For these 485 customers, all historical columns (`num_prev_loans`, `total_borrowed`, etc.) are filled with `NaN` (missing).
- The final master DataFrame has:
  - 5,000 rows (one per customer)
  - 19 columns (6 from performance + 6 from demographics + 7 from historical summary, minus duplicate `customerid`)
- This is the **master dataset** that combines all information: who they are (demographics), what they're applying for (performance), and how they've behaved in the past (historical summary).

---

### Step 6: Handle Missing Historical Features

Delete everything in `guide/week1/day2_practice.py` and replace it with this:

```python
"""
Day 2 Practice: Data Merging & Relationships
Credit Risk Scoring System
"""

import pandas as pd
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 2: DATA MERGING & RELATIONSHIPS")
print("="*80)

RAW_DATA_DIR = Path('credit-risk-api/data/raw')

demographics = pd.read_csv(RAW_DATA_DIR / 'traindemographics.csv')
performance = pd.read_csv(RAW_DATA_DIR / 'trainperf.csv')
prev_loans = pd.read_csv(RAW_DATA_DIR / 'trainprevloans.csv')

# Merge steps
base_df = performance.merge(demographics, on='customerid', how='left')

hist_summary = prev_loans.groupby('customerid').agg({
    'loanamount': ['count', 'sum', 'mean', 'max', 'min'],
    'totaldue': 'mean',
    'termdays': 'mean'
}).reset_index()

hist_summary.columns = ['customerid', 'num_prev_loans', 'total_borrowed',
                        'avg_loan_amount', 'max_loan_amount', 'min_loan_amount',
                        'avg_total_due', 'avg_term_days']

master_df = base_df.merge(hist_summary, on='customerid', how='left')

print(f"✅ Master DataFrame: {master_df.shape}")
print(f"   Missing history: {master_df['num_prev_loans'].isnull().sum():,} customers\n")

# --- NEW CODE BELOW ---

print("="*80)
print("STEP 6: Handle Missing Historical Features")
print("="*80)

print("\n🔍 Checking missing values per column:")
missing_counts = master_df.isnull().sum()
print(missing_counts[missing_counts > 0])

print("\n💡 Why are historical columns missing?")
print("   → 485 customers have NO previous loans in the dataset")
print("   → For them, ALL historical features are NaN")

print("\n🔧 Filling missing values with 0...")
print("   Logic: If no loan history exists, then:")
print("     • num_prev_loans = 0   (they've had zero loans)")
print("     • total_borrowed = 0   (they've borrowed nothing)")
print("     • avg_loan_amount = 0  (average of nothing is zero)")
print("     • etc.")

# Fill missing historical features with 0
hist_cols = ['num_prev_loans', 'total_borrowed', 'avg_loan_amount',
             'max_loan_amount', 'min_loan_amount', 'avg_total_due', 'avg_term_days']

master_df[hist_cols] = master_df[hist_cols].fillna(0)

print(f"\n✅ Missing values after filling:")
print(master_df.isnull().sum().sum())  # Total missing across entire DataFrame

print("\n👀 Sample customer with NO history (all 0s):")
no_history_customer = master_df[master_df['num_prev_loans'] == 0].iloc[0]
print(no_history_customer[['customerid', 'num_prev_loans', 'total_borrowed', 'avg_loan_amount']])

print("\n👀 Sample customer WITH history (real values):")
with_history_customer = master_df[master_df['num_prev_loans'] > 0].iloc[0]
print(with_history_customer[['customerid', 'num_prev_loans', 'total_borrowed', 'avg_loan_amount']])
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see** (new output at the bottom):

```
================================================================================
STEP 6: Handle Missing Historical Features
================================================================================

🔍 Checking missing values per column:
num_prev_loans       485
total_borrowed       485
avg_loan_amount      485
max_loan_amount      485
min_loan_amount      485
avg_total_due        485
avg_term_days        485
dtype: int64

💡 Why are historical columns missing?
   → 485 customers have NO previous loans in the dataset
   → For them, ALL historical features are NaN

🔧 Filling missing values with 0...
   Logic: If no loan history exists, then:
     • num_prev_loans = 0   (they've had zero loans)
     • total_borrowed = 0   (they've borrowed nothing)
     • avg_loan_amount = 0  (average of nothing is zero)
     • etc.

✅ Missing values after filling:
0

👀 Sample customer with NO history (all 0s):
customerid               CUST_000006
num_prev_loans                   0.0
total_borrowed                   0.0
avg_loan_amount                  0.0
Name: 5, dtype: object

👀 Sample customer WITH history (real values):
customerid               CUST_000001
num_prev_loans                   8.0
total_borrowed                395177.0
avg_loan_amount            49397.125
Name: 0, dtype: object
```

**What just happened:**

- `.fillna(0)` replaces all `NaN` values in the specified columns with `0`.
- This makes logical sense: If a customer has no loan history, then `num_prev_loans = 0`, `total_borrowed = 0`, etc.
- After filling, the DataFrame has **zero missing values** (`isnull().sum().sum() == 0`).
- Customer `CUST_000006` has all zeros in historical columns because they're a first-time borrower.
- Customer `CUST_000001` has real values because they have 8 previous loans.
- This is a simple imputation strategy. Tomorrow (Day 3) you'll explore more sophisticated feature engineering from the date columns in `prev_loans` (like on-time payment rate, days late, etc.).

---

### Step 7: Save the Master Dataset

Delete everything in `guide/week1/day2_practice.py` and replace it with this:

```python
"""
Day 2 Practice: Data Merging & Relationships
Credit Risk Scoring System

Final complete version with all steps
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

# --- NEW CODE BELOW ---

print("\n" + "="*80)
print("STEP 7: Save Master Dataset")
print("="*80)

# Create output directory
PROCESSED_DATA_DIR = Path('credit-risk-api/data/processed')
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Save to CSV
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
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see** (final output):

```
================================================================================
STEP 7: Save Master Dataset
================================================================================

💾 Saved master dataset to:
   credit-risk-api/data/processed/master_dataset_day2.csv
   Shape: (5000, 19)
   Size: 789.2 KB

================================================================================
📝 DAY 2 SUMMARY
================================================================================

✅ Successfully merged three datasets:
   1. traindemographics.csv  (5,000 rows) → Customer info
   2. trainperf.csv          (5,000 rows) → Target variable
   3. trainprevloans.csv     (15,312 rows) → Historical behavior

📊 Final Master Dataset:
   • Rows: 5,000 (one per customer)
   • Columns: 19 features
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
```

**What just happened:**

- `.mkdir(parents=True, exist_ok=True)` creates the output directory if it doesn't exist. `parents=True` means "create parent folders too if needed." `exist_ok=True` means "don't raise an error if the folder already exists."
- `.to_csv(output_path, index=False)` saves the DataFrame to a CSV file. `index=False` means "don't save the row numbers as a column." The file will have 19 columns (all the features) and 5,000 rows (one per customer).
- `.stat().st_size` gets the file size in bytes. We divide by 1024 to convert to kilobytes.
- You now have a **clean, merged master dataset** saved to disk. Tomorrow you can load it with `pd.read_csv('credit-risk-api/data/processed/master_dataset_day2.csv')` and continue building features without re-running the merge steps.

---

## Part 3: Key Concepts Summary

| Operation | What It Does |
|-----------|-------------|
| `df1.merge(df2, on='key', how='left')` | Join two DataFrames using a common column (`key`) |
| `how='left'` | Keep all rows from left DataFrame, add matching columns from right |
| `.groupby('col').size()` | Count rows per group |
| `.groupby('col').agg({...})` | Compute multiple aggregations per group |
| `.reset_index()` | Convert index back to a regular column |
| `.fillna(value)` | Replace missing values (`NaN`) with a specific value |
| `Path.mkdir(parents=True, exist_ok=True)` | Create directory (and parents) if it doesn't exist |
| `.to_csv(path, index=False)` | Save DataFrame to CSV without row numbers |

---

## Part 4: Commit Your Work

```bash
git add guide/week1/day2_practice.py credit-risk-api/data/processed/master_dataset_day2.csv
git commit -m "Day 2: Data merging practice (1:1 and 1:Many relationships)"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: traindemographics.csv` | Ensure CSV files are in `credit-risk-api/data/raw/`. Download from Kaggle if missing. |
| `KeyError: 'customerid'` | Column names are case-sensitive. Check exact spelling in CSV. |
| Merge produces wrong number of rows | Check `how=` parameter. Use `how='left'` to keep all rows from left DataFrame. |
| Historical columns still have NaN after fillna | Make sure you're filling the correct column list (`hist_cols`). |
| `PermissionError` when saving CSV | File may be open in Excel. Close it and try again. |

---

## Check Your Understanding

**1. What is the difference between a 1:1 merge and a 1:Many relationship?**

A 1:1 merge joins two tables where each row in the left table matches exactly one row in the right table. The result has the same number of rows as the left table, just more columns. Example: performance (5,000 rows) + demographics (5,000 rows) → base_df (5,000 rows, 12 columns).

A 1:Many relationship means one row in the left table corresponds to multiple rows in the right table. Example: one customer has many historical loans. If you merge directly, you get one row per historical loan, which duplicates customer data. The solution is to **aggregate the many rows into one summary row first**, then merge that summary.

**2. Why did we use `.fillna(0)` for missing historical features?**

485 customers have no previous loan history in the dataset. When we merged `hist_summary` (which only contains customers with history) onto `master_df` (which has all 5,000 customers), those 485 got `NaN` values for all historical columns. Filling with 0 makes logical sense: zero previous loans means `num_prev_loans = 0`, zero total borrowed means `total_borrowed = 0`, etc. This is a simple imputation strategy that lets the model know "this customer has no history" rather than "this value is unknown."

**3. What happens if you forget `how='left'` in the merge?**

The default merge type is `how='inner'`, which only keeps rows where the key (`customerid`) exists in **both** DataFrames. If you merge `base_df` (5,000 rows) with `hist_summary` (4,515 rows) using `how='inner'`, you'd lose the 485 customers with no history. The result would be 4,515 rows instead of 5,000. Always use `how='left'` when you want to keep all rows from the left DataFrame and add columns from the right.

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Practice script | `guide/week1/day2_practice.py` | Learned merging, aggregation, handling missing data |
| Master dataset | `credit-risk-api/data/processed/master_dataset_day2.csv` | Merged dataset ready for feature engineering |

**New pandas operations learned:**
`.merge()`, `.groupby()`, `.agg()`, `.reset_index()`, `.fillna()`, `.to_csv()`

**Tomorrow (Day 3):** Feature engineering from date columns (calculate on-time rate, days late, loan closure rate), encode categorical variables, create 30+ features total.
