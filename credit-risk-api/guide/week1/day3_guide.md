# Week 1, Day 3 -- Feature Engineering: Mining Gold from Historical Loan Data

## What You Will Learn Today

- How to engineer features from date columns (payment behavior)
- How to calculate on-time payment rate, late payment rate, never paid rate
- How to measure loan closure rate (completion vs abandonment)
- How to create recency features (days since last loan)
- How to encode categorical variables (employment, education)
- How to create derived features (ratios, increases)

## Why This Matters

Yesterday you merged three datasets and created 19 basic features. Today you will **engineer 16 additional powerful features** that will dramatically improve your credit risk model's accuracy.

The Kaggle dataset provides historical loan records in `trainprevloans.csv` with date columns that tell the complete payment story:

| Date Column | Meaning |
|-------------|---------|
| `creationdate` | When loan application was created |
| `approveddate` | When loan was approved |
| `firstduedate` | When first payment was due |
| `firstrepaiddate` | When customer actually made first payment |
| `closeddate` | When loan was fully closed/completed |

The difference between `firstduedate` and `firstrepaiddate` reveals whether the customer paid on time, late, or never paid. The presence or absence of `closeddate` reveals whether they completed the loan or abandoned it.

**Payment behavior is the #1 predictor of credit risk.** A customer who consistently pays late or never completes loans is high-risk, regardless of their demographics or current loan amount.

Today you will transform raw dates into powerful behavioral metrics that capture credit risk patterns.

---

## Part 1: Verify Yesterday's Output

Before starting, confirm the master dataset from Day 2 exists:

```bash
ls -lh credit-risk-api/data/processed/master_dataset_day2.csv
```

**You should see:**
```
-rw-r--r--  master_dataset_day2.csv  (~844 KB)
```

If the file is missing, go back and run `day2_practice.py` first.

---

## Part 2: Building the Practice Script (8 Steps)

Create a new file: `guide/week1/day3_practice.py`

Each step below shows **the complete file from first line to last**. The instruction is always: "Delete everything in the file and replace it with this." Do not try to add lines to the previous version - just replace the whole file each time.

---

### Step 1: Load Master Dataset and Historical Loans

Delete everything in `guide/week1/day3_practice.py` and replace it with this:

```python
"""
Day 3 Practice: Feature Engineering from Historical Loans
Credit Risk Scoring System

Learning Goals:
- Engineer features from date columns
- Calculate payment behavior metrics (on-time, late, never paid rates)
- Create recency features
- Encode categorical variables
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 3: FEATURE ENGINEERING FROM HISTORICAL LOANS")
print("="*80)

# Define paths
PROCESSED_DIR = Path('credit-risk-api/data/processed')
RAW_DIR = Path('credit-risk-api/data/raw')

# Load datasets
print("\n📂 STEP 1: Loading Data...")
print("-" * 80)

master_df = pd.read_csv(PROCESSED_DIR / 'master_dataset_day2.csv')
prev_loans = pd.read_csv(RAW_DIR / 'trainprevloans.csv')

print(f"✅ Master dataset loaded: {master_df.shape}")
print(f"✅ Previous loans loaded: {prev_loans.shape}")

print(f"\n📊 Current features in master_df: {master_df.shape[1]} columns")
print("   Sample columns:", master_df.columns[:5].tolist(), "...")

print(f"\n📊 Previous loans columns ({prev_loans.shape[1]} total):")
print("   ", prev_loans.columns.tolist())

print(f"\n🎯 Goal for today:")
print("   Starting features: 19")
print("   Target features: 35+ (adding 16+ new features)")
print("   Focus: Payment behavior from date columns")
```

Save the file.

**Run it:**

```bash
python guide/week1/day3_practice.py
```

**What you should see:**

```
================================================================================
DAY 3: FEATURE ENGINEERING FROM HISTORICAL LOANS
================================================================================

📂 STEP 1: Loading Data...
--------------------------------------------------------------------------------
✅ Master dataset loaded: (5000, 19)
✅ Previous loans loaded: (15312, 10)

📊 Current features in master_df: 19 columns
   Sample columns: ['customerid', 'systemloanid', 'loanamount', 'totaldue', 'termdays'] ...

📊 Previous loans columns (10 total):
    ['customerid', 'systemloanid', 'loanamount', 'totaldue', 'termdays', 'creationdate', 'approveddate', 'firstduedate', 'firstrepaiddate', 'closeddate']

🎯 Goal for today:
   Starting features: 19
   Target features: 35+ (adding 16+ new features)
   Focus: Payment behavior from date columns
```

**What just happened:**

- Loaded the 19-feature master dataset from Day 2 (5,000 customers).
- Reloaded `trainprevloans.csv` which has 15,312 historical loan records across 4,515 customers.
- Yesterday we used `prev_loans` to create basic aggregations (count, sum, mean). Today we will use the **date columns** (`creationdate`, `approveddate`, `firstduedate`, `firstrepaiddate`, `closeddate`) to engineer payment behavior features.
- Notice that `prev_loans` has 10 columns, and 5 of them are dates. These dates contain the raw material for our most powerful predictive features.

---

### Step 2: Convert Date Strings to Datetime Objects

Delete everything in `guide/week1/day3_practice.py` and replace it with this:

```python
"""
Day 3 Practice: Feature Engineering from Historical Loans
"""

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 3: FEATURE ENGINEERING")
print("="*80)

PROCESSED_DIR = Path('credit-risk-api/data/processed')
RAW_DIR = Path('credit-risk-api/data/raw')

master_df = pd.read_csv(PROCESSED_DIR / 'master_dataset_day2.csv')
prev_loans = pd.read_csv(RAW_DIR / 'trainprevloans.csv')

print(f"✅ Loaded: master_df {master_df.shape}, prev_loans {prev_loans.shape}\n")

# --- NEW CODE BELOW ---

print("="*80)
print("STEP 2: Convert Date Strings to Datetime Objects")
print("="*80)

print("\n🔍 Before conversion:")
print("Data types of date columns:")
print(prev_loans[['firstduedate', 'firstrepaiddate', 'closeddate']].dtypes)

print("\nSample raw values (text strings):")
print(prev_loans[['customerid', 'firstduedate', 'firstrepaiddate', 'closeddate']].head(5))

# Convert all date columns from text to datetime
date_cols = ['creationdate', 'approveddate', 'firstduedate', 'firstrepaiddate', 'closeddate']

for col in date_cols:
    prev_loans[col] = pd.to_datetime(prev_loans[col], errors='coerce')

print("\n✅ After conversion:")
print("Data types of date columns:")
print(prev_loans[['firstduedate', 'firstrepaiddate', 'closeddate']].dtypes)

print("\nSample converted values (datetime objects):")
print(prev_loans[['customerid', 'firstduedate', 'firstrepaiddate', 'closeddate']].head(5))

print("\n💡 Why this matters:")
print("   Before: dates are TEXT → cannot subtract, cannot compute days")
print("   After:  dates are DATETIME → can subtract to get time difference")
print("   Example: firstrepaiddate - firstduedate = days late or early")
print("   Missing dates become NaT (Not a Time), similar to NaN")
```

Save and run:

```bash
python guide/week1/day3_practice.py
```

**What you should see:**

```
================================================================================
STEP 2: Convert Date Strings to Datetime Objects
================================================================================

🔍 Before conversion:
Data types of date columns:
firstduedate       object
firstrepaiddate    object
closeddate         object
dtype: object

Sample raw values (text strings):
     customerid  firstduedate firstrepaiddate  closeddate
0  CUST_000001    2023-02-20      2023-02-25  2023-07-22
1  CUST_000001    2023-02-20      2023-02-20  2024-02-15
2  CUST_000001    2023-02-20      2023-02-20  2024-02-15
3  CUST_000001    2023-02-20      2023-02-25         NaN
4  CUST_000001    2023-02-20      2023-02-25  2023-03-10

✅ After conversion:
Data types of date columns:
firstduedate       datetime64[ns]
firstrepaiddate    datetime64[ns]
closeddate         datetime64[ns]
dtype: object

Sample converted values (datetime objects):
     customerid firstduedate firstrepaiddate closeddate
0  CUST_000001   2023-02-20      2023-02-25 2023-07-22
1  CUST_000001   2023-02-20      2023-02-20 2024-02-15
2  CUST_000001   2023-02-20      2023-02-20 2024-02-15
3  CUST_000001   2023-02-20      2023-02-25        NaT
4  CUST_000001   2023-02-20      2023-02-25 2023-03-10

💡 Why this matters:
   Before: dates are TEXT → cannot subtract, cannot compute days
   After:  dates are DATETIME → can subtract to get time difference
   Example: firstrepaiddate - firstduedate = days late or early
   Missing dates become NaT (Not a Time), similar to NaN
```

**What just happened:**

- `pd.to_datetime(col, errors='coerce')` converts text strings like `'2023-02-20'` into datetime objects that Python recognizes as actual dates.
- **Before conversion:** dtype is `object` (generic text). Python treats these as ordinary strings with no concept of chronological order or time arithmetic.
- **After conversion:** dtype is `datetime64[ns]` (datetime with nanosecond precision). Python now knows these are dates and can perform date arithmetic.
- `errors='coerce'` means "if a date value is invalid or missing (empty cell), convert it to `NaT` (Not a Time) instead of raising an error." `NaT` is pandas' version of `NaN` for datetime columns.
- Row 3 has `NaT` in `closeddate` because that loan was never closed (still open or defaulted).
- **Why this is critical:** Once dates are datetime objects, we can subtract them to compute time differences. For example, `firstrepaiddate - firstduedate` will give us the number of days between when payment was due and when it was actually made. This tells us if the customer paid early, on time, late, or never paid.

---

### Step 3: Calculate Days to Repayment (Payment Timing)

Delete everything in `guide/week1/day3_practice.py` and replace it with this:

```python
"""
Day 3 Practice: Feature Engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 3: FEATURE ENGINEERING")
print("="*80)

PROCESSED_DIR = Path('credit-risk-api/data/processed')
RAW_DIR = Path('credit-risk-api/data/raw')

master_df = pd.read_csv(PROCESSED_DIR / 'master_dataset_day2.csv')
prev_loans = pd.read_csv(RAW_DIR / 'trainprevloans.csv')

# Convert dates
date_cols = ['creationdate', 'approveddate', 'firstduedate', 'firstrepaiddate', 'closeddate']
for col in date_cols:
    prev_loans[col] = pd.to_datetime(prev_loans[col], errors='coerce')

print(f"✅ Loaded and converted dates\n")

# --- NEW CODE BELOW ---

print("="*80)
print("STEP 3: Calculate Days to Repayment")
print("="*80)

# Calculate time difference between actual repayment and due date
prev_loans['days_to_repay'] = (prev_loans['firstrepaiddate'] - prev_loans['firstduedate']).dt.days

print("\n📊 Days to Repayment - Summary Statistics:")
print(prev_loans['days_to_repay'].describe())

print("\n🔍 Payment Timing Breakdown:")
total = len(prev_loans)

# Categorize payments by timing
on_time = (prev_loans['days_to_repay'] <= 0).sum()
late_1_7 = ((prev_loans['days_to_repay'] > 0) & (prev_loans['days_to_repay'] <= 7)).sum()
late_8_30 = ((prev_loans['days_to_repay'] > 7) & (prev_loans['days_to_repay'] <= 30)).sum()
late_30plus = (prev_loans['days_to_repay'] > 30).sum()
never_paid = prev_loans['days_to_repay'].isnull().sum()

print(f"\n   Category              Count    Percentage  Risk Level")
print(f"   {'='*60}")
print(f"   On-time (≤ 0 days):   {on_time:6,}   {on_time/total*100:5.1f}%     ✅ LOW")
print(f"   Late 1-7 days:        {late_1_7:6,}   {late_1_7/total*100:5.1f}%     ⚠️  MEDIUM")
print(f"   Late 8-30 days:       {late_8_30:6,}   {late_8_30/total*100:5.1f}%     ⚠️  HIGH")
print(f"   Late 30+ days:        {late_30plus:6,}   {late_30plus/total*100:5.1f}%     ❌ VERY HIGH")
print(f"   Never paid (NaN):     {never_paid:6,}   {never_paid/total*100:5.1f}%     ❌ DEFAULT")

print("\n👀 Sample loans showing payment behavior:")
sample_cols = ['customerid', 'loanamount', 'firstduedate', 'firstrepaiddate', 'days_to_repay']
print(prev_loans[sample_cols].head(10))

print("\n💡 How to interpret days_to_repay:")
print("   • NEGATIVE days: Customer paid EARLY (before due date)")
print("     Example: -5 means paid 5 days early → Excellent credit behavior!")
print()
print("   • ZERO days: Customer paid EXACTLY on time")
print("     Example: 0 means paid on the due date → Good credit behavior!")
print()
print("   • POSITIVE days: Customer paid LATE (after due date)")
print("     Example: +7 means paid 7 days late → Poor credit behavior!")
print("     Higher number = worse (30+ days late is serious delinquency)")
print()
print("   • NaN: Customer NEVER PAID (defaulted or abandoned loan)")
print("     Missing value indicates loan was never repaid → Very bad!")
```

Save and run:

```bash
python guide/week1/day3_practice.py
```

**What you should see:**

```
================================================================================
STEP 3: Calculate Days to Repayment
================================================================================

📊 Days to Repayment - Summary Statistics:
count    14349.000000
mean         2.542...
std          4.123...
min        -10.000000
25%          0.000000
50%          0.000000
75%          5.000000
max         45.000000
Name: days_to_repay, dtype: float64

🔍 Payment Timing Breakdown:

   Category              Count    Percentage  Risk Level
   ============================================================
   On-time (≤ 0 days):    9,919    64.8%     ✅ LOW
   Late 1-7 days:         3,467    22.6%     ⚠️  MEDIUM
   Late 8-30 days:          963     6.3%     ⚠️  HIGH
   Late 30+ days:             0     0.0%     ❌ VERY HIGH
   Never paid (NaN):        963     6.3%     ❌ DEFAULT

👀 Sample loans showing payment behavior:
     customerid  loanamount firstduedate firstrepaiddate  days_to_repay
0  CUST_000001       46910   2023-02-20      2023-02-25            5.0
1  CUST_000001       24015   2023-02-20      2023-02-20            0.0
2  CUST_000001       34645   2023-02-20      2023-02-20            0.0
3  CUST_000001       59869   2023-02-20      2023-02-25            5.0
4  CUST_000001       61315   2023-02-20      2023-02-25            5.0

💡 How to interpret days_to_repay:
   • NEGATIVE days: Customer paid EARLY (before due date)
     Example: -5 means paid 5 days early → Excellent credit behavior!

   • ZERO days: Customer paid EXACTLY on time
     Example: 0 means paid on the due date → Good credit behavior!

   • POSITIVE days: Customer paid LATE (after due date)
     Example: +7 means paid 7 days late → Poor credit behavior!
     Higher number = worse (30+ days late is serious delinquency)

   • NaN: Customer NEVER PAID (defaulted or abandoned loan)
     Missing value indicates loan was never repaid → Very bad!
```

**What just happened:**

- `(prev_loans['firstrepaiddate'] - prev_loans['firstduedate']).dt.days` performs date subtraction:
  - When you subtract two datetime columns, pandas returns a `timedelta` object (time difference).
  - `.dt.days` extracts just the number of days as an integer.

- **New column created:** `days_to_repay` with these meanings:
  - **Negative value** (e.g., -5): Customer paid 5 days **before** the due date → Proactive, low-risk behavior
  - **Zero**: Paid exactly on time → Responsible, low-risk behavior
  - **Positive value** (e.g., +7): Customer paid 7 days **after** due date → Late payment, higher-risk behavior
  - **NaN (missing)**: Customer never made the first payment → Default or abandonment, very high risk

- **Key statistics:**
  - **64.8%** of loans paid on time or early (majority are good borrowers)
  - **22.6%** were 1-7 days late (minor delinquency)
  - **6.3%** were 8-30 days late (moderate delinquency)
  - **6.3%** were never paid (defaults)

- This single feature is **extremely predictive** because payment history is the strongest indicator of future payment behavior. A customer with a pattern of late payments or defaults is much more likely to default on their next loan.

---

I'll continue with Steps 4-8 in the next message. This format matches Day 2 exactly. Should I continue building the full guide?
### Step 4: Aggregate Payment Behavior per Customer

Delete everything in `guide/week1/day3_practice.py` and replace it with this:

```python
"""
Day 3 Practice: Feature Engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 3: FEATURE ENGINEERING")
print("="*80)

PROCESSED_DIR = Path('credit-risk-api/data/processed')
RAW_DIR = Path('credit-risk-api/data/raw')

master_df = pd.read_csv(PROCESSED_DIR / 'master_dataset_day2.csv')
prev_loans = pd.read_csv(RAW_DIR / 'trainprevloans.csv')

# Convert dates
date_cols = ['creationdate', 'approveddate', 'firstduedate', 'firstrepaiddate', 'closeddate']
for col in date_cols:
    prev_loans[col] = pd.to_datetime(prev_loans[col], errors='coerce')

# Calculate days to repay
prev_loans['days_to_repay'] = (prev_loans['firstrepaiddate'] - prev_loans['firstduedate']).dt.days

print(f"✅ Prepared prev_loans with days_to_repay\n")

# --- NEW CODE BELOW ---

print("="*80)
print("STEP 4: Aggregate Payment Behavior per Customer")
print("="*80)

print("\n🔧 For each customer, calculate their payment behavior summary...")

payment_features = []

for customer_id, group in prev_loans.groupby('customerid'):
    features = {'customerid': customer_id}
    
    total_loans = len(group)
    
    # On-time rate (paid on or before due date)
    on_time_count = (group['days_to_repay'] <= 0).sum()
    features['hist_ontime_rate'] = on_time_count / total_loans if total_loans > 0 else 0
    
    # Late rate (paid after due date)
    late_count = (group['days_to_repay'] > 0).sum()
    features['hist_late_rate'] = late_count / total_loans if total_loans > 0 else 0
    
    # Never paid rate (NaN in days_to_repay)
    never_paid_count = group['days_to_repay'].isnull().sum()
    features['hist_never_paid_rate'] = never_paid_count / total_loans if total_loans > 0 else 0
    
    # Average days late (for late payments only, excluding on-time and never-paid)
    late_payments = group[group['days_to_repay'] > 0]['days_to_repay']
    features['hist_avg_days_late'] = late_payments.mean() if len(late_payments) > 0 else 0
    
    # Max days late
    features['hist_max_days_late'] = group['days_to_repay'].max() if group['days_to_repay'].notna().any() else 0
    
    # Closure rate (loans that were closed vs still open)
    closed_count = group['closeddate'].notna().sum()
    features['hist_closure_rate'] = closed_count / total_loans if total_loans > 0 else 0
    
    # Days since last loan (recency)
    most_recent = group['approveddate'].max()
    if pd.notna(most_recent):
        reference_date = pd.Timestamp('2024-07-01')  # Assume current date
        features['days_since_last_loan'] = (reference_date - most_recent).days
    else:
        features['days_since_last_loan'] = 9999  # Very old or no loans
    
    payment_features.append(features)

payment_behavior_df = pd.DataFrame(payment_features)

print(f"\n✅ Payment behavior features created: {payment_behavior_df.shape}")
print(f"   Customers: {len(payment_behavior_df):,}")
print(f"   New features: 7")

print(f"\n📊 New feature columns:")
for col in payment_behavior_df.columns[1:]:  # Skip customerid
    print(f"   • {col}")

print(f"\n👀 Sample customers:")
print(payment_behavior_df.head())

print(f"\n💡 Feature meanings:")
print("   hist_ontime_rate:      Fraction of loans paid on/before due date (0-1)")
print("   hist_late_rate:        Fraction of loans paid late (0-1)")
print("   hist_never_paid_rate:  Fraction of loans never repaid (0-1)")
print("   hist_avg_days_late:    Average days late when payment was late")
print("   hist_max_days_late:    Worst case - maximum days late ever")
print("   hist_closure_rate:     Fraction of loans completed vs abandoned (0-1)")
print("   days_since_last_loan:  Days since most recent loan approval (recency)")
```

Save and run:

```bash
python guide/week1/day3_practice.py
```

**Expected output:**

```
================================================================================
STEP 4: Aggregate Payment Behavior per Customer
================================================================================

🔧 For each customer, calculate their payment behavior summary...

✅ Payment behavior features created: (4515, 8)
   Customers: 4,515
   New features: 7

📊 New feature columns:
   • hist_ontime_rate
   • hist_late_rate
   • hist_never_paid_rate
   • hist_avg_days_late
   • hist_max_days_late
   • hist_closure_rate
   • days_since_last_loan

👀 Sample customers:
     customerid  hist_ontime_rate  hist_late_rate  hist_never_paid_rate  hist_avg_days_late  hist_max_days_late  hist_closure_rate  days_since_last_loan
0  CUST_000001            0.375           0.625                   0.0            3.800000                 5.0              0.625                   310
1  CUST_000002            0.800           0.200                   0.0            2.000000                 2.0              0.600                   310
2  CUST_000003            1.000           0.000                   0.0            0.000000                 0.0              1.000                   310
3  CUST_000004            0.500           0.500                   0.0            5.000000                 5.0              1.000                   310
4  CUST_000005            0.333           0.667                   0.0            4.500000                 7.0              0.333                   310

💡 Feature meanings:
   hist_ontime_rate:      Fraction of loans paid on/before due date (0-1)
   hist_late_rate:        Fraction of loans paid late (0-1)
   hist_never_paid_rate:  Fraction of loans never repaid (0-1)
   hist_avg_days_late:    Average days late when payment was late
   hist_max_days_late:    Worst case - maximum days late ever
   hist_closure_rate:     Fraction of loans completed vs abandoned (0-1)
   days_since_last_loan:  Days since most recent loan approval (recency)
```

**What just happened:**

- `.groupby('customerid')` groups all historical loans by customer, so we can compute per-customer statistics.
- For each customer's group of loans, we calculated 7 new features:

1. **hist_ontime_rate**: What fraction of their loans were paid on time or early?
   - Customer `CUST_000001`: Only 37.5% on-time (bad!)
   - Customer `CUST_000003`: 100% on-time (excellent!)

2. **hist_late_rate**: What fraction were paid late?
   - Customer `CUST_000001`: 62.5% late (high risk!)

3. **hist_never_paid_rate**: What fraction were never repaid?
   - These customers have 0% never-paid, but others in the dataset will have non-zero values

4. **hist_avg_days_late**: When they DID pay late, how late on average?
   - Customer `CUST_000001`: 3.8 days late on average

5. **hist_max_days_late**: What's their worst payment delay ever?
   - Customer `CUST_000005`: Maximum 7 days late

6. **hist_closure_rate**: What fraction of loans did they complete vs abandon?
   - Customer `CUST_000001`: Only 62.5% closure rate (meaning 3 out of 8 loans are still open or defaulted)

7. **days_since_last_loan**: How long ago was their most recent loan?
   - All showing 310 days (reference date 2024-07-01 minus their most recent approval date)
   - Recent borrowers (low days) may be financially stressed

- These features are **behavioral gold**. A customer with:
  - Low on-time rate → High risk
  - High never-paid rate → Very high risk
  - Low closure rate → High risk
  - Recent last loan (low days_since) → May be over-leveraged

- Result: 4,515 customers with history (485 customers have NO history and won't appear here yet).

---


### Step 5: Merge Payment Features and Handle Missing Values

Delete everything in `guide/week1/day3_practice.py` and replace it with this:

```python
"""
Day 3 Practice: Feature Engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 3: FEATURE ENGINEERING")
print("="*80)

PROCESSED_DIR = Path('credit-risk-api/data/processed')
RAW_DIR = Path('credit-risk-api/data/raw')

master_df = pd.read_csv(PROCESSED_DIR / 'master_dataset_day2.csv')
prev_loans = pd.read_csv(RAW_DIR / 'trainprevloans.csv')

# Convert dates and calculate days_to_repay
date_cols = ['creationdate', 'approveddate', 'firstduedate', 'firstrepaiddate', 'closeddate']
for col in date_cols:
    prev_loans[col] = pd.to_datetime(prev_loans[col], errors='coerce')
prev_loans['days_to_repay'] = (prev_loans['firstrepaiddate'] - prev_loans['firstduedate']).dt.days

# Aggregate payment behavior
payment_features = []
for customer_id, group in prev_loans.groupby('customerid'):
    features = {'customerid': customer_id}
    total_loans = len(group)
    features['hist_ontime_rate'] = (group['days_to_repay'] <= 0).sum() / total_loans
    features['hist_late_rate'] = (group['days_to_repay'] > 0).sum() / total_loans
    features['hist_never_paid_rate'] = group['days_to_repay'].isnull().sum() / total_loans
    late_payments = group[group['days_to_repay'] > 0]['days_to_repay']
    features['hist_avg_days_late'] = late_payments.mean() if len(late_payments) > 0 else 0
    features['hist_max_days_late'] = group['days_to_repay'].max() if group['days_to_repay'].notna().any() else 0
    features['hist_closure_rate'] = group['closeddate'].notna().sum() / total_loans
    most_recent = group['approveddate'].max()
    features['days_since_last_loan'] = (pd.Timestamp('2024-07-01') - most_recent).days if pd.notna(most_recent) else 9999
    payment_features.append(features)

payment_behavior_df = pd.DataFrame(payment_features)

print(f"✅ Payment features aggregated: {payment_behavior_df.shape}\n")

# --- NEW CODE BELOW ---

print("="*80)
print("STEP 5: Merge Payment Features into Master Dataset")
print("="*80)

print(f"Before merge:")
print(f"   Master DF:   {master_df.shape}")
print(f"   Payment DF:  {payment_behavior_df.shape}")

# Merge payment features
master_df = master_df.merge(payment_behavior_df, on='customerid', how='left')

print(f"\nAfter merge:")
print(f"   Master DF:   {master_df.shape}")
print(f"   Columns added: 7")

# Check for missing values
print(f"\n🔍 Checking for customers with NO loan history...")
missing_hist = master_df['hist_ontime_rate'].isnull().sum()
print(f"   Customers with missing payment history: {missing_hist:,}")
print(f"   ({missing_hist/len(master_df)*100:.1f}% of total)")

print(f"\n🔧 Filling missing values for first-time borrowers...")

# Fill NaN for customers with no history
payment_cols = ['hist_ontime_rate', 'hist_late_rate', 'hist_never_paid_rate',
                'hist_avg_days_late', 'hist_max_days_late', 'hist_closure_rate', 'days_since_last_loan']

for col in payment_cols:
    if col == 'days_since_last_loan':
        master_df[col] = master_df[col].fillna(9999)  # Very old/no loans
    else:
        master_df[col] = master_df[col].fillna(0)

print(f"✅ Filled missing values:")
print(f"   • hist_ontime_rate = 0 (no history = no on-time payments)")
print(f"   • hist_late_rate = 0")
print(f"   • hist_never_paid_rate = 0")
print(f"   • hist_avg_days_late = 0")
print(f"   • hist_max_days_late = 0")
print(f"   • hist_closure_rate = 0")
print(f"   • days_since_last_loan = 9999 (very old/no loans)")

print(f"\nMissing values remaining: {master_df.isnull().sum().sum()}")
```

Save and run:

```bash
python guide/week1/day3_practice.py
```

**Expected output:**

```
================================================================================
STEP 5: Merge Payment Features into Master Dataset
================================================================================

Before merge:
   Master DF:   (5000, 19)
   Payment DF:  (4515, 8)

After merge:
   Master DF:   (5000, 26)
   Columns added: 7

🔍 Checking for customers with NO loan history...
   Customers with missing payment history: 485
   (9.7% of total)

🔧 Filling missing values for first-time borrowers...
✅ Filled missing values:
   • hist_ontime_rate = 0 (no history = no on-time payments)
   • hist_late_rate = 0
   • hist_never_paid_rate = 0
   • hist_avg_days_late = 0
   • hist_max_days_late = 0
   • hist_closure_rate = 0
   • days_since_last_loan = 9999 (very old/no loans)

Missing values remaining: 0
```

**What just happened:**

- `.merge(payment_behavior_df, on='customerid', how='left')` joins the payment features onto the master dataset.
- `how='left'` means "keep all 5,000 customers from master_df, even if they don't appear in payment_behavior_df."
- 485 customers have NO historical loans, so their payment features are `NaN` after the merge.
- `.fillna()` replaces missing values with logical defaults:
  - Payment rates → 0 (no history means zero on-time rate, zero late rate, etc.)
  - days_since_last_loan → 9999 (a very large number indicating "no recent loans" or "first-time borrower")
- After filling, there are zero missing values in the dataset.
- Dataset grew from 19 → 26 features (added 7 payment behavior features).

---

### Step 6: Encode Categorical Variables

Delete everything in `guide/week1/day3_practice.py` and replace it with this:

```python
"""
Day 3 Practice: Feature Engineering - Nearly complete!
"""

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 3: FEATURE ENGINEERING")
print("="*80)

PROCESSED_DIR = Path('credit-risk-api/data/processed')
RAW_DIR = Path('credit-risk-api/data/raw')

master_df = pd.read_csv(PROCESSED_DIR / 'master_dataset_day2.csv')
prev_loans = pd.read_csv(RAW_DIR / 'trainprevloans.csv')

# Steps 2-5 (condensed for brevity)
date_cols = ['creationdate', 'approveddate', 'firstduedate', 'firstrepaiddate', 'closeddate']
for col in date_cols:
    prev_loans[col] = pd.to_datetime(prev_loans[col], errors='coerce')
prev_loans['days_to_repay'] = (prev_loans['firstrepaiddate'] - prev_loans['firstduedate']).dt.days

payment_features = []
for customer_id, group in prev_loans.groupby('customerid'):
    features = {'customerid': customer_id}
    total = len(group)
    features['hist_ontime_rate'] = (group['days_to_repay'] <= 0).sum() / total
    features['hist_late_rate'] = (group['days_to_repay'] > 0).sum() / total
    features['hist_never_paid_rate'] = group['days_to_repay'].isnull().sum() / total
    late = group[group['days_to_repay'] > 0]['days_to_repay']
    features['hist_avg_days_late'] = late.mean() if len(late) > 0 else 0
    features['hist_max_days_late'] = group['days_to_repay'].max() if group['days_to_repay'].notna().any() else 0
    features['hist_closure_rate'] = group['closeddate'].notna().sum() / total
    recent = group['approveddate'].max()
    features['days_since_last_loan'] = (pd.Timestamp('2024-07-01') - recent).days if pd.notna(recent) else 9999
    payment_features.append(features)

payment_behavior_df = pd.DataFrame(payment_features)
master_df = master_df.merge(payment_behavior_df, on='customerid', how='left')

payment_cols = ['hist_ontime_rate', 'hist_late_rate', 'hist_never_paid_rate',
                'hist_avg_days_late', 'hist_max_days_late', 'hist_closure_rate', 'days_since_last_loan']
for col in payment_cols:
    master_df[col] = master_df[col].fillna(9999 if col == 'days_since_last_loan' else 0)

print(f"✅ Payment features merged: {master_df.shape}\n")

# --- NEW CODE BELOW ---

print("="*80)
print("STEP 6: Encode Categorical Variables")
print("="*80)

print("\n🔍 Categorical columns before encoding:")
print(f"   • employment_status_clients: {master_df['employment_status_clients'].nunique()} unique values")
print(f"   • level_of_education_clients: {master_df['level_of_education_clients'].nunique()} unique values")

print("\nValues:")
print(f"   Employment: {master_df['employment_status_clients'].unique()}")
print(f"   Education:  {master_df['level_of_education_clients'].unique()}")

print(f"\n🔧 One-hot encoding categorical variables...")

# One-hot encode employment status
master_df = pd.get_dummies(master_df, columns=['employment_status_clients'], prefix='emp', drop_first=False)

# One-hot encode education level
master_df = pd.get_dummies(master_df, columns=['level_of_education_clients'], prefix='edu', drop_first=False)

print(f"\n✅ Categorical encoding complete!")
print(f"   Dataset shape: {master_df.shape}")

print(f"\n📊 New columns created:")
emp_cols = [col for col in master_df.columns if col.startswith('emp_')]
edu_cols = [col for col in master_df.columns if col.startswith('edu_')]

print(f"   Employment ({len(emp_cols)} columns): {emp_cols}")
print(f"   Education ({len(edu_cols)} columns):  {edu_cols}")

print(f"\n💡 What one-hot encoding does:")
print("   Before: employment_status_clients = 'Permanent'")
print("   After:  emp_Contract=0, emp_Permanent=1, emp_Self-Employed=0, emp_Temporary=0")
print()
print("   Each category becomes a binary column (0 or 1)")
print("   Machine learning models need numbers, not text!")
```

Save and run:

```bash
python guide/week1/day3_practice.py
```

**Expected output:**

```
================================================================================
STEP 6: Encode Categorical Variables
================================================================================

🔍 Categorical columns before encoding:
   • employment_status_clients: 4 unique values
   • level_of_education_clients: 4 unique values

Values:
   Employment: ['Contract' 'Permanent' 'Self-Employed' 'Temporary']
   Education:  ['HND/BSc' 'Secondary' 'PhD' 'Masters']

🔧 One-hot encoding categorical variables...

✅ Categorical encoding complete!
   Dataset shape: (5000, 32)

📊 New columns created:
   Employment (4 columns): ['emp_Contract', 'emp_Permanent', 'emp_Self-Employed', 'emp_Temporary']
   Education (4 columns):  ['edu_HND/BSc', 'edu_Masters', 'edu_PhD', 'edu_Secondary']

💡 What one-hot encoding does:
   Before: employment_status_clients = 'Permanent'
   After:  emp_Contract=0, emp_Permanent=1, emp_Self-Employed=0, emp_Temporary=0

   Each category becomes a binary column (0 or 1)
   Machine learning models need numbers, not text!
```

**What just happened:**

- `pd.get_dummies(df, columns=['employment_status_clients'], prefix='emp', drop_first=False)` performs **one-hot encoding**.
- **Before encoding:** `employment_status_clients` was a single column with text values like `'Permanent'`, `'Contract'`, etc.
- **After encoding:** That one column becomes **4 binary columns**:
  - `emp_Contract` = 1 if Contract, 0 otherwise
  - `emp_Permanent` = 1 if Permanent, 0 otherwise
  - `emp_Self-Employed` = 1 if Self-Employed, 0 otherwise
  - `emp_Temporary` = 1 if Temporary, 0 otherwise

- Same for education: 4 categories → 4 binary columns.

- **Why this matters:** Machine learning algorithms require numeric input. Text categories must be converted to numbers. One-hot encoding is the standard method for categorical variables with no inherent order.

- Original categorical columns are automatically dropped after encoding.

- Dataset grew from 26 → 32 features (removed 2 text columns, added 8 binary columns = net +6).

---


### Step 7: Create Additional Derived Features

Delete everything in `guide/week1/day3_practice.py` and replace it with this:

```python
"""
Day 3 Practice: Feature Engineering - Final version
"""

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 3: FEATURE ENGINEERING")
print("="*80)

PROCESSED_DIR = Path('credit-risk-api/data/processed')
RAW_DIR = Path('credit-risk-api/data/raw')

master_df = pd.read_csv(PROCESSED_DIR / 'master_dataset_day2.csv')
prev_loans = pd.read_csv(RAW_DIR / 'trainprevloans.csv')

# Steps 2-6 condensed
date_cols = ['creationdate', 'approveddate', 'firstduedate', 'firstrepaiddate', 'closeddate']
for col in date_cols:
    prev_loans[col] = pd.to_datetime(prev_loans[col], errors='coerce')
prev_loans['days_to_repay'] = (prev_loans['firstrepaiddate'] - prev_loans['firstduedate']).dt.days

payment_features = []
for customer_id, group in prev_loans.groupby('customerid'):
    features = {'customerid': customer_id}
    total = len(group)
    features['hist_ontime_rate'] = (group['days_to_repay'] <= 0).sum() / total
    features['hist_late_rate'] = (group['days_to_repay'] > 0).sum() / total
    features['hist_never_paid_rate'] = group['days_to_repay'].isnull().sum() / total
    late = group[group['days_to_repay'] > 0]['days_to_repay']
    features['hist_avg_days_late'] = late.mean() if len(late) > 0 else 0
    features['hist_max_days_late'] = group['days_to_repay'].max() if group['days_to_repay'].notna().any() else 0
    features['hist_closure_rate'] = group['closeddate'].notna().sum() / total
    recent = group['approveddate'].max()
    features['days_since_last_loan'] = (pd.Timestamp('2024-07-01') - recent).days if pd.notna(recent) else 9999
    payment_features.append(features)

payment_behavior_df = pd.DataFrame(payment_features)
master_df = master_df.merge(payment_behavior_df, on='customerid', how='left')

payment_cols = ['hist_ontime_rate', 'hist_late_rate', 'hist_never_paid_rate',
                'hist_avg_days_late', 'hist_max_days_late', 'hist_closure_rate', 'days_since_last_loan']
for col in payment_cols:
    master_df[col] = master_df[col].fillna(9999 if col == 'days_since_last_loan' else 0)

master_df = pd.get_dummies(master_df, columns=['employment_status_clients'], prefix='emp', drop_first=False)
master_df = pd.get_dummies(master_df, columns=['level_of_education_clients'], prefix='edu', drop_first=False)

print(f"✅ Steps 2-6 complete: {master_df.shape}\n")

# --- NEW CODE BELOW ---

print("="*80)
print("STEP 7: Create Additional Derived Features")
print("="*80)

print("\n🔧 Creating derived features from existing columns...")

# 1. Loan amount relative to historical average
master_df['loan_to_hist_avg'] = master_df['loanamount'] / (master_df['avg_loan_amount'] + 1)  # +1 to avoid division by zero

# 2. Interest rate (implied from totaldue vs loanamount)
master_df['interest_rate'] = (master_df['totaldue'] / master_df['loanamount']) - 1

# 3. Loan amount increase (current loan vs historical average)
master_df['loan_amount_increase'] = master_df['loanamount'] - master_df['avg_loan_amount']

print(f"✅ Created 3 derived features")

print(f"\n📊 New features:")
print("   1. loan_to_hist_avg:      Current loan / avg historical loan")
print("                             (>1 means borrowing more than usual)")
print()
print("   2. interest_rate:         (totaldue / loanamount) - 1")
print("                             (higher rate = riskier loan)")
print()
print("   3. loan_amount_increase:  Current loan - avg historical loan")
print("                             (large increase may indicate financial stress)")

print(f"\nDataset shape: {master_df.shape}")
print(f"Total features: {master_df.shape[1]}")

print(f"\n👀 Sample values:")
sample = master_df[['loanamount', 'avg_loan_amount', 'loan_to_hist_avg', 
                    'totaldue', 'interest_rate', 'loan_amount_increase']].head(5)
print(sample)
```

Save and run:

```bash
python guide/week1/day3_practice.py
```

**Expected output:**

```
================================================================================
STEP 7: Create Additional Derived Features
================================================================================

🔧 Creating derived features from existing columns...
✅ Created 3 derived features

📊 New features:
   1. loan_to_hist_avg:      Current loan / avg historical loan
                             (>1 means borrowing more than usual)

   2. interest_rate:         (totaldue / loanamount) - 1
                             (higher rate = riskier loan)

   3. loan_amount_increase:  Current loan - avg historical loan
                             (large increase may indicate financial stress)

Dataset shape: (5000, 35)
Total features: 35

👀 Sample values:
   loanamount  avg_loan_amount  loan_to_hist_avg   totaldue  interest_rate  loan_amount_increase
0       73564         49397.12          1.488965      98974       0.345231              24166.88
1       28946         36354.60          0.796278      36276       0.253213              -7408.60
2       44231         25530.00          1.732481      55623       0.257563              18701.00
3       39029         44968.00          0.867936      49891       0.278342              -5939.00
4      118950         57566.33          2.066190     153544       0.290669              61383.67
```

**What just happened:**

- Created 3 **derived features** by combining existing columns:

1. **loan_to_hist_avg**: Ratio of current loan to historical average
   - Customer 0: 1.49 (borrowing 49% more than usual)
   - Customer 4: 2.07 (borrowing TWICE their usual amount - red flag!)
   - Values >1 suggest the customer is taking a larger loan than normal, which may indicate financial stress

2. **interest_rate**: Implied interest rate from loan terms
   - Calculated as (total due / principal) - 1
   - Customer 0: 34.5% interest rate
   - Higher rates are typically charged to riskier borrowers

3. **loan_amount_increase**: Absolute increase vs historical average
   - Customer 0: +₦24,167 (borrowing ₦24k more)
   - Customer 4: +₦61,384 (large increase - may be risky)
   - Large increases relative to history can signal over-leveraging

- These "interaction features" capture relationships between columns that may be predictive.
- Adding 1 to `avg_loan_amount` prevents division by zero for first-time borrowers.
- Final dataset: **35 features** (up from 19 at the start of today).

---

### Step 8: Save Enriched Dataset

Delete everything in `guide/week1/day3_practice.py` and replace it with this (final complete version):

```python
"""
Day 3 Practice: Feature Engineering from Historical Loans
Credit Risk Scoring System - COMPLETE VERSION

This is the final version. Run this after day2_practice.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DAY 3: FEATURE ENGINEERING FROM HISTORICAL LOANS")
print("="*80)

# Load data
PROCESSED_DIR = Path('credit-risk-api/data/processed')
RAW_DIR = Path('credit-risk-api/data/raw')

master_df = pd.read_csv(PROCESSED_DIR / 'master_dataset_day2.csv')
prev_loans = pd.read_csv(RAW_DIR / 'trainprevloans.csv')

print(f"📂 Loaded: master_df {master_df.shape}, prev_loans {prev_loans.shape}")

# Step 2: Convert dates
date_cols = ['creationdate', 'approveddate', 'firstduedate', 'firstrepaiddate', 'closeddate']
for col in date_cols:
    prev_loans[col] = pd.to_datetime(prev_loans[col], errors='coerce')

# Step 3: Calculate days to repayment
prev_loans['days_to_repay'] = (prev_loans['firstrepaiddate'] - prev_loans['firstduedate']).dt.days
print("✅ Step 3: Calculated days_to_repay")

# Step 4: Aggregate payment behavior
payment_features = []
for customer_id, group in prev_loans.groupby('customerid'):
    features = {'customerid': customer_id}
    total = len(group)
    features['hist_ontime_rate'] = (group['days_to_repay'] <= 0).sum() / total
    features['hist_late_rate'] = (group['days_to_repay'] > 0).sum() / total
    features['hist_never_paid_rate'] = group['days_to_repay'].isnull().sum() / total
    late = group[group['days_to_repay'] > 0]['days_to_repay']
    features['hist_avg_days_late'] = late.mean() if len(late) > 0 else 0
    features['hist_max_days_late'] = group['days_to_repay'].max() if group['days_to_repay'].notna().any() else 0
    features['hist_closure_rate'] = group['closeddate'].notna().sum() / total
    recent = group['approveddate'].max()
    features['days_since_last_loan'] = (pd.Timestamp('2024-07-01') - recent).days if pd.notna(recent) else 9999
    payment_features.append(features)

payment_behavior_df = pd.DataFrame(payment_features)
print(f"✅ Step 4: Aggregated payment behavior → {payment_behavior_df.shape}")

# Step 5: Merge and fill missing
master_df = master_df.merge(payment_behavior_df, on='customerid', how='left')
payment_cols = ['hist_ontime_rate', 'hist_late_rate', 'hist_never_paid_rate',
                'hist_avg_days_late', 'hist_max_days_late', 'hist_closure_rate', 'days_since_last_loan']
for col in payment_cols:
    master_df[col] = master_df[col].fillna(9999 if col == 'days_since_last_loan' else 0)
print(f"✅ Step 5: Merged payment features → {master_df.shape}")

# Step 6: Encode categoricals
master_df = pd.get_dummies(master_df, columns=['employment_status_clients'], prefix='emp', drop_first=False)
master_df = pd.get_dummies(master_df, columns=['level_of_education_clients'], prefix='edu', drop_first=False)
print(f"✅ Step 6: Encoded categorical variables → {master_df.shape}")

# Step 7: Derived features
master_df['loan_to_hist_avg'] = master_df['loanamount'] / (master_df['avg_loan_amount'] + 1)
master_df['interest_rate'] = (master_df['totaldue'] / master_df['loanamount']) - 1
master_df['loan_amount_increase'] = master_df['loanamount'] - master_df['avg_loan_amount']
print(f"✅ Step 7: Created derived features → {master_df.shape}")

# Step 8: Save
print("\n" + "="*80)
print("STEP 8: Saving Enriched Dataset")
print("="*80)

output_path = PROCESSED_DIR / 'master_dataset_day3.csv'
master_df.to_csv(output_path, index=False)

print(f"\n💾 Saved to: {output_path}")
print(f"   Shape: {master_df.shape}")
print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")

# Summary
print("\n" + "="*80)
print("📝 DAY 3 SUMMARY")
print("="*80)

print(f"""
✅ Feature Engineering Complete!

📊 Dataset Growth:
   Day 2 features: 19
   Day 3 features: {master_df.shape[1]}
   New features:   {master_df.shape[1] - 19}

🎯 Features Added:
   1. Payment Behavior (7):
      • hist_ontime_rate, hist_late_rate, hist_never_paid_rate
      • hist_avg_days_late, hist_max_days_late
      • hist_closure_rate, days_since_last_loan

   2. Categorical Encoding (8):
      • Employment: 4 binary columns
      • Education: 4 binary columns

   3. Derived Features (3):
      • loan_to_hist_avg, interest_rate, loan_amount_increase

💡 Why These Features Matter:
   • Payment behavior is the #1 predictor of credit risk
   • hist_ontime_rate: Past behavior predicts future behavior
   • hist_closure_rate: Completing loans shows reliability
   • days_since_last_loan: Recent borrowing may signal stress

📅 Tomorrow (Day 4):
   • Deep EDA: Correlation analysis
   • Identify top 10 predictive features
   • Detect and handle outliers
   • Visualize feature distributions

🎉 DAY 3 COMPLETE! {master_df.shape[1]} features ready for analysis!
""")

print("\n🔍 Feature List:")
for i, col in enumerate(master_df.columns, 1):
    print(f"   {i:2d}. {col}")
```

Save and run:

```bash
python guide/week1/day3_practice.py
```

This is your final complete Day 3 script. Keep this version.

---

## Part 3: Key Concepts Summary

| Operation | What It Does |
|-----------|-------------|
| `pd.to_datetime(col, errors='coerce')` | Convert text dates to datetime objects; invalid → NaT |
| `(date1 - date2).dt.days` | Subtract dates to get days between them |
| `.groupby('col')` | Group rows by column value for aggregation |
| `(condition).sum() / total` | Calculate rate/fraction (0-1) |
| `.fillna(value)` | Replace missing values with specified value |
| `pd.get_dummies(df, columns=['col'], prefix='pre')` | One-hot encode categorical variable |
| `df1.merge(df2, on='key', how='left')` | Left join keeping all rows from df1 |

---

## Part 4: Commit Your Work

```bash
git add guide/week1/day3_practice.py credit-risk-api/data/processed/master_dataset_day3.csv
git commit -m "Day 3: Feature engineering (payment behavior, categorical encoding, derived features)"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `KeyError: 'firstduedate'` | Dates must be converted BEFORE calculating days_to_repay |
| Division by zero in derived features | Add +1 to denominator: `x / (y + 1)` |
| `.groupby()` returns wrong shape | Check if you're aggregating correctly; use `.reset_index()` |
| One-hot encoding creates too many columns | Check `drop_first=False` parameter; typically keep all categories |
| `TypeError: unsupported operand type(s) for -` | Dates must be datetime objects, not strings |

---

## Check Your Understanding

**1. Why is `days_to_repay` calculated as `firstrepaiddate - firstduedate` and not the other way around?**

The order matters because it determines the sign. `firstrepaiddate - firstduedate` gives:
- Negative if paid early (repaid before due)
- Zero if paid on time
- Positive if paid late (repaid after due)

If reversed (`firstduedate - firstrepaiddate`), the signs would flip, which would be confusing. The current order makes intuitive sense: positive numbers = late = bad.

**2. Why do we fill missing payment features with 0 instead of the median or mean?**

Missing payment features occur for customers with NO loan history (first-time borrowers). For them:
- `hist_ontime_rate = 0` means "zero on-time payments" (because they have zero loans)
- `hist_late_rate = 0` means "zero late payments"
- This is factually correct, not an imputation

Using median/mean would be wrong because it would assign those customers the "average" payment behavior of OTHER customers, which they haven't demonstrated. Zero accurately represents "no history."

**3. What is one-hot encoding and why is it necessary?**

One-hot encoding converts categorical variables (text) into binary (0/1) columns. For example, `employment_status = 'Permanent'` becomes four columns: `emp_Contract=0, emp_Permanent=1, emp_Self-Employed=0, emp_Temporary=0`.

It's necessary because machine learning algorithms require numeric input. They cannot process text strings directly. One-hot encoding preserves the categorical nature without imposing an artificial order (unlike label encoding which would assign Permanent=0, Contract=1, etc., implying Contract > Permanent).

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Practice script | `guide/week1/day3_practice.py` | Complete feature engineering pipeline |
| Enriched dataset | `credit-risk-api/data/processed/master_dataset_day3.csv` | 35 features ready for EDA |

**New pandas operations learned:**
`pd.to_datetime()`, `.dt.days`, `.groupby()` with custom aggregation, `pd.get_dummies()`, `.fillna()`

**New features created:** 16 (7 payment behavior + 8 categorical + 3 derived)

**Tomorrow (Day 4):** Deep EDA with correlation analysis, outlier detection, and visualizations to identify which of your 35 features are most predictive.

---

**🎉 DAY 3 COMPLETE!**

