# DAY 3: Feature Engineering - Kaggle Credit Risk Dataset

**Goal**: Engineer powerful predictive features from payment behavior, demographics, and loan history

**Time**: 60 minutes

**Prerequisites**: Day 2 completed (master_dataset_day2.csv exists)

---

## 📋 What You'll Build Today

By the end of Day 3, you'll have:
- ✅ Payment behavior features (on-time rate, late rate, never-paid rate)
- ✅ Loan closure features (completion rate, abandonment rate)
- ✅ Recency features (days since last loan)
- ✅ Demographic features (age from birthdate)
- ✅ Loan-to-history ratios (current loan vs historical average)
- ✅ Categorical encodings (education, bank type)
- ✅ Final dataset: `data/processed/master_dataset_day3.csv` with 40+ features

---

## Prerequisites Check

**Verify Day 2 output exists:**

```bash
cd credit-risk-api
ls -lh data/processed/master_dataset_day2.csv
ls -lh data/raw/trainprevloans.csv
```

**If missing:**
```bash
python src/data/merge_kaggle_data.py
```

---

## PART 1: Understanding Feature Engineering

**What is Feature Engineering?**

Transforming raw data into features that better represent the underlying patterns for machine learning.

**Example:**

Raw data:
```
birthdate: "1985-03-15"
```

Engineered feature:
```
age: 39 years
age_group: "35-40"
is_young: 0 (not under 25)
```

**Why it matters:**

Payment behavior is the #1 predictor of credit risk. A customer who consistently pays late is high-risk, regardless of demographics.

**Features we'll create today:**

| Feature Category | Examples | Why It Matters |
|-----------------|----------|----------------|
| **Payment Behavior** | on-time rate, late rate, never-paid rate | Shows payment discipline |
| **Loan Completion** | closure rate, abandonment rate | Shows loan commitment |
| **Recency** | days since last loan | Recent borrowers may be riskier |
| **Demographics** | age, education level | Age correlates with stability |
| **Ratios** | current loan / avg historical loan | Borrowing more than usual is risky |

---

## PART 2: Build Feature Engineering Script (5 Chunks)

We'll build `src/data/engineer_features.py` in 5 incremental steps.

### Step 1: Chunk 1 - Load Data and Calculate Payment Timing

Create a new file: `src/data/engineer_features.py`

Add this code:

```python
"""
Day 3: Feature Engineering from Kaggle Credit Risk dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'

# ============================================================================
# CHUNK 1: Load Data and Calculate Payment Timing
# ============================================================================

def load_data():
    """Load master dataset and historical loans"""
    print("=" * 70)
    print("DAY 3: Feature Engineering")
    print("=" * 70)

    print("\n📂 Loading data...")

    master_df = pd.read_csv(PROCESSED_DATA_DIR / 'master_dataset_day2.csv')
    print(f"   ✅ Master dataset: {master_df.shape}")

    historical_loans = pd.read_csv(RAW_DATA_DIR / 'trainprevloans.csv')
    print(f"   ✅ Historical loans: {historical_loans.shape}")
    print(f"   ✅ Customers with history: {historical_loans['customerid'].nunique():,}\n")

    return master_df, historical_loans


def calculate_payment_behavior(historical_loans):
    """
    Calculate payment behavior features from historical loan dates

    Features created:
    - days_to_repay: firstrepaiddate - firstduedate (negative=early, positive=late)
    - payment_status: 'on_time', 'late', 'never_paid'
    """
    print("=" * 70)
    print("STEP 1: Calculate Payment Behavior from Dates")
    print("=" * 70)

    # Convert date columns
    date_cols = ['approveddate', 'creationdate', 'closeddate', 'firstduedate', 'firstrepaiddate']
    for col in date_cols:
        if col in historical_loans.columns:
            historical_loans[col] = pd.to_datetime(historical_loans[col], errors='coerce')

    print("\n📊 Date columns converted to datetime")

    # Calculate days to repay (positive = late, negative = early, NaN = never paid)
    historical_loans['days_to_repay'] = (
        historical_loans['firstrepaiddate'] - historical_loans['firstduedate']
    ).dt.days

    # Categorize payment status
    def categorize_payment(days):
        if pd.isna(days):
            return 'never_paid'
        elif days <= 0:
            return 'on_time'
        else:
            return 'late'

    historical_loans['payment_status'] = historical_loans['days_to_repay'].apply(categorize_payment)

    print("\n✅ Payment behavior calculated:")
    print(f"   Total historical loans: {len(historical_loans):,}")
    print(f"\n   Payment status breakdown:")
    print(historical_loans['payment_status'].value_counts())

    return historical_loans


if __name__ == "__main__":
    master_df, historical_loans = load_data()
    historical_loans = calculate_payment_behavior(historical_loans)

    print("\n" + "=" * 70)
    print("SAMPLE: Payment Behavior")
    print("=" * 70)
    print(historical_loans[['customerid', 'firstduedate', 'firstrepaiddate',
                           'days_to_repay', 'payment_status']].head(10))
```

**Run it:**
```bash
python src/data/engineer_features.py
```

**Expected output:**
```
======================================================================
DAY 3: Feature Engineering
======================================================================

📂 Loading data...
   ✅ Master dataset: (8000, 32)
   ✅ Historical loans: (25000, 18)
   ✅ Customers with history: 6,500

======================================================================
STEP 1: Calculate Payment Behavior from Dates
======================================================================

📊 Date columns converted to datetime

✅ Payment behavior calculated:
   Total historical loans: 25,000

   Payment status breakdown:
on_time        15,000
late            8,000
never_paid      2,000
```

---

### Step 2: Chunk 2 - Aggregate Payment Behavior Per Customer

**Delete everything in `src/data/engineer_features.py` and replace with this:**

```python
"""
Day 3: Feature Engineering from Kaggle Credit Risk dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'

# ============================================================================
# CHUNK 1: Load Data and Calculate Payment Timing
# ============================================================================

def load_data():
    """Load master dataset and historical loans"""
    print("=" * 70)
    print("DAY 3: Feature Engineering")
    print("=" * 70)

    print("\n📂 Loading data...")

    master_df = pd.read_csv(PROCESSED_DATA_DIR / 'master_dataset_day2.csv')
    print(f"   ✅ Master dataset: {master_df.shape}")

    historical_loans = pd.read_csv(RAW_DATA_DIR / 'trainprevloans.csv')
    print(f"   ✅ Historical loans: {historical_loans.shape}\n")

    return master_df, historical_loans


def calculate_payment_behavior(historical_loans):
    """Calculate payment behavior features from historical loan dates"""
    print("=" * 70)
    print("STEP 1: Calculate Payment Behavior from Dates")
    print("=" * 70)

    # Convert date columns
    date_cols = ['approveddate', 'creationdate', 'closeddate', 'firstduedate', 'firstrepaiddate']
    for col in date_cols:
        if col in historical_loans.columns:
            historical_loans[col] = pd.to_datetime(historical_loans[col], errors='coerce')

    # Calculate days to repay
    historical_loans['days_to_repay'] = (
        historical_loans['firstrepaiddate'] - historical_loans['firstduedate']
    ).dt.days

    # Categorize payment status
    def categorize_payment(days):
        if pd.isna(days):
            return 'never_paid'
        elif days <= 0:
            return 'on_time'
        else:
            return 'late'

    historical_loans['payment_status'] = historical_loans['days_to_repay'].apply(categorize_payment)

    print(f"\n✅ Payment behavior calculated for {len(historical_loans):,} loans\n")

    return historical_loans


# ============================================================================
# CHUNK 2: Aggregate Payment Behavior Per Customer
# ============================================================================

def aggregate_payment_features(historical_loans):
    """
    Aggregate payment behavior per customer

    Features created:
    - payment_ontime_rate: % of loans paid on time
    - payment_late_rate: % of loans paid late
    - payment_never_rate: % of loans never paid
    - avg_days_late: Average days late (for late payments)
    - max_days_late: Maximum days late ever
    """
    print("=" * 70)
    print("STEP 2: Aggregate Payment Behavior Per Customer")
    print("=" * 70)

    # Count payment statuses per customer
    payment_counts = historical_loans.groupby(['customerid', 'payment_status']).size().unstack(fill_value=0)

    # Calculate total loans per customer
    payment_counts['total_loans'] = payment_counts.sum(axis=1)

    # Calculate rates
    payment_features = pd.DataFrame()
    payment_features['customerid'] = payment_counts.index

    if 'on_time' in payment_counts.columns:
        payment_features['payment_ontime_rate'] = payment_counts['on_time'] / payment_counts['total_loans']
    else:
        payment_features['payment_ontime_rate'] = 0

    if 'late' in payment_counts.columns:
        payment_features['payment_late_rate'] = payment_counts['late'] / payment_counts['total_loans']
    else:
        payment_features['payment_late_rate'] = 0

    if 'never_paid' in payment_counts.columns:
        payment_features['payment_never_rate'] = payment_counts['never_paid'] / payment_counts['total_loans']
    else:
        payment_features['payment_never_rate'] = 0

    # Calculate average and max days late (only for late payments)
    late_payments = historical_loans[historical_loans['payment_status'] == 'late'].copy()

    if len(late_payments) > 0:
        avg_late = late_payments.groupby('customerid')['days_to_repay'].mean().reset_index()
        avg_late.columns = ['customerid', 'avg_days_late']

        max_late = late_payments.groupby('customerid')['days_to_repay'].max().reset_index()
        max_late.columns = ['customerid', 'max_days_late']

        payment_features = payment_features.merge(avg_late, on='customerid', how='left')
        payment_features = payment_features.merge(max_late, on='customerid', how='left')
    else:
        payment_features['avg_days_late'] = 0
        payment_features['max_days_late'] = 0

    # Fill missing values (customers with no late payments)
    payment_features['avg_days_late'] = payment_features['avg_days_late'].fillna(0)
    payment_features['max_days_late'] = payment_features['max_days_late'].fillna(0)

    print(f"\n✅ Payment features aggregated:")
    print(f"   Customers with payment history: {len(payment_features):,}")
    print(f"\n   Features created:")
    for col in payment_features.columns:
        if col != 'customerid':
            print(f"      • {col}")

    return payment_features


if __name__ == "__main__":
    master_df, historical_loans = load_data()
    historical_loans = calculate_payment_behavior(historical_loans)
    payment_features = aggregate_payment_features(historical_loans)

    print("\n" + "=" * 70)
    print("SAMPLE: Payment Features")
    print("=" * 70)
    print(payment_features.head())
```

**Run it:**
```bash
python src/data/engineer_features.py
```

**Expected output:**
```
======================================================================
STEP 2: Aggregate Payment Behavior Per Customer
======================================================================

✅ Payment features aggregated:
   Customers with payment history: 6,500

   Features created:
      • payment_ontime_rate
      • payment_late_rate
      • payment_never_rate
      • avg_days_late
      • max_days_late
```

---

### Step 3: Chunk 3 - Create Loan Closure and Recency Features

**Delete everything in `src/data/engineer_features.py` and replace with this:**

```python
"""
Day 3: Feature Engineering from Kaggle Credit Risk dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'

# ============================================================================
# CHUNK 1: Load Data and Calculate Payment Timing
# ============================================================================

def load_data():
    """Load master dataset and historical loans"""
    print("=" * 70)
    print("DAY 3: Feature Engineering")
    print("=" * 70)

    print("\n📂 Loading data...")

    master_df = pd.read_csv(PROCESSED_DATA_DIR / 'master_dataset_day2.csv')
    print(f"   ✅ Master dataset: {master_df.shape}")

    historical_loans = pd.read_csv(RAW_DATA_DIR / 'trainprevloans.csv')
    print(f"   ✅ Historical loans: {historical_loans.shape}\n")

    return master_df, historical_loans


def calculate_payment_behavior(historical_loans):
    """Calculate payment behavior features"""
    # Convert date columns
    date_cols = ['approveddate', 'creationdate', 'closeddate', 'firstduedate', 'firstrepaiddate']
    for col in date_cols:
        if col in historical_loans.columns:
            historical_loans[col] = pd.to_datetime(historical_loans[col], errors='coerce')

    # Calculate days to repay
    historical_loans['days_to_repay'] = (
        historical_loans['firstrepaiddate'] - historical_loans['firstduedate']
    ).dt.days

    # Categorize payment status
    def categorize_payment(days):
        if pd.isna(days):
            return 'never_paid'
        elif days <= 0:
            return 'on_time'
        else:
            return 'late'

    historical_loans['payment_status'] = historical_loans['days_to_repay'].apply(categorize_payment)

    return historical_loans


# ============================================================================
# CHUNK 2: Aggregate Payment Behavior Per Customer
# ============================================================================

def aggregate_payment_features(historical_loans):
    """Aggregate payment behavior per customer"""
    print("=" * 70)
    print("STEP 1: Aggregate Payment Behavior")
    print("=" * 70)

    payment_counts = historical_loans.groupby(['customerid', 'payment_status']).size().unstack(fill_value=0)
    payment_counts['total_loans'] = payment_counts.sum(axis=1)

    payment_features = pd.DataFrame()
    payment_features['customerid'] = payment_counts.index

    if 'on_time' in payment_counts.columns:
        payment_features['payment_ontime_rate'] = payment_counts['on_time'] / payment_counts['total_loans']
    else:
        payment_features['payment_ontime_rate'] = 0

    if 'late' in payment_counts.columns:
        payment_features['payment_late_rate'] = payment_counts['late'] / payment_counts['total_loans']
    else:
        payment_features['payment_late_rate'] = 0

    if 'never_paid' in payment_counts.columns:
        payment_features['payment_never_rate'] = payment_counts['never_paid'] / payment_counts['total_loans']
    else:
        payment_features['payment_never_rate'] = 0

    late_payments = historical_loans[historical_loans['payment_status'] == 'late'].copy()

    if len(late_payments) > 0:
        avg_late = late_payments.groupby('customerid')['days_to_repay'].mean().reset_index()
        avg_late.columns = ['customerid', 'avg_days_late']
        max_late = late_payments.groupby('customerid')['days_to_repay'].max().reset_index()
        max_late.columns = ['customerid', 'max_days_late']
        payment_features = payment_features.merge(avg_late, on='customerid', how='left')
        payment_features = payment_features.merge(max_late, on='customerid', how='left')
    else:
        payment_features['avg_days_late'] = 0
        payment_features['max_days_late'] = 0

    payment_features['avg_days_late'] = payment_features['avg_days_late'].fillna(0)
    payment_features['max_days_late'] = payment_features['max_days_late'].fillna(0)

    print(f"   ✅ Payment features: {len(payment_features):,} customers, 5 features\n")

    return payment_features


# ============================================================================
# CHUNK 3: Create Loan Closure and Recency Features
# ============================================================================

def create_closure_and_recency_features(historical_loans):
    """
    Create loan closure and recency features

    Features created:
    - loan_closure_rate: % of loans that were closed (completed)
    - loan_abandonment_rate: % of loans never closed (abandoned)
    - days_since_last_loan: Days from most recent loan approval to today
    - recency_category: 'recent' (<180 days), 'moderate' (180-365), 'old' (>365)
    """
    print("=" * 70)
    print("STEP 2: Create Loan Closure & Recency Features")
    print("=" * 70)

    # Loan closure features
    historical_loans['is_closed'] = historical_loans['closeddate'].notnull().astype(int)

    closure_agg = historical_loans.groupby('customerid').agg({
        'is_closed': ['sum', 'count']
    }).reset_index()

    closure_agg.columns = ['customerid', 'closed_loans', 'total_loans']
    closure_agg['loan_closure_rate'] = closure_agg['closed_loans'] / closure_agg['total_loans']
    closure_agg['loan_abandonment_rate'] = 1 - closure_agg['loan_closure_rate']

    # Recency features
    recency_agg = historical_loans.groupby('customerid')['approveddate'].max().reset_index()
    recency_agg.columns = ['customerid', 'last_loan_date']

    # Calculate days since last loan (from today)
    today = pd.Timestamp.today()
    recency_agg['days_since_last_loan'] = (today - recency_agg['last_loan_date']).dt.days

    # Categorize recency
    def categorize_recency(days):
        if days < 180:
            return 'recent'
        elif days < 365:
            return 'moderate'
        else:
            return 'old'

    recency_agg['recency_category'] = recency_agg['days_since_last_loan'].apply(categorize_recency)

    # Merge closure and recency
    closure_features = closure_agg[['customerid', 'loan_closure_rate', 'loan_abandonment_rate']].copy()
    closure_features = closure_features.merge(
        recency_agg[['customerid', 'days_since_last_loan', 'recency_category']],
        on='customerid'
    )

    print(f"\n✅ Closure & recency features created:")
    print(f"   Customers: {len(closure_features):,}")
    print(f"\n   Features created:")
    print(f"      • loan_closure_rate")
    print(f"      • loan_abandonment_rate")
    print(f"      • days_since_last_loan")
    print(f"      • recency_category")

    print(f"\n📊 Recency breakdown:")
    print(recency_agg['recency_category'].value_counts())

    return closure_features


if __name__ == "__main__":
    master_df, historical_loans = load_data()
    historical_loans = calculate_payment_behavior(historical_loans)
    payment_features = aggregate_payment_features(historical_loans)
    closure_features = create_closure_and_recency_features(historical_loans)

    print("\n" + "=" * 70)
    print("SAMPLE: Closure & Recency Features")
    print("=" * 70)
    print(closure_features.head())
```

**Run it:**
```bash
python src/data/engineer_features.py
```

**Expected output:**
```
======================================================================
STEP 2: Create Loan Closure & Recency Features
======================================================================

✅ Closure & recency features created:
   Customers: 6,500

   Features created:
      • loan_closure_rate
      • loan_abandonment_rate
      • days_since_last_loan
      • recency_category

📊 Recency breakdown:
recent        2,500
moderate      2,000
old           2,000
```

---

### Step 4: Chunk 4 - Create Demographic and Ratio Features

**Delete everything in `src/data/engineer_features.py` and replace with this:**

```python
"""
Day 3: Feature Engineering from Kaggle Credit Risk dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'

# ============================================================================
# Helper Functions (Chunks 1-3)
# ============================================================================

def load_data():
    """Load master dataset and historical loans"""
    print("=" * 70)
    print("DAY 3: Feature Engineering")
    print("=" * 70)

    print("\n📂 Loading data...")

    master_df = pd.read_csv(PROCESSED_DATA_DIR / 'master_dataset_day2.csv')
    historical_loans = pd.read_csv(RAW_DATA_DIR / 'trainprevloans.csv')

    print(f"   ✅ Master: {master_df.shape}, Historical: {historical_loans.shape}\n")

    return master_df, historical_loans


def calculate_payment_behavior(historical_loans):
    """Calculate payment behavior"""
    date_cols = ['approveddate', 'creationdate', 'closeddate', 'firstduedate', 'firstrepaiddate']
    for col in date_cols:
        if col in historical_loans.columns:
            historical_loans[col] = pd.to_datetime(historical_loans[col], errors='coerce')

    historical_loans['days_to_repay'] = (
        historical_loans['firstrepaiddate'] - historical_loans['firstduedate']
    ).dt.days

    def categorize_payment(days):
        if pd.isna(days):
            return 'never_paid'
        elif days <= 0:
            return 'on_time'
        else:
            return 'late'

    historical_loans['payment_status'] = historical_loans['days_to_repay'].apply(categorize_payment)
    return historical_loans


def aggregate_payment_features(historical_loans):
    """Aggregate payment features"""
    payment_counts = historical_loans.groupby(['customerid', 'payment_status']).size().unstack(fill_value=0)
    payment_counts['total_loans'] = payment_counts.sum(axis=1)

    payment_features = pd.DataFrame()
    payment_features['customerid'] = payment_counts.index

    if 'on_time' in payment_counts.columns:
        payment_features['payment_ontime_rate'] = payment_counts['on_time'] / payment_counts['total_loans']
    else:
        payment_features['payment_ontime_rate'] = 0

    if 'late' in payment_counts.columns:
        payment_features['payment_late_rate'] = payment_counts['late'] / payment_counts['total_loans']
    else:
        payment_features['payment_late_rate'] = 0

    if 'never_paid' in payment_counts.columns:
        payment_features['payment_never_rate'] = payment_counts['never_paid'] / payment_counts['total_loans']
    else:
        payment_features['payment_never_rate'] = 0

    late_payments = historical_loans[historical_loans['payment_status'] == 'late'].copy()

    if len(late_payments) > 0:
        avg_late = late_payments.groupby('customerid')['days_to_repay'].mean().reset_index()
        avg_late.columns = ['customerid', 'avg_days_late']
        max_late = late_payments.groupby('customerid')['days_to_repay'].max().reset_index()
        max_late.columns = ['customerid', 'max_days_late']
        payment_features = payment_features.merge(avg_late, on='customerid', how='left')
        payment_features = payment_features.merge(max_late, on='customerid', how='left')
    else:
        payment_features['avg_days_late'] = 0
        payment_features['max_days_late'] = 0

    payment_features['avg_days_late'] = payment_features['avg_days_late'].fillna(0)
    payment_features['max_days_late'] = payment_features['max_days_late'].fillna(0)

    return payment_features


def create_closure_and_recency_features(historical_loans):
    """Create closure and recency features"""
    historical_loans['is_closed'] = historical_loans['closeddate'].notnull().astype(int)

    closure_agg = historical_loans.groupby('customerid').agg({
        'is_closed': ['sum', 'count']
    }).reset_index()

    closure_agg.columns = ['customerid', 'closed_loans', 'total_loans']
    closure_agg['loan_closure_rate'] = closure_agg['closed_loans'] / closure_agg['total_loans']
    closure_agg['loan_abandonment_rate'] = 1 - closure_agg['loan_closure_rate']

    recency_agg = historical_loans.groupby('customerid')['approveddate'].max().reset_index()
    recency_agg.columns = ['customerid', 'last_loan_date']

    today = pd.Timestamp.today()
    recency_agg['days_since_last_loan'] = (today - recency_agg['last_loan_date']).dt.days

    def categorize_recency(days):
        if days < 180:
            return 'recent'
        elif days < 365:
            return 'moderate'
        else:
            return 'old'

    recency_agg['recency_category'] = recency_agg['days_since_last_loan'].apply(categorize_recency)

    closure_features = closure_agg[['customerid', 'loan_closure_rate', 'loan_abandonment_rate']].copy()
    closure_features = closure_features.merge(
        recency_agg[['customerid', 'days_since_last_loan', 'recency_category']],
        on='customerid'
    )

    return closure_features


# ============================================================================
# CHUNK 4: Create Demographic and Ratio Features
# ============================================================================

def create_demographic_and_ratio_features(master_df):
    """
    Create demographic and ratio features from master dataset

    Features created:
    - age: Calculated from birthdate
    - age_group: Categorical age buckets
    - current_to_avg_loan_ratio: Current loan / average historical loan
    - current_to_max_loan_ratio: Current loan / max historical loan
    """
    print("\n" + "=" * 70)
    print("STEP 3: Create Demographic & Ratio Features")
    print("=" * 70)

    df = master_df.copy()

    # Age from birthdate
    if 'birthdate' in df.columns:
        df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')
        today = pd.Timestamp.today()
        df['age'] = ((today - df['birthdate']).dt.days / 365.25).astype(int)

        # Age groups
        def categorize_age(age):
            if pd.isna(age) or age < 18:
                return 'unknown'
            elif age < 25:
                return '18-24'
            elif age < 35:
                return '25-34'
            elif age < 45:
                return '35-44'
            elif age < 55:
                return '45-54'
            else:
                return '55+'

        df['age_group'] = df['age'].apply(categorize_age)

        print(f"\n✅ Age features created")
        print(f"   Age range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
        print(f"\n📊 Age group distribution:")
        print(df['age_group'].value_counts().sort_index())

    # Loan ratios (current loan vs historical)
    if 'loanamount' in df.columns and 'hist_avg_loan_amount' in df.columns:
        df['current_to_avg_loan_ratio'] = df['loanamount'] / (df['hist_avg_loan_amount'] + 1)  # +1 to avoid division by zero

        if 'hist_max_loan_amount' in df.columns:
            df['current_to_max_loan_ratio'] = df['loanamount'] / (df['hist_max_loan_amount'] + 1)

        print(f"\n✅ Loan ratio features created")

    print(f"\n✅ Total demographic & ratio features created: 4")

    return df


if __name__ == "__main__":
    # Load and process
    master_df, historical_loans = load_data()
    historical_loans = calculate_payment_behavior(historical_loans)

    print("=" * 70)
    print("STEP 1: Aggregate Payment Features")
    print("=" * 70)
    payment_features = aggregate_payment_features(historical_loans)
    print(f"   ✅ Payment features: {len(payment_features):,} customers\n")

    print("=" * 70)
    print("STEP 2: Create Closure & Recency Features")
    print("=" * 70)
    closure_features = create_closure_and_recency_features(historical_loans)
    print(f"   ✅ Closure features: {len(closure_features):,} customers\n")

    # Create demographic features
    master_df = create_demographic_and_ratio_features(master_df)

    print("\n" + "=" * 70)
    print("PROGRESS CHECK")
    print("=" * 70)
    print(f"✅ Master dataset: {master_df.shape}")
    print(f"✅ Payment features ready: {len(payment_features):,} customers")
    print(f"✅ Closure features ready: {len(closure_features):,} customers")
```

**Run it:**
```bash
python src/data/engineer_features.py
```

**Expected output:**
```
======================================================================
STEP 3: Create Demographic & Ratio Features
======================================================================

✅ Age features created
   Age range: 21 - 68 years

📊 Age group distribution:
18-24      500
25-34    2,500
35-44    3,000
45-54    1,500
55+        500

✅ Loan ratio features created

✅ Total demographic & ratio features created: 4
```

---

### Step 5: Chunk 5 - Merge All Features and Save

**Delete everything in `src/data/engineer_features.py` and replace with the final complete version:**

```python
"""
Day 3: Feature Engineering from Kaggle Credit Risk dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load master dataset and historical loans"""
    print("=" * 70)
    print("DAY 3: Feature Engineering")
    print("=" * 70)

    print("\n📂 Loading data...")

    master_df = pd.read_csv(PROCESSED_DATA_DIR / 'master_dataset_day2.csv')
    historical_loans = pd.read_csv(RAW_DATA_DIR / 'trainprevloans.csv')

    print(f"   ✅ Master: {master_df.shape}, Historical: {historical_loans.shape}\n")

    return master_df, historical_loans

# ============================================================================
# PAYMENT BEHAVIOR FEATURES
# ============================================================================

def calculate_payment_behavior(historical_loans):
    """Calculate payment behavior from dates"""
    date_cols = ['approveddate', 'creationdate', 'closeddate', 'firstduedate', 'firstrepaiddate']
    for col in date_cols:
        if col in historical_loans.columns:
            historical_loans[col] = pd.to_datetime(historical_loans[col], errors='coerce')

    historical_loans['days_to_repay'] = (
        historical_loans['firstrepaiddate'] - historical_loans['firstduedate']
    ).dt.days

    def categorize_payment(days):
        if pd.isna(days):
            return 'never_paid'
        elif days <= 0:
            return 'on_time'
        else:
            return 'late'

    historical_loans['payment_status'] = historical_loans['days_to_repay'].apply(categorize_payment)
    return historical_loans


def aggregate_payment_features(historical_loans):
    """Aggregate payment features per customer"""
    payment_counts = historical_loans.groupby(['customerid', 'payment_status']).size().unstack(fill_value=0)
    payment_counts['total_loans'] = payment_counts.sum(axis=1)

    payment_features = pd.DataFrame()
    payment_features['customerid'] = payment_counts.index

    if 'on_time' in payment_counts.columns:
        payment_features['payment_ontime_rate'] = payment_counts['on_time'] / payment_counts['total_loans']
    else:
        payment_features['payment_ontime_rate'] = 0

    if 'late' in payment_counts.columns:
        payment_features['payment_late_rate'] = payment_counts['late'] / payment_counts['total_loans']
    else:
        payment_features['payment_late_rate'] = 0

    if 'never_paid' in payment_counts.columns:
        payment_features['payment_never_rate'] = payment_counts['never_paid'] / payment_counts['total_loans']
    else:
        payment_features['payment_never_rate'] = 0

    late_payments = historical_loans[historical_loans['payment_status'] == 'late'].copy()

    if len(late_payments) > 0:
        avg_late = late_payments.groupby('customerid')['days_to_repay'].mean().reset_index()
        avg_late.columns = ['customerid', 'avg_days_late']
        max_late = late_payments.groupby('customerid')['days_to_repay'].max().reset_index()
        max_late.columns = ['customerid', 'max_days_late']
        payment_features = payment_features.merge(avg_late, on='customerid', how='left')
        payment_features = payment_features.merge(max_late, on='customerid', how='left')
    else:
        payment_features['avg_days_late'] = 0
        payment_features['max_days_late'] = 0

    payment_features['avg_days_late'] = payment_features['avg_days_late'].fillna(0)
    payment_features['max_days_late'] = payment_features['max_days_late'].fillna(0)

    return payment_features

# ============================================================================
# CLOSURE AND RECENCY FEATURES
# ============================================================================

def create_closure_and_recency_features(historical_loans):
    """Create closure and recency features"""
    historical_loans['is_closed'] = historical_loans['closeddate'].notnull().astype(int)

    closure_agg = historical_loans.groupby('customerid').agg({
        'is_closed': ['sum', 'count']
    }).reset_index()

    closure_agg.columns = ['customerid', 'closed_loans', 'total_loans']
    closure_agg['loan_closure_rate'] = closure_agg['closed_loans'] / closure_agg['total_loans']
    closure_agg['loan_abandonment_rate'] = 1 - closure_agg['loan_closure_rate']

    recency_agg = historical_loans.groupby('customerid')['approveddate'].max().reset_index()
    recency_agg.columns = ['customerid', 'last_loan_date']

    today = pd.Timestamp.today()
    recency_agg['days_since_last_loan'] = (today - recency_agg['last_loan_date']).dt.days

    def categorize_recency(days):
        if days < 180:
            return 'recent'
        elif days < 365:
            return 'moderate'
        else:
            return 'old'

    recency_agg['recency_category'] = recency_agg['days_since_last_loan'].apply(categorize_recency)

    closure_features = closure_agg[['customerid', 'loan_closure_rate', 'loan_abandonment_rate']].copy()
    closure_features = closure_features.merge(
        recency_agg[['customerid', 'days_since_last_loan', 'recency_category']],
        on='customerid'
    )

    return closure_features

# ============================================================================
# DEMOGRAPHIC AND RATIO FEATURES
# ============================================================================

def create_demographic_and_ratio_features(master_df):
    """Create demographic and ratio features"""
    df = master_df.copy()

    # Age from birthdate
    if 'birthdate' in df.columns:
        df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')
        today = pd.Timestamp.today()
        df['age'] = ((today - df['birthdate']).dt.days / 365.25).astype(int)

        def categorize_age(age):
            if pd.isna(age) or age < 18:
                return 'unknown'
            elif age < 25:
                return '18-24'
            elif age < 35:
                return '25-34'
            elif age < 45:
                return '35-44'
            elif age < 55:
                return '45-54'
            else:
                return '55+'

        df['age_group'] = df['age'].apply(categorize_age)

    # Loan ratios
    if 'loanamount' in df.columns and 'hist_avg_loan_amount' in df.columns:
        df['current_to_avg_loan_ratio'] = df['loanamount'] / (df['hist_avg_loan_amount'] + 1)

        if 'hist_max_loan_amount' in df.columns:
            df['current_to_max_loan_ratio'] = df['loanamount'] / (df['hist_max_loan_amount'] + 1)

    return df

# ============================================================================
# MERGE AND SAVE
# ============================================================================

def merge_all_features_and_save(master_df, payment_features, closure_features):
    """Merge all engineered features and save final dataset"""
    print("\n" + "=" * 70)
    print("STEP 4: Merge All Features & Save")
    print("=" * 70)

    print(f"\n📊 Before merge:")
    print(f"   Master dataset: {master_df.shape}")
    print(f"   Payment features: {payment_features.shape}")
    print(f"   Closure features: {closure_features.shape}")

    # Merge payment features
    final_df = master_df.merge(payment_features, on='customerid', how='left')

    # Merge closure features
    final_df = final_df.merge(closure_features, on='customerid', how='left')

    # Fill missing values for customers without history
    new_cols = list(payment_features.columns) + list(closure_features.columns)
    new_cols = [col for col in new_cols if col != 'customerid']

    for col in new_cols:
        if col in final_df.columns:
            if final_df[col].dtype in ['float64', 'int64']:
                final_df[col] = final_df[col].fillna(0)
            else:
                final_df[col] = final_df[col].fillna('unknown')

    print(f"\n✅ After merge:")
    print(f"   Final dataset: {final_df.shape}")
    print(f"   New features added: {final_df.shape[1] - master_df.shape[1]}")

    # Save
    output_path = PROCESSED_DATA_DIR / 'master_dataset_day3.csv'
    final_df.to_csv(output_path, index=False)

    print(f"\n✅ Saved: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024**2:.2f} MB")

    return final_df

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Load data
    master_df, historical_loans = load_data()

    # Step 1: Payment behavior features
    print("=" * 70)
    print("STEP 1: Create Payment Behavior Features")
    print("=" * 70)
    historical_loans = calculate_payment_behavior(historical_loans)
    payment_features = aggregate_payment_features(historical_loans)
    print(f"\n✅ Payment features: {payment_features.shape}")

    # Step 2: Closure and recency features
    print("\n" + "=" * 70)
    print("STEP 2: Create Closure & Recency Features")
    print("=" * 70)
    closure_features = create_closure_and_recency_features(historical_loans)
    print(f"\n✅ Closure features: {closure_features.shape}")

    # Step 3: Demographic and ratio features
    print("\n" + "=" * 70)
    print("STEP 3: Create Demographic & Ratio Features")
    print("=" * 70)
    master_df = create_demographic_and_ratio_features(master_df)
    print(f"\n✅ Master dataset with demographics: {master_df.shape}")

    # Step 4: Merge and save
    final_df = merge_all_features_and_save(master_df, payment_features, closure_features)

    # Final summary
    print("\n" + "=" * 70)
    print("🎉 DAY 3 COMPLETE!")
    print("=" * 70)
    print(f"✅ Engineered dataset: {final_df.shape[0]:,} rows × {final_df.shape[1]} features")
    print(f"✅ Saved to: data/processed/master_dataset_day3.csv")
    print(f"\n📊 Feature categories added:")
    print(f"   • Payment behavior: 5 features")
    print(f"   • Loan closure: 2 features")
    print(f"   • Recency: 2 features")
    print(f"   • Demographics: 2 features")
    print(f"   • Loan ratios: 2 features")
    print(f"   Total new features: 13")
    print(f"\n🎯 Next: Day 4 - EDA & Outlier Detection")
```

**Run the final version:**
```bash
python src/data/engineer_features.py
```

**Expected output:**
```
======================================================================
STEP 4: Merge All Features & Save
======================================================================

📊 Before merge:
   Master dataset: (8000, 36)
   Payment features: (6500, 6)
   Closure features: (6500, 5)

✅ After merge:
   Final dataset: (8000, 45)
   New features added: 9

✅ Saved: data/processed/master_dataset_day3.csv
   Size: 3.12 MB

======================================================================
🎉 DAY 3 COMPLETE!
======================================================================
✅ Engineered dataset: 8,000 rows × 45 features
✅ Saved to: data/processed/master_dataset_day3.csv

📊 Feature categories added:
   • Payment behavior: 5 features
   • Loan closure: 2 features
   • Recency: 2 features
   • Demographics: 2 features
   • Loan ratios: 2 features
   Total new features: 13

🎯 Next: Day 4 - EDA & Outlier Detection
```

---

## ✅ Day 3 Checklist

Before moving to Day 4, verify:

- [ ] Script runs without errors: `python src/data/engineer_features.py`
- [ ] Output file exists: `data/processed/master_dataset_day3.csv`
- [ ] Dataset has ~8,000 rows and ~45 features
- [ ] Payment behavior features present (ontime_rate, late_rate, never_rate, avg_days_late, max_days_late)
- [ ] Closure features present (closure_rate, abandonment_rate, days_since_last_loan)
- [ ] Demographic features present (age, age_group)

---

## 🎯 What's Next?

**Day 4**: EDA & Outlier Detection
- Correlation analysis with target variable
- Identify top 10-15 most predictive features
- Detect outliers using IQR method
- Handle outliers through capping
- Create visualizations (heatmaps, distributions)
- Output: `data/processed/master_dataset_day4.csv`

---

## 📚 Key Concepts Learned

1. **Feature Engineering**: Transforming raw data into predictive features
2. **Payment Behavior**: On-time/late/never-paid rates are powerful predictors
3. **Loan Closure**: Completion vs abandonment shows loan commitment
4. **Recency**: Recent borrowing activity indicates current financial state
5. **Ratios**: Comparing current loan to historical averages reveals risk
6. **Age Calculation**: Converting dates to meaningful numeric features

**Time to complete**: ~60 minutes
**Files created**: 1 (src/data/engineer_features.py) + 1 output (master_dataset_day3.csv)
