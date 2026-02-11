# DAY 2: Data Merging - Kaggle Credit Risk Dataset

**Goal**: Merge demographics + current loans + historical loan behavior into one unified dataset

**Time**: 45 minutes

**Prerequisites**: Day 1 completed (3 CSV files in data/processed/)

---

## 📋 What You'll Build Today

By the end of Day 2, you'll have:
- ✅ Demographics + current loans merged (one row per loan with customer info)
- ✅ Historical loan behavior aggregated per customer (payment patterns, default history)
- ✅ Final master dataset with all 3 data sources merged
- ✅ Saved to `data/processed/master_dataset_day2.csv`

---

## Prerequisites Check

**Before starting Day 2, verify Day 1 outputs exist:**

```bash
cd credit-risk-api
ls -lh data/processed/demographics_day1.csv
ls -lh data/processed/current_loans_day1.csv
ls -lh data/processed/historical_loans_day1.csv
```

**If missing, complete Day 1 first:**
```bash
python src/data/load_kaggle_data.py
```

---

## PART 1: Understanding the Merge Strategy

**We have 3 datasets from Day 1:**

1. **demographics_day1.csv** (8,000 rows) - One row per customer
   - `customerid`, `birthdate`, `bank_account_type`, `education`, etc.

2. **current_loans_day1.csv** (8,000 rows) - One row per current loan
   - `customerid`, `loanamount`, `totaldue`, `approveddate`, etc.

3. **historical_loans_day1.csv** (~25,000 rows) - Multiple rows per customer
   - `customerid`, `loannumber`, `loanamount`, `closeddate`, `firstduedate`, `firstrepaiddate`, etc.

**Merge Strategy:**
```
Step 1: Demographics + Current Loans → 8,000 rows (one row per loan with customer info)
Step 2: Aggregate Historical Loans → 6,500 rows (one summary per customer with history)
Step 3: Merge Historical Summary → 8,000 rows (final master dataset)
```

**Key Insight**: Not all customers have historical data (~1,500 new customers). We'll use a **left join** to keep all current loans.

---

## PART 2: Build the Data Merging Script (4 Chunks)

We'll build `src/data/merge_kaggle_data.py` in 4 incremental steps.

### Step 1: Chunk 1 - Load Day 1 Outputs

Create a new file: `src/data/merge_kaggle_data.py`

Add this code:

```python
"""
Day 2: Merge Kaggle Credit Risk datasets
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'

# ============================================================================
# CHUNK 1: Load Day 1 Outputs
# ============================================================================

def load_day1_outputs():
    """Load the 3 datasets from Day 1"""
    print("=" * 70)
    print("DAY 2: Merge Kaggle Credit Risk Data")
    print("=" * 70)

    print("\n📂 Loading Day 1 outputs...")

    demographics = pd.read_csv(PROCESSED_DATA_DIR / 'demographics_day1.csv')
    print(f"   ✅ Demographics: {len(demographics):,} customers, {demographics.shape[1]} columns")

    current_loans = pd.read_csv(PROCESSED_DATA_DIR / 'current_loans_day1.csv')
    print(f"   ✅ Current loans: {len(current_loans):,} loans, {current_loans.shape[1]} columns")

    historical_loans = pd.read_csv(PROCESSED_DATA_DIR / 'historical_loans_day1.csv')
    print(f"   ✅ Historical loans: {len(historical_loans):,} records, {historical_loans.shape[1]} columns")
    print(f"   ✅ Unique customers with history: {historical_loans['customerid'].nunique():,}\n")

    return demographics, current_loans, historical_loans

if __name__ == "__main__":
    demographics, current_loans, historical_loans = load_day1_outputs()

    print("\n" + "=" * 70)
    print("PREVIEW: Demographics")
    print("=" * 70)
    print(demographics.head(3))

    print("\n" + "=" * 70)
    print("PREVIEW: Current Loans")
    print("=" * 70)
    print(current_loans.head(3))

    print("\n" + "=" * 70)
    print("PREVIEW: Historical Loans")
    print("=" * 70)
    print(historical_loans.head(3))
```

**Run it:**
```bash
python src/data/merge_kaggle_data.py
```

**Expected output:**
```
======================================================================
DAY 2: Merge Kaggle Credit Risk Data
======================================================================

📂 Loading Day 1 outputs...
   ✅ Demographics: 8,000 customers, 10 columns
   ✅ Current loans: 8,000 loans, 15 columns
   ✅ Historical loans: 25,000 records, 18 columns
   ✅ Unique customers with history: 6,500
```

---

### Step 2: Chunk 2 - Merge Demographics + Current Loans

**Delete everything in `src/data/merge_kaggle_data.py` and replace with this:**

```python
"""
Day 2: Merge Kaggle Credit Risk datasets
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'

# ============================================================================
# CHUNK 1: Load Day 1 Outputs
# ============================================================================

def load_day1_outputs():
    """Load the 3 datasets from Day 1"""
    print("=" * 70)
    print("DAY 2: Merge Kaggle Credit Risk Data")
    print("=" * 70)

    print("\n📂 Loading Day 1 outputs...")

    demographics = pd.read_csv(PROCESSED_DATA_DIR / 'demographics_day1.csv')
    print(f"   ✅ Demographics: {len(demographics):,} customers, {demographics.shape[1]} columns")

    current_loans = pd.read_csv(PROCESSED_DATA_DIR / 'current_loans_day1.csv')
    print(f"   ✅ Current loans: {len(current_loans):,} loans, {current_loans.shape[1]} columns")

    historical_loans = pd.read_csv(PROCESSED_DATA_DIR / 'historical_loans_day1.csv')
    print(f"   ✅ Historical loans: {len(historical_loans):,} records, {historical_loans.shape[1]} columns")
    print(f"   ✅ Unique customers with history: {historical_loans['customerid'].nunique():,}\n")

    return demographics, current_loans, historical_loans

# ============================================================================
# CHUNK 2: Merge Demographics + Current Loans
# ============================================================================

def merge_demographics_and_loans(demographics, current_loans):
    """
    Merge demographics and current loans on customerid
    Result: One row per loan with customer demographic information
    """
    print("=" * 70)
    print("STEP 1: Merge Demographics + Current Loans")
    print("=" * 70)

    print(f"\n📊 Before merge:")
    print(f"   Demographics: {len(demographics):,} rows")
    print(f"   Current loans: {len(current_loans):,} rows")

    # Left join: Keep all current loans, add customer info
    merged = current_loans.merge(
        demographics,
        on='customerid',
        how='left',
        validate='many_to_one'  # Many loans can belong to one customer
    )

    print(f"\n✅ After merge:")
    print(f"   Merged dataset: {len(merged):,} rows, {merged.shape[1]} columns")

    # Check for missing matches
    missing_customers = merged['customerid'].isnull().sum()
    if missing_customers > 0:
        print(f"   ⚠️  WARNING: {missing_customers:,} loans have no customer info")
    else:
        print(f"   ✅ All loans have customer info")

    return merged

if __name__ == "__main__":
    demographics, current_loans, historical_loans = load_day1_outputs()

    # Merge demographics + current loans
    merged = merge_demographics_and_loans(demographics, current_loans)

    print("\n" + "=" * 70)
    print("MERGED PREVIEW (Demographics + Current Loans)")
    print("=" * 70)
    print(merged.head(3))
    print("\nColumns:", merged.columns.tolist())
```

**Run it:**
```bash
python src/data/merge_kaggle_data.py
```

**Expected output:**
```
======================================================================
STEP 1: Merge Demographics + Current Loans
======================================================================

📊 Before merge:
   Demographics: 8,000 rows
   Current loans: 8,000 rows

✅ After merge:
   Merged dataset: 8,000 rows, 24 columns
   ✅ All loans have customer info
```

---

### Step 3: Chunk 3 - Aggregate Historical Loans Per Customer

**Delete everything in `src/data/merge_kaggle_data.py` and replace with this:**

```python
"""
Day 2: Merge Kaggle Credit Risk datasets
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'

# ============================================================================
# CHUNK 1: Load Day 1 Outputs
# ============================================================================

def load_day1_outputs():
    """Load the 3 datasets from Day 1"""
    print("=" * 70)
    print("DAY 2: Merge Kaggle Credit Risk Data")
    print("=" * 70)

    print("\n📂 Loading Day 1 outputs...")

    demographics = pd.read_csv(PROCESSED_DATA_DIR / 'demographics_day1.csv')
    print(f"   ✅ Demographics: {len(demographics):,} customers, {demographics.shape[1]} columns")

    current_loans = pd.read_csv(PROCESSED_DATA_DIR / 'current_loans_day1.csv')
    print(f"   ✅ Current loans: {len(current_loans):,} loans, {current_loans.shape[1]} columns")

    historical_loans = pd.read_csv(PROCESSED_DATA_DIR / 'historical_loans_day1.csv')
    print(f"   ✅ Historical loans: {len(historical_loans):,} records, {historical_loans.shape[1]} columns")
    print(f"   ✅ Unique customers with history: {historical_loans['customerid'].nunique():,}\n")

    return demographics, current_loans, historical_loans

# ============================================================================
# CHUNK 2: Merge Demographics + Current Loans
# ============================================================================

def merge_demographics_and_loans(demographics, current_loans):
    """
    Merge demographics and current loans on customerid
    Result: One row per loan with customer demographic information
    """
    print("=" * 70)
    print("STEP 1: Merge Demographics + Current Loans")
    print("=" * 70)

    print(f"\n📊 Before merge:")
    print(f"   Demographics: {len(demographics):,} rows")
    print(f"   Current loans: {len(current_loans):,} rows")

    # Left join: Keep all current loans, add customer info
    merged = current_loans.merge(
        demographics,
        on='customerid',
        how='left',
        validate='many_to_one'
    )

    print(f"\n✅ After merge:")
    print(f"   Merged dataset: {len(merged):,} rows, {merged.shape[1]} columns")

    # Check for missing matches
    missing_customers = merged['customerid'].isnull().sum()
    if missing_customers > 0:
        print(f"   ⚠️  WARNING: {missing_customers:,} loans have no customer info")
    else:
        print(f"   ✅ All loans have customer info")

    return merged

# ============================================================================
# CHUNK 3: Aggregate Historical Loans Per Customer
# ============================================================================

def aggregate_historical_loans(historical_loans):
    """
    Aggregate historical loan behavior per customer
    From 25,000 rows → 6,500 rows (one row per customer with history)

    Features created:
    - hist_total_loans: Total number of historical loans
    - hist_avg_loan_amount: Average historical loan amount
    - hist_total_loan_amount: Total amount borrowed historically
    - hist_closed_loans: Number of closed loans
    - hist_open_loans: Number of still-open loans
    """
    print("\n" + "=" * 70)
    print("STEP 2: Aggregate Historical Loans Per Customer")
    print("=" * 70)

    print(f"\n📊 Before aggregation:")
    print(f"   Historical loans: {len(historical_loans):,} rows")
    print(f"   Unique customers: {historical_loans['customerid'].nunique():,}")

    # Convert dates if needed
    date_cols = ['approveddate', 'creationdate', 'closeddate', 'firstduedate', 'firstrepaiddate']
    for col in date_cols:
        if col in historical_loans.columns:
            historical_loans[col] = pd.to_datetime(historical_loans[col], errors='coerce')

    # Aggregate per customer
    agg_dict = {
        'loannumber': 'count',  # Total loans
        'loanamount': ['mean', 'sum', 'max', 'min'],  # Loan amount stats
    }

    # Check if closeddate exists (indicates closed loans)
    if 'closeddate' in historical_loans.columns:
        historical_loans['is_closed'] = historical_loans['closeddate'].notnull().astype(int)
        agg_dict['is_closed'] = 'sum'  # Count closed loans

    hist_agg = historical_loans.groupby('customerid').agg(agg_dict).reset_index()

    # Flatten column names
    hist_agg.columns = ['customerid', 'hist_total_loans', 'hist_avg_loan_amount',
                        'hist_total_loan_amount', 'hist_max_loan_amount', 'hist_min_loan_amount',
                        'hist_closed_loans']

    # Calculate open loans
    hist_agg['hist_open_loans'] = hist_agg['hist_total_loans'] - hist_agg['hist_closed_loans']

    # Add flag for having history
    hist_agg['has_history'] = 1

    print(f"\n✅ After aggregation:")
    print(f"   Aggregated dataset: {len(hist_agg):,} rows (one per customer), {hist_agg.shape[1]} features")
    print(f"\n   Features created:")
    for col in hist_agg.columns:
        if col != 'customerid':
            print(f"      • {col}")

    return hist_agg

if __name__ == "__main__":
    demographics, current_loans, historical_loans = load_day1_outputs()

    # Merge demographics + current loans
    merged = merge_demographics_and_loans(demographics, current_loans)

    # Aggregate historical loans
    hist_agg = aggregate_historical_loans(historical_loans)

    print("\n" + "=" * 70)
    print("AGGREGATED HISTORICAL LOANS PREVIEW")
    print("=" * 70)
    print(hist_agg.head(3))
```

**Run it:**
```bash
python src/data/merge_kaggle_data.py
```

**Expected output:**
```
======================================================================
STEP 2: Aggregate Historical Loans Per Customer
======================================================================

📊 Before aggregation:
   Historical loans: 25,000 rows
   Unique customers: 6,500

✅ After aggregation:
   Aggregated dataset: 6,500 rows (one per customer), 9 features

   Features created:
      • hist_total_loans
      • hist_avg_loan_amount
      • hist_total_loan_amount
      • hist_max_loan_amount
      • hist_min_loan_amount
      • hist_closed_loans
      • hist_open_loans
      • has_history
```

---

### Step 4: Chunk 4 - Final Merge + Save

**Delete everything in `src/data/merge_kaggle_data.py` and replace with this:**

```python
"""
Day 2: Merge Kaggle Credit Risk datasets
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'

# ============================================================================
# CHUNK 1: Load Day 1 Outputs
# ============================================================================

def load_day1_outputs():
    """Load the 3 datasets from Day 1"""
    print("=" * 70)
    print("DAY 2: Merge Kaggle Credit Risk Data")
    print("=" * 70)

    print("\n📂 Loading Day 1 outputs...")

    demographics = pd.read_csv(PROCESSED_DATA_DIR / 'demographics_day1.csv')
    print(f"   ✅ Demographics: {len(demographics):,} customers, {demographics.shape[1]} columns")

    current_loans = pd.read_csv(PROCESSED_DATA_DIR / 'current_loans_day1.csv')
    print(f"   ✅ Current loans: {len(current_loans):,} loans, {current_loans.shape[1]} columns")

    historical_loans = pd.read_csv(PROCESSED_DATA_DIR / 'historical_loans_day1.csv')
    print(f"   ✅ Historical loans: {len(historical_loans):,} records, {historical_loans.shape[1]} columns")
    print(f"   ✅ Unique customers with history: {historical_loans['customerid'].nunique():,}\n")

    return demographics, current_loans, historical_loans

# ============================================================================
# CHUNK 2: Merge Demographics + Current Loans
# ============================================================================

def merge_demographics_and_loans(demographics, current_loans):
    """
    Merge demographics and current loans on customerid
    Result: One row per loan with customer demographic information
    """
    print("=" * 70)
    print("STEP 1: Merge Demographics + Current Loans")
    print("=" * 70)

    print(f"\n📊 Before merge:")
    print(f"   Demographics: {len(demographics):,} rows")
    print(f"   Current loans: {len(current_loans):,} rows")

    # Left join: Keep all current loans, add customer info
    merged = current_loans.merge(
        demographics,
        on='customerid',
        how='left',
        validate='many_to_one'
    )

    print(f"\n✅ After merge:")
    print(f"   Merged dataset: {len(merged):,} rows, {merged.shape[1]} columns")

    # Check for missing matches
    missing_customers = merged['customerid'].isnull().sum()
    if missing_customers > 0:
        print(f"   ⚠️  WARNING: {missing_customers:,} loans have no customer info")
    else:
        print(f"   ✅ All loans have customer info")

    return merged

# ============================================================================
# CHUNK 3: Aggregate Historical Loans Per Customer
# ============================================================================

def aggregate_historical_loans(historical_loans):
    """
    Aggregate historical loan behavior per customer
    From 25,000 rows → 6,500 rows (one row per customer with history)
    """
    print("\n" + "=" * 70)
    print("STEP 2: Aggregate Historical Loans Per Customer")
    print("=" * 70)

    print(f"\n📊 Before aggregation:")
    print(f"   Historical loans: {len(historical_loans):,} rows")
    print(f"   Unique customers: {historical_loans['customerid'].nunique():,}")

    # Convert dates if needed
    date_cols = ['approveddate', 'creationdate', 'closeddate', 'firstduedate', 'firstrepaiddate']
    for col in date_cols:
        if col in historical_loans.columns:
            historical_loans[col] = pd.to_datetime(historical_loans[col], errors='coerce')

    # Aggregate per customer
    agg_dict = {
        'loannumber': 'count',
        'loanamount': ['mean', 'sum', 'max', 'min'],
    }

    if 'closeddate' in historical_loans.columns:
        historical_loans['is_closed'] = historical_loans['closeddate'].notnull().astype(int)
        agg_dict['is_closed'] = 'sum'

    hist_agg = historical_loans.groupby('customerid').agg(agg_dict).reset_index()

    # Flatten column names
    hist_agg.columns = ['customerid', 'hist_total_loans', 'hist_avg_loan_amount',
                        'hist_total_loan_amount', 'hist_max_loan_amount', 'hist_min_loan_amount',
                        'hist_closed_loans']

    hist_agg['hist_open_loans'] = hist_agg['hist_total_loans'] - hist_agg['hist_closed_loans']
    hist_agg['has_history'] = 1

    print(f"\n✅ After aggregation:")
    print(f"   Aggregated dataset: {len(hist_agg):,} rows, {hist_agg.shape[1]} features")

    return hist_agg

# ============================================================================
# CHUNK 4: Final Merge + Save
# ============================================================================

def merge_historical_and_save(merged, hist_agg):
    """
    Merge historical loan aggregates into main dataset
    Left join: Keep all current loans, add history where available
    """
    print("\n" + "=" * 70)
    print("STEP 3: Merge Historical Aggregates + Save")
    print("=" * 70)

    print(f"\n📊 Before final merge:")
    print(f"   Main dataset: {len(merged):,} rows")
    print(f"   Historical aggregates: {len(hist_agg):,} rows")

    # Left join: Keep all loans, add history if available
    final_merged = merged.merge(
        hist_agg,
        on='customerid',
        how='left'
    )

    # Fill missing historical values with 0 (customers with no history)
    hist_cols = [col for col in final_merged.columns if col.startswith('hist_')]
    final_merged[hist_cols] = final_merged[hist_cols].fillna(0)

    # Fill has_history flag
    final_merged['has_history'] = final_merged['has_history'].fillna(0).astype(int)

    print(f"\n✅ After final merge:")
    print(f"   Final dataset: {len(final_merged):,} rows, {final_merged.shape[1]} columns")

    # Summary stats
    customers_with_history = final_merged['has_history'].sum()
    customers_without_history = len(final_merged) - customers_with_history

    print(f"\n📊 Customer history breakdown:")
    print(f"   • Customers WITH history: {customers_with_history:,} ({customers_with_history/len(final_merged)*100:.1f}%)")
    print(f"   • Customers WITHOUT history: {customers_without_history:,} ({customers_without_history/len(final_merged)*100:.1f}%)")

    # Save
    output_path = PROCESSED_DATA_DIR / 'master_dataset_day2.csv'
    final_merged.to_csv(output_path, index=False)
    print(f"\n✅ Saved: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024**2:.2f} MB")

    return final_merged

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Load Day 1 outputs
    demographics, current_loans, historical_loans = load_day1_outputs()

    # Step 1: Merge demographics + current loans
    merged = merge_demographics_and_loans(demographics, current_loans)

    # Step 2: Aggregate historical loans
    hist_agg = aggregate_historical_loans(historical_loans)

    # Step 3: Final merge + save
    final_merged = merge_historical_and_save(merged, hist_agg)

    print("\n" + "=" * 70)
    print("FINAL MASTER DATASET PREVIEW")
    print("=" * 70)
    print(final_merged.head(3))
    print("\n" + "=" * 70)
    print("COLUMN SUMMARY")
    print("=" * 70)
    print(f"Total columns: {final_merged.shape[1]}")
    print("\nColumn names:")
    for i, col in enumerate(final_merged.columns, 1):
        print(f"   {i:2d}. {col}")

    print("\n" + "=" * 70)
    print("🎉 DAY 2 COMPLETE!")
    print("=" * 70)
    print(f"✅ Master dataset created: {len(final_merged):,} rows × {final_merged.shape[1]} columns")
    print(f"✅ Saved to: data/processed/master_dataset_day2.csv")
    print(f"\n🎯 Next: Day 3 - Feature Engineering (create 20+ ML-ready features)")
```

**Run the final version:**
```bash
python src/data/merge_kaggle_data.py
```

**Expected output:**
```
======================================================================
STEP 3: Merge Historical Aggregates + Save
======================================================================

📊 Before final merge:
   Main dataset: 8,000 rows
   Historical aggregates: 6,500 rows

✅ After final merge:
   Final dataset: 8,000 rows, 32 columns

📊 Customer history breakdown:
   • Customers WITH history: 6,500 (81.2%)
   • Customers WITHOUT history: 1,500 (18.8%)

✅ Saved: data/processed/master_dataset_day2.csv
   Size: 2.45 MB

======================================================================
🎉 DAY 2 COMPLETE!
======================================================================
✅ Master dataset created: 8,000 rows × 32 columns
✅ Saved to: data/processed/master_dataset_day2.csv

🎯 Next: Day 3 - Feature Engineering (create 20+ ML-ready features)
```

---

## ✅ Day 2 Checklist

Before moving to Day 3, verify:

- [ ] Script runs without errors: `python src/data/merge_kaggle_data.py`
- [ ] Output file exists: `data/processed/master_dataset_day2.csv`
- [ ] Dataset has ~8,000 rows and ~32 columns
- [ ] No missing customerid values
- [ ] Historical features filled with 0 for customers without history

---

## 🎯 What's Next?

**Day 3**: Feature Engineering
- Create age from birthdate
- Calculate loan-to-income ratio
- Encode categorical variables (education, bank type)
- Create payment delay features from historical data
- Create 20+ ML-ready features
- Output: `data/processed/engineered_day3.csv`

---

## 📚 Key Concepts Learned

1. **Data Merging**: Left joins to preserve all current loans
2. **Aggregation**: Converting multiple rows per customer to one summary row
3. **Missing Data Handling**: Filling 0 for customers without historical data
4. **Feature Creation**: Aggregating historical loan behavior into features
5. **Data Validation**: Checking merge results and customer coverage

**Time to complete**: ~45 minutes
**Files created**: 2 (1 script + 1 master dataset CSV)
