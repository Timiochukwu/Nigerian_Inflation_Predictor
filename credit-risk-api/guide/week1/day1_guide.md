# DAY 1: Project Setup & Data Loading - Kaggle Credit Risk Dataset

**Goal**: Set up project structure, install dependencies, load the 3 Kaggle CSV files, and understand each dataset

**Time**: 45 minutes

**Dataset**: [Kaggle Credit Risk Dataset](https://www.kaggle.com/)
- `traindemographics.csv` - Customer demographics (age, education, location, bank details)
- `trainperf.csv` - Current loan performance (loan amount, due date, approval status)
- `trainprevloans.csv` - Historical loan payment behavior (payment dates, closure status)

---

## 📋 What You'll Build Today

By the end of Day 1, you'll have:
- ✅ Complete project folder structure (src/, data/, models/, results/)
- ✅ All dependencies installed (pandas, numpy, matplotlib, seaborn)
- ✅ The 3 Kaggle CSV files loaded into data/raw/
- ✅ Data exploration script that shows statistics for all 3 datasets
- ✅ Understanding of the schema and relationships between the tables

---

## PART 1: Create Project Structure

### Step 1: Create All Required Folders

Run these commands in your terminal from the `credit-risk-api/` directory:

```bash
cd credit-risk-api
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/final
mkdir -p models
mkdir -p src/data
mkdir -p results
```

Verify your structure:
```bash
tree -L 2
# Or if tree is not installed:
ls -R
```

You should see:
```
credit-risk-api/
├── data/
│   ├── raw/           ← Kaggle CSV files go here
│   ├── processed/     ← Intermediate steps (merged, engineered, cleaned)
│   └── final/         ← ML-ready files (X_train, X_test, y_train, y_test)
├── models/            ← Saved trained models (.pkl files)
├── src/
│   └── data/          ← Data loading and processing scripts
└── results/           ← Visualizations and reports
```

---

## PART 2: Install Dependencies

### Step 2: Create requirements.txt

Create a new file: `credit-risk-api/requirements.txt`

Add these dependencies:

```txt
pandas==2.1.4
numpy==1.26.3
matplotlib==3.8.2
seaborn==0.13.1
scikit-learn==1.4.0
xgboost==2.0.3
```

### Step 3: Install All Dependencies

Run this command:

```bash
pip install -r requirements.txt
```

Wait for installation to complete (~2-3 minutes).

Verify installation:
```bash
python -c "import pandas, numpy, matplotlib, seaborn; print('✅ All libraries installed!')"
```

---

## PART 3: Prepare Your Kaggle Dataset

### Step 4: Download and Place Your Kaggle Data

**You should have downloaded 3 CSV files from Kaggle:**

1. `traindemographics.csv` - Customer demographics
2. `trainperf.csv` - Current loan performance
3. `trainprevloans.csv` - Historical loan payment behavior

**Copy your 3 CSV files into the `data/raw/` folder:**

```bash
# Example: Copy from your Downloads folder
cp ~/Downloads/traindemographics.csv credit-risk-api/data/raw/
cp ~/Downloads/trainperf.csv credit-risk-api/data/raw/
cp ~/Downloads/trainprevloans.csv credit-risk-api/data/raw/
```

Verify the files are there:
```bash
ls -lh credit-risk-api/data/raw/
```

You should see:
```
traindemographics.csv  (~200 KB)
trainperf.csv          (~400 KB)
trainprevloans.csv     (~800 KB)
```

**If you don't have the Kaggle files yet:**
- Go to [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets)
- Download `traindemographics.csv`, `trainperf.csv`, `trainprevloans.csv`
- Place them in `data/raw/`

---

## PART 4: Build the Data Loading Script (5 Chunks)

We'll build `src/data/load_kaggle_data.py` in 5 incremental steps.

### Step 5: Chunk 1 - Basic Setup and Load Demographics

Create a new file: `src/data/load_kaggle_data.py`

Add this code:

```python
"""
Day 1: Load and explore Kaggle Credit Risk dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CHUNK 1: Load traindemographics.csv
# ============================================================================

def load_demographics():
    """Load customer demographics data"""
    print("📊 Loading traindemographics.csv...")

    df = pd.read_csv(RAW_DATA_DIR / 'traindemographics.csv')

    print(f"   ✅ Loaded {len(df):,} customers")
    print(f"   ✅ Columns: {df.shape[1]}")
    print(f"   ✅ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    return df

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 1: Load Kaggle Credit Risk Data")
    print("=" * 70)

    demographics = load_demographics()
    print(demographics.head())
    print("\nColumns:")
    print(demographics.columns.tolist())
```

**Run it:**
```bash
python src/data/load_kaggle_data.py
```

**Expected output:**
```
======================================================================
DAY 1: Load Kaggle Credit Risk Data
======================================================================
📊 Loading traindemographics.csv...
   ✅ Loaded 8,000 customers
   ✅ Columns: 10
   ✅ Memory usage: 0.62 MB

   customerid  birthdate  bank_account_type  ...
0  C0001      1985-03-15  Savings           ...
```

---

### Step 6: Chunk 2 - Add Load Current Loans (trainperf.csv)

**Delete everything in `src/data/load_kaggle_data.py` and replace with this:**

```python
"""
Day 1: Load and explore Kaggle Credit Risk dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CHUNK 1: Load traindemographics.csv
# ============================================================================

def load_demographics():
    """Load customer demographics data"""
    print("📊 Loading traindemographics.csv...")

    df = pd.read_csv(RAW_DATA_DIR / 'traindemographics.csv')

    print(f"   ✅ Loaded {len(df):,} customers")
    print(f"   ✅ Columns: {df.shape[1]}")
    print(f"   ✅ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    return df

# ============================================================================
# CHUNK 2: Load trainperf.csv (Current Loan Performance)
# ============================================================================

def load_current_performance():
    """Load current loan performance data"""
    print("📊 Loading trainperf.csv...")

    df = pd.read_csv(RAW_DATA_DIR / 'trainperf.csv')

    print(f"   ✅ Loaded {len(df):,} current loans")
    print(f"   ✅ Columns: {df.shape[1]}")
    print(f"   ✅ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    return df

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 1: Load Kaggle Credit Risk Data")
    print("=" * 70)

    demographics = load_demographics()
    current_loans = load_current_performance()

    print("\n" + "=" * 70)
    print("CURRENT LOANS PREVIEW")
    print("=" * 70)
    print(current_loans.head())
    print("\nColumns:")
    print(current_loans.columns.tolist())
```

**Run it:**
```bash
python src/data/load_kaggle_data.py
```

**Expected output:**
```
📊 Loading traindemographics.csv...
   ✅ Loaded 8,000 customers

📊 Loading trainperf.csv...
   ✅ Loaded 8,000 current loans

======================================================================
CURRENT LOANS PREVIEW
======================================================================
   customerid  systemloanid  loannumber  loanamount  totaldue  ...
0  C0001       SL00001      1           50000       55000     ...
```

---

### Step 7: Chunk 3 - Add Load Historical Loans (trainprevloans.csv)

**Delete everything in `src/data/load_kaggle_data.py` and replace with this:**

```python
"""
Day 1: Load and explore Kaggle Credit Risk dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CHUNK 1: Load traindemographics.csv
# ============================================================================

def load_demographics():
    """Load customer demographics data"""
    print("📊 Loading traindemographics.csv...")

    df = pd.read_csv(RAW_DATA_DIR / 'traindemographics.csv')

    print(f"   ✅ Loaded {len(df):,} customers")
    print(f"   ✅ Columns: {df.shape[1]}")
    print(f"   ✅ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    return df

# ============================================================================
# CHUNK 2: Load trainperf.csv (Current Loan Performance)
# ============================================================================

def load_current_performance():
    """Load current loan performance data"""
    print("📊 Loading trainperf.csv...")

    df = pd.read_csv(RAW_DATA_DIR / 'trainperf.csv')

    print(f"   ✅ Loaded {len(df):,} current loans")
    print(f"   ✅ Columns: {df.shape[1]}")
    print(f"   ✅ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    return df

# ============================================================================
# CHUNK 3: Load trainprevloans.csv (Historical Loan Behavior)
# ============================================================================

def load_historical_loans():
    """Load historical loan payment behavior"""
    print("📊 Loading trainprevloans.csv...")

    df = pd.read_csv(RAW_DATA_DIR / 'trainprevloans.csv')

    print(f"   ✅ Loaded {len(df):,} historical loan records")
    print(f"   ✅ Columns: {df.shape[1]}")
    print(f"   ✅ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"   ✅ Unique customers with history: {df['customerid'].nunique():,}\n")

    return df

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 1: Load Kaggle Credit Risk Data")
    print("=" * 70)

    demographics = load_demographics()
    current_loans = load_current_performance()
    historical_loans = load_historical_loans()

    print("\n" + "=" * 70)
    print("HISTORICAL LOANS PREVIEW")
    print("=" * 70)
    print(historical_loans.head())
    print("\nColumns:")
    print(historical_loans.columns.tolist())
```

**Run it:**
```bash
python src/data/load_kaggle_data.py
```

**Expected output:**
```
📊 Loading trainprevloans.csv...
   ✅ Loaded 25,000 historical loan records
   ✅ Unique customers with history: 6,500

   customerid  systemloanid  loannumber  approveddate  closeddate  ...
0  C0001       SL00001      1           2020-01-15    2020-06-15  ...
```

---

### Step 8: Chunk 4 - Add Data Exploration Statistics

**Delete everything in `src/data/load_kaggle_data.py` and replace with this:**

```python
"""
Day 1: Load and explore Kaggle Credit Risk dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CHUNK 1: Load traindemographics.csv
# ============================================================================

def load_demographics():
    """Load customer demographics data"""
    print("📊 Loading traindemographics.csv...")

    df = pd.read_csv(RAW_DATA_DIR / 'traindemographics.csv')

    print(f"   ✅ Loaded {len(df):,} customers")
    print(f"   ✅ Columns: {df.shape[1]}")
    print(f"   ✅ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    return df

# ============================================================================
# CHUNK 2: Load trainperf.csv (Current Loan Performance)
# ============================================================================

def load_current_performance():
    """Load current loan performance data"""
    print("📊 Loading trainperf.csv...")

    df = pd.read_csv(RAW_DATA_DIR / 'trainperf.csv')

    print(f"   ✅ Loaded {len(df):,} current loans")
    print(f"   ✅ Columns: {df.shape[1]}")
    print(f"   ✅ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    return df

# ============================================================================
# CHUNK 3: Load trainprevloans.csv (Historical Loan Behavior)
# ============================================================================

def load_historical_loans():
    """Load historical loan payment behavior"""
    print("📊 Loading trainprevloans.csv...")

    df = pd.read_csv(RAW_DATA_DIR / 'trainprevloans.csv')

    print(f"   ✅ Loaded {len(df):,} historical loan records")
    print(f"   ✅ Columns: {df.shape[1]}")
    print(f"   ✅ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"   ✅ Unique customers with history: {df['customerid'].nunique():,}\n")

    return df

# ============================================================================
# CHUNK 4: Data Exploration Statistics
# ============================================================================

def explore_datasets(demographics, current_loans, historical_loans):
    """Show key statistics about each dataset"""
    print("\n" + "=" * 70)
    print("DATA EXPLORATION SUMMARY")
    print("=" * 70)

    # Demographics stats
    print("\n1️⃣  DEMOGRAPHICS (traindemographics.csv)")
    print(f"   Total customers: {len(demographics):,}")
    print(f"   Missing values per column:")
    for col in demographics.columns:
        missing = demographics[col].isnull().sum()
        if missing > 0:
            print(f"      • {col}: {missing:,} ({missing/len(demographics)*100:.1f}%)")

    # Current loans stats
    print("\n2️⃣  CURRENT LOANS (trainperf.csv)")
    print(f"   Total current loans: {len(current_loans):,}")
    if 'loanamount' in current_loans.columns:
        print(f"   Loan amount range: ${current_loans['loanamount'].min():,} - ${current_loans['loanamount'].max():,}")
        print(f"   Average loan amount: ${current_loans['loanamount'].mean():,.2f}")

    # Historical loans stats
    print("\n3️⃣  HISTORICAL LOANS (trainprevloans.csv)")
    print(f"   Total historical records: {len(historical_loans):,}")
    print(f"   Unique customers: {historical_loans['customerid'].nunique():,}")
    print(f"   Avg loans per customer: {len(historical_loans) / historical_loans['customerid'].nunique():.1f}")

    # Check how many customers have historical data
    customers_with_history = set(historical_loans['customerid'].unique())
    customers_without_history = set(demographics['customerid'].unique()) - customers_with_history
    print(f"   Customers WITH history: {len(customers_with_history):,}")
    print(f"   Customers WITHOUT history: {len(customers_without_history):,}")

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 1: Load Kaggle Credit Risk Data")
    print("=" * 70)

    demographics = load_demographics()
    current_loans = load_current_performance()
    historical_loans = load_historical_loans()

    explore_datasets(demographics, current_loans, historical_loans)
```

**Run it:**
```bash
python src/data/load_kaggle_data.py
```

**Expected output:**
```
======================================================================
DATA EXPLORATION SUMMARY
======================================================================

1️⃣  DEMOGRAPHICS (traindemographics.csv)
   Total customers: 8,000
   Missing values per column:
      • birthdate: 120 (1.5%)

2️⃣  CURRENT LOANS (trainperf.csv)
   Total current loans: 8,000
   Loan amount range: $5,000 - $500,000
   Average loan amount: $75,342.56

3️⃣  HISTORICAL LOANS (trainprevloans.csv)
   Total historical records: 25,000
   Unique customers: 6,500
   Avg loans per customer: 3.8
   Customers WITH history: 6,500
   Customers WITHOUT history: 1,500
```

---

### Step 9: Chunk 5 - Save Processed Data

**Delete everything in `src/data/load_kaggle_data.py` and replace with this:**

```python
"""
Day 1: Load and explore Kaggle Credit Risk dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CHUNK 1: Load traindemographics.csv
# ============================================================================

def load_demographics():
    """Load customer demographics data"""
    print("📊 Loading traindemographics.csv...")

    df = pd.read_csv(RAW_DATA_DIR / 'traindemographics.csv')

    print(f"   ✅ Loaded {len(df):,} customers")
    print(f"   ✅ Columns: {df.shape[1]}")
    print(f"   ✅ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    return df

# ============================================================================
# CHUNK 2: Load trainperf.csv (Current Loan Performance)
# ============================================================================

def load_current_performance():
    """Load current loan performance data"""
    print("📊 Loading trainperf.csv...")

    df = pd.read_csv(RAW_DATA_DIR / 'trainperf.csv')

    print(f"   ✅ Loaded {len(df):,} current loans")
    print(f"   ✅ Columns: {df.shape[1]}")
    print(f"   ✅ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    return df

# ============================================================================
# CHUNK 3: Load trainprevloans.csv (Historical Loan Behavior)
# ============================================================================

def load_historical_loans():
    """Load historical loan payment behavior"""
    print("📊 Loading trainprevloans.csv...")

    df = pd.read_csv(RAW_DATA_DIR / 'trainprevloans.csv')

    print(f"   ✅ Loaded {len(df):,} historical loan records")
    print(f"   ✅ Columns: {df.shape[1]}")
    print(f"   ✅ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"   ✅ Unique customers with history: {df['customerid'].nunique():,}\n")

    return df

# ============================================================================
# CHUNK 4: Data Exploration Statistics
# ============================================================================

def explore_datasets(demographics, current_loans, historical_loans):
    """Show key statistics about each dataset"""
    print("\n" + "=" * 70)
    print("DATA EXPLORATION SUMMARY")
    print("=" * 70)

    # Demographics stats
    print("\n1️⃣  DEMOGRAPHICS (traindemographics.csv)")
    print(f"   Total customers: {len(demographics):,}")
    print(f"   Missing values per column:")
    for col in demographics.columns:
        missing = demographics[col].isnull().sum()
        if missing > 0:
            print(f"      • {col}: {missing:,} ({missing/len(demographics)*100:.1f}%)")

    # Current loans stats
    print("\n2️⃣  CURRENT LOANS (trainperf.csv)")
    print(f"   Total current loans: {len(current_loans):,}")
    if 'loanamount' in current_loans.columns:
        print(f"   Loan amount range: ${current_loans['loanamount'].min():,} - ${current_loans['loanamount'].max():,}")
        print(f"   Average loan amount: ${current_loans['loanamount'].mean():,.2f}")

    # Historical loans stats
    print("\n3️⃣  HISTORICAL LOANS (trainprevloans.csv)")
    print(f"   Total historical records: {len(historical_loans):,}")
    print(f"   Unique customers: {historical_loans['customerid'].nunique():,}")
    print(f"   Avg loans per customer: {len(historical_loans) / historical_loans['customerid'].nunique():.1f}")

    # Check how many customers have historical data
    customers_with_history = set(historical_loans['customerid'].unique())
    customers_without_history = set(demographics['customerid'].unique()) - customers_with_history
    print(f"   Customers WITH history: {len(customers_with_history):,}")
    print(f"   Customers WITHOUT history: {len(customers_without_history):,}")

# ============================================================================
# CHUNK 5: Save Processed Data
# ============================================================================

def save_day1_outputs(demographics, current_loans, historical_loans):
    """Save loaded datasets to processed/ folder"""
    print("\n" + "=" * 70)
    print("SAVING DAY 1 OUTPUTS")
    print("=" * 70)

    demographics.to_csv(PROCESSED_DATA_DIR / 'demographics_day1.csv', index=False)
    print(f"✅ Saved: data/processed/demographics_day1.csv")

    current_loans.to_csv(PROCESSED_DATA_DIR / 'current_loans_day1.csv', index=False)
    print(f"✅ Saved: data/processed/current_loans_day1.csv")

    historical_loans.to_csv(PROCESSED_DATA_DIR / 'historical_loans_day1.csv', index=False)
    print(f"✅ Saved: data/processed/historical_loans_day1.csv")

    print("\n🎉 Day 1 Complete! You now have all 3 datasets loaded and ready for Day 2.")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 1: Load Kaggle Credit Risk Data")
    print("=" * 70)

    # Load all 3 datasets
    demographics = load_demographics()
    current_loans = load_current_performance()
    historical_loans = load_historical_loans()

    # Explore the data
    explore_datasets(demographics, current_loans, historical_loans)

    # Save outputs
    save_day1_outputs(demographics, current_loans, historical_loans)
```

**Run the final version:**
```bash
python src/data/load_kaggle_data.py
```

**Expected output:**
```
======================================================================
SAVING DAY 1 OUTPUTS
======================================================================
✅ Saved: data/processed/demographics_day1.csv
✅ Saved: data/processed/current_loans_day1.csv
✅ Saved: data/processed/historical_loans_day1.csv

🎉 Day 1 Complete! You now have all 3 datasets loaded and ready for Day 2.
```

---

## ✅ Day 1 Checklist

Before moving to Day 2, verify:

- [ ] Folder structure created (data/raw, data/processed, data/final, models, src/data, results)
- [ ] Dependencies installed (pandas, numpy, matplotlib, seaborn)
- [ ] 3 Kaggle CSV files in `data/raw/`
- [ ] Script runs without errors: `python src/data/load_kaggle_data.py`
- [ ] 3 output files in `data/processed/`:
  - `demographics_day1.csv`
  - `current_loans_day1.csv`
  - `historical_loans_day1.csv`

---

## 🎯 What's Next?

**Day 2**: Merge the 3 datasets into one master dataset
- Join demographics + current loans (on customerid)
- Aggregate historical loan behavior per customer
- Create 15+ basic features from the merge
- Output: `data/processed/master_dataset_day2.csv`

---

## 📚 Key Concepts Learned

1. **Project Structure**: Organized folders for raw data, processed data, models, and results
2. **Kaggle Dataset Schema**: Understanding the 3 tables and their relationships
3. **Data Loading**: Using pandas to read CSV files efficiently
4. **Data Exploration**: Checking shape, columns, missing values, and statistics
5. **Incremental Development**: Building code in chunks, testing as you go

**Time to complete**: ~45 minutes
**Files created**: 7 (1 script + 3 requirements.txt + 3 processed CSVs)
