# Week 1, Day 3 — Data Ingestion: Building the Loader Step by Step

## What You Will Learn Today

- How to place your real CBN data file into the project
- What the 34 columns in your dataset represent
- How to build a reusable data ingestion script **one piece at a time**
- What "schema validation" means and why you need it
- How to separate core variables from the full dataset

## Why This Matters

The ingestion script is the front door of your entire system. Every model, every plot, every result starts here. If bad data gets through this door, everything downstream is wrong. That is why we validate the data on the way in -- checking column names, data types, and date formats.

## Why This Matters for Nigeria

Nigerian macro data from the CBN Statistical Bulletin sometimes has formatting issues -- dates in different formats, missing months, values stored as text instead of numbers. Your ingestion script must handle these robustly. Unlike the old version of this guide, which used synthetic placeholder data, you are now working with your **own real data** from the Central Bank of Nigeria. This means your results are thesis-ready from day one.

---

## Part 1: Place Your Data

You should already have a file called `cbn_infl_data.csv` that you downloaded or exported from the CBN Statistical Bulletin. This file contains monthly Nigerian macroeconomic data with 34 columns.

Copy your `cbn_infl_data.csv` file into the `data/raw/` folder inside the project:

```
Nigerian_Inflation_Predictor/
    data/
        raw/
            cbn_infl_data.csv   <-- put it here
```

Open the file in a text editor (not Excel) and verify that the first few lines look something like this:

```
date,mpr,omo,crr,lr,tbr,infl,inflcore,inflfood,fcpi,ccpi,acpi,...
2007-01-01,10.0,... (values for all columns)
2007-02-01,10.0,... (values for all columns)
2007-03-01,10.0,... (values for all columns)
```

The exact values and start date depend on your data. The important things are: (1) the first row is a header with column names, and (2) values are separated by commas. Your dates are in CBN's abbreviated format (`8-Jan`, `8-Feb`, etc.) — the ingestion script handles this automatically using `format="%y-%b"`.

### Understanding All 34 Columns

Your dataset contains 34 columns spanning five categories. Here is the complete reference:

**Monetary Policy (4 columns)**

| Column | Full Name | Description |
|--------|-----------|-------------|
| `mpr` | Monetary Policy Rate | The CBN's benchmark interest rate (%). This is the primary tool the CBN uses to control inflation. When the CBN raises the MPR, borrowing becomes more expensive across the economy. |
| `omo` | Open Market Operations | Rate on CBN open market operations (%). The CBN buys and sells government securities to control money supply. |
| `crr` | Cash Reserve Ratio | Percentage of deposits banks must hold with CBN (%). A higher CRR means banks have less money to lend. |
| `lr` | Liquidity Ratio | Minimum ratio of liquid assets banks must maintain (%). Another tool to control how much banks can lend. |

**Inflation (9 columns)**

| Column | Full Name | Description |
|--------|-----------|-------------|
| `infl` | Headline Inflation | Year-over-year change in the all-items Consumer Price Index (%). This is the number you see in news headlines. |
| `inflcore` | Core Inflation | Inflation excluding volatile food and energy prices (%). Shows the underlying trend. |
| `inflfood` | Food Inflation | Year-over-year change in food prices (%). Often the largest component of Nigerian inflation. |
| `fcpi` | Food CPI | Food component of the Consumer Price Index (index level). |
| `ccpi` | Core CPI | Core component of the Consumer Price Index (index level). |
| `acpi` | All-items CPI | The full Consumer Price Index (index level). |
| `fcpi_r` / `ccpi_r` / `acpi_r` | Rural CPI breakdowns | Food, core, and all-items CPI for rural areas. |
| `fcpi_u` / `ccpi_u` / `acpi_u` | Urban CPI breakdowns | Food, core, and all-items CPI for urban areas. |

**Exchange Rates (4 columns)**

| Column | Full Name | Description |
|--------|-----------|-------------|
| `exo` | Official Exchange Rate | Naira per US dollar at the official CBN rate. |
| `exbdc` | Bureau De Change Rate | Naira per US dollar at bureau de change (parallel market). Often higher than the official rate. |
| `neer` | Nominal Effective Exchange Rate | Trade-weighted average of the Naira against a basket of currencies (index). |
| `reer` | Real Effective Exchange Rate | NEER adjusted for inflation differentials (index). Shows real competitiveness. |

**Interest Rates (10 columns)**

| Column | Full Name | Description |
|--------|-----------|-------------|
| `tbr` | Treasury Bill Rate | Yield on Nigerian Treasury Bills (%). The benchmark short-term risk-free rate. |
| `sr` | Savings Rate | Interest rate on savings deposits (%). |
| `ir7d` | 7-Day Rate | Interbank rate for 7-day lending (%). |
| `ir1m` | 1-Month Rate | Interbank rate for 1-month lending (%). |
| `ir3m` | 3-Month Rate | Interbank rate for 3-month lending (%). |
| `ir6m` | 6-Month Rate | Interbank rate for 6-month lending (%). |
| `ir12m` | 12-Month Rate | Interbank rate for 12-month lending (%). |
| `irover12m` | Over 12-Month Rate | Interbank rate for lending over 12 months (%). |
| `plr` | Prime Lending Rate | Rate banks charge their most creditworthy customers (%). |
| `mlr` | Maximum Lending Rate | Highest rate banks charge on loans (%). |
| `icr` | Interbank Call Rate | Overnight lending rate between banks (%). |
| `obb` | Open Buy Back | Rate on repurchase agreements in the interbank market (%). |

**Capital Market (2 columns)**

| Column | Full Name | Description |
|--------|-----------|-------------|
| `mktcap` | Market Capitalization | Total value of all listed shares on the Nigerian Stock Exchange (billions of Naira). |
| `asi` | All Share Index | The main Nigerian stock market index. |

### The 4 Core Variables

Our inflation prediction model uses 4 core variables:

- **mpr** -- Monetary Policy Rate (%). The CBN's main policy lever.
- **infl** -- Headline inflation (%). The variable we are trying to predict.
- **exo** -- Official exchange rate (Naira per dollar). Captures import price pressure.
- **tbr** -- Treasury Bill Rate (%). Captures market expectations of monetary policy.

The model uses these 4 because they represent the key channels of monetary policy transmission: the policy rate itself (mpr), the outcome we are forecasting (infl), the exchange rate channel (exo), and market interest rates (tbr).

The full 34-column dataset is preserved for future extensions. You might later add food inflation, the parallel exchange rate, or stock market indicators as additional features. But we start simple.

---

## Part 2: Build `data_ingestion/ingest.py` -- Step by Step

We are going to build the ingestion script **one piece at a time**. After each step, you will run the script and see output. This way, if something breaks, you know exactly which piece caused it.

Do **not** skip ahead. Type each step, run it, and verify the output before moving on.

At each step, we show you the **complete file**. Delete everything in `data_ingestion/ingest.py` and replace it with exactly what is shown. No guessing where to put things.

---

### Step 1: Bare Minimum -- Load and Print Shape

Delete everything in `data_ingestion/ingest.py` and replace it with this:

```python
"""Data ingestion module for the Nigerian Inflation Predictor."""

import os
import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DEFAULT_FILE = "cbn_infl_data.csv"

CORE_COLUMNS = ["mpr", "infl", "exo", "tbr"]


def load_raw_data(filename=None):
    """Load raw macroeconomic data from a CSV file."""

    if filename is None:
        filename = DEFAULT_FILE

    filepath = os.path.join(RAW_DATA_DIR, filename)

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {filename}")
    print(f"Columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
```

**Run it:**

```bash
python -m data_ingestion.ingest
```

**Expected output** (the exact numbers depend on your data):

```
Loaded XX rows and 34 columns from cbn_infl_data.csv
Columns: ['date', 'mpr', 'omo', 'crr', 'lr', 'tbr', 'infl', ...]
   (first 5 rows of your data)
```

If you see `FileNotFoundError`, your CSV is not in the right place. It must be at `data/raw/cbn_infl_data.csv`.

**What this code does:** `os.path.join()` builds file paths in a way that works on any operating system. `os.path.dirname(__file__)` means "the folder this script lives in" -- so we start from `data_ingestion/` and go up one level (`..`) to the project root, then down into `data/raw/`. `pd.read_csv(filepath)` reads the CSV and returns a DataFrame. `CORE_COLUMNS` defines the 4 variables our model will use -- we will use this list later.

---

### Step 2: Add Date Parsing

Delete everything in `data_ingestion/ingest.py` and replace it with this:

```python
"""Data ingestion module for the Nigerian Inflation Predictor."""

import os
import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DEFAULT_FILE = "cbn_infl_data.csv"

CORE_COLUMNS = ["mpr", "infl", "exo", "tbr"]


def load_raw_data(filename=None):
    """Load raw macroeconomic data from a CSV file."""

    if filename is None:
        filename = DEFAULT_FILE

    filepath = os.path.join(RAW_DATA_DIR, filename)

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {filename}")

    # Convert the date column from text to actual datetime objects.
    # CBN data uses "8-Jan", "8-Feb" format: 2-digit year + abbreviated month.
    # format="%y-%b" tells pandas: %y = 2-digit year, %b = month abbreviation.
    df["date"] = pd.to_datetime(df["date"], format="%y-%b")

    # Sort by date (oldest first) and set date as the index
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    df.index.freq = "MS"

    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
```

**Run it:**

```bash
python -m data_ingestion.ingest
```

**Expected output** (your dates will depend on your data):

```
Loaded 126 rows and 35 columns from cbn_infl_data.csv
Date range: 2008-01-01 to 2019-01-01
            mpr   omo   crr  ...
date
2008-01-01  ...   ...   ...  ...
```

**What changed:** Three new operations.

`pd.to_datetime(df["date"], format="%y-%b")` converts the date column from the CBN's abbreviated format (`"8-Jan"`, `"19-Jan"`) into proper datetime objects. The `format` parameter tells pandas exactly how to read the date: `%y` means a 2-digit year (so `8` becomes `2008`, `19` becomes `2019`) and `%b` means an abbreviated month name (like `Jan`, `Feb`, `Mar`). Without the `format` parameter, pandas would fail because it cannot guess this unusual format automatically.

`sort_values("date")` ensures rows are in chronological order, even if the CSV was not sorted. `set_index("date")` makes the date column the row label -- notice how dates now appear on the left side of the table instead of numbered rows.

`df.index.freq = "MS"` tells pandas this is a monthly time series where each observation falls on the first of the month. `"MS"` stands for "Month Start." This is important because time series operations like `.diff()` and `.shift()` need to know the frequency to work correctly.

---

### Step 3: Add Column Validation

Delete everything in `data_ingestion/ingest.py` and replace it with this:

```python
"""Data ingestion module for the Nigerian Inflation Predictor."""

import os
import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DEFAULT_FILE = "cbn_infl_data.csv"

CORE_COLUMNS = ["mpr", "infl", "exo", "tbr"]


def load_raw_data(filename=None):
    """Load raw macroeconomic data from a CSV file."""

    if filename is None:
        filename = DEFAULT_FILE

    filepath = os.path.join(RAW_DATA_DIR, filename)

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {filename}")

    # Validate that all core columns exist
    missing = set(CORE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required core columns: {missing}\n"
            f"Your CSV has these columns: {list(df.columns)}\n"
            f"Required core columns are: {CORE_COLUMNS}"
        )
    print(f"Core columns verified: {CORE_COLUMNS}")

    # Convert the date column from text to actual datetime objects.
    # CBN data uses "8-Jan", "8-Feb" format: 2-digit year + abbreviated month.
    df["date"] = pd.to_datetime(df["date"], format="%y-%b")

    # Sort by date (oldest first) and set date as the index
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    df.index.freq = "MS"

    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
```

**Run it:**

```bash
python -m data_ingestion.ingest
```

**Expected output** (new line in the middle):

```
Loaded 126 rows and 35 columns from cbn_infl_data.csv
Core columns verified: ['mpr', 'infl', 'exo', 'tbr']
Date range: 2008-01-01 to 2019-01-01
```

If you see a `ValueError` about missing columns, open your CSV in a text editor and check the very first line (the header row). The column names must match exactly -- lowercase, no spaces. For example, if your CSV uses `inflation` instead of `infl`, or `exchange_rate` instead of `exo`, you will need to rename the columns in your CSV file.

**What the new code does:** `set(CORE_COLUMNS) - set(df.columns)` computes the **set difference** -- it takes everything in `CORE_COLUMNS` and subtracts everything in `df.columns`. Whatever is left over is missing. If nothing is left, the set is empty (which Python treats as `False`), so the `if` block does not run. If something IS left, we crash immediately with a clear error message. This is called **schema validation** -- checking that the data matches the expected structure before doing anything else. Catching this here saves you from a confusing `KeyError` deep inside a model script weeks from now.

---

### Step 4: Add Type Conversion

Delete everything in `data_ingestion/ingest.py` and replace it with this:

```python
"""Data ingestion module for the Nigerian Inflation Predictor."""

import os
import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DEFAULT_FILE = "cbn_infl_data.csv"

CORE_COLUMNS = ["mpr", "infl", "exo", "tbr"]


def load_raw_data(filename=None):
    """Load raw macroeconomic data from a CSV file."""

    if filename is None:
        filename = DEFAULT_FILE

    filepath = os.path.join(RAW_DATA_DIR, filename)

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {filename}")

    # Validate that all core columns exist
    missing = set(CORE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required core columns: {missing}\n"
            f"Your CSV has these columns: {list(df.columns)}\n"
            f"Required core columns are: {CORE_COLUMNS}"
        )
    print(f"Core columns verified: {CORE_COLUMNS}")

    # Convert the date column from text to actual datetime objects.
    # CBN data uses "8-Jan", "8-Feb" format: 2-digit year + abbreviated month.
    df["date"] = pd.to_datetime(df["date"], format="%y-%b")

    # Convert core columns to float, replacing bad values with NaN
    for col in CORE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by date (oldest first) and set date as the index
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    df.index.freq = "MS"

    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"Core column types:\n{df[CORE_COLUMNS].dtypes}")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
```

**Run it:**

```bash
python -m data_ingestion.ingest
```

**Expected output** (new lines at the bottom of the summary):

```
Loaded 126 rows and 35 columns from cbn_infl_data.csv
Core columns verified: ['mpr', 'infl', 'exo', 'tbr']
Date range: 2008-01-01 to 2019-01-01
Core column types:
mpr     float64
infl    float64
exo     float64
tbr     float64
dtype: object
```

All four core columns should show `float64`. If any show `object`, that means some cells contain text that could not be converted to numbers -- those cells are now `NaN` and will be handled during cleaning on Day 4.

**What the new code does:** `pd.to_numeric(df[col], errors="coerce")` converts each value in the column to a number. The `errors="coerce"` parameter is the key part: if a value cannot be converted (for example, if someone typed "N/A", a dash, or an empty string in the CSV), instead of crashing the entire script, it quietly replaces that value with `NaN` (Not a Number). You can detect and handle those missing values later in the cleaning step. Without `errors="coerce"`, a single bad cell in your entire dataset would crash the script with a `ValueError`.

---

### Step 5: Add Core Variable Selection (Final Version)

Delete everything in `data_ingestion/ingest.py` and replace it with this:

```python
"""
Data ingestion module for the Nigerian Inflation Predictor.

This script loads raw CSV data from data/raw/, validates that it has
the correct columns and data types, and returns a pandas DataFrame
ready for cleaning and analysis.

The raw CSV contains 34 columns from the CBN Statistical Bulletin.
The model uses 4 core variables: mpr, infl, exo, tbr.
The full dataset is preserved; core variable selection is a separate step.

Usage:
    python -m data_ingestion.ingest
"""

import os
import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DEFAULT_FILE = "cbn_infl_data.csv"

CORE_COLUMNS = ["mpr", "infl", "exo", "tbr"]


def load_raw_data(filename=None):
    """Load raw macroeconomic data from a CSV file.

    Returns a DataFrame with ALL columns from the CSV, indexed by date.
    The full dataset is preserved so that downstream scripts can access
    any of the 34 variables for extensions or exploratory analysis.
    """

    if filename is None:
        filename = DEFAULT_FILE

    filepath = os.path.join(RAW_DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Make sure you have placed your CSV in the data/raw/ folder."
        )

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {filename}")

    # Validate that all core columns exist
    missing = set(CORE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required core columns: {missing}\n"
            f"Your CSV has these columns: {list(df.columns)}\n"
            f"Required core columns are: {CORE_COLUMNS}"
        )
    print(f"Core columns verified: {CORE_COLUMNS}")

    # Convert the date column from text to actual datetime objects.
    # CBN data uses "8-Jan", "8-Feb" format: 2-digit year + abbreviated month.
    df["date"] = pd.to_datetime(df["date"], format="%y-%b")

    # Convert core columns to float, replacing bad values with NaN
    for col in CORE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by date (oldest first) and set date as the index
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    df.index.freq = "MS"

    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"Missing values in core columns:\n{df[CORE_COLUMNS].isnull().sum()}")
    return df


def select_core_variables(df):
    """Select only the 4 core variables used by the prediction model.

    Parameters:
        df: DataFrame returned by load_raw_data() (contains all 34 columns)

    Returns:
        DataFrame with only the columns: mpr, infl, exo, tbr
    """
    missing = set(CORE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Cannot select core variables. Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    return df[CORE_COLUMNS].copy()


if __name__ == "__main__":
    print("=" * 60)
    print("NIGERIAN INFLATION PREDICTOR -- DATA INGESTION")
    print("=" * 60)

    # Load the full dataset (all 34 columns)
    df = load_raw_data()

    print(f"\nFull dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nFirst 5 rows (showing core columns only):")
    print(df[CORE_COLUMNS].head())

    print(f"\nLast 5 rows (showing core columns only):")
    print(df[CORE_COLUMNS].tail())

    # Select only the 4 core variables
    core_df = select_core_variables(df)

    print(f"\nCore dataset shape: {core_df.shape[0]} rows, {core_df.shape[1]} columns")
    print(f"\nCore data types:")
    print(core_df.dtypes)

    print(f"\nCore summary statistics:")
    print(core_df.describe().round(2))
```

Step 5 above is your final complete file.

**Run it:**

```bash
python -m data_ingestion.ingest
```

**Expected output** (exact numbers depend on your data):

```
============================================================
NIGERIAN INFLATION PREDICTOR -- DATA INGESTION
============================================================
Loaded XX rows and 34 columns from cbn_infl_data.csv
Core columns verified: ['mpr', 'infl', 'exo', 'tbr']
Date range: YYYY-MM-DD to YYYY-MM-DD
Missing values in core columns:
mpr     0
infl    0
exo     0
tbr     0
dtype: int64

Full dataset shape: XX rows, 33 columns
(33 because date is now the index, not a column)

First 5 rows (showing core columns only):
              mpr   infl     exo    tbr
date
YYYY-MM-DD   ...    ...     ...    ...

Last 5 rows (showing core columns only):
              mpr   infl     exo    tbr
date
YYYY-MM-DD   ...    ...     ...    ...

Core dataset shape: XX rows, 4 columns

Core data types:
mpr     float64
infl    float64
exo     float64
tbr     float64
dtype: object

Core summary statistics:
         mpr    infl     exo     tbr
count  XX.00  XX.00   XX.00   XX.00
mean    ...    ...     ...     ...
std     ...    ...     ...     ...
min     ...    ...     ...     ...
25%     ...    ...     ...     ...
50%     ...    ...     ...     ...
75%     ...    ...     ...     ...
max     ...    ...     ...     ...
```

**What the new code does:** We added two things.

First, the `select_core_variables(df)` function takes the full 34-column DataFrame and returns a new DataFrame with only the 4 core columns. The `.copy()` at the end is important -- it creates an independent copy of the data, so modifying the core DataFrame later will not accidentally change the full DataFrame. This function is separate from `load_raw_data()` on purpose: other scripts might want access to all 34 columns (for example, to plot food inflation or the parallel exchange rate), so the loader always returns everything.

Second, the `if __name__ == "__main__"` block now demonstrates both functions: it loads the full dataset, prints a summary, then selects the core variables and prints their summary statistics. `df.describe()` computes count, mean, standard deviation, min, max, and quartiles for every column -- a quick sanity check that the values look reasonable for Nigerian macro data.

---

## Part 3: Commit

```bash
git add data/raw/cbn_infl_data.csv data_ingestion/ingest.py
git commit -m "Day 3: Add CBN dataset and data ingestion script"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: Data file not found` | Your CSV is not in the right place. It must be at `data/raw/cbn_infl_data.csv` -- not in the project root, not in `data/`, not in `data_ingestion/`. Check the path. |
| `ValueError: Missing required core columns` | Your CSV header row has different column names. They must include `mpr`, `infl`, `exo`, and `tbr` -- lowercase, no spaces. Open the CSV in a text editor (not Excel) and check the very first line. |
| `KeyError: 'date'` | Your CSV does not have a column called `date`. Check the header row. If the date column has a different name (like `Date` or `period`), rename it to `date` in the CSV. |
| `ParserError` or garbled output | Your CSV might be using semicolons or tabs instead of commas. Open it in a text editor and verify each value is separated by a comma. |
| Output shows NaN values in the "Missing values" summary | One or more cells in your CSV contain text that cannot be converted to a number (like "N/A", a dash, or an empty cell). These will be handled during cleaning on Day 4. |
| `ModuleNotFoundError: No module named 'pandas'` | Your virtual environment is not activated. Run `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows) before running the script. |
| `ModuleNotFoundError: No module named 'data_ingestion'` | You are not in the project root directory. `cd` into `Nigerian_Inflation_Predictor/` before running the command. The `-m` flag expects to find a `data_ingestion/` folder in the current directory. |
| `ValueError: cannot set freq "MS"` | Your data may have irregular date spacing (not exactly monthly). Check your CSV for duplicate months or dates that do not fall on the first of the month. |

---

## Check Your Understanding

**1. "Why do we validate columns instead of just reading the file?"**

If someone gives you a CSV with different column names (for example, "MPR" instead of "mpr", or "inflation" instead of "infl"), the script will not crash at the loading step -- it will crash much later, deep inside a model or a plot, with a confusing `KeyError`. Validating upfront gives you a clear, immediate error message at the exact point of failure. This is a general principle: catch problems as early as possible.

**2. "Why does load_raw_data return all 34 columns instead of just the 4 core columns?"**

Because the full dataset has analytical value beyond the core model. You might want to plot food inflation against headline inflation, compare the official and parallel exchange rates, or add stock market data as an additional predictor. By loading everything and selecting core columns as a separate step, the ingestion script is flexible enough for any downstream use. The `select_core_variables()` function exists as a convenience for scripts that only need the 4 core columns.

**3. "What does `errors='coerce'` do in `pd.to_numeric()`?"**

It replaces values that cannot be converted to numbers with `NaN` (Not a Number) instead of crashing the entire script. Without it, a single bad value like "N/A" or "-" in one cell of your CSV would stop everything. With `coerce`, the script keeps going and you can deal with the missing values later in the cleaning step (Day 4). This is called **defensive programming** -- handling bad input gracefully instead of crashing.

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Dataset | `data/raw/cbn_infl_data.csv` | Real CBN data with 34 monthly macroeconomic variables |
| Ingestion script | `data_ingestion/ingest.py` | Loads all 34 columns, validates core columns, parses dates, sets index |

**Functions in `data_ingestion/ingest.py`:**

- `load_raw_data(filename)` -- Loads the full CSV, parses dates, validates core columns, converts types, returns a DataFrame with all 34 columns indexed by date.
- `select_core_variables(df)` -- Takes the full DataFrame and returns only the 4 core columns (mpr, infl, exo, tbr).

**Packages installed today:** None (using pandas from Day 2)

**Tomorrow (Day 4):** You will build the data cleaning pipeline -- handling missing values, enforcing monthly frequency, and validating value ranges for the core variables.
