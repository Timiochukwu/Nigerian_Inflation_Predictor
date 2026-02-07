# Week 1, Day 4 — Data Cleaning: Missing Values, Frequency & Validation

## What You'll Learn Today

- How to enforce a strict monthly frequency on your time series (no gaps allowed)
- How to detect and handle missing values using interpolation
- How to validate that each variable falls within a realistic range
- How to save the cleaned dataset so every downstream script uses the same data

## Why This Matters

Econometric models — ARDL, VAR, Granger causality — all require **complete, regularly spaced time series**. If your data has a missing month, your lag structure is broken. If February 2015 is missing, then what your model thinks is "one month ago" is actually "two months ago," and every coefficient it estimates is wrong.

This is not a theoretical problem. Real-world Nigerian macro data from the CBN Statistical Bulletin and the National Bureau of Statistics frequently has:

- **Missing months** where data was not published or was published late.
- **Impossible values** from data entry errors (e.g., an inflation rate of 999% or a negative exchange rate).
- **Irregular frequency** where some observations are quarterly instead of monthly, or a month is duplicated.

The cleaning script you build today is the wall between messy reality and the clean, reliable dataset your models need. Everything downstream — your EDA, your stationarity tests, your ARDL model, your VAR impulse responses — depends on this step being done correctly.

---

## Packages Installed Today: None

pandas (installed on Day 2) provides everything you need for data cleaning: reindexing, interpolation, range checking, and CSV export. No new packages required.

---

## Full Code

Create this file inside the `data_processing/` folder. If you followed Day 3, this folder already exists and contains an `__init__.py` file. If it does not, create the folder and add an empty `__init__.py` inside it.

**File: `data_processing/clean.py`**

```python
"""
Data cleaning module for the Nigerian Inflation Predictor.

This script takes the raw DataFrame loaded by the ingestion module (Day 3)
and produces a clean, validated, gap-free monthly time series ready for
analysis.

Cleaning steps:
1. Enforce monthly frequency — ensure every month from start to end is present.
2. Handle missing values — fill small gaps using linear interpolation.
3. Validate ranges — flag or clip values that fall outside realistic bounds.
4. Drop any remaining rows with missing data.
5. Save the cleaned data to data/processed/cleaned_data.csv.

Usage:
    python -m data_processing.clean
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
# os.path.dirname(__file__) gives the folder this script lives in
# (data_processing/). We go up one level (..) to reach the project root,
# then into the data/processed/ subfolder.

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# The filename for the cleaned output. Every downstream script (EDA,
# stationarity tests, models) will read from this exact file.
PROCESSED_FILE = "cleaned_data.csv"

# ---------------------------------------------------------------------------
# Valid ranges for each variable
# ---------------------------------------------------------------------------
# These ranges define what values are considered "realistic" for Nigerian
# macroeconomic data over the 2000-2024 period. Values outside these
# bounds are likely data entry errors or unit conversion mistakes.
#
# How these ranges were chosen:
#   mpr (1.0 to 35.0):
#       The CBN Monetary Policy Rate has historically ranged from about
#       6% to 27.5%. We use a wider range (1-35%) to allow for future
#       changes without falsely flagging valid data.
#
#   inflation (-5.0 to 50.0):
#       Nigeria has never had deflation (negative inflation), but we
#       allow down to -5% just in case. The highest inflation in our
#       sample is about 35%, but we allow up to 50% for safety.
#
#   exchange_rate (50.0 to 2000.0):
#       The Naira/USD rate started around 92 in 2000 and reached about
#       1,680 in late 2024. The range 50-2000 covers this with room
#       to spare.
#
#   m2 (100.0 to 500000.0):
#       Broad money supply (in billions of Naira) started around 1,070
#       in 2000 and grew to about 285,000 by 2024. The range 100-500,000
#       accommodates both historical and near-future values.

VALID_RANGES = {
    "mpr": (1.0, 35.0),
    "inflation": (-5.0, 50.0),
    "exchange_rate": (50.0, 2000.0),
    "m2": (100.0, 500000.0),
}


def enforce_monthly_frequency(df):
    """
    Ensure the DataFrame has exactly one row per month with no gaps.

    Why this matters:
    -----------------
    Time series models use "lags" — they look at the value 1 month ago,
    2 months ago, etc. If a month is missing, the lag structure is wrong.
    For example, if March 2015 is missing, then what the model sees as
    "the value 1 month before April 2015" is actually February 2015
    (2 months before), not March. This silently corrupts every result.

    How it works:
    -------------
    1. We create a complete date range from the first to the last date
       in the data, with freq="MS" (Month Start — the 1st of each month).
    2. We reindex the DataFrame to this complete range.
    3. Any month that was missing in the original data will now appear
       as a row filled with NaN (Not a Number = missing).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex (from the ingestion script).

    Returns
    -------
    pd.DataFrame
        DataFrame reindexed to a complete monthly frequency.
        Missing months are filled with NaN values.
    """

    # pd.date_range() creates a sequence of dates.
    # start = the earliest date in the data
    # end = the latest date in the data
    # freq="MS" means "Month Start" — generates the 1st of each month.
    #
    # Example: if your data runs from 2000-01-01 to 2024-12-01,
    # this creates a list of 300 dates: 2000-01-01, 2000-02-01, ..., 2024-12-01
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="MS")

    # Count how many months SHOULD exist vs how many we HAVE
    expected_months = len(full_range)
    actual_months = len(df)

    # reindex() reshapes the DataFrame to match the new index.
    # If a date in full_range does not exist in df, that row gets NaN values.
    # If a date in df does not exist in full_range, that row is dropped.
    df = df.reindex(full_range)

    # Give the index a clear name so downstream scripts know what it is
    df.index.name = "date"

    # Report what happened
    gaps = expected_months - actual_months
    if gaps > 0:
        print(f"  Monthly frequency enforced: {gaps} missing month(s) detected and inserted as NaN.")
    else:
        print(f"  Monthly frequency verified: all {expected_months} months present, no gaps.")

    return df


def handle_missing_values(df):
    """
    Fill missing values using linear interpolation.

    What is interpolation?
    ----------------------
    Interpolation estimates missing values by drawing a straight line
    between the known values on either side of the gap.

    Example: if January = 10.0 and March = 14.0, but February is missing,
    linear interpolation fills February with 12.0 (the midpoint).

    For a longer gap: if January = 10.0, February = NaN, March = NaN,
    April = 16.0, then interpolation fills:
        February = 12.0 (one-third of the way from 10 to 16)
        March    = 14.0 (two-thirds of the way from 10 to 16)

    Why linear interpolation?
    -------------------------
    - It is simple and transparent. You can explain exactly how each
      missing value was estimated.
    - It preserves the local trend of the data.
    - It is conservative — it does not introduce artificial patterns
      or oscillations.
    - More sophisticated methods (spline, polynomial) can overfit and
      create unrealistic values, especially for economic data.

    Why limit=3?
    ------------
    The limit=3 parameter means: only fill gaps of 3 or fewer consecutive
    missing months. If 4 or more months in a row are missing, the gap is
    too large to trust a simple interpolation, and those values stay as
    NaN (they will be dropped later).

    This is a judgment call. For monthly macro data, 1-3 missing months
    can usually be estimated reasonably. More than that, and you risk
    inventing data rather than estimating it.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that may contain NaN values (from enforce_monthly_frequency
        or from the original data).

    Returns
    -------
    pd.DataFrame
        DataFrame with small gaps (up to 3 months) filled by interpolation.
        Larger gaps remain as NaN.
    """

    # Count missing values BEFORE interpolation so we can report what changed
    missing_before = df.isnull().sum().sum()

    # df.interpolate() fills NaN values.
    # method="linear" means: draw a straight line between the nearest
    #   known values on each side and use that line to estimate the missing ones.
    # limit=3 means: only fill if the gap is 3 or fewer consecutive NaNs.
    #   Gaps of 4+ consecutive NaNs are left untouched.
    # limit_direction="both" means: interpolate forward and backward.
    #   Without this, the first or last rows (if missing) cannot be filled
    #   because there is no "previous" or "next" value to interpolate from.
    df = df.interpolate(method="linear", limit=3, limit_direction="both")

    # Count missing values AFTER interpolation
    missing_after = df.isnull().sum().sum()
    filled = missing_before - missing_after

    # Report what happened
    if filled > 0:
        print(f"  Interpolation filled {filled} missing value(s).")
    else:
        print(f"  No missing values to interpolate.")

    if missing_after > 0:
        print(f"  WARNING: {missing_after} missing value(s) remain (gaps too large to interpolate).")

    return df


def validate_ranges(df):
    """
    Check that each variable falls within its expected realistic range.

    Why validate ranges?
    --------------------
    Data entry errors are common in Nigerian economic datasets. A decimal
    point in the wrong place can turn an inflation rate of 15.0% into
    150.0%. An exchange rate recorded in kobo instead of Naira would be
    100x too large. These errors will not cause your code to crash, but
    they will silently corrupt your model's results.

    Range validation catches these problems early, before they propagate
    through your entire analysis.

    What this function does:
    ------------------------
    For each variable, it checks whether any values fall outside the
    bounds defined in VALID_RANGES. If out-of-range values are found,
    it clips them to the nearest valid boundary and prints a warning.

    Clipping means:
    - If a value is below the minimum, it is set to the minimum.
    - If a value is above the maximum, it is set to the maximum.

    Example: if inflation = 999.0 and the valid range is (-5, 50),
    the value is clipped to 50.0.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric columns to validate.

    Returns
    -------
    pd.DataFrame
        DataFrame with out-of-range values clipped to valid bounds.
    """

    print("  Validating value ranges:")

    for col, (low, high) in VALID_RANGES.items():
        # Skip columns that do not exist in the DataFrame
        # (in case the user has a subset of columns)
        if col not in df.columns:
            print(f"    {col}: column not found, skipping.")
            continue

        # Count how many values are below the minimum
        below = (df[col] < low).sum()

        # Count how many values are above the maximum
        above = (df[col] > high).sum()

        if below > 0 or above > 0:
            # df[col].clip(lower, upper) replaces:
            #   values < lower  with  lower
            #   values > upper  with  upper
            #   values in range are left unchanged
            df[col] = df[col].clip(lower=low, upper=high)
            print(f"    {col}: {below} below {low}, {above} above {high} — clipped to [{low}, {high}]")
        else:
            print(f"    {col}: all values within [{low}, {high}] — OK")

    return df


def clean_data(df):
    """
    Run the full cleaning pipeline: frequency, interpolation, validation, drop NaN.

    This function orchestrates all the individual cleaning steps in the
    correct order:

    1. enforce_monthly_frequency() — fill any date gaps with NaN rows.
    2. handle_missing_values() — interpolate small gaps (up to 3 months).
    3. validate_ranges() — clip any out-of-range values.
    4. dropna() — remove any rows that still have missing values
       (gaps too large to interpolate).

    The order matters:
    - You must enforce frequency BEFORE interpolation, because interpolation
      needs NaN placeholders to know where the gaps are.
    - You must interpolate BEFORE range validation, because NaN values
      cannot be compared to numeric ranges.
    - You must drop NaN LAST, after all attempts to fill them have been made.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from the ingestion script (Day 3).

    Returns
    -------
    pd.DataFrame
        Fully cleaned DataFrame with no missing values, monthly frequency,
        and all values within valid ranges.
    """

    print("\n--- Step 1: Enforce Monthly Frequency ---")
    df = enforce_monthly_frequency(df)

    print("\n--- Step 2: Handle Missing Values (Linear Interpolation) ---")
    df = handle_missing_values(df)

    print("\n--- Step 3: Validate Value Ranges ---")
    df = validate_ranges(df)

    print("\n--- Step 4: Drop Remaining Missing Values ---")
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    dropped = rows_before - rows_after

    if dropped > 0:
        print(f"  Dropped {dropped} row(s) with remaining missing values.")
    else:
        print(f"  No rows dropped — dataset is complete.")

    # Print a final summary of the cleaned data
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"  Final row count:  {len(df)}")
    print(f"  Date range:       {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"  Missing values:   {df.isnull().sum().sum()}")
    print(f"  Columns:          {list(df.columns)}")
    print("=" * 60)

    return df


def save_cleaned_data(df):
    """
    Save the cleaned DataFrame to data/processed/cleaned_data.csv.

    Why save to a separate file?
    ----------------------------
    We NEVER modify the raw data file. The raw file in data/raw/ is your
    original source of truth. The cleaned file in data/processed/ is a
    derived product. If you ever need to change your cleaning logic (for
    example, using a different interpolation method), you can re-run this
    script and it will overwrite the processed file without touching the
    raw data.

    This separation between raw and processed data is a fundamental
    principle of reproducible research.

    Parameters
    ----------
    df : pd.DataFrame
        The fully cleaned DataFrame to save.

    Returns
    -------
    str
        The full file path where the data was saved.
    """

    # os.makedirs() creates the directory if it does not exist.
    # exist_ok=True means: do not raise an error if the directory
    # already exists. Without this, running the script twice would crash.
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Build the full output path
    output_path = os.path.join(PROCESSED_DIR, PROCESSED_FILE)

    # df.to_csv() writes the DataFrame to a CSV file.
    # The index (dates) is included by default, which is what we want
    # because downstream scripts will read it back with index_col="date".
    df.to_csv(output_path)

    print(f"\nCleaned data saved to: {output_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    return output_path


# ---------------------------------------------------------------------------
# Main execution block
# ---------------------------------------------------------------------------
# This runs only when you execute the script directly:
#   python -m data_processing.clean
# It does NOT run when another script imports functions from this file.

if __name__ == "__main__":
    # Import the ingestion function from the Day 3 module.
    # We import here (inside the if block) rather than at the top of the
    # file so that this module can be imported by other scripts without
    # triggering the ingestion import. This avoids circular import issues.
    from data_ingestion.ingest import load_raw_data

    print("=" * 60)
    print("NIGERIAN INFLATION PREDICTOR — DATA CLEANING")
    print("=" * 60)

    # Step 1: Load the raw data using the Day 3 ingestion script
    print("\nLoading raw data...")
    df = load_raw_data()

    # Step 2: Run the full cleaning pipeline
    print("\nCleaning data...")
    df = clean_data(df)

    # Step 3: Save the cleaned data
    save_cleaned_data(df)

    # Step 4: Show a preview of the cleaned data
    print(f"\nFirst 5 rows of cleaned data:")
    print(df.head())

    print(f"\nLast 5 rows of cleaned data:")
    print(df.tail())

    print(f"\nData types:")
    print(df.dtypes)
```

---

## How to Run

Make sure you are in the project root directory (`Nigerian_Inflation_Predictor/`), then run:

```bash
python -m data_processing.clean
```

The `-m` flag tells Python to run `data_processing/clean.py` as a module. This is the same pattern you used for the ingestion script on Day 3.

**Important:** The cleaning script depends on the ingestion script from Day 3. If you have not completed Day 3 yet, go back and do that first. The cleaning script calls `load_raw_data()` from `data_ingestion/ingest.py` to get the raw data.

---

## Expected Output

When you run the script, you should see something like this in your terminal:

```
============================================================
NIGERIAN INFLATION PREDICTOR — DATA CLEANING
============================================================

Loading raw data...
Loaded 300 observations
Date range: 2000-01-01 to 2024-12-01

Missing values per column:
mpr              0
inflation        0
exchange_rate    0
m2               0

Cleaning data...

--- Step 1: Enforce Monthly Frequency ---
  Monthly frequency verified: all 300 months present, no gaps.

--- Step 2: Handle Missing Values (Linear Interpolation) ---
  No missing values to interpolate.

--- Step 3: Validate Value Ranges ---
  Validating value ranges:
    mpr: all values within [1.0, 35.0] — OK
    inflation: all values within [-5.0, 50.0] — OK
    exchange_rate: all values within [50.0, 2000.0] — OK
    m2: all values within [100.0, 500000.0] — OK

--- Step 4: Drop Remaining Missing Values ---
  No rows dropped — dataset is complete.

============================================================
CLEANING SUMMARY
============================================================
  Final row count:  300
  Date range:       2000-01-01 to 2024-12-01
  Missing values:   0
  Columns:          ['mpr', 'inflation', 'exchange_rate', 'm2']
============================================================

Cleaned data saved to: data_processing/../data/processed/cleaned_data.csv
  Rows: 300
  Columns: ['mpr', 'inflation', 'exchange_rate', 'm2']

First 5 rows of cleaned data:
             mpr  inflation  exchange_rate      m2
date
2000-01-01  13.5       6.62          92.34  1070.5
2000-02-01  13.5       6.93          92.55  1078.2
2000-03-01  13.5       7.87          92.69  1090.6
2000-04-01  13.5       8.13          93.05  1095.3
2000-05-01  13.5       8.68          95.10  1100.7
...
```

If you see the cleaning summary with 300 rows and 0 missing values, your cleaning pipeline is working correctly. The file `data/processed/cleaned_data.csv` has been created and is ready for the EDA script on Day 5.

---

## Step 5: Commit

```bash
git add data_processing/clean.py
git commit -m "Day 4: Add data cleaning script with frequency enforcement, interpolation, and range validation"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: Data file not found` | The cleaning script calls the ingestion script, which looks for `data/raw/nigeria_macro_data.csv`. Make sure that file exists. If you have not completed Day 3, go back and create the dataset first. |
| `ModuleNotFoundError: No module named 'data_ingestion'` | You are not running from the project root directory. Make sure you `cd` into `Nigerian_Inflation_Predictor/` before running the command. The `-m` flag requires you to be in the project root so Python can find the `data_ingestion` package. |
| `AttributeError: 'DataFrame' object has no attribute 'interpolate'` | This should not happen with pandas 2.x. If you see this, check your pandas version with `python -c "import pandas; print(pandas.__version__)"`. It should be 2.1.4 or later. |
| `PermissionError: [Errno 13] Permission denied` when saving | The script cannot write to the `data/processed/` folder. Check the folder permissions. On Linux/Mac, try `chmod -R 755 data/`. On Windows, right-click the folder, go to Properties, and make sure it is not read-only. |
| Output shows `WARNING: X missing value(s) remain` | Your data has a gap of 4 or more consecutive months that interpolation cannot fill. Check your raw CSV for large missing sections. You may need to obtain the missing data from CBN/NBS sources. |
| `KeyError: 'date'` | The ingestion script did not set the date column as the index. Make sure `load_raw_data()` in `data_ingestion/ingest.py` calls `df.set_index("date")` before returning the DataFrame. |

---

## Check Your Understanding

These are the kinds of questions an examiner will ask about your data cleaning process. Practice answering them before your defense.

### 1. "Why not just delete rows with missing values instead of interpolating?"

**Answer:** "Deleting rows with missing values (listwise deletion) breaks the time series structure. If March 2015 is missing and we delete it, then the gap between February and April is two months, not one. Every lag-based calculation — autocorrelation, ARDL lag terms, VAR impulse responses — assumes equal spacing between observations. Deletion violates that assumption.

Additionally, deletion wastes information. If we have valid data for February and April, we can make a reasonable estimate for March using interpolation. Deleting the row throws away the information contained in the neighbouring months.

The only time deletion is preferable is when a large block of data is missing (more than 3 consecutive months in our case), because interpolation over long gaps produces unreliable estimates. That is why we set `limit=3` — small gaps are interpolated, large gaps result in dropped rows."

### 2. "What does interpolation do, and why did you choose linear interpolation?"

**Answer:** "Interpolation estimates missing values by using the known values around the gap. Linear interpolation specifically draws a straight line between the last known value before the gap and the first known value after the gap, then reads off the estimated values along that line.

I chose linear interpolation for three reasons. First, it is simple and transparent — I can explain exactly how each filled value was calculated, which matters for thesis defense and reproducibility. Second, it preserves the local trend without introducing artificial oscillations or overshooting, which more complex methods like polynomial or spline interpolation can produce. Third, for monthly macroeconomic data where gaps are typically 1-2 months, the assumption of a roughly linear change between neighbouring months is reasonable — economic variables do not usually jump erratically from month to month.

More sophisticated alternatives exist — Kalman filtering, EM algorithm, multiple imputation — but for the small number of gaps typical in CBN data, linear interpolation is standard practice in the applied econometrics literature."

### 3. "Why enforce monthly frequency before doing anything else?"

**Answer:** "Enforcing monthly frequency must come first because it is the step that makes hidden gaps visible. In the raw data, if March 2015 is missing, the CSV simply does not have a row for that date. The DataFrame jumps from February to April, and pandas does not know a month is missing — it just sees consecutive rows.

By reindexing to a complete monthly date range using `pd.date_range(freq='MS')`, we explicitly insert a row for every month that should exist. Missing months appear as rows filled with NaN. Only after these NaN placeholders exist can the interpolation step detect and fill them.

If we tried to interpolate first (before enforcing frequency), there would be nothing to interpolate — the DataFrame would have no NaN values because the missing months are simply absent, not marked as NaN. The data would appear complete but would actually have temporal gaps that corrupt all subsequent time series analysis."

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Cleaning script | `data_processing/clean.py` | Enforces monthly frequency, interpolates missing values, validates ranges, saves cleaned data |
| Cleaned dataset | `data/processed/cleaned_data.csv` | Gap-free, validated monthly time series ready for EDA and modelling |

**Packages installed today:** None (using pandas from Day 2)

**Tomorrow (Day 5):** You will build the Exploratory Data Analysis (EDA) script — computing summary statistics, creating time series plots for each variable, and generating a correlation matrix heatmap.
