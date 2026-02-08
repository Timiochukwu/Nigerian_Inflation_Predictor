# Week 1, Day 4 — Data Cleaning: Building It Function by Function

## What You'll Learn Today

- How to enforce monthly frequency (no missing months)
- How to handle missing values with interpolation
- What value ranges are reasonable for Nigerian macro data
- How to save cleaned data

## Why This Matters

Econometric models need **complete time series** -- one value per month, no gaps. If March 2015 is missing from your data, what the model thinks is "one month before April 2015" is actually February 2015 (two months before). Every lag coefficient it estimates is silently wrong.

This is not a theoretical problem. Real-world Nigerian macro data from the CBN Statistical Bulletin and the National Bureau of Statistics frequently has gaps where data was not published, impossible values from data entry errors, or months that are simply absent from the file. The cleaning script you build today is the wall between that messy reality and the clean, reliable dataset your models need.

Today we build the cleaning script **one function at a time**. You will write a small piece, run it, see output, understand what it does, then add the next piece. By the end, you will have the complete pipeline -- but you will understand every line because you built it yourself.

---

## Packages Installed Today: None

pandas (installed on Day 2) provides everything you need: reindexing, interpolation, range checking, and CSV export. No new packages required.

---

## Building `data_processing/clean.py` -- Step by Step

If you followed Day 3, the `data_processing/` folder already exists and contains an `__init__.py` file. If it does not, create the folder and add an empty `__init__.py` inside it.

---

### Step 1: Create the File with Just Imports and Constants

Create a new file called `clean.py` inside the `data_processing/` folder. Start with only this.

**Your complete `data_processing/clean.py` should look like this:**

```python
import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PROCESSED_FILE = "cleaned_data.csv"

VALID_RANGES = {
    "mpr": (1.0, 35.0),
    "inflation": (-5.0, 50.0),
    "exchange_rate": (50.0, 2000.0),
    "m2": (100.0, 500000.0),
}

print("Constants loaded successfully.")
print(f"Cleaned data will be saved to: {PROCESSED_DIR}")
```

**Run it:**

```bash
python -m data_processing.clean
```

**Expected output:**

```
Constants loaded successfully.
Cleaned data will be saved to: data_processing/../data/processed
```

**What just happened:**

- `PROCESSED_DIR` builds a file path relative to the script's location. `os.path.dirname(__file__)` gives the folder this script lives in (`data_processing/`), then `..` goes up one level to the project root, then into `data/processed/`. This is where we will save the cleaned output.
- `PROCESSED_FILE` is the filename for the cleaned CSV. Every downstream script (EDA, stationarity tests, models) will read from this exact file.
- `VALID_RANGES` defines what values are "realistic" for Nigerian macro data between 2000 and 2024:
  - **mpr (1.0 to 35.0):** The CBN Monetary Policy Rate has historically ranged from about 6% to 27.5%. We use a wider range to allow for future changes without falsely flagging valid data.
  - **inflation (-5.0 to 50.0):** Nigeria has never had deflation, but we allow a small buffer below zero. The highest inflation in our sample is about 35%, but we allow up to 50% for safety.
  - **exchange_rate (50.0 to 2000.0):** The Naira/USD rate started around 92 in 2000 and reached about 1,680 in late 2024. This range covers the full history with room to spare.
  - **m2 (100.0 to 500000.0):** Broad money supply (in billions of Naira) started around 1,070 in 2000 and grew to about 285,000 by 2024. This range accommodates both historical and near-future values.

The `print` lines are just temporary -- they confirm the file loads without errors. We will replace them in the next step.

---

### Step 2: Add the First Function -- `enforce_monthly_frequency`

Now replace the two temporary `print(...)` lines with the first real function and a test block.

**Your complete `data_processing/clean.py` should look like this:**

```python
import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PROCESSED_FILE = "cleaned_data.csv"

VALID_RANGES = {
    "mpr": (1.0, 35.0),
    "inflation": (-5.0, 50.0),
    "exchange_rate": (50.0, 2000.0),
    "m2": (100.0, 500000.0),
}


def enforce_monthly_frequency(df):
    """Make sure there is exactly one row per month, no gaps."""
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="MS"
    )
    df_reindexed = df.reindex(full_range)
    df_reindexed.index.name = "date"

    added = len(df_reindexed) - len(df)
    if added > 0:
        print(f"  Inserted {added} missing month(s)")
    else:
        print(f"  No gaps found. All months present.")
    return df_reindexed


# Quick test
if __name__ == "__main__":
    from data_ingestion.ingest import load_raw_data
    df = load_raw_data()
    print("\nStep 1: Enforcing monthly frequency...")
    df = enforce_monthly_frequency(df)
    print(f"  Rows after: {len(df)}")
```

**Run it:**

```bash
python -m data_processing.clean
```

**Expected output:**

```
Loaded 300 observations
Date range: 2000-01-01 to 2024-12-01

Missing values per column:
mpr              0
inflation        0
exchange_rate    0
m2               0

Step 1: Enforcing monthly frequency...
  No gaps found. All months present.
  Rows after: 300
```

**What this function does:**

- `pd.date_range(start, end, freq="MS")` creates a complete list of dates from the first to the last month in your data. `freq="MS"` means "Month Start" -- it generates the 1st of every month. So if your data runs from 2000-01-01 to 2024-12-01, this creates a list of 300 dates.
- `df.reindex(full_range)` reshapes the DataFrame to match that complete date list. If a month exists in `full_range` but not in your data, that row gets filled with NaN (missing). If a month exists in your data but not in `full_range`, that row is dropped.
- Why does this matter? In the raw CSV, if March 2015 is missing, there is simply no row for that date. The DataFrame jumps from February to April, and pandas does not know a month is missing. By reindexing, we make hidden gaps visible -- they become NaN rows that the next step (interpolation) can detect and fill.

If your data has no gaps, the function reports "No gaps found" and the row count stays the same. If gaps exist, you will see how many were inserted.

---

### Step 3: Add the Second Function -- `handle_missing_values`

Now add the `handle_missing_values` function and update the `if __name__` block to also call it.

**Your complete `data_processing/clean.py` should look like this:**

```python
import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PROCESSED_FILE = "cleaned_data.csv"

VALID_RANGES = {
    "mpr": (1.0, 35.0),
    "inflation": (-5.0, 50.0),
    "exchange_rate": (50.0, 2000.0),
    "m2": (100.0, 500000.0),
}


def enforce_monthly_frequency(df):
    """Make sure there is exactly one row per month, no gaps."""
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="MS"
    )
    df_reindexed = df.reindex(full_range)
    df_reindexed.index.name = "date"

    added = len(df_reindexed) - len(df)
    if added > 0:
        print(f"  Inserted {added} missing month(s)")
    else:
        print(f"  No gaps found. All months present.")
    return df_reindexed


def handle_missing_values(df):
    """Fill short gaps (up to 3 months) using interpolation."""
    missing_before = df.isnull().sum()
    df_clean = df.interpolate(method="linear", limit=3)
    missing_after = df_clean.isnull().sum()

    for col in df_clean.columns:
        filled = missing_before[col] - missing_after[col]
        if filled > 0:
            print(f"  {col}: filled {filled} missing values")
        remaining = missing_after[col]
        if remaining > 0:
            print(f"  WARNING: {col} still has {remaining} missing values")

    if missing_before.sum() == 0:
        print("  No missing values found.")
    return df_clean


# Quick test
if __name__ == "__main__":
    from data_ingestion.ingest import load_raw_data
    df = load_raw_data()
    print("\nStep 1: Enforcing monthly frequency...")
    df = enforce_monthly_frequency(df)
    print(f"  Rows after: {len(df)}")

    print("\nStep 2: Handling missing values...")
    df = handle_missing_values(df)
```

**Run it:**

```bash
python -m data_processing.clean
```

**Expected output (new lines at the bottom):**

```
Step 2: Handling missing values...
  No missing values found.
```

**What interpolation means in plain English:**

Interpolation fills in missing values by drawing a straight line between the known values on either side of the gap.

Example: if January's inflation is 10.0 and March's inflation is 14.0, but February is missing, linear interpolation fills February with 12.0 -- the midpoint.

For a longer gap: if January = 10.0, February = NaN, March = NaN, April = 16.0, then:
- February gets 12.0 (one-third of the way from 10 to 16)
- March gets 14.0 (two-thirds of the way from 10 to 16)

The `limit=3` parameter means: only fill gaps of 3 or fewer consecutive missing months. If 4 or more months in a row are missing, the gap is too large to trust a simple straight-line estimate, and those values stay as NaN (they will be dropped later). This is a judgment call -- for monthly macro data, 1-3 missing months can usually be estimated reasonably. More than that, and you risk inventing data rather than estimating it.

Since our dataset currently has no missing values, the function reports "No missing values found." But if you ever use real CBN data with gaps, this function will catch and fill them.

---

### Step 4: Add the Third Function -- `validate_ranges`

Now add the `validate_ranges` function and update the `if __name__` block to also call it.

**Your complete `data_processing/clean.py` should look like this:**

```python
import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PROCESSED_FILE = "cleaned_data.csv"

VALID_RANGES = {
    "mpr": (1.0, 35.0),
    "inflation": (-5.0, 50.0),
    "exchange_rate": (50.0, 2000.0),
    "m2": (100.0, 500000.0),
}


def enforce_monthly_frequency(df):
    """Make sure there is exactly one row per month, no gaps."""
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="MS"
    )
    df_reindexed = df.reindex(full_range)
    df_reindexed.index.name = "date"

    added = len(df_reindexed) - len(df)
    if added > 0:
        print(f"  Inserted {added} missing month(s)")
    else:
        print(f"  No gaps found. All months present.")
    return df_reindexed


def handle_missing_values(df):
    """Fill short gaps (up to 3 months) using interpolation."""
    missing_before = df.isnull().sum()
    df_clean = df.interpolate(method="linear", limit=3)
    missing_after = df_clean.isnull().sum()

    for col in df_clean.columns:
        filled = missing_before[col] - missing_after[col]
        if filled > 0:
            print(f"  {col}: filled {filled} missing values")
        remaining = missing_after[col]
        if remaining > 0:
            print(f"  WARNING: {col} still has {remaining} missing values")

    if missing_before.sum() == 0:
        print("  No missing values found.")
    return df_clean


def validate_ranges(df):
    """Check if values are within reasonable ranges for Nigeria."""
    for col, (low, high) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        outliers = df[(df[col] < low) | (df[col] > high)]
        if len(outliers) > 0:
            print(f"  WARNING: {col} has {len(outliers)} values outside [{low}, {high}]")
            print(f"    Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")
        else:
            print(f"  {col}: all values within range")


# Quick test
if __name__ == "__main__":
    from data_ingestion.ingest import load_raw_data
    df = load_raw_data()
    print("\nStep 1: Enforcing monthly frequency...")
    df = enforce_monthly_frequency(df)
    print(f"  Rows after: {len(df)}")

    print("\nStep 2: Handling missing values...")
    df = handle_missing_values(df)

    print("\nStep 3: Validating ranges...")
    validate_ranges(df)
```

**Run it:**

```bash
python -m data_processing.clean
```

**Expected output (new lines at the bottom):**

```
Step 3: Validating ranges...
  mpr: all values within range
  inflation: all values within range
  exchange_rate: all values within range
  m2: all values within range
```

**Why we check ranges:**

Data entry errors are common in Nigerian economic datasets. A decimal point in the wrong place turns an inflation rate of 15.0% into 150.0%. An exchange rate recorded in kobo instead of Naira would be 100x too large. These errors will not cause your code to crash -- they will silently corrupt your model's results. Range validation catches these problems early, before they propagate through your entire analysis.

If any values fall outside the valid range, the function prints a warning with the actual min and max so you know exactly what is wrong. Since our dataset was carefully constructed, all values are within range. But with real CBN data, you would likely see warnings here.

---

### Step 5: Add the Wrapper and Save Functions

Now we tie everything together. Add the `clean_data` wrapper function and the `save_cleaned_data` function, then replace the `if __name__` block with the final version.

**Your complete `data_processing/clean.py` should look like this:**

```python
import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PROCESSED_FILE = "cleaned_data.csv"

VALID_RANGES = {
    "mpr": (1.0, 35.0),
    "inflation": (-5.0, 50.0),
    "exchange_rate": (50.0, 2000.0),
    "m2": (100.0, 500000.0),
}


def enforce_monthly_frequency(df):
    """Make sure there is exactly one row per month, no gaps."""
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="MS"
    )
    df_reindexed = df.reindex(full_range)
    df_reindexed.index.name = "date"

    added = len(df_reindexed) - len(df)
    if added > 0:
        print(f"  Inserted {added} missing month(s)")
    else:
        print(f"  No gaps found. All months present.")
    return df_reindexed


def handle_missing_values(df):
    """Fill short gaps (up to 3 months) using interpolation."""
    missing_before = df.isnull().sum()
    df_clean = df.interpolate(method="linear", limit=3)
    missing_after = df_clean.isnull().sum()

    for col in df_clean.columns:
        filled = missing_before[col] - missing_after[col]
        if filled > 0:
            print(f"  {col}: filled {filled} missing values")
        remaining = missing_after[col]
        if remaining > 0:
            print(f"  WARNING: {col} still has {remaining} missing values")

    if missing_before.sum() == 0:
        print("  No missing values found.")
    return df_clean


def validate_ranges(df):
    """Check if values are within reasonable ranges for Nigeria."""
    for col, (low, high) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        outliers = df[(df[col] < low) | (df[col] > high)]
        if len(outliers) > 0:
            print(f"  WARNING: {col} has {len(outliers)} values outside [{low}, {high}]")
            print(f"    Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")
        else:
            print(f"  {col}: all values within range")


def clean_data(df):
    """Run the full cleaning pipeline."""
    print("Step 1: Enforcing monthly frequency...")
    df = enforce_monthly_frequency(df)
    print("Step 2: Handling missing values...")
    df = handle_missing_values(df)
    print("Step 3: Validating ranges...")
    validate_ranges(df)
    print("Step 4: Dropping any remaining NaN...")
    rows_before = len(df)
    df = df.dropna()
    dropped = rows_before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows")
    print(f"\nCleaning complete. Final: {len(df)} observations")
    return df


def save_cleaned_data(df):
    """Save to data/processed/cleaned_data.csv"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    filepath = os.path.join(PROCESSED_DIR, PROCESSED_FILE)
    df.to_csv(filepath)
    print(f"Saved cleaned data to {filepath}")


if __name__ == "__main__":
    from data_ingestion.ingest import load_raw_data

    print("=" * 60)
    print("DATA CLEANING PIPELINE")
    print("=" * 60)
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    save_cleaned_data(clean_df)
    print("\nFirst 5 rows of cleaned data:")
    print(clean_df.head())
```

Step 5 above is your final complete file.

**Run it:**

```bash
python -m data_processing.clean
```

**Expected output:**

```
============================================================
DATA CLEANING PIPELINE
============================================================
Loaded 300 observations
Date range: 2000-01-01 to 2024-12-01

Missing values per column:
mpr              0
inflation        0
exchange_rate    0
m2               0
Step 1: Enforcing monthly frequency...
  No gaps found. All months present.
Step 2: Handling missing values...
  No missing values found.
Step 3: Validating ranges...
  mpr: all values within range
  inflation: all values within range
  exchange_rate: all values within range
  m2: all values within range
Step 4: Dropping any remaining NaN...

Cleaning complete. Final: 300 observations
Saved cleaned data to data_processing/../data/processed/cleaned_data.csv

First 5 rows of cleaned data:
             mpr  inflation  exchange_rate      m2
date
2000-01-01  13.5       6.62          92.34  1070.5
2000-02-01  13.5       6.93          92.55  1078.2
2000-03-01  13.5       7.87          92.69  1090.6
2000-04-01  13.5       8.13          93.05  1095.3
2000-05-01  13.5       8.68          95.10  1100.7
```

Now you have the complete cleaning pipeline. The `clean_data()` function runs all three steps in the correct order, then drops any rows that still have missing values after interpolation. The `save_cleaned_data()` function writes the result to `data/processed/cleaned_data.csv`, creating the folder if it does not exist.

**Why the order matters:**

1. **Frequency first** -- so interpolation knows WHERE the gaps are (they become NaN rows).
2. **Interpolation second** -- so range validation does not have to deal with NaN values.
3. **Range validation third** -- so the final dataset has only realistic values.
4. **Drop NaN last** -- after all attempts to fill gaps have been made.

---

## Verify: Check That the Cleaned Data Was Saved

```bash
ls -la data/processed/
```

You should see `cleaned_data.csv` in that folder. If the folder or file does not exist, something went wrong -- re-run `python -m data_processing.clean` and check for error messages.

---

## Commit

```bash
git add data_processing/clean.py
git commit -m "Day 4: Add data cleaning pipeline"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: Data file not found` | The cleaning script calls the ingestion script, which looks for `data/raw/nigeria_macro_data.csv`. Make sure that file exists. If you have not completed Day 3, go back and create the dataset first. |
| `ModuleNotFoundError: No module named 'data_ingestion'` | You are not running from the project root directory. Make sure you `cd` into `Nigerian_Inflation_Predictor/` before running the command. The `-m` flag requires you to be in the project root so Python can find the `data_ingestion` package. |
| `KeyError: 'date'` | The ingestion script did not set the date column as the index. Make sure `load_raw_data()` in `data_ingestion/ingest.py` calls `df.set_index("date")` before returning the DataFrame. |
| Output shows `WARNING: still has X missing values` | Your data has a gap of 4 or more consecutive months that interpolation cannot fill. Check your raw CSV for large missing sections. You may need to obtain the missing data from CBN/NBS sources, or accept that those rows will be dropped. |
| `PermissionError: [Errno 13] Permission denied` when saving | The script cannot write to the `data/processed/` folder. On Linux/Mac, try `chmod -R 755 data/`. On Windows, right-click the folder, go to Properties, and make sure it is not read-only. |
| `AttributeError: 'DataFrame' object has no attribute 'interpolate'` | Check your pandas version with `python -c "import pandas; print(pandas.__version__)"`. It should be 2.1.4 or later. If not, run `pip install pandas==2.1.4`. |

---

## Check Your Understanding

These are the kinds of questions an examiner will ask about your data cleaning process. Practice answering them before your defense.

### 1. "Why not just delete rows with missing values?"

**Answer:** "Deleting rows breaks the time series structure. If March 2015 is missing and we delete it, the gap between February and April is two months, not one. Every lag-based calculation -- autocorrelation, ARDL lag terms, VAR impulse responses -- assumes equal spacing between observations. Deletion violates that assumption. Interpolation preserves the monthly spacing by estimating a reasonable value for the missing month, using the known values on either side of the gap."

### 2. "What does interpolation do?"

**Answer:** "Linear interpolation draws a straight line between the last known value before a gap and the first known value after the gap, then reads off the estimated values along that line. For example, if January = 10.0 and March = 14.0 but February is missing, interpolation fills February with 12.0 -- the midpoint. We limit it to gaps of 3 months or fewer, because interpolating over longer gaps produces unreliable estimates."

### 3. "Why enforce monthly frequency first?"

**Answer:** "Because it makes hidden gaps visible. In the raw CSV, if March 2015 is missing, there is simply no row for that date -- the DataFrame jumps from February to April. Pandas does not know a month is missing. By reindexing to a complete monthly date range, we insert a NaN row for every missing month. Only then can the interpolation step detect those NaN values and fill them. If we tried to interpolate first, there would be nothing to interpolate -- the data would look complete but actually have temporal gaps."

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Cleaning script | `data_processing/clean.py` | Enforces monthly frequency, interpolates missing values, validates ranges, saves cleaned data |
| Cleaned dataset | `data/processed/cleaned_data.csv` | Gap-free, validated monthly time series ready for EDA and modelling |

**Packages installed today:** None (using pandas from Day 2)

**Tomorrow (Day 5):** You will build the Exploratory Data Analysis (EDA) script -- computing summary statistics, creating time series plots for each variable, and generating a correlation matrix heatmap.
