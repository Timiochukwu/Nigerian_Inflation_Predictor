# Week 1, Day 4 -- Data Cleaning: Building It Function by Function

## What You Will Learn Today

- Why cleaning matters and what goes wrong with Nigerian economic data
- How to enforce monthly frequency so your time series has no hidden gaps
- How to handle missing values with interpolation instead of deletion
- How to validate that values fall within realistic ranges for Nigeria
- How to save cleaned data so every downstream script reads the same file

## Why This Matters

Econometric models need **complete time series** -- one value per month, no gaps, no impossible numbers. If March 2015 is missing from your data, what the model thinks is "one month before April 2015" is actually February 2015 (two months before). Every lag coefficient it estimates is silently wrong. Your ARDL bounds test, your VAR impulse responses, your Granger causality results -- all of them depend on the assumption that consecutive rows are exactly one month apart.

This is not a theoretical problem. Real-world Nigerian data from the CBN Statistical Bulletin frequently has:

- **Missing months** where data was not published or was delayed
- **Impossible values** from data entry errors (a decimal in the wrong place turns 15.0% inflation into 150.0%)
- **Gaps in reporting** where an entire variable is absent for several months

Your raw CBN data file (`data/raw/cbn_infl_data.csv`) has 34 columns. On Day 3, the ingestion module selected the 4 core variables you need: `mpr`, `infl`, `exo`, and `tbr`. Today you build the wall between that raw extraction and the clean, reliable dataset your models require.

We build the cleaning script **one function at a time**. You will write a small piece, run it, see output, understand what it does, then add the next piece. By the end, you will have a complete pipeline -- but you will understand every line because you built it yourself.

---

## Packages Installed Today: None

pandas (installed on Day 2) provides everything you need: reindexing, interpolation, range checking, and CSV export. numpy (also from Day 2) is used for NaN handling. No new packages required.

---

## Part 1: Building `data_processing/clean.py` -- Step by Step

If you followed Day 3, the `data_processing/` folder already exists and contains an `__init__.py` file. If it does not, create the folder and add an empty `__init__.py` inside it.

At each step, we show you the **complete file**. Delete everything in `data_processing/clean.py` and replace it with exactly what is shown. No guessing where to put things. No snippets. The entire file, every time.

---

### Step 1: Imports, Constants, and VALID_RANGES Only

Delete everything in `data_processing/clean.py` and replace it with this:

```python
import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PROCESSED_FILE = "cleaned_data.csv"

VALID_RANGES = {
    "mpr": (1.0, 35.0),
    "infl": (-5.0, 50.0),
    "exo": (50.0, 2500.0),
    "tbr": (0.1, 40.0),
}

if __name__ == "__main__":
    print("VALID_RANGES:")
    for col, (lo, hi) in VALID_RANGES.items():
        print(f"  {col}: {lo} to {hi}")
```

**Run it:**

```bash
python -m data_processing.clean
```

**Expected output:**

```
VALID_RANGES:
  mpr: 1.0 to 35.0
  infl: -5.0 to 50.0
  exo: 50.0 to 2500.0
  tbr: 0.1 to 40.0
```

**What this code does -- line by line:**

`import os` -- gives you functions for working with file paths and directories. You need it to build the path to the output folder.

`import pandas as pd` -- the data manipulation library you installed on Day 2. `pd` is the standard short name that every tutorial and Stack Overflow answer uses.

`import numpy as np` -- the numerical computing library. pandas is built on top of numpy. You need it here because `np.nan` (Not a Number) is how Python represents missing values.

`PROCESSED_DIR` -- builds a file path relative to the script's location. `os.path.dirname(__file__)` gives the folder this script lives in (`data_processing/`), then `".."` goes up one level to the project root, then into `data/processed/`. This is where cleaned output will be saved.

`PROCESSED_FILE` -- the filename for the cleaned CSV. Every downstream script (EDA, stationarity tests, ARDL, VAR) will read from this exact file.

`VALID_RANGES` -- a dictionary that defines what values are "realistic" for each Nigerian macro variable. Each entry is a tuple of (minimum, maximum). Here is why each range is set the way it is:

- **mpr (1.0 to 35.0):** The CBN Monetary Policy Rate. Historically, the CBN has never set the MPR below 6% (the 2009 low during the global financial crisis) or above about 27.50% (the 2024 tightening cycle). We use a wider range of 1.0 to 35.0 to allow for future policy changes without falsely flagging valid data. If you see a value of 0.5 or 99.0, that is almost certainly a data entry error.

- **infl (-5.0 to 50.0):** Headline CPI inflation. Nigeria has rarely experienced deflation (negative inflation), but we allow a small buffer below zero. The highest inflation in the modern CBN data is around 34% (late 2024). We allow up to 50% for safety. If you see a value of 150.0, someone put the decimal in the wrong place.

- **exo (50.0 to 2500.0):** The official Naira/USD exchange rate. The Naira was around 90/$ in 2000 and has depreciated to roughly 1500+/$ by late 2024. We use 50 as the floor (well below any historical value) and 2500 as the ceiling (allowing room for further depreciation). A value of 15000 would suggest the data was recorded in kobo instead of Naira.

- **tbr (0.1 to 40.0):** The 91-day Treasury Bill Rate. T-bill rates in Nigeria have ranged from sub-1% (during periods of excess liquidity) to over 25% (during aggressive monetary tightening). We use 0.1 to 40.0 to capture this full range with a safety margin.

The `if __name__ == "__main__"` block is a quick test -- it confirms the file loads without syntax errors and prints the ranges so you can verify them.

---

### Step 2: Add the enforce_monthly_frequency Function

Delete everything in `data_processing/clean.py` and replace it with this:

```python
import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PROCESSED_FILE = "cleaned_data.csv"

VALID_RANGES = {
    "mpr": (1.0, 35.0),
    "infl": (-5.0, 50.0),
    "exo": (50.0, 2500.0),
    "tbr": (0.1, 40.0),
}


def enforce_monthly_frequency(df):
    """Make sure there is exactly one row per month, no gaps."""
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="MS",
    )
    df_reindexed = df.reindex(full_range)
    df_reindexed.index.name = "date"

    added = len(df_reindexed) - len(df)
    if added > 0:
        print(f"  Inserted {added} missing month(s) as NaN rows")
    else:
        print(f"  No gaps found. All months present.")
    print(f"  Rows after: {len(df_reindexed)}")
    return df_reindexed


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data_ingestion.ingest import load_raw_data, select_core_variables

    raw = load_raw_data()
    core = select_core_variables(raw)

    print("\nEnforcing monthly frequency...")
    result = enforce_monthly_frequency(core)
    print(f"\nFirst 5 rows:")
    print(result.head())
```

**Run it:**

```bash
python -m data_processing.clean
```

**Expected output (the exact numbers depend on your data):**

```
Loaded ... observations
...

Enforcing monthly frequency...
  No gaps found. All months present.
  Rows after: ...

First 5 rows:
             mpr   infl     exo    tbr
date
2000-01-01  ...   ...      ...    ...
...
```

If your data has no missing months, you will see "No gaps found." If it does have gaps, you will see how many were inserted.

**What this function does:**

`pd.date_range(start, end, freq="MS")` creates a complete list of dates from the first to the last month in your data. `freq="MS"` means "Month Start" -- it generates the 1st of every month. So if your data runs from January 2000 to December 2024, this creates 300 dates, one for every month.

`df.reindex(full_range)` reshapes the DataFrame to match that complete date list. If a month exists in `full_range` but not in your data, that row is created and filled with NaN (missing). If a month exists in your data but not in `full_range`, it is kept as-is.

**Why does this matter?** In the raw CSV, if March 2015 is missing, there is simply no row for that date. The DataFrame jumps from February to April, and pandas does not know a month is missing. By reindexing, we make hidden gaps visible -- they become NaN rows that the interpolation step can detect and fill.

The `sys.path.insert(0, ...)` line in the `__main__` block tells Python to look in the project root directory when importing modules. This is necessary because `data_processing/clean.py` needs to import from `data_ingestion/ingest.py`, which lives in a sibling folder.

---

### Step 3: Add the handle_missing_values Function

Delete everything in `data_processing/clean.py` and replace it with this:

```python
import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PROCESSED_FILE = "cleaned_data.csv"

VALID_RANGES = {
    "mpr": (1.0, 35.0),
    "infl": (-5.0, 50.0),
    "exo": (50.0, 2500.0),
    "tbr": (0.1, 40.0),
}


def enforce_monthly_frequency(df):
    """Make sure there is exactly one row per month, no gaps."""
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="MS",
    )
    df_reindexed = df.reindex(full_range)
    df_reindexed.index.name = "date"

    added = len(df_reindexed) - len(df)
    if added > 0:
        print(f"  Inserted {added} missing month(s) as NaN rows")
    else:
        print(f"  No gaps found. All months present.")
    print(f"  Rows after: {len(df_reindexed)}")
    return df_reindexed


def handle_missing_values(df):
    """Fill short gaps (up to 3 consecutive months) using linear interpolation."""
    missing_before = df.isnull().sum()

    df_filled = df.interpolate(method="linear", limit=3)

    missing_after = df_filled.isnull().sum()

    for col in df_filled.columns:
        filled = missing_before[col] - missing_after[col]
        if filled > 0:
            print(f"  {col}: filled {filled} missing value(s) via interpolation")
        remaining = missing_after[col]
        if remaining > 0:
            print(f"  WARNING: {col} still has {remaining} missing value(s) (gap > 3 months)")

    if missing_before.sum() == 0:
        print("  No missing values found. All columns complete.")

    return df_filled


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data_ingestion.ingest import load_raw_data, select_core_variables

    raw = load_raw_data()
    core = select_core_variables(raw)

    print("\nStep 1: Enforcing monthly frequency...")
    core = enforce_monthly_frequency(core)

    print("\nStep 2: Handling missing values...")
    core = handle_missing_values(core)
```

**Run it:**

```bash
python -m data_processing.clean
```

**Expected new output at the bottom:**

```
Step 2: Handling missing values...
  No missing values found. All columns complete.
```

Or, if your real CBN data has gaps:

```
Step 2: Handling missing values...
  infl: filled 2 missing value(s) via interpolation
  tbr: filled 1 missing value(s) via interpolation
```

**What interpolation means in plain English:**

Interpolation fills in missing values by drawing a straight line between the known values on either side of the gap.

Example: if January's inflation is 10.0% and March's inflation is 14.0%, but February is missing, linear interpolation fills February with 12.0% -- the midpoint.

For a longer gap: if January = 10.0, February = NaN, March = NaN, April = 16.0, then interpolation fills February with 12.0 (one-third of the way from 10 to 16) and March with 14.0 (two-thirds of the way).

The `limit=3` parameter means: only fill gaps of 3 or fewer consecutive missing months. If 4 or more months in a row are missing, the gap is too large to trust a simple straight-line estimate, and those values stay as NaN. They will be dropped in the final step. This is a judgment call -- for monthly macro data, 1-3 missing months can usually be estimated reasonably. More than that, and you risk inventing data rather than estimating it.

---

### Step 4: Add the validate_ranges Function

Delete everything in `data_processing/clean.py` and replace it with this:

```python
import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PROCESSED_FILE = "cleaned_data.csv"

VALID_RANGES = {
    "mpr": (1.0, 35.0),
    "infl": (-5.0, 50.0),
    "exo": (50.0, 2500.0),
    "tbr": (0.1, 40.0),
}


def enforce_monthly_frequency(df):
    """Make sure there is exactly one row per month, no gaps."""
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="MS",
    )
    df_reindexed = df.reindex(full_range)
    df_reindexed.index.name = "date"

    added = len(df_reindexed) - len(df)
    if added > 0:
        print(f"  Inserted {added} missing month(s) as NaN rows")
    else:
        print(f"  No gaps found. All months present.")
    print(f"  Rows after: {len(df_reindexed)}")
    return df_reindexed


def handle_missing_values(df):
    """Fill short gaps (up to 3 consecutive months) using linear interpolation."""
    missing_before = df.isnull().sum()

    df_filled = df.interpolate(method="linear", limit=3)

    missing_after = df_filled.isnull().sum()

    for col in df_filled.columns:
        filled = missing_before[col] - missing_after[col]
        if filled > 0:
            print(f"  {col}: filled {filled} missing value(s) via interpolation")
        remaining = missing_after[col]
        if remaining > 0:
            print(f"  WARNING: {col} still has {remaining} missing value(s) (gap > 3 months)")

    if missing_before.sum() == 0:
        print("  No missing values found. All columns complete.")

    return df_filled


def validate_ranges(df):
    """Check each column against VALID_RANGES. Replace out-of-range values with NaN."""
    total_violations = 0

    for col, (low, high) in VALID_RANGES.items():
        if col not in df.columns:
            continue

        out_of_range = (df[col] < low) | (df[col] > high)
        n_violations = out_of_range.sum()

        if n_violations > 0:
            print(f"  WARNING: {col} has {n_violations} value(s) outside [{low}, {high}]")
            print(f"    Actual min: {df[col].min():.2f}, Actual max: {df[col].max():.2f}")
            df.loc[out_of_range, col] = np.nan
            print(f"    Replaced {n_violations} out-of-range value(s) with NaN")
            total_violations += n_violations
        else:
            print(f"  {col}: all values within [{low}, {high}]")

    if total_violations == 0:
        print("  All variables passed range validation.")

    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data_ingestion.ingest import load_raw_data, select_core_variables

    raw = load_raw_data()
    core = select_core_variables(raw)

    print("\nStep 1: Enforcing monthly frequency...")
    core = enforce_monthly_frequency(core)

    print("\nStep 2: Handling missing values...")
    core = handle_missing_values(core)

    print("\nStep 3: Validating ranges...")
    core = validate_ranges(core)
```

**Run it:**

```bash
python -m data_processing.clean
```

**Expected new output at the bottom:**

```
Step 3: Validating ranges...
  mpr: all values within [1.0, 35.0]
  infl: all values within [-5.0, 50.0]
  exo: all values within [50.0, 2500.0]
  tbr: all values within [0.1, 40.0]
  All variables passed range validation.
```

If your data has values outside the valid ranges, you will see warnings like:

```
Step 3: Validating ranges...
  WARNING: exo has 2 value(s) outside [50.0, 2500.0]
    Actual min: 45.30, Actual max: 1680.00
    Replaced 2 out-of-range value(s) with NaN
```

**Why we check ranges and replace violations with NaN:**

Data entry errors are common in Nigerian economic datasets. A decimal point in the wrong place turns an inflation rate of 15.0% into 150.0%. An exchange rate recorded in kobo instead of Naira would be 100x too large. A T-bill rate of -5.0% is economically impossible.

These errors will not cause your code to crash -- they will silently corrupt your model's results. An inflation spike of 150% in one month will dominate every regression coefficient, every impulse response, and every forecast. Range validation catches these problems before they propagate.

Notice that `validate_ranges` does not just report problems -- it **replaces** out-of-range values with `np.nan`. This is important because the orchestrator function (Step 5) will run a final `dropna()` to remove any rows that still have NaN after all cleaning attempts. So the sequence is: detect the bad value, mark it as missing, then let the final cleanup handle it.

---

### Step 5: Add the clean_data Orchestrator, save_cleaned_data, and the Final __main__ Block

Delete everything in `data_processing/clean.py` and replace it with this:

```python
import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PROCESSED_FILE = "cleaned_data.csv"

VALID_RANGES = {
    "mpr": (1.0, 35.0),
    "infl": (-5.0, 50.0),
    "exo": (50.0, 2500.0),
    "tbr": (0.1, 40.0),
}


def enforce_monthly_frequency(df):
    """Make sure there is exactly one row per month, no gaps."""
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="MS",
    )
    df_reindexed = df.reindex(full_range)
    df_reindexed.index.name = "date"

    added = len(df_reindexed) - len(df)
    if added > 0:
        print(f"  Inserted {added} missing month(s) as NaN rows")
    else:
        print(f"  No gaps found. All months present.")
    print(f"  Rows after: {len(df_reindexed)}")
    return df_reindexed


def handle_missing_values(df):
    """Fill short gaps (up to 3 consecutive months) using linear interpolation."""
    missing_before = df.isnull().sum()

    df_filled = df.interpolate(method="linear", limit=3)

    missing_after = df_filled.isnull().sum()

    for col in df_filled.columns:
        filled = missing_before[col] - missing_after[col]
        if filled > 0:
            print(f"  {col}: filled {filled} missing value(s) via interpolation")
        remaining = missing_after[col]
        if remaining > 0:
            print(f"  WARNING: {col} still has {remaining} missing value(s) (gap > 3 months)")

    if missing_before.sum() == 0:
        print("  No missing values found. All columns complete.")

    return df_filled


def validate_ranges(df):
    """Check each column against VALID_RANGES. Replace out-of-range values with NaN."""
    total_violations = 0

    for col, (low, high) in VALID_RANGES.items():
        if col not in df.columns:
            continue

        out_of_range = (df[col] < low) | (df[col] > high)
        n_violations = out_of_range.sum()

        if n_violations > 0:
            print(f"  WARNING: {col} has {n_violations} value(s) outside [{low}, {high}]")
            print(f"    Actual min: {df[col].min():.2f}, Actual max: {df[col].max():.2f}")
            df.loc[out_of_range, col] = np.nan
            print(f"    Replaced {n_violations} out-of-range value(s) with NaN")
            total_violations += n_violations
        else:
            print(f"  {col}: all values within [{low}, {high}]")

    if total_violations == 0:
        print("  All variables passed range validation.")

    return df


def clean_data(df):
    """Run the full cleaning pipeline: frequency -> interpolation -> validation -> drop NaN."""
    print("Step 1: Enforcing monthly frequency...")
    df = enforce_monthly_frequency(df)

    print("\nStep 2: Handling missing values...")
    df = handle_missing_values(df)

    print("\nStep 3: Validating ranges...")
    df = validate_ranges(df)

    print("\nStep 4: Dropping any remaining NaN rows...")
    rows_before = len(df)
    df = df.dropna()
    dropped = rows_before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} row(s) that still had missing values")
    else:
        print(f"  No rows dropped. Dataset is complete.")

    print(f"\nCleaning complete. Final dataset: {len(df)} observations")
    return df


def save_cleaned_data(df):
    """Save the cleaned DataFrame to data/processed/cleaned_data.csv."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    filepath = os.path.join(PROCESSED_DIR, PROCESSED_FILE)
    df.to_csv(filepath)
    print(f"Saved cleaned data to: {filepath}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data_ingestion.ingest import load_raw_data, select_core_variables

    raw = load_raw_data()
    core = select_core_variables(raw)
    cleaned = clean_data(core)
    save_cleaned_data(cleaned)
    print("\nCleaned data summary:")
    print(cleaned.describe())
```

Step 5 above is your final complete file.

**Run it:**

```bash
python -m data_processing.clean
```

**Expected output (the exact numbers depend on your CBN data):**

```
Loaded ... observations
...

Step 1: Enforcing monthly frequency...
  No gaps found. All months present.
  Rows after: ...

Step 2: Handling missing values...
  No missing values found. All columns complete.

Step 3: Validating ranges...
  mpr: all values within [1.0, 35.0]
  infl: all values within [-5.0, 50.0]
  exo: all values within [50.0, 2500.0]
  tbr: all values within [0.1, 40.0]
  All variables passed range validation.

Step 4: Dropping any remaining NaN rows...
  No rows dropped. Dataset is complete.

Cleaning complete. Final dataset: ... observations
Saved cleaned data to: data_processing/../data/processed/cleaned_data.csv

Cleaned data summary:
              mpr        infl  ...         tbr
count  ...        ...         ...         ...
mean   ...        ...         ...         ...
std    ...        ...         ...         ...
min    ...        ...         ...         ...
25%    ...        ...         ...         ...
50%    ...        ...         ...         ...
75%    ...        ...         ...         ...
max    ...        ...         ...         ...
```

The `cleaned.describe()` output gives you the count, mean, standard deviation, minimum, quartiles, and maximum for each of your 4 core variables. This is a quick sanity check -- you should be able to glance at these numbers and confirm they make sense for Nigerian data. For example, if the mean MPR is around 12-14% and the mean inflation is around 12-15%, that is consistent with what you know about Nigeria's macroeconomic history.

**Why the order of operations matters:**

1. **Frequency first** -- so interpolation knows WHERE the gaps are. Without this step, missing months are invisible (there is simply no row for that date). After reindexing, missing months become NaN rows that the next step can detect.

2. **Interpolation second** -- so we fill small gaps before checking ranges. If we validated ranges first and a gap existed between two valid values, the NaN would be flagged as a range violation, which is wrong -- it is a gap, not a bad value.

3. **Range validation third** -- so we catch impossible values after all legitimate gaps have been filled. Out-of-range values are replaced with NaN.

4. **Drop NaN last** -- after all attempts to fill and fix have been made. Any row that still has a NaN at this point either had a gap too large to interpolate (more than 3 months) or had an out-of-range value that was replaced. These rows are removed.

---

## Part 2: Verify the Output

Check that the cleaned data file was created:

```bash
ls -la data/processed/
```

You should see `cleaned_data.csv` in that folder. If the folder or file does not exist, something went wrong -- re-read the error messages from the script and fix them before continuing.

---

## Part 3: Commit

```bash
git add data_processing/clean.py
git commit -m "Day 4: Add data cleaning pipeline with frequency enforcement, interpolation, and range validation"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'data_ingestion'` | You are not running from the project root directory. Make sure you `cd` into `Nigerian_Inflation_Predictor/` before running the command. The `-m` flag requires you to be in the project root so Python can find the `data_ingestion` package. |
| `FileNotFoundError: Data file not found` | The cleaning script calls the ingestion script, which looks for `data/raw/cbn_infl_data.csv`. Make sure that file exists. If you have not completed Day 3, go back and do it first. |
| `KeyError: 'mpr'` or `KeyError: 'infl'` | The `select_core_variables()` function from Day 3 did not find the expected columns in your CSV. Open your raw CSV and verify the column names match what the ingestion script expects. Column names are case-sensitive. |
| `AttributeError: 'DataFrame' object has no attribute 'interpolate'` | Check your pandas version with `python -c "import pandas; print(pandas.__version__)"`. You need pandas 2.0 or later. If your version is older, run `pip install --upgrade pandas`. |
| Output shows `WARNING: still has X missing value(s) (gap > 3 months)` | Your data has a gap of 4 or more consecutive months that interpolation cannot safely fill. Check your raw CSV for large missing sections. You may need to obtain the missing data from CBN sources, or accept that those rows will be dropped in Step 4. |
| `PermissionError` when saving to `data/processed/` | The script cannot write to that folder. On Linux/Mac, try `chmod -R 755 data/`. On Windows, right-click the folder, go to Properties, and uncheck read-only. |

---

## Check Your Understanding

These are the kinds of questions an examiner will ask about your data cleaning process. Practice answering them before your defense.

### 1. "Why not just delete rows with missing values instead of interpolating?"

**Answer:** Deleting rows breaks the time series structure. If March 2015 is missing and we delete it, the gap between February and April is two months, not one. Every lag-based calculation -- autocorrelation, ARDL lag terms, VAR impulse responses -- assumes equal spacing between observations. Deletion violates that assumption. Interpolation preserves the monthly spacing by estimating a reasonable value for the missing month, using the known values on either side of the gap. We limit interpolation to gaps of 3 months or fewer because longer gaps produce unreliable estimates.

### 2. "Why do you enforce monthly frequency before interpolating?"

**Answer:** Because reindexing makes hidden gaps visible. In the raw CSV, if March 2015 is missing, there is simply no row for that date -- the DataFrame jumps from February to April. pandas does not know a month is missing. By reindexing to a complete monthly date range, we insert a NaN row for March 2015. Only then can the interpolation step detect that NaN and fill it. If we tried to interpolate first, there would be nothing to interpolate -- the data would look complete but actually have a two-month temporal gap between February and April.

### 3. "How do your VALID_RANGES relate to Nigerian economic history?"

**Answer:** Each range is set based on the historical bounds of that variable with a safety margin. The MPR range of 1% to 35% covers the full CBN policy rate history from the 6% low in 2009 to the 27.50% high in late 2024, with room for future changes. The inflation range of -5% to 50% covers the 3% low during the 2006-2007 oil boom to the 34%+ highs in late 2024. The exchange rate range of 50 to 2500 covers the Naira's journey from around 90/$ in 2000 to 1500+/$ in 2024. The T-bill rate range of 0.1% to 40% captures the full range of Nigerian money market conditions. Any value outside these bounds is almost certainly a data entry error rather than a genuine economic observation, so we replace it with NaN and let the final cleanup decide whether to drop or interpolate.

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Cleaning script | `data_processing/clean.py` | Enforces monthly frequency, interpolates gaps, validates ranges, saves cleaned data |
| Cleaned dataset | `data/processed/cleaned_data.csv` | Gap-free, validated monthly time series ready for EDA and modelling |

**Functions you built:**

| Function | What it does |
|----------|-------------|
| `enforce_monthly_frequency(df)` | Creates a complete monthly date range and reindexes the DataFrame so hidden gaps become visible NaN rows |
| `handle_missing_values(df)` | Fills gaps of 1-3 consecutive months using linear interpolation; reports what was filled and what remains |
| `validate_ranges(df)` | Checks each variable against known Nigerian data bounds; replaces impossible values with NaN |
| `clean_data(df)` | Orchestrator that calls the three functions in the correct order, then drops remaining NaN rows |
| `save_cleaned_data(df)` | Writes the cleaned DataFrame to `data/processed/cleaned_data.csv` |

**Packages installed today:** None (using pandas and numpy from Day 2)

**Tomorrow (Day 5):** You will build the Exploratory Data Analysis (EDA) script -- computing summary statistics for each variable, creating time series plots, and generating a correlation matrix heatmap. This is where you start to see and understand the story in your data.
