# Week 1, Day 2 — Python & Pandas Crash Course for This Project

## What You'll Learn Today

- How to create a virtual environment (isolated Python workspace)
- How to install your first package (`pandas`)
- What a DataFrame is and why it matters
- How to read a CSV file into Python
- How to inspect, filter, and manipulate data
- How to run a Python script from the terminal

## Why This Matters

Every piece of code in this project uses `pandas` — the Python library for working with tabular data (rows and columns). You cannot proceed without understanding DataFrames. Today is about building that foundation.

---

## Step 1: Create a Virtual Environment

A virtual environment is an isolated Python workspace. Packages you install here don't affect your system Python.

```bash
cd Nigerian_Inflation_Predictor
python -m venv venv
```

**Activate it:**

```bash
# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

**How you know it worked:** Your terminal prompt will show `(venv)` at the beginning:
```
(venv) $
```

**Important:** You must activate the virtual environment every time you open a new terminal to work on this project.

---

## Step 2: Install pandas

**Update `requirements.txt`:**
```
pandas==2.1.4
```

**Install:**
```bash
pip install pandas==2.1.4
```

**What you should see:** A bunch of download/install messages ending with:
```
Successfully installed pandas-2.1.4 ...
```

**Verify it works:**
```bash
python -c "import pandas; print(pandas.__version__)"
```

Should print: `2.1.4`

---

## Step 3: Create the Practice Script

We'll create a small practice script to learn pandas basics using Nigerian macro data. This is NOT part of the final project — it's a learning exercise.

**File: `data_processing/pandas_practice.py`**

```python
"""
Pandas crash course — learning the basics with Nigerian macro data.
Run this script to see how pandas works.
You can delete this file after you understand the concepts.
"""

import pandas as pd

# ============================================================
# PART 1: CREATING A DATAFRAME
# ============================================================
# A DataFrame is a table — rows and columns, like a spreadsheet.
# Each column has a name and all values in a column are the same type.

print("=" * 60)
print("PART 1: Creating a DataFrame")
print("=" * 60)

# Create a small DataFrame with Nigerian macro data
data = {
    "date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"],
    "mpr": [18.75, 22.75, 24.75, 24.75, 26.25],
    "inflation": [29.90, 31.70, 33.20, 33.69, 33.95],
    "exchange_rate": [895.0, 1510.0, 1550.0, 1400.0, 1480.0],
    "m2": [222990.3, 227925.5, 232985.8, 238175.4, 243500.6],
}

# pd.DataFrame() turns a dictionary into a table
df = pd.DataFrame(data)

# Print the DataFrame — this shows the table
print(df)
print()

# .shape tells you (rows, columns)
print(f"Shape: {df.shape}")        # Output: (5, 5)
print(f"Rows: {df.shape[0]}")      # Output: 5
print(f"Columns: {df.shape[1]}")   # Output: 5
print()


# ============================================================
# PART 2: EXAMINING DATA
# ============================================================
print("=" * 60)
print("PART 2: Examining Data")
print("=" * 60)

# .head(n) shows the first n rows (default is 5)
print("First 3 rows:")
print(df.head(3))
print()

# .tail(n) shows the last n rows
print("Last 2 rows:")
print(df.tail(2))
print()

# .dtypes shows the data type of each column
print("Data types:")
print(df.dtypes)
print()
# Note: 'date' shows as 'object' (text) — we'll fix this below

# .describe() gives summary statistics for numeric columns
print("Summary statistics:")
print(df.describe())
print()


# ============================================================
# PART 3: SELECTING COLUMNS
# ============================================================
print("=" * 60)
print("PART 3: Selecting Columns")
print("=" * 60)

# Select one column — returns a Series (a single column of data)
inflation = df["inflation"]
print("Inflation column:")
print(inflation)
print(f"Type: {type(inflation)}")  # pandas Series
print()

# Select multiple columns — returns a new DataFrame
subset = df[["mpr", "inflation"]]
print("MPR and Inflation columns:")
print(subset)
print()


# ============================================================
# PART 4: DATES AND INDEXING
# ============================================================
print("=" * 60)
print("PART 4: Dates and Indexing")
print("=" * 60)

# Convert the 'date' column from text to actual dates
# pd.to_datetime() understands date formats automatically
df["date"] = pd.to_datetime(df["date"])
print("After date conversion:")
print(df.dtypes)
print()

# Set date as the index (row labels)
# This is standard for time series data — each row is identified by its date
df = df.set_index("date")
print("With date as index:")
print(df)
print()

# Now you can select rows by date
print("Data for March 2024:")
print(df.loc["2024-03-01"])
print()

# Select a date range
print("Data from Feb to April 2024:")
print(df.loc["2024-02-01":"2024-04-01"])
print()


# ============================================================
# PART 5: BASIC CALCULATIONS
# ============================================================
print("=" * 60)
print("PART 5: Basic Calculations")
print("=" * 60)

# Mean, median, min, max of a column
print(f"Average inflation: {df['inflation'].mean():.2f}%")
print(f"Median inflation: {df['inflation'].median():.2f}%")
print(f"Min inflation: {df['inflation'].min():.2f}%")
print(f"Max inflation: {df['inflation'].max():.2f}%")
print(f"Std deviation: {df['inflation'].std():.2f}%")
print()

# .diff() computes the change from one row to the next
# This is important — in econometrics, "first differencing" means
# computing change from period to period
print("Monthly CHANGE in inflation:")
print(df["inflation"].diff())
print()
# Note: The first value is NaN (Not a Number) because there's
# no previous row to compute the change from. This is normal.


# ============================================================
# PART 6: MISSING VALUES (NaN)
# ============================================================
print("=" * 60)
print("PART 6: Missing Values")
print("=" * 60)

# NaN = "Not a Number" — pandas uses this for missing data
# Let's create some missing data to see how it works

df_with_gaps = df.copy()
df_with_gaps.loc["2024-03-01", "mpr"] = None  # Set one value to missing

print("Data with a missing value:")
print(df_with_gaps)
print()

# .isnull() checks for missing values
print("Missing values per column:")
print(df_with_gaps.isnull().sum())
print()

# .dropna() removes rows with any missing value
print("After dropping rows with NaN:")
print(df_with_gaps.dropna())
print()

# .fillna() fills missing values with a specific value
# .interpolate() fills by computing the average of neighbors
print("After interpolating:")
print(df_with_gaps.interpolate())
print()


# ============================================================
# PART 7: READING FROM CSV
# ============================================================
print("=" * 60)
print("PART 7: Reading from CSV")
print("=" * 60)

# This is how you'll actually load data in the project:
# df = pd.read_csv("data/raw/nigeria_macro_data.csv")
#
# If the CSV has a date column you want as the index:
# df = pd.read_csv("data/raw/nigeria_macro_data.csv",
#                   index_col="date", parse_dates=True)
#
# We'll create the actual CSV file tomorrow. For now, just know
# that read_csv() is how data gets into pandas.

print("CSV reading syntax:")
print('  df = pd.read_csv("data/raw/nigeria_macro_data.csv")')
print('  df = pd.read_csv("path", index_col="date", parse_dates=True)')
print()

# SAVING to CSV
# df.to_csv("data/processed/cleaned_data.csv")
print("CSV saving syntax:")
print('  df.to_csv("data/processed/cleaned_data.csv")')


# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 60)
print("SUMMARY — Key pandas operations for this project")
print("=" * 60)
print("""
  pd.read_csv(path)          — Load a CSV file
  df.head() / df.tail()      — See first/last rows
  df.shape                   — (rows, columns)
  df.dtypes                  — Data types of each column
  df.describe()              — Summary statistics
  df["column"]               — Select one column
  df[["col1", "col2"]]      — Select multiple columns
  df.set_index("date")       — Make a column the row index
  df.loc["2024-01-01"]       — Select row by index value
  df["col"].mean()           — Average
  df["col"].diff()           — Period-to-period change
  df.isnull().sum()          — Count missing values
  df.dropna()                — Remove rows with missing values
  df.interpolate()           — Fill gaps with interpolation
  df.to_csv(path)            — Save to CSV
""")
```

---

## Step 4: Run the Practice Script

```bash
python -m data_processing.pandas_practice
```

**What you should see:** Each section prints its output with clear headers. Read through every line of output and make sure you understand what happened.

---

## Step 5: Experiment

Open `data_processing/pandas_practice.py` in your editor and try these modifications:

1. **Add a new row** to the `data` dictionary (e.g., June 2024: mpr=26.25, inflation=34.19, exchange_rate=1505, m2=248965.3)
2. **Calculate the average MPR** using `df["mpr"].mean()`
3. **Try `df.corr()`** — this computes the correlation matrix (how strongly the variables move together)

---

## Step 6: Commit

```bash
git add data_processing/pandas_practice.py requirements.txt
git commit -m "Day 2: Add pandas practice script and install pandas"
```

---

## Verify Your Work

After running the script, you should be able to answer:

- What is a DataFrame? → A table with rows and columns
- How do you select a column? → `df["column_name"]`
- What does `.diff()` do? → Computes the change from one row to the next
- What is NaN? → A missing value
- How do you read a CSV? → `pd.read_csv("path/to/file.csv")`

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'pandas'` | You didn't activate the venv. Run `source venv/bin/activate` first |
| `pip install` fails with permission error | Make sure you're in the venv. Never use `sudo pip install` |
| Numbers look wrong (too many decimals) | Use f-strings: `f"{value:.2f}"` to format to 2 decimal places |
| `KeyError: 'inflation'` | Check your column name — Python is case-sensitive |

---

## Check Your Understanding

1. **"What is a pandas DataFrame?"**
   > A two-dimensional labeled data structure — like a spreadsheet or SQL table. Each column has a name and data type. Each row has an index (usually a date for time series).

2. **"Why do we set the date as the index?"**
   > For time series analysis, the date is the primary identifier for each observation. Setting it as the index allows us to select data by date, align multiple series, and ensure proper temporal ordering.

3. **"What does first differencing mean?"**
   > Subtracting the previous period's value from the current period's value. If inflation was 31.7% in February and 33.2% in March, the first difference is +1.5 percentage points. This tells us the rate of *change* rather than the level.

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Practice script | `data_processing/pandas_practice.py` | Learn pandas fundamentals |
| Requirements | `requirements.txt` | Now lists `pandas==2.1.4` |

**Packages installed today:** `pandas==2.1.4`

**Tomorrow (Day 3):** You'll create the actual dataset and build the data ingestion script that loads it into pandas.
