# Week 1, Day 2 -- Python & Pandas: Learning by Doing

## What You Will Learn Today

- How to create a virtual environment (an isolated Python workspace)
- How to install your first package (`pandas`)
- What a DataFrame is and why it matters for this project
- How to select columns, convert dates, compute changes, and handle missing data
- How to run a Python script from the terminal

## Why This Matters

Every single file in this project uses `pandas`. It is the Python library for working with tabular data -- rows and columns, like a spreadsheet. You cannot move forward without understanding DataFrames. Today you will build that foundation one small piece at a time using the same variable names you will see in the real CBN dataset.

The four core variables in this project are:

| Variable | Meaning |
|----------|---------|
| `mpr` | Monetary Policy Rate (%) -- the interest rate set by the CBN |
| `infl` | Headline CPI inflation (%, year-on-year) |
| `exo` | Official exchange rate (Naira per US Dollar, NGN/$) |
| `tbr` | Treasury Bill Rate (%) |

You will work with all four today using made-up numbers. Tomorrow you will load the real data from `data/raw/cbn_infl_data.csv`, which has 34 columns of actual CBN data.

---

## Part 1: Setup (Virtual Environment + Install pandas)

### 1A: Create the Virtual Environment

A virtual environment is an isolated Python workspace. Packages you install inside it do not affect your system Python. Every time you work on this project, you will activate it first.

Open your terminal, make sure you are inside the project root folder, and run:

```bash
cd Nigerian_Inflation_Predictor
python -m venv venv
```

This creates a new folder called `venv/` inside your project. That folder contains a private copy of Python and a place to install packages. You never need to open or edit anything inside `venv/` -- just pretend it is a black box.

### 1B: Activate It

You must run one of these commands every time you open a new terminal window to work on this project.

**Mac / Linux:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**How you know it worked:** Your terminal prompt now starts with `(venv)`:
```
(venv) $
```

If you do not see `(venv)` at the start of your prompt, the virtual environment is not active. Nothing you install will be available to your scripts. Go back and run the activate command again.

### 1C: Update requirements.txt

Delete everything in `requirements.txt` and replace it with this:

```
pandas==2.1.4
```

That is the entire file -- one line, no comments, nothing else. Save the file.

`requirements.txt` is a standard Python convention. It lists every package your project depends on, pinned to an exact version number. Anyone who clones your project can install the same packages by running `pip install -r requirements.txt`. Pinning the version (`==2.1.4`) means everyone gets the exact same version, which prevents "it works on my machine" problems.

### 1D: Install pandas

```bash
pip install -r requirements.txt
```

You will see a wall of download and install messages. Pip is downloading pandas and all of its dependencies (numpy, pytz, python-dateutil, etc.). The last line should say something like:

```
Successfully installed numpy-1.26.2 pandas-2.1.4 python-dateutil-2.8.2 pytz-2023.3.post1 six-1.16.0 tzdata-2023.3
```

The exact dependency versions may differ slightly. That is fine. What matters is that `pandas-2.1.4` appears in the list.

### 1E: Verify It Works

Run this one-liner in your terminal:

```bash
python -c "import pandas; print(pandas.__version__)"
```

**What you should see:**
```
2.1.4
```

If you see that version number, pandas is installed and working. If you get `ModuleNotFoundError: No module named 'pandas'`, your virtual environment is not activated. Go back to step 1B.

---

## Part 2: Building the Practice Script (6 Steps)

We are going to create a practice file at `guide/week1/day2_practice.py`. This is NOT part of the main project code. It is purely for learning. You will write a few lines, run the file, see the output, understand what happened, and then replace the file with a slightly longer version.

Every step below shows you the **complete file from first line to last line**. The instruction is always: "Delete everything in the file and replace it with this." Do not try to add lines to the middle of the previous version. Just replace the whole file each time.

---

### Step 1: Create the File and Print a DataFrame

Create a new file at this path: `guide/week1/day2_practice.py`

Delete everything in `guide/week1/day2_practice.py` and replace it with this:

```python
import pandas as pd

data = {
    "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
    "mpr": [18.75, 22.75, 24.75],
    "infl": [29.90, 31.70, 33.20],
}

df = pd.DataFrame(data)
print(df)
print(f"\nShape: {df.shape}")
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see:**

```
         date    mpr   infl
0  2024-01-01  18.75  29.90
1  2024-02-01  22.75  31.70
2  2024-03-01  24.75  33.20

Shape: (3, 3)
```

**What just happened:**

- `import pandas as pd` loads the pandas library and gives it the short name `pd`. Everyone in the Python world uses this convention. You will see `pd.something` in every file of this project.
- `data = { ... }` is a Python **dictionary**. A dictionary stores key-value pairs. Each key (`"date"`, `"mpr"`, `"infl"`) becomes a column name. Each value is a list of entries for that column. The keys are strings (text in quotes) and the values are lists (items inside square brackets).
- `pd.DataFrame(data)` turns that dictionary into a **DataFrame** -- a table with rows and columns, like a spreadsheet. This is the single most important object in pandas. Almost everything you do in this project starts with a DataFrame.
- `df` is just a variable name. You could call it anything, but `df` is the universal convention for "the DataFrame I am working with right now."
- `df.shape` returns a tuple `(rows, columns)`. Here it is `(3, 3)` because we have 3 rows and 3 columns. The numbers on the left side of the output (0, 1, 2) are the **index** -- row labels that pandas assigns automatically starting from 0.

---

### Step 2: Select a Column and Compute an Average

Delete everything in `guide/week1/day2_practice.py` and replace it with this:

```python
import pandas as pd

data = {
    "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
    "mpr": [18.75, 22.75, 24.75],
    "infl": [29.90, 31.70, 33.20],
}

df = pd.DataFrame(data)
print(df)
print(f"\nShape: {df.shape}")

print("\n--- Column selection and average ---")
infl_column = df["infl"]
print(infl_column)
print(f"\nAverage inflation: {df['infl'].mean():.2f}%")
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see** (new output at the bottom):

```
--- Column selection and average ---
0    29.90
1    31.70
2    33.20
Name: infl, dtype: float64

Average inflation: 31.60%
```

**What just happened:**

- `df["infl"]` selects a single column from the DataFrame. You pass the column name as a string inside square brackets. The result is called a **Series** -- think of it as one column pulled out of the table. A Series is like a list of numbers with an index (the 0, 1, 2 on the left).
- `.mean()` computes the average of all values in that Series: (29.90 + 31.70 + 33.20) / 3 = 31.60. Pandas has dozens of built-in methods like this: `.sum()`, `.min()`, `.max()`, `.std()`, and more. You do not need to write loops.
- `f"Average inflation: {df['infl'].mean():.2f}%"` is an **f-string**. The `f` before the opening quote tells Python to evaluate anything inside curly braces `{}`. The `:.2f` is a format specifier that means "display this number with exactly 2 decimal places." Notice we used single quotes `'infl'` inside the curly braces because the f-string itself uses double quotes. Python requires you to alternate quote types when nesting.

---

### Step 3: Convert Dates and Set the Index

Delete everything in `guide/week1/day2_practice.py` and replace it with this:

```python
import pandas as pd

data = {
    "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
    "mpr": [18.75, 22.75, 24.75],
    "infl": [29.90, 31.70, 33.20],
}

df = pd.DataFrame(data)
print(df)
print(f"\nShape: {df.shape}")

print("\n--- Column selection and average ---")
infl_column = df["infl"]
print(infl_column)
print(f"\nAverage inflation: {df['infl'].mean():.2f}%")

print("\n--- Converting dates and setting index ---")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")
print(df)
print(f"\nIndex type: {type(df.index)}")
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see** (new output at the bottom):

```
--- Converting dates and setting index ---
              mpr   infl
date
2024-01-01  18.75  29.90
2024-02-01  22.75  31.70
2024-03-01  24.75  33.20

Index type: <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
```

**What just happened:**

- `pd.to_datetime(df["date"])` converts the "date" column from plain text strings (like `"2024-01-01"`) into actual datetime objects that Python understands as real dates. Before this call, pandas treated those values as ordinary text -- it did not know that `"2024-02-01"` comes after `"2024-01-01"`. After this call, pandas knows they are dates and can do date arithmetic, sorting, and slicing by date range.
- `df.set_index("date")` moves the date column from being a regular column to being the **row label** (the index). Notice how "date" moved from a column header to the left side of the table. The DataFrame now has only 2 columns (`mpr`, `infl`) instead of 3, because "date" is the index, not a column.
- The index type is `DatetimeIndex`, which confirms pandas is treating our row labels as real dates. This is called a **DatetimeIndex** and it is the standard setup for time series data. Every row is identified by its date. This matters because later, when you load the real CBN data from `data/raw/cbn_infl_data.csv`, you will set the date column as the index in exactly the same way.

---

### Step 4: First Differencing

Delete everything in `guide/week1/day2_practice.py` and replace it with this:

```python
import pandas as pd

data = {
    "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
    "mpr": [18.75, 22.75, 24.75],
    "infl": [29.90, 31.70, 33.20],
}

df = pd.DataFrame(data)
print(df)
print(f"\nShape: {df.shape}")

print("\n--- Column selection and average ---")
infl_column = df["infl"]
print(infl_column)
print(f"\nAverage inflation: {df['infl'].mean():.2f}%")

print("\n--- Converting dates and setting index ---")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")
print(df)
print(f"\nIndex type: {type(df.index)}")

print("\n--- First differencing ---")
infl_diff = df["infl"].diff()
print(infl_diff)
print("\nThe first value is NaN because there is no previous month to subtract from.")
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see** (new output at the bottom):

```
--- First differencing ---
date
2024-01-01    NaN
2024-02-01    1.8
2024-03-01    1.5
Name: infl, dtype: float64

The first value is NaN because there is no previous month to subtract from.
```

**What just happened:**

- `.diff()` computes the **change from one row to the next**. This is called "first differencing." For each row, it subtracts the previous row's value from the current row's value:
  - January: no previous row, so the result is `NaN` (Not a Number -- pandas' way of saying "this value is missing").
  - February: 31.70 - 29.90 = 1.8 (inflation rose by 1.8 percentage points).
  - March: 33.20 - 31.70 = 1.5 (inflation rose by 1.5 percentage points).
- `NaN` always appears in the first row after `.diff()` because there is no row before it to subtract from. This is normal and expected. You will see this throughout the project.
- First differencing is extremely important in econometrics and time series analysis. Many statistical models (including the ARDL and VAR models you will build later) require the data to be **stationary** -- meaning the statistical properties (mean, variance) do not change over time. Raw inflation numbers tend to trend upward or downward, which is non-stationary. Differencing removes that trend by converting levels into changes. Instead of asking "what is inflation?" you ask "how much did inflation change this month?" The differenced series is often stationary even when the original is not.

---

### Step 5: Missing Values

Delete everything in `guide/week1/day2_practice.py` and replace it with this:

```python
import pandas as pd

data = {
    "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
    "mpr": [18.75, 22.75, 24.75],
    "infl": [29.90, 31.70, 33.20],
}

df = pd.DataFrame(data)
print(df)
print(f"\nShape: {df.shape}")

print("\n--- Column selection and average ---")
infl_column = df["infl"]
print(infl_column)
print(f"\nAverage inflation: {df['infl'].mean():.2f}%")

print("\n--- Converting dates and setting index ---")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")
print(df)
print(f"\nIndex type: {type(df.index)}")

print("\n--- First differencing ---")
infl_diff = df["infl"].diff()
print(infl_diff)
print("\nThe first value is NaN because there is no previous month to subtract from.")

print("\n--- Missing values ---")
df.loc["2024-02-01", "mpr"] = None
print("DataFrame with one missing value:")
print(df)
print(f"\nMissing values per column:\n{df.isnull().sum()}")
print(f"\nAfter interpolation:")
df_filled = df.interpolate()
print(df_filled)
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see** (new output at the bottom):

```
--- Missing values ---
DataFrame with one missing value:
              mpr   infl
date
2024-01-01  18.75  29.90
2024-02-01    NaN  31.70
2024-03-01  24.75  33.20

Missing values per column:
mpr     1
infl    0
dtype: int64

After interpolation:
              mpr   infl
date
2024-01-01  18.75  29.90
2024-02-01  21.75  31.70
2024-03-01  24.75  33.20
```

**What just happened:**

- `df.loc["2024-02-01", "mpr"] = None` manually sets the MPR value for February to "missing." In Python, `None` means "no value." When pandas stores `None` in a numeric column, it displays it as `NaN` (Not a Number). `NaN` is the universal marker for missing data in pandas.
- `df.isnull().sum()` is two operations chained together. First, `df.isnull()` creates a DataFrame of `True`/`False` values -- `True` wherever a value is missing, `False` wherever a value exists. Then `.sum()` counts the `True` values in each column. The result tells you exactly how many missing values each column has. Here, `mpr` has 1 and `infl` has 0.
- `df.interpolate()` fills in missing values by estimating what they should be based on the values on either side. February's MPR was missing, and the values around it were 18.75 (January) and 24.75 (March). Linear interpolation fills it with the midpoint: (18.75 + 24.75) / 2 = 21.75. Notice we saved the result to a new variable `df_filled` instead of overwriting `df` -- this is good practice because it lets you compare before and after.
- Real-world datasets almost always have missing values. The CBN dataset in `data/raw/cbn_infl_data.csv` may have gaps where data was not reported for certain months. Knowing how to detect missing values (`isnull()`) and fill them (`interpolate()`) is essential. You will use these exact techniques when cleaning the real data later in this project.

---

### Step 6: Preview of Reading CSV Files

Delete everything in `guide/week1/day2_practice.py` and replace it with this:

```python
import pandas as pd

data = {
    "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
    "mpr": [18.75, 22.75, 24.75],
    "infl": [29.90, 31.70, 33.20],
}

df = pd.DataFrame(data)
print(df)
print(f"\nShape: {df.shape}")

print("\n--- Column selection and average ---")
infl_column = df["infl"]
print(infl_column)
print(f"\nAverage inflation: {df['infl'].mean():.2f}%")

print("\n--- Converting dates and setting index ---")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")
print(df)
print(f"\nIndex type: {type(df.index)}")

print("\n--- First differencing ---")
infl_diff = df["infl"].diff()
print(infl_diff)
print("\nThe first value is NaN because there is no previous month to subtract from.")

print("\n--- Missing values ---")
df.loc["2024-02-01", "mpr"] = None
print("DataFrame with one missing value:")
print(df)
print(f"\nMissing values per column:\n{df.isnull().sum()}")
print(f"\nAfter interpolation:")
df_filled = df.interpolate()
print(df_filled)

print("\n--- CSV preview (what you will use tomorrow) ---")
print("To load the real CBN data, you will write:")
print()
print('  df = pd.read_csv("data/raw/cbn_infl_data.csv")')
print()
print("To load it with dates as the index in one step:")
print()
print('  df = pd.read_csv("data/raw/cbn_infl_data.csv", index_col="date", parse_dates=True)')
print()
print("The real CSV has 34 columns. The four we care about most are:")
print("  mpr, infl, exo, tbr")
print()
print("You will build this for real tomorrow in data_ingestion/ingest.py")
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see** (new output at the bottom):

```
--- CSV preview (what you will use tomorrow) ---
To load the real CBN data, you will write:

  df = pd.read_csv("data/raw/cbn_infl_data.csv")

To load it with dates as the index in one step:

  df = pd.read_csv("data/raw/cbn_infl_data.csv", index_col="date", parse_dates=True)

The real CSV has 34 columns. The four we care about most are:
  mpr, infl, exo, tbr

You will build this for real tomorrow in data_ingestion/ingest.py
```

**What just happened:**

- This step does not run any new pandas operations on data. It simply prints the syntax you will use starting tomorrow.
- `pd.read_csv("data/raw/cbn_infl_data.csv")` loads a CSV file from disk into a DataFrame. This is how real data enters your program. You give it a file path, and it returns a DataFrame with all the rows and columns from that file.
- `index_col="date"` tells pandas to use the column named "date" as the row index, instead of the default 0, 1, 2, ... numbering. This is the same thing `set_index("date")` did in Step 3, but done at load time so you do not need a separate step.
- `parse_dates=True` tells pandas to automatically detect and convert date strings into datetime objects. This is the same thing `pd.to_datetime()` did in Step 3, but again done at load time.
- The real CBN dataset has 34 columns, but the four we will focus on are `mpr` (Monetary Policy Rate), `infl` (headline inflation), `exo` (official exchange rate), and `tbr` (Treasury Bill Rate). You will learn about the other columns as the project progresses.

---

Step 6 above is your final complete file. If your output matches all the expected outputs shown above, you are done with Part 2.

---

## Part 3: Key Concepts Summary

Here is a quick-reference table of every pandas operation you used today. Keep this handy -- you will use all of these repeatedly throughout the project.

| Code | What It Does |
|------|-------------|
| `import pandas as pd` | Load the pandas library with the short name `pd` |
| `pd.DataFrame(data)` | Create a table from a dictionary |
| `df.shape` | Returns `(rows, columns)` as a tuple |
| `df["column_name"]` | Select one column (returns a Series) |
| `df["col"].mean()` | Compute the average of a column |
| `pd.to_datetime(col)` | Convert text strings to datetime objects |
| `df.set_index("col")` | Make a column the row label (index) |
| `df["col"].diff()` | Compute change from previous row (first differencing) |
| `df.loc[row, col] = None` | Set a specific cell to missing |
| `df.isnull().sum()` | Count missing values per column |
| `df.interpolate()` | Fill missing values using linear interpolation |
| `pd.read_csv("path")` | Load a CSV file into a DataFrame |
| `f"{value:.2f}"` | Format a number to 2 decimal places in an f-string |

---

## Part 4: Commit Your Work

You made two changes today: the practice script and the updated requirements file. Commit them both.

```bash
git add guide/week1/day2_practice.py requirements.txt
git commit -m "Day 2: pandas practice script and requirements"
```

**What you should see:**

```
[main xxxxxxx] Day 2: pandas practice script and requirements
 2 files changed, ...
```

If git says "nothing to commit," make sure you saved both files and that you are in the project root directory.

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'pandas'` | Your virtual environment is not activated. Run `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows) and try again. |
| `pip install` fails with a permission error | You are not inside the venv. Never use `sudo pip install`. Activate the venv first, then run pip. |
| `FileNotFoundError` when running the script | Make sure you are in the project root directory (`Nigerian_Inflation_Predictor/`) when you run `python guide/week1/day2_practice.py`. Do not cd into `guide/week1/` first. |
| `KeyError: 'infl'` | Column names are case-sensitive. Check that you typed `"infl"` exactly as shown, not `"Infl"` or `"INFL"` or `"inflation"`. |
| `IndentationError: unexpected indent` | Python uses spaces to define code blocks. Make sure you are not mixing tabs and spaces. Use 4 spaces per indent level. Most code editors have a setting for this. |
| Numbers display with too many decimals | Use f-strings with format specifiers: `f"{value:.2f}"` rounds the display to 2 decimal places. This only affects the display -- the actual stored value is unchanged. |
| `SyntaxError` on the f-string line | Make sure the outer quotes and inner quotes are different types. If the f-string uses double quotes `"..."`, use single quotes `'infl'` inside the curly braces, or vice versa. |

---

## Check Your Understanding

These are questions an examiner, interviewer, or thesis advisor might ask. Read the question, think about your answer, then check below.

**1. What is a pandas DataFrame, and how did you create one today?**

A DataFrame is a two-dimensional table with rows and columns, similar to a spreadsheet or a SQL table. Each column has a name and holds one type of data. Each row has an index (a label). Today you created one by passing a Python dictionary to `pd.DataFrame()`. The dictionary keys became column names (`"date"`, `"mpr"`, `"infl"`) and the dictionary values (lists of numbers or strings) became the data in each column.

**2. Why do we convert dates with `pd.to_datetime()` and set them as the index?**

By default, pandas treats date strings like `"2024-01-01"` as plain text. Converting them to datetime objects means pandas understands chronological order and can do date arithmetic (like calculating the time between two dates). Setting the date as the index means each row is identified by its date rather than by a meaningless number (0, 1, 2, ...). This is the standard setup for time series data. It also allows you to select data by date range, like `df["2024-01":"2024-06"]`, which would not work with text dates.

**3. What does `.diff()` do, and why does it matter for econometric modeling?**

`.diff()` computes the period-to-period change by subtracting each row's value from the next row's value. The first row always becomes `NaN` because there is no previous row to subtract from. This operation is called "first differencing" and it is critical for time series econometrics because many models (ARDL, VAR) require the input data to be stationary -- meaning the statistical properties do not change over time. Raw economic variables like inflation or exchange rates often have trends, which makes them non-stationary. Differencing removes the trend by converting the data from levels to changes, which is often enough to achieve stationarity.

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Practice script | `guide/week1/day2_practice.py` | Learned DataFrames, column selection, dates, differencing, missing values, CSV syntax |
| Updated requirements | `requirements.txt` | Now contains `pandas==2.1.4` |

**Packages installed today:** `pandas==2.1.4` (plus its automatic dependencies: `numpy`, `pytz`, `tzdata`, `python-dateutil`, `six`)

**Core variables introduced today:** `mpr` (Monetary Policy Rate), `infl` (headline inflation), `exo` (official exchange rate), `tbr` (Treasury Bill Rate)

**Tomorrow (Day 3):** You will load the real CBN dataset from `data/raw/cbn_infl_data.csv` and build the data ingestion script (`data_ingestion/ingest.py`) that reads all 34 columns into a pandas DataFrame.
