# Week 1, Day 2 — Python & Pandas: Learning by Doing

## What You Will Learn Today

- How to create a virtual environment (an isolated Python workspace)
- How to install your first package (`pandas`)
- What a DataFrame is and why it matters for this project
- How to select columns, convert dates, compute changes, and handle missing data
- How to run a Python script from the terminal

## Why This Matters

Every single file in this project uses `pandas`. It is the Python library for working with tabular data (rows and columns, like a spreadsheet). You cannot move forward without understanding DataFrames. Today you will build that foundation one small piece at a time.

---

## Part 1: Setup (Virtual Environment + Install pandas)

### 1A: Create the Virtual Environment

A virtual environment is an isolated Python workspace. Packages you install inside it do not affect your system Python. Every time you work on this project, you will activate it first.

```bash
cd Nigerian_Inflation_Predictor
python -m venv venv
```

### 1B: Activate It

**Mac / Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

**How you know it worked:** Your terminal prompt now starts with `(venv)`:
```
(venv) $
```

You must activate the virtual environment every time you open a new terminal to work on this project. If you do not see `(venv)` at the start of your prompt, nothing you install will be available.

### 1C: Update requirements.txt

Open the file `requirements.txt` in the root of the project. Add this single line (or replace what is there):

```
pandas==2.1.4
```

Save the file.

### 1D: Install pandas

```bash
pip install -r requirements.txt
```

You will see a wall of download and install messages. The last line should say something like:

```
Successfully installed pandas-2.1.4 ...
```

### 1E: Verify It Works

Run this one-liner in your terminal:

```bash
python -c "import pandas; print(pandas.__version__)"
```

**What you should see:**
```
2.1.4
```

If you see that version number, pandas is installed and working. If you get `ModuleNotFoundError`, your virtual environment is not activated. Go back to step 1B.

---

## Part 2: Your First Python Script (Build It Piece by Piece)

We are going to create a practice file at `guide/week1/day2_practice.py`. This is NOT part of the main project code. It is purely for learning. You will write a few lines, run the file, see the output, understand what happened, and then add more lines.

Do not skip ahead. Do not copy the whole thing at once. The learning happens when you see each piece work on its own.

---

### Step 1: Create the File and Print a DataFrame

Create a new file at this path: `guide/week1/day2_practice.py`

Write exactly these lines in it:

```python
import pandas as pd

data = {
    "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
    "mpr": [18.75, 22.75, 24.75],
    "inflation": [29.90, 31.70, 33.20],
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
         date    mpr  inflation
0  2024-01-01  18.75      29.90
1  2024-02-01  22.75      31.70
2  2024-03-01  24.75      33.20

Shape: (3, 3)
```

**What just happened:**

- `import pandas as pd` loads the pandas library and gives it the short name `pd`. Everyone uses this convention.
- `data = { ... }` is a Python dictionary. Each key (`"date"`, `"mpr"`, `"inflation"`) becomes a column name. Each value is a list of entries for that column.
- `pd.DataFrame(data)` turns that dictionary into a **DataFrame** -- a table with rows and columns, like a spreadsheet. This is the single most important object in pandas. Almost everything you do in this project starts with a DataFrame.
- `df.shape` returns a tuple `(rows, columns)`. Here it is `(3, 3)` because we have 3 rows and 3 columns.

---

### Step 2: Select a Column and Compute an Average

Open `guide/week1/day2_practice.py` again. Add these lines **to the very bottom** of the file, below what you already have:

```python
print("\n--- Selecting columns ---")
print(df["inflation"])
print(f"\nAverage inflation: {df['inflation'].mean():.2f}%")
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see** (new output at the bottom):

```
--- Selecting columns ---
0    29.90
1    31.70
2    33.20
Name: inflation, dtype: float64

Average inflation: 31.60%
```

**What just happened:**

- `df["inflation"]` selects a single column from the DataFrame. The result is called a **Series** -- think of it as one column pulled out of the table. The numbers on the left (0, 1, 2) are the row index.
- `.mean()` computes the average of all values in that column: (29.90 + 31.70 + 33.20) / 3 = 31.60.
- `:.2f` inside the f-string means "format this number with exactly 2 decimal places."

---

### Step 3: Convert Dates and Set the Index

Add these lines **to the very bottom** of `guide/week1/day2_practice.py`:

```python
print("\n--- Converting dates ---")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")
print(df)
print(f"\nData type of index: {type(df.index)}")
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see** (new output at the bottom):

```
--- Converting dates ---
              mpr  inflation
date
2024-01-01  18.75      29.90
2024-02-01  22.75      31.70
2024-03-01  24.75      33.20

Data type of index: <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
```

**What just happened:**

- `pd.to_datetime(df["date"])` converts the "date" column from plain text strings (like `"2024-01-01"`) into actual datetime objects that Python understands as dates. Before this, pandas treated those dates as ordinary text. After this, it knows they are dates and can do date math with them.
- `df.set_index("date")` makes the date column the **row labels** instead of a regular column. Notice how "date" moved from being a column to being on the left side of the table. This is standard practice for time series data -- each row is identified by its date.
- The index type is `DatetimeIndex`, which confirms pandas now treats our dates as real dates, not text.

---

### Step 4: First Differencing

Add these lines **to the very bottom** of `guide/week1/day2_practice.py`:

```python
print("\n--- First differencing ---")
print(df["inflation"].diff())
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
Name: inflation, dtype: float64

The first value is NaN because there is no previous month to subtract from.
```

**What just happened:**

- `.diff()` computes the **change from one row to the next**. This is called "first differencing." For each row, it subtracts the previous row's value from the current row's value.
  - January: no previous row, so the result is `NaN` (Not a Number -- pandas' way of saying "missing").
  - February: 31.70 - 29.90 = 1.8
  - March: 33.20 - 31.70 = 1.5
- First differencing is extremely important in economics and time series analysis. Instead of looking at the level of inflation (29.9%, 31.7%, 33.2%), you look at the **change** in inflation (+1.8, +1.5). This tells you whether inflation is accelerating or slowing down.
- `NaN` always appears in the first row after `.diff()` because there is no row before it to subtract from. This is normal and expected.

---

### Step 5: Missing Values

Add these lines **to the very bottom** of `guide/week1/day2_practice.py`:

```python
print("\n--- Missing values ---")
df.loc["2024-02-01", "mpr"] = None
print(df)
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nAfter interpolation:")
print(df.interpolate())
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see** (new output at the bottom):

```
--- Missing values ---
              mpr  inflation
date
2024-01-01  18.75      29.90
2024-02-01    NaN      31.70
2024-03-01  24.75      33.20

Missing values:
mpr          1
inflation    0
dtype: int64

After interpolation:
              mpr  inflation
date
2024-01-01  18.75      29.90
2024-02-01  21.75      31.70
2024-03-01  24.75      33.20
```

**What just happened:**

- `df.loc["2024-02-01", "mpr"] = None` manually sets the MPR value for February to "missing." In pandas, missing values are displayed as `NaN`.
- `df.isnull().sum()` counts how many missing values exist in each column. Here, `mpr` has 1 missing value and `inflation` has 0.
- `df.interpolate()` fills in missing values by estimating what they should be based on the surrounding values. February's MPR was missing, and the values around it were 18.75 (January) and 24.75 (March). Interpolation fills it with the midpoint: (18.75 + 24.75) / 2 = 21.75.
- Real-world datasets almost always have missing values. Knowing how to detect them (`isnull()`) and fill them (`interpolate()`) is essential. You will use these techniques when cleaning Nigerian economic data later in this project.

---

### Step 6: A Preview of Reading CSV Files

Add these lines **to the very bottom** of `guide/week1/day2_practice.py`:

```python
print("\n--- Reading from CSV ---")
print("In this project, we load data like this:")
print('  df = pd.read_csv("data/raw/nigeria_macro_data.csv")')
print('  df = pd.read_csv("path", index_col="date", parse_dates=True)')
print("\nYou will build this for real tomorrow in data_ingestion/ingest.py")
```

Save the file.

**Run it:**

```bash
python guide/week1/day2_practice.py
```

**What you should see** (new output at the bottom):

```
--- Reading from CSV ---
In this project, we load data like this:
  df = pd.read_csv("data/raw/nigeria_macro_data.csv")
  df = pd.read_csv("path", index_col="date", parse_dates=True)

You will build this for real tomorrow in data_ingestion/ingest.py
```

**What just happened:**

- This step does not run any actual pandas code on data. It simply prints the syntax you will use tomorrow.
- `pd.read_csv("path")` loads a CSV file from disk into a DataFrame. This is how real data enters your program.
- `index_col="date"` tells pandas to use the "date" column as the row index (instead of the default 0, 1, 2, ...).
- `parse_dates=True` tells pandas to automatically convert date strings into datetime objects.
- Tomorrow on Day 3, you will create the actual CSV file and write the ingestion script. Today was about learning what a DataFrame is and how to work with one.

---

### Complete File (Verify Yours Matches)

After all 6 steps, your `guide/week1/day2_practice.py` should look exactly like this:

```python
import pandas as pd

data = {
    "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
    "mpr": [18.75, 22.75, 24.75],
    "inflation": [29.90, 31.70, 33.20],
}

df = pd.DataFrame(data)
print(df)
print(f"\nShape: {df.shape}")

print("\n--- Selecting columns ---")
print(df["inflation"])
print(f"\nAverage inflation: {df['inflation'].mean():.2f}%")

print("\n--- Converting dates ---")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")
print(df)
print(f"\nData type of index: {type(df.index)}")

print("\n--- First differencing ---")
print(df["inflation"].diff())
print("\nThe first value is NaN because there is no previous month to subtract from.")

print("\n--- Missing values ---")
df.loc["2024-02-01", "mpr"] = None
print(df)
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nAfter interpolation:")
print(df.interpolate())

print("\n--- Reading from CSV ---")
print("In this project, we load data like this:")
print('  df = pd.read_csv("data/raw/nigeria_macro_data.csv")')
print('  df = pd.read_csv("path", index_col="date", parse_dates=True)')
print("\nYou will build this for real tomorrow in data_ingestion/ingest.py")
```

If your file does not match, delete it and retype it from this complete version. Then run it one final time to confirm everything works:

```bash
python guide/week1/day2_practice.py
```

---

## Part 3: Key Concepts Summary

Here is a quick-reference table of every pandas operation you used today. Keep this handy.

```
pd.DataFrame(data)     -- Create a table from a dictionary
df.head() / df.tail()  -- See first/last rows
df.shape               -- (rows, columns)
df["column"]           -- Select one column
df["col"].mean()       -- Average of a column
pd.to_datetime()       -- Convert text to dates
df.set_index("date")   -- Make date the row labels
df["col"].diff()       -- Change from previous row
df.isnull().sum()      -- Count missing values
df.interpolate()       -- Fill gaps with estimated values
pd.read_csv("path")   -- Load a CSV file
```

---

## Part 4: Commit Your Work

```bash
git add guide/week1/day2_practice.py requirements.txt
git commit -m "Day 2: Pandas practice script"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'pandas'` | Your virtual environment is not activated. Run `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows) and try again. |
| `pip install` fails with a permission error | You are not inside the venv. Never use `sudo pip install`. Activate the venv first. |
| `FileNotFoundError` when running the script | Make sure you are in the project root directory (`Nigerian_Inflation_Predictor/`) when you run the command, not inside the `guide/` folder. |
| `KeyError: 'inflation'` | Column names are case-sensitive. Check that you typed `"inflation"` exactly, not `"Inflation"` or `"INFLATION"`. |
| Numbers have too many decimals | Use f-strings with format specifiers: `f"{value:.2f}"` rounds the display to 2 decimal places. |
| `IndentationError` | Python uses spaces to define code blocks. Make sure you are not mixing tabs and spaces. Use 4 spaces per indent level. |

---

## Check Your Understanding

**1. What is a pandas DataFrame?**

A table with rows and columns, like a spreadsheet. Each column has a name and holds one type of data. Each row has an index (usually a date for time series). You created one from a dictionary using `pd.DataFrame(data)`.

**2. Why do we convert dates with `pd.to_datetime()` and then set them as the index?**

Because pandas treats date text like `"2024-01-01"` as a plain string by default. Converting to datetime lets pandas understand the chronological order, do date math, and select date ranges. Setting dates as the index means each row is labeled by its date, which is the standard setup for time series analysis.

**3. What does `.diff()` do and why does the first value come back as NaN?**

`.diff()` subtracts the previous row's value from the current row's value, giving you the period-to-period change. The first row has no previous row to subtract from, so the result is `NaN` (Not a Number), which is pandas' way of saying "this value is missing." This is completely normal and expected.

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Practice script | `guide/week1/day2_practice.py` | Learned DataFrames, column selection, dates, differencing, missing values |
| Updated requirements | `requirements.txt` | Now lists `pandas==2.1.4` |

**Packages installed today:** `pandas==2.1.4` (plus its automatic dependencies: `numpy`, `pytz`, `tzdata`, `python-dateutil`, `six`)

**Tomorrow (Day 3):** You will create the actual Nigerian macro dataset and build the data ingestion script (`data_ingestion/ingest.py`) that loads it into pandas for real.
