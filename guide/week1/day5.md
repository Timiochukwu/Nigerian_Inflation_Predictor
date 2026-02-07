# Week 1, Day 5 — Exploratory Data Analysis: Statistics & Plots, Built Step by Step

## What You'll Learn Today

- How to install matplotlib for plotting
- How to compute summary statistics and what they mean
- How to create time series plots
- How to create a correlation matrix
- What the data tells us about Nigeria's economy

## Why This Matters

Exploratory Data Analysis (EDA) is NOT optional. It is the most important step between loading your data and building models. Every examiner, every thesis committee, every job interviewer will ask you the same question before they care about your model: **"Describe your data."**

If you cannot describe what your data looks like — its central tendency, its spread, its trends, its anomalies — then no amount of fancy modelling will save you. You must know your data better than anyone in the room.

EDA serves three critical purposes:

1. **It reveals structural breaks.** A time series plot will show you that Nigeria's exchange rate did not move gradually — it jumped violently in June 2016 and again in June 2023. These jumps (structural breaks) affect which econometric models are appropriate. If you skip EDA, you will choose the wrong model.

2. **It reveals trends and non-stationarity.** If a variable trends upward over 25 years (like M2 money supply), it is not stationary. You need to know this before fitting any regression, because regressing two non-stationary series on each other can produce "spurious regression" — results that look significant but are meaningless.

3. **It reveals outliers and data errors.** Maybe one month has an inflation value of 999 because of a data entry mistake. Maybe the exchange rate is recorded in different units for different periods. Plots catch these problems instantly; staring at a spreadsheet does not.

Bottom line: EDA is where you build the story of your data. That story is what your thesis or report is really about. The models just quantify what you already see.

---

## Step 1: Install matplotlib

matplotlib is Python's most widely used plotting library. It creates publication-quality charts, and it is the foundation that many other plotting libraries (like seaborn) are built on.

Update your `requirements.txt` so it reads:

```
pandas==2.1.4
matplotlib==3.8.2
```

Then install it:

```bash
pip install matplotlib==3.8.2
```

**What you should see:** A lot of download and installation text ending with `Successfully installed matplotlib-3.8.2` (plus some dependencies like `contourpy`, `cycler`, `kiwisolver`, `pillow`, etc.).

Verify it installed correctly:

```bash
python -c "import matplotlib; print(matplotlib.__version__)"
```

That should print `3.8.2`.

**Why we pin the version:** Writing `matplotlib==3.8.2` instead of just `matplotlib` ensures everyone who runs your project gets the same version. This prevents the "it works on my machine" problem.

---

## Building `data_processing/eda.py` — STEP BY STEP

We are NOT going to dump one giant script on you. Instead, we will build the file piece by piece. You will write a small chunk, run it, see the output, understand what it does, and THEN add the next piece. By the end you will have the same complete file, but you will actually understand every line because you watched it come to life.

Make sure you have a `data_processing/` folder with an `__init__.py` inside it (from Day 4). If not, create them now.

---

### Build Step 1: Create the File with Imports and Data Loading

Create a new file called `data_processing/eda.py` and type this in:

```python
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend (no window pops up)
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

def load_cleaned_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)

if __name__ == "__main__":
    df = load_cleaned_data()
    print(f"Loaded {len(df)} rows")
    print(df.head())
```

**Save the file.** Now run it:

```bash
python -m data_processing.eda
```

**What you should see:**

```
Loaded 300 rows
              mpr  inflation  exchange_rate       m2
date
2000-01-01  13.5       6.62          92.34   1070.50
2000-02-01  13.5       7.80          97.50   1085.40
...
```

**What just happened — line by line:**

- `import matplotlib` and `matplotlib.use("Agg")` — This tells matplotlib to save plots to files instead of trying to open a window on your screen. The "Agg" backend is non-interactive. This MUST come before `import matplotlib.pyplot as plt`. If you put them in the wrong order, you will get errors on servers or remote machines that have no display.
- `RESULTS_DIR` and `PROCESSED_DIR` — These are paths relative to where the script lives. `os.path.dirname(__file__)` means "the folder this script is in" (which is `data_processing/`). The `".."` goes up one level to the project root, then into the target subfolder.
- `load_cleaned_data()` — Reads the CSV that Day 4's cleaning script created. `index_col="date"` makes the date column the row index (standard for time series). `parse_dates=True` converts the date strings to actual datetime objects.
- `if __name__ == "__main__":` — This block only runs when you execute the script directly. It does NOT run if another script imports functions from this file.

If that ran and printed your data, you are ready for the next piece.

---

### Build Step 2: Add the summary_statistics Function

Open `data_processing/eda.py`. Add this function **ABOVE** the `if __name__` block (between `load_cleaned_data` and `if __name__`):

```python
def summary_statistics(df):
    stats = df.describe().T
    stats["skewness"] = df.skew()
    stats["kurtosis"] = df.kurtosis()

    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(stats.round(2).to_string())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stats.round(4).to_csv(os.path.join(RESULTS_DIR, "summary_statistics.csv"))
    print(f"\nSaved to results/summary_statistics.csv")
    return stats
```

Now **update** the `if __name__` block at the bottom to call this new function:

```python
if __name__ == "__main__":
    df = load_cleaned_data()
    summary_statistics(df)
```

**Save and run:**

```bash
python -m data_processing.eda
```

**What you should see:**

```
======================================================================
SUMMARY STATISTICS
======================================================================
               count    mean       std    min     25%      50%       75%        max  skewness  kurtosis
mpr            300.0   12.68     3.71    6.00   11.00    13.00     14.00      27.50      1.47      3.24
inflation      300.0   13.48     6.52    3.00    8.65    11.61     17.10      34.80      1.32      1.22
exchange_rate  300.0   268.43   258.71   92.34  132.50   197.00    386.00   1680.00      2.35      6.18
m2             300.0  45218.37 62845.20 1070.50 3195.30 10140.80  62505.80 284975.80     1.70      2.26
======================================================================

Saved to results/summary_statistics.csv
```

(Your exact numbers may differ slightly depending on your data.)

**What each statistic means:**

- `df.describe()` computes, in one call: count (number of non-missing values), mean (average), std (standard deviation — how spread out the data is), min, 25% (first quartile), 50% (median), 75% (third quartile), and max.
- `.T` transposes the table so each variable is a row instead of a column. This makes it easier to read when printed.
- `df.skew()` measures how lopsided the distribution is. Positive skewness means the right tail is longer (most values are low, but there are a few very high ones). Negative means the left tail is longer. Zero means symmetric. For economic data, positive skew often means a few extreme high values are pulling the mean above the median.
- `df.kurtosis()` measures how heavy the tails are compared to a normal distribution. Positive kurtosis means more extreme values than a bell curve would predict. Negative means fewer. Pandas uses "excess kurtosis," so a normal distribution gives 0.

You now have a CSV file in `results/` that you can paste into your thesis appendix. Move on.

---

### Build Step 3: Add plot_time_series — Start with Just ONE Plot

We will not create all the plots at once. Let us start with a single plot — inflation — so you can see how matplotlib works.

Add this function **ABOVE** the `if __name__` block:

```python
def plot_time_series(df):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Start with just one plot — inflation
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["inflation"], color="#d62728", linewidth=1.2)
    ax.set_title("Headline Inflation Rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("Percent (%, YoY)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ts_inflation.png"), dpi=150)
    plt.close(fig)
    print("Saved: ts_inflation.png")
```

**Update** the `if __name__` block:

```python
if __name__ == "__main__":
    df = load_cleaned_data()
    summary_statistics(df)
    plot_time_series(df)
```

**Save and run:**

```bash
python -m data_processing.eda
```

**What you should see** (after the summary stats output):

```
Saved: ts_inflation.png
```

Now go open `results/ts_inflation.png` and **look at it**. This is important. Do not skip this step.

**What you are looking at:** A line chart showing Nigeria's monthly inflation rate from 2000 to 2024. You should see that inflation was low in 2006-2007 (the oil boom years), spiked around 2016 (when the Naira was devalued), and then exploded upward after 2020, reaching above 30% by 2024.

**How the matplotlib code works:**

- `fig, ax = plt.subplots(figsize=(12, 4))` — Creates a figure (the blank canvas) and an axes object (where the plot goes). `figsize=(12, 4)` means 12 inches wide, 4 inches tall. Wide is good for time series because time goes left to right.
- `ax.plot(df.index, df["inflation"], ...)` — Draws a line. `df.index` is the dates (x-axis), `df["inflation"]` is the values (y-axis). `color="#d62728"` is red. `linewidth=1.2` makes the line slightly thicker than the default.
- `ax.set_title(...)`, `ax.set_ylabel(...)`, `ax.set_xlabel(...)` — Add labels. Without these, the plot has no context and is useless in a thesis.
- `ax.grid(True, alpha=0.3)` — Adds light gridlines. `alpha=0.3` means 30% opaque, so they are subtle and do not compete with the data.
- `plt.tight_layout()` — Automatically adjusts spacing so nothing gets cut off.
- `fig.savefig(...)` — Saves the plot to a file. `dpi=150` means 150 dots per inch, which is good for screen viewing and thesis printing.
- `plt.close(fig)` — Frees the memory used by this figure. Without this, creating many plots in a loop will consume more and more RAM.

Good. One plot works. Now let us make it plot all four variables.

---

### Build Step 4: Expand plot_time_series to Plot ALL Four Variables

Now **REPLACE** the entire `plot_time_series` function with this expanded version:

```python
def plot_time_series(df):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    variables = {
        "mpr": ("Monetary Policy Rate (MPR)", "Percent (%)", "#1f77b4"),
        "inflation": ("Headline Inflation Rate", "Percent (%, YoY)", "#d62728"),
        "exchange_rate": ("Exchange Rate (NGN/USD)", "Naira per USD", "#2ca02c"),
        "m2": ("Broad Money Supply (M2)", "NGN Billions", "#9467bd"),
    }

    # Individual plots
    for col, (title, ylabel, color) in variables.items():
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df[col], color=color, linewidth=1.2)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"ts_{col}.png"), dpi=150)
        plt.close(fig)
        print(f"Saved: ts_{col}.png")

    # Combined 4-panel plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    for ax, (col, (title, ylabel, color)) in zip(axes, variables.items()):
        ax.plot(df.index, df[col], color=color, linewidth=1.2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date", fontsize=12)
    fig.suptitle("Nigerian Macroeconomic Variables", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ts_panel_all_variables.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: ts_panel_all_variables.png")
```

The `if __name__` block stays the same as before. **Save and run:**

```bash
python -m data_processing.eda
```

**What you should see:**

```
Saved: ts_mpr.png
Saved: ts_inflation.png
Saved: ts_exchange_rate.png
Saved: ts_m2.png
Saved: ts_panel_all_variables.png
```

Now you have 5 plot files in `results/`. Open each one and look at it.

**What changed from the previous version:**

- Instead of hardcoding one variable, we use a dictionary called `variables` that maps each column name to its display title, y-axis label, and colour. This makes the loop clean and easy to extend if you ever add more variables.
- The `for col, (title, ylabel, color) in variables.items():` loop creates one plot per variable. Same logic as before, just repeated.
- The second section creates a combined 4-panel plot using `plt.subplots(4, 1, ...)` — that means 4 rows, 1 column. `sharex=True` makes all four panels share the same x-axis (date), which is what you want so the time periods line up visually. `axes[-1]` is the bottom panel — we only put the x-axis label on the bottom one to avoid redundancy.

---

### Build Step 5: Add the Correlation Matrix

Add this function **ABOVE** the `if __name__` block:

```python
def plot_correlation_matrix(df):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    corr = df.corr()

    print("\nCorrelation Matrix:")
    print(corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    plt.colorbar(im)
    ax.set_title("Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "correlation_matrix.png"), dpi=150)
    plt.close(fig)
    print("Saved: correlation_matrix.png")
```

Now **update** the `if __name__` block one final time:

```python
if __name__ == "__main__":
    df = load_cleaned_data()
    summary_statistics(df)
    plot_time_series(df)
    plot_correlation_matrix(df)
    print("\nEDA complete.")
```

**Save and run:**

```bash
python -m data_processing.eda
```

**What you should see** (at the end, after all the previous output):

```
Correlation Matrix:
                  mpr  inflation  exchange_rate      m2
mpr            1.000      0.XXX          0.XXX   0.XXX
inflation      0.XXX      1.000          0.XXX   0.XXX
exchange_rate  0.XXX      0.XXX          1.000   0.XXX
m2             0.XXX      0.XXX          0.XXX   1.000
Saved: correlation_matrix.png

EDA complete.
```

(The `0.XXX` values will be actual correlation numbers when you run it.)

Open `results/correlation_matrix.png` and look at it.

**How the correlation code works:**

- `df.corr()` computes the Pearson correlation coefficient between every pair of columns. The result is a square matrix. The diagonal is always 1.0 (every variable is perfectly correlated with itself).
- `ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)` displays the matrix as a coloured grid. `cmap="RdBu_r"` means red = positive correlation, blue = negative, white = near zero. `vmin=-1, vmax=1` fixes the colour scale so the colours always mean the same thing.
- The nested `for i / for j` loop places the actual numbers inside each cell of the heatmap. `color="white" if abs(corr.iloc[i, j]) > 0.5 else "black"` makes the text readable: white text on dark cells, black text on light cells.
- `plt.colorbar(im)` adds the colour legend on the right side.

**What correlation means:**

- **+1.0** = the two variables move in the same direction perfectly
- **-1.0** = the two variables move in opposite directions perfectly
- **0.0** = no linear relationship at all

---

## The Complete Final File

After all five build steps, your `data_processing/eda.py` should look exactly like this:

```python
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend (no window pops up)
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

def load_cleaned_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def summary_statistics(df):
    stats = df.describe().T
    stats["skewness"] = df.skew()
    stats["kurtosis"] = df.kurtosis()

    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(stats.round(2).to_string())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stats.round(4).to_csv(os.path.join(RESULTS_DIR, "summary_statistics.csv"))
    print(f"\nSaved to results/summary_statistics.csv")
    return stats


def plot_time_series(df):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    variables = {
        "mpr": ("Monetary Policy Rate (MPR)", "Percent (%)", "#1f77b4"),
        "inflation": ("Headline Inflation Rate", "Percent (%, YoY)", "#d62728"),
        "exchange_rate": ("Exchange Rate (NGN/USD)", "Naira per USD", "#2ca02c"),
        "m2": ("Broad Money Supply (M2)", "NGN Billions", "#9467bd"),
    }

    # Individual plots
    for col, (title, ylabel, color) in variables.items():
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df[col], color=color, linewidth=1.2)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"ts_{col}.png"), dpi=150)
        plt.close(fig)
        print(f"Saved: ts_{col}.png")

    # Combined 4-panel plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    for ax, (col, (title, ylabel, color)) in zip(axes, variables.items()):
        ax.plot(df.index, df[col], color=color, linewidth=1.2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date", fontsize=12)
    fig.suptitle("Nigerian Macroeconomic Variables", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ts_panel_all_variables.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: ts_panel_all_variables.png")


def plot_correlation_matrix(df):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    corr = df.corr()

    print("\nCorrelation Matrix:")
    print(corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    plt.colorbar(im)
    ax.set_title("Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "correlation_matrix.png"), dpi=150)
    plt.close(fig)
    print("Saved: correlation_matrix.png")


if __name__ == "__main__":
    df = load_cleaned_data()
    summary_statistics(df)
    plot_time_series(df)
    plot_correlation_matrix(df)
    print("\nEDA complete.")
```

Compare your file against this. If something is different, fix it now before moving on.

---

## Interpreting Your Results — NIGERIA SPECIFIC

This is the most important section of today's guide. The numbers and plots are useless unless you know what they mean. This is what separates a student who runs code from a student who understands their research.

### What the MPR Plot Shows

The Monetary Policy Rate (MPR) is the interest rate set by the Central Bank of Nigeria (CBN). It is the price of borrowing money in Nigeria and the CBN's primary tool for controlling inflation.

When you look at the MPR plot, you should see **distinct policy regimes** — periods where the rate was held steady, followed by sudden jumps or drops. This is because the CBN does not adjust the MPR smoothly. The Monetary Policy Committee (MPC) meets every two months and decides to hold, raise, or cut the rate.

Key features you should see:

- **2000-2005 (13-15%):** Relatively high rates during the early democratic period. The CBN was fighting inflation inherited from the military era.
- **2006-2007 (~10%):** Rate cuts during the oil boom. Nigeria was running current account surpluses, foreign reserves were high, and inflation was low.
- **2008-2009 (drop to 6%):** The global financial crisis hit. The CBN slashed rates to stimulate the economy, mirroring central banks worldwide.
- **2011 (jump to 12%):** Sanusi Lamido Sanusi hiked the MPR dramatically to defend the Naira and fight imported inflation. This is one of the most famous CBN policy moves.
- **2011-2019 (12-14%):** A long period of relatively stable high rates.
- **2020 (cut to 11.5%):** COVID-19 response. The CBN cut rates to support the economy during lockdowns.
- **2022-2024 (surge to 27.5%):** The most aggressive tightening cycle in Nigerian history. The CBN under Cardoso raised rates from 11.5% to 27.5% as inflation spiralled out of control.

**What to tell your examiner:** "The MPR series shows distinct policy regimes with structural breaks, suggesting that a model allowing for regime changes may be more appropriate than one assuming a stable relationship."

### What the Inflation Plot Shows

The inflation rate measures how fast prices are rising. It is the variable you are trying to predict.

Key features you should see:

- **2000-2007 (volatile, 3-18%):** Inflation bounced around considerably but averaged in the single digits during the oil boom years of 2006-2007.
- **2008-2012 (10-16%):** Elevated inflation following the global financial crisis and food price shocks.
- **2013-2019 (8-18%):** A wide range. Inflation spiked in 2016-2017 after the Naira was devalued, then gradually came back down as the exchange rate stabilized.
- **2020-2024 (acceleration from 12% to 35%):** This is the defining feature of the series. Starting in 2020, inflation began a relentless climb. COVID disrupted supply chains, the Naira collapsed, food prices surged, and money supply grew rapidly. By late 2024, inflation was near 35%.

**What to tell your examiner:** "The inflation series shows a clear upward trend since 2020, with acceleration in 2023-2024 coinciding with exchange rate unification and Naira depreciation. The series appears non-stationary in levels, which we will formally test in Week 2."

### What the Exchange Rate Plot Shows

The exchange rate (NGN/USD) shows how many Naira you need to buy one US dollar. When this number goes up, the Naira is losing value (depreciating).

This is probably the most dramatic plot in your dataset:

- **2000-2008 (~92-132 NGN/USD):** Relatively stable. The CBN maintained a managed float, intervening heavily in the foreign exchange market.
- **2008-2009 (jump to 150):** The global financial crisis reduced oil revenue, and the CBN could no longer defend the old rate.
- **2009-2015 (~150-197):** Another period of managed stability. The CBN burned through foreign reserves to keep the rate steady.
- **2016 (jump from 199 to 310):** The CBN allowed the Naira to float (partially) after oil prices crashed. The official rate jumped overnight. This is a textbook structural break.
- **2017-2019 (~360):** The Naira settled at a new managed rate around 360.
- **2020-early 2023 (~360-462):** Gradual depreciation as COVID and falling oil prices depleted reserves.
- **June 2023 (jump from 462 to 750):** President Tinubu's government unified the exchange rate, ending the CBN's multiple exchange rate regime. The Naira instantly lost nearly 40% of its value.
- **2024 (surge to 1,500-1,680):** Continued freefall as markets adjusted to the new floating regime.

**What to tell your examiner:** "The exchange rate series contains at least three major structural breaks — 2016, June 2023, and early 2024. These step-changes violate the assumption of a smooth, continuous process. Any econometric model must account for these breaks."

### What the M2 Plot Shows

M2 (broad money supply) is the total amount of money circulating in the economy. It includes cash, checking accounts, savings accounts, and short-term deposits. It is measured in billions of Naira.

The key feature of this plot is that it is **exponential**:

- **2000 (~1,000 billion):** Nigeria's money supply was about 1 trillion Naira at the start of your sample.
- **2012 (~13,000 billion):** It grew to 13 trillion in 12 years.
- **2020 (~100,000 billion):** It exploded to 100 trillion by 2020.
- **2024 (~285,000 billion):** It nearly tripled in 4 years, reaching 285 trillion Naira by end-2024.

This exponential growth is important for two reasons:

1. **It suggests you should use the log of M2, not the raw value.** When a series grows exponentially, its variance increases over time (bigger numbers = bigger fluctuations). Taking the natural log makes the growth rate constant and stabilizes the variance. This is a standard transformation in econometrics.

2. **It explains much of the inflation story.** The quantity theory of money (MV = PQ) says that if money supply grows faster than real output, prices must rise. Nigeria's M2 has grown by roughly 20-30% per year, while real GDP grew maybe 2-4% per year. The difference shows up as inflation.

**What to tell your examiner:** "M2 shows exponential growth, with the growth rate accelerating in 2020-2024. The series is clearly non-stationary and will likely require log transformation and differencing before it can be used in regression models."

### What the Correlation Matrix Tells You

The correlation matrix shows the Pearson correlation coefficient between every pair of variables. Here is how to interpret the key relationships:

- **exchange_rate and m2 (very high positive, likely 0.95+):** These two variables have grown together over the 25-year period. As the money supply expanded, the Naira depreciated. This makes economic sense: more Naira chasing the same amount of foreign currency pushes the exchange rate up. However, be cautious — this high correlation is partly because both series have strong upward trends. Two trending series will always appear correlated even if there is no causal link. This is the "spurious regression" problem you will address in Week 2.

- **exchange_rate and inflation (high positive, likely 0.70-0.85):** When the Naira loses value, imported goods become more expensive, pushing up inflation. Nigeria imports a lot of what it consumes — refined petroleum, machinery, food products. So exchange rate depreciation feeds directly into consumer prices. Economists call this "exchange rate pass-through."

- **m2 and inflation (moderate to high positive):** Money supply growth and inflation tend to move together, consistent with the quantity theory of money.

- **mpr and inflation (moderate positive):** This might seem counterintuitive. If the CBN raises rates to fight inflation, should the correlation not be negative? Not necessarily. The CBN raises the MPR *in response to* rising inflation. So in the data, you see high MPR and high inflation at the same time — because the MPR was raised to fight inflation that was already happening. The causal effect (higher MPR eventually reduces inflation) takes time (lags) and is not visible in a simple correlation. This is exactly why you need the ARDL and VAR models in later weeks — they capture lagged, dynamic relationships that correlation cannot.

**What to tell your examiner:** "The correlation matrix shows strong positive associations between exchange rate, money supply, and inflation, consistent with economic theory. However, these correlations may be driven by common trends rather than causal relationships. We address this by testing for stationarity and using cointegration analysis in subsequent sections."

---

## Step 3: Commit

```bash
git add data_processing/eda.py requirements.txt
git commit -m "Day 5: Add EDA script with summary statistics, time series plots, and correlation matrix"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: Cleaned data not found` or similar | You need to run the cleaning script first: `python -m data_processing.clean`. The EDA script reads from `data/processed/cleaned_data.csv`, which is created by the Day 4 cleaning step. |
| `ModuleNotFoundError: No module named 'matplotlib'` | You forgot to install matplotlib. Run `pip install matplotlib==3.8.2` and try again. |
| `_tkinter.TclError: no display name` or `Tkinter not found` | This happens when matplotlib tries to open a plot window on a system without a display (like a remote server or WSL). The script handles this with `matplotlib.use("Agg")`, but make sure that line comes BEFORE `import matplotlib.pyplot as plt`. If you moved the imports around, put them back in the correct order. |
| `KeyError: 'mpr'` or `KeyError: 'inflation'` | Your cleaned CSV does not have the expected column names. Open `data/processed/cleaned_data.csv` and check the header row. The columns must be exactly: `date`, `mpr`, `inflation`, `exchange_rate`, `m2`. |
| `ValueError: could not convert string to float` | One of your numeric columns contains text values. Go back to Day 4's cleaning script and make sure all columns are properly converted to numbers. |
| Plot files are created but appear blank or all-white | Check that your DataFrame actually has data. Add `print(df.head())` after loading to verify. An empty DataFrame will produce blank plots. |

---

## Check Your Understanding

These are the kinds of questions an examiner will ask about your EDA. Practice answering them out loud before your defense.

### 1. "Describe the key features of your inflation series."

**Answer:** "The inflation series covers January 2000 to December 2024 — 300 monthly observations. The mean inflation rate is approximately 13.5% with a standard deviation of about 6.5 percentage points, indicating considerable variability. The minimum is around 3% (achieved during the 2006-2007 oil boom under single-digit inflation) and the maximum is approximately 35% (reached in late 2024).

The series has positive skewness, meaning the distribution has a long right tail — there are more periods of extremely high inflation than extremely low inflation. This is typical for developing economies.

Visually, the most prominent feature is the sustained acceleration beginning in 2020. Inflation climbed from about 12% in early 2020 to nearly 35% by end-2024, driven by COVID-19 supply chain disruptions, Naira depreciation (especially the June 2023 exchange rate unification), and rapid monetary expansion.

The series appears non-stationary in levels, with a clear upward trend in the latter half of the sample. This will need to be formally tested using ADF and KPSS tests before fitting any regression model."

### 2. "What does a high positive correlation between exchange_rate and inflation mean?"

**Answer:** "A high positive correlation between the exchange rate (NGN/USD) and inflation means that when the Naira depreciates (the exchange rate rises), inflation also tends to rise. This is consistent with the exchange rate pass-through mechanism: Nigeria is heavily import-dependent for fuel, food, and manufactured goods. When the Naira loses value, the Naira-denominated price of imports increases, which directly pushes up the Consumer Price Index.

However, I would caution that this correlation alone does not establish causation. Both variables may be driven by a common underlying factor — for example, expansionary monetary policy can simultaneously cause both Naira depreciation and domestic inflation. Additionally, the high correlation may be partly spurious, arising because both series have strong upward trends over the sample period. To establish a genuine causal relationship with appropriate dynamics and lags, we use the ARDL bounds testing approach and Granger causality tests in our econometric analysis."

### 3. "Why might M2 and exchange_rate both correlate with inflation?"

**Answer:** "M2 and exchange rate are both correlated with inflation because they represent two of the main transmission channels through which macroeconomic policy affects consumer prices in Nigeria.

M2 operates through the monetary channel. When the CBN expands the money supply (through deficit financing, Ways and Means advances to the federal government, or open market operations), there is more Naira chasing the same quantity of goods and services. This is the classic quantity theory of money: if money grows faster than output, prices rise.

The exchange rate operates through the external channel. When the Naira depreciates, everything Nigeria imports becomes more expensive. Since Nigeria imports refined petroleum products, vehicles, machinery, chemicals, and even a significant portion of its food, exchange rate depreciation feeds into consumer prices across many CPI categories.

These two channels are also connected to each other. Excessive money supply growth can cause Naira depreciation (more Naira relative to foreign currency pushes the exchange rate up), which in turn pushes up import prices and inflation. So M2, exchange rate, and inflation can form a mutually reinforcing cycle — which is part of the reason Nigeria experienced such rapid inflation in 2023-2024.

This interconnectedness is precisely why we use a Vector Autoregression (VAR) model, which treats all variables as jointly determined and allows each variable to depend on lagged values of all others, rather than assuming a simple one-way causal chain."

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| EDA script | `data_processing/eda.py` | Computes summary statistics, generates time series plots and correlation heatmap |
| Summary statistics table | `results/summary_statistics.csv` | Mean, std, min, max, skewness, kurtosis for all 4 variables |
| MPR time series plot | `results/ts_mpr.png` | Visual history of CBN policy rate decisions |
| Inflation time series plot | `results/ts_inflation.png` | Visual history of Nigerian inflation |
| Exchange rate time series plot | `results/ts_exchange_rate.png` | Visual history of Naira depreciation |
| M2 time series plot | `results/ts_m2.png` | Visual history of money supply expansion |
| Combined panel plot | `results/ts_panel_all_variables.png` | All 4 variables in one figure for comparison |
| Correlation heatmap | `results/correlation_matrix.png` | Pairwise correlation between all variables |

**Package installed today:** `matplotlib==3.8.2`

---

## What You Know After Week 1

Congratulations. You have completed the first week of the Nigerian Inflation Predictor project. Let us take stock of everything you have accomplished.

### The Project Structure is Set Up (Day 1)

You created a professional, organized project folder with separate directories for raw data, processed data, ingestion scripts, processing scripts, models, results, and documentation. You initialized git for version control. This structure is not arbitrary — it mirrors how real data science and economics research projects are organized.

### Your Environment is Ready (Day 2)

You created a virtual environment to isolate your project's dependencies, installed pandas for data manipulation, and set up a requirements.txt file so anyone can reproduce your exact environment. You understand why virtual environments matter and how pip works.

### Data is Loaded and Validated (Day 3)

You built an ingestion script that loads raw Nigerian macroeconomic data from CSV, validates the column names and data types, converts dates properly, and returns a clean DataFrame. You have 300 monthly observations covering January 2000 to December 2024.

### Data is Cleaned (Day 4)

You built a cleaning pipeline that handles missing values, enforces monthly frequency, validates value ranges, and saves the processed data to a new location. Your raw data is never modified — the cleaning step reads from `data/raw/` and writes to `data/processed/`.

### You Understand Your Data (Day 5 — Today)

You computed summary statistics and know the mean, standard deviation, range, skewness, and kurtosis of each variable. You created time series plots that reveal the trends, structural breaks, and regime changes in Nigerian macroeconomic data. You built a correlation matrix that shows how the variables relate to each other. Most importantly, you can now tell the **story** of this data:

> Nigeria's economy between 2000 and 2024 experienced multiple monetary policy regimes, two major exchange rate collapses (2016 and 2023), exponential money supply growth, and a sustained inflation crisis that accelerated dramatically after 2020. The data shows strong positive correlations between exchange rate depreciation, money supply growth, and inflation — consistent with both the quantity theory of money and exchange rate pass-through theory. However, these correlations may be partly driven by common trends, requiring formal stationarity and cointegration testing before any causal conclusions can be drawn.

### You Are Ready for Week 2

In Week 2, you will move from description to formal testing. Here is what is coming:

- **Day 6-7:** Stationarity testing with ADF and KPSS tests — you will formally determine whether each variable is stationary or has a unit root, and how many times you need to difference it.
- **Day 8-9:** Cointegration testing — you will test whether the non-stationary variables share a long-run equilibrium relationship.
- **Day 10:** Lag selection — you will determine the optimal number of lags for your econometric models.

The descriptive work you did this week is the foundation everything else builds on. If an examiner asks "why did you difference the exchange rate?" your answer starts with "as shown in the EDA, the exchange rate has a clear upward trend and structural breaks, suggesting non-stationarity..." You need the EDA to justify every modelling decision you make later.

Well done. See you in Week 2.
