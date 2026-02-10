# Week 1, Day 5 — Exploratory Data Analysis: Statistics & Plots, Built Step by Step

## What You Will Learn Today

- How to install matplotlib for creating plots
- How to compute summary statistics and what each one means for Nigerian economic data
- How to create time series plots that reveal trends and structural breaks
- How to build a correlation matrix heatmap and interpret pairwise relationships
- What your data tells you about the Nigerian economy before you build any model

## Why This Matters

Exploratory Data Analysis (EDA) is NOT optional. It is the single most important step between loading your data and building models. Every examiner, every thesis committee, every job interviewer will ask you the same question before they care about your model: **"Describe your data."**

If you cannot describe what your data looks like -- its central tendency, its spread, its trends, its anomalies -- then no amount of fancy modelling will save you. You must know your data better than anyone in the room.

EDA serves three critical purposes:

1. **It reveals structural breaks.** A time series plot will show you that Nigeria's official exchange rate did not depreciate gradually -- it jumped violently in June 2016 and again in June 2023. These jumps (structural breaks) affect which econometric models are appropriate. If you skip EDA, you will choose the wrong model.

2. **It reveals trends and non-stationarity.** If a variable trends upward over 20 years (like the exchange rate), it is not stationary. You need to know this before fitting any regression, because regressing two non-stationary series on each other can produce "spurious regression" -- results that look significant but are meaningless.

3. **It reveals outliers and data errors.** Maybe one month has an inflation value of 999 because of a data entry mistake. Maybe the Treasury Bill Rate is recorded in basis points instead of percentages for some period. Plots catch these problems instantly; staring at a spreadsheet does not.

Bottom line: EDA is where you build the story of your data. That story is what your thesis or report is really about. The models just quantify what you already see.

---

## The Four Variables in Your Dataset

Before we start coding, make sure you know what you are working with. Your cleaned dataset (`data/processed/cleaned_data.csv`) contains four columns of monthly data from the Central Bank of Nigeria (CBN):

| Column | Full Name | Unit | What It Measures |
|--------|-----------|------|------------------|
| `mpr` | Monetary Policy Rate | Percent (%) | The benchmark interest rate set by the CBN's Monetary Policy Committee (MPC). This is the CBN's primary tool for controlling inflation. |
| `infl` | Headline Inflation | Percent (%, year-on-year) | The annual rate of change in the Consumer Price Index, published by the National Bureau of Statistics (NBS). This is the variable you are trying to predict. |
| `exo` | Official Exchange Rate | Naira per US Dollar (NGN/USD) | The official CBN exchange rate for the Naira against the US Dollar. When this number goes up, the Naira is losing value. |
| `tbr` | Treasury Bill Rate | Percent (%) | The yield (interest rate) on Nigerian Treasury Bills at auction. This reflects short-term borrowing costs for the Federal Government and is closely linked to the MPR. |

---

## Part 1: Install matplotlib

matplotlib is Python's most widely used plotting library. It creates publication-quality charts, and it is the foundation that many other plotting libraries (like seaborn) are built on.

### Update requirements.txt

Delete everything in `requirements.txt` and replace it with this:

```
pandas==2.1.4
matplotlib==3.8.2
```

That is the complete file -- two lines, two packages.

### Install matplotlib

```bash
pip install matplotlib==3.8.2
```

**What you should see:** A lot of download and installation text ending with `Successfully installed matplotlib-3.8.2` (plus some dependencies like `contourpy`, `cycler`, `kiwisolver`, `pillow`, etc.). If matplotlib was already installed, you will see `Requirement already satisfied`.

### Verify the installation

```bash
python -c "import matplotlib; print(matplotlib.__version__)"
```

That should print `3.8.2`. If it does, you are ready to plot.

**Why we pin the version:** Writing `matplotlib==3.8.2` instead of just `matplotlib` ensures everyone who runs your project gets the exact same version. Different versions of matplotlib can produce slightly different-looking plots or even change function signatures. Pinning prevents the "it works on my machine" problem.

---

## Part 2: Building `data_processing/eda.py` -- Step by Step

We are NOT going to dump one giant script on you. Instead, we build the file piece by piece. You write a small chunk, run it, see the output, understand what it does, and THEN add the next piece. By the end you will have the complete file, but you will understand every line because you watched it come to life.

Make sure you have a `data_processing/` folder with an `__init__.py` inside it (from Day 4). If not, create them now.

At each step, we show you the **complete file from the first line to the last line**. Delete everything in `data_processing/eda.py` and replace it with exactly what is shown. No guessing where to put things.

---

### Build Step 1: Load Cleaned Data Only

Create a new file called `data_processing/eda.py`. Delete everything in `data_processing/eda.py` and replace it with this:

```python
"""Exploratory Data Analysis for the Nigerian Inflation Predictor."""
import os
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

def load_cleaned_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df

if __name__ == "__main__":
    df = load_cleaned_data()
    print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    print(f"\nColumns: {list(df.columns)}")
```

**Run it:**

```bash
python -m data_processing.eda
```

**What you should see:**

```
Loaded cleaned data: 300 rows, 4 columns
Period: 2000-01 to 2024-12

Columns: ['mpr', 'infl', 'exo', 'tbr']
```

(Your row count may differ slightly depending on your dataset. The key thing is that you see all four columns: `mpr`, `infl`, `exo`, `tbr`.)

**What just happened -- line by line:**

- `"""Exploratory Data Analysis..."""` -- This is a module docstring. It describes what this file does. It is the first thing another developer (or your future self) sees when they open the file.
- `PROCESSED_DIR` builds a path relative to where the script lives. `os.path.dirname(__file__)` means "the folder this script is in" (which is `data_processing/`). The `".."` goes up one level to the project root, then into `data/processed/`. This is where Day 4's cleaning script saved the cleaned CSV.
- `RESULTS_DIR` does the same thing but points to the `results/` folder, where we will save plots.
- `load_cleaned_data()` reads the CSV that Day 4 created. `index_col="date"` makes the date column the row index (standard for time series in pandas). `parse_dates=True` converts the date strings into actual datetime objects so matplotlib can put them on the x-axis properly.
- `if __name__ == "__main__":` -- This block only runs when you execute the script directly with `python -m data_processing.eda`. It does NOT run if another script imports functions from this file.
- `df.shape[0]` is the number of rows, `df.shape[1]` is the number of columns.
- `df.index[0].strftime('%Y-%m')` formats the first date as "2000-01" instead of the full datetime string.

If that ran and printed your column names, you are ready for the next piece.

---

### Build Step 2: Add the summary_statistics Function

Delete everything in `data_processing/eda.py` and replace it with this:

```python
"""Exploratory Data Analysis for the Nigerian Inflation Predictor."""
import os
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_cleaned_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def summary_statistics(df):
    """Compute and print descriptive statistics for all variables."""
    stats = df.describe().T
    stats["skewness"] = df.skew()
    stats["kurtosis"] = df.kurtosis()

    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(stats.round(2).to_string())
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stats.round(4).to_csv(os.path.join(RESULTS_DIR, "summary_statistics.csv"))
    print(f"\nSaved to results/summary_statistics.csv")
    return stats


if __name__ == "__main__":
    df = load_cleaned_data()
    print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    print()
    summary_statistics(df)
```

**Run it:**

```bash
python -m data_processing.eda
```

**What you should see:**

```
Loaded cleaned data: 300 rows, 4 columns
Period: 2000-01 to 2024-12

======================================================================
SUMMARY STATISTICS
======================================================================
      count   mean     std    min     25%     50%      75%      max  skewness  kurtosis
mpr   300.0  12.68    3.71   6.00   11.00   13.00    14.00    27.50      1.47      3.24
infl  300.0  13.48    6.52   3.00    8.65   11.61    17.10    34.80      1.32      1.22
exo   300.0 268.43  258.71  92.34  132.50  197.00   386.00  1680.00      2.35      6.18
tbr   300.0  10.25    4.38   1.04    5.75   10.50    13.85    26.00      0.31     -0.32
======================================================================

Saved to results/summary_statistics.csv
```

(Your exact numbers will depend on your dataset. The important thing is you see all four rows.)

**What each statistic means -- explained with Nigerian economic context:**

`df.describe()` computes, in one call: count (number of non-missing values), mean (average), std (standard deviation -- how spread out the data is), min, 25% (first quartile), 50% (median), 75% (third quartile), and max. The `.T` transposes the table so each variable is a row instead of a column, which makes it easier to read.

Here is what the numbers tell you about each variable:

**mpr (Monetary Policy Rate):**
- Mean around 12-14% reflects the CBN's generally tight monetary policy stance over the past two decades. Nigeria has kept rates relatively high compared to advanced economies because it faces persistent inflationary pressure.
- Standard deviation of roughly 3-4 percentage points shows moderate variation. The MPR does not move every month -- the MPC meets every two months and often holds the rate unchanged for long stretches.
- The range from about 6% (the 2009 crisis low) to 27.5% (the 2024 tightening peak) captures the full span of CBN policy actions. That 21.5 percentage-point range is enormous by global standards.

**infl (Headline Inflation):**
- Mean around 12-15% confirms that Nigeria has chronic moderate inflation. This is not hyperinflation (like Zimbabwe) but it is well above the CBN's implicit target of single digits.
- Positive skewness means the distribution has a long right tail -- there are more periods of extremely high inflation than extremely low inflation. This is typical for developing economies where supply shocks (food prices, fuel costs, exchange rate collapses) create occasional spikes.

**exo (Official Exchange Rate):**
- The huge range from roughly 90 to 1500+ NGN/USD is the most striking feature. The Naira has lost over 90% of its value against the dollar since 2000. This is not a subtle trend -- it is a dramatic collapse, mostly concentrated in a few step-devaluations.
- Very high positive skewness (above 2.0) confirms the distribution is heavily right-skewed: the Naira spent most of the sample period at lower exchange rates, with the extreme depreciation concentrated in recent years.
- High kurtosis (above 6.0) indicates fat tails -- extreme values are more common than a normal distribution would predict.

**tbr (Treasury Bill Rate):**
- Mean around 10% with a standard deviation of roughly 4 percentage points. The TBR generally follows the MPR but with a spread (premium) that varies with market conditions.
- The TBR range is typically slightly below the MPR, because Treasury Bills are considered risk-free instruments backed by the full faith of the Federal Government.
- Lower skewness than the other variables suggests a more symmetric distribution -- the TBR can go both very high and very low relative to its mean.

`df.skew()` measures how lopsided the distribution is. Positive skewness means the right tail is longer. `df.kurtosis()` measures how heavy the tails are compared to a normal distribution. Pandas uses "excess kurtosis," so a normal distribution gives 0.

You now have a CSV file in `results/` that you can paste into your thesis appendix. Move on.

---

### Build Step 3: Add the plot_time_series Function

Now we add the ability to plot each variable over time. This is the most visually informative part of EDA -- you will literally see the story of Nigeria's economy unfold.

Delete everything in `data_processing/eda.py` and replace it with this:

```python
"""Exploratory Data Analysis for the Nigerian Inflation Predictor."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_cleaned_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def summary_statistics(df):
    """Compute and print descriptive statistics for all variables."""
    stats = df.describe().T
    stats["skewness"] = df.skew()
    stats["kurtosis"] = df.kurtosis()

    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(stats.round(2).to_string())
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stats.round(4).to_csv(os.path.join(RESULTS_DIR, "summary_statistics.csv"))
    print(f"\nSaved to results/summary_statistics.csv")
    return stats


def plot_time_series(df):
    """Create a 2x2 panel of time series plots, one per variable."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    variables = {
        "mpr": ("Monetary Policy Rate (MPR)", "Percent (%)", "#1f77b4"),
        "infl": ("Headline Inflation (infl)", "Percent (%, YoY)", "#d62728"),
        "exo": ("Official Exchange Rate (exo)", "NGN per USD", "#2ca02c"),
        "tbr": ("Treasury Bill Rate (tbr)", "Percent (%)", "#ff7f0e"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (col, (title, ylabel, color)) in zip(axes, variables.items()):
        ax.plot(df.index, df[col], color=color, linewidth=1.2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Date", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Nigerian Macroeconomic Variables — Time Series",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, "time_series.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


if __name__ == "__main__":
    df = load_cleaned_data()
    print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    print()
    summary_statistics(df)
    print()
    plot_time_series(df)
```

**Run it:**

```bash
python -m data_processing.eda
```

**What you should see** (after the summary statistics output):

```
Saved: data_processing/../results/time_series.png
```

Now go open `results/time_series.png` and **look at it**. This is important. Do not skip this step. You should see a 2x2 grid with four panels.

**How the matplotlib code works:**

- `matplotlib.use("Agg")` tells matplotlib to save plots to files instead of trying to open a window on your screen. The "Agg" backend is non-interactive. This MUST come before `import matplotlib.pyplot as plt`. If you put them in the wrong order, you will get errors on servers or remote machines that have no display.
- `fig, axes = plt.subplots(2, 2, figsize=(14, 10))` creates a figure with a 2x2 grid of subplots. `figsize=(14, 10)` means 14 inches wide by 10 inches tall. This gives each panel enough space to be readable.
- `axes = axes.flatten()` converts the 2x2 array of axes into a flat list of 4 axes, so we can loop over them easily.
- `ax.plot(df.index, df[col], ...)` draws a line on each subplot. `df.index` is the dates (x-axis), `df[col]` is the values (y-axis).
- `ax.grid(True, alpha=0.3)` adds light gridlines. `alpha=0.3` means 30% opaque, so they do not compete with the data.
- `fig.suptitle(...)` adds an overall title above all four panels.
- `plt.tight_layout()` automatically adjusts spacing so labels and titles do not overlap.
- `fig.savefig(filepath, dpi=150, bbox_inches="tight")` saves the plot to a PNG file. `dpi=150` means 150 dots per inch, which is good for screen viewing and thesis printing. `bbox_inches="tight"` trims any extra whitespace around the edges.
- `plt.close(fig)` frees the memory used by this figure. Without this, creating many plots will consume more and more RAM.

**DETAILED INTERPRETATION -- What you should see in each panel:**

**Top-Left: MPR Panel**

You should see a **stepped pattern** that looks like a staircase, not a smooth curve. This is because the Monetary Policy Committee (MPC) meets every two months and either holds, raises, or cuts the rate by discrete amounts (usually 25, 50, or 100 basis points). Between meetings, the rate does not change.

Key features:
- 2000-2005 (13-15%): Relatively high rates during the early democratic period. The CBN was fighting inflation inherited from the military era.
- 2006-2007 (drop to ~10%): Rate cuts during the oil boom. Nigeria was running current account surpluses, foreign reserves were high.
- 2008-2009 (drop to 6%): The global financial crisis hit. The CBN slashed rates to stimulate the economy.
- 2011 (jump to 12%): Governor Sanusi hiked the MPR dramatically to defend the Naira and fight imported inflation.
- 2011-2019 (12-14%): A long period of relatively stable high rates.
- 2020 (cut to 11.5%): COVID-19 response.
- 2022-2024 (surge to 27.5%): The most aggressive tightening cycle in Nigerian history under Governor Cardoso.

**Top-Right: Inflation Panel**

You should see inflation cycling around 10-15% for most of the sample, with a dramatic surge above 30% in the final years.

Key features:
- 2000-2007 (volatile, 3-18%): Inflation bounced around considerably but averaged in the single digits during the 2006-2007 oil boom.
- 2008-2012 (10-16%): Elevated inflation following the global financial crisis and food price shocks.
- 2016-2017 (spike to ~18%): Inflation jumped after the CBN devalued the Naira in June 2016.
- 2020-2024 (acceleration from ~12% to ~35%): The defining feature of the series. COVID disrupted supply chains, the Naira collapsed, food prices surged, and the exchange rate unification in June 2023 sent inflation spiralling higher.

**Bottom-Left: Exchange Rate Panel**

This is probably the most dramatic panel. You should NOT see a gradual upward slope. Instead, you should see **long flat periods interrupted by sudden vertical jumps** -- step devaluations.

Key features:
- 2000-2008 (~92-132): Relatively stable. The CBN maintained a managed float, intervening heavily in the forex market.
- 2016 (jump from ~199 to ~310): The CBN partially floated the Naira after oil prices crashed. The official rate jumped overnight. This is a textbook structural break.
- 2023 (jump from ~462 to ~750): President Tinubu's government unified the exchange rate, ending the CBN's multiple exchange rate regime.
- 2024 (surge toward 1500+): Continued freefall as markets adjusted to the new floating regime.

This step-devaluation pattern means the exchange rate violates many econometric assumptions. It is not normally distributed. It is not stationary. It has structural breaks. Your models will need to account for this.

**Bottom-Right: TBR Panel**

The Treasury Bill Rate should visually co-move with the MPR but with more noise and volatility. This is because the TBR is market-determined at auction, while the MPR is an administrative rate set by the MPC.

Key features:
- The TBR generally tracks the MPR but with a spread. When the MPR is 13%, the TBR might be anywhere from 8% to 15% depending on market liquidity.
- The TBR is more volatile -- it can move in any auction cycle, not just at MPC meetings.
- In periods of excess liquidity (too much cash in the banking system), the TBR drops well below the MPR because banks have surplus funds to lend to the government.
- In tight liquidity conditions, the TBR rises closer to or above the MPR.

---

### Build Step 4: Add the plot_correlation_matrix Function

Now we add the correlation matrix heatmap. This shows how strongly each pair of variables moves together.

Delete everything in `data_processing/eda.py` and replace it with this:

```python
"""Exploratory Data Analysis for the Nigerian Inflation Predictor."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_cleaned_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def summary_statistics(df):
    """Compute and print descriptive statistics for all variables."""
    stats = df.describe().T
    stats["skewness"] = df.skew()
    stats["kurtosis"] = df.kurtosis()

    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(stats.round(2).to_string())
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stats.round(4).to_csv(os.path.join(RESULTS_DIR, "summary_statistics.csv"))
    print(f"\nSaved to results/summary_statistics.csv")
    return stats


def plot_time_series(df):
    """Create a 2x2 panel of time series plots, one per variable."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    variables = {
        "mpr": ("Monetary Policy Rate (MPR)", "Percent (%)", "#1f77b4"),
        "infl": ("Headline Inflation (infl)", "Percent (%, YoY)", "#d62728"),
        "exo": ("Official Exchange Rate (exo)", "NGN per USD", "#2ca02c"),
        "tbr": ("Treasury Bill Rate (tbr)", "Percent (%)", "#ff7f0e"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (col, (title, ylabel, color)) in zip(axes, variables.items()):
        ax.plot(df.index, df[col], color=color, linewidth=1.2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Date", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Nigerian Macroeconomic Variables — Time Series",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, "time_series.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_correlation_matrix(df):
    """Compute correlation matrix and plot as annotated heatmap."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    corr = df.corr()

    print("=" * 70)
    print("CORRELATION MATRIX")
    print("=" * 70)
    print(corr.round(3).to_string())
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(corr.columns, fontsize=11)

    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=text_color,
            )

    plt.colorbar(im, label="Pearson Correlation")
    ax.set_title("Correlation Matrix — Nigerian Macro Variables",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, "correlation_matrix.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")

    return corr


if __name__ == "__main__":
    df = load_cleaned_data()
    print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    print()
    summary_statistics(df)
    print()
    plot_time_series(df)
    print()
    plot_correlation_matrix(df)
```

**Run it:**

```bash
python -m data_processing.eda
```

**What you should see** (at the end, after the time series output):

```
======================================================================
CORRELATION MATRIX
======================================================================
       mpr   infl    exo    tbr
mpr  1.000  0.XXX  0.XXX  0.XXX
infl 0.XXX  1.000  0.XXX  0.XXX
exo  0.XXX  0.XXX  1.000  0.XXX
tbr  0.XXX  0.XXX  0.XXX  1.000
======================================================================
Saved: data_processing/../results/correlation_matrix.png
```

(The `0.XXX` values will be actual correlation numbers when you run it.)

Now open `results/correlation_matrix.png` and look at it. You should see a 4x4 coloured grid. Red cells mean strong positive correlation. Blue cells mean strong negative correlation. White cells mean no correlation. Each cell has its numeric value printed on it.

**How the correlation code works:**

- `df.corr()` computes the Pearson correlation coefficient between every pair of columns. The result is a square matrix where each cell shows how strongly two variables move together. The diagonal is always 1.0 (every variable is perfectly correlated with itself).
- `ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)` displays the matrix as a coloured grid. `cmap="RdBu_r"` means red for positive, blue for negative, white for near zero. `vmin=-1, vmax=1` fixes the colour scale so the meaning of each colour is always consistent.
- The nested `for i / for j` loop places the actual numbers inside each cell. `color="white" if abs(value) > 0.5 else "black"` makes the text readable: white text on dark (strongly correlated) cells, black text on light (weakly correlated) cells.
- `plt.colorbar(im, label="Pearson Correlation")` adds the colour legend on the right side so readers know what the colours mean.

**DETAILED INTERPRETATION -- Expected correlations and what they mean:**

**MPR and TBR: Strong positive correlation (likely 0.60 to 0.85)**

This is the strongest expected correlation and it makes perfect economic sense. The MPR is the CBN's benchmark rate, and Treasury Bill yields at auction are directly influenced by it. When the CBN raises the MPR, banks demand higher yields on Treasury Bills because their cost of funds has increased. This is called **policy transmission** -- the mechanism through which the CBN's interest rate decisions flow into the broader financial system.

If you see a correlation above 0.6 between MPR and TBR, that tells you monetary policy transmission is working in Nigeria. The CBN's rate decisions are reaching the money market.

**MPR and Inflation: Direction depends on lead/lag**

This is the trickiest one. You might expect a negative correlation (higher rates should reduce inflation), but the contemporaneous (same-month) correlation may actually be **positive**. Why? Because the CBN raises rates BECAUSE inflation is already high. In the data, you see high MPR and high inflation at the same time -- not because high rates cause high inflation, but because the CBN is reacting to inflation.

The causal effect (higher MPR eventually reduces inflation) takes time -- typically 6 to 18 months in Nigeria. This lag means a simple correlation cannot capture the true relationship. This is exactly why you need ARDL and VAR models in later weeks. Those models capture lagged, dynamic relationships that correlation cannot.

**Exchange Rate and Inflation: Positive (likely 0.50 to 0.80)**

When the Naira depreciates (exo goes up), imported goods become more expensive, which pushes up the Consumer Price Index. Nigeria imports a lot of what it consumes -- refined petroleum, machinery, food products, chemicals. So exchange rate depreciation feeds directly into consumer prices. Economists call this **exchange rate pass-through**.

The strength of this correlation tells you how much of Nigeria's inflation is imported versus domestically generated. A very high correlation (above 0.7) suggests that exchange rate movements are a dominant driver of inflation.

**Exchange Rate and TBR: May be weak (likely 0.10 to 0.40)**

The exchange rate and the Treasury Bill Rate operate through different channels. The exchange rate is driven by dollar supply and demand (oil revenue, foreign investment, remittances, import demand). The TBR is driven by domestic money market conditions (CBN policy, bank liquidity, government borrowing). These two channels can move independently.

However, there may be an indirect link: when the Naira is depreciating, the CBN sometimes raises rates to attract foreign capital and defend the currency, which would push up the TBR. The correlation depends on which effect dominates in your sample.

**IMPORTANT CAUTION:** Be very careful interpreting these correlations. Two variables that both trend upward over time will show a high positive correlation even if there is no causal relationship between them. For example, Nigeria's population also grew from 2000 to 2024 -- if you added population to your dataset, it would correlate strongly with the exchange rate, but that does not mean population growth causes Naira depreciation. This is the **spurious correlation** problem, and it is why you will need stationarity tests and cointegration analysis in Week 2.

---

### Build Step 5: Full Final File with Main Block That Runs Everything

This is the final version. Delete everything in `data_processing/eda.py` and replace it with this:

```python
"""Exploratory Data Analysis for the Nigerian Inflation Predictor."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_cleaned_data():
    """Load the cleaned dataset from data/processed/cleaned_data.csv."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def summary_statistics(df):
    """Compute and print descriptive statistics for all variables."""
    stats = df.describe().T
    stats["skewness"] = df.skew()
    stats["kurtosis"] = df.kurtosis()

    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(stats.round(2).to_string())
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stats.round(4).to_csv(os.path.join(RESULTS_DIR, "summary_statistics.csv"))
    print(f"\nSaved to results/summary_statistics.csv")
    return stats


def plot_time_series(df):
    """Create a 2x2 panel of time series plots, one per variable."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    variables = {
        "mpr": ("Monetary Policy Rate (MPR)", "Percent (%)", "#1f77b4"),
        "infl": ("Headline Inflation (infl)", "Percent (%, YoY)", "#d62728"),
        "exo": ("Official Exchange Rate (exo)", "NGN per USD", "#2ca02c"),
        "tbr": ("Treasury Bill Rate (tbr)", "Percent (%)", "#ff7f0e"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (col, (title, ylabel, color)) in zip(axes, variables.items()):
        ax.plot(df.index, df[col], color=color, linewidth=1.2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Date", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Nigerian Macroeconomic Variables — Time Series",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, "time_series.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_correlation_matrix(df):
    """Compute correlation matrix and plot as annotated heatmap."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    corr = df.corr()

    print("=" * 70)
    print("CORRELATION MATRIX")
    print("=" * 70)
    print(corr.round(3).to_string())
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(corr.columns, fontsize=11)

    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=text_color,
            )

    plt.colorbar(im, label="Pearson Correlation")
    ax.set_title("Correlation Matrix — Nigerian Macro Variables",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, "correlation_matrix.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")

    return corr


if __name__ == "__main__":
    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS — Nigerian Inflation Predictor")
    print("=" * 70)

    # Step 1: Load the cleaned data
    df = load_cleaned_data()
    print(f"\nLoaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    print(f"Columns: {list(df.columns)}")

    # Step 2: Summary statistics
    print()
    summary_statistics(df)

    # Step 3: Time series plots
    print()
    print("Generating time series plots...")
    plot_time_series(df)

    # Step 4: Correlation matrix
    print()
    print("Generating correlation matrix...")
    plot_correlation_matrix(df)

    # Done
    print()
    print("=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)
    print(f"All plots saved to the results/ folder.")
    print(f"  - results/time_series.png")
    print(f"  - results/correlation_matrix.png")
    print(f"  - results/summary_statistics.csv")
```

**Run it:**

```bash
python -m data_processing.eda
```

**What you should see:**

```
======================================================================
EXPLORATORY DATA ANALYSIS — Nigerian Inflation Predictor
======================================================================

Loaded cleaned data: 300 rows, 4 columns
Period: 2000-01 to 2024-12
Columns: ['mpr', 'infl', 'exo', 'tbr']

======================================================================
SUMMARY STATISTICS
======================================================================
      count   mean     std    min     25%     50%      75%      max  skewness  kurtosis
mpr   300.0  12.68    3.71   6.00   11.00   13.00    14.00    27.50      1.47      3.24
infl  300.0  13.48    6.52   3.00    8.65   11.61    17.10    34.80      1.32      1.22
exo   300.0 268.43  258.71  92.34  132.50  197.00   386.00  1680.00      2.35      6.18
tbr   300.0  10.25    4.38   1.04    5.75   10.50    13.85    26.00      0.31     -0.32
======================================================================

Saved to results/summary_statistics.csv

Generating time series plots...
Saved: data_processing/../results/time_series.png

Generating correlation matrix...
======================================================================
CORRELATION MATRIX
======================================================================
       mpr   infl    exo    tbr
mpr  1.000  0.XXX  0.XXX  0.XXX
infl 0.XXX  1.000  0.XXX  0.XXX
exo  0.XXX  0.XXX  1.000  0.XXX
tbr  0.XXX  0.XXX  0.XXX  1.000
======================================================================
Saved: data_processing/../results/correlation_matrix.png

======================================================================
EDA COMPLETE
======================================================================
All plots saved to the results/ folder.
  - results/time_series.png
  - results/correlation_matrix.png
  - results/summary_statistics.csv
```

(Your exact numbers will depend on your dataset. The `0.XXX` placeholders will be real correlation values.)

That is your complete, final `data_processing/eda.py` file. It loads data, computes statistics, creates a 2x2 time series panel, and generates an annotated correlation heatmap.

---

## Part 3: Commit Your Work

```bash
git add data_processing/eda.py requirements.txt
git commit -m "Day 5: Add EDA script with summary statistics, time series plots, and correlation matrix"
```

If you also want to commit the generated outputs:

```bash
git add results/summary_statistics.csv results/time_series.png results/correlation_matrix.png
git commit -m "Day 5: Add EDA output files (statistics CSV, plots)"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `FileNotFoundError` when loading cleaned data | You need to run the cleaning script first: `python -m data_processing.clean`. The EDA script reads from `data/processed/cleaned_data.csv`, which is created by Day 4. |
| `ModuleNotFoundError: No module named 'matplotlib'` | You forgot to install matplotlib. Run `pip install matplotlib==3.8.2` and try again. |
| `_tkinter.TclError: no display name` or `Tkinter not found` | This happens when matplotlib tries to open a plot window on a system without a display (like a remote server or WSL). The script handles this with `matplotlib.use("Agg")`, but make sure that line comes BEFORE `import matplotlib.pyplot as plt`. If you moved the imports around, put them back in the correct order. |
| `KeyError: 'mpr'` or `KeyError: 'infl'` | Your cleaned CSV does not have the expected column names. Open `data/processed/cleaned_data.csv` and check the header row. The columns must be exactly: `date`, `mpr`, `infl`, `exo`, `tbr`. If your columns have different names (like `inflation` or `exchange_rate`), go back to Day 3-4 and make sure the ingestion and cleaning scripts use the correct names. |
| `ValueError: could not convert string to float` | One of your numeric columns contains text values. Go back to Day 4 and make sure all columns are properly converted to numbers during cleaning. |
| Plot files are created but appear blank or all-white | Check that your DataFrame actually has data. Add `print(df.head())` after loading to verify. An empty DataFrame will produce blank plots. Also check that the column names match exactly. |
| `UserWarning: FigureCanvasAgg is non-interactive` | This is a warning, not an error. It just means matplotlib is using the Agg backend (which is what we want). Your plots are still saved correctly. You can safely ignore this message. |
| Plots look squished or labels are cut off | Make sure `plt.tight_layout()` is called before `fig.savefig()`. If labels are still cut off, try increasing `figsize` (for example, change `figsize=(14, 10)` to `figsize=(16, 12)`). |

---

## Check Your Understanding

These are the kinds of questions an examiner will ask about your EDA. Practice answering them out loud before your defense.

### 1. "Describe the key features of your inflation series."

**Answer:** "The inflation series (infl) covers monthly observations of Nigeria's headline year-on-year inflation rate. The mean is approximately 13-15%, with a standard deviation of about 6-7 percentage points, indicating considerable variability. The series has positive skewness, meaning the distribution has a long right tail -- there are more periods of extremely high inflation than extremely low inflation. This is typical for developing economies where supply shocks create occasional spikes.

Visually, the most prominent feature is the sustained acceleration beginning in 2020. Inflation climbed from about 12% to nearly 35%, driven by COVID-19 supply chain disruptions, Naira depreciation (especially the June 2023 exchange rate unification), and monetary expansion. The series appears non-stationary in levels, with a clear upward trend in the latter half of the sample. This will need to be formally tested using ADF and KPSS tests before fitting any regression model."

### 2. "Why might the correlation between MPR and inflation be positive even though higher rates are supposed to reduce inflation?"

**Answer:** "The correlation between MPR and inflation is contemporaneous -- it measures the same-month relationship. The CBN raises the MPR in response to inflation that is already high. So in the data, you see high MPR and high inflation at the same time -- not because high rates cause inflation, but because the CBN is reacting to inflation that already exists.

The causal effect of monetary tightening (higher MPR reducing inflation) operates with a lag, typically 6 to 18 months in Nigeria. This lagged effect is invisible in a simple Pearson correlation. To capture this dynamic, we use the ARDL model and Granger causality tests, which explicitly model how past values of MPR affect current inflation with appropriate time delays."

### 3. "What does the exchange rate plot tell you about modelling challenges?"

**Answer:** "The exchange rate series (exo) shows step-devaluations rather than gradual movements -- the Naira was held at managed levels for years, then jumped dramatically in 2016 and again in 2023. These step-changes are structural breaks that violate the assumption of a smooth, continuous process.

This has three modelling implications. First, the series is clearly non-stationary and will require differencing. Second, the structural breaks may affect unit root test results -- the ADF test can fail to reject a unit root in the presence of structural breaks. Third, any model that assumes a stable linear relationship between the exchange rate and inflation may perform poorly across these break points, because the exchange rate pass-through may differ before and after a devaluation."

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| EDA script | `data_processing/eda.py` | Computes summary statistics, generates time series plots and correlation heatmap |
| Summary statistics table | `results/summary_statistics.csv` | Mean, std, min, max, skewness, kurtosis for mpr, infl, exo, tbr |
| Time series panel plot | `results/time_series.png` | 2x2 grid showing each variable over time, revealing trends and breaks |
| Correlation heatmap | `results/correlation_matrix.png` | Annotated heatmap of pairwise Pearson correlations |

**Package installed today:** `matplotlib==3.8.2`

**Tomorrow (Day 6):** You will begin Week 2 by testing each variable for stationarity using the Augmented Dickey-Fuller (ADF) and KPSS tests. This is where you formally determine whether each series has a unit root and how many times you need to difference it before it can be used in regression models. The EDA you did today -- especially the time series plots showing trends and structural breaks -- will help you interpret those test results.
