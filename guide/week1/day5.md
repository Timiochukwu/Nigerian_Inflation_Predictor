# Week 1, Day 5 — Exploratory Data Analysis: Summary Statistics & Time Series Plots

## What You'll Learn Today

- How to install and use matplotlib for plotting
- How to compute and interpret summary statistics (mean, std, min, max, skewness, kurtosis)
- How to create time series plots for each of the 4 variables
- How to create a correlation matrix and understand what it means
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

Update your `requirements.txt` to include it:

```
pandas==2.1.4
matplotlib==3.8.2
```

Then install it:

```bash
pip install matplotlib==3.8.2
```

**What you should see:** A lot of download and installation text ending with `Successfully installed matplotlib-3.8.2` (plus some dependencies like `contourpy`, `cycler`, `kiwisolver`, `pillow`, etc.).

**Why we pin the version:** Writing `matplotlib==3.8.2` instead of just `matplotlib` ensures everyone who runs your project gets the same version. This prevents the "it works on my machine" problem.

---

## Full Code

Create this file. If you already have a `data_processing/` folder with an `__init__.py` inside it (from Day 4), you are ready. If not, make sure the folder exists and contains an empty `__init__.py`.

**File: `data_processing/eda.py`**

```python
"""
Exploratory Data Analysis (EDA) for the Nigerian Inflation Predictor.

This script computes summary statistics, generates time series plots for
each macroeconomic variable, and produces a correlation matrix heatmap.
All outputs are saved to the results/ folder.

Usage:
    python -m data_processing.eda
"""

import os

import pandas as pd

# matplotlib.use("Agg") MUST come before importing pyplot.
# "Agg" is a non-interactive backend — it renders plots to files
# instead of trying to open a window. This is important because:
# 1. On servers or remote machines, there is no display to open a window on.
# 2. It avoids the "Tkinter not found" error on some systems.
# 3. It makes the script work in automated pipelines.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
# os.path.dirname(__file__) gives the folder this script lives in
# (data_processing/). We go up one level (..) to reach the project root,
# then into the target subfolder.

# Where cleaned data lives (output of Day 4's cleaning script)
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Where we save all results — plots, tables, statistics
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------------------
# Colour palette for consistent plotting
# ---------------------------------------------------------------------------
# Using distinct colours makes each variable instantly recognizable
# across all plots in your thesis or report.
COLORS = {
    "mpr": "#1f77b4",            # Blue — policy rate, cool/authoritative
    "inflation": "#d62728",      # Red — inflation, urgency/heat
    "exchange_rate": "#2ca02c",  # Green — exchange rate, money/currency
    "m2": "#9467bd",             # Purple — money supply, complex/abstract
}

# Human-readable labels for plot titles and axes
LABELS = {
    "mpr": "Monetary Policy Rate (%)",
    "inflation": "Inflation Rate (%)",
    "exchange_rate": "Exchange Rate (NGN/USD)",
    "m2": "Broad Money Supply M2 (NGN Billion)",
}


def load_cleaned_data():
    """
    Load the cleaned dataset from data/processed/cleaned_data.csv.

    This function reads the CSV file that was created by the cleaning
    script (Day 4). It expects:
    - A 'date' column that can be parsed as datetime
    - Numeric columns: mpr, inflation, exchange_rate, m2

    Returns
    -------
    pd.DataFrame
        DataFrame with a DatetimeIndex and four numeric columns.

    Raises
    ------
    FileNotFoundError
        If cleaned_data.csv does not exist.
    """

    # Build the full path to the cleaned data file
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")

    # Check the file exists before trying to read it
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Cleaned data not found at: {filepath}\n"
            f"Have you run the cleaning script first?\n"
            f"  python -m data_processing.clean"
        )

    # Read the CSV. parse_dates=["date"] tells pandas to convert the
    # date column from text to actual datetime objects automatically.
    # index_col="date" makes the date column the row index, which is
    # standard for time series data.
    df = pd.read_csv(filepath, parse_dates=["date"], index_col="date")

    # Print a quick confirmation so you know the load worked
    print(f"Loaded cleaned data: {len(df)} observations")
    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to "
          f"{df.index.max().strftime('%Y-%m-%d')}")
    print(f"Columns: {list(df.columns)}")

    return df


def summary_statistics(df):
    """
    Compute and display summary statistics for all variables.

    This function calculates:
    - describe(): count, mean, std, min, 25%, 50%, 75%, max
    - skewness: how asymmetric the distribution is
      (positive = right-skewed, negative = left-skewed)
    - kurtosis: how heavy the tails are compared to a normal distribution
      (>0 = heavier tails than normal, <0 = lighter tails)

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned data with numeric columns.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing all summary statistics.
    """

    # df.describe() computes the standard summary statistics in one call:
    # count — number of non-missing observations
    # mean  — average value
    # std   — standard deviation (spread around the mean)
    # min   — smallest value
    # 25%   — first quartile (25% of values are below this)
    # 50%   — median (middle value)
    # 75%   — third quartile (75% of values are below this)
    # max   — largest value
    stats = df.describe()

    # .T transposes the table (swaps rows and columns) so that
    # each variable is a row — this is easier to read when printed.
    # Without .T, each variable would be a column, which gets too wide.
    stats = stats.T

    # Add skewness as a new column.
    # Skewness measures how lopsided the distribution is:
    #   skew > 0 means the right tail is longer (most values are low)
    #   skew < 0 means the left tail is longer (most values are high)
    #   skew = 0 means the distribution is symmetric (like a bell curve)
    # For economic data, positive skew often means a few extreme high values
    # are pulling the mean above the median.
    stats["skewness"] = df.skew()

    # Add kurtosis as a new column.
    # Kurtosis measures how "peaked" or "fat-tailed" the distribution is
    # compared to a normal distribution:
    #   kurtosis > 0 means heavier tails (more extreme values than normal)
    #   kurtosis < 0 means lighter tails (fewer extreme values than normal)
    #   kurtosis = 0 means approximately normal tails
    # pandas uses "excess kurtosis" by default, so a normal distribution = 0.
    stats["kurtosis"] = df.kurtosis()

    # Print the results in a readable format
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # .round(2) limits everything to 2 decimal places for readability.
    # .to_string() converts the DataFrame to a nicely formatted string.
    print(stats.round(2).to_string())
    print("=" * 70)

    # Save to CSV so you can include it in your thesis appendix
    # or load it later without re-running the script.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "summary_statistics.csv")
    stats.round(4).to_csv(output_path)
    print(f"\nSaved summary statistics to: {output_path}")

    return stats


def plot_time_series(df):
    """
    Create time series plots for each variable and a combined panel plot.

    This function produces:
    1. One individual plot per variable (4 plots total)
    2. One combined 2x2 panel plot with all 4 variables

    Each plot shows the variable's value on the y-axis and the date on
    the x-axis, with a title, axis labels, and grid lines.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned data with a DatetimeIndex and numeric columns.
    """

    # Make sure the results folder exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # List of variables to plot — these must match your DataFrame columns
    variables = ["mpr", "inflation", "exchange_rate", "m2"]

    # ------------------------------------------------------------------
    # Part 1: Individual plots for each variable
    # ------------------------------------------------------------------
    # We loop through each variable and create a separate plot file.
    # This is useful when you want to include a single variable's plot
    # in a specific section of your thesis.

    for var in variables:
        # plt.figure() creates a new blank canvas for the plot.
        # figsize=(12, 5) sets the width to 12 inches and height to 5 inches.
        # This wide aspect ratio works well for time series because time
        # goes left-to-right and you want to see the full date range clearly.
        fig, ax = plt.subplots(figsize=(12, 5))

        # ax.plot() draws a line on the axes.
        # df.index is the date (x-axis), df[var] is the values (y-axis).
        # color= sets the line colour using our predefined palette.
        # linewidth=1.2 makes the line slightly thicker than the default (1.0)
        # for better visibility in printed documents.
        ax.plot(df.index, df[var], color=COLORS[var], linewidth=1.2)

        # ax.set_title() adds a title above the plot.
        # fontsize=14 is large enough to read in a thesis.
        # fontweight="bold" makes it stand out.
        ax.set_title(f"{LABELS[var]} — Nigeria (2000-2024)",
                      fontsize=14, fontweight="bold")

        # ax.set_xlabel() labels the horizontal axis
        ax.set_xlabel("Date", fontsize=12)

        # ax.set_ylabel() labels the vertical axis using our human-readable label
        ax.set_ylabel(LABELS[var], fontsize=12)

        # ax.grid(True, alpha=0.3) adds light gridlines behind the data.
        # alpha=0.3 makes them 30% opaque (very subtle) so they don't
        # compete with the actual data line.
        ax.grid(True, alpha=0.3)

        # plt.tight_layout() automatically adjusts spacing so nothing
        # gets cut off (titles, axis labels, etc.).
        plt.tight_layout()

        # Save the plot as a PNG image file.
        # dpi=150 means 150 dots per inch — good enough for on-screen
        # viewing and thesis printing. (300 dpi would be publication quality.)
        output_path = os.path.join(RESULTS_DIR, f"ts_{var}.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

        # plt.close(fig) frees the memory used by this figure.
        # Without this, creating many plots in a loop will use more
        # and more RAM until Python crashes.
        plt.close(fig)

        print(f"Saved: {output_path}")

    # ------------------------------------------------------------------
    # Part 2: Combined 2x2 panel plot
    # ------------------------------------------------------------------
    # A panel plot puts all 4 variables side by side so you can compare
    # their trends visually. This is one of the most useful plots in
    # any time series thesis — examiners love it because it gives a
    # full overview of your dataset in a single figure.

    # plt.subplots(2, 2) creates a 2-row, 2-column grid of subplots.
    # fig is the overall figure, axes is a 2x2 array of individual axes.
    # figsize=(14, 10) makes the panel plot larger to fit 4 charts comfortably.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # axes.flatten() turns the 2x2 array into a flat list of 4 axes,
    # so we can loop through them alongside our variable list.
    axes_flat = axes.flatten()

    # zip() pairs each variable name with its corresponding axes object:
    #   ("mpr", axes_flat[0]), ("inflation", axes_flat[1]), etc.
    for var, ax in zip(variables, axes_flat):
        # Plot the data on this subplot
        ax.plot(df.index, df[var], color=COLORS[var], linewidth=1.0)

        # Add title and labels to this subplot
        ax.set_title(LABELS[var], fontsize=12, fontweight="bold")
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel(LABELS[var], fontsize=10)

        # Add subtle gridlines
        ax.grid(True, alpha=0.3)

        # Rotate x-axis date labels by 45 degrees so they don't overlap
        ax.tick_params(axis="x", rotation=45)

    # Add an overall title for the entire panel figure.
    # y=1.02 pushes the title slightly above the top of the figure
    # so it doesn't overlap with the subplot titles.
    fig.suptitle("Nigerian Macroeconomic Variables (2000-2024)",
                 fontsize=16, fontweight="bold", y=1.02)

    # plt.tight_layout() adjusts the spacing between subplots so they
    # don't overlap with each other.
    plt.tight_layout()

    # Save the combined panel plot
    output_path = os.path.join(RESULTS_DIR, "ts_panel_all_variables.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_correlation_matrix(df):
    """
    Compute and plot the correlation matrix as a heatmap.

    A correlation matrix shows how strongly each pair of variables
    moves together:
    - Correlation near +1.0 means they move in the same direction
    - Correlation near -1.0 means they move in opposite directions
    - Correlation near 0.0 means they have no linear relationship

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned data with numeric columns.

    Returns
    -------
    pd.DataFrame
        The correlation matrix.
    """

    # df.corr() computes the Pearson correlation coefficient between
    # every pair of columns. The result is a square matrix where
    # row i, column j contains the correlation between variable i and j.
    # The diagonal is always 1.0 (every variable is perfectly correlated
    # with itself).
    corr = df.corr()

    # Print the correlation matrix to the console
    print("\n" + "=" * 70)
    print("CORRELATION MATRIX")
    print("=" * 70)
    print(corr.round(4).to_string())
    print("=" * 70)

    # ------------------------------------------------------------------
    # Create the heatmap using matplotlib's imshow()
    # ------------------------------------------------------------------
    # imshow() displays a 2D array as a coloured grid. Each cell's
    # colour represents the correlation value.

    # Create a new figure. figsize=(8, 6) is a good size for a
    # 4x4 correlation matrix — not too cramped, not too spread out.
    fig, ax = plt.subplots(figsize=(8, 6))

    # ax.imshow() renders the correlation matrix as a colour grid.
    # cmap="RdBu_r" uses a red-blue colour scale:
    #   Red = positive correlation (variables move together)
    #   Blue = negative correlation (variables move opposite)
    #   White = near zero (no linear relationship)
    # vmin=-1, vmax=1 fixes the colour scale to the full correlation
    # range, so the colours are always meaningful regardless of the
    # actual values in your data.
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

    # Get the variable names for labelling the axes
    labels = [LABELS.get(col, col) for col in corr.columns]

    # Short labels for the axis ticks (full labels are too long)
    short_labels = ["MPR", "Inflation", "Exch. Rate", "M2"]

    # Set the tick positions and labels for both axes.
    # range(len(corr)) gives [0, 1, 2, 3] — the position of each cell.
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(short_labels, fontsize=11, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(short_labels, fontsize=11)

    # ------------------------------------------------------------------
    # Add text labels inside each cell
    # ------------------------------------------------------------------
    # This is important because colour alone can be hard to read precisely.
    # Printing the actual number inside each cell makes the values unambiguous.

    for i in range(len(corr)):
        for j in range(len(corr)):
            # Get the correlation value for this cell
            value = corr.iloc[i, j]

            # Choose text colour: white text on dark backgrounds,
            # black text on light backgrounds. If the absolute correlation
            # is above 0.7, the cell colour will be dark (deep red or
            # deep blue), so we use white text. Otherwise, black.
            text_color = "white" if abs(value) > 0.7 else "black"

            # ax.text() places text at position (j, i) — note that
            # imshow uses (column, row) for x, y coordinates.
            # f"{value:.2f}" formats the number to 2 decimal places.
            # ha="center", va="center" centres the text in the cell.
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

    # Add a colour bar on the right side that explains what the colours mean
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson Correlation", fontsize=11)

    # Add a title
    ax.set_title("Correlation Matrix — Nigerian Macro Variables",
                  fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()

    # Save the heatmap
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "correlation_matrix.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved correlation matrix to: {output_path}")

    return corr


# ---------------------------------------------------------------------------
# Main execution block
# ---------------------------------------------------------------------------
# This runs only when you execute the script directly:
#   python -m data_processing.eda
# It does NOT run when another script imports functions from this file.

if __name__ == "__main__":
    print("=" * 70)
    print("NIGERIAN INFLATION PREDICTOR — EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # Step 1: Load the cleaned data from Day 4's output
    df = load_cleaned_data()

    # Step 2: Compute and display summary statistics
    stats = summary_statistics(df)

    # Step 3: Create time series plots (individual + panel)
    print("\nGenerating time series plots...")
    plot_time_series(df)

    # Step 4: Compute and plot the correlation matrix
    print("\nComputing correlation matrix...")
    corr = plot_correlation_matrix(df)

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to the results/ folder.")
    print("Files created:")
    print("  - results/summary_statistics.csv")
    print("  - results/ts_mpr.png")
    print("  - results/ts_inflation.png")
    print("  - results/ts_exchange_rate.png")
    print("  - results/ts_m2.png")
    print("  - results/ts_panel_all_variables.png")
    print("  - results/correlation_matrix.png")
```

---

## Step 2: Run the EDA Script

Make sure you are in the project root directory (`Nigerian_Inflation_Predictor/`), then run:

```bash
python -m data_processing.eda
```

The `-m` flag tells Python to run `data_processing/eda.py` as a module. This is the same pattern we used for ingestion and cleaning.

---

## Expected Output

When you run the script, you should see something like this printed in your terminal:

```
======================================================================
NIGERIAN INFLATION PREDICTOR — EXPLORATORY DATA ANALYSIS
======================================================================
Loaded cleaned data: 300 observations
Date range: 2000-01-01 to 2024-12-01
Columns: ['mpr', 'inflation', 'exchange_rate', 'm2']

======================================================================
SUMMARY STATISTICS
======================================================================
               count    mean       std    min     25%      50%       75%        max  skewness  kurtosis
mpr            300.0   12.68     3.71    6.00   11.00    13.00     14.00      27.50      1.47      3.24
inflation      300.0   13.48     6.52    3.00    8.65    11.61     17.10      34.80      1.32      1.22
exchange_rate  300.0   268.43   258.71   92.34  132.50   197.00    386.00   1680.00      2.35      6.18
m2             300.0  45218.37 62845.20 1070.50 3195.30 10140.80  62505.80 284975.80     1.70      2.26
======================================================================

Saved summary statistics to: results/summary_statistics.csv

Generating time series plots...
Saved: results/ts_mpr.png
Saved: results/ts_inflation.png
Saved: results/ts_exchange_rate.png
Saved: results/ts_m2.png
Saved: results/ts_panel_all_variables.png

Computing correlation matrix...

======================================================================
CORRELATION MATRIX
======================================================================
                  mpr  inflation  exchange_rate      m2
mpr            1.0000     0.XXXX         0.XXXX  0.XXXX
inflation      0.XXXX     1.0000         0.XXXX  0.XXXX
exchange_rate  0.XXXX     0.XXXX         1.0000  0.XXXX
m2             0.XXXX     0.XXXX         0.XXXX  1.0000
======================================================================

Saved correlation matrix to: results/correlation_matrix.png

======================================================================
EDA COMPLETE
======================================================================

All results saved to the results/ folder.
Files created:
  - results/summary_statistics.csv
  - results/ts_mpr.png
  - results/ts_inflation.png
  - results/ts_exchange_rate.png
  - results/ts_m2.png
  - results/ts_panel_all_variables.png
  - results/correlation_matrix.png
```

(The `0.XXXX` values will be replaced with actual correlation numbers when you run it on your data.)

**Files saved to `results/`:**

| File | Description |
|------|-------------|
| `summary_statistics.csv` | Table of all summary stats — embed this in your thesis appendix |
| `ts_mpr.png` | Time series plot of the Monetary Policy Rate |
| `ts_inflation.png` | Time series plot of the Inflation Rate |
| `ts_exchange_rate.png` | Time series plot of the Exchange Rate |
| `ts_m2.png` | Time series plot of Broad Money Supply (M2) |
| `ts_panel_all_variables.png` | Combined 2x2 panel plot of all 4 variables |
| `correlation_matrix.png` | Heatmap showing pairwise correlations |

---

## Interpreting Your Results

This is the most important section of today's guide. The numbers and plots are useless unless you know what they mean. Let us walk through each one.

### What the MPR Plot Shows

The Monetary Policy Rate (MPR) is the interest rate set by the Central Bank of Nigeria (CBN). It is the price of borrowing money in Nigeria, and it is the CBN's primary tool for controlling inflation.

When you look at the MPR plot, you should see **distinct policy regimes** — periods where the rate was held steady, followed by sudden jumps or drops. This is because the CBN does not adjust the MPR smoothly. The Monetary Policy Committee (MPC) meets every two months and decides to hold, raise, or cut the rate.

Key features you should see:

- **2000-2005 (13-15%):** Relatively high rates during the early democratic period. The CBN was fighting inflation inherited from the military era.
- **2006-2007 (~10%):** The CBN under Soludo cut rates aggressively during the oil boom. Nigeria was running current account surpluses, foreign reserves were high, and inflation was low. This was the "good times" era.
- **2008-2009 (drop to 6%):** The global financial crisis hit. The CBN slashed rates to stimulate the economy, mirroring what central banks worldwide were doing.
- **2011 (jump to 12%):** Sanusi Lamido Sanusi hiked the MPR dramatically to defend the Naira and fight imported inflation. This is one of the most famous CBN policy moves.
- **2011-2019 (12-14%):** A long period of relatively stable high rates. The CBN kept rates elevated even when growth was weak.
- **2020 (cut to 11.5%):** COVID-19 response. The CBN cut rates to support the economy during lockdowns.
- **2022-2024 (surge to 27.5%):** The most aggressive tightening cycle in Nigerian history. The CBN under Cardoso raised rates from 11.5% to 27.5% as inflation spiralled out of control.

**What to tell your examiner:** "The MPR series shows distinct policy regimes with structural breaks, suggesting that a model allowing for regime changes or structural breaks may be more appropriate than one assuming a stable relationship."

### What the Inflation Plot Shows

The inflation rate measures how fast prices are rising. It is the variable you are trying to predict.

Key features you should see:

- **2000-2007 (volatile, 3-18%):** Inflation bounced around considerably but averaged in the single digits during the oil boom years of 2006-2007.
- **2008-2012 (10-16%):** Elevated inflation following the global financial crisis and food price shocks.
- **2013-2019 (8-18%):** A wide range. Inflation spiked in 2016-2017 after the Naira was devalued, then gradually came back down as the exchange rate stabilized.
- **2020-2024 (acceleration from 12% to 35%):** This is the defining feature of the series. Starting in 2020, inflation began a relentless climb. COVID disrupted supply chains, the Naira collapsed, food prices surged, and money supply grew rapidly. By late 2024, inflation was near 35% — the highest in nearly two decades.

**What to tell your examiner:** "The inflation series shows a clear upward trend since 2020, with acceleration in 2023-2024 coinciding with exchange rate unification and Naira depreciation. The series appears non-stationary in levels, which we will formally test in Week 2."

### What the Exchange Rate Plot Shows

The exchange rate (NGN/USD) shows how many Naira you need to buy one US dollar. When this number goes up, the Naira is losing value (depreciating).

This is probably the most dramatic plot in your dataset:

- **2000-2008 (~92-132 NGN/USD):** Relatively stable. The CBN maintained a managed float, intervening heavily in the foreign exchange market to keep the Naira steady.
- **2008-2009 (jump to 150):** The global financial crisis reduced oil revenue, and the CBN could no longer defend the old rate. The Naira was devalued.
- **2009-2015 (~150-197):** Another period of managed stability. The CBN burned through foreign reserves to keep the rate around 150-197.
- **2016 (jump from 199 to 310):** The CBN finally allowed the Naira to float (partially) after oil prices crashed in 2015. The official rate jumped overnight. This is a textbook structural break.
- **2017-2019 (~360):** The Naira settled at a new managed rate around 360, held in place by CBN intervention.
- **2020-early 2023 (~360-462):** Gradual depreciation as COVID and falling oil prices depleted reserves.
- **June 2023 (jump from 462 to 750):** President Tinubu's government unified the exchange rate, ending the CBN's multiple exchange rate regime. The Naira instantly lost nearly 40% of its value.
- **2024 (surge to 1,500-1,680):** Continued freefall. The Naira depreciated further as markets adjusted to the new floating regime and foreign investment remained cautious.

**What to tell your examiner:** "The exchange rate series contains at least three major structural breaks — 2016, June 2023, and early 2024. These step-changes violate the assumption of a smooth, continuous process. Any econometric model must account for these breaks, either through dummy variables, structural break tests, or by working with differenced data."

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

- **exchange_rate and m2 (very high positive, likely 0.95+):** These two variables have grown together over the 25-year period. As the money supply expanded, the Naira depreciated. This makes economic sense: more Naira chasing the same amount of foreign currency pushes the exchange rate up. However, be cautious — this high correlation is partly because both series have strong upward trends. Two trending series will always appear correlated even if there is no causal link. This is the "spurious regression" problem you will address in Week 2 with stationarity tests.

- **exchange_rate and inflation (high positive, likely 0.70-0.85):** When the Naira loses value, imported goods become more expensive, pushing up inflation. Nigeria imports a lot of what it consumes — refined petroleum, machinery, food products, raw materials. So exchange rate depreciation feeds directly into consumer prices. Economists call this "exchange rate pass-through."

- **m2 and inflation (moderate to high positive):** Money supply growth and inflation tend to move together, consistent with the quantity theory of money. More money in the economy pushes prices up.

- **mpr and inflation (moderate positive):** This might seem counterintuitive. If the CBN raises rates to fight inflation, shouldn't the correlation be negative? Not necessarily. The CBN raises the MPR *in response to* rising inflation. So in the data, you see high MPR and high inflation at the same time — because the MPR was raised to fight the inflation that was already happening. The causal effect (higher MPR reduces inflation) takes time (lags) and is not visible in a simple correlation. This is exactly why you need the ARDL and VAR models in later weeks — they capture the lagged, dynamic relationships that correlation cannot.

**What to tell your examiner:** "The correlation matrix shows strong positive associations between exchange rate, money supply, and inflation, consistent with economic theory. However, these correlations may be driven by common trends rather than causal relationships. We address this by testing for stationarity and using cointegration analysis in subsequent sections."

---

## Step 3: Commit

```bash
git add data_processing/eda.py requirements.txt
git commit -m "Day 5: Add EDA script with summary statistics, time series plots, and correlation matrix"
```

---

## Packages Installed Today

| Package | Version | Purpose |
|---------|---------|---------|
| `matplotlib` | `3.8.2` | Creating all plots — time series charts, heatmaps, panel figures |

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: Cleaned data not found` | You need to run the cleaning script first: `python -m data_processing.clean`. The EDA script reads from `data/processed/cleaned_data.csv`, which is created by the Day 4 cleaning step. |
| `ModuleNotFoundError: No module named 'matplotlib'` | You forgot to install matplotlib. Run `pip install matplotlib==3.8.2` and try again. |
| `_tkinter.TclError: no display name` or `Tkinter not found` | This happens when matplotlib tries to open a plot window on a system without a display (like a remote server or WSL). The script already handles this with `matplotlib.use("Agg")`, but make sure that line comes BEFORE `import matplotlib.pyplot as plt`. If you moved the imports around, put them back in the correct order. |
| `KeyError: 'mpr'` or `KeyError: 'inflation'` | Your cleaned CSV does not have the expected column names. Open `data/processed/cleaned_data.csv` and check the header row. The columns must be exactly: `date`, `mpr`, `inflation`, `exchange_rate`, `m2`. |
| `ValueError: could not convert string to float` | One of your numeric columns contains text values. Go back to Day 4's cleaning script to make sure all columns are properly converted to numbers. |
| Plot files are created but appear blank or all-white | Check that your DataFrame actually has data. Add `print(df.head())` after loading to verify. An empty DataFrame will produce blank plots. |

---

## Check Your Understanding

These are the kinds of questions an examiner will ask about your EDA. Practice answering them out loud before your defense.

### 1. "Describe the key features of your inflation series."

**Answer:** "The inflation series covers January 2000 to December 2024 — 300 monthly observations. The mean inflation rate is approximately 13.5% with a standard deviation of about 6.5 percentage points, indicating considerable variability. The minimum is around 3% (achieved during the 2006-2007 oil boom under single-digit inflation) and the maximum is approximately 35% (reached in late 2024).

The series has positive skewness, meaning the distribution has a long right tail — there are more periods of extremely high inflation than extremely low inflation. This is typical for developing economies.

Visually, the most prominent feature is the sustained acceleration beginning in 2020. Inflation climbed from about 12% in early 2020 to nearly 35% by end-2024, driven by COVID-19 supply chain disruptions, Naira depreciation (especially the June 2023 exchange rate unification), and rapid monetary expansion.

The series appears non-stationary in levels, with a clear upward trend in the latter half of the sample. This will need to be formally tested using ADF and KPSS tests before fitting any regression model."

### 2. "What does a high positive correlation between exchange_rate and inflation mean economically?"

**Answer:** "A high positive correlation between the exchange rate (NGN/USD) and inflation means that when the Naira depreciates (the exchange rate rises), inflation also tends to rise. This is consistent with the exchange rate pass-through mechanism: Nigeria is heavily import-dependent for fuel, food, and manufactured goods. When the Naira loses value, the Naira-denominated price of imports increases, which directly pushes up the Consumer Price Index.

However, I would caution that this correlation alone does not establish causation. Both variables may be driven by a common underlying factor — for example, expansionary monetary policy can simultaneously cause both Naira depreciation and domestic inflation. Additionally, the high correlation may be partly spurious, arising because both series have strong upward trends over the sample period. To establish a genuine causal relationship with appropriate dynamics and lags, we use the ARDL bounds testing approach and Granger causality tests in our econometric analysis."

### 3. "Why might M2 and exchange_rate both be correlated with inflation?"

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

**Packages installed today:** `matplotlib==3.8.2`

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
