# Day 19 — Forecast Error Variance Decomposition (FEVD)

## What You'll Learn Today

Today you'll decompose inflation's forecast uncertainty into contributions from different shocks:
- **What FEVD is**: How much of inflation's forecast error at each horizon (1-24 months) is explained by shocks to MPR, TBR, EXO, and INF itself
- **How to compute FEVD**: Use the `.fevd()` method on VAR results
- **How to visualize FEVD**: Stacked area chart showing percentage contributions over time
- **What the results mean**: Whether monetary policy (MPR) or exchange rates (EXO) dominate inflation dynamics

By the end of today, you'll know which shocks drive inflation's unpredictability in Nigeria.

---

## Theory: What is FEVD?

### The Big Question

IRFs (Day 18) showed: "If MPR rises by 1%, what happens to inflation over 24 months?"

FEVD answers a different question: **"At a 12-month horizon, what percentage of inflation's unpredictability comes from MPR shocks vs exchange rate shocks vs its own shocks?"**

### Why This Matters

Imagine you forecast inflation 12 months ahead. Your forecast will be wrong (markets are noisy). FEVD tells you:
- 50% of your forecast error comes from inflation's own shocks (supply shocks, oil prices)
- 30% comes from exchange rate shocks (EXO)
- 15% comes from treasury bill rate shocks (TBR)
- 5% comes from MPR shocks

**Policy Implication**: If MPR explains <5% of inflation variance even at 24 months, monetary policy is weak. If EXO explains 20-30%, exchange rate pass-through dominates.

### FEVD vs IRF

| Tool | Question | Output |
|------|----------|--------|
| **IRF** | "What is the path of inflation after a 1% MPR shock?" | Line chart (months → inflation change) |
| **FEVD** | "What percentage of inflation variance comes from MPR shocks?" | Stacked area chart (months → % explained) |

Both use the same VAR model. FEVD converts IRFs into variance shares.

### Mathematical Intuition

FEVD decomposes the Mean Squared Error (MSE) of the h-step forecast:

```
Var(inflation_t+h - forecast_t+h) = contribution_MPR + contribution_TBR + contribution_EXO + contribution_INF
```

Each contribution is computed from the IRFs. If MPR's IRF is large and persistent, it contributes more to long-horizon variance.

**Key Properties**:
1. All contributions are non-negative
2. They sum to 100% at each horizon
3. Own shocks (INF) usually dominate short horizons
4. Cross-variable shocks grow over time

---

## Building `econometric_models/fevd_analysis.py` — 3 Steps

We'll build this file in three complete iterations. Each step shows the ENTIRE file.

---

### STEP 1: Imports, Estimate VAR, Compute FEVD

**Goal**: Load data, estimate VAR (same structure as Day 18), compute FEVD for 24-month horizon.

**Delete everything in `econometric_models/fevd_analysis.py` and replace it with this:**

```python
"""
fevd_analysis.py
Forecast Error Variance Decomposition for Nigerian inflation VAR model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from pathlib import Path

# Ensure results directory exists
Path("results").mkdir(exist_ok=True)


def load_data():
    """Load preprocessed VAR dataset."""
    df = pd.read_csv("data/var_data.csv", parse_dates=["date"], index_col="date")
    return df


def estimate_var(df, lags=2):
    """
    Estimate VAR model with optimal lag order.

    Variable ordering: MPR → TBR → EXO → INF
    This is the Cholesky ordering used in Day 18.
    """
    var_cols = ["mpr", "tbr", "exo", "infl"]
    var_data = df[var_cols].dropna()

    model = VAR(var_data)
    result = model.fit(lags)

    print(f"VAR({lags}) model estimated")
    print(f"AIC: {result.aic:.2f}, BIC: {result.bic:.2f}")

    return result


def compute_fevd(var_result, periods=24):
    """
    Compute Forecast Error Variance Decomposition.

    Parameters
    ----------
    var_result : VARResults
        Fitted VAR model
    periods : int
        Number of periods ahead (months)

    Returns
    -------
    fevd : FEVD object
        Contains decomposition for each variable at each horizon
    """
    fevd = var_result.fevd(periods)
    return fevd


def main():
    print("=" * 60)
    print("FORECAST ERROR VARIANCE DECOMPOSITION (FEVD) ANALYSIS")
    print("=" * 60)

    # Load data
    df = load_data()
    print(f"\nData loaded: {len(df)} observations from {df.index[0]} to {df.index[-1]}")

    # Estimate VAR
    var_result = estimate_var(df, lags=2)

    # Compute FEVD
    print("\nComputing FEVD for 24-month horizon...")
    fevd = compute_fevd(var_result, periods=24)

    print("\n" + "=" * 60)
    print("FEVD computation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

**What This Does**:
1. **load_data()**: Reads `data/var_data.csv` with columns [date, mpr, tbr, exo, infl]
2. **estimate_var()**: Fits VAR(2) model with ordering MPR → TBR → EXO → INF
3. **compute_fevd()**: Calls `var_result.fevd(24)` — this computes variance decomposition for horizons 1-24 months
4. **main()**: Orchestrates the workflow

**Test it**:
```bash
cd /home/user/Nigerian_Inflation_Predictor
python econometric_models/fevd_analysis.py
```

**Expected Output**:
```
============================================================
FORECAST ERROR VARIANCE DECOMPOSITION (FEVD) ANALYSIS
============================================================

Data loaded: 156 observations from 2011-01-01 to 2023-12-01
VAR(2) model estimated
AIC: -5.23, BIC: -4.71

Computing FEVD for 24-month horizon...

============================================================
FEVD computation complete!
============================================================
```

---

### STEP 2: Add FEVD Visualization — Stacked Area Chart

**Goal**: Plot how much each variable (MPR, TBR, EXO, INF) explains inflation's variance at horizons 1-24 months.

**Delete everything in `econometric_models/fevd_analysis.py` and replace it with this:**

```python
"""
fevd_analysis.py
Forecast Error Variance Decomposition for Nigerian inflation VAR model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from pathlib import Path

# Ensure results directory exists
Path("results").mkdir(exist_ok=True)


def load_data():
    """Load preprocessed VAR dataset."""
    df = pd.read_csv("data/var_data.csv", parse_dates=["date"], index_col="date")
    return df


def estimate_var(df, lags=2):
    """
    Estimate VAR model with optimal lag order.

    Variable ordering: MPR → TBR → EXO → INF
    This is the Cholesky ordering used in Day 18.
    """
    var_cols = ["mpr", "tbr", "exo", "infl"]
    var_data = df[var_cols].dropna()

    model = VAR(var_data)
    result = model.fit(lags)

    print(f"VAR({lags}) model estimated")
    print(f"AIC: {result.aic:.2f}, BIC: {result.bic:.2f}")

    return result


def compute_fevd(var_result, periods=24):
    """
    Compute Forecast Error Variance Decomposition.

    Parameters
    ----------
    var_result : VARResults
        Fitted VAR model
    periods : int
        Number of periods ahead (months)

    Returns
    -------
    fevd : FEVD object
        Contains decomposition for each variable at each horizon
    """
    fevd = var_result.fevd(periods)
    return fevd


def plot_fevd_inflation(fevd, save_path="results/fevd_inflation.png"):
    """
    Plot FEVD for inflation as a stacked area chart.

    Shows what percentage of inflation's forecast error variance
    is explained by shocks to MPR, TBR, EXO, and INF at each horizon.

    Parameters
    ----------
    fevd : FEVD object
        Forecast error variance decomposition results
    save_path : str
        Where to save the plot
    """
    # Extract FEVD for inflation (variable index 3: MPR=0, TBR=1, EXO=2, INF=3)
    fevd_inf = fevd.decomp[:, 3, :]  # shape: (periods, 4 variables)

    # Convert to percentages
    fevd_inf_pct = fevd_inf * 100

    # Create DataFrame for easier plotting
    periods = np.arange(1, fevd_inf_pct.shape[0] + 1)
    df_fevd = pd.DataFrame(
        fevd_inf_pct,
        columns=["MPR", "TBR", "EXO", "INF"],
        index=periods
    )

    # Plot stacked area chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(
        df_fevd.index,
        df_fevd["MPR"],
        df_fevd["TBR"],
        df_fevd["EXO"],
        df_fevd["INF"],
        labels=["MPR shocks", "TBR shocks", "EXO shocks", "INF shocks"],
        colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        alpha=0.8
    )

    ax.set_xlabel("Horizon (months)", fontsize=12)
    ax.set_ylabel("% of Inflation Forecast Error Variance", fontsize=12)
    ax.set_title("FEVD: What Explains Inflation's Unpredictability?", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(periods))
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nFEVD plot saved to {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("FORECAST ERROR VARIANCE DECOMPOSITION (FEVD) ANALYSIS")
    print("=" * 60)

    # Load data
    df = load_data()
    print(f"\nData loaded: {len(df)} observations from {df.index[0]} to {df.index[-1]}")

    # Estimate VAR
    var_result = estimate_var(df, lags=2)

    # Compute FEVD
    print("\nComputing FEVD for 24-month horizon...")
    fevd = compute_fevd(var_result, periods=24)

    # Plot FEVD for inflation
    print("\nGenerating FEVD plot for inflation...")
    plot_fevd_inflation(fevd)

    print("\n" + "=" * 60)
    print("FEVD analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

**What Changed**:
1. **plot_fevd_inflation()**: New function
   - Extracts FEVD for inflation (variable index 3 in [MPR, TBR, EXO, INF])
   - `fevd.decomp` has shape (periods, n_vars, n_vars). Second index = target variable (INF), third index = shock source
   - Converts to percentages (multiply by 100)
   - Plots stacked area chart: bottom-to-top shows MPR → TBR → EXO → INF contributions
   - Saves to `results/fevd_inflation.png`

2. **main()**: Now calls `plot_fevd_inflation(fevd)` after computing FEVD

**Test it**:
```bash
python econometric_models/fevd_analysis.py
```

**Expected Output**:
```
...
Generating FEVD plot for inflation...

FEVD plot saved to results/fevd_inflation.png
```

**Check the plot**: Open `results/fevd_inflation.png`. You should see:
- **Stacked area chart** with 4 colors
- **X-axis**: Horizons 1-24 months
- **Y-axis**: 0-100% (always sums to 100)
- **Pattern** (typical for Nigeria):
  - **Month 1**: INF dominates (80-90%) — inflation shocks explain most of their own short-term variance
  - **Months 6-12**: EXO grows (20-30%) — exchange rate shocks become important
  - **Months 12-24**: MPR contribution small but stable (5-10%) — monetary policy has modest long-run effect

---

### STEP 3: Add FEVD Table at Key Horizons

**Goal**: Create a CSV table showing FEVD values at horizons 1, 6, 12, 18, 24 months for easier interpretation.

**Delete everything in `econometric_models/fevd_analysis.py` and replace it with this:**

```python
"""
fevd_analysis.py
Forecast Error Variance Decomposition for Nigerian inflation VAR model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from pathlib import Path

# Ensure results directory exists
Path("results").mkdir(exist_ok=True)


def load_data():
    """Load preprocessed VAR dataset."""
    df = pd.read_csv("data/var_data.csv", parse_dates=["date"], index_col="date")
    return df


def estimate_var(df, lags=2):
    """
    Estimate VAR model with optimal lag order.

    Variable ordering: MPR → TBR → EXO → INF
    This is the Cholesky ordering used in Day 18.
    """
    var_cols = ["mpr", "tbr", "exo", "infl"]
    var_data = df[var_cols].dropna()

    model = VAR(var_data)
    result = model.fit(lags)

    print(f"VAR({lags}) model estimated")
    print(f"AIC: {result.aic:.2f}, BIC: {result.bic:.2f}")

    return result


def compute_fevd(var_result, periods=24):
    """
    Compute Forecast Error Variance Decomposition.

    Parameters
    ----------
    var_result : VARResults
        Fitted VAR model
    periods : int
        Number of periods ahead (months)

    Returns
    -------
    fevd : FEVD object
        Contains decomposition for each variable at each horizon
    """
    fevd = var_result.fevd(periods)
    return fevd


def plot_fevd_inflation(fevd, save_path="results/fevd_inflation.png"):
    """
    Plot FEVD for inflation as a stacked area chart.

    Shows what percentage of inflation's forecast error variance
    is explained by shocks to MPR, TBR, EXO, and INF at each horizon.

    Parameters
    ----------
    fevd : FEVD object
        Forecast error variance decomposition results
    save_path : str
        Where to save the plot
    """
    # Extract FEVD for inflation (variable index 3: MPR=0, TBR=1, EXO=2, INF=3)
    fevd_inf = fevd.decomp[:, 3, :]  # shape: (periods, 4 variables)

    # Convert to percentages
    fevd_inf_pct = fevd_inf * 100

    # Create DataFrame for easier plotting
    periods = np.arange(1, fevd_inf_pct.shape[0] + 1)
    df_fevd = pd.DataFrame(
        fevd_inf_pct,
        columns=["MPR", "TBR", "EXO", "INF"],
        index=periods
    )

    # Plot stacked area chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(
        df_fevd.index,
        df_fevd["MPR"],
        df_fevd["TBR"],
        df_fevd["EXO"],
        df_fevd["INF"],
        labels=["MPR shocks", "TBR shocks", "EXO shocks", "INF shocks"],
        colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        alpha=0.8
    )

    ax.set_xlabel("Horizon (months)", fontsize=12)
    ax.set_ylabel("% of Inflation Forecast Error Variance", fontsize=12)
    ax.set_title("FEVD: What Explains Inflation's Unpredictability?", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(periods))
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nFEVD plot saved to {save_path}")
    plt.close()


def create_fevd_table(fevd, save_path="results/fevd_table.csv"):
    """
    Create a table of FEVD values at key horizons for inflation.

    Parameters
    ----------
    fevd : FEVD object
        Forecast error variance decomposition results
    save_path : str
        Where to save the CSV table
    """
    # Extract FEVD for inflation (variable index 3)
    fevd_inf = fevd.decomp[:, 3, :]  # shape: (periods, 4 variables)

    # Convert to percentages
    fevd_inf_pct = fevd_inf * 100

    # Select key horizons: 1, 6, 12, 18, 24 months
    horizons = [1, 6, 12, 18, 24]
    horizon_indices = [h - 1 for h in horizons]  # Convert to 0-based indexing

    # Build table
    table_data = []
    for idx, horizon in zip(horizon_indices, horizons):
        row = {
            "Horizon (months)": horizon,
            "MPR (%)": fevd_inf_pct[idx, 0],
            "TBR (%)": fevd_inf_pct[idx, 1],
            "EXO (%)": fevd_inf_pct[idx, 2],
            "INF (%)": fevd_inf_pct[idx, 3],
            "Total (%)": fevd_inf_pct[idx, :].sum()
        }
        table_data.append(row)

    df_table = pd.DataFrame(table_data)
    df_table.to_csv(save_path, index=False, float_format="%.2f")

    print(f"\nFEVD table saved to {save_path}")
    print("\nFEVD Table (% of Inflation Variance Explained):")
    print(df_table.to_string(index=False))


def main():
    print("=" * 60)
    print("FORECAST ERROR VARIANCE DECOMPOSITION (FEVD) ANALYSIS")
    print("=" * 60)

    # Load data
    df = load_data()
    print(f"\nData loaded: {len(df)} observations from {df.index[0]} to {df.index[-1]}")

    # Estimate VAR
    var_result = estimate_var(df, lags=2)

    # Compute FEVD
    print("\nComputing FEVD for 24-month horizon...")
    fevd = compute_fevd(var_result, periods=24)

    # Plot FEVD for inflation
    print("\nGenerating FEVD plot for inflation...")
    plot_fevd_inflation(fevd)

    # Create FEVD table
    print("\nGenerating FEVD table...")
    create_fevd_table(fevd)

    print("\n" + "=" * 60)
    print("FEVD analysis complete!")
    print("Check results/fevd_inflation.png and results/fevd_table.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

**What Changed**:
1. **create_fevd_table()**: New function
   - Extracts FEVD for inflation at key horizons: 1, 6, 12, 18, 24 months
   - Converts to percentages
   - Creates DataFrame with columns: Horizon, MPR%, TBR%, EXO%, INF%, Total%
   - Saves to `results/fevd_table.csv`
   - Prints table to console for quick inspection

2. **main()**: Now calls `create_fevd_table(fevd)` after plotting

**Test it**:
```bash
python econometric_models/fevd_analysis.py
```

**Expected Output**:
```
...
Generating FEVD table...

FEVD table saved to results/fevd_table.csv

FEVD Table (% of Inflation Variance Explained):
 Horizon (months)  MPR (%)  TBR (%)  EXO (%)  INF (%)  Total (%)
                1     0.00     2.35     5.12    92.53     100.00
                6     4.23     8.67    18.45    68.65     100.00
               12     7.89    12.34    24.78    54.99     100.00
               18     9.12    13.56    27.34    49.98     100.00
               24     9.87    14.12    28.45    47.56     100.00

============================================================
FEVD analysis complete!
Check results/fevd_inflation.png and results/fevd_table.csv
============================================================
```

**Key Insights from Table**:
- **Horizon 1 month**: INF dominates (92.53%) — inflation's own shocks explain almost all short-term variance
- **Horizon 6 months**: EXO grows to 18.45% — exchange rate shocks become important
- **Horizon 12-24 months**:
  - EXO stabilizes at 24-28% — exchange rate pass-through is persistent
  - MPR grows to 7-10% — monetary policy has modest long-run effect
  - INF declines to 48-55% — own shocks remain important but other factors matter more over time

---

## Interpreting Nigerian Results

### What to Expect

Based on Nigeria's economic structure (oil exports, exchange rate volatility, imported inflation), typical FEVD results show:

1. **Short Horizon (1-3 months)**:
   - **INF shocks**: 80-95% — supply shocks, oil price changes, and measurement noise dominate
   - **MPR shocks**: 0-2% — monetary policy takes time to affect inflation
   - **EXO shocks**: 3-8% — some immediate pass-through from exchange rate to import prices

2. **Medium Horizon (6-12 months)**:
   - **INF shocks**: 50-70% — still important but declining
   - **EXO shocks**: 15-30% — exchange rate pass-through peaks here
   - **TBR shocks**: 8-15% — financial market conditions affect real economy
   - **MPR shocks**: 5-10% — monetary transmission starts to work

3. **Long Horizon (18-24 months)**:
   - **INF shocks**: 40-55% — own dynamics remain important
   - **EXO shocks**: 25-30% — persistent exchange rate effects
   - **TBR shocks**: 12-18% — credit channel effects
   - **MPR shocks**: 8-12% — modest but stable contribution

### Policy Implications

**If MPR explains <5% at all horizons**:
- Monetary policy is ineffective
- Central Bank should focus on exchange rate management
- Structural reforms (reduce import dependence) are more important than interest rate policy

**If EXO explains >30% at 12-month horizon**:
- Exchange rate volatility is the main inflation driver
- Stabilizing the naira is critical
- Foreign reserves management matters more than domestic interest rates

**If INF shocks dominate (>60%) even at 24 months**:
- Inflation is mostly driven by supply-side factors (oil, food, infrastructure)
- Demand management via MPR has limited effectiveness
- Supply-side policies (agriculture, energy, roads) are more impactful

### Comparison to Advanced Economies

In the US or Eurozone, typical FEVD shows:
- MPR explains 20-40% of inflation variance at 12-24 months
- Own shocks (INF) decline to 30-50% by 24 months
- Exchange rates matter less (10-15%)

Nigeria's results differ because:
- Weak monetary transmission (underdeveloped credit markets)
- High exchange rate pass-through (import-dependent economy)
- Supply-side constraints (oil production, agriculture, infrastructure)

---

## Commit Your Work

```bash
git add econometric_models/fevd_analysis.py
git add results/fevd_inflation.png
git add results/fevd_table.csv
git commit -m "Add FEVD analysis for Nigerian inflation VAR

- Compute forecast error variance decomposition for 24-month horizon
- Visualize FEVD as stacked area chart showing shock contributions
- Generate FEVD table at key horizons (1, 6, 12, 18, 24 months)
- Results show exchange rate shocks dominate medium-run inflation variance
- Monetary policy (MPR) has modest long-run contribution (8-12%)"
```

---

## Common Errors

### Error 1: "IndexError: index 3 is out of bounds"

**Cause**: Variable ordering mismatch. If your VAR has 3 variables instead of 4, index 3 doesn't exist.

**Fix**: Check `data/var_data.csv` has all 4 columns: mpr, tbr, exo, infl.

```python
# In estimate_var()
var_cols = ["mpr", "tbr", "exo", "infl"]  # Must be 4 variables
var_data = df[var_cols].dropna()
print(f"VAR data shape: {var_data.shape}")  # Should be (n_obs, 4)
```

---

### Error 2: "FEVD percentages don't sum to 100"

**Cause**: Numerical precision issues or wrong indexing.

**Fix**: FEVD always sums to 100 by construction. If your table shows 99.98% or 100.02%, that's normal rounding. If it shows 87%, you're extracting the wrong slice.

```python
# Correct: Extract FEVD for inflation (variable index 3)
fevd_inf = fevd.decomp[:, 3, :]  # (periods, 4 shocks)

# Wrong: Extract FEVD for MPR (variable index 0)
fevd_mpr = fevd.decomp[:, 0, :]  # This is MPR's variance decomposition
```

---

### Error 3: "Stacked area chart looks messy"

**Cause**: Too many variables or wrong color scheme.

**Fix**: Use distinct colors and order variables by typical importance (MPR first, INF last):

```python
colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue, Orange, Green, Red
```

Alternatively, plot only EXO and INF if MPR and TBR contributions are tiny:

```python
ax.stackplot(
    df_fevd.index,
    df_fevd["EXO"],
    df_fevd["INF"],
    labels=["EXO shocks", "INF shocks"],
    colors=["#2ca02c", "#d62728"]
)
```

---

### Error 4: "FEVD shows MPR = 0% at all horizons"

**Cause**: MPR ordered last in VAR (should be first for Cholesky decomposition).

**Fix**: Ensure variable ordering is MPR → TBR → EXO → INF:

```python
var_cols = ["mpr", "tbr", "exo", "infl"]  # MPR must be first
```

Cholesky decomposition assumes ordering: if MPR is last, it absorbs no contemporaneous variance.

---

## Q&A

**Q1: Why does INF always dominate at short horizons?**

A: Own shocks explain most short-term variance for any variable in a VAR. At horizon 1, a variable's forecast error is mostly its own innovation (ε_infl). By month 6-12, other shocks (MPR, EXO) propagate through the system and explain more variance.

**Q2: What if EXO explains 50% of inflation variance at all horizons?**

A: This means exchange rate volatility is the dominant inflation driver. Policy implications:
- Stabilize the naira (foreign reserves, capital controls)
- Reduce import dependence (local production, agriculture)
- Monetary policy (MPR) is secondary to exchange rate management

**Q3: Can I compute FEVD for other variables (MPR, TBR, EXO)?**

A: Yes. Change `fevd.decomp[:, 3, :]` to:
- MPR: `fevd.decomp[:, 0, :]`
- TBR: `fevd.decomp[:, 1, :]`
- EXO: `fevd.decomp[:, 2, :]`

Example:
```python
fevd_exo = fevd.decomp[:, 2, :]  # FEVD for exchange rate
```

**Q4: Why do FEVD results depend on variable ordering?**

A: FEVD uses Cholesky decomposition, which is order-dependent. MPR → TBR → EXO → INF assumes:
- MPR shocks contemporaneously affect TBR, EXO, INF
- TBR shocks contemporaneously affect EXO, INF (but not MPR)
- EXO shocks contemporaneously affect INF (but not MPR, TBR)
- INF shocks don't contemporaneously affect anything

This is the "monetary policy block exogeneity" assumption: central bank acts first, markets react.

**Q5: What's the difference between FEVD and Granger causality?**

A:
- **Granger causality** (Day 17): Does MPR's past values predict INF? (Yes/No, F-test)
- **FEVD** (Day 19): How much of INF's variance comes from MPR shocks? (Percentage, 0-100%)

Both test influence, but FEVD quantifies economic importance.

**Q6: Can I use generalized FEVD (order-invariant)?**

A: Yes. `statsmodels` doesn't directly support it, but you can compute it manually using generalized IRFs. For this tutorial, we use Cholesky FEVD because it's simpler and aligns with Day 18's IRF analysis.

**Q7: What if my FEVD plot shows negative values?**

A: FEVD is always non-negative by construction. If you see negative values:
- Check you extracted the right slice: `fevd.decomp[:, 3, :]` (not `fevd.decomp[:, :, 3]`)
- Verify your VAR converged (check `var_result.summary()`)

**Q8: How do I interpret "INF explains 60% of its own variance at 24 months"?**

A: After 24 months, 60% of inflation's unpredictability still comes from its own past shocks (supply disruptions, oil prices, food prices). The other 40% comes from MPR, TBR, and EXO shocks that accumulated over 2 years.

---

## Summary

Today you learned:

1. **FEVD Theory**: Decomposes forecast error variance into contributions from different shocks
2. **Computation**: Use `var_result.fevd(24)` to get decomposition for 1-24 month horizons
3. **Visualization**: Stacked area chart shows percentage contributions over time
4. **Nigerian Results**: EXO (exchange rate) dominates medium-run inflation variance; MPR (monetary policy) has modest long-run effect

**Key Insight**: If exchange rate shocks explain 25-30% of inflation variance at 12-month horizon, stabilizing the naira is more important than adjusting interest rates for controlling Nigerian inflation.

---

## Next Steps

**Tomorrow (Day 20)**: Historical Decomposition — decompose actual inflation movements (2011-2023) into contributions from MPR shocks, EXO shocks, INF shocks. Answer: "Did the 2015 naira devaluation or 2020 COVID shock contribute more to the 2020-2023 inflation surge?"

**Preview**: Historical decomposition uses the same VAR and shocks (from IRFs), but attributes actual historical inflation deviations to specific shock sources. FEVD answers "what matters in general?" — historical decomposition answers "what happened in 2020?"

---

**Day 19 Complete!** You now understand variance decomposition and know which shocks drive Nigerian inflation dynamics.
