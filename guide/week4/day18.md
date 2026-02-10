# Day 18 — Impulse Response Functions (IRFs)

## What You'll Learn Today

Today you'll learn how to trace the dynamic effects of shocks through your VAR system using **Impulse Response Functions (IRFs)**. Specifically, you'll:

- Understand what an IRF is: tracing a 1-unit shock to one variable through the entire system over time
- Compute the key IRF: what happens to TBR, EXO, and INF when MPR increases by 1 unit
- Create IRF plots with 95% confidence bands to assess statistical significance
- Understand the difference between orthogonalized and non-orthogonalized IRFs
- Interpret Nigerian monetary policy transmission through IRF plots

By the end, you'll answer: "If the CBN raises the MPR by 1 percentage point today, what happens to inflation over the next 24 months?"

---

## Theory: What is an Impulse Response Function?

### The Core Idea

An **Impulse Response Function (IRF)** traces the effect of a one-time shock to one variable on all variables in the system over time.

Think of it like dropping a stone in a pond:
- The stone (shock) hits one spot (one variable)
- Ripples (effects) spread across the entire pond (all variables)
- You measure the ripples at 1 month, 2 months, ..., 24 months after the stone dropped

### The Nigerian Monetary Policy Question

"If the Central Bank of Nigeria raises the Monetary Policy Rate (MPR) by 1 percentage point today, what happens to:
- Treasury Bill Rate (TBR) over the next 24 months?
- Exchange Rate (EXO) over the next 24 months?
- Inflation (INF) over the next 24 months?"

The IRF answers this question by simulating the shock and tracking the responses.

### Orthogonalized vs Non-Orthogonalized IRFs

**Problem:** In reality, shocks happen to multiple variables simultaneously. How do you isolate a shock to just MPR?

**Solution:** Use **Cholesky decomposition** to "orthogonalize" the shocks. This makes shocks independent of each other.

**Ordering matters:** With ordering MPR → TBR → EXO → INF:
- A shock to MPR can affect TBR, EXO, and INF contemporaneously
- A shock to TBR can affect EXO and INF contemporaneously (but not MPR)
- A shock to EXO can affect INF contemporaneously (but not MPR or TBR)
- A shock to INF affects nothing contemporaneously

This ordering reflects Nigerian monetary policy transmission:
1. CBN changes MPR (policy rate)
2. TBR (market rate) responds immediately
3. EXO (exchange rate) adjusts
4. INF (inflation) is the final target variable

### Confidence Bands

IRFs come with 95% confidence bands:
- If the band includes zero at a given time horizon, the response is **not statistically significant** at that horizon
- If the band is entirely above (or below) zero, the response is **statistically significant**

This tells you: "Can we confidently say MPR affects inflation 12 months later, or is it just noise?"

---

## Building econometric_models/irf_analysis.py

You'll build this script in **3 steps**. Each step shows the **complete file**.

---

### STEP 1: Estimate VAR and Compute IRF

Delete everything in **econometric_models/irf_analysis.py** and replace it with this:

```python
"""Impulse Response Function analysis for the Nigerian Inflation Predictor."""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path to import from other modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Directories
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Variable ordering: MPR → TBR → EXO → INF
VARIABLE_ORDER = ["mpr", "tbr", "exo", "infl"]


def load_and_estimate_var():
    """
    Load cleaned data and estimate VAR model.

    Returns:
        var_result: Fitted VAR model result
    """
    # Load cleaned data
    df = pd.read_csv(
        os.path.join(PROCESSED_DIR, "cleaned_data.csv"),
        index_col="date",
        parse_dates=True
    )

    # Select variables in correct order
    df = df[VARIABLE_ORDER]

    # First difference to achieve stationarity
    df_diff = df.diff().dropna()

    # Estimate VAR
    from statsmodels.tsa.api import VAR
    model = VAR(df_diff)

    # Select lag order using BIC
    lag_order = model.select_order(maxlags=12)
    optimal_lag = lag_order.bic

    print(f"Estimating VAR with {optimal_lag} lags...")
    result = model.fit(optimal_lag)

    return result


def compute_irf(var_result, periods=24):
    """
    Compute Impulse Response Functions.

    Args:
        var_result: Fitted VAR model
        periods: Number of periods to compute IRF (default 24 months)

    Returns:
        irf: IRF object from statsmodels
    """
    print(f"Computing IRFs for {periods} periods...")
    irf = var_result.irf(periods)
    return irf


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Estimate VAR
    result = load_and_estimate_var()

    # Compute IRF
    irf = compute_irf(result, periods=24)

    print("\nIRF computed successfully for 24 periods.")
    print("IRF represents orthogonalized responses using Cholesky decomposition.")
    print(f"Variable ordering: {' → '.join(VARIABLE_ORDER)}")
```

**What this does:**
- Loads cleaned_data.csv and selects variables in the correct order: MPR → TBR → EXO → INF
- Differences the data to make it stationary
- Estimates VAR with optimal lag order (using BIC)
- Computes IRF for 24 periods (2 years) using `var_result.irf(24)`

**Run it:**
```bash
cd /home/user/Nigerian_Inflation_Predictor
python econometric_models/irf_analysis.py
```

You should see:
```
Estimating VAR with X lags...
Computing IRFs for 24 periods...

IRF computed successfully for 24 periods.
IRF represents orthogonalized responses using Cholesky decomposition.
Variable ordering: mpr → tbr → exo → infl
```

---

### STEP 2: Plot MPR Shock IRF (4-Panel Plot)

Delete everything in **econometric_models/irf_analysis.py** and replace it with this:

```python
"""Impulse Response Function analysis for the Nigerian Inflation Predictor."""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path to import from other modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Directories
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Variable ordering: MPR → TBR → EXO → INF
VARIABLE_ORDER = ["mpr", "tbr", "exo", "infl"]


def load_and_estimate_var():
    """
    Load cleaned data and estimate VAR model.

    Returns:
        var_result: Fitted VAR model result
    """
    # Load cleaned data
    df = pd.read_csv(
        os.path.join(PROCESSED_DIR, "cleaned_data.csv"),
        index_col="date",
        parse_dates=True
    )

    # Select variables in correct order
    df = df[VARIABLE_ORDER]

    # First difference to achieve stationarity
    df_diff = df.diff().dropna()

    # Estimate VAR
    from statsmodels.tsa.api import VAR
    model = VAR(df_diff)

    # Select lag order using BIC
    lag_order = model.select_order(maxlags=12)
    optimal_lag = lag_order.bic

    print(f"Estimating VAR with {optimal_lag} lags...")
    result = model.fit(optimal_lag)

    return result


def compute_irf(var_result, periods=24):
    """
    Compute Impulse Response Functions.

    Args:
        var_result: Fitted VAR model
        periods: Number of periods to compute IRF (default 24 months)

    Returns:
        irf: IRF object from statsmodels
    """
    print(f"Computing IRFs for {periods} periods...")
    irf = var_result.irf(periods)
    return irf


def plot_mpr_shock_irf(irf):
    """
    Plot IRF of all variables to a 1-unit shock in MPR.
    Creates a 2x2 panel plot showing response of MPR, TBR, EXO, and INF.

    Args:
        irf: IRF object from statsmodels
    """
    print("Plotting MPR shock IRF...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Impulse Response Functions: Response to 1 Unit Shock in MPR", fontsize=16, fontweight="bold")

    # Shock variable
    shock_var = "mpr"

    # Response variables and titles
    response_vars = ["mpr", "tbr", "exo", "infl"]
    titles = [
        "Response of MPR to MPR Shock",
        "Response of TBR to MPR Shock",
        "Response of EXO to MPR Shock",
        "Response of INF to MPR Shock"
    ]
    ylabels = ["MPR (pp)", "TBR (pp)", "EXO (Naira/$)", "Inflation (pp)"]

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for idx, (response_var, title, ylabel) in enumerate(zip(response_vars, titles, ylabels)):
        ax = axes[idx]

        # Get IRF data
        irf_data = irf.irfs[:, VARIABLE_ORDER.index(response_var), VARIABLE_ORDER.index(shock_var)]

        # Get confidence intervals (95%)
        lower_band = irf.ci[:, VARIABLE_ORDER.index(response_var), VARIABLE_ORDER.index(shock_var), 0]
        upper_band = irf.ci[:, VARIABLE_ORDER.index(response_var), VARIABLE_ORDER.index(shock_var), 1]

        # Time axis (0 to 23 months)
        periods = len(irf_data)
        time_axis = np.arange(periods)

        # Plot IRF
        ax.plot(time_axis, irf_data, linewidth=2, color="blue", label="IRF")

        # Plot confidence bands
        ax.fill_between(time_axis, lower_band, upper_band, alpha=0.3, color="blue", label="95% CI")

        # Zero line
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

        # Labels and title
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Months", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(RESULTS_DIR, "irf_mpr_shock.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"MPR shock IRF plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Estimate VAR
    result = load_and_estimate_var()

    # Compute IRF
    irf = compute_irf(result, periods=24)

    # Plot MPR shock IRF (4-panel plot)
    plot_mpr_shock_irf(irf)

    print("\n" + "="*60)
    print("IRF analysis complete!")
    print(f"Variable ordering: {' → '.join(VARIABLE_ORDER)}")
    print("="*60)
```

**What's new:**
- Added `plot_mpr_shock_irf(irf)` function
- Creates 2×2 panel plot (4 subplots)
- Each subplot shows response of one variable (MPR, TBR, EXO, INF) to MPR shock
- Includes 95% confidence intervals (shaded blue area)
- Zero line (dashed black) shows where response is zero
- Saves to results/irf_mpr_shock.png

**Run it:**
```bash
python econometric_models/irf_analysis.py
```

Check **results/irf_mpr_shock.png**. You should see 4 panels showing how each variable responds to an MPR shock over 24 months.

---

### STEP 3: Plot All IRFs (4×4 Grid) and Save CSV

Delete everything in **econometric_models/irf_analysis.py** and replace it with this:

```python
"""Impulse Response Function analysis for the Nigerian Inflation Predictor."""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path to import from other modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Directories
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Variable ordering: MPR → TBR → EXO → INF
VARIABLE_ORDER = ["mpr", "tbr", "exo", "infl"]


def load_and_estimate_var():
    """
    Load cleaned data and estimate VAR model.

    Returns:
        var_result: Fitted VAR model result
    """
    # Load cleaned data
    df = pd.read_csv(
        os.path.join(PROCESSED_DIR, "cleaned_data.csv"),
        index_col="date",
        parse_dates=True
    )

    # Select variables in correct order
    df = df[VARIABLE_ORDER]

    # First difference to achieve stationarity
    df_diff = df.diff().dropna()

    # Estimate VAR
    from statsmodels.tsa.api import VAR
    model = VAR(df_diff)

    # Select lag order using BIC
    lag_order = model.select_order(maxlags=12)
    optimal_lag = lag_order.bic

    print(f"Estimating VAR with {optimal_lag} lags...")
    result = model.fit(optimal_lag)

    return result


def compute_irf(var_result, periods=24):
    """
    Compute Impulse Response Functions.

    Args:
        var_result: Fitted VAR model
        periods: Number of periods to compute IRF (default 24 months)

    Returns:
        irf: IRF object from statsmodels
    """
    print(f"Computing IRFs for {periods} periods...")
    irf = var_result.irf(periods)
    return irf


def plot_mpr_shock_irf(irf):
    """
    Plot IRF of all variables to a 1-unit shock in MPR.
    Creates a 2x2 panel plot showing response of MPR, TBR, EXO, and INF.

    Args:
        irf: IRF object from statsmodels
    """
    print("Plotting MPR shock IRF...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Impulse Response Functions: Response to 1 Unit Shock in MPR", fontsize=16, fontweight="bold")

    # Shock variable
    shock_var = "mpr"

    # Response variables and titles
    response_vars = ["mpr", "tbr", "exo", "infl"]
    titles = [
        "Response of MPR to MPR Shock",
        "Response of TBR to MPR Shock",
        "Response of EXO to MPR Shock",
        "Response of INF to MPR Shock"
    ]
    ylabels = ["MPR (pp)", "TBR (pp)", "EXO (Naira/$)", "Inflation (pp)"]

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for idx, (response_var, title, ylabel) in enumerate(zip(response_vars, titles, ylabels)):
        ax = axes[idx]

        # Get IRF data
        irf_data = irf.irfs[:, VARIABLE_ORDER.index(response_var), VARIABLE_ORDER.index(shock_var)]

        # Get confidence intervals (95%)
        lower_band = irf.ci[:, VARIABLE_ORDER.index(response_var), VARIABLE_ORDER.index(shock_var), 0]
        upper_band = irf.ci[:, VARIABLE_ORDER.index(response_var), VARIABLE_ORDER.index(shock_var), 1]

        # Time axis (0 to 23 months)
        periods = len(irf_data)
        time_axis = np.arange(periods)

        # Plot IRF
        ax.plot(time_axis, irf_data, linewidth=2, color="blue", label="IRF")

        # Plot confidence bands
        ax.fill_between(time_axis, lower_band, upper_band, alpha=0.3, color="blue", label="95% CI")

        # Zero line
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

        # Labels and title
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Months", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(RESULTS_DIR, "irf_mpr_shock.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"MPR shock IRF plot saved to: {output_path}")
    plt.close()


def plot_all_irfs(irf):
    """
    Plot all IRFs in a 4x4 grid.
    Each row represents a response variable.
    Each column represents a shock variable.

    Args:
        irf: IRF object from statsmodels
    """
    print("Plotting all IRFs (4x4 grid)...")

    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    fig.suptitle("All Impulse Response Functions (4×4 Grid)", fontsize=18, fontweight="bold")

    var_labels = ["MPR", "TBR", "EXO", "INF"]

    for row_idx, response_var in enumerate(VARIABLE_ORDER):
        for col_idx, shock_var in enumerate(VARIABLE_ORDER):
            ax = axes[row_idx, col_idx]

            # Get IRF data
            irf_data = irf.irfs[:, row_idx, col_idx]

            # Get confidence intervals (95%)
            lower_band = irf.ci[:, row_idx, col_idx, 0]
            upper_band = irf.ci[:, row_idx, col_idx, 1]

            # Time axis
            periods = len(irf_data)
            time_axis = np.arange(periods)

            # Plot IRF
            ax.plot(time_axis, irf_data, linewidth=1.5, color="darkblue")

            # Plot confidence bands
            ax.fill_between(time_axis, lower_band, upper_band, alpha=0.2, color="blue")

            # Zero line
            ax.axhline(y=0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)

            # Title (only on top row)
            if row_idx == 0:
                ax.set_title(f"Shock: {var_labels[col_idx]}", fontsize=11, fontweight="bold")

            # Y-axis label (only on left column)
            if col_idx == 0:
                ax.set_ylabel(f"{var_labels[row_idx]}", fontsize=11, fontweight="bold")

            # X-axis label (only on bottom row)
            if row_idx == 3:
                ax.set_xlabel("Months", fontsize=9)

            ax.grid(True, alpha=0.2, linewidth=0.5)
            ax.tick_params(labelsize=8)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(RESULTS_DIR, "irf_all.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"All IRFs plot saved to: {output_path}")
    plt.close()


def save_mpr_to_infl_irf_csv(irf):
    """
    Save the IRF of inflation response to MPR shock as CSV.
    This is the most important IRF for policy analysis.

    Args:
        irf: IRF object from statsmodels
    """
    print("Saving MPR → INF IRF to CSV...")

    # Get IRF data for INF response to MPR shock
    irf_data = irf.irfs[:, VARIABLE_ORDER.index("infl"), VARIABLE_ORDER.index("mpr")]

    # Get confidence intervals
    lower_band = irf.ci[:, VARIABLE_ORDER.index("infl"), VARIABLE_ORDER.index("mpr"), 0]
    upper_band = irf.ci[:, VARIABLE_ORDER.index("infl"), VARIABLE_ORDER.index("mpr"), 1]

    # Create DataFrame
    periods = len(irf_data)
    df = pd.DataFrame({
        "month": np.arange(periods),
        "irf": irf_data,
        "lower_95ci": lower_band,
        "upper_95ci": upper_band
    })

    # Save to CSV
    output_path = os.path.join(RESULTS_DIR, "irf_mpr_to_infl.csv")
    df.to_csv(output_path, index=False)
    print(f"MPR → INF IRF data saved to: {output_path}")


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Estimate VAR
    result = load_and_estimate_var()

    # Compute IRF
    irf = compute_irf(result, periods=24)

    # Plot MPR shock IRF (4-panel plot)
    plot_mpr_shock_irf(irf)

    # Plot all IRFs (4x4 grid)
    plot_all_irfs(irf)

    # Save MPR → INF IRF to CSV
    save_mpr_to_infl_irf_csv(irf)

    print("\n" + "="*60)
    print("IRF analysis complete!")
    print(f"Variable ordering: {' → '.join(VARIABLE_ORDER)}")
    print("Output files:")
    print("  - results/irf_mpr_shock.png (4-panel MPR shock)")
    print("  - results/irf_all.png (4×4 grid of all IRFs)")
    print("  - results/irf_mpr_to_infl.csv (MPR → INF data)")
    print("="*60)
```

**What's new:**
- Added `plot_all_irfs(irf)` — creates 4×4 grid showing all 16 shock-response combinations
  - Rows = response variables (what reacts)
  - Columns = shock variables (what causes the shock)
  - Example: cell (row=INF, col=MPR) shows inflation's response to MPR shock
- Added `save_mpr_to_infl_irf_csv(irf)` — saves the most important IRF (MPR → INF) to CSV for further analysis
- CSV contains: month, IRF value, lower 95% CI, upper 95% CI

**Run it:**
```bash
python econometric_models/irf_analysis.py
```

You should now have:
- **results/irf_mpr_shock.png** — 4-panel plot of MPR shock
- **results/irf_all.png** — 4×4 grid of all IRFs
- **results/irf_mpr_to_infl.csv** — CSV data for MPR → inflation IRF

---

## Interpreting Nigerian Results

### Expected IRF Pattern for MPR Shock

When you look at **results/irf_mpr_shock.png**, here's what you should expect to see:

**1. MPR response to MPR shock (top-left panel):**
- Immediate spike to 1.0 at month 0 (the shock itself)
- Gradual decay back toward zero over 12-24 months
- This shows the shock is temporary, not permanent

**2. TBR response to MPR shock (top-right panel):**
- Immediate positive response (TBR rises when MPR rises)
- Peak at 0.5-0.8 in first 1-3 months
- Gradual return to zero
- **Interpretation:** Market rates follow policy rates immediately (strong transmission)

**3. EXO response to MPR shock (bottom-left panel):**
- Small positive response initially (Naira appreciates when MPR rises)
- May turn slightly negative later
- Often **not statistically significant** (confidence band includes zero)
- **Interpretation:** Higher interest rates attract foreign capital, strengthening Naira, but effect is weak

**4. INF response to MPR shock (bottom-right panel):**
- Initial response near zero (inflation doesn't react immediately)
- Becomes **negative** after 6-12 months (inflation falls when MPR rises)
- Peak effect around 12-18 months
- Gradual return to zero after 24 months
- **Interpretation:** Monetary policy works, but with a long lag (6-12 months)

### Policy Implications

**If confidence bands include zero:** The effect is not statistically significant. MPR may not be effectively controlling inflation.

**If confidence bands are below zero for INF response:** MPR increases successfully reduce inflation with high confidence. This validates CBN's use of MPR as a policy tool.

**Lag structure:** The 6-12 month lag between MPR changes and inflation changes explains why CBN must be forward-looking. They can't wait to see inflation rising before acting — they must anticipate it.

### Looking at results/irf_all.png

The 4×4 grid shows all possible shocks and responses. Key insights:

- **Diagonal elements:** How each variable responds to its own shock (should spike at month 0)
- **Lower triangle:** Ordering restrictions at work (Cholesky decomposition)
  - Example: MPR shock affects all variables, but INF shock only affects INF itself at month 0
- **Cross-responses:** Look for statistically significant cross-effects
  - Does EXO shock affect INF? (Exchange rate pass-through)
  - Does TBR shock affect INF independently of MPR?

---

## Commit Your Work

```bash
cd /home/user/Nigerian_Inflation_Predictor
git add econometric_models/irf_analysis.py results/irf_mpr_shock.png results/irf_all.png results/irf_mpr_to_infl.csv
git commit -m "Add IRF analysis: orthogonalized impulse responses with 95% CI bands"
```

---

## Common Errors and Solutions

### Error 1: "IRF computation failed"

**Problem:** VAR model not estimated properly (possibly due to insufficient data or wrong lag order).

**Solution:**
- Check that cleaned_data.csv has at least 50-100 observations
- Ensure variables are in correct order: mpr, tbr, exo, infl
- Try reducing maxlags in `model.select_order(maxlags=12)` to maxlags=6

### Error 2: "Confidence bands are very wide"

**Problem:** High uncertainty in IRF estimates, often due to small sample size or high residual variance.

**Solution:**
- This is not necessarily an error — it reflects genuine uncertainty
- Consider using more data if available
- Wide bands mean effects are not statistically significant (which is itself a finding)

### Error 3: "IRF plot shows unexpected patterns"

**Problem:**
- Shocks don't decay to zero (model may be non-stationary)
- Confidence bands explode at long horizons

**Solution:**
- Verify that you differenced the data (`df.diff()`) before estimating VAR
- Check stationarity using Day 16's ADF tests
- Consider reducing IRF horizon from 24 to 12 months

### Error 4: "All IRFs look the same"

**Problem:** Variables are too highly correlated, or data has limited variation.

**Solution:**
- This is a data issue, not a code issue
- Check correlation matrix from Day 14
- Consider adding more exogenous variables to capture different shocks

---

## Q&A

**Q1: Why use orthogonalized IRFs instead of simple shocks?**

**A:** In reality, shocks happen to multiple variables at once. For example, when oil prices spike, both EXO and INF might jump simultaneously. Orthogonalized IRFs use Cholesky decomposition to "disentangle" these shocks so you can isolate the effect of a shock to just one variable. Without orthogonalization, you couldn't answer "What happens if ONLY MPR changes?"

**Q2: What if I change the variable ordering?**

**A:** Ordering affects **contemporaneous effects** only (month 0). If you put INF first instead of MPR, then an inflation shock can affect MPR contemporaneously, which doesn't make economic sense (CBN can't react instantly to inflation). After month 0, ordering doesn't matter as much. Always use an ordering that reflects causal priority: policy variables first, target variables last.

**Q3: Why are confidence bands wider at longer horizons?**

**A:** Uncertainty compounds over time. Predicting what happens in month 1 is easier than predicting month 24. This is normal and reflects genuine uncertainty about long-run effects.

**Q4: What does it mean if the confidence band includes zero?**

**A:** The response is **not statistically significant** at that time horizon. You cannot confidently say the shock has a non-zero effect. For policy purposes, this means MPR might not be effectively controlling the target variable at that horizon.

**Q5: Can I compute IRFs for cumulative responses?**

**A:** Yes. IRFs show **marginal** effects (effect in each period). Cumulative IRFs sum these up to show total accumulated effect. In statsmodels, use `irf.cum_effects` instead of `irf.irfs`. Cumulative IRFs are useful for understanding long-run total impact.

**Q6: How do I interpret an IRF that doesn't return to zero?**

**A:** If the IRF doesn't return to zero after 24 months, it suggests:
1. The shock has a **permanent** effect (unlikely if you differenced the data properly), or
2. 24 months is not long enough to observe full decay — try extending to 36 or 48 months

For differenced data, IRFs should always decay to zero eventually.

**Q7: What's the difference between IRF and FEVD?**

**A:**
- **IRF:** Shows the size and direction of response over time (measured in units of the response variable)
- **FEVD (Forecast Error Variance Decomposition):** Shows what percentage of forecast error variance is due to each shock (measured in percentages)

IRF tells you "how much inflation changes," FEVD tells you "how important is each shock in explaining inflation volatility." You'll learn FEVD on Day 19.

**Q8: Can I use IRFs for forecasting?**

**A:** Not directly. IRFs show responses to hypothetical shocks, not forecasts of future values. However, IRFs inform forecasts by revealing dynamic relationships. For actual forecasting, you'll use `var_result.forecast()` (covered on Day 20).

---

## Summary

Today you learned:

1. **What IRFs are:** Tools for tracing the effect of a shock through the VAR system over time
2. **Orthogonalized IRFs:** Use Cholesky decomposition with ordering MPR → TBR → EXO → INF to isolate shocks
3. **Plotting IRFs:** Created 4-panel plot for MPR shock and 4×4 grid for all shocks
4. **Confidence bands:** 95% CI tells you whether effects are statistically significant
5. **Nigerian policy transmission:** MPR affects inflation with a 6-12 month lag; TBR responds immediately; EXO effect is weak

**Next up:** Day 19 — Forecast Error Variance Decomposition (FEVD). You'll decompose inflation forecast errors to see what percentage is due to MPR shocks vs. other shocks. This answers: "How much of inflation volatility is under CBN's control?"
