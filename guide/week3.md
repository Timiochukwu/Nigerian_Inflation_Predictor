# Week 3: VAR Model, Impulse Response Functions & FEVD

**Goal:** Estimate a VAR model with Cholesky identification, compute IRFs and FEVD, and interpret the results in Nigerian policy context.

**Prerequisite:** Week 2 complete. ARDL model estimated and diagnosed.

---

## Day 1 — VAR Model Specification & Estimation

**Objective:** Estimate the VAR model with the correct variable ordering for Cholesky identification.

### Why VAR?

The VAR (Vector Autoregression) treats all four variables as endogenous — each depends on its own lags and the lags of all other variables. Unlike ARDL (which has one dependent variable), the VAR captures the full system of interactions.

### The Cholesky Ordering

We impose this recursive ordering:
```
MPR → Exchange Rate → M2 → Inflation
```

**Why this order?**
1. **MPR first**: The CBN sets the policy rate exogenously based on forward-looking assessments. It is not determined by the other variables within the same month.
2. **Exchange Rate second**: Asset prices respond quickly to policy signals. The Naira adjusts to MPR changes within the month.
3. **M2 third**: Money supply adjusts more slowly through bank credit channels.
4. **Inflation last**: Prices are the most sluggish — they respond to everything else but feed back only with a lag.

### Step 1: Create the VAR estimation script

**File: `econometric_models/var_model.py`**

```python
"""
VAR Model Estimation for the Nigerian Inflation Predictor.

Estimates a Vector Autoregression with Cholesky identification.
Variable ordering: MPR -> Exchange Rate -> M2 -> Inflation
"""

import os
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Cholesky ordering — this is the structural identification
VAR_ORDERING = ["mpr", "exchange_rate", "m2", "inflation"]


def load_cleaned_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df[VAR_ORDERING]


def check_stationarity_decision(df):
    """
    Check integration orders and decide whether to difference.

    For VAR, all variables should be stationary. If any are I(1),
    we difference them.
    """
    # Load integration orders from Week 1
    orders_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "integration_orders.csv"
    )
    if os.path.exists(orders_path):
        orders = pd.read_csv(orders_path)
        print("Integration orders from Week 1:")
        for _, row in orders.iterrows():
            print(f"  {row['variable']:20s} -> I({row['integration_order']})")

        needs_diff = orders["integration_order"].max() > 0
        if needs_diff:
            print("\nSome variables are I(1). Differencing for VAR estimation.")
        else:
            print("\nAll variables are I(0). VAR in levels is appropriate.")
        return needs_diff
    else:
        print("No integration orders file found. Defaulting to first differences.")
        return True


def estimate_var(df, use_differences=True):
    """
    Estimate the VAR model.

    Parameters
    ----------
    df : pd.DataFrame
        Data in Cholesky ordering.
    use_differences : bool
        If True, first-difference the data before estimation.

    Returns
    -------
    VARResults
        Fitted VAR model.
    """
    if use_differences:
        data = df.diff().dropna()
        print("\nUsing first differences for VAR estimation.")
    else:
        data = df.dropna()
        print("\nUsing levels for VAR estimation.")

    # Select optimal lag
    model = VAR(data)
    lag_selection = model.select_order(maxlags=12)
    print("\nLag selection:")
    print(lag_selection.summary())

    # Use BIC for parsimony
    optimal_lag = lag_selection.bic
    print(f"\nSelected lag (BIC): {optimal_lag}")

    # Estimate
    results = model.fit(optimal_lag)

    print("\n" + "=" * 70)
    print("VAR MODEL SUMMARY")
    print("=" * 70)
    print(results.summary())

    return results, data


def check_stability(results):
    """
    Check VAR stability — all eigenvalues of companion matrix
    should be inside the unit circle.
    """
    print("\n" + "=" * 70)
    print("STABILITY CHECK")
    print("=" * 70)

    eigenvalues = np.abs(np.linalg.eigvals(
        results.params.values[1:].T  # Companion form
    )) if hasattr(results, 'params') else []

    # Use built-in method
    is_stable = results.is_stable()
    roots = results.roots

    print(f"\nModel is stable: {is_stable}")
    print(f"\nCharacteristic roots (modulus):")
    for i, root in enumerate(sorted(np.abs(roots), reverse=True)):
        inside = "inside" if root < 1 else "OUTSIDE"
        print(f"  Root {i+1}: {root:.4f} ({inside} unit circle)")

    if is_stable:
        print("\nAll roots inside unit circle. VAR is stable and stationary.")
    else:
        print("\nWARNING: Some roots outside unit circle. Model may be unstable.")

    return is_stable


def save_var_results(results, is_stable, use_differences):
    """Save VAR results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Summary text
    with open(os.path.join(RESULTS_DIR, "var_summary.txt"), "w") as f:
        f.write(str(results.summary()))

    # Key metrics
    metrics = {
        "lag_order": results.k_ar,
        "variables": VAR_ORDERING,
        "n_obs": int(results.nobs),
        "is_stable": is_stable,
        "aic": float(results.aic),
        "bic": float(results.bic),
        "hqic": float(results.hqic),
        "used_differences": use_differences,
    }
    with open(os.path.join(RESULTS_DIR, "var_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved: var_summary.txt, var_metrics.json")


if __name__ == "__main__":
    df = load_cleaned_data()
    use_diff = check_stationarity_decision(df)
    results, data = estimate_var(df, use_differences=use_diff)
    is_stable = check_stability(results)
    save_var_results(results, is_stable, use_diff)
```

### Step 2: Run

```bash
python -m econometric_models.var_model
```

### Step 3: Key checks

- **Stability**: All characteristic roots must be inside the unit circle (modulus < 1)
- **Lag order**: BIC typically suggests 2-4 lags for monthly data
- If unstable, try fewer lags or re-check your differencing

### Step 4: Commit

```bash
git add econometric_models/var_model.py
git commit -m "Week 3 Day 1: Add VAR model estimation with stability diagnostics"
```

---

## Day 2 — Impulse Response Functions (IRFs)

**Objective:** Compute and plot IRFs showing how a +1 standard deviation shock to MPR affects all four variables over 24 months.

### Step 1: Create the IRF script

**File: `econometric_models/irf_analysis.py`**

```python
"""
Impulse Response Function analysis for the Nigerian Inflation Predictor.

Computes orthogonalized IRFs using Cholesky decomposition.
Focus: How does a shock to MPR transmit to inflation?
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

VAR_ORDERING = ["mpr", "exchange_rate", "m2", "inflation"]


def load_and_estimate_var():
    """Load data, difference if needed, estimate VAR."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)[VAR_ORDERING]

    # Check if we need differencing
    orders_path = os.path.join(RESULTS_DIR, "integration_orders.csv")
    use_diff = True
    if os.path.exists(orders_path):
        orders = pd.read_csv(orders_path)
        use_diff = orders["integration_order"].max() > 0

    data = df.diff().dropna() if use_diff else df.dropna()

    model = VAR(data)
    optimal_lag = model.select_order(maxlags=12).bic
    results = model.fit(optimal_lag)

    return results, use_diff


def compute_irfs(results, periods=24):
    """
    Compute orthogonalized IRFs using Cholesky decomposition.

    Parameters
    ----------
    results : VARResults
    periods : int
        IRF horizon in months.

    Returns
    -------
    IRAnalysis
        IRF results object.
    """
    irf = results.irf(periods=periods)
    print(f"Computed IRFs for {periods} periods")
    print(f"Ordering: {VAR_ORDERING}")
    return irf


def plot_irfs(irf, results):
    """
    Plot IRFs — focusing on the monetary policy transmission channel.

    Key plots:
    1. MPR shock -> all variables (the full transmission)
    2. MPR shock -> Inflation (the key policy question)
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    variable_labels = {
        "mpr": "MPR (%)",
        "exchange_rate": "Exchange Rate (Δ₦/$)",
        "m2": "Money Supply (ΔM2)",
        "inflation": "Inflation (%)",
    }

    # Plot 1: MPR shock -> all variables (2x2 panel)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    periods = irf.irfs.shape[0]
    x = range(periods)

    for i, var in enumerate(VAR_ORDERING):
        ax = axes[i]
        # IRF values: irf.irfs[period, response_var, shock_var]
        # Shock is MPR (index 0)
        response = irf.irfs[:, i, 0]

        # Confidence bands
        lower = irf.ci[:, i, 0, 0]  # lower bound
        upper = irf.ci[:, i, 0, 1]  # upper bound

        ax.plot(x, response, color="steelblue", linewidth=2, label="IRF")
        ax.fill_between(x, lower, upper, alpha=0.2, color="steelblue", label="95% CI")
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax.set_title(f"Response of {variable_labels[var]} to MPR Shock",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Months")
        ax.set_ylabel("Response")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Impulse Response Functions — MPR Shock (Cholesky Identification)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "irf_mpr_shock.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: irf_mpr_shock.png")

    # Plot 2: All shocks -> Inflation
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    shock_labels = {
        0: "MPR Shock",
        1: "Exchange Rate Shock",
        2: "M2 Shock",
        3: "Inflation Shock (own)",
    }

    inflation_idx = VAR_ORDERING.index("inflation")

    for shock_idx in range(4):
        ax = axes[shock_idx]
        response = irf.irfs[:, inflation_idx, shock_idx]
        lower = irf.ci[:, inflation_idx, shock_idx, 0]
        upper = irf.ci[:, inflation_idx, shock_idx, 1]

        ax.plot(x, response, color="#d62728", linewidth=2)
        ax.fill_between(x, lower, upper, alpha=0.2, color="#d62728")
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax.set_title(f"Inflation Response to {shock_labels[shock_idx]}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Months")
        ax.set_ylabel("Response")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Inflation Response to Various Shocks",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "irf_inflation_responses.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: irf_inflation_responses.png")


def interpret_irfs(irf):
    """Print economic interpretation of IRF results."""
    print("\n" + "=" * 70)
    print("IRF INTERPRETATION — Nigerian Context")
    print("=" * 70)

    # MPR shock -> Inflation
    inf_idx = VAR_ORDERING.index("inflation")
    mpr_to_inf = irf.irfs[:, inf_idx, 0]

    peak_response = mpr_to_inf[np.argmax(np.abs(mpr_to_inf))]
    peak_month = np.argmax(np.abs(mpr_to_inf))

    print(f"\n  MPR shock -> Inflation:")
    print(f"    Peak response: {peak_response:.4f} at month {peak_month}")

    if peak_response < 0:
        print(f"    A positive MPR shock REDUCES inflation, peaking at month {peak_month}.")
        print(f"    This is consistent with conventional monetary policy theory.")
    else:
        print(f"    A positive MPR shock INCREASES inflation initially (price puzzle).")
        print(f"    This is common in developing economies where:")
        print(f"    - Cost-push channels dominate (higher rates -> higher business costs)")
        print(f"    - Supply-side constraints limit the demand channel effectiveness")

    # Cumulative effect
    cumulative = np.cumsum(mpr_to_inf)
    print(f"\n    Cumulative 12-month effect: {cumulative[11]:.4f}")
    print(f"    Cumulative 24-month effect: {cumulative[-1]:.4f}")

    # MPR shock -> Exchange Rate
    exr_idx = VAR_ORDERING.index("exchange_rate")
    mpr_to_exr = irf.irfs[:, exr_idx, 0]
    peak_exr = mpr_to_exr[np.argmax(np.abs(mpr_to_exr))]

    print(f"\n  MPR shock -> Exchange Rate:")
    print(f"    Peak response: {peak_exr:.4f}")
    if peak_exr < 0:
        print(f"    Naira appreciates following MPR hike (expected — capital inflows).")
    else:
        print(f"    Naira depreciates — may reflect capital flight or credibility issues.")

    # Save interpretation data
    os.makedirs(RESULTS_DIR, exist_ok=True)
    irf_data = {
        "mpr_to_inflation": {
            "peak_response": round(float(peak_response), 4),
            "peak_month": int(peak_month),
            "cumulative_12m": round(float(cumulative[11]), 4),
            "cumulative_24m": round(float(cumulative[-1]), 4),
        },
        "mpr_to_exchange_rate": {
            "peak_response": round(float(peak_exr), 4),
        },
        "horizon": len(mpr_to_inf),
    }
    with open(os.path.join(RESULTS_DIR, "irf_interpretation.json"), "w") as f:
        json.dump(irf_data, f, indent=2)


if __name__ == "__main__":
    results, use_diff = load_and_estimate_var()
    irf = compute_irfs(results, periods=24)
    plot_irfs(irf, results)
    interpret_irfs(irf)
    print("\nIRF analysis complete.")
```

### Step 2: Run

```bash
python -m econometric_models.irf_analysis
```

### Step 3: Commit

```bash
git add econometric_models/irf_analysis.py
git commit -m "Week 3 Day 2: Add IRF analysis with Cholesky identification and plots"
```

---

## Day 3 — Forecast Error Variance Decomposition (FEVD)

**Objective:** Compute FEVD to quantify how much of inflation's forecast variance is explained by each variable.

### Step 1: Create the FEVD script

**File: `econometric_models/fevd_analysis.py`**

```python
"""
Forecast Error Variance Decomposition for the Nigerian Inflation Predictor.

Quantifies the proportion of forecast error variance in each variable
that is attributable to shocks from each variable in the system.

Key question: How much of inflation's unpredictability is driven by
MPR shocks vs exchange rate shocks vs money supply shocks?
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

VAR_ORDERING = ["mpr", "exchange_rate", "m2", "inflation"]


def load_and_estimate_var():
    """Load data and estimate VAR (same as IRF module)."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)[VAR_ORDERING]

    orders_path = os.path.join(RESULTS_DIR, "integration_orders.csv")
    use_diff = True
    if os.path.exists(orders_path):
        orders = pd.read_csv(orders_path)
        use_diff = orders["integration_order"].max() > 0

    data = df.diff().dropna() if use_diff else df.dropna()
    model = VAR(data)
    optimal_lag = model.select_order(maxlags=12).bic
    results = model.fit(optimal_lag)
    return results


def compute_fevd(results, periods=24):
    """
    Compute FEVD for all variables.

    Returns
    -------
    FEVD object from statsmodels
    """
    fevd = results.fevd(periods=periods)
    print("=" * 70)
    print("FORECAST ERROR VARIANCE DECOMPOSITION")
    print("=" * 70)
    print(fevd.summary())
    return fevd


def create_fevd_tables(fevd, periods=24):
    """Create formatted FEVD tables and save as CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Focus on inflation FEVD
    inf_idx = VAR_ORDERING.index("inflation")

    # FEVD at selected horizons
    horizons = [1, 3, 6, 12, 18, 24]
    horizons = [h for h in horizons if h <= periods]

    print("\n" + "=" * 70)
    print("FEVD FOR INFLATION AT SELECTED HORIZONS")
    print("=" * 70)
    print(f"\n{'Horizon':>8} | {'MPR':>8} | {'EXR':>8} | {'M2':>8} | {'INF':>8}")
    print("-" * 50)

    rows = []
    for h in horizons:
        decomp = fevd.decomp[h - 1, inf_idx, :]  # h-1 because 0-indexed
        row = {
            "horizon": h,
            "mpr": round(float(decomp[0]) * 100, 2),
            "exchange_rate": round(float(decomp[1]) * 100, 2),
            "m2": round(float(decomp[2]) * 100, 2),
            "inflation": round(float(decomp[3]) * 100, 2),
        }
        rows.append(row)
        print(f"{h:>8} | {row['mpr']:>7.2f}% | {row['exchange_rate']:>7.2f}% | "
              f"{row['m2']:>7.2f}% | {row['inflation']:>7.2f}%")

    fevd_df = pd.DataFrame(rows)
    fevd_df.to_csv(os.path.join(RESULTS_DIR, "fevd_inflation.csv"), index=False)
    print(f"\nSaved: fevd_inflation.csv")

    return fevd_df


def plot_fevd(fevd):
    """Plot FEVD as stacked area charts."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    periods = fevd.decomp.shape[0]
    x = range(1, periods + 1)

    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728"]
    labels = ["MPR", "Exchange Rate", "M2", "Inflation"]

    # Plot FEVD for inflation
    inf_idx = VAR_ORDERING.index("inflation")

    fig, ax = plt.subplots(figsize=(12, 6))

    decomp_data = fevd.decomp[:, inf_idx, :] * 100  # Convert to percentages
    ax.stackplot(x, decomp_data.T, labels=labels, colors=colors, alpha=0.8)
    ax.set_title("FEVD of Inflation — Contribution of Each Variable",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Forecast Horizon (months)")
    ax.set_ylabel("Percentage of Forecast Error Variance (%)")
    ax.set_xlim(1, periods)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fevd_inflation.png"), dpi=150)
    plt.close(fig)
    print("Saved: fevd_inflation.png")

    # Plot FEVD for all four variables (2x2 panel)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for var_idx, var_name in enumerate(VAR_ORDERING):
        ax = axes[var_idx]
        decomp_data = fevd.decomp[:, var_idx, :] * 100
        ax.stackplot(x, decomp_data.T, labels=labels, colors=colors, alpha=0.8)
        ax.set_title(f"FEVD of {var_name.upper()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Months")
        ax.set_ylabel("Variance Share (%)")
        ax.set_xlim(1, periods)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Forecast Error Variance Decomposition — All Variables",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fevd_all_variables.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: fevd_all_variables.png")


def interpret_fevd(fevd_df):
    """Interpret FEVD results in Nigerian context."""
    print("\n" + "=" * 70)
    print("FEVD INTERPRETATION — Nigerian Context")
    print("=" * 70)

    # Get 12-month horizon
    row_12 = fevd_df[fevd_df["horizon"] == 12].iloc[0] if 12 in fevd_df["horizon"].values else fevd_df.iloc[-1]

    print(f"\n  At the 12-month horizon:")
    print(f"    MPR explains {row_12['mpr']:.1f}% of inflation forecast variance")
    print(f"    Exchange Rate explains {row_12['exchange_rate']:.1f}%")
    print(f"    M2 explains {row_12['m2']:.1f}%")
    print(f"    Own shocks explain {row_12['inflation']:.1f}%")

    # Identify dominant driver
    drivers = {"MPR": row_12["mpr"], "Exchange Rate": row_12["exchange_rate"],
               "M2": row_12["m2"]}
    top_driver = max(drivers, key=drivers.get)

    print(f"\n  Dominant external driver: {top_driver}")

    if top_driver == "Exchange Rate":
        print("  This suggests exchange rate pass-through is the primary")
        print("  transmission channel for inflation in Nigeria — consistent")
        print("  with Nigeria's import dependence and managed float regime.")
    elif top_driver == "MPR":
        print("  Monetary policy has significant direct influence on inflation,")
        print("  suggesting the interest rate channel is active in Nigeria.")
    elif top_driver == "M2":
        print("  Money supply growth is the primary inflation driver,")
        print("  consistent with monetarist theory and fiscal dominance.")


if __name__ == "__main__":
    results = load_and_estimate_var()
    fevd = compute_fevd(results, periods=24)
    fevd_df = create_fevd_tables(fevd, periods=24)
    plot_fevd(fevd)
    interpret_fevd(fevd_df)
    print("\nFEVD analysis complete.")
```

### Step 2: Run

```bash
python -m econometric_models.fevd_analysis
```

### Step 3: Commit

```bash
git add econometric_models/fevd_analysis.py
git commit -m "Week 3 Day 3: Add FEVD analysis with tables, plots, and interpretation"
```

---

## Day 4 — VAR Diagnostics

**Objective:** Run diagnostic tests on the VAR model to validate its specification.

### Step 1: Create the VAR diagnostics script

**File: `econometric_models/var_diagnostics.py`**

```python
"""
VAR Model Diagnostics.

Tests:
1. Stability (eigenvalue check)
2. Serial correlation (Portmanteau test)
3. Normality (multivariate Jarque-Bera)
4. Granger causality (does MPR Granger-cause inflation?)
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

VAR_ORDERING = ["mpr", "exchange_rate", "m2", "inflation"]


def load_and_estimate_var():
    """Load data and estimate VAR."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)[VAR_ORDERING]

    orders_path = os.path.join(RESULTS_DIR, "integration_orders.csv")
    use_diff = True
    if os.path.exists(orders_path):
        orders = pd.read_csv(orders_path)
        use_diff = orders["integration_order"].max() > 0

    data = df.diff().dropna() if use_diff else df.dropna()
    model = VAR(data)
    optimal_lag = model.select_order(maxlags=12).bic
    results = model.fit(optimal_lag)
    return results


def run_var_diagnostics(results):
    """Run all VAR diagnostic tests."""
    print("=" * 70)
    print("VAR DIAGNOSTIC TESTS")
    print("=" * 70)

    diagnostics = {}

    # 1. Stability
    print("\n1. STABILITY")
    is_stable = results.is_stable()
    print(f"   Stable: {is_stable}")
    diagnostics["stability"] = {"passed": is_stable}

    # 2. Serial Correlation (Portmanteau / Ljung-Box)
    print("\n2. SERIAL CORRELATION (Portmanteau Test)")
    print("   H0: No serial correlation up to lag h")
    try:
        test_result = results.test_whiteness(nlags=12, signif=0.05)
        port_stat = float(test_result.test_statistic)
        port_pvalue = float(test_result.pvalue)
        serial_ok = port_pvalue > 0.05
        print(f"   Test statistic: {port_stat:.4f}")
        print(f"   p-value: {port_pvalue:.4f}")
        print(f"   Result: {'No serial correlation' if serial_ok else 'Serial correlation detected'}")
        diagnostics["serial_correlation"] = {
            "statistic": round(port_stat, 4),
            "p_value": round(port_pvalue, 4),
            "passed": serial_ok
        }
    except Exception as e:
        print(f"   Could not run: {e}")

    # 3. Normality
    print("\n3. NORMALITY (Multivariate Jarque-Bera)")
    print("   H0: Residuals are multivariate normal")
    try:
        norm_test = results.test_normality()
        norm_stat = float(norm_test.test_statistic)
        norm_pvalue = float(norm_test.pvalue)
        normal_ok = norm_pvalue > 0.05
        print(f"   Test statistic: {norm_stat:.4f}")
        print(f"   p-value: {norm_pvalue:.4f}")
        print(f"   Result: {'Normal' if normal_ok else 'Non-normal (acceptable for large samples)'}")
        diagnostics["normality"] = {
            "statistic": round(norm_stat, 4),
            "p_value": round(norm_pvalue, 4),
            "passed": normal_ok
        }
    except Exception as e:
        print(f"   Could not run: {e}")

    # 4. Granger Causality
    print("\n4. GRANGER CAUSALITY TESTS")
    print("   Does X Granger-cause Y?")
    granger_results = {}
    test_pairs = [
        ("mpr", "inflation", "Does MPR Granger-cause Inflation?"),
        ("exchange_rate", "inflation", "Does Exchange Rate Granger-cause Inflation?"),
        ("m2", "inflation", "Does M2 Granger-cause Inflation?"),
        ("inflation", "mpr", "Does Inflation Granger-cause MPR?"),
    ]

    for cause, effect, description in test_pairs:
        try:
            gc_test = results.test_causality(effect, [cause], kind="f")
            gc_stat = float(gc_test.test_statistic)
            gc_pvalue = float(gc_test.pvalue)
            causes = gc_pvalue < 0.05
            print(f"\n   {description}")
            print(f"   F-stat: {gc_stat:.4f}, p-value: {gc_pvalue:.4f}")
            print(f"   Result: {'Yes' if causes else 'No'}")
            granger_results[f"{cause}_to_{effect}"] = {
                "f_statistic": round(gc_stat, 4),
                "p_value": round(gc_pvalue, 4),
                "granger_causes": causes
            }
        except Exception as e:
            print(f"   Could not test {cause}->{effect}: {e}")

    diagnostics["granger_causality"] = granger_results

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "var_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\nSaved: var_diagnostics.json")

    return diagnostics


if __name__ == "__main__":
    results = load_and_estimate_var()
    diagnostics = run_var_diagnostics(results)
```

### Step 2: Run

```bash
python -m econometric_models.var_diagnostics
```

### Step 3: Commit

```bash
git add econometric_models/var_diagnostics.py
git commit -m "Week 3 Day 4: Add VAR diagnostics (stability, serial correlation, Granger causality)"
```

---

## Day 5 — Week 3 Review & Model Comparison

**Objective:** Create a unified model comparison summary and validate all econometric outputs.

### Step 1: Create the comparison script

**File: `econometric_models/model_summary.py`**

```python
"""
Model Comparison Summary — ARDL vs VAR findings.
"""

import os
import json
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def generate_summary():
    """Generate a comprehensive model comparison summary."""
    print("=" * 70)
    print("NIGERIAN INFLATION PREDICTOR — MODEL COMPARISON")
    print("=" * 70)

    # ARDL Results
    ardl = load_json("ardl_ecm_results.json")
    bounds = load_json("bounds_test.json")
    ardl_diag = load_json("ardl_diagnostics.json")

    print("\n1. ARDL MODEL")
    print("-" * 40)
    if ardl:
        print(f"   Order: ARDL{ardl.get('ardl_order', 'N/A')}")
        print(f"   R-squared: {ardl.get('r_squared', 'N/A'):.4f}")
        print(f"   ECT speed of adjustment: {ardl.get('ect_speed', 'N/A'):.4f}")
        print(f"   Long-run coefficients:")
        for var, coef in ardl.get("long_run_coefficients", {}).items():
            print(f"     {var}: {coef:.6f}")
    if bounds:
        print(f"   Bounds test F-stat: {bounds.get('f_statistic', 'N/A')}")
        print(f"   Cointegration: {bounds.get('conclusion', 'N/A')}")

    # VAR Results
    var_metrics = load_json("var_metrics.json")
    irf_data = load_json("irf_interpretation.json")
    var_diag = load_json("var_diagnostics.json")

    print("\n2. VAR MODEL")
    print("-" * 40)
    if var_metrics:
        print(f"   Lag order: {var_metrics.get('lag_order', 'N/A')}")
        print(f"   Stable: {var_metrics.get('is_stable', 'N/A')}")
        print(f"   Observations: {var_metrics.get('n_obs', 'N/A')}")

    if irf_data:
        mpr_inf = irf_data.get("mpr_to_inflation", {})
        print(f"   IRF — MPR shock -> Inflation:")
        print(f"     Peak response: {mpr_inf.get('peak_response', 'N/A')} "
              f"at month {mpr_inf.get('peak_month', 'N/A')}")
        print(f"     Cumulative 12m: {mpr_inf.get('cumulative_12m', 'N/A')}")

    # FEVD
    fevd_path = os.path.join(RESULTS_DIR, "fevd_inflation.csv")
    if os.path.exists(fevd_path):
        fevd_df = pd.read_csv(fevd_path)
        print("\n3. FEVD — Inflation variance at 12 months:")
        row_12 = fevd_df[fevd_df["horizon"] == 12]
        if not row_12.empty:
            row = row_12.iloc[0]
            print(f"     MPR: {row['mpr']:.1f}%")
            print(f"     Exchange Rate: {row['exchange_rate']:.1f}%")
            print(f"     M2: {row['m2']:.1f}%")
            print(f"     Own: {row['inflation']:.1f}%")

    # Diagnostics summary
    print("\n4. DIAGNOSTIC SUMMARY")
    print("-" * 40)
    if ardl_diag:
        for test, result in ardl_diag.items():
            status = "PASS" if result.get("passed") else "FAIL"
            print(f"   ARDL {test}: [{status}]")
    if var_diag:
        for test, result in var_diag.items():
            if isinstance(result, dict) and "passed" in result:
                status = "PASS" if result["passed"] else "FAIL"
                print(f"   VAR {test}: [{status}]")

    print("\n" + "=" * 70)
    print("Econometric estimation complete. Ready for simulation (Week 4).")


if __name__ == "__main__":
    generate_summary()
```

### Step 2: Run

```bash
python -m econometric_models.model_summary
```

### Step 3: Commit

```bash
git add econometric_models/model_summary.py
git commit -m "Week 3 Day 5: Add model comparison summary"
```

### What You Know After Week 3

- How VAR captures the full system dynamics
- How a monetary policy shock transmits to inflation (IRFs)
- What proportion of inflation uncertainty is driven by each variable (FEVD)
- Whether MPR Granger-causes inflation
- The relative strengths of exchange rate vs money supply channels
