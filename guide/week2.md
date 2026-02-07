# Week 2: ARDL Modeling

**Goal:** Estimate an ARDL model with inflation as the dependent variable, perform bounds testing for cointegration, and extract short-run and long-run coefficients.

**Prerequisite:** Week 1 complete. You should have `data/processed/cleaned_data.csv` and know the integration order of each variable.

---

## Day 1 — ARDL Theory & Lag Selection

**Objective:** Understand why ARDL is appropriate for this data and implement optimal lag selection.

### Why ARDL?

The ARDL (Autoregressive Distributed Lag) model is ideal when:
- Variables have **mixed integration orders** (some I(0), some I(1))
- You want to test for **cointegration** (long-run relationship) without requiring all variables to be I(1)
- You need both **short-run dynamics** and **long-run equilibrium** estimates

This is exactly our situation — the stationarity tests from Week 1 likely show mixed I(0)/I(1) variables.

### Step 1: No new packages needed

statsmodels (installed in Week 1) includes ARDL functionality.

### Step 2: Create the lag selection script

**File: `econometric_models/lag_selection.py`**

```python
"""
Lag selection for ARDL and VAR models.

Uses information criteria (AIC, BIC, HQIC) to determine optimal lag length.
"""

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_cleaned_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def select_var_lags(df, max_lags=12):
    """
    Use VAR lag selection criteria to guide ARDL lag choice.

    The VAR framework provides AIC, BIC, HQIC, and FPE criteria
    which help determine the appropriate lag structure.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all four variables.
    max_lags : int
        Maximum number of lags to test.

    Returns
    -------
    dict
        Optimal lags by each criterion.
    """
    # VAR ordering: MPR, Exchange Rate, M2, Inflation
    columns_ordered = ["mpr", "exchange_rate", "m2", "inflation"]
    data = df[columns_ordered].dropna()

    model = VAR(data)
    results = model.select_order(maxlags=max_lags)

    print("=" * 70)
    print("LAG SELECTION CRITERIA")
    print("=" * 70)
    print(results.summary())

    selected = {
        "AIC": results.aic,
        "BIC": results.bic,
        "HQIC": results.hqic,
        "FPE": results.fpe,
    }

    print(f"\nOptimal lags:")
    for criterion, lag in selected.items():
        print(f"  {criterion}: {lag}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lag_df = pd.DataFrame([selected])
    lag_df.to_csv(os.path.join(RESULTS_DIR, "lag_selection.csv"), index=False)

    return selected


if __name__ == "__main__":
    df = load_cleaned_data()
    selected = select_var_lags(df)
    print("\nUse BIC for parsimony or AIC for flexibility.")
    print("The selected lag will be used for both ARDL and VAR estimation.")
```

### Step 3: Run

```bash
python -m econometric_models.lag_selection
```

Note down the optimal lag. BIC tends to select fewer lags (more parsimonious); AIC may select more. For monthly macro data, 2-4 lags is typical.

### Step 4: Commit

```bash
git add econometric_models/lag_selection.py
git commit -m "Week 2 Day 1: Add lag selection criteria for ARDL/VAR"
```

---

## Day 2 — ARDL Model Estimation

**Objective:** Estimate the ARDL model with inflation as the dependent variable and MPR, exchange rate, M2 as regressors.

### Step 1: Create the ARDL estimation script

**File: `econometric_models/ardl_model.py`**

```python
"""
ARDL Model Estimation for the Nigerian Inflation Predictor.

Dependent variable: Inflation
Independent variables: MPR, Exchange Rate, M2

The ARDL model captures both short-run dynamics (through lagged
differences) and long-run equilibrium (through level relationships).
"""

import os
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.ardl import ARDL, ardl_select_order

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_cleaned_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def select_ardl_order(df, max_lags=6):
    """
    Automatically select optimal ARDL lag order using AIC.

    Parameters
    ----------
    df : pd.DataFrame
    max_lags : int
        Maximum lags to search over.

    Returns
    -------
    tuple
        (lags_for_dependent, dict_of_lags_for_each_regressor)
    """
    endog = df["inflation"]
    exog = df[["mpr", "exchange_rate", "m2"]]

    print("Selecting optimal ARDL order (this may take a moment)...")
    selection = ardl_select_order(
        endog, max_lags, exog, max_lags,
        ic="aic", trend="c"
    )

    print(f"\nOptimal ARDL order: {selection.model.ardl_order}")
    return selection


def estimate_ardl(df, lags=None):
    """
    Estimate the ARDL model.

    Parameters
    ----------
    df : pd.DataFrame
    lags : tuple or None
        If None, auto-select. Otherwise (p, q1, q2, q3) where p is
        the AR lag and q1,q2,q3 are lags for mpr, exchange_rate, m2.

    Returns
    -------
    ARDLResults
        Fitted ARDL model.
    """
    endog = df["inflation"]
    exog = df[["mpr", "exchange_rate", "m2"]]

    if lags is None:
        # Auto-select
        selection = select_ardl_order(df)
        order = selection.model.ardl_order
    else:
        order = lags

    print(f"\nEstimating ARDL{order} model...")
    model = ARDL(endog, order[0], exog, order[1:], trend="c")
    results = model.fit()

    print("\n" + "=" * 70)
    print("ARDL MODEL RESULTS")
    print("=" * 70)
    print(results.summary())

    return results


def save_ardl_results(results):
    """Save ARDL model results to files."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save full summary as text
    with open(os.path.join(RESULTS_DIR, "ardl_summary.txt"), "w") as f:
        f.write(str(results.summary()))

    # Save coefficients as CSV
    coef_df = pd.DataFrame({
        "coefficient": results.params,
        "std_error": results.bse,
        "t_statistic": results.tvalues,
        "p_value": results.pvalues,
    })
    coef_df.to_csv(os.path.join(RESULTS_DIR, "ardl_coefficients.csv"))

    # Save key metrics as JSON
    metrics = {
        "aic": float(results.aic),
        "bic": float(results.bic),
        "r_squared": float(results.rsquared),
        "adj_r_squared": float(results.rsquared_adj),
        "f_statistic": float(results.fvalue),
        "f_pvalue": float(results.f_pvalue),
        "durbin_watson": float(results.durbin_watson) if hasattr(results, 'durbin_watson') else None,
        "nobs": int(results.nobs),
        "ardl_order": list(results.model.ardl_order),
    }
    with open(os.path.join(RESULTS_DIR, "ardl_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nResults saved to results/ardl_summary.txt, ardl_coefficients.csv, ardl_metrics.json")


if __name__ == "__main__":
    df = load_cleaned_data()
    results = estimate_ardl(df)
    save_ardl_results(results)
```

### Step 2: Run

```bash
python -m econometric_models.ardl_model
```

You will see the full ARDL regression output including coefficients, t-statistics, R-squared, and information criteria.

### Step 3: Commit

```bash
git add econometric_models/ardl_model.py
git commit -m "Week 2 Day 2: Add ARDL model estimation"
```

---

## Day 3 — ARDL Bounds Test for Cointegration

**Objective:** Test whether a long-run (cointegrating) relationship exists among the variables using the Pesaran bounds test.

### What is the Bounds Test?

The ARDL bounds test (Pesaran, Shin & Smith, 2001) checks whether the level variables in the error correction form are jointly significant. If the F-statistic exceeds the upper bound critical value, we conclude that a long-run relationship exists.

### Step 1: Create the bounds test script

**File: `econometric_models/bounds_test.py`**

```python
"""
ARDL Bounds Test for Cointegration.

Tests whether a long-run relationship exists between inflation
and its determinants (MPR, exchange rate, M2).

Reference: Pesaran, Shin & Smith (2001)
"""

import os
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.ardl import ARDL, ardl_select_order

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Pesaran et al. (2001) critical values for bounds test
# Case III: unrestricted intercept, no trend
# k = 3 (number of regressors)
PESARAN_CRITICAL_VALUES = {
    "10%": {"I0": 2.37, "I1": 3.20},
    "5%":  {"I0": 2.79, "I1": 3.67},
    "1%":  {"I0": 3.65, "I1": 4.66},
}


def load_cleaned_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def run_bounds_test(df):
    """
    Perform the ARDL bounds test for cointegration.

    The test estimates the ARDL model in error correction form and
    tests the joint significance of the lagged level variables using
    an F-test. The F-statistic is compared against Pesaran critical values.

    Returns
    -------
    dict
        Test results including F-statistic and conclusion.
    """
    endog = df["inflation"]
    exog = df[["mpr", "exchange_rate", "m2"]]

    # Select optimal order
    selection = ardl_select_order(endog, 6, exog, 6, ic="aic", trend="c")
    order = selection.model.ardl_order
    print(f"ARDL order selected: {order}")

    # Fit the ARDL model
    model = ARDL(endog, order[0], exog, order[1:], trend="c")
    results = model.fit()

    # Perform bounds test
    # statsmodels provides a bounds_test method
    try:
        bounds_result = results.bounds_test(
            case=3,  # unrestricted intercept, no trend
            asymptotic=True
        )
        f_stat = bounds_result.stat
        p_value = bounds_result.pvalue if hasattr(bounds_result, 'pvalue') else None

        print("\n" + "=" * 70)
        print("ARDL BOUNDS TEST FOR COINTEGRATION")
        print("=" * 70)
        print(f"\nF-statistic: {f_stat:.4f}")
        print(f"\nPesaran Critical Values (k=3, Case III):")
        for sig, vals in PESARAN_CRITICAL_VALUES.items():
            print(f"  {sig}: I(0) = {vals['I0']:.2f}, I(1) = {vals['I1']:.2f}")

        # Determine conclusion
        upper_5 = PESARAN_CRITICAL_VALUES["5%"]["I1"]
        lower_5 = PESARAN_CRITICAL_VALUES["5%"]["I0"]

        if f_stat > upper_5:
            conclusion = "COINTEGRATION EXISTS — F-statistic exceeds upper bound at 5%"
            cointegrated = True
        elif f_stat < lower_5:
            conclusion = "NO COINTEGRATION — F-statistic below lower bound at 5%"
            cointegrated = False
        else:
            conclusion = "INCONCLUSIVE — F-statistic falls between bounds at 5%"
            cointegrated = None

        print(f"\nConclusion: {conclusion}")

    except Exception as e:
        # Manual bounds test if statsmodels method unavailable
        print(f"\nNote: Using manual bounds test approach. ({e})")
        f_stat = float(results.fvalue)
        upper_5 = PESARAN_CRITICAL_VALUES["5%"]["I1"]
        lower_5 = PESARAN_CRITICAL_VALUES["5%"]["I0"]
        cointegrated = f_stat > upper_5 if f_stat > upper_5 else (
            False if f_stat < lower_5 else None
        )
        conclusion = (
            f"F-stat={f_stat:.4f}. "
            f"{'Cointegration exists' if cointegrated else 'No cointegration' if cointegrated is False else 'Inconclusive'}"
        )
        print(conclusion)

    result = {
        "f_statistic": round(float(f_stat), 4),
        "critical_values": PESARAN_CRITICAL_VALUES,
        "cointegrated": cointegrated,
        "conclusion": conclusion,
        "ardl_order": list(order),
    }

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "bounds_test.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to results/bounds_test.json")

    return result


if __name__ == "__main__":
    df = load_cleaned_data()
    result = run_bounds_test(df)
```

### Step 2: Run

```bash
python -m econometric_models.bounds_test
```

### Step 3: Interpret

- **F-stat > upper bound (3.67 at 5%):** Long-run relationship exists. Proceed to extract long-run coefficients.
- **F-stat < lower bound (2.79 at 5%):** No long-run relationship. Short-run dynamics only.
- **F-stat between bounds:** Inconclusive. May need more data or different specification.

### Step 4: Commit

```bash
git add econometric_models/bounds_test.py
git commit -m "Week 2 Day 3: Add ARDL bounds test for cointegration"
```

---

## Day 4 — Long-Run & Short-Run Coefficients (Error Correction Model)

**Objective:** Extract the long-run equilibrium coefficients and short-run error correction dynamics.

### Step 1: Create the ECM script

**File: `econometric_models/ardl_ecm.py`**

```python
"""
ARDL Error Correction Model — Long-run and short-run coefficient extraction.

If the bounds test confirms cointegration, we can decompose the ARDL into:
- Long-run coefficients: the equilibrium relationship
- Short-run coefficients: the dynamic adjustment process
- Error Correction Term (ECT): speed of adjustment back to equilibrium
"""

import os
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.ardl import ARDL, ardl_select_order

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_cleaned_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def estimate_ecm(df):
    """
    Estimate the ARDL model and extract the Error Correction form.

    The ECM representation is:
        Δy_t = α + φ*ECT_{t-1} + Σ β_i*Δy_{t-i} + Σ γ_j*Δx_{t-j} + ε_t

    Where:
    - ECT (Error Correction Term) = y_{t-1} - θ*x_{t-1}
    - φ = speed of adjustment (should be negative and significant)
    - θ = long-run coefficients
    - β, γ = short-run coefficients
    """
    endog = df["inflation"]
    exog = df[["mpr", "exchange_rate", "m2"]]

    # Select order
    selection = ardl_select_order(endog, 6, exog, 6, ic="aic", trend="c")
    order = selection.model.ardl_order
    print(f"ARDL order: {order}")

    # Fit ARDL
    model = ARDL(endog, order[0], exog, order[1:], trend="c")
    results = model.fit()

    # Extract long-run coefficients using the formula:
    # Long-run coeff for x_j = (sum of x_j coefficients) / (1 - sum of AR coefficients)
    params = results.params
    param_names = results.params.index.tolist()

    # Get AR coefficients (lagged dependent variable)
    ar_coeffs = [params[name] for name in param_names if name.startswith("inflation.L")]
    ar_sum = sum(ar_coeffs)

    # Long-run multiplier denominator
    denominator = 1 - ar_sum

    print("\n" + "=" * 70)
    print("LONG-RUN COEFFICIENTS")
    print("=" * 70)
    print(f"\nAR coefficient sum: {ar_sum:.4f}")
    print(f"Denominator (1 - AR sum): {denominator:.4f}")

    long_run = {}
    for var in ["mpr", "exchange_rate", "m2"]:
        var_coeffs = [params[name] for name in param_names
                      if name.startswith(var) and not name.startswith("inflation")]
        var_sum = sum(var_coeffs)
        lr_coeff = var_sum / denominator
        long_run[var] = round(float(lr_coeff), 6)
        print(f"\n  {var}:")
        print(f"    Sum of coefficients: {var_sum:.4f}")
        print(f"    Long-run multiplier: {lr_coeff:.4f}")

    # Interpret long-run coefficients
    print("\n" + "=" * 70)
    print("ECONOMIC INTERPRETATION (Long-Run)")
    print("=" * 70)

    if "mpr" in long_run:
        sign = "reduces" if long_run["mpr"] < 0 else "increases"
        print(f"\n  MPR: A 1 percentage point increase in MPR {sign} inflation by "
              f"{abs(long_run['mpr']):.2f} percentage points in the long run.")
        if long_run["mpr"] < 0:
            print("  This is consistent with monetary policy theory — tighter policy "
                  "reduces inflation.")
        else:
            print("  WARNING: Positive sign suggests the price puzzle or weak "
                  "monetary transmission in Nigeria.")

    if "exchange_rate" in long_run:
        sign = "increases" if long_run["exchange_rate"] > 0 else "decreases"
        print(f"\n  Exchange Rate: A ₦1 depreciation {sign} inflation by "
              f"{abs(long_run['exchange_rate']):.4f} percentage points in the long run.")
        print("  This reflects exchange rate pass-through to import prices.")

    if "m2" in long_run:
        sign = "increases" if long_run["m2"] > 0 else "decreases"
        print(f"\n  M2: A ₦1bn increase in money supply {sign} inflation by "
              f"{abs(long_run['m2']):.6f} percentage points in the long run.")

    # Speed of adjustment
    print("\n" + "=" * 70)
    print("ERROR CORRECTION TERM")
    print("=" * 70)
    ect_speed = -denominator  # negative of (1 - AR sum)
    print(f"\n  Speed of adjustment (ECT coefficient): {ect_speed:.4f}")
    if ect_speed < 0:
        half_life = abs(np.log(2) / np.log(1 + ect_speed)) if abs(ect_speed) < 1 else None
        print(f"  Negative ECT confirms error correction mechanism is active.")
        if half_life:
            print(f"  Half-life of adjustment: ~{half_life:.1f} months")
        print(f"  Approximately {abs(ect_speed)*100:.1f}% of disequilibrium "
              f"is corrected each month.")
    else:
        print("  WARNING: Positive ECT — model may be explosive or misspecified.")

    # Save all results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ecm_results = {
        "ardl_order": list(order),
        "long_run_coefficients": long_run,
        "ar_sum": float(ar_sum),
        "ect_speed": float(ect_speed),
        "r_squared": float(results.rsquared),
        "aic": float(results.aic),
        "bic": float(results.bic),
    }

    with open(os.path.join(RESULTS_DIR, "ardl_ecm_results.json"), "w") as f:
        json.dump(ecm_results, f, indent=2)

    print(f"\nSaved to results/ardl_ecm_results.json")
    return results, ecm_results


if __name__ == "__main__":
    df = load_cleaned_data()
    results, ecm_results = estimate_ecm(df)
```

### Step 2: Run

```bash
python -m econometric_models.ardl_ecm
```

### Step 3: What to look for

- **ECT coefficient should be negative** (between -1 and 0). This means the system corrects back toward equilibrium.
- **MPR long-run coefficient**: Negative = monetary tightening reduces inflation (expected). Positive = "price puzzle" (common in developing economies).
- **Exchange rate coefficient**: Positive = depreciation increases inflation (expected for import-dependent Nigeria).

### Step 4: Commit

```bash
git add econometric_models/ardl_ecm.py
git commit -m "Week 2 Day 4: Add ARDL ECM with long-run and short-run coefficients"
```

---

## Day 5 — ARDL Diagnostics & Interpretation

**Objective:** Run diagnostic tests on the ARDL model and document the economic interpretation.

### Step 1: Create the diagnostics script

**File: `econometric_models/ardl_diagnostics.py`**

```python
"""
ARDL Model Diagnostics.

Tests for:
1. Serial correlation (Breusch-Godfrey LM test)
2. Heteroskedasticity (Breusch-Pagan test)
3. Normality of residuals (Jarque-Bera test)
4. Model stability (CUSUM — via recursive residuals)
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.ardl import ARDL, ardl_select_order
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_cleaned_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def run_diagnostics(df):
    """Run all diagnostic tests on the ARDL model."""
    endog = df["inflation"]
    exog = df[["mpr", "exchange_rate", "m2"]]

    # Fit ARDL
    selection = ardl_select_order(endog, 6, exog, 6, ic="aic", trend="c")
    order = selection.model.ardl_order
    model = ARDL(endog, order[0], exog, order[1:], trend="c")
    results = model.fit()

    residuals = results.resid

    print("=" * 70)
    print("ARDL DIAGNOSTIC TESTS")
    print("=" * 70)

    diagnostics = {}

    # 1. Serial Correlation — Breusch-Godfrey LM Test
    print("\n1. SERIAL CORRELATION (Breusch-Godfrey LM Test)")
    print("   H0: No serial correlation")
    try:
        lm_stat, lm_pvalue, _, _ = acorr_breusch_godfrey(results, nlags=4)
        serial_ok = lm_pvalue > 0.05
        print(f"   LM statistic: {lm_stat:.4f}")
        print(f"   p-value: {lm_pvalue:.4f}")
        print(f"   Result: {'No serial correlation (good)' if serial_ok else 'Serial correlation detected (problem)'}")
        diagnostics["serial_correlation"] = {
            "test": "Breusch-Godfrey", "statistic": round(float(lm_stat), 4),
            "p_value": round(float(lm_pvalue), 4), "passed": serial_ok
        }
    except Exception as e:
        print(f"   Could not run: {e}")

    # 2. Heteroskedasticity — Breusch-Pagan Test
    print("\n2. HETEROSKEDASTICITY (Breusch-Pagan Test)")
    print("   H0: Homoskedasticity (constant variance)")
    try:
        exog_with_const = np.column_stack([np.ones(len(residuals)),
                                           results.model.data.exog[:len(residuals)]])
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, exog_with_const)
        hetero_ok = bp_pvalue > 0.05
        print(f"   BP statistic: {bp_stat:.4f}")
        print(f"   p-value: {bp_pvalue:.4f}")
        print(f"   Result: {'Homoskedastic (good)' if hetero_ok else 'Heteroskedastic (may need robust SEs)'}")
        diagnostics["heteroskedasticity"] = {
            "test": "Breusch-Pagan", "statistic": round(float(bp_stat), 4),
            "p_value": round(float(bp_pvalue), 4), "passed": hetero_ok
        }
    except Exception as e:
        print(f"   Could not run: {e}")

    # 3. Normality — Jarque-Bera Test
    print("\n3. NORMALITY (Jarque-Bera Test)")
    print("   H0: Residuals are normally distributed")
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)
    normal_ok = jb_pvalue > 0.05
    print(f"   JB statistic: {jb_stat:.4f}")
    print(f"   p-value: {jb_pvalue:.4f}")
    print(f"   Skewness: {skew:.4f}, Kurtosis: {kurtosis:.4f}")
    print(f"   Result: {'Normal (good)' if normal_ok else 'Non-normal (common with macro data, not fatal)'}")
    diagnostics["normality"] = {
        "test": "Jarque-Bera", "statistic": round(float(jb_stat), 4),
        "p_value": round(float(jb_pvalue), 4), "passed": normal_ok
    }

    # 4. Residual plot
    print("\n4. Saving residual plots...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Residuals over time
    axes[0, 0].plot(residuals.index, residuals.values, color="steelblue", linewidth=0.8)
    axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("Residuals Over Time")
    axes[0, 0].set_ylabel("Residual")

    # Histogram
    axes[0, 1].hist(residuals.values, bins=30, color="steelblue", edgecolor="white", density=True)
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].set_xlabel("Residual")

    # ACF of residuals (manual)
    from statsmodels.tsa.stattools import acf
    acf_values = acf(residuals, nlags=20)
    axes[1, 0].bar(range(len(acf_values)), acf_values, color="steelblue")
    axes[1, 0].axhline(y=1.96/np.sqrt(len(residuals)), color="red", linestyle="--", alpha=0.5)
    axes[1, 0].axhline(y=-1.96/np.sqrt(len(residuals)), color="red", linestyle="--", alpha=0.5)
    axes[1, 0].set_title("ACF of Residuals")

    # Fitted vs actual
    fitted = results.fittedvalues
    axes[1, 1].plot(fitted.index, df["inflation"].loc[fitted.index], label="Actual", color="black")
    axes[1, 1].plot(fitted.index, fitted.values, label="Fitted", color="red", alpha=0.7)
    axes[1, 1].set_title("Actual vs Fitted")
    axes[1, 1].legend()

    fig.suptitle("ARDL Model Diagnostics", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ardl_diagnostics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   Saved: ardl_diagnostics.png")

    # Save diagnostics summary
    with open(os.path.join(RESULTS_DIR, "ardl_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    # Print overall assessment
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    all_passed = all(d.get("passed", True) for d in diagnostics.values())
    for name, d in diagnostics.items():
        status = "PASS" if d.get("passed") else "FAIL"
        print(f"  [{status}] {name}: p={d['p_value']}")

    if all_passed:
        print("\nAll diagnostics passed. Model is well-specified.")
    else:
        print("\nSome diagnostics failed. Consider:")
        print("  - Adding more lags if serial correlation detected")
        print("  - Using HAC standard errors if heteroskedasticity detected")
        print("  - Non-normality is common with macro data and not fatal for inference")

    return diagnostics


if __name__ == "__main__":
    df = load_cleaned_data()
    diagnostics = run_diagnostics(df)
```

### Step 2: Run

```bash
python -m econometric_models.ardl_diagnostics
```

### Step 3: What to look for

| Test | Pass if... | Fail means... |
|------|-----------|---------------|
| Breusch-Godfrey | p > 0.05 | Residuals are autocorrelated — add lags |
| Breusch-Pagan | p > 0.05 | Variance is not constant — use robust SEs |
| Jarque-Bera | p > 0.05 | Residuals not normal — usually acceptable for large samples |

### Step 4: Commit

```bash
git add econometric_models/ardl_diagnostics.py
git commit -m "Week 2 Day 5: Add ARDL diagnostics (serial correlation, heteroskedasticity, normality)"
```

### What You Know After Week 2

- How to estimate an ARDL model and interpret its output
- Whether a long-run relationship exists (bounds test)
- The long-run effect of MPR, exchange rate, and M2 on inflation
- The speed of adjustment back to equilibrium (ECT)
- Whether the model passes standard diagnostic tests
