# DAY 24 — Robustness Checks

## What You'll Learn Today

Robustness checks are the backbone of any credible econometric thesis. They answer the question: "Do my main results hold when I change something reasonable?" Today you'll learn:

- **Why robustness checks matter** — Reviewers and examiners will scrutinize whether your findings are fragile or robust
- **Alternative variable specifications** — Using `inflcore` instead of `infl`, or `exbdc` instead of `exo`
- **Alternative lag orders** — Testing whether optimal lags ±1 changes your conclusions
- **Sub-sample analysis** — Splitting data at structural break points (pre-2015 vs post-2015) to check stability

By the end of today, you'll have a systematic robustness framework that strengthens your thesis defense.

---

## Building econometric_models/robustness.py

We'll build this file in **3 steps**. Each step shows the **COMPLETE file** — no snippets.

---

### STEP 1: Imports, Load Full Dataset, Alternative Inflation Measure

**Goal**: Run ARDL with `inflcore` (core inflation) instead of `infl` (headline inflation) and compare coefficients.

**Why this matters**: If your main results used headline inflation (`infl`), showing that core inflation (`inflcore`) gives similar long-run coefficients proves your findings aren't driven by volatile food/energy prices.

**Delete everything in econometric_models/robustness.py and replace it with this:**

```python
"""
Robustness Checks for Nigerian Inflation ARDL Model
====================================================
Tests whether main results hold under:
1. Alternative variable specifications (inflcore, exbdc, etc.)
2. Sub-sample analysis (pre-2015 vs post-2015)
3. Alternative lag orders (±1 from optimal)

Author: Your Name
Date: Day 24
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load Full Dataset and Run Baseline + Alternative Inflation
# ============================================================================

def load_full_data():
    """
    Load the complete dataset with all 34 columns.
    Returns a DataFrame with datetime index.
    """
    df = pd.read_csv('data/processed/cleaned_data.csv', parse_dates=['date'])
    df.set_index('date', inplace=True)
    print(f"✓ Loaded {len(df)} rows with {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)[:10]}... (showing first 10)")
    return df


def run_baseline_ardl(df, dep_var='infl', exog_vars=['mpr', 'exo', 'tbr'], lags=4):
    """
    Run the baseline ARDL model (your main specification).

    Parameters:
    -----------
    df : DataFrame
        Full dataset
    dep_var : str
        Dependent variable (default 'infl')
    exog_vars : list
        Exogenous variables
    lags : int or dict
        Lag order

    Returns:
    --------
    fitted_model : ARDLResults
        Fitted ARDL model
    """
    # Drop missing values
    vars_used = [dep_var] + exog_vars
    df_clean = df[vars_used].dropna()

    print(f"\n{'='*70}")
    print(f"BASELINE ARDL: {dep_var} ~ {' + '.join(exog_vars)}")
    print(f"{'='*70}")
    print(f"Sample size: {len(df_clean)} observations")
    print(f"Lag order: {lags}")

    # Fit ARDL
    model = ARDL(endog=df_clean[dep_var],
                 exog=df_clean[exog_vars],
                 lags=lags,
                 trend='c')
    fitted = model.fit()

    # Extract long-run coefficients
    print("\n--- Long-Run Coefficients (Baseline) ---")
    try:
        lr_params = fitted.ardl_params  # Long-run parameters
        for var in exog_vars:
            if var in lr_params.index:
                coef = lr_params[var]
                print(f"  {var:8s}: {coef:8.4f}")
    except:
        print("  (Long-run coefficients not directly available)")

    # ECM term coefficient (speed of adjustment)
    try:
        ecm_coef = fitted.params['ec']
        print(f"\n  ECM coefficient: {ecm_coef:.4f}")
        print(f"  Half-life (months): {np.log(0.5) / np.log(1 + ecm_coef):.2f}")
    except:
        print("\n  ECM coefficient not available in this specification")

    return fitted


def run_alternative_inflation_ardl(df, alt_var='inflcore', exog_vars=['mpr', 'exo', 'tbr'], lags=4):
    """
    Run ARDL with alternative inflation measure (e.g., inflcore).

    Parameters:
    -----------
    df : DataFrame
        Full dataset
    alt_var : str
        Alternative dependent variable (default 'inflcore')
    exog_vars : list
        Same exogenous variables as baseline
    lags : int or dict
        Same lag order as baseline

    Returns:
    --------
    fitted_model : ARDLResults
        Fitted ARDL model
    """
    # Check if alternative variable exists
    if alt_var not in df.columns:
        print(f"\n⚠ WARNING: {alt_var} not found in dataset. Skipping.")
        return None

    # Drop missing values
    vars_used = [alt_var] + exog_vars
    df_clean = df[vars_used].dropna()

    print(f"\n{'='*70}")
    print(f"ALTERNATIVE ARDL: {alt_var} ~ {' + '.join(exog_vars)}")
    print(f"{'='*70}")
    print(f"Sample size: {len(df_clean)} observations")
    print(f"Lag order: {lags}")

    # Fit ARDL
    model = ARDL(endog=df_clean[alt_var],
                 exog=df_clean[exog_vars],
                 lags=lags,
                 trend='c')
    fitted = model.fit()

    # Extract long-run coefficients
    print("\n--- Long-Run Coefficients (Alternative) ---")
    try:
        lr_params = fitted.ardl_params
        for var in exog_vars:
            if var in lr_params.index:
                coef = lr_params[var]
                print(f"  {var:8s}: {coef:8.4f}")
    except:
        print("  (Long-run coefficients not directly available)")

    # ECM term
    try:
        ecm_coef = fitted.params['ec']
        print(f"\n  ECM coefficient: {ecm_coef:.4f}")
        print(f"  Half-life (months): {np.log(0.5) / np.log(1 + ecm_coef):.2f}")
    except:
        print("\n  ECM coefficient not available")

    return fitted


def compare_baseline_vs_alternative(baseline_model, alt_model, exog_vars):
    """
    Compare long-run coefficients between baseline and alternative specifications.
    """
    print(f"\n{'='*70}")
    print("COMPARISON: Baseline vs Alternative")
    print(f"{'='*70}")
    print(f"{'Variable':<12} {'Baseline':<12} {'Alternative':<12} {'Difference':<12}")
    print("-" * 70)

    try:
        baseline_lr = baseline_model.ardl_params
        alt_lr = alt_model.ardl_params

        for var in exog_vars:
            base_coef = baseline_lr[var] if var in baseline_lr.index else np.nan
            alt_coef = alt_lr[var] if var in alt_lr.index else np.nan
            diff = base_coef - alt_coef if not np.isnan(base_coef) and not np.isnan(alt_coef) else np.nan

            print(f"{var:<12} {base_coef:>11.4f} {alt_coef:>11.4f} {diff:>11.4f}")
    except:
        print("⚠ Could not extract long-run coefficients for comparison")


if __name__ == "__main__":
    print("="*70)
    print(" ROBUSTNESS CHECKS — STEP 1: Alternative Inflation Measure")
    print("="*70)

    # Load data
    df = load_full_data()

    # Define core variables
    exog_vars = ['mpr', 'exo', 'tbr']
    lags = 4  # Use optimal lag from Day 23

    # Run baseline ARDL (infl)
    baseline = run_baseline_ardl(df, dep_var='infl', exog_vars=exog_vars, lags=lags)

    # Run alternative ARDL (inflcore)
    alternative = run_alternative_inflation_ardl(df, alt_var='inflcore', exog_vars=exog_vars, lags=lags)

    # Compare
    if alternative is not None:
        compare_baseline_vs_alternative(baseline, alternative, exog_vars)
        print("\n✓ If coefficients are similar, your main results are ROBUST to inflation measure.")

    print("\n" + "="*70)
    print("STEP 1 COMPLETE")
    print("="*70)
```

**What we did**:
1. **`load_full_data()`** — Loads all 34 columns from `cleaned_data.csv`
2. **`run_baseline_ardl()`** — Runs your main ARDL model with `infl` as dependent variable
3. **`run_alternative_inflation_ardl()`** — Runs ARDL with `inflcore` (core inflation) instead
4. **`compare_baseline_vs_alternative()`** — Shows side-by-side comparison of long-run coefficients

**Interpretation**: If `mpr`, `exo`, and `tbr` have similar coefficients in both models, your findings are robust.

---

### STEP 2: Add Sub-Sample Analysis

**Goal**: Split data at a structural break (e.g., 2015 when CBN changed monetary policy framework) and estimate ARDL on each sub-sample.

**Why this matters**: If coefficients differ dramatically across sub-periods, it suggests regime changes. If they're similar, your model is stable over time.

**Delete everything in econometric_models/robustness.py and replace it with this:**

```python
"""
Robustness Checks for Nigerian Inflation ARDL Model
====================================================
Tests whether main results hold under:
1. Alternative variable specifications (inflcore, exbdc, etc.)
2. Sub-sample analysis (pre-2015 vs post-2015)
3. Alternative lag orders (±1 from optimal)

Author: Your Name
Date: Day 24
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load Full Dataset and Run Baseline + Alternative Inflation
# ============================================================================

def load_full_data():
    """
    Load the complete dataset with all 34 columns.
    Returns a DataFrame with datetime index.
    """
    df = pd.read_csv('data/processed/cleaned_data.csv', parse_dates=['date'])
    df.set_index('date', inplace=True)
    print(f"✓ Loaded {len(df)} rows with {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)[:10]}... (showing first 10)")
    return df


def run_baseline_ardl(df, dep_var='infl', exog_vars=['mpr', 'exo', 'tbr'], lags=4):
    """
    Run the baseline ARDL model (your main specification).

    Parameters:
    -----------
    df : DataFrame
        Full dataset
    dep_var : str
        Dependent variable (default 'infl')
    exog_vars : list
        Exogenous variables
    lags : int or dict
        Lag order

    Returns:
    --------
    fitted_model : ARDLResults
        Fitted ARDL model
    """
    # Drop missing values
    vars_used = [dep_var] + exog_vars
    df_clean = df[vars_used].dropna()

    print(f"\n{'='*70}")
    print(f"BASELINE ARDL: {dep_var} ~ {' + '.join(exog_vars)}")
    print(f"{'='*70}")
    print(f"Sample size: {len(df_clean)} observations")
    print(f"Lag order: {lags}")

    # Fit ARDL
    model = ARDL(endog=df_clean[dep_var],
                 exog=df_clean[exog_vars],
                 lags=lags,
                 trend='c')
    fitted = model.fit()

    # Extract long-run coefficients
    print("\n--- Long-Run Coefficients (Baseline) ---")
    try:
        lr_params = fitted.ardl_params  # Long-run parameters
        for var in exog_vars:
            if var in lr_params.index:
                coef = lr_params[var]
                print(f"  {var:8s}: {coef:8.4f}")
    except:
        print("  (Long-run coefficients not directly available)")

    # ECM term coefficient (speed of adjustment)
    try:
        ecm_coef = fitted.params['ec']
        print(f"\n  ECM coefficient: {ecm_coef:.4f}")
        print(f"  Half-life (months): {np.log(0.5) / np.log(1 + ecm_coef):.2f}")
    except:
        print("\n  ECM coefficient not available in this specification")

    return fitted


def run_alternative_inflation_ardl(df, alt_var='inflcore', exog_vars=['mpr', 'exo', 'tbr'], lags=4):
    """
    Run ARDL with alternative inflation measure (e.g., inflcore).

    Parameters:
    -----------
    df : DataFrame
        Full dataset
    alt_var : str
        Alternative dependent variable (default 'inflcore')
    exog_vars : list
        Same exogenous variables as baseline
    lags : int or dict
        Same lag order as baseline

    Returns:
    --------
    fitted_model : ARDLResults
        Fitted ARDL model
    """
    # Check if alternative variable exists
    if alt_var not in df.columns:
        print(f"\n⚠ WARNING: {alt_var} not found in dataset. Skipping.")
        return None

    # Drop missing values
    vars_used = [alt_var] + exog_vars
    df_clean = df[vars_used].dropna()

    print(f"\n{'='*70}")
    print(f"ALTERNATIVE ARDL: {alt_var} ~ {' + '.join(exog_vars)}")
    print(f"{'='*70}")
    print(f"Sample size: {len(df_clean)} observations")
    print(f"Lag order: {lags}")

    # Fit ARDL
    model = ARDL(endog=df_clean[alt_var],
                 exog=df_clean[exog_vars],
                 lags=lags,
                 trend='c')
    fitted = model.fit()

    # Extract long-run coefficients
    print("\n--- Long-Run Coefficients (Alternative) ---")
    try:
        lr_params = fitted.ardl_params
        for var in exog_vars:
            if var in lr_params.index:
                coef = lr_params[var]
                print(f"  {var:8s}: {coef:8.4f}")
    except:
        print("  (Long-run coefficients not directly available)")

    # ECM term
    try:
        ecm_coef = fitted.params['ec']
        print(f"\n  ECM coefficient: {ecm_coef:.4f}")
        print(f"  Half-life (months): {np.log(0.5) / np.log(1 + ecm_coef):.2f}")
    except:
        print("\n  ECM coefficient not available")

    return fitted


def compare_baseline_vs_alternative(baseline_model, alt_model, exog_vars):
    """
    Compare long-run coefficients between baseline and alternative specifications.
    """
    print(f"\n{'='*70}")
    print("COMPARISON: Baseline vs Alternative")
    print(f"{'='*70}")
    print(f"{'Variable':<12} {'Baseline':<12} {'Alternative':<12} {'Difference':<12}")
    print("-" * 70)

    try:
        baseline_lr = baseline_model.ardl_params
        alt_lr = alt_model.ardl_params

        for var in exog_vars:
            base_coef = baseline_lr[var] if var in baseline_lr.index else np.nan
            alt_coef = alt_lr[var] if var in alt_lr.index else np.nan
            diff = base_coef - alt_coef if not np.isnan(base_coef) and not np.isnan(alt_coef) else np.nan

            print(f"{var:<12} {base_coef:>11.4f} {alt_coef:>11.4f} {diff:>11.4f}")
    except:
        print("⚠ Could not extract long-run coefficients for comparison")


# ============================================================================
# STEP 2: Sub-Sample Analysis (Pre-2015 vs Post-2015)
# ============================================================================

def subsample_analysis(df, dep_var='infl', exog_vars=['mpr', 'exo', 'tbr'],
                       lags=4, breakpoint='2015-01-01'):
    """
    Split data at a structural break and estimate ARDL on each sub-sample.

    Parameters:
    -----------
    df : DataFrame
        Full dataset with datetime index
    dep_var : str
        Dependent variable
    exog_vars : list
        Exogenous variables
    lags : int
        Lag order
    breakpoint : str
        Date to split the sample (default '2015-01-01')

    Returns:
    --------
    results_dict : dict
        Contains fitted models for each sub-sample
    """
    print(f"\n{'='*70}")
    print(f"SUB-SAMPLE ANALYSIS: Split at {breakpoint}")
    print(f"{'='*70}")

    # Convert breakpoint to datetime
    breakpoint = pd.to_datetime(breakpoint)

    # Split data
    df_pre = df[df.index < breakpoint].copy()
    df_post = df[df.index >= breakpoint].copy()

    print(f"Pre-break sample:  {df_pre.index.min()} to {df_pre.index.max()} ({len(df_pre)} obs)")
    print(f"Post-break sample: {df_post.index.min()} to {df_post.index.max()} ({len(df_post)} obs)")

    results = {}

    # Estimate ARDL on pre-break sample
    print(f"\n--- PRE-BREAK ARDL ({dep_var}) ---")
    vars_used = [dep_var] + exog_vars
    df_pre_clean = df_pre[vars_used].dropna()

    if len(df_pre_clean) > 30:  # Minimum sample size check
        model_pre = ARDL(endog=df_pre_clean[dep_var],
                         exog=df_pre_clean[exog_vars],
                         lags=lags,
                         trend='c')
        fitted_pre = model_pre.fit()
        results['pre'] = fitted_pre

        print(f"Sample size: {len(df_pre_clean)}")
        try:
            lr_params_pre = fitted_pre.ardl_params
            for var in exog_vars:
                if var in lr_params_pre.index:
                    print(f"  {var:8s}: {lr_params_pre[var]:8.4f}")
            ecm_pre = fitted_pre.params['ec']
            print(f"  ECM: {ecm_pre:.4f}")
        except:
            print("  (Could not extract coefficients)")
    else:
        print("⚠ Insufficient data for pre-break estimation")
        results['pre'] = None

    # Estimate ARDL on post-break sample
    print(f"\n--- POST-BREAK ARDL ({dep_var}) ---")
    df_post_clean = df_post[vars_used].dropna()

    if len(df_post_clean) > 30:
        model_post = ARDL(endog=df_post_clean[dep_var],
                          exog=df_post_clean[exog_vars],
                          lags=lags,
                          trend='c')
        fitted_post = model_post.fit()
        results['post'] = fitted_post

        print(f"Sample size: {len(df_post_clean)}")
        try:
            lr_params_post = fitted_post.ardl_params
            for var in exog_vars:
                if var in lr_params_post.index:
                    print(f"  {var:8s}: {lr_params_post[var]:8.4f}")
            ecm_post = fitted_post.params['ec']
            print(f"  ECM: {ecm_post:.4f}")
        except:
            print("  (Could not extract coefficients)")
    else:
        print("⚠ Insufficient data for post-break estimation")
        results['post'] = None

    # Compare sub-samples
    if results['pre'] is not None and results['post'] is not None:
        print(f"\n{'='*70}")
        print("COMPARISON: Pre-Break vs Post-Break")
        print(f"{'='*70}")
        print(f"{'Variable':<12} {'Pre-Break':<12} {'Post-Break':<12} {'Difference':<12}")
        print("-" * 70)

        try:
            lr_pre = results['pre'].ardl_params
            lr_post = results['post'].ardl_params

            for var in exog_vars:
                pre_coef = lr_pre[var] if var in lr_pre.index else np.nan
                post_coef = lr_post[var] if var in lr_post.index else np.nan
                diff = pre_coef - post_coef if not np.isnan(pre_coef) and not np.isnan(post_coef) else np.nan

                print(f"{var:<12} {pre_coef:>11.4f} {post_coef:>11.4f} {diff:>11.4f}")
        except:
            print("⚠ Could not compare coefficients")

    return results


if __name__ == "__main__":
    print("="*70)
    print(" ROBUSTNESS CHECKS — STEPS 1 & 2")
    print("="*70)

    # Load data
    df = load_full_data()

    # Define core variables
    exog_vars = ['mpr', 'exo', 'tbr']
    lags = 4  # Use optimal lag from Day 23

    # ========================================================================
    # STEP 1: Alternative Inflation Measure
    # ========================================================================
    print("\n" + "="*70)
    print(" STEP 1: Alternative Inflation Measure")
    print("="*70)

    baseline = run_baseline_ardl(df, dep_var='infl', exog_vars=exog_vars, lags=lags)
    alternative = run_alternative_inflation_ardl(df, alt_var='inflcore', exog_vars=exog_vars, lags=lags)

    if alternative is not None:
        compare_baseline_vs_alternative(baseline, alternative, exog_vars)
        print("\n✓ If coefficients are similar, your main results are ROBUST to inflation measure.")

    # ========================================================================
    # STEP 2: Sub-Sample Analysis
    # ========================================================================
    print("\n" + "="*70)
    print(" STEP 2: Sub-Sample Analysis")
    print("="*70)

    subsample_results = subsample_analysis(df, dep_var='infl', exog_vars=exog_vars,
                                           lags=lags, breakpoint='2015-01-01')

    print("\n✓ If pre/post coefficients are similar, your model is STABLE over time.")

    print("\n" + "="*70)
    print("STEPS 1 & 2 COMPLETE")
    print("="*70)
```

**What we added**:
- **`subsample_analysis()`** — Splits data at a breakpoint (default: 2015-01-01), estimates ARDL on each sub-sample, and compares long-run coefficients
- **Minimum sample size check** — Ensures each sub-sample has at least 30 observations

**Interpretation**: If `mpr`, `exo`, `tbr` coefficients are similar pre/post 2015, your model is structurally stable.

---

### STEP 3: Alternative Lag Orders and Compile Robustness Table

**Goal**: Test lags ±1 from optimal (e.g., if optimal = 4, test lags 3 and 5), then save all robustness results to `results/robustness_checks.csv`.

**Delete everything in econometric_models/robustness.py and replace it with this:**

```python
"""
Robustness Checks for Nigerian Inflation ARDL Model
====================================================
Tests whether main results hold under:
1. Alternative variable specifications (inflcore, exbdc, etc.)
2. Sub-sample analysis (pre-2015 vs post-2015)
3. Alternative lag orders (±1 from optimal)

Author: Your Name
Date: Day 24
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load Full Dataset and Run Baseline + Alternative Inflation
# ============================================================================

def load_full_data():
    """
    Load the complete dataset with all 34 columns.
    Returns a DataFrame with datetime index.
    """
    df = pd.read_csv('data/processed/cleaned_data.csv', parse_dates=['date'])
    df.set_index('date', inplace=True)
    print(f"✓ Loaded {len(df)} rows with {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)[:10]}... (showing first 10)")
    return df


def run_baseline_ardl(df, dep_var='infl', exog_vars=['mpr', 'exo', 'tbr'], lags=4):
    """
    Run the baseline ARDL model (your main specification).

    Parameters:
    -----------
    df : DataFrame
        Full dataset
    dep_var : str
        Dependent variable (default 'infl')
    exog_vars : list
        Exogenous variables
    lags : int or dict
        Lag order

    Returns:
    --------
    fitted_model : ARDLResults
        Fitted ARDL model
    """
    # Drop missing values
    vars_used = [dep_var] + exog_vars
    df_clean = df[vars_used].dropna()

    print(f"\n{'='*70}")
    print(f"BASELINE ARDL: {dep_var} ~ {' + '.join(exog_vars)}")
    print(f"{'='*70}")
    print(f"Sample size: {len(df_clean)} observations")
    print(f"Lag order: {lags}")

    # Fit ARDL
    model = ARDL(endog=df_clean[dep_var],
                 exog=df_clean[exog_vars],
                 lags=lags,
                 trend='c')
    fitted = model.fit()

    # Extract long-run coefficients
    print("\n--- Long-Run Coefficients (Baseline) ---")
    try:
        lr_params = fitted.ardl_params  # Long-run parameters
        for var in exog_vars:
            if var in lr_params.index:
                coef = lr_params[var]
                print(f"  {var:8s}: {coef:8.4f}")
    except:
        print("  (Long-run coefficients not directly available)")

    # ECM term coefficient (speed of adjustment)
    try:
        ecm_coef = fitted.params['ec']
        print(f"\n  ECM coefficient: {ecm_coef:.4f}")
        print(f"  Half-life (months): {np.log(0.5) / np.log(1 + ecm_coef):.2f}")
    except:
        print("\n  ECM coefficient not available in this specification")

    return fitted


def run_alternative_inflation_ardl(df, alt_var='inflcore', exog_vars=['mpr', 'exo', 'tbr'], lags=4):
    """
    Run ARDL with alternative inflation measure (e.g., inflcore).

    Parameters:
    -----------
    df : DataFrame
        Full dataset
    alt_var : str
        Alternative dependent variable (default 'inflcore')
    exog_vars : list
        Same exogenous variables as baseline
    lags : int or dict
        Same lag order as baseline

    Returns:
    --------
    fitted_model : ARDLResults
        Fitted ARDL model
    """
    # Check if alternative variable exists
    if alt_var not in df.columns:
        print(f"\n⚠ WARNING: {alt_var} not found in dataset. Skipping.")
        return None

    # Drop missing values
    vars_used = [alt_var] + exog_vars
    df_clean = df[vars_used].dropna()

    print(f"\n{'='*70}")
    print(f"ALTERNATIVE ARDL: {alt_var} ~ {' + '.join(exog_vars)}")
    print(f"{'='*70}")
    print(f"Sample size: {len(df_clean)} observations")
    print(f"Lag order: {lags}")

    # Fit ARDL
    model = ARDL(endog=df_clean[alt_var],
                 exog=df_clean[exog_vars],
                 lags=lags,
                 trend='c')
    fitted = model.fit()

    # Extract long-run coefficients
    print("\n--- Long-Run Coefficients (Alternative) ---")
    try:
        lr_params = fitted.ardl_params
        for var in exog_vars:
            if var in lr_params.index:
                coef = lr_params[var]
                print(f"  {var:8s}: {coef:8.4f}")
    except:
        print("  (Long-run coefficients not directly available)")

    # ECM term
    try:
        ecm_coef = fitted.params['ec']
        print(f"\n  ECM coefficient: {ecm_coef:.4f}")
        print(f"  Half-life (months): {np.log(0.5) / np.log(1 + ecm_coef):.2f}")
    except:
        print("\n  ECM coefficient not available")

    return fitted


def compare_baseline_vs_alternative(baseline_model, alt_model, exog_vars):
    """
    Compare long-run coefficients between baseline and alternative specifications.
    """
    print(f"\n{'='*70}")
    print("COMPARISON: Baseline vs Alternative")
    print(f"{'='*70}")
    print(f"{'Variable':<12} {'Baseline':<12} {'Alternative':<12} {'Difference':<12}")
    print("-" * 70)

    try:
        baseline_lr = baseline_model.ardl_params
        alt_lr = alt_model.ardl_params

        for var in exog_vars:
            base_coef = baseline_lr[var] if var in baseline_lr.index else np.nan
            alt_coef = alt_lr[var] if var in alt_lr.index else np.nan
            diff = base_coef - alt_coef if not np.isnan(base_coef) and not np.isnan(alt_coef) else np.nan

            print(f"{var:<12} {base_coef:>11.4f} {alt_coef:>11.4f} {diff:>11.4f}")
    except:
        print("⚠ Could not extract long-run coefficients for comparison")


# ============================================================================
# STEP 2: Sub-Sample Analysis (Pre-2015 vs Post-2015)
# ============================================================================

def subsample_analysis(df, dep_var='infl', exog_vars=['mpr', 'exo', 'tbr'],
                       lags=4, breakpoint='2015-01-01'):
    """
    Split data at a structural break and estimate ARDL on each sub-sample.

    Parameters:
    -----------
    df : DataFrame
        Full dataset with datetime index
    dep_var : str
        Dependent variable
    exog_vars : list
        Exogenous variables
    lags : int
        Lag order
    breakpoint : str
        Date to split the sample (default '2015-01-01')

    Returns:
    --------
    results_dict : dict
        Contains fitted models for each sub-sample
    """
    print(f"\n{'='*70}")
    print(f"SUB-SAMPLE ANALYSIS: Split at {breakpoint}")
    print(f"{'='*70}")

    # Convert breakpoint to datetime
    breakpoint = pd.to_datetime(breakpoint)

    # Split data
    df_pre = df[df.index < breakpoint].copy()
    df_post = df[df.index >= breakpoint].copy()

    print(f"Pre-break sample:  {df_pre.index.min()} to {df_pre.index.max()} ({len(df_pre)} obs)")
    print(f"Post-break sample: {df_post.index.min()} to {df_post.index.max()} ({len(df_post)} obs)")

    results = {}

    # Estimate ARDL on pre-break sample
    print(f"\n--- PRE-BREAK ARDL ({dep_var}) ---")
    vars_used = [dep_var] + exog_vars
    df_pre_clean = df_pre[vars_used].dropna()

    if len(df_pre_clean) > 30:  # Minimum sample size check
        model_pre = ARDL(endog=df_pre_clean[dep_var],
                         exog=df_pre_clean[exog_vars],
                         lags=lags,
                         trend='c')
        fitted_pre = model_pre.fit()
        results['pre'] = fitted_pre

        print(f"Sample size: {len(df_pre_clean)}")
        try:
            lr_params_pre = fitted_pre.ardl_params
            for var in exog_vars:
                if var in lr_params_pre.index:
                    print(f"  {var:8s}: {lr_params_pre[var]:8.4f}")
            ecm_pre = fitted_pre.params['ec']
            print(f"  ECM: {ecm_pre:.4f}")
        except:
            print("  (Could not extract coefficients)")
    else:
        print("⚠ Insufficient data for pre-break estimation")
        results['pre'] = None

    # Estimate ARDL on post-break sample
    print(f"\n--- POST-BREAK ARDL ({dep_var}) ---")
    df_post_clean = df_post[vars_used].dropna()

    if len(df_post_clean) > 30:
        model_post = ARDL(endog=df_post_clean[dep_var],
                          exog=df_post_clean[exog_vars],
                          lags=lags,
                          trend='c')
        fitted_post = model_post.fit()
        results['post'] = fitted_post

        print(f"Sample size: {len(df_post_clean)}")
        try:
            lr_params_post = fitted_post.ardl_params
            for var in exog_vars:
                if var in lr_params_post.index:
                    print(f"  {var:8s}: {lr_params_post[var]:8.4f}")
            ecm_post = fitted_post.params['ec']
            print(f"  ECM: {ecm_post:.4f}")
        except:
            print("  (Could not extract coefficients)")
    else:
        print("⚠ Insufficient data for post-break estimation")
        results['post'] = None

    # Compare sub-samples
    if results['pre'] is not None and results['post'] is not None:
        print(f"\n{'='*70}")
        print("COMPARISON: Pre-Break vs Post-Break")
        print(f"{'='*70}")
        print(f"{'Variable':<12} {'Pre-Break':<12} {'Post-Break':<12} {'Difference':<12}")
        print("-" * 70)

        try:
            lr_pre = results['pre'].ardl_params
            lr_post = results['post'].ardl_params

            for var in exog_vars:
                pre_coef = lr_pre[var] if var in lr_pre.index else np.nan
                post_coef = lr_post[var] if var in lr_post.index else np.nan
                diff = pre_coef - post_coef if not np.isnan(pre_coef) and not np.isnan(post_coef) else np.nan

                print(f"{var:<12} {pre_coef:>11.4f} {post_coef:>11.4f} {diff:>11.4f}")
        except:
            print("⚠ Could not compare coefficients")

    return results


# ============================================================================
# STEP 3: Alternative Lag Orders
# ============================================================================

def alternative_lag_orders(df, dep_var='infl', exog_vars=['mpr', 'exo', 'tbr'],
                           optimal_lag=4):
    """
    Test alternative lag orders (optimal ±1) to check robustness.

    Parameters:
    -----------
    df : DataFrame
        Full dataset
    dep_var : str
        Dependent variable
    exog_vars : list
        Exogenous variables
    optimal_lag : int
        Optimal lag order from AIC/BIC selection

    Returns:
    --------
    results_dict : dict
        Contains fitted models for each lag order
    """
    print(f"\n{'='*70}")
    print(f"ALTERNATIVE LAG ORDERS: Testing lags {optimal_lag-1}, {optimal_lag}, {optimal_lag+1}")
    print(f"{'='*70}")

    # Prepare data
    vars_used = [dep_var] + exog_vars
    df_clean = df[vars_used].dropna()

    lag_orders = [optimal_lag - 1, optimal_lag, optimal_lag + 1]
    results = {}

    for lag in lag_orders:
        if lag < 1:
            continue

        print(f"\n--- ARDL with {lag} lags ---")

        try:
            model = ARDL(endog=df_clean[dep_var],
                         exog=df_clean[exog_vars],
                         lags=lag,
                         trend='c')
            fitted = model.fit()
            results[f'lag_{lag}'] = fitted

            print(f"AIC: {fitted.aic:.2f}")
            print(f"BIC: {fitted.bic:.2f}")

            # Long-run coefficients
            try:
                lr_params = fitted.ardl_params
                for var in exog_vars:
                    if var in lr_params.index:
                        print(f"  {var:8s}: {lr_params[var]:8.4f}")
            except:
                print("  (Could not extract long-run coefficients)")

            # ECM
            try:
                ecm_coef = fitted.params['ec']
                print(f"  ECM: {ecm_coef:.4f}")
            except:
                pass

        except Exception as e:
            print(f"⚠ Could not estimate model with {lag} lags: {e}")
            results[f'lag_{lag}'] = None

    # Compare across lag orders
    print(f"\n{'='*70}")
    print("COMPARISON: Across Lag Orders")
    print(f"{'='*70}")
    print(f"{'Variable':<12} {'Lag-1':<12} {'Optimal':<12} {'Lag+1':<12}")
    print("-" * 70)

    try:
        lr_minus = results[f'lag_{optimal_lag-1}'].ardl_params if f'lag_{optimal_lag-1}' in results and results[f'lag_{optimal_lag-1}'] else None
        lr_optimal = results[f'lag_{optimal_lag}'].ardl_params if f'lag_{optimal_lag}' in results and results[f'lag_{optimal_lag}'] else None
        lr_plus = results[f'lag_{optimal_lag+1}'].ardl_params if f'lag_{optimal_lag+1}' in results and results[f'lag_{optimal_lag+1}'] else None

        for var in exog_vars:
            coef_minus = lr_minus[var] if lr_minus is not None and var in lr_minus.index else np.nan
            coef_optimal = lr_optimal[var] if lr_optimal is not None and var in lr_optimal.index else np.nan
            coef_plus = lr_plus[var] if lr_plus is not None and var in lr_plus.index else np.nan

            print(f"{var:<12} {coef_minus:>11.4f} {coef_optimal:>11.4f} {coef_plus:>11.4f}")
    except:
        print("⚠ Could not compare coefficients across lag orders")

    return results


# ============================================================================
# STEP 4: Compile Robustness Table and Save
# ============================================================================

def compile_robustness_table(baseline_model, alt_model, subsample_results,
                             lag_results, exog_vars, optimal_lag=4):
    """
    Compile all robustness results into a single DataFrame and save to CSV.

    Parameters:
    -----------
    baseline_model : ARDLResults
        Baseline ARDL model
    alt_model : ARDLResults
        Alternative inflation model
    subsample_results : dict
        Pre/post break results
    lag_results : dict
        Alternative lag order results
    exog_vars : list
        Exogenous variables
    optimal_lag : int
        Optimal lag order

    Returns:
    --------
    robustness_df : DataFrame
        Compiled robustness table
    """
    print(f"\n{'='*70}")
    print("COMPILING ROBUSTNESS TABLE")
    print(f"{'='*70}")

    # Initialize results dictionary
    results_dict = {'Variable': exog_vars}

    # Extract baseline coefficients
    try:
        baseline_lr = baseline_model.ardl_params
        results_dict['Baseline'] = [baseline_lr[var] if var in baseline_lr.index else np.nan
                                     for var in exog_vars]
    except:
        results_dict['Baseline'] = [np.nan] * len(exog_vars)

    # Extract alternative inflation coefficients
    if alt_model is not None:
        try:
            alt_lr = alt_model.ardl_params
            results_dict['Alt_Inflation'] = [alt_lr[var] if var in alt_lr.index else np.nan
                                             for var in exog_vars]
        except:
            results_dict['Alt_Inflation'] = [np.nan] * len(exog_vars)
    else:
        results_dict['Alt_Inflation'] = [np.nan] * len(exog_vars)

    # Extract pre-break coefficients
    if subsample_results.get('pre') is not None:
        try:
            pre_lr = subsample_results['pre'].ardl_params
            results_dict['Pre_2015'] = [pre_lr[var] if var in pre_lr.index else np.nan
                                        for var in exog_vars]
        except:
            results_dict['Pre_2015'] = [np.nan] * len(exog_vars)
    else:
        results_dict['Pre_2015'] = [np.nan] * len(exog_vars)

    # Extract post-break coefficients
    if subsample_results.get('post') is not None:
        try:
            post_lr = subsample_results['post'].ardl_params
            results_dict['Post_2015'] = [post_lr[var] if var in post_lr.index else np.nan
                                         for var in exog_vars]
        except:
            results_dict['Post_2015'] = [np.nan] * len(exog_vars)
    else:
        results_dict['Post_2015'] = [np.nan] * len(exog_vars)

    # Extract lag-1 coefficients
    if lag_results.get(f'lag_{optimal_lag-1}') is not None:
        try:
            lag_minus_lr = lag_results[f'lag_{optimal_lag-1}'].ardl_params
            results_dict[f'Lag_{optimal_lag-1}'] = [lag_minus_lr[var] if var in lag_minus_lr.index else np.nan
                                                     for var in exog_vars]
        except:
            results_dict[f'Lag_{optimal_lag-1}'] = [np.nan] * len(exog_vars)
    else:
        results_dict[f'Lag_{optimal_lag-1}'] = [np.nan] * len(exog_vars)

    # Extract lag+1 coefficients
    if lag_results.get(f'lag_{optimal_lag+1}') is not None:
        try:
            lag_plus_lr = lag_results[f'lag_{optimal_lag+1}'].ardl_params
            results_dict[f'Lag_{optimal_lag+1}'] = [lag_plus_lr[var] if var in lag_plus_lr.index else np.nan
                                                     for var in exog_vars]
        except:
            results_dict[f'Lag_{optimal_lag+1}'] = [np.nan] * len(exog_vars)
    else:
        results_dict[f'Lag_{optimal_lag+1}'] = [np.nan] * len(exog_vars)

    # Create DataFrame
    robustness_df = pd.DataFrame(results_dict)

    # Save to CSV
    output_path = 'results/robustness_checks.csv'
    robustness_df.to_csv(output_path, index=False)
    print(f"\n✓ Robustness table saved to {output_path}")

    # Display table
    print("\n" + "="*70)
    print("ROBUSTNESS TABLE (Long-Run Coefficients)")
    print("="*70)
    print(robustness_df.to_string(index=False))

    return robustness_df


if __name__ == "__main__":
    print("="*70)
    print(" ROBUSTNESS CHECKS — COMPLETE ANALYSIS")
    print("="*70)

    # Load data
    df = load_full_data()

    # Define core variables
    exog_vars = ['mpr', 'exo', 'tbr']
    optimal_lag = 4  # From Day 23 lag selection

    # ========================================================================
    # STEP 1: Alternative Inflation Measure
    # ========================================================================
    print("\n" + "="*70)
    print(" STEP 1: Alternative Inflation Measure")
    print("="*70)

    baseline = run_baseline_ardl(df, dep_var='infl', exog_vars=exog_vars, lags=optimal_lag)
    alternative = run_alternative_inflation_ardl(df, alt_var='inflcore', exog_vars=exog_vars, lags=optimal_lag)

    if alternative is not None:
        compare_baseline_vs_alternative(baseline, alternative, exog_vars)

    # ========================================================================
    # STEP 2: Sub-Sample Analysis
    # ========================================================================
    print("\n" + "="*70)
    print(" STEP 2: Sub-Sample Analysis")
    print("="*70)

    subsample_results = subsample_analysis(df, dep_var='infl', exog_vars=exog_vars,
                                           lags=optimal_lag, breakpoint='2015-01-01')

    # ========================================================================
    # STEP 3: Alternative Lag Orders
    # ========================================================================
    print("\n" + "="*70)
    print(" STEP 3: Alternative Lag Orders")
    print("="*70)

    lag_results = alternative_lag_orders(df, dep_var='infl', exog_vars=exog_vars,
                                         optimal_lag=optimal_lag)

    # ========================================================================
    # STEP 4: Compile and Save Robustness Table
    # ========================================================================
    print("\n" + "="*70)
    print(" STEP 4: Compile Robustness Table")
    print("="*70)

    robustness_table = compile_robustness_table(baseline, alternative, subsample_results,
                                                lag_results, exog_vars, optimal_lag)

    print("\n" + "="*70)
    print("ALL ROBUSTNESS CHECKS COMPLETE")
    print("="*70)
    print("\n✓ Your main results are ROBUST if:")
    print("  1. Alternative inflation (inflcore) gives similar coefficients")
    print("  2. Pre/post-2015 sub-samples show stable coefficients")
    print("  3. Lag orders ±1 from optimal don't change signs or significance")
    print("\n✓ Use results/robustness_checks.csv in your thesis appendix!")
```

**What we added**:
- **`alternative_lag_orders()`** — Tests lags 3, 4, 5 (if optimal = 4) and compares long-run coefficients
- **`compile_robustness_table()`** — Aggregates all robustness results into a single DataFrame and saves to `results/robustness_checks.csv`

**Output**: A CSV table with columns: `Variable`, `Baseline`, `Alt_Inflation`, `Pre_2015`, `Post_2015`, `Lag_3`, `Lag_5`.

---

## How to Present Robustness in Your Thesis

### Table Format (Appendix or Chapter 4)

```
Table 6: Robustness Checks — Long-Run Coefficients
=====================================================================
Variable    Baseline  Alt_Infl  Pre-2015  Post-2015  Lag-1  Lag+1
---------------------------------------------------------------------
mpr          0.452     0.438     0.461     0.449     0.447   0.456
exo          1.234     1.201     1.189     1.267     1.222   1.245
tbr         -0.123    -0.118    -0.130    -0.115    -0.120  -0.126
---------------------------------------------------------------------
Note: All specifications use ARDL framework. Alt_Infl = inflcore.
Pre/Post-2015 = sub-sample analysis. Lag-1 = 3 lags, Lag+1 = 5 lags.
```

### Narrative (Chapter 4)

> "To ensure the reliability of our main findings, we conducted three robustness checks. First, we re-estimated the model using core inflation (inflcore) instead of headline inflation. The long-run coefficient on MPR remained positive and significant (0.438 vs. 0.452 in the baseline), confirming that our results are not driven by volatile food/energy prices.
>
> Second, we split the sample at 2015 to test for structural breaks following the CBN's monetary policy framework change. Pre- and post-2015 coefficients were statistically similar (MPR: 0.461 vs. 0.449), indicating model stability.
>
> Third, we tested alternative lag orders (3 and 5 lags vs. optimal 4). Coefficients remained consistent across specifications, with no sign reversals or substantial magnitude changes. These robustness checks strengthen confidence in our main conclusions."

---

## Why Robustness Matters for Your Thesis Defense

1. **Anticipates examiner questions**: "What if you used a different inflation measure?"
2. **Demonstrates thoroughness**: Shows you tested alternative reasonable specifications
3. **Strengthens credibility**: If results hold across specifications, they're not fragile
4. **Earns methodological points**: Examiners reward systematic robustness analysis

---

## Commit Your Work

```bash
cd /home/user/Nigerian_Inflation_Predictor
python econometric_models/robustness.py
git add econometric_models/robustness.py results/robustness_checks.csv
git commit -m "Day 24: Robustness checks (alt variables, sub-samples, lag orders)"
git push origin main
```

---

## Common Errors and Solutions

### Error 1: "inflcore not found in dataset"

**Cause**: Your cleaned dataset doesn't have `inflcore` column.

**Solution**: Check `data/processed/cleaned_data.csv` columns. If missing, use another alternative like `inflfood` or skip the alternative variable test.

```python
# Check available columns
df = pd.read_csv('data/processed/cleaned_data.csv')
print(df.columns)
```

### Error 2: "Insufficient data for pre-break estimation"

**Cause**: Breakpoint is too early/late, leaving too few observations in one sub-sample.

**Solution**: Change breakpoint to a date that splits data more evenly.

```python
subsample_analysis(df, breakpoint='2016-01-01')  # Try 2016 instead of 2015
```

### Error 3: "Could not extract long-run coefficients"

**Cause**: ARDL model didn't converge, or `ardl_params` attribute not available.

**Solution**: Use `fitted.params` instead of `fitted.ardl_params`, or check model diagnostics.

```python
# Alternative: extract coefficients manually
params = fitted.params
print(params)
```

### Error 4: "KeyError: 'ec'" when accessing ECM coefficient

**Cause**: ECM specification wasn't used, or error correction term not estimated.

**Solution**: Wrap ECM extraction in try-except (already done in code above), or use bounds test instead.

---

## Q&A

**Q1: How many robustness checks should I do for a thesis?**

**A:** Minimum 3:
1. Alternative variable specification (e.g., inflcore)
2. Sub-sample analysis (structural break)
3. Alternative lag orders

For top-tier theses, add:
4. Alternative estimator (e.g., DOLS, FMOLS)
5. Exclusion/inclusion of control variables
6. Outlier analysis (drop extreme observations)

---

**Q2: What if robustness checks give DIFFERENT results?**

**A:** That's valuable information! Report it honestly:
- "Results are sensitive to inflation measure, suggesting headline vs. core inflation matter"
- "Pre-2015 coefficients differ significantly, indicating a structural break"

Then discuss WHY differences exist (policy changes, external shocks, etc.). Examiners appreciate nuanced analysis.

---

**Q3: Should I use the same lag order for all robustness checks?**

**A:** Generally yes, for comparability. But in Step 3, you explicitly test different lags. For sub-sample analysis, you might re-select optimal lags for each sub-sample if sample sizes differ greatly.

---

**Q4: Can I use different exogenous variables for robustness?**

**A:** Yes! For example:
- Replace `exo` with `exbdc` (official exchange rate vs. parallel market)
- Add additional controls like `ppi` (producer price index)
- Test minimal specification (drop `tbr`) to see if main results hold

---

**Q5: How do I know if my results are "robust enough"?**

**A:** Rule of thumb:
- **Coefficients within ±20% of baseline** → Robust
- **Same sign and significance** → Robust
- **Magnitude changes >50% or sign flip** → Fragile (needs explanation)

---

**Q6: What if my dataset doesn't have 34 columns?**

**A:** Use what you have! The principle is the same:
- Test alternative dependent variables (e.g., `infl_sa` instead of `infl`)
- Test alternative exogenous variables from your available set
- The code will skip missing variables with a warning

---

**Q7: Should robustness checks go in main chapters or appendix?**

**A:** Hybrid approach:
- **Main chapter (4)**: Summarize key robustness findings (1 paragraph + 1 table)
- **Appendix**: Full robustness tables, additional specifications, diagnostic plots

---

## Tomorrow: Day 25 — Policy Implications and Forecasting

Tomorrow you'll learn:
- How to generate out-of-sample forecasts using your ARDL model
- How to simulate policy scenarios (e.g., "What if CBN raises MPR by 2%?")
- How to present policy recommendations in your thesis conclusion

---

**END OF DAY 24**
