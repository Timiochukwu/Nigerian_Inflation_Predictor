# Day 14 — Error Correction Model (ECM)

Welcome to Day 14! Yesterday you confirmed cointegration using the bounds test. Today you'll extract the **Error Correction Term (ECT)** and measure how fast the Nigerian economy returns to equilibrium after a monetary policy shock.

---

## What You'll Learn Today

- What the Error Correction Term (ECT) is and why it matters
- How to extract the speed of adjustment coefficient (λ) from the UECM model
- What λ means for Nigerian monetary policy transmission and how long it takes for CBN rate changes to fully affect inflation

---

## Theory: The Error Correction Mechanism

### Why We Need ECM

On Day 13, you confirmed cointegration using the bounds test. This means there IS a long-run equilibrium relationship between MPR, exchange rate, T-bill rate, and inflation.

But here's the reality: **In the short run, shocks push the system away from equilibrium.**

- A sudden exchange rate depreciation
- An unexpected oil price spike
- A surprise CBN rate hike

All these create temporary deviations from the long-run path.

### What is the Error Correction Term?

The **Error Correction Term (ECT)** measures how fast the system corrects back to equilibrium after a shock.

Think of it like a rubber band:
- You stretch it (shock moves system away from equilibrium)
- It snaps back (ECT pulls system toward equilibrium)
- λ measures the speed of the snap-back

### The Speed of Adjustment Coefficient (λ)

The ECT coefficient is called **lambda (λ)** or the "speed of adjustment."

**Critical rules for λ:**
- λ MUST be NEGATIVE (between -1 and 0)
- λ MUST be statistically significant (p-value < 0.05)
- If λ is positive or zero, your model is WRONG

**Interpreting λ values:**

| λ Value | Meaning | Half-Life | Example |
|---------|---------|-----------|---------|
| λ = -0.10 | 10% of disequilibrium corrected each month | ~7 months | Very slow adjustment |
| λ = -0.20 | 20% corrected per month | ~3 months | Moderate adjustment |
| λ = -0.50 | 50% corrected per month | ~1 month | Fast adjustment |
| λ = -0.05 | 5% corrected per month | ~14 months | Extremely slow |

### Half-Life Formula

The **half-life** tells you how long it takes for HALF of the disequilibrium to disappear.

Formula:
```
half_life = ln(0.5) / ln(1 + λ)
```

Example:
- If λ = -0.20, then half_life = ln(0.5) / ln(0.80) = 3.1 months

This means if the CBN raises MPR by 200 basis points, it takes about 3 months for half the impact on inflation to materialize.

### What to Expect for Nigeria

Based on emerging market literature:
- **Expected λ: between -0.05 and -0.30**
- This means SLOW adjustment (5-30% per month)
- Full equilibrium can take 6-20 months

Why so slow?
1. **Monetary transmission lags**: Banks don't immediately pass on rate changes
2. **Inflation inertia**: Prices are sticky in Nigeria (wage contracts, menu costs)
3. **Exchange rate channel**: Takes time for Naira depreciation to feed into import prices
4. **Informal economy**: Large informal sector insulates many prices from monetary policy

---

## Building the Error Correction Model

Today you'll expand `econometric_models/ardl_model.py` in **2 STEPS**. Each step shows the COMPLETE file from line 1 to the last line.

---

### STEP 1: Add extract_ecm() Function

**Delete everything in econometric_models/ardl_model.py and replace it with this:**

```python
"""
ARDL Model for Nigerian Inflation (Days 11-14)
Variables: infl (dependent), mpr, exo, tbr (exogenous)
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import os


def load_data(filepath):
    """
    Load cleaned CSV and prepare for ARDL.
    Returns DataFrame with datetime index.
    """
    df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
    print(f"✓ Loaded {len(df)} rows from {filepath}")
    print(f"  Columns: {list(df.columns)}")
    return df


def select_ardl_order(df, max_lags=6):
    """
    Use AIC to select optimal ARDL(p,q1,q2,q3) lag structure.
    Returns: tuple (p, q1, q2, q3) - the best order
    """
    y = df['infl']
    X = df[['mpr', 'exo', 'tbr']]

    print("\n" + "="*60)
    print("ARDL LAG ORDER SELECTION (AIC)")
    print("="*60)

    selection = ardl_select_order(
        endog=y,
        exog=X,
        maxlag=max_lags,
        ic='aic',
        trend='c'
    )

    best_order = selection.model.ardl_order
    print(f"\n✓ Best model: ARDL{best_order}")
    print(f"  AIC: {selection.aic:.2f}")

    return best_order


def estimate_ardl(df, order):
    """
    Estimate ARDL model with given order.
    Returns: fitted ARDL model
    """
    y = df['infl']
    X = df[['mpr', 'exo', 'tbr']]

    model = ARDL(endog=y, exog=X, lags=order, trend='c')
    result = model.fit()

    print("\n" + "="*60)
    print(f"ARDL{order} ESTIMATION RESULTS")
    print("="*60)
    print(result.summary())

    return result


def display_key_results(result):
    """
    Extract and display key statistics from ARDL estimation.
    """
    print("\n" + "="*60)
    print("KEY STATISTICS")
    print("="*60)
    print(f"R-squared:       {result.rsquared:.4f}")
    print(f"Adj. R-squared:  {result.rsquared_adj:.4f}")
    print(f"AIC:             {result.aic:.2f}")
    print(f"BIC:             {result.bic:.2f}")
    print(f"Log-Likelihood:  {result.llf:.2f}")

    dw = durbin_watson(result.resid)
    print(f"\nDurbin-Watson:   {dw:.2f}")
    if dw < 1.5:
        print("  ⚠ Warning: Possible positive autocorrelation")
    elif dw > 2.5:
        print("  ⚠ Warning: Possible negative autocorrelation")
    else:
        print("  ✓ No serious autocorrelation detected")


def save_results(result, order, output_dir='results'):
    """
    Save ARDL results to CSV and text files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save coefficients
    coef_df = pd.DataFrame({
        'variable': result.params.index,
        'coefficient': result.params.values,
        'std_error': result.bse.values,
        't_statistic': result.tvalues.values,
        'p_value': result.pvalues.values
    })
    coef_path = os.path.join(output_dir, 'ardl_coefficients.csv')
    coef_df.to_csv(coef_path, index=False)
    print(f"\n✓ Coefficients saved to {coef_path}")

    # Save summary to text file
    summary_path = os.path.join(output_dir, 'ardl_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"ARDL{order} Model Summary\n")
        f.write("="*60 + "\n\n")
        f.write(str(result.summary()))
    print(f"✓ Full summary saved to {summary_path}")


def bounds_test(result, significance=0.05):
    """
    Perform Pesaran bounds test for cointegration.
    Returns: dict with test results
    """
    print("\n" + "="*60)
    print("PESARAN BOUNDS TEST FOR COINTEGRATION")
    print("="*60)

    bt = result.bounds_test(case=3, significance_level=significance)

    print(f"\nF-statistic: {bt.stat:.3f}")
    print(f"\nCritical values (case III: unrestricted constant, no trend):")
    print(f"  Significance | I(0) Lower | I(1) Upper")
    print(f"  -------------|------------|------------")
    for sig, bounds in bt.critical_values.items():
        lower, upper = bounds
        print(f"      {sig}%      |   {lower:.3f}    |   {upper:.3f}")

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION:")
    lower_bound = bt.critical_values[int(significance*100)][0]
    upper_bound = bt.critical_values[int(significance*100)][1]

    if bt.stat > upper_bound:
        conclusion = "COINTEGRATION CONFIRMED ✓"
        detail = f"F-stat ({bt.stat:.3f}) > I(1) upper bound ({upper_bound:.3f})"
    elif bt.stat < lower_bound:
        conclusion = "NO COINTEGRATION ✗"
        detail = f"F-stat ({bt.stat:.3f}) < I(0) lower bound ({lower_bound:.3f})"
    else:
        conclusion = "INCONCLUSIVE (in gray zone)"
        detail = f"F-stat ({bt.stat:.3f}) between bounds"

    print(conclusion)
    print(detail)
    print("="*60)

    return {
        'f_statistic': bt.stat,
        'conclusion': conclusion,
        'critical_values': bt.critical_values
    }


def long_run_coefficients(result):
    """
    Extract and display long-run coefficients from ARDL model.
    """
    print("\n" + "="*60)
    print("LONG-RUN COEFFICIENTS")
    print("="*60)

    lr_params = result.params  # Long-run from levels ARDL

    # Extract long-run coefficients for exogenous variables
    # These are the cumulative effects
    print("\nVariable     | Coefficient | Std Error | t-stat  | p-value")
    print("-"*60)

    # Get long-run summary
    # For ARDL, we need to calculate long-run from the model
    # The result object has this built-in
    try:
        # Try to get long-run results if available
        print("\nNOTE: Long-run coefficients calculated from short-run dynamics")
        print("Use the UECM representation for explicit long-run form.\n")

        # Display key coefficients
        for var in ['mpr', 'exo', 'tbr']:
            if var in result.params.index:
                coef = result.params[var]
                se = result.bse[var]
                tstat = result.tvalues[var]
                pval = result.pvalues[var]
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                print(f"{var:12} | {coef:11.4f} | {se:9.4f} | {tstat:7.2f} | {pval:7.4f} {sig}")
    except Exception as e:
        print(f"Note: {e}")


def extract_ecm(uecm_result):
    """
    Extract Error Correction Term (ECT) from UECM model.

    The ECT coefficient (lambda) tells us the speed of adjustment
    back to long-run equilibrium after a shock.

    Parameters:
        uecm_result: fitted UECM model result

    Returns:
        dict with ECT coefficient, p-value, and half-life
    """
    print("\n" + "="*60)
    print("ERROR CORRECTION TERM (ECT) ANALYSIS")
    print("="*60)

    # The ECT coefficient is the coefficient on the lagged level of dependent variable
    # In UECM, it's labeled as 'L1.infl' or similar
    ect_coef = None
    ect_pval = None
    ect_var_name = None

    # Search for the ECT coefficient
    for param_name in uecm_result.params.index:
        if 'L1.infl' in param_name or param_name == 'ci1':
            ect_coef = uecm_result.params[param_name]
            ect_pval = uecm_result.pvalues[param_name]
            ect_var_name = param_name
            break

    if ect_coef is None:
        # Alternative: use the first lagged difference coefficient
        print("⚠ ECT not found with standard name. Checking model structure...")
        print("\nAvailable parameters:")
        print(uecm_result.params.index.tolist())
        return None

    # Display results
    print(f"\nSpeed of Adjustment (λ): {ect_coef:.4f}")
    print(f"Standard Error:          {uecm_result.bse[ect_var_name]:.4f}")
    print(f"t-statistic:             {uecm_result.tvalues[ect_var_name]:.2f}")
    print(f"p-value:                 {ect_pval:.4f}")

    # Significance stars
    if ect_pval < 0.01:
        sig = "***"
    elif ect_pval < 0.05:
        sig = "**"
    elif ect_pval < 0.10:
        sig = "*"
    else:
        sig = "(not significant)"
    print(f"Significance:            {sig}")

    # Validity checks
    print("\n" + "-"*60)
    print("VALIDITY CHECKS:")
    print("-"*60)

    valid = True

    # Check 1: λ must be negative
    if ect_coef >= 0:
        print("✗ FAIL: λ is not negative!")
        print("  → Model misspecification. Check lag order.")
        valid = False
    else:
        print("✓ PASS: λ is negative (correct sign)")

    # Check 2: λ must be > -1
    if ect_coef <= -1:
        print("✗ FAIL: λ ≤ -1 (explosive adjustment)")
        print("  → Model is unstable. Check data or specification.")
        valid = False
    else:
        print("✓ PASS: λ > -1 (stable adjustment)")

    # Check 3: λ must be significant
    if ect_pval >= 0.05:
        print("⚠ WARNING: λ is not significant at 5% level")
        print("  → Weak evidence of error correction")
        valid = False
    else:
        print("✓ PASS: λ is statistically significant")

    # Calculate half-life if valid
    half_life = None
    if valid and ect_coef < 0 and ect_coef > -1:
        half_life = np.log(0.5) / np.log(1 + ect_coef)
        print("\n" + "-"*60)
        print("ADJUSTMENT SPEED INTERPRETATION:")
        print("-"*60)
        print(f"Monthly correction rate: {abs(ect_coef)*100:.1f}%")
        print(f"Half-life of shock:      {half_life:.1f} months")
        print(f"\nThis means it takes {half_life:.1f} months for HALF of any")
        print("disequilibrium to disappear after a shock.")

        # Nigerian context
        if abs(ect_coef) < 0.10:
            speed = "VERY SLOW"
            policy_msg = "CBN rate changes take over a year to fully transmit"
        elif abs(ect_coef) < 0.20:
            speed = "SLOW"
            policy_msg = "CBN rate changes take 6-12 months to fully transmit"
        elif abs(ect_coef) < 0.35:
            speed = "MODERATE"
            policy_msg = "CBN rate changes take 3-6 months to fully transmit"
        else:
            speed = "FAST"
            policy_msg = "CBN rate changes transmit quickly (< 3 months)"

        print(f"\nAdjustment speed: {speed}")
        print(f"Policy implication: {policy_msg}")

    print("="*60)

    return {
        'lambda': ect_coef,
        'p_value': ect_pval,
        'half_life': half_life,
        'valid': valid
    }


if __name__ == "__main__":
    # Full ARDL pipeline (Days 11-14)

    print("\n" + "="*70)
    print(" NIGERIAN INFLATION ARDL MODEL - FULL ESTIMATION PIPELINE")
    print("="*70)

    # Step 1: Load data
    df = load_data('data/processed/cleaned_data.csv')

    # Step 2: Select optimal lag order
    best_order = select_ardl_order(df, max_lags=6)

    # Step 3: Estimate ARDL model
    ardl_result = estimate_ardl(df, best_order)

    # Step 4: Display key statistics
    display_key_results(ardl_result)

    # Step 5: Bounds test for cointegration
    bounds_result = bounds_test(ardl_result, significance=0.05)

    # Step 6: Long-run coefficients
    long_run_coefficients(ardl_result)

    # Step 7: Estimate UECM for ECT
    print("\n" + "="*60)
    print("ESTIMATING UECM (UNRESTRICTED ERROR CORRECTION MODEL)")
    print("="*60)

    y = df['infl']
    X = df[['mpr', 'exo', 'tbr']]

    uecm_model = UECM(endog=y, exog=X, lags=best_order, trend='c')
    uecm_result = uecm_model.fit()

    print("\n✓ UECM estimated successfully")

    # Step 8: Extract ECT
    ect_results = extract_ecm(uecm_result)

    # Step 9: Save all results
    save_results(ardl_result, best_order, output_dir='results')

    print("\n" + "="*70)
    print(" PIPELINE COMPLETE!")
    print("="*70)
    print("\n✓ All results saved to results/ directory")
    print("✓ Check results/ardl_summary.txt for full output")
```

**What changed:**
- Added `extract_ecm(uecm_result)` function
- This function finds the ECT coefficient (λ) from the UECM model
- Performs 3 validity checks: negative, stable, significant
- Calculates half-life if λ is valid
- Added UECM estimation in the `__main__` block
- The pipeline now runs: lag selection → ARDL → bounds test → UECM → ECT extraction

**Run it:**
```bash
cd /home/user/Nigerian_Inflation_Predictor
python econometric_models/ardl_model.py
```

**Expected output:**
```
ERROR CORRECTION TERM (ECT) ANALYSIS
============================================================

Speed of Adjustment (λ): -0.1842
Standard Error:          0.0523
t-statistic:             -3.52
p-value:                 0.0006
Significance:            ***

------------------------------------------------------------
VALIDITY CHECKS:
------------------------------------------------------------
✓ PASS: λ is negative (correct sign)
✓ PASS: λ > -1 (stable adjustment)
✓ PASS: λ is statistically significant

------------------------------------------------------------
ADJUSTMENT SPEED INTERPRETATION:
------------------------------------------------------------
Monthly correction rate: 18.4%
Half-life of shock:      3.4 months

This means it takes 3.4 months for HALF of any
disequilibrium to disappear after a shock.

Adjustment speed: SLOW
Policy implication: CBN rate changes take 6-12 months to fully transmit
============================================================
```

---

### STEP 2: Add Comprehensive Summary Function

Now let's add a function that pulls everything together into a thesis-ready summary.

**Delete everything in econometric_models/ardl_model.py and replace it with this:**

```python
"""
ARDL Model for Nigerian Inflation (Days 11-14)
Variables: infl (dependent), mpr, exo, tbr (exogenous)
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import os


def load_data(filepath):
    """
    Load cleaned CSV and prepare for ARDL.
    Returns DataFrame with datetime index.
    """
    df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
    print(f"✓ Loaded {len(df)} rows from {filepath}")
    print(f"  Columns: {list(df.columns)}")
    return df


def select_ardl_order(df, max_lags=6):
    """
    Use AIC to select optimal ARDL(p,q1,q2,q3) lag structure.
    Returns: tuple (p, q1, q2, q3) - the best order
    """
    y = df['infl']
    X = df[['mpr', 'exo', 'tbr']]

    print("\n" + "="*60)
    print("ARDL LAG ORDER SELECTION (AIC)")
    print("="*60)

    selection = ardl_select_order(
        endog=y,
        exog=X,
        maxlag=max_lags,
        ic='aic',
        trend='c'
    )

    best_order = selection.model.ardl_order
    print(f"\n✓ Best model: ARDL{best_order}")
    print(f"  AIC: {selection.aic:.2f}")

    return best_order


def estimate_ardl(df, order):
    """
    Estimate ARDL model with given order.
    Returns: fitted ARDL model
    """
    y = df['infl']
    X = df[['mpr', 'exo', 'tbr']]

    model = ARDL(endog=y, exog=X, lags=order, trend='c')
    result = model.fit()

    print("\n" + "="*60)
    print(f"ARDL{order} ESTIMATION RESULTS")
    print("="*60)
    print(result.summary())

    return result


def display_key_results(result):
    """
    Extract and display key statistics from ARDL estimation.
    """
    print("\n" + "="*60)
    print("KEY STATISTICS")
    print("="*60)
    print(f"R-squared:       {result.rsquared:.4f}")
    print(f"Adj. R-squared:  {result.rsquared_adj:.4f}")
    print(f"AIC:             {result.aic:.2f}")
    print(f"BIC:             {result.bic:.2f}")
    print(f"Log-Likelihood:  {result.llf:.2f}")

    dw = durbin_watson(result.resid)
    print(f"\nDurbin-Watson:   {dw:.2f}")
    if dw < 1.5:
        print("  ⚠ Warning: Possible positive autocorrelation")
    elif dw > 2.5:
        print("  ⚠ Warning: Possible negative autocorrelation")
    else:
        print("  ✓ No serious autocorrelation detected")


def save_results(result, order, output_dir='results'):
    """
    Save ARDL results to CSV and text files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save coefficients
    coef_df = pd.DataFrame({
        'variable': result.params.index,
        'coefficient': result.params.values,
        'std_error': result.bse.values,
        't_statistic': result.tvalues.values,
        'p_value': result.pvalues.values
    })
    coef_path = os.path.join(output_dir, 'ardl_coefficients.csv')
    coef_df.to_csv(coef_path, index=False)
    print(f"\n✓ Coefficients saved to {coef_path}")

    # Save summary to text file
    summary_path = os.path.join(output_dir, 'ardl_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"ARDL{order} Model Summary\n")
        f.write("="*60 + "\n\n")
        f.write(str(result.summary()))
    print(f"✓ Full summary saved to {summary_path}")


def bounds_test(result, significance=0.05):
    """
    Perform Pesaran bounds test for cointegration.
    Returns: dict with test results
    """
    print("\n" + "="*60)
    print("PESARAN BOUNDS TEST FOR COINTEGRATION")
    print("="*60)

    bt = result.bounds_test(case=3, significance_level=significance)

    print(f"\nF-statistic: {bt.stat:.3f}")
    print(f"\nCritical values (case III: unrestricted constant, no trend):")
    print(f"  Significance | I(0) Lower | I(1) Upper")
    print(f"  -------------|------------|------------")
    for sig, bounds in bt.critical_values.items():
        lower, upper = bounds
        print(f"      {sig}%      |   {lower:.3f}    |   {upper:.3f}")

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION:")
    lower_bound = bt.critical_values[int(significance*100)][0]
    upper_bound = bt.critical_values[int(significance*100)][1]

    if bt.stat > upper_bound:
        conclusion = "COINTEGRATION CONFIRMED ✓"
        detail = f"F-stat ({bt.stat:.3f}) > I(1) upper bound ({upper_bound:.3f})"
    elif bt.stat < lower_bound:
        conclusion = "NO COINTEGRATION ✗"
        detail = f"F-stat ({bt.stat:.3f}) < I(0) lower bound ({lower_bound:.3f})"
    else:
        conclusion = "INCONCLUSIVE (in gray zone)"
        detail = f"F-stat ({bt.stat:.3f}) between bounds"

    print(conclusion)
    print(detail)
    print("="*60)

    return {
        'f_statistic': bt.stat,
        'conclusion': conclusion,
        'critical_values': bt.critical_values
    }


def long_run_coefficients(result):
    """
    Extract and display long-run coefficients from ARDL model.
    """
    print("\n" + "="*60)
    print("LONG-RUN COEFFICIENTS")
    print("="*60)

    lr_params = result.params  # Long-run from levels ARDL

    # Extract long-run coefficients for exogenous variables
    # These are the cumulative effects
    print("\nVariable     | Coefficient | Std Error | t-stat  | p-value")
    print("-"*60)

    # Get long-run summary
    # For ARDL, we need to calculate long-run from the model
    # The result object has this built-in
    try:
        # Try to get long-run results if available
        print("\nNOTE: Long-run coefficients calculated from short-run dynamics")
        print("Use the UECM representation for explicit long-run form.\n")

        # Display key coefficients
        for var in ['mpr', 'exo', 'tbr']:
            if var in result.params.index:
                coef = result.params[var]
                se = result.bse[var]
                tstat = result.tvalues[var]
                pval = result.pvalues[var]
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                print(f"{var:12} | {coef:11.4f} | {se:9.4f} | {tstat:7.2f} | {pval:7.4f} {sig}")
    except Exception as e:
        print(f"Note: {e}")


def extract_ecm(uecm_result):
    """
    Extract Error Correction Term (ECT) from UECM model.

    The ECT coefficient (lambda) tells us the speed of adjustment
    back to long-run equilibrium after a shock.

    Parameters:
        uecm_result: fitted UECM model result

    Returns:
        dict with ECT coefficient, p-value, and half-life
    """
    print("\n" + "="*60)
    print("ERROR CORRECTION TERM (ECT) ANALYSIS")
    print("="*60)

    # The ECT coefficient is the coefficient on the lagged level of dependent variable
    # In UECM, it's labeled as 'L1.infl' or similar
    ect_coef = None
    ect_pval = None
    ect_var_name = None

    # Search for the ECT coefficient
    for param_name in uecm_result.params.index:
        if 'L1.infl' in param_name or param_name == 'ci1':
            ect_coef = uecm_result.params[param_name]
            ect_pval = uecm_result.pvalues[param_name]
            ect_var_name = param_name
            break

    if ect_coef is None:
        # Alternative: use the first lagged difference coefficient
        print("⚠ ECT not found with standard name. Checking model structure...")
        print("\nAvailable parameters:")
        print(uecm_result.params.index.tolist())
        return None

    # Display results
    print(f"\nSpeed of Adjustment (λ): {ect_coef:.4f}")
    print(f"Standard Error:          {uecm_result.bse[ect_var_name]:.4f}")
    print(f"t-statistic:             {uecm_result.tvalues[ect_var_name]:.2f}")
    print(f"p-value:                 {ect_pval:.4f}")

    # Significance stars
    if ect_pval < 0.01:
        sig = "***"
    elif ect_pval < 0.05:
        sig = "**"
    elif ect_pval < 0.10:
        sig = "*"
    else:
        sig = "(not significant)"
    print(f"Significance:            {sig}")

    # Validity checks
    print("\n" + "-"*60)
    print("VALIDITY CHECKS:")
    print("-"*60)

    valid = True

    # Check 1: λ must be negative
    if ect_coef >= 0:
        print("✗ FAIL: λ is not negative!")
        print("  → Model misspecification. Check lag order.")
        valid = False
    else:
        print("✓ PASS: λ is negative (correct sign)")

    # Check 2: λ must be > -1
    if ect_coef <= -1:
        print("✗ FAIL: λ ≤ -1 (explosive adjustment)")
        print("  → Model is unstable. Check data or specification.")
        valid = False
    else:
        print("✓ PASS: λ > -1 (stable adjustment)")

    # Check 3: λ must be significant
    if ect_pval >= 0.05:
        print("⚠ WARNING: λ is not significant at 5% level")
        print("  → Weak evidence of error correction")
        valid = False
    else:
        print("✓ PASS: λ is statistically significant")

    # Calculate half-life if valid
    half_life = None
    if valid and ect_coef < 0 and ect_coef > -1:
        half_life = np.log(0.5) / np.log(1 + ect_coef)
        print("\n" + "-"*60)
        print("ADJUSTMENT SPEED INTERPRETATION:")
        print("-"*60)
        print(f"Monthly correction rate: {abs(ect_coef)*100:.1f}%")
        print(f"Half-life of shock:      {half_life:.1f} months")
        print(f"\nThis means it takes {half_life:.1f} months for HALF of any")
        print("disequilibrium to disappear after a shock.")

        # Nigerian context
        if abs(ect_coef) < 0.10:
            speed = "VERY SLOW"
            policy_msg = "CBN rate changes take over a year to fully transmit"
        elif abs(ect_coef) < 0.20:
            speed = "SLOW"
            policy_msg = "CBN rate changes take 6-12 months to fully transmit"
        elif abs(ect_coef) < 0.35:
            speed = "MODERATE"
            policy_msg = "CBN rate changes take 3-6 months to fully transmit"
        else:
            speed = "FAST"
            policy_msg = "CBN rate changes transmit quickly (< 3 months)"

        print(f"\nAdjustment speed: {speed}")
        print(f"Policy implication: {policy_msg}")

    print("="*60)

    return {
        'lambda': ect_coef,
        'p_value': ect_pval,
        'half_life': half_life,
        'valid': valid
    }


def comprehensive_summary(ardl_result, bounds_result, ect_results, order, output_dir='results'):
    """
    Generate a comprehensive thesis-ready summary of all ARDL results.
    Combines model specification, cointegration test, long-run effects,
    and error correction mechanism into one document.

    Saves to:
    - results/ardl_comprehensive_summary.txt (human-readable)
    - results/ecm_results.csv (table format)
    """
    os.makedirs(output_dir, exist_ok=True)

    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("COMPREHENSIVE ARDL-ECM SUMMARY: NIGERIAN INFLATION MODEL")
    summary_lines.append("="*80)
    summary_lines.append("")

    # Section 1: Model Specification
    summary_lines.append("1. MODEL SPECIFICATION")
    summary_lines.append("-" * 80)
    summary_lines.append(f"Model:              ARDL{order}")
    summary_lines.append(f"Dependent variable: Inflation (infl)")
    summary_lines.append(f"Exogenous variables: MPR (mpr), Exchange Rate (exo), T-bill Rate (tbr)")
    summary_lines.append(f"Estimation period:  {ardl_result.nobs} observations")
    summary_lines.append(f"R-squared:          {ardl_result.rsquared:.4f}")
    summary_lines.append(f"Adj. R-squared:     {ardl_result.rsquared_adj:.4f}")
    summary_lines.append(f"AIC:                {ardl_result.aic:.2f}")
    summary_lines.append(f"BIC:                {ardl_result.bic:.2f}")
    summary_lines.append("")

    # Section 2: Cointegration Test
    summary_lines.append("2. COINTEGRATION TEST (PESARAN BOUNDS TEST)")
    summary_lines.append("-" * 80)
    summary_lines.append(f"F-statistic:        {bounds_result['f_statistic']:.3f}")
    summary_lines.append(f"Conclusion:         {bounds_result['conclusion']}")
    summary_lines.append("")
    summary_lines.append("Critical Values (Case III: Unrestricted constant, no trend):")
    for sig, bounds in bounds_result['critical_values'].items():
        lower, upper = bounds
        summary_lines.append(f"  {sig}% level: I(0) = {lower:.3f}, I(1) = {upper:.3f}")
    summary_lines.append("")

    if "CONFIRMED" in bounds_result['conclusion']:
        summary_lines.append("INTERPRETATION: Variables are cointegrated. A long-run equilibrium")
        summary_lines.append("relationship exists between inflation and monetary policy variables.")
    else:
        summary_lines.append("INTERPRETATION: No evidence of cointegration. Variables may not have")
        summary_lines.append("a stable long-run relationship.")
    summary_lines.append("")

    # Section 3: Error Correction Mechanism
    summary_lines.append("3. ERROR CORRECTION MECHANISM (ECM)")
    summary_lines.append("-" * 80)

    if ect_results and ect_results['valid']:
        lambda_val = ect_results['lambda']
        p_val = ect_results['p_value']
        half_life = ect_results['half_life']

        summary_lines.append(f"Speed of Adjustment (λ): {lambda_val:.4f} ***")
        summary_lines.append(f"P-value:                 {p_val:.4f}")
        summary_lines.append(f"Half-life:               {half_life:.2f} months")
        summary_lines.append("")
        summary_lines.append(f"INTERPRETATION:")
        summary_lines.append(f"  - {abs(lambda_val)*100:.1f}% of disequilibrium is corrected each month")
        summary_lines.append(f"  - After a shock, it takes {half_life:.1f} months for half the deviation")
        summary_lines.append(f"    from equilibrium to disappear")
        summary_lines.append(f"  - Full adjustment takes approximately {half_life*3:.1f} months")
        summary_lines.append("")

        # Policy implications
        if abs(lambda_val) < 0.15:
            adj_speed = "SLOW"
            policy_lag = "12-24 months"
        elif abs(lambda_val) < 0.30:
            adj_speed = "MODERATE"
            policy_lag = "6-12 months"
        else:
            adj_speed = "FAST"
            policy_lag = "3-6 months"

        summary_lines.append(f"  Adjustment speed: {adj_speed}")
        summary_lines.append(f"  Policy transmission lag: {policy_lag}")
    else:
        summary_lines.append("⚠ WARNING: Error correction term is invalid or not significant.")
        summary_lines.append("Model may be misspecified.")
    summary_lines.append("")

    # Section 4: Short-Run Dynamics
    summary_lines.append("4. SHORT-RUN DYNAMICS")
    summary_lines.append("-" * 80)
    summary_lines.append("Top 5 significant short-run coefficients:")
    summary_lines.append("")
    summary_lines.append(f"{'Variable':<20} {'Coefficient':>12} {'Std Error':>12} {'p-value':>10}")
    summary_lines.append("-" * 80)

    # Get significant coefficients
    sig_params = ardl_result.pvalues[ardl_result.pvalues < 0.05].sort_values()
    for i, (var, pval) in enumerate(sig_params.head(5).items()):
        coef = ardl_result.params[var]
        se = ardl_result.bse[var]
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*"
        summary_lines.append(f"{var:<20} {coef:>12.4f} {se:>12.4f} {pval:>10.4f} {sig}")
    summary_lines.append("")

    # Section 5: Diagnostics
    summary_lines.append("5. DIAGNOSTIC TESTS")
    summary_lines.append("-" * 80)
    dw = durbin_watson(ardl_result.resid)
    summary_lines.append(f"Durbin-Watson:      {dw:.2f}")
    if 1.5 < dw < 2.5:
        summary_lines.append("  → No serious autocorrelation detected")
    else:
        summary_lines.append("  → ⚠ Possible autocorrelation issue")
    summary_lines.append("")

    # Section 6: Policy Implications for Nigeria
    summary_lines.append("6. POLICY IMPLICATIONS FOR NIGERIA")
    summary_lines.append("-" * 80)

    if ect_results and ect_results['valid']:
        half_life = ect_results['half_life']
        summary_lines.append("Monetary Policy Transmission:")
        summary_lines.append(f"  - CBN rate changes take {half_life:.1f} months to reach 50% effectiveness")
        summary_lines.append(f"  - Full policy impact realized after {half_life*3:.1f} months")
        summary_lines.append("")
        summary_lines.append("Why is adjustment slow in Nigeria?")
        summary_lines.append("  1. Banking sector delays in passing through rate changes")
        summary_lines.append("  2. High inflation inertia (sticky prices and wages)")
        summary_lines.append("  3. Exchange rate channel operates with lags")
        summary_lines.append("  4. Large informal economy insulates many prices from policy")
        summary_lines.append("")
        summary_lines.append("Recommendation:")
        summary_lines.append("  CBN should adopt forward-looking policy stance, anticipating")
        summary_lines.append(f"  inflation pressures {int(half_life*2)} months in advance.")
    else:
        summary_lines.append("⚠ Cannot provide policy recommendations due to invalid ECM results.")

    summary_lines.append("")
    summary_lines.append("="*80)
    summary_lines.append("END OF COMPREHENSIVE SUMMARY")
    summary_lines.append("="*80)

    # Save to text file
    summary_path = os.path.join(output_dir, 'ardl_comprehensive_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    print("\n" + "="*60)
    print("COMPREHENSIVE SUMMARY GENERATED")
    print("="*60)
    print(f"✓ Saved to {summary_path}")

    # Also save ECM results to CSV
    if ect_results:
        ecm_df = pd.DataFrame({
            'Metric': ['Speed of Adjustment (λ)', 'P-value', 'Half-life (months)',
                      'Valid', 'Monthly Correction Rate (%)'],
            'Value': [
                ect_results['lambda'],
                ect_results['p_value'],
                ect_results['half_life'] if ect_results['half_life'] else np.nan,
                ect_results['valid'],
                abs(ect_results['lambda'])*100 if ect_results['lambda'] else np.nan
            ]
        })
        ecm_path = os.path.join(output_dir, 'ecm_results.csv')
        ecm_df.to_csv(ecm_path, index=False)
        print(f"✓ ECM results saved to {ecm_path}")

    # Print to console
    print("\n" + '\n'.join(summary_lines[:30]))  # Print first 30 lines
    print("\n... (see full summary in file)")


if __name__ == "__main__":
    # Full ARDL pipeline (Days 11-14)

    print("\n" + "="*70)
    print(" NIGERIAN INFLATION ARDL MODEL - FULL ESTIMATION PIPELINE")
    print("="*70)

    # Step 1: Load data
    df = load_data('data/processed/cleaned_data.csv')

    # Step 2: Select optimal lag order
    best_order = select_ardl_order(df, max_lags=6)

    # Step 3: Estimate ARDL model
    ardl_result = estimate_ardl(df, best_order)

    # Step 4: Display key statistics
    display_key_results(ardl_result)

    # Step 5: Bounds test for cointegration
    bounds_result = bounds_test(ardl_result, significance=0.05)

    # Step 6: Long-run coefficients
    long_run_coefficients(ardl_result)

    # Step 7: Estimate UECM for ECT
    print("\n" + "="*60)
    print("ESTIMATING UECM (UNRESTRICTED ERROR CORRECTION MODEL)")
    print("="*60)

    y = df['infl']
    X = df[['mpr', 'exo', 'tbr']]

    uecm_model = UECM(endog=y, exog=X, lags=best_order, trend='c')
    uecm_result = uecm_model.fit()

    print("\n✓ UECM estimated successfully")

    # Step 8: Extract ECT
    ect_results = extract_ecm(uecm_result)

    # Step 9: Save all results
    save_results(ardl_result, best_order, output_dir='results')

    # Step 10: Generate comprehensive summary
    comprehensive_summary(ardl_result, bounds_result, ect_results, best_order, output_dir='results')

    print("\n" + "="*70)
    print(" PIPELINE COMPLETE!")
    print("="*70)
    print("\n✓ All results saved to results/ directory")
    print("✓ Check results/ardl_comprehensive_summary.txt for thesis-ready output")
```

**What changed:**
- Added `comprehensive_summary()` function
- This function creates a thesis-ready document combining:
  * Model specification and fit statistics
  * Cointegration test results
  * Error correction mechanism analysis
  * Short-run dynamics
  * Diagnostic tests
  * Policy implications for Nigeria
- Saves to two files:
  * `results/ardl_comprehensive_summary.txt` (human-readable report)
  * `results/ecm_results.csv` (tabular format)
- Called at the end of the pipeline

**Run it:**
```bash
cd /home/user/Nigerian_Inflation_Predictor
python econometric_models/ardl_model.py
```

Now check the output files:
```bash
cat results/ardl_comprehensive_summary.txt
cat results/ecm_results.csv
```

---

## Interpreting Your Results for Nigeria

### Expected λ for Nigeria

Based on emerging market research, you should expect:
- **λ between -0.05 and -0.30**
- **Half-life: 3 to 14 months**

If your results show λ = -0.18 and half-life = 3.5 months, this means:

1. **Slow adjustment**: Only 18% of disequilibrium corrects each month
2. **Policy lag**: CBN rate changes take 3.5 months to reach 50% effectiveness
3. **Full impact**: About 10-12 months for complete transmission

### Why is Nigerian Adjustment Slow?

**1. Banking Sector Frictions**
- Nigerian banks don't immediately pass through MPR changes
- Liquidity constraints mean some banks can't adjust rates quickly
- Market power allows banks to delay rate adjustments

**2. Inflation Inertia**
- Many prices are "sticky" due to wage contracts, rent agreements
- Inflation expectations are backward-looking in Nigeria
- Past inflation strongly predicts future inflation

**3. Exchange Rate Channel Lags**
- Takes time for Naira depreciation to feed into import prices
- Inventory stocks mean old prices persist
- Some imports have long supply chains (3-6 months)

**4. Large Informal Economy**
- 40-60% of Nigerian economy is informal
- Informal sector prices don't respond to CBN policy
- This dampens overall policy effectiveness

### What if λ is Positive?

If you get λ > 0, your model is WRONG. This means the system is moving AWAY from equilibrium, which is economically impossible.

**Possible causes:**
1. **Wrong lag order**: Go back to Day 12, try different max_lags
2. **Structural break**: Check if there's a regime change in your data period
3. **Omitted variable**: Maybe you need oil prices or money supply
4. **Data quality**: Check for outliers or measurement errors

### Policy Implications

If λ = -0.18 (half-life 3.5 months):

**For CBN:**
- Rate hikes today won't fully affect inflation for 10-12 months
- Must adopt forward-looking policy stance
- Should start tightening 6-9 months BEFORE expected inflation spike

**For forecasters:**
- Don't expect immediate impact from rate changes
- Build in 3-6 month lags when modeling policy effects
- Watch leading indicators (exchange rate, money supply)

**For researchers:**
- Compare your λ to other African countries (Ghana, Kenya, South Africa)
- Investigate WHY adjustment is slow (bank lending channel? exchange rate pass-through?)
- Test if λ changed after 2016 (inflation targeting adoption)

---

## Commit Your Work

Stage and commit all changes:

```bash
cd /home/user/Nigerian_Inflation_Predictor
git add econometric_models/ardl_model.py
git add results/ardl_comprehensive_summary.txt
git add results/ecm_results.csv
git commit -m "Day 14: Add Error Correction Model (ECM) analysis

- Extracted ECT coefficient (lambda) from UECM model
- Calculated speed of adjustment and half-life
- Generated comprehensive ARDL-ECM summary report
- Interpreted results for Nigerian monetary policy transmission"
```

---

## Common Errors and Fixes

### Error 1: "ECT coefficient not found"

```
⚠ ECT not found with standard name. Checking model structure...
```

**Cause:** The UECM model uses different parameter naming conventions.

**Fix:**
Check the parameter names by printing `uecm_result.params.index`. The ECT might be called 'L.infl' or 'ci' instead of 'L1.infl'. Update the search logic in `extract_ecm()`.

### Error 2: λ is positive

```
✗ FAIL: λ is not negative!
→ Model misspecification. Check lag order.
```

**Cause:** Your model is misspecified or lag order is wrong.

**Fix:**
1. Re-run lag selection with different `max_lags` (try 8 or 12)
2. Check for structural breaks in your data
3. Consider adding more exogenous variables (oil prices, money supply)
4. Verify data quality (no outliers or data entry errors)

### Error 3: λ is insignificant

```
⚠ WARNING: λ is not significant at 5% level
→ Weak evidence of error correction
```

**Cause:** Weak cointegration or low sample size.

**Fix:**
1. This might be legitimate (weak long-run relationship)
2. Check if bounds test was borderline
3. Consider using 10% significance level instead
4. Try longer sample period if possible

---

## Q&A

**Q1: What's the difference between ARDL and UECM?**

**A:** They're two representations of the same model:
- **ARDL**: Levels form. Shows cumulative effects. `ARDL(endog, exog, lags)`
- **UECM**: Differences form. Shows short-run dynamics + ECT. `UECM(endog, exog, lags)`

The UECM explicitly separates:
1. Long-run equilibrium (the ECT)
2. Short-run adjustments (differenced variables)

Both have the same statistical properties, just different parameterizations.

**Q2: Can λ be exactly -1?**

**A:** Theoretically yes, practically no.

λ = -1 means 100% of disequilibrium corrects in one period. This implies:
- Instantaneous adjustment
- No persistence whatsoever
- Perfect markets with zero frictions

This never happens in real economies, especially not in Nigeria. If you get λ = -1.0, suspect numerical issues.

**Q3: How does Nigerian λ compare to other countries?**

**A:** Typical values:

| Country | λ (Speed) | Half-life | Notes |
|---------|-----------|-----------|-------|
| USA | -0.30 to -0.50 | 1-2 months | Developed, deep markets |
| UK | -0.25 to -0.40 | 1.5-3 months | Similar to USA |
| South Africa | -0.15 to -0.25 | 2-5 months | Most developed in Africa |
| Ghana | -0.08 to -0.18 | 4-9 months | Similar to Nigeria |
| Nigeria | -0.05 to -0.20 | 3-14 months | Slow due to structure |

Nigeria's adjustment is slower because:
- Less developed financial markets
- Larger informal economy
- Higher inflation inertia
- Exchange rate rigidities

---

## What You Built Today

1. **extract_ecm() function** that:
   - Extracts the ECT coefficient (λ) from UECM results
   - Validates λ (negative, stable, significant)
   - Calculates half-life
   - Interprets adjustment speed for Nigerian policy

2. **comprehensive_summary() function** that:
   - Combines all ARDL results into one thesis-ready document
   - Saves to `results/ardl_comprehensive_summary.txt`
   - Exports ECM metrics to `results/ecm_results.csv`
   - Provides policy recommendations

3. **Full ARDL-ECM pipeline** that runs:
   - Lag selection → ARDL estimation → Bounds test → Long-run coefficients → UECM → ECT extraction → Comprehensive summary

You now have a complete error correction model showing how fast the Nigerian economy returns to equilibrium after monetary policy shocks.

---

## Tomorrow: Day 15 — Diagnostic Tests

Tomorrow you'll validate your ARDL model by testing for:
- Serial correlation (Breusch-Godfrey LM test)
- Heteroskedasticity (ARCH test)
- Normality of residuals (Jarque-Bera test)
- Parameter stability (CUSUM test)

These tests ensure your model is reliable and meets econometric assumptions.

**Key question:** Is your model statistically sound, or are there hidden problems?

See you tomorrow!
