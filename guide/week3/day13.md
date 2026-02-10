# Week 3, Day 13 — ARDL Bounds Test & Long-Run Coefficients

## What You'll Learn Today

- The Pesaran-Shin-Smith (2001) bounds test for cointegration
- What the F-statistic means relative to I(0) and I(1) bounds
- How to extract long-run coefficients from the UECM (Unrestricted Error Correction Model)
- What the long-run coefficients mean for Nigerian monetary policy

## Why This Matters

On Day 11 you learned the theory behind ARDL and selected the optimal lag order for each variable. On Day 12 you estimated the ARDL model itself and confirmed that the specification was statistically sound. But fitting a model is not enough. You need to answer the central econometric question: **do inflation, MPR, Treasury Bill Rate, and the exchange rate actually share a long-run equilibrium relationship?**

The bounds test answers that question. If the answer is yes -- if cointegration exists -- then the long-run coefficients tell you the exact nature of the equilibrium. They tell you how much inflation moves when MPR, TBR, or the exchange rate changes permanently. Those numbers are the headline results of your thesis. An examiner will look at your long-run coefficients table before anything else.

Today you will add the bounds test and long-run coefficient extraction to your ARDL script, run both, and save the results to CSV files that you will reference in your final write-up.

---

## No New Packages

Everything you need is already in statsmodels. The `UECM` class (Unrestricted Error Correction Model) and its `bounds_test` method have been available since Day 11. No changes to `requirements.txt`.

---

## Theory: The Bounds Test

This is the most important hypothesis test in your entire ARDL analysis. Read this section carefully.

### The Central Question

You have four variables: `infl` (inflation, the dependent variable), `mpr` (Monetary Policy Rate), `tbr` (Treasury Bill Rate), and `exo` (Naira/USD exchange rate). All four are I(1) -- non-stationary in levels, stationary in first differences. The question is: **do these variables share a long-run equilibrium?**

If they do, it means that even though each variable wanders on its own (because they are I(1)), they are tied together by an invisible thread. When one drifts too far from the equilibrium, forces pull the system back. This "pulling back" is what the Error Correction Model captures. But first, you need to prove the thread exists. That is what the bounds test does.

### The UECM Form

The bounds test is based on the UECM (Unrestricted Error Correction Model) representation of the ARDL model. The UECM rewrites the ARDL equation so that it contains two types of terms:

1. **Lagged level terms:** `infl_(t-1)`, `mpr_(t-1)`, `tbr_(t-1)`, `exo_(t-1)`. These capture the long-run relationship. If there is no long-run relationship, the coefficients on these terms will all be zero.

2. **First-difference terms:** `delta_infl_(t-1)`, `delta_mpr_t`, `delta_tbr_t`, `delta_exo_t`, and their lags. These capture the short-run dynamics -- how the variables respond to recent changes.

The UECM form looks like this in general notation:

```
delta_infl_t = c + phi*infl_(t-1) + theta1*mpr_(t-1) + theta2*tbr_(t-1) + theta3*exo_(t-1)
               + [lagged differences of infl, mpr, tbr, exo]
               + error_t
```

The bounds test focuses on the lagged level terms: `phi`, `theta1`, `theta2`, `theta3`.

### The Hypotheses

- **Null hypothesis (H0):** phi = theta1 = theta2 = theta3 = 0. All the lagged level coefficients are zero. There is NO long-run relationship. The variables may move together in the short run, but they have no equilibrium pulling them back when they diverge.

- **Alternative hypothesis (H1):** At least one of the lagged level coefficients is non-zero. A long-run relationship EXISTS.

### The F-Statistic and the Two Bounds

The test computes an F-statistic for the joint significance of the lagged level terms. This F-statistic is then compared against TWO sets of critical values, not one. This is what makes the bounds test unique.

- **Lower bound (I(0) bound):** Assumes all the regressors (mpr, tbr, exo) are I(0) -- stationary in levels. This is the most favourable scenario for finding cointegration.

- **Upper bound (I(1) bound):** Assumes all the regressors are I(1) -- non-stationary. This is the least favourable scenario.

The decision rule has three possible outcomes:

| Condition | Decision |
|-----------|----------|
| F-statistic > upper bound | **Reject H0.** Cointegration EXISTS. This holds regardless of whether the regressors are I(0) or I(1). |
| F-statistic < lower bound | **Fail to reject H0.** No cointegration. The variables do not share a long-run equilibrium. |
| Lower bound < F-statistic < upper bound | **Inconclusive.** The test cannot decide. You need additional evidence (like the Johansen test from Day 9). |

The critical values are tabulated at standard significance levels: 10%, 5%, 2.5%, and 1%. For your thesis, the 5% level is the primary benchmark. If the F-statistic exceeds the upper bound at 5%, you can confidently state that cointegration exists.

### Case 3: Unrestricted Constant

The bounds test comes in five "cases" depending on how the constant and trend are treated. We use **Case 3: unrestricted intercept, no trend**. This means the model includes a constant that is not restricted to lie within the cointegrating relationship. Case 3 is the standard choice for macroeconomic data that has no deterministic time trend in the long-run equilibrium.

### What Long-Run Coefficients Are

Once the bounds test confirms cointegration, you extract the **long-run coefficients**. These tell you the equilibrium relationship between the variables. Specifically, they answer: "If MPR permanently increases by 1 percentage point and everything else stays constant, what is the eventual long-run change in inflation once the system reaches its new equilibrium?"

The long-run coefficients are computed from the UECM parameters. The statsmodels `UECM` class provides them directly through the `.long_run` attribute, so you do not need to compute them by hand.

---

## Building `econometric_models/ardl_model.py` -- Step by Step

You are expanding the file you built on Days 11 and 12. At each step, we show you the **complete file** from the first line to the last line. Delete everything in the file and replace it with exactly what is shown.

---

### Build Step 1: Add the Bounds Test Function

Delete everything in `econometric_models/ardl_model.py` and replace it with this:

```python
import os
import pandas as pd
from statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

DEPENDENT = "infl"
EXOG_VARS = ["mpr", "tbr", "exo"]


def load_data():
    """Load the cleaned dataset with a monthly frequency index."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    df.index.freq = "MS"
    return df


def select_ardl_order(df, max_lags=6, ic="aic"):
    """Select optimal ARDL lag order using an information criterion."""
    y = df[DEPENDENT]
    X = df[EXOG_VARS]
    sel = ardl_select_order(y, max_lags, X, max_lags, ic=ic, trend="c")
    return sel


def fit_ardl(df, p, q_dict):
    """
    Fit the ARDL model in levels using the selected lag order.

    Parameters
    ----------
    df : DataFrame with columns infl, mpr, tbr, exo.
    p  : int, number of lags of the dependent variable (infl).
    q_dict : dict mapping each exogenous variable name to its lag order,
             e.g. {"mpr": 2, "tbr": 1, "exo": 3}.

    Returns
    -------
    result : ARDLResults, the fitted model.
    """
    y = df[DEPENDENT]
    X = df[EXOG_VARS]
    order = [q_dict[var] for var in EXOG_VARS]

    model = ARDL(y, lags=p, exog=X, order=order, trend="c")
    result = model.fit()
    return result


def bounds_test(df, p, q_dict):
    """
    Perform the Pesaran-Shin-Smith bounds test for cointegration.
    Uses the UECM (Unrestricted Error Correction Model) form.

    Parameters
    ----------
    df : DataFrame with columns infl, mpr, tbr, exo.
    p  : int, lags of the dependent variable.
    q_dict : dict mapping each exogenous variable to its lag order.

    Returns
    -------
    uecm_result : the fitted UECM model.
    bounds : the bounds test result object.
    """
    y = df[DEPENDENT]
    X = df[EXOG_VARS]
    order = [q_dict[var] for var in EXOG_VARS]

    uecm = UECM(y, lags=p, exog=X, order=order)
    uecm_result = uecm.fit()

    bounds = uecm_result.bounds_test(case=3)

    print("=" * 70)
    print("PESARAN-SHIN-SMITH BOUNDS TEST")
    print("=" * 70)
    print(f"\nF-statistic: {bounds.stat:.4f}")
    print(f"\nCritical Values:")
    print(f"  {'Significance':<15} {'I(0) Bound':>12} {'I(1) Bound':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12}")
    for sig, (i0, i1) in bounds.critical_values.items():
        print(f"  {sig:<15} {i0:>12.4f} {i1:>12.4f}")

    # Determine conclusion at 5% significance
    five_pct = bounds.critical_values["5%"]
    upper_5 = five_pct[1]
    lower_5 = five_pct[0]

    print(f"\nConclusion at 5% significance:")
    if bounds.stat > upper_5:
        print(f"  F = {bounds.stat:.4f} > upper bound {upper_5:.4f}")
        print(f"  REJECT the null. Cointegration EXISTS.")
    elif bounds.stat < lower_5:
        print(f"  F = {bounds.stat:.4f} < lower bound {lower_5:.4f}")
        print(f"  FAIL TO REJECT the null. No cointegration.")
    else:
        print(f"  {lower_5:.4f} < F = {bounds.stat:.4f} < {upper_5:.4f}")
        print(f"  INCONCLUSIVE. Use Johansen results as backup evidence.")

    return uecm_result, bounds


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows  |  Frequency: {df.index.freq}")

    # --- Lag selection (from Day 11/12) ---
    sel = select_ardl_order(df, max_lags=6, ic="aic")
    order = sel.model.ardl_order
    p = order[0]
    q_dict = {var: order[i + 1] for i, var in enumerate(EXOG_VARS)}
    print(f"\nSelected ARDL order (AIC): ARDL{order}")

    # --- Bounds test ---
    uecm_result, bounds = bounds_test(df, p, q_dict)
```

**Run it:**

```bash
python -m econometric_models.ardl_model
```

**What you should see:**

```
Loaded 300 rows  |  Frequency: MS

Selected ARDL order (AIC): ARDL(X, Y, Z, W)

======================================================================
PESARAN-SHIN-SMITH BOUNDS TEST
======================================================================

F-statistic: N.NNNN

Critical Values:
  Significance       I(0) Bound   I(1) Bound
  --------------- ------------ ------------
  10%                   X.XXXX       X.XXXX
  5%                    X.XXXX       X.XXXX
  2.5%                  X.XXXX       X.XXXX
  1%                    X.XXXX       X.XXXX

Conclusion at 5% significance:
  F = N.NNNN > upper bound X.XXXX
  REJECT the null. Cointegration EXISTS.
```

(Your numbers will differ. The F-statistic and critical values depend on your data.)

**What just happened -- function by function:**

- `bounds_test(df, p, q_dict)` -- This is the new function. It creates a UECM (Unrestricted Error Correction Model) using the same lag structure as your ARDL model from Day 12, fits it, and then runs the bounds test with `case=3` (unrestricted constant, no trend).

- `UECM(y, lags=p, exog=X, order=order)` -- Creates the UECM representation of the ARDL model. The UECM is mathematically equivalent to the ARDL model -- it is just written in a different form that separates the long-run level terms from the short-run difference terms. The `lags` and `order` arguments are the same as for the ARDL model.

- `uecm_result.bounds_test(case=3)` -- Runs the Pesaran-Shin-Smith bounds test. The `case=3` argument specifies "unrestricted intercept, no trend." The result object contains the F-statistic and a dictionary of critical values at standard significance levels.

- `bounds.critical_values` -- A dictionary where keys are significance levels (like `"5%"`) and values are tuples of `(lower_bound, upper_bound)`. The loop prints each significance level with both bounds, so you can see the full picture.

- The conclusion block compares the F-statistic against the 5% critical values. If F exceeds the upper bound, cointegration is confirmed. If F falls below the lower bound, there is no cointegration. If F falls between the bounds, the test is inconclusive.

If that ran and printed the bounds test results, move on.

---

### Build Step 2: Add Long-Run Coefficients Extraction

Delete everything in `econometric_models/ardl_model.py` and replace it with this:

```python
import os
import pandas as pd
from statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

DEPENDENT = "infl"
EXOG_VARS = ["mpr", "tbr", "exo"]


def load_data():
    """Load the cleaned dataset with a monthly frequency index."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    df.index.freq = "MS"
    return df


def select_ardl_order(df, max_lags=6, ic="aic"):
    """Select optimal ARDL lag order using an information criterion."""
    y = df[DEPENDENT]
    X = df[EXOG_VARS]
    sel = ardl_select_order(y, max_lags, X, max_lags, ic=ic, trend="c")
    return sel


def fit_ardl(df, p, q_dict):
    """
    Fit the ARDL model in levels using the selected lag order.

    Parameters
    ----------
    df : DataFrame with columns infl, mpr, tbr, exo.
    p  : int, number of lags of the dependent variable (infl).
    q_dict : dict mapping each exogenous variable name to its lag order,
             e.g. {"mpr": 2, "tbr": 1, "exo": 3}.

    Returns
    -------
    result : ARDLResults, the fitted model.
    """
    y = df[DEPENDENT]
    X = df[EXOG_VARS]
    order = [q_dict[var] for var in EXOG_VARS]

    model = ARDL(y, lags=p, exog=X, order=order, trend="c")
    result = model.fit()
    return result


def bounds_test(df, p, q_dict):
    """
    Perform the Pesaran-Shin-Smith bounds test for cointegration.
    Uses the UECM (Unrestricted Error Correction Model) form.

    Parameters
    ----------
    df : DataFrame with columns infl, mpr, tbr, exo.
    p  : int, lags of the dependent variable.
    q_dict : dict mapping each exogenous variable to its lag order.

    Returns
    -------
    uecm_result : the fitted UECM model.
    bounds : the bounds test result object.
    """
    y = df[DEPENDENT]
    X = df[EXOG_VARS]
    order = [q_dict[var] for var in EXOG_VARS]

    uecm = UECM(y, lags=p, exog=X, order=order)
    uecm_result = uecm.fit()

    bounds = uecm_result.bounds_test(case=3)

    print("=" * 70)
    print("PESARAN-SHIN-SMITH BOUNDS TEST")
    print("=" * 70)
    print(f"\nF-statistic: {bounds.stat:.4f}")
    print(f"\nCritical Values:")
    print(f"  {'Significance':<15} {'I(0) Bound':>12} {'I(1) Bound':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12}")
    for sig, (i0, i1) in bounds.critical_values.items():
        print(f"  {sig:<15} {i0:>12.4f} {i1:>12.4f}")

    five_pct = bounds.critical_values["5%"]
    upper_5 = five_pct[1]
    lower_5 = five_pct[0]

    print(f"\nConclusion at 5% significance:")
    if bounds.stat > upper_5:
        print(f"  F = {bounds.stat:.4f} > upper bound {upper_5:.4f}")
        print(f"  REJECT the null. Cointegration EXISTS.")
    elif bounds.stat < lower_5:
        print(f"  F = {bounds.stat:.4f} < lower bound {lower_5:.4f}")
        print(f"  FAIL TO REJECT the null. No cointegration.")
    else:
        print(f"  {lower_5:.4f} < F = {bounds.stat:.4f} < {upper_5:.4f}")
        print(f"  INCONCLUSIVE. Use Johansen results as backup evidence.")

    return uecm_result, bounds


def long_run_coefficients(uecm_result):
    """
    Extract and interpret long-run coefficients from the fitted UECM.

    The long-run coefficients represent the equilibrium relationship:
    a permanent 1-unit increase in an exogenous variable is associated
    with a change of [coefficient] units in inflation in the long run.

    Parameters
    ----------
    uecm_result : the fitted UECM result from bounds_test().

    Returns
    -------
    lr_df : DataFrame with coefficient, std error, t-stat, p-value.
    """
    lr = uecm_result.long_run

    print("\n" + "=" * 70)
    print("LONG-RUN COEFFICIENTS")
    print("=" * 70)

    names = EXOG_VARS
    coeffs = lr.params
    std_errs = lr.bse
    tvalues = lr.tvalues
    pvalues = lr.pvalues

    print(f"\n  {'Variable':<12} {'Coefficient':>14} {'Std Error':>12} "
          f"{'t-stat':>10} {'p-value':>10} {'Sig':>6}")
    print(f"  {'-'*12} {'-'*14} {'-'*12} {'-'*10} {'-'*10} {'-'*6}")

    rows = []
    for i, name in enumerate(names):
        coeff = coeffs.iloc[i]
        se = std_errs.iloc[i]
        t = tvalues.iloc[i]
        p = pvalues.iloc[i]

        if p < 0.01:
            sig = "***"
        elif p < 0.05:
            sig = "**"
        elif p < 0.10:
            sig = "*"
        else:
            sig = ""

        print(f"  {name:<12} {coeff:>14.4f} {se:>12.4f} {t:>10.4f} {p:>10.4f} {sig:>6}")
        rows.append({
            "variable": name,
            "coefficient": round(coeff, 6),
            "std_error": round(se, 6),
            "t_statistic": round(t, 4),
            "p_value": round(p, 4),
        })

    print(f"\n  Significance codes: *** p<0.01, ** p<0.05, * p<0.10")

    # Interpret each coefficient in plain language
    print(f"\n  Interpretation (long-run multipliers):")
    for i, name in enumerate(names):
        coeff = coeffs.iloc[i]
        direction = "increase" if coeff > 0 else "decrease"
        print(f"    A permanent 1-pp rise in {name} is associated with a "
              f"{abs(coeff):.4f}-pp {direction} in infl in the long run.")

    lr_df = pd.DataFrame(rows)
    return lr_df


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows  |  Frequency: {df.index.freq}")

    # --- Lag selection ---
    sel = select_ardl_order(df, max_lags=6, ic="aic")
    order = sel.model.ardl_order
    p = order[0]
    q_dict = {var: order[i + 1] for i, var in enumerate(EXOG_VARS)}
    print(f"\nSelected ARDL order (AIC): ARDL{order}")

    # --- Bounds test ---
    uecm_result, bounds = bounds_test(df, p, q_dict)

    # --- Long-run coefficients ---
    lr_df = long_run_coefficients(uecm_result)
```

**Run it:**

```bash
python -m econometric_models.ardl_model
```

**What you should see (in addition to the bounds test output):**

```
======================================================================
LONG-RUN COEFFICIENTS
======================================================================

  Variable      Coefficient    Std Error     t-stat    p-value    Sig
  ------------ -------------- ------------ ---------- ---------- ------
  mpr                -X.XXXX       X.XXXX    -X.XXXX     X.XXXX    **
  tbr                 X.XXXX       X.XXXX     X.XXXX     X.XXXX
  exo                 X.XXXX       X.XXXX     X.XXXX     X.XXXX   ***

  Significance codes: *** p<0.01, ** p<0.05, * p<0.10

  Interpretation (long-run multipliers):
    A permanent 1-pp rise in mpr is associated with a X.XXXX-pp decrease in infl in the long run.
    A permanent 1-pp rise in tbr is associated with a X.XXXX-pp increase in infl in the long run.
    A permanent 1-pp rise in exo is associated with a X.XXXX-pp increase in infl in the long run.
```

(Your numbers will differ. The signs and magnitudes depend on your data.)

**What just happened -- function by function:**

- `long_run_coefficients(uecm_result)` -- This function extracts the long-run multipliers from the fitted UECM. The long-run coefficients are the equilibrium effect of each exogenous variable on inflation. They answer: "After all the short-run dynamics have played out and the system has settled, what is the permanent effect of a 1-percentage-point change in this variable?"

- `uecm_result.long_run` -- The statsmodels UECM result object provides a `.long_run` attribute that contains the long-run coefficient estimates, standard errors, t-statistics, and p-values. You do not need to compute these yourself. Internally, statsmodels derives them from the UECM coefficients using the formula: long-run coefficient of variable X = (sum of coefficients on current and lagged X) / (1 - sum of coefficients on lagged infl). The `.long_run` object handles this algebra for you.

- The significance stars (`***`, `**`, `*`) indicate whether each long-run coefficient is statistically different from zero. A coefficient with `***` (p < 0.01) means you are very confident that the variable has a genuine long-run effect on inflation.

- The interpretation lines translate the coefficients into plain language. "A permanent 1-pp rise in mpr is associated with a 0.35-pp decrease in infl" means that if the CBN permanently raises the MPR by 1 percentage point, inflation will eventually settle at a level 0.35 percentage points lower.

If that ran and printed both the bounds test and long-run coefficients, move on to the final version.

---

### Build Step 3: Save All Results and Full Main Block (FINAL)

Delete everything in `econometric_models/ardl_model.py` and replace it with this:

```python
import os
import pandas as pd
from statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

DEPENDENT = "infl"
EXOG_VARS = ["mpr", "tbr", "exo"]


def load_data():
    """Load the cleaned dataset with a monthly frequency index."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    df.index.freq = "MS"
    return df


def select_ardl_order(df, max_lags=6, ic="aic"):
    """Select optimal ARDL lag order using an information criterion."""
    y = df[DEPENDENT]
    X = df[EXOG_VARS]
    sel = ardl_select_order(y, max_lags, X, max_lags, ic=ic, trend="c")
    return sel


def fit_ardl(df, p, q_dict):
    """
    Fit the ARDL model in levels using the selected lag order.

    Parameters
    ----------
    df : DataFrame with columns infl, mpr, tbr, exo.
    p  : int, number of lags of the dependent variable (infl).
    q_dict : dict mapping each exogenous variable name to its lag order,
             e.g. {"mpr": 2, "tbr": 1, "exo": 3}.

    Returns
    -------
    result : ARDLResults, the fitted model.
    """
    y = df[DEPENDENT]
    X = df[EXOG_VARS]
    order = [q_dict[var] for var in EXOG_VARS]

    model = ARDL(y, lags=p, exog=X, order=order, trend="c")
    result = model.fit()
    return result


def bounds_test(df, p, q_dict):
    """
    Perform the Pesaran-Shin-Smith bounds test for cointegration.
    Uses the UECM (Unrestricted Error Correction Model) form.

    Parameters
    ----------
    df : DataFrame with columns infl, mpr, tbr, exo.
    p  : int, lags of the dependent variable.
    q_dict : dict mapping each exogenous variable to its lag order.

    Returns
    -------
    uecm_result : the fitted UECM model.
    bounds : the bounds test result object.
    """
    y = df[DEPENDENT]
    X = df[EXOG_VARS]
    order = [q_dict[var] for var in EXOG_VARS]

    uecm = UECM(y, lags=p, exog=X, order=order)
    uecm_result = uecm.fit()

    bounds = uecm_result.bounds_test(case=3)

    print("=" * 70)
    print("PESARAN-SHIN-SMITH BOUNDS TEST")
    print("=" * 70)
    print(f"\nF-statistic: {bounds.stat:.4f}")
    print(f"\nCritical Values:")
    print(f"  {'Significance':<15} {'I(0) Bound':>12} {'I(1) Bound':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12}")
    for sig, (i0, i1) in bounds.critical_values.items():
        print(f"  {sig:<15} {i0:>12.4f} {i1:>12.4f}")

    five_pct = bounds.critical_values["5%"]
    upper_5 = five_pct[1]
    lower_5 = five_pct[0]

    print(f"\nConclusion at 5% significance:")
    if bounds.stat > upper_5:
        print(f"  F = {bounds.stat:.4f} > upper bound {upper_5:.4f}")
        print(f"  REJECT the null. Cointegration EXISTS.")
    elif bounds.stat < lower_5:
        print(f"  F = {bounds.stat:.4f} < lower bound {lower_5:.4f}")
        print(f"  FAIL TO REJECT the null. No cointegration.")
    else:
        print(f"  {lower_5:.4f} < F = {bounds.stat:.4f} < {upper_5:.4f}")
        print(f"  INCONCLUSIVE. Use Johansen results as backup evidence.")

    return uecm_result, bounds


def long_run_coefficients(uecm_result):
    """
    Extract and interpret long-run coefficients from the fitted UECM.

    The long-run coefficients represent the equilibrium relationship:
    a permanent 1-unit increase in an exogenous variable is associated
    with a change of [coefficient] units in inflation in the long run.

    Parameters
    ----------
    uecm_result : the fitted UECM result from bounds_test().

    Returns
    -------
    lr_df : DataFrame with coefficient, std error, t-stat, p-value.
    """
    lr = uecm_result.long_run

    print("\n" + "=" * 70)
    print("LONG-RUN COEFFICIENTS")
    print("=" * 70)

    names = EXOG_VARS
    coeffs = lr.params
    std_errs = lr.bse
    tvalues = lr.tvalues
    pvalues = lr.pvalues

    print(f"\n  {'Variable':<12} {'Coefficient':>14} {'Std Error':>12} "
          f"{'t-stat':>10} {'p-value':>10} {'Sig':>6}")
    print(f"  {'-'*12} {'-'*14} {'-'*12} {'-'*10} {'-'*10} {'-'*6}")

    rows = []
    for i, name in enumerate(names):
        coeff = coeffs.iloc[i]
        se = std_errs.iloc[i]
        t = tvalues.iloc[i]
        p = pvalues.iloc[i]

        if p < 0.01:
            sig = "***"
        elif p < 0.05:
            sig = "**"
        elif p < 0.10:
            sig = "*"
        else:
            sig = ""

        print(f"  {name:<12} {coeff:>14.4f} {se:>12.4f} {t:>10.4f} {p:>10.4f} {sig:>6}")
        rows.append({
            "variable": name,
            "coefficient": round(coeff, 6),
            "std_error": round(se, 6),
            "t_statistic": round(t, 4),
            "p_value": round(p, 4),
        })

    print(f"\n  Significance codes: *** p<0.01, ** p<0.05, * p<0.10")

    print(f"\n  Interpretation (long-run multipliers):")
    for i, name in enumerate(names):
        coeff = coeffs.iloc[i]
        direction = "increase" if coeff > 0 else "decrease"
        print(f"    A permanent 1-pp rise in {name} is associated with a "
              f"{abs(coeff):.4f}-pp {direction} in infl in the long run.")

    lr_df = pd.DataFrame(rows)
    return lr_df


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows  |  Frequency: {df.index.freq}")

    # ---- Step 1: Lag selection ----
    print("\n" + "=" * 70)
    print("STEP 1: ARDL LAG ORDER SELECTION")
    print("=" * 70)

    sel = select_ardl_order(df, max_lags=6, ic="aic")
    order = sel.model.ardl_order
    p = order[0]
    q_dict = {var: order[i + 1] for i, var in enumerate(EXOG_VARS)}

    print(f"\n  AIC optimal order: ARDL{order}")
    print(f"    {DEPENDENT}: {p} lags")
    for var in EXOG_VARS:
        print(f"    {var}: {q_dict[var]} lags")

    # ---- Step 2: Estimate ARDL ----
    print("\n" + "=" * 70)
    print("STEP 2: ARDL MODEL ESTIMATION")
    print("=" * 70)

    ardl_result = fit_ardl(df, p, q_dict)
    print(ardl_result.summary())

    # ---- Step 3: Bounds test ----
    print()
    uecm_result, bounds = bounds_test(df, p, q_dict)

    # ---- Step 4: Long-run coefficients ----
    lr_df = long_run_coefficients(uecm_result)

    # ---- Step 5: Save results ----
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save bounds test results
    five_pct = bounds.critical_values["5%"]
    bounds_rows = []
    for sig, (i0, i1) in bounds.critical_values.items():
        bounds_rows.append({
            "significance": sig,
            "I0_bound": round(i0, 4),
            "I1_bound": round(i1, 4),
        })
    bounds_df = pd.DataFrame(bounds_rows)
    bounds_df.loc[len(bounds_df)] = {
        "significance": "F-statistic",
        "I0_bound": round(bounds.stat, 4),
        "I1_bound": round(bounds.stat, 4),
    }
    bounds_path = os.path.join(RESULTS_DIR, "bounds_test.csv")
    bounds_df.to_csv(bounds_path, index=False)
    print(f"\nSaved bounds test results  -> {bounds_path}")

    # Save long-run coefficients
    lr_path = os.path.join(RESULTS_DIR, "long_run_coefficients.csv")
    lr_df.to_csv(lr_path, index=False)
    print(f"Saved long-run coefficients -> {lr_path}")

    # Save ARDL order for downstream scripts
    order_path = os.path.join(RESULTS_DIR, "ardl_lag_selection.csv")
    order_data = {
        "variable": [DEPENDENT] + EXOG_VARS,
        "lag_order": list(order),
    }
    pd.DataFrame(order_data).to_csv(order_path, index=False)
    print(f"Saved lag order             -> {order_path}")

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)
```

This is your **final complete file**. It will not change again today.

**Run it:**

```bash
python -m econometric_models.ardl_model
```

**What you should see:**

The script runs through all five steps in sequence: lag selection, ARDL estimation, bounds test, long-run coefficients, and saving results. The full output will look something like:

```
Loaded 300 rows  |  Frequency: MS

======================================================================
STEP 1: ARDL LAG ORDER SELECTION
======================================================================

  AIC optimal order: ARDL(X, Y, Z, W)
    infl: X lags
    mpr: Y lags
    tbr: Z lags
    exo: W lags

======================================================================
STEP 2: ARDL MODEL ESTIMATION
======================================================================
[... full ARDL summary table ...]

======================================================================
PESARAN-SHIN-SMITH BOUNDS TEST
======================================================================

F-statistic: N.NNNN

Critical Values:
  ...

Conclusion at 5% significance:
  F = N.NNNN > upper bound X.XXXX
  REJECT the null. Cointegration EXISTS.

======================================================================
LONG-RUN COEFFICIENTS
======================================================================

  Variable      Coefficient    Std Error     t-stat    p-value    Sig
  ...

  Interpretation (long-run multipliers):
    ...

Saved bounds test results  -> .../results/bounds_test.csv
Saved long-run coefficients -> .../results/long_run_coefficients.csv
Saved lag order             -> .../results/ardl_lag_selection.csv

======================================================================
ALL DONE
======================================================================
```

**What the main block does -- step by step:**

1. **Lag selection:** Calls `select_ardl_order()` to find the AIC-optimal ARDL lag structure. Extracts `p` (lags of infl) and `q_dict` (lags of each exogenous variable) for use in all subsequent steps.

2. **ARDL estimation:** Calls `fit_ardl()` to fit the ARDL model in levels and prints the full summary table. This is the same model from Day 12, repeated here so the full pipeline runs in one go.

3. **Bounds test:** Calls `bounds_test()` to run the Pesaran-Shin-Smith test and determine whether cointegration exists.

4. **Long-run coefficients:** Calls `long_run_coefficients()` to extract the equilibrium multipliers and print their interpretation.

5. **Save results:** Writes three CSV files to the `results/` directory. `bounds_test.csv` stores the F-statistic and critical values. `long_run_coefficients.csv` stores the coefficient estimates with their standard errors, t-statistics, and p-values. `ardl_lag_selection.csv` stores the selected lag order so downstream scripts can read it without re-running the selection.

---

## Interpreting Results -- Nigeria Specific

### Expected Outcome: F-Statistic Above the Upper Bound

For Nigerian macroeconomic data, you should expect the F-statistic to exceed the upper bound at the 5% level. This means cointegration is confirmed -- inflation, MPR, TBR, and the exchange rate share a genuine long-run equilibrium. This result is consistent with the Johansen test from Day 9, which also found cointegration among these variables. Having two independent tests pointing to the same conclusion (Johansen and bounds test) makes your evidence very strong.

### Long-Run MPR Coefficient: Likely Small and Negative

The long-run coefficient on `mpr` will likely be small and negative. This means that a permanent 1-percentage-point increase in the Monetary Policy Rate is associated with a small decrease in inflation in the long run. The effect is small because the MPR-to-inflation transmission channel in Nigeria is weak. The CBN raises MPR, commercial banks slowly adjust their lending rates, borrowing and spending decline, and prices eventually ease -- but each link in this chain is imperfect. Many Nigerian businesses rely on cash transactions and informal credit, bypassing the formal banking system entirely. The MPR signal gets diluted as it travels through the economy.

If the coefficient is, say, -0.35, you would say: "A permanent 1-percentage-point increase in the MPR is associated with a 0.35-percentage-point decrease in inflation in the long run. This modest effect is consistent with the documented weakness of the interest rate channel in Nigeria (Adebiyi and Mordi, 2016)."

### Long-Run EXO Coefficient: Likely Positive

The long-run coefficient on `exo` (exchange rate) will likely be positive and may be the largest in magnitude. A positive coefficient means that Naira depreciation (a higher exchange rate -- more Naira per dollar) leads to higher inflation in the long run. This is the **exchange rate pass-through** effect and it is one of the strongest channels driving Nigerian inflation.

Nigeria imports a large share of its consumer goods, refined petroleum, and industrial inputs. When the Naira weakens, import costs rise, which pushes up domestic prices. The pass-through is especially strong for food prices, which carry a heavy weight in Nigeria's CPI basket. If the coefficient is 0.08, it means a permanent 1-Naira depreciation per dollar is associated with a 0.08-percentage-point rise in inflation.

### Long-Run TBR Coefficient: May Be Positive

The long-run coefficient on `tbr` (Treasury Bill Rate) may be positive, which can seem counterintuitive. Higher interest rates are supposed to reduce inflation, not increase it. But the Treasury Bill Rate is a market-determined rate, not a policy rate. It reflects market conditions. When inflation is high, investors demand higher TBR to compensate for the erosion of purchasing power. So high TBR and high inflation tend to go together -- not because TBR causes inflation, but because both respond to the same underlying conditions (excess liquidity, fiscal pressures, exchange rate shocks).

If the TBR coefficient is positive, explain it this way in your thesis: "The positive long-run coefficient on TBR reflects the Fisher effect -- market interest rates incorporate inflation expectations, so higher expected inflation drives up the Treasury Bill Rate. The coefficient captures the equilibrium relationship between inflation and market rates, not a causal direction from TBR to inflation."

### What If the Bounds Test Is Inconclusive?

If the F-statistic falls between the lower and upper bounds, the bounds test cannot determine whether cointegration exists. This is not a disaster. You have backup evidence from the Johansen test on Day 9. If Johansen found cointegration rank >= 1, you can state: "The ARDL bounds test was inconclusive at the 5% level. However, the Johansen trace and maximum eigenvalue tests both rejected the null of zero cointegrating relationships at the 5% level (Day 9). On the balance of evidence, we proceed with the assumption that a long-run equilibrium exists among the variables."

You can also try re-running the bounds test with the BIC-optimal lag order instead of AIC. A different lag structure may push the F-statistic above the upper bound.

---

## Commit

```bash
git add econometric_models/ardl_model.py
git commit -m "Day 13: ARDL bounds test and long-run coefficients"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `AttributeError: 'ARDLResults' object has no attribute 'bounds_test'` | You called `bounds_test()` on an `ARDL` result instead of a `UECM` result. The bounds test method belongs to the `UECM` class, not the `ARDL` class. Make sure you are fitting the model with `UECM(y, lags=p, exog=X, order=order)`, not `ARDL(...)`. The UECM is the error correction form of the ARDL and only that form supports the bounds test. |
| `KeyError: '5%'` when reading critical values | The critical value dictionary keys depend on your statsmodels version. In some versions the keys are `"5%"`, in others they may be `"5.0%"` or numeric. Print `bounds.critical_values.keys()` to see the exact key names and adjust the code accordingly. |
| `ValueError: order must be a list of integers` | The `order` parameter for UECM must be a list of integers matching the number of exogenous variables. If EXOG_VARS has 3 elements, `order` must be a list of exactly 3 integers. Check that `q_dict` contains an entry for every variable in EXOG_VARS and that `[q_dict[var] for var in EXOG_VARS]` produces the correct list. |
| Long-run coefficients have unexpected signs | This is not necessarily an error. Coefficient signs depend on the data. If MPR has a positive long-run coefficient (contrary to theory), it may indicate that the CBN raises rates in response to high inflation rather than the other way around (reverse causality). Discuss this in your interpretation -- do not change the model to force a particular sign. |
| `LinAlgError: Singular matrix` or convergence failure | The UECM estimation may fail if the selected lag order creates a near-singular design matrix. Try reducing `max_lags` from 6 to 4 and re-running, or use the BIC-optimal order instead of AIC. BIC selects fewer lags, which reduces the chance of multicollinearity. |
| `FileNotFoundError: cleaned_data.csv` | You need to run the Day 4 cleaning script first: `python -m data_processing.clean`. All econometric scripts read from `data/processed/cleaned_data.csv`. |

---

## Check Your Understanding

### 1. "What does the bounds test tell you that the Johansen test does not?"

**Answer:** "The Johansen test and the bounds test both test for cointegration, but they do so in different frameworks with different advantages. The Johansen test is a system-based test -- it examines all variables symmetrically in a VAR/VECM system and determines the cointegration rank (how many independent long-run relationships exist). It requires all variables to be I(1) and works best with large samples.

The ARDL bounds test is a single-equation test. It tests whether there is a long-run relationship between one dependent variable (inflation) and a set of regressors (MPR, TBR, exchange rate). Its key advantage is that it does not require pre-testing the integration order of the variables -- it is valid whether the regressors are I(0), I(1), or a mix. It also performs better in small and moderate samples.

In my project, I use both tests. The Johansen test (Day 9) confirmed cointegration in the multivariate system. The bounds test (Day 13) confirmed cointegration in the specific ARDL equation with inflation as the dependent variable. Having both tests agree strengthens the evidence for a long-run equilibrium."

### 2. "Your long-run MPR coefficient is -0.35. What does that mean for the CBN?"

**Answer:** "The coefficient of -0.35 means that if the Central Bank of Nigeria permanently raises the Monetary Policy Rate by 1 percentage point, inflation will eventually decrease by 0.35 percentage points in the long run, after all the short-run dynamics have played out and the economy has settled into its new equilibrium.

This has two important implications for the CBN. First, the negative sign confirms that monetary tightening does reduce inflation in the long run -- the policy works in the expected direction. Second, the small magnitude tells us that the transmission is weak. To reduce inflation by 1 full percentage point, the CBN would need to raise the MPR by roughly 3 percentage points (1 / 0.35 = 2.86). This is consistent with the well-documented weakness of the interest rate channel in Nigeria, where a large informal sector, limited financial intermediation, and structural supply-side constraints dilute the impact of monetary policy changes."

### 3. "What happens if the F-statistic falls in the inconclusive region?"

**Answer:** "If the F-statistic falls between the lower and upper bounds, the test cannot determine whether cointegration exists based on the asymptotic critical values alone. The inconclusive region exists because the test does not know the exact integration order of the regressors -- the lower bound assumes all regressors are I(0) and the upper bound assumes all are I(1). When the F-statistic falls between them, the conclusion depends on which assumption is correct.

In practice, I have already established the integration order of all variables using the ADF and KPSS tests on Days 6 and 7. All four variables are I(1). Since the regressors are all I(1), the relevant critical value is the upper bound. If the F-statistic is closer to the upper bound, I would lean toward concluding cointegration exists.

Additionally, I have the Johansen test from Day 9 as independent supporting evidence. If Johansen found cointegration, I would state that the bounds test was inconclusive but the Johansen test provides strong supporting evidence for a long-run equilibrium, and I proceed with the ARDL error correction model on that basis."

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Bounds test function | `econometric_models/ardl_model.py` | Runs the Pesaran-Shin-Smith bounds test using the UECM form |
| Long-run coefficients function | `econometric_models/ardl_model.py` | Extracts and interprets equilibrium multipliers from the UECM |
| Bounds test results | `results/bounds_test.csv` | F-statistic and critical values at all significance levels |
| Long-run coefficients | `results/long_run_coefficients.csv` | Coefficient, std error, t-stat, p-value for each exogenous variable |
| Lag order record | `results/ardl_lag_selection.csv` | Selected ARDL lag order for use in downstream scripts |

**No new packages installed today.** The `UECM` class and `bounds_test` method are part of statsmodels, which was installed on Day 6.

---

**Tomorrow (Day 14):** You will build the Error Correction Model (ECM). Now that the bounds test has confirmed a long-run equilibrium, the ECM tells you how quickly inflation corrects back toward that equilibrium after a shock. The key result is the **error correction coefficient** -- a number between -1 and 0 that measures the speed of adjustment. A value of -0.15, for example, means that 15% of any deviation from the long-run equilibrium is corrected each month. Day 14 is where the short-run dynamics come alive.
