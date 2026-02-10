# Week 2, Day 7 -- KPSS Test & Integration Order Confirmation

## What You'll Learn Today

- What the KPSS test is and why its opposite null hypothesis matters
- The ADF+KPSS confirmatory strategy for stationarity conclusions
- How to formally determine the integration order of each variable
- Why confirmed I(1) for all variables is good news for ARDL bounds testing

---

## Why Both ADF and KPSS?

Yesterday you ran the ADF test and concluded that all four variables are non-stationary in levels but stationary in first differences. That was a good start, but one test alone is not enough for a thesis. Here is why.

Every hypothesis test can make two kinds of mistakes:

- **Type I error:** You reject the null hypothesis when it is actually true. (False positive.)
- **Type II error:** You fail to reject the null hypothesis when it is actually false. (False negative.)

The ADF test has a well-known weakness: it has **low statistical power**. "Low power" means it has a high probability of Type II error. In plain language, when a series really IS stationary, the ADF test sometimes fails to detect it and incorrectly says "non-stationary." So when the ADF test says "I cannot reject the presence of a unit root," you are left wondering: is the series genuinely non-stationary, or is the ADF test just too weak to detect stationarity?

The KPSS test solves this problem by asking the question from the opposite direction.

**ADF test:**
- Null hypothesis (H0): The series HAS a unit root (it IS non-stationary)
- Alternative hypothesis (H1): The series does NOT have a unit root (it IS stationary)
- If p-value < 0.05: reject H0 -- conclude the series is STATIONARY
- If p-value >= 0.05: fail to reject H0 -- conclude the series is NON-STATIONARY

**KPSS test (Kwiatkowski-Phillips-Schmidt-Shin, 1992):**
- Null hypothesis (H0): The series IS stationary (no unit root)
- Alternative hypothesis (H1): The series IS non-stationary (has a unit root)
- If p-value < 0.05: reject H0 -- conclude the series is NON-STATIONARY
- If p-value >= 0.05: fail to reject H0 -- conclude the series IS STATIONARY

Notice the nulls are exact opposites. The ADF test assumes non-stationarity and looks for evidence of stationarity. The KPSS test assumes stationarity and looks for evidence of non-stationarity. By running both, you get a cross-check. Think of it like asking two witnesses who are biased in opposite directions. If they both say the same thing, you can trust the conclusion.

### The Four Possible Outcomes

When you combine the two tests, there are exactly four outcomes:

| ADF Result | KPSS Result | Conclusion |
|------------|-------------|------------|
| Rejects H0 (stationary) | Fails to reject H0 (stationary) | **Definitely stationary -- I(0)** |
| Fails to reject H0 (non-stationary) | Rejects H0 (non-stationary) | **Definitely non-stationary -- I(1)** |
| Both reject their nulls | Both reject their nulls | **Inconclusive** |
| Both fail to reject their nulls | Both fail to reject their nulls | **Inconclusive** |

The first row is the best case for stationarity: the ADF test found enough evidence to reject the unit root, and the KPSS test could not find evidence against stationarity. Both tests agree the series is stationary.

The second row is the best case for non-stationarity: the ADF test could not reject the unit root, and the KPSS test found enough evidence to reject stationarity. Both tests agree the series is non-stationary.

The third and fourth rows are the ambiguous cases. Both tests reject, or both tests fail to reject. This can happen when a series has a structural break (a sudden jump or shift in level), which confuses the tests. If you encounter this, you may need a structural break test like Zivot-Andrews, but that is beyond this project.

For Nigerian macroeconomic data, we expect clean results: all four variables should be **definitely non-stationary in levels** and **definitely stationary in first differences**.

---

## No New Packages

The KPSS test is included in the `statsmodels` library, which you installed on Day 6. You do not need to install anything new today. The import path is the same module: `statsmodels.tsa.stattools` contains both `adfuller` and `kpss`.

---

## Building `econometric_models/stationarity.py` -- Step by Step

We are expanding the file you built yesterday. At each step, we show you the **complete file** from the first line to the last line. You delete everything and replace it with exactly what is shown. No guessing where code goes, no partial snippets.

Make sure you have the `econometric_models/` folder with an `__init__.py` file inside it (from Day 1). If not, create them now.

---

### Build Step 1: Everything from Day 6 Plus the New kpss_test Function

This step takes your entire Day 6 file (imports, constants, `load_data`, `adf_test`) and adds a new `kpss_test` function. The main block runs KPSS on all four variables in levels and prints the results.

Delete everything in `econometric_models/stationarity.py` and replace it with this:

```python
"""Stationarity testing for the Nigerian Inflation Predictor."""
import os
import sys
import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    """Load the cleaned dataset from data/processed/cleaned_data.csv."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def adf_test(series, variable_name, significance=0.05):
    """
    Run the Augmented Dickey-Fuller test.

    Null hypothesis: the series HAS a unit root (non-stationary).
    If p-value < significance, reject null -> series is STATIONARY.

    Returns a dict with variable, adf_statistic, p_value, lags_used,
    is_stationary.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    adf_stat = result[0]
    p_value = result[1]
    lags_used = result[2]
    n_obs = result[3]
    critical_values = result[4]

    is_stationary = p_value < significance

    print(f"\nADF Test for: {variable_name}")
    print(f"  ADF Statistic : {adf_stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")
    print(f"  Lags Used     : {lags_used}")
    print(f"  Observations  : {n_obs}")
    print(f"  Critical Values:")
    for key, val in critical_values.items():
        print(f"    {key}: {val:.4f}")
    print(f"  Result: {'STATIONARY' if is_stationary else 'NON-STATIONARY (has unit root)'}")

    return {
        "variable": variable_name,
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "lags_used": lags_used,
        "is_stationary": is_stationary,
    }


def kpss_test(series, variable_name, regression="c", significance=0.05):
    """
    Run the KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test.

    Null hypothesis: the series IS stationary.
    This is the OPPOSITE of the ADF test.

    If p-value < significance, reject null -> series is NON-STATIONARY.
    If p-value >= significance, fail to reject null -> series IS STATIONARY.

    The regression parameter controls what kind of stationarity to test:
      "c"  = level stationarity (stationary around a constant mean)
      "ct" = trend stationarity (stationary around a linear trend)
    We use "c" because we want to know if the series has a stable mean.

    Returns a dict with variable, kpss_statistic, p_value, lags_used,
    is_stationary.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_value, lags_used, critical_values = kpss(
            series.dropna(), regression=regression, nlags="auto"
        )

    is_stationary = p_value >= significance

    print(f"\nKPSS Test for: {variable_name}")
    print(f"  KPSS Statistic: {stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")
    print(f"  Lags Used     : {lags_used}")
    print(f"  Critical Values:")
    for key, val in critical_values.items():
        print(f"    {key}: {val:.4f}")
    print(f"  Result: {'STATIONARY' if is_stationary else 'NON-STATIONARY (reject stationarity)'}")

    return {
        "variable": variable_name,
        "kpss_statistic": stat,
        "p_value": p_value,
        "lags_used": lags_used,
        "is_stationary": is_stationary,
    }


if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded: {df.shape[0]} observations, {df.shape[1]} variables")
    print(f"Period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")

    print("\n" + "=" * 60)
    print("KPSS TESTS -- LEVELS (original data)")
    print("=" * 60)

    for col in df.columns:
        kpss_test(df[col], col)
```

**Run it:**

```bash
python -m econometric_models.stationarity
```

**What you should see:**

```
Data loaded: 300 observations, 4 variables
Period: 2000-01 to 2024-12

============================================================
KPSS TESTS -- LEVELS (original data)
============================================================

KPSS Test for: mpr
  KPSS Statistic: X.XXXX
  p-value       : 0.0100
  Lags Used     : XX
  Critical Values:
    10%: 0.347
    5%: 0.463
    2.5%: 0.574
    1%: 0.739
  Result: NON-STATIONARY (reject stationarity)

KPSS Test for: infl
  KPSS Statistic: X.XXXX
  p-value       : 0.0100
  ...
  Result: NON-STATIONARY (reject stationarity)

KPSS Test for: exo
  KPSS Statistic: X.XXXX
  p-value       : 0.0100
  ...
  Result: NON-STATIONARY (reject stationarity)

KPSS Test for: tbr
  KPSS Statistic: X.XXXX
  p-value       : 0.0100
  ...
  Result: NON-STATIONARY (reject stationarity)
```

(Your exact numbers will depend on your data. The `X.XXXX` values will be actual numbers.)

All four variables should show NON-STATIONARY. The KPSS test rejected its null hypothesis of stationarity for each variable. This agrees with what the ADF test told us yesterday -- both tests now point in the same direction.

**What each new piece does -- line by line:**

- `import warnings` -- We import this at the top of the file (not inside the function) because it is a standard Python module. We need it to suppress a warning that the KPSS function generates.

- `from statsmodels.tsa.stattools import adfuller, kpss` -- We now import both test functions from the same module. The path `tsa.stattools` stands for "time series analysis, statistical tools."

- `warnings.catch_warnings()` -- This creates a temporary context where we can control how Python handles warnings. Inside this `with` block, we suppress all warnings. Once the block ends, warnings go back to normal. We do this because the KPSS function in statsmodels raises an `InterpolationWarning` almost every time it runs, telling you that the p-value has been truncated. We handle this ourselves, so the warning is just noise.

- `warnings.simplefilter("ignore")` -- This tells Python to ignore all warnings inside the `with` block. It only applies inside the block, not globally.

- `kpss(series.dropna(), regression="c", nlags="auto")` -- This is the core KPSS test call. It returns four values (not six like ADF): the test statistic, the p-value, the number of lags used, and a dictionary of critical values.

- `regression="c"` -- Tests for stationarity around a constant mean (level stationarity). The other option is `"ct"` for trend stationarity, but `"c"` is the standard choice for macroeconomic variables.

- `nlags="auto"` -- Lets statsmodels automatically choose the number of lags for the spectral estimation using the Schwert (1989) rule. This is the standard approach.

- `is_stationary = p_value >= significance` -- This is the **opposite** of the ADF test. For the ADF test, a low p-value means stationary (`p_value < significance`). For the KPSS test, a **high** p-value means stationary (`p_value >= significance`). This is because the KPSS null is stationarity. If we fail to reject the null (high p-value), the series is stationary. If we reject the null (low p-value), the series is non-stationary.

This opposite logic is the single most important thing to understand about the KPSS test. Get this wrong and your conclusions flip.

---

### Build Step 2: Add determine_integration_order and Run on All 4 Variables

Now we add the function that brings ADF and KPSS together. It runs both tests on levels AND first differences for a single variable, then uses the combined results to formally determine the integration order.

Delete everything in `econometric_models/stationarity.py` and replace it with this:

```python
"""Stationarity testing for the Nigerian Inflation Predictor."""
import os
import sys
import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    """Load the cleaned dataset from data/processed/cleaned_data.csv."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def adf_test(series, variable_name, significance=0.05):
    """
    Run the Augmented Dickey-Fuller test.

    Null hypothesis: the series HAS a unit root (non-stationary).
    If p-value < significance, reject null -> series is STATIONARY.

    Returns a dict with variable, adf_statistic, p_value, lags_used,
    is_stationary.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    adf_stat = result[0]
    p_value = result[1]
    lags_used = result[2]
    n_obs = result[3]
    critical_values = result[4]

    is_stationary = p_value < significance

    print(f"\nADF Test for: {variable_name}")
    print(f"  ADF Statistic : {adf_stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")
    print(f"  Lags Used     : {lags_used}")
    print(f"  Observations  : {n_obs}")
    print(f"  Critical Values:")
    for key, val in critical_values.items():
        print(f"    {key}: {val:.4f}")
    print(f"  Result: {'STATIONARY' if is_stationary else 'NON-STATIONARY (has unit root)'}")

    return {
        "variable": variable_name,
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "lags_used": lags_used,
        "is_stationary": is_stationary,
    }


def kpss_test(series, variable_name, regression="c", significance=0.05):
    """
    Run the KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test.

    Null hypothesis: the series IS stationary.
    This is the OPPOSITE of the ADF test.

    If p-value < significance, reject null -> series is NON-STATIONARY.
    If p-value >= significance, fail to reject null -> series IS STATIONARY.

    Returns a dict with variable, kpss_statistic, p_value, lags_used,
    is_stationary.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_value, lags_used, critical_values = kpss(
            series.dropna(), regression=regression, nlags="auto"
        )

    is_stationary = p_value >= significance

    print(f"\nKPSS Test for: {variable_name}")
    print(f"  KPSS Statistic: {stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")
    print(f"  Lags Used     : {lags_used}")
    print(f"  Critical Values:")
    for key, val in critical_values.items():
        print(f"    {key}: {val:.4f}")
    print(f"  Result: {'STATIONARY' if is_stationary else 'NON-STATIONARY (reject stationarity)'}")

    return {
        "variable": variable_name,
        "kpss_statistic": stat,
        "p_value": p_value,
        "lags_used": lags_used,
        "is_stationary": is_stationary,
    }


def determine_integration_order(series, variable_name, significance=0.05):
    """
    Determine the integration order of a variable using both ADF and KPSS.

    Runs both tests on the variable in levels first. If both confirm
    stationarity, the variable is I(0). If not, it differences once and
    tests again. If both confirm stationarity after one differencing,
    the variable is I(1).

    Returns a dict with variable, adf_level_pval, kpss_level_pval,
    adf_diff_pval, kpss_diff_pval, integration_order.
    """
    # --- Test in levels ---
    print(f"\n{'='*60}")
    print(f"TESTING {variable_name.upper()} IN LEVELS")
    print(f"{'='*60}")

    adf_level = adf_test(series, f"{variable_name} (level)", significance)
    kpss_level = kpss_test(series, f"{variable_name} (level)", significance=significance)

    # --- Test in first differences ---
    diff_series = series.diff().dropna()

    print(f"\n{'='*60}")
    print(f"TESTING {variable_name.upper()} IN FIRST DIFFERENCES")
    print(f"{'='*60}")

    adf_diff = adf_test(diff_series, f"d_{variable_name} (1st diff)", significance)
    kpss_diff = kpss_test(diff_series, f"d_{variable_name} (1st diff)", significance=significance)

    # --- Determine integration order ---
    if adf_level["is_stationary"] and kpss_level["is_stationary"]:
        integration_order = 0
        confirmation = "CONFIRMED I(0): Both ADF and KPSS agree -- stationary in levels"
    elif not adf_level["is_stationary"] and not kpss_level["is_stationary"]:
        if adf_diff["is_stationary"] and kpss_diff["is_stationary"]:
            integration_order = 1
            confirmation = "CONFIRMED I(1): Non-stationary in levels, stationary in 1st differences"
        else:
            integration_order = 2
            confirmation = "WARNING I(2)+: Still non-stationary after first differencing"
    else:
        if adf_diff["is_stationary"] and kpss_diff["is_stationary"]:
            integration_order = 1
            confirmation = "LIKELY I(1): Level tests inconclusive, but 1st differences are stationary"
        else:
            integration_order = -1
            confirmation = "INCONCLUSIVE: ADF and KPSS disagree at both levels and differences"

    print(f"\n>>> {variable_name}: {confirmation}")

    return {
        "variable": variable_name,
        "adf_level_pval": round(adf_level["p_value"], 4),
        "kpss_level_pval": round(kpss_level["p_value"], 4),
        "adf_diff_pval": round(adf_diff["p_value"], 4),
        "kpss_diff_pval": round(kpss_diff["p_value"], 4),
        "integration_order": integration_order,
    }


if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded: {df.shape[0]} observations, {df.shape[1]} variables")
    print(f"Period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")

    all_results = []
    for col in df.columns:
        result = determine_integration_order(df[col], col)
        all_results.append(result)

    # --- Print summary table ---
    print("\n\n" + "=" * 80)
    print("INTEGRATION ORDER SUMMARY (ADF + KPSS Confirmatory Strategy)")
    print("=" * 80)
    print(f"{'Variable':<10} {'ADF Level':<12} {'KPSS Level':<12} "
          f"{'ADF Diff':<12} {'KPSS Diff':<12} {'Order':<8}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['variable']:<10} "
              f"{r['adf_level_pval']:<12.4f} "
              f"{r['kpss_level_pval']:<12.4f} "
              f"{r['adf_diff_pval']:<12.4f} "
              f"{r['kpss_diff_pval']:<12.4f} "
              f"I({r['integration_order']})")
    print("=" * 80)
```

**Run it:**

```bash
python -m econometric_models.stationarity
```

**What you should see:**

For each of the four variables (mpr, infl, exo, tbr), you see ADF and KPSS results in levels, then ADF and KPSS results in first differences, then a confirmation line. At the very end, a summary table:

```
================================================================================
INTEGRATION ORDER SUMMARY (ADF + KPSS Confirmatory Strategy)
================================================================================
Variable   ADF Level    KPSS Level   ADF Diff     KPSS Diff    Order
--------------------------------------------------------------------------------
mpr        0.XXXX       0.0100       0.XXXX       0.1000       I(1)
infl       0.XXXX       0.0100       0.XXXX       0.1000       I(1)
exo        0.XXXX       0.0100       0.XXXX       0.1000       I(1)
tbr        0.XXXX       0.0100       0.XXXX       0.1000       I(1)
================================================================================
```

(The `X.XXXX` values will be actual p-values. The KPSS values will likely show `0.0100` for levels and `0.1000` for differences because of the bounded p-value behavior explained later in the Common Errors section.)

**How determine_integration_order works -- line by line:**

- The function takes a pandas Series, a variable name, and a significance level (default 0.05).

- It first tests the original series (levels) with both ADF and KPSS. If both agree the series is stationary, it returns I(0) immediately.

- Then it computes the first difference with `series.diff().dropna()`. The `.diff()` method subtracts each value from the previous one: `value_this_month - value_last_month`. The `.dropna()` removes the first row, which becomes NaN because there is no previous month to subtract from.

- It tests the differenced series with both ADF and KPSS. If the levels were confirmed non-stationary AND the differences are confirmed stationary, the variable is I(1).

- The logic handles inconclusive cases too. If the level tests disagree (ADF says one thing, KPSS says another), but the differenced series is clearly stationary, the function reports "likely I(1)." If nothing is clear, it reports "inconclusive."

- The function returns a dictionary with all five p-values and the integration order. This dictionary format makes it easy to collect results from all variables and build a summary table.

---

### Build Step 3: Add Save Functionality and Final Formatted Output

This is the final, complete version of the file. It adds the ability to save all results to `results/integration_order.csv` and prints a polished summary with interpretation notes.

Delete everything in `econometric_models/stationarity.py` and replace it with this:

```python
"""Stationarity testing for the Nigerian Inflation Predictor."""
import os
import sys
import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    """Load the cleaned dataset from data/processed/cleaned_data.csv."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def adf_test(series, variable_name, significance=0.05):
    """
    Run the Augmented Dickey-Fuller test.

    Null hypothesis: the series HAS a unit root (non-stationary).
    If p-value < significance, reject null -> series is STATIONARY.

    Returns a dict with variable, adf_statistic, p_value, lags_used,
    is_stationary.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    adf_stat = result[0]
    p_value = result[1]
    lags_used = result[2]
    n_obs = result[3]
    critical_values = result[4]

    is_stationary = p_value < significance

    print(f"\nADF Test for: {variable_name}")
    print(f"  ADF Statistic : {adf_stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")
    print(f"  Lags Used     : {lags_used}")
    print(f"  Observations  : {n_obs}")
    print(f"  Critical Values:")
    for key, val in critical_values.items():
        print(f"    {key}: {val:.4f}")
    print(f"  Result: {'STATIONARY' if is_stationary else 'NON-STATIONARY (has unit root)'}")

    return {
        "variable": variable_name,
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "lags_used": lags_used,
        "is_stationary": is_stationary,
    }


def kpss_test(series, variable_name, regression="c", significance=0.05):
    """
    Run the KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test.

    Null hypothesis: the series IS stationary.
    This is the OPPOSITE of the ADF test.

    If p-value < significance, reject null -> series is NON-STATIONARY.
    If p-value >= significance, fail to reject null -> series IS STATIONARY.

    The regression parameter controls what kind of stationarity to test:
      "c"  = level stationarity (stationary around a constant mean)
      "ct" = trend stationarity (stationary around a linear trend)
    We use "c" because we want to know if the series has a stable mean.

    Returns a dict with variable, kpss_statistic, p_value, lags_used,
    is_stationary.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_value, lags_used, critical_values = kpss(
            series.dropna(), regression=regression, nlags="auto"
        )

    is_stationary = p_value >= significance

    print(f"\nKPSS Test for: {variable_name}")
    print(f"  KPSS Statistic: {stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")
    print(f"  Lags Used     : {lags_used}")
    print(f"  Critical Values:")
    for key, val in critical_values.items():
        print(f"    {key}: {val:.4f}")
    print(f"  Result: {'STATIONARY' if is_stationary else 'NON-STATIONARY (reject stationarity)'}")

    return {
        "variable": variable_name,
        "kpss_statistic": stat,
        "p_value": p_value,
        "lags_used": lags_used,
        "is_stationary": is_stationary,
    }


def determine_integration_order(series, variable_name, significance=0.05):
    """
    Determine the integration order of a variable using both ADF and KPSS.

    Runs both tests on the variable in levels first. If both confirm
    stationarity, the variable is I(0). If not, it differences once and
    tests again. If both confirm stationarity after one differencing,
    the variable is I(1).

    Returns a dict with variable, adf_level_pval, kpss_level_pval,
    adf_diff_pval, kpss_diff_pval, integration_order.
    """
    # --- Test in levels ---
    print(f"\n{'='*60}")
    print(f"TESTING {variable_name.upper()} IN LEVELS")
    print(f"{'='*60}")

    adf_level = adf_test(series, f"{variable_name} (level)", significance)
    kpss_level = kpss_test(series, f"{variable_name} (level)", significance=significance)

    # --- Test in first differences ---
    diff_series = series.diff().dropna()

    print(f"\n{'='*60}")
    print(f"TESTING {variable_name.upper()} IN FIRST DIFFERENCES")
    print(f"{'='*60}")

    adf_diff = adf_test(diff_series, f"d_{variable_name} (1st diff)", significance)
    kpss_diff = kpss_test(diff_series, f"d_{variable_name} (1st diff)", significance=significance)

    # --- Determine integration order ---
    if adf_level["is_stationary"] and kpss_level["is_stationary"]:
        integration_order = 0
        confirmation = "CONFIRMED I(0): Both ADF and KPSS agree -- stationary in levels"
    elif not adf_level["is_stationary"] and not kpss_level["is_stationary"]:
        if adf_diff["is_stationary"] and kpss_diff["is_stationary"]:
            integration_order = 1
            confirmation = "CONFIRMED I(1): Non-stationary in levels, stationary in 1st differences"
        else:
            integration_order = 2
            confirmation = "WARNING I(2)+: Still non-stationary after first differencing"
    else:
        if adf_diff["is_stationary"] and kpss_diff["is_stationary"]:
            integration_order = 1
            confirmation = "LIKELY I(1): Level tests inconclusive, but 1st differences are stationary"
        else:
            integration_order = -1
            confirmation = "INCONCLUSIVE: ADF and KPSS disagree at both levels and differences"

    print(f"\n>>> {variable_name}: {confirmation}")

    return {
        "variable": variable_name,
        "adf_level_pval": round(adf_level["p_value"], 4),
        "kpss_level_pval": round(kpss_level["p_value"], 4),
        "adf_diff_pval": round(adf_diff["p_value"], 4),
        "kpss_diff_pval": round(kpss_diff["p_value"], 4),
        "integration_order": integration_order,
    }


if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded: {df.shape[0]} observations, {df.shape[1]} variables")
    print(f"Period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")

    all_results = []
    for col in df.columns:
        result = determine_integration_order(df[col], col)
        all_results.append(result)

    # --- Build and save results DataFrame ---
    results_df = pd.DataFrame(all_results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "integration_order.csv")
    results_df.to_csv(output_path, index=False)

    # --- Print final formatted summary ---
    print("\n\n" + "=" * 80)
    print("INTEGRATION ORDER SUMMARY (ADF + KPSS Confirmatory Strategy)")
    print("=" * 80)
    print(f"{'Variable':<10} {'ADF Level':<12} {'KPSS Level':<12} "
          f"{'ADF Diff':<12} {'KPSS Diff':<12} {'Order':<8}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['variable']:<10} "
              f"{r['adf_level_pval']:<12.4f} "
              f"{r['kpss_level_pval']:<12.4f} "
              f"{r['adf_diff_pval']:<12.4f} "
              f"{r['kpss_diff_pval']:<12.4f} "
              f"I({r['integration_order']})")
    print("=" * 80)

    print("\nHow to read this table:")
    print("  ADF Level  p >= 0.05 -> non-stationary in levels (fail to reject unit root)")
    print("  KPSS Level p <  0.05 -> non-stationary in levels (reject stationarity)")
    print("  ADF Diff   p <  0.05 -> stationary after differencing (reject unit root)")
    print("  KPSS Diff  p >= 0.05 -> stationary after differencing (fail to reject stationarity)")
    print("  I(1) = variable needs one round of differencing to become stationary")

    print(f"\nResults saved to: {output_path}")
    print("You can open this CSV in Excel or paste it into your thesis appendix.")
```

Build Step 3 above is your final complete file.

**Run it:**

```bash
python -m econometric_models.stationarity
```

**What you should see:**

After all the individual test outputs (which are lengthy because each variable gets four tests), you see:

```
================================================================================
INTEGRATION ORDER SUMMARY (ADF + KPSS Confirmatory Strategy)
================================================================================
Variable   ADF Level    KPSS Level   ADF Diff     KPSS Diff    Order
--------------------------------------------------------------------------------
mpr        0.XXXX       0.0100       0.XXXX       0.1000       I(1)
infl       0.XXXX       0.0100       0.XXXX       0.1000       I(1)
exo        0.XXXX       0.0100       0.XXXX       0.1000       I(1)
tbr        0.XXXX       0.0100       0.XXXX       0.1000       I(1)
================================================================================

How to read this table:
  ADF Level  p >= 0.05 -> non-stationary in levels (fail to reject unit root)
  KPSS Level p <  0.05 -> non-stationary in levels (reject stationarity)
  ADF Diff   p <  0.05 -> stationary after differencing (reject unit root)
  KPSS Diff  p >= 0.05 -> stationary after differencing (fail to reject stationarity)
  I(1) = variable needs one round of differencing to become stationary

Results saved to: /path/to/results/integration_order.csv
You can open this CSV in Excel or paste it into your thesis appendix.
```

**What changed from Step 2:**

- `results_df = pd.DataFrame(all_results)` -- Converts the list of dictionaries into a pandas DataFrame. Each dictionary (one per variable) becomes one row. The keys become column names.

- `os.makedirs(RESULTS_DIR, exist_ok=True)` -- Creates the `results/` folder if it does not already exist. The `exist_ok=True` parameter means it will not crash if the folder is already there from Day 6.

- `results_df.to_csv(output_path, index=False)` -- Saves the summary to a CSV file. The `index=False` parameter means it does not write row numbers into the file -- we do not need them because the variable name identifies each row.

- The formatted print block at the end uses Python f-string formatting with width specifiers like `{r['variable']:<10}`. The `<10` means "left-align in a field 10 characters wide." The `.4f` means "format as a floating-point number with 4 decimal places." This produces the clean aligned table you see in the output.

- The "How to read this table" section is printed below the table so that anyone running the script understands what the numbers mean without needing to refer back to the guide.

---

## Interpreting the Results -- Nigerian Data

All four variables -- mpr, infl, exo, and tbr -- are confirmed I(1) by the ADF+KPSS confirmatory strategy. Here is what this means for each variable and for your project.

### mpr (Monetary Policy Rate %)

**In levels: CONFIRMED NON-STATIONARY.** Both ADF and KPSS agree. The MPR might seem stable because the CBN holds it constant for months at a time (the Monetary Policy Committee meets roughly every two months). But over the full 2000-2024 sample, the MPR shows a clear pattern: it was 13.5% in 2000, fell to 6% in 2009, rose to 14% in 2014, fell again to 11.5% in 2020, then surged to 27.5% by late 2024 as the CBN aggressively tightened policy. These large, persistent shifts make the overall series non-stationary.

**In first differences: CONFIRMED STATIONARY.** The month-to-month change in the MPR is mostly zero (when the rate is held constant) with occasional non-zero jumps (when the MPC changes it). This differenced series fluctuates around a stable mean near zero, which is the definition of stationarity.

**Integration order: I(1).**

### infl (Headline Inflation Rate %)

**In levels: CONFIRMED NON-STATIONARY.** Inflation climbed from about 6% in early 2000 to above 33% by 2024, with a strong upward acceleration since 2020. The mean inflation in the first half of the sample is dramatically lower than in the second half. The ADF test could not reject the unit root, and the KPSS test rejected stationarity.

**In first differences: CONFIRMED STATIONARY.** The month-to-month change in inflation fluctuates around zero without drifting in one direction. Even though there are occasional sharp spikes (like the jump after the June 2023 exchange rate unification), the differenced series does not trend upward or downward over time.

**Integration order: I(1).**

### exo (Official Exchange Rate, naira per dollar)

**In levels: CONFIRMED NON-STATIONARY.** The exchange rate has the most extreme non-stationarity in the dataset. It went from about 92 naira per dollar in 2000 to over 1,500 in 2024, with massive structural breaks in June 2016 (when the CBN allowed the naira to float from its rigid peg) and June 2023 (when the new Tinubu government unified the exchange rate windows). The KPSS test statistic for this variable is very large, and the ADF p-value is very high. Both tests strongly agree: non-stationary.

**In first differences: CONFIRMED STATIONARY.** The month-to-month change in the exchange rate does not trend persistently in one direction. There are occasional large jumps (June 2016, June 2023), but between those jumps the differenced series reverts to a mean near zero.

**Integration order: I(1).**

### tbr (Treasury Bill Rate %)

**In levels: CONFIRMED NON-STATIONARY.** The Treasury Bill Rate tracks the MPR closely but has more month-to-month variation because it is set by market auctions rather than a committee. Like the MPR, the TBR shows a strong upward trend in 2022-2024 as monetary policy tightened. Over the full sample, it does not revert to a fixed mean.

**In first differences: CONFIRMED STATIONARY.** The month-to-month change in the TBR fluctuates around zero. Rate movements in one direction are typically followed by movements in the other direction, without a persistent drift.

**Integration order: I(1).**

### Why All Being I(1) Is Good News for ARDL Bounds Testing

This is the key takeaway from today. All four variables are I(1), and none are I(2). This finding has three important consequences:

1. **The ARDL bounds testing framework is valid.** The Pesaran, Shin, and Smith (2001) bounds test explicitly requires that all variables be either I(0) or I(1). If even one variable were I(2), the entire framework would be invalid and you would need a different approach. Since all your variables are I(1), ARDL is appropriate.

2. **Cointegration testing is the natural next step.** When you have multiple I(1) variables, the key question becomes: do they share a long-run equilibrium relationship? If they do (cointegration), you can model both the short-run dynamics and the long-run equilibrium in a single framework. If they do not, you can only model the short-run dynamics by working with first differences. We test for cointegration starting tomorrow.

3. **You cannot run simple OLS in levels.** Regressing infl on mpr, exo, and tbr in their original (undifferenced) form would produce spurious results. The high R-squared and significant coefficients would be artifacts of common trends, not genuine relationships. This is the Granger-Newbold (1974) spurious regression problem. The stationarity tests you have done today formally justify why you use ARDL or error correction models instead of naive OLS.

### What to Tell Your Examiner

> "Using the confirmatory ADF-KPSS strategy, all four variables -- the monetary policy rate (mpr), headline inflation (infl), the official exchange rate (exo), and the Treasury Bill rate (tbr) -- are confirmed to be integrated of order one, I(1), at the 5% significance level. Both the ADF test (null: unit root exists) and the KPSS test (null: series is stationary) agree on the classification for all variables in both levels and first differences. The absence of any I(2) variable validates the applicability of the ARDL bounds testing approach for cointegration analysis."

This is a statement you can put directly into your thesis methodology section.

---

## Commit

```bash
git add econometric_models/stationarity.py results/integration_order.csv
git commit -m "Day 7: KPSS test, confirmatory ADF+KPSS strategy, integration order determination"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `InterpolationWarning: The test statistic is outside of the range of p-values available in the look-up table` | This is the KPSS bounded p-value warning. Our code suppresses it with `warnings.catch_warnings()`. If you see this warning, make sure the `kpss()` call is INSIDE the `with warnings.catch_warnings():` block, not outside it. Check that the indentation is correct -- the `kpss()` line must be indented further than the `with` line. |
| KPSS p-values show exactly `0.01` or `0.10` for every variable | This is expected behavior, not an error. The KPSS test uses a lookup table of critical values rather than computing an exact p-value. When the true p-value is below 0.01, statsmodels reports `0.01`. When it is above 0.10, statsmodels reports `0.10`. For levels (where our variables are strongly non-stationary), you will see `0.01`. For differences (where they are clearly stationary), you will see `0.10`. These truncated values are still perfectly valid for our conclusions because `0.01 < 0.05` (reject stationarity) and `0.10 > 0.05` (fail to reject stationarity). |
| `TypeError: 'NoneType' object is not iterable` when unpacking KPSS results | The KPSS function returns exactly four values: `stat, p_value, lags, critical_values`. Make sure you unpack exactly four. The ADF function returns six values -- do not confuse them. |
| `ValueError: regression must be 'c' or 'ct'` | You passed a wrong value for the `regression` parameter. It must be exactly `"c"` (lower-case letter c in quotes) or `"ct"`. Check for typos like `"C"`, `"constant"`, or missing quotes. |
| `ModuleNotFoundError: No module named 'statsmodels'` | statsmodels was installed on Day 6. Run `pip install -r requirements.txt` again. Make sure your virtual environment is activated first. |
| `FileNotFoundError` when loading `cleaned_data.csv` | The cleaned data file does not exist yet. Run `python -m data_processing.clean` first (from Day 4). The stationarity script reads from `data/processed/cleaned_data.csv`. |
| `KeyError: 'infl'` or `KeyError: 'mpr'` | Your cleaned CSV does not have the expected column names. Open `data/processed/cleaned_data.csv` and check the header row. The columns must be exactly: `mpr`, `infl`, `exo`, `tbr` with `date` as the index. |
| `PermissionError` when saving to `results/` | The script cannot create or write to the results folder. On Linux/Mac, try `chmod -R 755 results/`. On Windows, check that the folder is not read-only. |

---

## Check Your Understanding

These are the kinds of questions an examiner will ask about your stationarity testing strategy. Practice answering them before your defense.

### 1. "Why do you use both the ADF test and the KPSS test instead of just one?"

**Answer:** "The ADF and KPSS tests have opposite null hypotheses, which makes them a natural confirmatory pair. The ADF test assumes non-stationarity as its null -- it needs strong evidence to conclude a series is stationary. The KPSS test assumes stationarity as its null -- it needs strong evidence to conclude a series is non-stationary. The ADF test is known to have low statistical power, meaning it sometimes fails to detect stationarity when stationarity is actually present. By running both tests, I get a cross-check: if the ADF test says non-stationary AND the KPSS test also says non-stationary, then both tests agree despite starting from opposite assumptions, and I can be confident in the conclusion. If the two tests disagree, it flags an ambiguous case that may require further investigation with structural break tests."

### 2. "The KPSS test reported a p-value of exactly 0.01. Is that the true p-value?"

**Answer:** "No. The KPSS test in statsmodels uses a lookup table of critical values rather than computing an exact p-value from a continuous distribution. The lookup table covers p-values between 0.01 and 0.10. When the true p-value falls below 0.01, statsmodels reports it as 0.01. When the true p-value is above 0.10, it reports 0.10. So a reported p-value of 0.01 means the true p-value is at most 0.01 -- it could be much smaller. For our purposes this does not matter, because all we need to know is whether the p-value is above or below the 0.05 significance threshold. A p-value of 0.01 is clearly below 0.05, so we reject the null hypothesis of stationarity and conclude the series is non-stationary in levels."

### 3. "What would it mean for your analysis if one of the variables were I(2)?"

**Answer:** "If any variable were I(2), it would mean that even after first differencing, the series remains non-stationary -- you would need to difference it twice to achieve stationarity. This would be a serious problem for our project because the ARDL bounds testing framework, developed by Pesaran, Shin, and Smith in 2001, explicitly requires all variables to be either I(0) or I(1). An I(2) variable would invalidate the ARDL approach entirely, and I would need to either transform the variable (for example, by taking its logarithm) to reduce it to I(1), or switch to a different modelling framework that can handle I(2) variables. Fortunately, all four of our variables are confirmed I(1), so the ARDL bounds test is valid."

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| KPSS test function | `econometric_models/stationarity.py` | Tests stationarity with opposite null hypothesis to ADF |
| Integration order function | `econometric_models/stationarity.py` | Runs ADF+KPSS at levels and 1st differences, determines I(d) |
| Integration order CSV | `results/integration_order.csv` | All p-values and integration orders, ready for thesis appendix |

**Key finding:** All four variables (mpr, infl, exo, tbr) are confirmed I(1) by both the ADF test and the KPSS test. No variable is I(0) or I(2). This validates the ARDL bounds testing approach that we will use later.

**Tomorrow (Day 8):** We test whether these I(1) variables are **cointegrated** -- whether they share a long-run equilibrium relationship even though each one individually wanders. This is the Engle-Granger cointegration test. If cointegration exists, it means the variables drift apart in the short run but are pulled back together over time by some economic force. That finding would justify modelling both short-run dynamics and long-run equilibrium in a single framework, which is far more informative than just modelling first differences.
