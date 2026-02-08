# Week 2, Day 7 — KPSS Test & Determining Integration Order

## What You'll Learn Today

- Why one stationarity test is not enough
- How the KPSS test works and how it differs from the ADF test
- The confirmatory ADF-KPSS strategy for reliable stationarity conclusions
- What integration order means: I(0), I(1), I(2)
- How to automatically determine integration order for each variable
- Why all of this matters for the ARDL bounds test

## Why This Matters

Yesterday you ran the ADF test on all four variables and found that each one is non-stationary in levels but stationary in first differences. That is a useful result, but it is only half the picture.

The ADF test has a known weakness: it has **low power**. "Low power" means that when a series really IS stationary, the ADF test sometimes fails to detect it. In technical terms, the ADF test has a high probability of Type II error (failing to reject a false null hypothesis). So when the ADF test says "I cannot reject non-stationarity," you cannot be fully confident. Maybe the series really is non-stationary. Or maybe the series is stationary and the ADF test just was not powerful enough to detect it.

The solution is to run a second test that approaches the question from the opposite direction. That test is the KPSS test.

---

## No New Packages

The KPSS test is included in `statsmodels`, which you installed on Day 6. You do not need to install anything new today.

---

## Theory: Why We Need BOTH ADF and KPSS

This section is long. Read it carefully. If your examiner asks about stationarity testing (and they will), this is what you need to know.

### The Core Problem: Opposite Null Hypotheses

Every statistical hypothesis test has a **null hypothesis** (H0) -- the default assumption -- and an **alternative hypothesis** (H1). The test tries to find evidence against H0. If it finds enough evidence, you reject H0. If it does not, you fail to reject H0.

Here is the critical difference between the two tests:

**ADF Test (Augmented Dickey-Fuller):**
- H0 (null): The series IS non-stationary (has a unit root)
- H1 (alternative): The series IS stationary (no unit root)
- If p-value < 0.05: reject H0 -- conclude STATIONARY
- If p-value >= 0.05: fail to reject H0 -- conclude NON-STATIONARY

**KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin):**
- H0 (null): The series IS stationary (trend-stationary)
- H1 (alternative): The series IS non-stationary (has a unit root)
- If p-value < 0.05: reject H0 -- conclude NON-STATIONARY
- If p-value >= 0.05: fail to reject H0 -- conclude STATIONARY

Notice they are mirror images. The ADF test assumes non-stationarity and looks for evidence of stationarity. The KPSS test assumes stationarity and looks for evidence of non-stationarity. Using them together is like asking two witnesses who are predisposed to give opposite answers. If they BOTH point in the same direction, you can be confident.

### The Confirmatory Strategy

This approach was formalized by Kwiatkowski, Phillips, Schmidt, and Shin in their 1992 paper. Using both tests together gives you four possible outcomes:

| ADF Result | KPSS Result | Conclusion |
|------------|-------------|------------|
| Fail to reject H0 (non-stationary) | Reject H0 (non-stationary) | **Confirmed non-stationary** |
| Reject H0 (stationary) | Fail to reject H0 (stationary) | **Confirmed stationary** |
| Reject H0 (stationary) | Reject H0 (non-stationary) | **Inconclusive** -- both say different things |
| Fail to reject H0 (non-stationary) | Fail to reject H0 (stationary) | **Inconclusive** -- neither test is decisive |

The first two rows are the clear cases. When ADF and KPSS agree, you have a confirmed result. The last two rows are the ambiguous cases -- they can arise when a series has a structural break (a sudden shift in level or trend), which confuses both tests. If you encounter an inconclusive case, you may need structural break tests (like the Zivot-Andrews test), but that is beyond the scope of this project.

For Nigerian macroeconomic data, we expect clear results: all four variables should be confirmed non-stationary in levels and confirmed stationary in first differences.

### A Note on KPSS P-Values

The KPSS test in statsmodels reports bounded p-values. This means:

- If the true p-value is greater than 0.10, statsmodels reports it as **0.10**
- If the true p-value is less than 0.01, statsmodels reports it as **0.01**

This happens because the KPSS test uses a lookup table of critical values rather than computing an exact p-value. Outside the table's range, the best it can say is "greater than 0.10" or "less than 0.01."

Statsmodels also prints a warning about this. We will suppress that warning in our code because we handle the bounded p-value ourselves.

### Integration Order

The integration order of a variable tells you how many times you need to difference it before it becomes stationary:

- **I(0)** -- integrated of order zero. The series is already stationary in levels. No differencing needed.
- **I(1)** -- integrated of order one. The series is non-stationary in levels but becomes stationary after one round of differencing (subtracting each value from the previous one).
- **I(2)** -- integrated of order two. The series needs two rounds of differencing to become stationary. This is rare in practice and usually signals a problem (exponential growth that even differencing once cannot tame).

Yesterday's ADF test suggested all four variables are I(1). Today we confirm this with the KPSS test. The confirmation matters because our modelling strategy depends on it:

- **For ARDL bounds testing (Week 3-4):** The ARDL framework requires all variables to be I(0) or I(1). It does NOT work with I(2) variables. If any variable were I(2), we would need a different approach entirely.
- **For VAR modelling (Week 5-6):** If variables are I(1) and cointegrated, we estimate a Vector Error Correction Model (VECM). If they are I(1) but not cointegrated, we estimate a VAR in first differences.

So today's work is not just a box-ticking exercise. It directly determines which models we can and cannot use.

---

## Building `econometric_models/stationarity.py` -- Step by Step

We are expanding the file you built yesterday. At each step, we show you the **complete file** from the first line to the last line. Delete everything and replace it with exactly what is shown.

Make sure you have the `econometric_models/` folder with an `__init__.py` inside it (from Day 6). If not, create them now.

---

### Build Step 1: Add the KPSS Test Function

This step takes your Day 6 file and adds the KPSS test alongside the existing ADF test. Delete everything in `econometric_models/stationarity.py` and replace it with this:

```python
import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    """Load the cleaned dataset."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def adf_test(series, name):
    """
    Run the Augmented Dickey-Fuller test on a series.

    Null hypothesis: the series IS non-stationary (has a unit root).
    If p-value < 0.05, reject null -> series is STATIONARY.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    stat, p_value, usedlag, nobs, critical_values, icbest = result

    print(f"\n{'='*50}")
    print(f"ADF Test: {name}")
    print(f"{'='*50}")
    print(f"Test Statistic:  {stat:.4f}")
    print(f"P-Value:         {p_value:.4f}")
    print(f"Lags Used:       {usedlag}")
    print(f"Observations:    {nobs}")
    for level, cv in critical_values.items():
        print(f"Critical Value ({level}): {cv:.4f}")

    if p_value < 0.05:
        print(f"Conclusion: STATIONARY (reject H0)")
    else:
        print(f"Conclusion: NON-STATIONARY (fail to reject H0)")

    return {"test_statistic": stat, "p_value": p_value, "stationary": p_value < 0.05}


def kpss_test(series, name):
    """
    Run the KPSS test on a series.

    Null hypothesis: the series IS stationary (trend-stationary).
    If p-value < 0.05, reject null -> series is NON-STATIONARY.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_value, lags, critical_values = kpss(series.dropna(), regression="c", nlags="auto")

    print(f"\n{'='*50}")
    print(f"KPSS Test: {name}")
    print(f"{'='*50}")
    print(f"Test Statistic:  {stat:.4f}")
    print(f"P-Value:         {p_value:.4f}")
    print(f"Lags Used:       {lags}")
    for level, cv in critical_values.items():
        print(f"Critical Value ({level}): {cv:.4f}")

    if p_value < 0.05:
        print(f"Conclusion: NON-STATIONARY (reject H0)")
    else:
        print(f"Conclusion: STATIONARY (fail to reject H0)")

    return {"test_statistic": stat, "p_value": p_value, "stationary": p_value >= 0.05}


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows\n")

    variables = ["inflation", "mpr", "exchange_rate", "m2"]

    for var in variables:
        adf_result = adf_test(df[var], var)
        kpss_result = kpss_test(df[var], var)

        # Confirmatory interpretation
        print(f"\n--- Confirmatory Result for {var} (Level) ---")
        if not adf_result["stationary"] and not kpss_result["stationary"]:
            print(f"ADF: non-stationary | KPSS: non-stationary -> CONFIRMED NON-STATIONARY")
        elif adf_result["stationary"] and kpss_result["stationary"]:
            print(f"ADF: stationary | KPSS: stationary -> CONFIRMED STATIONARY")
        else:
            print(f"ADF and KPSS disagree -> INCONCLUSIVE")
```

**Run it:**

```bash
python -m econometric_models.stationarity
```

**What you should see:**

For each of the four variables, you get two test outputs (ADF and KPSS) followed by a confirmatory interpretation. For all four variables in levels, you should see:

```
--- Confirmatory Result for inflation (Level) ---
ADF: non-stationary | KPSS: non-stationary -> CONFIRMED NON-STATIONARY
```

This means both tests agree: each variable is non-stationary in levels. The ADF test failed to reject its null (non-stationarity), and the KPSS test rejected its null (stationarity). Both point the same direction.

**What just happened -- line by line:**

- `from statsmodels.tsa.stattools import adfuller, kpss` -- We now import BOTH test functions from the same module.
- `import warnings` and `warnings.catch_warnings()` -- The KPSS function in statsmodels raises an `InterpolationWarning` whenever the p-value falls outside its lookup table. This happens frequently -- almost every time you run the test -- so the warning is more noise than signal. The `with warnings.catch_warnings()` block creates a temporary context where we suppress all warnings. Once the `with` block ends, warnings go back to normal. We do this INSIDE the function so it only affects the KPSS calculation, not the rest of the program.
- `regression="c"` -- This tells KPSS to test for level stationarity (constant only). The alternative is `regression="ct"` which tests for trend stationarity. For macroeconomic variables that may have a trend, `"c"` is the standard choice because we want to know if the series is stationary around a constant mean (not around a trend line).
- `nlags="auto"` -- This lets statsmodels automatically choose the number of lags for the KPSS test's spectral estimation. It uses the Schwert (1989) rule, which is the standard approach.
- `"stationary": p_value >= 0.05` -- Notice this is the OPPOSITE of the ADF function, which uses `p_value < 0.05`. This is because the KPSS null is stationarity. A high p-value means we fail to reject stationarity, so the series IS stationary. A low p-value means we reject stationarity, so the series is NOT stationary.

If all four variables show "CONFIRMED NON-STATIONARY," you are ready for the next step.

---

### Build Step 2: Add the Integration Order Function

Now we add a function that systematically determines the integration order of each variable. It tests levels first, then first differences, then second differences -- and stops as soon as both ADF and KPSS confirm stationarity.

Delete everything in `econometric_models/stationarity.py` and replace it with this:

```python
import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    """Load the cleaned dataset."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def adf_test(series, name):
    """
    Run the Augmented Dickey-Fuller test on a series.

    Null hypothesis: the series IS non-stationary (has a unit root).
    If p-value < 0.05, reject null -> series is STATIONARY.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    stat, p_value, usedlag, nobs, critical_values, icbest = result

    print(f"\n{'='*50}")
    print(f"ADF Test: {name}")
    print(f"{'='*50}")
    print(f"Test Statistic:  {stat:.4f}")
    print(f"P-Value:         {p_value:.4f}")
    print(f"Lags Used:       {usedlag}")
    print(f"Observations:    {nobs}")
    for level, cv in critical_values.items():
        print(f"Critical Value ({level}): {cv:.4f}")

    if p_value < 0.05:
        print(f"Conclusion: STATIONARY (reject H0)")
    else:
        print(f"Conclusion: NON-STATIONARY (fail to reject H0)")

    return {"test_statistic": stat, "p_value": p_value, "stationary": p_value < 0.05}


def kpss_test(series, name):
    """
    Run the KPSS test on a series.

    Null hypothesis: the series IS stationary (trend-stationary).
    If p-value < 0.05, reject null -> series is NON-STATIONARY.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_value, lags, critical_values = kpss(series.dropna(), regression="c", nlags="auto")

    print(f"\n{'='*50}")
    print(f"KPSS Test: {name}")
    print(f"{'='*50}")
    print(f"Test Statistic:  {stat:.4f}")
    print(f"P-Value:         {p_value:.4f}")
    print(f"Lags Used:       {lags}")
    for level, cv in critical_values.items():
        print(f"Critical Value ({level}): {cv:.4f}")

    if p_value < 0.05:
        print(f"Conclusion: NON-STATIONARY (reject H0)")
    else:
        print(f"Conclusion: STATIONARY (fail to reject H0)")

    return {"test_statistic": stat, "p_value": p_value, "stationary": p_value >= 0.05}


def determine_integration_order(series, name, max_diff=2):
    """
    Determine the integration order of a series using ADF + KPSS.
    Tests levels first, then first difference, then second difference.
    """
    for d in range(max_diff + 1):
        if d == 0:
            test_series = series.dropna()
            label = f"{name} (Level)"
        else:
            test_series = series.diff(d).dropna() if d == 1 else series.diff().diff().dropna()
            label = f"{name} (d={d})"

        adf_result = adf_test(test_series, label)
        kpss_result = kpss_test(test_series, label)

        if adf_result["stationary"] and kpss_result["stationary"]:
            print(f"\n>>> {name} is I({d}) — both ADF and KPSS confirm stationarity at d={d}")
            return d

    print(f"\n>>> {name}: Could not determine integration order (may be I({max_diff}+))")
    return max_diff + 1


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows\n")

    variables = ["inflation", "mpr", "exchange_rate", "m2"]

    for var in variables:
        order = determine_integration_order(df[var], var)
        print(f"\n{'*'*50}")
        print(f"RESULT: {var} is I({order})")
        print(f"{'*'*50}\n")
```

**Run it:**

```bash
python -m econometric_models.stationarity
```

**What you should see:**

For each variable, the function first tests levels. Both ADF and KPSS say non-stationary, so it does NOT stop there. It then tests first differences. Both ADF and KPSS say stationary, so it stops and reports I(1).

```
>>> inflation is I(1) — both ADF and KPSS confirm stationarity at d=1

**************************************************
RESULT: inflation is I(1)
**************************************************
```

You should see this pattern for all four variables. None should be I(0) in levels (they all have trends), and none should need second differencing (I(2)).

**How the determine_integration_order function works:**

- The `for d in range(max_diff + 1)` loop tries d=0 (levels), d=1 (first difference), d=2 (second difference).
- For d=0, it uses the raw series. For d=1, it uses `series.diff(1)` which computes the change from one period to the next. For d=2, it uses `series.diff().diff()` which differences twice.
- `.dropna()` is essential because differencing creates NaN values. The first observation after one differencing has no predecessor to subtract from, so it becomes NaN. After two differencings, the first TWO observations are NaN. We drop these before testing.
- As soon as BOTH ADF and KPSS agree the series is stationary, the function returns the current differencing order `d` and stops. It does not test further.
- If neither 0, 1, nor 2 differencings produce confirmed stationarity, it returns `max_diff + 1` as a flag that something unusual is going on.

If all four variables show I(1), you are ready for the final step.

---

### Build Step 3: Add Summary Table and CSV Export (Final Version)

This is the final, complete version of the file. It adds a summary table that you can paste into your thesis and saves the results to a CSV file for future reference.

Delete everything in `econometric_models/stationarity.py` and replace it with this:

```python
import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    """Load the cleaned dataset."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def adf_test(series, name):
    """
    Run the Augmented Dickey-Fuller test on a series.

    Null hypothesis: the series IS non-stationary (has a unit root).
    If p-value < 0.05, reject null -> series is STATIONARY.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    stat, p_value, usedlag, nobs, critical_values, icbest = result

    print(f"\n{'='*50}")
    print(f"ADF Test: {name}")
    print(f"{'='*50}")
    print(f"Test Statistic:  {stat:.4f}")
    print(f"P-Value:         {p_value:.4f}")
    print(f"Lags Used:       {usedlag}")
    print(f"Observations:    {nobs}")
    for level, cv in critical_values.items():
        print(f"Critical Value ({level}): {cv:.4f}")

    if p_value < 0.05:
        print(f"Conclusion: STATIONARY (reject H0)")
    else:
        print(f"Conclusion: NON-STATIONARY (fail to reject H0)")

    return {"test_statistic": stat, "p_value": p_value, "stationary": p_value < 0.05}


def kpss_test(series, name):
    """
    Run the KPSS test on a series.

    Null hypothesis: the series IS stationary (trend-stationary).
    If p-value < 0.05, reject null -> series is NON-STATIONARY.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_value, lags, critical_values = kpss(series.dropna(), regression="c", nlags="auto")

    print(f"\n{'='*50}")
    print(f"KPSS Test: {name}")
    print(f"{'='*50}")
    print(f"Test Statistic:  {stat:.4f}")
    print(f"P-Value:         {p_value:.4f}")
    print(f"Lags Used:       {lags}")
    for level, cv in critical_values.items():
        print(f"Critical Value ({level}): {cv:.4f}")

    if p_value < 0.05:
        print(f"Conclusion: NON-STATIONARY (reject H0)")
    else:
        print(f"Conclusion: STATIONARY (fail to reject H0)")

    return {"test_statistic": stat, "p_value": p_value, "stationary": p_value >= 0.05}


def determine_integration_order(series, name, max_diff=2):
    """
    Determine the integration order of a series using ADF + KPSS.
    Tests levels first, then first difference, then second difference.
    """
    for d in range(max_diff + 1):
        if d == 0:
            test_series = series.dropna()
            label = f"{name} (Level)"
        else:
            test_series = series.diff(d).dropna() if d == 1 else series.diff().diff().dropna()
            label = f"{name} (d={d})"

        adf_result = adf_test(test_series, label)
        kpss_result = kpss_test(test_series, label)

        if adf_result["stationary"] and kpss_result["stationary"]:
            print(f"\n>>> {name} is I({d}) — both ADF and KPSS confirm stationarity at d={d}")
            return d

    print(f"\n>>> {name}: Could not determine integration order (may be I({max_diff}+))")
    return max_diff + 1


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows\n")

    variables = ["inflation", "mpr", "exchange_rate", "m2"]
    results = []

    for var in variables:
        # --- Test in levels ---
        level_series = df[var].dropna()
        adf_level = adf_test(level_series, f"{var} (Level)")
        kpss_level = kpss_test(level_series, f"{var} (Level)")

        # --- Test in first differences ---
        diff_series = df[var].diff().dropna()
        adf_diff = adf_test(diff_series, f"{var} (d=1)")
        kpss_diff = kpss_test(diff_series, f"{var} (d=1)")

        # --- Determine integration order ---
        if adf_diff["stationary"] and kpss_diff["stationary"]:
            order = 1
        elif adf_level["stationary"] and kpss_level["stationary"]:
            order = 0
        else:
            order = 2

        results.append({
            "variable": var,
            "adf_level_pvalue": round(adf_level["p_value"], 4),
            "kpss_level_pvalue": round(kpss_level["p_value"], 4),
            "adf_diff_pvalue": round(adf_diff["p_value"], 4),
            "kpss_diff_pvalue": round(kpss_diff["p_value"], 4),
            "integration_order": order,
        })

    # --- Print summary table ---
    print("\n")
    print("=" * 90)
    print("INTEGRATION ORDER SUMMARY")
    print("=" * 90)
    print(f"{'Variable':<18} {'ADF Level':<14} {'KPSS Level':<14} {'ADF Diff':<14} {'KPSS Diff':<14} {'I(d)':<6}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['variable']:<18} "
            f"{r['adf_level_pvalue']:<14} "
            f"{r['kpss_level_pvalue']:<14} "
            f"{r['adf_diff_pvalue']:<14} "
            f"{r['kpss_diff_pvalue']:<14} "
            f"I({r['integration_order']})"
        )
    print("=" * 90)
    print("\nInterpretation:")
    print("  ADF Level p >= 0.05  -> non-stationary in levels (fail to reject unit root)")
    print("  KPSS Level p < 0.05  -> non-stationary in levels (reject stationarity)")
    print("  ADF Diff p < 0.05    -> stationary in differences (reject unit root)")
    print("  KPSS Diff p >= 0.05  -> stationary in differences (fail to reject stationarity)")
    print("  I(1) = one differencing needed to achieve stationarity")

    # --- Save to CSV ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df = pd.DataFrame(results)
    output_path = os.path.join(RESULTS_DIR, "integration_order.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: results/integration_order.csv")
```

Build Step 3 above is your final complete file.

**Run it:**

```bash
python -m econometric_models.stationarity
```

**What you should see:**

After all the individual test outputs, you will see a clean summary table:

```
==========================================================================================
INTEGRATION ORDER SUMMARY
==========================================================================================
Variable           ADF Level      KPSS Level     ADF Diff       KPSS Diff      I(d)
------------------------------------------------------------------------------------------
inflation          0.XXXX         0.0100         0.XXXX         0.1000         I(1)
mpr                0.XXXX         0.0100         0.XXXX         0.1000         I(1)
exchange_rate      0.XXXX         0.0100         0.XXXX         0.1000         I(1)
m2                 0.XXXX         0.0100         0.XXXX         0.1000         I(1)
==========================================================================================

Interpretation:
  ADF Level p >= 0.05  -> non-stationary in levels (fail to reject unit root)
  KPSS Level p < 0.05  -> non-stationary in levels (reject stationarity)
  ADF Diff p < 0.05    -> stationary in differences (reject unit root)
  KPSS Diff p >= 0.05  -> stationary in differences (fail to reject stationarity)
  I(1) = one differencing needed to achieve stationarity

Results saved to: results/integration_order.csv
```

(The `0.XXXX` values will be actual p-values when you run it. The KPSS values will likely show `0.0100` for levels and `0.1000` for differences because of the bounded p-value behavior explained earlier.)

**What the summary table tells you:**

- **ADF Level column:** These p-values should all be ABOVE 0.05 (non-stationary in levels). The ADF test could not reject the null of a unit root for any variable.
- **KPSS Level column:** These p-values should all be BELOW 0.05 (non-stationary in levels). The KPSS test rejected the null of stationarity for every variable. ADF and KPSS agree: all variables are non-stationary in levels.
- **ADF Diff column:** These p-values should all be BELOW 0.05 (stationary in first differences). After one round of differencing, the ADF test rejects the unit root null.
- **KPSS Diff column:** These p-values should all be ABOVE 0.05 (stationary in first differences). After one round of differencing, the KPSS test cannot reject its stationarity null. ADF and KPSS agree again: all variables are stationary in first differences.
- **I(d) column:** All variables are I(1). Confirmed by both tests.

**What changed from Step 2:**

In Step 2, we used the `determine_integration_order` function which was clean but did not save intermediate p-values. In Step 3, the main block runs both tests at both levels and differences explicitly, stores every p-value, and builds a summary table. The `determine_integration_order` function is still in the file (it is useful as a reusable utility), but the main block does its own detailed analysis so we can build the complete summary table with all the p-values.

---

## Interpreting Results -- Nigeria Specific

All four variables -- inflation, MPR, exchange rate, and M2 -- are confirmed as I(1) by the ADF-KPSS confirmatory strategy. Here is what this means for your project.

### What I(1) Means in Plain Language

An I(1) variable is one that "wanders" -- it does not return to a fixed mean over time. If you look at the time series plots from Day 5, this is visually obvious. The exchange rate does not fluctuate around some average value -- it trends upward and then jumps. M2 does not oscillate -- it grows exponentially. Even inflation, which you might expect to cycle, has shown a persistent upward drift since 2020.

When you take first differences of an I(1) variable, you are looking at the month-to-month CHANGES rather than the levels. The change in the exchange rate from one month to the next does fluctuate around zero (some months it goes up, some months down). The change in M2 from one month to the next is roughly stable. These differenced series ARE stationary -- they have a stable mean, stable variance, and no persistent trends.

### Why I(1) Is the Most Common Finding

Almost all macroeconomic time series are I(1). This is not a coincidence. Macroeconomic variables like GDP, prices, money supply, and exchange rates are cumulative -- today's value builds on yesterday's value. The price level does not reset each month; it accumulates changes. M2 does not start from zero each quarter; new money is added to the existing stock. This cumulative nature is exactly what makes a series I(1): each value equals the previous value plus a random shock, and those shocks accumulate over time.

### What This Means for Our Modelling Strategy

Since all four variables are I(1), we face the classic macro-econometric question:

1. **Option A: Difference everything.** We could take first differences of all variables and model the relationships between the CHANGES. A VAR in differences would work. But this throws away information about the LEVELS of the variables -- we lose the ability to say anything about long-run equilibrium relationships.

2. **Option B: Test for cointegration.** If two or more I(1) variables share a common stochastic trend, they are **cointegrated**. Cointegration means that even though each variable individually wanders (is non-stationary), there exists a linear combination of them that IS stationary. In economic terms, the variables share a long-run equilibrium -- they may drift apart in the short run, but they are pulled back together over time.

   If cointegration exists, we can model both the short-run dynamics AND the long-run equilibrium using an Error Correction Model (ECM) or ARDL bounds test. This is far more informative than just modelling differences.

We will test for cointegration on Days 8-9. If the variables are cointegrated (which is likely, given the strong economic theory linking money supply, exchange rates, monetary policy, and inflation in Nigeria), we will use the ARDL bounds testing framework in Weeks 3-4.

### Why No Variable Being I(2) Is Important

If any variable were I(2), it would be a serious problem:

- The ARDL bounds test framework (Pesaran, Shin, and Smith, 2001) explicitly requires all variables to be I(0) or I(1). An I(2) variable would invalidate the approach entirely.
- I(2) variables are hard to handle in any framework. They imply that even the CHANGES in a variable are non-stationary, meaning the changes themselves trend over time. For economic variables, this would mean accelerating growth or accelerating decline with no bound -- which is economically implausible over long horizons.

Today's results confirm that none of our variables are I(2), which validates the ARDL bounds testing approach we will use later.

### What to Tell Your Examiner

"Using the confirmatory ADF-KPSS strategy, all four variables are confirmed to be integrated of order one, I(1). No variable is I(2), which validates the use of the ARDL bounds testing framework. The I(1) finding for all variables also motivates Johansen cointegration testing to determine whether long-run equilibrium relationships exist among the variables."

---

## Commit

```bash
git add econometric_models/stationarity.py results/integration_order.csv
git commit -m "Day 7: Add KPSS test and integration order determination"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `InterpolationWarning: The test statistic is outside of the range of p-values available in the look-up table` | This is the KPSS bounded p-value warning. Our code suppresses it with `warnings.catch_warnings()`. If you see it, make sure the `with warnings.catch_warnings():` and `warnings.simplefilter("ignore")` lines are inside your `kpss_test` function and that the `kpss()` call is INSIDE the `with` block, not outside it. |
| `ModuleNotFoundError: No module named 'statsmodels'` | You did not install statsmodels. Run `pip install statsmodels==0.14.1` and try again. This was installed on Day 6. |
| `TypeError: 'NoneType' object is not iterable` when unpacking KPSS results | Make sure you are unpacking exactly four values: `stat, p_value, lags, critical_values = kpss(...)`. The KPSS function returns four items, not six like ADF. |
| `ValueError: regression must be 'c' or 'ct'` | You passed a wrong value for the `regression` parameter in `kpss()`. It must be exactly `"c"` (constant) or `"ct"` (constant + trend). Check for typos. Our code uses `"c"`. |
| `KeyError: 'inflation'` or similar column name error | Your cleaned CSV does not have the expected column names. Open `data/processed/cleaned_data.csv` and check the header row. The columns must be exactly: `date`, `mpr`, `inflation`, `exchange_rate`, `m2`. If they are different, go back to Day 4 and re-run the cleaning script. |
| KPSS p-values show exactly `0.01` or `0.10` for every variable | This is expected behaviour, not an error. The KPSS test uses a lookup table with a limited range. P-values below 0.01 are reported as 0.01, and p-values above 0.10 are reported as 0.10. For levels, you should see 0.01 (strongly non-stationary). For differences, you should see 0.10 (clearly stationary). These bounded values are still perfectly valid for drawing conclusions. |

---

## Check Your Understanding

These are the kinds of questions an examiner will ask about your stationarity testing. Practice answering them out loud.

### 1. "Why use both ADF and KPSS instead of just one?"

**Answer:** "The ADF and KPSS tests have opposite null hypotheses. The ADF test assumes non-stationarity as its null -- it requires evidence to confirm stationarity. The KPSS test assumes stationarity as its null -- it requires evidence to confirm non-stationarity. Using only one test means you are only asking one question. Using both provides a confirmatory strategy: if ADF says non-stationary AND KPSS also says non-stationary, both tests point the same direction despite starting from opposite assumptions, so you can be confident in the conclusion. If the two tests disagree, it signals that the result is ambiguous and may require additional investigation, such as structural break tests."

### 2. "What is the difference between I(0), I(1), and I(2)?"

**Answer:** "The integration order tells you how many times a series must be differenced to achieve stationarity. I(0) means the series is already stationary in levels -- it fluctuates around a stable mean and has constant variance. No differencing is needed. I(1) means the series is non-stationary in levels but becomes stationary after one round of differencing -- that is, the month-to-month changes are stationary even though the levels are not. Most macroeconomic variables are I(1). I(2) means the series requires two rounds of differencing, implying that even the changes are non-stationary. I(2) is rare in economic data and would be problematic for our analysis because the ARDL bounds testing framework requires all variables to be I(0) or I(1)."

### 3. "All your variables are I(1). What does this imply for your modelling strategy?"

**Answer:** "Since all four variables are I(1), we cannot simply run an OLS regression in levels -- that would produce spurious regression results with inflated R-squared and misleading t-statistics. We have two options. The first is to difference all variables and model the relationships between the changes, but this discards information about long-run equilibrium relationships. The second, and preferred, option is to test for cointegration. If the I(1) variables are cointegrated -- meaning they share a common long-run equilibrium that they revert to over time -- then we can use an Error Correction Model or ARDL bounds test that captures both short-run dynamics and the long-run equilibrium relationship. The fact that all variables are I(1) and none are I(2) validates the use of the ARDL bounds testing framework, which is our primary modelling approach."

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| KPSS test function | `econometric_models/stationarity.py` | Tests stationarity with opposite null to ADF |
| Integration order function | `econometric_models/stationarity.py` | Automatically determines I(d) using both tests |
| Summary table | Printed to terminal | Shows all p-values and integration orders at a glance |
| Integration order CSV | `results/integration_order.csv` | Machine-readable record of stationarity results |

**Key finding:** All four variables (inflation, MPR, exchange rate, M2) are I(1) -- non-stationary in levels, stationary in first differences. Confirmed by both ADF and KPSS tests. No variable is I(2), validating the ARDL bounds testing approach.

**Tomorrow (Day 8):** We test whether these I(1) variables are cointegrated -- that is, whether they share a long-run equilibrium relationship despite each one being individually non-stationary. This is the Johansen cointegration test, and its result determines whether we model in differences only or can capture the long-run relationship.
