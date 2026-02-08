# Week 2, Day 9 — Johansen Cointegration Test: Testing All Variables Together

## What You'll Learn Today

- Why the Engle-Granger test from Day 8 is not enough
- How the Johansen test works and what it tells you that Engle-Granger cannot
- How to read the Trace test and Max Eigenvalue test tables
- What the cointegration rank means for your model choice
- How to check robustness by testing with different lag orders

## Why This Matters

On Day 8 you tested every pair of variables for cointegration using the Engle-Granger method. That gave you 6 pairwise results. But here is the problem: the Engle-Granger test only looks at TWO variables at a time. Your Nigerian inflation system has FOUR variables -- MPR, inflation, exchange rate, and M2 -- all interacting simultaneously. There could be multiple long-run equilibrium relationships tying these variables together, and the pairwise approach simply cannot see them.

The Johansen test fixes this. It examines all four variables at once and tells you exactly how many cointegrating relationships exist. This number -- called the cointegration rank -- is one of the most important results in your entire thesis, because it determines which class of econometric model you should use.

---

## Packages Installed Today: None

The `coint_johansen` function is already included in statsmodels, which you installed earlier. It lives in `statsmodels.tsa.vector_ar.vecm`. No new packages are needed.

---

## Theory: Johansen vs Engle-Granger

This section is long because the Johansen test is one of the most important tools in applied time series econometrics. If an examiner asks about cointegration, they will almost certainly ask about Johansen. Read this carefully.

### The Limitation of Engle-Granger

The Engle-Granger test you ran on Day 8 works like this: take two I(1) variables, run a regression of one on the other, and test whether the residuals are stationary. If the residuals are stationary, the two variables are cointegrated -- they share a long-run equilibrium.

This works fine when you only have two variables. But you have four: MPR, inflation, exchange_rate, and M2. With four I(1) variables, there can be **up to 3 cointegrating relationships** (the maximum is n-1, where n is the number of variables). The Engle-Granger test cannot detect this. It tests one pair at a time, so it can only ever find one relationship per test. Worse, the results can depend on which variable you put on the left side of the regression -- regressing inflation on exchange_rate can give a different answer than regressing exchange_rate on inflation.

In summary, Engle-Granger has three problems with more than two variables:

1. **It cannot determine how many cointegrating relationships exist.** It only tests for one at a time.
2. **It is sensitive to the normalization.** Which variable you put on the left side of the regression affects the result.
3. **It ignores the multivariate structure.** When four variables interact, you need a method that sees all four at once.

### The Johansen Test (1988, 1991)

Soren Johansen published two papers -- one in 1988 and one in 1991 -- that solved all three problems. His method tests ALL variables simultaneously in a multivariate framework. It determines three things:

1. **Whether cointegration exists at all** among the set of variables.
2. **How many cointegrating relationships exist** -- this number is called the **cointegration rank**, written as *r*.
3. **The cointegrating vectors themselves** -- the actual coefficients of the long-run equilibrium equations.

The Johansen test is based on a Vector Error Correction Model (VECM), which is a restricted form of the VAR model you will build later. The mathematical details involve eigenvalues of a matrix, which is why the results report "eigenvalue statistics." You do not need to understand the matrix algebra to use and interpret the test, but you do need to understand the two test statistics it produces.

### Two Test Statistics

The Johansen procedure produces two test statistics. Both test for the cointegration rank, but they frame the hypothesis differently.

**1. The Trace Test**

The Trace test asks: "Is the cointegration rank *greater than* r?"

- **Null hypothesis (H0):** The cointegration rank is less than or equal to r. That is, there are at most r cointegrating relationships.
- **Alternative hypothesis (H1):** The cointegration rank is greater than r. There are more than r cointegrating relationships.

The test starts at r=0 and works upward:

- First, test H0: rank <= 0 (no cointegration at all). If the trace statistic exceeds the critical value, reject H0. Conclusion: there IS cointegration. Move on.
- Next, test H0: rank <= 1 (at most one cointegrating relationship). If the trace statistic exceeds the critical value, reject again. Conclusion: there are at least 2 cointegrating relationships. Move on.
- Next, test H0: rank <= 2. And so on.
- When you fail to reject for the first time, you stop. The number of rejections is your cointegration rank.

**2. The Max Eigenvalue Test**

The Max Eigenvalue test asks: "Is the cointegration rank *exactly* r, or is it r+1?"

- **Null hypothesis (H0):** The cointegration rank is exactly r.
- **Alternative hypothesis (H1):** The cointegration rank is exactly r+1.

This is more precise than the Trace test -- it tests for exactly one additional relationship at each step, rather than "more than r." However, it tends to have less statistical power, especially in small samples.

**Which one to report?** Most applied papers report both, but if they disagree, the Trace test is generally preferred. Many textbooks (including Lutkepohl's "New Introduction to Multiple Time Series Analysis") recommend the Trace test for its better small-sample properties.

### How to Read the Results Table

When you run the Johansen test, you get a table that looks like this:

```
--- Trace Test ---
H0              Trace Stat      5% CV           Reject?
r <= 0          65.1234         47.8561         Yes
r <= 1          28.4567         29.7971         No
r <= 2          10.1234         15.4947         No
r <= 3          2.3456          3.8415          No
```

Reading this table from top to bottom:

1. **r <= 0:** The trace statistic (65.12) is greater than the 5% critical value (47.86). Reject H0. There IS cointegration.
2. **r <= 1:** The trace statistic (28.46) is less than the 5% critical value (29.80). Fail to reject. We cannot say there are more than 1 cointegrating relationships.

Since we rejected once (at r=0) and failed to reject at r=1, the cointegration rank is **1**. There is one cointegrating relationship among the four variables.

### What the Rank Means for Nigeria

With your 4 variables (MPR, inflation, exchange_rate, M2), the cointegration rank can be 0, 1, 2, or 3. Here is what each means:

- **r = 0 (no cointegration):** The four variables do not share any long-run equilibrium relationship. They wander independently in the long run. If this is the result, you should model the data using a VAR in first differences. This would be a disappointing result for your thesis because it means you cannot say anything about long-run relationships.

- **r = 1 (one cointegrating relationship):** There is one long-run equilibrium that ties the four variables together. This is the most common result for macroeconomic data and the most useful for your thesis. It means you can estimate a VECM (Vector Error Correction Model) or use the ARDL bounds test to capture both the long-run equilibrium and short-run dynamics. The single cointegrating relationship likely captures the monetary policy transmission mechanism: the CBN sets the MPR, which affects money supply, which affects the exchange rate, which affects inflation. All four are bound together in one long-run equilibrium.

- **r = 2 (two cointegrating relationships):** There are two independent long-run equilibria. This is a stronger result. For Nigeria, the second relationship might capture the exchange_rate-M2 link: the CBN's foreign exchange interventions create a separate equilibrium between the exchange rate and the money supply. With r=2, a VECM with two error correction terms is appropriate.

- **r = 3 (three cointegrating relationships):** Three independent equilibria. This is unusual with only 4 variables and would suggest the variables are very tightly linked in the long run. If you get r=3, double-check your data and lag specification before reporting it.

### Deterministic Terms

The Johansen test requires you to specify what deterministic terms (constants, trends) are included. The parameter `det_order` controls this:

- `det_order=0`: Constant in the cointegrating equation only (no trend). This is the most common specification for macroeconomic data and the one we use. It allows the long-run equilibrium to have a non-zero mean, which makes economic sense -- the long-run relationship between inflation and exchange rate is not centred at zero.

- `det_order=1`: Constant plus a linear trend in the cointegrating equation. Use this only if you believe there is a deterministic trend in the long-run relationship itself, not just in the individual variables.

- `det_order=-1`: No deterministic terms at all. Rarely used in practice.

We use `det_order=0` throughout this guide. If your examiner asks why, the answer is: "A constant-only specification is standard for macroeconomic data. It allows for a non-zero mean in the cointegrating relationship without imposing a deterministic trend, which would be difficult to justify economically."

---

## Building `econometric_models/cointegration.py` -- Step by Step

On Day 8, you built `econometric_models/cointegration.py` with the Engle-Granger pairwise tests. Today you will expand it to include the Johansen test. As always, each step shows the **complete file** from the first line to the last line. No snippets.

---

### Build Step 1: Add the Johansen Test Function

This step takes your Day 8 file and adds the `johansen_test()` function plus a main block that runs it on all four variables.

Delete everything in `econometric_models/cointegration.py` and replace it with this:

```python
import os
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def engle_granger_test(df, var1, var2):
    """
    Run the Engle-Granger two-step cointegration test on a pair of variables.

    Returns a dictionary with the test statistic, p-value, and whether
    the null hypothesis of no cointegration is rejected at 5%.
    """
    series1 = df[var1].dropna()
    series2 = df[var2].dropna()
    common_index = series1.index.intersection(series2.index)
    series1 = series1.loc[common_index]
    series2 = series2.loc[common_index]

    stat, pvalue, crit_values = coint(series1, series2)

    result = {
        "var1": var1,
        "var2": var2,
        "test_statistic": stat,
        "p_value": pvalue,
        "crit_1pct": crit_values[0],
        "crit_5pct": crit_values[1],
        "crit_10pct": crit_values[2],
        "cointegrated_5pct": pvalue < 0.05,
    }

    print(f"\nEngle-Granger: {var1} & {var2}")
    print(f"  Test statistic: {stat:.4f}")
    print(f"  P-value:        {pvalue:.4f}")
    print(f"  Critical values: 1%={crit_values[0]:.4f}, 5%={crit_values[1]:.4f}, 10%={crit_values[2]:.4f}")
    print(f"  Cointegrated at 5%: {'Yes' if pvalue < 0.05 else 'No'}")

    return result


def johansen_test(df, variables, det_order=0, k_ar_diff=1):
    """
    Run the Johansen cointegration test on multiple variables.

    Parameters:
    - df: DataFrame with the variables
    - variables: list of column names to test
    - det_order: deterministic term (0 = constant only, 1 = constant + trend)
    - k_ar_diff: number of lagged differences in the VECM (like lag order minus 1)
    """
    data = df[variables].dropna()
    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

    print(f"\n{'='*60}")
    print(f"JOHANSEN COINTEGRATION TEST")
    print(f"Variables: {', '.join(variables)}")
    print(f"Deterministic term: {'Constant only' if det_order == 0 else 'Constant + Trend'}")
    print(f"Lag order (k_ar_diff): {k_ar_diff}")
    print(f"{'='*60}")

    print(f"\n--- Trace Test ---")
    print(f"{'H0':<15} {'Trace Stat':<15} {'5% CV':<15} {'Reject?':<10}")
    for i in range(len(variables)):
        trace_stat = result.lr1[i]
        cv_5 = result.cvt[i, 1]  # 5% critical value
        reject = "Yes" if trace_stat > cv_5 else "No"
        print(f"r <= {i:<10} {trace_stat:<15.4f} {cv_5:<15.4f} {reject}")

    print(f"\n--- Max Eigenvalue Test ---")
    print(f"{'H0':<15} {'Max Eig Stat':<15} {'5% CV':<15} {'Reject?':<10}")
    for i in range(len(variables)):
        max_stat = result.lr2[i]
        cv_5 = result.cvm[i, 1]  # 5% critical value
        reject = "Yes" if max_stat > cv_5 else "No"
        print(f"r = {i:<11} {max_stat:<15.4f} {cv_5:<15.4f} {reject}")

    # Determine cointegration rank
    rank = 0
    for i in range(len(variables)):
        if result.lr1[i] > result.cvt[i, 1]:
            rank = i + 1
        else:
            break

    print(f"\nCointegration Rank (Trace): {rank}")
    return result, rank


if __name__ == "__main__":
    df = load_data()
    variables = ["mpr", "inflation", "exchange_rate", "m2"]

    print("=" * 60)
    print("JOHANSEN COINTEGRATION TEST")
    print("=" * 60)

    result, rank = johansen_test(df, variables, det_order=0, k_ar_diff=1)
    print(f"\nResult: Cointegration rank = {rank}")
```

**Run it:**

```bash
python -m econometric_models.cointegration
```

**What you should see:**

```
============================================================
JOHANSEN COINTEGRATION TEST
============================================================

============================================================
JOHANSEN COINTEGRATION TEST
Variables: mpr, inflation, exchange_rate, m2
Deterministic term: Constant only
Lag order (k_ar_diff): 1
============================================================

--- Trace Test ---
H0              Trace Stat      5% CV           Reject?
r <= 0          XX.XXXX         47.8561         Yes
r <= 1          XX.XXXX         29.7971         Yes/No
r <= 2          XX.XXXX         15.4947         No
r <= 3          XX.XXXX         3.8415          No

--- Max Eigenvalue Test ---
H0              Max Eig Stat    5% CV           Reject?
r = 0           XX.XXXX         27.5843         Yes
r = 1           XX.XXXX         21.1316         Yes/No
r = 2           XX.XXXX         14.2646         No
r = 3           XX.XXXX         3.8415          No

Cointegration Rank (Trace): 1 or 2

Result: Cointegration rank = 1 or 2
```

(Your exact numbers will depend on your data. The critical values are fixed -- they come from statistical tables built into statsmodels.)

**What just happened -- line by line:**

- `from statsmodels.tsa.vector_ar.vecm import coint_johansen` -- This imports the Johansen test function. It lives in the VECM (Vector Error Correction Model) module because the Johansen test is based on the VECM representation.
- `data = df[variables].dropna()` -- Selects only the columns we want to test and drops any rows with missing values. The Johansen test needs a complete rectangular matrix of data.
- `coint_johansen(data, det_order=0, k_ar_diff=1)` -- Runs the test. `det_order=0` means constant only (no trend). `k_ar_diff=1` means one lagged difference in the underlying VECM. This is NOT the same as the VAR lag order -- it is the VAR lag order minus 1. So `k_ar_diff=1` corresponds to a VAR(2).
- `result.lr1` -- The trace test statistics. This is an array with one value for each hypothesis (r<=0, r<=1, r<=2, r<=3).
- `result.cvt` -- The critical values for the trace test. `result.cvt[i, 1]` gives the 5% critical value for the i-th hypothesis. Column 0 is 10%, column 1 is 5%, column 2 is 1%.
- `result.lr2` -- The max eigenvalue test statistics.
- `result.cvm` -- The critical values for the max eigenvalue test.
- The rank determination loop starts at r=0 and counts how many consecutive rejections there are. The first time we fail to reject, we stop.

If this ran successfully, you now have the Johansen test working. But we need to check whether the result is robust to the lag specification. That is the next step.

---

### Build Step 2: Test Robustness with Different Lag Orders

The Johansen test can be sensitive to the number of lags you specify. A result that only holds for one specific lag order is fragile and should not be trusted. A robust result gives the same cointegration rank across different reasonable lag choices.

Delete everything in `econometric_models/cointegration.py` and replace it with this:

```python
import os
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def engle_granger_test(df, var1, var2):
    """
    Run the Engle-Granger two-step cointegration test on a pair of variables.

    Returns a dictionary with the test statistic, p-value, and whether
    the null hypothesis of no cointegration is rejected at 5%.
    """
    series1 = df[var1].dropna()
    series2 = df[var2].dropna()
    common_index = series1.index.intersection(series2.index)
    series1 = series1.loc[common_index]
    series2 = series2.loc[common_index]

    stat, pvalue, crit_values = coint(series1, series2)

    result = {
        "var1": var1,
        "var2": var2,
        "test_statistic": stat,
        "p_value": pvalue,
        "crit_1pct": crit_values[0],
        "crit_5pct": crit_values[1],
        "crit_10pct": crit_values[2],
        "cointegrated_5pct": pvalue < 0.05,
    }

    print(f"\nEngle-Granger: {var1} & {var2}")
    print(f"  Test statistic: {stat:.4f}")
    print(f"  P-value:        {pvalue:.4f}")
    print(f"  Critical values: 1%={crit_values[0]:.4f}, 5%={crit_values[1]:.4f}, 10%={crit_values[2]:.4f}")
    print(f"  Cointegrated at 5%: {'Yes' if pvalue < 0.05 else 'No'}")

    return result


def johansen_test(df, variables, det_order=0, k_ar_diff=1):
    """
    Run the Johansen cointegration test on multiple variables.

    Parameters:
    - df: DataFrame with the variables
    - variables: list of column names to test
    - det_order: deterministic term (0 = constant only, 1 = constant + trend)
    - k_ar_diff: number of lagged differences in the VECM (like lag order minus 1)
    """
    data = df[variables].dropna()
    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

    print(f"\n{'='*60}")
    print(f"JOHANSEN COINTEGRATION TEST")
    print(f"Variables: {', '.join(variables)}")
    print(f"Deterministic term: {'Constant only' if det_order == 0 else 'Constant + Trend'}")
    print(f"Lag order (k_ar_diff): {k_ar_diff}")
    print(f"{'='*60}")

    print(f"\n--- Trace Test ---")
    print(f"{'H0':<15} {'Trace Stat':<15} {'5% CV':<15} {'Reject?':<10}")
    for i in range(len(variables)):
        trace_stat = result.lr1[i]
        cv_5 = result.cvt[i, 1]  # 5% critical value
        reject = "Yes" if trace_stat > cv_5 else "No"
        print(f"r <= {i:<10} {trace_stat:<15.4f} {cv_5:<15.4f} {reject}")

    print(f"\n--- Max Eigenvalue Test ---")
    print(f"{'H0':<15} {'Max Eig Stat':<15} {'5% CV':<15} {'Reject?':<10}")
    for i in range(len(variables)):
        max_stat = result.lr2[i]
        cv_5 = result.cvm[i, 1]  # 5% critical value
        reject = "Yes" if max_stat > cv_5 else "No"
        print(f"r = {i:<11} {max_stat:<15.4f} {cv_5:<15.4f} {reject}")

    # Determine cointegration rank
    rank = 0
    for i in range(len(variables)):
        if result.lr1[i] > result.cvt[i, 1]:
            rank = i + 1
        else:
            break

    print(f"\nCointegration Rank (Trace): {rank}")
    return result, rank


if __name__ == "__main__":
    df = load_data()
    variables = ["mpr", "inflation", "exchange_rate", "m2"]

    print("=" * 60)
    print("JOHANSEN TEST — ROBUSTNESS CHECK ACROSS LAG ORDERS")
    print("=" * 60)

    for lag in [1, 2]:
        result, rank = johansen_test(df, variables, det_order=0, k_ar_diff=lag)
        print(f"\nWith lag={lag}: Cointegration rank = {rank}")

    print("\n" + "=" * 60)
    print("If the rank is the same for both lags, the result is ROBUST.")
    print("If the rank differs, investigate further with information criteria.")
    print("=" * 60)
```

**Run it:**

```bash
python -m econometric_models.cointegration
```

**What you should see:**

Two complete Johansen test tables -- one for `k_ar_diff=1` and one for `k_ar_diff=2`. At the bottom, the script prints the rank for each lag and a note about robustness.

```
============================================================
JOHANSEN TEST — ROBUSTNESS CHECK ACROSS LAG ORDERS
============================================================

============================================================
JOHANSEN COINTEGRATION TEST
Variables: mpr, inflation, exchange_rate, m2
Deterministic term: Constant only
Lag order (k_ar_diff): 1
============================================================

--- Trace Test ---
...
Cointegration Rank (Trace): X

With lag=1: Cointegration rank = X

============================================================
JOHANSEN COINTEGRATION TEST
Variables: mpr, inflation, exchange_rate, m2
Deterministic term: Constant only
Lag order (k_ar_diff): 2
============================================================

--- Trace Test ---
...
Cointegration Rank (Trace): X

With lag=2: Cointegration rank = X

============================================================
If the rank is the same for both lags, the result is ROBUST.
If the rank differs, investigate further with information criteria.
============================================================
```

**Why test with different lags?** The `k_ar_diff` parameter controls how many lagged differences are included in the underlying VECM. Different lag orders can absorb different amounts of short-run dynamics, which can change the test statistics and therefore the rank. If your rank is 1 with lag=1 and also 1 with lag=2, you can confidently report "the Johansen test finds one cointegrating relationship, and this result is robust to the lag specification." If the rank changes (say, 1 with lag=1 but 2 with lag=2), the result is fragile and you need to use information criteria (AIC, BIC) to determine the optimal lag before drawing conclusions.

---

### Build Step 3: Final Version with Engle-Granger + Johansen + Summary + Save (FINAL)

This is the complete, final version of the file. It runs all 6 Engle-Granger pairwise tests from Day 8, runs the Johansen test with two different lag orders, prints a comprehensive summary comparing both methods, and saves the results to CSV files.

Delete everything in `econometric_models/cointegration.py` and replace it with this:

```python
import os
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def engle_granger_test(df, var1, var2):
    """
    Run the Engle-Granger two-step cointegration test on a pair of variables.

    Returns a dictionary with the test statistic, p-value, and whether
    the null hypothesis of no cointegration is rejected at 5%.
    """
    series1 = df[var1].dropna()
    series2 = df[var2].dropna()
    common_index = series1.index.intersection(series2.index)
    series1 = series1.loc[common_index]
    series2 = series2.loc[common_index]

    stat, pvalue, crit_values = coint(series1, series2)

    result = {
        "var1": var1,
        "var2": var2,
        "test_statistic": stat,
        "p_value": pvalue,
        "crit_1pct": crit_values[0],
        "crit_5pct": crit_values[1],
        "crit_10pct": crit_values[2],
        "cointegrated_5pct": pvalue < 0.05,
    }

    print(f"\nEngle-Granger: {var1} & {var2}")
    print(f"  Test statistic: {stat:.4f}")
    print(f"  P-value:        {pvalue:.4f}")
    print(f"  Critical values: 1%={crit_values[0]:.4f}, 5%={crit_values[1]:.4f}, 10%={crit_values[2]:.4f}")
    print(f"  Cointegrated at 5%: {'Yes' if pvalue < 0.05 else 'No'}")

    return result


def johansen_test(df, variables, det_order=0, k_ar_diff=1):
    """
    Run the Johansen cointegration test on multiple variables.

    Parameters:
    - df: DataFrame with the variables
    - variables: list of column names to test
    - det_order: deterministic term (0 = constant only, 1 = constant + trend)
    - k_ar_diff: number of lagged differences in the VECM (like lag order minus 1)
    """
    data = df[variables].dropna()
    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

    print(f"\n{'='*60}")
    print(f"JOHANSEN COINTEGRATION TEST")
    print(f"Variables: {', '.join(variables)}")
    print(f"Deterministic term: {'Constant only' if det_order == 0 else 'Constant + Trend'}")
    print(f"Lag order (k_ar_diff): {k_ar_diff}")
    print(f"{'='*60}")

    print(f"\n--- Trace Test ---")
    print(f"{'H0':<15} {'Trace Stat':<15} {'5% CV':<15} {'Reject?':<10}")
    for i in range(len(variables)):
        trace_stat = result.lr1[i]
        cv_5 = result.cvt[i, 1]  # 5% critical value
        reject = "Yes" if trace_stat > cv_5 else "No"
        print(f"r <= {i:<10} {trace_stat:<15.4f} {cv_5:<15.4f} {reject}")

    print(f"\n--- Max Eigenvalue Test ---")
    print(f"{'H0':<15} {'Max Eig Stat':<15} {'5% CV':<15} {'Reject?':<10}")
    for i in range(len(variables)):
        max_stat = result.lr2[i]
        cv_5 = result.cvm[i, 1]  # 5% critical value
        reject = "Yes" if max_stat > cv_5 else "No"
        print(f"r = {i:<11} {max_stat:<15.4f} {cv_5:<15.4f} {reject}")

    # Determine cointegration rank
    rank = 0
    for i in range(len(variables)):
        if result.lr1[i] > result.cvt[i, 1]:
            rank = i + 1
        else:
            break

    print(f"\nCointegration Rank (Trace): {rank}")
    return result, rank


if __name__ == "__main__":
    df = load_data()
    variables = ["mpr", "inflation", "exchange_rate", "m2"]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Part 1: Engle-Granger Pairwise Tests ----
    print("=" * 60)
    print("PART 1: ENGLE-GRANGER PAIRWISE COINTEGRATION TESTS")
    print("=" * 60)

    eg_results = []
    pairs = list(combinations(variables, 2))
    for var1, var2 in pairs:
        res = engle_granger_test(df, var1, var2)
        eg_results.append(res)

    eg_df = pd.DataFrame(eg_results)
    eg_df.to_csv(os.path.join(RESULTS_DIR, "engle_granger_cointegration.csv"), index=False)
    print(f"\nSaved Engle-Granger results to results/engle_granger_cointegration.csv")

    # ---- Part 2: Johansen Test with Multiple Lags ----
    print("\n" + "=" * 60)
    print("PART 2: JOHANSEN COINTEGRATION TESTS")
    print("=" * 60)

    johansen_results = []
    for lag in [1, 2]:
        result, rank = johansen_test(df, variables, det_order=0, k_ar_diff=lag)
        johansen_results.append({
            "k_ar_diff": lag,
            "det_order": 0,
            "cointegration_rank": rank,
        })
        print(f"\nWith k_ar_diff={lag}: Cointegration rank = {rank}")

    johansen_df = pd.DataFrame(johansen_results)
    johansen_df.to_csv(os.path.join(RESULTS_DIR, "johansen_cointegration.csv"), index=False)
    print(f"\nSaved Johansen results to results/johansen_cointegration.csv")

    # ---- Part 3: Comprehensive Summary ----
    print("\n" + "=" * 60)
    print("COMPREHENSIVE COINTEGRATION SUMMARY")
    print("=" * 60)

    print("\n--- Engle-Granger Pairwise Results ---")
    cointegrated_pairs = [r for r in eg_results if r["cointegrated_5pct"]]
    not_cointegrated_pairs = [r for r in eg_results if not r["cointegrated_5pct"]]

    print(f"Total pairs tested: {len(eg_results)}")
    print(f"Cointegrated at 5%: {len(cointegrated_pairs)}")
    if cointegrated_pairs:
        print("  Pairs with cointegration:")
        for r in cointegrated_pairs:
            print(f"    {r['var1']} & {r['var2']} (p={r['p_value']:.4f})")
    if not_cointegrated_pairs:
        print("  Pairs without cointegration:")
        for r in not_cointegrated_pairs:
            print(f"    {r['var1']} & {r['var2']} (p={r['p_value']:.4f})")

    print(f"\n--- Johansen Multivariate Results ---")
    for jr in johansen_results:
        print(f"  k_ar_diff={jr['k_ar_diff']}: rank = {jr['cointegration_rank']}")

    ranks = [jr["cointegration_rank"] for jr in johansen_results]
    if len(set(ranks)) == 1:
        print(f"\nRobustness: PASS — Rank is consistently {ranks[0]} across lag orders.")
    else:
        print(f"\nRobustness: CAUTION — Rank varies across lag orders ({ranks}).")
        print("Consider using information criteria to select the optimal lag.")

    print(f"\n--- Model Implication ---")
    final_rank = ranks[0]
    if final_rank == 0:
        print("Rank = 0: No cointegration found.")
        print("Recommendation: Use a VAR model on differenced (stationary) data.")
    elif final_rank == 1:
        print("Rank = 1: One cointegrating relationship found.")
        print("Recommendation: Use a VECM or ARDL bounds test to capture")
        print("both the long-run equilibrium and short-run dynamics.")
    elif final_rank == 2:
        print("Rank = 2: Two cointegrating relationships found.")
        print("Recommendation: Use a VECM with two error correction terms,")
        print("or ARDL bounds test for individual equation estimation.")
    else:
        print(f"Rank = {final_rank}: {final_rank} cointegrating relationships found.")
        print("Recommendation: Verify with alternative lag specifications.")
        print("A high rank with few variables may indicate model misspecification.")

    print("\n" + "=" * 60)
    print("Cointegration analysis complete.")
    print("=" * 60)
```

**Run it:**

```bash
python -m econometric_models.cointegration
```

**What you should see:**

A long output with three parts:

1. **Part 1** -- All 6 Engle-Granger pairwise test results (same as Day 8).
2. **Part 2** -- Two Johansen test tables (lag=1 and lag=2), each showing the Trace test, Max Eigenvalue test, and the determined rank.
3. **Part 3** -- A comprehensive summary that lists which pairs are cointegrated (Engle-Granger), what the Johansen rank is at each lag, whether the rank is robust, and what model you should use based on the results.

You should also see two new CSV files saved in `results/`:

```
Saved Engle-Granger results to results/engle_granger_cointegration.csv
Saved Johansen results to results/johansen_cointegration.csv
```

**What changed from the previous step:**

- The main block now runs BOTH the Engle-Granger tests (from Day 8) and the Johansen tests.
- The Johansen test is run with `k_ar_diff=1` and `k_ar_diff=2` to check robustness.
- A comprehensive summary section compares the two methods and tells you what model to use.
- Both result sets are saved to CSV files so you can include them in your thesis appendix.

This is the final version of `econometric_models/cointegration.py`. It contains everything from Day 8 (Engle-Granger) plus everything from Day 9 (Johansen).

---

## Interpreting Your Results -- Nigeria Specific

### What the Cointegration Rank Means for the Nigerian Economy

The cointegration rank tells you how many independent long-run equilibrium relationships bind your four Nigerian macroeconomic variables together. This is not just a statistical result -- it has real economic meaning.

**If the rank is 1 (most likely):**

There is one long-run equilibrium relationship tying MPR, inflation, exchange rate, and M2 together. In economic terms, this likely captures the **monetary policy transmission channel**: the CBN sets the MPR, which influences commercial bank lending rates, which affects the money supply (M2), which interacts with the exchange rate (through capital flows and forex demand), which ultimately feeds into consumer prices (inflation).

All four variables are bound together by this one equilibrium. When the system is disturbed -- say, the CBN raises the MPR sharply -- the variables will deviate from their long-run relationship temporarily, but the error correction mechanism will pull them back toward equilibrium over time. The speed of this adjustment is what the VECM captures.

For Nigeria specifically, the single equilibrium interpretation makes sense because:
- The CBN explicitly targets inflation through the MPR (monetary policy channel).
- Exchange rate depreciation passes through to consumer prices quickly because Nigeria is heavily import-dependent (exchange rate channel).
- Money supply growth finances government deficits, which adds demand pressure and depreciates the Naira (fiscal-monetary channel).
- All four channels ultimately converge on inflation.

**If the rank is 2:**

There are two independent long-run equilibria. The second relationship likely captures a distinct **exchange rate-money supply equilibrium**: the CBN's foreign exchange interventions (selling dollars from reserves to defend the Naira) directly affect the domestic money supply. When the CBN sells dollars, it absorbs Naira from the economy, reducing M2. When it stops defending the rate (as in 2016 and 2023), M2 expands because the CBN is no longer mopping up liquidity.

This second equilibrium is independent of the first (the inflation-monetary policy channel), meaning two separate long-run forces are at work in the Nigerian economy. A VECM with two error correction terms would capture both.

### How This Determines Your Model Choice

The cointegration rank is the single most important input to your model selection decision:

| Rank | Model | Why |
|------|-------|-----|
| 0 | VAR in first differences | No long-run relationship exists. Model the short-run dynamics of the differenced (stationary) data. |
| 1 | VECM with 1 error correction term, or ARDL bounds test | One long-run equilibrium. The VECM/ARDL captures both the equilibrium and the short-run adjustment toward it. |
| 2 | VECM with 2 error correction terms | Two equilibria. The VECM includes two error correction terms, each pulling the system toward its respective equilibrium. |
| 3 | Unusual -- investigate further | With only 4 variables, 3 cointegrating relationships means the system is almost fully determined in the long run. Check your data and lag specification. |

### The Cointegrating Vector

The Johansen test also produces the cointegrating vector -- the coefficients of the long-run equilibrium equation. You can access it from `result.evec` (the eigenvectors). The first column of `result.evec` corresponds to the first cointegrating relationship. If the rank is 1, this vector tells you the long-run relationship:

```
beta1 * mpr + beta2 * inflation + beta3 * exchange_rate + beta4 * m2 = 0
```

You can normalize this by dividing through by the coefficient on inflation (or any variable of interest) to get an equation of the form:

```
inflation = c1 * mpr + c2 * exchange_rate + c3 * m2
```

This is the long-run equilibrium equation -- it tells you the long-run effect of each variable on inflation. You will use this in later weeks when building the VECM.

### What to Tell Your Examiner

Here is a template answer you can adapt based on your actual results:

> "The Johansen cointegration test was applied to the four-variable system comprising the Monetary Policy Rate, headline inflation, the NGN/USD exchange rate, and broad money supply (M2). Using a constant-only specification (det_order=0) and testing with k_ar_diff values of 1 and 2, both the Trace test and the Maximum Eigenvalue test indicate a cointegration rank of [your rank]. This result is robust across the two lag specifications tested.
>
> The finding of [your rank] cointegrating relationship(s) implies that the four variables share [your rank] independent long-run equilibrium relationship(s). This is consistent with economic theory, which predicts that monetary policy, money supply, the exchange rate, and inflation are bound together through the monetary policy transmission mechanism [and, if rank=2, the exchange rate-money supply channel].
>
> Based on the cointegration rank, a Vector Error Correction Model (VECM) with [your rank] error correction term(s) is the appropriate modelling framework, as it captures both the long-run equilibrium relationship(s) and the short-run adjustment dynamics."

---

## Step 5: Commit

```bash
git add econometric_models/cointegration.py
git commit -m "Day 9: Johansen cointegration test with trace and max-eigenvalue"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `LinAlgError: Singular matrix` or `matrix is singular` | This usually means you have too many lags for the amount of data. With 300 observations and 4 variables, `k_ar_diff` up to about 12 should work. If you set `k_ar_diff=20` or higher, the model runs out of degrees of freedom and the matrix becomes singular. Reduce the lag order. |
| `ValueError: det_order must be -1, 0, or 1` | You passed an invalid value for `det_order`. The only valid options are -1 (no deterministic terms), 0 (constant only), or 1 (constant plus trend). We use 0. |
| `KeyError: 'mpr'` or similar column error | Your DataFrame does not have the expected column names. Make sure `cleaned_data.csv` has columns named exactly `mpr`, `inflation`, `exchange_rate`, `m2`. Check with `print(df.columns)`. |
| Confusing `k_ar_diff` with the VAR lag order | `k_ar_diff` is the number of lagged *differences* in the VECM, which equals the VAR lag order minus 1. So `k_ar_diff=1` corresponds to VAR(2), `k_ar_diff=2` corresponds to VAR(3), and so on. If your VAR analysis suggests an optimal lag of 2, set `k_ar_diff=1`. |
| `ImportError: cannot import name 'coint_johansen'` | Make sure you are importing from the correct module: `from statsmodels.tsa.vector_ar.vecm import coint_johansen`. Older versions of statsmodels had this in a different location. If you installed statsmodels correctly on Day 6, this should work. |
| Trace and Max Eigenvalue tests give different ranks | This happens. The two tests frame the hypothesis differently, so they can disagree, especially when a test statistic is close to the critical value. Report both results. When they disagree, the Trace test is generally preferred because it has better small-sample properties. |

---

## Check Your Understanding

These are the kinds of questions an examiner will ask about the Johansen test. Practice answering them out loud.

### 1. "What is the difference between the Trace test and the Max Eigenvalue test?"

**Answer:** "Both tests are part of the Johansen procedure for determining the cointegration rank, but they differ in how they formulate the hypotheses.

The Trace test tests H0: the cointegration rank is less than or equal to r, against H1: the rank is greater than r. This means the alternative hypothesis allows for the possibility that there could be many more cointegrating relationships beyond r -- it just says 'more than r.'

The Max Eigenvalue test tests H0: the rank is exactly r, against H1: the rank is exactly r+1. This is a sharper test because the alternative is more specific -- it says there is exactly one additional cointegrating relationship.

In practice, the Trace test tends to be more commonly reported and is generally preferred when the two tests disagree. This is because the Trace test has been shown to have better finite-sample (small-sample) properties in simulation studies. Most applied econometrics textbooks recommend reporting both but relying on the Trace test when they conflict."

### 2. "What does the cointegration rank tell you?"

**Answer:** "The cointegration rank tells you the number of independent long-run equilibrium relationships that exist among a set of non-stationary variables. With n I(1) variables, the rank can range from 0 to n-1.

A rank of 0 means there are no long-run equilibrium relationships -- the variables wander independently and do not share a common stochastic trend. In this case, you should model the data using a VAR in first differences.

A rank of 1 means there is one long-run equilibrium that ties the variables together. Any deviation from this equilibrium is temporary -- the system will correct itself back toward the equilibrium over time. This is modelled with a VECM containing one error correction term.

A rank of 2 or higher means there are multiple independent equilibria. Each one represents a separate long-run constraint on the system. A VECM with multiple error correction terms is used.

For our four-variable Nigerian system, a rank of 1 would mean that MPR, inflation, exchange rate, and M2 share one common long-run equilibrium, likely representing the monetary policy transmission mechanism."

### 3. "Why test with different lag orders?"

**Answer:** "The Johansen test results can be sensitive to the lag specification -- specifically, the `k_ar_diff` parameter, which controls how many lagged differences are included in the underlying Vector Error Correction Model. Different lag orders capture different amounts of short-run dynamics, which can change the test statistics and potentially the cointegration rank.

If the cointegration rank is the same when I test with k_ar_diff=1 and k_ar_diff=2, I can be confident that the result is robust -- it does not depend on a particular lag choice. This is what we want.

If the rank changes when I change the lag -- for example, rank=1 with one lag but rank=2 with two lags -- the result is fragile. In this case, I would use information criteria such as the AIC or BIC to determine the optimal lag order for the underlying VAR, then set k_ar_diff to that optimal lag minus 1. The rank obtained at the optimal lag would be the one I report.

Testing robustness to lag specification is standard practice in applied econometrics and demonstrates methodological rigour to an examiner."

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Johansen test function | `econometric_models/cointegration.py` | Tests all 4 variables simultaneously for multivariate cointegration |
| Robustness check | `econometric_models/cointegration.py` | Runs Johansen test with multiple lag orders to verify results are stable |
| Comprehensive summary | `econometric_models/cointegration.py` | Compares Engle-Granger and Johansen results and recommends a model |
| Engle-Granger CSV | `results/engle_granger_cointegration.csv` | Pairwise cointegration test results for thesis appendix |
| Johansen CSV | `results/johansen_cointegration.csv` | Multivariate cointegration rank results for thesis appendix |

**No new packages installed today.** The `coint_johansen` function is part of statsmodels, which was installed on Day 6.

---

**Tomorrow (Day 10):** You will determine the optimal lag order for your VAR/VECM model using information criteria (AIC, BIC, HQIC). The lag order is a critical input -- too few lags and the model misses important dynamics; too many and you waste degrees of freedom and risk overfitting. Day 10 ties together the stationarity results (Days 6-7), the cointegration results (Days 8-9), and prepares you for model estimation in Week 3.
