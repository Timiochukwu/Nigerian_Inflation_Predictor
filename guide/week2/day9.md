# Week 2, Day 9 -- Johansen Cointegration Test

## What You'll Learn Today

- Why the Engle-Granger test from Day 8 is not enough (it only handles two variables at a time)
- How the Johansen test examines all 4 variables simultaneously and can find multiple long-run relationships
- The difference between the Trace test and the Max-Eigenvalue test
- What the cointegrating rank means: 0, 1, 2, or 3 for a system of 4 variables
- How to check robustness by running the test with different lag orders

## Why This Matters

On Day 8 you tested every pair of variables for cointegration using the Engle-Granger method. That gave you 6 pairwise results. But the Engle-Granger test only looks at TWO variables at a time. Your Nigerian inflation system has FOUR variables -- mpr, infl, exo, and tbr -- all interacting simultaneously. There could be a long-run equilibrium that involves all four variables together, and the pairwise approach cannot see it.

The Johansen test fixes this. It examines all four variables at once and tells you exactly how many independent cointegrating relationships exist. This number -- the **cointegration rank** -- is one of the most important results in your entire thesis, because it determines which class of econometric model you should use going forward.

---

## No New Packages

The `coint_johansen` function is already included in statsmodels, which you installed on Day 6. It lives in `statsmodels.tsa.vector_ar.vecm`. No new packages are needed today.

---

## Theory: The Johansen Test

### Why Engle-Granger Is Not Enough

The Engle-Granger test works by regressing one variable on another and testing whether the residuals are stationary. This is fine for two variables. But with four I(1) variables, there can be **up to 3 cointegrating relationships** (the maximum is always n-1, where n is the number of variables). Engle-Granger tests one pair at a time and can only ever detect one relationship per test. Worse, the result can change depending on which variable you put on the left side of the regression.

In summary, Engle-Granger has three problems in a multivariate system:

1. It cannot tell you how many cointegrating relationships exist -- it only tests for one at a time.
2. It is sensitive to normalization -- swapping the dependent and independent variable can change the result.
3. It ignores the multivariate structure -- when four variables interact, you need a method that sees all four at once.

### What the Johansen Test Does

Soren Johansen (1988, 1991) developed a method that solves all three problems. His test examines all variables simultaneously in a multivariate framework and determines:

1. **Whether cointegration exists at all** among the set of variables.
2. **How many cointegrating relationships exist** -- the cointegration rank, written as *r*.
3. **The cointegrating vectors** -- the actual coefficients of each long-run equilibrium equation.

The test is based on a Vector Error Correction Model (VECM). The mathematics involve eigenvalues of a matrix, which is why the output reports "eigenvalue statistics." You do not need to understand the linear algebra to use and interpret the test correctly.

### Two Test Statistics: Trace and Max-Eigenvalue

The Johansen procedure produces two test statistics. Both determine the cointegration rank, but they frame the hypothesis differently.

**1. The Trace Test**

The Trace test asks: "Is the cointegration rank greater than r?"

- H0: The rank is less than or equal to r (at most r relationships).
- H1: The rank is greater than r (more than r relationships).

The test starts at r=0 and works upward. Each time the trace statistic exceeds the 5% critical value, you reject H0 and move to the next r. The first time you fail to reject, you stop. The number of rejections is your rank.

**2. The Max-Eigenvalue Test**

The Max-Eigenvalue test asks: "Is the rank exactly r, or is it r+1?"

- H0: The rank is exactly r.
- H1: The rank is exactly r+1.

This is a sharper hypothesis -- it tests for exactly one additional relationship at each step. However, it tends to have less statistical power in small samples.

**Which to trust?** Report both. If they disagree, prefer the Trace test. It has better small-sample properties and is recommended by most textbooks, including Lutkepohl's *New Introduction to Multiple Time Series Analysis*.

### What the Rank Means

With 4 variables (mpr, infl, exo, tbr), the cointegration rank can be 0, 1, 2, or 3:

- **r = 0 (no cointegration):** The four variables do not share any long-run equilibrium. They wander independently. You would model the data using a VAR in first differences. This is the least useful result for your thesis because you cannot say anything about long-run relationships.

- **r = 1 (one cointegrating relationship):** There is one long-run equilibrium tying the four variables together. This is the most common finding for Nigerian macroeconomic data. It likely captures the monetary policy transmission mechanism: the CBN sets the mpr, which influences treasury bill rates (tbr), which interacts with the exchange rate (exo), which feeds into consumer prices (infl). A VECM with one error correction term or the ARDL bounds test is appropriate.

- **r = 2 (two independent long-run relationships):** Two separate equilibria bind the system. The second might capture a distinct interest rate channel (mpr-tbr link) separate from the inflation-exchange rate channel. A VECM with two error correction terms is appropriate.

- **r = 3 (three relationships):** This is rare with only 4 variables. It implies the system is almost fully determined in the long run -- near-stationarity. If you get r=3, double-check your data and lag specification before reporting it.

### Deterministic Terms

The `det_order` parameter controls what deterministic components are included:

- `det_order=0`: Constant in the cointegrating equation only (no trend). This is the standard specification for macroeconomic data. It allows the long-run equilibrium to have a non-zero mean.
- `det_order=1`: Constant plus a linear trend.
- `det_order=-1`: No deterministic terms at all. Rarely used.

We use `det_order=0` throughout. If your examiner asks why: "A constant-only specification is standard for macroeconomic data. It allows for a non-zero mean in the cointegrating relationship without imposing a deterministic trend."

---

## Building `econometric_models/cointegration.py` -- 3 Steps

On Day 8 you built this file with the Engle-Granger pairwise tests. Today you expand it to include the Johansen test. Each step shows the **complete file** from the first line to the last. No snippets.

---

### Build Step 1: All Day 8 Code Plus Johansen Import and Stub

This step carries forward everything from Day 8 (load_data, engle_granger_test) and adds the new import for the Johansen test. The main block just prints a header and loads the data so you can confirm the new import works.

Delete everything in `econometric_models/cointegration.py` and replace it with this:

```python
"""Cointegration testing: Engle-Granger (Day 8) and Johansen (Day 9)."""
import os
from itertools import combinations

import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    """Load the cleaned dataset from data/processed/cleaned_data.csv."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def engle_granger_test(series1, series2, name1, name2):
    """
    Run the Engle-Granger cointegration test between two series.

    Null hypothesis: the two series are NOT cointegrated.
    If p-value < 0.05: reject null -> they ARE cointegrated.
    """
    score, p_value, critical_values = coint(series1, series2)

    print(f"\n{'='*55}")
    print(f"Engle-Granger Test: {name1} & {name2}")
    print(f"{'='*55}")
    print(f"Test Statistic:  {score:.4f}")
    print(f"P-Value:         {p_value:.4f}")
    print(f"Critical Values: 1%={critical_values[0]:.4f}, "
          f"5%={critical_values[1]:.4f}, "
          f"10%={critical_values[2]:.4f}")

    if p_value < 0.05:
        print(f"Conclusion: COINTEGRATED (reject H0 at 5%)")
    else:
        print(f"Conclusion: NOT cointegrated (fail to reject H0)")

    return {
        "pair": f"{name1} & {name2}",
        "test_stat": score,
        "p_value": p_value,
        "cointegrated": p_value < 0.05,
    }


if __name__ == "__main__":
    df = load_data()
    print("=" * 60)
    print("DAY 9 -- JOHANSEN COINTEGRATION TEST")
    print("=" * 60)
    print(f"Loaded {len(df)} rows")
    print(f"Variables: {list(df.columns)}")
    print(f"Johansen import OK: coint_johansen ready")
```

**Run it:**

```bash
python -m econometric_models.cointegration
```

**What you should see:**

```
============================================================
DAY 9 -- JOHANSEN COINTEGRATION TEST
============================================================
Loaded 300 rows
Variables: ['mpr', 'infl', 'exo', 'tbr']
Johansen import OK: coint_johansen ready
```

(Your exact row count may differ depending on your data.)

**What just happened:**

- `from statsmodels.tsa.vector_ar.vecm import coint_johansen` -- This imports the Johansen test function from the VECM module. It lives in a different part of statsmodels than the `coint` function you used on Day 8. If this line runs without error, you are ready to use it.
- All the Day 8 code (`load_data`, `engle_granger_test`) is preserved unchanged. You will still be able to run the pairwise tests whenever you need them.
- The main block confirms the data loads and the new import works. Nothing else runs yet.

---

### Build Step 2: Add the johansen_test Function

Delete everything in `econometric_models/cointegration.py` and replace it with this:

```python
"""Cointegration testing: Engle-Granger (Day 8) and Johansen (Day 9)."""
import os
from itertools import combinations

import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    """Load the cleaned dataset from data/processed/cleaned_data.csv."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def engle_granger_test(series1, series2, name1, name2):
    """
    Run the Engle-Granger cointegration test between two series.

    Null hypothesis: the two series are NOT cointegrated.
    If p-value < 0.05: reject null -> they ARE cointegrated.
    """
    score, p_value, critical_values = coint(series1, series2)

    print(f"\n{'='*55}")
    print(f"Engle-Granger Test: {name1} & {name2}")
    print(f"{'='*55}")
    print(f"Test Statistic:  {score:.4f}")
    print(f"P-Value:         {p_value:.4f}")
    print(f"Critical Values: 1%={critical_values[0]:.4f}, "
          f"5%={critical_values[1]:.4f}, "
          f"10%={critical_values[2]:.4f}")

    if p_value < 0.05:
        print(f"Conclusion: COINTEGRATED (reject H0 at 5%)")
    else:
        print(f"Conclusion: NOT cointegrated (fail to reject H0)")

    return {
        "pair": f"{name1} & {name2}",
        "test_stat": score,
        "p_value": p_value,
        "cointegrated": p_value < 0.05,
    }


def johansen_test(df, det_order=0, k_ar_diff=1):
    """
    Run the Johansen cointegration test on all columns of df.

    Parameters:
        df         -- DataFrame whose columns are the variables to test.
        det_order  -- Deterministic term: 0 = constant only, 1 = constant + trend.
        k_ar_diff  -- Number of lagged differences in the VECM (VAR lag minus 1).

    Returns:
        dict with keys: trace_rank, max_eig_rank, details (list of per-row dicts).
    """
    data = df.dropna()
    n_vars = data.shape[1]
    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

    # ---- Trace test ----
    print(f"\n{'='*65}")
    print(f"JOHANSEN TEST  |  det_order={det_order}, k_ar_diff={k_ar_diff}")
    print(f"Variables: {', '.join(data.columns)}")
    print(f"{'='*65}")

    print(f"\n--- Trace Test ---")
    print(f"{'H0':<12} {'Trace Stat':>12} {'5% CV':>12} {'Reject?':>10}")
    print("-" * 48)

    trace_rank = 0
    details = []
    for i in range(n_vars):
        trace_stat = result.lr1[i]
        cv_5 = result.cvt[i, 1]
        reject = trace_stat > cv_5
        tag = "Yes ***" if reject else "No"
        print(f"r <= {i:<7} {trace_stat:>12.4f} {cv_5:>12.4f} {tag:>10}")
        details.append({
            "hypothesis": f"r <= {i}",
            "trace_stat": round(trace_stat, 4),
            "trace_cv_5pct": round(cv_5, 4),
            "trace_reject": reject,
        })
        if reject:
            trace_rank = i + 1

    # ---- Max-eigenvalue test ----
    print(f"\n--- Max-Eigenvalue Test ---")
    print(f"{'H0':<12} {'Max-Eig Stat':>12} {'5% CV':>12} {'Reject?':>10}")
    print("-" * 48)

    max_eig_rank = 0
    for i in range(n_vars):
        max_stat = result.lr2[i]
        cv_5 = result.cvm[i, 1]
        reject = max_stat > cv_5
        tag = "Yes ***" if reject else "No"
        print(f"r = {i:<8} {max_stat:>12.4f} {cv_5:>12.4f} {tag:>10}")
        details[i]["max_eig_stat"] = round(max_stat, 4)
        details[i]["max_eig_cv_5pct"] = round(cv_5, 4)
        details[i]["max_eig_reject"] = reject
        if reject:
            max_eig_rank = i + 1

    print(f"\nRank from Trace test:          {trace_rank}")
    print(f"Rank from Max-Eigenvalue test: {max_eig_rank}")

    if trace_rank == max_eig_rank:
        print(f"Both tests agree: rank = {trace_rank}")
    else:
        print(f"Tests DISAGREE. Prefer Trace test: rank = {trace_rank}")

    return {
        "trace_rank": trace_rank,
        "max_eig_rank": max_eig_rank,
        "details": details,
    }


if __name__ == "__main__":
    df = load_data()
    variables = ["mpr", "infl", "exo", "tbr"]

    print("=" * 65)
    print("DAY 9 -- JOHANSEN COINTEGRATION TEST")
    print("=" * 65)

    result = johansen_test(df[variables], det_order=0, k_ar_diff=1)
    print(f"\nFinal rank (Trace): {result['trace_rank']}")
```

**Run it:**

```bash
python -m econometric_models.cointegration
```

**What you should see:**

```
=================================================================
DAY 9 -- JOHANSEN COINTEGRATION TEST
=================================================================

=================================================================
JOHANSEN TEST  |  det_order=0, k_ar_diff=1
Variables: mpr, infl, exo, tbr
=================================================================

--- Trace Test ---
H0            Trace Stat        5% CV    Reject?
------------------------------------------------
r <= 0          XX.XXXX       47.8561    Yes ***
r <= 1          XX.XXXX       29.7971         No
r <= 2          XX.XXXX       15.4947         No
r <= 3          XX.XXXX        3.8415         No

--- Max-Eigenvalue Test ---
H0           Max-Eig Stat        5% CV    Reject?
------------------------------------------------
r = 0           XX.XXXX       27.5843    Yes ***
r = 1           XX.XXXX       21.1316         No
r = 2           XX.XXXX       14.2646         No
r = 3           XX.XXXX        3.8415         No

Rank from Trace test:          1
Rank from Max-Eigenvalue test: 1
Both tests agree: rank = 1

Final rank (Trace): 1
```

(Your exact numbers will differ. The critical values are fixed -- they come from statistical tables built into statsmodels.)

**What just happened -- line by line:**

- `coint_johansen(data, det_order=0, k_ar_diff=1)` -- Runs the full Johansen procedure on all columns of the DataFrame at once. `det_order=0` means constant only. `k_ar_diff=1` means one lagged difference in the underlying VECM, which corresponds to a VAR(2).
- `result.lr1` -- Array of trace test statistics, one for each hypothesis (r<=0, r<=1, r<=2, r<=3).
- `result.cvt` -- Matrix of critical values for the trace test. `result.cvt[i, 1]` is the 5% critical value for hypothesis i. Column 0 is 10%, column 1 is 5%, column 2 is 1%.
- `result.lr2` -- Array of max-eigenvalue test statistics.
- `result.cvm` -- Matrix of critical values for the max-eigenvalue test.
- The rank determination counts consecutive rejections starting from r=0. Each time the test statistic exceeds the 5% critical value, we increment the rank. The rank equals the highest r+1 where rejection occurs sequentially.
- The function returns a dictionary with both ranks and the detailed per-hypothesis results, so downstream code can use them programmatically.

---

### Build Step 3: Robustness Check and Save Results (Final Version)

The Johansen test can be sensitive to the lag specification. A robust result gives the same rank across different reasonable lag orders. This final version runs the test with k_ar_diff = 1, 2, and 3, prints a comparative table, and saves everything to CSV.

Delete everything in `econometric_models/cointegration.py` and replace it with this:

```python
"""Cointegration testing: Engle-Granger (Day 8) and Johansen (Day 9)."""
import os
from itertools import combinations

import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    """Load the cleaned dataset from data/processed/cleaned_data.csv."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def engle_granger_test(series1, series2, name1, name2):
    """
    Run the Engle-Granger cointegration test between two series.

    Null hypothesis: the two series are NOT cointegrated.
    If p-value < 0.05: reject null -> they ARE cointegrated.
    """
    score, p_value, critical_values = coint(series1, series2)

    print(f"\n{'='*55}")
    print(f"Engle-Granger Test: {name1} & {name2}")
    print(f"{'='*55}")
    print(f"Test Statistic:  {score:.4f}")
    print(f"P-Value:         {p_value:.4f}")
    print(f"Critical Values: 1%={critical_values[0]:.4f}, "
          f"5%={critical_values[1]:.4f}, "
          f"10%={critical_values[2]:.4f}")

    if p_value < 0.05:
        print(f"Conclusion: COINTEGRATED (reject H0 at 5%)")
    else:
        print(f"Conclusion: NOT cointegrated (fail to reject H0)")

    return {
        "pair": f"{name1} & {name2}",
        "test_stat": score,
        "p_value": p_value,
        "cointegrated": p_value < 0.05,
    }


def johansen_test(df, det_order=0, k_ar_diff=1):
    """
    Run the Johansen cointegration test on all columns of df.

    Parameters:
        df         -- DataFrame whose columns are the variables to test.
        det_order  -- Deterministic term: 0 = constant only, 1 = constant + trend.
        k_ar_diff  -- Number of lagged differences in the VECM (VAR lag minus 1).

    Returns:
        dict with keys: trace_rank, max_eig_rank, details (list of per-row dicts).
    """
    data = df.dropna()
    n_vars = data.shape[1]
    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

    # ---- Trace test ----
    print(f"\n{'='*65}")
    print(f"JOHANSEN TEST  |  det_order={det_order}, k_ar_diff={k_ar_diff}")
    print(f"Variables: {', '.join(data.columns)}")
    print(f"{'='*65}")

    print(f"\n--- Trace Test ---")
    print(f"{'H0':<12} {'Trace Stat':>12} {'5% CV':>12} {'Reject?':>10}")
    print("-" * 48)

    trace_rank = 0
    details = []
    for i in range(n_vars):
        trace_stat = result.lr1[i]
        cv_5 = result.cvt[i, 1]
        reject = trace_stat > cv_5
        tag = "Yes ***" if reject else "No"
        print(f"r <= {i:<7} {trace_stat:>12.4f} {cv_5:>12.4f} {tag:>10}")
        details.append({
            "hypothesis": f"r <= {i}",
            "trace_stat": round(trace_stat, 4),
            "trace_cv_5pct": round(cv_5, 4),
            "trace_reject": reject,
        })
        if reject:
            trace_rank = i + 1

    # ---- Max-eigenvalue test ----
    print(f"\n--- Max-Eigenvalue Test ---")
    print(f"{'H0':<12} {'Max-Eig Stat':>12} {'5% CV':>12} {'Reject?':>10}")
    print("-" * 48)

    max_eig_rank = 0
    for i in range(n_vars):
        max_stat = result.lr2[i]
        cv_5 = result.cvm[i, 1]
        reject = max_stat > cv_5
        tag = "Yes ***" if reject else "No"
        print(f"r = {i:<8} {max_stat:>12.4f} {cv_5:>12.4f} {tag:>10}")
        details[i]["max_eig_stat"] = round(max_stat, 4)
        details[i]["max_eig_cv_5pct"] = round(cv_5, 4)
        details[i]["max_eig_reject"] = reject
        if reject:
            max_eig_rank = i + 1

    print(f"\nRank from Trace test:          {trace_rank}")
    print(f"Rank from Max-Eigenvalue test: {max_eig_rank}")

    if trace_rank == max_eig_rank:
        print(f"Both tests agree: rank = {trace_rank}")
    else:
        print(f"Tests DISAGREE. Prefer Trace test: rank = {trace_rank}")

    return {
        "trace_rank": trace_rank,
        "max_eig_rank": max_eig_rank,
        "details": details,
    }


if __name__ == "__main__":
    df = load_data()
    variables = ["mpr", "infl", "exo", "tbr"]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("DAY 9 -- JOHANSEN COINTEGRATION TEST (ROBUSTNESS CHECK)")
    print("=" * 65)

    # --- Run Johansen with k_ar_diff = 1, 2, 3 ---
    robustness_rows = []
    for lag in [1, 2, 3]:
        res = johansen_test(df[variables], det_order=0, k_ar_diff=lag)
        robustness_rows.append({
            "k_ar_diff": lag,
            "det_order": 0,
            "trace_rank": res["trace_rank"],
            "max_eig_rank": res["max_eig_rank"],
        })

    # --- Comparative table ---
    print(f"\n\n{'='*65}")
    print("ROBUSTNESS SUMMARY")
    print(f"{'='*65}")
    print(f"{'k_ar_diff':>10} {'Trace Rank':>12} {'Max-Eig Rank':>14}")
    print("-" * 38)
    for row in robustness_rows:
        print(f"{row['k_ar_diff']:>10} {row['trace_rank']:>12} "
              f"{row['max_eig_rank']:>14}")

    trace_ranks = [r["trace_rank"] for r in robustness_rows]
    if len(set(trace_ranks)) == 1:
        print(f"\nTrace rank is {trace_ranks[0]} at all lag orders -- ROBUST.")
    else:
        print(f"\nTrace rank varies across lags: {trace_ranks} -- SENSITIVE.")
        print("Use information criteria (Day 10) to pick the optimal lag.")

    # --- Save to CSV ---
    rob_df = pd.DataFrame(robustness_rows)
    outpath = os.path.join(RESULTS_DIR, "cointegration_johansen.csv")
    rob_df.to_csv(outpath, index=False)
    print(f"\nSaved to {outpath}")

    print("\n" + "=" * 65)
    print("Johansen cointegration analysis complete.")
    print("=" * 65)
```

**Run it:**

```bash
python -m econometric_models.cointegration
```

**What you should see:**

Three full Johansen test tables (one for each lag), followed by a robustness summary:

```
=================================================================
DAY 9 -- JOHANSEN COINTEGRATION TEST (ROBUSTNESS CHECK)
=================================================================

=================================================================
JOHANSEN TEST  |  det_order=0, k_ar_diff=1
Variables: mpr, infl, exo, tbr
=================================================================
... (trace and max-eigenvalue tables) ...

=================================================================
JOHANSEN TEST  |  det_order=0, k_ar_diff=2
Variables: mpr, infl, exo, tbr
=================================================================
... (trace and max-eigenvalue tables) ...

=================================================================
JOHANSEN TEST  |  det_order=0, k_ar_diff=3
Variables: mpr, infl, exo, tbr
=================================================================
... (trace and max-eigenvalue tables) ...


=================================================================
ROBUSTNESS SUMMARY
=================================================================
 k_ar_diff   Trace Rank   Max-Eig Rank
--------------------------------------
         1            1              1
         2            1              1
         3            1              1

Trace rank is 1 at all lag orders -- ROBUST.

Saved to results/cointegration_johansen.csv

=================================================================
Johansen cointegration analysis complete.
=================================================================
```

(Your numbers will vary. The example above shows a rank of 1 for illustration.)

**What changed from Step 2:**

- The main block now runs the Johansen test three times with `k_ar_diff = 1, 2, 3` instead of once. This checks whether the cointegration rank is stable across different lag specifications.
- A robustness summary table is printed at the end, making it easy to see whether the rank changes.
- Results are saved to `results/cointegration_johansen.csv` with one row per lag specification. This CSV goes into your thesis appendix.
- `os.makedirs(RESULTS_DIR, exist_ok=True)` ensures the results folder exists before writing.

**Why test with three lags?** The `k_ar_diff` parameter controls how many lagged differences the underlying VECM includes. Different lag orders absorb different amounts of short-run dynamics. If the rank is the same across lags 1, 2, and 3, the result is robust. If it changes, the result is fragile and you need the information criteria from Day 10 to determine the optimal lag before drawing conclusions.

---

## Interpreting Your Results

### What to Expect for Nigerian Data

For the mpr-infl-exo-tbr system, expect a cointegration rank of **1 or 2**.

**Rank = 1** is the most common finding. It means there is one long-run equilibrium tying the four variables together. Economically, this captures the monetary policy transmission mechanism: the CBN sets the mpr, which anchors the tbr (treasury bill rates track the policy rate with a spread), which interacts with the exo (interest rate differentials affect capital flows and forex demand), which feeds into infl (import prices pass through to consumer prices). All four are bound by one equilibrium "leash."

**Rank = 2** is also plausible. The second equilibrium might capture a distinct interest rate channel -- the mpr-tbr relationship is so tight (the CBN directly influences tbr through open market operations) that it forms its own long-run equilibrium separate from the inflation-exchange rate channel.

### When Trace and Max-Eigenvalue Disagree

This happens when a test statistic is close to the critical value. One test rejects and the other does not. When this occurs:

1. Report both results honestly. Do not hide the disagreement.
2. Prefer the Trace test. It has better finite-sample (small-sample) power. Simulation studies by Lutkepohl, Saikkonen, and Trenkler (2001) confirmed this.
3. Explain to your examiner: "The Trace test and Max-Eigenvalue test gave different ranks. Following the recommendation in the literature, I report the Trace test result of rank = X, noting that the Max-Eigenvalue test suggested rank = Y."

### What This Means for Your Model

The cointegration rank confirms that the ARDL bounds testing approach is appropriate for your thesis. Here is why:

- If rank >= 1, your variables share at least one long-run equilibrium. The ARDL model can capture this equilibrium in its long-run equation and the adjustment toward it in its short-run (error correction) equation.
- The ARDL bounds test has an advantage over the Johansen approach: it works regardless of whether variables are I(0) or I(1), so you do not need every variable to be the same integration order. But knowing the Johansen rank gives you a cross-check -- if the ARDL finds a long-run relationship and the Johansen rank is >= 1, the two methods agree, which strengthens your thesis.

---

## Commit

```bash
git add econometric_models/cointegration.py results/cointegration_johansen.csv
git commit -m "Day 9: Johansen cointegration test with robustness check"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `ImportError: cannot import name 'coint_johansen'` | Make sure you import from the correct path: `from statsmodels.tsa.vector_ar.vecm import coint_johansen`. If this still fails, run `pip install --upgrade statsmodels`. |
| `LinAlgError: Singular matrix` | You have too many lags for your data. With 300 observations and 4 variables, `k_ar_diff` up to about 12 is safe. If you set it higher, reduce it. |
| `ValueError: det_order must be -1, 0, or 1` | You passed an invalid value for `det_order`. The only valid options are -1, 0, or 1. We use 0. |
| `KeyError: 'infl'` or similar column name error | Your cleaned CSV does not have the expected column names. Open `data/processed/cleaned_data.csv` and check the header. Columns must be exactly `mpr`, `infl`, `exo`, `tbr`. |
| Trace and Max-Eigenvalue give different ranks | This is normal, not an error. Report both. Prefer the Trace test result. See the interpretation section above. |
| Confusing `k_ar_diff` with the VAR lag order | `k_ar_diff` is the number of lagged differences in the VECM, which equals the VAR lag order minus 1. So `k_ar_diff=1` corresponds to VAR(2), `k_ar_diff=2` corresponds to VAR(3), and so on. |

---

## Check Your Understanding

### 1. "Why did you run the Johansen test after already running Engle-Granger?"

**Answer:** "The Engle-Granger test only examines two variables at a time. With four variables in my system -- mpr, infl, exo, and tbr -- there can be up to three cointegrating relationships. The Engle-Granger test cannot detect multiple relationships and it is sensitive to which variable you place on the left side of the regression. The Johansen test examines all four variables simultaneously, determines exactly how many cointegrating relationships exist, and does not depend on an arbitrary normalization choice. I ran Engle-Granger first as a pairwise screening step, then used Johansen for the definitive multivariate result."

### 2. "What does a cointegration rank of 1 mean economically?"

**Answer:** "A rank of 1 means there is one independent long-run equilibrium relationship among the four variables. Individually, mpr, infl, exo, and tbr are all non-stationary -- they wander without reverting to a fixed mean. But there exists one specific linear combination of all four that IS stationary. This means the four variables cannot drift apart forever. When a shock pushes them away from their equilibrium, an error correction mechanism pulls them back. Economically, this single equilibrium likely captures the monetary policy transmission mechanism: the CBN's policy rate influences treasury bill rates, which interact with the exchange rate, which feeds into consumer price inflation. All four variables are bound together by this one long-run relationship."

### 3. "Why did you test with k_ar_diff = 1, 2, and 3?"

**Answer:** "The Johansen test results can be sensitive to the lag specification. The parameter k_ar_diff controls how many lagged differences are included in the underlying Vector Error Correction Model. Different lag orders capture different amounts of short-run dynamics, which can change the test statistics and potentially the rank. By testing with three different lag orders, I check whether my result is robust. If the rank is the same across all three -- say, rank = 1 at lags 1, 2, and 3 -- I can report with confidence that the result does not depend on a particular lag choice. If the rank varies, I flag it as sensitive and use information criteria to select the optimal lag before drawing conclusions."

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Johansen test function | `econometric_models/cointegration.py` | Tests all 4 variables simultaneously for multivariate cointegration |
| Robustness check | `econometric_models/cointegration.py` | Runs Johansen with k_ar_diff = 1, 2, 3 to verify the rank is stable |
| Johansen results CSV | `results/cointegration_johansen.csv` | Summary table with rank at each lag order, ready for your thesis appendix |

**No new packages installed today.** The `coint_johansen` function is part of statsmodels, which was installed on Day 6.

**Tomorrow (Day 10):** You will determine the optimal lag order for your VAR/VECM model using information criteria (AIC, BIC, HQIC). The lag order is a critical input -- too few lags and the model misses important dynamics; too many and you waste degrees of freedom. Day 10 ties together the stationarity results (Days 6-7) and the cointegration results (Days 8-9) to prepare you for model estimation in Week 3.
