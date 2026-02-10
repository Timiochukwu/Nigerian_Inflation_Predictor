# Week 2, Day 6 -- Stationarity Testing: The ADF Test

## What You'll Learn Today

- How to install numpy and statsmodels
- What stationarity is and why it matters for econometrics
- What the Augmented Dickey-Fuller (ADF) test does
- How to test all four variables in levels and first differences
- How to determine the integration order of each variable
- How to interpret the results for Nigerian macroeconomic data

## Why This Matters

You are about to cross the line from descriptive analysis into formal econometrics. Everything you did in Week 1 -- loading data, cleaning it, computing statistics, making plots -- was preparation. Starting today, you are doing the actual statistical testing that your thesis or research paper will be built on.

The single most important concept in time series econometrics is **stationarity**. If you do not test for stationarity before fitting a model, your results will be wrong. Not slightly wrong -- completely, fundamentally wrong. You will get regression results that look impressive (high R-squared, significant t-statistics) but are actually meaningless. This is called **spurious regression**, and it has ruined more undergraduate theses than any other mistake.

Here is the practical problem. Suppose you regress the exchange rate (exo) on the monetary policy rate (mpr). Both variables trend upward over 2000-2024. Your regression will report a high R-squared and a significant coefficient -- not because the MPR genuinely explains the exchange rate, but because both variables happen to increase over time. They are correlated with time itself, not necessarily with each other. Granger and Newbold (1974) proved this formally: when you regress one non-stationary (trending) series on another, the standard t-tests and F-tests give you false positives almost every time.

The solution is simple: **test for stationarity before you fit any model.** That is what today is about. By the end of today, you will have a formal, statistical answer to the question every examiner asks: "Are your variables stationary?"

---

## Step 1: Install New Packages

You need two new packages today. Delete everything in `requirements.txt` and replace it with this:

```
pandas==2.1.4
matplotlib==3.8.2
numpy==1.26.2
statsmodels==0.14.1
```

Then install everything:

```bash
pip install -r requirements.txt
```

**What you should see:** A lot of download and installation text. statsmodels has several dependencies (scipy, patsy, packaging), so expect more output than usual. It should end with `Successfully installed` followed by a list of packages, or it will say the requirements are already satisfied if you have them.

Verify both packages installed correctly:

```bash
python -c "import numpy; print(numpy.__version__)"
python -c "import statsmodels; print(statsmodels.__version__)"
```

The first should print `1.26.2`. The second should print `0.14.1`.

**What these packages are:**

- **numpy** (Numerical Python) is Python's foundational library for numerical computing. It provides fast array operations, linear algebra, random number generation, and mathematical functions. Almost every scientific Python library (pandas, statsmodels, matplotlib) is built on top of numpy. When statsmodels runs the ADF test, it is doing matrix math with numpy under the hood. You will not call numpy functions directly today, but statsmodels needs it to work.

- **statsmodels** is Python's econometrics library. It contains everything you need for time series analysis: ADF tests, KPSS tests, ARDL models, VAR models, impulse response functions, Granger causality tests, cointegration tests, and much more. If pandas is your data manipulation tool and matplotlib is your plotting tool, statsmodels is your econometrics tool. You will use it every day from now until the end of this project.

---

## Theory: What Is Stationarity?

This section is long. Read it carefully. Understanding stationarity is not optional -- it is the foundation of everything you will do for the next several weeks.

A **stationary** time series has three properties that remain constant over time:

1. **Constant mean.** The average value of the series does not drift upward or downward. If you split the series in half and compute the mean of each half, the two means should be roughly the same.

2. **Constant variance.** The spread of the series does not change over time. If it fluctuates by plus or minus 2 percentage points in the first half of the sample, it should fluctuate by roughly the same amount in the second half.

3. **Autocovariance depends only on the lag, not on time.** The correlation between an observation and the observation 3 months earlier is the same whether you are looking at 2005 or 2020.

Informally: a stationary series "looks the same" no matter where you slice it. If you covered up the x-axis dates, you would not be able to tell which part of the series came first and which came last.

Think about it with Nigerian data. Inflation (infl) has been accelerating since 2020, climbing from about 12% to above 33%. If you compute the average inflation for 2000-2012 and then for 2013-2024, you get very different numbers. That tells you the mean is not constant -- inflation is non-stationary. Now think about the exchange rate (exo). It went from about 92 naira per dollar in 2000 to over 1,500 in 2024. That is obviously trending -- the mean changes dramatically over time. Non-stationary.

**Why this matters: spurious regression.** When you regress one non-stationary series on another, the standard statistical tests (t-tests, F-tests, R-squared) become unreliable. You will almost always get a significant result, even when the two variables have no genuine relationship. The residuals from such a regression are not white noise -- they are autocorrelated and non-stationary themselves. Every conclusion you draw is built on a false foundation.

**What the ADF test does.** The Augmented Dickey-Fuller test is the most widely used test for stationarity. It works like this:

- **Null hypothesis (H0):** The series HAS a unit root -- it is non-stationary.
- **Alternative hypothesis (H1):** The series does NOT have a unit root -- it is stationary.

Notice the null is non-stationarity. This makes the test conservative: you need strong evidence (p-value below 0.05) to conclude stationarity. If the evidence is ambiguous, the default conclusion is "non-stationary."

- If the **p-value < 0.05**: reject the null. The series IS stationary.
- If the **p-value >= 0.05**: fail to reject the null. The series is NOT stationary (it has a unit root).

When a series is non-stationary in levels (its original form), we take its **first difference** -- the change from one month to the next. If the first difference is stationary, we say the original series is **I(1)** -- "integrated of order 1." If the original series is already stationary without differencing, it is **I(0)** -- "integrated of order 0." Most Nigerian macroeconomic variables are I(1).

---

## Building `econometric_models/stationarity.py` -- Step by Step

At each step, we show you the **complete file** from the first line to the last line. Delete everything in `econometric_models/stationarity.py` and replace it with exactly what is shown. No guessing where to put things.

Make sure you have an `econometric_models/` folder with an `__init__.py` inside it (from Day 1). If not, create them now.

---

### Build Step 1: Imports, Load Data, Print First 5 Rows

Create a new file called `econometric_models/stationarity.py`. Delete everything in `econometric_models/stationarity.py` and replace it with this:

```python
"""Stationarity testing for the Nigerian Inflation Predictor."""
import os
import sys
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded: {df.shape[0]} observations, {df.shape[1]} variables")
    print(f"Period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    print(f"\nFirst 5 rows:")
    print(df.head())
```

**Run it:**

```bash
python -m econometric_models.stationarity
```

**What you should see:**

```
Data loaded: 300 observations, 4 variables
Period: 2000-01 to 2024-12

First 5 rows:
             mpr   infl    exo    tbr
date
2000-01-01  13.5   6.62  92.34  12.00
2000-02-01  13.5   6.93  92.55  11.50
2000-03-01  13.5   7.87  92.69  11.80
2000-04-01  13.5   8.13  93.05  12.30
2000-05-01  13.5   8.68  95.10  13.00
```

(Your exact numbers will depend on your data.)

**What just happened -- line by line:**

- `"""Stationarity testing..."""` -- This is a module-level docstring. It describes what the file does. It appears when someone runs `help()` on your module or reads the code later.
- `import sys` -- We import this for safety. It gives access to system-level operations. We do not use it directly today, but it is a standard import for scripts that might need to exit early with error messages.
- `import numpy as np` -- Imports numpy and gives it the short name `np`. This is a universal convention in Python data science. We do not call numpy directly today, but statsmodels requires it internally.
- `from statsmodels.tsa.stattools import adfuller` -- This imports the ADF test function. The path `tsa.stattools` means "time series analysis, statistical tools." If this line runs without error, statsmodels is installed correctly.
- `PROCESSED_DIR` and `RESULTS_DIR` -- Paths relative to the script's location. `os.path.dirname(__file__)` is the folder this script lives in (`econometric_models/`), and `".."` goes up one level to the project root.
- `load_data()` -- Reads the cleaned CSV from Day 4. `index_col="date"` makes dates the row index. `parse_dates=True` converts strings to datetime objects.
- `df.shape[0]` is the number of rows (observations). `df.shape[1]` is the number of columns (variables).
- `df.index[0]` is the first date in the dataset. `.strftime('%Y-%m')` formats it as "2000-01" for clean display.

If that ran and printed your data, statsmodels is working. Move on.

---

### Build Step 2: Add the adf_test Function and Test All Variables in Levels

Delete everything in `econometric_models/stationarity.py` and replace it with this:

```python
"""Stationarity testing for the Nigerian Inflation Predictor."""
import os
import sys
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def adf_test(series, variable_name, significance=0.05):
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


if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded: {df.shape[0]} observations, {df.shape[1]} variables")
    print(f"Period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")

    print("\n" + "=" * 60)
    print("ADF TESTS -- LEVELS (original data)")
    print("=" * 60)

    results_levels = []
    for col in df.columns:
        result = adf_test(df[col], col)
        results_levels.append(result)
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
ADF TESTS -- LEVELS (original data)
============================================================

ADF Test for: mpr
  ADF Statistic : -1.XXXX
  p-value       : 0.XXXX
  Lags Used     : X
  Observations  : XXX
  Critical Values:
    1%: -3.XXXX
    5%: -2.XXXX
    10%: -2.XXXX
  Result: NON-STATIONARY (has unit root)

ADF Test for: infl
  ADF Statistic : -1.XXXX
  p-value       : 0.XXXX
  ...
  Result: NON-STATIONARY (has unit root)

ADF Test for: exo
  ADF Statistic : -0.XXXX
  p-value       : 0.XXXX
  ...
  Result: NON-STATIONARY (has unit root)

ADF Test for: tbr
  ADF Statistic : -1.XXXX
  p-value       : 0.XXXX
  ...
  Result: NON-STATIONARY (has unit root)
```

All four variables should show NON-STATIONARY with p-values well above 0.05. This is exactly what we predicted -- the EDA plots from Day 5 showed clear trends in these series.

**What each piece of the adf_test function does:**

- `adfuller(series.dropna(), autolag="AIC")` -- This is the core line. It runs the Augmented Dickey-Fuller test.
  - `series.dropna()` removes any NaN (missing) values first. The ADF test cannot handle NaN values and will crash if they are present.
  - `autolag="AIC"` tells the function to automatically choose the best number of lags using the Akaike Information Criterion. The ADF test includes lagged differences of the series to account for serial correlation. Too few lags means the test is biased; too many lags waste degrees of freedom. AIC finds the optimal balance automatically.

- `result = adfuller(...)` returns a tuple with six elements:
  - `result[0]` -- The ADF test statistic. More negative means stronger evidence of stationarity.
  - `result[1]` -- The p-value. Below 0.05 means stationary.
  - `result[2]` -- The number of lags used (chosen by AIC).
  - `result[3]` -- The number of observations used (total rows minus lags).
  - `result[4]` -- A dictionary of critical values at the 1%, 5%, and 10% significance levels.
  - `result[5]` -- The AIC value (we do not use this directly).

- The `significance=0.05` parameter lets you change the threshold if needed, but 5% is the standard in economics.

- The function returns a dictionary so we can collect all results into a table later.

---

### Build Step 3: Add First Difference Testing

Delete everything in `econometric_models/stationarity.py` and replace it with this:

```python
"""Stationarity testing for the Nigerian Inflation Predictor."""
import os
import sys
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def adf_test(series, variable_name, significance=0.05):
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


if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded: {df.shape[0]} observations, {df.shape[1]} variables")
    print(f"Period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")

    print("\n" + "=" * 60)
    print("ADF TESTS -- LEVELS (original data)")
    print("=" * 60)

    results_levels = []
    for col in df.columns:
        result = adf_test(df[col], col)
        results_levels.append(result)

    print("\n" + "=" * 60)
    print("ADF TESTS -- FIRST DIFFERENCES (delta)")
    print("=" * 60)

    df_diff = df.diff().dropna()
    results_diff = []
    for col in df_diff.columns:
        result = adf_test(df_diff[col], f"d_{col}")
        results_diff.append(result)
```

**Run it:**

```bash
python -m econometric_models.stationarity
```

**What you should see:** First, the four level tests (all NON-STATIONARY), then four first difference tests (all STATIONARY):

```
============================================================
ADF TESTS -- FIRST DIFFERENCES (delta)
============================================================

ADF Test for: d_mpr
  ADF Statistic : -X.XXXX
  p-value       : 0.0000
  ...
  Result: STATIONARY

ADF Test for: d_infl
  ADF Statistic : -X.XXXX
  p-value       : 0.00XX
  ...
  Result: STATIONARY

ADF Test for: d_exo
  ADF Statistic : -X.XXXX
  p-value       : 0.0000
  ...
  Result: STATIONARY

ADF Test for: d_tbr
  ADF Statistic : -X.XXXX
  p-value       : 0.0000
  ...
  Result: STATIONARY
```

All four first differences should show STATIONARY with p-values well below 0.05.

**What just happened:**

- `df.diff()` computes the first difference of every column. For each row, it calculates `value_this_month - value_last_month`. The first row becomes NaN because there is no previous month to subtract from.
- `.dropna()` removes that first NaN row. Without this, the ADF test would crash on the missing value.
- We prefix the variable names with `d_` (short for "delta" or "difference") so the output clearly shows these are differenced series, not levels. In your thesis, you would write this as the Greek letter delta: `infl` for the level and `d_infl` (or the symbol delta-infl) for the first difference.

**What this means:**

If a variable is non-stationary in levels but stationary in first differences, it is **I(1)** -- integrated of order one. The "1" means you need to difference it once to make it stationary.

If a variable were already stationary in levels, it would be **I(0)** -- integrated of order zero. No differencing needed.

If the first difference were still non-stationary (very rare), you would take the second difference and test again. If that were stationary, the series would be I(2). In practice, almost all Nigerian macroeconomic variables are either I(0) or I(1). I(2) is extremely uncommon.

For the ARDL bounds testing approach that we will use later, we need all variables to be either I(0) or I(1). The ARDL model **cannot handle I(2) variables**. So this test is not just academic -- it determines whether your chosen modelling framework is valid.

---

### Build Step 4: Add Summary Table and Save to CSV (Final Version)

Delete everything in `econometric_models/stationarity.py` and replace it with this:

```python
"""Stationarity testing for the Nigerian Inflation Predictor."""
import os
import sys
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def adf_test(series, variable_name, significance=0.05):
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


if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded: {df.shape[0]} observations, {df.shape[1]} variables")
    print(f"Period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")

    print("\n" + "=" * 60)
    print("ADF TESTS -- LEVELS (original data)")
    print("=" * 60)

    results_levels = []
    for col in df.columns:
        result = adf_test(df[col], col)
        results_levels.append(result)

    print("\n" + "=" * 60)
    print("ADF TESTS -- FIRST DIFFERENCES (delta)")
    print("=" * 60)

    df_diff = df.diff().dropna()
    results_diff = []
    for col in df_diff.columns:
        result = adf_test(df_diff[col], f"d_{col}")
        results_diff.append(result)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary = []
    for lev, dif in zip(results_levels, results_diff):
        order = "I(0)" if lev["is_stationary"] else ("I(1)" if dif["is_stationary"] else "I(2)+")
        summary.append({
            "variable": lev["variable"],
            "level_pvalue": lev["p_value"],
            "level_stationary": lev["is_stationary"],
            "diff_pvalue": dif["p_value"],
            "diff_stationary": dif["is_stationary"],
            "integration_order": order,
        })

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "adf_results.csv"), index=False)
    print(f"\nResults saved to results/adf_results.csv")
```

Build Step 4 above is your final complete file.

**Run it:**

```bash
python -m econometric_models.stationarity
```

**What you should see:** All the previous output, plus a summary table at the end:

```
============================================================
SUMMARY
============================================================
variable  level_pvalue  level_stationary  diff_pvalue  diff_stationary integration_order
     mpr        0.XXXX             False       0.XXXX             True              I(1)
    infl        0.XXXX             False       0.XXXX             True              I(1)
     exo        0.XXXX             False       0.XXXX             True              I(1)
     tbr        0.XXXX             False       0.XXXX             True              I(1)

Results saved to results/adf_results.csv
```

(The `X.XXXX` values will be actual numbers when you run it.)

**What changed from the previous version:**

- The `zip(results_levels, results_diff)` pairs up each variable's level result with its difference result. `zip` takes two lists and walks through them together: the first item from each list, then the second from each, and so on.

- The integration order logic is straightforward:
  - If the level is stationary, the variable is I(0). No differencing needed.
  - If the level is non-stationary but the first difference is stationary, the variable is I(1). Difference once.
  - If both are non-stationary, it is I(2) or higher. This would be a problem for ARDL.

- `pd.DataFrame(summary)` converts the list of dictionaries into a clean table. Each dictionary becomes one row.

- `summary_df.to_string(index=False)` prints the table without row numbers. We do not need them -- the variable name identifies each row.

- `os.makedirs(RESULTS_DIR, exist_ok=True)` creates the `results/` folder if it does not exist. `exist_ok=True` means it will not crash if the folder is already there.

- The CSV file `results/adf_results.csv` is a permanent record of your stationarity tests. You can paste it directly into your thesis appendix.

---

## Interpreting the Results -- Nigeria Specific

You now have formal statistical evidence about the stationarity of each variable. Here is what the results mean and what you should say about them.

### mpr (Monetary Policy Rate)

**In levels: NON-STATIONARY.** The MPR might seem like it should be stationary because it stays at the same level for months at a time -- the MPC meets every two months and usually holds the rate steady. But the recent aggressive tightening cycle (from 11.5% in 2020 to 27.5% in late 2024) creates a strong upward trend in the latter portion of the sample, making the overall series non-stationary. The ADF test picks up this trend.

**In first differences: STATIONARY.** The month-to-month change in the MPR is mostly zero (because the rate is held constant between MPC meetings), with occasional non-zero values when the rate is changed. This differenced series has no persistent trend and reverts to a mean near zero.

**Conclusion: mpr is I(1).**

### infl (Headline Inflation Rate)

**In levels: NON-STATIONARY.** The ADF test confirms what the time series plot showed on Day 5. Inflation has a strong upward trend since 2020, climbing from around 12% to above 33%. The mean inflation rate in the second half of the sample is much higher than in the first half. The test statistic is not negative enough to reject the null of a unit root.

**In first differences: STATIONARY.** The first difference of inflation -- the month-to-month *change* in the rate -- fluctuates around a roughly constant mean close to zero. While there are occasional spikes (like when inflation jumped sharply after the June 2023 exchange rate unification), the differenced series does not drift upward or downward over time.

**Conclusion: infl is I(1).**

### exo (Official Exchange Rate)

**In levels: NON-STATIONARY.** The exchange rate has the most dramatic non-stationarity of any variable in the dataset. It went from about 92 naira per dollar in 2000 to over 1,500 in 2024, with massive structural breaks in June 2016 (when the CBN allowed the naira to float) and June 2023 (exchange rate unification under Tinubu). The p-value is almost certainly very high.

**In first differences: STATIONARY.** The month-to-month change in the exchange rate does not have a persistent trend. Even though there are occasional massive jumps (June 2016, June 2023, early 2024), between those jumps the differenced series reverts to a mean near zero.

**Conclusion: exo is I(1).** Note: the structural breaks may reduce the ADF test's power (its ability to correctly reject the null when the series is actually stationary around a broken trend). This is a known limitation, but the I(1) classification is consistent with both economic theory and visual inspection.

### tbr (Treasury Bill Rate)

**In levels: NON-STATIONARY.** The Treasury Bill Rate tracks the MPR closely but with more month-to-month variation because it is determined by market auctions rather than committee decisions. Like the MPR, the TBR shows a strong upward trend in 2022-2024 as the CBN tightened monetary policy, making the overall series non-stationary over the full sample period.

**In first differences: STATIONARY.** The month-to-month change in the TBR does not exhibit a persistent trend. Rate movements fluctuate around zero without drifting in one direction.

**Conclusion: tbr is I(1).**

### What "Integration Order" Means and Why It Matters

All four variables are I(1). This single finding determines the entire direction of your modelling strategy:

1. **You cannot run a simple OLS regression in levels.** Regressing infl on mpr, exo, and tbr in their original form would produce spurious results. The high R-squared and significant coefficients would be artifacts of common trends, not genuine relationships.

2. **You can regress the first differences on each other.** If you difference everything, the variables become stationary and standard regression is valid. But differencing throws away the long-run level information -- you only capture short-run month-to-month dynamics.

3. **You should test for cointegration.** If these I(1) variables share a long-run equilibrium relationship, they are cointegrated, and you can use an Error Correction Model or ARDL bounds test to capture both short-run and long-run dynamics. This is what we test on Day 8-9.

4. **The ARDL bounds testing approach is ideal for your data.** The ARDL model can handle a mix of I(0) and I(1) variables, but it cannot handle I(2). Since all our variables are I(1) -- none are I(2) -- the ARDL framework is valid.

### What to Tell Your Examiner

> "All four variables -- the monetary policy rate (mpr), headline inflation (infl), the official exchange rate (exo), and the Treasury Bill rate (tbr) -- are integrated of order one, I(1), as determined by the Augmented Dickey-Fuller test at the 5% significance level. All variables are non-stationary in levels but become stationary after first differencing. The absence of I(2) variables confirms the applicability of the ARDL bounds testing approach. The presence of unit roots in all variables further motivates cointegration testing to determine whether these variables share a long-run equilibrium relationship."

This is a statement you can put directly into your thesis methodology section.

---

## Commit

```bash
git add econometric_models/stationarity.py requirements.txt
git commit -m "Day 6: ADF stationarity test for all variables in levels and first differences"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'statsmodels'` | You forgot to install statsmodels. Run `pip install -r requirements.txt` and try again. Make sure your virtual environment is activated first. |
| `ModuleNotFoundError: No module named 'numpy'` | numpy is a dependency of statsmodels and should install automatically. If it did not, run `pip install numpy==1.26.2` explicitly. |
| `FileNotFoundError` when loading `cleaned_data.csv` | You need to run the cleaning script first: `python -m data_processing.clean`. The stationarity script reads from `data/processed/cleaned_data.csv`, which is created by the Day 4 cleaning step. |
| `KeyError: 'infl'` or `KeyError: 'mpr'` | Your cleaned CSV does not have the expected column names. Open `data/processed/cleaned_data.csv` and check the header row. The columns must be exactly: `mpr`, `infl`, `exo`, `tbr` with `date` as the index. |
| `ValueError: x is constant` or `ValueError: zero-size array` | One of your columns has no variation (all the same value) or is empty after dropping NaN. Check your data with `print(df.describe())` after loading to see if any column has zero standard deviation. This usually means the data was not loaded correctly. |
| `TypeError: No numeric types to aggregate` | Your columns are being read as strings instead of numbers. Add `print(df.dtypes)` after loading to check. All four variable columns should be `float64`. If they are `object`, go back to Day 4 and fix the cleaning script. |
| `PermissionError` when saving to results/ | The script cannot write to the folder. On Linux/Mac, try `chmod -R 755 results/`. On Windows, check folder permissions. |

---

## Check Your Understanding

These are the kinds of questions an examiner will ask about stationarity. Practice answering them out loud before your defense.

### 1. "What is the null hypothesis of the ADF test, and what does your result mean?"

**Answer:** "The null hypothesis of the Augmented Dickey-Fuller test is that the series has a unit root -- that is, the series is non-stationary. The alternative hypothesis is that the series is stationary. This means the test is conservative: we need strong evidence, specifically a p-value below 0.05, to conclude that a series IS stationary. If the evidence is ambiguous, the default conclusion is non-stationarity. In our analysis, all four variables -- mpr, infl, exo, and tbr -- failed to reject the null in levels, indicating the presence of unit roots. However, all four strongly rejected the null after first differencing, confirming they are integrated of order one, I(1). This finding is consistent with the time series plots from our exploratory data analysis, which showed clear upward trends in all four series."

### 2. "What does I(1) mean, and why does it matter for your model choice?"

**Answer:** "I(1) means integrated of order one. A variable is I(1) if it is non-stationary in its original form but becomes stationary after being differenced once. The '1' refers to the number of times you need to difference the series to achieve stationarity. An I(0) variable is already stationary and needs no differencing. An I(2) variable requires differencing twice, which is rare in practice. The integration order matters because it determines which econometric models are valid. You cannot run ordinary least squares regression on I(1) variables in levels -- the results will be spurious. Instead, you must either difference everything first, losing long-run information, or use models designed for non-stationary data, such as the ARDL bounds testing approach. The ARDL model accommodates both I(0) and I(1) variables but breaks down if any variable is I(2), which is why we test the integration order before selecting our model."

### 3. "Why can you not just run a regression on non-stationary data?"

**Answer:** "Because of the spurious regression problem, demonstrated by Granger and Newbold in 1974. When you regress one non-stationary series on another, the standard statistical tests -- t-tests, F-tests, and R-squared -- become unreliable. You will get a high R-squared and statistically significant coefficients even when the two variables have no genuine causal relationship. This happens because both series are correlated with time: they both trend, and that common trend inflates the apparent relationship between them. The regression residuals will be autocorrelated and non-stationary themselves, violating the classical OLS assumptions. To avoid this, we must either difference the variables to make them stationary before running a regression, or use cointegration-based techniques like the ARDL bounds test or the Johansen procedure, which are specifically designed for non-stationary data and can separate genuine long-run relationships from spurious correlations."

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Stationarity test script | `econometric_models/stationarity.py` | Runs the ADF test on all 4 variables in levels and first differences |
| ADF results table | `results/adf_results.csv` | Summary of test statistics, p-values, and integration orders for all variables |
| Updated dependencies | `requirements.txt` | Added numpy==1.26.2 and statsmodels==0.14.1 |

**Packages installed today:** `numpy==1.26.2`, `statsmodels==0.14.1`

**Key finding:** All four variables (mpr, infl, exo, tbr) are I(1) -- non-stationary in levels, stationary in first differences. No variable is I(2), confirming the ARDL bounds testing approach is applicable.

**Tomorrow (Day 7):** We run the KPSS test as a confirmatory stationarity test. The KPSS test has the *opposite* null hypothesis from the ADF test -- it assumes stationarity under the null. Using both tests together gives you much stronger evidence. If the ADF says non-stationary AND the KPSS also says non-stationary, you can be confident in your conclusion. Examiners expect to see both tests, not just one. Running only the ADF and claiming your variables are I(1) invites the question "did you confirm with a second test?" Tomorrow you will be able to answer yes.
