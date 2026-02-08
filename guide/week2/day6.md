# Week 2, Day 6 — Stationarity Testing: The ADF Test

## What You'll Learn Today

- How to install numpy and statsmodels
- What stationarity is and why it matters for econometrics
- What the Augmented Dickey-Fuller (ADF) test does
- How to test all four variables in levels and first differences
- How to interpret the results for Nigerian macroeconomic data

## Why This Matters

You are about to cross the line from descriptive analysis into formal econometrics. Everything you did in Week 1 — loading data, cleaning it, computing statistics, making plots — was preparation. Starting today, you are doing the actual statistical testing that your thesis or research paper will be built on.

The single most important concept in time series econometrics is **stationarity**. If you do not test for stationarity before fitting a model, your results will be wrong. Not slightly wrong — completely, fundamentally wrong. You will get regression results that look impressive (high R-squared, significant t-statistics) but are actually meaningless. This is called "spurious regression," and it has ruined more undergraduate theses than any other mistake.

Today you will learn what stationarity means, why it matters, how to test for it using the Augmented Dickey-Fuller (ADF) test, and what the results mean for your Nigerian data. By the end of today, you will have a clear answer to the question every examiner asks: "Are your variables stationary?"

---

## Step 1: Install New Packages

You need two new packages today. Update your `requirements.txt` so it reads exactly:

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

**What you should see:** A lot of download and installation text. statsmodels has several dependencies (scipy, patsy, packaging), so expect more output than usual. It should end with `Successfully installed` followed by a list of packages.

Verify both packages installed correctly:

```bash
python -c "import numpy; print(numpy.__version__)"
python -c "import statsmodels; print(statsmodels.__version__)"
```

The first should print `1.26.2`. The second should print `0.14.1`.

**What these packages are:**

- **numpy** (Numerical Python) is Python's foundational library for numerical computing. It provides fast array operations, linear algebra, random number generation, and mathematical functions. Almost every scientific Python library (pandas, statsmodels, matplotlib) is built on top of numpy. When statsmodels runs the ADF test, it is doing matrix math with numpy under the hood.

- **statsmodels** is Python's econometrics library. It contains everything you need for time series analysis: ADF tests, KPSS tests, ARDL models, VAR models, impulse response functions, Granger causality tests, cointegration tests, and much more. If pandas is your data manipulation tool and matplotlib is your plotting tool, statsmodels is your econometrics tool. You will use it every day from now until the end of this project.

---

## Theory: What is Stationarity and Why It Matters

This section is long. Read it carefully. Understanding stationarity is not optional — it is the foundation of everything you will do for the next seven weeks.

### What Stationarity Means

A **stationary** time series has three properties:

1. **Constant mean.** The average value of the series does not change over time. If you take the mean of the first 100 observations and the mean of the last 100 observations, they should be roughly the same.

2. **Constant variance.** The spread of the series does not change over time. If the series fluctuates by plus or minus 2 in the first half, it should fluctuate by roughly plus or minus 2 in the second half as well.

3. **Autocovariance depends only on the lag, not on time.** The correlation between an observation and the observation 3 months ago is the same whether you are looking at 2005 or 2020.

Informally: a stationary series "looks the same" no matter where you slice it. If you covered up the x-axis dates, you would not be able to tell which part of the series came first and which came last.

A **non-stationary** series violates one or more of these properties. The most common violation in economic data is a **trend** — the mean changes over time. A series that trends upward has a higher mean at the end than at the beginning. That alone makes it non-stationary.

### Why It Matters: Spurious Regression

Here is the critical problem. Suppose you have two time series that are both trending upward over 25 years, but they have absolutely nothing to do with each other. Maybe one is Nigeria's exchange rate and the other is the number of mobile phone subscriptions in Brazil. If you run a simple regression of one on the other, you will get:

- A high R-squared (maybe 0.85 or higher)
- A highly significant t-statistic (p-value near zero)
- The appearance that one variable "explains" the other

But the relationship is completely fake. The only reason you got significant results is that both series happen to trend upward. They are correlated with time, not with each other.

This is called **spurious regression**. Granger and Newbold (1974) demonstrated this problem in a famous paper, and it is one of the most important results in econometrics. The key finding: when you regress one non-stationary series on another, the standard statistical tests (t-tests, F-tests, R-squared) are unreliable. They will almost always tell you the relationship is significant, even when it is not.

The solution: **you must test for stationarity before fitting any model.** If the variables are non-stationary, you need to either difference them (make them stationary) or use special techniques like cointegration that are designed for non-stationary data. You cannot just run a regression and hope for the best.

### The Augmented Dickey-Fuller (ADF) Test

The ADF test is the most widely used test for stationarity. Here is how it works:

- **Null hypothesis (H0):** The series HAS a unit root. In other words, the series is non-stationary.
- **Alternative hypothesis (H1):** The series does NOT have a unit root. The series is stationary.

Notice that the null hypothesis is non-stationarity, not stationarity. This means the test is conservative — you need strong evidence to conclude that a series is stationary. If the evidence is ambiguous, the test defaults to "non-stationary."

**How to read the results:**

- If the **p-value < 0.05**: reject the null hypothesis. Conclude the series IS stationary. We call this I(0) — "integrated of order zero."
- If the **p-value >= 0.05**: fail to reject the null. Conclude the series is NON-stationary. It needs differencing.

**What about the test statistic and critical values?** The test statistic is a number (usually negative). The more negative it is, the stronger the evidence against the null. The critical values tell you the threshold: if the test statistic is more negative than the critical value at the 5% level, you reject the null. The p-value gives you the same information in a more intuitive form, so most people just look at the p-value.

### First Differencing and Integration Order

When a series is non-stationary, we take its **first difference**: the change from one period to the next.

If the original series is `y_t`, the first difference is `y_t - y_{t-1}`. In pandas, this is `df["inflation"].diff()`.

For example, if inflation in January is 15% and in February is 16%, the first difference is +1 percentage point. The first difference captures the *change* in the variable rather than its *level*.

We then run the ADF test on the first difference. If the first difference is stationary, we say the original series is **I(1)** — "integrated of order 1." This means you need to difference it once to make it stationary.

If the first difference is still non-stationary, we take the second difference (the change in the change) and test again. If that is stationary, the series is I(2). In practice, most economic variables are either I(0) or I(1). I(2) is rare.

### What to Expect for Nigerian Data

Before you run the test, let us predict what you will find based on the EDA you did on Day 5:

- **Inflation:** Likely non-stationary in levels because of the strong upward trend since 2020. Inflation went from 12% to 35% — the mean is clearly not constant. After differencing (the month-to-month change in inflation), it should become stationary. Prediction: I(1).

- **MPR:** Likely non-stationary in levels. The MPR was around 11-14% for years, then shot up to 27.5% in 2023-2024. That is a trend, even though it moves in discrete steps. After differencing, the month-to-month changes should be stationary. Prediction: I(1). Note: the MPR might be borderline because it is held constant for months at a time. The ADF test may struggle with this "step function" behaviour.

- **Exchange rate:** Definitely non-stationary in levels. You saw the structural breaks in 2016 and 2023 on Day 5. The rate went from 92 to 1,680 over the sample period. After differencing, the month-to-month changes should be stationary. Prediction: I(1).

- **M2 (money supply):** Exponential growth makes it non-stationary. The mean, variance, and every other property change dramatically over the sample. After differencing, the month-to-month changes in M2 should be stationary. Prediction: I(1).

If all four variables turn out to be I(1), that is a very important result. It means they all need differencing before standard regression, but it also opens the door to **cointegration testing** — the possibility that these non-stationary variables share a long-run equilibrium relationship. We will explore that on Day 8-9.

---

## Building `econometric_models/stationarity.py` — Step by Step

At each step, we show you the **complete file** from the first line to the last line. Delete everything in `econometric_models/stationarity.py` and replace it with exactly what is shown. No guessing where to put things.

Make sure you have an `econometric_models/` folder with an `__init__.py` inside it (from Day 1). If not, create them now.

---

### Build Step 1: Create the File with Imports and Data Loading

Create a new file called `econometric_models/stationarity.py`. Your complete file should look like this:

```python
import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(df.head())
```

**Run it:**

```bash
python -m econometric_models.stationarity
```

**What you should see:**

```
Loaded 300 rows
Columns: ['mpr', 'inflation', 'exchange_rate', 'm2']
              mpr  inflation  exchange_rate       m2
date
2000-01-01  13.5       6.62          92.34   1070.50
2000-02-01  13.5       6.93          92.55   1078.20
...
```

(Your exact numbers will depend on your data.)

**What just happened — line by line:**

- `from statsmodels.tsa.stattools import adfuller` — This imports the ADF test function from statsmodels. The path `tsa.stattools` means "time series analysis, statistical tools." If this line runs without error, statsmodels is installed correctly.
- `PROCESSED_DIR` and `RESULTS_DIR` — Same pattern as Day 5. Paths relative to the script's location. `os.path.dirname(__file__)` is the folder this script lives in (`econometric_models/`), and `".."` goes up one level to the project root.
- `load_data()` — Reads the cleaned CSV from Day 4. Same logic as the EDA script: `index_col="date"` makes dates the row index, `parse_dates=True` converts strings to datetime objects.
- The `if __name__ == "__main__":` block loads the data and prints basic info to confirm everything works.

If that ran and printed your data, statsmodels is working. Move on.

---

### Build Step 2: Add the adf_test Function and Test One Variable

Delete everything in `econometric_models/stationarity.py` and replace it with this:

```python
import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def adf_test(series, name):
    """
    Run the Augmented Dickey-Fuller test on a series.

    Null hypothesis: the series has a unit root (non-stationary).
    If p-value < 0.05, we reject the null and conclude the series IS stationary.
    """
    result = adfuller(series.dropna(), autolag="AIC")

    test_stat = result[0]
    p_value = result[1]
    lags_used = result[2]
    n_obs = result[3]
    critical_values = result[4]

    print(f"\n{'='*50}")
    print(f"ADF Test: {name}")
    print(f"{'='*50}")
    print(f"Test Statistic:  {test_stat:.4f}")
    print(f"P-Value:         {p_value:.4f}")
    print(f"Lags Used:       {lags_used}")
    print(f"Observations:    {n_obs}")
    for level, cv in critical_values.items():
        print(f"Critical Value ({level}): {cv:.4f}")

    if p_value < 0.05:
        print(f"Conclusion: STATIONARY (reject H0 at 5%)")
    else:
        print(f"Conclusion: NON-STATIONARY (fail to reject H0)")

    return {"test_statistic": test_stat, "p_value": p_value,
            "lags_used": lags_used, "stationary": p_value < 0.05}


if __name__ == "__main__":
    df = load_data()
    adf_test(df["inflation"], "Inflation (Level)")
```

**Run it:**

```bash
python -m econometric_models.stationarity
```

**What you should see:**

```
==================================================
ADF Test: Inflation (Level)
==================================================
Test Statistic:  -1.XXXX
P-Value:         0.XXXX
Lags Used:       XX
Observations:    XXX
Critical Value (1%): -3.XXXX
Critical Value (5%): -2.XXXX
Critical Value (10%): -2.XXXX
Conclusion: NON-STATIONARY (fail to reject H0)
```

The p-value should be well above 0.05, and the conclusion should say NON-STATIONARY. This is exactly what we predicted — inflation has a clear upward trend and is not stationary in levels.

**What each line of the adf_test function does:**

- `adfuller(series.dropna(), autolag="AIC")` — This is the core line. It runs the Augmented Dickey-Fuller test.
  - `series.dropna()` removes any NaN (missing) values first. The ADF test cannot handle NaN values and will crash if they are present.
  - `autolag="AIC"` tells the function to automatically choose the best number of lags using the Akaike Information Criterion. The ADF test includes lagged differences of the series to account for serial correlation. Too few lags means the test is invalid; too many lags waste degrees of freedom. AIC finds the optimal balance.

- `result = adfuller(...)` returns a tuple (a fixed-length list) with six elements:
  - `result[0]` — The test statistic. This is the ADF t-statistic. More negative means stronger evidence of stationarity.
  - `result[1]` — The p-value. If this is less than 0.05, reject the null (series is stationary).
  - `result[2]` — The number of lags the test used (chosen automatically by AIC).
  - `result[3]` — The number of observations used in the test (total rows minus lags).
  - `result[4]` — A dictionary of critical values at the 1%, 5%, and 10% significance levels.
  - `result[5]` — The AIC value (we do not use this directly).

- The test statistic is compared against critical values. If the test statistic is MORE NEGATIVE than the critical value at 5%, we reject the null. For example, if the test statistic is -3.50 and the 5% critical value is -2.87, then -3.50 < -2.87, so we reject. The p-value gives the same information in a more intuitive form: if p < 0.05, reject the null.

- The function returns a dictionary with the key results so we can collect them into a table later.

---

### Build Step 3: Test All Four Variables in Levels

Delete everything in `econometric_models/stationarity.py` and replace it with this:

```python
import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def adf_test(series, name):
    """
    Run the Augmented Dickey-Fuller test on a series.

    Null hypothesis: the series has a unit root (non-stationary).
    If p-value < 0.05, we reject the null and conclude the series IS stationary.
    """
    result = adfuller(series.dropna(), autolag="AIC")

    test_stat = result[0]
    p_value = result[1]
    lags_used = result[2]
    n_obs = result[3]
    critical_values = result[4]

    print(f"\n{'='*50}")
    print(f"ADF Test: {name}")
    print(f"{'='*50}")
    print(f"Test Statistic:  {test_stat:.4f}")
    print(f"P-Value:         {p_value:.4f}")
    print(f"Lags Used:       {lags_used}")
    print(f"Observations:    {n_obs}")
    for level, cv in critical_values.items():
        print(f"Critical Value ({level}): {cv:.4f}")

    if p_value < 0.05:
        print(f"Conclusion: STATIONARY (reject H0 at 5%)")
    else:
        print(f"Conclusion: NON-STATIONARY (fail to reject H0)")

    return {"test_statistic": test_stat, "p_value": p_value,
            "lags_used": lags_used, "stationary": p_value < 0.05}


if __name__ == "__main__":
    df = load_data()

    print("=" * 60)
    print("STATIONARITY TESTING — ADF TEST (LEVELS)")
    print("=" * 60)

    variables = ["mpr", "inflation", "exchange_rate", "m2"]
    for var in variables:
        adf_test(df[var], f"{var} (Level)")
```

**Run it:**

```bash
python -m econometric_models.stationarity
```

**What you should see:** Four ADF test outputs, one for each variable. All four should show `NON-STATIONARY (fail to reject H0)` with p-values above 0.05.

```
============================================================
STATIONARITY TESTING — ADF TEST (LEVELS)
============================================================

==================================================
ADF Test: mpr (Level)
==================================================
Test Statistic:  -X.XXXX
P-Value:         0.XXXX
...
Conclusion: NON-STATIONARY (fail to reject H0)

==================================================
ADF Test: inflation (Level)
==================================================
...
Conclusion: NON-STATIONARY (fail to reject H0)

==================================================
ADF Test: exchange_rate (Level)
==================================================
...
Conclusion: NON-STATIONARY (fail to reject H0)

==================================================
ADF Test: m2 (Level)
==================================================
...
Conclusion: NON-STATIONARY (fail to reject H0)
```

All four variables are non-stationary in levels. This is exactly what we expected from the time series plots on Day 5 — all four have clear trends. But we need to formally test this rather than just eyeball it, because examiners want statistical evidence, not visual impressions.

Now we need to test whether the first differences are stationary.

---

### Build Step 4: Add First Difference Testing and Save Results (Final Version)

Delete everything in `econometric_models/stationarity.py` and replace it with this:

```python
import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def adf_test(series, name):
    """
    Run the Augmented Dickey-Fuller test on a series.

    Null hypothesis: the series has a unit root (non-stationary).
    If p-value < 0.05, we reject the null and conclude the series IS stationary.
    """
    result = adfuller(series.dropna(), autolag="AIC")

    test_stat = result[0]
    p_value = result[1]
    lags_used = result[2]
    n_obs = result[3]
    critical_values = result[4]

    print(f"\n{'='*50}")
    print(f"ADF Test: {name}")
    print(f"{'='*50}")
    print(f"Test Statistic:  {test_stat:.4f}")
    print(f"P-Value:         {p_value:.4f}")
    print(f"Lags Used:       {lags_used}")
    print(f"Observations:    {n_obs}")
    for level, cv in critical_values.items():
        print(f"Critical Value ({level}): {cv:.4f}")

    if p_value < 0.05:
        print(f"Conclusion: STATIONARY (reject H0 at 5%)")
    else:
        print(f"Conclusion: NON-STATIONARY (fail to reject H0)")

    return {"test_statistic": test_stat, "p_value": p_value,
            "lags_used": lags_used, "stationary": p_value < 0.05}


if __name__ == "__main__":
    df = load_data()
    variables = ["mpr", "inflation", "exchange_rate", "m2"]
    results = []

    # Test in levels
    print("=" * 60)
    print("STATIONARITY TESTING — ADF TEST")
    print("=" * 60)

    print("\n>>> TESTING IN LEVELS (original data)")
    for var in variables:
        r = adf_test(df[var], f"{var} (Level)")
        results.append({"variable": var, "form": "Level",
                        "test_stat": r["test_statistic"],
                        "p_value": r["p_value"], "stationary": r["stationary"]})

    # Test in first differences
    print("\n\n>>> TESTING IN FIRST DIFFERENCES (change from previous month)")
    for var in variables:
        diff_series = df[var].diff().dropna()
        r = adf_test(diff_series, f"{var} (First Difference)")
        results.append({"variable": var, "form": "First Difference",
                        "test_stat": r["test_statistic"],
                        "p_value": r["p_value"], "stationary": r["stationary"]})

    # Summary table
    results_df = pd.DataFrame(results)
    print("\n\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(RESULTS_DIR, "adf_test_results.csv"), index=False)
    print(f"\nSaved to results/adf_test_results.csv")
```

Build Step 4 above is your final complete file.

**Run it:**

```bash
python -m econometric_models.stationarity
```

**What you should see:** First, the levels tests (all NON-STATIONARY), then the first difference tests (all STATIONARY), then the summary table:

```
============================================================
STATIONARITY TESTING — ADF TEST
============================================================

>>> TESTING IN LEVELS (original data)

==================================================
ADF Test: mpr (Level)
==================================================
...
Conclusion: NON-STATIONARY (fail to reject H0)

==================================================
ADF Test: inflation (Level)
==================================================
...
Conclusion: NON-STATIONARY (fail to reject H0)

==================================================
ADF Test: exchange_rate (Level)
==================================================
...
Conclusion: NON-STATIONARY (fail to reject H0)

==================================================
ADF Test: m2 (Level)
==================================================
...
Conclusion: NON-STATIONARY (fail to reject H0)


>>> TESTING IN FIRST DIFFERENCES (change from previous month)

==================================================
ADF Test: mpr (First Difference)
==================================================
...
Conclusion: STATIONARY (reject H0 at 5%)

==================================================
ADF Test: inflation (First Difference)
==================================================
...
Conclusion: STATIONARY (reject H0 at 5%)

==================================================
ADF Test: exchange_rate (First Difference)
==================================================
...
Conclusion: STATIONARY (reject H0 at 5%)

==================================================
ADF Test: m2 (First Difference)
==================================================
...
Conclusion: STATIONARY (reject H0 at 5%)


============================================================
SUMMARY TABLE
============================================================
      variable             form  test_stat  p_value  stationary
           mpr            Level     -X.XXX    0.XXX       False
     inflation            Level     -X.XXX    0.XXX       False
 exchange_rate            Level     -X.XXX    0.XXX       False
            m2            Level     -X.XXX    0.XXX       False
           mpr  First Difference    -X.XXX    0.XXX        True
     inflation  First Difference    -X.XXX    0.XXX        True
 exchange_rate  First Difference    -X.XXX    0.XXX        True
            m2  First Difference    -X.XXX    0.XXX        True

Saved to results/adf_test_results.csv
```

(The `X.XXX` values will be actual numbers when you run it.)

**What changed from the previous version:**

- We now test both **levels** and **first differences** for every variable.
- `df[var].diff().dropna()` computes the first difference. `.diff()` calculates `y_t - y_{t-1}` for every row. The first row becomes NaN (because there is no previous month to subtract), so `.dropna()` removes it.
- Results are collected into a list of dictionaries, then converted to a DataFrame with `pd.DataFrame(results)`. This gives us a clean summary table.
- `results_df.to_string(index=False)` prints the table without the row index numbers (we do not need them).
- `os.makedirs(RESULTS_DIR, exist_ok=True)` creates the `results/` folder if it does not already exist. `exist_ok=True` means it will not crash if the folder is already there.
- The results are saved to `results/adf_test_results.csv` so you can reference them later or paste them into your thesis.

---

## Interpreting the Results — Nigeria Specific

You now have formal statistical evidence about the stationarity of each variable. Here is what the results mean and what you should say about them.

### Inflation

**In levels: NON-STATIONARY.** The ADF test confirms what the time series plot showed on Day 5. Inflation has a strong upward trend since 2020, rising from about 12% to over 34%. The mean is not constant — it is much higher in the second half of the sample than the first. The test statistic is not negative enough to reject the null of a unit root.

**In first differences: STATIONARY.** The first difference of inflation is the month-to-month *change* in the inflation rate. While the inflation rate itself trends upward, the monthly changes fluctuate around a roughly constant mean (close to zero, with occasional spikes). The ADF test strongly rejects the null in first differences.

**Conclusion: Inflation is I(1).** Integrated of order one. It needs to be differenced once to become stationary.

### Exchange Rate

**In levels: NON-STATIONARY.** The exchange rate has the most dramatic non-stationarity of any variable in the dataset. It went from 92 NGN/USD in 2000 to over 1,680 NGN/USD in 2024, with massive structural breaks in 2016 and 2023. The p-value is almost certainly very high (well above 0.05).

**In first differences: STATIONARY.** The month-to-month change in the exchange rate does not have a persistent trend. Even though there are occasional large jumps (June 2016, June 2023), the differenced series reverts to a mean near zero between those jumps. The ADF test rejects the null.

**Conclusion: The exchange rate is I(1).** Note that the structural breaks in the exchange rate may cause the ADF test to have low power (it may fail to reject the null even when the series is actually stationary around a broken trend). This is a known limitation, but for our purposes, the I(1) result is consistent with economic theory and our visual analysis.

### M2 (Money Supply)

**In levels: NON-STATIONARY.** M2 shows exponential growth. It grew from about 1 trillion Naira in 2000 to about 285 trillion in 2024. The mean, variance, and every other statistical property change dramatically over the sample period. This is as non-stationary as a series can get.

**In first differences: STATIONARY.** The month-to-month change in M2 (how much the money supply grew in a given month) fluctuates around a mean without a persistent trend. The ADF test strongly rejects the null.

**Conclusion: M2 is I(1).** In practice, many researchers take the log of M2 before differencing, because the log transform converts exponential growth into linear growth. The first difference of log(M2) is approximately the monthly percentage growth rate of money supply. We may revisit this transformation later, but for now, the key finding is that M2 is non-stationary in levels and stationary in first differences.

### MPR (Monetary Policy Rate)

**In levels: NON-STATIONARY.** The MPR might seem like it should be stationary because it stays at the same level for months at a time. But the recent aggressive tightening cycle (from 11.5% to 27.5% in under two years) creates an upward trend in the second half of the sample, making the overall series non-stationary.

**In first differences: STATIONARY.** The month-to-month change in the MPR is mostly zero (because the rate is held constant between MPC meetings), with occasional non-zero values when the rate is changed. This differenced series has no trend and is stationary.

**Conclusion: MPR is I(1).** The MPR is a borderline case — it has characteristics of both stationary and non-stationary behaviour. But the formal ADF test says non-stationary in levels, and that is what we report.

### The Big Picture

All four variables are I(1). This is the single most important result so far, because it determines everything about how you build your models:

1. **You cannot run a simple OLS regression of inflation on MPR, exchange rate, and M2 in levels.** The results would be spurious. The high R-squared and significant coefficients would be artifacts of common trends, not real relationships.

2. **You can regress the first differences on each other.** If you difference everything, the variables are stationary and standard regression is valid. But differencing throws away the long-run level information — you only capture short-run dynamics.

3. **You should test for cointegration.** If the I(1) variables share a long-run equilibrium relationship, they are cointegrated, and you can use an Error Correction Model or ARDL bounds test to capture both short-run and long-run dynamics. This is what we do on Day 8-9.

4. **The ARDL bounds testing approach is ideal.** The ARDL model can handle a mix of I(0) and I(1) variables (though not I(2)). Since all our variables are I(1), the ARDL approach is valid. We will build this model in Week 3-4.

### What to Tell Your Examiner

> "All four variables — the monetary policy rate, headline inflation, the nominal exchange rate, and broad money supply M2 — are integrated of order one, I(1), as determined by the Augmented Dickey-Fuller test. All variables are non-stationary in levels but become stationary after first differencing. The presence of unit roots in all variables motivates the use of the ARDL bounds testing approach, which can handle a mix of I(0) and I(1) variables, and the Johansen cointegration framework, which tests whether these I(1) variables share long-run equilibrium relationships."

This is a statement you can put directly into your thesis methodology section.

---

## Step 2: Commit

```bash
git add econometric_models/stationarity.py requirements.txt
git commit -m "Day 6: ADF stationarity test for all variables"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'statsmodels'` | You forgot to install statsmodels. Run `pip install -r requirements.txt` and try again. Make sure your virtual environment is activated. |
| `ModuleNotFoundError: No module named 'numpy'` | numpy is a dependency of statsmodels but should install automatically. If it did not, run `pip install numpy==1.26.2` explicitly. |
| `FileNotFoundError` when loading `cleaned_data.csv` | You need to run the cleaning script first: `python -m data_processing.clean`. The stationarity script reads from `data/processed/cleaned_data.csv`, which is created by the Day 4 cleaning step. |
| `KeyError: 'inflation'` or `KeyError: 'mpr'` | Your cleaned CSV does not have the expected column names. Open `data/processed/cleaned_data.csv` and check the header row. The columns must be exactly: `date`, `mpr`, `inflation`, `exchange_rate`, `m2`. |
| `ValueError: x is constant` or `ValueError: zero-size array` | One of your columns has no variation (all the same value) or is empty after dropping NaN. Check your data with `print(df.describe())` to see if any column has zero standard deviation. This usually means the data was not loaded correctly. |
| `TypeError: No numeric types to aggregate` | Your columns are being read as strings instead of numbers. Add `print(df.dtypes)` after loading to check. All four variable columns should be `float64`. If they are `object`, go back to Day 4 and fix the cleaning script. |

---

## Check Your Understanding

These are the kinds of questions an examiner will ask about stationarity. Practice answering them out loud before your defense.

### 1. "What is the null hypothesis of the ADF test?"

**Answer:** "The null hypothesis of the Augmented Dickey-Fuller test is that the series has a unit root — that is, the series is non-stationary. The alternative hypothesis is that the series is stationary. This means the test is conservative: we need strong evidence (a p-value below 0.05) to conclude that a series IS stationary. If the evidence is ambiguous, the default conclusion is non-stationarity. In our analysis, all four variables failed to reject the null in levels, indicating unit roots, but strongly rejected the null after first differencing, confirming they are I(1)."

### 2. "What does I(1) mean?"

**Answer:** "I(1) means integrated of order one. A variable is I(1) if it is non-stationary in its original form (in levels) but becomes stationary after being differenced once. The 'one' refers to the number of times you need to difference the series to achieve stationarity. An I(0) variable is already stationary in levels and does not need differencing at all. An I(2) variable would require differencing twice, but this is rare in practice. All four of our variables — MPR, inflation, exchange rate, and M2 — are I(1), meaning they contain stochastic trends in levels that are removed by taking first differences."

### 3. "Why can't you just run a regression on non-stationary data?"

**Answer:** "Because of the spurious regression problem, demonstrated by Granger and Newbold in 1974. When you regress one non-stationary series on another, the standard statistical tests — t-tests, F-tests, R-squared — become unreliable. You will get a high R-squared and significant t-statistics even when the two variables have no genuine relationship. This happens because both series are correlated with time (they both trend), not necessarily with each other. The standard OLS assumptions break down: residuals are not white noise, they are autocorrelated and non-stationary themselves. To avoid this, we must either difference the variables to make them stationary before regression, or use cointegration techniques that are specifically designed for non-stationary data, such as the ARDL bounds test or the Johansen procedure."

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Stationarity test script | `econometric_models/stationarity.py` | Runs the ADF test on all 4 variables in levels and first differences |
| ADF results table | `results/adf_test_results.csv` | Summary of test statistics, p-values, and stationarity conclusions for all 8 tests |
| Updated dependencies | `requirements.txt` | Added numpy and statsmodels |

**Packages installed today:** `numpy==1.26.2`, `statsmodels==0.14.1`

**Key finding:** All four variables (MPR, inflation, exchange rate, M2) are I(1) — non-stationary in levels, stationary in first differences.

**Tomorrow (Day 7):** We run the KPSS test as a confirmatory stationarity test. The KPSS test has the *opposite* null hypothesis from the ADF test (it assumes stationarity under the null), so using both tests together gives you much stronger evidence. If ADF says non-stationary AND KPSS says non-stationary, you can be confident. This is what examiners expect to see.
