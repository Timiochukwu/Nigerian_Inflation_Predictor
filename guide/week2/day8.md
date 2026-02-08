# Week 2, Day 8 — Cointegration: Do These Variables Share a Long-Run Relationship?

## What You'll Learn Today

- What cointegration means and why it matters for your model
- How two non-stationary series can still have a meaningful long-run relationship
- How to run the Engle-Granger pairwise cointegration test
- How to interpret cointegration results in the context of the Nigerian economy
- Why pairwise testing is only the first step (Johansen comes tomorrow)

## Why This Matters

On Days 6-7 you established that all four of your variables — MPR, inflation, exchange_rate, and M2 — are I(1). They are non-stationary. They wander without a fixed mean, and if you difference them once, they become stationary.

This creates a problem. If you run an ordinary least squares (OLS) regression of one I(1) variable on another I(1) variable, you can get results that look highly significant — high R-squared, low p-values — but are completely meaningless. This is the **spurious regression** problem, first demonstrated by Granger and Newbold (1974). Two completely unrelated series that both trend upward will appear to be related simply because they both trend upward. The regression is picking up the common trend, not a genuine economic relationship.

So here is the question Day 8 answers: **Are our I(1) variables genuinely connected to each other, or would regressing them together just give us garbage?**

The answer comes from cointegration testing. If two I(1) series are cointegrated, their relationship is real — not spurious. You can safely model them together in levels. If they are NOT cointegrated, you must work with their differences instead. This decision shapes your entire modelling strategy for the rest of the project.

---

## No New Packages

You do not need to install anything new today. The `statsmodels` package you installed on Day 6 already includes the `coint` function for the Engle-Granger cointegration test. Everything you need is already in your environment.

---

## Theory: What Is Cointegration?

This is one of the most important concepts in time series econometrics, and it is also one of the hardest to grasp on first encounter. Read this section carefully. Read it twice if you need to. It is worth the time.

### The Problem: Spurious Regression with I(1) Variables

You proved on Days 6-7 that all four variables are I(1) — they are non-stationary in levels and stationary in first differences. This means each variable wanders over time without reverting to a fixed mean. MPR drifts, inflation drifts, the exchange rate drifts, M2 drifts.

Now imagine you regress inflation on the exchange rate using OLS. Both series trend upward over the 2000-2024 period. OLS will find a "relationship" — it will say the exchange rate is a statistically significant predictor of inflation with a high R-squared. But this could be completely meaningless. OLS is just picking up the fact that both series went up. If you regressed inflation on the number of mobile phone subscribers in Nigeria (which also went up over this period), you would get an equally "significant" result. That does not mean mobile phones cause inflation.

This is why Granger and Newbold warned econometricians: **never regress one I(1) series on another I(1) series without first checking for cointegration.** If the series are not cointegrated, the regression is spurious and its results cannot be trusted.

### The Key Insight: Engle and Granger (1987)

Robert Engle and Clive Granger made a discovery that earned them the Nobel Prize in Economics (2003). They showed that even though two I(1) series individually wander without a fixed mean, there might exist a **linear combination** of them that IS stationary.

What does that mean in practice? Suppose inflation and MPR are both I(1). Individually, they wander. But suppose there exists some constant `b` such that:

```
inflation - b * MPR = stationary residual
```

If this residual (the gap between inflation and `b * MPR`) is stationary — meaning it fluctuates around a fixed mean and always reverts back — then inflation and MPR are **cointegrated**. They share a long-run equilibrium. They may diverge from each other in the short run, but they always come back together.

This is a profound result. It means that even though neither series has a fixed mean on its own, the GAP between them does. The relationship between them is real, not spurious.

### The Drunk and the Dog: An Analogy

Think of a drunk person walking their dog on a leash. The drunk staggers randomly — left, right, forward, back. Their path is non-stationary (a random walk). The dog also wanders — sniffing here, running there. The dog's path is also non-stationary.

But here is the crucial point: the drunk and the dog are connected by a leash. They can never get too far apart. If the dog runs ahead, the leash pulls it back. If the drunk staggers too far left, the leash drags the dog along.

The **distance between the drunk and the dog** is stationary. It fluctuates — sometimes the dog is ahead, sometimes behind — but it always reverts to the length of the leash. It has a fixed mean.

The drunk and the dog are cointegrated. Their individual paths are non-stationary, but they share a long-run equilibrium (defined by the leash). The "error" (the distance between them) is a stationary, mean-reverting process.

Now remove the leash. The drunk and the dog wander independently. Sometimes they happen to be close, sometimes far apart, but there is no force pulling them back together. The distance between them is non-stationary — it can drift forever. Without the leash, they are NOT cointegrated.

### Economic Meaning for Nigeria

**If MPR and inflation are cointegrated**, it means the Central Bank of Nigeria's policy rate and the inflation rate share a long-run equilibrium. When inflation rises above the equilibrium level, the CBN eventually raises the MPR to bring it back. When inflation falls below equilibrium, the CBN eventually cuts the MPR. They may diverge in the short run — the CBN may be slow to respond, or inflation may overshoot — but there is an invisible leash connecting them. The CBN's policy is systematic enough to maintain a long-run link.

This is exactly what you would hope to find if the CBN is doing its job. A central bank that targets inflation should have a cointegrating relationship between its policy rate and the inflation rate. If they are NOT cointegrated, it suggests the CBN's response to inflation has been too inconsistent, too delayed, or too disrupted by political interference to create a stable long-run equilibrium.

**If exchange_rate and M2 are cointegrated**, it means money supply growth and Naira depreciation share a long-run link. This is consistent with the quantity theory of money applied to the exchange rate: when the CBN prints more Naira (M2 grows) without a corresponding increase in dollar-earning capacity, the Naira loses value. The exchange rate and M2 are connected by a theoretical leash — the fundamental relationship between money supply and currency value.

**If exchange_rate and inflation are cointegrated**, it means exchange rate depreciation and consumer price inflation share a long-run equilibrium — the exchange rate pass-through is a permanent, structural feature of the Nigerian economy, not just a short-run phenomenon.

### The Engle-Granger Cointegration Test

The Engle-Granger test is the simplest and most intuitive cointegration test. It works in four steps:

1. **Regress Y on X using OLS in levels.** For example, regress inflation on MPR. This gives you a fitted line and a set of residuals (the differences between actual inflation and what the OLS line predicts).

2. **Extract the residuals.** These residuals represent the "gap" between the two series after accounting for their linear relationship. If the series are cointegrated, this gap should be stationary.

3. **Test the residuals for stationarity using the ADF test.** Run an Augmented Dickey-Fuller test on the residuals. If the residuals are stationary (have no unit root), the gap reverts to a mean — the series are cointegrated.

4. **Make the decision:**
   - If the residuals are stationary: Y and X **are cointegrated**. Their relationship is genuine, not spurious. OLS in levels gives valid (in fact, "super-consistent") estimates.
   - If the residuals are non-stationary: Y and X are **NOT cointegrated**. Their apparent relationship may be spurious. You should not model them together in levels.

The good news is that `statsmodels` wraps all four steps into a single function called `coint()`. You pass it two series, and it returns a test statistic, a p-value, and critical values.

**Hypotheses:**
- **Null hypothesis (H0):** The two series are NOT cointegrated (no long-run equilibrium exists)
- **Alternative hypothesis (H1):** The two series ARE cointegrated (a long-run equilibrium exists)

**Decision rule:**
- If p-value < 0.05: **reject H0** — the series are cointegrated at the 5% significance level
- If p-value >= 0.05: **fail to reject H0** — no evidence of cointegration

Note that the critical values for the Engle-Granger test are NOT the same as the standard ADF critical values. They are more negative (harder to reject) because you are testing residuals from a regression, not a raw series. The `coint()` function uses the correct critical values automatically.

---

## Building `econometric_models/cointegration.py` — Step by Step

At each step, we show you the **complete file** from the first line to the last line. Delete everything in the file and replace it with exactly what is shown. No guessing where to put things.

Make sure you have an `econometric_models/` folder with an `__init__.py` inside it (from Days 6-7). If not, create them now.

---

### Build Step 1: Create the File with Imports and Data Loading

Create a new file called `econometric_models/cointegration.py`. Delete everything in `econometric_models/cointegration.py` and replace it with this:

```python
import os
import pandas as pd
from statsmodels.tsa.stattools import coint

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows")
    print(f"Variables: {list(df.columns)}")
```

**Run it:**

```bash
python -m econometric_models.cointegration
```

**What you should see:**

```
Loaded 300 rows
Variables: ['mpr', 'inflation', 'exchange_rate', 'm2']
```

(Your exact row count may differ depending on your data.)

**What just happened — line by line:**

- `from statsmodels.tsa.stattools import coint` — This imports the Engle-Granger cointegration test function. It is in the same `stattools` module as the `adfuller` function you used on Days 6-7 for the ADF test. The `coint` function wraps the entire Engle-Granger procedure (OLS regression, residual extraction, ADF test on residuals) into a single call.
- `PROCESSED_DIR` and `RESULTS_DIR` — Paths relative to where the script lives. `os.path.dirname(__file__)` means "the folder this script is in" (which is `econometric_models/`). The `".."` goes up one level to the project root, then into the target subfolder.
- `load_data()` — Reads the cleaned CSV from Day 4. Same pattern you have used in every script so far. `index_col="date"` makes the date column the row index, `parse_dates=True` converts it to datetime objects.
- The `if __name__ == "__main__":` block loads the data and prints a quick confirmation so you know it works before adding anything else.

If that ran and printed your data summary, you are ready for the next step.

---

### Build Step 2: Add the Engle-Granger Test Function

Delete everything in `econometric_models/cointegration.py` and replace it with this:

```python
import os
import pandas as pd
from statsmodels.tsa.stattools import coint

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
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
    print(f"Loaded {len(df)} rows\n")

    # Test the key policy pair: inflation vs MPR
    result = engle_granger_test(df["inflation"], df["mpr"],
                                "inflation", "mpr")
```

**Run it:**

```bash
python -m econometric_models.cointegration
```

**What you should see:**

```
Loaded 300 rows

=======================================================
Engle-Granger Test: inflation & mpr
=======================================================
Test Statistic:  -X.XXXX
P-Value:         0.XXXX
Critical Values: 1%=-X.XXXX, 5%=-X.XXXX, 10%=-X.XXXX
Conclusion: COINTEGRATED (reject H0 at 5%)
```

or:

```
Conclusion: NOT cointegrated (fail to reject H0)
```

(The exact numbers depend on your data. Either result is valid — see the interpretation section below.)

**What just happened — line by line:**

- `coint(series1, series2)` — This is the entire Engle-Granger test in one call. Behind the scenes, `statsmodels` performs OLS regression of `series1` on `series2`, extracts the residuals, and runs an ADF test on those residuals with adjusted critical values. It returns three things:
  - `score` — The ADF test statistic on the residuals. More negative = stronger evidence of cointegration.
  - `p_value` — The probability of observing this test statistic if the null hypothesis (no cointegration) is true. Small p-value = reject the null = cointegrated.
  - `critical_values` — The threshold values at the 1%, 5%, and 10% significance levels. If `score` is more negative than the critical value, you reject the null at that level.
- The function prints a formatted report and returns a dictionary with the results so we can collect them into a summary table later.
- In the main block, we test the single most important pair first: inflation and MPR. This is the key policy question — does the CBN's interest rate share a long-run equilibrium with inflation?

**Understanding the output:**

The test statistic is negative (like the ADF test). The more negative it is, the stronger the evidence that the residuals are stationary and the series are cointegrated. Compare it to the critical values:
- If the test statistic is more negative than the 5% critical value: cointegrated at 5% significance
- If it is more negative than the 1% critical value: cointegrated at 1% significance (very strong evidence)
- If it is less negative than the 10% critical value: no evidence of cointegration

The p-value gives you the same information in a simpler format. Below 0.05 means cointegrated at 5%.

Good. The function works for one pair. Now let us test all pairs.

---

### Build Step 3: Test All Pairs and Save Results (Final Version)

Delete everything in `econometric_models/cointegration.py` and replace it with this:

```python
import os
from itertools import combinations

import pandas as pd
from statsmodels.tsa.stattools import coint

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
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
    variables = ["mpr", "inflation", "exchange_rate", "m2"]

    print("=" * 60)
    print("COINTEGRATION TESTING — ENGLE-GRANGER (PAIRWISE)")
    print("=" * 60)

    results = []
    for var1, var2 in combinations(variables, 2):
        r = engle_granger_test(df[var1], df[var2], var1, var2)
        results.append(r)

    # Summary
    results_df = pd.DataFrame(results)
    print("\n\n" + "=" * 60)
    print("SUMMARY: PAIRWISE COINTEGRATION")
    print("=" * 60)
    print(results_df.to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(
        os.path.join(RESULTS_DIR, "cointegration_engle_granger.csv"),
        index=False,
    )
    print(f"\nSaved to results/cointegration_engle_granger.csv")
```

Build Step 3 above is your final complete file.

**Run it:**

```bash
python -m econometric_models.cointegration
```

**What you should see:**

```
============================================================
COINTEGRATION TESTING — ENGLE-GRANGER (PAIRWISE)
============================================================

=======================================================
Engle-Granger Test: mpr & inflation
=======================================================
Test Statistic:  -X.XXXX
P-Value:         0.XXXX
Critical Values: 1%=-X.XXXX, 5%=-X.XXXX, 10%=-X.XXXX
Conclusion: ... (COINTEGRATED or NOT cointegrated)

=======================================================
Engle-Granger Test: mpr & exchange_rate
=======================================================
...

=======================================================
Engle-Granger Test: mpr & m2
=======================================================
...

=======================================================
Engle-Granger Test: inflation & exchange_rate
=======================================================
...

=======================================================
Engle-Granger Test: inflation & m2
=======================================================
...

=======================================================
Engle-Granger Test: exchange_rate & m2
=======================================================
...


============================================================
SUMMARY: PAIRWISE COINTEGRATION
============================================================
                  pair  test_stat  p_value  cointegrated
     mpr & inflation    -X.XXXX   0.XXXX          True
  mpr & exchange_rate   -X.XXXX   0.XXXX         False
            mpr & m2   -X.XXXX   0.XXXX         False
inflation & exchange_rate -X.XXXX 0.XXXX          True
          inflation & m2 -X.XXXX  0.XXXX          True
     exchange_rate & m2  -X.XXXX  0.XXXX          True

Saved to results/cointegration_engle_granger.csv
```

(Your exact numbers and True/False values will depend on your data. The example above is illustrative — your results may differ.)

**What changed from the previous version:**

- `from itertools import combinations` — This is a Python standard library function that generates all unique pairs from a list. Given `["mpr", "inflation", "exchange_rate", "m2"]`, it produces the 6 pairs: (mpr, inflation), (mpr, exchange_rate), (mpr, m2), (inflation, exchange_rate), (inflation, m2), (exchange_rate, m2). This is exactly C(4,2) = 6 combinations.
- The `for var1, var2 in combinations(variables, 2):` loop runs the Engle-Granger test on each pair and collects the results into a list.
- `results_df = pd.DataFrame(results)` converts the list of dictionaries into a DataFrame, which gives us a clean summary table.
- The summary table is printed and saved to CSV. This CSV file goes into your thesis appendix.

**Why 6 pairs?** With 4 variables, the number of unique pairs is 4! / (2! * 2!) = 6. These are:

| Pair # | Variable 1 | Variable 2 |
|--------|-----------|-----------|
| 1 | mpr | inflation |
| 2 | mpr | exchange_rate |
| 3 | mpr | m2 |
| 4 | inflation | exchange_rate |
| 5 | inflation | m2 |
| 6 | exchange_rate | m2 |

Each pair is tested independently. This is pairwise cointegration testing.

---

## Interpreting Your Results — Nigeria Specific

Your results will fall into one of several patterns. Here is how to interpret the most likely outcomes for Nigerian data.

### Pairs Likely to Show Cointegration

**exchange_rate and m2:** This pair is the most likely to be cointegrated. Both series have been driven by the same fundamental force over the past 25 years — monetary expansion. As the CBN expanded the money supply (through deficit monetization, Ways and Means advances to the federal government, and other channels), the Naira lost value against the dollar. The quantity theory of money predicts this: more Naira chasing the same amount of foreign currency pushes the exchange rate up. The long-run link between money supply and currency value acts as the "leash" in our analogy.

**inflation and exchange_rate:** This pair often shows cointegration because exchange rate pass-through is a structural feature of the Nigerian economy. Nigeria imports a significant share of what it consumes — refined petroleum, food, machinery, chemicals. When the Naira depreciates, import prices rise, and those price increases feed into the CPI. This pass-through creates a long-run equilibrium between the exchange rate and inflation.

**inflation and m2:** Money supply growth and inflation are linked through the quantity theory of money (MV = PQ). If money supply grows faster than real output, prices must eventually rise. In Nigeria, M2 has grown at 20-30% per year while real GDP grew at 2-4%. The difference shows up as inflation over the long run.

### Pairs That May or May Not Show Cointegration

**mpr and inflation:** This is the most interesting pair from a policy perspective. If they are cointegrated, it means the CBN's monetary policy has been systematic enough to maintain a long-run equilibrium with inflation — the CBN raises rates when inflation rises and cuts when inflation falls, in a consistent, predictable way.

If they are NOT cointegrated, it does not mean the CBN is irrelevant. It means the CBN's response to inflation has been too inconsistent to create a stable long-run equilibrium. This could be because:
- The CBN was slow to respond to inflation shocks (delayed tightening cycles)
- Political interference disrupted monetary policy (the CBN's independence has been questioned at various points in Nigerian history)
- The CBN pursued multiple conflicting objectives simultaneously (price stability, exchange rate stability, growth, development finance) and could not maintain a consistent inflation response
- Structural breaks in the data (the 2016 recession, COVID-19, the 2023 exchange rate unification) disrupted the equilibrium

Both outcomes are valid findings. If they are cointegrated, you report that monetary policy and inflation share a long-run equilibrium. If they are not, you report that the relationship is unstable and discuss the structural and institutional reasons why.

**mpr and exchange_rate / mpr and m2:** These pairs test whether the CBN's policy rate has a long-run relationship with the exchange rate or money supply. In theory, higher interest rates should attract foreign capital (supporting the Naira) and slow money supply growth. In practice, Nigeria's capital controls, multiple exchange rate regimes, and the dominance of oil revenues in determining the exchange rate may weaken or break these links.

### What If Most Pairs Are NOT Cointegrated?

This is a perfectly valid finding. It means the pairwise long-run relationships between your variables are weak or nonexistent. There are two important things to keep in mind:

1. **The Engle-Granger test only checks PAIRWISE cointegration.** It tests whether two specific variables share a long-run equilibrium. But in a system of four variables, there may be a cointegrating relationship that involves THREE or FOUR variables simultaneously. For example, there might be no pairwise cointegration between inflation and MPR, or between inflation and exchange_rate, but a linear combination of ALL FOUR variables might be stationary. The Engle-Granger test cannot detect this.

2. **Tomorrow you will run the Johansen test (Day 9).** The Johansen test examines all four variables simultaneously and can detect MULTIPLE cointegrating vectors. It is more powerful than the Engle-Granger test for multivariate systems. So if you find few or no cointegrating pairs today, the Johansen test may still find cointegration in the full system.

### What If ALL Pairs Are Cointegrated?

This is also a valid finding, especially for Nigerian macroeconomic data. If all four variables are I(1) and all driven by the same underlying forces (monetary expansion, Naira depreciation, CBN policy responses), then multiple pairwise cointegrating relationships are plausible. This would suggest a strongly interconnected system where all variables share long-run equilibria — consistent with the macroeconomic theory that MPR, inflation, exchange rate, and money supply are all part of the same monetary transmission mechanism.

### What to Tell Your Examiner

Here is how to discuss your cointegration results in your thesis defense:

**If you found cointegration in most pairs:** "The Engle-Granger pairwise tests indicate significant cointegrating relationships between [list the pairs]. This suggests that despite being individually non-stationary, these variables share long-run equilibria — consistent with [monetary transmission theory / quantity theory of money / exchange rate pass-through]. The presence of cointegration means that modelling these variables in levels using an Error Correction Model (ECM) or VECM framework is appropriate and will not produce spurious results."

**If you found cointegration in few pairs:** "The Engle-Granger pairwise tests found cointegration between [list the pairs], but not between [list the others]. The absence of pairwise cointegration in some pairs does not rule out multivariate cointegration. The Johansen test, which examines the full system simultaneously, may detect cointegrating relationships that the pairwise approach misses. Additionally, the limited cointegration may reflect structural instability in the Nigerian economy — particularly the exchange rate regime changes and shifts in monetary policy frameworks over the sample period."

**Regardless of results:** "It is important to note that the Engle-Granger test has lower statistical power than the Johansen test in multivariate settings, and it is sensitive to the choice of which variable is used as the dependent variable in the first-step regression. The Johansen results in the next section provide a more comprehensive picture of the cointegrating structure."

---

## Commit

```bash
git add econometric_models/cointegration.py
git commit -m "Day 8: Engle-Granger pairwise cointegration tests"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: cleaned_data.csv not found` or similar | You need to run the cleaning script first: `python -m data_processing.clean`. The cointegration script reads from `data/processed/cleaned_data.csv`, which is created by the Day 4 cleaning step. |
| `ModuleNotFoundError: No module named 'statsmodels'` | You need statsmodels installed. Run `pip install statsmodels` (it should already be installed from Day 6). Check that your virtual environment is activated. |
| `ImportError: cannot import name 'coint' from 'statsmodels.tsa.stattools'` | Your version of statsmodels may be too old. Run `pip install --upgrade statsmodels` to get the latest version. The `coint` function has been available since statsmodels 0.8, so any recent version should work. |
| `KeyError: 'mpr'` or `KeyError: 'inflation'` | Your cleaned CSV does not have the expected column names. Open `data/processed/cleaned_data.csv` and check the header row. The columns must be exactly: `date`, `mpr`, `inflation`, `exchange_rate`, `m2`. If the names differ, update the `variables` list in the script to match. |
| `ValueError: x and y must have the same length` | One of your series has missing values that created a length mismatch. Go back to Day 4's cleaning script and make sure all missing values are handled. The `coint` function requires both series to have the same number of observations with no NaN values. |
| `InfestimableError` or `LinAlgError: Singular matrix` | This happens when one of the series has zero variance (all values are the same) or when the two series are perfectly collinear. Check your data with `df.describe()` to make sure all columns have variation. |

---

## Check Your Understanding

These are the kinds of questions an examiner will ask about cointegration. Practice answering them out loud before your defense.

### 1. "What is cointegration in plain English?"

**Answer:** "Cointegration means that two time series that individually wander — they are non-stationary, they do not revert to a fixed mean — are nonetheless connected to each other in the long run. The gap between them is stationary. It fluctuates, but it always reverts back to a mean. They may diverge in the short run, but they always come back together.

The analogy I use is a drunk person walking their dog on a leash. Both the drunk and the dog follow random, unpredictable paths — both are non-stationary. But the distance between them is bounded by the leash and always reverts to a manageable range. That distance is stationary. The drunk and the dog are cointegrated.

In economic terms, cointegration means two variables share a long-run equilibrium. Short-run shocks can push them apart, but economic forces pull them back together over time."

### 2. "If two variables are cointegrated, what does that tell you about running OLS on them in levels?"

**Answer:** "If two I(1) variables are cointegrated, it means that running OLS on them in levels is NOT spurious — the relationship the regression finds is genuine, not an artifact of common trends. In fact, OLS gives what econometricians call 'super-consistent' estimates of the cointegrating vector. The OLS estimates converge to the true parameter values faster than they would in a standard stationary regression.

This is the foundation of the Engle-Granger two-step method: first, estimate the long-run relationship using OLS in levels; second, use the residuals from that regression as an error correction term in a short-run dynamic model. The first step works precisely because cointegration makes OLS in levels valid.

However, the standard errors from the first-step OLS are not reliable for inference — you cannot use them for hypothesis testing. For valid inference on the cointegrating parameters, you need methods like the Johansen procedure, Dynamic OLS (DOLS), or Fully Modified OLS (FMOLS)."

### 3. "Why might MPR and inflation not be cointegrated even though the CBN targets inflation?"

**Answer:** "Cointegration requires a SYSTEMATIC long-run equilibrium — a consistent, predictable link between the two variables that persists over the entire sample period. Several factors specific to Nigeria could prevent this:

First, the CBN's response to inflation has not always been timely. There have been long periods where the MPC held the MPR constant despite changing inflation conditions. This delayed response weakens the equilibrium relationship.

Second, political interference has disrupted monetary policy at various points. The CBN's operational independence has been questioned, particularly during periods where the government needed deficit financing. When political considerations override inflation targeting, the systematic link breaks down.

Third, the CBN has pursued multiple conflicting objectives — price stability, exchange rate stability, economic growth, financial inclusion, development finance. When the central bank tries to do too many things at once, it cannot maintain a consistent inflation response function.

Fourth, structural breaks in the data — the 2016 recession, COVID-19 in 2020, the 2023 exchange rate unification — may have disrupted the equilibrium relationship. A cointegrating relationship that held before 2016 might not hold after 2023 because the economic structure changed fundamentally.

The absence of cointegration does not mean the MPR is irrelevant to inflation. It means the relationship is not stable enough over the full sample period to qualify as a long-run equilibrium. The CBN may still influence inflation — just not in the systematic, predictable way that cointegration requires."

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Cointegration test script | `econometric_models/cointegration.py` | Runs Engle-Granger pairwise cointegration tests on all 6 variable pairs |
| Cointegration results CSV | `results/cointegration_engle_granger.csv` | Summary table with test statistics, p-values, and cointegration decisions for all pairs |

**No new packages installed today.** The `coint` function comes from `statsmodels`, which you installed on Day 6.

**Tomorrow (Day 9):** You will run the Johansen cointegration test, which examines all four variables simultaneously and can detect multiple cointegrating vectors. This is more powerful than today's pairwise approach and will give you a complete picture of the long-run relationships in your system.
