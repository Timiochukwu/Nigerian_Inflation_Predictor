# Week 2, Day 8 -- Engle-Granger Cointegration Test

## What You'll Learn Today

- What cointegration is and why it matters for modelling Nigerian macroeconomic data
- The Engle-Granger two-step procedure for testing whether two I(1) series share a long-run equilibrium
- How to test all 6 pairwise combinations of your 4 variables and interpret every result

## Why This Matters

On Days 6-7 you established that all four variables -- mpr, infl, exo, and tbr -- are integrated of order one, I(1). Each one wanders over time without returning to a fixed mean. You also learned that regressing one I(1) variable on another I(1) variable can produce spurious results: high R-squared, significant coefficients, but no genuine economic relationship. The regression is just picking up the fact that both series trend in the same direction.

So here is the critical question for Day 8: **Are any of these wandering variables actually connected to each other in the long run?** Or would regressing them together just give you garbage?

Cointegration testing answers this question. If two I(1) series are cointegrated, their relationship is genuine and you can safely model them together in levels. If they are NOT cointegrated, you must work with their differences instead or risk spurious results. This decision shapes your entire modelling strategy for the rest of the project.

---

## No New Packages

You do not need to install anything new today. The `coint` function for the Engle-Granger test is included in `statsmodels`, which you installed on Day 6. Everything you need is already in your environment.

---

## Theory: What Is Cointegration?

This is one of the most important concepts in time series econometrics. If you only understand one idea from this entire project, make it this one. Read this section slowly. Read it twice if you need to.

### The Problem You Face

You have four variables that are all I(1). They all wander. MPR drifts up and down as the CBN adjusts policy. Inflation drifts as supply shocks and demand pressures come and go. The exchange rate drifts as oil revenues fluctuate and the CBN intervenes in the forex market. The Treasury Bill Rate drifts as the government borrows more or less.

If you naively regress inflation on the exchange rate using OLS, both series trend upward over the 2000-2024 period. OLS will report a "significant" relationship with a high R-squared. But this could be completely meaningless. OLS is just detecting that both series went up. You would get an equally "significant" result if you regressed inflation on the number of registered cars in Lagos (which also went up). That does not mean cars cause inflation.

This is the spurious regression problem. You cannot trust OLS results when both variables are I(1) -- unless you first confirm that the two series are cointegrated.

### The Key Insight: Engle and Granger (1987)

Robert Engle and Clive Granger made a discovery that earned them the 2003 Nobel Prize in Economics. They showed that even though two I(1) series individually wander without a fixed mean, there might exist a **linear combination** of them that IS stationary.

What does that mean in practice? Suppose inflation (infl) and the Monetary Policy Rate (mpr) are both I(1). Individually, they wander. But suppose there exists some constant `b` such that:

```
infl - b * mpr = stationary residual
```

If this residual -- the gap between actual inflation and `b * mpr` -- is stationary (meaning it fluctuates around a fixed mean and always reverts back), then infl and mpr are **cointegrated**. They share a long-run equilibrium. They may drift apart in the short run, but economic forces always pull them back together.

### The Drunk and the Dog Analogy

Think of a drunk person walking their dog on a leash at night. The drunk staggers randomly -- left, right, forward, back. The drunk's path is non-stationary (a random walk). The dog also wanders -- sniffing lampposts, chasing shadows. The dog's path is also non-stationary.

But here is the crucial point: the drunk and the dog are connected by a leash. They can never get too far apart. If the dog runs too far ahead, the leash yanks it back. If the drunk staggers sideways, the leash drags the dog along.

The **distance between the drunk and the dog** is stationary. It fluctuates -- sometimes the dog is ahead, sometimes behind -- but it always reverts to roughly the length of the leash. It has a fixed mean.

The drunk and the dog are cointegrated. Each individual path is non-stationary, but they share a long-run equilibrium defined by the leash. Deviations from that equilibrium are temporary.

Now remove the leash. The drunk and the dog wander independently. Sometimes they happen to be close, sometimes far apart, but there is no force pulling them back together. The distance between them can drift forever without reverting. Without the leash, they are NOT cointegrated.

### Nigerian Example: MPR and TBR

The best Nigerian example of cointegration is the relationship between the Monetary Policy Rate (mpr) and the Treasury Bill Rate (tbr).

The MPR is the benchmark rate set by the CBN's Monetary Policy Committee. The TBR is the yield on Treasury Bills at auction. In theory, when the CBN raises the MPR, Treasury Bill rates should also rise because:

1. The MPR signals the CBN's desired interest rate level for the economy.
2. Banks use the MPR as a reference point when bidding at T-bill auctions.
3. If the MPR goes up but T-bill rates stay low, banks would prefer to deposit money at the CBN (at the MPR-linked Standing Deposit Facility) rather than buy T-bills. This reduces demand for T-bills, pushing their yields up until they align with the new MPR level.

So the MPR and TBR are connected by an invisible leash -- the CBN's monetary policy transmission mechanism. Each rate can wander on its own (both are I(1)), but the **spread** between them stays bounded. When the spread gets too wide, market forces pull it back.

If you run the Engle-Granger test and find that mpr and tbr are cointegrated, it confirms that the CBN's policy rate transmission is working -- the MPR genuinely anchors T-bill rates in the long run. If they are NOT cointegrated, it suggests transmission is broken, perhaps because of excess liquidity in the banking system, government borrowing that distorts T-bill pricing, or other structural problems.

### If Cointegrated: What It Means

If two variables are cointegrated, three things follow:

1. **The relationship is genuine, not spurious.** OLS in levels gives valid -- in fact, "super-consistent" -- estimates of the long-run relationship.
2. **Deviations are temporary.** When the variables drift apart from their equilibrium, error correction forces pull them back. This is the basis for the Error Correction Model (ECM) you will build later.
3. **You can model in levels.** You do not have to throw away information by differencing. The ARDL bounds test and VECM frameworks both exploit cointegration to capture long-run AND short-run dynamics.

### If NOT Cointegrated: What It Means

If two variables are NOT cointegrated, the apparent relationship between them may be spurious. You should NOT model them together in levels using OLS. Instead, you must either:

- Work with first differences (model the changes, not the levels), or
- Use a framework like ARDL bounds testing that does not require pre-testing for cointegration.

### The Engle-Granger Two-Step Procedure

The Engle-Granger test is the simplest cointegration test. It works in two steps:

**Step 1 -- Regress Y on X using OLS in levels.** For example, regress infl on mpr. This gives you a fitted line and a set of residuals (the differences between actual infl and what the OLS line predicts).

**Step 2 -- Test the residuals for stationarity using the ADF test.** If the residuals are stationary (no unit root), the gap between the two series reverts to a mean. The series are cointegrated. If the residuals are non-stationary, the gap can drift forever. The series are NOT cointegrated.

The `statsmodels` library wraps both steps into a single function called `coint()`. You pass it two series, and it returns a test statistic, a p-value, and critical values.

**Hypotheses:**

- **Null hypothesis (H0):** The two series are NOT cointegrated (no long-run equilibrium)
- **Alternative hypothesis (H1):** The two series ARE cointegrated (a long-run equilibrium exists)

**Decision rule:**

- If p-value < 0.05: **reject H0** -- the series are cointegrated at the 5% significance level
- If p-value >= 0.05: **fail to reject H0** -- no evidence of cointegration

Note: the critical values for the Engle-Granger test are NOT the same as regular ADF critical values. They are more stringent (harder to reject) because you are testing residuals from a regression, not a raw series. The `coint()` function uses the correct critical values automatically.

---

## Building `econometric_models/cointegration.py` -- Step by Step

We will build this file in 3 steps. At each step, we show you the **complete file** from the first line to the last line. Delete everything and replace it with exactly what is shown. No guessing where code goes.

Make sure you have the `econometric_models/` folder with an `__init__.py` inside it (from Day 6). If not, create them now.

---

### Build Step 1: Imports, Data Loading, and Basic Structure

Create a new file called `econometric_models/cointegration.py`. Delete everything in `econometric_models/cointegration.py` and replace it with this:

```python
"""Cointegration testing for the Nigerian Inflation Predictor."""
import os
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    """Load the cleaned dataset from data/processed/cleaned_data.csv."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


if __name__ == "__main__":
    df = load_data()
    print(f"Data: {df.shape[0]} observations, variables: {list(df.columns)}")
    print("All variables confirmed I(1) from Day 7.")
    print("Now testing pairwise for long-run equilibrium relationships.\n")
```

**Run it:**

```bash
python -m econometric_models.cointegration
```

**What you should see:**

```
Data: 300 observations, variables: ['mpr', 'infl', 'exo', 'tbr']
All variables confirmed I(1) from Day 7.
Now testing pairwise for long-run equilibrium relationships.
```

(Your exact row count may differ depending on your data.)

**What just happened -- line by line:**

- `"""Cointegration testing..."""` -- A module-level docstring. It describes what this file does. Good practice for every Python file.
- `from itertools import combinations` -- This imports a function from Python's standard library that generates all unique pairs from a list. We will use it in Step 2 to generate all 6 pairs from our 4 variables.
- `from statsmodels.tsa.stattools import coint` -- This imports the Engle-Granger cointegration test. It is in the same `stattools` module as the `adfuller` function you used on Days 6-7. The `coint` function wraps the entire two-step procedure (OLS regression + ADF on residuals) into a single call.
- `PROCESSED_DIR` and `RESULTS_DIR` -- Paths relative to where the script lives. `os.path.dirname(__file__)` means "the folder this script is in" (which is `econometric_models/`). The `".."` goes up one level to the project root, then into the target subfolder.
- `load_data()` -- Reads the cleaned CSV from Day 4. `index_col="date"` makes the date column the row index. `parse_dates=True` converts it to datetime objects.
- The `if __name__ == "__main__":` block loads the data and prints a confirmation so you know the data is loading correctly before you add anything else.

If that ran and printed your data summary, you are ready for Step 2.

---

### Build Step 2: Add the Engle-Granger Test Function and Test All 6 Pairs

Now we add the core testing function and loop through every pairwise combination. Delete everything in `econometric_models/cointegration.py` and replace it with this:

```python
"""Cointegration testing for the Nigerian Inflation Predictor."""
import os
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    """Load the cleaned dataset from data/processed/cleaned_data.csv."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def engle_granger_test(y, x, y_name, x_name, significance=0.05):
    """
    Run the Engle-Granger two-step cointegration test on two series.

    Parameters
    ----------
    y : pd.Series
        The dependent variable (left-hand side of the cointegrating regression).
    x : pd.Series
        The independent variable (right-hand side of the cointegrating regression).
    y_name : str
        Display name for y.
    x_name : str
        Display name for x.
    significance : float
        Significance level for the test (default 0.05).

    Returns
    -------
    dict
        Dictionary with test_statistic, p_value, crit_1pct, crit_5pct,
        crit_10pct, and cointegrated (bool).

    The null hypothesis is: the two series are NOT cointegrated.
    If p_value < significance, reject the null -> they ARE cointegrated.
    """
    stat, p_value, crit_values = coint(y, x)

    result = {
        "pair": f"{y_name} & {x_name}",
        "test_statistic": round(stat, 4),
        "p_value": round(p_value, 4),
        "crit_1pct": round(crit_values[0], 4),
        "crit_5pct": round(crit_values[1], 4),
        "crit_10pct": round(crit_values[2], 4),
        "cointegrated": p_value < significance,
    }

    print(f"\n{'='*55}")
    print(f"Engle-Granger Test: {y_name} & {x_name}")
    print(f"{'='*55}")
    print(f"  Test Statistic:  {stat:.4f}")
    print(f"  P-Value:         {p_value:.4f}")
    print(f"  Critical Values: 1%={crit_values[0]:.4f}, "
          f"5%={crit_values[1]:.4f}, "
          f"10%={crit_values[2]:.4f}")

    if p_value < significance:
        print(f"  Conclusion: COINTEGRATED (reject H0 at {int(significance*100)}%)")
    else:
        print(f"  Conclusion: NOT cointegrated (fail to reject H0)")

    return result


if __name__ == "__main__":
    df = load_data()
    variables = ["mpr", "infl", "exo", "tbr"]

    print("=" * 55)
    print("ENGLE-GRANGER PAIRWISE COINTEGRATION TESTS")
    print(f"Variables: {', '.join(variables)}")
    print(f"All confirmed I(1) from Day 7.")
    print("=" * 55)

    results = []
    for var1, var2 in combinations(variables, 2):
        r = engle_granger_test(df[var1], df[var2], var1, var2)
        results.append(r)

    print(f"\nTested {len(results)} pairs.")
```

**Run it:**

```bash
python -m econometric_models.cointegration
```

**What you should see:**

Six test outputs, one for each pair:

```
=======================================================
ENGLE-GRANGER PAIRWISE COINTEGRATION TESTS
Variables: mpr, infl, exo, tbr
All confirmed I(1) from Day 7.
=======================================================

=======================================================
Engle-Granger Test: mpr & infl
=======================================================
  Test Statistic:  -X.XXXX
  P-Value:         0.XXXX
  Critical Values: 1%=-X.XXXX, 5%=-X.XXXX, 10%=-X.XXXX
  Conclusion: COINTEGRATED (reject H0 at 5%)

=======================================================
Engle-Granger Test: mpr & exo
=======================================================
  ...

=======================================================
Engle-Granger Test: mpr & tbr
=======================================================
  ...

=======================================================
Engle-Granger Test: infl & exo
=======================================================
  ...

=======================================================
Engle-Granger Test: infl & tbr
=======================================================
  ...

=======================================================
Engle-Granger Test: exo & tbr
=======================================================
  ...

Tested 6 pairs.
```

(Your exact numbers and conclusions will depend on your data.)

**What just happened -- line by line:**

- `engle_granger_test(y, x, y_name, x_name, significance=0.05)` -- The function takes two series (y and x), their display names, and a significance level. It calls `coint(y, x)` which performs the full Engle-Granger procedure internally: regress y on x with OLS, extract residuals, run ADF on residuals with adjusted critical values. It returns three things: the ADF test statistic on the residuals, the p-value, and an array of critical values at 1%, 5%, and 10%.
- `round(stat, 4)` -- We round to 4 decimal places for clean output and CSV storage.
- `"cointegrated": p_value < significance` -- A boolean: `True` if the p-value is below our threshold, `False` otherwise. The default threshold is 0.05 (5%).
- `combinations(variables, 2)` -- Given `["mpr", "infl", "exo", "tbr"]`, this generates all unique pairs without repetition: (mpr, infl), (mpr, exo), (mpr, tbr), (infl, exo), (infl, tbr), (exo, tbr). That is C(4,2) = 4!/(2!*2!) = 6 pairs.
- The loop runs the test on each pair and collects results into a list.

**Why 6 pairs?** With 4 variables, the number of unique unordered pairs is 6:

| Pair # | Variable 1 | Variable 2 | Economic Question |
|--------|-----------|-----------|-------------------|
| 1 | mpr | infl | Does monetary policy anchor inflation long-run? |
| 2 | mpr | exo | Does the policy rate anchor the exchange rate? |
| 3 | mpr | tbr | Does the policy rate transmit to T-bill yields? |
| 4 | infl | exo | Does exchange rate pass-through create a long-run link? |
| 5 | infl | tbr | Do inflation expectations drive T-bill pricing? |
| 6 | exo | tbr | Are currency risk and short-term yields linked? |

Good. The function works for all 6 pairs. Now let us add the summary and export.

---

### Build Step 3: Summary Table, CSV Export, and Formatted Output (Final Version)

This is the final, complete version of the file. It adds a summary DataFrame, prints a formatted table, and saves results to `results/cointegration_engle_granger.csv`. Delete everything in `econometric_models/cointegration.py` and replace it with this:

```python
"""Cointegration testing for the Nigerian Inflation Predictor."""
import os
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    """Load the cleaned dataset from data/processed/cleaned_data.csv."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def engle_granger_test(y, x, y_name, x_name, significance=0.05):
    """
    Run the Engle-Granger two-step cointegration test on two series.

    Parameters
    ----------
    y : pd.Series
        The dependent variable (left-hand side of the cointegrating regression).
    x : pd.Series
        The independent variable (right-hand side of the cointegrating regression).
    y_name : str
        Display name for y.
    x_name : str
        Display name for x.
    significance : float
        Significance level for the test (default 0.05).

    Returns
    -------
    dict
        Dictionary with test_statistic, p_value, crit_1pct, crit_5pct,
        crit_10pct, and cointegrated (bool).

    The null hypothesis is: the two series are NOT cointegrated.
    If p_value < significance, reject the null -> they ARE cointegrated.
    """
    stat, p_value, crit_values = coint(y, x)

    result = {
        "pair": f"{y_name} & {x_name}",
        "test_statistic": round(stat, 4),
        "p_value": round(p_value, 4),
        "crit_1pct": round(crit_values[0], 4),
        "crit_5pct": round(crit_values[1], 4),
        "crit_10pct": round(crit_values[2], 4),
        "cointegrated": p_value < significance,
    }

    print(f"\n{'='*55}")
    print(f"Engle-Granger Test: {y_name} & {x_name}")
    print(f"{'='*55}")
    print(f"  Test Statistic:  {stat:.4f}")
    print(f"  P-Value:         {p_value:.4f}")
    print(f"  Critical Values: 1%={crit_values[0]:.4f}, "
          f"5%={crit_values[1]:.4f}, "
          f"10%={crit_values[2]:.4f}")

    if p_value < significance:
        print(f"  Conclusion: COINTEGRATED (reject H0 at {int(significance*100)}%)")
    else:
        print(f"  Conclusion: NOT cointegrated (fail to reject H0)")

    return result


if __name__ == "__main__":
    df = load_data()
    variables = ["mpr", "infl", "exo", "tbr"]

    print("=" * 60)
    print("ENGLE-GRANGER PAIRWISE COINTEGRATION TESTS")
    print(f"Variables: {', '.join(variables)}")
    print("All confirmed I(1) from Day 7.")
    print("=" * 60)

    # --- Test all 6 pairwise combinations ---
    results = []
    for var1, var2 in combinations(variables, 2):
        r = engle_granger_test(df[var1], df[var2], var1, var2)
        results.append(r)

    # --- Build summary DataFrame ---
    results_df = pd.DataFrame(results)

    # --- Print formatted summary table ---
    print("\n\n" + "=" * 60)
    print("SUMMARY: PAIRWISE ENGLE-GRANGER COINTEGRATION")
    print("=" * 60)
    print(f"\n{'Pair':<18} {'Test Stat':<12} {'P-Value':<10} {'Cointegrated?':<15}")
    print("-" * 55)
    for _, row in results_df.iterrows():
        verdict = "Yes" if row["cointegrated"] else "No"
        print(f"{row['pair']:<18} {row['test_statistic']:<12} {row['p_value']:<10} {verdict:<15}")
    print("-" * 55)

    cointegrated_count = results_df["cointegrated"].sum()
    print(f"\nCointegrated pairs: {cointegrated_count} out of {len(results_df)}")

    if cointegrated_count > 0:
        print("\nPairs with cointegration (long-run equilibrium exists):")
        for _, row in results_df[results_df["cointegrated"]].iterrows():
            print(f"  {row['pair']} (p={row['p_value']:.4f})")

    if cointegrated_count < len(results_df):
        print("\nPairs without cointegration:")
        for _, row in results_df[~results_df["cointegrated"]].iterrows():
            print(f"  {row['pair']} (p={row['p_value']:.4f})")

    # --- Save to CSV ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "cointegration_engle_granger.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: results/cointegration_engle_granger.csv")
```

Build Step 3 above is your final complete file.

**Run it:**

```bash
python -m econometric_models.cointegration
```

**What you should see:**

After all six individual test outputs, you will see a clean summary table:

```
============================================================
SUMMARY: PAIRWISE ENGLE-GRANGER COINTEGRATION
============================================================

Pair               Test Stat    P-Value    Cointegrated?
-------------------------------------------------------
mpr & infl         -X.XXXX      0.XXXX     Yes
mpr & exo          -X.XXXX      0.XXXX     No
mpr & tbr          -X.XXXX      0.XXXX     Yes
infl & exo         -X.XXXX      0.XXXX     Yes
infl & tbr         -X.XXXX      0.XXXX     No
exo & tbr          -X.XXXX      0.XXXX     No
-------------------------------------------------------

Cointegrated pairs: 3 out of 6

Pairs with cointegration (long-run equilibrium exists):
  mpr & infl (p=0.XXXX)
  mpr & tbr (p=0.XXXX)
  infl & exo (p=0.XXXX)

Pairs without cointegration:
  mpr & exo (p=0.XXXX)
  infl & tbr (p=0.XXXX)
  exo & tbr (p=0.XXXX)

Results saved to: results/cointegration_engle_granger.csv
```

(The `X.XXXX` values will be actual numbers when you run it. The Yes/No verdicts above are illustrative -- your actual results depend on your data.)

**What changed from Step 2:**

- `results_df = pd.DataFrame(results)` -- Converts the list of dictionaries into a DataFrame. Each dictionary (one per pair) becomes a row. The keys become column names. This gives us a clean tabular structure for printing and saving.
- The formatted summary table uses f-string alignment (`:<18`, `:<12`, `:<10`) to create neat columns. The `<` means left-align, the number is the field width.
- `results_df["cointegrated"].sum()` -- Since `True` equals 1 and `False` equals 0, summing a boolean column counts the number of `True` values.
- `results_df[results_df["cointegrated"]]` -- Boolean indexing: this selects only the rows where `cointegrated` is `True`. The tilde `~` inverts the boolean, selecting the rows where it is `False`.
- `os.makedirs(RESULTS_DIR, exist_ok=True)` -- Creates the results directory if it does not already exist. The `exist_ok=True` prevents an error if the directory is already there.
- `results_df.to_csv(output_path, index=False)` -- Saves the DataFrame to CSV. `index=False` means do not write the row numbers as a column.

---

## Interpreting Your Results -- Nigeria Specific

Your results will fall into different patterns depending on your data. Here is how to interpret the most important pairs.

### MPR and TBR: Likely Cointegrated

This is the pair most likely to show cointegration. The CBN's Monetary Policy Rate is the anchor for the entire short-term interest rate structure in Nigeria. When the MPC raises the MPR, the Standing Lending Facility rate rises automatically (MPR + 1 percentage point), the Standing Deposit Facility rate rises (MPR - 1 percentage point), and commercial banks adjust their T-bill bidding accordingly.

The transmission is direct and institutional. The CBN does not just hope that T-bill rates follow the MPR -- the entire framework is designed so they must. The spread between MPR and TBR may widen during periods of excess liquidity (when banks are flush with cash and T-bill demand is high, pushing yields down) or during periods of fiscal dominance (when the government borrows heavily, pushing yields up). But these deviations are temporary. The spread always reverts because the MPR is the price of last resort for bank liquidity.

If your test confirms cointegration, it validates the CBN's interest rate corridor system. The "leash" between MPR and TBR is the institutional structure of monetary policy itself.

### EXO and INFL: May Be Cointegrated

Exchange rate pass-through is a structural feature of the Nigerian economy. Nigeria imports a large share of what it consumes -- refined petroleum products, food (especially wheat, rice, sugar), machinery, chemicals, and consumer goods. When the Naira depreciates (exo goes up), import prices rise in Naira terms. Those higher import costs feed through to consumer prices (infl goes up).

If exo and infl are cointegrated, it means pass-through is not just a short-term phenomenon. It means the exchange rate and the price level share a permanent, structural long-run equilibrium. A permanently weaker Naira means a permanently higher price level. The 2023 exchange rate unification -- where the official rate moved from around 460 to over 750 NGN/USD -- would predict a permanent increase in the inflation level, not just a temporary spike.

If they are NOT cointegrated, it may be because the exchange rate regime changes (the CBN has switched between fixed, managed float, and multiple exchange rate systems several times) break the long-run equilibrium. The relationship exists within each regime but not across regimes.

### Other Pairs: Depends on the Data

**MPR and INFL:** If cointegrated, the CBN's monetary policy has been systematic enough to maintain a long-run link with inflation -- the CBN raises rates when inflation rises and cuts when it falls, consistently enough to create a stable equilibrium. If not cointegrated, it suggests the CBN's inflation response has been too inconsistent (delayed tightening, political interference, conflicting objectives) to create a stable long-run equilibrium. Both outcomes are valid thesis findings.

**MPR and EXO:** In theory, higher interest rates should support the Naira by attracting capital inflows. In practice, Nigeria's capital controls and the dominance of oil revenues in determining the exchange rate may weaken this link.

**INFL and TBR:** The Fisher equation predicts that nominal interest rates (like T-bill rates) should reflect expected inflation. If cointegrated, it means the T-bill market prices in inflation expectations systematically over the long run.

**EXO and TBR:** This tests whether currency depreciation risk is priced into short-term government borrowing costs. A weaker Naira should raise yields because investors demand compensation for currency risk.

### What If Few or No Pairs Are Cointegrated?

Do not panic. This is a perfectly valid finding, and it does NOT mean your project has failed. Remember two things:

1. **The Engle-Granger test only checks pairwise cointegration.** It tests two variables at a time. But in a system of four variables, there may be a cointegrating relationship that involves THREE or FOUR variables simultaneously. For example, there might be no pairwise cointegration between mpr and infl alone, but a linear combination of ALL FOUR variables might be stationary. The Engle-Granger test simply cannot see this.

2. **Tomorrow you will run the Johansen test (Day 9).** The Johansen test examines all four variables simultaneously and can detect MULTIPLE cointegrating vectors. It is strictly more powerful than the pairwise approach for multivariate systems. So even if you find zero cointegrating pairs today, the Johansen test may still find cointegration in the full four-variable system.

3. **The ARDL bounds test (Week 3) does not require pre-testing for cointegration.** One of the key advantages of the ARDL framework (Pesaran, Shin, and Smith, 2001) is that it tests for cointegration as part of the model estimation itself. It works regardless of whether variables are I(0) or I(1), and it does not require you to establish cointegration beforehand. So even if both Engle-Granger and Johansen find nothing, the ARDL approach remains valid.

### What If All Pairs Are Cointegrated?

This is also a valid finding for Nigerian macroeconomic data. If all four variables are I(1) and all driven by the same underlying forces -- monetary expansion, Naira depreciation, CBN policy responses, fiscal dominance -- then multiple pairwise equilibria are plausible. This would suggest a tightly interconnected system where every variable is linked to every other variable in the long run, consistent with the theory that MPR, inflation, exchange rate, and Treasury Bill rates are all part of the same monetary transmission mechanism.

---

## Commit

```bash
git add econometric_models/cointegration.py results/cointegration_engle_granger.csv
git commit -m "Day 8: Engle-Granger pairwise cointegration tests for all 6 pairs"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: cleaned_data.csv not found` | You need to run the cleaning script first: `python -m data_processing.clean`. The cointegration script reads from `data/processed/cleaned_data.csv`, which is created by the Day 4 cleaning step. |
| `ModuleNotFoundError: No module named 'statsmodels'` | You need statsmodels installed. Run `pip install statsmodels==0.14.1` (it should already be installed from Day 6). Make sure your virtual environment is activated. |
| `ImportError: cannot import name 'coint'` | Your version of statsmodels may be too old. Run `pip install --upgrade statsmodels`. The `coint` function has been available since statsmodels 0.8. |
| `KeyError: 'mpr'` or `KeyError: 'infl'` | Your cleaned CSV does not have the expected column names. Open `data/processed/cleaned_data.csv` and check the header row. The columns must be exactly: `date`, `mpr`, `infl`, `exo`, `tbr`. If the names differ, update the `variables` list in the script to match your actual column names. |
| `ValueError: x and y must have the same length` | One of your series has missing values that created a length mismatch. Go back to Day 4's cleaning script and ensure all NaN values are handled. The `coint` function requires both series to have the same number of observations with no NaN values. |
| `LinAlgError: Singular matrix` | This happens when one of the series has zero variance (all values are the same) or the two series are perfectly collinear. Check your data with `df.describe()` to make sure all columns have variation. |

---

## Check Your Understanding

These are questions an examiner will ask. Practice answering them out loud.

### 1. "What is cointegration in plain English?"

**Answer:** "Cointegration means that two time series that individually wander -- they are non-stationary, they do not revert to a fixed mean -- are nonetheless connected to each other in the long run. The gap between them is stationary. It fluctuates, but it always reverts back to a mean. They may diverge in the short run, but economic forces always pull them back together.

The analogy I use is a drunk person walking their dog on a leash. Both the drunk and the dog follow random, unpredictable paths -- both are non-stationary. But the distance between them is bounded by the leash and always reverts to a manageable range. That distance is stationary. The drunk and the dog are cointegrated.

In economic terms, cointegration means two variables share a long-run equilibrium. For Nigeria, the MPR and Treasury Bill Rate are a natural example. Both rates wander over time, but the CBN's monetary policy framework acts as a leash that keeps the spread between them bounded."

### 2. "Why did you test all 6 pairs instead of just the pairs you care about?"

**Answer:** "With four variables, there are C(4,2) = 6 unique pairwise combinations. I tested all of them for two reasons.

First, completeness. Each pair represents a different economic hypothesis. The mpr-tbr pair tests monetary policy transmission. The exo-infl pair tests exchange rate pass-through. The mpr-infl pair tests whether the CBN's rate-setting behaviour has a stable long-run link to inflation. Testing all pairs gives a complete picture of which bilateral long-run relationships exist in the data.

Second, the Engle-Granger test is a pairwise procedure by design. It can only examine two variables at a time. By testing all 6 pairs, I extract the maximum information possible from this method before moving to the Johansen test, which examines all four variables simultaneously. Comparing the pairwise results with the Johansen results helps validate the findings."

### 3. "The Engle-Granger test found no cointegration for a pair you expected to be cointegrated. What could explain this?"

**Answer:** "Several factors could prevent the Engle-Granger test from detecting cointegration even when a long-run relationship exists in theory.

First, the Engle-Granger test has lower statistical power than multivariate methods like the Johansen test. With a limited sample size, it may fail to reject the null of no cointegration even when the relationship is real. This is a Type II error.

Second, the Engle-Granger test is sensitive to which variable is placed on the left side of the regression. Regressing infl on mpr can give a different result than regressing mpr on infl. With only two variables, the test has no way to determine the correct normalization.

Third, structural breaks in the data can disrupt the equilibrium relationship. If the relationship between two variables changed fundamentally at some point -- for example, if the CBN's exchange rate regime changed -- the test may not find cointegration over the full sample even though it existed within each sub-period.

Fourth, the relationship may genuinely require more than two variables. The monetary transmission mechanism involves mpr, infl, exo, and tbr simultaneously. A pairwise test might miss a cointegrating relationship that only appears when all four are considered together. This is exactly what the Johansen test on Day 9 is designed to detect."

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Cointegration test script | `econometric_models/cointegration.py` | Runs Engle-Granger pairwise tests on all 6 variable pairs |
| Test function | `engle_granger_test()` | Reusable function that tests any two series and returns a dict |
| Summary table | Printed to terminal | Shows test stats, p-values, and verdicts at a glance |
| Results CSV | `results/cointegration_engle_granger.csv` | Machine-readable record of all pairwise cointegration results |

**Key finding:** You now know which pairs of your four Nigerian macroeconomic variables share a long-run equilibrium and which do not. The MPR-TBR pair is likely cointegrated (policy rate transmission). The EXO-INFL pair may be cointegrated (exchange rate pass-through). Other pairs depend on your specific data.

**No new packages installed today.** The `coint` function comes from `statsmodels`, which you installed on Day 6.

**Tomorrow (Day 9):** You will run the Johansen cointegration test, which examines all four variables simultaneously and determines how many cointegrating relationships exist in the full system. The Johansen test is more powerful than today's pairwise approach and can detect multivariate equilibria that the Engle-Granger test cannot see. Its result -- the cointegration rank -- is one of the most important numbers in your entire thesis.
