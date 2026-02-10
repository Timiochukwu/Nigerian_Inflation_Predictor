# Week 3, Day 11 — ARDL Model: Theory & Lag Selection

## What You'll Learn Today

- What ARDL is and why it is ideal for this project
- ARDL notation: ARDL(p, q1, q2, q3) where p = lags of infl, q1 = lags of mpr,
  q2 = lags of tbr, q3 = lags of exo
- How to use the statsmodels `ardl_select_order()` function to find optimal lags
- Why AIC and BIC may disagree, and which one to trust for forecasting

---

## Theory: What Is ARDL?

ARDL stands for **Autoregressive Distributed Lag**. If you have never seen this
model before, here is a plain-English breakdown of the two words in that name.

### Autoregressive

"Auto" means self. "Regressive" means regression. So autoregressive means
**a variable is regressed on its own past values**. In our project, the
dependent variable is `infl` (Nigerian headline inflation). When we say the
model is autoregressive, we mean that current inflation depends partly on
what inflation was last month, two months ago, three months ago, and so on.

Why would inflation depend on its own past? Because of inertia. Prices do not
jump randomly each month. If inflation was 18 percent last month, it is very
likely to be somewhere near 18 percent this month too. Past inflation carries
forward through contracts, expectations, and sticky prices.

### Distributed Lag

"Distributed lag" means the model also includes the **current and past values
of other variables** as predictors. In our project those other variables are
`mpr` (Monetary Policy Rate), `tbr` (Treasury Bill Rate), and `exo` (exchange
rate proxy). The word "distributed" tells you that the effect of a change in,
say, `mpr` does not hit inflation all at once. Instead, the effect is
**distributed across several time periods** — it trickles in over months.

### Put Them Together

ARDL is a regression where `infl` depends on its own past values AND on
the current and past values of `mpr`, `tbr`, and `exo`. Written out in
equation form:

```
infl_t = c
       + a1 * infl_(t-1) + a2 * infl_(t-2) + ... + ap * infl_(t-p)
       + b0 * mpr_t  + b1 * mpr_(t-1)  + ... + bq1 * mpr_(t-q1)
       + g0 * tbr_t  + g1 * tbr_(t-1)  + ... + gq2 * tbr_(t-q2)
       + d0 * exo_t  + d1 * exo_(t-1)  + ... + dq3 * exo_(t-q3)
       + error_t
```

Where:

- `c` is a constant (intercept).
- `a1 ... ap` are coefficients on lagged inflation (the autoregressive part).
- `b0 ... bq1` are coefficients on current and lagged `mpr`.
- `g0 ... gq2` are coefficients on current and lagged `tbr`.
- `d0 ... dq3` are coefficients on current and lagged `exo`.
- `error_t` is the residual — what the model cannot explain.

### ARDL(p, q1, q2, q3) Notation

We write the model as **ARDL(p, q1, q2, q3)** where:

| Symbol | Meaning                            |
|--------|------------------------------------|
| p      | Number of lags of `infl`           |
| q1     | Number of lags of `mpr`            |
| q2     | Number of lags of `tbr`            |
| q3     | Number of lags of `exo`            |

For example, ARDL(3, 2, 1, 4) means:

- 3 lags of `infl` (inflation from 1, 2, and 3 months ago)
- 2 lags of `mpr` (current mpr plus mpr from 1 and 2 months ago)
- 1 lag of `tbr` (current tbr plus tbr from 1 month ago)
- 4 lags of `exo` (current exo plus exo from 1, 2, 3, and 4 months ago)

Each variable gets its **own** optimal lag count. This is one of the biggest
advantages over VAR, which forces every variable to share the same lag.

### Why ARDL vs VAR

You built a VAR model earlier, so you already know how vector autoregressions
work. Here is why ARDL is the better tool for this project:

1. **Mixed integration orders.** VAR requires all variables to be the same
   order of integration — typically all I(1), differenced to I(0). ARDL works
   with a MIX of I(0) and I(1) variables. The only restriction is that nothing
   can be I(2). If your ADF or KPSS tests are ambiguous for one variable, ARDL
   still works.

2. **Built-in cointegration test.** ARDL comes with the bounds test (see
   below), which checks for a long-run equilibrium relationship. VAR has no
   equivalent built-in test — you need the separate Johansen procedure.

3. **Clear short-run AND long-run coefficients.** When ARDL is rewritten in
   its error correction form (the UECM), it separates short-run dynamics from
   the long-run equilibrium. You get both sets of numbers from a single model.

4. **Different lags per variable.** VAR forces one lag order on every variable.
   ARDL lets each variable have its own lag, matching the actual speed at which
   each economic mechanism operates.

5. **Better small-sample performance.** The Johansen test relies on asymptotic
   theory. With moderate sample sizes (a few hundred observations), the ARDL
   bounds test has been shown in simulation studies to be more reliable.

### Pesaran, Shin, and Smith (2001) — The Seminal Paper

The theoretical foundation for ARDL cointegration testing comes from a single
paper: Pesaran, Shin, and Smith (2001), "Bounds testing approaches to the
analysis of level relationships." This paper introduced the bounds test that
lets you test for cointegration without first determining whether each variable
is I(0) or I(1). You will cite this paper in your thesis every time you
mention the ARDL bounds test.

### Nigerian Context

Why does ARDL fit Nigeria especially well? Because monetary policy
transmission in Nigeria is slow and uneven.

- When the CBN changes the MPR, commercial banks do not adjust lending rates
  overnight. It takes weeks or months. That transmission flows through `tbr`
  first (the Treasury Bill Rate responds quickly to MPR changes) and then
  through `exo` (the exchange rate adjusts as capital flows respond to interest
  rate differentials).
- The exchange rate channel is sluggish. A Naira depreciation raises import
  costs, but those costs take months to reach consumer prices because of
  supply chain lags, inventory buffers, and administered pricing.
- ARDL captures this **gradual, variable-speed transmission** by allowing
  different lag lengths for `mpr`, `tbr`, and `exo`.

---

## Building `econometric_models/ardl_model.py` — 3 Steps

We build the script incrementally. At every step you see the **complete file**
from the very first line to the very last line. No snippets. No "add this
below the previous code." Each step is a self-contained, runnable program.

Make sure the `econometric_models/` directory exists and contains an
`__init__.py` file (created on a previous day). If not, create both now.

---

### STEP 1: Bare Minimum — Imports, Load Data, Print Summary

Delete everything in `econometric_models/ardl_model.py` and replace it with this:

```python
"""ARDL model for the Nigerian Inflation Predictor."""
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.ardl import ardl_select_order

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

DEPENDENT = "infl"
EXOG_VARS = ["mpr", "tbr", "exo"]


def load_data():
    """Load the cleaned dataset and return only the columns we need."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    df = df[[DEPENDENT] + EXOG_VARS]
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Data: {df.shape[0]} observations")
    print(f"Dependent variable: {DEPENDENT}")
    print(f"Exogenous variables: {EXOG_VARS}")
    print(f"\nFirst 5 rows:")
    print(df.head())
```

**Run it:**

```bash
python -m econometric_models.ardl_model
```

**What you should see:**

```
Data: 300 observations
Dependent variable: infl
Exogenous variables: ['mpr', 'tbr', 'exo']

First 5 rows:
             infl    mpr    tbr     exo
date
2000-01-01  ...     ...    ...     ...
...
```

Your exact row count and numbers will depend on your cleaned data. The
important thing is that the script runs without errors and prints four columns.

**Line-by-line explanation:**

- `DEPENDENT = "infl"` — The variable we are trying to predict. This is the
  left-hand side of the ARDL equation.
- `EXOG_VARS = ["mpr", "tbr", "exo"]` — The right-hand-side variables, in
  the order that maps to q1, q2, q3 in the ARDL(p, q1, q2, q3) notation.
- `load_data()` — Reads `data/processed/cleaned_data.csv`, sets the date
  column as the index, and keeps only the four columns we need. Dropping
  unused columns prevents accidental contamination.
- The `if __name__` block runs only when you execute the script directly. It
  loads the data and prints a quick sanity check so you can confirm everything
  looks right before moving on.

If that ran successfully, move on to Step 2.

---

### STEP 2: Add the `select_ardl_order` Function

Delete everything in `econometric_models/ardl_model.py` and replace it with this:

```python
"""ARDL model for the Nigerian Inflation Predictor."""
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.ardl import ardl_select_order

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

DEPENDENT = "infl"
EXOG_VARS = ["mpr", "tbr", "exo"]


def load_data():
    """Load the cleaned dataset and return only the columns we need."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    df = df[[DEPENDENT] + EXOG_VARS]
    return df


def select_ardl_order(df, maxlag=12, ic="aic"):
    """
    Use information criteria to find optimal ARDL(p, q1, q2, q3) specification.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: infl, mpr, tbr, exo.
    maxlag : int
        Maximum number of lags to search over for every variable.
    ic : str
        Information criterion — "aic" or "bic".

    Returns
    -------
    result : ARDLSelectionResults
        The statsmodels selection result object.
    p : int
        Optimal lag for the dependent variable (infl).
    q_vals : list of int
        Optimal lags for each exogenous variable [mpr, tbr, exo].
    """
    y = df[DEPENDENT]
    X = df[EXOG_VARS]

    print(f"\nSearching for optimal ARDL order (maxlag={maxlag}, criterion={ic})...")
    print("This may take a minute — it tests thousands of lag combinations.\n")

    result = ardl_select_order(y, maxlag, X, maxorder=maxlag, ic=ic, trend="c")

    p = result.ar_lags[-1] if len(result.ar_lags) > 0 else 0
    q_vals = []
    for var in EXOG_VARS:
        lags = result.dl_lags.get(var, [])
        q = max(lags) if len(lags) > 0 else 0
        q_vals.append(q)

    print(f"Optimal specification: ARDL({p}, {', '.join(str(q) for q in q_vals)})")
    print(f"  p  (lags of {DEPENDENT}): {p}")
    for var, q in zip(EXOG_VARS, q_vals):
        print(f"  q  (lags of {var}): {q}")

    return result, p, q_vals


if __name__ == "__main__":
    df = load_data()
    print(f"Data: {df.shape[0]} observations")
    print(f"Dependent variable: {DEPENDENT}")
    print(f"Exogenous variables: {EXOG_VARS}")
    print(f"\nFirst 5 rows:")
    print(df.head())

    # --- AIC selection ---
    print("\n" + "=" * 60)
    print("ARDL LAG SELECTION — AIC")
    print("=" * 60)
    result_aic, p_aic, q_aic = select_ardl_order(df, maxlag=12, ic="aic")

    # --- BIC selection ---
    print("\n" + "=" * 60)
    print("ARDL LAG SELECTION — BIC")
    print("=" * 60)
    result_bic, p_bic, q_bic = select_ardl_order(df, maxlag=12, ic="bic")
```

**Run it:**

```bash
python -m econometric_models.ardl_model
```

**What you should see:**

```
Data: 300 observations
...

============================================================
ARDL LAG SELECTION — AIC
============================================================

Searching for optimal ARDL order (maxlag=12, criterion=aic)...
This may take a minute — it tests thousands of lag combinations.

Optimal specification: ARDL(X, Y, Z, W)
  p  (lags of infl): X
  q  (lags of mpr): Y
  q  (lags of tbr): Z
  q  (lags of exo): W

============================================================
ARDL LAG SELECTION — BIC
============================================================

Searching for optimal ARDL order (maxlag=12, criterion=bic)...
...
Optimal specification: ARDL(A, B, C, D)
  p  (lags of infl): A
  ...
```

X, Y, Z, W and A, B, C, D will be real numbers when you run it.

**Function walkthrough:**

- `y = df[DEPENDENT]` — Extracts the `infl` column as a pandas Series. This
  is the left-hand-side variable.
- `X = df[EXOG_VARS]` — Extracts `mpr`, `tbr`, `exo` as a DataFrame. These
  are the right-hand-side variables.
- `ardl_select_order(y, maxlag, X, maxorder=maxlag, ic=ic, trend="c")` — The
  core statsmodels function. It fits every possible combination of lags from
  0 (or 1 for the dependent) up to `maxlag` for each variable, computes the
  information criterion for each combination, and returns the specification
  with the lowest value.
  - `y` — dependent variable.
  - `maxlag` (positional) — max AR lags to test for `infl`.
  - `X` — exogenous variables.
  - `maxorder=maxlag` — max distributed lags to test for each exogenous var.
  - `ic="aic"` — which information criterion to minimize.
  - `trend="c"` — include a constant. Standard for macroeconomic data.
- `result.ar_lags` — A list of selected AR lag indices, e.g. `[1, 2, 3]`.
  The last element is p.
- `result.dl_lags` — A dictionary mapping each exogenous variable name to
  its selected lag indices. For example, `{"mpr": [0, 1], "tbr": [0, 1, 2],
  "exo": [0]}`. The max of each list gives the q value.

**Note on computation time:** With `maxlag=12` and 3 exogenous variables, the
function evaluates 12 x 13 x 13 x 13 = 26,364 different ARDL specifications.
This can take one to five minutes depending on your hardware. Be patient.

If both AIC and BIC results printed successfully, move on to Step 3.

---

### STEP 3: Add AIC vs BIC Comparison + Save Results

Delete everything in `econometric_models/ardl_model.py` and replace it with this:

```python
"""ARDL model for the Nigerian Inflation Predictor."""
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.ardl import ardl_select_order

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

DEPENDENT = "infl"
EXOG_VARS = ["mpr", "tbr", "exo"]


def load_data():
    """Load the cleaned dataset and return only the columns we need."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    df = df[[DEPENDENT] + EXOG_VARS]
    return df


def select_ardl_order(df, maxlag=12, ic="aic"):
    """
    Use information criteria to find optimal ARDL(p, q1, q2, q3) specification.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: infl, mpr, tbr, exo.
    maxlag : int
        Maximum number of lags to search over for every variable.
    ic : str
        Information criterion — "aic" or "bic".

    Returns
    -------
    result : ARDLSelectionResults
        The statsmodels selection result object.
    p : int
        Optimal lag for the dependent variable (infl).
    q_vals : list of int
        Optimal lags for each exogenous variable [mpr, tbr, exo].
    """
    y = df[DEPENDENT]
    X = df[EXOG_VARS]

    print(f"\nSearching for optimal ARDL order (maxlag={maxlag}, criterion={ic})...")
    print("This may take a minute — it tests thousands of lag combinations.\n")

    result = ardl_select_order(y, maxlag, X, maxorder=maxlag, ic=ic, trend="c")

    p = result.ar_lags[-1] if len(result.ar_lags) > 0 else 0
    q_vals = []
    for var in EXOG_VARS:
        lags = result.dl_lags.get(var, [])
        q = max(lags) if len(lags) > 0 else 0
        q_vals.append(q)

    print(f"Optimal specification: ARDL({p}, {', '.join(str(q) for q in q_vals)})")
    print(f"  p  (lags of {DEPENDENT}): {p}")
    for var, q in zip(EXOG_VARS, q_vals):
        print(f"  q  (lags of {var}): {q}")

    return result, p, q_vals


def compare_and_save(p_aic, q_aic, p_bic, q_bic):
    """
    Compare AIC and BIC lag selections and save both to a CSV file.

    Parameters
    ----------
    p_aic, p_bic : int
        Optimal AR lag under AIC and BIC.
    q_aic, q_bic : list of int
        Optimal distributed lags under AIC and BIC for [mpr, tbr, exo].
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    rows.append({
        "criterion": "AIC",
        "ardl_order": f"ARDL({p_aic}, {', '.join(str(q) for q in q_aic)})",
        "infl_lags": p_aic,
        "mpr_lags": q_aic[0],
        "tbr_lags": q_aic[1],
        "exo_lags": q_aic[2],
    })
    rows.append({
        "criterion": "BIC",
        "ardl_order": f"ARDL({p_bic}, {', '.join(str(q) for q in q_bic)})",
        "infl_lags": p_bic,
        "mpr_lags": q_bic[0],
        "tbr_lags": q_bic[1],
        "exo_lags": q_bic[2],
    })

    results_df = pd.DataFrame(rows)
    outpath = os.path.join(RESULTS_DIR, "ardl_lag_selection.csv")
    results_df.to_csv(outpath, index=False)

    print(f"\nResults saved to {outpath}")
    print(results_df.to_string(index=False))

    # --- Recommendation ---
    total_aic = p_aic + sum(q_aic)
    total_bic = p_bic + sum(q_bic)

    print("\n--- Recommendation ---")
    if total_aic == total_bic:
        print("AIC and BIC agree on the same specification. This is a strong")
        print("signal — use it with confidence.")
    elif total_aic > total_bic:
        print("AIC selected more lags than BIC. This is typical.")
        print("AIC favours richer models (better for forecasting).")
        print("BIC favours parsimonious models (closer to the 'true' model).")
        print("Use the AIC specification as your primary model and report")
        print("the BIC specification as a robustness check in your appendix.")
    else:
        print("Unusually, BIC selected more lags than AIC. Double-check your")
        print("data for irregularities. Proceed with AIC as primary.")


if __name__ == "__main__":
    df = load_data()

    print("=" * 60)
    print("ARDL LAG ORDER SELECTION")
    print(f"Dependent variable : {DEPENDENT}")
    print(f"Exogenous variables: {EXOG_VARS}")
    print(f"Observations       : {df.shape[0]}")
    print("=" * 60)

    # --- AIC selection ---
    print("\n>>> AIC criterion")
    result_aic, p_aic, q_aic = select_ardl_order(df, maxlag=12, ic="aic")

    # --- BIC selection ---
    print("\n>>> BIC criterion")
    result_bic, p_bic, q_bic = select_ardl_order(df, maxlag=12, ic="bic")

    # --- Compare and save ---
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    compare_and_save(p_aic, q_aic, p_bic, q_bic)

    print(f"\nPrimary model: ARDL({p_aic}, {', '.join(str(q) for q in q_aic)})")
    print("Tomorrow we estimate this model and run the bounds test.")
```

**Run it:**

```bash
python -m econometric_models.ardl_model
```

**Expected output (your numbers will differ):**

```
============================================================
ARDL LAG ORDER SELECTION
Dependent variable : infl
Exogenous variables: ['mpr', 'tbr', 'exo']
Observations       : 300
============================================================

>>> AIC criterion

Searching for optimal ARDL order (maxlag=12, criterion=aic)...
This may take a minute — it tests thousands of lag combinations.

Optimal specification: ARDL(3, 2, 1, 4)
  p  (lags of infl): 3
  q  (lags of mpr): 2
  q  (lags of tbr): 1
  q  (lags of exo): 4

>>> BIC criterion

Searching for optimal ARDL order (maxlag=12, criterion=bic)...
This may take a minute — it tests thousands of lag combinations.

Optimal specification: ARDL(2, 1, 1, 2)
  p  (lags of infl): 2
  q  (lags of mpr): 1
  q  (lags of tbr): 1
  q  (lags of exo): 2

============================================================
COMPARISON
============================================================

Results saved to .../results/ardl_lag_selection.csv
criterion              ardl_order  infl_lags  mpr_lags  tbr_lags  exo_lags
      AIC        ARDL(3, 2, 1, 4)          3         2         1         4
      BIC        ARDL(2, 1, 1, 2)          2         1         1         2

--- Recommendation ---
AIC selected more lags than BIC. This is typical.
AIC favours richer models (better for forecasting).
BIC favours parsimonious models (closer to the 'true' model).
Use the AIC specification as your primary model and report
the BIC specification as a robustness check in your appendix.

Primary model: ARDL(3, 2, 1, 4)
Tomorrow we estimate this model and run the bounds test.
```

---

## Interpreting the Results

### What the ARDL(p, q1, q2, q3) Numbers Mean in Practice

Suppose AIC selected ARDL(3, 2, 1, 4). Here is what each number tells you
about how the Nigerian economy works:

**p = 3 (lags of infl):** Current inflation depends on the previous three
months of inflation. This captures price inertia. A high p (3 or 4) means
inflation is persistent — once it rises, it stays elevated for several months
before the effects of monetary policy bring it down. For Nigerian data,
p = 2 to 4 is typical.

**q1 = 2 (lags of mpr):** The Monetary Policy Rate affects inflation with up
to a two-month lag. When the CBN raises the MPR at its bi-monthly MPC meeting,
the effect begins filtering into inflation within one to two months. This is
the direct interest rate channel — higher MPR leads to higher commercial bank
lending rates, which reduces borrowing and aggregate demand.

**q2 = 1 (lags of tbr):** The Treasury Bill Rate transmits to inflation
quickly, within about one month. TBR is a market-determined rate that responds
almost immediately to MPR changes, so its own lag tends to be short. The TBR
captures the cost-of-funds channel through which monetary tightening reduces
money available for lending.

**q3 = 4 (lags of exo):** The exchange rate takes the longest to pass through
to consumer prices — up to four months. This is the exchange rate pass-through
channel. When the Naira depreciates, import prices rise at the ports. Those
higher costs then propagate through wholesalers, distributors, and retailers
before finally showing up in the CPI. Each link in the supply chain adds a
month or so of delay.

### Expected Values for Nigerian Data

Based on the empirical literature on Nigerian monetary policy transmission:

| Variable | Expected lags | Reason                                         |
|----------|---------------|-------------------------------------------------|
| infl (p) | 2 to 4        | Inflation persistence due to expectations       |
| mpr (q1) | 1 to 3        | Policy rate transmission through bank rates     |
| tbr (q2) | 0 to 2        | Market rate responds quickly to policy signals  |
| exo (q3) | 2 to 5        | Slow exchange rate pass-through via supply chain|

If your results fall roughly within these ranges, your model is capturing
plausible economic dynamics. If a lag is outside these ranges, do not panic —
let the data speak. But do mention in your thesis why the lag might be higher
or lower than expected.

### Why AIC and BIC May Disagree

AIC and BIC penalize model complexity differently:

- **AIC** penalty per parameter: 2
- **BIC** penalty per parameter: ln(n), where n is the number of observations

With 300 observations, ln(300) is approximately 5.7. So BIC penalizes each
additional lag almost three times as harshly as AIC does. The result is that
BIC almost always selects the same number of lags or fewer than AIC. It never
selects more (in practice).

For a **forecasting** project like ours, AIC is the primary criterion because
it optimizes predictive accuracy. BIC is the robustness check because it
identifies the most parsimonious model that still captures the essential
dynamics.

---

## Commit

```bash
git add econometric_models/ardl_model.py results/ardl_lag_selection.csv
git commit -m "Day 11: ARDL lag order selection with AIC and BIC comparison"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `KeyError: 'infl'` or `KeyError: 'mpr'` | Your `cleaned_data.csv` does not have the expected column names. Open `data/processed/cleaned_data.csv` and verify the header row contains exactly `infl`, `mpr`, `tbr`, `exo`. Column names are case-sensitive. |
| `FileNotFoundError: cleaned_data.csv` | Run the Day 4 cleaning script first. The ARDL script reads from `data/processed/cleaned_data.csv`, which must exist before you run this. |
| `ValueError: No frequency information` or `ValueWarning: A date index has been provided, but it has no associated frequency information` | statsmodels ARDL functions need a frequency on the DatetimeIndex. Add `df.index.freq = "MS"` after loading the data, or ensure your CSV dates are at regular monthly intervals with no gaps. |
| Script runs for more than five minutes | With `maxlag=12` and three exogenous variables, the function evaluates 26,364 combinations. On a slow machine this can take several minutes. Reduce `maxlag` to 6 (2,058 combinations) if you need faster results, then re-run with 12 overnight. |
| `ConvergenceWarning` during lag search | Some lag combinations cause the optimizer to struggle. This is a warning, not an error. The final selected model is the one that converged and had the best criterion value. You can safely ignore convergence warnings during the search phase. |
| AIC and BIC select identical orders | Not an error. It means both criteria agree, which strengthens your result. Report it positively in your thesis. |

---

## Check Your Understanding

### Q1: "Why did you choose ARDL instead of VAR for this project?"

**Answer:** "I chose ARDL for three reasons. First, ARDL accommodates mixed
integration orders. My unit root tests showed that some variables may be I(0)
and others I(1), and ARDL works regardless of this mix, provided nothing is
I(2). VAR requires all variables to be the same order. Second, ARDL has the
Pesaran, Shin, and Smith (2001) bounds test built in, which lets me test for
cointegration directly within the model rather than relying on a separate
Johansen procedure. Third, ARDL allows different lag lengths for each
variable, which matches the economic reality that MPR, TBR, and exchange rate
transmit to inflation at different speeds."

### Q2: "What does ARDL(3, 2, 1, 4) mean in plain language?"

**Answer:** "It means inflation today is explained by the last three months
of inflation itself, the current and last two months of the Monetary Policy
Rate, the current and last month of the Treasury Bill Rate, and the current
and last four months of the exchange rate. Each variable gets its own number
of lags based on how quickly it affects inflation. The exchange rate has the
most lags because exchange rate pass-through to consumer prices is slow — it
takes up to four months for a Naira depreciation to fully show up in the
Consumer Price Index."

### Q3: "Why might AIC and BIC give different lag orders?"

**Answer:** "AIC and BIC use different penalty functions for model complexity.
AIC penalizes each additional parameter by 2, while BIC penalizes by ln(n),
which is about 5.7 for 300 observations. Because BIC is stricter, it selects
fewer lags — it prefers a simpler model that captures only the strongest
dynamics. AIC is more generous with lags because it optimizes for predictive
accuracy rather than parsimony. In applied forecasting work, we use the AIC
specification as our primary model and report the BIC specification as a
robustness check in the appendix."

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| ARDL lag selection script | `econometric_models/ardl_model.py` | Finds optimal ARDL(p, q1, q2, q3) using AIC and BIC |
| Lag selection results | `results/ardl_lag_selection.csv` | Stores the selected lags for both criteria |

No new packages were installed. The `ardl_select_order` function is part of
`statsmodels`, which you installed earlier.

---

**Tomorrow (Day 12):** You will estimate the actual ARDL model using the lag
orders selected today, run the Pesaran bounds test for cointegration, and
extract both short-run and long-run coefficients. Day 12 is where you finally
get numbers that tell you: "A one-percentage-point increase in the MPR reduces
Nigerian inflation by X percent in the short run and Y percent in the long
run."
