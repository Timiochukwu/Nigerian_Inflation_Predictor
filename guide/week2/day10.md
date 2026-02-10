# Week 2, Day 10 -- Lag Selection: How Many Lags to Include

## What You'll Learn Today

- Information criteria: AIC, BIC, HQIC, FPE
- How to use `VAR.select_order()` for optimal lag determination
- How to interpret and choose among conflicting criteria

## Theory: Why Lag Selection Matters

Every VAR or ARDL model needs a lag order -- the number of past time periods each variable uses to predict the present. This number is called **p**. Getting it right is critical.

### Too few lags: the model misses dynamics

Nigerian monetary policy takes **6 to 18 months** to transmit through the economy. When the CBN raises the MPR, the effect passes through commercial bank lending rates, then credit volumes, then aggregate demand, then finally consumer prices. If your model only looks back 1 month, it misses this entire chain. The result: autocorrelated residuals, biased standard errors, and impulse response functions that understate the true policy effect.

### Too many lags: overfitting and lost degrees of freedom

With 4 variables and p lags, each VAR equation has 4 x p coefficients plus a constant. At p=12, that is 49 parameters per equation and 196 across the system. With roughly 300 observations (299 after differencing), you burn through degrees of freedom fast. Parameter estimates become imprecise, confidence intervals balloon, and out-of-sample forecasts deteriorate.

### Information criteria: the formal solution

Information criteria balance **goodness of fit** against **model complexity**. You fit the VAR at every lag from 1 to some maximum, compute the criterion at each, and pick the lag that **minimizes** it. Lower value = better model.

**AIC (Akaike Information Criterion)** -- Penalizes complexity lightly. Tends to select more lags. Better for forecasting.

**BIC (Bayesian Information Criterion)** -- Penalizes complexity heavily (penalty grows with sample size). Tends to select fewer lags. Statistically consistent -- converges to the true model.

**HQIC (Hannan-Quinn Information Criterion)** -- A compromise between AIC and BIC. Also consistent, converges more slowly.

**FPE (Final Prediction Error)** -- Similar to AIC. Designed for prediction. Tends to agree with AIC.

The typical pattern: AIC suggests 4-8 lags, BIC suggests 1-3 lags, HQIC falls in between. When they disagree, we use a majority-vote approach with BIC as the tiebreaker, because BIC's parsimony avoids overfitting.

---

## No New Packages

Everything you need is already in `statsmodels` from Day 6. The `VAR` model class has a built-in `select_order()` method that computes all four criteria.

---

## Building `econometric_models/lag_selection.py` -- Step by Step

Each step shows the **complete file**. Delete everything and replace with exactly what is shown.

---

### Build Step 1: Imports, Load Data, Basic Print

Delete everything in `econometric_models/lag_selection.py` and replace it with this:

```python
"""Lag selection for the Nigerian Inflation Predictor."""
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


if __name__ == "__main__":
    df = load_data()
    print(f"Data: {df.shape[0]} observations, variables: {list(df.columns)}")
    print("Testing on FIRST DIFFERENCES because all variables are I(1).")
```

**Run it:**

```bash
python -m econometric_models.lag_selection
```

**What you should see:**

```
Data: 300 observations, variables: ['mpr', 'infl', 'exo', 'tbr']
Testing on FIRST DIFFERENCES because all variables are I(1).
```

- `from statsmodels.tsa.api import VAR` -- Imports the VAR model class. Today we only use its lag selection feature.
- `load_data()` -- Reads the cleaned CSV from Day 4.
- The print confirms your data loaded and reminds you why we difference: Days 6-7 established all four variables are I(1).

If that ran, move on.

---

### Build Step 2: Add select_lag_order Function

Delete everything in `econometric_models/lag_selection.py` and replace it with this:

```python
"""Lag selection for the Nigerian Inflation Predictor."""
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def select_lag_order(df, maxlags=12):
    """
    Select optimal VAR lag order using information criteria.

    Takes a DataFrame of level data, computes first differences,
    fits VAR models from lag 1 to maxlags, and returns the optimal
    lag according to each criterion: AIC, BIC, HQIC, FPE.
    """
    df_diff = df.diff().dropna()
    print(f"After differencing: {df_diff.shape[0]} observations")

    model = VAR(df_diff)
    lag_results = model.select_order(maxlags=maxlags)

    print("\n" + "=" * 60)
    print("VAR LAG ORDER SELECTION RESULTS")
    print("=" * 60)
    print(lag_results.summary())

    selected = {
        "aic": lag_results.aic,
        "bic": lag_results.bic,
        "hqic": lag_results.hqic,
        "fpe": lag_results.fpe,
    }

    print(f"\nOptimal lags per criterion:")
    print(f"  AIC:  {selected['aic']}")
    print(f"  BIC:  {selected['bic']}")
    print(f"  HQIC: {selected['hqic']}")
    print(f"  FPE:  {selected['fpe']}")

    return selected


if __name__ == "__main__":
    df = load_data()
    print(f"Data: {df.shape[0]} observations, variables: {list(df.columns)}")
    print("Testing on FIRST DIFFERENCES because all variables are I(1).\n")

    var_columns = ["mpr", "tbr", "exo", "infl"]
    selected = select_lag_order(df[var_columns], maxlags=12)
```

**Run it:**

```bash
python -m econometric_models.lag_selection
```

**What you should see:**

A table with each criterion value at each lag from 0 to 12. An asterisk (*) marks the minimum for each column.

```
After differencing: 299 observations

============================================================
VAR LAG ORDER SELECTION RESULTS
============================================================
  AIC          BIC          HQIC         FPE
==================================================
0  ...          ...          ...          ...
1  ...          ...          ...          ...
...
12 ...          ...          ...          ...
==================================================

Optimal lags per criterion:
  AIC:  X
  BIC:  Y
  HQIC: Z
  FPE:  W
```

**How to read the table:** Each row is a lag order. Each column is a criterion. The asterisk marks the best (lowest) value per criterion. If all asterisks are in the same row, the criteria agree. If not, you need a decision rule -- which is what Step 3 adds.

**Key detail:** `var_columns = ["mpr", "tbr", "exo", "infl"]` sets the VAR ordering: MPR (policy instrument), TBR (market rate), EXO (exchange rate channel), INF (target variable). This ordering matters for Cholesky decomposition in impulse response analysis later.

If that ran, move on.

---

### Build Step 3: Add Recommendation Logic and Save Results (FINAL)

Delete everything in `econometric_models/lag_selection.py` and replace it with this:

```python
"""Lag selection for the Nigerian Inflation Predictor."""
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def select_lag_order(df, maxlags=12):
    """
    Select optimal VAR lag order using information criteria.

    Takes a DataFrame of level data, computes first differences,
    fits VAR models from lag 1 to maxlags, and returns the optimal
    lag according to each criterion: AIC, BIC, HQIC, FPE.
    """
    df_diff = df.diff().dropna()
    print(f"After differencing: {df_diff.shape[0]} observations")

    model = VAR(df_diff)
    lag_results = model.select_order(maxlags=maxlags)

    print("\n" + "=" * 60)
    print("VAR LAG ORDER SELECTION RESULTS")
    print("=" * 60)
    print(lag_results.summary())

    selected = {
        "aic": lag_results.aic,
        "bic": lag_results.bic,
        "hqic": lag_results.hqic,
        "fpe": lag_results.fpe,
    }

    print(f"\nOptimal lags per criterion:")
    print(f"  AIC:  {selected['aic']}")
    print(f"  BIC:  {selected['bic']}")
    print(f"  HQIC: {selected['hqic']}")
    print(f"  FPE:  {selected['fpe']}")

    return selected


def recommend_lag(selected):
    """
    Recommend a single lag order using majority vote, BIC as tiebreaker.

    If 3 or 4 criteria agree, use that value.
    If tied (2 vs 2), use BIC (parsimonious, avoids overfitting).
    If all four disagree, default to BIC.
    """
    votes = [selected["aic"], selected["bic"],
             selected["hqic"], selected["fpe"]]

    from collections import Counter
    counts = Counter(votes)
    most_common_lag, most_common_count = counts.most_common(1)[0]

    print("\n" + "=" * 60)
    print("LAG RECOMMENDATION")
    print("=" * 60)

    if most_common_count >= 3:
        recommended = most_common_lag
        print(f"Clear majority: {most_common_count}/4 criteria select {recommended} lags.")
        reason = f"{most_common_count}/4 criteria agree on {recommended} lags"
    elif most_common_count == 2:
        recommended = selected["bic"]
        print(f"No majority: votes are {dict(counts)}.")
        print(f"Using BIC ({selected['bic']}) as tiebreaker (most parsimonious).")
        reason = f"Tie broken by BIC: {recommended} lags"
    else:
        recommended = selected["bic"]
        print(f"All criteria disagree: {dict(counts)}.")
        print(f"Defaulting to BIC ({selected['bic']}) for parsimony.")
        reason = f"No agreement; BIC default: {recommended} lags"

    print(f"\n>>> RECOMMENDED LAG ORDER: {recommended} <<<")
    print(f"    Reason: {reason}")

    return recommended, reason


if __name__ == "__main__":
    df = load_data()
    print(f"Data: {df.shape[0]} observations, variables: {list(df.columns)}")
    print("Testing on FIRST DIFFERENCES because all variables are I(1).\n")

    var_columns = ["mpr", "tbr", "exo", "infl"]
    selected = select_lag_order(df[var_columns], maxlags=12)
    recommended, reason = recommend_lag(selected)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df = pd.DataFrame([{
        "aic_lag": selected["aic"],
        "bic_lag": selected["bic"],
        "hqic_lag": selected["hqic"],
        "fpe_lag": selected["fpe"],
        "recommended_lag": recommended,
        "reason": reason,
    }])
    save_path = os.path.join(RESULTS_DIR, "lag_selection.csv")
    results_df.to_csv(save_path, index=False)
    print(f"\nSaved to {save_path}")

    # Final context
    print("\n" + "=" * 60)
    print("CONTEXT FOR NIGERIAN DATA")
    print("=" * 60)
    print(f"Selected lag: {recommended} months of history per variable.")
    print("For monthly macro data, 2-6 lags is typical.")
    print("CBN monetary policy transmission takes 6-18 months,")
    print("but cumulative impulse responses propagate the effect")
    print("far beyond the direct lag window.")
```

This is your **final complete file**.

**Run it:**

```bash
python -m econometric_models.lag_selection
```

**What you should see:**

```
Data: 300 observations, variables: ['mpr', 'infl', 'exo', 'tbr']
Testing on FIRST DIFFERENCES because all variables are I(1).

After differencing: 299 observations

============================================================
VAR LAG ORDER SELECTION RESULTS
============================================================
  AIC          BIC          HQIC         FPE
==================================================
0  ...          ...          ...          ...
...
12 ...          ...          ...          ...
==================================================

Optimal lags per criterion:
  AIC:  X       BIC:  Y       HQIC: Z       FPE:  W

============================================================
LAG RECOMMENDATION
============================================================
Clear majority: 3/4 criteria select N lags.

>>> RECOMMENDED LAG ORDER: N <<<
    Reason: 3/4 criteria agree on N lags

Saved to results/lag_selection.csv

============================================================
CONTEXT FOR NIGERIAN DATA
============================================================
Selected lag: N months of history per variable.
For monthly macro data, 2-6 lags is typical.
CBN monetary policy transmission takes 6-18 months,
but cumulative impulse responses propagate the effect
far beyond the direct lag window.
```

**What the new function does:**

- `recommend_lag(selected)` -- Takes the four optimal lags and applies majority voting. If 3 or 4 criteria agree, that lag wins. If 2-vs-2 tie, BIC wins as tiebreaker. If all disagree, BIC is the default. Returns the recommended lag and a reason string.
- The `__main__` block saves everything to `results/lag_selection.csv` so later weeks can read the recommendation without re-running lag selection.

---

## After the Code: Interpretation and Context

### Expected results for Nigerian data

AIC typically suggests **4-8 lags** (generous, captures more dynamics). BIC typically suggests **1-3 lags** (parsimonious, avoids overfitting). For monthly Nigerian macro data, a final recommendation of **2-6 months** is reasonable.

### For ARDL: ardl_select_order() does its own lag selection

The ARDL model you build in Week 3 selects lags per-variable independently using `ardl_select_order()`. Today's VAR lag selection does not constrain it. But it gives a useful benchmark -- if VAR selects 3 lags, the ARDL grid should include at least 3.

### For VAR: use this recommendation directly

In Week 5 you will use today's result:

```python
model = VAR(df_diff)
fitted = model.fit(maxlags=recommended)
```

### Nigerian context: why 2-6 lags is reasonable for monetary transmission

The CBN estimates policy transmission takes 6-18 months. So why do criteria select only 2-4 lags? Because information criteria penalize models that consume too many degrees of freedom. With 299 observations and 4 variables, a VAR(12) would have 196 parameters -- far too many.

But a VAR(3) does not mean the model forgets everything beyond 3 months. A shock in month 1 affects month 2 directly. Month 2's changed value affects month 3. Month 3 affects month 4. The effect propagates recursively through the system. The impulse response function (Week 5) captures this propagation over 12, 24, or 36 months -- far beyond the lag order.

---

## Week 2 Recap: Stationarity, Integration Order, Cointegration, Lag Selection

| Day | Test | Key Result |
|-----|------|------------|
| Day 6 | ADF stationarity | All 4 variables non-stationary in levels, stationary in first differences |
| Day 7 | KPSS confirmation | Confirmed: mpr, infl, exo, tbr are all I(1) |
| Day 8 | Engle-Granger cointegration | Identified which pairs share long-run equilibria |
| Day 9 | Johansen cointegration | Determined cointegration rank of the 4-variable system |
| Day 10 | Lag selection (today) | Selected optimal lag using AIC, BIC, HQIC, FPE |

These five results answer the three pre-estimation questions:

1. **Are the variables stationary?** No -- all I(1). Must difference or use ARDL/VECM.
2. **Are they cointegrated?** Johansen determined the rank. If rank > 0, error correction is needed.
3. **How many lags?** Information criteria selected the optimal order, saved in `results/lag_selection.csv`.

Every model from Week 3 onward depends on these answers.

---

## Commit

```bash
git add econometric_models/lag_selection.py results/lag_selection.csv
git commit -m "Day 10: VAR lag selection using information criteria"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `ValueError: maxlags is too large` | Data has fewer rows than expected. Reduce `maxlags`. Rule of thumb: maxlags <= (T / (k + 1)) - 1. |
| `UserWarning: date index has no associated frequency` | Warning, not error. Results are valid. To silence: `df_diff.index.freq = "MS"` after differencing. |
| `KeyError: 'mpr'` or `KeyError: 'infl'` | Column names in `cleaned_data.csv` do not match. Must be exactly: `mpr`, `infl`, `exo`, `tbr`. |
| `FileNotFoundError: cleaned_data.csv` | Run Day 4 cleaning first: `python -m data_processing.clean`. |
| All four criteria select the same lag | Not an error. Report: "All four criteria unanimously selected N lags." |
| `ModuleNotFoundError: statsmodels` | Run `pip install statsmodels`. Should already be installed from Day 6. |

---

## Check Your Understanding

### 1. "Why do you use first-differenced data for VAR lag selection?"

**Answer:** "The standard VAR assumes stationary data. All four variables -- mpr, infl, exo, tbr -- are I(1) as established on Days 6-7. I take first differences to achieve stationarity before selecting the lag order.

If I performed lag selection on non-stationary level data, the information criteria would be unreliable -- the VAR in levels would be misspecified and the asymptotic theory underlying the criteria would not apply.

If cointegration exists (Days 8-9), I could alternatively use a VECM in levels with an error correction term. The VECM lag order is typically p-1, where p is the VAR lag order selected today."

### 2. "AIC says 6 lags, BIC says 2. Which do you choose and why?"

**Answer:** "AIC penalizes complexity lightly, favouring more lags -- better for forecasting. BIC penalizes heavily and is consistent -- better for finding the true model.

I use majority voting with BIC as the tiebreaker. If 3 or 4 criteria agree, I use that. If tied, BIC wins because parsimony avoids overfitting with our 300-observation sample. I report all four criteria and present robustness checks under both specifications."

### 3. "If CBN policy takes 6-18 months to transmit, why do criteria select only 2-4 lags?"

**Answer:** "Information criteria balance dynamics against degrees of freedom. A VAR(12) with 4 variables has 196 parameters -- too many for 299 observations.

But the effective memory of a VAR is much longer than its lag order. In a VAR(3), a shock in month 1 affects month 2, which affects month 3, which affects month 4, and so on. The cumulative impulse response function traces this propagation over 12, 24, or 36 months. A lag order of 3 means each equation directly references 3 months, but the indirect effect cascades far beyond that."

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Lag selection script | `econometric_models/lag_selection.py` | Selects optimal VAR lag using AIC, BIC, HQIC, FPE with majority-vote recommendation |
| Lag selection results | `results/lag_selection.csv` | Stores optimal lag per criterion, recommended lag, and reasoning |

---

## Week 2 Complete -- All Files Built

| Day | File | Purpose |
|-----|------|---------|
| Day 6 | `econometric_models/stationarity.py` | ADF unit root test on all 4 variables |
| Day 7 | `econometric_models/stationarity_kpss.py` | KPSS stationarity test to confirm ADF results |
| Day 8 | `econometric_models/cointegration.py` | Engle-Granger pairwise cointegration tests |
| Day 9 | `econometric_models/cointegration.py` | Johansen multivariate cointegration (added to Day 8 file) |
| Day 10 | `econometric_models/lag_selection.py` | VAR lag selection using information criteria |

---

## Next Week Preview: Week 3 -- ARDL Model Building

In Week 3 you begin building the **ARDL (Autoregressive Distributed Lag) model** and applying the **bounds testing approach to cointegration**. ARDL works regardless of whether variables are I(0), I(1), or a mix -- making it the most flexible approach for applied macroeconomic research. You will:

- Learn the theory behind ARDL and the Pesaran-Shin-Smith bounds test
- Build the ARDL model specification using the stationarity results from Week 2
- Estimate short-run and long-run coefficients
- Conduct the bounds test for cointegration

The lag order, stationarity findings, and cointegration results from Week 2 all feed directly into the ARDL specification. Everything you built this week was preparation for this.

See you in Week 3.
