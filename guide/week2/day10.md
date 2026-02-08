# Week 2, Day 10 — Lag Selection: How Many Lags Do Your Models Need?

## What You'll Learn Today

- Why the number of lags matters in time series models
- What information criteria (AIC, BIC, HQIC, FPE) are and how they work
- How to use statsmodels to select the optimal lag order for a VAR model
- How to interpret the results for Nigerian macroeconomic data
- How the selected lag order carries forward into your VAR and ARDL models

## Why This Matters

Days 6 through 9 answered the foundational questions: Are the variables stationary? (No — they are all I(1).) Do they share a long-run equilibrium? (Cointegration testing answered that.) Now you face a new question that is just as important: **how many lags should your models include?**

This is not a trivial decision. Choose wrong and your entire model is compromised. Too few lags and you miss critical dynamics — your residuals will be autocorrelated, your standard errors will be wrong, and your impulse responses will be biased. Too many lags and you burn through degrees of freedom, your parameter estimates become imprecise, and your model may become unstable.

Today you will learn the formal, systematic way to choose the lag order using information criteria. By the end of today, you will have a concrete number — "my VAR model should use X lags" — backed by statistical evidence. That number will be used in the VAR model you build in Week 5 and will inform the ARDL specification in Weeks 3-4.

---

## No New Packages

You do not need to install anything new today. The `statsmodels` library you installed on Day 6 includes everything you need. Specifically, the `VAR` model class in `statsmodels.tsa.api` has a built-in `select_order()` method that computes all four information criteria across a range of lag lengths. No additional dependencies required.

---

## Theory: Why Lag Length Matters

This section is long. Read it carefully. Choosing the wrong lag length is one of the most common mistakes in applied time series econometrics, and examiners know this. They will ask you why you chose the number of lags you did.

### What "lags" mean in a time series model

In time series models like VAR (Vector Autoregression) and ARDL (Autoregressive Distributed Lag), each variable depends on its **own past values** and the **past values of other variables**. The number of past time periods you include is called the **lag order**, denoted by **p**.

Think of it this way. If you are modelling four variables — MPR, inflation, exchange rate, and M2 — and you set p=2, your model says:

> "This month's inflation depends on:
> - inflation 1 month ago, inflation 2 months ago,
> - MPR 1 month ago, MPR 2 months ago,
> - exchange rate 1 month ago, exchange rate 2 months ago,
> - M2 1 month ago, M2 2 months ago."

And the same for each of the other three variables. Every variable is explained by two months of history from every variable, including itself.

If you set p=3, you add a third month of history for every variable. If p=6, six months. And so on.

### The problem with too few lags

If you choose too few lags, the model misses important dynamics that are actually present in the data.

Here is a concrete example for Nigeria. The Central Bank of Nigeria (CBN) raises the Monetary Policy Rate (MPR) to fight inflation. But the effect of that rate hike does not show up in the inflation numbers immediately. It takes time for higher interest rates to slow lending, reduce spending, and eventually bring prices down. CBN research estimates this transmission takes **6 to 12 months**.

If you set p=2 (only 2 lags), your model can only see 2 months into the past. It will completely miss the effect of an MPR hike that takes 6 months to affect inflation. The model will conclude that MPR has no effect on inflation — not because it truly has no effect, but because you gave the model too narrow a window to detect it.

When you choose too few lags, the symptoms are:

- **Autocorrelated residuals.** The leftover errors from the model will be correlated with each other, because the model is systematically missing dynamics that are actually in the data.
- **Wrong standard errors.** If the residuals are autocorrelated, the standard errors on your coefficients are biased, which means your t-statistics and p-values are unreliable.
- **Biased impulse responses.** The impulse response functions (which show how a shock to one variable affects another over time) will be distorted because the model has an incomplete picture of the dynamics.

### The problem with too many lags

If you choose too many lags, the model becomes bloated with parameters it does not need.

Consider the arithmetic. In a VAR model with **k** variables and **p** lags, each equation has **k x p** coefficient parameters, plus a constant. With 4 variables:

- p=2: Each equation has 4 x 2 = 8 coefficients + 1 constant = 9 parameters. Total across 4 equations: 36 parameters.
- p=6: Each equation has 4 x 6 = 24 coefficients + 1 constant = 25 parameters. Total: 100 parameters.
- p=12: Each equation has 4 x 12 = 48 coefficients + 1 constant = 49 parameters. Total: **196 parameters**.

With only about 300 observations (and 299 after differencing), estimating 196 parameters leaves very few degrees of freedom. The symptoms of too many lags are:

- **Imprecise estimates.** Each additional parameter is estimated from the same fixed amount of data. More parameters means less data per parameter, which means wider confidence intervals and less certainty about the true values.
- **Overfitting.** The model fits the noise in your specific sample rather than the true underlying patterns. It will look great in-sample but forecast terribly out-of-sample.
- **Instability.** With too many parameters, the VAR companion matrix may develop eigenvalues near or outside the unit circle, making the model dynamically unstable.

### Information criteria: the solution

Information criteria solve this problem by formally balancing **goodness of fit** against **model complexity**. They work like this: you fit the model with p=1, then p=2, then p=3, all the way up to some maximum (we use 12 for monthly data). For each lag length, you compute the criterion value. The optimal lag is the one that **minimizes** the criterion.

There are four commonly used information criteria:

**AIC (Akaike Information Criterion)**

AIC = -2 x (log-likelihood) + 2 x (number of parameters)

The first term rewards fit (lower is better — higher log-likelihood means better fit). The second term penalizes complexity. AIC penalizes complexity **lightly**, which means it tends to choose **more lags**. AIC is better for **forecasting** because it allows the model to capture more of the data dynamics, even at the cost of some extra parameters.

**BIC (Bayesian Information Criterion, also called Schwarz Criterion)**

BIC = -2 x (log-likelihood) + log(n) x (number of parameters)

BIC replaces the "2" penalty in AIC with "log(n)", where n is the sample size. For any sample with more than 7 observations (which is always true in practice), log(n) > 2, so BIC penalizes complexity **more heavily** than AIC. This means BIC tends to choose **fewer lags**. BIC is better for **finding the true model** because it is "consistent" — as the sample size grows to infinity, BIC will select the correct model with probability 1.

**HQIC (Hannan-Quinn Information Criterion)**

HQIC = -2 x (log-likelihood) + 2 x log(log(n)) x (number of parameters)

The penalty is between AIC and BIC. HQIC is less commonly used but provides a middle ground. It is also consistent (like BIC) but converges more slowly.

**FPE (Final Prediction Error)**

FPE is similar to AIC and is specifically designed for prediction. It estimates the prediction error of the model on new data. Like AIC, it tends to favour more lags.

**The key takeaway:** Lower values are better for all four criteria. You fit models from p=1 to p=max_lags, compute each criterion at each lag, and pick the lag that minimizes each one.

### For this project

We will select lags for the VAR model. Since we are working with monthly macroeconomic data, the typical range to test is **1 to 12 lags** (1 month to 1 year of history). We test up to 12 because monetary policy effects in Nigeria can take up to a year to fully transmit through the economy.

### What to expect for Nigerian data

Information criteria on monthly macroeconomic data typically select **1 to 4 lags**. Here is what usually happens:

- **AIC** might choose a higher lag, like 3 or 4, because it penalizes complexity less and wants to capture more dynamics.
- **BIC** might choose a lower lag, like 1 or 2, because it penalizes complexity more and prefers parsimony.
- **HQIC** often falls between the two.
- **FPE** usually agrees with AIC or is close to it.

When the criteria disagree (which is common), you report all of them and justify your choice. For forecasting projects like ours, AIC is typically the primary specification, with BIC reported as a robustness check.

---

## Building `econometric_models/lag_selection.py` — Step by Step

At each step, we show you the **complete file** from the first line to the last line. No snippets, no "add this to the existing file." Delete everything and replace it with exactly what is shown.

Make sure you have an `econometric_models/` folder with an `__init__.py` inside it (from earlier days). If not, create them now.

---

### Build Step 1: Create the File with Imports, Data Loading, and First Differences

Delete everything in `econometric_models/lag_selection.py` and replace it with this:

```python
import os
import pandas as pd
from statsmodels.tsa.api import VAR

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows, {len(df.columns)} variables")
    print(f"Variables: {list(df.columns)}")

    # VAR requires stationary data — use first differences
    df_diff = df.diff().dropna()
    print(f"\nAfter differencing: {len(df_diff)} rows")
    print(df_diff.head())
```

**Run it:**

```bash
python -m econometric_models.lag_selection
```

**What you should see:**

```
Loaded 300 rows, 4 variables
Variables: ['mpr', 'inflation', 'exchange_rate', 'm2']

After differencing: 299 rows
                 mpr  inflation  exchange_rate       m2
date
2000-02-01      0.0       0.31           0.21     7.70
2000-03-01      0.0       0.28          -0.13    15.30
2000-04-01      0.0      -0.22           0.07   -12.40
2000-05-01      0.0       0.15           0.34    22.10
2000-06-01     -0.5       0.42           0.45    18.90
```

(Your exact numbers will depend on your data. The important thing is that you see 299 rows — one less than the original 300 because differencing loses the first observation.)

**What just happened — line by line:**

- `from statsmodels.tsa.api import VAR` — Imports the Vector Autoregression model class. This is the same `VAR` you will use to fit the actual model in Week 5. Today we are only using its lag selection feature.
- `PROCESSED_DIR` and `RESULTS_DIR` — Paths relative to where the script lives. `os.path.dirname(__file__)` is the `econometric_models/` folder. `".."` goes up one level to the project root.
- `load_data()` — Reads the cleaned CSV from Day 4. `index_col="date"` makes the date the row index. `parse_dates=True` converts date strings to datetime objects.
- `df.diff()` — Computes the first difference: each value minus the value from the previous row. For inflation, if January was 6.62 and February was 6.93, the first difference is 6.93 - 6.62 = 0.31. This transforms our I(1) variables into stationary I(0) series.
- `.dropna()` — The first row of the differenced data is NaN (there is no "previous row" for January 2000 to subtract), so we drop it. This is why we go from 300 to 299 rows.

If that ran and printed your differenced data, move on.

---

### Build Step 2: Add VAR Lag Selection Using select_order

Delete everything in `econometric_models/lag_selection.py` and replace it with this:

```python
import os
import pandas as pd
from statsmodels.tsa.api import VAR

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows, {len(df.columns)} variables")
    print(f"Variables: {list(df.columns)}")

    # VAR requires stationary data — use first differences
    df_diff = df.diff().dropna()
    print(f"\nAfter differencing: {len(df_diff)} rows")

    # Fit VAR and select optimal lag
    model = VAR(df_diff)
    lag_order_results = model.select_order(maxlags=12)
    print(lag_order_results.summary())
```

**Run it:**

```bash
python -m econometric_models.lag_selection
```

**What you should see:**

A table that looks something like this:

```
Loaded 300 rows, 4 variables
Variables: ['mpr', 'inflation', 'exchange_rate', 'm2']

After differencing: 299 rows
  AIC          BIC          HQIC         FPE
==================================================
0  ...          ...          ...          ...
1  ...          ...          ...          ...
2  ...          ...          ...          ...
3  ...          ...          ...          ...
...
12 ...          ...          ...          ...
==================================================
```

The exact format depends on your version of statsmodels, but the table will show the value of each information criterion for each lag order from 0 to 12. An asterisk (*) will appear next to the value that is the minimum for each criterion. For example, if AIC is minimized at lag 3, you will see an asterisk next to the AIC value in row 3.

**How to read the output table:**

- Each **row** is a lag order (0, 1, 2, ..., 12). Lag 0 means no lagged values at all (just a constant), lag 1 means one month of history, lag 2 means two months, and so on.
- Each **column** is an information criterion (AIC, BIC, HQIC, FPE).
- The **asterisk (*)** marks the lag order that minimizes each criterion. This is the optimal lag according to that criterion.
- If all four asterisks are in the same row, all criteria agree. If they are in different rows, the criteria disagree and you need to make a judgment call.

For example, if you see the asterisk for AIC in row 3 and the asterisk for BIC in row 1, that means AIC recommends 3 lags and BIC recommends 1 lag. This is a very common pattern — AIC almost always suggests the same or more lags than BIC.

If that ran and printed the table, move on to the final version.

---

### Build Step 3: Full Version with Custom Display, Recommendation, and Save (FINAL)

Delete everything in `econometric_models/lag_selection.py` and replace it with this:

```python
import os
import pandas as pd
from statsmodels.tsa.api import VAR

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df


def select_var_lag_order(df_diff, max_lags=12):
    """
    Determine the optimal lag order for a VAR model using information criteria.
    """
    model = VAR(df_diff)
    lag_results = model.select_order(maxlags=max_lags)

    print("=" * 60)
    print("VAR LAG ORDER SELECTION")
    print("=" * 60)
    print(lag_results.summary())

    # Extract optimal lags
    aic_lag = lag_results.aic
    bic_lag = lag_results.bic
    hqic_lag = lag_results.hqic
    fpe_lag = lag_results.fpe

    print(f"\nOptimal Lags:")
    print(f"  AIC:  {aic_lag}")
    print(f"  BIC:  {bic_lag}")
    print(f"  HQIC: {hqic_lag}")
    print(f"  FPE:  {fpe_lag}")

    return {"aic": aic_lag, "bic": bic_lag, "hqic": hqic_lag, "fpe": fpe_lag}


if __name__ == "__main__":
    df = load_data()
    variables = ["mpr", "inflation", "exchange_rate", "m2"]

    print("=" * 60)
    print("LAG SELECTION FOR VAR MODEL")
    print("=" * 60)

    # VAR requires stationary data — use first differences
    df_diff = df[variables].diff().dropna()
    print(f"Using first-differenced data: {len(df_diff)} observations")

    # Select optimal lag
    optimal = select_var_lag_order(df_diff, max_lags=12)

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    if optimal["aic"] == optimal["bic"]:
        print(f"AIC and BIC agree: use {optimal['aic']} lags.")
        recommended = optimal["aic"]
    else:
        print(f"AIC suggests {optimal['aic']} lags, BIC suggests {optimal['bic']} lags.")
        print(f"When they disagree, BIC is preferred for inference (hypothesis testing),")
        print(f"while AIC is preferred for forecasting. For this project, we will use")
        print(f"the AIC suggestion ({optimal['aic']}) as our primary specification")
        print(f"and report BIC ({optimal['bic']}) as a robustness check.")
        recommended = optimal["aic"]

    print(f"\nSelected lag order: {recommended}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df = pd.DataFrame([optimal])
    results_df["recommended"] = recommended
    results_df.to_csv(os.path.join(RESULTS_DIR, "lag_selection.csv"), index=False)
    print(f"Saved to results/lag_selection.csv")
```

This is your **final complete file**. It will not change again today.

**Run it:**

```bash
python -m econometric_models.lag_selection
```

**What you should see:**

```
============================================================
LAG SELECTION FOR VAR MODEL
============================================================
Using first-differenced data: 299 observations
============================================================
VAR LAG ORDER SELECTION
============================================================
  AIC          BIC          HQIC         FPE
==================================================
0  ...          ...          ...          ...
1  ...          ...          ...          ...
2  ...          ...          ...          ...
...
12 ...          ...          ...          ...
==================================================

Optimal Lags:
  AIC:  X
  BIC:  Y
  HQIC: Z
  FPE:  W

============================================================
RECOMMENDATION
============================================================
AIC suggests X lags, BIC suggests Y lags.
When they disagree, BIC is preferred for inference (hypothesis testing),
while AIC is preferred for forecasting. For this project, we will use
the AIC suggestion (X) as our primary specification
and report BIC (Y) as a robustness check.

Selected lag order: X
Saved to results/lag_selection.csv
```

(X, Y, Z, W will be actual numbers when you run it. Typically 1-4 for Nigerian monthly macro data.)

**What just happened — function by function:**

- `select_var_lag_order(df_diff, max_lags=12)` — This is the core function. It creates a `VAR` model object from the differenced data, then calls `select_order(maxlags=12)`. Under the hood, statsmodels fits the VAR with p=0, p=1, p=2, ..., p=12 and computes all four information criteria at each lag length. The `.aic`, `.bic`, `.hqic`, and `.fpe` attributes of the result object give you the lag that minimizes each criterion.
- The `if __name__ == "__main__":` block first loads the data and selects only the four variables we are modelling. It then differences the data (because VAR requires stationary data and all our variables are I(1)). It calls `select_var_lag_order()` to get the optimal lags, prints a recommendation, and saves the results to CSV.
- The recommendation logic checks whether AIC and BIC agree. If they agree, the choice is easy. If they disagree (the more common case), the script explains the trade-off and defaults to AIC for forecasting purposes.
- `results_df.to_csv(...)` saves all four optimal lags plus the recommended lag to a CSV file. You will read this file in later weeks when building the VAR and ARDL models so you do not have to re-run the lag selection.

---

## Interpreting Results — Nigeria Specific

### What the optimal lag means practically

If AIC says 2, it means the model works best when each variable depends on **2 months of history** from every variable. In concrete terms:

- This month's change in inflation is best explained by the past 2 months of changes in inflation, MPR, exchange rate, and M2.
- This month's change in the exchange rate is best explained by the past 2 months of changes in all four variables.
- And so on for MPR and M2.

If AIC says 3, extend everything to 3 months. If it says 1, each variable only needs last month's values.

### The Nigerian monetary policy context

The CBN estimates that the MPR-to-inflation transmission mechanism takes **6 to 12 months** to fully play out. You might wonder: if the effect takes 6-12 months, why do information criteria typically select only 1-4 lags?

The answer is that information criteria balance **capturing dynamics** against **preserving degrees of freedom given your sample size**. With 299 observations and 4 variables, a VAR(12) would have 196 coefficient parameters. That is too many for the data to reliably estimate. The information criteria know this — they are designed to penalize models that consume too many degrees of freedom relative to their improvement in fit.

This does not mean the 6-12 month transmission mechanism is absent from your model. Even with 2-3 lags, the **cumulative** impulse response (which you will compute in Week 5) can build up over many periods. A shock in month 1 affects month 2 directly, then month 2 affects month 3, and so on. The effect propagates through the system even though each equation only looks back a few months. This is a key insight about VAR models — the effective memory of the system is much longer than the lag order p.

### How this connects to your other models

- **VAR model (Week 5):** You will use the lag order selected today directly. When you fit `VAR(df_diff).fit(maxlags=recommended)`, the `recommended` value comes from today's analysis.
- **ARDL model (Weeks 3-4):** The ARDL framework does its own lag selection — it selects the number of lags for each variable independently (which is one of its advantages over VAR). However, today's result gives you a reasonable upper bound. If VAR selects 3 lags, you know the ARDL search should include at least 3 lags in its grid.
- **VECM (if applicable):** If you found cointegration in Day 9, you may use a Vector Error Correction Model instead of a plain VAR. The VECM lag order is typically p-1 where p is the VAR lag order.

### What to tell your examiner

"The optimal lag order for the VAR model was selected using information criteria on the first-differenced data. AIC suggested X lags and BIC suggested Y lags. We proceed with X lags as our primary specification, as AIC is preferred for forecasting purposes, and report the BIC specification as a robustness check. The selected lag order balances capturing the monetary policy transmission mechanism (which the CBN estimates takes 6-12 months) against preserving degrees of freedom given our 300-observation sample. Even with a modest lag order, the cumulative impulse response functions allow us to trace the full dynamic effect of policy shocks over longer horizons."

(Replace X and Y with your actual numbers when you present.)

---

## Step 2: Commit

```bash
git add econometric_models/lag_selection.py
git commit -m "Day 10: VAR lag selection using information criteria"
```

---

## Week 2 Summary

Congratulations. You have completed Week 2 of the Nigerian Inflation Predictor project. Let us take stock of everything you have accomplished.

### Day 6: ADF Test for Stationarity

You ran the Augmented Dickey-Fuller test on all four variables (MPR, inflation, exchange rate, M2) in levels and first differences. The result: **all four variables are non-stationary in levels** — they have unit roots. After first differencing, all four become stationary. This told you that the variables are integrated of order one.

### Day 7: KPSS Confirmation and Integration Order

You ran the KPSS test to confirm the ADF results (because relying on a single test is poor practice). KPSS tests from the opposite direction — it assumes stationarity as the null hypothesis. The KPSS results confirmed the ADF findings: all four variables are non-stationary in levels and stationary in first differences. You formally classified all variables as **I(1) — integrated of order one**.

### Day 8: Engle-Granger Pairwise Cointegration

You tested whether pairs of I(1) variables share a long-run equilibrium relationship using the Engle-Granger two-step procedure. This told you which pairs of variables move together in the long run, even though they individually wander like random walks.

### Day 9: Johansen Multivariate Cointegration

You moved from pairwise (Engle-Granger) to multivariate (Johansen) cointegration testing. The Johansen test can detect multiple cointegrating relationships among all four variables simultaneously. It determined the **cointegration rank** — the number of independent long-run equilibrium relationships in the system.

### Day 10: Lag Selection (Today)

You selected the optimal lag order for the VAR model using four information criteria (AIC, BIC, HQIC, FPE) on first-differenced data. You now have a specific, statistically justified lag order to use when building the VAR model. The result is saved in `results/lag_selection.csv`.

### What This All Means Together

At the end of Week 2, you have answered the three fundamental questions that every time series econometric analysis must address before building any model:

1. **Are the variables stationary?** No. All four are I(1). You must either difference the data or use a model that handles non-stationarity (like ARDL or VECM).
2. **Are the non-stationary variables cointegrated?** The Engle-Granger and Johansen tests determined whether there are long-run equilibrium relationships. If cointegration exists, an error correction term should be included in the model.
3. **How many lags should the model include?** Information criteria selected the optimal lag order, balancing fit against complexity.

These three answers dictate your modelling strategy for the rest of the project.

### Preview of Week 3

In Week 3, you begin building the **ARDL (Autoregressive Distributed Lag) model** and applying the **bounds testing approach to cointegration**. The ARDL model has a major advantage: it works regardless of whether variables are I(0), I(1), or a mix. This makes it the most flexible and commonly used approach in applied macroeconomic research. You will:

- Learn the theory behind ARDL and bounds testing
- Build the ARDL model specification
- Estimate the model and interpret short-run and long-run coefficients
- Conduct the bounds test for cointegration

The stationarity results, cointegration findings, and lag order from Week 2 will all feed directly into the ARDL model. Everything you have done so far was building toward this.

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `ValueError: maxlags is too large for the available data` or similar | You are asking for more lags than the data can support. With 299 observations and 4 variables, `maxlags=12` should work fine, but if your data has fewer rows (maybe you have missing values or a shorter sample), reduce `maxlags` to a smaller number. A rule of thumb is maxlags should be no more than (T / (k + 1)) - 1, where T is the number of observations and k is the number of variables. |
| `ValueError: The data must be stationary` or residual autocorrelation warnings | VAR lag selection should be performed on stationary data. If you forgot to difference the data (or are passing the raw levels), the model will either error out or produce unreliable results. Make sure you are using `df.diff().dropna()` before passing the data to `VAR()`. |
| `UserWarning: A date index has been provided, but it has no associated frequency information` | Your date index does not have a frequency set. This is a warning, not an error — the lag selection will still work. To silence it, add `df_diff.index.freq = "MS"` (for month-start frequency) after differencing. Or, ignore the warning — it does not affect the results. |
| `KeyError: 'mpr'` or `KeyError: 'inflation'` | Your cleaned CSV does not have the expected column names. Open `data/processed/cleaned_data.csv` and check the header row. The columns must be exactly: `mpr`, `inflation`, `exchange_rate`, `m2`. |
| `FileNotFoundError: cleaned_data.csv not found` | You need to run the Day 4 cleaning script first: `python -m data_processing.clean`. The lag selection script reads from `data/processed/cleaned_data.csv`. |
| All four criteria select the same lag (e.g., all say 1) | This is not an error — it means the criteria agree, which makes your life easier. Just report it: "All four information criteria unanimously selected 1 lag." |

---

## Check Your Understanding

These are the kinds of questions an examiner will ask about lag selection. Practice answering them out loud before your defense.

### 1. "Why do you use first-differenced data for VAR lag selection?"

**Answer:** "A standard VAR model assumes that the data is stationary. Since all four of our variables — MPR, inflation, exchange rate, and M2 — are integrated of order one, I(1), as established by the ADF and KPSS tests on Days 6 and 7, they are not stationary in their raw levels. We therefore take first differences to achieve stationarity before fitting the VAR and selecting the lag order.

If we performed lag selection on the non-stationary level data, the information criteria would be unreliable because the VAR in levels would be misspecified — the residuals would be non-stationary and the asymptotic theory underlying the information criteria would not apply.

It is worth noting that if we find cointegration (as tested on Days 8 and 9), we could alternatively use a Vector Error Correction Model (VECM), which works with the data in levels and includes an error correction term. The VECM lag order is typically p-1, where p is the VAR lag order we selected today."

### 2. "AIC says 3 lags, BIC says 1. Which do you choose and why?"

**Answer:** "The disagreement between AIC and BIC is expected and well-understood in the econometrics literature. AIC penalizes model complexity less heavily, so it tends to select more lags — it prioritizes capturing dynamics even at the cost of some extra parameters. BIC penalizes complexity more heavily because its penalty term grows with the sample size, so it tends to select fewer lags — it prioritizes parsimony and is statistically consistent, meaning it converges to the true model as the sample grows.

For this project, which is primarily a forecasting exercise, I use the AIC suggestion of 3 lags as the primary specification, because AIC is generally preferred for forecasting — it captures more of the data dynamics that are useful for prediction. I report the BIC suggestion of 1 lag as a robustness check, to show that the main conclusions hold even with a more parsimonious specification.

In my results, I present impulse response functions and Granger causality tests for both specifications and note any differences. If the qualitative conclusions are the same under both lag orders, this strengthens the findings."

### 3. "What happens if you choose too few or too many lags?"

**Answer:** "Choosing too few lags means the model misses important dynamics that are actually present in the data. The symptoms are autocorrelated residuals, biased standard errors, and distorted impulse response functions. For example, if the Central Bank of Nigeria's interest rate changes take 6 months to affect inflation but I only include 2 lags, the model would incorrectly conclude that monetary policy has no effect on inflation.

Choosing too many lags means the model has too many parameters relative to the available data. With 4 variables and 12 lags, a VAR would have 196 coefficient parameters — far too many for a 299-observation sample. The symptoms are imprecise parameter estimates with wide confidence intervals, overfitting to noise in the sample (which produces poor out-of-sample forecasts), and potential dynamic instability where the VAR companion matrix has eigenvalues near the unit circle.

Information criteria optimally balance these two risks. They reward better fit but penalize additional parameters, finding the lag order that best trades off capturing genuine dynamics against wasting degrees of freedom on noise."

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Lag selection script | `econometric_models/lag_selection.py` | Selects optimal VAR lag order using AIC, BIC, HQIC, and FPE |
| Lag selection results | `results/lag_selection.csv` | Stores optimal lag for each criterion plus recommended lag |

---

## Week 2 Complete — All Files Built

| Day | File | Purpose |
|-----|------|---------|
| Day 6 | `econometric_models/stationarity_adf.py` | ADF unit root test on all 4 variables |
| Day 7 | `econometric_models/stationarity_kpss.py` | KPSS stationarity test to confirm ADF results |
| Day 8 | `econometric_models/cointegration_eg.py` | Engle-Granger pairwise cointegration tests |
| Day 9 | `econometric_models/cointegration_johansen.py` | Johansen multivariate cointegration test |
| Day 10 | `econometric_models/lag_selection.py` | VAR lag selection using information criteria |

You now have a complete pre-estimation diagnostic toolkit. Every one of these results — stationarity classification, integration order, cointegration rank, and optimal lag order — feeds directly into the models you will build starting in Week 3.

See you in Week 3, where you begin building the ARDL model.
