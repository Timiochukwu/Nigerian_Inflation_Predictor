# Day 12 — ARDL Model Estimation

## What You'll Learn Today

By the end of today, you will:

1. **Estimate the ARDL model** using statsmodels with the optimal lag structure you found on Day 11
2. **Read and interpret regression output** including coefficients, standard errors, t-statistics, p-values, and R²
3. **Understand what each coefficient means** in the context of Nigerian monetary policy and inflation
4. **Extract and save model results** for further analysis and reporting

This is where your project transforms from data preparation to actual econometric modeling—you'll run the regression that tells you how MPR, TBR, and exchange rates relate to inflation.

---

## Why This Matters

The ARDL (Autoregressive Distributed Lag) model is the heart of your inflation predictor. Here's what it does:

- **Quantifies relationships**: Instead of just saying "MPR affects inflation," the model tells you *by how much* and *how quickly*
- **Captures dynamics**: It shows both immediate effects (what happens this month) and delayed effects (what happens over the next few months)
- **Tests statistical significance**: You'll know which variables actually matter and which might just be noise
- **Provides prediction power**: Once estimated, you can use this model to forecast future inflation

For the Central Bank of Nigeria, this means understanding:
- Does raising the MPR actually help control inflation?
- How long does it take for policy changes to affect prices?
- What role do exchange rates and Treasury bill rates play?

---

## No New Packages

Good news—you already have everything you need! You installed statsmodels on Day 11, and that same package includes the ARDL model estimation functions. No new installations today.

---

## Building the ARDL Model Estimator

You'll expand the `econometric_models/ardl_model.py` file in three steps:

1. **Add ARDL estimation function** to fit the model and display full output
2. **Add key results display** to extract the most important statistics in a readable format
3. **Add results saving** to export coefficients to CSV for later use

Each step shows the COMPLETE file from line 1 to the end.

---

## STEP 1: Add ARDL Estimation Function

**Instruction**: Delete everything in `econometric_models/ardl_model.py` and replace it with this:

```python
"""
ARDL Model for Nigerian Inflation
This module selects optimal lags and estimates the ARDL model.
"""

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.ardl import ARDL, ardl_select_order

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")

DEPENDENT = "infl"
EXOG_VARS = ["mpr", "tbr", "exo"]


def load_data():
    """
    Load cleaned data with date index.

    Returns:
        DataFrame with columns: mpr, infl, exo, tbr (index: date)
    """
    df = pd.read_csv(CLEANED_DATA_PATH, index_col=0, parse_dates=True)
    print(f"✓ Loaded {len(df)} observations from {CLEANED_DATA_PATH}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    return df


def select_ardl_order(df, maxlag=12, ic="aic"):
    """
    Select optimal ARDL lag structure using information criteria.

    Parameters:
        df: DataFrame with dependent and exogenous variables
        maxlag: Maximum lag to consider (default 12 for monthly data)
        ic: Information criterion - "aic" or "bic" (default "aic")

    Returns:
        dict with keys "p" (dependent lags) and "q" (exog lags dict)
    """
    print(f"\n{'='*60}")
    print(f"SELECTING OPTIMAL ARDL LAG ORDER")
    print(f"{'='*60}")
    print(f"Criterion: {ic.upper()}")
    print(f"Max lags: {maxlag}")
    print(f"Dependent variable: {DEPENDENT}")
    print(f"Exogenous variables: {EXOG_VARS}")

    # Prepare data
    y = df[DEPENDENT]
    X = df[EXOG_VARS]

    # Select order
    print("\nSearching for optimal lags... (this may take 30-60 seconds)")
    selection = ardl_select_order(
        endog=y,
        exog=X,
        maxlag=maxlag,
        ic=ic,
        trend="c"  # "c" = constant term (intercept)
    )

    # Extract results
    model_spec = selection.model
    p = model_spec.ardl_order[0]  # lags of dependent variable
    q_dict = {}
    for i, var in enumerate(EXOG_VARS):
        q_dict[var] = model_spec.ardl_order[i + 1]

    print("\n" + "="*60)
    print("OPTIMAL LAG STRUCTURE")
    print("="*60)
    print(f"Dependent variable lags (p): {p}")
    print(f"Exogenous variable lags (q):")
    for var, lag in q_dict.items():
        print(f"  {var}: {lag} lags")
    print(f"\n{ic.upper()}: {selection.aic if ic == 'aic' else selection.bic:.2f}")
    print("="*60)

    return {"p": p, "q": q_dict}


def estimate_ardl(df, p, q_dict):
    """
    Estimate ARDL model with specified lag structure.

    Parameters:
        df: DataFrame with dependent and exogenous variables
        p: Number of lags for dependent variable
        q_dict: Dictionary mapping exog variable names to their lags

    Returns:
        Fitted ARDL model result object
    """
    print(f"\n{'='*60}")
    print(f"ESTIMATING ARDL({p}, {', '.join([str(q_dict[v]) for v in EXOG_VARS])}) MODEL")
    print(f"{'='*60}")

    # Prepare data
    y = df[DEPENDENT]
    X = df[EXOG_VARS]

    # Build order tuple: (p, q_mpr, q_tbr, q_exo)
    # Must match the order of variables in EXOG_VARS
    order = {var: q_dict[var] for var in EXOG_VARS}

    print(f"\nDependent: {DEPENDENT} with {p} lags")
    print(f"Exogenous:")
    for var in EXOG_VARS:
        print(f"  {var} with {order[var]} lags")

    # Create ARDL model
    model = ARDL(
        endog=y,
        lags=p,
        exog=X,
        order=order,
        trend="c"  # Include constant term
    )

    # Fit the model
    print("\nFitting model...")
    result = model.fit()

    print("✓ Model estimation complete!")
    print(f"  Observations: {result.nobs}")
    print(f"  R-squared: {result.rsquared:.4f}")
    print(f"  Adjusted R-squared: {result.rsquared_adj:.4f}")

    # Display full regression output
    print("\n" + "="*60)
    print("FULL REGRESSION OUTPUT")
    print("="*60)
    print(result.summary())

    return result


if __name__ == "__main__":
    print("="*60)
    print("ARDL MODEL FOR NIGERIAN INFLATION")
    print("="*60)

    # Load data
    df = load_data()

    # Select optimal lag order
    order_result = select_ardl_order(df, maxlag=12, ic="aic")
    p = order_result["p"]
    q_dict = order_result["q"]

    # Estimate ARDL model
    ardl_result = estimate_ardl(df, p, q_dict)

    print("\n" + "="*60)
    print("ESTIMATION COMPLETE")
    print("="*60)
    print("Next steps:")
    print("  1. Review the coefficient table above")
    print("  2. Check which variables are statistically significant (p < 0.05)")
    print("  3. Note the R-squared value (how well the model fits)")
```

**What's new in this step**:

1. **`estimate_ardl()` function**: This is the core function that:
   - Takes your data and the optimal lag structure from Day 11
   - Creates an ARDL model object using `ARDL()` from statsmodels
   - Fits the model to your data with `.fit()` (this is where the actual estimation happens)
   - Prints the full regression output table
   - Returns the result object for further analysis

2. **Model specification**:
   - `endog=y`: The dependent variable (inflation)
   - `lags=p`: Number of lags of inflation to include
   - `exog=X`: The exogenous variables (MPR, TBR, exchange rate)
   - `order=order`: Dictionary telling how many lags of each exogenous variable to include
   - `trend="c"`: Include a constant term (intercept) in the regression

3. **Main block**: Now runs the complete workflow—load data, select lags, estimate model

**Run it**:

```bash
python econometric_models/ardl_model.py
```

**Expected output pattern**:

```
============================================================
ARDL MODEL FOR NIGERIAN INFLATION
============================================================
✓ Loaded 150 observations from data/processed/cleaned_data.csv
  Columns: ['mpr', 'infl', 'exo', 'tbr']
  Date range: 2010-01-01 to 2022-06-01

============================================================
SELECTING OPTIMAL ARDL LAG ORDER
============================================================
...
[lag selection output from Day 11]
...

============================================================
ESTIMATING ARDL(4, 2, 1, 3) MODEL
============================================================

Dependent: infl with 4 lags
Exogenous:
  mpr with 2 lags
  tbr with 1 lags
  exo with 3 lags

Fitting model...
✓ Model estimation complete!
  Observations: 146
  R-squared: 0.8234
  Adjusted R-squared: 0.8076

============================================================
FULL REGRESSION OUTPUT
============================================================
                            ARDL Regression Results
==============================================================================
Dep. Variable:                   infl   No. Observations:                  146
Model:                     ARDL(4, 2)   Log Likelihood                -345.67
Date:                Mon, 10 Feb 2026   AIC                            723.34
Time:                        14:23:45   BIC                            756.89
Sample:                    01-01-2010   HQIC                           737.12
                         - 06-01-2022
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.1456      0.893      2.403      0.018       0.379       3.912
L1.infl        0.7234      0.082      8.823      0.000       0.562       0.885
L2.infl        0.1456      0.091      1.601      0.112      -0.034       0.325
L3.infl       -0.0234      0.088     -0.266      0.791      -0.197       0.150
L4.infl        0.0823      0.079      1.042      0.299      -0.074       0.239
mpr            0.1234      0.045      2.742      0.007       0.034       0.213
L1.mpr         0.0567      0.051      1.112      0.268      -0.044       0.157
L2.mpr        -0.0345      0.043     -0.802      0.424      -0.120       0.051
tbr           -0.0234      0.038     -0.616      0.539      -0.098       0.051
L1.tbr         0.0456      0.039      1.169      0.244      -0.031       0.122
exo            0.0045      0.002      2.250      0.026       0.001       0.008
L1.exo         0.0023      0.002      1.150      0.252      -0.002       0.007
L2.exo         0.0012      0.002      0.600      0.550      -0.003       0.005
L3.exo        -0.0008      0.002     -0.400      0.690      -0.005       0.003
==============================================================================
Omnibus:                        3.456   Durbin-Watson:                   1.987
Prob(Omnibus):                  0.178   Jarque-Bera (JB):                3.234
Skew:                           0.345   Prob(JB):                        0.198
Kurtosis:                       2.876   Cond. No.                         456.
==============================================================================
```

(Note: The actual numbers will vary based on your real data and optimal lag selection)

**What you're seeing**:

1. **Model header**: Shows it's an ARDL model with your lag structure
2. **Coefficient table**: Each row is a variable in your model
   - `const`: The intercept term
   - `L1.infl`, `L2.infl`, etc.: Lagged values of inflation (L1 = 1 month ago, L2 = 2 months ago)
   - `mpr`, `L1.mpr`, etc.: Current and lagged values of Monetary Policy Rate
   - `tbr`, `L1.tbr`, etc.: Treasury Bill Rate
   - `exo`, `L1.exo`, etc.: Exchange rate
3. **Statistics**: Each coefficient has:
   - `coef`: The estimated coefficient (effect size)
   - `std err`: Standard error (uncertainty in the estimate)
   - `t`: t-statistic (coef ÷ std err)
   - `P>|t|`: p-value (if < 0.05, the variable is statistically significant)
   - `[0.025 0.975]`: 95% confidence interval
4. **Model fit statistics**: R², AIC, BIC, etc.

---

## STEP 2: Add Key Results Display Function

The full regression table is great for detailed analysis, but it's overwhelming. Let's extract the key results in a cleaner format.

**Instruction**: Delete everything in `econometric_models/ardl_model.py` and replace it with this:

```python
"""
ARDL Model for Nigerian Inflation
This module selects optimal lags and estimates the ARDL model.
"""

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.ardl import ARDL, ardl_select_order

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")

DEPENDENT = "infl"
EXOG_VARS = ["mpr", "tbr", "exo"]


def load_data():
    """
    Load cleaned data with date index.

    Returns:
        DataFrame with columns: mpr, infl, exo, tbr (index: date)
    """
    df = pd.read_csv(CLEANED_DATA_PATH, index_col=0, parse_dates=True)
    print(f"✓ Loaded {len(df)} observations from {CLEANED_DATA_PATH}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    return df


def select_ardl_order(df, maxlag=12, ic="aic"):
    """
    Select optimal ARDL lag structure using information criteria.

    Parameters:
        df: DataFrame with dependent and exogenous variables
        maxlag: Maximum lag to consider (default 12 for monthly data)
        ic: Information criterion - "aic" or "bic" (default "aic")

    Returns:
        dict with keys "p" (dependent lags) and "q" (exog lags dict)
    """
    print(f"\n{'='*60}")
    print(f"SELECTING OPTIMAL ARDL LAG ORDER")
    print(f"{'='*60}")
    print(f"Criterion: {ic.upper()}")
    print(f"Max lags: {maxlag}")
    print(f"Dependent variable: {DEPENDENT}")
    print(f"Exogenous variables: {EXOG_VARS}")

    # Prepare data
    y = df[DEPENDENT]
    X = df[EXOG_VARS]

    # Select order
    print("\nSearching for optimal lags... (this may take 30-60 seconds)")
    selection = ardl_select_order(
        endog=y,
        exog=X,
        maxlag=maxlag,
        ic=ic,
        trend="c"  # "c" = constant term (intercept)
    )

    # Extract results
    model_spec = selection.model
    p = model_spec.ardl_order[0]  # lags of dependent variable
    q_dict = {}
    for i, var in enumerate(EXOG_VARS):
        q_dict[var] = model_spec.ardl_order[i + 1]

    print("\n" + "="*60)
    print("OPTIMAL LAG STRUCTURE")
    print("="*60)
    print(f"Dependent variable lags (p): {p}")
    print(f"Exogenous variable lags (q):")
    for var, lag in q_dict.items():
        print(f"  {var}: {lag} lags")
    print(f"\n{ic.upper()}: {selection.aic if ic == 'aic' else selection.bic:.2f}")
    print("="*60)

    return {"p": p, "q": q_dict}


def estimate_ardl(df, p, q_dict):
    """
    Estimate ARDL model with specified lag structure.

    Parameters:
        df: DataFrame with dependent and exogenous variables
        p: Number of lags for dependent variable
        q_dict: Dictionary mapping exog variable names to their lags

    Returns:
        Fitted ARDL model result object
    """
    print(f"\n{'='*60}")
    print(f"ESTIMATING ARDL({p}, {', '.join([str(q_dict[v]) for v in EXOG_VARS])}) MODEL")
    print(f"{'='*60}")

    # Prepare data
    y = df[DEPENDENT]
    X = df[EXOG_VARS]

    # Build order tuple: (p, q_mpr, q_tbr, q_exo)
    # Must match the order of variables in EXOG_VARS
    order = {var: q_dict[var] for var in EXOG_VARS}

    print(f"\nDependent: {DEPENDENT} with {p} lags")
    print(f"Exogenous:")
    for var in EXOG_VARS:
        print(f"  {var} with {order[var]} lags")

    # Create ARDL model
    model = ARDL(
        endog=y,
        lags=p,
        exog=X,
        order=order,
        trend="c"  # Include constant term
    )

    # Fit the model
    print("\nFitting model...")
    result = model.fit()

    print("✓ Model estimation complete!")
    print(f"  Observations: {result.nobs}")
    print(f"  R-squared: {result.rsquared:.4f}")
    print(f"  Adjusted R-squared: {result.rsquared_adj:.4f}")

    # Display full regression output
    print("\n" + "="*60)
    print("FULL REGRESSION OUTPUT")
    print("="*60)
    print(result.summary())

    return result


def display_key_results(result):
    """
    Extract and display key results from ARDL estimation.

    Parameters:
        result: Fitted ARDL model result object
    """
    print(f"\n{'='*60}")
    print("KEY MODEL RESULTS")
    print("="*60)

    # Model fit statistics
    print("\nMODEL FIT:")
    print(f"  Observations:       {int(result.nobs)}")
    print(f"  R-squared:          {result.rsquared:.4f}")
    print(f"  Adjusted R-squared: {result.rsquared_adj:.4f}")
    print(f"  AIC:                {result.aic:.2f}")
    print(f"  BIC:                {result.bic:.2f}")
    print(f"  Log-Likelihood:     {result.llf:.2f}")

    # Coefficient table
    print("\nCOEFFICIENT TABLE:")
    print("-" * 85)
    print(f"{'Variable':<15} {'Coefficient':>12} {'Std Error':>12} {'t-stat':>10} {'p-value':>10} {'Sig':>5}")
    print("-" * 85)

    # Extract coefficients, standard errors, t-stats, p-values
    params = result.params
    std_err = result.bse
    t_stats = result.tvalues
    p_values = result.pvalues

    for var in params.index:
        coef = params[var]
        se = std_err[var]
        t_stat = t_stats[var]
        p_val = p_values[var]

        # Determine significance stars
        if p_val < 0.01:
            sig = "***"
        elif p_val < 0.05:
            sig = "**"
        elif p_val < 0.10:
            sig = "*"
        else:
            sig = ""

        print(f"{var:<15} {coef:>12.4f} {se:>12.4f} {t_stat:>10.3f} {p_val:>10.4f} {sig:>5}")

    print("-" * 85)
    print("Significance levels: *** p<0.01, ** p<0.05, * p<0.10")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("ARDL MODEL FOR NIGERIAN INFLATION")
    print("="*60)

    # Load data
    df = load_data()

    # Select optimal lag order
    order_result = select_ardl_order(df, maxlag=12, ic="aic")
    p = order_result["p"]
    q_dict = order_result["q"]

    # Estimate ARDL model
    ardl_result = estimate_ardl(df, p, q_dict)

    # Display key results in clean format
    display_key_results(ardl_result)

    print("\n" + "="*60)
    print("ESTIMATION COMPLETE")
    print("="*60)
    print("Next steps:")
    print("  1. Review the coefficient table above")
    print("  2. Check which variables are statistically significant")
    print("  3. Interpret the economic meaning of key coefficients")
```

**What's new in this step**:

1. **`display_key_results()` function**: Extracts and formats the most important results:
   - **Model fit statistics**: How well the model explains inflation variation
   - **Clean coefficient table**: Shows each variable with its:
     - Coefficient: The estimated effect size
     - Standard error: Uncertainty measure
     - t-statistic: How many standard errors away from zero (bigger = more significant)
     - p-value: Probability the true effect is zero (smaller = more significant)
     - Significance stars: Quick visual indicator (*** = very significant, ** = significant, * = weakly significant)

2. **Significance levels**:
   - `p < 0.01` (***): Very strong evidence the variable matters
   - `p < 0.05` (**): Strong evidence (conventional threshold)
   - `p < 0.10` (*): Weak evidence
   - `p > 0.10` (no star): Not statistically significant

**Run it**:

```bash
python econometric_models/ardl_model.py
```

**Expected output pattern** (after the full regression output):

```
============================================================
KEY MODEL RESULTS
============================================================

MODEL FIT:
  Observations:       146
  R-squared:          0.8234
  Adjusted R-squared: 0.8076
  AIC:                723.34
  BIC:                756.89
  Log-Likelihood:     -345.67

COEFFICIENT TABLE:
-------------------------------------------------------------------------------------
Variable        Coefficient    Std Error     t-stat    p-value   Sig
-------------------------------------------------------------------------------------
const                2.1456       0.8930      2.403     0.0180    **
L1.infl              0.7234       0.0820      8.823     0.0000   ***
L2.infl              0.1456       0.0910      1.601     0.1120
L3.infl             -0.0234       0.0880     -0.266     0.7910
L4.infl              0.0823       0.0790      1.042     0.2990
mpr                  0.1234       0.0450      2.742     0.0070   ***
L1.mpr               0.0567       0.0510      1.112     0.2680
L2.mpr              -0.0345       0.0430     -0.802     0.4240
tbr                 -0.0234       0.0380     -0.616     0.5390
L1.tbr               0.0456       0.0390      1.169     0.2440
exo                  0.0045       0.0020      2.250     0.0260    **
L1.exo               0.0023       0.0020      1.150     0.2520
L2.exo               0.0012       0.0020      0.600     0.5500
L3.exo              -0.0008       0.0020     -0.400     0.6900
-------------------------------------------------------------------------------------
Significance levels: *** p<0.01, ** p<0.05, * p<0.10
============================================================
```

**What this tells you**:

- **R² = 0.8234**: The model explains 82.34% of inflation variation—very good!
- **L1.infl = 0.7234 (***)**:: Last month's inflation strongly predicts this month's inflation (inflation is persistent)
- **mpr = 0.1234 (***)**:: A 1 percentage point increase in MPR is associated with 0.12 percentage point higher inflation this month (but this doesn't mean causation—see interpretation below!)
- **exo = 0.0045 (**)**:: A ₦1 depreciation in exchange rate increases inflation by 0.0045 percentage points

---

## STEP 3: Add Results Saving Function

Now let's save the coefficients to a CSV file for documentation and future reference.

**Instruction**: Delete everything in `econometric_models/ardl_model.py` and replace it with this:

```python
"""
ARDL Model for Nigerian Inflation
This module selects optimal lags and estimates the ARDL model.
"""

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.ardl import ARDL, ardl_select_order

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
COEF_OUTPUT_PATH = os.path.join(RESULTS_DIR, "ardl_coefficients.csv")

DEPENDENT = "infl"
EXOG_VARS = ["mpr", "tbr", "exo"]


def load_data():
    """
    Load cleaned data with date index.

    Returns:
        DataFrame with columns: mpr, infl, exo, tbr (index: date)
    """
    df = pd.read_csv(CLEANED_DATA_PATH, index_col=0, parse_dates=True)
    print(f"✓ Loaded {len(df)} observations from {CLEANED_DATA_PATH}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    return df


def select_ardl_order(df, maxlag=12, ic="aic"):
    """
    Select optimal ARDL lag structure using information criteria.

    Parameters:
        df: DataFrame with dependent and exogenous variables
        maxlag: Maximum lag to consider (default 12 for monthly data)
        ic: Information criterion - "aic" or "bic" (default "aic")

    Returns:
        dict with keys "p" (dependent lags) and "q" (exog lags dict)
    """
    print(f"\n{'='*60}")
    print(f"SELECTING OPTIMAL ARDL LAG ORDER")
    print(f"{'='*60}")
    print(f"Criterion: {ic.upper()}")
    print(f"Max lags: {maxlag}")
    print(f"Dependent variable: {DEPENDENT}")
    print(f"Exogenous variables: {EXOG_VARS}")

    # Prepare data
    y = df[DEPENDENT]
    X = df[EXOG_VARS]

    # Select order
    print("\nSearching for optimal lags... (this may take 30-60 seconds)")
    selection = ardl_select_order(
        endog=y,
        exog=X,
        maxlag=maxlag,
        ic=ic,
        trend="c"  # "c" = constant term (intercept)
    )

    # Extract results
    model_spec = selection.model
    p = model_spec.ardl_order[0]  # lags of dependent variable
    q_dict = {}
    for i, var in enumerate(EXOG_VARS):
        q_dict[var] = model_spec.ardl_order[i + 1]

    print("\n" + "="*60)
    print("OPTIMAL LAG STRUCTURE")
    print("="*60)
    print(f"Dependent variable lags (p): {p}")
    print(f"Exogenous variable lags (q):")
    for var, lag in q_dict.items():
        print(f"  {var}: {lag} lags")
    print(f"\n{ic.upper()}: {selection.aic if ic == 'aic' else selection.bic:.2f}")
    print("="*60)

    return {"p": p, "q": q_dict}


def estimate_ardl(df, p, q_dict):
    """
    Estimate ARDL model with specified lag structure.

    Parameters:
        df: DataFrame with dependent and exogenous variables
        p: Number of lags for dependent variable
        q_dict: Dictionary mapping exog variable names to their lags

    Returns:
        Fitted ARDL model result object
    """
    print(f"\n{'='*60}")
    print(f"ESTIMATING ARDL({p}, {', '.join([str(q_dict[v]) for v in EXOG_VARS])}) MODEL")
    print(f"{'='*60}")

    # Prepare data
    y = df[DEPENDENT]
    X = df[EXOG_VARS]

    # Build order tuple: (p, q_mpr, q_tbr, q_exo)
    # Must match the order of variables in EXOG_VARS
    order = {var: q_dict[var] for var in EXOG_VARS}

    print(f"\nDependent: {DEPENDENT} with {p} lags")
    print(f"Exogenous:")
    for var in EXOG_VARS:
        print(f"  {var} with {order[var]} lags")

    # Create ARDL model
    model = ARDL(
        endog=y,
        lags=p,
        exog=X,
        order=order,
        trend="c"  # Include constant term
    )

    # Fit the model
    print("\nFitting model...")
    result = model.fit()

    print("✓ Model estimation complete!")
    print(f"  Observations: {result.nobs}")
    print(f"  R-squared: {result.rsquared:.4f}")
    print(f"  Adjusted R-squared: {result.rsquared_adj:.4f}")

    # Display full regression output
    print("\n" + "="*60)
    print("FULL REGRESSION OUTPUT")
    print("="*60)
    print(result.summary())

    return result


def display_key_results(result):
    """
    Extract and display key results from ARDL estimation.

    Parameters:
        result: Fitted ARDL model result object
    """
    print(f"\n{'='*60}")
    print("KEY MODEL RESULTS")
    print("="*60)

    # Model fit statistics
    print("\nMODEL FIT:")
    print(f"  Observations:       {int(result.nobs)}")
    print(f"  R-squared:          {result.rsquared:.4f}")
    print(f"  Adjusted R-squared: {result.rsquared_adj:.4f}")
    print(f"  AIC:                {result.aic:.2f}")
    print(f"  BIC:                {result.bic:.2f}")
    print(f"  Log-Likelihood:     {result.llf:.2f}")

    # Coefficient table
    print("\nCOEFFICIENT TABLE:")
    print("-" * 85)
    print(f"{'Variable':<15} {'Coefficient':>12} {'Std Error':>12} {'t-stat':>10} {'p-value':>10} {'Sig':>5}")
    print("-" * 85)

    # Extract coefficients, standard errors, t-stats, p-values
    params = result.params
    std_err = result.bse
    t_stats = result.tvalues
    p_values = result.pvalues

    for var in params.index:
        coef = params[var]
        se = std_err[var]
        t_stat = t_stats[var]
        p_val = p_values[var]

        # Determine significance stars
        if p_val < 0.01:
            sig = "***"
        elif p_val < 0.05:
            sig = "**"
        elif p_val < 0.10:
            sig = "*"
        else:
            sig = ""

        print(f"{var:<15} {coef:>12.4f} {se:>12.4f} {t_stat:>10.3f} {p_val:>10.4f} {sig:>5}")

    print("-" * 85)
    print("Significance levels: *** p<0.01, ** p<0.05, * p<0.10")
    print("="*60)


def save_results(result):
    """
    Save ARDL coefficients to CSV file.

    Parameters:
        result: Fitted ARDL model result object
    """
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print("="*60)

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Create DataFrame with all coefficient information
    coef_df = pd.DataFrame({
        "variable": result.params.index,
        "coefficient": result.params.values,
        "std_error": result.bse.values,
        "t_statistic": result.tvalues.values,
        "p_value": result.pvalues.values,
        "conf_int_lower": result.conf_int()[0].values,
        "conf_int_upper": result.conf_int()[1].values
    })

    # Add significance indicator
    coef_df["significant"] = coef_df["p_value"] < 0.05

    # Save to CSV
    coef_df.to_csv(COEF_OUTPUT_PATH, index=False)

    print(f"✓ Coefficients saved to: {COEF_OUTPUT_PATH}")
    print(f"  Variables saved: {len(coef_df)}")
    print(f"  Significant variables: {coef_df['significant'].sum()}")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("ARDL MODEL FOR NIGERIAN INFLATION")
    print("="*60)

    # Load data
    df = load_data()

    # Select optimal lag order
    order_result = select_ardl_order(df, maxlag=12, ic="aic")
    p = order_result["p"]
    q_dict = order_result["q"]

    # Estimate ARDL model
    ardl_result = estimate_ardl(df, p, q_dict)

    # Display key results in clean format
    display_key_results(ardl_result)

    # Save results to CSV
    save_results(ardl_result)

    print("\n" + "="*60)
    print("ESTIMATION COMPLETE")
    print("="*60)
    print("Next steps:")
    print("  1. Review coefficient table above")
    print("  2. Open results/ardl_coefficients.csv in Excel/Sheets")
    print("  3. On Day 13, you'll compute long-run elasticities")
```

**What's new in this step**:

1. **`save_results()` function**:
   - Creates a DataFrame with all coefficient information
   - Includes confidence intervals (the range where the true coefficient likely lies)
   - Adds a `significant` column (True if p < 0.05)
   - Saves to `results/ardl_coefficients.csv`

2. **Results directory**: Automatically creates the `results/` folder if it doesn't exist

3. **Complete workflow**: The main block now runs all four steps:
   - Select optimal lags
   - Estimate the model
   - Display key results
   - Save coefficients to CSV

**Run it**:

```bash
python econometric_models/ardl_model.py
```

**Expected additional output**:

```
============================================================
SAVING RESULTS
============================================================
✓ Coefficients saved to: /home/user/Nigerian_Inflation_Predictor/results/ardl_coefficients.csv
  Variables saved: 14
  Significant variables: 3
============================================================

============================================================
ESTIMATION COMPLETE
============================================================
Next steps:
  1. Review coefficient table above
  2. Open results/ardl_coefficients.csv in Excel/Sheets
  3. On Day 13, you'll compute long-run elasticities
```

**Check your results file**:

```bash
cat results/ardl_coefficients.csv
```

You'll see all coefficients with their statistics in CSV format, ready for reporting or further analysis.

---

## Understanding the Coefficient Table

Let's decode what you're seeing in the coefficient table.

### Variable Names

- **`const`**: The intercept—the baseline inflation rate when all other variables are zero
- **`L1.infl`, `L2.infl`, etc.**: Lagged inflation values
  - `L1` = 1 month ago (lag 1)
  - `L2` = 2 months ago (lag 2)
  - `L3` = 3 months ago (lag 3), and so on
- **`mpr`**: Current month's Monetary Policy Rate
- **`L1.mpr`, `L2.mpr`**: MPR from 1 month ago, 2 months ago, etc.
- **`tbr`**, **`L1.tbr`**: Treasury Bill Rate (current and lagged)
- **`exo`**, **`L1.exo`, etc.**: Exchange rate (current and lagged)

### Reading a Coefficient Row

Let's say you see:

```
mpr                  0.1234       0.0450      2.742     0.0070   ***
```

This means:

1. **Coefficient = 0.1234**: A 1 percentage point increase in MPR is associated with a 0.1234 percentage point increase in inflation this month
2. **Std Error = 0.0450**: The uncertainty in this estimate is ±0.045
3. **t-statistic = 2.742**: The coefficient is 2.742 standard errors away from zero (the bigger, the more "real" the effect)
4. **p-value = 0.0070**: There's only a 0.7% chance this relationship happened by random chance (very strong evidence!)
5. ***** = Very significant (p < 0.01)

### Statistical Significance

- **p < 0.01 (***)***: Very strong evidence—you can be highly confident this variable matters
- **p < 0.05 (**)***: Strong evidence—the conventional threshold for "significant"
- **p < 0.10 (*)***: Weak evidence—suggestive but not conclusive
- **p > 0.10 (no star)***: Not significant—could just be noise

**Important**: Only interpret coefficients that are statistically significant!

---

## Critical Interpretation Warning

### The Monetary Policy Paradox

If you see a **positive coefficient on MPR** (e.g., `mpr = 0.1234 ***`), your first instinct might be:

> "Higher interest rates cause MORE inflation? That doesn't make sense!"

**BUT WAIT!** This is where understanding **correlation vs. causation** is crucial.

### What's Really Happening

The Central Bank of Nigeria **raises MPR in response to high inflation**. They're reacting, not causing. So:

1. Inflation goes up (e.g., from supply shocks, exchange rate depreciation)
2. CBN sees this and raises MPR to combat it
3. Your data shows: high MPR and high inflation occur together
4. The ARDL model picks up this correlation

**The model shows association, not causation.**

### The Solution: Long-Run Analysis

On Day 13, you'll calculate **long-run coefficients** and **error correction terms**. These separate:

- **Short-run effects**: The immediate reaction (what you see now—positive because CBN reacts to inflation)
- **Long-run effects**: The ultimate impact after all adjustments play out (likely negative—higher MPR eventually reduces inflation)

**For now**: Just note which variables are significant. Don't draw strong causal conclusions yet!

---

## Commit Your Progress

Stage and commit your expanded ARDL model:

```bash
git add econometric_models/ardl_model.py results/ardl_coefficients.csv
git commit -m "Add ARDL estimation, display, and save functions

- Implement estimate_ardl() to fit model and show full output
- Add display_key_results() for clean coefficient table
- Add save_results() to export coefficients to CSV
- Complete workflow: select → estimate → display → save
- 146 observations, R² ≈ 0.82, 3 significant variables

Day 12: ARDL Model Estimation"
```

---

## Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: endog and exog must have same length` | Data lengths don't match (missing values) | Check that all columns have the same number of rows; review data cleaning from Day 7-8 |
| `LinAlgError: Singular matrix` | Perfect collinearity between variables | Remove duplicate or perfectly correlated variables; check for data entry errors |
| `KeyError: 'L1.infl'` | Result object doesn't have expected lags | Verify that `p > 0` in your lag selection; check that model actually fitted |
| `FileNotFoundError: [Errno 2] No such file or directory: 'results/...'` | Results directory doesn't exist | The `save_results()` function should create it automatically; check that `os.makedirs()` is running |
| Model estimation hangs/freezes | Too many lags (e.g., maxlag=24) creates huge search space | Reduce `maxlag` to 12 or use `ic="bic"` (BIC penalizes complexity more) |
| R² is very low (< 0.3) | Model doesn't fit data well | Check data quality; verify variable relationships; consider data transformations |
| All p-values > 0.05 | No variables are significant | May indicate data issues, wrong model specification, or genuinely weak relationships |

---

## Check Your Understanding

### Question 1: Reading Coefficients

You see this in your coefficient table:

```
L1.infl              0.7500       0.0800      9.375     0.0000   ***
```

**What does this mean in plain English?**

<details>
<summary>Click to see answer</summary>

**Answer**:

Last month's inflation strongly predicts this month's inflation. Specifically:

- If inflation was 1 percentage point higher last month, we expect it to be 0.75 percentage points higher this month, all else equal
- The coefficient is 0.75, meaning 75% of last month's inflation "persists" into this month
- The p-value is essentially 0 (p < 0.0001), so we're extremely confident this relationship is real, not random
- The t-statistic of 9.375 is very high, indicating a very strong effect

**Economic interpretation**: Nigerian inflation has high persistence—once it rises, it tends to stay elevated. This is why the CBN needs to act quickly and decisively to control inflation before it becomes entrenched.

</details>

---

### Question 2: Significance Testing

Your model shows:

```
mpr                  0.1200       0.0450      2.667     0.0085   ***
L1.mpr               0.0500       0.0460      1.087     0.2790
tbr                 -0.0300       0.0400     -0.750     0.4545
```

**Which of these variables should you focus on when interpreting the model? Why?**

<details>
<summary>Click to see answer</summary>

**Answer**:

**Focus only on `mpr` (the current MPR)** because it's the only statistically significant variable among these three.

- **`mpr`**: p = 0.0085 < 0.05 → significant (***)
- **`L1.mpr`**: p = 0.2790 > 0.05 → NOT significant (could be random noise)
- **`tbr`**: p = 0.4545 > 0.05 → NOT significant (could be random noise)

**Why ignore non-significant variables?**

When p > 0.05, there's a >5% chance the relationship you see is just random variation, not a real pattern. The coefficient might be non-zero in your sample, but you can't confidently say it's non-zero in the population.

**Practical implication**:

The current MPR matters for inflation, but what MPR was 1 month ago doesn't seem to add additional explanatory power. Treasury bill rates don't appear to significantly affect inflation after controlling for MPR and other variables.

</details>

---

### Question 3: Model Fit

Your ARDL model reports:

```
Observations:       146
R-squared:          0.8234
Adjusted R-squared: 0.8076
```

**What does the R-squared tell you? Is this a good model?**

<details>
<summary>Click to see answer</summary>

**Answer**:

**R² = 0.8234 means your model explains 82.34% of the variation in inflation.** The remaining 17.66% is due to factors not in your model (other economic variables, measurement error, truly random shocks, etc.).

**Is this good?**

**Yes, very good!** For economic time series data:
- R² > 0.70 is generally considered good
- R² > 0.80 is excellent
- R² > 0.90 might actually be suspicious (possible overfitting)

**Why adjusted R²?**

- **Regular R²** always increases when you add more variables (even useless ones)
- **Adjusted R²** (0.8076) penalizes you for adding variables that don't improve the model much
- Since adjusted R² is only slightly lower than regular R² (0.8234 vs 0.8076), your variables are genuinely useful, not just inflating the fit

**Context**: Inflation is influenced by many global and local factors (oil prices, food supply, expectations, etc.). Explaining 82% with just three monetary policy variables is impressive and suggests these tools are indeed important for understanding Nigerian inflation dynamics.

</details>

---

## What You Built Today

Today you crossed a major milestone—you estimated your first econometric model! Here's what you accomplished:

1. ✅ **Estimated an ARDL model** using the optimal lag structure from Day 11
2. ✅ **Ran your first regression** and generated professional statistical output
3. ✅ **Interpreted coefficients, p-values, and R²** to understand model fit and variable significance
4. ✅ **Extracted key results** into a clean, readable format
5. ✅ **Saved coefficients to CSV** for documentation and reporting
6. ✅ **Learned to read regression tables** like an econometrician

### Your Project Structure Now

```
Nigerian_Inflation_Predictor/
├── econometric_models/
│   └── ardl_model.py          # Complete: select lags → estimate → display → save
├── results/
│   └── ardl_coefficients.csv  # NEW: All coefficients with statistics
└── [other files...]
```

### What You've Learned

- **ARDL estimation**: How to fit a distributed lag model to time series data
- **Regression output**: How to read summary tables with coefficients, t-stats, p-values
- **Statistical significance**: What p-values mean and how to use significance stars
- **Model fit**: How to interpret R² and assess model quality
- **Correlation vs. causation**: Why a positive MPR coefficient doesn't mean rates cause inflation

### Coming Up on Day 13

Tomorrow you'll unlock the **economic meaning** of these coefficients:

- **Long-run multipliers**: What's the total effect of MPR on inflation after all lags play out?
- **Error correction model**: How fast does inflation return to equilibrium after a shock?
- **Impulse response**: If the CBN raises MPR by 1%, what happens to inflation over the next 12 months?

These analyses will separate short-run reactions from long-run impacts—the key to understanding whether monetary policy actually works!

---

**You're doing great!** You've built a real econometric model from scratch. Tomorrow you'll interpret what it all means for Nigerian monetary policy.
