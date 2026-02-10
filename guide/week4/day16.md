# Day 16 — VAR Model: Theory & Estimation

## What You'll Learn Today
- What a VAR (Vector Autoregression) is and how it differs from ARDL
- Why Cholesky ordering matters for identification
- How to estimate a VAR using statsmodels
- How to check model stability and interpret results

## Theory: Understanding VAR Models

### What is VAR?
A **Vector Autoregression (VAR)** is a multivariate time series model where ALL variables are treated as endogenous. Unlike ARDL (where we had one dependent variable), VAR treats all variables symmetrically—each variable is a function of its own lags AND the lags of all other variables in the system.

### ARDL vs VAR
| Feature | ARDL | VAR |
|---------|------|-----|
| Dependent variables | One (e.g., inflation) | All variables |
| Endogeneity | Only dependent variable | All variables are endogenous |
| Asymmetry | Explains Y with X | Symmetric system |
| Best for | Long-run relationships, ECM | IRFs, FEVD, Granger causality |

### The VAR System
For our four variables (MPR, TBR, EXO, INF), the VAR system is:

```
MPR_t = α₁ + Σ β₁ᵢ MPR_{t-i} + Σ γ₁ᵢ TBR_{t-i} + Σ δ₁ᵢ EXO_{t-i} + Σ θ₁ᵢ INF_{t-i} + ε₁t
TBR_t = α₂ + Σ β₂ᵢ MPR_{t-i} + Σ γ₂ᵢ TBR_{t-i} + Σ δ₂ᵢ EXO_{t-i} + Σ θ₂ᵢ INF_{t-i} + ε₂t
EXO_t = α₃ + Σ β₃ᵢ MPR_{t-i} + Σ γ₃ᵢ TBR_{t-i} + Σ δ₃ᵢ EXO_{t-i} + Σ θ₃ᵢ INF_{t-i} + ε₃t
INF_t = α₄ + Σ β₄ᵢ MPR_{t-i} + Σ γ₄ᵢ TBR_{t-i} + Σ δ₄ᵢ EXO_{t-i} + Σ θ₄ᵢ INF_{t-i} + ε₄t
```

Each equation has the SAME right-hand side variables (all lags of all variables).

### Why Use VAR?
VAR is ideal for:
1. **Impulse Response Functions (IRFs)**: Trace how a shock to one variable affects all others over time
2. **Forecast Error Variance Decomposition (FEVD)**: Attribute forecast variance to different shocks
3. **Granger Causality**: Test whether variable X helps predict variable Y
4. **Dynamic analysis**: Understand system-wide dynamics without imposing economic theory upfront

### Cholesky Ordering and Identification
VAR errors (ε₁t, ε₂t, ε₃t, ε₄t) can be contemporaneously correlated. To identify structural shocks for IRFs, we use **Cholesky decomposition**, which imposes a recursive ordering:

**Our ordering: MPR → TBR → EXO → INF**

This means:
1. **MPR is most exogenous**: The Central Bank of Nigeria sets MPR independently; it reacts to nothing within the same month
2. **TBR (Treasury Bill Rate)**: Reacts to MPR contemporaneously (monetary policy affects interbank rates), but not to EXO or INF
3. **EXO (Exchange Rate)**: Reacts to MPR and TBR contemporaneously, but not to INF
4. **INF (Inflation) is most endogenous**: Reacts to all other variables contemporaneously (inflation is slow-moving)

This ordering reflects economic intuition: policy variables (MPR) are fastest, inflation is slowest.

### Stationarity Requirement
VAR requires stationary variables. Since our variables have unit roots (we tested in Week 2), we estimate VAR on **first differences** (Δ variables). This is called a **VAR in differences**.

---

## Building econometric_models/var_model.py

We'll build this in 3 steps:
1. Load data, estimate VAR, select optimal lag
2. Add coefficient display and significance testing
3. Add stability check and save results

---

### STEP 1: Basic VAR Estimation

Delete everything in `econometric_models/var_model.py` and replace it with this:

```python
"""VAR model for the Nigerian Inflation Predictor."""
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

# Paths
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Cholesky ordering: MPR → TBR → EXO → INF
VARIABLE_ORDER = ["mpr", "tbr", "exo", "infl"]


def load_data():
    """Load cleaned data with variables in Cholesky order."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df[VARIABLE_ORDER]


def estimate_var(df, maxlags=12):
    """
    Estimate VAR on first differences.

    Parameters
    ----------
    df : pd.DataFrame
        Data with variables in Cholesky order
    maxlags : int
        Maximum lags to consider for lag selection

    Returns
    -------
    result : VARResults
        Fitted VAR model
    df_diff : pd.DataFrame
        First-differenced data used for estimation
    """
    # Take first differences (VAR requires stationarity)
    df_diff = df.diff().dropna()

    print("=" * 70)
    print("VAR MODEL ESTIMATION")
    print("=" * 70)
    print(f"\nVariables (Cholesky order): {VARIABLE_ORDER}")
    print(f"Sample size: {len(df_diff)} observations")
    print(f"Date range: {df_diff.index[0]} to {df_diff.index[-1]}")

    # Lag order selection
    model = VAR(df_diff)
    lag_order = model.select_order(maxlags=maxlags)
    print("\n" + "=" * 70)
    print("LAG ORDER SELECTION")
    print("=" * 70)
    print(lag_order.summary())

    # Use BIC (tends to choose more parsimonious models)
    optimal_lag = lag_order.bic
    print(f"\nUsing BIC-optimal lag: {optimal_lag}")

    # Fit VAR
    result = model.fit(optimal_lag)
    print("\n" + "=" * 70)
    print("VAR ESTIMATION RESULTS")
    print("=" * 70)
    print(result.summary())

    return result, df_diff


if __name__ == "__main__":
    df = load_data()
    result, df_diff = estimate_var(df)
```

**Run it:**
```bash
cd /home/user/Nigerian_Inflation_Predictor
python econometric_models/var_model.py
```

**What you'll see:**
- Lag order selection criteria (AIC, BIC, FPE, HQIC) for lags 1-12
- The BIC-optimal lag (usually 1-3 for monthly data)
- Full VAR regression results showing coefficients for each equation

**Interpretation:**
- If BIC selects lag = 2, then each variable depends on 2 lags of itself and 2 lags of all other variables
- That's 4 variables × 2 lags = 8 coefficients per equation (plus constant)
- With 4 equations, that's 36 total parameters (8 × 4 + 4 constants)

---

### STEP 2: Add Coefficient Display and Significance

Delete everything in `econometric_models/var_model.py` and replace it with this:

```python
"""VAR model for the Nigerian Inflation Predictor."""
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

# Paths
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Cholesky ordering: MPR → TBR → EXO → INF
VARIABLE_ORDER = ["mpr", "tbr", "exo", "infl"]


def load_data():
    """Load cleaned data with variables in Cholesky order."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df[VARIABLE_ORDER]


def estimate_var(df, maxlags=12):
    """
    Estimate VAR on first differences.

    Parameters
    ----------
    df : pd.DataFrame
        Data with variables in Cholesky order
    maxlags : int
        Maximum lags to consider for lag selection

    Returns
    -------
    result : VARResults
        Fitted VAR model
    df_diff : pd.DataFrame
        First-differenced data used for estimation
    """
    # Take first differences (VAR requires stationarity)
    df_diff = df.diff().dropna()

    print("=" * 70)
    print("VAR MODEL ESTIMATION")
    print("=" * 70)
    print(f"\nVariables (Cholesky order): {VARIABLE_ORDER}")
    print(f"Sample size: {len(df_diff)} observations")
    print(f"Date range: {df_diff.index[0]} to {df_diff.index[-1]}")

    # Lag order selection
    model = VAR(df_diff)
    lag_order = model.select_order(maxlags=maxlags)
    print("\n" + "=" * 70)
    print("LAG ORDER SELECTION")
    print("=" * 70)
    print(lag_order.summary())

    # Use BIC (tends to choose more parsimonious models)
    optimal_lag = lag_order.bic
    print(f"\nUsing BIC-optimal lag: {optimal_lag}")

    # Fit VAR
    result = model.fit(optimal_lag)
    print("\n" + "=" * 70)
    print("VAR ESTIMATION RESULTS")
    print("=" * 70)
    print(result.summary())

    return result, df_diff


def display_coefficient_matrices(result):
    """
    Display VAR coefficient matrices in a readable format.

    Shows which lags of which variables significantly affect each variable.
    """
    print("\n" + "=" * 70)
    print("COEFFICIENT MATRICES (by lag)")
    print("=" * 70)

    k = result.k_ar  # Number of lags
    neqs = result.neqs  # Number of equations

    # Get coefficients and p-values
    params = result.params.T  # Transpose so each row is an equation
    pvalues = result.pvalues.T

    for lag in range(1, k + 1):
        print(f"\n--- LAG {lag} ---")

        # Extract lag-specific coefficients
        lag_cols = [f"{var}.L{lag}" for var in VARIABLE_ORDER]

        # Create table
        table = []
        for i, eq_var in enumerate(VARIABLE_ORDER):
            row = [eq_var]
            for col in lag_cols:
                if col in params.columns:
                    coef = params.loc[eq_var, col]
                    pval = pvalues.loc[eq_var, col]
                    # Add significance stars
                    if pval < 0.01:
                        sig = "***"
                    elif pval < 0.05:
                        sig = "**"
                    elif pval < 0.10:
                        sig = "*"
                    else:
                        sig = ""
                    row.append(f"{coef:>8.4f}{sig:<3}")
                else:
                    row.append("    N/A")
            table.append(row)

        # Print table
        header = ["Equation"] + [f"{var}(-{lag})" for var in VARIABLE_ORDER]
        print(f"{'Equation':<10}", end="")
        for var in VARIABLE_ORDER:
            print(f"{var}(-{lag}):>12", end="")
        print()
        print("-" * 70)

        for row in table:
            print(f"{row[0]:<10}", end="")
            for val in row[1:]:
                print(f"{val:>12}", end="")
            print()

    print("\nSignificance: *** p<0.01, ** p<0.05, * p<0.10")
    print("\nInterpretation:")
    print("- Each row is an equation (dependent variable)")
    print("- Each column shows how lag k of a variable affects the equation")
    print("- Example: Row 'infl', Column 'mpr(-1)' shows effect of MPR lag 1 on inflation")


def interpret_var_equations(result):
    """
    Provide economic interpretation of key VAR equations.
    """
    print("\n" + "=" * 70)
    print("ECONOMIC INTERPRETATION")
    print("=" * 70)

    # Get significant coefficients (p < 0.05)
    params = result.params.T
    pvalues = result.pvalues.T

    for eq_var in VARIABLE_ORDER:
        print(f"\n{eq_var.upper()} EQUATION:")
        print(f"Change in {eq_var} depends on...")

        sig_vars = []
        for col in params.columns:
            if col == "const":
                continue
            pval = pvalues.loc[eq_var, col]
            if pval < 0.05:
                coef = params.loc[eq_var, col]
                sig_vars.append((col, coef, pval))

        if sig_vars:
            for var, coef, pval in sig_vars:
                direction = "increases" if coef > 0 else "decreases"
                print(f"  - {var}: {direction} {eq_var} (coef={coef:.4f}, p={pval:.4f})")
        else:
            print("  - No significant predictors at 5% level")


if __name__ == "__main__":
    df = load_data()
    result, df_diff = estimate_var(df)

    # Display detailed coefficient information
    display_coefficient_matrices(result)
    interpret_var_equations(result)
```

**Run it:**
```bash
python econometric_models/var_model.py
```

**What's new:**
- `display_coefficient_matrices()`: Shows coefficients for each lag in matrix form with significance stars
- `interpret_var_equations()`: Lists which lags significantly affect each variable
- You can now quickly see, for example, "Does MPR lag 1 significantly affect inflation?"

---

### STEP 3: Add Stability Check and Save Results

Delete everything in `econometric_models/var_model.py` and replace it with this:

```python
"""VAR model for the Nigerian Inflation Predictor."""
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import pickle

# Paths
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Cholesky ordering: MPR → TBR → EXO → INF
VARIABLE_ORDER = ["mpr", "tbr", "exo", "infl"]


def load_data():
    """Load cleaned data with variables in Cholesky order."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    return df[VARIABLE_ORDER]


def estimate_var(df, maxlags=12):
    """
    Estimate VAR on first differences.

    Parameters
    ----------
    df : pd.DataFrame
        Data with variables in Cholesky order
    maxlags : int
        Maximum lags to consider for lag selection

    Returns
    -------
    result : VARResults
        Fitted VAR model
    df_diff : pd.DataFrame
        First-differenced data used for estimation
    """
    # Take first differences (VAR requires stationarity)
    df_diff = df.diff().dropna()

    print("=" * 70)
    print("VAR MODEL ESTIMATION")
    print("=" * 70)
    print(f"\nVariables (Cholesky order): {VARIABLE_ORDER}")
    print(f"Sample size: {len(df_diff)} observations")
    print(f"Date range: {df_diff.index[0]} to {df_diff.index[-1]}")

    # Lag order selection
    model = VAR(df_diff)
    lag_order = model.select_order(maxlags=maxlags)
    print("\n" + "=" * 70)
    print("LAG ORDER SELECTION")
    print("=" * 70)
    print(lag_order.summary())

    # Use BIC (tends to choose more parsimonious models)
    optimal_lag = lag_order.bic
    print(f"\nUsing BIC-optimal lag: {optimal_lag}")

    # Fit VAR
    result = model.fit(optimal_lag)
    print("\n" + "=" * 70)
    print("VAR ESTIMATION RESULTS")
    print("=" * 70)
    print(result.summary())

    return result, df_diff


def display_coefficient_matrices(result):
    """
    Display VAR coefficient matrices in a readable format.

    Shows which lags of which variables significantly affect each variable.
    """
    print("\n" + "=" * 70)
    print("COEFFICIENT MATRICES (by lag)")
    print("=" * 70)

    k = result.k_ar  # Number of lags
    neqs = result.neqs  # Number of equations

    # Get coefficients and p-values
    params = result.params.T  # Transpose so each row is an equation
    pvalues = result.pvalues.T

    for lag in range(1, k + 1):
        print(f"\n--- LAG {lag} ---")

        # Extract lag-specific coefficients
        lag_cols = [f"{var}.L{lag}" for var in VARIABLE_ORDER]

        # Create table
        table = []
        for i, eq_var in enumerate(VARIABLE_ORDER):
            row = [eq_var]
            for col in lag_cols:
                if col in params.columns:
                    coef = params.loc[eq_var, col]
                    pval = pvalues.loc[eq_var, col]
                    # Add significance stars
                    if pval < 0.01:
                        sig = "***"
                    elif pval < 0.05:
                        sig = "**"
                    elif pval < 0.10:
                        sig = "*"
                    else:
                        sig = ""
                    row.append(f"{coef:>8.4f}{sig:<3}")
                else:
                    row.append("    N/A")
            table.append(row)

        # Print table
        header = ["Equation"] + [f"{var}(-{lag})" for var in VARIABLE_ORDER]
        print(f"{'Equation':<10}", end="")
        for var in VARIABLE_ORDER:
            print(f"{var}(-{lag}):>12", end="")
        print()
        print("-" * 70)

        for row in table:
            print(f"{row[0]:<10}", end="")
            for val in row[1:]:
                print(f"{val:>12}", end="")
            print()

    print("\nSignificance: *** p<0.01, ** p<0.05, * p<0.10")
    print("\nInterpretation:")
    print("- Each row is an equation (dependent variable)")
    print("- Each column shows how lag k of a variable affects the equation")
    print("- Example: Row 'infl', Column 'mpr(-1)' shows effect of MPR lag 1 on inflation")


def interpret_var_equations(result):
    """
    Provide economic interpretation of key VAR equations.
    """
    print("\n" + "=" * 70)
    print("ECONOMIC INTERPRETATION")
    print("=" * 70)

    # Get significant coefficients (p < 0.05)
    params = result.params.T
    pvalues = result.pvalues.T

    for eq_var in VARIABLE_ORDER:
        print(f"\n{eq_var.upper()} EQUATION:")
        print(f"Change in {eq_var} depends on...")

        sig_vars = []
        for col in params.columns:
            if col == "const":
                continue
            pval = pvalues.loc[eq_var, col]
            if pval < 0.05:
                coef = params.loc[eq_var, col]
                sig_vars.append((col, coef, pval))

        if sig_vars:
            for var, coef, pval in sig_vars:
                direction = "increases" if coef > 0 else "decreases"
                print(f"  - {var}: {direction} {eq_var} (coef={coef:.4f}, p={pval:.4f})")
        else:
            print("  - No significant predictors at 5% level")


def check_stability(result):
    """
    Check VAR stability condition.

    A VAR is stable if all eigenvalues of the companion matrix
    lie inside the unit circle. If stable, IRFs will converge.
    """
    print("\n" + "=" * 70)
    print("STABILITY CHECK")
    print("=" * 70)

    # Get roots (inverse of eigenvalues)
    roots = result.roots

    print(f"\nNumber of roots: {len(roots)}")
    print("\nRoots of characteristic polynomial:")
    for i, root in enumerate(roots, 1):
        modulus = np.abs(root)
        print(f"  Root {i}: {root:.4f} (modulus = {modulus:.4f})")

    # Check stability
    max_root = np.max(np.abs(roots))

    if max_root < 1.0:
        print(f"\n✓ VAR is STABLE: All roots have modulus < 1 (max = {max_root:.4f})")
        print("  → Impulse responses will converge to zero")
        print("  → Forecasts are meaningful")
        stable = True
    else:
        print(f"\n✗ VAR is UNSTABLE: Some roots have modulus ≥ 1 (max = {max_root:.4f})")
        print("  → Impulse responses may not converge")
        print("  → Consider reducing lag order or checking data")
        stable = False

    return stable


def save_results(result, df_diff):
    """Save VAR results for use in IRF and FEVD analysis."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save fitted model
    model_path = os.path.join(RESULTS_DIR, "var_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\n✓ Saved VAR model to {model_path}")

    # Save differenced data (needed for IRF)
    data_path = os.path.join(RESULTS_DIR, "var_data_diff.csv")
    df_diff.to_csv(data_path)
    print(f"✓ Saved differenced data to {data_path}")

    # Save summary statistics
    summary_path = os.path.join(RESULTS_DIR, "var_summary.txt")
    with open(summary_path, "w") as f:
        f.write("VAR MODEL SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Variables: {', '.join(VARIABLE_ORDER)}\n")
        f.write(f"Lag order: {result.k_ar}\n")
        f.write(f"Number of observations: {result.nobs}\n")
        f.write(f"AIC: {result.aic:.2f}\n")
        f.write(f"BIC: {result.bic:.2f}\n")
        f.write(f"Log-likelihood: {result.llf:.2f}\n\n")
        f.write(str(result.summary()))
    print(f"✓ Saved summary to {summary_path}")


if __name__ == "__main__":
    df = load_data()
    result, df_diff = estimate_var(df)

    # Display detailed coefficient information
    display_coefficient_matrices(result)
    interpret_var_equations(result)

    # Check stability
    stable = check_stability(result)

    # Save results
    if stable:
        save_results(result, df_diff)
        print("\n" + "=" * 70)
        print("VAR estimation complete! Next steps:")
        print("  - Day 17: Impulse Response Functions (IRFs)")
        print("  - Day 18: Forecast Error Variance Decomposition (FEVD)")
        print("=" * 70)
    else:
        print("\n⚠ Warning: Model is unstable. Review lag order or data.")
```

**Run it:**
```bash
python econometric_models/var_model.py
```

**What's new:**
- `check_stability()`: Computes eigenvalues of the companion matrix. If all are inside the unit circle (modulus < 1), the VAR is stable
- `save_results()`: Saves the fitted VAR model, differenced data, and summary to `results/` for use in IRF and FEVD analysis tomorrow
- Clear indication of whether the model is ready for IRF analysis

---

## Interpreting VAR Results

### Lag Order Selection
The output shows four criteria:
- **AIC** (Akaike Information Criterion): Penalizes complexity, tends to choose more lags
- **BIC** (Bayesian Information Criterion): Penalizes complexity more heavily, chooses fewer lags
- **FPE** (Final Prediction Error): Similar to AIC
- **HQIC** (Hannan-Quinn IC): Between AIC and BIC

**Rule of thumb**: Use BIC for parsimony. For monthly data, 1-3 lags is typical. For quarterly data, 4-8 lags.

### Coefficient Interpretation
Each equation has coefficients for all lags of all variables. Example:

```
INFL EQUATION:
  - mpr.L1: -0.0234 (p=0.03) **
  - infl.L1: 0.4521 (p=0.00) ***
```

**Interpretation:**
- A 1 percentage point increase in MPR last month reduces inflation change by 0.0234 pp this month
- A 1 pp increase in inflation last month leads to a 0.4521 pp increase in inflation change this month (persistence)

### Stability Condition
**Stable VAR**: All roots have modulus < 1. This means:
- Impulse responses converge to zero (shocks die out)
- Forecasts are reliable
- The system is not explosive

**Unstable VAR**: At least one root has modulus ≥ 1. This suggests:
- Model misspecification (wrong lag order)
- Data issues (remaining unit root)
- Consider re-checking stationarity or reducing lags

---

## Cholesky Ordering Justification

### Why MPR → TBR → EXO → INF?

1. **MPR (Monetary Policy Rate)**
   - Set by the CBN Monetary Policy Committee (MPC)
   - MPC meets every two months
   - Decision is based on past data, not contemporaneous values of TBR, EXO, or INF
   - **Most exogenous**: MPR does not react to anything within the same month

2. **TBR (Treasury Bill Rate)**
   - Market-determined (auction)
   - Reacts immediately to MPR changes (banks adjust their bids)
   - Does not react to EXO or INF contemporaneously (takes time)
   - **Second in order**: Reacts to MPR only

3. **EXO (Exchange Rate)**
   - Determined by forex market supply and demand
   - Reacts to MPR (interest rate parity) and TBR (capital flows) contemporaneously
   - Inflation affects exchange rate with a lag (purchasing power parity takes time)
   - **Third in order**: Reacts to MPR and TBR

4. **INF (Inflation)**
   - Slow-moving (sticky prices)
   - Reacts to all variables with lags
   - Does not react to MPR, TBR, or EXO within the same month
   - **Most endogenous**: Reacts to everything, but slowly

### Alternative Orderings
Different orderings imply different economic assumptions:
- **INF → MPR → TBR → EXO**: Assumes CBN reacts to inflation contemporaneously (inflation targeting)
- **EXO → MPR → TBR → INF**: Assumes exchange rate is most exogenous (small open economy)

For our analysis, **MPR → TBR → EXO → INF** reflects Nigeria's monetary policy framework: the CBN uses MPR as its main tool, and inflation responds slowly.

---

## Commit Your Work

```bash
git add econometric_models/var_model.py
git commit -m "Day 16: VAR model estimation with stability check

- Estimate VAR on first differences (stationary data)
- Cholesky ordering: MPR → TBR → EXO → INF
- Display coefficient matrices with significance stars
- Check model stability (all roots inside unit circle)
- Save results for IRF and FEVD analysis

https://claude.ai/code/session_xyz"
```

---

## Common Errors and Solutions

### Error 1: "ValueError: endog must not have nan or inf"
**Cause:** Missing values in data after differencing.

**Solution:**
```python
df_diff = df.diff().dropna()  # Already in our code
```

### Error 2: "All roots inside unit circle, but results seem wrong"
**Cause:** Lag order too low or too high.

**Solution:**
Try different lag selection criteria:
```python
# Compare all criteria
print("AIC selects:", lag_order.aic)
print("BIC selects:", lag_order.bic)
print("HQIC selects:", lag_order.hqic)

# Try AIC instead of BIC
result = model.fit(lag_order.aic)
```

### Error 3: "Model unstable: roots outside unit circle"
**Cause:** Data may still be non-stationary or lag order too high.

**Solutions:**
1. Check stationarity again (ADF test on differences)
2. Reduce lag order manually:
   ```python
   result = model.fit(2)  # Force lag=2
   ```
3. Check for outliers in data

### Error 4: "No significant coefficients in any equation"
**Cause:** Weak relationships, over-differencing, or too much noise.

**Solutions:**
1. Check data quality (outliers, measurement error)
2. Consider longer sample period
3. Review variable definitions (are they economically meaningful?)

---

## Q&A

**Q1: Why do we difference the data?**
A: VAR requires stationary variables. Our unit root tests (Week 2) showed MPR, TBR, EXO, and INF are I(1)—they become stationary after first differencing. Estimating VAR on levels (non-stationary data) would give spurious results.

**Q2: VAR vs VECM—when to use which?**
A:
- Use **VAR in differences** (what we did) if variables are I(1) but not cointegrated
- Use **VECM** (Vector Error Correction Model) if variables are I(1) AND cointegrated (Week 3)
- VECM = VAR + error correction term
- We'll compare them in Week 5

**Q3: Why not include more lags?**
A: More lags = more parameters = fewer degrees of freedom. BIC penalizes complexity, preventing overfitting. For monthly data, 12 lags means you're saying "what happened a year ago affects today," which may not be economically meaningful.

**Q4: What does "BIC selects 2 lags" mean?**
A: It means the BIC is minimized at lag=2. The model with 2 lags has the best balance between fit and parsimony.

**Q5: Can I use levels instead of differences?**
A: Only if all variables are stationary (I(0)) or if they're cointegrated (then use VECM). Using levels when variables are I(1) and not cointegrated leads to spurious regression.

**Q6: What if my model is unstable?**
A:
1. Check stationarity of differences (should all be I(0))
2. Try fewer lags
3. Check for data errors or outliers
4. Consider alternative specifications

**Q7: Do I need to test Granger causality before estimating VAR?**
A: No. VAR includes all variables regardless of Granger causality. Granger causality tests (Week 4, Day 18) are done AFTER VAR estimation to interpret the system.

**Q8: Why do some equations have no significant coefficients?**
A: Some variables may be weakly related or the sample size may be too small to detect effects. This is fine—VAR is still useful for IRFs and FEVD, which pool information across all equations.

**Q9: Should I remove insignificant lags?**
A: No. VAR is a system—removing lags from one equation but not others breaks the symmetry. If you want to simplify, reduce the lag order for ALL equations.

**Q10: How do I interpret the constant term?**
A: The constant is the average change (drift) in each variable. For inflation, a positive constant means inflation tends to increase over time. Often not economically interesting—we focus on lag coefficients.

---

## Summary

Today you learned:
1. **VAR treats all variables as endogenous**, unlike ARDL (one dependent variable)
2. **Cholesky ordering** (MPR → TBR → EXO → INF) imposes recursive contemporaneous restrictions for identification
3. **Lag selection**: Use BIC for parsimony; typical lag order is 1-3 for monthly data
4. **Stability**: All roots inside unit circle means IRFs converge and forecasts are valid
5. **Estimation**: Use first differences to ensure stationarity

**Next steps:**
- **Day 17**: Impulse Response Functions (how a shock to MPR affects inflation over time)
- **Day 18**: Forecast Error Variance Decomposition (how much of inflation variance is due to MPR shocks)
- **Day 19**: Granger Causality (does MPR Granger-cause inflation?)

---

## Further Reading

1. **Lütkepohl (2005)**: "New Introduction to Multiple Time Series Analysis" — the VAR bible
2. **Stock & Watson (2001)**: "Vector Autoregressions" — accessible review
3. **Sims (1980)**: "Macroeconomics and Reality" — seminal VAR paper
4. **Christiano, Eichenbaum, Evans (1999)**: "Monetary policy shocks: What have we learned and to what end?" — VAR for monetary policy

**Key takeaway**: VAR is a system approach. We don't interpret individual coefficients much; we use VAR as a tool for IRFs, FEVD, and causality tests. Tomorrow, we'll see VAR's real power: tracing shocks through the system.
