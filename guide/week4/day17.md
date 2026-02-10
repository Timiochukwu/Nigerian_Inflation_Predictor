# DAY 17 — Granger Causality: Does MPR "Cause" Inflation?

## What You'll Learn Today

By the end of today, you will:

- Understand what Granger causality means (and critically, what it does NOT mean)
- Learn how to test all pairwise Granger causality relationships in your VAR model
- Interpret Granger causality results for Nigerian monetary policy
- Understand whether MPR changes predictably lead to inflation changes
- Save and document your causality test results

## Theory: What is Granger Causality?

### The Concept

Granger causality is a statistical concept of causality based on prediction:

**Variable X "Granger-causes" variable Y if:**
- Past values of X help predict Y
- This prediction is better than using only past values of Y alone

### What Granger Causality IS

- A test of **predictive precedence**
- A way to determine if one time series contains useful information for forecasting another
- Based on temporal ordering: X must come before Y in time
- Useful for understanding policy transmission mechanisms

### What Granger Causality IS NOT

- **NOT true causality** in the philosophical or experimental sense
- Does not prove that X actually causes Y
- Does not account for confounding variables (Z might cause both X and Y)
- Does not imply a structural or economic causal relationship

### Why This Matters for Nigerian Monetary Policy

The Central Bank of Nigeria (CBN) uses the Monetary Policy Rate (MPR) as its primary tool to control inflation. But does it work?

**Key Questions:**
1. Does MPR Granger-cause inflation? If yes, CBN rate changes predictably affect prices
2. Does MPR Granger-cause TBR? If yes, policy transmission to money markets works
3. Does inflation Granger-cause MPR? If yes, CBN responds to inflation (reactive policy)
4. What about exchange rates? Do they Granger-cause inflation (pass-through effect)?

### The Test Statistic

The Granger causality test uses an F-test:

**Null Hypothesis (H0):** X does NOT Granger-cause Y
**Alternative (H1):** X does Granger-cause Y

- **F-statistic:** Measures the joint significance of lagged X terms in predicting Y
- **p-value:** Probability of observing this F-statistic if H0 is true
- **Decision rule:** If p-value < 0.05, reject H0 and conclude X Granger-causes Y

### All Pairwise Tests

With 4 variables (mpr, tbr, exo, infl), we have 12 directional tests:

1. MPR → TBR (does policy rate affect Treasury Bill rate?)
2. MPR → EXO (does policy rate affect exchange rate?)
3. MPR → INF (does policy rate affect inflation?)
4. TBR → MPR (does market rate influence policy decisions?)
5. TBR → EXO (does market rate affect exchange rate?)
6. TBR → INF (does market rate affect inflation?)
7. EXO → MPR (does exchange rate influence policy decisions?)
8. EXO → TBR (does exchange rate affect market rate?)
9. EXO → INF (does exchange rate affect inflation - pass-through?)
10. INF → MPR (does inflation drive policy changes - reactive?)
11. INF → TBR (does inflation affect market rates?)
12. INF → EXO (does inflation affect exchange rate?)

## Building: Expand econometric_models/var_model.py

We will add Granger causality testing to your existing VAR model code.

---

### STEP 1: Add Granger Causality Test Function

**Delete everything in `econometric_models/var_model.py` and replace it with this:**

```python
"""
VAR (Vector Autoregression) Model for Nigerian Inflation Prediction
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """
    Load cleaned data for VAR modeling
    """
    df = pd.read_csv('data/processed/cleaned_data.csv', index_col=0, parse_dates=True)
    return df


def prepare_var_data(df, columns=['mpr', 'tbr', 'exo', 'infl']):
    """
    Prepare data for VAR model - extract and order variables
    """
    var_data = df[columns].copy()
    print(f"VAR data shape: {var_data.shape}")
    print(f"Date range: {var_data.index.min()} to {var_data.index.max()}")
    print(f"\nVariable ordering: {' → '.join(columns)}")
    return var_data


def check_stationarity(data):
    """
    Check stationarity for all variables using ADF test
    """
    print("\n" + "="*60)
    print("STATIONARITY CHECK (ADF Test)")
    print("="*60)

    results = {}
    for col in data.columns:
        adf_result = adfuller(data[col].dropna())
        results[col] = {
            'ADF Statistic': adf_result[0],
            'p-value': adf_result[1],
            'Stationary': adf_result[1] < 0.05
        }

        print(f"\n{col.upper()}:")
        print(f"  ADF Statistic: {adf_result[0]:.4f}")
        print(f"  p-value: {adf_result[1]:.4f}")
        print(f"  Stationary: {'YES' if adf_result[1] < 0.05 else 'NO'}")

    return pd.DataFrame(results).T


def difference_data(data):
    """
    First difference the data to achieve stationarity
    """
    data_diff = data.diff().dropna()
    print(f"\nData after differencing: {data_diff.shape}")
    return data_diff


def select_lag_order(data, maxlags=12):
    """
    Select optimal lag order using information criteria
    """
    print("\n" + "="*60)
    print("LAG ORDER SELECTION")
    print("="*60)

    model = VAR(data)
    lag_order = model.select_order(maxlags=maxlags)
    print(lag_order.summary())

    return lag_order


def estimate_var_model(data, lags=4):
    """
    Estimate VAR model with specified lag order
    """
    print("\n" + "="*60)
    print(f"ESTIMATING VAR MODEL (Lag order: {lags})")
    print("="*60)

    model = VAR(data)
    result = model.fit(lags)

    print(f"\nVAR model estimated successfully")
    print(f"Number of equations: {result.neqs}")
    print(f"Number of observations: {result.nobs}")
    print(f"Lag order: {result.k_ar}")

    return result


def print_var_summary(result):
    """
    Print VAR model summary
    """
    print("\n" + "="*60)
    print("VAR MODEL SUMMARY")
    print("="*60)
    print(result.summary())


def plot_var_results(result, save_path='results/var_results.png'):
    """
    Plot VAR results: coefficients and residuals
    """
    fig = result.plot()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVAR results plot saved to {save_path}")
    plt.close()


def forecast_var(result, steps=12):
    """
    Forecast future values using VAR model
    """
    print("\n" + "="*60)
    print(f"FORECASTING {steps} STEPS AHEAD")
    print("="*60)

    lag_order = result.k_ar
    forecast = result.forecast(result.endog[-lag_order:], steps=steps)

    forecast_df = pd.DataFrame(
        forecast,
        columns=result.names,
        index=pd.date_range(
            start=result.endog_names[0],
            periods=steps,
            freq='M'
        )
    )

    print(f"\nForecast shape: {forecast_df.shape}")
    print(forecast_df)

    return forecast_df


def granger_causality_tests(result, maxlag=4):
    """
    Perform pairwise Granger causality tests for all variable combinations

    Tests whether variable X Granger-causes variable Y for all pairs.
    Reports F-statistic, p-value, and significance at 5% level.
    """
    print("\n" + "="*70)
    print("GRANGER CAUSALITY TESTS")
    print("="*70)
    print("\nNull Hypothesis (H0): [Causing variable] does NOT Granger-cause [Caused variable]")
    print("Reject H0 if p-value < 0.05")
    print("-"*70)

    variables = result.names
    causality_results = []

    # Test all pairwise combinations (X → Y)
    for caused in variables:
        for causing in variables:
            if caused != causing:  # Don't test a variable causing itself
                # Perform Granger causality test
                test_result = result.test_causality(caused=caused, causing=causing, kind='f')

                # Extract test statistics
                f_stat = test_result.test_statistic
                p_value = test_result.pvalue
                significant = "YES" if p_value < 0.05 else "NO"

                # Store results
                causality_results.append({
                    'Causing': causing.upper(),
                    'Caused': caused.upper(),
                    'F-statistic': f_stat,
                    'p-value': p_value,
                    'Significant (5%)': significant
                })

                # Print formatted result
                arrow = "→"
                print(f"{causing.upper()} {arrow} {caused.upper():<4} | F-stat: {f_stat:8.4f} | p-value: {p_value:.4f} | Significant: {significant}")

    print("-"*70)
    print(f"\nTotal tests performed: {len(causality_results)}")
    print(f"Significant relationships (p < 0.05): {sum(1 for r in causality_results if r['Significant (5%)'] == 'YES')}")

    # Convert to DataFrame
    causality_df = pd.DataFrame(causality_results)

    return causality_df


def main():
    """
    Main execution function
    """
    print("\n" + "="*70)
    print("NIGERIAN INFLATION PREDICTOR - VAR MODEL WITH GRANGER CAUSALITY")
    print("="*70)

    # Load data
    df = load_data()

    # Prepare VAR data (MPR → TBR → EXO → INF ordering)
    var_data = prepare_var_data(df, columns=['mpr', 'tbr', 'exo', 'infl'])

    # Check stationarity
    stationarity_results = check_stationarity(var_data)

    # Difference data if needed
    var_data_diff = difference_data(var_data)

    # Check stationarity of differenced data
    print("\n--- Stationarity Check After Differencing ---")
    stationarity_diff = check_stationarity(var_data_diff)

    # Select lag order
    lag_order = select_lag_order(var_data_diff, maxlags=12)

    # Use AIC optimal lag (or default to 4)
    optimal_lag = lag_order.aic
    if optimal_lag > 8:
        optimal_lag = 4  # Cap at 4 for stability

    # Estimate VAR model
    result = estimate_var_model(var_data_diff, lags=optimal_lag)

    # Print summary
    print_var_summary(result)

    # Plot results
    plot_var_results(result)

    # Forecast
    forecast_df = forecast_var(result, steps=12)

    # Granger causality tests
    causality_df = granger_causality_tests(result, maxlag=optimal_lag)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
```

**What Changed:**

1. **New function: `granger_causality_tests(result, maxlag=4)`**
   - Takes the fitted VAR model result
   - Loops through all variable pairs (causing → caused)
   - Uses `result.test_causality()` to perform F-test
   - Extracts F-statistic and p-value
   - Determines significance at 5% level
   - Prints formatted table with clear arrows (MPR → INF)
   - Returns DataFrame with all results

2. **Updated `main()` function**
   - Calls `granger_causality_tests()` after forecasting
   - Stores results in `causality_df`

**Test your code:**

```bash
cd /home/user/Nigerian_Inflation_Predictor
python econometric_models/var_model.py
```

You should see a table like:

```
======================================================================
GRANGER CAUSALITY TESTS
======================================================================

Null Hypothesis (H0): [Causing variable] does NOT Granger-cause [Caused variable]
Reject H0 if p-value < 0.05
----------------------------------------------------------------------
MPR → TBR  | F-stat:  12.4567 | p-value: 0.0001 | Significant: YES
MPR → EXO  | F-stat:   2.3456 | p-value: 0.0732 | Significant: NO
MPR → INFL | F-stat:   3.8901 | p-value: 0.0156 | Significant: YES
TBR → MPR  | F-stat:   1.2345 | p-value: 0.3124 | Significant: NO
...
```

---

### STEP 2: Add Summary and Save Results

**Delete everything in `econometric_models/var_model.py` and replace it with this:**

```python
"""
VAR (Vector Autoregression) Model for Nigerian Inflation Prediction
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """
    Load cleaned data for VAR modeling
    """
    df = pd.read_csv('data/processed/cleaned_data.csv', index_col=0, parse_dates=True)
    return df


def prepare_var_data(df, columns=['mpr', 'tbr', 'exo', 'infl']):
    """
    Prepare data for VAR model - extract and order variables
    """
    var_data = df[columns].copy()
    print(f"VAR data shape: {var_data.shape}")
    print(f"Date range: {var_data.index.min()} to {var_data.index.max()}")
    print(f"\nVariable ordering: {' → '.join(columns)}")
    return var_data


def check_stationarity(data):
    """
    Check stationarity for all variables using ADF test
    """
    print("\n" + "="*60)
    print("STATIONARITY CHECK (ADF Test)")
    print("="*60)

    results = {}
    for col in data.columns:
        adf_result = adfuller(data[col].dropna())
        results[col] = {
            'ADF Statistic': adf_result[0],
            'p-value': adf_result[1],
            'Stationary': adf_result[1] < 0.05
        }

        print(f"\n{col.upper()}:")
        print(f"  ADF Statistic: {adf_result[0]:.4f}")
        print(f"  p-value: {adf_result[1]:.4f}")
        print(f"  Stationary: {'YES' if adf_result[1] < 0.05 else 'NO'}")

    return pd.DataFrame(results).T


def difference_data(data):
    """
    First difference the data to achieve stationarity
    """
    data_diff = data.diff().dropna()
    print(f"\nData after differencing: {data_diff.shape}")
    return data_diff


def select_lag_order(data, maxlags=12):
    """
    Select optimal lag order using information criteria
    """
    print("\n" + "="*60)
    print("LAG ORDER SELECTION")
    print("="*60)

    model = VAR(data)
    lag_order = model.select_order(maxlags=maxlags)
    print(lag_order.summary())

    return lag_order


def estimate_var_model(data, lags=4):
    """
    Estimate VAR model with specified lag order
    """
    print("\n" + "="*60)
    print(f"ESTIMATING VAR MODEL (Lag order: {lags})")
    print("="*60)

    model = VAR(data)
    result = model.fit(lags)

    print(f"\nVAR model estimated successfully")
    print(f"Number of equations: {result.neqs}")
    print(f"Number of observations: {result.nobs}")
    print(f"Lag order: {result.k_ar}")

    return result


def print_var_summary(result):
    """
    Print VAR model summary
    """
    print("\n" + "="*60)
    print("VAR MODEL SUMMARY")
    print("="*60)
    print(result.summary())


def plot_var_results(result, save_path='results/var_results.png'):
    """
    Plot VAR results: coefficients and residuals
    """
    fig = result.plot()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVAR results plot saved to {save_path}")
    plt.close()


def forecast_var(result, steps=12):
    """
    Forecast future values using VAR model
    """
    print("\n" + "="*60)
    print(f"FORECASTING {steps} STEPS AHEAD")
    print("="*60)

    lag_order = result.k_ar
    forecast = result.forecast(result.endog[-lag_order:], steps=steps)

    forecast_df = pd.DataFrame(
        forecast,
        columns=result.names,
        index=pd.date_range(
            start=result.endog_names[0],
            periods=steps,
            freq='M'
        )
    )

    print(f"\nForecast shape: {forecast_df.shape}")
    print(forecast_df)

    return forecast_df


def granger_causality_tests(result, maxlag=4):
    """
    Perform pairwise Granger causality tests for all variable combinations

    Tests whether variable X Granger-causes variable Y for all pairs.
    Reports F-statistic, p-value, and significance at 5% level.
    """
    print("\n" + "="*70)
    print("GRANGER CAUSALITY TESTS")
    print("="*70)
    print("\nNull Hypothesis (H0): [Causing variable] does NOT Granger-cause [Caused variable]")
    print("Reject H0 if p-value < 0.05")
    print("-"*70)

    variables = result.names
    causality_results = []

    # Test all pairwise combinations (X → Y)
    for caused in variables:
        for causing in variables:
            if caused != causing:  # Don't test a variable causing itself
                # Perform Granger causality test
                test_result = result.test_causality(caused=caused, causing=causing, kind='f')

                # Extract test statistics
                f_stat = test_result.test_statistic
                p_value = test_result.pvalue
                significant = "YES" if p_value < 0.05 else "NO"

                # Store results
                causality_results.append({
                    'Causing': causing.upper(),
                    'Caused': caused.upper(),
                    'F-statistic': f_stat,
                    'p-value': p_value,
                    'Significant (5%)': significant
                })

                # Print formatted result
                arrow = "→"
                print(f"{causing.upper()} {arrow} {caused.upper():<4} | F-stat: {f_stat:8.4f} | p-value: {p_value:.4f} | Significant: {significant}")

    print("-"*70)
    print(f"\nTotal tests performed: {len(causality_results)}")

    # Count significant relationships
    sig_count = sum(1 for r in causality_results if r['Significant (5%)'] == 'YES')
    print(f"Significant relationships (p < 0.05): {sig_count}")

    # Convert to DataFrame
    causality_df = pd.DataFrame(causality_results)

    return causality_df


def summarize_granger_results(causality_df):
    """
    Summarize Granger causality results by policy relevance
    """
    print("\n" + "="*70)
    print("GRANGER CAUSALITY SUMMARY - POLICY IMPLICATIONS")
    print("="*70)

    # Filter significant results
    significant = causality_df[causality_df['Significant (5%)'] == 'YES']

    print("\n--- MONETARY POLICY TRANSMISSION (MPR Effects) ---")
    mpr_effects = significant[significant['Causing'] == 'MPR']
    if len(mpr_effects) > 0:
        for _, row in mpr_effects.iterrows():
            print(f"  MPR → {row['Caused']}: F={row['F-statistic']:.4f}, p={row['p-value']:.4f}")
    else:
        print("  No significant MPR effects found")

    print("\n--- POLICY RESPONSIVENESS (What drives MPR changes?) ---")
    mpr_drivers = significant[significant['Caused'] == 'MPR']
    if len(mpr_drivers) > 0:
        for _, row in mpr_drivers.iterrows():
            print(f"  {row['Causing']} → MPR: F={row['F-statistic']:.4f}, p={row['p-value']:.4f}")
    else:
        print("  No significant MPR drivers found")

    print("\n--- INFLATION DYNAMICS (What predicts inflation?) ---")
    infl_drivers = significant[significant['Caused'] == 'INFL']
    if len(infl_drivers) > 0:
        for _, row in infl_drivers.iterrows():
            print(f"  {row['Causing']} → INFL: F={row['F-statistic']:.4f}, p={row['p-value']:.4f}")
    else:
        print("  No significant inflation drivers found")

    print("\n--- EXCHANGE RATE EFFECTS (Pass-through mechanism) ---")
    exo_to_infl = significant[(significant['Causing'] == 'EXO') & (significant['Caused'] == 'INFL')]
    if len(exo_to_infl) > 0:
        row = exo_to_infl.iloc[0]
        print(f"  EXO → INFL: F={row['F-statistic']:.4f}, p={row['p-value']:.4f}")
        print("  Exchange rate pass-through to inflation is significant")
    else:
        print("  No significant exchange rate pass-through found")

    print("\n" + "="*70)


def save_granger_results(causality_df, save_path='results/granger_causality.csv'):
    """
    Save Granger causality results to CSV
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save to CSV
    causality_df.to_csv(save_path, index=False)
    print(f"\nGranger causality results saved to {save_path}")

    # Also create a summary file
    summary_path = 'results/granger_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("GRANGER CAUSALITY TEST RESULTS - NIGERIAN INFLATION PREDICTOR\n")
        f.write("="*70 + "\n\n")

        f.write("SIGNIFICANT RELATIONSHIPS (p < 0.05):\n")
        f.write("-"*70 + "\n")

        significant = causality_df[causality_df['Significant (5%)'] == 'YES']
        for _, row in significant.iterrows():
            f.write(f"{row['Causing']} → {row['Caused']}: F={row['F-statistic']:.4f}, p={row['p-value']:.4f}\n")

        f.write("\n" + "="*70 + "\n")
        f.write(f"Total significant relationships: {len(significant)} out of {len(causality_df)}\n")
        f.write("="*70 + "\n")

    print(f"Granger causality summary saved to {summary_path}")


def main():
    """
    Main execution function
    """
    print("\n" + "="*70)
    print("NIGERIAN INFLATION PREDICTOR - VAR MODEL WITH GRANGER CAUSALITY")
    print("="*70)

    # Load data
    df = load_data()

    # Prepare VAR data (MPR → TBR → EXO → INF ordering)
    var_data = prepare_var_data(df, columns=['mpr', 'tbr', 'exo', 'infl'])

    # Check stationarity
    stationarity_results = check_stationarity(var_data)

    # Difference data if needed
    var_data_diff = difference_data(var_data)

    # Check stationarity of differenced data
    print("\n--- Stationarity Check After Differencing ---")
    stationarity_diff = check_stationarity(var_data_diff)

    # Select lag order
    lag_order = select_lag_order(var_data_diff, maxlags=12)

    # Use AIC optimal lag (or default to 4)
    optimal_lag = lag_order.aic
    if optimal_lag > 8:
        optimal_lag = 4  # Cap at 4 for stability

    # Estimate VAR model
    result = estimate_var_model(var_data_diff, lags=optimal_lag)

    # Print summary
    print_var_summary(result)

    # Plot results
    plot_var_results(result)

    # Forecast
    forecast_df = forecast_var(result, steps=12)

    # Granger causality tests
    causality_df = granger_causality_tests(result, maxlag=optimal_lag)

    # Summarize Granger results
    summarize_granger_results(causality_df)

    # Save results
    save_granger_results(causality_df)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nOutputs generated:")
    print("  - results/var_results.png")
    print("  - results/granger_causality.csv")
    print("  - results/granger_summary.txt")


if __name__ == "__main__":
    main()
```

**What Changed:**

1. **New function: `summarize_granger_results(causality_df)`**
   - Groups results by policy relevance
   - Shows MPR effects (policy transmission)
   - Shows MPR drivers (policy responsiveness)
   - Shows inflation drivers
   - Shows exchange rate pass-through

2. **New function: `save_granger_results(causality_df, save_path)`**
   - Saves full results to `results/granger_causality.csv`
   - Creates `results/granger_summary.txt` with significant relationships only
   - Creates results directory if needed

3. **Updated `main()` function**
   - Calls `summarize_granger_results()`
   - Calls `save_granger_results()`
   - Lists all output files

**Test your code:**

```bash
cd /home/user/Nigerian_Inflation_Predictor
python econometric_models/var_model.py
```

You should now see:
1. Full Granger causality table
2. Policy-focused summary
3. Two new files created:
   - `results/granger_causality.csv` (full results)
   - `results/granger_summary.txt` (significant relationships only)

**Check the saved files:**

```bash
cat results/granger_summary.txt
head results/granger_causality.csv
```

---

## Interpreting Your Results

### Expected Findings for Nigerian Monetary Policy

Based on economic theory and Nigerian context, here's what you might find:

#### 1. MPR → TBR (Strong, Significant)

**Expected: YES (p < 0.05)**

- When CBN changes MPR, Treasury Bill rates should follow
- This is the first step in policy transmission
- If NOT significant: Policy transmission is broken at the first step

#### 2. MPR → INF (Weak or Lagged)

**Expected: MAYBE (could be p > 0.05 at short lags, p < 0.05 at longer lags)**

- Monetary policy affects inflation with long and variable lags (6-18 months)
- In Nigeria, this link is often weak due to:
  - Supply-side inflation (food, fuel)
  - Exchange rate shocks
  - Fiscal dominance
- If NOT significant: MPR may not be the primary inflation driver

#### 3. INF → MPR (Strong, Significant)

**Expected: YES (p < 0.05)**

- CBN responds to inflation by raising/lowering MPR
- This is reactive monetary policy (Taylor rule)
- Nigeria follows inflation-targeting framework
- If significant: CBN is responsive to inflation

#### 4. EXO → INF (Strong, Significant)

**Expected: YES (p < 0.05)**

- Nigeria is import-dependent
- Naira depreciation raises import costs
- Exchange rate pass-through to consumer prices
- If significant: Exchange rate is a major inflation driver

#### 5. TBR → INF (Possible)

**Expected: MAYBE**

- Market interest rates affect credit and demand
- Higher rates → lower demand → lower inflation
- In Nigeria, credit channel is weak
- If NOT significant: Interest rate channel is not the main transmission mechanism

#### 6. MPR → EXO (Weak)

**Expected: MAYBE NOT**

- Higher rates should attract foreign capital → Naira appreciation
- In Nigeria, exchange rate is often driven by oil prices and CBN interventions
- If NOT significant: Exchange rate is exogenous to domestic policy

### What Does "Not Granger-Cause" Mean?

If MPR does NOT Granger-cause inflation (p > 0.05):

**This does NOT mean:**
- MPR has zero effect on inflation
- CBN should stop using MPR

**This DOES mean:**
- Past MPR changes don't help predict inflation beyond what inflation's own past predicts
- The effect may be too delayed to show up in the VAR
- Other factors (exchange rate, fiscal policy, supply shocks) dominate
- Policy transmission may be weak or broken

### Policy Implications

**Strong MPR → TBR:**
- Policy transmission to money markets works
- CBN has control over short-term rates

**Weak MPR → INF:**
- Monetary policy alone may not control inflation
- Need complementary policies (exchange rate stability, fiscal discipline)
- Supply-side reforms needed (food, energy)

**Strong EXO → INF:**
- Exchange rate stability is crucial for inflation control
- CBN's FX interventions matter more than MPR changes
- Import substitution policies could reduce pass-through

**Strong INF → MPR:**
- CBN is reactive (responds to inflation)
- Consider proactive policy (forward-looking)

---

## Commit Your Work

```bash
git add econometric_models/var_model.py results/granger_causality.csv results/granger_summary.txt results/var_results.png
git commit -m "Add Granger causality tests to VAR model

- Implement pairwise Granger causality testing for all 4 variables
- Test 12 directional relationships (MPR→INF, EXO→INF, etc.)
- Add policy-focused summary (transmission, responsiveness, pass-through)
- Save results to granger_causality.csv and granger_summary.txt
- Identify significant predictive relationships for Nigerian monetary policy

https://claude.ai/code/session_XXXXX"
```

---

## Common Errors and Fixes

### Error 1: "ValueError: The model has not been estimated"

**Problem:** Calling `result.test_causality()` on unfitted model

**Fix:**
```python
# Wrong
model = VAR(data)
model.test_causality(...)  # Model not fitted yet!

# Right
model = VAR(data)
result = model.fit(lags=4)  # Fit first
result.test_causality(...)  # Now call test_causality on result
```

### Error 2: "KeyError: 'mpr'"

**Problem:** Variable name mismatch (uppercase vs lowercase)

**Fix:**
```python
# Wrong
result.test_causality(caused='MPR', causing='INF')  # Wrong case

# Right
result.test_causality(caused='mpr', causing='infl')  # Use lowercase (as in your data)
```

Or check your actual variable names:
```python
print(result.names)  # Print actual variable names
```

### Error 3: Empty or all non-significant results

**Problem:** Model lag order too short or data issues

**Fix:**
```python
# Try longer lags
causality_df = granger_causality_tests(result, maxlag=8)  # Increase from 4 to 8

# Or check if you're using differenced data (you should be)
var_data_diff = var_data.diff().dropna()  # Use differenced data
result = estimate_var_model(var_data_diff, lags=4)
```

### Error 4: "FileNotFoundError: results/granger_causality.csv"

**Problem:** Results directory doesn't exist

**Fix:**
```python
import os
os.makedirs('results', exist_ok=True)  # Create directory first
causality_df.to_csv('results/granger_causality.csv', index=False)
```

Already included in the code above!

---

## Practice Questions

### Q1: What is the difference between Granger causality and true causality?

**Answer:**

**Granger causality:**
- Statistical concept based on prediction
- X Granger-causes Y if past X helps predict Y
- Does NOT prove X actually causes Y
- Could be spurious (both driven by Z)

**True causality:**
- X directly influences Y through a mechanism
- Requires experimental or structural evidence
- Accounts for confounders
- Implies counterfactual: if X changed, Y would change

**Example:**
- Ice cream sales Granger-cause drowning deaths (both peak in summer)
- But ice cream doesn't cause drowning (both caused by hot weather)

### Q2: Why might MPR NOT Granger-cause inflation in Nigeria?

**Answer:**

1. **Long and variable lags:** Policy affects inflation after 6-18 months, may not show in 4-lag VAR
2. **Supply-side inflation:** Food and fuel prices drive Nigerian inflation (not demand-side)
3. **Weak transmission:** Interest rate channel is weak (low credit penetration)
4. **Exchange rate dominance:** Naira depreciation drives inflation more than MPR
5. **Fiscal dominance:** Government spending and deficits overwhelm monetary policy
6. **Structural factors:** Infrastructure, insecurity, poor distribution networks

### Q3: If EXO → INF is significant but MPR → INF is not, what does this mean for CBN?

**Answer:**

**Implication:**
- Exchange rate is the primary inflation channel
- MPR changes don't predictably affect prices
- CBN should focus on exchange rate stability rather than interest rates alone

**Policy recommendations:**
1. Coordinate MPR and FX policy
2. Build foreign reserves to defend Naira
3. Reduce import dependence (domestic production)
4. Implement structural reforms to reduce pass-through

**Caveat:**
- MPR may still affect inflation through EXO (MPR → EXO → INF)
- Need to check MPR → EXO relationship too

### Q4: What does it mean if both MPR → INF and INF → MPR are significant?

**Answer:**

**Interpretation:**
- Bidirectional feedback loop
- CBN responds to inflation (INF → MPR)
- MPR affects inflation (MPR → INF)
- Dynamic interaction between policy and target

**Example:**
1. Inflation rises to 18%
2. CBN raises MPR from 14% to 16% (INF → MPR)
3. Higher MPR slows credit and demand
4. Inflation falls to 16% (MPR → INF)
5. CBN lowers MPR to 15% (INF → MPR)

**Implication:**
- Active monetary policy regime
- Policy is both effective (MPR → INF) and responsive (INF → MPR)

### Q5: How do I interpret the F-statistic in Granger causality tests?

**Answer:**

**F-statistic:**
- Measures joint significance of lagged X terms in predicting Y
- Higher F-statistic = stronger predictive relationship
- Compare to critical value or use p-value

**Interpretation:**
- **Large F (e.g., 15.0) + small p (e.g., 0.0001):** Very strong evidence X Granger-causes Y
- **Medium F (e.g., 3.5) + small p (e.g., 0.04):** Moderate evidence, barely significant
- **Small F (e.g., 1.2) + large p (e.g., 0.35):** No evidence, fail to reject H0

**Example:**
```
MPR → INF: F=12.45, p=0.0001  → Strong Granger causality
TBR → INF: F=3.89, p=0.0456   → Weak but significant
EXO → TBR: F=1.23, p=0.3124   → Not significant
```

---

## What You Learned Today

1. **Granger causality concept:**
   - Predictive precedence, not true causality
   - X Granger-causes Y if past X helps predict Y
   - Uses F-test on lagged coefficients

2. **Testing all pairwise relationships:**
   - 4 variables → 12 directional tests (X → Y, Y → X are different)
   - Use `result.test_causality(caused, causing)`
   - Interpret F-statistic and p-value

3. **Policy implications:**
   - MPR → TBR: Policy transmission to money markets
   - MPR → INF: Policy effectiveness
   - INF → MPR: Policy responsiveness
   - EXO → INF: Exchange rate pass-through

4. **Nigerian context:**
   - Expect strong EXO → INF (import-dependent)
   - Expect weak MPR → INF (supply-side inflation)
   - Expect strong INF → MPR (reactive policy)

5. **Saving and documenting results:**
   - Save full results to CSV
   - Create summary file with significant relationships only
   - Organize by policy relevance

---

## Tomorrow: Day 18

**Topic: Impulse Response Functions (IRFs)**

We'll visualize how a shock to MPR affects inflation over time:
- What happens to inflation when CBN raises MPR by 1 percentage point?
- How long does the effect last?
- Does the effect peak immediately or after several months?

IRFs will bring your Granger causality results to life with dynamic plots showing policy impact paths.

See you tomorrow!
