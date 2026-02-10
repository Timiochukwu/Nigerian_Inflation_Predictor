# DAY 23 — Model Comparison: ARDL vs VAR

## What You'll Learn Today

Today you'll compare the two main models you've built: the ARDL model (Week 3) and the VAR model (Week 4). You'll learn:

- How to compare models using in-sample fit metrics (R², AIC, BIC)
- How to compare models using out-of-sample forecast accuracy (RMSE, MAE)
- When to use ARDL vs VAR for different economic questions
- How to create comparison tables and visualizations

**Why This Matters**: Different models excel at different tasks. ARDL is powerful for long-run relationships and cointegration analysis, while VAR excels at capturing dynamic interactions and generating impulse response functions. Understanding which model to use is crucial for applied econometrics.

---

## Building econometric_models/model_comparison.py

You'll build this file in 3 steps, with each step showing the COMPLETE file.

---

### STEP 1: Load Data and Extract In-Sample Fit Metrics

First, we'll import libraries, load the data, re-estimate both ARDL and VAR models, and extract their in-sample fit metrics.

**Delete everything in econometric_models/model_comparison.py and replace it with this:**

```python
"""
Model Comparison: ARDL vs VAR
Compare the ARDL and VAR models using in-sample and out-of-sample metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, meanabs
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Load the data
data = pd.read_csv('data/processed/model_data.csv', index_col=0, parse_dates=True)

# Our key variables
mpr = data['mpr']       # Monetary Policy Rate
infl = data['infl']     # Inflation (our target)
exo = data['exo']       # Exchange Rate
tbr = data['tbr']       # Treasury Bill Rate

print("=" * 70)
print("MODEL COMPARISON: ARDL vs VAR")
print("=" * 70)
print(f"\nData period: {data.index[0]} to {data.index[-1]}")
print(f"Total observations: {len(data)}")

# ============================================================================
# PART 1: RE-ESTIMATE BOTH MODELS ON FULL DATA
# ============================================================================

print("\n" + "=" * 70)
print("PART 1: RE-ESTIMATING BOTH MODELS (FULL SAMPLE)")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1A. ARDL Model
# -----------------------------------------------------------------------------

print("\n[1] ARDL Model")
print("-" * 70)

# ARDL(2,1,1,1) - 2 lags of infl, 1 lag each of mpr, exo, tbr
ardl_model = ARDL(endog=infl, exog=data[['mpr', 'exo', 'tbr']],
                   lags=2, order=(1, 1, 1))
ardl_fit = ardl_model.fit()

# Extract in-sample metrics
ardl_rsquared = ardl_fit.rsquared
ardl_aic = ardl_fit.aic
ardl_bic = ardl_fit.bic
ardl_fitted = ardl_fit.fittedvalues
ardl_residuals = ardl_fit.resid

print(f"ARDL(2,1,1,1) Results:")
print(f"  R-squared:  {ardl_rsquared:.4f}")
print(f"  AIC:        {ardl_aic:.2f}")
print(f"  BIC:        {ardl_bic:.2f}")
print(f"  Obs:        {len(ardl_fitted)}")

# -----------------------------------------------------------------------------
# 1B. VAR Model
# -----------------------------------------------------------------------------

print("\n[2] VAR Model")
print("-" * 70)

# VAR with 2 lags (all variables treated symmetrically)
var_data = data[['infl', 'mpr', 'exo', 'tbr']]
var_model = VAR(var_data)
var_fit = var_model.fit(maxlags=2)

# Extract in-sample metrics for the inflation equation only
var_aic = var_fit.aic
var_bic = var_fit.bic

# Get fitted values for inflation (first variable)
var_fitted = var_fit.fittedvalues['infl']
var_residuals = var_fit.resid['infl']

# Calculate R² for inflation equation
ss_res = np.sum(var_residuals**2)
ss_tot = np.sum((infl[var_fit.k_ar:] - infl[var_fit.k_ar:].mean())**2)
var_rsquared = 1 - (ss_res / ss_tot)

print(f"VAR(2) Results (Inflation Equation):")
print(f"  R-squared:  {var_rsquared:.4f}")
print(f"  AIC:        {var_aic:.2f}")
print(f"  BIC:        {var_bic:.2f}")
print(f"  Obs:        {len(var_fitted)}")

# -----------------------------------------------------------------------------
# 1C. In-Sample Comparison
# -----------------------------------------------------------------------------

print("\n[3] In-Sample Fit Comparison")
print("-" * 70)

print(f"\n{'Metric':<15} {'ARDL':<15} {'VAR':<15} {'Winner':<15}")
print("-" * 60)

# R² (higher is better)
r2_winner = "ARDL" if ardl_rsquared > var_rsquared else "VAR"
print(f"{'R-squared':<15} {ardl_rsquared:<15.4f} {var_rsquared:<15.4f} {r2_winner:<15}")

# AIC (lower is better)
aic_winner = "ARDL" if ardl_aic < var_aic else "VAR"
print(f"{'AIC':<15} {ardl_aic:<15.2f} {var_aic:<15.2f} {aic_winner:<15}")

# BIC (lower is better)
bic_winner = "ARDL" if ardl_bic < var_bic else "VAR"
print(f"{'BIC':<15} {ardl_bic:<15.2f} {var_bic:<15.2f} {bic_winner:<15}")

print("\nInterpretation:")
print("- R²: Proportion of variance explained (higher = better fit)")
print("- AIC/BIC: Information criteria (lower = better, penalize complexity)")
print("- BIC penalizes model complexity more than AIC")
```

**What This Does:**

1. **Imports**: Brings in NumPy, pandas, matplotlib, and statsmodels tools
2. **Load Data**: Reads the processed model data with our four key variables
3. **Re-estimate ARDL**: Fits ARDL(2,1,1,1) on full sample, extracts R², AIC, BIC
4. **Re-estimate VAR**: Fits VAR(2) on full sample, extracts metrics for inflation equation
5. **Compare Metrics**: Creates a table showing which model wins on each metric

**Run it:**

```bash
cd /home/user/Nigerian_Inflation_Predictor
python econometric_models/model_comparison.py
```

You'll see in-sample fit metrics for both models. But in-sample fit can be misleading—a more complex model always fits the training data better. That's why we need out-of-sample testing.

---

### STEP 2: Add Out-of-Sample Forecast Comparison

Now we'll add the crucial out-of-sample test: hold out the last 12 months, estimate both models on the training set, forecast 12 steps ahead, and compare accuracy.

**Delete everything in econometric_models/model_comparison.py and replace it with this:**

```python
"""
Model Comparison: ARDL vs VAR
Compare the ARDL and VAR models using in-sample and out-of-sample metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, meanabs
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Load the data
data = pd.read_csv('data/processed/model_data.csv', index_col=0, parse_dates=True)

# Our key variables
mpr = data['mpr']       # Monetary Policy Rate
infl = data['infl']     # Inflation (our target)
exo = data['exo']       # Exchange Rate
tbr = data['tbr']       # Treasury Bill Rate

print("=" * 70)
print("MODEL COMPARISON: ARDL vs VAR")
print("=" * 70)
print(f"\nData period: {data.index[0]} to {data.index[-1]}")
print(f"Total observations: {len(data)}")

# ============================================================================
# PART 1: RE-ESTIMATE BOTH MODELS ON FULL DATA
# ============================================================================

print("\n" + "=" * 70)
print("PART 1: RE-ESTIMATING BOTH MODELS (FULL SAMPLE)")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1A. ARDL Model
# -----------------------------------------------------------------------------

print("\n[1] ARDL Model")
print("-" * 70)

# ARDL(2,1,1,1) - 2 lags of infl, 1 lag each of mpr, exo, tbr
ardl_model = ARDL(endog=infl, exog=data[['mpr', 'exo', 'tbr']],
                   lags=2, order=(1, 1, 1))
ardl_fit = ardl_model.fit()

# Extract in-sample metrics
ardl_rsquared = ardl_fit.rsquared
ardl_aic = ardl_fit.aic
ardl_bic = ardl_fit.bic
ardl_fitted = ardl_fit.fittedvalues
ardl_residuals = ardl_fit.resid

print(f"ARDL(2,1,1,1) Results:")
print(f"  R-squared:  {ardl_rsquared:.4f}")
print(f"  AIC:        {ardl_aic:.2f}")
print(f"  BIC:        {ardl_bic:.2f}")
print(f"  Obs:        {len(ardl_fitted)}")

# -----------------------------------------------------------------------------
# 1B. VAR Model
# -----------------------------------------------------------------------------

print("\n[2] VAR Model")
print("-" * 70)

# VAR with 2 lags (all variables treated symmetrically)
var_data = data[['infl', 'mpr', 'exo', 'tbr']]
var_model = VAR(var_data)
var_fit = var_model.fit(maxlags=2)

# Extract in-sample metrics for the inflation equation only
var_aic = var_fit.aic
var_bic = var_fit.bic

# Get fitted values for inflation (first variable)
var_fitted = var_fit.fittedvalues['infl']
var_residuals = var_fit.resid['infl']

# Calculate R² for inflation equation
ss_res = np.sum(var_residuals**2)
ss_tot = np.sum((infl[var_fit.k_ar:] - infl[var_fit.k_ar:].mean())**2)
var_rsquared = 1 - (ss_res / ss_tot)

print(f"VAR(2) Results (Inflation Equation):")
print(f"  R-squared:  {var_rsquared:.4f}")
print(f"  AIC:        {var_aic:.2f}")
print(f"  BIC:        {var_bic:.2f}")
print(f"  Obs:        {len(var_fitted)}")

# -----------------------------------------------------------------------------
# 1C. In-Sample Comparison
# -----------------------------------------------------------------------------

print("\n[3] In-Sample Fit Comparison")
print("-" * 70)

print(f"\n{'Metric':<15} {'ARDL':<15} {'VAR':<15} {'Winner':<15}")
print("-" * 60)

# R² (higher is better)
r2_winner = "ARDL" if ardl_rsquared > var_rsquared else "VAR"
print(f"{'R-squared':<15} {ardl_rsquared:<15.4f} {var_rsquared:<15.4f} {r2_winner:<15}")

# AIC (lower is better)
aic_winner = "ARDL" if ardl_aic < var_aic else "VAR"
print(f"{'AIC':<15} {ardl_aic:<15.2f} {var_aic:<15.2f} {aic_winner:<15}")

# BIC (lower is better)
bic_winner = "ARDL" if ardl_bic < var_bic else "VAR"
print(f"{'BIC':<15} {ardl_bic:<15.2f} {var_bic:<15.2f} {bic_winner:<15}")

print("\nInterpretation:")
print("- R²: Proportion of variance explained (higher = better fit)")
print("- AIC/BIC: Information criteria (lower = better, penalize complexity)")
print("- BIC penalizes model complexity more than AIC")

# ============================================================================
# PART 2: OUT-OF-SAMPLE FORECAST COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: OUT-OF-SAMPLE FORECAST COMPARISON")
print("=" * 70)

# Split data: train on all but last 12 months, test on last 12 months
test_size = 12
train_data = data.iloc[:-test_size]
test_data = data.iloc[-test_size:]

print(f"\nTrain period: {train_data.index[0]} to {train_data.index[-1]}")
print(f"Test period:  {test_data.index[0]} to {test_data.index[-1]}")
print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

# -----------------------------------------------------------------------------
# 2A. ARDL Out-of-Sample Forecast
# -----------------------------------------------------------------------------

print("\n[1] ARDL Out-of-Sample Forecast")
print("-" * 70)

# Estimate ARDL on training data
ardl_train = ARDL(endog=train_data['infl'],
                  exog=train_data[['mpr', 'exo', 'tbr']],
                  lags=2, order=(1, 1, 1))
ardl_train_fit = ardl_train.fit()

# Forecast 12 steps ahead
# Need to provide exogenous variables for forecast period
ardl_forecast = ardl_train_fit.forecast(steps=test_size,
                                        exog_oos=test_data[['mpr', 'exo', 'tbr']])

# Calculate forecast errors
ardl_errors = test_data['infl'] - ardl_forecast
ardl_rmse = np.sqrt(np.mean(ardl_errors**2))
ardl_mae = np.mean(np.abs(ardl_errors))

print(f"ARDL Forecast Accuracy:")
print(f"  RMSE: {ardl_rmse:.4f}")
print(f"  MAE:  {ardl_mae:.4f}")

# -----------------------------------------------------------------------------
# 2B. VAR Out-of-Sample Forecast
# -----------------------------------------------------------------------------

print("\n[2] VAR Out-of-Sample Forecast")
print("-" * 70)

# Estimate VAR on training data
var_train_data = train_data[['infl', 'mpr', 'exo', 'tbr']]
var_train_model = VAR(var_train_data)
var_train_fit = var_train_model.fit(maxlags=2)

# Forecast 12 steps ahead
# VAR forecasts all variables simultaneously
var_forecast_all = var_train_fit.forecast(var_train_data.values[-var_train_fit.k_ar:],
                                           steps=test_size)

# Extract inflation forecasts (first column)
var_forecast = pd.Series(var_forecast_all[:, 0], index=test_data.index)

# Calculate forecast errors
var_errors = test_data['infl'] - var_forecast
var_rmse = np.sqrt(np.mean(var_errors**2))
var_mae = np.mean(np.abs(var_errors))

print(f"VAR Forecast Accuracy:")
print(f"  RMSE: {var_rmse:.4f}")
print(f"  MAE:  {var_mae:.4f}")

# -----------------------------------------------------------------------------
# 2C. Out-of-Sample Comparison
# -----------------------------------------------------------------------------

print("\n[3] Out-of-Sample Comparison")
print("-" * 70)

print(f"\n{'Metric':<15} {'ARDL':<15} {'VAR':<15} {'Winner':<15}")
print("-" * 60)

# RMSE (lower is better)
rmse_winner = "ARDL" if ardl_rmse < var_rmse else "VAR"
print(f"{'RMSE':<15} {ardl_rmse:<15.4f} {var_rmse:<15.4f} {rmse_winner:<15}")

# MAE (lower is better)
mae_winner = "ARDL" if ardl_mae < var_mae else "VAR"
print(f"{'MAE':<15} {ardl_mae:<15.4f} {var_mae:<15.4f} {mae_winner:<15}")

# Improvement percentage
if ardl_rmse < var_rmse:
    improvement = ((var_rmse - ardl_rmse) / var_rmse) * 100
    print(f"\nARDL outperforms VAR by {improvement:.1f}% (RMSE)")
else:
    improvement = ((ardl_rmse - var_rmse) / ardl_rmse) * 100
    print(f"\nVAR outperforms ARDL by {improvement:.1f}% (RMSE)")

print("\nInterpretation:")
print("- RMSE: Root Mean Squared Error (penalizes large errors)")
print("- MAE: Mean Absolute Error (treats all errors equally)")
print("- Out-of-sample accuracy is the gold standard for model comparison")

# Store forecast data for plotting in next step
forecast_results = pd.DataFrame({
    'actual': test_data['infl'].values,
    'ardl_forecast': ardl_forecast.values,
    'var_forecast': var_forecast.values
}, index=test_data.index)

print("\nForecast comparison completed.")
```

**What Changed:**

1. **Train/Test Split**: Last 12 months held out as test set
2. **ARDL Forecast**: Estimate on training data, forecast 12 steps with known exogenous values
3. **VAR Forecast**: Estimate on training data, forecast 12 steps (all variables)
4. **Accuracy Metrics**: Calculate RMSE and MAE for inflation forecasts
5. **Comparison**: Determine which model forecasts better out-of-sample

**Run it:**

```bash
python econometric_models/model_comparison.py
```

You'll now see both in-sample fit and out-of-sample forecast accuracy. Often, the model with the best in-sample fit doesn't have the best out-of-sample forecast accuracy.

---

### STEP 3: Add Comparison Table and Visualization

Finally, let's save a comprehensive comparison table and create a plot showing actual inflation vs both model forecasts.

**Delete everything in econometric_models/model_comparison.py and replace it with this:**

```python
"""
Model Comparison: ARDL vs VAR
Compare the ARDL and VAR models using in-sample and out-of-sample metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, meanabs
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Load the data
data = pd.read_csv('data/processed/model_data.csv', index_col=0, parse_dates=True)

# Our key variables
mpr = data['mpr']       # Monetary Policy Rate
infl = data['infl']     # Inflation (our target)
exo = data['exo']       # Exchange Rate
tbr = data['tbr']       # Treasury Bill Rate

print("=" * 70)
print("MODEL COMPARISON: ARDL vs VAR")
print("=" * 70)
print(f"\nData period: {data.index[0]} to {data.index[-1]}")
print(f"Total observations: {len(data)}")

# ============================================================================
# PART 1: RE-ESTIMATE BOTH MODELS ON FULL DATA
# ============================================================================

print("\n" + "=" * 70)
print("PART 1: RE-ESTIMATING BOTH MODELS (FULL SAMPLE)")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1A. ARDL Model
# -----------------------------------------------------------------------------

print("\n[1] ARDL Model")
print("-" * 70)

# ARDL(2,1,1,1) - 2 lags of infl, 1 lag each of mpr, exo, tbr
ardl_model = ARDL(endog=infl, exog=data[['mpr', 'exo', 'tbr']],
                   lags=2, order=(1, 1, 1))
ardl_fit = ardl_model.fit()

# Extract in-sample metrics
ardl_rsquared = ardl_fit.rsquared
ardl_aic = ardl_fit.aic
ardl_bic = ardl_fit.bic
ardl_fitted = ardl_fit.fittedvalues
ardl_residuals = ardl_fit.resid

print(f"ARDL(2,1,1,1) Results:")
print(f"  R-squared:  {ardl_rsquared:.4f}")
print(f"  AIC:        {ardl_aic:.2f}")
print(f"  BIC:        {ardl_bic:.2f}")
print(f"  Obs:        {len(ardl_fitted)}")

# -----------------------------------------------------------------------------
# 1B. VAR Model
# -----------------------------------------------------------------------------

print("\n[2] VAR Model")
print("-" * 70)

# VAR with 2 lags (all variables treated symmetrically)
var_data = data[['infl', 'mpr', 'exo', 'tbr']]
var_model = VAR(var_data)
var_fit = var_model.fit(maxlags=2)

# Extract in-sample metrics for the inflation equation only
var_aic = var_fit.aic
var_bic = var_fit.bic

# Get fitted values for inflation (first variable)
var_fitted = var_fit.fittedvalues['infl']
var_residuals = var_fit.resid['infl']

# Calculate R² for inflation equation
ss_res = np.sum(var_residuals**2)
ss_tot = np.sum((infl[var_fit.k_ar:] - infl[var_fit.k_ar:].mean())**2)
var_rsquared = 1 - (ss_res / ss_tot)

print(f"VAR(2) Results (Inflation Equation):")
print(f"  R-squared:  {var_rsquared:.4f}")
print(f"  AIC:        {var_aic:.2f}")
print(f"  BIC:        {var_bic:.2f}")
print(f"  Obs:        {len(var_fitted)}")

# -----------------------------------------------------------------------------
# 1C. In-Sample Comparison
# -----------------------------------------------------------------------------

print("\n[3] In-Sample Fit Comparison")
print("-" * 70)

print(f"\n{'Metric':<15} {'ARDL':<15} {'VAR':<15} {'Winner':<15}")
print("-" * 60)

# R² (higher is better)
r2_winner = "ARDL" if ardl_rsquared > var_rsquared else "VAR"
print(f"{'R-squared':<15} {ardl_rsquared:<15.4f} {var_rsquared:<15.4f} {r2_winner:<15}")

# AIC (lower is better)
aic_winner = "ARDL" if ardl_aic < var_aic else "VAR"
print(f"{'AIC':<15} {ardl_aic:<15.2f} {var_aic:<15.2f} {aic_winner:<15}")

# BIC (lower is better)
bic_winner = "ARDL" if ardl_bic < var_bic else "VAR"
print(f"{'BIC':<15} {ardl_bic:<15.2f} {var_bic:<15.2f} {bic_winner:<15}")

print("\nInterpretation:")
print("- R²: Proportion of variance explained (higher = better fit)")
print("- AIC/BIC: Information criteria (lower = better, penalize complexity)")
print("- BIC penalizes model complexity more than AIC")

# ============================================================================
# PART 2: OUT-OF-SAMPLE FORECAST COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: OUT-OF-SAMPLE FORECAST COMPARISON")
print("=" * 70)

# Split data: train on all but last 12 months, test on last 12 months
test_size = 12
train_data = data.iloc[:-test_size]
test_data = data.iloc[-test_size:]

print(f"\nTrain period: {train_data.index[0]} to {train_data.index[-1]}")
print(f"Test period:  {test_data.index[0]} to {test_data.index[-1]}")
print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

# -----------------------------------------------------------------------------
# 2A. ARDL Out-of-Sample Forecast
# -----------------------------------------------------------------------------

print("\n[1] ARDL Out-of-Sample Forecast")
print("-" * 70)

# Estimate ARDL on training data
ardl_train = ARDL(endog=train_data['infl'],
                  exog=train_data[['mpr', 'exo', 'tbr']],
                  lags=2, order=(1, 1, 1))
ardl_train_fit = ardl_train.fit()

# Forecast 12 steps ahead
# Need to provide exogenous variables for forecast period
ardl_forecast = ardl_train_fit.forecast(steps=test_size,
                                        exog_oos=test_data[['mpr', 'exo', 'tbr']])

# Calculate forecast errors
ardl_errors = test_data['infl'] - ardl_forecast
ardl_rmse = np.sqrt(np.mean(ardl_errors**2))
ardl_mae = np.mean(np.abs(ardl_errors))

print(f"ARDL Forecast Accuracy:")
print(f"  RMSE: {ardl_rmse:.4f}")
print(f"  MAE:  {ardl_mae:.4f}")

# -----------------------------------------------------------------------------
# 2B. VAR Out-of-Sample Forecast
# -----------------------------------------------------------------------------

print("\n[2] VAR Out-of-Sample Forecast")
print("-" * 70)

# Estimate VAR on training data
var_train_data = train_data[['infl', 'mpr', 'exo', 'tbr']]
var_train_model = VAR(var_train_data)
var_train_fit = var_train_model.fit(maxlags=2)

# Forecast 12 steps ahead
# VAR forecasts all variables simultaneously
var_forecast_all = var_train_fit.forecast(var_train_data.values[-var_train_fit.k_ar:],
                                           steps=test_size)

# Extract inflation forecasts (first column)
var_forecast = pd.Series(var_forecast_all[:, 0], index=test_data.index)

# Calculate forecast errors
var_errors = test_data['infl'] - var_forecast
var_rmse = np.sqrt(np.mean(var_errors**2))
var_mae = np.mean(np.abs(var_errors))

print(f"VAR Forecast Accuracy:")
print(f"  RMSE: {var_rmse:.4f}")
print(f"  MAE:  {var_mae:.4f}")

# -----------------------------------------------------------------------------
# 2C. Out-of-Sample Comparison
# -----------------------------------------------------------------------------

print("\n[3] Out-of-Sample Comparison")
print("-" * 70)

print(f"\n{'Metric':<15} {'ARDL':<15} {'VAR':<15} {'Winner':<15}")
print("-" * 60)

# RMSE (lower is better)
rmse_winner = "ARDL" if ardl_rmse < var_rmse else "VAR"
print(f"{'RMSE':<15} {ardl_rmse:<15.4f} {var_rmse:<15.4f} {rmse_winner:<15}")

# MAE (lower is better)
mae_winner = "ARDL" if ardl_mae < var_mae else "VAR"
print(f"{'MAE':<15} {ardl_mae:<15.4f} {var_mae:<15.4f} {mae_winner:<15}")

# Improvement percentage
if ardl_rmse < var_rmse:
    improvement = ((var_rmse - ardl_rmse) / var_rmse) * 100
    print(f"\nARDL outperforms VAR by {improvement:.1f}% (RMSE)")
else:
    improvement = ((ardl_rmse - var_rmse) / ardl_rmse) * 100
    print(f"\nVAR outperforms ARDL by {improvement:.1f}% (RMSE)")

print("\nInterpretation:")
print("- RMSE: Root Mean Squared Error (penalizes large errors)")
print("- MAE: Mean Absolute Error (treats all errors equally)")
print("- Out-of-sample accuracy is the gold standard for model comparison")

# ============================================================================
# PART 3: SAVE COMPARISON TABLE AND PLOT
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: CREATING COMPARISON TABLE AND PLOT")
print("=" * 70)

# -----------------------------------------------------------------------------
# 3A. Create Comprehensive Comparison Table
# -----------------------------------------------------------------------------

print("\n[1] Creating Comparison Table")
print("-" * 70)

comparison_table = pd.DataFrame({
    'Metric': ['R-squared (In-Sample)', 'AIC (In-Sample)', 'BIC (In-Sample)',
               'RMSE (Out-of-Sample)', 'MAE (Out-of-Sample)'],
    'ARDL': [ardl_rsquared, ardl_aic, ardl_bic, ardl_rmse, ardl_mae],
    'VAR': [var_rsquared, var_aic, var_bic, var_rmse, var_mae],
    'Better': [r2_winner, aic_winner, bic_winner, rmse_winner, mae_winner],
    'Direction': ['Higher', 'Lower', 'Lower', 'Lower', 'Lower']
})

# Save to CSV
comparison_table.to_csv('results/model_comparison.csv', index=False)
print("Saved: results/model_comparison.csv")

# Display the table
print("\nFull Comparison Table:")
print(comparison_table.to_string(index=False))

# -----------------------------------------------------------------------------
# 3B. Create Forecast Comparison Plot
# -----------------------------------------------------------------------------

print("\n[2] Creating Forecast Comparison Plot")
print("-" * 70)

# Prepare forecast data
forecast_results = pd.DataFrame({
    'Actual': test_data['infl'].values,
    'ARDL': ardl_forecast.values,
    'VAR': var_forecast.values
}, index=test_data.index)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot actual and forecasts
ax.plot(forecast_results.index, forecast_results['Actual'],
        marker='o', linewidth=2, markersize=6, label='Actual Inflation', color='black')
ax.plot(forecast_results.index, forecast_results['ARDL'],
        marker='s', linewidth=2, markersize=6, label='ARDL Forecast',
        color='blue', linestyle='--')
ax.plot(forecast_results.index, forecast_results['VAR'],
        marker='^', linewidth=2, markersize=6, label='VAR Forecast',
        color='red', linestyle=':')

# Add shaded region for test period
ax.axvspan(test_data.index[0], test_data.index[-1], alpha=0.1, color='gray',
           label='Test Period')

# Formatting
ax.set_xlabel('Date', fontsize=11, fontweight='bold')
ax.set_ylabel('Inflation Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('Out-of-Sample Forecast Comparison: ARDL vs VAR',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

# Add text box with accuracy metrics
textstr = f'ARDL RMSE: {ardl_rmse:.4f}\nVAR RMSE: {var_rmse:.4f}\n\nARDL MAE: {ardl_mae:.4f}\nVAR MAE: {var_mae:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: results/model_comparison.png")

plt.close()

# -----------------------------------------------------------------------------
# 3C. Summary Statistics
# -----------------------------------------------------------------------------

print("\n[3] Forecast Summary Statistics")
print("-" * 70)

summary_stats = pd.DataFrame({
    'Statistic': ['Mean', 'Std Dev', 'Min', 'Max'],
    'Actual': [forecast_results['Actual'].mean(),
               forecast_results['Actual'].std(),
               forecast_results['Actual'].min(),
               forecast_results['Actual'].max()],
    'ARDL': [forecast_results['ARDL'].mean(),
             forecast_results['ARDL'].std(),
             forecast_results['ARDL'].min(),
             forecast_results['ARDL'].max()],
    'VAR': [forecast_results['VAR'].mean(),
            forecast_results['VAR'].std(),
            forecast_results['VAR'].min(),
            forecast_results['VAR'].max()]
})

print(summary_stats.to_string(index=False, float_format='%.4f'))

print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETE")
print("=" * 70)
print("\nOutputs saved:")
print("  1. results/model_comparison.csv - Comparison table")
print("  2. results/model_comparison.png - Forecast plot")
```

**What's New:**

1. **Comparison Table**: Creates a comprehensive table with all metrics and saves to CSV
2. **Forecast Plot**: Visualizes actual inflation vs both model forecasts over test period
3. **Accuracy Display**: Shows RMSE/MAE in text box on the plot
4. **Summary Stats**: Compares mean, std dev, min, max of actual vs forecasts

**Run it:**

```bash
python econometric_models/model_comparison.py
```

**Check the outputs:**

```bash
cat results/model_comparison.csv
open results/model_comparison.png  # or xdg-open on Linux
```

---

## When to Use Which Model

Now that you've compared both models, here's guidance on when to use each:

### Use ARDL When:

1. **Long-Run Relationships**: You want to estimate equilibrium relationships and speed of adjustment (error correction)
2. **Cointegration Analysis**: Testing whether variables move together in the long run
3. **Clear Direction**: You have a single dependent variable (inflation) and clear exogenous drivers (mpr, exo, tbr)
4. **Policy Analysis**: Analyzing how policy variables affect the target over time
5. **Forecasting with Scenarios**: You can provide different paths for exogenous variables

**ARDL Strengths:**
- Explicitly models long-run equilibrium
- Distinguishes between short-run dynamics and long-run levels
- More economically interpretable coefficients
- Can handle I(0) and I(1) variables

**ARDL Weaknesses:**
- Assumes causality direction (infl depends on mpr, exo, tbr)
- Cannot model feedback effects (how does infl affect mpr?)
- Less flexible for system-wide shocks

### Use VAR When:

1. **Dynamic Interactions**: All variables influence each other (no assumed causality)
2. **Impulse Response Functions**: Want to trace out how shocks propagate through the system
3. **Forecast Error Variance Decomposition**: Understanding what drives forecast uncertainty
4. **Granger Causality**: Testing whether one variable helps predict another
5. **Policy Simulations**: Seeing system-wide effects of shocks

**VAR Strengths:**
- Treats all variables symmetrically (no assumed exogeneity)
- Captures feedback effects and dynamic interactions
- Great for IRFs and variance decomposition
- Flexible for structural analysis (SVAR)

**VAR Weaknesses:**
- Many parameters to estimate (can overfit)
- Less interpretable coefficients
- Requires forecasts for ALL variables (not just inflation)
- Doesn't explicitly model long-run equilibrium

### Practical Decision Rule:

- **For forecasting inflation alone**: Use whichever has better out-of-sample RMSE (usually ARDL if exog variables are predictable)
- **For understanding monetary policy transmission**: Use VAR to see feedback effects and IRFs
- **For long-run policy analysis**: Use ARDL to get error correction and equilibrium relationships
- **For rich scenario analysis**: Use ARDL if you have paths for exog variables; use VAR if you're simulating shocks

**In this project:** ARDL likely performs better for pure inflation forecasting, while VAR provides richer insight into how monetary policy shocks affect the entire system.

---

## Commit Your Work

You've successfully compared the ARDL and VAR models. Commit your progress:

```bash
cd /home/user/Nigerian_Inflation_Predictor
git add econometric_models/model_comparison.py results/model_comparison.csv results/model_comparison.png
git commit -m "Day 23: Add model comparison (ARDL vs VAR) with in-sample fit and out-of-sample forecast evaluation"
git push
```

---

## Common Errors and Solutions

### Error 1: Different Sample Sizes

**Error message:**
```
ValueError: operands could not be broadcast together with shapes...
```

**Problem:** ARDL and VAR start at different periods due to lags.

**Solution:** Align fitted values to same index before comparing:

```python
# Align to common index
common_idx = ardl_fitted.index.intersection(var_fitted.index)
ardl_fitted_aligned = ardl_fitted[common_idx]
var_fitted_aligned = var_fitted[common_idx]
```

### Error 2: VAR Forecast Shape Issues

**Error message:**
```
IndexError: too many indices for array
```

**Problem:** VAR forecast returns 2D array (all variables), need to extract inflation column.

**Solution:**

```python
# Correct way to extract inflation forecast
var_forecast = pd.Series(var_forecast_all[:, 0], index=test_data.index)
```

### Error 3: Test Set Too Small

**Problem:** Test set of 12 months might be too small for some datasets.

**Solution:**

```python
# Adjust test size based on sample size
test_size = max(12, int(0.2 * len(data)))  # At least 12 or 20% of data
```

### Error 4: Missing Exogenous Variables

**Error message:**
```
ValueError: exog_oos must be provided for forecasting
```

**Problem:** ARDL needs exogenous variables for forecast period.

**Solution:**

```python
# Always provide exog_oos when forecasting ARDL
ardl_forecast = ardl_train_fit.forecast(steps=test_size,
                                        exog_oos=test_data[['mpr', 'exo', 'tbr']])
```

---

## Q&A Section

**Q1: Why does VAR have lower in-sample fit but better out-of-sample forecast (or vice versa)?**

**A:** This is the bias-variance tradeoff. A complex model (more parameters) fits training data better (lower bias) but may overfit and forecast worse (high variance). A simpler model might underfit training data but generalize better to test data. Always trust out-of-sample results over in-sample fit.

**Q2: Should I use AIC or BIC for model selection?**

**A:** BIC penalizes complexity more strongly, so prefer it if you want a simpler model. AIC is better if you want the best fit. For forecasting, out-of-sample RMSE is more important than either.

**Q3: Can I combine ARDL and VAR forecasts?**

**A:** Yes! Forecast combination often outperforms individual models. Try:

```python
combined_forecast = 0.5 * ardl_forecast + 0.5 * var_forecast
```

Or use optimal weights based on historical forecast accuracy.

**Q4: What if both models forecast poorly?**

**A:** Consider:
1. Adding more lags (but watch for overfitting)
2. Including more exogenous variables (oil prices, global inflation)
3. Nonlinear models (LSTM, Random Forest)
4. Structural breaks (estimate on recent data only)

**Q5: How do I choose the test set size?**

**A:** Common choices:
- 12 months for monthly data (one year ahead)
- 20% of total sample
- Forecast horizon you care about (if forecasting 6 months, use 6-month test set)

**Q6: What's a "good" RMSE or MAE?**

**A:** It depends on the scale of your data. For Nigerian inflation (10-20%):
- RMSE < 1.5: Excellent
- RMSE 1.5-2.5: Good
- RMSE 2.5-4.0: Moderate
- RMSE > 4.0: Poor

Compare to a naive forecast (random walk) as baseline.

**Q7: Should I re-estimate the model on the full sample after comparison?**

**A:** Yes! The train/test split is only for evaluation. For your final model, estimate on all available data to get the most precise parameter estimates.

**Q8: Can I compare more than two models?**

**A:** Absolutely! Add LSTM (Day 24), GARCH (Day 22), or any other model to this framework. Create a loop over models:

```python
models = {'ARDL': ardl_forecast, 'VAR': var_forecast, 'LSTM': lstm_forecast}
for name, forecast in models.items():
    rmse = np.sqrt(np.mean((test_data['infl'] - forecast)**2))
    print(f"{name} RMSE: {rmse:.4f}")
```

---

## What's Next?

Tomorrow (Day 24), you'll implement a modern machine learning approach: **LSTM Neural Networks** for inflation forecasting. You'll compare the deep learning approach to your traditional econometric models.

**Key Takeaway:** Model comparison is essential. Don't rely on a single model—compare multiple approaches and use the one that performs best out-of-sample. ARDL excels at long-run relationships, VAR at dynamic interactions, and tomorrow you'll see what deep learning brings to the table.
