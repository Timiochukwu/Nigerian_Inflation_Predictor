# DAY 29 — Inflation Forecasting: 12-Month Ahead Predictions

## What You'll Learn Today

Today you'll learn how to generate actual inflation forecasts from your VAR model. This is where econometrics meets real-world decision-making. You'll create 12-month-ahead predictions with confidence intervals, visualize the forecast path, and export results for the API.

Key concepts:
- How to generate multi-step-ahead forecasts from a VAR model
- Forecast confidence intervals and uncertainty quantification
- Creating professional forecast visualizations
- Exporting forecast data for downstream applications

By the end of today, you'll have a complete forecasting pipeline that produces both visual and tabular outputs.

---

## Building simulation/forecast.py — 3 Steps

We'll build the forecasting module in three iterations. Remember: each step shows the COMPLETE file, not snippets.

---

### STEP 1: Imports, VAR Estimation, and Basic Forecasting

Delete everything in `simulation/forecast.py` and replace it with this:

```python
"""
Inflation Forecasting Module
Generates 12-month-ahead forecasts from the VAR model
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

def load_and_prepare_data():
    """Load the cleaned dataset and prepare for VAR estimation"""
    df = pd.read_csv('data/processed/cleaned_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Select VAR variables in order: mpr, infl, exo, tbr
    var_data = df[['mpr', 'infl', 'exo', 'tbr']].dropna()

    return var_data

def estimate_var_model(data, lag_order=2):
    """Estimate VAR model on full sample"""
    model = VAR(data)
    var_result = model.fit(lag_order)

    print("VAR Model Estimated Successfully")
    print(f"Sample size: {len(data)} observations")
    print(f"Lag order: {lag_order}")
    print(f"Last observation date: {data.index[-1]}")

    return var_result

def generate_forecast(var_result, steps=12):
    """
    Generate multi-step-ahead forecasts

    Parameters:
    -----------
    var_result : VARResultsWrapper
        Fitted VAR model
    steps : int
        Number of periods ahead to forecast (default: 12 months)

    Returns:
    --------
    forecast_df : DataFrame
        Point forecasts for each variable
    lower_bound : DataFrame
        Lower bound of 95% confidence interval
    upper_bound : DataFrame
        Upper bound of 95% confidence interval
    """
    # Get point forecasts
    forecast_values = var_result.forecast(var_result.endog[-var_result.k_ar:], steps=steps)

    # Get forecast confidence intervals
    forecast_intervals = var_result.forecast_interval(
        var_result.endog[-var_result.k_ar:],
        steps=steps,
        alpha=0.05  # 95% confidence interval
    )

    # Extract lower and upper bounds
    lower_bound = forecast_intervals[:, :, 0]  # Lower bound (2.5th percentile)
    upper_bound = forecast_intervals[:, :, 1]  # Upper bound (97.5th percentile)

    # Create date range for forecast period
    last_date = var_result.endog_names
    # For now, just use integer index
    forecast_index = range(1, steps + 1)

    # Create DataFrames
    column_names = ['mpr', 'infl', 'exo', 'tbr']

    forecast_df = pd.DataFrame(
        forecast_values,
        columns=column_names,
        index=forecast_index
    )

    lower_df = pd.DataFrame(
        lower_bound,
        columns=column_names,
        index=forecast_index
    )

    upper_df = pd.DataFrame(
        upper_bound,
        columns=column_names,
        index=forecast_index
    )

    return forecast_df, lower_df, upper_df

def main():
    """Main execution function"""
    print("=" * 60)
    print("INFLATION FORECASTING MODULE")
    print("=" * 60)
    print()

    # Load data
    print("Step 1: Loading data...")
    var_data = load_and_prepare_data()
    print(f"Loaded {len(var_data)} observations")
    print()

    # Estimate VAR model
    print("Step 2: Estimating VAR model...")
    var_result = estimate_var_model(var_data, lag_order=2)
    print()

    # Generate 12-month-ahead forecasts
    print("Step 3: Generating 12-month-ahead forecasts...")
    forecast_df, lower_df, upper_df = generate_forecast(var_result, steps=12)
    print()

    print("INFLATION FORECAST (next 12 months):")
    print(forecast_df['infl'])
    print()

    print("95% CONFIDENCE INTERVAL:")
    print(f"Lower bound: {lower_df['infl'].values}")
    print(f"Upper bound: {upper_df['infl'].values}")
    print()

    print("Forecasting complete!")

if __name__ == '__main__':
    main()
```

**What's happening here?**

1. **load_and_prepare_data()**: Loads the cleaned dataset and selects the four VAR variables
2. **estimate_var_model()**: Fits a VAR(2) model on the full sample
3. **generate_forecast()**:
   - Uses `var_result.forecast()` for point forecasts
   - Uses `var_result.forecast_interval()` for 95% confidence intervals
   - Returns three DataFrames: point forecast, lower bound, upper bound
4. **main()**: Orchestrates the workflow and prints results

Run this to verify it works: `python simulation/forecast.py`

---

### STEP 2: Add Forecast Visualization

Delete everything in `simulation/forecast.py` and replace it with this:

```python
"""
Inflation Forecasting Module
Generates 12-month-ahead forecasts from the VAR model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

def load_and_prepare_data():
    """Load the cleaned dataset and prepare for VAR estimation"""
    df = pd.read_csv('data/processed/cleaned_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Select VAR variables in order: mpr, infl, exo, tbr
    var_data = df[['mpr', 'infl', 'exo', 'tbr']].dropna()

    return var_data

def estimate_var_model(data, lag_order=2):
    """Estimate VAR model on full sample"""
    model = VAR(data)
    var_result = model.fit(lag_order)

    print("VAR Model Estimated Successfully")
    print(f"Sample size: {len(data)} observations")
    print(f"Lag order: {lag_order}")
    print(f"Last observation date: {data.index[-1]}")

    return var_result

def generate_forecast(var_result, steps=12):
    """
    Generate multi-step-ahead forecasts

    Parameters:
    -----------
    var_result : VARResultsWrapper
        Fitted VAR model
    steps : int
        Number of periods ahead to forecast (default: 12 months)

    Returns:
    --------
    forecast_df : DataFrame
        Point forecasts for each variable
    lower_bound : DataFrame
        Lower bound of 95% confidence interval
    upper_bound : DataFrame
        Upper bound of 95% confidence interval
    """
    # Get point forecasts
    forecast_values = var_result.forecast(var_result.endog[-var_result.k_ar:], steps=steps)

    # Get forecast confidence intervals
    forecast_intervals = var_result.forecast_interval(
        var_result.endog[-var_result.k_ar:],
        steps=steps,
        alpha=0.05  # 95% confidence interval
    )

    # Extract lower and upper bounds
    lower_bound = forecast_intervals[:, :, 0]  # Lower bound (2.5th percentile)
    upper_bound = forecast_intervals[:, :, 1]  # Upper bound (97.5th percentile)

    # Create date range for forecast period
    last_date = var_result.data.orig_endog.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=steps,
        freq='MS'  # Month start
    )

    # Create DataFrames
    column_names = ['mpr', 'infl', 'exo', 'tbr']

    forecast_df = pd.DataFrame(
        forecast_values,
        columns=column_names,
        index=forecast_dates
    )

    lower_df = pd.DataFrame(
        lower_bound,
        columns=column_names,
        index=forecast_dates
    )

    upper_df = pd.DataFrame(
        upper_bound,
        columns=column_names,
        index=forecast_dates
    )

    return forecast_df, lower_df, upper_df

def plot_inflation_forecast(historical_data, forecast_df, lower_df, upper_df):
    """
    Create a professional forecast visualization

    Shows:
    - Historical inflation (solid line)
    - Forecast path (dashed line)
    - 95% confidence interval (shaded band)
    """
    plt.figure(figsize=(12, 6))

    # Plot historical inflation (last 24 months for context)
    historical_infl = historical_data['infl'][-24:]
    plt.plot(historical_infl.index, historical_infl.values,
             color='#2E86AB', linewidth=2, label='Historical Inflation')

    # Connect historical to forecast with a marker
    last_historical_date = historical_infl.index[-1]
    last_historical_value = historical_infl.values[-1]
    first_forecast_date = forecast_df.index[0]
    first_forecast_value = forecast_df['infl'].values[0]

    plt.plot([last_historical_date, first_forecast_date],
             [last_historical_value, first_forecast_value],
             color='#A23B72', linestyle='--', linewidth=2)

    # Plot forecast
    plt.plot(forecast_df.index, forecast_df['infl'],
             color='#A23B72', linestyle='--', linewidth=2,
             label='12-Month Forecast')

    # Plot confidence interval as shaded area
    plt.fill_between(forecast_df.index,
                     lower_df['infl'],
                     upper_df['infl'],
                     color='#A23B72', alpha=0.2,
                     label='95% Confidence Interval')

    # Add vertical line to separate historical from forecast
    plt.axvline(x=last_historical_date, color='gray',
                linestyle=':', linewidth=1, alpha=0.7)

    # Formatting
    plt.title('Nigeria Inflation Forecast: 12-Month Ahead Projection',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Inflation Rate (%)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save the plot
    output_path = 'results/inflation_forecast.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Forecast chart saved to: {output_path}")

    plt.close()

def main():
    """Main execution function"""
    print("=" * 60)
    print("INFLATION FORECASTING MODULE")
    print("=" * 60)
    print()

    # Load data
    print("Step 1: Loading data...")
    var_data = load_and_prepare_data()
    print(f"Loaded {len(var_data)} observations")
    print()

    # Estimate VAR model
    print("Step 2: Estimating VAR model...")
    var_result = estimate_var_model(var_data, lag_order=2)
    print()

    # Generate 12-month-ahead forecasts
    print("Step 3: Generating 12-month-ahead forecasts...")
    forecast_df, lower_df, upper_df = generate_forecast(var_result, steps=12)
    print()

    print("INFLATION FORECAST (next 12 months):")
    print(forecast_df['infl'])
    print()

    # Create forecast visualization
    print("Step 4: Creating forecast visualization...")
    plot_inflation_forecast(var_data, forecast_df, lower_df, upper_df)
    print()

    print("Forecasting complete!")

if __name__ == '__main__':
    main()
```

**What changed?**

1. **plot_inflation_forecast()**: New function that creates a professional chart:
   - Historical inflation as solid blue line (last 24 months for context)
   - Forecast period as dashed purple line
   - 95% confidence interval as shaded purple band
   - Vertical dotted line separating historical from forecast period
   - Saves to `results/inflation_forecast.png` at high resolution (300 DPI)

2. Updated **generate_forecast()**: Now creates proper date index using `pd.date_range()` for the forecast period

Run this: `python simulation/forecast.py`

Check the output: `results/inflation_forecast.png`

---

### STEP 3: Add Forecast Table Export

Delete everything in `simulation/forecast.py` and replace it with this:

```python
"""
Inflation Forecasting Module
Generates 12-month-ahead forecasts from the VAR model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import json
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

def load_and_prepare_data():
    """Load the cleaned dataset and prepare for VAR estimation"""
    df = pd.read_csv('data/processed/cleaned_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Select VAR variables in order: mpr, infl, exo, tbr
    var_data = df[['mpr', 'infl', 'exo', 'tbr']].dropna()

    return var_data

def estimate_var_model(data, lag_order=2):
    """Estimate VAR model on full sample"""
    model = VAR(data)
    var_result = model.fit(lag_order)

    print("VAR Model Estimated Successfully")
    print(f"Sample size: {len(data)} observations")
    print(f"Lag order: {lag_order}")
    print(f"Last observation date: {data.index[-1]}")

    return var_result

def generate_forecast(var_result, steps=12):
    """
    Generate multi-step-ahead forecasts

    Parameters:
    -----------
    var_result : VARResultsWrapper
        Fitted VAR model
    steps : int
        Number of periods ahead to forecast (default: 12 months)

    Returns:
    --------
    forecast_df : DataFrame
        Point forecasts for each variable
    lower_bound : DataFrame
        Lower bound of 95% confidence interval
    upper_bound : DataFrame
        Upper bound of 95% confidence interval
    """
    # Get point forecasts
    forecast_values = var_result.forecast(var_result.endog[-var_result.k_ar:], steps=steps)

    # Get forecast confidence intervals
    forecast_intervals = var_result.forecast_interval(
        var_result.endog[-var_result.k_ar:],
        steps=steps,
        alpha=0.05  # 95% confidence interval
    )

    # Extract lower and upper bounds
    lower_bound = forecast_intervals[:, :, 0]  # Lower bound (2.5th percentile)
    upper_bound = forecast_intervals[:, :, 1]  # Upper bound (97.5th percentile)

    # Create date range for forecast period
    last_date = var_result.data.orig_endog.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=steps,
        freq='MS'  # Month start
    )

    # Create DataFrames
    column_names = ['mpr', 'infl', 'exo', 'tbr']

    forecast_df = pd.DataFrame(
        forecast_values,
        columns=column_names,
        index=forecast_dates
    )

    lower_df = pd.DataFrame(
        lower_bound,
        columns=column_names,
        index=forecast_dates
    )

    upper_df = pd.DataFrame(
        upper_bound,
        columns=column_names,
        index=forecast_dates
    )

    return forecast_df, lower_df, upper_df

def plot_inflation_forecast(historical_data, forecast_df, lower_df, upper_df):
    """
    Create a professional forecast visualization

    Shows:
    - Historical inflation (solid line)
    - Forecast path (dashed line)
    - 95% confidence interval (shaded band)
    """
    plt.figure(fig=(12, 6))

    # Plot historical inflation (last 24 months for context)
    historical_infl = historical_data['infl'][-24:]
    plt.plot(historical_infl.index, historical_infl.values,
             color='#2E86AB', linewidth=2, label='Historical Inflation')

    # Connect historical to forecast with a marker
    last_historical_date = historical_infl.index[-1]
    last_historical_value = historical_infl.values[-1]
    first_forecast_date = forecast_df.index[0]
    first_forecast_value = forecast_df['infl'].values[0]

    plt.plot([last_historical_date, first_forecast_date],
             [last_historical_value, first_forecast_value],
             color='#A23B72', linestyle='--', linewidth=2)

    # Plot forecast
    plt.plot(forecast_df.index, forecast_df['infl'],
             color='#A23B72', linestyle='--', linewidth=2,
             label='12-Month Forecast')

    # Plot confidence interval as shaded area
    plt.fill_between(forecast_df.index,
                     lower_df['infl'],
                     upper_df['infl'],
                     color='#A23B72', alpha=0.2,
                     label='95% Confidence Interval')

    # Add vertical line to separate historical from forecast
    plt.axvline(x=last_historical_date, color='gray',
                linestyle=':', linewidth=1, alpha=0.7)

    # Formatting
    plt.title('Nigeria Inflation Forecast: 12-Month Ahead Projection',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Inflation Rate (%)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save the plot
    output_path = 'results/inflation_forecast.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Forecast chart saved to: {output_path}")

    plt.close()

def export_forecast_table(forecast_df, lower_df, upper_df):
    """
    Export forecast results as CSV and JSON

    Creates a formatted table with:
    - Month (forecast horizon)
    - Date
    - Point forecast
    - Lower 95% CI
    - Upper 95% CI
    """
    # Build the forecast table (focus on inflation)
    forecast_table = pd.DataFrame({
        'month': range(1, len(forecast_df) + 1),
        'date': forecast_df.index.strftime('%Y-%m-%d'),
        'inflation_forecast': forecast_df['infl'].round(2),
        'lower_95_ci': lower_df['infl'].round(2),
        'upper_95_ci': upper_df['infl'].round(2)
    })

    # Add forecast horizon labels
    forecast_table['horizon'] = forecast_table['month'].apply(
        lambda x: f"{x}-month ahead"
    )

    # Reorder columns
    forecast_table = forecast_table[[
        'month', 'horizon', 'date',
        'inflation_forecast', 'lower_95_ci', 'upper_95_ci'
    ]]

    # Export to CSV
    csv_path = 'results/inflation_forecast.csv'
    forecast_table.to_csv(csv_path, index=False)
    print(f"Forecast table saved to: {csv_path}")

    # Export to JSON (for API consumption)
    json_data = {
        'metadata': {
            'generated_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': 'VAR(2)',
            'forecast_horizon': '12 months',
            'confidence_level': '95%'
        },
        'forecasts': forecast_table.to_dict(orient='records')
    }

    json_path = 'results/inflation_forecast.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Forecast JSON saved to: {json_path}")

    # Print summary table
    print("\nFORECAST SUMMARY TABLE:")
    print("=" * 70)
    print(forecast_table.to_string(index=False))
    print("=" * 70)

def main():
    """Main execution function"""
    print("=" * 60)
    print("INFLATION FORECASTING MODULE")
    print("=" * 60)
    print()

    # Load data
    print("Step 1: Loading data...")
    var_data = load_and_prepare_data()
    print(f"Loaded {len(var_data)} observations")
    print()

    # Estimate VAR model
    print("Step 2: Estimating VAR model...")
    var_result = estimate_var_model(var_data, lag_order=2)
    print()

    # Generate 12-month-ahead forecasts
    print("Step 3: Generating 12-month-ahead forecasts...")
    forecast_df, lower_df, upper_df = generate_forecast(var_result, steps=12)
    print()

    # Create forecast visualization
    print("Step 4: Creating forecast visualization...")
    plot_inflation_forecast(var_data, forecast_df, lower_df, upper_df)
    print()

    # Export forecast table
    print("Step 5: Exporting forecast table...")
    export_forecast_table(forecast_df, lower_df, upper_df)
    print()

    print("=" * 60)
    print("FORECASTING COMPLETE!")
    print("=" * 60)
    print("\nOutputs generated:")
    print("  1. results/inflation_forecast.png  (visualization)")
    print("  2. results/inflation_forecast.csv  (data table)")
    print("  3. results/inflation_forecast.json (API-ready format)")

if __name__ == '__main__':
    main()
```

**Final additions:**

1. **export_forecast_table()**: Creates a clean forecast table with:
   - Month number (1-12)
   - Horizon label ("1-month ahead", "2-month ahead", etc.)
   - Date
   - Point forecast
   - Lower and upper 95% confidence bounds

2. **Exports two formats**:
   - CSV: Human-readable table for Excel/spreadsheets
   - JSON: Structured format for the API, includes metadata (model type, confidence level, generation timestamp)

3. **Prints summary table** to console for quick inspection

Run the complete pipeline: `python simulation/forecast.py`

Check all three outputs:
- `results/inflation_forecast.png`
- `results/inflation_forecast.csv`
- `results/inflation_forecast.json`

---

## Understanding Forecast Uncertainty

The confidence intervals you see represent forecast uncertainty. Notice how the confidence bands widen as you move further into the future. This is expected: 1-month-ahead forecasts are more accurate than 12-month-ahead forecasts.

**Why forecasts become less certain over time:**

1. **Error accumulation**: Each forecast step uses previous forecasts as inputs, compounding uncertainty
2. **Unknown shocks**: Future unexpected events (oil price spikes, policy changes) can't be predicted
3. **Model limitations**: The VAR assumes relationships remain constant (they might not)

In Nigeria's case, the wide confidence intervals reflect the high volatility in inflation dynamics. A 12-month forecast might range from 15% to 25%, which is still useful for scenario planning even if imprecise.

---

## Caveats of Economic Forecasting

Forecasting is valuable but limited. Here are critical caveats:

### 1. **Structural Breaks**
Economic relationships change over time. A model estimated on 2010-2020 data might fail in 2025 if:
- CBN changes its policy framework
- Nigeria experiences regime change
- Oil market dynamics shift fundamentally

The VAR assumes the past predicts the future. When structure breaks, forecasts fail.

### 2. **The Lucas Critique**
Named after Nobel laureate Robert Lucas, this critique states: When policy changes, people's behavior changes, invalidating your model.

Example: If CBN announces a new inflation-targeting regime, the relationship between MPR and inflation might change. Your VAR, estimated on old data, won't capture this.

### 3. **Missing Variables**
Your VAR includes mpr, infl, exo, and tbr. But Nigerian inflation also depends on:
- Food supply shocks (security issues in farming regions)
- Petrol subsidy changes
- Electricity tariff adjustments
- Global commodity prices

These omitted factors create forecast errors.

### 4. **Parameter Uncertainty**
You estimated VAR coefficients from a finite sample. Those estimates are uncertain. Your forecast intervals account for this, but they might still be too narrow if your sample is small or unrepresentative.

### 5. **Model Misspecification**
Maybe VAR(2) isn't the right model. Maybe you need VAR(3), or a different model entirely (VECM, time-varying VAR, Bayesian VAR). No way to know for sure.

**Bottom line**: Treat forecasts as scenarios, not prophecies. Use them to understand possible futures, not to predict THE future.

---

## Nigerian Context: What Makes Forecasting Harder

Nigeria presents unique forecasting challenges:

1. **Frequent policy reversals**: CBN policy can shift rapidly with leadership changes. The Godwin Emefiele era (2014-2023) saw multiple forex regime changes, each invalidating previous models.

2. **Data quality issues**: Nigerian economic data has gaps, revisions, and measurement errors. Inflation data from NBS is sometimes revised months later, making real-time forecasting difficult.

3. **Oil dependence**: Nigeria's economy is highly oil-dependent. Oil price shocks (which your model doesn't include) drive inflation, exchange rates, and fiscal policy. Without oil prices in the VAR, you're missing a key driver.

4. **Security situation**: Farmer-herder conflicts and insurgency in the northeast disrupt food supply. These non-economic shocks cause inflation spikes that your model can't predict.

5. **Parallel forex markets**: The official exchange rate (in your data) differs from parallel market rates. Real economic activity often uses parallel rates, so your model might miss true inflation dynamics.

Despite these challenges, VAR forecasting still provides value:
- It gives you a baseline scenario
- It quantifies uncertainty (via confidence intervals)
- It can be updated monthly as new data arrives
- It's transparent and interpretable

Just don't over-rely on it.

---

## Commit Your Changes

Save your work to Git:

```bash
git add simulation/forecast.py
git add results/inflation_forecast.png
git add results/inflation_forecast.csv
git add results/inflation_forecast.json
git commit -m "Add 12-month inflation forecasting with visualization and export"
```

---

## Common Errors and Solutions

### Error 1: `ValueError: freq not specified and cannot be inferred`

**Cause**: Your date index in `cleaned_data.csv` isn't recognized as a proper time series.

**Fix**: In `load_and_prepare_data()`, ensure you convert to datetime and set frequency:
```python
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').asfreq('MS')  # Add .asfreq('MS') for monthly frequency
```

### Error 2: `IndexError: too many indices for array`

**Cause**: The `forecast_interval()` method returned a different array shape than expected.

**Fix**: Check the shape: `print(forecast_intervals.shape)`. It should be `(steps, n_vars, 2)`. If not, adjust the slicing:
```python
lower_bound = forecast_intervals[:, :, 0]
upper_bound = forecast_intervals[:, :, 1]
```

### Error 3: Confidence intervals look too narrow

**Cause**: You might be using `alpha=0.5` (50% CI) instead of `alpha=0.05` (95% CI).

**Fix**: Verify the `alpha` parameter:
```python
forecast_intervals = var_result.forecast_interval(
    var_result.endog[-var_result.k_ar:],
    steps=steps,
    alpha=0.05  # 95% confidence interval (not 0.5!)
)
```

### Error 4: JSON serialization error

**Cause**: Pandas/NumPy data types aren't directly JSON-serializable.

**Fix**: Convert to native Python types:
```python
forecast_table.to_dict(orient='records')  # Already handles this
# Or manually: float(value) for NumPy floats
```

---

## Q&A

**Q: Why 12 months? Why not 24 or 6?**

A: 12 months (1 year) is a standard horizon for monetary policy. Central banks typically plan policy 12-18 months ahead because interest rate changes take 6-12 months to fully affect inflation. You can change `steps=12` to any value.

**Q: How do I forecast all four variables, not just inflation?**

A: The code already forecasts all four (mpr, infl, exo, tbr). The visualization focuses on inflation because that's your target variable. To plot others, change `forecast_df['infl']` to `forecast_df['mpr']`, etc.

**Q: Can I forecast specific scenarios (e.g., "what if MPR rises by 2%")?**

A: Not with this code. Scenario forecasting requires conditional forecasting, where you fix MPR at a specific path. That's more advanced (requires manual calculation or specialized tools). Today's forecasts are unconditional: "what will happen if nothing changes?"

**Q: Why is the forecast path smooth while historical data is volatile?**

A: VAR forecasts converge to the long-run mean. They don't predict specific shocks (which cause volatility), only the average path. This is a limitation: forecasts underestimate actual volatility.

**Q: Should I re-estimate the model every month with new data?**

A: Yes! As new data arrives, re-run the entire pipeline:
1. Add new observations to `cleaned_data.csv`
2. Re-estimate the VAR
3. Generate new forecasts

This is called "rolling forecasts" or "real-time forecasting." You'll explore this next week.

**Q: How do I evaluate forecast accuracy?**

A: Wait 12 months, then compare your forecasts to actual outcomes. Common metrics:
- Mean Absolute Error (MAE): Average of |forecast - actual|
- Root Mean Squared Error (RMSE): Sqrt of average squared errors
- Coverage: Did 95% of actuals fall within 95% CI?

You'll implement forecast evaluation on Day 30.

**Q: Can I use this for other countries?**

A: Absolutely! Just replace the Nigerian data with data from any country. The code is country-agnostic. The caveats about structural breaks and Lucas critique apply everywhere.

---

## What's Next?

Tomorrow (Day 30), you'll evaluate forecast accuracy by conducting a pseudo out-of-sample forecast experiment. You'll:
- Split data into training and test sets
- Generate historical forecasts
- Compare forecasts to actual values
- Calculate accuracy metrics (MAE, RMSE)

This will show you how well your model really performs (spoiler: probably not as well as you hope, but better than guessing).

Then you'll move into advanced topics: Bayesian VAR, time-varying parameters, and structural models.

Keep going. You're doing real econometrics now.
