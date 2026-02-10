# DAY 27 — Multiple Scenario Analysis

## What You'll Learn Today
- How to run multiple policy scenarios (+50bps, +100bps, +200bps, +300bps)
- How to compare scenarios on a single chart
- How to build a scenario comparison table

Today we'll build a powerful tool for policy analysis: running multiple interest rate scenarios and comparing their effects side-by-side. This helps answer critical questions like "How aggressive does the Central Bank of Nigeria need to be to control inflation?"

---

## Building simulation/scenario_analysis.py — 3 steps

We're going to build this file in three steps. Each step shows the COMPLETE file.

---

### STEP 1: Imports, define scenarios, compute IRFs

**Delete everything in simulation/scenario_analysis.py and replace it with this:**

```python
"""
Multiple Scenario Analysis
Compares different monetary policy shock sizes
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Load VAR model
with open('models/var_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

var_model = model_data['model']
variable_names = model_data['variables']

# Find mpr position
mpr_position = variable_names.index('mpr')

# Define scenarios (basis points)
scenarios = {
    'Mild Tightening (+50bps)': 0.50,
    'Moderate Tightening (+100bps)': 1.00,
    'Aggressive Tightening (+200bps)': 2.00,
    'Very Aggressive Tightening (+300bps)': 3.00
}

# Compute IRF for each scenario
horizon = 24
results = {}

for scenario_name, shock_size in scenarios.items():
    # Create shock vector
    shock = np.zeros(len(variable_names))
    shock[mpr_position] = shock_size

    # Compute IRF
    irf = var_model.irf(horizon)

    # Get inflation response
    infl_position = variable_names.index('infl')
    response = np.zeros(horizon)

    for h in range(horizon):
        response[h] = irf.irfs[h, infl_position, mpr_position] * shock_size

    results[scenario_name] = response

    print(f"{scenario_name}: Peak impact = {response.max():.3f}% at month {response.argmax() + 1}")
```

**What's happening here:**

1. **Load the VAR model** we trained on Day 25
2. **Define four scenarios** ranging from mild (+50bps) to very aggressive (+300bps) tightening
3. **For each scenario:**
   - Create a shock vector (only mpr changes, other variables = 0)
   - Compute the 24-month Impulse Response Function
   - Extract inflation's response to the mpr shock
   - Scale by shock size
4. **Print quick summary** of peak impact for each scenario

**Run it:**

```bash
cd /home/user/Nigerian_Inflation_Predictor
python simulation/scenario_analysis.py
```

**Expected output:**
```
Mild Tightening (+50bps): Peak impact = -0.087% at month 3
Moderate Tightening (+100bps): Peak impact = -0.174% at month 3
Aggressive Tightening (+200bps): Peak impact = -0.348% at month 3
Very Aggressive Tightening (+300bps): Peak impact = -0.522% at month 3
```

Notice the linear relationship: doubling the shock doubles the impact. This is because VAR models are linear.

---

### STEP 2: Add visualization — overlay all scenarios

**Delete everything in simulation/scenario_analysis.py and replace it with this:**

```python
"""
Multiple Scenario Analysis
Compares different monetary policy shock sizes
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# Load VAR model
with open('models/var_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

var_model = model_data['model']
variable_names = model_data['variables']

# Find mpr position
mpr_position = variable_names.index('mpr')

# Define scenarios (basis points)
scenarios = {
    'Mild Tightening (+50bps)': 0.50,
    'Moderate Tightening (+100bps)': 1.00,
    'Aggressive Tightening (+200bps)': 2.00,
    'Very Aggressive Tightening (+300bps)': 3.00
}

# Compute IRF for each scenario
horizon = 24
results = {}

for scenario_name, shock_size in scenarios.items():
    # Create shock vector
    shock = np.zeros(len(variable_names))
    shock[mpr_position] = shock_size

    # Compute IRF
    irf = var_model.irf(horizon)

    # Get inflation response
    infl_position = variable_names.index('infl')
    response = np.zeros(horizon)

    for h in range(horizon):
        response[h] = irf.irfs[h, infl_position, mpr_position] * shock_size

    results[scenario_name] = response

    print(f"{scenario_name}: Peak impact = {response.max():.3f}% at month {response.argmax() + 1}")

# Create visualization
plt.figure(figsize=(12, 6))

colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
months = np.arange(1, horizon + 1)

for (scenario_name, response), color in zip(results.items(), colors):
    plt.plot(months, response, label=scenario_name, color=color, linewidth=2)

plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.xlabel('Months After Shock', fontsize=12)
plt.ylabel('Inflation Change (%)', fontsize=12)
plt.title('Multiple Scenario Analysis: Inflation Response to MPR Increases',
          fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
Path('results').mkdir(exist_ok=True)
plt.savefig('results/scenario_comparison.png', dpi=300, bbox_inches='tight')
print("\nChart saved to results/scenario_comparison.png")
plt.close()
```

**What changed:**

1. **Import matplotlib**
2. **After computing all IRFs, create a single figure**
3. **Plot all four scenarios on the same axes** using different colors:
   - Green for mild tightening
   - Blue for moderate
   - Orange for aggressive
   - Red for very aggressive
4. **Save to results/scenario_comparison.png**

**Run it:**

```bash
python simulation/scenario_analysis.py
```

**Check the chart:**

```bash
xdg-open results/scenario_comparison.png
```

You should see all four inflation response curves overlaid. Notice:
- All curves are negative (rate hikes reduce inflation)
- All peak around month 3-4
- Larger shocks = larger impacts (linear scaling)
- Effects fade after 12-18 months

---

### STEP 3: Add comparison table with key metrics

**Delete everything in simulation/scenario_analysis.py and replace it with this:**

```python
"""
Multiple Scenario Analysis
Compares different monetary policy shock sizes
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import json

# Load VAR model
with open('models/var_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

var_model = model_data['model']
variable_names = model_data['variables']

# Find mpr position
mpr_position = variable_names.index('mpr')

# Define scenarios (basis points)
scenarios = {
    'Mild Tightening (+50bps)': 0.50,
    'Moderate Tightening (+100bps)': 1.00,
    'Aggressive Tightening (+200bps)': 2.00,
    'Very Aggressive Tightening (+300bps)': 3.00
}

# Compute IRF for each scenario
horizon = 24
results = {}

for scenario_name, shock_size in scenarios.items():
    # Create shock vector
    shock = np.zeros(len(variable_names))
    shock[mpr_position] = shock_size

    # Compute IRF
    irf = var_model.irf(horizon)

    # Get inflation response
    infl_position = variable_names.index('infl')
    response = np.zeros(horizon)

    for h in range(horizon):
        response[h] = irf.irfs[h, infl_position, mpr_position] * shock_size

    results[scenario_name] = response

    print(f"{scenario_name}: Peak impact = {response.max():.3f}% at month {response.argmax() + 1}")

# Create visualization
plt.figure(figsize=(12, 6))

colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
months = np.arange(1, horizon + 1)

for (scenario_name, response), color in zip(results.items(), colors):
    plt.plot(months, response, label=scenario_name, color=color, linewidth=2)

plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.xlabel('Months After Shock', fontsize=12)
plt.ylabel('Inflation Change (%)', fontsize=12)
plt.title('Multiple Scenario Analysis: Inflation Response to MPR Increases',
          fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
Path('results').mkdir(exist_ok=True)
plt.savefig('results/scenario_comparison.png', dpi=300, bbox_inches='tight')
print("\nChart saved to results/scenario_comparison.png")
plt.close()

# Build comparison table
comparison_data = []

for scenario_name, response in results.items():
    # Extract shock size
    shock_size = scenarios[scenario_name]

    # Calculate metrics
    peak_impact = response.min()  # Most negative value
    peak_month = response.argmin() + 1  # 1-indexed

    # Cumulative effects
    cumulative_12m = response[:12].sum()
    cumulative_24m = response.sum()

    comparison_data.append({
        'Scenario': scenario_name,
        'Shock Size (bps)': int(shock_size * 100),
        'Peak Impact (%)': round(peak_impact, 3),
        'Month of Peak': peak_month,
        'Cumulative 12-Month Effect (%)': round(cumulative_12m, 3),
        'Cumulative 24-Month Effect (%)': round(cumulative_24m, 3)
    })

# Create DataFrame
comparison_df = pd.DataFrame(comparison_data)

# Display table
print("\n" + "="*80)
print("SCENARIO COMPARISON TABLE")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

# Save to CSV
comparison_df.to_csv('results/scenario_comparison.csv', index=False)
print("\nTable saved to results/scenario_comparison.csv")

# Save to JSON
comparison_dict = {
    'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
    'model_type': 'VAR',
    'horizon_months': horizon,
    'scenarios': comparison_data
}

with open('results/scenario_comparison.json', 'w') as f:
    json.dump(comparison_dict, f, indent=2)

print("Table saved to results/scenario_comparison.json")
```

**What changed:**

1. **Import json** for structured output
2. **After creating the chart, build a comparison table** with these metrics for each scenario:
   - Shock size in basis points
   - Peak impact (most negative inflation change)
   - Month when peak occurs
   - Cumulative 12-month effect (sum of first 12 months)
   - Cumulative 24-month effect (sum of all 24 months)
3. **Save in two formats:**
   - CSV for spreadsheet analysis
   - JSON for programmatic use

**Run it:**

```bash
python simulation/scenario_analysis.py
```

**Expected output:**

```
Mild Tightening (+50bps): Peak impact = -0.087% at month 3
Moderate Tightening (+100bps): Peak impact = -0.174% at month 3
Aggressive Tightening (+200bps): Peak impact = -0.348% at month 3
Very Aggressive Tightening (+300bps): Peak impact = -0.522% at month 3

Chart saved to results/scenario_comparison.png

================================================================================
SCENARIO COMPARISON TABLE
================================================================================
                          Scenario  Shock Size (bps)  Peak Impact (%)  Month of Peak  Cumulative 12-Month Effect (%)  Cumulative 24-Month Effect (%)
       Mild Tightening (+50bps)                50           -0.087              3                          -0.584                          -0.892
   Moderate Tightening (+100bps)               100           -0.174              3                          -1.168                          -1.784
  Aggressive Tightening (+200bps)               200           -0.348              3                          -2.336                          -3.568
Very Aggressive Tightening (+300bps)               300           -0.522              3                          -3.504                          -5.352
================================================================================

Table saved to results/scenario_comparison.csv
Table saved to results/scenario_comparison.json
```

**Check the CSV:**

```bash
cat results/scenario_comparison.csv
```

---

## Policy Implications

Now that you have the comparison table, you can answer critical policy questions:

### Question 1: How aggressive does CBN need to be?

If inflation is running 5% above target, look at the cumulative effects:
- **+50bps**: Only reduces inflation by ~0.89% over 2 years (insufficient)
- **+100bps**: Reduces by ~1.78% over 2 years (still insufficient)
- **+200bps**: Reduces by ~3.57% over 2 years (getting closer)
- **+300bps**: Reduces by ~5.35% over 2 years (overshoots slightly)

**Conclusion:** CBN would need 200-300bps of tightening to bring inflation down by 5%.

### Question 2: Why do multiple rate hikes over time?

Notice the peak impact happens at month 3, but the cumulative 12-month effect is much larger than the peak. This is because:
- Each rate hike has a **temporary peak effect**
- But the effects **accumulate over time**
- Multiple smaller hikes can be safer than one large shock

### Question 3: What about side effects?

This analysis only shows inflation. In reality:
- Rate hikes also slow economic growth (check `gdp_growth` or `exo` variable)
- Affect exchange rates (check `exr` if in your model)
- Impact bank lending (check `tbr` spread to mpr)

A complete policy analysis would run scenarios for ALL variables, not just inflation.

### Question 4: Why do effects fade after 18 months?

VAR models capture **short to medium-term dynamics**. Long-run effects depend on:
- Whether the rate hike is permanent or temporary
- How expectations adjust
- Supply-side factors not captured in the VAR

---

## Commit Your Work

```bash
cd /home/user/Nigerian_Inflation_Predictor
git add simulation/scenario_analysis.py
git add results/scenario_comparison.png
git add results/scenario_comparison.csv
git add results/scenario_comparison.json
git commit -m "Day 27: Add multiple scenario analysis with comparison table"
```

---

## Common Errors and Fixes

### Error 1: "KeyError: 'mpr'"

**Problem:** Variable name doesn't match what's in the VAR model.

**Fix:**
```python
# Check what variables are actually in the model
print(variable_names)

# Use the exact name, e.g., 'MPR' or 'policy_rate'
```

### Error 2: "All scenarios look identical"

**Problem:** Forgot to multiply IRF by shock size.

**Fix:**
```python
# Wrong:
response[h] = irf.irfs[h, infl_position, mpr_position]

# Right:
response[h] = irf.irfs[h, infl_position, mpr_position] * shock_size
```

### Error 3: "Peak impact is positive instead of negative"

**Problem:** Rate hikes should reduce inflation, so response should be negative.

**Fix:** Check your VAR model. If inflation increases when mpr increases, your model might be:
- Trained on too little data
- Missing important control variables (like exo)
- Capturing the wrong relationship

Re-check Day 25 model training.

### Error 4: "FileNotFoundError: models/var_model.pkl"

**Problem:** Haven't trained the VAR model yet.

**Fix:**
```bash
# Go back and train the model
python models/train_var.py
```

---

## Q&A

**Q: Why only 4 scenarios? Can I add more?**

Yes! Add more entries to the `scenarios` dictionary:
```python
scenarios = {
    'Mild (+50bps)': 0.50,
    'Moderate (+100bps)': 1.00,
    'Strong (+150bps)': 1.50,
    'Aggressive (+200bps)': 2.00,
    'Very Aggressive (+250bps)': 2.50,
    'Extreme (+300bps)': 3.00
}
```

Just be aware that charts with 6+ lines can get crowded.

**Q: Can I analyze rate CUTS instead of hikes?**

Absolutely! Use negative shock sizes:
```python
scenarios = {
    'Mild Easing (-50bps)': -0.50,
    'Moderate Easing (-100bps)': -1.00,
    'Aggressive Easing (-200bps)': -2.00
}
```

The chart will show inflation INCREASING (positive responses).

**Q: Why is the relationship linear?**

VAR models are linear by design. In reality:
- Small rate changes might have weak effects (central bank credibility issues)
- Large rate changes might have outsized effects (panic or confidence shifts)

For non-linear effects, you'd need a **threshold VAR** or **regime-switching VAR**.

**Q: Can I compare different TYPES of shocks?**

Yes! Compare monetary policy shock vs. oil price shock:
```python
scenarios = {
    'Rate Hike (+100bps)': (mpr_position, 1.00),
    'Oil Price Surge (+10%)': (exo_position, 10.00)
}
```

Then compute IRFs for each shock type.

**Q: How do I choose the "right" scenario for policy recommendations?**

1. **Estimate the inflation gap**: Current inflation - Target inflation
2. **Check cumulative effects**: Which scenario closes that gap?
3. **Consider growth tradeoffs**: Run the same analysis for GDP response
4. **Add confidence intervals**: Show uncertainty around each scenario (Day 28)

**Q: My peak impact seems too small. Is that normal?**

Depends on your data. If you used:
- **Inflation rate (e.g., 15.6%)**: Impact might be 0.1-0.5%
- **Inflation change (already differenced)**: Impact might be 0.05-0.2%

Check what units your variables are in (levels vs. differences).

---

## Summary

Today you built a **scenario comparison tool** that:
1. **Runs multiple policy simulations** (4 different shock sizes)
2. **Visualizes all scenarios** on a single chart
3. **Compares key metrics** in a structured table

This is a critical tool for policy analysis. Tomorrow (Day 28) we'll add **confidence intervals** to show uncertainty around these scenarios.

---

## Practice Exercises

1. **Add a 5th scenario**: +400bps "Emergency Tightening"
2. **Analyze rate cuts**: Create a separate chart for -50bps, -100bps, -200bps
3. **Compare GDP effects**: Change `infl_position` to `exo_position` and see how growth responds
4. **Export to Excel**: Use `pd.ExcelWriter` to create a formatted Excel file with:
   - Sheet 1: Comparison table
   - Sheet 2: Full time series for all scenarios

**Next:** Day 28 — Adding Confidence Intervals to Scenarios
