# DAY 28 — Exchange Rate Shock Simulation

## What You'll Learn Today
- How to simulate an exchange rate shock (Naira depreciation)
- Exchange rate pass-through: how much of a depreciation feeds into inflation?
- Compare MPR shock vs EXO shock effects on inflation

---

## Building `simulation/exchange_rate_shock.py` — 3 steps

We'll create a script that simulates a sudden Naira depreciation and measures how it feeds into inflation over time.

---

### STEP 1: Imports, estimate VAR, compute IRF for EXO shock

Delete everything in `simulation/exchange_rate_shock.py` and replace it with this:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import seaborn as sns

# Load the data
df = pd.read_csv("data/processed/final_dataset.csv", index_col=0, parse_dates=True)

# Select variables in order: mpr, infl, exo, tbr
var_columns = ["mpr", "infl", "exo", "tbr"]
data = df[var_columns].dropna()

# Estimate VAR model with 6 lags
model = VAR(data)
results = model.fit(maxlags=6, ic="aic")

print("VAR Model Summary:")
print(results.summary())

# Compute IRF for a 1-unit shock to EXO (exchange rate)
# This simulates a sudden Naira depreciation
irf = results.irf(24)

# Extract the effect of EXO shock on inflation
exo_shock_on_infl = irf.irfs[:, 1, 2]  # [periods, response_var=infl, impulse_var=exo]

print("\nExchange Rate Shock Effect on Inflation:")
print(exo_shock_on_infl)
```

**What this does:**
- Loads the final dataset
- Estimates a VAR model with our 4 variables: mpr, infl, exo, tbr
- Computes the Impulse Response Function (IRF) for a 1-unit shock to the exchange rate (exo)
- Extracts how inflation responds over 24 months

**Key concept:** When the Naira depreciates (exo increases), imported goods become more expensive. This feeds into inflation. We're measuring how much and how fast.

---

### STEP 2: Complete file with pass-through comparison plot

Delete everything in `simulation/exchange_rate_shock.py` and replace it with this:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import seaborn as sns

# Load the data
df = pd.read_csv("data/processed/final_dataset.csv", index_col=0, parse_dates=True)

# Select variables in order: mpr, infl, exo, tbr
var_columns = ["mpr", "infl", "exo", "tbr"]
data = df[var_columns].dropna()

# Estimate VAR model with 6 lags
model = VAR(data)
results = model.fit(maxlags=6, ic="aic")

print("VAR Model Summary:")
print(results.summary())

# Compute IRF for 24 months
irf = results.irf(24)

# Extract the effect of EXO shock on inflation
exo_shock_on_infl = irf.irfs[:, 1, 2]  # [periods, response_var=infl, impulse_var=exo]

# Also extract MPR shock on inflation for comparison
mpr_shock_on_infl = irf.irfs[:, 1, 0]  # [periods, response_var=infl, impulse_var=mpr]

print("\nExchange Rate Shock Effect on Inflation:")
print(exo_shock_on_infl)

# Plot the pass-through comparison
plt.figure(figsize=(12, 6))

# Plot EXO shock effect on inflation
plt.subplot(1, 2, 1)
plt.plot(range(24), exo_shock_on_infl, linewidth=2, color="red", label="EXO → Inflation")
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel("Months After Shock", fontsize=12)
plt.ylabel("Inflation Response (pp)", fontsize=12)
plt.title("Exchange Rate Pass-Through to Inflation", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend()

# Plot comparison: EXO vs MPR shock effects
plt.subplot(1, 2, 2)
plt.plot(range(24), exo_shock_on_infl, linewidth=2, color="red", label="EXO Shock → Inflation")
plt.plot(range(24), mpr_shock_on_infl, linewidth=2, color="blue", label="MPR Shock → Inflation")
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel("Months After Shock", fontsize=12)
plt.ylabel("Inflation Response (pp)", fontsize=12)
plt.title("Monetary Policy vs Exchange Rate Shocks", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("results/simulation/exchange_rate_pass_through.png", dpi=300, bbox_inches="tight")
print("\nPlot saved to: results/simulation/exchange_rate_pass_through.png")
plt.show()
```

**What changed:**
- Added extraction of MPR shock effect on inflation for comparison
- Created a two-panel plot:
  - Left: Exchange rate pass-through to inflation
  - Right: Comparison of EXO shock vs MPR shock effects
- Saved the plot to results folder

**What to look for:**
- Does the EXO shock cause inflation to rise? (It should, due to import dependency)
- How long does the effect last?
- Is the EXO shock effect larger than the MPR shock effect? (Often yes in import-dependent economies)

---

### STEP 3: Complete file with pass-through coefficient calculation

Delete everything in `simulation/exchange_rate_shock.py` and replace it with this:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import seaborn as sns
import os

# Create results directory if it doesn't exist
os.makedirs("results/simulation", exist_ok=True)

# Load the data
df = pd.read_csv("data/processed/final_dataset.csv", index_col=0, parse_dates=True)

# Select variables in order: mpr, infl, exo, tbr
var_columns = ["mpr", "infl", "exo", "tbr"]
data = df[var_columns].dropna()

# Estimate VAR model with 6 lags
model = VAR(data)
results = model.fit(maxlags=6, ic="aic")

print("VAR Model Summary:")
print(results.summary())

# Compute IRF for 24 months
irf = results.irf(24)

# Extract the effect of EXO shock on inflation
exo_shock_on_infl = irf.irfs[:, 1, 2]  # [periods, response_var=infl, impulse_var=exo]

# Also extract MPR shock on inflation for comparison
mpr_shock_on_infl = irf.irfs[:, 1, 0]  # [periods, response_var=infl, impulse_var=mpr]

# Extract EXO shock on itself (to verify shock magnitude)
exo_shock_on_exo = irf.irfs[:, 2, 2]  # [periods, response_var=exo, impulse_var=exo]

print("\nExchange Rate Shock Effect on Inflation:")
print(exo_shock_on_infl)

# Compute pass-through coefficient
# Pass-through = (cumulative inflation response) / (exchange rate shock magnitude)
cumulative_infl_response_6m = np.sum(exo_shock_on_infl[:6])
cumulative_infl_response_12m = np.sum(exo_shock_on_infl[:12])
cumulative_infl_response_24m = np.sum(exo_shock_on_infl)

shock_magnitude = exo_shock_on_exo[0]  # Initial shock to exchange rate

pass_through_6m = (cumulative_infl_response_6m / shock_magnitude) * 100
pass_through_12m = (cumulative_infl_response_12m / shock_magnitude) * 100
pass_through_24m = (cumulative_infl_response_24m / shock_magnitude) * 100

print("\n" + "="*60)
print("EXCHANGE RATE PASS-THROUGH COEFFICIENTS")
print("="*60)
print(f"Shock Magnitude: {shock_magnitude:.4f}")
print(f"6-Month Pass-Through:  {pass_through_6m:.2f}%")
print(f"12-Month Pass-Through: {pass_through_12m:.2f}%")
print(f"24-Month Pass-Through: {pass_through_24m:.2f}%")
print("="*60)

# Interpretation
print("\nInterpretation:")
print(f"A 1% depreciation of the Naira leads to:")
print(f"  - {pass_through_6m:.2f}% increase in inflation after 6 months")
print(f"  - {pass_through_12m:.2f}% increase in inflation after 12 months")
print(f"  - {pass_through_24m:.2f}% increase in inflation after 24 months")

# Plot the pass-through comparison
plt.figure(figsize=(14, 6))

# Plot 1: EXO shock effect on inflation
plt.subplot(1, 3, 1)
plt.plot(range(24), exo_shock_on_infl, linewidth=2.5, color="red", label="EXO → Inflation")
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel("Months After Shock", fontsize=11)
plt.ylabel("Inflation Response (pp)", fontsize=11)
plt.title("Exchange Rate Pass-Through", fontsize=13, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Comparison - EXO vs MPR shock effects
plt.subplot(1, 3, 2)
plt.plot(range(24), exo_shock_on_infl, linewidth=2.5, color="red", label="EXO Shock → Inflation")
plt.plot(range(24), mpr_shock_on_infl, linewidth=2.5, color="blue", label="MPR Shock → Inflation")
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel("Months After Shock", fontsize=11)
plt.ylabel("Inflation Response (pp)", fontsize=11)
plt.title("Monetary vs Exchange Rate Shocks", fontsize=13, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 3: Cumulative pass-through over time
cumulative_response = np.cumsum(exo_shock_on_infl)
cumulative_pass_through = (cumulative_response / shock_magnitude) * 100

plt.subplot(1, 3, 3)
plt.plot(range(24), cumulative_pass_through, linewidth=2.5, color="darkgreen", label="Cumulative Pass-Through")
plt.axhline(y=pass_through_12m, color="orange", linestyle="--", linewidth=1.5, label=f"12-Month: {pass_through_12m:.1f}%")
plt.xlabel("Months After Shock", fontsize=11)
plt.ylabel("Cumulative Pass-Through (%)", fontsize=11)
plt.title("Cumulative Exchange Rate Pass-Through", fontsize=13, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("results/simulation/exchange_rate_pass_through.png", dpi=300, bbox_inches="tight")
print("\nPlot saved to: results/simulation/exchange_rate_pass_through.png")
plt.show()

# Save results to CSV
results_df = pd.DataFrame({
    "Month": range(1, 25),
    "EXO_Shock_on_Inflation": exo_shock_on_infl,
    "MPR_Shock_on_Inflation": mpr_shock_on_infl,
    "Cumulative_Pass_Through_Percent": cumulative_pass_through
})

results_df.to_csv("results/simulation/exchange_rate_shock_results.csv", index=False)
print("Results saved to: results/simulation/exchange_rate_shock_results.csv")

# Save summary statistics
summary = {
    "Shock_Magnitude": shock_magnitude,
    "Pass_Through_6M_Percent": pass_through_6m,
    "Pass_Through_12M_Percent": pass_through_12m,
    "Pass_Through_24M_Percent": pass_through_24m,
    "Peak_Inflation_Response": np.max(exo_shock_on_infl),
    "Peak_Inflation_Response_Month": np.argmax(exo_shock_on_infl) + 1
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv("results/simulation/exchange_rate_pass_through_summary.csv", index=False)
print("Summary saved to: results/simulation/exchange_rate_pass_through_summary.csv")
```

**What changed:**
- Computed pass-through coefficients at 6, 12, and 24 months
- Added a third plot showing cumulative pass-through over time
- Saved detailed results and summary statistics to CSV files
- Added interpretation of results

**Pass-through coefficient:** This tells us what percentage of an exchange rate depreciation feeds into inflation. For example, if the pass-through is 30%, then a 10% Naira depreciation would cause inflation to rise by 3 percentage points.

---

## Run the Simulation

```bash
cd /home/user/Nigerian_Inflation_Predictor
python simulation/exchange_rate_shock.py
```

**Expected output:**
- VAR model summary
- Exchange rate pass-through coefficients
- Three-panel plot showing:
  1. Exchange rate pass-through to inflation
  2. Comparison of EXO vs MPR shocks
  3. Cumulative pass-through over time
- CSV files with detailed results

---

## Nigerian Context: Exchange Rate and Inflation

### June 2023 Devaluation
In June 2023, the CBN unified Nigeria's multiple exchange rates and allowed the Naira to float more freely. The Naira depreciated sharply from around ₦460/$ to over ₦750/$ within weeks. This is a real-world example of the exchange rate shock we're simulating.

**What happened to inflation?**
- Inflation rose from 22.4% (May 2023) to 25.8% (August 2023)
- Food inflation increased even more sharply due to import dependency
- The pass-through was significant but not 100% due to:
  - Local production substitution
  - Price controls on some goods
  - Delayed adjustment in some sectors

### Why Exchange Rate Matters for Nigerian Inflation

1. **Import Dependency**
   - Nigeria imports a large share of food (wheat, rice, sugar)
   - All petroleum products are imported (refined abroad)
   - Manufacturing inputs are heavily imported
   - When Naira weakens, all these become more expensive

2. **Fuel Subsidy Removal**
   - May 2023: Fuel subsidy was removed
   - Combined with Naira depreciation, petrol prices tripled
   - Fuel is used in transportation and production, so affects all prices

3. **Second-Round Effects**
   - Workers demand higher wages to cope with rising prices
   - Businesses raise prices to cover higher costs
   - Creates a wage-price spiral

### Pass-Through Estimates from Literature

Research on Nigeria's exchange rate pass-through:

- **Bada (2014):** Estimated 30-40% pass-through in first year
- **Imimole & Enoma (2011):** Found 15-25% pass-through after 6 months
- **Adebiyi et al. (2019):** Estimated 35-50% for food prices (higher than overall inflation)

**Why not 100%?**
- Some goods are produced locally and don't depend on imports
- Price stickiness: businesses are slow to adjust prices
- Policy interventions: price controls, subsidies
- Exchange rate expectations: if people expect Naira to strengthen again, they may not raise prices fully

Your model's estimate should fall somewhere in this range. If it's outside 15-50%, investigate why.

---

## Commit Your Work

```bash
git add simulation/exchange_rate_shock.py
git add results/simulation/exchange_rate_pass_through.png
git add results/simulation/exchange_rate_shock_results.csv
git add results/simulation/exchange_rate_pass_through_summary.csv
git commit -m "Add exchange rate shock simulation and pass-through analysis"
```

---

## Common Errors and Fixes

### Error 1: "File not found: data/processed/final_dataset.csv"
**Fix:** Make sure you've completed Day 1-7 (data preparation). Run the data processing scripts first.

### Error 2: VAR model doesn't converge
**Fix:** Try different lag orders. Replace `maxlags=6` with `maxlags=4` or `maxlags=8`.

### Error 3: Pass-through coefficient is negative
**Why:** In the data, exchange rate depreciation might be coded differently (negative values = depreciation). Or there's a structural break.
**Fix:** Check the sign of your exchange rate variable. You might need to multiply by -1.

### Error 4: "results/simulation directory does not exist"
**Fix:** The script creates it automatically with `os.makedirs()`. If it still fails, create it manually:
```bash
mkdir -p results/simulation
```

---

## Q&A

**Q1: What if my pass-through coefficient is very high (>80%)?**

**A:** This could mean:
- Your data period includes a major devaluation with high pass-through
- The exchange rate variable is measuring black market rates (which have higher pass-through)
- There's high import dependency in your sample period

Check your data period. If it includes June 2023 devaluation, high pass-through makes sense.

---

**Q2: Why compare EXO shock with MPR shock?**

**A:** To understand the relative importance of different shocks:
- If EXO shock has a bigger effect on inflation than MPR shock, it means exchange rate policy matters more than interest rate policy for inflation
- This is common in emerging markets with high import dependency
- Helps policymakers decide where to focus: defending the exchange rate vs raising interest rates

---

**Q3: What's the difference between "impact pass-through" and "cumulative pass-through"?**

**A:**
- **Impact pass-through:** Immediate effect in the first month after shock
- **Cumulative pass-through:** Total effect over several months

Cumulative is usually larger because effects take time to feed through:
- Month 1: Importers face higher costs
- Month 2-3: Retailers raise prices
- Month 4-6: Wages adjust, causing second-round effects

Our code calculates cumulative pass-through at 6, 12, and 24 months.

---

**Q4: How can the CBN reduce exchange rate pass-through?**

**A:** Several strategies:
1. **Reduce import dependency:** Promote local production
2. **Anchor expectations:** Convince people Naira won't depreciate further
3. **Price controls:** Limit how much businesses can raise prices (but causes other problems)
4. **Increase foreign reserves:** More reserves = more stable exchange rate = lower pass-through
5. **Improve productivity:** If businesses become more efficient, they can absorb cost increases

Nigeria has tried all of these with mixed success.

---

**Q5: Should the CBN defend the Naira to control inflation?**

**A:** This is a major policy debate:

**Arguments for defending Naira:**
- Reduces imported inflation
- Maintains purchasing power
- Prevents panic and capital flight

**Arguments against:**
- Burns through foreign reserves quickly
- Creates parallel market distortions
- Delays necessary adjustments
- Can't be sustained if economic fundamentals are weak

The June 2023 decision to let Naira float was partly because CBN reserves were low and the defense was unsustainable.

---

**Q6: How is this different from Day 27's shock scenarios?**

**A:**
- **Day 27:** Simulated multiple shock sizes (1%, 2%, 5%) for all variables
- **Day 28:** Deep dive into exchange rate shock specifically, calculating pass-through coefficients

Day 28 gives you more detailed analysis of exchange rate dynamics.

---

**Q7: What if I want to simulate the actual June 2023 devaluation?**

**A:** Modify the shock size:
- June 2023: Naira went from ₦460 to ₦750 = 63% depreciation
- Your shock should be scaled accordingly
- Change `irf = results.irf(24)` to include a custom shock size
- Or multiply the IRF results by 0.63 to scale to actual devaluation

---

## Tomorrow: Day 29

**Topic:** Oil Price Shock Simulation

We'll explore how oil price changes affect Nigeria's economy:
- Oil exports are Nigeria's main revenue source
- Higher oil prices = more government revenue
- But also higher fuel prices for consumers
- Measure the net effect on inflation

See you tomorrow!
