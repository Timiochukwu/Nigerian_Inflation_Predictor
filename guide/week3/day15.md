# Day 15 — ARDL Diagnostics: Validating Your Model

Welcome to Day 15! Yesterday you extracted the Error Correction Model from your ARDL. Today you'll learn the final critical step: **diagnostic testing**. Your model is only scientifically valid if the residuals behave properly. We'll run three statistical tests and create diagnostic plots.

---

## What You'll Learn Today

By the end of this session, you will:

1. **Understand why diagnostics are mandatory** — A model with poor diagnostics produces unreliable forecasts
2. **Run the Breusch-Godfrey test** — Detect serial correlation in residuals
3. **Run the Breusch-Pagan test** — Detect heteroscedasticity (non-constant variance)
4. **Run the Jarque-Bera test** — Test for normality of residuals
5. **Create residual diagnostic plots** — Visual inspection of residual behavior
6. **Interpret test results** — Know what to do when tests fail

**Key concept**: If residuals show patterns (serial correlation, heteroscedasticity), your model is misspecified and standard errors are incorrect.

---

## Theory: The 3 Diagnostic Tests

### 1. Breusch-Godfrey Test (Serial Correlation)

**What it tests**: Whether residuals are correlated with their past values.

**Why it matters**: Serial correlation means your model missed important dynamics. Standard errors become unreliable.

**Null hypothesis (H0)**: No serial correlation exists.

**Decision rule**:
- p-value > 0.05 → **PASS** (no serial correlation detected)
- p-value ≤ 0.05 → **FAIL** (serial correlation present)

**If you fail**: Increase the lag order of your ARDL model. For example, if ARDL(1,1,1,1) fails, try ARDL(2,2,2,2).

---

### 2. Breusch-Pagan Test (Heteroscedasticity)

**What it tests**: Whether residual variance is constant over time.

**Why it matters**: Heteroscedasticity (changing variance) makes confidence intervals and p-values unreliable.

**Null hypothesis (H0)**: Residuals have constant variance (homoscedasticity).

**Decision rule**:
- p-value > 0.05 → **PASS** (constant variance)
- p-value ≤ 0.05 → **FAIL** (heteroscedasticity detected)

**If you fail**: Use HAC (Heteroscedasticity and Autocorrelation Consistent) standard errors, also called Newey-West standard errors.

---

### 3. Jarque-Bera Test (Normality)

**What it tests**: Whether residuals follow a normal distribution.

**Why it matters**: Many statistical inferences assume normality. However, this is the **least critical** test — with large samples (100+ observations), violations are often acceptable.

**Null hypothesis (H0)**: Residuals are normally distributed.

**Decision rule**:
- p-value > 0.05 → **PASS** (residuals are normal)
- p-value ≤ 0.05 → **FAIL** (residuals are non-normal)

**If you fail**: Usually not a major problem with 100+ observations due to the Central Limit Theorem. Extreme outliers may indicate structural breaks.

---

## Building econometric_models/ardl_diagnostics.py

We'll build this file in **3 steps**. Each step shows the **COMPLETE file** — delete everything and replace it.

---

### STEP 1: Imports + Breusch-Godfrey Test

**Delete everything in `econometric_models/ardl_diagnostics.py` and replace it with this:**

```python
"""ARDL diagnostic tests for the Nigerian Inflation Predictor."""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from scipy.stats import jarque_bera

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Results directory for saving outputs
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_breusch_godfrey(result, nlags=4):
    """
    Run Breusch-Godfrey test for serial correlation.

    Parameters
    ----------
    result : RegressionResults
        Fitted ARDL model results from statsmodels
    nlags : int
        Number of lags to test for serial correlation

    Returns
    -------
    dict
        Dictionary with test results
    """
    lm_stat, lm_pvalue, f_stat, f_pvalue = acorr_breusch_godfrey(result, nlags=nlags)
    passed = lm_pvalue > 0.05

    print(f"\nBreusch-Godfrey Test (Serial Correlation):")
    print(f"  LM statistic: {lm_stat:.4f}")
    print(f"  p-value: {lm_pvalue:.4f}")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    if not passed:
        print(f"  → Serial correlation detected. Consider increasing lag order.")

    return {
        "test": "Breusch-Godfrey",
        "statistic": lm_stat,
        "p_value": lm_pvalue,
        "passed": passed
    }


if __name__ == "__main__":
    # Import ARDL model functions
    from econometric_models.ardl_model import load_data, estimate_ardl

    print("=" * 60)
    print("ARDL DIAGNOSTIC TESTS")
    print("=" * 60)

    # Load data
    print("\nLoading cleaned data...")
    data = load_data()
    print(f"Loaded {len(data)} observations")

    # Estimate ARDL model
    print("\nEstimating ARDL(1,1,1,1) model...")
    result = estimate_ardl(data, lags={"infl": 1, "mpr": 1, "exo": 1, "tbr": 1})

    # Run Breusch-Godfrey test
    bg_result = run_breusch_godfrey(result, nlags=4)

    print("\n" + "=" * 60)
```

**What this does**:
- Imports all necessary libraries for diagnostic testing
- Creates `run_breusch_godfrey()` function that tests for serial correlation
- The function uses the LM (Lagrange Multiplier) test with 4 lags by default
- Tests if p-value > 0.05 (PASS) or ≤ 0.05 (FAIL)
- In `__main__`: loads data, estimates ARDL model, runs BG test

**Run it**:
```bash
cd /home/user/Nigerian_Inflation_Predictor
python econometric_models/ardl_diagnostics.py
```

You should see the Breusch-Godfrey test result with a p-value.

---

### STEP 2: Add Breusch-Pagan, Jarque-Bera, and Summary Function

**Delete everything in `econometric_models/ardl_diagnostics.py` and replace it with this:**

```python
"""ARDL diagnostic tests for the Nigerian Inflation Predictor."""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from scipy.stats import jarque_bera

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Results directory for saving outputs
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_breusch_godfrey(result, nlags=4):
    """
    Run Breusch-Godfrey test for serial correlation.

    Parameters
    ----------
    result : RegressionResults
        Fitted ARDL model results from statsmodels
    nlags : int
        Number of lags to test for serial correlation

    Returns
    -------
    dict
        Dictionary with test results
    """
    lm_stat, lm_pvalue, f_stat, f_pvalue = acorr_breusch_godfrey(result, nlags=nlags)
    passed = lm_pvalue > 0.05

    print(f"\nBreusch-Godfrey Test (Serial Correlation):")
    print(f"  LM statistic: {lm_stat:.4f}")
    print(f"  p-value: {lm_pvalue:.4f}")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    if not passed:
        print(f"  → Serial correlation detected. Consider increasing lag order.")

    return {
        "test": "Breusch-Godfrey",
        "statistic": lm_stat,
        "p_value": lm_pvalue,
        "passed": passed
    }


def run_breusch_pagan(result):
    """
    Run Breusch-Pagan test for heteroscedasticity.

    Parameters
    ----------
    result : RegressionResults
        Fitted ARDL model results from statsmodels

    Returns
    -------
    dict
        Dictionary with test results
    """
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(result.resid, result.model.exog)
    passed = lm_pvalue > 0.05

    print(f"\nBreusch-Pagan Test (Heteroscedasticity):")
    print(f"  LM statistic: {lm_stat:.4f}")
    print(f"  p-value: {lm_pvalue:.4f}")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    if not passed:
        print(f"  → Heteroscedasticity detected. Consider using HAC standard errors.")

    return {
        "test": "Breusch-Pagan",
        "statistic": lm_stat,
        "p_value": lm_pvalue,
        "passed": passed
    }


def run_jarque_bera(result):
    """
    Run Jarque-Bera test for normality of residuals.

    Parameters
    ----------
    result : RegressionResults
        Fitted ARDL model results from statsmodels

    Returns
    -------
    dict
        Dictionary with test results
    """
    jb_stat, jb_pvalue = jarque_bera(result.resid)
    passed = jb_pvalue > 0.05

    print(f"\nJarque-Bera Test (Normality):")
    print(f"  JB statistic: {jb_stat:.4f}")
    print(f"  p-value: {jb_pvalue:.4f}")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    if not passed:
        print(f"  → Residuals are non-normal. Usually acceptable with large samples.")

    return {
        "test": "Jarque-Bera",
        "statistic": jb_stat,
        "p_value": jb_pvalue,
        "passed": passed
    }


def run_all_diagnostics(result, nlags=4):
    """
    Run all diagnostic tests and print summary table.

    Parameters
    ----------
    result : RegressionResults
        Fitted ARDL model results from statsmodels
    nlags : int
        Number of lags for Breusch-Godfrey test

    Returns
    -------
    list of dict
        List containing results from all diagnostic tests
    """
    print("\n" + "=" * 60)
    print("RUNNING ALL DIAGNOSTIC TESTS")
    print("=" * 60)

    # Run all three tests
    bg_result = run_breusch_godfrey(result, nlags=nlags)
    bp_result = run_breusch_pagan(result)
    jb_result = run_jarque_bera(result)

    # Combine results
    all_results = [bg_result, bp_result, jb_result]

    # Print summary table
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"{'Test':<25} {'Statistic':<12} {'p-value':<12} {'Result':<10}")
    print("-" * 60)

    for res in all_results:
        status = "PASS ✓" if res["passed"] else "FAIL ✗"
        print(f"{res['test']:<25} {res['statistic']:<12.4f} {res['p_value']:<12.4f} {status:<10}")

    # Overall assessment
    all_passed = all(res["passed"] for res in all_results)
    print("-" * 60)

    if all_passed:
        print("Overall: ALL TESTS PASSED ✓✓✓")
        print("Your ARDL model residuals are well-behaved.")
    else:
        failed_tests = [res["test"] for res in all_results if not res["passed"]]
        print(f"Overall: SOME TESTS FAILED")
        print(f"Failed tests: {', '.join(failed_tests)}")
        print("Review individual test recommendations above.")

    print("=" * 60)

    return all_results


if __name__ == "__main__":
    # Import ARDL model functions
    from econometric_models.ardl_model import load_data, estimate_ardl

    print("=" * 60)
    print("ARDL DIAGNOSTIC TESTS")
    print("=" * 60)

    # Load data
    print("\nLoading cleaned data...")
    data = load_data()
    print(f"Loaded {len(data)} observations")

    # Estimate ARDL model
    print("\nEstimating ARDL(1,1,1,1) model...")
    result = estimate_ardl(data, lags={"infl": 1, "mpr": 1, "exo": 1, "tbr": 1})

    # Run all diagnostic tests
    diagnostic_results = run_all_diagnostics(result, nlags=4)

    print("\nDiagnostic testing complete!")
```

**What's new**:
- `run_breusch_pagan()`: Tests for heteroscedasticity using the BP test
- `run_jarque_bera()`: Tests for normality of residuals
- `run_all_diagnostics()`: Runs all three tests and prints a clean summary table
- Shows whether each test passed or failed
- Provides overall assessment at the end

**Run it**:
```bash
python econometric_models/ardl_diagnostics.py
```

You should now see all three diagnostic tests plus a summary table.

---

### STEP 3: Add Residual Plots and Save Results

**Delete everything in `econometric_models/ardl_diagnostics.py` and replace it with this:**

```python
"""ARDL diagnostic tests for the Nigerian Inflation Predictor."""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from scipy.stats import jarque_bera, probplot

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Results directory for saving outputs
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_breusch_godfrey(result, nlags=4):
    """
    Run Breusch-Godfrey test for serial correlation.

    Parameters
    ----------
    result : RegressionResults
        Fitted ARDL model results from statsmodels
    nlags : int
        Number of lags to test for serial correlation

    Returns
    -------
    dict
        Dictionary with test results
    """
    lm_stat, lm_pvalue, f_stat, f_pvalue = acorr_breusch_godfrey(result, nlags=nlags)
    passed = lm_pvalue > 0.05

    print(f"\nBreusch-Godfrey Test (Serial Correlation):")
    print(f"  LM statistic: {lm_stat:.4f}")
    print(f"  p-value: {lm_pvalue:.4f}")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    if not passed:
        print(f"  → Serial correlation detected. Consider increasing lag order.")

    return {
        "test": "Breusch-Godfrey",
        "statistic": lm_stat,
        "p_value": lm_pvalue,
        "passed": passed
    }


def run_breusch_pagan(result):
    """
    Run Breusch-Pagan test for heteroscedasticity.

    Parameters
    ----------
    result : RegressionResults
        Fitted ARDL model results from statsmodels

    Returns
    -------
    dict
        Dictionary with test results
    """
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(result.resid, result.model.exog)
    passed = lm_pvalue > 0.05

    print(f"\nBreusch-Pagan Test (Heteroscedasticity):")
    print(f"  LM statistic: {lm_stat:.4f}")
    print(f"  p-value: {lm_pvalue:.4f}")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    if not passed:
        print(f"  → Heteroscedasticity detected. Consider using HAC standard errors.")

    return {
        "test": "Breusch-Pagan",
        "statistic": lm_stat,
        "p_value": lm_pvalue,
        "passed": passed
    }


def run_jarque_bera(result):
    """
    Run Jarque-Bera test for normality of residuals.

    Parameters
    ----------
    result : RegressionResults
        Fitted ARDL model results from statsmodels

    Returns
    -------
    dict
        Dictionary with test results
    """
    jb_stat, jb_pvalue = jarque_bera(result.resid)
    passed = jb_pvalue > 0.05

    print(f"\nJarque-Bera Test (Normality):")
    print(f"  JB statistic: {jb_stat:.4f}")
    print(f"  p-value: {jb_pvalue:.4f}")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    if not passed:
        print(f"  → Residuals are non-normal. Usually acceptable with large samples.")

    return {
        "test": "Jarque-Bera",
        "statistic": jb_stat,
        "p_value": jb_pvalue,
        "passed": passed
    }


def run_all_diagnostics(result, nlags=4):
    """
    Run all diagnostic tests and print summary table.

    Parameters
    ----------
    result : RegressionResults
        Fitted ARDL model results from statsmodels
    nlags : int
        Number of lags for Breusch-Godfrey test

    Returns
    -------
    list of dict
        List containing results from all diagnostic tests
    """
    print("\n" + "=" * 60)
    print("RUNNING ALL DIAGNOSTIC TESTS")
    print("=" * 60)

    # Run all three tests
    bg_result = run_breusch_godfrey(result, nlags=nlags)
    bp_result = run_breusch_pagan(result)
    jb_result = run_jarque_bera(result)

    # Combine results
    all_results = [bg_result, bp_result, jb_result]

    # Print summary table
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"{'Test':<25} {'Statistic':<12} {'p-value':<12} {'Result':<10}")
    print("-" * 60)

    for res in all_results:
        status = "PASS ✓" if res["passed"] else "FAIL ✗"
        print(f"{res['test']:<25} {res['statistic']:<12.4f} {res['p_value']:<12.4f} {status:<10}")

    # Overall assessment
    all_passed = all(res["passed"] for res in all_results)
    print("-" * 60)

    if all_passed:
        print("Overall: ALL TESTS PASSED ✓✓✓")
        print("Your ARDL model residuals are well-behaved.")
    else:
        failed_tests = [res["test"] for res in all_results if not res["passed"]]
        print(f"Overall: SOME TESTS FAILED")
        print(f"Failed tests: {', '.join(failed_tests)}")
        print("Review individual test recommendations above.")

    print("=" * 60)

    return all_results


def plot_residuals(result, data):
    """
    Create 4-panel residual diagnostic plots.

    Parameters
    ----------
    result : RegressionResults
        Fitted ARDL model results from statsmodels
    data : pd.DataFrame
        Original data with DatetimeIndex

    Returns
    -------
    str
        Path to saved plot file
    """
    print("\nCreating residual diagnostic plots...")

    # Extract residuals and fitted values
    residuals = result.resid
    fitted_values = result.fittedvalues

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ARDL Residual Diagnostics", fontsize=16, fontweight="bold")

    # Panel 1: Residuals over time
    ax1 = axes[0, 0]
    ax1.plot(residuals.index, residuals, linewidth=0.8, color="blue", alpha=0.7)
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    ax1.set_title("Residuals Over Time")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Residuals")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Histogram with normal curve overlay
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=30, density=True, alpha=0.7, color="skyblue", edgecolor="black")

    # Overlay normal distribution curve
    mu = residuals.mean()
    sigma = residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
             color="red", linewidth=2, label="Normal Distribution")

    ax2.set_title("Histogram of Residuals")
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Q-Q plot
    ax3 = axes[1, 0]
    (osm, osr), (slope, intercept, r) = probplot(residuals, dist="norm")
    ax3.plot(osm, osr, "o", color="blue", alpha=0.6, markersize=4)
    ax3.plot(osm, slope * osm + intercept, "r-", linewidth=2, label="45° line")
    ax3.set_title("Q-Q Plot")
    ax3.set_xlabel("Theoretical Quantiles")
    ax3.set_ylabel("Sample Quantiles")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Residuals vs Fitted Values
    ax4 = axes[1, 1]
    ax4.scatter(fitted_values, residuals, alpha=0.6, s=20, color="green")
    ax4.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    ax4.set_title("Residuals vs Fitted Values")
    ax4.set_xlabel("Fitted Values")
    ax4.set_ylabel("Residuals")
    ax4.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(RESULTS_DIR, "ardl_residual_diagnostics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Residual plots saved to: {plot_path}")

    return plot_path


if __name__ == "__main__":
    # Import ARDL model functions
    from econometric_models.ardl_model import load_data, estimate_ardl

    print("=" * 60)
    print("ARDL DIAGNOSTIC TESTS")
    print("=" * 60)

    # Load data
    print("\nLoading cleaned data...")
    data = load_data()
    print(f"Loaded {len(data)} observations")

    # Estimate ARDL model
    print("\nEstimating ARDL(1,1,1,1) model...")
    result = estimate_ardl(data, lags={"infl": 1, "mpr": 1, "exo": 1, "tbr": 1})

    # Run all diagnostic tests
    diagnostic_results = run_all_diagnostics(result, nlags=4)

    # Create residual plots
    plot_path = plot_residuals(result, data)

    # Save diagnostic results to CSV
    print("\nSaving diagnostic results...")
    diagnostic_df = pd.DataFrame(diagnostic_results)
    csv_path = os.path.join(RESULTS_DIR, "ardl_diagnostics.csv")
    diagnostic_df.to_csv(csv_path, index=False)
    print(f"Diagnostic results saved to: {csv_path}")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC TESTING COMPLETE")
    print("=" * 60)
    print(f"Summary CSV: {csv_path}")
    print(f"Plots: {plot_path}")
```

**What's new**:
- `plot_residuals()`: Creates a 4-panel figure with:
  1. **Residuals over time**: Should fluctuate randomly around zero
  2. **Histogram with normal curve**: Should roughly match the red normal curve
  3. **Q-Q plot**: Points should lie close to the 45-degree line if residuals are normal
  4. **Residuals vs fitted values**: Should show no patterns (random scatter)
- Saves plot to `results/ardl_residual_diagnostics.png`
- Saves diagnostic test results to `results/ardl_diagnostics.csv`
- Complete `__main__` that runs everything: estimate → test → plot → save

**Run it**:
```bash
python econometric_models/ardl_diagnostics.py
```

**Check your outputs**:
```bash
ls -lh results/ardl_diagnostics.csv
ls -lh results/ardl_residual_diagnostics.png
```

Open `ardl_residual_diagnostics.png` to visually inspect your residuals.

---

## Interpreting the Diagnostic Plots

### 1. Residuals Over Time
- **Good**: Random fluctuations around zero with no patterns
- **Bad**: Systematic patterns, trends, or clustering suggest misspecification

### 2. Histogram with Normal Curve
- **Good**: Histogram bars roughly match the red normal curve
- **Bad**: Extreme skewness or heavy tails (many outliers)

### 3. Q-Q Plot
- **Good**: Points lie close to the 45-degree red line
- **Bad**: Severe departures at the tails indicate non-normality

### 4. Residuals vs Fitted Values
- **Good**: Random scatter with no patterns
- **Bad**: Funnel shape (heteroscedasticity) or curves (nonlinearity)

---

## What to Do If Tests Fail

### Serial Correlation (Breusch-Godfrey Fails)

**Problem**: Residuals are correlated with their past values.

**Solutions**:
1. **Increase lag order**: Try ARDL(2,2,2,2) or ARDL(3,3,3,3)
2. **Add more variables**: You may have omitted a relevant variable
3. **Check for structural breaks**: Major events (2016 float, 2023 naira redesign) can cause correlation

**Example**:
```python
# If ARDL(1,1,1,1) fails, try:
result = estimate_ardl(data, lags={"infl": 2, "mpr": 2, "exo": 2, "tbr": 2})
```

---

### Heteroscedasticity (Breusch-Pagan Fails)

**Problem**: Residual variance is not constant over time.

**Solutions**:
1. **Use HAC standard errors**: Also called Newey-West standard errors
2. **Check for outliers**: Nigeria experienced extreme volatility in 2016 and 2023
3. **Consider GARCH models**: For explicitly modeling changing variance

**Using HAC standard errors**:
```python
from statsmodels.regression.linear_model import OLS

# Re-estimate with HAC standard errors
model = OLS(endog, exog)
result_hac = model.fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print(result_hac.summary())
```

---

### Non-Normality (Jarque-Bera Fails)

**Problem**: Residuals are not normally distributed.

**Solutions**:
1. **Usually acceptable**: With 100+ observations, non-normality is less critical due to the Central Limit Theorem
2. **Check for outliers**: Look for extreme observations (e.g., inflation spikes)
3. **Structural breaks**: Major policy changes can create outliers

**Nigerian context**: The 2016 naira float and 2023 cash crisis created extreme outliers. Consider:
- Dummy variables for structural breaks
- Robust regression methods
- Accepting the violation if sample size is large

---

## Nigerian Context: Why Tests May Fail

Nigeria's macroeconomic environment has several features that challenge ARDL diagnostics:

1. **2016 Naira Float**: Sharp devaluation created a structural break
2. **2023 Cash Crisis**: Currency redesign caused severe disruptions
3. **Policy Uncertainty**: Frequent MPR changes create volatility
4. **Oil Price Shocks**: External shocks create heteroscedasticity

**Recommendation**: Even if some tests fail, document the issue and proceed if:
- You've tried increasing lags
- You use robust standard errors (HAC)
- Your sample size is reasonably large (100+)

---

## Week 3 Recap: The Complete ARDL Workflow

Congratulations! You've completed Week 3 — the ARDL chapter. Here's what you built:

| Day | Topic | File Created |
|-----|-------|--------------|
| **Day 11** | Lag Selection | `econometric_models/lag_selection.py` |
| **Day 12** | ARDL Estimation | `econometric_models/ardl_model.py` |
| **Day 13** | Bounds Test | `econometric_models/bounds_test.py` |
| **Day 14** | Error Correction Model | `econometric_models/ecm_extraction.py` |
| **Day 15** | Diagnostics | `econometric_models/ardl_diagnostics.py` |

**The complete ARDL process**:
1. Select lag order using AIC/BIC
2. Estimate ARDL model
3. Test for cointegration using bounds test
4. Extract ECM to analyze short-run dynamics
5. Validate model using diagnostic tests

You now have a scientifically rigorous cointegration framework for Nigerian inflation.

---

## Commit Your Work

Save your progress to Git:

```bash
cd /home/user/Nigerian_Inflation_Predictor
git add econometric_models/ardl_diagnostics.py
git add results/ardl_diagnostics.csv
git add results/ardl_residual_diagnostics.png
git commit -m "Day 15: Add ARDL diagnostic tests (Breusch-Godfrey, Breusch-Pagan, Jarque-Bera) with residual plots"
git log --oneline -5
```

Your commit history should show all 5 ARDL files from Week 3.

---

## Common Errors and Solutions

### Error 1: ModuleNotFoundError for scipy.stats
```
ModuleNotFoundError: No module named 'scipy.stats'
```

**Solution**: Scipy should already be installed. If not:
```bash
pip install scipy
```

---

### Error 2: ValueError: probplot requires numeric input
```
ValueError: probplot requires numeric input
```

**Solution**: Ensure residuals don't contain NaN values:
```python
residuals = result.resid.dropna()
```

---

### Error 3: All tests fail with p < 0.001
```
Breusch-Godfrey: FAIL
Breusch-Pagan: FAIL
Jarque-Bera: FAIL
```

**Solution**: This suggests serious model misspecification. Try:
1. Increase lag order to ARDL(2,2,2,2) or higher
2. Add dummy variables for 2016 and 2023
3. Check if data has extreme outliers

---

## Q&A: Understanding Diagnostics

**Q1: My model fails the Breusch-Godfrey test but passes the others. Can I still use it?**

A: Serial correlation is serious because it invalidates standard errors. Try increasing lags first. If it still fails, use HAC standard errors:

```python
result_hac = model.fit(cov_type="HAC", cov_kwds={"maxlags": 4})
```

This adjusts standard errors for serial correlation.

---

**Q2: Why is the Jarque-Bera test less critical than the others?**

A: The Central Limit Theorem says that with large samples (100+), coefficient estimates are approximately normal even if residuals aren't. Serial correlation and heteroscedasticity, however, directly bias standard errors, making p-values unreliable regardless of sample size.

---

**Q3: My residual plot shows a few large spikes in 2016 and 2023. What should I do?**

A: These are likely the naira float (2016) and cash crisis (2023). Three options:

1. **Add dummy variables**:
```python
data["d2016"] = (data.index.year == 2016) & (data.index.month >= 6)
data["d2023"] = (data.index.year == 2023) & (data.index.month <= 3)
```

2. **Accept the outliers**: Document them in your analysis

3. **Use robust regression**: Methods like Huber regression are less sensitive to outliers

For a teaching project, option 2 (accept and document) is often sufficient.

---

## What You Built Today

You created `econometric_models/ardl_diagnostics.py` with:

1. **Breusch-Godfrey test**: Detects serial correlation in residuals
2. **Breusch-Pagan test**: Detects heteroscedasticity (non-constant variance)
3. **Jarque-Bera test**: Tests normality of residuals
4. **4-panel diagnostic plots**: Visual inspection of residual behavior
5. **Automated testing workflow**: Run all tests and save results

**Key skills learned**:
- Understanding why diagnostics are mandatory
- Running and interpreting statistical tests
- Creating publication-quality diagnostic plots
- Knowing what to do when tests fail

**Outputs**:
- `results/ardl_diagnostics.csv` — Test results table
- `results/ardl_residual_diagnostics.png` — 4-panel diagnostic plots

---

## Next Week: Vector Autoregression (VAR)

Week 4 introduces **VAR models** — a system of equations where each variable depends on its own lags and lags of all other variables.

**Preview**:
- Day 16: VAR model estimation
- Day 17: Granger causality tests (Does MPR Granger-cause inflation?)
- Day 18: Impulse Response Functions (What happens if MPR increases by 1%?)
- Day 19: Forecast Error Variance Decomposition
- Day 20: Cholesky identification and structural shocks

VAR models answer questions like: "If the CBN raises MPR by 100 basis points, what happens to inflation over the next 12 months?"

---

**Congratulations on completing Day 15 and Week 3!** You've mastered the ARDL cointegration framework from start to finish. Your model is now validated and ready for forecasting.

Tomorrow, you'll start building Vector Autoregression models — a powerful tool for analyzing dynamic relationships between multiple time series.
