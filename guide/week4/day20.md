# Day 20 — VAR Diagnostics & Model Validation

## What You'll Learn Today

After building the VAR model and analyzing impulse responses and forecast error variance decomposition, we need to validate our model. Today you'll learn:

- **Stability condition**: All eigenvalues of the companion matrix must be inside the unit circle
- **Portmanteau test**: Tests for serial correlation in VAR residuals
- **Normality test**: Tests whether residuals follow a multivariate normal distribution
- **Residual plots**: Visual inspection of model residuals

These diagnostic tests ensure your VAR model is properly specified and reliable for policy analysis.

---

## Building `econometric_models/var_diagnostics.py` in 3 Steps

### STEP 1: Imports, Load Data, and Stability Check

**Delete everything in `econometric_models/var_diagnostics.py` and replace it with this:**

```python
"""
VAR Diagnostic Tests for Nigerian Inflation Predictor.

This module runs diagnostic tests on the estimated VAR model:
- Stability condition (eigenvalues inside unit circle)
- Serial correlation test (Portmanteau)
- Normality test (Jarque-Bera)
- Residual plots
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Define directories
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Variable ordering: MPR → TBR → EXO → INFL
VARIABLE_ORDER = ["mpr", "tbr", "exo", "infl"]


def load_stationary_data():
    """Load the stationary data for VAR estimation."""
    file_path = os.path.join(PROCESSED_DIR, "stationary_data.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find {file_path}. Run data preparation first.")

    df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")

    # Select variables in correct order
    if not all(var in df.columns for var in VARIABLE_ORDER):
        raise ValueError(f"Missing variables. Expected: {VARIABLE_ORDER}")

    df_var = df[VARIABLE_ORDER].dropna()
    print(f"Loaded {len(df_var)} observations for VAR diagnostics")
    return df_var


def check_stability(var_result):
    """
    Check VAR stability condition.

    All eigenvalues of the companion matrix must be inside the unit circle.
    If stable, the VAR is stationary and IRFs will converge.

    Parameters
    ----------
    var_result : VARResultsWrapper
        Fitted VAR model

    Returns
    -------
    dict
        Stability test results
    """
    print("\n" + "="*60)
    print("STABILITY CONDITION TEST")
    print("="*60)

    # Check stability using built-in method
    is_stable = var_result.is_stable()

    # Get eigenvalues (roots of characteristic polynomial)
    # For stability, modulus of all eigenvalues must be < 1
    roots = var_result.roots
    max_root = np.max(np.abs(roots))

    stability_results = {
        "is_stable": is_stable,
        "max_eigenvalue_modulus": max_root,
        "all_roots": roots
    }

    print(f"VAR Model Stable: {is_stable}")
    print(f"Maximum eigenvalue modulus: {max_root:.4f}")
    print(f"Number of eigenvalues: {len(roots)}")

    if is_stable:
        print("✓ Model is STABLE (all eigenvalues inside unit circle)")
    else:
        print("✗ Model is UNSTABLE (some eigenvalues outside unit circle)")
        print("  → Consider re-specifying the model")

    return stability_results


def plot_eigenvalues(var_result):
    """
    Plot eigenvalues on complex plane with unit circle.

    Parameters
    ----------
    var_result : VARResultsWrapper
        Fitted VAR model
    """
    roots = var_result.roots

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot unit circle
    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', linewidth=2)
    ax.add_patch(circle)

    # Plot eigenvalues
    ax.scatter(roots.real, roots.imag, s=100, c='red', marker='o',
               edgecolors='black', linewidths=1.5, alpha=0.7, label='Eigenvalues')

    # Add reference lines
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # Labels and title
    ax.set_xlabel('Real Part', fontsize=12)
    ax.set_ylabel('Imaginary Part', fontsize=12)
    ax.set_title('VAR Stability: Eigenvalues of Companion Matrix', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Set limits
    max_val = max(1.2, np.max(np.abs(roots)) + 0.2)
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # Add text annotation
    is_stable = var_result.is_stable()
    status_text = "STABLE" if is_stable else "UNSTABLE"
    status_color = "green" if is_stable else "red"
    ax.text(0.05, 0.95, f"Model Status: {status_text}",
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor=status_color, alpha=0.3))

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "var_eigenvalues.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Eigenvalue plot saved to: {save_path}")


def main():
    """Run all VAR diagnostic tests."""
    print("="*60)
    print("VAR DIAGNOSTIC TESTS - Nigerian Inflation Predictor")
    print("="*60)

    # Load data
    df_var = load_stationary_data()

    # Estimate VAR model with optimal lag
    # In practice, you would load the optimal lag from previous analysis
    # For now, we use lag=2 (this should match your VAR estimation)
    optimal_lag = 2
    print(f"\nEstimating VAR({optimal_lag}) model...")

    model = VAR(df_var)
    var_result = model.fit(maxlags=optimal_lag)

    print(f"VAR({var_result.k_ar}) model estimated with {len(df_var)} observations")

    # Test 1: Stability
    stability_results = check_stability(var_result)
    plot_eigenvalues(var_result)

    print("\n" + "="*60)
    print("DIAGNOSTIC TESTS COMPLETE")
    print("="*60)
    print("Next steps:")
    print("- Test for serial correlation (Step 2)")
    print("- Test for normality (Step 2)")
    print("- Plot residuals (Step 3)")


if __name__ == "__main__":
    main()
```

**What this code does:**

1. **Loads stationary data**: Uses the same data you prepared for VAR estimation
2. **Estimates VAR model**: Fits a VAR(2) model to the data
3. **Stability check**: Uses `var_result.is_stable()` to check if all eigenvalues are inside the unit circle
4. **Eigenvalue plot**: Visualizes eigenvalues on the complex plane with unit circle overlay

**Run it:**

```bash
python econometric_models/var_diagnostics.py
```

You should see stability test results and an eigenvalue plot saved to `results/figures/var_eigenvalues.png`.

---

### STEP 2: Add Serial Correlation and Normality Tests

**Delete everything in `econometric_models/var_diagnostics.py` and replace it with this:**

```python
"""
VAR Diagnostic Tests for Nigerian Inflation Predictor.

This module runs diagnostic tests on the estimated VAR model:
- Stability condition (eigenvalues inside unit circle)
- Serial correlation test (Portmanteau)
- Normality test (Jarque-Bera)
- Residual plots
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Define directories
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Variable ordering: MPR → TBR → EXO → INFL
VARIABLE_ORDER = ["mpr", "tbr", "exo", "infl"]


def load_stationary_data():
    """Load the stationary data for VAR estimation."""
    file_path = os.path.join(PROCESSED_DIR, "stationary_data.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find {file_path}. Run data preparation first.")

    df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")

    # Select variables in correct order
    if not all(var in df.columns for var in VARIABLE_ORDER):
        raise ValueError(f"Missing variables. Expected: {VARIABLE_ORDER}")

    df_var = df[VARIABLE_ORDER].dropna()
    print(f"Loaded {len(df_var)} observations for VAR diagnostics")
    return df_var


def check_stability(var_result):
    """
    Check VAR stability condition.

    All eigenvalues of the companion matrix must be inside the unit circle.
    If stable, the VAR is stationary and IRFs will converge.

    Parameters
    ----------
    var_result : VARResultsWrapper
        Fitted VAR model

    Returns
    -------
    dict
        Stability test results
    """
    print("\n" + "="*60)
    print("STABILITY CONDITION TEST")
    print("="*60)

    # Check stability using built-in method
    is_stable = var_result.is_stable()

    # Get eigenvalues (roots of characteristic polynomial)
    # For stability, modulus of all eigenvalues must be < 1
    roots = var_result.roots
    max_root = np.max(np.abs(roots))

    stability_results = {
        "is_stable": is_stable,
        "max_eigenvalue_modulus": max_root,
        "all_roots": roots
    }

    print(f"VAR Model Stable: {is_stable}")
    print(f"Maximum eigenvalue modulus: {max_root:.4f}")
    print(f"Number of eigenvalues: {len(roots)}")

    if is_stable:
        print("✓ Model is STABLE (all eigenvalues inside unit circle)")
    else:
        print("✗ Model is UNSTABLE (some eigenvalues outside unit circle)")
        print("  → Consider re-specifying the model")

    return stability_results


def test_serial_correlation(var_result, nlags=10):
    """
    Portmanteau test for serial correlation in VAR residuals.

    H0: No serial correlation up to lag h
    H1: Serial correlation exists

    If p-value > 0.05, we fail to reject H0 (no serial correlation = good)

    Parameters
    ----------
    var_result : VARResultsWrapper
        Fitted VAR model
    nlags : int
        Number of lags to test

    Returns
    -------
    dict
        Test results
    """
    print("\n" + "="*60)
    print("SERIAL CORRELATION TEST (Portmanteau)")
    print("="*60)

    # Perform Portmanteau test
    test_result = var_result.test_whiteness(nlags=nlags, signif=0.05, adjusted=True)

    print(f"Testing for serial correlation up to lag {nlags}")
    print(f"Test statistic: {test_result.test_statistic:.4f}")
    print(f"P-value: {test_result.pvalue:.4f}")
    print(f"Critical value (5%): {test_result.crit_value:.4f}")

    if test_result.pvalue > 0.05:
        print("✓ No evidence of serial correlation (p > 0.05)")
        conclusion = "Pass"
    else:
        print("✗ Evidence of serial correlation (p < 0.05)")
        print("  → Consider adding more lags to the VAR model")
        conclusion = "Fail"

    return {
        "test_statistic": test_result.test_statistic,
        "pvalue": test_result.pvalue,
        "critical_value": test_result.crit_value,
        "conclusion": conclusion
    }


def test_normality(var_result):
    """
    Test for normality of VAR residuals using Jarque-Bera test.

    H0: Residuals are normally distributed
    H1: Residuals are not normally distributed

    If p-value > 0.05, we fail to reject H0 (normality = good)

    Parameters
    ----------
    var_result : VARResultsWrapper
        Fitted VAR model

    Returns
    -------
    dict
        Test results for each variable
    """
    print("\n" + "="*60)
    print("NORMALITY TEST (Jarque-Bera)")
    print("="*60)

    # Perform normality test
    test_result = var_result.test_normality()

    print(f"Testing normality of residuals for {len(VARIABLE_ORDER)} variables")
    print(f"Joint test statistic: {test_result.test_statistic:.4f}")
    print(f"Joint p-value: {test_result.pvalue:.4f}")

    if test_result.pvalue > 0.05:
        print("✓ Residuals appear normally distributed (p > 0.05)")
        conclusion = "Pass"
    else:
        print("✗ Residuals may not be normally distributed (p < 0.05)")
        print("  → VAR can still be used but inference may be affected")
        conclusion = "Fail"

    # Individual variable results
    print("\nIndividual variable tests:")
    results_by_var = {}

    for i, var_name in enumerate(VARIABLE_ORDER):
        print(f"  {var_name.upper()}: statistic={test_result.test_statistic_indiv[i]:.4f}, "
              f"p-value={test_result.pvalue_indiv[i]:.4f}")
        results_by_var[var_name] = {
            "test_statistic": test_result.test_statistic_indiv[i],
            "pvalue": test_result.pvalue_indiv[i]
        }

    return {
        "joint_test_statistic": test_result.test_statistic,
        "joint_pvalue": test_result.pvalue,
        "individual_tests": results_by_var,
        "conclusion": conclusion
    }


def plot_eigenvalues(var_result):
    """
    Plot eigenvalues on complex plane with unit circle.

    Parameters
    ----------
    var_result : VARResultsWrapper
        Fitted VAR model
    """
    roots = var_result.roots

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot unit circle
    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', linewidth=2)
    ax.add_patch(circle)

    # Plot eigenvalues
    ax.scatter(roots.real, roots.imag, s=100, c='red', marker='o',
               edgecolors='black', linewidths=1.5, alpha=0.7, label='Eigenvalues')

    # Add reference lines
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # Labels and title
    ax.set_xlabel('Real Part', fontsize=12)
    ax.set_ylabel('Imaginary Part', fontsize=12)
    ax.set_title('VAR Stability: Eigenvalues of Companion Matrix', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Set limits
    max_val = max(1.2, np.max(np.abs(roots)) + 0.2)
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # Add text annotation
    is_stable = var_result.is_stable()
    status_text = "STABLE" if is_stable else "UNSTABLE"
    status_color = "green" if is_stable else "red"
    ax.text(0.05, 0.95, f"Model Status: {status_text}",
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor=status_color, alpha=0.3))

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "var_eigenvalues.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Eigenvalue plot saved to: {save_path}")


def main():
    """Run all VAR diagnostic tests."""
    print("="*60)
    print("VAR DIAGNOSTIC TESTS - Nigerian Inflation Predictor")
    print("="*60)

    # Load data
    df_var = load_stationary_data()

    # Estimate VAR model with optimal lag
    # In practice, you would load the optimal lag from previous analysis
    # For now, we use lag=2 (this should match your VAR estimation)
    optimal_lag = 2
    print(f"\nEstimating VAR({optimal_lag}) model...")

    model = VAR(df_var)
    var_result = model.fit(maxlags=optimal_lag)

    print(f"VAR({var_result.k_ar}) model estimated with {len(df_var)} observations")

    # Test 1: Stability
    stability_results = check_stability(var_result)
    plot_eigenvalues(var_result)

    # Test 2: Serial correlation
    serial_corr_results = test_serial_correlation(var_result, nlags=10)

    # Test 3: Normality
    normality_results = test_normality(var_result)

    print("\n" + "="*60)
    print("DIAGNOSTIC TESTS COMPLETE")
    print("="*60)
    print("Summary:")
    print(f"  Stability: {'PASS' if stability_results['is_stable'] else 'FAIL'}")
    print(f"  Serial correlation: {serial_corr_results['conclusion']}")
    print(f"  Normality: {normality_results['conclusion']}")


if __name__ == "__main__":
    main()
```

**What's new:**

1. **`test_serial_correlation()`**: Uses the Portmanteau test (`test_whiteness()`) to check for autocorrelation in residuals
   - **Null hypothesis**: No serial correlation
   - **Good result**: p-value > 0.05 (fail to reject, no serial correlation)

2. **`test_normality()`**: Uses Jarque-Bera test (`test_normality()`) for multivariate normality
   - **Null hypothesis**: Residuals are normally distributed
   - **Good result**: p-value > 0.05 (fail to reject, normal distribution)

**Run it:**

```bash
python econometric_models/var_diagnostics.py
```

You'll now see three diagnostic test results printed to the console.

---

### STEP 3: Add Residual Plots and Save Results

**Delete everything in `econometric_models/var_diagnostics.py` and replace it with this:**

```python
"""
VAR Diagnostic Tests for Nigerian Inflation Predictor.

This module runs diagnostic tests on the estimated VAR model:
- Stability condition (eigenvalues inside unit circle)
- Serial correlation test (Portmanteau)
- Normality test (Jarque-Bera)
- Residual plots
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from scipy import stats

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Define directories
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Variable ordering: MPR → TBR → EXO → INFL
VARIABLE_ORDER = ["mpr", "tbr", "exo", "infl"]


def load_stationary_data():
    """Load the stationary data for VAR estimation."""
    file_path = os.path.join(PROCESSED_DIR, "stationary_data.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find {file_path}. Run data preparation first.")

    df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")

    # Select variables in correct order
    if not all(var in df.columns for var in VARIABLE_ORDER):
        raise ValueError(f"Missing variables. Expected: {VARIABLE_ORDER}")

    df_var = df[VARIABLE_ORDER].dropna()
    print(f"Loaded {len(df_var)} observations for VAR diagnostics")
    return df_var


def check_stability(var_result):
    """
    Check VAR stability condition.

    All eigenvalues of the companion matrix must be inside the unit circle.
    If stable, the VAR is stationary and IRFs will converge.

    Parameters
    ----------
    var_result : VARResultsWrapper
        Fitted VAR model

    Returns
    -------
    dict
        Stability test results
    """
    print("\n" + "="*60)
    print("STABILITY CONDITION TEST")
    print("="*60)

    # Check stability using built-in method
    is_stable = var_result.is_stable()

    # Get eigenvalues (roots of characteristic polynomial)
    # For stability, modulus of all eigenvalues must be < 1
    roots = var_result.roots
    max_root = np.max(np.abs(roots))

    stability_results = {
        "is_stable": is_stable,
        "max_eigenvalue_modulus": max_root,
        "all_roots": roots
    }

    print(f"VAR Model Stable: {is_stable}")
    print(f"Maximum eigenvalue modulus: {max_root:.4f}")
    print(f"Number of eigenvalues: {len(roots)}")

    if is_stable:
        print("✓ Model is STABLE (all eigenvalues inside unit circle)")
    else:
        print("✗ Model is UNSTABLE (some eigenvalues outside unit circle)")
        print("  → Consider re-specifying the model")

    return stability_results


def test_serial_correlation(var_result, nlags=10):
    """
    Portmanteau test for serial correlation in VAR residuals.

    H0: No serial correlation up to lag h
    H1: Serial correlation exists

    If p-value > 0.05, we fail to reject H0 (no serial correlation = good)

    Parameters
    ----------
    var_result : VARResultsWrapper
        Fitted VAR model
    nlags : int
        Number of lags to test

    Returns
    -------
    dict
        Test results
    """
    print("\n" + "="*60)
    print("SERIAL CORRELATION TEST (Portmanteau)")
    print("="*60)

    # Perform Portmanteau test
    test_result = var_result.test_whiteness(nlags=nlags, signif=0.05, adjusted=True)

    print(f"Testing for serial correlation up to lag {nlags}")
    print(f"Test statistic: {test_result.test_statistic:.4f}")
    print(f"P-value: {test_result.pvalue:.4f}")
    print(f"Critical value (5%): {test_result.crit_value:.4f}")

    if test_result.pvalue > 0.05:
        print("✓ No evidence of serial correlation (p > 0.05)")
        conclusion = "Pass"
    else:
        print("✗ Evidence of serial correlation (p < 0.05)")
        print("  → Consider adding more lags to the VAR model")
        conclusion = "Fail"

    return {
        "test_statistic": test_result.test_statistic,
        "pvalue": test_result.pvalue,
        "critical_value": test_result.crit_value,
        "conclusion": conclusion
    }


def test_normality(var_result):
    """
    Test for normality of VAR residuals using Jarque-Bera test.

    H0: Residuals are normally distributed
    H1: Residuals are not normally distributed

    If p-value > 0.05, we fail to reject H0 (normality = good)

    Parameters
    ----------
    var_result : VARResultsWrapper
        Fitted VAR model

    Returns
    -------
    dict
        Test results for each variable
    """
    print("\n" + "="*60)
    print("NORMALITY TEST (Jarque-Bera)")
    print("="*60)

    # Perform normality test
    test_result = var_result.test_normality()

    print(f"Testing normality of residuals for {len(VARIABLE_ORDER)} variables")
    print(f"Joint test statistic: {test_result.test_statistic:.4f}")
    print(f"Joint p-value: {test_result.pvalue:.4f}")

    if test_result.pvalue > 0.05:
        print("✓ Residuals appear normally distributed (p > 0.05)")
        conclusion = "Pass"
    else:
        print("✗ Residuals may not be normally distributed (p < 0.05)")
        print("  → VAR can still be used but inference may be affected")
        conclusion = "Fail"

    # Individual variable results
    print("\nIndividual variable tests:")
    results_by_var = {}

    for i, var_name in enumerate(VARIABLE_ORDER):
        print(f"  {var_name.upper()}: statistic={test_result.test_statistic_indiv[i]:.4f}, "
              f"p-value={test_result.pvalue_indiv[i]:.4f}")
        results_by_var[var_name] = {
            "test_statistic": test_result.test_statistic_indiv[i],
            "pvalue": test_result.pvalue_indiv[i]
        }

    return {
        "joint_test_statistic": test_result.test_statistic,
        "joint_pvalue": test_result.pvalue,
        "individual_tests": results_by_var,
        "conclusion": conclusion
    }


def plot_eigenvalues(var_result):
    """
    Plot eigenvalues on complex plane with unit circle.

    Parameters
    ----------
    var_result : VARResultsWrapper
        Fitted VAR model
    """
    roots = var_result.roots

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot unit circle
    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', linewidth=2)
    ax.add_patch(circle)

    # Plot eigenvalues
    ax.scatter(roots.real, roots.imag, s=100, c='red', marker='o',
               edgecolors='black', linewidths=1.5, alpha=0.7, label='Eigenvalues')

    # Add reference lines
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # Labels and title
    ax.set_xlabel('Real Part', fontsize=12)
    ax.set_ylabel('Imaginary Part', fontsize=12)
    ax.set_title('VAR Stability: Eigenvalues of Companion Matrix', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Set limits
    max_val = max(1.2, np.max(np.abs(roots)) + 0.2)
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # Add text annotation
    is_stable = var_result.is_stable()
    status_text = "STABLE" if is_stable else "UNSTABLE"
    status_color = "green" if is_stable else "red"
    ax.text(0.05, 0.95, f"Model Status: {status_text}",
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor=status_color, alpha=0.3))

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "var_eigenvalues.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Eigenvalue plot saved to: {save_path}")


def plot_residuals(var_result):
    """
    Plot VAR residuals for all variables.

    Creates a 4x2 grid:
    - Left column: Time series of residuals
    - Right column: Q-Q plots for normality check

    Parameters
    ----------
    var_result : VARResultsWrapper
        Fitted VAR model
    """
    residuals = var_result.resid

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))

    for i, var_name in enumerate(VARIABLE_ORDER):
        resid = residuals[var_name].values

        # Left column: Time series plot
        ax_ts = axes[i, 0]
        ax_ts.plot(residuals.index, resid, linewidth=1, color='steelblue', alpha=0.7)
        ax_ts.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_ts.set_ylabel('Residual', fontsize=10)
        ax_ts.set_title(f'{var_name.upper()} Residuals Over Time', fontsize=11, fontweight='bold')
        ax_ts.grid(True, alpha=0.3)

        # Add mean line
        mean_resid = np.mean(resid)
        ax_ts.axhline(y=mean_resid, color='green', linestyle=':', linewidth=1,
                     label=f'Mean={mean_resid:.4f}')
        ax_ts.legend(fontsize=8, loc='upper right')

        # Right column: Q-Q plot
        ax_qq = axes[i, 1]
        stats.probplot(resid, dist="norm", plot=ax_qq)
        ax_qq.set_title(f'{var_name.upper()} Q-Q Plot', fontsize=11, fontweight='bold')
        ax_qq.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "var_residuals.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Residual plots saved to: {save_path}")


def save_diagnostic_summary(stability_results, serial_corr_results, normality_results):
    """
    Save all diagnostic test results to CSV.

    Parameters
    ----------
    stability_results : dict
        Stability test results
    serial_corr_results : dict
        Serial correlation test results
    normality_results : dict
        Normality test results
    """
    # Create summary dataframe
    summary_data = {
        "Test": [
            "Stability (Eigenvalues)",
            "Serial Correlation (Portmanteau)",
            "Normality (Jarque-Bera)"
        ],
        "Result": [
            "PASS" if stability_results["is_stable"] else "FAIL",
            serial_corr_results["conclusion"],
            normality_results["conclusion"]
        ],
        "Test Statistic": [
            f"{stability_results['max_eigenvalue_modulus']:.4f}",
            f"{serial_corr_results['test_statistic']:.4f}",
            f"{normality_results['joint_test_statistic']:.4f}"
        ],
        "P-value": [
            "N/A",
            f"{serial_corr_results['pvalue']:.4f}",
            f"{normality_results['joint_pvalue']:.4f}"
        ]
    }

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    save_path = os.path.join(RESULTS_DIR, "var_diagnostic_summary.csv")
    summary_df.to_csv(save_path, index=False)
    print(f"\nDiagnostic summary saved to: {save_path}")

    # Also save individual normality test results
    normality_detail = []
    for var_name, results in normality_results["individual_tests"].items():
        normality_detail.append({
            "Variable": var_name.upper(),
            "Test Statistic": f"{results['test_statistic']:.4f}",
            "P-value": f"{results['pvalue']:.4f}",
            "Normal": "Yes" if results['pvalue'] > 0.05 else "No"
        })

    normality_df = pd.DataFrame(normality_detail)
    save_path = os.path.join(RESULTS_DIR, "var_normality_by_variable.csv")
    normality_df.to_csv(save_path, index=False)
    print(f"Individual normality tests saved to: {save_path}")


def main():
    """Run all VAR diagnostic tests."""
    print("="*60)
    print("VAR DIAGNOSTIC TESTS - Nigerian Inflation Predictor")
    print("="*60)

    # Load data
    df_var = load_stationary_data()

    # Estimate VAR model with optimal lag
    # In practice, you would load the optimal lag from previous analysis
    # For now, we use lag=2 (this should match your VAR estimation)
    optimal_lag = 2
    print(f"\nEstimating VAR({optimal_lag}) model...")

    model = VAR(df_var)
    var_result = model.fit(maxlags=optimal_lag)

    print(f"VAR({var_result.k_ar}) model estimated with {len(df_var)} observations")

    # Test 1: Stability
    stability_results = check_stability(var_result)
    plot_eigenvalues(var_result)

    # Test 2: Serial correlation
    serial_corr_results = test_serial_correlation(var_result, nlags=10)

    # Test 3: Normality
    normality_results = test_normality(var_result)

    # Plot residuals
    plot_residuals(var_result)

    # Save summary
    save_diagnostic_summary(stability_results, serial_corr_results, normality_results)

    print("\n" + "="*60)
    print("DIAGNOSTIC TESTS COMPLETE")
    print("="*60)
    print("Summary:")
    print(f"  Stability: {'PASS' if stability_results['is_stable'] else 'FAIL'}")
    print(f"  Serial correlation: {serial_corr_results['conclusion']}")
    print(f"  Normality: {normality_results['conclusion']}")
    print("\nFiles created:")
    print("  - results/figures/var_eigenvalues.png")
    print("  - results/figures/var_residuals.png")
    print("  - results/var_diagnostic_summary.csv")
    print("  - results/var_normality_by_variable.csv")


if __name__ == "__main__":
    main()
```

**What's new in Step 3:**

1. **`plot_residuals()`**: Creates a 4x2 grid of plots
   - **Left column**: Time series of residuals for each variable
   - **Right column**: Q-Q plots to visually assess normality

2. **`save_diagnostic_summary()`**: Saves all test results to CSV files
   - Overall summary in `var_diagnostic_summary.csv`
   - Individual normality tests in `var_normality_by_variable.csv`

**Run the complete diagnostic script:**

```bash
python econometric_models/var_diagnostics.py
```

**Check your results:**

```bash
ls results/figures/var_*.png
cat results/var_diagnostic_summary.csv
```

---

## Week 4 Recap: Vector Autoregression Complete

Congratulations! You've completed Week 4 and built a full VAR analysis pipeline:

### What You Built This Week

1. **Day 16**: VAR estimation and lag order selection
   - Fitted VAR model to stationary data
   - Selected optimal lag using AIC/BIC

2. **Day 17**: Granger causality testing
   - Tested if MPR "Granger-causes" inflation
   - Tested all pairwise relationships between variables

3. **Day 18**: Impulse Response Functions (IRFs)
   - Visualized dynamic effects of policy shocks
   - Saw how MPR shock affects inflation over time

4. **Day 19**: Forecast Error Variance Decomposition (FEVD)
   - Decomposed forecast error variance
   - Quantified relative importance of each variable

5. **Day 20** (today): Model diagnostics
   - Validated stability condition
   - Tested for serial correlation
   - Tested for normality
   - Plotted residuals

### Week 3 vs Week 4: Two Complementary Models

| Feature | ARDL (Week 3) | VAR (Week 4) |
|---------|---------------|--------------|
| **Focus** | Single equation for inflation | System of equations |
| **Variables** | Dependent: INFL<br>Independent: MPR, TBR, EXO | All variables treated symmetrically |
| **Stationarity** | Mix of I(0) and I(1) allowed | All variables must be stationary |
| **Long-run** | Cointegrating relationship | Not directly estimated |
| **Short-run** | Error correction term | Lag dynamics |
| **Use case** | Long-run policy impact | Dynamic interactions |

**Both models are valuable**:
- ARDL tells you about long-run equilibrium relationships
- VAR tells you about dynamic interactions and shock propagation

---

## Commit Your Work

```bash
git add econometric_models/var_diagnostics.py
git commit -m "Day 20: Add VAR diagnostics (stability, serial correlation, normality tests)"
```

---

## Common Errors and Fixes

### Error 1: "VAR model is unstable"

**Problem**: Some eigenvalues are outside the unit circle.

**Solution**:
- Try different lag orders
- Check if your data is truly stationary
- Consider adding dummy variables for outliers
- Re-examine variable transformations

### Error 2: "Evidence of serial correlation"

**Problem**: Portmanteau test rejects null hypothesis (p < 0.05).

**Solution**:
- Increase the number of lags in the VAR
- Check if you missed structural breaks
- Consider adding deterministic terms (trend, constant)

### Error 3: "Residuals not normally distributed"

**Problem**: Jarque-Bera test rejects normality (p < 0.05).

**Solution**:
- Don't panic! VAR is robust to moderate non-normality
- Check for outliers in the data
- Consider robust standard errors for inference
- Large sample sizes mitigate non-normality issues

---

## Q&A

**Q1: What if my VAR model fails all diagnostic tests?**

A: Start with stability. If the model is unstable, you need to fix that first (check stationarity, try different lags). If only normality fails but stability and serial correlation pass, you can still use the VAR with caution.

**Q2: Why do we care about eigenvalues?**

A: Eigenvalues determine if the VAR process is stationary. If any eigenvalue is outside the unit circle, the process is explosive (not stationary), and your IRFs will diverge to infinity instead of dying out.

**Q3: Can I use a VAR with non-normal residuals?**

A: Yes! VAR inference is asymptotically valid even with non-normal residuals, especially in large samples. However, small-sample inference (like confidence intervals) may be affected.

**Q4: What's the difference between AIC lag selection and diagnostic tests?**

A: AIC selects the lag that best balances fit and parsimony. But even the "optimal" lag may leave serial correlation in residuals. Diagnostic tests check if your chosen model is well-specified.

**Q5: Should I report all these tests in my analysis?**

A: Yes! Any credible VAR analysis should report:
- Stability condition
- Serial correlation test
- Normality test
- Lag order selection criteria

---

## Next Week Preview: Policy Simulation and Integration

Week 5 will bring everything together:

- **Day 21**: Policy shock scenarios (MPR increase/decrease simulations)
- **Day 22**: Combine ARDL and VAR forecasts
- **Day 23**: Dashboard for visualization
- **Day 24**: Documentation and reporting
- **Day 25**: Final integration and deployment

You now have TWO robust econometric models (ARDL + VAR) ready for policy analysis!

---

**Tomorrow**: Policy shock simulation - testing different monetary policy scenarios using your validated VAR model.
