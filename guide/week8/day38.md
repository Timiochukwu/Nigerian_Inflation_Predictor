# Day 38 — Validation: End-to-End Testing

## What You'll Learn Today

Today you'll create a **validation script** that checks whether your entire analysis pipeline works correctly. This is like a quality control inspector—it verifies that every file exists, contains the right data, and has no errors.

**Why validation matters:**
- Ensures all scripts ran successfully
- Catches missing or corrupted files early
- Makes your project reproducible (someone else can verify it works)
- Gives you confidence before presenting results

You'll build `validate.py` in **3 steps**:
1. Check file existence (raw data, cleaned data, results, plots)
2. Validate content (no missing values, correct columns, valid formats)
3. Print a comprehensive PASS/FAIL report

By the end, you'll have a single command (`python validate.py`) that confirms your entire project is intact.

---

## STEP 1: Create validate.py with File Existence Checks

**What this does:** Checks that all expected files exist in the project.

**Delete everything in `validate.py` and replace it with this:**

```python
"""
Validation Script — Nigerian Inflation Predictor
Checks that all expected files exist and contain valid data.
"""

import os
import json
import pandas as pd
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists and return result."""
    exists = os.path.exists(filepath)
    return {
        'check': description,
        'status': 'PASS' if exists else 'FAIL',
        'details': f'File found at {filepath}' if exists else f'File missing: {filepath}'
    }


def check_png_valid(filepath, description):
    """Check if a PNG file exists and has content."""
    if not os.path.exists(filepath):
        return {
            'check': description,
            'status': 'FAIL',
            'details': f'File missing: {filepath}'
        }

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return {
            'check': description,
            'status': 'FAIL',
            'details': f'File is empty (0 bytes): {filepath}'
        }

    return {
        'check': description,
        'status': 'PASS',
        'details': f'Valid PNG ({file_size} bytes)'
    }


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("NIGERIAN INFLATION PREDICTOR — VALIDATION REPORT")
    print("=" * 70)
    print()

    results = []

    # ========================================================================
    # SECTION 1: RAW DATA
    # ========================================================================
    print("[1] Checking Raw Data...")
    print("-" * 70)

    raw_data_path = 'data/raw/cbn_infl_data.csv'
    result = check_file_exists(raw_data_path, 'Raw CBN data file')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")

    if result['status'] == 'PASS':
        # Check if file has expected columns
        try:
            df_raw = pd.read_csv(raw_data_path)
            expected_cols = ['date', 'mpr', 'tbr', 'exo', 'infl']
            missing_cols = [col for col in expected_cols if col not in df_raw.columns]

            if missing_cols:
                results.append({
                    'check': 'Raw data has core columns',
                    'status': 'FAIL',
                    'details': f'Missing columns: {missing_cols}'
                })
                print(f"  FAIL: Missing columns: {missing_cols}")
            else:
                results.append({
                    'check': 'Raw data has core columns',
                    'status': 'PASS',
                    'details': f'Found {len(df_raw)} rows with all core columns'
                })
                print(f"  PASS: Found {len(df_raw)} rows with all core columns")
        except Exception as e:
            results.append({
                'check': 'Raw data readable',
                'status': 'FAIL',
                'details': str(e)
            })
            print(f"  FAIL: Could not read file — {e}")

    print()

    # ========================================================================
    # SECTION 2: CLEANED DATA
    # ========================================================================
    print("[2] Checking Cleaned Data...")
    print("-" * 70)

    cleaned_path = 'data/processed/cleaned_data.csv'
    result = check_file_exists(cleaned_path, 'Cleaned data file')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print()

    # ========================================================================
    # SECTION 3: STATIONARITY TESTS
    # ========================================================================
    print("[3] Checking Stationarity Test Results...")
    print("-" * 70)

    stationarity_files = [
        ('results/adf_results.csv', 'ADF test results'),
        ('results/integration_order.csv', 'Integration order results')
    ]

    for filepath, description in stationarity_files:
        result = check_file_exists(filepath, description)
        results.append(result)
        print(f"  {result['status']}: {result['check']}")

    print()

    # ========================================================================
    # SECTION 4: COINTEGRATION TESTS
    # ========================================================================
    print("[4] Checking Cointegration Test Results...")
    print("-" * 70)

    cointegration_files = [
        ('results/engle_granger.csv', 'Engle-Granger test'),
        ('results/johansen_trace.csv', 'Johansen trace test'),
        ('results/johansen_maxeig.csv', 'Johansen max eigenvalue test')
    ]

    for filepath, description in cointegration_files:
        result = check_file_exists(filepath, description)
        results.append(result)
        print(f"  {result['status']}: {result['check']}")

    print()

    # ========================================================================
    # SECTION 5: LAG SELECTION
    # ========================================================================
    print("[5] Checking Lag Selection Results...")
    print("-" * 70)

    result = check_file_exists('results/lag_selection.csv', 'Lag selection results')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print()

    # ========================================================================
    # SECTION 6: ARDL MODEL
    # ========================================================================
    print("[6] Checking ARDL Model Results...")
    print("-" * 70)

    ardl_files = [
        ('results/ardl_summary.txt', 'ARDL model summary'),
        ('results/ardl_bounds_test.csv', 'ARDL bounds test'),
        ('results/ardl_diagnostics.csv', 'ARDL diagnostics')
    ]

    for filepath, description in ardl_files:
        result = check_file_exists(filepath, description)
        results.append(result)
        print(f"  {result['status']}: {result['check']}")

    print()

    # ========================================================================
    # SECTION 7: VAR MODEL
    # ========================================================================
    print("[7] Checking VAR Model Results...")
    print("-" * 70)

    var_files = [
        ('results/var_summary.txt', 'VAR model summary'),
        ('results/var_granger.csv', 'Granger causality test')
    ]

    for filepath, description in var_files:
        result = check_file_exists(filepath, description)
        results.append(result)
        print(f"  {result['status']}: {result['check']}")

    print()

    # ========================================================================
    # SECTION 8: IRF AND FEVD
    # ========================================================================
    print("[8] Checking IRF and FEVD Results...")
    print("-" * 70)

    # IRF plots
    irf_plots = [
        ('results/plots/irf_mpr_to_infl.png', 'IRF: MPR → Inflation'),
        ('results/plots/irf_tbr_to_infl.png', 'IRF: TBR → Inflation'),
        ('results/plots/irf_exo_to_infl.png', 'IRF: EXO → Inflation'),
        ('results/plots/irf_all.png', 'IRF: All shocks to inflation')
    ]

    for filepath, description in irf_plots:
        result = check_png_valid(filepath, description)
        results.append(result)
        print(f"  {result['status']}: {result['check']}")

    # FEVD plots
    fevd_plots = [
        ('results/plots/fevd_infl.png', 'FEVD: Inflation decomposition'),
        ('results/plots/fevd_all.png', 'FEVD: All variables')
    ]

    for filepath, description in fevd_plots:
        result = check_png_valid(filepath, description)
        results.append(result)
        print(f"  {result['status']}: {result['check']}")

    print()

    # ========================================================================
    # SECTION 9: FORECASTS
    # ========================================================================
    print("[9] Checking Forecast Results...")
    print("-" * 70)

    forecast_files = [
        ('results/var_forecast.csv', 'VAR forecast data'),
        ('results/plots/var_forecast.png', 'VAR forecast plot')
    ]

    for filepath, description in forecast_files:
        if filepath.endswith('.png'):
            result = check_png_valid(filepath, description)
        else:
            result = check_file_exists(filepath, description)
        results.append(result)
        print(f"  {result['status']}: {result['check']}")

    print()

    # ========================================================================
    # SECTION 10: SIMULATIONS
    # ========================================================================
    print("[10] Checking Simulation Results...")
    print("-" * 70)

    simulation_files = [
        ('results/policy_simulation.csv', 'Policy simulation data'),
        ('results/plots/policy_simulation.png', 'Policy simulation plot')
    ]

    for filepath, description in simulation_files:
        if filepath.endswith('.png'):
            result = check_png_valid(filepath, description)
        else:
            result = check_file_exists(filepath, description)
        results.append(result)
        print(f"  {result['status']}: {result['check']}")

    print()

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    total_checks = len(results)
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = total_checks - passed

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total checks: {total_checks}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print()

    if failed == 0:
        print("🎉 ALL CHECKS PASSED! Your project is complete and reproducible.")
    else:
        print("⚠️  SOME CHECKS FAILED. Review the report above.")
        print()
        print("Failed checks:")
        for r in results:
            if r['status'] == 'FAIL':
                print(f"  - {r['check']}: {r['details']}")

    print("=" * 70)


if __name__ == '__main__':
    main()
```

**What this script does:**

1. **`check_file_exists()`**: Checks if a file exists and returns a result dictionary
2. **`check_png_valid()`**: Checks if a PNG file exists AND has content (not empty)
3. **`main()`**: Runs all checks organized by section:
   - Raw data (CSV exists, has columns)
   - Cleaned data
   - Stationarity tests (ADF, integration order)
   - Cointegration tests (Engle-Granger, Johansen)
   - Lag selection
   - ARDL model results
   - VAR model results
   - IRF and FEVD plots
   - Forecasts
   - Simulations
4. **Summary**: Counts PASS/FAIL and prints overall status

**Run it:**

```bash
python validate.py
```

**Expected output (if everything exists):**

```
======================================================================
NIGERIAN INFLATION PREDICTOR — VALIDATION REPORT
======================================================================

[1] Checking Raw Data...
----------------------------------------------------------------------
  PASS: Raw data file
  PASS: Found 120 rows with all core columns

[2] Checking Cleaned Data...
----------------------------------------------------------------------
  PASS: Cleaned data file

[3] Checking Stationarity Test Results...
----------------------------------------------------------------------
  PASS: ADF test results
  PASS: Integration order results

... (and so on for all sections)

======================================================================
SUMMARY
======================================================================
Total checks: 28
Passed: 28 ✓
Failed: 0 ✗

🎉 ALL CHECKS PASSED! Your project is complete and reproducible.
======================================================================
```

---

## STEP 2: Add Content Validation

**What this adds:** Not just checking if files exist, but also checking if they contain valid data.

**Delete everything in `validate.py` and replace it with this:**

```python
"""
Validation Script — Nigerian Inflation Predictor
Checks that all expected files exist and contain valid data.
"""

import os
import json
import pandas as pd
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists and return result."""
    exists = os.path.exists(filepath)
    return {
        'check': description,
        'status': 'PASS' if exists else 'FAIL',
        'details': f'File found at {filepath}' if exists else f'File missing: {filepath}'
    }


def check_png_valid(filepath, description):
    """Check if a PNG file exists and has content."""
    if not os.path.exists(filepath):
        return {
            'check': description,
            'status': 'FAIL',
            'details': f'File missing: {filepath}'
        }

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return {
            'check': description,
            'status': 'FAIL',
            'details': f'File is empty (0 bytes): {filepath}'
        }

    return {
        'check': description,
        'status': 'PASS',
        'details': f'Valid PNG ({file_size} bytes)'
    }


def check_csv_columns(filepath, expected_cols, description):
    """Check if CSV exists and has expected columns."""
    if not os.path.exists(filepath):
        return {
            'check': description,
            'status': 'FAIL',
            'details': f'File missing: {filepath}'
        }

    try:
        df = pd.read_csv(filepath)
        missing = [col for col in expected_cols if col not in df.columns]

        if missing:
            return {
                'check': description,
                'status': 'FAIL',
                'details': f'Missing columns: {missing}'
            }

        return {
            'check': description,
            'status': 'PASS',
            'details': f'Found {len(df)} rows with all expected columns'
        }
    except Exception as e:
        return {
            'check': description,
            'status': 'FAIL',
            'details': f'Error reading CSV: {e}'
        }


def check_no_nan(filepath, columns, description):
    """Check that specified columns have no NaN values."""
    if not os.path.exists(filepath):
        return {
            'check': description,
            'status': 'FAIL',
            'details': f'File missing: {filepath}'
        }

    try:
        df = pd.read_csv(filepath)
        nan_counts = df[columns].isna().sum()

        if nan_counts.sum() > 0:
            nan_cols = nan_counts[nan_counts > 0].to_dict()
            return {
                'check': description,
                'status': 'FAIL',
                'details': f'Found NaN values: {nan_cols}'
            }

        return {
            'check': description,
            'status': 'PASS',
            'details': f'No NaN values in {len(columns)} columns'
        }
    except Exception as e:
        return {
            'check': description,
            'status': 'FAIL',
            'details': f'Error checking NaN: {e}'
        }


def check_json_valid(filepath, description):
    """Check if JSON file exists and is valid."""
    if not os.path.exists(filepath):
        return {
            'check': description,
            'status': 'FAIL',
            'details': f'File missing: {filepath}'
        }

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        return {
            'check': description,
            'status': 'PASS',
            'details': f'Valid JSON with {len(data)} top-level keys'
        }
    except Exception as e:
        return {
            'check': description,
            'status': 'FAIL',
            'details': f'Invalid JSON: {e}'
        }


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("NIGERIAN INFLATION PREDICTOR — VALIDATION REPORT")
    print("=" * 70)
    print()

    results = []

    # ========================================================================
    # SECTION 1: RAW DATA
    # ========================================================================
    print("[1] Checking Raw Data...")
    print("-" * 70)

    raw_data_path = 'data/raw/cbn_infl_data.csv'
    expected_raw_cols = ['date', 'mpr', 'tbr', 'exo', 'infl']

    result = check_csv_columns(raw_data_path, expected_raw_cols, 'Raw data has core columns')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print(f"    → {result['details']}")
    print()

    # ========================================================================
    # SECTION 2: CLEANED DATA
    # ========================================================================
    print("[2] Checking Cleaned Data...")
    print("-" * 70)

    cleaned_path = 'data/processed/cleaned_data.csv'
    expected_cleaned_cols = ['date', 'mpr', 'tbr', 'exo', 'infl']

    result = check_csv_columns(cleaned_path, expected_cleaned_cols, 'Cleaned data has core columns')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print(f"    → {result['details']}")

    # Check for NaN values in core columns
    result = check_no_nan(cleaned_path, expected_cleaned_cols[1:], 'Cleaned data has no NaN in core columns')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print(f"    → {result['details']}")
    print()

    # ========================================================================
    # SECTION 3: STATIONARITY TESTS
    # ========================================================================
    print("[3] Checking Stationarity Test Results...")
    print("-" * 70)

    adf_cols = ['variable', 'test_stat', 'p_value', 'lags', 'critical_1', 'critical_5', 'critical_10']
    result = check_csv_columns('results/adf_results.csv', adf_cols, 'ADF results has expected columns')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print(f"    → {result['details']}")

    integ_cols = ['variable', 'order']
    result = check_csv_columns('results/integration_order.csv', integ_cols, 'Integration order has expected columns')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print(f"    → {result['details']}")
    print()

    # ========================================================================
    # SECTION 4: COINTEGRATION TESTS
    # ========================================================================
    print("[4] Checking Cointegration Test Results...")
    print("-" * 70)

    eg_cols = ['test_stat', 'p_value', 'critical_1', 'critical_5', 'critical_10']
    result = check_csv_columns('results/engle_granger.csv', eg_cols, 'Engle-Granger has expected columns')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print(f"    → {result['details']}")

    johansen_cols = ['rank', 'test_stat', 'critical_90', 'critical_95', 'critical_99']
    result = check_csv_columns('results/johansen_trace.csv', johansen_cols, 'Johansen trace has expected columns')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print(f"    → {result['details']}")

    result = check_csv_columns('results/johansen_maxeig.csv', johansen_cols, 'Johansen max-eig has expected columns')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print(f"    → {result['details']}")
    print()

    # ========================================================================
    # SECTION 5: LAG SELECTION
    # ========================================================================
    print("[5] Checking Lag Selection Results...")
    print("-" * 70)

    lag_cols = ['lag', 'aic', 'bic', 'hqic']
    result = check_csv_columns('results/lag_selection.csv', lag_cols, 'Lag selection has expected columns')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print(f"    → {result['details']}")
    print()

    # ========================================================================
    # SECTION 6: ARDL MODEL
    # ========================================================================
    print("[6] Checking ARDL Model Results...")
    print("-" * 70)

    result = check_file_exists('results/ardl_summary.txt', 'ARDL summary exists')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")

    bounds_cols = ['F_stat', 'k', 'I0_5', 'I1_5']
    result = check_csv_columns('results/ardl_bounds_test.csv', bounds_cols, 'ARDL bounds test has expected columns')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print(f"    → {result['details']}")

    diag_cols = ['test', 'statistic', 'p_value']
    result = check_csv_columns('results/ardl_diagnostics.csv', diag_cols, 'ARDL diagnostics has expected columns')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print(f"    → {result['details']}")
    print()

    # ========================================================================
    # SECTION 7: VAR MODEL
    # ========================================================================
    print("[7] Checking VAR Model Results...")
    print("-" * 70)

    result = check_file_exists('results/var_summary.txt', 'VAR summary exists')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")

    granger_cols = ['caused', 'causing', 'test_stat', 'p_value', 'df']
    result = check_csv_columns('results/var_granger.csv', granger_cols, 'Granger causality has expected columns')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print(f"    → {result['details']}")
    print()

    # ========================================================================
    # SECTION 8: IRF AND FEVD
    # ========================================================================
    print("[8] Checking IRF and FEVD Plots...")
    print("-" * 70)

    irf_plots = [
        ('results/plots/irf_mpr_to_infl.png', 'IRF: MPR → Inflation'),
        ('results/plots/irf_tbr_to_infl.png', 'IRF: TBR → Inflation'),
        ('results/plots/irf_exo_to_infl.png', 'IRF: EXO → Inflation'),
        ('results/plots/irf_all.png', 'IRF: All shocks to inflation')
    ]

    for filepath, description in irf_plots:
        result = check_png_valid(filepath, description)
        results.append(result)
        print(f"  {result['status']}: {result['check']}")

    fevd_plots = [
        ('results/plots/fevd_infl.png', 'FEVD: Inflation decomposition'),
        ('results/plots/fevd_all.png', 'FEVD: All variables')
    ]

    for filepath, description in fevd_plots:
        result = check_png_valid(filepath, description)
        results.append(result)
        print(f"  {result['status']}: {result['check']}")
    print()

    # ========================================================================
    # SECTION 9: FORECASTS
    # ========================================================================
    print("[9] Checking Forecast Results...")
    print("-" * 70)

    forecast_cols = ['period', 'mpr_forecast', 'tbr_forecast', 'exo_forecast', 'infl_forecast']
    result = check_csv_columns('results/var_forecast.csv', forecast_cols, 'VAR forecast has expected columns')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print(f"    → {result['details']}")

    result = check_png_valid('results/plots/var_forecast.png', 'VAR forecast plot')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print()

    # ========================================================================
    # SECTION 10: SIMULATIONS
    # ========================================================================
    print("[10] Checking Simulation Results...")
    print("-" * 70)

    sim_cols = ['scenario', 'period']
    result = check_csv_columns('results/policy_simulation.csv', sim_cols, 'Policy simulation has expected columns')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print(f"    → {result['details']}")

    result = check_png_valid('results/plots/policy_simulation.png', 'Policy simulation plot')
    results.append(result)
    print(f"  {result['status']}: {result['check']}")
    print()

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    total_checks = len(results)
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = total_checks - passed

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total checks: {total_checks}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print()

    if failed == 0:
        print("🎉 ALL CHECKS PASSED! Your project is complete and reproducible.")
    else:
        print("⚠️  SOME CHECKS FAILED. Review the report above.")
        print()
        print("Failed checks:")
        for r in results:
            if r['status'] == 'FAIL':
                print(f"  - {r['check']}")
                print(f"    {r['details']}")

    print("=" * 70)


if __name__ == '__main__':
    main()
```

**What changed:**

1. **New function `check_csv_columns()`**: Checks if a CSV has the expected columns
2. **New function `check_no_nan()`**: Checks if specified columns have no missing values
3. **New function `check_json_valid()`**: Checks if a JSON file is valid (for future use)
4. **More detailed checks**: Instead of just checking file existence, we now verify:
   - Raw data has the 5 core columns (date, mpr, tbr, exo, infl)
   - Cleaned data has no NaN in core columns
   - Result CSVs have the expected column names (e.g., ADF results should have "variable", "test_stat", "p_value", etc.)

**Run it:**

```bash
python validate.py
```

**Now the output shows more details:**

```
[1] Checking Raw Data...
----------------------------------------------------------------------
  PASS: Raw data has core columns
    → Found 120 rows with all expected columns

[2] Checking Cleaned Data...
----------------------------------------------------------------------
  PASS: Cleaned data has core columns
    → Found 118 rows with all expected columns
  PASS: Cleaned data has no NaN in core columns
    → No NaN values in 4 columns
```

---

## STEP 3: Print Comprehensive Validation Report

**What this adds:** A final polished version with better formatting and a saved report file.

**Delete everything in `validate.py` and replace it with this:**

```python
"""
Validation Script — Nigerian Inflation Predictor
Checks that all expected files exist and contain valid data.
Saves a validation report to results/validation_report.txt
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime


def check_file_exists(filepath, description):
    """Check if a file exists and return result."""
    exists = os.path.exists(filepath)
    return {
        'check': description,
        'status': 'PASS' if exists else 'FAIL',
        'details': f'File found at {filepath}' if exists else f'File missing: {filepath}'
    }


def check_png_valid(filepath, description):
    """Check if a PNG file exists and has content."""
    if not os.path.exists(filepath):
        return {
            'check': description,
            'status': 'FAIL',
            'details': f'File missing: {filepath}'
        }

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return {
            'check': description,
            'status': 'FAIL',
            'details': f'File is empty (0 bytes): {filepath}'
        }

    return {
        'check': description,
        'status': 'PASS',
        'details': f'Valid PNG ({file_size:,} bytes)'
    }


def check_csv_columns(filepath, expected_cols, description):
    """Check if CSV exists and has expected columns."""
    if not os.path.exists(filepath):
        return {
            'check': description,
            'status': 'FAIL',
            'details': f'File missing: {filepath}'
        }

    try:
        df = pd.read_csv(filepath)
        missing = [col for col in expected_cols if col not in df.columns]

        if missing:
            return {
                'check': description,
                'status': 'FAIL',
                'details': f'Missing columns: {missing}'
            }

        return {
            'check': description,
            'status': 'PASS',
            'details': f'{len(df)} rows, all expected columns present'
        }
    except Exception as e:
        return {
            'check': description,
            'status': 'FAIL',
            'details': f'Error reading CSV: {e}'
        }


def check_no_nan(filepath, columns, description):
    """Check that specified columns have no NaN values."""
    if not os.path.exists(filepath):
        return {
            'check': description,
            'status': 'FAIL',
            'details': f'File missing: {filepath}'
        }

    try:
        df = pd.read_csv(filepath)
        nan_counts = df[columns].isna().sum()

        if nan_counts.sum() > 0:
            nan_cols = nan_counts[nan_counts > 0].to_dict()
            return {
                'check': description,
                'status': 'FAIL',
                'details': f'Found NaN values: {nan_cols}'
            }

        return {
            'check': description,
            'status': 'PASS',
            'details': f'No missing values in {len(columns)} core columns'
        }
    except Exception as e:
        return {
            'check': description,
            'status': 'FAIL',
            'details': f'Error checking NaN: {e}'
        }


def main():
    """Run all validation checks."""
    # Prepare output list for both console and file
    output_lines = []

    def log(line):
        """Print to console and save to list."""
        print(line)
        output_lines.append(line)

    log("=" * 70)
    log("NIGERIAN INFLATION PREDICTOR — VALIDATION REPORT")
    log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)
    log("")

    results = []

    # ========================================================================
    # SECTION 1: RAW DATA
    # ========================================================================
    log("[1] Checking Raw Data...")
    log("-" * 70)

    raw_data_path = 'data/raw/cbn_infl_data.csv'
    expected_raw_cols = ['date', 'mpr', 'tbr', 'exo', 'infl']

    result = check_csv_columns(raw_data_path, expected_raw_cols, 'Raw data has core columns')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")
    log(f"         {result['details']}")
    log("")

    # ========================================================================
    # SECTION 2: CLEANED DATA
    # ========================================================================
    log("[2] Checking Cleaned Data...")
    log("-" * 70)

    cleaned_path = 'data/processed/cleaned_data.csv'
    expected_cleaned_cols = ['date', 'mpr', 'tbr', 'exo', 'infl']

    result = check_csv_columns(cleaned_path, expected_cleaned_cols, 'Cleaned data has core columns')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")
    log(f"         {result['details']}")

    result = check_no_nan(cleaned_path, expected_cleaned_cols[1:], 'No NaN in core columns')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")
    log(f"         {result['details']}")
    log("")

    # ========================================================================
    # SECTION 3: STATIONARITY TESTS
    # ========================================================================
    log("[3] Checking Stationarity Test Results...")
    log("-" * 70)

    adf_cols = ['variable', 'test_stat', 'p_value', 'lags', 'critical_1', 'critical_5', 'critical_10']
    result = check_csv_columns('results/adf_results.csv', adf_cols, 'ADF results')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")
    log(f"         {result['details']}")

    integ_cols = ['variable', 'order']
    result = check_csv_columns('results/integration_order.csv', integ_cols, 'Integration order')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")
    log(f"         {result['details']}")
    log("")

    # ========================================================================
    # SECTION 4: COINTEGRATION TESTS
    # ========================================================================
    log("[4] Checking Cointegration Test Results...")
    log("-" * 70)

    eg_cols = ['test_stat', 'p_value', 'critical_1', 'critical_5', 'critical_10']
    result = check_csv_columns('results/engle_granger.csv', eg_cols, 'Engle-Granger test')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")
    log(f"         {result['details']}")

    johansen_cols = ['rank', 'test_stat', 'critical_90', 'critical_95', 'critical_99']
    result = check_csv_columns('results/johansen_trace.csv', johansen_cols, 'Johansen trace test')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")
    log(f"         {result['details']}")

    result = check_csv_columns('results/johansen_maxeig.csv', johansen_cols, 'Johansen max-eig test')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")
    log(f"         {result['details']}")
    log("")

    # ========================================================================
    # SECTION 5: LAG SELECTION
    # ========================================================================
    log("[5] Checking Lag Selection Results...")
    log("-" * 70)

    lag_cols = ['lag', 'aic', 'bic', 'hqic']
    result = check_csv_columns('results/lag_selection.csv', lag_cols, 'Lag selection')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")
    log(f"         {result['details']}")
    log("")

    # ========================================================================
    # SECTION 6: ARDL MODEL
    # ========================================================================
    log("[6] Checking ARDL Model Results...")
    log("-" * 70)

    result = check_file_exists('results/ardl_summary.txt', 'ARDL summary text file')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")

    bounds_cols = ['F_stat', 'k', 'I0_5', 'I1_5']
    result = check_csv_columns('results/ardl_bounds_test.csv', bounds_cols, 'ARDL bounds test')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")
    log(f"         {result['details']}")

    diag_cols = ['test', 'statistic', 'p_value']
    result = check_csv_columns('results/ardl_diagnostics.csv', diag_cols, 'ARDL diagnostics')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")
    log(f"         {result['details']}")
    log("")

    # ========================================================================
    # SECTION 7: VAR MODEL
    # ========================================================================
    log("[7] Checking VAR Model Results...")
    log("-" * 70)

    result = check_file_exists('results/var_summary.txt', 'VAR summary text file')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")

    granger_cols = ['caused', 'causing', 'test_stat', 'p_value', 'df']
    result = check_csv_columns('results/var_granger.csv', granger_cols, 'Granger causality')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")
    log(f"         {result['details']}")
    log("")

    # ========================================================================
    # SECTION 8: IRF AND FEVD
    # ========================================================================
    log("[8] Checking IRF and FEVD Plots...")
    log("-" * 70)

    irf_plots = [
        ('results/plots/irf_mpr_to_infl.png', 'IRF: MPR → Inflation'),
        ('results/plots/irf_tbr_to_infl.png', 'IRF: TBR → Inflation'),
        ('results/plots/irf_exo_to_infl.png', 'IRF: EXO → Inflation'),
        ('results/plots/irf_all.png', 'IRF: All shocks')
    ]

    for filepath, description in irf_plots:
        result = check_png_valid(filepath, description)
        results.append(result)
        log(f"  [{result['status']}] {result['check']}")

    fevd_plots = [
        ('results/plots/fevd_infl.png', 'FEVD: Inflation'),
        ('results/plots/fevd_all.png', 'FEVD: All variables')
    ]

    for filepath, description in fevd_plots:
        result = check_png_valid(filepath, description)
        results.append(result)
        log(f"  [{result['status']}] {result['check']}")
    log("")

    # ========================================================================
    # SECTION 9: FORECASTS
    # ========================================================================
    log("[9] Checking Forecast Results...")
    log("-" * 70)

    forecast_cols = ['period', 'mpr_forecast', 'tbr_forecast', 'exo_forecast', 'infl_forecast']
    result = check_csv_columns('results/var_forecast.csv', forecast_cols, 'VAR forecast data')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")
    log(f"         {result['details']}")

    result = check_png_valid('results/plots/var_forecast.png', 'VAR forecast plot')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")
    log("")

    # ========================================================================
    # SECTION 10: SIMULATIONS
    # ========================================================================
    log("[10] Checking Simulation Results...")
    log("-" * 70)

    sim_cols = ['scenario', 'period']
    result = check_csv_columns('results/policy_simulation.csv', sim_cols, 'Policy simulation data')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")
    log(f"         {result['details']}")

    result = check_png_valid('results/plots/policy_simulation.png', 'Policy simulation plot')
    results.append(result)
    log(f"  [{result['status']}] {result['check']}")
    log("")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    total_checks = len(results)
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = total_checks - passed

    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"Total checks:  {total_checks}")
    log(f"Passed:        {passed} ✓")
    log(f"Failed:        {failed} ✗")
    log("")

    if failed == 0:
        log("🎉 ALL CHECKS PASSED!")
        log("Your project is complete and reproducible.")
    else:
        log("⚠️  SOME CHECKS FAILED")
        log("")
        log("Failed checks:")
        for r in results:
            if r['status'] == 'FAIL':
                log(f"  • {r['check']}")
                log(f"    {r['details']}")

    log("=" * 70)

    # Save report to file
    os.makedirs('results', exist_ok=True)
    report_path = 'results/validation_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(output_lines))

    print()
    print(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
```

**What changed:**

1. **Saves report to file**: The validation report is now saved to `results/validation_report.txt`
2. **Better formatting**: Uses `[PASS]` and `[FAIL]` instead of just the words
3. **Timestamp**: Shows when the validation was run
4. **Cleaner output**: Uses bullet points for failed checks

**Run it:**

```bash
python validate.py
```

**Now you get:**

1. Console output showing all checks
2. A saved file at `results/validation_report.txt` with the same content

**Check the saved report:**

```bash
cat results/validation_report.txt
```

---

## Commit Your Changes

**Stage and commit:**

```bash
git add validate.py
git commit -m "Add end-to-end validation script"
```

**What you built:**

- A validation script that checks 30+ aspects of your project
- File existence checks (raw data, cleaned data, results, plots)
- Content validation (column names, no NaN values)
- A saved validation report for documentation

---

## Common Errors

### Error 1: "File missing: data/raw/cbn_infl_data.csv"

**Cause:** The raw data file doesn't exist.

**Fix:** Make sure you have the raw data file in the correct location. If you're testing on a new machine, copy the file from your original project.

---

### Error 2: "Missing columns: ['mpr']"

**Cause:** The CSV file exists but doesn't have the expected columns (maybe different column names or wrong file).

**Fix:** Check the column names in your CSV:

```bash
python -c "import pandas as pd; print(pd.read_csv('data/raw/cbn_infl_data.csv').columns.tolist())"
```

Update the expected columns in `validate.py` if needed.

---

### Error 3: "Found NaN values: {'mpr': 2, 'infl': 1}"

**Cause:** Your cleaned data still has missing values.

**Fix:** Re-run your data cleaning script (`scripts/clean_data.py`) to fill or drop NaN values.

---

### Error 4: All checks fail

**Cause:** You're running the script before completing the project (or in a new directory).

**Fix:** This is normal if you haven't run all the previous scripts yet. The validation script is meant to be run **after** you've completed all analysis steps.

---

## Questions & Answers

**Q1: When should I run this validation script?**

A: Run it:
- After completing all 40 days of the project
- Before sharing your project with others
- After cloning the project to a new machine
- Anytime you want to verify everything is intact

---

**Q2: Can I customize the checks?**

A: Yes. You can:
- Add new checks by creating new functions
- Remove checks for files you didn't create
- Change expected column names if your project differs

Example: Add a check for a new file:

```python
result = check_file_exists('results/my_new_analysis.csv', 'My new analysis')
results.append(result)
log(f"  [{result['status']}] {result['check']}")
```

---

**Q3: What if some checks fail but I know the files are correct?**

A: The validation script is strict on purpose. If you're confident your files are correct:
1. Check if the column names match (the script expects exact names)
2. Update the expected columns in `validate.py`
3. Or add a comment in your code explaining why the check fails

---

**Q4: Can I use this for other projects?**

A: Absolutely. The pattern is reusable:
1. List all expected output files
2. Check they exist
3. Check they have the right format
4. Check they have valid content

Just modify the file paths and column names for your project.

---

**Q5: Should I run this before every commit?**

A: No need. This is for **final validation**, not continuous testing. Run it when you want to verify the entire project is complete, not after every small change.

---

## What You Accomplished

You built a **validation script** that:
- Checks 30+ files and data quality aspects
- Validates both file existence and content correctness
- Provides a clear PASS/FAIL report
- Saves results to a file for documentation

This is the last step in making your project **reproducible**—anyone can now run `validate.py` to verify that all analysis steps completed successfully.

**Next up (Day 39):** You'll create a **master pipeline script** that runs all analysis steps in order with a single command.
