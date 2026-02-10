# DAY 25 — Results Compilation: Building the Results Summary

## What You'll Learn Today
- How to compile all econometric results into a single summary
- Creating a master results JSON file for the API (Week 7)
- Generating publication-quality summary tables

Today you'll build `results/compile_results.py` to consolidate all your econometric findings from Weeks 2-4 into a single master file. This makes it easy to access key results for the policy simulation (Week 6) and API (Week 7).

---

## Building results/compile_results.py — 3 Steps

### STEP 1: Imports, Load All Result CSVs, Create Master Summary Dictionary

Delete everything in `results/compile_results.py` and replace it with this:

```python
import pandas as pd
import json
from pathlib import Path

# Set up paths
results_dir = Path(__file__).parent
data_dir = results_dir.parent / "data"

def load_all_results():
    """Load all result CSV files from results/ directory."""
    results = {}

    # Load integration order results (Week 2)
    try:
        integration_df = pd.read_csv(results_dir / "integration_order.csv")
        results['integration_order'] = integration_df.to_dict('records')
    except FileNotFoundError:
        results['integration_order'] = []

    # Load cointegration results (Week 2)
    try:
        coint_df = pd.read_csv(results_dir / "cointegration_results.csv")
        results['cointegration'] = coint_df.to_dict('records')
    except FileNotFoundError:
        results['cointegration'] = []

    # Load ARDL results (Week 3)
    try:
        ardl_bounds_df = pd.read_csv(results_dir / "ardl_bounds_test.csv")
        results['ardl_bounds'] = ardl_bounds_df.to_dict('records')
    except FileNotFoundError:
        results['ardl_bounds'] = []

    try:
        ardl_lr_df = pd.read_csv(results_dir / "ardl_longrun_coefficients.csv")
        results['ardl_longrun'] = ardl_lr_df.to_dict('records')
    except FileNotFoundError:
        results['ardl_longrun'] = []

    try:
        ardl_ect_df = pd.read_csv(results_dir / "ardl_ect.csv")
        results['ardl_ect'] = ardl_ect_df.to_dict('records')
    except FileNotFoundError:
        results['ardl_ect'] = []

    # Load VAR results (Week 4)
    try:
        granger_df = pd.read_csv(results_dir / "granger_causality.csv")
        results['granger_causality'] = granger_df.to_dict('records')
    except FileNotFoundError:
        results['granger_causality'] = []

    try:
        irf_summary_df = pd.read_csv(results_dir / "irf_summary.csv")
        results['irf_summary'] = irf_summary_df.to_dict('records')
    except FileNotFoundError:
        results['irf_summary'] = []

    try:
        fevd_df = pd.read_csv(results_dir / "fevd_summary.csv")
        results['fevd_summary'] = fevd_df.to_dict('records')
    except FileNotFoundError:
        results['fevd_summary'] = []

    # Load model comparison (Day 23)
    try:
        comparison_df = pd.read_csv(results_dir / "model_comparison.csv")
        results['model_comparison'] = comparison_df.to_dict('records')
    except FileNotFoundError:
        results['model_comparison'] = []

    return results

def create_master_summary():
    """Create master summary dictionary with all results."""
    master = {
        'project': 'Nigerian Inflation Predictor',
        'description': 'Econometric Analysis of Inflation Dynamics in Nigeria',
        'variables': {
            'mpr': 'Monetary Policy Rate (%)',
            'infl': 'Inflation Rate (%)',
            'exo': 'Exchange Rate (NGN/USD)',
            'tbr': 'Treasury Bill Rate (%)'
        },
        'results': load_all_results()
    }

    return master

if __name__ == "__main__":
    master = create_master_summary()
    print("Master summary created successfully!")
    print(f"Total sections: {len(master['results'])}")
```

**What this does:**
- `load_all_results()`: Loads all CSV files from `results/` directory, converts each to a list of dictionaries
- Uses try/except to handle missing files gracefully
- `create_master_summary()`: Creates the master dictionary with project metadata and all results
- This structure makes it easy to access any result via `master['results']['ardl_longrun']`

---

### STEP 2: Add Compilation Functions

Delete everything in `results/compile_results.py` and replace it with this:

```python
import pandas as pd
import json
from pathlib import Path

# Set up paths
results_dir = Path(__file__).parent
data_dir = results_dir.parent / "data"

def load_all_results():
    """Load all result CSV files from results/ directory."""
    results = {}

    # Load integration order results (Week 2)
    try:
        integration_df = pd.read_csv(results_dir / "integration_order.csv")
        results['integration_order'] = integration_df.to_dict('records')
    except FileNotFoundError:
        results['integration_order'] = []

    # Load cointegration results (Week 2)
    try:
        coint_df = pd.read_csv(results_dir / "cointegration_results.csv")
        results['cointegration'] = coint_df.to_dict('records')
    except FileNotFoundError:
        results['cointegration'] = []

    # Load ARDL results (Week 3)
    try:
        ardl_bounds_df = pd.read_csv(results_dir / "ardl_bounds_test.csv")
        results['ardl_bounds'] = ardl_bounds_df.to_dict('records')
    except FileNotFoundError:
        results['ardl_bounds'] = []

    try:
        ardl_lr_df = pd.read_csv(results_dir / "ardl_longrun_coefficients.csv")
        results['ardl_longrun'] = ardl_lr_df.to_dict('records')
    except FileNotFoundError:
        results['ardl_longrun'] = []

    try:
        ardl_ect_df = pd.read_csv(results_dir / "ardl_ect.csv")
        results['ardl_ect'] = ardl_ect_df.to_dict('records')
    except FileNotFoundError:
        results['ardl_ect'] = []

    # Load VAR results (Week 4)
    try:
        granger_df = pd.read_csv(results_dir / "granger_causality.csv")
        results['granger_causality'] = granger_df.to_dict('records')
    except FileNotFoundError:
        results['granger_causality'] = []

    try:
        irf_summary_df = pd.read_csv(results_dir / "irf_summary.csv")
        results['irf_summary'] = irf_summary_df.to_dict('records')
    except FileNotFoundError:
        results['irf_summary'] = []

    try:
        fevd_df = pd.read_csv(results_dir / "fevd_summary.csv")
        results['fevd_summary'] = fevd_df.to_dict('records')
    except FileNotFoundError:
        results['fevd_summary'] = []

    # Load model comparison (Day 23)
    try:
        comparison_df = pd.read_csv(results_dir / "model_comparison.csv")
        results['model_comparison'] = comparison_df.to_dict('records')
    except FileNotFoundError:
        results['model_comparison'] = []

    return results

def compile_integration_summary(results):
    """Extract integration order summary."""
    summary = {}
    for record in results.get('integration_order', []):
        var = record.get('variable', 'unknown')
        summary[var] = {
            'order': record.get('order', 'unknown'),
            'level_pvalue': record.get('adf_level_pvalue'),
            'diff_pvalue': record.get('adf_diff_pvalue')
        }
    return summary

def compile_cointegration_summary(results):
    """Extract cointegration summary."""
    summary = {}
    for record in results.get('cointegration', []):
        test_name = record.get('test', 'unknown')
        summary[test_name] = {
            'statistic': record.get('statistic'),
            'pvalue': record.get('pvalue'),
            'conclusion': record.get('conclusion', 'unknown')
        }
    return summary

def compile_ardl_summary(results):
    """Extract ARDL model summary."""
    summary = {}

    # Bounds test
    bounds_records = results.get('ardl_bounds', [])
    if bounds_records:
        record = bounds_records[0]
        summary['bounds_test'] = {
            'f_statistic': record.get('f_statistic'),
            'lower_bound': record.get('lower_bound'),
            'upper_bound': record.get('upper_bound'),
            'conclusion': record.get('conclusion', 'unknown')
        }

    # Long-run coefficients
    summary['longrun_coefficients'] = {}
    for record in results.get('ardl_longrun', []):
        var = record.get('variable', 'unknown')
        summary['longrun_coefficients'][var] = {
            'coefficient': record.get('coefficient'),
            'std_error': record.get('std_error'),
            'pvalue': record.get('pvalue')
        }

    # Error Correction Term
    ect_records = results.get('ardl_ect', [])
    if ect_records:
        record = ect_records[0]
        summary['error_correction'] = {
            'ect_coefficient': record.get('ect_coefficient'),
            'ect_pvalue': record.get('ect_pvalue'),
            'adjustment_speed': abs(record.get('ect_coefficient', 0))
        }

    return summary

def compile_var_summary(results):
    """Extract VAR model summary."""
    summary = {}

    # Granger causality
    summary['granger_causality'] = {}
    for record in results.get('granger_causality', []):
        cause = record.get('cause', 'unknown')
        effect = record.get('effect', 'unknown')
        key = f"{cause}_to_{effect}"
        summary['granger_causality'][key] = {
            'f_statistic': record.get('f_statistic'),
            'pvalue': record.get('pvalue'),
            'significant': record.get('significant', False)
        }

    # IRF summary (peak responses)
    summary['irf_peak_responses'] = {}
    for record in results.get('irf_summary', []):
        shock = record.get('shock', 'unknown')
        response = record.get('response', 'unknown')
        key = f"{shock}_to_{response}"
        summary['irf_peak_responses'][key] = {
            'peak_month': record.get('peak_month'),
            'peak_value': record.get('peak_value'),
            'cumulative_12m': record.get('cumulative_12m')
        }

    # FEVD at 12 and 24 months
    summary['fevd_12m'] = {}
    summary['fevd_24m'] = {}
    for record in results.get('fevd_summary', []):
        var = record.get('variable', 'unknown')
        horizon = record.get('horizon', 0)
        if horizon == 12:
            summary['fevd_12m'][var] = {
                'own_shock': record.get('own_shock'),
                'mpr_shock': record.get('mpr_shock'),
                'exo_shock': record.get('exo_shock'),
                'tbr_shock': record.get('tbr_shock')
            }
        elif horizon == 24:
            summary['fevd_24m'][var] = {
                'own_shock': record.get('own_shock'),
                'mpr_shock': record.get('mpr_shock'),
                'exo_shock': record.get('exo_shock'),
                'tbr_shock': record.get('tbr_shock')
            }

    return summary

def compile_model_comparison(results):
    """Extract model comparison metrics."""
    comparison = {}
    for record in results.get('model_comparison', []):
        model = record.get('model', 'unknown')
        comparison[model] = {
            'rmse': record.get('rmse'),
            'mae': record.get('mae'),
            'mape': record.get('mape'),
            'r2': record.get('r2')
        }
    return comparison

def create_master_summary():
    """Create master summary dictionary with all results."""
    results = load_all_results()

    master = {
        'project': 'Nigerian Inflation Predictor',
        'description': 'Econometric Analysis of Inflation Dynamics in Nigeria',
        'variables': {
            'mpr': 'Monetary Policy Rate (%)',
            'infl': 'Inflation Rate (%)',
            'exo': 'Exchange Rate (NGN/USD)',
            'tbr': 'Treasury Bill Rate (%)'
        },
        'results': {
            'raw_data': results,
            'integration_order': compile_integration_summary(results),
            'cointegration': compile_cointegration_summary(results),
            'ardl_model': compile_ardl_summary(results),
            'var_model': compile_var_summary(results),
            'model_comparison': compile_model_comparison(results)
        }
    }

    return master

if __name__ == "__main__":
    master = create_master_summary()
    print("Master summary created successfully!")
    print(f"Total sections: {len(master['results'])}")
```

**What this does:**
- `compile_integration_summary()`: Creates clean dictionary with integration order for each variable
- `compile_cointegration_summary()`: Extracts Johansen and Engle-Granger test results
- `compile_ardl_summary()`: Compiles bounds test, long-run coefficients, and ECT
- `compile_var_summary()`: Extracts Granger causality, IRF peaks, and FEVD at key horizons
- `compile_model_comparison()`: Creates model performance dictionary
- `create_master_summary()`: Now includes both raw data and compiled summaries

---

### STEP 3: Save JSON and Generate Text Summary

Delete everything in `results/compile_results.py` and replace it with this:

```python
import pandas as pd
import json
from pathlib import Path

# Set up paths
results_dir = Path(__file__).parent
data_dir = results_dir.parent / "data"

def load_all_results():
    """Load all result CSV files from results/ directory."""
    results = {}

    # Load integration order results (Week 2)
    try:
        integration_df = pd.read_csv(results_dir / "integration_order.csv")
        results['integration_order'] = integration_df.to_dict('records')
    except FileNotFoundError:
        results['integration_order'] = []

    # Load cointegration results (Week 2)
    try:
        coint_df = pd.read_csv(results_dir / "cointegration_results.csv")
        results['cointegration'] = coint_df.to_dict('records')
    except FileNotFoundError:
        results['cointegration'] = []

    # Load ARDL results (Week 3)
    try:
        ardl_bounds_df = pd.read_csv(results_dir / "ardl_bounds_test.csv")
        results['ardl_bounds'] = ardl_bounds_df.to_dict('records')
    except FileNotFoundError:
        results['ardl_bounds'] = []

    try:
        ardl_lr_df = pd.read_csv(results_dir / "ardl_longrun_coefficients.csv")
        results['ardl_longrun'] = ardl_lr_df.to_dict('records')
    except FileNotFoundError:
        results['ardl_longrun'] = []

    try:
        ardl_ect_df = pd.read_csv(results_dir / "ardl_ect.csv")
        results['ardl_ect'] = ardl_ect_df.to_dict('records')
    except FileNotFoundError:
        results['ardl_ect'] = []

    # Load VAR results (Week 4)
    try:
        granger_df = pd.read_csv(results_dir / "granger_causality.csv")
        results['granger_causality'] = granger_df.to_dict('records')
    except FileNotFoundError:
        results['granger_causality'] = []

    try:
        irf_summary_df = pd.read_csv(results_dir / "irf_summary.csv")
        results['irf_summary'] = irf_summary_df.to_dict('records')
    except FileNotFoundError:
        results['irf_summary'] = []

    try:
        fevd_df = pd.read_csv(results_dir / "fevd_summary.csv")
        results['fevd_summary'] = fevd_df.to_dict('records')
    except FileNotFoundError:
        results['fevd_summary'] = []

    # Load model comparison (Day 23)
    try:
        comparison_df = pd.read_csv(results_dir / "model_comparison.csv")
        results['model_comparison'] = comparison_df.to_dict('records')
    except FileNotFoundError:
        results['model_comparison'] = []

    return results

def compile_integration_summary(results):
    """Extract integration order summary."""
    summary = {}
    for record in results.get('integration_order', []):
        var = record.get('variable', 'unknown')
        summary[var] = {
            'order': record.get('order', 'unknown'),
            'level_pvalue': record.get('adf_level_pvalue'),
            'diff_pvalue': record.get('adf_diff_pvalue')
        }
    return summary

def compile_cointegration_summary(results):
    """Extract cointegration summary."""
    summary = {}
    for record in results.get('cointegration', []):
        test_name = record.get('test', 'unknown')
        summary[test_name] = {
            'statistic': record.get('statistic'),
            'pvalue': record.get('pvalue'),
            'conclusion': record.get('conclusion', 'unknown')
        }
    return summary

def compile_ardl_summary(results):
    """Extract ARDL model summary."""
    summary = {}

    # Bounds test
    bounds_records = results.get('ardl_bounds', [])
    if bounds_records:
        record = bounds_records[0]
        summary['bounds_test'] = {
            'f_statistic': record.get('f_statistic'),
            'lower_bound': record.get('lower_bound'),
            'upper_bound': record.get('upper_bound'),
            'conclusion': record.get('conclusion', 'unknown')
        }

    # Long-run coefficients
    summary['longrun_coefficients'] = {}
    for record in results.get('ardl_longrun', []):
        var = record.get('variable', 'unknown')
        summary['longrun_coefficients'][var] = {
            'coefficient': record.get('coefficient'),
            'std_error': record.get('std_error'),
            'pvalue': record.get('pvalue')
        }

    # Error Correction Term
    ect_records = results.get('ardl_ect', [])
    if ect_records:
        record = ect_records[0]
        summary['error_correction'] = {
            'ect_coefficient': record.get('ect_coefficient'),
            'ect_pvalue': record.get('ect_pvalue'),
            'adjustment_speed': abs(record.get('ect_coefficient', 0))
        }

    return summary

def compile_var_summary(results):
    """Extract VAR model summary."""
    summary = {}

    # Granger causality
    summary['granger_causality'] = {}
    for record in results.get('granger_causality', []):
        cause = record.get('cause', 'unknown')
        effect = record.get('effect', 'unknown')
        key = f"{cause}_to_{effect}"
        summary['granger_causality'][key] = {
            'f_statistic': record.get('f_statistic'),
            'pvalue': record.get('pvalue'),
            'significant': record.get('significant', False)
        }

    # IRF summary (peak responses)
    summary['irf_peak_responses'] = {}
    for record in results.get('irf_summary', []):
        shock = record.get('shock', 'unknown')
        response = record.get('response', 'unknown')
        key = f"{shock}_to_{response}"
        summary['irf_peak_responses'][key] = {
            'peak_month': record.get('peak_month'),
            'peak_value': record.get('peak_value'),
            'cumulative_12m': record.get('cumulative_12m')
        }

    # FEVD at 12 and 24 months
    summary['fevd_12m'] = {}
    summary['fevd_24m'] = {}
    for record in results.get('fevd_summary', []):
        var = record.get('variable', 'unknown')
        horizon = record.get('horizon', 0)
        if horizon == 12:
            summary['fevd_12m'][var] = {
                'own_shock': record.get('own_shock'),
                'mpr_shock': record.get('mpr_shock'),
                'exo_shock': record.get('exo_shock'),
                'tbr_shock': record.get('tbr_shock')
            }
        elif horizon == 24:
            summary['fevd_24m'][var] = {
                'own_shock': record.get('own_shock'),
                'mpr_shock': record.get('mpr_shock'),
                'exo_shock': record.get('exo_shock'),
                'tbr_shock': record.get('tbr_shock')
            }

    return summary

def compile_model_comparison(results):
    """Extract model comparison metrics."""
    comparison = {}
    for record in results.get('model_comparison', []):
        model = record.get('model', 'unknown')
        comparison[model] = {
            'rmse': record.get('rmse'),
            'mae': record.get('mae'),
            'mape': record.get('mape'),
            'r2': record.get('r2')
        }
    return comparison

def create_master_summary():
    """Create master summary dictionary with all results."""
    results = load_all_results()

    master = {
        'project': 'Nigerian Inflation Predictor',
        'description': 'Econometric Analysis of Inflation Dynamics in Nigeria',
        'variables': {
            'mpr': 'Monetary Policy Rate (%)',
            'infl': 'Inflation Rate (%)',
            'exo': 'Exchange Rate (NGN/USD)',
            'tbr': 'Treasury Bill Rate (%)'
        },
        'results': {
            'raw_data': results,
            'integration_order': compile_integration_summary(results),
            'cointegration': compile_cointegration_summary(results),
            'ardl_model': compile_ardl_summary(results),
            'var_model': compile_var_summary(results),
            'model_comparison': compile_model_comparison(results)
        }
    }

    return master

def generate_text_summary(master):
    """Generate formatted text summary for thesis/report."""
    lines = []
    lines.append("=" * 80)
    lines.append("ECONOMETRIC ANALYSIS SUMMARY")
    lines.append("Nigerian Inflation Predictor")
    lines.append("=" * 80)
    lines.append("")

    # Variables section
    lines.append("VARIABLES:")
    for var, desc in master['variables'].items():
        lines.append(f"  {var}: {desc}")
    lines.append("")

    # Integration order
    lines.append("-" * 80)
    lines.append("1. INTEGRATION ORDER (Week 2)")
    lines.append("-" * 80)
    integration = master['results']['integration_order']
    for var, data in integration.items():
        lines.append(f"\n{var}:")
        lines.append(f"  Order of Integration: {data['order']}")
        lines.append(f"  ADF (Level) p-value: {data['level_pvalue']:.4f}" if data['level_pvalue'] else "  ADF (Level) p-value: N/A")
        lines.append(f"  ADF (Diff) p-value: {data['diff_pvalue']:.4f}" if data['diff_pvalue'] else "  ADF (Diff) p-value: N/A")
    lines.append("")

    # Cointegration
    lines.append("-" * 80)
    lines.append("2. COINTEGRATION TESTS (Week 2)")
    lines.append("-" * 80)
    coint = master['results']['cointegration']
    for test, data in coint.items():
        lines.append(f"\n{test}:")
        lines.append(f"  Statistic: {data['statistic']:.4f}" if data['statistic'] else "  Statistic: N/A")
        lines.append(f"  P-value: {data['pvalue']:.4f}" if data['pvalue'] else "  P-value: N/A")
        lines.append(f"  Conclusion: {data['conclusion']}")
    lines.append("")

    # ARDL Model
    lines.append("-" * 80)
    lines.append("3. ARDL MODEL (Week 3)")
    lines.append("-" * 80)
    ardl = master['results']['ardl_model']

    if 'bounds_test' in ardl:
        lines.append("\nBounds Test for Cointegration:")
        bt = ardl['bounds_test']
        lines.append(f"  F-statistic: {bt['f_statistic']:.4f}" if bt['f_statistic'] else "  F-statistic: N/A")
        lines.append(f"  Lower Bound (I(0)): {bt['lower_bound']:.4f}" if bt['lower_bound'] else "  Lower Bound: N/A")
        lines.append(f"  Upper Bound (I(1)): {bt['upper_bound']:.4f}" if bt['upper_bound'] else "  Upper Bound: N/A")
        lines.append(f"  Conclusion: {bt['conclusion']}")

    if 'longrun_coefficients' in ardl:
        lines.append("\nLong-Run Coefficients:")
        for var, coef in ardl['longrun_coefficients'].items():
            lines.append(f"\n  {var}:")
            lines.append(f"    Coefficient: {coef['coefficient']:.4f}" if coef['coefficient'] is not None else "    Coefficient: N/A")
            lines.append(f"    Std Error: {coef['std_error']:.4f}" if coef['std_error'] else "    Std Error: N/A")
            lines.append(f"    P-value: {coef['pvalue']:.4f}" if coef['pvalue'] is not None else "    P-value: N/A")

    if 'error_correction' in ardl:
        lines.append("\nError Correction Term (ECT):")
        ect = ardl['error_correction']
        lines.append(f"  ECT Coefficient: {ect['ect_coefficient']:.4f}" if ect['ect_coefficient'] is not None else "  ECT Coefficient: N/A")
        lines.append(f"  P-value: {ect['ect_pvalue']:.4f}" if ect['ect_pvalue'] is not None else "  P-value: N/A")
        lines.append(f"  Adjustment Speed: {ect['adjustment_speed']:.4f}" if ect['adjustment_speed'] else "  Adjustment Speed: N/A")
    lines.append("")

    # VAR Model
    lines.append("-" * 80)
    lines.append("4. VAR MODEL ANALYSIS (Week 4)")
    lines.append("-" * 80)
    var_results = master['results']['var_model']

    if 'granger_causality' in var_results:
        lines.append("\nGranger Causality Tests:")
        for key, data in var_results['granger_causality'].items():
            lines.append(f"\n  {key}:")
            lines.append(f"    F-statistic: {data['f_statistic']:.4f}" if data['f_statistic'] else "    F-statistic: N/A")
            lines.append(f"    P-value: {data['pvalue']:.4f}" if data['pvalue'] is not None else "    P-value: N/A")
            lines.append(f"    Significant: {data['significant']}")

    if 'irf_peak_responses' in var_results:
        lines.append("\nImpulse Response Functions (Peak Responses):")
        for key, data in var_results['irf_peak_responses'].items():
            lines.append(f"\n  {key}:")
            lines.append(f"    Peak Month: {data['peak_month']}" if data['peak_month'] else "    Peak Month: N/A")
            lines.append(f"    Peak Value: {data['peak_value']:.4f}" if data['peak_value'] is not None else "    Peak Value: N/A")
            lines.append(f"    Cumulative (12m): {data['cumulative_12m']:.4f}" if data['cumulative_12m'] is not None else "    Cumulative (12m): N/A")

    if 'fevd_12m' in var_results:
        lines.append("\nForecast Error Variance Decomposition (12 months):")
        for var, data in var_results['fevd_12m'].items():
            lines.append(f"\n  {var}:")
            lines.append(f"    Own shock: {data['own_shock']:.2f}%" if data['own_shock'] is not None else "    Own shock: N/A")
            lines.append(f"    MPR shock: {data['mpr_shock']:.2f}%" if data['mpr_shock'] is not None else "    MPR shock: N/A")
            lines.append(f"    EXO shock: {data['exo_shock']:.2f}%" if data['exo_shock'] is not None else "    EXO shock: N/A")
            lines.append(f"    TBR shock: {data['tbr_shock']:.2f}%" if data['tbr_shock'] is not None else "    TBR shock: N/A")
    lines.append("")

    # Model comparison
    lines.append("-" * 80)
    lines.append("5. MODEL COMPARISON (Day 23)")
    lines.append("-" * 80)
    comparison = master['results']['model_comparison']
    for model, metrics in comparison.items():
        lines.append(f"\n{model}:")
        lines.append(f"  RMSE: {metrics['rmse']:.4f}" if metrics['rmse'] is not None else "  RMSE: N/A")
        lines.append(f"  MAE: {metrics['mae']:.4f}" if metrics['mae'] is not None else "  MAE: N/A")
        lines.append(f"  MAPE: {metrics['mape']:.2f}%" if metrics['mape'] is not None else "  MAPE: N/A")
        lines.append(f"  R²: {metrics['r2']:.4f}" if metrics['r2'] is not None else "  R²: N/A")
    lines.append("")

    lines.append("=" * 80)
    lines.append("END OF SUMMARY")
    lines.append("=" * 80)

    return "\n".join(lines)

def save_results(master):
    """Save master results to JSON and text summary."""
    # Save JSON
    json_path = results_dir / "master_results.json"
    with open(json_path, 'w') as f:
        json.dump(master, f, indent=2)
    print(f"Saved master results to: {json_path}")

    # Save text summary
    text_summary = generate_text_summary(master)
    txt_path = results_dir / "econometric_summary.txt"
    with open(txt_path, 'w') as f:
        f.write(text_summary)
    print(f"Saved text summary to: {txt_path}")

    return json_path, txt_path

if __name__ == "__main__":
    print("Compiling all econometric results...")
    master = create_master_summary()
    json_path, txt_path = save_results(master)
    print("\nResults compilation complete!")
    print(f"\nFiles created:")
    print(f"  - {json_path}")
    print(f"  - {txt_path}")
```

**What this does:**
- `generate_text_summary()`: Creates a publication-quality text summary with all key results, formatted for a thesis chapter
- `save_results()`: Saves both the master JSON (for API/programmatic access) and text summary (for reports/presentations)
- Includes proper formatting with section headers, indentation, and value formatting
- Handles missing values gracefully with "N/A" placeholders

---

## Week 5 Recap: Structural Analysis Complete

Congratulations! You've completed Week 5. Here's what you've accomplished:

**Day 21:** Built SVAR model with sign restrictions for policy shocks
**Day 22:** Conducted structural breaks analysis using Chow and CUSUM tests
**Day 23:** Compared ARDL vs VAR vs SVAR model performance
**Day 24:** Created comprehensive visualizations for all structural results
**Day 25:** Compiled all results into master summary files

You now have a complete structural analysis of Nigerian inflation dynamics with proper identification of policy shocks and model comparison.

---

## What the Master Results Contain

The `master_results.json` file contains:

1. **Raw Data Section:** All original CSV results preserved as lists of dictionaries
2. **Integration Order:** Summary of stationarity tests for each variable (I(0) vs I(1))
3. **Cointegration:** Johansen and Engle-Granger test results
4. **ARDL Model:** Bounds test, long-run coefficients, and error correction term
5. **VAR Model:** Granger causality, IRF peaks, and FEVD decomposition
6. **Model Comparison:** RMSE, MAE, MAPE, and R² for each model

This structure makes it easy to access any result programmatically via:
```python
master['results']['ardl_model']['longrun_coefficients']['mpr']['coefficient']
```

---

## How These Feed Into Week 6 and Week 7

**Week 6 (Policy Simulation):**
- You'll use the ARDL long-run coefficients and ECT for equilibrium calculations
- VAR IRFs will inform the dynamic response paths for policy scenarios
- Model comparison metrics will help choose the best model for forecasting

**Week 7 (API Development):**
- The `master_results.json` file will be served via API endpoints
- API users can query specific results (e.g., `/api/results/ardl/longrun`)
- The structured format makes it easy to build REST endpoints

---

## Running the Code

```bash
cd /home/user/Nigerian_Inflation_Predictor
python results/compile_results.py
```

**Expected Output:**
```
Compiling all econometric results...
Saved master results to: /home/user/Nigerian_Inflation_Predictor/results/master_results.json
Saved text summary to: /home/user/Nigerian_Inflation_Predictor/results/econometric_summary.txt

Results compilation complete!

Files created:
  - /home/user/Nigerian_Inflation_Predictor/results/master_results.json
  - /home/user/Nigerian_Inflation_Predictor/results/econometric_summary.txt
```

---

## Commit Your Work

```bash
git add results/compile_results.py results/master_results.json results/econometric_summary.txt
git commit -m "Day 25: Compile all econometric results into master summary"
```

---

## Common Errors and Solutions

**Error:** `FileNotFoundError: integration_order.csv not found`
**Solution:** The code handles this gracefully with try/except. If you haven't run all previous weeks, some CSVs may be missing. The script will still run and mark those sections as empty lists.

**Error:** `KeyError: 'order'`
**Solution:** Check that your CSV files have the expected column names. The code uses `.get()` to avoid KeyErrors, but you should verify your CSV structure matches Week 2-4 outputs.

**Error:** `TypeError: 'NoneType' object is not subscriptable`
**Solution:** This happens when a result section is missing. Add a check like `if ardl.get('bounds_test'):` before accessing nested keys.

**Error:** JSON file is too large
**Solution:** The raw_data section can be large. If needed, remove it from the master dictionary: `del master['results']['raw_data']`

---

## Q&A

**Q: Why save both JSON and text formats?**
A: JSON is for programmatic access (API, scripts), text is for human readers (thesis, presentations). Different audiences need different formats.

**Q: Can I add more metrics to the master summary?**
A: Yes! Add new compilation functions and include them in `create_master_summary()`. For example, you could add diagnostic test results or residual analysis.

**Q: How do I access a specific result from the JSON file?**
A: Load it and navigate the dictionary:
```python
import json
with open('results/master_results.json') as f:
    master = json.load(f)
mpr_coef = master['results']['ardl_model']['longrun_coefficients']['mpr']['coefficient']
```

**Q: Should I version control the JSON file?**
A: Yes. It's a small file (usually < 100KB) and serves as a snapshot of your analysis. Future you will thank current you for having this historical record.

**Q: What if I want to add new models later?**
A: Just update `load_all_results()` to load the new CSV, add a compilation function, and include it in `create_master_summary()`. The modular structure makes it easy to extend.

---

## Next Steps

**Tomorrow (Day 26):** Begin Week 6 - Policy Simulation. You'll use these compiled results to simulate different monetary policy scenarios and forecast inflation under various conditions.

**Week 5 Complete!** You've mastered structural analysis and results compilation. Week 6 will show you how to apply these findings to real-world policy questions.
