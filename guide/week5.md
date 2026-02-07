# Week 5: Integration Testing & Thesis-Ready Documentation

**Goal:** Run the full pipeline end-to-end, write thesis-grade documentation, and interpret all results in Nigerian policy context.

**Prerequisite:** Weeks 1-4 complete. All models estimated, simulation done, API built.

---

## Day 1 — Full Pipeline Integration Script

**Objective:** Create a master script that runs everything from ingestion to simulation in one command.

### Step 1: Create the master pipeline

**File: `run_all.py`** (project root)

```python
"""
Nigerian Inflation Predictor — Full Pipeline Runner.

Runs the entire analysis from data ingestion to policy simulation.
"""

import os
import sys
import time


def run_step(step_num, total, description, func, *args):
    """Run a pipeline step with timing and error handling."""
    print(f"\n[{step_num}/{total}] {description}")
    print("-" * 50)
    start = time.time()
    try:
        result = func(*args)
        elapsed = time.time() - start
        print(f"  Completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        print(f"  FAILED: {e}")
        raise


def main():
    print("=" * 70)
    print("NIGERIAN INFLATION PREDICTOR — FULL PIPELINE")
    print("=" * 70)
    start_time = time.time()

    # Step 1: Ingest
    from data_ingestion.ingest import load_raw_data
    raw_df = run_step(1, 8, "DATA INGESTION", load_raw_data)

    # Step 2: Clean
    from data_processing.clean import clean_data, save_cleaned_data
    clean_df = run_step(2, 8, "DATA CLEANING", clean_data, raw_df)
    save_cleaned_data(clean_df)

    # Step 3: EDA
    from data_processing.eda import summary_statistics, plot_time_series, plot_correlation_matrix
    run_step(3, 8, "EXPLORATORY DATA ANALYSIS", lambda: (
        summary_statistics(clean_df),
        plot_time_series(clean_df),
        plot_correlation_matrix(clean_df)
    ))

    # Step 4: Stationarity
    from data_processing.stationarity import run_all_tests, save_results
    def stationarity_step():
        results, orders = run_all_tests(clean_df)
        save_results(results, orders)
    run_step(4, 8, "STATIONARITY TESTING", stationarity_step)

    # Step 5: ARDL
    from econometric_models.ardl_model import estimate_ardl, save_ardl_results
    def ardl_step():
        results = estimate_ardl(clean_df)
        save_ardl_results(results)
    run_step(5, 8, "ARDL ESTIMATION", ardl_step)

    # Step 6: Bounds Test + ECM
    from econometric_models.bounds_test import run_bounds_test
    run_step(6, 8, "BOUNDS TEST", run_bounds_test, clean_df)

    from econometric_models.ardl_ecm import estimate_ecm
    run_step(6, 8, "ARDL ECM", estimate_ecm, clean_df)

    # Step 7: VAR + IRF + FEVD
    from econometric_models.irf_analysis import load_and_estimate_var, compute_irfs, plot_irfs, interpret_irfs
    def var_irf_step():
        results, use_diff = load_and_estimate_var()
        irf = compute_irfs(results, periods=24)
        plot_irfs(irf, results)
        interpret_irfs(irf)

    from econometric_models.fevd_analysis import load_and_estimate_var as fevd_load, compute_fevd, create_fevd_tables, plot_fevd, interpret_fevd
    def fevd_step():
        results = fevd_load()
        fevd = compute_fevd(results, periods=24)
        fevd_df = create_fevd_tables(fevd, periods=24)
        plot_fevd(fevd)
        interpret_fevd(fevd_df)

    run_step(7, 8, "VAR + IRF ANALYSIS", var_irf_step)
    run_step(7, 8, "FEVD ANALYSIS", fevd_step)

    # Step 8: Simulation
    from simulation.policy_shock import load_and_estimate_var as sim_load, simulate_mpr_shock, plot_simulation, interpret_simulation, save_simulation
    def simulation_step():
        results, data = sim_load()
        sim = simulate_mpr_shock(results, shock_size_bps=100, horizon=24)
        plot_simulation(sim, shock_size_bps=100, horizon=24)
        interpret_simulation(sim, shock_size_bps=100)
        save_simulation(sim, shock_size_bps=100)
    run_step(8, 8, "POLICY SIMULATION", simulation_step)

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s")

    # Validate outputs
    expected_files = [
        "data/processed/cleaned_data.csv",
        "results/summary_statistics.csv",
        "results/ts_panel_all_variables.png",
        "results/correlation_matrix.png",
        "results/stationarity_tests.json",
        "results/integration_orders.csv",
        "results/ardl_summary.txt",
        "results/ardl_coefficients.csv",
        "results/ardl_ecm_results.json",
        "results/bounds_test.json",
        "results/ardl_diagnostics.png",
        "results/var_summary.txt",
        "results/var_metrics.json",
        "results/irf_mpr_shock.png",
        "results/irf_inflation_responses.png",
        "results/irf_interpretation.json",
        "results/fevd_inflation.csv",
        "results/fevd_inflation.png",
        "results/fevd_all_variables.png",
        "results/simulation_mpr_shock.png",
        "results/simulation_inflation_response.png",
        "results/simulation_results.json",
    ]

    print("\nOutput validation:")
    base = os.path.dirname(__file__)
    missing = 0
    for f in expected_files:
        exists = os.path.exists(os.path.join(base, f))
        if not exists:
            print(f"  [MISSING] {f}")
            missing += 1

    if missing == 0:
        print(f"  All {len(expected_files)} outputs present.")
    else:
        print(f"  {missing} outputs missing.")


if __name__ == "__main__":
    main()
```

### Step 2: Run the full pipeline

```bash
python run_all.py
```

All 22 output files should be present.

### Step 3: Commit

```bash
git add run_all.py
git commit -m "Week 5 Day 1: Add full pipeline runner with output validation"
```

---

## Day 2 — Methodology Documentation (Thesis Chapter)

**Objective:** Complete the `docs/methodology.md` with formal econometric methodology and all results.

### Step 1: Update `docs/methodology.md`

Open `docs/methodology.md` (created in Week 1 with skeleton headers) and fill in every section based on your actual results. Use the JSON and CSV files in `results/` to populate the tables.

The key sections to complete:

**Section 3 — Stationarity Testing:**
- Read `results/integration_orders.csv` and `results/stationarity_tests.json`
- Fill in the ADF and KPSS tables with actual test statistics and p-values
- State the integration order of each variable

**Section 4 — ARDL Model:**
- Read `results/ardl_coefficients.csv`, `results/bounds_test.json`, `results/ardl_ecm_results.json`
- Fill in lag selection, bounds test F-statistic, long-run coefficients, ECT speed
- Fill in diagnostic test results from `results/ardl_diagnostics.json`

**Section 5 — VAR Model:**
- Read `results/var_metrics.json`, `results/var_diagnostics.json`
- Fill in lag order, stability check, Granger causality

**Section 7 — IRFs:**
- Read `results/irf_interpretation.json`
- Describe the dynamic response of inflation to an MPR shock
- Reference the IRF plots

**Section 8 — FEVD:**
- Read `results/fevd_inflation.csv`
- Create a table showing variance decomposition at 1, 6, 12, 24 months

**Section 9 — Simulation:**
- Read `results/simulation_results.json`
- Describe the +100bps scenario and its inflationary impact

### Step 2: Write it

This is a manual writing step. Use the data from your results files to fill in the methodology document. Write in formal academic English. Every number must come from your actual model output.

### Step 3: Commit

```bash
git add docs/methodology.md
git commit -m "Week 5 Day 2: Complete methodology documentation with results"
```

---

## Day 3 — Results Interpretation Document

**Objective:** Create a standalone results interpretation document focused on Nigerian policy context.

### Step 1: Create the interpretation document

**File: `docs/results_interpretation.md`**

Write this document using your actual results. The template below shows the structure — fill in the numbers from your model outputs.

```markdown
# Results Interpretation — Nigerian Inflation Predictor

## 1. Stationarity Findings

[Fill with your actual ADF/KPSS results]

The stationarity analysis reveals that [variable names] are I(1), requiring
first differencing for stationarity, while [variable names] are I(0). This
mixed integration order justifies the use of the ARDL bounds testing approach,
which accommodates variables of different integration orders (Pesaran et al., 2001).

## 2. ARDL Long-Run Relationship

The bounds test yields an F-statistic of [X.XX], which exceeds the upper bound
critical value of 3.67 at the 5% significance level. This confirms the existence
of a long-run equilibrium relationship between inflation and its determinants.

### Long-Run Coefficients

| Variable | Coefficient | Interpretation |
|----------|------------|----------------|
| MPR | [X.XX] | A 1pp increase in MPR [increases/reduces] inflation by [X.XX]pp |
| Exchange Rate | [X.XXXX] | A ₦1 depreciation [increases/reduces] inflation by [X.XX]pp |
| M2 | [X.XXXXXX] | A ₦1bn increase in M2 [increases/reduces] inflation by [X.XX]pp |

### Error Correction

The ECT coefficient of [X.XX] indicates that approximately [X]% of any
disequilibrium is corrected each month, implying a half-life of approximately
[X] months.

## 3. VAR and Impulse Responses

### Monetary Policy Transmission

A one-standard-deviation shock to the MPR [reduces/increases] inflation,
with the peak effect occurring at month [X]. The cumulative 12-month effect
is [X.XX] percentage points.

[Describe the transmission: MPR -> Exchange Rate -> M2 -> Inflation]

### Exchange Rate Pass-Through

[Describe how exchange rate shocks affect inflation in Nigeria]

## 4. Variance Decomposition

At the 12-month horizon, inflation forecast variance is explained by:
- Own shocks: [X]%
- MPR: [X]%
- Exchange Rate: [X]%
- M2: [X]%

[Interpret what this means for Nigerian monetary policy]

## 5. Policy Simulation: +100bps MPR Shock

A contractionary monetary policy shock of 100 basis points:
- [Reduces/Increases] inflation by [X.XX]pp at peak (month [X])
- Cumulative 12-month effect: [X.XX]pp
- [Appreciates/Depreciates] the Naira by [X.XX] units

### Policy Implications

1. [First implication — effectiveness of monetary policy]
2. [Second implication — transmission lag]
3. [Third implication — exchange rate channel importance]

## 6. Limitations

1. The analysis uses [development/actual] data covering [period].
2. Structural breaks (e.g., June 2023 exchange rate unification) may affect
   parameter stability.
3. The Cholesky identification assumes a recursive structure that may not
   fully capture simultaneous interactions.
4. The model does not account for fiscal policy, supply-side shocks, or
   expectations channels.
```

### Step 2: Fill in with your actual numbers

Open the JSON/CSV files in `results/` and replace every `[X.XX]` placeholder.

### Step 3: Commit

```bash
git add docs/results_interpretation.md
git commit -m "Week 5 Day 3: Add results interpretation document"
```

---

## Day 4 — README Finalization

**Objective:** Update the README to reflect the completed project with actual results.

### Step 1: Update README.md

Add these sections to the existing README:

- **Results Summary** — 3-4 bullet points with key findings
- **How to Run** — exact commands to reproduce the analysis
- **API Endpoints** — table of available endpoints
- **Requirements** — Python and Java versions

Example addition for the "How to Run" section:

```markdown
## How to Run

### Prerequisites
- Python 3.10+
- Java 17
- PostgreSQL 14+
- Maven 3.8+

### Python Pipeline
```bash
# Install dependencies
pip install -r requirements.txt

# Run full analysis pipeline
python run_all.py

# Store results in database
python -m data_ingestion.store_results
```

### API
```bash
cd api/
mvn spring-boot:run
# API available at http://localhost:8080
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/data | All macro monthly data |
| GET | /api/v1/data/latest | Most recent observation |
| GET | /api/v1/data/range?start=X&end=Y | Data in date range |
| GET | /api/v1/models/{name} | Results for a model (ardl, var, simulation) |
| GET | /api/v1/models/{name}/latest | Latest result for a model |
```

### Step 2: Commit

```bash
git add README.md
git commit -m "Week 5 Day 4: Finalize README with run instructions and API docs"
```

---

## Day 5 — Final Requirements & Dependency Audit

**Objective:** Finalize `requirements.txt` with exact versions and ensure everything runs clean.

### Step 1: Final `requirements.txt`

**File: `requirements.txt`**

```
pandas==2.1.4
numpy==1.26.2
matplotlib==3.8.2
statsmodels==0.14.1
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
```

### Step 2: Clean install test

```bash
# Create fresh virtual environment
python -m venv venv_test
source venv_test/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Run full pipeline
python run_all.py

# Deactivate and remove test environment
deactivate
rm -rf venv_test
```

### Step 3: Commit

```bash
git add requirements.txt
git commit -m "Week 5 Day 5: Finalize requirements and verify clean install"
```

### What You Know After Week 5

- The full pipeline runs end-to-end from raw data to simulation
- All documentation is thesis-grade and Nigeria-specific
- The API serves results from PostgreSQL
- Dependencies are locked and reproducible
