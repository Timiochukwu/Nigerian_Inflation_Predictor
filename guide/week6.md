# Week 6: Final Validation, Polish & Completion

**Goal:** Final validation, API testing, folder structure audit, and project sign-off.

**Prerequisite:** Weeks 1-5 complete. All models, simulation, API, and documentation done.

---

## Day 1 — API End-to-End Testing

**Objective:** Verify the Spring Boot API serves correct data from PostgreSQL.

### Step 1: Ensure database is populated

```bash
# Run Python results storage
python -m data_ingestion.store_results
```

### Step 2: Start the API

```bash
cd api/
mvn spring-boot:run
```

### Step 3: Test all endpoints

In a separate terminal, run these curl commands:

```bash
# 1. Health check — get latest macro data
curl -s http://localhost:8080/api/v1/data/latest | python -m json.tool

# 2. Get all data
curl -s http://localhost:8080/api/v1/data | python -m json.tool | head -30

# 3. Date range query
curl -s "http://localhost:8080/api/v1/data/range?start=2020-01-01&end=2024-12-01" | python -m json.tool | head -30

# 4. ARDL model results
curl -s http://localhost:8080/api/v1/models/ardl | python -m json.tool

# 5. VAR results
curl -s http://localhost:8080/api/v1/models/var | python -m json.tool

# 6. Simulation results
curl -s http://localhost:8080/api/v1/models/simulation/latest | python -m json.tool

# 7. Stationarity results
curl -s http://localhost:8080/api/v1/models/stationarity | python -m json.tool
```

### Step 4: Verify each response

For each endpoint, check:
- Response status is 200
- JSON is well-formed
- Data matches what is in your `results/` JSON files
- Date ranges are correct

### Step 5: Document any issues and fix

If any endpoint fails, trace the issue:
1. Check PostgreSQL has data: `psql -d nigerian_inflation -c "SELECT COUNT(*) FROM model_results;"`
2. Check Spring Boot logs for errors
3. Fix and re-test

### No commit needed today unless fixes were made.

---

## Day 2 — Folder Structure Audit

**Objective:** Verify the project structure is clean, consistent, and contains no ad-hoc files.

### Step 1: Run a structure check

```bash
find . -not -path './.git/*' -not -path './venv/*' -not -path './api/target/*' -type f | sort
```

### Step 2: Verify against expected structure

Your project should contain ONLY these files (plus results outputs):

```
./.gitignore
./README.md
./requirements.txt
./run_all.py
./api/.gitkeep (or api/ with Spring Boot source)
./api/pom.xml
./api/src/main/java/com/nigerianinflation/api/Application.java
./api/src/main/java/com/nigerianinflation/api/controller/DataController.java
./api/src/main/java/com/nigerianinflation/api/controller/ModelController.java
./api/src/main/java/com/nigerianinflation/api/model/MacroMonthly.java
./api/src/main/java/com/nigerianinflation/api/model/ModelResult.java
./api/src/main/java/com/nigerianinflation/api/repository/MacroMonthlyRepository.java
./api/src/main/java/com/nigerianinflation/api/repository/ModelResultRepository.java
./api/src/main/java/com/nigerianinflation/api/service/InflationService.java
./api/src/main/resources/application.properties
./data/raw/nigeria_macro_data.csv
./data/raw/.gitkeep
./data/processed/cleaned_data.csv
./data/processed/.gitkeep
./data_ingestion/__init__.py
./data_ingestion/ingest.py
./data_ingestion/db_setup.py
./data_ingestion/store_results.py
./data_processing/__init__.py
./data_processing/clean.py
./data_processing/eda.py
./data_processing/stationarity.py
./data_processing/run_pipeline.py
./econometric_models/__init__.py
./econometric_models/lag_selection.py
./econometric_models/ardl_model.py
./econometric_models/bounds_test.py
./econometric_models/ardl_ecm.py
./econometric_models/ardl_diagnostics.py
./econometric_models/var_model.py
./econometric_models/irf_analysis.py
./econometric_models/fevd_analysis.py
./econometric_models/var_diagnostics.py
./econometric_models/model_summary.py
./simulation/__init__.py
./simulation/policy_shock.py
./docs/data_sources.md
./docs/methodology.md
./docs/results_interpretation.md
./guide/week1.md
./guide/week2.md
./guide/week3.md
./guide/week4.md
./guide/week5.md
./guide/week6.md
./results/.gitkeep
./results/ (various .png, .csv, .json, .txt outputs)
```

### Step 3: Remove any stray files

If you find files outside this structure (temp files, notebook checkpoints, etc.), delete them.

### Step 4: Commit if any cleanup was needed

```bash
git add -A
git commit -m "Week 6 Day 2: Clean project structure audit"
```

---

## Day 3 — Model Validation Checklist

**Objective:** Go through every model output and verify it is statistically valid and economically interpretable.

### Checklist

Work through this checklist and verify each item:

**Stationarity:**
- [ ] ADF and KPSS tests run on all 4 variables
- [ ] Tests run on both levels and first differences
- [ ] Integration orders are consistent across both tests
- [ ] Results saved in `stationarity_tests.json` and `integration_orders.csv`

**ARDL:**
- [ ] Lag order was selected using AIC
- [ ] Bounds test F-statistic computed and compared to Pesaran critical values
- [ ] Long-run coefficients extracted and have correct signs
- [ ] ECT is negative (between -1 and 0)
- [ ] Serial correlation test passes (Breusch-Godfrey p > 0.05)
- [ ] Heteroskedasticity test passes (Breusch-Pagan p > 0.05)
- [ ] Residual plots saved in `ardl_diagnostics.png`

**VAR:**
- [ ] Model is stable (all eigenvalues inside unit circle)
- [ ] Lag order selected by BIC
- [ ] Granger causality tests computed
- [ ] VAR summary saved

**IRFs:**
- [ ] Computed for 24-month horizon
- [ ] MPR shock -> all variables plotted with 95% confidence bands
- [ ] All shocks -> Inflation plotted
- [ ] Peak response and timing identified and interpreted

**FEVD:**
- [ ] Computed for 24-month horizon
- [ ] Table at horizons 1, 3, 6, 12, 18, 24 months
- [ ] Stacked area plots generated
- [ ] Dominant inflation driver identified

**Simulation:**
- [ ] +100bps shock correctly scaled
- [ ] Response plots for all variables generated
- [ ] Focused inflation response plot generated
- [ ] Policy interpretation written

### If any item fails, go back and fix it before proceeding.

---

## Day 4 — Documentation Final Review

**Objective:** Read every documentation file end-to-end and ensure it is thesis-defensible.

### Step 1: Review checklist

Read each document and verify:

**README.md:**
- [ ] Project overview is clear and accurate
- [ ] Tech stack justification included
- [ ] "How to Run" instructions work
- [ ] API endpoints documented

**docs/data_sources.md:**
- [ ] All 4 variables documented with source, URL, units
- [ ] Sample period specified
- [ ] Data format requirements clear

**docs/methodology.md:**
- [ ] All sections filled in (no "To be completed" remaining)
- [ ] Cholesky ordering justified in formal academic language
- [ ] All numbers match actual model outputs
- [ ] References included

**docs/results_interpretation.md:**
- [ ] Every result has economic interpretation
- [ ] Nigeria-specific context provided
- [ ] Policy implications stated
- [ ] Limitations acknowledged

### Step 2: Fix any gaps and commit

```bash
git add docs/
git commit -m "Week 6 Day 4: Final documentation review and corrections"
```

---

## Day 5 — Project Sign-Off

**Objective:** Final commit. Project is complete.

### Step 1: Run the full pipeline one final time

```bash
python run_all.py
```

All outputs should generate without errors.

### Step 2: Final git status

```bash
git status
```

Should show: `nothing to commit, working tree clean`

If there are uncommitted changes:
```bash
git add -A
git commit -m "Final: Project complete — Nigerian Inflation Predictor"
```

### Step 3: Verify the completion criteria

**The project is COMPLETE when:**

| Criterion | Status |
|-----------|--------|
| Stationarity tests (ADF/KPSS) produce valid results | |
| ARDL model estimated with bounds test | |
| ARDL long-run and short-run coefficients extracted | |
| ARDL diagnostics pass (or failures explained) | |
| VAR model estimated and stable | |
| IRF plots generated with confidence bands | |
| FEVD tables and plots generated | |
| +100bps simulation completed with interpretation | |
| Spring Boot API serves all results | |
| PostgreSQL stores all data and results | |
| Folder structure is clean (no ad-hoc files) | |
| Documentation is thesis-grade and Nigeria-specific | |
| Full pipeline runs end-to-end without errors | |

### Step 4: Push final commit

```bash
git push
```

---

## Project Complete

You have built a research-grade Nigerian Inflation Predictor that:

1. **Ingests** monthly CBN/NBS macroeconomic data
2. **Tests** stationarity using ADF and KPSS
3. **Estimates** ARDL models with bounds testing and ECM
4. **Estimates** VAR models with Cholesky identification
5. **Computes** IRFs showing monetary policy transmission dynamics
6. **Computes** FEVD showing inflation variance drivers
7. **Simulates** a +100bps MPR policy shock
8. **Serves** results through a Spring Boot REST API
9. **Stores** everything in PostgreSQL
10. **Documents** methodology and results in thesis-grade language

**Do not add features. Do not refactor. The project is done.**
