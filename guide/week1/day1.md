# Week 1, Day 1 — Project Setup, Folder Structure & Git

## What You'll Learn Today

- How to create a professional project folder structure from scratch
- What each folder is for and why it matters in a research project
- How to use Git to track your work so nothing is ever lost
- How to create the foundational files every serious project needs
- The 34-column CBN dataset you will work with and the 4 core modeling variables

## Why This Matters

When you present this project in a thesis defense or job interview, the first thing
people look at is your project structure. A messy folder of random scripts says
"weekend experiment." A clean structure with documentation says "production-grade
research system." Today you build that structure — once, correctly — so every file
you create for the rest of the project has a clear home.

---

## Prerequisites

Make sure you have **Python 3.10+** and **Git** installed. Verify in your terminal:

```bash
python --version    # Should print Python 3.10 or higher
git --version       # Should print any git version
```

If missing: Python from https://www.python.org/downloads/, Git from https://git-scm.com/downloads.
On some Windows systems, use `python3` instead of `python`.

---

## Step 1: Create the Project Folder

```bash
mkdir Nigerian_Inflation_Predictor
cd Nigerian_Inflation_Predictor
```

`mkdir` creates a new empty folder. `cd` moves into it. This is your project root —
everything lives here. Think of it as the compound fence; every file stays inside.

---

## Step 2: Initialize Git

```bash
git init
```

This turns the folder into a Git repository — a logbook that records every change.
If you break something on Day 12, you can revert to the working version from Day 11.

---

## Step 3: Create the Folder Structure

```bash
mkdir -p data/raw data/processed
mkdir -p data_ingestion data_processing econometric_models simulation
mkdir -p api results docs guide
```

The `-p` flag creates parent folders automatically (`data/raw` creates both `data/`
and `raw/` inside it).

| Folder | Purpose | Example Contents |
|---|---|---|
| `data/raw/` | Original data exactly as downloaded — never modified | `cbn_infl_data.csv` |
| `data/processed/` | Cleaned, analysis-ready data | `processed_data.csv` |
| `data_ingestion/` | Python scripts that load and validate raw data | `ingest.py` |
| `data_processing/` | Scripts that clean data and run statistical tests | `clean.py`, `stationarity.py` |
| `econometric_models/` | ARDL, VAR, IRF, FEVD estimation scripts | `ardl_model.py`, `var_model.py` |
| `simulation/` | Policy shock simulation scripts | `policy_shock.py` |
| `api/` | Java Spring Boot REST API (built later) | Java source files |
| `results/` | Generated outputs — plots, tables, JSON | `irf_mpr_shock.png` |
| `docs/` | Methodology notes, data source records | `data_sources.md` |
| `guide/` | This build guide you are reading | `week1/day1.md` |

**Rule:** Every file goes in one of these folders. Only config files (`.gitignore`,
`requirements.txt`, `README.md`) belong in the root.

---

## Step 4: Create Python Package Files

Python needs `__init__.py` in each folder to treat it as a package. Without it,
`from data_ingestion.ingest import load_data` will not work.

**File: `data_ingestion/__init__.py`**
```python
# Nigerian Inflation Predictor — Data Ingestion Module
```

**File: `data_processing/__init__.py`**
```python
# Nigerian Inflation Predictor — Data Processing Module
```

**File: `econometric_models/__init__.py`**
```python
# Nigerian Inflation Predictor — Econometric Models Module
```

**File: `simulation/__init__.py`**
```python
# Nigerian Inflation Predictor — Simulation Module
```

Each file contains one comment line — nothing else. Create them in any text editor.

---

## Step 5: Create `.gitignore`

Tells Git which files to never track — temp files, virtual environments, IDE settings.

**File: `.gitignore`**
```
# Python temporary files
__pycache__/
*.py[cod]
*.egg-info/

# Virtual environment
venv/
.venv/

# Environment variables (may contain passwords)
.env

# IDE and editor settings
.idea/
.vscode/
*.swp
*.swo
*~

# Large Excel files (we use CSV)
data/raw/*.xlsx
data/raw/*.xls

# Java build output (Spring Boot API)
api/target/
api/build/
api/.gradle/

# Jupyter notebook checkpoints
.ipynb_checkpoints/

# OS-generated files
.DS_Store
Thumbs.db
```

Create this file in your project root. The leading dot makes it hidden on Mac/Linux.

---

## Step 6: Create `.gitkeep` Files

Git ignores empty folders. We add placeholder files so Git remembers them.

```bash
touch data/raw/.gitkeep data/processed/.gitkeep results/.gitkeep api/.gitkeep
```

These files are empty — their only job is to preserve the folder in Git.

> **Windows:** If `touch` is not recognised, use `type nul > data\raw\.gitkeep`.

---

## Step 7: Create `requirements.txt`

**File: `requirements.txt`**
```
# Python dependencies for Nigerian Inflation Predictor.
# Packages are added one at a time as each module is built.
# Do NOT install everything at once — add as you go.
```

---

## Step 8: Create `README.md`

**File: `README.md`**
```markdown
# Nigerian Inflation Predictor

## Overview

A research-grade inflation prediction system for Nigeria that models the transmission
of monetary policy shocks to consumer price inflation using time-series econometric
methods. The system traces how changes in the CBN Monetary Policy Rate (MPR) transmit
through the money market (via treasury bill rates) and the foreign exchange market
(via the official exchange rate) to ultimately affect headline consumer price inflation.
This chain — policy rate to money market to exchange rate to prices — is the core
identification strategy.

## Dataset: All 34 Columns

The raw data file `cbn_infl_data.csv` contains 34 monthly variables from the CBN and NBS.

### Monetary Policy Variables

| Column | Description |
|---|---|
| `date` | Observation date (CBN format: `8-Jan` = Jan 2008) |
| `mpr` | **Monetary Policy Rate (%)** — CBN benchmark interest rate |
| `omo` | Open Market Operations rate (%) |
| `crr` | Cash Reserve Ratio (%) |
| `lr` | Liquidity Ratio (%) |

### Inflation Variables

| Column | Description |
|---|---|
| `infl` | **Headline inflation rate (%, YoY)** — all-items CPI |
| `inflcore` | Core inflation rate (%, YoY) — excludes food and energy |
| `inflfood` | Food inflation rate (%, YoY) |

### Consumer Price Indices (CPI)

| Column | Description |
|---|---|
| `fcpi` | Food CPI (composite) |
| `ccpi` | Core CPI (composite) |
| `acpi` | All-items CPI (composite) |
| `fcpi_r` | Food CPI (rural) |
| `ccpi_r` | Core CPI (rural) |
| `acpi_r` | All-items CPI (rural) |
| `fcpi_u` | Food CPI (urban) |
| `ccpi_u` | Core CPI (urban) |
| `acpi_u` | All-items CPI (urban) |

### Exchange Rate Variables

| Column | Description |
|---|---|
| `exo` | **Official exchange rate (NGN/USD)** — CBN official rate |
| `exbdc` | Bureau de Change exchange rate (NGN/USD) |
| `neer` | Nominal Effective Exchange Rate (index) |
| `reer` | Real Effective Exchange Rate (index) |

### Interest Rate Variables

| Column | Description |
|---|---|
| `sr` | Savings rate (%) |
| `tbr` | **91-day Treasury Bill Rate (%)** — money market benchmark |
| `ir7d` | Interbank rate, 7-day (%) |
| `ir1m` | Interbank rate, 1-month (%) |
| `ir3m` | Interbank rate, 3-month (%) |
| `ir6m` | Interbank rate, 6-month (%) |
| `ir12m` | Interbank rate, 12-month (%) |
| `irover12m` | Interbank rate, over 12-month (%) |
| `plr` | Prime Lending Rate (%) |
| `mlr` | Maximum Lending Rate (%) |
| `icr` | Interbank Call Rate (%) |
| `obb` | Open Buy Back rate (%) |

### Capital Market Variables

| Column | Description |
|---|---|
| `mktcap` | Stock market capitalisation (NGN) |
| `asi` | All Share Index |

### Core 4 Modeling Variables

| Variable | Column | Role in Model |
|---|---|---|
| Monetary Policy Rate | `mpr` | Policy instrument — the shock variable |
| Treasury Bill Rate | `tbr` | Money market transmission channel |
| Official Exchange Rate | `exo` | Exchange rate pass-through channel |
| Headline Inflation | `infl` | Target variable — what we predict |

## Models

1. **ARDL Bounds Testing** — cointegration, short-run and long-run coefficients
2. **VAR** — Cholesky ordering: MPR -> TBR -> EXO -> INF
3. **Impulse Response Functions** — +100bps MPR shock over 12-24 months
4. **Forecast Error Variance Decomposition** — inflation variance attribution
5. **Policy Simulation** — +100bps MPR shock traced through TBR and EXO to inflation

## Project Structure

```
Nigerian_Inflation_Predictor/
├── data/raw/                  # Original datasets from CBN/NBS
├── data/processed/            # Cleaned, analysis-ready data
├── data_ingestion/            # Data loading and validation scripts
├── data_processing/           # Cleaning, EDA, stationarity testing
├── econometric_models/        # ARDL, VAR, IRF, FEVD estimation
├── simulation/                # Policy shock simulation
├── api/                       # Java Spring Boot REST API
├── results/                   # Generated plots, tables, JSON
├── docs/                      # Methodology and data documentation
├── guide/                     # Step-by-step build guide
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Modeling | Python 3.10+, pandas, statsmodels | Time-series econometrics and data processing |
| API | Java 17, Spring Boot | Production-grade REST API for serving predictions |
| Database | PostgreSQL | Structured time-series storage with ACID compliance |
```

---

## Step 9: Create `docs/data_sources.md`

Documents where your data comes from and what each variable means. An examiner will
ask — having this written down protects you.

**File: `docs/data_sources.md`**
```markdown
# Data Sources — Nigerian Inflation Predictor

All data is monthly frequency, sourced from official Nigerian statistical agencies.
The raw data file is `cbn_infl_data.csv` and contains 34 columns. The core econometric
model uses 4 of these: `mpr`, `tbr`, `exo`, and `infl`. The remaining 30 columns are
available for robustness checks.

---

## Core Variable 1: Monetary Policy Rate (MPR)

- **Column name in CSV:** `mpr`
- **Unit:** Percent per annum (%)
- **What it is:** The benchmark interest rate set by the CBN Monetary Policy
  Committee (MPC). This is the rate at which the CBN lends to commercial banks.
  When the MPC raises the MPR, borrowing becomes more expensive economy-wide.
- **How it is set:** The MPC meets roughly every two months (six times a year).
  Between meetings the rate stays constant — repeated values are normal, not
  missing data.
- **Source:** Central Bank of Nigeria
- **URL:** https://www.cbn.gov.ng/rates/mnymktind.asp
- **Role in model:** Shock variable. We simulate a +100 basis point MPR increase.

## Core Variable 2: Headline Inflation Rate

- **Column name in CSV:** `infl`
- **Unit:** Percent, year-on-year (%)
- **What it is:** How much consumer prices have risen compared to the same month
  one year ago. This is the number the CBN targets and newspapers report.
- **Important:** The column is `infl`, not `inflation`. Do not rename it.
- **Source:** National Bureau of Statistics (NBS) — CPI and Inflation Report
- **URL:** https://nigerianstat.gov.ng/
- **Role in model:** Target variable — what we predict.

## Core Variable 3: Official Exchange Rate

- **Column name in CSV:** `exo`
- **Unit:** Nigerian Naira per 1 US Dollar (NGN/USD)
- **What it is:** The official CBN exchange rate. When this rises the Naira has
  depreciated. This matters because Nigeria imports many goods — a weaker Naira
  means imports cost more, pushing up consumer prices (exchange rate pass-through).
- **Important:** The column is `exo`, not `exchange_rate`. Do not rename it.
- **Source:** Central Bank of Nigeria — Exchange Rate by Currency
- **URL:** https://www.cbn.gov.ng/rates/ExchRateByCurrency.asp
- **Note:** In June 2023 the CBN unified multiple exchange rate windows into a
  single market-determined rate, causing a large jump (roughly 460 to 750+). This
  is a real policy event, not a data error.

## Core Variable 4: 91-Day Treasury Bill Rate

- **Column name in CSV:** `tbr`
- **Unit:** Percent per annum (%)
- **What it is:** The interest rate on 91-day Nigerian Treasury Bills — the most
  liquid instrument in the money market, auctioned by the CBN.
- **Why it matters:** TBR captures how MPR transmits into the money market. When
  the CBN raises the MPR, commercial banks adjust and this shows up in T-bill
  auction results. TBR is the first link in the chain: MPR -> TBR -> EXO -> INF.
- **Source:** Central Bank of Nigeria — Treasury Bills Auction Results
- **URL:** https://www.cbn.gov.ng/rates/mnymktind.asp
- **Role in model:** Sits between MPR and EXO in the Cholesky ordering.

---

## Full CSV Format

| Column | Type | Format | Example |
|---|---|---|---|
| `date` | Date | 2-digit year + month abbreviation | 8-Jan, 19-Jan |
| `mpr` | Numeric | Percentage | 18.75 |
| `tbr` | Numeric | Percentage | 5.50 |
| `exo` | Numeric | NGN per USD | 750.42 |
| `infl` | Numeric | Percentage (YoY) | 22.79 |

### All 34 column names in order

```
date, mpr, omo, crr, lr, tbr, infl, inflcore, inflfood, fcpi, ccpi, acpi,
fcpi_r, ccpi_r, acpi_r, fcpi_u, ccpi_u, acpi_u, exo, exbdc, neer, reer,
sr, ir7d, ir1m, ir3m, ir6m, ir12m, irover12m, plr, mlr, icr, obb, mktcap, asi
```

Missing values: leave empty or write `NaN`. Never use 0 — zero is a valid number.
```

---

## Step 10: First Git Commit

Save everything to Git:

```bash
git add -A
git commit -m "Day 1: Initialize project skeleton, folder structure, and documentation"
```

`git add -A` stages every new file. `git commit -m` creates a permanent snapshot.

**Expected output:**
```
[main (root-commit) abc1234] Day 1: Initialize project skeleton, folder structure, and documentation
 13 files changed, X insertions(+)
 create mode 100644 .gitignore
 create mode 100644 README.md
 ...
```

---

## Verify Your Work

```bash
find . -not -path './.git/*' -type f | sort
```

**Expected output (13 files):**
```
./.gitignore
./README.md
./api/.gitkeep
./data/processed/.gitkeep
./data/raw/.gitkeep
./data_ingestion/__init__.py
./data_processing/__init__.py
./docs/data_sources.md
./econometric_models/__init__.py
./requirements.txt
./results/.gitkeep
./simulation/__init__.py
```

If you see all 13 files, Day 1 is complete.

---

## Common Errors

| Problem | Cause | Solution |
|---|---|---|
| `git: command not found` | Git not installed | https://git-scm.com/downloads |
| `python: command not found` | Python not installed or not on PATH | https://www.python.org/downloads/ — try `python3` on some systems |
| Cannot create `.gitignore` | Windows hides dot-files | Enable "Show hidden files" in Explorer, or create via terminal |
| `mkdir -p` fails on Windows CMD | `-p` is Linux/Mac only | Run `mkdir data\raw` and `mkdir data\processed` separately |
| Git says "nothing to commit" | Already committed or files not saved | Run `git status` — if clean, you already committed successfully |
| `touch` not recognised (Windows) | `touch` is Linux/Mac | Use `type nul > filename` in CMD |

---

## Check Your Understanding

Practice answering these out loud before moving on.

**1. "Why do you separate raw data from processed data?"**

> Raw data in `data/raw/` is the original file exactly as downloaded — never modified.
> Processed data in `data/processed/` is the cleaned version ready for modeling.
> Keeping them separate means I can always trace any result back to the original
> source. This is a basic principle of reproducible research.

**2. "Your CSV has 34 columns but your model uses 4. Why not use all of them?"**

> The 4 core variables — `mpr`, `tbr`, `exo`, `infl` — represent a specific economic
> transmission chain: policy rate to money market to exchange rate to prices. Adding
> variables without theoretical justification overfits the model. The other 30 columns
> are available for robustness checks, but the core model must be parsimonious.

**3. "What is the Cholesky ordering MPR, TBR, EXO, INF and why does it matter?"**

> The ordering imposes a recursive structure in the VAR: MPR is set by the CBN and
> does not respond to other variables within the same month. TBR responds to MPR but
> not to EXO or INF contemporaneously. EXO responds to policy rates but not to
> inflation within the month. Inflation is last because prices are slowest to adjust.
> This reflects the actual speed of transmission in the Nigerian economy.

---

## What You Built Today

| Item | File or Folder | Purpose |
|---|---|---|
| Project root | `Nigerian_Inflation_Predictor/` | Top-level container |
| Raw data folder | `data/raw/` | Stores `cbn_infl_data.csv` exactly as downloaded |
| Processed data folder | `data/processed/` | Stores cleaned, model-ready data |
| Ingestion package | `data_ingestion/__init__.py` | Python package for loading raw data |
| Processing package | `data_processing/__init__.py` | Python package for cleaning and testing |
| Models package | `econometric_models/__init__.py` | Python package for ARDL, VAR, IRF, FEVD |
| Simulation package | `simulation/__init__.py` | Python package for policy shock simulation |
| API folder | `api/` | Placeholder for Java Spring Boot API |
| Results folder | `results/` | Generated plots, tables, and JSON outputs |
| Git configuration | `.gitignore` | Tells Git which files to ignore |
| Dependencies | `requirements.txt` | Python packages (empty for now) |
| Project overview | `README.md` | Full description with all 34 variables |
| Data documentation | `docs/data_sources.md` | Documents the 4 core variables and sources |
| Git repository | `.git/` | Version control — tracks every change |

**Packages installed today:** None. Day 1 is structural only.

**Tomorrow (Day 2):** You will create a Python virtual environment, install `pandas`,
and write the data ingestion script that loads `cbn_infl_data.csv` and validates all
34 columns.
