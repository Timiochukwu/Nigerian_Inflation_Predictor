# Week 1, Day 1 — Project Setup, Folder Structure & Git

## What You'll Learn Today

- How to create a professional project folder structure
- What each folder is for and why it matters
- How to use git to track your work
- How to create the foundational files every project needs

## Why This Matters

When you present this project in a thesis defense or interview, the first thing people see is your folder structure. A messy project with random scripts says "student experiment." A clean, enforced structure says "production system." Today we build that structure.

---

## Prerequisites

Before starting, make sure you have:
- **Python 3.10 or higher** installed
- **Git** installed
- A terminal (Command Prompt on Windows, Terminal on Mac/Linux)

Check by running:
```bash
python --version    # Should show 3.10+
git --version       # Should show any version
```

If Python is not installed, download from https://www.python.org/downloads/

---

## Step 1: Create the Project Folder

Open your terminal and run:

```bash
mkdir Nigerian_Inflation_Predictor
cd Nigerian_Inflation_Predictor
```

**What this does:** Creates a new empty folder and moves into it. This is your project root — everything lives inside here.

---

## Step 2: Initialize Git

```bash
git init
```

**What this does:** Turns this folder into a git repository. Git tracks every change you make, so you can always go back to a working version if something breaks.

**What you should see:**
```
Initialized empty Git repository in /path/to/Nigerian_Inflation_Predictor/.git/
```

---

## Step 3: Create the Folder Structure

```bash
mkdir -p data/raw data/processed
mkdir data_ingestion data_processing econometric_models simulation api results docs guide
```

**What this does:** Creates all the folders your project will use. Here's what each one is for:

| Folder | Purpose | Example contents |
|--------|---------|-----------------|
| `data/raw/` | Original data files exactly as downloaded from CBN/NBS | `nigeria_macro_data.csv` |
| `data/processed/` | Cleaned, ready-to-use data files | `cleaned_data.csv` |
| `data_ingestion/` | Python scripts that load raw data | `ingest.py` |
| `data_processing/` | Python scripts that clean data and run tests | `clean.py`, `stationarity.py` |
| `econometric_models/` | Python scripts for ARDL, VAR, IRF, FEVD | `ardl_model.py`, `var_model.py` |
| `simulation/` | Python scripts for policy shock simulation | `policy_shock.py` |
| `api/` | Java Spring Boot REST API (built later in Week 6) | Java source files |
| `results/` | Generated outputs — plots, tables, JSON | `irf_mpr_shock.png` |
| `docs/` | Documentation — methodology, data sources | `methodology.md` |
| `guide/` | This build guide you're reading | `week1/day1.md` |

**Rule:** Every file you create must go in one of these folders. No random scripts in the root directory (except `run_all.py` and config files).

---

## Step 4: Create Python Package Files

Python needs a special file called `__init__.py` in each folder to treat it as a "package" (a collection of related code).

Create these four files:

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

**How to create these:** Open each file in any text editor (VS Code, Notepad++, nano), type the single comment line, and save.

**Why `__init__.py`?** Without this file, Python cannot import code between folders. For example, later you will write `from data_ingestion.ingest import load_raw_data` — this only works if `data_ingestion/__init__.py` exists.

---

## Step 5: Create `.gitignore`

This file tells git which files to NOT track. You don't want to commit temporary files, virtual environments, or large data files.

**File: `.gitignore`**
```
# Python temporary files
__pycache__/
*.py[cod]
*.egg-info/

# Virtual environment (you'll create this later)
venv/
.venv/

# Environment variables (may contain passwords)
.env

# IDE settings
.idea/
.vscode/
*.swp

# Large Excel files (we use CSV instead)
data/raw/*.xlsx
data/raw/*.xls

# Java build output (for the API later)
api/target/
api/build/

# Jupyter notebook checkpoints
.ipynb_checkpoints/
```

**How to create this:** Create a file called `.gitignore` (note the dot at the start) in your project root. The dot makes it a hidden file on Mac/Linux — that's normal.

---

## Step 6: Create `.gitkeep` Files

Git doesn't track empty folders. Since `data/raw/`, `data/processed/`, `results/`, and `api/` are empty right now, we add a tiny placeholder file to each so git remembers they exist.

Create four empty files:
- `data/raw/.gitkeep`
- `data/processed/.gitkeep`
- `results/.gitkeep`
- `api/.gitkeep`

These files have no content — just create them as empty files.

```bash
touch data/raw/.gitkeep data/processed/.gitkeep results/.gitkeep api/.gitkeep
```

(On Windows without `touch`, just create empty text files with those names.)

---

## Step 7: Create `requirements.txt`

This file lists the Python packages your project needs. We start empty and add packages as we need them.

**File: `requirements.txt`**
```
# Python dependencies for Nigerian Inflation Predictor.
# Added one at a time as each module is built.
# Do NOT install everything at once.
```

---

## Step 8: Create `README.md`

This is the first thing anyone sees when they open your project. Write it like a thesis abstract.

**File: `README.md`**
```markdown
# Nigerian Inflation Predictor

## Overview

A research-grade inflation prediction system for Nigeria that models
the transmission of monetary policy shocks to consumer price inflation
using time-series econometric methods.

**Central question:** How do changes in the CBN's Monetary Policy Rate
transmit through the exchange rate and money supply channels to affect
headline inflation, and over what horizon?

## Variables (Monthly Data)

| Variable | Description | Source |
|----------|-------------|--------|
| MPR | Monetary Policy Rate (%) | Central Bank of Nigeria |
| Inflation | Headline CPI inflation (%, YoY) | National Bureau of Statistics |
| Exchange Rate | Naira per US Dollar (NGN/USD) | CBN |
| M2 | Broad Money Supply (NGN billions) | CBN Statistical Bulletin |

## Models

1. **ARDL** — Bounds testing for cointegration, short-run and long-run coefficients
2. **VAR** — With Cholesky identification (ordering: MPR -> EXR -> M2 -> INF)
3. **Impulse Response Functions** — 12-24 month horizon
4. **Forecast Error Variance Decomposition**
5. **Policy simulation** — +100 basis point MPR shock

## Project Structure

    Nigerian_Inflation_Predictor/
    ├── data/raw/              # Original datasets from CBN/NBS
    ├── data/processed/        # Cleaned, analysis-ready data
    ├── data_ingestion/        # Data loading scripts
    ├── data_processing/       # Cleaning, EDA, stationarity testing
    ├── econometric_models/    # ARDL, VAR, IRF, FEVD estimation
    ├── simulation/            # Policy shock simulation
    ├── api/                   # Java Spring Boot REST API
    ├── results/               # Generated plots, tables, JSON
    ├── docs/                  # Methodology documentation
    ├── guide/                 # Step-by-step build guide
    └── requirements.txt       # Python dependencies

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Modeling | Python 3, pandas, numpy, statsmodels | Industry-standard for applied econometrics |
| API | Java 17, Spring Boot | Production-grade REST layer for financial systems |
| Database | PostgreSQL | Structured time-series storage with ACID compliance |
```

---

## Step 9: Create `docs/data_sources.md`

This documents exactly where your data comes from. An examiner will ask.

**File: `docs/data_sources.md`**
```markdown
# Data Sources — Nigerian Inflation Predictor

All data is monthly frequency, sourced from Nigerian official statistics.

## 1. Monetary Policy Rate (MPR)

- **Column name in CSV:** `mpr`
- **Unit:** Percent per annum (%)
- **What it is:** The interest rate set by the Central Bank of Nigeria's
  Monetary Policy Committee (MPC). This is the rate at which the CBN lends
  to commercial banks. When the CBN raises the MPR, it is "tightening"
  monetary policy — making borrowing more expensive.
- **Source:** Central Bank of Nigeria
- **URL:** https://www.cbn.gov.ng/rates/mnymktind.asp
- **Note:** The MPC meets roughly every 2 months. Between meetings, the
  MPR stays constant. This is normal — repeated values are NOT missing data.

## 2. Headline Inflation Rate

- **Column name in CSV:** `inflation`
- **Unit:** Percent, year-on-year (%)
- **What it is:** How much consumer prices have risen compared to the same
  month last year. If inflation is 15%, prices are 15% higher than a year ago.
  This is the number the CBN is trying to control.
- **Source:** National Bureau of Statistics (NBS)
- **URL:** https://nigerianstat.gov.ng/
- **Note:** We use headline inflation (all items), not core inflation.
  Headline is what the CBN targets and what the public experiences.

## 3. Exchange Rate (Naira/USD)

- **Column name in CSV:** `exchange_rate`
- **Unit:** Nigerian Naira per 1 US Dollar (NGN/USD)
- **What it is:** How many Naira you need to buy one US Dollar. If the
  exchange rate goes from 400 to 800, the Naira has lost half its value
  (depreciated). This matters for inflation because Nigeria imports many
  goods — when the Naira weakens, imports cost more, pushing up prices.
- **Source:** Central Bank of Nigeria
- **URL:** https://www.cbn.gov.ng/rates/ExchRateByCurrency.asp
- **Note:** Nigeria has had multiple exchange rate windows (official, BDC,
  I&E). Use one consistent series. The June 2023 unification caused a
  large jump — this is a real event, not a data error.

## 4. Broad Money Supply (M2)

- **Column name in CSV:** `m2`
- **Unit:** Nigerian Naira, billions (NGN billions)
- **What it is:** The total amount of money circulating in the economy.
  M2 includes cash, demand deposits (current accounts), savings deposits,
  and time deposits. When M2 grows fast, there is "too much money chasing
  too few goods" — which can cause inflation.
- **Source:** CBN Statistical Bulletin
- **URL:** https://www.cbn.gov.ng/documents/Statbulletin.asp
- **Note:** M2 is a stock variable (measured at month-end). For modeling,
  we will take the natural logarithm of M2 during data processing.

## Sample Period

January 2000 to December 2024 (300 months).

## CSV File Format

All raw data must be in this exact format:

| Column | Format | Example |
|--------|--------|---------|
| `date` | YYYY-MM-DD (first of month) | 2020-01-01 |
| `mpr` | Number | 13.50 |
| `inflation` | Number | 12.13 |
| `exchange_rate` | Number | 360.50 |
| `m2` | Number | 81950.30 |

Missing values: leave empty or write NaN. Never use 0 for missing data.
```

---

## Step 10: First Git Commit

Now save everything to git:

```bash
git add -A
git commit -m "Day 1: Initialize project skeleton, folder structure, and documentation"
```

**What you should see:**
```
[main (root-commit) xxxxxxx] Day 1: Initialize project skeleton...
 X files changed, Y insertions(+)
 create mode 100644 .gitignore
 create mode 100644 README.md
 ...
```

---

## Verify Your Work

Run this command to see your project structure:

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

If you see all of these, Day 1 is complete.

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `git: command not found` | Install git: https://git-scm.com/downloads |
| `python: command not found` | Install Python. On some systems, use `python3` instead of `python` |
| Can't create `.gitignore` (Windows hides it) | In File Explorer, make sure "Show hidden files" is enabled |
| `mkdir -p` doesn't work on Windows | Use `mkdir data\raw` and `mkdir data\processed` separately |

---

## Check Your Understanding

An examiner or interviewer might ask:

1. **"Why do you separate raw and processed data?"**
   > Raw data is exactly as downloaded from CBN/NBS — never modified. Processed data has been cleaned and transformed. Keeping them separate means you can always trace back to the original source.

2. **"Why use git for an academic project?"**
   > Git provides version control — every change is recorded. If a model breaks after an edit, I can see exactly what changed and revert. It also provides a complete audit trail of the research process.

3. **"Why not just put all Python files in one folder?"**
   > Separating by function (ingestion, processing, models, simulation) enforces modularity. Each module has a single responsibility. This makes the code easier to test, debug, and explain.

---

## What You Built Today

| Item | File/Folder | Purpose |
|------|------------|---------|
| Project root | `Nigerian_Inflation_Predictor/` | Everything lives here |
| Data folders | `data/raw/`, `data/processed/` | Raw and cleaned data |
| Python modules | `data_ingestion/`, `data_processing/`, `econometric_models/`, `simulation/` | Code organized by function |
| API folder | `api/` | Java Spring Boot (Week 6) |
| Output folder | `results/` | Generated plots and tables |
| Documentation | `docs/data_sources.md` | Where data comes from |
| Config files | `.gitignore`, `requirements.txt` | Project configuration |
| README | `README.md` | Project overview |

**Packages installed today:** None. Day 1 is structural only.

**Tomorrow (Day 2):** You'll install `pandas` and learn the Python fundamentals needed for this project.
