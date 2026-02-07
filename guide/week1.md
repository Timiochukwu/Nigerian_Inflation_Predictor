# Week 1: Project Setup, Data Ingestion & Stationarity Testing

**Goal:** Set up the project, ingest raw data, clean it, run EDA, test for stationarity, and load data into PostgreSQL.

---

## Day 1 — Project Skeleton & Data Ingestion

**Objective:** Create the folder structure, documentation, and data loading script.

### Step 1: Create the project folder and initialize git

```bash
mkdir Nigerian_Inflation_Predictor
cd Nigerian_Inflation_Predictor
git init
```

### Step 2: Create the folder structure

```bash
mkdir -p data/raw data/processed
mkdir data_ingestion data_processing econometric_models simulation api results docs guide
```

### Step 3: Create Python package files

Create an `__init__.py` in each Python module folder.

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

### Step 4: Create `.gitignore`

**File: `.gitignore`**
```
__pycache__/
*.py[cod]
*.egg-info/
venv/
.env
.idea/
.vscode/
*.swp
data/raw/*.xlsx
data/raw/*.xls
api/target/
api/build/
.ipynb_checkpoints/
```

### Step 5: Create empty `requirements.txt`

**File: `requirements.txt`**
```
# Dependencies added incrementally as needed.
```

### Step 6: Create `.gitkeep` files for empty directories

Create empty files at:
- `data/raw/.gitkeep`
- `data/processed/.gitkeep`
- `results/.gitkeep`
- `api/.gitkeep`

### Step 7: Create `README.md`

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

| Variable | Source |
|----------|--------|
| Monetary Policy Rate (MPR) | Central Bank of Nigeria |
| Headline Inflation Rate | National Bureau of Statistics / CBN |
| Exchange Rate (₦/$) | CBN |
| Broad Money Supply (M2) | CBN Statistical Bulletin |

## Models

1. ARDL — Bounds testing, short-run and long-run coefficients
2. VAR — With Cholesky identification (ordering: MPR → EXR → M2 → INF)
3. Impulse Response Functions — 12–24 month horizon
4. Forecast Error Variance Decomposition
5. Policy shock simulation — +100bps MPR shock

## Project Structure

    Nigerian_Inflation_Predictor/
    ├── data/raw/              # Original datasets
    ├── data/processed/        # Cleaned, analysis-ready data
    ├── data_ingestion/        # Data loading scripts
    ├── data_processing/       # Cleaning, transformation, testing
    ├── econometric_models/    # ARDL, VAR, IRF, FEVD
    ├── simulation/            # Policy shock simulation
    ├── api/                   # Java Spring Boot API
    ├── results/               # Generated outputs
    ├── docs/                  # Methodology and documentation
    ├── guide/                 # Step-by-step build guide
    └── requirements.txt

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Modeling | Python, statsmodels | Validated ARDL/VAR implementations |
| API | Java 17, Spring Boot | Production-grade REST layer |
| Database | PostgreSQL | Structured time-series storage |
```

### Step 8: Create `docs/data_sources.md`

**File: `docs/data_sources.md`**
```markdown
# Data Sources

## 1. Monetary Policy Rate (MPR)
- **Column name:** `mpr`
- **Unit:** Percent per annum (%)
- **Source:** Central Bank of Nigeria — Monetary Policy Decisions
- **URL:** https://www.cbn.gov.ng/rates/mnymktind.asp
- **Notes:** Set by the MPC. Constant between meetings. Use month-end rate.

## 2. Headline Inflation Rate
- **Column name:** `inflation`
- **Unit:** Percent, year-on-year (%)
- **Source:** National Bureau of Statistics (NBS)
- **URL:** https://nigerianstat.gov.ng/
- **Notes:** Headline (all items) CPI inflation. Published monthly.

## 3. Exchange Rate
- **Column name:** `exchange_rate`
- **Unit:** Nigerian Naira per 1 US Dollar (₦/$)
- **Source:** CBN — Exchange Rate Statistics
- **URL:** https://www.cbn.gov.ng/rates/ExchRateByCurrency.asp
- **Notes:** Use a single consistent rate series. Document which window.

## 4. Broad Money Supply (M2)
- **Column name:** `m2`
- **Unit:** Nigerian Naira, billions (₦ bn)
- **Source:** CBN — Money and Credit Statistics
- **URL:** https://www.cbn.gov.ng/documents/Statbulletin.asp
- **Notes:** End-of-month stock. Log-transform applied during processing.

## Sample Period
January 2000 to most recent available month (~290 observations).

## File Format
- CSV, UTF-8 encoded
- Date column: `date` in `YYYY-MM-DD` format (first of month)
- Missing values: empty or NaN (never zero)
```

### Step 9: Install pandas

**Update `requirements.txt`:**
```
pandas==2.1.4
```

**Install:**
```bash
pip install pandas==2.1.4
```

### Step 10: Create the sample dataset

Create a CSV file with realistic Nigerian macro data. In production, download from CBN. For development, create this file.

**File: `data/raw/nigeria_macro_data.csv`**

```csv
date,mpr,inflation,exchange_rate,m2
2000-01-01,13.50,6.62,92.34,1070.50
2000-02-01,13.50,6.93,92.55,1078.20
2000-03-01,13.50,7.87,92.69,1090.60
2000-04-01,13.50,8.13,93.05,1095.30
2000-05-01,13.50,8.68,95.10,1100.70
2000-06-01,13.50,9.01,98.20,1115.40
2000-07-01,13.50,9.54,100.50,1125.60
2000-08-01,13.50,9.21,101.00,1130.10
2000-09-01,13.50,8.76,102.30,1142.80
2000-10-01,13.50,8.54,103.80,1155.40
2000-11-01,13.50,8.02,104.20,1170.20
2000-12-01,13.50,6.94,105.50,1188.70
2001-01-01,14.00,6.22,109.20,1205.30
2001-02-01,14.00,7.35,110.50,1212.40
2001-03-01,14.00,8.50,111.00,1225.10
2001-04-01,14.00,9.10,111.80,1230.50
2001-05-01,14.00,10.30,112.20,1245.60
2001-06-01,14.00,11.40,112.50,1260.80
2001-07-01,14.00,12.80,113.00,1270.40
2001-08-01,14.00,13.50,113.60,1285.30
2001-09-01,14.00,14.90,115.00,1298.60
2001-10-01,14.00,16.10,117.00,1315.40
2001-11-01,14.00,17.30,119.00,1330.50
2001-12-01,14.00,18.87,120.00,1345.80
2002-01-01,14.50,16.80,120.50,1360.20
2002-02-01,14.50,15.40,121.00,1370.50
2002-03-01,14.50,13.10,123.00,1385.40
2002-04-01,14.50,12.80,124.50,1395.60
2002-05-01,14.50,11.90,126.00,1410.30
2002-06-01,14.50,11.50,127.50,1430.80
2002-07-01,14.50,10.20,128.00,1445.50
2002-08-01,14.50,9.10,129.30,1460.20
2002-09-01,14.50,7.60,130.50,1478.90
2002-10-01,14.50,6.80,131.00,1495.60
2002-11-01,14.50,5.90,132.50,1515.30
2002-12-01,14.50,12.17,133.50,1536.80
2003-01-01,15.00,10.90,134.00,1555.40
2003-02-01,15.00,9.20,135.00,1570.60
2003-03-01,15.00,8.50,135.50,1590.30
2003-04-01,15.00,8.10,136.00,1610.50
2003-05-01,15.00,10.50,136.50,1635.80
2003-06-01,15.00,11.80,137.00,1660.40
2003-07-01,15.00,13.00,136.80,1680.20
2003-08-01,15.00,14.80,136.50,1705.60
2003-09-01,15.00,16.50,137.00,1730.30
2003-10-01,15.00,15.90,137.50,1758.80
2003-11-01,15.00,15.20,138.00,1785.40
2003-12-01,15.00,14.03,137.00,1810.60
2004-01-01,15.00,12.50,137.50,1840.30
2004-02-01,15.00,11.80,138.00,1870.50
2004-03-01,15.00,10.30,137.80,1900.80
2004-04-01,15.00,9.40,133.50,1925.40
2004-05-01,15.00,10.70,133.00,1955.60
2004-06-01,15.00,12.30,132.80,1980.30
2004-07-01,15.00,13.50,133.20,2010.80
2004-08-01,15.00,14.00,133.50,2040.50
2004-09-01,15.00,14.60,132.80,2065.30
2004-10-01,15.00,15.80,132.50,2095.60
2004-11-01,15.00,15.20,132.00,2130.40
2004-12-01,15.00,10.02,132.35,2165.80
2005-01-01,13.00,9.10,132.00,2200.30
2005-02-01,13.00,8.50,132.50,2240.50
2005-03-01,13.00,11.60,132.00,2280.80
2005-04-01,13.00,13.00,131.50,2315.40
2005-05-01,13.00,14.70,131.00,2355.60
2005-06-01,13.00,15.30,131.20,2390.30
2005-07-01,13.00,16.00,130.80,2430.80
2005-08-01,13.00,16.90,131.00,2470.50
2005-09-01,13.00,17.90,131.50,2510.30
2005-10-01,13.00,17.50,131.00,2555.60
2005-11-01,13.00,16.60,130.50,2600.40
2005-12-01,13.00,11.56,131.00,2645.80
2006-01-01,10.00,10.50,129.00,2700.30
2006-02-01,10.00,8.60,128.50,2750.50
2006-03-01,10.00,7.50,128.00,2810.80
2006-04-01,10.00,6.80,128.50,2870.40
2006-05-01,10.00,6.50,128.00,2935.60
2006-06-01,10.00,6.20,128.30,2995.30
2006-07-01,10.00,5.70,128.00,3060.80
2006-08-01,10.00,4.30,128.50,3130.50
2006-09-01,10.00,3.50,128.00,3195.30
2006-10-01,10.00,3.00,128.30,3265.60
2006-11-01,10.00,5.60,128.00,3340.40
2006-12-01,10.00,8.53,128.50,3420.80
2007-01-01,10.00,6.50,127.00,3500.30
2007-02-01,10.00,5.20,127.50,3580.50
2007-03-01,10.00,4.80,126.80,3660.80
2007-04-01,10.00,4.20,126.50,3745.40
2007-05-01,10.00,3.80,126.00,3830.60
2007-06-01,10.00,4.30,126.50,3920.30
2007-07-01,10.00,4.80,126.00,4010.80
2007-08-01,10.00,5.20,127.00,4105.50
2007-09-01,10.00,4.10,127.50,4200.30
2007-10-01,10.00,5.00,127.00,4300.60
2007-11-01,10.00,5.50,126.50,4405.40
2007-12-01,10.00,6.56,127.00,4510.80
2008-01-01,10.00,7.80,117.00,4620.30
2008-02-01,10.00,8.60,117.50,4730.50
2008-03-01,10.00,9.50,118.00,4845.80
2008-04-01,10.00,10.20,117.50,4960.40
2008-05-01,10.00,11.30,117.00,5080.60
2008-06-01,10.00,12.00,117.50,5200.30
2008-07-01,10.00,12.40,117.00,5325.80
2008-08-01,10.00,12.00,117.50,5450.50
2008-09-01,10.25,11.60,117.00,5580.30
2008-10-01,10.25,11.00,118.50,5520.60
2008-11-01,10.25,10.50,120.00,5480.40
2008-12-01,10.25,15.06,132.00,5450.80
2009-01-01,9.75,14.60,145.00,5490.30
2009-02-01,9.75,14.30,148.00,5540.50
2009-03-01,9.75,13.90,149.00,5600.80
2009-04-01,9.75,13.30,149.50,5660.40
2009-05-01,9.75,12.40,150.00,5720.60
2009-06-01,8.00,11.20,150.50,5790.30
2009-07-01,8.00,10.40,150.00,5870.80
2009-08-01,8.00,10.40,149.50,5960.50
2009-09-01,6.00,10.50,149.00,6050.30
2009-10-01,6.00,10.30,149.50,6140.60
2009-11-01,6.00,12.40,150.00,6240.40
2009-12-01,6.00,13.93,150.50,6350.80
2010-01-01,6.00,14.40,150.00,6460.30
2010-02-01,6.00,14.90,150.50,6575.50
2010-03-01,6.00,14.60,150.00,6690.80
2010-04-01,6.00,13.10,150.50,6810.40
2010-05-01,6.00,11.70,150.00,6935.60
2010-06-01,6.00,10.30,150.50,7060.30
2010-07-01,6.25,12.80,151.00,7190.80
2010-08-01,6.25,13.20,151.50,7320.50
2010-09-01,6.25,13.60,152.00,7455.30
2010-10-01,6.25,13.40,152.50,7595.60
2010-11-01,6.25,12.80,153.00,7740.40
2010-12-01,6.25,11.80,153.50,7895.80
2011-01-01,6.50,12.10,153.00,8050.30
2011-02-01,6.50,11.30,153.50,8210.50
2011-03-01,7.50,12.80,155.00,8380.80
2011-04-01,7.50,11.60,155.50,8555.40
2011-05-01,8.00,12.40,156.00,8735.60
2011-06-01,8.00,10.20,156.50,8920.30
2011-07-01,8.75,10.30,157.00,9110.80
2011-08-01,8.75,9.30,158.00,9305.50
2011-09-01,9.25,10.30,160.00,9505.30
2011-10-01,12.00,10.50,162.00,9710.60
2011-11-01,12.00,10.40,162.50,9920.40
2011-12-01,12.00,10.30,162.00,10140.80
2012-01-01,12.00,12.60,162.50,10370.30
2012-02-01,12.00,11.90,157.50,10605.50
2012-03-01,12.00,12.10,157.80,10845.80
2012-04-01,12.00,12.90,157.50,11090.40
2012-05-01,12.00,12.70,162.00,11340.60
2012-06-01,12.00,12.80,162.30,11595.30
2012-07-01,12.00,12.80,157.50,11855.80
2012-08-01,12.00,11.70,157.80,12120.50
2012-09-01,12.00,11.30,157.50,12390.30
2012-10-01,12.00,11.70,157.30,12665.60
2012-11-01,12.00,12.30,157.50,12950.40
2012-12-01,12.00,12.00,157.30,13245.80
2013-01-01,12.00,9.00,157.50,13550.30
2013-02-01,12.00,9.50,157.30,13860.50
2013-03-01,12.00,8.60,158.00,14175.80
2013-04-01,12.00,9.10,158.50,14496.40
2013-05-01,12.00,9.00,158.00,14822.60
2013-06-01,12.00,8.40,158.50,15153.30
2013-07-01,12.00,8.70,159.00,15490.80
2013-08-01,12.00,8.20,159.50,15835.50
2013-09-01,12.00,8.00,160.00,16185.30
2013-10-01,12.00,7.80,160.50,16540.60
2013-11-01,12.00,7.90,160.00,16905.40
2013-12-01,12.00,8.00,160.50,17280.80
2014-01-01,12.00,8.00,160.00,17665.30
2014-02-01,12.00,7.70,162.00,18055.50
2014-03-01,12.00,7.80,163.00,18450.80
2014-04-01,12.00,7.90,162.50,18855.40
2014-05-01,12.00,8.00,162.80,19265.60
2014-06-01,12.00,8.20,163.00,19685.30
2014-07-01,12.00,8.30,163.50,20110.80
2014-08-01,12.00,8.50,164.00,20545.50
2014-09-01,12.00,8.30,165.00,20985.30
2014-10-01,12.00,8.10,167.00,21435.60
2014-11-01,13.00,7.90,175.00,21895.40
2014-12-01,13.00,8.00,180.00,22365.80
2015-01-01,13.00,8.20,185.00,22845.30
2015-02-01,13.00,8.40,198.00,23335.50
2015-03-01,13.00,8.50,199.00,23835.80
2015-04-01,13.00,8.70,197.00,24345.40
2015-05-01,13.00,9.00,197.50,24865.60
2015-06-01,13.00,9.20,197.00,25395.30
2015-07-01,13.00,9.20,197.50,25935.80
2015-08-01,13.00,9.30,197.00,26485.50
2015-09-01,13.00,9.40,197.50,27045.30
2015-10-01,11.00,9.30,197.00,27615.60
2015-11-01,11.00,9.40,197.50,28200.40
2015-12-01,11.00,9.55,197.00,28800.80
2016-01-01,11.00,9.60,197.50,29415.30
2016-02-01,11.00,11.40,197.00,30045.50
2016-03-01,12.00,12.80,199.00,30690.80
2016-04-01,12.00,13.70,199.50,31350.40
2016-05-01,12.00,15.60,199.00,32025.60
2016-06-01,12.00,16.50,283.00,32715.30
2016-07-01,14.00,17.10,310.00,33420.80
2016-08-01,14.00,17.60,318.00,34145.50
2016-09-01,14.00,17.90,315.00,34885.30
2016-10-01,14.00,18.30,312.00,35640.60
2016-11-01,14.00,18.48,320.00,36415.40
2016-12-01,14.00,18.55,305.00,37210.80
2017-01-01,14.00,18.72,315.00,38020.30
2017-02-01,14.00,17.78,315.50,38850.50
2017-03-01,14.00,17.26,316.00,39700.80
2017-04-01,14.00,16.25,315.50,40570.40
2017-05-01,14.00,16.25,315.00,41460.60
2017-06-01,14.00,16.10,364.00,42370.30
2017-07-01,14.00,16.05,363.50,43300.80
2017-08-01,14.00,16.01,363.00,44250.50
2017-09-01,14.00,15.98,362.50,45220.30
2017-10-01,14.00,15.91,362.00,46210.60
2017-11-01,14.00,15.90,362.50,47225.40
2017-12-01,14.00,15.37,360.00,48265.80
2018-01-01,14.00,15.13,360.50,49330.30
2018-02-01,14.00,14.33,361.00,50415.50
2018-03-01,14.00,13.34,360.50,51520.80
2018-04-01,14.00,12.48,360.00,52650.40
2018-05-01,14.00,11.61,360.50,53800.60
2018-06-01,14.00,11.23,361.00,54970.30
2018-07-01,14.00,11.14,360.50,56165.80
2018-08-01,14.00,11.28,361.00,57385.50
2018-09-01,14.00,11.28,360.50,58630.30
2018-10-01,14.00,11.26,360.00,59895.60
2018-11-01,14.00,11.28,360.50,61185.40
2018-12-01,14.00,11.44,361.00,62505.80
2019-01-01,14.00,11.37,360.50,63850.30
2019-02-01,14.00,11.31,360.00,65215.50
2019-03-01,13.50,11.25,360.50,66605.80
2019-04-01,13.50,11.37,360.00,68020.40
2019-05-01,13.50,11.40,360.50,69460.60
2019-06-01,13.50,11.22,361.00,70925.30
2019-07-01,13.50,11.08,360.50,72415.80
2019-08-01,13.50,11.02,360.00,73935.50
2019-09-01,13.50,11.24,360.50,75480.30
2019-10-01,13.50,11.61,360.00,77050.60
2019-11-01,13.50,11.85,360.50,78650.40
2019-12-01,13.50,11.98,361.00,80285.80
2020-01-01,13.50,12.13,360.50,81950.30
2020-02-01,13.50,12.20,360.00,83645.50
2020-03-01,13.50,12.26,360.50,85370.80
2020-04-01,12.50,12.34,380.00,87130.40
2020-05-01,12.50,12.40,385.00,88920.60
2020-06-01,12.50,12.56,388.00,90745.30
2020-07-01,12.50,12.82,386.00,92605.80
2020-08-01,11.50,13.22,386.50,94500.50
2020-09-01,11.50,13.71,386.00,96435.30
2020-10-01,11.50,14.23,386.50,98410.60
2020-11-01,11.50,14.89,386.00,100425.40
2020-12-01,11.50,15.75,394.00,102485.80
2021-01-01,11.50,16.47,394.50,104590.30
2021-02-01,11.50,17.33,410.00,106740.50
2021-03-01,11.50,18.17,411.00,108935.80
2021-04-01,11.50,18.12,411.50,111180.40
2021-05-01,11.50,17.93,412.00,113475.60
2021-06-01,11.50,17.75,411.50,115820.30
2021-07-01,11.50,17.38,411.00,118215.80
2021-08-01,11.50,17.01,411.50,120665.50
2021-09-01,11.50,16.63,412.00,123170.30
2021-10-01,11.50,15.99,413.00,125730.60
2021-11-01,11.50,15.40,414.00,128350.40
2021-12-01,11.50,15.63,415.00,131030.80
2022-01-01,11.50,15.60,416.00,133775.30
2022-02-01,11.50,15.70,416.50,136580.50
2022-03-01,11.50,15.92,417.00,139450.80
2022-04-01,11.50,16.82,418.00,142390.40
2022-05-01,13.00,17.71,418.50,145400.60
2022-06-01,13.00,18.60,419.00,148480.30
2022-07-01,14.00,19.64,420.00,151630.80
2022-08-01,14.00,20.52,430.00,154855.50
2022-09-01,15.50,20.77,435.00,158155.30
2022-10-01,15.50,21.09,440.00,161530.60
2022-11-01,16.50,21.47,445.00,164985.40
2022-12-01,16.50,21.34,448.00,168525.80
2023-01-01,17.50,21.82,461.00,172150.30
2023-02-01,17.50,21.91,461.50,175860.50
2023-03-01,18.00,22.04,461.00,179660.80
2023-04-01,18.00,22.22,461.50,183550.40
2023-05-01,18.00,22.41,462.00,187530.60
2023-06-01,18.50,22.79,750.00,191605.30
2023-07-01,18.75,24.08,780.00,195775.80
2023-08-01,18.75,25.80,790.00,200045.50
2023-09-01,18.75,26.72,785.00,204415.30
2023-10-01,18.75,27.33,790.00,208890.60
2023-11-01,18.75,28.20,815.00,213475.40
2023-12-01,18.75,28.92,890.00,218175.80
2024-01-01,18.75,29.90,895.00,222990.30
2024-02-01,22.75,31.70,1510.00,227925.50
2024-03-01,24.75,33.20,1550.00,232985.80
2024-04-01,24.75,33.69,1400.00,238175.40
2024-05-01,26.25,33.95,1480.00,243500.60
2024-06-01,26.25,34.19,1505.00,248965.30
2024-07-01,26.75,33.40,1590.00,254575.80
2024-08-01,26.75,32.15,1600.00,260335.50
2024-09-01,27.25,32.70,1650.00,266250.30
2024-10-01,27.25,33.88,1665.00,272325.60
2024-11-01,27.50,34.60,1680.00,278565.40
2024-12-01,27.50,34.80,1535.00,284975.80
```

> **Note:** This is development data based on publicly known Nigerian macro trends. For your thesis, replace with actual CBN/NBS data.

### Step 11: Create the ingestion script

**File: `data_ingestion/ingest.py`**

```python
"""
Data ingestion module for the Nigerian Inflation Predictor.

Loads raw CSV data from data/raw/, validates schema and types,
and returns a clean pandas DataFrame ready for processing.
"""

import os
import pandas as pd

EXPECTED_COLUMNS = {
    "date": "datetime64[ns]",
    "mpr": "float64",
    "inflation": "float64",
    "exchange_rate": "float64",
    "m2": "float64",
}

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DEFAULT_FILE = "nigeria_macro_data.csv"


def load_raw_data(filename=None):
    """
    Load raw macro data from CSV.

    Parameters
    ----------
    filename : str, optional
        Name of the CSV file in data/raw/.

    Returns
    -------
    pd.DataFrame
        DataFrame with validated columns and types, indexed by date.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    if filename is None:
        filename = DEFAULT_FILE

    filepath = os.path.join(RAW_DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Validate columns
    missing_cols = set(EXPECTED_COLUMNS.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Parse date
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # Convert numeric columns
    for col in ["mpr", "inflation", "exchange_rate", "m2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort and index
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")

    print(f"Loaded {len(df)} observations from {df.index.min()} to {df.index.max()}")
    print(f"Missing values:\n{df.isnull().sum()}")

    return df


if __name__ == "__main__":
    df = load_raw_data()
    print("\nFirst 5 rows:")
    print(df.head())
    print(f"\nShape: {df.shape}")
```

### Step 12: Test

```bash
python -m data_ingestion.ingest
```

You should see: `Loaded 300 observations from 2000-01-01 to 2024-12-01`

### Step 13: Commit

```bash
git add -A
git commit -m "Day 1: Initialize project skeleton, data sources, and ingestion pipeline"
```

---

## Day 2 — Data Cleaning & Exploratory Analysis

**Objective:** Build the cleaning pipeline and generate summary statistics with time series plots.

### Step 1: Install matplotlib

**Update `requirements.txt`:**
```
pandas==2.1.4
matplotlib==3.8.2
```

```bash
pip install matplotlib==3.8.2
```

### Step 2: Create the cleaning script

**File: `data_processing/clean.py`**

```python
"""
Data cleaning module for the Nigerian Inflation Predictor.

Handles missing values, enforces monthly frequency, validates value ranges.
"""

import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PROCESSED_FILE = "cleaned_data.csv"

VALID_RANGES = {
    "mpr": (1.0, 35.0),
    "inflation": (-5.0, 50.0),
    "exchange_rate": (50.0, 2000.0),
    "m2": (100.0, 500000.0),
}


def enforce_monthly_frequency(df):
    """Reindex to complete monthly frequency. Inserts NaN for gaps."""
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="MS")
    df_reindexed = df.reindex(full_range)
    df_reindexed.index.name = "date"

    added = len(df_reindexed) - len(df)
    if added > 0:
        print(f"  Inserted {added} missing month(s) to enforce monthly frequency.")
    return df_reindexed


def handle_missing_values(df):
    """Interpolate short gaps (up to 3 months). Flag longer gaps."""
    missing_before = df.isnull().sum()
    df_clean = df.interpolate(method="linear", limit=3)
    missing_after = df_clean.isnull().sum()

    for col in df_clean.columns:
        filled = missing_before[col] - missing_after[col]
        remaining = missing_after[col]
        if filled > 0:
            print(f"  {col}: interpolated {filled} values, {remaining} still missing")
        if remaining > 0:
            print(f"  WARNING: {col} has {remaining} unfilled missing values")
    return df_clean


def validate_ranges(df):
    """Flag values outside expected ranges (warnings only)."""
    for col, (low, high) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        outliers = df[(df[col] < low) | (df[col] > high)]
        if len(outliers) > 0:
            print(f"  WARNING: {col} has {len(outliers)} values outside [{low}, {high}]")


def clean_data(df):
    """Run the full cleaning pipeline."""
    print("Step 1: Enforcing monthly frequency...")
    df = enforce_monthly_frequency(df)

    print("Step 2: Handling missing values...")
    df = handle_missing_values(df)

    print("Step 3: Validating value ranges...")
    validate_ranges(df)

    print("Step 4: Dropping rows with remaining NaN...")
    rows_before = len(df)
    df = df.dropna()
    dropped = rows_before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with unfilled NaN values")

    print(f"\nCleaning complete. Final dataset: {len(df)} observations")
    return df


def save_cleaned_data(df):
    """Save cleaned data to data/processed/."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    filepath = os.path.join(PROCESSED_DIR, PROCESSED_FILE)
    df.to_csv(filepath)
    print(f"Saved cleaned data to {filepath}")


if __name__ == "__main__":
    from data_ingestion.ingest import load_raw_data

    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    save_cleaned_data(clean_df)
    print("\nFirst 5 rows:")
    print(clean_df.head())
```

### Step 3: Create the EDA script

**File: `data_processing/eda.py`**

```python
"""
Exploratory Data Analysis for the Nigerian Inflation Predictor.

Produces summary statistics and time series plots for all four variables.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_cleaned_data():
    """Load the cleaned dataset."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def summary_statistics(df):
    """Generate and save summary statistics."""
    stats = df.describe().T
    stats["skewness"] = df.skew()
    stats["kurtosis"] = df.kurtosis()

    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(stats.round(2).to_string())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stats.round(4).to_csv(os.path.join(RESULTS_DIR, "summary_statistics.csv"))
    return stats


def plot_time_series(df):
    """Plot individual and panel time series for each variable."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    variables = {
        "mpr": ("Monetary Policy Rate (MPR)", "Percent (%)", "#1f77b4"),
        "inflation": ("Headline Inflation Rate", "Percent (%, YoY)", "#d62728"),
        "exchange_rate": ("Exchange Rate (₦/USD)", "Naira per USD", "#2ca02c"),
        "m2": ("Broad Money Supply (M2)", "₦ Billions", "#9467bd"),
    }

    # Individual plots
    for col, (title, ylabel, color) in variables.items():
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df[col], color=color, linewidth=1.2)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"ts_{col}.png"), dpi=150)
        plt.close(fig)
        print(f"Saved: ts_{col}.png")

    # Panel plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    for ax, (col, (title, ylabel, color)) in zip(axes, variables.items()):
        ax.plot(df.index, df[col], color=color, linewidth=1.2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date", fontsize=12)
    fig.suptitle("Nigerian Macroeconomic Variables", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ts_panel_all_variables.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: ts_panel_all_variables.png")


def plot_correlation_matrix(df):
    """Plot and save the correlation matrix."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    corr = df.corr()

    print("\nCorrelation Matrix:")
    print(corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    plt.colorbar(im)
    ax.set_title("Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "correlation_matrix.png"), dpi=150)
    plt.close(fig)
    print("Saved: correlation_matrix.png")


if __name__ == "__main__":
    df = load_cleaned_data()
    summary_statistics(df)
    plot_time_series(df)
    plot_correlation_matrix(df)
    print("\nEDA complete.")
```

### Step 4: Run both scripts

```bash
python -m data_processing.clean
python -m data_processing.eda
```

### Step 5: Commit

```bash
git add data_processing/clean.py data_processing/eda.py requirements.txt
git commit -m "Day 2: Add data cleaning pipeline and exploratory data analysis"
```

---

## Day 3 — PostgreSQL Schema & Database Loader

**Objective:** Design the database schema and load cleaned data into PostgreSQL.

### Step 1: Install database dependencies

**Update `requirements.txt`:**
```
pandas==2.1.4
matplotlib==3.8.2
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
```

```bash
pip install psycopg2-binary==2.9.9 sqlalchemy==2.0.23
```

### Step 2: Set up PostgreSQL

```bash
# Create the database (run once)
psql -U postgres -c "CREATE DATABASE nigerian_inflation;"
```

### Step 3: Create the database setup script

**File: `data_ingestion/db_setup.py`**

```python
"""
PostgreSQL database setup and data loading.

Creates the schema and loads cleaned macro data into the database.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": os.environ.get("DB_PORT", "5432"),
    "database": os.environ.get("DB_NAME", "nigerian_inflation"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "postgres"),
}

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def get_engine():
    """Create SQLAlchemy engine."""
    url = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(url)


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS macro_monthly (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL UNIQUE,
    mpr             DOUBLE PRECISION NOT NULL,
    inflation       DOUBLE PRECISION NOT NULL,
    exchange_rate   DOUBLE PRECISION NOT NULL,
    m2              DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_macro_monthly_date ON macro_monthly(date);

CREATE TABLE IF NOT EXISTS model_results (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(50) NOT NULL,
    result_type     VARCHAR(50) NOT NULL,
    parameters      JSONB,
    result_data     JSONB NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stationarity_tests (
    id              SERIAL PRIMARY KEY,
    variable        VARCHAR(30) NOT NULL,
    test_name       VARCHAR(10) NOT NULL,
    test_statistic  DOUBLE PRECISION NOT NULL,
    p_value         DOUBLE PRECISION,
    critical_values JSONB,
    is_stationary   BOOLEAN NOT NULL,
    differencing    INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def create_tables(engine):
    """Create all required tables."""
    with engine.connect() as conn:
        conn.execute(text(CREATE_TABLES_SQL))
        conn.commit()
    print("Tables created successfully.")


def load_data_to_db(engine):
    """Load cleaned CSV data into macro_monthly table."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True).reset_index()

    with engine.connect() as conn:
        conn.execute(text("DELETE FROM macro_monthly"))
        conn.commit()

    df.to_sql("macro_monthly", engine, if_exists="append", index=False)
    print(f"Loaded {len(df)} rows into macro_monthly.")


def verify_load(engine):
    """Verify data was loaded correctly."""
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM macro_monthly")).scalar()
        row = conn.execute(text("SELECT MIN(date), MAX(date) FROM macro_monthly")).fetchone()
    print(f"Verification: {count} rows, {row[0]} to {row[1]}")


if __name__ == "__main__":
    engine = get_engine()
    create_tables(engine)
    load_data_to_db(engine)
    verify_load(engine)
```

### Step 4: Run

```bash
python -m data_ingestion.db_setup
```

### Step 5: Commit

```bash
git add data_ingestion/db_setup.py requirements.txt
git commit -m "Day 3: Add PostgreSQL schema and data loader"
```

---

## Day 4 — Stationarity Testing (ADF & KPSS)

**Objective:** Test all four variables for unit roots to determine integration order.

### Step 1: Install statsmodels and numpy

**Update `requirements.txt`:**
```
pandas==2.1.4
matplotlib==3.8.2
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
numpy==1.26.2
statsmodels==0.14.1
```

```bash
pip install numpy==1.26.2 statsmodels==0.14.1
```

### Step 2: Create the stationarity testing script

**File: `data_processing/stationarity.py`**

```python
"""
Stationarity testing for the Nigerian Inflation Predictor.

Performs ADF and KPSS tests on all four variables in levels and
first differences to determine integration order.

Key concepts:
- ADF: H0 = unit root (non-stationary). Reject H0 (p < 0.05) = stationary.
- KPSS: H0 = stationary. Reject H0 (p < 0.05) = non-stationary.
- I(0) = stationary in levels. I(1) = stationary after first differencing.
"""

import os
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_cleaned_data():
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    return pd.read_csv(filepath, index_col="date", parse_dates=True)


def run_adf_test(series, variable_name, significance=0.05):
    """Run ADF test. H0: unit root (non-stationary)."""
    result = adfuller(series.dropna(), autolag="AIC")
    test_stat, p_value, used_lag, nobs = result[0], result[1], result[2], result[3]
    critical_values = result[4]
    is_stationary = p_value < significance

    return {
        "variable": variable_name, "test": "ADF",
        "test_statistic": round(test_stat, 4),
        "p_value": round(p_value, 4),
        "lags_used": used_lag, "n_obs": nobs,
        "critical_values": {k: round(v, 4) for k, v in critical_values.items()},
        "is_stationary": is_stationary,
        "conclusion": "Stationary (reject H0)" if is_stationary
                      else "Non-stationary (fail to reject H0)",
    }


def run_kpss_test(series, variable_name, significance=0.05):
    """Run KPSS test. H0: stationary."""
    result = kpss(series.dropna(), regression="c", nlags="auto")
    test_stat, p_value, used_lag = result[0], result[1], result[2]
    critical_values = result[3]
    is_stationary = p_value > significance

    return {
        "variable": variable_name, "test": "KPSS",
        "test_statistic": round(test_stat, 4),
        "p_value": round(p_value, 4),
        "lags_used": used_lag,
        "critical_values": {k: round(v, 4) for k, v in critical_values.items()},
        "is_stationary": is_stationary,
        "conclusion": "Stationary (fail to reject H0)" if is_stationary
                      else "Non-stationary (reject H0)",
    }


def determine_integration_order(levels_adf, levels_kpss, diff_adf, diff_kpss):
    """Determine I(d) based on both tests."""
    if levels_adf["is_stationary"] and levels_kpss["is_stationary"]:
        return 0
    elif diff_adf["is_stationary"] and diff_kpss["is_stationary"]:
        return 1
    else:
        return 2


def run_all_tests(df):
    """Run ADF and KPSS on all variables in levels and first differences."""
    variables = ["mpr", "inflation", "exchange_rate", "m2"]
    all_results = []
    integration_orders = {}

    print("=" * 80)
    print("STATIONARITY TEST RESULTS")
    print("=" * 80)

    for var in variables:
        print(f"\n{'─' * 60}")
        print(f"Variable: {var.upper()}")
        print(f"{'─' * 60}")

        series = df[var]
        diff_series = df[var].diff().dropna()

        adf_levels = run_adf_test(series, f"{var}_levels")
        kpss_levels = run_kpss_test(series, f"{var}_levels")

        print(f"\n  LEVELS:")
        print(f"    ADF:  stat={adf_levels['test_statistic']}, "
              f"p={adf_levels['p_value']} -> {adf_levels['conclusion']}")
        print(f"    KPSS: stat={kpss_levels['test_statistic']}, "
              f"p={kpss_levels['p_value']} -> {kpss_levels['conclusion']}")

        adf_diff = run_adf_test(diff_series, f"{var}_diff")
        kpss_diff = run_kpss_test(diff_series, f"{var}_diff")

        print(f"\n  FIRST DIFFERENCE:")
        print(f"    ADF:  stat={adf_diff['test_statistic']}, "
              f"p={adf_diff['p_value']} -> {adf_diff['conclusion']}")
        print(f"    KPSS: stat={kpss_diff['test_statistic']}, "
              f"p={kpss_diff['p_value']} -> {kpss_diff['conclusion']}")

        order = determine_integration_order(adf_levels, kpss_levels, adf_diff, kpss_diff)
        integration_orders[var] = order
        print(f"\n  -> Integration order: I({order})")

        all_results.extend([adf_levels, kpss_levels, adf_diff, kpss_diff])

    print(f"\n{'=' * 80}")
    print("INTEGRATION ORDER SUMMARY")
    print(f"{'=' * 80}")
    for var, order in integration_orders.items():
        print(f"  {var:20s} -> I({order})")

    return all_results, integration_orders


def save_results(all_results, integration_orders):
    """Save results to JSON and CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, "stationarity_tests.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    summary = [{"variable": v, "integration_order": o} for v, o in integration_orders.items()]
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS_DIR, "integration_orders.csv"), index=False)
    print("Results saved.")


if __name__ == "__main__":
    df = load_cleaned_data()
    all_results, integration_orders = run_all_tests(df)
    save_results(all_results, integration_orders)
```

### Step 3: Run

```bash
python -m data_processing.stationarity
```

### Step 4: Commit

```bash
git add data_processing/stationarity.py requirements.txt
git commit -m "Day 4: Add ADF and KPSS stationarity tests"
```

---

## Day 5 — Week 1 Pipeline Validation & Review

**Objective:** Run the entire pipeline end-to-end. Verify all outputs. Document what you learned.

### Step 1: Create the pipeline runner

**File: `data_processing/run_pipeline.py`**

```python
"""
Week 1 end-to-end pipeline: Ingestion -> Cleaning -> EDA -> Stationarity.
"""

import os
import sys


def main():
    print("=" * 70)
    print("NIGERIAN INFLATION PREDICTOR — WEEK 1 PIPELINE")
    print("=" * 70)

    # Step 1: Ingest
    print("\n[1/4] DATA INGESTION")
    from data_ingestion.ingest import load_raw_data
    raw_df = load_raw_data()

    # Step 2: Clean
    print("\n[2/4] DATA CLEANING")
    from data_processing.clean import clean_data, save_cleaned_data
    clean_df = clean_data(raw_df)
    save_cleaned_data(clean_df)

    # Step 3: EDA
    print("\n[3/4] EXPLORATORY DATA ANALYSIS")
    from data_processing.eda import summary_statistics, plot_time_series, plot_correlation_matrix
    summary_statistics(clean_df)
    plot_time_series(clean_df)
    plot_correlation_matrix(clean_df)

    # Step 4: Stationarity
    print("\n[4/4] STATIONARITY TESTING")
    from data_processing.stationarity import run_all_tests, save_results
    all_results, integration_orders = run_all_tests(clean_df)
    save_results(all_results, integration_orders)

    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    expected = [
        "data/processed/cleaned_data.csv",
        "results/summary_statistics.csv",
        "results/ts_mpr.png",
        "results/ts_inflation.png",
        "results/ts_exchange_rate.png",
        "results/ts_m2.png",
        "results/ts_panel_all_variables.png",
        "results/correlation_matrix.png",
        "results/stationarity_tests.json",
        "results/integration_orders.csv",
    ]
    base = os.path.dirname(os.path.dirname(__file__))
    all_ok = True
    for f in expected:
        exists = os.path.exists(os.path.join(base, f))
        print(f"  [{'OK' if exists else 'MISSING'}] {f}")
        if not exists:
            all_ok = False

    if all_ok:
        print("\nAll outputs validated. Week 1 complete.")
    else:
        print("\nSome outputs missing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### Step 2: Run it

```bash
python -m data_processing.run_pipeline
```

All 10 files should show `[OK]`.

### Step 3: Commit

```bash
git add data_processing/run_pipeline.py
git commit -m "Day 5: Add pipeline runner, validate Week 1 outputs"
```

### What You Know After Week 1

- How Nigerian macro data is structured
- How each variable behaves over time (from your plots)
- The integration order of each variable (from ADF/KPSS)
- Why this matters: I(1) variables need differencing or cointegration for valid inference
