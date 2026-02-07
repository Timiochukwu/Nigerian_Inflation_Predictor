# Week 1, Day 3 — Data Ingestion: Loading & Validating the Dataset

## What You'll Learn Today

- How to create a CSV dataset with realistic Nigerian macro data
- How to write a reusable Python script that loads and validates data
- What "schema validation" means and why you need it
- How to run Python modules from the command line

## Why This Matters

The ingestion script is the front door of your entire system. Every model, every plot, every result starts here. If bad data gets through this door, everything downstream is wrong. That's why we validate the data on the way in — checking column names, data types, and date formats.

## Why This Matters for Nigeria

Nigerian macro data from the CBN Statistical Bulletin sometimes has formatting issues — dates in different formats, missing months, values stored as text instead of numbers. Your ingestion script must handle these robustly.

---

## Step 1: No New Packages

pandas (installed yesterday) is all you need for data ingestion.

---

## Step 2: Create the Dataset

This CSV file contains monthly Nigerian macro data from January 2000 to December 2024. The values are based on publicly known trends in MPR, inflation, exchange rate, and M2.

**For your thesis:** Replace this with actual CBN/NBS data from the sources listed in `docs/data_sources.md`. The structure will be identical.

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

**Notice key Nigerian economic events visible in this data:**
- **2006-2007**: Low inflation (~3-8%) during oil boom and Soludo-era reforms
- **2008-2009**: Global financial crisis — exchange rate jumps from 117 to 150
- **2011**: Emergency MPR hike to 12% (Sanusi era) to defend the Naira
- **2016**: Exchange rate jumps from 199 to 310 after CBN floats the Naira
- **2020-2021**: COVID — CBN cuts MPR, inflation rises
- **2023 June**: Exchange rate unification — Naira drops from 462 to 750 overnight
- **2024**: Aggressive tightening — MPR raised from 18.75% to 27.50%

---

## Step 3: Create the Ingestion Script

This script is the **official entry point** for all data into the system.

**File: `data_ingestion/ingest.py`**

```python
"""
Data ingestion module for the Nigerian Inflation Predictor.

This script loads raw CSV data from data/raw/, validates that it has
the correct columns and data types, and returns a pandas DataFrame
ready for cleaning and analysis.

Usage:
    python -m data_ingestion.ingest
"""

import os
import pandas as pd


# These are the EXACT column names your CSV must have.
# If any are missing, the script will raise an error.
REQUIRED_COLUMNS = ["date", "mpr", "inflation", "exchange_rate", "m2"]

# Where raw data files live (relative to this script's location)
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Default filename — change this if your CSV has a different name
DEFAULT_FILE = "nigeria_macro_data.csv"


def load_raw_data(filename=None):
    """
    Load raw macroeconomic data from a CSV file.

    What this function does, step by step:
    1. Finds the CSV file in data/raw/
    2. Reads it into a pandas DataFrame
    3. Checks that all required columns exist
    4. Converts the date column to proper datetime format
    5. Converts numeric columns to float (handling any text errors)
    6. Sorts by date and sets date as the index
    7. Reports basic info (number of rows, date range, missing values)

    Parameters
    ----------
    filename : str, optional
        Name of the CSV file. If not provided, uses 'nigeria_macro_data.csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with date index and four numeric columns:
        mpr, inflation, exchange_rate, m2

    Raises
    ------
    FileNotFoundError
        If the CSV file doesn't exist.
    ValueError
        If required columns are missing.
    """

    # Use default filename if none provided
    if filename is None:
        filename = DEFAULT_FILE

    # Build the full file path
    # os.path.join() handles different OS path separators (/ vs \)
    filepath = os.path.join(RAW_DATA_DIR, filename)

    # Check the file exists before trying to read it
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Make sure you have placed your CSV in the data/raw/ folder."
        )

    # Read the CSV file into a DataFrame
    # This is one line of code, but it does a lot:
    # - Opens the file
    # - Parses the comma-separated values
    # - Creates column headers from the first row
    # - Puts all data into a table
    df = pd.read_csv(filepath)

    # VALIDATION 1: Check that all required columns exist
    # set() turns a list into a collection that supports subtraction
    # If we subtract df.columns from REQUIRED_COLUMNS, anything left is missing
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Your CSV has these columns: {list(df.columns)}\n"
            f"Required columns are: {REQUIRED_COLUMNS}"
        )

    # VALIDATION 2: Convert 'date' column to datetime
    # pd.to_datetime() understands many date formats automatically
    # format="%Y-%m-%d" means we expect "2024-01-01" format
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # VALIDATION 3: Convert numeric columns to float
    # errors="coerce" means: if a value can't be converted (e.g., "N/A"),
    # replace it with NaN instead of crashing
    numeric_cols = ["mpr", "inflation", "exchange_rate", "m2"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by date (oldest first) and reset the row numbers
    df = df.sort_values("date").reset_index(drop=True)

    # Make 'date' the index (row labels) instead of a regular column
    # This is standard for time series data
    df = df.set_index("date")

    # Print a summary so you can verify the data looks right
    print(f"Loaded {len(df)} observations")
    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"\nMissing values per column:")
    print(df.isnull().sum().to_string())

    return df


# This block runs ONLY when you execute this file directly
# (not when another script imports from it)
if __name__ == "__main__":
    print("=" * 60)
    print("NIGERIAN INFLATION PREDICTOR — DATA INGESTION")
    print("=" * 60)

    df = load_raw_data()

    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nLast 5 rows:")
    print(df.tail())

    print(f"\nData types:")
    print(df.dtypes)

    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
```

---

## Step 4: Run the Ingestion Script

```bash
python -m data_ingestion.ingest
```

**Expected output:**
```
============================================================
NIGERIAN INFLATION PREDICTOR — DATA INGESTION
============================================================
Loaded 300 observations
Date range: 2000-01-01 to 2024-12-01

Missing values per column:
mpr              0
inflation        0
exchange_rate    0
m2               0

First 5 rows:
             mpr  inflation  exchange_rate      m2
date
2000-01-01  13.5       6.62          92.34  1070.5
2000-02-01  13.5       6.93          92.55  1078.2
2000-03-01  13.5       7.87          92.69  1090.6
2000-04-01  13.5       8.13          93.05  1095.3
2000-05-01  13.5       8.68          95.10  1100.7
...
Shape: 300 rows, 4 columns
```

If you see this, your ingestion pipeline is working.

---

## Step 5: Commit

```bash
git add data/raw/nigeria_macro_data.csv data_ingestion/ingest.py
git commit -m "Day 3: Add dataset and data ingestion script with validation"
```

---

## Common Errors

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: Data file not found` | Make sure `nigeria_macro_data.csv` is in `data/raw/`, not in the project root |
| `ValueError: Missing required columns` | Check your CSV header row. Column names must be exactly: `date,mpr,inflation,exchange_rate,m2` |
| `ParserError` or garbled output | Open your CSV in a text editor and check it's comma-separated, not semicolons or tabs |
| Output shows NaN values | Check the CSV — some values might have text or special characters |

---

## Check Your Understanding

1. **"Why validate columns instead of just reading the file?"**
   > If someone gives you a CSV with different column names (e.g., "MPR" instead of "mpr"), the script will crash later with a confusing error. Validating upfront gives a clear error message at the point of failure.

2. **"What does `errors='coerce'` do in `pd.to_numeric()`?"**
   > It replaces values that can't be converted to numbers with NaN (missing). Without it, a single bad value like "N/A" in your CSV would crash the entire script.

3. **"Why set date as the index?"**
   > In time series analysis, the date uniquely identifies each observation. Setting it as the index lets you select by date (e.g., `df.loc["2024-01-01"]`) and ensures the data is always in chronological order.

---

## What You Built Today

| Item | File | Purpose |
|------|------|---------|
| Dataset | `data/raw/nigeria_macro_data.csv` | 300 months of Nigerian macro data |
| Ingestion script | `data_ingestion/ingest.py` | Loads, validates, and returns clean DataFrame |

**Packages installed today:** None (using pandas from Day 2)

**Tomorrow (Day 4):** You'll build the data cleaning pipeline — handling missing values, enforcing monthly frequency, and validating value ranges.
