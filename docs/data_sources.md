# Data Sources — Nigerian Inflation Predictor

## Overview

This document specifies the exact variables, sources, frequency, units, and expected formats for all data used in the Nigerian Inflation Predictor. All data pertains to Nigeria and is sourced from official statistical agencies.

The system uses **four macroeconomic variables** observed at **monthly frequency**. No additional variables, proxies, or derived indicators are included.

---

## 1. Monetary Policy Rate (MPR)

| Field | Value |
|-------|-------|
| **Variable name** | `mpr` |
| **Description** | The benchmark interest rate set by the Central Bank of Nigeria's Monetary Policy Committee (MPC). It is the rate at which the CBN lends to commercial banks and serves as the primary instrument of monetary policy. |
| **Unit** | Percent per annum (%) |
| **Frequency** | Monthly (constant within months between MPC meetings; stepped to reflect meeting decisions) |
| **Source** | Central Bank of Nigeria (CBN) — Monetary Policy Decisions / CBN Statistical Bulletin |
| **URL** | https://www.cbn.gov.ng/rates/mnymktind.asp |
| **Notes** | The MPR is set at MPC meetings held approximately every two months. Between meetings, the rate is constant. For monthly analysis, the prevailing rate at month-end is used. |

---

## 2. Headline Inflation Rate (INF)

| Field | Value |
|-------|-------|
| **Variable name** | `inflation` |
| **Description** | Year-on-year percentage change in the Consumer Price Index (CPI), measuring the rate of increase in the general price level of goods and services consumed by households in Nigeria. |
| **Unit** | Percent, year-on-year (%) |
| **Frequency** | Monthly |
| **Source** | National Bureau of Statistics (NBS), also published in the CBN Statistical Bulletin |
| **URL** | https://nigerianstat.gov.ng/ (CPI and Inflation Report) |
| **Notes** | Headline inflation (all items) is used rather than core inflation, as it is the primary target referenced by the CBN in its policy communications. The NBS publishes the CPI report on or around the 15th of each month for the preceding month. |

---

## 3. Exchange Rate (EXR)

| Field | Value |
|-------|-------|
| **Variable name** | `exchange_rate` |
| **Description** | The Naira-to-US Dollar exchange rate. This may be the official CBN rate, the Investors' and Exporters' (I&E) window rate, or the Bureau de Change (BDC) rate, depending on the period and data availability. |
| **Unit** | Nigerian Naira per 1 US Dollar (₦/$) |
| **Frequency** | Monthly (end-of-month or monthly average) |
| **Source** | Central Bank of Nigeria (CBN) — Exchange Rate Statistics |
| **URL** | https://www.cbn.gov.ng/rates/ExchRateByCurrency.asp |
| **Notes** | Nigeria operates a managed float exchange rate regime with multiple windows. The choice of rate series must be documented. For consistency, a single rate series should be used throughout the sample period. If a structural break (e.g., the June 2023 unification) is present, it must be acknowledged in the analysis. |

---

## 4. Broad Money Supply (M2)

| Field | Value |
|-------|-------|
| **Variable name** | `m2` |
| **Description** | Broad money supply, defined as the sum of narrow money (M1 = currency in circulation + demand deposits) and quasi-money (savings deposits, time deposits, foreign currency deposits). M2 captures the total liquidity in the Nigerian economy. |
| **Unit** | Nigerian Naira, billions (₦ bn) |
| **Frequency** | Monthly |
| **Source** | Central Bank of Nigeria (CBN) — Money and Credit Statistics / CBN Statistical Bulletin |
| **URL** | https://www.cbn.gov.ng/documents/Statbulletin.asp |
| **Notes** | M2 is a stock variable measured at end-of-month. For econometric modeling, the natural logarithm of M2 (ln_m2) is typically used to linearise the series and express changes in percentage terms. This transformation is applied during data processing, not at ingestion. |

---

## Sample Period

The target sample period is **January 2000 to the most recent available month** (subject to data availability across all four variables). This provides:

- Sufficient observations for asymptotic properties of ARDL and VAR estimation (minimum 200+ observations)
- Coverage of multiple monetary policy regimes and structural events in Nigeria (banking consolidation 2004–2005, global financial crisis 2008–2009, oil price collapse 2014–2016, COVID-19 2020, exchange rate unification 2023)

---

## Data Format Requirements

All raw data files must be placed in `data/raw/` and conform to the following:

| Requirement | Specification |
|-------------|---------------|
| **File format** | CSV (comma-separated values) |
| **Encoding** | UTF-8 |
| **Date column** | Named `date`, formatted as `YYYY-MM-DD` (first day of the month) |
| **Variable columns** | Named exactly as specified above: `mpr`, `inflation`, `exchange_rate`, `m2` |
| **Missing values** | Empty cell or `NaN` — never zero, never a placeholder string |
| **Header row** | Required (first row) |

---

## Data Quality Considerations

1. **MPR step function**: The MPR does not change every month. Repeated values are expected and correct — they are not missing data.
2. **Exchange rate regime changes**: Nigeria has undergone multiple exchange rate policy shifts. The series may exhibit structural breaks that must be addressed in the econometric analysis.
3. **M2 revisions**: The CBN occasionally revises monetary aggregate figures. The most recent vintage of data should be used.
4. **Inflation rebasing**: The NBS has rebased the CPI multiple times (most recently in 2009). Year-on-year rates are less affected by rebasing than index levels, which is one reason for using the rate rather than the index.
