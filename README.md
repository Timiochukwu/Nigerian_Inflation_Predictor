# Nigerian Inflation Predictor

## Overview

This project implements a research-grade inflation prediction system for Nigeria. It models the transmission of monetary policy shocks to consumer price inflation using established time-series econometric methods, producing interpretable and policy-relevant outputs.

The system answers a central question in Nigerian macroeconomic policy:

> **How do changes in the Central Bank of Nigeria's Monetary Policy Rate (MPR) transmit through the exchange rate and money supply channels to affect headline inflation, and over what horizon?**

## Analytical Framework

The predictor is built on four macroeconomic variables observed at monthly frequency:

| Variable | Description | Source |
|----------|-------------|--------|
| **MPR** | Monetary Policy Rate (%) | Central Bank of Nigeria (CBN) |
| **INF** | Headline Inflation Rate (% year-on-year) | National Bureau of Statistics (NBS) / CBN |
| **EXR** | Naira/USD Exchange Rate (official or BDC) | CBN |
| **M2** | Broad Money Supply (₦ billions) | CBN Statistical Bulletin |

## Econometric Methods

The following models are estimated and analysed:

1. **Autoregressive Distributed Lag (ARDL)** — Bounds testing for cointegration; short-run and long-run coefficient estimation.
2. **Vector Autoregression (VAR)** — Reduced-form dynamics with Cholesky identification.
3. **Impulse Response Functions (IRFs)** — Tracing the dynamic effect of a +100 basis point MPR shock on inflation over a 12–24 month horizon.
4. **Forecast Error Variance Decomposition (FEVD)** — Quantifying the relative contribution of each variable to inflation forecast uncertainty.
5. **Policy Shock Simulation** — Simulating the inflationary impact of a contractionary monetary policy shock (+100bps to MPR).

### VAR Identification Ordering

The Cholesky decomposition imposes a recursive causal ordering:

```
MPR → Exchange Rate → Money Supply → Inflation
```

This ordering reflects the assumption that the CBN's policy rate is the most exogenous variable in the system — set by the Monetary Policy Committee (MPC) based on forward-looking assessments — while inflation is the most endogenous, responding contemporaneously to all other shocks. The exchange rate responds to policy signals before the money supply adjusts, consistent with the asset-pricing view of exchange rate determination and the lagged nature of monetary aggregates.

## Project Structure

```
Nigerian_Inflation_Predictor/
├── data/
│   ├── raw/                  # Original datasets as obtained from sources
│   └── processed/            # Cleaned, merged, analysis-ready datasets
├── data_ingestion/           # Scripts for loading raw data
├── data_processing/          # Cleaning, transformation, stationarity testing
├── econometric_models/       # ARDL, VAR, IRF, FEVD estimation
├── simulation/               # Policy shock simulation engine
├── api/                      # Java 17 / Spring Boot REST API layer
├── results/                  # Generated tables, plots, model outputs
├── docs/                     # Methodology, data sources, interpretation
├── requirements.txt          # Python dependencies (added incrementally)
├── .gitignore
└── README.md
```

## Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Econometric Modeling** | Python 3, pandas, numpy, statsmodels | statsmodels provides validated implementations of ARDL, VAR, IRF, and FEVD. pandas and numpy are the standard data manipulation stack. Python is the dominant language for applied econometrics and reproducible research. |
| **API** | Java 17, Spring Boot | Spring Boot provides a production-grade, strongly-typed REST API layer with mature tooling for deployment. Java enforces interface contracts and is widely used in financial systems engineering. The API serves model results — it does not perform econometric estimation. |
| **Database** | PostgreSQL | PostgreSQL is the standard open-source relational database for structured time-series storage, with strong support for numeric precision and ACID compliance. |

### Why Not Alternatives?

- **R / EViews**: Python offers superior integration with API layers and deployment pipelines. statsmodels covers all required econometric methods.
- **Flask / FastAPI (Python API)**: Spring Boot is chosen to demonstrate financial-systems-grade engineering. A Python-only stack would be simpler but would not showcase the architectural separation expected in production analytics systems.
- **MongoDB / NoSQL**: The data is structured, tabular, and relational. A document store would introduce unnecessary complexity with no analytical benefit.

## Target Audience

This system is designed to be defensible before:

- MSc thesis examination panels (economics, finance, or financial engineering)
- Quantitative finance and risk analytics interviewers
- Central bank policy research teams

## License

This project is developed for academic and research purposes.
