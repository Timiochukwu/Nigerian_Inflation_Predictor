# Day 36 — Methodology Documentation

## What You'll Learn Today
Today you'll write the **methodology chapter** for your thesis or research report. This is where you explain:
- What data you used and why
- How you measured each variable
- What econometric techniques you applied
- Why those techniques are appropriate for your research question

Think of methodology as your "instruction manual" — someone reading it should be able to replicate your entire analysis.

**By the end of today:** You'll have a complete `docs/methodology.md` file that you can adapt for your thesis, journal article, or policy brief.

---

## Step 1: Create the Methodology Document

Navigate to your project folder:
```bash
cd ~/Nigerian_Inflation_Predictor
```

Create the `docs` folder if it doesn't exist yet:
```bash
mkdir -p docs
```

Now create the methodology file:
```bash
touch docs/methodology.md
```

Open it in your text editor:
```bash
nano docs/methodology.md
```

**Delete everything in `docs/methodology.md` and replace it with this:**

```markdown
# Methodology

## 3.1 Research Design

This study employs a **quantitative, time-series econometric approach** to investigate the transmission of monetary policy shocks to inflation in Nigeria. The research design follows three stages:

1. **Stationarity and Cointegration Analysis**: Testing the order of integration of key variables and examining long-run equilibrium relationships
2. **Long-Run and Short-Run Dynamics**: Estimating an ARDL (Autoregressive Distributed Lag) model to separate long-run effects from short-run adjustments
3. **Dynamic Policy Simulation**: Using a Vector Autoregression (VAR) model with impulse response functions to trace out the temporal path of policy transmission

The methodology is guided by the theoretical framework of monetary transmission mechanisms, specifically the **interest rate channel** and **exchange rate channel**, which are dominant in Nigeria's economy.

---

## 3.2 Data Sources and Description

### 3.2.1 Data Source
All data used in this study were obtained from the **Central Bank of Nigeria (CBN) Statistical Database**, accessible at [https://www.cbn.gov.ng/rates/](https://www.cbn.gov.ng/rates/). The CBN publishes monthly time-series data on monetary policy instruments, interest rates, exchange rates, and inflation.

### 3.2.2 Sample Period
The analysis covers the period from **January 2010 to December 2024**, providing **180 monthly observations**. This period captures:
- The 2015-2017 oil price shock and currency crisis
- The 2020 COVID-19 pandemic and associated policy responses
- The 2023 currency redesign policy
- Multiple monetary policy regime shifts

This 15-year window is sufficient for reliable time-series estimation while being recent enough to reflect current policy transmission dynamics.

### 3.2.3 Variables
The dataset contains **34 variables** covering monetary policy instruments, interest rates, exchange rates, inflation indices, and financial market indicators. For this study, we focus on **four core variables** that represent the key nodes in the monetary transmission mechanism:

| Variable | Definition | Unit | Role |
|----------|-----------|------|------|
| **MPR** | Monetary Policy Rate | % per annum | Policy instrument |
| **TBR** | 91-day Treasury Bill Rate | % per annum | Intermediate target (interest rate channel) |
| **EXO** | Official Exchange Rate (NGN/USD) | Naira per US Dollar | Intermediate target (exchange rate channel) |
| **INFL** | Headline Inflation Rate | % year-on-year | Target variable |

The full list of 34 variables is documented in Appendix A.

---

## 3.3 Variable Definitions and Economic Justification

### 3.3.1 Monetary Policy Rate (MPR)
The **Monetary Policy Rate** is the Central Bank of Nigeria's primary policy instrument. It represents the interest rate at which the CBN lends to commercial banks and signals the stance of monetary policy (tight or loose). Changes in the MPR are expected to transmit through the financial system to influence inflation.

**Economic Justification**: In modern central banking, the policy rate is the anchor for all other interest rates in the economy. The CBN's Monetary Policy Committee (MPC) adjusts the MPR in response to inflation deviations from target.

### 3.3.2 Treasury Bill Rate (TBR)
The **91-day Treasury Bill Rate** is the yield on short-term government securities. It reflects the opportunity cost of holding money and is highly responsive to MPR changes. We use TBR as a proxy for the **money market interest rate** — the rate at which banks lend to each other and to creditworthy customers.

**Economic Justification**: In Nigeria, the interest rate channel is more reliable than the monetary quantity channel (M2) because:
- The CBN targets interest rates, not money supply quantities
- The relationship between M2 and inflation is unstable due to financial innovation and dollarization
- TBR directly affects investment and consumption decisions

TBR serves as the **first-stage transmission variable** from MPR to the real economy.

### 3.3.3 Official Exchange Rate (EXO)
The **Official Exchange Rate** (Naira per US Dollar) measures the external value of the Naira. Given Nigeria's import-dependent economy, exchange rate depreciation raises the domestic price of imported goods, directly affecting inflation.

**Economic Justification**: Nigeria imports over 70% of consumer goods, petroleum products (refined), and industrial inputs. A depreciation of the Naira increases the Naira-equivalent cost of imports, which feeds into the Consumer Price Index. The exchange rate channel is particularly strong in Nigeria.

EXO serves as the **second-stage transmission variable** — it is influenced by interest rates (through capital flows and CBN FX interventions) and directly affects inflation.

### 3.3.4 Headline Inflation Rate (INFL)
**Headline Inflation** is the year-on-year percentage change in the Consumer Price Index (CPI), measuring the average price increase for a basket of goods and services consumed by Nigerian households.

**Economic Justification**: Headline inflation is the CBN's primary target variable. Unlike core inflation (which excludes food and energy), headline inflation reflects the actual cost of living faced by citizens and is the focus of monetary policy decisions.

---

## 3.4 Theoretical Framework: The Monetary Transmission Mechanism

Monetary policy affects inflation through multiple channels. In Nigeria, the dominant channels are:

### 3.4.1 Interest Rate Channel
```
MPR ↑ → TBR ↑ → Borrowing costs ↑ → Investment & Consumption ↓ → Aggregate Demand ↓ → INFL ↓
```

An increase in the MPR raises the cost of borrowing throughout the economy. Higher interest rates discourage investment and consumption, reducing aggregate demand and putting downward pressure on prices.

### 3.4.2 Exchange Rate Channel
```
MPR ↑ → TBR ↑ → Capital inflows ↑ → Demand for Naira ↑ → EXO ↓ (Naira appreciates) → Import prices ↓ → INFL ↓
```

Higher interest rates attract foreign capital inflows, increasing demand for Naira and causing appreciation (a fall in EXO). A stronger Naira lowers the cost of imports, reducing inflation.

### 3.4.3 Integrated Transmission Path
Combining both channels, we have:
```
MPR → TBR → EXO → INFL
```

This **ordering** reflects the temporal sequence of transmission:
1. **Stage 1**: MPR changes immediately affect TBR (within days)
2. **Stage 2**: TBR changes influence EXO (within weeks to months, through capital flows and CBN FX interventions)
3. **Stage 3**: EXO changes affect INFL (within months, as import prices adjust)

This ordering is crucial for our VAR identification strategy (Section 3.6.3).

---

## 3.5 Econometric Methodology

### 3.5.1 Unit Root Testing
Time-series regressions with non-stationary data can produce **spurious results**. We test for stationarity using two complementary tests:

#### Augmented Dickey-Fuller (ADF) Test
- **Null hypothesis**: The series has a unit root (non-stationary)
- **Alternative hypothesis**: The series is stationary
- **Test statistic**: More negative values favor stationarity
- **Decision rule**: Reject H₀ if p-value < 0.05

The ADF test is powerful against I(1) alternatives but may over-reject when the true process is near-stationary.

#### KPSS Test
- **Null hypothesis**: The series is stationary
- **Alternative hypothesis**: The series has a unit root
- **Test statistic**: Larger values indicate non-stationarity
- **Decision rule**: Reject H₀ if test statistic > critical value

The KPSS test complements ADF by reversing the null hypothesis, reducing the risk of false conclusions.

**Joint Interpretation**:
- Both tests agree on stationarity → Conclude I(0)
- Both tests agree on non-stationarity → Conclude I(1)
- Tests conflict → Examine first differences

### 3.5.2 Cointegration Testing
If variables are I(1) but share a long-run equilibrium relationship, they are **cointegrated**. We apply two cointegration tests:

#### Engle-Granger Two-Step Test
1. Estimate the long-run relationship: INFL = β₀ + β₁·MPR + β₂·TBR + β₃·EXO + u
2. Extract residuals (û) and test for stationarity using ADF
3. If residuals are I(0), variables are cointegrated

**Advantage**: Simple and intuitive
**Limitation**: Assumes a single cointegrating vector; results depend on which variable is chosen as dependent

#### Johansen Test
Uses a maximum likelihood approach to test for multiple cointegrating vectors.
- **Trace statistic**: Tests H₀: r ≤ k cointegrating vectors
- **Maximum eigenvalue statistic**: Tests H₀: exactly r cointegrating vectors

**Advantage**: Can detect multiple cointegrating relationships
**Limitation**: Sensitive to lag length and small-sample properties

If cointegration is detected, we proceed to estimate the long-run relationship using ARDL.

---

## 3.6 Model Specifications

### 3.6.1 ARDL (Autoregressive Distributed Lag) Model

The ARDL model is ideal for our context because:
1. It allows variables to be I(0), I(1), or a mix (no need for pre-differencing)
2. It separates **long-run levels** from **short-run dynamics**
3. It handles endogeneity through lagged regressors
4. It works well with small to moderate sample sizes

#### ARDL Specification
```
INFL_t = α + Σ(i=1 to p) φᵢ·INFL_(t-i) + Σ(i=0 to q₁) β₁ᵢ·MPR_(t-i) + Σ(i=0 to q₂) β₂ᵢ·TBR_(t-i) + Σ(i=0 to q₃) β₃ᵢ·EXO_(t-i) + εₜ
```

Where:
- `p` = number of lags of the dependent variable (inflation)
- `q₁, q₂, q₃` = number of lags of each independent variable
- `α` = intercept
- `εₜ` = error term (white noise)

#### Lag Selection
We use the **Akaike Information Criterion (AIC)** to select optimal lag lengths:
- AIC balances model fit against complexity
- Lower AIC indicates a better model
- We test lag combinations from (1,0,0,0) to (4,4,4,4)

The selected lag structure is denoted as **ARDL(p, q₁, q₂, q₃)**.

#### Bounds Testing Procedure (Pesaran, Shin & Smith, 2001)
To test for cointegration in the ARDL framework, we estimate the **Error Correction Model (ECM)** representation:

```
ΔINFL_t = α + Σφᵢ·ΔINFL_(t-i) + Σβ₁ᵢ·ΔMPR_(t-i) + Σβ₂ᵢ·ΔTBR_(t-i) + Σβ₃ᵢ·ΔEXO_(t-i)
          + θ₁·INFL_(t-1) + θ₂·MPR_(t-1) + θ₃·TBR_(t-1) + θ₄·EXO_(t-1) + εₜ
```

The **F-statistic** tests the joint significance of lagged levels (θ₁, θ₂, θ₃, θ₄):
- **H₀**: θ₁ = θ₂ = θ₃ = θ₄ = 0 (no cointegration)
- **H₁**: At least one θᵢ ≠ 0 (cointegration exists)

**Decision rule**:
- F-stat > upper critical value → Reject H₀ (cointegration confirmed)
- F-stat < lower critical value → Fail to reject H₀ (no cointegration)
- F-stat between bounds → Inconclusive

Critical values depend on the number of regressors (k=3) and whether variables are I(0) or I(1).

#### Long-Run Coefficients
Once cointegration is confirmed, we estimate long-run coefficients:
```
β_LR = -θᵢ / θ₁
```

For example, the long-run effect of MPR on inflation is:
```
∂INFL/∂MPR (long-run) = -θ₂ / θ₁
```

#### Error Correction Term
The **error correction coefficient** (θ₁, often denoted ECT) measures the speed of adjustment back to long-run equilibrium:
- ECT should be negative and significant
- |ECT| ∈ (0, 1) indicates stable adjustment
- Interpretation: "X% of disequilibrium is corrected each month"

### 3.6.2 VAR (Vector Autoregression) Model

While ARDL focuses on long-run relationships, the VAR model captures **dynamic interactions** among all variables. Each variable is regressed on its own lags and lags of all other variables.

#### VAR Specification
```
Y_t = A₁·Y_(t-1) + A₂·Y_(t-2) + ... + Aₚ·Y_(t-p) + εₜ
```

Where:
- `Y_t = [MPR_t, TBR_t, EXO_t, INFL_t]'` is a 4×1 vector
- `Aᵢ` are 4×4 coefficient matrices
- `εₜ` is a 4×1 vector of reduced-form errors

**Lag Length Selection**: We use the Akaike Information Criterion (AIC) to choose `p`, testing lags 1 through 12.

#### Identification: Cholesky Decomposition
The reduced-form errors (εₜ) are correlated. To obtain **structural shocks**, we impose a recursive ordering using Cholesky decomposition:

**Ordering**: MPR → TBR → EXO → INFL

This implies:
1. **MPR** is contemporaneously exogenous (determined by CBN policy, not affected by other variables within the same month)
2. **TBR** responds contemporaneously to MPR but not to EXO or INFL
3. **EXO** responds contemporaneously to MPR and TBR but not to INFL
4. **INFL** responds contemporaneously to all variables

This ordering aligns with the transmission mechanism (Section 3.4) and institutional facts (the CBN sets MPR at discrete MPC meetings, while inflation is measured with a lag).

#### Impulse Response Functions (IRFs)
IRFs trace out the effect of a **one-standard-deviation shock** to one variable on all variables over time (up to 24 months ahead). For example:
- **IRF(INFL, MPR shock)** shows how inflation responds to a surprise increase in the policy rate

We report:
- Point estimates (the impulse response path)
- 95% confidence intervals (bootstrapped with 1,000 replications)

**Interpretation**: If the confidence band excludes zero, the response is statistically significant.

---

## 3.7 Diagnostic Testing

To ensure the validity of our estimates, we perform the following diagnostic tests:

### 3.7.1 ARDL Diagnostics
| Test | Purpose | H₀ | Decision Rule |
|------|---------|----|----|
| **Breusch-Godfrey LM Test** | Serial correlation | No autocorrelation | Reject if p < 0.05 |
| **Breusch-Pagan-Godfrey Test** | Heteroskedasticity | Homoskedastic errors | Reject if p < 0.05 |
| **Jarque-Bera Test** | Normality of residuals | Errors are normal | Reject if p < 0.05 |
| **Ramsey RESET Test** | Functional form | Model correctly specified | Reject if p < 0.05 |

### 3.7.2 VAR Diagnostics
| Test | Purpose | Decision Rule |
|------|---------|---------------|
| **Portmanteau Test** | Residual autocorrelation | p > 0.05 → No autocorrelation |
| **Granger Causality Test** | Predictive relationships | p < 0.05 → X Granger-causes Y |
| **Stability Check** | Eigenvalue roots | All roots < 1 → VAR is stable |

**Stability is critical**: If the VAR is unstable (roots ≥ 1), impulse responses and forecasts are unreliable.

---

## 3.8 Policy Simulation Approach

### 3.8.1 Counterfactual Scenarios
Using the estimated VAR, we simulate the effect of hypothetical policy interventions:

**Scenario 1: 200 basis point MPR hike**
- Shock: MPR increases by 2 percentage points in month t
- Trace: Response of TBR, EXO, and INFL over 24 months

**Scenario 2: 100 basis point MPR cut**
- Shock: MPR decreases by 1 percentage point in month t
- Trace: Response of TBR, EXO, and INFL over 24 months

### 3.8.2 Policy Effectiveness Metrics
We measure effectiveness using:
1. **Peak response**: Maximum deviation of INFL from baseline
2. **Time to peak**: Number of months until peak response
3. **Cumulative effect**: Total change in INFL over 24 months
4. **Persistence**: Half-life of the shock (months until 50% dissipation)

### 3.8.3 Robustness Checks
To validate results, we:
- Re-estimate with alternative lag lengths (p±1)
- Test alternative VAR orderings (e.g., MPR → EXO → TBR → INFL)
- Estimate on sub-samples (pre-2020 vs. post-2020)
- Use alternative inflation measures (core inflation, food inflation)

If results are qualitatively similar across specifications, we conclude that findings are robust.

---

## 3.9 Software and Computational Tools

All analyses are conducted in **Python 3.12** using the following libraries:

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `statsmodels` | ARDL, VAR, unit root tests, cointegration tests |
| `matplotlib` & `seaborn` | Visualization |
| `scipy` | Statistical tests |

Code is version-controlled using **Git** and hosted on **GitHub** for reproducibility. All scripts are documented and can be found in the `src/` directory of the project repository.

---

## 3.10 Limitations

1. **Data frequency**: Monthly data may obscure short-term dynamics (daily or weekly data would be ideal but are unavailable for inflation)
2. **Structural breaks**: The methodology assumes parameter stability; major policy regime changes (e.g., 2023 currency redesign) may cause structural breaks
3. **Omitted variables**: Other factors (fiscal policy, global commodity prices, supply shocks) are not explicitly modeled
4. **Cholesky ordering**: The recursive identification assumption is strong; alternative identification schemes (e.g., sign restrictions) could yield different results
5. **Small sample**: 180 observations is adequate for time-series analysis but limits the precision of long-lag estimates

---

## 3.11 Ethical Considerations

This research uses publicly available, aggregated macroeconomic data. No individual or firm-level data are used, and no ethical approval is required. Results are intended to inform policy discussions and are presented objectively, acknowledging uncertainty through confidence intervals and robustness checks.

---

## References

- Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing approaches to the analysis of level relationships. *Journal of Applied Econometrics*, 16(3), 289-326.
- Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica*, 55(2), 251-276.
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica*, 59(6), 1551-1580.
- Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1-48.

---

**END OF METHODOLOGY DOCUMENT**
```

Save and exit nano (`Ctrl+O`, `Enter`, `Ctrl+X`).

---

## Step 2: Update Data Sources Documentation (Optional)

If you haven't already created a detailed data sources document, now is a good time. Open (or create) `docs/data_sources.md`:

```bash
nano docs/data_sources.md
```

**Delete everything in `docs/data_sources.md` and replace it with this:**

```markdown
# Data Sources

## Primary Data Source
All data used in this research were obtained from the **Central Bank of Nigeria (CBN) Statistical Database**.

- **URL**: [https://www.cbn.gov.ng/rates/](https://www.cbn.gov.ng/rates/)
- **Access**: Publicly available, no authentication required
- **Format**: Downloaded as CSV from the CBN website

---

## Dataset Description

- **File**: `data/raw/cbn_infl_data.csv`
- **Period**: January 2010 – December 2024
- **Frequency**: Monthly
- **Observations**: 180
- **Variables**: 34

---

## Variable Definitions

| Column Name | Full Name | Unit | Description |
|-------------|-----------|------|-------------|
| `date` | Date | YYYY-MM-DD | Observation date (first day of month) |
| `mpr` | Monetary Policy Rate | % per annum | CBN's benchmark policy rate |
| `omo` | Open Market Operations Rate | % per annum | Rate on CBN OMO bills |
| `crr` | Cash Reserve Ratio | % | Required reserves as % of deposits |
| `lr` | Liquidity Ratio | % | Liquid assets as % of deposits |
| `tbr` | 91-Day Treasury Bill Rate | % per annum | Yield on 91-day T-bills (primary market) |
| `infl` | Headline Inflation | % YoY | Consumer Price Index inflation rate |
| `inflcore` | Core Inflation | % YoY | Inflation excluding food and energy |
| `inflfood` | Food Inflation | % YoY | Inflation in food sub-index |
| `fcpi` | Food CPI | Index | Consumer Price Index for food |
| `ccpi` | Core CPI | Index | CPI excluding food and energy |
| `acpi` | All Items CPI | Index | Headline CPI (all items) |
| `fcpi_r` | Food CPI (Rural) | Index | Food CPI for rural areas |
| `ccpi_r` | Core CPI (Rural) | Index | Core CPI for rural areas |
| `acpi_r` | All Items CPI (Rural) | Index | Headline CPI for rural areas |
| `fcpi_u` | Food CPI (Urban) | Index | Food CPI for urban areas |
| `ccpi_u` | Core CPI (Urban) | Index | Core CPI for urban areas |
| `acpi_u` | All Items CPI (Urban) | Index | Headline CPI for urban areas |
| `exo` | Official Exchange Rate | NGN/USD | NAFEX (official) rate |
| `exbdc` | BDC Exchange Rate | NGN/USD | Bureau de Change (parallel market) rate |
| `neer` | Nominal Effective Exchange Rate | Index | Trade-weighted nominal exchange rate |
| `reer` | Real Effective Exchange Rate | Index | Trade-weighted real exchange rate |
| `sr` | Savings Rate | % per annum | Interest rate on savings deposits |
| `ir7d` | 7-Day Interbank Rate | % per annum | Rate on 7-day interbank loans |
| `ir1m` | 1-Month Interbank Rate | % per annum | Rate on 1-month interbank loans |
| `ir3m` | 3-Month Interbank Rate | % per annum | Rate on 3-month interbank loans |
| `ir6m` | 6-Month Interbank Rate | % per annum | Rate on 6-month interbank loans |
| `ir12m` | 12-Month Interbank Rate | % per annum | Rate on 12-month interbank loans |
| `irover12m` | Interbank Rate (>12M) | % per annum | Rate on interbank loans over 12 months |
| `plr` | Prime Lending Rate | % per annum | Base rate for loans to best customers |
| `mlr` | Maximum Lending Rate | % per annum | Ceiling on commercial lending rates |
| `icr` | Inter-bank Call Rate | % per annum | Overnight interbank lending rate |
| `obb` | Open Buy Back Rate | % per annum | CBN's repurchase rate for T-bills |
| `mktcap` | Stock Market Capitalization | Billion NGN | Total value of listed equities (NSE) |
| `asi` | All Share Index | Index | Nigerian Stock Exchange All Share Index |

---

## Core Variables (Used in Analysis)

For this study, we focus on **four core variables** that represent the key nodes in the monetary transmission mechanism:

1. **MPR** (Monetary Policy Rate) — Policy instrument
2. **TBR** (91-day Treasury Bill Rate) — Intermediate target (interest rate channel)
3. **EXO** (Official Exchange Rate) — Intermediate target (exchange rate channel)
4. **INFL** (Headline Inflation) — Target variable

---

## Data Quality Notes

- **Missing values**: None (dataset is complete for all 180 observations)
- **Outliers**: Some extreme values during crisis periods (2016, 2020, 2023) are genuine and retained
- **Seasonality**: Inflation has seasonal patterns (food price cycles); not deseasonalized to preserve policy-relevant variation
- **Structural breaks**: Potential breaks in 2016 (FX regime change), 2020 (COVID-19), 2023 (currency redesign)

---

## Data Preprocessing

Raw data are cleaned and processed using the script `src/data_processing.py`:
- Date parsing (converting strings to datetime)
- Column name standardization (lowercase)
- Sorting by date (ascending)
- Validation (checking for duplicates, missing values, outliers)

Cleaned data are saved to `data/processed/cbn_data_clean.csv`.

---

## Citation

When citing this data, use:

> Central Bank of Nigeria (2024). *Statistical Database*. Retrieved from https://www.cbn.gov.ng/rates/ (Accessed: [Your access date])

---

**Last Updated**: February 10, 2026
```

Save and exit nano.

---

## Step 3: Verify Your Documentation

Check that both files were created:
```bash
ls -lh docs/
```

You should see:
```
methodology.md
data_sources.md
```

Take a quick look at the methodology file to confirm formatting:
```bash
head -n 50 docs/methodology.md
```

---

## Step 4: Commit Your Documentation

Stage the new documentation files:
```bash
git add docs/methodology.md docs/data_sources.md
```

Check status:
```bash
git status
```

Commit with a descriptive message:
```bash
git commit -m "Add comprehensive methodology and data sources documentation

- Created docs/methodology.md covering research design, data description,
  variable definitions, transmission mechanism, econometric methods (ADF,
  KPSS, Engle-Granger, Johansen, ARDL bounds testing, VAR), model specs,
  diagnostic tests, and policy simulation approach
- Updated docs/data_sources.md with complete variable list and definitions
- Thesis-quality writing suitable for academic submission

https://claude.ai/code/session_0173mGraLvrZwx6v1oTi7RnN"
```

Push to GitHub:
```bash
git push origin claude/inflation-predictor-platform-JG3LF
```

---

## What You've Learned Today

1. **How to structure a methodology chapter** with sections on research design, data, variables, theory, estimation methods, and diagnostics
2. **How to justify variable selection** by linking each variable to the transmission mechanism
3. **How to document econometric procedures** (stationarity testing, cointegration, ARDL, VAR) in plain language
4. **How to explain identification strategies** (Cholesky ordering) and their economic logic
5. **How to write transparently about limitations** and robustness checks

---

## Q&A: Defending Your Methodology to an Examiner

### Q1: Why did you use TBR instead of M2 (money supply)?
**A**: Nigeria's CBN operates an **interest rate targeting regime**, not a monetary quantity targeting regime. The relationship between M2 and inflation is unstable due to financial innovation, dollarization, and velocity shifts. TBR better captures the actual transmission of policy intentions to financial conditions.

### Q2: How did you choose the VAR ordering (MPR → TBR → EXO → INFL)?
**A**: The ordering reflects the **temporal structure of transmission**:
1. **MPR** is set by the CBN at discrete MPC meetings (6-8 times per year) and is exogenous within each month
2. **TBR** responds immediately to MPR changes (within days)
3. **EXO** responds to interest rates through capital flows and CBN FX interventions (within weeks)
4. **INFL** is measured with a lag (NBS publishes CPI data 2-3 weeks after month-end) and responds to exchange rate pass-through (within months)

This ordering is standard in monetary VAR literature and consistent with Nigerian institutional facts.

### Q3: Why use both Engle-Granger and Johansen cointegration tests?
**A**: They are **complementary**:
- **Engle-Granger** is simple and intuitive but assumes a single cointegrating vector and depends on which variable is chosen as dependent
- **Johansen** can detect multiple cointegrating vectors and is more efficient in multivariate settings

If both tests agree, we have strong evidence. If they disagree, we investigate further.

### Q4: What if the ARDL bounds test is inconclusive (F-stat falls between critical bounds)?
**A**: We would:
1. Check whether individual variables are I(0) or I(1) (if all are I(0) or all are I(1), we can use lower or upper bound)
2. Increase the sample size if possible (though our sample is already 180 observations)
3. Use alternative cointegration tests (Johansen) as a tiebreaker
4. Report the ambiguity transparently and interpret long-run coefficients with caution

### Q5: How do you ensure your VAR is stable?
**A**: We compute the **eigenvalues** of the companion matrix. If all eigenvalues have modulus < 1, the VAR is stable (i.e., shocks dissipate over time rather than exploding). In Python, `statsmodels` provides a plot of inverse roots of the characteristic polynomial — all roots should lie inside the unit circle.

If the VAR is unstable, impulse responses and variance decompositions are unreliable, and we would need to re-specify the model (e.g., reduce lag length, add differencing, or include exogenous variables).

### Q6: Why 24-month impulse response horizon?
**A**: Monetary policy effects on inflation typically take 12-24 months to fully materialize (the "long and variable lags" of policy transmission). A 24-month horizon captures:
- Short-term effects (0-6 months): Financial market reactions
- Medium-term effects (6-12 months): Exchange rate pass-through to import prices
- Long-term effects (12-24 months): Aggregate demand effects on domestic prices

Beyond 24 months, confidence intervals become very wide, making inference unreliable.

### Q7: What if diagnostic tests fail (e.g., residuals are not normal)?
**A**: We would:
1. **Serial correlation**: Add more lags to the ARDL or VAR
2. **Heteroskedasticity**: Use robust standard errors (White or HAC)
3. **Non-normality**: Check for outliers; consider robust estimation methods; note that ARDL/VAR are fairly robust to non-normality if the sample is large (N > 100)
4. **Misspecification**: Add omitted variables, test for structural breaks, or use non-linear specifications

We report all diagnostic results transparently, including any failures and remedial actions.

### Q8: How do you address structural breaks (e.g., 2016 FX crisis, 2023 currency redesign)?
**A**: We perform **robustness checks**:
1. **Chow test**: Formally test for a break at suspected dates
2. **Sub-sample analysis**: Estimate separately for 2010-2019 and 2020-2024
3. **Dummy variables**: Include crisis dummies in the ARDL/VAR
4. **Time-varying parameter models**: Allow coefficients to change over time (advanced)

If we find strong evidence of a break, we acknowledge it in limitations and discuss implications for policy interpretation.

### Q9: Why did you choose ARDL over a traditional Error Correction Model (ECM)?
**A**: ARDL is more flexible:
- **ECM** requires all variables to be I(1) and cointegrated
- **ARDL** works with I(0), I(1), or mixed integration
- ARDL automatically selects lag lengths via AIC, while ECM requires manual specification

ARDL is the modern best practice for long-run/short-run decomposition, especially when integration orders are uncertain.

### Q10: How would you respond to a reviewer who says "Your model is too simple — you omitted fiscal policy, oil prices, etc."?
**A**: Acknowledge and defend:
1. **Parsimony**: Simple models are easier to interpret and less prone to overfitting
2. **Objective**: The research question is specifically about **monetary policy transmission**, not a full macro model
3. **Data limitations**: Adding more variables reduces degrees of freedom and increases multicollinearity
4. **Robustness**: We test alternative specifications and sub-samples; if results are robust, omitted variables are not driving conclusions
5. **Future work**: We explicitly list omitted factors in the limitations section and suggest extensions for future research

A good methodology is not one that includes everything, but one that clearly justifies what is included and what is left out.

---

## Summary

Today you created **two critical documents** for your thesis:

1. **`docs/methodology.md`**: A comprehensive methodology chapter covering data, theory, estimation techniques, and diagnostics
2. **`docs/data_sources.md`**: Complete documentation of your data sources and variable definitions

These documents serve multiple purposes:
- **For your thesis**: Copy-paste into Chapter 3
- **For replication**: Other researchers can reproduce your work
- **For defense**: You can confidently answer examiner questions about your methods

**Next steps**: Tomorrow (Day 37) you'll write the **results and interpretation chapter**, translating your statistical output into economic insights.

---

**End of Day 36**
