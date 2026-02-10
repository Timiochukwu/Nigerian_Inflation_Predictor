# Day 37 — Results Interpretation Guide

## What You'll Learn Today
- How to interpret and present your econometric results
- Writing a results chapter for your thesis
- Creating comprehensive documentation for results interpretation
- Understanding what examiners look for in econometric analysis

**Time Required**: 60-90 minutes
**Difficulty**: Medium (Documentation & Interpretation)

---

## Why This Matters

You've run all the tests and models. Now what?

Many students struggle with **presenting** their findings. Your supervisor or examiner doesn't just want numbers — they want:
- Clear interpretation of statistical output
- Connection to economic theory
- Honest discussion of limitations
- Professional presentation

Today you'll create a **results interpretation guide** that serves as:
1. A reference document for writing your thesis
2. A preparation tool for your defense
3. A template for presenting econometric findings

---

## The Big Picture

Your econometric analysis follows this structure:

```
Chapter 4: Results and Discussion
├── 4.1 Preliminary Tests
│   ├── 4.1.1 Descriptive Statistics
│   ├── 4.1.2 Stationarity Tests (ADF, KPSS)
│   └── 4.1.3 Cointegration Tests (Engle-Granger, Johansen)
├── 4.2 Model Estimation
│   ├── 4.2.1 ARDL Bounds Test
│   ├── 4.2.2 Long-Run Relationships
│   ├── 4.2.3 Short-Run Dynamics
│   └── 4.2.4 Error Correction Term
├── 4.3 VAR Analysis
│   ├── 4.3.1 Granger Causality
│   ├── 4.3.2 Impulse Response Functions
│   └── 4.3.3 Forecast Error Variance Decomposition
├── 4.4 Policy Simulations
│   └── 4.4.1 MPR Shock Scenarios
└── 4.5 Discussion
    ├── Comparison with Theory
    ├── Comparison with Literature
    └── Policy Implications
```

Today you'll create a guide for interpreting each section.

---

## Step 1: Create the Documentation File

Open your terminal and create the file:

```bash
touch docs/results_interpretation.md
```

Now delete everything in `docs/results_interpretation.md` and replace it with this:

```markdown
# Results Interpretation Guide
*Nigerian Inflation Predictor Project*

## Table of Contents
1. [Descriptive Statistics](#descriptive-statistics)
2. [Stationarity Tests](#stationarity-tests)
3. [Cointegration Tests](#cointegration-tests)
4. [ARDL Model Results](#ardl-model-results)
5. [VAR Analysis Results](#var-analysis-results)
6. [Policy Simulation Results](#policy-simulation-results)
7. [Common Examiner Questions](#common-examiner-questions)
8. [Tables and Figures for Thesis](#tables-and-figures-for-thesis)

---

## Descriptive Statistics

### How to Present

**Table 4.1: Descriptive Statistics of Variables**

| Variable | Obs | Mean | Std Dev | Min | Max |
|----------|-----|------|---------|-----|-----|
| INFL | 120 | 12.45 | 4.23 | 8.01 | 18.72 |
| MPR | 120 | 11.85 | 2.15 | 6.00 | 14.00 |
| TBR | 120 | 10.23 | 3.45 | 4.50 | 16.80 |
| EXO | 120 | 305.67 | 89.34 | 150.00 | 460.00 |

**How to Interpret (Write This in Your Thesis)**:

"Table 4.1 presents descriptive statistics for the variables over the sample period (Month YYYY to Month YYYY). Average headline inflation stood at 12.45% with a standard deviation of 4.23 percentage points, ranging from 8.01% to 18.72%. The Monetary Policy Rate (MPR) averaged 11.85%, reflecting the Central Bank's efforts to anchor inflation expectations. The Treasury Bill Rate (TBR), a proxy for short-term interest rates, averaged 10.23%, while the official exchange rate (EXO) averaged ₦305.67/$ with substantial volatility (SD = 89.34), reflecting the managed float regime and periodic interventions."

**Key Points**:
- INFL mean > MPR mean suggests real rates were often negative
- High EXO standard deviation indicates exchange rate volatility
- TBR < MPR is expected (T-bill is risk-free benchmark)

---

## Stationarity Tests

### ADF Test Results

**How to Present**

**Table 4.2: Augmented Dickey-Fuller Test Results**

| Variable | Level (t-stat) | p-value | I(0)? | 1st Diff (t-stat) | p-value | I(1)? |
|----------|----------------|---------|-------|-------------------|---------|-------|
| INFL | -2.345 | 0.159 | No | -8.234 | 0.000 | Yes |
| MPR | -1.876 | 0.345 | No | -6.789 | 0.000 | Yes |
| TBR | -2.123 | 0.234 | No | -7.456 | 0.000 | Yes |
| EXO | -1.234 | 0.654 | No | -9.123 | 0.000 | Yes |

*Note: Critical value at 5% = -2.89 (with constant). H0: Unit root present.*

**How to Interpret**:

"Table 4.2 reports the results of the Augmented Dickey-Fuller (ADF) unit root test. At levels, all variables fail to reject the null hypothesis of a unit root at the 5% significance level (all p-values > 0.05). However, after first differencing, all variables become stationary (all p-values < 0.01), indicating they are integrated of order one, I(1). This finding is consistent with most macroeconomic time series and satisfies the prerequisite for cointegration analysis."

**What This Means Economically**:

Non-stationarity implies these series contain stochastic trends (random walks). For policy:
- Shocks to inflation have permanent effects (no automatic mean reversion)
- Active monetary policy is needed to stabilize inflation
- This justifies the CBN's inflation targeting framework

### KPSS Test Results

**How to Present**

**Table 4.3: KPSS Stationarity Test Results**

| Variable | Level (stat) | 5% CV | Stationary? | 1st Diff (stat) | 5% CV | Stationary? |
|----------|--------------|-------|-------------|-----------------|-------|-------------|
| INFL | 0.678 | 0.463 | No | 0.123 | 0.463 | Yes |
| MPR | 0.734 | 0.463 | No | 0.089 | 0.463 | Yes |
| TBR | 0.823 | 0.463 | No | 0.101 | 0.463 | Yes |
| EXO | 0.912 | 0.463 | No | 0.067 | 0.463 | Yes |

*Note: H0: Series is stationary (opposite of ADF).*

**How to Interpret**:

"To confirm the ADF results, Table 4.3 presents the KPSS test, which has stationarity as the null hypothesis. At levels, all test statistics exceed the 5% critical value (0.463), rejecting stationarity. However, first differences yield test statistics below the critical value, confirming stationarity. The KPSS results corroborate the ADF findings: all variables are I(1)."

**Why Both Tests?**:

ADF and KPSS have different null hypotheses:
- ADF: H0 = unit root (non-stationary)
- KPSS: H0 = stationary

Using both avoids the "indeterminate zone" where neither null is rejected due to low power.

---

## Cointegration Tests

### Engle-Granger Test

**How to Present**

**Table 4.4: Engle-Granger Cointegration Test**

| Dependent Var | ADF on Residuals | p-value | Cointegrated? |
|---------------|------------------|---------|---------------|
| INFL | -4.567 | 0.023 | Yes |

*Note: MacKinnon critical value at 5% = -3.85 (4 variables).*

**How to Interpret**:

"Table 4.4 presents the Engle-Granger cointegration test, regressing INFL on MPR, TBR, and EXO, and testing the residuals for stationarity. The ADF statistic (-4.567) exceeds the MacKinnon critical value (-3.85) at the 5% level, rejecting the null of no cointegration. This indicates a stable long-run equilibrium relationship exists among the variables, justifying the use of error correction models."

### Johansen Test

**How to Present**

**Table 4.5: Johansen Cointegration Test**

| H0: r = | Trace Stat | 5% CV | Max-Eigen Stat | 5% CV |
|---------|------------|-------|----------------|-------|
| 0 | 68.45** | 47.86 | 35.67** | 27.58 |
| ≤ 1 | 32.78 | 29.79 | 18.23 | 21.13 |
| ≤ 2 | 14.55 | 15.49 | 10.12 | 14.26 |
| ≤ 3 | 4.43 | 3.84 | 4.43 | 3.84 |

*Note: ** denotes rejection at 5% level. r = number of cointegrating vectors.*

**How to Interpret**:

"Table 4.5 reports the Johansen cointegration test results. Both the Trace and Max-Eigenvalue statistics reject the null hypothesis of no cointegration (r = 0) at the 5% level, but fail to reject r ≤ 1. This suggests exactly **one cointegrating relationship** among the four variables (INFL, MPR, TBR, EXO). The existence of a single cointegrating vector is theoretically consistent with a monetary policy reaction function where the CBN adjusts MPR to stabilize inflation, with TBR and EXO serving as transmission channels."

**Economic Interpretation**:

One cointegrating vector means:
- The variables share a common stochastic trend
- There's a long-run equilibrium "attractor"
- Deviations are temporary; forces pull the system back
- This validates structural models like ARDL or VECM

---

## ARDL Model Results

### Bounds Test

**How to Present**

**Table 4.6: ARDL Bounds Test for Cointegration**

| Test Statistic | Value | I(0) Lower | I(1) Upper | Decision |
|----------------|-------|------------|------------|----------|
| F-statistic | 5.234 | 2.79 | 3.67 | Cointegrated |

*Note: Critical values from Pesaran et al. (2001) for k=3 regressors at 5% level.*

**How to Interpret**:

"Table 4.6 presents the ARDL bounds test for cointegration. The calculated F-statistic (5.234) exceeds both the I(0) lower bound (2.79) and I(1) upper bound (3.67) at the 5% significance level, confirming a long-run relationship between inflation and its determinants. This result is robust regardless of whether the regressors are I(0) or I(1), which is the key advantage of the ARDL approach."

### Long-Run Coefficients

**How to Present**

**Table 4.7: ARDL Long-Run Coefficients**

| Variable | Coefficient | Std Error | t-statistic | p-value |
|----------|-------------|-----------|-------------|---------|
| MPR | -0.345 | 0.089 | -3.876 | 0.001*** |
| TBR | 0.234 | 0.067 | 3.493 | 0.002*** |
| EXO | 0.012 | 0.004 | 3.000 | 0.004*** |
| Constant | 8.456 | 2.345 | 3.605 | 0.001*** |

*Note: *** p<0.01, ** p<0.05, * p<0.10*

**How to Interpret**:

"Table 4.7 displays the estimated long-run coefficients from the ARDL(4,2,3,1) model selected by AIC.

- **MPR coefficient (-0.345)**: A 1 percentage point increase in MPR reduces inflation by 0.345 percentage points in the long run, significant at the 1% level. This negative relationship validates the monetary transmission mechanism.

- **TBR coefficient (+0.234)**: Surprisingly, TBR has a positive long-run effect on inflation. This may reflect the 'cost channel' of monetary policy, where higher interest rates increase firms' borrowing costs, leading to higher prices. Alternatively, it could indicate that TBR rises endogenously in response to inflation expectations.

- **EXO coefficient (+0.012)**: A ₦1 depreciation of the naira raises inflation by 0.012 percentage points, confirming exchange rate pass-through to domestic prices, consistent with Nigeria's high import dependence.

These findings align with the literature on emerging markets where exchange rate channels often dominate."

### Short-Run Dynamics

**How to Present**

**Table 4.8: ARDL Short-Run Coefficients**

| Variable | Coefficient | Std Error | t-statistic | p-value |
|----------|-------------|-----------|-------------|---------|
| D(INFL(-1)) | 0.234 | 0.089 | 2.629 | 0.010** |
| D(MPR) | -0.123 | 0.067 | -1.836 | 0.070* |
| D(MPR(-1)) | -0.089 | 0.054 | -1.648 | 0.103 |
| D(TBR) | 0.156 | 0.045 | 3.467 | 0.001*** |
| D(TBR(-1)) | 0.098 | 0.048 | 2.042 | 0.044** |
| D(EXO) | 0.008 | 0.003 | 2.667 | 0.009*** |
| ECT(-1) | -0.456 | 0.098 | -4.653 | 0.000*** |

**How to Interpret**:

"Table 4.8 presents the short-run dynamics. The coefficient on D(MPR) is -0.123, indicating that a 1 percentage point MPR hike reduces inflation by 0.123 percentage points within the same month, though this effect is only marginally significant (p=0.07). The immediate effect is weaker than the long-run effect (-0.345), which is typical.

The Error Correction Term (ECT) is -0.456 and highly significant (p<0.001). This means that 45.6% of any disequilibrium is corrected each month, implying a half-life of about 1.5 months. The negative sign confirms the system converges to equilibrium."

### Error Correction Term (ECT)

**Critical for Examiners!**

The ECT coefficient must be:
1. **Negative** (otherwise divergence, not convergence)
2. **Significant** (p < 0.05, ideally p < 0.01)
3. **Between -1 and 0** (otherwise overshooting oscillations)

**How to Explain ECT to Examiners**:

"The ECT captures the speed of adjustment toward long-run equilibrium. A value of -0.456 means that if inflation is 10 percentage points above its equilibrium level this month, it will fall by 4.56 percentage points next month (holding other factors constant). Full adjustment takes about 2-3 months, which is plausible given the CBN's monthly Monetary Policy Committee meetings."

---

## VAR Analysis Results

### Granger Causality

**How to Present**

**Table 4.9: Granger Causality Test Results**

| H0: X does not Granger-cause Y | Chi-sq | df | p-value | Decision |
|--------------------------------|--------|----|---------| ---------|
| MPR ↛ INFL | 12.345 | 4 | 0.015** | Reject H0 |
| INFL ↛ MPR | 18.678 | 4 | 0.001*** | Reject H0 |
| TBR ↛ INFL | 9.234 | 4 | 0.055* | Weak Reject |
| EXO ↛ INFL | 15.456 | 4 | 0.004*** | Reject H0 |

**How to Interpret**:

"Table 4.9 reports Granger causality tests from a VAR(4) model.

- **MPR → INFL**: We reject the null that MPR does not Granger-cause inflation (p=0.015), confirming that past values of MPR help predict current inflation.

- **INFL → MPR**: We also reject this null (p=0.001), indicating **bidirectional causality**. This makes economic sense: the CBN raises MPR in response to inflation (Taylor Rule), and MPR affects future inflation (transmission mechanism).

- **EXO → INFL**: Strong causality (p=0.004), confirming exchange rate pass-through.

These results validate the use of VAR for analyzing dynamic interactions."

### Impulse Response Functions (IRF)

**How to Present**

Include a figure:

**Figure 4.1: Impulse Response of INFL to MPR Shock**
```
[Graph showing INFL response over 12 months to a 1% MPR shock]
- Initial period: slight positive response (price puzzle)
- Months 2-3: turns negative
- Peak negative effect at month 4: -0.25%
- Converges to zero by month 10
```

**How to Interpret**:

"Figure 4.1 displays the impulse response of inflation to a one-standard-deviation shock in MPR (approximately 0.5 percentage points).

- **Month 0-1**: A slight positive response (the 'price puzzle'), potentially due to the cost channel or signaling effects where MPR hikes signal future inflation.

- **Months 2-6**: Inflation declines, reaching a peak negative effect of -0.25 percentage points at month 4. This lag is consistent with the transmission mechanism operating through credit, exchange rate, and expectations channels.

- **Months 7-10**: The effect gradually dissipates as the system returns to equilibrium.

The IRF confirms that monetary policy affects inflation with a lag of 2-4 months, supporting the CBN's forward-looking approach to policy."

**Common IRF Pitfalls**:
- Sign flips can occur (price puzzle) — acknowledge and explain
- Confidence bands matter — report 95% CI
- Ordering matters (Cholesky) — justify MPR → TBR → EXO → INFL

### Forecast Error Variance Decomposition (FEVD)

**How to Present**

**Table 4.10: FEVD of Inflation (% of variance explained)**

| Period | INFL | MPR | TBR | EXO |
|--------|------|-----|-----|-----|
| 1 | 100.0 | 0.0 | 0.0 | 0.0 |
| 3 | 85.3 | 5.2 | 3.1 | 6.4 |
| 6 | 68.7 | 12.3 | 8.9 | 10.1 |
| 12 | 54.2 | 18.5 | 13.4 | 13.9 |

**How to Interpret**:

"Table 4.10 presents the forecast error variance decomposition of inflation.

- **Period 1**: 100% of inflation variance is explained by its own shocks (by construction in Cholesky ordering).

- **Period 6**: After 6 months, own shocks explain 68.7% of inflation variance. MPR shocks account for 12.3%, while exchange rate (EXO) and TBR shocks contribute 10.1% and 8.9%, respectively.

- **Period 12**: By one year, MPR explains 18.5% of inflation variance, confirming a meaningful but not dominant role for monetary policy. EXO (13.9%) and TBR (13.4%) have comparable importance.

This suggests inflation in Nigeria is driven by multiple factors, with monetary policy playing a significant but not exclusive role. Supply shocks and exchange rate dynamics are equally important, consistent with the country's structural characteristics (oil dependence, import reliance)."

---

## Policy Simulation Results

**How to Present**

**Table 4.11: Policy Simulation Results (Baseline vs. MPR+1% Shock)**

| Month | Baseline INFL | Counterfactual INFL | Difference | Cumulative Effect |
|-------|---------------|---------------------|------------|-------------------|
| 1 | 12.50 | 12.48 | -0.02 | -0.02 |
| 2 | 12.55 | 12.45 | -0.10 | -0.12 |
| 3 | 12.60 | 12.38 | -0.22 | -0.34 |
| 6 | 12.80 | 12.42 | -0.38 | -1.12 |
| 12 | 13.00 | 12.65 | -0.35 | -2.15 |

**How to Interpret**:

"Table 4.11 simulates the effect of a sustained 1 percentage point increase in MPR over 12 months. Relative to the baseline scenario, inflation is 0.38 percentage points lower by month 6, with the peak effect occurring around that time. By month 12, the effect slightly diminishes (-0.35 pp), consistent with the IRF pattern. Cumulatively, the policy reduces inflation by 2.15 percentage points over the year.

**Policy Implication**: A sustained 100 basis point MPR hike can reduce inflation by approximately 0.35-0.40 percentage points within 6 months. To achieve a 2% reduction in inflation (e.g., from 12% to 10%), the CBN would need to raise MPR by roughly 5-6 percentage points, assuming no offsetting shocks."

---

## Common Examiner Questions

### Q1: "Why are all your variables I(1)? Isn't that too convenient?"

**Answer**:

"It's not convenience; it's a well-documented empirical regularity. Most macroeconomic time series (GDP, inflation, interest rates, exchange rates) are I(1) due to the presence of stochastic trends. This has been shown in hundreds of studies since Nelson and Plosser (1982). The economic intuition is that these variables have no inherent mean to revert to—shocks have permanent effects absent policy intervention. I confirmed this in my data using both ADF and KPSS tests with consistent results."

### Q2: "Your TBR coefficient is positive. Doesn't that contradict theory?"

**Answer**:

"The positive TBR coefficient in the long run can be explained by two mechanisms:

1. **Cost Channel**: Higher interest rates increase firms' financing costs, which they pass through to consumers as higher prices (Ravenna and Walsh, 2006).

2. **Endogeneity**: TBR may rise in response to inflation expectations, even after controlling for MPR. Market participants demand higher yields on T-bills when they anticipate inflation.

3. **Transmission Channel**: TBR reflects the full term structure effect, while MPR is a policy signal. Their distinct coefficients capture different aspects of monetary transmission.

I acknowledge this is a limitation and warrants further investigation with richer data (e.g., disaggregated inflation, bank lending rates)."

### Q3: "Why did you get the price puzzle in your IRF?"

**Answer**:

"The price puzzle—where inflation initially rises after a contractionary MPR shock—is common in VAR studies, especially in developing countries. Possible explanations include:

1. **Signaling Effect**: An MPR hike may signal the CBN's private information about future inflation, causing expectations to jump.

2. **Cost Channel**: Immediate cost increases dominate demand effects in the short run.

3. **Omitted Variables**: Lacking data on commodity prices or fiscal policy, which might correlate with MPR decisions.

Sims (1992) showed that including commodity prices often resolves this. Given data constraints, I acknowledge it as a limitation. Importantly, the puzzle reverses by month 2-3, and the overall dynamic aligns with theory."

### Q4: "How do you know your model is correctly specified?"

**Answer**:

"I conducted several diagnostic tests:

1. **Lag Selection**: Used AIC, BIC, and HQIC; all suggested 4 lags. I also checked that residuals become white noise.

2. **Residual Diagnostics**:
   - Jarque-Bera test for normality
   - Breusch-Godfrey test for serial correlation
   - ARCH test for heteroskedasticity
   - Results available in the appendix

3. **Stability**: Checked that all roots of the VAR characteristic polynomial lie inside the unit circle (confirmed stability).

4. **Cointegration Consistency**: Both Engle-Granger and Johansen confirmed cointegration; ARDL bounds test also confirmed.

While no model is perfect, these tests give me confidence in the specification."

### Q5: "What's the difference between ARDL and VAR? Why use both?"

**Answer**:

"They serve complementary purposes:

**ARDL**:
- Focuses on a single equation (inflation)
- Identifies long-run equilibrium and short-run dynamics
- Provides the Error Correction Term (speed of adjustment)
- Best for testing a specific hypothesis about inflation determinants

**VAR**:
- Treats all variables symmetrically
- Explores bidirectional causality (Granger tests)
- Generates IRFs and FEVDs for shock analysis
- Better for understanding the full system dynamics

Using both provides a more complete picture. ARDL tells me *what* the long-run relationships are; VAR tells me *how* shocks propagate through the system."

---

## Tables and Figures for Thesis

### Essential Tables

1. **Table 4.1**: Descriptive Statistics
2. **Table 4.2**: ADF Test Results (Levels and First Differences)
3. **Table 4.3**: KPSS Test Results
4. **Table 4.4**: Engle-Granger Cointegration Test
5. **Table 4.5**: Johansen Cointegration Test
6. **Table 4.6**: ARDL Bounds Test
7. **Table 4.7**: ARDL Long-Run Coefficients
8. **Table 4.8**: ARDL Short-Run Coefficients and ECT
9. **Table 4.9**: Granger Causality Test Results
10. **Table 4.10**: FEVD of Inflation

### Essential Figures

1. **Figure 4.1**: Time Series Plots of All Variables (raw data)
2. **Figure 4.2**: First Differences (show stationarity visually)
3. **Figure 4.3**: IRF of INFL to MPR Shock
4. **Figure 4.4**: IRF of INFL to EXO Shock
5. **Figure 4.5**: FEVD of Inflation (stacked bar chart over 12 periods)
6. **Figure 4.6**: CUSUM and CUSUMSQ Tests (stability)
7. **Figure 4.7**: Actual vs. Fitted Inflation (from ARDL model)
8. **Figure 4.8**: Policy Simulation Results (baseline vs. counterfactual)

### Pro Tips for Presentation

1. **Always report p-values** or significance stars (*, **, ***)
2. **Include notes** under tables explaining critical values, null hypotheses
3. **Number equations** in text (e.g., "Equation 4.1 shows the ARDL specification...")
4. **Reference tables/figures in text** ("As shown in Table 4.7...")
5. **Don't just describe numbers—interpret economically**
6. **Compare with literature** ("This elasticity of -0.345 is comparable to Balogun (2007) who found -0.40 for Nigeria")
7. **Discuss limitations honestly** (shows maturity, not weakness)

---

## Summary

Your results chapter should flow logically:

1. **Establish the data properties** (descriptive stats, stationarity)
2. **Test for long-run relationships** (cointegration)
3. **Estimate the model** (ARDL or VAR)
4. **Interpret economically** (not just statistically)
5. **Simulate policy** (show practical relevance)
6. **Discuss robustness and limitations**

Remember: **Examiners are not trying to trap you.** They want to see:
- You understand what you did
- You can connect statistics to economics
- You're honest about limitations
- You can defend your choices

This interpretation guide is your roadmap. Refer to it when writing Chapter 4 of your thesis.

```

Save the file (Ctrl+S).

---

## Step 2: Commit Your Work

Now add and commit this new documentation:

```bash
cd /home/user/Nigerian_Inflation_Predictor
git add docs/results_interpretation.md
git commit -m "Add comprehensive results interpretation guide

- Created docs/results_interpretation.md with detailed guidance on interpreting econometric results
- Covers stationarity tests, cointegration tests, ARDL results, VAR analysis
- Includes example tables and interpretations for thesis writing
- Provides answers to common examiner questions
- Lists essential tables and figures for results chapter

This guide serves as a reference for presenting findings in a thesis or research paper.

https://claude.ai/code/session_0173mGraLvrZwx6v1oTi7RnN"
```

Push to remote:

```bash
git push origin claude/inflation-predictor-platform-JG3LF
```

---

## What You Built Today

You created a **comprehensive results interpretation guide** (`docs/results_interpretation.md`) that:

1. ✅ Shows how to present stationarity tests (ADF, KPSS)
2. ✅ Explains cointegration results (Engle-Granger, Johansen)
3. ✅ Interprets ARDL outputs (bounds test, long-run, short-run, ECT)
4. ✅ Interprets VAR outputs (Granger causality, IRF, FEVD)
5. ✅ Guides policy simulation interpretation
6. ✅ Answers common examiner questions
7. ✅ Lists essential tables and figures for a thesis

This document will be invaluable when you write your thesis Chapter 4.

---

## Q&A

**Q1: Do I need to include ALL these tables in my thesis?**

A: Not necessarily. The essential minimum is:
- Descriptive statistics
- Stationarity tests (ADF is standard)
- Cointegration test (choose Johansen OR Engle-Granger)
- Your main model results (ARDL long-run and short-run)
- Key diagnostic tests

VAR results (IRF, FEVD) are highly recommended if you're doing VAR analysis. Check your department's thesis requirements.

**Q2: What if my ECT is positive?**

A: That's a **serious problem**. A positive ECT means the system diverges from equilibrium, which invalidates the error correction model. Possible causes:
- Misspecified model (wrong variables, wrong lags)
- No true cointegration (pre-tests were misleading)
- Structural break in the data

You'd need to revisit your model specification.

**Q3: How much interpretation is "enough" for a table?**

A: Good rule of thumb:
- 1 sentence describing what the table shows
- 1-2 sentences interpreting key coefficients/statistics
- 1 sentence connecting to economic theory or policy
- 1 sentence comparing to literature (if applicable)

Example (for ARDL long-run MPR coefficient):
- "Table 4.7 shows MPR has a coefficient of -0.345."
- "This means a 1pp MPR increase reduces inflation by 0.345pp in the long run."
- "This validates the monetary transmission mechanism."
- "The magnitude is similar to Adebiyi and Mordi (2012) who found -0.38 for Nigeria."

**Q4: Should I discuss every variable in every table?**

A: Focus on:
- **Your key variables** (MPR, INFL for this project)
- **Statistically significant** results
- **Economically surprising** results (e.g., positive TBR)
- **Policy-relevant** findings

You can briefly mention control variables, but don't need lengthy interpretation.

**Q5: What if I get different results from ARDL vs. VAR?**

A: That's actually normal and informative:
- ARDL focuses on long-run equilibrium (levels relationship)
- VAR focuses on short-run dynamics (changes/shocks)

If MPR is significant in ARDL but not in VAR Granger tests, it means MPR affects the long-run level of inflation but may not have strong predictive power for short-run fluctuations. This is economically plausible.

**Discuss** the differences in your thesis—it shows depth.

**Q6: How do I prepare for my thesis defense using this guide?**

A: Practice explaining:
1. Your stationarity results to a non-economist (What does I(1) mean in plain English?)
2. The ECT to your grandmother (How fast does inflation correct?)
3. The price puzzle to a skeptical examiner (Why might theory not hold initially?)
4. Policy implications to a central banker (What should the CBN do?)

If you can do these four things, you're ready.

---

## Tomorrow: Day 38

**Topic**: Diagnostic Tests and Robustness Checks

You'll learn:
- How to test your model for serial correlation, heteroskedasticity, normality
- Structural break tests (Chow test, CUSUM)
- Robustness checks (alternative lag lengths, subsamples)
- Creating `docs/diagnostic_tests.md`

---

**Congratulations!** You now have a professional interpretation guide that will support you through thesis writing and your defense. 🎓
