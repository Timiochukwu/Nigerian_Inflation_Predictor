# Econometric Methodology — Nigerian Inflation Predictor

> This document will be expanded incrementally as each modelling stage is completed. Section headers are defined upfront to enforce the analytical sequence.

---

## 1. Introduction and Research Question

**Central question:** How do monetary policy rate changes by the Central Bank of Nigeria transmit to headline inflation through the exchange rate and money supply channels, and over what time horizon?

**Analytical objective:** To estimate, quantify, and simulate the dynamic effects of MPR shocks on Nigerian inflation using time-series econometric methods applied to monthly data.

---

## 2. Variable Definitions and Data Description

*To be completed — Day 2–4 (data ingestion and exploratory analysis).*

---

## 3. Stationarity and Unit Root Testing

### 3.1 Augmented Dickey-Fuller (ADF) Test
*To be completed — Day 6.*

### 3.2 Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
*To be completed — Day 6.*

### 3.3 Integration Order Summary
*To be completed — Day 6.*

---

## 4. ARDL Model

### 4.1 Theoretical Basis for ARDL
*To be completed — Week 2.*

### 4.2 Lag Selection
*To be completed — Week 2.*

### 4.3 Bounds Test for Cointegration
*To be completed — Week 2.*

### 4.4 Long-Run Coefficients
*To be completed — Week 2.*

### 4.5 Short-Run Dynamics and Error Correction Term
*To be completed — Week 2.*

### 4.6 Diagnostic Tests
*To be completed — Week 2.*

---

## 5. VAR Model

### 5.1 Model Specification
*To be completed — Week 3.*

### 5.2 Lag Length Selection (AIC, BIC, HQIC)
*To be completed — Week 3.*

### 5.3 Stability Diagnostics
*To be completed — Week 3.*

---

## 6. Cholesky Identification and Structural Ordering

### 6.1 Identification Strategy

The VAR is identified using a Cholesky decomposition of the reduced-form residual covariance matrix. This imposes a recursive (lower-triangular) structure on the contemporaneous relationships among the variables.

### 6.2 Ordering Justification

The ordering imposed is:

```
MPR → Exchange Rate → Money Supply → Inflation
```

**Formal justification:**

1. **MPR is ordered first** because the Monetary Policy Rate is an administered rate, set by the CBN's Monetary Policy Committee based on forward-looking assessments of macroeconomic conditions. It is not contemporaneously determined by the other variables in the system within the same month. This reflects the standard assumption of **policy exogeneity** in the monetary transmission literature (Christiano, Eichenbaum & Evans, 1999).

2. **Exchange Rate is ordered second** because, as an asset price, the Naira/USD rate responds rapidly to monetary policy signals. A tightening (increase in MPR) is expected to appreciate the Naira through the interest rate parity channel, attracting capital inflows. The exchange rate adjusts within the contemporaneous period to MPR shocks but does not contemporaneously affect the policy rate.

3. **Money Supply (M2) is ordered third** because broad money adjusts more slowly than exchange rates. M2 responds to both the policy rate (through the credit channel and reserve requirements) and the exchange rate (through foreign asset valuation), but its adjustment involves commercial bank portfolio decisions that typically lag within the month.

4. **Inflation is ordered last** because consumer prices are the most sluggish variable in the system. Prices are set by firms with menu costs and information lags, and the CPI is a weighted average of thousands of goods and services. Inflation responds contemporaneously to all other shocks in the system but does not feed back to the other variables within the same month.

This ordering is consistent with the monetary transmission mechanism literature for developing economies and has been applied in studies of Nigerian monetary policy (Adebiyi & Mordi, 2012; Olofin & Afees, 2008).

---

## 7. Impulse Response Functions (IRFs)

### 7.1 Specification
*To be completed — Week 3.*

### 7.2 IRF Plots and Interpretation
*To be completed — Week 3.*

---

## 8. Forecast Error Variance Decomposition (FEVD)

### 8.1 Specification
*To be completed — Week 3.*

### 8.2 FEVD Tables and Interpretation
*To be completed — Week 3.*

---

## 9. Policy Shock Simulation (+100bps MPR)

### 9.1 Simulation Design
*To be completed — Week 4.*

### 9.2 Results and Policy Interpretation
*To be completed — Week 4.*

---

## 10. Results Summary and Policy Implications

*To be completed — final week.*

---

## 11. Limitations and Further Research

*To be completed — final week.*

---

## References

- Christiano, L.J., Eichenbaum, M. & Evans, C.L. (1999). Monetary policy shocks: What have we learned and to what end? *Handbook of Macroeconomics*, 1, 65–148.
- Pesaran, M.H., Shin, Y. & Smith, R.J. (2001). Bounds testing approaches to the analysis of level relationships. *Journal of Applied Econometrics*, 16(3), 289–326.
- Sims, C.A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1–48.
- Adebiyi, M.A. & Mordi, C.N.O. (2012). Monetary policy transmission mechanism in Nigeria. *CBN Understanding Monetary Policy Series*, No. 38.
