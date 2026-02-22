# CECL Credit Risk Modeling Framework
## Model Governance Document

**Document Version:** 1.0  
**Author:** Saurabh Chavan  
**Date:** February 2026  
**Model ID:** CECL-MRT-2026-001  
**Classification:** Credit Risk - Loss Forecasting  
**Status:** Development Complete, Pending Independent Validation

---

## Table of Contents

1. Executive Summary
2. Model Scope and Purpose
3. Data Description
4. Feature Engineering
5. PD Model Methodology
6. LGD Model Methodology
7. EAD Model and ECL Calculation
8. Model Validation Results
9. Stress Testing
10. Monte Carlo Simulation
11. Model Limitations and Assumptions
12. Ongoing Monitoring Plan
13. Appendices

---

## 1. Executive Summary

This document describes an end-to-end CECL-compliant credit risk modeling framework for a single-family residential mortgage portfolio. The framework estimates lifetime Expected Credit Loss (ECL) by independently modeling the three components of credit risk: Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD).

**Portfolio scope:** 3,804,801 loans originated between 2005 and 2007, representing $693.8 billion in total outstanding balance. These vintages span the pre-crisis, crisis, and peak-crisis origination periods, providing a rigorous test of model performance under extreme economic conditions.

**Key results:**

| Metric | Value |
|---|---|
| Portfolio size | 3,804,801 loans, $693.8B outstanding |
| Overall default rate | 14.4% (lifetime, crisis-era vintages) |
| PD model AUC (validation) | 0.6777 (Logistic Regression), 0.7529 (XGBoost) |
| LGD model R-squared (validation) | 0.1508 (OLS), 0.2465 (XGBoost) |
| Mean observed LGD | 38.55% |
| Baseline ECL | $115,562M (16.66% of portfolio) |
| Severely Adverse ECL | $221,865M (31.98% of portfolio) |
| Scenario-Weighted ECL (60/40) | $158,083M (22.78% of portfolio) |
| Monte Carlo VaR 99.9% | $99,208M (14.30% of portfolio) |
| Monte Carlo Expected Shortfall 99% | $89,071M (12.84% of portfolio) |

**Model recommendation:** The logistic regression PD model and OLS LGD model are recommended as the primary production models based on their interpretability, regulatory acceptability, and adequate calibration. XGBoost challenger models demonstrate superior discrimination and serve as performance benchmarks. The primary models are fit for purpose with the limitations noted in Section 11.

---

## 2. Model Scope and Purpose

### 2.1 Business Application

This modeling framework supports three business functions:

**CECL reserve calculation.** ASC 326 (Current Expected Credit Loss) requires financial institutions to estimate lifetime expected losses from the date of loan origination. The ECL output from this framework directly informs the allowance for credit losses reported on financial statements. The framework computes ECL = PD x LGD x EAD across the remaining life of each loan, discounted at the effective interest rate, under probability-weighted economic scenarios.

**Stress testing.** The Federal Reserve's annual CCAR/DFAST exercises require banks to project losses under hypothetical economic scenarios, including a severely adverse scenario. This framework ingests the Fed's 2025 published scenario paths (13 quarters, 2025Q1 through 2028Q1) and projects quarterly portfolio losses under baseline and severely adverse conditions.

**Risk quantification.** Monte Carlo simulation generates a full loss distribution from 10,000 correlated macroeconomic scenarios, producing Value at Risk (VaR) and Expected Shortfall (ES) metrics that inform economic capital adequacy assessment.

### 2.2 Portfolio Covered

The model applies to fixed-rate, fully amortizing single-family residential mortgages acquired by Fannie Mae. The portfolio consists of conforming conventional loans with the following characteristics:

| Characteristic | Range |
|---|---|
| Origination vintages | 2005Q1 through 2007Q4 |
| Loan size | Conforming limits (up to ~$417K in this period) |
| Property types | Single-family, condos, manufactured housing, 2-4 unit |
| Amortization | Fixed-rate, fully amortizing (FRM) |
| Geography | All 50 US states |

### 2.3 Regulatory Context

The framework addresses requirements under ASC 326 (CECL), the OCC's Supervisory Guidance on Model Risk Management (OCC 2011-12 / SR 11-7), and the Federal Reserve's Comprehensive Capital Analysis and Review (CCAR) / Dodd-Frank Act Stress Testing (DFAST) programs.

### 2.4 Model Tier

Given the portfolio materiality ($693.8B) and the direct impact on financial statement reserves, this model would be classified as Tier 1 (Critical) under a standard model risk governance framework, requiring annual independent validation and quarterly performance monitoring.

---

## 3. Data Description

### 3.1 Data Source

**Primary data:** Fannie Mae Single-Family Loan Performance Dataset, accessed through the Fannie Mae Data Dynamics portal. This dataset contains both origination-level attributes and monthly performance history for every loan in the Fannie Mae single-family credit guarantee book.

**Format:** Pipe-delimited text files with no header row. Each row represents one loan observed at one monthly reporting period, containing 108 data fields as defined in the Fannie Mae Single-Family Glossary and File Layout (February 2026 edition).

**Macroeconomic data:** Federal Reserve Economic Data (FRED) API, providing monthly observations for 8 economic series: unemployment rate (UNRATE), federal funds rate (FEDFUNDS), CPI (CPIAUCSL), 30-year mortgage rate (MORTGAGE30US), Case-Shiller National Home Price Index (CSUSHPINSA), 10-year Treasury yield (GS10), BAA corporate bond spread (BAA10Y), and GDP.

**Stress test scenarios:** Federal Reserve Board 2025 Stress Test Scenarios, providing quarterly macro variable paths for baseline and severely adverse scenarios across 13 quarters (2025Q1 through 2028Q1).

### 3.2 Data Volume

The raw dataset comprises 265,586,127 loan-month observations across 12 quarterly files:

| Quarter | Rows | Unique Loans |
|---|---|---|
| 2005Q1 | 24,674,933 | 303,611 |
| 2005Q2 | 27,054,781 | 339,372 |
| 2005Q3 | 35,640,066 | 440,521 |
| 2005Q4 | 29,612,454 | 378,311 |
| 2006Q1 | 18,159,238 | 253,043 |
| 2006Q2 | 19,695,956 | 291,165 |
| 2006Q3 | 16,523,630 | 271,373 |
| 2006Q4 | 18,020,864 | 280,979 |
| 2007Q1 | 16,736,975 | 253,279 |
| 2007Q2 | 18,562,209 | 287,263 |
| 2007Q3 | 18,268,039 | 314,703 |
| 2007Q4 | 22,636,982 | 391,181 |
| **Total** | **265,586,127** | **3,804,801** |

After loan-level aggregation, the modeling dataset contains 3,804,801 loans with 68 feature columns.

### 3.3 Data Ingestion Pipeline

Raw files are processed using a chunked reading approach (500,000 rows per chunk) to manage memory constraints. The pipeline selects 42 of 108 available fields relevant to credit risk modeling and converts each quarterly file to compressed Parquet format. Column alignment is verified programmatically against the glossary mapping (raw_index = glossary_field_number - 1) with automated checks on loan_id format, channel codes, property type codes, and FICO score ranges.

### 3.4 Data Quality Assessment

Data quality profiling was performed on a representative quarter (2006Q1, 253,043 loans):

| Feature | Missing Rate | Notes |
|---|---|---|
| borrower_credit_score | 0.55% | Very low; missing loans excluded or imputed |
| original_ltv | 0.00% | Complete |
| dti | 2.98% | Missing may indicate stated-income loans |
| coborrower_credit_score | 54.5% | Expected: many loans are single-borrower |
| mortgage_insurance_pct | 86.2% | Expected: most loans have LTV <= 80% |
| Loss fields (foreclosure costs, net sale proceeds, etc.) | 99.9% | Normal: only populated after default and disposition |

**Key distributional statistics (full dataset):**
- Mean FICO: 722, range 300-850
- Median LTV: 74%
- Mean loan size: $182,000
- Median loan term: 360 months (30-year fixed)

### 3.5 Default Definition

A loan is classified as defaulted if, at any point during its observed life, it meets either condition:

1. **Delinquency threshold:** Current loan delinquency status reaches 3 or higher (90+ days past due), consistent with the Basel II/III definition of default.
2. **Credit loss termination:** The loan exits the pool with a zero balance code indicating a credit loss event: 02 (Third Party Sale), 03 (Short Sale), 06 (Repurchased due to serious delinquency), 09 (REO Disposition), or 15 (Non-Performing Note Sale).

This is a lifetime binary default flag (ever-defaulted vs. never-defaulted), not a point-in-time indicator. The resulting default rate is 14.4% across the full portfolio, which is consistent with expectations for 2005-2007 vintage mortgages that experienced the 2008 financial crisis.

### 3.6 LGD Calculation

LGD is computed for defaulted loans using the realized loss approach:

```
Total Loss = EAD - Total Recovery + Total Costs
LGD = Total Loss / EAD
```

Where:
- **EAD** = Current actual UPB from the second-to-last monthly observation. Fannie Mae sets UPB to zero in the final observation when a loan is removed from the pool; therefore, the second-to-last observation captures the true exposure at default.
- **Total Recovery** = net_sale_proceeds + credit_enhancement_proceeds + repurchase_make_whole_proceeds + other_foreclosure_proceeds.
- **Total Costs** = foreclosure_costs + property_preservation_costs + asset_recovery_costs + misc_holding_expenses + holding_taxes.

**Data handling notes:**
- Fannie Mae encodes net_sale_proceeds as "C" (string) when the amount is confidential. These records are excluded from LGD computation (treated as missing, not zero).
- LGD is floored at 0.0 (negative LGD from insurance recoveries exceeding balance) and capped at 1.5 (extreme cost overruns).
- 278,220 defaulted loans have valid LGD values.
- Mean observed LGD: 0.3855, median: 0.3678.

### 3.7 Train/Validation/Test Split

A time-based split is used, consistent with credit risk modeling best practices. Random splits would allow information leakage from future economic conditions into training data.

| Split | Vintages | Loans | Default Rate | Purpose |
|---|---|---|---|---|
| Train | 2005 + pre-2005 | 1,613,840 | 11.75% | Model development |
| Validation | 2006 | 1,079,504 | 15.26% | Out-of-time hyperparameter selection |
| Test | 2007 | 1,111,457 | 17.41% | Out-of-time final performance assessment |

The increasing default rate across splits reflects the deteriorating origination quality as the housing bubble inflated. The test set (2007 vintages) provides the most severe performance test, as these loans entered the crisis with minimal seasoning.

---

## 4. Feature Engineering

### 4.1 Feature Construction

The feature engineering pipeline transforms 265 million loan-month observations into a 3.8 million loan-level dataset by extracting static origination features, computing default and LGD targets, merging macroeconomic conditions at origination, and creating derived risk indicators. Processing is performed one quarter at a time to accommodate memory constraints (8-16GB RAM).

**Static origination features** are extracted from the first monthly observation per loan using groupby aggregation. These represent the loan's characteristics at the time of underwriting and do not change over the loan's life.

**Binary flags** are derived from data patterns that carry risk information:
- `has_coborrower`: derived from coborrower_credit_score non-missingness. Dual-income borrowers have greater resilience to income shocks.
- `has_mortgage_insurance`: derived from mortgage_insurance_pct non-missingness. MI reduces LGD by transferring a portion of loss to the insurer.
- `is_first_time_buyer`, `is_investment_property`, `is_second_home`, `is_cashout_refi`, `is_condo`, `is_manufactured_housing`: binary indicators from categorical fields, each capturing a known risk dimension.

**Categorical bins** segment continuous risk drivers into industry-standard tiers:
- FICO buckets: <620 (subprime), 620-660, 660-700, 700-740, 740+ (super-prime)
- LTV buckets: <60, 60-70, 70-80, 80-90, 90+
- DTI buckets: <20, 20-30, 30-40, 40-50, 50+

**Interaction features** capture non-linear risk combinations:
- `fico_x_ltv`: Low FICO combined with high LTV represents the highest-risk borrower profile (weak credit and minimal equity).
- `fico_x_dti`: Low FICO combined with high DTI indicates a borrower who is both credit-impaired and heavily leveraged on income.
- `fico_x_unemployment`: Borrower quality interacted with the macroeconomic environment at origination. Weak borrowers are disproportionately affected by labor market deterioration.

**Macroeconomic features** are merged by origination month from FRED data. Each loan inherits the economic conditions prevailing when it was originated, including unemployment rate, federal funds rate, 30-year mortgage rate, Case-Shiller HPI, 10-year Treasury yield, BAA corporate spread, and derived variables (12-month unemployment change, 12-month HPI change, GDP growth rate).

### 4.2 Missing Value Treatment

| Feature | Missing Rate | Treatment | Rationale |
|---|---|---|---|
| borrower_credit_score | 0.55% | Median imputation (training set median) | Low missing rate; standard imputation |
| dti | 2.98% | Median imputation + `dti_missing` indicator | Missingness itself is informative (possible stated-income loan) |
| coborrower_credit_score | 54.5% | Not imputed; used to derive `has_coborrower` flag | Missingness means single borrower, not data error |
| mortgage_insurance_pct | 86.2% | Not imputed; used to derive `has_mortgage_insurance` flag | Missingness means no MI on loan |
| Macro features | <1% | Forward-fill from prior month | Standard for time series gaps |

---

## 5. PD Model Methodology

### 5.1 Modeling Approach

The primary PD model is a logistic regression trained on Weight of Evidence (WoE) transformed features. This combination is the industry standard for credit risk scorecards and was selected for the following reasons:

**Interpretability.** Logistic regression coefficients directly quantify each feature's contribution to default risk. When model risk reviewers ask "why is this loan rated high-risk?", the answer is traceable to specific coefficient magnitudes. Each WoE-transformed feature's coefficient represents the log-odds impact of moving from one risk bin to another.

**Monotonicity.** WoE transformation guarantees monotonic relationships between features and the log-odds of default. Higher FICO always produces lower risk scores, with no exceptions. Tree-based models can produce non-monotonic relationships that are difficult to justify economically.

**Stability.** Logistic regression coefficients remain stable across time periods and resampling, unlike tree-based models where small data perturbations can produce substantially different tree structures.

**Regulatory acceptance.** The OCC, Federal Reserve, and FDIC have decades of supervisory experience with logistic regression for credit risk. Using it reduces model risk review friction and aligns with examiner expectations.

An XGBoost gradient-boosted classifier serves as the challenger model, providing a benchmark for the discrimination gap inherent in the simpler primary model.

### 5.2 Weight of Evidence Transformation

For each feature, the WoE transformation proceeds as follows:

1. **Binning.** Continuous features are divided into decile bins using quantile-based binning (`pd.qcut` with 10 bins). Categorical features use each unique value as a bin. Bins with fewer than 5% of observations are flagged for potential instability.

2. **WoE calculation.** For each bin:
   ```
   WoE(bin) = ln(% of Non-Defaults in bin / % of Defaults in bin)
   ```
   Laplace smoothing (additive constant of 0.5) is applied to prevent division by zero and log(0) when a bin contains zero events or zero non-events.

3. **Feature replacement.** Original feature values are replaced by their bin's WoE value.

**Interpretation:** Positive WoE indicates a bin with proportionally more non-defaults (low risk). Negative WoE indicates proportionally more defaults (high risk). The magnitude reflects signal strength.

**Critical implementation detail:** When applying the WoE transformation to validation and test data, the bin edges from training data are preserved using `pd.cut` with fixed boundaries. Re-binning new data with `pd.qcut` would create different bin boundaries (because the validation/test distributions differ), assigning loans to incorrect WoE values and degrading model discrimination.

### 5.3 Information Value and Feature Selection

Information Value (IV) quantifies each feature's overall predictive power for separating defaults from non-defaults:

```
IV = SUM over bins: (% Non-Defaults in bin - % Defaults in bin) x WoE(bin)
```

**IV interpretation thresholds:**
| IV Range | Interpretation |
|---|---|
| < 0.02 | Not useful (drop) |
| 0.02 - 0.10 | Weak predictor |
| 0.10 - 0.30 | Medium predictor |
| 0.30 - 0.50 | Strong predictor |
| > 0.50 | Suspiciously strong (investigate for leakage) |

Starting from 38 candidate features (20 continuous + 18 categorical), the following 10 features were selected based on IV thresholds and redundancy removal:

| Feature | IV | Interpretation | Rationale |
|---|---|---|---|
| borrower_credit_score | 0.60 | Strong/Suspicious | Strongest single predictor; IV above 0.50 investigated and confirmed legitimate |
| fico_x_unemployment | 0.58 | Strong/Suspicious | Captures borrower-macro interaction; confirmed not leakage |
| original_ltv | 0.20 | Medium | Equity/leverage measure |
| original_cltv | 0.17 | Medium | Total leverage including second liens |
| original_interest_rate | 0.15 | Medium | Proxy for underwriter's risk assessment |
| dti | 0.13 | Medium | Payment capacity |
| original_loan_term | 0.11 | Medium | Term structure risk |
| has_mortgage_insurance | 0.06 | Weak | Loss mitigation indicator |
| is_cashout_refi | 0.04 | Weak | Equity extraction risk |
| has_coborrower | 0.04 | Weak | Income stability |

**Redundancy removal decisions:**
- `fico_bucket`, `ltv_bucket`, `dti_bucket`: Removed as they duplicate information already captured by the continuous versions with less granularity.
- `fico_x_ltv`, `fico_x_dti`: Removed as partially redundant with their component features. The `fico_x_unemployment` interaction is retained because it uniquely captures the macro-borrower channel.
- `number_of_borrowers`: Redundant with `has_coborrower`.
- `gdp_growth_pct`, `fed_funds_rate`, `hpi_national`: IV values near 0.02 on full training data; dropped for parsimony. Their economic signal is partially captured through `fico_x_unemployment` and `original_interest_rate`.

### 5.4 Model Specification

**Primary model (Logistic Regression):**
- Penalty: L2 (ridge regularization)
- Regularization strength: C = 1.0
- Class weight: None (no reweighting). This deliberate choice preserves calibration. Using `class_weight='balanced'` would inflate predicted probabilities for the minority class, producing miscalibrated PD estimates that overstate reserves.
- Solver: L-BFGS
- Input features: 10 WoE-transformed features

**Challenger model (XGBoost):**
- Trees: 300 estimators
- Max depth: 4 (kept shallow for partial interpretability)
- Learning rate: 0.05
- Class imbalance: scale_pos_weight = ratio of non-defaults to defaults
- Input features: 10 original (non-WoE) features, with label encoding for any categoricals
- Evaluation metric: log-loss

### 5.5 Coefficient Interpretation

All logistic regression coefficients are reported on WoE-transformed features. Because WoE is monotonically related to default risk by construction, a positive coefficient on a WoE feature means that higher WoE (i.e., lower risk bins) reduces the predicted probability of default, which is the economically expected direction.

The intercept captures the baseline log-odds of default when all WoE features are at zero (the "average" bin for each feature).

---

## 6. LGD Model Methodology

### 6.1 Modeling Approach

The primary LGD model is an OLS linear regression. The challenger is an XGBoost gradient-boosted regressor.

**Why OLS instead of Beta regression?** While LGD is theoretically bounded on [0, 1], the observed data includes LGD values up to 1.5 (foreclosure costs exceeding the outstanding balance). Beta regression requires values strictly in the open interval (0, 1) and cannot accommodate these observations without transformation. OLS with post-hoc prediction clipping to [0, 1.5] is the pragmatic industry approach and produces directly interpretable coefficients that are straightforward to document and explain to model risk reviewers.

**Training population:** Only the 278,220 defaulted loans with valid LGD values. Non-defaulted loans are excluded because they have no observed loss outcome. This is the standard approach for workout LGD modeling.

### 6.2 Feature Selection

Thirteen features are selected based on economic intuition about what drives loss severity after default:

| Feature | Economic Rationale |
|---|---|
| original_ltv | Higher LTV at origination means less equity buffer. Strongest LGD driver. |
| original_cltv | Combined LTV includes second liens, indicating total leverage. |
| has_mortgage_insurance | MI covers a portion of the loss, reducing LGD by approximately 15 percentage points. |
| borrower_credit_score | Lower FICO borrowers tend to delay foreclosure, accumulating costs. |
| dti | Higher DTI borrowers have less capacity to cure a default. |
| original_interest_rate | Proxy for unmeasured risk factors identified by the original underwriter. |
| original_upb | Larger loans may have different LGD profiles due to property market segment. |
| loan_age_at_default | Early defaults (within 2 years) differ from seasoned defaults. Early defaults may indicate fraud. |
| was_modified | Modified loans that re-default may have higher LGD because concessions were already made. |
| is_investment_property | Investors are more strategic about walking away. |
| is_condo | Condos may have different resale liquidity compared to single-family homes. |
| unemployment_rate | Higher unemployment at default correlates with worse recovery rates (fewer buyers). |
| hpi_national | Combined with current HPI, determines equity position. Loans originated at peak HPI (2006) had worst LGD. |

Missing values are filled with training set medians. For features specific to defaulted loans (`loan_age_at_default`, `was_modified`), median values from the training population of defaulted loans are used when scoring non-defaulted loans for ECL purposes.

### 6.3 Model Output

LGD predictions are clipped to [0, 1.0] for ECL calculation (capped at 100% loss). The raw model can produce values outside this range, which reflects the theoretical possibility of costs exceeding the balance, but for reserve computation a 100% loss cap is applied.

---

## 7. EAD Model and ECL Calculation

### 7.1 Exposure at Default

For fixed-rate fully-amortizing mortgages, EAD at any future month t is the scheduled remaining balance:

```
Balance(t) = P x [(1+r)^n - (1+r)^t] / [(1+r)^n - 1]
```

Where P is the original principal balance, r is the monthly interest rate (annual rate / 12), n is the total number of payments, and t is the number of payments already made. This is the standard amortization formula; no statistical model is required because the payment schedule is contractually determined.

### 7.2 PD Term Structure

The PD model produces a 12-month probability of default. CECL requires lifetime loss estimation, so the 12-month PD is extended to a monthly marginal PD curve using an empirical seasoning pattern:

- Months 1-12: Ramp-up phase (multipliers 0.4 to 1.0)
- Months 13-24: Continuing ramp (multipliers 1.05 to 1.30)
- Months 25-36: Approaching peak (multipliers 1.10 to 1.00)
- Months 37-48: Peak default period (multipliers 1.00 to 0.75)
- Months 49-60: Decline phase (multipliers 0.70 to 0.35)
- Months 61+: Exponential decay with floor at 5% of base rate

The curve is scaled so that the first 12 months sum to the PD model's 12-month estimate. Monthly hazard rates are converted to marginal PDs using a survival function to ensure cumulative PD never exceeds 100%:

```
Survival(t) = Survival(t-1) x (1 - hazard(t-1))
Marginal_PD(t) = hazard(t) x Survival(t)
```

Monthly hazard rates are capped at 5% to prevent numerical instability.

### 7.3 ECL Calculation

For each loan, lifetime ECL is computed as:

```
ECL = SUM over t=1 to remaining_term:
    Marginal_PD(t) x LGD x EAD(t) x Discount_Factor(t)
```

Where Discount_Factor(t) = 1 / (1 + r_monthly)^t, using the loan's contractual interest rate as the effective interest rate per CECL guidance. LGD is assumed constant over the loan's remaining life for computational simplicity.

### 7.4 Scenario Weighting

CECL requires consideration of multiple economic scenarios weighted by their probability of occurrence:

```
ECL_final = 0.60 x ECL_baseline + 0.40 x ECL_severely_adverse
```

The 60/40 weighting reflects a judgment that the baseline economic path is more probable but assigns meaningful weight to downside risk. In production, these weights would be reviewed quarterly by the Economic Scenarios Committee.

**Scenario-level ECL results:**

| Scenario | Total ECL | ECL Rate |
|---|---|---|
| Baseline | $115,562M | 16.66% |
| Severely Adverse | $221,865M | 31.98% |
| Weighted (60/40) | $158,083M | 22.78% |

---

## 8. Model Validation Results

### 8.1 PD Model Discrimination

Discrimination measures the model's ability to rank-order borrowers by risk level.

| Metric | Logistic Regression | XGBoost Challenger |
|---|---|---|
| Train AUC | 0.7048 | 0.7579 |
| Validation AUC (2006) | 0.6777 | 0.7529 |
| Test AUC (2007) | 0.6838 | 0.7606 |
| Train KS | 0.3142 | 0.3893 |
| Validation KS | 0.2651 | 0.3853 |

**Assessment:** The logistic regression validation AUC of 0.6777 is below the 0.75 threshold typically considered "good" for credit risk. However, this is explained by three factors: (1) the WoE transformation with only 10 features sacrifices some discrimination for interpretability, (2) the out-of-time validation set (2006 vintages) experienced regime change relative to 2005 training data, and (3) the XGBoost challenger confirms that additional discrimination is available through non-linear modeling (validation AUC 0.7529), bounding the interpretability cost at approximately 7.5 AUC points.

The KS statistic for the logistic regression on validation (0.2651) falls slightly below the 0.30 guideline, consistent with the AUC finding. The model still provides meaningful rank ordering.

### 8.2 PD Model Calibration

Calibration measures whether predicted probabilities match observed default rates. For CECL, calibration is more critical than discrimination because predicted PDs directly determine dollar reserves.

The calibration table (validation set, decile analysis) shows predicted PD versus actual default rate across risk deciles. The logistic regression model was trained without class reweighting specifically to preserve calibration. The ratio of actual to predicted default rate (calibration ratio) remains within acceptable bounds across deciles, confirming the model produces probability estimates suitable for reserve computation.

### 8.3 PD Model Stability

Population Stability Index (PSI) measures score distribution shift between populations:

| Comparison | Logistic Regression | XGBoost |
|---|---|---|
| PSI: Train vs Validation | 0.2916 | 0.1880 |
| PSI: Train vs Test | -- | -- |

**Assessment:** The logistic regression PSI of 0.2916 between training (2005 vintages) and validation (2006 vintages) exceeds the 0.25 threshold indicating significant population shift. This is expected given the structural change in origination quality between 2005 and 2006 as underwriting standards deteriorated during the housing bubble. The XGBoost PSI of 0.1880 indicates greater score stability, partly because tree-based models adapt more flexibly to distributional shifts in input features.

This elevated PSI is a known limitation and would trigger enhanced monitoring in production. For this historical dataset, the shift reflects genuine economic regime change rather than model deficiency.

### 8.4 LGD Model Performance

| Metric | OLS | XGBoost |
|---|---|---|
| Train R-squared | 0.1702 | 0.2985 |
| Validation R-squared | 0.1508 | 0.2465 |
| Test R-squared | 0.1574 | 0.2401 |
| Validation Calibration (mean predicted / mean actual) | 1.05 | 0.99 |

**Assessment:** R-squared values of 0.15-0.25 are realistic and expected for LGD modeling. Loss severity has high inherent variance driven by property-specific, market-timing, and legal-process factors that are not observable in origination-level data. The academic literature on mortgage LGD consistently reports R-squared values in this range. The OLS model's validation calibration ratio of 1.05 indicates the model slightly overpredicts LGD on average (5% conservative bias), which is preferred to underprediction from a prudential perspective. The XGBoost challenger achieves near-perfect calibration (0.99) with meaningfully higher R-squared.

### 8.5 Challenger Model Comparison

The XGBoost challenger consistently outperforms the logistic regression/OLS primary models on discrimination metrics while maintaining equal or better calibration. The performance gap is:

- PD: +7.5 AUC points (validation), +7.7 AUC points (test)
- LGD: +9.6 R-squared points (validation), +8.3 R-squared points (test)

This gap represents the cost of interpretability. The primary models are recommended for production because: (1) coefficient-level interpretability supports model risk review and regulatory examination, (2) the calibration of the primary models is adequate for reserve computation, and (3) the challenger models serve their intended purpose as performance benchmarks. If discrimination becomes insufficient for business needs, a transition to the XGBoost models could be considered with enhanced documentation of feature importance and partial dependence analysis.

---

## 9. Stress Testing

### 9.1 Methodology

The stress testing module uses a macro-sensitivity overlay approach rather than direct macro variable substitution. This design choice was driven by a fundamental challenge: the PD and LGD models use macroeconomic features at the time of origination (e.g., unemployment rate when the loan was underwritten). These origination-time features have opposite economic interpretations from point-in-time features.

For example, a loan originated when unemployment was 4% and HPI was at its peak (2006) is actually riskier than a loan originated during moderate conditions, because peak-HPI origination implies the borrower bought at the top of the market. Substituting a stressed unemployment rate of 10% into the origination-time feature would incorrectly suggest the loan was originated during a recession (and thus underwritten more conservatively).

The overlay approach resolves this by:
1. Scoring the portfolio once under current conditions to establish baseline PD and LGD for each loan.
2. Computing stress multipliers based on how much scenario macro variables deviate from baseline conditions.
3. Applying multipliers to baseline PD and LGD to produce stressed estimates.

### 9.2 Multiplier Calibration

**PD multiplier:** +25% per +1 percentage point of unemployment above baseline, plus +5% per -1 percentage point of GDP growth. This calibration is consistent with the empirical relationship observed during the 2008 crisis, where approximately 5 percentage points of unemployment increase roughly doubled default rates.

**LGD multiplier:** +15% per -10 percentage points of HPI decline from baseline, equivalent to +1.5% per -1% HPI. This reflects the 2008 experience where a roughly 33% national HPI decline approximately doubled loss severity.

Multipliers are floored at 0.5x (no scenario improves losses by more than 50%) and capped at 5.0x for PD and 3.0x for LGD to prevent extreme extrapolation.

### 9.3 Stress Test Results

| Scenario | 13-Quarter Total Loss | Loss Rate |
|---|---|---|
| Baseline | $99,615M | 14.36% |
| Severely Adverse | $311,945M | 44.96% |
| Stress Increment | +$212,330M | +30.60pp |

**Peak stress quarter:** 2025Q4, with PD multiplier of 2.52x and LGD multiplier of 1.46x, corresponding to peak unemployment of 10% and HPI trough of approximately -33% from baseline.

**Segment-level findings:** The stress increment is not uniform across the portfolio. Subprime borrowers (FICO < 620) with high LTV (> 90%) experience the largest absolute and relative stress increment, consistent with the 2008 experience where this segment sustained catastrophic losses. Prime borrowers (FICO > 740) with low LTV (< 60%) show the smallest stress multiplier, confirming portfolio resilience in the highest-quality segment.

---

## 10. Monte Carlo Simulation

### 10.1 Methodology

Monte Carlo simulation generates 10,000 correlated random macroeconomic scenarios to produce a full loss distribution. The simulation uses three macro variables: unemployment rate, annual HPI change, and annual GDP growth.

**Correlated scenario generation:** Independent standard normal random draws are transformed into correlated draws using Cholesky decomposition of the historical correlation matrix. This ensures that simulated scenarios reflect observed macro relationships (e.g., high unemployment co-occurs with falling home prices and negative GDP growth). The historical correlation matrix is estimated from the full FRED monthly dataset (312 observations).

**Loss computation:** For each simulation, correlated macro draws are converted to PD and LGD multipliers using the same calibration as the stress testing module. Portfolio loss equals the sum of multiplier-adjusted baseline expected losses across all loans.

### 10.2 Risk Metrics

| Metric | Value | % of Portfolio |
|---|---|---|
| Expected Loss (mean) | $44,454M | 6.41% |
| Standard Deviation | $16,000M | -- |
| VaR 99% | $81,805M | 11.79% |
| VaR 99.9% | $99,208M | 14.30% |
| Expected Shortfall 99% | $89,071M | 12.84% |

**Loss distribution percentiles:**

| Percentile | Loss | % of Portfolio |
|---|---|---|
| 5th | $22,000M | 3.2% |
| 25th | $32,000M | 4.6% |
| 50th (Median) | $42,000M | 6.1% |
| 75th | $54,000M | 7.8% |
| 95th | $74,000M | 10.7% |

### 10.3 Sensitivity Analysis

One-at-a-time sensitivity analysis identifies which macro variable drives the most portfolio risk:

**Unemployment rate** is the dominant risk driver. Increasing unemployment from 4% to 12% produces a 2.9x increase in portfolio loss, primarily through the PD channel. This confirms the model's economic intuition: job loss is the primary trigger for mortgage default.

**HPI decline** is the secondary risk driver, operating primarily through the LGD channel. A -30% HPI decline increases portfolio loss by approximately 1.5x through higher loss severity on each default.

**GDP growth** has the smallest marginal impact, as its effect on PD is modest relative to the unemployment channel.

---

## 11. Model Limitations and Assumptions

### 11.1 Data Limitations

**Fannie Mae scope.** The model is trained exclusively on conforming conventional mortgages acquired by Fannie Mae. It does not capture risk dynamics of jumbo loans, FHA/VA government-insured loans, non-QM products, or commercial real estate. Application to portfolios outside this scope would require re-development.

**Vintage concentration.** Training data covers 2005-2007 originations, a period of historically poor underwriting standards. The model may overweight crisis-era risk factors. Performance on modern vintage loans (post-Dodd-Frank, post-QM rule) has not been validated.

**Survivorship in FRED data.** Macroeconomic time series represent national aggregates. State-level or MSA-level economic variation is not captured.

### 11.2 Modeling Assumptions

**LGD constancy over loan life.** The ECL calculation assumes LGD does not vary with the timing of default. In practice, LGD may be higher for early defaults (less equity buildup) and lower for late defaults (more amortization).

**PD term structure shape.** The seasoning curve used to extend 12-month PD to lifetime is based on historical mortgage default patterns and is assumed to apply uniformly across risk segments. In practice, subprime loans may peak earlier than prime loans.

**Stress multiplier linearity.** The macro-sensitivity overlay assumes linear relationships between macro variable changes and PD/LGD multipliers. Actual relationships may be non-linear, with accelerating effects at extreme stress levels.

**Monte Carlo homogeneity.** The Monte Carlo simulation applies the same macro multiplier to all loans in each scenario. In practice, geographic concentration would cause some loans to experience more severe local conditions than the national average.

**Static portfolio.** The analysis treats the portfolio as static (no new originations, no runoff). In production, portfolio composition changes monthly through new originations and payoffs.

### 11.3 Known Model Weaknesses

**PD discrimination.** The logistic regression validation AUC of 0.6777 is below the typical 0.75 target. This is partially mitigated by acceptable calibration, but enhanced feature engineering or consideration of the XGBoost challenger for production could improve rank ordering.

**PSI instability.** The 0.29 PSI between training and validation populations reflects genuine regime change but would trigger a recalibration review in production.

**Geographic risk.** The model does not include state-level or MSA-level features, missing geographic concentration risk. During the 2008 crisis, losses were heavily concentrated in sand states (California, Nevada, Arizona, Florida).

**Prepayment modeling.** The EAD model uses contractual amortization schedules without prepayment adjustment. Actual exposures will differ from scheduled balances due to refinancing activity, particularly in falling-rate environments.

---

## 12. Ongoing Monitoring Plan

### 12.1 Monthly Monitoring

- **Predicted vs. actual default rates** by score decile and calendar month. A sustained deviation exceeding 20% in any decile triggers investigation.
- **Mean predicted PD** tracked as a time series. Sudden shifts indicate population change or data quality issues.

### 12.2 Quarterly Monitoring

- **PSI computation** comparing the current quarter's score distribution to the reference (training) distribution. PSI exceeding 0.10 triggers enhanced review; PSI exceeding 0.25 triggers recalibration assessment.
- **Calibration table refresh** comparing predicted PD to observed default rate by decile on the most recent 12-month outcome window.
- **LGD backtesting** comparing predicted LGD to realized LGD on newly resolved defaults.
- **ECL reasonableness check** comparing model-driven reserves to industry benchmarks and peer analysis.

### 12.3 Annual Activities

- **Full model re-development assessment** including re-estimation of WoE bins, IV recalculation, coefficient re-fitting, and challenger model refresh.
- **Macro scenario review** incorporating the latest Federal Reserve stress test scenarios and updated historical macro statistics for Monte Carlo calibration.
- **Independent validation** by the model risk management challenge function, reviewing all aspects of this document.

### 12.4 Trigger-Based Actions

| Trigger | Threshold | Action |
|---|---|---|
| PSI (monthly score distribution) | > 0.25 | Recalibration assessment within 30 days |
| Calibration deviation (any decile) | > 20% actual/predicted ratio deviation | Root cause investigation |
| AUC degradation | Drop > 0.03 from development | Re-development assessment |
| LGD mean bias | > 10% predicted/actual ratio deviation | LGD model recalibration |
| Macro regime change | Unemployment change > 2pp in 6 months | Stress multiplier recalibration |

---

## 13. Appendices

### Appendix A: Software and Dependencies

| Component | Version |
|---|---|
| Python | 3.9 |
| pandas | Latest stable |
| numpy | Latest stable |
| scikit-learn | Latest stable |
| xgboost | Latest stable |
| fredapi | Latest stable |
| joblib | Latest stable |
| pyarrow | Latest stable |

### Appendix B: Model Artifacts

| Artifact | Filename | Description |
|---|---|---|
| PD Logistic Regression | pd_logistic_regression.pkl | Trained primary PD model |
| PD XGBoost | pd_xgboost.pkl | Trained challenger PD model |
| WoE Mappings | woe_results.pkl | WoE bin definitions and values (critical for scoring) |
| XGBoost Encoders | xgb_label_encoders.pkl | Label encoders for XGBoost categorical features |
| LGD OLS | lgd_ols.pkl | Trained primary LGD model |
| LGD XGBoost | lgd_xgboost.pkl | Trained challenger LGD model |
| PD Feature List | selected_features.txt | 10 PD features |
| LGD Feature List | lgd_features.txt | 13 LGD features |
| IV Summary | iv_summary.csv | Information Value for all 38 candidate features |
| LR Coefficients | lr_coefficients.csv | Logistic regression coefficients |
| LGD Coefficients | lgd_ols_coefficients.csv | OLS regression coefficients |
| Calibration Table | lr_calibration_validation.csv | PD calibration by decile |
| Validation Summary | validation_summary.csv | PD model comparison metrics |
| LGD Validation | lgd_validation_summary.csv | LGD model comparison metrics |
| ECL Summary | ecl_summary.csv | Portfolio ECL by scenario |
| ECL Loan-Level | ecl_loan_level.parquet | Loan-level ECL results |
| Stress Test Results | stress_test_summary.csv, stress_severely_adverse_quarterly.csv | Scenario comparison and quarterly path |
| Monte Carlo Results | mc_loss_distribution.csv, mc_scenarios.csv, mc_risk_metrics.csv, mc_sensitivity.csv | Full simulation outputs |

### Appendix C: Code Repository Structure

```
cecl-credit-loss-modeling/
    src/
        data_pipeline.py          Phase 1: Raw data ingestion
        feature_engine.py         Phase 2: Feature engineering
        pd_model.py               Phase 3: PD model functions
        run_pd_model.py           Phase 3: PD training pipeline
        lgd_model.py              Phase 4: LGD model functions
        run_lgd_model.py          Phase 4: LGD training pipeline
        ecl_engine.py             Phase 5-6: EAD and ECL calculation
        run_ecl.py                Phase 6: ECL pipeline
        stress_testing.py         Phase 7: Stress testing functions
        run_stress_test.py        Phase 7: Stress testing pipeline
        monte_carlo.py            Phase 8: Monte Carlo functions
        run_monte_carlo.py        Phase 8: Monte Carlo pipeline
    models/                       Trained model artifacts
    data/
        raw/                      Fannie Mae quarterly CSVs
        processed/                Parquet files and loan-level dataset
        scenarios/                Fed 2025 stress test scenarios
    docs/
        model_documentation.md    This document
    dashboard/                    Streamlit dashboard (Phase 10)
```

### Appendix D: Glossary

| Term | Definition |
|---|---|
| AUC | Area Under the Receiver Operating Characteristic Curve |
| CECL | Current Expected Credit Loss (ASC 326) |
| CCAR | Comprehensive Capital Analysis and Review |
| CLTV | Combined Loan-to-Value ratio |
| DFAST | Dodd-Frank Act Stress Testing |
| DTI | Debt-to-Income ratio |
| EAD | Exposure at Default |
| ECL | Expected Credit Loss |
| ES | Expected Shortfall (Conditional VaR) |
| FICO | Fair Isaac Corporation credit score |
| HPI | Home Price Index |
| IV | Information Value |
| KS | Kolmogorov-Smirnov statistic |
| LGD | Loss Given Default |
| LTV | Loan-to-Value ratio |
| MI | Mortgage Insurance |
| OLS | Ordinary Least Squares |
| PD | Probability of Default |
| PSI | Population Stability Index |
| UPB | Unpaid Principal Balance |
| VaR | Value at Risk |
| WoE | Weight of Evidence |

---

*End of Model Governance Document*
