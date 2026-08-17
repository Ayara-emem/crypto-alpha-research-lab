# Crypto Alpha Research Laboratory (CARL)

# Research Methodology

## 1. Research Objective

CARL investigates whether the choice of covariance estimator materially
affects portfolio performance, risk, implementation characteristics,
and robustness when evaluated out of sample.

The research compares:

1. Sample covariance
2. Fixed-intensity shrinkage
3. Ledoit-Wolf shrinkage

All methodologies are evaluated within a global minimum-variance
portfolio framework.

---

## 2. Research Universe

The empirical analysis uses cryptocurrency market data.

The primary research universe consists of:

- Bitcoin (BTC)
- Ethereum (ETH)
- Solana (SOL)

Historical daily OHLCV data are loaded through CARL's data layer,
validated, normalized, and aligned before being used in the research
pipeline.

---

## 3. Data Preparation

CARL applies a controlled data pipeline before research begins.

The process is:

Raw Market Data
→ Data Loading
→ Normalization
→ Validation
→ Chronological Ordering
→ Duplicate Removal
→ Research Dataset

The research dataset is therefore separated from the modelling layer.

This helps ensure that downstream experiments operate on a consistent
data contract.

---

## 4. Return Construction

Portfolio research is performed using asset returns derived from
historical prices.

For price series P_t, the simple return is:

r_t = P_t / P_(t-1) - 1

Returns are calculated within each training and testing window rather
than using future observations when estimating the covariance matrix.

---

## 5. Covariance Estimation

Three covariance methodologies are evaluated.

### 5.1 Sample Covariance

The sample covariance matrix provides the conventional empirical
estimate of the covariance structure of asset returns.

It serves as the primary baseline against which the shrinkage methods
are evaluated.

### 5.2 Fixed-Intensity Shrinkage

The shrinkage estimator combines the sample covariance estimate with a
structured target using a specified shrinkage intensity.

CARL evaluates several shrinkage intensities during sensitivity analysis.

The shrinkage intensity controls the degree to which the empirical
covariance estimate is pulled toward the target.

### 5.3 Ledoit-Wolf

CARL also evaluates the Ledoit-Wolf covariance estimator.

The Ledoit-Wolf literature motivates shrinkage as a way to improve the
conditioning and estimation properties of covariance matrices and
demonstrates its relevance to portfolio selection. 

---

## 6. Portfolio Construction

For each covariance estimate, CARL constructs a global minimum-variance
(GMV) portfolio.

The portfolio weights are determined from the estimated covariance
matrix.

The key research principle is that covariance estimation occurs before
portfolio evaluation and that the portfolio is subsequently evaluated
using observations that were not used to estimate the covariance matrix.

---

## 7. Walk-Forward Validation

CARL uses walk-forward validation to separate model estimation from
subsequent portfolio evaluation.

For each fold:

1. A historical training window is selected.
2. Asset returns within the training window are calculated.
3. The covariance estimator is fitted using training data only.
4. Portfolio weights are constructed from the estimated covariance
   matrix.
5. The subsequent test window is treated as unseen data.
6. Portfolio performance is calculated on the test window.
7. The process is repeated through the historical sample.

The resulting performance series represents out-of-sample portfolio
returns.

---

## 8. Expanding Training Windows

CARL supports expanding training windows.

Under the expanding-window design, additional historical observations
are progressively incorporated into the training sample as the
walk-forward process advances.

This allows the research to examine how additional historical
information affects covariance estimation and portfolio performance.

---

## 9. Out-of-Sample Principle

A central research control is prevention of look-ahead bias.

Covariance estimation for each fold uses training observations only.

The subsequent test observations are not used when estimating the
covariance matrix or constructing the portfolio weights for that fold.

This separation is maintained throughout the walk-forward experiment.

---

## 10. Performance Evaluation

CARL evaluates each methodology using multiple performance and risk
metrics.

### Return Metrics

- Total return
- Annualized return

### Risk Metrics

- Annualized volatility
- Maximum drawdown

### Risk-Adjusted Metrics

- Sharpe ratio
- Sortino ratio
- Calmar ratio

### Implementation Metrics

- Average turnover
- Transaction costs

### Additional Metric

- Hit rate

The purpose of using multiple metrics is to avoid selecting a covariance
methodology solely on the basis of raw portfolio return.

---

## 11. Transaction Costs

CARL supports transaction-cost modelling within the backtesting
framework.

This allows portfolio performance to be evaluated not only on gross
returns but also under implementation assumptions.

Turnover is retained as an additional implementation diagnostic.

---

## 12. Baseline Comparison

The baseline research compares the three covariance methodologies using
a common experimental configuration.

The baseline comparison provides the initial evidence regarding:

- performance;
- risk;
- risk-adjusted performance; and
- implementation characteristics.

The baseline is subsequently subjected to additional robustness and
sensitivity analysis.

---

## 13. Scenario Analysis

CARL evaluates methodology preferences under different research
objectives.

Three scenarios are considered:

1. Performance-focused
2. Risk-control-focused
3. Implementation-focused

This recognizes that the preferred covariance methodology may depend on
the objective being optimized.

---

## 14. Training-Window Sensitivity

CARL evaluates multiple training-window lengths while holding the test
window constant.

The purpose is to determine whether conclusions are sensitive to the
amount of historical information used for covariance estimation.

The research evaluates training windows of:

- 126 observations
- 252 observations
- 504 observations

---

## 15. Shrinkage Sensitivity

CARL evaluates fixed shrinkage intensities across a range of values:

- 0.00
- 0.10
- 0.25
- 0.50
- 0.75
- 1.00

The purpose is to determine how portfolio outcomes respond to the
strength of shrinkage.

The analysis considers:

- return;
- volatility;
- Sharpe ratio;
- Sortino ratio;
- maximum drawdown;
- Calmar ratio; and
- hit rate.

---

## 16. Robustness Analysis

CARL evaluates covariance methodologies across multiple experimental
configurations.

The robustness framework varies training-window specifications while
maintaining a common test-window structure.

Each configuration is evaluated using the same out-of-sample research
engine.

The purpose is to determine whether the principal findings remain
informative when reasonable methodological assumptions are changed.

---

## 17. Regime Analysis

CARL divides the historical sample into multiple historical periods and
evaluates covariance-method performance within each period.

The analysis compares:

- annualized return;
- Sharpe ratio; and
- maximum drawdown.

The purpose is to determine whether methodology rankings remain stable
across different market environments.

---

## 18. Research Synthesis

The final research conclusion is not based on a single metric or
experiment.

Evidence is consolidated across:

1. baseline comparison;
2. scenario analysis;
3. training-window sensitivity;
4. shrinkage sensitivity;
5. robustness analysis; and
6. regime analysis.

The final methodology-selection framework therefore considers both
performance and stability.

---

## 19. Research Controls

CARL incorporates several controls designed to improve reproducibility
and research integrity.

These include:

- deterministic research configurations;
- validated input data;
- chronological data requirements;
- duplicate-date checks;
- explicit training and test windows;
- out-of-sample evaluation;
- metadata describing experiments;
- automated unit and integration tests; and
- reproducible research configurations.

---

## 20. Reproducibility

The research methodology is implemented through reusable Python modules
rather than notebook-only calculations.

The notebooks serve as the research and evidence layer, while the
underlying CARL package contains the reusable data, covariance,
portfolio, validation, and evaluation components.

This separation allows the experiments to be reproduced independently
of the presentation layer.

---

## 21. Methodological Scope

The methodology is designed to evaluate covariance estimation within a
specific portfolio construction problem.

It does not attempt to establish that any covariance estimator is
universally optimal across:

- all asset classes;
- all portfolio objectives;
- all market regimes; or
- all transaction-cost environments.

The empirical conclusions should therefore be interpreted within the
scope of the research design and dataset.
