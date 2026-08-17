# CARL Research Figure Catalog

This catalog documents the figures generated from the Crypto Alpha
Research Laboratory (CARL) covariance research pipeline.

All figures are generated reproducibly from the CARL research engine
using out-of-sample walk-forward experiments.

---

## Figure 01 — Cumulative Out-of-Sample Returns

**File:** `fig01_cumulative_oos_returns.png`

**Research question:**  
How do the covariance estimation methods perform over the complete
out-of-sample evaluation period?

**Methodology:**  
Global Minimum Variance portfolios constructed from:

- Sample covariance
- Fixed shrinkage covariance
- Ledoit-Wolf covariance

Covariance estimates are formed using training observations only and
evaluated on subsequent out-of-sample observations using walk-forward
validation.

**Evidence represented:**  
Cumulative portfolio wealth across the complete out-of-sample period.

**Key finding:**  
The sample covariance approach produced the highest cumulative return
in the baseline experiment, followed by fixed shrinkage and
Ledoit-Wolf.

---

## Figure 02 — Risk-Adjusted Performance

**File:** `fig02_risk_adjusted_performance.png`

**Research question:**  
Does the ranking of covariance methods remain similar when performance
is evaluated on a risk-adjusted basis?

**Metrics:**

- Sharpe ratio
- Sortino ratio
- Calmar ratio

**Methodology:**  
Performance metrics are calculated from the out-of-sample portfolio
return series produced by the covariance experiments.

**Key finding:**  
The sample covariance method leads the baseline comparison on the
primary risk-adjusted performance measures.

---

## Figure 03 — Maximum Drawdown

**File:** `fig03_maximum_drawdown.png`

**Research question:**  
How does covariance estimation affect downside portfolio risk?

**Methodology:**  
Maximum drawdown is calculated from the cumulative out-of-sample
portfolio return series.

**Key finding:**  
Ledoit-Wolf produces the least severe maximum drawdown in the baseline
experiment, while sample covariance produces the largest drawdown.

This highlights an important trade-off between return and downside
risk.

---

## Figure 04 — Portfolio Turnover

**File:** `fig04_turnover.png`

**Research question:**  
How do covariance estimation methods differ in implementation
intensity?

**Methodology:**  
Average portfolio turnover is calculated across the out-of-sample
walk-forward evaluation.

**Key finding:**  
Fixed shrinkage produces the lowest average turnover, while
Ledoit-Wolf produces the highest.

This makes implementation efficiency an important consideration when
selecting a covariance estimator.

---

## Figure 05 — Training-Window Sensitivity

**File:** `fig05_training_window_sensitivity.png`

**Research question:**  
Are the covariance-method conclusions sensitive to the amount of
historical information used for covariance estimation?

**Training windows evaluated:**

- 126 observations
- 252 observations
- 504 observations

**Methodology:**  
Each covariance methodology is independently evaluated under each
training-window configuration using the same 21-observation test
window.

**Key finding:**  
Performance varies substantially with training-window length.

The 504-observation configuration produces materially higher cumulative
returns for all three covariance methods than the shorter training
windows.

This demonstrates that covariance-method conclusions cannot be
interpreted independently of estimation-window choice.

---

## Figure 06 — Shrinkage Sensitivity

**File:** `fig06_shrinkage_sensitivity.png`

**Research question:**  
How sensitive is portfolio performance to the selected shrinkage
intensity?

**Shrinkage intensities evaluated:**

- 0.00
- 0.10
- 0.25
- 0.50
- 0.75
- 1.00

**Metrics evaluated:**

- Annualized return
- Sharpe ratio
- Maximum drawdown

**Key finding:**  
Increasing shrinkage intensity progressively reduces the observed
baseline return and Sharpe ratio while reducing the magnitude of
maximum drawdown.

The result demonstrates a clear return-risk trade-off rather than a
single universally optimal shrinkage intensity.

---

## Figure 07 — Regime Analysis

**File:** `fig07_regime_analysis.png`

**Research question:**  
Are covariance-method rankings stable across different market
regimes?

**Periods evaluated:**

- 2021–2022
- 2023–2024
- 2025

**Methodology:**  
The covariance experiments are independently evaluated within each
historical period.

**Key finding:**  
The preferred covariance methodology changes across regimes.

The results therefore provide evidence against treating one covariance
estimator as universally dominant.

---

# Evidence Interpretation

The figures should not be interpreted individually as proof that one
covariance estimator is universally superior.

Taken together, they demonstrate three broader findings:

1. **Performance trade-offs exist.**  
   The method with the highest return is not necessarily the method
   with the lowest drawdown or turnover.

2. **Research conclusions are configuration-sensitive.**  
   Training-window length and shrinkage intensity materially affect
   observed results.

3. **Market regime matters.**  
   Covariance-method rankings change across historical periods.

Therefore, the appropriate research conclusion is not simply to select
the highest-return estimator.

Instead, covariance-method selection should consider:

- expected performance,
- risk-adjusted performance,
- downside risk,
- turnover,
- estimation-window choice,
- and market regime.

---

## Reproducibility

Figures are generated using:

```text
scripts/generate_research_figures.py