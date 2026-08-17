# CARL — Executive Research Summary

## Cryptocurrency Covariance Estimation and Portfolio Construction

### Executive Summary

The Crypto Alpha Research Laboratory (CARL) investigates whether
different covariance estimation methods materially affect the
out-of-sample performance and implementation characteristics of
Global Minimum Variance portfolios in a cryptocurrency universe.

The research compares three covariance approaches:

- Sample covariance
- Fixed shrinkage covariance
- Ledoit-Wolf shrinkage

The central research question is:

> Does the choice of covariance estimator materially affect portfolio
> performance, risk, turnover, and robustness across different
> estimation configurations and market regimes?

---

## 1. Research Design

CARL uses a walk-forward out-of-sample framework.

For each evaluation fold:

1. Historical training observations are isolated.
2. Asset returns are calculated from the training data.
3. The covariance matrix is estimated using training data only.
4. A Global Minimum Variance portfolio is constructed.
5. Portfolio weights are evaluated on the subsequent test window.
6. Out-of-sample returns are aggregated across folds.
7. Performance and risk statistics are calculated from the resulting
   out-of-sample return series.

The baseline configuration uses:

| Component | Configuration |
|---|---|
| Portfolio | Global Minimum Variance |
| Training window | 252 observations |
| Test window | 21 observations |
| Shrinkage intensity | 0.25 |
| Annualization | 252 periods/year |
| Risk-free rate | 0.0 |

This design is intended to reduce look-ahead bias by ensuring that
covariance estimates are formed before the corresponding test-period
returns are observed.

---

## 2. Baseline Results

The current CARL evaluation layer produces the following baseline
results:

| Method | Total Return | Annualized Return | Volatility | Sharpe | Max Drawdown | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| Sample | 86.97% | 10.68% | 43.66% | 0.451 | -78.20% | 1.219 |
| Shrinkage | 78.17% | 9.82% | 43.54% | 0.433 | -76.74% | 1.111 |
| Ledoit-Wolf | 56.88% | 7.58% | 43.70% | 0.386 | -72.58% | 1.352 |

### Baseline interpretation

The sample covariance estimator produces the highest total and
annualized return and the highest Sharpe ratio in the baseline
experiment.

However, it also produces the largest maximum drawdown.

Fixed shrinkage produces slightly lower return and Sharpe performance
but reduces both maximum drawdown and average turnover relative to the
sample estimator.

Ledoit-Wolf produces the lowest return and Sharpe ratio in the baseline
configuration, but produces the least severe maximum drawdown.

The evidence therefore does not support selecting a covariance method
using return alone.

---

## 3. Robustness Findings

### Training-window sensitivity

CARL evaluates training windows of:

- 126 observations
- 252 observations
- 504 observations

The resulting total returns are:

| Training Window | Sample | Shrinkage | Ledoit-Wolf |
|---:|---:|---:|---:|
| 126 | 82.25% | 65.07% | 19.18% |
| 252 | 86.97% | 78.17% | 56.88% |
| 504 | 194.90% | 175.83% | 127.62% |

The magnitude of the results changes substantially with the training
window.

This indicates that covariance-method conclusions are sensitive to the
amount of historical information used for estimation.

---

### Shrinkage sensitivity

Fixed shrinkage intensity is evaluated from 0.00 to 1.00.

The research shows a systematic change in performance as shrinkage
intensity increases.

The observed total returns are:

| Shrinkage | Total Return |
|---:|---:|
| 0.00 | 86.97% |
| 0.10 | 83.29% |
| 0.25 | 78.17% |
| 0.50 | 70.50% |
| 0.75 | 63.53% |
| 1.00 | 56.88% |

The result indicates that shrinkage intensity is an economically
meaningful research parameter rather than merely an implementation
choice.

---

## 4. Regime Analysis

CARL evaluates covariance-method behaviour across:

- 2021–2022
- 2023–2024
- 2025

The preferred method changes across historical periods.

The regime-level evidence therefore does not support the claim that one
covariance estimator is universally superior.

Instead, the results suggest that covariance-method effectiveness is
conditional on the underlying market environment.

---

## 5. Main Research Findings

The evidence supports five principal conclusions.

### Finding 1 — Covariance estimation matters

Different covariance estimators produce materially different
portfolio outcomes.

### Finding 2 — Return is not sufficient

The highest-return method does not simultaneously minimize drawdown or
turnover.

### Finding 3 — Estimation-window choice matters

Changing the training-window length materially changes observed
out-of-sample performance.

### Finding 4 — Shrinkage introduces a trade-off

Increasing shrinkage changes the return, risk, and implementation
profile of the resulting portfolio.

### Finding 5 — Regime dependence matters

The preferred covariance method changes across historical periods.

---

## 6. Research Recommendation

The evidence does **not** justify a universal recommendation to always
use one covariance estimator.

Instead, covariance-method selection should be treated as a
multi-objective research decision.

A practical selection framework should consider:

1. Expected return
2. Risk-adjusted performance
3. Maximum drawdown
4. Portfolio turnover
5. Estimation-window sensitivity
6. Shrinkage sensitivity
7. Historical-regime behaviour

For a performance-focused objective, the sample covariance estimator
is strongest in the baseline experiment.

For an implementation-focused objective, fixed shrinkage provides a
more attractive trade-off because it reduces turnover while retaining
relatively strong performance.

For a risk-control objective, the lower drawdown produced by
Ledoit-Wolf becomes relevant despite its weaker baseline return.

The appropriate choice therefore depends on the investment objective
rather than on a single universal ranking.

---

## 7. Limitations

The results should be interpreted within the scope of the research
design.

### Asset universe

The analysis uses a defined cryptocurrency research universe and
therefore should not automatically be generalized to all digital
assets or traditional asset classes.

### Historical dependence

The conclusions are based on historical observations and may not hold
under future market conditions.

### Model specification

The analysis focuses on Global Minimum Variance portfolio construction.
Other portfolio objectives could produce different conclusions.

### Transaction costs

Turnover is evaluated as an implementation metric, but the baseline
research does not establish that observed results remain optimal under
all realistic transaction-cost assumptions.

### Parameter sensitivity

The results demonstrate that training-window length and shrinkage
intensity affect outcomes. Consequently, parameter selection itself
introduces model risk.

### Regime coverage

The regime analysis provides useful historical comparisons but does not
guarantee that future market regimes will resemble the selected
periods.

---

## 8. Reproducibility

The research is implemented as a modular Python research laboratory.

The principal components include:

```text
src/crypto_alpha_lab/research/
src/crypto_alpha_lab/evaluation/
tests/
scripts/generate_research_figures.py
scripts/validate_research_figures.py
docs/figures/