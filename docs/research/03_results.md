# Crypto Alpha Research Laboratory (CARL)

# Research Results and Evidence

## 1. Results Overview

The empirical analysis evaluates three covariance estimation methods:

1. Sample covariance
2. Fixed-intensity shrinkage
3. Ledoit-Wolf shrinkage

The methods are evaluated using out-of-sample portfolio returns and
multiple performance, risk, implementation, sensitivity, robustness,
and regime-based diagnostics.

The results show that covariance methodology materially affects
portfolio outcomes, but no single estimator dominates across every
objective and market environment.

---

# 2. Baseline Comparison

The baseline experiment used a common out-of-sample configuration for
all three covariance methodologies.

The resulting portfolio statistics were:

| Method | Total Return | Annualized Return | Volatility | Sharpe | Sortino | Max Drawdown | Calmar | Avg. Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sample | 86.97% | 15.83% | 52.55% | 0.542 | 0.0417 | -78.20% | 0.214 | 1.219 |
| Shrinkage | 78.17% | 14.53% | 52.40% | 0.521 | 0.0399 | -76.74% | 0.202 | **1.111** |
| Ledoit-Wolf | 56.88% | 11.16% | 52.59% | 0.464 | 0.0354 | **-72.58%** | 0.168 | 1.352 |

## Baseline Interpretation

Sample covariance generated the strongest baseline performance.

It produced the highest:

- total return;
- annualized return;
- Sharpe ratio;
- Sortino ratio; and
- Calmar ratio.

Shrinkage produced lower turnover than both alternatives.

Ledoit-Wolf produced the least severe maximum drawdown.

The baseline therefore presents a trade-off rather than a universal
winner.

The sample estimator was strongest from a performance perspective,
while shrinkage was more attractive from an implementation perspective,
and Ledoit-Wolf provided the strongest baseline drawdown control.

---

# 3. Scenario Analysis

CARL evaluated three decision scenarios:

1. Performance-focused
2. Risk-control-focused
3. Implementation-focused

The resulting scenario scores were:

| Method | Performance-Focused | Risk-Control-Focused | Implementation-Focused |
|---|---:|---:|---:|
| Sample | 0.755 | 0.455 | 0.625 |
| Shrinkage | 0.655 | **0.517** | **0.790** |
| Ledoit-Wolf | 0.200 | 0.500 | 0.150 |

The scenario winners were:

| Objective | Winner |
|---|---|
| Performance-focused | Sample |
| Risk-control-focused | Shrinkage |
| Implementation-focused | Shrinkage |

## Interpretation

The scenario analysis demonstrates that methodology selection depends on
the objective being optimized.

Sample covariance is preferred when performance is given the greatest
weight.

Shrinkage becomes preferable when risk control or implementation
characteristics receive greater importance.

This supports the conclusion that covariance estimation should be
treated as an investment-design decision rather than simply a technical
parameter choice.

---

# 4. Training-Window Sensitivity

CARL evaluated training windows of:

- 126 observations;
- 252 observations; and
- 504 observations.

The test window was held at 21 observations.

The resulting total returns were:

| Training Window | Sample | Shrinkage | Ledoit-Wolf |
|---:|---:|---:|---:|
| 126 | 82.25% | 65.07% | 19.18% |
| 252 | 86.97% | 78.17% | 56.88% |
| 504 | **194.90%** | 175.83% | 127.62% |

The results show substantial sensitivity to the amount of historical
information used to estimate the covariance matrix.

The 504-observation configuration generated substantially stronger
returns for all three methodologies.

However, the ranking remained consistent within these configurations:

**Sample > Shrinkage > Ledoit-Wolf**

in terms of total return.

---

# 5. Risk-Adjusted Training-Window Results

The training-window analysis also showed meaningful changes in
risk-adjusted performance.

For the 504-observation window:

| Method | Annualized Return | Volatility | Sharpe | Max Drawdown | Calmar |
|---|---:|---:|---:|---:|---:|
| Sample | 35.41% | 49.23% | **0.862** | -52.16% | **0.707** |
| Shrinkage | 32.90% | 49.14% | 0.824 | -49.87% | 0.687 |
| Ledoit-Wolf | 25.93% | 49.61% | 0.713 | **-45.19%** | 0.600 |

The longer training window improved the observed risk-adjusted
performance of all three methods.

Sample covariance maintained the highest Sharpe and Calmar ratios.

Ledoit-Wolf maintained the strongest drawdown control.

This reinforces the baseline finding that performance and downside-risk
objectives can lead to different methodology preferences.

---

# 6. Shrinkage-Intensity Sensitivity

CARL evaluated fixed shrinkage intensities from 0.00 to 1.00.

| Shrinkage | Total Return | Ann. Return | Volatility | Sharpe | Max Drawdown |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 86.97% | 15.83% | 52.55% | 0.542 | -78.20% |
| 0.10 | 83.29% | 15.29% | 52.46% | 0.533 | -77.60% |
| 0.25 | 78.17% | 14.53% | 52.40% | 0.521 | -76.74% |
| 0.50 | 70.50% | 13.35% | 52.38% | 0.501 | -75.37% |
| 0.75 | 63.53% | 12.25% | 52.45% | 0.482 | -74.04% |
| 1.00 | 56.88% | 11.16% | 52.59% | 0.464 | -72.58% |

## Interpretation

Increasing shrinkage intensity produced a systematic reduction in
portfolio return and Sharpe ratio within this experiment.

At the same time, maximum drawdown became progressively less severe.

This creates a clear performance-versus-downside-risk trade-off.

The results therefore suggest that shrinkage intensity should not be
selected solely according to return maximization.

---

# 7. Robustness Analysis

CARL expanded the research across multiple experimental
configurations.

The robustness experiment evaluated:

- 3 covariance methods;
- 3 training-window configurations;
- a constant 21-observation test window.

This produced:

**9 experiments**

and all experiments were evaluated out of sample.

The robustness metadata recorded:

```text
analysis: covariance_robustness
out_of_sample: True
methods: sample, shrinkage, ledoit_wolf
configuration_count: 3
experiment_count: 9