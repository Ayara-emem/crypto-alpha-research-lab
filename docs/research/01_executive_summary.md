# Crypto Alpha Research Laboratory (CARL)

## Research Executive Summary

### Research Problem

Portfolio construction depends critically on the quality of the estimated
covariance matrix.

CARL investigates whether different covariance estimation methods produce
materially different out-of-sample portfolio outcomes when applied to a
cryptocurrency portfolio.

The study compares three covariance methodologies:

1. Sample covariance
2. Fixed-intensity shrinkage
3. Ledoit-Wolf shrinkage

The methods are evaluated within a global minimum-variance portfolio
framework.

---

## Research Question

> Does the choice of covariance estimator materially affect
> out-of-sample portfolio performance, risk, implementation
> characteristics, and robustness across different market environments?

---

## Research Design

CARL uses a walk-forward out-of-sample research framework.

Covariance matrices are estimated using training observations only.
Portfolio weights are then constructed from those estimates and evaluated
on subsequent unseen observations.

The research evaluates:

- total return;
- annualized return;
- annualized volatility;
- Sharpe ratio;
- Sortino ratio;
- maximum drawdown;
- Calmar ratio;
- hit rate;
- turnover;
- transaction costs;
- training-window sensitivity;
- shrinkage sensitivity;
- robustness across configurations; and
- historical regime performance.

---

## Principal Findings

The baseline experiment showed that sample covariance produced the
strongest overall performance according to total return, annualized
return, Sharpe ratio, Sortino ratio, and Calmar ratio.

Shrinkage produced lower turnover than sample covariance and therefore
provided a stronger implementation-oriented profile.

Ledoit-Wolf produced the least severe maximum drawdown in the baseline
comparison.

However, the results did not establish universal superiority.

Scenario analysis showed that:

- sample covariance was preferred under a performance-focused objective;
- shrinkage was preferred under risk-control and implementation-focused
  objectives.

Training-window analysis showed that portfolio performance was sensitive
to the amount of historical information used for covariance estimation.

Regime analysis further demonstrated that the leading methodology
changed across historical periods.

During 2021–2022, shrinkage led annualized return and Sharpe ratio.

During 2023–2024, sample covariance led return, Sharpe ratio, and
maximum drawdown.

During 2025, Ledoit-Wolf led all three measures.

---

## Central Finding

The evidence does not support the hypothesis that one covariance
estimator universally dominates the alternatives.

Instead, covariance methodology selection appears to be context-dependent.

The appropriate methodology depends on:

- investment objective;
- risk preference;
- implementation constraints;
- training-window specification; and
- market regime.

---

## Research Contribution

CARL demonstrates a reproducible quantitative research workflow in which
a portfolio methodology is evaluated beyond a single backtest.

The project connects:

**Data → Covariance Estimation → Portfolio Construction →
Walk-Forward Validation → Performance Evaluation →
Sensitivity Analysis → Robustness Analysis →
Regime Analysis → Research Synthesis**

The resulting framework provides an empirical basis for selecting
covariance methodologies according to investment objectives rather than
assuming that a single estimator is universally optimal.

---

## Limitations

The findings should be interpreted within the scope of the experiment.

Key limitations include:

1. The empirical universe consists of a relatively small number of
   cryptocurrency assets.

2. Cryptocurrency returns exhibit substantial non-stationarity, making
   regime dependence particularly important.

3. The portfolio construction framework is based on the global
   minimum-variance objective and therefore does not represent every
   possible investment objective.

4. The fixed shrinkage intensity used in several experiments is not
   necessarily optimal for every market condition.

5. Transaction-cost assumptions may not fully represent real-world
   execution costs, market impact, or liquidity.

6. Historical backtest performance does not guarantee future performance.

---

## Future Research

Potential extensions include:

- expanding the asset universe;
- introducing additional covariance estimators;
- dynamically estimating shrinkage intensity;
- incorporating liquidity and market-impact models;
- testing alternative portfolio objectives;
- introducing portfolio constraints;
- investigating volatility and correlation regimes; and
- developing adaptive covariance-method selection.

---

## Conclusion

CARL demonstrates that covariance estimation is not merely a statistical
preprocessing choice.

It can materially influence portfolio performance, risk, turnover, and
robustness.

The research therefore supports a context-dependent approach to
covariance methodology selection and demonstrates the importance of
strict out-of-sample validation in quantitative portfolio research.