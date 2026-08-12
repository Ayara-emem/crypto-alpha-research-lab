from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.evaluation.performance import (
    PerformanceAnalyzer,
    PerformanceReport,
)
from crypto_alpha_lab.backtest.report import (
    BacktestReport,
    build_backtest_report,
    report_summary,
    report_table,
)


@pytest.fixture
def returns() -> pd.Series:
    """Deterministic strategy return series."""
    index = pd.date_range(
        "2024-01-01",
        periods=20,
        freq="D",
    )

    values = np.array(
        [
            0.010,
            0.005,
            -0.003,
            0.008,
            0.004,
            -0.002,
            0.006,
            0.003,
            -0.004,
            0.007,
            0.005,
            -0.001,
            0.009,
            0.002,
            -0.003,
            0.006,
            0.004,
            -0.002,
            0.008,
            0.003,
        ],
        dtype=float,
    )

    return pd.Series(
        values,
        index=index,
        name="portfolio_returns",
    )


@pytest.fixture
def analyzer() -> PerformanceAnalyzer:
    return PerformanceAnalyzer()


def test_analyzer_can_be_created():
    analyzer = PerformanceAnalyzer()

    assert analyzer.periods_per_year == 252
    assert analyzer.risk_free_rate == 0.0


def test_custom_analyzer_configuration_is_preserved():
    analyzer = PerformanceAnalyzer(
        periods_per_year=365,
        risk_free_rate=0.03,
    )

    assert analyzer.periods_per_year == 365
    assert analyzer.risk_free_rate == 0.03


def test_invalid_periods_per_year_are_rejected():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        PerformanceAnalyzer(
            periods_per_year=0,
        )


def test_non_finite_risk_free_rate_is_rejected():
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        PerformanceAnalyzer(
            risk_free_rate=np.nan,
        )


def test_total_return_is_compounded(
    analyzer,
    returns,
):
    expected = float(
        (1.0 + returns).prod() - 1.0
    )

    result = analyzer.total_return(
        returns
    )

    assert result == pytest.approx(
        expected
    )


def test_annualized_return_is_finite(
    analyzer,
    returns,
):
    result = analyzer.annualized_return(
        returns
    )

    assert np.isfinite(result)


def test_annualized_volatility_is_finite(
    analyzer,
    returns,
):
    result = analyzer.annualized_volatility(
        returns
    )

    assert np.isfinite(result)
    assert result >= 0.0


def test_cumulative_returns_are_series(
    analyzer,
    returns,
):
    result = analyzer.cumulative_returns(
        returns
    )

    assert isinstance(
        result,
        pd.Series,
    )


def test_cumulative_returns_preserve_index(
    analyzer,
    returns,
):
    result = analyzer.cumulative_returns(
        returns
    )

    pd.testing.assert_index_equal(
        result.index,
        returns.index,
    )


def test_cumulative_returns_match_compounding(
    analyzer,
    returns,
):
    expected = (
        (1.0 + returns)
        .cumprod()
        - 1.0
    )

    result = analyzer.cumulative_returns(
        returns
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_drawdown_series_is_non_positive(
    analyzer,
    returns,
):
    drawdown = analyzer.drawdown_series(
        returns
    )

    assert (
        drawdown <= 0.0
    ).all()


def test_maximum_drawdown_is_non_positive(
    analyzer,
    returns,
):
    result = analyzer.maximum_drawdown(
        returns
    )

    assert result <= 0.0
    assert np.isfinite(result)


def test_sharpe_ratio_is_finite(
    analyzer,
    returns,
):
    result = analyzer.sharpe_ratio(
        returns
    )

    assert np.isfinite(result)


def test_sortino_ratio_is_finite(
    analyzer,
    returns,
):
    result = analyzer.sortino_ratio(
        returns
    )

    assert np.isfinite(result)


def test_calmar_ratio_is_finite(
    analyzer,
    returns,
):
    result = analyzer.calmar_ratio(
        returns
    )

    assert np.isfinite(result)


def test_hit_rate_is_between_zero_and_one(
    analyzer,
    returns,
):
    result = analyzer.hit_rate(
        returns
    )

    assert 0.0 <= result <= 1.0


def test_hit_rate_matches_positive_observations(
    analyzer,
    returns,
):
    expected = float(
        (returns > 0.0).mean()
    )

    result = analyzer.hit_rate(
        returns
    )

    assert result == pytest.approx(
        expected
    )


def test_analyze_returns_performance_report(
    analyzer,
    returns,
):
    result = analyzer.analyze(
        returns
    )

    assert isinstance(
        result,
        PerformanceReport,
    )


def test_analyze_contains_all_metrics(
    analyzer,
    returns,
):
    result = analyzer.analyze(
        returns
    )

    assert np.isfinite(
        result.total_return
    )

    assert np.isfinite(
        result.annualized_return
    )

    assert np.isfinite(
        result.annualized_volatility
    )

    assert np.isfinite(
        result.sharpe_ratio
    )

    assert np.isfinite(
        result.maximum_drawdown
    )

    assert np.isfinite(
        result.calmar_ratio
    )

    assert np.isfinite(
        result.hit_rate
    )

    assert np.isfinite(
        result.sortino_ratio
    )

    assert (
        result.observation_count
        == len(returns)
    )


def test_analyze_is_deterministic(
    analyzer,
    returns,
):
    first = analyzer.analyze(
        returns
    )

    second = analyzer.analyze(
        returns
    )

    assert first == second


def test_non_series_returns_are_rejected(
    analyzer,
):
    with pytest.raises(
        TypeError,
        match="pandas Series",
    ):
        analyzer.analyze(
            np.array(
                [0.01, 0.02]
            )
        )


def test_empty_returns_are_rejected(
    analyzer,
):
    returns = pd.Series(
        dtype=float
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        analyzer.analyze(
            returns
        )


def test_duplicate_return_dates_are_rejected(
    analyzer,
    returns,
):
    duplicate = pd.concat(
        [
            returns,
            returns.iloc[[0]],
        ]
    )

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        analyzer.analyze(
            duplicate
        )


def test_non_finite_returns_are_rejected(
    analyzer,
    returns,
):
    invalid = returns.copy()

    invalid.iloc[0] = np.nan

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        analyzer.analyze(
            invalid
        )


def test_report_building_returns_backtest_report(
    benchmark_dataset,
):
    from crypto_alpha_lab.research.experiment import (
        ResearchExperiment,
    )
    from crypto_alpha_lab.backtest.engine import (
        BacktestEngine,
    )

    experiment = ResearchExperiment(
        dataset=benchmark_dataset,
        asset_universe=["Close"],
        price_columns=["Close"],
        portfolio=pd.Series(
            {"Close": 1.0},
            dtype=float,
        ),
    )

    engine = BacktestEngine()

    result = engine.run(
        experiment,
    )

    report = build_backtest_report(
        result
    )

    assert isinstance(
        report,
        BacktestReport,
    )


def test_report_contains_required_metrics(
    benchmark_dataset,
):
    from crypto_alpha_lab.research.experiment import (
        ResearchExperiment,
    )
    from crypto_alpha_lab.backtest.engine import (
        BacktestEngine,
    )

    experiment = ResearchExperiment(
    dataset=benchmark_dataset,
    asset_universe=["Close"],
    price_columns=["Close"],
    portfolio=pd.Series(
        {"Close": 1.0},
        dtype=float,
        ),
    )

    engine = BacktestEngine()
    result = engine.run(
        experiment,
        )

    report = build_backtest_report(
        result
    )

    required = {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "calmar_ratio",
        "hit_rate",
        "observation_count",
        "turnover",
        "transaction_costs",
    }

    assert required.issubset(
        report.summary.index
    )


def test_report_preserves_backtest_series(
    benchmark_dataset,
):
    from crypto_alpha_lab.research.experiment import (
        ResearchExperiment,
    )
    from crypto_alpha_lab.backtest.engine import (
        BacktestEngine,
    )

    experiment = ResearchExperiment(
        dataset=benchmark_dataset,
        asset_universe=["Close"],
        price_columns=["Close"],
        portfolio=pd.Series(
            {"Close": 1.0},
            dtype=float,
            ),
        )
    engine = BacktestEngine()
    result = engine.run(
        experiment,
            )

    report = build_backtest_report(
        result
    )

    pd.testing.assert_series_equal(
        report.portfolio_returns,
        result.portfolio_returns,
    )

    pd.testing.assert_series_equal(
        report.cumulative_returns,
        result.cumulative_returns,
    )


def test_report_preserves_weights(
    benchmark_dataset,
):
    from crypto_alpha_lab.research.experiment import (
        ResearchExperiment,
    )
    from crypto_alpha_lab.backtest.engine import (
        BacktestEngine,
    )

    experiment = ResearchExperiment(
            dataset=benchmark_dataset,
            asset_universe=["Close"],
            price_columns=["Close"],
            portfolio=pd.Series(
                {"Close": 1.0},
                dtype=float,
                ),
            )
    engine = BacktestEngine()
    result = engine.run(
            experiment,
                )

    report = build_backtest_report(
        result
    )

    pd.testing.assert_series_equal(
        report.weights,
        result.weights,
    )


def test_report_summary_returns_copy(
    benchmark_dataset,
):
    from crypto_alpha_lab.research.experiment import (
        ResearchExperiment,
    )
    from crypto_alpha_lab.backtest.engine import (
        BacktestEngine,
    )

    experiment = ResearchExperiment(
            dataset=benchmark_dataset,
            asset_universe=["Close"],
            price_columns=["Close"],
            portfolio=pd.Series(
                {"Close": 1.0},
                dtype=float,
                ),
            )
    engine = BacktestEngine()
    result = engine.run(
            experiment,
                )

    report = build_backtest_report(
        result
    )

    summary = report_summary(
        report
    )

    assert isinstance(
        summary,
        pd.Series,
    )

    pd.testing.assert_series_equal(
        summary,
        report.summary,
    )


def test_report_table_returns_dataframe(
    benchmark_dataset,
):
    from crypto_alpha_lab.research.experiment import (
        ResearchExperiment,
    )
    from crypto_alpha_lab.backtest.engine import (
        BacktestEngine,
    )

    experiment = ResearchExperiment(
            dataset=benchmark_dataset,
            asset_universe=["Close"],
            price_columns=["Close"],
            portfolio=pd.Series(
                {"Close": 1.0},
                dtype=float,
                ),
            )
    engine = BacktestEngine()
    result = engine.run(
            experiment,
                )

    report = build_backtest_report(
        result
    )

    table = report_table(
        report
    )

    assert isinstance(
        table,
        pd.DataFrame,
    )

    assert "value" in table.columns


def test_report_metadata_preserves_configuration(
    benchmark_dataset,
):
    from crypto_alpha_lab.research.experiment import (
        ResearchExperiment,
    )
    from crypto_alpha_lab.backtest.engine import (
        BacktestEngine,
    )

    experiment = ResearchExperiment(
            dataset=benchmark_dataset,
            asset_universe=["Close"],
            price_columns=["Close"],
            portfolio=pd.Series(
                {"Close": 1.0},
                dtype=float,
                ),
            )
    engine = BacktestEngine()
    result = engine.run(
            experiment,
                )
    
    report = build_backtest_report(
        result,
        periods_per_year=365,
        risk_free_rate=0.03,
    )

    assert (
        report.metadata[
            "periods_per_year"
        ]
        == 365
    )

    assert (
        report.metadata[
            "risk_free_rate"
        ]
        == 0.03
    )