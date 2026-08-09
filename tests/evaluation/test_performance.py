import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.evaluation.performance import (
    PerformanceAnalyzer,
    PerformanceReport,
)


@pytest.fixture
def returns():
    return pd.Series(
        [
            0.01,
            -0.005,
            0.02,
            0.01,
            -0.01,
            0.015,
        ],
        index=pd.date_range(
            "2024-01-01",
            periods=6,
            freq="D",
        ),
    )


@pytest.fixture
def analyzer():
    return PerformanceAnalyzer(
        periods_per_year=252,
    )


def test_analyzer_can_be_created():
    analyzer = PerformanceAnalyzer()

    assert analyzer.periods_per_year == 252
    assert analyzer.risk_free_rate == 0.0


def test_total_return_is_correct(
    analyzer,
    returns,
):
    expected = (
        (1.01)
        * (0.995)
        * (1.02)
        * (1.01)
        * (0.99)
        * (1.015)
        - 1.0
    )

    assert analyzer.total_return(
        returns
    ) == pytest.approx(expected)


def test_cumulative_returns_preserve_index(
    analyzer,
    returns,
):
    result = analyzer.cumulative_returns(
        returns
    )

    assert isinstance(result, pd.Series)
    assert result.index.equals(
        returns.index
    )


def test_cumulative_returns_are_correct(
    analyzer,
    returns,
):
    result = analyzer.cumulative_returns(
        returns
    )

    expected = (
        (1.0 + returns).cumprod()
        - 1.0
    )

    pd.testing.assert_series_equal(
        result,
        expected,
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


def test_sharpe_ratio_is_finite(
    analyzer,
    returns,
):
    result = analyzer.sharpe_ratio(
        returns
    )

    assert np.isfinite(result)


def test_drawdown_series_never_positive(
    analyzer,
    returns,
):
    result = analyzer.drawdown_series(
        returns
    )

    assert (result <= 0.0).all()


def test_maximum_drawdown_is_non_positive(
    analyzer,
    returns,
):
    result = analyzer.maximum_drawdown(
        returns
    )

    assert result <= 0.0


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


def test_report_contains_correct_observation_count(
    analyzer,
    returns,
):
    result = analyzer.analyze(
        returns
    )

    assert (
        result.observation_count
        == len(returns)
    )


def test_report_total_return_matches_analyzer(
    analyzer,
    returns,
):
    result = analyzer.analyze(
        returns
    )

    assert result.total_return == pytest.approx(
        analyzer.total_return(
            returns
        )
    )


def test_empty_returns_are_rejected(
    analyzer,
):
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        analyzer.analyze(
            pd.Series(dtype=float)
        )


def test_non_series_returns_are_rejected(
    analyzer,
):
    with pytest.raises(
        TypeError,
        match="pandas Series",
    ):
        analyzer.analyze(
            np.array([0.01, 0.02])
        )


def test_non_finite_returns_are_rejected(
    analyzer,
):
    returns = pd.Series(
        [0.01, np.nan, 0.02]
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        analyzer.analyze(
            returns
        )


def test_duplicate_index_is_rejected(
    analyzer,
):
    returns = pd.Series(
        [0.01, 0.02],
        index=[
            "2024-01-01",
            "2024-01-01",
        ],
    )

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        analyzer.analyze(
            returns
        )


def test_invalid_periods_per_year_is_rejected():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        PerformanceAnalyzer(
            periods_per_year=0
        )


def test_invalid_risk_free_rate_is_rejected():
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        PerformanceAnalyzer(
            risk_free_rate=np.nan
        )