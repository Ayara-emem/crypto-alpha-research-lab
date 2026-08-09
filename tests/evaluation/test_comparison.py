"""
Tests for CARL backtest comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.backtest.engine import (
    BacktestResult,
)

from crypto_alpha_lab.evaluation.comparison import (
    compare_backtests,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def result_a() -> BacktestResult:
    returns = pd.Series(
        [0.01, 0.02, -0.01, 0.03],
        index=pd.date_range(
            "2024-01-01",
            periods=4,
        ),
    )

    cumulative = (
        (1.0 + returns)
        .cumprod()
        - 1.0
    )

    return BacktestResult(
        asset_returns=pd.DataFrame(
            {
                "BTC": returns,
            }
        ),
        portfolio_returns=returns,
        cumulative_returns=cumulative,
        weights=pd.Series(
            {"BTC": 1.0},
        ),
        turnover=1.0,
        transaction_costs=0.01,
        metadata={},
    )


@pytest.fixture
def result_b() -> BacktestResult:
    returns = pd.Series(
        [0.02, 0.01, 0.01, 0.02],
        index=pd.date_range(
            "2024-01-01",
            periods=4,
        ),
    )

    cumulative = (
        (1.0 + returns)
        .cumprod()
        - 1.0
    )

    return BacktestResult(
        asset_returns=pd.DataFrame(
            {
                "BTC": returns,
            }
        ),
        portfolio_returns=returns,
        cumulative_returns=cumulative,
        weights=pd.Series(
            {"BTC": 1.0},
        ),
        turnover=0.5,
        transaction_costs=0.005,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Basic comparison
# ---------------------------------------------------------------------------


def test_compare_backtests_returns_dataframe(
    result_a,
    result_b,
):
    comparison = compare_backtests(
        {
            "Strategy A": result_a,
            "Strategy B": result_b,
        }
    )

    assert isinstance(
        comparison,
        pd.DataFrame,
    )


def test_compare_backtests_contains_strategy_names(
    result_a,
    result_b,
):
    comparison = compare_backtests(
        {
            "Strategy A": result_a,
            "Strategy B": result_b,
        }
    )

    assert list(
        comparison.index,
    ) == [
        "Strategy A",
        "Strategy B",
    ]


def test_compare_backtests_contains_expected_columns(
    result_a,
):
    comparison = compare_backtests(
        {
            "Strategy A": result_a,
        }
    )

    assert list(
        comparison.columns,
    ) == [
        "total_return",
        "mean_return",
        "volatility",
        "turnover",
        "transaction_costs",
    ]


# ---------------------------------------------------------------------------
# Metric correctness
# ---------------------------------------------------------------------------


def test_total_return_matches_final_cumulative_return(
    result_a,
):
    comparison = compare_backtests(
        {
            "Strategy A": result_a,
        }
    )

    assert comparison.loc[
        "Strategy A",
        "total_return",
    ] == pytest.approx(
        result_a.cumulative_returns.iloc[-1],
    )


def test_mean_return_is_correct(
    result_a,
):
    comparison = compare_backtests(
        {
            "Strategy A": result_a,
        }
    )

    expected = result_a.portfolio_returns.mean()

    assert comparison.loc[
        "Strategy A",
        "mean_return",
    ] == pytest.approx(
        expected,
    )


def test_volatility_is_correct(
    result_a,
):
    comparison = compare_backtests(
        {
            "Strategy A": result_a,
        }
    )

    expected = result_a.portfolio_returns.std()

    assert comparison.loc[
        "Strategy A",
        "volatility",
    ] == pytest.approx(
        expected,
    )


def test_turnover_is_correct(
    result_a,
):
    comparison = compare_backtests(
        {
            "Strategy A": result_a,
        }
    )

    assert comparison.loc[
        "Strategy A",
        "turnover",
    ] == pytest.approx(
        result_a.turnover,
    )


def test_transaction_costs_are_correct(
    result_a,
):
    comparison = compare_backtests(
        {
            "Strategy A": result_a,
        }
    )

    assert comparison.loc[
        "Strategy A",
        "transaction_costs",
    ] == pytest.approx(
        result_a.transaction_costs,
    )


# ---------------------------------------------------------------------------
# Multiple strategies
# ---------------------------------------------------------------------------


def test_multiple_strategies_are_compared(
    result_a,
    result_b,
):
    comparison = compare_backtests(
        {
            "Strategy A": result_a,
            "Strategy B": result_b,
        }
    )

    assert comparison.shape == (
        2,
        5,
    )


def test_strategies_remain_distinguishable(
    result_a,
    result_b,
):
    comparison = compare_backtests(
        {
            "Strategy A": result_a,
            "Strategy B": result_b,
        }
    )

    assert not comparison.loc[
        "Strategy A"
    ].equals(
        comparison.loc[
            "Strategy B"
        ]
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_compare_backtests_rejects_non_dict():
    with pytest.raises(TypeError):

        compare_backtests(
            [
                "invalid",
            ],
        )


def test_compare_backtests_rejects_empty_dict():
    with pytest.raises(ValueError):

        compare_backtests(
            {},
        )


def test_compare_backtests_rejects_invalid_result(
    result_a,
):
    with pytest.raises(TypeError):

        compare_backtests(
            {
                "Strategy A": result_a,
                "Invalid": object(),
            },
        )


# ---------------------------------------------------------------------------
# Numerical integrity
# ---------------------------------------------------------------------------


def test_comparison_values_are_finite(
    result_a,
    result_b,
):
    comparison = compare_backtests(
        {
            "Strategy A": result_a,
            "Strategy B": result_b,
        }
    )

    assert np.isfinite(
        comparison.to_numpy(),
    ).all()


def test_comparison_preserves_transaction_cost_precision(
    result_a,
):
    comparison = compare_backtests(
        {
            "Strategy A": result_a,
        }
    )

    assert comparison.loc[
        "Strategy A",
        "transaction_costs",
    ] == pytest.approx(
        0.01,
    )


def test_comparison_preserves_turnover_precision(
    result_b,
):
    comparison = compare_backtests(
        {
            "Strategy B": result_b,
        }
    )

    assert comparison.loc[
        "Strategy B",
        "turnover",
    ] == pytest.approx(
        0.5,
    )

