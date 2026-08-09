"""
Tests for CARL out-of-sample evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.backtest.engine import (
    BacktestResult,
)

from crypto_alpha_lab.strategy.strategy import (
    ResearchStrategy,
)

from crypto_alpha_lab.validation.out_of_sample import (
    OutOfSampleEvaluator,
    OutOfSampleResult,
)

from crypto_alpha_lab.validation.walk_forward import (
    WalkForwardSplit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def strategy() -> ResearchStrategy:
    return ResearchStrategy(
        name="OOS Test Strategy",
        signals=pd.Series(
            {
                "BTC": 0.6,
                "ETH": 0.4,
            },
            dtype=float,
        ),
        portfolio_method="signal_weighted",
    )


@pytest.fixture
def splits() -> list[WalkForwardSplit]:
    dates_1 = pd.date_range(
        "2024-01-01",
        periods=5,
        freq="D",
    )

    dates_2 = pd.date_range(
        "2024-01-06",
        periods=5,
        freq="D",
    )

    train_1 = pd.DataFrame(
        {
            "BTC": np.arange(
                100.0,
                105.0,
            ),
            "ETH": np.arange(
                50.0,
                55.0,
            ),
        },
        index=dates_1,
    )

    test_1 = pd.DataFrame(
        {
            "BTC": [
                0.01,
                0.02,
                -0.01,
                0.03,
                0.01,
            ],
            "ETH": [
                0.02,
                0.01,
                0.00,
                0.02,
                0.01,
            ],
        },
        index=dates_2,
    )

    dates_3 = pd.date_range(
        "2024-01-11",
        periods=5,
        freq="D",
    )

    train_2 = pd.concat(
        [
            train_1,
            test_1,
        ]
    )

    test_2 = pd.DataFrame(
        {
            "BTC": [
                0.02,
                0.01,
                0.03,
                -0.01,
                0.02,
            ],
            "ETH": [
                0.01,
                0.02,
                0.01,
                0.00,
                0.02,
            ],
        },
        index=dates_3,
    )

    return [
        WalkForwardSplit(
            train=train_1,
            test=test_1,
            fold=0,
        ),
        WalkForwardSplit(
            train=train_2,
            test=test_2,
            fold=1,
        ),
    ]


@pytest.fixture
def evaluator() -> OutOfSampleEvaluator:
    return OutOfSampleEvaluator()


# ---------------------------------------------------------------------------
# Basic evaluation
# ---------------------------------------------------------------------------


def test_evaluator_can_be_created():
    evaluator = OutOfSampleEvaluator()

    assert isinstance(
        evaluator,
        OutOfSampleEvaluator,
    )


def test_evaluate_returns_oos_result(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    assert isinstance(
        result,
        OutOfSampleResult,
    )


def test_oos_result_contains_returns(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    assert isinstance(
        result.returns,
        pd.Series,
    )


def test_oos_result_contains_fold_results(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    assert isinstance(
        result.folds,
        tuple,
    )

    assert len(
        result.folds,
    ) == len(splits)


# ---------------------------------------------------------------------------
# Fold results
# ---------------------------------------------------------------------------


def test_each_fold_is_a_backtest_result(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    assert all(
        isinstance(
            fold,
            BacktestResult,
        )
        for fold in result.folds
    )


def test_each_fold_is_marked_out_of_sample(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    for fold in result.folds:

        assert (
            fold.metadata["out_of_sample"]
            is True
        )


def test_each_fold_contains_walk_forward_metadata(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    for expected_fold, result_fold in enumerate(
        result.folds
    ):

        assert (
            result_fold.metadata["validation"]
            == "walk_forward"
        )

        assert (
            result_fold.metadata["fold"]
            == expected_fold
        )


# ---------------------------------------------------------------------------
# Portfolio returns
# ---------------------------------------------------------------------------


def test_fold_returns_use_strategy_weights(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    weights = strategy.weights()

    expected = (
        splits[0]
        .test[
            weights.index
        ]
        @ weights
    )

    pd.testing.assert_series_equal(
        result.folds[0].portfolio_returns,
        expected,
    )


def test_oos_returns_are_concatenated_in_order(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    expected = pd.concat(
        [
            split.test[
                strategy.weights().index
            ]
            @ strategy.weights()
            for split in splits
        ]
    )

    pd.testing.assert_series_equal(
        result.returns,
        expected,
    )


def test_oos_returns_have_expected_length(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    expected_length = sum(
        len(split.test)
        for split in splits
    )

    assert len(
        result.returns,
    ) == expected_length


# ---------------------------------------------------------------------------
# Cumulative returns
# ---------------------------------------------------------------------------


def test_cumulative_returns_is_series(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    assert isinstance(
        result.cumulative_returns,
        pd.Series,
    )


def test_cumulative_returns_aligns_with_oos_returns(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    assert len(
        result.cumulative_returns,
    ) == len(
        result.returns,
    )


def test_cumulative_returns_is_correct(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    expected = (
        (1.0 + result.returns)
        .cumprod()
        - 1.0
    )

    pd.testing.assert_series_equal(
        result.cumulative_returns,
        expected,
    )


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def test_fold_weights_match_strategy(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    expected = strategy.weights()

    for fold in result.folds:

        pd.testing.assert_series_equal(
            fold.weights,
            expected,
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_evaluator_rejects_invalid_strategy(
    evaluator,
    splits,
):
    with pytest.raises(TypeError):

        evaluator.evaluate(
            object(),
            splits,
        )


def test_evaluator_rejects_empty_splits(
    evaluator,
    strategy,
):
    with pytest.raises(ValueError):

        evaluator.evaluate(
            strategy,
            [],
        )


def test_evaluator_rejects_invalid_split(
    evaluator,
    strategy,
):
    with pytest.raises(TypeError):

        evaluator.evaluate(
            strategy,
            [object()],
        )


# ---------------------------------------------------------------------------
# Index integrity
# ---------------------------------------------------------------------------


def test_oos_returns_preserve_test_index(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    expected_index = splits[0].test.index.append(
        splits[1].test.index,
    )

    pd.testing.assert_index_equal(
        result.returns.index,
        expected_index,
    )


def test_oos_returns_are_chronological(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    assert result.returns.index.is_monotonic_increasing


def test_oos_returns_have_no_duplicate_dates(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    assert not result.returns.index.has_duplicates


# ---------------------------------------------------------------------------
# Numerical integrity
# ---------------------------------------------------------------------------


def test_oos_returns_are_finite(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    assert np.isfinite(
        result.returns.to_numpy(),
    ).all()


def test_oos_cumulative_returns_are_finite(
    evaluator,
    strategy,
    splits,
):
    result = evaluator.evaluate(
        strategy,
        splits,
    )

    assert np.isfinite(
        result.cumulative_returns.to_numpy(),
    ).all()


# ---------------------------------------------------------------------------
# Empty / malformed test data
# ---------------------------------------------------------------------------


def test_evaluator_rejects_empty_test_fold(
    evaluator,
    strategy,
):
    split = WalkForwardSplit(
        train=pd.DataFrame(
            {
                "BTC": [100.0],
                "ETH": [50.0],
            }
        ),
        test=pd.DataFrame(
            columns=[
                "BTC",
                "ETH",
            ]
        ),
        fold=0,
    )

    with pytest.raises(
        ValueError,
    ):
        evaluator.evaluate(
            strategy,
            [split],
        )


def test_evaluator_rejects_missing_strategy_asset(
    evaluator,
):
    strategy = ResearchStrategy(
        name="Invalid OOS Strategy",
        signals=pd.Series(
            {
                "BTC": 0.5,
                "SOL": 0.5,
            }
        ),
        portfolio_method="signal_weighted",
    )

    split = WalkForwardSplit(
        train=pd.DataFrame(
            {
                "BTC": [100.0],
                "ETH": [50.0],
            }
        ),
        test=pd.DataFrame(
            {
                "BTC": [0.01],
                "ETH": [0.02],
            }
        ),
        fold=0,
    )

    with pytest.raises(
        KeyError,
    ):
        evaluator.evaluate(
            strategy,
            [split],
        )