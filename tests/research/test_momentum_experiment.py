"""
Tests for the CARL cross-sectional momentum research experiment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.research.momentum_experiment import (
    MomentumExperimentResult,
    run_momentum_experiment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prices() -> pd.DataFrame:
    """
    Deterministic multi-asset price history.
    """

    index = pd.date_range(
        "2024-01-01",
        periods=60,
        freq="D",
    )

    return pd.DataFrame(
        {
            "BTC": np.linspace(
                100.0,
                160.0,
                len(index),
            ),
            "ETH": np.linspace(
                50.0,
                85.0,
                len(index),
            ),
            "SOL": np.linspace(
                20.0,
                45.0,
                len(index),
            ),
        },
        index=index,
    )


@pytest.fixture
def experiment_result(
    prices: pd.DataFrame,
) -> MomentumExperimentResult:
    """
    Standard deterministic momentum experiment.
    """

    return run_momentum_experiment(
        prices=prices,
        train_size=30,
        test_size=10,
        lookback=10,
    )


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_experiment_returns_expected_result_type(
    experiment_result: MomentumExperimentResult,
):
    """
    The experiment should return its public result type.
    """

    assert isinstance(
        experiment_result,
        MomentumExperimentResult,
    )


def test_result_contains_oos_returns(
    experiment_result: MomentumExperimentResult,
):
    """
    Result should contain a portfolio return series.
    """

    assert isinstance(
        experiment_result.returns,
        pd.Series,
    )

    assert not experiment_result.returns.empty


def test_result_contains_cumulative_returns(
    experiment_result: MomentumExperimentResult,
):
    """
    Result should contain cumulative OOS returns.
    """

    assert isinstance(
        experiment_result.cumulative_returns,
        pd.Series,
    )


def test_result_contains_fold_results(
    experiment_result: MomentumExperimentResult,
):
    """
    Every walk-forward fold should be retained.
    """

    assert isinstance(
        experiment_result.folds,
        tuple,
    )

    assert len(
        experiment_result.folds
    ) > 0


# ---------------------------------------------------------------------------
# Fold structure
# ---------------------------------------------------------------------------


def test_fold_count_is_correct(
    experiment_result: MomentumExperimentResult,
):
    """
    60 observations with train=30 and test=10
    should produce three folds.
    """

    assert len(
        experiment_result.folds
    ) == 3


def test_each_fold_contains_oos_returns(
    experiment_result: MomentumExperimentResult,
):
    """
    Every fold must contain non-empty OOS returns.
    """

    for fold in experiment_result.folds:

        assert isinstance(
            fold.portfolio_returns,
            pd.Series,
        )

        assert not fold.portfolio_returns.empty


def test_each_fold_contains_weights(
    experiment_result: MomentumExperimentResult,
):
    """
    Every fold should retain portfolio weights.
    """

    for fold in experiment_result.folds:

        assert isinstance(
            fold.weights,
            pd.Series,
        )

        assert not fold.weights.empty


# ---------------------------------------------------------------------------
# OOS integrity
# ---------------------------------------------------------------------------


def test_oos_returns_are_chronological(
    experiment_result: MomentumExperimentResult,
):
    """
    Aggregated OOS returns must be chronological.
    """

    assert (
        experiment_result.returns.index
        .is_monotonic_increasing
    )


def test_oos_returns_have_no_duplicate_dates(
    experiment_result: MomentumExperimentResult,
):
    """
    A date may not appear in more than one OOS fold.
    """

    assert not (
        experiment_result.returns.index
        .has_duplicates
    )


def test_oos_returns_are_finite(
    experiment_result: MomentumExperimentResult,
):
    """
    OOS returns must be finite.
    """

    assert np.isfinite(
        experiment_result.returns.to_numpy()
    ).all()


def test_oos_dates_belong_to_test_periods(
    prices: pd.DataFrame,
    experiment_result: MomentumExperimentResult,
):
    """
    Every aggregated OOS observation must correspond
    to a declared test period.
    """

    expected_dates: list[pd.Timestamp] = []

    for fold in experiment_result.folds:

        expected_dates.extend(
            fold.portfolio_returns.index
        )

    expected_index = pd.DatetimeIndex(
        expected_dates
    )

    assert experiment_result.returns.index.equals(
        expected_index.sort_values()
    )


# ---------------------------------------------------------------------------
# Return calculations
# ---------------------------------------------------------------------------


def test_cumulative_returns_match_oos_returns(
    experiment_result: MomentumExperimentResult,
):
    """
    Cumulative returns must equal compounded OOS returns.
    """

    expected = (
        (1.0 + experiment_result.returns)
        .cumprod()
        - 1.0
    )

    pd.testing.assert_series_equal(
        experiment_result.cumulative_returns,
        expected,
    )


def test_cumulative_returns_align_with_oos_returns(
    experiment_result: MomentumExperimentResult,
):
    """
    Cumulative and periodic returns must have
    identical indexes.
    """

    assert (
        experiment_result.cumulative_returns.index.equals(
            experiment_result.returns.index
        )
    )


def test_first_oos_return_is_retained(
    prices: pd.DataFrame,
):
    """
    The first test-period return should not be discarded.

    The final training price acts as the return anchor,
    allowing the first test observation to generate an
    OOS return.
    """

    result = run_momentum_experiment(
        prices=prices,
        train_size=30,
        test_size=10,
        lookback=10,
    )

    first_fold = result.folds[0]

    expected_first_date = prices.index[30]

    assert (
        first_fold.portfolio_returns.index[0]
        == expected_first_date
    )


# ---------------------------------------------------------------------------
# Train/test separation
# ---------------------------------------------------------------------------


def test_first_fold_starts_after_training_window(
    prices: pd.DataFrame,
    experiment_result: MomentumExperimentResult,
):
    """
    The first OOS return must occur after the complete
    initial training window.
    """

    first_fold = experiment_result.folds[0]

    assert (
        first_fold.portfolio_returns.index[0]
        >= prices.index[30]
    )


def test_fold_metadata_marks_oos(
    experiment_result: MomentumExperimentResult,
):
    """
    Every fold must explicitly identify itself as OOS.
    """

    for fold in experiment_result.folds:

        assert (
            fold.metadata["out_of_sample"]
            is True
        )

        assert (
            fold.metadata["validation"]
            == "walk_forward"
        )


def test_fold_metadata_contains_dates(
    experiment_result: MomentumExperimentResult,
):
    """
    Fold provenance should contain train/test boundaries.
    """

    for fold in experiment_result.folds:

        metadata = fold.metadata

        assert (
            "train_start"
            in metadata
        )

        assert (
            "train_end"
            in metadata
        )

        assert (
            "test_start"
            in metadata
        )

        assert (
            "test_end"
            in metadata
        )


# ---------------------------------------------------------------------------
# Parameter provenance
# ---------------------------------------------------------------------------


def test_parameters_are_recorded(
    experiment_result: MomentumExperimentResult,
):
    """
    Experiment parameters must be retained.
    """

    assert (
        experiment_result.parameters[
            "train_size"
        ]
        == 30
    )

    assert (
        experiment_result.parameters[
            "test_size"
        ]
        == 10
    )

    assert (
        experiment_result.parameters[
            "lookback"
        ]
        == 10
    )


def test_experiment_metadata_is_present(
    experiment_result: MomentumExperimentResult,
):
    """
    Result should identify the research experiment.
    """

    assert (
        experiment_result.metadata[
            "experiment"
        ]
        == "cross_sectional_momentum"
    )

    assert (
        experiment_result.metadata[
            "validation"
        ]
        == "walk_forward"
    )

    assert (
        experiment_result.metadata[
            "out_of_sample"
        ]
        is True
    )


# ---------------------------------------------------------------------------
# Expanding / rolling validation
# ---------------------------------------------------------------------------


def test_expanding_validation_is_supported(
    prices: pd.DataFrame,
):
    """
    Expanding walk-forward validation should execute.
    """

    result = run_momentum_experiment(
        prices=prices,
        train_size=20,
        test_size=10,
        lookback=5,
        expanding=True,
    )

    assert len(
        result.folds
    ) > 0


def test_rolling_validation_is_supported(
    prices: pd.DataFrame,
):
    """
    Rolling walk-forward validation should execute.
    """

    result = run_momentum_experiment(
        prices=prices,
        train_size=20,
        test_size=10,
        lookback=5,
        expanding=False,
    )

    assert len(
        result.folds
    ) > 0


# ---------------------------------------------------------------------------
# Lookback validation
# ---------------------------------------------------------------------------


def test_lookback_must_be_positive(
    prices: pd.DataFrame,
):
    """
    Zero lookback is invalid.
    """

    with pytest.raises(
        ValueError,
    ):
        run_momentum_experiment(
            prices=prices,
            train_size=30,
            test_size=10,
            lookback=0,
        )


def test_lookback_must_be_smaller_than_training_size(
    prices: pd.DataFrame,
):
    """
    Lookback cannot consume the entire training window.
    """

    with pytest.raises(
        ValueError,
    ):
        run_momentum_experiment(
            prices=prices,
            train_size=10,
            test_size=10,
            lookback=10,
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_prices_must_be_dataframe():
    """
    Non-DataFrame prices should be rejected.
    """

    with pytest.raises(
        TypeError,
    ):
        run_momentum_experiment(
            prices=np.ones(
                (20, 3)
            ),
            train_size=10,
            test_size=5,
            lookback=5,
        )


def test_empty_prices_are_rejected():
    """
    Empty price data should be rejected.
    """

    with pytest.raises(
        ValueError,
    ):
        run_momentum_experiment(
            prices=pd.DataFrame(),
            train_size=10,
            test_size=5,
            lookback=5,
        )


def test_invalid_train_size_is_rejected(
    prices: pd.DataFrame,
):
    """
    Training size must be positive.
    """

    with pytest.raises(
        ValueError,
    ):
        run_momentum_experiment(
            prices=prices,
            train_size=0,
            test_size=5,
            lookback=2,
        )


def test_invalid_test_size_is_rejected(
    prices: pd.DataFrame,
):
    """
    Test size must be positive.
    """

    with pytest.raises(
        ValueError,
    ):
        run_momentum_experiment(
            prices=prices,
            train_size=20,
            test_size=0,
            lookback=5,
        )


def test_non_chronological_prices_are_rejected(
    prices: pd.DataFrame,
):
    """
    Walk-forward research requires chronological prices.
    """

    shuffled = prices.iloc[
        ::-1
    ]

    with pytest.raises(
        ValueError,
    ):
        run_momentum_experiment(
            prices=shuffled,
            train_size=20,
            test_size=10,
            lookback=5,
        )


def test_duplicate_dates_are_rejected(
    prices: pd.DataFrame,
):
    """
    Duplicate timestamps would invalidate OOS provenance.
    """

    duplicated = pd.concat(
        [
            prices,
            prices.iloc[[0]],
        ]
    )

    with pytest.raises(
        ValueError,
    ):
        run_momentum_experiment(
            prices=duplicated,
            train_size=20,
            test_size=10,
            lookback=5,
        )


def test_empty_strategy_name_is_rejected(
    prices: pd.DataFrame,
):
    """
    Strategy names must be non-empty.
    """

    with pytest.raises(
        ValueError,
    ):
        run_momentum_experiment(
            prices=prices,
            train_size=20,
            test_size=10,
            lookback=5,
            strategy_name="",
        )


def test_insufficient_data_is_rejected(
    prices: pd.DataFrame,
):
    """
    The experiment must reject data that cannot produce
    a complete walk-forward evaluation.
    """

    with pytest.raises(
        ValueError,
    ):
        run_momentum_experiment(
            prices=prices.iloc[:15],
            train_size=20,
            test_size=10,
            lookback=5,
        )