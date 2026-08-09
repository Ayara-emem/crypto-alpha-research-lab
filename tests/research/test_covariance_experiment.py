"""
Tests for the walk-forward covariance comparison experiment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.research.covariance_experiment import (
    CovarianceExperimentResult,
    run_covariance_experiment,
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
        periods=40,
        freq="D",
    )

    t = np.arange(
        len(index),
        dtype=float,
    )

    return pd.DataFrame(
        {
            "BTC": 100.0 + 1.50 * t,
            "ETH": 60.0 + 0.90 * t + 0.02 * t**2,
            "SOL": 40.0 + 0.55 * t + 0.03 * t**2,
        },
        index=index,
    )


@pytest.fixture
def experiment_kwargs() -> dict:
    """
    Common walk-forward configuration.
    """

    return {
        "train_size": 12,
        "test_size": 5,
    }


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


def test_covariance_experiment_returns_expected_type(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    The experiment should return its dedicated result object.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    assert isinstance(
        result,
        CovarianceExperimentResult,
    )


def test_sample_covariance_experiment_runs(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Sample covariance should be a supported methodology.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    assert result.method == "sample"
    assert len(result.folds) > 0


def test_shrinkage_covariance_experiment_runs(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Explicit shrinkage should be supported.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="shrinkage",
        shrinkage=0.25,
        **experiment_kwargs,
    )

    assert result.method == "shrinkage"
    assert result.parameters["shrinkage"] == 0.25
    assert len(result.folds) > 0


def test_ledoit_wolf_experiment_runs(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Ledoit-Wolf covariance should be supported.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="ledoit_wolf",
        **experiment_kwargs,
    )

    assert result.method == "ledoit_wolf"
    assert len(result.folds) > 0


# ---------------------------------------------------------------------------
# Return integrity
# ---------------------------------------------------------------------------


def test_experiment_returns_are_series(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Aggregated OOS returns should be a Series.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    assert isinstance(
        result.returns,
        pd.Series,
    )


def test_cumulative_returns_are_series(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Cumulative OOS returns should be a Series.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    assert isinstance(
        result.cumulative_returns,
        pd.Series,
    )


def test_cumulative_returns_align_with_returns(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Cumulative returns must have exactly the same index
    as aggregated OOS returns.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    assert result.cumulative_returns.index.equals(
        result.returns.index,
    )


def test_oos_returns_are_finite(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    OOS returns must contain finite observations only.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    assert np.isfinite(
        result.returns.to_numpy(),
    ).all()


def test_cumulative_returns_are_finite(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Cumulative returns must contain finite observations only.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    assert np.isfinite(
        result.cumulative_returns.to_numpy(),
    ).all()


# ---------------------------------------------------------------------------
# Walk-forward / OOS integrity
# ---------------------------------------------------------------------------


def test_oos_returns_are_chronological(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Aggregated OOS returns must remain chronological.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    assert result.returns.index.is_monotonic_increasing


def test_oos_returns_have_no_duplicate_dates(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Walk-forward test windows must not create duplicate
    OOS observations.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    assert not result.returns.index.has_duplicates


def test_oos_returns_belong_to_test_periods(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Every OOS observation must belong to its fold's test period.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    for fold in result.folds:
        assert (
            fold.portfolio_returns.index.min()
            >= fold.metadata["test_start"]
        )

        assert (
            fold.portfolio_returns.index.max()
            <= fold.metadata["test_end"]
        )


def test_each_fold_is_out_of_sample(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Every fold must explicitly identify itself as OOS.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    assert all(
        fold.metadata["out_of_sample"] is True
        for fold in result.folds
    )


# ---------------------------------------------------------------------------
# Covariance methodology metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [
        "sample",
        "shrinkage",
        "ledoit_wolf",
    ],
)
def test_fold_metadata_records_covariance_method(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
    method: str,
):
    """
    Every fold should record the covariance methodology.
    """

    kwargs = dict(
        experiment_kwargs,
    )

    if method == "shrinkage":
        kwargs["shrinkage"] = 0.25

    result = run_covariance_experiment(
        prices=prices,
        method=method,
        **kwargs,
    )

    assert all(
        fold.metadata["covariance_method"] == method
        for fold in result.folds
    )


def test_shrinkage_intensity_is_recorded(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Explicit shrinkage intensity should be preserved
    in experiment and fold metadata.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="shrinkage",
        shrinkage=0.30,
        **experiment_kwargs,
    )

    assert result.parameters["shrinkage"] == 0.30

    assert all(
        fold.metadata["shrinkage"] == 0.30
        for fold in result.folds
    )


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------


def test_each_fold_contains_portfolio_weights(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Each fold must contain the resulting GMV weights.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    assert all(
        isinstance(
            fold.weights,
            pd.Series,
        )
        for fold in result.folds
    )


def test_each_fold_weights_are_finite(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Portfolio weights must be finite.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    assert all(
        np.isfinite(
            fold.weights.to_numpy()
        ).all()
        for fold in result.folds
    )


def test_each_fold_uses_global_minimum_variance(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    The experiment must explicitly record GMV construction.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    assert all(
        fold.metadata[
            "portfolio_method"
        ] == "global_minimum_variance"
        for fold in result.folds
    )


# ---------------------------------------------------------------------------
# Fold metadata
# ---------------------------------------------------------------------------


def test_fold_numbers_are_preserved(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Fold identifiers should be retained.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    fold_numbers = [
        fold.metadata["fold"]
        for fold in result.folds
    ]

    assert fold_numbers == list(
        range(len(result.folds))
    )


def test_fold_train_test_metadata_is_present(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Every fold should retain its train/test boundaries.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    for fold in result.folds:
        assert "train_start" in fold.metadata
        assert "train_end" in fold.metadata
        assert "test_start" in fold.metadata
        assert "test_end" in fold.metadata


def test_test_period_follows_training_period(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    The test period must begin after the training period.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    for fold in result.folds:
        assert (
            fold.metadata["test_start"]
            > fold.metadata["train_end"]
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_non_dataframe_prices_are_rejected(
    experiment_kwargs: dict,
):
    """
    Prices must be supplied as a DataFrame.
    """

    with pytest.raises(
        TypeError,
        match="prices must be a pandas DataFrame",
    ):
        run_covariance_experiment(
            prices=np.ones(
                (20, 3)
            ),
            method="sample",
            **experiment_kwargs,
        )


def test_empty_prices_are_rejected(
    experiment_kwargs: dict,
):
    """
    Empty price data must be rejected.
    """

    with pytest.raises(
        ValueError,
        match="prices cannot be empty",
    ):
        run_covariance_experiment(
            prices=pd.DataFrame(),
            method="sample",
            **experiment_kwargs,
        )


def test_non_chronological_prices_are_rejected(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Prices must be chronologically ordered.
    """

    shuffled = prices.iloc[
        ::-1
    ]

    with pytest.raises(
        ValueError,
        match="chronological",
    ):
        run_covariance_experiment(
            prices=shuffled,
            method="sample",
            **experiment_kwargs,
        )


def test_duplicate_price_dates_are_rejected(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Duplicate timestamps must be rejected.
    """

    duplicated = pd.concat(
    [
        prices,
        prices.iloc[[-1]],
    ]
)

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        run_covariance_experiment(
            prices=duplicated,
            method="sample",
            **experiment_kwargs,
        )


@pytest.mark.parametrize(
    "method",
    [
        "unknown",
        "minimum_variance",
        "oracle",
    ],
)
def test_unknown_covariance_method_is_rejected(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
    method: str,
):
    """
    Unsupported covariance methodologies must be rejected.
    """

    with pytest.raises(
        ValueError,
        match="Unsupported covariance method",
    ):
        run_covariance_experiment(
            prices=prices,
            method=method,
            **experiment_kwargs,
        )


def test_shrinkage_requires_intensity(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Explicit shrinkage must specify its intensity.
    """

    with pytest.raises(
        ValueError,
        match="shrinkage",
    ):
        run_covariance_experiment(
            prices=prices,
            method="shrinkage",
            **experiment_kwargs,
        )


@pytest.mark.parametrize(
    "shrinkage",
    [
        -0.01,
        1.01,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_shrinkage_is_rejected(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
    shrinkage: float,
):
    """
    Shrinkage intensity must lie in [0, 1].
    """

    with pytest.raises(
        ValueError,
    ):
        run_covariance_experiment(
            prices=prices,
            method="shrinkage",
            shrinkage=shrinkage,
            **experiment_kwargs,
        )


def test_zero_train_size_is_rejected(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Training size must be positive.
    """

    with pytest.raises(
        ValueError,
        match="train_size",
    ):
        run_covariance_experiment(
            prices=prices,
            method="sample",
            train_size=0,
            test_size=5,
        )


def test_zero_test_size_is_rejected(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Test size must be positive.
    """

    with pytest.raises(
        ValueError,
        match="test_size",
    ):
        run_covariance_experiment(
            prices=prices,
            method="sample",
            train_size=12,
            test_size=0,
        )


def test_insufficient_data_is_rejected(
    prices: pd.DataFrame,
):
    """
    The experiment should reject data that cannot produce
    a valid walk-forward split.
    """

    with pytest.raises(
        ValueError,
    ):
        run_covariance_experiment(
            prices=prices.iloc[:10],
            method="sample",
            train_size=20,
            test_size=5,
        )


# ---------------------------------------------------------------------------
# Experiment-level metadata
# ---------------------------------------------------------------------------


def test_experiment_metadata_identifies_research_design(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Result metadata should make the research design explicit.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    assert (
        result.metadata["experiment"]
        == "covariance_comparison"
    )

    assert (
        result.metadata["validation"]
        == "walk_forward"
    )

    assert (
        result.metadata["out_of_sample"]
        is True
    )


def test_fold_count_matches_result(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Metadata fold count must agree with the actual result.
    """

    result = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    assert (
        result.metadata["fold_count"]
        == len(result.folds)
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_experiment_is_deterministic(
    prices: pd.DataFrame,
    experiment_kwargs: dict,
):
    """
    Repeated runs on identical deterministic data should
    produce identical OOS results.
    """

    first = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    second = run_covariance_experiment(
        prices=prices,
        method="sample",
        **experiment_kwargs,
    )

    pd.testing.assert_series_equal(
        first.returns,
        second.returns,
    )

    pd.testing.assert_series_equal(
        first.cumulative_returns,
        second.cumulative_returns,
    )