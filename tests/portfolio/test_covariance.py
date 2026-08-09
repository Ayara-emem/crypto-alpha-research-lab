"""
Tests for CARL covariance estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.portfolio.covariance import (
    CovarianceEstimate,
    CovarianceEstimator,
)


@pytest.fixture
def returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BTC": [
                0.01,
                0.02,
                -0.01,
                0.03,
                0.01,
                -0.02,
            ],
            "ETH": [
                0.02,
                0.01,
                -0.02,
                0.02,
                0.03,
                -0.01,
            ],
            "SOL": [
                0.03,
                0.02,
                -0.03,
                0.04,
                0.01,
                -0.02,
            ],
        }
    )


def test_estimator_can_be_created():
    estimator = CovarianceEstimator()

    assert estimator.method == "sample"
    assert estimator.shrinkage is None


def test_sample_covariance_returns_estimate(
    returns,
):
    estimator = CovarianceEstimator(
        method="sample"
    )

    result = estimator.fit(
        returns
    )

    assert isinstance(
        result,
        CovarianceEstimate,
    )

    assert isinstance(
        result.matrix,
        pd.DataFrame,
    )


def test_sample_covariance_preserves_assets(
    returns,
):
    result = CovarianceEstimator(
        method="sample"
    ).fit(returns)

    assert list(
        result.matrix.index
    ) == list(
        returns.columns
    )

    assert list(
        result.matrix.columns
    ) == list(
        returns.columns
    )


def test_covariance_matrix_has_correct_shape(
    returns,
):
    result = CovarianceEstimator().fit(
        returns
    )

    assert result.matrix.shape == (
        returns.shape[1],
        returns.shape[1],
    )


def test_covariance_matrix_is_symmetric(
    returns,
):
    result = CovarianceEstimator().fit(
        returns
    )

    np.testing.assert_allclose(
        result.matrix.to_numpy(),
        result.matrix.to_numpy().T,
    )


def test_covariance_matrix_is_finite(
    returns,
):
    result = CovarianceEstimator().fit(
        returns
    )

    assert np.isfinite(
        result.matrix.to_numpy()
    ).all()


@pytest.mark.parametrize(
    "method",
    [
        "identity",
        "diagonal",
        "constant_correlation",
        "ledoit_wolf",
    ],
)
def test_supported_covariance_methods(
    returns,
    method,
):
    estimator = CovarianceEstimator(
        method=method
    )

    result = estimator.fit(
        returns
    )

    assert isinstance(
        result,
        CovarianceEstimate,
    )

    assert result.method == method


def test_explicit_shrinkage_requires_intensity():
    """
    Explicit shrinkage requires a shrinkage intensity
    at estimator construction time.
    """

    with pytest.raises(
        ValueError,
        match="shrinkage must be supplied",
    ):
        CovarianceEstimator(
            method="shrinkage"
        )


@pytest.mark.parametrize(
    "shrinkage",
    [
        -0.01,
        1.01,
        np.inf,
        -np.inf,
        np.nan,
    ],
)
def test_invalid_shrinkage_is_rejected(
    shrinkage,
):
    with pytest.raises(
        ValueError,
    ):
        CovarianceEstimator(
            method="shrinkage",
            shrinkage=shrinkage,
        )


def test_valid_shrinkage_is_preserved(
    returns,
):
    estimator = CovarianceEstimator(
        method="shrinkage",
        shrinkage=0.25,
    )

    result = estimator.fit(
        returns
    )

    assert result.shrinkage == 0.25


def test_shrinkage_matrix_is_finite(
    returns,
):
    result = CovarianceEstimator(
        method="shrinkage",
        shrinkage=0.25,
    ).fit(returns)

    assert np.isfinite(
        result.matrix.to_numpy()
    ).all()


def test_shrinkage_matrix_is_symmetric(
    returns,
):
    result = CovarianceEstimator(
        method="shrinkage",
        shrinkage=0.25,
    ).fit(returns)

    np.testing.assert_allclose(
        result.matrix.to_numpy(),
        result.matrix.to_numpy().T,
    )


def test_zero_shrinkage_matches_sample_covariance(
    returns,
):
    sample = CovarianceEstimator(
        method="sample"
    ).fit(returns)

    shrinkage = CovarianceEstimator(
        method="shrinkage",
        shrinkage=0.0,
    ).fit(returns)

    pd.testing.assert_frame_equal(
        sample.matrix,
        shrinkage.matrix,
    )


def test_full_shrinkage_matches_target(
    returns,
):
    """
    With lambda=1, shrinkage should equal the
    constant-correlation target.
    """

    shrinkage = CovarianceEstimator(
        method="shrinkage",
        shrinkage=1.0,
    ).fit(returns)

    target = CovarianceEstimator(
        method="constant_correlation"
    ).fit(returns)

    pd.testing.assert_frame_equal(
        shrinkage.matrix,
        target.matrix,
    )


def test_unknown_method_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported covariance method",
    ):
        CovarianceEstimator(
            method="unknown"
        )


def test_returns_must_be_dataframe():
    estimator = CovarianceEstimator()

    with pytest.raises(
        TypeError,
        match="DataFrame",
    ):
        estimator.fit(
            np.ones((10, 3))
        )


def test_empty_returns_are_rejected():
    estimator = CovarianceEstimator()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        estimator.fit(
            pd.DataFrame()
        )


def test_single_asset_returns_are_rejected():
    estimator = CovarianceEstimator()

    returns = pd.DataFrame(
        {
            "BTC": [
                0.01,
                0.02,
                0.03,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="at least two assets",
    ):
        estimator.fit(
            returns
        )


def test_duplicate_assets_are_rejected():
    estimator = CovarianceEstimator()

    returns = pd.DataFrame(
        [
            [0.01, 0.02],
            [0.02, 0.03],
        ],
        columns=[
            "BTC",
            "BTC",
        ],
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        estimator.fit(
            returns
        )


def test_non_finite_returns_are_rejected():
    estimator = CovarianceEstimator()

    returns = pd.DataFrame(
        {
            "BTC": [
                0.01,
                np.nan,
                0.02,
            ],
            "ETH": [
                0.02,
                0.01,
                0.03,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        estimator.fit(
            returns
        )