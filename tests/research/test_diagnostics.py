"""
Tests for CARL feature diagnostics.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.research.diagnostics import (
    feature_summary,
    feature_correlation,
    high_correlation_pairs,
    missing_feature_fraction,
)


@pytest.fixture
def sample_features():
    """
    Small deterministic feature matrix for diagnostics tests.
    """

    return pd.DataFrame(
        {
            "momentum": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
            "trend": [
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
            ],
            "inverse": [
                -1.0,
                -2.0,
                -3.0,
                -4.0,
                -5.0,
            ],
            "volume": [
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
            ],
        }
    )


def test_feature_summary_returns_dataframe(
    sample_features,
):
    result = feature_summary(
        sample_features
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_feature_summary_index(
    sample_features,
):
    result = feature_summary(
        sample_features
    )

    assert list(result.index) == list(
        sample_features.columns
    )


def test_feature_summary_expected_columns(
    sample_features,
):
    result = feature_summary(
        sample_features
    )

    expected_columns = [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]

    assert list(result.columns) == expected_columns


def test_feature_summary_values(
    sample_features,
):
    result = feature_summary(
        sample_features
    )

    assert result.loc[
        "momentum",
        "mean",
    ] == pytest.approx(3.0)

    assert result.loc[
        "momentum",
        "count",
    ] == pytest.approx(5.0)


def test_feature_summary_empty_dataframe():
    features = pd.DataFrame()

    with pytest.raises(ValueError):
        feature_summary(features)


def test_feature_correlation_returns_dataframe(
    sample_features,
):
    result = feature_correlation(
        sample_features
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_feature_correlation_shape(
    sample_features,
):
    result = feature_correlation(
        sample_features
    )

    expected_size = len(
        sample_features.columns
    )

    assert result.shape == (
        expected_size,
        expected_size,
    )


def test_feature_correlation_diagonal(
    sample_features,
):
    result = feature_correlation(
        sample_features
    )

    diagonal = np.diag(result)

    assert np.allclose(
        diagonal,
        1.0,
    )


def test_feature_correlation_symmetry(
    sample_features,
):
    result = feature_correlation(
        sample_features
    )

    pd.testing.assert_frame_equal(
        result,
        result.T,
    )


def test_feature_correlation_perfect_positive(
    sample_features,
):
    result = feature_correlation(
        sample_features
    )

    assert result.loc[
        "momentum",
        "trend",
    ] == pytest.approx(1.0)


def test_feature_correlation_perfect_negative(
    sample_features,
):
    result = feature_correlation(
        sample_features
    )

    assert result.loc[
        "momentum",
        "inverse",
    ] == pytest.approx(-1.0)


def test_feature_correlation_empty_dataframe():
    features = pd.DataFrame()

    with pytest.raises(ValueError):
        feature_correlation(features)


def test_high_correlation_pairs_returns_dataframe(
    sample_features,
):
    result = high_correlation_pairs(
        sample_features,
        threshold=0.90,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_high_correlation_pairs_columns(
    sample_features,
):
    result = high_correlation_pairs(
        sample_features,
        threshold=0.90,
    )

    expected_columns = [
        "feature_1",
        "feature_2",
        "correlation",
    ]

    assert list(result.columns) == expected_columns


def test_high_correlation_pairs_detects_positive(
    sample_features,
):
    result = high_correlation_pairs(
        sample_features,
        threshold=0.90,
    )

    pairs = set(
        zip(
            result["feature_1"],
            result["feature_2"],
        )
    )

    assert (
        "momentum",
        "trend",
    ) in pairs


def test_high_correlation_pairs_detects_negative(
    sample_features,
):
    result = high_correlation_pairs(
        sample_features,
        threshold=0.90,
    )

    pairs = set(
        zip(
            result["feature_1"],
            result["feature_2"],
        )
    )

    assert (
        "momentum",
        "inverse",
    ) in pairs


@pytest.mark.parametrize(
    "threshold",
    [
        0,
        -0.10,
        1.10,
    ],
)
def test_high_correlation_pairs_invalid_threshold(
    sample_features,
    threshold,
):
    with pytest.raises(ValueError):
        high_correlation_pairs(
            sample_features,
            threshold=threshold,
        )


def test_high_correlation_pairs_empty_dataframe():
    features = pd.DataFrame()

    with pytest.raises(ValueError):
        high_correlation_pairs(
            features
        )


def test_high_correlation_pairs_no_duplicates(
    sample_features,
):
    result = high_correlation_pairs(
        sample_features,
        threshold=0.90,
    )

    pairs = list(
        zip(
            result["feature_1"],
            result["feature_2"],
        )
    )

    reverse_pairs = [
        (feature_2, feature_1)
        for feature_1, feature_2
        in pairs
    ]

    assert not any(
        pair in pairs
        for pair in reverse_pairs
    )


def test_high_correlation_pairs_excludes_self_pairs(
    sample_features,
):
    result = high_correlation_pairs(
        sample_features,
        threshold=0.90,
    )

    assert not (
        result["feature_1"]
        == result["feature_2"]
    ).any()


def test_missing_feature_fraction_returns_series(
    sample_features,
):
    result = missing_feature_fraction(
        sample_features
    )

    assert isinstance(
        result,
        pd.Series,
    )


def test_missing_feature_fraction_no_missing(
    sample_features,
):
    result = missing_feature_fraction(
        sample_features
    )

    assert (
        result == 0.0
    ).all()


def test_missing_feature_fraction_values():
    features = pd.DataFrame(
        {
            "feature_a": [
                1.0,
                np.nan,
                3.0,
                np.nan,
            ],
            "feature_b": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "feature_c": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )

    result = missing_feature_fraction(
        features
    )

    assert result[
        "feature_a"
    ] == pytest.approx(0.50)

    assert result[
        "feature_b"
    ] == pytest.approx(0.0)

    assert result[
        "feature_c"
    ] == pytest.approx(1.0)


def test_missing_feature_fraction_empty_dataframe():
    features = pd.DataFrame()

    with pytest.raises(ValueError):
        missing_feature_fraction(
            features
        )