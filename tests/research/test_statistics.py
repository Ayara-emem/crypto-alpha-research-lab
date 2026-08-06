"""
Tests for research statistics.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.research.statistics import (
    feature_target_correlation,
    feature_target_rank_correlation,
    information_coefficient,
    correlation_matrix,
    feature_p_values,
    summary_statistics,
)

@pytest.fixture
def sample_research():

    rng = np.random.default_rng(42)

    n = 100

    return pd.DataFrame(
        {
            "feature_a": rng.normal(size=n),
            "feature_b": rng.normal(size=n),
            "feature_c": rng.normal(size=n),
            "target": rng.normal(size=n),
        }
    )

def test_invalid_target_raises(
    sample_research,
):
    with pytest.raises(ValueError):
        feature_target_correlation(
            sample_research,
            target="missing",
        )

def test_empty_dataframe_raises():

    with pytest.raises(ValueError):
        feature_target_correlation(
            pd.DataFrame(),
            target="target",
        )


def test_non_dataframe_raises():

    with pytest.raises(TypeError):
        feature_target_correlation(
            [],
            target="target",
        )

def test_feature_target_correlation_returns_dataframe(
    sample_research,
):

    result = feature_target_correlation(
        sample_research,
        target="target",
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

def test_feature_target_correlation_columns(
    sample_research,
):

    result = feature_target_correlation(
        sample_research,
        target="target",
    )

    assert list(result.columns) == [
        "feature",
        "correlation",
    ]

def test_all_features_present(
    sample_research,
):

    result = feature_target_correlation(
        sample_research,
        target="target",
    )

    assert set(result["feature"]) == {
        "feature_a",
        "feature_b",
        "feature_c",
    }

def test_correlations_are_finite(
    sample_research,
):

    result = feature_target_correlation(
        sample_research,
        target="target",
    )

    assert np.isfinite(
        result["correlation"],
    ).all()

def test_sorted_by_absolute_correlation(
    sample_research,
):

    result = feature_target_correlation(
        sample_research,
        target="target",
    )

    values = result[
        "correlation"
    ].abs()

    assert values.is_monotonic_decreasing

def test_rank_correlation_returns_dataframe(
    sample_research,
):

    result = feature_target_rank_correlation(
        sample_research,
        target="target",
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

def test_rank_columns(
    sample_research,
):

    result = feature_target_rank_correlation(
        sample_research,
        target="target",
    )

    assert list(result.columns) == [
        "feature",
        "correlation",
    ]

def test_rank_sorted(
    sample_research,
):

    result = feature_target_rank_correlation(
        sample_research,
        target="target",
    )

    assert (
        result["correlation"]
        .abs()
        .is_monotonic_decreasing
    )

def test_information_coefficient_matches_rank(
    sample_research,
):

    ic = information_coefficient(
        sample_research,
        target="target",
    )

    rank = feature_target_rank_correlation(
        sample_research,
        target="target",
    )

    pd.testing.assert_frame_equal(
        ic,
        rank,
    )

def test_correlation_matrix_returns_dataframe(
    sample_research,
):

    result = correlation_matrix(
        sample_research,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

def test_matrix_square(
    sample_research,
):

    result = correlation_matrix(
        sample_research,
    )

    assert result.shape[0] == result.shape[1]

def test_matrix_square(
    sample_research,
):

    result = correlation_matrix(
        sample_research,
    )

    assert result.shape[0] == result.shape[1]

def test_matrix_symmetric(
    sample_research,
):

    result = correlation_matrix(
        sample_research,
    )

    pd.testing.assert_frame_equal(
        result,
        result.T,
    )

def test_matrix_diagonal_one(
    sample_research,
):

    result = correlation_matrix(
        sample_research,
    )

    assert np.allclose(
        np.diag(result),
        1,
    )

def test_p_values_returns_dataframe(
    sample_research,
):

    result = feature_p_values(
        sample_research,
        target="target",
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

def test_p_value_columns(
    sample_research,
):

    result = feature_p_values(
        sample_research,
        target="target",
    )

    assert list(result.columns) == [
        "feature",
        "correlation",
        "p_value",
    ]

def test_p_values_between_zero_and_one(
    sample_research,
):

    result = feature_p_values(
        sample_research,
        target="target",
    )

    assert (
        (
            result["p_value"] >= 0
        )
        &
        (
            result["p_value"] <= 1
        )
    ).all()

def test_p_values_sorted(
    sample_research,
):

    result = feature_p_values(
        sample_research,
        target="target",
    )

    assert result[
        "p_value"
    ].is_monotonic_increasing

def test_summary_returns_dataframe(
    sample_research,
):

    result = summary_statistics(
        sample_research,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

def test_summary_contains_skew(
    sample_research,
):

    result = summary_statistics(
        sample_research,
    )

    assert "skew" in result.columns

def test_summary_contains_kurtosis(
    sample_research,
):

    result = summary_statistics(
        sample_research,
    )

    assert "kurtosis" in result.columns

def test_summary_rows(
    sample_research,
):

    result = summary_statistics(
        sample_research,
    )

    assert len(result) == len(
        sample_research.columns
    )

def test_functions_do_not_modify_input(
    sample_research,
):

    original = sample_research.copy()

    feature_target_correlation(
        sample_research,
        target="target",
    )

    pd.testing.assert_frame_equal(
        sample_research,
        original,
    )

def test_repeatability(
    sample_research,
):

    first = feature_target_correlation(
        sample_research,
        target="target",
    )

    second = feature_target_correlation(
        sample_research,
        target="target",
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )

