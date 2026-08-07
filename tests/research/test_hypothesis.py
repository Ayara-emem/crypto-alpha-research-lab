"""
Tests for hypothesis.py
"""

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.research.hypothesis import (
    evaluate_alpha,
    evaluate_alpha_universe,
    bonferroni_correction,
    benjamini_hochberg,
    bootstrap_ic,
    permutation_ic,
)

@pytest.fixture
def sample_research():

    rng = np.random.default_rng(42)

    n = 200

    return pd.DataFrame(
        {
            "momentum": rng.normal(size=n),
            "volatility": rng.normal(size=n),
            "volume": rng.normal(size=n),
            "target": rng.normal(size=n),
        }
    )


def test_evaluate_alpha_returns_series(
    sample_research,
):

    result = evaluate_alpha(
        sample_research,
        feature="momentum",
        target="target",
    )

    assert isinstance(
        result,
        pd.Series,
    )

def test_evaluate_alpha_contains_expected_fields(
    sample_research,
):

    result = evaluate_alpha(
        sample_research,
        feature="momentum",
        target="target",
    )

    expected = {
        "feature",
        "pearson",
        "pearson_p",
        "information_coefficient",
        "ic_p",
        "significant",
    }

    assert expected.issubset(
        result.index
    )


def test_invalid_feature_raises(
    sample_research,
):

    with pytest.raises(ValueError):

        evaluate_alpha(
            sample_research,
            feature="missing",
            target="target",
        )


def test_universe_returns_dataframe(
    sample_research,
):

    result = evaluate_alpha_universe(
        sample_research,
        target="target",
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_universe_contains_all_features(
    sample_research,
):

    result = evaluate_alpha_universe(
        sample_research,
        target="target",
    )

    assert set(
        result["feature"]
    ) == {
        "momentum",
        "volatility",
        "volume",
    }


def test_universe_contains_significance(
    sample_research,
):

    result = evaluate_alpha_universe(
        sample_research,
        target="target",
    )

    assert (
        "significant"
        in result.columns
    )

def test_bonferroni_returns_dataframe(
    sample_research,
):

    report = evaluate_alpha_universe(
        sample_research,
        target="target",
    )

    result = bonferroni_correction(
        report,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_bonferroni_columns(
    sample_research,
):

    report = evaluate_alpha_universe(
        sample_research,
        target="target",
    )

    result = bonferroni_correction(
        report,
    )

    assert (
        "bonferroni_alpha"
        in result.columns
    )

    assert (
        "bonferroni_significant"
        in result.columns
    )

def test_bh_returns_dataframe(
    sample_research,
):

    report = evaluate_alpha_universe(
        sample_research,
        target="target",
    )

    result = benjamini_hochberg(
        report,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_bh_columns(
    sample_research,
):

    report = evaluate_alpha_universe(
        sample_research,
        target="target",
    )

    result = benjamini_hochberg(
        report,
    )

    assert (
        "bh_threshold"
        in result.columns
    )

    assert (
        "bh_significant"
        in result.columns
    )

def test_bootstrap_returns_dataframe(
    sample_research,
):

    result = bootstrap_ic(
        sample_research,
        feature="momentum",
        target="target",
        n_bootstrap=100,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_bootstrap_columns(
    sample_research,
):

    result = bootstrap_ic(
        sample_research,
        feature="momentum",
        target="target",
        n_bootstrap=100,
    )

    expected = {
        "observed_ic",
        "bootstrap_mean",
        "bootstrap_std",
        "lower_ci",
        "upper_ci",
    }

    assert expected.issubset(
        result.columns
    )

def test_bootstrap_ci_order(
    sample_research,
):

    result = bootstrap_ic(
        sample_research,
        feature="momentum",
        target="target",
        n_bootstrap=100,
    )

    assert (
        result["lower_ci"].iloc[0]
        <=
        result["upper_ci"].iloc[0]
    )


def test_permutation_returns_dataframe(
    sample_research,
):

    result = permutation_ic(
        sample_research,
        feature="momentum",
        target="target",
        n_permutations=100,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

def test_permutation_columns(
    sample_research,
):

    result = permutation_ic(
        sample_research,
        feature="momentum",
        target="target",
        n_permutations=100,
    )

    assert {
        "observed_ic",
        "permutation_p_value",
    }.issubset(
        result.columns
    )

def test_permutation_probability(
    sample_research,
):

    result = permutation_ic(
        sample_research,
        feature="momentum",
        target="target",
        n_permutations=100,
    )

    value = result[
        "permutation_p_value"
    ].iloc[0]

    assert 0 <= value <= 1

def test_input_not_modified(
    sample_research,
):

    original = sample_research.copy()

    evaluate_alpha_universe(
        sample_research,
        target="target",
    )

    pd.testing.assert_frame_equal(
        original,
        sample_research,
    )


def test_repeatability(
    sample_research,
):

    first = evaluate_alpha_universe(
        sample_research,
        target="target",
    )

    second = evaluate_alpha_universe(
        sample_research,
        target="target",
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


