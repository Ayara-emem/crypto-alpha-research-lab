"""
Tests for signal transformation.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.signals.transform import (
    build_signal,
    invert_signal,
    rescale_signal,
)

@pytest.fixture
def sample_signal():

    return pd.Series(
        [-2.0, -1.0, 0.0, 0.5, 2.0],
        name="alpha",
    )


@pytest.fixture
def positive_signal():

    return pd.Series(
        [0.2, 0.5, 1.0, 1.5],
        name="alpha",
    )

def test_build_signal_returns_series(
    sample_signal,
):

    result = build_signal(
        sample_signal,
    )

    assert isinstance(
        result,
        pd.Series,
    )

def test_long_short_bounds(
    sample_signal,
):

    result = build_signal(
        sample_signal,
        direction="long_short",
    )

    assert (result <= 1).all()

    assert (result >= -1).all()


def test_long_only_non_negative(
    sample_signal,
):

    result = build_signal(
        sample_signal,
        direction="long_only",
    )

    assert (result >= 0).all()

def test_long_only_upper_bound(
    sample_signal,
):

    result = build_signal(
        sample_signal,
        direction="long_only",
    )

    assert (result <= 1).all()

def test_short_only_non_negative(
    sample_signal,
):

    result = build_signal(
        sample_signal,
        direction="short_only",
    )

    assert (result >= 0).all()

def test_invalid_direction_raises(
    sample_signal,
):

    with pytest.raises(ValueError):

        build_signal(
            sample_signal,
            direction="invalid",
        )


def test_build_signal_removes_nan():

    signal = pd.Series(
        [1.0, np.nan, 2.0],
    )

    result = build_signal(
        signal,
    )

    assert not result.isna().any()

def test_invert_returns_series(
    sample_signal,
):

    result = invert_signal(
        sample_signal,
    )

    assert isinstance(
        result,
        pd.Series,
    )

def test_invert_negates_values(
    sample_signal,
):

    result = invert_signal(
        sample_signal,
    )

    pd.testing.assert_series_equal(
        result,
        -sample_signal,
    )

def test_double_invert_returns_original(
    sample_signal,
):

    result = invert_signal(
        invert_signal(
            sample_signal,
        )
    )

    pd.testing.assert_series_equal(
        result,
        sample_signal,
    )

def test_rescale_returns_series(
    sample_signal,
):

    result = rescale_signal(
        sample_signal,
    )

    assert isinstance(
        result,
        pd.Series,
    )

def test_rescale_bounds(
    sample_signal,
):

    result = rescale_signal(
        sample_signal,
    )

    assert result.min() >= -1

    assert result.max() <= 1

def test_custom_bounds(
    sample_signal,
):

    result = rescale_signal(
        sample_signal,
        lower=0,
        upper=100,
    )

    assert result.min() >= 0

    assert result.max() <= 100

def test_invalid_bounds_raise(
    sample_signal,
):

    with pytest.raises(ValueError):

        rescale_signal(
            sample_signal,
            lower=5,
            upper=1,
        )

def test_constant_signal_returns_zero():

    signal = pd.Series(
        [5.0, 5.0, 5.0],
    )

    result = rescale_signal(
        signal,
    )

    assert (result == 0).all()

def test_build_signal_does_not_modify_input(
    sample_signal,
):

    original = sample_signal.copy()

    build_signal(
        sample_signal,
    )

    pd.testing.assert_series_equal(
        original,
        sample_signal,
    )


def test_rescale_does_not_modify_input(
    sample_signal,
):

    original = sample_signal.copy()

    rescale_signal(
        sample_signal,
    )

    pd.testing.assert_series_equal(
        original,
        sample_signal,
    )

def test_repeatability(
    sample_signal,
):

    first = build_signal(
        sample_signal,
    )

    second = build_signal(
        sample_signal,
    )

    pd.testing.assert_series_equal(
        first,
        second,
    )

def test_empty_signal_raises():

    with pytest.raises(ValueError):

        build_signal(
            pd.Series(
                dtype=float,
            )
        )


def test_non_series_raises():

    with pytest.raises(TypeError):

        build_signal(
            [1, 2, 3],
        )

def test_positive_signal_long_only(
    positive_signal,
):

    result = build_signal(
        positive_signal,
        direction="long_only",
    )

    assert (result >= 0).all()

def test_positive_signal_short_only(
    positive_signal,
):

    result = build_signal(
        positive_signal,
        direction="short_only",
    )

    assert (result >= 0).all()

