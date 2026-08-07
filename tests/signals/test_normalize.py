import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.signals.normalize import (
    normalize_signal,
)

@pytest.fixture
def sample_signal():

    return pd.Series(
        [1,2,3,4,5],
        dtype=float,
    )

def test_returns_series(sample_signal):

    result = normalize_signal(
        sample_signal,
    )

    assert isinstance(
        result,
        pd.Series,
    )

def test_zscore_mean_zero(
    sample_signal,
):

    result = normalize_signal(
        sample_signal,
        method="zscore",
    )

    assert np.isclose(
        result.mean(),
        0,
    )

def test_rank_between_zero_one(
    sample_signal,
):

    result = normalize_signal(
        sample_signal,
        method="rank",
    )

    assert (
        (result >= 0)
        &
        (result <= 1)
    ).all()

def test_minmax_between_zero_one(
    sample_signal,
):

    result = normalize_signal(
        sample_signal,
        method="minmax",
    )

    assert (
        (result >= 0)
        &
        (result <= 1)
    ).all()
def test_invalid_method():

    with pytest.raises(ValueError):

        normalize_signal(
            pd.Series([1,2]),
            method="bad",
        )

def test_constant_signal():

    signal = pd.Series(
        [5,5,5],
    )

    result = normalize_signal(
        signal,
    )

    assert (
        result == 0
    ).all()

def test_input_not_modified(
    sample_signal,
):

    original = sample_signal.copy()

    normalize_signal(
        sample_signal,
    )

    pd.testing.assert_series_equal(
        original,
        sample_signal,
    )


def test_repeatability(
    sample_signal,
):

    first = normalize_signal(
        sample_signal,
    )

    second = normalize_signal(
        sample_signal,
    )

    pd.testing.assert_series_equal(
        first,
        second,
    )
