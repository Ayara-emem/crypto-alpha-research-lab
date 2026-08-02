"""
Tests for research target engineering.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.research.targets import (
    future_return,
)

from crypto_alpha_lab.dataset import (
    ResearchDataset,
)


@pytest.fixture
def sample_dataset():
    """
    Small deterministic dataset.
    """

    prices = pd.DataFrame(
        {
            "Close": [
                100.0,
                102.0,
                105.0,
                110.0,
                120.0,
            ]
        },
        index=pd.date_range(
            "2024-01-01",
            periods=5,
        ),
    )

    return ResearchDataset(
        prices=prices,
    )


def test_future_return_returns_series(
    sample_dataset,
):
    result = future_return(
        sample_dataset,
    )

    assert isinstance(
        result,
        pd.Series,
    )


def test_future_return_preserves_length(
    sample_dataset,
):
    result = future_return(
        sample_dataset,
    )

    assert len(result) == len(
        sample_dataset.prices
    )


def test_future_return_preserves_index(
    sample_dataset,
):
    result = future_return(
        sample_dataset,
    )

    assert result.index.equals(
        sample_dataset.prices.index
    )


@pytest.mark.parametrize(
    "horizon",
    [
        0,
        -1,
        -20,
    ],
)
def test_future_return_invalid_horizon(
    sample_dataset,
    horizon,
):
    with pytest.raises(ValueError):
        future_return(
            sample_dataset,
            horizon=horizon,
        )


def test_future_return_last_value_nan(
    sample_dataset,
):
    result = future_return(
        sample_dataset,
        horizon=1,
    )

    assert pd.isna(
        result.iloc[-1]
    )


def test_future_return_two_period_nan(
    sample_dataset,
):
    result = future_return(
        sample_dataset,
        horizon=2,
    )

    assert result.iloc[-2:].isna().all()


def test_future_return_values_horizon_one(
    sample_dataset,
):
    close = sample_dataset.prices["Close"]

    expected = (
        close.shift(-1)
        .div(close)
        .sub(1.0)
    )

    result = future_return(
        sample_dataset,
        horizon=1,
    )

    pd.testing.assert_series_equal(
        result,
        expected,
        check_names=False,
    )


def test_future_return_values_horizon_three(
    sample_dataset,
):
    close = sample_dataset.prices["Close"]

    expected = (
        close.shift(-3)
        .div(close)
        .sub(1.0)
    )

    result = future_return(
        sample_dataset,
        horizon=3,
    )

    pd.testing.assert_series_equal(
        result,
        expected,
        check_names=False,
    )


def test_future_return_constant_prices():
    """
    Constant prices should produce
    zero future returns.
    """

    prices = pd.DataFrame(
        {
            "Close": [
                100,
                100,
                100,
                100,
                100,
            ]
        }
    )

    dataset = ResearchDataset(
        prices=prices,
    )

    result = future_return(
        dataset,
    )

    assert (
        result.dropna() == 0
    ).all()


def test_future_return_positive_prices():
    """
    Rising prices should produce
    non-negative future returns.
    """

    prices = pd.DataFrame(
        {
            "Close": [
                100,
                101,
                102,
                103,
                104,
            ]
        }
    )

    dataset = ResearchDataset(
        prices=prices,
    )

    result = future_return(
        dataset,
    )

    assert (
        result.dropna() >= 0
    ).all()