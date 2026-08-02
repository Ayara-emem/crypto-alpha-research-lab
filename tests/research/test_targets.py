"""
Tests for research target engineering.
"""
from __future__ import annotations

from asset_pricing_lab.returns import log_ratio
import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.research.targets import (
    future_return,
    future_log_return,
    future_direction,
    future_volatility,
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

def test_future_log_return_returns_series(
    sample_dataset,
):
    result = future_log_return(
        sample_dataset,
    )

    assert isinstance(
        result,
        pd.Series,
    )

def test_future_log_return_preserves_length(
    sample_dataset,
):
    result = future_log_return(
        sample_dataset,
    )

    assert len(result) == len(
        sample_dataset.prices
    )

def test_future_log_return_preserves_index(
    sample_dataset,
):
    result = future_log_return(
        sample_dataset,
    )

    assert result.index.equals(
        sample_dataset.prices.index
    )

@pytest.mark.parametrize(
    "horizon",
    [0, -1, -5],
)
def test_future_log_return_invalid_horizon(
    sample_dataset,
    horizon,
):
    with pytest.raises(ValueError):
        future_log_return(
            sample_dataset,
            horizon=horizon,
        )

def test_future_log_return_values(
    sample_dataset,
):
    close = sample_dataset.prices["Close"]
    expected = log_ratio(
        close.shift(-1),
        close,
    )

    result = future_log_return(
        sample_dataset,
        horizon=1,
    )

    pd.testing.assert_series_equal(
        result,
        expected,
        check_names=False,
    )


def test_future_log_return_constant_prices():
    prices = pd.DataFrame(
        {
            "Close": [
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

    result = future_log_return(
        dataset,
    )

    assert np.allclose(
        result.dropna(),
        0,
    )

def test_future_direction_returns_series(
    sample_dataset,
):
    result = future_direction(
        sample_dataset,
    )

    assert isinstance(
        result,
        pd.Series,
    )

def test_future_direction_preserves_length(
    sample_dataset,
):
    result = future_direction(
        sample_dataset,
    )

    assert len(result) == len(
        sample_dataset.prices
    )


def test_future_direction_preserves_index(
    sample_dataset,
):
    result = future_direction(
        sample_dataset,
    )

    assert result.index.equals(
        sample_dataset.prices.index
    )

@pytest.mark.parametrize(
    "horizon",
    [0, -1, -5],
)
def test_future_direction_invalid_horizon(
    sample_dataset,
    horizon,
):
    with pytest.raises(ValueError):
        future_direction(
            sample_dataset,
            horizon=horizon,
        )

def test_future_direction_rising_prices():
    prices = pd.DataFrame(
        {
            "Close": [
                100,
                101,
                102,
                103,
            ]
        }
    )

    dataset = ResearchDataset(
        prices=prices,
    )

    result = future_direction(
        dataset,
    )

    assert (result.dropna() == 1).all()

def test_future_direction_falling_prices():
    prices = pd.DataFrame(
        {
            "Close": [
                103,
                102,
                101,
                100,
            ]
        }
    )

    dataset = ResearchDataset(
        prices=prices,
    )

    result = future_direction(
        dataset,
    )

    assert (result.dropna() == 0).all()

def test_future_direction_last_value_nan(
    sample_dataset,
):
    result = future_direction(
        sample_dataset,
    )

    assert pd.isna(
        result.iloc[-1]
    )

def test_future_volatility_returns_series(
    sample_dataset,
):
    result = future_volatility(
        sample_dataset,
        horizon=2,
    )

    assert isinstance(
        result,
        pd.Series,
    )

def test_future_volatility_preserves_length(
    sample_dataset,
):
    result = future_volatility(
        sample_dataset,
        horizon=2,
    )

    assert len(result) == len(
        sample_dataset.prices
    )


def test_future_volatility_preserves_index(
    sample_dataset,
):
    result = future_volatility(
        sample_dataset,
        horizon=2,
    )

    assert result.index.equals(
        sample_dataset.prices.index
    )


@pytest.mark.parametrize(
    "horizon",
    [
        0,
        1,
        -5,
    ],
)
def test_future_volatility_invalid_horizon(
    sample_dataset,
    horizon,
):
    with pytest.raises(ValueError):
        future_volatility(
            sample_dataset,
            horizon=horizon,
        )

def test_future_volatility_constant_prices():
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

    result = future_volatility(
        dataset,
        horizon=2,
    )

    assert np.allclose(
        result.dropna(),
        0,
    )
def test_future_volatility_trailing_nan(
    sample_dataset,
):
    result = future_volatility(
        sample_dataset,
        horizon=2,
    )

    assert result.iloc[-2:].isna().all()


