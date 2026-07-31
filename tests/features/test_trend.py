import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.features.trend import (
    simple_moving_average,
    exponential_moving_average,
    price_to_moving_average,
    moving_average_spread,
    price_to_moving_average,
    moving_average_spread,
)

def test_simple_moving_average_shape(
    dataset_factory,
):
    dataset = dataset_factory(
        [100, 101, 102, 103, 104, 105]
    )

    result = simple_moving_average(
        dataset,
        window=3,
    )

    assert isinstance(result, pd.Series)
    assert len(result) == len(dataset.prices)
    assert result.index.equals(dataset.prices.index)


@pytest.mark.parametrize(
    "window",
    [0, -1, -20],
)
def test_simple_moving_average_invalid_window(
    dataset_factory,
    window,
):
    dataset = dataset_factory(
        [100, 101, 102]
    )

    with pytest.raises(ValueError):
        simple_moving_average(
            dataset,
            window=window,
        )

def test_simple_moving_average_nan_behavior(
    dataset_factory,
):
    dataset = dataset_factory(
        [100, 101, 102, 103, 104]
    )

    result = simple_moving_average(
        dataset,
        window=3,
    )

    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].notna().all()

def test_simple_moving_average_values(
    dataset_factory,
):
    dataset = dataset_factory(
        [100, 102, 104, 106, 108]
    )

    result = simple_moving_average(
        dataset,
        window=3,
    )

    expected = pd.Series(
        [
            np.nan,
            np.nan,
            102.0,
            104.0,
            106.0,
        ],
        index=dataset.prices.index,
        name="Close",
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )

def test_exponential_moving_average_shape(
    dataset_factory,
):
    dataset = dataset_factory(
        [100, 101, 102, 103, 104, 105]
    )

    result = exponential_moving_average(
        dataset,
        span=3,
    )

    assert isinstance(result, pd.Series)
    assert len(result) == len(dataset.prices)
    assert result.index.equals(dataset.prices.index)

@pytest.mark.parametrize(
    "span",
    [0, -1, -20],
)
def test_exponential_moving_average_invalid_span(
    dataset_factory,
    span,
):
    dataset = dataset_factory(
        [100, 101, 102]
    )

    with pytest.raises(ValueError):
        exponential_moving_average(
            dataset,
            span=span,
        )

def test_exponential_moving_average_values(
    dataset_factory,
):
    dataset = dataset_factory(
        [100, 102, 104, 106, 108]
    )

    close = dataset.prices["Close"]

    expected = close.ewm(
        span=3,
        adjust=False,
    ).mean()

    result = exponential_moving_average(
        dataset,
        span=3,
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )

def test_price_to_moving_average_shape(
    dataset_factory,
):
    dataset = dataset_factory(
        np.linspace(
            100,
            150,
            30,
        )
    )

    result = price_to_moving_average(
        dataset,
        window=5,
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 30
    assert result.index.equals(dataset.prices.index)

@pytest.mark.parametrize(
    "window",
    [0, -1],
)
def test_price_to_moving_average_invalid_window(
    dataset_factory,
    window,
):
    dataset = dataset_factory(
        [100, 101, 102]
    )

    with pytest.raises(ValueError):
        price_to_moving_average(
            dataset,
            window=window,
        )


def test_price_to_moving_average_values(
    dataset_factory,
):
    dataset = dataset_factory(
        [100, 102, 104, 106, 108]
    )

    close = dataset.prices["Close"]

    moving_average = close.rolling(3).mean()

    expected = (
        close
        .div(moving_average)
        .sub(1.0)
    )

    result = price_to_moving_average(
        dataset,
        window=3,
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )

def test_moving_average_spread_shape(
    dataset_factory,
):
    dataset = dataset_factory(
        np.linspace(
            100,
            200,
            100,
        )
    )

    result = moving_average_spread(
        dataset,
        short_window=10,
        long_window=30,
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 100
    assert result.index.equals(dataset.prices.index)

def test_moving_average_spread_invalid_short_window(
    dataset_factory,
):
    dataset = dataset_factory(
        np.linspace(100, 200, 100)
    )

    with pytest.raises(ValueError):
        moving_average_spread(
            dataset,
            short_window=0,
            long_window=30,
        )

def test_moving_average_spread_invalid_long_window(
    dataset_factory,
):
    dataset = dataset_factory(
        np.linspace(100, 200, 100)
    )

    with pytest.raises(ValueError):
        moving_average_spread(
            dataset,
            short_window=10,
            long_window=0,
        )

@pytest.mark.parametrize(
    ("short_window", "long_window"),
    [
        (30, 30),
        (40, 30),
    ],
)
def test_moving_average_spread_window_order(
    dataset_factory,
    short_window,
    long_window,
):
    dataset = dataset_factory(
        np.linspace(100, 200, 100)
    )

    with pytest.raises(ValueError):
        moving_average_spread(
            dataset,
            short_window=short_window,
            long_window=long_window,
        )

def test_moving_average_spread_values(
    dataset_factory,
):
    dataset = dataset_factory(
        np.linspace(
            100,
            150,
            50,
        )
    )

    close = dataset.prices["Close"]

    short_ma = close.rolling(5).mean()
    long_ma = close.rolling(10).mean()

    expected = (
        short_ma
        .div(long_ma)
        .sub(1.0)
    )

    result = moving_average_spread(
        dataset,
        short_window=5,
        long_window=10,
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_price_to_moving_average_constant_prices(
    dataset_factory,
):
    dataset = dataset_factory(
        [100, 100, 100, 100, 100, 100]
    )

    result = price_to_moving_average(
        dataset,
        window=3,
    )

    assert (
        result.dropna() == 0.0
    ).all()

def test_moving_average_spread_constant_prices(
    dataset_factory,
):
    dataset = dataset_factory(
        [100] * 20
    )

    result = moving_average_spread(
        dataset,
        short_window=3,
        long_window=5,
    )

    assert (
        result.dropna() == 0.0
    ).all()


