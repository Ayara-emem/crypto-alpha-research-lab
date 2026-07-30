import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.features.volatility import (
    rolling_volatility,
    realized_volatility,
    volatility_ratio,
    volatility_zscore,
)

def test_rolling_volatility_shape(dataset_factory):

    dataset = dataset_factory(
        [100, 101, 102, 103, 104, 105]
    )

    result = rolling_volatility(
        dataset,
        window=3,
    )

    assert isinstance(result, pd.Series)

    assert len(result) == 6

    assert result.index.equals(
        dataset.prices.index
    )

@pytest.mark.parametrize(
    "window",
    [0, -1, -20],
)
def test_rolling_volatility_invalid_window(
    dataset_factory,
    window,
):

    dataset = dataset_factory(
        [100, 101, 102]
    )

    with pytest.raises(ValueError):
        rolling_volatility(
            dataset,
            window=window,
        )


def test_rolling_volatility_nan_behavior(
    dataset_factory,
):

    dataset = dataset_factory(
        [100, 101, 102, 103, 104]
    )

    result = rolling_volatility(
        dataset,
        window=3,
    )

    assert result.iloc[:3].isna().all()

    assert result.iloc[3:].notna().all()

def test_rolling_volatility_values(
    dataset_factory,
):

    dataset = dataset_factory(
        [100, 101, 103, 102, 105, 108]
    )

    close = dataset.prices["Close"]

    expected = (
        close
        .pct_change()
        .rolling(3)
        .std()
    )

    result = rolling_volatility(
        dataset,
        window=3,
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )

def test_realized_volatility_shape(
    dataset_factory,
):

    dataset = dataset_factory(
        [100, 101, 102, 103, 104]
    )

    result = realized_volatility(
        dataset,
        window=3,
    )

    assert isinstance(result, pd.Series)

    assert len(result) == 5


@pytest.mark.parametrize(
    "window",
    [0, -1],
)
def test_realized_volatility_invalid_window(
    dataset_factory,
    window,
):

    dataset = dataset_factory(
        [100, 101, 102]
    )

    with pytest.raises(ValueError):
        realized_volatility(
            dataset,
            window=window,
        )


def test_volatility_ratio_shape(
    dataset_factory,
):

    dataset = dataset_factory(
        np.linspace(
            100,
            150,
            100,
        )
    )

    result = volatility_ratio(
        dataset,
        short_window=5,
        long_window=20,
    )

    assert isinstance(result, pd.Series)

    assert len(result) == 100

def test_volatility_ratio_invalid_windows(
    dataset_factory,
):

    dataset = dataset_factory(
        np.linspace(
            100,
            120,
            50,
        )
    )

    with pytest.raises(ValueError):
        volatility_ratio(
            dataset,
            short_window=20,
            long_window=10,
        )

def test_volatility_zscore_shape(
    dataset_factory,
):

    dataset = dataset_factory(
        np.linspace(
            100,
            150,
            100,
        )
    )

    result = volatility_zscore(
        dataset,
        volatility_window=5,
        zscore_window=20,
    )

    assert isinstance(result, pd.Series)

    assert len(result) == 100

def test_volatility_zscore_invalid_window(
    dataset_factory,
):

    dataset = dataset_factory(
        np.linspace(
            100,
            150,
            100,
        )
    )

    with pytest.raises(ValueError):
        volatility_zscore(
            dataset,
            volatility_window=0,
            zscore_window=20,
        )

    with pytest.raises(ValueError):
        volatility_zscore(
            dataset,
            volatility_window=5,
            zscore_window=0,
        )

def test_rolling_volatility_constant_prices(
    dataset_factory,
):
    dataset = dataset_factory([100, 100, 100, 100, 100, 100])

    result = rolling_volatility(dataset, window=3)

    assert (result.dropna() == 0).all()