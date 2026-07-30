import numpy as np
import pandas as pd
import pytest


from crypto_alpha_lab.features.momentum import (
    price_momentum,
    rolling_return,
    log_momentum,
    relative_momentum,
)


def test_price_momentum_shape(sample_dataset):

    result = price_momentum(
        sample_dataset,
        window=3,
    )

    assert len(result) == len(sample_dataset.prices)


@pytest.mark.parametrize(
    "window",
    [0, -1, -20],
)
def test_price_momentum_invalid_window(
    sample_dataset,
    window,
):
    with pytest.raises(ValueError):
        price_momentum(
            sample_dataset,
            window=window,
        )

def test_log_momentum_shape(sample_dataset):

    result = log_momentum(
        sample_dataset,
        window=3,
    )

    assert len(result) == len(sample_dataset.prices)

def test_relative_momentum_zero_when_equal(
    dataset_factory,
):
    dataset = dataset_factory(
        [100, 101, 102, 103, 104]
    )

    benchmark = dataset_factory(
        [100, 101, 102, 103, 104]
    )

    result = relative_momentum(
        dataset,
        benchmark,
        window=3,
    )

    assert (result.dropna() == 0).all()

def test_price_momentum_nan_behavior(sample_dataset):

    window = 4

    result = price_momentum(
        sample_dataset,
        window=window,
    )

    assert result.iloc[:window].isna().all()

def test_price_momentum_values(sample_dataset):

    result = price_momentum(
        sample_dataset,
        window=2,
    )

    expected = (
        sample_dataset.prices["Close"]
        / sample_dataset.prices["Close"].shift(2)
        - 1
    )

    pd.testing.assert_series_equal(
        result,
        expected,
        check_names=False,
    )
