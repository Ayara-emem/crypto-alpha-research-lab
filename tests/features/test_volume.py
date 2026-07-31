import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.features.volume import (
    rolling_average_volume,
    relative_volume,
    volume_momentum,
    volume_zscore,
)

def test_rolling_average_volume_shape(
    dataset_factory,
):

    dataset = dataset_factory(
        [100, 110, 120, 130, 140, 150]
    )

    dataset.prices["Volume"] = [
        1000,
        1100,
        1200,
        1300,
        1400,
        1500,
    ]

    result = rolling_average_volume(
        dataset,
        window=3,
    )

    assert isinstance(result, pd.Series)

    assert len(result) == len(dataset.prices)

    assert result.index.equals(
        dataset.prices.index
    )

@pytest.mark.parametrize(
    "window",
    [0, -1, -20],
)
def test_rolling_average_volume_invalid_window(
    dataset_factory,
    window,
):

    dataset = dataset_factory(
        [100, 101, 102]
    )

    dataset.prices["Volume"] = [
        1000,
        1100,
        1200,
    ]

    with pytest.raises(ValueError):
        rolling_average_volume(
            dataset,
            window=window,
        )

def test_rolling_average_volume_nan_behavior(
    dataset_factory,
):

    dataset = dataset_factory(
        [100, 101, 102, 103, 104]
    )

    dataset.prices["Volume"] = [
        100,
        200,
        300,
        400,
        500,
    ]

    result = rolling_average_volume(
        dataset,
        window=3,
    )

    assert result.iloc[:2].isna().all()

    assert result.iloc[2:].notna().all()

def test_rolling_average_volume_values(
    dataset_factory,
):

    dataset = dataset_factory(
        [100, 101, 102, 103, 104]
    )

    dataset.prices["Volume"] = [
        100,
        200,
        300,
        400,
        500,
    ]

    expected = (
        dataset.prices["Volume"]
        .rolling(3)
        .mean()
    )

    result = rolling_average_volume(
        dataset,
        window=3,
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )

def test_relative_volume_shape(
    dataset_factory,
):

    dataset = dataset_factory(
        np.linspace(
            100,
            120,
            30,
        )
    )

    dataset.prices["Volume"] = np.arange(
        1000,
        1030,
    )

    result = relative_volume(
        dataset,
        window=5,
    )

    assert isinstance(result, pd.Series)

    assert len(result) == 30

@pytest.mark.parametrize(
    "window",
    [0, -5],
)
def test_relative_volume_invalid_window(
    dataset_factory,
    window,
):

    dataset = dataset_factory(
        [100, 101, 102]
    )

    dataset.prices["Volume"] = [
        100,
        200,
        300,
    ]

    with pytest.raises(ValueError):
        relative_volume(
            dataset,
            window=window,
        )

def test_volume_momentum_shape(
    dataset_factory,
):

    dataset = dataset_factory(
        np.linspace(
            100,
            120,
            20,
        )
    )

    dataset.prices["Volume"] = np.arange(
        100,
        120,
    )

    result = volume_momentum(
        dataset,
        window=3,
    )

    assert isinstance(result, pd.Series)

    assert len(result) == 20

@pytest.mark.parametrize(
    "window",
    [0, -3],
)
def test_volume_momentum_invalid_window(
    dataset_factory,
    window,
):

    dataset = dataset_factory(
        [100, 101, 102]
    )

    dataset.prices["Volume"] = [
        100,
        200,
        300,
    ]

    with pytest.raises(ValueError):
        volume_momentum(
            dataset,
            window=window,
        )

def test_volume_zscore_shape(
    dataset_factory,
    ):

        dataset = dataset_factory(
        np.linspace(
            100,
            120,
            30,
        )
    )
        dataset.prices["Volume"] = np.arange(
        100,
        130,
    )
        result = volume_zscore(
        dataset,
        window=5,
    )
        assert isinstance(result, pd.Series)

        assert len(result) == 30

@pytest.mark.parametrize(
    "window",
    [0, -1],
)
def test_volume_zscore_invalid_window(
    dataset_factory,
    window,
):

    dataset = dataset_factory(
        [100, 101, 102]
    )

    dataset.prices["Volume"] = [
        100,
        200,
        300,
    ]

    with pytest.raises(ValueError):
        volume_zscore(
            dataset,
            window=window,
        )
def test_constant_volume_behavior(
    dataset_factory,
):

    dataset = dataset_factory(
        [100, 101, 102, 103, 104, 105]
    )

    dataset.prices["Volume"] = [
        1000,
        1000,
        1000,
        1000,
        1000,
        1000,
    ]

    result = relative_volume(
        dataset,
        window=3,
    )

    expected = pd.Series(
    [np.nan, np.nan, 1.0, 1.0, 1.0, 1.0],
    index=dataset.prices.index,
    name="Volume",
)

    pd.testing.assert_series_equal(
        result,
        expected,
    )


