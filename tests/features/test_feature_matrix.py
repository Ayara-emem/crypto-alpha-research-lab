"""
Tests for CARL feature matrix construction.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.research.feature_matrix import (
    build_feature_matrix,
)

from crypto_alpha_lab.features.momentum import (
    price_momentum,
    rolling_return,
    log_momentum,
)

from crypto_alpha_lab.features.volatility import (
    rolling_volatility,
    realized_volatility,
)

from crypto_alpha_lab.features.volume import (
    relative_volume,
    volume_momentum,
    volume_zscore,
)

from crypto_alpha_lab.features.trend import (
    price_to_moving_average,
    moving_average_spread,
)


def test_feature_matrix_returns_dataframe(
    dataset_factory,
):
    dataset = dataset_factory(
        close_prices=np.linspace(
            100,
            150,
            100,
        ),
        volume=np.linspace(
            1000,
            5000,
            100,
        ),
    )

    result = build_feature_matrix(
        dataset,
        window=20,
        trend_long_window=60,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_feature_matrix_preserves_length(
    dataset_factory,
):
    dataset = dataset_factory(
        close_prices=np.linspace(
            100,
            150,
            100,
        ),
        volume=np.linspace(
            1000,
            5000,
            100,
        ),
    )

    result = build_feature_matrix(
        dataset,
        window=20,
        trend_long_window=60,
    )

    assert len(result) == len(
        dataset.prices
    )


def test_feature_matrix_preserves_index(
    dataset_factory,
):
    dataset = dataset_factory(
        close_prices=np.linspace(
            100,
            150,
            100,
        ),
        volume=np.linspace(
            1000,
            5000,
            100,
        ),
    )

    result = build_feature_matrix(
        dataset,
        window=20,
        trend_long_window=60,
    )

    assert result.index.equals(
        dataset.prices.index
    )


def test_feature_matrix_columns(
    dataset_factory,
):
    dataset = dataset_factory(
        close_prices=np.linspace(
            100,
            150,
            100,
        ),
        volume=np.linspace(
            1000,
            5000,
            100,
        ),
    )

    result = build_feature_matrix(
        dataset,
        window=20,
        trend_long_window=60,
    )

    expected_columns = [
        "price_momentum",
        "rolling_return",
        "log_momentum",
        "rolling_volatility",
        "realized_volatility",
        "relative_volume",
        "volume_momentum",
        "volume_zscore",
        "price_to_moving_average",
        "moving_average_spread",
    ]

    assert list(result.columns) == expected_columns


@pytest.mark.parametrize(
    "window",
    [0, -1, -20],
)
def test_feature_matrix_invalid_window(
    dataset_factory,
    window,
):
    dataset = dataset_factory(
        close_prices=np.linspace(
            100,
            150,
            100,
        ),
        volume=np.linspace(
            1000,
            5000,
            100,
        ),
    )

    with pytest.raises(ValueError):
        build_feature_matrix(
            dataset,
            window=window,
            trend_long_window=60,
        )


@pytest.mark.parametrize(
    "trend_long_window",
    [0, -1, -60],
)
def test_feature_matrix_invalid_trend_long_window(
    dataset_factory,
    trend_long_window,
):
    dataset = dataset_factory(
        close_prices=np.linspace(
            100,
            150,
            100,
        ),
        volume=np.linspace(
            1000,
            5000,
            100,
        ),
    )

    with pytest.raises(ValueError):
        build_feature_matrix(
            dataset,
            window=20,
            trend_long_window=trend_long_window,
        )


@pytest.mark.parametrize(
    ("window", "trend_long_window"),
    [
        (20, 20),
        (60, 20),
    ],
)
def test_feature_matrix_invalid_trend_window_order(
    dataset_factory,
    window,
    trend_long_window,
):
    dataset = dataset_factory(
        close_prices=np.linspace(
            100,
            150,
            100,
        ),
        volume=np.linspace(
            1000,
            5000,
            100,
        ),
    )

    with pytest.raises(ValueError):
        build_feature_matrix(
            dataset,
            window=window,
            trend_long_window=trend_long_window,
        )


def test_feature_matrix_preserves_nan_warmup(
    dataset_factory,
):
    dataset = dataset_factory(
        close_prices=np.linspace(
            100,
            150,
            100,
        ),
        volume=np.linspace(
            1000,
            5000,
            100,
        ),
    )

    result = build_feature_matrix(
        dataset,
        window=20,
        trend_long_window=60,
    )

    assert result.iloc[0].isna().any()


def test_feature_matrix_preserves_long_trend_warmup(
    dataset_factory,
):
    dataset = dataset_factory(
        close_prices=np.linspace(
            100,
            200,
            100,
        ),
        volume=np.linspace(
            1000,
            5000,
            100,
        ),
    )

    result = build_feature_matrix(
        dataset,
        window=20,
        trend_long_window=60,
    )

    spread = result[
        "moving_average_spread"
    ]

    assert spread.iloc[:59].isna().all()

    assert spread.iloc[59:].notna().all()


def test_feature_matrix_price_momentum_consistency(
    dataset_factory,
):
    dataset = dataset_factory(
        close_prices=np.linspace(
            100,
            150,
            100,
        ),
        volume=np.linspace(
            1000,
            5000,
            100,
        ),
    )

    window = 20

    matrix = build_feature_matrix(
        dataset,
        window=window,
        trend_long_window=60,
    )

    expected = price_momentum(
        dataset,
        window=window,
    )

    pd.testing.assert_series_equal(
        matrix["price_momentum"],
        expected,
        check_names=False,
    )


def test_feature_matrix_matches_feature_functions(
    dataset_factory,
):
    dataset = dataset_factory(
        close_prices=np.linspace(
            100,
            150,
            100,
        ),
        volume=np.linspace(
            1000,
            5000,
            100,
        ),
    )

    window = 20
    trend_long_window = 60

    matrix = build_feature_matrix(
        dataset,
        window=window,
        trend_long_window=trend_long_window,
    )

    expected = {
        "price_momentum": price_momentum(
            dataset,
            window=window,
        ),
        "rolling_return": rolling_return(
            dataset,
            window=window,
        ),
        "log_momentum": log_momentum(
            dataset,
            window=window,
        ),
        "rolling_volatility": rolling_volatility(
            dataset,
            window=window,
        ),
        "realized_volatility": realized_volatility(
            dataset,
            window=window,
        ),
        "relative_volume": relative_volume(
            dataset,
            window=window,
        ),
        "volume_momentum": volume_momentum(
            dataset,
            window=window,
        ),
        "volume_zscore": volume_zscore(
            dataset,
            window=window,
        ),
        "price_to_moving_average": (
            price_to_moving_average(
                dataset,
                window=window,
            )
        ),
        "moving_average_spread": (
            moving_average_spread(
                dataset,
                short_window=window,
                long_window=trend_long_window,
            )
        ),
    }

    for column, expected_series in expected.items():
        pd.testing.assert_series_equal(
            matrix[column],
            expected_series,
            check_names=False,
        )