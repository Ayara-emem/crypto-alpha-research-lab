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


def test_feature_matrix_returns_dataframe(
    dataset_factory,
):
    dataset = dataset_factory(
        np.linspace(100, 150, 100)
    )

    dataset.prices["Volume"] = np.linspace(
        1000,
        5000,
        100,
    )

    result = build_feature_matrix(
        dataset,
        window=20,
    )

    assert isinstance(result, pd.DataFrame)


def test_feature_matrix_preserves_length(
    dataset_factory,
):
    dataset = dataset_factory(
        np.linspace(100, 150, 100)
    )

    dataset.prices["Volume"] = np.linspace(
        1000,
        5000,
        100,
    )

    result = build_feature_matrix(
        dataset,
        window=20,
    )

    assert len(result) == len(dataset.prices)

def test_feature_matrix_preserves_index(
    dataset_factory,
):
    dataset = dataset_factory(
        np.linspace(100, 150, 100)
    )

    dataset.prices["Volume"] = np.linspace(
        1000,
        5000,
        100,
    )

    result = build_feature_matrix(
        dataset,
        window=20,
    )

    assert result.index.equals(
        dataset.prices.index
    )

def test_feature_matrix_columns(
    dataset_factory,
):
    dataset = dataset_factory(
        np.linspace(100, 150, 100)
    )

    dataset.prices["Volume"] = np.linspace(
        1000,
        5000,
        100,
    )

    result = build_feature_matrix(
        dataset,
        window=20,
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
        [100, 101, 102]
    )

    dataset.prices["Volume"] = [
        1000,
        1100,
        1200,
    ]

    with pytest.raises(ValueError):
        build_feature_matrix(
            dataset,
            window=window,
        )

def test_feature_matrix_preserves_nan_warmup(
    dataset_factory,
):
    dataset = dataset_factory(
        np.linspace(100, 150, 100)
    )

    dataset.prices["Volume"] = np.linspace(
        1000,
        5000,
        100,
    )

    result = build_feature_matrix(
        dataset,
        window=20,
    )

    assert result.iloc[0].isna().any()


def test_feature_matrix_price_momentum_consistency(
    dataset_factory,
):
    dataset = dataset_factory(
        np.linspace(100, 150, 100)
    )

    dataset.prices["Volume"] = np.linspace(
        1000,
        5000,
        100,
    )

    window = 20

    matrix = build_feature_matrix(
        dataset,
        window=window,
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
        np.linspace(100, 150, 100)
    )

    dataset.prices["Volume"] = np.linspace(
        1000,
        5000,
        100,
    )

    window = 20

    matrix = build_feature_matrix(
        dataset,
        window=window,
    )

    expected = {
        "price_momentum": price_momentum(
            dataset,
            window,
        ),
        "rolling_return": rolling_return(
            dataset,
            window,
        ),
        "log_momentum": log_momentum(
            dataset,
            window,
        ),
        "rolling_volatility": rolling_volatility(
            dataset,
            window,
        ),
        "realized_volatility": realized_volatility(
            dataset,
            window,
        ),
        "relative_volume": relative_volume(
            dataset,
            window,
        ),
        "volume_momentum": volume_momentum(
            dataset,
            window,
        ),
        "volume_zscore": volume_zscore(
            dataset,
            window,
        ),
    }

    for column, expected_series in expected.items():
        pd.testing.assert_series_equal(
            matrix[column],
            expected_series,
            check_names=False,
        )

