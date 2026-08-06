"""
Tests for research dataset construction.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.dataset import (
    ResearchDataset,
)

from crypto_alpha_lab.research.dataset import (
    build_research_dataset,
)


@pytest.fixture
def sample_dataset():
    """
    Deterministic dataset for research dataset tests.
    """

    prices = pd.DataFrame(
        {
            "Close": np.linspace(
                100,
                200,
                120,
            ),
            "Volume": np.linspace(
                1000,
                5000,
                120,
            ),
        },
        index=pd.date_range(
            "2024-01-01",
            periods=120,
        ),
    )

    return ResearchDataset(
        prices=prices,
    )


# --------------------------------------------------------
# Basic API
# --------------------------------------------------------

def test_build_research_dataset_returns_dataframe(
    sample_dataset,
):
    result = build_research_dataset(
        sample_dataset,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_build_research_dataset_not_empty(
    sample_dataset,
):
    result = build_research_dataset(
        sample_dataset,
    )

    assert not result.empty


def test_build_research_dataset_unique_index(
    sample_dataset,
):
    result = build_research_dataset(
        sample_dataset,
    )

    assert result.index.is_unique


def test_build_research_dataset_sorted_index(
    sample_dataset,
):
    result = build_research_dataset(
        sample_dataset,
    )

    assert result.index.is_monotonic_increasing


# --------------------------------------------------------
# Target selection
# --------------------------------------------------------

@pytest.mark.parametrize(
    "target",
    [
        "future_return",
        "future_log_return",
        "future_direction",
        "future_volatility",
    ],
)
def test_all_targets_supported(
    sample_dataset,
    target,
):
    result = build_research_dataset(
        sample_dataset,
        target=target,
    )

    assert target in result.columns


def test_invalid_target():
    prices = pd.DataFrame(
        {
            "Close": np.arange(
                100,
                220,
            ),
            "Volume": np.arange(
                1000,
                1120,
            ),
        }
    )

    dataset = ResearchDataset(
        prices=prices,
    )

    with pytest.raises(ValueError):
        build_research_dataset(
            dataset,
            target="invalid_target",
        )


# --------------------------------------------------------
# Missing values
# --------------------------------------------------------

def test_drop_missing_true(
    sample_dataset,
):
    result = build_research_dataset(
        sample_dataset,
        drop_missing=True,
    )

    assert not result.isna().any().any()


def test_drop_missing_false(
    sample_dataset,
):
    result = build_research_dataset(
        sample_dataset,
        drop_missing=False,
    )

    assert result.isna().any().any()


# --------------------------------------------------------
# Alignment
# --------------------------------------------------------

def test_target_alignment(
    sample_dataset,
):
    result = build_research_dataset(
        sample_dataset,
        target="future_return",
        drop_missing=False,
    )

    assert result.index.equals(
        sample_dataset.prices.index
    )


def test_feature_columns_exist(
    sample_dataset,
):
    result = build_research_dataset(
        sample_dataset,
    )

    expected = [
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

    for column in expected:
        assert column in result.columns


# --------------------------------------------------------
# Parameter propagation
# --------------------------------------------------------

def test_target_horizon_changes_values(
    sample_dataset,
):
    one_day = build_research_dataset(
        sample_dataset,
        target_horizon=1,
    )

    five_day = build_research_dataset(
        sample_dataset,
        target_horizon=5,
    )

    assert not one_day.equals(
        five_day,
    )


def test_feature_window_changes_values(
    sample_dataset,
):
    short = build_research_dataset(
        sample_dataset,
        feature_window=10,
    )

    long = build_research_dataset(
        sample_dataset,
        feature_window=30,
    )

    assert not short.equals(
        long,
    )


# --------------------------------------------------------
# Research integrity
# --------------------------------------------------------

def test_no_duplicate_columns(
    sample_dataset,
):
    result = build_research_dataset(
        sample_dataset,
    )

    assert result.columns.is_unique


def test_contains_target_column(
    sample_dataset,
):
    result = build_research_dataset(
        sample_dataset,
    )

    assert "future_return" in result.columns


def test_contains_more_than_target(
    sample_dataset,
):
    result = build_research_dataset(
        sample_dataset,
    )

    assert result.shape[1] > 1


def test_drop_missing_reduces_rows(
    sample_dataset,
):
    complete = build_research_dataset(
        sample_dataset,
        drop_missing=False,
    )

    filtered = build_research_dataset(
        sample_dataset,
        drop_missing=True,
    )

    assert len(filtered) <= len(
        complete,
    )


def test_future_direction_binary(
    sample_dataset,
):
    result = build_research_dataset(
        sample_dataset,
        target="future_direction",
    )

    values = set(
        result["future_direction"]
    )

    assert values.issubset(
        {0.0, 1.0}
    )


def test_future_volatility_non_negative(
    sample_dataset,
):
    result = build_research_dataset(
        sample_dataset,
        target="future_volatility",
        volatility_window=20,
    )

    assert (
        result["future_volatility"]
        >= 0
    ).all()