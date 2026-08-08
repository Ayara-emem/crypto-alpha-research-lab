"""
Tests for CARL portfolio validation APIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.portfolio.validation import (
    validate_asset_alignment,
    validate_gross_exposure,
    validate_net_exposure,
    validate_portfolio,
    validate_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def weights() -> pd.Series:
    return pd.Series(
        {
            "BTC": 0.5,
            "ETH": 0.3,
            "SOL": 0.2,
        },
        dtype=float,
    )


@pytest.fixture
def assets() -> list[str]:
    return [
        "BTC",
        "ETH",
        "SOL",
    ]


# ---------------------------------------------------------------------------
# validate_weights
# ---------------------------------------------------------------------------


def test_validate_weights_accepts_valid_weights(
    weights,
):
    validate_weights(
        weights,
    )


def test_validate_weights_rejects_non_series():
    with pytest.raises(TypeError):

        validate_weights(
            {
                "BTC": 1.0,
            },
        )


def test_validate_weights_rejects_empty_series():
    with pytest.raises(ValueError):

        validate_weights(
            pd.Series(
                dtype=float,
            ),
        )


def test_validate_weights_rejects_duplicate_assets():
    weights = pd.Series(
        [0.5, 0.5],
        index=[
            "BTC",
            "BTC",
        ],
    )

    with pytest.raises(ValueError):

        validate_weights(
            weights,
        )


def test_validate_weights_rejects_nan():
    weights = pd.Series(
        {
            "BTC": 0.5,
            "ETH": np.nan,
        }
    )

    with pytest.raises(ValueError):

        validate_weights(
            weights,
        )


def test_validate_weights_rejects_infinite_values():
    weights = pd.Series(
        {
            "BTC": 0.5,
            "ETH": np.inf,
        }
    )

    with pytest.raises(ValueError):

        validate_weights(
            weights,
        )


# ---------------------------------------------------------------------------
# validate_asset_alignment
# ---------------------------------------------------------------------------


def test_validate_asset_alignment_accepts_matching_assets(
    weights,
    assets,
):
    validate_asset_alignment(
        weights,
        assets,
    )


def test_validate_asset_alignment_rejects_missing_asset(
    assets,
):
    weights = pd.Series(
        {
            "BTC": 0.5,
            "ETH": 0.5,
        }
    )

    with pytest.raises(ValueError):

        validate_asset_alignment(
            weights,
            assets,
        )


def test_validate_asset_alignment_rejects_extra_asset(
    assets,
):
    weights = pd.Series(
        {
            "BTC": 0.4,
            "ETH": 0.3,
            "SOL": 0.2,
            "ADA": 0.1,
        }
    )

    with pytest.raises(ValueError):

        validate_asset_alignment(
            weights,
            assets,
        )


# ---------------------------------------------------------------------------
# validate_gross_exposure
# ---------------------------------------------------------------------------


def test_validate_gross_exposure_accepts_valid_exposure():
    weights = pd.Series(
        {
            "BTC": 0.5,
            "ETH": 0.5,
        }
    )

    validate_gross_exposure(
        weights,
        max_gross_exposure=1.0,
    )


def test_validate_gross_exposure_rejects_excess_exposure():
    weights = pd.Series(
        {
            "BTC": 0.8,
            "ETH": 0.8,
        }
    )

    with pytest.raises(ValueError):

        validate_gross_exposure(
            weights,
            max_gross_exposure=1.0,
        )


def test_validate_gross_exposure_accepts_long_short_at_limit():
    weights = pd.Series(
        {
            "BTC": 0.5,
            "ETH": -0.5,
        }
    )

    validate_gross_exposure(
        weights,
        max_gross_exposure=1.0,
    )


def test_validate_gross_exposure_rejects_negative_limit(
    weights,
):
    with pytest.raises(ValueError):

        validate_gross_exposure(
            weights,
            max_gross_exposure=-1.0,
        )


# ---------------------------------------------------------------------------
# validate_net_exposure
# ---------------------------------------------------------------------------


def test_validate_net_exposure_accepts_valid_range(
    weights,
):
    validate_net_exposure(
        weights,
        min_net_exposure=0.0,
        max_net_exposure=1.0,
    )


def test_validate_net_exposure_rejects_below_minimum():
    weights = pd.Series(
        {
            "BTC": -0.5,
            "ETH": 0.2,
        }
    )

    with pytest.raises(ValueError):

        validate_net_exposure(
            weights,
            min_net_exposure=0.0,
        )


def test_validate_net_exposure_rejects_above_maximum():
    weights = pd.Series(
        {
            "BTC": 0.8,
            "ETH": 0.5,
        }
    )

    with pytest.raises(ValueError):

        validate_net_exposure(
            weights,
            max_net_exposure=1.0,
        )


def test_validate_net_exposure_rejects_invalid_range(
    weights,
):
    with pytest.raises(ValueError):

        validate_net_exposure(
            weights,
            min_net_exposure=1.0,
            max_net_exposure=0.5,
        )


def test_validate_net_exposure_accepts_market_neutral_portfolio():
    weights = pd.Series(
        {
            "BTC": 0.5,
            "ETH": -0.5,
        }
    )

    validate_net_exposure(
        weights,
        min_net_exposure=0.0,
        max_net_exposure=0.0,
    )


# ---------------------------------------------------------------------------
# validate_portfolio
# ---------------------------------------------------------------------------


def test_validate_portfolio_accepts_valid_portfolio(
    weights,
    assets,
):
    validate_portfolio(
        weights,
        assets=assets,
        max_gross_exposure=1.0,
        min_net_exposure=0.0,
        max_net_exposure=1.0,
    )


def test_validate_portfolio_validates_assets(
    weights,
):
    with pytest.raises(ValueError):

        validate_portfolio(
            weights,
            assets=[
                "BTC",
                "ETH",
                "SOL",
                "ADA",
            ],
        )


def test_validate_portfolio_validates_gross_exposure():
    weights = pd.Series(
        {
            "BTC": 0.8,
            "ETH": 0.8,
        }
    )

    with pytest.raises(ValueError):

        validate_portfolio(
            weights,
            max_gross_exposure=1.0,
        )


def test_validate_portfolio_validates_net_exposure():
    weights = pd.Series(
        {
            "BTC": 0.8,
            "ETH": 0.5,
        }
    )

    with pytest.raises(ValueError):

        validate_portfolio(
            weights,
            max_net_exposure=1.0,
        )


def test_validate_portfolio_accepts_long_short_portfolio():
    weights = pd.Series(
        {
            "BTC": 0.5,
            "ETH": -0.5,
        }
    )

    validate_portfolio(
        weights,
        max_gross_exposure=1.0,
        min_net_exposure=0.0,
        max_net_exposure=0.0,
    )

