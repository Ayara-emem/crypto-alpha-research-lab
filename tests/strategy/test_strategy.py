"""
Tests for CARL research strategy orchestration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.strategy.strategy import (
    ResearchStrategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def signals() -> pd.Series:
    """
    Asset-level research signals.
    """

    return pd.Series(
        {
            "BTC": 0.6,
            "ETH": 0.3,
            "SOL": 0.1,
        },
        dtype=float,
    )


@pytest.fixture
def covariance() -> pd.DataFrame:
    """
    Positive-definite covariance matrix.
    """

    return pd.DataFrame(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.01],
            [0.00, 0.01, 0.16],
        ],
        index=[
            "BTC",
            "ETH",
            "SOL",
        ],
        columns=[
            "BTC",
            "ETH",
            "SOL",
        ],
    )


# ---------------------------------------------------------------------------
# General strategy construction
# ---------------------------------------------------------------------------


def test_strategy_can_be_created(
    signals: pd.Series,
):
    strategy = ResearchStrategy(
        name="Test Strategy",
        signals=signals,
        portfolio_method="signal_weighted",
    )

    assert strategy.name == "Test Strategy"

    assert strategy.portfolio_method == (
        "signal_weighted"
    )

    pd.testing.assert_series_equal(
        strategy.signals,
        signals,
    )


def test_strategy_returns_series(
    signals: pd.Series,
):
    strategy = ResearchStrategy(
        name="Test Strategy",
        signals=signals,
        portfolio_method="signal_weighted",
    )

    weights = strategy.weights()

    assert isinstance(
        weights,
        pd.Series,
    )


def test_strategy_preserves_assets(
    signals: pd.Series,
):
    strategy = ResearchStrategy(
        name="Test Strategy",
        signals=signals,
        portfolio_method="signal_weighted",
    )

    weights = strategy.weights()

    assert set(weights.index) == set(
        signals.index
    )


# ---------------------------------------------------------------------------
# Equal-weight strategy
# ---------------------------------------------------------------------------


def test_equal_weight_strategy(
    signals: pd.Series,
):
    strategy = ResearchStrategy(
        name="Equal Weight",
        signals=signals,
        portfolio_method="equal_weight",
    )

    weights = strategy.weights()

    assert weights.sum() == pytest.approx(
        1.0,
    )

    assert np.allclose(
        weights.to_numpy(),
        1.0 / len(signals),
    )


def test_equal_weight_strategy_is_long_only(
    signals: pd.Series,
):
    strategy = ResearchStrategy(
        name="Equal Weight",
        signals=signals,
        portfolio_method="equal_weight",
    )

    weights = strategy.weights()

    assert (weights >= 0.0).all()


# ---------------------------------------------------------------------------
# Signal-weighted strategy
# ---------------------------------------------------------------------------


def test_signal_weighted_strategy(
    signals: pd.Series,
):
    strategy = ResearchStrategy(
        name="Signal Strategy",
        signals=signals,
        portfolio_method="signal_weighted",
    )

    weights = strategy.weights()

    assert weights.abs().sum() == pytest.approx(
        1.0,
    )


def test_signal_weighted_strategy_preserves_direction():
    signals = pd.Series(
        {
            "BTC": 0.8,
            "ETH": -0.4,
            "SOL": 0.2,
        }
    )

    strategy = ResearchStrategy(
        name="Long Short",
        signals=signals,
        portfolio_method="signal_weighted",
    )

    weights = strategy.weights()

    assert weights["BTC"] > 0.0

    assert weights["ETH"] < 0.0

    assert weights["SOL"] > 0.0


# ---------------------------------------------------------------------------
# Global minimum-variance strategy
# ---------------------------------------------------------------------------


def test_global_minimum_variance_strategy(
    signals: pd.Series,
    covariance: pd.DataFrame,
):
    strategy = ResearchStrategy(
        name="GMV",
        signals=signals,
        portfolio_method=(
            "global_minimum_variance"
        ),
        covariance=covariance,
    )

    weights = strategy.weights()

    assert isinstance(
        weights,
        pd.Series,
    )

    assert set(weights.index) == set(
        signals.index
    )

    assert np.isfinite(
        weights.to_numpy(),
    ).all()


def test_global_minimum_variance_requires_covariance(
    signals: pd.Series,
):
    strategy = ResearchStrategy(
        name="GMV",
        signals=signals,
        portfolio_method=(
            "global_minimum_variance"
        ),
    )

    with pytest.raises(ValueError):

        strategy.weights()


# ---------------------------------------------------------------------------
# Unsupported methods
# ---------------------------------------------------------------------------


def test_unsupported_portfolio_method_is_rejected(
    signals: pd.Series,
):
    strategy = ResearchStrategy(
        name="Invalid Strategy",
        signals=signals,
        portfolio_method="unsupported",
    )

    with pytest.raises(ValueError):

        strategy.weights()


# ---------------------------------------------------------------------------
# Exposure constraints
# ---------------------------------------------------------------------------


def test_max_gross_exposure_is_enforced():
    signals = pd.Series(
        {
            "BTC": 0.8,
            "ETH": 0.2,
        }
    )

    strategy = ResearchStrategy(
        name="Constrained Strategy",
        signals=signals,
        portfolio_method="signal_weighted",
        max_gross_exposure=0.5,
    )

    with pytest.raises(ValueError):

        strategy.weights()


def test_min_net_exposure_is_enforced():
    signals = pd.Series(
        {
            "BTC": 0.5,
            "ETH": -0.5,
        }
    )

    strategy = ResearchStrategy(
        name="Long Only Requirement",
        signals=signals,
        portfolio_method="signal_weighted",
        min_net_exposure=0.5,
    )

    with pytest.raises(ValueError):

        strategy.weights()


def test_max_net_exposure_is_enforced():
    signals = pd.Series(
        {
            "BTC": 0.8,
            "ETH": 0.2,
        }
    )

    strategy = ResearchStrategy(
        name="Market Neutral Requirement",
        signals=signals,
        portfolio_method="signal_weighted",
        max_net_exposure=0.5,
    )

    with pytest.raises(ValueError):

        strategy.weights()


def test_valid_exposure_constraints_are_accepted(
    signals: pd.Series,
):
    strategy = ResearchStrategy(
        name="Constrained Strategy",
        signals=signals,
        portfolio_method="signal_weighted",
        max_gross_exposure=1.0,
        min_net_exposure=0.0,
        max_net_exposure=1.0,
    )

    weights = strategy.weights()

    assert weights.abs().sum() == pytest.approx(
        1.0,
    )


# ---------------------------------------------------------------------------
# Strategy metadata
# ---------------------------------------------------------------------------


def test_strategy_name_is_preserved(
    signals: pd.Series,
):
    strategy = ResearchStrategy(
        name="Momentum Strategy",
        signals=signals,
        portfolio_method="signal_weighted",
    )

    assert strategy.name == (
        "Momentum Strategy"
    )


def test_strategy_method_is_preserved(
    signals: pd.Series,
):
    strategy = ResearchStrategy(
        name="Signal Strategy",
        signals=signals,
        portfolio_method="signal_weighted",
    )

    assert strategy.portfolio_method == (
        "signal_weighted"
    )


# ---------------------------------------------------------------------------
# Strategy result integrity
# ---------------------------------------------------------------------------


def test_strategy_weights_are_finite(
    signals: pd.Series,
):
    strategy = ResearchStrategy(
        name="Finite Weights",
        signals=signals,
        portfolio_method="signal_weighted",
    )

    weights = strategy.weights()

    assert np.isfinite(
        weights.to_numpy(),
    ).all()


def test_strategy_weights_have_weight_name(
    signals: pd.Series,
):
    strategy = ResearchStrategy(
        name="Named Weights",
        signals=signals,
        portfolio_method="signal_weighted",
    )

    weights = strategy.weights()

    assert weights.name == "weight"


