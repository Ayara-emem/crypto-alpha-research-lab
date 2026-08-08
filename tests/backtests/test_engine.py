"""
Tests for the CARL backtesting engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.backtest.cost import (
    ProportionalCostModel,
)

from crypto_alpha_lab.backtest.engine import (
    BacktestEngine,
    BacktestResult,
)

from crypto_alpha_lab.research.dataset import (
    ResearchDataset,
)

from crypto_alpha_lab.research.experiment import (
    ResearchExperiment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prices() -> pd.DataFrame:
    """
    Multi-asset price dataset.
    """

    index = pd.date_range(
        "2024-01-01",
        periods=10,
        freq="D",
    )

    return pd.DataFrame(
        {
            "BTC": np.linspace(
                100.0,
                110.0,
                10,
            ),
            "ETH": np.linspace(
                50.0,
                55.0,
                10,
            ),
            "Volume": np.linspace(
                1000.0,
                2000.0,
                10,
            ),
        },
        index=index,
    )


@pytest.fixture
def dataset(
    prices: pd.DataFrame,
) -> ResearchDataset:
    """
    Research dataset fixture.
    """

    return ResearchDataset(
        prices=prices,
    )


@pytest.fixture
def experiment(
    dataset: ResearchDataset,
) -> ResearchExperiment:
    """
    Multi-asset research experiment.
    """

    experiment = ResearchExperiment(
        dataset=dataset,
    )

    experiment.price_columns = [
        "BTC",
        "ETH",
    ]

    return experiment


@pytest.fixture
def portfolio() -> pd.Series:
    """
    Equal-weight portfolio.
    """

    return pd.Series(
        {
            "BTC": 0.5,
            "ETH": 0.5,
        },
        dtype=float,
    )


@pytest.fixture
def configured_experiment(
    experiment: ResearchExperiment,
    portfolio: pd.Series,
) -> ResearchExperiment:
    """
    Fully configured experiment.
    """

    experiment.asset_universe = [
        "BTC",
        "ETH",
    ]

    experiment.portfolio = portfolio

    return experiment


@pytest.fixture
def engine() -> BacktestEngine:
    """
    Backtest engine fixture.
    """

    return BacktestEngine()


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


def test_backtest_result_creation(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    BacktestResult should be created successfully.
    """

    result = engine.run(
        configured_experiment,
    )

    assert isinstance(
        result,
        BacktestResult,
    )


def test_backtest_result_contains_asset_returns(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Result should contain asset returns.
    """

    result = engine.run(
        configured_experiment,
    )

    assert isinstance(
        result.asset_returns,
        pd.DataFrame,
    )


def test_backtest_result_contains_portfolio_returns(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Result should contain portfolio returns.
    """

    result = engine.run(
        configured_experiment,
    )

    assert isinstance(
        result.portfolio_returns,
        pd.Series,
    )


def test_backtest_result_contains_cumulative_returns(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Result should contain cumulative returns.
    """

    result = engine.run(
        configured_experiment,
    )

    assert isinstance(
        result.cumulative_returns,
        pd.Series,
    )


def test_backtest_result_contains_weights(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Result should contain portfolio weights.
    """

    result = engine.run(
        configured_experiment,
    )

    pd.testing.assert_series_equal(
        result.weights,
        configured_experiment.portfolio,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_engine_rejects_invalid_experiment(
    engine: BacktestEngine,
):
    """
    Engine should reject non-ResearchExperiment inputs.
    """

    with pytest.raises(TypeError):

        engine.run(
            object(),
        )


def test_engine_rejects_missing_portfolio(
    experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Engine should reject experiments without portfolios.
    """

    with pytest.raises(ValueError):

        engine.run(
            experiment,
        )


def test_engine_rejects_empty_prices(
    engine: BacktestEngine,
):
    """
    Engine should reject empty price data.
    """

    empty_prices = pd.DataFrame()

    dataset = ResearchDataset(
        prices=empty_prices,
    )

    experiment = ResearchExperiment(
        dataset=dataset,
    )

    experiment.portfolio = pd.Series(
        {"BTC": 1.0},
    )

    with pytest.raises(ValueError):

        engine.run(
            experiment,
        )


def test_engine_rejects_unknown_price_columns(
    dataset: ResearchDataset,
    engine: BacktestEngine,
):
    """
    Engine should reject unknown price columns.
    """

    experiment = ResearchExperiment(
        dataset=dataset,
    )

    experiment.price_columns = [
        "NOT_A_PRICE",
    ]

    experiment.portfolio = pd.Series(
        {"BTC": 1.0},
    )

    with pytest.raises(ValueError):

        engine.run(
            experiment,
        )


def test_engine_rejects_empty_asset_universe(
    dataset: ResearchDataset,
    engine: BacktestEngine,
):
    """
    Empty asset universe should be rejected.
    """

    experiment = ResearchExperiment(
        dataset=dataset,
    )

    experiment.price_columns = [
        "BTC",
    ]

    experiment.asset_universe = []

    experiment.portfolio = pd.Series(
        {"BTC": 1.0},
    )

    with pytest.raises(ValueError):

        engine.run(
            experiment,
        )


# ---------------------------------------------------------------------------
# Price preparation
# ---------------------------------------------------------------------------


def test_prepare_asset_returns_uses_configured_columns(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Only explicitly configured price columns should
    be converted into asset returns.
    """

    result = engine.run(
        configured_experiment,
    )

    assert list(
        result.asset_returns.columns
    ) == [
        "BTC",
        "ETH",
    ]


def test_prepare_asset_returns_excludes_volume(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Volume must not be treated as an asset price.
    """

    result = engine.run(
        configured_experiment,
    )

    assert "Volume" not in (
        result.asset_returns.columns
    )


def test_prepare_asset_returns_default_close(
    dataset: ResearchDataset,
    engine: BacktestEngine,
):
    """
    A Close column should be selected by default
    when price_columns are not configured.
    """

    prices = dataset.prices.copy()

    prices["Close"] = np.linspace(
        100.0,
        110.0,
        len(prices),
    )

    close_dataset = ResearchDataset(
        prices=prices,
    )

    experiment = ResearchExperiment(
        dataset=close_dataset,
    )

    experiment.portfolio = pd.Series(
        {"Close": 1.0},
    )

    result = engine.run(
        experiment,
    )

    assert list(
        result.asset_returns.columns
    ) == ["Close"]


# ---------------------------------------------------------------------------
# Weight preparation
# ---------------------------------------------------------------------------


def test_weights_are_aligned_to_asset_universe(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Weights should follow the declared asset universe.
    """

    configured_experiment.asset_universe = [
        "ETH",
        "BTC",
    ]

    result = engine.run(
        configured_experiment,
    )

    assert list(
        result.weights.index
    ) == [
        "ETH",
        "BTC",
    ]


def test_missing_portfolio_assets_receive_zero_weight(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Assets in the declared universe but absent from
    the portfolio should receive zero weight.
    """

    configured_experiment.asset_universe = [
        "BTC",
        "ETH",
    ]

    configured_experiment.portfolio = pd.Series(
        {"BTC": 1.0},
    )

    result = engine.run(
        configured_experiment,
    )

    assert result.weights["BTC"] == 1.0

    assert result.weights["ETH"] == 0.0


def test_unknown_asset_is_rejected(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Unknown assets must be rejected.
    """

    configured_experiment.asset_universe = [
        "BTC",
        "UNKNOWN",
    ]

    configured_experiment.portfolio = pd.Series(
        {
            "BTC": 0.5,
            "UNKNOWN": 0.5,
        }
    )

    with pytest.raises(ValueError):

        engine.run(
            configured_experiment,
        )


# ---------------------------------------------------------------------------
# Portfolio return calculations
# ---------------------------------------------------------------------------


def test_portfolio_returns_are_series(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Portfolio returns should be a Series.
    """

    result = engine.run(
        configured_experiment,
    )

    assert isinstance(
        result.portfolio_returns,
        pd.Series,
    )


def test_portfolio_returns_have_expected_length(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Portfolio returns should preserve the return
    index generated by APRL.
    """

    result = engine.run(
        configured_experiment,
    )

    assert len(
        result.portfolio_returns
    ) == len(
        result.asset_returns,
    )


def test_cumulative_returns_match_result_length(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Cumulative returns should align with portfolio
    returns.
    """

    result = engine.run(
        configured_experiment,
    )

    assert len(
        result.cumulative_returns,
    ) == len(
        result.portfolio_returns,
    )


# ---------------------------------------------------------------------------
# Turnover
# ---------------------------------------------------------------------------


def test_initial_turnover_equals_gross_exposure(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Initial turnover should equal gross portfolio exposure.
    """

    result = engine.run(
        configured_experiment,
    )

    expected = (
        configured_experiment.portfolio
        .abs()
        .sum()
    )

    assert result.turnover == pytest.approx(
        expected,
    )


def test_turnover_is_non_negative(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Turnover cannot be negative.
    """

    result = engine.run(
        configured_experiment,
    )

    assert result.turnover >= 0.0


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------


def test_no_cost_model_produces_zero_cost(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Without a cost model, transaction costs should
    equal zero.
    """

    result = engine.run(
        configured_experiment,
    )

    assert result.transaction_costs == 0.0


def test_proportional_cost_model_is_applied(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Proportional transaction costs should be calculated
    from turnover.
    """

    cost_model = ProportionalCostModel(
        rate=0.01,
    )

    result = engine.run(
        configured_experiment,
        cost_model=cost_model,
    )

    expected = (
        result.turnover
        * 0.01
    )

    assert result.transaction_costs == pytest.approx(
        expected,
    )


def test_transaction_cost_does_not_change_return_series(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    v1.0 records transaction costs separately rather
    than subtracting a one-time cost from every
    return observation.
    """

    without_cost = engine.run(
        configured_experiment,
    )

    with_cost = engine.run(
        configured_experiment,
        cost_model=ProportionalCostModel(
            rate=0.01,
        ),
    )

    pd.testing.assert_series_equal(
        without_cost.portfolio_returns,
        with_cost.portfolio_returns,
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_default_metadata_is_present(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Backtest results should contain default metadata.
    """

    result = engine.run(
        configured_experiment,
    )

    assert result.metadata["engine"] == "CARL"

    assert (
        result.metadata["engine_component"]
        == "backtest"
    )

    assert result.metadata["version"] == "1.0"


# ---------------------------------------------------------------------------
# Result integrity
# ---------------------------------------------------------------------------


def test_result_weights_are_floats(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    Result weights should have floating-point dtype.
    """

    result = engine.run(
        configured_experiment,
    )

    assert pd.api.types.is_float_dtype(
        result.weights.dtype,
    )


def test_result_contains_all_required_fields(
    configured_experiment: ResearchExperiment,
    engine: BacktestEngine,
):
    """
    BacktestResult should expose all expected fields.
    """

    result = engine.run(
        configured_experiment,
    )

    assert hasattr(
        result,
        "asset_returns",
    )

    assert hasattr(
        result,
        "portfolio_returns",
    )

    assert hasattr(
        result,
        "cumulative_returns",
    )

    assert hasattr(
        result,
        "weights",
    )

    assert hasattr(
        result,
        "turnover",
    )

    assert hasattr(
        result,
        "transaction_costs",
    )

    assert hasattr(
        result,
        "metadata",
    )