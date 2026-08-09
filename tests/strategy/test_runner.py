"""
Tests for CARL strategy execution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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

from crypto_alpha_lab.strategy.runner import (
    StrategyRunner,
)

from crypto_alpha_lab.strategy.strategy import (
    ResearchStrategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prices() -> pd.DataFrame:
    """
    Multi-asset price data.
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
def experiment(
    prices: pd.DataFrame,
) -> ResearchExperiment:
    """
    Fully configured research experiment.
    """

    dataset = ResearchDataset(
        prices=prices,
    )

    experiment = ResearchExperiment(
        dataset=dataset,
    )

    experiment.price_columns = [
        "BTC",
        "ETH",
    ]

    experiment.asset_universe = [
        "BTC",
        "ETH",
    ]

    return experiment


@pytest.fixture
def signals() -> pd.Series:
    """
    Strategy signals.
    """

    return pd.Series(
        {
            "BTC": 0.6,
            "ETH": 0.4,
        },
        dtype=float,
    )


@pytest.fixture
def strategy(
    signals: pd.Series,
) -> ResearchStrategy:
    """
    Signal-weighted strategy.
    """

    return ResearchStrategy(
        name="Test Strategy",
        signals=signals,
        portfolio_method="signal_weighted",
    )


@pytest.fixture
def engine() -> BacktestEngine:
    """
    Backtest engine.
    """

    return BacktestEngine()


@pytest.fixture
def runner() -> StrategyRunner:
    """
    Default strategy runner.
    """

    return StrategyRunner()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_strategy_runner_can_be_created():
    runner = StrategyRunner()

    assert isinstance(
        runner,
        StrategyRunner,
    )


def test_strategy_runner_can_accept_custom_engine(
    engine: BacktestEngine,
):
    runner = StrategyRunner(
        engine=engine,
    )

    assert runner.engine is engine


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_runner_rejects_invalid_strategy(
    runner: StrategyRunner,
    experiment: ResearchExperiment,
):
    with pytest.raises(TypeError):

        runner.run(
            strategy=object(),
            experiment=experiment,
        )


def test_runner_rejects_invalid_experiment(
    runner: StrategyRunner,
    strategy: ResearchStrategy,
):
    with pytest.raises(TypeError):

        runner.run(
            strategy=strategy,
            experiment=object(),
        )


# ---------------------------------------------------------------------------
# Strategy execution
# ---------------------------------------------------------------------------


def test_runner_returns_backtest_result(
    runner: StrategyRunner,
    strategy: ResearchStrategy,
    experiment: ResearchExperiment,
):
    result = runner.run(
        strategy=strategy,
        experiment=experiment,
    )

    assert isinstance(
        result,
        BacktestResult,
    )


def test_runner_generates_portfolio_weights(
    runner: StrategyRunner,
    strategy: ResearchStrategy,
    experiment: ResearchExperiment,
):
    runner.run(
        strategy=strategy,
        experiment=experiment,
    )

    assert experiment.portfolio is not None

    assert isinstance(
        experiment.portfolio,
        pd.Series,
    )


def test_runner_weights_match_strategy(
    runner: StrategyRunner,
    strategy: ResearchStrategy,
    experiment: ResearchExperiment,
):
    expected_weights = strategy.weights()

    runner.run(
        strategy=strategy,
        experiment=experiment,
    )

    pd.testing.assert_series_equal(
        experiment.portfolio,
        expected_weights,
    )


def test_runner_preserves_asset_universe(
    runner: StrategyRunner,
    strategy: ResearchStrategy,
    experiment: ResearchExperiment,
):
    runner.run(
        strategy=strategy,
        experiment=experiment,
    )

    assert list(
        experiment.asset_universe,
    ) == [
        "BTC",
        "ETH",
    ]


def test_runner_produces_portfolio_returns(
    runner: StrategyRunner,
    strategy: ResearchStrategy,
    experiment: ResearchExperiment,
):
    result = runner.run(
        strategy=strategy,
        experiment=experiment,
    )

    assert isinstance(
        result.portfolio_returns,
        pd.Series,
    )

    assert not result.portfolio_returns.empty


def test_runner_produces_cumulative_returns(
    runner: StrategyRunner,
    strategy: ResearchStrategy,
    experiment: ResearchExperiment,
):
    result = runner.run(
        strategy=strategy,
        experiment=experiment,
    )

    assert isinstance(
        result.cumulative_returns,
        pd.Series,
    )

    assert len(
        result.cumulative_returns,
    ) == len(
        result.portfolio_returns,
    )


# ---------------------------------------------------------------------------
# Cost model propagation
# ---------------------------------------------------------------------------


def test_runner_passes_cost_model(
    runner: StrategyRunner,
    strategy: ResearchStrategy,
    experiment: ResearchExperiment,
):
    class DummyCostModel:

        def cost(
            self,
            turnover: float,
        ) -> float:
            return 0.01

    result = runner.run(
        strategy=strategy,
        experiment=experiment,
        cost_model=DummyCostModel(),
    )

    assert result.transaction_costs == pytest.approx(
        0.01,
    )


# ---------------------------------------------------------------------------
# Custom engine
# ---------------------------------------------------------------------------


def test_runner_uses_injected_engine(
    engine: BacktestEngine,
    strategy: ResearchStrategy,
    experiment: ResearchExperiment,
):
    runner = StrategyRunner(
        engine=engine,
    )

    result = runner.run(
        strategy=strategy,
        experiment=experiment,
    )

    assert isinstance(
        result,
        BacktestResult,
    )


# ---------------------------------------------------------------------------
# Strategy behavior
# ---------------------------------------------------------------------------


def test_runner_supports_equal_weight_strategy(
    runner: StrategyRunner,
    experiment: ResearchExperiment,
):
    strategy = ResearchStrategy(
        name="Equal Weight",
        signals=pd.Series(
            {
                "BTC": 1.0,
                "ETH": 1.0,
            }
        ),
        portfolio_method="equal_weight",
    )

    result = runner.run(
        strategy=strategy,
        experiment=experiment,
    )

    assert result.weights.sum() == pytest.approx(
        1.0,
    )


def test_runner_supports_long_short_strategy(
    runner: StrategyRunner,
    experiment: ResearchExperiment,
):
    strategy = ResearchStrategy(
        name="Long Short",
        signals=pd.Series(
            {
                "BTC": 1.0,
                "ETH": -1.0,
            }
        ),
        portfolio_method="signal_weighted",
    )

    result = runner.run(
        strategy=strategy,
        experiment=experiment,
    )

    assert result.weights["BTC"] > 0.0

    assert result.weights["ETH"] < 0.0


# ---------------------------------------------------------------------------
# Result integrity
# ---------------------------------------------------------------------------


def test_runner_result_contains_asset_returns(
    runner: StrategyRunner,
    strategy: ResearchStrategy,
    experiment: ResearchExperiment,
):
    result = runner.run(
        strategy=strategy,
        experiment=experiment,
    )

    assert isinstance(
        result.asset_returns,
        pd.DataFrame,
    )

    assert list(
        result.asset_returns.columns,
    ) == [
        "BTC",
        "ETH",
    ]


def test_runner_result_contains_weights(
    runner: StrategyRunner,
    strategy: ResearchStrategy,
    experiment: ResearchExperiment,
):
    result = runner.run(
        strategy=strategy,
        experiment=experiment,
    )

    assert isinstance(
        result.weights,
        pd.Series,
    )

    assert np.isfinite(
        result.weights.to_numpy(),
    ).all()


def test_runner_result_contains_metadata(
    runner: StrategyRunner,
    strategy: ResearchStrategy,
    experiment: ResearchExperiment,
):
    result = runner.run(
        strategy=strategy,
        experiment=experiment,
    )

    assert result.metadata["engine"] == "CARL"

    assert (
        result.metadata["engine_component"]
        == "backtest"
    )


def test_runner_result_turnover_is_non_negative(
    runner: StrategyRunner,
    strategy: ResearchStrategy,
    experiment: ResearchExperiment,
):
    result = runner.run(
        strategy=strategy,
        experiment=experiment,
    )

    assert result.turnover >= 0.0

