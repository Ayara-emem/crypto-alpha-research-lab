"""
Tests for the CARL research pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.backtest.engine import (
    BacktestEngine,
    BacktestResult,
)

from crypto_alpha_lab.pipeline.research_pipeline import (
    ResearchPipeline,
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
    Research signals.
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
    Signal-weighted research strategy.
    """

    return ResearchStrategy(
        name="Pipeline Strategy",
        signals=signals,
        portfolio_method="signal_weighted",
    )


@pytest.fixture
def runner() -> StrategyRunner:
    """
    Strategy runner.
    """

    return StrategyRunner()


@pytest.fixture
def pipeline() -> ResearchPipeline:
    """
    Default research pipeline.
    """

    return ResearchPipeline()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_pipeline_can_be_created():
    pipeline = ResearchPipeline()

    assert isinstance(
        pipeline,
        ResearchPipeline,
    )


def test_pipeline_can_accept_custom_runner(
    runner: StrategyRunner,
):
    pipeline = ResearchPipeline(
        runner=runner,
    )

    assert pipeline.runner is runner


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_pipeline_rejects_invalid_experiment(
    pipeline: ResearchPipeline,
    strategy: ResearchStrategy,
):
    with pytest.raises(TypeError):

        pipeline.run(
            experiment=object(),
            strategy=strategy,
        )


def test_pipeline_rejects_invalid_strategy(
    pipeline: ResearchPipeline,
    experiment: ResearchExperiment,
):
    with pytest.raises(TypeError):

        pipeline.run(
            experiment=experiment,
            strategy=object(),
        )


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def test_pipeline_returns_backtest_result(
    pipeline: ResearchPipeline,
    experiment: ResearchExperiment,
    strategy: ResearchStrategy,
):
    result = pipeline.run(
        experiment=experiment,
        strategy=strategy,
    )

    assert isinstance(
        result,
        BacktestResult,
    )


def test_pipeline_populates_experiment_portfolio(
    pipeline: ResearchPipeline,
    experiment: ResearchExperiment,
    strategy: ResearchStrategy,
):
    pipeline.run(
        experiment=experiment,
        strategy=strategy,
    )

    assert experiment.portfolio is not None

    assert isinstance(
        experiment.portfolio,
        pd.Series,
    )


def test_pipeline_weights_match_strategy(
    pipeline: ResearchPipeline,
    experiment: ResearchExperiment,
    strategy: ResearchStrategy,
):
    expected_weights = strategy.weights()

    pipeline.run(
        experiment=experiment,
        strategy=strategy,
    )

    pd.testing.assert_series_equal(
        experiment.portfolio,
        expected_weights,
    )


def test_pipeline_produces_asset_returns(
    pipeline: ResearchPipeline,
    experiment: ResearchExperiment,
    strategy: ResearchStrategy,
):
    result = pipeline.run(
        experiment=experiment,
        strategy=strategy,
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


def test_pipeline_produces_portfolio_returns(
    pipeline: ResearchPipeline,
    experiment: ResearchExperiment,
    strategy: ResearchStrategy,
):
    result = pipeline.run(
        experiment=experiment,
        strategy=strategy,
    )

    assert isinstance(
        result.portfolio_returns,
        pd.Series,
    )

    assert not result.portfolio_returns.empty


def test_pipeline_produces_cumulative_returns(
    pipeline: ResearchPipeline,
    experiment: ResearchExperiment,
    strategy: ResearchStrategy,
):
    result = pipeline.run(
        experiment=experiment,
        strategy=strategy,
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


def test_pipeline_passes_cost_model(
    pipeline: ResearchPipeline,
    experiment: ResearchExperiment,
    strategy: ResearchStrategy,
):
    class DummyCostModel:

        def cost(
            self,
            turnover: float,
        ) -> float:
            return 0.01

    result = pipeline.run(
        experiment=experiment,
        strategy=strategy,
        cost_model=DummyCostModel(),
    )

    assert result.transaction_costs == pytest.approx(
        0.01,
    )


# ---------------------------------------------------------------------------
# Custom runner
# ---------------------------------------------------------------------------


def test_pipeline_uses_injected_runner(
    runner: StrategyRunner,
    experiment: ResearchExperiment,
    strategy: ResearchStrategy,
):
    pipeline = ResearchPipeline(
        runner=runner,
    )

    result = pipeline.run(
        experiment=experiment,
        strategy=strategy,
    )

    assert isinstance(
        result,
        BacktestResult,
    )


# ---------------------------------------------------------------------------
# Different strategy methods
# ---------------------------------------------------------------------------


def test_pipeline_supports_equal_weight_strategy(
    pipeline: ResearchPipeline,
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

    result = pipeline.run(
        experiment=experiment,
        strategy=strategy,
    )

    assert result.weights.sum() == pytest.approx(
        1.0,
    )


def test_pipeline_supports_long_short_strategy(
    pipeline: ResearchPipeline,
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

    result = pipeline.run(
        experiment=experiment,
        strategy=strategy,
    )

    assert result.weights["BTC"] > 0.0

    assert result.weights["ETH"] < 0.0


# ---------------------------------------------------------------------------
# Result integrity
# ---------------------------------------------------------------------------


def test_pipeline_result_contains_weights(
    pipeline: ResearchPipeline,
    experiment: ResearchExperiment,
    strategy: ResearchStrategy,
):
    result = pipeline.run(
        experiment=experiment,
        strategy=strategy,
    )

    assert isinstance(
        result.weights,
        pd.Series,
    )

    assert np.isfinite(
        result.weights.to_numpy(),
    ).all()


def test_pipeline_result_contains_metadata(
    pipeline: ResearchPipeline,
    experiment: ResearchExperiment,
    strategy: ResearchStrategy,
):
    result = pipeline.run(
        experiment=experiment,
        strategy=strategy,
    )

    assert result.metadata["engine"] == "CARL"

    assert (
        result.metadata["engine_component"]
        == "backtest"
    )


def test_pipeline_result_contains_turnover(
    pipeline: ResearchPipeline,
    experiment: ResearchExperiment,
    strategy: ResearchStrategy,
):
    result = pipeline.run(
        experiment=experiment,
        strategy=strategy,
    )

    assert result.turnover >= 0.0


def test_pipeline_result_contains_transaction_costs(
    pipeline: ResearchPipeline,
    experiment: ResearchExperiment,
    strategy: ResearchStrategy,
):
    result = pipeline.run(
        experiment=experiment,
        strategy=strategy,
    )

    assert result.transaction_costs >= 0.0