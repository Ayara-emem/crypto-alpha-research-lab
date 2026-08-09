"""
Strategy execution.

Connects ResearchStrategy objects to the CARL
backtesting engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from crypto_alpha_lab.backtest.engine import (
    BacktestEngine,
    BacktestResult,
)

from crypto_alpha_lab.research.experiment import (
    ResearchExperiment,
)

from crypto_alpha_lab.strategy.strategy import (
    ResearchStrategy,
)


@dataclass(slots=True)
class StrategyRunner:
    """
    Execute research strategies through the
    CARL backtesting engine.

    Parameters
    ----------
    engine
        Backtesting engine. If omitted, a default
        BacktestEngine is created.
    """

    engine: BacktestEngine | None = None

    def run(
        self,
        strategy: ResearchStrategy,
        experiment: ResearchExperiment,
        cost_model=None,
    ) -> BacktestResult:
        """
        Execute a strategy against a research experiment.

        Parameters
        ----------
        strategy
            Research strategy.

        experiment
            Research experiment containing the dataset.

        cost_model
            Optional transaction cost model.

        Returns
        -------
        BacktestResult
            Completed strategy backtest.
        """

        if not isinstance(
            strategy,
            ResearchStrategy,
        ):
            raise TypeError(
                "strategy must be a ResearchStrategy."
            )

        if not isinstance(
            experiment,
            ResearchExperiment,
        ):
            raise TypeError(
                "experiment must be a ResearchExperiment."
            )

        weights = strategy.weights()

        experiment.portfolio = weights

        if self.engine is None:
            engine = BacktestEngine()
        else:
            engine = self.engine

        return engine.run(
            experiment,
            cost_model=cost_model,
        )

