"""
Research pipeline orchestration.

Coordinates research experiments, strategies,
and backtest execution.
"""

from __future__ import annotations

from dataclasses import dataclass

from crypto_alpha_lab.backtest.engine import (
    BacktestResult,
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


@dataclass(slots=True)
class ResearchPipeline:
    """
    Top-level CARL research execution pipeline.

    Parameters
    ----------
    runner
        Strategy runner used to execute the experiment.
        If omitted, a default StrategyRunner is created.
    """

    runner: StrategyRunner | None = None

    def run(
        self,
        experiment: ResearchExperiment,
        strategy: ResearchStrategy,
        cost_model=None,
    ) -> BacktestResult:
        """
        Execute a research experiment through a strategy.

        Parameters
        ----------
        experiment
            Research experiment containing the dataset
            and research configuration.

        strategy
            Research strategy responsible for portfolio
            construction.

        cost_model
            Optional transaction cost model.

        Returns
        -------
        BacktestResult
            Completed backtest result.
        """

        if not isinstance(
            experiment,
            ResearchExperiment,
        ):
            raise TypeError(
                "experiment must be a ResearchExperiment."
            )

        if not isinstance(
            strategy,
            ResearchStrategy,
        ):
            raise TypeError(
                "strategy must be a ResearchStrategy."
            )

        if self.runner is None:
            runner = StrategyRunner()
        else:
            runner = self.runner

        return runner.run(
            strategy=strategy,
            experiment=experiment,
            cost_model=cost_model,
        )
        