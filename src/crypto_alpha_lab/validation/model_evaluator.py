"""
Training-aware walk-forward model evaluation.

This module connects CARL's research model, strategy,
walk-forward validation, and backtesting layers.

The evaluator is responsible for orchestration only.
Portfolio-return and transaction-cost calculations remain
owned by BacktestEngine.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

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
from crypto_alpha_lab.strategy.strategy import (
    ResearchStrategy,
)
from crypto_alpha_lab.validation.model import (
    ResearchModel,
)
from crypto_alpha_lab.validation.walk_forward import (
    WalkForwardSplit,
)


@dataclass(frozen=True, slots=True)
class ModelEvaluationResult:
    """
    Aggregated result from training-aware
    walk-forward evaluation.

    Parameters
    ----------
    returns
        Concatenated out-of-sample portfolio returns.

    folds
        Individual out-of-sample backtest results.
    """

    returns: pd.Series

    folds: tuple[BacktestResult, ...]

    @property
    def cumulative_returns(self) -> pd.Series:
        """
        Cumulative out-of-sample return path.
        """

        return (
            (1.0 + self.returns)
            .cumprod()
            - 1.0
        )


class WalkForwardModelEvaluator:
    """
    Execute a ResearchModel through chronological
    walk-forward folds.

    The model is fitted independently on each training
    fold and then used to generate signals for the
    corresponding unseen test fold.
    """

    def __init__(
        self,
        backtest_engine: BacktestEngine | None = None,
    ) -> None:
        self.backtest_engine = (
            backtest_engine
            if backtest_engine is not None
            else BacktestEngine()
        )

    def evaluate(
        self,
        model: ResearchModel,
        splits: list[WalkForwardSplit],
        strategy_name: str = "WalkForward Strategy",
        portfolio_method: str = "signal_weighted",
        cost_model=None,
    ) -> ModelEvaluationResult:
        """
        Perform training-aware walk-forward evaluation.

        Parameters
        ----------
        model
            Trainable CARL research model.

        splits
            Chronological walk-forward splits.

        strategy_name
            Name assigned to each fold strategy.

        portfolio_method
            Portfolio construction method used to convert
            model signals into portfolio weights.

        cost_model
            Optional transaction-cost model passed directly
            to BacktestEngine.

        Returns
        -------
        ModelEvaluationResult
            Aggregated out-of-sample results.
        """

        if not isinstance(
            splits,
            list,
        ):
            raise TypeError(
                "splits must be a list."
            )

        if not splits:
            raise ValueError(
                "splits cannot be empty."
            )

        if not strategy_name:
            raise ValueError(
                "strategy_name cannot be empty."
            )

        fold_results: list[
            BacktestResult
        ] = []

        for split in splits:

            if not isinstance(
                split,
                WalkForwardSplit,
            ):
                raise TypeError(
                    "every split must be a "
                    "WalkForwardSplit."
                )

            if split.train.empty:
                raise ValueError(
                    f"Fold {split.fold} has "
                    "an empty training set."
                )

            if split.test.empty:
                raise ValueError(
                    f"Fold {split.fold} has "
                    "an empty test set."
                )

            # ---------------------------------------------------------------
            # 1. FIT USING TRAINING DATA ONLY
            # ---------------------------------------------------------------

            model.fit(
                split.train,
            )

            # ---------------------------------------------------------------
            # 2. GENERATE SIGNALS FOR UNSEEN TEST DATA
            # ---------------------------------------------------------------

            signals = model.predict(
                split.test,
            )

            if not isinstance(
                signals,
                pd.Series,
            ):
                raise TypeError(
                    "model.predict() must return "
                    "a pandas Series."
                )

            if signals.empty:
                raise ValueError(
                    f"Fold {split.fold} produced "
                    "empty signals."
                )

            # ---------------------------------------------------------------
            # 3. CONVERT SIGNALS INTO PORTFOLIO WEIGHTS
            # ---------------------------------------------------------------

            strategy = ResearchStrategy(
                name=strategy_name,
                signals=signals,
                portfolio_method=portfolio_method,
            )

            weights = strategy.weights()

            if not isinstance(
                weights,
                pd.Series,
            ):
                raise TypeError(
                    "strategy.weights() must return "
                    "a pandas Series."
                )

            # ---------------------------------------------------------------
            # 4. BUILD A TEST-FOLD RESEARCH EXPERIMENT
            # ---------------------------------------------------------------

            test_prices = split.test.copy()

            dataset = ResearchDataset(
                prices=test_prices,
            )

            experiment = ResearchExperiment(
                dataset=dataset,
            )

            experiment.price_columns = list(
                test_prices.columns
            )

            experiment.asset_universe = list(
                weights.index
            )

            experiment.portfolio = weights

            # ---------------------------------------------------------------
            # 5. BACKTEST USING THE EXISTING CARL ENGINE
            # ---------------------------------------------------------------

            result = self.backtest_engine.run(
                experiment,
                cost_model=cost_model,
            )

            # ---------------------------------------------------------------
            # 6. ADD RESEARCH-PROVENANCE METADATA
            # ---------------------------------------------------------------

            result.metadata.update(
                {
                    "validation": "walk_forward",
                    "out_of_sample": True,
                    "fold": split.fold,
                    "train_start": split.train.index[0],
                    "train_end": split.train.index[-1],
                    "test_start": split.test.index[0],
                    "test_end": split.test.index[-1],
                    "model": type(model).__name__,
                    "strategy": strategy_name,
                }
            )

            fold_results.append(
                result
            )

        returns = pd.concat(
            [
                result.portfolio_returns
                for result in fold_results
            ]
        )

        return ModelEvaluationResult(
            returns=returns,
            folds=tuple(
                fold_results
            ),
        )