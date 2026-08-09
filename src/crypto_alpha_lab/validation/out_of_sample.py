"""
Out-of-sample research execution.

Executes a research strategy across chronological
walk-forward folds and collects strictly out-of-sample
portfolio returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crypto_alpha_lab.backtest.engine import (
    BacktestResult,
)
from crypto_alpha_lab.strategy.strategy import (
    ResearchStrategy,
)
from crypto_alpha_lab.validation.walk_forward import (
    WalkForwardSplit,
)


@dataclass(frozen=True, slots=True)
class OutOfSampleResult:
    """
    Aggregated out-of-sample research result.

    Parameters
    ----------
    returns
        Concatenated out-of-sample portfolio returns.

    folds
        Individual fold backtest results.
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


class OutOfSampleEvaluator:
    """
    Evaluate a strategy across walk-forward test folds.
    """

    def evaluate(
        self,
        strategy: ResearchStrategy,
        splits: list[WalkForwardSplit],
    ) -> OutOfSampleResult:
        """
        Evaluate a strategy on chronological test folds.

        Each fold is evaluated independently using only
        the observations contained in that fold's test
        dataset.

        Parameters
        ----------
        strategy
            Research strategy.

        splits
            Walk-forward train/test splits.

        Returns
        -------
        OutOfSampleResult
            Aggregated out-of-sample results.
        """

        if not isinstance(
            strategy,
            ResearchStrategy,
        ):
            raise TypeError(
                "strategy must be a ResearchStrategy."
            )

        if not splits:
            raise ValueError(
                "splits cannot be empty."
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
            if split.test.empty:
                raise ValueError(
        "walk-forward test fold cannot be empty."
    )

            # The strategy produces the portfolio
            # using the information available to it.
            weights = strategy.weights()

            # The actual BacktestResult for each fold
            # will be produced by the CARL backtest
            # execution layer in the next integration
            # step.
            #
            # For now, the evaluator operates on the
            # test return matrix directly.
            test_returns = split.test

            portfolio_returns = (
                test_returns[
                    weights.index
                ]
                @ weights
            )

            fold_result = BacktestResult(
                asset_returns=test_returns[
                    weights.index
                ],
                portfolio_returns=portfolio_returns,
                cumulative_returns=(
                    (1.0 + portfolio_returns)
                    .cumprod()
                    - 1.0
                ),
                weights=weights,
                turnover=0.0,
                transaction_costs=0.0,
                metadata={
                    "validation": (
                        "walk_forward"
                    ),
                    "fold": split.fold,
                    "out_of_sample": True,
                },
            )

            fold_results.append(
                fold_result
            )

        returns = pd.concat(
            [
                result.portfolio_returns
                for result in fold_results
            ]
        )

        return OutOfSampleResult(
            returns=returns,
            folds=tuple(
                fold_results
            ),
        )