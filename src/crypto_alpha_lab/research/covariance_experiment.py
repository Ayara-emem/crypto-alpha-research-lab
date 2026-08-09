"""
Walk-forward covariance estimation research experiment.

Compares alternative covariance estimators through a
global minimum-variance portfolio constructed using
training data only and evaluated strictly out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from crypto_alpha_lab.dataset import (
    ResearchDataset,
)

from crypto_alpha_lab.backtest.engine import (
    BacktestEngine,
    BacktestResult,
)
from crypto_alpha_lab.portfolio.covariance import (
    CovarianceEstimator,
)
from crypto_alpha_lab.research.experiment import (
    ResearchExperiment,
)
from crypto_alpha_lab.strategy.strategy import (
    ResearchStrategy,
)
from crypto_alpha_lab.validation.walk_forward import (
    WalkForwardValidator,
)


CovarianceMethod = Literal[
    "sample",
    "shrinkage",
    "ledoit_wolf",
]


@dataclass(frozen=True, slots=True)
class CovarianceExperimentResult:
    """
    Complete result for one covariance methodology.
    """

    method: str

    returns: pd.Series

    cumulative_returns: pd.Series

    folds: tuple[BacktestResult, ...]

    parameters: dict[str, object]

    metadata: dict[str, object]


def _build_experiment(
    prices: pd.DataFrame,
    weights: pd.Series,
) -> ResearchExperiment:
    """
    Build a valid CARL ResearchExperiment.
    """

    dataset = ResearchDataset(
        prices=prices,
    )

    return ResearchExperiment(
        dataset=dataset,
        portfolio=weights,
    )

def run_covariance_experiment(
    prices: pd.DataFrame,
    method: CovarianceMethod,
    train_size: int,
    test_size: int,
    shrinkage: float | None = None,
    cost_model=None,
    expanding: bool = True,
) -> CovarianceExperimentResult:
    """
    Run a walk-forward covariance estimation experiment.

    Covariance is estimated using training data only.
    Portfolio weights are constructed from that covariance
    estimate and then evaluated on the subsequent test window.

    Parameters
    ----------
    prices
        Historical asset prices.

    method
        Covariance methodology.

    train_size
        Number of observations in the initial training window.

    test_size
        Number of observations in each test window.

    shrinkage
        Explicit shrinkage intensity when method='shrinkage'.

    cost_model
        Optional transaction-cost model.

    expanding
        Whether training windows expand over time.
    """

    if not isinstance(
        prices,
        pd.DataFrame,
    ):
        raise TypeError(
            "prices must be a pandas DataFrame."
        )

    if prices.empty:
        raise ValueError(
            "prices cannot be empty."
        )

    if not prices.index.is_monotonic_increasing:
        raise ValueError(
            "prices index must be chronological."
        )

    if prices.index.has_duplicates:
        raise ValueError(
            "prices index cannot contain duplicates."
        )

    if train_size <= 0:
        raise ValueError(
            "train_size must be positive."
        )

    if test_size <= 0:
        raise ValueError(
            "test_size must be positive."
        )

    supported_methods = {
        "sample",
        "shrinkage",
        "ledoit_wolf",
    }

    if method not in supported_methods:
        raise ValueError(
            f"Unsupported covariance method: {method!r}."
        )

    if method == "shrinkage":
        if shrinkage is None:
            raise ValueError(
                "shrinkage must be supplied for "
                "method='shrinkage'."
            )

    validator = WalkForwardValidator(
        train_size=train_size,
        test_size=test_size,
        expanding=expanding,
    )

    splits = validator.split(
        prices,
    )

    if not splits:
        raise ValueError(
            "No valid walk-forward splits were produced."
        )

    engine = BacktestEngine()

    fold_results: list[
        BacktestResult
    ] = []

    for split in splits:

        # ---------------------------------------------------------------
        # 1. TRAINING RETURNS ONLY
        # ---------------------------------------------------------------

        train_returns = (
            split.train
            .pct_change()
            .dropna()
        )

        if train_returns.empty:
            raise ValueError(
                f"Fold {split.fold} contains insufficient "
                "training observations to estimate covariance."
            )

        # ---------------------------------------------------------------
        # 2. ESTIMATE COVARIANCE FROM TRAINING DATA ONLY
        # ---------------------------------------------------------------

        estimator = CovarianceEstimator(
            method=method,
            shrinkage=shrinkage,
        )

        covariance_estimate = estimator.fit(
            train_returns,
        )

        # ---------------------------------------------------------------
        # 3. CONSTRUCT GMV PORTFOLIO
        # ---------------------------------------------------------------

        strategy = ResearchStrategy(
            name=(
                f"GMV - {method}"
            ),
            signals=pd.Series(
                1.0,
                index=covariance_estimate.matrix.index,
            ),
            portfolio_method=(
                "global_minimum_variance"
            ),
            covariance=(
                covariance_estimate.matrix
            ),
        )

        weights = strategy.weights()

        # ---------------------------------------------------------------
        # 4. RETAIN THE LAST TRAINING PRICE AS
        #    THE RETURN ANCHOR
        # ---------------------------------------------------------------

        test_prices = pd.concat(
            [
                split.train.iloc[[-1]],
                split.test,
            ]
        )

        experiment = _build_experiment(
            prices=test_prices,
            weights=weights,
        )

        # ---------------------------------------------------------------
        # 5. OUT-OF-SAMPLE BACKTEST
        # ---------------------------------------------------------------

        result = engine.run(
            experiment,
            cost_model=cost_model,
        )

        oos_returns = (
            result.portfolio_returns
            .reindex(
                split.test.index
            )
        )

        if oos_returns.empty:
            raise ValueError(
                f"Fold {split.fold} produced "
                "no OOS returns."
            )

        fold_result = BacktestResult(
            asset_returns=(
                result.asset_returns
                .reindex(
                    split.test.index
                )
            ),
            portfolio_returns=oos_returns,
            cumulative_returns=(
                (1.0 + oos_returns)
                .cumprod()
                - 1.0
            ),
            weights=result.weights,
            turnover=result.turnover,
            transaction_costs=result.transaction_costs,
            metadata={
                **result.metadata,
                "validation": (
                    "walk_forward"
                ),
                "out_of_sample": True,
                "fold": split.fold,
                "covariance_method": method,
                "shrinkage": shrinkage,
                "portfolio_method": (
                    "global_minimum_variance"
                ),
                "train_start": (
                    split.train.index[0]
                ),
                "train_end": (
                    split.train.index[-1]
                ),
                "test_start": (
                    split.test.index[0]
                ),
                "test_end": (
                    split.test.index[-1]
                ),
            },
        )

        fold_results.append(
            fold_result
        )

    # -------------------------------------------------------------------
    # 6. AGGREGATE OOS RETURNS
    # -------------------------------------------------------------------

    returns = pd.concat(
        [
            fold.portfolio_returns
            for fold in fold_results
        ]
    ).sort_index()

    if returns.index.has_duplicates:
        raise ValueError(
            "Aggregated OOS returns contain duplicates."
        )

    cumulative_returns = (
        (1.0 + returns)
        .cumprod()
        - 1.0
    )

    return CovarianceExperimentResult(
        method=method,
        returns=returns,
        cumulative_returns=cumulative_returns,
        folds=tuple(
            fold_results
        ),
        parameters={
            "method": method,
            "train_size": train_size,
            "test_size": test_size,
            "shrinkage": shrinkage,
            "expanding": expanding,
        },
        metadata={
            "experiment": (
                "covariance_comparison"
            ),
            "validation": (
                "walk_forward"
            ),
            "out_of_sample": True,
            "portfolio_method": (
                "global_minimum_variance"
            ),
            "fold_count": len(
                fold_results
            ),
        },
    )