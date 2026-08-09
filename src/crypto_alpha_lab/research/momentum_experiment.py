"""
Cross-sectional momentum research experiment.

Provides a reproducible end-to-end CARL workflow:

prices
    -> walk-forward splits
    -> model fitting
    -> signal generation
    -> portfolio construction
    -> backtesting
    -> transaction costs
    -> out-of-sample performance
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crypto_alpha_lab.backtest.engine import (
    BacktestEngine,
    BacktestResult,
)
from crypto_alpha_lab.models.momentum import (
    CrossSectionalMomentumModel,
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
from crypto_alpha_lab.validation.walk_forward import (
    WalkForwardSplit,
    WalkForwardValidator,
)


@dataclass(frozen=True, slots=True)
class MomentumExperimentResult:
    """
    Complete result of a walk-forward momentum experiment.
    """

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
    Build a CARL ResearchExperiment using the current
    ResearchExperiment public data attributes.
    """

    dataset = ResearchDataset(
        prices=prices,
    )

    experiment = ResearchExperiment(
        dataset=dataset,
    )

    experiment.price_columns = list(
        prices.columns
    )

    experiment.asset_universe = list(
        weights.index
    )

    experiment.portfolio = weights

    return experiment


def run_momentum_experiment(
    prices: pd.DataFrame,
    train_size: int,
    test_size: int,
    lookback: int = 20,
    expanding: bool = True,
    cost_model=None,
    strategy_name: str = (
        "Cross-Sectional Momentum"
    ),
) -> MomentumExperimentResult:
    """
    Run a complete walk-forward cross-sectional
    momentum experiment.

    Parameters
    ----------
    prices
        Historical asset prices.

    train_size
        Initial number of observations in each
        training window.

    test_size
        Number of observations in each test window.

    lookback
        Momentum lookback period.

    expanding
        Whether the training window expands over time.

    cost_model
        Optional transaction-cost model.

    strategy_name
        Research strategy name.

    Returns
    -------
    MomentumExperimentResult
        Complete out-of-sample research result.

    Notes
    -----
    The model is fitted exclusively on each training
    window.

    The last training price is retained solely as the
    return anchor needed to calculate the first test-period
    return. It is not passed to the model as test data.
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

    if prices.columns.empty:
        raise ValueError(
            "prices must contain at least one asset."
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

    if lookback <= 0:
        raise ValueError(
            "lookback must be positive."
        )

    if lookback >= train_size:
        raise ValueError(
            "lookback must be smaller than "
            "train_size."
        )

    if not strategy_name:
        raise ValueError(
            "strategy_name cannot be empty."
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
            "prices do not contain enough "
            "observations for the requested "
            "walk-forward configuration."
        )

    model = CrossSectionalMomentumModel(
        lookback=lookback,
    )

    engine = BacktestEngine()

    fold_results: list[
        BacktestResult
    ] = []

    for split in splits:

        # ---------------------------------------------------------------
        # 1. FIT MODEL ON TRAINING DATA ONLY
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

        # ---------------------------------------------------------------
        # 3. CONVERT SIGNALS INTO PORTFOLIO WEIGHTS
        # ---------------------------------------------------------------

        strategy = ResearchStrategy(
            name=strategy_name,
            signals=signals,
            portfolio_method=(
                "signal_weighted"
            ),
        )

        weights = strategy.weights()

        # ---------------------------------------------------------------
        # 4. INCLUDE THE LAST TRAINING PRICE ONLY AS
        #    THE RETURN CALCULATION ANCHOR
        # ---------------------------------------------------------------

        test_prices = pd.concat(
            [
                split.train.iloc[
                    [-1]
                ],
                split.test,
            ]
        )

        experiment = _build_experiment(
            prices=test_prices,
            weights=weights,
        )

        # ---------------------------------------------------------------
        # 5. BACKTEST USING THE EXISTING ENGINE
        # ---------------------------------------------------------------

        result = engine.run(
            experiment,
            cost_model=cost_model,
        )

        # ---------------------------------------------------------------
        # 6. RETAIN ONLY RETURNS BELONGING TO THE TEST WINDOW
        # ---------------------------------------------------------------

        oos_returns = result.portfolio_returns.reindex(
            split.test.index
        )

        if oos_returns.empty:
            raise ValueError(
                f"Fold {split.fold} produced "
                "no out-of-sample returns."
            )

        fold_result = BacktestResult(
            asset_returns=result.asset_returns.reindex(
                split.test.index
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
                "model": (
                    "CrossSectionalMomentumModel"
                ),
                "strategy": strategy_name,
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
                "lookback": lookback,
            },
        )

        fold_results.append(
            fold_result
        )

    # -------------------------------------------------------------------
    # 7. AGGREGATE PURE OOS RETURNS
    # -------------------------------------------------------------------

    returns = pd.concat(
        [
            fold.portfolio_returns
            for fold in fold_results
        ]
    )

    if returns.index.has_duplicates:
        raise ValueError(
            "Aggregated OOS returns contain "
            "duplicate dates."
        )

    returns = returns.sort_index()

    cumulative_returns = (
        (1.0 + returns)
        .cumprod()
        - 1.0
    )

    return MomentumExperimentResult(
        returns=returns,
        cumulative_returns=cumulative_returns,
        folds=tuple(
            fold_results
        ),
        parameters={
            "train_size": train_size,
            "test_size": test_size,
            "lookback": lookback,
            "expanding": expanding,
            "strategy_name": strategy_name,
            "cost_model": (
                type(cost_model).__name__
                if cost_model is not None
                else None
            ),
        },
        metadata={
            "experiment": (
                "cross_sectional_momentum"
            ),
            "validation": (
                "walk_forward"
            ),
            "out_of_sample": True,
            "fold_count": len(
                fold_results
            ),
        },
    )

