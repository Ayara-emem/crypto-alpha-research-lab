"""
Institutional research backtesting engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from asset_pricing_lab.returns import (
    arithmetic_returns,
)

from crypto_alpha_lab.research.experiment import (
    ResearchExperiment,
)


@dataclass(slots=True)
class BacktestResult:
    """
    Output of a completed research backtest.
    """

    asset_returns: pd.DataFrame

    portfolio_returns: pd.Series

    cumulative_returns: pd.Series

    weights: pd.Series

    turnover: float

    transaction_costs: float

    metadata: dict = field(
        default_factory=dict,
    )


class BacktestEngine:
    """
    Institutional research backtesting engine.
    """

    def run(
        self,
        experiment: ResearchExperiment,
        cost_model=None,
    ) -> BacktestResult:
        """
        Execute a backtest for a research experiment.

        Parameters
        ----------
        experiment
            Configured research experiment.

        cost_model
            Optional transaction cost model.

        Returns
        -------
        BacktestResult
            Completed backtest result.
        """

        self._validate_experiment(
            experiment,
        )

        asset_returns = (
            self._prepare_asset_returns(
                experiment,
            )
        )

        weights = self._prepare_weights(
            experiment,
            asset_returns,
        )

        portfolio_returns = (
            asset_returns[
                weights.index
            ]
            @ weights
        )

        turnover = self._compute_turnover(
            weights,
        )

        (
            portfolio_returns,
            transaction_costs,
        ) = self._apply_transaction_costs(
            portfolio_returns=portfolio_returns,
            weights=weights,
            turnover=turnover,
            cost_model=cost_model,
        )

        cumulative = (
    (1.0 + portfolio_returns)
    .cumprod()
    - 1.0
)

        metadata = self._default_metadata()

        return self._build_result(
            asset_returns=asset_returns,
            portfolio_returns=portfolio_returns,
            cumulative_returns=cumulative,
            weights=weights,
            turnover=turnover,
            transaction_costs=transaction_costs,
            metadata=metadata,
        )

    def _validate_experiment(
        self,
        experiment: ResearchExperiment,
    ) -> None:
        """
        Validate experiment inputs.
        """

        if not isinstance(
            experiment,
            ResearchExperiment,
        ):
            raise TypeError(
                "experiment must be a ResearchExperiment."
            )

        if experiment.dataset is None:
            raise ValueError(
                "Experiment contains no dataset."
            )

        if experiment.portfolio is None:
            raise ValueError(
                "Experiment contains no portfolio."
            )

        prices = experiment.dataset.prices

        if prices.empty:
            raise ValueError(
                "Dataset contains no price data."
            )

        if experiment.price_columns is not None:

            missing = (
                set(experiment.price_columns)
                - set(prices.columns)
            )

            if missing:
                raise ValueError(
                    "Unknown price columns: "
                    f"{sorted(missing)}"
                )

        if experiment.asset_universe is not None:

            if len(
                experiment.asset_universe
            ) == 0:
                raise ValueError(
                    "asset_universe cannot be empty."
                )
    def _prepare_asset_returns(
        self,
        experiment: ResearchExperiment,
    ) -> pd.DataFrame:
        """
        Compute arithmetic returns from configured
        price columns.

        APRL's arithmetic_returns() operates on a
        one-dimensional price series, so CARL applies
        the APRL implementation independently to each
        asset column.
        """

        prices = experiment.dataset.prices

        if prices.empty:
            raise ValueError(
                "Dataset contains no price data."
            )

        if experiment.price_columns is None:

            if "Close" in prices.columns:

                price_columns = [
                    "Close",
                ]

            else:

                price_columns = list(
                    prices.columns
                )

        else:

            price_columns = (
                experiment.price_columns
            )

        if len(price_columns) == 0:
            raise ValueError(
                "No price columns configured."
            )

        missing = (
            set(price_columns)
            - set(prices.columns)
        )

        if missing:
            raise ValueError(
                "Unknown price columns: "
                f"{sorted(missing)}"
            )

        returns = {}

        for column in price_columns:

            returns[column] = arithmetic_returns(
                prices[column],
            )

        asset_returns = pd.DataFrame(
            returns,
        )

        if asset_returns.empty:
            raise ValueError(
                "No asset returns were generated."
            )

        return asset_returns


    def _prepare_weights(
        self,
        experiment: ResearchExperiment,
        asset_returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Align portfolio weights with the asset universe.
        """

        if experiment.portfolio is None:
            raise ValueError(
                "Experiment contains no portfolio."
            )

        if experiment.asset_universe is None:

            asset_universe = list(
                asset_returns.columns
            )

        else:

            asset_universe = (
                experiment.asset_universe
            )

        missing_assets = (
            set(asset_universe)
            - set(asset_returns.columns)
        )

        if missing_assets:
            raise ValueError(
                "Unknown assets in asset_universe: "
                f"{sorted(missing_assets)}"
            )

        weights = experiment.portfolio.reindex(
            asset_universe,
            fill_value=0.0,
        )

        if weights.isna().any():
            raise ValueError(
                "Portfolio contains missing weights."
            )

        return weights.astype(float)

    def _compute_turnover(
        self,
        current_weights: pd.Series,
        previous_weights: pd.Series | None = None,
    ) -> float:
        """
        Compute portfolio turnover.

        For an initial portfolio, turnover is the
        gross amount traded into the portfolio.

        For subsequent rebalances, turnover is the
        absolute change in portfolio weights.
        """

        if previous_weights is None:

            return float(
                current_weights.abs().sum()
            )

        previous_weights = (
            previous_weights.reindex(
                current_weights.index,
                fill_value=0.0,
            )
        )

        return float(
            (
                current_weights
                - previous_weights
            )
            .abs()
            .sum()
        )

    def _apply_transaction_costs(
        self,
        portfolio_returns: pd.Series,
        weights: pd.Series,
        turnover: float,
        cost_model=None,
    ) -> tuple[pd.Series, float]:
        """
        Calculate transaction costs.

        The current v1.0 engine records transaction costs
        separately rather than subtracting a one-time
        cost from every return observation.
        """

        del weights

        transaction_cost = 0.0

        if cost_model is None:

            return (
                portfolio_returns,
                transaction_cost,
            )

        transaction_cost = float(
            cost_model.cost(
                turnover=turnover,
            )
        )

        return (
            portfolio_returns,
            transaction_cost,
        )

    def _build_result(
        self,
        asset_returns: pd.DataFrame,
        portfolio_returns: pd.Series,
        cumulative_returns: pd.Series,
        weights: pd.Series,
        turnover: float,
        transaction_costs: float,
        metadata: dict | None = None,
    ) -> BacktestResult:
        """
        Construct a BacktestResult.
        """

        if metadata is None:
            metadata = {}

        return BacktestResult(
            asset_returns=asset_returns,
            portfolio_returns=portfolio_returns,
            cumulative_returns=cumulative_returns,
            weights=weights,
            turnover=float(turnover),
            transaction_costs=float(
                transaction_costs,
            ),
            metadata=dict(metadata),
        )

    def _default_metadata(
        self,
    ) -> dict:
        """
        Create default backtest metadata.
        """

        return {
            "engine": "CARL",
            "engine_component": "backtest",
            "version": "1.0",
        }