"""
Strategy and model comparison utilities.

Provides lightweight comparison of completed CARL
backtest results.
"""

from __future__ import annotations

import pandas as pd

from crypto_alpha_lab.backtest.engine import (
    BacktestResult,
)


def compare_backtests(
    results: dict[str, BacktestResult],
) -> pd.DataFrame:
    """
    Compare multiple backtest results.

    Parameters
    ----------
    results
        Mapping from strategy name to BacktestResult.

    Returns
    -------
    pandas.DataFrame
        Comparison table containing core backtest
        statistics.
    """

    if not isinstance(
        results,
        dict,
    ):
        raise TypeError(
            "results must be a dictionary."
        )

    if not results:
        raise ValueError(
            "results cannot be empty."
        )

    rows = {}

    for name, result in results.items():

        if not isinstance(
            result,
            BacktestResult,
        ):
            raise TypeError(
                "Every result must be a BacktestResult."
            )

        returns = result.portfolio_returns

        cumulative = result.cumulative_returns

        if returns.empty:
            raise ValueError(
                f"Backtest '{name}' has no returns."
            )

        rows[name] = {
            "total_return": float(
                cumulative.iloc[-1]
            ),
            "mean_return": float(
                returns.mean()
            ),
            "volatility": float(
                returns.std()
            ),
            "turnover": float(
                result.turnover
            ),
            "transaction_costs": float(
                result.transaction_costs
            ),
        }

    return pd.DataFrame.from_dict(
        rows,
        orient="index",
    )


