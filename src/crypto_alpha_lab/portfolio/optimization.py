"""
Portfolio optimization.

This module provides research-oriented wrappers around
portfolio optimization methods implemented in APRL.
"""

from __future__ import annotations

import pandas as pd

from asset_pricing_lab.portfolio import (
    global_minimum_variance_portfolio,
    maximum_sharpe_portfolio,
)

def global_minimum_variance_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> pd.Series:
    """
    Global minimum variance portfolio.
    """

    weights = global_minimum_variance_portfolio(
        expected_returns=expected_returns,
        covariance=covariance,
    )

    return pd.Series(
        weights,
        index=expected_returns.index,
        name="weight",
    )

def maximum_sharpe_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    Maximum Sharpe portfolio.
    """

    weights = maximum_sharpe_portfolio(
        expected_returns=expected_returns,
        covariance=covariance,
        risk_free_rate=risk_free_rate,
    )

    return pd.Series(
        weights,
        index=expected_returns.index,
        name="weight",
    )

def optimize_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    method: str = "global_minimum_variance",
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    Portfolio optimization dispatcher.
    """

    if method == "global_minimum_variance":
        return global_minimum_variance_weights(
            expected_returns,
            covariance,
        )

    if method == "maximum_sharpe":
        return maximum_sharpe_weights(
            expected_returns,
            covariance,
            risk_free_rate=risk_free_rate,
        )

    raise ValueError(
        f"Unknown optimization method '{method}'."
    )


