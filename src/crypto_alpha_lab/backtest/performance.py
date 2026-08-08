"""
Backtest performance analytics.

This module evaluates completed CARL backtests while
delegating quantitative calculations to APRL.
"""

from __future__ import annotations

import pandas as pd

from asset_pricing_lab.returns import (
    annualized_return,
    annualized_volatility,
)

from asset_pricing_lab.risk import (
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
)

from crypto_alpha_lab.backtest.engine import (
    BacktestResult,
)

def annualized_return_from_backtest(
    result: BacktestResult,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized portfolio return.

    Parameters
    ----------
    result
        Completed backtest result.

    periods_per_year
        Number of return observations per year.

    Returns
    -------
    float
        Annualized portfolio return.
    """

    return float(
        annualized_return(
            result.portfolio_returns,
            periods_per_year=periods_per_year,
        )
    )

def annualized_return_from_backtest(
    result: BacktestResult,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized portfolio return.

    Parameters
    ----------
    result
        Completed backtest result.

    periods_per_year
        Number of return observations per year.

    Returns
    -------
    float
        Annualized portfolio return.
    """

    return float(
        annualized_return(
            result.portfolio_returns,
            periods_per_year=periods_per_year,
        )
    )

def annualized_volatility_from_backtest(
    result: BacktestResult,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized portfolio volatility.
    """

    return float(
        annualized_volatility(
            result.portfolio_returns,
            periods_per_year=periods_per_year,
        )
    )


def sharpe_ratio_from_backtest(
    result: BacktestResult,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute the portfolio Sharpe ratio.
    """

    return float(
        sharpe_ratio(
            result.portfolio_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
    )

def sortino_ratio_from_backtest(
    result: BacktestResult,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute the portfolio Sortino ratio.
    """

    return float(
        sortino_ratio(
            result.portfolio_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
    )


def calmar_ratio_from_backtest(
    result: BacktestResult,
    periods_per_year: int = 252,
) -> float:
    """
    Compute the portfolio Calmar ratio.
    """

    return float(
        calmar_ratio(
            result.portfolio_returns,
            periods_per_year=periods_per_year,
        )
    )

def performance_summary(
    result: BacktestResult,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Generate a performance summary for a backtest.

    Returns
    -------
    pandas.Series
        Standardized performance metrics.
    """

    return pd.Series(
        {
            "annualized_return": (
                annualized_return_from_backtest(
                    result,
                    periods_per_year=periods_per_year,
                )
            ),
            "annualized_volatility": (
                annualized_volatility_from_backtest(
                    result,
                    periods_per_year=periods_per_year,
                )
            ),
            "sharpe_ratio": (
                sharpe_ratio_from_backtest(
                    result,
                    risk_free_rate=risk_free_rate,
                    periods_per_year=periods_per_year,
                )
            ),
            "sortino_ratio": (
                sortino_ratio_from_backtest(
                    result,
                    risk_free_rate=risk_free_rate,
                    periods_per_year=periods_per_year,
                )
            ),
            "calmar_ratio": (
                calmar_ratio_from_backtest(
                    result,
                    periods_per_year=periods_per_year,
                )
            ),
            "turnover": result.turnover,
            "transaction_costs": (
                result.transaction_costs
            ),
        },
        dtype=float,
    )

