"""
Backtest reporting.

Provides structured reporting and presentation utilities
for completed CARL backtests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from crypto_alpha_lab.backtest.engine import (
    BacktestResult,
)

from crypto_alpha_lab.backtest.performance import (
    performance_summary,
)


@dataclass(slots=True)
class BacktestReport:
    """
    Structured report generated from a completed backtest.

    Parameters
    ----------
    summary
        Portfolio performance metrics.

    portfolio_returns
        Portfolio return series.

    cumulative_returns
        Cumulative portfolio returns.

    weights
        Portfolio weights.

    metadata
        Additional report metadata.
    """

    summary: pd.Series

    portfolio_returns: pd.Series

    cumulative_returns: pd.Series

    weights: pd.Series

    metadata: dict = field(
        default_factory=dict,
    )

    def to_frame(
        self,
    ) -> pd.DataFrame:
        """
        Convert the performance summary to a DataFrame.

        Returns
        -------
        pandas.DataFrame
            Performance metrics in tabular form.
        """

        return self.summary.to_frame(
            name="value",
        )

    def to_dict(
        self,
    ) -> dict:
        """
        Convert the report summary to a dictionary.

        Returns
        -------
        dict
            Performance metrics and metadata.
        """

        return {
            "summary": self.summary.to_dict(),
            "metadata": dict(self.metadata),
        }


def build_backtest_report(
    result: BacktestResult,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    metadata: dict | None = None,
) -> BacktestReport:
    """
    Build a structured report from a backtest result.

    Parameters
    ----------
    result
        Completed backtest result.

    risk_free_rate
        Annualized risk-free rate.

    periods_per_year
        Number of return observations per year.

    metadata
        Optional additional report metadata.

    Returns
    -------
    BacktestReport
        Structured backtest report.
    """

    summary = performance_summary(
        result,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    report_metadata = {}

    if metadata is not None:
        report_metadata.update(
            metadata,
        )

    report_metadata.setdefault(
        "periods_per_year",
        periods_per_year,
    )

    report_metadata.setdefault(
        "risk_free_rate",
        risk_free_rate,
    )

    return BacktestReport(
        summary=summary,
        portfolio_returns=result.portfolio_returns.copy(),
        cumulative_returns=result.cumulative_returns.copy(),
        weights=result.weights.copy(),
        metadata=report_metadata,
    )


def report_summary(
    report: BacktestReport,
) -> pd.Series:
    """
    Return the report performance summary.

    Parameters
    ----------
    report
        Backtest report.

    Returns
    -------
    pandas.Series
        Performance metrics.
    """

    return report.summary.copy()


def report_table(
    report: BacktestReport,
) -> pd.DataFrame:
    """
    Return the report as a tabular object.

    Parameters
    ----------
    report
        Backtest report.

    Returns
    -------
    pandas.DataFrame
        Performance metrics.
    """

    return report.to_frame()