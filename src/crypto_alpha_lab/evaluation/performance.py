"""
Institutional research performance analytics.

CARL owns research orchestration and reporting.
APRL owns the underlying quantitative-finance primitives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from asset_pricing_lab.risk import (
    calmar_ratio as aprl_calmar_ratio,
    sortino_ratio as aprl_sortino_ratio,
)
from asset_pricing_lab.returns import (
    annualized_return as aprl_annualized_return,
    annualized_volatility as aprl_annualized_volatility,
)


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    """
    Performance and risk summary for a strategy return series.
    """

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    maximum_drawdown: float
    calmar_ratio: float
    hit_rate: float
    observation_count: int
    sortino_ratio: float


class PerformanceAnalyzer:
    """
    Analyze strategy returns using CARL orchestration
    and APRL quantitative primitives.
    """

    def __init__(
        self,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ) -> None:
        if periods_per_year <= 0:
            raise ValueError(
                "periods_per_year must be positive."
            )

        if not np.isfinite(risk_free_rate):
            raise ValueError(
                "risk_free_rate must be finite."
            )

        self.periods_per_year = int(
            periods_per_year
        )

        self.risk_free_rate = float(
            risk_free_rate
        )

    @staticmethod
    def _validate_returns(
        returns: pd.Series,
    ) -> pd.Series:
        """
        Validate a strategy return series.
        """

        if not isinstance(
            returns,
            pd.Series,
        ):
            raise TypeError(
                "returns must be a pandas Series."
            )

        if returns.empty:
            raise ValueError(
                "returns cannot be empty."
            )

        if returns.index.has_duplicates:
            raise ValueError(
                "returns index cannot contain duplicates."
            )

        values = returns.to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "returns must contain only finite values."
            )

        return returns.astype(float)

    def total_return(
        self,
        returns: pd.Series,
    ) -> float:
        """
        Compute total compounded return.
        """

        returns = self._validate_returns(
            returns
        )

        return float(
            (1.0 + returns).prod() - 1.0
        )

    def annualized_return(
        self,
        returns: pd.Series,
    ) -> float:
        """
        Compute annualized return through APRL.
        """

        returns = self._validate_returns(
            returns
        )

        cumulative = self.total_return(
            returns
        )

        return float(
            aprl_annualized_return(
                cumulative_return=cumulative,
                periods=len(returns),
                periods_per_year=(
                    self.periods_per_year
                ),
            )
        )

    def annualized_volatility(
        self,
        returns: pd.Series,
    ) -> float:
        """
        Compute annualized volatility through APRL.
        """

        returns = self._validate_returns(
            returns
        )

        return float(
            aprl_annualized_volatility(
                returns.to_numpy(
                    dtype=float
                ),
                periods_per_year=(
                    self.periods_per_year
                ),
            )
        )

    def cumulative_returns(
        self,
        returns: pd.Series,
    ) -> pd.Series:
        """
        Compute cumulative compounded returns.
        """

        returns = self._validate_returns(
            returns
        )

        return (
            (1.0 + returns)
            .cumprod()
            - 1.0
        )

    def drawdown_series(
        self,
        returns: pd.Series,
    ) -> pd.Series:
        """
        Compute running drawdown.
        """

        returns = self._validate_returns(
            returns
        )

        wealth = (
            1.0 + returns
        ).cumprod()

        running_peak = wealth.cummax()

        return (
            wealth / running_peak
            - 1.0
        )

    def maximum_drawdown(
        self,
        returns: pd.Series,
    ) -> float:
        """
        Compute maximum peak-to-trough drawdown.
        """

        return float(
            self.drawdown_series(
                returns
            ).min()
        )

    def sharpe_ratio(
        self,
        returns: pd.Series,
    ) -> float:
        """
        Compute annualized Sharpe ratio.

        The risk-free rate is interpreted as an
        annual rate.
        """

        returns = self._validate_returns(
            returns
        )

        periodic_rf = (
            self.risk_free_rate
            / self.periods_per_year
        )

        excess = returns - periodic_rf

        volatility = float(
            excess.std(ddof=1)
        )

        if volatility == 0.0:
            return 0.0

        return float(
            excess.mean()
            / volatility
            * np.sqrt(
                self.periods_per_year
            )
        )

    def sortino_ratio(
        self,
        returns: pd.Series,
    ) -> float:
        """
        Compute the annualized Sortino ratio through APRL.

        If the downside deviation is zero, the Sortino
        ratio is mathematically undefined. CARL represents
        that condition as NaN rather than failing the
        complete research report.

        Other ValueError exceptions are propagated so that
        invalid inputs are not silently suppressed.
        """

        returns = self._validate_returns(
            returns
            )
        
        try:
            return float(
                aprl_sortino_ratio(
                    returns.to_numpy(
                        dtype=float
                        ),
                        risk_free_rate=(
                            self.risk_free_rate
                            ),
                            )
                            )
        except ValueError as exc:
            if (
                "undefined when downside deviation is zero"
                not in str(exc).lower()
                ):
                raise
            return float("nan")

    def calmar_ratio(
        self,
        returns: pd.Series,
    ) -> float:
            """
            Compute Calmar ratio through APRL.

            CARL converts the return series into a cumulative
            wealth index before delegating to APRL.

            A zero maximum drawdown makes the Calmar ratio
            undefined. CARL represents that undefined
            reporting value as zero.
            """

            returns = self._validate_returns(
                returns
            )

            wealth = (
                1.0 + returns
            ).cumprod()

            try:
                return float(
                    aprl_calmar_ratio(
                        prices=wealth,
                        periods=len(returns),
                        periods_per_year=(
                            self.periods_per_year
                        ),
                    )
                )
            except ValueError as exc:
                if (
                    "maximum drawdown is zero"
                    not in str(exc).lower()
                ):
                    raise

                return 0.0

    def hit_rate(
        self,
        returns: pd.Series,
    ) -> float:
        """
        Compute the fraction of observations
        with positive returns.
        """

        returns = self._validate_returns(
            returns
        )

        return float(
            (returns > 0.0).mean()
        )

    def analyze(
        self,
        returns: pd.Series,
    ) -> PerformanceReport:
            """
            Produce a complete performance report.
            """
            returns = self._validate_returns(
            returns
        )

            return PerformanceReport(
            total_return=self.total_return(
                returns
            ),
            annualized_return=(
                self.annualized_return(
                    returns
                )
            ),
            annualized_volatility=(
                self.annualized_volatility(
                    returns
                )
            ),
            sharpe_ratio=self.sharpe_ratio(
                returns
            ),
            maximum_drawdown=(
                self.maximum_drawdown(
                    returns
                )
            ),
            calmar_ratio=self.calmar_ratio(
                returns
            ),
            hit_rate=self.hit_rate(
                returns
            ),
            observation_count=len(
                returns
            ),
            sortino_ratio=self.sortino_ratio(
                returns
            ),
        )