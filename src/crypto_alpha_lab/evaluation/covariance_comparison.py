"""
Comparison of covariance-estimation methodologies.

This module compares already-computed, strictly out-of-sample
covariance experiments. It deliberately performs no covariance
estimation itself.
"""

from __future__ import annotations
from asset_pricing_lab import returns
from crypto_alpha_lab.research import experiment
import numpy as np

from dataclasses import dataclass
import numbers
import pandas as pd

from crypto_alpha_lab.evaluation.performance import (
    PerformanceAnalyzer,
    PerformanceReport,
)

from crypto_alpha_lab.research.covariance_experiment import (
    CovarianceExperimentResult,
)


@dataclass(frozen=True, slots=True)
class CovarianceComparisonResult:
    """
    Research comparison of covariance methodologies.

    Attributes
    ----------
    summary
        Performance comparison table indexed by covariance method.

    returns
        Out-of-sample return series by covariance method.

    cumulative_returns
        Cumulative out-of-sample return series by covariance method.

    metadata
        Research-design metadata.
    """

    summary: pd.DataFrame
    returns: dict[str, pd.Series]
    cumulative_returns: dict[str, pd.Series]
    metadata: dict[str, object]

class CovarianceComparison:
    """
    Compare completed covariance experiments.

    The comparison layer consumes experiment results only.
    It never estimates covariance or accesses training data.
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

        if not isinstance(
            risk_free_rate,
            (int, float),
        ):
            raise TypeError(
                "risk_free_rate must be numeric."
            )

        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

    @staticmethod
    def _validate_experiments(
        experiments: list[
            CovarianceExperimentResult
        ],
    ) -> None:
        """
        Validate the experiment collection.
        """

        if not isinstance(experiments, list):
            raise TypeError(
        "experiments must be a list."
    )

        if not experiments:
            raise ValueError(
        "experiments cannot be empty."
    )

        for experiment in experiments:

            if not isinstance(
        experiment,
        CovarianceExperimentResult,
    ):
                raise TypeError(
            "experiments must contain only "
            "CovarianceExperimentResult instances."
        )

        methods = [
            experiment.method
            for experiment in experiments
]

        if len(methods) != len(set(methods)):
            raise ValueError(
        "duplicate experiment methods are not allowed."
    )

        for experiment in experiments:

            if experiment.metadata.get(
        "out_of_sample"
        ) is not True:
                raise ValueError(
            f"experiment '{experiment.method}' "
            "must be strictly out-of-sample."
        )

        returns = experiment.returns

        if not isinstance(
        returns,
        pd.Series,
    ):
            raise TypeError(
            f"experiment '{experiment.method}' "
            "returns must be a pandas Series."
        )

        if returns.empty:
            raise ValueError(
            f"experiment '{experiment.method}' "
            "has no returns."
        )

        if returns.index.has_duplicates:
            raise ValueError(
                f"experiment '{experiment.method}' "
                "returns contain duplicate dates."
            )

        if not returns.index.is_monotonic_increasing:
            raise ValueError(
                f"experiment '{experiment.method}' "
                "returns must be chronological."
            )

        values = returns.to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"experiment '{experiment.method}' "
                "returns must contain only finite values."
            )

    def compare(
        self,
        experiments: list[
            CovarianceExperimentResult
        ],
    ) -> CovarianceComparisonResult:
        """
        Compare completed covariance experiments.

        The comparison is strictly out-of-sample and produces
        exactly one summary row per covariance methodology.
        """

        self._validate_experiments(
            experiments
        )

        analyzer = PerformanceAnalyzer(
            periods_per_year=(
                self.periods_per_year
            ),
            risk_free_rate=(
                self.risk_free_rate
            ),
        )

        summary_rows: list[
            dict[str, object]
        ] = []

        returns: dict[
            str,
            pd.Series,
        ] = {}

        cumulative_returns: dict[
            str,
            pd.Series,
        ] = {}

        for experiment in experiments:

            method = experiment.method
            series = experiment.returns

            report = analyzer.analyze(
                series
            )

            turnover_values = [
                float(fold.turnover)
                for fold in experiment.folds
                if fold.turnover is not None
            ]

            if turnover_values:
                average_turnover = float(
                    np.mean(turnover_values)
                )
            else:
                average_turnover = 0.0

            calmar = float(
                report.calmar_ratio
            )

            if not np.isfinite(calmar):
                calmar = 0.0

            summary_rows.append(
                {
                    "method": method,
                    "total_return": (
                        report.total_return
                    ),
                    "annualized_return": (
                        report.annualized_return
                    ),
                    "annualized_volatility": (
                        report.annualized_volatility
                    ),
                    "sharpe_ratio": (
                        report.sharpe_ratio
                    ),
                    "sortino_ratio": (
                        report.sortino_ratio
                    ),
                    "maximum_drawdown": (
                        report.maximum_drawdown
                    ),
                    "calmar_ratio": calmar,
                    "hit_rate": (
                        report.hit_rate
                    ),
                    "observation_count": (
                        report.observation_count
                    ),
                    "average_turnover": (
                        average_turnover
                    ),
                }
            )

            returns[method] = (
                series.copy()
            )

            cumulative_returns[method] = (
                experiment.cumulative_returns.copy()
            )

        summary = pd.DataFrame(
            summary_rows
        ).set_index(
            "method"
        )

        summary.index.name = "method"

        return CovarianceComparisonResult(
            summary=summary,
            returns=returns,
            cumulative_returns=(
                cumulative_returns
            ),
            metadata={
                "comparison": (
                    "covariance_methods"
                ),
                "out_of_sample": True,
                "method_count": len(
                    experiments
                ),
                "methods": [
                    experiment.method
                    for experiment in experiments
                ],
                "periods_per_year": (
                    self.periods_per_year
                ),
                "risk_free_rate": (
                    self.risk_free_rate
                ),
            },
        )


def compare_covariance_methods(
    experiments: list[
        CovarianceExperimentResult
    ],
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> CovarianceComparisonResult:
    """
    Convenience function for covariance comparison.
    """

    comparison = CovarianceComparison(
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )

    return comparison.compare(
        experiments
    )