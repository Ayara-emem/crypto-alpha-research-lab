"""
Robustness analysis for covariance-based portfolio research.

This module evaluates whether conclusions from covariance experiments
remain stable across reasonable experimental configurations.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crypto_alpha_lab.research.covariance_experiment import (
    CovarianceExperimentResult,
    run_covariance_experiment,
)


@dataclass(frozen=True, slots=True)
class RobustnessConfiguration:
    """Configuration for one covariance experiment."""

    train_size: int
    test_size: int
    shrinkage: float | None = None

    def __post_init__(self) -> None:
        if self.train_size <= 0:
            raise ValueError(
                "train_size must be positive."
            )

        if self.test_size <= 0:
            raise ValueError(
                "test_size must be positive."
            )

        if self.shrinkage is not None:
            if not 0.0 <= self.shrinkage <= 1.0:
                raise ValueError(
                    "shrinkage must lie in [0, 1]."
                )


@dataclass(frozen=True, slots=True)
class CovarianceRobustnessResult:
    """
    Results from a covariance robustness analysis.
    """

    experiments: tuple[
        CovarianceExperimentResult,
        ...
    ]

    summary: pd.DataFrame

    metadata: dict[str, object]


class CovarianceRobustnessAnalyzer:
    """
    Run covariance experiments across multiple configurations.

    The analyzer deliberately delegates all actual research
    computation to the existing covariance experiment layer.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
    ) -> None:

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

        if prices.index.has_duplicates:
            raise ValueError(
                "prices cannot contain duplicate dates."
            )

        if not prices.index.is_monotonic_increasing:
            raise ValueError(
                "prices must be chronological."
            )

        self.prices = prices.copy()

    @staticmethod
    def _validate_methods(
        methods: list[str],
    ) -> None:

        if not isinstance(
            methods,
            list,
        ):
            raise TypeError(
                "methods must be a list."
            )

        if not methods:
            raise ValueError(
                "methods cannot be empty."
            )

        supported = {
            "sample",
            "shrinkage",
            "ledoit_wolf",
        }

        unknown = set(methods) - supported

        if unknown:
            raise ValueError(
                f"Unsupported covariance methods: "
                f"{sorted(unknown)}"
            )

        if len(methods) != len(set(methods)):
            raise ValueError(
                "methods cannot contain duplicates."
            )

    def run(
        self,
        configurations: list[
            RobustnessConfiguration
        ],
        methods: list[str] | None = None,
    ) -> CovarianceRobustnessResult:

        if not isinstance(
            configurations,
            list,
        ):
            raise TypeError(
                "configurations must be a list."
            )

        if not configurations:
            raise ValueError(
                "configurations cannot be empty."
            )

        for configuration in configurations:
            if not isinstance(
                configuration,
                RobustnessConfiguration,
            ):
                raise TypeError(
                    "all configurations must be "
                    "RobustnessConfiguration instances."
                )

        if methods is None:
            methods = [
                "sample",
                "shrinkage",
                "ledoit_wolf",
            ]

        self._validate_methods(methods)

        experiments: list[
            CovarianceExperimentResult
        ] = []

        rows: list[dict[str, object]] = []

        for configuration in configurations:

            for method in methods:

                kwargs: dict[str, object] = {
                    "prices": self.prices,
                    "train_size": (
                        configuration.train_size
                    ),
                    "test_size": (
                        configuration.test_size
                    ),
                    "method": method,
                }

                if method == "shrinkage":
                    if configuration.shrinkage is None:
                        raise ValueError(
                            "shrinkage must be supplied "
                            "for method='shrinkage'."
                        )

                    kwargs["shrinkage"] = (
                        configuration.shrinkage
                    )

                experiment = (
                    run_covariance_experiment(
                        **kwargs
                    )
                )

                experiments.append(
                    experiment
                )

                rows.append(
                    {
                        "method": method,
                        "train_size": (
                            configuration.train_size
                        ),
                        "test_size": (
                            configuration.test_size
                        ),
                        "shrinkage": (
                            configuration.shrinkage
                            if method == "shrinkage"
                            else None
                        ),
                        "total_return": (
                            experiment.returns
                            .add(1.0)
                            .prod()
                            - 1.0
                        ),
                        "observation_count": (
                            len(experiment.returns)
                        ),
                        "fold_count": (
                            len(experiment.folds)
                        ),
                    }
                )

        summary = pd.DataFrame(
            rows
        )

        return CovarianceRobustnessResult(
            experiments=tuple(
                experiments
            ),
            summary=summary,
            metadata={
                "analysis": (
                    "covariance_robustness"
                ),
                "out_of_sample": True,
                "methods": list(methods),
                "configuration_count": (
                    len(configurations)
                ),
                "experiment_count": (
                    len(experiments)
                ),
            },
        )


def run_covariance_robustness(
    prices: pd.DataFrame,
    configurations: list[
        RobustnessConfiguration
    ],
    methods: list[str] | None = None,
) -> CovarianceRobustnessResult:
    """
    Convenience function for covariance robustness analysis.
    """

    analyzer = CovarianceRobustnessAnalyzer(
        prices
    )

    return analyzer.run(
        configurations=configurations,
        methods=methods,
    )