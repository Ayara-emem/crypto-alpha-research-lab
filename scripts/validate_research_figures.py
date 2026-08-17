"""
Validate CARL research evidence against the research engine.

This module checks that the numerical results used by the
professional evidence package remain consistent with the
underlying CARL research implementation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_alpha_lab.data import load_prices

from crypto_alpha_lab.research.covariance_experiment import (
    run_covariance_experiment,
)

from crypto_alpha_lab.evaluation.covariance_comparison import (
    compare_covariance_methods,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSETS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
}

START = "2021-01-01"
END = "2025-12-31"

TRAIN_SIZE = 252
TEST_SIZE = 21

SHRINKAGE = 0.25

PERIODS_PER_YEAR = 252
RISK_FREE_RATE = 0.0

METHODS = [
    "sample",
    "shrinkage",
    "ledoit_wolf",
]


def load_prices() -> pd.DataFrame:
    """Load and align the CARL research universe."""

    prices = pd.concat(
        {
            name: load_prices_from_carl(
                ticker,
                START,
                END,
            )["Close"]
            for name, ticker in ASSETS.items()
        },
        axis=1,
    )

    prices.columns = list(
        ASSETS.keys()
    )

    prices = prices.dropna()

    if prices.empty:
        raise ValueError(
            "Research dataset is empty."
        )

    return prices


def load_prices_from_carl(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Delegate market-data loading to CARL."""

    from crypto_alpha_lab.data import (
        load_prices as carl_load_prices,
    )

    return carl_load_prices(
        ticker=ticker,
        start=start,
        end=end,
    )


def run_baseline_experiments(
    prices: pd.DataFrame,
):
    """Run the baseline covariance experiments."""

    experiments = []

    for method in METHODS:

        kwargs = {
            "prices": prices,
            "method": method,
            "train_size": TRAIN_SIZE,
            "test_size": TEST_SIZE,
        }

        if method == "shrinkage":
            kwargs["shrinkage"] = SHRINKAGE

        experiments.append(
            run_covariance_experiment(
                **kwargs
            )
        )

    return experiments


def validate_baseline_results(
    comparison,
) -> None:
    """
    Validate the established baseline results.

    Tolerance allows for small numerical differences while
    still detecting meaningful research inconsistencies.
    """

    expected = {
    "sample": {
        "total_return": 0.869704,
        "annualized_return": 0.106805,
        "annualized_volatility": 0.436643,
        "sharpe_ratio": 0.450538,
        "maximum_drawdown": -0.781959,
        "average_turnover": 1.219442,
    },
    "shrinkage": {
        "total_return": 0.781683,
        "annualized_return": 0.098184,
        "annualized_volatility": 0.435377,
        "sharpe_ratio": 0.432726,
        "maximum_drawdown": -0.767388,
        "average_turnover": 1.110687,
    },
    "ledoit_wolf": {
        "total_return": 0.568794,
        "annualized_return": 0.075755,
        "annualized_volatility": 0.437010,
        "sharpe_ratio": 0.385554,
        "maximum_drawdown": -0.725831,
        "average_turnover": 1.352342,
    },
}

    tolerance = 1e-5

    for method, metrics in expected.items():

        row = comparison.summary.loc[
            method
        ]

        for metric, expected_value in metrics.items():

            actual_value = row[metric]

            if abs(
                actual_value
                - expected_value
            ) > tolerance:

                raise AssertionError(
                    f"{method} {metric} mismatch: "
                    f"expected {expected_value:.6f}, "
                    f"got {actual_value:.6f}"
                )


def validate_training_window_results(
    prices: pd.DataFrame,
) -> None:
    """Validate the training-window sensitivity results."""

    expected = {
        126: {
            "sample": 0.822511,
            "shrinkage": 0.650715,
            "ledoit_wolf": 0.191804,
        },
        252: {
            "sample": 0.869704,
            "shrinkage": 0.781683,
            "ledoit_wolf": 0.568794,
        },
        504: {
            "sample": 1.948960,
            "shrinkage": 1.758280,
            "ledoit_wolf": 1.276179,
        },
    }

    tolerance = 1e-5

    for train_size, methods in expected.items():

        for method, expected_return in methods.items():

            kwargs = {
                "prices": prices,
                "method": method,
                "train_size": train_size,
                "test_size": TEST_SIZE,
            }

            if method == "shrinkage":
                kwargs["shrinkage"] = SHRINKAGE

            experiment = run_covariance_experiment(
                **kwargs
            )

            actual_return = (
                (1.0 + experiment.returns)
                .prod()
                - 1.0
            )

            if abs(
                actual_return
                - expected_return
            ) > tolerance:

                raise AssertionError(
                    f"Training window mismatch: "
                    f"train_size={train_size}, "
                    f"method={method}; "
                    f"expected "
                    f"{expected_return:.6f}, "
                    f"got "
                    f"{actual_return:.6f}"
                )


def validate_figure_files() -> None:
    """Verify that all expected figure files exist."""

    expected_files = [
        "fig01_cumulative_oos_returns.png",
        "fig02_risk_adjusted_performance.png",
        "fig03_maximum_drawdown.png",
        "fig04_turnover.png",
        "fig05_training_window_sensitivity.png",
        "fig06_shrinkage_sensitivity.png",
        "fig07_regime_analysis.png",
    ]

    figures_dir = (
        PROJECT_ROOT
        / "docs"
        / "figures"
    )

    for filename in expected_files:

        path = figures_dir / filename

        if not path.exists():
            raise AssertionError(
                f"Missing figure: {path}"
            )

        if path.stat().st_size == 0:
            raise AssertionError(
                f"Empty figure file: {path}"
            )


def main() -> None:

    print(
        "Loading research data..."
    )

    prices = load_prices()

    print(
        "Running baseline experiments..."
    )

    experiments = (
        run_baseline_experiments(
            prices
        )
    )

    comparison = (
        compare_covariance_methods(
            experiments,
            periods_per_year=PERIODS_PER_YEAR,
            risk_free_rate=RISK_FREE_RATE,
        )
    )

    print(
        "Validating baseline results..."
    )

    validate_baseline_results(
        comparison
    )

    print(
        "Validating training-window results..."
    )

    validate_training_window_results(
        prices
    )

    print(
        "Validating figure files..."
    )

    validate_figure_files()

    print()
    print(
        "EP-4.2 VALIDATION PASSED"
    )
    print(
        "Research numbers and figure artifacts "
        "are internally consistent."
    )


if __name__ == "__main__":
    main()