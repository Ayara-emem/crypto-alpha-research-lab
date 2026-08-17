"""
Generate CARL research evidence figures.

This script is intentionally a presentation layer.

All portfolio and covariance calculations are delegated to the
existing CARL research engine. The script consumes CARL experiment
results and converts them into reproducible research figures.

Output directory
----------------
docs/figures/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from crypto_alpha_lab.data import load_prices
from crypto_alpha_lab.research.covariance_experiment import (
    run_covariance_experiment,
)
from crypto_alpha_lab.evaluation.covariance_comparison import (
    compare_covariance_methods,
)


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIGURES_DIR = (
    PROJECT_ROOT
    / "docs"
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Research configuration
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def load_research_prices() -> pd.DataFrame:
    """
    Load and align the CARL research universe.
    """

    prices = pd.concat(
        {
            name: load_prices(
                ticker=ticker,
                start=START,
                end=END,
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
            "Research price dataset is empty."
        )

    return prices


# ---------------------------------------------------------------------
# Experiment engine
# ---------------------------------------------------------------------

def run_baseline_experiments(
    prices: pd.DataFrame,
):
    """
    Run the baseline covariance experiments.

    Returns
    -------
    list[CovarianceExperimentResult]
    """

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

        experiment = run_covariance_experiment(
            **kwargs
        )

        experiments.append(
            experiment
        )

    return experiments


# ---------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------

def generate_cumulative_oos_returns(
    experiments,
) -> None:
    """
    Generate cumulative out-of-sample return figure.
    """

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for experiment in experiments:

        wealth = (
            1.0
            + experiment.cumulative_returns
        )

        ax.plot(
            wealth.index,
            wealth.values,
            label=experiment.method,
            linewidth=2,
        )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(
        "Cumulative Out-of-Sample Portfolio Returns"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Cumulative Wealth"
    )

    ax.legend(
        title="Covariance Method"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "fig01_cumulative_oos_returns.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------

def generate_risk_adjusted_performance(
    experiments,
) -> None:
    """
    Generate risk-adjusted performance comparison.
    """

    comparison = compare_covariance_methods(
        experiments,
        periods_per_year=PERIODS_PER_YEAR,
        risk_free_rate=RISK_FREE_RATE,
    )

    summary = comparison.summary[
        [
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
        ]
    ]

    ax = summary.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "Risk-Adjusted Performance Comparison"
    )

    ax.set_xlabel(
        "Covariance Method"
    )

    ax.set_ylabel(
        "Ratio"
    )

    ax.legend(
        title="Metric"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.xticks(
        rotation=0
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "fig02_risk_adjusted_performance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------

def generate_maximum_drawdown(
    experiments,
) -> None:
    """
    Generate maximum drawdown comparison.
    """

    comparison = compare_covariance_methods(
        experiments,
        periods_per_year=PERIODS_PER_YEAR,
        risk_free_rate=RISK_FREE_RATE,
    )

    summary = comparison.summary[
        [
            "maximum_drawdown"
        ]
    ]

    ax = summary.plot(
        kind="bar",
        figsize=(9, 6),
        legend=False,
    )

    ax.set_title(
        "Maximum Drawdown by Covariance Method"
    )

    ax.set_xlabel(
        "Covariance Method"
    )

    ax.set_ylabel(
        "Maximum Drawdown"
    )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.xticks(
        rotation=0
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "fig03_maximum_drawdown.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------

def generate_turnover(
    experiments,
) -> None:
    """
    Generate average turnover comparison.
    """

    comparison = compare_covariance_methods(
        experiments,
        periods_per_year=PERIODS_PER_YEAR,
        risk_free_rate=RISK_FREE_RATE,
    )

    summary = comparison.summary[
        [
            "average_turnover"
        ]
    ]

    ax = summary.plot(
        kind="bar",
        figsize=(9, 6),
        legend=False,
    )

    ax.set_title(
        "Average Portfolio Turnover"
    )

    ax.set_xlabel(
        "Covariance Method"
    )

    ax.set_ylabel(
        "Average Turnover"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.xticks(
        rotation=0
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "fig04_turnover.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Training-window experiments
# ---------------------------------------------------------------------

def run_training_window_experiments(
    prices: pd.DataFrame,
):
    """
    Run experiments for training-window sensitivity.
    """

    results = []

    for train_size in [
        126,
        252,
        504,
    ]:

        for method in METHODS:

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

            total_return = (
                (1.0 + experiment.returns)
                .prod()
                - 1.0
            )

            results.append(
                {
                    "train_size": train_size,
                    "method": method,
                    "total_return": total_return,
                }
            )

    return pd.DataFrame(
        results
    )


# ---------------------------------------------------------------------
# Figure 5
# ---------------------------------------------------------------------

def generate_training_window_sensitivity(
    prices: pd.DataFrame,
) -> None:
    """
    Generate training-window sensitivity figure.
    """

    results = run_training_window_experiments(
        prices
    )

    table = results.pivot(
        index="train_size",
        columns="method",
        values="total_return",
    )

    ax = table.plot(
        marker="o",
        figsize=(10, 6),
    )

    ax.set_title(
        "Training-Window Sensitivity"
    )

    ax.set_xlabel(
        "Training Window (Observations)"
    )

    ax.set_ylabel(
        "Total Out-of-Sample Return"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend(
        title="Covariance Method"
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "fig05_training_window_sensitivity.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Shrinkage sensitivity
# ---------------------------------------------------------------------

def run_shrinkage_sensitivity(
    prices: pd.DataFrame,
):
    """
    Evaluate fixed shrinkage intensities.
    """

    results = []

    for shrinkage in [
        0.00,
        0.10,
        0.25,
        0.50,
        0.75,
        1.00,
    ]:

        experiment = run_covariance_experiment(
            prices=prices,
            method="shrinkage",
            train_size=TRAIN_SIZE,
            test_size=TEST_SIZE,
            shrinkage=shrinkage,
        )

        comparison = compare_covariance_methods(
            [experiment],
            periods_per_year=PERIODS_PER_YEAR,
            risk_free_rate=RISK_FREE_RATE,
        )

        row = comparison.summary.iloc[0]

        results.append(
            {
                "shrinkage": shrinkage,
                "annualized_return": (
                    row["annualized_return"]
                ),
                "sharpe_ratio": (
                    row["sharpe_ratio"]
                ),
                "maximum_drawdown": (
                    row["maximum_drawdown"]
                ),
            }
        )

    return pd.DataFrame(
        results
    )


# ---------------------------------------------------------------------
# Figure 6
# ---------------------------------------------------------------------

def generate_shrinkage_sensitivity(
    prices: pd.DataFrame,
) -> None:
    """
    Generate shrinkage-intensity sensitivity figure.
    """

    results = run_shrinkage_sensitivity(
        prices
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 12),
        sharex=True,
    )

    axes[0].plot(
        results["shrinkage"],
        results["annualized_return"],
        marker="o",
    )

    axes[0].set_ylabel(
        "Annualized Return"
    )

    axes[0].set_title(
        "Shrinkage-Intensity Sensitivity"
    )

    axes[1].plot(
        results["shrinkage"],
        results["sharpe_ratio"],
        marker="o",
    )

    axes[1].set_ylabel(
        "Sharpe Ratio"
    )

    axes[2].plot(
        results["shrinkage"],
        results["maximum_drawdown"],
        marker="o",
    )

    axes[2].set_ylabel(
        "Maximum Drawdown"
    )

    axes[2].set_xlabel(
        "Shrinkage Intensity"
    )

    for axis in axes:
        axis.grid(
            alpha=0.3
        )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "fig06_shrinkage_sensitivity.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------

REGIMES = {
    "2021_2022": (
        "2021-01-01",
        "2022-12-31",
    ),
    "2023_2024": (
        "2023-01-01",
        "2024-12-31",
    ),
    "2025": (
        "2025-01-01",
        "2025-12-31",
    ),
}


def run_regime_experiments(
    prices: pd.DataFrame,
):
    """
    Run covariance experiments separately for each research regime.

    The regime windows are applied to the research dataset before the
    walk-forward experiment is executed.
    """

    results = []

    for period, (
        start,
        end,
    ) in REGIMES.items():

        period_prices = prices.loc[
            start:end
        ]

        if period_prices.empty:
            continue

        for method in METHODS:

            kwargs = {
                "prices": period_prices,
                "method": method,
                "train_size": TRAIN_SIZE,
                "test_size": TEST_SIZE,
            }

            if method == "shrinkage":
                kwargs["shrinkage"] = SHRINKAGE

            try:

                experiment = (
                    run_covariance_experiment(
                        **kwargs
                    )
                )

            except ValueError:
                continue

            comparison = (
                compare_covariance_methods(
                    [experiment],
                    periods_per_year=(
                        PERIODS_PER_YEAR
                    ),
                    risk_free_rate=(
                        RISK_FREE_RATE
                    ),
                )
            )

            row = comparison.summary.iloc[0]

            results.append(
                {
                    "period": period,
                    "method": method,
                    "annualized_return": (
                        row[
                            "annualized_return"
                        ]
                    ),
                    "sharpe_ratio": (
                        row[
                            "sharpe_ratio"
                        ]
                    ),
                    "maximum_drawdown": (
                        row[
                            "maximum_drawdown"
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        results
    )


# ---------------------------------------------------------------------
# Figure 7
# ---------------------------------------------------------------------

def generate_regime_analysis(
    prices: pd.DataFrame,
) -> None:
    """
    Generate regime-level Sharpe-ratio comparison.
    """

    results = run_regime_experiments(
        prices
    )

    if results.empty:
        raise ValueError(
            "No valid regime experiments were produced."
        )

    table = results.pivot(
        index="period",
        columns="method",
        values="sharpe_ratio",
    )

    ax = table.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "Sharpe Ratio Across Historical Regimes"
    )

    ax.set_xlabel(
        "Research Period"
    )

    ax.set_ylabel(
        "Sharpe Ratio"
    )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.legend(
        title="Covariance Method"
    )

    plt.xticks(
        rotation=0
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "fig07_regime_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """
    Generate the complete CARL research evidence figure set.
    """

    print(
        "Loading CARL research dataset..."
    )

    prices = load_research_prices()

    print(
        f"Research dataset: "
        f"{prices.shape[0]} observations × "
        f"{prices.shape[1]} assets"
    )

    print(
        "Running baseline experiments..."
    )

    experiments = (
        run_baseline_experiments(
            prices
        )
    )

    print(
        "Generating Figure 1..."
    )

    generate_cumulative_oos_returns(
        experiments
    )

    print(
        "Generating Figure 2..."
    )

    generate_risk_adjusted_performance(
        experiments
    )

    print(
        "Generating Figure 3..."
    )

    generate_maximum_drawdown(
        experiments
    )

    print(
        "Generating Figure 4..."
    )

    generate_turnover(
        experiments
    )

    print(
        "Generating Figure 5..."
    )

    generate_training_window_sensitivity(
        prices
    )

    print(
        "Generating Figure 6..."
    )

    generate_shrinkage_sensitivity(
        prices
    )

    print(
        "Generating Figure 7..."
    )

    generate_regime_analysis(
        prices
    )

    print(
        "\nCARL research figures generated successfully."
    )

    print(
        f"Output directory: {FIGURES_DIR}"
    )


if __name__ == "__main__":
    main()