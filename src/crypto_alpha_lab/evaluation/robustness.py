from __future__ import annotations

import pandas as pd


def rank_methods_by_configuration(
    summary: pd.DataFrame,
    metric: str = "total_return",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Rank covariance methods within each robustness configuration.
    """

    if not isinstance(
        summary,
        pd.DataFrame,
    ):
        raise TypeError(
            "summary must be a pandas DataFrame."
        )

    if summary.empty:
        raise ValueError(
            "summary cannot be empty."
        )

    required = {
        "method",
        "train_size",
        "test_size",
        metric,
    }

    missing = required - set(
        summary.columns
    )

    if missing:
        raise ValueError(
            "summary is missing required "
            f"columns: {sorted(missing)}"
        )

    result = summary.copy()

    result["rank"] = (
        result.groupby(
            [
                "train_size",
                "test_size",
            ]
        )[metric]
        .rank(
            ascending=ascending,
            method="min",
        )
    )

    return result

def ranking_stability(
    summary: pd.DataFrame,
    metric: str = "total_return",
) -> pd.DataFrame:
    """
    Measure how frequently each covariance method
    achieves each rank across robustness configurations.
    """

    ranked = rank_methods_by_configuration(
        summary=summary,
        metric=metric,
    )

    counts = (
        ranked
        .groupby(
            ["method", "rank"]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    counts.index.name = "method"

    counts.columns = [
        f"rank_{int(column)}"
        for column in counts.columns
    ]

    total = counts.sum(axis=1)

    for column in counts.columns:
        counts[column] = (
            counts[column] / total
        )

    return counts


def performance_stability(
    summary: pd.DataFrame,
    metric: str = "total_return",
) -> pd.DataFrame:

    if not isinstance(
        summary,
        pd.DataFrame,
    ):
        raise TypeError(
            "summary must be a pandas DataFrame."
        )

    if summary.empty:
        raise ValueError(
            "summary cannot be empty."
        )

    required = {
        "method",
        "train_size",
        "test_size",
        metric,
    }

    missing = required - set(
        summary.columns
    )

    if missing:
        raise ValueError(
            "summary is missing required "
            f"columns: {sorted(missing)}"
        )


    if summary.empty:
        raise ValueError(
            "summary cannot be empty."
        )

    result = (
        summary
        .groupby("method")[metric]
        .agg(
            mean="mean",
            standard_deviation="std",
            minimum="min",
            maximum="max",
        )
    )

    result["range"] = (
        result["maximum"]
        - result["minimum"]
    )

    result["observations"] = (
        summary.groupby("method")
        .size()
    )

    return result

def portfolio_weight_stability(
    experiments,
) -> pd.DataFrame:
    """
    Measure portfolio-weight stability for each
    covariance experiment.

    The metric is the mean L1 distance between
    consecutive out-of-sample fold portfolios.
    """

    if not experiments:
        raise ValueError(
            "experiments cannot be empty."
        )

    rows = []

    for experiment in experiments:

        if not experiment.folds:
            raise ValueError(
                f"experiment {experiment.method!r} "
                "contains no folds."
            )

        weights = [
            fold.weights
            for fold in experiment.folds
        ]

        distances = []

        for previous, current in zip(
            weights,
            weights[1:],
        ):

            previous, current = (
                previous.align(
                    current,
                    fill_value=0.0,
                )
            )

            distance = (
                previous
                .sub(current)
                .abs()
                .sum()
            )

            distances.append(
                float(distance)
            )

        if distances:
            mean_distance = sum(
                distances
            ) / len(distances)

            maximum_distance = max(
                distances
            )
        else:
            mean_distance = 0.0
            maximum_distance = 0.0

        rows.append(
            {
                "method": experiment.method,
                "mean_weight_change": (
                    mean_distance
                ),
                "maximum_weight_change": (
                    maximum_distance
                ),
                "fold_count": len(
                    weights
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("method")
    )
