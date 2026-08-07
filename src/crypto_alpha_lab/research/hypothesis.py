"""
Hypothesis testing for quantitative alpha research.

This module evaluates candidate alpha factors using
statistical significance tests and multiple-testing
procedures.

The hypothesis layer builds upon the research statistics
module and provides research-oriented APIs for deciding
whether an observed relationship is likely to represent
genuine predictive information rather than random noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
)

from crypto_alpha_lab.research.statistics import (
    feature_target_correlation,
    feature_target_rank_correlation,
)

def _validate_input(
    research: pd.DataFrame,
    target: str,
) -> None:
    """
    Validate a research dataset.
    """

    if not isinstance(
        research,
        pd.DataFrame,
    ):
        raise TypeError(
            "research must be a pandas DataFrame."
        )

    if research.empty:
        raise ValueError(
            "research dataset is empty."
        )

    if target not in research.columns:
        raise ValueError(
            f"Target '{target}' not found."
        )

def _collect_statistics(
    research: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman statistics for
    every feature.
    """

    _validate_input(
        research,
        target,
    )

    X = research.drop(
        columns=target,
    )

    y = research[target]

    rows = []

    for feature in X.columns:

        pearson_corr, pearson_p = pearsonr(
            X[feature],
            y,
        )

        spearman_corr, spearman_p = spearmanr(
            X[feature],
            y,
        )

        rows.append(
            {
                "feature": feature,
                "pearson": pearson_corr,
                "pearson_p": pearson_p,
                "information_coefficient": spearman_corr,
                "ic_p": spearman_p,
            }
        )

    return pd.DataFrame(rows)

def evaluate_alpha(
    research: pd.DataFrame,
    feature: str,
    target: str,
    alpha: float = 0.05,
) -> pd.Series:
    """
    Evaluate a single candidate alpha factor.
    """

    statistics = _collect_statistics(
        research,
        target,
    )

    result = statistics.loc[
        statistics["feature"] == feature
    ]

    if result.empty:
        raise ValueError(
            f"Feature '{feature}' not found."
        )

    result = result.squeeze()

    result["significant"] = (
        result["pearson_p"] < alpha
    ) and (
        result["ic_p"] < alpha
    )

    return result

def evaluate_alpha_universe(
    research: pd.DataFrame,
    target: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Evaluate every candidate alpha in a research dataset.
    """

    statistics = _collect_statistics(
        research,
        target,
    ).copy()

    statistics["significant"] = (
        (statistics["pearson_p"] < alpha)
        &
        (statistics["ic_p"] < alpha)
    )

    return (
        statistics
        .sort_values(
            "information_coefficient",
            key=lambda s: s.abs(),
            ascending=False,
        )
        .reset_index(drop=True)
    )

def bonferroni_correction(
    alpha_report: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply the Bonferroni correction for multiple testing.
    """

    report = alpha_report.copy()

    n_tests = len(report)

    adjusted_alpha = alpha / n_tests

    report["bonferroni_alpha"] = adjusted_alpha

    report["bonferroni_significant"] = (
        report["pearson_p"] < adjusted_alpha
    ) & (
        report["ic_p"] < adjusted_alpha
    )

    return report


def benjamini_hochberg(
    alpha_report: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply the Benjamini-Hochberg False Discovery Rate
    correction.
    """

    report = (
        alpha_report.copy()
        .sort_values("pearson_p")
        .reset_index(drop=True)
    )

    n = len(report)

    report["rank"] = np.arange(
        1,
        n + 1,
    )

    report["bh_threshold"] = (
        report["rank"] / n
    ) * alpha

    report["bh_significant"] = (
        report["pearson_p"]
        <= report["bh_threshold"]
    )

    return report

def bootstrap_ic(
    research: pd.DataFrame,
    feature: str,
    target: str,
    n_bootstrap: int = 1000,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Bootstrap the Information Coefficient.
    """

    _validate_input(
        research,
        target,
    )

    rng = np.random.default_rng(
        random_state,
    )

    data = research[
        [feature, target]
    ].dropna()

    estimates = []

    for _ in range(n_bootstrap):

        sample = data.sample(
            frac=1.0,
            replace=True,
            random_state=rng.integers(
                0,
                2**32 - 1,
            ),
        )

        ic, _ = spearmanr(
            sample[feature],
            sample[target],
        )

        estimates.append(ic)

    estimates = np.asarray(estimates)

    observed, _ = spearmanr(
        data[feature],
        data[target],
    )

    return pd.DataFrame(
        {
            "observed_ic": [observed],
            "bootstrap_mean": [
                estimates.mean()
            ],
            "bootstrap_std": [
                estimates.std(
                    ddof=1,
                )
            ],
            "lower_ci": [
                np.percentile(
                    estimates,
                    2.5,
                )
            ],
            "upper_ci": [
                np.percentile(
                    estimates,
                    97.5,
                )
            ],
        }
    )

def permutation_ic(
    research: pd.DataFrame,
    feature: str,
    target: str,
    n_permutations: int = 1000,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Perform a permutation test for the Information
    Coefficient.
    """

    _validate_input(
        research,
        target,
    )

    rng = np.random.default_rng(
        random_state,
    )

    data = research[
        [feature, target]
    ].dropna()

    observed, _ = spearmanr(
        data[feature],
        data[target],
    )

    permutations = []

    for _ in range(n_permutations):

        shuffled = rng.permutation(
            data[target]
        )

        ic, _ = spearmanr(
            data[feature],
            shuffled,
        )

        permutations.append(ic)

    permutations = np.asarray(
        permutations,
    )

    p_value = np.mean(
        np.abs(permutations)
        >= abs(observed)
    )

    return pd.DataFrame(
        {
            "observed_ic": [observed],
            "permutation_p_value": [
                p_value
            ],
        }
    )

