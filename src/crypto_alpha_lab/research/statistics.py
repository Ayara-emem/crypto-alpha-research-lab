from __future__ import annotations
import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
)

def _split_features_target(
    research: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Validate a research dataset and separate the
    predictor matrix from the target.
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

    X = research.drop(
        columns=target,
    )

    y = research[target]

    return X, y

def feature_target_correlation(
    research: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """
    Pearson correlation between every feature
    and the prediction target.
    """

    X, y = _split_features_target(
        research,
        target,
    )

    rows = []

    for feature in X.columns:

        correlation, _ = pearsonr(
            X[feature],
            y,
        )

        rows.append(
            {
                "feature": feature,
                "correlation": correlation,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "correlation",
            key=lambda s: s.abs(),
            ascending=False,
        )
        .reset_index(drop=True)
    )

def feature_target_rank_correlation(
    research: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """
    Compute the Spearman rank correlation between each feature
    and the prediction target.

    Parameters
    ----------
    research
        Research dataset containing engineered features and
        the prediction target.

    target
        Name of the target column.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the feature names and their
        Spearman rank correlations with the target.
    """

    X, y = _split_features_target(
        research,
        target,
    )

    rows = []

    for feature in X.columns:

        correlation, _ = spearmanr(
            X[feature],
            y,
        )

        rows.append(
            {
                "feature": feature,
                "correlation": correlation,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "correlation",
            key=lambda s: s.abs(),
            ascending=False,
        )
        .reset_index(drop=True)
    )

def information_coefficient(
    research: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """
    Compute the Information Coefficient (IC)
    between each feature and the prediction target.
    """
    return feature_target_rank_correlation(
        research,
        target,
    )

def correlation_matrix(
    research: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pairwise Pearson correlation matrix.
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

    return research.corr(
        numeric_only=True,
    )

def feature_p_values(
    research: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """
    Pearson p-values for every feature.
    """

    X, y = _split_features_target(
        research,
        target,
    )

    rows = []

    for feature in X.columns:

        correlation, p_value = pearsonr(
            X[feature],
            y,
        )

        rows.append(
            {
                "feature": feature,
                "correlation": correlation,
                "p_value": p_value,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "p_value",
        )
        .reset_index(drop=True)
    )

def summary_statistics(
    research: pd.DataFrame,
) -> pd.DataFrame:
    """
    Descriptive statistics for a research dataset.
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

    summary = research.describe().T

    summary["skew"] = research.skew()

    summary["kurtosis"] = research.kurtosis()

    return summary

from typing import Literal

def _safe_correlation(
    x: pd.Series,
    y: pd.Series,
    method: Literal["pearson", "spearman"],
) -> tuple[float, float]:
    """
    Compute a robust correlation and p-value between two series.
    """

    data = pd.concat(
        [x, y],
        axis=1,
    ).dropna()

    if len(data) < 2:
        return np.nan, np.nan

    x = data.iloc[:, 0]
    y = data.iloc[:, 1]

    if x.nunique() <= 1:
        return np.nan, np.nan

    if y.nunique() <= 1:
        return np.nan, np.nan

    if method == "pearson":
        return pearsonr(x, y)

    if method == "spearman":
        return spearmanr(x, y)

    raise ValueError(
        f"Unknown method '{method}'."
    )