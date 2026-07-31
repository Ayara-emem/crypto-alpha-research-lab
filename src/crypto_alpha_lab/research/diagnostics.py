"""
Feature Diagnostics

Statistical diagnostics for evaluating CARL research
features before predictive modeling and signal generation.
"""

from __future__ import annotations

import pandas as pd


def feature_summary(
    features: pd.DataFrame,
) -> pd.DataFrame:

    """
    Compute descriptive statistics for research features.

    Parameters
    ----------
    features
        CARL feature matrix.

    Returns
    -------
    pandas.DataFrame
        Descriptive statistics by feature.
    """

    if features.empty:
        raise ValueError(
            "features must not be empty."
        )

    return features.describe().T
def feature_correlation(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute pairwise feature correlations.
    """

    if features.empty:
        raise ValueError(
            "features must not be empty."
        )

    return features.corr()

def high_correlation_pairs(
    features: pd.DataFrame,
    threshold: float = 0.90,
) -> pd.DataFrame:
    """
    Identify feature pairs with high absolute correlation.
    """

    if features.empty:
        raise ValueError(
            "features must not be empty."
        )

    if not 0 < threshold <= 1:
        raise ValueError(
            "threshold must lie in (0, 1]."
        )

    correlation = features.corr()

    records = []

    columns = correlation.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):

            value = correlation.iloc[i, j]

            if abs(value) >= threshold:
                records.append(
                    {
                        "feature_1": columns[i],
                        "feature_2": columns[j],
                        "correlation": value,
                    }
                )

    return pd.DataFrame(
        records,
        columns=[
            "feature_1",
            "feature_2",
            "correlation",
        ],
    )

def missing_feature_fraction(
    features: pd.DataFrame,
) -> pd.Series:
    """
    Compute the fraction of missing observations
    for each research feature.
    """

    if features.empty:
        raise ValueError(
            "features must not be empty."
        )

    return features.isna().mean()
