"""
Portfolio constraints.
"""

from __future__ import annotations

import pandas as pd


def long_only_constraint(
    weights: pd.Series,
) -> pd.Series:
    """
    Remove short positions.
    """

    clipped = weights.clip(
        lower=0,
    )

    if clipped.sum() == 0:
        return clipped

    return clipped / clipped.sum()

def leverage_constraint(
    weights: pd.Series,
    leverage: float = 1.0,
) -> pd.Series:
    """
    Scale leverage.
    """

    if leverage <= 0:
        raise ValueError(
            "leverage must be positive."
        )

    gross = weights.abs().sum()

    if gross == 0:
        return weights.copy()

    return (
        weights
        * leverage
        / gross
    )

def max_weight_constraint(
    weights: pd.Series,
    maximum: float,
) -> pd.Series:
    """
    Cap individual positions.
    """

    if maximum <= 0:
        raise ValueError(
            "maximum must be positive."
        )

    clipped = weights.clip(
        upper=maximum,
    )

    return clipped / clipped.sum()



