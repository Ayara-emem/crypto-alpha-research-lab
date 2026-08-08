"""
Portfolio weighting methods.

Institutional portfolio construction algorithms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

def equal_weight(
    assets: list[str],
) -> pd.Series:
    """
    Equal-weight allocation.
    """

    if len(assets) == 0:
        raise ValueError(
            "assets cannot be empty."
        )

    weight = 1 / len(assets)

    return pd.Series(
        weight,
        index=assets,
        name="weight",
    )

def inverse_volatility_weight(
    volatility: pd.Series,
) -> pd.Series:
    """
    Inverse-volatility weighting.
    """

    if (volatility <= 0).any():
        raise ValueError(
            "volatility must be positive."
        )

    inverse = 1 / volatility

    weights = inverse / inverse.sum()

    return weights.rename(
        "weight",
    )

def proportional_weight(
    score: pd.Series,
) -> pd.Series:
    """
    Normalize positive scores.
    """

    if (score < 0).any():
        raise ValueError(
            "scores must be non-negative."
        )

    total = score.sum()

    if total == 0:
        raise ValueError(
            "sum of scores is zero."
        )

    return (
        score / total
    ).rename("weight")


def long_short_weight(
    signal: pd.Series,
) -> pd.Series:
    """
    Dollar-neutral weights.
    """

    if signal.abs().sum() == 0:
        raise ValueError(
            "signal contains no exposure."
        )

    return (
        signal / signal.abs().sum()
    ).rename("weight")

