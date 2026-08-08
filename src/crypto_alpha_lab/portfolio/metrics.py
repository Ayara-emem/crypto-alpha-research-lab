"""
Portfolio metrics.
"""

from __future__ import annotations

import pandas as pd

def gross_exposure(
    weights: pd.Series,
) -> float:
    """
    Gross portfolio exposure.
    """

    return float(
        weights.abs().sum()
    )


def net_exposure(
    weights: pd.Series,
) -> float:
    """
    Net portfolio exposure.
    """

    return float(
        weights.sum()
    )

def concentration(
    weights: pd.Series,
) -> float:
    """
    Portfolio concentration index.
    """

    return float(
        (weights ** 2).sum()
    )


def effective_number_of_positions(
    weights: pd.Series,
) -> float:
    """
    Effective diversification.
    """

    hhi = concentration(
        weights,
    )

    if hhi == 0:
        return 0.0

    return 1.0 / hhi

