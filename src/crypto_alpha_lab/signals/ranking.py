"""
Signal ranking.
"""

from __future__ import annotations

import pandas as pd

from crypto_alpha_lab.signals.base import (
    _validate_signal,
)


def rank_signals(
    signal: pd.Series,
    ascending: bool = False,
) -> pd.Series:
    """
    Rank signals.
    """

    _validate_signal(signal)

    return signal.rank(
        ascending=ascending,
        method="dense",
    )


def top_k_signals(
    signal: pd.Series,
    k: int,
) -> pd.Series:
    """
    Return the strongest k signals.
    """

    _validate_signal(signal)

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    return signal.nlargest(k)


def bottom_k_signals(
    signal: pd.Series,
    k: int,
) -> pd.Series:
    """
    Return the weakest k signals.
    """

    _validate_signal(signal)

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    return signal.nsmallest(k)