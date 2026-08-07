"""
Signal normalization.

Normalize trading signals onto a common scale.
"""

from __future__ import annotations

import pandas as pd

from crypto_alpha_lab.signals.base import (
    _validate_signal,
)

def _zscore(
    signal: pd.Series,
) -> pd.Series:

    std = signal.std()

    if std == 0:
        return pd.Series(
            0.0,
            index=signal.index,
            name=signal.name,
        )

    return (
        signal - signal.mean()
    ) / std

def _rank(
    signal: pd.Series,
) -> pd.Series:

    return signal.rank(
        pct=True,
    )

def _minmax(
    signal: pd.Series,
) -> pd.Series:

    minimum = signal.min()

    maximum = signal.max()

    if minimum == maximum:

        return pd.Series(
            0.0,
            index=signal.index,
            name=signal.name,
        )

    return (
        signal - minimum
    ) / (
        maximum - minimum
    )


def normalize_signal(
    signal: pd.Series,
    method: str = "zscore",
) -> pd.Series:
    """
    Normalize a trading signal.
    """

    _validate_signal(
        signal,
    )

    if method == "zscore":
        return _zscore(signal)

    if method == "rank":
        return _rank(signal)

    if method == "minmax":
        return _minmax(signal)

    raise ValueError(
        "Unknown normalization method."
    )

