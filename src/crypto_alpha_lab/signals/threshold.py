"""
Signal thresholding.

Convert continuous signals into actionable trading
signals by suppressing weak predictions.
"""

from __future__ import annotations

import pandas as pd

from crypto_alpha_lab.signals.base import (
    _validate_signal,
)


def threshold_signal(
    signal: pd.Series,
    threshold: float = 0.05,
) -> pd.Series:
    """
    Zero-out signals whose magnitude is below a threshold.
    """

    _validate_signal(signal)

    if threshold < 0:
        raise ValueError(
            "threshold must be non-negative."
        )

    result = signal.copy()

    result[result.abs() < threshold] = 0.0

    return result


def binary_signal(
    signal: pd.Series,
    threshold: float = 0.0,
) -> pd.Series:
    """
    Convert a continuous signal into
    {-1, 0, 1}.
    """

    _validate_signal(signal)

    if threshold < 0:
        raise ValueError(
            "threshold must be non-negative."
        )

    result = pd.Series(
        0,
        index=signal.index,
        dtype=int,
        name=signal.name,
    )

    result[signal > threshold] = 1
    result[signal < -threshold] = -1

    return result