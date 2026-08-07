"""
Signal transformation.

Convert validated alpha factors into investable
continuous trading signals.
"""

from __future__ import annotations

import pandas as pd

from crypto_alpha_lab.signals.base import (
    _align_signal,
    _clip_signal,
    _validate_signal,
)

from enum import Enum


class SignalDirection(str, Enum):

    LONG_ONLY = "long_only"

    SHORT_ONLY = "short_only"

    LONG_SHORT = "long_short"


def build_signal(
    alpha: pd.Series,
    direction=SignalDirection.LONG_SHORT,
) -> pd.Series:
    """
    Convert an alpha factor into a trading signal.

    Parameters
    ----------
    alpha
        Alpha factor.

    direction
        One of:

        - "long_short"
        - "long_only"
        - "short_only"

    Returns
    -------
    pandas.Series
        Trading signal.
    """

    signal = _align_signal(
        alpha,
    )

    if direction == "long_short":
        return _clip_signal(
            signal,
        )

    if direction == "long_only":
        return _clip_signal(
            signal.clip(lower=0),
            lower=0,
            upper=1,
        )

    if direction == "short_only":
        return _clip_signal(
            (-signal).clip(lower=0),
            lower=0,
            upper=1,
        )

    raise ValueError(
        "Unknown direction."
    )

def invert_signal(
    signal: pd.Series,
) -> pd.Series:
    """
    Reverse a signal.
    """

    _validate_signal(
        signal,
    )

    return -signal

def rescale_signal(
    signal: pd.Series,
    lower: float = -1.0,
    upper: float = 1.0,
) -> pd.Series:
    """
    Linearly rescale a signal.
    """

    _validate_signal(
        signal,
    )

    if lower >= upper:
        raise ValueError(
            "lower must be less than upper."
        )

    minimum = signal.min()
    maximum = signal.max()

    if minimum == maximum:
        return pd.Series(
            0.0,
            index=signal.index,
            name=signal.name,
        )

    scaled = (
        signal - minimum
    ) / (
        maximum - minimum
    )

    scaled = (
        scaled * (upper - lower)
        + lower
    )

    return scaled

