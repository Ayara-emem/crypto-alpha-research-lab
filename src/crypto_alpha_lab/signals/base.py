"""
Base utilities for signal engineering.

This module contains shared validation and helper
functions used throughout the CARL signal package.
"""

from __future__ import annotations

import pandas as pd


def _validate_signal(
    signal: pd.Series,
) -> None:
    """
    Validate a trading signal.
    """

    if not isinstance(
        signal,
        pd.Series,
    ):
        raise TypeError(
            "signal must be a pandas Series."
        )

    if signal.empty:
        raise ValueError(
            "signal is empty."
        )


def _align_signal(
    signal: pd.Series,
) -> pd.Series:
    """
    Return a cleaned signal with missing values removed.
    """

    _validate_signal(
        signal,
    )

    return signal.dropna()


def _clip_signal(
    signal: pd.Series,
    lower: float = -1.0,
    upper: float = 1.0,
) -> pd.Series:
    """
    Clip signal values to a fixed interval.
    """

    _validate_signal(
        signal,
    )

    if lower >= upper:
        raise ValueError(
            "lower must be less than upper."
        )

    return signal.clip(
        lower=lower,
        upper=upper,
    )