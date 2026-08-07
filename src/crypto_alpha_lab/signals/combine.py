"""
Signal combination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def combine_signals(
    signals: pd.DataFrame,
    weights: np.ndarray | None = None,
) -> pd.Series:
    """
    Combine multiple signals into one.
    """

    if not isinstance(
        signals,
        pd.DataFrame,
    ):
        raise TypeError(
            "signals must be a pandas DataFrame."
        )

    if signals.empty:
        raise ValueError(
            "signals are empty."
        )

    if weights is None:
        weights = np.ones(
            signals.shape[1],
        )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    if len(weights) != signals.shape[1]:
        raise ValueError(
            "weights length mismatch."
        )

    weights /= weights.sum()

    combined = signals.to_numpy() @ weights

    return pd.Series(
        combined,
        index=signals.index,
        name="combined_signal",
    )


def average_signal(
    signals: pd.DataFrame,
) -> pd.Series:
    """
    Equal-weight signal combination.
    """

    return combine_signals(
        signals,
    )


def weighted_signal(
    signals: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """
    Explicit weighted combination.
    """

    return combine_signals(
        signals,
        weights=weights,
    )