"""
Validation utilities for market data.
"""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def validate_prices(df: pd.DataFrame) -> None:
    """
    Validate historical market data.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical OHLCV data.

    Raises
    ------
    ValueError
        If validation fails.
    """

    if df.empty:
        raise ValueError("DataFrame is empty.")

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.isna().any().any():
        raise ValueError(
            "Data contains missing values."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Datetime index must be sorted."
        )

    if df.index.has_duplicates:
        raise ValueError(
            "Duplicate timestamps detected."
        )