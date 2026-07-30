"""
Internal utilities for feature engineering.

These helper functions support the feature modules and
are not part of CARL's public API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _align_to_prices(
    values: np.ndarray | pd.Series,
    prices: pd.Series,
) -> pd.Series:
    """
    Align APRL outputs with the original price index.

    Parameters
    ----------
    values
        Output returned by APRL.

    prices
        Original closing-price series.

    Returns
    -------
    pandas.Series
        Series aligned with the original price index.
    """

    if isinstance(values, pd.Series):
        return values.reindex(prices.index)

    return pd.Series(
        values,
        index=prices.index[1:],
        name=prices.name,
    ).reindex(prices.index)