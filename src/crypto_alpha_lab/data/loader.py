"""
High-level market data loader.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cache import (
    cache_path,
    load_cached_prices,
    save_prices,
)

from .download import download_prices

from .validation import validate_prices


def load_prices(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    auto_adjust: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Load historical market data.

    The function first checks the local cache.
    If cached data exists, it is loaded.

    Otherwise the data is downloaded,
    validated,
    cached,
    and returned.

    Parameters
    ----------
    ticker : str

    start : str

    end : str

    interval : str

    auto_adjust : bool

    refresh : bool
        Force a fresh download.

    Returns
    -------
    pandas.DataFrame
    """

    path = cache_path(ticker)

    if path.exists() and not refresh:

        return load_cached_prices(
            ticker,
        )

    df = download_prices(
        ticker=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
    )

    validate_prices(df)

    save_prices(
        df,
        ticker,
    )

    return df