"""
Cache utilities for market data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CACHE_DIR = Path("data/raw")


def ensure_cache_directory() -> None:
    """
    Create cache directory if it does not exist.
    """

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def cache_path(
    ticker: str,
) -> Path:
    """
    Return cache filename.
    """

    return CACHE_DIR / f"{ticker}.parquet"


def save_prices(
    df: pd.DataFrame,
    ticker: str,
) -> None:
    """
    Save prices to local cache.
    """

    ensure_cache_directory()

    path = cache_path(ticker)

    df.to_parquet(path)


def load_cached_prices(
    ticker: str,
) -> pd.DataFrame:
    """
    Load cached prices.

    Raises
    ------
    FileNotFoundError
        If cache does not exist.
    """

    path = cache_path(ticker)

    if not path.exists():
        raise FileNotFoundError(
            f"No cached data for {ticker}."
        )

    return pd.read_parquet(path)