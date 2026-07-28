from pathlib import Path
import pandas as pd

from crypto_alpha_lab.data.paths import CACHE_DIR


def cache_path(ticker: str) -> Path:
    """
    Return the cache file path for a ticker.

    Parameters
    ----------
    ticker : str
        Market symbol.

    Returns
    -------
    pathlib.Path
        Location of the cached parquet file.
    """
    return CACHE_DIR / f"{ticker}.parquet"

def cache_exists(ticker: str) -> bool:
    """
    Return True if a cache file exists.
    """
    return cache_path(ticker).exists()

from datetime import datetime
from datetime import timedelta


def cache_age(ticker: str) -> timedelta | None:
    """
    Return the age of a cached file.

    Returns
    -------
    timedelta
        Time since last modification.

    None
        If the cache does not exist.
    """
    path = cache_path(ticker)

    if not path.exists():
        return None

    modified = datetime.fromtimestamp(path.stat().st_mtime)

    return datetime.now() - modified

def load_cached_prices(ticker: str) -> pd.DataFrame:
    """
    Load cached market data.

    Parameters
    ----------
    ticker : str
        Market symbol.

    Returns
    -------
    pandas.DataFrame
        Cached OHLCV data.

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist.
    """

    path = cache_path(ticker)

    if not path.exists():
        raise FileNotFoundError(
            f"No cached data found for '{ticker}'."
        )

    return pd.read_parquet(path)


def save_cache_prices(
    prices: pd.DataFrame,
    ticker: str,
) -> None:
    """
    Save market data to the local cache.

    Parameters
    ----------
    prices : pandas.DataFrame
        Market data.

    ticker : str
        Market symbol.
    """

    path = cache_path(ticker)

    prices.to_parquet(path)

def clear_cache(ticker: str | None = None) -> None:
    """
    Delete cached data.

    Parameters
    ----------
    ticker : str, optional
        Remove one cache.

    None
        Remove every cache.
    """

    if ticker is None:

        for file in CACHE_DIR.glob("*.parquet"):
            file.unlink()

        return

    path = cache_path(ticker)

    if path.exists():
        path.unlink()

