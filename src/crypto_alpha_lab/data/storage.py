"""
Storage interface for the Crypto Alpha Research Laboratory (CARL).

This module provides the public API for loading and saving market data.
"""

from __future__ import annotations

import pandas as pd

from crypto_alpha_lab.data.cache import (
    cache_exists,
    load_cached_prices,
    save_cache_prices,
)

from crypto_alpha_lab.data.download import download_prices

from crypto_alpha_lab.data.validation import validate_prices


def _normalize_columns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Yahoo Finance column names.

    Single-ticker downloads from yfinance may contain a MultiIndex.
    This function converts them to standard OHLCV columns.
    """

    if isinstance(prices.columns, pd.MultiIndex):

        if prices.columns.get_level_values("Ticker").nunique() == 1:

            prices.columns = prices.columns.get_level_values("Price")

    return prices


def _finalize_dataframe(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Apply final preprocessing before returning data.
    """

    prices = prices.sort_index()

    prices = prices.loc[~prices.index.duplicated()]

    return prices


def _cache_covers_range(
    prices: pd.DataFrame,
    start: str,
    end: str,
    interval: str = "1D",
) -> bool:
    """
    Return whether cached prices cover the requested date range.

    The ``end`` date follows the data-provider convention used
    by CARL and is treated as an exclusive upper boundary.

    Parameters
    ----------
    prices
        Cached market data.

    start
        Requested start date.

    end
        Requested end date.

    interval
        Data frequency.

    Returns
    -------
    bool
        True when the cached data fully cover the requested range.
    """

    if prices.empty:
        return False

    if not prices.index.is_monotonic_increasing:
        return False

    requested_start = pd.Timestamp(start)
    requested_end = pd.Timestamp(end)

    cached_start = pd.Timestamp(
        prices.index[0]
    )

    cached_end = pd.Timestamp(
        prices.index[-1]
    )

    try:
        interval_offset = pd.tseries.frequencies.to_offset(
            interval
        )
    except ValueError:
        return False

    required_last_observation = (
        requested_end - interval_offset
    )

    return (
        cached_start <= requested_start
        and cached_end >= required_last_observation
    )


def load_prices(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1D",
    auto_adjust: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Load historical market data.

    Data are loaded from the local cache whenever possible.
    Otherwise, they are downloaded.

    Cached data are used only when they cover the requested
    date range.

    Regardless of the source, the returned DataFrame is
    normalized, validated, finalized, and guaranteed to satisfy
    the CARL data contract.
    """

    downloaded = False

    if cache_exists(ticker) and not refresh:

        cached_prices = load_cached_prices(ticker)

        if _cache_covers_range(
    cached_prices,
    start=start,
    end=end,
    interval=interval,
):
            prices = cached_prices

        else:
            prices = download_prices(
                ticker=ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
            )

            downloaded = True

    else:

        prices = download_prices(
            ticker=ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
        )

        downloaded = True

    # Apply the CARL data pipeline
    prices = _normalize_columns(prices)

    validate_prices(prices)

    prices = _finalize_dataframe(prices)

    # Save only normalized data
    if downloaded:

        save_cache_prices(
            prices,
            ticker,
        )

    return prices