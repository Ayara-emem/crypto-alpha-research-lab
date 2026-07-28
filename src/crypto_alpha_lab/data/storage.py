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

    Data are loaded from the local cache whenever possible.
    Otherwise, they are downloaded.

    Regardless of the source, the returned DataFrame is
    normalized, validated, finalized, and guaranteed to satisfy
    the CARL data contract.
    """

    downloaded = False

    if cache_exists(ticker) and not refresh:

        prices = load_cached_prices(ticker)

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