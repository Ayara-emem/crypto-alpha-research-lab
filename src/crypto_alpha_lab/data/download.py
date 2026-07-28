"""
Market data downloader for the Crypto Alpha Research Laboratory (CARL).

This module is responsible only for downloading historical market data
from external data providers. It does not perform caching, validation,
or data storage.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_prices(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download historical OHLCV market data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. "BTC-USD").

    start : str
        Start date in YYYY-MM-DD format.

    end : str
        End date in YYYY-MM-DD format.

    interval : str, default="1d"
        Data frequency.

    auto_adjust : bool, default=True
        Whether to automatically adjust historical prices.

    Returns
    -------
    pandas.DataFrame
        Historical OHLCV market data.

    Raises
    ------
    ValueError
        If no market data are returned.
    """

    prices = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if prices.empty:
        raise ValueError(
            f"No market data returned for ticker '{ticker}'."
        )

    return prices