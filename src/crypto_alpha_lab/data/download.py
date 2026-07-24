"""
Market data downloader for Crypto Alpha Research Laboratory.
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
    Download historical OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker.

    start : str
        Start date (YYYY-MM-DD).

    end : str
        End date (YYYY-MM-DD).

    interval : str, default="1d"
        Data frequency.

    auto_adjust : bool, default=True
        Whether to adjust prices.

    Returns
    -------
    pandas.DataFrame
        Historical market data.

    Raises
    ------
    ValueError
        If no data is returned.
    """

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if df.empty:
        raise ValueError(
            f"No market data returned for '{ticker}'."
        )

    return df