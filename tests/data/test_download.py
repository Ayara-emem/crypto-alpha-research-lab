"""
Unit tests for the market data downloader.
"""

import pandas as pd
import pytest

from crypto_alpha_lab.data.download import download_prices

from crypto_alpha_lab.data.cache import (save_cache_prices,
                                         cache_exists)


START_DATE = "2020-01-01"
END_DATE = "2020-12-31"

def test_download_returns_dataframe():
    """
    download_prices should return a pandas DataFrame.
    """

    prices = download_prices(
        ticker="BTC-USD",
        start=START_DATE,
        end=END_DATE,
    )

    assert isinstance(prices, pd.DataFrame)

def test_download_returns_non_empty_dataframe():
    """
    Downloaded market data should not be empty.
    """

    prices = download_prices(
        ticker="BTC-USD",
        start=START_DATE,
        end=END_DATE,
    )

    assert not prices.empty

def test_download_returns_datetime_index():
    """
    Downloaded prices should use a DatetimeIndex.
    """

    prices = download_prices(
        ticker="BTC-USD",
        start=START_DATE,
        end=END_DATE,
    )

    assert isinstance(
        prices.index,
        pd.DatetimeIndex,
    )

def test_download_contains_close_prices():
    """
    Downloaded data should include Close prices.
    """

    prices = download_prices(
        ticker="BTC-USD",
        start=START_DATE,
        end=END_DATE,
    )

    if isinstance(prices.columns, pd.MultiIndex):
        assert "Close" in prices.columns.get_level_values(0)
    else:
        assert "Close" in prices.columns

def test_invalid_ticker_raises_value_error():
    """
    Invalid tickers should raise ValueError.
    """

    with pytest.raises(ValueError):

        download_prices(
            ticker="THIS-TICKER-DOES-NOT-EXIST",
            start=START_DATE,
            end=END_DATE,
        )

def test_download_returns_sorted_index():
    """
    Downloaded prices should be ordered by date.
    """

    prices = download_prices(
        ticker="BTC-USD",
        start=START_DATE,
        end=END_DATE,
    )

    assert prices.index.is_monotonic_increasing

def test_download_has_unique_dates():
    """
    Downloaded prices should not contain duplicate dates.
    """

    prices = download_prices(
        ticker="BTC-USD",
        start=START_DATE,
        end=END_DATE,
    )

    assert not prices.index.has_duplicates

def test_save_prices_creates_cache():

    prices = download_prices(
        ticker="BTC-USD",
        start="2020-01-01",
        end="2020-12-31",
    )

    save_cache_prices(
        prices,
        "BTC-USD",
    )

    assert cache_exists("BTC-USD")