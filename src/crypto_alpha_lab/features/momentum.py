

from __future__ import annotations
import pandas as pd
import numpy as np

from asset_pricing_lab.returns import (
    arithmetic_returns,
    log_returns,
)

from crypto_alpha_lab.data.storage import load_prices

def price_momentum(
    ticker: str,
    start: str,
    end: str,
    window: int = 20,
):
    """
    Compute price momentum for a cryptocurrency.
    """
    if window <= 0:
        raise ValueError("window must be positive.")

    prices = load_prices(
        ticker=ticker,
        start=start,
        end=end,
    )

    close = prices["Close"]

    return close / close.shift(window) - 1

def rolling_return(
    ticker: str,
    start: str,
    end: str,
    window: int = 20,
):
    """
    Rolling cumulative arithmetic return.
    """
    if window <= 0:
        raise ValueError("window must be positive.")

    prices = load_prices(
        ticker=ticker,
        start=start,
        end=end,
    )
    returns = arithmetic_returns(prices["Close"])
    returns = returns.reindex(prices.index)
    return returns.rolling(window).sum()

def log_momentum(
    ticker: str,
    start: str,
    end: str,
    window: int = 20,
):
    """
    Rolling log-return momentum.
    """
    if window <= 0:
        raise ValueError("window must be positive.")

    prices = load_prices(
        ticker=ticker,
        start=start,
        end=end,
    )

    returns = log_returns(prices["Close"])
    returns = pd.Series(
    returns,
    index=prices.index[1:],
    name=prices["Close"].name,
    )
    returns = returns.reindex(prices.index)
    return returns.rolling(window).sum()

def relative_momentum(
    ticker: str,
    benchmark: str,
    start: str,
    end: str,
    window: int = 20,
):
    """
    Relative momentum against a benchmark asset.
    """
    if window <= 0:
        raise ValueError("window must be positive.")

    asset = price_momentum(
        ticker=ticker,
        start=start,
        end=end,
        window=window,
    )

    benchmark_momentum = price_momentum(
        ticker=benchmark,
        start=start,
        end=end,
        window=window,
    )

    return asset - benchmark_momentum

def _align_to_prices(
    values: np.ndarray | pd.Series,
    prices: pd.Series,
) -> pd.Series:
    """
    Convert APRL outputs into a pandas Series aligned with the
    original price index.
    """

