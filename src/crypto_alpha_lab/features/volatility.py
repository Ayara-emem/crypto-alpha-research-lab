"""
Volatility Features

Research-oriented volatility features built on top of APRL.

These features are designed for:

- alpha research
- market regime detection
- portfolio construction
- risk modelling
- signal generation
"""

from __future__ import annotations

import pandas as pd

from asset_pricing_lab.returns import (
    arithmetic_returns,
    log_returns,
)

from crypto_alpha_lab.dataset import ResearchDataset
from crypto_alpha_lab.features._utils import (
    _align_to_prices,
)

def rolling_volatility(
    dataset: ResearchDataset,
    window: int = 20,
) -> pd.Series:
    """
    Rolling volatility computed from arithmetic returns.

    Parameters
    ----------
    dataset
        Research dataset.

    window
        Rolling lookback window.

    Returns
    -------
    pandas.Series
        Rolling standard deviation of returns.
    """

    if window <= 0:
        raise ValueError("window must be positive.")

    close = dataset.prices["Close"]

    returns = _align_to_prices(
        arithmetic_returns(close),
        close,
    )

    return returns.rolling(window).std()

def realized_volatility(
    dataset: ResearchDataset,
    window: int = 20,
) -> pd.Series:
    """
    Rolling realized volatility computed from log returns.

    Parameters
    ----------
    dataset
        Research dataset.

    window
        Rolling lookback window.

    Returns
    -------
    pandas.Series
        Rolling standard deviation of log returns.
    """

    if window <= 0:
        raise ValueError("window must be positive.")

    close = dataset.prices["Close"]

    returns = _align_to_prices(
        log_returns(close),
        close,
    )

    return returns.rolling(window).std()

def volatility_ratio(
    dataset: ResearchDataset,
    short_window: int = 20,
    long_window: int = 60,
) -> pd.Series:
    """
    Ratio of short-term to long-term rolling volatility.

    Parameters
    ----------
    dataset
        Research dataset.

    short_window
        Short-term rolling window.

    long_window
        Long-term rolling window.

    Returns
    -------
    pandas.Series
        Volatility ratio.
    """

    if short_window <= 0:
        raise ValueError("short_window must be positive.")

    if long_window <= 0:
        raise ValueError("long_window must be positive.")

    if short_window >= long_window:
        raise ValueError(
            "short_window must be smaller than long_window."
        )

    short_vol = rolling_volatility(
        dataset,
        short_window,
    )

    long_vol = rolling_volatility(
        dataset,
        long_window,
    )

    return short_vol / long_vol

def volatility_zscore(
    dataset: ResearchDataset,
    volatility_window: int = 20,
    zscore_window: int = 60,
) -> pd.Series:
    """
    Standardized rolling volatility.

    Parameters
    ----------
    dataset
        Research dataset.

    volatility_window
        Window used to compute rolling volatility.

    zscore_window
        Window used for standardization.

    Returns
    -------
    pandas.Series
        Rolling volatility z-score.
    """

    if volatility_window <= 0:
        raise ValueError(
            "volatility_window must be positive."
        )

    if zscore_window <= 0:
        raise ValueError(
            "zscore_window must be positive."
        )

    volatility = rolling_volatility(
        dataset,
        volatility_window,
    )

    mean = volatility.rolling(zscore_window).mean()

    std = volatility.rolling(zscore_window).std()

    return (volatility - mean) / std

