"""
Trend Features

Research-oriented trend features for CARL.

These features characterize the direction and structure
of cryptocurrency price trends.
"""

from __future__ import annotations

from asset_pricing_lab.returns import percentage_change
import pandas as pd

from crypto_alpha_lab.dataset import ResearchDataset


def simple_moving_average(
    dataset: ResearchDataset,
    window: int = 20,
) -> pd.Series:
    """
    Simple moving average of closing prices.

    Parameters
    ----------
    dataset
        Research dataset.

    window
        Rolling lookback window.

    Returns
    -------
    pandas.Series
        Simple moving average.
    """

    if window <= 0:
        raise ValueError("window must be positive.")
    close = dataset.prices["Close"]
    return close.rolling(window).mean()


def exponential_moving_average(
    dataset: ResearchDataset,
    span: int = 20,
) -> pd.Series:
    """
    Exponential moving average of closing prices.

    Parameters
    ----------
    dataset
        Research dataset.

    span
        Exponential moving-average span.

    Returns
    -------
    pandas.Series
        Exponential moving average.
    """

    if span <= 0:
        raise ValueError("span must be positive.")

    close = dataset.prices["Close"]

    return close.ewm(
        span=span,
        adjust=False,
    ).mean()

def price_to_moving_average(
    dataset: ResearchDataset,
    window: int = 20,
) -> pd.Series:
    """
    Closing price relative to its simple moving average.
    """

    if window <= 0:
        raise ValueError("window must be positive.")

    close = dataset.prices["Close"]

    moving_average = simple_moving_average(
        dataset,
        window=window,
    )

    return percentage_change(
    close,
    moving_average,)


def moving_average_spread(
    dataset: ResearchDataset,
    short_window: int = 20,
    long_window: int = 60,
) -> pd.Series:
    """
    Normalized spread between short- and long-term
    simple moving averages.
    """

    if short_window <= 0:
        raise ValueError(
            "short_window must be positive."
        )

    if long_window <= 0:
        raise ValueError(
            "long_window must be positive."
        )

    if short_window >= long_window:
        raise ValueError(
            "short_window must be smaller than long_window."
        )

    short_ma = simple_moving_average(
        dataset,
        window=short_window,
    )

    long_ma = simple_moving_average(
        dataset,
        window=long_window,
    )

    return percentage_change(
    short_ma,
    long_ma,
)