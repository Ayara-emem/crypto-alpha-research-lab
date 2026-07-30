"""
Momentum Features

Research-oriented momentum indicators for CARL.

These functions operate on a validated ResearchDataset
and return feature series suitable for alpha research,
machine learning, and signal generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_pricing_lab.returns import (
    arithmetic_returns,
    log_returns,
)

from crypto_alpha_lab.dataset import ResearchDataset

from crypto_alpha_lab.features._utils import (
    _align_to_prices,
)

def price_momentum(
    dataset: ResearchDataset,
    window: int = 20,
) -> pd.Series:
    """
    Percentage price momentum.
    """

    if window <= 0:
        raise ValueError("window must be positive.")

    close = dataset.prices["Close"]

    return close.div(close.shift(window)).sub(1.0)

def rolling_return(
    dataset: ResearchDataset,
    window: int = 20,
) -> pd.Series:
    """
    Rolling cumulative arithmetic return.
    """

    if window <= 0:
        raise ValueError("window must be positive.")

    close = dataset.prices["Close"]

    returns = _align_to_prices(
        arithmetic_returns(close),
        close,
    )

    return returns.rolling(window).sum()

def log_momentum(
    dataset: ResearchDataset,
    window: int = 20,
) -> pd.Series:
    """
    Rolling log-return momentum.
    """

    if window <= 0:
        raise ValueError("window must be positive.")

    close = dataset.prices["Close"]

    returns = _align_to_prices(
        log_returns(close),
        close,
    )

    return returns.rolling(window).sum()

def relative_momentum(
    dataset: ResearchDataset,
    benchmark: ResearchDataset,
    window: int = 20,
) -> pd.Series:
    """
    Relative momentum versus a benchmark.
    """

    return (
        price_momentum(dataset, window)
        - price_momentum(benchmark, window)
    )
