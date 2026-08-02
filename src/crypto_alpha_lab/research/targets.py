"""
Target Engineering

Construction of forward-looking target variables for
predictive quantitative research.
"""

from __future__ import annotations

import numpy as np

import pandas as pd

from crypto_alpha_lab.dataset import ResearchDataset

from asset_pricing_lab.returns import (log_returns,
    log_ratio,
    percentage_change,
)


def future_return(
    dataset: ResearchDataset,
    horizon: int = 1,
) -> pd.Series:

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    close = dataset.prices["Close"]

    future_prices = close.shift(
        -horizon
    )

    return percentage_change(
        future_prices,
        close,
    )

def future_log_return(
    dataset: ResearchDataset,
    horizon: int = 1,
) -> pd.Series:

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    close = dataset.prices["Close"]

    future_prices = close.shift(
        -horizon
    )

    return log_ratio(
        future_prices,
        close,
    )

def future_direction(
    dataset: ResearchDataset,
    horizon: int = 1,
) -> pd.Series:
    """
    Compute a binary forward-direction target.

    Parameters
    ----------
    dataset
        Validated research dataset.

    horizon
        Prediction horizon.

    Returns
    -------
    pandas.Series
        Binary target where:

        1 = positive future return
        0 = zero or negative future return
    """

    returns = future_return(
        dataset,
        horizon=horizon,
    )

    result = (
        returns > 0
    ).astype("float")

    result[returns.isna()] = float("nan")

    return result

def future_volatility(
    dataset: ResearchDataset,
    horizon: int = 20,
) -> pd.Series:
    """
    Compute forward realized volatility.

    Parameters
    ----------
    dataset
        Validated research dataset.

    horizon
        Number of future periods over which
        realized volatility is measured.

    Returns
    -------
    pandas.Series
        Forward realized volatility.
    """

    if horizon <= 1:
        raise ValueError(
            "horizon must be greater than 1."
        )

    returns = future_return(
        dataset,
        horizon=1,
    )

    return (
        returns[::-1]
        .rolling(horizon)
        .std()
        [::-1]
    )
