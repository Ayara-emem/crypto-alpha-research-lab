"""
Target Engineering

Construction of forward-looking target variables for
predictive quantitative research.
"""

from __future__ import annotations

import pandas as pd

from crypto_alpha_lab.dataset import ResearchDataset


def future_return(
    dataset: ResearchDataset,
    horizon: int = 1,
) -> pd.Series:
    """
    Compute forward arithmetic returns.

    Parameters
    ----------
    dataset
        Validated research dataset.

    horizon
        Number of periods ahead.

    Returns
    -------
    pandas.Series
        Forward return aligned with today's observations.
    """

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    close = dataset.prices["Close"]

    return (
        close.shift(-horizon)
        .div(close)
        .sub(1.0)
    )