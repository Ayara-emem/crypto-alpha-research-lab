"""
Volume Features

Research-oriented volume features for CARL.

These features quantify trading activity and market participation
for alpha research and signal generation.
"""

from __future__ import annotations

import pandas as pd

from crypto_alpha_lab.dataset import ResearchDataset

def rolling_average_volume(
    dataset: ResearchDataset,
    window: int = 20,
) -> pd.Series:
    """
    Rolling average trading volume.
    """

    if window <= 0:
        raise ValueError("window must be positive.")

    volume = dataset.prices["Volume"]

    return volume.rolling(window).mean()

def relative_volume(
    dataset: ResearchDataset,
    window: int = 20,
) -> pd.Series:
    """
    Current volume divided by rolling average volume.
    """

    if window <= 0:
        raise ValueError("window must be positive.")

    volume = dataset.prices["Volume"]

    average = rolling_average_volume(
        dataset,
        window,
    )

    return volume / average

def volume_momentum(
    dataset: ResearchDataset,
    window: int = 20,
) -> pd.Series:
    """
    Percentage change in trading volume.
    """

    if window <= 0:
        raise ValueError("window must be positive.")

    volume = dataset.prices["Volume"]

    return volume.div(
        volume.shift(window)
    ).sub(1.0)

def volume_zscore(
    dataset: ResearchDataset,
    window: int = 20,
) -> pd.Series:
    """
    Standardized trading volume.
    """

    if window <= 0:
        raise ValueError("window must be positive.")

    volume = dataset.prices["Volume"]

    mean = volume.rolling(window).mean()

    std = volume.rolling(window).std()

    return (volume - mean) / std