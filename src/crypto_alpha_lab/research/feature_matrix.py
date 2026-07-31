"""
Feature Matrix Construction

Combines CARL research features into a date-aligned matrix
for quantitative analysis, signal research, and modeling.
"""

from __future__ import annotations

import pandas as pd

from crypto_alpha_lab.dataset import ResearchDataset

from crypto_alpha_lab.features.momentum import (
    price_momentum,
    rolling_return,
    log_momentum,
)

from crypto_alpha_lab.features.volatility import (
    rolling_volatility,
    realized_volatility,
    volatility_zscore,
)

from crypto_alpha_lab.features.volume import (
    relative_volume,
    volume_momentum,
    volume_zscore,
)

def build_feature_matrix(
    dataset: ResearchDataset,
    window: int = 20,
) -> pd.DataFrame:
    """
    Build a date-aligned quantitative research feature matrix.

    Parameters
    ----------
    dataset
        Validated CARL research dataset.

    window
        Lookback window used by feature calculations.

    Returns
    -------
    pandas.DataFrame
        Date-aligned research feature matrix.
    """

    if window <= 0:
        raise ValueError("window must be positive.")

    features = {
        "price_momentum": price_momentum(
            dataset,
            window=window,
        ),
        "rolling_return": rolling_return(
            dataset,
            window=window,
        ),
        "log_momentum": log_momentum(
            dataset,
            window=window,
        ),
        "rolling_volatility": rolling_volatility(
            dataset,
            window=window,
        ),
        "realized_volatility": realized_volatility(
            dataset,
            window=window,
        ),
        "relative_volume": relative_volume(
            dataset,
            window=window,
        ),
        "volume_momentum": volume_momentum(
            dataset,
            window=window,
        ),
        "volume_zscore": volume_zscore(
            dataset,
            window=window,
        ),
    }

    return pd.DataFrame(
        features,
        index=dataset.prices.index,
    )
