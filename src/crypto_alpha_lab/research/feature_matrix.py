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
)

from crypto_alpha_lab.features.volume import (
    relative_volume,
    volume_momentum,
    volume_zscore,
)

from crypto_alpha_lab.features.trend import (
    price_to_moving_average,
    moving_average_spread,
)


def build_feature_matrix(
    dataset: ResearchDataset,
    window: int = 20,
    trend_long_window: int = 60,
) -> pd.DataFrame:
    """
    Build a date-aligned quantitative research feature matrix.

    Parameters
    ----------
    dataset : ResearchDataset
        Validated CARL research dataset.

    window : int, default=20
        Primary lookback window used by momentum,
        volatility, volume, and short-term trend features.

    trend_long_window : int, default=60
        Long lookback window used for the moving-average
        trend spread.

    Returns
    -------
    pandas.DataFrame
        Date-aligned quantitative research feature matrix.

    Raises
    ------
    ValueError
        If window parameters are invalid.
    """

    if window <= 0:
        raise ValueError(
            "window must be positive."
        )

    if trend_long_window <= 0:
        raise ValueError(
            "trend_long_window must be positive."
        )

    if window >= trend_long_window:
        raise ValueError(
            "window must be smaller than trend_long_window."
        )

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
        "price_to_moving_average": price_to_moving_average(
            dataset,
            window=window,
        ),
        "moving_average_spread": moving_average_spread(
            dataset,
            short_window=window,
            long_window=trend_long_window,
        ),
    }

    return pd.DataFrame(
        features,
        index=dataset.prices.index,
    )