"""
Portfolio construction for quantitative research.

CARL orchestrates portfolio construction while
delegating portfolio mathematics to APRL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_pricing_lab.portfolio import (
    global_minimum_variance_portfolio,
)


def global_minimum_variance(
    covariance: pd.DataFrame,
) -> pd.Series:
    """
    Construct a global minimum-variance portfolio.
    """

    if not isinstance(
        covariance,
        pd.DataFrame,
    ):
        raise TypeError(
            "covariance must be a pandas DataFrame."
        )

    if covariance.empty:
        raise ValueError(
            "covariance cannot be empty."
        )

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            "covariance must be square."
        )

    if list(covariance.index) != list(
        covariance.columns
    ):
        raise ValueError(
            "covariance index and columns "
            "must contain the same assets "
            "in the same order."
        )

    if not np.isfinite(
        covariance.to_numpy()
    ).all():
        raise ValueError(
            "covariance contains non-finite values."
        )

    result = global_minimum_variance_portfolio(
        covariance.to_numpy(),
    )

    if isinstance(result, dict):

        raw_weights = result.get(
            "weights",
        )

        if raw_weights is None:
            raise ValueError(
                "APRL did not return portfolio weights."
            )

    elif isinstance(result, tuple):

        raw_weights = result[0]

    else:

        raw_weights = result

    weights = pd.Series(
        np.asarray(
            raw_weights,
            dtype=float,
        ).reshape(-1),
        index=covariance.index,
        name="weight",
    )

    if weights.isna().any():
        raise ValueError(
            "APRL returned incomplete portfolio weights."
        )

    return weights

def equal_weight_portfolio(
    assets: list[str],
) -> pd.Series:
    """
    Construct an equal-weight portfolio.
    """

    if not assets:
        raise ValueError(
            "assets cannot be empty."
        )

    if len(set(assets)) != len(assets):
        raise ValueError(
            "assets must be unique."
        )

    weight = 1.0 / len(assets)

    return pd.Series(
        weight,
        index=assets,
        dtype=float,
        name="weight",
    )


def signal_weighted_portfolio(
    signals: pd.Series,
    normalize: bool = True,
) -> pd.Series:
    """
    Construct portfolio weights from signals.
    """

    if not isinstance(
        signals,
        pd.Series,
    ):
        raise TypeError(
            "signals must be a pandas Series."
        )

    if signals.empty:
        raise ValueError(
            "signals cannot be empty."
        )

    if not np.isfinite(
        signals.to_numpy(
            dtype=float,
        )
    ).all():
        raise ValueError(
            "signals contain non-finite values."
        )

    weights = signals.astype(float).copy()

    if not normalize:
        weights.name = "weight"
        return weights

    gross_exposure = float(
        weights.abs().sum()
    )

    if gross_exposure == 0.0:
        raise ValueError(
            "Cannot normalize zero signals."
        )

    weights = (
        weights
        / gross_exposure
    )

    weights.name = "weight"

    return weights