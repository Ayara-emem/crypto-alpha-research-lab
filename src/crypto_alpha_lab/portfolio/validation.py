"""
Portfolio validation utilities.

Provides validation APIs for portfolio weights,
asset alignment, and exposure constraints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_weights(
    weights: pd.Series,
) -> None:
    """
    Validate portfolio weights.

    Parameters
    ----------
    weights
        Portfolio weights indexed by asset.

    Raises
    ------
    TypeError
        If weights is not a pandas Series.

    ValueError
        If weights are empty, non-finite, or duplicated.
    """

    if not isinstance(
        weights,
        pd.Series,
    ):
        raise TypeError(
            "weights must be a pandas Series."
        )

    if weights.empty:
        raise ValueError(
            "weights cannot be empty."
        )

    if weights.index.has_duplicates:
        raise ValueError(
            "weights index must contain unique assets."
        )

    values = weights.to_numpy(
        dtype=float,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "weights contain non-finite values."
        )


def validate_asset_alignment(
    weights: pd.Series,
    assets: list[str] | pd.Index,
) -> None:
    """
    Validate that portfolio weights align with
    the supplied asset universe.

    Parameters
    ----------
    weights
        Portfolio weights.

    assets
        Expected asset universe.

    Raises
    ------
    ValueError
        If assets are missing or extra assets exist.
    """

    validate_weights(
        weights,
    )

    expected = set(assets)
    actual = set(weights.index)

    missing = expected - actual
    extra = actual - expected

    if missing:
        raise ValueError(
            "Missing portfolio weights for assets: "
            f"{sorted(missing)}"
        )

    if extra:
        raise ValueError(
            "Portfolio contains assets outside "
            f"the asset universe: {sorted(extra)}"
        )


def validate_gross_exposure(
    weights: pd.Series,
    max_gross_exposure: float = 1.0,
) -> None:
    """
    Validate portfolio gross exposure.

    Gross exposure is defined as:

        sum(abs(weights))

    Parameters
    ----------
    weights
        Portfolio weights.

    max_gross_exposure
        Maximum permitted gross exposure.

    Raises
    ------
    ValueError
        If the exposure exceeds the specified limit.
    """

    validate_weights(
        weights,
    )

    if max_gross_exposure < 0:
        raise ValueError(
            "max_gross_exposure must be non-negative."
        )

    gross_exposure = float(
        weights.abs().sum()
    )

    if gross_exposure > (
        max_gross_exposure + 1e-12
    ):
        raise ValueError(
            "Portfolio gross exposure "
            f"{gross_exposure:.12f} exceeds "
            f"maximum allowed exposure "
            f"{max_gross_exposure:.12f}."
        )


def validate_net_exposure(
    weights: pd.Series,
    min_net_exposure: float | None = None,
    max_net_exposure: float | None = None,
) -> None:
    """
    Validate portfolio net exposure.

    Net exposure is defined as:

        sum(weights)

    Parameters
    ----------
    weights
        Portfolio weights.

    min_net_exposure
        Optional minimum permitted net exposure.

    max_net_exposure
        Optional maximum permitted net exposure.
    """

    validate_weights(
        weights,
    )

    if (
        min_net_exposure is not None
        and max_net_exposure is not None
        and min_net_exposure > max_net_exposure
    ):
        raise ValueError(
            "min_net_exposure cannot exceed "
            "max_net_exposure."
        )

    net_exposure = float(
        weights.sum()
    )

    tolerance = 1e-12
    if (
    min_net_exposure is not None
    and net_exposure
    < min_net_exposure - tolerance):

        raise ValueError(
        "Portfolio net exposure "
        f"{net_exposure:.12f} is below "
        f"minimum allowed exposure "
        f"{min_net_exposure:.12f}."
    )

    if (
    max_net_exposure is not None
    and net_exposure
    > max_net_exposure + tolerance
):
        raise ValueError(
        "Portfolio net exposure "
        f"{net_exposure:.12f} exceeds "
        f"maximum allowed exposure "
        f"{max_net_exposure:.12f}."
    )

def validate_portfolio(
    weights: pd.Series,
    assets: list[str] | pd.Index | None = None,
    max_gross_exposure: float | None = None,
    min_net_exposure: float | None = None,
    max_net_exposure: float | None = None,
) -> None:
    """
    Run complete portfolio validation.

    Parameters
    ----------
    weights
        Portfolio weights.

    assets
        Optional expected asset universe.

    max_gross_exposure
        Optional maximum gross exposure.

    min_net_exposure
        Optional minimum net exposure.

    max_net_exposure
        Optional maximum net exposure.
    """

    validate_weights(
        weights,
    )

    if assets is not None:
        validate_asset_alignment(
            weights,
            assets,
        )

    if max_gross_exposure is not None:
        validate_gross_exposure(
            weights,
            max_gross_exposure=max_gross_exposure,
        )

    if (
        min_net_exposure is not None
        or max_net_exposure is not None
    ):
        validate_net_exposure(
            weights,
            min_net_exposure=min_net_exposure,
            max_net_exposure=max_net_exposure,
        )