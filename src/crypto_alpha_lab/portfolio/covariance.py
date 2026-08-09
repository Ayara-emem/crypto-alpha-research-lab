"""
Covariance estimation interface for CARL.

CARL delegates covariance mathematics to APRL and provides
a research-oriented interface for selecting covariance
estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from asset_pricing_lab.covariance import (
    constant_correlation_target,
    diagonal_target,
    identity_target,
    ledoit_wolf_covariance,
    sample_covariance,
    shrink_covariance,
)


CovarianceMethod = Literal[
    "sample",
    "identity",
    "diagonal",
    "constant_correlation",
    "shrinkage",
    "ledoit_wolf",
]


@dataclass(frozen=True, slots=True)
class CovarianceEstimate:
    """
    Research covariance estimate.

    Attributes
    ----------
    matrix
        Estimated covariance matrix.

    method
        Covariance estimation method.

    assets
        Asset labels corresponding to matrix rows/columns.

    shrinkage
        Shrinkage intensity when applicable.
    """

    matrix: pd.DataFrame
    method: str
    assets: tuple[str, ...]
    shrinkage: float | None = None


class CovarianceEstimator:
    """
    Research-oriented covariance estimator.

    The estimator accepts a pandas DataFrame of asset returns
    and delegates the numerical covariance calculations to APRL.
    """

    SUPPORTED_METHODS = (
        "sample",
        "identity",
        "diagonal",
        "constant_correlation",
        "shrinkage",
        "ledoit_wolf",
    )

    def __init__(
        self,
        method: CovarianceMethod = "sample",
        shrinkage: float | None = None,
    ) -> None:

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported covariance method: {method!r}. "
                f"Supported methods are: "
                f"{self.SUPPORTED_METHODS}"
            )

        if shrinkage is not None:
            if not np.isfinite(shrinkage):
                raise ValueError(
                    "shrinkage must be finite."
                )

            if not 0.0 <= shrinkage <= 1.0:
                raise ValueError(
                    "shrinkage must lie in [0, 1]."
                )

        if method == "shrinkage" and shrinkage is None:
            raise ValueError(
                "shrinkage must be supplied when "
                "method='shrinkage'."
            )

        self.method = method
        self.shrinkage = shrinkage

    @staticmethod
    def _validate_returns(
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate and normalize return data.
        """

        if not isinstance(
            returns,
            pd.DataFrame,
        ):
            raise TypeError(
                "returns must be a pandas DataFrame."
            )

        if returns.empty:
            raise ValueError(
                "returns cannot be empty."
            )

        if returns.shape[1] < 2:
            raise ValueError(
                "returns must contain at least "
                "two assets."
            )

        if returns.columns.has_duplicates:
            raise ValueError(
                "returns cannot contain duplicate "
                "asset columns."
            )

        values = returns.to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "returns must contain only "
                "finite values."
            )

        return returns.astype(
    float
)

    @staticmethod
    def _as_dataframe(
        matrix: np.ndarray,
        assets: pd.Index,
    ) -> pd.DataFrame:
        """
        Convert an APRL covariance matrix into
        a labeled DataFrame.
        """

        matrix = np.asarray(
            matrix,
            dtype=float,
        )

        if matrix.ndim != 2:
            raise ValueError(
                "covariance matrix must be 2-dimensional."
            )

        if matrix.shape != (
            len(assets),
            len(assets),
        ):
            raise ValueError(
                "covariance matrix shape does not "
                "match the asset universe."
            )

        if not np.isfinite(matrix).all():
            raise ValueError(
                "covariance matrix must contain "
                "only finite values."
            )

        return pd.DataFrame(
            matrix,
            index=assets,
            columns=assets,
        )

    def _estimate_matrix(
        self,
        returns: pd.DataFrame,
    ) -> np.ndarray:
        """
        Dispatch covariance estimation to APRL.
        """

        values = returns.to_numpy(
            dtype=float
        )

        if self.method == "sample":
            return sample_covariance(
                values
            )

        sample = sample_covariance(
            values
        )

        if self.method == "identity":
            target = identity_target(
                sample
            )

            return target

        if self.method == "diagonal":
            target = diagonal_target(
                sample
            )

            return target

        if self.method == "constant_correlation":
            return constant_correlation_target(
                sample
            )

        if self.method == "shrinkage":
            assert self.shrinkage is not None

            target = constant_correlation_target(
                sample
            )

            return shrink_covariance(
                sample_covariance=sample,
                target_covariance=target,
                shrinkage=self.shrinkage,
            )

        if self.method == "ledoit_wolf":
            return ledoit_wolf_covariance(
                values
            )

        raise RuntimeError(
            "Unsupported covariance method."
        )

    def fit(
        self,
        returns: pd.DataFrame,
    ) -> CovarianceEstimate:
        """
        Estimate covariance from historical returns.

        Returns
        -------
        CovarianceEstimate
            Labeled covariance estimate suitable for
            portfolio construction and research.
        """

        returns = self._validate_returns(
            returns
        )

        matrix = self._estimate_matrix(
            returns
        )

        covariance = self._as_dataframe(
            matrix,
            returns.columns,
        )

        return CovarianceEstimate(
            matrix=covariance,
            method=self.method,
            assets=tuple(
                returns.columns.astype(str)
            ),
            shrinkage=self.shrinkage,
        )