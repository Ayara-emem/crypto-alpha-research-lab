"""
Cross-sectional momentum research model.

The model estimates trailing momentum from the training
sample and converts the resulting cross-sectional ranking
into standardized signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CrossSectionalMomentumModel:
    """
    Cross-sectional momentum model.

    Parameters
    ----------
    lookback
        Number of observations used to estimate momentum.

    demean
        Whether to demean the cross-sectional signal.

    normalize
        Whether to normalize the cross-sectional signal
        by its absolute sum.

    Notes
    -----
    The model is intentionally simple and interpretable.
    The purpose is to provide a transparent first research
    model for CARL's walk-forward framework.
    """

    def __init__(
        self,
        lookback: int = 20,
        demean: bool = True,
        normalize: bool = True,
    ) -> None:

        if lookback <= 0:
            raise ValueError(
                "lookback must be positive."
            )

        self.lookback = lookback
        self.demean = demean
        self.normalize = normalize

        self._fitted = False
        self._signals: pd.Series | None = None

    def fit(
        self,
        train: pd.DataFrame,
    ) -> None:
        """
        Estimate momentum signals using training data only.

        Parameters
        ----------
        train
            Historical price observations.

        Notes
        -----
        The final ``lookback + 1`` observations are used
        to estimate trailing price momentum.
        """

        if not isinstance(
            train,
            pd.DataFrame,
        ):
            raise TypeError(
                "train must be a pandas DataFrame."
            )

        if train.empty:
            raise ValueError(
                "train cannot be empty."
            )

        if len(train) <= self.lookback:
            raise ValueError(
                "train must contain more observations "
                "than lookback."
            )

        if train.columns.empty:
            raise ValueError(
                "train must contain at least one asset."
            )

        if not np.isfinite(
            train.to_numpy(
                dtype=float,
            )
        ).all():
            raise ValueError(
                "train contains non-finite values."
            )

        if (train <= 0).any().any():
            raise ValueError(
                "prices must be strictly positive."
            )

        prices = train.iloc[
            -self.lookback - 1 :
        ]

        momentum = (
            prices.iloc[-1]
            / prices.iloc[0]
            - 1.0
        )

        momentum = momentum.astype(
            float,
        )

        if self.demean:
            momentum = (
                momentum
                - momentum.mean()
            )

        if self.normalize:

            gross = momentum.abs().sum()

            if gross > 0.0:
                momentum = (
                    momentum / gross
                )

        momentum.name = "signal"

        self._signals = momentum
        self._fitted = True

    def predict(
        self,
        test: pd.DataFrame,
    ) -> pd.Series:
        """
        Generate signals for unseen observations.

        The fitted cross-sectional signal is held constant
        throughout the supplied test period.

        Parameters
        ----------
        test
            Unseen test observations.

        Returns
        -------
        pandas.Series
            Asset-level cross-sectional signals.
        """

        if not self._fitted:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        if not isinstance(
            test,
            pd.DataFrame,
        ):
            raise TypeError(
                "test must be a pandas DataFrame."
            )

        if test.empty:
            raise ValueError(
                "test cannot be empty."
            )

        assert self._signals is not None

        missing = set(
            self._signals.index
        ) - set(
            test.columns
        )

        if missing:
            raise ValueError(
                "Test data is missing assets: "
                f"{sorted(missing)}"
            )

        signals = self._signals.copy()

        signals.name = "signal"

        return signals
    