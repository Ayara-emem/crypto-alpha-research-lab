"""
Research model contracts.

Defines the minimal interface required for training-aware
walk-forward research in CARL.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class ResearchModel(Protocol):
    """
    Protocol for a trainable quantitative research model.

    A research model must learn exclusively from the training
    observations and subsequently generate signals for unseen
    observations.
    """

    def fit(
        self,
        train: pd.DataFrame,
    ) -> None:
        """
        Fit the model using training observations only.
        """
        ...

    def predict(
        self,
        test: pd.DataFrame,
    ) -> pd.Series:
        """
        Generate predictions/signals for unseen observations.
        """
        ...