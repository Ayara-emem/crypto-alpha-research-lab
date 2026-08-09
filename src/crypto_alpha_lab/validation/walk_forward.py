"""
Walk-forward validation utilities.

Provides chronological train/test splits for quantitative
research without random shuffling or future-information leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    """
    One chronological walk-forward train/test split.

    Attributes
    ----------
    train
        Training observations.

    test
        Out-of-sample test observations.

    fold
        Zero-based fold number.
    """

    train: pd.DataFrame

    test: pd.DataFrame

    fold: int


class WalkForwardValidator:
    """
    Generate chronological walk-forward train/test splits.

    Parameters
    ----------
    train_size
        Number of observations in the initial training window.

    test_size
        Number of observations in each test window.

    step_size
        Number of observations by which the window advances.
        Defaults to ``test_size``.

    expanding
        If True, the training window expands after each fold.
        If False, a rolling training window of fixed size is used.

    gap
        Number of observations between the end of the training
        window and beginning of the test window.
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        step_size: int | None = None,
        expanding: bool = True,
        gap: int = 0,
    ) -> None:

        if train_size <= 0:
            raise ValueError(
                "train_size must be positive."
            )

        if test_size <= 0:
            raise ValueError(
                "test_size must be positive."
            )

        if step_size is not None and step_size <= 0:
            raise ValueError(
                "step_size must be positive."
            )

        if gap < 0:
            raise ValueError(
                "gap must be non-negative."
            )

        self.train_size = train_size
        self.test_size = test_size
        self.step_size = (
            test_size
            if step_size is None
            else step_size
        )
        self.expanding = expanding
        self.gap = gap

    def split(
        self,
        data: pd.DataFrame,
    ) -> list[WalkForwardSplit]:
        """
        Generate chronological walk-forward splits.

        Parameters
        ----------
        data
            Time-ordered research observations.

        Returns
        -------
        list[WalkForwardSplit]
            Generated train/test splits.
        """

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "data cannot be empty."
            )

        if not data.index.is_monotonic_increasing:
            raise ValueError(
                "data index must be sorted "
                "in ascending chronological order."
            )

        if data.index.has_duplicates:
            raise ValueError(
                "data index must contain unique observations."
            )

        splits: list[WalkForwardSplit] = []

        n_observations = len(data)

        train_start = 0

        while True:

            train_end = (
                train_start
                + self.train_size
            )

            test_start = (
                train_end
                + self.gap
            )

            test_end = (
                test_start
                + self.test_size
            )

            if test_end > n_observations:
                break

            if self.expanding:
                actual_train_start = 0
            else:
                actual_train_start = train_start

            train = data.iloc[
                actual_train_start:train_end
            ].copy()

            test = data.iloc[
                test_start:test_end
            ].copy()

            splits.append(
                WalkForwardSplit(
                    train=train,
                    test=test,
                    fold=len(splits),
                )
            )

            train_start += self.step_size

        return splits

    def __iter__(self):
        """
        Prevent accidental iteration without data.

        Use ``split(data)`` explicitly so the data being
        validated remains visible in research code.
        """

        raise TypeError(
            "WalkForwardValidator is not directly iterable. "
            "Use validator.split(data)."
        )