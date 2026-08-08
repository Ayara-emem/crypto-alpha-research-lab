"""
Research experiment.

A ResearchExperiment encapsulates the complete
state of a quantitative research workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from crypto_alpha_lab.dataset import (
    ResearchDataset,
)

@dataclass(slots=True)
class ResearchExperiment:
    """
    Complete research experiment.

    This object carries every artifact generated
    throughout the CARL research pipeline.
    """

    dataset: ResearchDataset

    asset_universe: list[str] | None = None

    feature_matrix: pd.DataFrame | None = None

    target: pd.Series | None = None

    alpha_report: pd.DataFrame | None = None

    signals: pd.DataFrame | None = None

    portfolio: pd.Series | None = None
    asset_universe: list[str] | None = None

    price_columns: list[str] | None = None

    price_columns: list[str] | None = None

    metadata: dict = field(
        default_factory=dict,
    )

def set_asset_universe(
    self,
    assets: list[str],
) -> "ResearchExperiment":

    self.asset_universe = assets

    return self

    def validate(self) -> None:
        """
        Validate experiment consistency.
        """

        if self.feature_matrix is not None:

            if len(self.feature_matrix) != len(
                self.dataset.prices
            ):
                raise ValueError(
                    "Feature matrix length mismatch."
                )

        if self.target is not None:

            if len(self.target) != len(
                self.dataset.prices
            ):
                raise ValueError(
                    "Target length mismatch."
                )

    def add_metadata(
        self,
        key: str,
        value,
    ) -> None:
        """
        Store experiment metadata.
        """

        self.metadata[key] = value


    def get_metadata(
        self,
        key: str,
        default=None,
    ):
        """
        Retrieve experiment metadata.
        """

        return self.metadata.get(
            key,
            default,
        )

    def summary(
        self,
    ) -> pd.Series:
        """
        Summarize experiment contents.
        """

        return pd.Series(
            {
                "n_observations":
                    len(
                        self.dataset.prices
                    ),

                "n_features":
                    (
                        0
                        if self.feature_matrix is None
                        else self.feature_matrix.shape[1]
                    ),

                "target_available":
                    self.target is not None,

                "alpha_available":
                    self.alpha_report is not None,

                "signals_available":
                    self.signals is not None,

                "portfolio_available":
                    self.portfolio is not None,
            }
        )

def create_experiment(
    dataset: ResearchDataset,
) -> ResearchExperiment:
    """
    Create an empty experiment.
    """

    return ResearchExperiment(
        dataset=dataset,
    )

def set_features(
    self,
    feature_matrix: pd.DataFrame,
) -> "ResearchExperiment":
    """
    Attach engineered features.
    """

    self.feature_matrix = feature_matrix

    return self

def set_target(
    self,
    target: pd.Series,
) -> "ResearchExperiment":
    """
    Attach research target.
    """

    self.target = target

    return self


def set_alpha_report(
    self,
    report: pd.DataFrame,
) -> "ResearchExperiment":
    """
    Attach alpha report.
    """

    self.alpha_report = report

    return self


def set_signals(
    self,
    signals: pd.DataFrame,
) -> "ResearchExperiment":
    """
    Attach signals.
    """

    self.signals = signals

    return self

def set_portfolio(
    self,
    portfolio: pd.Series,
) -> "ResearchExperiment":
    """
    Attach portfolio weights.
    """

    self.portfolio = portfolio

    return self



