"""
Research strategy orchestration.

Connects research signals to portfolio construction
and validation while delegating quantitative portfolio
mathematics to CARL/APRL portfolio APIs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crypto_alpha_lab.portfolio.construction import (
    equal_weight_portfolio,
    global_minimum_variance,
    signal_weighted_portfolio,
)

from crypto_alpha_lab.portfolio.validation import (
    validate_portfolio,
)


@dataclass(slots=True)
class ResearchStrategy:
    """
    Research strategy connecting signals to portfolio
    construction.

    Parameters
    ----------
    name
        Strategy name.

    signals
        Asset-level research signals.

    portfolio_method
        Portfolio construction method.

    covariance
        Optional covariance matrix required by
        global minimum-variance construction.

    max_gross_exposure
        Optional maximum portfolio gross exposure.

    min_net_exposure
        Optional minimum portfolio net exposure.

    max_net_exposure
        Optional maximum portfolio net exposure.
    """

    name: str

    signals: pd.Series

    portfolio_method: str

    covariance: pd.DataFrame | None = None

    max_gross_exposure: float | None = None

    min_net_exposure: float | None = None

    max_net_exposure: float | None = None

    def weights(self) -> pd.Series:
        """
        Construct and validate portfolio weights.

        Returns
        -------
        pandas.Series
            Validated portfolio weights.
        """

        if self.portfolio_method == "equal_weight":

            weights = equal_weight_portfolio(
                list(self.signals.index),
            )

        elif self.portfolio_method == "signal_weighted":

            weights = signal_weighted_portfolio(
                self.signals,
            )

        elif (
            self.portfolio_method
            == "global_minimum_variance"
        ):

            if self.covariance is None:
                raise ValueError(
                    "covariance is required for "
                    "global_minimum_variance."
                )

            weights = global_minimum_variance(
                self.covariance,
            )

        else:

            raise ValueError(
                "Unsupported portfolio method: "
                f"'{self.portfolio_method}'."
            )

        validate_portfolio(
            weights,
            assets=list(self.signals.index),
            max_gross_exposure=(
                self.max_gross_exposure
            ),
            min_net_exposure=(
                self.min_net_exposure
            ),
            max_net_exposure=(
                self.max_net_exposure
            ),
        )

        return weights