from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crypto_alpha_lab.data.storage import load_prices


@dataclass(slots=True)
class ResearchDataset:
    """
    Validated market dataset used throughout the
    quantitative research pipeline.

    Parameters
    ----------
    prices : pandas.DataFrame
        Validated OHLCV price data.
    """

    prices: pd.DataFrame

    @classmethod
    def load(
        cls,
        ticker: str,
        start: str,
        end: str,
    ) -> "ResearchDataset":
        """
        Load a validated research dataset.

        Parameters
        ----------
        ticker : str
            Asset ticker.

        start : str
            Start date.

        end : str
            End date.

        Returns
        -------
        ResearchDataset
            Dataset containing validated OHLCV prices.
        """
        prices = load_prices(
            ticker=ticker,
            start=start,
            end=end,
        )

        return cls(prices=prices)