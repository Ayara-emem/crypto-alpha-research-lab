import pandas as pd
import pytest
import numpy as np


from crypto_alpha_lab.data import (
    download_prices,
    validate_prices,
)


def test_validation_passes():
    """
    Valid BTC data should pass validation.
    """

    df = download_prices(
        "BTC-USD",
        "2024-01-01",
        "2024-02-01",
    )

    validate_prices(df)


def test_validation_empty_dataframe():

    with pytest.raises(ValueError):

        validate_prices(
            pd.DataFrame()
        )


def test_validation_missing_column():

    df = download_prices(
        "BTC-USD",
        "2024-01-01",
        "2024-02-01",
    )

    df = df.drop(columns=["Close"])

    with pytest.raises(ValueError):

        validate_prices(df)
        