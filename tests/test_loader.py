import pandas as pd

from crypto_alpha_lab.data import load_prices


def test_loader_returns_dataframe():
    """Loader should return a DataFrame."""

    df = load_prices(
        ticker="BTC-USD",
        start="2024-01-01",
        end="2024-02-01",
    )

    assert isinstance(df, pd.DataFrame)


def test_loader_not_empty():
    """Loaded data should not be empty."""

    df = load_prices(
        ticker="BTC-USD",
        start="2024-01-01",
        end="2024-02-01",
    )

    assert len(df) > 0