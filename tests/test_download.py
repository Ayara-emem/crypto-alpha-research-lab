import pandas as pd

from crypto_alpha_lab.data import download_prices


def test_download_returns_dataframe():
    """Downloader returns a DataFrame."""

    df = download_prices(
        "BTC-USD",
        "2024-01-01",
        "2024-02-01",
    )

    assert isinstance(df, pd.DataFrame)


def test_download_not_empty():
    """Downloaded data should not be empty."""

    df = download_prices(
        "BTC-USD",
        "2024-01-01",
        "2024-02-01",
    )

    assert len(df) > 0


def test_download_contains_close():
    """Downloaded data should contain a Close column."""

    df = download_prices(
        "BTC-USD",
        "2024-01-01",
        "2024-02-01",
    )

    assert "Close" in df.columns