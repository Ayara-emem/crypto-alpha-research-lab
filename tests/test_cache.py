import pandas as pd

from crypto_alpha_lab.data import (
    download_prices,
    save_prices,
    load_cached_prices,
)


def test_save_and_load_cache():

    df = download_prices(
        "BTC-USD",
        "2024-01-01",
        "2024-02-01",
    )

    save_prices(
        df,
        "BTC-USD",
    )

    cached = load_cached_prices(
        "BTC-USD",
    )

    assert isinstance(
        cached,
        pd.DataFrame,
    )


def test_cache_not_empty():

    df = load_cached_prices(
        "BTC-USD",
    )

    assert len(df) > 0