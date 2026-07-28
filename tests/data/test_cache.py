from pathlib import Path

from crypto_alpha_lab.data.cache import (cache_path,
                                         cache_exists,
                                         cache_age,
                                         save_cache_prices,
)


def test_cache_path_returns_path():
    assert isinstance(cache_path("BTC-USD"), Path)

def test_cache_path_suffix():
    path = cache_path("BTC-USD")

    assert path.name == "BTC-USD.parquet"

from crypto_alpha_lab.data.cache import cache_exists


def test_cache_exists_returns_bool():
    assert isinstance(cache_exists("BTC-USD"), bool)

from crypto_alpha_lab.data.cache import cache_age


def test_cache_age_missing():
    assert cache_age("THIS-TICKER-SHOULD-NOT-EXIST") is None



