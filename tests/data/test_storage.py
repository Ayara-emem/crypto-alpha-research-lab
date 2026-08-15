import pandas as pd

from crypto_alpha_lab.data.storage import load_prices


def _valid_prices(
    start: str,
    periods: int,
) -> pd.DataFrame:

    index = pd.date_range(
        start,
        periods=periods,
        freq="D",
    )

    return pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 1_000_000.0,
        },
        index=index,
    )


def test_load_prices_returns_dataframe():

    prices = load_prices(
        ticker="BTC-USD",
        start="2020-01-01",
        end="2020-12-31",
    )

    assert isinstance(
        prices,
        pd.DataFrame,
    )


def test_load_prices_returns_flat_columns():

    prices = load_prices(
        ticker="BTC-USD",
        start="2020-01-01",
        end="2020-12-31",
    )

    assert not isinstance(
        prices.columns,
        pd.MultiIndex,
    )


def test_load_prices_contains_required_columns():

    prices = load_prices(
        ticker="BTC-USD",
        start="2020-01-01",
        end="2020-12-31",
    )

    expected = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    assert expected.issubset(
        prices.columns
    )


def test_load_prices_returns_sorted_index():

    prices = load_prices(
        ticker="BTC-USD",
        start="2020-01-01",
        end="2020-12-31",
    )

    assert prices.index.is_monotonic_increasing


def test_load_prices_has_unique_dates():

    prices = load_prices(
        ticker="BTC-USD",
        start="2020-01-01",
        end="2020-12-31",
    )

    assert not prices.index.has_duplicates


def test_compatible_cache_is_used(
    monkeypatch,
):

    cached = _valid_prices(
        start="2020-01-01",
        periods=365,
    )

    downloaded = False

    def fake_download(**kwargs):
        nonlocal downloaded
        downloaded = True
        return _valid_prices(
            start="2020-01-01",
            periods=365,
        )

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.cache_exists",
        lambda ticker: True,
    )

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.load_cached_prices",
        lambda ticker: cached,
    )

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.download_prices",
        fake_download,
    )

    result = load_prices(
        ticker="TEST",
        start="2020-01-01",
        end="2020-12-31",
    )

    assert not downloaded
    assert result.equals(cached)


def test_cache_starting_too_late_triggers_download(
    monkeypatch,
):

    cached = _valid_prices(
        start="2021-01-01",
        periods=365,
    )

    fresh = _valid_prices(
        start="2020-01-01",
        periods=731,
    )

    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return fresh

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.cache_exists",
        lambda ticker: True,
    )

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.load_cached_prices",
        lambda ticker: cached,
    )

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.download_prices",
        fake_download,
    )

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.save_cache_prices",
        lambda prices, ticker: None,
    )

    result = load_prices(
        ticker="TEST",
        start="2020-01-01",
        end="2021-12-31",
    )

    assert len(calls) == 1
    assert result.index.min() == pd.Timestamp(
        "2020-01-01"
    )


def test_cache_ending_too_early_triggers_download(
    monkeypatch,
):

    cached = _valid_prices(
        start="2020-01-01",
        periods=100,
    )

    fresh = _valid_prices(
        start="2020-01-01",
        periods=365,
    )

    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return fresh

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.cache_exists",
        lambda ticker: True,
    )

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.load_cached_prices",
        lambda ticker: cached,
    )

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.download_prices",
        fake_download,
    )

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.save_cache_prices",
        lambda prices, ticker: None,
    )

    result = load_prices(
        ticker="TEST",
        start="2020-01-01",
        end="2020-12-31",
    )

    assert len(calls) == 1
    assert result.index.max() == pd.Timestamp(
        "2020-12-30"
    )


def test_refresh_forces_download(
    monkeypatch,
):

    cached = _valid_prices(
        start="2020-01-01",
        periods=365,
    )

    fresh = _valid_prices(
        start="2021-01-01",
        periods=365,
    )

    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return fresh

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.cache_exists",
        lambda ticker: True,
    )

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.load_cached_prices",
        lambda ticker: cached,
    )

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.download_prices",
        fake_download,
    )

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.save_cache_prices",
        lambda prices, ticker: None,
    )

    result = load_prices(
        ticker="TEST",
        start="2021-01-01",
        end="2021-12-31",
        refresh=True,
    )

    assert len(calls) == 1
    assert result.equals(fresh)


def test_exact_cache_boundaries_are_accepted(
    monkeypatch,
):

    cached = _valid_prices(
        start="2021-01-01",
        periods=365,
    )

    def fail_download(**kwargs):
        raise AssertionError(
            "Download should not occur for a compatible cache."
        )

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.cache_exists",
        lambda ticker: True,
    )

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.load_cached_prices",
        lambda ticker: cached,
    )

    monkeypatch.setattr(
        "crypto_alpha_lab.data.storage.download_prices",
        fail_download,
    )

    result = load_prices(
        ticker="TEST",
        start="2021-01-01",
        end="2022-01-01",
    )

    assert result.equals(cached)