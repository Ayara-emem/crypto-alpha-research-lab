

import pandas as pd

from crypto_alpha_lab.data.storage import load_prices


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
