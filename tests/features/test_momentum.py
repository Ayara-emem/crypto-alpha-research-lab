import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.features.momentum import (
    price_momentum,
    rolling_return,
    log_momentum,
    relative_momentum,
)

@pytest.fixture
def sample_prices():
    dates = pd.date_range("2024-01-01", periods=10)

    return pd.DataFrame(
        {
            "Close": np.arange(100, 110, dtype=float)
        },
        index=dates,
    )

def test_price_momentum_shape(sample_prices, monkeypatch):
    monkeypatch.setattr(
        "crypto_alpha_lab.features.momentum.load_prices",
        lambda **kwargs: sample_prices,
    )

    result = price_momentum(
        ticker="BTC-USD",
        start="2024-01-01",
        end="2024-01-10",
        window=3,
    )

    assert len(result) == len(sample_prices)

@pytest.mark.parametrize("window", [0, -1, -5])
def test_price_momentum_invalid_window(window):
    with pytest.raises(ValueError):
        price_momentum(
            ticker="BTC-USD",
            start="2024-01-01",
            end="2024-01-10",
            window=window,
        )

def test_rolling_return_shape(sample_prices, monkeypatch):
    monkeypatch.setattr(
        "crypto_alpha_lab.features.momentum.load_prices",
        lambda **kwargs: sample_prices,
    )

    result = rolling_return(
        ticker="BTC-USD",
        start="2024-01-01",
        end="2024-01-10",
        window=3,
    )

    assert len(result) == len(sample_prices)

def test_log_momentum_shape(sample_prices, monkeypatch):
    monkeypatch.setattr(
        "crypto_alpha_lab.features.momentum.load_prices",
        lambda **kwargs: sample_prices,
    )

    result = log_momentum(
        ticker="BTC-USD",
        start="2024-01-01",
        end="2024-01-10",
        window=3,
    )

    assert len(result) == len(sample_prices)


def test_relative_momentum_zero_when_equal(sample_prices, monkeypatch):
    monkeypatch.setattr(
        "crypto_alpha_lab.features.momentum.load_prices",
        lambda **kwargs: sample_prices,
    )

    result = relative_momentum(
        ticker="BTC-USD",
        benchmark="ETH-USD",
        start="2024-01-01",
        end="2024-01-10",
        window=3,
    )

    np.testing.assert_allclose(
        result.dropna(),
        0.0,
    )


def test_price_momentum_nan_behavior(sample_prices, monkeypatch):
    monkeypatch.setattr(
        "crypto_alpha_lab.features.momentum.load_prices",
        lambda **kwargs: sample_prices,
    )

    window = 4

    result = price_momentum(
        ticker="BTC-USD",
        start="2024-01-01",
        end="2024-01-10",
        window=window,
    )

    assert result.iloc[:window].isna().all()

def test_price_momentum_values(sample_prices, monkeypatch):
    monkeypatch.setattr(
        "crypto_alpha_lab.features.momentum.load_prices",
        lambda **kwargs: sample_prices,
    )

    result = price_momentum(
        ticker="BTC-USD",
        start="2024-01-01",
        end="2024-01-10",
        window=2,
    )

    expected = sample_prices["Close"] / sample_prices["Close"].shift(2) - 1

    pd.testing.assert_series_equal(
        result,
        expected,
        check_names=False,
    )
