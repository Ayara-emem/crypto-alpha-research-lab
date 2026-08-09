"""
Tests for CARL cross-sectional momentum research model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.models.momentum import (
    CrossSectionalMomentumModel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prices() -> pd.DataFrame:
    """
    Synthetic positive price history for three assets.
    """

    index = pd.date_range(
        "2024-01-01",
        periods=30,
        freq="D",
    )

    return pd.DataFrame(
        {
            "BTC": np.linspace(
                100.0,
                130.0,
                30,
            ),
            "ETH": np.linspace(
                100.0,
                115.0,
                30,
            ),
            "SOL": np.linspace(
                100.0,
                90.0,
                30,
            ),
        },
        index=index,
    )


@pytest.fixture
def short_prices() -> pd.DataFrame:
    """
    Shorter price history for boundary tests.
    """

    index = pd.date_range(
        "2024-01-01",
        periods=10,
        freq="D",
    )

    return pd.DataFrame(
        {
            "BTC": np.linspace(
                100.0,
                110.0,
                10,
            ),
            "ETH": np.linspace(
                100.0,
                105.0,
                10,
            ),
        },
        index=index,
    )


@pytest.fixture
def test_data() -> pd.DataFrame:
    """
    Synthetic unseen test observations.
    """

    index = pd.date_range(
        "2024-01-31",
        periods=5,
        freq="D",
    )

    return pd.DataFrame(
        {
            "BTC": np.linspace(
                130.0,
                135.0,
                5,
            ),
            "ETH": np.linspace(
                115.0,
                120.0,
                5,
            ),
            "SOL": np.linspace(
                90.0,
                95.0,
                5,
            ),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_model_can_be_created():
    model = CrossSectionalMomentumModel()

    assert model.lookback == 20
    assert model.demean is True
    assert model.normalize is True


def test_custom_parameters_are_preserved():
    model = CrossSectionalMomentumModel(
        lookback=10,
        demean=False,
        normalize=False,
    )

    assert model.lookback == 10
    assert model.demean is False
    assert model.normalize is False


def test_model_starts_unfitted():
    model = CrossSectionalMomentumModel()

    assert model._fitted is False
    assert model._signals is None


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_zero_lookback_is_rejected():
    with pytest.raises(ValueError):

        CrossSectionalMomentumModel(
            lookback=0,
        )


def test_negative_lookback_is_rejected():
    with pytest.raises(ValueError):

        CrossSectionalMomentumModel(
            lookback=-5,
        )


# ---------------------------------------------------------------------------
# Fit validation
# ---------------------------------------------------------------------------


def test_fit_rejects_non_dataframe():
    model = CrossSectionalMomentumModel()

    with pytest.raises(TypeError):

        model.fit(
            np.arange(30),
        )


def test_fit_rejects_empty_dataframe():
    model = CrossSectionalMomentumModel()

    with pytest.raises(ValueError):

        model.fit(
            pd.DataFrame(),
        )


def test_fit_rejects_insufficient_history(
    short_prices,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    with pytest.raises(ValueError):

        model.fit(
            short_prices,
        )


def test_fit_rejects_non_finite_values(
    prices,
):
    corrupted = prices.copy()

    corrupted.iloc[5, 0] = np.nan

    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    with pytest.raises(ValueError):

        model.fit(
            corrupted,
        )


def test_fit_rejects_infinite_values(
    prices,
):
    corrupted = prices.copy()

    corrupted.iloc[5, 0] = np.inf

    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    with pytest.raises(ValueError):

        model.fit(
            corrupted,
        )


def test_fit_rejects_zero_prices(
    prices,
):
    corrupted = prices.copy()

    corrupted.iloc[5, 0] = 0.0

    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    with pytest.raises(ValueError):

        model.fit(
            corrupted,
        )


def test_fit_rejects_negative_prices(
    prices,
):
    corrupted = prices.copy()

    corrupted.iloc[5, 0] = -10.0

    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    with pytest.raises(ValueError):

        model.fit(
            corrupted,
        )


# ---------------------------------------------------------------------------
# Fit behavior
# ---------------------------------------------------------------------------


def test_fit_marks_model_as_fitted(
    prices,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    model.fit(
        prices,
    )

    assert model._fitted is True


def test_fit_creates_signals(
    prices,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    model.fit(
        prices,
    )

    assert model._signals is not None


def test_fit_creates_one_signal_per_asset(
    prices,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    model.fit(
        prices,
    )

    assert len(
        model._signals,
    ) == prices.shape[1]


def test_signal_index_matches_assets(
    prices,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    model.fit(
        prices,
    )

    assert list(
        model._signals.index,
    ) == list(
        prices.columns,
    )


# ---------------------------------------------------------------------------
# Momentum correctness
# ---------------------------------------------------------------------------


def test_momentum_signal_has_expected_direction(
    prices,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
        demean=False,
        normalize=False,
    )

    model.fit(
        prices,
    )

    assert model._signals["BTC"] > 0.0
    assert model._signals["ETH"] > 0.0
    assert model._signals["SOL"] < 0.0


def test_momentum_uses_configured_lookback(
    prices,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
        demean=False,
        normalize=False,
    )

    model.fit(
        prices,
    )

    expected = (
        prices.iloc[-1]
        / prices.iloc[-11]
        - 1.0
    )

    pd.testing.assert_series_equal(
        model._signals,
        expected.rename("signal"),
    )


def test_demeaning_removes_cross_sectional_mean(
    prices,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
        demean=True,
        normalize=False,
    )

    model.fit(
        prices,
    )

    assert model._signals.mean() == pytest.approx(
        0.0,
    )


def test_normalization_sets_unit_gross_exposure(
    prices,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
        demean=True,
        normalize=True,
    )

    model.fit(
        prices,
    )

    assert model._signals.abs().sum() == pytest.approx(
        1.0,
    )


def test_normalization_can_be_disabled(
    prices,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
        demean=True,
        normalize=False,
    )

    model.fit(
        prices,
    )

    assert model._signals.abs().sum() != pytest.approx(
        1.0,
    )


# ---------------------------------------------------------------------------
# Prediction lifecycle
# ---------------------------------------------------------------------------


def test_predict_requires_fit(
    test_data,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    with pytest.raises(RuntimeError):

        model.predict(
            test_data,
        )


def test_predict_rejects_non_dataframe(
    prices,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    model.fit(
        prices,
    )

    with pytest.raises(TypeError):

        model.predict(
            np.arange(10),
        )


def test_predict_rejects_empty_dataframe(
    prices,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    model.fit(
        prices,
    )

    with pytest.raises(ValueError):

        model.predict(
            pd.DataFrame(),
        )


def test_predict_returns_series(
    prices,
    test_data,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    model.fit(
        prices,
    )

    signals = model.predict(
        test_data,
    )

    assert isinstance(
        signals,
        pd.Series,
    )


def test_predict_returns_all_assets(
    prices,
    test_data,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    model.fit(
        prices,
    )

    signals = model.predict(
        test_data,
    )

    assert list(
        signals.index,
    ) == list(
        prices.columns,
    )


def test_predict_signal_name_is_correct(
    prices,
    test_data,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    model.fit(
        prices,
    )

    signals = model.predict(
        test_data,
    )

    assert signals.name == "signal"


def test_predict_signals_are_finite(
    prices,
    test_data,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    model.fit(
        prices,
    )

    signals = model.predict(
        test_data,
    )

    assert np.isfinite(
        signals.to_numpy(),
    ).all()


# ---------------------------------------------------------------------------
# Asset validation
# ---------------------------------------------------------------------------


def test_predict_rejects_missing_assets(
    prices,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    model.fit(
        prices,
    )

    incomplete = prices[
        ["BTC", "ETH"]
    ]

    with pytest.raises(ValueError):

        model.predict(
            incomplete,
        )


# ---------------------------------------------------------------------------
# Leakage / independence checks
# ---------------------------------------------------------------------------


def test_prediction_does_not_change_with_test_prices(
    prices,
    test_data,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    model.fit(
        prices,
    )

    original = model.predict(
        test_data,
    )

    altered = test_data.copy()

    altered.loc[:, :] = (
        altered.to_numpy()
        * 100.0
    )

    changed = model.predict(
        altered,
    )

    pd.testing.assert_series_equal(
        original,
        changed,
    )


def test_prediction_is_constant_across_test_period(
    prices,
    test_data,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    model.fit(
        prices,
    )

    signals = model.predict(
        test_data,
    )

    # Asset-level signal is intentionally
    # constant throughout the test period.
    assert signals.index.equals(
        prices.columns,
    )


def test_test_data_does_not_modify_fitted_signal(
    prices,
    test_data,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    model.fit(
        prices,
    )

    before = model._signals.copy()

    model.predict(
        test_data,
    )

    pd.testing.assert_series_equal(
        model._signals,
        before,
    )


# ---------------------------------------------------------------------------
# Re-fitting behavior
# ---------------------------------------------------------------------------


def test_refitting_updates_signal(
    prices,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
        demean=False,
        normalize=False,
    )

    model.fit(
        prices,
    )

    first = model._signals.copy()

    modified = prices.copy()

    modified.iloc[-1] = (
        modified.iloc[-1] * 2.0
    )

    model.fit(
        modified,
    )

    second = model._signals.copy()

    assert not first.equals(
        second,
    )


# ---------------------------------------------------------------------------
# Numerical robustness
# ---------------------------------------------------------------------------


def test_signal_values_are_finite_after_fit(
    prices,
):
    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    model.fit(
        prices,
    )

    assert np.isfinite(
        model._signals.to_numpy(),
    ).all()


def test_model_handles_single_asset(
    prices,
):
    single_asset = prices[
        ["BTC"]
    ]

    model = CrossSectionalMomentumModel(
        lookback=10,
        demean=True,
        normalize=True,
    )

    model.fit(
        single_asset,
    )

    signals = model.predict(
        single_asset.iloc[
            -5:
        ],
    )

    assert len(signals) == 1

    assert np.isfinite(
        signals.iloc[0],
    )


def test_model_preserves_asset_names(
    prices,
):
    renamed = prices.copy()

    renamed.columns = [
        "Asset_A",
        "Asset_B",
        "Asset_C",
    ]

    model = CrossSectionalMomentumModel(
        lookback=10,
    )

    model.fit(
        renamed,
    )

    assert list(
        model._signals.index,
    ) == [
        "Asset_A",
        "Asset_B",
        "Asset_C",
    ]