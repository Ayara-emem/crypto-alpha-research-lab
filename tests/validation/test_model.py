"""
Tests for the CARL ResearchModel contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.validation.model import (
    ResearchModel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def train_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BTC": [100.0, 101.0, 102.0],
            "ETH": [50.0, 51.0, 52.0],
        },
        index=pd.date_range(
            "2024-01-01",
            periods=3,
            freq="D",
        ),
    )


@pytest.fixture
def test_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BTC": [103.0, 104.0],
            "ETH": [53.0, 54.0],
        },
        index=pd.date_range(
            "2024-01-04",
            periods=2,
            freq="D",
        ),
    )


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class ValidResearchModel:
    """
    Minimal valid implementation of ResearchModel.
    """

    def __init__(self):
        self.fitted = False
        self.train_rows = None

    def fit(
        self,
        train: pd.DataFrame,
    ) -> None:
        self.fitted = True
        self.train_rows = len(train)

    def predict(
        self,
        test: pd.DataFrame,
    ) -> pd.Series:

        if not self.fitted:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        return pd.Series(
            1.0,
            index=test.index,
            name="signal",
        )


class IncompleteResearchModel:
    """
    Deliberately incomplete implementation.
    """

    def fit(
        self,
        train: pd.DataFrame,
    ) -> None:
        pass


# ---------------------------------------------------------------------------
# Protocol compatibility
# ---------------------------------------------------------------------------


def test_valid_model_satisfies_research_model_protocol(
    train_data,
    test_data,
):
    model = ValidResearchModel()

    research_model: ResearchModel = model

    research_model.fit(
        train_data,
    )

    predictions = research_model.predict(
        test_data,
    )

    assert isinstance(
        predictions,
        pd.Series,
    )


def test_fit_is_called_before_prediction(
    train_data,
    test_data,
):
    model = ValidResearchModel()

    with pytest.raises(
        RuntimeError,
    ):
        model.predict(
            test_data,
        )

    model.fit(
        train_data,
    )

    predictions = model.predict(
        test_data,
    )

    assert len(predictions) == len(
        test_data,
    )


# ---------------------------------------------------------------------------
# Training behavior
# ---------------------------------------------------------------------------


def test_model_receives_training_data_only(
    train_data,
):
    model = ValidResearchModel()

    model.fit(
        train_data,
    )

    assert model.train_rows == len(
        train_data,
    )


def test_model_fit_does_not_modify_training_data(
    train_data,
):
    original = train_data.copy()

    model = ValidResearchModel()

    model.fit(
        train_data,
    )

    pd.testing.assert_frame_equal(
        train_data,
        original,
    )


# ---------------------------------------------------------------------------
# Prediction behavior
# ---------------------------------------------------------------------------


def test_prediction_index_matches_test_index(
    train_data,
    test_data,
):
    model = ValidResearchModel()

    model.fit(
        train_data,
    )

    predictions = model.predict(
        test_data,
    )

    pd.testing.assert_index_equal(
        predictions.index,
        test_data.index,
    )


def test_prediction_length_matches_test_data(
    train_data,
    test_data,
):
    model = ValidResearchModel()

    model.fit(
        train_data,
    )

    predictions = model.predict(
        test_data,
    )

    assert len(predictions) == len(
        test_data,
    )


def test_prediction_values_are_finite(
    train_data,
    test_data,
):
    model = ValidResearchModel()

    model.fit(
        train_data,
    )

    predictions = model.predict(
        test_data,
    )

    assert np.isfinite(
        predictions.to_numpy(),
    ).all()


def test_prediction_is_series(
    train_data,
    test_data,
):
    model = ValidResearchModel()

    model.fit(
        train_data,
    )

    predictions = model.predict(
        test_data,
    )

    assert isinstance(
        predictions,
        pd.Series,
    )


# ---------------------------------------------------------------------------
# Structural contract
# ---------------------------------------------------------------------------


def test_model_has_fit_method():
    model = ValidResearchModel()

    assert callable(
        getattr(
            model,
            "fit",
            None,
        )
    )


def test_model_has_predict_method():
    model = ValidResearchModel()

    assert callable(
        getattr(
            model,
            "predict",
            None,
        )
    )


def test_incomplete_model_does_not_satisfy_protocol():
    model = IncompleteResearchModel()

    assert not hasattr(
        model,
        "predict",
    )


# ---------------------------------------------------------------------------
# Temporal discipline
# ---------------------------------------------------------------------------


def test_training_data_precedes_test_data(
    train_data,
    test_data,
):
    assert (
        train_data.index[-1]
        < test_data.index[0]
    )


def test_model_does_not_receive_test_data_during_fit(
    train_data,
    test_data,
):
    model = ValidResearchModel()

    model.fit(
        train_data,
    )

    assert model.train_rows == len(
        train_data,
    )

    assert model.train_rows != len(
        train_data
    ) + len(
        test_data
    )


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------


def test_model_starts_unfitted():
    model = ValidResearchModel()

    assert model.fitted is False


def test_model_becomes_fitted_after_fit(
    train_data,
):
    model = ValidResearchModel()

    model.fit(
        train_data,
    )

    assert model.fitted is True


def test_multiple_predictions_are_supported(
    train_data,
    test_data,
):
    model = ValidResearchModel()

    model.fit(
        train_data,
    )

    first = model.predict(
        test_data,
    )

    second = model.predict(
        test_data,
    )

    pd.testing.assert_series_equal(
        first,
        second,
    )
    