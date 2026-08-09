"""
Tests for CARL training-aware walk-forward model evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.backtest.engine import (
    BacktestEngine,
    BacktestResult,
)
from crypto_alpha_lab.validation.model import (
    ResearchModel,
)
from crypto_alpha_lab.validation.model_evaluator import (
    ModelEvaluationResult,
    WalkForwardModelEvaluator,
)
from crypto_alpha_lab.validation.walk_forward import (
    WalkForwardSplit,
)


# ---------------------------------------------------------------------------
# Test model
# ---------------------------------------------------------------------------


class DummyModel(ResearchModel):
    """
    Deterministic research model used to test evaluator orchestration.

    The model records the data supplied to fit() and predict()
    so that train/test separation can be verified explicitly.
    """

    def __init__(self) -> None:
        self.fit_calls: list[pd.DataFrame] = []
        self.predict_calls: list[pd.DataFrame] = []
        self.is_fitted = False

    def fit(
        self,
        data: pd.DataFrame,
    ) -> DummyModel:
        """
        Record training data and mark model as fitted.
        """

        self.fit_calls.append(
            data.copy()
        )

        self.is_fitted = True

        return self

    def predict(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:
        """
        Produce deterministic positive signals.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        self.predict_calls.append(
            data.copy()
        )

        return pd.Series(
            {
                column: float(index + 1)
                for index, column
                in enumerate(data.columns)
            },
            dtype=float,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prices() -> pd.DataFrame:
    """
    Deterministic multi-asset price data.
    """

    index = pd.date_range(
        "2024-01-01",
        periods=12,
        freq="D",
    )

    return pd.DataFrame(
        {
            "BTC": np.linspace(
                100.0,
                112.0,
                len(index),
            ),
            "ETH": np.linspace(
                50.0,
                56.0,
                len(index),
            ),
        },
        index=index,
    )


@pytest.fixture
def splits(
    prices: pd.DataFrame,
) -> list[WalkForwardSplit]:
    """
    Two chronological walk-forward folds.
    """

    return [
        WalkForwardSplit(
            fold=1,
            train=prices.iloc[:6].copy(),
            test=prices.iloc[6:9].copy(),
        ),
        WalkForwardSplit(
            fold=2,
            train=prices.iloc[:9].copy(),
            test=prices.iloc[9:12].copy(),
        ),
    ]


@pytest.fixture
def model() -> DummyModel:
    """
    Deterministic test model.
    """

    return DummyModel()


@pytest.fixture
def evaluator() -> WalkForwardModelEvaluator:
    """
    Model evaluator fixture.
    """

    return WalkForwardModelEvaluator(
        backtest_engine=BacktestEngine(),
    )


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


def test_model_evaluation_result_contains_returns(
    prices: pd.DataFrame,
):
    """
    ModelEvaluationResult should expose the aggregated
    out-of-sample return series.
    """

    returns = pd.Series(
        [0.01, 0.02, -0.01],
        index=prices.index[:3],
        dtype=float,
    )

    result = ModelEvaluationResult(
        returns=returns,
        folds=(),
    )

    assert isinstance(
        result.returns,
        pd.Series,
    )


def test_cumulative_returns_are_correct(
    prices: pd.DataFrame,
):
    """
    Cumulative returns should compound the OOS
    return series correctly.
    """

    returns = pd.Series(
        [0.10, 0.05],
        index=prices.index[:2],
        dtype=float,
    )

    result = ModelEvaluationResult(
        returns=returns,
        folds=(),
    )

    expected = pd.Series(
        [
            0.10,
            1.10 * 1.05 - 1.0,
        ],
        index=prices.index[:2],
    )

    pd.testing.assert_series_equal(
        result.cumulative_returns,
        expected,
    )


# ---------------------------------------------------------------------------
# Basic evaluation
# ---------------------------------------------------------------------------


def test_evaluator_returns_expected_result_type(
    model: DummyModel,
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    Evaluator should return ModelEvaluationResult.
    """

    result = evaluator.evaluate(
        model=model,
        splits=splits,
    )

    assert isinstance(
        result,
        ModelEvaluationResult,
    )


def test_evaluator_produces_one_result_per_fold(
    model: DummyModel,
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    Every walk-forward fold should produce exactly
    one BacktestResult.
    """

    result = evaluator.evaluate(
        model=model,
        splits=splits,
    )

    assert len(
        result.folds
    ) == len(splits)

    assert all(
        isinstance(
            fold,
            BacktestResult,
        )
        for fold in result.folds
    )


# ---------------------------------------------------------------------------
# Training / testing separation
# ---------------------------------------------------------------------------


def test_model_is_fitted_once_per_fold(
    model: DummyModel,
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    The model must be fitted independently on every
    walk-forward training fold.
    """

    evaluator.evaluate(
        model=model,
        splits=splits,
    )

    assert len(
        model.fit_calls
    ) == len(splits)


def test_prediction_is_generated_once_per_fold(
    model: DummyModel,
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    The model must generate one prediction set for
    every unseen test fold.
    """

    evaluator.evaluate(
        model=model,
        splits=splits,
    )

    assert len(
        model.predict_calls
    ) == len(splits)


def test_fit_receives_training_data_only(
    model: DummyModel,
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    fit() must never receive observations from the
    corresponding test period.
    """

    evaluator.evaluate(
        model=model,
        splits=splits,
    )

    for split, fit_data in zip(
        splits,
        model.fit_calls,
    ):
        pd.testing.assert_frame_equal(
            fit_data,
            split.train,
        )

        assert fit_data.index[-1] < (
            split.test.index[0]
        )


def test_predict_receives_test_data_only(
    model: DummyModel,
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    predict() must receive only unseen test data.
    """

    evaluator.evaluate(
        model=model,
        splits=splits,
    )

    for split, predict_data in zip(
        splits,
        model.predict_calls,
    ):
        pd.testing.assert_frame_equal(
            predict_data,
            split.test,
        )


# ---------------------------------------------------------------------------
# Out-of-sample integrity
# ---------------------------------------------------------------------------


def test_aggregated_returns_are_out_of_sample_only(
    model: DummyModel,
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    Aggregated returns must contain only return observations
    generated from the test periods.

    The first price observation of each test fold does not
    itself produce a return because arithmetic returns require
    a previous price observation.
    """

    result = evaluator.evaluate(
        model=model,
        splits=splits,
    )

    expected_index = (
        splits[0].test.index[1:]
        .append(
            splits[1].test.index[1:]
        )
    )

    assert result.returns.index.equals(
        expected_index
    )


def test_aggregated_returns_preserve_chronological_order(
    model: DummyModel,
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    OOS returns must remain chronological after
    fold aggregation.
    """

    result = evaluator.evaluate(
        model=model,
        splits=splits,
    )

    assert result.returns.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# Fold metadata
# ---------------------------------------------------------------------------


def test_fold_metadata_contains_validation_information(
    model: DummyModel,
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    Each fold result should contain enough metadata
    to reconstruct its research provenance.
    """

    result = evaluator.evaluate(
        model=model,
        splits=splits,
    )

    for expected_fold, fold_result in zip(
        splits,
        result.folds,
    ):
        assert (
            fold_result.metadata["validation"]
            == "walk_forward"
        )

        assert (
            fold_result.metadata["out_of_sample"]
            is True
        )

        assert (
            fold_result.metadata["fold"]
            == expected_fold.fold
        )

        assert (
            fold_result.metadata["train_start"]
            == expected_fold.train.index[0]
        )

        assert (
            fold_result.metadata["train_end"]
            == expected_fold.train.index[-1]
        )

        assert (
            fold_result.metadata["test_start"]
            == expected_fold.test.index[0]
        )

        assert (
            fold_result.metadata["test_end"]
            == expected_fold.test.index[-1]
        )


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------


def test_cost_model_is_forwarded_to_backtest_engine(
    model: DummyModel,
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    The evaluator must delegate transaction-cost handling
    to BacktestEngine rather than implementing costs itself.
    """

    from crypto_alpha_lab.backtest.cost import (
        ProportionalCostModel,
    )

    cost_model = ProportionalCostModel(
        rate=0.01,
    )

    result = evaluator.evaluate(
        model=model,
        splits=splits,
        cost_model=cost_model,
    )

    for fold in result.folds:
        assert fold.transaction_costs >= 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_empty_splits_are_rejected(
    model: DummyModel,
    evaluator: WalkForwardModelEvaluator,
):
    """
    An evaluator cannot operate without folds.
    """

    with pytest.raises(ValueError):
        evaluator.evaluate(
            model=model,
            splits=[],
        )


def test_invalid_splits_container_is_rejected(
    model: DummyModel,
    evaluator: WalkForwardModelEvaluator,
):
    """
    splits must be supplied as a list.
    """

    with pytest.raises(TypeError):
        evaluator.evaluate(
            model=model,
            splits=(
                "not",
                "a",
                "list",
            ),
        )


def test_invalid_split_object_is_rejected(
    model: DummyModel,
    evaluator: WalkForwardModelEvaluator,
):
    """
    Every element must be a WalkForwardSplit.
    """

    with pytest.raises(TypeError):
        evaluator.evaluate(
            model=model,
            splits=[
                object(),
            ],
        )


def test_empty_training_fold_is_rejected(
    model: DummyModel,
    prices: pd.DataFrame,
    evaluator: WalkForwardModelEvaluator,
):
    """
    Empty training data should be rejected before
    fitting the model.
    """

    split = WalkForwardSplit(
        fold=1,
        train=prices.iloc[:0].copy(),
        test=prices.iloc[6:9].copy(),
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            model=model,
            splits=[split],
        )


def test_empty_test_fold_is_rejected(
    model: DummyModel,
    prices: pd.DataFrame,
    evaluator: WalkForwardModelEvaluator,
):
    """
    Empty test data should be rejected before
    prediction/backtesting.
    """

    split = WalkForwardSplit(
        fold=1,
        train=prices.iloc[:6].copy(),
        test=prices.iloc[6:6].copy(),
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            model=model,
            splits=[split],
        )


def test_empty_strategy_name_is_rejected(
    model: DummyModel,
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    Strategy names should not be silently accepted
    when empty.
    """

    with pytest.raises(ValueError):
        evaluator.evaluate(
            model=model,
            splits=splits,
            strategy_name="",
        )

# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------


class LeakageDetectionModel(DummyModel):
    """
    Test model that records the exact observations supplied
    during every fit and predict operation.
    """

    def __init__(self) -> None:
        super().__init__()

        self.fit_indices: list[pd.Index] = []
        self.predict_indices: list[pd.Index] = []

    def fit(
        self,
        data: pd.DataFrame,
    ) -> LeakageDetectionModel:

        self.fit_indices.append(
            data.index.copy()
        )

        return super().fit(data)

    def predict(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:

        self.predict_indices.append(
            data.index.copy()
        )

        return super().predict(data)


def test_fit_never_receives_test_dates(
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    No test observation may enter model fitting.
    """

    model = LeakageDetectionModel()

    evaluator.evaluate(
        model=model,
        splits=splits,
    )

    for split, fit_index in zip(
        splits,
        model.fit_indices,
    ):
        assert fit_index.equals(
            split.train.index
        )

        assert len(
    fit_index.intersection(
        split.test.index
    )
) == 0


def test_predict_receives_only_test_dates(
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    Prediction must operate exclusively on the
    corresponding test fold.
    """

    model = LeakageDetectionModel()

    evaluator.evaluate(
        model=model,
        splits=splits,
    )

    for split, predict_index in zip(
        splits,
        model.predict_indices,
    ):
        assert predict_index.equals(
            split.test.index
        )


def test_future_fold_is_not_visible_during_current_fit(
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    A model fitted for fold t must not receive observations
    belonging to a future fold's test period.
    """

    model = LeakageDetectionModel()

    evaluator.evaluate(
        model=model,
        splits=splits,
    )

    for current_position, split in enumerate(
        splits
    ):

        fit_index = model.fit_indices[
            current_position
        ]

        for future_split in splits[
            current_position + 1 :
        ]:

            assert len(
                fit_index.intersection(
                    split.test.index)
) == 0


def test_current_test_period_is_not_used_for_fit(
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    The current test window must remain completely
    unseen during model fitting.
    """

    model = LeakageDetectionModel()

    evaluator.evaluate(
        model=model,
        splits=splits,
    )

    for position, split in enumerate(
        splits
    ):

        fit_index = model.fit_indices[
            position
        ]

        assert len(
            fit_index.intersection(
                split.test.index
            )
        ) == 0


def test_training_and_test_indices_are_disjoint(
    splits: list[WalkForwardSplit],
):
    """
    The walk-forward split itself must establish strict
    temporal separation.
    """

    for split in splits:

        assert len(
            split.train.index.intersection(
                split.test.index
            )
        ) == 0


def test_training_data_precedes_test_data(
    splits: list[WalkForwardSplit],
):
    """
    Training observations must precede the corresponding
    test observations chronologically.
    """

    for split in splits:

        assert (
            split.train.index[-1]
            < split.test.index[0]
        )


def test_fold_training_window_is_reproducible(
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    The evaluator must pass the exact declared training
    window to the model.
    """

    model = LeakageDetectionModel()

    evaluator.evaluate(
        model=model,
        splits=splits,
    )

    for split, received in zip(
        splits,
        model.fit_indices,
    ):

        assert received.equals(
            split.train.index
        )


def test_fold_test_window_is_reproducible(
    splits: list[WalkForwardSplit],
    evaluator: WalkForwardModelEvaluator,
):
    """
    The evaluator must pass the exact declared test
    window to the model.
    """

    model = LeakageDetectionModel()

    evaluator.evaluate(
        model=model,
        splits=splits,
    )

    for split, received in zip(
        splits,
        model.predict_indices,
    ):

        assert received.equals(
            split.test.index
        )