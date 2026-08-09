"""
Tests for CARL walk-forward validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.validation.walk_forward import (
    WalkForwardSplit,
    WalkForwardValidator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data() -> pd.DataFrame:
    """Chronological research observations."""

    index = pd.date_range(
        "2024-01-01",
        periods=30,
        freq="D",
    )

    return pd.DataFrame(
        {
            "feature": np.arange(30, dtype=float),
            "target": np.arange(
                100,
                130,
                dtype=float,
            ),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_validator_can_be_created():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    assert validator.train_size == 10
    assert validator.test_size == 5
    assert validator.step_size == 5
    assert validator.expanding is True
    assert validator.gap == 0


def test_custom_step_size_is_preserved():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        step_size=2,
    )

    assert validator.step_size == 2


def test_rolling_window_can_be_created():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        expanding=False,
    )

    assert validator.expanding is False


def test_gap_is_preserved():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        gap=3,
    )

    assert validator.gap == 3


# ---------------------------------------------------------------------------
# Basic splitting
# ---------------------------------------------------------------------------


def test_split_returns_list(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    splits = validator.split(data)

    assert isinstance(
        splits,
        list,
    )


def test_split_returns_walk_forward_objects(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    splits = validator.split(data)

    assert all(
        isinstance(
            split,
            WalkForwardSplit,
        )
        for split in splits
    )


def test_expected_number_of_splits(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    splits = validator.split(data)

    # 10 train + 5 test, advancing by 5:
    # folds start at 0, 5, 10, 15
    assert len(splits) == 4


def test_fold_numbers_are_sequential(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    splits = validator.split(data)

    assert [
        split.fold
        for split in splits
    ] == [
        0,
        1,
        2,
        3,
    ]


# ---------------------------------------------------------------------------
# Train/test sizes
# ---------------------------------------------------------------------------


def test_initial_train_size_is_correct(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    splits = validator.split(data)

    assert len(splits) > 0

    assert len(
        splits[0].train,
    ) == 10


def test_test_size_is_correct(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    splits = validator.split(data)

    assert all(
        len(split.test) == 5
        for split in splits
    )


def test_train_and_test_columns_are_preserved(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    split = validator.split(data)[0]

    assert list(
        split.train.columns,
    ) == [
        "feature",
        "target",
    ]

    assert list(
        split.test.columns,
    ) == [
        "feature",
        "target",
    ]


# ---------------------------------------------------------------------------
# Chronological integrity
# ---------------------------------------------------------------------------


def test_train_precedes_test(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    splits = validator.split(data)

    for split in splits:

        assert (
            split.train.index[-1]
            < split.test.index[0]
        )


def test_no_train_test_overlap(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    splits = validator.split(data)

    for split in splits:

        assert set(
            split.train.index
        ).isdisjoint(
            set(split.test.index)
        )


def test_test_windows_are_chronological(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    splits = validator.split(data)

    for previous, current in zip(
        splits,
        splits[1:],
    ):
        assert (
            previous.test.index[-1]
            < current.test.index[0]
        )


# ---------------------------------------------------------------------------
# Expanding-window behavior
# ---------------------------------------------------------------------------


def test_expanding_window_starts_at_first_observation(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        expanding=True,
    )

    splits = validator.split(data)

    for split in splits:

        assert (
            split.train.index[0]
            == data.index[0]
        )


def test_expanding_training_window_grows(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        expanding=True,
    )

    splits = validator.split(data)

    assert [
        len(split.train)
        for split in splits
    ] == [
        10,
        15,
        20,
        25,
    ]


def test_expanding_training_end_moves_forward(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        expanding=True,
    )

    splits = validator.split(data)

    assert [
        split.train.index[-1]
        for split in splits
    ] == [
        data.index[9],
        data.index[14],
        data.index[19],
        data.index[24],
    ]


# ---------------------------------------------------------------------------
# Rolling-window behavior
# ---------------------------------------------------------------------------


def test_rolling_training_window_remains_fixed(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        expanding=False,
    )

    splits = validator.split(data)

    assert [
        len(split.train)
        for split in splits
    ] == [
        10,
        10,
        10,
        10,
    ]


def test_rolling_training_window_moves_forward(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        expanding=False,
    )

    splits = validator.split(data)

    assert [
        split.train.index[0]
        for split in splits
    ] == [
        data.index[0],
        data.index[5],
        data.index[10],
        data.index[15],
    ]


# ---------------------------------------------------------------------------
# Step size
# ---------------------------------------------------------------------------


def test_custom_step_size_changes_fold_spacing(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        step_size=2,
    )

    splits = validator.split(data)

    assert [
        split.test.index[0]
        for split in splits[:3]
    ] == [
        data.index[10],
        data.index[12],
        data.index[14],
    ]


# ---------------------------------------------------------------------------
# Gap / embargo
# ---------------------------------------------------------------------------


def test_gap_creates_embargo_between_train_and_test(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        gap=3,
    )

    split = validator.split(data)[0]

    assert (
        split.train.index[-1]
        == data.index[9]
    )

    assert (
        split.test.index[0]
        == data.index[13]
    )


def test_gap_observations_are_not_in_train_or_test(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        gap=3,
    )

    split = validator.split(data)[0]

    gap_index = data.index[10:13]

    assert set(
        gap_index
    ).isdisjoint(
        set(split.train.index)
    )

    assert set(
        gap_index
    ).isdisjoint(
        set(split.test.index)
    )


def test_gap_is_applied_to_every_fold(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        gap=2,
    )

    splits = validator.split(data)

    for split in splits:

        position_train_end = (
            data.index.get_loc(
                split.train.index[-1]
            )
        )

        position_test_start = (
            data.index.get_loc(
                split.test.index[0]
            )
        )

        assert (
            position_test_start
            - position_train_end
            - 1
            == 2
        )


# ---------------------------------------------------------------------------
# Incomplete final folds
# ---------------------------------------------------------------------------


def test_incomplete_final_test_window_is_excluded(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=7,
    )

    splits = validator.split(data)

    assert all(
        len(split.test) == 7
        for split in splits
    )


def test_no_partial_test_window_is_returned(
    data,
):
    validator = WalkForwardValidator(
        train_size=20,
        test_size=7,
    )

    splits = validator.split(data)

    for split in splits:

        assert len(split.test) == 7


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_train_size_must_be_positive():
    with pytest.raises(ValueError):

        WalkForwardValidator(
            train_size=0,
            test_size=5,
        )


def test_negative_train_size_is_rejected():
    with pytest.raises(ValueError):

        WalkForwardValidator(
            train_size=-1,
            test_size=5,
        )


def test_test_size_must_be_positive():
    with pytest.raises(ValueError):

        WalkForwardValidator(
            train_size=10,
            test_size=0,
        )


def test_negative_test_size_is_rejected():
    with pytest.raises(ValueError):

        WalkForwardValidator(
            train_size=10,
            test_size=-5,
        )


def test_step_size_must_be_positive():
    with pytest.raises(ValueError):

        WalkForwardValidator(
            train_size=10,
            test_size=5,
            step_size=0,
        )


def test_negative_step_size_is_rejected():
    with pytest.raises(ValueError):

        WalkForwardValidator(
            train_size=10,
            test_size=5,
            step_size=-1,
        )


def test_gap_must_be_non_negative():
    with pytest.raises(ValueError):

        WalkForwardValidator(
            train_size=10,
            test_size=5,
            gap=-1,
        )


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------


def test_split_rejects_non_dataframe():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    with pytest.raises(TypeError):

        validator.split(
            np.arange(30),
        )


def test_split_rejects_empty_dataframe():
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    with pytest.raises(ValueError):

        validator.split(
            pd.DataFrame(),
        )


def test_split_rejects_unsorted_index(
    data,
):
    unsorted = data.iloc[
        ::-1
    ]

    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    with pytest.raises(ValueError):

        validator.split(
            unsorted,
        )


def test_split_rejects_duplicate_index(
    data,
):
    duplicated = data.copy()

    duplicated.index = [
        *data.index[:-1],
        data.index[-2],
    ]

    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    with pytest.raises(ValueError):

        validator.split(
            duplicated,
        )


# ---------------------------------------------------------------------------
# Boundary behavior
# ---------------------------------------------------------------------------


def test_no_split_when_data_is_too_short():
    data = pd.DataFrame(
        {
            "value": np.arange(10),
        },
        index=pd.date_range(
            "2024-01-01",
            periods=10,
        ),
    )

    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    splits = validator.split(data)

    assert splits == []


def test_no_split_when_gap_makes_test_impossible():
    data = pd.DataFrame(
        {
            "value": np.arange(15),
        },
        index=pd.date_range(
            "2024-01-01",
            periods=15,
        ),
    )

    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        gap=1,
    )

    splits = validator.split(data)

    assert splits == []


def test_split_returns_copies(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    split = validator.split(data)[0]

    split.train.iloc[0, 0] = -999.0

    assert (
        data.iloc[0, 0]
        != -999.0
    )


# ---------------------------------------------------------------------------
# Research integrity
# ---------------------------------------------------------------------------


def test_future_observations_never_enter_training_window(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
        expanding=True,
    )

    splits = validator.split(data)

    for split in splits:

        assert (
            split.train.index[-1]
            < split.test.index[0]
        )

        assert (
            split.train.index[-1]
            < data.index[-1]
        )


def test_test_observations_are_strictly_out_of_sample(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    splits = validator.split(data)

    for split in splits:

        assert (
            split.test.index[0]
            > split.train.index[-1]
        )


def test_split_data_reconstructs_expected_observations(
    data,
):
    validator = WalkForwardValidator(
        train_size=10,
        test_size=5,
    )

    split = validator.split(data)[0]

    pd.testing.assert_frame_equal(
        split.train,
        data.iloc[:10],
    )

    pd.testing.assert_frame_equal(
        split.test,
        data.iloc[10:15],
    )