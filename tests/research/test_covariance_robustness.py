from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.research.covariance_robustness import (
    CovarianceRobustnessAnalyzer,
    CovarianceRobustnessResult,
    RobustnessConfiguration,
    run_covariance_robustness,
)

@pytest.fixture
def robustness_prices() -> pd.DataFrame:
    """
    Deterministic multi-asset price history
    large enough for robustness experiments.
    """

    index = pd.date_range(
        "2022-01-01",
        periods=800,
        freq="D",
    )

    t = np.arange(
        len(index),
        dtype=float,
    )

    return pd.DataFrame(
        {
            "BTC": (
                100.0
                + 0.15 * t
                + 0.02 * np.sin(t / 10.0)
            ),
            "ETH": (
                60.0
                + 0.10 * t
                + 0.03 * np.sin(t / 15.0)
            ),
            "SOL": (
                40.0
                + 0.08 * t
                + 0.04 * np.sin(t / 20.0)
            ),
        },
        index=index,
    )



@pytest.fixture
def configurations():

    return [
        RobustnessConfiguration(
            train_size=20,
            test_size=5,
            shrinkage=0.25,
        ),
        RobustnessConfiguration(
            train_size=30,
            test_size=5,
            shrinkage=0.25,
        ),
    ]


def test_configuration_can_be_created():

    configuration = RobustnessConfiguration(
        train_size=20,
        test_size=5,
        shrinkage=0.25,
    )

    assert configuration.train_size == 20
    assert configuration.test_size == 5
    assert configuration.shrinkage == 0.25


def test_invalid_train_size_is_rejected():

    with pytest.raises(ValueError):
        RobustnessConfiguration(
            train_size=0,
            test_size=5,
        )


def test_invalid_test_size_is_rejected():

    with pytest.raises(ValueError):
        RobustnessConfiguration(
            train_size=20,
            test_size=0,
        )


@pytest.mark.parametrize(
    "shrinkage",
    [-0.01, 1.01, np.nan, np.inf],
)
def test_invalid_shrinkage_is_rejected(
    shrinkage,
):

    with pytest.raises(
        ValueError
    ):
        RobustnessConfiguration(
            train_size=20,
            test_size=5,
            shrinkage=shrinkage,
        )


def test_analyzer_can_be_created(
    prices,
):

    analyzer = CovarianceRobustnessAnalyzer(
        prices
    )

    assert analyzer.prices.equals(
        prices
    )


def test_non_dataframe_prices_are_rejected():

    with pytest.raises(TypeError):

        CovarianceRobustnessAnalyzer(
            np.ones((10, 3))
        )


def test_empty_prices_are_rejected():

    with pytest.raises(ValueError):

        CovarianceRobustnessAnalyzer(
            pd.DataFrame()
        )


def test_duplicate_price_dates_are_rejected(
    prices,
):

    duplicate = pd.concat(
        [
            prices,
            prices.iloc[[0]],
        ]
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):

        CovarianceRobustnessAnalyzer(
            duplicate
        )


def test_non_chronological_prices_are_rejected(
    prices,
):

    unordered = prices.iloc[::-1]

    with pytest.raises(
        ValueError,
        match="chronological",
    ):

        CovarianceRobustnessAnalyzer(
            unordered
        )


def test_empty_configurations_are_rejected(
    prices,
):

    analyzer = CovarianceRobustnessAnalyzer(
        prices
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):

        analyzer.run([])


def test_default_methods_are_used(
    prices,
    configurations,
):

    result = run_covariance_robustness(
        prices,
        configurations,
    )

    assert result.metadata[
        "methods"
    ] == [
        "sample",
        "shrinkage",
        "ledoit_wolf",
    ]


def test_expected_experiment_count(
    prices,
    configurations,
):

    result = run_covariance_robustness(
        prices,
        configurations,
    )

    assert len(
        result.experiments
    ) == 6


def test_expected_summary_rows(
    prices,
    configurations,
):

    result = run_covariance_robustness(
        prices,
        configurations,
    )

    assert len(
        result.summary
    ) == 6


def test_result_type(
    prices,
    configurations,
):

    result = run_covariance_robustness(
        prices,
        configurations,
    )

    assert isinstance(
        result,
        CovarianceRobustnessResult,
    )


def test_summary_contains_method(
    prices,
    configurations,
):

    result = run_covariance_robustness(
        prices,
        configurations,
    )

    assert "method" in result.summary


def test_summary_contains_configuration(
    prices,
    configurations,
):

    result = run_covariance_robustness(
        prices,
        configurations,
    )

    assert "train_size" in result.summary
    assert "test_size" in result.summary


def test_summary_contains_performance(
    prices,
    configurations,
):

    result = run_covariance_robustness(
        prices,
        configurations,
    )

    assert "total_return" in result.summary
    assert "observation_count" in result.summary
    assert "fold_count" in result.summary


def test_all_experiments_are_oos(
    prices,
    configurations,
):

    result = run_covariance_robustness(
        prices,
        configurations,
    )

    for experiment in result.experiments:

        assert experiment.metadata[
            "out_of_sample"
        ] is True


def test_experiment_returns_are_finite(
    prices,
    configurations,
):

    result = run_covariance_robustness(
        prices,
        configurations,
    )

    for experiment in result.experiments:

        assert np.isfinite(
            experiment.returns.to_numpy()
        ).all()


def test_summary_returns_are_finite(
    prices,
    configurations,
):

    result = run_covariance_robustness(
        prices,
        configurations,
    )

    assert np.isfinite(
        result.summary[
            "total_return"
        ].to_numpy()
    ).all()


def test_custom_methods_are_supported(
    prices,
    configurations,
):

    result = run_covariance_robustness(
        prices,
        configurations,
        methods=["sample"],
    )

    assert len(
        result.experiments
    ) == len(configurations)

    assert set(
        result.summary["method"]
    ) == {"sample"}


def test_unknown_method_is_rejected(
    prices,
    configurations,
):

    with pytest.raises(
        ValueError,
        match="Unsupported",
    ):

        run_covariance_robustness(
            prices,
            configurations,
            methods=["unknown"],
        )


def test_duplicate_methods_are_rejected(
    prices,
    configurations,
):

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):

        run_covariance_robustness(
            prices,
            configurations,
            methods=[
                "sample",
                "sample",
            ],
        )


def test_shrinkage_requires_intensity(
    prices,
):

    configuration = RobustnessConfiguration(
        train_size=20,
        test_size=5,
        shrinkage=None,
    )

    with pytest.raises(
        ValueError,
        match="shrinkage",
    ):

        run_covariance_robustness(
            prices,
            [configuration],
            methods=["shrinkage"],
        )


def test_metadata_records_experiment_count(
    prices,
    configurations,
):

    result = run_covariance_robustness(
        prices,
        configurations,
    )

    assert result.metadata[
        "experiment_count"
    ] == 6


def test_metadata_records_configuration_count(
    prices,
    configurations,
):

    result = run_covariance_robustness(
        prices,
        configurations,
    )

    assert result.metadata[
        "configuration_count"
    ] == 2


def test_robustness_run_is_deterministic(
    prices,
    configurations,
):

    first = run_covariance_robustness(
        prices,
        configurations,
    )

    second = run_covariance_robustness(
        prices,
        configurations,
    )

    pd.testing.assert_frame_equal(
        first.summary,
        second.summary,
    )


# ---------------------------------------------------------------------------
# Training-window sensitivity
# ---------------------------------------------------------------------------


def test_multiple_training_windows_are_supported(
    robustness_prices: pd.DataFrame,
):
    """
    Robustness analysis should support multiple training windows.
    """

    configurations = [
        RobustnessConfiguration(
            train_size=126,
            test_size=21,
            shrinkage=0.25,
        ),
        RobustnessConfiguration(
            train_size=252,
            test_size=21,
            shrinkage=0.25,
        ),
        RobustnessConfiguration(
            train_size=504,
            test_size=21,
            shrinkage=0.25,
        ),
    ]

    result = run_covariance_robustness(
        prices=robustness_prices,
        configurations=configurations,
    )

    assert isinstance(
        result,
        CovarianceRobustnessResult,
    )

    assert len(result.experiments) == (
        len(configurations) * 3
    )


def test_training_window_is_preserved_in_summary(
    robustness_prices: pd.DataFrame,
):
    """
    Each robustness experiment must preserve
    its training-window configuration.
    """

    configurations = [
        RobustnessConfiguration(
            train_size=126,
            test_size=21,
            shrinkage=0.25,
        ),
        RobustnessConfiguration(
            train_size=252,
            test_size=21,
            shrinkage=0.25,
        ),
        RobustnessConfiguration(
            train_size=504,
            test_size=21,
            shrinkage=0.25,
        ),
    ]

    result = run_covariance_robustness(
        prices=robustness_prices,
        configurations=configurations,
    )

    assert set(
        result.summary["train_size"]
    ) == {
        126,
        252,
        504,
    }


def test_test_window_is_constant_in_training_sensitivity(
    robustness_prices: pd.DataFrame,
):
    """
    Training-window sensitivity should hold
    the test horizon constant.
    """

    configurations = [
        RobustnessConfiguration(
            train_size=126,
            test_size=21,
            shrinkage=0.25,
        ),
        RobustnessConfiguration(
            train_size=252,
            test_size=21,
            shrinkage=0.25,
        ),
        RobustnessConfiguration(
            train_size=504,
            test_size=21,
            shrinkage=0.25,
        ),
    ]

    result = run_covariance_robustness(
        prices=robustness_prices,
        configurations=configurations,
    )

    assert result.summary[
        "test_size"
    ].nunique() == 1

    assert (
        result.summary[
            "test_size"
        ].iloc[0]
        == 21
    )


def test_all_covariance_methods_are_evaluated_for_each_training_window(
    robustness_prices: pd.DataFrame,
):
    """
    Every training window should evaluate
    all supported covariance methodologies.
    """

    configurations = [
        RobustnessConfiguration(
            train_size=126,
            test_size=21,
            shrinkage=0.25,
        ),
        RobustnessConfiguration(
            train_size=252,
            test_size=21,
            shrinkage=0.25,
        ),
        RobustnessConfiguration(
            train_size=504,
            test_size=21,
            shrinkage=0.25,
        ),
    ]

    result = run_covariance_robustness (
        prices = robustness_prices,
        configurations=configurations,)


    for train_size in (
        126,
        252,
        504,
    ):
        methods = set(
            result.summary.loc[
                result.summary["train_size"]
                == train_size,
                "method",
            ]
        )

        assert methods == {
            "sample",
            "shrinkage",
            "ledoit_wolf",
        }