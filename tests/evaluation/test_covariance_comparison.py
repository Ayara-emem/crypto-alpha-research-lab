from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.evaluation.covariance_comparison import (
    CovarianceComparison,
    CovarianceComparisonResult,
    compare_covariance_methods,
)
from crypto_alpha_lab.research.covariance_experiment import (
    run_covariance_experiment,
)


def test_comparison_can_be_created():
    comparison = CovarianceComparison()

    assert comparison.periods_per_year == 252
    assert comparison.risk_free_rate == 0.0


def test_compare_returns_expected_type(
    experiments,
):
    result = CovarianceComparison().compare(
        experiments
    )

    assert isinstance(
        result,
        CovarianceComparisonResult,
    )


def test_convenience_function_returns_expected_type(
    experiments,
):
    result = compare_covariance_methods(
        experiments
    )

    assert isinstance(
        result,
        CovarianceComparisonResult,
    )


def test_summary_contains_all_methods(
    experiments,
):
    result = compare_covariance_methods(
        experiments
    )

    assert set(
        result.summary.index
    ) == {
        "sample",
        "shrinkage",
        "ledoit_wolf",
    }


def test_summary_contains_required_metrics(
    experiments,
):
    result = compare_covariance_methods(
        experiments
    )

    required = {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "calmar_ratio",
        "hit_rate",
        "observation_count",
        "average_turnover",
    }

    assert required.issubset(
        result.summary.columns
    )


def test_summary_values_are_finite(
    experiments,
):
    result = compare_covariance_methods(
        experiments
    )

    numeric = result.summary[
        [
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "calmar_ratio",
            "hit_rate",
            "average_turnover",
        ]
    ]

    assert np.isfinite(
        numeric.to_numpy()
    ).all()


def test_observation_counts_are_positive(
    experiments,
):
    result = compare_covariance_methods(
        experiments
    )

    assert (
        result.summary[
            "observation_count"
        ] > 0
    ).all()


def test_returns_are_preserved(
    experiments,
):
    result = compare_covariance_methods(
        experiments
    )

    for experiment in experiments:
        pd.testing.assert_series_equal(
            result.returns[
                experiment.method
            ],
            experiment.returns,
        )


def test_cumulative_returns_are_preserved(
    experiments,
):
    result = compare_covariance_methods(
        experiments
    )

    for experiment in experiments:
        pd.testing.assert_series_equal(
            result.cumulative_returns[
                experiment.method
            ],
            experiment.cumulative_returns,
        )


def test_comparison_is_strictly_oos(
    experiments,
):
    result = compare_covariance_methods(
        experiments
    )

    assert (
        result.metadata["out_of_sample"]
        is True
    )


def test_method_count_is_correct(
    experiments,
):
    result = compare_covariance_methods(
        experiments
    )

    assert (
        result.metadata["method_count"]
        == 3
    )


def test_methods_are_recorded(
    experiments,
):
    result = compare_covariance_methods(
        experiments
    )

    assert result.metadata[
        "methods"
    ] == [
        "sample",
        "shrinkage",
        "ledoit_wolf",
    ]


def test_duplicate_methods_are_rejected(
    experiments,
):
    duplicate = [
        experiments[0],
        experiments[0],
    ]

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        compare_covariance_methods(
            duplicate
        )


def test_empty_experiments_are_rejected():
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        compare_covariance_methods([])


def test_non_list_experiments_are_rejected():
    with pytest.raises(
        TypeError,
        match="list",
    ):
        compare_covariance_methods(
            tuple()
        )


def test_invalid_experiment_type_is_rejected(
    experiments,
):
    with pytest.raises(
        TypeError,
        match="CovarianceExperimentResult",
    ):
        compare_covariance_methods(
            [experiments[0], object()]
        )


def test_non_oos_experiment_is_rejected(
    experiments,
):
    experiment = experiments[0]

    modified_metadata = dict(
        experiment.metadata
    )

    modified_metadata[
        "out_of_sample"
    ] = False

    from crypto_alpha_lab.research.covariance_experiment import (
        CovarianceExperimentResult,
    )

    invalid = CovarianceExperimentResult(
        method=experiment.method,
        returns=experiment.returns,
        cumulative_returns=(
            experiment.cumulative_returns
        ),
        folds=experiment.folds,
        parameters=experiment.parameters,
        metadata=modified_metadata,
    )

    with pytest.raises(
        ValueError,
        match="out-of-sample",
    ):
        compare_covariance_methods(
            [invalid]
        )


def test_empty_returns_are_rejected(
    experiments,
):
    from crypto_alpha_lab.research.covariance_experiment import (
        CovarianceExperimentResult,
    )

    experiment = experiments[0]

    invalid = CovarianceExperimentResult(
        method=experiment.method,
        returns=pd.Series(
            dtype=float
        ),
        cumulative_returns=pd.Series(
            dtype=float
        ),
        folds=experiment.folds,
        parameters=experiment.parameters,
        metadata=experiment.metadata,
    )

    with pytest.raises(
        ValueError,
        match="no returns",
    ):
        compare_covariance_methods(
            [invalid]
        )


def test_duplicate_return_dates_are_rejected(
    experiments,
):
    from crypto_alpha_lab.research.covariance_experiment import (
        CovarianceExperimentResult,
    )

    experiment = experiments[0]

    returns = pd.concat(
        [
            experiment.returns,
            experiment.returns.iloc[
                [0]
            ],
        ]
    )

    invalid = CovarianceExperimentResult(
        method=experiment.method,
        returns=returns,
        cumulative_returns=experiment.cumulative_returns,
        folds=experiment.folds,
        parameters=experiment.parameters,
        metadata=experiment.metadata,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        compare_covariance_methods(
            [invalid]
        )


def test_custom_periods_per_year_are_preserved(
    experiments,
):
    result = compare_covariance_methods(
        experiments,
        periods_per_year=365,
    )

    assert (
        result.metadata[
            "periods_per_year"
        ]
        == 365
    )


def test_custom_risk_free_rate_is_preserved(
    experiments,
):
    result = compare_covariance_methods(
        experiments,
        risk_free_rate=0.03,
    )

    assert (
        result.metadata[
            "risk_free_rate"
        ]
        == 0.03
    )


def test_average_turnover_is_reported(
    experiments,
):
    result = compare_covariance_methods(
        experiments
    )

    assert (
        "average_turnover"
        in result.summary.columns
    )

    assert np.isfinite(
        result.summary[
            "average_turnover"
        ].to_numpy()
    ).all()


def test_comparison_is_deterministic(
    experiments,
):
    first = compare_covariance_methods(
        experiments
    )

    second = compare_covariance_methods(
        experiments
    )

    pd.testing.assert_frame_equal(
        first.summary,
        second.summary,
    )