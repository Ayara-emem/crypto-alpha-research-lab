import numpy as np
import pandas as pd
import pytest

from crypto_alpha_lab.evaluation.robustness import (
    rank_methods_by_configuration,
    ranking_stability,
)

from crypto_alpha_lab.evaluation.robustness import (
    performance_stability,
    portfolio_weight_stability,
)

@pytest.fixture
def summary():

    return pd.DataFrame(
        {
            "method": [
                "sample",
                "shrinkage",
                "ledoit_wolf",
                "sample",
                "shrinkage",
                "ledoit_wolf",
            ],
            "train_size": [
                20,
                20,
                20,
                30,
                30,
                30,
            ],
            "test_size": [
                5,
                5,
                5,
                5,
                5,
                5,
            ],
            "total_return": [
                0.10,
                0.12,
                0.15,
                0.08,
                0.11,
                0.09,
            ],
        }
    )


def test_rank_methods_returns_dataframe(
    summary,
):

    result = rank_methods_by_configuration(
        summary
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_rank_column_is_present(
    summary,
):

    result = rank_methods_by_configuration(
        summary
    )

    assert "rank" in result.columns


def test_each_configuration_has_ranks(
    summary,
):

    result = rank_methods_by_configuration(
        summary
    )

    for _, group in result.groupby(
        ["train_size", "test_size"]
    ):
        assert set(
            group["rank"]
        ) == {1.0, 2.0, 3.0}


def test_best_method_receives_rank_one(
    summary,
):

    result = rank_methods_by_configuration(
        summary
    )

    first = result[
        (
            result["train_size"] == 20
        )
        & (
            result["method"]
            == "ledoit_wolf"
        )
    ]

    assert first["rank"].iloc[0] == 1.0


def test_ranking_stability_returns_dataframe(
    summary,
):

    result = ranking_stability(
        summary
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_ranking_stability_contains_methods(
    summary,
):

    result = ranking_stability(
        summary
    )

    assert set(
        result.index
    ) == {
        "sample",
        "shrinkage",
        "ledoit_wolf",
    }


def test_ranking_probabilities_sum_to_one(
    summary,
):

    result = ranking_stability(
        summary
    )

    assert np.allclose(
        result.sum(axis=1).to_numpy(),
        1.0,
    )


def test_ranking_columns_are_created(
    summary,
):

    result = ranking_stability(
        summary
    )

    assert set(
        result.columns
    ) == {
        "rank_1",
        "rank_2",
        "rank_3",
    }


def test_empty_summary_is_rejected():

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        ranking_stability(
            pd.DataFrame()
        )


def test_missing_metric_is_rejected(
    summary,
):

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        ranking_stability(
            summary,
            metric="sharpe_ratio",
        )

def test_performance_stability_returns_dataframe(
    summary,
):

    result = performance_stability(
        summary
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_performance_stability_contains_methods(
    summary,
):

    result = performance_stability(
        summary
    )

    assert set(
        result.index
    ) == {
        "sample",
        "shrinkage",
        "ledoit_wolf",
    }


def test_performance_stability_contains_required_metrics(
    summary,
):

    result = performance_stability(
        summary
    )

    assert set(
        [
            "mean",
            "standard_deviation",
            "minimum",
            "maximum",
            "range",
            "observations",
        ]
    ).issubset(
        result.columns
    )


def test_performance_range_is_correct(
    summary,
):

    result = performance_stability(
        summary
    )

    sample = result.loc[
        "sample"
    ]

    assert sample["range"] == pytest.approx(
        0.02
    )


def test_performance_observation_count_is_correct(
    summary,
):

    result = performance_stability(
        summary
    )

    assert (
        result.loc[
            "sample",
            "observations",
        ]
        == 2
    )


def test_empty_summary_is_rejected_by_performance_stability():

    with pytest.raises(
        ValueError,
        match="empty",
    ):

        performance_stability(
            pd.DataFrame()
        )


def test_missing_metric_is_rejected_by_performance_stability(
    summary,
):

    with pytest.raises(
        ValueError,
        match="missing",
    ):

        performance_stability(
            summary,
            metric="sharpe_ratio",
        )

def test_portfolio_weight_stability_returns_dataframe(
    experiments,
):

    result = portfolio_weight_stability(
        experiments
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_portfolio_weight_stability_contains_methods(
    experiments,
):

    result = portfolio_weight_stability(
        experiments
    )

    assert set(
        result.index
    ) == {
        "sample",
        "shrinkage",
        "ledoit_wolf",
    }


def test_portfolio_weight_stability_contains_required_metrics(
    experiments,
):

    result = portfolio_weight_stability(
        experiments
    )

    assert {
        "mean_weight_change",
        "maximum_weight_change",
        "fold_count",
    }.issubset(
        result.columns
    )


def test_portfolio_weight_stability_is_finite(
    experiments,
):

    result = portfolio_weight_stability(
        experiments
    )

    assert np.isfinite(
        result[
            [
                "mean_weight_change",
                "maximum_weight_change",
            ]
        ].to_numpy()
    ).all()


def test_portfolio_weight_changes_are_non_negative(
    experiments,
):

    result = portfolio_weight_stability(
        experiments
    )

    assert (
        result[
            "mean_weight_change"
        ]
        >= 0
    ).all()

    assert (
        result[
            "maximum_weight_change"
        ]
        >= 0
    ).all()


def test_empty_experiments_are_rejected_by_weight_stability():

    with pytest.raises(
        ValueError,
        match="empty",
    ):

        portfolio_weight_stability(
            []
        )


