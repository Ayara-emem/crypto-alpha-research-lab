from crypto_alpha_lab.research.covariance_robustness import RobustnessConfiguration
import pandas as pd
import pytest
import numpy as np

from crypto_alpha_lab.dataset import ResearchDataset
from crypto_alpha_lab.research.covariance_experiment import (
    run_covariance_experiment
)


@pytest.fixture
def dataset_factory():

    def _factory(
        close_prices,
        volume=None,
    ):
        prices = pd.DataFrame(
            {
                "Close": close_prices,
            },
            index=pd.date_range(
                "2024-01-01",
                periods=len(close_prices),
            ),
        )

        if volume is not None:
            prices["Volume"] = volume

        return ResearchDataset(
            prices=prices,
        )

    return _factory


@pytest.fixture
def sample_dataset():
    prices = pd.DataFrame(
        {
            "Close": np.arange(100, 110, dtype=float),
        },
        index=pd.date_range(
            "2024-01-01",
            periods=10,
        ),
    )

    return ResearchDataset(prices=prices)

@pytest.fixture
def benchmark_dataset(dataset_factory):
    return dataset_factory(
        [100, 101, 102, 103, 104]
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

@pytest.fixture
def prices() -> pd.DataFrame:
    """
    Deterministic multi-asset price history.
    """

    index = pd.date_range(
        "2024-01-01",
        periods=40,
        freq="D",
    )

    t = np.arange(
        len(index),
        dtype=float,
    )

    return pd.DataFrame(
        {
            "BTC": 100.0 + 1.50 * t,
            "ETH": 60.0 + 0.90 * t + 0.02 * t**2,
            "SOL": 40.0 + 0.55 * t + 0.03 * t**2,
        },
        index=index,
    )

@pytest.fixture
def experiments(prices):
    common = {
        "prices": prices,
        "train_size": 15,
        "test_size": 5,
    }

    return [
        run_covariance_experiment(
            method="sample",
            **common,
        ),
        run_covariance_experiment(
            method="shrinkage",
            shrinkage=0.25,
            **common,
        ),
        run_covariance_experiment(
            method="ledoit_wolf",
            **common,
        ),
    ]















