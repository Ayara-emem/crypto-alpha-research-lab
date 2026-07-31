import pandas as pd
import pytest
import numpy as np

from crypto_alpha_lab.dataset import ResearchDataset


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
