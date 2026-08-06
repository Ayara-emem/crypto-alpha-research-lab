"""
Research dataset construction.

Combine engineered features with forward-looking
targets into a research-ready dataset.
"""

from __future__ import annotations

import pandas as pd

from crypto_alpha_lab.dataset import ResearchDataset

from crypto_alpha_lab.research.feature_matrix import (
    build_feature_matrix,
)

from crypto_alpha_lab.research.targets import (
    future_return,
    future_log_return,
    future_direction,
    future_volatility,
)

_TARGETS = {
    "future_return": future_return,
    "future_log_return": future_log_return,
    "future_direction": future_direction,
    "future_volatility": future_volatility,
}

def build_research_dataset(
    dataset: ResearchDataset,
    target: str = "future_return",
    feature_window: int = 20,
    trend_long_window: int = 60,
    target_horizon: int = 1,
    volatility_window: int = 20,
    drop_missing: bool = True,
)-> pd.DataFrame:
    """
    Construct a research-ready dataset containing
    engineered features and a prediction target.
    """

    if target not in _TARGETS:
        raise ValueError(
            f"Unknown target '{target}'."
        )

    features = build_feature_matrix(
        dataset,
        window=feature_window,
        trend_long_window=trend_long_window,
    )

    if target == "future_volatility":
        target_values = future_volatility(
            dataset,
            horizon=volatility_window,
        )
    else:
        target_values = _TARGETS[target](
        dataset,
        horizon=target_horizon,
    )

    research = features.copy()

    research[target] = target_values

    if drop_missing:
        research = research.dropna()

    return research

