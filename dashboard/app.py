from __future__ import annotations

from pathlib import Path

import streamlit as st


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIGURES_DIR = (
    PROJECT_ROOT
    / "docs"
    / "figures"
)


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="CARL Research Dashboard",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.title(
    "Crypto Alpha Research Laboratory"
)

st.subheader(
    "Covariance Estimation & Portfolio Research Dashboard"
)

st.markdown(
    """
    CARL evaluates alternative covariance estimation methods for
    Global Minimum Variance portfolios using walk-forward,
    out-of-sample research.
    """
)

st.divider()


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.title(
    "Research Navigation"
)

section = st.sidebar.radio(
    "Select section",
    [
        "Executive Overview",
        "Method Comparison",
        "Risk Analysis",
        "Robustness",
        "Regime Analysis",
        "Research Conclusion",
    ],
)


# ---------------------------------------------------------------------
# Executive Overview
# ---------------------------------------------------------------------

if section == "Executive Overview":

    st.header(
        "Executive Overview"
    )

    st.markdown(
        """
        ### Research Question

        Does the choice of covariance estimator materially affect
        portfolio performance, risk, implementation characteristics,
        and robustness?

        ### Methods

        - Sample covariance
        - Fixed shrinkage covariance
        - Ledoit-Wolf covariance

        ### Portfolio

        Global Minimum Variance

        ### Validation

        Walk-forward out-of-sample evaluation
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Baseline Winner",
        "Sample",
    )

    col2.metric(
        "Best Baseline Sharpe",
        "0.451",
    )

    col3.metric(
        "Lowest Drawdown",
        "Ledoit-Wolf",
    )

    col4.metric(
        "Lowest Turnover",
        "Shrinkage",
    )

    st.divider()

    st.subheader(
        "Cumulative Out-of-Sample Performance"
    )

    figure = (
        FIGURES_DIR
        / "fig01_cumulative_oos_returns.png"
    )

    if figure.exists():
        st.image(
            str(figure),
            use_container_width=True,
        )
    else:
        st.warning(
            "Figure 01 has not been generated yet."
        )


# ---------------------------------------------------------------------
# Method Comparison
# ---------------------------------------------------------------------

elif section == "Method Comparison":

    st.header(
        "Covariance Method Comparison"
    )

    st.markdown(
        """
        The baseline comparison evaluates three covariance
        estimators under the same walk-forward out-of-sample
        framework.
        """
    )

    import pandas as pd

    comparison = pd.DataFrame(
        {
            "Total Return": [
                0.869704,
                0.781683,
                0.568794,
            ],
            "Annualized Return": [
                0.106805,
                0.098184,
                0.075755,
            ],
            "Volatility": [
                0.436643,
                0.435377,
                0.437010,
            ],
            "Sharpe": [
                0.450538,
                0.432726,
                0.385554,
            ],
            "Max Drawdown": [
                -0.781959,
                -0.767388,
                -0.725831,
            ],
            "Turnover": [
                1.219442,
                1.110687,
                1.352342,
            ],
        },
        index=[
            "Sample",
            "Shrinkage",
            "Ledoit-Wolf",
        ],
    )

    st.dataframe(
        comparison.style.format(
            {
                "Total Return": "{:.2%}",
                "Annualized Return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe": "{:.3f}",
                "Max Drawdown": "{:.2%}",
                "Turnover": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    figure = (
        FIGURES_DIR
        / "fig02_risk_adjusted_performance.png"
    )

    if figure.exists():
        st.image(
            str(figure),
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# Risk Analysis
# ---------------------------------------------------------------------

elif section == "Risk Analysis":

    st.header(
        "Risk & Implementation Analysis"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Maximum Drawdown"
        )

        figure = (
            FIGURES_DIR
            / "fig03_maximum_drawdown.png"
        )

        if figure.exists():
            st.image(
                str(figure),
                use_container_width=True,
            )

    with col2:

        st.subheader(
            "Portfolio Turnover"
        )

        figure = (
            FIGURES_DIR
            / "fig04_turnover.png"
        )

        if figure.exists():
            st.image(
                str(figure),
                use_container_width=True,
            )


# ---------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------

elif section == "Robustness":

    st.header(
        "Robustness Analysis"
    )

    st.markdown(
        """
        CARL evaluates whether the research conclusions remain stable
        when key experimental configurations change.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Training-Window Sensitivity"
        )

        figure = (
            FIGURES_DIR
            / "fig05_training_window_sensitivity.png"
        )

        if figure.exists():
            st.image(
                str(figure),
                use_container_width=True,
            )

    with col2:

        st.subheader(
            "Shrinkage Sensitivity"
        )

        figure = (
            FIGURES_DIR
            / "fig06_shrinkage_sensitivity.png"
        )

        if figure.exists():
            st.image(
                str(figure),
                use_container_width=True,
            )


# ---------------------------------------------------------------------
# Regime Analysis
# ---------------------------------------------------------------------

elif section == "Regime Analysis":

    st.header(
        "Historical Regime Analysis"
    )

    st.markdown(
        """
        Covariance-method rankings can change across historical
        market environments.
        """
    )

    figure = (
        FIGURES_DIR
        / "fig07_regime_analysis.png"
    )

    if figure.exists():
        st.image(
            str(figure),
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# Research Conclusion
# ---------------------------------------------------------------------

elif section == "Research Conclusion":

    st.header(
        "Research Conclusion"
    )

    st.success(
        """
        There is no universally dominant covariance estimator.
        Method selection depends on the investment objective,
        estimation configuration, and market regime.
        """
    )

    st.markdown(
        """
        ### Performance-focused

        **Sample covariance** leads the baseline comparison.

        ### Risk-control focused

        **Ledoit-Wolf and shrinkage** provide more attractive
        downside-risk characteristics.

        ### Implementation-focused

        **Fixed shrinkage** produces the lowest baseline turnover.

        ### Robustness

        Training-window, shrinkage-intensity, and regime analysis
        demonstrate that covariance-method rankings are not invariant
        across research configurations.
        """
    )

    st.info(
        """
        CARL is a quantitative research laboratory. Historical
        backtest results should not be interpreted as investment
        advice or a guarantee of future performance.
        """
    )