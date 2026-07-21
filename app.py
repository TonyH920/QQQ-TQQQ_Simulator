from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from simulation import simulate_all_rolling_periods


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="QQQ vs TQQQ Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# SMALL STYLE IMPROVEMENTS
# =========================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        [data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.06);
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 12px;
            padding: 14px;
        }

        div[data-testid="stExpander"] {
            border-radius: 12px;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# FILE LOCATION
# =========================================================
BASE_FOLDER = Path(__file__).resolve().parent

EXTENDED_HISTORY_FILE = (
    BASE_FOLDER / "extended_history_1938.csv"
)


# =========================================================
# LOAD AND CLEAN DATA
# =========================================================
@st.cache_data
def load_extended_history(
    file_path: Path,
) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            "extended_history_1938.csv was not found. "
            "Run build_extended_history.py first."
        )

    data = pd.read_csv(file_path)

    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
    )

    required_columns = {
        "Date",
        "QQQ Close",
        "TQQQ Close",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "The extended-history file is missing: "
            f"{sorted(missing_columns)}"
        )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
    )

    data["QQQ Close"] = pd.to_numeric(
        data["QQQ Close"],
        errors="coerce",
    )

    data["TQQQ Close"] = pd.to_numeric(
        data["TQQQ Close"],
        errors="coerce",
    )

    data = (
        data.dropna(
            subset=[
                "Date",
                "QQQ Close",
                "TQQQ Close",
            ]
        )
        .drop_duplicates(
            subset="Date",
            keep="last",
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return data


@st.cache_data(show_spinner=False)
def calculate_rolling_periods(
    prices: pd.DataFrame,
    years: int,
    initial_investment: float,
    monthly_investment: float,
) -> pd.DataFrame:
    return simulate_all_rolling_periods(
        prices=prices,
        years=years,
        initial_investment=initial_investment,
        monthly_investment=monthly_investment,
    )


try:
    full_history = load_extended_history(
        EXTENDED_HISTORY_FILE
    )

except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def format_period(years: int) -> str:
    if years == 1:
        return "1 year"

    return f"{years} years"


def create_distribution_chart(
    results: pd.DataFrame,
    years: int,
) -> plt.Figure:
    """
    Plot distributions of rolling QQQ and TQQQ returns.

    Returns are shown as percentage points.
    """

    qqq_returns = (
        results["QQQ Return"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        * 100
    )

    tqqq_returns = (
        results["TQQQ Return"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        * 100
    )

    combined_returns = pd.concat(
        [
            qqq_returns,
            tqqq_returns,
        ],
        ignore_index=True,
    )

    lower_limit = combined_returns.quantile(
        0.005
    )

    upper_limit = combined_returns.quantile(
        0.995
    )

    filtered_qqq = qqq_returns[
        qqq_returns.between(
            lower_limit,
            upper_limit,
        )
    ]

    filtered_tqqq = tqqq_returns[
        tqqq_returns.between(
            lower_limit,
            upper_limit,
        )
    ]

    bins = np.linspace(
        lower_limit,
        upper_limit,
        45,
    )

    figure, axis = plt.subplots(
        figsize=(11, 5.5)
    )

    axis.hist(
        filtered_qqq,
        bins=bins,
        alpha=0.55,
        label="QQQ",
        density=True,
    )

    axis.hist(
        filtered_tqqq,
        bins=bins,
        alpha=0.55,
        label="TQQQ",
        density=True,
    )

    axis.axvline(
        0,
        linewidth=1,
        linestyle="--",
    )

    axis.axvline(
        qqq_returns.median(),
        linewidth=1.5,
        linestyle="-",
        label="QQQ median",
    )

    axis.axvline(
        tqqq_returns.median(),
        linewidth=1.5,
        linestyle=":",
        label="TQQQ median",
    )

    axis.set_title(
        f"Distribution of {format_period(years)} "
        f"rolling investment returns"
    )

    axis.set_xlabel(
        "Return on total contributions (%)"
    )

    axis.set_ylabel(
        "Relative frequency"
    )

    axis.legend()

    axis.grid(
        alpha=0.20,
    )

    figure.tight_layout()

    return figure


def create_difference_distribution_chart(
    results: pd.DataFrame,
    years: int,
) -> plt.Figure:
    """
    Plot the distribution of the TQQQ-minus-QQQ
    ending-value difference.
    """

    differences = (
        results["TQQQ Minus QQQ"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    lower_limit = differences.quantile(
        0.005
    )

    upper_limit = differences.quantile(
        0.995
    )

    filtered_differences = differences[
        differences.between(
            lower_limit,
            upper_limit,
        )
    ]

    figure, axis = plt.subplots(
        figsize=(11, 4.8)
    )

    axis.hist(
        filtered_differences,
        bins=45,
        alpha=0.75,
    )

    axis.axvline(
        0,
        linewidth=1.5,
        linestyle="--",
        label="Tie",
    )

    axis.axvline(
        differences.median(),
        linewidth=1.5,
        linestyle=":",
        label="Median difference",
    )

    axis.set_title(
        f"Distribution of TQQQ's advantage "
        f"over {format_period(years)}"
    )

    axis.set_xlabel(
        "TQQQ ending value minus QQQ ending value ($)"
    )

    axis.set_ylabel(
        "Number of starting dates"
    )

    axis.legend()

    axis.grid(
        alpha=0.20,
    )

    figure.tight_layout()

    return figure


def return_summary_table(
    results: pd.DataFrame,
) -> pd.DataFrame:
    qqq_returns = results["QQQ Return"]
    tqqq_returns = results["TQQQ Return"]

    summary = pd.DataFrame(
        {
            "Statistic": [
                "Average return",
                "Median return",
                "Best return",
                "Worst return",
                "Positive-return periods",
                "25th percentile",
                "75th percentile",
            ],
            "QQQ": [
                qqq_returns.mean(),
                qqq_returns.median(),
                qqq_returns.max(),
                qqq_returns.min(),
                (qqq_returns > 0).mean(),
                qqq_returns.quantile(0.25),
                qqq_returns.quantile(0.75),
            ],
            "TQQQ": [
                tqqq_returns.mean(),
                tqqq_returns.median(),
                tqqq_returns.max(),
                tqqq_returns.min(),
                (tqqq_returns > 0).mean(),
                tqqq_returns.quantile(0.25),
                tqqq_returns.quantile(0.75),
            ],
        }
    )

    return summary


# =========================================================
# HEADER
# =========================================================
st.title("QQQ vs TQQQ Historical Backtest")

st.markdown(
    """
    Test how QQQ and TQQQ performed from **every possible
    historical starting date**, using an initial investment
    and consistent monthly contributions.
    """
)


# =========================================================
# METHODOLOGY
# =========================================================
with st.expander(
    "How the historical data was created",
    expanded=False,
):
    st.markdown(
        "### A quick explanation of the reconstructed history"
    )

    method_1, method_2, method_3 = st.columns(3)

    with method_1:
        st.markdown(
            """
            #### 1. Historical Nasdaq-100

            The source dataset contains Nasdaq-100-style history
            beginning in 1938.

            The portion before the Nasdaq-100 officially launched
            in 1985 is a **vendor reconstruction**, not a real
            investable index record.
            """
        )

    with method_2:
        st.markdown(
            """
            #### 2. Synthetic QQQ

            Actual QQQ prices are used beginning in 1999.

            Before 1999, QQQ is reconstructed from the daily
            percentage movements of the historical Nasdaq-100
            series and connected to the first actual QQQ price.
            """
        )

    with method_3:
        st.markdown(
            """
            #### 3. Synthetic TQQQ

            Actual TQQQ prices are used beginning in 2010.

            Earlier TQQQ returns are estimated using the observed
            daily relationship between actual QQQ and TQQQ:

            **TQQQ return = alpha + beta × QQQ return**
            """
        )

    st.divider()

    model_1, model_2, model_3, model_4 = (
        st.columns(4)
    )

    model_1.metric(
        "Estimated daily leverage",
        "2.9574×",
    )

    model_2.metric(
        "Daily model R²",
        "0.9978",
    )

    model_3.metric(
        "Model observations",
        "4,121",
    )

    model_4.metric(
        "History begins",
        "Jan 3, 1938",
    )

    st.info(
        """
        Synthetic results show what the funds might have done
        under the model. They are not actual ETF returns before
        each fund's launch.
        """
    )


# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.title("Backtest settings")

history_mode = st.sidebar.radio(
    "History to use",
    [
        "Extended reconstructed history",
        "QQQ era only",
    ],
    help=(
        "Extended history begins in 1938. "
        "QQQ-era history begins in 1999."
    ),
)

initial_investment = st.sidebar.number_input(
    "Initial investment",
    min_value=0.0,
    value=5000.0,
    step=500.0,
    format="%.2f",
)

monthly_investment = st.sidebar.number_input(
    "Monthly contribution",
    min_value=0.0,
    value=100.0,
    step=25.0,
    format="%.2f",
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **How each test works**

    The initial amount is invested on the starting date.
    The monthly contribution is then invested on the first
    available trading day of each following month.
    """
)


# =========================================================
# CHOOSE DATASET
# =========================================================
if history_mode == "Extended reconstructed history":
    analysis_prices = full_history[
        [
            "Date",
            "QQQ Close",
            "TQQQ Close",
        ]
    ].copy()

    st.warning(
        """
        **Extended reconstructed history selected:** Results
        before 1985 rely on vendor-reconstructed Nasdaq-100 data.
        QQQ is synthetic before 1999 and TQQQ is synthetic before
        2010.
        """
    )

else:
    qqq_start_date = pd.Timestamp(
        "1999-03-10"
    )

    analysis_prices = full_history[
        full_history["Date"] >= qqq_start_date
    ][
        [
            "Date",
            "QQQ Close",
            "TQQQ Close",
        ]
    ].copy()

    st.info(
        """
        **QQQ-era history selected:** This begins with actual QQQ
        history in 1999. TQQQ remains synthetic before its 2010
        launch.
        """
    )


analysis_prices = (
    analysis_prices.dropna(
        subset=[
            "Date",
            "QQQ Close",
            "TQQQ Close",
        ]
    )
    .sort_values("Date")
    .reset_index(drop=True)
)


# =========================================================
# DATASET SUMMARY
# =========================================================
dataset_1, dataset_2, dataset_3, dataset_4 = (
    st.columns(4)
)

dataset_1.metric(
    "First date",
    analysis_prices[
        "Date"
    ].min().strftime("%b %d, %Y"),
)

dataset_2.metric(
    "Last date",
    analysis_prices[
        "Date"
    ].max().strftime("%b %d, %Y"),
)

dataset_3.metric(
    "Trading days",
    f"{len(analysis_prices):,}",
)

dataset_4.metric(
    "Investment plan",
    (
        f"${initial_investment:,.0f} + "
        f"${monthly_investment:,.0f}/month"
    ),
)


# =========================================================
# BACKTEST INTRODUCTION
# =========================================================
st.divider()

st.header("Rolling historical results")

st.markdown(
    f"""
    The analysis tests **every possible starting trading day**
    for 1-, 5-, and 10-year holding periods.

    Each test begins with **\\${initial_investment:,.0f}** and
    adds **\\${monthly_investment:,.0f} per month**.
    """
)


# =========================================================
# RUN FIXED 1-, 5-, AND 10-YEAR TESTS
# =========================================================
PERIODS = [
    1,
    5,
    10,
]

period_tabs = st.tabs(
    [
        "1-year results",
        "5-year results",
        "10-year results",
    ]
)

all_results = []


for tab, years in zip(
    period_tabs,
    PERIODS,
):
    with tab:
        with st.spinner(
            f"Calculating all possible "
            f"{years}-year starting dates..."
        ):
            results = calculate_rolling_periods(
                prices=analysis_prices,
                years=years,
                initial_investment=(
                    initial_investment
                ),
                monthly_investment=(
                    monthly_investment
                ),
            )

        if results.empty:
            st.warning(
                f"There is not enough data to calculate "
                f"{format_period(years)} periods."
            )
            continue

        results = results.copy()
        results["Period Length"] = years

        all_results.append(results)

        total_periods = len(results)

        tqqq_wins = int(
            (
                results["Winner"]
                == "TQQQ"
            ).sum()
        )

        qqq_wins = int(
            (
                results["Winner"]
                == "QQQ"
            ).sum()
        )

        ties = int(
            (
                results["Winner"]
                == "Tie"
            ).sum()
        )

        tqqq_win_rate = (
            tqqq_wins / total_periods
        )

        qqq_win_rate = (
            qqq_wins / total_periods
        )

        median_qqq = results[
            "QQQ Ending Value"
        ].median()

        median_tqqq = results[
            "TQQQ Ending Value"
        ].median()

        median_difference = results[
            "TQQQ Minus QQQ"
        ].median()

        # =================================================
        # TOP RESULT SUMMARY
        # =================================================
        st.subheader(
            f"{format_period(years).title()} overview"
        )

        summary_1, summary_2, summary_3 = (
            st.columns(3)
        )

        summary_1.metric(
            "TQQQ finished ahead",
            f"{tqqq_win_rate:.1%}",
            delta=(
                f"{tqqq_wins:,} starting dates"
            ),
        )

        summary_2.metric(
            "QQQ finished ahead",
            f"{qqq_win_rate:.1%}",
            delta=(
                f"{qqq_wins:,} starting dates"
            ),
        )

        summary_3.metric(
            "Starting dates tested",
            f"{total_periods:,}",
            delta=f"{ties:,} ties",
        )

        balance_1, balance_2, balance_3 = (
            st.columns(3)
        )

        balance_1.metric(
            "Median QQQ ending balance",
            f"${median_qqq:,.2f}",
        )

        balance_2.metric(
            "Median TQQQ ending balance",
            f"${median_tqqq:,.2f}",
        )

        balance_3.metric(
            "Median TQQQ advantage",
            f"${median_difference:,.2f}",
        )

        st.caption(
            f"Starting dates range from "
            f"{results['Start Date'].min():%B %d, %Y} "
            f"through "
            f"{results['Start Date'].max():%B %d, %Y}."
        )

        # =================================================
        # INNER NAVIGATION
        # =================================================
        (
            performance_tab,
            distribution_tab,
            extremes_tab,
            data_tab,
        ) = st.tabs(
            [
                "Performance over time",
                "Return distributions",
                "Best and worst periods",
                "Detailed data",
            ]
        )

        # =================================================
        # PERFORMANCE TAB
        # =================================================
        with performance_tab:
            st.markdown(
                "### Ending balance by starting date"
            )

            st.caption(
                "Each point represents a separate historical "
                "starting date. Both portfolios received the "
                "same contributions."
            )

            ending_value_chart = results[
                [
                    "Start Date",
                    "QQQ Ending Value",
                    "TQQQ Ending Value",
                ]
            ].set_index("Start Date")

            st.line_chart(
                ending_value_chart,
                y_label="Ending portfolio value ($)",
                height=450,
            )

            st.markdown(
                "### TQQQ advantage or disadvantage"
            )

            st.caption(
                "Above zero means TQQQ ended with more money. "
                "Below zero means QQQ ended with more money."
            )

            difference_chart = results[
                [
                    "Start Date",
                    "TQQQ Minus QQQ",
                ]
            ].set_index("Start Date")

            st.line_chart(
                difference_chart,
                y_label="TQQQ minus QQQ ($)",
                height=350,
            )

        # =================================================
        # DISTRIBUTION TAB
        # =================================================
        with distribution_tab:
            st.markdown(
                "### Distribution of historical returns"
            )

            st.markdown(
                """
                A distribution shows how frequently different
                outcomes occurred.

                A wider distribution means results varied more
                depending on the starting date. Values below zero
                represent a loss relative to total contributions.
                """
            )

            distribution_figure = (
                create_distribution_chart(
                    results=results,
                    years=years,
                )
            )

            st.pyplot(
                distribution_figure,
                use_container_width=True,
            )

            plt.close(
                distribution_figure
            )

            st.caption(
                "For readability, the chart excludes only the "
                "most extreme 0.5% of observations on each side. "
                "All observations remain included in the summary "
                "statistics and win percentages."
            )

            st.markdown(
                "### Return statistics"
            )

            distribution_summary = (
                return_summary_table(
                    results
                )
            )

            st.dataframe(
                distribution_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Statistic": (
                        st.column_config.TextColumn(
                            "Statistic"
                        )
                    ),
                    "QQQ": (
                        st.column_config.NumberColumn(
                            "QQQ",
                            format="percent",
                        )
                    ),
                    "TQQQ": (
                        st.column_config.NumberColumn(
                            "TQQQ",
                            format="percent",
                        )
                    ),
                },
            )

            st.markdown(
                "### Distribution of TQQQ's dollar advantage"
            )

            difference_figure = (
                create_difference_distribution_chart(
                    results=results,
                    years=years,
                )
            )

            st.pyplot(
                difference_figure,
                use_container_width=True,
            )

            plt.close(
                difference_figure
            )

        # =================================================
        # EXTREMES TAB
        # =================================================
        with extremes_tab:
            best_tqqq_period = results.loc[
                results[
                    "TQQQ Minus QQQ"
                ].idxmax()
            ]

            worst_tqqq_period = results.loc[
                results[
                    "TQQQ Minus QQQ"
                ].idxmin()
            ]

            best_qqq_return_period = results.loc[
                results[
                    "QQQ Return"
                ].idxmax()
            ]

            worst_qqq_return_period = results.loc[
                results[
                    "QQQ Return"
                ].idxmin()
            ]

            best_column, worst_column = (
                st.columns(2)
            )

            with best_column:
                st.markdown(
                    "### Largest TQQQ victory"
                )

                st.metric(
                    "TQQQ advantage",
                    (
                        f"${best_tqqq_period['TQQQ Minus QQQ']:,.2f}"
                    ),
                )

                st.write(
                    f"**Start:** "
                    f"{best_tqqq_period['Start Date']:%B %d, %Y}"
                )

                st.write(
                    f"**End:** "
                    f"{best_tqqq_period['End Date']:%B %d, %Y}"
                )

                st.write(
                    f"QQQ ending balance: "
                    f"**${best_tqqq_period['QQQ Ending Value']:,.2f}**"
                )

                st.write(
                    f"TQQQ ending balance: "
                    f"**${best_tqqq_period['TQQQ Ending Value']:,.2f}**"
                )

            with worst_column:
                st.markdown(
                    "### Largest TQQQ defeat"
                )

                st.metric(
                    "TQQQ minus QQQ",
                    (
                        f"${worst_tqqq_period['TQQQ Minus QQQ']:,.2f}"
                    ),
                )

                st.write(
                    f"**Start:** "
                    f"{worst_tqqq_period['Start Date']:%B %d, %Y}"
                )

                st.write(
                    f"**End:** "
                    f"{worst_tqqq_period['End Date']:%B %d, %Y}"
                )

                st.write(
                    f"QQQ ending balance: "
                    f"**${worst_tqqq_period['QQQ Ending Value']:,.2f}**"
                )

                st.write(
                    f"TQQQ ending balance: "
                    f"**${worst_tqqq_period['TQQQ Ending Value']:,.2f}**"
                )

            st.divider()

            qqq_best_column, qqq_worst_column = (
                st.columns(2)
            )

            with qqq_best_column:
                st.markdown(
                    "### Best QQQ outcome"
                )

                st.metric(
                    "QQQ return",
                    (
                        f"{best_qqq_return_period['QQQ Return']:.1%}"
                    ),
                )

                st.write(
                    f"Started "
                    f"**{best_qqq_return_period['Start Date']:%B %d, %Y}**"
                )

            with qqq_worst_column:
                st.markdown(
                    "### Worst QQQ outcome"
                )

                st.metric(
                    "QQQ return",
                    (
                        f"{worst_qqq_return_period['QQQ Return']:.1%}"
                    ),
                )

                st.write(
                    f"Started "
                    f"**{worst_qqq_return_period['Start Date']:%B %d, %Y}**"
                )

        # =================================================
        # DATA TAB
        # =================================================
        with data_tab:
            st.markdown(
                "### Every historical starting date"
            )

            st.caption(
                "Use the column controls to sort and inspect "
                "individual periods."
            )

            table = results[
                [
                    "Start Date",
                    "End Date",
                    "Total Contributed",
                    "QQQ Ending Value",
                    "TQQQ Ending Value",
                    "QQQ Return",
                    "TQQQ Return",
                    "TQQQ Minus QQQ",
                    "Winner",
                ]
            ].copy()

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
                height=500,
                column_config={
                    "Start Date": (
                        st.column_config.DateColumn(
                            "Start date"
                        )
                    ),
                    "End Date": (
                        st.column_config.DateColumn(
                            "End date"
                        )
                    ),
                    "Total Contributed": (
                        st.column_config.NumberColumn(
                            "Contributed",
                            format="$%.2f",
                        )
                    ),
                    "QQQ Ending Value": (
                        st.column_config.NumberColumn(
                            "QQQ ending value",
                            format="$%.2f",
                        )
                    ),
                    "TQQQ Ending Value": (
                        st.column_config.NumberColumn(
                            "TQQQ ending value",
                            format="$%.2f",
                        )
                    ),
                    "QQQ Return": (
                        st.column_config.NumberColumn(
                            "QQQ return",
                            format="percent",
                        )
                    ),
                    "TQQQ Return": (
                        st.column_config.NumberColumn(
                            "TQQQ return",
                            format="percent",
                        )
                    ),
                    "TQQQ Minus QQQ": (
                        st.column_config.NumberColumn(
                            "TQQQ minus QQQ",
                            format="$%.2f",
                        )
                    ),
                },
            )

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label=(
                    f"Download all {years}-year periods"
                ),
                data=csv_data,
                file_name=(
                    f"qqq_tqqq_{years}_year_periods.csv"
                ),
                mime="text/csv",
                key=f"download_{years}",
            )


# =========================================================
# COMBINED DOWNLOAD
# =========================================================
if all_results:
    combined_results = pd.concat(
        all_results,
        ignore_index=True,
    )

    combined_csv = combined_results.to_csv(
        index=False
    ).encode("utf-8")

    st.divider()

    st.download_button(
        label="Download all 1-, 5-, and 10-year results",
        data=combined_csv,
        file_name="qqq_tqqq_all_backtests.csv",
        mime="text/csv",
    )


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    """
    Synthetic and reconstructed results do not represent actual
    investable ETF returns before each fund existed. Results are
    provided for research and educational purposes.
    """
)