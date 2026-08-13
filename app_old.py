from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from simulation import (
    simulate_all_rolling_periods,
    simulate_monthly_investment,
)

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="QQQ vs TQQQ Daily Return Analyzer",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("📈 QQQ vs TQQQ Daily Return Analyzer")

st.write(
    """
    This website compares TQQQ's actual daily return with exactly three times
    QQQ's daily return.

    The calculation used is:

    **Tracking difference = TQQQ daily return − 3 × QQQ daily return**
    """
)


# ---------------------------------------------------------
# FIND THE CSV FILES
# ---------------------------------------------------------

CURRENT_FOLDER = Path(__file__).parent

QQQ_FILE = CURRENT_FOLDER / "qqq.csv"
TQQQ_FILE = CURRENT_FOLDER / "tqqq.csv"


# Stop the app and explain the problem if a file is missing.
if not QQQ_FILE.exists():
    st.error(
        "The file qqq.csv was not found. "
        "Make sure your QQQ data file is renamed to qqq.csv "
        "and placed in the same folder as app.py."
    )
    st.stop()

if not TQQQ_FILE.exists():
    st.error(
        "The file tqqq.csv was not found. "
        "Make sure your TQQQ data file is renamed to tqqq.csv "
        "and placed in the same folder as app.py."
    )
    st.stop()


# ---------------------------------------------------------
# LOAD AND CLEAN DATA
# ---------------------------------------------------------

@st.cache_data
def load_price_data(file_path: Path, ticker: str) -> pd.DataFrame:
    """
    Load a stock-price CSV and calculate close-to-close daily returns.
    """

    data = pd.read_csv(file_path)

    # Remove extra spaces from column names.
    data.columns = data.columns.str.strip()

    required_columns = {"Date", "Close"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"{ticker} CSV must contain Date and Close columns. "
            f"Columns found: {list(data.columns)}"
        )

    # Convert dates and closing prices.
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

    # Remove bad rows.
    data = data.dropna(subset=["Date", "Close"])

    # Remove duplicate dates and sort chronologically.
    data = (
        data.drop_duplicates(subset="Date", keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Calculate close-to-close daily return.
    data[f"{ticker} Daily Return"] = data["Close"].pct_change()

    # Rename Close so the two datasets remain distinguishable.
    data = data.rename(columns={"Close": f"{ticker} Close"})

    return data[
        [
            "Date",
            f"{ticker} Close",
            f"{ticker} Daily Return",
        ]
    ]


try:
    qqq = load_price_data(QQQ_FILE, "QQQ")
    tqqq = load_price_data(TQQQ_FILE, "TQQQ")

except Exception as error:
    st.error(f"An error occurred while reading the files: {error}")
    st.stop()


# ---------------------------------------------------------
# ALIGN QQQ AND TQQQ BY DATE
# ---------------------------------------------------------

comparison = pd.merge(
    qqq,
    tqqq,
    on="Date",
    how="inner",
)

comparison = comparison.dropna(
    subset=[
        "QQQ Daily Return",
        "TQQQ Daily Return",
    ]
).reset_index(drop=True)


if comparison.empty:
    st.error(
        "The two files do not contain overlapping trading dates."
    )
    st.stop()


# ---------------------------------------------------------
# CALCULATIONS
# ---------------------------------------------------------

comparison["Exact 3x QQQ Return"] = (
    3 * comparison["QQQ Daily Return"]
)

comparison["TQQQ Minus Exact 3x"] = (
    comparison["TQQQ Daily Return"]
    - comparison["Exact 3x QQQ Return"]
)

comparison["Absolute Tracking Difference"] = (
    comparison["TQQQ Minus Exact 3x"].abs()
)


# Convert the return columns to NumPy arrays for regression.
qqq_returns = comparison["QQQ Daily Return"].to_numpy()
tqqq_returns = comparison["TQQQ Daily Return"].to_numpy()


# Regression:
# TQQQ return = intercept + slope × QQQ return
slope, intercept = np.polyfit(
    qqq_returns,
    tqqq_returns,
    1,
)

predicted_tqqq_returns = (
    intercept + slope * qqq_returns
)


# R-squared calculation.
residual_sum_squares = np.sum(
    (tqqq_returns - predicted_tqqq_returns) ** 2
)

total_sum_squares = np.sum(
    (tqqq_returns - tqqq_returns.mean()) ** 2
)

r_squared = 1 - (
    residual_sum_squares / total_sum_squares
)


# Summary statistics.
average_gap = comparison["TQQQ Minus Exact 3x"].mean()
median_gap = comparison["TQQQ Minus Exact 3x"].median()
average_absolute_gap = comparison[
    "Absolute Tracking Difference"
].mean()

within_010 = (
    comparison["Absolute Tracking Difference"] <= 0.001
).mean()

within_025 = (
    comparison["Absolute Tracking Difference"] <= 0.0025
).mean()

within_050 = (
    comparison["Absolute Tracking Difference"] <= 0.005
).mean()


# ---------------------------------------------------------
# SIDEBAR DATE FILTER
# ---------------------------------------------------------

st.sidebar.header("Date Settings")

minimum_date = comparison["Date"].min().date()
maximum_date = comparison["Date"].max().date()

selected_dates = st.sidebar.date_input(
    "Choose a date range",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date,
)


if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    selected_start_date = pd.Timestamp(selected_dates[0])
    selected_end_date = pd.Timestamp(selected_dates[1])

    filtered = comparison[
        comparison["Date"].between(
            selected_start_date,
            selected_end_date,
        )
    ].copy()

else:
    filtered = comparison.copy()


if filtered.empty:
    st.warning("No trading days were found in that date range.")
    st.stop()


# ---------------------------------------------------------
# DATASET INFORMATION
# ---------------------------------------------------------

st.subheader("Dataset Information")

information_column_1, information_column_2, information_column_3 = st.columns(3)

information_column_1.metric(
    "Matching Trading Days",
    f"{len(comparison):,}",
)

information_column_2.metric(
    "First Matching Date",
    comparison["Date"].min().strftime("%B %d, %Y"),
)

information_column_3.metric(
    "Last Matching Date",
    comparison["Date"].max().strftime("%B %d, %Y"),
)


# ---------------------------------------------------------
# MAIN RESULTS
# ---------------------------------------------------------

st.subheader("Daily Tracking Results")

metric_column_1, metric_column_2, metric_column_3 = st.columns(3)

metric_column_1.metric(
    "Average Daily Difference",
    f"{average_gap * 100:.4f} percentage points",
)

metric_column_2.metric(
    "Median Daily Difference",
    f"{median_gap * 100:.4f} percentage points",
)

metric_column_3.metric(
    "Average Absolute Difference",
    f"{average_absolute_gap * 100:.4f} percentage points",
)


metric_column_4, metric_column_5, metric_column_6 = st.columns(3)

metric_column_4.metric(
    "Within ±0.10 Percentage Points",
    f"{within_010:.1%}",
)

metric_column_5.metric(
    "Within ±0.25 Percentage Points",
    f"{within_025:.1%}",
)

metric_column_6.metric(
    "Within ±0.50 Percentage Points",
    f"{within_050:.1%}",
)


# ---------------------------------------------------------
# REGRESSION FORMULA
# ---------------------------------------------------------

st.subheader("Best-Fitting Daily Formula")

intercept_percentage = intercept * 100

if intercept_percentage >= 0:
    intercept_text = f"+ {intercept_percentage:.4f}%"
else:
    intercept_text = f"− {abs(intercept_percentage):.4f}%"

st.latex(
    rf"""
    R_{{TQQQ}}
    \approx
    {slope:.4f}
    R_{{QQQ}}
    {intercept_text}
    """
)

st.write(
    f"""
    According to the overlapping historical data, TQQQ behaved approximately
    like **{slope:.4f} times QQQ's daily return**, with a daily intercept of
    **{intercept_percentage:.4f}%**.

    The regression has an R² of **{r_squared:.6f}**.
    """
)


# ---------------------------------------------------------
# DAILY RETURN SCATTER PLOT
# ---------------------------------------------------------

st.subheader("Actual TQQQ Return vs Exact 3× QQQ")

fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))

exact_3x_percent = (
    filtered["Exact 3x QQQ Return"] * 100
)

actual_tqqq_percent = (
    filtered["TQQQ Daily Return"] * 100
)

ax_scatter.scatter(
    exact_3x_percent,
    actual_tqqq_percent,
    alpha=0.35,
    s=14,
)

minimum_plot_value = min(
    exact_3x_percent.min(),
    actual_tqqq_percent.min(),
)

maximum_plot_value = max(
    exact_3x_percent.max(),
    actual_tqqq_percent.max(),
)

ax_scatter.plot(
    [minimum_plot_value, maximum_plot_value],
    [minimum_plot_value, maximum_plot_value],
    linewidth=1.5,
    label="Perfect exact 3× tracking",
)

ax_scatter.set_xlabel(
    "Exact 3× QQQ Daily Return (%)"
)

ax_scatter.set_ylabel(
    "Actual TQQQ Daily Return (%)"
)

ax_scatter.set_title(
    "TQQQ Daily Return Compared with Exact 3× QQQ"
)

ax_scatter.legend()
ax_scatter.grid(True, alpha=0.25)

fig_scatter.tight_layout()

st.pyplot(fig_scatter)

plt.close(fig_scatter)


# ---------------------------------------------------------
# TRACKING DIFFERENCE HISTOGRAM
# ---------------------------------------------------------

st.subheader("Distribution of Daily Tracking Differences")

fig_histogram, ax_histogram = plt.subplots(figsize=(10, 6))

tracking_difference_percent = (
    filtered["TQQQ Minus Exact 3x"] * 100
)

ax_histogram.hist(
    tracking_difference_percent,
    bins=100,
)

ax_histogram.axvline(
    tracking_difference_percent.mean(),
    linestyle="--",
    linewidth=1.5,
    label="Average difference",
)

ax_histogram.set_xlabel(
    "TQQQ Return Minus Exact 3× QQQ "
    "(Percentage Points)"
)

ax_histogram.set_ylabel(
    "Number of Trading Days"
)

ax_histogram.set_title(
    "Distribution of TQQQ's Daily Tracking Difference"
)

ax_histogram.legend()
ax_histogram.grid(True, alpha=0.25)

fig_histogram.tight_layout()

st.pyplot(fig_histogram)

plt.close(fig_histogram)


# ---------------------------------------------------------
# CUMULATIVE COMPARISON
# ---------------------------------------------------------

st.subheader("Growth of $100")

cumulative = filtered.copy()

cumulative["Actual TQQQ Growth"] = (
    100
    * (
        1 + cumulative["TQQQ Daily Return"]
    ).cumprod()
)

cumulative["Exact 3x Growth"] = (
    100
    * (
        1 + cumulative["Exact 3x QQQ Return"]
    ).cumprod()
)

fig_cumulative, ax_cumulative = plt.subplots(
    figsize=(11, 6)
)

ax_cumulative.plot(
    cumulative["Date"],
    cumulative["Actual TQQQ Growth"],
    label="Actual TQQQ",
)

ax_cumulative.plot(
    cumulative["Date"],
    cumulative["Exact 3x Growth"],
    label="Synthetic Exact 3× QQQ",
)

use_log_scale = st.checkbox(
    "Use logarithmic scale",
    value=True,
)

if use_log_scale:
    ax_cumulative.set_yscale("log")

ax_cumulative.set_xlabel("Date")
ax_cumulative.set_ylabel("Growth of $100")

ax_cumulative.set_title(
    "Compounding Effect of Daily Tracking Differences"
)

ax_cumulative.legend()
ax_cumulative.grid(True, alpha=0.25)

fig_cumulative.tight_layout()

st.pyplot(fig_cumulative)

plt.close(fig_cumulative)


# ---------------------------------------------------------
# LARGEST TRACKING DIFFERENCES
# ---------------------------------------------------------

st.subheader("Largest Daily Tracking Differences")

largest_differences = (
    filtered.sort_values(
        "Absolute Tracking Difference",
        ascending=False,
    )
    .head(20)
    .copy()
)

display_largest = largest_differences[
    [
        "Date",
        "QQQ Daily Return",
        "Exact 3x QQQ Return",
        "TQQQ Daily Return",
        "TQQQ Minus Exact 3x",
    ]
].copy()

display_largest["Date"] = (
    display_largest["Date"].dt.strftime("%Y-%m-%d")
)

percentage_columns = [
    "QQQ Daily Return",
    "Exact 3x QQQ Return",
    "TQQQ Daily Return",
    "TQQQ Minus Exact 3x",
]

for column in percentage_columns:
    display_largest[column] = (
        display_largest[column] * 100
    )

st.dataframe(
    display_largest.style.format(
        {
            "QQQ Daily Return": "{:.3f}%",
            "Exact 3x QQQ Return": "{:.3f}%",
            "TQQQ Daily Return": "{:.3f}%",
            "TQQQ Minus Exact 3x": "{:+.3f}%",
        }
    ),
    use_container_width=True,
)


# ---------------------------------------------------------
# COMPLETE DATA TABLE
# ---------------------------------------------------------

with st.expander("View Complete Daily Dataset"):
    display_data = filtered.copy()

    display_data["Date"] = (
        display_data["Date"].dt.strftime("%Y-%m-%d")
    )

    st.dataframe(
        display_data,
        use_container_width=True,
    )


# ---------------------------------------------------------
# DOWNLOAD RESULTS
# ---------------------------------------------------------

download_data = comparison.copy()

download_data["Date"] = (
    download_data["Date"].dt.strftime("%Y-%m-%d")
)

csv_download = download_data.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Complete Comparison CSV",
    data=csv_download,
    file_name="qqq_tqqq_daily_comparison.csv",
    mime="text/csv",
)


# ---------------------------------------------------------
# EXPLANATION
# ---------------------------------------------------------

st.subheader("What the Difference Means")

st.write(
    """
    TQQQ targets approximately three times QQQ's **daily** performance,
    but it will not equal exactly three times QQQ every day.

    The difference can come from:

    - Fund expenses
    - Financing and borrowing costs
    - Derivative pricing
    - Daily rebalancing
    - Differences between market price and net asset value
    - Differences in the timestamps or pricing sources used by the CSV files

    This analysis uses close-to-close daily returns from the `Close` column.
    """
)
st.subheader("Historical TQQQ Back to 1999")

historical_tqqq = pd.read_csv(
    "historical_tqqq_1999.csv",
    parse_dates=["Date"],
)

st.line_chart(
    historical_tqqq.set_index("Date")[
        [
            "QQQ Close",
            "Historical TQQQ Close",
        ]
    ]
)

st.dataframe(
    historical_tqqq[
        [
            "Date",
            "QQQ Close",
            "Historical TQQQ Close",
            "Historical TQQQ Daily Return",
            "Series Source",
        ]
    ],
    use_container_width=True,
)
# ---------------------------------------------------------
# ALL ROLLING INVESTMENT PERIODS
# ---------------------------------------------------------
st.divider()

st.header("Every Possible QQQ vs TQQQ Investment Period")

st.markdown(
    """
    This backtest begins on **every possible shared trading date**.

    For each period, it invests **\\$5,000 initially** and then adds
    **\\$100 on the first available trading day of every following
    calendar month**. It compares the final QQQ and TQQQ balances
    after 1, 5, or 10 years.
    """
)

INITIAL_INVESTMENT = 5000
MONTHLY_INVESTMENT = 100

ROLLING_PERIODS = {
    "1 Year": 1,
    "5 Years": 5,
    "10 Years": 10,
}

HISTORICAL_TQQQ_FILE = (
    CURRENT_FOLDER / "historical_tqqq_1999.csv"
)

if not HISTORICAL_TQQQ_FILE.exists():
    st.error(
        "historical_tqqq_1999.csv was not found. "
        "Run build_historical_tqqq.py first."
    )
    st.stop()

historical_rolling_data = pd.read_csv(
    HISTORICAL_TQQQ_FILE
)

historical_rolling_data.columns = (
    historical_rolling_data.columns.str.strip()
)

historical_rolling_data["Date"] = pd.to_datetime(
    historical_rolling_data["Date"],
    errors="coerce",
)

historical_rolling_data["QQQ Close"] = pd.to_numeric(
    historical_rolling_data["QQQ Close"],
    errors="coerce",
)

historical_rolling_data["Historical TQQQ Close"] = (
    pd.to_numeric(
        historical_rolling_data[
            "Historical TQQQ Close"
        ],
        errors="coerce",
    )
)

rolling_prices = historical_rolling_data[
    [
        "Date",
        "QQQ Close",
        "Historical TQQQ Close",
    ]
].rename(
    columns={
        "Historical TQQQ Close": "TQQQ Close",
    }
)

rolling_prices = (
    rolling_prices.dropna(
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

rolling_tabs = st.tabs(
    list(ROLLING_PERIODS.keys())
)

all_rolling_results = []

for tab, (period_name, years) in zip(
    rolling_tabs,
    ROLLING_PERIODS.items(),
):
    with tab:
        with st.spinner(
            f"Calculating every possible "
            f"{period_name.lower()} period..."
        ):
            rolling_results = (
                simulate_all_rolling_periods(
                    prices=rolling_prices,
                    years=years,
                    initial_investment=(
                        INITIAL_INVESTMENT
                    ),
                    monthly_investment=(
                        MONTHLY_INVESTMENT
                    ),
                )
            )

        if rolling_results.empty:
            st.warning(
                f"There is not enough overlapping data "
                f"to calculate complete {period_name.lower()} "
                f"periods."
            )
            continue

        all_rolling_results.append(
            rolling_results.copy()
        )

        total_periods = len(rolling_results)

        tqqq_wins = int(
            (
                rolling_results["Winner"]
                == "TQQQ"
            ).sum()
        )

        qqq_wins = int(
            (
                rolling_results["Winner"]
                == "QQQ"
            ).sum()
        )

        ties = int(
            (
                rolling_results["Winner"]
                == "Tie"
            ).sum()
        )

        tqqq_win_percentage = (
            tqqq_wins / total_periods
        )

        qqq_win_percentage = (
            qqq_wins / total_periods
        )

        median_qqq_ending_value = (
            rolling_results[
                "QQQ Ending Value"
            ].median()
        )

        median_tqqq_ending_value = (
            rolling_results[
                "TQQQ Ending Value"
            ].median()
        )

        average_difference = (
            rolling_results[
                "TQQQ Minus QQQ"
            ].mean()
        )

        st.subheader(f"{period_name} Rolling Results")

        st.caption(
            f"Tested {total_periods:,} possible starting "
            f"trading dates from "
            f"{rolling_results['Start Date'].min():%B %d, %Y} "
            f"through "
            f"{rolling_results['Start Date'].max():%B %d, %Y}."
        )

        win_column, qqq_column, periods_column = (
            st.columns(3)
        )

        win_column.metric(
            "TQQQ Outperformed",
            f"{tqqq_win_percentage:.1%}",
            delta=(
                f"{tqqq_wins:,} of "
                f"{total_periods:,} periods"
            ),
        )

        qqq_column.metric(
            "QQQ Outperformed",
            f"{qqq_win_percentage:.1%}",
            delta=(
                f"{qqq_wins:,} of "
                f"{total_periods:,} periods"
            ),
        )

        periods_column.metric(
            "Periods Tested",
            f"{total_periods:,}",
            delta=f"{ties:,} ties",
        )

        qqq_median_column, tqqq_median_column, difference_column = (
            st.columns(3)
        )

        qqq_median_column.metric(
            "Median QQQ Ending Value",
            f"${median_qqq_ending_value:,.2f}",
        )

        tqqq_median_column.metric(
            "Median TQQQ Ending Value",
            f"${median_tqqq_ending_value:,.2f}",
        )

        difference_column.metric(
            "Average TQQQ Minus QQQ",
            f"${average_difference:,.2f}",
        )

        # -------------------------------------------------
        # ENDING VALUE BY STARTING DATE
        # -------------------------------------------------
        st.subheader(
            "Ending Value for Every Starting Date"
        )

        st.write(
            """
            Each point represents one complete investment period.
            The horizontal axis is the date on which the $5,000
            initial investment was made.
            """
        )

        ending_value_chart = (
            rolling_results[
                [
                    "Start Date",
                    "QQQ Ending Value",
                    "TQQQ Ending Value",
                ]
            ]
            .set_index("Start Date")
        )

        st.line_chart(
            ending_value_chart,
            y_label="Ending Portfolio Value ($)",
        )

        # -------------------------------------------------
        # TQQQ ADVANTAGE OR DISADVANTAGE
        # -------------------------------------------------
        st.subheader(
            "TQQQ Advantage Over QQQ"
        )

        st.write(
            """
            Values above zero mean TQQQ finished ahead.
            Values below zero mean QQQ finished ahead.
            """
        )

        difference_chart = (
            rolling_results[
                [
                    "Start Date",
                    "TQQQ Minus QQQ",
                ]
            ]
            .set_index("Start Date")
        )

        st.line_chart(
            difference_chart,
            y_label="TQQQ Minus QQQ ($)",
        )

        # -------------------------------------------------
        # RETURN COMPARISON
        # -------------------------------------------------
        st.subheader(
            "Return on Total Contributions"
        )

        return_chart = (
            rolling_results[
                [
                    "Start Date",
                    "QQQ Return",
                    "TQQQ Return",
                ]
            ]
            .set_index("Start Date")
            * 100
        )

        return_chart = return_chart.rename(
            columns={
                "QQQ Return": "QQQ Return (%)",
                "TQQQ Return": "TQQQ Return (%)",
            }
        )

        st.line_chart(
            return_chart,
            y_label="Return (%)",
        )

        # -------------------------------------------------
        # BEST AND WORST PERIODS
        # -------------------------------------------------
        st.subheader("Notable Periods")

        best_tqqq_period = rolling_results.loc[
            rolling_results[
                "TQQQ Minus QQQ"
            ].idxmax()
        ]

        worst_tqqq_period = rolling_results.loc[
            rolling_results[
                "TQQQ Minus QQQ"
            ].idxmin()
        ]

        best_column, worst_column = st.columns(2)

        with best_column:
            st.markdown(
                "#### Biggest TQQQ Victory"
            )

            st.metric(
                "TQQQ Advantage",
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
                f"QQQ ended with "
                f"**${best_tqqq_period['QQQ Ending Value']:,.2f}**."
            )

            st.write(
                f"TQQQ ended with "
                f"**${best_tqqq_period['TQQQ Ending Value']:,.2f}**."
            )

        with worst_column:
            st.markdown(
                "#### Biggest TQQQ Defeat"
            )

            st.metric(
                "TQQQ Minus QQQ",
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
                f"QQQ ended with "
                f"**${worst_tqqq_period['QQQ Ending Value']:,.2f}**."
            )

            st.write(
                f"TQQQ ended with "
                f"**${worst_tqqq_period['TQQQ Ending Value']:,.2f}**."
            )

        # -------------------------------------------------
        # FULL DATA TABLE
        # -------------------------------------------------
        st.subheader(
            f"All {total_periods:,} Periods"
        )

        display_results = rolling_results.copy()

        display_results["Start Date"] = (
            display_results["Start Date"].dt.strftime(
                "%Y-%m-%d"
            )
        )

        display_results["End Date"] = (
            display_results["End Date"].dt.strftime(
                "%Y-%m-%d"
            )
        )

        money_columns = [
            "Total Contributed",
            "QQQ Ending Value",
            "TQQQ Ending Value",
            "QQQ Profit",
            "TQQQ Profit",
            "TQQQ Minus QQQ",
        ]

        for column in money_columns:
            display_results[column] = (
                display_results[column].map(
                    lambda value: f"${value:,.2f}"
                )
            )

        display_results["QQQ Return"] = (
            display_results["QQQ Return"].map(
                lambda value: f"{value:.2%}"
            )
        )

        display_results["TQQQ Return"] = (
            display_results["TQQQ Return"].map(
                lambda value: f"{value:.2%}"
            )
        )

        st.dataframe(
            display_results[
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
            ],
            use_container_width=True,
            hide_index=True,
        )

        period_csv = rolling_results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label=(
                f"Download Every {period_name} Period"
            ),
            data=period_csv,
            file_name=(
                f"qqq_tqqq_all_"
                f"{years}_year_periods.csv"
            ),
            mime="text/csv",
            key=f"download_{years}_year_periods",
        )

# ---------------------------------------------------------
# COMBINED DOWNLOAD
# ---------------------------------------------------------
if all_rolling_results:
    combined_rolling_results = pd.concat(
        all_rolling_results,
        ignore_index=True,
    )

    combined_csv = (
        combined_rolling_results.to_csv(
            index=False
        ).encode("utf-8")
    )

    st.download_button(
        label="Download All Rolling Period Results",
        data=combined_csv,
        file_name=(
            "qqq_tqqq_all_rolling_periods.csv"
        ),
        mime="text/csv",
    )