from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


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