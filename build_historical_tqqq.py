from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# FILE LOCATIONS
# =========================================================

BASE_FOLDER = Path(__file__).resolve().parent

QQQ_FILE = BASE_FOLDER / "qqq.csv"
TQQQ_FILE = BASE_FOLDER / "tqqq.csv"

OUTPUT_CSV = BASE_FOLDER / "historical_tqqq_1999.csv"
OUTPUT_CHART = BASE_FOLDER / "historical_tqqq_1999.png"


# =========================================================
# LOAD AND CLEAN PRICE DATA
# =========================================================

def load_prices(file_path: Path, ticker: str) -> pd.DataFrame:
    """
    Load one price CSV and calculate daily close-to-close returns.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path.name}. "
            "Make sure it is in the same folder as this Python file."
        )

    data = pd.read_csv(file_path)

    # Remove accidental spaces from column names.
    data.columns = data.columns.str.strip()

    required_columns = {"Date", "Close"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"{file_path.name} must contain Date and Close columns. "
            f"Columns found: {list(data.columns)}"
        )

    # Convert columns to the proper types.
    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
    )

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce",
    )

    # Remove invalid rows and duplicate dates.
    data = (
        data.dropna(subset=["Date", "Close"])
        .drop_duplicates(subset="Date", keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Keep only the columns we need.
    data = data[
        [
            "Date",
            "Close",
        ]
    ].rename(
        columns={
            "Close": f"{ticker} Close",
        }
    )

    # Calculate daily percentage return.
    data[f"{ticker} Daily Return"] = (
        data[f"{ticker} Close"].pct_change()
    )

    return data


# =========================================================
# LOAD QQQ AND TQQQ
# =========================================================

qqq = load_prices(
    QQQ_FILE,
    "QQQ",
)

tqqq = load_prices(
    TQQQ_FILE,
    "TQQQ",
)


# =========================================================
# CALCULATE THE QQQ -> TQQQ RELATIONSHIP
# =========================================================

# Match QQQ and TQQQ by trading date.
overlap = pd.merge(
    qqq,
    tqqq,
    on="Date",
    how="inner",
)

overlap = overlap.dropna(
    subset=[
        "QQQ Daily Return",
        "TQQQ Daily Return",
    ]
)


if overlap.empty:
    raise ValueError(
        "The QQQ and TQQQ files do not have overlapping return data."
    )


# Regression formula:
#
# TQQQ daily return =
# intercept + slope × QQQ daily return

slope, intercept = np.polyfit(
    overlap["QQQ Daily Return"],
    overlap["TQQQ Daily Return"],
    1,
)


# Calculate predicted returns during the real overlap period.
predicted_overlap_returns = (
    intercept
    + slope * overlap["QQQ Daily Return"]
)


# Calculate R-squared.
residual_sum_squares = np.sum(
    (
        overlap["TQQQ Daily Return"]
        - predicted_overlap_returns
    ) ** 2
)

total_sum_squares = np.sum(
    (
        overlap["TQQQ Daily Return"]
        - overlap["TQQQ Daily Return"].mean()
    ) ** 2
)

r_squared = (
    1
    - residual_sum_squares / total_sum_squares
)


# =========================================================
# SIMULATE TQQQ RETURNS FOR EVERY QQQ DATE
# =========================================================

history = qqq.copy()


history["Simulated TQQQ Daily Return"] = (
    slope * history["QQQ Daily Return"]
    + intercept
)


# Make sure no simulated daily return is -100% or lower.
invalid_returns = (
    history["Simulated TQQQ Daily Return"]
    .dropna()
    .le(-1)
)

if invalid_returns.any():
    raise ValueError(
        "The formula generated a return at or below -100%. "
        "A valid price series cannot be created."
    )


# Add actual TQQQ prices where they exist.
history = pd.merge(
    history,
    tqqq[
        [
            "Date",
            "TQQQ Close",
            "TQQQ Daily Return",
        ]
    ],
    on="Date",
    how="left",
)


# =========================================================
# FIND THE FIRST ACTUAL TQQQ PRICE
# =========================================================

anchor_index = history[
    "TQQQ Close"
].first_valid_index()


if anchor_index is None:
    raise ValueError(
        "No actual TQQQ price was found."
    )


anchor_date = history.loc[
    anchor_index,
    "Date",
]

anchor_close = history.loc[
    anchor_index,
    "TQQQ Close",
]


# =========================================================
# CREATE A FULLY SIMULATED TQQQ PRICE SERIES
# =========================================================

history["Fully Simulated TQQQ Close"] = np.nan


# Anchor the simulated series to the first real TQQQ close.
history.loc[
    anchor_index,
    "Fully Simulated TQQQ Close",
] = anchor_close


# ---------------------------------------------------------
# Work backward from 2010 to 1999
# ---------------------------------------------------------

for row_index in range(
    anchor_index - 1,
    -1,
    -1,
):
    next_day_return = history.loc[
        row_index + 1,
        "Simulated TQQQ Daily Return",
    ]

    next_day_price = history.loc[
        row_index + 1,
        "Fully Simulated TQQQ Close",
    ]

    previous_day_price = (
        next_day_price
        / (1 + next_day_return)
    )

    history.loc[
        row_index,
        "Fully Simulated TQQQ Close",
    ] = previous_day_price


# ---------------------------------------------------------
# Work forward from the first real TQQQ date
# ---------------------------------------------------------

for row_index in range(
    anchor_index + 1,
    len(history),
):
    current_day_return = history.loc[
        row_index,
        "Simulated TQQQ Daily Return",
    ]

    previous_day_price = history.loc[
        row_index - 1,
        "Fully Simulated TQQQ Close",
    ]

    current_day_price = (
        previous_day_price
        * (1 + current_day_return)
    )

    history.loc[
        row_index,
        "Fully Simulated TQQQ Close",
    ] = current_day_price


# =========================================================
# CREATE THE FINAL HISTORICAL TQQQ SERIES
# =========================================================

# Before TQQQ existed:
# use simulated TQQQ.
#
# After TQQQ existed:
# use the actual TQQQ closing price.

history["Historical TQQQ Close"] = np.where(
    history["TQQQ Close"].notna(),
    history["TQQQ Close"],
    history["Fully Simulated TQQQ Close"],
)


history["Historical TQQQ Daily Return"] = (
    history["Historical TQQQ Close"]
    .pct_change()
)


history["Series Source"] = np.where(
    history["TQQQ Close"].notna(),
    "Actual TQQQ",
    "Simulated from QQQ",
)


# =========================================================
# SAVE THE HISTORICAL DATA
# =========================================================

output_columns = [
    "Date",
    "QQQ Close",
    "QQQ Daily Return",
    "Simulated TQQQ Daily Return",
    "Fully Simulated TQQQ Close",
    "TQQQ Close",
    "TQQQ Daily Return",
    "Historical TQQQ Close",
    "Historical TQQQ Daily Return",
    "Series Source",
]


history[
    output_columns
].to_csv(
    OUTPUT_CSV,
    index=False,
)


# =========================================================
# PRINT RESULTS IN THE TERMINAL
# =========================================================

print()
print("Historical TQQQ data created successfully.")
print()

print(
    f"First QQQ date: "
    f"{history['Date'].min().date()}"
)

print(
    f"First actual TQQQ date: "
    f"{anchor_date.date()}"
)

print(
    f"Last available date: "
    f"{history['Date'].max().date()}"
)

print()

print(
    "Fitted formula:"
)

print(
    f"TQQQ return = "
    f"{slope:.7f} × QQQ return "
    f"{intercept:+.9f}"
)

print(
    f"Daily intercept as percent: "
    f"{intercept * 100:+.6f}%"
)

print(
    f"R-squared: "
    f"{r_squared:.6f}"
)

print()

print(
    f"CSV saved as: "
    f"{OUTPUT_CSV.name}"
)


# =========================================================
# CREATE AND SAVE A GRAPH
# =========================================================

figure, axis = plt.subplots(
    figsize=(12, 7)
)


axis.plot(
    history["Date"],
    history["Historical TQQQ Close"],
    label=(
        "Historical TQQQ "
        "(simulated before launch, actual afterward)"
    ),
)


axis.plot(
    history["Date"],
    history["Fully Simulated TQQQ Close"],
    label="Fully simulated TQQQ",
    alpha=0.75,
)


axis.axvline(
    anchor_date,
    linestyle="--",
    linewidth=1.25,
    label="First actual TQQQ date",
)


axis.set_yscale("log")

axis.set_xlabel(
    "Date"
)

axis.set_ylabel(
    "TQQQ price-like series, log scale"
)

axis.set_title(
    "Backfilled TQQQ History Using QQQ Daily Returns"
)

axis.grid(
    True,
    alpha=0.25,
)

axis.legend()

figure.tight_layout()


figure.savefig(
    OUTPUT_CHART,
    dpi=180,
)


plt.show()


print(
    f"Chart saved as: "
    f"{OUTPUT_CHART.name}"
)