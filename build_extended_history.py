from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# FILE LOCATIONS
# =========================================================
BASE_FOLDER = Path(__file__).resolve().parent

NDX_FILE = BASE_FOLDER / "ndx_d.csv"
QQQ_FILE = BASE_FOLDER / "qqq.csv"
TQQQ_FILE = BASE_FOLDER / "tqqq.csv"

OUTPUT_FILE = BASE_FOLDER / "extended_history_1938.csv"


# =========================================================
# IMPORTANT DATES
# =========================================================
NASDAQ_100_LAUNCH_DATE = pd.Timestamp("1985-01-31")


# =========================================================
# LOAD AND CLEAN PRICE FILES
# =========================================================
def load_price_file(
    file_path: Path,
    output_price_name: str,
) -> pd.DataFrame:
    """
    Load a CSV with Date and Close columns.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path.name}."
        )

    data = pd.read_csv(file_path)

    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
    )

    required_columns = {"Date", "Close"}

    missing_columns = (
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    data = data[
        [
            "Date",
            "Close",
        ]
    ].copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
    )

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce",
    )

    data = (
        data.dropna(
            subset=[
                "Date",
                "Close",
            ]
        )
        .drop_duplicates(
            subset="Date",
            keep="last",
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if data.empty:
        raise ValueError(
            f"{file_path.name} contains no usable data."
        )

    if (data["Close"] <= 0).any():
        raise ValueError(
            f"{file_path.name} contains a price "
            f"that is zero or negative."
        )

    data = data.rename(
        columns={
            "Close": output_price_name,
        }
    )

    return data


# =========================================================
# ESTIMATE ACTUAL QQQ/TQQQ RELATIONSHIP
# =========================================================
def estimate_tqqq_model(
    actual_qqq: pd.DataFrame,
    actual_tqqq: pd.DataFrame,
) -> tuple[float, float, pd.DataFrame]:
    """
    Estimate:

        TQQQ return = alpha + beta * QQQ return

    from the actual overlapping QQQ and TQQQ history.
    """

    overlap = pd.merge(
        actual_qqq,
        actual_tqqq,
        on="Date",
        how="inner",
    )

    overlap = (
        overlap.sort_values("Date")
        .reset_index(drop=True)
    )

    overlap["Actual QQQ Return"] = (
        overlap["Actual QQQ Close"].pct_change()
    )

    overlap["Actual TQQQ Return"] = (
        overlap["Actual TQQQ Close"].pct_change()
    )

    overlap = overlap.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    overlap = overlap.dropna(
        subset=[
            "Actual QQQ Return",
            "Actual TQQQ Return",
        ]
    )

    # Remove obviously invalid or split-distorted rows.
    overlap = overlap[
        overlap["Actual QQQ Return"].between(
            -0.40,
            0.40,
        )
        & overlap["Actual TQQQ Return"].between(
            -0.99,
            2.00,
        )
    ].copy()

    if len(overlap) < 100:
        raise ValueError(
            "Not enough usable QQQ/TQQQ overlap "
            "was found to fit the model."
        )

    beta, alpha = np.polyfit(
        overlap["Actual QQQ Return"],
        overlap["Actual TQQQ Return"],
        1,
    )

    return (
        float(alpha),
        float(beta),
        overlap,
    )


# =========================================================
# MODEL STATISTICS
# =========================================================
def calculate_model_statistics(
    overlap: pd.DataFrame,
    alpha: float,
    beta: float,
) -> dict:
    predicted = (
        alpha
        + beta * overlap["Actual QQQ Return"]
    )

    actual = overlap[
        "Actual TQQQ Return"
    ]

    residuals = actual - predicted

    correlation = overlap[
        [
            "Actual QQQ Return",
            "Actual TQQQ Return",
        ]
    ].corr().iloc[0, 1]

    ss_residual = np.sum(
        residuals ** 2
    )

    ss_total = np.sum(
        (actual - actual.mean()) ** 2
    )

    r_squared = (
        1.0 - ss_residual / ss_total
        if ss_total > 0
        else np.nan
    )

    rmse = np.sqrt(
        np.mean(residuals ** 2)
    )

    return {
        "correlation": float(correlation),
        "r_squared": float(r_squared),
        "rmse": float(rmse),
        "observations": int(len(overlap)),
    }


# =========================================================
# BUILD EXTENDED QQQ
# =========================================================
def build_extended_qqq(
    ndx: pd.DataFrame,
    actual_qqq: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a synthetic QQQ series before QQQ existed.

    The synthetic QQQ is based on the percentage movement
    of NDX and scaled to equal QQQ on QQQ's first real date.
    """

    history = ndx.copy()

    first_actual_qqq_date = (
        actual_qqq["Date"].min()
    )

    anchor = pd.merge(
        history,
        actual_qqq,
        on="Date",
        how="inner",
    )

    if anchor.empty:
        raise ValueError(
            "NDX and QQQ do not share any dates."
        )

    anchor = anchor.sort_values("Date")

    anchor_row = anchor.iloc[0]

    anchor_date = anchor_row["Date"]
    anchor_ndx_price = anchor_row["NDX Close"]
    anchor_qqq_price = anchor_row[
        "Actual QQQ Close"
    ]

    # Scale NDX so the synthetic QQQ equals actual QQQ
    # on their first shared date.
    qqq_scale_factor = (
        anchor_qqq_price / anchor_ndx_price
    )

    history["Synthetic QQQ Close"] = (
        history["NDX Close"]
        * qqq_scale_factor
    )

    history = pd.merge(
        history,
        actual_qqq,
        on="Date",
        how="left",
    )

    # Use synthetic QQQ before the actual fund began.
    # Use actual QQQ from its first available date onward.
    history["QQQ Close"] = np.where(
        history["Date"] < anchor_date,
        history["Synthetic QQQ Close"],
        history["Actual QQQ Close"],
    )

    history["QQQ Source"] = np.where(
        history["Date"] < anchor_date,
        "Synthetic QQQ from NDX",
        "Actual QQQ",
    )

    history["NDX Classification"] = np.where(
        history["Date"]
        < NASDAQ_100_LAUNCH_DATE,
        "Vendor reconstructed pre-1985 NDX",
        "Official-era NDX",
    )

    # Rows after the final actual QQQ date are not used,
    # because the website comparison requires actual QQQ.
    final_actual_qqq_date = (
        actual_qqq["Date"].max()
    )

    history = history[
        history["Date"]
        <= final_actual_qqq_date
    ].copy()

    history["QQQ Return"] = (
        history["QQQ Close"].pct_change()
    )

    return (
        history.sort_values("Date")
        .reset_index(drop=True)
    )


# =========================================================
# BUILD EXTENDED TQQQ
# =========================================================
def build_extended_tqqq(
    qqq_history: pd.DataFrame,
    actual_tqqq: pd.DataFrame,
    alpha: float,
    beta: float,
) -> pd.DataFrame:
    """
    Create synthetic TQQQ before TQQQ existed.

    Daily synthetic return:

        alpha + beta * QQQ return

    The series is compounded forward from the beginning,
    then scaled to connect smoothly to the first actual
    TQQQ price.
    """

    history = qqq_history.copy()

    history["Synthetic TQQQ Return"] = (
        alpha
        + beta * history["QQQ Return"]
    )

    # The first row has no prior-day return.
    history.loc[
        history.index[0],
        "Synthetic TQQQ Return",
    ] = 0.0

    # Prevent a modeled loss of 100% or more.
    history["Synthetic TQQQ Return"] = (
        history["Synthetic TQQQ Return"]
        .clip(lower=-0.999999)
    )

    # Compound in log space for numerical stability.
    history["Synthetic TQQQ Log Growth"] = (
        np.log1p(
            history["Synthetic TQQQ Return"]
        ).cumsum()
    )

    history = pd.merge(
        history,
        actual_tqqq,
        on="Date",
        how="left",
    )

    actual_overlap = history[
        history["Actual TQQQ Close"].notna()
    ].copy()

    if actual_overlap.empty:
        raise ValueError(
            "No overlap was found with actual TQQQ."
        )

    first_actual_row = (
        actual_overlap.iloc[0]
    )

    first_actual_tqqq_date = (
        first_actual_row["Date"]
    )

    first_actual_tqqq_price = (
        first_actual_row[
            "Actual TQQQ Close"
        ]
    )

    anchor_log_growth = (
        first_actual_row[
            "Synthetic TQQQ Log Growth"
        ]
    )

    # Choose the scale so synthetic TQQQ equals actual
    # TQQQ on the first real TQQQ date.
    log_scale = (
        np.log(first_actual_tqqq_price)
        - anchor_log_growth
    )

    history["Synthetic TQQQ Close"] = np.exp(
        history["Synthetic TQQQ Log Growth"]
        + log_scale
    )

    history["TQQQ Close"] = np.where(
        history["Date"]
        < first_actual_tqqq_date,
        history["Synthetic TQQQ Close"],
        history["Actual TQQQ Close"],
    )

    history["TQQQ Source"] = np.where(
        history["Date"]
        < first_actual_tqqq_date,
        "Synthetic fitted TQQQ",
        "Actual TQQQ",
    )

    # The shared usable ending date is whichever actual
    # ETF dataset ends first.
    final_actual_tqqq_date = (
        actual_tqqq["Date"].max()
    )

    final_usable_date = min(
        history["Date"].max(),
        final_actual_tqqq_date,
    )

    history = history[
        history["Date"]
        <= final_usable_date
    ].copy()

    return (
        history.sort_values("Date")
        .reset_index(drop=True)
    )


# =========================================================
# MAIN BUILD PROCESS
# =========================================================
def main() -> None:
    print("=" * 60)
    print("BUILDING EXTENDED QQQ AND TQQQ HISTORY")
    print("=" * 60)
    print()

    print("Loading price files...")

    ndx = load_price_file(
        NDX_FILE,
        "NDX Close",
    )

    qqq = load_price_file(
        QQQ_FILE,
        "Actual QQQ Close",
    )

    tqqq = load_price_file(
        TQQQ_FILE,
        "Actual TQQQ Close",
    )

    print(
        f"NDX range: "
        f"{ndx['Date'].min().date()} through "
        f"{ndx['Date'].max().date()}"
    )

    print(
        f"QQQ range: "
        f"{qqq['Date'].min().date()} through "
        f"{qqq['Date'].max().date()}"
    )

    print(
        f"TQQQ range: "
        f"{tqqq['Date'].min().date()} through "
        f"{tqqq['Date'].max().date()}"
    )

    print()
    print(
        "Estimating actual QQQ/TQQQ relationship..."
    )

    alpha, beta, overlap = (
        estimate_tqqq_model(
            actual_qqq=qqq,
            actual_tqqq=tqqq,
        )
    )

    statistics = calculate_model_statistics(
        overlap=overlap,
        alpha=alpha,
        beta=beta,
    )

    print(
        f"Estimated beta: {beta:.6f}"
    )

    print(
        f"Estimated alpha: {alpha:.8f}"
    )

    print(
        f"Daily return correlation: "
        f"{statistics['correlation']:.6f}"
    )

    print(
        f"R-squared: "
        f"{statistics['r_squared']:.6f}"
    )

    print(
        f"Model observations: "
        f"{statistics['observations']:,}"
    )

    print()
    print("Building extended QQQ history...")

    qqq_history = build_extended_qqq(
        ndx=ndx,
        actual_qqq=qqq,
    )

    print("Building extended TQQQ history...")

    extended_history = build_extended_tqqq(
        qqq_history=qqq_history,
        actual_tqqq=tqqq,
        alpha=alpha,
        beta=beta,
    )

    output_columns = [
        "Date",
        "NDX Close",
        "QQQ Close",
        "TQQQ Close",
        "QQQ Return",
        "Synthetic TQQQ Return",
        "QQQ Source",
        "TQQQ Source",
        "NDX Classification",
    ]

    extended_history = (
        extended_history[
            output_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "Date",
                "QQQ Close",
                "TQQQ Close",
            ]
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    extended_history.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 60)
    print("EXTENDED HISTORY CREATED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print(
        f"Date range: "
        f"{extended_history['Date'].min().date()} "
        f"through "
        f"{extended_history['Date'].max().date()}"
    )

    print(
        f"Total rows: "
        f"{len(extended_history):,}"
    )

    print()
    print("QQQ source counts:")

    print(
        extended_history[
            "QQQ Source"
        ].value_counts()
    )

    print()
    print("TQQQ source counts:")

    print(
        extended_history[
            "TQQQ Source"
        ].value_counts()
    )

    print()
    print("NDX classification counts:")

    print(
        extended_history[
            "NDX Classification"
        ].value_counts()
    )

    print()
    print("First five rows:")

    print(
        extended_history.head()
    )

    print()
    print("Last five rows:")

    print(
        extended_history.tail()
    )


if __name__ == "__main__":
    main()