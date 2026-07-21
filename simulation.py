import pandas as pd
import numpy as np

def simulate_monthly_investment(
    prices: pd.DataFrame,
    initial_investment: float = 5000,
    monthly_investment: float = 100,
) -> tuple[pd.DataFrame, dict]:
    """
    Simulate investing an initial amount and then adding money on the
    first available trading day of each following calendar month.

    Required columns:
        Date
        Close
    """

    required_columns = {"Date", "Close"}

    if not required_columns.issubset(prices.columns):
        raise ValueError(
            "Price data must contain Date and Close columns."
        )

    data = prices[["Date", "Close"]].copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
    )

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce",
    )

    data = (
        data.dropna(subset=["Date", "Close"])
        .drop_duplicates(subset="Date", keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if data.empty:
        raise ValueError("No usable price data was found.")

    if (data["Close"] <= 0).any():
        raise ValueError("All closing prices must be greater than zero.")

    first_price = data.loc[0, "Close"]

    shares = initial_investment / first_price
    total_contributed = initial_investment

    previous_month = data.loc[0, "Date"].to_period("M")

    portfolio_values = []
    total_contributions = []
    shares_owned = []
    contribution_amounts = []

    for row_number, row in data.iterrows():
        current_date = row["Date"]
        current_price = row["Close"]
        current_month = current_date.to_period("M")

        contribution = 0.0

        # Do not add $100 on the initial investment date.
        # Add it on the first trading day of every following month.
        if row_number > 0 and current_month != previous_month:
            contribution = monthly_investment
            shares += contribution / current_price
            total_contributed += contribution
            previous_month = current_month

        portfolio_value = shares * current_price

        contribution_amounts.append(contribution)
        shares_owned.append(shares)
        total_contributions.append(total_contributed)
        portfolio_values.append(portfolio_value)

    data["Contribution"] = contribution_amounts
    data["Shares Owned"] = shares_owned
    data["Total Contributed"] = total_contributions
    data["Portfolio Value"] = portfolio_values
    data["Profit"] = (
        data["Portfolio Value"] - data["Total Contributed"]
    )

    ending_value = data["Portfolio Value"].iloc[-1]
    ending_contributions = data["Total Contributed"].iloc[-1]
    profit = ending_value - ending_contributions

    return_on_contributions = (
        profit / ending_contributions
        if ending_contributions > 0
        else 0
    )

    summary = {
        "start_date": data["Date"].iloc[0],
        "end_date": data["Date"].iloc[-1],
        "starting_price": data["Close"].iloc[0],
        "ending_price": data["Close"].iloc[-1],
        "ending_value": ending_value,
        "total_contributed": ending_contributions,
        "profit": profit,
        "return_on_contributions": return_on_contributions,
        "shares_owned": data["Shares Owned"].iloc[-1],
        "monthly_contribution_count": int(
            (data["Contribution"] > 0).sum()
        ),
    }

    return data, summary
def simulate_all_rolling_periods(
    prices: pd.DataFrame,
    years: int,
    initial_investment: float = 5000,
    monthly_investment: float = 100,
) -> pd.DataFrame:
    """
    Test every possible rolling period of the requested length.

    Each period:
    - Begins on every possible shared trading date.
    - Invests the initial amount on that starting date.
    - Adds the monthly amount on the first available trading day
      of each following calendar month.
    - Ends on the first available trading date on or after the
      requested anniversary.

    Required columns:
        Date
        QQQ Close
        TQQQ Close
    """

    required_columns = {
        "Date",
        "QQQ Close",
        "TQQQ Close",
    }

    if not required_columns.issubset(prices.columns):
        missing_columns = required_columns - set(prices.columns)

        raise ValueError(
            f"Rolling-period data is missing: "
            f"{sorted(missing_columns)}"
        )

    data = prices[
        [
            "Date",
            "QQQ Close",
            "TQQQ Close",
        ]
    ].copy()

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

    if data.empty:
        raise ValueError(
            "No usable overlapping price data was found."
        )

    if (
        (data["QQQ Close"] <= 0).any()
        or (data["TQQQ Close"] <= 0).any()
    ):
        raise ValueError(
            "All QQQ and TQQQ prices must be greater than zero."
        )

    dates = data["Date"].to_numpy(
        dtype="datetime64[ns]"
    )

    qqq_prices = data["QQQ Close"].to_numpy(
        dtype=float
    )

    tqqq_prices = data["TQQQ Close"].to_numpy(
        dtype=float
    )

    # A numeric code for each calendar month.
    month_codes = (
        data["Date"].dt.year.to_numpy() * 12
        + data["Date"].dt.month.to_numpy()
    )

    final_available_date = data["Date"].iloc[-1]

    results = []

    for start_index in range(len(data)):
        start_date = data["Date"].iloc[start_index]

        target_end_date = (
            start_date
            + pd.DateOffset(years=years)
        )

        # Stop once a complete period no longer fits in the dataset.
        if target_end_date > final_available_date:
            break

        # Use the first trading date on or after the anniversary.
        end_index = int(
            dates.searchsorted(
                np.datetime64(target_end_date),
                side="left",
            )
        )

        if end_index >= len(data):
            break

        # Find the first trading day of each new calendar month
        # after the initial investment date.
        period_month_codes = month_codes[
            start_index:end_index + 1
        ]

        month_change_positions = (
            np.flatnonzero(
                period_month_codes[1:]
                != period_month_codes[:-1]
            )
            + start_index
            + 1
        )

        number_of_monthly_investments = len(
            month_change_positions
        )

        total_contributed = (
            initial_investment
            + monthly_investment
            * number_of_monthly_investments
        )

        qqq_shares = (
            initial_investment
            / qqq_prices[start_index]
        )

        tqqq_shares = (
            initial_investment
            / tqqq_prices[start_index]
        )

        if number_of_monthly_investments > 0:
            qqq_shares += np.sum(
                monthly_investment
                / qqq_prices[month_change_positions]
            )

            tqqq_shares += np.sum(
                monthly_investment
                / tqqq_prices[month_change_positions]
            )

        qqq_ending_value = (
            qqq_shares * qqq_prices[end_index]
        )

        tqqq_ending_value = (
            tqqq_shares * tqqq_prices[end_index]
        )

        qqq_profit = (
            qqq_ending_value - total_contributed
        )

        tqqq_profit = (
            tqqq_ending_value - total_contributed
        )

        qqq_return = (
            qqq_profit / total_contributed
        )

        tqqq_return = (
            tqqq_profit / total_contributed
        )

        if tqqq_ending_value > qqq_ending_value:
            winner = "TQQQ"
        elif qqq_ending_value > tqqq_ending_value:
            winner = "QQQ"
        else:
            winner = "Tie"

        results.append(
            {
                "Start Date": start_date,
                "End Date": data["Date"].iloc[
                    end_index
                ],
                "Years": years,
                "Monthly Investments": (
                    number_of_monthly_investments
                ),
                "Total Contributed": (
                    total_contributed
                ),
                "QQQ Ending Value": (
                    qqq_ending_value
                ),
                "TQQQ Ending Value": (
                    tqqq_ending_value
                ),
                "QQQ Profit": qqq_profit,
                "TQQQ Profit": tqqq_profit,
                "QQQ Return": qqq_return,
                "TQQQ Return": tqqq_return,
                "TQQQ Minus QQQ": (
                    tqqq_ending_value
                    - qqq_ending_value
                ),
                "Winner": winner,
            }
        )

    return pd.DataFrame(results)