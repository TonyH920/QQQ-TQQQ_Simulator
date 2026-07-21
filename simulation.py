import pandas as pd


def simulate_investment(
    prices,
    start_index,
    months,
    initial=5000,
    monthly=100
):
    """
    Simulates investing into one ETF.

    prices = dataframe containing monthly prices
    start_index = first month to invest
    months = investment length
    """

    return {
        "ending_value": 0,
        "shares": 0,
        "contributions": 0,
        "gain": 0,
        "cagr": 0
    }
def load_data():

    qqq = pd.read_csv("qqq.csv")
    tqqq = pd.read_csv("tqqq.csv")

    qqq["Date"] = pd.to_datetime(qqq["Date"])
    tqqq["Date"] = pd.to_datetime(tqqq["Date"])

    return qqq, tqqq
def first_trading_day_each_month(df):
    """
    Returns the first trading day for each month.
    """

    monthly = (
        df.sort_values("Date")
          .groupby(df["Date"].dt.to_period("M"))
          .first()
          .reset_index(drop=True)
    )

    return monthly
if __name__ == "__main__":

    qqq, tqqq = load_data()

    qqq_monthly = first_trading_day_each_month(qqq)

    print(qqq_monthly.head())