import pandas as pd
from prophet import Prophet

def forecast_next_month(df: pd.DataFrame, value_col: str):
    """Predicts next month's value using Prophet."""
    if "date" not in df.columns:
        return None, "No 'date' column found for forecasting."

    df = df[["date", value_col]].dropna()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.rename(columns={"date": "ds", value_col: "y"})

    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    next_month = forecast.tail(30)["yhat"].mean()
    return next_month, forecast
