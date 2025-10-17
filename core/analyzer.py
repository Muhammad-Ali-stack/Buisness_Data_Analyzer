import pandas as pd

def get_basic_kpis(df: pd.DataFrame) -> dict:
    """Extracts key business metrics based on available columns."""
    kpis = {}
    numeric_cols = df.select_dtypes(include='number').columns

    if "revenue" in df.columns or "sales" in df.columns:
        col = "revenue" if "revenue" in df.columns else "sales"
        kpis["total_revenue"] = df[col].sum()
        kpis["average_revenue"] = df[col].mean()
        kpis["max_revenue"] = df[col].max()

    if "profit" in df.columns:
        kpis["total_profit"] = df["profit"].sum()
        kpis["profit_margin_avg"] = (df["profit"] / df[col]).mean() * 100

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        monthly = df.groupby(df["date"].dt.to_period("M"))[col].sum()
        if len(monthly) > 1:
            growth = ((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2]) * 100
            kpis["last_month_growth_%"] = round(growth, 2)

    for n in numeric_cols:
        kpis[f"{n}_mean"] = round(df[n].mean(), 2)

    return kpis
