import streamlit as st
import pandas as pd
from core.loader import load_csv
from core.analyzer import get_basic_kpis
from core.visualizer import visualize_data
from core.insights_groq import generate_ai_insights
from core.forecaster import forecast_next_month

# 🔑 Your Groq API key
groq_api_key = "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"

# Streamlit app setup
st.set_page_config(page_title="Business Data Analyzer", layout="wide")
st.title("💼 Business Data Analyzer")

# File uploader
uploaded = st.file_uploader("📤 Upload your business CSV file", type=["csv"])

if uploaded:
    df = load_csv(uploaded)
    st.write("### Preview", df.head())

    st.divider()
    st.subheader("📈 KPIs")
    kpis = get_basic_kpis(df)
    for k, v in kpis.items():
        st.metric(label=k, value=round(v, 2) if isinstance(v, (int, float)) else v)

    st.divider()
    visualize_data(df)

    # 🧠 Generate AI Insights
    if st.button("🧠 Generate AI Insights"):
        with st.spinner("Generating insights via Groq..."):
            try:
                insights = generate_ai_insights(df, groq_api_key)  # ✅ pass API key here
                st.markdown(insights)
            except Exception as e:
                st.error(str(e))

    # 🔮 Forecasting
    if "revenue" in df.columns or "sales" in df.columns:
        col = "revenue" if "revenue" in df.columns else "sales"
        if st.button("🔮 Forecast Next Month"):
            with st.spinner("Forecasting with Prophet..."):
                next_month, forecast = forecast_next_month(df, col)
                st.success(f"Predicted {col} next month: {round(next_month, 2)}")
