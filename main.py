import streamlit as st
import pandas as pd
import os
import requests
from groq import Groq

from core.loader import load_csv
from core.analyzer import get_basic_kpis
from core.visualizer import visualize_data
from core.forecaster import forecast_next_month
from core.insights_groq import generate_ai_insights

# 🔑 Load API key safely
groq_api_key = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", None))

# Streamlit config
st.set_page_config(page_title="Business Data Analyzer", layout="wide")
st.title("Business Data Analyzer")

uploaded = st.file_uploader("Upload your business CSV file", type=["csv"])

def get_available_groq_model(api_key: str) -> str:
    """
    Fetches the list of available models from Groq API and returns
    the first usable chat model.
    """
    try:
        url = "https://api.groq.com/openai/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Find any Llama or Mixtral chat model
        for m in data.get("data", []):
            if "llama" in m["id"] or "mixtral" in m["id"]:
                return m["id"]

        # Fallback
        return "llama-3.3-70b-versatile"
    except Exception as e:
        st.warning(f"⚠️ Could not auto-detect model: {e}")
        return "llama-3.3-70b-versatile"

# 🧠 Automatically choose a valid Groq model
groq_model = get_available_groq_model(groq_api_key)

if uploaded:
    df = load_csv(uploaded)
    st.write("### Preview", df.head())

    st.divider()
    st.subheader("KPIs")
    kpis = get_basic_kpis(df)
    for k, v in kpis.items():
        st.metric(label=k, value=round(v, 2) if isinstance(v, (int, float)) else v)

    st.divider()
    visualize_data(df)

    # 🧠 AI Insights (Groq)
    if st.button("Generate AI Insights"):
        with st.spinner("Analyzing data using Groq..."):
            insights = generate_ai_insights(df, groq_api_key)
            st.markdown(insights)

    # 🔮 Forecasting
    if "revenue" in df.columns or "sales" in df.columns:
        col = "revenue" if "revenue" in df.columns else "sales"
        if st.button("🔮 Forecast Next Month"):
            with st.spinner("Forecasting with Prophet..."):
                next_month, forecast = forecast_next_month(df, col)
                st.success(f"Predicted {col} next month: {round(next_month, 2)}")

    # 💬 CSV Question Answering
    st.divider()
    st.subheader("Ask a Question About Your Data")

    user_query = st.text_input(
        "Type your question here (e.g., Which product had the highest revenue?)"
    )

    if st.button("Ask AI"):
        if user_query.strip() == "":
            st.warning("Please enter a question.")
        elif not groq_api_key:
            st.error("GROQ API key missing. Please add it in .streamlit/secrets.toml")
        else:
            with st.spinner("Thinking..."):
                try:
                    client = Groq(api_key=groq_api_key)
                    sample_data = df.head(20).to_csv(index=False)

                    prompt = f"""
                    You are a business data analysis assistant.
                    Here is a sample of the uploaded CSV data:
                    {sample_data}

                    The user asks: {user_query}
                    Please answer clearly and concisely based only on the data provided.
                    """

                    response = client.chat.completions.create(
                        model=groq_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                        max_tokens=400,
                    )

                    answer = response.choices[0].message.content
                    st.success(answer)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Footer Section
st.markdown("""
---
<div style='text-align: center; padding-top: 20px; font-size: 15px; color: gray;'>
    Made by <b>Muhammad Ali</b>
</div>
""", unsafe_allow_html=True)
