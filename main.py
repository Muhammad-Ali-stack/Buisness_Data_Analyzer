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
st.title("📊 Business Data Analyzer")

uploaded = st.file_uploader("📁 Upload your business CSV file", type=["csv"])

def get_available_groq_model(api_key: str) -> str:
    """
    Fetches the list of available models from Groq API and returns
    the first usable chat model (e.g., Llama or Mixtral).
    """
    try:
        url = "https://api.groq.com/openai/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        for m in data.get("data", []):
            if "llama" in m["id"] or "mixtral" in m["id"]:
                return m["id"]

        return "llama-3.3-70b-versatile"
    except Exception as e:
        st.warning(f"⚠️ Could not auto-detect Groq model: {e}")
        return "llama-3.3-70b-versatile"

# 🧠 Auto-select model
groq_model = get_available_groq_model(groq_api_key)

if uploaded:
    # Load CSV
    df = load_csv(uploaded)
    st.write("### 👀 Preview", df.head())

    st.divider()

    # 📈 KPIs Section
    st.subheader("📌 Key Performance Indicators (KPIs)")
    kpis = get_basic_kpis(df)
    for k, v in kpis.items():
        st.metric(label=k, value=round(v, 2) if isinstance(v, (int, float)) else v)

    st.divider()

    # 📊 Visualization
    visualize_data(df)

    # 🧠 AI Insights
    if st.button("🧩 Generate AI Insights"):
        with st.spinner("Analyzing data using Groq..."):
            insights = generate_ai_insights(df, groq_api_key)
            st.markdown(insights)

    # 🔮 Forecasting (Optional)
    if any(col in df.columns for col in ["revenue", "sales"]):
        col = "revenue" if "revenue" in df.columns else "sales"
        if st.button("🔮 Forecast Next Month"):
            with st.spinner("Forecasting with Prophet..."):
                next_month, forecast = forecast_next_month(df, col)
                st.success(f"📅 Predicted {col} next month: **{round(next_month, 2)}**")

        # 💬 Ask AI a Question
    st.divider()
    st.subheader("💬 Ask a Question About Your Data")

    user_query = st.text_input(
        "Type your question here (e.g., Which product had the highest revenue?)"
    )

    if st.button("🤖 Ask AI"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        elif not groq_api_key:
            st.error("GROQ API key missing. Please add it in `.streamlit/secrets.toml`.")
        else:
            with st.spinner("Analyzing your data with Groq..."):
                try:
                    client = Groq(api_key=groq_api_key)

                    # ✅ Limit data size (to avoid context overflow)
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    sample_cols = numeric_cols[:5] if numeric_cols else df.columns[:5]
                    sample_data = df[sample_cols].sample(min(5, len(df))).to_csv(index=False)

                    prompt = f"""
                    You are a business data analysis assistant.
                    Here is a small sample of the uploaded CSV data (truncated for context safety):

                    {sample_data}

                    The user asks: "{user_query}"

                    Please answer concisely based only on this sample data.
                    """

                    # ✅ Hard cap prompt to 2500 characters
                    prompt = prompt[:2500]

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

# Footer
st.markdown(
    """
---
<div style='text-align: center; padding-top: 20px; font-size: 15px; color: gray;'>
    Made with ❤️ by <b>Muhammad Ali</b>
</div>
""",
    unsafe_allow_html=True,
)
