import requests
import json
import pandas as pd
import os

def generate_ai_insights(df: pd.DataFrame, groq_api_key: str = None) -> str:
    """
    Generate business insights from a DataFrame using Groq's LLaMA model.
    """
    try:
        # Get API key from environment if not passed
        groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return "❌ Missing Groq API key. Please set GROQ_API_KEY in Streamlit secrets or environment."

        # Summarize data
        summary = df.describe(include='all').to_string()
        sample = df.head(3).to_string()

        prompt = f"""
        You are a senior business analyst. Analyze the following dataset summary and sample rows.
        Write 5–7 clear insights about trends, anomalies, and opportunities.

        === Dataset Summary ===
        {summary}

        === Sample Data ===
        {sample}
        """

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "llama-3.3-70b-versatile",  # ✅ currently supported model
            "messages": [
                {"role": "system", "content": "You are a professional business data analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 700
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            return f"❌ API Error {response.status_code}: {response.text}"

        data = response.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"❌ Error generating AI insights: {e}"
