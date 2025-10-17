# core/insights_groq.py
import requests
import json
import pandas as pd

def generate_ai_insights(df: pd.DataFrame, groq_api_key: str) -> str:
    """
    Generate business insights from a DataFrame using Groq's Llama 3 model.
    """
    try:
        # Convert summary stats to text
        summary = df.describe(include='all').to_string()
        sample_rows = df.head(3).to_string()

        prompt = f"""
        You are a senior business data analyst. Analyze the following dataset summary and sample data.
        Give 5–7 concise insights in plain English about:
        - Key trends
        - Outliers
        - Correlations
        - Business opportunities

        === Dataset Summary ===
        {summary}

        === Sample Data ===
        {sample_rows}
        """

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
    # Use a supported (active) Groq model
    "model": "llama-3.3-70b-versatile",  
    "messages": [
        {"role": "system", "content": "You are an expert business data analyst."},
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
