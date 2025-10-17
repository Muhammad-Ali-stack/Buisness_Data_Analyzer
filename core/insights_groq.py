import requests
import pandas as pd

def generate_ai_insights(df: pd.DataFrame, groq_api_key: str, user_question: str = None) -> str:
    """
    Generate AI insights or answer user questions using Groq API.
    Automatically trims data to fit Groq token limits.
    """
    if not groq_api_key:
        return "❌ No Groq API key provided. Please set it in Streamlit secrets or environment."

    # Reduce large data: sample max 10 rows and 10 columns
    df_trimmed = df.sample(min(10, len(df)), random_state=42).iloc[:, :10]
    data_sample = df_trimmed.to_csv(index=False)

    # Base prompt
    if user_question:
        prompt = f"""
        You are a business data analyst. Here is a sample of my data:
        {data_sample}

        Based on this data, please answer this user question:
        {user_question}
        """
    else:
        prompt = f"""
        You are a business data analyst. Analyze this business dataset:
        {data_sample}

        Summarize insights, trends, performance, and key recommendations.
        """

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a professional business data analyst."},
            {"role": "user", "content": prompt[:4000]}  # Prevent long inputs
        ],
        "max_tokens": 500,
        "temperature": 0.7,
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        return f"❌ API Error: {e}"
    except KeyError:
        return f"❌ Unexpected API response: {response.text}"
