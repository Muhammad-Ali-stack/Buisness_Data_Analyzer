import pandas as pd

def load_csv(file_path: str) -> pd.DataFrame:
    """Loads a CSV and cleans column names."""
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")
