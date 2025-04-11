import os
import requests
import pandas as pd
from datetime import datetime, timezone
from utils.config import DATA_PATH

def get_fear_greed_data(path=DATA_PATH):
    """
    Fetches historical Fear & Greed Index data from alternative.me API,
    starting from the last date in the existing merged_data.csv + 1 day.
    Returns a DataFrame with columns: ['date', 'fear_greed']
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")

    # Load existing merged data and determine starting date
    df_merged = pd.read_csv(path, parse_dates=['date'])
    last_date = pd.to_datetime(df_merged['date'].max())
    start_date = last_date + pd.Timedelta(days=1)
    today = pd.to_datetime(datetime.now(timezone.utc).date())
    num_days = (today - start_date).days

    if num_days <= 0:
        print("Fear & Greed data is already up to date.")
        return pd.DataFrame()

    # Fetch data from API
    url = f"https://api.alternative.me/fng/?limit={num_days}&format=json"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} {response.text}")

    data = response.json().get("data", [])
    if not data:
        print("No Fear & Greed data returned.")
        return pd.DataFrame()

    # Format DataFrame
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
    df = df.sort_values('date')
    df = df[df['date'] >= start_date]

    if not df.empty:
        start_str = df['date'].min().strftime('%Y-%m-%d')
        end_str = df['date'].max().strftime('%Y-%m-%d')
        print(f"Fetched {len(df)} Fear & Greed rows from {start_str} to {end_str}")
    else:
        print("No new Fear & Greed data after filtering by start_date.")

    df = df.rename(columns={'value': 'fear_greed'})
    df = df[['date', 'fear_greed']].reset_index(drop=True)

    return df