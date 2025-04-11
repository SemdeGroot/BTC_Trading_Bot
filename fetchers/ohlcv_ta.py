import os
import pandas as pd
import requests
from datetime import datetime, timezone
from ta import add_all_ta_features
from utils.config import COINDESK_API_KEY, DATA_PATH

def get_coindesk_ohlcv_ta(market='cadli', instrument='BTC-USD', api_key=COINDESK_API_KEY, path=DATA_PATH):
    """
    Fetches new OHLCV data from CoinDesk, appends it to historical data,
    computes selected technical analysis (TA) features on the full dataset,
    and returns only the new rows with relevant TA features.

    Parameters:
        market (str): CoinDesk market code (default 'cadli')
        instrument (str): Trading pair (default 'BTC-USD')
        api_key (str): CoinDesk API key
        path (str): Path to the historical merged dataset

    Returns:
        pd.DataFrame: DataFrame containing only new rows with selected TA features
    """

    # Only keep these TA features (exclude on-chain and commodity indicators)
    ta_features_to_keep = [
        'others_dr',
        'volume_sma_em',
        'trend_dpo',
        'volatility_ui',
        'volume_em',
        'trend_adx',
        'trend_mass_index',
        'trend_stc',
        'volatility_atr'
    ]

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")

    # Load existing data
    df_merged = pd.read_csv(path, parse_dates=['date'])
    last_date = pd.to_datetime(df_merged['date'].max())
    start_date = last_date + pd.Timedelta(days=1)
    today = pd.to_datetime(datetime.now(timezone.utc).date())
    num_days = (today - start_date).days

    if num_days <= 0:
        print("Data is already up to date.")
        return pd.DataFrame()

    # Fetch new data from CoinDesk API
    start_ts = int(start_date.timestamp())
    now_ts = int(datetime.now(timezone.utc).timestamp())
    all_data = []
    to_ts = now_ts

    while to_ts > start_ts:
        limit = num_days
        params = {
            "market": market,
            "instrument": instrument,
            "limit": limit,
            "to_ts": to_ts,
            "aggregate": 1,
            "fill": "true",
            "apply_mapping": "true",
            "response_format": "JSON"
        }
        if api_key:
            params["api_key"] = api_key

        response = requests.get("https://data-api.coindesk.com/index/cc/v1/historical/days", params=params)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code} {response.text}")

        data = response.json().get("Data", [])
        if not data:
            print("No more data returned.")
            break

        all_data.extend(data)
        timestamps = [entry["TIMESTAMP"] for entry in data]
        earliest_ts = min(timestamps)
        latest_ts = max(timestamps)
        to_ts = earliest_ts - 86400  # move back one day

        print(f"Fetched {len(data)} rows from {datetime.fromtimestamp(earliest_ts, tz=timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(latest_ts, tz=timezone.utc).strftime('%Y-%m-%d')}")

    if not all_data:
        return pd.DataFrame()

    # Process new data
    df_new = pd.DataFrame(all_data)
    df_new['date'] = pd.to_datetime(df_new['TIMESTAMP'], unit='s')
    df_new = df_new.sort_values('date')
    df_new = df_new[['date', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]

    # Combine old and new
    df_all = pd.concat([df_merged[['date', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']], df_new], ignore_index=True)
    df_all = df_all.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)

    # Add all TA features
    df_all_ta = add_all_ta_features(
        df_all.copy(),
        open='OPEN', high='HIGH', low='LOW', close='CLOSE', volume='VOLUME'
    )

    # Keep only date, OHLCV, and selected TA features
    columns_to_keep = ['date', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'] + ta_features_to_keep
    df_all_ta = df_all_ta[[col for col in columns_to_keep if col in df_all_ta.columns]]

    # Return only newly added rows
    df_result = df_all_ta[df_all_ta['date'] >= start_date].copy()

    return df_result