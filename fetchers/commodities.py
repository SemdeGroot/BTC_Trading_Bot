import os
import requests
import pandas as pd
from datetime import datetime, timezone
from utils.config import DATA_PATH, ALPHA_VANTAGE_API_KEY

def get_commodity_data(path=DATA_PATH, apikey=ALPHA_VANTAGE_API_KEY):
    """
    Fetches historical commodity data (SUGAR, COFFEE) from Alpha Vantage starting from
    the last date in merged_data.csv + 1 day until today. Ensures daily rows with NaNs are added
    for missing dates to allow later interpolation.

    Returns:
    pd.DataFrame: DataFrame with ['date', 'SUGAR', 'COFFEE'] columns.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")

    # Determine date range for update
    df_merged = pd.read_csv(path, parse_dates=['date'])
    last_date = pd.to_datetime(df_merged['date'].max())
    start_date = last_date + pd.Timedelta(days=1)
    today = pd.to_datetime(datetime.now(timezone.utc).date())
    full_range = pd.date_range(start=start_date, end=today, freq='D')

    base_url = "https://www.alphavantage.co/query"
    commodities = {
        "SUGAR": "monthly",
        "COFFEE": "monthly"
    }
    all_dfs = []

    for commodity, interval in commodities.items():
        url = f"{base_url}?function={commodity}&interval={interval}&apikey={apikey}&datatype=json"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if 'Information' in data or 'Note' in data:
                raise requests.exceptions.HTTPError(
                    f"API Limit or Error Message from Alpha Vantage: {data.get('Information') or data.get('Note')}",
                    response=response
                )

            if 'data' not in data:
                print(f"{commodity}: 'data' key not found in API response.")
                continue

            df = pd.DataFrame(data['data'])
            df = df[df['value'] != '.']
            df['date'] = pd.to_datetime(df['date'])
            df[commodity] = pd.to_numeric(df['value'], errors='coerce')
            df = df[['date', commodity]]
            df = df.set_index('date').sort_index()

            # Filter to requested range and reindex to daily frequency
            df = df.loc[df.index <= today]
            df = df.reindex(full_range)
            df.index.name = 'date'

            all_dfs.append(df)

            print(f"Fetched {df.dropna().shape[0]} records for {commodity} from {start_date.date()} to {today.date()}.")

        except requests.exceptions.HTTPError as http_err:
            print(f"{commodity}: HTTP error {http_err.response.status_code if http_err.response else ''}: {http_err}")
        except Exception as e:
            print(f"{commodity}: Error: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, axis=1)
        combined_df.reset_index(inplace=True)
        return combined_df
    else:
        return pd.DataFrame(columns=["date", "SUGAR", "COFFEE"])