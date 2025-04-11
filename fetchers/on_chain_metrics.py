import os
import requests
import pandas as pd
from datetime import datetime, timezone
from utils.config import DATA_PATH

def get_bitcoin_metrics(path=DATA_PATH):
    """
    Fetches selected Bitcoin on-chain metrics from the BGeometrics API using startday and endday parameters.
    Falls back to full fetch if date filtering fails. Manually filters from start_date if needed.

    Returns:
        pd.DataFrame with ['date'] and selected metrics as columns.
    """
    metrics = ['puell-multiple', 'sth-mvrv', 'lth-mvrv', 'lth-sopr', 'cdd', 'out-flows']

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")

    df_merged = pd.read_csv(path, parse_dates=['date'])
    last_date = pd.to_datetime(df_merged['date'].max())
    start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    base_url = "https://bitcoin-data.com/api/v1/"
    metric_dataframes = []

    for metric in metrics:
        url = f"{base_url}{metric}"
        params = {"startday": start_date, "endday": end_date}

        try:
            response = requests.get(url, params=params)
            if response.status_code == 404:
                print(f"Date filtering not supported for {metric}, retrying full fetch.")
                response = requests.get(url)  # fallback: no params
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list) and data:
                records = []
                for entry in data:
                    date = entry.get("d") or entry.get("theDay")
                    if date is None:
                        continue

                    for key in entry:
                        if key not in ["d", "theDay", "unixTs"]:
                            value = pd.to_numeric(entry[key], errors='coerce')
                            break
                    else:
                        value = None

                    if value is not None:
                        records.append((date, value))

                temp_df = pd.DataFrame(records, columns=["date", metric])
                temp_df["date"] = pd.to_datetime(temp_df["date"])
                temp_df = temp_df[temp_df["date"] >= pd.to_datetime(start_date)]
                temp_df.set_index("date", inplace=True)
                metric_dataframes.append(temp_df)

                print(f"Fetched {len(temp_df)} rows for {metric} from {start_date} to {end_date}")
            else:
                print(f"No data returned for {metric}.")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {metric}: {e}")

    if metric_dataframes:
        df = pd.concat(metric_dataframes, axis=1).reset_index()
    else:
        df = pd.DataFrame()

    return df