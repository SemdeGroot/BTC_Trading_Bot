import os
import pandas as pd
from utils.config import DATA_PATH

def merge_and_append_with_existing(df_list, file_path=DATA_PATH):
    """
    Loads the existing df_merged.csv and appends new rows from a list of DataFrames based on 'date'.
    The new DataFrames are merged on 'date' using outer joins, and then appended to the existing data.

    Parameters:
        df_list (list): List of pd.DataFrame objects, each containing a 'date' column.
        file_path (str): Path to the existing df_merged.csv file.

    Returns:
        pd.DataFrame: The combined DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    # Load existing data
    df_existing = pd.read_csv(file_path, parse_dates=["date"])

    # Check and prepare incoming data
    for i, df in enumerate(df_list):
        if 'date' not in df.columns:
            raise ValueError(f"DataFrame at position {i} has no 'date' column.")
        df['date'] = pd.to_datetime(df['date'])

    # Merge all new dataframes on 'date'
    df_new_combined = df_list[0]
    for df in df_list[1:]:
        df_new_combined = pd.merge(df_new_combined, df, on='date', how='outer')

    # Concatenate with existing data
    df_full = pd.concat([df_existing, df_new_combined], ignore_index=True)
    df_full = df_full.sort_values("date")

    return df_full