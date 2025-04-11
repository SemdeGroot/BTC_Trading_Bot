import pandas as pd
from utils.config import DATA_PATH

def interpolate_missing_values(df):
    """
    Interpolates missing numeric values for each column using linear interpolation and overwrites existing df.

    - Sets 'date' as the index.
    - Ensures all columns are numeric.
    - Sorts by date.
    - Fills missing values in both directions (forward and backward).

    Parameters:
        df (pd.DataFrame): A DataFrame with a 'date' column and numeric features.

    Returns:
        pd.DataFrame: A cleaned DataFrame with interpolated values and restored index.
    """
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.interpolate(method='linear', limit_direction='both')

    df = df.reset_index()
    df.to_csv(DATA_PATH, index=False)

    return df