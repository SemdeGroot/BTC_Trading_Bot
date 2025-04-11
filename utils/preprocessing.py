import pandas as pd
import numpy as np
import joblib
import os
from utils.config import SCALER_PATH, X_TEST_PATH, X_TRAIN_PATH, DATA_PATH

def update_X_test():
    """
    Loads df_merged from DATA_PATH, extracts new rows beyond X_train and existing X_test,
    scales them with the stored MinMaxScaler, and appends to X_test.csv.

    Returns:
        pd.DataFrame: The updated X_test dataframe.
    """
    # Define the features used during training
    selected_features = [
        'others_dr', 'puell-multiple', 'sth-mvrv', 'lth-mvrv', 'volume_sma_em', 'trend_dpo',
        'volatility_ui', 'lth-sopr', 'volume_em', 'cdd', 'trend_adx', 'out-flows',
        'trend_mass_index', 'SUGAR', 'COFFEE', 'trend_stc', 'fear_greed'
    ]

    # Load merged data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"df_merged not found at: {DATA_PATH}")
    df_merged = pd.read_csv(DATA_PATH, parse_dates=['date'])

    # Load X_train to determine starting index
    if not os.path.exists(X_TRAIN_PATH):
        raise FileNotFoundError(f"X_train not found at: {X_TRAIN_PATH}")
    X_train = pd.read_csv(X_TRAIN_PATH)
    num_train = len(X_train)

    # Load existing X_test
    if os.path.exists(X_TEST_PATH):
        X_test_existing = pd.read_csv(X_TEST_PATH)
    else:
        X_test_existing = pd.DataFrame(columns=selected_features)

    num_existing_test = len(X_test_existing)

    # Determine new rows
    start_idx = num_train + num_existing_test
    df_new_rows = df_merged.iloc[start_idx:].copy()

    if df_new_rows.empty:
        print("No new rows to add to X_test.")
        return X_test_existing

    # Select and validate features
    try:
        X_new_unscaled = df_new_rows[selected_features].copy()
    except KeyError as e:
        missing = set(selected_features) - set(df_new_rows.columns)
        raise KeyError(f"Missing features in df_merged: {missing}") from e

    # Load scaler
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"MinMaxScaler not found at: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

    # Scale and assemble
    X_new_scaled = scaler.transform(X_new_unscaled)
    X_new_scaled_df = pd.DataFrame(X_new_scaled, columns=selected_features)

    # Append and save
    X_test_updated = pd.concat([X_test_existing, X_new_scaled_df], ignore_index=True)
    X_test_updated.to_csv(X_TEST_PATH, index=False)

    print(f"Added {len(X_new_scaled_df)} new row(s) to X_test. Total rows now: {len(X_test_updated)}")
    return X_test_updated