from fetchers.ohlcv_ta import get_coindesk_ohlcv_ta
from fetchers.on_chain_metrics import get_bitcoin_metrics
from fetchers.fear_and_greed import get_fear_greed_data
from fetchers.commodities import get_commodity_data

from utils.merge import merge_and_append_with_existing
from utils.intrapolate_missing_values import interpolate_missing_values
from utils.preprocessing import update_X_test

def run_pipeline():
    # Fetch all sources
    df_ohlcv_ta = get_coindesk_ohlcv_ta()
    df_onchain = get_bitcoin_metrics()
    df_fng = get_fear_greed_data()
    df_commodities = get_commodity_data()

    # Merge new data and append with existing data
    df_merged = merge_and_append_with_existing([
        df_ohlcv_ta,
        df_onchain,
        df_fng,
        df_commodities
    ])

    # Interpolate missing values
    interpolate_missing_values(df_merged)

    # Update X_test.csv with newly scaled rows
    update_X_test()


if __name__ == "__main__":
    run_pipeline()