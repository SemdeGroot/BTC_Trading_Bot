import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.config import DATA_PATH, PLOT_PATH

def plot_results(df_agent):
    """
    Plots and saves capital growth comparison between the Q-Learning Agent and a Buy & Hold strategy.
    Uses close prices from df_merged.csv and saves the plot as a PNG at the location specified in PLOT_PATH.

    Parameters:
        df_agent (pd.DataFrame): DataFrame with at least a 'Capital' column and matching number of days with df_merged.
    """
    # Load price data
    df_merged = pd.read_csv(DATA_PATH, parse_dates=["date"])
    close_prices = df_merged["CLOSE"].values[-len(df_agent):]
    x_dates = df_merged["date"].values[-len(df_agent):]
    initial_capital = 1000

    # Compute Buy & Hold benchmark
    buy_and_hold = [initial_capital * (price / close_prices[0]) for price in close_prices]
    agent_capital_history = df_agent["Capital"].tolist()

    # Plotting
    plt.figure(figsize=(14, 6))

    # Subplot 1: BTC close price
    plt.subplot(1, 2, 1)
    plt.plot(x_dates, close_prices, label="BTC Close Price", color="blue")
    plt.title("Bitcoin Close Price (per day)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.grid(True)
    plt.legend()
    ax1 = plt.gca()
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax1.xaxis.get_major_locator()))
    plt.xticks(rotation=45)

    # Subplot 2: Capital comparison
    plt.subplot(1, 2, 2)
    plt.plot(x_dates, agent_capital_history, label="Q-Learning Agent", color="green")
    plt.plot(x_dates, buy_and_hold, label="Buy & Hold", color="orange", linestyle="--")
    plt.title("Capital Growth Comparison")
    plt.xlabel("Date")
    plt.ylabel("Capital ($)")
    plt.grid(True)
    plt.legend()
    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax2.xaxis.get_major_locator()))
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save to PNG
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()