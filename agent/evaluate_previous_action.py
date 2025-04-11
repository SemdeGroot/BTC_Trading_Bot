import os
import pandas as pd
import numpy as np
import torch
import joblib
from glob import glob

from agent_model import DQNAgent, calculate_reward, ensemble_predict  # Externalized agent code

# === Constants ===
state_size = 19
action_size = 3
model_path = "models/dqn_agent.pth"
log_path = "data/trading_log.csv"
ensemble_paths = sorted(glob("data/models/ensemble_xgb/*.joblib"))
ensemble_models = [joblib.load(p) for p in ensemble_paths]

# === Utility ===
def load_data():
    X_test = pd.read_csv("data/X_test.csv")
    merged_df = pd.read_csv("data/df_merged.csv")
    df_test = merged_df.iloc[-len(X_test):].reset_index(drop=True)
    return X_test.values, df_test

# === Evaluate previous action ===
def evaluate_previous_action():
    if not os.path.exists(log_path) or os.path.getsize(log_path) < 2:
        print("No sufficient trading log to evaluate.")
        return

    # Load log
    df_log = pd.read_csv(log_path)
    if len(df_log) < 2:
        print("Not enough entries to evaluate previous action.")
        return

    last_evaluated = df_log.iloc[-2].copy()

    # Load input features and price data
    X_test, df_test = load_data()
    if len(X_test) < 2:
        print("Not enough data to evaluate.")
        return

    # Load agent
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)

    # Reconstruct state from t-1
    idx = -2
    x_prev = X_test[idx]
    y_proba_prev, unc_prev = ensemble_predict(x_prev.reshape(1, -1), ensemble_models)
    confidence_prev = abs(y_proba_prev - 0.5) * 2
    fear_greed_prev = df_test.iloc[idx]["fear_greed"]
    state_prev = np.concatenate([[y_proba_prev, unc_prev, fear_greed_prev], x_prev])

    action_map = {"SHORT": 0, "NO TRADE": 1, "LONG": 2}
    action_prev = action_map.get(last_evaluated["Action"], 1)

    # Use day t HIGH/LOW/CLOSE to evaluate what happened
    close_t = df_test.iloc[-2]["CLOSE"]
    close_t1 = df_test.iloc[-1]["CLOSE"]
    high_t1 = df_test.iloc[-1]["HIGH"]
    low_t1 = df_test.iloc[-1]["LOW"]
    atr = df_test.iloc[-2]["volatility_atr"]

    tp_level = close_t + atr * 1.4802669357528435 if action_prev == 2 else close_t - atr * 1.4802669357528435
    sl_level = close_t - atr * 0.2275244055029675 if action_prev == 2 else close_t + atr * 0.2275244055029675

    tp_hit = (high_t1 >= tp_level) if action_prev == 2 else (low_t1 <= tp_level)
    sl_hit = (low_t1 <= sl_level) if action_prev == 2 else (high_t1 >= sl_level)

    if tp_hit and sl_hit:
        close_reason = np.random.choice([-1, 1])  # Both hit
    elif tp_hit:
        close_reason = 1
    elif sl_hit:
        close_reason = -1
    else:
        close_reason = 0  # No hit, still open (or max hold to be evaluated elsewhere)

    pct_change = (close_t1 - close_t) / close_t
    reward = calculate_reward(pct_change, action_prev, confidence_prev, unc_prev)

    # Create next state for learning
    x_next = X_test[-1]
    y_proba_next, unc_next = ensemble_predict(x_next.reshape(1, -1), ensemble_models)
    fear_greed_next = df_test.iloc[-1]["fear_greed"]
    next_state = np.concatenate([[y_proba_next, unc_next, fear_greed_next], x_next])

    agent.remember(state_prev, action_prev, reward, next_state, done=False)
    agent.train()
    agent.save(model_path)

    # Update log at row -2
    df_log.at[df_log.index[-2], "Reward"] = round(reward, 4)
    df_log.at[df_log.index[-2], "pct_change"] = round(pct_change * 100, 4)
    df_log.at[df_log.index[-2], "Close Reason"] = close_reason
    df_log.to_csv(log_path, index=False)
    print("Previous action evaluated and agent trained.")