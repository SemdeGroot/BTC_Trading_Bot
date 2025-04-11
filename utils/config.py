import os
from dotenv import load_dotenv

load_dotenv()

# === API KEYS ===
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
COINDESK_API_KEY = os.getenv("COINDESK_API_KEY")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
BYBIT_IS_LIVE = os.getenv("BYBIT_IS_LIVE", "False") == "True"

# === DATA PATHS ===
DATA_PATH = os.getenv("DATA_PATH")
ENSEMBLE_XGB_PATH = os.getenv("ENSEMBLE_XGB_PATH")
SCALER_PATH = os.getenv("SCALER_PATH")
X_TEST_PATH = os.getenv("X_TEST_PATH")
X_TRAIN_PATH = os.getenv("X_TRAIN_PATH")
PLOT_PATH = os.getenv("PLOT_PATH")

# === EMAIL FOR RAPPORT ===
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT"))
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")