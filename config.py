"""
Complete Config Module for Kush Tracker Lite.
Provides all constants, relative paths, and settings for local and Streamlit Cloud execution.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")

os.makedirs(CACHE_DIR, exist_ok=True)

# Ticker file paths
TICKERS_IN_PATH = os.path.join(BASE_DIR, "tickers.txt")
TICKERS_ETF_IN_PATH = os.path.join(BASE_DIR, "tickers_etf.txt")
TICKERS_US_PATH = os.path.join(BASE_DIR, "tickers_us.txt")
TICKERS_ETF_US_PATH = os.path.join(BASE_DIR, "tickers_us_etf.txt")

# Database & History cache paths
DB_PATH = os.path.join(CACHE_DIR, "kush_tracker_lite.db")
DATABASE_PATH = DB_PATH
PRICE_HISTORY_CACHE_PATH = os.path.join(CACHE_DIR, "price_history.pkl")

# RS Weights
RS_WEIGHTS = {
    '1M': 0.40,
    '3M': 0.35,
    '6M': 0.25,
}

RS_THRESHOLDS = {
    'ELITE': 85,
    'VALID': 70,
    'WEAKENING': 65,
}

BENCHMARK_TICKER = '^CRSLDX'

DATA_SETTINGS = {
    'HISTORY_DAYS': 252,
    'TRADING_DAYS_1M': 21,
    'TRADING_DAYS_3M': 63,
    'TRADING_DAYS_6M': 126,
}

BATCH_SIZE = 100
DEFAULT_HISTORY_DAYS = 252
