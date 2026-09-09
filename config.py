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
    'HISTORY_DAYS': 400,
    'TRADING_DAYS_1M': 21,
    'TRADING_DAYS_3M': 63,
    'TRADING_DAYS_6M': 126,
}

BATCH_SIZE = 100
DEFAULT_HISTORY_DAYS = 400

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================
EMA_PERIODS = {
    'FAST': 8,
    'MEDIUM': 21,
    'SLOW': 50,
}

ATR_PERIODS = {
    'SHORT': 14,
    'LONG': 30,
}

# =============================================================================
# RISK MANAGEMENT THRESHOLDS
# =============================================================================
RISK_LIMITS = {
    'FULL_EXIT_LOSS_PCT': 2.0,
    'TRIM_LOSS_PCT': 1.0,
    'TRIM_PERCENTAGE': 50,
}

# =============================================================================
# TREND RULES
# =============================================================================
TREND_RULES = {
    'WARNING_CONSECUTIVE_DAYS': 5,
    'WARNING_DOWN_DAY_PCT': 5.0,
    'TRACKING_SIZE_MIN': 10,
    'TRACKING_SIZE_MAX': 25,
}

# =============================================================================
# STRUCTURAL FAILURE THRESHOLDS
# =============================================================================
STRUCTURAL_FAILURE = {
    'DOWN_FROM_52W_HIGH_PCT': 20,
    'SMA_RECLAIM_DAYS': 10,
    'WEEKLY_DISTRIBUTION_DROP_PCT': 8,
    'WEEKLY_VOL_MULTIPLIER': 1.5,
}

# =============================================================================
# AVERAGING UP (PYRAMIDING) CONDITIONS
# =============================================================================
AVERAGING_UP = {
    'MIN_DAYS_ABOVE_21EMA': 10,
    'ATR_CONTRACTION_RATIO': 0.85,
    'MAX_DOWN_DAY_PCT': 4.0,
    'MIN_RS_SCORE': 80,
    'MAX_DISTANCE_FROM_52W': 15,
    'ADD_SIZE_MIN': 25,
    'ADD_SIZE_MAX': 50,
    'MAX_EXTENSION_FROM_21EMA': 5.0,
    'MAX_EXTENSION_FROM_50SMA': 4.0,
    'MAX_PULLBACK_VOL': 1.0,
}

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================
DISPLAY = {
    'COLORS': {
        'SAFE': 'green',
        'CAUTION': 'yellow',
        'RISK': 'red',
    }
}

# =============================================================================
# CANSLIM SWING TRADING SYSTEM SETTINGS
# =============================================================================
CANSLIM_SWING_SETTINGS = {
    'MAX_POSITIONS': 8,                      # Target 12.5% allocation each
    'MIN_SALES_GROWTH_PCT': 25.0,            # Minimum 25% Sales growth YoY (Sole Fundamental Gate)
    'INITIAL_STOP_LOSS_PCT': 4.5,            # Target initial risk
    'HARD_MAX_STOP_PCT': 7.5,                # O'Neil maximum cutoff
    'BUY_ZONE_MAX_PCT': 5.0,                 # Max 5% above pivot (never chase extended)
    'BREAKOUT_VOL_MULTIPLIER': 1.4,          # 1.4x 50-day average volume
    'TRAILING_EXIT_CONSECUTIVE_DAYS_21EMA': 2, # 2 consecutive closes below 21 EMA
    'EIGHT_WEEK_HOLD_GAIN_PCT': 20.0,        # 20% gain within 3 weeks
    'EIGHT_WEEK_HOLD_MAX_DAYS': 15,          # 15 trading days (~3 weeks)
    # Pyramiding (Averaging Up)
    'PYRAMID_MIN_GAIN_PCT': 2.5,             # Position must be up at least 2.5% to add
    'PYRAMID_MAX_EXTENSION_21EMA': 5.0,      # Add only within 5% of 21 EMA
    'PYRAMID_TRANCHE_INITIAL': 0.50,         # Tranche 1: 50% size
    'PYRAMID_TRANCHE_ADD1': 0.30,            # Tranche 2: 30% size
    'PYRAMID_TRANCHE_ADD2': 0.20,            # Tranche 3: 20% size
}
