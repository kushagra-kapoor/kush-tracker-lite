"""
High-Speed Incremental Price History Caching Engine for Kush Tracker Lite.
Fetches multi-threaded yfinance price history and caches locally to disk (.cache/price_history.pkl).
"""
import os
import pickle
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import config

def load_price_history_cache():
    if os.path.exists(config.PRICE_HISTORY_CACHE_PATH):
        try:
            with open(config.PRICE_HISTORY_CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

def save_price_history_cache(cache_data):
    try:
        os.makedirs(os.path.dirname(config.PRICE_HISTORY_CACHE_PATH), exist_ok=True)
        with open(config.PRICE_HISTORY_CACHE_PATH, "wb") as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"[Price Cache] Failed to save cache: {e}")

def fetch_incremental_history(tickers: list, days: int = 252, force_today_refresh: bool = False) -> dict:
    """
    Fetch multi-ticker yfinance price history with high-speed disk caching.
    Returns dict mapping ticker -> DataFrame(Open, High, Low, Close, Volume).
    """
    if not tickers:
        return {}

    # Clean ticker symbols
    tickers = list(set([t.strip().upper() for t in tickers if t and t.strip()]))
    cache = load_price_history_cache()
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    missing_tickers = []
    result = {}
    
    for t in tickers:
        if not force_today_refresh and t in cache:
            df = cache[t]
            if not df.empty and len(df) >= 30:
                result[t] = df
                continue
        missing_tickers.append(t)
        
    if missing_tickers:
        print(f"[Price History Engine] Batch fetching {len(missing_tickers)} tickers via yfinance...")
        try:
            batch_df = yf.download(missing_tickers, period=f"{days}d", group_by="ticker", threads=True, progress=False)
            
            for t in missing_tickers:
                df_t = pd.DataFrame()
                if len(missing_tickers) == 1:
                    df_t = batch_df.copy()
                elif isinstance(batch_df.columns, pd.MultiIndex):
                    if t in batch_df.columns.levels[0]:
                        df_t = batch_df[t].copy()
                    elif t in batch_df.columns.levels[1]:
                        df_t = batch_df.xs(t, axis=1, level=1).copy()
                        
                if not df_t.empty and "Close" in df_t.columns:
                    df_t = df_t.dropna(subset=["Close"]).copy()
                    cache[t] = df_t
                    result[t] = df_t
        except Exception as e:
            print(f"[Price History Engine] Error batch downloading: {e}")

    # Return fetched result combined with cached tickers
    for t in tickers:
        if t in cache and t not in result:
            result[t] = cache[t]
            
    save_price_history_cache(cache)
    return result
