"""
Market Data & Universe Resolver for Kush Tracker Lite.
"""
import os
import pandas as pd
import numpy as np
import config
from price_history_manager import fetch_incremental_history
from nifty500_universe import get_nifty500_tickers

def fetch_nifty_total_market_tickers():
    """Fetch Indian Stock Universe (Nifty 500 / Total Market)."""
    if os.path.exists(config.TICKERS_IN_PATH):
        try:
            with open(config.TICKERS_IN_PATH, "r") as f:
                tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
            if tickers:
                return list(set(tickers))
        except Exception:
            pass
    return get_nifty500_tickers()

def fetch_us_tickers():
    """Fetch US Equities Stock Universe."""
    if os.path.exists(config.TICKERS_US_PATH):
        try:
            with open(config.TICKERS_US_PATH, "r") as f:
                tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
            if tickers:
                return list(set(tickers))
        except Exception:
            pass
    return ["AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD", "NFLX", "DELL", "DDOG", "FTNT", "ENVA", "GH"]

def get_top_5_rs_leaders(market: str = "INDIA") -> list:
    """
    Computes top 5 Relative Strength leaders for India or US.
    Returns list of dicts: [{'ticker': 'DIACABS', 'rs_score': 98.9, 'industry': 'SPECIALTY MACHINERY'}]
    """
    tickers = fetch_nifty_total_market_tickers()[:150] if market.upper() == "INDIA" else fetch_us_tickers()[:100]
    
    # Add .NS suffix for India yfinance lookup if missing
    yf_tickers = [t if (market.upper() != "INDIA" or t.endswith(".NS")) else f"{t}.NS" for t in tickers]
    
    histories = fetch_incremental_history(yf_tickers, days=120)
    scores = []
    
    for orig_t, yf_t in zip(tickers, yf_tickers):
        df = histories.get(yf_t, pd.DataFrame())
        if not df.empty and len(df) >= 40:
            try:
                close = df['Close']
                ret_3m = (close.iloc[-1] / close.iloc[-60] - 1) if len(close) >= 60 else (close.iloc[-1] / close.iloc[0] - 1)
                ret_1m = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0
                score = (ret_3m * 0.7 + ret_1m * 0.3) * 100
                scores.append({
                    'ticker': orig_t,
                    'raw_score': score,
                    'price': float(close.iloc[-1])
                })
            except Exception:
                continue
                
    if not scores:
        return []
        
    scores = sorted(scores, key=lambda x: x['raw_score'], reverse=True)
    total = len(scores)
    
    top_5 = []
    for idx, item in enumerate(scores[:5]):
        percentile = round(((total - idx) / total) * 99, 1)
        top_5.append({
            'ticker': item['ticker'],
            'rs_score': percentile,
            'industry': 'LEADERSHIP EQUITY'
        })
        
    return top_5
