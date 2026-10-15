import pandas as pd
from datetime import datetime, timedelta
from config import BENCHMARK_TICKER, RS_WEIGHTS

try:
    import streamlit as st
    cache_decorator = st.cache_data(ttl=600)
except Exception:
    def cache_decorator(f):
        return f

SAFE_HARBOR_TICKERS = {
    'LIQUIDCASE.NS': 'Liquid Fund (Cash Parking)',
    'GOLDBEES.NS': 'Gold (Safe Haven)',
    'MON100.NS': 'Nasdaq 100 (Geographical Diversification)',
    'SILVERBEES.NS': 'Silver (Precious Metal)'
}

def _fetch_series(ticker: str) -> pd.Series:
    """
    Fetches 1y daily close series using curl_cffi to bypass Yahoo Finance bot blocking,
    falling back to yfinance if needed.
    """
    try:
        from curl_cffi import requests
        url = f'https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range=1y&interval=1d'
        r = requests.get(url, impersonate='chrome', timeout=8)
        if r.status_code == 200:
            data = r.json()
            res = data.get('chart', {}).get('result', [])
            if res:
                close = res[0].get('indicators', {}).get('quote', [{}])[0].get('close', [])
                s = pd.Series(close).dropna()
                if len(s) >= 25:
                    return s
    except Exception:
        pass
        
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        df = t.history(period='1y').dropna(subset=['Close'])
        if not df.empty and len(df) >= 20:
            return df['Close'].dropna()
    except Exception:
        pass
        
    return pd.Series(dtype=float)

@cache_decorator
def evaluate_safe_harbor(days=150, cache_buster=2) -> list:
    """
    Evaluates momentum & relative strength of Safe Harbor assets against the Nifty 500 benchmark.
    Returns recommendations for capital rotation during weak or corrective market regimes.
    """
    # 1. Fetch Benchmark
    b_series = _fetch_series(BENCHMARK_TICKER)
    if b_series.empty or len(b_series) < 25:
        b_series = _fetch_series('^NSEI')
        
    b1, b3 = 0.0, 0.0
    if len(b_series) >= 25:
        b1 = float((b_series.iloc[-1] / b_series.iloc[-22]) - 1.0)
        b3 = float((b_series.iloc[-1] / b_series.iloc[-min(64, len(b_series))]) - 1.0)

    # 2. Fetch Safe Harbor Assets
    results = []
    
    for ticker, name in SAFE_HARBOR_TICKERS.items():
        s = _fetch_series(ticker)
        if len(s) < 25:
            continue
            
        r1 = float((s.iloc[-1] / s.iloc[-22]) - 1.0)
        r3 = float((s.iloc[-1] / s.iloc[-min(64, len(s))]) - 1.0)
        
        rel_1m = float(r1 - b1)
        rel_3m = float(r3 - b3)
        
        score = float((r1 * 0.7) + (r3 * 0.3))
        rel_score = float((rel_1m * 0.7) + (rel_3m * 0.3))
        
        close = float(s.iloc[-1])
        ema21 = float(s.ewm(span=21, adjust=False).mean().iloc[-1])
        sma50 = float(s.rolling(window=min(50, len(s)), min_periods=10).mean().iloc[-1])
        uptrend = (close > ema21) and (close > sma50)
        
        if ticker == 'LIQUIDCASE.NS':
            status = 'Excellent Parking'
        elif uptrend and rel_score > 0 and score > 0:
            status = 'Strong Rotation Candidate'
        elif uptrend and score > 0:
            status = 'Defensive Hold'
        elif rel_score > 0:
            status = 'Accumulating (Watch)'
        else:
            status = 'Weak (Avoid)'
            
        results.append({
            'Ticker': ticker.replace('.NS', ''),
            'Asset': name,
            'Status': status,
            '1M_Return': round(r1 * 100.0, 2),
            '3M_Return': round(r3 * 100.0, 2),
            'Rel_Strength_Score': round(rel_score * 100.0, 2),
            'Trend': 'Up' if uptrend else 'Down'
        })
        
    results = sorted(results, key=lambda x: x.get('Rel_Strength_Score', 0.0), reverse=True)
    return results
