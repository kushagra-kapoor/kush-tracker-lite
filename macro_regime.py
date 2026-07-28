import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_macro_data(ticker='^NSEI', period='2y'):
    """Fetch macro index data and calculate SMAs and Drawdowns."""
    try:
        tkr = yf.Ticker(ticker)
        df = tkr.history(period=period)
        if df.empty:
            return None
            
        df = df.dropna(subset=['Close'])
        if df.empty:
            return None
            
        df['ema_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_100'] = df['Close'].rolling(window=100).mean()
        df['sma_150'] = df['Close'].rolling(window=150).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()
        
        # 52-Week High (approx 252 trading days)
        df['52w_high'] = df['High'].rolling(window=252, min_periods=50).max()
        df['drawdown_pct'] = ((df['Close'] / df['52w_high']) - 1) * 100
        
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def determine_risk_regime(df):
    """
    Determine risk regime based on Kush Tracker Institutional rules:
    - Default: SAFE
    - Nifty SMA50 Break & Drawdown from High >4% -> ELEVATED (BSL 6%, TSL 15%, 1% Port Hit)
    - Nifty SMA100 Break & Drawdown 6% from High -> WARNING (BSL 6%, TSL 12%, 0.9% Port Hit)
    - Nifty SMA150 Break OR Drawdown 8% from High -> HIGH RISK (No New Buy, TSL 10%, 0.5% Port Hit)
    - Nifty 200 Break OR Drawdown 10% from High -> DANGER (No Buy, TSL 9%, 0.25% Port Hit)
    """
    if df is None or df.empty:
        return _default_regime()
        
    last = df.iloc[-1]
    c = last['Close']
    dd = abs(last['drawdown_pct']) if pd.notna(last['drawdown_pct']) else 0
    
    s50 = last.get('sma_50', 0)
    s100 = last.get('sma_100', 0)
    s150 = last.get('sma_150', 0)
    s200 = last.get('sma_200', 0)
    
    # Check conditions (Highest severity first)
    if (s200 > 0 and c < s200) or dd >= 10.0:
        return {
            'level': 'DANGER',
            'color': '#ef4444',
            'bsl': 'NO NEW BUYS',
            'tsl': '9%',
            'max_port_hit': 0.25,
            'desc': 'Nifty broke 200 SMA or >10% Drawdown. Maximum defense.'
        }
    elif (s150 > 0 and c < s150) or dd >= 8.0:
        return {
            'level': 'HIGH RISK',
            'color': '#f97316',
            'bsl': 'NO NEW BUYS',
            'tsl': '10%',
            'max_port_hit': 0.5,
            'desc': 'Nifty broke 150 SMA or >8% Drawdown. No new exposure.'
        }
    elif (s100 > 0 and c < s100) and dd >= 6.0:
        return {
            'level': 'WARNING',
            'color': '#eab308',
            'bsl': '6%',
            'tsl': '12%',
            'max_port_hit': 0.9,
            'desc': 'Nifty below 100 SMA & >6% Drawdown. Tighten stops.'
        }
    elif (s50 > 0 and c < s50) and dd >= 4.0:
        return {
            'level': 'ELEVATED',
            'color': '#38bdf8',
            'bsl': '6%',
            'tsl': '15%',
            'max_port_hit': 1.0,
            'desc': 'Nifty below 50 SMA & >4% Drawdown. Proceed with caution.'
        }
    else:
        return _default_regime()

def _default_regime():
    return {
        'level': 'SAFE',
        'color': '#10b981',
        'bsl': '8-10%',
        'tsl': '15-20%',
        'max_port_hit': 2.0,
        'desc': 'Market is healthy. Normal risk parameters apply.'
    }

def get_macro_snapshot():
    """Returns a full snapshot of Nifty 50 and Nifty 500 macro health."""
    n50_df = fetch_macro_data('^NSEI')
    n500_df = fetch_macro_data('^CRSLDX')
    sp500_df = fetch_macro_data('^GSPC')
    nasdaq_df = fetch_macro_data('^IXIC')
    
    n50_regime = determine_risk_regime(n50_df)
    
    snapshot = {
        'nifty50': {},
        'nifty500': {},
        'sp500': {},
        'nasdaq': {},
        'regime': n50_regime
    }
    
    if n50_df is not None and not n50_df.empty:
        last = n50_df.iloc[-1]
        snapshot['nifty50'] = {
            'close': last['Close'],
            'ema_21': last['ema_21'],
            'sma_50': last['sma_50'],
            'sma_100': last['sma_100'],
            'sma_150': last['sma_150'],
            'sma_200': last['sma_200'],
            'drawdown': last['drawdown_pct']
        }
        
    if n500_df is not None and not n500_df.empty:
        last = n500_df.iloc[-1]
        snapshot['nifty500'] = {
            'close': last['Close'],
            'ema_21': last['ema_21'],
            'sma_50': last['sma_50'],
            'sma_100': last['sma_100'],
            'sma_150': last['sma_150'],
            'sma_200': last['sma_200'],
            'drawdown': last['drawdown_pct']
        }
        
    if sp500_df is not None and not sp500_df.empty:
        last = sp500_df.iloc[-1]
        snapshot['sp500'] = {
            'close': last['Close'],
            'ema_21': last['ema_21'],
            'sma_50': last['sma_50'],
            'sma_100': last['sma_100'],
            'sma_150': last['sma_150'],
            'sma_200': last['sma_200'],
            'drawdown': last['drawdown_pct']
        }
        
    if nasdaq_df is not None and not nasdaq_df.empty:
        last = nasdaq_df.iloc[-1]
        snapshot['nasdaq'] = {
            'close': last['Close'],
            'ema_21': last['ema_21'],
            'sma_50': last['sma_50'],
            'sma_100': last['sma_100'],
            'sma_150': last['sma_150'],
            'sma_200': last['sma_200'],
            'drawdown': last['drawdown_pct']
        }
        
    return snapshot

if __name__ == '__main__':
    snap = get_macro_snapshot()
    print(snap['regime'])
