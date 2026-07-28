import pandas as pd
import numpy as np
import requests
import os
import sqlite3
from datetime import datetime
from database import save_market_liquidity

def get_nifty500_tickers():
    valid_tickers = []
    try:
        from market_data import fetch_nifty500_tickers
        valid_tickers = fetch_nifty500_tickers(show_progress=False)
        valid_tickers = [t.replace('.NS', '') for t in valid_tickers]
    except Exception as e:
        print(f"Failed to fetch nifty 500: {e}")
        
    if not valid_tickers and os.path.exists('tickers.txt'):
        with open('tickers.txt', 'r') as f:
            valid_tickers = [line.strip().upper() for line in f if line.strip()]
                
    valid_tickers = [t + '.NS' if not t.endswith('.NS') else t for t in valid_tickers]
    return valid_tickers

def compute_liquidity(market='IN'):
    matrix_path = 'historical_prices_matrix.pkl'
    if not os.path.exists(matrix_path):
        print("historical_prices_matrix.pkl not found.")
        return False
        
    try:
        df = pd.read_pickle(matrix_path)
    except Exception as e:
        print(f"Failed to load matrix: {e}")
        return False

    if df.empty:
        print("Matrix is empty.")
        return False

    if market == 'IN':
        nifty500_tickers = get_nifty500_tickers()
        if not nifty500_tickers:
            print("Could not retrieve Nifty 500 tickers, falling back to all available IN tickers.")
            nifty500_tickers = [t for t in df.columns.get_level_values(0).unique() if str(t).endswith('.NS')]

        # Filter matrix to only Nifty 500 stocks
        available_tickers = [t for t in nifty500_tickers if t in df.columns.get_level_values(0)]
    else:
        # For US, use all non-IN/BO tickers
        available_tickers = [t for t in df.columns.get_level_values(0).unique() if not str(t).endswith('.NS') and not str(t).endswith('.BO')]

    df_filtered = df[available_tickers]

    if df_filtered.empty:
        print(f"Filtered matrix for {market} is empty.")
        return False

    try:
        closes = df_filtered.xs('Close', level=1, axis=1)
        volumes = df_filtered.xs('Volume', level=1, axis=1)
    except KeyError:
        print("Close or Volume missing from matrix.")
        return False

    # Calculate daily turnover for each stock
    daily_turnover = closes * volumes
    
    # Sum across all stocks to get total daily turnover for the market
    total_daily_turnover = daily_turnover.sum(axis=1)
    
    # YFinance Bug Fix: Yahoo Finance returned literally 0 volume for Nifty 500
    # Replace any 0 turnover days with NaN and interpolate before smoothing
    total_daily_turnover = total_daily_turnover.replace(0, np.nan)
    total_daily_turnover = total_daily_turnover.interpolate(method='linear')
    
    # Calculate rolling 20-day sum (Monthly Trade)
    rolling_monthly_trade = total_daily_turnover.rolling(window=20, min_periods=10).sum()
    
    if market == 'IN':
        # Convert to K Crores (10,000,000,000)
        monthly_trade = rolling_monthly_trade / 10_000_000_000
    else:
        # Convert to Billion USD (1,000,000,000)
        monthly_trade = rolling_monthly_trade / 1_000_000_000
    
    # Calculate 200-day SMA of the monthly trade
    sma_200 = monthly_trade.rolling(window=200, min_periods=100).mean()
    
    # Combine into a dataframe
    result_df = pd.DataFrame({
        'monthly_turnover_k_cr': monthly_trade, # Using same column name to avoid DB schema changes, but meaning changes based on market
        'sma_200': sma_200
    }).dropna(subset=['monthly_turnover_k_cr'])
    
    # Keep only the last 3 years of data (approx 750 trading days) to avoid bloating the DB unnecessarily
    # But for a macro chart, we want as much history as possible, let's keep all valid non-NaN rows.
    result_df = result_df.dropna()
    
    records = []
    for idx, row in result_df.iterrows():
        # date, market, turnover, sma
        records.append((idx.strftime('%Y-%m-%d'), market, float(row['monthly_turnover_k_cr']), float(row['sma_200'])))
        
    save_market_liquidity(records)
    print(f"[{market}] Saved {len(records)} daily liquidity records to database.")
    return True

if __name__ == "__main__":
    import database
    database.init_database()
    compute_liquidity(market='IN')
    compute_liquidity(market='US')
