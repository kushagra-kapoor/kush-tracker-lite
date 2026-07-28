import os
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime
from config import DATABASE_PATH

# 250 trading days in a year
YEAR_TRADING_DAYS = 250
# If the stock is within 2% of its 52w high/low, we count it as a "New High/Low"
PROXIMITY_THRESHOLD = 0.02 

def get_connection():
    return sqlite3.connect(DATABASE_PATH)

def init_breadth_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_breadth_daily (
            date TEXT,
            market TEXT DEFAULT 'IN',
            above_50_pct REAL,
            above_200_pct REAL,
            net_new_highs INTEGER,
            new_highs_count INTEGER,
            new_lows_count INTEGER,
            total_stocks INTEGER,
            surge_extreme_5d INTEGER,
            panic_extreme_5d INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, market)
        )
    ''')
    conn.commit()
    conn.close()

def fetch_historical_breadth(market='IN', days=60):
    init_breadth_table()
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM market_breadth_daily WHERE market=? ORDER BY date DESC LIMIT ?", conn, params=(market, days))
    conn.close()
    if not df.empty:
        df = df.sort_values('date').set_index('date')
    return df

def sync_breadth_history(market='IN', lookback_days=60):
    init_breadth_table()
    
    matrix_path = 'historical_prices_matrix.pkl'
    if not os.path.exists(matrix_path):
        return

    try:
        df = pd.read_pickle(matrix_path)
    except Exception as e:
        print(f"[Breadth Engine] Failed to load matrix: {e}")
        return

    if df.empty:
        return
        
    valid_tickers = []
    if market == 'IN':
        try:
            from market_data import fetch_nifty_total_market_tickers
            valid_tickers = fetch_nifty_total_market_tickers(show_progress=False)
        except Exception as e:
            print(f"[Breadth Engine] Failed to fetch tickers: {e}")
            
        if not valid_tickers and os.path.exists('tickers.txt'):
            with open('tickers.txt', 'r') as f:
                valid_tickers = [line.strip().upper() for line in f if line.strip()]
                valid_tickers = [t + '.NS' if not t.endswith('.NS') else t for t in valid_tickers]
    else:
        if os.path.exists('tickers_us.txt'):
            with open('tickers_us.txt', 'r') as f:
                valid_tickers = [line.strip().upper() for line in f if line.strip()]

    if valid_tickers:
        available_tickers = [t for t in valid_tickers if t in df.columns.get_level_values(0)]
        df = df[available_tickers]
        
    if df.empty:
        return

    try:
        closes = df.xs('Close', level=1, axis=1).dropna(how='all')
        highs = df.xs('High', level=1, axis=1).dropna(how='all')
        lows = df.xs('Low', level=1, axis=1).dropna(how='all')
    except Exception:
        return

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM market_breadth_daily WHERE market=?", (market,))
    last_date_row = cursor.fetchone()
    last_date_str = last_date_row[0] if last_date_row and last_date_row[0] else None

    dates_in_matrix = closes.index.strftime('%Y-%m-%d').tolist()
    
    if not last_date_str:
        missing_dates = dates_in_matrix[-lookback_days:]
    else:
        missing_dates = [d for d in dates_in_matrix if d > last_date_str]

    if not missing_dates:
        conn.close()
        return

    sma_50 = closes.rolling(window=50, min_periods=40).mean()
    sma_200 = closes.rolling(window=200, min_periods=180).mean()
    
    # Pradeep Bonde 5-Day Extreme Signals
    ret_5d = closes / closes.shift(5) - 1
    
    year_highs = highs.rolling(window=YEAR_TRADING_DAYS, min_periods=200).max()
    year_lows = lows.rolling(window=YEAR_TRADING_DAYS, min_periods=200).min()

    records_to_insert = []
    
    for date_str in missing_dates:
        try:
            dt_index = closes.index[dates_in_matrix.index(date_str)]
            
            c = closes.loc[dt_index]
            s50 = sma_50.loc[dt_index]
            s200 = sma_200.loc[dt_index]
            yh = year_highs.loc[dt_index]
            yl = year_lows.loc[dt_index]
            r5 = ret_5d.loc[dt_index]
            
            valid_50 = c.notna() & s50.notna()
            total_50 = valid_50.sum()
            above_50 = (c[valid_50] > s50[valid_50]).sum()
            above_50_pct = (above_50 / total_50 * 100) if total_50 > 0 else 0
            
            valid_200 = c.notna() & s200.notna()
            total_200 = valid_200.sum()
            above_200 = (c[valid_200] > s200[valid_200]).sum()
            above_200_pct = (above_200 / total_200 * 100) if total_200 > 0 else 0
            
            valid_hl = c.notna() & yh.notna() & yl.notna()
            new_h = (c[valid_hl] >= (yh[valid_hl] * (1 - PROXIMITY_THRESHOLD))).sum()
            new_l = (c[valid_hl] <= (yl[valid_hl] * (1 + PROXIMITY_THRESHOLD))).sum()
            net_nh = int(new_h - new_l)
            
            valid_r5 = r5.notna()
            s_thresh = 0.12 if market == 'IN' else 0.20
            p_thresh = -0.12 if market == 'IN' else -0.20
            surge_count = (r5[valid_r5] >= s_thresh).sum()
            panic_count = (r5[valid_r5] <= p_thresh).sum()
            
            records_to_insert.append((
                date_str, market, float(above_50_pct), float(above_200_pct), 
                net_nh, int(new_h), int(new_l), int(total_50),
                int(surge_count), int(panic_count)
            ))
        except Exception as e:
            print(f"[Breadth Engine] Error calculating for {date_str}: {e}")

    if records_to_insert:
        cursor.executemany('''
            INSERT OR REPLACE INTO market_breadth_daily 
            (date, market, above_50_pct, above_200_pct, net_new_highs, new_highs_count, new_lows_count, total_stocks, surge_extreme_5d, panic_extreme_5d)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', records_to_insert)
        conn.commit()
    conn.close()

def compute_daily_breadth(force=False, market='IN'):
    init_breadth_table()
    sync_breadth_history(market=market, lookback_days=60)
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM market_breadth_daily WHERE market = ? ORDER BY date DESC LIMIT 1", (market,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'date': row[0],
            'market': row[1],
            'above_50_pct': float(row[2]),
            'above_200_pct': float(row[3]),
            'net_new_highs': int(row[4]),
            'new_highs_count': int(row[5]),
            'new_lows_count': int(row[6]),
            'total_stocks': int(row[7]),
            'surge_extreme_5d': int(row[8]),
            'panic_extreme_5d': int(row[9])
        }
    return None

# removed the redundant .iloc[-1] logic here as it is moved into sync_breadth_history

if __name__ == "__main__":
    print("Testing Breadth Engine...")
    result = compute_daily_breadth(force=True)
    if result:
        print(f"Breadth for {result['date']}:")
        print(f"  Total Universe: {result['total_stocks']}")
        print(f"  % Above 50 SMA: {result['above_50_pct']:.1f}%")
        print(f"  % Above 200 SMA: {result['above_200_pct']:.1f}%")
        print(f"  Net New Highs: {result['net_new_highs']} ({result['new_highs_count']} NH, {result['new_lows_count']} NL)")
    else:
        print("Failed to compute breadth.")
