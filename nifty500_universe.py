# NIFTY 500 Universe for Kush Tracker
# Provides RS reference data for market-normalized calculations

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sqlite3
import requests
from io import StringIO
from config import DATABASE_PATH, DATA_SETTINGS, RS_WEIGHTS

# Fallback tickers if NSE download fails (top 100 liquid stocks)
FALLBACK_TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN", "BHARTIARTL",
    "ITC", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "BAJFINANCE", "HCLTECH",
    "TITAN", "SUNPHARMA", "WIPRO", "ULTRACEMCO", "ONGC", "NTPC", "POWERGRID", "NESTLEIND",
    "M&M", "TATAMOTORS", "JSWSTEEL", "TATASTEEL", "ADANIENT", "ADANIPORTS", "COALINDIA",
    "BAJAJFINSV", "TECHM", "HINDALCO", "INDUSINDBK", "GRASIM", "DRREDDY", "BRITANNIA",
    "DIVISLAB", "CIPLA", "APOLLOHOSP", "EICHERMOT", "BPCL", "HEROMOTOCO", "TATACONSUM",
    "SHREECEM", "DABUR", "GODREJCP", "PIDILITIND", "HAVELLS", "SIEMENS", "BAJAJ-AUTO",
    "BERGEPAINT", "MARICO", "INDIGO", "HINDPETRO", "BANKBARODA", "IOC", "SBILIFE",
    "HDFCLIFE", "ICICIPRULI", "DLF", "COLPAL", "AMBUJACEM", "ACC", "TATAPOWER", "VEDL",
    "GAIL", "BOSCHLTD", "MOTHERSON", "PIIND", "LUPIN", "TORNTPHARM", "AUROPHARMA",
    "BIOCON", "ZYDUSLIFE", "ALKEM", "PERSISTENT", "LTIM", "MPHASIS", "COFORGE",
    "TRENT", "MAXHEALTH", "POLYCAB", "KEI", "ASTRAL", "VOLTAS", "CROMPTON", "DIXON",
    "ZOMATO", "SHRIRAMFIN", "MUTHOOTFIN", "CHOLAFIN", "PFC", "RECLTD", "NATIONALUM",
    "HINDZINC", "NMDC", "JINDALSTEL", "MCX", "IRCTC"
]

# Cache for fetched tickers
_cached_tickers = None
_cache_date = None


def fetch_nifty500_tickers(show_progress: bool = True) -> list:
    """
    Fetch NIFTY 500 constituent tickers from NSE website.

    
    Falls back to hardcoded list if download fails.
    
    Returns:
        List of ticker symbols
    """
    global _cached_tickers, _cache_date
    
    # Return cached if already fetched today
    today = datetime.now().strftime('%Y-%m-%d')
    if _cached_tickers and _cache_date == today:
        return _cached_tickers
    
    if show_progress:
        print(f"  -> Fetching NIFTY 500 constituents from NSE...")
    
    try:
        from market_data import fetch_nifty500_tickers as md_fetch_nifty500
        tickers = md_fetch_nifty500(show_progress)
        
        # Strip .NS since this file's callers expect no suffix
        tickers = [t.replace('.NS', '') for t in tickers]
        
        if len(tickers) >= 100:
            if show_progress:
                print(f"  [OK] Fetched {len(tickers)} NIFTY 500 constituents from NSE")
            _cached_tickers = tickers
            _cache_date = today
            return tickers
        else:
            raise ValueError(f"Only {len(tickers)} tickers found, expected 500+")
            
    except Exception as e:
        print(f"  [!] Failed to fetch NIFTY 500 from NSE: {e}")
        print(f"  -> Using fallback list of {len(FALLBACK_TICKERS)} stocks")
        return FALLBACK_TICKERS


def get_connection():
    """Get database connection."""
    return sqlite3.connect(DATABASE_PATH)


def init_rs_reference_table():
    """Initialize the RS reference table for NIFTY 500 data."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rs_reference_nifty500 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            rs_raw REAL,
            r1 REAL,
            r3 REAL,
            r6 REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, ticker)
        )
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_rs_ref_date 
        ON rs_reference_nifty500(date)
    ''')
    
    conn.commit()
    conn.close()


def check_rs_reference_fresh(date: str = None) -> bool:
    """Check if RS reference data exists for today."""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT COUNT(*) FROM rs_reference_nifty500 
            WHERE date = ?
        ''', (date,))
        count = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        count = 0
    finally:
        conn.close()
    
    # Consider fresh if we have at least 100 stocks computed
    return count >= 100


def get_rs_reference_distribution(date: str = None) -> list:
    """Get RS raw distribution for NIFTY 500 on given date."""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT rs_raw FROM rs_reference_nifty500 
            WHERE date = ? AND rs_raw IS NOT NULL
            ORDER BY rs_raw
        ''', (date,))
        results = [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        results = []
    finally:
        conn.close()
    
    return results


def calculate_returns(close_prices: pd.Series, periods: int) -> float:
    """Calculate return over specified periods."""
    if len(close_prices) < periods + 1:
        return None
    
    current_price = close_prices.iloc[-1]
    past_price = close_prices.iloc[-(periods + 1)]
    
    if past_price == 0 or pd.isna(past_price):
        return None
    
    return (current_price / past_price) - 1


def fetch_stock_returns(ticker: str, days: int = 180) -> dict:
    """Fetch stock data and calculate returns."""
    try:
        yf_ticker = f"{ticker}.NS"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)
        
        stock = yf.Ticker(yf_ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty or len(df) < DATA_SETTINGS['TRADING_DAYS_6M']:
            return None
        
        close = df['Close']
        
        r1 = calculate_returns(close, DATA_SETTINGS['TRADING_DAYS_1M'])
        r3 = calculate_returns(close, DATA_SETTINGS['TRADING_DAYS_3M'])
        r6 = calculate_returns(close, DATA_SETTINGS['TRADING_DAYS_6M'])
        
        if r1 is None or r3 is None or r6 is None:
            return None
        
        return {'R1': r1, 'R3': r3, 'R6': r6}
    except Exception as e:
        return None


def compute_nifty500_rs_reference(show_progress: bool = True):
    """
    Compute RS_raw for all NIFTY 500 stocks and store in database.
    This should be run once per trading day.
    """
    from config import BENCHMARK_TICKER
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Check if already computed today
    if check_rs_reference_fresh(today):
        if show_progress:
            print(f"[OK] RS reference data already exists for {today}")
        return True
    
    if show_progress:
        print(f"Computing NIFTY 500 RS reference for {today}...")
    
    # Initialize table
    init_rs_reference_table()
    
    # Get benchmark returns
    if show_progress:
        print("  -> Fetching benchmark (NIFTY 500 Index)...")
    
    try:
        benchmark = yf.Ticker(BENCHMARK_TICKER)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=210)
        bench_df = benchmark.history(start=start_date, end=end_date)
        
        if bench_df.empty:
            print("  [!] Benchmark data unavailable")
            return False
        
        bench_close = bench_df['Close']
        b1 = calculate_returns(bench_close, DATA_SETTINGS['TRADING_DAYS_1M'])
        b3 = calculate_returns(bench_close, DATA_SETTINGS['TRADING_DAYS_3M'])
        b6 = calculate_returns(bench_close, DATA_SETTINGS['TRADING_DAYS_6M'])
        
        if b1 is None or b3 is None or b6 is None:
            print("  [!] Benchmark returns incomplete")
            return False
        
        if show_progress:
            print(f"  [OK] Benchmark returns: 1M={b1:.2%}, 3M={b3:.2%}, 6M={b6:.2%}")
    except Exception as e:
        print(f"  [X] Error fetching benchmark: {e}")
        return False
    
    # Fetch NIFTY 500 tickers dynamically from NSE
    nifty500_tickers = fetch_nifty500_tickers(show_progress=show_progress)
    
    # Compute RS for each NIFTY 500 stock
    conn = get_connection()
    cursor = conn.cursor()
    
    success_count = 0
    total = len(nifty500_tickers)
    
    if show_progress:
        print(f"  -> Bulk downloading {total} NIFTY 500 stocks (180 days)...")
        
    yf_tickers = [f"{t}.NS" for t in nifty500_tickers]
    
    try:
        bulk_data = yf.download(yf_tickers, start=start_date, end=end_date, group_by='ticker', threads=True, progress=show_progress)
    except Exception as e:
        print(f"  [X] Error during bulk download: {e}")
        return False

    for i, ticker in enumerate(nifty500_tickers):
        yf_ticker = f"{ticker}.NS"
        
        try:
            if len(yf_tickers) == 1 or not isinstance(bulk_data.columns, pd.MultiIndex):
                df = bulk_data.copy()
            else:
                df = bulk_data[yf_ticker].copy() if yf_ticker in bulk_data.columns.get_level_values(0) else pd.DataFrame()
                
            if df.empty or len(df) < DATA_SETTINGS['TRADING_DAYS_6M']:
                continue
                
            if 'Close' in df.columns:
                close = df['Close']
            elif 'close' in df.columns:
                close = df['close']
            else:
                continue
                
            r1 = calculate_returns(close, DATA_SETTINGS['TRADING_DAYS_1M'])
            r3 = calculate_returns(close, DATA_SETTINGS['TRADING_DAYS_3M'])
            r6 = calculate_returns(close, DATA_SETTINGS['TRADING_DAYS_6M'])
            
            if r1 is None or r3 is None or r6 is None:
                continue
            
            # Calculate relative returns
            rr1 = r1 - b1
            rr3 = r3 - b3
            rr6 = r6 - b6
            
            # Weighted RS_raw
            rs_raw = (
                rr1 * RS_WEIGHTS['1M'] +
                rr3 * RS_WEIGHTS['3M'] +
                rr6 * RS_WEIGHTS['6M']
            )
            
            # Store in database
            cursor.execute('''
                INSERT OR REPLACE INTO rs_reference_nifty500 
                (date, ticker, rs_raw, r1, r3, r6)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (today, ticker, rs_raw, r1, r3, r6))
            success_count += 1
        except Exception:
            pass
            
    conn.commit()
    conn.close()
    
    if show_progress:
        print(f"  [OK] Computed RS for {success_count}/{total} NIFTY 500 stocks")
    
    return success_count >= 100


def get_percentile_rank(value: float, distribution: list) -> float:
    """
    Calculate percentile rank of value against distribution.
    
    Args:
        value: The RS_raw value to rank
        distribution: Sorted list of RS_raw values from NIFTY 500
    
    Returns:
        Percentile rank (0-100)
    """
    if not distribution or value is None:
        return None
    
    # Count how many values in distribution are less than our value
    count_below = sum(1 for v in distribution if v < value)
    
    # Percentile = (count below / total) * 100
    percentile = (count_below / len(distribution)) * 100
    
    return round(percentile, 1)


if __name__ == '__main__':
    print("Testing NIFTY 500 Universe module...")
    compute_nifty500_rs_reference(show_progress=True)
    
    distribution = get_rs_reference_distribution()
    print(f"\nDistribution size: {len(distribution)}")
    if distribution:
        print(f"Min RS_raw: {min(distribution):.4f}")
        print(f"Max RS_raw: {max(distribution):.4f}")
        print(f"Median RS_raw: {distribution[len(distribution)//2]:.4f}")

get_nifty500_tickers = fetch_nifty500_tickers

