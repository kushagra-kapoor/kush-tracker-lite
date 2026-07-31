import sqlite3
import yfinance as yf
import concurrent.futures
import time
import random
from datetime import datetime, timedelta
import sys
import os

# Ensure we can import from database.py in the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database import get_connection

def save_fundamentals_cache_batch(data_list: list):
    """
    Batch save successful fundamentals to bypass Yahoo Finance rate limits securely.
    Args:
        data_list: List of dicts [{'ticker': 'AAPL', 'eps_growth': 10.0, ...}]
    """
    if not data_list:
        return
        
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        for item in data_list:
            cursor.execute('''
                INSERT OR REPLACE INTO fundamentals_cache 
                (ticker, eps_yoy, sales_yoy, roe, industry, market_cap, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                item['ticker'],
                item['eps_growth'],
                item['sales_growth'],
                item['roe'],
                item['industry'],
                item['market_cap']
            ))
        conn.commit()
    except Exception as e:
        print(f"Error saving fundamentals batch cache: {e}")
    finally:
        conn.close()


def get_stale_tickers(all_tickers: list) -> list:
    """
    Returns tickers that are either not in the cache OR whose updated_at is older than 24 hours.
    This provides resumability if the script crashes midway.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT ticker, updated_at FROM fundamentals_cache")
        rows = cursor.fetchall()
        cache = {r['ticker']: r['updated_at'] for r in rows}
    except Exception as e:
        print(f"Error reading fundamentals cache: {e}")
        cache = {}
    finally:
        conn.close()

    stale_tickers = []
    now = datetime.now()
    
    for ticker in all_tickers:
        if ticker not in cache:
            stale_tickers.append(ticker)
        else:
            updated_at_str = cache[ticker]
            if updated_at_str:
                try:
                    last_updated = datetime.strptime(updated_at_str, '%Y-%m-%d %H:%M:%S')
                    # If updated more than 7 days (168 hours) ago, consider it stale
                    if (now - last_updated).total_seconds() > (7 * 24 * 3600):
                        stale_tickers.append(ticker)
                except Exception:
                    stale_tickers.append(ticker)
            else:
                stale_tickers.append(ticker)
                
    return stale_tickers


def fetch_full_fundamentals(ticker, max_retries=3):
    """Fetch EPS growth, Sales growth, ROE, Industry, and Market Cap with retry logic for rate limits."""
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(ticker)
            eps_g = 0.0
            sales_g = 0.0
            roe = 0.0
            mcap = 0.0
            industry = "Unknown"
            
            info = t.info
            if info:
                industry = info.get('industry', 'Unknown')
                mcap = info.get('marketCap', 0.0)
                roe = info.get('returnOnEquity', 0)
                if roe is not None: roe *= 100
                else: roe = 0.0
                    
                rev_g = info.get('revenueGrowth', 0)
                if rev_g is not None: sales_g = rev_g * 100
                    
                ern_g = info.get('earningsGrowth', 0)
                if ern_g is not None: eps_g = ern_g * 100

            if eps_g == 0 and sales_g == 0:
                fins = t.quarterly_financials
                if fins is not None and not fins.empty and len(fins.columns) >= 5:
                    if 'Net Income' in fins.index:
                        curr_ni = fins.loc['Net Income'].iloc[0]
                        prev_ni = fins.loc['Net Income'].iloc[4]
                        if prev_ni and prev_ni > 0:
                            eps_g = ((curr_ni / prev_ni) - 1) * 100
                            
                    rev_keys = ['Total Revenue', 'Operating Revenue', 'Revenue']
                    rel_row = next((r for r in rev_keys if r in fins.index), None)
                    if rel_row:
                        curr_rev = fins.loc[rel_row].iloc[0]
                        prev_rev = fins.loc[rel_row].iloc[4]
                        if prev_rev and prev_rev > 0:
                            sales_g = ((curr_rev / prev_rev) - 1) * 100
                            
            if roe == 0:
                bal = t.quarterly_balance_sheet
                fins = t.quarterly_financials
                if bal is not None and not bal.empty and fins is not None and not fins.empty:
                    eq_keys = ['Common Stock Equity', 'Stockholders Equity', 'Total Equity Gross Minority Interest']
                    eq_row = next((r for r in eq_keys if r in bal.index), None)
                    if eq_row and 'Net Income' in fins.index:
                        ni_series = fins.loc['Net Income'].dropna()
                        eq_series = bal.loc[eq_row].dropna()
                        if len(ni_series) > 0 and len(eq_series) > 0:
                            if len(ni_series) >= 4:
                                trailing_ni = ni_series.iloc[:4].sum()
                            else:
                                trailing_ni = ni_series.iloc[0] * 4
                                
                            latest_eq = eq_series.iloc[0]
                            if latest_eq > 0:
                                roe = (trailing_ni / latest_eq) * 100
                                
            # Anti-Poisoning Check: Check if yfinance totally failed silently without throwing an exception (shadowban protection)
            if eps_g == 0 and sales_g == 0 and roe == 0 and industry == "Unknown" and mcap == 0:
                if attempt < max_retries - 1:
                    wait_time = random.uniform(2.0, 4.0) * (attempt + 1)
                    time.sleep(wait_time)
                    continue
                else:
                    return None
                    
            return {
                'ticker': ticker,
                'eps_growth': float(eps_g) if eps_g is not None else 0.0,
                'sales_growth': float(sales_g) if sales_g is not None else 0.0,
                'roe': float(roe) if roe is not None else 0.0,
                'market_cap': float(mcap) if mcap is not None else 0.0,
                'industry': industry
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = random.uniform(2.0, 4.0) * (attempt + 1)
                time.sleep(wait_time)
                continue
            return None


def update_all_fundamentals():
    print("Starting Cloud Fundamentals Sync Engine...")
    start_time = time.time()
    
    # 1. Load all tickers
    all_tickers = []
    try:
        with open('tickers.txt', 'r') as f:
            all_tickers.extend([line.strip().upper() for line in f if line.strip() and '-' not in line])
    except Exception as e:
        print(f"Warning: tickers.txt not found or error reading: {e}")
        
    try:
        with open('tickers_us.txt', 'r') as f:
            all_tickers.extend([line.strip().upper() for line in f if line.strip() and '-' not in line])
    except Exception as e:
        print(f"Warning: tickers_us.txt not found or error reading: {e}")
        
    all_tickers = list(set(all_tickers))
    print(f"Total Unique Tickers in Universe: {len(all_tickers)}")
    
    # 2. Resumability Check: Filter out already updated tickers
    tickers = get_stale_tickers(all_tickers)
    print(f"Target: {len(tickers)} stale tickers require fetching.")
    
    if not tickers:
        print("All tickers are fully up to date. Exiting.")
        return
        
    print("Fetching Fundamentals via YFinance (Rate-Limit Safe ThreadPool)...")
    results = []
    failed = 0
    completed = 0
    
    # Use max_workers=2 to prevent triggering rapid Yahoo Finance shadow bans 
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_ticker = {executor.submit(fetch_full_fundamentals, ticker): ticker for ticker in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            data = future.result()
            if data is not None:
                results.append(data)
                # Batch save every 20 successful fetches to ensure progress is not lost
                if len(results) % 20 == 0:
                    save_fundamentals_cache_batch(results[-20:])
            else:
                failed += 1
                
            completed += 1
            if completed % 10 == 0:
                print(f"   Progress: {completed}/{len(tickers)} | Success: {len(results)} | Failed: {failed}")
                
    # Save any remaining results
    if len(results) % 20 != 0 and len(results) > 0:
        remainder = len(results) % 20
        save_fundamentals_cache_batch(results[-remainder:])
        
    elapsed = time.time() - start_time
    
    print("\n==========================================")
    print("            FINAL STATISTICS              ")
    print("==========================================")
    print(f" Total Universe Target: {len(all_tickers)}")
    print(f" Actually Fetched:      {len(tickers)}")
    print(f" Successfully Fetched:  {len(results)}")
    print(f" Successfully Updated:  {len(results)} (Upserted into DB)")
    print(f" Stale Data Remaining:  {failed} (Failed to fetch/Delisted)")
    print(f" Total Time Taken:      {elapsed:.1f} seconds")
    print("==========================================\n")

if __name__ == "__main__":
    update_all_fundamentals()
