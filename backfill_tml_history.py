import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3

# Add root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import get_connection, save_tml_snapshot, get_all_fundamentals_cache
from views.intraday_monitor import fetch_yfinance_batch

# India imports
from views.true_market_leader import (
    get_cached_universe as get_india_universe,
    compute_rs_scores_fast as compute_rs_india,
    run_technical_prescreen as prescreen_india,
    score_leaders as score_india
)

# US imports
from views.true_market_leader_us import (
    get_cached_universe as get_us_universe,
    compute_rs_scores_fast as compute_rs_us,
    run_technical_prescreen as prescreen_us,
    score_leaders as score_us
)

def backfill_market(market, universe_mode, asset_class, get_universe_fn, compute_rs_fn, prescreen_fn, score_fn):
    print(f"\n--- Backfilling TML History for {market} ---")
    
    # 1. Get dates that already exist
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT date FROM tml_snapshot WHERE market = ?", (market,))
    existing_dates = set([row[0] for row in cursor.fetchall()])
    conn.close()
    
    # 2. Get Universe
    if market == 'INDIA':
        tickers = get_universe_fn(universe_mode)
    else:
        tickers = get_universe_fn(mode=asset_class)
        
    print(f"Loaded {len(tickers)} tickers for {market}.")
    
    # 3. Fetch large history block once (350 days so we can look back 90 days and still have 250 days of history)
    print("Fetching historical price matrix (this may take a minute)...")
    full_history_df = fetch_yfinance_batch(tickers, days=350)
    
    if full_history_df.empty or full_history_df.columns.nlevels < 2:
        print("Failed to fetch historical data.")
        return
        
    # Get all distinct trading dates from the history
    try:
        all_dates = full_history_df.xs('Close', level=1, axis=1).index
        # Convert to YYYY-MM-DD strings
        date_strs = [d.strftime('%Y-%m-%d') for d in all_dates]
    except Exception as e:
        print(f"Error parsing dates: {e}")
        return
        
    # We want to backfill the last 90 trading days
    target_dates = date_strs[-90:]
    
    # 4. Get Fundamentals Cache
    db_cache = get_all_fundamentals_cache()
    
    # 5. Iterate through target dates
    for i, target_date_str in enumerate(target_dates):
        if target_date_str in existing_dates:
            print(f"[{target_date_str}] Already exists. Skipping.")
            continue
            
        print(f"[{target_date_str}] Backfilling...")
        
        # Slice history up to this date
        target_ts = pd.to_datetime(target_date_str)
        # We need to make sure we slice properly. The index might be tz-aware.
        # Let's just slice by integer index for safety if we know the positions.
        # target_dates is just the last 90 of date_strs.
        # If target_date_str is the Nth from the end, we slice up to there.
        idx_pos = date_strs.index(target_date_str)
        sliced_df = full_history_df.iloc[:idx_pos + 1]
        
        # Run TML logic on sliced df
        rs_scores = compute_rs_fn(sliced_df, tickers)
        pre_screened, funnel = prescreen_fn(sliced_df, tickers, rs_scores, is_etf=False)
        
        # Inject fundamentals from cache (simulate fetch_fundamentals)
        final_leaders = []
        for stock in pre_screened:
            ticker = stock['ticker']
            funds = db_cache.get(ticker, {})
            
            eps_g = funds.get('eps_growth', 0.0)
            sales_g = funds.get('sales_growth', 0.0)
            roe = funds.get('roe', 0.0)
            industry = funds.get('industry', 'Unknown')
            mcap = funds.get('market_cap', 0.0)
            
            if market == 'INDIA':
                mcap_val = (mcap / 10000000.0) if mcap else 0.0
                fund_dict = {'eps_growth': eps_g, 'sales_growth': sales_g, 'roe': roe, 'mcap_cr': mcap_val, 'industry': industry}
            else:
                mcap_val = (mcap / 1000000000.0) if mcap else 0.0
                fund_dict = {'eps_growth': eps_g, 'sales_growth': sales_g, 'roe': roe, 'mcap_b': mcap_val, 'industry': industry}
                
            final_leaders.append({**stock, **fund_dict})
            
        final_leaders = score_fn(final_leaders, is_etf=False)
        
        if final_leaders:
            # Save using custom SQL to insert with specific date (save_tml_snapshot uses today's date)
            conn = get_connection()
            cursor = conn.cursor()
            sorted_leaders = sorted(final_leaders, key=lambda x: x.get('tml_score', 0), reverse=True)
            top_leaders = sorted_leaders[:20]
            
            for rank, leader in enumerate(top_leaders):
                ticker = leader.get('ticker', '')
                tml_score = leader.get('tml_score', 0.0)
                rs_score = leader.get('rs_score', 0.0)
                action_status = leader.get('Action_Status', 'Unknown')
                industry = leader.get('industry', 'Unknown')
                
                cursor.execute('''
                    INSERT OR REPLACE INTO tml_snapshot 
                    (date, market, rank, ticker, tml_score, rs_score, action_status, industry)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (target_date_str, market, rank + 1, ticker, tml_score, rs_score, action_status, industry))
                
            conn.commit()
            conn.close()
            print(f"  -> Saved {len(top_leaders)} leaders.")
        else:
            print("  -> No leaders found.")

if __name__ == "__main__":
    print("Starting TML Backfill Process...")
    
    # Backfill India (Deep Market)
    backfill_market('INDIA', 'Deep Market (2500+ NSE Stocks)', 'Stocks', get_india_universe, compute_rs_india, prescreen_india, score_india)
    
    # Backfill US (Stocks)
    backfill_market('US', None, 'Stocks', get_us_universe, compute_rs_us, prescreen_us, score_us)
    
    print("\nBackfill Complete!")

