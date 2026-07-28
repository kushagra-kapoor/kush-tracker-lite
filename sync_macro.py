import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime
from database import init_database, save_global_regime, save_global_etf_momentum
from market_data import fetch_nifty_total_market_tickers
from price_history_manager import fetch_incremental_history
from liquidity_engine import compute_liquidity
from breadth_engine import sync_breadth_history
from macro_regime_engine import (
    standardize_columns, calculate_distribution_days, 
    detect_follow_through_day, detect_change_of_character, get_market_regime_label
)

def get_tickers_from_file(filename):
    if not os.path.exists(filename):
        return []
    with open(filename, 'r') as f:
        return [line.strip().upper() for line in f if line.strip()]

def sync_all_data():
    print("Starting Global Macro Sync Engine...")
    
    # 1. Initialize Database
    init_database()
    
    # 2. Gather All Universes
    print("Gathering ticker universes...")
    try:
        in_tickers = fetch_nifty_total_market_tickers(show_progress=False)
        in_tickers = [t + '.NS' if not t.endswith('.NS') else t for t in in_tickers]
    except Exception as e:
        print(f"Failed to fetch IN tickers: {e}")
        in_tickers = get_tickers_from_file('tickers.txt')
        in_tickers = [t + '.NS' if not t.endswith('.NS') else t for t in in_tickers]
        
    us_tickers = get_tickers_from_file('tickers_us.txt')
    us_etfs = get_tickers_from_file('tickers_us_etf.txt')
    in_etfs = get_tickers_from_file('tickers_etf.txt')
    in_etfs = [t + '.NS' if not t.endswith('.NS') else t for t in in_etfs]
    
    all_tickers = list(set(in_tickers + us_tickers + us_etfs + in_etfs))
    
    # 3. Fetch Matrix History
    print(f"Fetching history for {len(all_tickers)} global assets...")
    def print_prog(msg):
        print(f"  > {msg}")
    fetch_incremental_history(all_tickers, days=252, progress_callback=print_prog, force_today_refresh=True)
    
    # 4. Update Engine Tables
    print("Updating Breadth and Liquidity for IN Market...")
    compute_liquidity(market='IN')
    sync_breadth_history(market='IN')
    
    print("Updating Breadth and Liquidity for US Market...")
    compute_liquidity(market='US')
    sync_breadth_history(market='US')
    
    # 5. Global Benchmark Regime
    print("Calculating Global Regimes...")
    regime_data = []
    # Using Nifty 500 for India and S&P 500 for US
    for market, benchmark in [('IN', '^CRSLDX'), ('US', '^GSPC')]:
        try:
            idx = yf.download(benchmark, period="250d", progress=False)
            idx_std = standardize_columns(idx)
            if not idx_std.empty and 'close' in idx_std.columns:
                dd_count, _ = calculate_distribution_days(idx)
                ftd_detected, _ = detect_follow_through_day(idx)
                choch = detect_change_of_character(idx)
                
                curr = idx_std['close'].iloc[-1]
                sma50 = idx_std['close'].rolling(50).mean().iloc[-1]
                sma200 = idx_std['close'].rolling(200).mean().iloc[-1] if len(idx_std) >= 200 else curr
                regime_label, _, _ = get_market_regime_label(dd_count, curr, sma50)
                choch_label = choch['label'] if choch and choch['detected'] else "None"
                
                date_val = idx_std.index[-1].strftime('%Y-%m-%d')
                
                regime_data.append({
                    'date': date_val,
                    'market': market,
                    'benchmark_ticker': benchmark,
                    'close': float(curr),
                    'sma50': float(sma50),
                    'sma200': float(sma200),
                    'dd_count': int(dd_count),
                    'ftd_detected': bool(ftd_detected),
                    'regime_label': regime_label,
                    'choch_label': choch_label
                })
        except Exception as e:
            print(f"Failed regime calc for {benchmark}: {e}")
            
    if regime_data:
        save_global_regime(regime_data)
        
    # 6. Global ETF Momentum
    print("Calculating ETF Momentum...")
    if os.path.exists('historical_prices_matrix.pkl'):
        df = pd.read_pickle('historical_prices_matrix.pkl')
        if not df.empty:
            close_prices = df.xs('Close', level=1, axis=1)
            etf_results = []
            
            for etf in set(us_etfs + in_etfs):
                if etf in close_prices.columns:
                    series = close_prices[etf].dropna()
                    if len(series) > 0:
                        r1 = (series.iloc[-1] / series.iloc[-21]) - 1 if len(series) > 21 else 0
                        r3 = (series.iloc[-1] / series.iloc[-63]) - 1 if len(series) > 63 else 0
                        r6 = (series.iloc[-1] / series.iloc[-126]) - 1 if len(series) > 126 else 0
                        
                        etf_results.append({
                            'date': series.index[-1].strftime('%Y-%m-%d'),
                            'ticker': etf,
                            'name': etf.replace('.NS', ''),
                            'return_1m': float(r1) * 100,
                            'return_3m': float(r3) * 100,
                            'return_6m': float(r6) * 100
                        })
            if etf_results:
                save_global_etf_momentum(etf_results)
                
    print("Global Macro Sync Completed!")

if __name__ == "__main__":
    sync_all_data()
