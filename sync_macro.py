import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime
from database import (
    init_database, save_global_regime, 
    save_global_etf_momentum, save_global_sovereign_returns
)
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
    all_etfs = list(set(us_etfs + in_etfs))
    if all_etfs:
        try:
            data = yf.download(all_etfs, period="1y", progress=False)
            close_prices = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
            etf_results = []
            for etf in all_etfs:
                if etf in close_prices.columns:
                    series = close_prices[etf].dropna()
                    if len(series) >= 20:
                        curr_close = float(series.iloc[-1])
                        high_52w = float(series.max())
                        dist_52w_high = float(((curr_close / high_52w) - 1.0) * 100.0) if high_52w > 0 else 0.0

                        r1 = (curr_close / series.iloc[-21]) - 1 if len(series) > 21 else 0
                        r3 = (curr_close / series.iloc[-63]) - 1 if len(series) > 63 else 0
                        r6 = (curr_close / series.iloc[-126]) - 1 if len(series) > 126 else 0
                        
                        etf_results.append({
                            'date': series.index[-1].strftime('%Y-%m-%d'),
                            'ticker': etf,
                            'name': etf.replace('.NS', ''),
                            'return_1m': float(r1) * 100,
                            'return_3m': float(r3) * 100,
                            'return_6m': float(r6) * 100,
                            'close': curr_close,
                            'high_52w': high_52w,
                            'dist_52w_high': dist_52w_high
                        })
            if etf_results:
                save_global_etf_momentum(etf_results)
                print(f"Saved momentum for {len(etf_results)} global ETFs.")
        except Exception as e:
            print(f"Failed ETF momentum calc: {e}")
                
    # 7. Update NIFTY 500 RS Reference
    print("Computing NIFTY 500 RS Reference...")
    try:
        from nifty500_universe import compute_nifty500_rs_reference
        compute_nifty500_rs_reference(show_progress=True)
    except Exception as e:
        print(f"Failed to compute RS Reference: {e}")

    # 8. Global Sovereign Normalized USD Returns
    print("Calculating Global Sovereign Normalized USD Returns...")
    sync_global_sovereign_returns()
                
    print("Global Macro Sync Completed!")

SOVEREIGN_MARKETS = [
    {'country': 'United States', 'flag': '🇺🇸', 'index_ticker': '^GSPC', 'etf_ticker': 'SPY', 'index_name': 'S&P 500 Core', 'currency': 'USD', 'fx_ticker': None, 'inverse_fx': False},
    {'country': 'United States', 'flag': '🇺🇸', 'index_ticker': '^NDX', 'etf_ticker': 'QQQ', 'index_name': 'Nasdaq 100 Growth', 'currency': 'USD', 'fx_ticker': None, 'inverse_fx': False},
    {'country': 'India', 'flag': '🇮🇳', 'index_ticker': '^NSEI', 'etf_ticker': 'INDY', 'index_name': 'NIFTY 50 LargeCap', 'currency': 'INR', 'fx_ticker': 'USDINR=X', 'inverse_fx': True},
    {'country': 'India', 'flag': '🇮🇳', 'index_ticker': '^CRSLDX', 'etf_ticker': 'INDA', 'index_name': 'NIFTY 500 BroadMarket', 'currency': 'INR', 'fx_ticker': 'USDINR=X', 'inverse_fx': True},
    {'country': 'Japan', 'flag': '🇯🇵', 'index_ticker': '^N225', 'etf_ticker': 'EWJ', 'index_name': 'Nikkei 225', 'currency': 'JPY', 'fx_ticker': 'JPY=X', 'inverse_fx': True},
    {'country': 'Germany', 'flag': '🇩🇪', 'index_ticker': '^GDAXI', 'etf_ticker': 'EWG', 'index_name': 'DAX 40', 'currency': 'EUR', 'fx_ticker': 'EURUSD=X', 'inverse_fx': False},
    {'country': 'United Kingdom', 'flag': '🇬🇧', 'index_ticker': '^FTSE', 'etf_ticker': 'EWU', 'index_name': 'FTSE 100', 'currency': 'GBP', 'fx_ticker': 'GBPUSD=X', 'inverse_fx': False},
    {'country': 'China', 'flag': '🇨🇳', 'index_ticker': '^HSI', 'etf_ticker': 'MCHI', 'index_name': 'Hang Seng Index', 'currency': 'HKD', 'fx_ticker': 'HKD=X', 'inverse_fx': True},
    {'country': 'France', 'flag': '🇫🇷', 'index_ticker': '^FCHI', 'etf_ticker': 'EWQ', 'index_name': 'CAC 40', 'currency': 'EUR', 'fx_ticker': 'EURUSD=X', 'inverse_fx': False},
    {'country': 'Australia', 'flag': '🇦🇺', 'index_ticker': '^AXJO', 'etf_ticker': 'EWA', 'index_name': 'S&P/ASX 200', 'currency': 'AUD', 'fx_ticker': 'AUDUSD=X', 'inverse_fx': False},
    {'country': 'Canada', 'flag': '🇨🇦', 'index_ticker': '^GSPTSE', 'etf_ticker': 'EWC', 'index_name': 'S&P/TSX Composite', 'currency': 'CAD', 'fx_ticker': 'CAD=X', 'inverse_fx': True},
    {'country': 'Brazil', 'flag': '🇧🇷', 'index_ticker': '^BVSP', 'etf_ticker': 'EWZ', 'index_name': 'Ibovespa', 'currency': 'BRL', 'fx_ticker': 'BRL=X', 'inverse_fx': True},
    {'country': 'South Korea', 'flag': '🇰🇷', 'index_ticker': '^KS11', 'etf_ticker': 'EWY', 'index_name': 'KOSPI', 'currency': 'KRW', 'fx_ticker': 'KRW=X', 'inverse_fx': True},
    {'country': 'Taiwan', 'flag': '🇹🇼', 'index_ticker': '^TWII', 'etf_ticker': 'EWT', 'index_name': 'TAIEX', 'currency': 'TWD', 'fx_ticker': 'TWD=X', 'inverse_fx': True},
]

def sync_global_sovereign_returns():
    """Fetches global sovereign indices and FX pairs, normalizing returns into USD across 1M, 3M, 6M, 1Y."""
    init_database()
    idx_tickers = [m['index_ticker'] for m in SOVEREIGN_MARKETS]
    fx_tickers = [m['fx_ticker'] for m in SOVEREIGN_MARKETS if m['fx_ticker']]
    all_syms = list(set(idx_tickers + fx_tickers))
    
    try:
        data = yf.download(all_syms, period="15mo", progress=False)
        close_prices = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
        close_prices = close_prices.ffill().bfill()
        
        parsed_markets = []
        sp500_usd_returns = {'1m': 0.0, '3m': 0.0, '6m': 0.0, '1y': 0.0}
        
        for m in SOVEREIGN_MARKETS:
            t = m['index_ticker']
            if t not in close_prices.columns:
                continue
            idx_series = close_prices[t].dropna()
            if len(idx_series) < 25:
                continue
                
            fx_sym = m['fx_ticker']
            if fx_sym and fx_sym in close_prices.columns:
                fx_raw = close_prices[fx_sym].reindex(idx_series.index).ffill().bfill()
                if m['inverse_fx']:
                    fx_mult = 1.0 / fx_raw
                else:
                    fx_mult = fx_raw
                fx_latest = float(fx_raw.iloc[-1])
            else:
                fx_mult = pd.Series(1.0, index=idx_series.index)
                fx_latest = 1.0
                
            usd_series = idx_series * fx_mult
            curr_usd = float(usd_series.iloc[-1])
            curr_local = float(idx_series.iloc[-1])
            high_52w_usd = float(usd_series.iloc[-252:].max()) if len(usd_series) >= 252 else float(usd_series.max())
            dist_52w = ((curr_usd / high_52w_usd) - 1.0) * 100.0 if high_52w_usd > 0 else 0.0
            
            def calc_metrics(h_offset):
                target_dt = idx_series.index[-1] - h_offset
                past_idx = idx_series.loc[:target_dt]
                if len(past_idx) > 0:
                    past_local = float(past_idx.iloc[-1])
                    past_mult = float(fx_mult.loc[:target_dt].iloc[-1])
                    past_usd = float(usd_series.loc[:target_dt].iloc[-1])
                    ret_local = ((curr_local / past_local) - 1.0) * 100.0
                    ret_fx = ((float(fx_mult.iloc[-1]) / past_mult) - 1.0) * 100.0
                    ret_usd = ((curr_usd / past_usd) - 1.0) * 100.0
                else:
                    ret_local, ret_fx, ret_usd = 0.0, 0.0, 0.0
                return ret_local, ret_fx, ret_usd
                
            r1_loc, r1_fx, r1_usd = calc_metrics(pd.DateOffset(months=1))
            r3_loc, r3_fx, r3_usd = calc_metrics(pd.DateOffset(months=3))
            r6_loc, r6_fx, r6_usd = calc_metrics(pd.DateOffset(months=6))
            ry_loc, ry_fx, ry_usd = calc_metrics(pd.DateOffset(years=1))
            
            if t == '^GSPC':
                sp500_usd_returns = {'1m': r1_usd, '3m': r3_usd, '6m': r6_usd, '1y': ry_usd}
                
            date_str = idx_series.index[-1].strftime('%Y-%m-%d')
            parsed_markets.append({
                'date': date_str,
                'country': m['country'],
                'index_ticker': t,
                'etf_ticker': m.get('etf_ticker', ''),
                'index_name': m['index_name'],
                'currency': m['currency'],
                'fx_ticker': fx_sym if fx_sym else 'USD',
                'local_close': curr_local,
                'fx_rate': fx_latest,
                'return_1m_local': r1_loc,
                'return_1m_fx': r1_fx,
                'return_1m_usd': r1_usd,
                'return_3m_local': r3_loc,
                'return_3m_fx': r3_fx,
                'return_3m_usd': r3_usd,
                'return_6m_local': r6_loc,
                'return_6m_fx': r6_fx,
                'return_6m_usd': r6_usd,
                'return_1y_local': ry_loc,
                'return_1y_fx': ry_fx,
                'return_1y_usd': ry_usd,
                'dist_52w_high': dist_52w
            })
            
        # Calculate USD Alpha vs S&P 500
        for p in parsed_markets:
            p['alpha_1m_usd'] = p['return_1m_usd'] - sp500_usd_returns['1m']
            p['alpha_3m_usd'] = p['return_3m_usd'] - sp500_usd_returns['3m']
            p['alpha_6m_usd'] = p['return_6m_usd'] - sp500_usd_returns['6m']
            p['alpha_1y_usd'] = p['return_1y_usd'] - sp500_usd_returns['1y']
            
        if parsed_markets:
            save_global_sovereign_returns(parsed_markets)
            print(f"[Macro Engine] Successfully saved {len(parsed_markets)} sovereign USD return records.")
        return parsed_markets
    except Exception as e:
        print(f"[Macro Engine] Failed sovereign USD calculation: {e}")
        return []

if __name__ == "__main__":
    sync_all_data()
