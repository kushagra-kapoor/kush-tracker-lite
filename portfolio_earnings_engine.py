# portfolio_earnings_engine.py
# Engine for fetching, caching, and analyzing portfolio company quarterly earnings & sales results

import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

CACHE_DIR = '.cache'
CACHE_FILE = os.path.join(CACHE_DIR, 'portfolio_earnings.json')

ETF_KEYWORDS = ['BEES', 'ETF', 'GOLD', 'SILVER', 'LIQUID', 'MON100', 'MAFANG', 'ALPHA', 'MOMENTUM', 'MOM50', 'MOM30', 'HDFCBANK-P']


def _ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)


def is_etf_ticker(ticker: str) -> bool:
    t_upper = ticker.upper()
    return any(k in t_upper for k in ETF_KEYWORDS)


def load_earnings_cache() -> dict:
    """Load cached portfolio earnings results."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Earnings Engine] Error loading cache: {e}")
    return {}


def save_earnings_cache(cache_data: dict):
    """Save portfolio earnings results to cache."""
    _ensure_cache_dir()
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"[Earnings Engine] Saved {len(cache_data)} portfolio tickers to cache.")
    except Exception as e:
        print(f"[Earnings Engine] Error saving cache: {e}")


def _find_yoy_prior_quarter_idx(dates: list, current_date) -> int:
    """Find column index in quarterly_financials representing the same quarter ~1 year prior (330-395 days)."""
    curr_dt = pd.to_datetime(current_date)
    for j, d in enumerate(dates):
        dt = pd.to_datetime(d)
        diff_days = (curr_dt - dt).days
        if 330 <= diff_days <= 395:
            return j
    return None


def fetch_ticker_quarterly_data(ticker: str) -> dict:
    """
    Fetch quarterly EPS, Sales, OPM, and calendar data for a single ticker via yfinance (Tier 3 fallback)
    or local database cache.
    """
    clean_ticker = ticker.replace('.NS', '').replace('.BO', '').strip()
    
    # 1. Handle ETF / Non-Reporting Assets
    if is_etf_ticker(clean_ticker):
        return {
            'ticker': clean_ticker,
            'yf_symbol': clean_ticker,
            'latest_quarter': 'ETF',
            'reported_date': 'N/A',
            'eps_yoy_pct': None,
            'sales_yoy_pct': None,
            'opm_pct': None,
            'eps_trajectory': [],
            'sales_trajectory': [],
            'quarter_labels': [],
            'is_new': False,
            'is_etf': True,
            'upcoming_earnings_date': 'N/A',
            'days_to_earnings': None,
            'dual_acceleration': False,
            'verdict': '💵 ETF / Gold',
        }

    if clean_ticker.isdigit():
        yf_symbol = f"{clean_ticker}.BO"
    else:
        yf_symbol = f"{clean_ticker}.NS" if not (clean_ticker.endswith('.NS') or clean_ticker.endswith('.BO')) else clean_ticker

    record = {
        'ticker': clean_ticker,
        'yf_symbol': yf_symbol,
        'latest_quarter': 'N/A',
        'reported_date': 'N/A',
        'eps_yoy_pct': None,
        'sales_yoy_pct': None,
        'opm_pct': None,
        'eps_trajectory': [],     # List of dicts: [{'type': 'YoY'|'QoQ'|'RAW', 'val': float, 'label': str}]
        'sales_trajectory': [],   # List of dicts: [{'type': 'YoY'|'QoQ'|'RAW', 'val': float, 'label': str}]
        'quarter_labels': [],     # e.g. ["Jun 25", "Dec 25", "Mar 26", "Jun 26"]
        'is_new': False,
        'is_etf': False,
        'upcoming_earnings_date': 'N/A',
        'days_to_earnings': None,
        'dual_acceleration': False,
        'verdict': 'Neutral',
    }

    try:
        t_obj = yf.Ticker(yf_symbol)
        
        # Check Calendar for upcoming earnings date
        try:
            cal = t_obj.calendar
            if cal is not None and not (isinstance(cal, pd.DataFrame) and cal.empty):
                if isinstance(cal, dict) and 'Earnings Date' in cal:
                    ed_list = cal['Earnings Date']
                    if ed_list:
                        next_ed = pd.to_datetime(ed_list[0])
                        record['upcoming_earnings_date'] = next_ed.strftime('%Y-%m-%d')
                        days_left = (next_ed.date() - datetime.now().date()).days
                        record['days_to_earnings'] = days_left
                elif isinstance(cal, pd.DataFrame) and 'Earnings Date' in cal.index:
                    ed_list = cal.loc['Earnings Date'].values
                    if len(ed_list) > 0:
                        next_ed = pd.to_datetime(ed_list[0])
                        record['upcoming_earnings_date'] = next_ed.strftime('%Y-%m-%d')
                        days_left = (next_ed.date() - datetime.now().date()).days
                        record['days_to_earnings'] = days_left
        except Exception:
            pass

        # Fetch quarterly financials
        inc = t_obj.quarterly_financials
        if inc is not None and not inc.empty and len(inc.columns) >= 2:
            dates = list(inc.columns)
            
            rev_row = None
            for idx in ['Total Revenue', 'Revenue', 'Operating Revenue']:
                if idx in inc.index:
                    rev_row = inc.loc[idx]
                    break
                    
            eps_row = None
            for idx in ['Diluted EPS', 'Basic EPS', 'Net Income']:
                if idx in inc.index:
                    eps_row = inc.loc[idx]
                    break

            if rev_row is not None and len(dates) >= 2:
                latest_date = dates[0]
                record['latest_quarter'] = pd.to_datetime(latest_date).strftime('%b %Y')
                record['reported_date'] = pd.to_datetime(latest_date).strftime('%Y-%m-%d')
                
                # Check if reported within last 7 days
                days_since_report = (datetime.now().date() - pd.to_datetime(latest_date).date()).days
                if 0 <= days_since_report <= 7:
                    record['is_new'] = True
                
                # Compute YoY / QoQ hybrid trajectory across available quarters
                eps_traj = []
                sales_traj = []
                q_labels = []
                
                num_q = min(4, len(dates))
                for i in range(num_q - 1, -1, -1):
                    current_q_date = dates[i]
                    q_lbl = pd.to_datetime(current_q_date).strftime('%b %y')
                    q_labels.append(q_lbl)
                    
                    prior_yr_idx = _find_yoy_prior_quarter_idx(dates, current_q_date)
                    
                    # Sales metric
                    curr_sales = rev_row.iloc[i]
                    if prior_yr_idx is not None and prior_yr_idx < len(dates):
                        prev_sales = rev_row.iloc[prior_yr_idx]
                        if prev_sales and prev_sales > 0 and pd.notna(curr_sales):
                            s_yoy = ((curr_sales - prev_sales) / abs(prev_sales)) * 100.0
                            sales_traj.append({'type': 'YoY', 'val': round(s_yoy, 1), 'label': f"+{s_yoy:.1f}% YoY" if s_yoy > 0 else f"{s_yoy:.1f}% YoY"})
                        else:
                            sales_traj.append({'type': 'RAW', 'val': None, 'label': '-'})
                    elif i + 1 < len(dates):
                        prev_sales = rev_row.iloc[i + 1]
                        if prev_sales and prev_sales > 0 and pd.notna(curr_sales):
                            s_qoq = ((curr_sales - prev_sales) / abs(prev_sales)) * 100.0
                            sales_traj.append({'type': 'QoQ', 'val': round(s_qoq, 1), 'label': f"+{s_qoq:.1f}% QoQ" if s_qoq > 0 else f"{s_qoq:.1f}% QoQ"})
                        else:
                            sales_traj.append({'type': 'RAW', 'val': None, 'label': '-'})
                    else:
                        sales_traj.append({'type': 'RAW', 'val': round(curr_sales / 1e7, 1) if pd.notna(curr_sales) else None, 'label': f"₹{curr_sales/1e7:.1f}Cr" if pd.notna(curr_sales) else '-'})

                    # EPS metric
                    if eps_row is not None:
                        curr_eps = eps_row.iloc[i]
                        if prior_yr_idx is not None and prior_yr_idx < len(dates):
                            prev_eps = eps_row.iloc[prior_yr_idx]
                            if prev_eps and prev_eps != 0 and pd.notna(curr_eps):
                                e_yoy = ((curr_eps - prev_eps) / abs(prev_eps)) * 100.0
                                eps_traj.append({'type': 'YoY', 'val': round(e_yoy, 1), 'label': f"+{e_yoy:.1f}% YoY" if e_yoy > 0 else f"{e_yoy:.1f}% YoY"})
                            else:
                                eps_traj.append({'type': 'RAW', 'val': None, 'label': '-'})
                        elif i + 1 < len(dates):
                            prev_eps = eps_row.iloc[i + 1]
                            if prev_eps and prev_eps != 0 and pd.notna(curr_eps):
                                e_qoq = ((curr_eps - prev_eps) / abs(prev_eps)) * 100.0
                                eps_traj.append({'type': 'QoQ', 'val': round(e_qoq, 1), 'label': f"+{e_qoq:.1f}% QoQ" if e_qoq > 0 else f"{e_qoq:.1f}% QoQ"})
                            else:
                                eps_traj.append({'type': 'RAW', 'val': None, 'label': '-'})
                        else:
                            eps_traj.append({'type': 'RAW', 'val': round(curr_eps, 2) if pd.notna(curr_eps) else None, 'label': f"₹{curr_eps:.1f}" if pd.notna(curr_eps) else '-'})
                    else:
                        eps_traj.append({'type': 'RAW', 'val': None, 'label': '-'})

                record['eps_trajectory'] = eps_traj
                record['sales_trajectory'] = sales_traj
                record['quarter_labels'] = q_labels

                # Extract current quarter YoY metrics if present
                if eps_traj and eps_traj[-1]['type'] == 'YoY':
                    record['eps_yoy_pct'] = eps_traj[-1]['val']
                elif eps_traj and eps_traj[-1]['type'] == 'QoQ':
                    record['eps_yoy_pct'] = eps_traj[-1]['val']

                if sales_traj and sales_traj[-1]['type'] == 'YoY':
                    record['sales_yoy_pct'] = sales_traj[-1]['val']
                elif sales_traj and sales_traj[-1]['type'] == 'QoQ':
                    record['sales_yoy_pct'] = sales_traj[-1]['val']

                # Dual acceleration check: both EPS and Sales accelerating across at least 2 valid consecutive points
                valid_eps_vals = [e['val'] for e in eps_traj if e['val'] is not None]
                valid_sales_vals = [s['val'] for s in sales_traj if s['val'] is not None]
                
                if len(valid_eps_vals) >= 2 and len(valid_sales_vals) >= 2:
                    if valid_eps_vals[-1] > valid_eps_vals[-2] and valid_sales_vals[-1] > valid_sales_vals[-2]:
                        record['dual_acceleration'] = True
                        record['verdict'] = '🟢 EXPLOSIVE (Code 33)'
                    else:
                        eyoy = record.get('eps_yoy_pct')
                        if eyoy is not None and eyoy > 25:
                            record['verdict'] = '🟢 STRONG'
                        elif eyoy is not None and eyoy < 0:
                            record['verdict'] = '🔴 DECRYING'
                        else:
                            record['verdict'] = '🟡 IN-LINE'
                else:
                    eyoy = record.get('eps_yoy_pct')
                    if eyoy is not None and eyoy > 25:
                        record['verdict'] = '🟢 STRONG'
                    elif eyoy is not None and eyoy < 0:
                        record['verdict'] = '🔴 DECRYING'
                    else:
                        record['verdict'] = '🟡 IN-LINE'

    except Exception as e:
        print(f"[Earnings Engine] Could not fetch yfinance data for {ticker}: {e}")

    # Fallback to local DB cache if yfinance returned empty financials
    if record['latest_quarter'] == 'N/A':
        try:
            from database import get_all_fundamentals_cache
            f_cache = get_all_fundamentals_cache()
            f_data = f_cache.get(clean_ticker) or f_cache.get(f"{clean_ticker}.NS")
            if f_data:
                record['latest_quarter'] = 'Annual (Screener)'
                record['eps_yoy_pct'] = f_data.get('eps_growth')
                record['sales_yoy_pct'] = f_data.get('sales_growth')
                
                eps_g = f_data.get('eps_growth')
                sales_g = f_data.get('sales_growth')
                
                record['eps_trajectory'] = [{'type': 'YoY', 'val': eps_g, 'label': f"+{eps_g:.1f}%" if eps_g and eps_g > 0 else f"{eps_g:.1f}%" if eps_g else '-'}]
                record['sales_trajectory'] = [{'type': 'YoY', 'val': sales_g, 'label': f"+{sales_g:.1f}%" if sales_g and sales_g > 0 else f"{sales_g:.1f}%" if sales_g else '-'}]
                
                if record['eps_yoy_pct'] is not None and record['eps_yoy_pct'] > 25:
                    record['verdict'] = '🟢 STRONG (Annual)'
                elif record['eps_yoy_pct'] is not None and record['eps_yoy_pct'] < 0:
                    record['verdict'] = '🔴 DECRYING (Annual)'
                else:
                    record['verdict'] = '🟡 IN-LINE (Annual)'
        except Exception:
            pass

    return record


def update_portfolio_earnings(tickers: list, force_refresh: bool = False) -> dict:
    """
    Update earnings data for active portfolio tickers and return the cache dictionary.
    Prunes stale non-portfolio tickers from the cache file.
    """
    cache = load_earnings_cache()
    updated = False

    clean_tickers = set([t.replace('.NS', '').replace('.BO', '').strip() for t in tickers if t and t != 'CASH'])

    # Prune non-portfolio keys from cache
    keys_to_remove = [k for k in cache.keys() if k not in clean_tickers]
    if keys_to_remove:
        for k in keys_to_remove:
            del cache[k]
        updated = True

    for ticker in clean_tickers:
        if force_refresh or ticker not in cache:
            data = fetch_ticker_quarterly_data(ticker)
            cache[ticker] = data
            updated = True

    if updated:
        save_earnings_cache(cache)

    return cache


def get_portfolio_new_results_alerts(cache: dict, active_tickers: set = None) -> list:
    """Return list of newly reported earnings results (within last 7 days) for active holdings."""
    alerts = []
    for ticker, data in cache.items():
        if active_tickers and ticker not in active_tickers:
            continue
        if data.get('is_new', False) and not data.get('is_etf', False):
            alerts.append(data)
    return alerts



def get_pre_earnings_risk_tickers(cache: dict, portfolio_df: pd.DataFrame, max_days: int = 5) -> list:
    """
    Identify portfolio stocks reporting earnings in <= max_days.
    """
    risk_list = []
    if portfolio_df is None or portfolio_df.empty:
        return risk_list

    ticker_col = 'ticker' if 'ticker' in portfolio_df.columns else 'NSE TICKER'
    
    for _, row in portfolio_df.iterrows():
        raw_t = str(row.get(ticker_col, '')).replace('.NS', '').replace('.BO', '').strip()
        if not raw_t or raw_t == 'CASH' or is_etf_ticker(raw_t):
            continue
            
        data = cache.get(raw_t, {})
        days_left = data.get('days_to_earnings')
        if days_left is not None and 0 <= days_left <= max_days:
            upcoming_date = data.get('upcoming_earnings_date', 'N/A')
            risk_list.append({
                'ticker': raw_t,
                'upcoming_date': upcoming_date,
                'days_left': days_left,
                'data': data
            })

    return risk_list
