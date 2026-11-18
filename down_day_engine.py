"""
Down-Day Resilience & Point-in-Time Alpha Engine
================================================
Institutional CANSLIM & Minervini quantitative engine for discovering
submerged leaders, Beach Ball accumulation (VB), and RS divergences
on historical and live market correction days.

Philosophy (William J. O'Neil & David Ryan):
"When the market averages drop hard, 90% of stocks fall with the tide.
The 10% that close green, hit new RS highs, or thrust up on volume
are the true market leaders holding coiled springs."
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st

import database
if not hasattr(database, 'save_down_day_resilience_events') or not hasattr(database, 'get_saved_down_day_resilience_events'):
    import importlib
    importlib.reload(database)

from database import (
    get_connection,
    read_sql_df,
    save_down_day_resilience_events,
    get_saved_down_day_resilience_events,
    get_available_resilience_dates
)

MATRIX_PATH = 'historical_prices_matrix.pkl'

def load_price_matrix():
    """Loads the cached multi-ticker price matrix with automatic disk mtime+size invalidation."""
    from price_history_manager import get_matrix_cache_signature, load_cached_price_matrix
    if not os.path.exists(MATRIX_PATH):
        return None
    try:
        df = load_cached_price_matrix(MATRIX_PATH, cache_sig=get_matrix_cache_signature(MATRIX_PATH))
        return df if df is not None and not df.empty else None
    except Exception as e:
        print(f"[DownDay Engine] Error loading price matrix: {e}")
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def get_historical_market_down_dates(
    market: str = 'INDIA',
    min_drop_pct: float = 0.75,
    max_breadth_pct: float = 35.0,
    lookback_days: int = 365
) -> pd.DataFrame:
    """
    Identifies all historical trading sessions where:
      1. Major index experienced an institutional distribution down-day (Drop >= min_drop_pct), OR
      2. 50-day SMA market breadth washed out below max_breadth_pct (< 35%), OR
      3. Net New Lows surged aggressively.
    """
    mkt_code = 'US' if market.upper() in ['US', 'USA'] else 'IN'
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    query = """
    SELECT *
    FROM market_breadth_daily
    WHERE market = ? AND date >= ?
    ORDER BY date DESC
    """
    try:
        df_breadth = read_sql_df(query, params=(mkt_code, cutoff))
    except Exception as e:
        print(f"[DownDay Engine] Error querying breadth history: {e}")
        df_breadth = pd.DataFrame()

    if df_breadth.empty:
        return pd.DataFrame()

    # Fill optional columns safely across varying schemas
    if 'advances' not in df_breadth.columns:
        df_breadth['advances'] = 0
    if 'declines' not in df_breadth.columns:
        df_breadth['declines'] = 0
    if 'above_50_pct' not in df_breadth.columns:
        df_breadth['above_50_pct'] = 50.0
    if 'net_new_highs' not in df_breadth.columns:
        df_breadth['net_new_highs'] = 0

    df_breadth['advances'] = df_breadth['advances'].fillna(0).astype(int)
    df_breadth['declines'] = df_breadth['declines'].fillna(0).astype(int)
    df_breadth['above_50_pct'] = df_breadth['above_50_pct'].fillna(50.0).astype(float)
    df_breadth['net_new_highs'] = df_breadth['net_new_highs'].fillna(0).astype(int)

    # Convert date to datetime
    df_breadth['date_dt'] = pd.to_datetime(df_breadth['date'])
    
    # Calculate Advance/Decline washouts
    df_breadth['ad_ratio'] = df_breadth['advances'] / df_breadth['declines'].clip(lower=1)
    
    # Flag washouts
    df_breadth['is_breadth_washout'] = df_breadth['above_50_pct'] <= max_breadth_pct
    df_breadth['is_panic_breadth'] = df_breadth['above_50_pct'] <= 25.0
    df_breadth['is_heavy_declines'] = df_breadth['ad_ratio'] <= 0.35
    df_breadth['is_nnh_crushed'] = df_breadth['net_new_highs'] <= -50

    # Filter for qualifying institutional down/washout days
    washout_df = df_breadth[
        df_breadth['is_breadth_washout'] | 
        df_breadth['is_heavy_declines'] | 
        df_breadth['is_nnh_crushed']
    ].copy()
    
    # Assign institutional down-day severity tier
    def score_severity(r):
        pts = 0
        if r['above_50_pct'] <= 20.0: pts += 3
        elif r['above_50_pct'] <= 35.0: pts += 2
        if r['ad_ratio'] <= 0.25: pts += 2
        elif r['ad_ratio'] <= 0.50: pts += 1
        if r['net_new_highs'] <= -100: pts += 2
        elif r['net_new_highs'] <= -50: pts += 1
        return pts

    washout_df['severity_score'] = washout_df.apply(score_severity, axis=1)
    washout_df['severity_badge'] = washout_df['severity_score'].apply(
        lambda s: "🚨 Critical Washout" if s >= 5 else ("⚠️ Deep Correction" if s >= 3 else "📉 Down Session")
    )
    
    return washout_df.sort_values('date', ascending=False)

@st.cache_data(ttl=1800, show_spinner=False)
def get_point_in_time_rs_series(target_date_str: str, market: str = 'INDIA') -> pd.Series:
    """
    Computes Point-in-Time CANSLIM Relative Strength Rating (1-99) for all tickers
    as of target_date_str using 1M (40%), 3M (35%), and 6M (25%) momentum.
    Filters to clean exchange trading sessions to scrub holiday artifacts (e.g. Oct 2).
    """
    is_us = market.upper() in ['US', 'USA']
    matrix = load_price_matrix()
    if matrix is None:
        return pd.Series(dtype=float)
    try:
        close_panel = matrix.xs('Close', level=1, axis=1)
    except Exception:
        return pd.Series(dtype=float)

    # 1. Isolate universe tickers for this market
    if is_us:
        mask = ~close_panel.columns.str.endswith('.NS') & ~close_panel.columns.str.endswith('.BO') & ~close_panel.columns.str.startswith('^')
    else:
        mask = (close_panel.columns.str.endswith('.NS') | close_panel.columns.str.endswith('.BO')) & ~close_panel.columns.str.startswith('^')

    sub_close = close_panel.loc[:, mask]
    if sub_close.empty:
        return pd.Series(dtype=float)

    # 2. Filter to clean active exchange trading sessions (eliminates national holidays)
    clean_close = get_clean_active_trading_panel(sub_close, is_us=is_us)
    if clean_close.empty:
        return pd.Series(dtype=float)

    target_dt = pd.to_datetime(target_date_str)
    valid_dates = clean_close.index[clean_close.index <= target_dt]
    if len(valid_dates) < 22:
        return pd.Series(dtype=float)

    # 3. Forward fill up to 3 days to handle non-trading circuit / illiquid gap days
    clean_close_ffill = clean_close.ffill(limit=3)

    t_idx = clean_close_ffill.index.get_loc(valid_dates[-1])
    c_t = clean_close_ffill.iloc[t_idx]
    c_1m = clean_close_ffill.iloc[max(0, t_idx - 21)]
    c_3m = clean_close_ffill.iloc[max(0, t_idx - 63)]
    c_6m = clean_close_ffill.iloc[max(0, t_idx - 126)]

    r_1m = (c_t / c_1m.replace(0, np.nan)) - 1.0
    r_3m = (c_t / c_3m.replace(0, np.nan)) - 1.0
    r_6m = (c_t / c_6m.replace(0, np.nan)) - 1.0

    # 4. Composite momentum with adaptive weights if 6M history is missing (e.g. recent IPO)
    full_mask = r_1m.notna() & r_3m.notna() & r_6m.notna()
    partial_mask = r_1m.notna() & r_3m.notna() & r_6m.isna()

    weighted_raw = pd.Series(np.nan, index=sub_close.columns)
    weighted_raw[full_mask] = (r_1m[full_mask] * 0.40) + (r_3m[full_mask] * 0.35) + (r_6m[full_mask] * 0.25)
    weighted_raw[partial_mask] = (r_1m[partial_mask] * 0.55) + (r_3m[partial_mask] * 0.45)

    sub_raw = weighted_raw.dropna()
    if sub_raw.empty:
        return pd.Series(dtype=float)

    # CANSLIM 1 to 99 percentile ranking
    rs_ranked = (sub_raw.rank(pct=True) * 98.0 + 1.0).round(1)
    
    # Also index clean ticker symbols so both 'TICKER.NS' and 'TICKER' resolve in .map()
    clean_idx_map = {t.replace('.NS', '').replace('.BO', ''): val for t, val in rs_ranked.items()}
    combined_rs = pd.concat([rs_ranked, pd.Series(clean_idx_map)])
    # Deduplicate index so tickers (especially US symbols like AAPL) are uniquely indexed
    combined_rs = combined_rs[~combined_rs.index.duplicated(keep='first')]
    return combined_rs

def enrich_down_day_resilience_records(df: pd.DataFrame, target_date_str: str, market: str = 'INDIA') -> pd.DataFrame:
    """
    Enriches resilience dataframe with:
      - Point-in-Time RS Rating (0-99)
      - Canonical Industry Group (resolving any 'Unknown')
      - IBD Letter Grade ('A+', 'A', 'B', etc.)
      - Livermore / O'Neil Sister Tandem Role ('Alpha Leader', 'Sister Tandem', etc.)
      - Alpha Score alias
    """
    if df.empty:
        for col in ['RS_Rating', 'Alpha_Score', 'Industry', 'Ind_Grade', 'Sister_Role']:
            if col not in df.columns:
                df[col] = []
        return df

    is_us = market.upper() in ['US', 'USA']
    rs_series = get_point_in_time_rs_series(target_date_str, market)

    try:
        from industry_group_engine import get_stock_group_intelligence
        intel = get_stock_group_intelligence(universe_scope="US (S&P 500 / Nasdaq)" if is_us else "India (NSE/BSE)")
    except Exception:
        intel = {}

    try:
        from database import get_all_fundamentals_cache
        fc = get_all_fundamentals_cache()
    except Exception:
        fc = {}

    us_gics = {}
    if is_us and os.path.exists('data/sp500_gics_map.json'):
        try:
            import json
            with open('data/sp500_gics_map.json', 'r') as f:
                us_gics = json.load(f)
        except Exception:
            pass

    rs_list = []
    ind_list = []
    grd_list = []
    role_list = []

    for _, row in df.iterrows():
        raw_t = str(row.get('Ticker', row.get('ticker', '')))
        clean_t = str(row.get('Clean_Ticker', row.get('clean_ticker', '')))
        if not clean_t:
            clean_t = raw_t.replace('.NS', '').replace('.BO', '')

        # 1. RS Rating
        rs_val = rs_series.get(raw_t, rs_series.get(clean_t, np.nan))
        if isinstance(rs_val, (pd.Series, np.ndarray, list)):
            rs_val = rs_val.iloc[0] if hasattr(rs_val, 'iloc') and len(rs_val) > 0 else (rs_val[0] if len(rs_val) > 0 else np.nan)
        if pd.isna(rs_val) or rs_val is None:
            f_info = fc.get(raw_t) or fc.get(clean_t) or {}
            rs_val = float(f_info.get('rs_score', 80.0) or 80.0)
        rs_list.append(round(float(rs_val), 1))

        # 2. Industry, Grade, Role
        existing_ind = str(row.get('Industry', row.get('industry', 'Unknown')) or '')
        existing_grd = str(row.get('Ind_Grade', row.get('ind_grade', '—')) or '')
        existing_role = str(row.get('Sister_Role', row.get('sister_role', '—')) or '')

        if existing_ind not in ['Unknown', '—', '', 'None', 'nan'] and existing_grd != '—':
            ind_list.append(existing_ind)
            grd_list.append(existing_grd)
            role_list.append(existing_role)
            continue

        info = intel.get(clean_t) or intel.get(raw_t)
        f_info = fc.get(raw_t) or fc.get(clean_t) or {}

        if info:
            ind_list.append(info.get('group_name', 'Broad Market'))
            grd_list.append(info.get('group_grade_badge', '—'))
            role_list.append(info.get('stock_role', '—'))
        elif clean_t in us_gics:
            g_meta = us_gics[clean_t]
            ind_list.append(g_meta.get('industry', 'Broad Market'))
            grd_list.append('—')
            role_list.append('—')
        elif f_info and f_info.get('industry'):
            ind_list.append(f_info.get('industry', 'Broad Market'))
            grd_list.append('—')
            role_list.append('—')
        else:
            ind_list.append('Broad Market')
            grd_list.append('—')
            role_list.append('—')

    df['RS_Rating'] = rs_list
    df['Industry'] = ind_list
    df['Ind_Grade'] = grd_list
    df['Sister_Role'] = role_list

    if 'Resilience_Score' in df.columns:
        df['Alpha_Score'] = df['Resilience_Score']
    elif 'resilience_score' in df.columns:
        df['Alpha_Score'] = df['resilience_score']
        df['Resilience_Score'] = df['resilience_score']
    else:
        df['Alpha_Score'] = 50.0
        df['Resilience_Score'] = 50.0

    return df

@st.cache_data(ttl=1800, show_spinner=False)
def compute_down_day_resilience(
    target_date_str: str,
    market: str = 'INDIA',
    min_price: float = 20.0,
    max_dist_high: float = 15.0
) -> pd.DataFrame:
    """
    Computes Point-in-Time resilience metrics for all stocks on the chosen target date.
    
    Returns DataFrame classified into 3 CANSLIM Resilience Archetypes:
      1. 🏖️ BEACH BALL (VB): Daily Return >= +3.8% on Volume >= 1.0x ADTV.
      2. 🟢 GREEN ON RED (Relative Anchor): Daily Return > 0%, Dist from 52W High >= -max_dist_high%.
      3. 🚀 NEW 52W HIGH ON DOWN DAY: Closing at absolute 252-day high on market down day.
    """
    mkt_tag = 'US' if market.upper() in ['US', 'USA'] else 'INDIA'
    is_us = mkt_tag == 'US'
    curr_symbol = "$" if is_us else "₹"

    # 1. First check if already persisted in database
    try:
        saved_rows = get_saved_down_day_resilience_events(target_date_str, mkt_tag)
        if saved_rows:
            df_saved = pd.DataFrame(saved_rows)
            col_map = {
                'ticker': 'Ticker',
                'clean_ticker': 'Clean_Ticker',
                'archetype': 'Archetype',
                'resilience_score': 'Resilience_Score',
                'return_1d': 'Return_1D',
                'dist_52w_high': 'Dist_52W_High',
                'volume_mult': 'Volume_Mult',
                'close_price': 'Close',
                'turnover_cr': 'Turnover_Cr',
                'industry': 'Industry',
                'ind_grade': 'Ind_Grade',
                'sister_role': 'Sister_Role'
            }
            for old_c, new_c in col_map.items():
                if old_c in df_saved.columns and new_c not in df_saved.columns:
                    df_saved[new_c] = df_saved[old_c]
            
            # Apply dynamic price & distance filters
            if 'Close' in df_saved.columns:
                df_saved = df_saved[df_saved['Close'] >= min_price]
            if 'Dist_52W_High' in df_saved.columns:
                df_saved = df_saved[df_saved['Dist_52W_High'] >= -abs(max_dist_high)]
            
            # Enrich with real RS Rating, Canonical Industry, IBD Grade, and Sister Role
            df_saved = enrich_down_day_resilience_records(df_saved, target_date_str, mkt_tag)
            return df_saved.sort_values(['Resilience_Score', 'Return_1D'], ascending=[False, False])
    except Exception as e:
        print(f"[DownDay Engine] DB fetch error: {e}")

    matrix = load_price_matrix()
    if matrix is None:
        return pd.DataFrame()
    
    try:
        close_panel = matrix.xs('Close', level=1, axis=1)
        vol_panel = matrix.xs('Volume', level=1, axis=1)
    except Exception as e:
        print(f"[DownDay Engine] Error indexing matrix levels: {e}")
        return pd.DataFrame()

    target_dt = pd.to_datetime(target_date_str)
    
    # Find exact or nearest preceding date
    available_dates = close_panel.index.sort_values()
    valid_dates = available_dates[available_dates <= target_dt]
    if len(valid_dates) < 2:
        return pd.DataFrame()
        
    t_date = valid_dates[-1]
    t_idx = close_panel.index.get_loc(t_date)
    p_date = close_panel.index[t_idx - 1]
    
    # Filter universe tickers (US vs India)
    if is_us:
        tickers = [c for c in close_panel.columns if not c.endswith('.NS') and not c.endswith('.BO')]
    else:
        tickers = [c for c in close_panel.columns if c.endswith(('.NS', '.BO'))]

    if not tickers:
        return pd.DataFrame()

    sub_close = close_panel[tickers]
    sub_vol = vol_panel[tickers]

    c_today = sub_close.loc[t_date]
    c_prev = sub_close.loc[p_date]
    ret_1d = ((c_today / c_prev) - 1.0) * 100.0

    # 52-week (252d) high as of target date
    start_252 = max(0, t_idx - 251)
    window_252 = sub_close.iloc[start_252:t_idx + 1]
    high_252 = window_252.max()
    dist_high = ((c_today / high_252) - 1.0) * 100.0

    # 20D Volume Multiple
    vol_today = sub_vol.loc[t_date]
    start_20 = max(0, t_idx - 20)
    vol_20d = sub_vol.iloc[start_20:t_idx].mean().replace(0, np.nan)
    vol_mult = vol_today / vol_20d

    # Turnover / Liquidity
    turnover_cr = (c_today * vol_today) / 10000000.0 if not is_us else (c_today * vol_today) / 1000000.0

    # Build Raw DataFrame
    df = pd.DataFrame({
        'Ticker': tickers,
        'Close': c_today.values,
        'Return_1D': ret_1d.values,
        'Dist_52W_High': dist_high.values,
        'Volume_Mult': vol_mult.values,
        'Turnover_Cr': turnover_cr.values
    }).dropna()

    # Apply Base Filters
    df = df[df['Close'] >= min_price]
    min_turnover = 3.0 if not is_us else 5.0
    df = df[df['Turnover_Cr'] >= min_turnover]

    # Filter to Resilient Names: positive return and within threshold of 52W high
    resilient = df[(df['Return_1D'] > 0.0) & (df['Dist_52W_High'] >= -abs(max_dist_high))].copy()
    if resilient.empty:
        return pd.DataFrame()

    # Archetype Classification
    def classify_resilience_archetype(row):
        r1d = row['Return_1D']
        dh = row['Dist_52W_High']
        vm = row['Volume_Mult']
        
        if dh >= -0.5:
            return "🚀 New 52W High on Down Day"
        elif r1d >= 3.8 and vm >= 1.0:
            return "🏖️ Beach Ball (VB Thrust)"
        elif r1d >= 2.0 and vm >= 1.2:
            return "⚡ Institutional Accumulation"
        else:
            return "🟢 Green on Red (Relative Anchor)"

    resilient['Archetype'] = resilient.apply(classify_resilience_archetype, axis=1)

    # Clean ticker display
    resilient['Clean_Ticker'] = resilient['Ticker'].apply(lambda t: t.replace('.NS', '').replace('.BO', ''))
    
    # Add Industry mapping if available
    try:
        from industry_group_engine import get_stock_group_intelligence
        intel = get_stock_group_intelligence(universe_scope="US (S&P 500 / Nasdaq)" if is_us else "India (NSE/BSE)")
        resilient['Industry'] = resilient['Clean_Ticker'].apply(lambda t: intel.get(t, {}).get('group_name', 'Unknown'))
        resilient['Ind_Grade'] = resilient['Clean_Ticker'].apply(lambda t: intel.get(t, {}).get('group_grade_badge', '—'))
        resilient['Sister_Role'] = resilient['Clean_Ticker'].apply(lambda t: intel.get(t, {}).get('stock_role', '—'))
    except Exception:
        resilient['Industry'] = 'Unknown'
        resilient['Ind_Grade'] = '—'
        resilient['Sister_Role'] = '—'

    # Composite Resilience Score (Weights: 1D return 40%, Volume Expansion 30%, Proximity to 52W High 30%)
    norm_ret = resilient['Return_1D'].clip(0, 15) / 15.0
    norm_vol = resilient['Volume_Mult'].clip(0.5, 5.0) / 5.0
    norm_prox = (15.0 + resilient['Dist_52W_High'].clip(-15.0, 0.0)) / 15.0
    resilient['Resilience_Score'] = ((norm_ret * 40.0) + (norm_vol * 30.0) + (norm_prox * 30.0)).round(1)

    # Persist computed resilient events to database for instant future retrieval
    try:
        events_to_save = []
        for _, row in resilient.iterrows():
            events_to_save.append({
                'date': target_date_str,
                'market': mkt_tag,
                'ticker': row['Ticker'],
                'clean_ticker': row['Clean_Ticker'],
                'archetype': row['Archetype'],
                'resilience_score': float(row['Resilience_Score']),
                'return_1d': float(row['Return_1D']),
                'dist_52w_high': float(row['Dist_52W_High']),
                'volume_mult': float(row['Volume_Mult']),
                'close_price': float(row['Close']),
                'turnover_cr': float(row['Turnover_Cr']),
                'industry': str(row['Industry']),
                'ind_grade': str(row['Ind_Grade']),
                'sister_role': str(row['Sister_Role']),
                'breadth_pct': 0.0
            })
        if events_to_save:
            save_down_day_resilience_events(events_to_save)
    except Exception as e:
        print(f"[DownDay Engine] DB persistence warning: {e}")

    # Enrich with real RS Rating, Canonical Industry, IBD Grade, and Sister Role
    resilient = enrich_down_day_resilience_records(resilient, target_date_str, mkt_tag)

    return resilient.sort_values(['Resilience_Score', 'Return_1D'], ascending=[False, False])

def sync_all_historical_down_days(market: str = 'INDIA', lookback_days: int = 180, max_dates: int = 40) -> int:
    """
    Backfills and persists all qualifying historical down days into SQLite/Turso.
    """
    mkt_tag = 'US' if market.upper() in ['US', 'USA'] else 'INDIA'
    washouts = get_historical_market_down_dates(market=mkt_tag, lookback_days=lookback_days)
    if washouts.empty:
        return 0
    
    dates = washouts['date'].head(max_dates).tolist()
    total_events = 0
    for d in dates:
        df_res = compute_down_day_resilience(d, market=mkt_tag)
        total_events += len(df_res)
    return total_events

def format_tradingview_watchlist(tickers: list, market: str = 'INDIA') -> str:
    """Formats a list of tickers into a clean 1-click TradingView import string."""
    is_us = market.upper() in ['US', 'USA']
    prefix = "" if is_us else "NSE:"
    clean = [f"{prefix}{t.replace('.NS', '').replace('.BO', '')}" for t in tickers]
    return ",".join(clean)

def format_notion_markdown_table(df_resilient: pd.DataFrame, target_date: str) -> str:
    """Formats resilient leaders into a copyable Markdown table optimized for Notion paste."""
    if df_resilient.empty:
        return "No resilient stocks recorded for this session."
    
    lines = [
        f"### 🏖️ Down-Day Resilient Leaders — {target_date}",
        "| Symbol | RS Rating | Alpha Score | Archetype | 1D Gain % | Dist 52W High | Volume Mult | Industry & Grade | Sister Role |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for _, r in df_resilient.head(30).iterrows():
        t = r.get('Clean_Ticker', '')
        rs = f"{r.get('RS_Rating', 80):.0f}"
        alpha = f"{r.get('Alpha_Score', r.get('Resilience_Score', 0)):.1f}"
        arch = r.get('Archetype', '')
        g = f"+{r.get('Return_1D', 0):.2f}%"
        dh = f"{r.get('Dist_52W_High', 0):.1f}%"
        vm = f"{r.get('Volume_Mult', 1.0):.2f}x"
        ind = f"{r.get('Industry', 'Broad Market')} ({r.get('Ind_Grade', '—')})"
        role = r.get('Sister_Role', '—')
        lines.append(f"| **{t}** | {rs} | {alpha} | {arch} | {g} | {dh} | {vm} | {ind} | {role} |")
        
    return "\n".join(lines)

def format_telegram_alert(
    target_date: str,
    breadth_pct: float,
    net_highs: int,
    df_resilient: pd.DataFrame,
    market: str = 'INDIA'
) -> str:
    """Formats an institutional Telegram alert message."""
    tv_str = format_tradingview_watchlist(df_resilient['Clean_Ticker'].head(15).tolist(), market=market)
    
    lines = [
        f"🚨 *CANSLIM MARKET DOWN-DAY BRIEFING* ({target_date})",
        f"🌊 *Breadth Washout*: {breadth_pct:.1f}% > 50-SMA | Net 52W Highs: {net_highs}",
        "",
        "🏖️ *TOP RESILIENT LEADERS (RS ≥ 80 & Alpha ≥ 50)*:"
    ]
    for i, (_, r) in enumerate(df_resilient.head(8).iterrows()):
        t = r.get('Clean_Ticker', '')
        rs = r.get('RS_Rating', 80)
        alpha = r.get('Alpha_Score', r.get('Resilience_Score', 0))
        arch = str(r.get('Archetype', '')).split()[0] if r.get('Archetype') else '🏖️'
        g = r.get('Return_1D', 0)
        vm = r.get('Volume_Mult', 1.0)
        ind = r.get('Industry', 'Broad Market')
        lines.append(f"{i+1}. *{t}* (RS {rs:.0f} • Alpha {alpha:.1f}): {arch} +{g:.1f}% on {vm:.1f}x vol — _{ind}_")
        
    lines.append("")
    lines.append(f"📋 *TradingView Watchlist*:\n`{tv_str}`")
    return "\n".join(lines)

def format_cluster_notion_table(df_cluster: pd.DataFrame, dates_list: list) -> str:
    """Formats an institutional markdown table for Notion copy-paste across a multi-day cluster."""
    if df_cluster.empty:
        return "No cluster leaders found."
    d_start = sorted(dates_list)[0] if dates_list else ""
    d_end = sorted(dates_list)[-1] if dates_list else ""
    n_days = len(dates_list)
    lines = [
        f"### 🗂️ Down-Day Washout Cluster Leaders — {d_start} to {d_end} ({n_days} Sessions)",
        "| Symbol | RS Rating | Hit Rate | Cluster Gain % | Avg Alpha | Archetype Footprint | Industry & Grade | Sister Role |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for _, r in df_cluster.head(50).iterrows():
        t = r.get('Clean_Ticker', '')
        rs = f"{r.get('RS_Rating', 80):.0f}"
        hit = r.get('Hit_Rate_Label', '')
        gain = f"{r.get('Cluster_Gain', 0):+.2f}%"
        alpha = f"{r.get('Avg_Alpha', 0):.1f}"
        arch = r.get('Archetype_Footprint', '')
        ind = f"{r.get('Industry', 'Broad Market')} ({r.get('Ind_Grade', '—')})"
        role = r.get('Sister_Role', '—')
        lines.append(f"| **{t}** | {rs} | {hit} | {gain} | {alpha} | {arch} | {ind} | {role} |")
    return "\n".join(lines)

def format_cluster_telegram_alert(df_cluster: pd.DataFrame, dates_list: list, market: str = 'INDIA') -> str:
    """Formats a concise cluster alert for Telegram."""
    if df_cluster.empty:
        return "No cluster leaders found."
    d_start = sorted(dates_list)[0] if dates_list else ""
    d_end = sorted(dates_list)[-1] if dates_list else ""
    n_days = len(dates_list)
    tv_str = format_tradingview_watchlist(df_cluster['Clean_Ticker'].head(15).tolist(), market=market)
    lines = [
        f"🚨 *CANSLIM MULTI-DAY WASHOUT CLUSTER REPORT*",
        f"🗓️ *Correction Window*: {d_start} → {d_end} ({n_days} down sessions)",
        f"🛡️ *Resilient Universe*: {len(df_cluster)} stocks held green on ≥1 down day",
        "",
        "💎 *TOP CLUSTER GENERALS (Top Hit Rate & RS ≥ 80)*:"
    ]
    for i, (_, r) in enumerate(df_cluster.head(8).iterrows()):
        t = r.get('Clean_Ticker', '')
        rs = r.get('RS_Rating', 80)
        hit = r.get('Hit_Rate_Label', '')
        gain = r.get('Cluster_Gain', 0)
        arch = r.get('Archetype_Footprint', '')
        ind = r.get('Industry', 'Broad Market')
        lines.append(f"{i+1}. *{t}* (RS {rs:.0f} • Hit {hit}): {gain:+.1f}% | {arch} — _{ind}_")
    lines.append("")
    lines.append(f"📋 *TradingView Watchlist*:\n`{tv_str}`")
    return "\n".join(lines)

@st.cache_data(ttl=1800, show_spinner=False)
def get_cumulative_thrust_map(market: str = 'INDIA', lookback_days: int = 90) -> dict:
    """
    Returns a dict {clean_ticker: {'count_3m': int, 'count_1m': int, 'badge': str}}
    aggregating beach ball power plays across recent low-breadth sessions.
    """
    conn = get_connection()
    c = conn.cursor()
    clean_m = str(market or 'INDIA').strip().upper()
    if clean_m == 'US_IXIC':
        mkt_tags = ['US_IXIC', 'US']
    elif clean_m in ['US', 'USA']:
        mkt_tags = ['US']
    else:
        mkt_tags = ['INDIA']
    
    cutoff_3m = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    cutoff_1m = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    result = {}
    try:
        # Check beach_ball_events (US_IXIC prefers US_IXIC events first, falling back to US)
        placeholders = ','.join(['?'] * len(mkt_tags))
        c.execute(f'''
            SELECT ticker, 
                   COUNT(CASE WHEN event_date >= ? THEN 1 END) as c_3m,
                   COUNT(CASE WHEN event_date >= ? THEN 1 END) as c_1m
            FROM beach_ball_events
            WHERE market IN ({placeholders}) AND event_date >= ?
            GROUP BY ticker
        ''', (cutoff_3m, cutoff_1m, *mkt_tags, cutoff_3m))
        rows = c.fetchall()
        for r in rows:
            clean = r[0].replace('.NS', '').replace('.BO', '')
            c3 = int(r[1] or 0)
            c1 = int(r[2] or 0)
            if c3 > 0:
                result[clean] = {
                    'count_3m': c3,
                    'count_1m': c1,
                    'badge': f"🏖️ {c3}x Power Plays" if c3 >= 2 else "🏖️ 1x Thrust"
                }
    except Exception as e:
        print(f"[DownDay Engine] get_cumulative_thrust_map error: {e}")
    finally:
        conn.close()
    return result


_ETF_EXACT_SET = {
    # US Major Index, Sector, Bond, Commodity & Thematic ETFs
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "IVV", "RSP", "QQQM", "TQQQ", "SQQQ",
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    "SMH", "SOXX", "IGV", "ARKK", "ARKG", "ARKW", "XBI", "IBB", "KRE", "KBE", "XOP",
    "OIH", "GDX", "GDXJ", "GLD", "SLV", "IAU", "USO", "UNG", "DBC", "PDBC",
    "TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "JNK", "IBIT", "FBTC", "GBTC",
    "SCHD", "VIG", "VYM", "JEPI", "JEPQ", "VXUS", "EFA", "EEM", "FXI", "KWEB", "INDA",
    # India NSE ETFs that lack standard suffixes
    "CPSEETF", "BHARAT22", "MAFANG", "MON100", "HNGSNGBEES", "SHARIABEES", "GILT5YBEES",
    "LIQUIDCASE", "GOLDCASE", "SILVERCASE", "MID150CASE", "TOP100CASE",
}

_ETF_SUBSTRINGS = (
    "BEES", "IETF", "ETF", "LIQUID", "GOLDBEES", "SILVERBEES", "NIFTY", "SENSEX",
    "JUNIORBEES", "BANKBEES", "PSUBNKBEES", "ITBEES", "PHARMABEES", "CONSUMBEES",
    "INFRABEES", "AUTOBEES", "MID150BEES", "NV20", "BHARATBOND", "GOLD1", "SILVER1",
)


def is_etf_or_non_equity(ticker: str) -> bool:
    """
    Returns True if the ticker is an Index/Liquid/Gold/Silver/Factor ETF or Debt instrument
    that should be excluded from pure-equity CANSLIM leadership screeners.
    """
    if not ticker or not isinstance(ticker, str):
        return True
    clean = ticker.strip().upper().replace(".NS", "").replace(".BO", "")
    if clean in _ETF_EXACT_SET:
        return True
    for sub in _ETF_SUBSTRINGS:
        if sub in clean:
            return True
    return False


def get_clean_active_trading_panel(close_panel: pd.DataFrame, is_us: bool = False) -> pd.DataFrame:
    """
    Filters price panel to valid trading sessions where the exchange was open.
    Eliminates national holidays (e.g. Gandhi Jayanti Oct 2), partial sessions,
    and sparse data artifacts that cause mathematical distortion.
    """
    min_stocks = 200 if is_us else 500
    active_mask = close_panel.notna().sum(axis=1) >= min_stocks
    return close_panel.loc[active_mask].copy()

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_official_benchmark(market: str = 'INDIA', benchmark_symbol: str = None) -> tuple:
    """
    Downloads or fetches the official broad market index benchmark.
    India: ^CRSLDX (NIFTY 500) with fallback to ^NSEI (NIFTY 50)
    US default: ^GSPC (S&P 500) with fallback to SPY; explicit override (e.g. '^IXIC') honored cleanly.
    Normalizes Series index (tz-naive, midnight-normalized) & settles US completed sessions.
    """
    is_us = market.upper() in ['US', 'USA', 'US_IXIC']
    if benchmark_symbol:
        symbols = [str(benchmark_symbol).strip()]
        if symbols[0] == '^IXIC':
            symbols.append('QQQ')
        dl_period = '400d'
    else:
        symbols = ['^GSPC', 'SPY'] if is_us else ['^CRSLDX', '^NSEI']
        dl_period = '1y'

    name_map = {
        '^GSPC': "S&P 500 (^GSPC)",
        'SPY': "SPY (S&P 500 ETF)",
        '^IXIC': "Nasdaq Composite (^IXIC)",
        'QQQ': "QQQ (Nasdaq Proxy)",
        '^CRSLDX': "NIFTY 500 (^CRSLDX)",
        '^NSEI': "NIFTY 50 (^NSEI)",
    }

    last_us_dt = None
    if is_us:
        try:
            from price_history_manager import get_latest_completed_session_date
            last_us_dt = pd.Timestamp(get_latest_completed_session_date('US'))
        except Exception:
            last_us_dt = None

    for s in symbols:
        try:
            df = yf.download(s, period=dl_period, progress=False)
            if df is not None and not df.empty:
                c = df['Close']
                if hasattr(c, 'iloc') and len(c.shape) > 1:
                    c = c.iloc[:, 0]
                c = c.dropna()
                if len(c) >= 21:
                    c.index = pd.to_datetime(c.index)
                    if c.index.tz is not None:
                        c.index = c.index.tz_localize(None)
                    c.index = c.index.normalize()
                    if last_us_dt is not None:
                        c_set = c.loc[c.index <= last_us_dt]
                        if len(c_set) >= 21:
                            c = c_set
                    return c, name_map.get(s, s)
        except Exception as e:
            print(f"[DownDay Engine] Benchmark download warning for {s}: {e}")
    return None, ""

def get_session_advance_decline(target_date_str: str, market: str = 'INDIA') -> tuple:
    """
    Computes exact point-in-time advances and declines from clean price matrix
    if market_breadth_daily has 0 or null recorded.
    """
    matrix = load_price_matrix()
    if matrix is None:
        return 0, 0
    is_us = market.upper() in ['US', 'USA']
    try:
        close_panel = matrix.xs('Close', level=1, axis=1)
        if is_us:
            tickers = [c for c in close_panel.columns if not c.endswith('.NS') and not c.endswith('.BO')]
        else:
            tickers = [c for c in close_panel.columns if c.endswith(('.NS', '.BO'))]
        sub_close = close_panel[tickers]
        clean_close = get_clean_active_trading_panel(sub_close, is_us=is_us)
        if target_date_str in clean_close.index:
            idx = clean_close.index.get_loc(target_date_str)
            if idx > 0:
                c_cur = clean_close.iloc[idx]
                c_prv = clean_close.iloc[idx - 1]
                ret = (c_cur / c_prv) - 1.0
                adv = int((ret > 0.0).sum())
                dec = int((ret < 0.0).sum())
                return adv, dec
    except Exception as e:
        print(f"[DownDay Engine] get_session_advance_decline error: {e}")
    return 0, 0

@st.cache_data(ttl=1800, show_spinner=False)
def compute_multi_horizon_correction_analytics(
    market: str = 'INDIA',
    horizon_days: int = 21,
    min_price: float = 20.0,
    _cache_v: str = "v2_1_clean_institutional"
) -> tuple:
    """
    Multi-Horizon Macro Correction & Counter-Trend Divergence Engine:
    Evaluates market performance over the last N active trading days (e.g. 1M / 21D, 3M / 63D, 6M / 126D).
    Identifies stocks that decoupled from broad market weakness and generated positive excess alpha.
    Scrubs holiday artifacts and unadjusted corporate action split spikes.
    
    Returns:
      (df_divergent, bench_summary, df_group_alpha)
    """
    matrix = load_price_matrix()
    if matrix is None:
        return pd.DataFrame(), {}, pd.DataFrame()
        
    is_us = market.upper() in ['US', 'USA']
    mkt_tag = 'US' if is_us else 'INDIA'
    
    try:
        close_panel = matrix.xs('Close', level=1, axis=1)
        vol_panel = matrix.xs('Volume', level=1, axis=1)
    except Exception as e:
        print(f"[DownDay Engine] Horizon matrix indexing error: {e}")
        return pd.DataFrame(), {}, pd.DataFrame()

    # Universe tickers
    if is_us:
        tickers = [c for c in close_panel.columns if not c.endswith('.NS') and not c.endswith('.BO') and not c.startswith('^')]
    else:
        tickers = [c for c in close_panel.columns if c.endswith(('.NS', '.BO')) and not c.startswith('^')]

    if not tickers:
        return pd.DataFrame(), {}, pd.DataFrame()

    sub_close = close_panel[tickers]
    sub_vol = vol_panel[tickers]

    # Clean active trading days (completely scrubs national holidays & non-trading dates)
    clean_close = get_clean_active_trading_panel(sub_close, is_us=is_us)
    clean_vol = sub_vol.loc[clean_close.index]

    n_active = len(clean_close)
    if n_active < horizon_days + 1:
        return pd.DataFrame(), {}, pd.DataFrame()

    # Download or get official benchmark
    bench_series, bench_name = get_cached_official_benchmark(market=mkt_tag)
    
    if bench_series is not None:
        # Align official benchmark with clean active dates
        common_dates = clean_close.index.intersection(bench_series.index)
        if len(common_dates) >= horizon_days + 1:
            clean_close = clean_close.loc[common_dates]
            clean_vol = clean_vol.loc[common_dates]
            b_window = bench_series.loc[common_dates].tail(horizon_days + 1)
        elif len(common_dates) >= horizon_days:
            clean_close = clean_close.loc[common_dates]
            clean_vol = clean_vol.loc[common_dates]
            b_window = bench_series.loc[common_dates].tail(horizon_days)
        else:
            b_window = bench_series.tail(horizon_days + 1)
    else:
        # Fallback to holiday-scrubbed equal-weight median index
        bench_series = clean_close.median(axis=1)
        bench_name = "NIFTY Broad Market Index (Equal-Weight)" if not is_us else "US Broad Market Index (Equal-Weight)"
        b_window = bench_series.tail(horizon_days + 1)

    if len(b_window) < 2:
        return pd.DataFrame(), {}, pd.DataFrame()

    t_start = b_window.index[0]
    t_end = b_window.index[-1]

    # Mathematically exact benchmark return & max drawdown over horizon
    bench_ret_pct = float(((b_window.iloc[-1] / b_window.iloc[0]) - 1.0) * 100.0)
    b_rolling_max = b_window.cummax()
    b_drawdown = ((b_window - b_rolling_max) / b_rolling_max) * 100.0
    b_max_dd = float(b_drawdown.min())

    # Daily benchmark returns in window to identify down days
    b_daily_rets = b_window.pct_change().dropna()
    down_days_mask = b_daily_rets < 0.0
    down_day_dates = b_daily_rets[down_days_mask].index
    total_down_days = len(down_day_dates)

    # Window slice for stocks
    window_close = clean_close.loc[t_start:t_end]
    c_start = window_close.iloc[0]
    c_end = window_close.iloc[-1]
    stock_rets = ((c_end / c_start) - 1.0) * 100.0

    # 52W High Proximity (evaluated over active trading sessions)
    window_252 = clean_close.tail(252)
    high_252 = window_252.max()
    dist_high = ((c_end / high_252) - 1.0) * 100.0

    # 20D Volume Multiple
    vol_latest = clean_vol.loc[t_end]
    vol_20d_avg = clean_vol.tail(21).iloc[:-1].mean().replace(0, np.nan)
    vol_mult = vol_latest / vol_20d_avg

    # Corporate Action Split Scrubbing:
    # Detect any unadjusted corporate consolidation/reverse split spike (> 100% daily jump)
    window_daily = window_close.pct_change()
    has_corporate_split = (window_daily > 1.0).any(axis=0)

    # 1. Horizon Daily RS Outperformance (Ameet Rai Persistence across Horizon H)
    aligned_daily = window_daily.loc[b_daily_rets.index]
    daily_outperf = aligned_daily.gt(b_daily_rets, axis=0)
    rs_days_count = daily_outperf.sum(axis=0)
    total_h_bars = len(b_daily_rets)
    rs_days_pct = (rs_days_count / max(1, total_h_bars)) * 100.0

    # 2. Recent 20D Daily RS Outperformance (Active Accumulation Momentum)
    close_up_to_end = clean_close.loc[:t_end]
    recent_bench_win = bench_series.loc[:t_end].tail(21)
    recent_b_rets = recent_bench_win.pct_change().dropna()
    if len(recent_b_rets) > 0:
        recent_stock_rets = close_up_to_end.pct_change().loc[recent_b_rets.index]
        recent_outperf = recent_stock_rets.gt(recent_b_rets, axis=0)
        c20 = recent_outperf.sum(axis=0)
        p20 = (c20 / len(recent_b_rets)) * 100.0
    else:
        c20 = rs_days_count
        p20 = rs_days_pct

    # 3. Moving Average Proximity & Structure Adherence (21 EMA & 50 SMA)
    ema_21_series = close_up_to_end.ewm(span=21, adjust=False).mean().iloc[-1]
    sma_50_series = close_up_to_end.rolling(window=50, min_periods=35).mean().iloc[-1]
    dist_21 = ((c_end - ema_21_series) / ema_21_series) * 100.0
    dist_50 = ((c_end - sma_50_series) / sma_50_series) * 100.0
    holding_21 = (dist_21 >= -2.0) & (c_end >= sma_50_series.fillna(0.0))
    actionable_pocket = (dist_21 >= -2.0) & (dist_21 <= 3.5) & (c_end >= sma_50_series.fillna(0.0))

    # Down-Day Hit Rate (% of days stock was GREEN when Benchmark was RED)
    if total_down_days > 0 and len(down_day_dates) > 0:
        down_stock_rets = window_daily.loc[window_daily.index.isin(down_day_dates)]
        green_on_down_count = (down_stock_rets > 0.0).sum(axis=0)
        down_day_win_rate = (green_on_down_count / total_down_days) * 100.0
    else:
        down_day_win_rate = pd.Series(50.0, index=clean_close.columns)

    # Turnover
    turnover_cr = (c_end * vol_latest) / 10000000.0 if not is_us else (c_end * vol_latest) / 1000000.0

    # Assemble DataFrame
    df = pd.DataFrame({
        'Ticker': clean_close.columns,
        'Close': c_end.values,
        'Price': c_end.values,
        'Return_Window': stock_rets.values,
        'Alpha_Delta': (stock_rets - bench_ret_pct).values,
        'Down_Day_Win_Rate': down_day_win_rate.values,
        'Dist_52W_High': dist_high.values,
        'Volume_Mult': vol_mult.values,
        'Turnover_Cr': turnover_cr.values,
        'Has_Corporate_Split': has_corporate_split.values,
        'RS_Days_Count': rs_days_count.values,
        'RS_Days_Pct': rs_days_pct.round(1).values,
        'Recent_20D_Count': c20.values,
        'Recent_20D_RS_Pct': p20.round(1).values,
        'Dist_21EMA': dist_21.round(2).values,
        'Dist_50SMA': dist_50.round(2).values,
        'Holding_21EMA_Zone': holding_21.values,
        'Actionable_21EMA_Pocket': actionable_pocket.values
    }).dropna(subset=['Close', 'Return_Window', 'Dist_52W_High'])

    # Scrub corporate action reverse splits / consolidation jumps (e.g. CCAVENUE)
    df = df[~df['Has_Corporate_Split']]

    # Fill volume mult if missing
    df['Volume_Mult'] = df['Volume_Mult'].fillna(1.0)
    df['Turnover_Cr'] = df['Turnover_Cr'].fillna(0.0)

    # Clean ticker display
    df['Clean_Ticker'] = df['Ticker'].apply(lambda t: t.replace('.NS', '').replace('.BO', ''))

    # Minimum liquidity and price floor
    df = df[df['Close'] >= min_price]
    min_to = 2.0 if not is_us else 3.0
    df = df[df['Turnover_Cr'] >= min_to]

    # Map Industry Intelligence
    try:
        from industry_group_engine import get_stock_group_intelligence
        intel = get_stock_group_intelligence(universe_scope="US (S&P 500 / Nasdaq)" if is_us else "India (NSE/BSE)")
    except Exception:
        intel = {}

    try:
        from database import get_all_fundamentals_cache
        fc = get_all_fundamentals_cache()
    except Exception:
        fc = {}

    us_gics = {}
    if is_us and os.path.exists('data/sp500_gics_map.json'):
        try:
            import json
            with open('data/sp500_gics_map.json', 'r') as f:
                us_gics = json.load(f)
        except Exception:
            pass

    ind_list = []
    grd_list = []
    role_list = []
    for _, r in df.iterrows():
        ct = r['Clean_Ticker']
        rt = r['Ticker']
        info = intel.get(ct) or intel.get(rt)
        f_info = fc.get(rt) or fc.get(ct) or {}
        if info:
            ind_list.append(info.get('group_name', 'General'))
            grd_list.append(info.get('group_grade_badge', '—'))
            role_list.append(info.get('stock_role', '—'))
        elif ct in us_gics:
            ind_list.append(us_gics[ct].get('industry', 'General'))
            grd_list.append('—')
            role_list.append('—')
        elif f_info and f_info.get('industry'):
            ind_list.append(f_info.get('industry', 'General'))
            grd_list.append('—')
            role_list.append('—')
        else:
            ind_list.append('General')
            grd_list.append('—')
            role_list.append('—')

    df['Industry'] = ind_list
    df['Ind_Grade'] = grd_list
    df['Sister_Role'] = role_list

    # Map Point-in-Time CANSLIM RS Rating as of horizon end date
    try:
        rs_series = get_point_in_time_rs_series(t_end.strftime('%Y-%m-%d'), market=mkt_tag)
        df['RS_Rating'] = df['Ticker'].map(rs_series)
        df['RS_Rating'] = df['RS_Rating'].fillna(df['Clean_Ticker'].map(rs_series))
        
        # Fallback to fundamentals cache or group RS if missing
        missing_mask = df['RS_Rating'].isna()
        if missing_mask.any():
            for idx in df[missing_mask].index:
                tkr = df.loc[idx, 'Ticker']
                ctkr = df.loc[idx, 'Clean_Ticker']
                f_val = fc.get(tkr, {}).get('rs_score') or fc.get(ctkr, {}).get('rs_score')
                if f_val is not None and not pd.isna(f_val):
                    df.loc[idx, 'RS_Rating'] = float(f_val)
            df['RS_Rating'] = df['RS_Rating'].fillna(50.0)
        df['RS_Rating'] = df['RS_Rating'].round(1)
    except Exception as e:
        print(f"[DownDay Engine] Horizon RS map error: {e}")
        df['RS_Rating'] = 50.0

    # Map Cumulative Beach Ball Count
    thrust_map = get_cumulative_thrust_map(market=market, lookback_days=horizon_days * 2)
    df['Beach_Ball_Count'] = df['Clean_Ticker'].apply(lambda t: thrust_map.get(t, {}).get('count_3m', 0))
    df['BB_Badge'] = df['Clean_Ticker'].apply(lambda t: thrust_map.get(t, {}).get('badge', '—'))

    # Horizon-Calibrated Persistence Thresholds
    if horizon_days <= 21:
        core_thresh, elite_thresh = 61.9, 71.4
    elif horizon_days <= 63:
        core_thresh, elite_thresh = 58.7, 63.5
    else:
        core_thresh, elite_thresh = 56.3, 60.3

    df['RS_Days_Label'] = df.apply(lambda r: f"{r['RS_Days_Pct']:.1f}% ({int(r['RS_Days_Count'])}/{total_h_bars})", axis=1)
    df['Recent_20D_Label'] = df.apply(lambda r: f"{r['Recent_20D_RS_Pct']:.1f}% ({int(r['Recent_20D_Count'])}/{max(1, len(recent_b_rets))})", axis=1)
    df['Persistence_Momentum'] = (df['Recent_20D_RS_Pct'] - df['RS_Days_Pct']).round(1)

    def get_ma_status(row):
        d21 = row['Dist_21EMA']
        h21 = bool(row['Holding_21EMA_Zone'])
        d50 = row.get('Dist_50SMA', np.nan)
        if pd.isna(d21):
            return "❌ Violated Trend"
        if not h21:
            if d21 >= -2.0 and (pd.isna(d50) or d50 < 0.0):
                return "⚠️ Below 50 SMA (Overhead Supply)"
            return "❌ Violated Trend"
        if d21 >= 5.0:
            return "🚀 Power Extended (>+5%)"
        elif 3.0 < d21 < 5.0:
            return "🛡️ 21 EMA Cushion (+3-5%)"
        elif 0.0 <= d21 <= 3.0:
            return "🟢 21 EMA Pullback Pocket (0-3%)"
        else:
            return "🧲 21 EMA Test / Shakeout (-2 to 0%)"

    df['MA_Status'] = df.apply(get_ma_status, axis=1)

    # Exclude ETFs / Non-Equities from universe
    df = df[~df['Clean_Ticker'].apply(is_etf_or_non_equity)].copy()

    def get_rai_tier(pct):
        if pct >= elite_thresh:
            return "🔥 Elite Super Leader"
        elif pct >= core_thresh:
            return "🟢 Rai Core Leader"
        elif pct >= 45.0:
            return "⚪ Neutral Market Drift"
        else:
            return "⚠️ Relative Laggard"

    df['Rai_Tier'] = df['RS_Days_Pct'].apply(get_rai_tier)

    # 4-Pillar Unified Apex Horizon Score (0-100)
    alpha_norm = np.clip(df['Alpha_Delta'] / 50.0, 0.0, 1.0) * 100.0
    rai_norm = np.clip((df['RS_Days_Pct'] - 35.0) / (75.0 - 35.0), 0.0, 1.0) * 100.0
    win_norm = np.clip(df['Down_Day_Win_Rate'], 0.0, 100.0)
    rs_norm = np.clip(df['RS_Rating'], 0.0, 99.0)

    base_score = (0.30 * rai_norm) + (0.25 * alpha_norm) + (0.25 * win_norm) + (0.20 * rs_norm)
    bonus = np.where(df['Holding_21EMA_Zone'], 5.0, 0.0)
    df['Apex_Horizon_Score'] = np.clip(base_score + bonus, 0.0, 100.0).round(1)

    # Filter for Decoupled / Counter-Trend Leaders:
    # Must outperform benchmark (Alpha_Delta > 0.0) AND have either positive return OR Alpha_Delta >= 5.0%
    divergent = df[(df['Alpha_Delta'] > 0.0) & ((df['Return_Window'] > 0.0) | (df['Alpha_Delta'] >= 5.0))].copy()

    # Sort by Apex Horizon Score descending by default, then Alpha Delta
    divergent = divergent.sort_values(['Apex_Horizon_Score', 'Alpha_Delta', 'RS_Days_Pct'], ascending=[False, False, False])

    # Summary Stats
    target_bars = int(np.ceil((core_thresh / 100.0) * total_h_bars))
    rai_persistent_count = int(((df['Return_Window'] > 0.0) & (df['RS_Days_Pct'] >= core_thresh) & (df['RS_Rating'] >= 80)).sum())
    apex_generals_count = int(((df['Return_Window'] > 0.0) & (df['RS_Days_Pct'] >= core_thresh) & (df['RS_Rating'] >= 80) & (df['Holding_21EMA_Zone']) & (df['Alpha_Delta'] >= 15.0)).sum())

    bench_summary = {
        'bench_name': bench_name,
        'start_date': t_start.strftime('%Y-%m-%d'),
        'end_date': t_end.strftime('%Y-%m-%d'),
        'horizon_bars': total_h_bars,
        'bench_return_pct': round(bench_ret_pct, 2),
        'bench_max_dd_pct': round(b_max_dd, 2),
        'total_down_days': total_down_days,
        'outperforming_count': int((df['Alpha_Delta'] > 0.0).sum()),
        'total_universe': len(df),
        'positive_absolute_count': int((df['Return_Window'] > 0.0).sum()),
        'calibrated_rai_threshold': core_thresh,
        'calibrated_rai_target_bars': target_bars,
        'rai_persistent_count': rai_persistent_count,
        'apex_generals_count': apex_generals_count
    }

    # Group-Level Alpha Aggregates
    if not divergent.empty:
        group_stats = divergent.groupby('Industry').agg(
            Avg_Alpha=('Alpha_Delta', 'mean'),
            Avg_RS_Days=('RS_Days_Pct', 'mean'),
            Leader_Count=('Clean_Ticker', 'count'),
            Top_Stock=('Clean_Ticker', 'first'),
            Top_Stock_Return=('Return_Window', 'first')
        ).reset_index()
        group_stats['Avg_Alpha'] = group_stats['Avg_Alpha'].round(1)
        group_stats['Avg_RS_Days'] = group_stats['Avg_RS_Days'].round(1)
        group_stats = group_stats[group_stats['Leader_Count'] >= 2].sort_values(['Leader_Count', 'Avg_Alpha'], ascending=[False, False])
    else:
        group_stats = pd.DataFrame()

    return divergent, bench_summary, group_stats

@st.cache_data(ttl=1800, show_spinner=False)
def compute_multi_day_resilience_cluster(
    dates_list: list,
    market: str = 'INDIA',
    min_price: float = 20.0,
    max_dist_high: float = 20.0
) -> tuple:
    """
    Aggregates institutional down-day resilience across a cluster of multiple selected down days.
    
    Returns:
      (df_cluster_leaders, cluster_summary)
    """
    if dates_list is None:
        return pd.DataFrame(), {}
    if isinstance(dates_list, pd.DataFrame):
        dates_list = dates_list['date'].tolist() if 'date' in dates_list.columns else (dates_list['Date'].tolist() if 'Date' in dates_list.columns else dates_list.iloc[:, 0].tolist())
    elif isinstance(dates_list, (pd.Series, pd.Index)):
        dates_list = dates_list.tolist()
    if len(dates_list) == 0:
        return pd.DataFrame(), {}
        
    mkt_tag = 'US' if market.upper() in ['US', 'USA'] else 'INDIA'
    sorted_dates = sorted(dates_list)
    n_days = len(sorted_dates)
    
    dfs = []
    for d in sorted_dates:
        df_d = compute_down_day_resilience(d, market=mkt_tag, min_price=min_price, max_dist_high=max_dist_high)
        if not df_d.empty:
            df_d = df_d.copy()
            df_d['Session_Date'] = d
            dfs.append(df_d)
            
    if not dfs:
        return pd.DataFrame(), {}
        
    all_df = pd.concat(dfs, ignore_index=True)

    # Defensive schema check: guarantee all required aggregation columns exist
    for col, default_val in [
        ('Alpha_Score', 50.0),
        ('RS_Rating', 50.0),
        ('Return_1D', 0.0),
        ('Archetype', '🟢 Green on Red'),
        ('Industry', 'Broad Market'),
        ('Ind_Grade', '—'),
        ('Sister_Role', '—'),
        ('Close', 0.0),
        ('Turnover_Cr', 0.0),
        ('Dist_52W_High', 0.0),
        ('Volume_Mult', 1.0)
    ]:
        if col not in all_df.columns:
            if col == 'Alpha_Score' and 'Resilience_Score' in all_df.columns:
                all_df['Alpha_Score'] = all_df['Resilience_Score']
            elif col == 'Alpha_Score' and 'resilience_score' in all_df.columns:
                all_df['Alpha_Score'] = all_df['resilience_score']
            else:
                all_df[col] = default_val
    
    # Aggregate by ticker
    def summarize_archetypes(s):
        parts = []
        n_bb = int((s.str.contains('Beach Ball')).sum())
        n_52 = int((s.str.contains('52W High')).sum())
        n_gr = int((s.str.contains('Green on Red')).sum())
        if n_bb > 0: parts.append(f"🏖️ {n_bb}x BB")
        if n_52 > 0: parts.append(f"🚀 {n_52}x 52W")
        if n_gr > 0: parts.append(f"🟢 {n_gr}x Green")
        return " • ".join(parts) if parts else "🟢 Green on Red"

    agg = all_df.groupby('Clean_Ticker').agg(
        Green_Days_Count=('Session_Date', 'nunique'),
        Avg_Return_Down=('Return_1D', 'mean'),
        Avg_Alpha=('Alpha_Score', 'mean'),
        Beach_Ball_Plays=('Archetype', lambda s: int((s.str.contains('Beach Ball')).sum())),
        High_52W_Plays=('Archetype', lambda s: int((s.str.contains('52W High')).sum())),
        Archetype_Footprint=('Archetype', summarize_archetypes),
        RS_Rating=('RS_Rating', 'max'),
        Industry=('Industry', 'last'),
        Ind_Grade=('Ind_Grade', 'last'),
        Sister_Role=('Sister_Role', 'last'),
        Close=('Close', 'last'),
        Turnover_Cr=('Turnover_Cr', 'last'),
        Dist_52W_High=('Dist_52W_High', 'last'),
        Volume_Mult=('Volume_Mult', 'mean')
    ).reset_index()
    
    agg['Total_Down_Days'] = n_days
    agg['Hit_Rate_Pct'] = (agg['Green_Days_Count'] / n_days) * 100.0
    agg['Hit_Rate_Label'] = agg.apply(lambda r: f"{int(r['Green_Days_Count'])} / {n_days} ({r['Hit_Rate_Pct']:.0f}%)", axis=1)
    
    # Calculate Cluster Return from earliest date to latest date
    matrix = load_price_matrix()
    if matrix is not None:
        try:
            close_panel = matrix.xs('Close', level=1, axis=1)
            t_first = pd.to_datetime(sorted_dates[0])
            t_last = pd.to_datetime(sorted_dates[-1])
            avail = close_panel.index.sort_values()
            v_start = avail[avail <= t_first]
            v_end = avail[avail <= t_last]
            if len(v_start) >= 2 and len(v_end) >= 1:
                idx_start = close_panel.index.get_loc(v_start[-1])
                idx_prev = max(0, idx_start - 1)
                p_base = close_panel.iloc[idx_prev]
                p_cur = close_panel.loc[v_end[-1]]
                c_ret = ((p_cur / p_base) - 1.0) * 100.0
                
                # Clean ticker map
                clean_ret_map = {}
                for k, v in c_ret.dropna().items():
                    c_k = k.replace('.NS', '').replace('.BO', '')
                    if isinstance(v, (pd.Series, np.ndarray)):
                        v = v.iloc[0] if len(v) > 0 else np.nan
                    if not pd.isna(v):
                        clean_ret_map[c_k] = round(float(v), 2)
                    
                agg['Cluster_Gain'] = agg['Clean_Ticker'].map(clean_ret_map).fillna(agg['Avg_Return_Down'])
            else:
                agg['Cluster_Gain'] = agg['Avg_Return_Down']
        except Exception as e:
            print(f"[DownDay Engine] Cluster return calc note: {e}")
            agg['Cluster_Gain'] = agg['Avg_Return_Down']
    else:
        agg['Cluster_Gain'] = agg['Avg_Return_Down']
        
    agg = agg.sort_values(['Hit_Rate_Pct', 'Avg_Alpha', 'RS_Rating', 'Cluster_Gain'], ascending=[False, False, False, False])
    
    # Cluster summary
    summary = {
        'total_down_days': n_days,
        'earliest_date': sorted_dates[0],
        'latest_date': sorted_dates[-1],
        'dates_list': sorted_dates,
        'total_candidates': len(agg),
        'perfect_hit_count': int((agg['Hit_Rate_Pct'] >= 80.0).sum()),
        'rs80_hit_count': int(((agg['Hit_Rate_Pct'] >= 60.0) & (agg['RS_Rating'] >= 80)).sum()),
        'apex_bb_count': int((agg['Beach_Ball_Plays'] >= 2).sum())
    }
    
    return agg, summary


# -----------------------------------------------------------------------------
# AMEET RAI 62.5% RS DAYS PERSISTENCE & APEX DUAL TOPPER CROSS-REFERENCE ENGINE
# -----------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def compute_rolling_rs_days_matrix(
    market: str = 'INDIA',
    windows: tuple = (15, 20, 40),
    _cache_v: str = "v2_clean_rai_liquidity"
) -> dict:
    """
    Vectorized computation of Ameet Rai Relative Strength Days Persistence.
    Evaluates every trading session across rolling windows to measure the percentage
    of sessions where stock return exceeded benchmark return (R_stock > R_bench).
    Enforces 20D average turnover and unadjusted split jump detection.
    """
    matrix = load_price_matrix()
    if matrix is None:
        return {}

    is_us = market.upper() in ['US', 'USA']
    mkt_tag = 'US' if is_us else 'INDIA'

    try:
        close_panel = matrix.xs('Close', level=1, axis=1)
        vol_panel = matrix.xs('Volume', level=1, axis=1) if ('Volume' in matrix.columns.levels[1] if hasattr(matrix.columns, 'levels') else 'Volume' in matrix.columns.get_level_values(1)) else None
    except Exception as e:
        print(f"[Ameet Rai Engine] Panel extraction error: {e}")
        return {}

    if is_us:
        tickers = [c for c in close_panel.columns if not c.endswith('.NS') and not c.endswith('.BO') and not c.startswith('^')]
    else:
        tickers = [c for c in close_panel.columns if c.endswith(('.NS', '.BO')) and not c.startswith('^')]

    if not tickers:
        return {}

    sub_close = close_panel[tickers]
    clean_close = get_clean_active_trading_panel(sub_close, is_us=is_us)
    if clean_close.empty or len(clean_close) < max(windows) + 2:
        return {}

    if vol_panel is not None:
        vol_cols = [c for c in tickers if c in vol_panel.columns]
        clean_vol = vol_panel[vol_cols].reindex(index=clean_close.index, columns=clean_close.columns)
    else:
        clean_vol = None

    bench_series, bench_name = get_cached_official_benchmark(market=mkt_tag)
    if bench_series is None or len(bench_series) < 10:
        bench_series = clean_close.median(axis=1)

    common_dates = clean_close.index.intersection(bench_series.index)
    if len(common_dates) < max(windows) + 2:
        return {}

    clean_close = clean_close.loc[common_dates]
    bench_window = bench_series.loc[common_dates]

    stock_rets = clean_close.pct_change()
    bench_rets = bench_window.pct_change()

    # Boolean outperformance: True if stock outpaced benchmark on session t
    outperf = stock_rets.gt(bench_rets, axis=0).astype(float)

    results = {}
    for w in windows:
        min_p = max(5, w // 2)
        results[f'pct_{w}d'] = (outperf.rolling(window=w, min_periods=min_p).mean() * 100.0).iloc[-1]
        results[f'cnt_{w}d'] = (outperf.rolling(window=w, min_periods=min_p).sum()).iloc[-1]

    # Technical MAs
    latest_close = clean_close.iloc[-1]
    results['ema_21'] = clean_close.ewm(span=21, adjust=False).mean().iloc[-1]
    results['sma_50'] = clean_close.rolling(window=50, min_periods=35).mean().iloc[-1]
    results['latest_close'] = latest_close

    # Institutional Turnover & Split Detection
    if clean_vol is not None:
        vol_20d_avg = clean_vol.tail(20).mean(axis=0)
        turnover_20d = (latest_close * vol_20d_avg) / (1e6 if is_us else 1e7)
    else:
        turnover_20d = pd.Series(999.0, index=clean_close.columns)

    has_split_jump = (stock_rets.tail(45).abs() > 0.85).any(axis=0).fillna(False)

    results['turnover_20d'] = turnover_20d
    results['has_split_jump'] = has_split_jump

    return results


@st.cache_data(ttl=1800, show_spinner=False)
def compute_ameet_rai_persistence_universe(
    market: str = 'INDIA',
    min_price: float = 20.0,
    _cache_v: str = "v2_clean_rai_liquidity"
) -> pd.DataFrame:
    """
    Computes universe-wide Ameet Rai RS Days persistence and structures it for
    Tab 3 of Beach Ball & Down-Day Resilience Terminal.
    Enforces institutional liquidity (Turnover >= 2 Cr / $3M), split hygiene, and ETF exclusion.
    """
    mkt_tag = 'US' if market.upper() in ['US', 'USA'] else 'INDIA'
    is_us = mkt_tag == 'US'

    rai_matrix = compute_rolling_rs_days_matrix(market=mkt_tag, windows=(15, 20, 40))
    if not rai_matrix:
        return pd.DataFrame()

    pct_15 = rai_matrix.get('pct_15d', pd.Series(dtype=float))
    cnt_15 = rai_matrix.get('cnt_15d', pd.Series(dtype=float))
    pct_20 = rai_matrix.get('pct_20d', pd.Series(dtype=float))
    cnt_20 = rai_matrix.get('cnt_20d', pd.Series(dtype=float))
    pct_40 = rai_matrix.get('pct_40d', pd.Series(dtype=float))
    cnt_40 = rai_matrix.get('cnt_40d', pd.Series(dtype=float))
    ema_21 = rai_matrix.get('ema_21', pd.Series(dtype=float))
    sma_50 = rai_matrix.get('sma_50', pd.Series(dtype=float))
    closes = rai_matrix.get('latest_close', pd.Series(dtype=float))
    turnover_series = rai_matrix.get('turnover_20d', pd.Series(dtype=float))
    split_series = rai_matrix.get('has_split_jump', pd.Series(dtype=bool))
    min_turnover = 3.0 if is_us else 2.0

    today_str = datetime.now().strftime('%Y-%m-%d')
    rs_series = get_point_in_time_rs_series(today_str, market=mkt_tag)

    try:
        from industry_group_engine import get_stock_group_intelligence
        intel = get_stock_group_intelligence(taxonomy="canonical", universe_scope="US (S&P 500 / Nasdaq)" if is_us else "India (NSE/BSE)")
    except Exception:
        intel = {}

    us_gics = {}
    if is_us and os.path.exists('data/sp500_gics_map.json'):
        try:
            import json
            with open('data/sp500_gics_map.json', 'r') as f:
                us_gics = json.load(f)
        except Exception:
            pass

    records = []
    for raw_t in closes.index:
        clean_t = raw_t.replace('.NS', '').replace('.BO', '')
        if raw_t.startswith('^') or is_etf_or_non_equity(clean_t):
            continue
        if bool(split_series.get(raw_t, False)) or bool(split_series.get(clean_t, False)):
            continue

        c = float(closes.get(raw_t, 0.0))
        if c < min_price or pd.isna(c):
            continue

        t_cr = float(turnover_series.get(raw_t, turnover_series.get(clean_t, 0.0)))
        if pd.isna(t_cr) or t_cr < min_turnover:
            continue

        e21 = float(ema_21.get(raw_t, 0.0))
        s50 = float(sma_50.get(raw_t, 0.0))

        dist_21 = ((c - e21) / e21) * 100.0 if e21 > 0 else 0.0
        dist_50 = ((c - s50) / s50) * 100.0 if s50 > 0 else 0.0

        p15 = round(float(pct_15.get(raw_t, 0.0)), 1)
        c15 = int(cnt_15.get(raw_t, 0))
        p20 = round(float(pct_20.get(raw_t, 0.0)), 1)
        c20 = int(cnt_20.get(raw_t, 0))
        p40 = round(float(pct_40.get(raw_t, 0.0)), 1)
        c40 = int(cnt_40.get(raw_t, 0))

        rs_val = float(rs_series.get(clean_t, rs_series.get(raw_t, 70.0)) or 70.0)

        if is_us and clean_t in us_gics:
            ind_info = us_gics[clean_t]
            ind_name = ind_info.get('industry', 'Broad Market')
            ind_badge = ind_info.get('sector', 'US Equities')
            grd = 'B'
            rnk = 50
            is_top_20 = False
        elif clean_t in intel or (clean_t + '.NS') in intel:
            info = intel.get(clean_t) or intel.get(clean_t + '.NS', {})
            ind_name = info.get('group_name', info.get('industry', 'Broad Market'))
            grd = info.get('group_grade', info.get('grade', 'B'))
            rnk = info.get('group_rank', info.get('rank', 99))
            is_top_20 = bool(info.get('is_top_20_pct', grd in ['A+', 'A'] or rnk <= 22))
            ind_badge = f"{ind_name} ({grd} #{rnk})"
        else:
            ind_name = 'Broad Market'
            ind_badge = 'General Market'
            grd = 'C'
            rnk = 99
            is_top_20 = False

        if p20 >= 75.0:
            tier_badge = "🔥 Elite Super Leader (≥75%)"
        elif p20 >= 62.5:
            tier_badge = "🟢 Rai Core Leader (≥62.5%)"
        elif p20 >= 45.0:
            tier_badge = "⚪ Neutral Market Drift"
        else:
            tier_badge = "⚠️ Relative Laggard"

        holding_21 = bool(dist_21 >= -2.0 and c >= s50)

        if not holding_21:
            if dist_21 >= -2.0 and c < s50:
                ma_status = "⚠️ Below 50 SMA (Overhead Supply)"
            else:
                ma_status = "❌ Violated Trend"
        elif dist_21 >= 5.0:
            ma_status = "🚀 Power Extended (>+5%)"
        elif 3.0 < dist_21 < 5.0:
            ma_status = "🛡️ 21 EMA Cushion (+3-5%)"
        elif 0.0 <= dist_21 <= 3.0:
            ma_status = "🟢 21 EMA Pullback Pocket (0-3%)"
        else:
            ma_status = "🧲 21 EMA Test / Shakeout (-2 to 0%)"

        records.append({
            'Ticker': clean_t,
            'Raw_Ticker': raw_t,
            'Price': round(c, 2),
            'Turnover_20D': round(t_cr, 2),
            'Industry': ind_name,
            'Industry_Grade': grd,
            'Group_Rank': rnk,
            'Is_Top_20_Group': is_top_20,
            'Industry_Badge': ind_badge,
            'RS_Days_15D_Pct': p15,
            'RS_Days_15D_Count': c15,
            'RS_Days_15D_Label': f"{p15:.1f}% ({c15}/15)",
            'RS_Days_20D_Pct': p20,
            'RS_Days_20D_Count': c20,
            'RS_Days_20D_Label': f"{p20:.1f}% ({c20}/20)",
            'RS_Days_40D_Pct': p40,
            'RS_Days_40D_Count': c40,
            'RS_Days_40D_Label': f"{p40:.1f}% ({c40}/40)",
            'RS_Rating': round(rs_val, 1),
            'Dist_21EMA': round(dist_21, 2),
            'Dist_50SMA': round(dist_50, 2),
            'Holding_21EMA_Zone': holding_21,
            'Tier_Badge': tier_badge,
            'MA_Status': ma_status
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    return df.sort_values(by=['RS_Days_20D_Pct', 'RS_Rating', 'Dist_21EMA'], ascending=[False, False, False])


@st.cache_data(ttl=1800, show_spinner=False)
def compute_apex_dual_toppers(
    df_cluster: pd.DataFrame,
    df_rai: pd.DataFrame,
    market: str = 'INDIA'
) -> pd.DataFrame:
    """
    Computes the permanent Apex Dual Toppers cross-reference table.
    Fuses Beach Ball down-day shock resilience with Ameet Rai continuous persistence.
    """
    if df_cluster.empty or df_rai.empty:
        return pd.DataFrame()

    rai_map = df_rai.set_index('Ticker').to_dict(orient='index')

    merged_records = []
    for _, r in df_cluster.iterrows():
        t = str(r.get('Clean_Ticker', ''))
        if not t or is_etf_or_non_equity(t):
            continue
        rai_info = rai_map.get(t, {})
        if not rai_info:
            continue

        hit_pct = float(r.get('Hit_Rate_Pct', 0.0))
        rai_20d_pct = float(rai_info.get('RS_Days_20D_Pct', 0.0))
        rai_40d_pct = float(rai_info.get('RS_Days_40D_Pct', 0.0))
        rs_rating = float(r.get('RS_Rating', rai_info.get('RS_Rating', 75.0)))
        dist_21 = float(rai_info.get('Dist_21EMA', 0.0))
        dist_50 = float(rai_info.get('Dist_50SMA', 0.0))
        holding_21 = bool(rai_info.get('Holding_21EMA_Zone', False))

        apex_score = (hit_pct * 0.40) + (rai_20d_pct * 0.40) + (rs_rating * 0.20)
        if holding_21:
            apex_score += 5.0

        if hit_pct >= 80.0 and rai_20d_pct >= 65.0 and holding_21:
            verdict = f"👑 Apex Dual Topper ({r.get('Hit_Rate_Label', '')} + {rai_20d_pct:.0f}%)"
            is_apex = True
        elif hit_pct >= 60.0 and rai_20d_pct >= 60.0 and holding_21:
            verdict = f"🟢 Dual Persistent Leader ({r.get('Hit_Rate_Label', '')} + {rai_20d_pct:.0f}%)"
            is_apex = True
        elif hit_pct >= 80.0 and rai_20d_pct < 60.0:
            verdict = "🌊 Washout Specialist"
            is_apex = False
        elif rai_20d_pct >= 70.0 and hit_pct < 60.0:
            verdict = "📈 Steady Climber"
            is_apex = False
        else:
            verdict = "⚪ Emerging Runner"
            is_apex = False

        if is_apex:
            is_top_20_apex = bool(
                rai_info.get('Is_Top_20_Group', False) or 
                rai_info.get('Industry_Grade') in ['A+', 'A'] or 
                r.get('Ind_Grade') in ['A+', 'A'] or
                '(A+' in str(rai_info.get('Industry_Badge', '')) or
                '(A #' in str(rai_info.get('Industry_Badge', ''))
            )
            merged_records.append({
                'Ticker': t,
                'Price': r.get('Latest_Price', rai_info.get('Price', 0.0)),
                'Industry_Badge': rai_info.get('Industry_Badge', r.get('Industry', 'Broad Market')),
                'Industry_Grade': rai_info.get('Industry_Grade', r.get('Ind_Grade', 'B')),
                'Is_Top_20_Group': is_top_20_apex,
                'Beach_Ball_Hit': r.get('Hit_Rate_Label', ''),
                'Hit_Rate_Pct': hit_pct,
                'Rai_20D_Label': rai_info.get('RS_Days_20D_Label', ''),
                'Rai_20D_Pct': rai_20d_pct,
                'Rai_40D_Label': rai_info.get('RS_Days_40D_Label', ''),
                'Rai_40D_Pct': rai_40d_pct,
                'RS_Rating': rs_rating,
                'Dist_21EMA': dist_21,
                'Dist_50SMA': dist_50,
                'Apex_Score': round(apex_score, 1),
                'Institutional_Verdict': verdict,
                'Archetype': r.get('Archetype_Footprint', 'Beach Ball Leader'),
                'Volume_Mult': r.get('Volume_Mult', 1.0)
            })

    if not merged_records:
        return pd.DataFrame()

    df_apex = pd.DataFrame(merged_records)
    return df_apex.sort_values(by=['Apex_Score', 'Hit_Rate_Pct', 'Rai_20D_Pct'], ascending=[False, False, False])


