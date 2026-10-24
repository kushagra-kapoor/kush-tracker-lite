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
import sqlite3
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
    save_down_day_resilience_events,
    get_saved_down_day_resilience_events,
    get_available_resilience_dates
)

MATRIX_PATH = 'historical_prices_matrix.pkl'

@st.cache_data(ttl=3600, show_spinner=False)
def load_price_matrix():
    """Loads the cached multi-ticker price matrix."""
    if not os.path.exists(MATRIX_PATH):
        return None
    try:
        with open(MATRIX_PATH, 'rb') as f:
            return pickle.load(f)
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
    conn = get_connection()
    mkt_code = 'US' if market.upper() in ['US', 'USA'] else 'IN'
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    query = """
    SELECT *
    FROM market_breadth_daily
    WHERE market = ? AND date >= ?
    ORDER BY date DESC
    """
    try:
        df_breadth = pd.read_sql_query(query, conn, params=(mkt_code, cutoff))
    except Exception as e:
        print(f"[DownDay Engine] Error querying breadth history: {e}")
        df_breadth = pd.DataFrame()
    finally:
        conn.close()

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
    """
    is_us = market.upper() in ['US', 'USA']
    matrix = load_price_matrix()
    if matrix is None:
        return pd.Series(dtype=float)
    try:
        close_panel = matrix.xs('Close', level=1, axis=1)
    except Exception:
        return pd.Series(dtype=float)

    target_dt = pd.to_datetime(target_date_str)
    valid_dates = close_panel.index[close_panel.index <= target_dt]
    if len(valid_dates) < 2:
        return pd.Series(dtype=float)

    t_idx = close_panel.index.get_loc(valid_dates[-1])
    c_t = close_panel.iloc[t_idx]
    c_1m = close_panel.iloc[max(0, t_idx - 21)]
    c_3m = close_panel.iloc[max(0, t_idx - 63)]
    c_6m = close_panel.iloc[max(0, t_idx - 126)]

    r_1m = (c_t / c_1m.replace(0, np.nan)) - 1.0
    r_3m = (c_t / c_3m.replace(0, np.nan)) - 1.0
    r_6m = (c_t / c_6m.replace(0, np.nan)) - 1.0

    weighted_raw = (r_1m * 0.40) + (r_3m * 0.35) + (r_6m * 0.25)

    if is_us:
        mask = ~close_panel.columns.str.endswith('.NS') & ~close_panel.columns.str.endswith('.BO')
    else:
        mask = close_panel.columns.str.endswith('.NS')

    sub_raw = weighted_raw[mask].dropna()
    if sub_raw.empty:
        return pd.Series(dtype=float)

    # CANSLIM 1 to 99 percentile ranking
    rs_ranked = (sub_raw.rank(pct=True) * 98.0 + 1.0).round(1)
    return rs_ranked

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
        tickers = [c for c in close_panel.columns if c.endswith('.NS')]

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
    mkt_tag = 'US' if market.upper() in ['US', 'USA'] else 'INDIA'
    
    cutoff_3m = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    cutoff_1m = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    result = {}
    try:
        # Check beach_ball_events
        c.execute('''
            SELECT ticker, 
                   COUNT(CASE WHEN event_date >= ? THEN 1 END) as c_3m,
                   COUNT(CASE WHEN event_date >= ? THEN 1 END) as c_1m
            FROM beach_ball_events
            WHERE market = ? AND event_date >= ?
            GROUP BY ticker
        ''', (cutoff_3m, cutoff_1m, mkt_tag, cutoff_3m))
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
def get_cached_official_benchmark(market: str = 'INDIA') -> tuple:
    """
    Downloads or fetches the official broad market index benchmark.
    India: ^CRSLDX (NIFTY 500) with fallback to ^NSEI (NIFTY 50)
    US: ^GSPC (S&P 500) with fallback to SPY
    """
    is_us = market.upper() in ['US', 'USA']
    symbols = ['^GSPC', 'SPY'] if is_us else ['^CRSLDX', '^NSEI']
    
    for s in symbols:
        try:
            df = yf.download(s, period='1y', progress=False)
            if df is not None and not df.empty:
                c = df['Close']
                if hasattr(c, 'iloc') and len(c.shape) > 1:
                    c = c.iloc[:, 0]
                c = c.dropna()
                if len(c) >= 21:
                    c.index = pd.to_datetime(c.index)
                    name = "S&P 500 (^GSPC)" if s == '^GSPC' else ("SPY (S&P 500 ETF)" if s == 'SPY' else ("NIFTY 500 (^CRSLDX)" if s == '^CRSLDX' else "NIFTY 50 (^NSEI)"))
                    return c, name
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
            tickers = [c for c in close_panel.columns if c.endswith('.NS')]
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
    min_price: float = 20.0
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
        tickers = [c for c in close_panel.columns if c.endswith('.NS') and not c.startswith('^')]

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
        if len(common_dates) >= horizon_days:
            clean_close = clean_close.loc[common_dates]
            clean_vol = clean_vol.loc[common_dates]
            b_window = bench_series.loc[common_dates].tail(horizon_days)
        else:
            b_window = bench_series.tail(horizon_days)
    else:
        # Fallback to holiday-scrubbed equal-weight median index
        bench_series = clean_close.median(axis=1)
        bench_name = "NIFTY Broad Market Index (Equal-Weight)" if not is_us else "US Broad Market Index (Equal-Weight)"
        b_window = bench_series.tail(horizon_days)

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

    # Down-Day Hit Rate (% of days stock was GREEN when Benchmark was RED)
    if total_down_days > 0 and len(down_day_dates) > 0:
        down_stock_rets = window_daily.loc[window_daily.index.isin(down_dates if 'down_dates' in locals() else down_day_dates)]
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
        'Return_Window': stock_rets.values,
        'Alpha_Delta': (stock_rets - bench_ret_pct).values,
        'Down_Day_Win_Rate': down_day_win_rate.values,
        'Dist_52W_High': dist_high.values,
        'Volume_Mult': vol_mult.values,
        'Turnover_Cr': turnover_cr.values,
        'Has_Corporate_Split': has_corporate_split.values
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
        df['RS_Rating'] = df['RS_Rating'].fillna(df['Clean_Ticker'].map(rs_series)).fillna(80.0).round(1)
    except Exception as e:
        print(f"[DownDay Engine] Horizon RS map error: {e}")
        df['RS_Rating'] = 80.0

    # Map Cumulative Beach Ball Count
    thrust_map = get_cumulative_thrust_map(market=market, lookback_days=horizon_days * 2)
    df['Beach_Ball_Count'] = df['Clean_Ticker'].apply(lambda t: thrust_map.get(t, {}).get('count_3m', 0))
    df['BB_Badge'] = df['Clean_Ticker'].apply(lambda t: thrust_map.get(t, {}).get('badge', '—'))

    # Filter for Decoupled / Counter-Trend Leaders:
    # Stocks with positive return OR Excess Alpha >= +5.0%
    divergent = df[(df['Return_Window'] > 0.0) | (df['Alpha_Delta'] >= 5.0)].copy()

    # Sort by Alpha Delta (Excess Return) descending
    divergent = divergent.sort_values(['Alpha_Delta', 'Down_Day_Win_Rate'], ascending=[False, False])

    # Summary Stats
    bench_summary = {
        'bench_name': bench_name,
        'start_date': t_start.strftime('%Y-%m-%d'),
        'end_date': t_end.strftime('%Y-%m-%d'),
        'horizon_bars': len(b_window),
        'bench_return_pct': round(bench_ret_pct, 2),
        'bench_max_dd_pct': round(b_max_dd, 2),
        'total_down_days': total_down_days,
        'outperforming_count': int((df['Alpha_Delta'] > 0.0).sum()),
        'total_universe': len(df),
        'positive_absolute_count': int((df['Return_Window'] > 0.0).sum())
    }

    # Group-Level Alpha Aggregates
    if not divergent.empty:
        group_stats = divergent.groupby('Industry').agg(
            Avg_Alpha=('Alpha_Delta', 'mean'),
            Leader_Count=('Clean_Ticker', 'count'),
            Top_Stock=('Clean_Ticker', 'first'),
            Top_Stock_Return=('Return_Window', 'first')
        ).reset_index()
        group_stats['Avg_Alpha'] = group_stats['Avg_Alpha'].round(1)
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
    if not dates_list:
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

