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
        "| Ticker | Archetype | 1D Gain % | Dist 52W High | Volume (x ADTV) | Industry & Grade | Sister Role |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for _, r in df_resilient.head(25).iterrows():
        t = r['Clean_Ticker']
        arch = r['Archetype']
        g = f"+{r['Return_1D']:.1f}%"
        dh = f"{r['Dist_52W_High']:.1f}%"
        vm = f"{r['Volume_Mult']:.1f}x"
        ind = f"{r['Industry']} ({r['Ind_Grade']})"
        role = r['Sister_Role']
        lines.append(f"| **{t}** | {arch} | {g} | {dh} | {vm} | {ind} | {role} |")
        
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
        "🏖️ *TOP RESILIENT LEADERS (Coiled Springs Holding Green)*:"
    ]
    for i, (_, r) in enumerate(df_resilient.head(7).iterrows()):
        t = r['Clean_Ticker']
        arch = r['Archetype'].split()[0] # Icon
        g = f"+{r['Return_1D']:.1f}%"
        vm = f"{r['Volume_Mult']:.1f}x Vol"
        dh = f"{r['Dist_52W_High']:.1f}% from ATH"
        ind = r['Industry']
        lines.append(f"{i+1}. *{t}* ({arch} {g}, {vm}, {dh}) — _{ind}_")
        
    lines.append("")
    lines.append(f"📋 *TradingView 1-Click Import*:\n`{tv_str}`")
    return "\n".join(lines)

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

@st.cache_data(ttl=1800, show_spinner=False)
def compute_multi_horizon_correction_analytics(
    market: str = 'INDIA',
    horizon_days: int = 21,
    min_price: float = 20.0
) -> tuple:
    """
    Multi-Horizon Macro Correction & Counter-Trend Divergence Engine:
    Evaluates market performance over the last N trading days (e.g. 1M / 21D, 3M / 63D, 6M / 126D).
    Identifies stocks that decoupled from broad market weakness and generated positive excess alpha.
    
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

    n_bars = len(close_panel)
    if n_bars < horizon_days + 1:
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

    # Time window indices
    start_idx = max(0, n_bars - horizon_days - 1)
    t_start = close_panel.index[start_idx]
    t_end = close_panel.index[-1]

    # Determine Benchmark Series with clean fallback
    bench_series = None
    bench_name = "NIFTY 500 (^CRSLDX)" if not is_us else "S&P 500 (^GSPC)"
    
    if not is_us:
        if '^CRSLDX' in close_panel.columns and close_panel['^CRSLDX'].dropna().index[-1] >= t_end:
            bench_series = close_panel['^CRSLDX'].dropna()
        elif 'NIFTYBEES.NS' in close_panel.columns and close_panel['NIFTYBEES.NS'].dropna().index[-1] >= t_end:
            bench_series = close_panel['NIFTYBEES.NS'].dropna()
            bench_name = "NIFTY 50 (NIFTYBEES)"
        else:
            bench_series = sub_close.median(axis=1).dropna()
            bench_name = "NIFTY Broad Market Index (Equal-Weight)"
    else:
        if '^GSPC' in close_panel.columns and close_panel['^GSPC'].dropna().index[-1] >= t_end:
            bench_series = close_panel['^GSPC'].dropna()
        elif 'SPY' in close_panel.columns and close_panel['SPY'].dropna().index[-1] >= t_end:
            bench_series = close_panel['SPY'].dropna()
            bench_name = "SPY (S&P 500 ETF)"
        else:
            bench_series = sub_close.median(axis=1).dropna()
            bench_name = "US Broad Market Index (Equal-Weight)"

    # Benchmark metrics over horizon
    b_window = bench_series.loc[t_start:t_end]
    if len(b_window) < 2:
        return pd.DataFrame(), {}, pd.DataFrame()
        
    bench_ret_pct = ((b_window.iloc[-1] / b_window.iloc[0]) - 1.0) * 100.0
    
    # Daily benchmark returns in window to identify down days
    b_daily_rets = b_window.pct_change().dropna()
    down_days_mask = b_daily_rets < 0.0
    down_day_dates = b_daily_rets[down_days_mask].index
    total_down_days = len(down_day_dates)

    # Benchmark Max Drawdown in window
    b_rolling_max = b_window.cummax()
    b_drawdown = ((b_window - b_rolling_max) / b_rolling_max) * 100.0
    b_max_dd = b_drawdown.min()

    # Stock metrics over horizon
    c_start = sub_close.loc[t_start]
    c_end = sub_close.loc[t_end]
    stock_rets = ((c_end / c_start) - 1.0) * 100.0
    
    # 52W High Proximity
    window_252 = sub_close.iloc[max(0, n_bars - 252):]
    high_252 = window_252.max()
    dist_high = ((c_end / high_252) - 1.0) * 100.0

    # 20D Volume Multiple
    vol_latest = sub_vol.loc[t_end]
    vol_20d_avg = sub_vol.iloc[max(0, n_bars - 21):-1].mean().replace(0, np.nan)
    vol_mult = vol_latest / vol_20d_avg

    # Down-Day Hit Rate (% of days stock was GREEN when Benchmark was RED)
    if total_down_days > 0 and len(down_day_dates) > 0:
        sub_window = sub_close.loc[t_start:t_end]
        sub_daily_rets = sub_window.pct_change()
        down_day_stock_rets = sub_daily_rets.loc[sub_daily_rets.index.isin(down_day_dates)]
        green_on_down_count = (down_day_stock_rets > 0.0).sum(axis=0)
        down_day_win_rate = (green_on_down_count / total_down_days) * 100.0
    else:
        down_day_win_rate = pd.Series(50.0, index=tickers)

    # Turnover
    turnover_cr = (c_end * vol_latest) / 10000000.0 if not is_us else (c_end * vol_latest) / 1000000.0

    # Assemble DataFrame with clean handling
    df = pd.DataFrame({
        'Ticker': tickers,
        'Close': c_end.values,
        'Return_Window': stock_rets.values,
        'Alpha_Delta': (stock_rets - bench_ret_pct).values,
        'Down_Day_Win_Rate': down_day_win_rate.values,
        'Dist_52W_High': dist_high.values,
        'Volume_Mult': vol_mult.values,
        'Turnover_Cr': turnover_cr.values
    }).dropna(subset=['Close', 'Return_Window', 'Dist_52W_High'])

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
        df['Industry'] = df['Clean_Ticker'].apply(lambda t: intel.get(t, {}).get('group_name', 'General'))
        df['Ind_Grade'] = df['Clean_Ticker'].apply(lambda t: intel.get(t, {}).get('group_grade_badge', '—'))
        df['Sister_Role'] = df['Clean_Ticker'].apply(lambda t: intel.get(t, {}).get('stock_role', '—'))
    except Exception:
        df['Industry'] = 'General'
        df['Ind_Grade'] = '—'
        df['Sister_Role'] = '—'

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

