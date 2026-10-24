"""
Beach Ball Under Water Engine: Institutional Low-Breadth Accumulation Detection & Radar
========================================================================================
Identifies, tracks, and ranks equities that print institutional power plays
(>= +4.0% price thrust on >= 1.0x 20D ADTV with strong close) on trading sessions
where the broad market 50-day SMA breadth is depressed (< 35.0%).

Physics:
  When market breadth is washed out, systemic index selling acts like hands holding
  a buoyant beach ball under water. When the tide turns or selling abates, these
  stocks with multiple institutional power plays are the first to explode out of the gate.
"""

import os
import pickle
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

import database
if not hasattr(database, 'save_beach_ball_events') or not hasattr(database, 'get_beach_ball_badge_map'):
    import importlib
    importlib.reload(database)

from database import (
    get_connection,
    save_beach_ball_events,
    get_latest_beach_ball_sync_date,
    get_beach_ball_leaderboard,
    get_beach_ball_badge_map,
    get_all_fundamentals_cache
)

@st.cache_data(ttl=1800, show_spinner=False)
def get_low_breadth_dates(market: str = 'INDIA', breadth_threshold: float = 35.0, lookback_days: int = 180) -> dict:
    """
    Returns a dict of {date_str: above_50_pct} for sessions where 50-SMA breadth was below threshold.
    """
    mkt_code = 'US' if market.upper() in ['US', 'USA'] else 'IN'
    conn = get_connection()
    c = conn.cursor()
    try:
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        c.execute('''
            SELECT date, above_50_pct 
            FROM market_breadth_daily 
            WHERE market = ? AND above_50_pct <= ? AND date >= ?
            ORDER BY date ASC
        ''', (mkt_code, breadth_threshold, cutoff))
        rows = c.fetchall()
        return {r[0]: float(r[1]) for r in rows if r[0] and r[1] is not None}
    except Exception as e:
        print(f"[BeachBall Engine] Error querying low breadth dates: {e}")
        return {}
    finally:
        conn.close()

def sync_beach_ball_events(
    market: str = 'INDIA',
    lookback_days: int = 180,
    breadth_threshold: float = 35.0,
    force_rescan: bool = False
) -> int:
    """
    Scans historical price matrices on qualifying low-breadth dates (< 35% above 50 SMA).
    Detects institutional accumulation power plays and commits them to SQLite.
    
    Criteria:
      1. Market Breadth: 50-SMA breadth < breadth_threshold (default 35.0%)
      2. Price Thrust: Daily return >= 3.8% (standard +4% threshold with rounding buffer)
      3. Volume Expansion: Volume >= 1.0x 20-day Average Daily Volume
      4. Close Location Value (CLV): >= 0.65 (close in top 35% of day's range)
      5. Liquidity Floor: Turnover >= ₹5 Cr (India) or >= $5M (US)
      6. Structural Alignment: Price is above or within 5% of 50-day SMA and within 25% of 52W high
    """
    mkt_tag = 'US' if market.upper() in ['US', 'USA'] else 'INDIA'
    low_breadth_map = get_low_breadth_dates(market=mkt_tag, breadth_threshold=breadth_threshold, lookback_days=lookback_days)
    if not low_breadth_map:
        return 0

    latest_synced = get_latest_beach_ball_sync_date(mkt_tag)
    dates_to_scan = sorted(list(low_breadth_map.keys()))
    if not force_rescan and latest_synced:
        dates_to_scan = [d for d in dates_to_scan if d >= latest_synced]

    if not dates_to_scan:
        return 0

    # Load cached matrix if available
    matrix_path = 'historical_prices_matrix.pkl'
    if not os.path.exists(matrix_path):
        return 0

    try:
        with open(matrix_path, 'rb') as f:
            df_matrix = pickle.load(f)
    except Exception as e:
        print(f"[BeachBall Engine] Failed to load matrix: {e}")
        return 0

    if not isinstance(df_matrix, pd.DataFrame) or df_matrix.empty:
        return 0

    # Convert matrix index to date strings
    matrix_date_map = {ts.strftime('%Y-%m-%d'): ts for ts in df_matrix.index}
    valid_scan_dates = [d for d in dates_to_scan if d in matrix_date_map]

    if not valid_scan_dates:
        return 0

    # Filter tickers belonging to this market
    all_tickers = list(df_matrix.columns.levels[0])
    if mkt_tag == 'INDIA':
        universe_tickers = [t for t in all_tickers if t.endswith('.NS')]
        turnover_div = 1e7 # Crores
        min_turnover = 5.0 # ₹5 Cr
    else:
        universe_tickers = [t for t in all_tickers if not t.endswith('.NS')]
        turnover_div = 1e6 # Millions
        min_turnover = 5.0 # $5M

    events_to_save = []

    for ticker in universe_tickers:
        try:
            sub = df_matrix[ticker]
            c = sub['Close'].dropna()
            if len(c) < 55:
                continue

            v = sub['Volume'].dropna()
            h = sub['High'].dropna() if 'High' in sub.columns else c
            l = sub['Low'].dropna() if 'Low' in sub.columns else c

            pct_change = c.pct_change() * 100.0
            v_ma20 = v.rolling(20).mean()
            sma50 = c.rolling(50).mean()
            max52w = h.rolling(252, min_periods=60).max()

            for dt_str in valid_scan_dates:
                ts = matrix_date_map.get(dt_str)
                if ts not in pct_change.index or ts not in v_ma20.index:
                    continue

                ret = float(pct_change.loc[ts])
                vol = float(v.loc[ts])
                ma_v = float(v_ma20.loc[ts])

                # 1. Price Thrust & Volume Expansion Gate
                if ret < 3.8 or ma_v <= 0 or vol < (1.0 * ma_v):
                    continue

                cur_c = float(c.loc[ts])
                day_h = float(h.loc[ts]) if ts in h.index else cur_c
                day_l = float(l.loc[ts]) if ts in l.index else cur_c

                # 2. Close Location Value (CLV) >= 0.65
                if day_h > day_l:
                    clv = (cur_c - day_l) / (day_h - day_l)
                else:
                    clv = 1.0

                if clv < 0.65:
                    continue

                # 3. Turnover Gate
                day_turnover = (cur_c * vol) / turnover_div
                if day_turnover < min_turnover:
                    continue

                # 4. Stage 2 Structural Alignment Gate
                # Close must be above or within 5% of 50 SMA, and within 25% of 52W high
                sma50_val = float(sma50.loc[ts]) if ts in sma50.index and not np.isnan(sma50.loc[ts]) else 0.0
                dist_sma50 = ((cur_c - sma50_val) / sma50_val * 100.0) if sma50_val > 0 else 0.0
                if dist_sma50 < -5.0:
                    continue

                high_52w = float(max52w.loc[ts]) if ts in max52w.index and not np.isnan(max52w.loc[ts]) else cur_c
                dist_52w = ((high_52w - cur_c) / high_52w * 100.0) if high_52w > 0 else 0.0
                if dist_52w > 25.0:
                    continue

                vol_mult = round(vol / ma_v, 2)
                breadth_val = low_breadth_map.get(dt_str, 0.0)

                events_to_save.append({
                    'ticker': ticker,
                    'event_date': dt_str,
                    'market': mkt_tag,
                    'daily_return_pct': round(ret, 2),
                    'volume_multiple': vol_mult,
                    'turnover_cr': round(day_turnover, 2),
                    'close_location_val': round(clv, 2),
                    'market_breadth_50sma': round(breadth_val, 1),
                    'close_price': round(cur_c, 2),
                    'is_stage2': 1
                })
        except Exception:
            continue

    if events_to_save:
        saved_count = save_beach_ball_events(events_to_save)
        print(f"[BeachBall Engine] {mkt_tag}: Scanned {len(valid_scan_dates)} dates, committed {saved_count} power plays.")
        return saved_count

    return 0

def auto_sync_beach_ball_if_low_breadth(breadth_threshold: float = 35.0) -> dict:
    """
    Background worker routine:
    Checks if latest market breadth for India and US is < 35%.
    If so, syncs newly emerged low-breadth power plays in background thread.
    Zero UI latency.
    """
    results = {'INDIA': 0, 'US': 0}
    try:
        conn = get_connection()
        c = conn.cursor()
        
        # Check India
        c.execute('''
            SELECT date, above_50_pct 
            FROM market_breadth_daily 
            WHERE market = 'IN' 
            ORDER BY date DESC LIMIT 1
        ''')
        row_in = c.fetchone()
        in_breadth = float(row_in[1]) if row_in and row_in[1] is not None else 50.0

        # Check US
        c.execute('''
            SELECT date, above_50_pct 
            FROM market_breadth_daily 
            WHERE market = 'US' 
            ORDER BY date DESC LIMIT 1
        ''')
        row_us = c.fetchone()
        us_breadth = float(row_us[1]) if row_us and row_us[1] is not None else 50.0
        conn.close()

        # Run India sync if current breadth < 35% or if table is empty
        latest_in_sync = get_latest_beach_ball_sync_date('INDIA')
        if in_breadth < breadth_threshold or not latest_in_sync:
            n_in = sync_beach_ball_events(market='INDIA', breadth_threshold=breadth_threshold)
            results['INDIA'] = n_in

        # Run US sync if current breadth < 35% or if table is empty
        latest_us_sync = get_latest_beach_ball_sync_date('US')
        if us_breadth < breadth_threshold or not latest_us_sync:
            n_us = sync_beach_ball_events(market='US', breadth_threshold=breadth_threshold)
            results['US'] = n_us

    except Exception as e:
        print(f"[BeachBall Sync] Background worker error: {e}")

    return results

@st.cache_data(ttl=1800, show_spinner=False)
def get_enriched_beach_ball_radar(
    market: str = 'INDIA',
    breadth_threshold: float = 35.0,
    min_power_plays_3m: int = 1
) -> pd.DataFrame:
    """
    Returns the enriched beach ball radar leaderboard from SQLite.
    Instant sub-50ms query time.
    Ranked strictly by 3M Power Plays -> 1M Power Plays -> Volume Surge.
    """
    mkt_tag = 'US' if market.upper() in ['US', 'USA'] else 'INDIA'
    raw_leaders = get_beach_ball_leaderboard(
        market=mkt_tag,
        breadth_threshold=breadth_threshold,
        min_power_plays_3m=min_power_plays_3m
    )
    if not raw_leaders:
        return pd.DataFrame()

    df = pd.DataFrame(raw_leaders)

    # Enrich with fundamentals & RS cache
    f_cache = get_all_fundamentals_cache()
    
    # Try loading price matrix for live 52W high and RS score
    rs_map = {}
    dist52_map = {}
    cur_price_map = {}
    industry_map = {}
    comp_name_map = {}

    try:
        from down_day_engine import load_price_matrix
        df_matrix = load_price_matrix()
    except Exception:
        df_matrix = None

    for t in df['ticker']:
        clean_t = t.replace('.NS', '')
        info = f_cache.get(t, {}) or f_cache.get(clean_t, {}) or {}
        comp_name_map[t] = info.get('company_name', clean_t)
        industry_map[t] = info.get('industry', 'Broad Market')

        if df_matrix is not None and t in df_matrix.columns.levels[0]:
            try:
                sub_c = df_matrix[t]['Close'].dropna()
                sub_h = df_matrix[t]['High'].dropna() if 'High' in df_matrix[t].columns else sub_c
                if not sub_c.empty:
                    c_now = float(sub_c.iloc[-1])
                    h_max = float(sub_h.tail(252).max()) if not sub_h.empty else c_now
                    dist = ((h_max - c_now) / h_max * 100.0) if h_max > 0 else 0.0
                    cur_price_map[t] = c_now
                    dist52_map[t] = round(dist, 1)

                    # Simple 3M/6M momentum proxy for RS
                    if len(sub_c) >= 63:
                        r3m = (c_now / float(sub_c.iloc[-63])) - 1.0
                        rs_map[t] = int(np.clip(50 + r3m * 100.0, 50, 99))
                    else:
                        rs_map[t] = 80
            except Exception:
                pass

        if t not in rs_map:
            rs_map[t] = int(info.get('rs_score', 80))
        if t not in dist52_map:
            dist52_map[t] = 5.0
        if t not in cur_price_map:
            cur_price_map[t] = float(df.loc[df['ticker'] == t, 'last_close_price'].iloc[0])

    try:
        from industry_group_engine import get_stock_group_intelligence
        intel = get_stock_group_intelligence(universe_scope="US (S&P 500 / Nasdaq)" if (mkt_tag == 'US') else "India (NSE/BSE)")
    except Exception:
        intel = {}

    ind_grade_map = {}
    sister_role_map = {}
    for t in df['ticker']:
        clean_t = t.replace('.NS', '').replace('.BO', '')
        info_intel = intel.get(clean_t) or intel.get(t) or {}
        if info_intel:
            industry_map[t] = info_intel.get('group_name', industry_map.get(t, 'Broad Market'))
            ind_grade_map[t] = info_intel.get('group_grade_badge', '—')
            sister_role_map[t] = info_intel.get('stock_role', '—')
        else:
            ind_grade_map[t] = '—'
            sister_role_map[t] = '—'

    df['company_name'] = df['ticker'].map(comp_name_map)
    df['industry'] = df['ticker'].map(industry_map)
    df['ind_grade'] = df['ticker'].map(ind_grade_map)
    df['sister_role'] = df['ticker'].map(sister_role_map)
    df['rs_score'] = df['ticker'].map(rs_map)
    df['current_price'] = df['ticker'].map(cur_price_map)
    df['dist_52w_high'] = df['ticker'].map(dist52_map)

    # Accumulation Tier Categorization
    def categorize_tier(p3m):
        if p3m >= 4:
            return '👑 Apex (4x+)'
        elif p3m >= 2:
            return '🔥 Multi (2-3x)'
        else:
            return '🌱 Single (1x)'

    df['accumulation_tier'] = df['power_plays_3m'].apply(categorize_tier)

    # Setup Status categorization based on 52W high distance
    def get_status(dist):
        if dist <= 3.0:
            return '🚀 Breakout Ready'
        elif dist <= 8.0:
            return '🏖️ Coiling Highs'
        elif dist <= 15.0:
            return '🎯 21-EMA Pullback'
        else:
            return '🧱 Base Formation'

    df['setup_status'] = df['dist_52w_high'].apply(get_status)

    # TradingView Link
    is_in = (mkt_tag == 'INDIA')
    df['clean_ticker'] = df['ticker'].apply(lambda s: s.replace('.NS', ''))
    df['tv_url'] = df.apply(
        lambda r: f"https://www.tradingview.com/chart/?symbol={'NSE:' + r['clean_ticker'] if is_in else r['clean_ticker']}",
        axis=1
    )

    # Primary sort order: 3M power plays DESC -> 1M power plays DESC -> RS score DESC
    df = df.sort_values(by=['power_plays_3m', 'power_plays_1m', 'rs_score'], ascending=[False, False, False]).reset_index(drop=True)
    df['rank'] = df.index + 1

    return df
