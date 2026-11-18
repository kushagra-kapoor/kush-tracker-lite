from __future__ import annotations

import os
import pandas as pd
import numpy as np
from datetime import datetime
from database import get_connection, safe_execute, read_sql_df, _fetch_all_dicts

# 250 trading days in a year
YEAR_TRADING_DAYS = 250
# If the stock is within 2% of its 52w high/low, we count it as a "New High/Low"
PROXIMITY_THRESHOLD = 0.02

def init_breadth_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_breadth_daily (
            date TEXT,
            market TEXT DEFAULT 'IN',
            above_20_pct REAL,
            above_50_pct REAL,
            above_200_pct REAL,
            net_new_highs INTEGER,
            new_highs_count INTEGER,
            new_lows_count INTEGER,
            advances INTEGER,
            declines INTEGER,
            ad_line_cumulative INTEGER,
            double_52w_count INTEGER,
            quadruple_52w_count INTEGER,
            total_stocks INTEGER,
            surge_extreme_5d INTEGER,
            panic_extreme_5d INTEGER,
            up_volume REAL DEFAULT 0,
            down_volume REAL DEFAULT 0,
            up_down_volume_ratio REAL DEFAULT 1.0,
            cumulative_volume_ad_line REAL DEFAULT 0,
            ad_grade_a_count INTEGER DEFAULT 0,
            ad_grade_b_count INTEGER DEFAULT 0,
            ad_grade_c_count INTEGER DEFAULT 0,
            ad_grade_d_count INTEGER DEFAULT 0,
            ad_grade_e_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, market)
        )
    ''')
    
    # Safe migration for existing tables
    new_cols = [
        ("quadruple_52w_count", "INTEGER DEFAULT 0"),
        ("up_volume", "REAL DEFAULT 0"),
        ("down_volume", "REAL DEFAULT 0"),
        ("up_down_volume_ratio", "REAL DEFAULT 1.0"),
        ("cumulative_volume_ad_line", "REAL DEFAULT 0"),
        ("ad_grade_a_count", "INTEGER DEFAULT 0"),
        ("ad_grade_b_count", "INTEGER DEFAULT 0"),
        ("ad_grade_c_count", "INTEGER DEFAULT 0"),
        ("ad_grade_d_count", "INTEGER DEFAULT 0"),
        ("ad_grade_e_count", "INTEGER DEFAULT 0"),
        ("advance_ratio", "REAL DEFAULT 0.5"),
        ("zbt_10_ema", "REAL DEFAULT 0.5"),
        ("zbt_signal", "INTEGER DEFAULT 0"),
        ("zbt_active", "INTEGER DEFAULT 0"),
        ("is_9_to_1_up_day", "INTEGER DEFAULT 0"),
        ("double_9_to_1_active", "INTEGER DEFAULT 0")
    ]
    for col_name, col_def in new_cols:
        try:
            cursor.execute(f"ALTER TABLE market_breadth_daily ADD COLUMN {col_name} {col_def}")
        except Exception:
            pass # Column already exists
            
    conn.commit()
    conn.close()

def compute_zweig_thrust_and_volume_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Dr. Martin Zweig Breadth Thrust (ZBT) and 9-to-1 Up-Volume Days.
    
    Mathematical Specifications (Dr. Martin Zweig, Winning on Wall Street):
      - Advance Ratio = advances / (advances + declines)
      - ZBT_10 = 10-day EMA of Advance Ratio
      - ZBT Signal: Surges from < 0.40 to >= 0.615 within 10 consecutive trading days
      - ZBT Active: Remains active for 30 trading days post-signal
      - 9-to-1 Up-Volume Day: Up Volume >= 90% of (Up Volume + Down Volume) or Up/Down Ratio >= 9.0
      - Double 9:1 Blast-Off: >= 2 9-to-1 days within rolling 65 trading days (3 months)
    """
    if df is None or df.empty:
        return df
        
    df = df.copy()
    
    # Advances / Declines Ratio & 10 EMA
    if 'advances' in df.columns and 'declines' in df.columns:
        tot_issues = (df['advances'].fillna(0) + df['declines'].fillna(0)).replace(0, 1)
        adv_ratio = (df['advances'].fillna(0) / tot_issues).clip(0.0, 1.0)
        df['advance_ratio'] = adv_ratio
        zbt_ema = adv_ratio.ewm(span=10, adjust=False).mean()
        df['zbt_10_ema'] = zbt_ema
        
        # ZBT Signal detection
        zbt_sig = np.zeros(len(df), dtype=int)
        zbt_act = np.zeros(len(df), dtype=int)
        zbt_vals = zbt_ema.values
        
        for i in range(len(df)):
            if zbt_vals[i] >= 0.615:
                start_w = max(0, i - 10)
                window = zbt_vals[start_w:i]
                if len(window) > 0 and np.min(window) < 0.40:
                    zbt_sig[i] = 1
                    
        # Active counter (30 sessions)
        act_counter = 0
        for i in range(len(df)):
            if zbt_sig[i] == 1:
                act_counter = 30
            elif act_counter > 0:
                act_counter -= 1
            zbt_act[i] = 1 if act_counter > 0 else 0
            
        df['zbt_signal'] = zbt_sig
        df['zbt_active'] = zbt_act

    # Up / Down Volume 9:1 Metrics
    if 'up_volume' in df.columns and 'down_volume' in df.columns:
        tot_vol = df['up_volume'].fillna(0) + df['down_volume'].fillna(0)
        up_pct = np.where(tot_vol > 0, df['up_volume'].fillna(0) / tot_vol, 0.5)
        df['up_vol_pct'] = up_pct
        
        ud_ratio = df['up_down_volume_ratio'].fillna(1.0) if 'up_down_volume_ratio' in df.columns else (df['up_volume'] / df['down_volume'].replace(0, 1))
        is_9to1 = ((up_pct >= 0.90) | (ud_ratio >= 9.0)).astype(int)
        df['is_9_to_1_up_day'] = is_9to1
        
        rolling_9to1 = pd.Series(is_9to1, index=df.index).rolling(window=65, min_periods=1).sum()
        df['count_9_to_1_65d'] = rolling_9to1.astype(int)
        df['double_9_to_1_active'] = (rolling_9to1 >= 2).astype(int)
        
    return df

def get_zweig_tape_summary(market='IN', lookback=120) -> dict:
    """
    Returns an institutional summary of the Zweig Breadth Thrust & 9:1 Volume Model.
    """
    df = fetch_historical_breadth(market=market, days=lookback)
    if df.empty:
        return {
            'zbt_10_ema': 0.50,
            'zbt_signal': False,
            'zbt_active': False,
            'days_since_zbt': None,
            'is_9_to_1_today': False,
            'count_9_to_1_65d': 0,
            'double_9_to_1_active': False,
            'recent_9_to_1_days': []
        }
        
    latest = df.iloc[-1]
    
    # Days since last ZBT signal
    zbt_indices = df.index[df.get('zbt_signal', 0) == 1].tolist()
    days_since_zbt = None
    if zbt_indices:
        last_zbt_date = zbt_indices[-1]
        days_since_zbt = (len(df) - 1) - df.index.get_loc(last_zbt_date)
        
    # Recent 9-to-1 days in last 65 sessions
    tail_65 = df.tail(65)
    recent_9to1 = []
    if 'is_9_to_1_up_day' in tail_65.columns:
        for dt, row in tail_65[tail_65['is_9_to_1_up_day'] == 1].iterrows():
            idx_loc = df.index.get_loc(dt)
            days_ago = (len(df) - 1) - idx_loc
            recent_9to1.append({
                'date': str(dt)[:10],
                'days_ago': int(days_ago),
                'up_vol_pct': round(float(row.get('up_vol_pct', 0.9)) * 100, 1),
                'ud_ratio': round(float(row.get('up_down_volume_ratio', 9.0)), 2),
                'advances': int(row.get('advances', 0)),
                'declines': int(row.get('declines', 0)),
                'up_volume': float(row.get('up_volume', 0)),
                'down_volume': float(row.get('down_volume', 0))
            })
            
    return {
        'zbt_10_ema': round(float(latest.get('zbt_10_ema', 0.50)), 3),
        'zbt_signal': bool(latest.get('zbt_signal', 0) == 1),
        'zbt_active': bool(latest.get('zbt_active', 0) == 1),
        'days_since_zbt': days_since_zbt,
        'is_9_to_1_today': bool(latest.get('is_9_to_1_up_day', 0) == 1),
        'count_9_to_1_65d': int(latest.get('count_9_to_1_65d', len(recent_9to1))),
        'double_9_to_1_active': bool(latest.get('double_9_to_1_active', 0) == 1),
        'recent_9_to_1_days': recent_9to1
    }

def fetch_historical_breadth(market='IN', days=60):
    init_breadth_table()
    df = read_sql_df("SELECT * FROM market_breadth_daily WHERE market=? ORDER BY date DESC LIMIT ?", params=(market, days))
    if not df.empty:
        df = df.sort_values('date').set_index('date')
        df = compute_zweig_thrust_and_volume_signals(df)
    return df

def fetch_full_breadth_history(market='IN', days=1500):
    init_breadth_table()
    if days:
        query = """
            SELECT * FROM (
                SELECT * FROM market_breadth_daily 
                WHERE market=? 
                ORDER BY date DESC 
                LIMIT ?
            ) ORDER BY date ASC
        """
        df = read_sql_df(query, params=(market, days))
    else:
        query = "SELECT * FROM market_breadth_daily WHERE market=? ORDER BY date ASC"
        df = read_sql_df(query, params=(market,))
    if not df.empty:
        # Exclude zero-trade phantom holiday rows if any
        if 'advances' in df.columns and 'declines' in df.columns:
            df = df[~((df['advances'] == 0) & (df['declines'] == 0) & (df['date'] > '2022-09-01'))]
        df = df.set_index('date')
        # Guarantee unbroken mathematical continuity of cumulative A/D lines
        if 'advances' in df.columns and 'declines' in df.columns:
            df['ad_line_cumulative'] = (df['advances'].fillna(0) - df['declines'].fillna(0)).cumsum()
        if 'up_volume' in df.columns and 'down_volume' in df.columns:
            df['cumulative_volume_ad_line'] = (df['up_volume'].fillna(0) - df['down_volume'].fillna(0)).cumsum()
        df = compute_zweig_thrust_and_volume_signals(df)
    return df

def sync_breadth_history(market='IN', lookback_days=1000):
    init_breadth_table()
    
    matrix_path = 'breadth_matrix.pkl'
    valid_tickers = []
    sp500_set = set()
    
    if market == 'IN':
        try:
            from market_data import fetch_nifty_total_market_tickers
            valid_tickers = fetch_nifty_total_market_tickers(show_progress=False)
        except Exception as e:
            print(f"[Breadth Engine] Failed to fetch tickers: {e}")
            
        if not valid_tickers:
            try:
                from universe_security_master import load_universe_tickers
                valid_tickers = load_universe_tickers("IN")
            except Exception:
                if os.path.exists('tickers.txt'):
                    with open('tickers.txt', 'r') as f:
                        valid_tickers = [line.strip().upper() for line in f if line.strip()]
                        valid_tickers = [t if t.endswith(('.NS', '.BO')) else f"{t}.NS" for t in valid_tickers]
    else:
        # US Market Breadth follows the exact India Nifty-750 symmetry:
        # Restrict breadth aggregates cleanly to institutional S&P 500 core (~503)
        # while retaining Nasdaq 100 in breadth_matrix.pkl for the NDX 100 slice toggle.
        try:
            from us_universe_manager import get_nasdaq100_constituents, get_sp500_constituents
            sp500_list = get_sp500_constituents()
            ndx100_list = get_nasdaq100_constituents()
            sp500_set = set(sp500_list)
            valid_tickers = sorted(set(sp500_list).union(set(ndx100_list)))
        except Exception as e:
            print(f"[Breadth Engine] Failed to fetch US S&P 500 constituents: {e}")

        if not valid_tickers and os.path.exists('tickers_us.txt'):
            with open('tickers_us.txt', 'r') as f:
                valid_tickers = [line.strip().upper() for line in f if line.strip()]

    if not valid_tickers:
        return

    # Trigger isolated 4-year download cache just for the breadth engine
    try:
        from price_history_manager import fetch_incremental_history
        print(f"[Breadth Engine] Syncing isolated 4-year Breadth Matrix for {len(valid_tickers)} stocks...")
        df = fetch_incremental_history(valid_tickers, cache_file=matrix_path, period="4y")
    except Exception as e:
        print(f"[Breadth Engine] Failed to sync 4-year matrix: {e}")
        return

    if df.empty:
        return

    try:
        closes = df.xs('Close', level=1, axis=1).dropna(how='all')
        highs = df.xs('High', level=1, axis=1).dropna(how='all')
        lows = df.xs('Low', level=1, axis=1).dropna(how='all')
        volumes = df.xs('Volume', level=1, axis=1).dropna(how='all') if 'Volume' in df.columns.levels[1] else pd.DataFrame()
    except Exception:
        return

    # Strictly restrict US aggregate breadth calculation to S&P 500 core (exclude NDX-only stocks from denominator)
    if market == 'US' and sp500_set:
        sp_cols = [c for c in closes.columns if str(c).upper() in sp500_set]
        if len(sp_cols) >= 400:
            closes = closes[sp_cols].dropna(how='all')
            highs = highs[[c for c in sp_cols if c in highs.columns]].reindex(closes.index)
            lows = lows[[c for c in sp_cols if c in lows.columns]].reindex(closes.index)
            if not volumes.empty:
                volumes = volumes[[c for c in sp_cols if c in volumes.columns]].reindex(closes.index)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT date, up_volume, ad_grade_a_count, total_stocks 
        FROM market_breadth_daily 
        WHERE market = ? 
        ORDER BY date DESC 
        LIMIT 1
    """, (market,))
    last_date_row = cursor.fetchone()
    last_date_str = last_date_row[0] if last_date_row and last_date_row[0] else None
    has_up_vol = (
        last_date_row is not None 
        and last_date_row[1] is not None and last_date_row[1] > 0
        and len(last_date_row) > 2 and last_date_row[2] is not None and last_date_row[2] > 0
    )
    last_total_stocks = int(last_date_row[3] or 0) if (last_date_row and len(last_date_row) > 3 and last_date_row[3] is not None) else 0
    
    dates_in_matrix = closes.index.strftime('%Y-%m-%d').tolist()
    
    # Migrate legacy >600-stock US breadth smoothly to clean S&P 500 (~503) continuous baseline
    if market == 'US' and sp500_set and last_total_stocks > 600:
        print(f"[Breadth Engine] Migrating US market_breadth_daily from legacy {last_total_stocks}-stock denominator to pure S&P 500 baseline...")
        cursor.execute("DELETE FROM market_breadth_daily WHERE market = 'US'")
        conn.commit()
        missing_dates = dates_in_matrix[-lookback_days:]
    elif not last_date_str:
        missing_dates = dates_in_matrix[-lookback_days:]
    elif not has_up_vol:
        # Re-sync recent window to backfill volume and A-E metrics
        missing_dates = dates_in_matrix[-min(len(dates_in_matrix), 180):]
    else:
        missing_dates = [d for d in dates_in_matrix if d > last_date_str]

    if not missing_dates:
        conn.close()
        return

    # Fetch cumulative values immediately prior to the first missing date to avoid resets/jumps
    first_missing_date = missing_dates[0]
    cursor.execute('''
        SELECT ad_line_cumulative, cumulative_volume_ad_line 
        FROM market_breadth_daily 
        WHERE market=? AND date < ? 
        ORDER BY date DESC LIMIT 1
    ''', (market, first_missing_date))
    prev_ad_row = cursor.fetchone()
    current_ad_cumulative = prev_ad_row[0] if prev_ad_row and prev_ad_row[0] is not None else 0
    current_vol_ad_cumulative = prev_ad_row[1] if prev_ad_row and len(prev_ad_row) > 1 and prev_ad_row[1] is not None else 0.0

    sma_20 = closes.rolling(window=20, min_periods=15).mean()
    sma_50 = closes.rolling(window=50, min_periods=40).mean()
    sma_200 = closes.rolling(window=200, min_periods=180).mean()
    daily_returns = closes / closes.shift(1) - 1
    
    # Pradeep Bonde 5-Day Extreme Signals
    ret_5d = closes / closes.shift(5) - 1
    
    year_highs = highs.rolling(window=YEAR_TRADING_DAYS, min_periods=200).max()
    year_lows = lows.rolling(window=YEAR_TRADING_DAYS, min_periods=200).min()

    # Precalculate vectorized daily institutional flow for A-E Grade distribution
    if not volumes.empty:
        vol_sma50 = volumes.rolling(50, min_periods=20).mean()
        vol_ratio_matrix = (volumes / vol_sma50).clip(0.2, 3.5)
        hl_range = (highs - lows).replace(0, np.nan)
        clv_matrix = (((closes - lows) - (highs - closes)) / hl_range).fillna(0).clip(-1, 1)
        daily_flow_matrix = (0.6 * np.sign(daily_returns).fillna(0) + 0.4 * clv_matrix) * vol_ratio_matrix
        weights_65 = np.linspace(1.0, 2.0, 65)
        weights_norm_65 = weights_65 / weights_65.sum()
    else:
        daily_flow_matrix = pd.DataFrame()

    records_to_insert = []
    
    for date_str in missing_dates:
        try:
            idx_pos = dates_in_matrix.index(date_str)
            dt_index = closes.index[idx_pos]
            
            c = closes.loc[dt_index]
            s20 = sma_20.loc[dt_index]
            s50 = sma_50.loc[dt_index]
            s200 = sma_200.loc[dt_index]
            yh = year_highs.loc[dt_index]
            yl = year_lows.loc[dt_index]
            r5 = ret_5d.loc[dt_index]
            d_ret = daily_returns.loc[dt_index]
            
            valid_20 = c.notna() & s20.notna()
            total_20 = valid_20.sum()
            above_20 = (c[valid_20] > s20[valid_20]).sum()
            above_20_pct = (above_20 / total_20 * 100) if total_20 > 0 else 0

            valid_50 = c.notna() & s50.notna()
            total_50 = valid_50.sum()
            above_50 = (c[valid_50] > s50[valid_50]).sum()
            above_50_pct = (above_50 / total_50 * 100) if total_50 > 0 else 0
            
            valid_200 = c.notna() & s200.notna()
            total_200 = valid_200.sum()
            above_200 = (c[valid_200] > s200[valid_200]).sum()
            above_200_pct = (above_200 / total_200 * 100) if total_200 > 0 else 0
            
            valid_hl = c.notna() & yh.notna() & yl.notna()
            new_h = (c[valid_hl] >= (yh[valid_hl] * (1 - PROXIMITY_THRESHOLD))).sum()
            new_l = (c[valid_hl] <= (yl[valid_hl] * (1 + PROXIMITY_THRESHOLD))).sum()
            net_nh = int(new_h - new_l)
            
            double_52w = (c[valid_hl] >= (yl[valid_hl] * 2)).sum()
            quad_52w = (c[valid_hl] >= (yl[valid_hl] * 4)).sum()
            
            valid_ret = d_ret.notna()
            advances = int((d_ret[valid_ret] > 0).sum())
            declines = int((d_ret[valid_ret] < 0).sum())
            current_ad_cumulative += int(advances - declines)
            
            valid_r5 = r5.notna()
            s_thresh = 0.12 if market == 'IN' else 0.20
            p_thresh = -0.12 if market == 'IN' else -0.20
            surge_count = (r5[valid_r5] >= s_thresh).sum()
            panic_count = (r5[valid_r5] <= p_thresh).sum()

            # Up / Down Volume Breadth
            up_vol = 0.0
            down_vol = 0.0
            ud_ratio = 1.0
            if not volumes.empty and dt_index in volumes.index:
                v = volumes.loc[dt_index]
                up_vol = float(v[valid_ret & (d_ret > 0)].sum())
                down_vol = float(v[valid_ret & (d_ret < 0)].sum())
                ud_ratio = float(up_vol / down_vol) if down_vol > 0 else 1.0
                current_vol_ad_cumulative += float(up_vol - down_vol)

            # Universe A/B/C/D/E Distribution Counts
            a_cnt, b_cnt, c_cnt, d_cnt, e_cnt = 0, 0, 0, 0, 0
            if not daily_flow_matrix.empty and idx_pos >= 64:
                sub_flow = daily_flow_matrix.iloc[idx_pos-64:idx_pos+1]
                w_scores = np.dot(weights_norm_65, sub_flow.fillna(0).values) * 20.0
                a_cnt = int((w_scores >= 2.0).sum())
                b_cnt = int(((w_scores >= 0.6) & (w_scores < 2.0)).sum())
                c_cnt = int(((w_scores >= -0.6) & (w_scores < 0.6)).sum())
                d_cnt = int(((w_scores >= -2.0) & (w_scores < -0.6)).sum())
                e_cnt = int((w_scores < -2.0).sum())
            
            records_to_insert.append((
                date_str, market, float(above_20_pct), float(above_50_pct), float(above_200_pct), 
                net_nh, int(new_h), int(new_l), int(advances), int(declines), int(current_ad_cumulative), int(double_52w), int(quad_52w),
                int(total_50), int(surge_count), int(panic_count),
                float(up_vol), float(down_vol), round(float(ud_ratio), 2), float(current_vol_ad_cumulative),
                int(a_cnt), int(b_cnt), int(c_cnt), int(d_cnt), int(e_cnt)
            ))
        except Exception as e:
            print(f"[Breadth Engine] Error calculating for {date_str}: {e}")

    if records_to_insert:
        cursor.executemany('''
            INSERT OR REPLACE INTO market_breadth_daily 
            (date, market, above_20_pct, above_50_pct, above_200_pct, net_new_highs, new_highs_count, new_lows_count, advances, declines, ad_line_cumulative, double_52w_count, quadruple_52w_count, total_stocks, surge_extreme_5d, panic_extreme_5d, up_volume, down_volume, up_down_volume_ratio, cumulative_volume_ad_line, ad_grade_a_count, ad_grade_b_count, ad_grade_c_count, ad_grade_d_count, ad_grade_e_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', records_to_insert)
        conn.commit()

        # Guarantee continuous mathematical cumulative sums across whole table
        try:
            cursor.execute('''
                WITH cum AS (
                    SELECT date, market, 
                           SUM(advances - declines) OVER (PARTITION BY market ORDER BY date ASC) as true_ad,
                           SUM(up_volume - down_volume) OVER (PARTITION BY market ORDER BY date ASC) as true_vol
                    FROM market_breadth_daily
                    WHERE market = ?
                )
                UPDATE market_breadth_daily
                SET ad_line_cumulative = (SELECT true_ad FROM cum WHERE cum.date = market_breadth_daily.date AND cum.market = market_breadth_daily.market),
                    cumulative_volume_ad_line = (SELECT true_vol FROM cum WHERE cum.date = market_breadth_daily.date AND cum.market = market_breadth_daily.market)
                WHERE market = ?
            ''', (market, market))
            conn.commit()
        except Exception as e_rec:
            print(f"[Breadth Engine] Cumulative reconciliation warning: {e_rec}")

    conn.close()

def get_latest_volume_and_ad_distribution(market='IN') -> dict:
    """
    Retrieves latest Exchange Volume Breadth & Universe A-E Distribution with daily delta.
    Defensively auto-heals and bypasses hollow rows so UI never shows 0-volume or broken deltas.
    """
    init_breadth_table()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM market_breadth_daily 
        WHERE market = ? 
        ORDER BY date DESC 
        LIMIT 5
    """, (market,))
    rows = _fetch_all_dicts(cursor)
    conn.close()

    if not rows:
        return {}

    curr = dict(rows[0])

    # Defensive auto-heal: if latest row is hollow (missing volume and A-E distribution)
    if curr.get('up_volume', 0.0) == 0.0 and curr.get('ad_grade_a_count', 0) == 0:
        try:
            print("[Breadth Engine] Hollow breadth row detected on latest date. Auto-healing from price matrix...")
            sync_breadth_history(market=market, lookback_days=10)
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM market_breadth_daily 
                WHERE market = ? 
                ORDER BY date DESC 
                LIMIT 5
            """, (market,))
            rows = _fetch_all_dicts(cursor)
            conn.close()
            if rows:
                curr = dict(rows[0])
        except Exception as e:
            print(f"[Breadth Engine] Auto-heal warning: {e}")

    # Fallback: if curr is still hollow, pick the most recent row that has valid volume / grades
    if curr.get('up_volume', 0.0) == 0.0 and curr.get('ad_grade_a_count', 0) == 0:
        for r in rows:
            rd = dict(r)
            if rd.get('up_volume', 0.0) > 0 or rd.get('ad_grade_a_count', 0) > 0:
                curr = rd
                break

    # Locate prev row immediately before curr
    prev = curr
    for r in rows:
        rd = dict(r)
        if rd.get('date') < curr.get('date'):
            prev = rd
            break

    # Deltas
    delta_a = curr.get('ad_grade_a_count', 0) - prev.get('ad_grade_a_count', 0)
    delta_b = curr.get('ad_grade_b_count', 0) - prev.get('ad_grade_b_count', 0)
    delta_c = curr.get('ad_grade_c_count', 0) - prev.get('ad_grade_c_count', 0)
    delta_d = curr.get('ad_grade_d_count', 0) - prev.get('ad_grade_d_count', 0)
    delta_e = curr.get('ad_grade_e_count', 0) - prev.get('ad_grade_e_count', 0)

    a_plus_b = curr.get('ad_grade_a_count', 0) + curr.get('ad_grade_b_count', 0)
    d_plus_e = curr.get('ad_grade_d_count', 0) + curr.get('ad_grade_e_count', 0)
    smart_money_ratio = round(a_plus_b / d_plus_e, 2) if d_plus_e > 0 else 1.0

    return {
        'date': curr['date'],
        'market': curr['market'],
        'advances': curr.get('advances', 0),
        'declines': curr.get('declines', 0),
        'up_volume': curr.get('up_volume', 0.0),
        'down_volume': curr.get('down_volume', 0.0),
        'up_down_volume_ratio': curr.get('up_down_volume_ratio', 1.0),
        'cumulative_volume_ad_line': curr.get('cumulative_volume_ad_line', 0.0),
        'ad_grade_a': curr.get('ad_grade_a_count', 0),
        'ad_grade_b': curr.get('ad_grade_b_count', 0),
        'ad_grade_c': curr.get('ad_grade_c_count', 0),
        'ad_grade_d': curr.get('ad_grade_d_count', 0),
        'ad_grade_e': curr.get('ad_grade_e_count', 0),
        'delta_a': delta_a,
        'delta_b': delta_b,
        'delta_c': delta_c,
        'delta_d': delta_d,
        'delta_e': delta_e,
        'smart_money_ratio': smart_money_ratio
    }


def compute_daily_breadth(force=False, market='IN'):
    init_breadth_table()
    sync_breadth_history(market=market, lookback_days=1000)
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM market_breadth_daily WHERE market = ? ORDER BY date DESC LIMIT 1", (market,))
    rows = _fetch_all_dicts(cursor)
    conn.close()
    row = rows[0] if rows else None
    
    if row:
        return {
            'date': row['date'],
            'market': row['market'],
            'above_20_pct': float(row['above_20_pct']) if row['above_20_pct'] is not None else 0.0,
            'above_50_pct': float(row['above_50_pct']) if row['above_50_pct'] is not None else 0.0,
            'above_200_pct': float(row['above_200_pct']) if row['above_200_pct'] is not None else 0.0,
            'net_new_highs': int(row['net_new_highs']) if row['net_new_highs'] is not None else 0,
            'new_highs_count': int(row['new_highs_count']) if row['new_highs_count'] is not None else 0,
            'new_lows_count': int(row['new_lows_count']) if row['new_lows_count'] is not None else 0,
            'total_stocks': int(row['total_stocks']) if row['total_stocks'] is not None else 0,
            'surge_extreme_5d': int(row['surge_extreme_5d']) if row['surge_extreme_5d'] is not None else 0,
            'panic_extreme_5d': int(row['panic_extreme_5d']) if row['panic_extreme_5d'] is not None else 0
        }
    return None

# removed the redundant .iloc[-1] logic here as it is moved into sync_breadth_history

def compute_top_rs_breadth_history(market='IN', lookback_days=300):
    """
    Computes Top 100 RS Stock Breadth (% above 21 EMA and 50 SMA).
    Institutional leading stock breadth indicator.
    """
    matrix_path = 'breadth_matrix.pkl'
    if not os.path.exists(matrix_path):
        matrix_path = 'historical_prices_matrix.pkl'
    if not os.path.exists(matrix_path):
        try:
            from database import load_cloud_json_cache_cached
            is_us = str(market or "IN").upper().strip().startswith("US")
            bundle, _ = load_cloud_json_cache_cached("breadth_aux_bundle_us" if is_us else "breadth_aux_bundle_in")
            if isinstance(bundle, dict) and isinstance(bundle.get("top_rs_breadth"), pd.DataFrame):
                df_c = bundle["top_rs_breadth"].copy()
                return df_c.tail(int(lookback_days)) if lookback_days else df_c
        except Exception:
            pass
        return pd.DataFrame()
        
    try:
        df = pd.read_pickle(matrix_path)
        if isinstance(df.columns, pd.MultiIndex):
            closes = df.xs('Close', level=1, axis=1).dropna(how='all')
        else:
            closes = df
    except Exception as e:
        print(f"[Breadth Engine] Error reading matrix for Top RS breadth: {e}")
        return pd.DataFrame()

    if closes.empty or len(closes) < 70:
        return pd.DataFrame()

    # Filter strictly by market
    from price_history_manager import filter_panel_active_market_dates, is_excluded_etf_or_index
    if market == 'IN':
        in_cols = [c for c in closes.columns if (str(c).endswith('.NS') or str(c).endswith('.BO')) and not is_excluded_etf_or_index(c)]
        try:
            from market_data import fetch_nifty_total_market_tickers
            n750 = set(fetch_nifty_total_market_tickers(show_progress=False))
            clean_750 = {t.replace('.NS', '').replace('.BO', '') for t in n750}
            in_cols = [c for c in in_cols if str(c) in n750 or str(c).replace('.NS', '').replace('.BO', '') in clean_750]
        except Exception:
            pass
        if in_cols:
            closes = filter_panel_active_market_dates(closes[in_cols], min_coverage_ratio=0.15, min_abs=10)
    else:
        us_cols = [c for c in closes.columns if not str(c).upper().endswith(('.NS', '.BO')) and not is_excluded_etf_or_index(c)]
        try:
            from us_universe_manager import get_sp500_constituents
            sp_set = set(get_sp500_constituents())
            sp_cols = [c for c in us_cols if str(c).upper() in sp_set]
            if len(sp_cols) >= 400:
                us_cols = sp_cols
        except Exception:
            pass
        if us_cols:
            closes = filter_panel_active_market_dates(closes[us_cols], min_coverage_ratio=0.15, min_abs=10)

    ema_20 = closes.ewm(span=20, adjust=False).mean()
    ema_21 = closes.ewm(span=21, adjust=False).mean()
    sma_50 = closes.rolling(window=50, min_periods=40).mean()
    sma_200 = closes.rolling(window=200, min_periods=120).mean()

    ret_1w = closes.pct_change(5)
    ret_1m = closes.pct_change(21)
    ret_3m = closes.pct_change(63)
    rs_raw = (ret_1w * 0.20) + (ret_1m * 0.40) + (ret_3m * 0.40)

    dates = closes.index[-lookback_days:]
    records = []

    for dt in dates:
        rs_series = rs_raw.loc[dt].dropna()
        if len(rs_series) < 50:
            continue
            
        top_count = min(100, len(rs_series))
        top_tickers = rs_series.nlargest(top_count).index
        
        c_vals = closes.loc[dt, top_tickers]
        e21_vals = ema_21.loc[dt, top_tickers]
        s50_vals = sma_50.loc[dt, top_tickers]
        
        above_21 = float((c_vals > e21_vals).sum() / top_count * 100.0)
        above_50 = float((c_vals > s50_vals).sum() / top_count * 100.0)
        
        # Broad market moving average breadth
        s50_all = sma_50.loc[dt].dropna()
        c_all = closes.loc[dt].reindex(s50_all.index).dropna()
        s50_all = s50_all.reindex(c_all.index)
        broad_50 = float((c_all > s50_all).sum() / len(s50_all) * 100.0) if len(s50_all) > 0 else 0.0

        e20_all = ema_20.loc[dt].reindex(c_all.index).dropna()
        broad_20 = float((c_all.reindex(e20_all.index) > e20_all).sum() / len(e20_all) * 100.0) if len(e20_all) > 0 else 0.0

        s200_all = sma_200.loc[dt].reindex(c_all.index).dropna()
        broad_200 = float((c_all.reindex(s200_all.index) > s200_all).sum() / len(s200_all) * 100.0) if len(s200_all) > 0 else 0.0

        spread = above_50 - broad_50
        
        records.append({
            'date': dt.strftime('%Y-%m-%d'),
            'top100_above_21_ema': round(above_21, 1),
            'top100_above_50_sma': round(above_50, 1),
            'broad_above_20_ema': round(broad_20, 1),
            'broad_above_50_sma': round(broad_50, 1),
            'broad_above_200_sma': round(broad_200, 1),
            'leadership_spread': round(spread, 1)
        })
        
    if not records:
        return pd.DataFrame()
        
    res_df = pd.DataFrame(records).set_index('date')
    res_df.index = pd.to_datetime(res_df.index)
    return res_df


try:
    import streamlit as st
    _cache_lead = st.cache_data(ttl=900, show_spinner=False)
except Exception:
    _cache_lead = lambda f: f


def _compute_heavy_dist_break_10d(
    c_top: pd.DataFrame,
    s50_top: pd.DataFrame,
    v_top: pd.DataFrame | None,
    h_top: pd.DataFrame | None,
    l_top: pd.DataFrame | None,
    vol_avg_50_s: pd.Series,
) -> pd.Series:
    """
    Detects unreclaimed heavy-volume distribution breakdowns below the 50 SMA over the trailing 10 sessions.
    Rescues high-volume absorption hammers closing in the upper range (DCR >= 45%).
    """
    import numpy as np

    cols = list(c_top.columns)
    out = pd.Series(False, index=cols, dtype=bool)
    if v_top is None or c_top.empty or len(c_top) < 2:
        return out

    win = min(10, len(c_top))
    c_w = c_top.tail(win)
    pc_w = c_top.shift(1).tail(win)
    s50_w = s50_top.tail(win)
    v_w = v_top.tail(win)
    v_ref = vol_avg_50_s.reindex(cols).replace(0, np.nan)

    if h_top is not None and l_top is not None:
        h_w = h_top.tail(win)
        l_w = l_top.tail(win)
        span_w = (h_w - l_w).replace(0, np.nan)
        dcr_w = (((c_w - l_w) / span_w) * 100.0).fillna(50.0)
    else:
        dcr_w = pd.DataFrame(50.0, index=c_w.index, columns=cols)

    is_heavy = v_w.ge(v_ref * 1.50, axis=1).fillna(False)
    bad_bar = (c_w < s50_w) & (c_w < pc_w) & (dcr_w < 45.0) & is_heavy
    reclaim_bar = (c_w >= s50_w) & v_w.ge(v_ref * 1.0, axis=1).fillna(False)

    for col in cols:
        bad_vals = bad_bar[col].to_numpy(dtype=bool)
        if not bad_vals.any():
            continue
        last_bad_pos = int(np.flatnonzero(bad_vals)[-1])
        reclaim_vals = reclaim_bar[col].to_numpy(dtype=bool)
        if not reclaim_vals[last_bad_pos + 1:].any():
            out.loc[col] = True
    return out


@_cache_lead
def _compute_leader_21ema_table_cached(
    market='IN',
    top_n=100,
    universe_scope='750',
    min_adtv_cr=None,
    _cache_v='v11_50sma_65ema',
    benchmark_symbol=None,
    _cache_sig=(0, 0),
):
    """Cached inner implementation keyed on matrix mtime+size cache signature (`_cache_sig`)."""
    import os
    import json
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from price_history_manager import (
        load_cached_price_matrix,
        extract_market_ohlcv_panels,
    )

    is_us = str(market).upper() in ['US', 'USA', 'US_IXIC']
    scope_clean = str(universe_scope).upper().strip()

    # 1. Select the correct price matrix file based on universe_scope (Gap 1 Fix)
    if scope_clean == 'ALL':
        matrix_path = 'historical_prices_matrix.pkl' if os.path.exists('historical_prices_matrix.pkl') else 'breadth_matrix.pkl'
    else:
        matrix_path = 'breadth_matrix.pkl' if os.path.exists('breadth_matrix.pkl') else 'historical_prices_matrix.pkl'

    if not os.path.exists(matrix_path):
        return pd.DataFrame()

    try:
        df = load_cached_price_matrix(matrix_path, cache_sig=_cache_sig)
    except Exception as e:
        print(f"[Breadth Engine] Matrix error in compute_leader_21ema_table: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    allowed_in_750 = None
    if (not is_us) and scope_clean == '750':
        n750 = set()
        try:
            from market_data import fetch_nifty_total_market_tickers
            n750 = set(fetch_nifty_total_market_tickers(show_progress=False))
        except Exception:
            pass
        if not n750 and os.path.exists('data/ind_niftytotalmarket_list.csv'):
            try:
                csv_df = pd.read_csv('data/ind_niftytotalmarket_list.csv')
                sym_col = 'Symbol' if 'Symbol' in csv_df.columns else csv_df.columns[0]
                n750 = {f"{str(s).strip()}.NS" for s in csv_df[sym_col].dropna()}
            except Exception:
                pass
        if n750:
            allowed_in_750 = sorted(n750)

    is_multi = isinstance(df.columns, pd.MultiIndex)
    panels = extract_market_ohlcv_panels(
        df,
        market='US' if is_us else 'IN',
        allowed_tickers=allowed_in_750,
        include_etfs=False,
        deduplicate_dual_in=True,
        min_coverage_ratio=0.35,
        min_abs=3,
    )
    closes = panels['close'].dropna(how='all')
    if closes.empty or len(closes) < 70:
        return pd.DataFrame()

    highs = panels['high'] if (is_multi and not panels['high'].empty and not panels['high'].isna().all().all()) else None
    lows = panels['low'] if (is_multi and not panels['low'].empty and not panels['low'].isna().all().all()) else None
    vols = panels['volume'] if (is_multi and not panels['volume'].empty and not panels['volume'].isna().all().all()) else None

    from down_day_engine import (
        is_etf_or_non_equity,
        get_cached_official_benchmark,
        get_point_in_time_rs_series,
        get_cumulative_thrust_map
    )

    min_price = 5.0 if is_us else 20.0
    if min_adtv_cr is not None:
        min_turnover = float(min_adtv_cr)
    else:
        min_turnover = 3.0 if is_us else (2.0 if scope_clean == '750' else 0.15)

    latest_dt = closes.index[-1]
    latest_all = closes.loc[latest_dt]
    # Guard only interior single-day halts (<=2 bars) without masking today-halted NaNs
    closes_calc = closes.ffill(limit=2)

    # 3. Robust 20D ADTV & Circuit-Lock Microstructure Hygiene (Gap 3 & Gap 4 Fix)
    scale_factor = 1e6 if is_us else 1e7
    if vols is not None:
        c_20 = closes.tail(20)
        v_20 = vols.tail(20)
        daily_turn_20 = (c_20 * v_20) / scale_factor
        mean_turn_20 = daily_turn_20.mean(axis=0).fillna(0.0)
        med_turn_20 = daily_turn_20.median(axis=0).fillna(0.0)
        # Cap single-day block deal distortion at 2.0x median daily turnover
        turnover_20d = np.minimum(mean_turn_20, med_turn_20 * 2.0)
        active_vol_days = (v_20 > 0).sum(axis=0).fillna(0)
    else:
        turnover_20d = pd.Series(999.0, index=closes.columns)
        mean_turn_20 = turnover_20d
        med_turn_20 = turnover_20d
        active_vol_days = pd.Series(20, index=closes.columns)

    if highs is not None and lows is not None:
        h_20 = highs.tail(20)
        l_20 = lows.tail(20)
        c_20_valid = closes.tail(20).notna()
        # Intraday range < 0.05% indicates locked upper/lower circuit bar
        is_circuit_bar = (((h_20 - l_20) / l_20.replace(0, np.nan)).abs() < 0.0005) & c_20_valid
        circuit_days_20d = is_circuit_bar.sum(axis=0).fillna(0)
    else:
        circuit_days_20d = pd.Series(0, index=closes.columns)

    daily_rets = closes.pct_change(fill_method=None)
    if is_us and vols is not None:
        vma_65 = vols.rolling(20, min_periods=5).mean().tail(65)
        # Preserve genuine US high-volume Episodic Pivots (e.g. MRNA/KOD/SLN); reject synthetic thin splits
        has_split_jump = ((daily_rets.tail(65).abs() > 0.85) & (vols.tail(65) < vma_65 * 2.0)).any(axis=0).fillna(False)
    else:
        has_split_jump = (daily_rets.tail(65).abs() > 0.85).any(axis=0).fillna(False)

    # 4. Benchmark return series & Stale Index Protection (C7 Fix)
    bench_series, _ = get_cached_official_benchmark(
        market='US' if is_us else 'INDIA',
        benchmark_symbol=benchmark_symbol,
    )
    bench_stale_days = 0
    bench_last_dt = latest_dt
    if bench_series is not None and not bench_series.empty:
        bench_series = bench_series.copy()
        bench_series.index = pd.to_datetime(bench_series.index)
        if bench_series.index.tz is not None:
            bench_series.index = bench_series.index.tz_localize(None)
        bench_series.index = bench_series.index.normalize()
        bench_last_dt = bench_series.index.max()
        bench_stale_days = int((closes.index > bench_last_dt).sum())
        bench_ret = bench_series.pct_change(fill_method=None).reindex(daily_rets.index)
    else:
        bench_ret = daily_rets.median(axis=1)

    outperform_all = daily_rets.tail(40).gt(bench_ret.tail(40), axis=0) & daily_rets.tail(40).notna()
    valid_all = daily_rets.tail(40).notna()

    rai_15d_cnt_s = outperform_all.tail(15).sum(axis=0)
    rai_15d_pct_s = (rai_15d_cnt_s / valid_all.tail(15).sum(axis=0).clip(lower=1)) * 100.0

    rai_20d_count_s = outperform_all.tail(20).sum(axis=0)
    rai_20d_pct_s = (rai_20d_count_s / valid_all.tail(20).sum(axis=0).clip(lower=1)) * 100.0

    rai_40d_cnt_s = outperform_all.tail(40).sum(axis=0)
    rai_40d_pct_s = (rai_40d_cnt_s / valid_all.tail(40).sum(axis=0).clip(lower=1)) * 100.0

    # 52-Week (252-Bar) High Proximity (%) — Both Intraday High & Closing High
    c_252 = closes_calc.tail(252)
    close_52w = c_252.max(axis=0)
    if highs is not None:
        h_252 = highs.tail(252)
        high_52w = h_252.max(axis=0)
        price_peak_idx_s = h_252.idxmax(axis=0)
    else:
        high_52w = close_52w
        price_peak_idx_s = c_252.idxmax(axis=0)

    ema_21 = closes_calc.ewm(span=21, adjust=False).mean()
    sma_50 = closes_calc.rolling(window=50, min_periods=35).mean()
    ema_65 = closes_calc.ewm(span=65, adjust=False, min_periods=45).mean()
    sma_200 = closes_calc.rolling(window=200, min_periods=140).mean()

    # Vectorized Slope Lookbacks & Bar Counts (D3/D4/D5)
    e21_5d = ema_21.shift(5)
    s50_10d = sma_50.shift(10)
    e65_10d = ema_65.shift(10)
    s200_20d = sma_200.shift(20)
    valid_bars_s = closes.notna().sum(axis=0)

    # Item 2: Vectorized O'Neil RS Line (Stock Close / Benchmark Close) 52W, 3M & 5D High Memory
    if bench_series is not None and not bench_series.empty:
        bench_close_aligned = bench_series.reindex(closes.index).ffill().bfill()
    else:
        bench_close_aligned = (1.0 + daily_rets.median(axis=1).fillna(0.0)).cumprod()

    b_252 = bench_close_aligned.tail(252).replace(0, np.nan)
    rs_line_panel = c_252.div(b_252, axis=0)
    # When benchmark index lags the stock matrix, evaluate new RS highs strictly up to settled bench_last_dt (C7)
    rs_eval_panel = rs_line_panel.loc[:bench_last_dt] if (bench_stale_days >= 1 and not rs_line_panel.loc[:bench_last_dt].empty) else rs_line_panel
    rs_line_cur_s = rs_eval_panel.iloc[-1]
    rs_line_5d_max_s = rs_eval_panel.tail(5).max(axis=0)
    rs_line_max_252_s = rs_eval_panel.max(axis=0)
    rs_line_max_63_s = rs_eval_panel.tail(63).max(axis=0)
    rs_line_dist_252_s = (((rs_line_cur_s - rs_line_max_252_s) / rs_line_max_252_s.replace(0, np.nan)) * 100.0).fillna(-99.0)
    rs_line_5d_dist_252_s = (((rs_line_5d_max_s - rs_line_max_252_s) / rs_line_max_252_s.replace(0, np.nan)) * 100.0).fillna(-99.0)
    rs_line_dist_63_s = (((rs_line_cur_s - rs_line_max_63_s) / rs_line_max_63_s.replace(0, np.nan)) * 100.0).fillna(-99.0)
    rs_line_5d_dist_63_s = (((rs_line_5d_max_s - rs_line_max_63_s) / rs_line_max_63_s.replace(0, np.nan)) * 100.0).fillna(-99.0)

    # Item 5: Vectorized 50-Day Up/Down Volume Ratio (UDVR), 10D Pocket Pivot & Volume Dry-Up (VDU)
    if vols is not None:
        v_60 = vols.tail(60)
        c_diff_60 = closes_calc.tail(60).diff()
        up_vol_50 = v_60.where(c_diff_60 > 0, 0.0).tail(50).sum(axis=0)
        dn_vol_50 = v_60.where(c_diff_60 < 0, 0.0).tail(50).sum(axis=0).replace(0, np.nan)
        udvr_50_s = (up_vol_50 / dn_vol_50).fillna(1.0).clip(0.1, 9.9)

        v_50_win = v_60.tail(50)
        v_50_med = v_50_win.median(axis=0).replace(0, np.nan)
        vol_avg_50_s = v_50_win.clip(upper=v_50_med * 3.0, axis=1).mean(axis=0).fillna(v_50_win.mean(axis=0)).replace(0, np.nan)
        cur_vol_s = v_60.iloc[-1]
        dn_vol_only = v_60.where(c_diff_60 < 0, 0.0)
        max_dn_vol_10_s = dn_vol_only.iloc[-11:-1].max(axis=0).fillna(0.0) if len(dn_vol_only) >= 11 else pd.Series(0.0, index=closes.columns)
        is_up_day_s = (c_diff_60.iloc[-1] > 0).fillna(False)
        is_pocket_pivot_s = (is_up_day_s & (cur_vol_s > max_dn_vol_10_s) & (cur_vol_s >= vol_avg_50_s * 0.85)).fillna(False)
        is_vdu_s = ((cur_vol_s <= vol_avg_50_s * 0.55) & (cur_vol_s > 0)).fillna(False)
    else:
        udvr_50_s = pd.Series(1.0, index=closes.columns)
        vol_avg_50_s = pd.Series(np.nan, index=closes.columns)
        is_pocket_pivot_s = pd.Series(False, index=closes.columns)
        is_vdu_s = pd.Series(False, index=closes.columns)

    # 5. Short-Term Momentum Composite (Strict Lock: 20% 1W + 40% 1M + 40% 3M)
    ret_1w = closes_calc.pct_change(5, fill_method=None)
    ret_1m = closes_calc.pct_change(21, fill_method=None)
    ret_3m = closes_calc.pct_change(63, fill_method=None)
    rs_raw = (ret_1w * 0.20) + (ret_1m * 0.40) + (ret_3m * 0.40)

    rs_series = rs_raw.loc[latest_dt].dropna()
    if rs_series.empty:
        return pd.DataFrame()

    # 6. Pre-Ranking Candidate Eligibility Filter
    valid_tickers = []
    for t in rs_series.index:
        clean_t = str(t).replace('.NS', '').replace('.BO', '')
        if str(t).startswith('^') or is_etf_or_non_equity(clean_t):
            continue
        if bool(has_split_jump.get(t, False)):
            continue
        c_val = float(latest_all.get(t, np.nan))
        if pd.isna(c_val) or c_val < min_price:
            continue
        if int(active_vol_days.get(t, 0)) < 16:
            continue
        if int(circuit_days_20d.get(t, 0)) >= 4:
            continue
        t_cr = float(turnover_20d.get(t, 0.0))
        if pd.isna(t_cr) or t_cr < min_turnover:
            continue
        valid_tickers.append(t)

    if not valid_tickers:
        return pd.DataFrame()

    rs_clean_series = rs_series.loc[valid_tickers]
    top_count = min(int(top_n), len(rs_clean_series))
    top_tickers = rs_clean_series.nlargest(top_count).index

    heavy_dist_10d_s = _compute_heavy_dist_break_10d(
        c_top=closes_calc[top_tickers],
        s50_top=sma_50[top_tickers],
        v_top=vols[top_tickers] if vols is not None else None,
        h_top=highs[top_tickers] if highs is not None else None,
        l_top=lows[top_tickers] if lows is not None else None,
        vol_avg_50_s=vol_avg_50_s,
    )

    # 7. Enrichments: Risk, Industry Intelligence + Fallback, Beach Ball, RS Rating
    # Vectorized 3M Sortino Ratio (63 trading days, MAR = 4.5% US / 6.5% India)
    mar_ann = 0.045 if is_us else 0.065
    daily_mar = mar_ann / 252.0
    rets_63 = daily_rets[top_tickers].tail(63)
    ann_ret_s = rets_63.mean(axis=0) * 252.0
    down_diff = np.minimum(0.0, rets_63 - daily_mar)
    down_dev_s = np.sqrt((down_diff ** 2).mean(axis=0)) * np.sqrt(252.0)
    sortino_3m_s = ((ann_ret_s - mar_ann) / down_dev_s.replace(0, np.nan)).fillna(0.0).clip(-10.0, 25.0)

    # Vectorized Smooth RS / ATR (Composite 1M/3M/6M Return % divided by 20D ATR %)
    c_top = closes_calc[top_tickers]
    cur_c_s = c_top.iloc[-1]
    r21_s = (cur_c_s / c_top.iloc[max(0, len(c_top) - 22)].replace(0, np.nan)) - 1.0
    r63_s = (cur_c_s / c_top.iloc[max(0, len(c_top) - 64)].replace(0, np.nan)) - 1.0
    if len(c_top) >= 127:
        r126_s = (cur_c_s / c_top.iloc[-127].replace(0, np.nan)) - 1.0
        comp_rs_pct_s = np.where(
            r126_s.notna(),
            (0.40 * r21_s + 0.35 * r63_s + 0.25 * r126_s) * 100.0,
            (0.55 * r21_s + 0.45 * r63_s) * 100.0
        )
        comp_rs_pct_s = pd.Series(comp_rs_pct_s, index=top_tickers)
    else:
        comp_rs_pct_s = (0.55 * r21_s + 0.45 * r63_s) * 100.0

    if highs is not None and lows is not None:
        h_21 = highs[top_tickers].tail(21)
        l_21 = lows[top_tickers].tail(21)
        pc_21 = c_top.tail(21).shift(1)
        tr_20 = np.maximum(h_21 - l_21, np.maximum((h_21 - pc_21).abs(), (l_21 - pc_21).abs())).tail(20)
        atr_pct_s = ((tr_20.mean(axis=0) / cur_c_s.replace(0, np.nan)) * 100.0).fillna(2.5)
    else:
        atr_pct_s = (daily_rets[top_tickers].tail(20).abs().mean(axis=0) * 125.0).fillna(2.5)

    smooth_rs_s = (comp_rs_pct_s / atr_pct_s.clip(lower=1.0)).fillna(0.0).clip(-10.0, 40.0)

    # USIC Edge 1, 2 & 3: Vectorized 20D ADR %, VCP Price Tightness (ID, NR7, 5D Coil, 3WT), DCR % & RVOL
    if highs is not None and lows is not None:
        h_top_21 = highs[top_tickers].tail(21)
        l_top_21 = lows[top_tickers].tail(21)
        # 20D ADR % = 20-day mean of ((High - Low) / Low) * 100
        adr_20d_s = (((h_top_21 - l_top_21) / l_top_21.replace(0, np.nan)).tail(20).mean(axis=0) * 100.0).fillna(atr_pct_s).clip(0.5, 25.0)

        # Daily Closing Range (DCR %): (Close - Low) / (High - Low) * 100 on latest bar
        cur_h_s = h_top_21.iloc[-1]
        cur_l_s = l_top_21.iloc[-1]
        bar_span_s = (cur_h_s - cur_l_s).replace(0, np.nan)
        dcr_pct_s = (((cur_c_s - cur_l_s) / bar_span_s) * 100.0).fillna(50.0).clip(0.0, 100.0)

        # Inside Day (ID): Today's High <= Yesterday's High AND Today's Low >= Yesterday's Low
        if len(h_top_21) >= 2:
            prev_h_s = h_top_21.iloc[-2]
            prev_l_s = l_top_21.iloc[-2]
            is_inside_day_s = ((cur_h_s <= prev_h_s * 1.0005) & (cur_l_s >= prev_l_s * 0.9995) & (cur_h_s > cur_l_s)).fillna(False)
        else:
            is_inside_day_s = pd.Series(False, index=top_tickers)

        # NR7 (Narrowest Range of last 7 sessions)
        if len(h_top_21) >= 7:
            rng_7 = ((h_top_21.tail(7) - l_top_21.tail(7)) / c_top.tail(7).replace(0, np.nan)) * 100.0
            cur_rng_s = rng_7.iloc[-1]
            min_rng_7_s = rng_7.min(axis=0)
            is_nr7_s = ((cur_rng_s <= min_rng_7_s * 1.01) & (cur_rng_s > 0)).fillna(False)
        else:
            is_nr7_s = pd.Series(False, index=top_tickers)

        # 5-Day VCP Coil Range %: (Max High of 5D - Min Low of 5D) / Latest Close * 100
        if len(h_top_21) >= 5:
            h_5d_s = h_top_21.tail(5).max(axis=0)
            l_5d_s = l_top_21.tail(5).min(axis=0)
            range_5d_pct_s = (((h_5d_s - l_5d_s) / cur_c_s.replace(0, np.nan)) * 100.0).fillna(99.0)
        else:
            range_5d_pct_s = pd.Series(99.0, index=top_tickers)
    else:
        adr_20d_s = atr_pct_s.clip(0.5, 25.0)
        dcr_pct_s = pd.Series(50.0, index=top_tickers)
        is_inside_day_s = pd.Series(False, index=top_tickers)
        is_nr7_s = pd.Series(False, index=top_tickers)
        range_5d_pct_s = (((c_top.tail(5).max(axis=0) - c_top.tail(5).min(axis=0)) / cur_c_s.replace(0, np.nan)) * 100.0).fillna(99.0)

    # 3-Weeks Tight (3WT): Last 3 weekly closes (t, t-5, t-10) within <= 1.6% of each other
    if len(c_top) >= 11:
        w0 = c_top.iloc[-1]
        w1 = c_top.iloc[-6]
        w2 = c_top.iloc[-11]
        w_max = np.maximum(w0, np.maximum(w1, w2))
        w_min = np.minimum(w0, np.minimum(w1, w2))
        three_wk_spread_s = (((w_max - w_min) / w_min.replace(0, np.nan)) * 100.0).fillna(99.0)
        is_3wt_s = (three_wk_spread_s <= 1.6).fillna(False)
    else:
        is_3wt_s = pd.Series(False, index=top_tickers)

    # Today's RVOL (Today's Volume / Block-Deal Capped 50D Average Volume)
    if vols is not None:
        v_top_50 = vols[top_tickers].tail(50)
        v_med_50_top = v_top_50.median(axis=0).replace(0, np.nan)
        v_avg_50_top = v_top_50.clip(upper=v_med_50_top * 3.0, axis=1).mean(axis=0).fillna(v_top_50.mean(axis=0)).replace(0, np.nan)
        rvol_today_s = (v_top_50.iloc[-1] / v_avg_50_top).fillna(1.0).clip(0.05, 15.0)
    else:
        rvol_today_s = pd.Series(1.0, index=top_tickers)

    try:
        from industry_group_engine import get_stock_group_intelligence
        intel = get_stock_group_intelligence(
            taxonomy="canonical",
            universe_scope="US (S&P 500 / Nasdaq)" if is_us else "India (NSE/BSE)"
        )
    except Exception:
        intel = {}

    fallback_ind_map = {}
    if (not is_us) and os.path.exists('industry_map.json'):
        try:
            with open('industry_map.json', 'r', encoding='utf-8') as f:
                fallback_ind_map = json.load(f)
        except Exception:
            pass

    us_gics = {}
    us_sm_ind = {}
    us_tv_map = {}
    if is_us and os.path.exists('data/sp500_gics_map.json'):
        try:
            with open('data/sp500_gics_map.json', 'r', encoding='utf-8') as f:
                us_gics = json.load(f)
        except Exception:
            pass
    format_tradingview_symbol = None
    if is_us:
        try:
            from industry_group_engine import load_exporter_enrichment_maps, format_tradingview_symbol
            us_tv_map, _ = load_exporter_enrichment_maps(is_us=True)
        except Exception:
            format_tradingview_symbol = None
            us_tv_map = {}

    try:
        bb_mkt_arg = "US_IXIC" if (is_us and str(benchmark_symbol or '').upper() == '^IXIC') else ("US" if is_us else "INDIA")
        bb_map = get_cumulative_thrust_map(market=bb_mkt_arg)
    except Exception:
        bb_map = {}

    matrix_dt_str = latest_dt.strftime('%Y-%m-%d') if hasattr(latest_dt, 'strftime') else str(latest_dt)[:10]
    try:
        from price_history_manager import get_latest_completed_session_date
        today_str = get_latest_completed_session_date('US' if is_us else 'IN')
    except Exception:
        today_str = datetime.now().strftime('%Y-%m-%d')
    try:
        rs_map = get_point_in_time_rs_series(matrix_dt_str, market="US" if is_us else "INDIA")
        if (rs_map is None or rs_map.empty) and today_str != matrix_dt_str:
            rs_map = get_point_in_time_rs_series(today_str, market="US" if is_us else "INDIA")
    except Exception:
        rs_map = pd.Series(dtype=float)

    # Gap 6: Strictly partition fundamentals_cache by market suffix so HAL/META/CCL/PTC never collide!
    funda_map = {}
    spons_map = {}
    try:
        from database import get_fundamentals_map_for_engine, read_sql_df

        mkt_f = "US" if is_us else "IN"
        raw_funda, _ = get_fundamentals_map_for_engine(mkt_f)
        for raw_tk, info in (raw_funda or {}).items():
            if not raw_tk or not isinstance(info, dict):
                continue
            sym = str(raw_tk).replace('.NS', '').replace('.BO', '').upper()
            funda_map.setdefault(sym, {
                'sales_growth': float(info.get('sales_growth', 0.0) or 0.0),
                'eps_growth': float(info.get('eps_growth', 0.0) or 0.0),
                'roe': float(info.get('roe', 0.0) or 0.0),
                'market_cap': float(info.get('market_cap', 0.0) or 0.0),
                'industry': str(info.get('industry') or ''),
            })

        if is_us:
            try:
                df_sm = read_sql_df(
                    "SELECT yf_ticker, industry FROM security_master WHERE market='US' AND status='ACTIVE' AND industry IS NOT NULL"
                )
                if df_sm is not None and not df_sm.empty:
                    for tk_sm, ind_sm in zip(df_sm["yf_ticker"].tolist(), df_sm["industry"].tolist()):
                        if tk_sm and ind_sm:
                            us_sm_ind[str(tk_sm).upper()] = str(ind_sm)
            except Exception:
                pass
        else:
            try:
                df_sp = read_sql_df(
                    "SELECT symbol, sponsorship_grade, inst_change_qoq, total_inst_pct FROM institutional_sponsorship"
                )
                if df_sp is not None and not df_sp.empty:
                    for row_d in df_sp.to_dict(orient="records"):
                        sym = str(row_d.get("symbol") or "").replace('.NS', '').replace('.BO', '').upper()
                        if sym:
                            spons_map[sym] = {
                                'grade': str(row_d.get('sponsorship_grade') or 'C'),
                                'inst_change': float(row_d.get('inst_change_qoq') or 0.0),
                                'inst_pct': float(row_d.get('total_inst_pct') or 0.0),
                            }
            except Exception:
                pass
    except Exception as e:
        print(f"[Breadth Engine] Funda/Spons error: {e}")

    records = []
    for rank_idx, t in enumerate(top_tickers, 1):
        c = float(closes.loc[latest_dt, t])
        e21 = float(ema_21.loc[latest_dt, t])
        s50 = float(sma_50.loc[latest_dt, t]) if pd.notna(sma_50.loc[latest_dt, t]) else np.nan
        e65 = float(ema_65.loc[latest_dt, t]) if pd.notna(ema_65.loc[latest_dt, t]) else np.nan
        d21 = ((c - e21) / e21) * 100.0 if e21 > 0 else 0.0
        d50 = ((c - s50) / s50) * 100.0 if (not np.isnan(s50) and s50 > 0) else np.nan
        d65 = ((c - e65) / e65) * 100.0 if (not np.isnan(e65) and e65 > 0) else np.nan

        h52 = float(high_52w.get(t, np.nan))
        c52 = float(close_52w.get(t, np.nan))
        d52w = ((c - h52) / h52) * 100.0 if (not np.isnan(h52) and h52 > 0) else 0.0
        d52w_close = ((c - c52) / c52) * 100.0 if (not np.isnan(c52) and c52 > 0) else 0.0

        clean_t = str(t).replace('.NS', '').replace('.BO', '')

        s200 = float(sma_200.loc[latest_dt, t]) if pd.notna(sma_200.loc[latest_dt, t]) else 0.0
        d200 = ((c - s200) / s200) * 100.0 if s200 > 0 else 0.0
        is_stage2 = bool((not np.isnan(s50)) and c >= s50 and (s200 <= 0.0 or (c >= s200 and s50 >= s200)))
        is_fresh_stage2 = bool(is_stage2 and (0.0 <= d200 <= 60.0))
        is_climax_ext = bool(d200 > 100.0)

        # Item 2: True MarketSurge / O'Neil RS Line Detection (Wick-Safe + 5D Memory + Peak RS Comparison)
        rs_d252 = float(rs_line_dist_252_s.get(t, -99.0))
        rs_5d_d252 = float(rs_line_5d_dist_252_s.get(t, -99.0))
        rs_d63 = float(rs_line_dist_63_s.get(t, -99.0))
        rs_5d_d63 = float(rs_line_5d_dist_63_s.get(t, -99.0))

        # Check RS Line level today vs RS Line level on the date Price hit its 52W High
        pk_dt = price_peak_idx_s.get(t, None)
        rs_cur_val = float(rs_line_cur_s.get(t, 0.0))
        rs_at_pk_val = float(rs_eval_panel.loc[pk_dt, t]) if (pk_dt is not None and pk_dt in rs_eval_panel.index) else 0.0
        rs_above_price_peak = bool(rs_at_pk_val > 0 and pd.notna(rs_cur_val) and rs_cur_val >= rs_at_pk_val * 0.995)

        # Price is at a 52W Breakout if within 1.5% of 52W Intraday High OR at 52W Closing High (within 0.5% and d52w >= -3.0%)
        is_price_at_52w = bool(d52w >= -1.5 or (d52w_close >= -0.5 and d52w >= -3.0))
        is_rs_52w_high = bool(rs_d252 >= -0.75 or (rs_5d_d252 >= -0.5 and rs_d252 >= -4.5) or rs_above_price_peak)
        is_rs_3m_high = bool(rs_d63 >= -0.75 or (rs_5d_d63 >= -0.5 and rs_d63 >= -3.5))
        is_blue_dot = bool((not is_price_at_52w) and (d52w >= -25.0) and is_rs_52w_high and bench_stale_days == 0)

        if is_blue_dot:
            rs_line_badge = "🔵 Blue Dot (RS Leading Price)"
        elif is_rs_52w_high and is_price_at_52w:
            rs_line_badge = "🟢 RS & Price 52W High"
        elif is_rs_52w_high:
            rs_line_badge = "🔵 Blue Dot (RS Leading Price)"
        elif is_rs_3m_high and not is_price_at_52w:
            rs_line_badge = "🔷 3M RS High Before Price"
        elif is_rs_3m_high:
            rs_line_badge = "🔹 3M RS Line High"
        else:
            rs_line_badge = "—"

        rs_rating_val = float(rs_map.get(t, rs_map.get(clean_t, 0.0)) or 0.0)
        if rs_rating_val == 0.0 and len(rs_clean_series) > 1:
            rs_rating_val = round(float((rs_clean_series < rs_clean_series.get(t, 0.0)).mean() * 99.0), 1)
        elif rs_rating_val == 0.0:
            rs_rating_val = 90.0

        s3m = float(sortino_3m_s.get(t, 0.0))
        sm_rs = float(smooth_rs_s.get(t, 0.0))

        clean_up = str(clean_t).upper()
        info = (
            intel.get(clean_t)
            or intel.get(t)
            or intel.get(clean_up)
            or intel.get(clean_up.replace('.', '-'))
            or intel.get(clean_up.replace('-', '.'))
            or (intel.get(clean_t + '.NS') if not is_us else None)
            or (intel.get(clean_t + '.BO') if not is_us else None)
        )
        if isinstance(info, dict) and info:
            ind_name = info.get('group_name', info.get('industry', 'Broad Market'))
            grd = info.get('group_grade', info.get('grade', 'B'))
            rnk = info.get('group_rank', info.get('rank', 99))
            is_top_20 = bool(info.get('is_top_20_pct', grd in ['A+', 'A'] or rnk <= 22))
            ind_badge = f"{ind_name} ({grd} #{rnk})"
            grp_role = info.get('stock_role', '—')
            role_code = info.get('role_code', 'UNCLASSIFIED')
        elif is_us and (clean_t.upper() in us_sm_ind or clean_t.upper() in us_gics or funda_map.get(clean_t.upper(), {}).get('industry')):
            ind_name = us_sm_ind.get(clean_t.upper()) or (us_gics.get(clean_t, {}).get('industry') or us_gics.get(clean_t, {}).get('sector')) or funda_map.get(clean_t.upper(), {}).get('industry') or 'US Equities'
            grd = 'NR'
            rnk = 99
            is_top_20 = False
            ind_badge = f"{ind_name} (NR)"
            grp_role = '— Unranked Group'
            role_code = 'UNCLASSIFIED'
        else:
            fb_ind = None if is_us else (fallback_ind_map.get(t) or fallback_ind_map.get(clean_t + '.NS') or fallback_ind_map.get(clean_t))
            ind_name = fb_ind if fb_ind else 'Emerging / Unclassified'
            grd = 'NR'
            rnk = 99
            is_top_20 = False
            ind_badge = f"{ind_name} (NR)"
            grp_role = '— Unranked Group'
            role_code = 'UNCLASSIFIED'

        # Item 5: Volume Accumulation Metrics (50D U/D Volume Ratio, Pocket Pivot, VDU, Beach Ball)
        udvr_val = float(udvr_50_s.get(t, 1.0))
        is_pp = bool(is_pocket_pivot_s.get(t, False))
        is_vdu_raw = bool(is_vdu_s.get(t, False))
        is_vdu = bool(is_vdu_raw and (not np.isnan(s50)) and c >= s50)

        bb_info = bb_map.get(clean_t, {}) or bb_map.get(t, {})
        bb_cnt = int(bb_info.get('count_3m', 0) or 0)
        has_bb = bb_cnt > 0 or bool(bb_info.get('has_thrust', False))

        vol_tags = []
        if has_bb:
            vol_tags.append(f"👑 BB ({bb_cnt}x)" if bb_cnt >= 1 else "👑 BB Active")
        if is_pp:
            vol_tags.append("⚡ Pocket Pivot")
        elif is_vdu:
            vol_tags.append("🔇 VDU Dry-Up")
        apex_badge = " • ".join(vol_tags) if vol_tags else "—"

        # Item 1: Full CANSLIM Techno-Fundamental Confluence (EPS + Sales + ROE + Inst/UDVR)
        f_data = funda_map.get(clean_t.upper(), {})
        s_data = spons_map.get(clean_t.upper(), {})
        sales_g = float(f_data.get('sales_growth', 0.0) or 0.0)
        eps_g = float(f_data.get('eps_growth', 0.0) or 0.0)
        roe_val = float(f_data.get('roe', 0.0) or 0.0)
        sp_grade = str(s_data.get('grade', 'NR') or 'NR')
        sp_change = float(s_data.get('inst_change', 0.0) or 0.0)

        has_high_sales = bool(sales_g >= 25.0)
        has_high_eps = bool(eps_g >= 25.0)
        has_double_25 = bool(eps_g >= 25.0 and sales_g >= 25.0)
        has_quality_roe = bool(roe_val >= 15.0 and eps_g >= 0.0)
        has_inst_accum = bool(sp_grade in ['A+', 'A'] or sp_change >= 0.5 or (udvr_val >= 1.25 and has_bb) or udvr_val >= 1.50)

        # True Market Leader (TML) Confluence: Strong C+A growth supported by ROE >= 15% or Institutional Accumulation
        is_c_and_a_strong = bool((eps_g >= 25.0 and sales_g >= 20.0) or (sales_g >= 35.0 and eps_g >= 15.0) or (eps_g >= 50.0 and sales_g >= 15.0))
        is_tml_confluence = bool(is_c_and_a_strong and (roe_val >= 15.0 or has_inst_accum) and roe_val >= 0.0)

        # Build transparent, full-context CANSLIM badge
        parts = []
        if eps_g != 0.0:
            e_disp = f"+{eps_g:.0f}%" if eps_g > 0 else f"{eps_g:.0f}%"
            parts.append(f"EPS {e_disp}")
        if sales_g != 0.0:
            s_disp = f"+{sales_g:.0f}%" if sales_g > 0 else f"{sales_g:.0f}%"
            parts.append(f"Sales {s_disp}")
        if roe_val != 0.0:
            parts.append(f"ROE {roe_val:.0f}%")
        if sp_grade in ['A+', 'A', 'B']:
            parts.append(f"Inst {sp_grade}")
        if sp_change != 0.0:
            parts.append(f"Inst {sp_change:+.1f}% QoQ")

        core_str = " • ".join(parts)
        if is_tml_confluence:
            funda_badge = f"💎 TML ({core_str})"
        elif roe_val < 0.0 or eps_g < 0.0:
            funda_badge = f"⚠️ {core_str}" if core_str else "—"
        elif has_double_25 or has_high_eps or has_high_sales:
            funda_badge = f"🚀 {core_str}"
        elif sp_grade in ['A+', 'A']:
            funda_badge = f"🏛️ {core_str}"
        elif core_str:
            funda_badge = core_str
        else:
            funda_badge = "—"

        # Market Cap Tier & Supply Classification
        raw_mcap = float(f_data.get('market_cap', 0.0) or 0.0)
        if not is_us:
            mcap_cr = raw_mcap / 1e7 if raw_mcap > 0 else 0.0
            if mcap_cr >= 40000.0:
                cap_tier = "Large Cap"
                mcap_disp = f"Large ₹{mcap_cr/1000.0:.1f}k Cr"
            elif mcap_cr >= 10000.0:
                cap_tier = "Mid Cap"
                mcap_disp = f"Mid ₹{mcap_cr/1000.0:.1f}k Cr"
            elif mcap_cr >= 1500.0:
                cap_tier = "Small Cap"
                mcap_disp = f"Small ₹{mcap_cr/1000.0:.1f}k Cr"
            elif mcap_cr > 0.0:
                cap_tier = "Micro Cap"
                mcap_disp = f"Micro ₹{mcap_cr:.0f} Cr"
            else:
                cap_tier = "Mid Cap"
                mcap_disp = "—"
        else:
            mcap_b = raw_mcap / 1e9 if raw_mcap > 0 else 0.0
            if mcap_b >= 100.0:
                cap_tier = "Mega Cap"
                mcap_disp = f"Mega ${mcap_b:.0f}B"
            elif mcap_b >= 15.0:
                cap_tier = "Large Cap"
                mcap_disp = f"Large ${mcap_b:.1f}B"
            elif mcap_b >= 2.0:
                cap_tier = "Mid Cap"
                mcap_disp = f"Mid ${mcap_b:.1f}B"
            elif mcap_b > 0.0:
                cap_tier = "Small Cap"
                mcap_disp = f"Small ${mcap_b*1000.0:.0f}M"
            else:
                cap_tier = "Mid Cap"
                mcap_disp = "—"

        # Stage & Cap Badge (Mutually Exclusive Weinstein / Minervini Stage Classification)
        if s200 > 0:
            if d200 < 0.0:
                stg_label = f"⚠️ S4 Below 200D ({d200:.0f}%)"
            elif is_climax_ext:
                stg_label = f"⚠️ Climax Ext (+{d200:.0f}% 200D)"
            elif is_fresh_stage2:
                stg_label = f"🌱 Fresh S2 (+{d200:.0f}% 200D)"
            elif is_stage2:
                stg_label = f"🔥 Adv S2 (+{d200:.0f}% 200D)"
            elif c >= s50:
                stg_label = f"🔄 S1→S2 Turn (+{d200:.0f}% 200D)"
            else:
                stg_label = f"🛡️ S2 Pullback (+{d200:.0f}% 200D)"
        else:
            stg_label = "Stage 2" if is_stage2 else "Stage 4"

        cap_stage_badge = f"{stg_label} • {mcap_disp}" if mcap_disp != "—" else stg_label

        # USIC VCP Tightness, ADR & Bar Action
        adr_val = float(adr_20d_s.get(t, 2.5))
        dcr_val = float(dcr_pct_s.get(t, 50.0))
        rvol_val = float(rvol_today_s.get(t, 1.0))
        rng_5d_val = float(range_5d_pct_s.get(t, 99.0))
        is_id = bool(is_inside_day_s.get(t, False))
        is_nr7 = bool(is_nr7_s.get(t, False))
        is_3wt = bool(is_3wt_s.get(t, False))
        # Tight 5D VCP Coil requires both relative contraction (<= 1.5x ADR) and an absolute cap (<= 7.5% 5D range)
        is_5d_coil_raw = bool(rng_5d_val <= min(max(adr_val * 1.50, 4.0), 7.5))
        is_5d_coil = bool(is_5d_coil_raw and (not np.isnan(s50)) and c >= s50)
        is_id_nr7 = bool(is_id or is_nr7)
        is_vcp_tight = bool(is_id or is_nr7 or is_5d_coil or is_3wt)

        tight_tags = []
        if is_id and is_nr7:
            tight_tags.append("🎯 ID+NR7")
        elif is_id:
            tight_tags.append("🎯 Inside Day")
        elif is_nr7:
            tight_tags.append("🎯 NR7 Tight")
        if is_3wt:
            tight_tags.append("🧬 3Wks Tight")
        elif is_5d_coil and not (is_id or is_nr7):
            tight_tags.append(f"🌀 5D Coil ({rng_5d_val:.1f}%)")

        if dcr_val >= 75.0:
            bar_str = f"🟢 DCR {dcr_val:.0f}%"
        elif dcr_val <= 25.0:
            bar_str = f"⚠️ DCR {dcr_val:.0f}%"
        else:
            bar_str = f"DCR {dcr_val:.0f}%"

        if rvol_val >= 1.50:
            bar_str += f" ({rvol_val:.1f}x Vol)"
        elif rvol_val <= 0.55:
            bar_str += f" ({rvol_val:.1f}x Dry)"

        vcp_action_badge = f"{' • '.join(tight_tags)} • {bar_str}" if tight_tags else bar_str

        r15_pct = float(rai_15d_pct_s.get(t, 0.0))
        r15_cnt = int(rai_15d_cnt_s.get(t, 0))
        r20_pct = float(rai_20d_pct_s.get(t, 0.0))
        r20_cnt = int(rai_20d_count_s.get(t, 0))
        r40_pct = float(rai_40d_pct_s.get(t, 0.0))
        r40_cnt = int(rai_40d_cnt_s.get(t, 0))

        adtv_val = float(turnover_20d.get(t, 0.0))
        adtv_mean_val = float(mean_turn_20.get(t, 0.0))
        adtv_med_val = float(med_turn_20.get(t, 0.0))

        # Immutable 21-EMA zone flag (D8: preserves MarketPersona / Beach Ball parity)
        holding_21 = bool((not np.isnan(s50)) and d21 >= -2.0 and c >= s50)

        # Asymmetric slopes, mature 200 SMA & IPO checks (D3, D4, D5)
        e21_5d_val = e21_5d.loc[latest_dt, t] if latest_dt in e21_5d.index else np.nan
        s50_10d_val = s50_10d.loc[latest_dt, t] if latest_dt in s50_10d.index else np.nan
        e65_10d_val = e65_10d.loc[latest_dt, t] if latest_dt in e65_10d.index else np.nan
        s200_20d_val = s200_20d.loc[latest_dt, t] if latest_dt in s200_20d.index else np.nan
        n_bars = int(valid_bars_s.get(t, 0))

        e21_rising = bool(pd.notna(e21) and pd.notna(e21_5d_val) and e21 >= float(e21_5d_val))
        sma50_rising = bool((not np.isnan(s50)) and pd.notna(s50_10d_val) and s50 >= float(s50_10d_val))
        ema65_rising = bool((not np.isnan(e65)) and pd.notna(e65_10d_val) and e65 >= float(e65_10d_val))

        below_50 = bool(np.isnan(s50) or c < s50)

        if n_bars >= 220 and s200 > 0 and pd.notna(s200_20d_val):
            strict_s200 = bool(s200 >= float(s200_20d_val) * 0.995 and c >= s200)
            # Allow Stage 1->2 breakouts above 50 & 200 SMA where trailing 200D back-edge slightly flattens (<-2.5%/20D)
            stage2_breakout_s200 = bool(
                (not below_50)
                and c >= s200
                and s50 >= s200
                and (sma50_rising or ema65_rising or e21_rising)
                and s200 >= float(s200_20d_val) * 0.975
            )
            sma200_slope_ok = bool(strict_s200 or stage2_breakout_s200)
        elif s200 > 0:
            sma200_slope_ok = bool(c >= s200 and d52w >= -30.0)
        else:
            sma200_slope_ok = bool(d52w >= -25.0 and (not np.isnan(e65)) and e65 > 0)

        near_52w_ok = bool(d52w >= -25.0)
        slope_ok_50 = bool(sma50_rising and ema65_rising) if below_50 else bool(sma50_rising or e21_rising)
        curling_reclaim_65 = bool(
            e21_rising
            and c >= e21
            and pd.notna(e65_10d_val)
            and e65 >= float(e65_10d_val) * 0.995
        )
        slope_ok_65 = bool(ema65_rising or curling_reclaim_65) if below_50 else bool(ema65_rising or e21_rising)

        unreclaimed_10d = bool(heavy_dist_10d_s.get(t, False))
        heavy_dist_break = bool(below_50 and (unreclaimed_10d or udvr_val < 0.85))
        d50_atrx = round(((c - s50) / (s50 * max(adr_val, 0.5) / 100.0)), 2) if (not np.isnan(s50) and s50 > 0) else 0.0

        holding_50 = bool(
            (not np.isnan(d50))
            and d50 >= -4.0
            and sma200_slope_ok
            and slope_ok_50
            and (not heavy_dist_break)
        )
        holding_65 = bool(
            (not np.isnan(d65))
            and d65 >= -2.0
            and sma200_slope_ok
            and slope_ok_65
            and (not heavy_dist_break)
        )
        bear_pivot_pocket = bool(
            near_52w_ok
            and (
                (holding_50 and (not np.isnan(d50)) and (-4.0 <= d50 <= 3.0))
                or (holding_65 and (not np.isnan(d65)) and (-2.0 <= d65 <= 2.5))
            )
        )

        # Strict isolation of bull vs bear floor coils (C5)
        is_bear_vdu = bool(is_vdu_raw and below_50 and holding_65)
        is_bear_coil = bool(is_5d_coil_raw and below_50 and holding_65)

        if below_50 and (holding_50 or holding_65):
            if is_pp or (dcr_val >= 65.0 and d21 >= -2.0):
                setup_trigger_state = "⚡ Right-Side Reclaim / PP"
            elif is_id_nr7 or is_bear_coil or is_bear_vdu:
                setup_trigger_state = "🌀 Tight 65E/50S Floor Coil"
            else:
                setup_trigger_state = "👁️ Passive Support Watch"
        elif holding_21 and (is_pp or (dcr_val >= 65.0 and -2.0 <= d21 <= 3.0)):
            setup_trigger_state = "⚡ 21E Pullback / PP Trigger"
        else:
            setup_trigger_state = "—"

        if not holding_21:
            if heavy_dist_break:
                ma_status = "🚨 Heavy Vol Break < 50 SMA"
            elif holding_50:
                ma_status = "🛡️ Holding 50 SMA Floor (≥ -4%)"
            elif holding_65:
                ma_status = "📐 Holding Ross Haber 65 EMA"
            elif d21 >= -2.0 and below_50:
                ma_status = "⚠️ Below 50 SMA (Overhead Supply)"
            else:
                ma_status = "❌ Violated Trend"
        elif d21 >= 5.0:
            ma_status = "🚀 Power Extended (>+5%)"
        elif 3.0 < d21 < 5.0:
            ma_status = "🛡️ 21 EMA Cushion (+3-5%)"
        elif 0.0 <= d21 <= 3.0:
            ma_status = "🟢 21 EMA Pullback Pocket (0-3%)"
        else:
            ma_status = "🧲 21 EMA Test / Shakeout (-2 to 0%)"

        if is_us:
            if format_tradingview_symbol:
                tv_sym_str = format_tradingview_symbol(
                    ticker=clean_t,
                    raw_ticker=str(t),
                    is_us=True,
                    us_tv_map=us_tv_map,
                    prefix_us=True,
                )
            else:
                tv_sym_str = (
                    us_tv_map.get(clean_t.upper())
                    or us_tv_map.get(str(t).upper())
                    or clean_t.replace('-', '.')
                )
            exch_str = tv_sym_str.split(':', 1)[0] if ':' in str(tv_sym_str) else 'US'
            tv_url_str = f"https://www.tradingview.com/chart/?symbol={tv_sym_str}"
        else:
            exch_str = 'BSE' if str(t).endswith('.BO') else 'NSE'
            base_tv = clean_t.replace('-SM', '').replace('-ST', '').replace('-BE', '')
            tv_sym_str = f"{exch_str}:{base_tv}"
            tv_url_str = f"https://in.tradingview.com/chart/?symbol={tv_sym_str}"

        records.append({
            'Rank': rank_idx,
            'Ticker': clean_t,
            'Raw_Ticker': t,
            'Exchange': exch_str,
            'TV_Symbol': tv_sym_str,
            'TradingView_URL': tv_url_str,
            'Matrix_Bar_Date': matrix_dt_str,
            'Price': round(c, 2),
            'RS_Rating': round(rs_rating_val, 1),
            'ADTV_20D': round(adtv_val, 2),
            'ADTV_Mean_20D': round(adtv_mean_val, 2),
            'ADTV_Median_20D': round(adtv_med_val, 2),
            'Industry': ind_name,
            'Industry_Grade': grd,
            'Group_Rank': rnk,
            'Is_Top_20_Group': is_top_20,
            'Industry_Badge': ind_badge,
            'Group_Role': grp_role,
            'Role_Code': role_code,
            'RS_Line_Status': rs_line_badge,
            'Is_RS_Blue_Dot': is_blue_dot,
            'Is_RS_52W_High': is_rs_52w_high,
            'Is_RS_3M_High': is_rs_3m_high,
            'Is_Stage2': is_stage2,
            'Funda_Badge': funda_badge,
            'EPS_Growth': round(eps_g, 1),
            'Sales_Growth': round(sales_g, 1),
            'ROE': round(roe_val, 1),
            'Sponsorship_Grade': sp_grade,
            'Inst_Change_QoQ': round(sp_change, 2),
            'Is_TML_Confluence': is_tml_confluence,
            'Has_Double_25': has_double_25,
            'Has_High_Sales': bool(has_high_sales or has_high_eps),
            'Has_Quality_ROE': has_quality_roe,
            'Has_Inst_Accum': has_inst_accum,
            'UDVR_50D': round(udvr_val, 2),
            'Is_Pocket_Pivot': is_pp,
            'Is_VDU': is_vdu,
            'Dist_21EMA': round(d21, 2),
            'Dist_50SMA': round(float(d50), 2) if not np.isnan(d50) else np.nan,
            'Dist_65EMA': round(float(d65), 2) if not np.isnan(d65) else np.nan,
            'Dist_50SMA_ATRx': d50_atrx,
            'Dist_52W_High': round(d52w, 2),
            'RS_Days_15D_Pct': round(r15_pct, 1),
            'RS_Days_15D_Label': f"{r15_pct:.1f}% ({r15_cnt}/15)",
            'RS_Days_20D_Pct': round(r20_pct, 1),
            'RS_Days_20D_Label': f"{r20_pct:.1f}% ({r20_cnt}/20)",
            'RS_Days_40D_Pct': round(r40_pct, 1),
            'RS_Days_40D_Label': f"{r40_pct:.1f}% ({r40_cnt}/40)",
            'Sortino_3M': round(float(s3m), 2) if s3m else 0.0,
            'Smooth_RS': round(float(sm_rs), 2) if sm_rs else 0.0,
            'Holding_21EMA_Zone': holding_21,
            'Holding_50SMA_Zone': holding_50,
            'Holding_65EMA_Zone': holding_65,
            'Bear_Pivot_Pocket': bear_pivot_pocket,
            'SMA50_Rising': sma50_rising,
            'EMA65_Rising': ema65_rising,
            'SMA200_Slope_OK': sma200_slope_ok,
            'Heavy_Dist_Break': heavy_dist_break,
            'Is_Bear_VDU': is_bear_vdu,
            'Is_Bear_Coil': is_bear_coil,
            'Setup_Trigger_State': setup_trigger_state,
            'MA_Status': ma_status,
            'Apex_Status': apex_badge,
            'Has_Beach_Ball': has_bb,
            'ADR_20D': round(adr_val, 2),
            'DCR_Pct': round(dcr_val, 0),
            'RVOL_Today': round(rvol_val, 2),
            'Range_5D_Pct': round(rng_5d_val, 2),
            'Is_Inside_Day': is_id,
            'Is_NR7': is_nr7,
            'Is_3WT': is_3wt,
            'Is_5D_Coil': is_5d_coil,
            'Is_ID_NR7': is_id_nr7,
            'Is_VCP_Tight': is_vcp_tight,
            'VCP_Action_Badge': vcp_action_badge,
            'Dist_200SMA': round(d200, 2),
            'Is_Fresh_Stage2': is_fresh_stage2,
            'Is_Climax_Ext': is_climax_ext,
            'Cap_Tier': cap_tier,
            'Market_Cap_Display': mcap_disp,
            'Cap_Stage_Badge': cap_stage_badge
        })

    res_df = pd.DataFrame(records)
    res_df.attrs['diagnostics'] = {
        'bench_stale_days': int(bench_stale_days),
        'bench_last_dt': bench_last_dt.strftime('%Y-%m-%d') if hasattr(bench_last_dt, 'strftime') else str(bench_last_dt)[:10],
        'matrix_last_dt': matrix_dt_str,
        'universe_scope': scope_clean,
        'market': 'US' if is_us else 'IN',
    }
    return res_df


def compute_leader_21ema_table(
    market='IN',
    top_n=100,
    universe_scope='750',
    min_adtv_cr=None,
    _cache_v='v11_50sma_65ema',
    benchmark_symbol=None,
):
    """
    Computes real-time 21 EMA, 50 SMA (>= -4%) & Ross Haber 65 EMA adherence ledger
    for Relative Strength leaders with automatic matrix mtime+size cache invalidation.
    """
    scope_clean = str(universe_scope).upper().strip()
    if scope_clean == 'ALL':
        matrix_path = 'historical_prices_matrix.pkl' if os.path.exists('historical_prices_matrix.pkl') else 'breadth_matrix.pkl'
    else:
        matrix_path = 'breadth_matrix.pkl' if os.path.exists('breadth_matrix.pkl') else 'historical_prices_matrix.pkl'

    if not os.path.exists(matrix_path):
        try:
            from database import load_cloud_json_cache_cached
            is_us = str(market or "IN").upper().strip() in ("US", "USA", "US_IXIC")
            bm_u = str(benchmark_symbol or "").upper()
            if is_us:
                if "GSPC" in bm_u or "SP500" in scope_clean or "S&P" in scope_clean:
                    keys = ["leader_radar_us_gspc", "leader_radar_us_ixic"]
                else:
                    keys = ["leader_radar_us_ixic", "leader_radar_us_gspc"]
            else:
                if scope_clean == "ALL" or "FULL" in scope_clean:
                    keys = ["leader_radar_in_all", "leader_radar_in_750"]
                else:
                    keys = ["leader_radar_in_750", "leader_radar_in_all"]
            for ck in keys:
                cached_df, _ = load_cloud_json_cache_cached(ck)
                if isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
                    diag = dict(cached_df.attrs.get("diagnostics") or {})
                    out = cached_df.head(int(top_n)).copy() if top_n else cached_df.copy()
                    out.attrs["diagnostics"] = diag
                    return out
        except Exception:
            pass
        return pd.DataFrame()

    try:
        from price_history_manager import get_matrix_cache_signature
        sig = get_matrix_cache_signature(matrix_path)
    except Exception:
        sig = (0, 0)

    return _compute_leader_21ema_table_cached(
        market=market,
        top_n=top_n,
        universe_scope=universe_scope,
        min_adtv_cr=min_adtv_cr,
        _cache_v=_cache_v,
        benchmark_symbol=benchmark_symbol,
        _cache_sig=sig,
    )


compute_leader_21ema_table.clear = getattr(_compute_leader_21ema_table_cached, 'clear', lambda: None)


# =============================================================================
# INSTITUTIONAL SECTOR BREADTH MATRIX (INDIA & US GICS)
# =============================================================================

def compute_sector_breadth_matrix_in(min_stocks: int = 5) -> pd.DataFrame:
    """
    Computes real-time moving average breadth (% > 50 SMA, % > 200 SMA, % > 20 EMA)
    for granular sub-industries across the Indian equity universe (750+ stocks).
    Returns a sorted DataFrame with constituent stock names and direct TradingView links.
    """
    import json
    from price_history_manager import load_cached_price_matrix, extract_market_ohlcv_panels, get_matrix_cache_signature
    matrix_path = 'historical_prices_matrix.pkl' if os.path.exists('historical_prices_matrix.pkl') else 'breadth_matrix.pkl'
    if not os.path.exists(matrix_path):
        try:
            from database import load_cloud_json_cache_cached
            bundle, _ = load_cloud_json_cache_cached("breadth_aux_bundle_in")
            if isinstance(bundle, dict) and isinstance(bundle.get("sector_breadth"), pd.DataFrame):
                return bundle["sector_breadth"].copy()
        except Exception:
            pass
        return pd.DataFrame()

    try:
        df = load_cached_price_matrix(matrix_path, cache_sig=get_matrix_cache_signature(matrix_path))
        panels = extract_market_ohlcv_panels(
            df,
            market='IN',
            include_etfs=False,
            deduplicate_dual_in=True,
            min_coverage_ratio=0.35,
            min_abs=20,
        )
        closes_in = panels['close']
    except Exception:
        return pd.DataFrame()

    ind_map = {}
    if os.path.exists('industry_map.json'):
        try:
            with open('industry_map.json', 'r', encoding='utf-8') as f:
                ind_map = json.load(f)
        except Exception:
            pass

    comp_map = {}
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ind_niftytotalmarket_list.csv')
    if os.path.exists(csv_path):
        try:
            cdf = pd.read_csv(csv_path)
            if 'Symbol' in cdf.columns and 'Company Name' in cdf.columns:
                comp_map = cdf.set_index('Symbol')['Company Name'].to_dict()
        except Exception:
            pass

    if closes_in.empty:
        return pd.DataFrame()

    curr = closes_in.iloc[-1]
    sma50 = closes_in.rolling(50, min_periods=35).mean().iloc[-1]
    sma200 = closes_in.rolling(200, min_periods=150).mean().iloc[-1]
    ema20 = closes_in.ewm(span=20, adjust=False).mean().iloc[-1]

    valid = curr.notna() & sma50.notna()
    c_valid = curr[valid]
    s50_v = sma50[valid]
    s200_v = sma200.reindex(c_valid.index)
    e20_v = ema20.reindex(c_valid.index)

    above50 = (c_valid > s50_v)
    above200 = (c_valid > s200_v).fillna(False)
    above20 = (c_valid > e20_v).fillna(False)
    pct_s50_dist = ((c_valid / s50_v) - 1) * 100

    data_rows = []
    for col in c_valid.index:
        clean_t = str(col).replace('.NS', '').replace('.BO', '')
        ind = ind_map.get(col, ind_map.get(clean_t, 'Other'))
        if ind == 'Other' or not ind:
            continue
        data_rows.append({
            'ticker': clean_t,
            'company': comp_map.get(clean_t, clean_t),
            'sector': ind,
            'above_50': bool(above50[col]),
            'above_200': bool(above200[col]),
            'above_20': bool(above20[col]),
            'pct_above_50': round(float(pct_s50_dist[col]), 1)
        })

    if not data_rows:
        return pd.DataFrame()

    df_stocks = pd.DataFrame(data_rows)
    grouped = df_stocks.groupby('sector')

    sector_records = []
    for sec_name, grp in grouped:
        total = len(grp)
        if total < min_stocks:
            continue
        a50 = int(grp['above_50'].sum())
        a200 = int(grp['above_200'].sum())
        a20 = int(grp['above_20'].sum())
        p50 = round(a50 / total * 100, 1)
        p200 = round(a200 / total * 100, 1)
        p20 = round(a20 / total * 100, 1)

        grp_sorted = grp.sort_values('pct_above_50', ascending=False)
        stock_list = grp_sorted[['ticker', 'company', 'above_50', 'above_200', 'pct_above_50']].to_dict('records')

        is_decoupled = bool(p50 >= 70.0 and p20 >= 60.0)
        if is_decoupled:
            status = '👑 DECOUPLED LEADER'
        elif p50 >= 50.0:
            status = '🟢 STAGE 2 BULL'
        elif p50 >= 40.0:
            status = '🟡 CONSOLIDATING'
        else:
            status = '🔴 LIQUIDATING'

        sector_records.append({
            'sector': sec_name,
            'total_stocks': total,
            'above_50_count': a50,
            'above_50_pct': p50,
            'above_200_count': a200,
            'above_200_pct': p200,
            'above_20_count': a20,
            'above_20_pct': p20,
            'is_decoupled': is_decoupled,
            'status': status,
            'stocks': stock_list
        })

    if not sector_records:
        return pd.DataFrame()

    return pd.DataFrame(sector_records).sort_values('above_50_pct', ascending=False).reset_index(drop=True)


def compute_sector_breadth_matrix_us() -> pd.DataFrame:
    """
    Computes real-time moving average breadth (% > 50 SMA, % > 200 SMA, % > 20 EMA)
    for the 11 official S&P 500 GICS sectors.
    Returns a sorted DataFrame with constituent stock names and direct TradingView links.
    """
    import json
    matrix_path = 'historical_prices_matrix.pkl' if os.path.exists('historical_prices_matrix.pkl') else 'breadth_matrix.pkl'
    if not os.path.exists(matrix_path):
        try:
            from database import load_cloud_json_cache_cached
            bundle, _ = load_cloud_json_cache_cached("breadth_aux_bundle_us")
            if isinstance(bundle, dict) and isinstance(bundle.get("sector_breadth"), pd.DataFrame):
                return bundle["sector_breadth"].copy()
        except Exception:
            pass
        return pd.DataFrame()

    try:
        df = pd.read_pickle(matrix_path)
        closes = df.xs('Close', level=1, axis=1) if df.columns.nlevels > 1 else df[['Close']]
    except Exception:
        return pd.DataFrame()

    gics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sp500_gics_map.json')
    if not os.path.exists(gics_path):
        return pd.DataFrame()

    try:
        with open(gics_path, 'r', encoding='utf-8') as f:
            gics_map = json.load(f)
    except Exception:
        return pd.DataFrame()

    from price_history_manager import filter_panel_active_market_dates, is_excluded_etf_or_index
    us_cols = [c for c in gics_map.keys() if c in closes.columns and not is_excluded_etf_or_index(c)]
    if not us_cols:
        return pd.DataFrame()
    closes_us = filter_panel_active_market_dates(closes[us_cols], min_coverage_ratio=0.20, min_abs=20)
    if closes_us.empty:
        return pd.DataFrame()

    curr = closes_us.iloc[-1]
    sma50 = closes_us.rolling(50, min_periods=35).mean().iloc[-1]
    sma200 = closes_us.rolling(200, min_periods=150).mean().iloc[-1]
    ema20 = closes_us.ewm(span=20, adjust=False).mean().iloc[-1]

    data_rows = []
    for sym in us_cols:
        info = gics_map.get(sym, {})
        if pd.notna(curr.get(sym)) and pd.notna(sma50.get(sym)):
            c_val = curr[sym]
            s50_val = sma50[sym]
            s200_val = sma200.get(sym, np.nan)
            e20_val = ema20.get(sym, np.nan)

            above_50 = bool(c_val > s50_val)
            above_200 = bool(c_val > s200_val) if pd.notna(s200_val) else False
            above_20 = bool(c_val > e20_val) if pd.notna(e20_val) else False
            pct_50 = round(float((c_val / s50_val - 1) * 100), 1)

            data_rows.append({
                'ticker': sym,
                'company': info.get('company', sym),
                'sector': info.get('sector', 'Other'),
                'sub_industry': info.get('sub_industry', ''),
                'above_50': above_50,
                'above_200': above_200,
                'above_20': above_20,
                'pct_above_50': pct_50
            })

    if not data_rows:
        return pd.DataFrame()

    df_us = pd.DataFrame(data_rows)
    grouped = df_us.groupby('sector')

    records = []
    for sec_name, grp in grouped:
        tot = len(grp)
        a50 = int(grp['above_50'].sum())
        a200 = int(grp['above_200'].sum())
        a20 = int(grp['above_20'].sum())
        p50 = round(a50 / tot * 100, 1)
        p200 = round(a200 / tot * 100, 1)
        p20 = round(a20 / tot * 100, 1)

        grp_sorted = grp.sort_values('pct_above_50', ascending=False)
        stocks = grp_sorted[['ticker', 'company', 'above_50', 'above_200', 'pct_above_50']].to_dict('records')

        is_dec = bool(p50 >= 70.0 and p20 >= 60.0)
        status = '👑 DECOUPLED LEADER' if is_dec else ('🟢 STAGE 2 BULL' if p50 >= 50.0 else ('🟡 CONSOLIDATING' if p50 >= 40.0 else '🔴 LIQUIDATING'))
        records.append({
            'sector': sec_name,
            'total_stocks': tot,
            'above_50_count': a50,
            'above_50_pct': p50,
            'above_200_count': a200,
            'above_200_pct': p200,
            'above_20_count': a20,
            'above_20_pct': p20,
            'is_decoupled': is_dec,
            'status': status,
            'stocks': stocks
        })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values('above_50_pct', ascending=False).reset_index(drop=True)


# =============================================================================
# COMPREHENSIVE 7-PILLAR CHECKLIST WITH INFLECTION TIMING GUARDRAILS
# =============================================================================

def evaluate_comprehensive_breadth_checklist(
    df_breadth,
    df_top_rs=None,
    vol_dist=None,
    n500_telem=None,
    zweig_tape=None,
    benchmark_name: str = "Nifty 500",
    breadth_alpha: dict = None,
    sector_breadth_df: pd.DataFrame = None
) -> dict:
    """
    Evaluates the 7-Pillar Market Breadth Checklist derived from the 2023-2024 Indian Bull Market Backtest
    with institutional inflection timing guardrails:
      1. % Stocks > 50-Day SMA (Tactical Momentum & Thrust)
      2. % Stocks > 200-Day SMA (Secular Stage 2 Backbone)
      3. Net New 52-Week Highs (Expansion Truth-Teller & Canary Alert)
      4. Top 100 RS Leaders > 21-Day EMA (Leadership Velocity)
      5. Exchange Volume Breadth & Smart Money Ratio (Money Flow)
      6. Dr. Martin Zweig Breadth Thrust & 9:1 Up-Volume (Stampede Edge)
      7. Distribution Days & Acute Cluster Alert (Institutional Supply)

    Inflection Timing Guardrails:
      - V-Bottom Thrust Fast-Track Override: Waives 200 SMA lock on active Zweig / 9:1 thrusts.
      - Bear Market Relief Rally Guard: Caps exposure at 20-30% on unconfirmed 50 SMA bounces under Stage 4 ceilings.
      - Mega-Cap Divergence Detection: Flags narrow rallies where Cap-Weight masks Equal-Weight liquidation.
      - Two-Tiered Concentration Exception: Unlocks 15-20% allocation into decoupled sector leaders during macro yellow/red.
    """
    if df_breadth is None or df_breadth.empty:
        return {
            'regime': 'YELLOW',
            'regime_sub_type': 'STANDARD_YELLOW',
            'regime_title': '🟡 Neutral / Insufficient Breadth Telemetry',
            'regime_desc': 'Broad market breadth telemetry is loading or unavailable. Maintain defensive posture with selective exposure.',
            'badge_color': '#f59e0b',
            'badge_bg': 'rgba(245, 158, 11, 0.15)',
            'target_exposure': '25% – 50%',
            'cash_target': '50% – 75%',
            'portfolio_sizing': 'Hold 2 to 3 selective pilots',
            'action_directives': {
                'new_buys': '⛔ Caution: Incomplete breadth telemetry. Avoid aggressive exposure.',
                'trailing_stops': '🛡️ Strict 21-day EMA trailing stops.',
                'cash_strategy': 'Hold conservative cash balance.',
                'risk_posture': 'Defensive observation.'
            },
            'checklist_items': [],
            'summary_stats': {'pass_count': 0, 'caution_count': 0, 'fail_count': 0}
        }

    latest_b = df_breadth.iloc[-1]
    
    # -------------------------------------------------------------
    # 1. PILLAR 1: % STOCKS > 50-DAY SMA (TACTICAL THRUST)
    # -------------------------------------------------------------
    s50 = float(latest_b.get('above_50_pct', 0.0))
    thrust_active = False
    if len(df_breadth) >= 10:
        tail_15 = df_breadth['above_50_pct'].tail(15)
        min_prev = float(tail_15.iloc[:-1].min()) if len(tail_15) > 1 else 100.0
        if min_prev < 35.0 and s50 >= 50.0:
            thrust_active = True
            
    if s50 >= 50.0 or thrust_active:
        p1_status = 'PASS'
        p1_color = '#10b981'
        p1_icon = '🟢'
        p1_note = '⚡ BREADTH THRUST ACTIVE: Explosive institutional momentum' if thrust_active else 'Bullish: Broad market holding intermediate-term institutional support'
    elif s50 >= 40.0:
        p1_status = 'CAUTION'
        p1_color = '#f59e0b'
        p1_icon = '🟡'
        p1_note = 'Warning: Intermediate breadth under pressure; >50% of stocks slipping below 50 SMA'
    else:
        p1_status = 'FAIL'
        p1_color = '#ef4444'
        p1_icon = '🔴'
        p1_note = 'Severe Breakdown: Deep technical liquidation across >60% of stocks'

    # -------------------------------------------------------------
    # 2. PILLAR 2: % STOCKS > 200-DAY SMA (SECULAR BACKBONE)
    # -------------------------------------------------------------
    s200 = float(latest_b.get('above_200_pct', 0.0))
    if s200 == 0.0 and len(df_breadth) < 200:
        p2_status = 'PASS' if s50 >= 50.0 else 'CAUTION'
        p2_color = '#10b981' if p2_status == 'PASS' else '#f59e0b'
        p2_icon = '🟢' if p2_status == 'PASS' else '🟡'
        p2_note = 'Warmup Phase (Using 50 SMA proxy): Macro long-term trend initializing'
        s200_display = f"{s50:.1f}% (Proxy)"
    elif s200 >= 60.0:
        p2_status = 'PASS'
        p2_color = '#10b981'
        p2_icon = '🟢'
        p2_note = 'Secular Bull: Over 60% of universe in Stan Weinstein Stage 2 long-term advance'
        s200_display = f"{s200:.1f}%"
    elif s200 >= 52.0:
        p2_status = 'CAUTION'
        p2_color = '#f59e0b'
        p2_icon = '🟡'
        p2_note = 'Secular Transition: 200 SMA health weakening toward structural bear boundary'
        s200_display = f"{s200:.1f}%"
    else:
        p2_status = 'FAIL'
        p2_color = '#ef4444'
        p2_icon = '🔴'
        p2_note = 'Secular Bear Breakdown: Structural Stage 4 market environment'
        s200_display = f"{s200:.1f}%"

    # -------------------------------------------------------------
    # 3. PILLAR 3: NET NEW 52-WEEK HIGHS (EXPANSION TRUTH-TELLER)
    # -------------------------------------------------------------
    nh = int(latest_b.get('new_highs_count', 0))
    nl = int(latest_b.get('new_lows_count', 0))
    nnh = int(latest_b.get('net_new_highs', nh - nl))
    
    canary_warning = (nl >= 15) or (nnh < 0 and s50 >= 45.0)
    
    if nnh > 0 and nh >= 25 and nl <= 8:
        p3_status = 'PASS'
        p3_color = '#10b981'
        p3_icon = '🟢'
        p3_note = f'Healthy Expansion: Highs ({nh}) dominate Lows ({nl}); leadership actively expanding'
    elif nnh >= 0 and nl <= 15:
        p3_status = 'CAUTION'
        p3_color = '#f59e0b'
        p3_icon = '🟡'
        p3_note = f'Subdued Expansion: Net +{nnh} ({nh} Highs, {nl} Lows); momentum pace cooling'
    else:
        p3_status = 'FAIL'
        p3_color = '#ef4444'
        p3_icon = '🔴'
        p3_note = f'🚨 DISTRIBUTION CANARY: Net {nnh:+d} ({nh} Highs vs {nl} Lows); institutional liquidation expanding' if canary_warning else f'Liquidation: Net {nnh:+d} New Highs'

    # -------------------------------------------------------------
    # 4. PILLAR 4: TOP 100 RS LEADERS > 21-DAY EMA (VELOCITY)
    # -------------------------------------------------------------
    top_21 = 0.0
    top_50 = 0.0
    if df_top_rs is not None and not df_top_rs.empty:
        top_21 = float(df_top_rs['top100_above_21_ema'].iloc[-1])
        top_50 = float(df_top_rs['top100_above_50_sma'].iloc[-1])
    else:
        top_21 = float(latest_b.get('above_20_pct', s50))
        top_50 = s50

    if top_21 >= 70.0:
        p4_status = 'PASS'
        p4_color = '#10b981'
        p4_icon = '🟢'
        p4_note = f'Super-Leader Surge: {top_21:.1f}% of top RS leaders holding fast 21 EMA'
    elif top_21 >= 50.0:
        p4_status = 'CAUTION'
        p4_color = '#f59e0b'
        p4_icon = '🟡'
        p4_note = f'Leader Digestion: {top_21:.1f}% above 21 EMA; leaders consolidating gains'
    else:
        p4_status = 'FAIL'
        p4_color = '#ef4444'
        p4_icon = '🔴'
        p4_note = f'Leader Breakdown: Only {top_21:.1f}% above 21 EMA; key momentum drivers failing'

    # -------------------------------------------------------------
    # 5. PILLAR 5: EXCHANGE VOLUME BREADTH & SMART MONEY RATIO
    # -------------------------------------------------------------
    ud_ratio = 1.0
    sm_ratio = 1.0
    if vol_dist:
        ud_ratio = float(vol_dist.get('up_down_volume_ratio', 1.0))
        sm_ratio = float(vol_dist.get('smart_money_ratio', 1.0))
    elif 'up_down_volume_ratio' in latest_b:
        ud_ratio = float(latest_b.get('up_down_volume_ratio', 1.0))

    if ud_ratio >= 1.2 or sm_ratio >= 1.15:
        p5_status = 'PASS'
        p5_color = '#10b981'
        p5_icon = '🟢'
        p5_note = f'Institutional Inflow: Up/Down Vol {ud_ratio:.2f}x | Smart Money {sm_ratio:.2f}x'
    elif ud_ratio >= 0.85 and sm_ratio >= 0.85:
        p5_status = 'CAUTION'
        p5_color = '#f59e0b'
        p5_icon = '🟡'
        p5_note = f'Balanced Flow: Up/Down Vol {ud_ratio:.2f}x | Smart Money {sm_ratio:.2f}x'
    else:
        p5_status = 'FAIL'
        p5_color = '#ef4444'
        p5_icon = '🔴'
        p5_note = f'Heavy Outflow: Up/Down Vol {ud_ratio:.2f}x | Smart Money {sm_ratio:.2f}x'

    # -------------------------------------------------------------
    # 6. PILLAR 6: ZWEIG BREADTH THRUST (ZBT) & 9:1 VOLUME ENGINE
    # -------------------------------------------------------------
    zbt_val = 0.50
    zbt_act = False
    d91_act = False
    if zweig_tape:
        zbt_val = float(zweig_tape.get('zbt_10_ema', 0.50))
        zbt_act = bool(zweig_tape.get('zbt_active', False))
        d91_act = bool(zweig_tape.get('double_9_to_1_active', False))
    elif 'zbt_10_ema' in latest_b:
        zbt_val = float(latest_b.get('zbt_10_ema', 0.50))
        zbt_act = bool(latest_b.get('zbt_active', 0) == 1)
        d91_act = bool(latest_b.get('double_9_to_1_active', 0) == 1)

    if zbt_act or d91_act:
        p6_status = 'PASS'
        p6_color = '#10b981'
        p6_icon = '🟢'
        p6_note = '🔥 ZBT / DOUBLE 9:1 ACTIVE: Maximum institutional buying stampede edge'
    elif zbt_val >= 0.55:
        p6_status = 'PASS'
        p6_color = '#10b981'
        p6_icon = '🟢'
        p6_note = f'Bullish Accumulation: Advance Ratio 10-EMA at {zbt_val:.3f}'
    elif zbt_val >= 0.40:
        p6_status = 'CAUTION'
        p6_color = '#f59e0b'
        p6_icon = '🟡'
        p6_note = f'Neutral Tape: ZBT 10-EMA at {zbt_val:.3f} (No active thrust)'
    else:
        p6_status = 'FAIL'
        p6_color = '#ef4444'
        p6_icon = '🔴'
        p6_note = f'Oversold / Washout: ZBT 10-EMA at {zbt_val:.3f} (<0.400)'

    # -------------------------------------------------------------
    # 7. PILLAR 7: DISTRIBUTION DAYS & ACUTE CLUSTERS
    # -------------------------------------------------------------
    dd_count = 0
    cluster_detected = False
    cluster_count_7d = 0
    if n500_telem:
        dd_count = int(n500_telem.get('dd_count', 0))
        cluster_detected = bool(n500_telem.get('cluster_detected', False))
        cluster_count_7d = int(n500_telem.get('cluster_count_7d', 0))

    if cluster_detected:
        p7_status = 'FAIL'
        p7_color = '#ef4444'
        p7_icon = '🔴'
        p7_note = f'🚨 ACUTE CLUSTER: {cluster_count_7d} Distribution Days in 7 sessions (Ross Haber Warning)'
    elif dd_count >= 5:
        p7_status = 'CAUTION'
        p7_color = '#f59e0b'
        p7_icon = '🟡'
        p7_note = f'Elevated Supply: {dd_count} Distribution Days in last 25 sessions (Uptrend Under Pressure)'
    else:
        p7_status = 'PASS'
        p7_color = '#10b981'
        p7_icon = '🟢'
        p7_note = f'Low Institutional Selling: Only {dd_count} Distribution Days in last 25 sessions'

    # Build checklist items list
    checklist_items = [
        {
            'id': 'p1_50sma',
            'name': '% Stocks > 50-Day SMA',
            'icon': '🚀',
            'category': 'Tactical Thrust & Entry Gate',
            'reading_str': f"{s50:.1f}%",
            'threshold_str': '≥ 50.0% (Thrust <35%→>50%)',
            'status': p1_status,
            'status_color': p1_color,
            'status_icon': p1_icon,
            'guidance': p1_note
        },
        {
            'id': 'p2_200sma',
            'name': '% Stocks > 200-Day SMA',
            'icon': '🏛️',
            'category': 'Secular Stage 2 Backbone',
            'reading_str': s200_display,
            'threshold_str': '≥ 60.0% (Stage 2 Floor)',
            'status': p2_status,
            'status_color': p2_color,
            'status_icon': p2_icon,
            'guidance': p2_note
        },
        {
            'id': 'p3_new_highs',
            'name': 'Net New 52-Week Highs',
            'icon': '🌟',
            'category': 'Leadership Expansion Engine',
            'reading_str': f"{nnh:+d} ({nh}H / {nl}L)",
            'threshold_str': '> 0 Net (H>25, L<10)',
            'status': p3_status,
            'status_color': p3_color,
            'status_icon': p3_icon,
            'guidance': p3_note
        },
        {
            'id': 'p4_top_rs',
            'name': 'Top 100 RS Leaders > 21 EMA',
            'icon': '👑',
            'category': 'Capital Velocity & Momentum',
            'reading_str': f"{top_21:.1f}%",
            'threshold_str': '≥ 70.0% (Bull Runway)',
            'status': p4_status,
            'status_color': p4_color,
            'status_icon': p4_icon,
            'guidance': p4_note
        },
        {
            'id': 'p5_volume',
            'name': 'Exchange Volume Breadth',
            'icon': '🌊',
            'category': 'Institutional Smart Money Flow',
            'reading_str': f"{ud_ratio:.2f}x Vol | {sm_ratio:.2f}x Flow",
            'threshold_str': '≥ 1.20x Vol | ≥ 1.10x Flow',
            'status': p5_status,
            'status_color': p5_color,
            'status_icon': p5_icon,
            'guidance': p5_note
        },
        {
            'id': 'p6_zweig',
            'name': 'Zweig Breadth Thrust & 9:1 Up-Vol',
            'icon': '⚡',
            'category': 'Rare Buying Stampede Edge',
            'reading_str': 'ACTIVE 🔥' if (zbt_act or d91_act) else f"{zbt_val:.3f} (10-EMA)",
            'threshold_str': '≥ 0.615 (ZBT Active)',
            'status': p6_status,
            'status_color': p6_color,
            'status_icon': p6_icon,
            'guidance': p6_note
        },
        {
            'id': 'p7_distribution',
            'name': f'{benchmark_name} Distribution Pressure',
            'icon': '🛑',
            'category': 'Institutional Selling Pressure',
            'reading_str': f"{dd_count} DDs" + (" (Cluster!)" if cluster_detected else ""),
            'threshold_str': '< 5 DDs & No Clusters',
            'status': p7_status,
            'status_color': p7_color,
            'status_icon': p7_icon,
            'guidance': p7_note
        }
    ]

    pass_cnt = sum(1 for it in checklist_items if it['status'] == 'PASS')
    caution_cnt = sum(1 for it in checklist_items if it['status'] == 'CAUTION')
    fail_cnt = sum(1 for it in checklist_items if it['status'] == 'FAIL')

    # -------------------------------------------------------------
    # INFLECTION TIMING GUARDRAILS (V-BOTTOM, BEAR TRAP, DIVERGENCE)
    # -------------------------------------------------------------
    # 1. V-Bottom Thrust Fast-Track Override:
    # A confirmed buying stampede (ZBT active or Double 9:1 or fast 50-SMA thrust)
    # with intermediate momentum holding (s50 >= 45%) and NO acute distribution clusters.
    thrust_fast_track = bool((thrust_active or zbt_act or d91_act) and (not cluster_detected) and (s50 >= 45.0))

    # 2. Bear Market Relief Rally Guard:
    # Tactical 50-SMA bounce under a depressed Stage 4 secular ceiling (<42%)
    # without 52W Highs expansion (nh < 15) and NO Zweig stampede.
    is_bear_rally = bool(
        s50 >= 45.0 and
        (s200 > 0 and s200 < 42.0) and
        (nh < 15) and
        (not thrust_fast_track)
    )

    # 3. Mega-Cap Divergence Detection (The "Index Lie"):
    mega_cap_divergence = False
    div_reason = ""
    if breadth_alpha:
        ew_cw_spread = float(breadth_alpha.get('spread_n50_1m', breadth_alpha.get('spread_sp_1m', breadth_alpha.get('spread_sp500_1m', 0.0))))
        pct_beat = float(breadth_alpha.get('pct_beat_n500_1m', breadth_alpha.get('pct_beat_sp_1m', breadth_alpha.get('pct_beat_sp500_1m', 50.0))))
        if (ew_cw_spread < -3.0 or pct_beat < 38.0) and (canary_warning or s50 >= 50.0):
            mega_cap_divergence = True
            div_reason = f"Narrow rally: Equal-weight lags Cap-weight by {ew_cw_spread:+.1f}% | Only {pct_beat:.0f}% beating benchmark"

    # 4. Decoupled Leading Sectors (Two-Tiered Markets):
    decoupled_sectors = []
    if sector_breadth_df is not None and not sector_breadth_df.empty and 'is_decoupled' in sector_breadth_df.columns:
        decoupled_sectors = sector_breadth_df[sector_breadth_df['is_decoupled']]['sector'].tolist()

    # -------------------------------------------------------------
    # REGIME SYNTHESIS ENGINE
    # -------------------------------------------------------------
    # RED Trigger:
    if thrust_fast_track:
        # Thrust Active: waives s200 lock; only acute breakdown triggers Red
        is_red = bool((s50 < 35.0) or (fail_cnt >= 5) or cluster_detected)
    elif is_bear_rally:
        # Bear Market Relief Rally: intermediate bounce under secular ceiling
        # Capped at defensive Yellow (20-30% pilot exposure)
        is_red = False
    else:
        is_red = bool(
            (s50 < 40.0 and (nnh < 0 or (s200 > 0 and s200 < 58.0) or cluster_detected)) or
            (s50 < 30.0) or
            (s200 > 0 and s200 < 50.0) or
            (fail_cnt >= 4)
        )

    # GREEN Trigger:
    is_green = bool(
        (not is_red) and
        (not is_bear_rally) and
        (not mega_cap_divergence) and
        (s50 >= 50.0 or thrust_fast_track) and
        (nnh >= 0) and
        ((s200 >= 58.0 or s200 == 0.0) or thrust_fast_track) and
        (not cluster_detected) and
        (pass_cnt >= 4 and fail_cnt <= 1)
    )

    if is_red:
        regime = 'RED'
        regime_sub_type = 'LOCKDOWN'
        regime_title = '🔴 RED LIGHT: DEFENSIVE LOCKDOWN (CAPITAL PRESERVATION)'
        regime_desc = 'Market breadth has broken down across multiple timeframes. Distribution volume dominates and leadership is failing. Move to heavy cash to preserve multi-month profits.'
        badge_color = '#ef4444'
        badge_bg = 'rgba(239, 68, 68, 0.15)'
        target_exposure = '0% – 20%'
        cash_target = '80% – 100%'
        portfolio_sizing = '0 to 1 defensive hedge or 100% Cash / Liquid BeES (Preserve dry powder)'
        action_directives = {
            'new_buys': '🚫 TOTAL BUYING HALT: Zero new purchases permitted. Breakouts in Red regimes fail >85% of the time (Bear DD: -37.5% without halt vs -16.8% with lockdown).',
            'trailing_stops': '🚨 FAST CAPITAL PRESERVATION: Exit immediately on 1-day 21 EMA or 50 SMA break (no cushions in Red). Tighten surviving stops to breakeven.',
            'cash_strategy': '🏦 CASH LOCKDOWN: Park 80%–100% of capital in Liquid BeES, Overnight Cash, or Treasury Bills.',
            'risk_posture': 'Maximum capital defense. Cash is an active, alpha-preserving position.'
        }
    elif is_bear_rally:
        regime = 'YELLOW'
        regime_sub_type = 'BEAR_RALLY'
        regime_title = '⚠️ YELLOW LIGHT: BEAR MARKET RELIEF RALLY (COUNTER-TREND TRAP)'
        regime_desc = f'Tactical bounce (% > 50 SMA is {s50:.1f}%) trapped beneath a Stage 4 secular ceiling (% > 200 SMA is {s200:.1f}% with only {nh} New Highs). High breakout failure rate.'
        badge_color = '#f59e0b'
        badge_bg = 'rgba(245, 158, 11, 0.15)'
        target_exposure = '20% – 30%'
        cash_target = '70% – 80%'
        portfolio_sizing = 'Max 1 to 2 defensive pilot positions only (Strict 5%-7% stops)'
        action_directives = {
            'new_buys': '🚫 STRICT DEFENSIVE PILOTS ONLY: Counter-trend rally beneath declining 200 SMA ceiling. Breakouts fail >75% of the time. Zero margin.',
            'trailing_stops': '🛡️ FAST PROFIT HARVESTING: Sell into overhead 21 EMA / 50 SMA resistance. Trail stops strictly at 5%–7% loss max.',
            'cash_strategy': '🏦 HEAVY CASH PRESERVATION: Keep 70%–80% dry powder for true secular expansion.',
            'risk_posture': 'Defensive skepticism. Treat all rallies as guilty until proven innocent.'
        }
    elif is_green:
        regime = 'GREEN'
        if thrust_fast_track and (s200 > 0 and s200 < 58.0):
            regime_sub_type = 'THRUST_BLASTOFF'
            regime_title = '🚀 GREEN LIGHT: BREADTH THRUST BLAST-OFF (EARLY STAGE BULL)'
            regime_desc = f'Rare buying stampede confirmed (ZBT/Thrust active, 50 SMA at {s50:.1f}%). 200 SMA requirement waived as secular accumulation initiates.'
            badge_color = '#10b981'
            badge_bg = 'rgba(16, 185, 129, 0.15)'
            target_exposure = '60% – 80%'
            cash_target = '20% – 40%'
            portfolio_sizing = 'Aggressive pilot entry into emerging Stage 1/2 Smooth RS leaders'
            action_directives = {
                'new_buys': '🚀 EARLY STAGE AGGRESSION: Aggressively buy top Smooth RS leaders breaking out of Stage 1/2 bases. Waive 200 SMA requirement because institutional accumulation has initiated.',
                'trailing_stops': '🛡️ FAST RUNNER TRAIL: Trail tightly on 21 EMA or Gap Day Low. Do not allow winning pilot trades to become losses.',
                'cash_strategy': '⚡ RAPID CAPITAL DEPLOYMENT: Deploy capital into 4-6 leading stocks as volume confirms follow-through.',
                'risk_posture': 'Offensive opportunism. Ride the initial thrust blast-off.'
            }
        else:
            regime_sub_type = 'CONFIRMED_BULL'
            regime_title = '🟢 GREEN LIGHT: CONFIRMED BULL MARKET (FULL AGGRESSION)'
            regime_desc = 'Market breadth confirms an active institutional accumulation cycle. Overwhelming participation across 50 SMA, Net New Highs, and top RS leaders warrants maximum portfolio exposure.'
            badge_color = '#10b981'
            badge_bg = 'rgba(16, 185, 129, 0.15)'
            target_exposure = '80% – 100%'
            cash_target = '0% – 10%'
            portfolio_sizing = 'ATR Risk-Parity ("Backing Into Size": 1.15% Equity Risk ÷ 2×ATR%) across K=6 to 8 stocks (Max 1 stock / industry)'
            action_directives = {
                'new_buys': '🟢 FULL AGGRESSION: Aggressively buy top Smooth RS (Composite RS / 20D ATR%) leaders breaking out of Stage 2 bases above 50-day SMA. Enforce Max 1 stock per sub-industry.',
                'trailing_stops': '🛡️ 21-DAY EMA TRAIL (OR 23 EMA CUSHION): Trail on 2 consecutive closes below 21/23 EMA. *True 8-Week Hold*: If stock advances +20% in <=15 days, trail strictly along 50 SMA floor for 40 days.',
                'cash_strategy': '⚡ MAXIMUM CAPITAL VELOCITY: Immediately redeploy 100% of realized proceeds into the next emerging Smooth RS leader.',
                'risk_posture': 'Offensive compounding. Stay fully invested in highest relative strength market leaders.'
            }
    else:
        regime = 'YELLOW'
        if mega_cap_divergence:
            regime_sub_type = 'MEGA_CAP_DIVERGENCE'
            regime_title = '🚨 YELLOW LIGHT: MEGA-CAP DIVERGENCE (NARROW INDEX LIE)'
            regime_desc = f'Headline benchmark is propped up by mega-caps while broader universe distributes ({div_reason}). High risk of broader correction.'
            badge_color = '#f59e0b'
            badge_bg = 'rgba(245, 158, 11, 0.15)'
            target_exposure = '35% – 50%'
            cash_target = '50% – 65%'
            portfolio_sizing = 'Hold 2 to 3 high-quality leaders only (Freeze new small/mid cap purchases)'
            action_directives = {
                'new_buys': '⛔ FREEZE SMALL/MID BUYS: Halt new purchases in high-beta small/mid caps. Mega-caps masking distribution.',
                'trailing_stops': '🛡️ TIGHTEN TO BREAKEVEN: Move stops to breakeven on existing winners; sell on 21 EMA close.',
                'cash_strategy': '💰 EXPAND CASH BUFFER: Allow cash to grow to 50%–65% as trailing stops trigger.',
                'risk_posture': 'Heightened caution. Do not trust the headline index.'
            }
        else:
            regime_sub_type = 'STANDARD_YELLOW'
            regime_title = '🟡 YELLOW LIGHT: TACTICAL DEFENSE (FREEZE NEW BUYS)'
            regime_desc = 'Market breadth is diverging or entering an intermediate pullback. Leading stocks are consolidating and breakout failure rates are elevated. Freeze new buys and let cash build naturally.'
            badge_color = '#f59e0b'
            badge_bg = 'rgba(245, 158, 11, 0.15)'
            target_exposure = '50% – 65%'
            cash_target = '35% – 50%'
            portfolio_sizing = 'Hold 3 to 4 best-performing leaders only (De-leverage using ATR Risk-Parity, cap exposure at 50%–65%)'
            action_directives = {
                'new_buys': '⛔ FREEZE NEW BUYS: Halt all new breakout buying. Breakouts during Yellow regimes have a 70%+ failure rate.',
                'trailing_stops': '🛡️ STRICT 21 EMA ENFORCEMENT: Honor existing trailing stops. If a stock closes 2 days below 21 EMA, sell without hesitation (no 23 EMA cushion in Yellow).',
                'cash_strategy': '💰 NATURAL CASH ACCUMULATION: When a trailing stop triggers, DO NOT replace the position. Let cash buffer expand to 35%–50%.',
                'risk_posture': 'Tactical defense. Protect open bull market profits; avoid churn.'
            }

    # Two-Tiered Concentration Exception Directive
    if decoupled_sectors and regime in ['YELLOW', 'RED']:
        top_sec_str = ", ".join(decoupled_sectors[:3])
        action_directives['two_tiered_exception'] = (
            f"🎯 TWO-TIERED CONCENTRATION EXCEPTION: Macro tape is in {regime}, but {top_sec_str} "
            f"have decoupled into Stage 2 advances (≥70% > 50 SMA). Authorized to allocate 15%–20% "
            f"capital strictly to confirmed leaders within these groups using 21 EMA trailing stops."
        )

    return {
        'regime': regime,
        'regime_sub_type': regime_sub_type,
        'regime_title': regime_title,
        'regime_desc': regime_desc,
        'badge_color': badge_color,
        'badge_bg': badge_bg,
        'target_exposure': target_exposure,
        'cash_target': cash_target,
        'portfolio_sizing': portfolio_sizing,
        'action_directives': action_directives,
        'checklist_items': checklist_items,
        'summary_stats': {
            'pass_count': pass_cnt,
            'caution_count': caution_cnt,
            'fail_count': fail_cnt,
            'total_items': len(checklist_items),
            's50_current': s50,
            's200_current': s200,
            'nnh_current': nnh,
            'nh_current': nh,
            'nl_current': nl,
            'thrust_active': thrust_active,
            'thrust_fast_track': thrust_fast_track,
            'is_bear_rally': is_bear_rally,
            'mega_cap_divergence': mega_cap_divergence,
            'decoupled_sectors': decoupled_sectors,
            'canary_warning': canary_warning
        }
    }


def compute_rolling_ad_divergence(
    df_idx_close: pd.Series,
    df_ad_line: pd.Series,
    lookback: int = 20
) -> dict:
    """
    Delta 5: Cumulative Advance/Decline (A/D) Breadth Divergence Early Warning.
    Derived from William O'Neil, David Ryan, and Michael Steinhardt.
    
    Zero-lag structural divergence check:
    - Checks if Benchmark Index (Nifty 500 / S&P 500) trades within 1.5% of its 20-day high.
    - Checks if Cumulative Net A/D Line is lagging its 20-day rolling high by >= 1.0 rolling std.
    - Counts consecutive active divergence sessions (X sessions).
    """
    default_res = {
        'is_diverging': False,
        'sessions_count': 0,
        'severity': 'NONE',
        'status_badge': '🟢 CONFIRMED ADVANCE',
        'status_color': '#10b981',
        'status_bg': 'rgba(16, 185, 129, 0.15)',
        'headline': '🟢 Cumulative Advance/Decline Confirmed: Broad Inflows Match Index Progress',
        'directive': 'No structural stealth decay detected. Advance/Decline participation confirms benchmark price highs.',
        'price_prox_pct': 100.0,
        'ad_deficit': 0,
        'div_mask': pd.Series(dtype=bool)
    }

    if df_idx_close is None or df_ad_line is None or len(df_idx_close) < lookback + 5 or len(df_ad_line) < lookback + 5:
        return default_res

    try:
        s_idx = df_idx_close.copy()
        s_ad = df_ad_line.copy()

        s_idx.index = pd.to_datetime(s_idx.index)
        if hasattr(s_idx.index, 'tz') and s_idx.index.tz is not None:
            s_idx.index = s_idx.index.tz_localize(None)

        s_ad.index = pd.to_datetime(s_ad.index)
        if hasattr(s_ad.index, 'tz') and s_ad.index.tz is not None:
            s_ad.index = s_ad.index.tz_localize(None)

        aligned = pd.concat([s_idx.rename('price'), s_ad.rename('ad_line')], axis=1, join='inner').dropna()
        if len(aligned) < lookback + 5:
            return default_res

        prices = aligned['price']
        ad_line = aligned['ad_line']

        # 20-day rolling highs and volatility
        price_20h = prices.rolling(window=lookback, min_periods=lookback // 2).max()
        ad_20h = ad_line.rolling(window=lookback, min_periods=lookback // 2).max()
        ad_std = ad_line.rolling(window=lookback, min_periods=lookback // 2).std().replace(0, 1)

        # Price near rolling 20-day peak (within 1.5% of high)
        price_near_high = (prices >= price_20h * 0.985)
        # A/D lagging its 20-day high by >= 1.0 standard deviation
        ad_lagging = (ad_line <= ad_20h - (1.0 * ad_std))

        div_condition = price_near_high & ad_lagging

        # Count consecutive active divergence sessions up to latest date
        consecutive_sessions = 0
        for i in range(len(div_condition) - 1, -1, -1):
            if div_condition.iloc[i]:
                consecutive_sessions += 1
            else:
                break

        latest_price = float(prices.iloc[-1])
        latest_high = float(price_20h.iloc[-1])
        price_prox_pct = round((latest_price / latest_high) * 100.0, 1) if latest_high > 0 else 100.0
        latest_ad = float(ad_line.iloc[-1])
        latest_ad_high = float(ad_20h.iloc[-1])
        ad_deficit = int(latest_ad_high - latest_ad)

        is_diverging = (consecutive_sessions >= 3)

        if consecutive_sessions >= 8:
            severity = 'CRITICAL'
            status_badge = f'🔴 STEALTH DECAY ({consecutive_sessions}d Divergence)'
            status_color = '#ef4444'
            status_bg = 'rgba(239, 68, 68, 0.2)'
        elif consecutive_sessions >= 3:
            severity = 'ALERT'
            status_badge = f'🟠 ACTIVE DIVERGENCE ({consecutive_sessions}d Divergence)'
            status_color = '#f97316'
            status_bg = 'rgba(249, 115, 22, 0.2)'
        elif consecutive_sessions >= 1:
            severity = 'EMERGING'
            status_badge = f'🟡 EMERGING DIVERGENCE ({consecutive_sessions}d)'
            status_color = '#f59e0b'
            status_bg = 'rgba(245, 158, 11, 0.15)'
        else:
            severity = 'NONE'
            status_badge = '🟢 CONFIRMED ADVANCE'
            status_color = '#10b981'
            status_bg = 'rgba(16, 185, 129, 0.15)'

        headline = (
            f"⚠️ STEALTH BREADTH DECAY: Mega-Cap Masking (A/D Line Diverging for {consecutive_sessions} Sessions)"
            if is_diverging else
            (f"🟡 Emerging Breadth Divergence (A/D Line Diverging for {consecutive_sessions} Sessions)"
             if consecutive_sessions > 0 else
             "🟢 Cumulative Advance/Decline Confirmed: Broad Inflows Match Index Progress")
        )
        directive = (
            "Classic institutional distribution: Headline index is being held up by narrow mega-caps while broader liquid universe is distributing. Stop buying new breakouts; raise trailing stops to lock in gains."
            if is_diverging else
            "No structural stealth decay detected. Advance/Decline participation confirms benchmark price highs."
        )

        return {
            'is_diverging': is_diverging,
            'sessions_count': consecutive_sessions,
            'severity': severity,
            'status_badge': status_badge,
            'status_color': status_color,
            'status_bg': status_bg,
            'headline': headline,
            'directive': directive,
            'price_prox_pct': price_prox_pct,
            'ad_deficit': ad_deficit,
            'latest_date': aligned.index[-1].strftime('%Y-%m-%d') if not aligned.empty else None,
            'div_mask': div_condition
        }
    except Exception as e:
        return default_res


def render_stealth_decay_banner_html(div_data: dict) -> str:
    """
    Renders high-visibility Delta 5 top-of-page alert banner when structural breadth decay is active.
    """
    if not div_data or not div_data.get('is_diverging', False):
        if not div_data or div_data.get('sessions_count', 0) == 0:
            return ""
        # If emerging (1-2 days), render a subtle amber notice
        cnt = div_data.get('sessions_count', 0)
        return f"""
        <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(15, 23, 42, 0.95) 100%);
                    border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 10px; padding: 12px 18px; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.25rem;">🟡</span>
                    <div>
                        <span style="font-size: 0.88rem; font-weight: 700; color: #fbbf24;">EMERGING BREADTH DIVERGENCE: Cumulative A/D Line Diverging ({cnt} Sessions)</span>
                        <div style="font-size: 0.76rem; color: #94a3b8; margin-top: 2px;">
                            Index at <b>{div_data.get('price_prox_pct', 100):.1f}%</b> of 20-Day High while Cumulative A/D is lagging by <b>-{div_data.get('ad_deficit', 0):,}</b> issues.
                        </div>
                    </div>
                </div>
                <span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); padding: 3px 10px; border-radius: 6px; font-size: 0.72rem; font-weight: 700;">
                    EARLY WATCH
                </span>
            </div>
        </div>
        """

    cnt = div_data.get('sessions_count', 0)
    badge = div_data.get('status_badge', '🔴 ACTIVE DIVERGENCE')
    c_border = "#ef4444" if cnt >= 8 else "#f97316"
    c_bg = "rgba(239, 68, 68, 0.18)" if cnt >= 8 else "rgba(249, 115, 22, 0.16)"
    c_text = "#f87171" if cnt >= 8 else "#fb923c"

    return f"""
    <div style="background: linear-gradient(135deg, {c_bg} 0%, rgba(15, 23, 42, 0.95) 100%);
                border: 1px solid {c_border}; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);">
        <div style="display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
            <div style="display: flex; align-items: flex-start; gap: 12px; flex: 1; min-width: 280px;">
                <span style="font-size: 1.6rem; line-height: 1.1;">⚠️</span>
                <div>
                    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                        <span style="font-size: 0.96rem; font-weight: 800; color: {c_text}; letter-spacing: 0.5px;">
                            DELTA 5 ALERT: STEALTH BREADTH DECAY (MEGA-CAP MASKING)
                        </span>
                        <span style="background: {c_bg}; color: {c_text}; border: 1px solid {c_border}; padding: 2px 8px; border-radius: 5px; font-size: 0.72rem; font-weight: 700;">
                            {cnt} CONSECUTIVE SESSIONS
                        </span>
                    </div>
                    <div style="font-size: 0.82rem; color: #e2e8f0; margin-top: 5px; line-height: 1.4;">
                        <b>Historical Precedent (David Ryan & Michael Steinhardt):</b> Index trades near highs (<b>{div_data.get('price_prox_pct', 100):.1f}%</b> of 20-day high) while broad liquid universe A/D has shed <b>-{div_data.get('ad_deficit', 0):,}</b> net advancing issues.
                    </div>
                    <div style="font-size: 0.78rem; color: #fbbf24; font-weight: 600; margin-top: 6px;">
                        🛡️ Action Directive: {div_data.get('directive')}
                    </div>
                </div>
            </div>
            <div style="text-align: right; background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 8px 14px;">
                <div style="font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Divergence Status</div>
                <div style="font-size: 0.88rem; font-weight: 800; color: {c_text}; margin-top: 2px;">{badge}</div>
                <div style="font-size: 0.70rem; color: #64748b; margin-top: 2px;">Lead Time: 2–6 Weeks</div>
            </div>
        </div>
    </div>
    """


def evaluate_early_inflection_radar(
    breadth_eval: dict,
    zweig_tape: dict = None,
    breadth_alpha: dict = None,
    telem: dict = None,
    top_rs_21: float = 0.0,
    bullish_divs: list = None,
    bearish_divs: list = None,
    latest_date = None,
    market: str = "IN",
    rolling_ad_div: dict = None
) -> dict:
    """
    Evaluates the 8 Leading Inflection Signals (4 Early Bull Bottoms, 4 Early Bear Tops)
    with live traffic lights, lead-times, readings, and tactical execution directives.
    """
    stats = breadth_eval.get('summary_stats', {}) if breadth_eval else {}

    # -------------------------------------------------------------
    # 1. EARLY BULL BLAST-OFF SIGNALS (CYCLE BOTTOMS)
    # -------------------------------------------------------------
    # Signal 1: Zweig Breadth Thrust (ZBT)
    zbt_val = float(zweig_tape.get('zbt_10_ema', 0.50)) if zweig_tape else 0.50
    zbt_act = bool(zweig_tape.get('zbt_active', False)) if zweig_tape else bool(stats.get('thrust_active', False))
    days_zbt = zweig_tape.get('days_since_zbt', None) if zweig_tape else None

    if zbt_act:
        s1_status = 'ACTIVE'
        s1_badge = '🟢 ACTIVE THRUST'
        s1_color = '#10b981'
        s1_bg = 'rgba(16, 185, 129, 0.2)'
        s1_reading = f"{zbt_val:.3f} (10-EMA) • Fired {days_zbt}d ago" if days_zbt is not None else f"{zbt_val:.3f} (10-EMA) • Active"
        s1_directive = "Stage 4 200 SMA lock waived. Scale exposure to 100% into top Stage 2 leaders immediately."
    elif zbt_val >= 0.55:
        s1_status = 'ACCUMULATING'
        s1_badge = '🟡 ACCUMULATING'
        s1_color = '#f59e0b'
        s1_bg = 'rgba(245, 158, 11, 0.15)'
        s1_reading = f"{zbt_val:.3f} (10-EMA)"
        s1_directive = "Advance ratio expanding toward thrust threshold (≥0.615). Buying stampede approaching."
    else:
        s1_status = 'DORMANT'
        s1_badge = '⚪ DORMANT'
        s1_color = '#94a3b8'
        s1_bg = 'rgba(148, 163, 184, 0.12)'
        s1_reading = f"{zbt_val:.3f} (10-EMA)"
        s1_directive = "Dormant / neutral advance ratio. Awaiting explosive 10-day buying surge across universe."

    # Signal 2: Double 9:1 Up-Volume Stampede
    d91_act = bool(zweig_tape.get('double_9_to_1_active', False)) if zweig_tape else False
    c91 = int(zweig_tape.get('count_9_to_1_65d', 0)) if zweig_tape else 0
    is_today_91 = bool(zweig_tape.get('is_9_to_1_today', False)) if zweig_tape else False

    if d91_act:
        s2_status = 'ACTIVE'
        s2_badge = '🟢 DOUBLE 9:1 ACTIVE'
        s2_color = '#10b981'
        s2_bg = 'rgba(16, 185, 129, 0.2)'
        s2_reading = f"{c91} Stampedes in 65D"
        s2_directive = "Rare institutional stampede active (+30.2% 1-yr forward historical edge). Maximum aggressive exposure."
    elif c91 == 1 or is_today_91:
        s2_status = 'WATCH'
        s2_badge = '🟡 1x 9:1 DAY (1/2)'
        s2_color = '#f59e0b'
        s2_bg = 'rgba(245, 158, 11, 0.15)'
        s2_reading = "1 Day (Need 2nd to Trigger)"
        s2_directive = "Initial 90% up-volume shock logged. Watch for 2nd stampede day within 65 trading sessions."
    else:
        s2_status = 'DORMANT'
        s2_badge = '⚪ 0 IN LAST 65D'
        s2_color = '#94a3b8'
        s2_bg = 'rgba(148, 163, 184, 0.12)'
        s2_reading = "0 Days"
        s2_directive = "No recent 9:1 volume stampede detected. Broad exchange volume remains balanced or defensive."

    # Signal 3: Bullish Net A/D Line Divergence
    has_bull_div = False
    bull_reading = "A/D tracking index trend"
    if bullish_divs and len(bullish_divs) > 0:
        last_dt = pd.to_datetime(bullish_divs[-1][0])
        ref_dt = pd.to_datetime(latest_date) if latest_date is not None else pd.Timestamp.now()
        days_ago_bull = (ref_dt - last_dt).days
        if 0 <= days_ago_bull <= 45:
            has_bull_div = True
            bull_reading = f"Higher Low ({last_dt.strftime('%b %d')}, {days_ago_bull}d ago)"

    if has_bull_div:
        s3_status = 'ACTIVE'
        s3_badge = '🟢 ACTIVE DIVERGENCE'
        s3_color = '#10b981'
        s3_bg = 'rgba(16, 185, 129, 0.2)'
        s3_directive = "Stealth accumulation: Index lower low rejected by advancing stocks. Stop selling panic; build focus list."
    else:
        s3_status = 'DORMANT'
        s3_badge = '⚪ NO DIVERGENCE'
        s3_color = '#94a3b8'
        s3_bg = 'rgba(148, 163, 184, 0.12)'
        s3_directive = "No active stealth accumulation divergence. Cumulative Net A/D line aligned with price action."

    # Signal 4: Top 100 RS Leaders > 21 EMA Ignition
    t21 = float(top_rs_21)
    if t21 >= 70.0:
        s4_status = 'ACTIVE'
        s4_badge = '🟢 LEADERS SURGING (≥70%)'
        s4_color = '#10b981'
        s4_bg = 'rgba(16, 185, 129, 0.2)'
        s4_reading = f"{t21:.1f}% > 21 EMA"
        s4_directive = "Super-leaders decoupled and holding 21 EMA floor. Aggressively buy high-tight bases and initial breakouts."
    elif t21 >= 50.0:
        s4_status = 'WATCH'
        s4_badge = '🟡 EARLY RECOVERY (50–69%)'
        s4_color = '#f59e0b'
        s4_bg = 'rgba(245, 158, 11, 0.15)'
        s4_reading = f"{t21:.1f}% > 21 EMA"
        s4_directive = "Emerging leadership base: Half of top RS leaders reclaiming fast moving averages before the broad index."
    else:
        s4_status = 'INACTIVE'
        s4_badge = '🔴 NO IGNITION (<50%)'
        s4_color = '#ef4444'
        s4_bg = 'rgba(239, 68, 68, 0.15)'
        s4_reading = f"{t21:.1f}% > 21 EMA"
        s4_directive = "True leaders still below 21 EMA. Wait for top RS cohort to reclaim 21 EMA before committing heavy capital."

    bull_signals = [
        {
            'id': 'bull_zbt',
            'name': 'Dr. Martin Zweig Breadth Thrust (ZBT)',
            'icon': '⚡',
            'lead_time': '⚡ Day 0–10 at Low',
            'rule': 'Advance Ratio 10-EMA surges <0.400 → ≥0.615 in ≤10 sessions',
            'reading': s1_reading,
            'status': s1_status,
            'status_badge': s1_badge,
            'status_color': s1_color,
            'status_bg': s1_bg,
            'directive': s1_directive
        },
        {
            'id': 'bull_91',
            'name': 'Double 9:1 Up-Volume Stampede',
            'icon': '🌊',
            'lead_time': '🌊 Day 0–15 at Low',
            'rule': 'Two separate sessions with ≥90% Advancing Volume in 65 sessions',
            'reading': s2_reading,
            'status': s2_status,
            'status_badge': s2_badge,
            'status_color': s2_color,
            'status_bg': s2_bg,
            'directive': s2_directive
        },
        {
            'id': 'bull_div',
            'name': 'Bullish Net A/D Line Divergence',
            'icon': '🔭',
            'lead_time': '🔭 1 to 3 Weeks Ahead',
            'rule': 'Index prints lower price low while cumulative Net A/D Line holds higher low',
            'reading': bull_reading,
            'status': s3_status,
            'status_badge': s3_badge,
            'status_color': s3_color,
            'status_bg': s3_bg,
            'directive': s3_directive
        },
        {
            'id': 'bull_rs',
            'name': 'Top 100 RS Leaders > 21 EMA Ignition',
            'icon': '👑',
            'lead_time': '👑 2 to 4 Weeks Ahead',
            'rule': 'Market leaders bottom first and reclaim 21 EMA (≥70%) before index',
            'reading': s4_reading,
            'status': s4_status,
            'status_badge': s4_badge,
            'status_color': s4_color,
            'status_bg': s4_bg,
            'directive': s4_directive
        }
    ]

    # -------------------------------------------------------------
    # 2. EARLY BEAR DISTRIBUTION SIGNALS (CYCLE TOPS)
    # -------------------------------------------------------------
    # Signal 5: Distribution Canary (New 52W Lows)
    nl = int(stats.get('nl_current', 0))
    canary = bool(stats.get('canary_warning', False))

    if canary or nl >= 15:
        s5_status = 'ALERT'
        s5_badge = '🔴 CANARY FIRING (ALERT)'
        s5_color = '#ef4444'
        s5_bg = 'rgba(239, 68, 68, 0.2)'
        s5_reading = f"🚨 {nl} New Lows"
        s5_directive = "Distribution Canary Active: Severe internal breakdown while index masks weakness. Stop buying breakouts."
    elif nl >= 8:
        s5_status = 'WATCH'
        s5_badge = '🟡 CANARY CREEPING (8–14)'
        s5_color = '#f59e0b'
        s5_bg = 'rgba(245, 158, 11, 0.15)'
        s5_reading = f"{nl} New Lows"
        s5_directive = "Subtle expansion of new lows. Institutional support weakening under the surface; raise stop discipline."
    else:
        s5_status = 'CALM'
        s5_badge = '🟢 CALM (<8 LOWS)'
        s5_color = '#10b981'
        s5_bg = 'rgba(16, 185, 129, 0.15)'
        s5_reading = f"{nl} New Lows"
        s5_directive = "New lows safely subdued. No stealth institutional liquidation beneath the headline surface."

    # Signal 6: Acute Distribution Clusters (Ross Haber Shock)
    clust_act = bool(telem.get('cluster_detected', False)) if telem else False
    c_7d = int(telem.get('cluster_count_7d', 0)) if telem else 0
    dd_tot = int(telem.get('dd_count', 0)) if telem else 0

    if clust_act or c_7d >= 3:
        s6_status = 'ALERT'
        s6_badge = '🔴 ACUTE CLUSTER (≥3 IN 7D)'
        s6_color = '#ef4444'
        s6_bg = 'rgba(239, 68, 68, 0.2)'
        s6_reading = f"🚨 {c_7d} DDs in 7 Sessions ({dd_tot} total)"
        s6_directive = "Institutional dump shock: Multiple heavy selling sessions packed in 1 week. Immediately slash exposure to 25%–50%."
    elif dd_tot >= 5:
        s6_status = 'WATCH'
        s6_badge = '🟡 ELEVATED SUPPLY (≥5 DDs)'
        s6_color = '#f59e0b'
        s6_bg = 'rgba(245, 158, 11, 0.15)'
        s6_reading = f"{dd_tot} Distribution Days"
        s6_directive = "Uptrend Under Pressure: Substantial institutional selling in last 25 sessions. Avoid new risk; let cash build."
    else:
        s6_status = 'CALM'
        s6_badge = '🟢 ORDERLY TAPE (<5 DDs)'
        s6_color = '#10b981'
        s6_bg = 'rgba(16, 185, 129, 0.15)'
        s6_reading = f"{dd_tot} Distribution Days"
        s6_directive = "Orderly institutional flow. Low systematic distribution across benchmark."

    # Signal 7: Mega-Cap Divergence (The "Index Lie")
    mega_cap_div = bool(stats.get('mega_cap_divergence', False))
    if market == 'US':
        p_beat = float(breadth_alpha.get('pct_beat_sp500_1m', breadth_alpha.get('pct_beat_n500_1m', 50.0))) if breadth_alpha else 50.0
        sp_val = float(breadth_alpha.get('spread_sp500_1m', breadth_alpha.get('spread_n50_1m', 0.0))) if breadth_alpha else 0.0
    else:
        p_beat = float(breadth_alpha.get('pct_beat_n500_1m', 50.0)) if breadth_alpha else 50.0
        sp_val = float(breadth_alpha.get('spread_n50_1m', 0.0)) if breadth_alpha else 0.0

    if mega_cap_div or (p_beat < 38.0 and sp_val < -0.5):
        s7_status = 'ALERT'
        s7_badge = '🔴 SEVERE INDEX LIE (<38%)'
        s7_color = '#ef4444'
        s7_bg = 'rgba(239, 68, 68, 0.2)'
        s7_reading = f"🚨 {p_beat:.1f}% Beating • EW {sp_val:+.2f}%"
        s7_directive = "Index Lie Active: Cap-weight index advancing on narrow mega-caps while majority liquidates. Do not trust index."
    elif p_beat < 45.0 or sp_val < -0.3:
        s7_status = 'WATCH'
        s7_badge = '🟡 SELECTIVE RALLY'
        s7_color = '#f59e0b'
        s7_bg = 'rgba(245, 158, 11, 0.15)'
        s7_reading = f"{p_beat:.1f}% Beating • EW {sp_val:+.2f}%"
        s7_directive = "Narrowing breadth: Fewer than 45% of universe beating benchmark. Exercise selective restraint."
    else:
        s7_status = 'CALM'
        s7_badge = '🟢 BROAD PARTICIPATION'
        s7_color = '#10b981'
        s7_bg = 'rgba(16, 185, 129, 0.15)'
        s7_reading = f"{p_beat:.1f}% Beating • EW {sp_val:+.2f}%"
        s7_directive = "Broad market participation confirmed. Active momentum outpacing or matching blue chips."

    # Signal 8: Bearish Net A/D Line Divergence (Delta 5 Zero-Lag + Peak Divergence)
    has_bear_div = False
    bear_reading = "Cumulative A/D confirms trend"
    if rolling_ad_div and rolling_ad_div.get('is_diverging', False):
        has_bear_div = True
        cnt_d = rolling_ad_div.get('sessions_count', 0)
        s8_status = 'ALERT'
        s8_badge = rolling_ad_div.get('status_badge', f'🔴 STEALTH DECAY ({cnt_d}d)')
        s8_color = rolling_ad_div.get('status_color', '#ef4444')
        s8_bg = rolling_ad_div.get('status_bg', 'rgba(239, 68, 68, 0.2)')
        bear_reading = f"🚨 {cnt_d}d Active Div • Deficit: -{rolling_ad_div.get('ad_deficit', 0):,} issues"
        s8_directive = rolling_ad_div.get('directive')
    elif rolling_ad_div and rolling_ad_div.get('sessions_count', 0) > 0:
        cnt_d = rolling_ad_div.get('sessions_count', 0)
        s8_status = 'WATCH'
        s8_badge = f'🟡 EMERGING DIVERGENCE ({cnt_d}d)'
        s8_color = '#f59e0b'
        s8_bg = 'rgba(245, 158, 11, 0.15)'
        bear_reading = f"⚠️ {cnt_d}d Emerging Div • Deficit: -{rolling_ad_div.get('ad_deficit', 0):,} issues"
        s8_directive = "Emerging divergence: Benchmark near high while Net A/D is lagging. Tighten entry filters."
    elif bearish_divs and len(bearish_divs) > 0:
        last_dt = pd.to_datetime(bearish_divs[-1][0])
        ref_dt = pd.to_datetime(latest_date) if latest_date is not None else pd.Timestamp.now()
        days_ago_bear = (ref_dt - last_dt).days
        if 0 <= days_ago_bear <= 45:
            has_bear_div = True
            bear_reading = f"Lower High ({last_dt.strftime('%b %d')}, {days_ago_bear}d ago)"
            s8_status = 'ALERT'
            s8_badge = '🔴 ACTIVE DIVERGENCE'
            s8_color = '#ef4444'
            s8_bg = 'rgba(239, 68, 68, 0.2)'
            s8_directive = "Classic macro top signal: Index sets higher high while Net A/D makes lower high. Lock profits; tighten stops."
        else:
            s8_status = 'CALM'
            s8_badge = '🟢 CONFIRMED ADVANCE'
            s8_color = '#10b981'
            s8_bg = 'rgba(16, 185, 129, 0.15)'
            s8_directive = "No bearish divergence. Cumulative Advance/Decline line confirming price strength."
    else:
        s8_status = 'CALM'
        s8_badge = '🟢 CONFIRMED ADVANCE'
        s8_color = '#10b981'
        s8_bg = 'rgba(16, 185, 129, 0.15)'
        s8_directive = "No bearish divergence. Cumulative Advance/Decline line confirming price strength."

    bear_signals = [
        {
            'id': 'bear_canary',
            'name': 'Distribution Canary (New 52W Lows)',
            'icon': '🐤',
            'lead_time': '🐤 2 to 4 Weeks Ahead',
            'rule': 'New 52W Lows expand to ≥15 while index trades near highs',
            'reading': s5_reading,
            'status': s5_status,
            'status_badge': s5_badge,
            'status_color': s5_color,
            'status_bg': s5_bg,
            'directive': s5_directive
        },
        {
            'id': 'bear_cluster',
            'name': 'Acute Distribution Clusters',
            'icon': '⚡',
            'lead_time': '⚡ 1 to 2 Weeks Ahead',
            'rule': '≥3 heavy-volume distribution days packed inside 7 sessions',
            'reading': s6_reading,
            'status': s6_status,
            'status_badge': s6_badge,
            'status_color': s6_color,
            'status_bg': s6_bg,
            'directive': s6_directive
        },
        {
            'id': 'bear_megacap',
            'name': 'Mega-Cap Divergence (The "Index Lie")',
            'icon': '🎭',
            'lead_time': '🎭 2 to 4 Weeks Ahead',
            'rule': 'Index hits highs on 3–5 mega-caps while <38% beat it and EW Spread <-0.5%',
            'reading': s7_reading,
            'status': s7_status,
            'status_badge': s7_badge,
            'status_color': s7_color,
            'status_bg': s7_bg,
            'directive': s7_directive
        },
        {
            'id': 'bear_div',
            'name': 'Bearish Net A/D Line Divergence',
            'icon': '📉',
            'lead_time': '📉 2 to 6 Weeks Ahead',
            'rule': 'Index prints higher price high while cumulative Net A/D Line prints lower high',
            'reading': bear_reading,
            'status': s8_status,
            'status_badge': s8_badge,
            'status_color': s8_color,
            'status_bg': s8_bg,
            'directive': s8_directive
        }
    ]

    return {
        'bull_signals': bull_signals,
        'bear_signals': bear_signals,
        'bull_active_count': sum(1 for s in bull_signals if s['status'] == 'ACTIVE'),
        'bear_alert_count': sum(1 for s in bear_signals if s['status'] == 'ALERT')
    }


def render_early_inflection_radar_html(radar_data: dict) -> str:
    """
    Renders the HTML for the dual-panel Early Inflection Radar (Bull Bottoms vs Bear Tops).
    """
    bull_items = radar_data.get('bull_signals', [])
    bear_items = radar_data.get('bear_signals', [])
    bull_active_count = radar_data.get('bull_active_count', 0)
    bear_alert_count = radar_data.get('bear_alert_count', 0)

    def render_card(it):
        return f"""
        <div class="mb-radar-card" style="border-left: 3px solid {it['status_color']};">
            <div class="mb-radar-card-header">
                <span class="mb-radar-card-name">{it['icon']} {it['name']}</span>
                <span class="mb-radar-lead-pill">{it['lead_time']}</span>
            </div>
            <div class="mb-radar-card-body">
                <span class="mb-radar-reading" style="color: {it['status_color']};">{it['reading']}</span>
                <span class="mb-radar-status-pill" style="background: {it['status_bg']}; color: {it['status_color']}; border: 1.5px solid {it['status_color']}77; font-size: 0.76rem; font-weight: 800; padding: 4px 10px;">
                    {it['status_badge']}
                </span>
            </div>
            <div class="mb-radar-directive">
                <div style="font-size: 0.68rem; color: #64748b; margin-bottom: 2px;"><b>RULE:</b> {it['rule']}</div>
                <div><b style="color: #cbd5e1;">DIRECTIVE:</b> <span style="color: #e2e8f0;">{it['directive']}</span></div>
            </div>
        </div>
        """

    bull_cards = "\n".join(render_card(it) for it in bull_items)
    bear_cards = "\n".join(render_card(it) for it in bear_items)

    if bull_active_count > 0:
        bull_badge_style = "background: rgba(16, 185, 129, 0.25); color: #34d399; border: 2px solid #10b981; box-shadow: 0 0 18px rgba(16, 185, 129, 0.4); padding: 7px 16px; border-radius: 24px; font-size: 0.95rem; font-weight: 900; letter-spacing: 0.03em; white-space: nowrap;"
        bull_badge_txt = f"🟢 {bull_active_count} OF 4 ACTIVE"
    else:
        bull_badge_style = "background: rgba(148, 163, 184, 0.16); color: #cbd5e1; border: 1.5px solid rgba(148, 163, 184, 0.4); padding: 7px 16px; border-radius: 24px; font-size: 0.92rem; font-weight: 800; letter-spacing: 0.02em; white-space: nowrap;"
        bull_badge_txt = "⚪ 0 OF 4 ACTIVE (DORMANT)"

    if bear_alert_count > 0:
        bear_badge_style = "background: rgba(239, 68, 68, 0.3); color: #fca5a5; border: 2.5px solid #ef4444; box-shadow: 0 0 22px rgba(239, 68, 68, 0.55); padding: 7px 16px; border-radius: 24px; font-size: 0.95rem; font-weight: 900; letter-spacing: 0.03em; white-space: nowrap;"
        bear_badge_txt = f"🚨 {bear_alert_count} OF 4 ALERTS FIRING"
    else:
        bear_badge_style = "background: rgba(16, 185, 129, 0.18); color: #34d399; border: 2px solid #10b981; box-shadow: 0 0 14px rgba(16, 185, 129, 0.25); padding: 7px 16px; border-radius: 24px; font-size: 0.92rem; font-weight: 900; letter-spacing: 0.02em; white-space: nowrap;"
        bear_badge_txt = "🟢 0 ALERTS (ORDERLY TAPE)"

    html = f"""
    <div class="mb-panel" style="margin-top: 14px; margin-bottom: 20px;">
        <div class="mb-panel-header" style="margin-bottom: 12px; padding-bottom: 10px;">
            <div>
                <h4 class="mb-panel-title">🚦 Institutional Early Inflection Radar (Live Traffic Lights)</h4>
                <p class="mb-panel-desc">Real-time quantitative early warning system: surfaces institutional turning points 1 to 6 weeks before index confirmation</p>
            </div>
            <span style="background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); padding: 3px 10px; border-radius: 6px; font-size: 0.74rem; font-weight: 700;">
                Leading Horizon Telemetry
            </span>
        </div>

        <div class="mb-radar-grid">
            <!-- Left Panel: Bull Bottoms -->
            <div class="mb-radar-col">
                <div class="mb-radar-col-header">
                    <div>
                        <div style="font-size: 0.94rem; font-weight: 800; color: #34d399; display: flex; align-items: center; gap: 6px;">
                            <span>🚀 Early Bull Blast-Off Radar (Cycle Bottoms)</span>
                        </div>
                        <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 2px;">Fires at Lows & Leads Index Confirmation by 1 to 4 Weeks</div>
                    </div>
                    <span style="{bull_badge_style}">
                        {bull_badge_txt}
                    </span>
                </div>
                {bull_cards}
            </div>

            <!-- Right Panel: Bear Tops -->
            <div class="mb-radar-col">
                <div class="mb-radar-col-header">
                    <div>
                        <div style="font-size: 0.94rem; font-weight: 800; color: #f87171; display: flex; align-items: center; gap: 6px;">
                            <span>🚨 Early Bear Distribution Radar (Cycle Tops)</span>
                        </div>
                        <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 2px;">Warns of Smart Money Liquidation 1 to 6 Weeks Before Breakdown</div>
                    </div>
                    <span style="{bear_badge_style}">
                        {bear_badge_txt}
                    </span>
                </div>
                {bear_cards}
            </div>
        </div>
    </div>
    """
    return html


def render_early_inflection_cheat_sheet_html() -> str:
    """
    Returns an institutional-grade Cheat Sheet table explaining the 8 signals,
    their lead times, triggers, historical precedent, and execution directives.
    """
    return """
    <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;">
        <h5 style="color: #38bdf8; margin: 0 0 10px 0; font-size: 0.98rem; font-weight: 800;">
            📚 Institutional Early Inflection Lead-Time Playbook (The Complete Cheat Sheet)
        </h5>
        <p style="margin: 0 0 14px 0; font-size: 0.76rem; color: #94a3b8;">
            Empirical lead times and execution directives derived from historical market cycles (1982–2024 US & 2003–2024 India).
            Never wait for lagging index moving averages when quantitative inflection signals fire.
        </p>

        <div style="overflow-x: auto; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; margin-bottom: 14px;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.76rem; text-align: left;">
                <thead>
                    <tr style="background: rgba(15, 23, 42, 0.9); border-bottom: 1px solid rgba(255,255,255,0.1); color: #f8fafc;">
                        <th style="padding: 10px 12px; font-weight: 700;">Signal Name</th>
                        <th style="padding: 10px 12px; font-weight: 700;">Regime</th>
                        <th style="padding: 10px 12px; font-weight: 700;">Lead Time</th>
                        <th style="padding: 10px 12px; font-weight: 700;">Quantitative Signal Rule</th>
                        <th style="padding: 10px 12px; font-weight: 700;">Empirical Precedent / Edge</th>
                        <th style="padding: 10px 12px; font-weight: 700;">Tactical Execution Directive</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.04); background: rgba(16, 185, 129, 0.04);">
                        <td style="padding: 10px 12px; font-weight: 700; color: #34d399;">⚡ Zweig Breadth Thrust (ZBT)</td>
                        <td style="padding: 10px 12px;"><span style="background: rgba(16,185,129,0.2); color: #34d399; padding: 2px 6px; border-radius: 4px; font-weight: 800;">🚀 BULL</span></td>
                        <td style="padding: 10px 12px; font-family: 'JetBrains Mono'; font-weight: 700; color: #38bdf8;">Day 0–10 at Low</td>
                        <td style="padding: 10px 12px;">10-day EMA of Advances ÷ (Adv + Dec) surges from &lt;0.400 to &ge;0.615 in &le;10 sessions.</td>
                        <td style="padding: 10px 12px; color: #94a3b8;">Never failed in 80 years without double-digit forward gains; triggered Mar 2023 India (+529% cycle).</td>
                        <td style="padding: 10px 12px; color: #f8fafc; font-weight: 600;">Waives 200 SMA lock; aggressively ramp exposure to 100% in top Smooth RS leaders.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.04); background: rgba(16, 185, 129, 0.02);">
                        <td style="padding: 10px 12px; font-weight: 700; color: #34d399;">🌊 Double 9:1 Up-Volume Stampede</td>
                        <td style="padding: 10px 12px;"><span style="background: rgba(16,185,129,0.2); color: #34d399; padding: 2px 6px; border-radius: 4px; font-weight: 800;">🚀 BULL</span></td>
                        <td style="padding: 10px 12px; font-family: 'JetBrains Mono'; font-weight: 700; color: #38bdf8;">Day 0–15 at Low</td>
                        <td style="padding: 10px 12px;">Two separate sessions with Advancing Volume &ge;90% of total exchange volume in 65 sessions.</td>
                        <td style="padding: 10px 12px; color: #94a3b8;">Historical 1-year forward return edge of +30.2% with &gt;90% win rate across US & Indian markets.</td>
                        <td style="padding: 10px 12px; color: #f8fafc; font-weight: 600;">Immediate maximum portfolio exposure; buy initial Stage 2 pivot breakouts without waiting for moving averages.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.04); background: rgba(16, 185, 129, 0.04);">
                        <td style="padding: 10px 12px; font-weight: 700; color: #34d399;">🔭 Bullish Net A/D Line Divergence</td>
                        <td style="padding: 10px 12px;"><span style="background: rgba(16,185,129,0.2); color: #34d399; padding: 2px 6px; border-radius: 4px; font-weight: 800;">🚀 BULL</span></td>
                        <td style="padding: 10px 12px; font-family: 'JetBrains Mono'; font-weight: 700; color: #38bdf8;">1 to 3 Wks Ahead</td>
                        <td style="padding: 10px 12px;">Index carves lower price low while cumulative Net A/D Line holds higher low (stealth accumulation).</td>
                        <td style="padding: 10px 12px; color: #94a3b8;">Flagged Oct 2020 and Mar 2023 cycle bottoms weeks before index breakout confirmations.</td>
                        <td style="padding: 10px 12px; color: #f8fafc; font-weight: 600;">Stop selling into panic; finalize high-conviction focus list; pilot test initial base breakouts.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.08); background: rgba(16, 185, 129, 0.02);">
                        <td style="padding: 10px 12px; font-weight: 700; color: #34d399;">👑 Top 100 RS Leaders > 21 EMA</td>
                        <td style="padding: 10px 12px;"><span style="background: rgba(16,185,129,0.2); color: #34d399; padding: 2px 6px; border-radius: 4px; font-weight: 800;">🚀 BULL</span></td>
                        <td style="padding: 10px 12px; font-family: 'JetBrains Mono'; font-weight: 700; color: #38bdf8;">2 to 4 Wks Ahead</td>
                        <td style="padding: 10px 12px;">Percentage of top 100 Relative Strength leaders reclaiming 21-day EMA surges &ge;70%.</td>
                        <td style="padding: 10px 12px; color: #94a3b8;">True leaders bottom 2–4 weeks before the index (Ross Haber Institutional Rule).</td>
                        <td style="padding: 10px 12px; color: #f8fafc; font-weight: 600;">Super-leaders decoupled; buy high-tight bases and initial breakouts regardless of index chop.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.04); background: rgba(239, 68, 68, 0.04);">
                        <td style="padding: 10px 12px; font-weight: 700; color: #f87171;">🐤 Distribution Canary (New Lows)</td>
                        <td style="padding: 10px 12px;"><span style="background: rgba(239,68,68,0.2); color: #f87171; padding: 2px 6px; border-radius: 4px; font-weight: 800;">🚨 BEAR</span></td>
                        <td style="padding: 10px 12px; font-family: 'JetBrains Mono'; font-weight: 700; color: #fbbf24;">2 to 4 Wks Ahead</td>
                        <td style="padding: 10px 12px;">New 52-Week Lows expand to &ge;15 while benchmark index is near highs.</td>
                        <td style="padding: 10px 12px; color: #94a3b8;">Preceded Jan 2022 and Oct 2021 market tops while blue chips were still making all-time highs.</td>
                        <td style="padding: 10px 12px; color: #f8fafc; font-weight: 600;">Stop buying new breakouts; raise cash on strength; tighten stops to break-even or 21 EMA.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.04); background: rgba(239, 68, 68, 0.02);">
                        <td style="padding: 10px 12px; font-weight: 700; color: #f87171;">⚡ Acute Distribution Clusters</td>
                        <td style="padding: 10px 12px;"><span style="background: rgba(239,68,68,0.2); color: #f87171; padding: 2px 6px; border-radius: 4px; font-weight: 800;">🚨 BEAR</span></td>
                        <td style="padding: 10px 12px; font-family: 'JetBrains Mono'; font-weight: 700; color: #fbbf24;">1 to 2 Wks Ahead</td>
                        <td style="padding: 10px 12px;">&ge;3 heavy-volume distribution days concentrated within just 7 trading sessions.</td>
                        <td style="padding: 10px 12px; color: #94a3b8;">Ross Haber distribution shock: institutional smart money systematically liquidating inventory into retail.</td>
                        <td style="padding: 10px 12px; color: #f8fafc; font-weight: 600;">Emergency defense: immediately cut exposure to 25%–50%; cash is a position.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.04); background: rgba(239, 68, 68, 0.04);">
                        <td style="padding: 10px 12px; font-weight: 700; color: #f87171;">🎭 Mega-Cap Divergence (Index Lie)</td>
                        <td style="padding: 10px 12px;"><span style="background: rgba(239,68,68,0.2); color: #f87171; padding: 2px 6px; border-radius: 4px; font-weight: 800;">🚨 BEAR</span></td>
                        <td style="padding: 10px 12px; font-family: 'JetBrains Mono'; font-weight: 700; color: #fbbf24;">2 to 4 Wks Ahead</td>
                        <td style="padding: 10px 12px;">Index rises on 3–5 mega-caps while &lt;38% of universe beats it and Equal-Weight spread &lt;-0.5%.</td>
                        <td style="padding: 10px 12px; color: #94a3b8;">Occurred at 2000 dot-com peak, Sep 2024 Nifty peak, and 2021 speculative blow-offs.</td>
                        <td style="padding: 10px 12px; color: #f8fafc; font-weight: 600;">Do NOT trust index breakouts; average stocks are liquidating; take partial profits and trail stops.</td>
                    </tr>
                    <tr style="background: rgba(239, 68, 68, 0.02);">
                        <td style="padding: 10px 12px; font-weight: 700; color: #f87171;">📉 Bearish Net A/D Line Divergence</td>
                        <td style="padding: 10px 12px;"><span style="background: rgba(239,68,68,0.2); color: #f87171; padding: 2px 6px; border-radius: 4px; font-weight: 800;">🚨 BEAR</span></td>
                        <td style="padding: 10px 12px; font-family: 'JetBrains Mono'; font-weight: 700; color: #fbbf24;">2 to 6 Wks Ahead</td>
                        <td style="padding: 10px 12px;">Index prints higher price high while cumulative Net A/D Line prints lower high.</td>
                        <td style="padding: 10px 12px; color: #94a3b8;">Classic Stan Weinstein / IBD divergence warning; signals underlying participation evaporation.</td>
                        <td style="padding: 10px 12px; color: #f8fafc; font-weight: 600;">Lock in swing profits; reduce position sizing; prepare for structural trend breakdown.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 10px 14px; font-size: 0.78rem; line-height: 1.5; color: #e0f2fe;">
            💡 <b>The Two Golden Rules of Market Inflection:</b><br>
            • <b>Bottoms Are Explosive:</b> Zweig Thrusts & 9:1 Up-Volume Stampedes ignite instantly at cycle lows. Leaders bottom 2 to 4 weeks before the index. Never wait for 200 SMA confirmation when a ZBT or Double 9:1 is active!<br>
            • <b>Tops Are Deceptive:</b> Distribution starts quietly underneath. The Canary (New Lows) and A/D Divergence appear while the index still looks invincible. If 3 Distribution Days cluster in 7 days, step aside immediately.
        </div>
    </div>
    """


def compute_institutional_breadth_alpha_in_core(df_nifty500_close, df_nifty50_close) -> dict | None:
    """Pure zero-Streamlit computation of India dual-benchmark Alpha & Multi-TF Highs/Lows."""
    matrix_path = 'historical_prices_matrix.pkl'
    if not os.path.exists(matrix_path) and os.path.exists('breadth_matrix.pkl'):
        matrix_path = 'breadth_matrix.pkl'
    if not os.path.exists(matrix_path):
        return None

    try:
        matrix_df = pd.read_pickle(matrix_path)
        closes = matrix_df.xs('Close', level=1, axis=1) if matrix_df.columns.nlevels > 1 else matrix_df[['Close']]
        highs = matrix_df.xs('High', level=1, axis=1) if matrix_df.columns.nlevels > 1 else matrix_df[['High']]
        lows = matrix_df.xs('Low', level=1, axis=1) if matrix_df.columns.nlevels > 1 else matrix_df[['Low']]
    except Exception as e:
        print(f"Error reading matrix for breadth alpha: {e}")
        return None

    try:
        from nifty500_universe import fetch_nifty500_tickers, FALLBACK_TICKERS
        n500_symbols = fetch_nifty500_tickers(show_progress=False)
        fallback_list = FALLBACK_TICKERS[:50]
    except Exception:
        n500_symbols = []
        fallback_list = []

    n500_cols = [f"{t}.NS" for t in n500_symbols if f"{t}.NS" in closes.columns]
    if len(n500_cols) < 50:
        n500_cols = [c for c in closes.columns if str(c).endswith('.NS')][:500]
    if not n500_cols:
        return None

    n50_cols = [f"{t}.NS" for t in fallback_list if f"{t}.NS" in closes.columns]
    if len(n50_cols) < 20:
        n50_cols = n500_cols[:50]

    from price_history_manager import filter_panel_active_market_dates
    closes = filter_panel_active_market_dates(closes[n500_cols], min_coverage_ratio=0.20, min_abs=20)
    if closes.empty:
        return None
    for _pnl in (highs, lows):
        if getattr(_pnl.index, "tz", None) is not None:
            _pnl.index = _pnl.index.tz_localize(None)
        _pnl.index = pd.to_datetime(_pnl.index).normalize()
    highs = highs[~highs.index.duplicated(keep='last')][n500_cols].reindex(closes.index)
    lows = lows[~lows.index.duplicated(keep='last')][n500_cols].reindex(closes.index)

    n500_s = df_nifty500_close.dropna()
    n50_s = df_nifty50_close.dropna()
    ret_n500_1m = ((n500_s.iloc[-1] / n500_s.iloc[-22]) - 1) * 100 if len(n500_s) >= 22 else 0.0
    ret_n500_3m = ((n500_s.iloc[-1] / n500_s.iloc[-64]) - 1) * 100 if len(n500_s) >= 64 else 0.0
    ret_n50_1m = ((n50_s.iloc[-1] / n50_s.iloc[-22]) - 1) * 100 if len(n50_s) >= 22 else 0.0
    ret_n50_3m = ((n50_s.iloc[-1] / n50_s.iloc[-64]) - 1) * 100 if len(n50_s) >= 64 else 0.0

    s_1m = ((closes[n500_cols].iloc[-1] / closes[n500_cols].iloc[-22]) - 1).dropna() * 100 if len(closes) >= 22 else pd.Series()
    s_3m = ((closes[n500_cols].iloc[-1] / closes[n500_cols].iloc[-64]) - 1).dropna() * 100 if len(closes) >= 64 else pd.Series()
    pct_beat_n500_1m = float((s_1m > ret_n500_1m).mean() * 100) if not s_1m.empty else 50.0
    pct_beat_n500_3m = float((s_3m > ret_n500_3m).mean() * 100) if not s_3m.empty else 50.0
    pct_beat_n50_1m = float((s_1m > ret_n50_1m).mean() * 100) if not s_1m.empty else 50.0
    pct_beat_n50_3m = float((s_3m > ret_n50_3m).mean() * 100) if not s_3m.empty else 50.0

    ew_n500_1m = float(s_1m.mean()) if not s_1m.empty else 0.0
    spread_n500_1m = ew_n500_1m - ret_n500_1m
    s50_1m = ((closes[n50_cols].iloc[-1] / closes[n50_cols].iloc[-22]) - 1).dropna() * 100 if len(closes) >= 22 else pd.Series()
    ew_n50_1m = float(s50_1m.mean()) if not s50_1m.empty else 0.0
    spread_n50_1m = ew_n50_1m - ret_n50_1m

    horizons = {'1W (5D)': 5, '1M (21D)': 21, '3M (63D)': 63, '1Y (52W)': 250}
    c_curr = closes[n500_cols].iloc[-1]
    multi_hl = {}
    for name, window in horizons.items():
        w_act = min(window, len(closes))
        h_win = highs[n500_cols].iloc[-w_act:].max()
        l_win = lows[n500_cols].iloc[-w_act:].min()
        valid = c_curr.notna() & h_win.notna() & l_win.notna()
        nh = int((c_curr[valid] >= (h_win[valid] * 0.98)).sum())
        nl = int((c_curr[valid] <= (l_win[valid] * 1.02)).sum())
        multi_hl[name] = {'Highs': nh, 'Lows': nl, 'Net': nh - nl}

    return {
        'ret_n500_1m': ret_n500_1m,
        'ret_n500_3m': ret_n500_3m,
        'ret_n50_1m': ret_n50_1m,
        'ret_n50_3m': ret_n50_3m,
        'pct_beat_n500_1m': pct_beat_n500_1m,
        'pct_beat_n500_3m': pct_beat_n500_3m,
        'pct_beat_n50_1m': pct_beat_n50_1m,
        'pct_beat_n50_3m': pct_beat_n50_3m,
        'ew_n50_1m': ew_n50_1m,
        'spread_n50_1m': spread_n50_1m,
        'ew_n500_1m': ew_n500_1m,
        'spread_n500_1m': spread_n500_1m,
        'multi_hl': multi_hl,
    }


def compute_institutional_breadth_alpha_us_core(df_sp500_close, df_nasdaq_close) -> dict | None:
    """Pure zero-Streamlit computation of US dual-benchmark Alpha & Multi-TF Highs/Lows."""
    from price_history_manager import load_cached_price_matrix, extract_market_ohlcv_panels, get_matrix_cache_signature
    matrix_path = 'historical_prices_matrix.pkl' if os.path.exists('historical_prices_matrix.pkl') else 'breadth_matrix.pkl'
    if not os.path.exists(matrix_path):
        return None
    try:
        matrix_df = load_cached_price_matrix(matrix_path, cache_sig=get_matrix_cache_signature(matrix_path))
        panels = extract_market_ohlcv_panels(
            matrix_df, market='US', include_etfs=False, min_coverage_ratio=0.15, min_abs=20
        )
        closes, highs, lows = panels['close'], panels['high'], panels['low']
    except Exception as e:
        print(f"Error reading matrix for US breadth alpha: {e}")
        return None

    if closes.empty or len(closes.columns) < 50:
        if matrix_path != 'breadth_matrix.pkl' and os.path.exists('breadth_matrix.pkl'):
            try:
                matrix_df = load_cached_price_matrix('breadth_matrix.pkl', cache_sig=get_matrix_cache_signature('breadth_matrix.pkl'))
                panels = extract_market_ohlcv_panels(matrix_df, market='US', include_etfs=False, min_coverage_ratio=0.15, min_abs=20)
                closes, highs, lows = panels['close'], panels['high'], panels['low']
            except Exception:
                return None
        if closes.empty or len(closes.columns) < 50:
            return None

    sp500_s = df_sp500_close.dropna()
    nasdaq_s = df_nasdaq_close.dropna()
    ret_sp500_1m = ((sp500_s.iloc[-1] / sp500_s.iloc[-22]) - 1) * 100 if len(sp500_s) >= 22 else 0.0
    ret_sp500_3m = ((sp500_s.iloc[-1] / sp500_s.iloc[-64]) - 1) * 100 if len(sp500_s) >= 64 else 0.0
    ret_ndx_1m = ((nasdaq_s.iloc[-1] / nasdaq_s.iloc[-22]) - 1) * 100 if len(nasdaq_s) >= 22 else 0.0
    ret_ndx_3m = ((nasdaq_s.iloc[-1] / nasdaq_s.iloc[-64]) - 1) * 100 if len(nasdaq_s) >= 64 else 0.0

    s_1m = ((closes.iloc[-1] / closes.iloc[-22]) - 1).dropna() * 100 if len(closes) >= 22 else pd.Series()
    s_3m = ((closes.iloc[-1] / closes.iloc[-64]) - 1).dropna() * 100 if len(closes) >= 64 else pd.Series()
    pct_beat_sp_1m = float((s_1m > ret_sp500_1m).mean() * 100) if not s_1m.empty else 50.0
    pct_beat_sp_3m = float((s_3m > ret_sp500_3m).mean() * 100) if not s_3m.empty else 50.0
    pct_beat_ndx_1m = float((s_1m > ret_ndx_1m).mean() * 100) if not s_1m.empty else 50.0
    pct_beat_ndx_3m = float((s_3m > ret_ndx_3m).mean() * 100) if not s_3m.empty else 50.0

    ew_us_1m = float(s_1m.mean()) if not s_1m.empty else 0.0
    spread_sp_1m = ew_us_1m - ret_sp500_1m
    spread_ndx_1m = ew_us_1m - ret_ndx_1m

    horizons = {'1W (5D)': 5, '1M (21D)': 21, '3M (63D)': 63, '1Y (52W)': 250}
    c_curr = closes.iloc[-1]
    multi_hl = {}
    for name, window in horizons.items():
        w_act = min(window, len(closes))
        h_win = highs.iloc[-w_act:].max()
        l_win = lows.iloc[-w_act:].min()
        valid = c_curr.notna() & h_win.notna() & l_win.notna()
        nh = int((c_curr[valid] >= (h_win[valid] * 0.98)).sum())
        nl = int((c_curr[valid] <= (l_win[valid] * 1.02)).sum())
        multi_hl[name] = {'Highs': nh, 'Lows': nl, 'Net': nh - nl}

    return {
        'ret_sp500_1m': ret_sp500_1m,
        'ret_sp500_3m': ret_sp500_3m,
        'ret_ndx_1m': ret_ndx_1m,
        'ret_ndx_3m': ret_ndx_3m,
        'pct_beat_sp_1m': pct_beat_sp_1m,
        'pct_beat_sp_3m': pct_beat_sp_3m,
        'pct_beat_ndx_1m': pct_beat_ndx_1m,
        'pct_beat_ndx_3m': pct_beat_ndx_3m,
        'ew_us_1m': ew_us_1m,
        'spread_sp_1m': spread_sp_1m,
        'spread_ndx_1m': spread_ndx_1m,
        'multi_hl': multi_hl,
    }


def slice_custom_scope_breadth_us_core(scope_name: str) -> pd.DataFrame | None:
    """Pure zero-Streamlit helper slicing S&P 500 or Nasdaq 100 breadth history from disk matrix."""
    matrix_path = 'breadth_matrix.pkl' if os.path.exists('breadth_matrix.pkl') else 'historical_prices_matrix.pkl'
    if not os.path.exists(matrix_path):
        return None
    try:
        matrix_df = pd.read_pickle(matrix_path)
        closes = matrix_df.xs('Close', level=1, axis=1).dropna(how='all') if matrix_df.columns.nlevels > 1 else matrix_df.dropna(how='all')
        highs = matrix_df.xs('High', level=1, axis=1).dropna(how='all') if matrix_df.columns.nlevels > 1 else matrix_df.dropna(how='all')
        lows = matrix_df.xs('Low', level=1, axis=1).dropna(how='all') if matrix_df.columns.nlevels > 1 else matrix_df.dropna(how='all')
    except Exception:
        return None

    try:
        from us_universe_manager import get_sp500_constituents, get_nasdaq100_constituents
        if "S&P 500" in scope_name:
            target_symbols = get_sp500_constituents()
        elif "Nasdaq 100" in scope_name:
            target_symbols = get_nasdaq100_constituents()
        else:
            return None
    except Exception:
        return None

    valid_cols = [c for c in target_symbols if c in closes.columns]
    if len(valid_cols) < 30:
        return None

    from price_history_manager import filter_panel_active_market_dates
    sub_closes = filter_panel_active_market_dates(closes[valid_cols], min_coverage_ratio=0.50, min_abs=15)
    if sub_closes.empty:
        return None
    for _pnl_s in (highs, lows):
        if getattr(_pnl_s.index, 'tz', None) is not None:
            _pnl_s.index = _pnl_s.index.tz_localize(None)
        _pnl_s.index = pd.to_datetime(_pnl_s.index).normalize()
    sub_highs = highs[~highs.index.duplicated(keep='last')][sub_closes.columns].reindex(sub_closes.index)
    sub_lows = lows[~lows.index.duplicated(keep='last')][sub_closes.columns].reindex(sub_closes.index)

    sma_20 = sub_closes.rolling(20, min_periods=15).mean()
    sma_50 = sub_closes.rolling(50, min_periods=40).mean()
    sma_200 = sub_closes.rolling(200, min_periods=120).mean()
    yh = sub_highs.rolling(250, min_periods=200).max()
    yl = sub_lows.rolling(250, min_periods=200).min()

    c_valid_counts = sub_closes.notna().sum(axis=1)
    above_20_pct = ((sub_closes > sma_20).sum(axis=1) / c_valid_counts * 100.0).fillna(0)
    above_50_pct = ((sub_closes > sma_50).sum(axis=1) / c_valid_counts * 100.0).fillna(0)
    above_200_pct = ((sub_closes > sma_200).sum(axis=1) / c_valid_counts * 100.0).fillna(0)

    new_h = (sub_closes >= (yh * 0.98)).sum(axis=1)
    new_l = (sub_closes <= (yl * 1.02)).sum(axis=1)
    net_nh = (new_h - new_l).astype(int)

    d_ret = sub_closes.pct_change()
    adv = (d_ret > 0).sum(axis=1)
    dec = (d_ret < 0).sum(axis=1)
    ad_line = (adv - dec).cumsum()

    return pd.DataFrame({
        'above_20_pct': above_20_pct,
        'above_50_pct': above_50_pct,
        'above_200_pct': above_200_pct,
        'net_new_highs': net_nh,
        'new_highs_count': new_h,
        'new_lows_count': new_l,
        'advances': adv,
        'declines': dec,
        'ad_line_cumulative': ad_line,
        'total_stocks': len(valid_cols),
    })


if __name__ == "__main__":

    print("Testing Breadth Engine...")
    result = compute_daily_breadth(force=True)
    if result:
        print(f"Breadth for {result['date']}:")
        print(f"  Total Universe: {result['total_stocks']}")
        print(f"  % Above 50 SMA: {result['above_50_pct']:.1f}%")
        print(f"  % Above 200 SMA: {result['above_200_pct']:.1f}%")
        print(f"  Net New Highs: {result['net_new_highs']} ({result['new_highs_count']} NH, {result['new_lows_count']} NL)")
    else:
        print("Failed to compute breadth.")
