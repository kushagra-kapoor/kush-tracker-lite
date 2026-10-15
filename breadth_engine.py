import os
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime
from config import DATABASE_PATH

# 250 trading days in a year
YEAR_TRADING_DAYS = 250
# If the stock is within 2% of its 52w high/low, we count it as a "New High/Low"
PROXIMITY_THRESHOLD = 0.02 

def get_connection():
    return sqlite3.connect(DATABASE_PATH)

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
        ("ad_grade_e_count", "INTEGER DEFAULT 0")
    ]
    for col_name, col_def in new_cols:
        try:
            cursor.execute(f"ALTER TABLE market_breadth_daily ADD COLUMN {col_name} {col_def}")
        except sqlite3.OperationalError:
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
    zbt_indices = []
    if 'zbt_signal' in df.columns:
        zbt_indices = df.index[df['zbt_signal'] == 1].tolist()
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
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM market_breadth_daily WHERE market=? ORDER BY date DESC LIMIT ?", conn, params=(market, days))
    conn.close()
    if not df.empty:
        df = df.sort_values('date').set_index('date')
        df = compute_zweig_thrust_and_volume_signals(df)
    return df

def fetch_full_breadth_history(market='IN', days=300):
    init_breadth_table()
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM market_breadth_daily WHERE market=? ORDER BY date ASC LIMIT ?", conn, params=(market, days))
    conn.close()
    if not df.empty:
        df = df.set_index('date')
        df = compute_zweig_thrust_and_volume_signals(df)
    return df

def sync_breadth_history(market='IN', lookback_days=1000):
    init_breadth_table()
    
    matrix_path = 'breadth_matrix.pkl'
    valid_tickers = []
    
    if market == 'IN':
        try:
            from market_data import fetch_nifty_total_market_tickers
            valid_tickers = fetch_nifty_total_market_tickers(show_progress=False)
        except Exception as e:
            print(f"[Breadth Engine] Failed to fetch tickers: {e}")
            
        if not valid_tickers and os.path.exists('tickers.txt'):
            with open('tickers.txt', 'r') as f:
                valid_tickers = [line.strip().upper() for line in f if line.strip()]
                valid_tickers = [t + '.NS' if not t.endswith('.NS') else t for t in valid_tickers]
    else:
        if os.path.exists('tickers_us.txt'):
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

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date), up_volume FROM market_breadth_daily WHERE market=?", (market,))
    last_date_row = cursor.fetchone()
    last_date_str = last_date_row[0] if last_date_row and last_date_row[0] else None
    has_up_vol = (last_date_row is not None and last_date_row[1] is not None and last_date_row[1] > 0)
    
    cursor.execute("SELECT ad_line_cumulative, cumulative_volume_ad_line FROM market_breadth_daily WHERE market=? ORDER BY date DESC LIMIT 1", (market,))
    last_ad_row = cursor.fetchone()
    current_ad_cumulative = last_ad_row[0] if last_ad_row and last_ad_row[0] is not None else 0
    current_vol_ad_cumulative = last_ad_row[1] if last_ad_row and len(last_ad_row) > 1 and last_ad_row[1] is not None else 0.0

    dates_in_matrix = closes.index.strftime('%Y-%m-%d').tolist()
    
    if not last_date_str:
        missing_dates = dates_in_matrix[-lookback_days:]
    elif not has_up_vol:
        # Re-sync recent window to backfill volume and A-E metrics
        missing_dates = dates_in_matrix[-min(len(dates_in_matrix), 180):]
    else:
        missing_dates = [d for d in dates_in_matrix if d > last_date_str]

    if not missing_dates:
        conn.close()
        return

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
    conn.close()

def get_latest_volume_and_ad_distribution(market='IN') -> dict:
    """
    Retrieves latest Exchange Volume Breadth & Universe A-E Distribution with daily delta.
    """
    init_breadth_table()
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM market_breadth_daily 
        WHERE market = ? 
        ORDER BY date DESC 
        LIMIT 2
    """, (market,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {}

    curr = dict(rows[0])
    prev = dict(rows[1]) if len(rows) > 1 else curr

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
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM market_breadth_daily WHERE market = ? ORDER BY date DESC LIMIT 1", (market,))
    row = cursor.fetchone()
    conn.close()
    
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
    if market == 'IN':
        in_cols = [c for c in closes.columns if str(c).endswith('.NS')]
        if in_cols:
            closes = closes[in_cols]
    else:
        us_cols = [c for c in closes.columns if not str(c).endswith('.NS') and not str(c).endswith('.BO')]
        if us_cols:
            closes = closes[us_cols]

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
