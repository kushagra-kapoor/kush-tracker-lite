"""
New Highs & Blue Sky Quantitative Engine for Kush Tracker Lite.

Calculates multi-tiered new high breakouts (Lifetime ATH, 3-Year, 2-Year, 52-Week),
Highs Frequency (counts in last 1W and 1M), custom RS Rating (0.40*1M + 0.40*3M + 0.20*6M),
Dual Momentum (Antonacci Velocity & Relative Strength), and Sortino Ratios (3M & 6M).
"""

import os
import pickle
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from risk_metrics import calculate_sortino_ratio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MATRIX_PKL_PATH = os.path.join(BASE_DIR, "historical_prices_matrix.pkl")
MAP_JSON_PATH = os.path.join(BASE_DIR, "industry_map.json")
DB_PATH = os.path.join(BASE_DIR, ".cache", "kush_tracker_lite.db")
if not os.path.exists(DB_PATH):
    DB_PATH = os.path.join(BASE_DIR, "kush_tracker.db")

def load_price_matrix(matrix_path: str = MATRIX_PKL_PATH, tickers: list = None, days: int = 252, *args, **kwargs):
    """
    Loads historical prices matrix from pickle.
    If pickle does not exist or is empty and tickers is provided,
    downloads price history on the go via price_history_manager.
    Returns close_df, high_df, low_df, volume_df.
    """
    if tickers is None and 'tickers' in kwargs:
        tickers = kwargs['tickers']
    if 'days' in kwargs:
        days = kwargs['days']
    if args:
        if len(args) >= 1 and tickers is None:
            tickers = args[0]
        if len(args) >= 2:
            days = args[1]
    matrix = None
    if os.path.exists(matrix_path):
        try:
            with open(matrix_path, "rb") as f:
                matrix = pickle.load(f)
        except Exception as e:
            print(f"Error loading {matrix_path}: {e}")
            matrix = None

    if (matrix is None or matrix.empty) and tickers:
        try:
            from price_history_manager import fetch_incremental_history
            print(f"[NewHighsEngine] Downloading price history on the go for {len(tickers)} tickers...")
            matrix = fetch_incremental_history(tickers, days=days)
        except Exception as e:
            print(f"[NewHighsEngine] Auto-download error: {e}")

    if matrix is None or matrix.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    try:
        if not isinstance(matrix.columns, pd.MultiIndex):
            return matrix, matrix, matrix, pd.DataFrame()
            
        # Safe extraction of price attributes
        def get_level_slice(target_name):
            if matrix.columns.nlevels >= 2:
                for col_type in [target_name, target_name.capitalize(), target_name.upper(), target_name.lower()]:
                    if col_type in matrix.columns.levels[1]:
                        return matrix.xs(col_type, level=1, axis=1)
                for col_type in [target_name, target_name.capitalize(), target_name.upper(), target_name.lower()]:
                    if col_type in matrix.columns.levels[0]:
                        return matrix.xs(col_type, level=0, axis=1)
            return pd.DataFrame()

        close_df = get_level_slice("Close")
        high_df = get_level_slice("High")
        low_df = get_level_slice("Low")
        volume_df = get_level_slice("Volume")
        
        if high_df.empty and not close_df.empty:
            high_df = close_df.copy()
        if low_df.empty and not close_df.empty:
            low_df = close_df.copy()
            
        return close_df, high_df, low_df, volume_df
    except Exception as e:
        print(f"Error extracting price matrix slices: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def load_industry_map(map_path: str = MAP_JSON_PATH, db_path: str = DB_PATH) -> dict:
    """
    Loads ticker to industry mapping from industry_map.json or sqlite db.
    """
    industry_map = {}
    if os.path.exists(map_path):
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                industry_map = json.load(f)
        except Exception:
            pass

    if not industry_map and os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT ticker, industry FROM fundamentals_cache WHERE industry IS NOT NULL AND industry != ''")
            for t, ind in cursor.fetchall():
                industry_map[t] = ind
            conn.close()
        except Exception:
            pass
            
    return industry_map

def compute_custom_rs_rating(close_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Computes custom user-specified Relative Strength Rating:
    Formula: RS Raw = 0.40 * Return_1M + 0.40 * Return_3M + 0.20 * Return_6M
    Percentile ranked cross-sectionally to integer 1-99.
    
    Returns:
        (rs_rating_series, rs_raw_series)
    """
    if close_df.empty or len(close_df) < 22:
        empty_s = pd.Series(50, index=close_df.columns if not close_df.empty else [])
        return empty_s, empty_s
        
    n_bars = len(close_df)
    c0 = close_df.iloc[-1]
    
    idx_1m = max(0, n_bars - 22)
    idx_3m = max(0, n_bars - 64)
    idx_6m = max(0, n_bars - 127)
    
    c_1m = close_df.iloc[idx_1m].replace(0, np.nan)
    c_3m = close_df.iloc[idx_3m].replace(0, np.nan)
    c_6m = close_df.iloc[idx_6m].replace(0, np.nan)
    
    ret_1m = ((c0 - c_1m) / c_1m) * 100.0
    ret_3m = ((c0 - c_3m) / c_3m) * 100.0
    ret_6m = ((c0 - c_6m) / c_6m) * 100.0
    
    # Custom formula: 0.40 * 1M + 0.40 * 3M + 0.20 * 6M
    rs_raw = (0.40 * ret_1m.fillna(0)) + (0.40 * ret_3m.fillna(0)) + (0.20 * ret_6m.fillna(0))
    
    # Cross-sectional percentile rank to 1-99 integer
    rs_pct = rs_raw.rank(pct=True, na_option='bottom')
    rs_rating = (rs_pct * 98.0 + 1.0).round().clip(lower=1, upper=99).fillna(50).astype(int)
    
    return rs_rating, rs_raw

def compute_dual_momentum_metrics(close_df: pd.DataFrame, rs_rating: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes Gary Antonacci & Manish Dhawan style Dual Momentum:
    - Smooth Velocity Score: 10% 1W + 40% 1M + 50% 3M
    - Dual Score: 50% RS Percentile + 50% Velocity Percentile
    - Dual Rank: Cross-sectional ranking in universe (Stage 2 prioritized)
    
    Returns:
        (velocity_score, dual_score, dual_rank)
    """
    if close_df.empty or len(close_df) < 10:
        empty_s = pd.Series(0.0, index=close_df.columns if not close_df.empty else [])
        return empty_s, empty_s, empty_s.astype(int)
        
    n_bars = len(close_df)
    c0 = close_df.iloc[-1]
    
    idx_1w = max(0, n_bars - 6)
    idx_1m = max(0, n_bars - 22)
    idx_3m = max(0, n_bars - 64)
    
    c_1w = close_df.iloc[idx_1w].replace(0, np.nan)
    c_1m = close_df.iloc[idx_1m].replace(0, np.nan)
    c_3m = close_df.iloc[idx_3m].replace(0, np.nan)
    
    roc_1w = ((c0 - c_1w) / c_1w) * 100.0
    roc_1m = ((c0 - c_1m) / c_1m) * 100.0
    roc_3m = ((c0 - c_3m) / c_3m) * 100.0
    
    # Velocity Score
    velocity_score = (0.10 * roc_1w.fillna(0)) + (0.40 * roc_1m.fillna(0)) + (0.50 * roc_3m.fillna(0))
    
    # Percentiles
    rs_pct = rs_rating.rank(pct=True, na_option='bottom') * 100.0
    vel_pct = velocity_score.rank(pct=True, na_option='bottom') * 100.0
    
    dual_score = (0.50 * rs_pct) + (0.50 * vel_pct)
    
    # 200 SMA Absolute Gate Check
    if n_bars >= 150:
        sma_200 = close_df.iloc[-200:].mean()
        above_200 = c0 >= sma_200
    else:
        above_200 = pd.Series(True, index=close_df.columns)
        
    passed_gate = above_200 & (velocity_score > 0)
    gate_score = passed_gate.astype(int) * 1000.0 + dual_score
    dual_rank = gate_score.rank(ascending=False, method='min').fillna(9999).astype(int)
    
    return velocity_score, dual_score, dual_rank

def compute_highs_frequency_and_tiers(high_df: pd.DataFrame, close_df: pd.DataFrame):
    """
    Computes:
    - 52W Prior High, 2Y Prior High, 3Y Prior High, Lifetime ATH
    - Today's Breakout Tier: Lifetime ATH, 3Y High, 2Y High, 52W High, or Coiling
    - 1W (5-day) and 1M (21-day) count of new highs printed
    - Pre-breakout distance to ATH and 52W High
    """
    if high_df.empty or len(high_df) < 5:
        return {}
        
    n_bars = len(high_df)
    filled_high = high_df.ffill()
    today_high = filled_high.iloc[-1]
    today_close = close_df.iloc[-1] if not close_df.empty else today_high
    
    # Shifted history (prior to today)
    prior_highs = filled_high.iloc[:-1]
    
    # Rolling and Cumulative Prior Highs
    prior_252_high = prior_highs.iloc[-252:].max() if len(prior_highs) >= 252 else prior_highs.max()
    prior_504_high = prior_highs.iloc[-504:].max() if len(prior_highs) >= 504 else pd.Series(np.nan, index=high_df.columns)
    prior_756_high = prior_highs.iloc[-756:].max() if len(prior_highs) >= 756 else pd.Series(np.nan, index=high_df.columns)
    prior_ath = prior_highs.cummax().iloc[-1] if not prior_highs.empty else today_high
    
    # Check Breakouts Today
    is_ath_today = (today_high >= prior_ath) & (prior_ath > 0)
    is_3y_today = (today_high >= prior_756_high) & (prior_756_high > 0)
    is_2y_today = (today_high >= prior_504_high) & (prior_504_high > 0)
    is_52w_today = (today_high >= prior_252_high) & (prior_252_high > 0)
    
    # Tier classification
    tiers = pd.Series("Normal", index=high_df.columns)
    tiers[is_52w_today] = "🌟 52W High"
    tiers[is_2y_today] = "🔷 2-Year High"
    tiers[is_3y_today] = "💎 3-Year High"
    tiers[is_ath_today] = "🚀 Lifetime ATH"
    
    # Frequency: Count of new highs in last 1W (5 trading days) & 1M (21 trading days)
    # Rolling 252-day peak comparison day-by-day
    rolling_252 = filled_high.shift(1).rolling(252, min_periods=40).max()
    is_52w_matrix = filled_high >= rolling_252
    
    # ATH Frequency (specific to ATH prints)
    rolling_ath_matrix = filled_high.shift(1).cummax()
    is_ath_matrix = filled_high >= rolling_ath_matrix
    
    # Combined: ANY new high (either 52W High OR All-Time High)
    is_any_high_matrix = is_52w_matrix | is_ath_matrix
    
    count_1w = is_any_high_matrix.iloc[-5:].sum().astype(int)
    count_1m = is_any_high_matrix.iloc[-21:].sum().astype(int)
    
    ath_count_1w = is_ath_matrix.iloc[-5:].sum().astype(int)
    ath_count_1m = is_ath_matrix.iloc[-21:].sum().astype(int)

    # Days since most recent High (checked across last 21 trading sessions for 1M horizon)
    sub_52w = is_52w_matrix.iloc[-21:][::-1]
    has_52w = sub_52w.any(axis=0)
    days_since_52w = pd.Series(999, index=high_df.columns)
    days_since_52w[has_52w] = sub_52w.values.argmax(axis=0)[has_52w]

    sub_ath = is_ath_matrix.iloc[-21:][::-1]
    has_ath = sub_ath.any(axis=0)
    days_since_ath = pd.Series(999, index=high_df.columns)
    days_since_ath[has_ath] = sub_ath.values.argmax(axis=0)[has_ath]

    sub_any = is_any_high_matrix.iloc[-21:][::-1]
    has_any = sub_any.any(axis=0)
    days_since_high = pd.Series(999, index=high_df.columns)
    days_since_high[has_any] = sub_any.values.argmax(axis=0)[has_any]
    
    # Distance to ATH and 52W High (%)
    dist_ath_pct = ((prior_ath - today_close) / prior_ath * 100.0).clip(lower=0.0)
    dist_52w_pct = ((prior_252_high - today_close) / prior_252_high * 100.0).clip(lower=0.0)
    
    # Coiling within 5% of ATH (not broken out today)
    is_coiling = (~is_ath_today) & (dist_ath_pct <= 5.0) & (dist_ath_pct >= 0.0)
    
    # Tier classification (Reflects both today and recent breakout leadership)
    tiers = pd.Series("Normal", index=high_df.columns)
    
    # 1-Month recent leaders
    is_recent_52w = (days_since_52w <= 20)
    is_recent_ath = (days_since_ath <= 20)
    
    tiers[is_recent_52w] = "🌟 52W High (Recent)"
    tiers[is_recent_ath] = "🚀 Lifetime ATH (Recent)"
    
    # Today's live breakouts
    tiers[is_52w_today] = "🌟 52W High"
    tiers[is_2y_today] = "🔷 2-Year High"
    tiers[is_3y_today] = "💎 3-Year High"
    tiers[is_ath_today] = "🚀 Lifetime ATH"
    
    tiers[is_coiling & (tiers == "Normal")] = "🔭 Coiling Near ATH"
    
    return {
        'today_high': today_high,
        'today_close': today_close,
        'prior_ath': prior_ath,
        'prior_52w': prior_252_high,
        'prior_2y': prior_504_high,
        'prior_3y': prior_756_high,
        'is_ath_today': is_ath_today,
        'is_3y_today': is_3y_today,
        'is_2y_today': is_2y_today,
        'is_52w_today': is_52w_today,
        'tier': tiers,
        'count_1w': count_1w,
        'count_1m': count_1m,
        'ath_count_1w': ath_count_1w,
        'ath_count_1m': ath_count_1m,
        'days_since_high': days_since_high,
        'days_since_ath': days_since_ath,
        'days_since_52w': days_since_52w,
        'dist_ath_pct': dist_ath_pct,
        'dist_52w_pct': dist_52w_pct,
        'is_coiling': is_coiling
    }

def compute_new_highs_universe(
    close_df: pd.DataFrame = None,
    high_df: pd.DataFrame = None,
    low_df: pd.DataFrame = None,
    volume_df: pd.DataFrame = None,
    industry_map: dict = None,
    tickers: list = None,
    min_price: float = 10.0,
    min_volume_expansion: float = 0.0,
    compute_sortino: bool = True
) -> pd.DataFrame:
    """
    Orchestrates full universe new highs screening with RS, Dual Momentum,
    Sortino 3M/6M, and Frequency counts.
    """
    if close_df is None or high_df is None:
        close_df, high_df, low_df, volume_df = load_price_matrix()
        
    if industry_map is None:
        industry_map = load_industry_map()
        
    if close_df.empty or high_df.empty:
        return pd.DataFrame()
        
    # Filter tickers if specified
    if tickers:
        valid_tickers = [t for t in tickers if t in close_df.columns]
        if valid_tickers:
            close_df = close_df[valid_tickers]
            high_df = high_df[[t for t in valid_tickers if t in high_df.columns]]
            if not low_df.empty:
                low_df = low_df[[t for t in valid_tickers if t in low_df.columns]]
            if not volume_df.empty:
                volume_df = volume_df[[t for t in valid_tickers if t in volume_df.columns]]

    # Filter out newly listed tickers with < 40 trading days of history
    valid_hist_counts = high_df.notna().sum()
    mature_tickers = valid_hist_counts[valid_hist_counts >= 40].index.tolist()
    if mature_tickers:
        close_df = close_df[mature_tickers]
        high_df = high_df[mature_tickers]
        if not low_df.empty:
            low_df = low_df[[t for t in mature_tickers if t in low_df.columns]]
        if not volume_df.empty:
            volume_df = volume_df[[t for t in mature_tickers if t in volume_df.columns]]

    n_bars = len(close_df)
    if n_bars < 10:
        return pd.DataFrame()
        
    # 1. Highs and Tiers
    high_stats = compute_highs_frequency_and_tiers(high_df, close_df)
    if not high_stats:
        return pd.DataFrame()
        
    # 2. Custom RS Rating (0.40 * 1M + 0.40 * 3M + 0.20 * 6M)
    rs_rating, rs_raw = compute_custom_rs_rating(close_df)
    
    # 3. Dual Momentum Metrics
    velocity_score, dual_score, dual_rank = compute_dual_momentum_metrics(close_df, rs_rating)
    
    # 4. Intraday & Execution Metrics
    c0 = close_df.iloc[-1]
    c_prev = close_df.iloc[-2] if n_bars >= 2 else c0
    today_pct = ((c0 - c_prev) / c_prev * 100.0).fillna(0.0)
    
    h0 = high_df.iloc[-1]
    l0 = low_df.iloc[-1] if not low_df.empty else c0
    daily_range = (h0 - l0).replace(0, np.nan)
    close_range_pct = (((c0 - l0) / daily_range) * 100.0).fillna(100.0).clip(0.0, 100.0)
    
    # Volume Expansion vs 20D Average
    if volume_df is not None and not volume_df.empty and len(volume_df) >= 21:
        v0 = volume_df.iloc[-1]
        v_20d = volume_df.iloc[-21:-1].mean().replace(0, np.nan)
        vol_expansion = (v0 / v_20d).fillna(1.0)
        dollar_vol_cr = ((v_20d.fillna(0) * c0) / 10000000.0).fillna(0.0)
    else:
        vol_expansion = pd.Series(1.0, index=close_df.columns)
        dollar_vol_cr = pd.Series(5.0, index=close_df.columns)
        
    # Distance to MAs
    if n_bars >= 21:
        ema_21 = close_df.ewm(span=21, adjust=False).mean().iloc[-1]
        dist_21ema = ((c0 - ema_21) / ema_21 * 100.0).fillna(0.0)
    else:
        dist_21ema = pd.Series(0.0, index=close_df.columns)
        
    if n_bars >= 50:
        sma_50 = close_df.iloc[-50:].mean()
        dist_50sma = ((c0 - sma_50) / sma_50 * 100.0).fillna(0.0)
    else:
        dist_50sma = pd.Series(0.0, index=close_df.columns)
        
    # Setup Tag for recent high leaders (CANSLIM low-risk pullback entries)
    def assign_setup_tag(d_since, d21, d50):
        if d_since == 0:
            return "⚡ Breaking Out Today"
        elif d21 > 7.0:
            return "⏳ Extended (>7%)"
        elif 0.0 <= d21 <= 3.5 and d50 >= 0:
            return "🎯 Near 21 EMA (Buy Zone)"
        elif -2.5 <= d21 < 0.0 and d50 >= 0:
            return "🛡️ 21 EMA Shakeout"
        elif 0.0 <= d50 <= 3.5:
            return "🛡️ Near 50 SMA"
        elif d50 < 0:
            return "⚠️ Below 50 SMA"
        else:
            return "✨ Consolidating"

    # Build Universe DataFrame
    records = []
    for t in close_df.columns:
        price = float(c0.get(t, 0.0))
        if price < min_price or pd.isna(price):
            continue
            
        tier = high_stats['tier'].get(t, "Normal")
        is_ath = bool(high_stats['is_ath_today'].get(t, False))
        is_52w = bool(high_stats['is_52w_today'].get(t, False))
        is_coil = bool(high_stats['is_coiling'].get(t, False))
        
        c_1w = int(high_stats['count_1w'].get(t, 0))
        c_1m = int(high_stats['count_1m'].get(t, 0))
        
        d_val = int(high_stats['days_since_high'].get(t, 999))
        d_ath = int(high_stats['days_since_ath'].get(t, 999))
        d_52w = int(high_stats['days_since_52w'].get(t, 999))
        
        d21_val = float(dist_21ema.get(t, 0.0))
        d50_val = float(dist_50sma.get(t, 0.0))
        setup_tag = assign_setup_tag(d_val, d21_val, d50_val)
        
        if d_val == 0:
            days_text = "⚡ Today"
            horizon_cat = "Today"
        elif d_val == 1:
            days_text = "1D ago"
            horizon_cat = "1-Week"
        elif d_val <= 5:
            days_text = f"{d_val}D ago"
            horizon_cat = "1-Week"
        elif d_val <= 21:
            days_text = f"{d_val}D ago"
            horizon_cat = "1-Month"
        else:
            days_text = "> 1M"
            horizon_cat = "Older"

        # Retain stock in universe if:
        # 1. Made new high today or in last 1 month (d_val <= 21)
        # 2. Or is coiling near ATH (is_coil)
        is_recent_leader = (d_val <= 21) or is_ath or is_52w or is_coil
        
        # Sortino computation (on-demand or precomputed)
        sortino_3m = 0.0
        sortino_6m = 0.0
        if compute_sortino and is_recent_leader:
            s_close = close_df[t].dropna()
            if len(s_close) >= 63:
                sortino_3m = float(calculate_sortino_ratio(s_close, window=63, mar_annual=0.065))
            if len(s_close) >= 126:
                sortino_6m = float(calculate_sortino_ratio(s_close, window=126, mar_annual=0.065))

        clean_t = str(t).replace('.NS', '').replace('.BO', '')
        ind = industry_map.get(t, 'Unknown')
        if ind == 'Unknown':
            ind = industry_map.get(clean_t, 'Unknown')

        records.append({
            'Ticker': clean_t,
            'FullTicker': t,
            'TradingView': f"https://in.tradingview.com/chart/?symbol=NSE:{clean_t}",
            'Industry': ind,
            'Price': price,
            'Today %': float(today_pct.get(t, 0.0)),
            'Tier': tier,
            'Is ATH': is_ath,
            'Is 52W High': is_52w,
            'Is Coiling': is_coil,
            'Days Since High': d_val,
            'Days Since ATH': d_ath,
            'Days Since 52W': d_52w,
            'Days Since High Text': days_text,
            'Horizon': horizon_cat,
            'Setup Tag': setup_tag,
            'Dist ATH %': float(high_stats['dist_ath_pct'].get(t, 0.0)),
            'Dist 52W %': float(high_stats['dist_52w_pct'].get(t, 0.0)),
            'Highs 1W': c_1w,
            'Highs 1M': c_1m,
            'ATH 1W': int(high_stats['ath_count_1w'].get(t, 0)),
            'ATH 1M': int(high_stats['ath_count_1m'].get(t, 0)),
            'RS Rating': int(rs_rating.get(t, 50)),
            'RS Raw': float(rs_raw.get(t, 0.0)),
            'Velocity Score': float(velocity_score.get(t, 0.0)),
            'Dual Score': float(dual_score.get(t, 50.0)),
            'Dual Rank': int(dual_rank.get(t, 9999)),
            'Sortino 3M': sortino_3m,
            'Sortino 6M': sortino_6m,
            'Vol Exp': float(vol_expansion.get(t, 1.0)),
            'Close Range %': float(close_range_pct.get(t, 100.0)),
            'Dist 21EMA %': d21_val,
            'Dist 50SMA %': d50_val,
            'Turnover Cr': float(dollar_vol_cr.get(t, 0.0)),
            'ADTV (Cr)': round(float(dollar_vol_cr.get(t, 0.0)), 2)
        })
        
    df = pd.DataFrame(records)
    if not df.empty:
        # Default sort: ATHs first, then 52W Highs, then by Dual Score descending
        tier_order = {
            "🚀 Lifetime ATH": 0,
            "💎 3-Year High": 1,
            "🔷 2-Year High": 2,
            "🌟 52W High": 3,
            "🔭 Coiling Near ATH": 4,
            "Normal": 5
        }
        df['Tier_Order'] = df['Tier'].map(tier_order).fillna(6)
        df = df.sort_values(['Tier_Order', 'Dual Score'], ascending=[True, False]).reset_index(drop=True)
        df.drop(columns=['Tier_Order'], inplace=True)
        
    return df

def get_industry_clustering_stats(highs_df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """
    Computes industry group concentration for stocks printing new highs today or in the last 1W.
    """
    if highs_df.empty or 'Industry' not in highs_df.columns:
        return pd.DataFrame()
        
    # Consider active breakouts or frequent new high leaders
    valid_mask = (highs_df['Tier'].str.contains("High|ATH", na=False)) | (highs_df['Highs 1W'] >= 1)
    sub_df = highs_df[valid_mask]
    
    if sub_df.empty:
        return pd.DataFrame()
        
    grouped = sub_df.groupby('Industry').agg(
        Total_Highs=('Ticker', 'count'),
        Lifetime_ATH_Count=('Is ATH', 'sum'),
        Avg_RS=('RS Rating', 'mean'),
        Avg_Sortino_3M=('Sortino 3M', 'mean'),
        Leaders=('Ticker', lambda s: ", ".join(list(s)[:3]))
    ).reset_index()
    
    grouped = grouped[grouped['Industry'] != 'Unknown'].sort_values('Total_Highs', ascending=False).head(top_n).reset_index(drop=True)
    return grouped
