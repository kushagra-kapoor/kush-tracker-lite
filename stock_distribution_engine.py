"""
stock_distribution_engine.py - Institutional Stock Distribution & Liquidation Engine for Kush Tracker

Implements the 4 Pillars of Individual Stock Distribution based on the principles of:
- William J. O'Neil (How to Make Money in Stocks)
- David Ryan (U.S. Investing Champion)
- Mark Minervini (SEPA - Trade Like a Stock Market Wizard)
- Ross Haber (Former O'Neil PM - 21 SMA Double Close & Acute Cluster Rules)

Pillars:
1. Institutional Down-Volume Day (IDVD): Price down <= -1.5%, Vol > 50 SMA Vol, DCR < 0.40.
2. Churning / Stalling Day: Price -0.5% to +0.5%, Vol >= 1.25x 50 SMA Vol, DCR < 0.40.
3. 50-Day Up/Down Volume Ratio (U/D Ratio) & Volume Dominance.
4. Distribution Clustering (>= 3 sell days in 15 sessions) & Structural MA Breaches.
"""

import pandas as pd
import numpy as np


def calculate_daily_closing_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Computes Daily Closing Range (DCR):
    DCR = (Close - Low) / (High - Low)
    Bounded in [0.0, 1.0]. Upper half (>= 0.50) is accumulative, lower half (< 0.50) is distributive.
    """
    spread = (high - low).replace(0, np.nan)
    dcr = ((close - low) / spread).fillna(0.5).clip(0.0, 1.0)
    return dcr


def analyze_stock_distribution(df: pd.DataFrame, lookback_cluster: int = 15, lookback_ud: int = 50) -> dict:
    """
    Evaluates individual stock distribution across the 4 pillars.
    Requires DataFrame with columns: ['Close', 'High', 'Low', 'Volume'] or standardized lowercase.
    
    Returns structured dictionary with:
    - idvd_count_15d: int
    - churning_count_15d: int
    - total_dist_days_15d: int
    - cluster_detected: bool (>= 3 distribution sessions in 15 trading days)
    - ud_volume_ratio_50d: float (Up/Down Volume ratio over 50 sessions)
    - consecutive_closes_below_21: int
    - haber_21_breach: bool (>= 2 closes below 21 EMA)
    - oneil_50_breach: bool (Close < 50 SMA on heavy volume >= 1.2x average)
    - ids_score: int (Institutional Distribution Score 0 to 10)
    - status: str ('🟢 Clean', '🟡 Stalling', '🔴 Heavy Distribution')
    - badge_label: str (Compact glanceable text e.g. '🟢 0 DD (1.4x)')
    - prescription: str (Actionable execution instruction)
    - dist_sessions: list of dicts (details of individual sell sessions for deep drill-down)
    """
    empty_result = {
        'idvd_count_15d': 0,
        'churning_count_15d': 0,
        'total_dist_days_15d': 0,
        'cluster_detected': False,
        'ud_volume_ratio_50d': 1.0,
        'consecutive_closes_below_21': 0,
        'haber_21_breach': False,
        'oneil_50_breach': False,
        'ids_score': 0,
        'status': '🟢 Clean',
        'badge_label': '🟢 0 DD (1.0x)',
        'prescription': 'Hold core position; institutional accumulation remains constructive.',
        'dist_sessions': []
    }

    if df is None or df.empty or len(df) < 20:
        return empty_result

    # Standardize column access
    c_map = {col.lower(): col for col in df.columns}
    close_col = c_map.get('close')
    high_col = c_map.get('high')
    low_col = c_map.get('low')
    vol_col = c_map.get('volume')

    if not close_col or not vol_col:
        return empty_result

    close = df[close_col].astype(float)
    volume = df[vol_col].astype(float).replace(0, np.nan).ffill().fillna(1.0)
    high = df[high_col].astype(float) if high_col else close
    low = df[low_col].astype(float) if low_col else close

    n_bars = len(close)
    if n_bars < 20:
        return empty_result

    # Daily percentage return
    pct_change = close.pct_change() * 100.0

    # 50-day SMA Volume (min_periods=15 for newly listed/young names)
    vol_sma50 = volume.rolling(50, min_periods=min(15, n_bars)).mean().replace(0, 1.0)

    # Daily Closing Range (DCR)
    dcr = calculate_daily_closing_range(high, low, close)

    # -------------------------------------------------------------
    # Pillar 1: Institutional Down-Volume Day (IDVD)
    # Price drop <= -1.5%, Volume > 50 SMA Vol, Close in lower range DCR < 0.40
    # -------------------------------------------------------------
    is_idvd = (pct_change <= -1.5) & (volume > vol_sma50) & (dcr < 0.40)

    # -------------------------------------------------------------
    # Pillar 2: Churning / Stalling Day
    # Price stagnant (-0.5% to +0.5%), Volume >= 1.25x 50 SMA Vol, DCR < 0.40
    # Heavy turnover with zero upward progress closing near lows
    # -------------------------------------------------------------
    is_churning = (pct_change >= -0.5) & (pct_change <= 0.5) & (volume >= 1.25 * vol_sma50) & (dcr < 0.40)

    # -------------------------------------------------------------
    # Pillar 3: 50-Day Up/Down Volume Ratio (U/D Ratio)
    # -------------------------------------------------------------
    ud_window = min(n_bars, lookback_ud)
    recent_rets = pct_change.tail(ud_window)
    recent_vols = volume.tail(ud_window)

    up_volume = recent_vols[recent_rets > 0].sum()
    down_volume = recent_vols[recent_rets < 0].sum()

    if down_volume > 0:
        ud_ratio = float(round(up_volume / down_volume, 2))
    elif up_volume > 0:
        ud_ratio = 3.0
    else:
        ud_ratio = 1.0

    # -------------------------------------------------------------
    # Pillar 4: 15-Day Rolling Clustering & Moving Average Breaches
    # -------------------------------------------------------------
    cluster_window = min(n_bars, lookback_cluster)
    recent_idvd = is_idvd.tail(cluster_window)
    recent_churning = is_churning.tail(cluster_window)

    idvd_count = int(recent_idvd.sum())
    churning_count = int(recent_churning.sum())
    total_dist_days = idvd_count + churning_count
    cluster_detected = total_dist_days >= 3

    # Moving Average calculations
    ema21 = close.ewm(span=21, adjust=False).mean()
    sma50 = close.rolling(50, min_periods=min(15, n_bars)).mean()

    # Consecutive Closes below 21 EMA (Ross Haber Rule)
    closes_below_21 = 0
    for i in range(1, min(60, n_bars)):
        if close.iloc[-i] < ema21.iloc[-i]:
            closes_below_21 += 1
        else:
            break
    haber_21_breach = (closes_below_21 >= 2)

    # William O'Neil 50 SMA Breakdown Rule
    # Current close below 50 SMA on volume >= 1.2x 50 SMA volume
    curr_close = float(close.iloc[-1])
    curr_sma50 = float(sma50.iloc[-1])
    curr_vol = float(volume.iloc[-1])
    curr_vol_sma50 = float(vol_sma50.iloc[-1])

    oneil_50_breach = (curr_close < curr_sma50) and (curr_vol >= 1.20 * curr_vol_sma50)

    # -------------------------------------------------------------
    # Session Details Drill-Down (for expander / modal)
    # -------------------------------------------------------------
    dist_sessions = []
    tail_dates = df.index[-cluster_window:]
    for idx_pos, dt in enumerate(tail_dates):
        pos = -(cluster_window - idx_pos)
        dt_str = str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10]
        if bool(is_idvd.iloc[pos]):
            dist_sessions.append({
                'date': dt_str,
                'type': '🔴 Down-Volume Day (IDVD)',
                'chg_pct': float(round(pct_change.iloc[pos], 2)),
                'vol_ratio': float(round(volume.iloc[pos] / vol_sma50.iloc[pos], 2)),
                'dcr': float(round(dcr.iloc[pos], 2))
            })
        elif bool(is_churning.iloc[pos]):
            dist_sessions.append({
                'date': dt_str,
                'type': '🟡 Churning / Stalling',
                'chg_pct': float(round(pct_change.iloc[pos], 2)),
                'vol_ratio': float(round(volume.iloc[pos] / vol_sma50.iloc[pos], 2)),
                'dcr': float(round(dcr.iloc[pos], 2))
            })

    # -------------------------------------------------------------
    # Composite Institutional Distribution Score (IDS: 0 to 10)
    # -------------------------------------------------------------
    ids_score = 0
    # Points for IDVD (up to 4 points)
    ids_score += min(4, idvd_count * 2)
    # Points for Churning (up to 2 points)
    ids_score += min(2, churning_count)
    # Points for Cluster (2 points)
    if cluster_detected:
        ids_score += 2
    # Points for U/D Volume Ratio degradation
    if ud_ratio < 0.75:
        ids_score += 2
    elif ud_ratio < 0.90:
        ids_score += 1
    # Points for Moving Average breaches
    if haber_21_breach:
        ids_score += 1
    if oneil_50_breach:
        ids_score += 2

    ids_score = min(10, ids_score)

    # Status classification & Prescription
    if ids_score >= 5 or cluster_detected or oneil_50_breach:
        status = '🔴 Heavy Distribution'
        badge_label = f"🔴 {total_dist_days} DD 🚨"
        if oneil_50_breach:
            prescription = "🚨 Heavy 50 SMA volume break! Liquidate or trim 50% immediately to protect capital."
        elif haber_21_breach and cluster_detected:
            prescription = f"🚨 {total_dist_days} DD cluster + {closes_below_21} closes < 21 EMA! Ross Haber rule: Sell 1/3 to 1/2 position now."
        elif cluster_detected:
            prescription = f"🚨 Acute Cluster: {total_dist_days} distribution days in last 15 sessions! Stop new buys; tighten stop to 21 EMA."
        else:
            prescription = f"🔴 Heavy institutional liquidation (IDS {ids_score}/10, U/D {ud_ratio:.2f}x). Enforce tight stops."
    elif ids_score >= 3 or total_dist_days >= 2 or haber_21_breach:
        status = '🟡 Stalling / Early Distribution'
        badge_label = f"🟡 {total_dist_days} DD ({ud_ratio:.2f}x)"
        if haber_21_breach:
            prescription = f"⚠️ Testing 21 EMA ({closes_below_21} closes below). Watch for absorption or execute Haber defense."
        else:
            prescription = f"⚠️ {total_dist_days} distribution sessions detected in 15d. Halt pyramiding additions; raise stops to breakeven."
    else:
        status = '🟢 Clean Accumulation'
        badge_label = f"🟢 {total_dist_days} DD ({ud_ratio:.2f}x)"
        prescription = "🟢 Healthy institutional sponsorship. Normal base-building; let winners ride."

    return {
        'idvd_count_15d': idvd_count,
        'churning_count_15d': churning_count,
        'total_dist_days_15d': total_dist_days,
        'cluster_detected': cluster_detected,
        'ud_volume_ratio_50d': ud_ratio,
        'consecutive_closes_below_21': closes_below_21,
        'haber_21_breach': haber_21_breach,
        'oneil_50_breach': oneil_50_breach,
        'ids_score': ids_score,
        'status': status,
        'badge_label': badge_label,
        'prescription': prescription,
        'dist_sessions': dist_sessions
    }
