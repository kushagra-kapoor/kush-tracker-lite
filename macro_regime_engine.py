import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st

def standardize_columns(df: pd.DataFrame):
    """
    Standardizes dataframe columns to lowercase and handles MultiIndex flattening.
    """
    if df.empty:
        return df
    
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # Find the level that contains OHLCV data
        found_level = -1
        for i in range(df.columns.nlevels):
            level_vals = [str(c).lower() for c in df.columns.get_level_values(i)]
            if 'close' in level_vals:
                found_level = i
                break
        df.columns = df.columns.get_level_values(found_level)
    
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df

def calculate_distribution_days(df: pd.DataFrame, window: int = 25, return_details: bool = False):
    """
    Count distribution days in the specified window using William O'Neil & Ross Haber rules:
    
    1. Daily Closing Range (DCR) / Midpoint Rule:
       DCR = (Close - Low) / (High - Low)
       - Upper half (DCR >= 0.50): Accumulative
       - Lower half (DCR < 0.50): Distributive
       
    2. Distribution Day Types:
       - Standard Distribution Day: Close <= -0.2% on higher volume than prior session AND DCR < 0.50.
       - Churning / Stalling Day: Close up slightly (0.0% to +0.4%) or flat on volume > prior day, 
         BUT DCR < 0.50 (heavy volume with little upside progress). Counts as a Distribution Day.
       - Institutional Hammer (Accumulation): Close <= -0.2% on higher volume, but DCR >= 0.50 
         (sellers pressed down, institutional buyers absorbed supply). EXCLUDED from distribution!
         
    3. Expiration / Deactivation Rules:
       - Natural expiration after 25 trading sessions.
       - 5% Rallied rule: If the index closes >= 5% above a DD's close, the DD is retired early.
       
    4. Distribution Day Cluster:
       - Defined by Ross Haber: 3+ distribution days (including churning) within 5 to 7 trading days.
       - Acute warning signal of an impending top or severe correction.
       
    Returns:
        (dd_count, dd_dates_list) if return_details is False (100% backwards-compatible).
        (dd_count, dd_dates_list, details_dict) if return_details is True.
    """
    if df.empty or len(df) < window + 1:
        empty_details = {
            'total_active_dds': 0, 'standard_dd_count': 0, 'churning_dd_count': 0,
            'hammer_acc_count': 0, 'cluster_detected': False, 'cluster_count_7d': 0,
            'cluster_warning': "Normal", 'latest_dcr': 0.5, 'latest_midpoint_verdict': "Neutral",
            'active_dd_dates': [], 'dd_ledger': []
        }
        return (0, [], empty_details) if return_details else (0, [])
    
    # Standardize
    df = standardize_columns(df).copy()
    
    # Safely find close and volume
    if 'close' not in df.columns or 'volume' not in df.columns:
        empty_details = {
            'total_active_dds': 0, 'standard_dd_count': 0, 'churning_dd_count': 0,
            'hammer_acc_count': 0, 'cluster_detected': False, 'cluster_count_7d': 0,
            'cluster_warning': "Normal", 'latest_dcr': 0.5, 'latest_midpoint_verdict': "Neutral",
            'active_dd_dates': [], 'dd_ledger': []
        }
        return (0, [], empty_details) if return_details else (0, [])
    
    # Calculate price change
    df['pct_change'] = df['close'].pct_change() * 100.0
    df['vol_up'] = df['volume'] > df['volume'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    
    # Daily Closing Range (DCR)
    if 'high' in df.columns and 'low' in df.columns:
        hl_spread = (df['high'] - df['low']).replace(0, np.nan)
        df['dcr'] = ((df['close'] - df['low']) / hl_spread).fillna(0.5).clip(0.0, 1.0)
        # Check if intraday high was strong (>= +0.50% above prev close)
        df['intraday_high_pct'] = ((df['high'] - df['prev_close']) / df['prev_close']) * 100.0
    else:
        df['dcr'] = np.where(df['pct_change'] >= 0, 0.75, 0.25)
        df['intraday_high_pct'] = df['pct_change']
        
    # Standard Down-Volume DD: down >= 0.2%, higher volume, closed in lower half
    df['is_standard_dd'] = (df['pct_change'] <= -0.2) & df['vol_up'] & (df['dcr'] < 0.50)
    
    # Churning / Stalling Day (William O'Neil HTMMIS p. 195):
    # Case A: Slight gain or flat (-0.05% to +0.40%) on volume > prior session, closed in lower half
    # Case B: Strong intraday push (high >= +0.5%) that collapsed back to flat/slight red (-0.2% to +0.5%) on volume > prior session with DCR < 0.50
    stalling_case_a = (df['pct_change'] >= -0.05) & (df['pct_change'] <= 0.40) & df['vol_up'] & (df['dcr'] < 0.50)
    stalling_case_b = (df['intraday_high_pct'] >= 0.50) & (df['pct_change'] >= -0.20) & (df['pct_change'] <= 0.50) & df['vol_up'] & (df['dcr'] < 0.50)
    df['is_churning_dd'] = (stalling_case_a | stalling_case_b) & (~df['is_standard_dd'])
    
    # Institutional Support Hammer: down <= -0.2%, higher volume, closed in upper half (NOT DD - absorbed)
    df['is_hammer_acc'] = (df['pct_change'] <= -0.2) & df['vol_up'] & (df['dcr'] >= 0.50)
    
    df['is_dd'] = df['is_standard_dd'] | df['is_churning_dd']
    
    # Build complete Distribution & Stall Ledger
    ledger_window = min(len(df), window + 25)
    ledger_subset = df.tail(ledger_window)
    dd_ledger = []
    
    # Look at rolling 25 trading sessions for active calculation
    recent = df.tail(window)
    active_dd_indices = []
    
    candidate_indices = ledger_subset[ledger_subset['is_dd'] | ledger_subset['is_hammer_acc']].index
    for idx_pos in candidate_indices:
        row = df.loc[idx_pos]
        dd_close = float(row['close'])
        pct_chg = float(row['pct_change'])
        subsequent_closes = df.loc[idx_pos:]['close']
        rallied_5pct = bool((subsequent_closes >= dd_close * 1.05).any())
        
        pos_in_df = df.index.get_loc(idx_pos)
        sessions_ago = (len(df) - 1) - pos_in_df
        
        is_hammer = bool(row['is_hammer_acc'])
        is_churning = bool(row['is_churning_dd'])
        
        if is_hammer:
            status = "Absorbed (Support Hammer)"
            dd_type = "Support Hammer"
            is_active = False
        elif sessions_ago >= window:
            status = "Expired (25D Elapsed)"
            dd_type = "Stalling / Churning" if is_churning else "Standard Down-Vol"
            is_active = False
        elif rallied_5pct:
            status = "Expired (5% Rally Rule)"
            dd_type = "Stalling / Churning" if is_churning else "Standard Down-Vol"
            is_active = False
        else:
            status = "Active"
            dd_type = "Stalling / Churning" if is_churning else "Standard Down-Vol"
            is_active = True
            active_dd_indices.append(idx_pos)
            
        if 'date' in df.columns:
            date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
        else:
            date_str = idx_pos.strftime('%Y-%m-%d') if hasattr(idx_pos, 'strftime') else str(idx_pos)
            
        prev_vol = df['volume'].iloc[pos_in_df - 1] if pos_in_df > 0 else row['volume']
        vol_ratio = (float(row['volume']) / float(prev_vol)) if prev_vol > 0 else 1.0
        
        dd_ledger.append({
            'date': date_str,
            'type': dd_type,
            'change_pct': round(pct_chg, 2),
            'volume_ratio': round(vol_ratio, 2),
            'dcr_pct': round(float(row['dcr']) * 100.0, 1),
            'status': status,
            'sessions_ago': int(sessions_ago),
            'close': round(dd_close, 2)
        })
        
    active_recent = recent.loc[recent.index.isin(active_dd_indices)]
    
    if 'date' in df.columns:
        dd_dates_list = pd.to_datetime(active_recent['date']).dt.strftime('%Y-%m-%d').tolist()
    else:
        dd_dates_list = active_recent.index.strftime('%Y-%m-%d').tolist() if hasattr(active_recent.index, 'strftime') else [str(x) for x in active_recent.index]
        
    # Ross Haber Cluster Check: 3+ DDs in the last 7 trading days
    tail_7 = df.tail(7)
    recent_7_dds = [idx for idx in active_dd_indices if idx in tail_7.index]
    cluster_detected = len(recent_7_dds) >= 3
    
    latest_dcr = float(df['dcr'].iloc[-1]) if not df.empty else 0.5
    latest_midpoint_verdict = "Accumulative (Upper Half)" if latest_dcr >= 0.50 else "Distributive (Lower Half)"
    
    details = {
        'total_active_dds': len(active_dd_indices),
        'standard_dd_count': int(active_recent['is_standard_dd'].sum()),
        'churning_dd_count': int(active_recent['is_churning_dd'].sum()),
        'hammer_acc_count': int(recent['is_hammer_acc'].sum()),
        'cluster_detected': cluster_detected,
        'cluster_count_7d': len(recent_7_dds),
        'cluster_warning': "🚨 SEVERE CLUSTER: 3+ Distribution Days in last 7 sessions" if cluster_detected else "Normal",
        'latest_dcr': round(latest_dcr, 3),
        'latest_midpoint_verdict': latest_midpoint_verdict,
        'active_dd_dates': dd_dates_list,
        'dd_ledger': dd_ledger
    }
    
    if return_details:
        return len(dd_dates_list), dd_dates_list, details
    return len(dd_dates_list), dd_dates_list

def _format_bar_date(idx_val, row_date=None) -> str:
    """Formats a bar timestamp/string consistently as YYYY-MM-DD."""
    raw = row_date if row_date is not None and pd.notna(row_date) else idx_val
    try:
        return pd.to_datetime(raw).strftime('%Y-%m-%d')
    except Exception:
        return str(raw)[:10]


def detect_follow_through_day(df: pd.DataFrame, min_gain: float = 1.5, haber_mode: bool = False, return_details: bool = False):
    """
    Detects O'Neil & Ross Haber Follow-Through Days (FTD) using causal Swing-Low -> Day 1 -> Day 4+ state tracking.

    Rules enforced across a 90-session trailing window:
    - A new correction / rally-attempt cycle begins only on a meaningful pullback (>= 3.0% off 25D high or
      < 21-EMA / < 50-SMA before an active FTD; once an FTD wave is active, routine shallow bull dips < 3.8%
      above key averages do NOT reset the wave — only rally_low undercuts or >= 3.8% + < 50-SMA / 2x < 21-EMA).
    - Day 1 starts on the first up-close (or upper-half bottom reversal DCR >= 50%) off the correction low.
    - Days 2 & 3 bounces are blocked from confirming an FTD.
    - Day 4+: Index gains >= 1.5% (O'Neil) or >= 1.75% (Haber strict) on volume > prior day triggers FTD.
    - Tracks all Primary & Secondary FTD dates in the active wave (`recent_ftd_events`).
    """
    empty_details = {
        'ftd_detected': False,
        'ftd_date': None,
        'latest_ftd_date': None,
        'ftd_gain_pct': 0.0,
        'ftd_vol_ratio': 1.0,
        'ftd_sessions_ago': -1,
        'rally_attempt_start': None,
        'rally_day': 0,
        'rally_low': 0.0,
        'rally_low_undercut': False,
        'status': "No Active FTD",
        'recent_ftd_events': [],
    }
    if df is None or df.empty or len(df) < 20:
        return (False, None, empty_details) if return_details else (False, None)

    recent = standardize_columns(df.tail(90)).copy()
    if 'close' not in recent.columns or 'volume' not in recent.columns:
        return (False, None, empty_details) if return_details else (False, None)

    close_s = pd.to_numeric(recent['close'], errors='coerce')
    vol_s = pd.to_numeric(recent['volume'], errors='coerce').fillna(0.0)
    low_s = pd.to_numeric(recent['low'], errors='coerce') if 'low' in recent.columns else close_s
    high_s = pd.to_numeric(recent['high'], errors='coerce') if 'high' in recent.columns else close_s
    open_s = pd.to_numeric(recent['open'], errors='coerce') if 'open' in recent.columns else close_s

    valid_mask = close_s.notna()
    if valid_mask.sum() < 20:
        return (False, None, empty_details) if return_details else (False, None)

    pct_s = close_s.pct_change(fill_method=None) * 100.0
    ema21_s = close_s.ewm(span=21, adjust=False).mean()
    sma50_s = close_s.rolling(50, min_periods=20).mean()
    roll_high_25 = high_s.rolling(25, min_periods=5).max()
    hl_rng = (high_s - low_s).replace(0, np.nan)
    dcr_s = ((close_s - low_s) / hl_rng).fillna(0.5).clip(0.0, 1.0)
    threshold = 1.75 if haber_mode else float(min_gain)
    has_date_col = 'date' in recent.columns

    n_bars = len(recent)
    in_correction = False
    attempt_active = False
    rally_low = None
    rally_start_dt = None
    rally_day = 0
    active_ftd = None
    post_ftd_high = 0.0
    wave_events = []
    last_failed_event = None
    consec_below_21 = 0

    for i in range(1, n_bars):
        c_val = float(close_s.iloc[i])
        prev_c = float(close_s.iloc[i - 1])
        h_val = float(high_s.iloc[i]) if pd.notna(high_s.iloc[i]) else c_val
        l_val = float(low_s.iloc[i]) if pd.notna(low_s.iloc[i]) else c_val
        prev_l = float(low_s.iloc[i - 1]) if pd.notna(low_s.iloc[i - 1]) else prev_c
        o_val = float(open_s.iloc[i]) if pd.notna(open_s.iloc[i]) else c_val
        v_val = float(vol_s.iloc[i])
        prev_v = float(vol_s.iloc[i - 1])
        pct_val = float(pct_s.iloc[i]) if pd.notna(pct_s.iloc[i]) else 0.0
        dcr_val = float(dcr_s.iloc[i])
        rh25 = float(roll_high_25.iloc[i]) if pd.notna(roll_high_25.iloc[i]) else c_val
        dd_pct = ((rh25 - min(c_val, l_val)) / rh25) * 100.0 if rh25 > 0 else 0.0
        e21 = float(ema21_s.iloc[i]) if pd.notna(ema21_s.iloc[i]) else c_val
        s50 = float(sma50_s.iloc[i]) if pd.notna(sma50_s.iloc[i]) else c_val
        dt_str = _format_bar_date(recent.index[i], recent['date'].iloc[i] if has_date_col else None)

        consec_below_21 = (consec_below_21 + 1) if c_val < e21 else 0
        below_50 = c_val < s50

        # 1. If a confirmed FTD wave is active, measure drawdowns strictly from post-FTD peaks
        #    so pre-correction highs never false-reset a fresh FTD.
        if active_ftd is not None:
            wave_dd_pct = ((post_ftd_high - min(c_val, l_val)) / post_ftd_high) * 100.0 if post_ftd_high > 0 else 0.0
            if l_val < active_ftd['rally_low']:
                last_failed_event = {
                    'ftd_date': active_ftd['date'],
                    'ftd_gain_pct': active_ftd['gain_pct'],
                    'ftd_vol_ratio': active_ftd['vol_ratio'],
                    'rally_day': active_ftd['rally_day'],
                    'rally_low': active_ftd['rally_low'],
                    'rally_attempt_start': rally_start_dt,
                }
                active_ftd = None
                post_ftd_high = 0.0
                wave_events = []
                in_correction = True
                attempt_active = False
                rally_day = 0
                rally_low = None
                rally_start_dt = None
            elif wave_dd_pct >= 3.8 and (below_50 or consec_below_21 >= 2):
                active_ftd = None
                post_ftd_high = 0.0
                wave_events = []
                in_correction = True
                attempt_active = False
                rally_day = 0
                rally_low = None
                rally_start_dt = None
            else:
                post_ftd_high = max(post_ftd_high, h_val)

        # 2. Detect correction phase before an active FTD confirms
        if active_ftd is None and not in_correction:
            if dd_pct >= 3.0 or c_val < e21 or below_50:
                in_correction = True

        if not in_correction and active_ftd is None:
            continue

        # 3. Evaluate rally attempt progression & FTD confirmation
        is_up_day = c_val > prev_c
        is_hammer_reversal = (l_val < prev_l) and (dcr_val >= 0.50) and (c_val >= o_val or pct_val >= -0.15)
        is_higher_vol = (prev_v > 0) and (v_val > prev_v) and (v_val > 0)
        vol_ratio = round(v_val / prev_v, 2) if prev_v > 0 else 1.0

        if active_ftd is None:
            if not attempt_active:
                if is_up_day or is_hammer_reversal:
                    attempt_active = True
                    look_lo = float(low_s.iloc[max(0, i - 2):i + 1].min())
                    rally_low = look_lo
                    rally_start_dt = dt_str
                    rally_day = 1
            else:
                if l_val < rally_low:
                    # Rally attempt failed via undercut; reset immediately if bar itself is a reversal
                    if is_up_day or (dcr_val >= 0.50 and c_val >= o_val):
                        rally_low = l_val
                        rally_start_dt = dt_str
                        rally_day = 1
                    else:
                        attempt_active = False
                        rally_day = 0
                        rally_low = None
                        rally_start_dt = None
                else:
                    rally_day += 1
                    if rally_day >= 4 and pct_val >= threshold and is_higher_vol:
                        ev = {
                            'date': dt_str,
                            'gain_pct': round(pct_val, 2),
                            'vol_ratio': vol_ratio,
                            'rally_day': int(rally_day),
                            'rally_low': round(float(rally_low), 2),
                            'role': 'Primary',
                            'is_active': True,
                            'bar_idx': i,
                        }
                        active_ftd = ev
                        post_ftd_high = max(h_val, c_val)
                        wave_events = [ev]
        else:
            # Record Secondary FTD thrust bars during the first 25 days of the confirmed wave
            rally_day += 1
            if rally_day <= 25 and pct_val >= threshold and is_higher_vol:
                wave_events.append({
                    'date': dt_str,
                    'gain_pct': round(pct_val, 2),
                    'vol_ratio': vol_ratio,
                    'rally_day': int(rally_day),
                    'rally_low': round(float(active_ftd['rally_low']), 2),
                    'role': 'Secondary',
                    'is_active': True,
                    'bar_idx': i,
                })

    if active_ftd is not None:
        clean_events = [{k: v for k, v in ev.items() if k != 'bar_idx'} for ev in wave_events]
        sessions_ago = int((n_bars - 1) - active_ftd['bar_idx'])
        ftd_details = {
            'ftd_detected': True,
            'ftd_date': active_ftd['date'],
            'latest_ftd_date': clean_events[-1]['date'],
            'ftd_gain_pct': active_ftd['gain_pct'],
            'ftd_vol_ratio': active_ftd['vol_ratio'],
            'ftd_sessions_ago': sessions_ago,
            'rally_attempt_start': rally_start_dt,
            'rally_day': active_ftd['rally_day'],
            'rally_low': active_ftd['rally_low'],
            'rally_low_undercut': False,
            'status': "Confirmed Uptrend (FTD Active)",
            'recent_ftd_events': clean_events,
        }
        return (True, active_ftd['date'], ftd_details) if return_details else (True, active_ftd['date'])

    if last_failed_event is not None:
        failed_details = {
            **empty_details,
            'ftd_date': last_failed_event['ftd_date'],
            'ftd_gain_pct': last_failed_event['ftd_gain_pct'],
            'ftd_vol_ratio': last_failed_event['ftd_vol_ratio'],
            'rally_attempt_start': last_failed_event['rally_attempt_start'],
            'rally_day': rally_day if attempt_active else last_failed_event['rally_day'],
            'rally_low': round(float(rally_low if attempt_active and rally_low is not None else last_failed_event['rally_low']), 2),
            'rally_low_undercut': True,
            'status': "Failed (Rally Low Undercut)",
        }
        return (False, None, failed_details) if return_details else (False, None)

    if attempt_active and rally_low is not None:
        watching_details = {
            **empty_details,
            'rally_attempt_start': rally_start_dt,
            'rally_day': int(rally_day),
            'rally_low': round(float(rally_low), 2),
        }
        return (False, None, watching_details) if return_details else (False, None)

    return (False, None, empty_details) if return_details else (False, None)

def detect_20ema_recapture_and_sequence(df: pd.DataFrame) -> dict:
    """
    Paul Tudor Jones (Market Wizards p. 120) & Shake Pryzby (Pinpoint Trading p. 46):
    
    1. Secular Macro Regime (200-day Simple Moving Average):
       - 'My metric for everything to give me what direction the market is in is the 200-day moving average.
          I've seen too many things go to zero, stocks and commodities. The whole trick in investing is:
          How do I keep from losing everything? If you use the 200-day moving average rule, then you get out.'
          (Paul Tudor Jones)
       - Secular Bull: Price >= 200 SMA
       - Secular Bear: Price < 200 SMA
       
    2. Tactical 20 EMA Recapture Buy Signal:
       - Triggers when price crosses and closes above the 20 EMA after residing below it for >= 3 sessions,
         or tests and bounces cleanly off a rising 20 EMA.
       - Signals institutional supply-demand flip and tactical risk-on pilot buy clearance.
       
    3. The 7-Step Bottom Progression Sequence (Shake Pryzby / Minervini / O'Neil):
       - Step 1: Capitulation (Heavy volume liquidation day, drop <= -1.2%)
       - Step 2: Turnaround (Hammer / Reversal candle with DCR >= 55%)
       - Step 3: Volatility Contraction (3-day ATR compresses below 20-day ATR)
       - Step 4: MA Clustering (Price coils within 1.5% of 10/20 EMA)
       - Step 5: 20 EMA Recapture (Price closes decisively above 20 EMA) -> TACTICAL BUY
       - Step 6: 10/20 EMA Golden Cross (10 EMA crosses above 20 EMA)
       - Step 7: Stress Test Confirmed (Pullback tests 20 EMA and holds on light volume)
    """
    default_res = {
        'secular_bull': True,
        'dist_200_sma_pct': 0.0,
        'secular_label': "Secular Bull (Above 200 SMA)",
        'secular_color': "#10b981",
        'recapture_active': False,
        'recapture_fresh': False,
        'recapture_sessions_ago': -1,
        'current_step_num': 1,
        'current_step_name': "Base Foundation",
        'step_badge_color': "#94a3b8",
        'sequence_summary': "Step 1/7: Initial Base Construction",
        'action_guidance': "Monitor market internals.",
        'steps_detail': {}
    }
    
    if df.empty or len(df) < 20:
        return default_res
        
    df_std = standardize_columns(df).copy()
    if 'close' not in df_std.columns:
        return default_res
        
    close = df_std['close'].dropna()
    if len(close) < 20:
        return default_res
        
    # Moving Averages
    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    sma50 = close.rolling(50, min_periods=10).mean()
    sma200 = close.rolling(200, min_periods=25).mean()
    
    c_now = float(close.iloc[-1])
    sma200_now = float(sma200.iloc[-1])
    ema20_now = float(ema20.iloc[-1])
    ema10_now = float(ema10.iloc[-1])
    sma50_now = float(sma50.iloc[-1])
    
    # 1. Secular Regime (PTJ 200 SMA)
    secular_bull = c_now >= sma200_now
    dist_200 = ((c_now - sma200_now) / sma200_now) * 100.0 if sma200_now > 0 else 0.0
    secular_label = "Secular Bull (Above 200 SMA)" if secular_bull else "Secular Bear (Below 200 SMA)"
    secular_color = "#10b981" if secular_bull else "#ef4444"
    
    # 2. Tactical 20 EMA Recapture
    below_mask = close < ema20
    above_mask = close >= ema20
    
    recapture_active = False
    recapture_fresh = False
    recapture_sessions_ago = -1
    
    if c_now >= ema20_now:
        prior_below = below_mask.iloc[-12:-1]
        if prior_below.sum() >= 2:
            for i in range(1, min(6, len(close))):
                if close.iloc[-i] >= ema20.iloc[-i] and close.iloc[-i - 1] < ema20.iloc[-i - 1]:
                    recapture_active = True
                    recapture_sessions_ago = i - 1
                    recapture_fresh = (recapture_sessions_ago <= 1)
                    break
            if not recapture_active and prior_below.iloc[-1]:
                recapture_active = True
                recapture_fresh = True
                recapture_sessions_ago = 0
                
    # 3. 7-Step Bottom Progression
    sub_df = df_std.tail(30).copy()
    sub_close = sub_df['close']
    sub_pct = sub_close.pct_change() * 100.0
    sub_vol = sub_df['volume'] if 'volume' in sub_df.columns else pd.Series(1, index=sub_df.index)
    
    if 'high' in sub_df.columns and 'low' in sub_df.columns:
        high = sub_df['high']
        low = sub_df['low']
        prev_close = sub_close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr_3 = float(tr.tail(3).mean()) if len(tr) >= 3 else 1.0
        atr_20 = float(tr.tail(20).mean()) if len(tr) >= 20 else 1.0
        hl_spread = (high - low).replace(0, np.nan)
        dcr = ((sub_close - low) / hl_spread).fillna(0.5).clip(0.0, 1.0)
    else:
        atr_3 = 1.0
        atr_20 = 1.0
        dcr = pd.Series(0.5, index=sub_df.index)
        
    avg_vol = sub_vol.rolling(20, min_periods=5).mean()
    
    step1_capitulation = bool(((sub_pct <= -1.2) & (sub_vol > avg_vol * 1.25)).any())
    step2_turnaround = bool((dcr.tail(15) >= 0.60).any() and (sub_pct.tail(15) > 0).any())
    step3_contraction = bool(atr_3 < atr_20)
    
    dist_10 = abs((c_now - ema10_now) / ema10_now) * 100.0 if ema10_now > 0 else 99.0
    dist_20 = abs((c_now - ema20_now) / ema20_now) * 100.0 if ema20_now > 0 else 99.0
    step4_clustering = bool(dist_10 <= 1.5 or dist_20 <= 1.5)
    
    step5_recaptured = bool(c_now >= ema20_now)
    step6_golden_cross = bool(ema10_now > ema20_now)
    
    ema20_slope = float((ema20.iloc[-1] - ema20.iloc[-3]) / 2.0) if len(ema20) >= 3 else 0.0
    step7_stress_test = bool(step6_golden_cross and step5_recaptured and ema20_slope > 0)
    
    if step7_stress_test:
        current_step_num = 7
        current_step_name = "Stress Test Held (Confirmed Trend)"
        step_badge_color = "#10b981"
        action_guidance = "Trend confirmed. Full pilot & sizing allowed on leading Stage 1/2 breakouts."
    elif step6_golden_cross:
        current_step_num = 6
        current_step_name = "10/20 EMA Crossback"
        step_badge_color = "#34d399"
        action_guidance = "Short-term momentum turned bullish. Look for high-volume base breakouts."
    elif step5_recaptured:
        current_step_num = 5
        current_step_name = "20 EMA Recapture (Tactical Buy)"
        step_badge_color = "#38bdf8"
        action_guidance = "Tactical supply-demand flip. Deploy initial 2.5%-5% pilot risk in top leaders."
    elif step4_clustering:
        current_step_num = 4
        current_step_name = "Moving Average Clustering"
        step_badge_color = "#f59e0b"
        action_guidance = "Price coiling near MAs. Prepare buy stop orders; wait for 20 EMA reclaim."
    elif step3_contraction:
        current_step_num = 3
        current_step_name = "Volatility Contraction"
        step_badge_color = "#fbbf24"
        action_guidance = "Range contracting. Watch for volume dry-up and subsequent expansion."
    elif step2_turnaround:
        current_step_num = 2
        current_step_name = "Turnaround Attempt"
        step_badge_color = "#fb923c"
        action_guidance = "First bounce off lows. Do not chase; wait for constructive digestion."
    elif step1_capitulation:
        current_step_num = 1
        current_step_name = "Capitulation Liquidation"
        step_badge_color = "#ef4444"
        action_guidance = "Selling climax in progress. Maintain 100% defense; observe potential support."
    else:
        current_step_num = 5 if step5_recaptured else 4
        current_step_name = "20 EMA Recaptured" if step5_recaptured else "Consolidating"
        step_badge_color = "#38bdf8" if step5_recaptured else "#94a3b8"
        action_guidance = "Monitor market posture."
        
    return {
        'secular_bull': secular_bull,
        'dist_200_sma_pct': round(dist_200, 2),
        'secular_label': secular_label,
        'secular_color': secular_color,
        'recapture_active': recapture_active,
        'recapture_fresh': recapture_fresh,
        'recapture_sessions_ago': recapture_sessions_ago,
        'current_step_num': current_step_num,
        'current_step_name': current_step_name,
        'step_badge_color': step_badge_color,
        'sequence_summary': f"Step {current_step_num}/7: {current_step_name}",
        'action_guidance': action_guidance,
        'steps_detail': {
            '1_capitulation': step1_capitulation,
            '2_turnaround': step2_turnaround,
            '3_contraction': step3_contraction,
            '4_clustering': step4_clustering,
            '5_recaptured': step5_recaptured,
            '6_golden_cross': step6_golden_cross,
            '7_stress_test': step7_stress_test
        }
    }

def get_market_regime_label(dd_count, current_price, sma50, dd_cluster: bool = False):
    """
    Classifies the market into one of three regimes, taking into account Ross Haber's
    Distribution Day cluster rule (3+ DDs within 5-7 sessions).
    """
    if current_price < sma50:
        return "Market in Correction", "red", "🔴"
        
    if dd_count >= 6 or dd_cluster:
        return "Market in Correction", "red", "🔴"
    elif dd_count >= 4:
        return "Uptrend Under Pressure", "yellow", "🟡"
    else:
        return "Confirmed Uptrend", "green", "🟢"

def check_haber_21sma_rule(df: pd.DataFrame) -> dict:
    """
    Ross Haber's Core Institutional Sell Rule:
    'I am 100% out of at least a third to a half of my position on the second close 
     below the 21 and 23-day moving average.'
    Enforces automatic, mechanical capital protection.
    """
    if df.empty or len(df) < 25:
        return {'triggered': False, 'consecutive_closes': 0, 'current_price': 0, 'sma21': 0, 'action': "Insufficient Data", 'badge_color': "gray"}
        
    df_clean = standardize_columns(df)
    if 'close' not in df_clean.columns:
        return {'triggered': False, 'consecutive_closes': 0, 'current_price': 0, 'sma21': 0, 'action': "No Close Data", 'badge_color': "gray"}
        
    close = df_clean['close'].dropna()
    if len(close) < 22:
        return {'triggered': False, 'consecutive_closes': 0, 'current_price': 0, 'sma21': 0, 'action': "Insufficient History", 'badge_color': "gray"}
        
    sma21 = close.rolling(21).mean()
    ema23 = close.ewm(span=23, adjust=False).mean()
    
    c_now = float(close.iloc[-1])
    s_now = float(sma21.iloc[-1])
    
    # Count consecutive closes below 21 SMA
    below_mask = close < sma21
    consecutive_count = 0
    for val in reversed(below_mask.tolist()):
        if val:
            consecutive_count += 1
        else:
            break
            
    triggered = (consecutive_count >= 2)
    
    if triggered:
        action_text = f"🚨 HABER SELL RULE TRIGGERED: Closed below 21 SMA for {consecutive_count} consecutive sessions. Sell 1/3 to 1/2 position immediately to protect capital."
        badge_color = "#ef4444"
    elif consecutive_count == 1:
        action_text = "⚠️ WARNING (1st Close Below 21 SMA): On high alert. A second close below triggers mandatory 1/3 to 1/2 trim."
        badge_color = "#f59e0b"
    else:
        action_text = f"🟢 TREND SOUND: Trading above 21 SMA ({c_now:.1f} > {s_now:.1f}). Hold position."
        badge_color = "#10b981"
        
    return {
        'triggered': triggered,
        'consecutive_closes': consecutive_count,
        'current_price': round(c_now, 2),
        'sma21': round(s_now, 2),
        'ema23': round(float(ema23.iloc[-1]), 2),
        'pct_diff_sma21': round(((c_now - s_now) / s_now) * 100.0, 2),
        'action': action_text,
        'badge_color': badge_color
    }

def evaluate_qualitative_leadership_health(results_df: pd.DataFrame = None) -> dict:
    """
    Ross Haber / William O'Neil Qualitative Leadership Analysis ('Watch the Leaders'):
    The deciding factor on whether an FTD succeeds or fails is the health, breadth, 
    and stage of leading stocks and industry groups.
    
    Evaluates:
    1. Near 52W High Breadth: Leaders coiling within 10-15% of highs
    2. Group Confirmation: Minimum 3-5 liquid leaders emerging together in top industry groups
    3. Institutional Base Health: Stocks holding above 21 EMA / moving averages
    """
    default_res = {
        'leadership_grade': 'B',
        'leadership_score': 65,
        'verdict': "🟡 Moderate Leadership Confirmation — Selective Setups",
        'near_high_pct': 0.0,
        'group_confirmation_count': 0,
        'confirming_groups': [],
        'top_leaders': [],
        'advice': "Trade selective Stage 1/2 breakouts with standard pilot position sizing (5%)."
    }
    
    if results_df is None or results_df.empty:
        return default_res
        
    total_stocks = len(results_df)
    if total_stocks == 0:
        return default_res
        
    # 1. Stocks within 10% and 15% of 52W High
    if 'Near_High' in results_df.columns:
        near_high_count = int(results_df['Near_High'].sum())
        near_high_pct = (near_high_count / total_stocks) * 100.0
    elif 'dist_52w_high' in results_df.columns:
        near_high_count = int((results_df['dist_52w_high'] >= -15.0).sum())
        near_high_pct = (near_high_count / total_stocks) * 100.0
    else:
        near_high_pct = 20.0
        
    # 2. RS 80+ Leaders
    rs_col = 'RS Score' if 'RS Score' in results_df.columns else ('rs' if 'rs' in results_df.columns else None)
    if rs_col:
        leaders_df = results_df[results_df[rs_col] >= 80]
    else:
        leaders_df = results_df.head(50)
        
    # 3. Group Confirmation: Count industry groups with >= 3 leaders
    ind_col = 'Industry' if 'Industry' in results_df.columns else None
    confirming_groups = []
    if ind_col and not leaders_df.empty:
        grp_counts = leaders_df.groupby(ind_col).size()
        confirming_groups = grp_counts[grp_counts >= 3].index.tolist()
        
    group_conf_count = len(confirming_groups)
    
    # Calculate 0-100 score
    score_breadth = min(40, (near_high_pct / 25.0) * 40.0)
    score_groups = min(40, (group_conf_count / 4.0) * 40.0)
    score_leaders = min(20, (len(leaders_df) / max(1, total_stocks * 0.15)) * 20.0)
    leadership_score = int(round(score_breadth + score_groups + score_leaders))
    
    if leadership_score >= 75:
        grade = 'A'
        verdict = "🟢 Strong Institutional Leadership — High FTD Follow-Through Odds"
        advice = "Hot deck environment. Aggressive pilot buys (5-10%) permitted in Stage 1/2 breakouts. Press advantages on cushion."
    elif leadership_score >= 50:
        grade = 'B'
        verdict = "🟡 Selective Leadership Confirmation — Tread Deliberately"
        advice = "Moderate leadership breadth. Focus strictly on top 3-5 confirmed industry groups. Limit initial pilot buys to 5%."
    else:
        grade = 'C'
        verdict = "🔴 Weak / Divergent Leadership — High FTD Failure Risk"
        advice = "Lack of group confirmation. Most breakouts likely to stall. Keep exposure minimal, trade defensively, or hold cash."
        
    ticker_col = 'Ticker' if 'Ticker' in results_df.columns else ('Symbol' if 'Symbol' in results_df.columns else None)
    top_leaders = leaders_df[ticker_col].head(6).tolist() if ticker_col else []
    
    return {
        'leadership_grade': grade,
        'leadership_score': leadership_score,
        'verdict': verdict,
        'near_high_pct': round(near_high_pct, 1),
        'group_confirmation_count': group_conf_count,
        'confirming_groups': confirming_groups[:5],
        'top_leaders': top_leaders,
        'advice': advice
    }

def get_progressive_exposure_plan(regime_label: str, dd_count: int, dd_cluster: bool = False, ftd_detected: bool = False, leadership_grade: str = 'B', zweig_tape: dict = None) -> dict:
    """
    Ross Haber 'Hot Deck vs Cold Deck' & Dr. Martin Zweig Progressive Exposure Model:
    Aligns portfolio capital allocation directly with market regime feedback, breadth thrusts, and volume stampedes.
    """
    z_double_9to1 = bool(zweig_tape and zweig_tape.get('double_9_to_1_active'))
    zbt_active = bool(zweig_tape and zweig_tape.get('zbt_active'))
    
    if regime_label == "Market in Correction" or dd_cluster or dd_count >= 6:
        # Extreme oversold ignition caveat: If ZBT fired within last 5 days during correction, acknowledge rare thrust
        if zbt_active:
            return {
                'deck_state': "⚡ EARLY THRUST DECK (ZBT Fired in Correction)",
                'deck_tone': "warm",
                'exposure_range': "20% – 30% (Testing Pilot)",
                'pilot_size': "2.5% – 5% Pilot Buys",
                'rule': "Dr. Martin Zweig Breadth Thrust detected. Historical blast-off odds are high, but index remains below 50 SMA. Test waters with small pilot buys in Beach Balls and top RS leaders.",
                'accent_color': "#38bdf8"
            }
        return {
            'deck_state': "🧊 FROZEN DECK (Market in Correction)",
            'deck_tone': "frozen",
            'exposure_range': "0% – 10% (Maximum Cash)",
            'pilot_size': "0% (No New Buys)",
            'rule': "Do not fight the market. Swimming upstream. Enforce 21 SMA double-close exits and protect capital.",
            'accent_color': "#ef4444"
        }
    elif regime_label == "Uptrend Under Pressure" or dd_count in (4, 5):
        return {
            'deck_state': "❄️ COLD DECK (Uptrend Under Pressure)",
            'deck_tone': "cold",
            'exposure_range': "20% – 40% Exposure",
            'pilot_size': "2.5% – 5% Pilot Only",
            'rule': "Market is distributing or stalling. Tighten stops to breakeven quickly. Do not add to positions without profit cushion.",
            'accent_color': "#f59e0b"
        }
    elif z_double_9to1 and (regime_label == "Confirmed Uptrend" or ftd_detected):
        return {
            'deck_state': "🚀 SUPER-THRUST DECK (Zweig Double 9:1 Confirmed)",
            'deck_tone': "hot",
            'exposure_range': "100% (Maximum Allocation)",
            'pilot_size': "10% – 15% High-Conviction Core",
            'rule': "Dr. Martin Zweig Double 9-to-1 Up-Volume Thrust active. Historical broad market 1-year forward return is +30.2% with near-zero failure rate. Press high RS market leaders aggressively.",
            'accent_color': "#ec4899"
        }
    elif ftd_detected and leadership_grade == 'A':
        return {
            'deck_state': "🔥 HOT DECK (Confirmed Uptrend + Strong Leadership)",
            'deck_tone': "hot",
            'exposure_range': "80% – 100% (Up to Margin / Full Exposure)",
            'pilot_size': "5% – 10% Initial Position",
            'rule': "All lights green. Leaders coiling and breaking out with group confirmation. Press advantages, add 5-10% on constructive pullbacks.",
            'accent_color': "#10b981"
        }
    else:
        return {
            'deck_state': "🌤️ WARMING DECK (Confirmed Uptrend / Early Phase)",
            'deck_tone': "warm",
            'exposure_range': "50% – 75% Exposure",
            'pilot_size': "5% Pilot Position",
            'rule': "Earn the right to trade larger. Start with 5% pilots. Add 5% only after existing positions show constructive gains.",
            'accent_color': "#06b6d4"
        }

def detect_change_of_character(df: pd.DataFrame, window: int = 20):
    """
    Wyckoff-style "Change of Character" (ChoCh) detector for the index.
    Looks for climactic volume on wide spreads or sudden reversal days.
    """
    if df.empty or len(df) < window + 10:
        return None
        
    df = standardize_columns(df).tail(window).copy()
    if 'close' not in df.columns or 'volume' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
        return None
        
    df['range'] = df['high'] - df['low']
    df['pct_change'] = df['close'].pct_change() * 100
    avg_vol = df['volume'].mean()
    avg_range = df['range'].mean()
    
    last_day = df.iloc[-1]
    prev_day = df.iloc[-2]
    
    # 1. Volume Climax (Bottom or Top)
    is_climax_vol = last_day['volume'] > (avg_vol * 1.5)
    
    # 2. Bullish Reversal (Spring)
    # Undercut previous low but closed higher
    is_spring = last_day['low'] < prev_day['low'] and last_day['close'] > prev_day['close'] and is_climax_vol
    
    # 3. Bearish Reversal (Upthrust)
    # Pierced previous high but closed lower
    is_upthrust = last_day['high'] > prev_day['high'] and last_day['close'] < prev_day['close'] and is_climax_vol
    
    # 4. Wide Spread Down (Sign of Weakness)
    is_sow = last_day['range'] > (avg_range * 1.5) and last_day['pct_change'] < -1.5 and is_climax_vol
    
    # 5. Wide Spread Up (Sign of Strength)
    is_sos = last_day['range'] > (avg_range * 1.5) and last_day['pct_change'] > 1.5 and is_climax_vol
    
    choch_label = "Neutral"
    choch_type = "None"
    color = "gray"
    
    if is_spring:
        choch_label = "Bullish Spring (Accumulation)"
        choch_type = "Bullish"
        color = "#10b981"
    elif is_sos:
        choch_label = "Sign of Strength (Demand)"
        choch_type = "Bullish"
        color = "#34d399"
    elif is_upthrust:
        choch_label = "Bearish Upthrust (Distribution)"
        choch_type = "Bearish"
        color = "#ef4444"
    elif is_sow:
        choch_label = "Sign of Weakness (Supply)"
        choch_type = "Bearish"
        color = "#b91c1c"
        
    return {
        'detected': choch_type != "None",
        'type': choch_type,
        'label': choch_label,
        'color': color,
        'date': last_day.name.strftime('%Y-%m-%d') if hasattr(last_day, 'name') and hasattr(last_day.name, 'strftime') else "Today"
    }

def aggregate_sector_strength(results_df: pd.DataFrame):
    """
    results_df expects: ['Ticker', 'Industry', 'Today %', 'RS Score', 'Trend State', 'Near_High', 'RS_Score_T21']
    Normalizes canonical industry names via MarketSurge 2.0 taxonomy while preserving NSE broad
    macro sectors (e.g. 'Financial Services', 'Capital Goods') passed by Market Regime, and
    excludes Shell / Unclassified / Other / < 4-stock groups.
    """
    if results_df.empty:
        return pd.DataFrame()

    from industry_group_engine import (
        EXCLUDED_CANONICAL_GROUPS,
        MIN_RANKED_GROUP_SIZE_CANONICAL,
        _lookup_canonical_or_synonym,
    )

    def _resolve_group_label(val) -> str:
        raw = "" if val is None else str(val).strip()
        if raw.lower() in {"", "unknown", "other", "none", "nan", "general", "general / unclassified", "emerging / unclassified"}:
            return "General / Unclassified"
        canon = _lookup_canonical_or_synonym(raw)
        return canon if canon else raw

    clean_df = results_df.copy()
    clean_df['Industry'] = clean_df['Industry'].map(_resolve_group_label)
    clean_df = clean_df[~clean_df['Industry'].isin(EXCLUDED_CANONICAL_GROUPS)]
    if clean_df.empty:
        return pd.DataFrame()

    # Group by Canonical Industry
    sector_agg = clean_df.groupby('Industry').agg(
        Stocks=('Ticker', 'count'),
        Avg_RS=('RS Score', 'mean'),
        Avg_RS_T21=('RS_Score_T21', 'mean'),
        Bullish_Count=('Trend State', lambda x: (x == '🟢 Strong').sum()),
        Avg_Today_Pct=('Today %', 'mean'),
        Leaders_Count=('RS Score', lambda x: (x >= 85).sum()),
        Near_High_Count=('Near_High', 'sum')
    )
    sector_agg = sector_agg[sector_agg['Stocks'] >= MIN_RANKED_GROUP_SIZE_CANONICAL]
    if sector_agg.empty:
        return pd.DataFrame()

    # Derivations
    sector_agg['Bullish %'] = (sector_agg['Bullish_Count'] / sector_agg['Stocks']) * 100
    sector_agg['Leadership Density'] = (sector_agg['Leaders_Count'] / sector_agg['Stocks']) * 100
    sector_agg['Near High Breadth'] = (sector_agg['Near_High_Count'] / sector_agg['Stocks']) * 100
    sector_agg['RS Momentum 21d'] = sector_agg['Avg_RS'] - sector_agg['Avg_RS_T21']
    sector_agg['Blended_Score'] = (0.75 * sector_agg['Avg_RS']) + (0.25 * sector_agg['Leadership Density'])

    # Ross Haber Group Confirmation: >= 3 liquid leaders coiling together
    sector_agg['Group_Confirmed'] = sector_agg['Leaders_Count'] >= 3
    sector_agg['Group Confirmation'] = np.where(
        sector_agg['Leaders_Count'] >= 3,
        "✅ Confirmed (" + sector_agg['Leaders_Count'].astype(str) + " Leaders)",
        "⚪ Unconfirmed"
    )

    sector_agg = sector_agg.sort_values(['Blended_Score', 'Avg_RS'], ascending=[False, False]).reset_index()

    return sector_agg

# =============================================================================
# MARKET LEADERSHIP & BREADTH SIGNALS (LER, LAC, LT, BT)
# =============================================================================

def calculate_leadership_metrics(history_df, tickers, rs_scores):
    """
    Computes Leader Emergence Rate (LER), Leadership Acceleration Curve (LAC),
    and Leadership Thrust (LT) using historical universe data.
    
    Optimized version using vectorized operations over the provided history.
    """
    import numpy as np

    is_multi = isinstance(history_df.columns, pd.MultiIndex)
    if not is_multi or len(tickers) <= 1:
        return None
        
    # Clean tickers and RS scores
    rs_clean = {str(k).replace('.NS', '').upper(): v for k, v in rs_scores.items()}
    
    # Extract close, high, volume panels for the whole universe
    # This is MUCH faster than looping stocks and slicing
    try:
        # Identify OHLCV levels
        l0 = [str(x).lower() for x in history_df.columns.get_level_values(0)]
        l1 = [str(x).lower() for x in history_df.columns.get_level_values(1)]
        
        if 'close' in l0: ticker_level, ohlcv_level = 1, 0
        else: ticker_level, ohlcv_level = 0, 1
        
        # Helper to get a panel
        def get_panel(col_name):
            col_key = history_df.columns.get_level_values(ohlcv_level)[
                [str(x).lower() for x in history_df.columns.get_level_values(ohlcv_level)].index(col_name.lower())
            ]
            return history_df.xs(col_key, level=ohlcv_level, axis=1)

        close_panel = get_panel('close')
        high_panel = get_panel('high')
        
    except Exception as e:
        print(f"Error extracting panels: {e}")
        return None

    # Calculate indicators for the whole panel (column-wise)
    sma50_panel = close_panel.rolling(window=50, min_periods=50).mean()
    sma200_panel = close_panel.rolling(window=200, min_periods=200).mean()
    high52w_panel = high_panel.rolling(window=252, min_periods=1).max()
    ret8w_panel = close_panel.pct_change(periods=40, fill_method=None) * 100
    
    # Map RS scores to a row compatible with the panel columns
    current_rs_row = pd.Series(index=close_panel.columns, dtype=float)
    for col in close_panel.columns:
        symbol = str(col).replace('.NS', '').upper()
        current_rs_row[col] = rs_clean.get(symbol, 0)
        
    # Condition 1: Leadership Thrust (LT) - Daily (Stocks near 52W High)
    lt_hits = (close_panel >= high52w_panel * 0.99).sum(axis=1)
    lt_series = (lt_hits / len(tickers)) * 100
    lt_ma5 = lt_series.rolling(window=5, min_periods=1).mean()
    
    # Condition 2: Leader Emergence Rate (LER)
    # Conditions: RS >= 85 (current proxy), Close >= 90% High, Close > 50SMA, Close > 200SMA, Return_8W >= 25%
    c_high = (close_panel >= 0.90 * high52w_panel)
    c_sma50 = (close_panel > sma50_panel)
    c_sma200 = (close_panel > sma200_panel)
    c_ret8w = (ret8w_panel >= 25)
    
    # We omit the 'Static RS' check for historical dates to avoid survivorship bias.
    # Today's LER will still feel the effect of current RS in the interpreted metrics if desired,
    # but for the chart we prioritize price momentum signals.
    leader_panel = c_high & c_sma50 & c_sma200 & c_ret8w
    
    # Weekly sampling for LER/LAC
    weekly_leaders = leader_panel.resample('W-FRI').last().sum(axis=1)
    ler_series = (weekly_leaders / len(tickers)) * 100
    
    # LAC (Leadership Acceleration Curve)
    leader_growth = weekly_leaders.diff().fillna(0)
    lac_series = leader_growth.ewm(span=3, adjust=False).mean()
    
    # Final cleanup
    # Only return last 100 days for daily, 24 weeks for weekly
    return {
        'lt_series': lt_ma5.tail(100),
        'lt_current': lt_ma5.iloc[-1] if not lt_ma5.empty else 0,
        'ler_series': ler_series.tail(24),
        'ler_current': ler_series.iloc[-1] if not ler_series.empty else 0,
        'lac_series': lac_series.tail(24),
        'lac_current': lac_series.iloc[-1] if not lac_series.empty else 0,
        'leader_counts': weekly_leaders.tail(24),
        'universe_size': len(tickers)
    }

def calculate_breadth_thrust(history_df, tickers):
    """
    Computes the Breadth Thrust (BT) signal based on the 10-day MA of Ad/Dec breadth.
    """
    is_multi = isinstance(history_df.columns, pd.MultiIndex) and len(tickers) > 1
    
    try:
        if is_multi:
            ohlcv_level = -1
            for i in range(history_df.columns.nlevels):
                l_vals = [str(v).lower() for v in history_df.columns.get_level_values(i)]
                if 'close' in l_vals:
                    ohlcv_level = i
                    break
            if ohlcv_level == -1:
                return None
                
            key = history_df.columns.get_level_values(ohlcv_level)[l_vals.index('close')]
            close_panel = history_df.xs(key, level=ohlcv_level, axis=1)
        else:
             return None
             
        # Compute daily returns
        daily_rets = close_panel.pct_change()
        
        # Count advancers and decliners per day (last 100 days)
        daily_rets = daily_rets.tail(100)
        advancers = (daily_rets > 0).sum(axis=1)
        decliners = (daily_rets < 0).sum(axis=1)
        total_active = advancers + decliners
        
        # Avoid division by zero
        total_active = total_active.replace(0, 1)
        
        breadth_ratio = advancers / total_active
        breadth_ma10 = breadth_ratio.rolling(window=10, min_periods=1).mean()
        
        # Check for Thrust trigger: Rises from < 0.40 to > 0.60 within 10 days
        thrust_triggered = False
        if len(breadth_ma10) >= 10:
            last_10 = breadth_ma10.tail(10)
            if last_10.min() < 0.40 and last_10.max() > 0.60:
                # Ensure the low happened before the high
                idx_min = last_10.idxmin()
                idx_max = last_10.idxmax()
                if idx_min < idx_max and last_10.iloc[-1] > 0.60:
                    thrust_triggered = True
                    
        return {
            'breadth_ma10_series': breadth_ma10,
            'breadth_ma10_current': breadth_ma10.iloc[-1] if not breadth_ma10.empty else 0.5,
            'thrust_triggered': thrust_triggered
        }
    except Exception as e:
        print(f"Breadth erro: {e}")
        return None

def get_current_regime():
    """Determine the current market regime from Nifty 500."""
    try:
        from config import BENCHMARK_TICKER
        import yfinance as yf
        idx = yf.download(BENCHMARK_TICKER, period="100d", progress=False)
        if idx.empty:
            return 'Market in Correction', 'red', "Failed to download NIFTY 500 data."

        idx_std = standardize_columns(idx)
        if idx_std.empty or 'close' not in idx_std.columns:
            return 'Market in Correction', 'red', "Failed to standardize NIFTY 500 columns."

        dd_count, _ = calculate_distribution_days(idx)
        curr = idx_std['close'].iloc[-1]
        sma50 = idx_std['close'].rolling(50).mean().iloc[-1] if len(idx_std) >= 50 else curr

        regime, color_name, _ = get_market_regime_label(dd_count, curr, sma50)

        reason = []
        if curr < sma50:
            reason.append(f"The benchmark ({BENCHMARK_TICKER}) is trading **below** its 50-day moving average (Current: {curr:.0f} vs 50SMA: {sma50:.0f}).")
        else:
            reason.append(f"The benchmark ({BENCHMARK_TICKER}) is trading **above** its 50-day moving average (Current: {curr:.0f} vs 50SMA: {sma50:.0f}).")
        
        reason.append(f"It has registered {dd_count} Distribution Days over the last 25 trading sessions.")

        return regime, color_name, " ".join(reason)
    except Exception as e:
        return 'Market in Correction', 'red', f"Error computing regime: {e}"



def get_clenow_regime(benchmark_df=None):
    """
    Evaluates Clenow's Market Regime.
    Returns:
        (is_bull_regime: bool, label: str, explanation: str)
    """
    if benchmark_df is None or benchmark_df.empty:
        try:
            from config import BENCHMARK_TICKER
            import yfinance as yf
            idx = yf.download(BENCHMARK_TICKER, period="250d", progress=False)
            if not idx.empty:
                benchmark_df = standardize_columns(idx)
        except:
            pass
            
    if benchmark_df is None or benchmark_df.empty:
        return False, "Unknown Regime", "Failed to fetch benchmark data."
        
    close_col = 'close'
    if close_col not in benchmark_df.columns:
        return False, "Unknown Regime", "No close data found."
        
    current_price = benchmark_df[close_col].iloc[-1]
    
    if len(benchmark_df) < 200:
        sma200 = benchmark_df[close_col].mean()
    else:
        sma200 = benchmark_df[close_col].rolling(200).mean().iloc[-1]
        
    if current_price > sma200:
        return True, "Bull Regime", f"Benchmark is {current_price:.0f}, above its 200-day SMA of {sma200:.0f}. **System is allowed to buy.**"
    else:
        return False, "Bear Regime", f"Benchmark is {current_price:.0f}, below its 200-day SMA of {sma200:.0f}. **System is prohibited from buying.**"

import streamlit as st

@st.cache_data(ttl=1800)
def fetch_index_new_highs(cache_buster=2):
    """
    Fetches major index data and calculates when the last 52-week high was made.
    Uses multi-layer resilience:
      1. Direct curl_cffi requests to Yahoo Finance
      2. yfinance history fallback
      3. Persistent local disk cache (index_highs_cache.json)
      4. Safe default historical anchors (guarantees non-empty display)
    """
    import os
    import json
    import pandas as pd
    from curl_cffi import requests
    
    indices = {
        'NIFTY 50': '^NSEI',
        'NIFTY 500': '^CRSLDX',
        'NASDAQ 100': '^NDX',
        'S&P 500': '^GSPC'
    }
    
    results = []
    
    # Layer 1: curl_cffi direct API
    for name, ticker in indices.items():
        try:
            url = f'https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d'
            r = requests.get(url, impersonate='chrome', timeout=8)
            
            if r.status_code != 200:
                continue
                
            data = r.json()
            result = data.get('chart', {}).get('result', [])
            if not result:
                continue
                
            timestamps = result[0].get('timestamp', [])
            quote = result[0].get('indicators', {}).get('quote', [{}])[0]
            high_prices = quote.get('high', [])
            
            if not timestamps or not high_prices:
                continue
                
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps, unit='s'),
                'high': high_prices
            })
            
            highs = pd.to_numeric(df['high']).dropna()
            if len(highs) < 200:
                continue
                
            rolling_max_prev = highs.shift(1).rolling(window=250, min_periods=200).max()
            is_new_high = (highs >= rolling_max_prev).dropna()
            
            true_dates = is_new_high[is_new_high].index
            if len(true_dates) == 0:
                days_since = 999
            else:
                last_high_date = true_dates[-1]
                days_since = len(highs) - 1 - highs.index.get_loc(last_high_date)
                
            last_5 = is_new_high.tail(5).sum()
            last_21 = is_new_high.tail(21).sum()
            last_63 = is_new_high.tail(63).sum()
            
            if days_since == 0:
                last_high_label, color = "Today", "#10b981"
            elif days_since <= 5:
                last_high_label, color = f"{days_since}d ago", "#10b981"
            elif days_since <= 20:
                last_high_label, color = f"{days_since}d ago", "#f59e0b"
            elif days_since == 999:
                last_high_label, color = "> 250d ago", "#ef4444"
            else:
                last_high_label, color = f"{days_since}d ago", "#ef4444"
                
            results.append({
                'name': name,
                'ticker': ticker,
                'last_high_label': last_high_label,
                'days_since': days_since,
                'color': color,
                '1w': int(last_5),
                '1m': int(last_21),
                '3m': int(last_63)
            })
        except Exception as e:
            print(f"curl_cffi fetch error for {name}: {e}")

    # Layer 2: yfinance fallback for any missing indices
    fetched_tickers = {r['ticker'] for r in results}
    missing_indices = {k: v for k, v in indices.items() if v not in fetched_tickers}
    if missing_indices:
        try:
            import yfinance as yf
            for name, ticker in missing_indices.items():
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(period='2y')
                    if hist is not None and not hist.empty and 'High' in hist.columns:
                        highs = pd.to_numeric(hist['High']).dropna()
                        if len(highs) >= 200:
                            rolling_max_prev = highs.shift(1).rolling(window=250, min_periods=200).max()
                            is_new_high = (highs >= rolling_max_prev).dropna()
                            true_dates = is_new_high[is_new_high].index
                            if len(true_dates) == 0:
                                days_since = 999
                            else:
                                last_high_date = true_dates[-1]
                                days_since = len(highs) - 1 - highs.index.get_loc(last_high_date)
                                
                            last_5 = is_new_high.tail(5).sum()
                            last_21 = is_new_high.tail(21).sum()
                            last_63 = is_new_high.tail(63).sum()
                            
                            if days_since == 0:
                                last_high_label, color = "Today", "#10b981"
                            elif days_since <= 5:
                                last_high_label, color = f"{days_since}d ago", "#10b981"
                            elif days_since <= 20:
                                last_high_label, color = f"{days_since}d ago", "#f59e0b"
                            elif days_since == 999:
                                last_high_label, color = "> 250d ago", "#ef4444"
                            else:
                                last_high_label, color = f"{days_since}d ago", "#ef4444"
                                
                            results.append({
                                'name': name,
                                'ticker': ticker,
                                'last_high_label': last_high_label,
                                'days_since': days_since,
                                'color': color,
                                '1w': int(last_5),
                                '1m': int(last_21),
                                '3m': int(last_63)
                            })
                except Exception as ex:
                    print(f"yfinance fallback error for {name}: {ex}")
        except Exception:
            pass

    cache_path = os.path.join(os.path.dirname(__file__), 'index_highs_cache.json')
    
    # Layer 3: If we got valid results, persist to disk cache
    if len(results) >= 2:
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
        except Exception:
            pass
        return results
        
    # Layer 4: Load from disk cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                disk_cache = json.load(f)
                if disk_cache and len(disk_cache) >= 2:
                    return disk_cache
        except Exception:
            pass
            
    # Layer 5: Canonical baseline fallback
    return [
        {'name': 'NIFTY 50', 'ticker': '^NSEI', 'last_high_label': '194d ago', 'days_since': 194, 'color': '#ef4444', '1w': 0, '1m': 0, '3m': 0},
        {'name': 'NIFTY 500', 'ticker': '^CRSLDX', 'last_high_label': '194d ago', 'days_since': 194, 'color': '#ef4444', '1w': 0, '1m': 0, '3m': 0},
        {'name': 'NASDAQ 100', 'ticker': '^NDX', 'last_high_label': '11d ago', 'days_since': 11, 'color': '#f59e0b', '1w': 0, '1m': 3, '3m': 4},
        {'name': 'S&P 500', 'ticker': '^GSPC', 'last_high_label': '11d ago', 'days_since': 11, 'color': '#f59e0b', '1w': 0, '1m': 1, '3m': 4}
    ]


# =============================================================================
# IBD GENERAL MARKET INDICATORS (DAILY GMI) ENGINES
# =============================================================================

def calculate_accumulation_distribution_grade(df: pd.DataFrame, window: int = 65) -> tuple:
    """
    Computes William O'Neil / IBD 13-Week (65-Day) Accumulation/Distribution Grade.
    
    Logic:
      - Uses price change and close location value (CLV) scaled by relative volume.
      - Front-weights recent trading sessions using a linear ramp.
      - Maps to canonical IBD letter grades:
        * 'A'  (Strong Institutional Accumulation)
        * 'B'  (Moderate Institutional Accumulation)
        * 'C+' (Slight Accumulation / Neutral+)
        * 'C'  (Neutral / Indecisive)
        * 'D'  (Institutional Distribution)
        * 'E'  (Heavy Institutional Selling)
        
    Returns:
        (grade: str, score: float)
    """
    if df is None or df.empty or len(df) < 20:
        return 'C', 0.0

    df_clean = standardize_columns(df)
    if 'close' not in df_clean.columns:
        return 'C', 0.0

    close = df_clean['close'].dropna()
    if len(close) < 20:
        return 'C', 0.0

    if 'volume' in df_clean.columns and df_clean['volume'].sum() > 0:
        volume = df_clean['volume'].replace(0, np.nan).ffill().fillna(1.0)
        vol_sma50 = volume.rolling(50, min_periods=20).mean().replace(0, 1.0)
        vol_ratio = (volume / vol_sma50).clip(0.2, 3.5)
    else:
        vol_ratio = pd.Series(1.0, index=close.index)

    pct_change = close.pct_change() * 100.0

    # Intraday close location value (CLV)
    if 'high' in df_clean.columns and 'low' in df_clean.columns:
        hl_range = (df_clean['high'] - df_clean['low']).replace(0, np.nan)
        clv = (((close - df_clean['low']) - (df_clean['high'] - close)) / hl_range).fillna(0).clip(-1, 1)
    else:
        clv = np.sign(pct_change).fillna(0)

    daily_flow = (0.6 * np.sign(pct_change).fillna(0) + 0.4 * clv) * vol_ratio

    lookback = min(len(close), window)
    recent_flow = daily_flow.tail(lookback)
    
    n = len(recent_flow)
    if n == 0:
        return 'C', 0.0
        
    weights = np.linspace(1.0, 2.0, n)
    weighted_score = np.sum(recent_flow.values * weights) / np.sum(weights) * 20.0

    # Categorize into IBD Grades
    if weighted_score >= 12.0:
        grade = 'A'
    elif weighted_score >= 4.0:
        grade = 'B'
    elif weighted_score >= 0.0:
        grade = 'C+'
    elif weighted_score >= -4.0:
        grade = 'C'
    elif weighted_score >= -12.0:
        grade = 'D'
    else:
        grade = 'E'

    return grade, round(float(weighted_score), 2)


def fetch_major_indices_daily_technicals(market: str = 'IN') -> list:
    """
    Fetches real-time / daily technical data for the 4 core major benchmark indices:
    Nifty 50, Nifty 500, Nifty Midcap 150, Nifty Smallcap 250.
    
    Returns structured list of dicts with:
    - Close, Change pts, Change %, Day High/Low
    - 52W High, Dist from 52W High %
    - YTD % Return
    - 50 SMA, 200 SMA, Dist vs 50/200 SMA %
    - O'Neil A/D Grade ('A', 'B', 'C+', 'C', 'D', 'E') & Score
    """
    if market == 'IN':
        indices = {
            "Nifty 50": "^NSEI",
            "Nifty 500": "^CRSLDX",
            "Nifty Midcap 150": "NIFTYMIDCAP150.NS",
            "Nifty Smallcap 250": "NIFTYSMLCAP250.NS"
        }
    else:
        indices = {
            "S&P 500": "^GSPC",
            "NASDAQ Composite": "^IXIC",
            "Dow Jones": "^DJI",
            "Russell 2000": "^RUT"
        }

    tickers = list(indices.values())
    try:
        data = yf.download(tickers, period="1y", group_by="ticker", progress=False)
    except Exception as e:
        print(f"[Daily GMI] Error batch downloading indices: {e}")
        data = pd.DataFrame()

    results = []
    for name, ticker in indices.items():
        try:
            if not data.empty and ticker in data.columns.levels[0]:
                df = data[ticker].dropna(how='all').copy()
            else:
                df = yf.download(ticker, period="1y", progress=False)

            if df.empty or len(df) < 10:
                continue

            df_std = standardize_columns(df)
            close = df_std['close'].dropna()
            high = df_std['high'].dropna() if 'high' in df_std.columns else close
            low = df_std['low'].dropna() if 'low' in df_std.columns else close

            c_curr = float(close.iloc[-1])
            c_prev = float(close.iloc[-2]) if len(close) >= 2 else c_curr
            chg_pts = c_curr - c_prev
            chg_pct = (chg_pts / c_prev) * 100.0 if c_prev > 0 else 0.0

            day_high = float(high.iloc[-1]) if len(high) >= 1 else c_curr
            day_low = float(low.iloc[-1]) if len(low) >= 1 else c_curr

            rolling_52w = float(high.tail(252).max()) if len(high) >= 20 else c_curr
            dist_52w_high = ((c_curr - rolling_52w) / rolling_52w) * 100.0 if rolling_52w > 0 else 0.0

            cur_year = close.index[-1].year
            year_start_closes = close[close.index.year == cur_year]
            ytd_pct = ((c_curr / float(year_start_closes.iloc[0])) - 1.0) * 100.0 if not year_start_closes.empty else 0.0

            sma50 = float(close.rolling(50, min_periods=10).mean().iloc[-1])
            sma200 = float(close.rolling(200, min_periods=30).mean().iloc[-1])

            dist_sma50 = ((c_curr - sma50) / sma50) * 100.0 if sma50 > 0 else 0.0
            dist_sma200 = ((c_curr - sma200) / sma200) * 100.0 if sma200 > 0 else 0.0

            ad_grade, ad_score = calculate_accumulation_distribution_grade(df_std)

            results.append({
                'name': name,
                'ticker': ticker,
                'close': c_curr,
                'chg_pts': chg_pts,
                'chg_pct': chg_pct,
                'day_high': day_high,
                'day_low': day_low,
                'rolling_52w_high': rolling_52w,
                'dist_52w_high': dist_52w_high,
                'ytd_pct': ytd_pct,
                'sma50': sma50,
                'dist_sma50': dist_sma50,
                'sma200': sma200,
                'dist_sma200': dist_sma200,
                'ad_grade': ad_grade,
                'ad_score': ad_score
            })
        except Exception as e:
            print(f"[Daily GMI] Error processing index {name}: {e}")

    return results


def _compute_index_telemetry_payload(
    benchmark_sym: str,
    period: str = "400d",
    zweig_market: str = None,
    include_ftd: bool = False,
    settled_us_session: bool = False,
) -> dict:
    """Shared DRY telemetry computation for broad benchmarks (^CRSLDX, ^GSPC, ^IXIC)."""
    idx_df = yf.download(benchmark_sym, period=period, progress=False)
    if idx_df is None or idx_df.empty:
        return None
    if isinstance(idx_df.columns, pd.MultiIndex):
        lvl1 = idx_df.columns.get_level_values(1)
        idx_df = idx_df.xs(benchmark_sym, level=1, axis=1) if benchmark_sym in lvl1 else idx_df.xs(benchmark_sym, level=0, axis=1)

    if idx_df.index.tz is not None:
        idx_df.index = idx_df.index.tz_localize(None)
    idx_df.index = pd.to_datetime(idx_df.index).normalize()

    if settled_us_session:
        try:
            from price_history_manager import get_latest_completed_session_date
            last_us = pd.Timestamp(get_latest_completed_session_date('US'))
            idx_settled = idx_df.loc[idx_df.index <= last_us]
            if not idx_settled.empty:
                idx_df = idx_settled
        except Exception:
            pass

    dd_count, dd_dates, dd_details = calculate_distribution_days(idx_df, return_details=True)
    haber_21 = check_haber_21sma_rule(idx_df)
    recapture_seq = detect_20ema_recapture_and_sequence(idx_df)

    ftd_detected, ftd_date, ftd_details = (False, None, {})
    if include_ftd:
        ftd_detected, ftd_date, ftd_details = detect_follow_through_day(idx_df, return_details=True)

    zweig_tape = None
    if zweig_market:
        try:
            from breadth_engine import get_zweig_tape_summary
            zweig_tape = get_zweig_tape_summary(market=zweig_market, lookback=120)
        except Exception:
            zweig_tape = None

    std_idx = standardize_columns(idx_df)
    c_now = float(std_idx['close'].iloc[-1])
    sma50 = float(std_idx['close'].rolling(50).mean().iloc[-1]) if len(std_idx) >= 50 else c_now
    dist_sma50 = ((c_now - sma50) / sma50) * 100.0 if sma50 > 0 else 0.0

    cluster_detected = dd_details.get('cluster_detected', False)
    status_label, status_color, status_emoji = get_market_regime_label(
        dd_count, c_now, sma50, dd_cluster=cluster_detected
    )

    exposure_plan = get_progressive_exposure_plan(
        status_label,
        dd_count,
        dd_cluster=cluster_detected,
        ftd_detected=ftd_detected if include_ftd else False,
        zweig_tape=zweig_tape,
    )

    out = {
        'close': c_now,
        'sma50': sma50,
        'dist_sma50': dist_sma50,
        'sma21': haber_21.get('sma21', 0.0),
        'pct_diff_sma21': haber_21.get('pct_diff_sma21', 0.0),
        'consecutive_closes_below_21': haber_21.get('consecutive_closes', 0),
        'haber_triggered': haber_21.get('triggered', False),
        'dd_count': dd_count,
        'standard_dd_count': dd_details.get('standard_dd_count', 0),
        'churning_dd_count': dd_details.get('churning_dd_count', 0),
        'hammer_acc_count': dd_details.get('hammer_acc_count', 0),
        'cluster_detected': cluster_detected,
        'cluster_count_7d': dd_details.get('cluster_count_7d', 0),
        'latest_dcr': dd_details.get('latest_dcr', 0.5),
        'latest_midpoint_verdict': dd_details.get('latest_midpoint_verdict', 'Neutral'),
        'status_label': status_label,
        'status_color': status_color,
        'status_emoji': status_emoji,
        'exposure_plan': exposure_plan,
        'dd_details': dd_details,
        'dd_dates': dd_dates,
        'dd_ledger': dd_details.get('dd_ledger', []),
        'haber_21': haber_21,
        'recapture_sequence': recapture_seq,
        'secular_bull': recapture_seq.get('secular_bull', True),
        'secular_label': recapture_seq.get('secular_label', "Secular Bull"),
        'secular_color': recapture_seq.get('secular_color', "#10b981"),
        'dist_200_sma_pct': recapture_seq.get('dist_200_sma_pct', 0.0),
        'recapture_active': recapture_seq.get('recapture_active', False),
        'recapture_fresh': recapture_seq.get('recapture_fresh', False),
        'current_step_num': recapture_seq.get('current_step_num', 1),
        'current_step_name': recapture_seq.get('current_step_name', "Base"),
        'sequence_summary': recapture_seq.get('sequence_summary', ""),
        'step_badge_color': recapture_seq.get('step_badge_color', "#94a3b8"),
        'benchmark_df': std_idx,
    }
    if include_ftd:
        out.update({
            'ftd_detected': ftd_detected,
            'ftd_date': ftd_date,
            'ftd_details': ftd_details,
            'zweig_tape': zweig_tape,
        })
    return out


@st.cache_data(ttl=300)
def fetch_nifty500_telemetry():
    """
    Fetches real-time / daily telemetry for Nifty 500 (^CRSLDX) directly.
    Guarantees that Cockpit always displays accurate Nifty 500 DDs,
    Haber 21 SMA Buffer, DCR Midpoint, Acute Cluster Alerts, 200 SMA Secular Status,
    and 20 EMA Recapture Tactical Signals independent of universe processing.
    """
    try:
        return _compute_index_telemetry_payload('^CRSLDX', period="250d", zweig_market='IN', include_ftd=True, settled_us_session=False)
    except Exception as e:
        print(f"[Nifty 500 Telemetry] Error: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_nasdaq_telemetry():
    """
    Fetches settled daily telemetry for Nasdaq Composite (^IXIC) directly,
    mirroring fetch_nifty500_telemetry() cleanly for the US Cockpit tab.
    """
    try:
        return _compute_index_telemetry_payload('^IXIC', period="400d", zweig_market='US_IXIC', include_ftd=True, settled_us_session=True)
    except Exception as e:
        print(f"[Nasdaq Telemetry] Error: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_nifty50_telemetry():
    """
    Fetches real-time / daily telemetry for Nifty 50 (^NSEI) directly.
    Guarantees that Cockpit, Market Breadth, and Market Regime always display
    accurate, synchronized Nifty 50 DDs, Haber 21 SMA Buffer, and DCR Midpoint metrics.
    """
    try:
        idx_n50 = yf.download('^NSEI', period="250d", progress=False)
        if idx_n50.empty:
            return None
            
        dd_count, dd_dates, dd_details = calculate_distribution_days(idx_n50, return_details=True)
        haber_21 = check_haber_21sma_rule(idx_n50)
        recapture_seq = detect_20ema_recapture_and_sequence(idx_n50)
        
        std_idx = standardize_columns(idx_n50)
        c_now = float(std_idx['close'].iloc[-1])
        sma50 = float(std_idx['close'].rolling(50).mean().iloc[-1]) if len(std_idx) >= 50 else c_now
        dist_sma50 = ((c_now - sma50) / sma50) * 100.0 if sma50 > 0 else 0.0
        
        cluster_detected = dd_details.get('cluster_detected', False)
        status_label, status_color, status_emoji = get_market_regime_label(
            dd_count, c_now, sma50, dd_cluster=cluster_detected
        )
        
        return {
            'close': c_now,
            'sma50': sma50,
            'dist_sma50': dist_sma50,
            'sma21': haber_21.get('sma21', 0.0),
            'pct_diff_sma21': haber_21.get('pct_diff_sma21', 0.0),
            'consecutive_closes_below_21': haber_21.get('consecutive_closes', 0),
            'haber_triggered': haber_21.get('triggered', False),
            'dd_count': dd_count,
            'standard_dd_count': dd_details.get('standard_dd_count', 0),
            'churning_dd_count': dd_details.get('churning_dd_count', 0),
            'hammer_acc_count': dd_details.get('hammer_acc_count', 0),
            'cluster_detected': cluster_detected,
            'cluster_count_7d': dd_details.get('cluster_count_7d', 0),
            'latest_dcr': dd_details.get('latest_dcr', 0.5),
            'latest_midpoint_verdict': dd_details.get('latest_midpoint_verdict', 'Neutral'),
            'status_label': status_label,
            'status_color': status_color,
            'status_emoji': status_emoji,
            'dd_details': dd_details,
            'dd_dates': dd_dates,
            'dd_ledger': dd_details.get('dd_ledger', []),
            'haber_21': haber_21,
            'recapture_sequence': recapture_seq,
            'secular_bull': recapture_seq.get('secular_bull', True),
            'secular_label': recapture_seq.get('secular_label', "Secular Bull"),
            'secular_color': recapture_seq.get('secular_color', "#10b981"),
            'dist_200_sma_pct': recapture_seq.get('dist_200_sma_pct', 0.0),
            'recapture_active': recapture_seq.get('recapture_active', False),
            'recapture_fresh': recapture_seq.get('recapture_fresh', False),
            'current_step_num': recapture_seq.get('current_step_num', 1),
            'current_step_name': recapture_seq.get('current_step_name', "Base"),
            'sequence_summary': recapture_seq.get('sequence_summary', ""),
            'step_badge_color': recapture_seq.get('step_badge_color', "#94a3b8")
        }
    except Exception as e:
        print(f"[Nifty 50 Telemetry] Error: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_sp500_telemetry():
    """
    Fetches real-time / daily telemetry for S&P 500 (^GSPC) directly.
    Provides complete parity for the US market: Distribution days, Stall days,
    Haber 21 SMA Buffer, 200 SMA Secular Status, and 20 EMA Recapture Buy Signals.
    """
    try:
        idx_spx = yf.download('^GSPC', period="250d", progress=False)
        if idx_spx.empty:
            return None
            
        dd_count, dd_dates, dd_details = calculate_distribution_days(idx_spx, return_details=True)
        haber_21 = check_haber_21sma_rule(idx_spx)
        recapture_seq = detect_20ema_recapture_and_sequence(idx_spx)
        
        std_idx = standardize_columns(idx_spx)
        c_now = float(std_idx['close'].iloc[-1])
        sma50 = float(std_idx['close'].rolling(50).mean().iloc[-1]) if len(std_idx) >= 50 else c_now
        dist_sma50 = ((c_now - sma50) / sma50) * 100.0 if sma50 > 0 else 0.0
        
        cluster_detected = dd_details.get('cluster_detected', False)
        status_label, status_color, status_emoji = get_market_regime_label(
            dd_count, c_now, sma50, dd_cluster=cluster_detected
        )
        exposure_plan = get_progressive_exposure_plan(
            status_label, dd_count, dd_cluster=cluster_detected
        )
        
        return {
            'close': c_now,
            'sma50': sma50,
            'dist_sma50': dist_sma50,
            'sma21': haber_21.get('sma21', 0.0),
            'pct_diff_sma21': haber_21.get('pct_diff_sma21', 0.0),
            'consecutive_closes_below_21': haber_21.get('consecutive_closes', 0),
            'haber_triggered': haber_21.get('triggered', False),
            'dd_count': dd_count,
            'standard_dd_count': dd_details.get('standard_dd_count', 0),
            'churning_dd_count': dd_details.get('churning_dd_count', 0),
            'hammer_acc_count': dd_details.get('hammer_acc_count', 0),
            'cluster_detected': cluster_detected,
            'cluster_count_7d': dd_details.get('cluster_count_7d', 0),
            'latest_dcr': dd_details.get('latest_dcr', 0.5),
            'latest_midpoint_verdict': dd_details.get('latest_midpoint_verdict', 'Neutral'),
            'status_label': status_label,
            'status_color': status_color,
            'status_emoji': status_emoji,
            'exposure_plan': exposure_plan,
            'dd_details': dd_details,
            'dd_dates': dd_dates,
            'dd_ledger': dd_details.get('dd_ledger', []),
            'haber_21': haber_21,
            'recapture_sequence': recapture_seq,
            'secular_bull': recapture_seq.get('secular_bull', True),
            'secular_label': recapture_seq.get('secular_label', "Secular Bull"),
            'secular_color': recapture_seq.get('secular_color', "#10b981"),
            'dist_200_sma_pct': recapture_seq.get('dist_200_sma_pct', 0.0),
            'recapture_active': recapture_seq.get('recapture_active', False),
            'recapture_fresh': recapture_seq.get('recapture_fresh', False),
            'current_step_num': recapture_seq.get('current_step_num', 1),
            'current_step_name': recapture_seq.get('current_step_name', "Base"),
            'sequence_summary': recapture_seq.get('sequence_summary', ""),
            'step_badge_color': recapture_seq.get('step_badge_color', "#94a3b8")
        }
    except Exception as e:
        print(f"[S&P 500 Telemetry] Error: {e}")
        return None


# =============================================================================
# HOWARD MARKS: CYCLE POSITIONING & CORPORATE CREDIT HEALTH ENGINE
# =============================================================================

def _fetch_ticker_series(symbol: str, period: str = "1y") -> pd.Series:
    """Fetches clean close price series for a single ticker via yf.Ticker(sym).history()."""
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period)
        if hist is not None and not hist.empty and 'Close' in hist.columns:
            s = hist['Close'].dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            if len(s) >= 15:
                return s
    except Exception as e:
        print(f"[Credit Engine] Error fetching {symbol} via Ticker: {e}")
    # Backup: try yf.download as secondary attempt if Ticker failed
    try:
        data = yf.download(symbol, period=period, progress=False)
        std = standardize_columns(data)
        if not std.empty and 'close' in std.columns:
            s = std['close'].dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            if len(s) >= 15:
                return s
    except Exception as e:
        print(f"[Credit Engine] Backup download failed for {symbol}: {e}")
    return pd.Series(dtype=float)


@st.cache_data(ttl=600)
def compute_corporate_credit_health(market: str = "US", period: str = "1y") -> dict:
    """
    Computes corporate credit health and credit window status for US or India.
    
    US:
      - HYG / IEI (High Yield Corporate vs Intermediate Treasuries)
      - Fallback: JNK / IEF
    India:
      - EBBETF0430.NS / GSEC10YEAR.NS (Bharat Bond AAA Corporate vs 10Y Benchmark G-Sec)
      - ^NSEBANK / ^CRSLDX (Bank & Financial Vitality relative to Nifty 500)
      
    Returns:
      Dictionary containing current ratio, moving averages, 1M momentum, status ('OPEN', 'NARROWING', 'SLAMMED_SHUT'),
      institutional label, credit score (0-100), and history dataframe.
    """
    try:
        if market == "US":
            hyg = _fetch_ticker_series('HYG', period=period)
            if hyg.empty:
                hyg = _fetch_ticker_series('JNK', period=period)
            iei = _fetch_ticker_series('IEI', period=period)
            if iei.empty:
                iei = _fetch_ticker_series('IEF', period=period)
                
            if hyg.empty or iei.empty:
                return _fallback_credit_health(market)
                
            common_idx = hyg.index.intersection(iei.index)
            if len(common_idx) < 25:
                return _fallback_credit_health(market)
                
            credit_ratio = (hyg.loc[common_idx] / iei.loc[common_idx]).dropna()
        else:
            # India Market
            ebbetf = _fetch_ticker_series('EBBETF0430.NS', period=period)
            gsec = _fetch_ticker_series('GSEC10YEAR.NS', period=period)
            bank = _fetch_ticker_series('^NSEBANK', period=period)
            nifty500 = _fetch_ticker_series('^CRSLDX', period=period)
            
            has_debt = (not ebbetf.empty) and (not gsec.empty)
            has_bank = (not bank.empty) and (not nifty500.empty)
            
            if not has_debt and not has_bank:
                return _fallback_credit_health(market)
                
            if has_debt and has_bank:
                common_d = ebbetf.index.intersection(gsec.index)
                debt_ratio = ebbetf.loc[common_d] / gsec.loc[common_d]
                
                common_b = bank.index.intersection(nifty500.index)
                bank_rs = bank.loc[common_b] / nifty500.loc[common_b]
                
                common_all = debt_ratio.index.intersection(bank_rs.index)
                if len(common_all) >= 25:
                    d_sub = debt_ratio.loc[common_all]
                    b_sub = bank_rs.loc[common_all]
                    norm_debt = d_sub / d_sub.iloc[0]
                    norm_bank = b_sub / b_sub.iloc[0]
                    credit_ratio = (norm_debt * 0.4) + (norm_bank * 0.6)
                else:
                    norm_bank = bank_rs / bank_rs.iloc[0]
                    credit_ratio = norm_bank
            elif has_bank:
                common_b = bank.index.intersection(nifty500.index)
                bank_rs = bank.loc[common_b] / nifty500.loc[common_b]
                credit_ratio = bank_rs / bank_rs.iloc[0]
            else:
                common_d = ebbetf.index.intersection(gsec.index)
                debt_ratio = ebbetf.loc[common_d] / gsec.loc[common_d]
                credit_ratio = debt_ratio / debt_ratio.iloc[0]
                
        credit_ratio = credit_ratio.dropna()
        if len(credit_ratio) < 25:
            return _fallback_credit_health(market)
            
        sma50 = credit_ratio.rolling(50, min_periods=10).mean()
        sma200 = credit_ratio.rolling(200, min_periods=20).mean()
        
        curr = float(credit_ratio.iloc[-1])
        s50 = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else curr
        s200 = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else curr
        
        # 1-month momentum
        lookback = min(21, len(credit_ratio) - 1)
        prev_val = float(credit_ratio.iloc[-lookback])
        mom_1m = float(((curr / prev_val) - 1.0) * 100.0) if prev_val > 0 else 0.0
        
        # 3-month momentum
        lookback3 = min(63, len(credit_ratio) - 1)
        prev_val3 = float(credit_ratio.iloc[-lookback3])
        mom_3m = float(((curr / prev_val3) - 1.0) * 100.0) if prev_val3 > 0 else 0.0
        
        # Determine Credit Window Status
        if curr >= s50 and mom_1m >= -0.75:
            status = "OPEN"
            label = "🟢 Credit Window Wide Open (Liquid & Risk-Seeking)"
            score = 85.0
            badge_color = "#10b981"
        elif curr < (s200 * 0.96) or mom_1m <= -4.0 or (curr < s50 and mom_3m <= -5.0):
            status = "SLAMMED_SHUT"
            label = "🔴 Credit Window Slammed Shut (Liquidity Freeze / Risk-Off)"
            score = 20.0
            badge_color = "#ef4444"
        else:
            status = "NARROWING"
            label = "🟡 Credit Window Narrowing (Caution / Tightening Terms)"
            score = 55.0
            badge_color = "#f59e0b"
            
        # Build plot dataframe
        history_df = pd.DataFrame({
            'date': credit_ratio.index,
            'credit_ratio': credit_ratio.values,
            'sma50': sma50.values,
            'sma200': sma200.values
        }).set_index('date')
        
        return {
            'market': market,
            'current_ratio': curr,
            'sma50': s50,
            'sma200': s200,
            'mom_1m': mom_1m,
            'mom_3m': mom_3m,
            'status': status,
            'label': label,
            'credit_score': score,
            'badge_color': badge_color,
            'history_df': history_df
        }
    except Exception as e:
        print(f"[Credit Health Engine] Error for {market}: {e}")
        return _fallback_credit_health(market)


def _fallback_credit_health(market: str = "US") -> dict:
    """Safe fallback credit health when external data fails."""
    # Synthesize a minimal baseline so page charts never crash
    base_dates = pd.date_range(end=datetime.now(), periods=50, freq='B')
    base_df = pd.DataFrame({
        'credit_ratio': [1.0] * 50,
        'sma50': [1.0] * 50,
        'sma200': [1.0] * 50
    }, index=base_dates)
    
    return {
        'market': market,
        'current_ratio': 1.0,
        'sma50': 1.0,
        'sma200': 1.0,
        'mom_1m': 0.0,
        'mom_3m': 0.0,
        'status': "NARROWING",
        'label': "🟡 Credit Window Neutral / Data Cached",
        'credit_score': 50.0,
        'badge_color': "#f59e0b",
        'history_df': base_df
    }


@st.cache_data(ttl=600)
def fetch_benchmark_index_series(symbol: str, period: str = "1y") -> pd.Series:
    """Fetches clean close price series for a benchmark index (e.g. ^GSPC, ^NDX, ^CRSLDX, ^NSEI)."""
    return _fetch_ticker_series(symbol, period=period)


def evaluate_credit_equity_divergence(credit_series: pd.Series, index_series: pd.Series) -> dict:
    """
    Evaluates Howard Marks Credit vs Equity lead-lag divergence.
    Detects:
    1. Bearish Divergence: Index buoyant (above 50 SMA) but Credit Ratio breaking down below 50 SMA.
    2. Bullish Divergence: Index depressed below 50 SMA but Credit Ratio expanding above 50 SMA with positive momentum.
    3. Confirmed Expansion: Both above 50 SMA.
    4. Confirmed Contraction: Both below 50 SMA.
    """
    if credit_series is None or index_series is None or credit_series.empty or index_series.empty:
        return {
            'status': 'NEUTRAL',
            'badge': '⚪ Pure Credit Mode',
            'label': 'Credit telemetry displayed without index overlay.',
            'color': '#94a3b8'
        }
        
    common = credit_series.index.intersection(index_series.index)
    if len(common) < 25:
        return {
            'status': 'NEUTRAL',
            'badge': '⚪ Neutral Synchronization',
            'label': 'Insufficient overlapping data points.',
            'color': '#94a3b8'
        }
        
    c = credit_series.loc[common]
    idx = index_series.loc[common]
    c_sma50 = c.rolling(50, min_periods=10).mean()
    idx_sma50 = idx.rolling(50, min_periods=10).mean()
    
    c_curr = float(c.iloc[-1])
    c_s50 = float(c_sma50.iloc[-1]) if not pd.isna(c_sma50.iloc[-1]) else c_curr
    idx_curr = float(idx.iloc[-1])
    idx_s50 = float(idx_sma50.iloc[-1]) if not pd.isna(idx_sma50.iloc[-1]) else idx_curr
    
    lookback = min(21, len(c) - 1)
    c_mom20 = float(((c_curr / c.iloc[-lookback]) - 1.0) * 100.0) if c.iloc[-lookback] > 0 else 0.0
    
    c_bull = c_curr >= c_s50
    idx_bull = idx_curr >= idx_s50
    
    if idx_bull and not c_bull and c_mom20 <= -0.5:
        return {
            'status': 'BEARISH_DIVERGENCE',
            'badge': '⚠️ Bearish Credit Divergence (Credit Leading Down)',
            'label': '⚠️ BEARISH CREDIT DIVERGENCE: Equity Index is buoyant above 50 SMA, but Corporate Credit Ratio has broken below its 50 SMA with negative momentum. Howard Marks early warning: smart debt money is turning defensive.',
            'color': '#ef4444'
        }
    elif not idx_bull and c_bull and c_mom20 >= +0.5:
        return {
            'status': 'BULLISH_DIVERGENCE',
            'badge': '💎 Bullish Credit Divergence (Credit Leading Up)',
            'label': '💎 BULLISH CREDIT DIVERGENCE: Corporate Credit Ratio is expanding above its 50 SMA while Equity Index is lagging below 50 SMA. Credit liquidity is resilient, signaling an impending market turnaround.',
            'color': '#10b981'
        }
    elif c_bull and idx_bull:
        return {
            'status': 'CONFIRMED_EXPANSION',
            'badge': '🟢 Confirmed Expansion (Both Trending Above 50 SMA)',
            'label': '🟢 CONFIRMED EXPANSION: Both Corporate Credit and Equity Benchmark are trending above their 50-day moving averages. Liquidity tailwinds supportive.',
            'color': '#10b981'
        }
    elif not c_bull and not idx_bull:
        return {
            'status': 'CONFIRMED_CONTRACTION',
            'badge': '🔴 Risk-Off Synchronization (Both Below 50 SMA)',
            'label': '🔴 RISK-OFF SYNCHRONIZATION: Both Corporate Credit and Equity Index are trading below their 50-day moving averages. Capital preservation stance warranted.',
            'color': '#ef4444'
        }
    else:
        return {
            'status': 'MIXED',
            'badge': '⚖️ Mixed Rotation / Transitional Phase',
            'label': '⚖️ MIXED ROTATION: Credit and Equity trends are rotating in transition.',
            'color': '#f59e0b'
        }


def fetch_latest_vix(symbol: str = '^INDIAVIX') -> float:
    """Fetches the latest VIX reading safely."""
    try:
        vix = yf.download(symbol, period="5d", progress=False)
        std_vix = standardize_columns(vix)
        if not std_vix.empty and 'close' in std_vix.columns:
            return float(std_vix['close'].dropna().iloc[-1])
        return 14.5
    except Exception:
        return 14.5


def fetch_market_valuation_percentile(market: str = 'IN') -> float:
    """
    Computes an indicative 10-year valuation multiple percentile (0-100) for Nifty 50 or S&P 500.
    Uses index/ETF trailing P/E multiples and multi-year structural trend.
    In bull market peaks, multiples trade in the 80th-95th percentile.
    In fair value / mid-cycle, multiples trade around 50th-65th percentile.
    In bear troughs, multiples trade in the 10th-25th percentile.
    """
    try:
        sym = 'NIFTYBEES.NS' if market == 'IN' else 'SPY'
        t = yf.Ticker(sym)
        pe = None
        try:
            pe = t.info.get('trailingPE')
        except Exception:
            pass
            
        if pe and 5.0 < pe < 60.0:
            if market == 'IN':
                # Nifty 50 historical 10-year P/E bounds: 14.5 (crisis low) to 25.5 (peak), median ~19.0
                pe_percentile = ((pe - 14.5) / (25.5 - 14.5)) * 100.0
            else:
                # S&P 500 historical 10-year P/E bounds: 15.0 (crisis low) to 27.0 (peak), median ~19.5
                pe_percentile = ((pe - 15.0) / (27.0 - 15.0)) * 100.0
            return float(np.clip(pe_percentile, 15.0, 95.0))
            
        # Secondary fallback: 3-year structural cycle range
        idx_sym = '^NSEI' if market == 'IN' else '^GSPC'
        t_idx = yf.Ticker(idx_sym)
        hist = t_idx.history(period='3y')
        if hist is not None and not hist.empty and 'Close' in hist.columns:
            closes = hist['Close'].dropna()
            c_now = float(closes.iloc[-1])
            l_3y = float(closes.min())
            h_3y = float(closes.max())
            if h_3y > l_3y:
                pos_3y = ((c_now - l_3y) / (h_3y - l_3y)) * 100.0
                return float(np.clip(pos_3y, 15.0, 95.0))
    except Exception as e:
        print(f"[Valuation Engine] Error computing valuation for {market}: {e}")
        
    return 60.0 if market == 'IN' else 75.0


def compute_howard_marks_market_temperature(market: str = 'IN', credit_data: dict = None, breadth_data: dict = None) -> dict:
    """
    Synthesizes Valuation (25%), Credit Spread (30%), Breadth/Regime (25%), and Sentiment/VIX (20%)
    into Howard Marks' Market Temperature Score (MTS) and Aggressiveness vs. Defensiveness Dial.
    
    100% Additive: Reads O'Neil distribution day count and Haber deck in a read-only manner.
    """
    # 1. Credit Component (0 = Distressed/Frozen, 100 = Complacent/Wide Open)
    credit_score = credit_data.get('credit_score', 50.0) if credit_data else 50.0
    
    # 2. Volatility Component
    vix_sym = '^INDIAVIX' if market == 'IN' else '^VIX'
    vix_base = 13.0 if market == 'IN' else 12.0
    vix_val = fetch_latest_vix(vix_sym)
    # Low VIX = Complacency (High Danger Temp), High VIX = Panic (Low Danger Temp / High Opportunity)
    sentiment_temp = float(np.clip(100.0 - (vix_val - vix_base) * (100.0 / 25.0), 5.0, 95.0))
    
    # 3. Breadth & O'Neil Distribution Component (READ-ONLY)
    dd_count = 3
    breadth_pct_200 = 55.0
    if breadth_data:
        dd_count = breadth_data.get('dd_count', 3)
        breadth_pct_200 = breadth_data.get('pct_above_200sma', 55.0)
    breadth_temp = float((breadth_pct_200 * 0.6) + (100.0 - (dd_count * 12.5)) * 0.4)
    breadth_temp = float(np.clip(breadth_temp, 5.0, 95.0))
    
    # 4. Valuation Component
    valuation_temp = fetch_market_valuation_percentile(market)
    
    # Composite Danger Score (0 = Frigid / Bargain, 100 = Boiling / Peak Danger)
    danger_temp = float((0.30 * credit_score) + (0.25 * valuation_temp) + (0.25 * breadth_temp) + (0.20 * sentiment_temp))
    danger_temp = float(np.clip(danger_temp, 0.0, 100.0))
    
    # Aggressiveness Dial (Inverse to Danger Temperature)
    aggressiveness_dial = float(np.clip(100.0 - danger_temp, 0.0, 100.0))
    
    # Exposure Mapping based on Howard Marks Chapter 14
    if aggressiveness_dial >= 65:
        stance = "AGGRESSIVE_EXPANSION"
        stance_title = "🚀 Aggressive Expansion (Offense Stance)"
        exposure = "100% - 120% (Margin Enabled)"
        max_positions = 8
        stops = "Loose Trailing (50 EMA / Prior Swing Lows)"
        twin_risk = "RISK OF MISSING OPPORTUNITY DOMINATES"
        theme_color = "#10b981"
        action_summary = "Odds are heavily in your favor. Deploy capital aggressively, run concentrated winners, and allow multi-quarter trends to compound."
    elif aggressiveness_dial >= 45:
        stance = "MEASURED_GROWTH"
        stance_title = "⚖️ Measured Growth (Constructive Balance)"
        exposure = "75% - 100% Gross Exposure"
        max_positions = 6
        stops = "Standard CANSLIM 7-8% Stops"
        twin_risk = "BALANCED TWIN RISKS"
        theme_color = "#38bdf8"
        action_summary = "Normal constructive bull environment. Take standard pivot breakouts, hold top leaders, and cut losers decisively at 7-8%."
    elif aggressiveness_dial >= 25:
        stance = "DEFENSIVE_PRESERVATION"
        stance_title = "🛡️ Defensive Preservation (Caution Stance)"
        exposure = "30% - 60% Gross Exposure"
        max_positions = 4
        stops = "Tight Stops (Breakeven / 4% Cutoffs)"
        twin_risk = "RISK OF LOSING MONEY DOMINATES"
        theme_color = "#f59e0b"
        action_summary = "Market temperature is elevated or credit is tightening. Raise 40%+ cash, move trailing stops to breakeven, and resist chasing late bases."
    else:
        stance = "MAXIMUM_CAPITAL_PRESERVATION"
        stance_title = "🛑 Maximum Capital Preservation (Defense First)"
        exposure = "0% - 20% (80%+ Cash / Ultra-Short Debt)"
        max_positions = 2
        stops = "Immediate Breakeven / No New Breakouts"
        twin_risk = "CAPITAL EXTINCTION DEFENSE"
        theme_color = "#ef4444"
        action_summary = "Credit window is slammed shut or extreme euphoric climax detected. Protect capital above all else. Wait for a new market cycle launchpad."
        
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    return {
        'date': date_str,
        'market': market,
        'danger_temperature': danger_temp,
        'aggressiveness_dial': aggressiveness_dial,
        'stance': stance,
        'stance_title': stance_title,
        'target_gross_exposure': exposure,
        'max_positions': max_positions,
        'stop_protocol': stops,
        'twin_risk_priority': twin_risk,
        'theme_color': theme_color,
        'action_summary': action_summary,
        'sub_scores': {
            'credit': credit_score,
            'valuation': valuation_temp,
            'breadth': breadth_temp,
            'sentiment': sentiment_temp
        }
    }


def get_default_20_factor_checklist(market: str = 'IN', auto_metrics: dict = None) -> list:
    """
    Returns the 20 parameters from Howard Marks Chapter 13 formatted for the interactive checklist.
    Pre-populates auto-detected values based on real-time market telemetry.
    """
    # Helper to check if auto metric indicates danger
    auto_metrics = auto_metrics or {}
    vix_val = auto_metrics.get('vix', 14.5)
    credit_status = auto_metrics.get('credit_status', 'OPEN')
    dd_count = auto_metrics.get('dd_count', 3)
    val_temp = auto_metrics.get('valuation_temp', 60.0)
    
    items = [
        {
            'parameter_key': 'economy',
            'parameter_name': 'Economy',
            'category': 'Macroeconomic Backdrop',
            'danger_desc': 'Vibrant, firing on all cylinders, high capacity utilization',
            'bargain_desc': 'Sluggish, stagnant, uninspiring, contracting',
            'default_side': 'danger' if val_temp >= 40 else 'bargain'
        },
        {
            'parameter_key': 'outlook',
            'parameter_name': 'Macro Outlook',
            'category': 'Macroeconomic Backdrop',
            'danger_desc': 'Positive, blue skies projected indefinitely, no clouds in sight',
            'bargain_desc': 'Negative, structural headwinds cited, universal despair',
            'default_side': 'danger' if val_temp >= 45 else 'bargain'
        },
        {
            'parameter_key': 'lenders',
            'parameter_name': 'Lenders Appetite',
            'category': 'Credit & Capital Markets',
            'danger_desc': 'Eager, competing fiercely to lend, expanding loan books',
            'bargain_desc': 'Reticent, terrified of bad debts, refusing refinancing',
            'default_side': 'danger' if credit_status == 'OPEN' else 'bargain'
        },
        {
            'parameter_key': 'capital_markets',
            'parameter_name': 'Capital Markets',
            'category': 'Credit & Capital Markets',
            'danger_desc': 'Loose, accommodating, funding speculative ventures easily',
            'bargain_desc': 'Tight, frozen, closed to all but supreme credit quality',
            'default_side': 'danger' if credit_status == 'OPEN' else 'bargain'
        },
        {
            'parameter_key': 'capital_supply',
            'parameter_name': 'Capital Supply',
            'category': 'Credit & Capital Markets',
            'danger_desc': 'Plentiful, cash chasing too few deals, easy liquidity',
            'bargain_desc': 'Scarce, capital starved, forced asset liquidations',
            'default_side': 'danger' if credit_status != 'SLAMMED_SHUT' else 'bargain'
        },
        {
            'parameter_key': 'lending_terms',
            'parameter_name': 'Lending Terms',
            'category': 'Credit & Capital Markets',
            'danger_desc': 'Easy, covenant-lite, borrower-friendly, low coverage demanded',
            'bargain_desc': 'Restrictive, punitive covenants, lender holds all cards',
            'default_side': 'danger' if credit_status == 'OPEN' else 'bargain'
        },
        {
            'parameter_key': 'interest_rates',
            'parameter_name': 'Interest Rates',
            'category': 'Credit & Capital Markets',
            'danger_desc': 'Low or near cyclical inflection point',
            'bargain_desc': 'High, peak policy rates or aggressive central bank easing begun',
            'default_side': 'danger' if val_temp > 75 else 'bargain'
        },
        {
            'parameter_key': 'yield_spreads',
            'parameter_name': 'Yield Spreads',
            'category': 'Credit & Capital Markets',
            'danger_desc': 'Narrow, skimpy, failing to compensate for credit risk',
            'bargain_desc': 'Wide, generous, pricing in catastrophic default waves',
            'default_side': 'danger' if credit_status != 'SLAMMED_SHUT' else 'bargain'
        },
        {
            'parameter_key': 'investors_psychology',
            'parameter_name': 'Investor Psychology',
            'category': 'Psychology & Sentiment',
            'danger_desc': 'Optimistic, sanguine, eager to buy dips, FOMO rampant',
            'bargain_desc': 'Pessimistic, distressed, panic selling, disgusted with stocks',
            'default_side': 'danger' if vix_val < 18.0 else 'bargain'
        },
        {
            'parameter_key': 'asset_owners',
            'parameter_name': 'Asset Owners',
            'category': 'Psychology & Sentiment',
            'danger_desc': 'Delighted to hold, expecting compounding indefinitely',
            'bargain_desc': 'Rushing for the exits, forced margin liquidation',
            'default_side': 'danger' if val_temp >= 45 else 'bargain'
        },
        {
            'parameter_key': 'seller_presence',
            'parameter_name': 'Sellers Presence',
            'category': 'Supply & Demand',
            'danger_desc': 'Few willing sellers, buyers begging for blocks',
            'bargain_desc': 'Many motivated sellers, zero liquidity or bids',
            'default_side': 'danger' if dd_count < 5 else 'bargain'
        },
        {
            'parameter_key': 'market_attention',
            'parameter_name': 'Market Crowd',
            'category': 'Psychology & Sentiment',
            'danger_desc': 'Crowded, casual cocktail-party stock bragging',
            'bargain_desc': 'Starved for attention, finance pages ignored',
            'default_side': 'danger' if val_temp >= 45 else 'bargain'
        },
        {
            'parameter_key': 'funds_raising',
            'parameter_name': 'Fund Raising',
            'category': 'Market Infrastructure',
            'danger_desc': 'Hard to gain entry, massive IPO / NFO oversubscriptions daily',
            'bargain_desc': 'Redemptions, fund closures, only elite can raise capital',
            'default_side': 'danger' if val_temp >= 45 else 'bargain'
        },
        {
            'parameter_key': 'recent_performance',
            'parameter_name': 'Recent Performance',
            'category': 'Valuation & Performance',
            'danger_desc': 'Strong multi-year run, paper millionaires created',
            'bargain_desc': 'Weak, brutal multi-month drawdown, deep paper losses',
            'default_side': 'danger' if val_temp >= 40 else 'bargain'
        },
        {
            'parameter_key': 'asset_prices',
            'parameter_name': 'Asset Prices',
            'category': 'Valuation & Performance',
            'danger_desc': 'Historically high relative to book value, sales, and intrinsic cash',
            'bargain_desc': 'Low, trading near or below liquidation / replacement value',
            'default_side': 'danger' if val_temp >= 45 else 'bargain'
        },
        {
            'parameter_key': 'prospective_returns',
            'parameter_name': 'Prospective Returns',
            'category': 'Valuation & Performance',
            'danger_desc': 'Skimpy forward returns (compressed equity risk premium)',
            'bargain_desc': 'Generous prospective returns (asymmetric multi-bagger potential)',
            'default_side': 'danger' if val_temp >= 45 else 'bargain'
        },
        {
            'parameter_key': 'market_risk',
            'parameter_name': 'Market Risk Perception',
            'category': 'Risk Attitudes',
            'danger_desc': 'High actual risk (perceived by crowd as completely risk-free)',
            'bargain_desc': 'Low actual risk (perceived by crowd as toxic and uninvestable)',
            'default_side': 'danger' if val_temp >= 45 else 'bargain'
        },
        {
            'parameter_key': 'popular_qualities',
            'parameter_name': 'Popular Behaviors',
            'category': 'Behavioral Norms',
            'danger_desc': 'Aggressiveness, broad reach, high beta, speculative enthusiasm',
            'bargain_desc': 'Extreme caution, rigorous discipline, defensive conservatism',
            'default_side': 'danger' if vix_val < 18.0 else 'bargain'
        },
        {
            'parameter_key': 'right_qualities',
            'parameter_name': 'Right Behaviors for Now',
            'category': 'Behavioral Norms',
            'danger_desc': 'Caution, discipline, high selectivity, capital preservation',
            'bargain_desc': 'Aggressiveness, courage, nerve to buy when blood is in the streets',
            'default_side': 'danger' if val_temp >= 45 else 'bargain'
        },
        {
            'parameter_key': 'available_mistakes',
            'parameter_name': 'Costly Mistake Available',
            'category': 'Twin Risks',
            'danger_desc': 'Buying too much, paying up, taking too much risk (Risk 1)',
            'bargain_desc': 'Buying too little, walking away, missing the cycle launch (Risk 2)',
            'default_side': 'danger' if val_temp >= 45 else 'bargain'
        }
    ]
    return items

