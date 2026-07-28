# =============================================================================
# SIGNAL ENGINE v2 — Regime-Adaptive Setup Detection & Outcome Tracking
# Enhanced with Minervini Trend Template, TraderLion HVE, merged setups,
# ADTV position sizing, and weighted RS.
# =============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from database import (
    get_open_signals, update_signal_outcome, mark_squat_alert
)
from config import RS_WEIGHTS

# =============================================================================
# REGIME → ALLOWED SETUPS MAPPING (6 merged setups)
# =============================================================================

REGIME_SETUPS = {
    'Confirmed Uptrend': {
        'label': '🚀 Expansion',
        'setups': ['High Tight Flag', 'Power Breakout', 'Momentum Continuation', 'VCP Breakout', 'Episodic Pivot', 'Pyramid (21 EMA)', 'EMA Crossback'],
        'holding': 'Positional (2–8 weeks)',
        'max_trades': 8,
        'expiry_days': 60,
    },
    'Uptrend Under Pressure': {
        'label': '📈 Trend',
        'setups': ['High Tight Flag', 'VCP Breakout', 'Episodic Pivot', 'Pyramid (21 EMA)', 'EMA Crossback'],
        'holding': 'Swing (2–4 weeks)',
        'max_trades': 8,
        'expiry_days': 30,
    },
    'Transition': {
        'label': '⚠️ Transition',
        'setups': ['Pyramid (21 EMA)', 'Momentum Continuation', 'EMA Crossback'],
        'holding': 'Swing (5–15 days)',
        'max_trades': 5,
        'expiry_days': 15,
    },
    'Market in Correction': {
        'label': '🔻 Defense',
        'setups': ['Mean Reversion', 'Reversal Extension'],
        'holding': 'Short Swing (3–7 days)',
        'max_trades': 3,
        'expiry_days': 15,
    },
}

DEFAULT_REGIME_KEY = 'Market in Correction'


# =============================================================================
# MINERVINI TREND TEMPLATE PRE-FILTER
# =============================================================================

def passes_trend_template(df, rs_score=None, relax_template=False):
    """
    Minervini's 8-point Trend Template checklist.
    Returns True only if ALL conditions pass:
    1. Close > 150 SMA
    2. Close > 200 SMA
    3. 150 SMA > 200 SMA
    4. 200 SMA rising for >= 20 trading days
    5. Close >= 25% above 52W low
    6. Close within 25% of 52W high
    7. RS >= 70
    8. Close > 50 SMA (price above all key MAs)
    
    If relax_template=True, we bypass the tightest structural rules to catch
    deeper tectonic secular shifts (e.g. US mega-caps recovering from 30% DD).
    """
    if df.empty or len(df) < 252:
        return False

    today = df.iloc[-1]
    close = today['close']

    # Calculate SMAs if not present
    sma_50 = df['close'].rolling(50).mean().iloc[-1]
    sma_150 = df['close'].rolling(150).mean().iloc[-1]
    sma_200 = df['close'].rolling(200).mean().iloc[-1]
    
    if relax_template:
        if close <= sma_200:
            return False
        if rs_score is not None and rs_score < 60: # Extremely relaxed RS
            return False
        return True

    # 200 SMA 20 days ago for rising check
    sma_200_20d_ago = df['close'].rolling(200).mean().iloc[-21] if len(df) >= 221 else sma_200

    # 52W high and low
    high_52w = df['high'].iloc[-252:].max()
    low_52w = df['low'].iloc[-252:].min()

    # Check all conditions purely mechanical
    if close <= sma_150:
        return False
    if close <= sma_200:
        return False
    if sma_150 <= sma_200:
        return False
    if sma_200 <= sma_200_20d_ago:  # 200 SMA not rising
        return False
    if low_52w <= 0 or ((close - low_52w) / low_52w * 100) < 25:
        return False
    if high_52w <= 0 or ((high_52w - close) / high_52w * 100) > 25:
        return False
    if rs_score is not None and rs_score < 70:
        return False
    if close <= sma_50:
        return False

    return True


# =============================================================================
# ADTV MAX POSITION SIZING
# =============================================================================

def calculate_adtv_max_position(df):
    """
    Max safe position = 5% of 21-day Average Daily Traded Value.
    ADTV = average(close × volume) over last 21 days.
    """
    if df.empty or len(df) < 21:
        return 0

    recent = df.tail(21)
    adtv = (recent['close'] * recent['volume']).mean()
    return round(adtv * 0.05, 0)


# =============================================================================
# WEIGHTED RS (APP STANDARD: 1M×0.40 + 3M×0.35 + 6M×0.25)
# =============================================================================

def calculate_weighted_rs_for_universe(history_df, tickers):
    """
    Compute weighted RS scores using the app's standard formula:
    RS_raw = (1M_return × 0.40) + (3M_return × 0.35) + (6M_return × 0.25)
    Then percentile-rank within the universe.

    Args:
        history_df: Multi-ticker DataFrame from yf.download
        tickers: list of ticker strings

    Returns:
        dict mapping clean ticker -> RS percentile (0-100)
    """
    rs_scores = {}
    w1 = RS_WEIGHTS.get('1M', 0.40)
    w3 = RS_WEIGHTS.get('3M', 0.35)
    w6 = RS_WEIGHTS.get('6M', 0.25)

    try:
        if not isinstance(history_df.columns, pd.MultiIndex):
            return rs_scores

        # Find the OHLCV level
        ohlcv_level = -1
        for i in range(history_df.columns.nlevels):
            l_vals = [str(v).lower() for v in history_df.columns.get_level_values(i)]
            if 'close' in l_vals:
                ohlcv_level = i
                break
        if ohlcv_level == -1:
            return rs_scores

        l_vals = [str(v).lower() for v in history_df.columns.get_level_values(ohlcv_level)]
        if 'close' not in l_vals:
            return rs_scores

        key = history_df.columns.get_level_values(ohlcv_level)[l_vals.index('close')]
        close_panel = history_df.xs(key, level=ohlcv_level, axis=1)

        raw_scores = {}
        for col in close_panel.columns:
            series = close_panel[col].dropna()
            if len(series) < 127:  # Need at least 6M of data
                continue

            # 1M return (21 days)
            r1 = (series.iloc[-1] / series.iloc[-22] - 1) * 100 if len(series) >= 22 else 0
            # 3M return (63 days)
            r3 = (series.iloc[-1] / series.iloc[-64] - 1) * 100 if len(series) >= 64 else 0
            # 6M return (126 days)
            r6 = (series.iloc[-1] / series.iloc[-127] - 1) * 100 if len(series) >= 127 else 0

            rs_raw = (r1 * w1) + (r3 * w3) + (r6 * w6)
            clean = col.replace('.NS', '') if isinstance(col, str) else str(col)
            raw_scores[clean] = rs_raw

        if not raw_scores:
            return rs_scores

        # Percentile rank
        all_raw = sorted(raw_scores.values())
        n = len(all_raw)
        for tk, raw in raw_scores.items():
            pct_rank = (sum(1 for r in all_raw if r <= raw) / n) * 100
            rs_scores[tk] = round(pct_rank, 1)

    except Exception:
        pass

    return rs_scores


# =============================================================================
# SETUP DETECTORS (6 merged categories)
# =============================================================================

def _vol_stats(df):
    """Helper: compute volume stats for confidence and detection."""
    avg_vol_20 = df['volume'].iloc[-21:-1].mean() if len(df) >= 22 else df['volume'].mean()
    avg_vol_50 = df['volume'].iloc[-51:-1].mean() if len(df) >= 52 else avg_vol_20
    avg_vol_5 = df['volume'].iloc[-6:-1].mean() if len(df) >= 6 else avg_vol_20
    today_vol = df.iloc[-1]['volume']
    return avg_vol_20, avg_vol_50, avg_vol_5, today_vol


def _prox_high(df):
    """Helper: proximity to 52W high."""
    lookback = df.iloc[-252:] if len(df) >= 252 else df
    max_high = lookback['high'].max()
    close = df.iloc[-1]['close']
    return round(abs((close - max_high) / max_high * 100), 2) if max_high > 0 else 100


def _tightness_score(df):
    """
    Tightness: 10-day price range / close.
    Lower = tighter consolidation = better quality base.
    """
    if len(df) < 11:
        return 0
    recent_10 = df.tail(10)
    range_10 = recent_10['high'].max() - recent_10['low'].min()
    close = df.iloc[-1]['close']
    return round(range_10 / close * 100, 2) if close > 0 else 100


def _prior_day_tight(df):
    """Check if prior day's range is < 50% of 20-day avg range (Zanger tight close)."""
    if len(df) < 22:
        return False
    prior = df.iloc[-2]
    prior_range = prior['high'] - prior['low']
    avg_range_20 = (df['high'].iloc[-21:-1] - df['low'].iloc[-21:-1]).mean()
    return prior_range < 0.5 * avg_range_20 if avg_range_20 > 0 else False


def _vol_dryup_score(df):
    """Volume dry-up: 5-day avg vol < 60% of 50-day avg vol (pre-breakout contraction)."""
    _, avg_vol_50, avg_vol_5, _ = _vol_stats(df)
    if avg_vol_50 <= 0:
        return 0
    ratio = avg_vol_5 / avg_vol_50
    if ratio < 0.6:
        return min(10, round((0.6 - ratio) * 50, 1))  # Max 10 points
    return 0


# ----- SETUP 7: EPISODIC PIVOT (TraderLion) -----
def detect_episodic_pivot(df):
    """
    Gap-up with extreme (>3x) volume expansion.
    """
    if df.empty or len(df) < 22:
        return None

    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    avg_vol_20, _, _, today_vol = _vol_stats(df)
    
    if today['low'] > yesterday['high'] and today_vol >= 3.0 * avg_vol_20 and today_vol > 0:
        # Closing range rule to ensure it's not a severe squat initially
        day_range = today['high'] - today['low']
        if day_range > 0:
            closing_range = (today['close'] - today['low']) / day_range
            if closing_range >= 0.25:  # At least not closing completely at the lows
                return {
                    'setup_type': 'Episodic Pivot',
                    'sub_type': 'Extreme Volume Gap',
                    'entry_price': round(today['close'], 2),
                    'stop_price': round(today['low'], 2),
                    'today_pct': round(today.get('daily_pct_change', 0), 2),
                    'vol_expansion': round(today_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 0,
                    'prox_high': _prox_high(df),
                    'tightness': _tightness_score(df),
                    'vol_dryup': _vol_dryup_score(df),
                    'tight_close': _prior_day_tight(df),
                }
    return None


# ----- SETUP 1: POWER BREAKOUT -----
def detect_power_breakout(df, rs_score=None, breadth_pct=None):
    """
    Merged: 52W High Breakout + HV1 (Yearly Volume Breakout) + Gap-Up Breakout.
    Requires RS >= 80 and Price within 20% of 52W High.
    Triggers if ANY of:
      A) Today High >= 252D High AND Volume >= 1.5× 20D avg
      B) Today Volume = highest in 252 days AND close up > 2%
      C) Gap-up (today low > yesterday high) AND Volume >= 1.5×
    Stop = breakout day low.
    """
    if df.empty or len(df) < 252:
        return None

    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    avg_vol_20, avg_vol_50, avg_vol_5, today_vol = _vol_stats(df)
    max_high_252 = df['high'].iloc[-253:-1].max()
    max_vol_252 = df['volume'].iloc[-253:-1].max()
    pct_change = today.get('daily_pct_change', 0)

    # [NEW AI REFACTOR]: Strict Stage 2 & Market Health Filters
    sma_50 = today.get('sma_50', 0)
    sma_200 = today.get('sma_200', 0)
    if today['close'] <= sma_50 or sma_50 <= sma_200:
        return None
        
    if breadth_pct is not None and breadth_pct <= 40.0:
        return None

    # [NEW AI REFACTOR]: Elite Filters (RS >= 80, Within 20% of 52W High)
    if rs_score is None or rs_score < 80:
        return None
        
    if pd.isna(max_high_252) or max_high_252 <= 0:
        return None
        
    dist_from_high = (max_high_252 - today['close']) / max_high_252 * 100
    if dist_from_high > 20.0:
        return None

    triggered = False
    sub_type = ''

    # A) 52W High Breakout
    if today['high'] >= max_high_252 and today_vol >= 1.5 * avg_vol_20:
        triggered = True
        sub_type = '52W High'

    # B) Highest Volume in 1 Year (HV1)
    if not triggered and today_vol > max_vol_252 and pct_change > 2:
        triggered = True
        sub_type = 'HV1 (Yearly Vol)'

    # C) Gap-Up Breakout
    if not triggered and today['low'] > yesterday['high'] and today_vol >= 1.5 * avg_vol_20:
        # Close should be in upper 50% of day's range (TraderLion closing range rule)
        day_range = today['high'] - today['low']
        if day_range > 0:
            closing_range = (today['close'] - today['low']) / day_range
            if closing_range >= 0.5:
                triggered = True
                sub_type = 'Gap-Up'

    if not triggered:
        return None

    # [NEW AI REFACTOR]: Require structural base contraction
    # Eliminate loose, extended breakouts that destroy win rate Expectancy.
    if _tightness_score(df) >= 15.0 and not _prior_day_tight(df):
        return None

    return {
        'setup_type': 'Power Breakout',
        'sub_type': sub_type,
        'entry_price': round(today['close'], 2),
        'stop_price': round(today['low'], 2),
        'today_pct': round(pct_change, 2),
        'vol_expansion': round(today_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 0,
        'prox_high': _prox_high(df),
        'tightness': _tightness_score(df),
        'vol_dryup': _vol_dryup_score(df),
        'tight_close': _prior_day_tight(df),
    }


# ----- SETUP 2: MOMENTUM CONTINUATION -----
def detect_momentum_continuation(df, rs_score=None, breadth_pct=None):
    """
    Merged: High Volume Breakout + Short Breakout Continuation.
    Close > yesterday's high on 1.5× volume.
    Requires RS >= 80 and Price within 20% of 52W High.
    """
    if df.empty or len(df) < 22:
        return None

    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    avg_vol_20, _, _, today_vol = _vol_stats(df)

    # [NEW AI REFACTOR]: Strict Stage 2 & Market Health Filters
    sma_50 = today.get('sma_50', 0)
    sma_200 = today.get('sma_200', 0)
    if today['close'] <= sma_50 or sma_50 <= sma_200:
        return None
        
    if breadth_pct is not None and breadth_pct <= 40.0:
        return None

    # [NEW AI REFACTOR]: Elite Filters (RS >= 80, Within 20% of 52W High)
    if rs_score is None or rs_score < 80:
        return None
        
    lookback = min(252, len(df))
    max_high_52w = df['high'].iloc[-lookback:-1].max() if lookback > 1 else df['high'].iloc[0]
    if pd.isna(max_high_52w) or max_high_52w <= 0:
        return None
        
    dist_from_high = (max_high_52w - today['close']) / max_high_52w * 100
    if dist_from_high > 20.0:
        return None

    if today['close'] > yesterday['high'] and today_vol >= 1.5 * avg_vol_20:
        # [NEW AI REFACTOR]: Require structural base contraction for breakouts
        if _tightness_score(df) >= 15.0 and not _prior_day_tight(df):
            return None
            
        return {
            'setup_type': 'Momentum Continuation',
            'sub_type': '',
            'entry_price': round(today['close'], 2),
            'stop_price': round(today['low'], 2),
            'today_pct': round(today.get('daily_pct_change', 0), 2),
            'vol_expansion': round(today_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 0,
            'prox_high': _prox_high(df),
            'tightness': _tightness_score(df),
            'vol_dryup': _vol_dryup_score(df),
            'tight_close': _prior_day_tight(df),
        }
    return None


# ----- SETUP 3: PULLBACK TO 21 EMA (PYRAMID ONLY) -----
def detect_pyramid_21ema(df, rs_score):
    """
    Holy Grail Pyramid Setup:
    Only fires if RS >= 80 (Elite RS) and price touches within ±2% of 21 EMA.
    Used exclusively to add a half-size position to an existing winner.
    """
    if df.empty or len(df) < 50 or rs_score is None or rs_score < 80:
        return None

    today = df.iloc[-1]
    ema_21 = today.get('ema_21')
    if ema_21 is None or ema_21 <= 0:
        return None

    # Focus strictly on the "Touch" / Pullback
    distance_pct = abs((today['close'] - ema_21) / ema_21 * 100)
    if distance_pct <= 2.0 and today['close'] >= ema_21:
        avg_vol_20, _, _, _ = _vol_stats(df)
        return {
            'setup_type': 'Pyramid (21 EMA)',
            'sub_type': 'RS>80 Pullback',
            'entry_price': round(today['close'], 2),
            'stop_price': round(ema_21 * 0.99, 2), # Typical structural stop below 21
            'today_pct': round(today.get('daily_pct_change', 0), 2),
            'vol_expansion': round(today['volume'] / avg_vol_20, 2) if avg_vol_20 > 0 else 0,
            'prox_high': _prox_high(df),
            'tightness': _tightness_score(df),
            'vol_dryup': _vol_dryup_score(df),
            'tight_close': _prior_day_tight(df),
        }
    return None


# ----- SETUP 4: PULLBACK TO 50 SMA -----
def detect_pullback_50sma(df, rs_score):
    """
    Price within ±3% of 50 SMA on RS >= 75.
    """
    if df.empty or len(df) < 60 or rs_score is None or rs_score < 75:
        return None

    today = df.iloc[-1]
    sma_50 = today.get('sma_50')
    if sma_50 is None or sma_50 <= 0:
        return None

    distance_pct = abs((today['close'] - sma_50) / sma_50 * 100)
    if distance_pct > 3.0:
        return None

    avg_vol_20, _, _, _ = _vol_stats(df)

    return {
        'setup_type': 'Pullback to 50 SMA',
        'sub_type': '',
        'entry_price': round(today['close'], 2),
        'stop_price': round(sma_50 * 0.99, 2),
        'today_pct': round(today.get('daily_pct_change', 0), 2),
        'vol_expansion': round(today['volume'] / avg_vol_20, 2) if avg_vol_20 > 0 else 0,
        'prox_high': _prox_high(df),
        'tightness': _tightness_score(df),
        'vol_dryup': _vol_dryup_score(df),
        'tight_close': _prior_day_tight(df),
    }


# ----- SETUP 5: VCP BREAKOUT (Minervini) -----
def detect_vcp_breakout(df, rs_score=None, breadth_pct=None):
    """
    Volatility Contraction Pattern:
    - ATR(14) / ATR(50) < 0.85 (range contracting)
    - Last 10-day range < prior 10-day range (tightening)
    - Today closes above the 10-day high on volume >= 1.1×
    Requires RS >= 80 and Price within 20% of 52W High.
    """
    if df.empty or len(df) < 60:
        return None

    today = df.iloc[-1]

    # [NEW AI REFACTOR]: Strict Stage 2 & Market Health Filters
    sma_50 = today.get('sma_50', 0)
    sma_200 = today.get('sma_200', 0)
    if today['close'] <= sma_50 or sma_50 <= sma_200:
        return None
        
    if breadth_pct is not None and breadth_pct <= 40.0:
        return None

    # [NEW AI REFACTOR]: Elite Filters (RS >= 80, Within 20% of 52W High)
    if rs_score is None or rs_score < 80:
        return None
        
    lookback = min(252, len(df))
    max_high_52w = df['high'].iloc[-lookback:-1].max() if lookback > 1 else df['high'].iloc[0]
    if pd.isna(max_high_52w) or max_high_52w <= 0:
        return None
        
    dist_from_high = (max_high_52w - today['close']) / max_high_52w * 100
    if dist_from_high > 20.0:
        return None

    # ATR contraction
    atr_14 = today.get('atr_14')
    atr_30 = today.get('atr_30')
    if atr_14 is None or atr_30 is None or atr_30 <= 0:
        # Compute manually
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_14 = tr.iloc[-14:].mean()
        atr_50 = tr.iloc[-50:].mean() if len(df) >= 50 else tr.mean()
    else:
        atr_50 = atr_30  # Use available longer ATR

    if atr_50 <= 0 or (atr_14 / atr_50) >= 0.85: # [AI REFACTOR]: Relaxed from 0.75 for better capture rate
        return None

    # Range tightening: last 10d range < prior 10d range
    if len(df) < 21:
        return None
    range_recent = df['high'].iloc[-10:].max() - df['low'].iloc[-10:].min()
    range_prior = df['high'].iloc[-20:-10].max() - df['low'].iloc[-20:-10].min()
    if range_prior <= 0 or range_recent > range_prior * 1.05: # [AI REFACTOR]: Relaxed strict contraction
        return None

    # Breakout: close > 10-day high on volume >= 1.1×
    high_10d = df['high'].iloc[-11:-1].max()
    avg_vol_20, _, _, today_vol = _vol_stats(df)
    if today['close'] <= high_10d or today_vol < 1.1 * avg_vol_20: # [AI REFACTOR]: Relaxed vol from 1.2x
        return None

    return {
        'setup_type': 'VCP Breakout',
        'sub_type': f"ATR ratio {atr_14/atr_50:.2f}",
        'entry_price': round(today['close'], 2),
        'stop_price': round(df['low'].iloc[-10:].min(), 2),  # Stop at VCP low
        'today_pct': round(today.get('daily_pct_change', 0), 2),
        'vol_expansion': round(today_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 0,
        'prox_high': _prox_high(df),
        'tightness': _tightness_score(df),
        'vol_dryup': _vol_dryup_score(df),
        'tight_close': _prior_day_tight(df),
    }


# ----- SETUP 7: HIGH TIGHT FLAG (O'Neil / Minervini / Ryan) -----
def detect_high_tight_flag(df, breadth_pct=None):
    """
    O'Neil's 'most bullish of all chart patterns' — adapted for broad market:
    1. POLE: Price advanced >= 50% within the last 60 trading days
       (Relaxed from 90% per Minervini large-cap 'Power Play' applicability)
    2. FLAG: Pullback from peak is <= 25%
    3. FLAG DURATION: 7-35 day consolidation 
    4. TIGHTNESS: Flag range <= 20% of flag high 
    5. VOLUME: Volume drying up during flag
    6. BREAKOUT: Today's close > flag high on volume >= 1.5x
    7. STAGE 2 + MARKET HEALTH guards
    """
    if df.empty or len(df) < 60:
        return None

    today = df.iloc[-1]

    # --- Stage 2 & Market Health Guards ---
    sma_50 = today.get('sma_50', 0)
    sma_200 = today.get('sma_200', 0)
    if today['close'] <= sma_50 or sma_50 <= sma_200:
        return None
    if breadth_pct is not None and breadth_pct <= 40.0:
        return None

    # --- Step 1: Find the POLE ---
    # Look back up to 95 days (60d pole + 35d flag max)
    lookback = min(95, len(df) - 1)
    window = df.iloc[-lookback:]

    # Find the lowest low in the window (potential pole start)
    pole_start_idx = window['low'].idxmin()
    pole_start_pos = df.index.get_loc(pole_start_idx)

    # Find the highest high AFTER the pole start (potential pole end / flag start)
    after_pole_start = df.iloc[pole_start_pos:]
    if len(after_pole_start) < 12:
        return None

    peak_idx = after_pole_start['high'].idxmax()
    peak_pos = df.index.get_loc(peak_idx)
    peak_price = df.loc[peak_idx, 'high']
    pole_start_price = df.loc[pole_start_idx, 'low']

    if pole_start_price <= 0:
        return None

    # Calculate pole gain
    pole_gain_pct = (peak_price - pole_start_price) / pole_start_price * 100

    # Pole must be >= 70% gain (Power Play criteria)
    if pole_gain_pct < 70.0:
        return None

    # Pole duration must be <= 60 trading days 
    pole_duration = peak_pos - pole_start_pos
    if pole_duration > 60 or pole_duration < 5:
        return None

    # --- Step 2: Analyze the FLAG ---
    current_pos = len(df) - 1
    flag_duration = current_pos - peak_pos

    # Flag must be 7-30 trading days (widened from 10-25 per Bulkowski)
    if flag_duration < 7 or flag_duration > 30:
        return None

    flag_data = df.iloc[peak_pos:]

    # Max pullback from peak must be <= 25%
    flag_low = flag_data['low'].min()
    pullback_pct = (peak_price - flag_low) / peak_price * 100
    if pullback_pct > 25.0:
        return None

    # Flag tightness: range must be <= 20% of the flag high (Bulkowski: 10-34%)
    flag_high = flag_data['high'].max()
    flag_range_pct = (flag_high - flag_low) / flag_high * 100
    if flag_range_pct > 20.0:
        return None

    # --- Step 3: Volume Dry-Up during flag ---
    avg_vol_20, _, avg_vol_5, today_vol = _vol_stats(df)
    if avg_vol_20 > 0 and avg_vol_5 > 0:
        # Volume during flag should be less than 20-day average
        flag_avg_vol = df['volume'].iloc[-min(10, flag_duration):].mean()
        if flag_avg_vol > avg_vol_20 * 1.15:  # Slightly more tolerant
            return None

    # --- Step 4: BREAKOUT Confirmation ---
    # Today's close must break above the flag's high (excluding today)
    flag_high_prior = df['high'].iloc[peak_pos:-1].max() if peak_pos < current_pos else peak_price
    if today['close'] <= flag_high_prior:
        return None

    # Volume must be >= 1.5x the 20-day average
    if today_vol < 1.5 * avg_vol_20:
        return None

    return {
        'setup_type': 'High Tight Flag',
        'sub_type': f"Pole +{pole_gain_pct:.0f}%, Flag {flag_duration}d, PB {pullback_pct:.0f}%",
        'entry_price': round(today['close'], 2),
        'stop_price': round(flag_low, 2),  # Stop at flag low
        'today_pct': round(today.get('daily_pct_change', 0), 2),
        'vol_expansion': round(today_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 0,
        'prox_high': _prox_high(df),
        'tightness': _tightness_score(df),
        'vol_dryup': _vol_dryup_score(df),
        'tight_close': _prior_day_tight(df),
    }


# ----- SETUP 6: MEAN REVERSION -----
def detect_mean_reversion(df):
    """
    Merged: Mean Reversion Bounce + 3-Day Decline Reversal.
    Either:
      A) 3 consecutive red candles, then close > yesterday high
      B) 3 consecutive declining closes, then close > yesterday high
    Stop = today's low.
    NOTE: Trend Template is NOT required for mean reversion.
    """
    if df.empty or len(df) < 5:
        return None

    today = df.iloc[-1]
    yesterday = df.iloc[-2]

    if today['close'] <= yesterday['high']:
        return None

    # A) 3 red candles
    three_red = all(df.iloc[i]['close'] < df.iloc[i]['open'] for i in range(-4, -1))

    # B) 3 declining closes
    three_decline = all(df.iloc[i]['close'] < df.iloc[i - 1]['close'] for i in range(-3, 0))

    if not three_red and not three_decline:
        return None

    sub_type = '3 Red Days' if three_red else '3 Declining Closes'
    avg_vol_20 = df['volume'].iloc[-21:-1].mean() if len(df) >= 22 else df['volume'].mean()

    return {
        'setup_type': 'Mean Reversion',
        'sub_type': sub_type,
        'entry_price': round(today['close'], 2),
        'stop_price': round(today['low'], 2),
        'today_pct': round(today.get('daily_pct_change', 0), 2),
        'vol_expansion': round(today['volume'] / avg_vol_20, 2) if avg_vol_20 > 0 else 0,
        'prox_high': _prox_high(df) if len(df) >= 252 else 50,
        'tightness': 0,
        'vol_dryup': 0,
        'tight_close': False,
    }


# ----- SETUP 8: EMA CROSSBACK (Oliver Kell) -----
def detect_ema_crossback_setup(df, rs_score=None, breadth_pct=None):
    """
    Price pulls back below the 10 EMA for a few days to shake out weak hands,
    then violently crosses back above it today.
    """
    if df.empty or len(df) < 20:
        return None
        
    from technical_indicators import detect_ema_crossback
    if not detect_ema_crossback(df, ema_period=10, max_days_below=5):
        return None
        
    today = df.iloc[-1]
    
    # Stage 2 Guards
    sma_50 = today.get('sma_50', 0)
    sma_200 = today.get('sma_200', 0)
    if today['close'] <= sma_50 or sma_50 <= sma_200:
        return None
        
    avg_vol_20, _, _, today_vol = _vol_stats(df)
    
    return {
        'setup_type': 'EMA Crossback',
        'sub_type': '10 EMA Bounce',
        'entry_price': round(today['close'], 2),
        'stop_price': round(df['low'].iloc[-3:].min(), 2), # Stop at recent swing low
        'today_pct': round(today.get('daily_pct_change', 0), 2),
        'vol_expansion': round(today_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 0,
        'prox_high': _prox_high(df),
        'tightness': _tightness_score(df),
        'vol_dryup': _vol_dryup_score(df),
        'tight_close': False,
    }


# ----- SETUP 9: REVERSAL EXTENSION (Oliver Kell) -----
def detect_reversal_extension_setup(df):
    """
    Price is significantly extended to the downside (>12% below 10 EMA) 
    and forms a climax bottom or reversal.
    """
    if df.empty or len(df) < 20:
        return None
        
    from technical_indicators import detect_reversal_extension
    if not detect_reversal_extension(df, ema_period=10, extension_threshold=12.0):
        return None
        
    today = df.iloc[-1]
    avg_vol_20, _, _, today_vol = _vol_stats(df)
    
    return {
        'setup_type': 'Reversal Extension',
        'sub_type': '>12% Below 10 EMA',
        'entry_price': round(today['close'], 2),
        'stop_price': round(today['low'], 2), # Very tight stop at reversal low
        'today_pct': round(today.get('daily_pct_change', 0), 2),
        'vol_expansion': round(today_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 0,
        'prox_high': _prox_high(df),
        'tightness': 0,
        'vol_dryup': 0,
        'tight_close': False,
    }


# =============================================================================
# ENHANCED CONFIDENCE SCORE
# =============================================================================

def calculate_confidence(rs_score, vol_expansion, prox_high_pct, regime, breadth_pct,
                          tightness=0, vol_dryup=0, tight_close=False):
    """
    Enhanced Confidence Score (0–100):
    - RS Component:       0–25
    - Volume Expansion:   0–15
    - Proximity to High:  0–10
    - Regime Alignment:   0–20
    - Breadth Strength:   0–10
    - Volume Dry-Up:      0–10  (NEW)
    - Tightness:          0–5   (NEW)
    - Tight Close Bonus:  0–5   (NEW)
    """
    # RS Component (0–25)
    rs = min(25, max(0, (rs_score or 0) * 0.25))

    # Volume Expansion (0–15): 1x = 0, 3x+ = 15
    vol = min(15, max(0, (vol_expansion - 1.0) * 7.5))

    # Proximity to High (0–10): 0% = 10, 25%+ = 0
    prox = max(0, 10 - prox_high_pct * 0.4)

    # Regime Alignment (0–20)
    regime_scores = {
        'Confirmed Uptrend': 20,
        'Uptrend Under Pressure': 12,
        'Transition': 6,
        'Market in Correction': 3,
    }
    reg = regime_scores.get(regime, 3)

    # Breadth Strength (0–10)
    brd = min(10, max(0, (breadth_pct or 50) * 0.10))

    # Volume Dry-Up bonus (0–10): already computed
    vdu = min(10, max(0, vol_dryup))

    # Tightness bonus (0–5): 10d range/close < 8% = 5, > 15% = 0
    tight = 0
    if tightness > 0:
        tight = max(0, min(5, (15 - tightness) * 0.71))

    # Tight Close bonus (0–5): prior day had tight range
    tc = 5 if tight_close else 0

    return round(rs + vol + prox + reg + brd + vdu + tight + tc, 1)


# =============================================================================
# RISK FILTER
# =============================================================================

def calculate_risk_pct(entry, stop):
    """Risk % = (entry - stop) / entry * 100."""
    if entry <= 0:
        return 100
    return round(abs(entry - stop) / entry * 100, 2)


def passes_weekly_tightness(df: pd.DataFrame, max_allowed_range_pct: float = 0.05) -> bool:
    """
    Check if the last 3 weekly bars are tight (David Ryan / Wyckoff logic).
    Resamples daily data into weekly, calculates the max/min close of the last 3 weeks,
    and returns True if the percentage difference between Max Close and Min Close is <= max_allowed_range_pct.
    """
    if len(df) < 20: 
        return False
        
    try:
        # Resample daily to weekly (Friday close)
        weekly = df.resample('W-FRI').agg({'close': 'last'}).dropna()
        if len(weekly) < 4:  # Need at least 3 completed weeks + current
            return False
            
        # Look at the 3 full weeks prior to the current week
        last_3_weeks = weekly.iloc[-4:-1]['close']
        max_c = last_3_weeks.max()
        min_c = last_3_weeks.min()
        
        if min_c > 0:
            tightness = (max_c - min_c) / min_c
            return tightness <= max_allowed_range_pct
        return False
    except Exception:
        return False

# =============================================================================
# MASTER SCAN
# =============================================================================

def scan_universe_for_setups(history_df: pd.DataFrame, tickers: list, regime: str, 
                           rs_scores: dict = None, breadth_pct: float = 50.0,
                           require_weekly_tightness: bool = False) -> list:
    """
    Scan the entire universe for trade setups given the current regime.
    All setups (except Mean Reversion) must pass Minervini Trend Template.
    """
    regime_info = REGIME_SETUPS.get(regime, REGIME_SETUPS[DEFAULT_REGIME_KEY])
    allowed = regime_info['setups']
    max_trades = regime_info['max_trades']
    holding = regime_info['holding']

    from technical_indicators import add_technical_indicators

    results = []

    is_multi = isinstance(history_df.columns, pd.MultiIndex) and len(tickers) > 1

    for ticker in tickers:
        clean_ticker = ticker.replace('.NS', '')
        rs = (rs_scores or {}).get(clean_ticker, 50)

        try:
            if is_multi:
                if ticker not in history_df.columns.get_level_values(0):
                    continue
                df = history_df[ticker].copy()
            else:
                df = history_df.copy()

            df.columns = [str(c).lower().strip() for c in df.columns]
            df = df.dropna(subset=['close'])
            if len(df) < 22:
                continue

            df = add_technical_indicators(df)

            # Pre-compute Trend Template pass
            trend_ok = passes_trend_template(df, rs)

            # Pre-compute ADTV max position
            max_pos = calculate_adtv_max_position(df)

            # Run detectors based on allowed setups
            setups_found = []

            if 'High Tight Flag' in allowed and trend_ok:
                s = detect_high_tight_flag(df, breadth_pct)
                if s:
                    setups_found.append(s)

            if 'Power Breakout' in allowed and trend_ok:
                s = detect_power_breakout(df, rs, breadth_pct)
                if s:
                    setups_found.append(s)

            if 'Momentum Continuation' in allowed and trend_ok:
                s = detect_momentum_continuation(df, rs, breadth_pct)
                if s:
                    setups_found.append(s)

            if 'Pyramid (21 EMA)' in allowed and trend_ok:
                s = detect_pyramid_21ema(df, rs)
                if s:
                    setups_found.append(s)

            if 'VCP Breakout' in allowed and trend_ok:
                s = detect_vcp_breakout(df, rs, breadth_pct)
                if s:
                    setups_found.append(s)

            if 'Episodic Pivot' in allowed and trend_ok:
                s = detect_episodic_pivot(df)
                if s:
                    setups_found.append(s)

            if 'Mean Reversion' in allowed:
                # Mean Reversion does NOT require trend template
                s = detect_mean_reversion(df)
                if s:
                    setups_found.append(s)

            if 'EMA Crossback' in allowed and trend_ok:
                s = detect_ema_crossback_setup(df, rs, breadth_pct)
                if s:
                    setups_found.append(s)

            if 'Reversal Extension' in allowed:
                # Capitulation setup, does not require trend template
                s = detect_reversal_extension_setup(df)
                if s:
                    setups_found.append(s)

            # Apply Weekly Tightness Filter if requested and applicable
            if require_weekly_tightness and setups_found:
                # We specifically enforce this for Power Breakouts and Momentum Continuation
                wt_passed = None # Lazy evaluate only if needed
                filtered_setups = []
                for setup in setups_found:
                    stype = setup['setup_type']
                    if stype in ['Power Breakout', 'Momentum Continuation']:
                        if wt_passed is None:
                            wt_passed = passes_weekly_tightness(df, 0.05)
                        if wt_passed:
                            filtered_setups.append(setup)
                    else:
                        filtered_setups.append(setup)
                setups_found = filtered_setups

            for setup in setups_found:
                risk = calculate_risk_pct(setup['entry_price'], setup['stop_price'])
                max_allowed_risk = 15.0 if setup['setup_type'] == 'Episodic Pivot' else 8.0
                
                if risk > max_allowed_risk or risk <= 0:
                    continue

                confidence = calculate_confidence(
                    rs_score=rs,
                    vol_expansion=setup.get('vol_expansion', 1),
                    prox_high_pct=setup.get('prox_high', 0),
                    regime=regime,
                    breadth_pct=breadth_pct,
                    tightness=setup.get('tightness', 0),
                    vol_dryup=setup.get('vol_dryup', 0),
                    tight_close=setup.get('tight_close', False),
                )

                if confidence < 70:
                    continue

                results.append({
                    'ticker': clean_ticker,
                    'setup_type': setup['setup_type'],
                    'sub_type': setup.get('sub_type', ''),
                    'entry_price': setup['entry_price'],
                    'stop_price': setup['stop_price'],
                    'risk_percent': risk,
                    'confidence_score': confidence,
                    'holding_bias': holding,
                    'regime': regime,
                    'today_pct': setup.get('today_pct', 0),
                    'max_position': max_pos,
                    'date_generated': datetime.now().strftime('%Y-%m-%d'),
                })
        except Exception:
            continue

    results.sort(key=lambda x: x['confidence_score'], reverse=True)
    return results[:max_trades]


# =============================================================================
# AUTOMATED OUTCOME TRACKING
# =============================================================================

def update_open_signal_outcomes():
    """
    Load all Open signals, fetch latest data, and check:
    1. Stop Hit  → Loss (R = -1)
    2. Target Hit → Win  (R = +2)
    3. Time Expiry → close at last price, compute R
    Also tracks MFE and MAE.
    """
    open_signals = get_open_signals()
    if not open_signals:
        return 0

    today_str = datetime.now().strftime('%Y-%m-%d')
    updated = 0

    tickers_needed = list(set(s['ticker'] for s in open_signals))
    price_cache = {}

    for ticker in tickers_needed:
        try:
            yf_ticker = f"{ticker}.NS"
            stock = yf.Ticker(yf_ticker)
            # Need 1y for accurate 200 SMA and Climax computations
            hist = stock.history(period="1y")
            if not hist.empty:
                hist.columns = [str(c).lower() for c in hist.columns]
                from technical_indicators import add_technical_indicators
                hist = add_technical_indicators(hist)
                price_cache[ticker] = hist
        except Exception:
            continue

    for signal in open_signals:
        ticker = signal['ticker']
        hist = price_cache.get(ticker)
        if hist is None or hist.empty:
            continue

        entry = signal['entry_price']
        stop = signal['stop_price']
        risk_amt = abs(entry - stop)
        if risk_amt <= 0:
            continue

        target = entry + (2 * risk_amt)
        date_gen = signal['date_generated']

        try:
            signal_date = pd.Timestamp(date_gen)
            if getattr(hist.index, 'tz', None) is not None:
                signal_date = signal_date.tz_localize(hist.index.tz)
            post_signal = hist[hist.index >= signal_date]
        except Exception:
            post_signal = hist

        if post_signal.empty:
            continue

        # Day 1 Squat Check
        if not signal.get('squat_alert'):
            day1 = post_signal.iloc[0]
            day1_range = day1['high'] - day1['low']
            if day1_range > 0:
                closing_range = (day1['close'] - day1['low']) / day1_range
                if closing_range < 0.25:
                    mark_squat_alert(signal['signal_id'])

        mfe = round(((post_signal['high'].max() - entry) / risk_amt), 2) if risk_amt > 0 else 0
        mae = round(((post_signal['low'].min() - entry) / risk_amt), 2) if risk_amt > 0 else 0

        status = None
        exit_price = None
        r_multiple = None
        exit_date_str = None

        for idx, row in post_signal.iterrows():
            curr_low, curr_high, curr_close = row['low'], row['high'], row['close']
            days_held = (idx.date() - pd.Timestamp(date_gen).date()).days
            
            # 1. Hard Initial Stop Loss
            if curr_low <= stop:
                status = 'Loss'
                exit_price = stop
                exit_date_str = idx.strftime('%Y-%m-%d')
                r_multiple = -1.0
                break
                
            # 2. Trailing Stops (Activate after 5 days)
            if days_held > 5:
                sma_200 = row.get('sma_200', 0)
                ema_10 = row.get('ema_10', 0)
                sma_50 = row.get('sma_50', 0)
                climax_score = row.get('climax_score', 0)
                
                # A. Parabolic Climax (10 EMA break on high climax score)
                if climax_score >= 4 and ema_10 > 0 and curr_close < ema_10:
                    status = 'Parabolic Climax'
                    exit_price = curr_close
                    exit_date_str = idx.strftime('%Y-%m-%d')
                    r_multiple = round((curr_close - entry) / risk_amt, 2) if risk_amt > 0 else 0
                    break
                    
                # B. Primary Structural Trail (50 SMA)
                if sma_50 > 0 and curr_close < sma_50:
                    curr_vol = row['volume']
                    avg_vol_20 = row.get('avg_vol_20', 0)
                    
                    if curr_vol > avg_vol_20:
                        status = '50 SMA Vol Break'
                        exit_price = curr_close
                        exit_date_str = idx.strftime('%Y-%m-%d')
                        r_multiple = round((curr_close - entry) / risk_amt, 2) if risk_amt > 0 else 0
                        break
                    else:
                        row_idx = post_signal.index.get_loc(idx)
                        if row_idx > 0:
                            prev_row = post_signal.iloc[row_idx - 1]
                            if prev_row['close'] < prev_row.get('sma_50', 0):
                                status = '50 SMA 2-Day Break'
                                exit_price = curr_close
                                exit_date_str = idx.strftime('%Y-%m-%d')
                                r_multiple = round((curr_close - entry) / risk_amt, 2) if risk_amt > 0 else 0
                                break
                                
            # 3. Expiry (25 days stagnant)
            if days_held > 25 and curr_close < (entry * 1.05):
                status = 'Time Stop'
                exit_price = curr_close
                exit_date_str = idx.strftime('%Y-%m-%d')
                r_multiple = round((curr_close - entry) / risk_amt, 2) if risk_amt > 0 else 0
                break

        if status:
            update_signal_outcome(
                signal_id=signal['signal_id'],
                status=status,
                exit_price=exit_price,
                exit_date=exit_date_str,
                r_multiple=r_multiple,
                mfe=mfe,
                mae=mae,
            )
            updated += 1

    return updated


# =============================================================================
# 15-DAY BACKFILL — Replay setup detection for missed days
# =============================================================================

def _scan_single_day(df_slice, ticker_clean, regime, rs, breadth_pct, holding, max_pos, date_str):
    """
    Run all 6 detectors on a single-day slice (df ending at that day).
    Returns list of signal dicts for that day.
    """
    from technical_indicators import add_technical_indicators

    if df_slice.empty or len(df_slice) < 22:
        return []

    df = df_slice.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    df = df.dropna(subset=['close'])
    if len(df) < 22:
        return []

    df = add_technical_indicators(df)

    regime_info = REGIME_SETUPS.get(regime, REGIME_SETUPS[DEFAULT_REGIME_KEY])
    allowed = regime_info['setups']

    trend_ok = passes_trend_template(df, rs)

    setups_found = []

    if 'High Tight Flag' in allowed and trend_ok:
        s = detect_high_tight_flag(df, breadth_pct)
        if s:
            setups_found.append(s)

    if 'Power Breakout' in allowed and trend_ok:
        s = detect_power_breakout(df, rs, breadth_pct)
        if s:
            setups_found.append(s)

    if 'Momentum Continuation' in allowed and trend_ok:
        s = detect_momentum_continuation(df, rs, breadth_pct)
        if s:
            setups_found.append(s)

    if 'Pyramid (21 EMA)' in allowed and trend_ok:
        s = detect_pyramid_21ema(df, rs)
        if s:
            setups_found.append(s)



    if 'VCP Breakout' in allowed and trend_ok:
        s = detect_vcp_breakout(df, rs, breadth_pct)
        if s:
            setups_found.append(s)

    if 'Episodic Pivot' in allowed and trend_ok:
        s = detect_episodic_pivot(df)
        if s:
            setups_found.append(s)

    if 'Mean Reversion' in allowed:
        s = detect_mean_reversion(df)
        if s:
            setups_found.append(s)

    results = []
    for setup in setups_found:
        risk = calculate_risk_pct(setup['entry_price'], setup['stop_price'])
        max_allowed_risk = 15.0 if setup['setup_type'] == 'Episodic Pivot' else 8.0
        
        if risk > max_allowed_risk or risk <= 0:
            continue

        confidence = calculate_confidence(
            rs_score=rs,
            vol_expansion=setup.get('vol_expansion', 1),
            prox_high_pct=setup.get('prox_high', 0),
            regime=regime,
            breadth_pct=breadth_pct,
            tightness=setup.get('tightness', 0),
            vol_dryup=setup.get('vol_dryup', 0),
            tight_close=setup.get('tight_close', False),
        )

        if confidence < 70:
            continue

        results.append({
            'ticker': ticker_clean,
            'setup_type': setup['setup_type'],
            'sub_type': setup.get('sub_type', ''),
            'entry_price': setup['entry_price'],
            'stop_price': setup['stop_price'],
            'risk_percent': risk,
            'confidence_score': confidence,
            'holding_bias': holding,
            'regime': regime,
            'today_pct': setup.get('today_pct', 0),
            'max_position': max_pos,
            'date_generated': date_str,
        })

    return results


def backfill_signals(history_df, tickers, regime, rs_scores=None, breadth_pct=50, lookback_days=15):
    """
    Backfill signals for the last N trading days.

    For each of the last `lookback_days` trading days, slices the history
    so that day becomes the "latest" row and runs all detectors.
    This ensures users who don't open the app daily still have
    a complete 15-day signal history with outcomes.

    Args:
        history_df: Multi-ticker DataFrame from yf.download (1Y of data)
        tickers: list of ticker strings (with .NS suffix)
        regime: current market regime string
        rs_scores: dict mapping clean ticker -> RS percentile
        breadth_pct: current positive breadth %
        lookback_days: number of trading days to backfill (default 15)

    Returns:
        total_new: count of newly saved signals
    """
    from database import save_signal

    regime_info = REGIME_SETUPS.get(regime, REGIME_SETUPS[DEFAULT_REGIME_KEY])
    holding = regime_info['holding']
    max_trades_per_day = regime_info['max_trades']

    is_multi = isinstance(history_df.columns, pd.MultiIndex) and len(tickers) > 1

    # Determine the trading dates from the index
    try:
        if is_multi:
            dates = history_df.index
        else:
            dates = history_df.index
        dates = pd.DatetimeIndex(dates).sort_values()
    except Exception:
        return 0

    if len(dates) < lookback_days + 252:
        # Not enough data to backfill with trend template (needs 252 days lookback)
        available = max(0, len(dates) - 252)
        lookback_days = min(lookback_days, available)

    if lookback_days <= 0:
        return 0

    # The last N dates to backfill (exclude today since main scan handles it)
    backfill_dates = dates[-(lookback_days + 1):-1]  # Last 15 trading days before today

    total_new = 0

    for bf_date in backfill_dates:
        date_str = bf_date.strftime('%Y-%m-%d')
        day_signals = []

        for ticker in tickers:
            clean_ticker = ticker.replace('.NS', '')
            rs = (rs_scores or {}).get(clean_ticker, 50)

            try:
                if is_multi:
                    if ticker not in history_df.columns.get_level_values(0):
                        continue
                    df_full = history_df[ticker].copy()
                else:
                    df_full = history_df.copy()

                # Slice data up to and including the backfill date
                df_slice = df_full[df_full.index <= bf_date]
                if df_slice.empty or len(df_slice) < 252:
                    continue

                # Standardize columns for ADTV calc
                df_std = df_slice.copy()
                df_std.columns = [str(c).lower().strip() for c in df_std.columns]
                max_pos = calculate_adtv_max_position(df_std)

                signals = _scan_single_day(
                    df_slice, clean_ticker, regime, rs, breadth_pct,
                    holding, max_pos, date_str,
                )
                day_signals.extend(signals)

            except Exception:
                continue

        # Sort by confidence, cap per day
        day_signals.sort(key=lambda x: x['confidence_score'], reverse=True)
        day_signals = day_signals[:max_trades_per_day]

        for sig in day_signals:
            if save_signal(sig):
                total_new += 1

    # After backfill, run outcome tracking on all open signals
    update_open_signal_outcomes()

    return total_new
