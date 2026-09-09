# =============================================================================
# CANSLIM SWING TRADING ENGINE
# Vectorized, High-Precision Momentum & Execution System for NIFTY 750
# Inspired by William J. O'Neil, David Ryan, and Jim Roppel
# =============================================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3

from config import CANSLIM_SWING_SETTINGS, BENCHMARK_TICKER
from database import get_all_fundamentals_cache, get_connection
from multibagger_analyzer import detect_vcp_pattern, check_minervini_trend_template, calculate_base_stage, check_volume_dryup
from macro_regime_engine import calculate_distribution_days, detect_follow_through_day, get_market_regime_label


# =============================================================================
# 1. UNIVERSE & MOMENTUM RADAR (TOP 50 CANDIDATES)
# =============================================================================

def get_top_momentum_candidates(score_matrix, date=None, n=50):
    """
    Extracts the Top N highest Z-score cross-sectional momentum leaders for a date.
    Returns: list of (ticker, score) tuples sorted descending.
    """
    if score_matrix is None or score_matrix.empty:
        return []
        
    if date is None or date not in score_matrix.index:
        date = score_matrix.index[-1]
        
    scores_series = score_matrix.loc[date].dropna()
    top_n = scores_series.nlargest(n)
    
    candidates = []
    for ticker, score in top_n.items():
        candidates.append((ticker, float(score)))
    return candidates


# =============================================================================
# 2. CANSLIM FUNDAMENTAL GATE (EPS & SALES >= 25%)
# =============================================================================

def check_canslim_fundamentals(ticker: str, fundamentals_cache: dict = None) -> dict:
    """
    Fundamental Qualification Gate:
    - Minimum Sales Growth YoY >= 25% (EPS and ROE removed per user configuration)
    """
    if fundamentals_cache is None:
        fundamentals_cache = get_all_fundamentals_cache()
        
    clean_t = ticker.replace('.NS', '').upper()
    full_t = f"{clean_t}.NS"
    
    data = fundamentals_cache.get(full_t) or fundamentals_cache.get(clean_t) or {}
    
    sales = data.get('sales_growth', 0.0) or 0.0
    eps = data.get('eps_growth', 0.0) or 0.0
    roe = data.get('roe', 0.0) or 0.0
    industry = data.get('industry', 'Unknown')
    mcap = data.get('market_cap', 0.0) or 0.0
    
    min_sales = CANSLIM_SWING_SETTINGS.get('MIN_SALES_GROWTH_PCT', 25.0)
    sales_passed = sales >= min_sales
    
    # Sole qualification is Sales Growth >= 25%
    fully_passed = sales_passed
    
    return {
        'passed': bool(fully_passed),
        'sales_growth': float(sales),
        'eps_growth': float(eps),
        'roe': float(roe),
        'industry': industry,
        'market_cap': float(mcap),
        'sales_passed': bool(sales_passed),
        'has_data': bool(data)
    }


# =============================================================================
# 3. DAVID RYAN RS LINE NEW HIGH (RSNH)
# =============================================================================

def detect_rs_line_new_high(stock_closes: pd.Series, benchmark_closes: pd.Series, lookback: int = 252, threshold_days: int = 10) -> dict:
    """
    David Ryan's premier chart indicator:
    Detects if the Relative Strength line (Stock / Benchmark) reached a new 52-week (lookback)
    high within the last `threshold_days` while the stock price itself is NOT yet extended at all-time highs.
    """
    result = {'rsnh_active': False, 'rs_line_new_high_days_ago': None, 'rs_ahead_of_price': False}
    
    if stock_closes is None or benchmark_closes is None:
        return result
        
    if isinstance(stock_closes, pd.DataFrame):
        stock_closes = stock_closes.iloc[:, 0]
    if isinstance(benchmark_closes, pd.DataFrame):
        benchmark_closes = benchmark_closes.iloc[:, 0]
        
    stock_closes = pd.to_numeric(stock_closes, errors='coerce').dropna()
    benchmark_closes = pd.to_numeric(benchmark_closes, errors='coerce').dropna()
    
    if len(stock_closes) < 126 or len(benchmark_closes) < 126:
        return result
        
    common_idx = stock_closes.index.intersection(benchmark_closes.index)
    if len(common_idx) < 126:
        return result
        
    s_close = stock_closes.loc[common_idx]
    b_close = benchmark_closes.loc[common_idx]
    
    rs_line = s_close / b_close
    eff_lookback = min(lookback, len(rs_line))
    
    rolling_rs_high = rs_line.rolling(window=eff_lookback, min_periods=eff_lookback // 2).max()
    rolling_price_high = s_close.rolling(window=eff_lookback, min_periods=eff_lookback // 2).max()
    
    # Check within the last threshold_days
    recent_rs = rs_line.tail(threshold_days)
    recent_rs_highs = rolling_rs_high.tail(threshold_days)
    
    # Hits if RS Line is within 0.5% of its rolling high
    is_rs_high = recent_rs >= (recent_rs_highs * 0.995)
    
    if is_rs_high.any():
        result['rsnh_active'] = True
        hit_positions = np.where(is_rs_high.values)[0]
        if len(hit_positions) > 0:
            result['rs_line_new_high_days_ago'] = int((len(recent_rs) - 1) - hit_positions[-1])
            
        current_price = s_close.iloc[-1]
        current_52w_high = rolling_price_high.iloc[-1]
        if current_52w_high > 0 and (current_52w_high - current_price) / current_52w_high >= 0.02:
            result['rs_ahead_of_price'] = True
            
    return result


# =============================================================================
# 4. PRECISION BUY SETUPS GENERATOR
# =============================================================================

def analyze_stock_setup(ticker: str, stock_df: pd.DataFrame, benchmark_df: pd.DataFrame = None, rs_score: float = 85.0, fundamentals_cache: dict = None) -> dict:
    """
    Evaluates whether a stock currently offers a valid, actionable CANSLIM Buy Trigger.
    Setups evaluated:
      1. Base Breakout (Cup-with-Handle, Flat Base, VCP) within 5% of pivot on volume >= 1.4x
      2. David Ryan RS Line New High (RSNH ahead of price)
      3. Jim Roppel 21 EMA / 50 SMA Institutional Support Bounce with Volume Dry-Up (VDU)
      4. Tight Range Coil / Consolidation Pivot
    """
    res = {
        'ticker': ticker,
        'clean_ticker': ticker.replace('.NS', ''),
        'actionable': False,
        'setup_type': 'None',
        'entry_price': 0.0,
        'pivot_price': 0.0,
        'buy_zone_max': 0.0,
        'current_price': 0.0,
        'dist_from_pivot_pct': 0.0,
        'stop_loss': 0.0,
        'risk_pct': 0.0,
        'volume_ratio': 1.0,
        'adtv_cr': 0.0,
        'industry': 'Unknown',
        'is_thematic_cluster': False,
        'cluster_count': 0,
        'vdu_detected': False,
        'rs_score': float(rs_score),
        'z_score': round(float(rs_score), 2),
        'fundamentals': check_canslim_fundamentals(ticker, fundamentals_cache),
        'minervini_passed': False,
        'vcp_detected': False,
        'base_stage': 'Unknown',
        'rsnh': {'rsnh_active': False, 'rs_ahead_of_price': False}
    }
    
    if stock_df.empty or len(stock_df) < 50:
        return res
        
    df = stock_df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    curr_price = float(close.iloc[-1])
    res['current_price'] = curr_price
    
    # 20-Day Average Daily Turnover in ₹ Crores (Close * Volume / 10,000,000)
    try:
        recent_20 = df.tail(20)
        turnover_s = recent_20['close'] * recent_20['volume']
        res['adtv_cr'] = round(float(turnover_s.mean() / 1e7), 2)
    except Exception:
        res['adtv_cr'] = 0.0
        
    res['industry'] = res['fundamentals'].get('industry', 'Unknown')
    
    # Moving averages
    ema_10 = close.ewm(span=10, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    sma_50 = close.rolling(50).mean()
    
    avg_vol_50 = volume.rolling(50).mean().iloc[-1]
    curr_vol = volume.iloc[-1]
    vol_ratio = curr_vol / avg_vol_50 if avg_vol_50 > 0 else 1.0
    res['volume_ratio'] = round(float(vol_ratio), 2)
    
    # VDU check
    vdu = check_volume_dryup(df)
    res['vdu_detected'] = bool(vdu)
    
    # Minervini & VCP & Base Stage
    res['minervini_passed'] = check_minervini_trend_template(df, rs_score=rs_score).get('passed', False)
    vcp_info = detect_vcp_pattern(df)
    res['vcp_detected'] = vcp_info.get('detected', False)
    stage_info = calculate_base_stage(df)
    res['base_stage'] = stage_info.get('stage_label', 'Unknown')
    
    # David Ryan RSNH
    if benchmark_df is not None and not benchmark_df.empty:
        b_close = benchmark_df['close'] if 'close' in benchmark_df.columns else benchmark_df.iloc[:, 0]
        res['rsnh'] = detect_rs_line_new_high(close, b_close)
        
    c = curr_price
    e21 = float(ema_21.iloc[-1])
    s50 = float(sma_50.iloc[-1]) if pd.notna(sma_50.iloc[-1]) else e21
    
    # Lookback swing high (20-day high excluding today for pivot)
    pivot = float(high.iloc[-25:-1].max()) if len(high) >= 26 else float(high.max())
    res['pivot_price'] = round(pivot, 2)
    
    dist_pct = ((c - pivot) / pivot) * 100
    res['dist_from_pivot_pct'] = round(dist_pct, 2)
    res['buy_zone_max'] = round(pivot * 1.05, 2) # Max 5% buy zone
    
    # 1. Base Breakout (Cup/Flat/VCP)
    is_breakout = (c >= pivot) and (dist_pct <= CANSLIM_SWING_SETTINGS.get('BUY_ZONE_MAX_PCT', 5.0)) and (vol_ratio >= CANSLIM_SWING_SETTINGS.get('BREAKOUT_VOL_MULTIPLIER', 1.4))
    
    # 2. 21 EMA / 50 SMA Institutional Bounce
    yesterday_low = float(low.iloc[-2]) if len(low) >= 2 else float(low.iloc[-1])
    today_low = float(low.iloc[-1])
    touched_21 = abs(min(yesterday_low, today_low) - e21) / e21 <= 0.015 or min(yesterday_low, today_low) <= e21
    # Bounce must show genuine reversal, normal/above-average volume (>= 0.8x), and not be deep inside base (>= pivot * 0.97)
    bounced_21 = (c > e21) and (c > close.iloc[-2]) and touched_21 and (c <= e21 * 1.05) and (vol_ratio >= 0.80) and (c >= pivot * 0.97)
    
    # 3. Consolidation Coil Breakout (3-day tight range)
    recent_3 = df.tail(3)
    tight_range = (recent_3['high'].max() - recent_3['low'].min()) / c * 100
    coil_break = (tight_range <= 3.5) and (c > close.iloc[-2]) and (c >= e21) and (vol_ratio >= 1.2)
    
    fund_pass = res['fundamentals']['passed']
    
    # Select Trigger
    if is_breakout and fund_pass:
        res['actionable'] = True
        res['setup_type'] = '🔥 Stage 1/2 Base Breakout' if not res['vcp_detected'] else '🔥 VCP Base Breakout'
        res['entry_price'] = round(c, 2)
        stop = max(float(low.iloc[-1]), pivot * 0.95)
        res['stop_loss'] = round(stop, 2)
        res['risk_pct'] = round(((c - stop) / c) * 100, 2)
        
    elif bounced_21 and fund_pass:
        res['actionable'] = True
        res['setup_type'] = '🛡️ 21 EMA Institutional Support Bounce'
        res['entry_price'] = round(c, 2)
        stop = min(today_low, e21 * 0.975)
        res['stop_loss'] = round(stop, 2)
        res['risk_pct'] = round(((c - stop) / c) * 100, 2)
        
    elif res['rsnh']['rs_ahead_of_price'] and (abs(dist_pct) <= 3.0) and fund_pass:
        res['actionable'] = True
        res['setup_type'] = '👑 David Ryan RS Line New High (RSNH)'
        res['entry_price'] = round(c, 2)
        stop = float(low.iloc[-5:].min())
        res['stop_loss'] = round(stop, 2)
        res['risk_pct'] = round(((c - stop) / c) * 100, 2)
        
    elif (c > pivot * 1.05):
        res['actionable'] = False
        res['setup_type'] = '🚨 Extended from Base (>5% above Pivot)'
        
    elif touched_21 and (c > e21) and (vol_ratio < 0.80):
        res['actionable'] = False
        res['setup_type'] = '⏳ Setting Up: 21 EMA Support Test (Low Vol/VDU)'
        
    elif (-4.5 <= dist_pct < 0.0) and (c >= e21 * 0.985):
        res['actionable'] = False
        res['setup_type'] = '⏳ Setting Up in Base (Within 4% of Pivot)'
        
    elif (c < pivot):
        res['actionable'] = False
        res['setup_type'] = 'Base Consolidation'
        
    return res


def extract_ticker_ohlcv(history_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Safely extracts and standardizes OHLCV price series for a single ticker from history_df.
    Gracefully handles MultiIndex columns, single-level columns, Series/DataFrame variants,
    and missing tickers. Returns a clean DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
    """
    if history_df is None or history_df.empty or not ticker:
        return pd.DataFrame()
        
    clean_t = str(ticker).replace('.NS', '').strip().upper()
    full_t = f"{clean_t}.NS"
    
    sub = None
    
    try:
        if isinstance(history_df.columns, pd.MultiIndex):
            l0_values = set(history_df.columns.get_level_values(0))
            if full_t in l0_values:
                sub = history_df[full_t]
            elif clean_t in l0_values:
                sub = history_df[clean_t]
            else:
                l1_values = set(history_df.columns.get_level_values(1))
                if full_t in l1_values:
                    sub = history_df.xs(full_t, level=1, axis=1)
                elif clean_t in l1_values:
                    sub = history_df.xs(clean_t, level=1, axis=1)
        else:
            cols = set(history_df.columns)
            if full_t in cols:
                sub = history_df[[full_t]]
            elif clean_t in cols:
                sub = history_df[[clean_t]]
    except Exception:
        return pd.DataFrame()
        
    if sub is None or sub.empty:
        return pd.DataFrame()
        
    # If sub is a Series (e.g. only Close series)
    if isinstance(sub, pd.Series):
        s_clean = pd.to_numeric(sub, errors='coerce').dropna()
        if s_clean.empty:
            return pd.DataFrame()
        return pd.DataFrame({
            'open': s_clean,
            'high': s_clean,
            'low': s_clean,
            'close': s_clean,
            'volume': pd.Series(1, index=s_clean.index)
        })
        
    # Standardize columns to lowercase
    col_dict = {}
    for c in sub.columns:
        c_low = str(c).lower().strip()
        if c_low not in col_dict:
            col_dict[c_low] = c
            
    res = pd.DataFrame(index=sub.index)
    
    if 'close' in col_dict:
        c_val = sub[col_dict['close']]
        if isinstance(c_val, pd.DataFrame):
            c_val = c_val.iloc[:, 0]
        res['close'] = pd.to_numeric(c_val, errors='coerce')
    elif not sub.empty:
        c_val = sub.iloc[:, 0]
        res['close'] = pd.to_numeric(c_val, errors='coerce')
    else:
        return pd.DataFrame()
        
    for target_col in ['open', 'high', 'low']:
        if target_col in col_dict:
            val = sub[col_dict[target_col]]
            if isinstance(val, pd.DataFrame):
                val = val.iloc[:, 0]
            res[target_col] = pd.to_numeric(val, errors='coerce')
        else:
            res[target_col] = res['close']
            
    if 'volume' in col_dict:
        v_val = sub[col_dict['volume']]
        if isinstance(v_val, pd.DataFrame):
            v_val = v_val.iloc[:, 0]
        res['volume'] = pd.to_numeric(v_val, errors='coerce').fillna(1)
    else:
        res['volume'] = pd.Series(1, index=res.index)
        
    return res.dropna(subset=['close'])


# =============================================================================
# 5. AVERAGING UP / PYRAMIDING SIGNALS
# =============================================================================

def detect_averaging_up_signals(active_positions: list, history_df: pd.DataFrame) -> list:
    """
    Evaluates open winning positions for disciplined pyramiding / averaging-up opportunities:
    - Position must be in profit >= +2.5%
    - Price must be coiling near 21 EMA (<= 5% extension)
    - Low-volume test (VDU) or 3-day tight consolidation
    - Elevates stop-loss to higher low to guarantee capital preservation!
    """
    pyramid_signals = []
    min_gain = CANSLIM_SWING_SETTINGS.get('PYRAMID_MIN_GAIN_PCT', 2.5)
    max_ext = CANSLIM_SWING_SETTINGS.get('PYRAMID_MAX_EXTENSION_21EMA', 5.0)
    
    for pos in active_positions:
        try:
            ticker = pos.get('ticker', '')
            avg_cost = float(pos.get('avg_buy_price', 0.0) or 0.0)
            qty = float(pos.get('quantity', 0.0) or 0.0)
            
            clean_t = str(ticker).replace('.NS', '').strip().upper()
            
            stock_df = extract_ticker_ohlcv(history_df, ticker)
            if stock_df.empty or len(stock_df) < 25:
                continue
                
            close = stock_df['close']
            volume = stock_df['volume']
            
            curr_price = float(close.iloc[-1])
            gain_pct = ((curr_price - avg_cost) / avg_cost) * 100 if avg_cost > 0 else 0.0
            
            if gain_pct < min_gain:
                continue
                
            ema_21 = close.ewm(span=21, adjust=False).mean()
            curr_e21 = float(ema_21.iloc[-1])
            dist_from_21 = ((curr_price - curr_e21) / curr_e21) * 100
            
            # Position must be holding at or above 21 EMA (not closing below it) and within max extension
            if dist_from_21 > max_ext or curr_price < curr_e21:
                continue
                
            vol_50_series = volume.rolling(50).mean()
            vol_50 = float(vol_50_series.iloc[-1]) if not vol_50_series.empty and pd.notna(vol_50_series.iloc[-1]) else 1.0
            curr_vol = float(volume.iloc[-1])
            vdu = curr_vol < (vol_50 * 0.85)
            
            recent_hl = float(close.tail(5).min())
            new_raised_stop = max(recent_hl, avg_cost * 1.01)
            
            cur_stop = float(pos.get('current_stop') or round(avg_cost * 0.95, 2))
            
            # 20-Day Average Daily Turnover in ₹ Crores
            try:
                recent_20 = stock_df.tail(20)
                turnover_s = recent_20['close'] * recent_20['volume']
                adtv_cr = round(float(turnover_s.mean() / 1e7), 2)
            except Exception:
                adtv_cr = 0.0
                
            ind = pos.get('industry', 'Unknown')
            
            pyramid_signals.append({
                'ticker': clean_t,
                'current_price': round(curr_price, 2),
                'avg_cost': round(avg_cost, 2),
                'current_gain_pct': round(gain_pct, 2),
                'extension_from_21ema': round(dist_from_21, 2),
                'trigger_reason': '🛡️ 21 EMA Support with Volume Dry-Up' if vdu else '⚡ Tight Coil Consolidation',
                'recommended_tranche': '30% (Secondary Add)',
                'current_stop': round(cur_stop, 2),
                'new_raised_stop': round(new_raised_stop, 2),
                'risk_locked_in': round(((new_raised_stop - avg_cost) / avg_cost) * 100, 2),
                'adtv_cr': adtv_cr,
                'industry': ind
            })
        except Exception as ex:
            print(f"[CANSLIM Engine] Pyramiding check skip {pos.get('ticker')}: {ex}")
            continue
        
    return pyramid_signals


# =============================================================================
# 6. ACTIVE SWING SELL & STOP-LOSS SIGNALS
# =============================================================================

def evaluate_swing_sell_signals(ticker: str, entry_price: float, entry_date: str, stock_df: pd.DataFrame, is_8_week_hold: bool = False, current_stop: float = None, industry: str = 'Unknown') -> dict:
    """
    Evaluates active swing exit triggers:
      1. Hard Stop-Loss Breach: Price <= initial logical stop or loss >= 7.5%
      2. The "2-Close 21 EMA Rule": 2 consecutive daily closes below 21 EMA triggers a 100% exit!
      3. Climax Exhaustion (CED): Parabolic run (>20% above 10 EMA) triggers 50% trim into strength.
      4. 8-Week Hold Rule Protection: Suppresses 21 EMA exit if +20% gain was achieved in <= 3 weeks.
    """
    res = {
        'ticker': ticker.replace('.NS', ''),
        'action': 'HOLD',
        'badge': '🟢 HEALTHY',
        'reason': 'Position holding above 21 EMA',
        'avg_buy_price': round(entry_price, 2),
        'current_price': entry_price,
        'pnl_pct': 0.0,
        'consecutive_below_21ema': 0,
        'current_stop': current_stop or round(entry_price * 0.95, 2),
        'is_8_week_hold': is_8_week_hold,
        'adtv_cr': 0.0,
        'industry': industry
    }
    
    if stock_df.empty or len(stock_df) < 25:
        return res
        
    close = stock_df['close']
    high = stock_df['high']
    volume = stock_df['volume']
    
    curr_price = float(close.iloc[-1])
    res['current_price'] = round(curr_price, 2)
    
    # 20-Day Average Daily Turnover in ₹ Crores
    try:
        recent_20 = stock_df.tail(20)
        turnover_s = recent_20['close'] * recent_20['volume']
        res['adtv_cr'] = round(float(turnover_s.mean() / 1e7), 2)
    except Exception:
        res['adtv_cr'] = 0.0
    pnl_pct = ((curr_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
    res['pnl_pct'] = round(pnl_pct, 2)
    
    # 8-Week Hold Check
    if not is_8_week_hold and pnl_pct >= CANSLIM_SWING_SETTINGS.get('EIGHT_WEEK_HOLD_GAIN_PCT', 20.0):
        res['is_8_week_hold'] = True
        is_8_week_hold = True
        
    # Check Initial Stop Loss
    eff_stop = current_stop if (current_stop and current_stop > 0) else entry_price * (1 - CANSLIM_SWING_SETTINGS.get('INITIAL_STOP_LOSS_PCT', 4.5) / 100)
    hard_max_stop = entry_price * (1 - CANSLIM_SWING_SETTINGS.get('HARD_MAX_STOP_PCT', 7.5) / 100)
    
    if curr_price <= eff_stop or curr_price <= hard_max_stop:
        res['action'] = 'EXIT'
        res['badge'] = '🔴 STOP-LOSS BREACH'
        res['reason'] = f"Price breached stop level ₹{eff_stop:.2f} (Loss: {pnl_pct:.2f}%)"
        return res
        
    # Moving Averages
    ema_10 = close.ewm(span=10, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    sma_50 = close.rolling(50).mean()
    
    c_today = float(close.iloc[-1])
    c_prev = float(close.iloc[-2])
    e21_today = float(ema_21.iloc[-1])
    e21_prev = float(ema_21.iloc[-2])
    
    below_today = c_today < e21_today
    below_prev = c_prev < e21_prev
    
    consec_below = 2 if (below_today and below_prev) else (1 if below_today else 0)
    res['consecutive_below_21ema'] = consec_below
    
    # Climax Exhaustion check
    e10_today = float(ema_10.iloc[-1])
    ext_10ema = ((c_today - e10_today) / e10_today) * 100
    if ext_10ema >= 20.0 and pnl_pct >= 25.0:
        res['action'] = 'TRIM'
        res['badge'] = '🟠 CLIMAX EXHAUSTION'
        res['reason'] = f"Parabolic extension: +{ext_10ema:.1f}% above 10 EMA. Trim 50% profit into strength."
        return res
        
    # 2 Closes Below 21 EMA Rule
    if consec_below >= CANSLIM_SWING_SETTINGS.get('TRAILING_EXIT_CONSECUTIVE_DAYS_21EMA', 2):
        if is_8_week_hold:
            s50_today = float(sma_50.iloc[-1]) if pd.notna(sma_50.iloc[-1]) else e21_today
            if c_today < s50_today:
                res['action'] = 'EXIT'
                res['badge'] = '🔴 50 SMA BREAK (8-WEEK HOLD VIOLATED)'
                res['reason'] = "Institutional floor (50 SMA) breached while under 8-Week Hold rule."
                return res
            else:
                res['action'] = 'HOLD'
                res['badge'] = '🔒 8-WEEK HOLD ACTIVE'
                res['reason'] = "2 closes below 21 EMA absorbed under O'Neil 8-Week Hold protection (holding above 50 SMA)."
                return res
        else:
            res['action'] = 'EXIT'
            res['badge'] = '🔴 SELL: 2 CLOSES BELOW 21 EMA'
            res['reason'] = f"Confirmed loss of swing momentum: 2 consecutive closes below 21 EMA (₹{e21_today:.2f})."
            return res
            
    if consec_below == 1:
        res['action'] = 'WATCH'
        res['badge'] = '⚠️ 1 CLOSE BELOW 21 EMA'
        res['reason'] = f"Warning: Closed below 21 EMA today (₹{c_today:.2f} < ₹{e21_today:.2f}). Watch tomorrow for confirm/reclaim."
        return res
        
    dist_above_21 = ((c_today - e21_today) / e21_today) * 100
    res['badge'] = f"🟢 +{dist_above_21:.1f}% ABOVE 21 EMA"
    return res


# =============================================================================
# 7. MULTI-DAY BREAKOUT TRIGGER SCANNER (TRAILING 10 TRADING DAYS / 2 WEEKS)
# =============================================================================

def scan_recent_breakout_triggers(top50: list, history_df: pd.DataFrame, fundamentals_cache: dict, lookback_days: int = 10) -> list:
    """
    Scans the Top 50 CANSLIM momentum pool across the trailing N trading days (default 10 days / 2 weeks).
    Preserves breakout signals so users who check weekly don't miss entries.
    Classifies current actionable state:
      - 🟢 In Buy Zone (Actionable) (0% to +5.5% from pivot)
      - 🎯 Retesting Pivot Support (-1.5% to 0% from pivot)
      - 🚨 Extended (>5% from Pivot)
      - 🔴 Failed Breakout (< 21 EMA)
    """
    recent_triggers = []
    seen_tickers = set()

    for rank, (ticker, z_score) in enumerate(top50, 1):
        clean_t = str(ticker).replace('.NS', '').strip().upper()
        if clean_t in seen_tickers:
            continue
            
        fd = check_canslim_fundamentals(ticker, fundamentals_cache)
        if not fd['passed']:
            continue
            
        sdf = extract_ticker_ohlcv(history_df, ticker)
        if sdf.empty or len(sdf) < 50:
            continue
            
        close = sdf['close']
        high = sdf['high']
        low = sdf['low']
        volume = sdf['volume']
        
        cmp_now = float(close.iloc[-1])
        ema_21_s = close.ewm(span=21, adjust=False).mean()
        e21_now = float(ema_21_s.iloc[-1])
        
        # Check each bar backwards in trailing lookback window
        for i in range(lookback_days):
            idx = len(sdf) - 1 - i
            if idx < 30: 
                break
                
            c_bar = float(close.iloc[idx])
            date_bar = str(sdf.index[idx]).split(' ')[0]
            
            # Lookback 25-day pivot prior to that bar
            pivot = float(high.iloc[idx-25:idx].max()) if idx >= 26 else float(high.iloc[:idx].max())
            vol_avg = float(volume.iloc[idx-50:idx].mean()) if idx >= 50 else float(volume.iloc[:idx].mean())
            vol_bar = float(volume.iloc[idx])
            vol_ratio = vol_bar / vol_avg if vol_avg > 0 else 1.0
            
            dist_at_bar = ((c_bar - pivot) / pivot) * 100
            
            # 1. Base Breakout on heavy volume
            is_breakout = (c_bar >= pivot) and (0.0 <= dist_at_bar <= 5.0) and (vol_ratio >= 1.3)
            
            # 2. Institutional 21 EMA Bounce on volume
            e21_bar = float(ema_21_s.iloc[idx])
            low_bar = float(low.iloc[idx])
            touched_21 = abs(low_bar - e21_bar) / e21_bar <= 0.015 or low_bar <= e21_bar
            is_bounce = (c_bar > e21_bar) and (c_bar > float(close.iloc[idx-1])) and touched_21 and (c_bar >= pivot * 0.97) and (vol_ratio >= 1.0)
            
            if is_breakout or is_bounce:
                setup_name = "🔥 Base Breakout" if is_breakout else "🛡️ 21 EMA Support Bounce"
                dist_now = ((cmp_now - pivot) / pivot) * 100
                
                # Actionable classification right now:
                if cmp_now < e21_now * 0.97 or dist_now < -5.0:
                    cur_status = "🔴 Failed Breakout (< 21 EMA)"
                    badge_style = "failed"
                    is_actionable = False
                elif dist_now > 5.5:
                    cur_status = "🚨 Extended (>5% from Pivot)"
                    badge_style = "extended"
                    is_actionable = False
                elif -1.5 <= dist_now < 0.0:
                    cur_status = "🎯 Retesting Pivot Support"
                    badge_style = "retest"
                    is_actionable = True
                elif 0.0 <= dist_now <= 5.5:
                    cur_status = "🟢 In Buy Zone (Actionable)"
                    badge_style = "in_zone"
                    is_actionable = True
                else:
                    cur_status = "⏳ Inside Base (< Pivot)"
                    badge_style = "base"
                    is_actionable = False
                    
                stop = round(max(float(low.iloc[idx]), pivot * 0.95), 2)
                risk_pct = round(((cmp_now - stop) / cmp_now) * 100, 2)
                
                # 20-Day Average Daily Turnover in ₹ Crores
                try:
                    recent_20 = sdf.tail(20)
                    turnover_s = recent_20['close'] * recent_20['volume']
                    adtv_cr = round(float(turnover_s.mean() / 1e7), 2)
                except Exception:
                    adtv_cr = 0.0
                    
                recent_triggers.append({
                    'ticker': clean_t,
                    'clean_ticker': clean_t,
                    'setup_type': setup_name,
                    'trigger_date': date_bar,
                    'days_ago': i,
                    'pivot_price': round(pivot, 2),
                    'trigger_price': round(c_bar, 2),
                    'current_price': round(cmp_now, 2),
                    'dist_now_pct': round(dist_now, 2),
                    'stop_loss': stop,
                    'risk_pct': risk_pct,
                    'volume_ratio': round(vol_ratio, 2),
                    'adtv_cr': adtv_cr,
                    'industry': fd.get('industry', 'Unknown'),
                    'is_thematic_cluster': False,
                    'cluster_count': 0,
                    'status': cur_status,
                    'badge_style': badge_style,
                    'actionable': is_actionable,
                    'sales_growth': fd.get('sales_growth', 0.0),
                    'z_score': round(float(z_score), 2)
                })
                seen_tickers.add(clean_t)
                break # take the most recent breakout for this ticker

    # Persist to database signal_history for permanent archiving
    try:
        from database import save_signal
        for trig in recent_triggers:
            save_signal({
                'date_generated': trig['trigger_date'],
                'ticker': trig['clean_ticker'],
                'setup_type': trig['setup_type'],
                'regime': 'CANSLIM Leader Breakout',
                'entry_price': trig['pivot_price'],
                'stop_price': trig['stop_loss'],
                'risk_percent': trig['risk_pct'],
                'holding_bias': 'Swing',
                'confidence_score': trig['volume_ratio']
            })
    except Exception as ex:
        print(f"[CANSLIM Engine] Note on signal persistence: {ex}")

    # Sort recent triggers: actionable first, then most recent days_ago ascending
    recent_triggers.sort(key=lambda x: (not x['actionable'], x['days_ago'], -x['sales_growth']))
    return recent_triggers


# =============================================================================
# 8. UNIFIED DAILY SWING ENGINE ORCHESTRATOR
# =============================================================================

def generate_daily_canslim_swing_state(today_date=None, portfolio_positions=None, universe_mode="nifty_750"):
    """
    Main entry point for the CANSLIM Swing Trading Terminal.
    Executes the entire multi-step pipeline across the selected universe:
      1. Computes / loads Fast Momentum Matrix
      2. Pulls Top 50 Momentum Leaders
      3. Filters through CANSLIM Fundamentals (EPS/Sales >= 25%)
      4. Detects today's precision Buy Signals
      5. Scans open positions for Averaging-Up / Pyramiding
      6. Evaluates open positions for 2-Close 21 EMA Exits
      7. Prepares Top 50 Leaderboard
    
    Args:
        universe_mode: "nifty_750" (default) or "deep_market" (~2500+ NSE stocks)
    """
    from systematic_engine import compute_live_fast_momentum_matrix
    from price_history_manager import fetch_incremental_history
    import yfinance as yf
    
    score_matrix, close_df = compute_live_fast_momentum_matrix(universe_mode=universe_mode)
    if score_matrix is None or score_matrix.empty:
        return None
        
    z_score_map = {}
    for col, val in score_matrix.iloc[-1].items():
        if pd.notna(val):
            c_sym = str(col).replace('.NS', '').replace('.BO', '').strip().upper()
            z_score_map[c_sym] = round(float(val), 2)
        
    if today_date is None:
        today_date = score_matrix.index[-1]
        
    # 1. Fetch benchmark for RS Line and Market Regime
    benchmark_data = None
    try:
        benchmark_df = yf.download(BENCHMARK_TICKER, period="400d", progress=False)
        if benchmark_df is not None and not benchmark_df.empty:
            if isinstance(benchmark_df.columns, pd.MultiIndex):
                l0 = [str(c).lower() for c in benchmark_df.columns.get_level_values(0)]
                price_lvl = 0 if 'close' in l0 else 1
                b_flat = benchmark_df.copy()
                b_flat.columns = [str(c).lower().strip() for c in benchmark_df.columns.get_level_values(price_lvl)]
            else:
                b_flat = benchmark_df.copy()
                b_flat.columns = [str(c).lower().strip() for c in benchmark_df.columns]
                
            def _get_1d(df, col):
                if col in df.columns:
                    s = df[col]
                    if isinstance(s, pd.DataFrame):
                        s = s.iloc[:, 0]
                    return pd.to_numeric(s, errors='coerce')
                return pd.Series(dtype=float, index=df.index)
                
            benchmark_data = pd.DataFrame({
                'close': _get_1d(b_flat, 'close'),
                'open': _get_1d(b_flat, 'open'),
                'high': _get_1d(b_flat, 'high'),
                'low': _get_1d(b_flat, 'low'),
                'volume': _get_1d(b_flat, 'volume')
            }).dropna(subset=['close'])
    except Exception as e:
        print(f"[CANSLIM Engine] Benchmark download fallback: {e}")
        
    # 2. Market Regime Evaluation
    dd_count = 0
    ftd_active = False
    regime_label = "Confirmed Uptrend"
    exposure_pct = 100
    if benchmark_data is not None and not benchmark_data.empty and 'close' in benchmark_data.columns:
        try:
            dd_count, _ = calculate_distribution_days(benchmark_data)
            ftd_active, _ = detect_follow_through_day(benchmark_data)
            curr = float(benchmark_data['close'].iloc[-1])
            sma50 = float(benchmark_data['close'].rolling(50).mean().iloc[-1]) if len(benchmark_data) >= 50 else curr
            regime_label, _, _ = get_market_regime_label(dd_count, curr, sma50)
            if dd_count >= 5:
                regime_label = "Market Under Heavy Pressure (Defensive)"
                exposure_pct = 40
            elif dd_count >= 3:
                exposure_pct = 70
        except Exception as e:
            print(f"[CANSLIM Engine] Market regime error: {e}")
            
    # 3. Top 50 Momentum Leaders
    top50 = get_top_momentum_candidates(score_matrix, today_date, n=50)
    top50_tickers = [t[0] for t in top50]
    
    # 4. Load full history for top 50 + portfolio positions
    portfolio_tickers = []
    if portfolio_positions:
        for p in portfolio_positions:
            t = p.get('ticker')
            if t:
                clean_t = str(t).replace('.NS', '').replace('.BO', '').strip().upper()
                if clean_t.isdigit():
                    portfolio_tickers.append(f"{clean_t}.BO")
                else:
                    portfolio_tickers.append(f"{clean_t}.NS")
                
    all_needed_tickers = list(dict.fromkeys(top50_tickers + portfolio_tickers))
    history_df = fetch_incremental_history(all_needed_tickers, days=350)
    fundamentals_cache = get_all_fundamentals_cache()
    
    # 4.5 Dynamically calculate Momentum Z-scores for any portfolio positions missing from the NIFTY 750 matrix
    if portfolio_positions and close_df is not None and not close_df.empty:
        try:
            from systematic_engine import calculate_returns_and_volatility
            _, u_vol, u_r1, u_r3, u_r6 = calculate_returns_and_volatility(close_df)
            u_safe_vol = u_vol.replace(0, np.nan).clip(lower=0.01)
            u_mr1 = u_r1 / u_safe_vol
            u_mr3 = u_r3 / u_safe_vol
            u_mr6 = u_r6 / u_safe_vol
            
            mean_mr1, std_mr1 = float(u_mr1.iloc[-1].mean()), float(u_mr1.iloc[-1].std())
            mean_mr3, std_mr3 = float(u_mr3.iloc[-1].mean()), float(u_mr3.iloc[-1].std())
            mean_mr6, std_mr6 = float(u_mr6.iloc[-1].mean()), float(u_mr6.iloc[-1].std())
            
            for pos in portfolio_positions:
                t = pos.get('ticker')
                if not t:
                    continue
                clean_t = str(t).replace('.NS', '').replace('.BO', '').strip().upper()
                if clean_t in z_score_map:
                    continue
                    
                stock_df = extract_ticker_ohlcv(history_df, t)
                if stock_df.empty or len(stock_df) < 25:
                    continue
                    
                c = stock_df['close']
                log_ret = np.log(c / c.shift(1))
                s_vol = float(log_ret.tail(252).std() * np.sqrt(252))
                if pd.isna(s_vol) or s_vol < 0.01:
                    s_vol = float(log_ret.std() * np.sqrt(252)) if len(log_ret) >= 25 else 0.30
                safe_s_vol = max(s_vol, 0.01)
                
                s_r1 = float(c.iloc[-1] / c.iloc[-22] - 1) if len(c) >= 22 else 0.0
                s_r3 = float(c.iloc[-1] / c.iloc[-64] - 1) if len(c) >= 64 else s_r1
                s_r6 = float(c.iloc[-1] / c.iloc[-127] - 1) if len(c) >= 127 else s_r3
                
                z1 = (s_r1 / safe_s_vol - mean_mr1) / std_mr1 if std_mr1 > 0 else 0.0
                z3 = (s_r3 / safe_s_vol - mean_mr3) / std_mr3 if std_mr3 > 0 else 0.0
                z6 = (s_r6 / safe_s_vol - mean_mr6) / std_mr6 if std_mr6 > 0 else 0.0
                w_z = 0.4 * z1 + 0.4 * z3 + 0.2 * z6
                norm_score = 1 + w_z if w_z >= 0 else 1 / (1 - w_z)
                z_score_map[clean_t] = round(float(norm_score), 2)
        except Exception as ex:
            print(f"[CANSLIM Engine] Note on dynamic portfolio momentum scores: {ex}")
    
    # 5. Pre-scan industries across Top 50 to detect Thematic Clusters (>= 3 stocks in same industry)
    industry_cluster_counts = {}
    for rank, (ticker, z_score) in enumerate(top50, 1):
        fd = check_canslim_fundamentals(ticker, fundamentals_cache)
        ind = fd.get('industry', 'Unknown')
        if ind and ind != 'Unknown':
            industry_cluster_counts[ind] = industry_cluster_counts.get(ind, 0) + 1

    buys_today = []
    radar_setups = []
    leaderboard = []
    
    for rank, (ticker, z_score) in enumerate(top50, 1):
        try:
            clean_t = str(ticker).replace('.NS', '').strip().upper()
            stock_df = extract_ticker_ohlcv(history_df, ticker)
            
            setup_data = analyze_stock_setup(ticker, stock_df, benchmark_data, rs_score=float(z_score), fundamentals_cache=fundamentals_cache)
            
            ind = setup_data.get('industry', 'Unknown')
            c_cnt = industry_cluster_counts.get(ind, 0)
            is_cluster = c_cnt >= 3
            setup_data['is_thematic_cluster'] = is_cluster
            setup_data['cluster_count'] = c_cnt
            
            leaderboard.append({
                'Rank': int(rank),
                'Ticker': clean_t,
                'Z_Score': round(float(z_score), 2),
                'Price': float(setup_data['current_price']),
                'Setup': str(setup_data['setup_type']),
                'In_Buy_Zone': bool(setup_data['actionable']),
                'Pivot': float(setup_data['pivot_price']),
                'Dist_Pivot': float(setup_data['dist_from_pivot_pct']),
                'Vol_Ratio': float(setup_data['volume_ratio']),
                'ADTV_Cr': float(setup_data.get('adtv_cr', 0.0)),
                'Industry': str(ind),
                'Is_Cluster': is_cluster,
                'Cluster_Count': int(c_cnt),
                'EPS_YoY': float(setup_data['fundamentals']['eps_growth']),
                'Sales_YoY': float(setup_data['fundamentals']['sales_growth']),
                'ROE': float(setup_data['fundamentals']['roe']),
                'Fund_Pass': bool(setup_data['fundamentals']['passed']),
                'Base_Stage': str(setup_data['base_stage']),
                'RSNH': bool(setup_data['rsnh']['rsnh_active'])
            })
            
            if setup_data['actionable']:
                buys_today.append(setup_data)
            elif 'Setting Up' in setup_data['setup_type']:
                radar_setups.append(setup_data)
        except Exception as ex:
            print(f"[CANSLIM Engine] Error scanning top50 candidate {ticker}: {ex}")
            continue
            
    # 6. Scan Active Model Portfolio & Universe for Averaging Up and Exits
    pyramids_today = []
    exits_today = []
    active_portfolio_status = []
    
    # If no external portfolio passed, autonomously construct the Systematic Swing Model Portfolio
    # from the top qualifying CANSLIM leaders holding above 21 EMA in the NIFTY 750 universe!
    if not portfolio_positions:
        model_positions = []
        for rank, (ticker, z_score) in enumerate(top50, 1):
            clean_t = str(ticker).replace('.NS', '').strip().upper()
            stock_df = extract_ticker_ohlcv(history_df, ticker)
            if stock_df.empty or len(stock_df) < 30:
                continue
                
            close = stock_df['close']
            high = stock_df['high']
            
            c_today = float(close.iloc[-1])
            ema_21 = close.ewm(span=21, adjust=False).mean()
            e21_today = float(ema_21.iloc[-1])
            
            fund_data = check_canslim_fundamentals(ticker, fundamentals_cache)
            pivot = float(high.iloc[-25:-1].max()) if len(high) >= 26 else float(high.max())
            
            # Must pass CANSLIM fundamentals (Sales >= 25%), hold above 21 EMA, and have broken out of base pivot
            if fund_data['passed'] and (c_today >= e21_today * 0.985) and (c_today >= pivot * 0.98):
                model_positions.append({
                    'ticker': clean_t,
                    'avg_buy_price': round(pivot, 2),
                    'quantity': 100,
                    'current_stop': round(pivot * 0.95, 2),
                    'buy_date': 'Model Pivot',
                    'is_8_week_hold': ((c_today - pivot) / pivot) >= 0.20
                })
                if len(model_positions) >= CANSLIM_SWING_SETTINGS.get('MAX_POSITIONS', 8):
                    break
        positions_to_evaluate = model_positions
    else:
        positions_to_evaluate = portfolio_positions
        
    if positions_to_evaluate:
        pyramids_today = detect_averaging_up_signals(positions_to_evaluate, history_df)
        
        for pos in positions_to_evaluate:
            try:
                t = pos.get('ticker', '')
                entry_p = float(pos.get('avg_buy_price', 0.0) or 0.0)
                entry_d = str(pos.get('buy_date', ''))
                is_hold = bool(pos.get('is_8_week_hold', False))
                cur_stop = pos.get('current_stop', None)
                if cur_stop is not None:
                    cur_stop = float(cur_stop)
                
                clean_t = str(t).replace('.NS', '').strip().upper()
                stock_df = extract_ticker_ohlcv(history_df, t)
                
                sell_eval = evaluate_swing_sell_signals(clean_t, entry_p, entry_d, stock_df, is_8_week_hold=is_hold, current_stop=cur_stop, industry=pos.get('industry', 'Unknown'))
                active_portfolio_status.append(sell_eval)
                
                if sell_eval['action'] in ['EXIT', 'TRIM']:
                    exits_today.append(sell_eval)
            except Exception as ex:
                print(f"[CANSLIM Engine] Error evaluating swing position {pos.get('ticker')}: {ex}")
                continue
                
    # If evaluating the autonomous model portfolio, also scan remaining Top 50 momentum pool for any leader that suffered 2 consecutive closes below 21 EMA
    if not portfolio_positions:
        existing_exit_tickers = {e['ticker'] for e in exits_today}
        for rank, (ticker, z_score) in enumerate(top50, 1):
            clean_t = str(ticker).replace('.NS', '').strip().upper()
            if clean_t in existing_exit_tickers:
                continue
            try:
                stock_df = extract_ticker_ohlcv(history_df, ticker)
                if stock_df.empty or len(stock_df) < 25:
                    continue
                close = stock_df['close']
                c_today = float(close.iloc[-1])
                c_prev = float(close.iloc[-2])
                ema_21 = close.ewm(span=21, adjust=False).mean()
                e21_today = float(ema_21.iloc[-1])
                e21_prev = float(ema_21.iloc[-2])
                
                if c_today < e21_today and c_prev < e21_prev:
                    pivot = float(stock_df['high'].iloc[-25:-1].max()) if len(stock_df) >= 26 else float(stock_df['high'].max())
                    pnl_pct = ((c_today - pivot) / pivot) * 100
                    recent_20 = stock_df.tail(20)
                    adtv_cr = round(float((recent_20['close'] * recent_20['volume']).mean() / 1e7), 2) if not recent_20.empty else 0.0
                    fd = check_canslim_fundamentals(ticker, fundamentals_cache)
                    exits_today.append({
                        'ticker': clean_t,
                        'action': 'EXIT',
                        'badge': '🔴 SELL: 2 CLOSES BELOW 21 EMA',
                        'reason': f"Momentum leader lost 21 EMA swing support (₹{c_today:.2f} < ₹{e21_today:.2f}).",
                        'current_price': round(c_today, 2),
                        'pnl_pct': round(pnl_pct, 2),
                        'consecutive_below_21ema': 2,
                        'current_stop': round(pivot * 0.95, 2),
                        'is_8_week_hold': False,
                        'adtv_cr': adtv_cr,
                        'industry': fd.get('industry', 'Unknown')
                    })
                    existing_exit_tickers.add(clean_t)
            except Exception:
                continue
            
    # 7. Strict Mutual Exclusion Enforcement:
    # A ticker can NEVER appear in both Buy Triggers and Sell/Exit Alerts.
    # An active Buy Trigger takes precedence over an exit scan alert.
    # Likewise, a stock triggering an Exit can never be an Averaging Up signal.
    buy_tickers = {b['clean_ticker'] for b in buys_today}
    exits_today = [e for e in exits_today if e['ticker'] not in buy_tickers]
    exit_tickers = {e['ticker'] for e in exits_today}
    buys_today = [b for b in buys_today if b['clean_ticker'] not in exit_tickers]
    pyramids_today = [p for p in pyramids_today if p['ticker'] not in exit_tickers]
    
    # 8. Trailing Multi-Day Breakout Scanner (10 Trading Days / 2 Weeks)
    recent_buy_triggers = scan_recent_breakout_triggers(top50, history_df, fundamentals_cache, lookback_days=10)
    for r_trig in recent_buy_triggers:
        t_ind = r_trig.get('industry', 'Unknown')
        c_cnt = industry_cluster_counts.get(t_ind, 0)
        r_trig['is_thematic_cluster'] = c_cnt >= 3
        r_trig['cluster_count'] = c_cnt
        if r_trig['ticker'] in exit_tickers:
            r_trig['actionable'] = False
            r_trig['status'] = "🔴 Exited / Below 21 EMA"
            r_trig['badge_style'] = "failed"
            
    # 9. Swing Radar Refinement & Mutual Exclusion:
    # Swing Radar represents the "On-Deck Circle" — strictly PRE-BREAKOUT watch setups!
    # Stocks that have ALREADY triggered breakouts (in recent_buy_triggers) are removed from Radar.
    # Also strictly enforces the CANSLIM Sales Growth gate (>= 25%) and excludes exit tickers.
    triggered_tickers = {b['clean_ticker'] for b in recent_buy_triggers}
    radar_setups = [
        r for r in radar_setups 
        if r['clean_ticker'] not in triggered_tickers 
        and r['clean_ticker'] not in exit_tickers
        and r['fundamentals']['passed']
    ]
    
    # 10. Ensure explicit z_score on every single card across all panels
    for p in pyramids_today:
        p['z_score'] = z_score_map.get(p.get('ticker'), None)
    for e in exits_today:
        e['z_score'] = z_score_map.get(e.get('ticker'), None)
    for s in active_portfolio_status:
        s['z_score'] = z_score_map.get(s.get('ticker'), None)
    for r in radar_setups:
        if 'z_score' not in r or r.get('z_score') is None:
            r['z_score'] = z_score_map.get(r.get('clean_ticker'), None)
    for b in recent_buy_triggers:
        if 'z_score' not in b or b.get('z_score') is None:
            b['z_score'] = z_score_map.get(b.get('clean_ticker'), None)
                
    return {
        'date': today_date,
        'market_regime': {
            'label': regime_label,
            'distribution_days': dd_count,
            'ftd_active': ftd_active,
            'recommended_exposure_pct': exposure_pct
        },
        'buys_today': buys_today,
        'recent_buy_triggers': recent_buy_triggers,
        'pyramids_today': pyramids_today,
        'exits_today': exits_today,
        'radar_setups': radar_setups,
        'active_portfolio_status': active_portfolio_status,
        'leaderboard': pd.DataFrame(leaderboard),
        'z_score_map': z_score_map
    }
