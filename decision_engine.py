# Decision Engine for Kush Tracker
# Implements layered decision logic for trade actions

import pandas as pd
from config import (
    RISK_LIMITS, TREND_RULES, STRUCTURAL_FAILURE, 
    AVERAGING_UP, RS_THRESHOLDS
)
from trend_analyzer import (
    TREND_STRONG, TREND_PULLBACK, TREND_WARNING, TREND_BROKEN,
    determine_trend_state, check_warning_triggers, check_structural_failure
)
from technical_indicators import (
    count_days_above_ema, count_days_below_ema, has_big_down_day, get_atr_state,
    detect_3_weeks_tight, calculate_atr_chandelier_stop, classify_stock_ma_dna
)


# Action constants
ACTION_EXIT = 'EXIT'
ACTION_TRIM = 'TRIM'
ACTION_HOLD = 'HOLD'
ACTION_ADD = 'ADD'


def calculate_position_loss(
    current_price: float, 
    avg_buy_price: float, 
    quantity: float, 
    total_portfolio_value: float
) -> dict:
    """
    Calculate position loss as percentage of portfolio.
    
    Args:
        current_price: Current stock price
        avg_buy_price: Average buy price
        quantity: Number of shares
        total_portfolio_value: Total portfolio value
    
    Returns:
        Dictionary with loss metrics
    """
    if total_portfolio_value <= 0:
        return {'loss_pct': 0, 'loss_amount': 0, 'gain_loss_pct': 0}
    
    import pandas as pd
    import math
    
    if pd.isna(current_price) or math.isnan(current_price):
        current_price = avg_buy_price
        
    position_value = current_price * quantity
    cost_basis = avg_buy_price * quantity
    gain_loss = position_value - cost_basis
    gain_loss_pct = ((current_price / avg_buy_price) - 1) * 100 if avg_buy_price > 0 else 0
    
    # Loss as percentage of total portfolio
    loss_pct = (gain_loss / total_portfolio_value) * 100 if gain_loss < 0 else 0
    
    return {
        'loss_pct': abs(loss_pct),  # Positive number representing loss
        'loss_amount': abs(gain_loss) if gain_loss < 0 else 0,
        'gain_loss_pct': gain_loss_pct,
        'position_value': position_value,
        'portfolio_weight': (position_value / total_portfolio_value) * 100,
    }


def check_layer1_risk(loss_metrics: dict) -> tuple:
    """
    Layer 1: Portfolio-level risk management (HIGHEST PRIORITY).
    
    Rules:
    - Loss >= 2% of portfolio → EXIT 100%
    - Loss >= 1% of portfolio → TRIM 50%
    
    Args:
        loss_metrics: Dictionary with loss calculations
    
    Returns:
        Tuple of (action, reason) or (None, None) if no trigger
    """
    loss_pct = loss_metrics.get('loss_pct', 0)
    
    if loss_pct >= RISK_LIMITS['FULL_EXIT_LOSS_PCT']:
        return (ACTION_EXIT, f"Portfolio loss {loss_pct:.2f}% >= {RISK_LIMITS['FULL_EXIT_LOSS_PCT']}% threshold")
    
    if loss_pct >= RISK_LIMITS['TRIM_LOSS_PCT']:
        return (ACTION_TRIM, f"Portfolio loss {loss_pct:.2f}% >= {RISK_LIMITS['TRIM_LOSS_PCT']}% threshold")
    
    return (None, None)


def check_layer1_5_volatility_stop(df: pd.DataFrame) -> tuple:
    """
    Layer 1.5: Volatility / Chandelier Stop (HIGH PRIORITY).
    
    Rules:
    - Price < 3.5x ATR(21) below 20-day high → EXIT
    """
    if df.empty:
        return (None, None)
        
    current_price = df.iloc[-1]['close']
    stop_price = calculate_atr_chandelier_stop(df, atr_multiplier=3.5, high_window=20, atr_period=21)
    
    if stop_price > 0 and current_price < stop_price:
        return (ACTION_EXIT, f"Volatility Stop: Price dropped below {stop_price:.2f} (3.5x ATR)")
        
    return (None, None)


def check_layer1_5_dead_money(df: pd.DataFrame, loss_metrics: dict) -> tuple:
    """
    Layer 1.5: Opportunity Cost / Dead Money Check.
    
    Rules:
    - Position gain_loss_pct is < 8.0% (no profit cushion)
    - AND either:
      - 1-Month Return (21 trading days) < 4.0%
      - OR 2-Month Return (42 trading days) < 12.0%
      
    If triggered, suggests rotating capital to better acting stocks.
    """
    if df.empty or len(df) < 21:
        return (None, None)
        
    gain_loss_pct = loss_metrics.get('gain_loss_pct', 0)
    
    # If the user has a cushion >= 8%, we allow normal base-building pullbacks.
    if gain_loss_pct >= 8.0:
        return (None, None)
        
    # Calculate returns
    current_price = df.iloc[-1]['close']
    
    ret_21d = 0
    if len(df) >= 21:
        price_21d = df.iloc[-21]['close']
        ret_21d = ((current_price / price_21d) - 1) * 100
        
    ret_42d = 0
    if len(df) >= 42:
        price_42d = df.iloc[-42]['close']
        ret_42d = ((current_price / price_42d) - 1) * 100
        
    # Check Dead Money conditions
    is_dead_money = False
    reason = ""
    
    if len(df) >= 21 and ret_21d < 4.0:
        is_dead_money = True
        reason = f"Dead Money: Up only {ret_21d:.1f}% in 1M with <8% profit cushion"
    elif len(df) >= 42 and ret_42d < 12.0:
        is_dead_money = True
        reason = f"Dead Money: Up only {ret_42d:.1f}% in 2M (8-Week Rule Failure)"
        
    if is_dead_money:
        return (ACTION_TRIM, reason)
        
    return (None, None)


def check_layer2_trend(df: pd.DataFrame, trend_state: str, gain_loss_pct: float = 0.0, ma_dna: dict = None) -> tuple:
    """
    Layer 2: Trend health based actions (upgraded with 21-EMA Low Anti-Wick Floor
    and John Boik 50-SMA Monster Stock Immunity).
    
    Rules:
    - John Boik Monster Immunity: >= 15% profit cushion + Boik 50-SMA DNA + holding 50 SMA -> Suppress 21-EMA trim
    - 21-EMA Low Rider: Only triggers Haber Trim if >= 2 consecutive closes below 21-EMA Low (ema_21_low)
    - Warning + high-vol down day → TRIM 25-50%
    - Broken trend → Reduce to tracking size / Exit
    """
    triggers = check_warning_triggers(df)
    if ma_dna is None:
        ma_dna = classify_stock_ma_dna(df)
    
    # Ross Haber 21-Day Moving Average Double-Close Rule (upgraded with 21-EMA Low & Boik Monster Immunity):
    if triggers.get('haber_double_close_triggered', False) or triggers.get('days_below_21ema', 0) >= 2:
        last_close = float(df['close'].iloc[-1]) if not df.empty and 'close' in df.columns else 0.0
        sma_50 = float(df['sma_50'].iloc[-1]) if (not df.empty and 'sma_50' in df.columns and pd.notna(df['sma_50'].iloc[-1])) else 0.0
        ema_21_low = float(df['ema_21_low'].iloc[-1]) if (not df.empty and 'ema_21_low' in df.columns and pd.notna(df['ema_21_low'].iloc[-1])) else 0.0
        days_below_21_low = count_days_below_ema(df, 'ema_21_low', lookback=10) if ('ema_21_low' in df.columns) else triggers.get('days_below_21ema', 0)

        # 1. John Boik Monster Stock 50-SMA Immunity Check
        is_monster_eligible = (
            gain_loss_pct >= 15.0
            and sma_50 > 0
            and last_close >= sma_50 * 0.99
            and not triggers.get('high_vol_down_day', False)
            and (
                ma_dna.get('dna_type') == '50_SMA_MONSTER'
                or (ma_dna.get('boik_launch_detected', False) and ma_dna.get('zero_heavy_vol_50sma_breaches', True))
            )
        )
        if is_monster_eligible:
            return (
                None,
                f"🦖 Monster 50-SMA Ride Active (+{gain_loss_pct:.1f}% cushion; holding rising 50 SMA ₹{sma_50:.1f})"
            )

        # 2. 21-EMA Low Anti-Wick Channel Check
        if ema_21_low > 0 and days_below_21_low < 2 and last_close >= ema_21_low * 0.995:
            return (
                None,
                f"⚡ 21-EMA Low Floor Defended (₹{ema_21_low:.1f} held despite 21-EMA Close undercut)"
            )

        days = max(days_below_21_low, triggers.get('days_below_21ema', 2))
        return (ACTION_TRIM, f"Ross Haber Sell Rule: {days} consecutive closes below 21-EMA Low (Trim 1/3 to 1/2)")
        
    if trend_state == TREND_WARNING:
        if triggers['high_vol_down_day']:
            return (ACTION_TRIM, "Warning: High-volume down day >5%")
    
    if trend_state == TREND_BROKEN:
        if len(df) >= 50 and 'sma_50' in df.columns and not pd.isna(df['sma_50'].iloc[-1]):
            return (ACTION_EXIT, "Trend broken: 3 consecutive closes below 50 SMA")
        return (None, None)
    
    return (None, None)


def check_layer3_structural(df: pd.DataFrame, rs_score: float, days_below_50sma: int = 0) -> tuple:
    """
    Layer 3: Structural failure exit triggers.
    
    Args:
        df: DataFrame with indicators
        rs_score: Current RS score
        days_below_50sma: Days below 50 SMA
    
    Returns:
        Tuple of (action, reason) or (None, None) if no trigger
    """
    failures = check_structural_failure(df, rs_score, days_below_50sma)
    
    if any([
        failures['down_from_52w_high'],
        failures['failed_50sma_reclaim'],
        failures['weekly_distribution'],
        failures['rs_leadership_lost'],
    ]):
        reason = failures['triggered_reason'] or "Structural failure detected"
        return (ACTION_EXIT, f"Structural failure: {reason}")
    
    return (None, None)


def check_averaging_up(df: pd.DataFrame, rs_score: float, trend_state: str) -> tuple:
    """
    Opportunity Layer: Check if eligible for averaging up (Pyramiding).
    
    Professional Add Setups:
    1. 3-Weeks Tight (Ultimate O'Neil setup)
    2. 21-EMA Squat/Bounce (Tight, low volume rest near 21-EMA)
    3. 50-SMA Pullback (Quiet test of 10-week line)
    
    Args:
        df: DataFrame with indicators
        rs_score: Current RS score
        trend_state: Current trend state
    
    Returns:
        Tuple of (action, add_on_ready, reason)
    """
    if df.empty or len(df) < 50:
        return (ACTION_HOLD, False, "")
    
    last_row = df.iloc[-1]
    close = last_row.get('close', 0)
    ema_21 = last_row.get('ema_21', 0)
    sma_50 = last_row.get('sma_50', 0)
    
    # Base requirements for any add
    if rs_score is None or rs_score < AVERAGING_UP['MIN_RS_SCORE']:
        rs_str = f"{rs_score:.1f}" if rs_score is not None else "N/A"
        return (ACTION_HOLD, False, f"RS {rs_str} < {AVERAGING_UP['MIN_RS_SCORE']}")
        
    dist_from_high = last_row.get('dist_from_52w_high', 100)
    if dist_from_high > AVERAGING_UP['MAX_DISTANCE_FROM_52W']:
        return (ACTION_HOLD, False, f"{dist_from_high:.1f}% from 52W high")
        
    # Setup 3: 3-Weeks Tight
    if detect_3_weeks_tight(df):
        return (ACTION_ADD, True, "Add: 3-Weeks Tight breakout setup")
        
    # For MA pullbacks, we require tight ATR and no recent distribution
    atr_ratio = last_row.get('atr_ratio', 1.0)
    rel_vol = last_row.get('rel_vol', 1.0)
    is_down_day = last_row.get('daily_pct_change', 0) < 0
    
    if is_down_day and rel_vol > AVERAGING_UP['MAX_PULLBACK_VOL']:
        return (ACTION_HOLD, False, "Pullback volume too high")
        
    if atr_ratio > AVERAGING_UP['ATR_CONTRACTION_RATIO']:
        return (ACTION_HOLD, False, f"ATR ratio {atr_ratio:.2f} not contracted")
        
    if has_big_down_day(df, AVERAGING_UP['MAX_DOWN_DAY_PCT'], lookback=10):
        return (ACTION_HOLD, False, f"Down day >{AVERAGING_UP['MAX_DOWN_DAY_PCT']}% in last 10 days")
        
    # Setup 1: 21-EMA / 21-EMA Low Squat
    ema_21_low = last_row.get('ema_21_low', ema_21)
    if ema_21 > 0 and close >= ema_21:
        ext_21 = ((close - ema_21) / ema_21) * 100
        if ext_21 <= AVERAGING_UP['MAX_EXTENSION_FROM_21EMA'] and trend_state == TREND_STRONG:
            return (ACTION_ADD, True, "Add: 21-EMA Squat on light volume")
    elif ema_21_low and ema_21_low > 0 and close >= ema_21_low:
        ext_21l = ((close - ema_21_low) / ema_21_low) * 100
        if ext_21l <= AVERAGING_UP['MAX_EXTENSION_FROM_21EMA']:
            return (ACTION_ADD, True, "Add: 21-EMA Low Channel Bounce on light volume")
            
    # Setup 2: 50-SMA Pullback
    if sma_50 > 0 and close >= sma_50:
        ext_50 = ((close - sma_50) / sma_50) * 100
        if ext_50 <= AVERAGING_UP['MAX_EXTENSION_FROM_50SMA']:
            # For 50 SMA bounce, we tolerate TREND_WARNING since price is below 21-EMA
            return (ACTION_ADD, True, "Add: 50-SMA Pullback / Support Test")
            
    return (ACTION_HOLD, False, "No valid add setup (Extended from MAs)")


def make_decision(
    df: pd.DataFrame,
    ticker: str,
    current_price: float,
    avg_buy_price: float,
    quantity: float,
    total_portfolio_value: float,
    rs_score: float,
) -> dict:
    """
    Make trading decision using layered logic.
    
    Priority:
    1. Layer 1: Risk management (portfolio loss)
    2. Layer 2: Trend health
    3. Layer 3: Structural failure
    4. Opportunity: Averaging up
    5. Default: HOLD
    
    Args:
        df: DataFrame with indicators
        ticker: Stock ticker
        current_price: Current price
        avg_buy_price: Average buy price
        quantity: Position quantity
        total_portfolio_value: Total portfolio value
        rs_score: Relative strength score
    
    Returns:
        Decision dictionary with action, reason, and details
    """
    # Get trend state
    trend_state = determine_trend_state(df)
    
    # Calculate loss metrics
    loss_metrics = calculate_position_loss(
        current_price, avg_buy_price, quantity, total_portfolio_value
    )
    
    # Get ATR state
    atr_ratio = df.iloc[-1].get('atr_ratio') if not df.empty else None
    atr_state = get_atr_state(atr_ratio)

    # Classify Moving Average Character DNA & Tom Hougaard House-Money Cushion
    ma_dna = classify_stock_ma_dna(df)
    ema_21_low_val = float(ma_dna.get('ema_21_low', 0.0) or 0.0)
    active_spine_val = float(ma_dna.get('active_spine_price', 0.0) or ema_21_low_val or 0.0)
    locked_profit_ps = max(0.0, active_spine_val - avg_buy_price) if (active_spine_val > 0 and avg_buy_price > 0 and current_price > active_spine_val) else 0.0
    locked_profit_inr = round(locked_profit_ps * quantity, 2)
    if active_spine_val > 0 and active_spine_val < current_price:
        stop_for_add = active_spine_val
    elif ema_21_low_val > 0 and ema_21_low_val < current_price:
        stop_for_add = ema_21_low_val
    else:
        stop_for_add = round(current_price * 0.94, 2) if current_price > 0 else 0.0
    new_share_risk = max(current_price * 0.03, current_price - stop_for_add) if current_price > 0 else 1.0
    if locked_profit_inr > 0 and new_share_risk > 0 and current_price > stop_for_add:
        # 1. Financed by at most 35% of locked profit cushion (preserving 65%+ profit floor)
        sh_by_cushion = int((locked_profit_inr * 0.35) // new_share_risk)
        # 2. O'Neil Pyramid Taper Rule: add at most 50% of existing share count
        sh_by_taper = max(1, int(quantity * 0.50))
        # 3. O'Neil 20% Max Portfolio Weight Cap
        rem_nav_cap = max(0.0, (total_portfolio_value * 0.20) - (quantity * current_price)) if total_portfolio_value > 0 else (sh_by_taper * current_price)
        sh_by_nav = int(rem_nav_cap // current_price) if current_price > 0 else sh_by_taper
        free_pyramid_shares = max(0, min(sh_by_cushion, sh_by_taper, sh_by_nav if sh_by_nav > 0 else sh_by_taper))
    else:
        free_pyramid_shares = 0

    def _enrich(res_dict: dict) -> dict:
        res_dict['ma_dna'] = ma_dna
        res_dict['ema_21_low'] = ema_21_low_val
        res_dict['locked_profit_inr'] = locked_profit_inr
        res_dict['free_pyramid_shares'] = free_pyramid_shares
        res_dict['new_share_stop'] = round(stop_for_add, 2)
        return res_dict
    
    # Layer 1: Risk management
    action, reason = check_layer1_risk(loss_metrics)
    if action:
        return _enrich({
            'ticker': ticker,
            'action': action,
            'reason': reason,
            'trend_state': trend_state,
            'rs_score': rs_score,
            'atr_state': atr_state,
            'add_on_ready': False,
            'loss_metrics': loss_metrics,
            'layer': 'L1-Risk',
        })
        
    # Layer 1.5: Volatility Risk (ATR Chandelier Stop)
    action, reason = check_layer1_5_volatility_stop(df)
    if action:
        return _enrich({
            'ticker': ticker,
            'action': action,
            'reason': reason,
            'trend_state': trend_state,
            'rs_score': rs_score,
            'atr_state': atr_state,
            'add_on_ready': False,
            'loss_metrics': loss_metrics,
            'layer': 'L1.5-VolRisk',
        })
    
    # Cash equivalents should be ignored from trend and structural exits
    is_cash_etf = ticker.upper() in {'LIQUIDBEES', 'LIQUIDCASE', 'LIQUIDETF'}

    # Layer 2: Trend health (with 21-EMA Low & Boik 50-SMA Monster Immunity)
    action, l2_reason = check_layer2_trend(df, trend_state, gain_loss_pct=loss_metrics.get('gain_loss_pct', 0.0), ma_dna=ma_dna)
    if action and not is_cash_etf:
        return _enrich({
            'ticker': ticker,
            'action': action,
            'reason': l2_reason,
            'trend_state': trend_state,
            'rs_score': rs_score,
            'atr_state': atr_state,
            'add_on_ready': False,
            'loss_metrics': loss_metrics,
            'layer': 'L2-Trend',
        })
    
    # Layer 3: Structural failure
    days_below_50sma = count_days_below_ema(df, 'sma_50', lookback=30) if not df.empty else 0
    action, reason = check_layer3_structural(df, rs_score, days_below_50sma)
    if action and not is_cash_etf:
        return _enrich({
            'ticker': ticker,
            'action': action,
            'reason': reason,
            'trend_state': trend_state,
            'rs_score': rs_score,
            'atr_state': atr_state,
            'add_on_ready': False,
            'loss_metrics': loss_metrics,
            'layer': 'L3-Structural',
        })
        
    # Layer 4: Opportunity Cost (Dead Money)
    action, reason = check_layer1_5_dead_money(df, loss_metrics)
    if action and not is_cash_etf:
        return _enrich({
            'ticker': ticker,
            'action': action,
            'reason': reason,
            'trend_state': trend_state,
            'rs_score': rs_score,
            'atr_state': atr_state,
            'add_on_ready': False,
            'loss_metrics': loss_metrics,
            'layer': 'L4-DeadMoney',
        })
    
    # Opportunity: Averaging up
    action, add_on_ready, reason = check_averaging_up(df, rs_score, trend_state)
    if action == ACTION_ADD:
        return _enrich({
            'ticker': ticker,
            'action': action,
            'reason': reason,
            'trend_state': trend_state,
            'rs_score': rs_score,
            'atr_state': atr_state,
            'add_on_ready': True,
            'loss_metrics': loss_metrics,
            'layer': 'Opportunity',
        })
    
    # Default: HOLD (surface Monster 50-SMA or 21-EMA Low defense reason if active)
    hold_reason = l2_reason if l2_reason else ("Position healthy - no action needed" if trend_state == TREND_STRONG else f"Monitoring: {reason or trend_state}")
    return _enrich({
        'ticker': ticker,
        'action': ACTION_HOLD,
        'reason': hold_reason,
        'trend_state': trend_state,
        'rs_score': rs_score,
        'atr_state': atr_state,
        'add_on_ready': add_on_ready,
        'loss_metrics': loss_metrics,
        'layer': 'Default',
    })


if __name__ == '__main__':
    print("Decision Engine loaded successfully")
    print(f"Actions: {ACTION_EXIT}, {ACTION_TRIM}, {ACTION_HOLD}, {ACTION_ADD}")
