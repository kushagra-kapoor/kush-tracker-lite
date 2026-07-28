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
    count_days_above_ema, count_days_below_ema, has_big_down_day, get_atr_state, detect_3_weeks_tight, calculate_atr_chandelier_stop
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
    - Price < 2.5x ATR below 20-day high → EXIT
    """
    if df.empty:
        return (None, None)
        
    current_price = df.iloc[-1]['close']
    stop_price = calculate_atr_chandelier_stop(df, atr_multiplier=2.5, high_window=20)
    
    if stop_price > 0 and current_price < stop_price:
        return (ACTION_EXIT, f"Volatility Stop: Price dropped below {stop_price:.2f} (2.5x ATR)")
        
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


def check_layer2_trend(df: pd.DataFrame, trend_state: str) -> tuple:
    """
    Layer 2: Trend health based actions.
    
    Rules:
    - Warning + 5 days below 21 EMA → TRIM 25-50%
    - Warning + high-vol down day → TRIM 25-50%
    - Broken trend → Reduce to tracking size
    
    Args:
        df: DataFrame with indicators
        trend_state: Current trend state
    
    Returns:
        Tuple of (action, reason) or (None, None) if no trigger
    """
    if trend_state == TREND_WARNING:
        triggers = check_warning_triggers(df)
        
        if triggers['consecutive_below_21ema']:
            days = triggers['days_below_21ema']
            return (ACTION_TRIM, f"Warning: {days} consecutive days below 21 EMA")
        
        if triggers['high_vol_down_day']:
            return (ACTION_TRIM, "Warning: High-volume down day >5%")
    
    if trend_state == TREND_BROKEN:
        return (ACTION_EXIT, "Trend broken: 3 consecutive closes below 50 SMA")
    
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
        
    # Setup 1: 21-EMA Squat
    if ema_21 > 0 and close >= ema_21:
        ext_21 = ((close - ema_21) / ema_21) * 100
        if ext_21 <= AVERAGING_UP['MAX_EXTENSION_FROM_21EMA'] and trend_state == TREND_STRONG:
            return (ACTION_ADD, True, "Add: 21-EMA Squat on light volume")
            
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
    
    # Layer 1: Risk management
    action, reason = check_layer1_risk(loss_metrics)
    if action:
        return {
            'ticker': ticker,
            'action': action,
            'reason': reason,
            'trend_state': trend_state,
            'rs_score': rs_score,
            'atr_state': atr_state,
            'add_on_ready': False,
            'loss_metrics': loss_metrics,
            'layer': 'L1-Risk',
        }
        
    # Layer 1.5: Volatility Risk (ATR Chandelier Stop)
    action, reason = check_layer1_5_volatility_stop(df)
    if action:
        return {
            'ticker': ticker,
            'action': action,
            'reason': reason,
            'trend_state': trend_state,
            'rs_score': rs_score,
            'atr_state': atr_state,
            'add_on_ready': False,
            'loss_metrics': loss_metrics,
            'layer': 'L1.5-VolRisk',
        }
    
    # Cash equivalents should be ignored from trend and structural exits
    is_cash_etf = ticker.upper() in {'LIQUIDBEES', 'LIQUIDCASE', 'LIQUIDETF'}

    # Layer 2: Trend health
    action, reason = check_layer2_trend(df, trend_state)
    if action and not is_cash_etf:
        return {
            'ticker': ticker,
            'action': action,
            'reason': reason,
            'trend_state': trend_state,
            'rs_score': rs_score,
            'atr_state': atr_state,
            'add_on_ready': False,
            'loss_metrics': loss_metrics,
            'layer': 'L2-Trend',
        }
    
    # Layer 3: Structural failure
    days_below_50sma = count_days_below_ema(df, 'sma_50', lookback=30) if not df.empty else 0
    action, reason = check_layer3_structural(df, rs_score, days_below_50sma)
    if action and not is_cash_etf:
        return {
            'ticker': ticker,
            'action': action,
            'reason': reason,
            'trend_state': trend_state,
            'rs_score': rs_score,
            'atr_state': atr_state,
            'add_on_ready': False,
            'loss_metrics': loss_metrics,
            'layer': 'L3-Structural',
        }
        
    # Layer 4: Opportunity Cost (Dead Money)
    action, reason = check_layer1_5_dead_money(df, loss_metrics)
    if action and not is_cash_etf:
        return {
            'ticker': ticker,
            'action': action,
            'reason': reason,
            'trend_state': trend_state,
            'rs_score': rs_score,
            'atr_state': atr_state,
            'add_on_ready': False,
            'loss_metrics': loss_metrics,
            'layer': 'L4-DeadMoney',
        }
    
    # Opportunity: Averaging up
    action, add_on_ready, reason = check_averaging_up(df, rs_score, trend_state)
    if action == ACTION_ADD:
        return {
            'ticker': ticker,
            'action': action,
            'reason': reason,
            'trend_state': trend_state,
            'rs_score': rs_score,
            'atr_state': atr_state,
            'add_on_ready': True,
            'loss_metrics': loss_metrics,
            'layer': 'Opportunity',
        }
    
    # Default: HOLD
    return {
        'ticker': ticker,
        'action': ACTION_HOLD,
        'reason': "Position healthy - no action needed" if trend_state == TREND_STRONG else f"Monitoring: {reason or trend_state}",
        'trend_state': trend_state,
        'rs_score': rs_score,
        'atr_state': atr_state,
        'add_on_ready': add_on_ready,
        'loss_metrics': loss_metrics,
        'layer': 'Default',
    }


if __name__ == '__main__':
    print("Decision Engine loaded successfully")
    print(f"Actions: {ACTION_EXIT}, {ACTION_TRIM}, {ACTION_HOLD}, {ACTION_ADD}")
