# Trend Analyzer for Kush Tracker

import pandas as pd
from config import TREND_RULES
from technical_indicators import count_days_below_ema, has_high_volume_down_day


# Trend state constants
TREND_STRONG = '🟢 Strong'
TREND_PULLBACK = '🟡 Pullback'
TREND_WARNING = '🟠 Warning'
TREND_BROKEN = '🔴 Broken'


def determine_trend_state(df: pd.DataFrame) -> str:
    """
    Determine the current trend state based on EMA positions.
    
    Rules:
    - Strong: Close > 8 EMA AND > 21 EMA
    - Pullback: Close < 8 EMA, > 21 EMA
    - Warning: Close < 21 EMA, > 50 SMA
    - Broken: Close < 50 SMA
    
    Args:
        df: DataFrame with close and EMA/SMA columns
    
    Returns:
        Trend state string
    """
    if df.empty:
        return TREND_BROKEN
    
    last_row = df.iloc[-1]
    
    close = last_row.get('close')
    ema_8 = last_row.get('ema_8')
    ema_21 = last_row.get('ema_21')
    sma_50 = last_row.get('sma_50')
    
    # Handle missing values
    if any(pd.isna([close, ema_8, ema_21, sma_50])):
        return TREND_BROKEN
    
    # Determine state
    if close > ema_8 and close > ema_21:
        return TREND_STRONG
    elif close < ema_8 and close > ema_21:
        return TREND_PULLBACK
    elif close < ema_21 and close >= sma_50:
        return TREND_WARNING
    else:
        # We are below 50 SMA. Check if this is day 3+ (3 consecutive closes below 50 SMA)
        days_below = count_days_below_ema(df, 'sma_50', lookback=5)
        if days_below >= 3:
            return TREND_BROKEN
        
        # Day 1-2 below 50 SMA -> Warning (let it potentially undercut and rally)
        return TREND_WARNING


def check_warning_triggers(df: pd.DataFrame) -> dict:
    """
    Check for warning-level trim triggers.
    
    Triggers:
    1. Close < 21 EMA for 5+ consecutive days
    2. Single down day > 5% on above-average volume
    
    Args:
        df: DataFrame with indicators
    
    Returns:
        Dictionary with trigger information
    """
    triggers = {
        'consecutive_below_21ema': False,
        'high_vol_down_day': False,
        'days_below_21ema': 0,
    }
    
    if df.empty:
        return triggers
    
    # Check consecutive days below 21 EMA
    days_below = count_days_below_ema(df, 'ema_21', lookback=10)
    triggers['days_below_21ema'] = days_below
    
    if days_below >= TREND_RULES['WARNING_CONSECUTIVE_DAYS']:
        triggers['consecutive_below_21ema'] = True
    
    # Check for high-volume down day
    if has_high_volume_down_day(df, TREND_RULES['WARNING_DOWN_DAY_PCT']):
        triggers['high_vol_down_day'] = True
    
    return triggers


def check_structural_failure(df: pd.DataFrame, rs_score: float, days_below_50sma: int = 0) -> dict:
    """
    Check for structural failure exit triggers.
    
    Triggers:
    1. Stock 20% down from 52-week high
    2. Below 50 SMA for 10-15 days without reclaim
    3. Weekly close down 8% on 1.5x volume (simplified to daily check)
    4. RS score < 65
    
    Args:
        df: DataFrame with indicators
        rs_score: Current RS score
        days_below_50sma: Days stock has been below 50 SMA
    
    Returns:
        Dictionary with failure triggers
    """
    from config import STRUCTURAL_FAILURE, RS_THRESHOLDS
    
    failures = {
        'down_from_52w_high': False,
        'failed_50sma_reclaim': False,
        'weekly_distribution': False,
        'rs_leadership_lost': False,
        'triggered_reason': None,
    }
    
    if df.empty:
        return failures
    
    last_row = df.iloc[-1]
    
    # Check distance from 52-week high
    dist_from_high = last_row.get('dist_from_52w_high', 0)
    if dist_from_high >= STRUCTURAL_FAILURE['DOWN_FROM_52W_HIGH_PCT']:
        failures['down_from_52w_high'] = True
        failures['triggered_reason'] = f"Down {dist_from_high:.1f}% from 52W high"
    
    # Check 50 SMA reclaim failure
    if days_below_50sma >= STRUCTURAL_FAILURE['SMA_RECLAIM_DAYS']:
        failures['failed_50sma_reclaim'] = True
        failures['triggered_reason'] = f"Below 50 SMA for {days_below_50sma} days"
    
    # Check Weekly Distribution (Down 8% on 1.5x volume)
    # We need to resample to weekly candles
    if 'date' in df.columns and len(df) > 100:
        try:
            # Resample to weekly (ending Friday)
            df_weekly = df.set_index('date').resample('W-FRI').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # Calculate weekly returns and volume MA
            df_weekly['pct_change'] = df_weekly['close'].pct_change() * 100
            df_weekly['vol_ma_10'] = df_weekly['volume'].rolling(window=10).mean() # 10 weeks ~ 2.5 months
            
            # Check last completed week (iloc[-1] might be current partial week)
            # We check the last CLOSED candle. If today is Friday after market close, -1.
            # Safe bet: check last 2 weeks to capture recent distribution
            recent_weeks = df_weekly.tail(2)
            
            for idx, week in recent_weeks.iterrows():
                # Check for drop > 8%
                if week['pct_change'] <= -STRUCTURAL_FAILURE['WEEKLY_DISTRIBUTION_DROP_PCT']:
                    # Check for volume > 1.5x average
                    if week['volume'] > (week['vol_ma_10'] * STRUCTURAL_FAILURE['WEEKLY_VOL_MULTIPLIER']):
                        failures['weekly_distribution'] = True
                        failures['triggered_reason'] = f"Weekly distribution: Down {abs(week['pct_change']):.1f}% on high vol"
                        break
        except Exception:
            # Fallback if resampling fails
            pass

    # Check RS leadership
    if rs_score is not None and rs_score < RS_THRESHOLDS['WEAKENING']:
        failures['rs_leadership_lost'] = True
        failures['triggered_reason'] = f"RS score {rs_score:.1f} below threshold"
    
    return failures


def get_trend_details(df: pd.DataFrame) -> dict:
    """
    Get detailed trend information for dashboard display.
    
    Args:
        df: DataFrame with indicators
    
    Returns:
        Dictionary with trend details
    """
    if df.empty:
        return {
            'above_8': False,
            'above_21': False,
            'above_50': False,
            'distance_from_52w_high': None,
        }
    
    last_row = df.iloc[-1]
    close_price = last_row.get('close', 0)
    ema_21 = last_row.get('ema_21', None)
    sma_50 = last_row.get('sma_50', None)
    sma_200 = last_row.get('sma_200', None)
    
    dist_21 = None
    if ema_21 is not None and ema_21 > 0:
        dist_21 = abs(close_price - ema_21) / ema_21 * 100
        
    ext_50sma = None
    if sma_50 is not None and sma_50 > 0:
        ext_50sma = ((close_price / sma_50) - 1) * 100
        
    ext_200sma = None
    if sma_200 is not None and sma_200 > 0:
        ext_200sma = ((close_price / sma_200) - 1) * 100
        
    return {
        'above_8': bool(close_price > last_row.get('ema_8', float('inf'))),
        'above_21': bool(close_price > ema_21) if ema_21 is not None else False,
        'above_50': bool(close_price > last_row.get('sma_50', float('inf'))),
        'distance_from_52w_high': last_row.get('dist_from_52w_high'),
        'dist_from_21ema': dist_21,
        'ext_50sma': ext_50sma,
        'ext_200sma': ext_200sma,
    }


if __name__ == '__main__':
    print("Testing trend analyzer...")
    print(f"Trend states: {TREND_STRONG}, {TREND_PULLBACK}, {TREND_WARNING}, {TREND_BROKEN}")
