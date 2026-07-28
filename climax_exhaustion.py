# Climax / Exhaustion Detector (CED)
# Identifies stocks entering parabolic, late-stage acceleration phase

import pandas as pd
import numpy as np
from technical_indicators import calculate_slope, calculate_atr

def detect_climax_exhaustion(df: pd.DataFrame, ticker: str = "Unknown") -> dict:
    """
    Analyze a stock for Climax / Exhaustion signals.
    
    Args:
        df: DataFrame with OHLCV data and indicators.
        ticker: Ticker symbol for reporting.
        
    Returns:
        Dictionary containing scores, logic checks, and status.
    """
    if df.empty or len(df) < 50:
        return {
            'ticker': ticker,
            'climax_score': 0,
            'status': 'Insufficient Data',
            'status_color': 'gray',
            'return_15d': 0.0,
            'pct_above_10ema': 0.0,
            'vol_spike_ratio': 0.0,
            'atr_ratio': 0.0,
            'slope_10': 0.0,
            'slope_30': 0.0,
            'slope_ratio': 0.0,
            'climax_risk': False,
            'high_climax_prob': False,
            'details': {}
        }
        
    current_close = df['close'].iloc[-1]
    
    # --- Condition 1: 3-Week Acceleration ---
    # Return_15 = (Close_today / Close_15_days_ago) - 1
    # If Return_15 >= 25% -> Score +1
    close_15_ago = df['close'].iloc[-16] if len(df) >= 16 else df['close'].iloc[0]
    return_15 = (current_close / close_15_ago) - 1
    cond1 = return_15 >= 0.25
    
    # --- Condition 2: Extreme Distance from 10 EMA ---
    # (Current Close - 10 EMA) / 10 EMA
    # If >= 15% -> Score +1
    # If >= 20% -> Strong Extension Flag
    ema_10 = df['ema_10'].iloc[-1]
    dist_10ema = (current_close - ema_10) / ema_10
    cond2 = dist_10ema >= 0.15
    strong_ext = dist_10ema >= 0.20
    
    # --- Condition 3: Volume Expansion During Acceleration ---
    # Avg Volume last 10 days vs Avg Volume prior 50 days
    # Using simple moving averages for this comparison
    vol_10d = df['volume'].tail(10).mean()
    vol_50d = df['volume'].tail(50).mean()
    cond3 = vol_10d >= 1.75 * vol_50d
    
    # --- Condition 4: Range Expansion ---
    # ATR 10 >= 1.5 * ATR 30
    # Calculate ATR 10 specifically for this check if not present
    atr_10_series = calculate_atr(df, 10)
    atr_10 = atr_10_series.iloc[-1]
    # Re-calculate ATR 30 to be sure (though likely in df)
    atr_30_series = calculate_atr(df, 30)
    atr_30 = atr_30_series.iloc[-1]
    
    # Handle division by zero or NaN
    atr_ratio = atr_10 / atr_30 if atr_30 > 0 else 0
    cond4 = atr_10 >= 1.5 * atr_30
    
    # --- Condition 5: Slope Acceleration ---
    # Slope_10 >= 1.8 * Slope_30
    slope_10 = calculate_slope(df['close'], 10)
    slope_30 = calculate_slope(df['close'], 30)
    
    # Handle negative slopes or zero slopes (acceleration implies positive slopes usually)
    # If slope_30 is negative and slope_10 is positive, that's huge acceleration (turnaround)
    # But usually this metric compares two positive slopes.
    # If slope_30 is very small, ratio blows up. 
    # Let's stick to strict math: Slope_10 >= 1.8 * Slope_30
    cond5 = False
    if slope_30 > 0:
        cond5 = slope_10 >= 1.8 * slope_30
    elif slope_30 <= 0 and slope_10 > 0:
        # If long term slope is flat/down and short term is up, it IS accelerating, 
        # but is it "parabolic late stage"? Maybe just breakout.
        # Strict interpretation of prompt:
        cond5 = slope_10 >= 1.8 * slope_30
        
    
    # --- Scoring ---
    score = sum([cond1, cond2, cond3, cond4, cond5])
    
    # --- Classification ---
    status = "Normal"
    color = "green" # Default/Safe
    
    if score >= 4:
        status = "🔥 High Climax Probability"
        color = "red"
    elif score == 3:
        status = "⚠ Climax Risk"
        color = "orange"
    elif score == 2:
        status = "Elevated Momentum"
        color = "yellow"
        
    return {
        'ticker': ticker,
        'climax_score': score,
        'status': status,
        'status_color': color,
        'return_15d': return_15,
        'pct_above_10ema': dist_10ema,
        'vol_spike_ratio': vol_10d / vol_50d if vol_50d > 0 else 0,
        'atr_ratio': atr_ratio,
        'slope_10': slope_10,
        'slope_30': slope_30,
        'slope_ratio': slope_10 / slope_30 if slope_30 != 0 else 0,
        'cond_1_accel': cond1,
        'cond_2_extension': cond2,
        'cond_3_vol': cond3,
        'cond_4_range': cond4,
        'cond_5_slope': cond5,
        'strong_extension': strong_ext
    }
