import numpy as np
import pandas as pd

def calculate_adjusted_slope(df: pd.DataFrame, window: int = 90) -> dict:
    """
    Calculates the Clenow Adjusted Momentum Score (Annualized Slope * R^2)
    
    Args:
        df: DataFrame containing 'close' prices
        window: The lookback period (default 90 days)
        
    Returns:
        dict: {'score': float, 'r_squared': float, 'annualized_slope': float}
    """
    if df.empty or len(df) < window:
        return {'score': 0.0, 'r_squared': 0.0, 'annualized_slope': 0.0}
        
    # Standardize column naming if needed
    col = 'close'
    if 'Close' in df.columns:
        col = 'Close'
        
    recent = df.tail(window).copy()
    
    # Drop NaNs
    recent = recent.dropna(subset=[col])
    if len(recent) < window // 2:
        return {'score': 0.0, 'r_squared': 0.0, 'annualized_slope': 0.0}
        
    y = np.log(recent[col].values)
    x = np.arange(len(y))
    
    # Calculate linear regression slope and intercept
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    
    r_squared = 0.0
    if ss_tot > 0:
        r_squared = 1 - (ss_res / ss_tot)
        
    # Annualize slope (250 trading days)
    # Annualized return = (exp(slope)^250) - 1
    # Clenow multiplies this annual return by R^2
    annualized_slope = (np.exp(slope) ** 250) - 1
    
    # Adjusted Momentum Score
    score = annualized_slope * r_squared
    
    return {
        'score': score * 100,  # Convert to percentage
        'r_squared': r_squared,
        'annualized_slope': annualized_slope * 100 # Convert to percentage
    }

def calculate_atr(df: pd.DataFrame, period: int = 20) -> float:
    """
    Calculates the 20-day Average True Range (ATR).
    """
    if df.empty or len(df) < period + 1:
        return 0.0
        
    high = 'high' if 'high' in df.columns else 'High'
    low = 'low' if 'low' in df.columns else 'Low'
    close = 'close' if 'close' in df.columns else 'Close'
    
    df_calc = df.copy()
    df_calc['prev_close'] = df_calc[close].shift(1)
    
    tr1 = df_calc[high] - df_calc[low]
    tr2 = np.abs(df_calc[high] - df_calc['prev_close'])
    tr3 = np.abs(df_calc[low] - df_calc['prev_close'])
    
    df_calc['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = df_calc['tr'].rolling(window=period).mean().iloc[-1]
    return atr

def check_disqualification_filters(df: pd.DataFrame, sma_period: int = 100, gap_window: int = 90, max_gap: float = 0.15) -> dict:
    """
    Applies Clenow's disqualification filters:
    1. Must track above 100-day SMA.
    2. No daily gaps downwards > 15% in the last 90 days.
    
    Returns True if PASS (valid to trade), False if FAIL (disqualified).
    """
    if df.empty or len(df) < sma_period:
        return {'pass': False, 'reason': 'Not enough data'}
        
    close = 'close' if 'close' in df.columns else 'Close'
    open_p = 'open' if 'open' in df.columns else 'Open' # Using open and low to detect full gaps
    
    recent = df.copy()
    
    # Filter 1: 100-day SMA
    sma = recent[close].rolling(window=sma_period).mean()
    current_price = recent[close].iloc[-1]
    current_sma = sma.iloc[-1]
    
    if current_price < current_sma:
        return {'pass': False, 'reason': 'Below 100-day SMA'}
        
    # Filter 2: Max Gap Down
    # Gap down means Open < Prev Close by 15% OR Low < Prev Close by 15% (depending on interpretation). 
    # Usually Clenow refers to a single day move/gap > 15%.
    recent_90 = recent.tail(gap_window).copy()
    recent_90['prev_close'] = recent_90[close].shift(1)
    
    if not recent_90.empty and len(recent_90) > 1:
        # Check percentage drop from prev close to current day's open or close
        # Using close-to-close or open-to-close. A true gap is open << prev_close.
        # But a 15% drop in general is also bad. Let's use standard return.
        returns = (recent_90[close] - recent_90['prev_close']) / recent_90['prev_close']
        
        # Did it drop by more than 15%?
        if (returns < -max_gap).any():
            return {'pass': False, 'reason': f'> {max_gap*100}% Drop in last 90d'}
            
    return {'pass': True, 'reason': 'Valid'}

def calculate_risk_parity_shares(portfolio_value: float, risk_bps: float, atr: float) -> float:
    """
    Clenow Risk Parity position sizing.
    Shares = (Portfolio_Value * Risk_Factor) / ATR
    """
    if atr <= 0 or portfolio_value <= 0:
        return 0.0
        
    risk_factor = risk_bps / 10000.0  # e.g. 10 bps = 0.001
    dollar_risk = portfolio_value * risk_factor
    
    shares = dollar_risk / atr
    return np.floor(shares)
