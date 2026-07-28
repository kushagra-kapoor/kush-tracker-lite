# Technical Indicators for Kush Tracker

import pandas as pd
import numpy as np
from config import EMA_PERIODS, ATR_PERIODS, DATA_SETTINGS


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        series: Price series
        period: EMA period
    
    Returns:
        EMA series
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        series: Price series
        period: SMA period
    
    Returns:
        SMA series
    """
    return series.rolling(window=period).mean()


def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period
    
    Returns:
        ATR series
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_slope(series: pd.Series, period: int) -> float:
    # Retained for backwards compatibility with single-row calls in the app
    if len(series) < period:
        return np.nan
    y = series.tail(period).values
    x = np.arange(period)
    slope, _ = np.polyfit(x, y, 1)
    return slope

def fast_rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Sub-millisecond vectorized linear regression slope over a rolling window.
    Leverages numpy stride tricks for lightning fast performance over millions of rows.
    """
    arr = series.values
    if len(arr) < window:
        return pd.Series(np.nan, index=series.index)
        
    windows = np.lib.stride_tricks.sliding_window_view(arr, window)
    x = np.arange(window)
    x_mean = x.mean()
    x_diff = x - x_mean
    x_var = np.sum(x_diff**2)
    
    slopes = windows.dot(x_diff) / x_var
    
    pad = np.full(window - 1, np.nan)
    res = np.concatenate((pad, slopes))
    return pd.Series(res, index=series.index)


def calculate_returns(series: pd.Series, periods: int) -> float:
    """
    Calculate return over specified periods.
    
    Args:
        series: Price series (most recent at end)
        periods: Number of periods to look back
    
    Returns:
        Return as decimal (e.g., 0.10 for 10%)
    """
    if len(series) < periods + 1:
        return None
    
    current_price = series.iloc[-1]
    past_price = series.iloc[-(periods + 1)]
    
    if past_price == 0:
        return None
    
    return (current_price / past_price) - 1


def add_technical_indicators(df: pd.DataFrame, benchmark_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add all technical indicators to a stock DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        benchmark_df: Optional benchmark DataFrame to calculate RS Blue Dot
    
    Returns:
        DataFrame with added indicator columns
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # EMAs
    df['ema_8'] = calculate_ema(df['close'], EMA_PERIODS['FAST'])
    df['ema_10'] = calculate_ema(df['close'], 10)  # Added for Climax Detector
    df['ema_21'] = calculate_ema(df['close'], EMA_PERIODS['MEDIUM'])
    df['sma_50'] = calculate_sma(df['close'], EMA_PERIODS['SLOW'])
    df['sma_200'] = calculate_sma(df['close'], 200)
    
    # ATRs
    df['atr_14'] = calculate_atr(df, ATR_PERIODS['SHORT'])
    df['atr_30'] = calculate_atr(df, ATR_PERIODS['LONG'])
    
    # ATR ratio for volatility contraction detection
    df['atr_ratio'] = df['atr_14'] / df['atr_30']
    
    # 52-week high
    df['high_52w'] = df['high'].rolling(window=252, min_periods=1).max()
    
    # Distance from 52-week high (as percentage)
    df['dist_from_52w_high'] = ((df['high_52w'] - df['close']) / df['high_52w']) * 100
    
    # Daily percentage change
    df['daily_pct_change'] = df['close'].pct_change() * 100
    
    # Volume moving average (20-day)
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    
    # Relative Volume (Current Volume / 20-day MA)
    # Avoid division by zero
    df['rel_vol'] = df['volume'] / df['volume_ma_20'].replace(0, 1)
    
    # ADR% (Average Daily Range as percentage of price)
    # Formula: ((High - Low) / Close) * 100
    # We use 20-day rolling average of this daily range %
    df['daily_range_pct'] = ((df['high'] - df['low']) / df['close']) * 100
    df['adr_pct_20'] = df['daily_range_pct'].rolling(window=20).mean()
    
    # Phase 28: Institutional Footprint Identifiers (Pocket Pivot & VDU)
    # 1. Volume Dry-Up (VDU): Daily volume drops > 50% below 50-day average
    vol_50d_ma = df['volume'].rolling(window=50).mean()
    df['is_vdu'] = df['volume'] < (vol_50d_ma * 0.5)
    
    # 2. Pocket Pivot: Up day AND volume > highest down volume over prior 10 days
    is_up_day = df['close'] > df['close'].shift(1)
    is_down_day = df['close'] < df['close'].shift(1)
    down_volume = df['volume'].where(is_down_day, 0)
    highest_down_vol_10d = down_volume.shift(1).rolling(window=10).max()
    df['is_pocket_pivot'] = is_up_day & (df['volume'] > highest_down_vol_10d)
    
    # Above/below indicators
    df['above_sma_50'] = df['close'] > df['sma_50']

    # Phase 27: Elite Climax Exhaustion Score (Vectorized 5-Point Rule)
    return_15 = (df['close'] / df['close'].shift(15)) - 1
    cond1 = (return_15 >= 0.25).astype(int)
    
    dist_10ema = (df['close'] - df['ema_10']) / df['ema_10']
    cond2 = (dist_10ema >= 0.15).astype(int)
    
    vol_10d = df['volume'].rolling(10).mean()
    vol_50d = df['volume'].rolling(50).mean()
    cond3 = (vol_10d >= 1.75 * vol_50d).astype(int)
    
    atr_10 = calculate_atr(df, 10)
    cond4 = (atr_10 >= 1.5 * df['atr_30']).astype(int)
    
    slope_10 = fast_rolling_slope(df['close'], 10)
    slope_30 = fast_rolling_slope(df['close'], 30)
    cond5 = ((slope_10 >= 1.8 * slope_30) & (slope_30 > 0)).astype(int)
    
    df['climax_score'] = cond1 + cond2 + cond3 + cond4 + cond5
    
    # RS Blue Dot calculation (William O'Neil)
    if benchmark_df is not None:
        try:
            from relative_strength import calculate_rs_blue_dot
            df['rs_blue_dot'] = calculate_rs_blue_dot(df, benchmark_df)
        except Exception as e:
            print(f"[Indicators] RS Blue Dot calculation error: {e}")
            df['rs_blue_dot'] = False
    else:
        df['rs_blue_dot'] = False
        
    return df


def get_atr_state(atr_ratio: float) -> str:
    """
    Determine ATR state based on ATR ratio.
    
    Args:
        atr_ratio: ATR(14) / ATR(30)
    
    Returns:
        'Tight', 'Normal', or 'Expanding'
    """
    if atr_ratio is None or pd.isna(atr_ratio):
        return 'Unknown'
    
    if atr_ratio <= 0.85:
        return 'Tight'
    elif atr_ratio <= 1.15:
        return 'Normal'
    else:
        return 'Expanding'


def count_days_above_ema(df: pd.DataFrame, ema_column: str, lookback: int = 10) -> int:
    """
    Count consecutive days the close was above EMA.
    
    Args:
        df: DataFrame with close and EMA column
        ema_column: Name of EMA column to check
        lookback: Number of days to look back
    
    Returns:
        Number of consecutive days above EMA
    """
    if df.empty or ema_column not in df.columns:
        return 0
    
    recent = df.tail(lookback)
    above = recent['close'] > recent[ema_column]
    
    # Count consecutive True from the end
    count = 0
    for val in above.iloc[::-1]:
        if val:
            count += 1
        else:
            break
    
    return count


def count_days_below_ema(df: pd.DataFrame, ema_column: str, lookback: int = 10) -> int:
    """
    Count consecutive days the close was below EMA.
    
    Args:
        df: DataFrame with close and EMA column
        ema_column: Name of EMA column to check
        lookback: Number of days to look back
    
    Returns:
        Number of consecutive days below EMA
    """
    if df.empty or ema_column not in df.columns:
        return 0
    
    recent = df.tail(lookback)
    below = recent['close'] < recent[ema_column]
    
    # Count consecutive True from the end
    count = 0
    for val in below.iloc[::-1]:
        if val:
            count += 1
        else:
            break
    
    return count


def has_big_down_day(df: pd.DataFrame, threshold_pct: float, lookback: int = 10) -> bool:
    """
    Check if there was a big down day in recent trading days.
    
    Args:
        df: DataFrame with daily_pct_change column
        threshold_pct: Threshold percentage (e.g., 4.0 for 4%)
        lookback: Number of days to look back
    
    Returns:
        True if there was a down day exceeding threshold
    """
    if df.empty or 'daily_pct_change' not in df.columns:
        return False
    
    recent = df.tail(lookback)['daily_pct_change']
    return (recent < -threshold_pct).any()


def calculate_power_days(df: pd.DataFrame, lookback: int = 65, threshold_pct: float = 4.0) -> tuple:
    """
    Calculate the number of power days (up moves) and distribution days (down moves)
    in the last `lookback` days.
    
    Args:
        df: DataFrame with daily_pct_change column
        lookback: Number of days to look back (default 65 = ~3 months)
        threshold_pct: Threshold percentage (e.g., 4.0 for 4%)
    
    Returns:
        Tuple of (power_days_count, dist_days_count)
    """
    if df.empty or 'daily_pct_change' not in df.columns:
        return (0, 0)
    
    recent = df.tail(lookback)['daily_pct_change']
    power_days = (recent >= threshold_pct).sum()
    dist_days = (recent <= -threshold_pct).sum()
    
    return int(power_days), int(dist_days)


def has_high_volume_down_day(df: pd.DataFrame, down_pct: float = 5.0, vol_mult: float = 1.5) -> bool:
    """
    Check for high-volume distribution day.
    
    Args:
        df: DataFrame with daily_pct_change, volume, and volume_ma_20 columns
        down_pct: Minimum down percentage
        vol_mult: Volume multiplier threshold
    
    Returns:
        True if there was a high-volume down day
    """
    if df.empty:
        return False
    
    last_row = df.iloc[-1]
    
    pct_change = last_row.get('daily_pct_change', 0)
    volume = last_row.get('volume', 0)
    vol_ma = last_row.get('volume_ma_20', volume)
    
    if vol_ma == 0:
        return False
    
    is_down = pct_change < -down_pct
    is_high_vol = volume >= (vol_mult * vol_ma)
    
    return is_down and is_high_vol


def detect_high_tight_flag(df: pd.DataFrame, min_thrust_pct: float = 70.0, max_thrust_days: int = 40, max_drawdown_pct: float = 25.0, min_flag_days: int = 10, max_flag_days: int = 30) -> dict:
    """
    Detects William O'Neil / Minervini's High Tight Flag (Power Play) setup.
    
    Args:
        df: DataFrame with at least 'high', 'low', 'close' columns
        min_thrust_pct: Minimum percentage gain for the thrust (default 70.0)
        max_thrust_days: Maximum days allowed for the thrust to occur (default 40 days)
        max_drawdown_pct: Maximum allowed drawdown from the peak during the flag
        min_flag_days: Minimum number of days for the consolidation flag
        max_flag_days: Maximum number of days for the consolidation flag
        
    Returns:
        Dictionary with boolean 'is_htf' and metrics.
    """
    result = {'is_htf': False, 'thrust_pct': 0.0, 'drawdown_pct': 0.0, 'flag_days': 0}
    
    if df is None or len(df) < max_thrust_days + min_flag_days:
        return result
        
    recent_history = df.tail(max_flag_days + 5)
    if recent_history.empty:
        return result
        
    high_col = 'High' if 'High' in df.columns else 'high'
    low_col = 'Low' if 'Low' in df.columns else 'low'
    
    peak_val = recent_history[high_col].max()
    peak_idx = recent_history[high_col].idxmax()
    
    try:
        # Get integer index of peak relative to the whole df
        if isinstance(df.index, pd.DatetimeIndex):
             peak_pos = df.index.get_loc(peak_idx)
             # Handle duplicate index case
             if isinstance(peak_pos, slice):
                 peak_pos = peak_pos.start
             elif isinstance(peak_pos, np.ndarray):
                 peak_pos = np.where(peak_pos)[0][0]
        else:
             peak_pos = df.index.get_loc(peak_idx)
             
        current_pos = len(df) - 1
        flag_days = current_pos - peak_pos
    except Exception:
        return result
        
    if not (min_flag_days <= flag_days <= max_flag_days):
        return result
        
    flag_period = df.iloc[peak_pos:]
    lowest_low = flag_period[low_col].min()
    
    drawdown_pct = ((lowest_low - peak_val) / peak_val) * 100.0
    if abs(drawdown_pct) > max_drawdown_pct:
        return result
        
    start_pos = max(0, peak_pos - max_thrust_days)
    thrust_period = df.iloc[start_pos:peak_pos+1]
    
    if len(thrust_period) < 5:
        return result
        
    lowest_before_peak = thrust_period[low_col].min()
    if lowest_before_peak <= 0:
        return result
        
    thrust_pct = ((peak_val - lowest_before_peak) / lowest_before_peak) * 100.0
    
    if thrust_pct >= min_thrust_pct:
        result['is_htf'] = True
        result['thrust_pct'] = round(thrust_pct, 1)
        result['drawdown_pct'] = round(drawdown_pct, 1)
        result['flag_days'] = flag_days
        
    return result


def detect_ants_momentum(df: pd.DataFrame, lookback: int = 15, min_up_days: int = 12) -> bool:
    """
    Detects "Ants" Momentum - A Deepvue concept where a stock has unrelenting 
    buying pressure (e.g., 12 out of 15 days closing higher).
    """
    if df.empty or len(df) < lookback:
        return False
        
    close_col = 'Close' if 'Close' in df.columns else 'close'
    
    recent_period = df.tail(lookback + 1)
    if len(recent_period) < lookback + 1:
        return False
        
    # Check if close is higher than previous close
    is_up_day = recent_period[close_col].diff() > 0
    
    # We drop the first NaN from diff
    up_days_count = is_up_day.iloc[1:].sum()
    
    return bool(up_days_count >= min_up_days)


def calculate_hv1_avwap(df: pd.DataFrame, lookback: int = 252) -> float:
    """
    Calculates the Anchored VWAP from the Highest Volume Day (HV1) in the lookback period.
    Returns the current AVWAP value. Returns 0.0 if not enough data.
    """
    if df.empty or len(df) < 5:
        return 0.0
        
    vol_col = 'Volume' if 'Volume' in df.columns else 'volume'
    close_col = 'Close' if 'Close' in df.columns else 'close'
    high_col = 'High' if 'High' in df.columns else 'high'
    low_col = 'Low' if 'Low' in df.columns else 'low'
    
    # Needs volume to calculate VWAP
    if vol_col not in df.columns:
        return 0.0
        
    lookback_df = df.tail(lookback)
    if lookback_df.empty:
        return 0.0
        
    # Find Highest Volume Day (HV1)
    hv1_idx = lookback_df[vol_col].idxmax()
    
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            hv1_pos = df.index.get_loc(hv1_idx)
            if isinstance(hv1_pos, slice): hv1_pos = hv1_pos.start
            elif isinstance(hv1_pos, np.ndarray): hv1_pos = np.where(hv1_pos)[0][0]
        else:
            hv1_pos = df.index.get_loc(hv1_idx)
    except Exception:
        return 0.0
        
    # Calculate AVWAP from HV1 to current
    period_df = df.iloc[hv1_pos:].copy()
    if period_df.empty:
        return 0.0
        
    # Typical price = (High + Low + Close) / 3
    # If High/Low not available, use Close
    if high_col in period_df.columns and low_col in period_df.columns:
        typical_price = (period_df[high_col] + period_df[low_col] + period_df[close_col]) / 3.0
    else:
        typical_price = period_df[close_col]
        
    pv = typical_price * period_df[vol_col]
    cum_pv = pv.cumsum()
    cum_vol = period_df[vol_col].cumsum()
    
    avwap = cum_pv / cum_vol.replace(0, np.nan)
    
    current_avwap = avwap.iloc[-1]
    if pd.isna(current_avwap):
        return 0.0
        
    return float(current_avwap)


def detect_ema_crossback(df: pd.DataFrame, ema_period: int = 10, max_days_below: int = 5) -> bool:
    """
    Oliver Kell's EMA Crossback (Bounce) setup.
    Price pulls back below the EMA for a few days to shake out weak hands, 
    then violently crosses back above it today.
    """
    if df.empty or len(df) < ema_period + max_days_below:
        return False
        
    close_col = 'Close' if 'Close' in df.columns else 'close'
    vol_col = 'Volume' if 'Volume' in df.columns else 'volume'
    
    # Calculate EMA if not present
    ema_col = f'ema_{ema_period}'
    if ema_col not in df.columns:
        ema_series = df[close_col].ewm(span=ema_period, adjust=False).mean()
    else:
        ema_series = df[ema_col]
        
    # We need to look at the last `max_days_below + 1` days
    # Today: Price > EMA
    # Previous 1 to max_days_below days: Price < EMA
    
    if len(df) < max_days_below + 2:
        return False
        
    today_close = df[close_col].iloc[-1]
    today_ema = ema_series.iloc[-1]
    
    yesterday_close = df[close_col].iloc[-2]
    yesterday_ema = ema_series.iloc[-2]
    
    # Rule 1: Today must have crossed back ABOVE the EMA
    if not (yesterday_close < yesterday_ema and today_close > today_ema):
        return False
        
    # Rule 2: It must have been below the EMA for at most `max_days_below` days.
    # Check backwards from yesterday
    days_below = 0
    for i in range(2, max_days_below + 3):
        if len(df) < i: break
        c = df[close_col].iloc[-i]
        e = ema_series.iloc[-i]
        if c < e:
            days_below += 1
        else:
            break
            
    if days_below == 0 or days_below > max_days_below:
        return False
        
    # Rule 3: Volume expansion (optional but preferred for "violent" crossback)
    # Check if today's volume is > yesterday's volume
    if vol_col in df.columns:
        today_vol = df[vol_col].iloc[-1]
        yest_vol = df[vol_col].iloc[-2]
        if today_vol <= yest_vol:
            return False
            
    return True


def detect_reversal_extension(df: pd.DataFrame, ema_period: int = 10, extension_threshold: float = 12.0) -> bool:
    """
    Oliver Kell's Reversal Extension setup.
    Price is significantly extended to the downside (e.g. >12% below EMA)
    and forms a climax bottom or reversal.
    """
    if df.empty or len(df) < ema_period + 5:
        return False
        
    close_col = 'Close' if 'Close' in df.columns else 'close'
    open_col = 'Open' if 'Open' in df.columns else 'open'
    
    ema_col = f'ema_{ema_period}'
    if ema_col not in df.columns:
        ema_series = df[close_col].ewm(span=ema_period, adjust=False).mean()
    else:
        ema_series = df[ema_col]
        
    today_close = df[close_col].iloc[-1]
    today_open = df[open_col].iloc[-1] if open_col in df.columns else today_close
    today_ema = ema_series.iloc[-1]
    
    if today_ema <= 0:
        return False
        
    # Calculate downside extension
    extension_pct = ((today_ema - today_close) / today_ema) * 100.0
    
    # Must be deeply extended to the downside
    if extension_pct < extension_threshold:
        return False
        
    # Reversal confirmation: Needs to close higher than open (Green candle)
    if today_close <= today_open:
        return False
        
    # Plus it should be up from yesterday's close (or strong intraday reversal)
    yesterday_close = df[close_col].iloc[-2]
    if today_close <= yesterday_close:
        # Not a true reversal if it still closed red vs yesterday
        return False
        
    return True


def detect_power_trend(df: pd.DataFrame) -> bool:
    """
    Deepvue Power Trend / William O'Neil Power Trend
    Strict multi-timeframe regime filter.
    1. Close > 21 EMA
    2. 21 EMA > 50 SMA for at least 20 consecutive days
    3. 50 SMA has been trending upwards for at least 20 consecutive days
    """
    if df.empty or len(df) < 50:
        return False
        
    close_col = 'Close' if 'Close' in df.columns else 'close'
    
    ema_21_col = 'ema_21'
    sma_50_col = 'sma_50'
    
    if ema_21_col not in df.columns:
        ema_21 = df[close_col].ewm(span=21, adjust=False).mean()
    else:
        ema_21 = df[ema_21_col]
        
    if sma_50_col not in df.columns:
        sma_50 = df[close_col].rolling(window=50, min_periods=50).mean()
    else:
        sma_50 = df[sma_50_col]
        
    # Condition 1: Close > 21 EMA
    if df[close_col].iloc[-1] <= ema_21.iloc[-1]:
        return False
        
    # Need at least 21 days for the lookback check
    if len(ema_21) < 21 or len(sma_50) < 21:
        return False
        
    last_21_ema = ema_21.tail(21)
    last_21_sma = sma_50.tail(21)
    
    # Condition 2: 21 EMA > 50 SMA for 20+ days
    if not (last_21_ema.iloc[1:] > last_21_sma.iloc[1:]).all():
        return False
        
    # Condition 3: 50 SMA trending upwards for 20+ days (slope >= 0)
    sma_diff = last_21_sma.diff().dropna()
    if not (sma_diff >= 0).all():
        return False
        
    return True


def detect_3_weeks_tight(df: pd.DataFrame, max_variance_pct: float = 2.0) -> bool:
    """
    Detects David Ryan's '3-Weeks Tight' pattern.
    Requires the last 3 weekly closes to be within a max_variance_pct of each other,
    and the stock to be in a general uptrend (above 50 SMA).
    """
    if df.empty or len(df) < 20:
        return False
        
    close_col = 'Close' if 'Close' in df.columns else 'close'
    
    # 1. Ensure stock is in a macro uptrend (Close > 50 SMA and 200 SMA)
    sma_50_col = 'sma_50'
    if sma_50_col not in df.columns:
        sma_50 = df[close_col].rolling(window=50, min_periods=20).mean()
    else:
        sma_50 = df[sma_50_col]
        
    sma_200_col = 'sma_200'
    if sma_200_col not in df.columns:
        sma_200 = df[close_col].rolling(window=200, min_periods=50).mean()
    else:
        sma_200 = df[sma_200_col]
        
    current_close = df[close_col].iloc[-1]
        
    if current_close < sma_50.iloc[-1] or current_close < sma_200.iloc[-1]:
        return False
        
    # 1.5. Ensure stock is within 20% of its 52-Week High
    high_col = 'High' if 'High' in df.columns else ('high' if 'high' in df.columns else close_col)
    high_52w = df[high_col].tail(252).max()
    if high_52w > 0 and current_close < (high_52w * 0.8):
        return False
        
    # 2. Resample to weekly timeframe (Friday close)
    # Ensure index is datetime for resampling
    try:
        temp_df = df.copy()
        if not isinstance(temp_df.index, pd.DatetimeIndex):
            if 'Date' in temp_df.columns:
                temp_df.set_index('Date', inplace=True)
            elif 'date' in temp_df.columns:
                temp_df.set_index('date', inplace=True)
            temp_df.index = pd.to_datetime(temp_df.index)
            
        weekly_closes = temp_df[close_col].resample('W-FRI').last().dropna()
    except Exception:
        # Fallback if we can't cleanly resample (e.g. integer index without dates)
        # Just grab the last 15 days and look at every 5th day as a rough proxy
        closes = df[close_col].values
        if len(closes) < 15: return False
        weekly_closes = [closes[-11], closes[-6], closes[-1]]
        weekly_closes = pd.Series(weekly_closes)
        
    if len(weekly_closes) < 3:
        return False
        
    # Get last 3 weeks
    last_3_weeks = weekly_closes.iloc[-3:]
    
    max_c = last_3_weeks.max()
    min_c = last_3_weeks.min()
    
    if min_c <= 0:
        return False
        
    variance_pct = ((max_c - min_c) / min_c) * 100.0
    
    return bool(variance_pct <= max_variance_pct)


def calculate_webster_sell_signals(df: pd.DataFrame, index_df: pd.DataFrame) -> dict:
    """
    Mike Webster's Quick, Quicksand, and Grateful Dead sell signals based on RS Line.
    RS Line = Stock Close / Index Close
    
    Returns a dict with the signal state and MAs.
    """
    if df.empty or index_df is None or index_df.empty or len(df) < 50 or len(index_df) < 50:
        return {'signal': '⚪ Unknown', 'color': '#6b7280'}
        
    close_col = 'Close' if 'Close' in df.columns else 'close'
    idx_close_col = 'Close' if 'Close' in index_df.columns else 'close'
    
    # Simple alignment
    if isinstance(df.index, pd.DatetimeIndex) and isinstance(index_df.index, pd.DatetimeIndex):
        joined = pd.DataFrame({'stock': df[close_col], 'index': index_df[idx_close_col]}).dropna()
    else:
        min_len = min(len(df), len(index_df))
        joined = pd.DataFrame({
            'stock': df[close_col].values[-min_len:],
            'index': index_df[idx_close_col].values[-min_len:]
        })
        
    if len(joined) < 50:
        return {'signal': '⚪ Unknown', 'color': '#6b7280'}
        
    rs_line = joined['stock'] / joined['index']
    
    # Calculate RS MAs (10 EMA fast, 21 EMA med, 50 SMA slow)
    rs_ema_10 = rs_line.ewm(span=10, adjust=False).mean()
    rs_ema_21 = rs_line.ewm(span=21, adjust=False).mean()
    rs_sma_50 = rs_line.rolling(window=50, min_periods=50).mean()
    
    rs_val = rs_line.iloc[-1]
    fast_ma = rs_ema_10.iloc[-1]
    med_ma = rs_ema_21.iloc[-1]
    slow_ma = rs_sma_50.iloc[-1]
    
    # Evaluate State
    if pd.isna(slow_ma):
        signal = '⚪ Unknown'
        color = '#6b7280'
    elif rs_val < slow_ma:
        signal = '🔴 Grateful Dead (Exit)'
        color = '#ef4444'
    elif rs_val < med_ma:
        signal = '🟠 Quicksand (Warning)'
        color = '#f97316'
    elif rs_val < fast_ma:
        signal = '🟡 Quick (Trim)'
        color = '#f59e0b'
    else:
        signal = '🟢 Hold (Strong)'
        color = '#10b981'
        
    return {
        'signal': signal,
        'color': color,
        'rs_val': rs_val,
        'fast_ma': fast_ma,
        'med_ma': med_ma,
        'slow_ma': slow_ma
    }


def calculate_atr_chandelier_stop(df: pd.DataFrame, atr_multiplier: float = 2.5, high_window: int = 20) -> float:
    """
    Calculate the ATR Chandelier trailing stop value.
    Anchors to the highest close over the last `high_window` days,
    and subtracts `atr_multiplier` * 14-day ATR.
    
    Args:
        df: DataFrame with OHLCV data, must have 'close' and 'atr_14' columns.
        atr_multiplier: The multiplier for ATR (default 2.5).
        high_window: Number of days to look back for the highest close (default 20).
        
    Returns:
        The stop price as a float, or 0.0 if unable to calculate.
    """
    if df.empty or len(df) < high_window:
        return 0.0
        
    # Get the highest close in the lookback window
    highest_close = df['close'].tail(high_window).max()
    
    # Get the latest 14-day ATR
    latest_atr = df['atr_14'].iloc[-1]
    
    if pd.isna(latest_atr) or pd.isna(highest_close):
        return 0.0
        
    stop_price = highest_close - (atr_multiplier * latest_atr)
    return stop_price
    # Test with sample data
    print("Testing technical indicators...")
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    sample_df = pd.DataFrame({
        'date': dates,
        'open': np.random.randint(100, 110, 100),
        'high': np.random.randint(110, 120, 100),
        'low': np.random.randint(90, 100, 100),
        'close': np.random.randint(100, 110, 100),
        'volume': np.random.randint(1000000, 5000000, 100),
    })
    
    result = add_technical_indicators(sample_df)
    print(result.tail())
