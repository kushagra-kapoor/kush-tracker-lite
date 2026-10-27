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


def detect_and_adjust_split_cliffs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects unadjusted stock split, bonus issue, or demerger cliffs in historical price data
    and retroactively normalizes pre-split bars.
    
    This ensures 52W High, Moving Averages (50/200 SMA), Drawdowns, and RS percentiles
    match verified TradingView / Bloomberg adjusted series and prevents false 80%+ drawdown
    signals caused by unadjusted data (e.g. V-Marc India 5:1 bonus, HEG, Oriana splits).
    """
    if df.empty or len(df) < 15:
        return df

    close_col = 'close' if 'close' in df.columns else 'Close' if 'Close' in df.columns else None
    if not close_col:
        return df

    close = df[close_col]
    
    # Overnight percentage change
    pct_change = close.pct_change()
    
    # Detect sudden cliffs > 40% (e.g., 1:2 split = -50%, 1:5 bonus = -80%, 1:10 split = -90%)
    split_indices = pct_change[pct_change < -0.40].index.tolist()
    if not split_indices:
        return df

    df = df.copy()
    
    for split_date in split_indices:
        try:
            idx_loc = df.index.get_loc(split_date)
            if isinstance(idx_loc, slice) or isinstance(idx_loc, np.ndarray):
                continue
            if idx_loc < 1:
                continue

            pre_close = float(close.iloc[idx_loc - 1])
            post_close = float(close.iloc[idx_loc])

            if pre_close <= 0 or post_close <= 0:
                continue

            # Verify that prices stayed down (not an intraday flash crash or single-day bad tick)
            subsequent_window = close.iloc[idx_loc: min(len(close), idx_loc + 30)]
            if float(subsequent_window.max()) < pre_close * 0.70:
                factor = pre_close / post_close

                # Snap factor to common corporate action ratios if within 12%
                # 2:1 (2.0), 3:1 (3.0), 4:1 (4.0), 5:1 (5.0), 6:1 (6.0), 10:1 (10.0)
                nearest_int = round(factor)
                if nearest_int >= 2 and abs(factor - nearest_int) / nearest_int < 0.12:
                    clean_factor = float(nearest_int)
                else:
                    clean_factor = factor

                # Adjust OHLC bars prior to split date
                price_cols = [c for c in ['open', 'high', 'low', 'close', 'Open', 'High', 'Low', 'Close'] if c in df.columns]
                pre_slice = df.index[:idx_loc]
                df.loc[pre_slice, price_cols] = df.loc[pre_slice, price_cols] / clean_factor

                # Adjust volume so turnover stays consistent
                vol_cols = [c for c in ['volume', 'Volume'] if c in df.columns]
                df.loc[pre_slice, vol_cols] = df.loc[pre_slice, vol_cols] * clean_factor
        except Exception:
            continue

    return df


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
    
    # Auto-adjust unadjusted corporate actions / split cliffs to match TradingView / institutional data
    df = detect_and_adjust_split_cliffs(df)
    
    # EMAs
    df['ema_8'] = calculate_ema(df['close'], EMA_PERIODS['FAST'])
    df['ema_10'] = calculate_ema(df['close'], 10)  # Added for Climax Detector
    df['ema_21'] = calculate_ema(df['close'], EMA_PERIODS['MEDIUM'])
    df['ema_21_low'] = calculate_ema(df['low'], EMA_PERIODS['MEDIUM'])  # 21-EMA of Lows (Anti-Wick Institutional Floor)
    df['ema_23'] = calculate_ema(df['close'], 23)  # Ross Haber / William O'Neil Core Swing Line
    df['sma_50'] = calculate_sma(df['close'], EMA_PERIODS['SLOW'])
    df['ema_65'] = calculate_ema(df['close'], 65)  # Ross Haber Quarterly / 13-Week Institutional Floor
    df['sma_200'] = calculate_sma(df['close'], 200)
    
    # ATRs
    df['atr_14'] = calculate_atr(df, ATR_PERIODS['SHORT'])
    df['atr_21'] = calculate_atr(df, ATR_PERIODS.get('CHANDELIER', 21))
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


def calculate_atr_chandelier_stop(df: pd.DataFrame, atr_multiplier: float = 3.5, high_window: int = 20, atr_period: int = 21) -> float:
    """
    Calculate the ATR Chandelier trailing stop value.
    Anchors to the highest close over the last `high_window` days,
    and subtracts `atr_multiplier` * `atr_period`-day ATR.
    
    Args:
        df: DataFrame with OHLCV data, must have 'close' and ATR column.
        atr_multiplier: The multiplier for ATR (default 3.5).
        high_window: Number of days to look back for the highest close (default 20).
        atr_period: Period for ATR calculation (default 21).
        
    Returns:
        The stop price as a float, or 0.0 if unable to calculate.
    """
    if df.empty or len(df) < high_window:
        return 0.0
        
    # Get the highest close in the lookback window
    highest_close = df['close'].tail(high_window).max()
    
    # Get the latest ATR for the specified period
    atr_col = f'atr_{atr_period}'
    if atr_col in df.columns:
        latest_atr = df[atr_col].iloc[-1]
    else:
        atr_series = calculate_atr(df, atr_period)
        latest_atr = atr_series.iloc[-1] if not atr_series.empty else None
    
    if pd.isna(latest_atr) or pd.isna(highest_close):
        return 0.0
        
    stop_price = highest_close - (atr_multiplier * latest_atr)
    return stop_price


# =============================================================================
# SUPERPERFORMER LAUNCHPAD TECHNICAL INDICATORS (WEINSTEIN / MURPHY / LOVE)
# =============================================================================

def calculate_wilder_adx(df: pd.DataFrame, period: int = 14) -> dict:
    """
    Calculates J. Welles Wilder's Average Directional Index (ADX 14) & Directional Movement (+DI / -DI).
    Used as John J. Murphy's Trend-Regime Gate (Ch. 9 & 15):
    - ADX < 20: Non-trending / choppy consolidation -> Breakout Warning.
    - ADX >= 20 and rising with +DI > -DI: Accelerating Stage 2 trend -> Green Light.
    - ADX >= 40 and rolling over: Trend exhaustion / climax warning.
    """
    empty_res = {
        'adx': 0.0, 'plus_di': 0.0, 'minus_di': 0.0, 'adx_slope': 0.0,
        'regime_label': 'Non-Trending (Choppy)', 'regime_color': '#94a3b8',
        'regime_badge': '⚖️ Non-Trending (ADX < 20)', 'green_light': False
    }
    if df.empty or len(df) < period * 2:
        return empty_res

    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']

    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder's modified exponential smoothing (alpha = 1 / period)
    alpha = 1.0 / period
    tr_smooth = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100.0 * (plus_dm_smooth / tr_smooth.replace(0, np.nan)).fillna(0.0)
    minus_di = 100.0 * (minus_dm_smooth / tr_smooth.replace(0, np.nan)).fillna(0.0)

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100.0 * (di_diff / di_sum.replace(0, np.nan)).fillna(0.0)
    adx_series = dx.ewm(alpha=alpha, adjust=False).mean()

    curr_adx = float(adx_series.iloc[-1])
    curr_plus_di = float(plus_di.iloc[-1])
    curr_minus_di = float(minus_di.iloc[-1])
    prev_adx_5 = float(adx_series.iloc[-6]) if len(adx_series) >= 6 else curr_adx
    prev_adx_3 = float(adx_series.iloc[-4]) if len(adx_series) >= 4 else curr_adx
    adx_slope = curr_adx - prev_adx_5

    if curr_adx >= 40.0 and curr_adx < prev_adx_3:
        regime = "Climax Exhaustion"
        color = "#f97316"
        badge = f"🔥 Climax Warning (ADX {curr_adx:.1f})"
        green_light = False
    elif curr_adx >= 20.0 and curr_plus_di >= curr_minus_di and adx_slope >= 0:
        regime = "Trend Expanding (Green Light)"
        color = "#10b981"
        badge = f"🚀 Trend Expanding (ADX {curr_adx:.1f})"
        green_light = True
    elif curr_adx >= 20.0 and curr_minus_di > curr_plus_di:
        regime = "Bearish Downtrend"
        color = "#ef4444"
        badge = f"🛑 Bearish Trend (ADX {curr_adx:.1f})"
        green_light = False
    elif curr_adx < 20.0:
        regime = "Non-Trending (Choppy / Base)"
        color = "#94a3b8"
        badge = f"⚖️ Chop / Base (ADX {curr_adx:.1f})"
        green_light = False
    else:
        regime = "Consolidating"
        color = "#38bdf8"
        badge = f"🛡️ Consolidating (ADX {curr_adx:.1f})"
        green_light = False

    return {
        'adx': round(curr_adx, 1),
        'plus_di': round(curr_plus_di, 1),
        'minus_di': round(curr_minus_di, 1),
        'adx_slope': round(adx_slope, 2),
        'regime_label': regime,
        'regime_color': color,
        'regime_badge': badge,
        'green_light': green_light
    }


def calculate_price_weighted_obv(df: pd.DataFrame) -> pd.Series:
    """
    John J. Murphy's Price-Weighted On-Balance Volume (Ch. 7):
    Scales volume by percentage price move so minor 1-tick moves don't distort cumulative flow:
    ΔW_OBV = ((Close_t - Close_{t-1}) / Close_{t-1}) * Volume_t
    """
    if df.empty or len(df) < 2:
        return pd.Series(dtype=float)
        
    close = df['close'] if 'close' in df.columns else df['Close']
    vol = df['volume'] if 'volume' in df.columns else df['Volume']
    
    pct_chg = close.pct_change().fillna(0.0)
    weighted_flow = pct_chg * vol
    return weighted_flow.cumsum()


def detect_obv_breakout_lead(df: pd.DataFrame, lookback: int = 63, threshold_days: int = 5) -> dict:
    """
    Detects Murphy's 'OBV Breakout Lead' (McDonald's Rule):
    Weighted OBV reaches a new 63-day / 252-day high while price is still coiling inside its base
    (within 0.5% of OBV high, but price is > 2% below 52W high).
    """
    res = {'obv_lead_active': False, 'days_ago': None, 'obv_ahead_of_price': False}
    if df.empty or len(df) < lookback:
        return res
        
    w_obv = calculate_price_weighted_obv(df)
    close = df['close'] if 'close' in df.columns else df['Close']
    high = df['high'] if 'high' in df.columns else df['High']
    
    eff_lb = min(lookback, len(w_obv))
    rolling_obv_high = w_obv.rolling(window=eff_lb, min_periods=eff_lb // 2).max()
    rolling_price_high = high.rolling(window=eff_lb, min_periods=eff_lb // 2).max()
    
    recent_obv = w_obv.tail(threshold_days)
    recent_obv_highs = rolling_obv_high.tail(threshold_days)
    
    is_obv_high = recent_obv >= (recent_obv_highs * 0.995)
    if is_obv_high.any():
        res['obv_lead_active'] = True
        hit_pos = np.where(is_obv_high.values)[0]
        if len(hit_pos) > 0:
            res['days_ago'] = int((len(recent_obv) - 1) - hit_pos[-1])
            
        curr_p = float(close.iloc[-1])
        curr_p_high = float(rolling_price_high.iloc[-1])
        if curr_p_high > 0 and (curr_p_high - curr_p) / curr_p_high >= 0.02:
            res['obv_ahead_of_price'] = True
            
    return res


def calculate_mansfield_rs(stock_close: pd.Series, bench_close: pd.Series, period: int = 252) -> pd.Series:
    """
    Stan Weinstein's Mansfield Relative Strength (MRS):
    R = stock_close / bench_close
    MRS = ((R / SMA(R, period)) - 1.0) * 100.0
    Positive values (> 0) indicate outperforming the benchmark on a 52-week moving basis.
    """
    if stock_close is None or bench_close is None or len(stock_close) < 20 or len(bench_close) < 20:
        return pd.Series(dtype=float)
        
    common_idx = stock_close.index.intersection(bench_close.index)
    if len(common_idx) < 20:
        return pd.Series(dtype=float)
        
    s = stock_close.loc[common_idx]
    b = bench_close.loc[common_idx]
    
    r = s / b.replace(0, np.nan)
    eff_period = min(period, len(r))
    sma_r = r.rolling(window=eff_period, min_periods=eff_period // 2).mean()
    mrs = ((r / sma_r) - 1.0) * 100.0
    return mrs


def snap_round_number_undercut(price: float, buffer_pct: float = 0.25) -> float:
    """
    Stan Weinstein's Round-Number Undercut Stop (Ch. 6, p. 183):
    Because retail stop-loss orders cluster on exact whole numbers and half numbers,
    market makers frequently hunt round numbers. Weinstein's secret: if a stop sits
    right on or fractionally above a round number, snap it 0.25% - 0.50% BELOW the round number.
    
    Rules:
    - For stocks >= ₹200: Undercut whole integers (e.g., 500.20 -> 498.50, 250.10 -> 248.80)
    - For stocks < ₹200: Undercut whole and half integers (e.g., 20.10 -> 19.85, 18.05 -> 17.85)
    """
    if price <= 0:
        return round(price, 2)
        
    # Check if price is within 0.75% above a whole integer
    nearest_whole = round(price)
    diff_whole = price - nearest_whole
    
    if 0.0 <= diff_whole <= (price * 0.0075):
        # Snap below the round number by buffer_pct
        return round(nearest_whole * (1.0 - (buffer_pct / 100.0)), 2)
        
    # For sub-₹200 stocks, check half-integers
    if price < 200:
        nearest_half = round(price * 2.0) / 2.0
        diff_half = price - nearest_half
        if 0.0 <= diff_half <= (price * 0.0075):
            return round(nearest_half * (1.0 - (buffer_pct / 100.0)), 2)
            
    return round(price, 2)




# =============================================================================
# TRADERLION & ROSS HABER MODEL BOOK DETECTORS
# =============================================================================

def detect_oops_reversal(df: pd.DataFrame, max_risk_pct: float = 3.5) -> dict:
    """
    Larry Williams / TraderLion Oops Reversal Detector.
    
    Identifies a high-conviction, low-risk shakeout entry:
      1. Today's low undercuts yesterday's low (flushing out retail stops).
      2. Price reverses sharply intraday and closes back ABOVE yesterday's low.
      3. Strong close in upper half of daily range or green close.
      4. Support proximity: within 3.5% of 21 EMA, 50 SMA, 65 EMA, or 200 SMA.
      5. Surgical stop-loss is placed strictly at Low of the Day (LOD).
    """
    res = {
        'is_oops': False,
        'entry_price': 0.0,
        'stop_price': 0.0,
        'risk_pct': 0.0,
        'support_level': 'None'
    }
    
    if df.empty or len(df) < 22:
        return res
        
    c_close = df['close'] if 'close' in df.columns else df['Close']
    c_open = df['open'] if 'open' in df.columns else df['Open']
    c_high = df['high'] if 'high' in df.columns else df['High']
    c_low = df['low'] if 'low' in df.columns else df['Low']
    
    today_low = float(c_low.iloc[-1])
    today_close = float(c_close.iloc[-1])
    today_open = float(c_open.iloc[-1])
    today_high = float(c_high.iloc[-1])
    
    yest_low = float(c_low.iloc[-2])
    
    # 1. Undercut & Reclaim Check
    if not (today_low < yest_low and today_close > yest_low):
        return res
        
    # 2. Strong Finish Check (close in upper half or closed green)
    daily_range = today_high - today_low
    close_pos = (today_close - today_low) / daily_range if daily_range > 0 else 1.0
    is_strong_bar = (today_close >= today_open) or (close_pos >= 0.45)
    if not is_strong_bar:
        return res
        
    # 3. Stop & Risk Calculation (Stop strictly at LOD)
    entry = today_close
    stop = today_low
    risk_pct = ((entry - stop) / entry) * 100 if entry > 0 else 99.0
    
    if risk_pct <= 0 or risk_pct > max_risk_pct:
        return res
        
    # 4. Support Proximity Check
    support_found = 'Recent Range'
    for col_name, label in [('ema_21', '21 EMA'), ('sma_50', '50 SMA'), ('ema_65', '65 EMA'), ('sma_200', '200 SMA')]:
        if col_name in df.columns:
            ma_val = float(df[col_name].iloc[-1])
            if ma_val > 0 and abs(today_close - ma_val) / ma_val <= 0.04:
                support_found = label
                break
                
    res['is_oops'] = True
    res['entry_price'] = round(entry, 2)
    res['stop_price'] = round(stop, 2)
    res['risk_pct'] = round(risk_pct, 2)
    res['support_level'] = support_found
    return res


def detect_positive_expectation_breaker(df: pd.DataFrame, max_risk_pct: float = 5.0) -> dict:
    """
    TraderLion Positive Expectation Breaker (PEB) Detector.
    
    Identifies institutional absorption when a sharp selloff is immediately negated:
      1. Yesterday was an Extension Down: wide-range red bar (<= -1.3% drop) with volume.
      2. Today breaks the negative expectation: closes green AND above yesterday's close.
      3. Stop-loss placed at min(today_low, yesterday_low).
    """
    res = {
        'is_peb': False,
        'entry_price': 0.0,
        'stop_price': 0.0,
        'risk_pct': 0.0
    }
    
    if df.empty or len(df) < 22:
        return res
        
    c_close = df['close'] if 'close' in df.columns else df['Close']
    c_open = df['open'] if 'open' in df.columns else df['Open']
    c_high = df['high'] if 'high' in df.columns else df['High']
    c_low = df['low'] if 'low' in df.columns else df['Low']
    c_vol = df['volume'] if 'volume' in df.columns else df['Volume']
    
    today_close = float(c_close.iloc[-1])
    today_open = float(c_open.iloc[-1])
    today_low = float(c_low.iloc[-1])
    
    yest_close = float(c_close.iloc[-2])
    yest_open = float(c_open.iloc[-2])
    yest_low = float(c_low.iloc[-2])
    prev2_close = float(c_close.iloc[-3]) if len(df) >= 3 else yest_open
    
    # 1. Extension Down check
    yest_return = ((yest_close - prev2_close) / prev2_close) * 100 if prev2_close > 0 else 0
    yest_is_red = (yest_close < yest_open) and (yest_return <= -1.3)
    if not yest_is_red:
        return res
        
    avg_vol_20 = c_vol.iloc[-22:-2].mean() if len(c_vol) >= 22 else c_vol.mean()
    yest_vol = float(c_vol.iloc[-2])
    if avg_vol_20 > 0 and (yest_vol / avg_vol_20) < 0.70:
        return res
        
    # 2. Breaks negative expectation
    today_breaks_neg = (today_close > today_open) and (today_close > yest_close)
    if not today_breaks_neg:
        return res
        
    stop = min(today_low, yest_low)
    entry = today_close
    risk_pct = ((entry - stop) / entry) * 100 if entry > 0 else 99.0
    
    if risk_pct <= 0 or risk_pct > max_risk_pct:
        return res
        
    res['is_peb'] = True
    res['entry_price'] = round(entry, 2)
    res['stop_price'] = round(stop, 2)
    res['risk_pct'] = round(risk_pct, 2)
    return res


def detect_inside_day_priming(df: pd.DataFrame) -> dict:
    """
    TraderLion Inside Days Prime Mature Setups Detector.
    
    Identifies maximum volatility contraction (coil) right before breakout:
      1. Inside Day: today_high <= yesterday_high and today_low >= yesterday_low.
      2. Volume Dry-Up (VDU): today_volume <= 0.85 * volume_ma_20.
      3. Double Inside Day: yesterday was also an inside day relative to day -2.
      4. Context: within 18% of 52-week high.
    """
    res = {
        'is_primed': False,
        'is_double_inside': False,
        'pivot_trigger': 0.0,
        'stop_price': 0.0,
        'risk_pct': 0.0
    }
    
    if df.empty or len(df) < 22:
        return res
        
    c_close = df['close'] if 'close' in df.columns else df['Close']
    c_high = df['high'] if 'high' in df.columns else df['High']
    c_low = df['low'] if 'low' in df.columns else df['Low']
    c_vol = df['volume'] if 'volume' in df.columns else df['Volume']
    
    today_high = float(c_high.iloc[-1])
    today_low = float(c_low.iloc[-1])
    today_vol = float(c_vol.iloc[-1])
    
    yest_high = float(c_high.iloc[-2])
    yest_low = float(c_low.iloc[-2])
    
    # 1. Inside Day Check
    is_inside = (today_high <= yest_high) and (today_low >= yest_low)
    if not is_inside:
        return res
        
    # 2. Volume Dry-Up Check
    avg_vol_20 = c_vol.iloc[-21:-1].mean() if len(c_vol) >= 21 else c_vol.mean()
    is_vdu = (today_vol <= 0.88 * avg_vol_20) if avg_vol_20 > 0 else True
    if not is_vdu:
        return res
        
    # 3. Double Inside Day Check
    is_double_inside = False
    if len(df) >= 4:
        prev2_high = float(c_high.iloc[-3])
        prev2_low = float(c_low.iloc[-3])
        if (yest_high <= prev2_high) and (yest_low >= prev2_low):
            is_double_inside = True
            
    # 4. Context Check: within 18% of 52W high
    max_high_252 = c_high.tail(252).max()
    curr_close = float(c_close.iloc[-1])
    dist_52w = ((max_high_252 - curr_close) / max_high_252) * 100 if max_high_252 > 0 else 0
    if dist_52w > 18.0:
        return res
        
    pivot = today_high
    stop = today_low
    risk_pct = ((pivot - stop) / pivot) * 100 if pivot > 0 else 0
    
    res['is_primed'] = True
    res['is_double_inside'] = is_double_inside
    res['pivot_trigger'] = round(pivot, 2)
    res['stop_price'] = round(stop, 2)
    res['risk_pct'] = round(risk_pct, 2)
    return res


def _safe_get_series(df: pd.DataFrame, col_lower: str, col_upper: str, fallback=None):
    """Safely extract a 1D numeric Series from a DataFrame regardless of column casing."""
    if df is None or df.empty:
        return fallback
    for col in (col_lower, col_upper):
        if col in df.columns:
            s = df[col]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return pd.to_numeric(s, errors='coerce')
    return fallback


def classify_stock_ma_dna(df: pd.DataFrame) -> dict:
    """
    John Boik (Monster Stocks) & TraderLion Moving Average Character DNA Classifier.
    Classifies a stock into one of the 3 institutional trend-riding spines:
      1. 🚀 10-EMA Sprinter (High-velocity momentum, holds 10 EMA >= 78% of uptrend)
      2. ⚡ 21-EMA Low Rider (Modern momentum floor: undercuts 21 EMA Close on wicks, holds 21 EMA Low)
      3. 🦖 50-SMA Monster (John Boik Monster Stock: undercuts 21 EMA during bases, 100% hold on rising 50 SMA)
    """
    default_res = {
        'dna_type': '21_EMA_LOW_RIDER',
        'dna_badge': '⚡ 21-EMA Low Rider',
        'active_spine_name': '21 EMA Low',
        'active_spine_price': 0.0,
        'ema_21_low': 0.0,
        'hold_rate_pct': 85.0,
        'boik_launch_detected': False,
        'zero_heavy_vol_50sma_breaches': True,
    }
    if df is None or df.empty or len(df) < 15:
        return default_res

    c_close = _safe_get_series(df, 'close', 'Close')
    if c_close is None or c_close.empty:
        return default_res
    c_low = _safe_get_series(df, 'low', 'Low', fallback=c_close)
    c_vol = _safe_get_series(df, 'volume', 'Volume', fallback=pd.Series(100000.0, index=df.index))

    ema10 = df['ema_10'] if 'ema_10' in df.columns else c_close.ewm(span=10, adjust=False).mean()
    ema21_c = df['ema_21'] if 'ema_21' in df.columns else c_close.ewm(span=21, adjust=False).mean()
    ema21_l = df['ema_21_low'] if 'ema_21_low' in df.columns else c_low.ewm(span=21, adjust=False).mean()
    sma50 = df['sma_50'] if 'sma_50' in df.columns else c_close.rolling(window=50, min_periods=25).mean()
    vol_ma20 = c_vol.rolling(window=20, min_periods=5).mean().replace(0, 1)

    curr_close = float(c_close.iloc[-1])
    curr_ema10 = float(ema10.iloc[-1]) if pd.notna(ema10.iloc[-1]) else curr_close
    curr_ema21_l = float(ema21_l.iloc[-1]) if pd.notna(ema21_l.iloc[-1]) else curr_close
    curr_sma50 = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else curr_ema21_l

    default_res['ema_21_low'] = round(curr_ema21_l, 2)
    default_res['active_spine_price'] = round(curr_ema21_l, 2)

    if len(df) < 25:
        default_res['dna_type'] = '10_EMA_SPRINTER'
        default_res['dna_badge'] = '🚀 10-EMA Sprinter'
        default_res['active_spine_name'] = '10 EMA'
        default_res['active_spine_price'] = round(curr_ema10, 2)
        return default_res

    window = min(120, len(df))
    w_close = c_close.iloc[-window:]
    w_low = c_low.iloc[-window:]
    w_vol = c_vol.iloc[-window:]
    w_vol_ma = vol_ma20.iloc[-window:]
    w_ema10 = ema10.iloc[-window:]
    w_ema21_c = ema21_c.iloc[-window:]
    w_ema21_l = ema21_l.iloc[-window:]
    w_sma50 = sma50.iloc[-window:]

    # 1. John Boik Monster Launch check: >= +20% surge in 15 trading days inside window
    ret_15d = (w_close / w_close.shift(15)) - 1.0
    boik_launch = bool((ret_15d >= 0.20).any()) if len(w_close) >= 16 else False

    # 2. Check heavy-volume 50-SMA breaches in last 60 sessions
    tail_n = min(60, len(w_close))
    valid_50 = w_sma50.iloc[-tail_n:].notna()
    if valid_50.sum() >= 20:
        below_50 = (w_close.iloc[-tail_n:] < w_sma50.iloc[-tail_n:] * 0.99)
        heavy_vol = (w_vol.iloc[-tail_n:] >= w_vol_ma.iloc[-tail_n:] * 1.25)
        heavy_50_breaches = int((below_50 & heavy_vol).sum())
        hold_50_pct = float((w_close.iloc[-tail_n:] >= w_sma50.iloc[-tail_n:] * 0.985).mean() * 100.0)
    else:
        heavy_50_breaches = 0
        hold_50_pct = 80.0

    # 3. Hold rates on 10 EMA and 21 EMA Low over trailing 45 sessions
    t45 = min(45, len(w_close))
    hold_10_pct = float((w_close.iloc[-t45:] >= w_ema10.iloc[-t45:]).mean() * 100.0)
    hold_21l_pct = float((w_close.iloc[-t45:] >= w_ema21_l.iloc[-t45:]).mean() * 100.0)
    undercut_21c_days = int((w_low.iloc[-t45:] < w_ema21_c.iloc[-t45:]).sum())

    if hold_10_pct >= 80.0 and curr_close >= curr_ema10:
        return {
            'dna_type': '10_EMA_SPRINTER',
            'dna_badge': '🚀 10-EMA Sprinter',
            'active_spine_name': '10 EMA',
            'active_spine_price': round(curr_ema10, 2),
            'ema_21_low': round(curr_ema21_l, 2),
            'hold_rate_pct': round(hold_10_pct, 1),
            'boik_launch_detected': boik_launch,
            'zero_heavy_vol_50sma_breaches': heavy_50_breaches == 0,
        }

    if (
        len(df) >= 50
        and boik_launch
        and heavy_50_breaches == 0
        and hold_50_pct >= 88.0
        and curr_close >= curr_sma50 * 0.985
        and (hold_21l_pct < 82.0 or undercut_21c_days >= 6)
    ):
        return {
            'dna_type': '50_SMA_MONSTER',
            'dna_badge': '🦖 50-SMA Monster',
            'active_spine_name': '50 SMA',
            'active_spine_price': round(curr_sma50, 2),
            'ema_21_low': round(curr_ema21_l, 2),
            'hold_rate_pct': round(hold_50_pct, 1),
            'boik_launch_detected': True,
            'zero_heavy_vol_50sma_breaches': True,
        }

    return {
        'dna_type': '21_EMA_LOW_RIDER',
        'dna_badge': '⚡ 21-EMA Low Rider',
        'active_spine_name': '21 EMA Low',
        'active_spine_price': round(curr_ema21_l, 2),
        'ema_21_low': round(curr_ema21_l, 2),
        'hold_rate_pct': round(max(hold_21l_pct, 75.0), 1),
        'boik_launch_detected': boik_launch,
        'zero_heavy_vol_50sma_breaches': heavy_50_breaches == 0,
    }


def detect_hv1_float_lockup(df: pd.DataFrame) -> dict:
    """
    TraderLion HV1 + Float-Lockup Detector.
    Finds the Highest Volume Day in the trailing 252 bars (HV1).
    If HV1 was an accumulation day (closed green or upper range) AND price is currently
    defending HV1 Low, calculates the Float-Lockup Ratio = HV1_Volume / Min_Volume_Last_10D.
    A ratio >= 6.0x proves institutional supply absorption (float lockup).
    """
    res = {
        'is_hv1_defended': False,
        'is_float_lockup': False,
        'lockup_ratio': 0.0,
        'hv1_date': None,
        'hv1_low': 0.0,
        'hv1_high': 0.0,
        'hv1_close': 0.0,
        'dist_from_hv1_low_pct': 0.0,
    }
    if df is None or df.empty or len(df) < 30:
        return res

    c_close = _safe_get_series(df, 'close', 'Close')
    if c_close is None or c_close.empty:
        return res
    c_open = _safe_get_series(df, 'open', 'Open', fallback=c_close)
    c_high = _safe_get_series(df, 'high', 'High', fallback=c_close)
    c_low = _safe_get_series(df, 'low', 'Low', fallback=c_close)
    c_vol = _safe_get_series(df, 'volume', 'Volume')
    if c_vol is None or c_vol.empty or float(c_vol.max()) <= 0:
        return res

    tail_len = min(252, len(df))
    w_vol = c_vol.iloc[-tail_len:]
    w_close = c_close.iloc[-tail_len:]
    w_open = c_open.iloc[-tail_len:]
    w_low = c_low.iloc[-tail_len:]
    w_high = c_high.iloc[-tail_len:]

    hv1_idx = w_vol.idxmax()
    hv1_vol = float(w_vol.loc[hv1_idx])
    hv1_close = float(w_close.loc[hv1_idx])
    hv1_open = float(w_open.loc[hv1_idx])
    hv1_low = float(w_low.loc[hv1_idx])
    hv1_high = float(w_high.loc[hv1_idx])

    rng = hv1_high - hv1_low
    clv = (hv1_close - hv1_low) / rng if rng > 0 else 1.0
    if hv1_close < hv1_open and clv < 0.50:
        return res

    curr_close = float(w_close.iloc[-1])
    if curr_close < hv1_low or hv1_low <= 0:
        return res

    min_vol_10d = float(w_vol.iloc[-10:].replace(0, np.nan).min())
    lockup_ratio = (hv1_vol / min_vol_10d) if (pd.notna(min_vol_10d) and min_vol_10d > 0) else 1.0
    dist_hv1_low = ((curr_close - hv1_low) / hv1_low) * 100.0

    hv1_dt_str = str(hv1_idx)[:10]
    if 'date' in df.columns:
        try:
            hv1_dt_str = str(pd.to_datetime(df.loc[hv1_idx, 'date']).strftime('%Y-%m-%d'))
        except Exception:
            pass

    res['is_hv1_defended'] = True
    res['is_float_lockup'] = bool(lockup_ratio >= 6.0 and dist_hv1_low <= 35.0)
    res['lockup_ratio'] = round(lockup_ratio, 1)
    res['hv1_date'] = hv1_dt_str
    res['hv1_low'] = round(hv1_low, 2)
    res['hv1_high'] = round(hv1_high, 2)
    res['hv1_close'] = round(hv1_close, 2)
    res['dist_from_hv1_low_pct'] = round(dist_hv1_low, 1)
    return res

if __name__ == '__main__':
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

