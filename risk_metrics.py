"""
Quantitative Risk and Performance Metrics for Kush Tracker
Implements downside risk metrics:
- Downside Deviation (Semi-deviation)
- Sortino Ratio (Dual-horizon: 3M Tactical & 6M Structural)
- Calmar Ratio (CAGR / Maximum Drawdown)
- Rolling Equity Curve Risk Metrics
"""

import numpy as np
import pandas as pd

def calculate_downside_deviation(
    prices_or_returns,
    window: int = 126,
    mar: float = 0.0,
    annualize: bool = True,
    is_returns: bool = False
) -> float:
    """
    Calculates downside deviation (semi-deviation below Minimum Acceptable Return MAR).
    
    Args:
        prices_or_returns: pd.Series, list, or np.ndarray of prices or daily returns.
        window: Lookback window (e.g. 63 for 3M, 126 for 6M).
        mar: Minimum Acceptable Return (daily). Default 0.0 (losses only).
        annualize: If True, scales daily downside deviation by sqrt(252).
        is_returns: If True, input is already daily returns. If False, converts price series to returns.
        
    Returns:
        float: Downside deviation.
    """
    if prices_or_returns is None:
        return 0.0
        
    if isinstance(prices_or_returns, pd.Series):
        arr = prices_or_returns.dropna().values
    else:
        arr = np.array(prices_or_returns, dtype=float)
        arr = arr[~np.isnan(arr)]
        
    if len(arr) < 2:
        return 0.0
        
    if not is_returns:
        # Input is prices, take tail(window + 1)
        if len(arr) > window + 1:
            arr = arr[-(window + 1):]
        # Calculate daily percentage returns
        rets = np.diff(arr) / arr[:-1]
    else:
        if len(arr) > window:
            rets = arr[-window:]
        else:
            rets = arr
            
    if len(rets) < 5:
        return 0.0
        
    # Underperformance below MAR
    downside_diff = np.minimum(0.0, rets - mar)
    sum_sq = np.sum(downside_diff ** 2)
    downside_dev = np.sqrt(sum_sq / len(rets))
    
    if annualize:
        downside_dev *= np.sqrt(252)
        
    return float(downside_dev)

def calculate_sortino_ratio(
    prices_or_returns,
    window: int = 126,
    mar_annual: float = 0.065,
    is_returns: bool = False
) -> float:
    """
    Calculates the annualized Sortino Ratio.
    
    Formula:
        Sortino = (Annualized Return - Annual MAR) / Annualized Downside Deviation
        
    Args:
        prices_or_returns: Price series or daily returns.
        window: Lookback period (e.g. 63 for 3M, 126 for 6M).
        mar_annual: Annualized Minimum Acceptable Return / Risk-free hurdle (default 6.5% = 0.065).
        is_returns: True if input is returns, False if prices.
        
    Returns:
        float: Annualized Sortino Ratio.
    """
    if prices_or_returns is None:
        return 0.0
        
    if isinstance(prices_or_returns, pd.Series):
        arr = prices_or_returns.dropna().values
    else:
        arr = np.array(prices_or_returns, dtype=float)
        arr = arr[~np.isnan(arr)]
        
    if len(arr) < 10:
        return 0.0
        
    if not is_returns:
        if len(arr) > window + 1:
            arr = arr[-(window + 1):]
        if len(arr) < 10 or arr[0] <= 0:
            return 0.0
            
        n_days = len(arr) - 1
        # Calculate daily percentage returns
        rets = np.diff(arr) / arr[:-1]
        daily_mar = mar_annual / 252.0
        # Standard arithmetic annualization (industry standard for daily asset returns)
        ann_ret = float(np.mean(rets)) * 252.0
        downside_dev = calculate_downside_deviation(arr, window=window, mar=daily_mar, annualize=True, is_returns=False)
    else:
        if len(arr) > window:
            arr = arr[-window:]
        n_days = len(arr)
        daily_mar = mar_annual / 252.0
        ann_ret = float(np.mean(arr)) * 252.0
        downside_dev = calculate_downside_deviation(arr, window=window, mar=daily_mar, annualize=True, is_returns=True)
        
    excess_ret = ann_ret - mar_annual
    
    # Handle zero or near-zero downside deviation (e.g. continuous up days)
    if downside_dev < 1e-4:
        if excess_ret > 0:
            return 20.0  # Cap cleanly at exceptional institutional ceiling
        else:
            return 0.0
            
    sortino = excess_ret / downside_dev
    
    # Bound between -10.0 and +20.0 for clean display and numerical sanity
    return float(np.clip(sortino, -10.0, 20.0))

def calculate_calmar_ratio(
    cagr_pct: float,
    max_drawdown_pct: float
) -> float:
    """
    Calculates the Calmar Ratio: CAGR (%) / Max Drawdown (%).
    
    Args:
        cagr_pct: Annualized growth rate as a percentage (e.g. 25.0 for 25%).
        max_drawdown_pct: Peak-to-trough max drawdown as a positive percentage (e.g. 10.0 for 10%).
        
    Returns:
        float: Calmar Ratio.
    """
    dd = abs(float(max_drawdown_pct))
    if dd < 0.5:
        # Under 0.5% drawdown, avoid division by zero
        return 10.0 if cagr_pct > 0 else 0.0
        
    calmar = float(cagr_pct) / dd
    return float(np.clip(calmar, -10.0, 15.0))

def calculate_rolling_portfolio_metrics(
    equity_df: pd.DataFrame,
    mar_annual: float = 0.065
) -> dict:
    """
    Calculates rolling Sortino (63D, 126D) and Calmar ratio from an equity curve DataFrame.
    Expected columns: 'Date', 'TotalValue'.
    """
    if equity_df is None or equity_df.empty or len(equity_df) < 5:
        return {
            'sortino_63d': 0.0,
            'sortino_126d': 0.0,
            'calmar': 0.0,
            'downside_dev_63d': 0.0,
            'cagr': 0.0,
            'max_drawdown': 0.0
        }
        
    vals = equity_df['TotalValue'].dropna().values
    if len(vals) < 5:
        return {
            'sortino_63d': 0.0,
            'sortino_126d': 0.0,
            'calmar': 0.0,
            'downside_dev_63d': 0.0,
            'cagr': 0.0,
            'max_drawdown': 0.0
        }
        
    # Calculate daily returns
    daily_rets = np.diff(vals) / vals[:-1]
    
    # 63D (3-Month) and 126D (6-Month) Sortino
    sortino_63d = calculate_sortino_ratio(daily_rets, window=min(63, len(daily_rets)), mar_annual=mar_annual, is_returns=True)
    sortino_126d = calculate_sortino_ratio(daily_rets, window=min(126, len(daily_rets)), mar_annual=mar_annual, is_returns=True)
    downside_dev_63d = calculate_downside_deviation(daily_rets, window=min(63, len(daily_rets)), mar=mar_annual/252.0, annualize=True, is_returns=True)
    
    # Maximum Drawdown calculation across the entire log
    running_max = np.maximum.accumulate(vals)
    dds = (vals - running_max) / running_max * 100.0
    max_dd = float(abs(np.min(dds))) if len(dds) > 0 else 0.0
    
    # Lifetime CAGR
    n_days = len(vals)
    total_ret = (vals[-1] / vals[0]) - 1.0 if vals[0] > 0 else 0.0
    if n_days > 1:
        if total_ret > 0:
            cagr = (((1.0 + total_ret) ** (252.0 / n_days)) - 1.0) * 100.0
        else:
            cagr = -(((1.0 + abs(total_ret)) ** (252.0 / n_days)) - 1.0) * 100.0
    else:
        cagr = 0.0
        
    calmar = calculate_calmar_ratio(cagr, max_dd)
    
    return {
        'sortino_63d': round(sortino_63d, 2),
        'sortino_126d': round(sortino_126d, 2),
        'calmar': round(calmar, 2),
        'downside_dev_63d': round(downside_dev_63d * 100.0, 1),
        'cagr': round(cagr, 1),
        'max_drawdown': round(max_dd, 1)
    }


def calculate_smooth_rs_atr(
    closes,
    highs=None,
    lows=None,
    atr_window: int = 20
) -> float:
    """
    Calculates Smooth RS (Composite RS % / 20D ATR%), measuring volatility-efficiency
    and tight trend progression relative to daily True Range expansion.

    Formula:
        Numerator (Composite RS %):
            (0.40 * R_21d + 0.35 * R_63d + 0.25 * R_126d) * 100
            (graceful fallback for shorter histories < 126 bars)
        Denominator (20D ATR%):
            Mean_20(TrueRange) / Close_t * 100
            (falls back to 20D mean absolute return % * 1.25 if highs/lows are omitted)
            Floored at 1.0% to prevent division-by-zero blowups on halted/illiquid bars.
        Smooth RS = clip(Composite RS % / max(ATR_20%, 1.0%), -10.0, 40.0)

    Args:
        closes: pd.Series, list, or np.ndarray of closing prices.
        highs: Optional pd.Series, list, or np.ndarray of high prices.
        lows: Optional pd.Series, list, or np.ndarray of low prices.
        atr_window: Lookback window for ATR% (default 20 trading days).

    Returns:
        float: Smooth RS ratio (Composite RS % / 20D ATR%), rounded to 2 decimals.
    """
    if closes is None:
        return 0.0

    try:
        if isinstance(closes, pd.Series):
            c_arr = closes.dropna().values.astype(float)
        else:
            c_arr = np.array(closes, dtype=float)
            c_arr = c_arr[~np.isnan(c_arr)]

        n = len(c_arr)
        if n < 22 or c_arr[-1] <= 0:
            return 0.0

        cur_c = float(c_arr[-1])

        # 1. Multi-Horizon Composite Momentum (%)
        r21 = (cur_c / float(c_arr[-22])) - 1.0 if n >= 22 and c_arr[-22] > 0 else 0.0
        if n >= 127 and c_arr[-64] > 0 and c_arr[-127] > 0:
            r63 = (cur_c / float(c_arr[-64])) - 1.0
            r126 = (cur_c / float(c_arr[-127])) - 1.0
            comp_rs_pct = (0.40 * r21 + 0.35 * r63 + 0.25 * r126) * 100.0
        elif n >= 64 and c_arr[-64] > 0:
            r63 = (cur_c / float(c_arr[-64])) - 1.0
            comp_rs_pct = (0.55 * r21 + 0.45 * r63) * 100.0
        else:
            comp_rs_pct = r21 * 100.0

        # 2. 20D ATR% (True Range % of current close)
        atr_pct = None
        if highs is not None and lows is not None:
            try:
                if isinstance(closes, pd.Series) and isinstance(highs, pd.Series) and isinstance(lows, pd.Series):
                    df_hlc = pd.DataFrame({'c': closes, 'h': highs, 'l': lows}).dropna().tail(atr_window + 2)
                    if len(df_hlc) >= 5:
                        h_v = df_hlc['h'].values.astype(float)
                        l_v = df_hlc['l'].values.astype(float)
                        c_v = df_hlc['c'].values.astype(float)
                        prev_c = c_v[:-1]
                        tr = np.maximum(
                            h_v[1:] - l_v[1:],
                            np.maximum(np.abs(h_v[1:] - prev_c), np.abs(l_v[1:] - prev_c))
                        )
                        atr_val = float(np.mean(tr[-atr_window:]))
                        if c_v[-1] > 0 and atr_val > 0:
                            atr_pct = (atr_val / float(c_v[-1])) * 100.0
                else:
                    h_arr = np.array(highs, dtype=float)
                    l_arr = np.array(lows, dtype=float)
                    m_len = min(len(c_arr), len(h_arr), len(l_arr))
                    if m_len >= 5:
                        w = min(m_len, atr_window + 2)
                        c_v = c_arr[-w:]
                        h_v = h_arr[-w:]
                        l_v = l_arr[-w:]
                        valid = ~(np.isnan(c_v) | np.isnan(h_v) | np.isnan(l_v))
                        c_v, h_v, l_v = c_v[valid], h_v[valid], l_v[valid]
                        if len(c_v) >= 5:
                            prev_c = c_v[:-1]
                            tr = np.maximum(
                                h_v[1:] - l_v[1:],
                                np.maximum(np.abs(h_v[1:] - prev_c), np.abs(l_v[1:] - prev_c))
                            )
                            atr_val = float(np.mean(tr[-atr_window:]))
                            if c_v[-1] > 0 and atr_val > 0:
                                atr_pct = (atr_val / float(c_v[-1])) * 100.0
            except Exception:
                atr_pct = None

        # Fallback if High/Low not provided: 20D close-to-close mean absolute % change * 1.25
        if atr_pct is None or np.isnan(atr_pct) or atr_pct <= 0:
            tail_c = c_arr[-(atr_window + 1):]
            if len(tail_c) >= 5:
                abs_rets = np.abs(np.diff(tail_c) / tail_c[:-1]) * 100.0
                atr_pct = float(np.mean(abs_rets)) * 1.25
            else:
                atr_pct = 2.5

        atr_pct_safe = max(float(atr_pct), 1.0)
        smooth_rs = comp_rs_pct / atr_pct_safe
        return round(float(np.clip(smooth_rs, -10.0, 40.0)), 2)
    except Exception:
        return 0.0

