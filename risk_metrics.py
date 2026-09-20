"""
Quantitative Risk and Performance Metrics for Kush Tracker Lite
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
