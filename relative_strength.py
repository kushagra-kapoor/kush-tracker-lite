# Relative Strength Calculator for Kush Tracker
# Market-Normalized RS (vs NIFTY 500 Universe)

import pandas as pd
import numpy as np
from datetime import datetime
from config import RS_WEIGHTS, RS_THRESHOLDS, DATA_SETTINGS
from nifty500_universe import (
    compute_nifty500_rs_reference,
    get_rs_reference_distribution,
    get_percentile_rank,
    check_rs_reference_fresh
)


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
    
    if past_price == 0 or pd.isna(past_price):
        return None
    
    return (current_price / past_price) - 1


def calculate_stock_returns(df: pd.DataFrame) -> dict:
    """
    Calculate 1M, 3M, and 6M returns for a stock.
    
    Args:
        df: DataFrame with 'close' column
    
    Returns:
        Dictionary with R1, R3, R6 returns
    """
    if df.empty or 'close' not in df.columns:
        return {'R1': None, 'R3': None, 'R6': None}
    
    close = df['close']
    
    r1 = calculate_returns(close, DATA_SETTINGS['TRADING_DAYS_1M'])
    r3 = calculate_returns(close, DATA_SETTINGS['TRADING_DAYS_3M'])
    r6 = calculate_returns(close, DATA_SETTINGS['TRADING_DAYS_6M'])
    
    return {'R1': r1, 'R3': r3, 'R6': r6}


def calculate_relative_returns(stock_returns: dict, benchmark_returns: dict) -> dict:
    """
    Calculate relative returns (stock - benchmark).
    
    Args:
        stock_returns: Dictionary with R1, R3, R6
        benchmark_returns: Dictionary with R1, R3, R6 for benchmark
    
    Returns:
        Dictionary with RR1, RR3, RR6
    """
    rr1 = None
    rr3 = None
    rr6 = None
    
    if stock_returns['R1'] is not None and benchmark_returns['R1'] is not None:
        rr1 = stock_returns['R1'] - benchmark_returns['R1']
    
    if stock_returns['R3'] is not None and benchmark_returns['R3'] is not None:
        rr3 = stock_returns['R3'] - benchmark_returns['R3']
    
    if stock_returns['R6'] is not None and benchmark_returns['R6'] is not None:
        rr6 = stock_returns['R6'] - benchmark_returns['R6']
    
    return {'RR1': rr1, 'RR3': rr3, 'RR6': rr6}


def calculate_weighted_rs_raw(relative_returns: dict) -> float:
    """
    Calculate weighted RS raw score.
    
    Formula: RS_raw = (RR1 × 0.40) + (RR3 × 0.35) + (RR6 × 0.25)
    
    Args:
        relative_returns: Dictionary with RR1, RR3, RR6
    
    Returns:
        Raw RS score (not normalized)
    """
    rr1 = relative_returns.get('RR1')
    rr3 = relative_returns.get('RR3')
    rr6 = relative_returns.get('RR6')
    
    # Handle missing values
    if rr1 is None or rr3 is None or rr6 is None:
        # Use available values with adjusted weights
        available = []
        weights = []
        
        if rr1 is not None:
            available.append(rr1)
            weights.append(RS_WEIGHTS['1M'])
        if rr3 is not None:
            available.append(rr3)
            weights.append(RS_WEIGHTS['3M'])
        if rr6 is not None:
            available.append(rr6)
            weights.append(RS_WEIGHTS['6M'])
        
        if not available:
            return None
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        return sum(a * w for a, w in zip(available, normalized_weights))
    
    return (
        rr1 * RS_WEIGHTS['1M'] +
        rr3 * RS_WEIGHTS['3M'] +
        rr6 * RS_WEIGHTS['6M']
    )


def normalize_rs_against_nifty500(rs_raw: float) -> float:
    """
    Normalize RS_raw against NIFTY 500 universe distribution.
    
    This is the CORRECT normalization method:
    - Ranks the holding's RS_raw against all NIFTY 500 stocks
    - NOT against other holdings in portfolio
    
    Args:
        rs_raw: Raw RS score for a holding
    
    Returns:
        Normalized RS score (0-100) against market universe
    """
    if rs_raw is None:
        return None
    
    # Try today's NIFTY 500 RS distribution first
    today = datetime.now().strftime('%Y-%m-%d')
    distribution = get_rs_reference_distribution(today)
    
    # If today's data is missing (market holiday, stale cache), try the most recent date
    if not distribution or len(distribution) < 50:
        from nifty500_universe import get_connection
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT date FROM rs_reference_nifty500 
                ORDER BY date DESC LIMIT 1
            ''')
            row = cursor.fetchone()
            conn.close()
            if row:
                fallback_date = row[0]
                distribution = get_rs_reference_distribution(fallback_date)
                if distribution and len(distribution) >= 50:
                    print(f"  → Using RS reference from {fallback_date} (today's not available)")
        except Exception:
            pass
    
    if not distribution or len(distribution) < 50:
        print("  ! No usable NIFTY 500 RS reference data found in DB, using mathematical fallback")
        # Mathematical fallback scaling
        fallback_score = min(99.0, max(1.0, float(50 + (rs_raw * 60))))
        return fallback_score
    
    # Get percentile rank against NIFTY 500 universe
    rs_score = get_percentile_rank(rs_raw, distribution)
    
    return rs_score


def get_rs_rating(rs_score: float) -> str:
    """
    Get RS rating based on score.
    
    Args:
        rs_score: Normalized RS score (0-100) vs NIFTY 500
    
    Returns:
        Rating string
    """
    if rs_score is None:
        return 'Unknown'
    
    if rs_score >= RS_THRESHOLDS['ELITE']:
        return 'Elite Leader'
    elif rs_score >= RS_THRESHOLDS['VALID']:
        return 'Valid Leader'
    elif rs_score >= RS_THRESHOLDS['WEAKENING']:
        return 'Weakening'
    else:
        return 'Leadership Lost'


def calculate_all_rs_scores(market_data: dict) -> dict:
    """
    Calculate RS scores for all holdings.
    
    RS is normalized against NIFTY 500 universe (NOT within holdings).
    This ensures strong leaders don't show false RS deterioration
    just because the portfolio is already concentrated in leaders.
    
    Args:
        market_data: Dictionary mapping ticker to DataFrame
    
    Returns:
        Dictionary mapping ticker to RS data (score, rating, details)
    """
    print("[*] Calculating Relative Strength scores (Market-Normalized)...")
    
    # Step 1: Ensure NIFTY 500 reference data is fresh
    today = datetime.now().strftime('%Y-%m-%d')
    if not check_rs_reference_fresh(today):
        print("  → Computing NIFTY 500 RS reference (this may take 2-3 minutes)...")
        success = compute_nifty500_rs_reference(show_progress=True)
        if not success:
            print("  ! Failed to compute NIFTY 500 reference. RS scores may be unavailable.")
    else:
        print("  + Using cached NIFTY 500 RS reference")
    
    # Get benchmark returns
    benchmark_df = market_data.get('BENCHMARK')
    if benchmark_df is None or benchmark_df.empty:
        print("! No benchmark data available. RS scores may be inaccurate.")
        benchmark_returns = {'R1': 0, 'R3': 0, 'R6': 0}
    else:
        benchmark_returns = calculate_stock_returns(benchmark_df)
        if all(v is not None for v in benchmark_returns.values()):
            print(f"  Benchmark returns: 1M={benchmark_returns['R1']:.2%}, "
                  f"3M={benchmark_returns['R3']:.2%}, 6M={benchmark_returns['R6']:.2%}")
    
    # Calculate RS for each holding
    result = {}
    raw_scores = {}  # Collect rs_raw for fallback percentile if NIFTY 500 ref is unavailable
    
    for ticker, df in market_data.items():
        if ticker == 'BENCHMARK':
            continue
        
        # Step 2: Calculate stock returns
        stock_returns = calculate_stock_returns(df)
        
        # Step 3: Calculate relative returns vs benchmark
        relative_returns = calculate_relative_returns(stock_returns, benchmark_returns)
        
        # Step 4: Calculate weighted RS_raw
        rs_raw = calculate_weighted_rs_raw(relative_returns)
        
        # Step 5: Normalize against NIFTY 500 universe (NOT within holdings!)
        rs_score = normalize_rs_against_nifty500(rs_raw)
        
        if rs_raw is not None:
            raw_scores[ticker] = rs_raw
        
        result[ticker] = {
            'rs_score': rs_score,
            'rs_raw': rs_raw,
            'rs_rating': get_rs_rating(rs_score),
            'details': {
                'stock_returns': stock_returns,
                'relative_returns': relative_returns,
            },
        }
        
        if rs_score is not None:
            print(f"  {ticker}: RS={rs_score:.1f} ({get_rs_rating(rs_score)})")
        else:
            print(f"  {ticker}: RS=N/A (raw={rs_raw})")
    
    # Fallback: if ALL rs_scores are None (no NIFTY 500 reference at all),
    # compute a relative percentile within the portfolio itself so the chart is not blank
    all_none = all(v.get('rs_score') is None for v in result.values())
    if all_none and len(raw_scores) >= 2:
        print("  ⚠️ NIFTY 500 reference unavailable — using intra-portfolio RS ranking as fallback")
        sorted_raws = sorted(raw_scores.values())
        for ticker in result:
            raw = raw_scores.get(ticker)
            if raw is not None:
                count_below = sum(1 for v in sorted_raws if v < raw)
                fallback_score = round((count_below / len(sorted_raws)) * 100, 1)
                result[ticker]['rs_score'] = fallback_score
                result[ticker]['rs_rating'] = get_rs_rating(fallback_score)
                result[ticker]['rs_fallback'] = True
                print(f"  {ticker}: RS={fallback_score:.1f} (fallback intra-portfolio)")
    elif all_none and len(raw_scores) == 1:
        # Single holding — just set to 50 (neutral)
        for ticker in result:
            if raw_scores.get(ticker) is not None:
                result[ticker]['rs_score'] = 50.0
                result[ticker]['rs_rating'] = 'Unknown'
                result[ticker]['rs_fallback'] = True
    
    return result


def calculate_rs_blue_dot(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
    """
    Calculate the RS Blue Dot indicator.
    An RS Blue Dot occurs when the Relative Strength (RS) line reaches a new 52-week high
    before the stock's price does.
    
    Formula:
      RS_Line = Stock_Close / Benchmark_Close
      Condition 1: RS_Line == 52-week high (rolling max)
      Condition 2: Stock_Close < 52-week high (rolling max)
      
    Args:
        stock_df: Stock price DataFrame containing 'close' column
        benchmark_df: Benchmark price DataFrame containing 'close' column
        lookback_window: Rolling window size (252 trading days for 52-week high)
        
    Returns:
        Pandas Series of boolean values (True where RS Blue Dot is active)
    """
    if stock_df.empty or benchmark_df.empty:
        return pd.Series(False, index=stock_df.index if not stock_df.empty else [])
        
    # Standardize columns to lowercase for consistency
    s_df = stock_df.copy()
    s_df.columns = [c.lower() for c in s_df.columns]
    b_df = benchmark_df.copy()
    b_df.columns = [c.lower() for c in b_df.columns]
    
    if 'close' not in s_df.columns or 'close' not in b_df.columns:
        return pd.Series(False, index=stock_df.index)
        
    # Align both DataFrames on their indices
    aligned_df = pd.DataFrame({
        'stock_close': s_df['close'],
        'benchmark_close': b_df['close']
    }).dropna()
    
    if len(aligned_df) < 5:
        return pd.Series(False, index=stock_df.index)
        
    # Calculate RS Line
    rs_line = aligned_df['stock_close'] / aligned_df['benchmark_close']
    
    # Calculate rolling maximums
    rs_line_roll_max = rs_line.rolling(window=lookback_window, min_periods=min(10, len(rs_line))).max()
    price_roll_max = aligned_df['stock_close'].rolling(window=lookback_window, min_periods=min(10, len(aligned_df))).max()
    
    # Condition 1: RS Line is at a 52-week High
    rs_high = rs_line >= rs_line_roll_max
    
    # Condition 2: Price is NOT at a 52-week High
    price_not_high = aligned_df['stock_close'] < price_roll_max
    
    # Combine conditions
    blue_dot_aligned = rs_high & price_not_high
    
    # Reindex to match the original stock_df
    return blue_dot_aligned.reindex(stock_df.index, fill_value=False)


if __name__ == '__main__':
    print("Testing Market-Normalized Relative Strength calculator...")
    
    # Test percentile ranking
    sample_distribution = [-0.05, -0.02, 0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    test_value = 0.10
    
    from nifty500_universe import get_percentile_rank
    percentile = get_percentile_rank(test_value, sample_distribution)
    print(f"\nTest: Value {test_value} in distribution = {percentile}th percentile")
