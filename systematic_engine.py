import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def calculate_returns_and_volatility(close_df):
    """
    Computes rolling returns and volatility across the matrix.
    close_df: pd.DataFrame where index is dates and columns are tickers.
    """
    # Calculate daily log returns
    log_returns = np.log(close_df / close_df.shift(1))
    
    # Calculate rolling 1-year annualized volatility (252 trading days)
    # Using min_periods=126 (6 months) as a fallback so we don't lose too much data
    volatility = log_returns.rolling(window=252, min_periods=126).std() * np.sqrt(252)
    
    # Calculate Rolling Returns
    # Using standard trading day windows
    r1 = (close_df / close_df.shift(21)) - 1   # 1 Month ~ 21 days
    r3 = (close_df / close_df.shift(63)) - 1   # 3 Months ~ 63 days
    r6 = (close_df / close_df.shift(126)) - 1  # 6 Months ~ 126 days
    
    return log_returns, volatility, r1, r3, r6

def _cross_sectional_z(df):
    """
    Calculates cross-sectional Z-Score row by row.
    z = (x - mean) / std
    """
    mean_val = df.mean(axis=1)
    std_val = df.std(axis=1)
    
    # Subtract mean across rows, divide by std across rows
    z_scores = df.sub(mean_val, axis=0).div(std_val, axis=0)
    return z_scores

def calculate_fast_momentum_scores(close_df):
    """
    Implements the custom faster momentum scoring for the live app.
    Combines computing volatility, returns, and z-score aggregations.
    
    Returns a dataframe of normalized scores, indexed exactly as close_df.
    """
    returns_log, vol, r1, r3, r6 = calculate_returns_and_volatility(close_df)
    
    # 1. Momentum Ratios (Return / Volatility)
    # If volatility is 0, replace to avoid Inf. Floor at 1%.
    safe_vol = vol.replace(0, np.nan).clip(lower=0.01)
    
    mr1 = r1 / safe_vol
    mr3 = r3 / safe_vol
    mr6 = r6 / safe_vol
    
    # 2. Cross-Sectional Z-Scores
    z1 = _cross_sectional_z(mr1)
    z3 = _cross_sectional_z(mr3)
    z6 = _cross_sectional_z(mr6)
    
    # 3. Weighted Z-Score 
    # (50% 1M, 30% 3M, 20% 6M)
    weighted_z = (z1 * 0.50) + (z3 * 0.30) + (z6 * 0.20)
    
    # 4. Normalized Momentum Score (NSE Formula)
    # If Z >= 0: 1 + Z
    # If Z < 0: 1 / (1 - Z)
    
    normalized_score = pd.DataFrame(index=weighted_z.index, columns=weighted_z.columns)
    
    z_numpy = weighted_z.values
    pos_mask = z_numpy >= 0
    neg_mask = z_numpy < 0
    
    out = np.full_like(z_numpy, np.nan)
    
    with np.errstate(invalid='ignore'):
        out[pos_mask] = 1 + z_numpy[pos_mask]
        out[neg_mask] = 1 / (1 - z_numpy[neg_mask])
        
    normalized_score.iloc[:, :] = out
    return normalized_score.astype(float)

def compute_live_fast_momentum_matrix():
    """
    Orchestrates the data loads from the cache and returns the scoring matrix.
    Ties deeply into the existing Kush Tracker app caching logic.
    """
    from market_data import fetch_nifty_total_market_tickers
    from price_history_manager import fetch_incremental_history
    
    # 1. Fetch Universe
    universe_tickers = fetch_nifty_total_market_tickers(show_progress=False)
    if not universe_tickers:
        return None, None
        
    # 2. Load History (Min 252 days to get 1Yr volatility properly + 6M prior window buffer)
    # 400 trading days is roughly 1.5 calendar years, plenty to compute current scores.
    history_df = fetch_incremental_history(universe_tickers, days=400)
    
    if history_df.empty:
        return None, None
        
    # 3. Clean and process close prices
    if isinstance(history_df.columns, pd.MultiIndex):
        if 'Close' in history_df.columns.get_level_values(1):
            close_df = history_df.xs('Close', level=1, axis=1)
        else:
            close_df = history_df
    else:
        close_df = history_df
        
    # Standardize column keys to match raw format without .NS for cleaner UI
    # We will keep .NS if it exists, let the UI clean it
    
    close_df = close_df.copy()
    close_df.ffill(inplace=True)
    
    # Reindex date if timezones exist
    if close_df.index.tz is not None:
        close_df.index = close_df.index.tz_localize(None)
    close_df.index = close_df.index.normalize()
    
    # 4. Generate absolute score matrix
    score_matrix = calculate_fast_momentum_scores(close_df)
    
    return score_matrix, close_df

def get_target_portfolio(date, score_matrix, close_df, n=15):
    """
    Extracts the Top N Equal Weight portfolio for a specific date.
    Returns:
        dict: ticker -> dictionary of {score, latest_price}
    """
    if date not in score_matrix.index:
        return {}
        
    scores_t = score_matrix.loc[date].dropna()
    top_n_stocks = scores_t.nlargest(n)
    
    portfolio = {}
    for ticker, score in top_n_stocks.items():
        price = close_df.loc[date, ticker] if ticker in close_df.columns else None
        portfolio[ticker] = {
            'Score': score,
            'Price': price,
            'Target_Weight': 1.0 / n
        }
        
    return portfolio

import streamlit as st

@st.cache_data(ttl=3600 * 24)
def get_momentum_badges():
    """
    Computes and caches the Fast Momentum outputs globally.
    Returns: (super_compounders_set, rockets_set)
    """
    score_matrix, close_df = compute_live_fast_momentum_matrix()
    if score_matrix is None or score_matrix.empty:
        return set(), set()
        
    dates = pd.Series(score_matrix.index, index=score_matrix.index)
    today = dates.iloc[-1]
    first_day_this_month = today.replace(day=1)
    
    past_dates = dates[dates < first_day_this_month]
    if not past_dates.empty:
        last_month_rebalance_date = past_dates.iloc[-1]
    else:
        last_month_rebalance_date = dates.iloc[0]
        
    is_month_end = dates.dt.month != dates.shift(-1).dt.month
    valid_historical_ends = dates[is_month_end][dates[is_month_end] <= last_month_rebalance_date].tail(6)
    
    hist_portfolios = {}
    for d in valid_historical_ends:
        hist_portfolios[d] = get_target_portfolio(d, score_matrix, close_df, n=50)
        
    hist_dates_sorted = sorted(hist_portfolios.keys(), reverse=True)
    
    super_compounders = set()
    if hist_dates_sorted:
        base_tickers = list(hist_portfolios[hist_dates_sorted[0]].keys())
        for ticker in base_tickers:
            count = sum(1 for hd in hist_dates_sorted if ticker in hist_portfolios[hd])
            if count >= 2:
                super_compounders.add(ticker.replace('.NS', ''))
                
    rocket_dict = {}
    scores_last = score_matrix.loc[last_month_rebalance_date].dropna()
    scores_today = score_matrix.loc[today].dropna()
    ranks_last = scores_last.rank(ascending=False, method='first')
    ranks_today = scores_today.rank(ascending=False, method='first')
    
    common_tickers = ranks_last.index.intersection(ranks_today.index)
    for t in common_tickers:
        velocity = int(ranks_last[t] - ranks_today[t])
        current_rank = int(ranks_today[t])
        if current_rank <= 50 and velocity > 0:
            rocket_dict[t] = velocity
            
    rockets = set()
    sorted_rockets = sorted(rocket_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    for t, _ in sorted_rockets:
        rockets.add(t.replace('.NS', ''))
            
    return super_compounders, rockets
