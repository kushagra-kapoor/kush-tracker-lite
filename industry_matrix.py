import pandas as pd
import numpy as np

def calculate_industry_leadership(history_df: pd.DataFrame, tickers: list, industry_map: dict, rs_scores: dict) -> pd.DataFrame:
    """
    Ranks Industry Groups based on the average Relative Strength of their constituents.
    Takes O'Neil's top-down approach: leading stocks command leading groups.
    
    Args:
        history_df: The daily multi-index OHLCV dataframe.
        tickers: List of all fetched tickers.
        industry_map: Dictionary mapping 'TICKER.NS' to 'Industry Name'.
        rs_scores: Dictionary mapping 'TICKER.NS' to its Weighted RS score.
        
    Returns:
        DataFrame ranking the top industries.
    """
    industry_data = []

    # Map tickers to their industry and RS score
    for t in tickers:
        clean_t = t.replace('.NS', '')
        # RS scores dict uses clean tickers or full tickers depending on where it's called from
        rs = rs_scores.get(clean_t) or rs_scores.get(t)
        industry = industry_map.get(t, "Unknown")
        
        if rs is not None and industry != "Unknown" and pd.notna(industry):
            industry_data.append({
                'Ticker': t,
                'Industry': industry,
                'RS_Score': rs
            })
            
    if not industry_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(industry_data)
    
    # Calculate metrics per industry
    group_stats = df.groupby('Industry').agg(
        Constituent_Count=('Ticker', 'count'),
        Avg_RS=('RS_Score', 'mean'),
        Max_RS=('RS_Score', 'max'),
        Leaders_80_Plus=('RS_Score', lambda x: (x >= 80).sum())
    ).reset_index()
    
    # Filter out statistically insignificant groups (e.g. groups with only 1-2 stocks)
    group_stats = group_stats[group_stats['Constituent_Count'] >= 3]
    
    # Minervini / O'Neil Methodology: Pure Average RS can be skewed by a single massive penny stock outperformer.
    # We sort fundamentally by Breadth Concentration: How many True Leaders (RS > 80) does the sector have? 
    # Average RS is used strictly as a tie-breaker.
    group_stats = group_stats.sort_values(by=['Leaders_80_Plus', 'Avg_RS'], ascending=[False, False]).reset_index(drop=True)
    
    # Add Breadth Participation metric (What % of the sector is breaking out?)
    group_stats['Participation_%'] = (group_stats['Leaders_80_Plus'] / group_stats['Constituent_Count']) * 100.0
    group_stats['Participation_%'] = group_stats['Participation_%'].round(1)
    
    # Add a rank column
    group_stats.index += 1
    group_stats['Rank'] = group_stats.index
    
    return group_stats

def calculate_industry_strength_cycle(history_df: pd.DataFrame, tickers: list, industry_map: dict, weight_mode: str = "Equal", db_cache: dict = None) -> pd.DataFrame:
    """
    Constructs an institutional strength cycle matrix similar to StockScans.
    Computes purely vectorized historical performance to yield a 1-Month trend, 
    Current Score, and Outperforming/Accumulating classifications.
    
    Args:
        history_df: The daily multi-index OHLCV dataframe.
        tickers: List of all fetched tickers.
        industry_map: Dictionary mapping 'TICKER.NS' to 'Industry Name'.
        
    Returns:
        DataFrame ranking industries with Cycle Status and 21-day sparkline arrays.
    """
    if history_df.columns.nlevels < 2:
        return pd.DataFrame()
        
    try:
        close_df = history_df.xs('Close', level=1, axis=1)
    except:
        try:
            close_df = history_df.xs('Close', level=0, axis=1)
        except:
            return pd.DataFrame()
    
    # Forward fill to prevent dropouts
    close_df = close_df.ffill()
    
    # Keep only common tickers
    valid_tickers = [t for t in tickers if t in close_df.columns]
    close_df = close_df[valid_tickers]
    
    # Create reverse map
    rev_map = {}
    for t in valid_tickers:
        ind = industry_map.get(t, "Unknown")
        if ind != "Unknown" and pd.notna(ind):
            if ind not in rev_map:
                rev_map[ind] = []
            rev_map[ind].append(t)
            
    results = []
    
    # Calculate a synthetic index for each sector over the last ~126 days (6 months)
    # We need 21 days of "Score" history.
    # Score = 40% 1M + 35% 3M + 25% 6M scaled to 100.
    # But since we might only have 100 days of history from the cash, let's just use 1W and 1M composite.
    
    for industry, ind_tickers in rev_map.items():
        if len(ind_tickers) < 3: # Ignore insignificant sectors
            continue
            
        ind_closes = close_df[ind_tickers]
        
        daily_returns = ind_closes.pct_change().fillna(0)
        
        if weight_mode == "Market Cap" and db_cache:
            weights = []
            for t in ind_tickers:
                clean_t = t.replace('.NS', '')
                mc = db_cache.get(clean_t, {}).get('market_cap', 0)
                if not mc: mc = db_cache.get(t, {}).get('market_cap', 0)
                weights.append(max(float(mc), 1.0))
            
            total_cap = sum(weights)
            weights_arr = np.array(weights) / total_cap
            weighted_returns = (daily_returns * weights_arr).sum(axis=1)
            sector_index = (1 + weighted_returns).cumprod() * 100
        else:
            # Calculate equal-weighted daily return of the sector
            sector_index = (1 + daily_returns.mean(axis=1)).cumprod() * 100
        
        # We need historical scores. Let's calculate a rolling momentum score over the last 21 days
        # We will use purely the sector's own index to compute its strength vs a theoretical flatline.
        # We use the standard 1M (40%), 3M (35%), 6M (25%) momentum weights.
        
        roll_126 = sector_index.pct_change(periods=126).fillna(0)
        roll_63  = sector_index.pct_change(periods=63).fillna(0)
        roll_21  = sector_index.pct_change(periods=21).fillna(0)
        
        # Composite raw intermediate score for every day
        raw_score = (roll_21 * 0.40) + (roll_63 * 0.35) + (roll_126 * 0.25)
        
        # Grab the last 21 days of synthetic scores (to measure 1M velocity)
        last_21_raw = raw_score.tail(21)
        
        results.append({
            'Industry': industry,
            'Constituents': len(ind_tickers),
            'raw_scores_21': last_21_raw.values,
            'latest_index_val': sector_index.iloc[-1]
        })
        
    if not results:
        return pd.DataFrame()
        
    df = pd.DataFrame(results)
    
    # Cross-sectionally normalize the raw scores for EACH of the 21 days into a 0-99 percentile score
    # This precisely perfectly matches the "Average RS" and gives a relative "Score"
    arr_21 = np.vstack(df['raw_scores_21'].values) # Shape: (Num_Sectors, 21)
    
    # Rank along columns (for each day, rank the sectors)
    ranks_21 = pd.DataFrame(arr_21).rank(pct=True, axis=0) * 99.0
    
    df['Scores_Array'] = ranks_21.values.tolist()
    
    # Current Score is the last element
    df['Score'] = df['Scores_Array'].apply(lambda x: round(x[-1], 2))
    df['1M_Ago_Score'] = df['Scores_Array'].apply(lambda x: round(x[0], 2))
    df['Score_Change_1M'] = df['Score'] - df['1M_Ago_Score']
    
    def classify_status(row):
        score = row['Score']
        change = row['Score_Change_1M']
        
        if score >= 60.0 and change > 5.0:
            return "OUTPERFORMING"
        elif score >= 60.0 and change <= 5.0:
            return "CONSOLIDATING"
        elif score < 60.0 and change > 10.0:
            return "ACCUMULATING"
        else:
            return "UNDERPERFORMING"
            
    df['Status'] = df.apply(classify_status, axis=1)
    
    # Format changes
    df['Score'] = df['Score'].round(2)
    df['Score_Change_1M'] = df['Score_Change_1M'].round(2)
    
    # Sort by highest current score
    df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df.index += 1
    df['Rank'] = df.index
    
    return df
    
def calculate_constituent_strength_cycle(history_df, industry_tickers=None):
    """
    Computes individual stock momentum cycle statuses and sparklines.
    Ranks them cross-sectionally against the ENTIRE universe before returning just the requested industry.
    """
    if history_df.columns.nlevels < 2:
        return pd.DataFrame()
        
    try:
        close_df = history_df.xs('Close', level=1, axis=1)
    except:
        try:
            close_df = history_df.xs('Close', level=0, axis=1)
        except:
            return pd.DataFrame()
            
    close_df = close_df.ffill().fillna(0)
    
    # Calculate for the entire universe simultaneously to get global percentiles
    roll_126 = close_df.pct_change(periods=126).fillna(0)
    roll_63  = close_df.pct_change(periods=63).fillna(0)
    roll_21  = close_df.pct_change(periods=21).fillna(0)
    
    raw_scores = (roll_21 * 0.40) + (roll_63 * 0.35) + (roll_126 * 0.25)
    last_21_raw = raw_scores.tail(21)
    
    # Rank cross-sectionally for every day (row) across all columns (stocks)
    ranks_21 = last_21_raw.rank(pct=True, axis=1) * 99.0
    
    # Isolate tickers that belong to the requested industry, or keep all if None
    if industry_tickers is not None and len(industry_tickers) > 0:
        valid_tickers = [t for t in industry_tickers if t in ranks_21.columns]
        ranks_21 = ranks_21[valid_tickers]
    
    if ranks_21.empty:
        return pd.DataFrame()
        
    # Transpose so tickers are rows, dates are columns
    ranks_t = ranks_21.T
    
    df = pd.DataFrame(index=ranks_t.index)
    df['Scores_Array'] = ranks_t.values.tolist()
    # Ensure they are standard lists of floats
    df['Score'] = df['Scores_Array'].apply(lambda x: round(float(x[-1]), 2))
    df['1M_Ago_Score'] = df['Scores_Array'].apply(lambda x: round(float(x[0]), 2))
    df['Score_Change_1M'] = df['Score'] - df['1M_Ago_Score']
    
    def classify_status(row):
        score = row['Score']
        change = row['Score_Change_1M']
        if score >= 60.0 and change > 5.0: return "OUTPERFORMING"
        elif score >= 60.0 and change <= 5.0: return "CONSOLIDATING"
        elif score < 60.0 and change > 10.0: return "ACCUMULATING"
        else: return "UNDERPERFORMING"
        
    df['Status'] = df.apply(classify_status, axis=1)
    df['Score'] = df['Score'].round(2)
    df['Score_Change_1M'] = df['Score_Change_1M'].round(2)
    
    df = df.reset_index()
    # Rename the multi-index leftover if it exists
    if 'Symbols' in df.columns:
        df = df.rename(columns={'Symbols': 'Ticker'})
    elif 'index' in df.columns:
        df = df.rename(columns={'index': 'Ticker'})
        
    return df.sort_values(by='Score', ascending=False).reset_index(drop=True)

def get_sector_heat_rankings_data(history_df: pd.DataFrame, tickers: list, industry_map: dict, rs_scores: dict, weight_mode: str = "Equal", db_cache: dict = None) -> pd.DataFrame:
    """
    Generates the comprehensive data payload for the Sector Heat Rankings CSS grid.
    Combines RS Rating, 1M Velocity, Apex Predators, and Stage 2 Breadth.
    """
    # 1. Get cycle status and 1M velocity
    cycle_df = calculate_industry_strength_cycle(history_df, tickers, industry_map, weight_mode, db_cache)
    if cycle_df.empty:
        return pd.DataFrame()
        
    # 2. Get Breadth and base RS
    leadership_df = calculate_industry_leadership(history_df, tickers, industry_map, rs_scores)
    
    if leadership_df.empty:
        return pd.DataFrame()
        
    # 3. Merge them together
    merged = pd.merge(cycle_df, leadership_df[['Industry', 'Leaders_80_Plus', 'Participation_%', 'Constituent_Count']], on='Industry', how='inner')
    
    # 4. Find Apex Predators (Top 2 stocks by RS per industry)
    apex_map = {}
    
    # Group valid tickers by industry
    ind_to_tickers = {}
    for t in tickers:
        clean_t = t.replace('.NS', '')
        rs = rs_scores.get(clean_t) or rs_scores.get(t)
        if rs is None: continue
            
        ind = industry_map.get(t, "Unknown")
        if ind == "Unknown" or not pd.notna(ind): continue
            
        if ind not in ind_to_tickers: ind_to_tickers[ind] = []
        ind_to_tickers[ind].append({'ticker': clean_t, 'rs': rs})
        
    for ind, t_list in ind_to_tickers.items():
        # Sort by RS descending
        sorted_t = sorted(t_list, key=lambda x: x['rs'], reverse=True)
        top_2 = [x['ticker'] for x in sorted_t[:2]]
        apex_map[ind] = ", ".join(top_2)
        
    merged['Apex_Predators'] = merged['Industry'].map(apex_map)
    
    # Re-sort entirely by current RS Score (O'Neil methodology)
    merged = merged.sort_values(by='Score', ascending=False).reset_index(drop=True)
    merged.index += 1
    merged['Rank'] = merged.index
    
    return merged

