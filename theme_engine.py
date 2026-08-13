import pandas as pd
import numpy as np
from market_data import fetch_nifty_total_market_tickers
from price_history_manager import fetch_incremental_history
from signal_engine import calculate_weighted_rs_for_universe
import streamlit as st

@st.cache_data(ttl=3600*2) # Cache for 2 hours
def calculate_theme_alignment():
    """
    Calculates the Top Sectors based on Net New Highs over the last 5 days
    using the Nifty Total Market universe.
    
    Returns:
        dict: Containing top sectors, alignment scores, and sector leaders.
    """
    try:
        # 1. Fetch Universe & Sectors
        tickers, industry_map = fetch_nifty_total_market_tickers(show_progress=False, return_industry_map=True)
        if not tickers:
            return None
            
        # 2. Fetch Historical Data (260 trading days ~ 1 year)
        history_df = fetch_incremental_history(tickers, days=260)
        if history_df.empty:
            return None
            
        # Get RS scores to identify leaders later
        rs_scores = calculate_weighted_rs_for_universe(history_df, tickers)
        
        # 3. Calculate 52-Week Highs and Lows
        if 'Close' not in history_df.columns.get_level_values(1):
            return None
            
        close_df = history_df.xs('Close', level=1, axis=1)
        high_df = history_df.xs('High', level=1, axis=1) if 'High' in history_df.columns.get_level_values(1) else close_df
        low_df = history_df.xs('Low', level=1, axis=1) if 'Low' in history_df.columns.get_level_values(1) else close_df
        
        # Calculate rolling 250-day max and min (excluding the current day to see if today broke it)
        # We actually just want to see if the current day's high >= 250-day high.
        # To be safe, we look at the last 5 days.
        last_5_days_high = high_df.tail(5)
        last_5_days_low = low_df.tail(5)
        
        # Max of the previous 250 days (shifting by 1 to exclude the current day of evaluation)
        rolling_250_max = high_df.rolling(window=250, min_periods=100).max().shift(1).tail(5)
        rolling_250_min = low_df.rolling(window=250, min_periods=100).min().shift(1).tail(5)
        
        # Boolean dataframes: True if the day's high was >= 52W high
        is_new_high = last_5_days_high >= rolling_250_max
        is_new_low = last_5_days_low <= rolling_250_min
        
        # Sum over the last 5 days for each ticker
        new_highs_count = is_new_high.sum(axis=0)
        new_lows_count = is_new_low.sum(axis=0)
        
        # 4. Aggregate by Sector
        sector_data = []
        for ticker in close_df.columns:
            clean_ticker = ticker.replace('.NS', '')
            sector = industry_map.get(ticker, industry_map.get(clean_ticker, "Unknown"))
            
            # Map specific industries to broader themes if needed, but the CSV usually has good macro sectors
            if sector == "Unknown" or pd.isna(sector):
                continue
                
            sector_data.append({
                'Ticker': ticker,
                'CleanTicker': clean_ticker,
                'Sector': sector,
                'NewHighs': 1 if new_highs_count.get(ticker, 0) > 0 else 0,
                'NewLows': 1 if new_lows_count.get(ticker, 0) > 0 else 0,
                'RS': rs_scores.get(clean_ticker, rs_scores.get(ticker, 0))
            })
            
        theme_df = pd.DataFrame(sector_data)
        if theme_df.empty:
            return None
            
        # Group by sector
        sector_group = theme_df.groupby('Sector').agg(
            TotalStocks=('Ticker', 'count'),
            NewHighs=('NewHighs', 'sum'),
            NewLows=('NewLows', 'sum')
        ).reset_index()
        
        sector_group['NetNewHighs'] = sector_group['NewHighs'] - sector_group['NewLows']
        
        # Filter sectors with at least some activity, sort by Net New Highs
        top_sectors = sector_group[sector_group['NetNewHighs'] > 0].sort_values(by='NetNewHighs', ascending=False).head(6)
        
        if top_sectors.empty:
            return None
            
        # 5. Extract Leaders for Top Sectors
        theme_results = []
        for _, row in top_sectors.iterrows():
            sector_name = row['Sector']
            
            # Find the top 3 stocks in this sector that made a new high, sorted by RS
            sector_stocks = theme_df[(theme_df['Sector'] == sector_name) & (theme_df['NewHighs'] > 0)]
            leaders = sector_stocks.sort_values(by='RS', ascending=False).head(3)
            
            leader_list = []
            for _, l_row in leaders.iterrows():
                leader_list.append({
                    'ticker': l_row['CleanTicker'],
                    'rs': l_row['RS']
                })
                
            theme_results.append({
                'sector': sector_name,
                'net_highs': int(row['NetNewHighs']),
                'total_stocks': int(row['TotalStocks']),
                'leaders': leader_list
            })
            
        return {
            'top_themes': theme_results,
            'universe_size': len(theme_df)
        }
    except Exception as e:
        print(f"[!] Theme Engine Error: {e}")
        return None

def calculate_portfolio_alignment(theme_data, portfolio_decisions):
    """
    Cross-references the top themes with the current portfolio decisions 
    to calculate exposure percentage.
    """
    if not theme_data or not portfolio_decisions:
        return 0.0, {}
        
    from market_data import fetch_nifty_total_market_tickers
    _, industry_map = fetch_nifty_total_market_tickers(show_progress=False, return_industry_map=True)
        
    top_themes = [t['sector'] for t in theme_data['top_themes']]
    
    total_capital = 0.0
    aligned_capital = 0.0
    theme_exposure = {theme: 0.0 for theme in top_themes}
    
    for d in portfolio_decisions:
        loss_metrics = d.get('loss_metrics', {})
        weight = loss_metrics.get('portfolio_weight', 0)
        ticker = d.get('ticker', '')
        
        # Override the Yahoo Finance sector with the official NSE Macro Sector
        clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
        ns_ticker = f"{clean_ticker}.NS"
        sector = industry_map.get(ticker, industry_map.get(ns_ticker, d.get('sector', 'Unknown')))
        
        total_capital += weight
        
        # Fuzzy match or exact match
        for theme in top_themes:
            if sector.lower() == theme.lower() or sector.lower() in theme.lower() or theme.lower() in sector.lower():
                theme_exposure[theme] += weight
                aligned_capital += weight
                break # count once
                
    alignment_pct = aligned_capital if total_capital > 0 else 0.0
    
    return alignment_pct, theme_exposure
