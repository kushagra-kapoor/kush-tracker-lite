import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime
import time

# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import (
    get_all_fundamentals_cache, 
    auto_backfill_footprints, 
    log_volume_shock, 
    save_intraday_signals,
    get_connection
)
from market_data import fetch_yfinance_batch
from signal_engine import process_intraday_data
from llm_utils import robust_llm_call

def read_tickers(filename):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def process_market(tickers, market_label, index_ticker):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing {len(tickers)} {market_label} stocks...")
    if not tickers: return []
    
    db_cache = get_all_fundamentals_cache()
    ind_map = {t: db_cache.get(t, {}).get('industry', 'Unknown') for t in tickers}
    
    print(f"  Downloading YFinance Data...")
    history_df = fetch_yfinance_batch(tickers, days=252, force_today_refresh=True)
    index_df = yf.download(index_ticker, period="1y", progress=False)
    
    print(f"  Computing Intraday Metrics...")
    results_df, market_state = process_intraday_data(history_df, tickers, index_df, ind_map)
    
    # Auto-Backfill Footprints
    if len(tickers) == 1 or not isinstance(history_df.columns, pd.MultiIndex):
        ticker_data = {tickers[0]: history_df} if not history_df.empty else {}
    else:
        ticker_data = {t: history_df[t] for t in tickers if t in history_df.columns.get_level_values(0).unique()}
    
    auto_backfill_footprints(ticker_data, market_label, lookback_days=30)
    
    # Extract Signals
    signals_to_save = []
    apex_count = 0
    shock_count = 0
    
    for _, row in results_df.iterrows():
        t = str(row.get('Ticker', '')).replace('👑', '').replace('🔥', '').strip()
        if not t: continue
        
        if row.get('Is Elite Breakout', False): signals_to_save.append({'ticker': t, 'signal_name': 'Stage 2'})
        if row.get('Is Launchpad', False): signals_to_save.append({'ticker': t, 'signal_name': 'Launchpad'})
        if row.get('Is GLB', False): signals_to_save.append({'ticker': t, 'signal_name': 'GLB Breakout'})
        if row.get('Is 3WT', False): 
            signals_to_save.append({'ticker': t, 'signal_name': '3WT'})
            signals_to_save.append({'ticker': t, 'signal_name': 'Ryan'})
        if row.get('Is EP', False): signals_to_save.append({'ticker': t, 'signal_name': 'EP'})
        if row.get('Is HV1', False): signals_to_save.append({'ticker': t, 'signal_name': 'HV1'})
        if row.get('Is New High', False): signals_to_save.append({'ticker': t, 'signal_name': '52W High'})
        
        # Apex
        is_apex = (row.get('Dollar Volume (Cr)', 0) >= 20.0) and \
                  (row.get('ADR %', 0) >= 4.0) and \
                  (row.get('1M Return', 0) >= 40.0 or row.get('1M Ret Pctile', 0) >= 90.0 or row.get('3M Ret Pctile', 0) >= 90.0) and \
                  (not row.get('Is Extended', False)) and \
                  (row.get('Is EP', False) or row.get('Is Breakout', False) or row.get('Is Launchpad', False) or row.get('Is GLB', False))
        if is_apex: 
            signals_to_save.append({'ticker': t, 'signal_name': 'Apex'})
            apex_count += 1
            
        # Institutional Footprint (Volume Shock)
        vol_exp = row.get('Volume Expansion', 0)
        today_ret = row.get('Today %', 0)
        adtv_cr = row.get('Dollar Volume (Cr)', 0)
        if pd.notnull(vol_exp) and vol_exp >= 5.0 and pd.notnull(today_ret) and today_ret >= 4.8 and pd.notnull(adtv_cr) and adtv_cr >= 5.0:
            shock_date = datetime.now().strftime('%Y-%m-%d')
            log_volume_shock(t, shock_date, float(vol_exp), float(row.get('Close', 0)), float(row.get('Today High', 0)), float(row.get('Today Low', 0)), market_label)
            shock_count += 1
            
    if signals_to_save:
        save_intraday_signals(market_label, signals_to_save)
        
    print(f"  Saved {len(signals_to_save)} {market_label} signals ({apex_count} Apex, {shock_count} Volume Shocks).")
    
    # Return summary for AI
    return {
        "market": market_label,
        "apex_count": apex_count,
        "shock_count": shock_count,
        "index_pct": market_state.get('index_pct', 0.0),
        "total_signals": len(signals_to_save)
    }

def run_daily_monitor():
    print("--- STARTING DAILY MONITOR ---")
    
    nse_tickers = read_tickers('tickers.txt')
    us_tickers = read_tickers('tickers_us.txt')
    
    nse_summary = process_market(nse_tickers, "INDIA", "^CRSLDX")
    us_summary = process_market(us_tickers, "USA", "^IXIC")
    
    # Generate Nightly Briefing via LLM
    print("Generating AI Nightly Briefing...")
    prompt = f"""
    You are an elite CANSLIM/Minervini momentum trader. Write a 2-paragraph Nightly Battle Plan summarizing the day's market action based on this data:
    
    INDIA MARKET: 
    - Nifty 500 Performance: {nse_summary.get('index_pct', 0.0):.2f}%
    - Apex Setups Triggered (The absolute highest quality momentum breakouts): {nse_summary.get('apex_count', 0)}
    - Institutional Volume Shocks (Massive footprints > 500% average volume): {nse_summary.get('shock_count', 0)}
    - Total Quality Signals: {nse_summary.get('total_signals', 0)}
    
    US MARKET (Nasdaq):
    - Nasdaq Composite Performance: {us_summary.get('index_pct', 0.0):.2f}%
    - Apex Setups Triggered: {us_summary.get('apex_count', 0)}
    - Institutional Volume Shocks: {us_summary.get('shock_count', 0)}
    - Total Quality Signals: {us_summary.get('total_signals', 0)}
    
    INSTRUCTIONS:
    - Paragraph 1: Analyze the market tone (Is it a distribution day? Are institutions buying? Is it risk-on or risk-off based on the index % and volume shocks?).
    - Paragraph 2: Comment on the breakout health (Apex count) and give actionable advice for tomorrow (e.g., "Press winners", "Tighten stops", or "Cash is a position").
    - Be aggressive, professional, and concise. No fluff. Do not use generic financial jargon, use CANSLIM terminology.
    """
    
    try:
        ai_briefing = robust_llm_call(prompt)
    except Exception as e:
        print(f"AI generation failed: {e}")
        ai_briefing = "AI Briefing unavailable due to API error."
        
    if not ai_briefing:
        ai_briefing = "AI Briefing unavailable. Please check LLM API Keys."
        
    print(f"AI BRIEFING:\n{ai_briefing}")
    
    # Save to daily_journal table
    print("Saving to database...")
    date_str = datetime.now().strftime('%Y-%m-%d')
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO daily_journal (date, review, mistakes, lessons, next_day_plan, is_distribution_day)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (date_str, ai_briefing, "", "", "Automatically generated by Daily Monitor.", 0))
    conn.commit()
    conn.close()
    
    print("--- DAILY MONITOR COMPLETE ---")

if __name__ == "__main__":
    run_daily_monitor()
