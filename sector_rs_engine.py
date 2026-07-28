import yfinance as yf
import pandas as pd
import streamlit as st
import time

# Categorized for potential future use, but merged for processing
SECTORS = {
    # Core Sectors
    "IT": "^CNXIT",
    "Banking": "^NSEBANK",
    "PSU Banks": "^CNXPSUBANK",
    "FMCG": "^CNXFMCG",
    "Automobiles": "^CNXAUTO",
    "Pharmaceuticals": "^CNXPHARMA",
    "Metals": "^CNXMETAL",
    "Real Estate": "^CNXREALTY",
    "Energy": "^CNXENERGY",
    "Infrastructure": "^CNXINFRA",
    "Media": "^CNXMEDIA",
    "MNC": "^CNXMNC",
    "PSE": "^CNXPSE",
    
    # Emerging Themes & Smart Beta (Using ETFs for reliable history)
    "Manufacturing": "MAKEINDIA.NS",
    "Consumption": "CONSUMBEES.NS",
    "Mid & Smallcap": "MIDSMALL.NS",
    "Momentum Factor": "MOMENTUM50.NS",
    "Low Vol Factor": "LOWVOLIETF.NS",
    "Alpha Factor": "ALPHAETF.NS",
    "CPSE (Public Sector)": "CPSEETF.NS",
    "Gold": "GOLDBEES.NS",
    "Silver": "SILVERBEES.NS"
}

BENCHMARK = "^NSEI"  # Nifty 50

@st.cache_data(ttl=3600)
def fetch_sector_rs_data(lookback_months=3):
    """
    Fetches historical data for sectors and calculates their relative return vs Nifty 50.
    """
    period = f"{lookback_months}mo"
    
    # Download Benchmark Data
    try:
        bench_data = yf.download(BENCHMARK, period=period, progress=False)
        if not bench_data.empty and len(bench_data) >= 2:
            # Handle possible MultiIndex from yfinance
            if isinstance(bench_data.columns, pd.MultiIndex):
                close_series = bench_data['Close'].iloc[:, 0].dropna()
            else:
                close_series = bench_data['Close'].dropna()
                
            if len(close_series) >= 2:
                bench_start = float(close_series.iloc[0])
                bench_end = float(close_series.iloc[-1])
                bench_return = ((bench_end - bench_start) / bench_start) * 100
            else:
                bench_return = 0.0
        else:
            bench_return = 0.0
    except Exception as e:
        print(f"Failed benchmark calculation: {e}")
        bench_return = 0.0
        
    results = []
    
    # Calculate Sector Returns sequentially to avoid yf batch rate-limit bugs
    for name, ticker in SECTORS.items():
        try:
            time.sleep(0.5)  # Avoid Yahoo Finance Rate Limits
            sec_data = yf.download(ticker, period=period, progress=False)
            if sec_data.empty:
                continue
                
            if isinstance(sec_data.columns, pd.MultiIndex):
                close_series = sec_data['Close'].iloc[:, 0].dropna()
            else:
                close_series = sec_data['Close'].dropna()
                
            if len(close_series) >= 2:
                sec_start = float(close_series.iloc[0])
                sec_end = float(close_series.iloc[-1])
                abs_return = ((sec_end - sec_start) / sec_start) * 100
                rel_return = abs_return - bench_return
                
                results.append({
                    "Sector": name,
                    "Absolute Return": abs_return,
                    "Relative Return": rel_return
                })
        except Exception:
            continue
            
    df = pd.DataFrame(results)
    if not df.empty:
        # Sort by relative return (Leaders at the top)
        df = df.sort_values(by="Relative Return", ascending=False).reset_index(drop=True)
        
    return df, bench_return
