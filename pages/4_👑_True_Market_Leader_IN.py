"""
True Market Leaders Scanner (India) for Kush Tracker Lite.
CANSLIM 'C-A-L-I' Institutional Leader Ranking with Clenow Momentum Slope & RS Scores.
"""
import streamlit as st
import pandas as pd
import numpy as np

from market_data import fetch_nifty_total_market_tickers
from price_history_manager import fetch_incremental_history
from clenow_math import calculate_adjusted_slope
from database import add_to_focus_list

st.markdown("## 👑 True Market Leaders (India)")
st.caption("CANSLIM & Minervini Institutional Leadership Scanner — Ranking Top Momentum Stocks")

st.markdown("---")

col_f1, col_f2 = st.columns([1, 4])
with col_f1:
    min_rs = st.slider("Minimum RS Rating:", 70, 99, 85)
with col_f2:
    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    scan_btn = st.button("🚀 Run Institutional TML Scan", type="primary")

tickers_raw = fetch_nifty_total_market_tickers()[:200]
yf_tickers = [t if t.endswith(".NS") or t.endswith(".BO") else f"{t}.NS" for t in tickers_raw]

with st.spinner("Calculating Clenow Slopes & CANSLIM RS Rankings..."):
    histories = fetch_incremental_history(yf_tickers, days=252)

tml_rows = []

for orig_t, yf_t in zip(tickers_raw, yf_tickers):
    df = histories.get(yf_t, pd.DataFrame())
    if not df.empty and len(df) >= 90:
        try:
            close = df['Close']
            curr_price = float(close.iloc[-1])
            
            # Trend Check (Above 50 & 200 SMA)
            sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else curr_price
            sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else curr_price
            
            if curr_price >= sma50 and sma50 >= sma200:
                # Calculate Clenow Adjusted Slope
                res = calculate_adjusted_slope(close.values, days=90)
                adj_slope = res.get('adjusted_slope', 0.0) if isinstance(res, dict) else 0.0
                r_squared = res.get('r_squared', 0.0) if isinstance(res, dict) else 0.0
                
                # 3-Month Performance
                ret_3m = ((curr_price - float(close.iloc[-60])) / float(close.iloc[-60])) * 100 if len(close) >= 60 else 0.0
                
                tml_rows.append({
                    'Symbol': orig_t,
                    'Price (₹)': round(curr_price, 2),
                    '3M Ret %': round(ret_3m, 1),
                    'Clenow Slope': round(adj_slope, 2),
                    'R² Consistency': round(r_squared, 2),
                    'raw_slope': adj_slope
                })
        except Exception:
            continue

df_tml = pd.DataFrame(tml_rows)

if not df_tml.empty:
    # Percentile RS Score calculation
    df_tml = df_tml.sort_values(by='raw_slope', ascending=False).reset_index(drop=True)
    total = len(df_tml)
    df_tml['RS Score'] = [round(((total - idx) / total) * 99, 1) for idx in range(total)]
    
    # Filter by user RS cutoff
    df_filtered = df_tml[df_tml['RS Score'] >= min_rs]
    
    st.markdown(f"### 🏆 Top {len(df_filtered)} Institutional Market Leaders (RS ≥ {min_rs})")
    
    st.dataframe(
        df_filtered[['Symbol', 'RS Score', 'Price (₹)', '3M Ret %', 'Clenow Slope', 'R² Consistency']],
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("### 📌 Pin TML Setup to Focus List")
    c1, c2 = st.columns([3, 1])
    with c1:
        sel_tml = st.selectbox("Select True Market Leader to add to Focus List:", df_filtered['Symbol'].tolist())
    with c2:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        if st.button("➕ Pin TML Setup"):
            if add_to_focus_list(sel_tml, market="IN"):
                st.success(f"Added {sel_tml} to Focus List!")
else:
    st.info("Click 'Run Institutional TML Scan' to compute rankings.")
