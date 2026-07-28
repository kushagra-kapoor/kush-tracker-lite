"""
US True Market Leaders Scanner for Kush Tracker Lite.
CANSLIM & Minervini US Institutional Leadership Scanner.
"""
import streamlit as st
import pandas as pd
import numpy as np

from market_data import fetch_us_tickers
from price_history_manager import fetch_incremental_history
from clenow_math import calculate_adjusted_slope
from database import add_to_focus_list

st.markdown("## 🦅 US True Market Leaders")
st.caption("US Institutional Leaders — S&P 500 & NASDAQ High-Relative Strength Scans")

st.markdown("---")

col_f1, col_f2 = st.columns([1, 4])
with col_f1:
    min_rs = st.slider("US Min RS Rating:", 70, 99, 85)
with col_f2:
    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    scan_btn = st.button("🚀 Run US TML Scan", type="primary")

us_tickers = fetch_us_tickers()

with st.spinner("Calculating US Clenow Slopes & RS Rankings..."):
    histories = fetch_incremental_history(us_tickers, days=252)

tml_rows = []

for t in us_tickers:
    df = histories.get(t, pd.DataFrame())
    if not df.empty and len(df) >= 90:
        try:
            close = df['Close']
            curr_price = float(close.iloc[-1])
            
            sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else curr_price
            sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else curr_price
            
            if curr_price >= sma50 and sma50 >= sma200:
                res = calculate_adjusted_slope(close.values, days=90)
                adj_slope = res.get('adjusted_slope', 0.0) if isinstance(res, dict) else 0.0
                r_squared = res.get('r_squared', 0.0) if isinstance(res, dict) else 0.0
                
                ret_3m = ((curr_price - float(close.iloc[-60])) / float(close.iloc[-60])) * 100 if len(close) >= 60 else 0.0
                
                tml_rows.append({
                    'Symbol': t,
                    'Price ($)': round(curr_price, 2),
                    '3M Ret %': round(ret_3m, 1),
                    'Clenow Slope': round(adj_slope, 2),
                    'R² Consistency': round(r_squared, 2),
                    'raw_slope': adj_slope
                })
        except Exception:
            continue

df_tml = pd.DataFrame(tml_rows)

if not df_tml.empty:
    df_tml = df_tml.sort_values(by='raw_slope', ascending=False).reset_index(drop=True)
    total = len(df_tml)
    df_tml['RS Score'] = [round(((total - idx) / total) * 99, 1) for idx in range(total)]
    
    df_filtered = df_tml[df_tml['RS Score'] >= min_rs]
    
    st.markdown(f"### 🦅 Top {len(df_filtered)} US Institutional Leaders (RS ≥ {min_rs})")
    
    st.dataframe(
        df_filtered[['Symbol', 'RS Score', 'Price ($)', '3M Ret %', 'Clenow Slope', 'R² Consistency']],
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("### 📌 Pin US TML Setup to Focus List")
    c1, c2 = st.columns([3, 1])
    with c1:
        sel_tml = st.selectbox("Select US TML to add to Focus List:", df_filtered['Symbol'].tolist())
    with c2:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        if st.button("➕ Pin US TML Setup"):
            if add_to_focus_list(sel_tml, market="US"):
                st.success(f"Added {sel_tml} to Focus List!")
else:
    st.info("Click 'Run US TML Scan' to compute rankings.")
