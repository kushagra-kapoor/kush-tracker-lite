"""
US Intraday Monitor for Kush Tracker Lite.
Scans US Equities & Sector ETFs for Volume Surges & Breakouts.
Auto-logs Volume Shock Breakouts to Turso DB and provides 1-click Pin to Focus List.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from market_data import fetch_us_tickers
from price_history_manager import fetch_incremental_history
from database import log_volume_shock, add_to_focus_list

st.markdown("## 🇺🇸 US Intraday Monitor")
st.caption("Live Intelligence & Volume Surges for US Equities (NYSE / NASDAQ)")

st.markdown("---")

col_ctrl1, col_ctrl2 = st.columns([1, 4])
with col_ctrl1:
    force_refresh = st.button("🔄 Refresh US Data", type="primary")

us_tickers = fetch_us_tickers()

with st.spinner("Fetching live US market data..."):
    histories = fetch_incremental_history(us_tickers, days=60, force_today_refresh=force_refresh)

scan_rows = []
vol_shocks_count = 0

for t in us_tickers:
    df = histories.get(t, pd.DataFrame())
    if not df.empty and len(df) >= 20:
        try:
            close = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2])
            high = float(df['High'].iloc[-1])
            low = float(df['Low'].iloc[-1])
            vol = float(df['Volume'].iloc[-1])
            avg_vol = float(df['Volume'].iloc[-20:-1].mean())
            
            pct_change = ((close - prev_close) / prev_close) * 100
            vol_mult = (vol / avg_vol) if avg_vol > 0 else 1.0
            
            is_shock = (vol_mult >= 2.0 and pct_change >= 1.5)
            if is_shock:
                vol_shocks_count += 1
                today_str = datetime.now().strftime("%Y-%m-%d")
                log_volume_shock(t, today_str, vol_mult, close, high, low, market="US")
                
            scan_rows.append({
                'Symbol': t,
                'Price ($)': round(close, 2),
                'Change %': round(pct_change, 2),
                'Vol Multiple': f"{round(vol_mult, 1)}x",
                'Shock Status': '🔥 VOLUME SHOCK' if is_shock else 'Normal',
                'raw_vol_mult': vol_mult
            })
        except Exception:
            continue

df_scan = pd.DataFrame(scan_rows)

# Glassmorphic Metric Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
gainers = len(df_scan[df_scan['Change %'] > 0]) if not df_scan.empty else 0
avg_vol_exp = round(df_scan['raw_vol_mult'].mean(), 1) if not df_scan.empty else 1.0

with kpi1:
    st.markdown(f"""
    <div style="background: rgba(15,23,42,0.85); border: 1px solid rgba(0,243,255,0.3); border-radius: 12px; padding: 14px; text-align: center;">
        <p style="color:#94a3b8; font-size:11px; margin:0;">US NET GAINERS</p>
        <h2 style="color:#00F3FF; margin:4px 0;">{gainers} / {len(df_scan)}</h2>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div style="background: rgba(15,23,42,0.85); border: 1px solid rgba(255,215,0,0.3); border-radius: 12px; padding: 14px; text-align: center;">
        <p style="color:#94a3b8; font-size:11px; margin:0;">US VOLUME SHOCKS</p>
        <h2 style="color:#FFD700; margin:4px 0;">⚡ {vol_shocks_count}</h2>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div style="background: rgba(15,23,42,0.85); border: 1px solid rgba(16,185,129,0.3); border-radius: 12px; padding: 14px; text-align: center;">
        <p style="color:#94a3b8; font-size:11px; margin:0;">AVG VOL EXPANSION</p>
        <h2 style="color:#10B981; margin:4px 0;">{avg_vol_exp}x</h2>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown("""
    <div style="background: rgba(15,23,42,0.85); border: 1px solid rgba(239,68,68,0.3); border-radius: 12px; padding: 14px; text-align: center;">
        <p style="color:#94a3b8; font-size:11px; margin:0;">US SESSION</p>
        <h2 style="color:#EF4444; margin:4px 0;">🦅 NY SESSION</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("### 📊 US Breakouts & Momentum Stocks")
if not df_scan.empty:
    df_sorted = df_scan.sort_values(by='raw_vol_mult', ascending=False)
    
    st.dataframe(
        df_sorted[['Symbol', 'Price ($)', 'Change %', 'Vol Multiple', 'Shock Status']],
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("### 📌 Pin US Stock to Focus List")
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_ticker = st.selectbox("Select breakout US stock for Focus List:", df_sorted['Symbol'].tolist())
    with col_btn:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        if st.button("➕ Pin US Setup"):
            if add_to_focus_list(selected_ticker, market="US"):
                st.success(f"Added {selected_ticker} to Focus List!")
else:
    st.info("No US data fetched yet. Click 'Refresh US Data' above.")
