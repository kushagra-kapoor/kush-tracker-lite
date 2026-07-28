"""
Intraday Monitor (India) for Kush Tracker Lite.
Scans Nifty Total Market / ETFs for Volume Surges, Breakouts & Squats.
Auto-logs Volume Shock Breakouts to Turso DB and provides 1-click Pin to Focus List.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from market_data import fetch_nifty_total_market_tickers
from price_history_manager import fetch_incremental_history
from database import log_volume_shock, add_to_focus_list

st.markdown("## ⚡ Intraday Monitor (India)")
st.caption("Live Breadth, Volume Surges & Institutional Footprints for Indian Equities (NSE)")

st.markdown("---")

# Refresh Controls
col_ctrl1, col_ctrl2 = st.columns([1, 4])
with col_ctrl1:
    force_refresh = st.button("🔄 Refresh Market Data", type="primary")

tickers_raw = fetch_nifty_total_market_tickers()
yf_tickers = [t if t.endswith(".NS") or t.endswith(".BO") else f"{t}.NS" for t in tickers_raw[:150]]

with st.spinner("Fetching live market data..."):
    histories = fetch_incremental_history(yf_tickers, days=60, force_today_refresh=force_refresh)

scan_rows = []
vol_shocks_count = 0

for orig_t, yf_t in zip(tickers_raw[:150], yf_tickers):
    df = histories.get(yf_t, pd.DataFrame())
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
            
            # Check for Volume Shock Breakout (Vol >= 2.0x & Change >= 1.5%)
            is_shock = (vol_mult >= 2.0 and pct_change >= 1.5)
            if is_shock:
                vol_shocks_count += 1
                today_str = datetime.now().strftime("%Y-%m-%d")
                log_volume_shock(orig_t, today_str, vol_mult, close, high, low, market="IN")
                
            scan_rows.append({
                'Symbol': orig_t,
                'Price (₹)': round(close, 2),
                'Change %': round(pct_change, 2),
                'Vol Multiple': f"{round(vol_mult, 1)}x",
                'Shock Status': '🔥 VOLUME SHOCK' if is_shock else 'Normal',
                'raw_vol_mult': vol_mult
            })
        except Exception:
            continue

df_scan = pd.DataFrame(scan_rows)

# Glassmorphic Metric Cards (100% Reliable HTML Rendering)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
gainers = len(df_scan[df_scan['Change %'] > 0]) if not df_scan.empty else 0
avg_vol_exp = round(df_scan['raw_vol_mult'].mean(), 1) if not df_scan.empty else 1.0

with kpi1:
    st.markdown(f"""
    <div style="background: rgba(15,23,42,0.85); border: 1px solid rgba(0,243,255,0.3); border-radius: 12px; padding: 14px; text-align: center;">
        <p style="color:#94a3b8; font-size:11px; margin:0;">NET GAINERS</p>
        <h2 style="color:#00F3FF; margin:4px 0;">{gainers} / {len(df_scan)}</h2>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div style="background: rgba(15,23,42,0.85); border: 1px solid rgba(255,215,0,0.3); border-radius: 12px; padding: 14px; text-align: center;">
        <p style="color:#94a3b8; font-size:11px; margin:0;">VOLUME SHOCKS</p>
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
        <p style="color:#94a3b8; font-size:11px; margin:0;">MARKET STATUS</p>
        <h2 style="color:#EF4444; margin:4px 0;">🔥 LIVE SCAN</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("---")

# Stock Table
st.markdown("### 📊 Scanned Breakouts & Momentum Stocks")
if not df_scan.empty:
    df_sorted = df_scan.sort_values(by='raw_vol_mult', ascending=False)
    
    st.dataframe(
        df_sorted[['Symbol', 'Price (₹)', 'Change %', 'Vol Multiple', 'Shock Status']],
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("### 📌 Pin Stock to Focus List")
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_ticker = st.selectbox("Select breakout stock to add to your execution Focus List:", df_sorted['Symbol'].tolist())
    with col_btn:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        if st.button("➕ Pin to Focus List"):
            if add_to_focus_list(selected_ticker, market="IN"):
                st.success(f"Added {selected_ticker} to Focus List!")
else:
    st.info("No market data fetched yet. Click 'Refresh Market Data' above.")
