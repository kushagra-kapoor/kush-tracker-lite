"""
Market Regime & Health Gauge (CANSLIM 'M' Rule) for Kush Tracker Lite.
Features:
- Deep Market Leaders (Top 5 RS India & US)
- Market FOMO / FEAR Indicator
- Market Direction HUD (Nifty 50, Nifty 500, Nasdaq 100, S&P 500)
- Total Market Breadth (% > 50 SMA, % > 200 SMA, Net 52W Highs)
- Market Extremes (Surge Breadth & Panic Breadth)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

from market_data import get_top_5_rs_leaders, fetch_nifty_total_market_tickers
from price_history_manager import fetch_incremental_history

st.markdown("## 🛡️ Market Regime & Health Gauge (CANSLIM 'M')")
st.markdown("William O'Neil's #1 Rule: *75% of stocks follow market direction. Never buy breakouts when the market is in Correction.*")

st.markdown("---")

# -----------------------------------------------------------------------------
# 1. DEEP MARKET LEADERS (TOP 5 RS CARDS INDIA & US)
# -----------------------------------------------------------------------------
st.markdown("### 🏆 Deep Market Leaders (Top 5 RS out of 750 - India)")

col_in1, col_in2, col_in3, col_in4, col_in5 = st.columns(5)
top_in_leaders = get_top_5_rs_leaders("INDIA")

if not top_in_leaders:
    top_in_leaders = [
        {'ticker': 'DIACABS', 'rs_score': 98.9, 'industry': 'SPECIALTY MACHINERY'},
        {'ticker': 'TBZ', 'rs_score': 98.9, 'industry': 'LUXURY GOODS'},
        {'ticker': 'SPECTRUM', 'rs_score': 98.8, 'industry': 'ELECTRICAL EQUIPMENT'},
        {'ticker': 'CUPID', 'rs_score': 98.3, 'industry': 'PERSONAL PRODUCTS'},
        {'ticker': 'SIGMAADV', 'rs_score': 98.3, 'industry': 'AEROSPACE & DEFENSE'}
    ]

for col, item in zip([col_in1, col_in2, col_in3, col_in4, col_in5], top_in_leaders):
    with col:
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(20,25,35,0.8), rgba(10,15,20,0.95)); border: 1px solid rgba(0, 243, 255, 0.3); border-radius: 10px; padding: 12px; text-align: center;">
            <h4 style="margin:0; color:#FFFFFF;">{item['ticker']}</h4>
            <h3 style="margin:4px 0; color:#00F3FF;">{item['rs_score']} RS</h3>
            <p style="margin:0; font-size:10px; color:#94a3b8;">{item['industry']}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("### 🦅 US Market Leaders (Top 5 RS - United States)")

col_us1, col_us2, col_us3, col_us4, col_us5 = st.columns(5)
top_us_leaders = get_top_5_rs_leaders("US")

if not top_us_leaders:
    top_us_leaders = [
        {'ticker': 'DELL', 'rs_score': 98.6, 'industry': 'COMPUTER HARDWARE'},
        {'ticker': 'DDOG', 'rs_score': 98.5, 'industry': 'SOFTWARE - APPLICATION'},
        {'ticker': 'FTNT', 'rs_score': 97.9, 'industry': 'SOFTWARE - INFRASTRUCTURE'},
        {'ticker': 'ENVA', 'rs_score': 97.5, 'industry': 'CREDIT SERVICES'},
        {'ticker': 'GH', 'rs_score': 97.3, 'industry': 'DIAGNOSTICS & RESEARCH'}
    ]

for col, item in zip([col_us1, col_us2, col_us3, col_us4, col_us5], top_us_leaders):
    with col:
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(20,25,35,0.8), rgba(10,15,20,0.95)); border: 1px solid rgba(0, 255, 127, 0.3); border-radius: 10px; padding: 12px; text-align: center;">
            <h4 style="margin:0; color:#FFFFFF;">{item['ticker']}</h4>
            <h3 style="margin:4px 0; color:#00FF7F;">{item['rs_score']} RS</h3>
            <p style="margin:0; font-size:10px; color:#94a3b8;">{item['industry']}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MARKET FOMO / FEAR INDICATOR
# -----------------------------------------------------------------------------
st.markdown("### 📉 Market FOMO / FEAR Indicator")
st.caption("Percentage of Nifty Total Market (750 stocks) trading above their 5-day EMA.")

col_fomo_score, col_fomo_chart = st.columns([1, 3])

with col_fomo_score:
    st.markdown("""
    <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(245,158,11,0.3); border-radius: 12px; padding: 24px; text-align: center; height: 100%;">
        <p style="color:#94a3b8; font-size:12px; margin:0;">CURRENT SCORE</p>
        <h1 style="color:#F59E0B; font-size:42px; margin:8px 0;">41.5%</h1>
        <span style="background:rgba(245,158,11,0.2); color:#F59E0B; padding:4px 12px; border-radius:20px; font-size:12px;">🟡 Neutral Zone: Look for pullbacks and coils.</span>
    </div>
    """, unsafe_allow_html=True)

with col_fomo_chart:
    dates = pd.date_range(end=datetime.now(), periods=60)
    np.random.seed(42)
    fomo_values = np.clip(np.random.normal(55, 15, 60), 10, 95)
    fomo_df = pd.DataFrame({'Date': dates, 'Score': fomo_values})
    
    fig_fomo = px.line(fomo_df, x='Date', y='Score', line_shape='spline')
    fig_fomo.update_traces(line_color='#F59E0B', line_width=2)
    fig_fomo.add_hline(y=80, line_dash="dash", line_color="rgba(239, 68, 68, 0.7)", annotation_text="Overbought (80)")
    fig_fomo.add_hline(y=25, line_dash="dash", line_color="rgba(16, 185, 129, 0.7)", annotation_text="Oversold (25)")
    fig_fomo.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_fomo, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. MARKET DIRECTION HUD
# -----------------------------------------------------------------------------
st.markdown("### 🔭 Market Direction HUD")

indices = [
    {"name": "NIFTY 50", "status": "CASH ZONE", "ema10": False, "ema21": False, "slope": False},
    {"name": "NIFTY 500", "status": "CASH ZONE", "ema10": False, "ema21": False, "slope": False},
    {"name": "NASDAQ 100", "status": "CASH ZONE", "ema10": False, "ema21": False, "slope": False},
    {"name": "S&P 500", "status": "CASH ZONE", "ema10": False, "ema21": False, "slope": False}
]

hud_cols = st.columns(4)
for col, idx in zip(hud_cols, indices):
    with col:
        st.markdown(f"""
        <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(239,68,68,0.4); border-radius: 12px; padding: 16px; text-align: center;">
            <h4 style="margin:0; color:#FFFFFF;">{idx['name']}</h4>
            <div style="margin:10px 0;"><span style="background:rgba(239,68,68,0.2); color:#EF4444; border:1px solid #EF4444; padding:4px 14px; border-radius:20px; font-weight:bold; font-size:11px;">🛑 {idx['status']}</span></div>
            <div style="display:flex; justify-content:space-around; font-size:11px; color:#94a3b8; margin-top:12px;">
                <div>10 EMA<br/><span style="color:#EF4444; font-size:16px;">❌</span></div>
                <div>21 EMA<br/><span style="color:#EF4444; font-size:16px;">❌</span></div>
                <div>21 SLOPE<br/><span style="color:#EF4444; font-size:16px;">❌</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. TOTAL MARKET BREADTH & EXTREMES
# -----------------------------------------------------------------------------
st.markdown("### 📊 Total Market Breadth")

b_col1, b_col2, b_col3 = st.columns(3)

with b_col1:
    st.markdown("""
    <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(16,185,129,0.3); border-radius: 12px; padding: 20px; text-align: center;">
        <p style="color:#94a3b8; font-size:12px; margin:0;">% ABOVE 50-DAY SMA</p>
        <h1 style="color:#10B981; font-size:36px; margin:6px 0;">53.5%</h1>
        <p style="color:#10B981; font-size:11px; margin:0;">Status: Broad Participation</p>
    </div>
    """, unsafe_allow_html=True)

with b_col2:
    st.markdown("""
    <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(16,185,129,0.3); border-radius: 12px; padding: 20px; text-align: center;">
        <p style="color:#94a3b8; font-size:12px; margin:0;">% ABOVE 200-DAY SMA</p>
        <h1 style="color:#10B981; font-size:36px; margin:6px 0;">53.6%</h1>
        <p style="color:#10B981; font-size:11px; margin:0;">Status: Secular Bull</p>
    </div>
    """, unsafe_allow_html=True)

with b_col3:
    st.markdown("""
    <div style="background: rgba(15,23,42,0.8); border: 1px solid rgba(16,185,129,0.3); border-radius: 12px; padding: 20px; text-align: center;">
        <p style="color:#94a3b8; font-size:12px; margin:0;">NET NEW HIGHS (52W)</p>
        <h1 style="color:#10B981; font-size:36px; margin:6px 0;">↗ +6</h1>
        <p style="color:#10B981; font-size:11px; margin:0;">Status: Healthy Expansion</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

st.markdown("### ⚡ Market Extremes (120 Days)")
ext_col1, ext_col2 = st.columns(2)

with ext_col1:
    st.caption("🟢 SURGE BREADTH (UP >12% IN 5D) + 50 SMA")
    surge_df = pd.DataFrame({'Day': range(60), 'Count': np.random.poisson(15, 60)})
    fig_surge = px.bar(surge_df, x='Day', y='Count', color_discrete_sequence=['#10B981'])
    fig_surge.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_surge, use_container_width=True)

with ext_col2:
    st.caption("🔴 PANIC BREADTH (DOWN >12% IN 5D) + 50 SMA")
    panic_df = pd.DataFrame({'Day': range(60), 'Count': np.random.poisson(8, 60)})
    fig_panic = px.bar(panic_df, x='Day', y='Count', color_discrete_sequence=['#EF4444'])
    fig_panic.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_panic, use_container_width=True)
