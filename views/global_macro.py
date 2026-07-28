import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from styles import load_css
    load_css()
except ImportError:
    pass

from database import get_connection, get_latest_global_regime, get_latest_global_etf_momentum

def fetch_latest_breadth(market):
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM market_breadth_daily WHERE market=? ORDER BY date DESC LIMIT 100", conn, params=(market,))
    conn.close()
    return df

def fetch_latest_liquidity(market):
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM market_liquidity_daily WHERE market=? ORDER BY date DESC LIMIT 252", conn, params=(market,))
    conn.close()
    return df

def render_market_column(market_code, market_name):
    st.subheader(f"🌐 {market_name}")
    
    # 1. Regime
    regimes = get_latest_global_regime(market=market_code)
    if not regimes:
        st.warning(f"No regime data for {market_code}. Please run sync_macro.py.")
        return
        
    reg = regimes[0]
    
    color = "green" if "Uptrend" in reg['regime_label'] else "red" if "Correction" in reg['regime_label'] else "yellow"
    st.markdown(f"""
    <div class="status-box {color}-status" style="margin-bottom: 20px;">
        <div class="status-title">{reg['regime_label']}</div>
        <div class="status-subtitle" style="font-size: 0.9em;">
            {reg['benchmark_ticker']} | Dist Days: <strong>{reg['dd_count']}</strong> | vs 50SMA: <strong>{'+' if reg['close'] > reg['sma50'] else ''}{((reg['close']/reg['sma50'])-1)*100:.1f}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Breadth
    breadth_df = fetch_latest_breadth(market_code)
    if not breadth_df.empty:
        latest_b = breadth_df.iloc[0]
        nnh = latest_b['net_new_highs']
        
        col1, col2 = st.columns(2)
        col1.metric("Net New Highs", f"{nnh}", delta="Expansion" if nnh > 50 else "Contraction" if nnh < -50 else "Neutral", delta_color="normal" if nnh > 0 else "inverse")
        col2.metric("Stocks > 50 SMA", f"{latest_b['above_50_pct']:.1f}%")
        
        # Breadth Chart
        breadth_df['date'] = pd.to_datetime(breadth_df['date'])
        breadth_df = breadth_df.sort_values('date')
        fig_b = px.bar(breadth_df.tail(60), x='date', y='net_new_highs', title=f"Net New Highs (60 Days)")
        fig_b.update_traces(marker_color=['#10b981' if val > 0 else '#ef4444' for val in breadth_df.tail(60)['net_new_highs']])
        fig_b.update_layout(height=220, margin=dict(l=0, r=0, t=30, b=0), template='plotly_dark')
        st.plotly_chart(fig_b, use_container_width=True)
        
    # 3. Liquidity
    liq_df = fetch_latest_liquidity(market_code)
    if not liq_df.empty:
        liq_df['date'] = pd.to_datetime(liq_df['date'])
        liq_df = liq_df.sort_values('date')
        
        fig_l = px.line(liq_df, x='date', y='monthly_turnover_k_cr', title=f"Monthly Turnover & 200SMA")
        fig_l.add_scatter(x=liq_df['date'], y=liq_df['sma_200'], mode='lines', name='200 SMA', line=dict(color='orange', dash='dot'))
        fig_l.update_layout(height=220, margin=dict(l=0, r=0, t=30, b=0), template='plotly_dark', showlegend=False)
        st.plotly_chart(fig_l, use_container_width=True)


def main():
    from components import render_header
    render_header("🌍 Global Macro View", "Continuous Cross-Market Aggregation & Capital Flow Tracking.")
    
    # Check overall Global Flow
    in_regime = get_latest_global_regime(market='IN')
    us_regime = get_latest_global_regime(market='US')
    
    if in_regime and us_regime:
        in_label = in_regime[0]['regime_label']
        us_label = us_regime[0]['regime_label']
        
        global_status = "Neutral / Mixed Flow"
        g_color = "#eab308" # yellow
        if "Uptrend" in in_label and "Uptrend" in us_label:
            global_status = "🔥 FULL RISK ON"
            g_color = "#10b981" # green
        elif "Correction" in in_label and "Correction" in us_label:
            global_status = "❄️ FULL RISK OFF"
            g_color = "#ef4444" # red
            
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; border-left: 5px solid {g_color}; margin-bottom: 25px; text-align: center;">
            <h2 style="margin:0; color: {g_color}; letter-spacing: 2px;">{global_status}</h2>
            <p style="margin:5px 0 0 0; color: #a1a1aa; font-size: 0.9em;">Synthesized from Nifty 500 & S&P 500 Macro States</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Split Screen
    col_in, col_us = st.columns(2)
    with col_in:
        render_market_column('IN', 'India (Nifty 500)')
    with col_us:
        render_market_column('US', 'United States (S&P 500)')
        
    st.markdown("---")
    
    # Global Asset Rotation
    st.subheader("🔄 Global Asset Rotation (ETF Momentum)")
    st.caption("Visualizing international capital flows over 1M, 3M, and 6M windows.")
    
    etf_data = get_latest_global_etf_momentum()
    if etf_data:
        df_etf = pd.DataFrame(etf_data)
        
        t1, t2, t3 = st.tabs(["1-Month Rotation", "3-Month Rotation", "6-Month Rotation"])
        
        with t1:
            df_1m = df_etf.sort_values('return_1m', ascending=False).head(15)
            fig_1 = px.bar(df_1m, x='ticker', y='return_1m', title="Top 15 Global ETFs (1M Return %)", text_auto='.1f', color='return_1m', color_continuous_scale='RdYlGn')
            fig_1.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig_1, use_container_width=True)
            
        with t2:
            df_3m = df_etf.sort_values('return_3m', ascending=False).head(15)
            fig_3 = px.bar(df_3m, x='ticker', y='return_3m', title="Top 15 Global ETFs (3M Return %)", text_auto='.1f', color='return_3m', color_continuous_scale='RdYlGn')
            fig_3.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig_3, use_container_width=True)
            
        with t3:
            df_6m = df_etf.sort_values('return_6m', ascending=False).head(15)
            fig_6 = px.bar(df_6m, x='ticker', y='return_6m', title="Top 15 Global ETFs (6M Return %)", text_auto='.1f', color='return_6m', color_continuous_scale='RdYlGn')
            fig_6.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig_6, use_container_width=True)
            
        with st.expander("View All ETF Data"):
            st.dataframe(
                df_etf[['ticker', 'return_1m', 'return_3m', 'return_6m']].sort_values('return_6m', ascending=False), 
                column_config={
                    "ticker": "ETF Ticker",
                    "return_1m": st.column_config.NumberColumn("1M Return (%)", format="%.2f"),
                    "return_3m": st.column_config.NumberColumn("3M Return (%)", format="%.2f"),
                    "return_6m": st.column_config.NumberColumn("6M Return (%)", format="%.2f")
                },
                use_container_width=True, hide_index=True
            )
    else:
        st.info("No Global ETF Momentum data available. Please run sync_macro.py.")

main()
