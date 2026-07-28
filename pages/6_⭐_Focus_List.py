"""
Focus List & Breakout Execution Hub (India & US) for Kush Tracker Lite.
Features:
- Auto-populated Volume Shock Breakouts from Turso DB
- User-pinned setups with custom Entry Triggers, Stop Loss levels & Trade Notes
- Live 15-second price auto-refresh & breakdown risk tracking
"""
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

from database import (
    get_focus_list,
    get_active_volume_shocks,
    remove_from_focus_list,
    update_focus_list_trade_plan,
    mark_shock_failed
)

st.markdown("## ⭐ Focus List & Breakout Execution Hub")
st.caption("Your Nightly & Live Trading Workspace — Monitor Breakouts, Entry Triggers & Stop Losses")

st.markdown("---")

tab_in, tab_us = st.tabs(["🇮🇳 India Setups", "🇺🇸 US Setups"])

# Helper function to render focus list tab
def render_focus_tab(market_code: str):
    auto_refresh = st.checkbox(f"🔄 Live 15s Auto-Refresh ({market_code})", value=False, key=f"ar_{market_code}")
    
    # 1. Fetch Auto Volume Shock Breakouts from Turso DB
    shocks = get_active_volume_shocks(market_code)
    
    if shocks:
        st.markdown(f"### ⚡ Automated Institutional Volume Shock Breakouts ({len(shocks)})")
        st.caption("Auto-populated from Intraday Monitor scans when Volume Expansion ≥ 2.0x")
        
        shock_data = []
        for s in shocks:
            t = s['ticker']
            yf_sym = t if (market_code == "US" or t.endswith(".NS")) else f"{t}.NS"
            try:
                ticker_obj = yf.Ticker(yf_sym)
                hist = ticker_obj.history(period="5d")
                curr_price = float(hist['Close'].iloc[-1]) if not hist.empty else float(s.get('shock_close', 0.0))
            except Exception:
                curr_price = float(s.get('shock_close', 0.0))
                
            entry_trig = float(s.get('shock_high', 0.0))
            stop_lvl = float(s.get('shock_low', 0.0))
            
            # Check breakdown status
            status_badge = "🔥 BREAKOUT ACTIVE"
            if curr_price < stop_lvl and stop_lvl > 0:
                status_badge = "🔴 BROKE SHOCK LOW (STOP TRIGGERED)"
                mark_shock_failed(t, "Broke Shock Low")
            elif curr_price >= entry_trig and entry_trig > 0:
                status_badge = "🟢 TRIGGERED ABOVE HIGH"
                
            shock_data.append({
                'Symbol': t,
                'Shock Date': s.get('shock_date', 'N/A'),
                'Vol Expansion': f"{round(float(s.get('shock_vol_multiple', 1.0)), 1)}x",
                'Live Price': round(curr_price, 2),
                'Entry Trigger (High)': round(entry_trig, 2),
                'Stop Loss (Low)': round(stop_lvl, 2),
                'Execution Status': status_badge
            })
            
        st.dataframe(pd.DataFrame(shock_data), use_container_width=True, hide_index=True)
    else:
        st.info("No active volume shock breakouts logged yet. Run Intraday Monitor to auto-detect breakouts.")
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # 2. Fetch User Pinned Focus List Setups
    user_list = get_focus_list(market_code)
    st.markdown(f"### 📌 User Pinned Focus Setups ({len(user_list)})")
    
    if user_list:
        for item in user_list:
            t = item['ticker']
            with st.expander(f"⭐ {t} — Entry: ₹{item.get('entry_trigger', 0.0)} | Stop: ₹{item.get('stop_loss', 0.0)}", expanded=True):
                col_e1, col_e2, col_e3, col_del = st.columns([2, 2, 4, 1])
                with col_e1:
                    new_entry = st.number_input("Entry Trigger", value=float(item.get('entry_trigger') or 0.0), key=f"e_{t}")
                with col_e2:
                    new_stop = st.number_input("Stop Loss", value=float(item.get('stop_loss') or 0.0), key=f"s_{t}")
                with col_e3:
                    new_notes = st.text_input("Trade Plan / Notes", value=str(item.get('notes') or ""), key=f"n_{t}")
                with col_del:
                    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                    if st.button("🗑️ Remove", key=f"r_{t}"):
                        remove_from_focus_list(t)
                        st.rerun()
                        
                if st.button("💾 Save Plan", key=f"save_{t}"):
                    update_focus_list_trade_plan(t, new_entry, new_stop, new_notes)
                    st.success(f"Updated trade plan for {t}!")
    else:
        st.info("No pinned setups in your Focus List. Pin stocks from Intraday Monitor or True Market Leaders.")

with tab_in:
    render_focus_tab("IN")

with tab_us:
    render_focus_tab("US")
