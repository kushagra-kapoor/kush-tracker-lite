import streamlit as st
import pandas as pd
import sys
import os

# Ensure the root directory is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_focus_list, update_focus_list_trade_plan, remove_from_focus_list, get_active_volume_shocks, mark_shock_failed
from components import render_header
import time
import yfinance as yf

try:
    from win11toast import toast
    TOAST_AVAILABLE = True
except ImportError:
    TOAST_AVAILABLE = False

# st.set_page_config(
#     page_title="Focus List - Kush Tracker",
#     page_icon="⭐",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

try:
    from styles import load_css
    load_css()
except ImportError:
    pass

# --- Session State for Notifications ---
if 'notified_tickers' not in st.session_state:
    st.session_state.notified_tickers = set()

if 'batched_notifications' not in st.session_state:
    st.session_state.batched_notifications = []

render_header("⭐ Focus List", "Your nightly action plan for tomorrow's session")

st.markdown("""
This is your isolated execution workspace. Every night, pin your top 3-5 setups here from the TML page. 
Set your strict **Entry Trigger** and **Stop Loss** levels. During the trading day, only watch this page.
""")

# --- Live Auto Refresh Logic ---
auto_refresh = st.sidebar.checkbox("🔄 Live Auto-Refresh (15s)", value=False, help="Automatically fetch live prices every 15 seconds")

st.markdown("---")

tab_in, tab_us = st.tabs(["🇮🇳 India (NSE)", "🇺🇸 United States"])

@st.cache_data(ttl=60)
def fetch_live_price(ticker: str, market: str) -> float:
    """Fetch live price formatting the ticker correctly based on market."""
    if market == "IN":
        clean = ticker.upper().replace("NSE:", "").strip()
        if not clean.endswith(".NS"):
            clean += ".NS"
    else:
        clean = ticker.upper().strip()
        
    try:
        stock = yf.Ticker(clean)
        info = stock.info
        return info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
    except Exception:
        return None

def render_focus_table(market: str):
    # Quick Add UI
    st.markdown(f"### {market} Action Plan")
    col1, col2, _ = st.columns([2, 1, 3])
    with col1:
        new_ticker = st.text_input(f"Quick Add Ticker to {market} Focus List:", key=f"add_focus_{market}", placeholder="e.g. RELIANCE" if market == "IN" else "e.g. AAPL")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add to List", key=f"btn_add_{market}", use_container_width=True):
            if new_ticker:
                from database import add_to_focus_list
                if add_to_focus_list(new_ticker.upper().strip(), market):
                    st.success(f"Added {new_ticker.upper()}!")
                    st.rerun()
                else:
                    st.error("Failed to add.")
                    
    stocks = get_focus_list(market)
    
    if not stocks:
        st.info(f"No stocks pinned for {market}. Use the Quick Add above or go to the True Market Leader page to pin setups.")
        return
        
    # Prepare Dataframe
    df_data = []
    for s in stocks:
        ticker = s['ticker']
        current_price = fetch_live_price(ticker, market)
        
        # Determine Status
        status = "⏳ Waiting"
        
        if current_price is not None:
            if s['entry_trigger'] > 0 and current_price >= s['entry_trigger']:
                status = "🟢 TRIGGERED"
                
                # Add to batch Notification if not already sent
                if TOAST_AVAILABLE and ticker not in st.session_state.notified_tickers:
                    st.session_state.batched_notifications.append(f"🟢 {ticker} crossed entry of {s['entry_trigger']}")
                    st.session_state.notified_tickers.add(ticker)
                        
            elif s['stop_loss'] > 0 and current_price <= s['stop_loss']:
                status = "🔴 STOPPED OUT"
                
                # Add to batch Notification if not already sent
                if TOAST_AVAILABLE and f"{ticker}_STOP" not in st.session_state.notified_tickers:
                    st.session_state.batched_notifications.append(f"🔴 {ticker} breached stop of {s['stop_loss']}")
                    st.session_state.notified_tickers.add(f"{ticker}_STOP")
        else:
            current_price = 0.0
            status = "⚠️ No Price Data"
            
        df_data.append({
            "Ticker": ticker,
            "Live Price": current_price,
            "Entry Trigger": s['entry_trigger'],
            "Stop Loss": s['stop_loss'],
            "Status": status,
            "Notes": s['notes'],
            "❌ Remove": False
        })
        
    df = pd.DataFrame(df_data)
    
    # Render Data Editor
    
    currency_symbol = "₹" if market == "IN" else "$"
    
    edited_df = st.data_editor(
        df,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
            "Live Price": st.column_config.NumberColumn("Live Price", format=f"{currency_symbol}%.2f", disabled=True),
            "Status": st.column_config.TextColumn("Status", disabled=True),
            "Entry Trigger": st.column_config.NumberColumn("Entry Trigger", format="%.2f", step=0.05),
            "Stop Loss": st.column_config.NumberColumn("Stop Loss", format="%.2f", step=0.05),
            "Notes": st.column_config.TextColumn("Notes"),
            "❌ Remove": st.column_config.CheckboxColumn("❌ Remove", default=False)
        },
        hide_index=True,
        use_container_width=True,
        key=f"editor_{market}"
    )
    
    # Check for changes
    if st.button(f"💾 Save Changes ({market})", type="primary"):
        changes_made = False
        for i, row in edited_df.iterrows():
            ticker = row['Ticker']
            
            # Check for removal
            if row['❌ Remove']:
                remove_from_focus_list(ticker)
                changes_made = True
                continue
                
            # Check for updates
            orig_row = df[df['Ticker'] == ticker].iloc[0]
            if (row['Entry Trigger'] != orig_row['Entry Trigger'] or 
                row['Stop Loss'] != orig_row['Stop Loss'] or 
                row['Notes'] != orig_row['Notes']):
                
                update_focus_list_trade_plan(
                    ticker, 
                    float(row['Entry Trigger']), 
                    float(row['Stop Loss']), 
                    str(row['Notes'])
                )
                
                # Reset notification state if triggers are modified
                if ticker in st.session_state.notified_tickers:
                    st.session_state.notified_tickers.remove(ticker)
                if f"{ticker}_STOP" in st.session_state.notified_tickers:
                    st.session_state.notified_tickers.remove(f"{ticker}_STOP")
                    
                changes_made = True
                
        if changes_made:
            st.success("Focus list updated successfully!")
            st.rerun()
        else:
            st.info("No changes detected.")

def render_institutional_footprints(market: str):
    db_market = "INDIA" if market == "IN" else "USA"
    shocks = get_active_volume_shocks(db_market)
    
    if not shocks:
        return
        
    st.markdown("---")
    
    # Neon World-Class Header with Animation
    col_hdr, col_btn = st.columns([4, 1])
    with col_hdr:
        st.markdown("""
        <style>
        @keyframes pulse-neon {
            0% { box-shadow: 0 0 10px rgba(245, 158, 11, 0.2), inset 0 1px 0 rgba(255,255,255,0.1); }
            50% { box-shadow: 0 0 25px rgba(245, 158, 11, 0.5), inset 0 1px 0 rgba(255,255,255,0.1); }
            100% { box-shadow: 0 0 10px rgba(245, 158, 11, 0.2), inset 0 1px 0 rgba(255,255,255,0.1); }
        }
        .neon-cyber-box {
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.8) 0%, rgba(2, 6, 23, 0.95) 100%);
            padding: 1.5rem 2rem;
            border-radius: 16px;
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-left: 5px solid #f59e0b;
            animation: pulse-neon 3s infinite alternate;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }
        .neon-cyber-box::before {
            content: '';
            position: absolute;
            top: -50%; left: -10%; width: 50%; height: 200%;
            background: radial-gradient(circle, rgba(245, 158, 11, 0.15) 0%, transparent 70%);
            transform: rotate(15deg);
            pointer-events: none;
        }
        .cyber-text {
            color: #fcfcfc;
            margin: 0;
            font-weight: 800;
            font-size: 1.5rem;
            display: flex;
            align-items: center;
            gap: 12px;
            letter-spacing: 1px;
            text-transform: uppercase;
            text-shadow: 0 0 15px rgba(245, 158, 11, 0.8);
        }
        .cyber-subtext {
            color: #94a3b8;
            margin: 8px 0 0 0;
            font-size: 0.95rem;
            font-weight: 400;
            letter-spacing: 0.5px;
        }
        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(245, 158, 11, 0.2);
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.8);
        }
        </style>
        
        <div class="neon-cyber-box">
            <h3 class="cyber-text">
                <span style="font-size: 1.8rem;">⚡</span> 
                Institutional Footprints
            </h3>
            <p class="cyber-subtext">
                Massive volume shocks (>5x) tracked over T+30 days to catch multibaggers.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_btn:
        st.write("")
        if st.button(f"🔄 Sync Missing Days", key=f"btn_sync_{market}", use_container_width=True, help="Scans the last 30 days of market data to find any missed footprints while you were away."):
            with st.spinner(f"Running 30-Day Deep Scan for {market}..."):
                import os as _os
                file_name = 'tickers.txt' if market == "IN" else 'tickers_us.txt'
                file_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), file_name)
                with open(file_path, 'r') as f:
                    sync_tickers = [line.strip().upper() for line in f if line.strip() and '-' not in line]
                
                if market == "IN":
                    sync_tickers = [t + ".NS" for t in sync_tickers]
                sync_tickers = list(set(sync_tickers))
                
                from database import auto_backfill_footprints
                
                # Use the safe chunked downloader to prevent YF rate limits
                if market == "IN":
                    from views.intraday_monitor import fetch_yfinance_batch
                else:
                    from views.intraday_monitor_us import fetch_yfinance_batch
                    
                hist_df = fetch_yfinance_batch(sync_tickers, days=60, force_today_refresh=True)
                
                if hist_df is not None and not hist_df.empty:
                    if len(sync_tickers) == 1 or not isinstance(hist_df.columns, pd.MultiIndex):
                        t_data = {sync_tickers[0]: hist_df}
                    else:
                        t_data = {t: hist_df[t] for t in sync_tickers if t in hist_df.columns.get_level_values(0).unique()}
                    
                    found_count = auto_backfill_footprints(t_data, market, lookback_days=30)
                else:
                    found_count = 0
                
                if found_count > 0:
                    st.success(f"Successfully restored {found_count} missed footprints!")
                else:
                    st.info("Scan complete. No missed footprints found.")
                
                # Rerun to show new footprints
                st.rerun()
    
    with st.expander("ℹ️ How to interpret Institutional Footprints"):
        st.markdown("""
        **What is an Episodic Pivot (EP)?**
        An EP occurs when a stock gaps up or surges on massive volume (usually >5x average). This is the "footprint" of an institution aggressively accumulating shares.
        
        **Behavior Interpretation:**
        * 🎾 **Tennis Ball (Tight):** The stock surged and is holding near its highs. The drawdown from the post-shock high is less than 50% of the shock day's range. This shows institutions are supporting the price (accumulation).
        * 🥚 **Egg (Fading):** The stock is giving up too much ground. 
            * *Why do some show as Egg on T+0?* If a stock is an Egg on the exact same day it triggered, it means it formed a massive upper wick (Shooting Star). Despite huge volume, it closed in the bottom half of its daily range. This is a severe warning sign of **hidden institutional distribution** (they sold heavily into the retail buying frenzy).
        
        **Advanced Indicators:**
        * 🪫 **VDU (Volume Dry Up):** When volume drops below 40% of the 20-day average. This indicates sellers are exhausted. A VDU resting on a moving average is a perfect low-risk "squat" entry.
        * 👑 **HTF Warning (Power Play):** The stock has surged >20% since the shock and is now consolidating tightly (<15% range). This is the most explosive CANSLIM setup.
        """)
    
    valid_shocks = []
    
    # Dramatically speed up load times by fetching all active footprint data concurrently
    if shocks:
        benchmark_ticker = "^CRSLDX" if market == "IN" else "SPY"
        clean_tickers_list = [shock['ticker'] if market == "IN" and shock['ticker'].endswith(".NS") else f"{shock['ticker']}.NS" if market == "IN" else shock['ticker'] for shock in shocks]
        clean_tickers_list.append(benchmark_ticker)
        clean_tickers_unique = list(set(clean_tickers_list))
        bulk_history = yf.download(clean_tickers_unique, period='1y', group_by='ticker', threads=True, progress=False)
        
        # Calculate benchmark returns for RS scoring
        from relative_strength import calculate_stock_returns, calculate_relative_returns, calculate_weighted_rs_raw, normalize_rs_against_nifty500
        if len(clean_tickers_unique) == 1 or not isinstance(bulk_history.columns, pd.MultiIndex):
            benchmark_df = bulk_history if not bulk_history.empty else pd.DataFrame()
        else:
            benchmark_df = bulk_history[benchmark_ticker].copy() if benchmark_ticker in bulk_history.columns.get_level_values(0) else pd.DataFrame()
            
        if not benchmark_df.empty:
            b_df = benchmark_df.copy()
            b_df.columns = [str(c).lower() for c in b_df.columns]
            if 'close' in b_df.columns:
                b_df = b_df.dropna(subset=['close'])
            benchmark_returns = calculate_stock_returns(b_df)
        else:
            benchmark_returns = {'R1': None, 'R3': None, 'R6': None}
    else:
        bulk_history = pd.DataFrame()
        benchmark_returns = {'R1': None, 'R3': None, 'R6': None}
    for shock in shocks:
        t = shock['ticker']
        try:
            clean_ticker = t if market == "IN" and t.endswith(".NS") else f"{t}.NS" if market == "IN" else t
            
            # Extract from the bulk downloaded dataframe
            if bulk_history.empty:
                df = pd.DataFrame()
            elif len(clean_tickers_unique) == 1 or not isinstance(bulk_history.columns, pd.MultiIndex):
                df = bulk_history.copy()
            else:
                df = bulk_history[clean_ticker].copy() if clean_ticker in bulk_history.columns.get_level_values(0) else pd.DataFrame()
            
            if not df.empty:
                if 'Close' in df.columns:
                    df = df.dropna(subset=['Close'])
                elif 'close' in df.columns:
                    df = df.dropna(subset=['close'])
                
            if df.empty:
                valid_shocks.append({
                    'Ticker': f"https://in.tradingview.com/chart/?symbol={'NSE:' if market=='IN' else ''}{t.replace('.NS', '')}",
                    'Shock Date': shock['shock_date'],
                    'Days': "⚠️ YF Rate Limit",
                    'Vol Mult': float(shock['shock_vol_multiple']),
                    'Last Price': float(shock['shock_close']),
                    'Close Range': 0.0,
                    'Behavior': "⚠️ API Blocked",
                    'VDU': "⚫ Normal",
                    'HTF Warning': "⚫ None",
                    'RS Score': 0,
                    'Dist to 52W High': 0.0
                })
                continue
                
            shock_date_dt = pd.to_datetime(shock['shock_date']).tz_localize(None)
            df.index = df.index.tz_localize(None)
            post_shock_df = df[df.index >= shock_date_dt]
            
            if post_shock_df.empty:
                # If shock was logged today after market close, maybe Yahoo finance hasn't updated its EOD yet for the history endpoint.
                # Just use the shock day's close for now and mark as T+0
                valid_shocks.append({
                    'Ticker': f"https://in.tradingview.com/chart/?symbol={'NSE:' if market=='IN' else ''}{t.replace('.NS', '')}",
                    'Shock Date': shock['shock_date'],
                    'Days': f"T+0",
                    'Vol Mult': float(shock['shock_vol_multiple']),
                    'Last Price': float(shock['shock_close']),
                    'Close Range': 50.0, # Placeholder if missing
                    'Behavior': "🎾 Tennis Ball (Tight)",
                    'VDU': "⚫ Normal",
                    'HTF Warning': "⚫ None",
                    'RS Score': 0,
                    'Dist to 52W High': 0.0
                })
                continue
                
            current_close = float(post_shock_df['Close'].iloc[-1])
            current_low = float(post_shock_df['Low'].iloc[-1])
            current_high = float(post_shock_df['High'].iloc[-1])
            current_vol = float(post_shock_df['Volume'].iloc[-1])
            
            # Calculate Close Range
            if current_high > current_low:
                close_range = ((current_close - current_low) / (current_high - current_low)) * 100
            else:
                close_range = 0.0
                
            trading_days = len(post_shock_df) - 1 # T+0 is the shock day itself
            
            # Expiration Logic
            if current_close < shock['shock_low']:
                mark_shock_failed(t, "Closed below Shock Low")
                continue
            if trading_days > 30:
                mark_shock_failed(t, "Time Expired (T+30 reached)")
                continue
                
            # CANSLIM Mechanics
            max_high_since_shock = float(post_shock_df['High'].max())
            shock_range = shock['shock_high'] - shock['shock_low']
            drawdown_from_high = max_high_since_shock - current_close
            
            is_egg = drawdown_from_high > (0.5 * shock_range) if shock_range > 0 else False
            behavior = "🥚 Egg (Fading)" if is_egg else "🎾 Tennis Ball (Tight)"
            
            # VDU Check
            avg_vol_20d = df['Volume'].iloc[-21:-1].mean() if len(df) >= 21 else df['Volume'].mean()
            vdu_flag = (current_vol < 0.4 * avg_vol_20d) if avg_vol_20d > 0 else False
            
            # Power Play Check
            is_power_play = False
            if trading_days <= 15 and max_high_since_shock >= shock['shock_close'] * 1.20:
                if max_high_since_shock > 0:
                    consolidation = (max_high_since_shock - current_low) / max_high_since_shock
                    if consolidation <= 0.15:
                        is_power_play = True
                        is_power_play = True
                
            # RS Score Calculation
            s_df = df.copy()
            s_df.columns = [str(c).lower() for c in s_df.columns]
            stock_returns = calculate_stock_returns(s_df)
            relative_returns = calculate_relative_returns(stock_returns, benchmark_returns)
            rs_raw = calculate_weighted_rs_raw(relative_returns)
            if rs_raw is not None and market == "IN":
                rs_score = normalize_rs_against_nifty500(rs_raw)
                if rs_score is None:
                    rs_score = min(99, max(1, int(50 + (rs_raw * 60))))
            else:
                # Fallback for US or missing
                rs_score = min(99, max(1, int(50 + (rs_raw * 60)))) if rs_raw is not None else 0
                
            # 52-Week High Calculation
            high_52w = float(df['High'].max()) if not df.empty else current_high
            dist_to_52w = ((current_close - high_52w) / high_52w * 100) if high_52w > 0 else 0.0
            
            valid_shocks.append({
                'Ticker': f"https://in.tradingview.com/chart/?symbol={'NSE:' if market=='IN' else ''}{t.replace('.NS', '')}",
                'Shock Date': shock['shock_date'],
                'Days': f"T+{trading_days}",
                'Vol Mult': float(shock['shock_vol_multiple']),
                'Last Price': float(current_close),
                'Close Range': float(close_range),
                'Behavior': behavior,
                'VDU': "🟢 ACTIVE" if vdu_flag else "⚫ Normal",
                'HTF Warning': "🔥 TRIGGERED" if is_power_play else "⚫ None",
                'RS Score': int(rs_score) if rs_score else 0,
                'Dist to 52W High': float(dist_to_52w)
            })
        except Exception as e:
            print(f"Error evaluating footprint {t}: {e}")
            continue
            
    if valid_shocks:
        df_shocks = pd.DataFrame(valid_shocks)
        # Sort by Vol Mult descending by default for better UX
        df_shocks = df_shocks.sort_values(by="Vol Mult", ascending=False)
        
        # --- ACTIONABLE FILTERS ---
        st.write("")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            show_tennis = st.checkbox("🎾 Tennis Balls Only", value=False, key=f"filter_tennis_{market}")
        with filter_col2:
            show_vdu = st.checkbox("🟢 VDU Squats Only", value=False, key=f"filter_vdu_{market}")
        with filter_col3:
            show_htf = st.checkbox("🔥 Power Plays Only", value=False, key=f"filter_htf_{market}")
            
        slider_col1, slider_col2 = st.columns(2)
        with slider_col1:
            min_rs = st.slider("Min RS Score", min_value=0, max_value=99, value=0, key=f"rs_{market}")
        with slider_col2:
            max_dist_input = st.slider("Max Dist from 52W High (%)", min_value=0, max_value=100, value=100, key=f"dist_{market}")
            max_dist = -max_dist_input
            
        if show_tennis:
            df_shocks = df_shocks[df_shocks['Behavior'].str.contains("Tennis Ball", na=False)]
        if show_vdu:
            df_shocks = df_shocks[df_shocks['VDU'].str.contains("ACTIVE", na=False)]
        if show_htf:
            df_shocks = df_shocks[df_shocks['HTF Warning'].str.contains("TRIGGERED", na=False)]
            
        df_shocks = df_shocks[df_shocks['RS Score'] >= min_rs]
        df_shocks = df_shocks[df_shocks['Dist to 52W High'] >= max_dist]
            
        df_shocks = df_shocks[['Ticker', 'Shock Date', 'Days', 'Vol Mult', 'Last Price', 'Dist to 52W High', 'Close Range', 'RS Score', 'Behavior', 'VDU', 'HTF Warning']]
        
        # --- DYNAMIC SORTING ---
        sort_col, _ = st.columns([1, 3])
        with sort_col:
            sort_metric = st.selectbox(
                "Sort Table By:", 
                ["Vol Mult (High to Low)", "RS Score (High to Low)", "Shock Date (Newest First)", "Distance to 52W High (Closest First)", "Close Range (High to Low)", "Days (Newest First)", "Ticker (A-Z)"],
                key=f"sort_{market}"
            )
            
        if sort_metric == "Vol Mult (High to Low)":
            df_shocks = df_shocks.sort_values(by="Vol Mult", ascending=False)
        elif sort_metric == "RS Score (High to Low)":
            df_shocks = df_shocks.sort_values(by="RS Score", ascending=False)
        elif sort_metric == "Shock Date (Newest First)":
            df_shocks = df_shocks.sort_values(by="Shock Date", ascending=False)
        elif sort_metric == "Distance to 52W High (Closest First)":
            df_shocks = df_shocks.sort_values(by="Dist to 52W High", ascending=False)
        elif sort_metric == "Close Range (High to Low)":
            df_shocks = df_shocks.sort_values(by="Close Range", ascending=False)
        elif sort_metric == "Days (Newest First)":
            df_shocks['Days_Int'] = df_shocks['Days'].str.extract(r'(\d+)').astype(float)
            df_shocks = df_shocks.sort_values(by="Days_Int", ascending=True)
            df_shocks = df_shocks.drop(columns=['Days_Int'])
        elif sort_metric == "Ticker (A-Z)":
            df_shocks = df_shocks.sort_values(by="Ticker", ascending=True)
        
        # World-Class HTML Rendering
        html_table = """
<style>
.kush-table { width: 100%; border-collapse: separate; border-spacing: 0 8px; font-family: 'Inter', sans-serif; }
.kush-table th { color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; text-align: left; padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.05); white-space: nowrap; }
.kush-row { background: rgba(255,255,255,0.03); transition: all 0.2s ease; border-radius: 8px; }
.kush-row:hover { background: rgba(255,255,255,0.06); transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
.kush-row td { padding: 14px 16px; font-size: 0.85rem; color: #f8fafc; border-top: 1px solid rgba(255,255,255,0.02); border-bottom: 1px solid rgba(255,255,255,0.02); white-space: nowrap; }
.kush-row td:first-child { border-left: 1px solid rgba(255,255,255,0.02); border-top-left-radius: 8px; border-bottom-left-radius: 8px; font-weight: 600; }
.kush-row td:last-child { border-right: 1px solid rgba(255,255,255,0.02); border-top-right-radius: 8px; border-bottom-right-radius: 8px; }

@keyframes pulse-vdu-green {
    0% { box-shadow: inset 4px 0 0 #22c55e, inset 0 0 10px rgba(34, 197, 94, 0.05); }
    50% { box-shadow: inset 4px 0 0 #22c55e, inset 0 0 30px rgba(34, 197, 94, 0.35); background: rgba(34, 197, 94, 0.1); }
    100% { box-shadow: inset 4px 0 0 #22c55e, inset 0 0 10px rgba(34, 197, 94, 0.05); }
}
@keyframes pulse-htf-orange {
    0% { box-shadow: inset 4px 0 0 #f97316, inset 0 0 10px rgba(249, 115, 22, 0.05); }
    50% { box-shadow: inset 4px 0 0 #f97316, inset 0 0 30px rgba(249, 115, 22, 0.35); background: rgba(249, 115, 22, 0.1); }
    100% { box-shadow: inset 4px 0 0 #f97316, inset 0 0 10px rgba(249, 115, 22, 0.05); }
}
.row-blink-green { animation: pulse-vdu-green 2s infinite !important; border-left: 4px solid #22c55e; }
.row-blink-orange { animation: pulse-htf-orange 1.5s infinite !important; border-left: 4px solid #f97316; }

.rs-bar-bg { background: rgba(255, 255, 255, 0.1); width: 60px; height: 6px; border-radius: 3px; overflow: hidden; display: inline-block; vertical-align: middle; margin-left: 8px; }
.rs-bar-fill { height: 100%; border-radius: 3px; }

.progress-bg { background: rgba(255, 255, 255, 0.1); width: 80px; height: 6px; border-radius: 3px; overflow: hidden; display: inline-block; vertical-align: middle; margin-right: 8px; }
.progress-fill { height: 100%; border-radius: 3px; background: #3b82f6; }
</style>
"""
        html_lines = [html_table]
        html_lines.append('<div style="overflow-x: auto; background-color: #0f172a; padding: 20px; border-radius: 16px; border: 1px solid #1e293b; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">')
        html_lines.append('<table class="kush-table">')
        html_lines.append('<thead><tr><th>Ticker</th><th>Shock Date</th><th>Days</th><th>Vol Mult</th><th>Last Price</th><th>Dist to High</th><th>Close Range</th><th>RS</th><th>Action</th><th>Volume Dry Up</th><th>HTF Warning</th></tr></thead>')
        html_lines.append('<tbody>')
        
        for _, r in df_shocks.iterrows():
            row_cls = "kush-row"
            vdu = str(r.get('VDU', ''))
            htf = str(r.get('HTF Warning', ''))
            
            if "TRIGGERED" in htf:
                row_cls += " row-blink-orange"
            elif "ACTIVE" in vdu:
                row_cls += " row-blink-green"
                
            rs = r.get('RS Score', 0)
            rs_color = "#22c55e" if rs >= 80 else "#38bdf8" if rs >= 60 else "#eab308" if rs >= 40 else "#ef4444"
            rs_html = f"<span style='color: {rs_color}; font-weight: 700; width: 20px; display: inline-block;'>{rs}</span> <div class='rs-bar-bg'><div class='rs-bar-fill' style='width: {rs}%; background-color: {rs_color};'></div></div>"
            
            cr = float(r.get('Close Range', 0))
            cr_html = f"<div class='progress-bg'><div class='progress-fill' style='width: {cr}%;'></div></div> {cr:.0f}%"
            
            ticker_url = r.get('Ticker', '')
            import re
            match = re.search(r'symbol=(?:NSE:)?(.*)', ticker_url)
            t_display = match.group(1) if match else "UNKNOWN"
            t_html = f"<a href='{ticker_url}' target='_blank' style='color: #38bdf8; text-decoration: none; font-weight: bold;'>{t_display}</a>"
            
            price_fmt = f"₹{r.get('Last Price', 0):.2f}" if market == "IN" else f"${r.get('Last Price', 0):.2f}"
            
            html_lines.append(f'<tr class="{row_cls}">')
            html_lines.append(f"<td>{t_html}</td>")
            html_lines.append(f"<td>{r.get('Shock Date', '')}</td>")
            html_lines.append(f"<td><span style='color: #94a3b8;'>{r.get('Days', '')}</span></td>")
            html_lines.append(f"<td><span style='color: #f97316; font-weight: 700;'>{r.get('Vol Mult', 0):.1f}x</span></td>")
            html_lines.append(f"<td>{price_fmt}</td>")
            html_lines.append(f"<td>{r.get('Dist to 52W High', 0):.1f}%</td>")
            html_lines.append(f"<td>{cr_html}</td>")
            html_lines.append(f"<td>{rs_html}</td>")
            html_lines.append(f"<td>{r.get('Behavior', '')}</td>")
            
            vdu_color = "#22c55e" if "ACTIVE" in vdu else "#94a3b8"
            html_lines.append(f"<td><span style='color: {vdu_color}; font-weight: 600;'>{vdu}</span></td>")
            
            htf_color = "#f97316" if "TRIGGERED" in htf else "#94a3b8"
            html_lines.append(f"<td><span style='color: {htf_color}; font-weight: 600;'>{htf}</span></td>")
            
            html_lines.append("</tr>")
            
        html_lines.append('</tbody></table></div>')
        st.markdown("\n".join(html_lines), unsafe_allow_html=True)
        
        st.write("")
        with st.expander("📊 TradingView Export"):
            import re
            def extract_tv_ticker(url):
                match = re.search(r'symbol=(.*)', url)
                return match.group(1) if match else ""
                
            export_tickers = [extract_tv_ticker(x) for x in df_shocks['Ticker'].tolist()]
            export_str = ",".join(export_tickers)
            if not export_str:
                export_str = "No tickers match the current filters."
            st.code(export_str, language="text")

with tab_in:
    render_institutional_footprints("IN")
    render_focus_table("IN")
    
with tab_us:
    render_institutional_footprints("US")
    render_focus_table("US")

# Fire batched notifications at the end of the script rendering
if st.session_state.batched_notifications and TOAST_AVAILABLE:
    try:
        msg = "\n".join(st.session_state.batched_notifications)
        toast('🔔 Kush Tracker Alerts', msg[:250])
    except Exception:
        pass
    st.session_state.batched_notifications = []

if auto_refresh:
    time.sleep(15)
    st.rerun()

