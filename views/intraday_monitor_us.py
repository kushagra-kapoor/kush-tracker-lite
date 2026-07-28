import streamlit as st

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from styles import load_css
    load_css()
except ImportError:
    pass

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
from io import StringIO
import pytz
import numpy as np
import concurrent.futures
import plotly.express as px
import plotly.graph_objects as go

# Local imports (US version does not use market_data for universe)

# Page config
def main():
    # st.set_page_config(page_title="US Intraday Monitor | Kush Tracker", page_icon="🗽", layout="wide")
    
    import styles
    import importlib
    importlib.reload(styles)
    styles.load_css()

# Custom CSS


# -----------------------------------------------------------------------------
# DATA FETCHING HELPERS
# -----------------------------------------------------------------------------

# Data fetching logic is routed via market_data.py

def fetch_yfinance_batch(tickers, days=252, force_today_refresh=False):
    """Fetch data for multiple tickers using the High-Speed Incremental Disk Cache."""
    from price_history_manager import fetch_incremental_history
    return fetch_incremental_history(tickers, days, force_today_refresh=force_today_refresh)

# -----------------------------------------------------------------------------
# METRIC CALCULATIONS
# -----------------------------------------------------------------------------

def process_historical_breadth(history_df, days=60):
    """
    Computes historical breadth, highs, distribution, and squats.
    Extremely robust to yfinance MultiIndex variations and case-sensitivity.
    """
    if history_df.empty:
        return pd.DataFrame()

    try:
        # Determine OHLCV level in MultiIndex
        if not isinstance(history_df.columns, pd.MultiIndex):
            return pd.DataFrame()

        ohlcv_level = -1
        for i in range(history_df.columns.nlevels):
            l_vals = [str(v).lower() for v in history_df.columns.get_level_values(i)]
            if 'close' in l_vals or 'adj close' in l_vals:
                ohlcv_level = i
                break
        
        if ohlcv_level == -1:
            return pd.DataFrame()

        def extract_all_for(attr):
            l_vals = [str(v).lower() for v in history_df.columns.get_level_values(ohlcv_level)]
            # Try direct match
            if attr.lower() in l_vals:
                key = history_df.columns.get_level_values(ohlcv_level)[l_vals.index(attr.lower())]
                return history_df.xs(key, level=ohlcv_level, axis=1)
            # Try adj close fallback for close
            if attr.lower() == 'close' and 'adj close' in l_vals:
                key = history_df.columns.get_level_values(ohlcv_level)[l_vals.index('adj close')]
                return history_df.xs(key, level=ohlcv_level, axis=1)
            return pd.DataFrame()

        close_panel = extract_all_for('close')
        high_panel = extract_all_for('high')
        low_panel = extract_all_for('low')
        vol_panel = extract_all_for('volume')

        if close_panel.empty or high_panel.empty:
            return pd.DataFrame()

        # Vectorized calculations for all dates
        # 1. Positive Breadth
        daily_rets = close_panel.pct_change()
        pos_counts = (daily_rets > 0).sum(axis=1)
        total_active = daily_rets.notna().sum(axis=1)
        breadth_pct = (pos_counts / total_active.replace(0, np.nan)) * 100

        # 2. 52W Highs
        rolling_max_252 = high_panel.rolling(window=252, min_periods=100).max().shift(1)
        is_new_high = (high_panel >= rolling_max_252)
        new_highs_count = is_new_high.sum(axis=1)

        # 3. Institutional Distribution
        near_high_20 = high_panel >= (0.80 * rolling_max_252)
        avg_vol_50 = vol_panel.rolling(window=50, min_periods=20).mean().shift(1)
        is_dist = (daily_rets <= -0.03) & (vol_panel >= 1.5 * avg_vol_50) & near_high_20
        dist_count = is_dist.sum(axis=1)

        # 4. Squats
        near_high_10 = high_panel >= (0.90 * rolling_max_252)
        prev_high = high_panel.shift(1)
        avg_vol_20 = vol_panel.rolling(window=20, min_periods=10).mean().shift(1)
        vol_exp = vol_panel / avg_vol_20
        is_breakout_try = (high_panel > prev_high) & (vol_exp >= 1.3)
        tried_up = (is_breakout_try | is_new_high) & near_high_10
        
        daily_range = high_panel - low_panel
        close_range = ((close_panel - low_panel) / daily_range.replace(0, np.nan)) * 100
        is_squat = tried_up & (close_range < 40.0)
        squat_count = is_squat.sum(axis=1)

        # Build Result DF
        result_df = pd.DataFrame({
            'Positive Breadth %': breadth_pct,
            'New 52W Highs': new_highs_count,
            'Distribution Count': dist_count,
            'Squat Count': squat_count
        })

        # Calculate MAs
        result_df['Breadth_MA10'] = result_df['Positive Breadth %'].rolling(window=10).mean()
        result_df['Highs_MA10'] = result_df['New 52W Highs'].rolling(window=10).mean()
        result_df['Dist_MA10'] = result_df['Distribution Count'].rolling(window=10).mean()
        result_df['Squat_MA10'] = result_df['Squat Count'].rolling(window=10).mean()
        
        result_df.index = pd.to_datetime(result_df.index)
        result_df['Date'] = result_df.index.date
        
        # Drop only the first row (the pct_change NaN) and keep the last 150 days to ensure enough for MA
        # then tail to the requested days
        result_df = result_df.iloc[1:].tail(days)
        return result_df

    except Exception as e:
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

from technical_indicators import detect_high_tight_flag, detect_ants_momentum, calculate_hv1_avwap, detect_ema_crossback, detect_reversal_extension, detect_power_trend

def process_intraday_data(history_df, tickers, index_data=None, industry_map=None):
    """
    Process the bulk historical dataframe to compute intraday metrics for all tickers.
    history_df is a MultiIndex dataframe if len(tickers) > 1, or single ticker df.
    """
    results = []
    
    # Handle case where yf.download returns single index (only 1 ticker succeeded/requested)
    if len(tickers) == 1 or not isinstance(history_df.columns, pd.MultiIndex):
        if not history_df.empty:
            t = tickers[0]
            # Convert to standard format to process below
            # We'll just wrap it in a dict-like structure to loop once
            ticker_data = {t: history_df}
        else:
            ticker_data = {}
    else:
        # Group by ticker level 0
        ticker_data = {t: history_df[t] for t in tickers if t in history_df.columns.get_level_values(0).unique()}

    market_state = {
        'green_light': False,
        'ema_10_slope': 0.0,
        'ema_20_slope': 0.0,
        'index_pct': 0.0
    }
    
    if index_data is not None and not index_data.empty and len(index_data) >= 5:
        # Handle cases where 'Close' might be a DataFrame containing a Series
        try:
            close_series = index_data['Close']
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
            
            idx_prev_close = float(close_series.iloc[-2])
            idx_curr = float(close_series.iloc[-1])
            
            if idx_prev_close > 0:
                market_state['index_pct'] = ((idx_curr - idx_prev_close) / idx_prev_close) * 100
                
            idx_ema_10 = close_series.ewm(span=10, adjust=False).mean()
            idx_ema_20 = close_series.ewm(span=20, adjust=False).mean()
            
            if len(idx_ema_10) >= 2:
                ema_10_val = idx_ema_10.iloc[-1]
                ema_20_val = idx_ema_20.iloc[-1]
                market_state['ema_10_slope'] = idx_ema_10.iloc[-1] - idx_ema_10.iloc[-2]
                market_state['ema_20_slope'] = idx_ema_20.iloc[-1] - idx_ema_20.iloc[-2]
                market_state['green_light'] = (ema_10_val > ema_20_val) and (market_state['ema_10_slope'] > 0) and (market_state['ema_20_slope'] > 0)
        except Exception:
            pass

    index_today_pct = market_state.get('index_pct', 0.0)

    from database import get_current_tml_leaders, get_latest_top_sectors, get_hat_stocks
    current_tml = get_current_tml_leaders("US")
    hat_stocks = get_hat_stocks("US")
    top_sectors = get_latest_top_sectors(5)

    for ticker, df in ticker_data.items():
        # Drop only if Close is missing. Yfinance often returns NaN for volume on Friday evenings
        df = df.dropna(subset=['Close']).copy()
        df['Volume'] = df['Volume'].fillna(0)
        if len(df) < 2:
            continue
        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        today_open = df['Open'].iloc[-1]
        today_high = df['High'].iloc[-1]
        today_low = df['Low'].iloc[-1]
        yesterday_high = df['High'].iloc[-2]
        today_volume = df['Volume'].iloc[-1]
        
        if prev_close <= 0 or today_high <= 0:
            continue
            
        # Today %
        today_pct = ((current_price - prev_close) / prev_close) * 100
        
        # 20D Avg Vol & Expansion
        avg_vol_20d = df['Volume'].iloc[-21:-1].mean() if len(df) >= 21 else df['Volume'].iloc[:-1].mean()
        vol_expansion = (today_volume / avg_vol_20d) if avg_vol_20d > 0 else 0
        
        # Dollar Volume ($M)
        dollar_volume_m = (avg_vol_20d * current_price) / 1000000
        
        # ADR% (Average Daily Range 20D)
        if len(df) >= 20:
            daily_ranges = ((df['High'].iloc[-20:] - df['Low'].iloc[-20:]) / df['Close'].iloc[-20:]) * 100
            adr_pct = daily_ranges.mean()
        else:
            adr_pct = 0.0
        
        # Distance from Day High
        dist_day_high = ((current_price - today_high) / today_high) * 100
        
        # Max Last 252D High (excluding today)
        lookback_df = df.iloc[-253:-1] if len(df) >= 253 else df.iloc[:-1]
        max_252d_high = lookback_df['High'].max() if not lookback_df.empty else today_high
        is_new_high = current_price >= max_252d_high and not lookback_df.empty
        
        # ---------------------------------------------------------------------
        # NEW: Green Line Breakout (GLB) Detector
        # A stock breaking above its 252-day high that was set AT LEAST 90 days ago.
        # ---------------------------------------------------------------------
        is_glb = False
        if is_new_high and not lookback_df.empty and len(lookback_df) >= 90:
            max_high_date = lookback_df['High'].idxmax()
            if max_high_date:
                max_high_date = pd.to_datetime(max_high_date)
                today_date = pd.to_datetime(df.index[-1])
                days_since_high = (today_date - max_high_date).days
                if days_since_high >= 90 and vol_expansion >= 1.5:
                    is_glb = True
        
        # Breakout % Above Yesterday High
        breakout_pct = ((current_price - yesterday_high) / yesterday_high) * 100 if current_price > yesterday_high else 0
        is_breakout = current_price > yesterday_high and vol_expansion >= 1.3
        
        # Episodic Pivot (Gap Up Catalyst)
        gap_pct = ((today_open - prev_close) / prev_close) * 100 if prev_close > 0 else 0
        is_ep = (gap_pct >= 4.0) and (vol_expansion >= 2.0) and (current_price >= today_open)
        
        # Close Range % (0 to 100%, where 100% = closed at exact high)
        daily_range = today_high - today_low
        close_range_pct = ((current_price - today_low) / daily_range * 100) if daily_range > 0 else 100.0
        
        # Squat Detector (Tried to break out but closed in bottom 40% of daily range)
        is_squat = (is_breakout or is_new_high) and (close_range_pct < 40.0)
        
        # 1M Return (21D)
        ret_1m = 0
        if len(df) >= 22:
            close_21d = df['Close'].iloc[-22]
            if close_21d > 0:
                ret_1m = ((current_price - close_21d) / close_21d) * 100
                
        # --- Stage 2 & Moving Average Clusters ---
        ema_10 = df['Close'].ewm(span=10, adjust=False, min_periods=5).mean().iloc[-1] if len(df) >= 10 else 0
        sma_21 = df['Close'].rolling(window=21, min_periods=10).mean().iloc[-1] if len(df) >= 21 else 0
        sma_50 = df['Close'].rolling(window=50, min_periods=20).mean().iloc[-1] if len(df) >= 50 else 0
        sma_200 = df['Close'].rolling(window=200, min_periods=50).mean().iloc[-1] if len(df) >= 200 else 0
        ema_65 = df['Close'].ewm(span=65, adjust=False, min_periods=20).mean().iloc[-1] if len(df) >= 65 else 0
        
        dist_ema10 = ((current_price - ema_10) / ema_10 * 100) if ema_10 > 0 else 100
        is_extended_10ema = dist_ema10 > 15.0
        
        dist_52w_high_breakout = ((max_252d_high - current_price) / max_252d_high * 100) if max_252d_high > 0 else 0
        
        # User defined relaxed conditions for Elite Breakout:
        # 1. Stage 2 Proxy: Price > 50 SMA > 200 SMA
        # 2. Within 15% of 52W High
        # 3. Up > 1% today
        # 4. Volume > 1.0x 20D Average
        is_stage_2 = (current_price > sma_50) and (sma_50 > sma_200) and (sma_200 > 0)
        is_elite_breakout = (is_stage_2) and (dist_52w_high_breakout <= 15.0) and (today_pct > 1.0) and (vol_expansion > 1.0)
                
        # 3M Return (63D)
        ret_3m = 0
        if len(df) >= 64:
            close_63d = df['Close'].iloc[-64]
            if close_63d > 0:
                ret_3m = ((current_price - close_63d) / close_63d) * 100

        # --- Deepvue Launch Pad (Early Entry Convergence) ---
        dist_21 = (abs(current_price - sma_21) / sma_21 * 100) if sma_21 > 0 else 100
        dist_50 = (abs(current_price - sma_50) / sma_50 * 100) if sma_50 > 0 else 100
        dist_65 = (abs(current_price - ema_65) / ema_65 * 100) if ema_65 > 0 else 100
        
        max_compression = max(dist_21, dist_50, dist_65)
        is_launchpad = (max_compression <= 3.5) and (sma_50 > sma_200) and (sma_200 > 0)

        # --- TraderLion HV1 (Highest Volume in 1 Year) ---
        max_vol_252d = lookback_df['Volume'].max() if not lookback_df.empty else 0
        vol_vs_1y_max = (today_volume / max_vol_252d) if max_vol_252d > 0 else 0
        is_hv1 = (today_volume > max_vol_252d) and (max_vol_252d > 0) and (close_range_pct >= 50) and (today_pct > 0)

        # --- High Tight Flag (Power Play) ---
        htf_data = detect_high_tight_flag(df, min_thrust_pct=70.0)
        is_htf = htf_data['is_htf']
        htf_thrust = htf_data['thrust_pct']
        htf_drawdown = htf_data['drawdown_pct']

        # --- Ants Momentum (Deepvue) ---
        is_ants = detect_ants_momentum(df, lookback=15, min_up_days=12)
        
        # --- HV1 AVWAP Defense ---
        hv1_avwap = calculate_hv1_avwap(df, lookback=252)
        is_avwap_defended = (hv1_avwap > 0) and (current_price > hv1_avwap) and (abs(current_price - hv1_avwap)/hv1_avwap < 0.05) # Within 5% of AVWAP
        
        # --- Oliver Kell Setups ---
        is_ema_crossback = is_stage_2 and detect_ema_crossback(df, ema_period=10, max_days_below=5)
        is_reversal_ext = detect_reversal_extension(df, ema_period=10, extension_threshold=12.0)
        
        # --- Deepvue Power Trend ---
        is_power_trend = detect_power_trend(df)

        # --- 50 SMA Reclaim (Key Level Recatch) ---
        low_3d = df['Low'].iloc[-3:].min() if len(df) >= 3 else current_price
        yest_close = df['Close'].iloc[-2] if len(df) >= 2 else current_price
        is_50_reclaim = (is_stage_2) and (low_3d < sma_50) and (current_price > sma_50) and (current_price > yest_close)

        # --- RideWinners 12-Day Trending Heatmap (Option A Logic) ---
        last_13 = df.iloc[-13:] if len(df) >= 13 else df
        heatmap_array = []
        for j in range(1, len(last_13)):
            is_green = (last_13['Close'].iloc[j] > last_13['Open'].iloc[j]) and \
                       (last_13['Close'].iloc[j] > last_13['Close'].iloc[j-1])
            heatmap_array.append(1 if is_green else 0)
            
        trend_score = sum(heatmap_array)
        heatmap_str = "".join(["🟩" if val == 1 else "⬛" for val in heatmap_array])

        # RS Intraday vs Index
        rs_intraday = today_pct - index_today_pct
        
        # RS Blue Dot (William O'Neil 🔵)
        is_blue_dot = False
        if index_data is not None and not index_data.empty:
            try:
                from relative_strength import calculate_rs_blue_dot
                rs_blue_dot_series = calculate_rs_blue_dot(df, index_data)
                is_blue_dot = bool(rs_blue_dot_series.iloc[-1]) if not rs_blue_dot_series.empty else False
            except Exception:
                is_blue_dot = False
        
        clean_ticker = ticker
        is_tml = clean_ticker in current_tml
        is_hat = clean_ticker in hat_stocks
        ticker_display = f"{clean_ticker}👑" if is_tml else (f"{clean_ticker}🎩" if is_hat else clean_ticker)
        if is_blue_dot:
            ticker_display += "🔵"
        
        industry = industry_map.get(ticker, 'Unknown') if industry_map else 'Unknown'
        is_hot_sector = industry in top_sectors
        if is_hot_sector and industry != 'Unknown':
            ticker_display += "🔥"
            
        results.append({
            'Ticker': ticker_display,
            'TickerLink': f"https://www.tradingview.com/chart/?symbol={clean_ticker}&name={ticker_display}",
            'Industry': industry,
            'Today %': today_pct,
            'Volume Expansion': vol_expansion,
            'Dollar Volume ($M)': dollar_volume_m,
            'ADR %': adr_pct,
            'Dist from 10 EMA %': dist_ema10,
            'Is Extended': is_extended_10ema,
            'Dist from Day High %': dist_day_high,
            '1M Return': ret_1m,
            '3M Return': ret_3m,
            'Is New High': is_new_high,
            'Breakout %': breakout_pct,
            'Is Breakout': is_breakout,
            'Is Elite Breakout': is_elite_breakout,
            'Is GLB': is_glb,
            'Close Range %': close_range_pct,
            'Is Squat': is_squat,
            'Gap %': gap_pct,
            'Is EP': is_ep,
            'Launchpad Compression %': max_compression,
            'Is Launchpad': is_launchpad,
            'Vol vs 1Y Max': vol_vs_1y_max,
            'Is HV1': is_hv1,
            '12D Trend Score': trend_score,
            '12D Heatmap': heatmap_str,
            'RS Intraday': rs_intraday,
            'Today Volume': today_volume,
            'Has Positive Return': today_pct > 0,
            'Has Negative Return': today_pct < 0,
            'Above Prev Close': current_price > prev_close,
            'Is HTF': is_htf,
            'HTF Thrust %': htf_thrust,
            'HTF Drawdown %': htf_drawdown,
            'Is Ants': is_ants,
            'HV1 AVWAP': hv1_avwap,
            'HV1 Defended': is_avwap_defended,
            'Is EMA Crossback': is_ema_crossback,
            'Is Reversal Ext': is_reversal_ext,
            'Is Power Trend': is_power_trend,
            'Is 50 Reclaim': is_50_reclaim,
            'Dist from 50 SMA %': dist_50,
            'Is RS Blue Dot': is_blue_dot
        })
        
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        # Cross-sectional percentiles for absolute momentum ranking
        df_res['1M Ret Pctile'] = df_res['1M Return'].rank(pct=True, ascending=True) * 100
        df_res['3M Ret Pctile'] = df_res['3M Return'].rank(pct=True, ascending=True) * 100
        
    return df_res, market_state

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

def main():
    # Header with Refresh button
    col1, col2, col3 = st.columns([2, 1, 1], vertical_alignment="center")
    
    with col1:
        from components import render_header
        render_header("🇺🇸 Intraday Monitor US", "Live intelligence for US Equities")
        
    with col2:
        if st.button("🔄 Refresh Snapshot", use_container_width=True, type="primary"):
            st.session_state['us_refresh_trigger'] = True
    
    with col3:
        last_updated = st.session_state.get('us_last_updated', 'Never')
        st.markdown(f"<div style='text-align: right; color: #94a3b8;'>Data Updated:<br><strong style='color: #f59e0b;'>{last_updated}</strong></div>", unsafe_allow_html=True)
        
    # -------------------------------------------------------------------------
    # MACRO SENTIMENT DASHBOARD
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🧭 Macro Sentiment & Exposure")
    import sentiment_engine as se
    fg_val, fg_desc = se.fetch_fear_and_greed()
    us_vix_val = se.get_us_vix()
    us_pcr_val = se.get_us_pcr()
    
    mac_col1, mac_col2 = st.columns(2)
    
    with mac_col1:
        with st.container(border=True):
            fig_fg = se.create_gauge(fg_val, "CNN Fear & Greed", 0, 100, [35, 65], inverse_colors=False, valueformat=".0f")
            st.plotly_chart(fig_fg, use_container_width=True, key="us_mac_fg")
            if fg_val is not None:
                if fg_val < 35:
                    st.markdown(f"<p style='text-align: center; color: #10b981; font-size: 0.95rem; margin-top:-10px;'>🟢 <b>{fg_desc}:</b> Contrarian opportunity zone. Buy the panic.</p>", unsafe_allow_html=True)
                elif fg_val > 65:
                    st.markdown(f"<p style='text-align: center; color: #ef4444; font-size: 0.95rem; margin-top:-10px;'>🔴 <b>{fg_desc}:</b> Caution, market overextended. Trim.</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='text-align: center; color: #e2e8f0; font-size: 0.95rem; margin-top:-10px;'>🟡 <b>{fg_desc}:</b> No extreme sentiment edge. Follow price action.</p>", unsafe_allow_html=True)
                
    with mac_col2:
        with st.container(border=True):
            fig_vix = se.create_gauge(us_vix_val, "US VIX (Volatility)", 10, 40, [15, 25], inverse_colors=False, valueformat=".1f")
            st.plotly_chart(fig_vix, use_container_width=True, key="us_mac_vix")
            if us_vix_val is not None:
                if us_vix_val < 15:
                    st.markdown("<p style='text-align: center; color: #10b981; font-size: 0.95rem; margin-top:-10px;'>🟢 <b>Low Volatility:</b> Complacent market. Breakouts stick.</p>", unsafe_allow_html=True)
                elif us_vix_val > 25:
                    st.markdown("<p style='text-align: center; color: #ef4444; font-size: 0.95rem; margin-top:-10px;'>🔴 <b>High Volatility:</b> Fear and sharp moves. Whipsaws likely.</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='text-align: center; color: #e2e8f0; font-size: 0.95rem; margin-top:-10px;'>🟡 <b>Normal Volatility:</b> Average market conditions.</p>", unsafe_allow_html=True)
    
    # Focus List quick pin
    st.markdown("---")
    pin_cols = st.columns([2, 1, 3])
    with pin_cols[0]:
        pin_ticker = st.text_input("⭐ Pin Ticker to Focus List (e.g. AAPL):", key="pin_intra_us_text")
    with pin_cols[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Pin to Focus List", key="btn_pin_intra_us", use_container_width=True):
            if pin_ticker:
                from database import add_to_focus_list
                if add_to_focus_list(pin_ticker.upper().strip(), "US"):
                    st.success(f"Pinned {pin_ticker.upper()}!")
                else:
                    st.error("Failed to pin.")
    st.markdown("---")
    
    from database import get_latest_top_sectors, get_market_regime
    regime = get_market_regime()
    st.markdown(f"### 🔭 Scanner Configuration | Market Regime: {regime}")
    
    top_sectors = get_latest_top_sectors(5)
    if not top_sectors:
        st.warning("⚠️ **Hot Sectors Not Cached:** Please run a scan on the Sector Leadership page to enable 🔥 Hot Sector badges on your intraday charts.")
    else:
        st.info(f"💡 **Symbol Guide:** 👑 = Top 20 True Market Leader &nbsp;&nbsp;|&nbsp;&nbsp; 🎩 = Next 20 Leader (RS > 85) &nbsp;&nbsp;|&nbsp;&nbsp; 🔥 = Hot Sector ({', '.join(top_sectors)})")
        
    st.caption("Universe: US Large/Mid-Cap stocks from `tickers_us.txt`")
    
    with st.expander("ℹ️ Understanding Institutional Footprints (Power Trend, Ants, HV1)"):
        st.markdown("""
        **Institutional Footprints** indicate heavy, sustained accumulation by large funds. These metrics help filter out noisy, low-quality breakouts from true market leaders.

        *   ⚡ **Power Trend (Deepvue / O'Neil)**: A strict, multi-timeframe uptrend regime. Price > 21 EMA; 21 EMA > 50 SMA for at least 20 days; 50 SMA strictly sloping up. Shows unshakeable momentum.
        *   🐜 **Ants Momentum (TraderLion)**: Aggressive momentum where a stock closes higher on 12 out of 15 consecutive days. Shows relentless institutional buying that ignores minor market pullbacks.
        *   🛡️ **HV1 Defender (TraderLion)**: Price is within 5% of the Anchored VWAP drawn from its **Highest Volume Day in 1 Year**. Institutions fiercely defend their largest cost basis, making this a powerful support level.
        """)

    # State management to prevent reloading on every UI interaction
    if 'us_results_df' not in st.session_state:
        st.session_state['us_results_df'] = pd.DataFrame()
    if 'batched_notifications' not in st.session_state:
        st.session_state['batched_notifications'] = []
        # Automatically trigger the refresh on first load
        st.session_state['us_refresh_trigger'] = True
        
    # Execute fetch if triggered
    if st.session_state.get('us_refresh_trigger', False):
        st.session_state['us_refresh_trigger'] = False # Reset
        
        with st.spinner("Loading US Universe from tickers_us.txt..."):
            import os as _os
            file_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), 'tickers_us.txt')
            with open(file_path, 'r') as f:
                tickers = [line.strip().upper() for line in f if line.strip() and '-' not in line]
            tickers = list(set(tickers))
            # Industry map from DB cache
            from database import get_all_fundamentals_cache
            db_cache = get_all_fundamentals_cache()
            ind_map = {t: db_cache.get(t, {}).get('industry', 'Unknown') for t in tickers}
            
        with st.spinner(f"Downloading historical & live data for {len(tickers)} US stocks (~30-60s)..."):
            history_df = fetch_yfinance_batch(tickers, days=252, force_today_refresh=True)
            # Fetch S&P 500 for benchmark (1y to compute RS Blue Dot 52W breakouts)
            index_df = yf.download('^GSPC', period="1y", progress=False)
            
        with st.spinner("Computing Intraday Metrics..."):
            results_df, market_state = process_intraday_data(history_df, tickers, index_df, ind_map)
            breadth_history_df = process_historical_breadth(history_df, days=60)
            
            # --- AUTO-BACKFILL ENGINE ---
            if len(tickers) == 1 or not isinstance(history_df.columns, pd.MultiIndex):
                ticker_data = {tickers[0]: history_df} if not history_df.empty else {}
            else:
                ticker_data = {t: history_df[t] for t in tickers if t in history_df.columns.get_level_values(0).unique()}
            
            from database import auto_backfill_footprints
            backfilled = auto_backfill_footprints(ticker_data, "USA", lookback_days=30)
            if backfilled > 0:
                st.session_state.batched_notifications.append(f"🔄 US Auto-Backfill: Found & restored {backfilled} historical footprints!")
            
            # Extract and save signals
            signals_to_save = []
            for _, row in results_df.iterrows():
                t = str(row['Ticker']).replace('👑', '').replace('🔥', '').strip()
                if row.get('Is Elite Breakout', False): signals_to_save.append({'ticker': t, 'signal_name': 'Stage 2'})
                if row.get('Is Launchpad', False): signals_to_save.append({'ticker': t, 'signal_name': 'Launchpad'})
                if row.get('Is GLB', False): signals_to_save.append({'ticker': t, 'signal_name': 'GLB Breakout'})
                if row.get('Is 3WT', False): 
                    signals_to_save.append({'ticker': t, 'signal_name': '3WT'})
                    signals_to_save.append({'ticker': t, 'signal_name': 'Ryan'})
                if row.get('Is EP', False): signals_to_save.append({'ticker': t, 'signal_name': 'EP'})
                if row.get('Is HV1', False): signals_to_save.append({'ticker': t, 'signal_name': 'HV1'})
                if row.get('Is New High', False): signals_to_save.append({'ticker': t, 'signal_name': '52W High'})
                
                # Apex Screener conditions
                is_apex = (row.get('Dollar Volume ($M)', 0) >= 20.0) and \
                          (row.get('ADR %', 0) >= 4.0) and \
                          (row.get('1M Return', 0) >= 40.0 or row.get('1M Ret Pctile', 0) >= 90.0 or row.get('3M Ret Pctile', 0) >= 90.0) and \
                          (not row.get('Is Extended', False)) and \
                          (row.get('Is EP', False) or row.get('Is Breakout', False) or row.get('Is Launchpad', False) or row.get('Is GLB', False))
                if is_apex: signals_to_save.append({'ticker': t, 'signal_name': 'Apex'})
                
            if signals_to_save:
                from database import save_intraday_signals
                save_intraday_signals("US", signals_to_save)
                
            # ---- LOG INSTITUTIONAL FOOTPRINTS (EPISODIC PIVOTS) TO DATABASE ----
            for _, row in results_df.iterrows():
                t = str(row['Ticker']).replace('👑', '').replace('🔥', '').strip()
                vol_exp = row.get('Volume Expansion', 0)
                today_pct = row.get('Today %', 0)
                dollar_vol = row.get('Dollar Volume ($M)', 0)
                
                # Criteria: >5x volume, >4.8% up, and Dollar Volume > $5M
                if vol_exp >= 5.0 and today_pct >= 4.8 and dollar_vol >= 5.0:
                    shock_date = datetime.now().strftime('%Y-%m-%d')
                    from database import log_volume_shock
                    log_volume_shock(t, shock_date, float(vol_exp), float(row.get('Close', 0)), float(row.get('Today High', 0)), float(row.get('Today Low', 0)), "USA")
            
            # Store in session state (US-prefixed keys to avoid collision)
            st.session_state['us_results_df'] = results_df
            st.session_state['us_breadth_history_df'] = breadth_history_df
            st.session_state['us_market_state'] = market_state
            st.session_state['us_index_pct'] = market_state.get('index_pct', 0.0)
            
            # Update timestamp
            local_tz = pytz.timezone('US/Eastern')
            now_et = datetime.now(local_tz)
            st.session_state['us_last_updated'] = now_et.strftime('%d %b, %H:%M:%S ET')
            st.rerun()

    results_df = st.session_state.get('us_results_df')
    
    if results_df.empty:
        st.info("👆 Click **Refresh Snapshot** to fetch live intraday data.")
        return
        
    # --- RENDER DASHBOARD ---
    
    # helper for styling
    def format_pct(val):
        return f"{val:+.2f}%" if pd.notnull(val) else "N/A"
        
    def val_color(val):
        if val > 0: return 'green-text'
        if val < 0: return 'red-text'
        return ''
        
    # Helper for TradingView exports
    def generate_tv_export(df, ticker_col='TickerLink'):
        import re
        # Extract base ticker from URL or direct list
        if df.empty:
            return ""
        if 'Ticker' in df.columns:
            tickers = [re.sub(r'[^\w\.\-\_]', '', str(t)).strip() for t in df['Ticker']]
        else:
            # Fallback if only link is present
            tickers = [re.sub(r'[^\w\.\-\_]', '', str(t)).strip() for t in df[ticker_col].str.extract(r"symbol=([^&]*)")[0]]
        return ",".join([t for t in tickers if t])

    def style_dataframe(df):
        df = df.copy()
        # Round all float columns to 2 decimals to prevent ugly unformatted 6-decimal renders
        for col in df.select_dtypes(include=['float', 'float64']).columns:
            df[col] = df[col].round(2)
            
        # Applies standard red/green text styling to percentage columns
        pct_cols = [c for c in df.columns if '%' in c or 'Ret' in c or 'Breakout' in c or 'Intraday' in c]
        
        def color_negative_red(val):
            try:
                if val < 0: return 'color: #ef4444'
                if val > 0: return 'color: #10b981'
            except:
                pass
            return ''
            
        def highlight_intraday_rows(row):
            # Check for TML
            is_tml = '👑' in str(row.get('Ticker', row.get('TickerLink', '')))
            
            # Check for Volume Expansion / Vol Multiplier
            vol_mult = row.get('Volume Expansion', row.get('Vol Multiplier', 0))
            is_bo = False
            if 'Setup Trigger' in row and isinstance(row['Setup Trigger'], str) and 'BO' in row['Setup Trigger']:
                is_bo = True
                
            if pd.notna(vol_mult) and float(vol_mult) > 5.0:
                return ['background-color: rgba(217, 70, 239, 0.15); color: #e879f9'] * len(row)
            elif is_bo and pd.notna(vol_mult) and float(vol_mult) > 2.0:
                return ['background-color: rgba(56, 189, 248, 0.15); color: #7dd3fc'] * len(row)
            elif is_tml:
                return ['background-color: rgba(234, 179, 8, 0.15); color: #facc15'] * len(row)
            return [''] * len(row)
            
        return df.style.map(color_negative_red, subset=pct_cols).apply(highlight_intraday_rows, axis=1)

    # Ensure market_state exists from earlier execution
    market_state = st.session_state.get('us_market_state', {})
    qull_green = market_state.get('green_light', False)
    ema_10_sl = market_state.get('ema_10_slope', 0.0)
    ema_20_sl = market_state.get('ema_20_slope', 0.0)

    # --- QULLAMAGGIE MARKET FILTER BANNER ---
    st.markdown("### 🚦 The Market Filter (S&P 500 Regime)")
    if qull_green:
        banner_color = "#10b981"
        banner_icon = "🟢"
        banner_title = "GREEN LIGHT: CLEAR TO TRADE BREAKOUTS"
        banner_desc = f"Index 10 EMA > 20 EMA. Both moving averages are sloping upwards. (10Slope: {ema_10_sl:+.2f}, 20Slope: {ema_20_sl:+.2f})"
    else:
        if ema_10_sl < 0 and ema_20_sl < 0:
            banner_color = "#ef4444"
            banner_icon = "🔴"
            banner_title = "RED LIGHT: DO NOT BUY BREAKOUTS"
            banner_desc = f"Both MAs are sloping downwards! Avoid long setups. (10Slope: {ema_10_sl:+.2f}, 20Slope: {ema_20_sl:+.2f})"
        else:
            banner_color = "#f59e0b"
            banner_icon = "🟡"
            banner_title = "CAUTION: TREND IS WEAKENING OR MIXED"
            banner_desc = f"Index 10 EMA vs 20 EMA is unaligned. Reduce size massively. (10Slope: {ema_10_sl:+.2f}, 20Slope: {ema_20_sl:+.2f})"

    st.markdown(f"""
    <div style="background-color: {banner_color}20; border: 2px solid {banner_color}; border-radius: 8px; padding: 15px; margin-bottom: 25px;">
        <h4 style="color: {banner_color}; margin: 0; padding: 0;">{banner_icon} {banner_title}</h4>
        <p style="margin: 5px 0 0 0; color: #cbd5e1;">{banner_desc}</p>
    </div>
    """, unsafe_allow_html=True)

    # SEC 1: MARKET BREADTH SNAPSHOT
    st.info("💡 **Color & Symbol Guide:** 🟪 **Purple** = Volume Spike (>5x) | 🟦 **Blue** = Breakout (>2x Vol) | 🟩 **Green** = GLB Breakout | 🟨 **Golden** = True Market Leader (TML) | 👑 **Apex Predator** | ⚡ **Super Compounder** | 🚀 **Rocket Rank**")
    st.markdown("### 📊 SEC 1: Market Breadth Snapshot")
    
    total_stocks = len(results_df)
    positive_count = results_df['Has Positive Return'].sum()
    negative_count = results_df['Has Negative Return'].sum()
    positive_pct = (positive_count / total_stocks * 100) if total_stocks > 0 else 0
    negative_pct = (negative_count / total_stocks * 100) if total_stocks > 0 else 0
    avg_return = results_df['Today %'].mean()
    new_highs_count = results_df['Is New High'].sum()
    
    # Interpret Context
    if positive_pct >= 70:
        context_msg = "🟢 <strong>Strong Bullish Breadth:</strong> Broad market participation with the vast majority of stocks advancing."
        glow_color = "rgba(16, 185, 129, 0.15)" # Green glow
    elif positive_pct >= 55:
        context_msg = "🟢 <strong>Positive Breadth:</strong> Slight bullish tilt, buyers are in control."
        glow_color = "rgba(16, 185, 129, 0.05)"
    elif positive_pct >= 45:
        context_msg = "🟡 <strong>Neutral / Mixed Breadth:</strong> Market is finding balance, no clear directional dominance intraday."
        glow_color = "rgba(245, 158, 11, 0.1)" # Amber glow
    elif positive_pct >= 30:
        context_msg = "🔴 <strong>Negative Breadth:</strong> Sellers are in control, broad market is sliding."
        glow_color = "rgba(239, 68, 68, 0.1)" # Red glow
    else:
        context_msg = "🔴 <strong>Severe Bearish Breadth:</strong> Heavy distribution across the total market."
        glow_color = "rgba(239, 68, 68, 0.2)"
        
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, #0f172a 0%, #020617 100%); border-radius: 20px; padding: 2.5rem; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7); border: 1px solid rgba(255, 255, 255, 0.05); position: relative; overflow: hidden; margin-bottom: 2rem;">
        <div style="position: absolute; top: -50%; left: -10%; width: 60%; height: 200%; background: radial-gradient(circle, {glow_color} 0%, transparent 70%); transform: rotate(-15deg); pointer-events: none;"></div>
        <h3 style="color: #f8fafc; margin: 0 0 1.5rem 0; font-family: 'Inter', sans-serif; font-size: 1.5rem; display: flex; align-items: center; gap: 0.75rem;">📊 Market Breadth Snapshot</h3>
        <div style="display: flex; gap: 1.5rem; flex-wrap: wrap;">
            <div style="background: rgba(0,0,0,0.4); padding: 1.5rem; border-radius: 14px; border: 1px solid rgba(255,255,255,0.05); flex: 1; min-width: 200px; text-align: center; transition: transform 0.2s;">
                <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 0.5rem;">Positive %</div>
                <div style="color: #10b981; font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 800; text-shadow: 0 0 20px rgba(16,185,129,0.4);">{positive_pct:.1f}%</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">({positive_count}/{total_stocks} stocks)</div>
            </div>
            <div style="background: rgba(0,0,0,0.4); padding: 1.5rem; border-radius: 14px; border: 1px solid rgba(255,255,255,0.05); flex: 1; min-width: 200px; text-align: center; transition: transform 0.2s;">
                <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 0.5rem;">Negative %</div>
                <div style="color: #ef4444; font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 800; text-shadow: 0 0 20px rgba(239,68,68,0.4);">{negative_pct:.1f}%</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">({negative_count}/{total_stocks} stocks)</div>
            </div>
            <div style="background: rgba(0,0,0,0.4); padding: 1.5rem; border-radius: 14px; border: 1px solid rgba(255,255,255,0.05); flex: 1; min-width: 200px; text-align: center; transition: transform 0.2s;">
                <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 0.5rem;">Avg Universe Return</div>
                <div style="color: {'#10b981' if avg_return > 0 else '#ef4444'}; font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 800; text-shadow: 0 0 20px {'rgba(16,185,129,0.4)' if avg_return > 0 else 'rgba(239,68,68,0.4)'};">{avg_return:+.2f}%</div>
            </div>
            <div style="background: rgba(0,0,0,0.4); padding: 1.5rem; border-radius: 14px; border: 1px solid rgba(245, 158, 11, 0.3); flex: 1; min-width: 200px; text-align: center; transition: transform 0.2s; box-shadow: inset 0 0 20px rgba(245,158,11,0.05);">
                <div style="color: #f59e0b; font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 0.5rem;">New 52W Highs</div>
                <div style="color: #fcd34d; font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 800; text-shadow: 0 0 20px rgba(245,158,11,0.4);">{new_highs_count}</div>
            </div>
        </div>
        <div style="margin-top: 2rem; padding: 1rem 1.5rem; border-radius: 12px; background: rgba(255,255,255,0.03); border-left: 4px solid {'#10b981' if positive_pct >= 55 else '#ef4444' if positive_pct < 45 else '#f59e0b'}; font-size: 0.95rem; color: #e2e8f0; letter-spacing: 0.3px;">
            {context_msg}
        </div>
    </div>
    """, unsafe_allow_html=True)
        
    # SEC 1.5: SECTOR / INDUSTRY HEATMAP
    st.markdown("---")
    st.markdown("### 🗺️ SEC 2: Institutional Sector Heatmap")
    st.caption("Aggregates the momentum of all US universe components to identify macro thematic leadership.")
    
    if 'Industry' in results_df.columns:
        industry_gb = results_df.groupby('Industry').agg(
            Stocks_Count=('Ticker', 'count'),
            Avg_Today_Pct=('Today %', 'mean'),
            Avg_1M_Ret=('1M Return', 'mean'),
            Positive_Pct=('Has Positive Return', lambda x: x.mean() * 100)
        )
        # Filter for sectors with at least 3 components to prevent skew
        valid_sectors = industry_gb[industry_gb['Stocks_Count'] >= 3].copy()
        
        if not valid_sectors.empty:
            valid_sectors = valid_sectors.sort_values('Avg_Today_Pct', ascending=False).reset_index()
            
            top_5 = valid_sectors.head(5)
            bottom_3 = valid_sectors.tail(3)
            combine_bar = pd.concat([top_5, bottom_3])
            
            # Helper for Plotly aesthetics
            def apply_institutional_theme(fig):
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
                    margin=dict(l=0, r=0, t=40, b=0),
                    hovermode="x unified",
                    hoverlabel=dict(bgcolor="rgba(15,23,42,0.9)", font_size=13, font_family="JetBrains Mono")
                )
                fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linecolor="rgba(255,255,255,0.1)", tickfont=dict(color="#64748b"))
                fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)", zeroline=True, zerolinecolor="rgba(255,255,255,0.1)", showline=False, tickfont=dict(color="#64748b"))
                return fig

            sc1, sc2 = st.columns([6, 4])
            with sc1:
                with st.container(border=True):
                    fig_sector = px.bar(
                        combine_bar,
                        x='Industry',
                        y='Avg_Today_Pct',
                        color='Avg_Today_Pct',
                        title="🔥 Hottest (Top 5) & Weakest (Bottom 3) Sectors",
                        color_continuous_scale=['#ef4444', '#f59e0b', '#10b981'],
                        color_continuous_midpoint=0,
                        text_auto='.2f'
                    )
                    fig_sector = apply_institutional_theme(fig_sector)
                    fig_sector.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(fig_sector, use_container_width=True)
                
            with sc2:
                with st.container(border=True):
                    st.markdown("#### 📊 Sector Breadth Rankings")
                    top_10 = valid_sectors.head(10).copy()
                    st.dataframe(
                        top_10[['Industry', 'Avg_Today_Pct', 'Positive_Pct', 'Stocks_Count']].style.format({
                            "Avg_Today_Pct": "{:+.2f}%", 
                            "Positive_Pct": "{:.0f}%"
                        }).bar(subset=["Avg_Today_Pct"], align="mid", color=['#ef4444', '#10b981']),
                        use_container_width=True,
                        hide_index=True,
                        height=350
                    )
    
    st.markdown("---")

    # Historical Breadth Charts
    breadth_history_df = st.session_state.get('us_breadth_history_df', pd.DataFrame())
    if not breadth_history_df.empty:
        st.markdown("### 📈 SEC 3: Historical Breadth (Last 60 Days)")
        
        hc_col1, hc_col2 = st.columns(2)
        
        with hc_col1:
            with st.container(border=True):
                fig_breadth = px.bar(
                    breadth_history_df, 
                    x='Date', 
                    y='Positive Breadth %',
                    title="Positive Breadth % (Stocks > 0)",
                    color='Positive Breadth %',
                    color_continuous_scale=['#ef4444', '#f59e0b', '#10b981'],
                    range_color=[30, 70]
                )
                if 'Breadth_MA10' in breadth_history_df.columns:
                    fig_breadth.add_trace(go.Scatter(x=breadth_history_df['Date'], y=breadth_history_df['Breadth_MA10'], mode='lines', name='10 SMA', line=dict(color='#3b82f6', width=2)))
                fig_breadth.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.2)", opacity=0.8)
                fig_breadth = apply_institutional_theme(fig_breadth)
                fig_breadth.update_layout(height=280, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_breadth, use_container_width=True)
                st.caption("**Condition:** Daily % of US universe stocks closing positive. **Use:** Gauge broad market health. **Line:** 10 DMA.")
            
        with hc_col2:
            with st.container(border=True):
                fig_highs = px.bar(
                    breadth_history_df, 
                    x='Date', 
                    y='New 52W Highs',
                    title="New 52-Week Highs Count",
                    color_discrete_sequence=['#f59e0b']
                )
                if 'Highs_MA10' in breadth_history_df.columns:
                    fig_highs.add_trace(go.Scatter(x=breadth_history_df['Date'], y=breadth_history_df['Highs_MA10'], mode='lines', name='10 SMA', line=dict(color='#3b82f6', width=2)))
                fig_highs = apply_institutional_theme(fig_highs)
                fig_highs.update_layout(height=280, showlegend=False)
                st.plotly_chart(fig_highs, use_container_width=True)
                st.caption("**Condition:** Daily Count of stocks hitting 252-day high. **Use:** Identify leadership expansion. **Line:** 10 DMA.")
            
        hc_col3, hc_col4 = st.columns(2)
        
        with hc_col3:
            with st.container(border=True):
                fig_dist = px.bar(
                    breadth_history_df, 
                    x='Date', 
                    y='Distribution Count',
                    title="Historical Institutional Distribution (Selling Pressure)",
                    color_discrete_sequence=['#ef4444']
                )
                if 'Dist_MA10' in breadth_history_df.columns:
                    fig_dist.add_trace(go.Scatter(x=breadth_history_df['Date'], y=breadth_history_df['Dist_MA10'], mode='lines', name='10 SMA', line=dict(color='#3b82f6', width=2)))
                fig_dist = apply_institutional_theme(fig_dist)
                fig_dist.update_layout(height=280, showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True)
                st.caption("**Condition:** Stocks within 20% of 52W High closing down >3% on 1.5x Vol. **Use:** Detect early institutional exit from leaders. **Line:** 10 DMA.")
            
        with hc_col4:
            with st.container(border=True):
                fig_squat = px.bar(
                    breadth_history_df, 
                    x='Date', 
                    y='Squat Count',
                    title="Historical Breakout Failures (Squat Detector)",
                    color_discrete_sequence=['#8b949e']
                )
                if 'Squat_MA10' in breadth_history_df.columns:
                    fig_squat.add_trace(go.Scatter(x=breadth_history_df['Date'], y=breadth_history_df['Squat_MA10'], mode='lines', name='10 SMA', line=dict(color='#3b82f6', width=2)))
                fig_squat = apply_institutional_theme(fig_squat)
                fig_squat.update_layout(height=280, showlegend=False)
                st.plotly_chart(fig_squat, use_container_width=True)
                st.caption("**Condition:** Stocks within 10% of 52W High attempting breakout but closing in bottom 40% of range. **Use:** Gauge quality of current breakouts. **Line:** 10 DMA.")
        
        st.markdown("---")

    # Add spacing using Streamlit
    st.write("") 

    # --- MODULE B: ALPHA EXECUTION (THE MONEY ZONES) ---

    # SEC 4: THE QULLAMAGGIE SCREENER (ELITE MOMENTUM BREAKOUTS)
    st.markdown("---")
    st.markdown("### 🧙‍♂️ SEC 4: The Apex Screener (Qullamaggie Strict Rules)")
    st.caption("Matches institutional momentum rules: Dollar Volume > $10M, ADR > 4%, 1M Ret > 40% (or >90th Percentile 1M/3M), and Not Extended (>15% above 10EMA). Setup trigger: EP, Breakout, or Launchpad.")
    
    # Apply strict Qullamaggie rules
    qull_df = results_df[
        (results_df['Dollar Volume ($M)'] >= 10.0) &
        (results_df['ADR %'] >= 4.0) &
        ((results_df['1M Return'] >= 40.0) | (results_df.get('1M Ret Pctile', 0) >= 90.0) | (results_df.get('3M Ret Pctile', 0) >= 90.0)) &
        (~results_df['Is Extended']) &
        (results_df['Is EP'] | results_df['Is Breakout'] | results_df['Is Launchpad'] | results_df['Is GLB'])
    ]
    
    if qull_df.empty:
        st.info("No stocks meet the strict Qullamaggie criteria today. **Sit on your hands!** Cash is a position.")
    else:
        # Map trigger strings for display
        def get_qull_trigger(row):
            triggers = []
            if row['Is EP']: triggers.append("EP")
            if row['Is Breakout']: triggers.append("BO")
            if row['Is Launchpad']: triggers.append("Launchpad")
            if row['Is GLB']: triggers.append("GLB")
            return " + ".join(triggers)
            
        display_qull = qull_df.copy()
        display_qull['Setup Trigger'] = display_qull.apply(get_qull_trigger, axis=1)
        
        display_qull = display_qull[['Ticker', 'TickerLink', 'Setup Trigger', 'Today %', 'Dollar Volume ($M)', 'ADR %', '1M Return', '1M Ret Pctile', '3M Return', '3M Ret Pctile', 'Dist from 10 EMA %', 'Is Power Trend', 'Is Ants', 'HV1 Defended']].copy()
        display_qull = display_qull.sort_values(by='1M Ret Pctile', ascending=False)
        
        tv_text_qull = generate_tv_export(display_qull)
        with st.expander("📋 Copy TradingView List (Qullamaggie Setups)"):
            st.code(tv_text_qull, language="text")
            
        styled_qull = style_dataframe(display_qull[['TickerLink', 'Setup Trigger', 'Today %', 'Dollar Volume ($M)', 'ADR %', '1M Return', '1M Ret Pctile', '3M Return', '3M Ret Pctile', 'Dist from 10 EMA %', 'Is Power Trend', 'Is Ants', 'HV1 Defended']])
        
        st.dataframe(
            styled_qull,
            column_config={
                "TickerLink": st.column_config.LinkColumn("Ticker", display_text=r"name=(.*)", width="small"),
                "Setup Trigger": st.column_config.TextColumn("Setup Trigger"),
                "Today %": st.column_config.NumberColumn("Today %", format="%+.2f%%"),
                "Dollar Volume ($M)": st.column_config.NumberColumn("Dollar Vol $M", format="%.1f"),
                "ADR %": st.column_config.NumberColumn("ADR %", format="%.2f%%"),
                "1M Return": st.column_config.NumberColumn("1M Ret", format="%+.2f%%"),
                "1M Ret Pctile": st.column_config.NumberColumn("1M Pctile", format="%.1fth"),
                "3M Return": st.column_config.NumberColumn("3M Ret", format="%+.2f%%"),
                "3M Ret Pctile": st.column_config.NumberColumn("3M Pctile", format="%.1fth"),
                "Dist from 10 EMA %": st.column_config.NumberColumn("Dist 10 EMA", format="%.2f%%"),
                "Is Power Trend": st.column_config.CheckboxColumn("⚡ Power Trend", help="Strict MA alignment & uptrend"),
                "Is Ants": st.column_config.CheckboxColumn("🐜 Ants", help="12+ up days in last 15"),
                "HV1 Defended": st.column_config.CheckboxColumn("🛡️ HV1 Defender", help="Within 5% of 1Y HV1 Anchored VWAP"),
            },
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")
    
    # ---------------------------------------------------------
    # NEW: DEDICATED GLB SECTION
    # ---------------------------------------------------------
    st.markdown("### 🟩 SEC 4.5: Green Line Breakouts (GLB)")
    st.caption("Condition: The stock is breaking out to a new 52-week high after consolidating below that high for AT LEAST 90 calendar days. This indicates all trapped supply has been absorbed.")
    
    glb_df = results_df[results_df.get('Is GLB', False)].copy()
    if glb_df.empty:
        from components import render_empty_state
        render_empty_state("No Green Line Breakouts detected today.", "🟩")
    else:
        glb_df = glb_df.sort_values('Volume Expansion', ascending=False)
        display_glb = glb_df[['Ticker', 'TickerLink', 'Today %', 'Volume Expansion', 'Dollar Volume ($M)', '1M Return', 'Dist from 10 EMA %', 'Is Power Trend', 'Is Ants', 'HV1 Defended']].copy()
        
        tv_text_glb = generate_tv_export(display_glb)
        with st.expander("📋 Copy TradingView List (GLBs)"):
            st.code(tv_text_glb, language="text")
            
        styled_glb = style_dataframe(display_glb[['TickerLink', 'Today %', 'Volume Expansion', 'Dollar Volume ($M)', '1M Return', 'Dist from 10 EMA %', 'Is Power Trend', 'Is Ants', 'HV1 Defended']])
        
        st.dataframe(
            styled_glb,
            column_config={
                "TickerLink": st.column_config.LinkColumn("Ticker", display_text=r"name=(.*)", width="small"),
                "Today %": st.column_config.NumberColumn("Today %", format="%+.2f%%"),
                "Volume Expansion": st.column_config.NumberColumn("Vol Exp", format="%.2fx", help="Current Vol / 20D Avg Vol"),
                "Dollar Volume ($M)": st.column_config.NumberColumn("Dollar Vol $M", format="%.1f"),
                "1M Return": st.column_config.NumberColumn("1M Ret", format="%+.2f%%"),
                "Dist from 10 EMA %": st.column_config.NumberColumn("Dist 10 EMA", format="%.2f%%"),
                "Is Power Trend": st.column_config.CheckboxColumn("⚡ Power Trend"),
                "Is Ants": st.column_config.CheckboxColumn("🐜 Ants"),
                "HV1 Defended": st.column_config.CheckboxColumn("🛡️ HV1 Defender"),
            },
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")
    col1, col2 = st.columns(2)

    # SEC 5: ELITE BREAKOUT SETUPS (STAGE 2)
    with col1:
        st.markdown("### 🏆 SEC 5: Stage 2 Elite Breakouts (Minervini)")
        st.caption("Condition: Price > 50 SMA > 200 SMA AND Within 15% of 52W High AND Up > 1% Today AND Volume > Avg")
        
        elite_df = results_df[results_df.get('Is Elite Breakout', False)].copy()
        elite_df = elite_df.sort_values('Volume Expansion', ascending=False)
        
        if elite_df.empty:
            st.info("No Stage 2 Elite Breakout setups detected today.")
        else:
            display_df = elite_df[['Ticker', 'TickerLink', '12D Trend Score', '12D Heatmap', 'Today %', 'Volume Expansion', 'Close Range %', '1M Return', '3M Return', 'Is Power Trend', 'Is Ants', 'HV1 Defended']].copy()
            
            tv_text = generate_tv_export(display_df)
            with st.expander("📋 Copy TradingView List (Elite Breakouts)"):
                st.code(tv_text, language="text")
                
            styled_df = style_dataframe(display_df[['TickerLink', '12D Trend Score', '12D Heatmap', 'Today %', 'Volume Expansion', 'Close Range %', '1M Return', '3M Return', 'Is Power Trend', 'Is Ants', 'HV1 Defended']])
            
            st.dataframe(
                styled_df,
                column_config={
                    "TickerLink": st.column_config.LinkColumn("Ticker", display_text=r"name=(.*)", width="small"),
                    "12D Trend Score": st.column_config.NumberColumn("12D Score", format="%d/12", help="Count of green days in the last 12 sessions."),
                    "12D Heatmap": st.column_config.TextColumn("12D Heatmap", help="Oldest (Left) -> Newest (Right)"),
                    "Today %": st.column_config.NumberColumn("Today %", format="%+.2f%%"),
                    "Volume Expansion": st.column_config.NumberColumn("Vol Exp", format="%.1fx"),
                    "Close Range %": st.column_config.ProgressColumn("Close Range", min_value=0, max_value=100, format="%d%%"),
                    "1M Return": st.column_config.NumberColumn("1M Ret", format="%+.2f%%"),
                    "3M Return": st.column_config.NumberColumn("3M Ret", format="%+.2f%%"),
                    "Is Power Trend": st.column_config.CheckboxColumn("⚡ Power Trend"),
                    "Is Ants": st.column_config.CheckboxColumn("🐜 Ants"),
                    "HV1 Defended": st.column_config.CheckboxColumn("🛡️ HV1 Defender"),
                },
                hide_index=True,
                height=400,
                use_container_width=True
            )

    # SEC 6: EPISODIC PIVOTS (QULLAMAGGIE)
    with col2:
        st.markdown("### 🔥 SEC 6: Episodic Pivots (Catalyst Gappers)")
        st.caption("Condition: Massive fundamental gap up (Gap ≥ 4%) on extreme institutional volume (Vol ≥ 2.0x 20D Avg), holding its gains (Close ≥ Open).")
        
        ep_df = results_df[results_df.get('Is EP', False)].copy()
        ep_df = ep_df.sort_values('Volume Expansion', ascending=False)
        
        if ep_df.empty:
            st.info("No Episodic Pivots detected today. Waiting for blowout earnings or catalysts.")
        else:
            display_df = ep_df[['Ticker', 'TickerLink', '12D Trend Score', '12D Heatmap', 'Gap %', 'Today %', 'Volume Expansion', 'Close Range %']].copy()
            
            tv_text = generate_tv_export(display_df)
            with st.expander("📋 Copy TradingView List (Episodic Pivots)"):
                st.code(tv_text, language="text")
                
            styled_df = style_dataframe(display_df[['TickerLink', '12D Trend Score', '12D Heatmap', 'Gap %', 'Today %', 'Volume Expansion', 'Close Range %']])
            
            st.dataframe(
                styled_df,
                column_config={
                    "TickerLink": st.column_config.LinkColumn("Ticker", display_text=r"name=(.*)", width="small"),
                    "12D Trend Score": st.column_config.NumberColumn("12D Score", format="%d/12", help="Count of green days in the last 12 sessions."),
                    "12D Heatmap": st.column_config.TextColumn("12D Heatmap", help="Oldest (Left) -> Newest (Right)"),
                    "Gap %": st.column_config.NumberColumn("Gap %", format="%+.2f%%"),
                    "Today %": st.column_config.NumberColumn("Total %", format="%+.2f%%"),
                    "Volume Expansion": st.column_config.NumberColumn("Vol Exp", format="%.1fx"),
                    "Close Range %": st.column_config.ProgressColumn("Close Range", min_value=0, max_value=100, format="%d%%"),
                },
                hide_index=True,
                height=400,
            )

    st.markdown("---")
    col3, col4 = st.columns(2)

    # SEC 7: LAUNCH PAD (EARLY ENTRY / TIGHT CONSOLIDATION)
    with col3:
        st.markdown("### 🚀 SEC 7: Launch Pads (VCP Deep Consolidation)")
        st.caption("Condition: Volatility Contraction Pattern. Price compressed within 3.5% of 21 SMA, 50 SMA, and 65 EMA simultaneously in an Uptrend.")
        
        lp_df = results_df[results_df.get('Is Launchpad', False)].copy()
        lp_df = lp_df.sort_values('Launchpad Compression %', ascending=True)
        
        if lp_df.empty:
            st.info("No Launch Pad setups detected today. Price action across the universe is too loose.")
        else:
            display_df = lp_df[['Ticker', 'TickerLink', '12D Trend Score', '12D Heatmap', 'Launchpad Compression %', 'Today %', 'Volume Expansion', '1M Return']].copy()
            
            tv_text = generate_tv_export(display_df)
            with st.expander("📋 Copy TradingView List (Launch Pads)"):
                st.code(tv_text, language="text")
                
            styled_df = style_dataframe(display_df[['TickerLink', '12D Trend Score', '12D Heatmap', 'Launchpad Compression %', 'Today %', 'Volume Expansion', '1M Return']])
            
            st.dataframe(
                styled_df,
                column_config={
                    "TickerLink": st.column_config.LinkColumn("Ticker", display_text=r"name=(.*)", width="small"),
                    "12D Trend Score": st.column_config.NumberColumn("12D Score", format="%d/12", help="Count of green days in the last 12 sessions."),
                    "12D Heatmap": st.column_config.TextColumn("12D Heatmap", help="Oldest (Left) -> Newest (Right)"),
                    "Launchpad Compression %": st.column_config.NumberColumn("Compression (Tightness)", format="%.2f%%"),
                    "Today %": st.column_config.NumberColumn("Today %", format="%+.2f%%"),
                    "Volume Expansion": st.column_config.NumberColumn("Vol Exp", format="%.1fx"),
                    "1M Return": st.column_config.NumberColumn("1M Ret", format="%+.2f%%"),
                },
                hide_index=True,
                height=400,
                use_container_width=True
            )

    # SEC 8: HV1 VOLUME EDGE (TRADERLION)
    with col4:
        st.markdown("### 🔊 SEC 8: HV1 Volume Edge (Historic Accumulation)")
        st.caption("Condition: Today's volume exceeds the highest single-day volume over the past 252 trading days (1 year).")
        
        hv1_df = results_df[results_df.get('Is HV1', False)].copy()
        hv1_df = hv1_df.sort_values('Vol vs 1Y Max', ascending=False)
        
        if hv1_df.empty:
            st.info("No HV1 signals detected today. Waiting for record-breaking institutional volume.")
        else:
            display_df = hv1_df[['Ticker', 'TickerLink', 'Today %', 'Gap %', 'Vol vs 1Y Max', 'Volume Expansion', 'Close Range %', '1M Return']].copy()
            
            tv_text = generate_tv_export(display_df)
            with st.expander("📋 Copy TradingView List (HV1 Edge)"):
                st.code(tv_text, language="text")
                
            styled_df = style_dataframe(display_df[['TickerLink', 'Today %', 'Gap %', 'Vol vs 1Y Max', 'Volume Expansion', 'Close Range %', '1M Return']])
            
            st.dataframe(
                styled_df,
                column_config={
                    "TickerLink": st.column_config.LinkColumn("Ticker", display_text=r"name=(.*)", width="small"),
                    "Today %": st.column_config.NumberColumn("Today %", format="%+.2f%%"),
                    "Gap %": st.column_config.NumberColumn("Gap %", format="%+.2f%%"),
                    "Vol vs 1Y Max": st.column_config.NumberColumn("Vol vs 1Y Max", format="%.2fx"),
                    "Volume Expansion": st.column_config.NumberColumn("Vol Exp", format="%.1fx"),
                    "Close Range %": st.column_config.ProgressColumn("Close Range", min_value=0, max_value=100, format="%d%%"),
                    "1M Return": st.column_config.NumberColumn("1M Ret", format="%+.2f%%"),
                },
                hide_index=True,
                height=400,
                use_container_width=True
            )

    st.markdown("---")
    
    # SEC 8.5: HIGH TIGHT FLAG (POWER PLAY)
    st.markdown("### 🔥 SEC 8.5: High Tight Flag (Power Play)")
    st.caption("Condition: Massive >70% thrust in <40 days, followed by tight <25% drawdown flag for 10-30 days. The nuclear momentum setup.")
    
    htf_df = results_df[results_df.get('Is HTF', False)].copy()
    htf_df = htf_df.sort_values('HTF Thrust %', ascending=False)
    
    if htf_df.empty:
        st.info("No High Tight Flags (Power Plays) detected today. This is an extremely rare and explosive setup.")
    else:
        display_df = htf_df[['Ticker', 'TickerLink', 'HTF Thrust %', 'HTF Drawdown %', 'Today %', 'Volume Expansion', 'Close Range %']].copy()
        
        tv_text = generate_tv_export(display_df)
        with st.expander("📋 Copy TradingView List (Power Plays)"):
            st.code(tv_text, language="text")
            
        styled_df = style_dataframe(display_df[['TickerLink', 'HTF Thrust %', 'HTF Drawdown %', 'Today %', 'Volume Expansion', 'Close Range %']])
        
        st.dataframe(
            styled_df,
            column_config={
                "TickerLink": st.column_config.LinkColumn("Ticker", display_text=r"name=(.*)", width="small"),
                "HTF Thrust %": st.column_config.NumberColumn("Pole Thrust %", format="%+.1f%%"),
                "HTF Drawdown %": st.column_config.NumberColumn("Flag Drawdown %", format="%+.1f%%"),
                "Today %": st.column_config.NumberColumn("Today %", format="%+.2f%%"),
                "Volume Expansion": st.column_config.NumberColumn("Vol Exp", format="%.1fx"),
                "Close Range %": st.column_config.ProgressColumn("Close Range", min_value=0, max_value=100, format="%d%%"),
            },
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")
    col_kell1, col_kell2 = st.columns(2)
    
    # SEC 8.6: EMA CROSSBACK (OLIVER KELL)
    with col_kell1:
        st.markdown("### 🏹 SEC 8.6: EMA Crossback (Oliver Kell)")
        st.caption("Condition: Price pulled below 10 EMA for <5 days to shake out weak hands, then violently crossed back above it today.")
        
        crossback_df = results_df[results_df.get('Is EMA Crossback', False)].copy()
        crossback_df = crossback_df.sort_values('Volume Expansion', ascending=False)
        
        if crossback_df.empty:
            st.info("No EMA Crossback bounce setups detected today.")
        else:
            display_df = crossback_df[['Ticker', 'TickerLink', 'Today %', 'Volume Expansion', 'Close Range %', '1M Return']].copy()
            
            tv_text = generate_tv_export(display_df)
            with st.expander("📋 Copy TradingView List (Crossbacks)"):
                st.code(tv_text, language="text")
                
            styled_df = style_dataframe(display_df[['TickerLink', 'Today %', 'Volume Expansion', 'Close Range %', '1M Return']])
            
            st.dataframe(
                styled_df,
                column_config={
                    "TickerLink": st.column_config.LinkColumn("Ticker", display_text=r"name=(.*)", width="small"),
                    "Today %": st.column_config.NumberColumn("Today %", format="%+.2f%%"),
                    "Volume Expansion": st.column_config.NumberColumn("Vol Exp", format="%.1fx"),
                    "Close Range %": st.column_config.ProgressColumn("Close Range", min_value=0, max_value=100, format="%d%%"),
                    "1M Return": st.column_config.NumberColumn("1M Ret", format="%+.2f%%"),
                },
                hide_index=True,
                height=300,
                use_container_width=True
            )

    # SEC 8.7: REVERSAL EXTENSION
    with col_kell2:
        st.markdown("### 🪃 SEC 8.7: Reversal Extension (Capitulation)")
        st.caption("Condition: Price extended >12% below 10 EMA (panic selling) and formed a violent intraday reversal today.")
        
        rev_df = results_df[results_df.get('Is Reversal Ext', False)].copy()
        rev_df = rev_df.sort_values('Volume Expansion', ascending=False)
        
        if rev_df.empty:
            st.info("No Reversal Extension (Capitulation) setups detected today.")
        else:
            display_df = rev_df[['Ticker', 'TickerLink', 'Dist from 10 EMA %', 'Today %', 'Volume Expansion', 'Close Range %']].copy()
            
            tv_text = generate_tv_export(display_df)
            with st.expander("📋 Copy TradingView List (Reversal Ext)"):
                st.code(tv_text, language="text")
                
            styled_df = style_dataframe(display_df[['TickerLink', 'Dist from 10 EMA %', 'Today %', 'Volume Expansion', 'Close Range %']])
            
            st.dataframe(
                styled_df,
                column_config={
                    "TickerLink": st.column_config.LinkColumn("Ticker", display_text=r"name=(.*)", width="small"),
                    "Dist from 10 EMA %": st.column_config.NumberColumn("Dist 10 EMA", format="%.1f%%"),
                    "Today %": st.column_config.NumberColumn("Today %", format="%+.2f%%"),
                    "Volume Expansion": st.column_config.NumberColumn("Vol Exp", format="%.1fx"),
                    "Close Range %": st.column_config.ProgressColumn("Close Range", min_value=0, max_value=100, format="%d%%"),
                },
                hide_index=True,
                height=300,
                use_container_width=True
            )

    st.markdown("---")
    
    # SEC 8.8: 50 SMA RECLAIM
    st.markdown("### 🎣 SEC 8.8: 50 SMA Reclaim (The Bear Trap)")
    st.caption("Condition: Stock is in a Stage 2 uptrend, dropped below its 50 SMA within the last 3 days, and is actively reclaiming it today.")
    
    reclaim_df = results_df[results_df.get('Is 50 Reclaim', False)].copy()
    reclaim_df = reclaim_df.sort_values('Volume Expansion', ascending=False)
    
    if reclaim_df.empty:
        st.info("No 50 SMA Reclaims detected today.")
    else:
        display_df = reclaim_df[['Ticker', 'TickerLink', 'Dist from 50 SMA %', 'Today %', 'Volume Expansion', 'Close Range %']].copy()
        
        tv_text = generate_tv_export(display_df)
        with st.expander("📋 Copy TradingView List (50 SMA Reclaim)"):
            st.code(tv_text, language="text")
            
        styled_df = style_dataframe(display_df[['TickerLink', 'Dist from 50 SMA %', 'Today %', 'Volume Expansion', 'Close Range %']])
        
        st.dataframe(
            styled_df,
            column_config={
                "TickerLink": st.column_config.LinkColumn("Ticker", display_text=r"name=(.*)", width="small"),
                "Dist from 50 SMA %": st.column_config.NumberColumn("Dist 50 SMA", format="%.2f%%"),
                "Today %": st.column_config.NumberColumn("Today %", format="%+.2f%%"),
                "Volume Expansion": st.column_config.NumberColumn("Vol Exp", format="%.1fx"),
                "Close Range %": st.column_config.ProgressColumn("Close Range", min_value=0, max_value=100, format="%d%%"),
            },
            hide_index=True,
            height=350,
            use_container_width=True
        )

    # --- MODULE C: RISK & WARNING RADARS (THE DEFENSIVE BLOCKS) ---
    st.markdown("---")
    st.markdown("## 🛡️ Risk & Warning Radars")
    col5, col6 = st.columns(2)
    
    # SEC 9: SQUAT DETECTOR (Failed Breakouts)
    with col5:
        st.markdown("### 📉 SEC 9: Breakout Failures (Squat Detector)")
        st.caption("Condition: Hit Breakout/52W High intraday BUT closed in bottom 40% of daily range")
        
        squat_df = results_df[results_df['Is Squat']].copy()
        squat_df = squat_df.sort_values('Volume Expansion', ascending=False)
        
        if squat_df.empty:
            st.success("No squatting breakouts detected today.")
        else:
            display_df = squat_df[['Ticker', 'TickerLink', 'Today %', 'Volume Expansion', 'Close Range %']].copy()
            
            tv_text = generate_tv_export(display_df)
            with st.expander("📋 Copy TradingView List"):
                st.code(tv_text, language="text")
                
            styled_df = style_dataframe(display_df[['TickerLink', 'Today %', 'Volume Expansion', 'Close Range %']])
            
            st.dataframe(
                styled_df,
                column_config={
                    "TickerLink": st.column_config.LinkColumn("Ticker", display_text=r"name=(.*)", width="small"),
                    "Today %": st.column_config.NumberColumn("Today %", format="%+.2f%%"),
                    "Volume Expansion": st.column_config.NumberColumn("Vol Exp", format="%.1fx"),
                    "Close Range %": st.column_config.ProgressColumn("Close Range", min_value=0, max_value=100, format="%d%%", help="Low = Heavy intraday selloff from highs"),
                },
                hide_index=True,
                height=400,
                use_container_width=True
            )

    # SEC 10: INSTITUTIONAL DISTRIBUTION
    with col6:
        st.markdown("### 🔴 SEC 10: Institutional Distribution (Selling Pressure)")
        st.caption("Condition: Today % ≤ -3.0% AND Volume ≥ 1.25x 20D Avg")
        
        dist_df = results_df[
            (results_df['Today %'] <= -3.0) & 
            (results_df['Volume Expansion'] >= 1.25)
        ].copy()
        
        dist_df = dist_df.sort_values('Today %', ascending=True)
        
        if dist_df.empty:
            st.success("No heavy distribution detected among the universe.")
        else:
            display_df = dist_df[['Ticker', 'TickerLink', 'Today %', 'Volume Expansion', 'Close Range %']].copy()
            
            tv_text = generate_tv_export(display_df)
            with st.expander("📋 Copy TradingView List"):
                st.code(tv_text, language="text")
                
            styled_df = style_dataframe(display_df[['TickerLink', 'Today %', 'Volume Expansion', 'Close Range %']])
            
            st.dataframe(
                styled_df,
                column_config={
                    "TickerLink": st.column_config.LinkColumn("Ticker", display_text=r"name=(.*)", width="small"),
                    "Today %": st.column_config.NumberColumn("Today %", format="%+.2f%%"),
                    "Volume Expansion": st.column_config.NumberColumn("Vol Exp", format="%.1fx"),
                    "Close Range %": st.column_config.ProgressColumn("Close Range", min_value=0, max_value=100, format="%d%%"),
                },
                hide_index=True,
                height=400,
                use_container_width=True
            )

    # SEC 11: NEW 52-WEEK HIGHS TODAY
    st.markdown("---")
    st.markdown("### 🌟 SEC 11: General Scanner (New 52-Week Highs Today)")
    st.caption("Condition: Today's High ≥ Max(Last 252D High)")
    
    highs_df = results_df[results_df['Is New High']].copy()
    highs_df = highs_df.sort_values('Volume Expansion', ascending=False)
    
    if highs_df.empty:
        st.info("No new 52-week highs detected today.")
    else:
        display_df = highs_df[['Ticker', 'TickerLink', 'Today %', 'Volume Expansion', 'Close Range %', 'Dist from Day High %', '3M Return', 'Is Power Trend', 'Is Ants', 'HV1 Defended']].copy()
        
        # Export Button (Copy to Clipboard)
        tv_text = generate_tv_export(display_df)
        with st.expander("📋 Copy TradingView List"):
            st.code(tv_text, language="text")

        styled_df = style_dataframe(display_df[['TickerLink', 'Today %', 'Volume Expansion', 'Close Range %', 'Dist from Day High %', '3M Return', 'Is Power Trend', 'Is Ants', 'HV1 Defended']])
        
        st.dataframe(
            styled_df,
            column_config={
                "TickerLink": st.column_config.LinkColumn("Ticker", display_text=r"name=(.*)", width="small"),
                "Today %": st.column_config.NumberColumn("Today %", format="%+.2f%%"),
                "Volume Expansion": st.column_config.NumberColumn("Vol Exp", format="%.1fx"),
                "Close Range %": st.column_config.ProgressColumn("Close Range %", min_value=0, max_value=100, format="%d%%"),
                "Dist from Day High %": st.column_config.NumberColumn("Dist from High", format="%.2f%%"),
                "3M Return": st.column_config.NumberColumn("3M Ret", format="%+.2f%%"),
                "Is Power Trend": st.column_config.CheckboxColumn("⚡ Power Trend"),
                "Is Ants": st.column_config.CheckboxColumn("🐜 Ants"),
                "HV1 Defended": st.column_config.CheckboxColumn("🛡️ HV1 Defender"),
            },
            hide_index=True,
            use_container_width=True
        )


main()
