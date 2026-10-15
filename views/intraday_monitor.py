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
import requests
from io import StringIO
import pytz
import numpy as np
import concurrent.futures
import plotly.express as px
import plotly.graph_objects as go

# Local imports
from market_data import fetch_nifty_total_market_tickers
from intraday_ui import (
    render_intraday_hud,
    generate_tv_export,
    render_tv_copy_box,
    style_dataframe,
    get_intraday_column_config,
    render_setup_table,
    get_active_setups_tags,
    apply_institutional_theme,
    detect_beach_ball_divergence,
    get_beach_ball_timing_state,
    render_beach_ball_timing_banner
)
from macro_regime_engine import detect_20ema_recapture_and_sequence

# Page config (Handled by app.py router in Lite)
# st.set_page_config(
#     page_title="Intraday Snapshot Monitor",
#     page_icon="📡",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# Custom CSS


# -----------------------------------------------------------------------------
# DATA FETCHING HELPERS
# -----------------------------------------------------------------------------

# Data fetching logic is routed via market_data.py

@st.cache_data(ttl=300, show_spinner=False)
def fetch_yfinance_batch(tickers, days=252, force_today_refresh=False):
    """Fetch data for multiple tickers using the High-Speed Incremental Disk Cache."""
    from price_history_manager import fetch_incremental_history
    return fetch_incremental_history(tickers, days, force_today_refresh=force_today_refresh)

@st.cache_data(ttl=900, show_spinner=False)
def get_cached_index_history(ticker: str = '^CRSLDX') -> pd.DataFrame:
    """Fetch and cache benchmark index to avoid repeated network calls."""
    try:
        return yf.download(ticker, period="1y", progress=False)
    except Exception:
        return pd.DataFrame()

def is_market_open_ist() -> bool:
    """Check if Indian stock market (NSE) is actively trading right now."""
    try:
        now = datetime.now(pytz.timezone('Asia/Kolkata'))
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=35, second=0, microsecond=0)
        return market_open <= now <= market_close
    except Exception:
        return True

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

from technical_indicators import detect_high_tight_flag, detect_ants_momentum, calculate_hv1_avwap, detect_ema_crossback, detect_reversal_extension, detect_power_trend, detect_3_weeks_tight

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
    
    idx_ret_1m = 0.0
    idx_ret_3m = 0.0
    idx_ret_6m = 0.0
    
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
                
            if len(close_series) >= 22:
                idx_21d = float(close_series.iloc[-22])
                if idx_21d > 0:
                    idx_ret_1m = ((idx_curr - idx_21d) / idx_21d) * 100
            if len(close_series) >= 64:
                idx_63d = float(close_series.iloc[-64])
                if idx_63d > 0:
                    idx_ret_3m = ((idx_curr - idx_63d) / idx_63d) * 100
            if len(close_series) >= 127:
                idx_126d = float(close_series.iloc[-127])
                if idx_126d > 0:
                    idx_ret_6m = ((idx_curr - idx_126d) / idx_126d) * 100

            # 20 EMA Recapture & 7-Step Bottom Sequence
            try:
                market_state['recapture_sequence'] = detect_20ema_recapture_and_sequence(index_data)
            except Exception as e:
                print(f"[Intraday Monitor] Recapture calc error: {e}")
        except Exception:
            pass

    index_today_pct = market_state['index_pct']
    
    from database import get_current_tml_leaders, get_latest_top_sectors, get_hat_stocks
    from systematic_engine import get_momentum_badges
    
    current_tml = get_current_tml_leaders("INDIA")
    hat_stocks = get_hat_stocks("INDIA")
    top_sectors = get_latest_top_sectors(5)
    
    # Fetch cross-conviction overlay sets
    try:
        super_compounders, rockets = get_momentum_badges()
    except Exception as e:
        print(f"Failed to fetch momentum badges: {e}")
        super_compounders, rockets = set(), set()

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
        
        # Dollar Volume (Cr)
        dollar_volume_cr = (avg_vol_20d * current_price) / 10000000
        
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
        dist_52w_high = max(0.0, ((max_252d_high - current_price) / max_252d_high * 100)) if max_252d_high > 0 else 0.0
        
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
                # GLB triggers if it's been at least 3 months since the peak, and volume is strong
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
        sma_200 = df['Close'].rolling(window=200, min_periods=50).mean().iloc[-1] if len(df) >= 50 else 0
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

        # 6M Return (126D)
        ret_6m = 0
        if len(df) >= 127:
            close_126d = df['Close'].iloc[-127]
            if close_126d > 0:
                ret_6m = ((current_price - close_126d) / close_126d) * 100
        else:
            ret_6m = ret_3m

        # 1Y Return (252D)
        ret_1y = 0
        if len(df) >= 253:
            close_252d = df['Close'].iloc[-253]
            if close_252d > 0:
                ret_1y = ((current_price - close_252d) / close_252d) * 100
        else:
            ret_1y = ret_6m

        # Market-Normalized Weighted RS Score: 0.40 * RR1M + 0.35 * RR3M + 0.25 * RR6M
        rr_1m = ret_1m - idx_ret_1m
        rr_3m = ret_3m - idx_ret_3m
        rr_6m = ret_6m - idx_ret_6m
        raw_rs_score = 0.40 * rr_1m + 0.35 * rr_3m + 0.25 * rr_6m

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
        
        # --- 3-Weeks Tight (David Ryan) ---
        is_3wt = detect_3_weeks_tight(df, max_variance_pct=2.0)
        
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

        # --- Jim Roppel Institutional Support Bounce (Monster Stocks Step 4) ---
        from intraday_ui import detect_roppel_support_bounce, compute_sister_stock_tandems
        roppel_bounce = detect_roppel_support_bounce(
            current_price=current_price,
            today_low=today_low,
            sma_21=sma_21,
            sma_50=sma_50,
            sma_200=sma_200,
            vol_expansion=vol_expansion,
            close_range_pct=close_range_pct,
            today_pct=today_pct
        )
        is_roppel_bounce = roppel_bounce['is_bounce']
        roppel_bounce_type = roppel_bounce['bounce_type']
        roppel_bounce_dist = roppel_bounce['bounce_dist_pct']

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
        
        clean_ticker = ticker.replace('.NS', '')
        is_tml = clean_ticker in current_tml
        is_hat = clean_ticker in hat_stocks
        ticker_display = f"{clean_ticker}👑" if is_tml else (f"{clean_ticker}🎩" if is_hat else clean_ticker)
        
        # Fast Momentum Overlays
        if clean_ticker in super_compounders:
            ticker_display += "⚡"
        if clean_ticker in rockets:
            ticker_display += "🚀"
            
        if is_blue_dot:
            ticker_display += "🔵"
        
        industry = industry_map.get(ticker, 'Unknown') if industry_map else 'Unknown'
        is_hot_sector = industry in top_sectors
        if is_hot_sector and industry != 'Unknown':
            ticker_display += "🔥"
            
        results.append({
            'Ticker': ticker_display,
            'TickerLink': f"https://in.tradingview.com/chart/?symbol=NSE:{clean_ticker}&name={ticker_display}",
            'Industry': industry,
            'Today %': today_pct,
            'Volume Expansion': vol_expansion,
            'Dollar Volume (Cr)': dollar_volume_cr,
            'Close': current_price,
            'Today High': today_high,
            'Today Low': today_low,
            'ADR %': adr_pct,
            'Dist from 10 EMA %': dist_ema10,
            'Is Extended': is_extended_10ema,
            'Dist from Day High %': dist_day_high,
            'Dist from 52W High %': dist_52w_high,
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
            'Is 3WT': is_3wt,
            'HV1 AVWAP': hv1_avwap,
            'HV1 Defended': is_avwap_defended,
            'Is EMA Crossback': is_ema_crossback,
            'Is Reversal Ext': is_reversal_ext,
            'Is Power Trend': is_power_trend,
            'Is 50 Reclaim': is_50_reclaim,
            'Dist from 50 SMA %': dist_50,
            'Is RS Blue Dot': is_blue_dot,
            'Is Roppel Bounce': is_roppel_bounce,
            'Support Bounce Type': roppel_bounce_type,
            'Bounce Dist %': roppel_bounce_dist,
            '6M Return': ret_6m,
            '1Y Return': ret_1y,
            'Raw_RS': raw_rs_score
        })
        
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        # Cross-sectional percentiles for absolute momentum ranking
        df_res['1M Ret Pctile'] = df_res['1M Return'].rank(pct=True, ascending=True) * 100
        df_res['3M Ret Pctile'] = df_res['3M Return'].rank(pct=True, ascending=True) * 100
        if 'Raw_RS' in df_res.columns:
            df_res['RS Rating'] = (df_res['Raw_RS'].rank(pct=True, ascending=True) * 99).fillna(0).round(0).astype(int)
        else:
            df_res['RS Rating'] = df_res['1M Ret Pctile'].fillna(0).round(0).astype(int)
            
        # Detect Steve Cohen / Shake Pryzby Beach Ball Divergence
        df_res = detect_beach_ball_divergence(df_res, index_state=market_state, is_us=False)
        
        # Jesse Livermore Sister Stock Tandem Confirmation Engine
        df_res = compute_sister_stock_tandems(df_res)
        
    return df_res, market_state

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

def main():
    # 1. Header with Refresh button & Data Updated timestamp
    col1, col2, col3 = st.columns([2.5, 1, 1], vertical_alignment="center")
    with col1:
        from components import render_header
        render_header("📡 Intraday Snapshot Monitor", "Institutional Execution Cockpit for NSE Equities")
    with col2:
        if st.button("🔄 Refresh Snapshot", use_container_width=True, type="primary"):
            st.session_state['refresh_trigger'] = True
    with col3:
        last_updated = st.session_state.get('last_updated', 'Never')
        st.markdown(
            f"<div style='text-align: right; color: #94a3b8; font-size: 0.85rem;'>Data Updated:<br>"
            f"<strong style='color: #f59e0b;'>{last_updated}</strong></div>",
            unsafe_allow_html=True
        )

    # 2. Compact Controls & Guides Expander
    with st.expander("🎛️ Scanner Controls, Liquidity & Footprint Guides", expanded=False):
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.2, 1.3])
        with ctrl_col1:
            universe_mode = st.radio(
                "Select Universe:",
                ["NIFTY 750 (High Conviction)", "Deep Market (2500+ NSE Stocks)"],
                horizontal=True,
                key='universe_mode_radio'
            )
            min_adtv = st.slider(
                "Min ADTV (₹ Cr):",
                min_value=0, max_value=50,
                value=5 if st.session_state.get('universe_mode_radio') == "Deep Market (2500+ NSE Stocks)" else 0,
                step=1,
                help="Filter out illiquid stocks based on 20-day Average Dollar Volume."
            )
        with ctrl_col2:
            st.markdown("⭐ **Quick Pin to Focus List**")
            pin_ticker = st.text_input("Enter Ticker (e.g. RELIANCE.NS):", key="pin_intra_in_text")
            if st.button("Pin to Focus List", key="btn_pin_intra_in", use_container_width=True):
                if pin_ticker:
                    from database import add_to_focus_list
                    if add_to_focus_list(pin_ticker.upper().strip(), "IN"):
                        st.success(f"Pinned {pin_ticker.upper()}!")
                    else:
                        st.error("Failed to pin.")
        with ctrl_col3:
            from database import get_latest_top_sectors
            top_sectors = get_latest_top_sectors(5)
            sector_txt = f"🔥 Hot Sectors: {', '.join(top_sectors)}" if top_sectors else "Hot Sectors not cached"
            st.markdown(f"**Symbol & Footprint Guide** ({sector_txt})")
            st.caption(
                "👑 **TML**: True Market Leader | 🎩 **Next 20** | ⚡ **Compounder** | 🚀 **Rocket**\\n"
                "⚡ **Power Trend**: Price > 21 EMA > 50 SMA (sloping up)\\n"
                "🐜 **Ants Momentum**: 12+ up closes in last 15 days\\n"
                "🛡️ **HV1 Defender**: Within 5% of 1Y Max-Volume AVWAP"
            )

    # 3. State Management & Data Fetch Trigger
    if 'results_df' not in st.session_state:
        st.session_state['results_df'] = pd.DataFrame()
        st.session_state['refresh_trigger'] = True

    if 'batched_notifications' not in st.session_state:
        st.session_state.batched_notifications = []

    if st.session_state.get('refresh_trigger', False):
        st.session_state['refresh_trigger'] = False

        with st.spinner(f"Fetching Universe List & Sectors ({st.session_state.get('universe_mode_radio', 'NIFTY 750 (High Conviction)')})..."):
            selected_mode = st.session_state.get('universe_mode_radio', 'NIFTY 750 (High Conviction)')
            from database import get_all_fundamentals_cache
            db_cache = get_all_fundamentals_cache()

            if selected_mode == "NIFTY 750 (High Conviction)":
                tickers, nse_map = fetch_nifty_total_market_tickers(return_industry_map=True)
                ind_map = {t: db_cache.get(t, {}).get('industry', nse_map.get(t, 'Unknown')) for t in tickers}
            else:
                try:
                    from views.true_market_leader import get_cached_universe
                except ImportError:
                    from pages.true_market_leader import get_cached_universe
                tickers = get_cached_universe("Deep Market (2500+ NSE Stocks)")
                ind_map = {t: db_cache.get(t, {}).get('industry', 'Unknown') for t in tickers}

        with st.spinner(f"Downloading historical & live data for {len(tickers)} stocks (~30-60s)..."):
            history_df = fetch_yfinance_batch(tickers, days=252, force_today_refresh=True)
            index_df = yf.download('^CRSLDX', period="1y", progress=False)

        with st.spinner("Computing Intraday Metrics..."):
            results_df, market_state = process_intraday_data(history_df, tickers, index_df, ind_map)
            breadth_history_df = process_historical_breadth(history_df, days=60)

            # Auto-Backfill Engine
            if len(tickers) == 1 or not isinstance(history_df.columns, pd.MultiIndex):
                ticker_data = {tickers[0]: history_df} if not history_df.empty else {}
            else:
                ticker_data = {t: history_df[t] for t in tickers if t in history_df.columns.get_level_values(0).unique()}

            from database import auto_backfill_footprints
            backfilled = auto_backfill_footprints(ticker_data, "INDIA", lookback_days=30)
            if backfilled > 0:
                st.session_state.batched_notifications.append(f"🔄 Auto-Backfill: Found & restored {backfilled} historical footprints from the last 30 days!")

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

                is_apex = (row.get('Dollar Volume (Cr)', 0) >= 20.0) and \
                          (row.get('ADR %', 0) >= 4.0) and \
                          (row.get('1M Return', 0) >= 40.0 or row.get('1M Ret Pctile', 0) >= 90.0 or row.get('3M Ret Pctile', 0) >= 90.0) and \
                          (not row.get('Is Extended', False)) and \
                          (row.get('Is EP', False) or row.get('Is Breakout', False) or row.get('Is Launchpad', False) or row.get('Is GLB', False))
                if is_apex: signals_to_save.append({'ticker': t, 'signal_name': 'Apex'})

                vol_exp = row.get('Volume Expansion', 0)
                today_ret = row.get('Today %', 0)
                adtv_cr = row.get('Dollar Volume (Cr)', 0)
                if pd.notnull(vol_exp) and vol_exp >= 5.0 and pd.notnull(today_ret) and today_ret >= 4.8 and pd.notnull(adtv_cr) and adtv_cr >= 5.0:
                    from database import log_volume_shock
                    shock_date = datetime.now().strftime('%Y-%m-%d')
                    log_volume_shock(t, shock_date, float(vol_exp), float(row.get('Close', 0)), float(row.get('Today High', 0)), float(row.get('Today Low', 0)), "INDIA")

            if signals_to_save:
                from database import save_intraday_signals
                save_intraday_signals("INDIA", signals_to_save)

            st.session_state['results_df'] = results_df
            st.session_state['breadth_history_df'] = breadth_history_df
            st.session_state['market_state'] = market_state
            st.session_state['index_pct'] = market_state.get('index_pct', 0.0)

            local_tz = pytz.timezone('Asia/Kolkata')
            now_ist = datetime.now(local_tz)
            st.session_state['last_updated'] = now_ist.strftime('%d %b, %H:%M:%S IST')
            st.rerun()

    results_df = st.session_state.get('results_df')
    if results_df is None or results_df.empty:
        st.info("👆 Click **Refresh Snapshot** to fetch live intraday data.")
        return

    # Ensure all expected columns exist with defaults
    expected_bool_cols = [
        'Is EP', 'Is Elite Breakout', 'Is Breakout', 'Is 50 Reclaim', 'Is Launchpad',
        'Is GLB', 'Is HV1', 'Is HTF', 'Is Ants', 'Is 3WT', 'Is EMA Crossback',
        'Is Reversal Ext', 'Is Power Trend', 'Is New High', 'Is Focus', 'Is Squat',
        'HV1 Defended', 'Has Positive Return', 'Has Negative Return', 'Above Prev Close',
        'Is Extended', 'Is RS Blue Dot', 'Is Roppel Bounce'
    ]
    for col in expected_bool_cols:
        if col not in results_df.columns:
            results_df[col] = False
        else:
            results_df[col] = results_df[col].fillna(False).astype(bool)

    expected_num_cols = {
        'RS Rating': 0, 'Today %': 0.0, 'Volume Expansion': 0.0,
        'Close Range %': 50.0, 'Dist from 10 EMA %': 0.0, 'ADR %': 0.0,
        '1M Return': 0.0, '3M Return': 0.0, '1M Ret Pctile': 0.0, '3M Ret Pctile': 0.0
    }
    for col, default_val in expected_num_cols.items():
        if col not in results_df.columns:
            results_df[col] = default_val

    # Apply Min ADTV filtering
    min_adtv_val = min_adtv if 'min_adtv' in locals() else 0
    if min_adtv_val > 0 and 'Dollar Volume (Cr)' in results_df.columns:
        results_df = results_df[results_df['Dollar Volume (Cr)'] >= min_adtv_val].copy()

    # Pre-compute Focus List flags & active setups
    try:
        from database import get_focus_list
        focus_tickers = get_focus_list("IN")
        clean_focus = [re.sub(r'[^\w]', '', str(t)).upper() for t in focus_tickers]
    except Exception:
        clean_focus = []

    results_df['Is Focus'] = results_df['Ticker'].apply(
        lambda x: any(cf == re.sub(r'[^\w]', '', str(x)).upper() for cf in clean_focus)
    )
    results_df['Active Setups'] = results_df.apply(get_active_setups_tags, axis=1)

    # 4. Macro Sentiment & Internals for HUD
    import sentiment_engine as se
    mmi_val = se.get_tickertape_mmi()
    india_vix_val = se.get_india_vix()
    market_state = st.session_state.get('market_state', {})

    if mmi_val is not None:
        if mmi_val < 30: mmi_lbl = "Extreme Fear"
        elif mmi_val < 45: mmi_lbl = "Fear"
        elif mmi_val <= 55: mmi_lbl = "Neutral"
        elif mmi_val <= 70: mmi_lbl = "Greed"
        else: mmi_lbl = "Extreme Greed"
    else:
        mmi_lbl = "Neutral"

    if india_vix_val is not None:
        if india_vix_val < 15: vix_lbl = "Low Volatility (<15)"
        elif india_vix_val <= 22: vix_lbl = "Normal Volatility"
        else: vix_lbl = "High Volatility (>22)"
    else:
        vix_lbl = "Normal"

    total_stocks = len(results_df)
    positive_count = int(results_df['Has Positive Return'].sum()) if 'Has Positive Return' in results_df.columns else 0
    negative_count = int(results_df['Has Negative Return'].sum()) if 'Has Negative Return' in results_df.columns else 0
    avg_return = float(results_df['Today %'].mean()) if 'Today %' in results_df.columns else 0.0
    new_highs_count = int(results_df['Is New High'].sum()) if 'Is New High' in results_df.columns else 0
    positive_pct = (positive_count / total_stocks * 100) if total_stocks > 0 else 0.0
    negative_pct = (negative_count / total_stocks * 100) if total_stocks > 0 else 0.0

    # 5. Render Executive HUD
    render_intraday_hud(
        market_state=market_state,
        sentiment_val=mmi_val,
        sentiment_label=mmi_lbl,
        vix_val=india_vix_val,
        vix_label=vix_lbl,
        total_stocks=total_stocks,
        pos_count=positive_count,
        neg_count=negative_count,
        avg_ret=avg_return,
        highs_count=new_highs_count,
        is_us=False
    )

    # -------------------------------------------------------------------------
    # 4 MASTER WORKSPACES (TABS)
    # -------------------------------------------------------------------------
    tab_setups, tab_internals, tab_defense, tab_explorer = st.tabs([
        "⚡ Live Actionable Setups",
        "🌊 Market Internals & Sectors",
        "🛡️ Risk & Defense Radar",
        "🔍 Master Universe Explorer"
    ])

    # =========================================================================
    # WORKSPACE 1: ⚡ LIVE ACTIONABLE SETUPS
    # =========================================================================
    with tab_setups:
        # Pre-compute all setup masks & counts
        ibd_base_mask = (
            ((results_df['RS Rating'] >= 80) | results_df['Ticker'].str.contains('👑')) &
            (results_df['Today %'] > 0.0) &
            (results_df['Volume Expansion'] >= 1.25) &
            (results_df['Close Range %'] >= 50)
        )
        ibd_df = results_df[ibd_base_mask].copy()
        ep_df = results_df[results_df['Is EP']].copy()
        elite_df = results_df[results_df['Is Elite Breakout']].copy()
        reclaim_df = results_df[results_df['Is 50 Reclaim']].copy()
        lp_df = results_df[results_df['Is Launchpad']].copy()
        qull_df = results_df[
            (results_df['Dollar Volume (Cr)'] >= 20.0) &
            (results_df['ADR %'] >= 4.0) &
            ((results_df['1M Return'] >= 40.0) | (results_df['1M Ret Pctile'] >= 90.0) | (results_df['3M Ret Pctile'] >= 90.0)) &
            (~results_df['Is Extended']) &
            (results_df['Is EP'] | results_df['Is Breakout'] | results_df['Is Launchpad'] | results_df['Is GLB'])
        ].copy()
        glb_df = results_df[results_df['Is GLB']].copy()
        hv1_df = results_df[results_df['Is HV1']].copy()
        htf_df = results_df[results_df['Is HTF']].copy()
        crossback_df = results_df[results_df['Is EMA Crossback']].copy()
        rev_df = results_df[results_df['Is Reversal Ext']].copy()
        ryan_df = results_df[results_df.get('Is 3WT', False)].copy()
        highs_df = results_df[results_df.get('Is New High', False)].copy()
        focus_df = results_df[results_df.get('Is Focus', False)].copy()
        bb_df = results_df[results_df.get('Is Beach Ball', False)].copy()
        bounce_df = results_df[results_df.get('Is Roppel Bounce', False)].copy()

        all_bullish_mask = (
            results_df.get('Is EP', False) |
            results_df.get('Is Elite Breakout', False) |
            results_df.get('Is 50 Reclaim', False) |
            results_df.get('Is Launchpad', False) |
            results_df.get('Is GLB', False) |
            results_df.get('Is HV1', False) |
            results_df.get('Is HTF', False) |
            results_df.get('Is EMA Crossback', False) |
            results_df.get('Is Reversal Ext', False) |
            results_df.get('Is 3WT', False) |
            results_df.get('Is Beach Ball', False) |
            results_df.get('Is Roppel Bounce', False) |
            results_df.get('Is New High', False) |
            ibd_base_mask |
            results_df.get('Is Focus', False)
        )
        all_setups_df = results_df[all_bullish_mask].copy()

        # Beach Ball Timing State Assessment (Steve Cohen & Shake Pryzby)
        bb_timing = get_beach_ball_timing_state(market_state, is_us=False)
        bb_badge_str = f" [{bb_timing['badge']}]" if bb_timing.get('badge') else ""

        # Setup Matrix Ribbon via st.pills
        setup_pills = [
            f"🔥 All Setups ({len(all_setups_df)})",
            f"🚀 IBD Leaders ({len(ibd_df)})",
            f"🏖️ Beach Balls ({len(bb_df)}){bb_badge_str}",
            f"🎯 21/50 Support Bounces ({len(bounce_df)})",
            f"💥 Episodic Pivots ({len(ep_df)})",
            f"🏆 Minervini Breakouts ({len(elite_df)})",
            f"🎣 50 SMA Reclaims ({len(reclaim_df)})",
            f"📦 Launch Pads VCP ({len(lp_df)})",
            f"⚡ Qullamaggie Apex ({len(qull_df)})",
            f"🟢 Green Line Breakouts ({len(glb_df)})",
            f"🔊 HV1 Volume Edge ({len(hv1_df)})",
            f"🚩 High Tight Flags ({len(htf_df)})",
            f"🏹 Oliver Kell Cross ({len(crossback_df)})",
            f"🪃 Reversal Ext ({len(rev_df)})",
            f"⏳ 3-Weeks Tight ({len(ryan_df)})",
            f"🌟 52W Highs ({len(highs_df)})",
            f"⭐ Focus List ({len(focus_df)})"
        ]

        selected_pill = st.pills(
            "Select Setup View:",
            setup_pills,
            default=setup_pills[0],
            label_visibility="collapsed"
        )

        if not selected_pill:
            selected_pill = setup_pills[0]

        # Route to selected setup view
        if selected_pill.startswith("🔥 All Setups"):
            st.markdown(f"### 🔥 Active Confluence Setups ({len(all_setups_df)} Stocks)")
            st.caption("Stocks triggering one or more institutional momentum setups today. Confluence is the ultimate alpha edge.")
            render_tv_copy_box(all_setups_df, "All Setups", is_us=False)
            render_setup_table(
                all_setups_df.sort_values(by=['RS Rating', 'Volume Expansion'], ascending=[False, False]),
                ['TickerLink', 'Active Setups', 'Sister Stock Tandem', 'RS Rating', 'Today %', 'Volume Expansion', 'Dollar Volume (Cr)', 'Close Range %', 'Dist from 10 EMA %', 'ADR %', 'Industry', 'Is Power Trend', 'Is Ants', 'HV1 Defended'],
                is_us=False,
                height=520
            )

        elif selected_pill.startswith("🚀 IBD Leaders"):
            st.markdown("### 🚀 IBD 50 / Apex Growth Leaders Up on Volume")
            st.caption("CANSLIM Accumulation Tracker: Top-tier leaders (RS ≥ 80 or 👑 TML) advancing on abnormal institutional volume surges with strong close range.")

            ibd_col1, ibd_col2, ibd_col3, ibd_col4 = st.columns([1.2, 1.2, 1.2, 1.4])
            with ibd_col1:
                min_ibd_rs = st.slider("Min RS Rating:", min_value=70, max_value=95, value=80, step=1, key="ibd_min_rs")
            with ibd_col2:
                min_ibd_vol = st.slider("Min Vol Expansion:", min_value=1.0, max_value=3.0, value=1.25, step=0.05, format="%.2fx", key="ibd_min_vol")
            with ibd_col3:
                min_ibd_cr = st.slider("Min Close Range %:", min_value=30, max_value=80, value=50, step=5, format="%d%%", key="ibd_min_cr")
            with ibd_col4:
                only_stage2 = st.checkbox("Only Stage 2 / Power Trend", value=False, key="ibd_only_s2")

            ibd_active_mask = (
                ((results_df.get('RS Rating', 0) >= min_ibd_rs) | results_df['Ticker'].str.contains('👑')) &
                (results_df['Today %'] > 0.0) &
                (results_df['Volume Expansion'] >= min_ibd_vol) &
                (results_df.get('Close Range %', 100) >= min_ibd_cr)
            )
            if only_stage2:
                ibd_active_mask = ibd_active_mask & (results_df.get('Is Elite Breakout', False) | results_df.get('Is Power Trend', False))
            ibd_active_df = results_df[ibd_active_mask].copy()

            if not ibd_active_df.empty:
                ibd_m1, ibd_m2, ibd_m3, ibd_m4 = st.columns(4)
                with ibd_m1:
                    st.metric("Leaders Up on Volume", f"{len(ibd_active_df)} Stocks")
                with ibd_m2:
                    st.metric("Median Volume Surge", f"{ibd_active_df['Volume Expansion'].median():.2f}x")
                with ibd_m3:
                    st.metric("Median Daily Gain", f"+{ibd_active_df['Today %'].median():.2f}%")
                with ibd_m4:
                    top_surger = ibd_active_df.sort_values('Volume Expansion', ascending=False).iloc[0]
                    clean_top = str(top_surger['Ticker']).replace('👑', '').replace('🔥', '').replace('⚡', '').replace('🚀', '').strip()
                    st.metric("Top Surge", f"{clean_top} ({top_surger['Volume Expansion']:.1f}x)", f"+{top_surger['Today %']:.1f}%")

            render_tv_copy_box(ibd_active_df, "Leaders Up on Volume", is_us=False)
            render_setup_table(
                ibd_active_df.sort_values(by=['RS Rating', 'Volume Expansion'], ascending=[False, False]),
                ['TickerLink', 'Sister Stock Tandem', 'RS Rating', 'Today %', 'Volume Expansion', 'Dollar Volume (Cr)', 'Close Range %', 'Dist from 10 EMA %', 'ADR %', 'Industry', 'Is Power Trend', 'Is Ants', 'HV1 Defended'],
                is_us=False,
                empty_msg="No leaders currently meeting criteria. Waiting for institutional volume surge.",
                empty_icon="🚀",
                height=450
            )

        elif selected_pill.startswith("🏖️ Beach Balls"):
            st.markdown(f"### 🏖️ Beach Ball Under Water Correction Divergence ({len(bb_df)} Stocks)")
            render_beach_ball_timing_banner(bb_timing)
            st.caption("Steve Cohen & Shake Pryzby CANSLIM Edge: Resilient growth leaders holding above 20 EMA and 50 SMA, printing accumulation candles on dry volume while the market is under pressure. Like a beach ball held underwater, they explode higher the instant index headwinds subside.")

            bb_col1, bb_col2, bb_col3, bb_col4 = st.columns([1.2, 1.2, 1.2, 1.4])
            with bb_col1:
                min_bb_rs = st.slider("Min RS Rating:", min_value=60, max_value=95, value=70, step=1, key="bb_min_rs")
            with bb_col2:
                min_bb_score = st.slider("Min 12D Green Days:", min_value=4, max_value=10, value=6, step=1, key="bb_min_12d")
            with bb_col3:
                max_bb_dist_high = st.slider("Max Dist from 52W High %:", min_value=2, max_value=25, value=15, step=1, format="%d%%", key="bb_max_dist")
            with bb_col4:
                only_stage2_bb = st.checkbox("Only Stage 2 / Power Trend", value=False, key="bb_only_s2")

            bb_active_mask = (
                (results_df.get('Is Beach Ball', False)) &
                (results_df.get('RS Rating', 0) >= min_bb_rs) &
                (results_df.get('12D Trend Score', 0) >= min_bb_score) &
                (results_df.get('Dist from 52W High %', 0) <= max_bb_dist_high)
            )
            if only_stage2_bb:
                bb_active_mask = bb_active_mask & (results_df.get('Is Elite Breakout', False) | results_df.get('Is Power Trend', False))
            bb_active_df = results_df[bb_active_mask].copy()

            if not bb_active_df.empty:
                bb_m1, bb_m2, bb_m3, bb_m4 = st.columns(4)
                with bb_m1:
                    st.metric("Beach Ball Leaders", f"{len(bb_active_df)} Stocks")
                with bb_m2:
                    st.metric("Median RS Rating", f"{bb_active_df['RS Rating'].median():.0f}")
                with bb_m3:
                    st.metric("Median 12D Green Days", f"{bb_active_df['12D Trend Score'].median():.0f}/12")
                with bb_m4:
                    top_bb = bb_active_df.sort_values('RS Rating', ascending=False).iloc[0]
                    clean_top = str(top_bb['Ticker']).replace('👑', '').replace('🔥', '').replace('⚡', '').replace('🚀', '').strip()
                    st.metric("Top Leader", clean_top, f"+{top_bb['Today %']:.1f}%")

            render_tv_copy_box(bb_active_df, "Beach Ball Leaders", is_us=False)
            render_setup_table(
                bb_active_df.sort_values(by=['Beach Ball Score', 'RS Rating'], ascending=[False, False]),
                ['TickerLink', 'Beach Ball Score', 'Sister Stock Tandem', 'RS Rating', 'Today %', 'Dist from 52W High %', 'Volume Expansion', 'Dollar Volume (Cr)', '12D Trend Score', 'Dist from 10 EMA %', 'Dist from 50 SMA %', 'Close Range %', 'Industry', 'Is Power Trend'],
                is_us=False,
                empty_msg="No beach ball divergence setups meeting strict criteria today.",
                empty_icon="🏖️",
                height=450
            )

        elif selected_pill.startswith("🎯 21/50 Support Bounces"):
            st.markdown(f"### 🎯 Jim Roppel 21 EMA & 50 SMA Institutional Support Bounces ({len(bounce_df)} Stocks)")
            st.caption("Jim Roppel & John Boik CANSLIM Edge (Monster Stocks Step 4): High RS Stage 2 leaders quietly testing their rising 21-day EMA or 50-day SMA on contracting volume and reversing off support intraday. Pyramiding and adding on support bounces captures the meatiest portion of monster runs.")

            bnc_col1, bnc_col2, bnc_col3, bnc_col4 = st.columns([1.2, 1.2, 1.2, 1.4])
            with bnc_col1:
                min_bnc_rs = st.slider("Min RS Rating:", min_value=60, max_value=95, value=75, step=1, key="bnc_min_rs")
            with bnc_col2:
                bnc_type_filter = st.selectbox("Moving Average:", ["All Bounces", "21 EMA Only", "50 SMA Only"], key="bnc_ma_type")
            with bnc_col3:
                max_bnc_vol = st.slider("Max Volume Surge:", min_value=0.5, max_value=2.0, value=1.4, step=0.05, format="%.2fx", key="bnc_max_vol", help="Quiet dry-up volume on touch")
            with bnc_col4:
                only_stage2_bnc = st.checkbox("Only Stage 2 / Power Trend", value=True, key="bnc_only_s2")

            bnc_active_mask = (
                (results_df.get('Is Roppel Bounce', False)) &
                (results_df.get('RS Rating', 0) >= min_bnc_rs) &
                (results_df.get('Volume Expansion', 1.0) <= max_bnc_vol)
            )
            if bnc_type_filter == "21 EMA Only":
                bnc_active_mask = bnc_active_mask & (results_df.get('Support Bounce Type', '') == '21 EMA')
            elif bnc_type_filter == "50 SMA Only":
                bnc_active_mask = bnc_active_mask & (results_df.get('Support Bounce Type', '') == '50 SMA')
            if only_stage2_bnc:
                bnc_active_mask = bnc_active_mask & (results_df.get('Is Elite Breakout', False) | results_df.get('Is Power Trend', False) | (results_df.get('Close', 0) > results_df.get('Dist from 50 SMA %', 0)))

            bnc_active_df = results_df[bnc_active_mask].copy()

            if not bnc_active_df.empty:
                bnc_m1, bnc_m2, bnc_m3, bnc_m4 = st.columns(4)
                with bnc_m1:
                    st.metric("Support Bounce Leaders", f"{len(bnc_active_df)} Stocks")
                with bnc_m2:
                    st.metric("Median RS Rating", f"{bnc_active_df['RS Rating'].median():.0f}")
                with bnc_m3:
                    st.metric("Median Vol Surge", f"{bnc_active_df['Volume Expansion'].median():.2f}x")
                with bnc_m4:
                    top_bnc = bnc_active_df.sort_values('RS Rating', ascending=False).iloc[0]
                    clean_top = str(top_bnc['Ticker']).replace('👑', '').replace('🔥', '').replace('⚡', '').replace('🚀', '').strip()
                    st.metric("Top Bounce Leader", clean_top, f"{top_bnc.get('Support Bounce Type', '21 EMA')} ({top_bnc['Today %']:+.1f}%)")

            render_tv_copy_box(bnc_active_df, "Institutional Support Bounces", is_us=False)
            render_setup_table(
                bnc_active_df.sort_values(by=['RS Rating', 'Today %'], ascending=[False, False]),
                ['TickerLink', 'Support Bounce Type', 'Bounce Dist %', 'Sister Stock Tandem', 'RS Rating', 'Today %', 'Volume Expansion', 'Dollar Volume (Cr)', 'Close Range %', 'Dist from 52W High %', 'Industry', 'Is Power Trend'],
                is_us=False,
                empty_msg="No institutional support bounce setups meeting criteria today.",
                empty_icon="🎯",
                height=450
            )

        elif selected_pill.startswith("💥 Episodic Pivots"):
            st.markdown("### 💥 Episodic Pivots (Catalyst Gappers)")
            st.caption("Condition: Massive fundamental gap up (Gap ≥ 4%) on extreme volume (≥ 2.0x 20D Avg), holding gains (Close ≥ Open).")
            ep_df = ep_df.sort_values('Volume Expansion', ascending=False)
            render_tv_copy_box(ep_df, "Episodic Pivots", is_us=False)
            render_setup_table(
                ep_df,
                ['TickerLink', 'RS Rating', '12D Trend Score', '12D Heatmap', 'Gap %', 'Today %', 'Volume Expansion', 'Close Range %', 'Dollar Volume (Cr)', 'Is Power Trend'],
                is_us=False,
                empty_msg="No Episodic Pivots detected today. Waiting for blowout earnings or catalyst gaps.",
                empty_icon="💥",
                height=420
            )

        elif selected_pill.startswith("🏆 Minervini Breakouts"):
            st.markdown("### 🏆 Stage 2 Elite Breakouts (Minervini)")
            st.caption("Condition: Price > 50 SMA > 200 SMA AND Within 15% of 52W High AND Up > 1% Today AND Volume > Avg.")
            elite_df = elite_df.sort_values('Volume Expansion', ascending=False)
            render_tv_copy_box(elite_df, "Elite Breakouts", is_us=False)
            render_setup_table(
                elite_df,
                ['TickerLink', 'RS Rating', '12D Trend Score', '12D Heatmap', 'Today %', 'Volume Expansion', 'Close Range %', '1M Return', '3M Return', 'Is Power Trend', 'Is Ants', 'HV1 Defended'],
                is_us=False,
                empty_msg="No Stage 2 Elite Breakouts detected today.",
                empty_icon="🏆",
                height=420
            )

        elif selected_pill.startswith("🎣 50 SMA Reclaims"):
            st.markdown("### 🎣 50 SMA Reclaim (The Bear Trap)")
            st.caption("Condition: Stock in Stage 2 uptrend, dropped below 50 SMA within last 3 days, and is actively reclaiming it today.")
            reclaim_df = reclaim_df.sort_values('Volume Expansion', ascending=False)
            render_tv_copy_box(reclaim_df, "50 SMA Reclaims", is_us=False)
            render_setup_table(
                reclaim_df,
                ['TickerLink', 'RS Rating', 'Dist from 50 SMA %', 'Today %', 'Volume Expansion', 'Close Range %', '1M Return', 'Is Power Trend'],
                is_us=False,
                empty_msg="No 50 SMA Reclaims detected today.",
                empty_icon="🎣",
                height=420
            )

        elif selected_pill.startswith("📦 Launch Pads VCP"):
            st.markdown("### 📦 Launch Pads (VCP Deep Consolidation)")
            st.caption("Condition: Volatility Contraction Pattern. Price compressed within 3.5% of 21 SMA, 50 SMA, and 65 EMA simultaneously in an Uptrend.")
            lp_df = lp_df.sort_values('Launchpad Compression %', ascending=True)
            render_tv_copy_box(lp_df, "Launch Pads", is_us=False)
            render_setup_table(
                lp_df,
                ['TickerLink', 'RS Rating', '12D Trend Score', '12D Heatmap', 'Launchpad Compression %', 'Today %', 'Volume Expansion', '1M Return', 'Is Power Trend'],
                is_us=False,
                empty_msg="No Launch Pad setups detected today. Volatility is too loose.",
                empty_icon="📦",
                height=420
            )

        elif selected_pill.startswith("⚡ Qullamaggie Apex"):
            st.markdown("### ⚡ The Apex Screener (Qullamaggie Strict Rules)")
            st.caption("Condition: Turnover > ₹20Cr, ADR > 4%, 1M Ret > 40% (or >90th Percentile), Not Extended (>15% above 10EMA). Trigger: EP, Breakout, Launchpad, or GLB.")
            if not qull_df.empty:
                def get_qull_trigger(row):
                    triggers = []
                    if row.get('Is EP', False): triggers.append("EP")
                    if row.get('Is Breakout', False): triggers.append("BO")
                    if row.get('Is Launchpad', False): triggers.append("Launchpad")
                    if row.get('Is GLB', False): triggers.append("GLB")
                    return " + ".join(triggers)
                qull_df['Setup Trigger'] = qull_df.apply(get_qull_trigger, axis=1)
                qull_df = qull_df.sort_values(by='1M Ret Pctile', ascending=False)

            render_tv_copy_box(qull_df, "Qullamaggie Apex", is_us=False)
            render_setup_table(
                qull_df,
                ['TickerLink', 'Setup Trigger', 'RS Rating', 'Today %', 'Dollar Volume (Cr)', 'ADR %', '1M Return', '1M Ret Pctile', '3M Ret Pctile', 'Dist from 10 EMA %', 'Is Power Trend', 'Is Ants', 'HV1 Defended'],
                is_us=False,
                empty_msg="No stocks meet the strict Qullamaggie criteria today. Sit on your hands! Cash is a position.",
                empty_icon="🧘‍♂️",
                height=420
            )

        elif selected_pill.startswith("🟢 Green Line Breakouts"):
            st.markdown("### 🟢 Green Line Breakouts (GLB)")
            st.caption("Condition: Breaking out to a new 52-week high after consolidating below that high for AT LEAST 90 calendar days.")
            glb_df = glb_df.sort_values('Volume Expansion', ascending=False)
            render_tv_copy_box(glb_df, "GLB Breakouts", is_us=False)
            render_setup_table(
                glb_df,
                ['TickerLink', 'RS Rating', 'Today %', 'Volume Expansion', 'Dollar Volume (Cr)', '1M Return', 'Dist from 10 EMA %', 'Is Power Trend', 'Is Ants', 'HV1 Defended'],
                is_us=False,
                empty_msg="No Green Line Breakouts detected today.",
                empty_icon="🟢",
                height=420
            )

        elif selected_pill.startswith("🔊 HV1 Volume Edge"):
            st.markdown("### 🔊 HV1 Volume Edge (Historic Accumulation)")
            st.caption("Condition: Today's volume exceeds the highest single-day volume over the past 252 trading days (1 year).")
            hv1_df = hv1_df.sort_values('Vol vs 1Y Max', ascending=False)
            render_tv_copy_box(hv1_df, "HV1 Volume Edge", is_us=False)
            render_setup_table(
                hv1_df,
                ['TickerLink', 'RS Rating', 'Today %', 'Gap %', 'Vol vs 1Y Max', 'Volume Expansion', 'Close Range %', '1M Return', 'Is Power Trend'],
                is_us=False,
                empty_msg="No HV1 signals detected today. Waiting for record-breaking institutional volume.",
                empty_icon="🔊",
                height=420
            )

        elif selected_pill.startswith("🚩 High Tight Flags"):
            st.markdown("### 🚩 High Tight Flag (Power Play)")
            st.caption("Condition: Massive >70% thrust in <40 days, followed by tight <25% drawdown flag for 10-30 days. The nuclear momentum setup.")
            htf_df = htf_df.sort_values('HTF Thrust %', ascending=False)
            render_tv_copy_box(htf_df, "High Tight Flags", is_us=False)
            render_setup_table(
                htf_df,
                ['TickerLink', 'RS Rating', 'HTF Thrust %', 'HTF Drawdown %', 'Today %', 'Volume Expansion', 'Close Range %', 'Is Power Trend'],
                is_us=False,
                empty_msg="No High Tight Flags (Power Plays) detected today. Extremely rare setup.",
                empty_icon="🚩",
                height=420
            )

        elif selected_pill.startswith("🏹 Oliver Kell Cross"):
            st.markdown("### 🏹 EMA Crossback (Oliver Kell)")
            st.caption("Condition: Price pulled below 10 EMA for <5 days to shake out weak hands, then violently crossed back above today.")
            crossback_df = crossback_df.sort_values('Volume Expansion', ascending=False)
            render_tv_copy_box(crossback_df, "EMA Crossbacks", is_us=False)
            render_setup_table(
                crossback_df,
                ['TickerLink', 'RS Rating', 'Today %', 'Volume Expansion', 'Close Range %', '1M Return', 'Is Power Trend'],
                is_us=False,
                empty_msg="No EMA Crossback bounce setups detected today.",
                empty_icon="🏹",
                height=420
            )

        elif selected_pill.startswith("🪃 Reversal Ext"):
            st.markdown("### 🪃 Reversal Extension (Capitulation)")
            st.caption("Condition: Price extended >12% below 10 EMA (panic selling) and formed a violent intraday reversal today.")
            rev_df = rev_df.sort_values('Volume Expansion', ascending=False)
            render_tv_copy_box(rev_df, "Reversal Extensions", is_us=False)
            render_setup_table(
                rev_df,
                ['TickerLink', 'RS Rating', 'Dist from 10 EMA %', 'Today %', 'Volume Expansion', 'Close Range %'],
                is_us=False,
                empty_msg="No Reversal Extension (Capitulation) setups detected today.",
                empty_icon="🪃",
                height=420
            )

        elif selected_pill.startswith("⏳ 3-Weeks Tight"):
            st.markdown("### ⏳ The David Ryan 3-Weeks Tight Scanner")
            st.caption("Condition: Variance between last 3 weekly closing prices is <= 2.0% and stock is above 50 SMA.")
            ryan_df = ryan_df.sort_values('1M Return', ascending=False)
            render_tv_copy_box(ryan_df, "3-Weeks Tight", is_us=False)
            render_setup_table(
                ryan_df,
                ['TickerLink', 'RS Rating', 'Today %', 'Volume Expansion', 'Dollar Volume (Cr)', '1M Return', 'Dist from 10 EMA %', 'Is Power Trend', 'Is Ants', 'HV1 Defended'],
                is_us=False,
                empty_msg="No 3-Weeks Tight setups detected today.",
                empty_icon="⏳",
                height=420
            )

        elif selected_pill.startswith("🌟 52W Highs"):
            st.markdown("### 🌟 General Scanner: New 52-Week Highs Today")
            st.caption("Condition: Today's High >= Max(Last 252D High).")
            highs_df = highs_df.sort_values('Volume Expansion', ascending=False)
            render_tv_copy_box(highs_df, "52W Highs", is_us=False)
            render_setup_table(
                highs_df,
                ['TickerLink', 'RS Rating', 'Today %', 'Volume Expansion', 'Close Range %', 'Dist from Day High %', '3M Return', 'Is Power Trend', 'Is Ants', 'HV1 Defended'],
                is_us=False,
                empty_msg="No new 52-week highs detected today.",
                empty_icon="🌟",
                height=420
            )

        elif selected_pill.startswith("⭐ Focus List"):
            st.markdown(f"### ⭐ Pinned Focus List ({len(focus_df)} Stocks)")
            st.caption("Your pinned high-conviction watchlist monitored for live intraday momentum.")
            focus_df = focus_df.sort_values('Volume Expansion', ascending=False)
            render_tv_copy_box(focus_df, "Focus List", is_us=False)
            render_setup_table(
                focus_df,
                ['TickerLink', 'Active Setups', 'RS Rating', 'Today %', 'Volume Expansion', 'Close Range %', 'Dist from 10 EMA %', '1M Return', 'Is Power Trend', 'Is Ants', 'HV1 Defended'],
                is_us=False,
                empty_msg="No pinned stocks currently in your Focus List. Pin tickers using the Controls expander above.",
                empty_icon="⭐",
                height=420
            )

    # =========================================================================
    # WORKSPACE 2: 🌊 MARKET INTERNALS & SECTORS
    # =========================================================================
    with tab_internals:
        with st.expander("🧭 Interactive Sentiment & Volatility Gauges (MMI & India VIX)", expanded=False):
            mg_c1, mg_c2 = st.columns(2)
            with mg_c1:
                with st.container(border=True):
                    fig_mmi = se.create_gauge(mmi_val, "Tickertape MMI (Fear/Greed)", 0, 100, [30, 70], inverse_colors=False, valueformat=".0f")
                    st.plotly_chart(fig_mmi, use_container_width=True)
            with mg_c2:
                with st.container(border=True):
                    fig_vix = se.create_gauge(india_vix_val, "India VIX (Volatility)", 10, 30, [15, 22], inverse_colors=False, valueformat=".1f")
                    st.plotly_chart(fig_vix, use_container_width=True)

        # SEC 1: Market Breadth Snapshot
        st.markdown("### 📊 SEC 1: Market Breadth Snapshot")
        if positive_pct >= 70:
            context_msg = "🟢 <strong>Strong Bullish Breadth:</strong> Broad market participation with the vast majority of stocks advancing."
            glow_color = "rgba(16, 185, 129, 0.15)"
        elif positive_pct >= 55:
            context_msg = "🟢 <strong>Positive Breadth:</strong> Slight bullish tilt, buyers are in control."
            glow_color = "rgba(16, 185, 129, 0.05)"
        elif positive_pct >= 45:
            context_msg = "🟡 <strong>Neutral / Mixed Breadth:</strong> Market is finding balance, no clear directional dominance intraday."
            glow_color = "rgba(245, 158, 11, 0.1)"
        elif positive_pct >= 30:
            context_msg = "🔴 <strong>Negative Breadth:</strong> Sellers are in control, broad market is sliding."
            glow_color = "rgba(239, 68, 68, 0.1)"
        else:
            context_msg = "🔴 <strong>Severe Bearish Breadth:</strong> Heavy distribution across the total market."
            glow_color = "rgba(239, 68, 68, 0.2)"

        st.markdown(f"""
        <div style="background: linear-gradient(145deg, #0f172a 0%, #020617 100%); border-radius: 16px; padding: 2rem; box-shadow: 0 20px 40px -12px rgba(0, 0, 0, 0.6); border: 1px solid rgba(255, 255, 255, 0.05); position: relative; overflow: hidden; margin-bottom: 2rem;">
            <div style="position: absolute; top: -50%; left: -10%; width: 60%; height: 200%; background: radial-gradient(circle, {glow_color} 0%, transparent 70%); transform: rotate(-15deg); pointer-events: none;"></div>
            <div style="display: flex; gap: 1.25rem; flex-wrap: wrap;">
                <div style="background: rgba(0,0,0,0.4); padding: 1.25rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); flex: 1; min-width: 180px; text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 0.35rem;">Positive %</div>
                    <div style="color: #10b981; font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 800; text-shadow: 0 0 20px rgba(16,185,129,0.4);">{positive_pct:.1f}%</div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.35rem;">({positive_count}/{total_stocks} stocks)</div>
                </div>
                <div style="background: rgba(0,0,0,0.4); padding: 1.25rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); flex: 1; min-width: 180px; text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 0.35rem;">Negative %</div>
                    <div style="color: #ef4444; font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 800; text-shadow: 0 0 20px rgba(239,68,68,0.4);">{negative_pct:.1f}%</div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.35rem;">({negative_count}/{total_stocks} stocks)</div>
                </div>
                <div style="background: rgba(0,0,0,0.4); padding: 1.25rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); flex: 1; min-width: 180px; text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 0.35rem;">Avg Return</div>
                    <div style="color: {'#10b981' if avg_return > 0 else '#ef4444'}; font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 800;">{avg_return:+.2f}%</div>
                </div>
                <div style="background: rgba(0,0,0,0.4); padding: 1.25rem; border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.3); flex: 1; min-width: 180px; text-align: center;">
                    <div style="color: #f59e0b; font-size: 0.75rem; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 0.35rem;">New 52W Highs</div>
                    <div style="color: #fcd34d; font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 800; text-shadow: 0 0 20px rgba(245,158,11,0.4);">{new_highs_count}</div>
                </div>
            </div>
            <div style="margin-top: 1.5rem; padding: 0.85rem 1.25rem; border-radius: 10px; background: rgba(255,255,255,0.03); border-left: 4px solid {'#10b981' if positive_pct >= 55 else '#ef4444' if positive_pct < 45 else '#f59e0b'}; font-size: 0.9rem; color: #e2e8f0;">
                {context_msg}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # SEC 2: Institutional Sector Heatmap
        st.markdown("### 🗺️ SEC 2: Institutional Sector Heatmap")
        st.caption("Aggregates momentum across sectors to reveal macro leadership.")

        if 'Industry' in results_df.columns:
            industry_gb = results_df.groupby('Industry').agg(
                Stocks_Count=('Ticker', 'count'),
                Avg_Today_Pct=('Today %', 'mean'),
                Avg_1M_Ret=('1M Return', 'mean'),
                Positive_Pct=('Has Positive Return', lambda x: x.mean() * 100)
            )
            valid_sectors = industry_gb[industry_gb['Stocks_Count'] >= 3].copy()
            if not valid_sectors.empty:
                valid_sectors = valid_sectors.sort_values('Avg_Today_Pct', ascending=False).reset_index()
                top_5 = valid_sectors.head(5)
                bottom_3 = valid_sectors.tail(3)
                combine_bar = pd.concat([top_5, bottom_3])

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

        # SEC 3: Historical Breadth (Last 60 Days)
        breadth_history_df = st.session_state.get('breadth_history_df', pd.DataFrame())
        if not breadth_history_df.empty:
            st.markdown("---")
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
                    st.caption("**Condition:** Daily % of stocks closing positive. **Use:** Broad market health. **Line:** 10 DMA.")

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
                    st.caption("**Condition:** Daily count of stocks hitting 252-day high. **Use:** Expansion. **Line:** 10 DMA.")

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
                    st.caption("**Condition:** Stocks within 20% of 52W High falling >3% on 1.5x Vol. **Use:** Selling pressure. **Line:** 10 DMA.")

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
                    st.caption("**Condition:** Breakout attempt closing in bottom 40% of range. **Use:** Breakout failure rate. **Line:** 10 DMA.")

    # =========================================================================
    # WORKSPACE 3: 🛡️ RISK & DEFENSE RADAR
    # =========================================================================
    with tab_defense:
        st.markdown("### 🛡️ Risk & Warning Radars (Defensive Cockpit)")
        st.caption("Real-time monitoring of institutional selling pressure and failed breakouts to protect trading capital.")

        squat_df = results_df[results_df['Is Squat']].copy().sort_values('Volume Expansion', ascending=False)
        dist_df = results_df[
            (results_df['Today %'] <= -3.0) &
            (results_df['Volume Expansion'] >= 1.25)
        ].copy().sort_values('Today %', ascending=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            sq_color = "normal" if len(squat_df) < 5 else "inverse"
            st.metric("Breakout Failures (Squats)", f"{len(squat_df)} Stocks", delta="Failed Breakouts" if len(squat_df) > 0 else "Clean Action", delta_color=sq_color)
        with r2:
            dist_color = "normal" if len(dist_df) < 5 else "inverse"
            st.metric("Institutional Distribution", f"{len(dist_df)} Stocks", delta="Heavy Selling (>3% drop on vol)" if len(dist_df) > 0 else "No Distribution", delta_color=dist_color)
        with r3:
            risk_status = "🟢 Low Risk" if (len(squat_df) < 3 and len(dist_df) < 5) else ("🟡 Elevated Risk" if (len(squat_df) < 8 and len(dist_df) < 15) else "🔴 High Selling Pressure")
            st.metric("Market Defense State", risk_status)

        st.markdown("---")
        st.markdown("#### 📉 SEC 9: Breakout Failures (Squat Detector)")
        st.caption("Condition: Attempted breakout or 52W high intraday BUT closed in bottom 40% of daily range on high volume.")
        render_tv_copy_box(squat_df, "Squats", is_us=False)
        render_setup_table(
            squat_df,
            ['TickerLink', 'RS Rating', 'Today %', 'Volume Expansion', 'Close Range %', 'Dollar Volume (Cr)', 'Dist from Day High %'],
            is_us=False,
            empty_msg="No squatting breakouts detected today. Breakouts are holding their intraday gains!",
            empty_icon="🛡️",
            height=350
        )

        st.markdown("---")
        st.markdown("#### 🔴 SEC 10: Institutional Distribution (Selling Pressure)")
        st.caption("Condition: Today % <= -3.0% AND Volume >= 1.25x 20D Average Volume.")
        render_tv_copy_box(dist_df, "Distribution Stocks", is_us=False)
        render_setup_table(
            dist_df,
            ['TickerLink', 'RS Rating', 'Today %', 'Volume Expansion', 'Close Range %', 'Dollar Volume (Cr)', 'Dist from 10 EMA %'],
            is_us=False,
            empty_msg="No heavy institutional distribution detected among the universe today.",
            empty_icon="✨",
            height=350
        )

    # =========================================================================
    # WORKSPACE 4: 🔍 MASTER UNIVERSE EXPLORER
    # =========================================================================
    with tab_explorer:
        st.markdown("### 🔍 Master Universe Explorer")
        st.caption("Universal multi-dimensional screener across all loaded stocks.")

        ex_c1, ex_c2, ex_c3 = st.columns([1.5, 1.5, 1.5])
        with ex_c1:
            search_txt = st.text_input("🔍 Search Symbol or Name:", key="explorer_search_txt")
        with ex_c2:
            all_industries = sorted([str(ind) for ind in results_df['Industry'].dropna().unique() if ind != 'Unknown'])
            selected_industries = st.multiselect("Filter by Sector / Theme:", options=all_industries, key="explorer_industry_select")
        with ex_c3:
            setup_tags = [
                "💥 EP", "🏆 Stage 2 BO", "⚡ BO", "🎣 50 SMA", "📦 VCP", "🟢 GLB",
                "🚩 HTF", "🏹 Kell Cross", "🪃 Rev Ext", "⏳ 3WT", "🔊 HV1", "🌟 52W High",
                "⚡ Power Trend", "🐜 Ants", "🛡️ HV1 Defender", "⭐ Focus"
            ]
            selected_setups = st.multiselect("Filter by Setup Trigger:", options=setup_tags, key="explorer_setups_select")

        ex_c4, ex_c5, ex_c6 = st.columns(3)
        with ex_c4:
            min_rs_exp = st.slider("Min RS Score:", min_value=0, max_value=95, value=0, step=5, key="exp_min_rs")
        with ex_c5:
            min_vol_exp = st.slider("Min Volume Surge:", min_value=0.0, max_value=5.0, value=0.0, step=0.25, format="%.2fx", key="exp_min_vol")
        with ex_c6:
            min_ret_exp = st.slider("Min Today %:", min_value=-10.0, max_value=15.0, value=-10.0, step=0.5, format="%.1f%%", key="exp_min_ret")

        exp_filtered = results_df.copy()
        if search_txt:
            s_clean = search_txt.upper().strip()
            exp_filtered = exp_filtered[exp_filtered['Ticker'].str.upper().str.contains(s_clean, na=False)]
        if selected_industries:
            exp_filtered = exp_filtered[exp_filtered['Industry'].isin(selected_industries)]
        if selected_setups:
            for st_tag in selected_setups:
                if st_tag == "💥 EP": exp_filtered = exp_filtered[exp_filtered.get('Is EP', False)]
                elif st_tag == "🏆 Stage 2 BO": exp_filtered = exp_filtered[exp_filtered.get('Is Elite Breakout', False)]
                elif st_tag == "⚡ BO": exp_filtered = exp_filtered[exp_filtered.get('Is Breakout', False)]
                elif st_tag == "🎣 50 SMA": exp_filtered = exp_filtered[exp_filtered.get('Is 50 Reclaim', False)]
                elif st_tag == "📦 VCP": exp_filtered = exp_filtered[exp_filtered.get('Is Launchpad', False)]
                elif st_tag == "🟢 GLB": exp_filtered = exp_filtered[exp_filtered.get('Is GLB', False)]
                elif st_tag == "🚩 HTF": exp_filtered = exp_filtered[exp_filtered.get('Is HTF', False)]
                elif st_tag == "🏹 Kell Cross": exp_filtered = exp_filtered[exp_filtered.get('Is EMA Crossback', False)]
                elif st_tag == "🪃 Rev Ext": exp_filtered = exp_filtered[exp_filtered.get('Is Reversal Ext', False)]
                elif st_tag == "⏳ 3WT": exp_filtered = exp_filtered[exp_filtered.get('Is 3WT', False)]
                elif st_tag == "🔊 HV1": exp_filtered = exp_filtered[exp_filtered.get('Is HV1', False)]
                elif st_tag == "🌟 52W High": exp_filtered = exp_filtered[exp_filtered.get('Is New High', False)]
                elif st_tag == "⚡ Power Trend": exp_filtered = exp_filtered[exp_filtered.get('Is Power Trend', False)]
                elif st_tag == "🐜 Ants": exp_filtered = exp_filtered[exp_filtered.get('Is Ants', False)]
                elif st_tag == "🛡️ HV1 Defender": exp_filtered = exp_filtered[exp_filtered.get('HV1 Defended', False)]
                elif st_tag == "⭐ Focus": exp_filtered = exp_filtered[exp_filtered.get('Is Focus', False)]

        if min_rs_exp > 0:
            exp_filtered = exp_filtered[exp_filtered.get('RS Rating', 0) >= min_rs_exp]
        if min_vol_exp > 0.0:
            exp_filtered = exp_filtered[exp_filtered['Volume Expansion'] >= min_vol_exp]
        if min_ret_exp > -10.0:
            exp_filtered = exp_filtered[exp_filtered['Today %'] >= min_ret_exp]

        if not exp_filtered.empty:
            st.markdown(
                f"**Showing {len(exp_filtered)} of {total_stocks} stocks** &nbsp;|&nbsp; "
                f"Median Return: **{exp_filtered['Today %'].median():+.2f}%** &nbsp;|&nbsp; "
                f"Median RS: **{exp_filtered['RS Rating'].median():.0f}**"
            )

        render_tv_copy_box(exp_filtered, "Filtered Universe", is_us=False)
        render_setup_table(
            exp_filtered.sort_values(by=['RS Rating', 'Today %'], ascending=[False, False]),
            ['TickerLink', 'Active Setups', 'RS Rating', 'Today %', 'Volume Expansion', 'Dollar Volume (Cr)', 'Close Range %', 'Dist from 10 EMA %', 'ADR %', '1M Return', '3M Return', 'Industry', 'Is Power Trend', 'Is Ants', 'HV1 Defended'],
            is_us=False,
            empty_msg="No stocks in the universe match the selected multi-filter combination.",
            empty_icon="🔍",
            height=550
        )


if __name__ == "__main__":
    main()

