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
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import plotly.express as px
import plotly.graph_objects as go
import time
import concurrent.futures

# Local Imports
from pages.intraday_monitor import fetch_nifty_total_market_tickers, fetch_yfinance_batch
from clenow_math import calculate_adjusted_slope
from database import (
    get_all_fundamentals_cache,
    save_fundamentals_cache,
    save_tml_snapshot,
    get_recent_intraday_signals,
    get_tml_persistence,
    add_to_focus_list,
    get_tml_hall_of_fame
)

# st.set_page_config(
#     page_title="True Market Leader (US)",
#     page_icon="👑",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )


# Custom CSS


# -----------------------------------------------------------------------------
# CACHED DATA FETCHERS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_universe(mode="Stocks"):
    import os
    if mode == "ETFs":
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tickers_us_etf.txt')
    else:
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tickers_us.txt')
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip() and '-' not in line]
        return list(set(tickers))
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_history(tickers, days):
    return fetch_yfinance_batch(tickers, days=days)

from components import render_disk_cache_sidebar
render_disk_cache_sidebar(get_cached_universe)

# -----------------------------------------------------------------------------
# CORE LOGIC
# -----------------------------------------------------------------------------
def compute_rs_scores_fast(history_df, tickers):
    """Compute fast proxy RS scores (1M, 3M, 6M weighted) to filter leaders."""
    rs_results = {}
    if history_df.columns.nlevels < 2:
        return rs_results
        
    try:
        close_df = history_df.xs('Close', level=1, axis=1)
    except:
        try:
            close_df = history_df.xs('Close', level=0, axis=1)
        except:
            return rs_results
        
    valid_tickers = [t for t in tickers if t in close_df.columns]
    current_prices = close_df.iloc[-1]
    
    ret_1w_series = pd.Series(index=valid_tickers, dtype=float)
    ret_1m_series = pd.Series(index=valid_tickers, dtype=float)
    ret_3m_series = pd.Series(index=valid_tickers, dtype=float)
    
    for t in valid_tickers:
        s = close_df[t].dropna()
        if len(s) < 6:
            ret_1w_series[t] = 0
        else:
            ret_1w_series[t] = (s.iloc[-1] / s.iloc[-6]) - 1
            
        if len(s) < 22:
            ret_1m_series[t] = 0
        else:
            ret_1m_series[t] = (s.iloc[-1] / s.iloc[-22]) - 1
            
        if len(s) < 64:
            ret_3m_series[t] = 0
        else:
            ret_3m_series[t] = (s.iloc[-1] / s.iloc[-64]) - 1
            
    # Ultra-responsive momentum RS weights
    composite = (ret_1w_series * 0.2) + (ret_1m_series * 0.4) + (ret_3m_series * 0.4)
    ranks = composite.rank(pct=True) * 99
    
    for t in valid_tickers:
        rs_results[t] = ranks[t]
        
    return rs_results

def run_technical_prescreen(history_df, tickers, rs_scores, is_etf=False):
    """
    Scans stocks and tracks funnel drop-offs.
    """
    passed_stocks = []
    
    # Download benchmark for RS Blue Dot (S&P 500)
    try:
        bench_df = yf.download('^GSPC', period="252d", progress=False)
        bench_df.columns = [c.lower() for c in bench_df.columns]
    except Exception as e:
        print(f"[TML US] Failed to download benchmark index: {e}")
        bench_df = pd.DataFrame()
    funnel = {
        'Target Universe': len(tickers),
        'Valid Data Fetched': 0,
        'RS > 80 & Price > $20': 0,
        'Stage 2 (SMA 50 > 200)': 0,
        'Near Highs (<20%)': 0,
        'Highly Liquid (ADTV > $10M)': 0
    }
    
    if history_df.columns.nlevels < 2:
        return passed_stocks, funnel
        
    try:
        close_panel = history_df.xs('Close', level=1, axis=1)
        vol_panel = history_df.xs('Volume', level=1, axis=1)
        high_panel = history_df.xs('High', level=1, axis=1)
        low_panel = history_df.xs('Low', level=1, axis=1)
    except:
        try:
            close_panel = history_df.xs('Close', level=0, axis=1)
            vol_panel = history_df.xs('Volume', level=0, axis=1)
            high_panel = history_df.xs('High', level=0, axis=1)
            low_panel = history_df.xs('Low', level=0, axis=1)
        except:
            return passed_stocks, funnel

    for t in tickers:
        if t not in close_panel.columns:
            continue
            
        c_series = close_panel[t].dropna()
        if len(c_series) < 200:
            continue
            
        funnel['Valid Data Fetched'] += 1
            
        close = c_series.iloc[-1]
        rs = rs_scores.get(t, 0)
        
        # GATE 1: RS & Price
        if not is_etf and (rs < 80 or close < 20.0):
            continue
        funnel['RS > 80 & Price > $20'] += 1
            
        # GATE 2: Stage 2 Moving Averages
        sma50 = c_series.tail(50).mean()
        sma200 = c_series.tail(200).mean()
        ema10 = c_series.ewm(span=10, adjust=False).mean().iloc[-1] if len(c_series) >= 10 else 0.0
        ema21 = c_series.ewm(span=21, adjust=False).mean().iloc[-1] if len(c_series) >= 21 else 0.0
        if not is_etf and not (close > sma50 and sma50 > sma200):
            continue
        funnel['Stage 2 (SMA 50 > 200)'] += 1
            
        # GATE 3: 52W High Proximity
        h_series = high_panel[t].dropna()
        high_252 = h_series.tail(252).max()
        if high_252 <= 0:
            continue
        dist_high = ((high_252 - close) / high_252) * 100
        if not is_etf and (dist_high > 20.0):
            continue
        funnel['Near Highs (<20%)'] += 1
            
        # GATE 4: Liquidity
        v_series = vol_panel[t].dropna()
        vol_21 = v_series.tail(21).mean()
        adtv_m = (vol_21 * close) / 1000000.0
        if not is_etf and (adtv_m < 10.0):
            continue
        funnel['Highly Liquid (ADTV > $10M)'] += 1
        
        ext_50d = ((close / sma50) - 1.0) * 100.0 if sma50 > 0 else 0.0
        ext_50d_atr = 0.0
        trend_6m = c_series.tail(126).tolist()
        
        # --- Ants & EMA Crossback ---
        is_ants = False
        if len(c_series) >= 16:
            recent_16 = c_series.tail(16)
            up_days = (recent_16 > recent_16.shift(1)).iloc[1:].sum()
            is_ants = bool(up_days >= 12)
            
        is_crossback = False
        is_hold_the_line = False
        if len(c_series) >= 20 and t in low_panel.columns:
            try:
                temp_df = pd.DataFrame({
                    'Close': c_series,
                    'High': high_panel[t],
                    'Low': low_panel[t],
                    'Volume': vol_panel[t]
                }).dropna()
                from technical_indicators import detect_ema_crossback, detect_power_trend, detect_3_weeks_tight, calculate_atr
                is_crossback = detect_ema_crossback(temp_df, ema_period=10, max_days_below=5)
                is_power_trend = detect_power_trend(temp_df)
                is_3wt = detect_3_weeks_tight(temp_df, max_variance_pct=2.0)
                
                temp_df_lower = temp_df.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close'})
                atr_series = calculate_atr(temp_df_lower, 21)
                if not atr_series.empty and pd.notna(atr_series.iloc[-1]) and atr_series.iloc[-1] > 0:
                    ext_50d_atr = (close - sma50) / atr_series.iloc[-1]
                
                # --- Hold the Line (21-EMA Pullback) ---
                if len(temp_df) >= 21:
                    ema21_series = temp_df['Close'].ewm(span=21, adjust=False).mean()
                    if len(temp_df) >= 16:
                        past_15_c = temp_df['Close'].iloc[-16:-1]
                        past_15_ema21 = ema21_series.iloc[-16:-1]
                        if (past_15_c > past_15_ema21).all():
                            today_low = temp_df['Low'].iloc[-1]
                            today_close = temp_df['Close'].iloc[-1]
                            today_ema21 = ema21_series.iloc[-1]
                            # Today's Low drops into the +5% zone, and Close holds the -3% zone
                            if today_low <= today_ema21 * 1.05 and today_close >= today_ema21 * 0.97:
                                today_vol = temp_df['Volume'].iloc[-1]
                                if len(temp_df) >= 50:
                                    vol_50d = temp_df['Volume'].tail(50).mean()
                                    if today_vol < vol_50d:
                                        is_hold_the_line = True
            except Exception:
                is_power_trend = False
                is_3wt = False
                is_hold_the_line = False
        
        # --- Institutional Footprint Metrics ---
        pocket_pivot_5d = False
        vdu_5d = False
        if len(c_series) >= 16 and len(v_series) >= 55:
            recent_c = c_series.tail(16)
            recent_v = v_series.tail(55)
            
            # VDU Check (Last 5 days against their rolling 50-day MA)
            vol_50d_ma = recent_v.rolling(50).mean()
            vdu_mask = recent_v < (vol_50d_ma * 0.5)
            vdu_5d = bool(vdu_mask.tail(5).any())
            
            # Pocket Pivot Check (Last 5 days)
            is_up_day = recent_c > recent_c.shift(1)
            is_down_day = recent_c < recent_c.shift(1)
            
            recent_v_16 = recent_v.tail(16)
            down_volume = recent_v_16.where(is_down_day, 0)
            highest_down_vol_10d = down_volume.shift(1).rolling(10).max()
            
            pp_mask = is_up_day & (recent_v_16 > highest_down_vol_10d)
            pocket_pivot_5d = bool(pp_mask.tail(5).any())

        hv1_detected = False
        if len(v_series) >= 250:
            vol_250 = v_series.tail(250)
            max_vol_250 = vol_250.max()
            last_21_vols = vol_250.tail(21)
            hv1_detected = bool(last_21_vols.max() >= max_vol_250)
        elif len(v_series) >= 21:
            max_vol_all = v_series.max()
            last_21_vols = v_series.tail(21)
            hv1_detected = bool(last_21_vols.max() >= max_vol_all)
        
        ud_ratio = 1.0
        if len(c_series) >= 51 and len(v_series) >= 50:
            recent_50_closes = c_series.tail(51)
            recent_50_vols = v_series.tail(50)
            daily_rets = recent_50_closes.pct_change().iloc[1:]
            daily_rets = daily_rets.reindex(recent_50_vols.index)
            up_vol = recent_50_vols.loc[daily_rets > 0].sum()
            down_vol = recent_50_vols.loc[daily_rets < 0].sum()
            ud_ratio = round(float(up_vol / down_vol), 2) if down_vol > 0 else (round(float(up_vol), 2) if up_vol > 0 else 1.0)
            
        # --- Power Days (3M) ---
        power_days_3m = 0
        dist_days_3m = 0
        if len(c_series) >= 66:
            recent_66_closes = c_series.tail(66)
            daily_rets_65 = (recent_66_closes.pct_change().iloc[1:] * 100.0)
            power_days_3m = int((daily_rets_65 >= 4.0).sum())
            dist_days_3m = int((daily_rets_65 <= -4.0).sum())
            
        # Liquidity Expansion Check (Roppel)
        liq_expansion = False
        if len(v_series) >= 50:
            vol_50d_avg = v_series.tail(50).mean()
            if vol_50d_avg > 0:
                last_5_vols = v_series.tail(5)
                vol_5d_avg = last_5_vols.mean()
                # Condition 1: Any of last 5 days > 300%
                if (last_5_vols.max() / vol_50d_avg) > 3.0:
                    liq_expansion = True
                # Condition 2: 5-day average > 150%
                elif (vol_5d_avg / vol_50d_avg) > 1.5:
                    liq_expansion = True
                    
        # --- Volatility & Momentum Stats ---
        vol_3m = 0.0
        ret_3m = 0.0
        if len(c_series) >= 64:
            recent_64_closes = c_series.tail(64)
            daily_returns = recent_64_closes.pct_change().dropna()
            if len(daily_returns) > 0:
                vol_3m = daily_returns.std() * np.sqrt(252) * 100
            if recent_64_closes.iloc[0] > 0:
                ret_3m = ((recent_64_closes.iloc[-1] / recent_64_closes.iloc[0]) - 1) * 100

        # --- Relative Measured Volatility (RMV) 15D ---
        rmv_15d = 100.0
        try:
            if t in low_panel.columns:
                df_rmv = pd.DataFrame({
                    'high': h_series,
                    'low': low_panel[t].dropna()
                }).dropna()
                if len(df_rmv) >= 15:
                    recent_ranges = df_rmv['high'].tail(15) - df_rmv['low'].tail(15)
                    # Filter out zero-range dummy bars appended by yfinance on weekends
                    valid_ranges = recent_ranges[recent_ranges > 0]
                    if len(valid_ranges) >= 5:
                        current_range = valid_ranges.iloc[-1]
                        min_range = valid_ranges.min()
                        max_range = valid_ranges.max()
                        if max_range > min_range:
                            rmv_15d = float(((current_range - min_range) / (max_range - min_range)) * 100.0)
                        else:
                            rmv_15d = 0.0
        except Exception:
            pass
            
        # --- High Tight Flag (Power Play) ---
        htf_detected = False
        if len(c_series) >= 60 and t in low_panel.columns:
            df_htf = pd.DataFrame({
                'high': h_series,
                'low': low_panel[t].dropna(),
                'close': c_series
            }).dropna()
            if len(df_htf) >= 60:
                from technical_indicators import detect_high_tight_flag
                htf_data = detect_high_tight_flag(df_htf, min_thrust_pct=70.0)
                htf_detected = htf_data['is_htf']

        # --- Actionability Status ---
        ext_21ema = ((close / ema21) - 1.0) * 100.0 if ema21 > 0 else 0.0
        
        if ext_21ema > 15.0 or ext_50d > 25.0:
            action_status = '🚨 Extended'
        elif is_hold_the_line:
            action_status = '🛡️ Buy@21EMA'
        elif rmv_15d < 15.0 and dist_high < 10.0 and vdu_5d:
            action_status = '🔥 Tight VCP'
        elif is_power_trend and ema10 > 0 and abs((close - ema10) / ema10) * 100.0 < 2.0:
            action_status = '🚀 10 EMA Power Trend'
        elif ema21 > 0 and abs((close - ema21) / ema21) * 100.0 <= 5.0 and ema21 > sma50:
            action_status = '📉 21 EMA Pullback'
        else:
            action_status = '🟡 Building Base'

        # --- Clenow Trend Quality (Display Only - Not Scored) ---
        clenow_mom = {'score': 0.0, 'r_squared': 0.0, 'annualized_slope': 0.0}
        if len(c_series) >= 90:
            df_clenow = pd.DataFrame({'close': c_series})
            clenow_mom = calculate_adjusted_slope(df_clenow, window=90)
            
        # --- RS Blue Dot ---
        rs_blue_dot = False
        if not bench_df.empty:
            try:
                from relative_strength import calculate_rs_blue_dot
                stock_df = pd.DataFrame({'close': c_series})
                rs_blue_dot_series = calculate_rs_blue_dot(stock_df, bench_df)
                rs_blue_dot = bool(rs_blue_dot_series.iloc[-1])
            except Exception:
                pass

        passed_stocks.append({
            'ticker': t,
            'close': close,
            'rs_score': rs,
            'rs_blue_dot': rs_blue_dot,
            'dist_high': dist_high,
            'adtv_m': adtv_m,
            'sma50': sma50,
            'sma200': sma200,
            'ext_50d': ext_50d,
            'ext_50d_atr': ext_50d_atr,
            'trend_6m': trend_6m,
            'hv1': hv1_detected,
            'ud_ratio': ud_ratio,
            'clenow_score': clenow_mom['score'],
            'clenow_r2': clenow_mom['r_squared'],
            'pocket_pivot': pocket_pivot_5d,
            'vdu': vdu_5d,
            'htf': htf_detected,
            'vol_3m': vol_3m,
            'ret_3m': ret_3m,
            'rmv_15d': rmv_15d,
            'is_etf': is_etf,
            'is_ants': is_ants,
            'Is_Crossback': is_crossback,
            'Is_Power_Trend': is_power_trend,
            'Is_3WT': is_3wt,
            'Is_Hold_The_Line': is_hold_the_line,
            'Power_Days_3M': power_days_3m,
            'Dist_Days_3M': dist_days_3m,
            'Action_Status': action_status,
            'Liq_Expansion': liq_expansion
        })
        
    return passed_stocks, funnel

def fetch_fundamentals(ticker, db_cache=None, max_retries=3):
    """Fetch EPS growth, Sales growth, ROE with retry logic for rate limits and DB Fallback."""
    
    # --- 21 DAY DYNAMIC CACHE VALIDATION ---
    if db_cache and ticker in db_cache:
        try:
            last_dt_str = db_cache[ticker].get('updated_at', None)
            if last_dt_str:
                last_updated = datetime.strptime(last_dt_str, '%Y-%m-%d %H:%M:%S')
                if (datetime.now() - last_updated).days < 21:
                    cached = db_cache[ticker].copy()
                    mcap = cached.get('market_cap', 0.0)
                    cached['mcap_b'] = mcap / 1000000000.0 if mcap else 0.0
                    return cached, False
        except Exception:
            pass
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(ticker)
            eps_g = 0.0
            sales_g = 0.0
            roe = 0.0
            mcap_b = 0.0
            industry = "Unknown"
            
            info = t.info
            if info:
                industry = info.get('industry', 'Unknown')
                mcap = info.get('marketCap', 0)
                if mcap is not None: mcap_b = mcap / 1000000000.0
                roe = info.get('returnOnEquity', 0)
                if roe is not None: roe *= 100
                else: roe = 0.0
                    
                rev_g = info.get('revenueGrowth', 0)
                if rev_g is not None: sales_g = rev_g * 100
                    
                ern_g = info.get('earningsGrowth', 0)
                if ern_g is not None: eps_g = ern_g * 100

            if eps_g == 0 and sales_g == 0:
                fins = t.quarterly_financials
                if fins is not None and not fins.empty and len(fins.columns) >= 5:
                    if 'Net Income' in fins.index:
                        curr_ni = fins.loc['Net Income'].iloc[0]
                        prev_ni = fins.loc['Net Income'].iloc[4]
                        if prev_ni and prev_ni > 0:
                            eps_g = ((curr_ni / prev_ni) - 1) * 100
                            
                    rev_keys = ['Total Revenue', 'Operating Revenue', 'Revenue']
                    rel_row = next((r for r in rev_keys if r in fins.index), None)
                    if rel_row:
                        curr_rev = fins.loc[rel_row].iloc[0]
                        prev_rev = fins.loc[rel_row].iloc[4]
                        if prev_rev and prev_rev > 0:
                            sales_g = ((curr_rev / prev_rev) - 1) * 100
                            
            if roe == 0:
                bal = t.quarterly_balance_sheet
                fins = t.quarterly_financials
                if bal is not None and not fins.empty and fins is not None and not fins.empty:
                    eq_keys = ['Common Stock Equity', 'Stockholders Equity', 'Total Equity Gross Minority Interest']
                    eq_row = next((r for r in eq_keys if r in bal.index), None)
                    if eq_row and 'Net Income' in fins.index:
                        ni_series = fins.loc['Net Income'].dropna()
                        eq_series = bal.loc[eq_row].dropna()
                        if len(ni_series) > 0 and len(eq_series) > 0:
                            if len(ni_series) >= 4:
                                trailing_ni = ni_series.iloc[:4].sum()
                            else:
                                trailing_ni = ni_series.iloc[0] * 4
                                
                            latest_eq = eq_series.iloc[0]
                            if latest_eq > 0:
                                roe = (trailing_ni / latest_eq) * 100
                                
            # Check if yfinance totally failed silently without throwing an exception (shadowban protection)
            if eps_g == 0 and sales_g == 0 and roe == 0 and industry == "Unknown" and mcap_b == 0:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    time.sleep(wait_time)
                    continue
                else:
                    if db_cache and ticker in db_cache:
                        cached = db_cache[ticker].copy()
                        mcap = cached.get('market_cap', 0.0)
                        cached['mcap_b'] = mcap / 1000000000.0 if mcap else 0.0
                        return cached, False
                    
            return {'eps_growth': float(eps_g), 'sales_growth': float(sales_g), 'roe': float(roe), 'market_cap': mcap, 'mcap_b': float(mcap_b), 'industry': industry}, True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # Wait 2s, 4s, 8s universally for any network error
                time.sleep(wait_time)
                continue
            
            if db_cache and ticker in db_cache:
                cached = db_cache[ticker].copy()
                mcap = cached.get('market_cap', 0.0)
                cached['mcap_b'] = mcap / 1000000000.0 if mcap else 0.0
                return cached, False
                
            return {'eps_growth': 0.0, 'sales_growth': 0.0, 'roe': 0.0, 'mcap_b': 0.0, 'industry': 'Unknown'}, False

def score_leaders(candidates, is_etf=False):
    """
    Score candidates and save component breakpoints for UI transparency.
    Total: 120 points (Growth 40 + Tech 40 + Liquidity 10 + Quality 10 + Institutional 20)
    """
    scored = []
    for c in candidates:
        if is_etf:
            growth_pts = 0
            qual_pts = 0
        else:
            # Growth (Max 40)
            eps = c.get('eps_growth', 0)
            sales = c.get('sales_growth', 0)
            eps_pts = min(20, max(0, eps / 2.5)) if eps > 0 else 0 
            sales_pts = min(20, max(0, sales / 2.5)) if sales > 0 else 0
            growth_pts = eps_pts + sales_pts
        
        # RS & Technicals (Max 40)
        rs = c['rs_score']
        dist = c['dist_high']
        rs_pts = max(0, ((rs - 80) / 19.0) * 25)
        dist_pts = max(0, ((20 - dist) / 20.0) * 15)
        tech_pts = rs_pts + dist_pts
        
        # Liquidity (Max 10)
        adtv = c['adtv_m']
        liq_pts = min(10, max(0, (adtv / 200.0) * 10))
        
        # Quality (Max 10)
        if not is_etf:
            roe = c.get('roe', 0)
            qual_pts = min(10, max(0, (roe / 25.0) * 10)) 
        
        # Institutional Footprints (Max 20)
        hv1_pts = 10 if c.get('hv1', False) else 0
        ud = c.get('ud_ratio', 1.0)
        if ud >= 2.0:
            ud_pts = 10
        elif ud >= 1.5:
            ud_pts = 7
        elif ud >= 1.1:
            ud_pts = 3
        else:
            ud_pts = 0
        inst_pts = hv1_pts + ud_pts
        
        score = growth_pts + tech_pts + liq_pts + qual_pts + inst_pts
        c['tml_score'] = round(score, 1)
        c['breakdown'] = {
            'growth': round(growth_pts, 1),
            'tech': round(tech_pts, 1),
            'liq': round(liq_pts, 1),
            'qual': round(qual_pts, 1),
            'inst': round(inst_pts, 1)
        }
        scored.append(c)
        
    scored.sort(key=lambda x: x['tml_score'], reverse=True)
    return scored
    
# -----------------------------------------------------------------------------
# MAIN APP UI
# -----------------------------------------------------------------------------
def main():
    from components import render_header
    render_header("👑 True Market Leader (US)", "Locating the Apex Predators of the US Market.")
    
    with st.expander("📖 True Market Leader Scoring Methodology (120 Points)", expanded=False):
        st.markdown("""
        **What makes a True Market Leader?**
        Inspired by CANSLIM practitioners Jim Roppel and David Ryan, the algorithm assigns a **120-point** composite score:
        
        🥇 **1. Explosive Growth (Max 40 Points)**
        - **Earnings Growth (up to 20 pts)**: Scales linearly. 50% YoY EPS growth equates to a perfect 20 points.
        - **Sales Growth (up to 20 pts)**: Scales linearly. 50% YoY Sales growth equates to a perfect 20 points.
        
        📈 **2. Price & Relative Strength (Max 40 Points)**
        - **Elite Relative Strength (up to 25 pts)**: Scales from an RS of 80 (0 pts) to an RS of 99 (25 pts). *Highest-weighted single metric — RS is king.*
          *(Formula: Percentile Rank of [1W Return × 20% + 1M Return × 40% + 3M Return × 40%])*
        - **Proximity to 52W High (up to 15 pts)**: The closer to the absolute high, the better. 0% distance = 15 pts.
        
        🌊 **3. Institutional Liquidity (Max 10 Points)**
        - **Average Daily Trading Value (up to 10 pts)**: TMLs must handle large fund inflows. Scales up to $200+ Million ADTV.
        
        🛡️ **4. Fundamental Quality (Max 10 Points)**
        - **Return on Equity (up to 10 pts)**: Measures capital efficiency. 25% ROE equates to 10 points.
        
        🏦 **5. Institutional Footprints (Max 20 Points)** *(NEW)*
        - **Highest Volume in 1 Year — HV1 (10 pts)**: If the stock printed its highest volume in 250 trading days at any point during the last 21 sessions, it signals fresh heavyweight accumulation.
        - **50-Day Up/Down Volume Ratio (10 pts)**: Measures sustained institutional buying pressure. Ratio ≥ 2.0 = 10 pts | 1.5–1.9 = 7 pts | 1.1–1.4 = 3 pts.
        
        👣 **Institutional Volume Footprints (Timing Filters)**
        - **Pocket Pivot (🟢)**: Triggers when today's up-volume is strictly greater than the highest down-volume over the prior 10 days. This indicates institutions are stepping in to accumulate quietly within a base. *Action: Use as an early, aggressive entry signal before a traditional breakout.*
        - **Volume Dry-Up / VDU (🔵)**: Triggers when daily volume drops > 50% below its 50-day average. This indicates sellers are completely exhausted and the stock is "dry". *Action: Watch closely; a VDU often precedes a violent breakout, as any slight demand will immediately push the price higher.*
        
        ⏱️ **Timing & Entry Indicator: Relative Measured Volatility (RMV)**
        - **What it is**: RMV identifies Volatility Contraction Patterns (VCP). It shows when a stock has absorbed selling pressure and is "coiling" tightly, signaling a high-probability entry point for an explosive move.
        - **How it's calculated**: A 15-day stochastic oscillator measuring the current day's True Range (High - Low) relative to the High/Low ranges of the last 3 weeks.
        - **How to read it**: It scales from **0 to 100**. A score between **0 to 15** indicates extreme tightness (Green/Actionable). A score near **100** indicates wide, loose price action (Red/Avoid). It is not part of the 120-point TML score, but acts as a pure timing filter.
        
        🎯 **Actionability Framework (Setup Status)**
        - 🚨 **Extended**: Price is extremely frothy (>15% above 21 EMA or >25% above 50 SMA). Too risky to buy.
        - 🔥 **Tight VCP**: Volatility is dead (RMV < 15), price is near highs, and volume is drying up. High conviction breakout setup.
        - 📉 **21 EMA Pullback**: Price has pulled back to within 5% of the 21-day EMA in a primary uptrend. Low risk entry.
        - 🚀 **10 EMA Power Trend**: Stock is in a confirmed Power Trend and riding within 2% of its 10-day EMA. Aggressive momentum add.
        - 🟡 **Building Base**: Normal consolidation. Not yet actionable.
        
        *Engine Pipeline: It dynamically pre-screens the US Market stocks, dropping any that fail to meet elite technical baselines (Stage 2 Uptrend, RS > 80, Close > $20). It then fetches live fundamentals only for survivors.*
        """)

    # Initialize session state for scan results
    if 'tml_results' not in st.session_state:
        st.session_state.tml_results = None
    if 'tml_funnel' not in st.session_state:
        st.session_state.tml_funnel = None

    st.markdown("### 🔭 Scanner Configuration")
    asset_class = st.radio("Asset Class:", ["Stocks", "ETFs"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab_apex, tab_sectors = st.tabs(['🦅 Apex Predators', '🔥 Sector Heat Rankings'])
    with tab_sectors:
        r1, r2, r3, r4 = st.columns([1, 1, 1, 1])
        with r1:
            heat_sort_us = st.radio("Sort Grid By:", ["RS Rating", "1M Velocity"], horizontal=True)
        with r2:
            weight_mode_us = st.radio("Weighting Method:", ["Equal", "Market Cap"], horizontal=True)
        with r3:
            min_rs_filter_us = st.slider("Min RS Rating:", min_value=0, max_value=99, value=0, step=5)
        with r4:
            min_breadth_filter_us = st.slider("Min Stage 2 Breadth (%):", min_value=0, max_value=100, value=0, step=10)
            
        if st.button("🔥 Load Sector Heat Rankings", use_container_width=True):
            with st.spinner("Fetching historical data and computing momentum..."):
                tickers = get_cached_universe(mode=asset_class)
                history_df = get_cached_history(tickers, days=252)
                db_cache = get_all_fundamentals_cache()
                rs_scores = compute_rs_scores_fast(history_df, tickers)
                st.session_state.heat_data_us = (history_df, tickers, db_cache, rs_scores)
        
        if 'heat_data_us' in st.session_state:
            history_df, tickers, db_cache, rs_scores = st.session_state.heat_data_us
            render_sector_heat_rankings(history_df, tickers, db_cache, rs_scores, sort_by=heat_sort_us, min_rs=min_rs_filter_us, min_breadth=min_breadth_filter_us, weight_mode=weight_mode_us)
    with tab_apex:
        c1, c2 = st.columns([1, 4])
        with c1:
            run_scan = st.button("🚀 Execute Apex Scan", type="primary", use_container_width=True)
        with c2:
            clear_cache = st.button("🔄 Force Refresh Data Cache")
            if clear_cache:
                st.cache_data.clear()
                st.session_state.tml_results = None
                st.session_state.tml_funnel = None
                st.rerun()

        if run_scan:
            start_time = time.time()
            
            # Use st.status to bundle all the phantom loading text neatly 
            with st.status("Deploying TML Engine...", expanded=True) as status:
                
                st.write(f"1. Parsing Universe ({asset_class})...")
                tickers = get_cached_universe(mode=asset_class)
                
                st.write("2. Downloading Cached Matrix Technicals (Last 252 Days)...")
                history_df = get_cached_history(tickers, days=252)
                
                st.write("3. Calculating Institutional Alpha & Stage 2 Baselines...")
                rs_scores = compute_rs_scores_fast(history_df, tickers)
                pre_screened, funnel = run_technical_prescreen(history_df, tickers, rs_scores, is_etf=(asset_class == 'ETFs'))
                
                st.write(f"4. Found {len(pre_screened)} Stage 2 elites. Initiating Yahoo Finance fundamental JIT pipeline...")
                
                if len(pre_screened) > 0:
                    if asset_class == "ETFs":
                        st.write(f"4. Found {len(pre_screened)} ETFs. Skipping fundamentals fetch for ETFs...")
                        final_leaders = pre_screened
                        for etf in final_leaders:
                            etf['eps_growth'] = 0.0
                            etf['sales_growth'] = 0.0
                            etf['roe'] = 0.0
                            etf['mcap_b'] = 0.0
                            etf['industry'] = 'ETF'
                    else:
                        prog_bar = st.progress(0)
                        
                        db_cache = get_all_fundamentals_cache()
                        new_successful_fetches = []
                        
                        def process_stock(stock, index):
                            ticker = stock['ticker']
                            time.sleep(np.random.uniform(0.1, 0.5) + (index * 0.1))
                            funds, is_new = fetch_fundamentals(ticker, db_cache=db_cache)
                            
                            if is_new and not (funds['eps_growth'] == 0 and funds['sales_growth'] == 0 and funds['roe'] == 0 and funds['industry'] == 'Unknown'):
                                new_dict = funds.copy()
                                new_dict['ticker'] = ticker
                                new_successful_fetches.append(new_dict)
                                
                            return {**stock, **funds}
        
                        completed = 0
                        final_leaders = []
                        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                            future_to_stock = {executor.submit(process_stock, c_stock, idx): c_stock for idx, c_stock in enumerate(pre_screened)}
                            for future in concurrent.futures.as_completed(future_to_stock):
                                final_leaders.append(future.result())
                                completed += 1
                                prog_bar.progress(completed / len(pre_screened), text=f"Analyzed SEC Filings ({completed}/{len(pre_screened)})")
                        
                        if new_successful_fetches:
                            save_fundamentals_cache(new_successful_fetches)
                
                final_leaders = score_leaders(final_leaders, is_etf=(asset_class == "ETFs"))
                
                # Save top 40 snapshot to database
                if final_leaders:
                    save_tml_snapshot("US", final_leaders, top_n=40)
                
                st.session_state.tml_results = final_leaders
                st.session_state.tml_funnel = funnel
                
                elapsed = time.time() - start_time
                status.update(label=f"Apex Scan Complete! {len(final_leaders)} stocks vetted in {elapsed:.1f}s", state="complete", expanded=False)

        # -------------------------------------------------------------------------
        # RENDER RESULTS FROM SESSION STATE
        # -------------------------------------------------------------------------
        if st.session_state.tml_results is not None:
            final_leaders = st.session_state.tml_results
            funnel = st.session_state.tml_funnel
            
            st.markdown("---")
            
            from components import render_metric_card
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                target = funnel.get('Target Universe', 0)
                valid = funnel.get('Valid Data Fetched', 0)
                render_metric_card("Stocks Scanned (Valid)", f"{valid} / {target}")
            with mc2:
                top_score = final_leaders[0]['tml_score'] if final_leaders else 0
                max_score = 70 if asset_class == "ETFs" else 120
                threshold = 55 if asset_class == "ETFs" else 90
                render_metric_card("Highest TML Score", f"{top_score}/{max_score}", color_class="green-text" if top_score > threshold else "blue-text")
            with mc3:
                top_ind = "N/A"
                if final_leaders:
                    ind_counts = {}
                    for d in final_leaders[:10]:
                        ind = d.get('industry', 'Unknown')
                        ind_counts[ind] = ind_counts.get(ind, 0) + 1
                    if ind_counts:
                        top_ind = sorted(ind_counts.items(), key=lambda x: x[1], reverse=True)[0][0]
                render_metric_card("Leading Sector", top_ind)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 1. Visualization of the Drop Funnel
            st.markdown("### 🌪️ The Rejection Funnel")
            st.caption("Jim Roppel says to brutally filter the market noise. Here is how the US Universe was decimated:")
            
            funnel_df = pd.DataFrame(list(funnel.items()), columns=['Stage', 'Count'])
            fig_funnel = px.funnel(funnel_df, x='Count', y='Stage', 
                                   color_discrete_sequence=['#4338ca'])
            fig_funnel.update_layout(template='plotly_dark', margin=dict(l=0, r=0, t=10, b=0), height=300)
            st.plotly_chart(fig_funnel, use_container_width=True)
                    
            st.markdown("<br>", unsafe_allow_html=True)
            
            if not final_leaders:
                # Empty State Analytics
                st.error("📉 **Zero True Market Leaders Detected**")
                st.info("Market Context: The strict institutional algorithm found 0 stocks passing the requirements. Total Market Breadth has likely collapsed, forcing stocks under their 50-day moving averages and crushing Relative Strength. Cash is highly recommended during severe stage 4 market conditions.")
                return
                
            st.markdown("### 🏆 Apex Predator & Sector Leadership")
            c_radar, c_sector = st.columns([1, 1])
            
            with c_radar:
                apex = final_leaders[0]
                clean_apex_ticker = apex['ticker']
                st.markdown(f"**#1 Ranked: {clean_apex_ticker}**")
                
                categories = ['Growth', 'Technicals', 'Liquidity', 'Quality']
                b = apex['breakdown']
                g_pct = (b['growth'] / 40.0) * 100 if 'growth' in b else 0
                t_pct = (b['tech'] / 30.0) * 100 if 'tech' in b else 0
                l_pct = (b['liq'] / 20.0) * 100 if 'liq' in b else 0
                q_pct = (b['qual'] / 10.0) * 100 if 'qual' in b else 0
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=[g_pct, t_pct, l_pct, q_pct, g_pct],
                    theta=categories + [categories[0]],
                    fill='toself',
                    fillcolor='rgba(67, 56, 202, 0.5)',
                    line=dict(color='#6366f1'),
                    name=clean_apex_ticker
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], ticksuffix='%')
                    ),
                    showlegend=False,
                    margin=dict(l=40, r=40, t=20, b=20),
                    height=250,
                    template='plotly_dark'
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
            with c_sector:
                st.markdown("**Leading Industries (Top 10)**")
                top_10 = final_leaders[:10]
                ind_counts = {}
                for d in top_10:
                    ind = d.get('industry', 'Unknown')
                    ind_counts[ind] = ind_counts.get(ind, 0) + 1
                sorted_inds = sorted(ind_counts.items(), key=lambda x: x[1], reverse=True)
                for ind, count in sorted_inds:
                    st.markdown(f"- **{ind}**: {count} leaders")
                    
            st.markdown("### 📈 True Market Leaders")
            st.info("💡 **Symbol Guide:** &nbsp;&nbsp; 👑 **Apex Predator** (RS > 95, EPS/Sales Growth > 40%) &nbsp;&nbsp;|&nbsp;&nbsp; 🐺 **Sector Wolfpack** (3+ leaders) &nbsp;&nbsp;|&nbsp;&nbsp; 🐘 **Liq Expansion** &nbsp;&nbsp;|&nbsp;&nbsp; 🛡️ **Buy@21EMA** &nbsp;&nbsp;|&nbsp;&nbsp; 🚨 **Micro Signal**")
            
            show_top_20 = st.toggle("Show Top 20 Leaders", value=False)
            show_rs_blue_dot_only = st.checkbox("Show RS Blue Dot Breakouts Only (William O'Neil 🔵)", value=False)
            
            active_leaders = final_leaders
            if show_rs_blue_dot_only:
                active_leaders = [x for x in active_leaders if x.get('rs_blue_dot', False)]
                
            leaders_to_show = active_leaders[:20] if show_top_20 else active_leaders[:10]
            
            recent_signals = get_recent_intraday_signals("US", days=3)
            
            from multibagger_analyzer import analyze_multibagger_stocks
            
            # Fetch Readiness Scores for the displayed leaders
            top_20_raw = [r['ticker'] for r in leaders_to_show]
            hist_df = get_cached_history(top_20_raw, 430)
            
            pre_fetched = {}
            if hist_df is not None and not hist_df.empty:
                for r in leaders_to_show:
                    t = r['ticker']
                    raw_t = t
                    try:
                        df = pd.DataFrame()
                        if hist_df.columns.nlevels == 2:
                            try:
                                df['close'] = hist_df.xs('Close', level=0, axis=1)[t]
                                df['high'] = hist_df.xs('High', level=0, axis=1)[t]
                                df['low'] = hist_df.xs('Low', level=0, axis=1)[t]
                                df['volume'] = hist_df.xs('Volume', level=0, axis=1)[t]
                            except KeyError:
                                df['close'] = hist_df.xs('Close', level=1, axis=1)[t]
                                df['high'] = hist_df.xs('High', level=1, axis=1)[t]
                                df['low'] = hist_df.xs('Low', level=1, axis=1)[t]
                                df['volume'] = hist_df.xs('Volume', level=1, axis=1)[t]
                        elif hist_df.columns.nlevels == 1 and len(leaders_to_show) == 1:
                            df['close'] = hist_df['Close']
                            df['high'] = hist_df['High']
                            df['low'] = hist_df['Low']
                            df['volume'] = hist_df['Volume']
                        
                        # Inject fundamental/RS data into the last row for the analyzer
                        df['rs_score'] = r.get('rs_score', None)
                        df['eps_growth'] = r.get('eps_growth', None)
                        df['sales_growth'] = r.get('sales_growth', None)
                        df['roe'] = r.get('roe', None)
                        df['opm'] = r.get('opm', None)
                        df['debt_to_equity'] = r.get('debt_to_equity', None)
                        
                        df.index.name = 'Date'
                        df.dropna(subset=['close'], inplace=True)
                        pre_fetched[raw_t] = df
                    except Exception as e:
                        print(f"Error extracting {t} for readiness: {e}")
            
            readiness_results = analyze_multibagger_stocks({t: 'US' for t in top_20_raw}, pre_fetched_data=pre_fetched)
            readiness_map = {res['ticker']: res.get('buy_readiness', {}) for res in readiness_results}
            
            display_data = []
            for i, rank in enumerate(leaders_to_show):
                clean_ticker = rank['ticker']
                breakdown_str = f"Growth: {rank['breakdown']['growth']}/{40} | Tech: {rank['breakdown']['tech']}/{40} | Liq: {rank['breakdown']['liq']}/{10} | Qual: {rank['breakdown']['qual']}/{10} | Inst: {rank['breakdown'].get('inst', 0)}/{20}"
                
                is_apex = (rank['rs_score'] >= 95 and 
                           rank.get('eps_growth', 0) >= 40 and 
                           rank.get('sales_growth', 0) >= 40 and 
                           rank['close'] > rank.get('ema_21', 0))
                ticker_display = f"{clean_ticker}"
                if is_apex:
                    ticker_display += " 👑"
                if rank.get('rs_blue_dot', False):
                    ticker_display += " 🔵"
                if rank.get('Is_Hold_The_Line', False):
                    ticker_display += " 🛡️"
                    
                # Check for recent signals
                if clean_ticker in recent_signals and recent_signals[clean_ticker]:
                    # Grab the most recent signal for this ticker
                    sig_name, sig_date = recent_signals[clean_ticker][0]
                    delta = (datetime.now() - datetime.strptime(sig_date, "%Y-%m-%d")).days
                    ago_str = "Today" if delta == 0 else f"{delta}d ago"
                    ticker_display += f" 🚨 {sig_name} ({ago_str})"
                    
                readiness = readiness_map.get(clean_ticker, {})
                read_score = readiness.get('score', 0)
                read_label = readiness.get('label', '⚪ Not Ready')
                read_breakdown = " | ".join([f"{k}: {v}" for k, v in readiness.get('breakdown', {}).items()])
                    
                display_data.append({
                    'Rank': len(display_data) + 1,
                    'Setup Status': rank.get('Action_Status', '🟡 Building Base'),
                    'Readiness': f"{read_label} [{read_score}/15]",
                    'Why?': read_breakdown,
                    'Chart': f"https://www.tradingview.com/chart/?symbol={clean_ticker}",
                    'TML Score': rank['tml_score'],
                    'Breakdown Detail': breakdown_str,
                    'Ticker': ticker_display,
                    'Industry': rank.get('industry', 'Unknown'),
                    'Trend (6M)': rank.get('trend_6m', []),
                    'Price': f"${rank['close']:.1f}",
                    'RS Score': round(rank['rs_score'], 1),
                    'Ext 50D': round(rank.get('ext_50d_atr', 0), 1),
                    'EPS Gr %': round(rank['eps_growth'], 1),
                    'Sales Gr %': round(rank['sales_growth'], 1),
                    'ADTV ($M)': round(rank['adtv_m'], 1),
                    'Dist High %': round(rank['dist_high'], 1),
                    'ROE %': round(rank.get('roe', 0), 1),
                    'HV1_Flag': rank.get('hv1', False),
                    'PocketPivot': rank.get('pocket_pivot', False),
                    'VDU_5D': rank.get('vdu', False),
                    'Power Trend': rank.get('Is_Power_Trend', False),
                    'Ants': rank.get('is_ants', False),
                    'Crossback': rank.get('Is_Crossback', False),
                    '3WT': rank.get('Is_3WT', False),
                    'Power Play': '🔥' if rank.get('htf', False) else '—',
                    'UD_Ratio': round(rank.get('ud_ratio', 1.0), 2),
                    'Mom Score': round(rank.get('clenow_score', 0), 1),
                    'R²': round(rank.get('clenow_r2', 0), 2),
                    'RMV (15D)': round(rank.get('rmv_15d', 100.0), 1),
                    'TML_Days': 0, # Will be populated below
                    'Liq_Expansion': rank.get('Liq_Expansion', False),
                })
                
            # Fetch Persistence Data
            persistence_dict = get_tml_persistence("US", days=90)
            for d in display_data:
                clean_t = d['Ticker'].split()[0]
                d['TML_Days'] = persistence_dict.get(clean_t, 1 if d['TML Score'] > 0 else 0)
                
            df_disp = pd.DataFrame(display_data)

            # Sector Wolfpack Detection
            wolfpack_industries = []
            if final_leaders:
                top_20_inds = [d.get('industry', 'Unknown') for d in final_leaders[:20]]
                from collections import Counter
                ind_counts_20 = Counter(top_20_inds)
                wolfpack_industries = [ind for ind, count in ind_counts_20.items() if count >= 3 and ind != 'Unknown']
                
                if wolfpack_industries:
                    alert_text = "🐺 **WOLFPACKS DETECTED:** "
                    details = []
                    for ind in wolfpack_industries:
                        count = ind_counts_20[ind]
                        details.append(f"**{ind}** ({count} stocks)")
                    st.warning(alert_text + " | ".join(details))
                    
            # Append 🐺 to the industry name in the dataframe
            if 'Industry' in df_disp.columns and wolfpack_industries:
                df_disp['Industry'] = df_disp['Industry'].apply(lambda x: f"🐺 {x}" if x in wolfpack_industries else x)
                    
            # Format Liquidity Expansion visually
            if 'Liq_Expansion' in df_disp.columns:
                df_disp['Liq_Exp'] = df_disp['Liq_Expansion'].apply(lambda x: "🌊 YES" if x else "")
                
            # Table View Selector
            view_mode = st.radio(
                "Select Table View:",
                ["📊 Main Overview", "🏢 Fundamentals", "🐋 Institutional Flow", "📐 Technicals"],
                horizontal=True
            )
            
            base_cols = ['Rank', 'Setup Status', 'Readiness', 'Why?', 'Chart', 'Ticker', 'Industry']
            
            if "Overview" in view_mode:
                show_cols = base_cols + ['TML Score', 'Trend (6M)', 'Price', 'RS Score', 'Ext 50D', 'Dist High %']
            elif "Fundamentals" in view_mode:
                show_cols = base_cols + ['EPS Gr %', 'Sales Gr %', 'ROE %', 'ADTV ($M)', 'Breakdown Detail']
            elif "Institutional" in view_mode:
                show_cols = base_cols + ['Liq_Exp', 'UD_Ratio', 'HV1_Flag', 'PocketPivot', 'VDU_5D']
            else: # Technicals
                show_cols = base_cols + ['RMV (15D)', '3WT', 'Ants', 'Crossback', 'Power Trend', 'Mom Score', 'R²', 'TML_Days']
                
            if df_disp.empty:
                df_render = pd.DataFrame(columns=show_cols)
            else:
                df_render = df_disp[show_cols]
            
            format_dict = {
                'TML Score': "{:.1f}",
                'Ext 50D': "{:+.1f}x",
                'EPS Gr %': "{:+.1f}%",
                'Sales Gr %': "{:+.1f}%",
                'Dist High %': "{:.1f}%",
                'ROE %': "{:.1f}%"
            }
            active_format = {k: v for k, v in format_dict.items() if k in df_render.columns}
            
            styled_df = df_render.style.format(active_format)
            
            if 'EPS Gr %' in df_render.columns and 'Sales Gr %' in df_render.columns:
                styled_df = styled_df.background_gradient(cmap='Greens', subset=['EPS Gr %', 'Sales Gr %'])
            if 'Ext 50D' in df_render.columns:
                styled_df = styled_df.background_gradient(cmap='Reds', subset=['Ext 50D'])
            if 'RMV (15D)' in df_render.columns:
                styled_df = styled_df.background_gradient(cmap='RdYlGn_r', subset=['RMV (15D)'])
                
            # Color styling
            st.dataframe(
                styled_df,
                column_config={
                    "Setup Status": st.column_config.TextColumn("Setup Status", help="Actionability based on VCP and EMA pullbacks"),
                    "Readiness": st.column_config.TextColumn("Readiness", help="Composite Setup Score (Minervini + VCP)"),
                    "Why?": st.column_config.TextColumn("Why?", help="Detailed Breakdown of Buy Readiness"),
                    "Chart": st.column_config.LinkColumn("Chart", display_text="View 📈"),
                    "Trend (6M)": st.column_config.LineChartColumn("Trend (6M)", y_min=0),
                    "TML Score": st.column_config.ProgressColumn("TML Score", format="%.1f", min_value=0, max_value=120),
                    "RS Score": st.column_config.ProgressColumn("RS Score", format="%.1f", min_value=0, max_value=99),
                    "Dist High %": st.column_config.ProgressColumn("Dist High %", format="%.1f%%", min_value=0, max_value=100),
                    "Ext 50D": st.column_config.NumberColumn("Ext 50D (ATR)", format="%+.1fx", help="Distance from 50-day SMA in terms of ATR(21)"),
                    "Power Trend": st.column_config.CheckboxColumn("⚡ Power Trend", help="Strict Uptrend Regime"),
                    "Ants": st.column_config.CheckboxColumn("🐜 Ants", help="12/15 days up"),
                    "3WT": st.column_config.CheckboxColumn("🗜️ 3WT", help="David Ryan 3-Weeks Tight (Variance <= 2%)"),
                    "Crossback": st.column_config.CheckboxColumn("⚔️ Crossback", help="Oliver Kell 10 EMA Shakeout & Reclaim"),
                    "Mom Score": st.column_config.NumberColumn("Mom Score", format="%.1f", help="Clenow Momentum: Annualized Slope × R² (trend quality, not part of TML score)"),
                    "R²": st.column_config.ProgressColumn("R² Smooth", format="%.2f", min_value=0.0, max_value=1.0, help="Trend smoothness (1.0 = perfect staircase)"),
                    "RMV (15D)": st.column_config.NumberColumn("RMV (15D)", format="%.1f", help="Relative Measured Volatility (15-Day). 0-15 = Tight Coiling. 100 = High Volatility."),
                    "TML_Days": st.column_config.NumberColumn("TML Days (90d)", format="%d", help="Days appeared in Top 20 over the last 90 days"),
                    "Liq_Expansion": None, # Hide raw boolean
                    "Liq_Exp": st.column_config.TextColumn("🐘 Liq Exp", help="Volume Expansion (Any of last 5 days >300% OR 5d Avg >150% of 50d Avg)")
                },
                use_container_width=True,
                hide_index=True,
                height=430
            )
            
            # Quick Copy for TradingView
            tv_list = ",".join([d['Ticker'].split()[0] for d in display_data])
            with st.expander(f"📋 Copy TML Leaders for TradingView ({len(display_data)})"):
                st.code(tv_list, language="text")
                
            # =========================================================================
            # TML INTELLIGENCE (NEWS RADAR)
            # =========================================================================
            st.markdown("---")
            st.markdown("### 📡 TML Intelligence Radar")
            st.markdown("Real-time institutional narrative and catalysts for current US True Market Leaders (Last 48 hours).")
            
            active_tickers = [d['Ticker'].split()[0] for d in display_data]
            
            if active_tickers:
                try:
                    from news_fetcher import fetch_portfolio_news
                    with st.spinner("Scanning global feeds for US TML catalysts..."):
                        portfolio_news = fetch_portfolio_news(active_tickers, max_per_ticker=2)
                        
                    if portfolio_news:
                        # Render Deepvue-inspired sleek grid
                        news_html = """
<style>
.news-marquee-wrapper {
    position: relative;
    width: 100%;
    overflow: hidden;
    margin-top: 20px;
    margin-bottom: 30px;
    padding: 10px 0;
}
.news-marquee-wrapper::before,
.news-marquee-wrapper::after {
    content: '';
    position: absolute;
    top: 0;
    bottom: 0;
    width: 120px;
    z-index: 2;
    pointer-events: none;
}
.news-marquee-wrapper::before {
    left: 0;
    background: linear-gradient(to right, #0b1121, transparent);
}
.news-marquee-wrapper::after {
    right: 0;
    background: linear-gradient(to left, #0b1121, transparent);
}
.news-marquee-track {
    display: flex;
    gap: 16px;
    width: max-content;
    animation: scroll-marquee 40s linear infinite;
}
.news-marquee-wrapper:hover .news-marquee-track {
    animation-play-state: paused;
}
@keyframes scroll-marquee {
    0% { transform: translateX(0); }
    100% { transform: translateX(calc(-50% - 8px)); }
}
.news-card {
    flex: 0 0 320px;
    background: rgba(20, 25, 40, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 16px;
    text-decoration: none !important;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 140px;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.news-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: #334155;
    transition: all 0.2s ease;
}
.news-card:hover {
    background: rgba(30, 40, 60, 0.8);
    border-color: rgba(255, 255, 255, 0.1);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.news-card.news-card-fresh:hover {
    border-color: rgba(0, 255, 157, 0.3);
}
.news-card:hover::before {
    background: #3b82f6;
}
.news-card.news-card-fresh {
    background: linear-gradient(145deg, rgba(20, 30, 45, 0.8), rgba(15, 25, 35, 0.9));
}
.news-card.news-card-fresh::before {
    background: #00ff9d;
    box-shadow: 0 0 10px #00ff9d;
}
.fresh-badge {
    color: #00ff9d;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    background: rgba(0, 255, 157, 0.1);
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid rgba(0, 255, 157, 0.2);
}
.news-badge {
    background: rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 8px;
    letter-spacing: 0.5px;
}
.news-title {
    color: #f8fafc;
    font-size: 0.9rem;
    font-weight: 600;
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-bottom: 8px;
}
.news-card:hover .news-title {
    color: #60a5fa;
}
.news-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: auto;
}
.news-publisher {
    color: #94a3b8;
    font-size: 0.75rem;
    display: flex;
    align-items: center;
    gap: 4px;
}
.news-time {
    color: #64748b;
    font-size: 0.75rem;
    display: flex;
    align-items: center;
    gap: 4px;
}
</style>
<div class="news-marquee-wrapper">
    <div class="news-marquee-track">
"""
                        cards_html = ""
                        import re
                        for item in portfolio_news:
                            # Sanitize text
                            title_safe = item['title'].replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                            pub_safe = item['publisher'].replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                            ticker_safe = item['clean_ticker']
                            link = item['link']
                            time_ago = item['time_ago']
                            
                            # Determine if fresh (<= 5 hours)
                            is_fresh = False
                            if "minute" in time_ago.lower():
                                is_fresh = True
                            elif "hour" in time_ago.lower():
                                match = re.search(r'\d+', time_ago)
                                if match and int(match.group()) <= 5:
                                    is_fresh = True
                                    
                            fresh_class = " news-card-fresh" if is_fresh else ""
                            time_html = f'<span class="fresh-badge">🔥 FRESH ({time_ago})</span>' if is_fresh else f'<span class="news-time"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>{time_ago}</span>'
                            
                            cards_html += f"""
<a href="{link}" target="_blank" class="news-card{fresh_class}">
    <div>
        <span class="news-badge">{ticker_safe}</span>
        <div class="news-title">{title_safe}</div>
    </div>
    <div class="news-meta">
        <span class="news-publisher">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"></path><path d="M18 14h-8"></path><path d="M15 18h-5"></path><path d="M10 6h8v4h-8V6Z"></path></svg>
            {pub_safe}
        </span>
        {time_html}
    </div>
</a>
"""
                        # Duplicate the cards exactly once for the seamless infinite loop
                        news_html += cards_html + cards_html
                        news_html += """
    </div>
</div>
"""
                        st.markdown(news_html, unsafe_allow_html=True)
                    else:
                        st.info("No major catalysts or news stories detected for US TML stocks in the last 48 hours.")
                except Exception as e:
                    st.error(f"📡 News radar temporarily offline. Error: {e}")
            else:
                st.info("No US TML stocks available to scan for news.")

            st.markdown("---")
            # =========================================================================
            # YOUTUBE MEDIA & INTERVIEWS
            # =========================================================================
            try:
                import media_fetcher
                import importlib
                importlib.reload(media_fetcher)
                from media_fetcher import extract_portfolio_interviews
                
                with st.spinner("Fetching latest management interviews..."):
                    interviews = extract_portfolio_interviews(active_tickers, [], region="US")
                    
                if interviews:
                    st.markdown("### 🎙️ Media & Interviews")
                    st.markdown("<p style='color: #94a3b8; font-size: 0.9rem;'>Recent management appearances on leading financial networks</p>", unsafe_allow_html=True)
                    
                    css_and_html = """
<style>
.media-marquee-wrapper {
    width: 100%;
    overflow: hidden;
    white-space: nowrap;
    position: relative;
    padding: 10px 0 30px 0;
}
.media-marquee-wrapper::before,
.media-marquee-wrapper::after {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    width: 100px;
    z-index: 2;
    pointer-events: none;
}
.media-marquee-wrapper::before {
    left: 0;
    background: linear-gradient(to right, #0b1120, transparent);
}
.media-marquee-wrapper::after {
    right: 0;
    background: linear-gradient(to left, #0b1120, transparent);
}
.media-marquee-track {
    display: inline-flex;
    gap: 20px;
    animation: marquee-media 45s linear infinite;
    padding-left: 20px;
}
.media-marquee-wrapper:hover .media-marquee-track {
    animation-play-state: paused;
}
@keyframes marquee-media {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
.media-card {
    background: linear-gradient(145deg, rgba(30,41,59,0.4) 0%, rgba(15,23,42,0.8) 100%);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 16px;
    padding: 15px;
    display: inline-flex;
    flex-direction: column;
    gap: 12px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    text-decoration: none !important;
    position: relative;
    overflow: hidden;
    width: 320px;
    white-space: normal;
    flex-shrink: 0;
}
.media-card:hover {
    transform: translateY(-5px);
    border-color: rgba(239, 68, 68, 0.4);
    box-shadow: 0 12px 24px -10px rgba(239, 68, 68, 0.25);
    background: linear-gradient(145deg, rgba(30,41,59,0.6) 0%, rgba(15,23,42,0.95) 100%);
}
.media-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, rgba(239, 68, 68, 0.9), transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
}
.media-card:hover::before {
    opacity: 1;
}
.media-thumb-wrapper {
    width: 100%;
    aspect-ratio: 16/9;
    border-radius: 10px;
    overflow: hidden;
    position: relative;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.media-thumb {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.5s ease;
}
.media-card:hover .media-thumb {
    transform: scale(1.05);
}
.play-btn {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(0,0,0,0.7);
    border-radius: 50%;
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0;
    transition: opacity 0.3s ease;
    backdrop-filter: blur(2px);
}
.media-card:hover .play-btn {
    opacity: 1;
}
.play-btn svg {
    width: 20px;
    height: 20px;
    fill: #f8fafc;
    margin-left: 3px;
}
.media-title {
    color: #f1f5f9 !important;
    font-weight: 700;
    font-size: 0.95rem;
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin: 0;
}
.media-card:hover .media-title {
    color: #ffffff !important;
}
.media-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: auto;
    padding-top: 12px;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.channel-badge {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 4px;
    border: 1px solid rgba(239, 68, 68, 0.2);
}
.time-badge {
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 4px;
}
</style>
<div class="media-marquee-wrapper">
"""
                    media_duration = max(15, len(interviews) * 5)
                    css_and_html += f"""<div class="media-marquee-track" style="animation: marquee-media {media_duration}s linear infinite;">\n"""
                    
                    cards_html = ""
                    for v in interviews:
                        video_url = f"https://www.youtube.com/watch?v={v['video_id']}"
                        cards_html += f"""
<a href="{video_url}" target="_blank" class="media-card">
    <div class="media-thumb-wrapper">
        <img src="{v['thumbnail']}" class="media-thumb">
        <div class="play-btn">
            <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
        </div>
    </div>
    <div class="media-title">{v['title']}</div>
    <div class="media-meta">
        <span class="channel-badge">📺 {v['channel']}</span>
        <span class="time-badge">🕒 {v['display_date']}</span>
    </div>
</a>
"""
                    css_and_html += cards_html + cards_html
                    css_and_html += """</div>
</div>
"""
                    st.markdown(css_and_html, unsafe_allow_html=True)
                else:
                    st.markdown("### 🎙️ Media & Interviews")
                    st.info("No recent management interviews or updates were found for US TML stocks on the major networks in the past 14 days.")
                    
            except Exception as e:
                print(f"Error loading media fetcher: {e}")
                
            # Pin to Focus List
            st.markdown("---")
            st.markdown("### ⭐ Focus List Quick Pin")
            st.markdown("Push top setups to your nightly actionable Focus List.")
            pin_cols = st.columns([2, 1, 3])
            with pin_cols[0]:
                pin_ticker = st.selectbox("Select Stock to Pin:", [d['Ticker'].split()[0] for d in display_data], key="pin_tml_us")
            with pin_cols[1]:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("⭐ Pin to Focus List", key="btn_pin_tml_us", use_container_width=True):
                    if add_to_focus_list(pin_ticker, "US"):
                        st.success(f"Pinned {pin_ticker} to Focus List!")
                    else:
                        st.error("Failed to pin stock.")

            # -----------------------------------------------------------
            # LIQUIDITY MONSTERS (Top 15 by ADTV)
            # -----------------------------------------------------------
            st.markdown("### 🐋 Institutional Heavyweights (Liquidity Monsters)")
            st.caption("These stocks passed the strict technical gates and have the absolute highest daily dollar volume. Massive ADTV confirms institutional footprints.")
            
            c_liq1, c_liq2 = st.columns([1, 3])
            with c_liq1:
                liq_rs_filter = st.slider("Minimum RS Score", min_value=80, max_value=99, value=85, key="liq_rs_us")
                
            filtered_liq = [x for x in final_leaders if x['rs_score'] >= liq_rs_filter]
            liquidity_leaders = sorted(filtered_liq, key=lambda x: x['adtv_m'], reverse=True)[:15]
            
            liq_data = []
            for rank in liquidity_leaders:
                clean_ticker = rank['ticker']
                liq_data.append({
                    'Setup Status': rank.get('Action_Status', '🟡 Building Base'),
                    'Ticker': clean_ticker,
                    'ADTV ($M)': round(rank['adtv_m'], 1),
                    'TML Score': rank['tml_score'],
                    'RS Score': round(rank['rs_score'], 1),
                    'Industry': rank.get('industry', 'Unknown'),
                    'Price': f"${rank['close']:.1f}",
                    'HV1': '🟢' if rank.get('hv1', False) else '—',
                    'U/D Vol': round(rank.get('ud_ratio', 1.0), 2),
                    'Mom Score': round(rank.get('clenow_score', 0), 1),
                    'R²': round(rank.get('clenow_r2', 0), 2),
                    'RMV (15D)': round(rank.get('rmv_15d', 100.0), 1),
                    'TML_Days': persistence_dict.get(rank['ticker'], 1)
                })
                
            if not liq_data:
                st.info("No liquidity monsters found for the current criteria.")
            else:
                df_liq = pd.DataFrame(liq_data)
                st.dataframe(
                    df_liq.style.format({
                        'TML Score': "{:.1f}"
                    }).background_gradient(cmap='RdYlGn_r', subset=['RMV (15D)']),
                    column_config={
                        "Setup Status": st.column_config.TextColumn("Setup Status", help="Actionability based on VCP and EMA pullbacks"),
                        "ADTV ($M)": st.column_config.ProgressColumn("ADTV ($M)", format="$%.1fM", min_value=0, max_value=float(max([d['ADTV ($M)'] for d in liq_data] + [100.0]))),
                        "TML Score": st.column_config.NumberColumn("TML Score", format="%.1f"),
                        "RS Score": st.column_config.NumberColumn("RS Score", format="%.1f"),
                        "Mom Score": st.column_config.NumberColumn("Mom Score", format="%.1f", help="Clenow Momentum: Annualized Slope × R² (display only)"),
                        "R²": st.column_config.ProgressColumn("R² Smooth", format="%.2f", min_value=0.0, max_value=1.0, help="Trend smoothness (1.0 = perfect)"),
                        "RMV (15D)": st.column_config.NumberColumn("RMV (15D)", format="%.1f", help="Relative Measured Volatility (15-Day). 0-15 = Tight Coiling. 100 = High Volatility."),
                        "TML_Days": st.column_config.NumberColumn("TML Days (90d)", format="%d")
                    },
                    use_container_width=True,
                    hide_index=True
                )

            # -----------------------------------------------------------
            # HALL OF FAME
            # -----------------------------------------------------------
            st.markdown("### 🏆 TML Hall of Fame (Consistency Leaders)")
            st.caption("Stocks that have appeared in the Top 20 most frequently over the last 90 days. These are the true, enduring market leaders.")
            
            hof_data = get_tml_hall_of_fame("US", days=90, limit=20)
            
            if hof_data:
                df_hof = pd.DataFrame(hof_data)
                df_hof['Chart'] = df_hof['ticker'].apply(lambda x: f"https://www.tradingview.com/chart/?symbol={x}")
                leaders_map = {x['ticker']: x for x in final_leaders}
                def get_hof_ticker_display(t_clean):
                    is_apex = False
                    is_blue_dot = False
                    is_htt = False
                    if t_clean in leaders_map:
                        rank = leaders_map[t_clean]
                        is_apex = (rank['rs_score'] >= 95 and 
                                   rank.get('eps_growth', 0) >= 40 and 
                                   rank.get('sales_growth', 0) >= 40 and 
                                   rank['close'] > rank.get('ema_21', 0))
                        is_blue_dot = rank.get('rs_blue_dot', False)
                        is_htt = rank.get('Is_Hold_The_Line', False)
                    
                    display_val = f"{t_clean}"
                    if is_apex:
                        display_val += " 👑"
                    if is_blue_dot:
                        display_val += " 🔵"
                    if is_htt:
                        display_val += " 🛡️"
                    return display_val
                    
                df_hof['ticker'] = df_hof['ticker'].apply(get_hof_ticker_display)
                
                # Reorder columns
                df_hof = df_hof[['Chart', 'ticker', 'days_on_list', 'industry', 'tml_score', 'rs_score', 'action_status', 'last_seen']]
                
                st.dataframe(
                    df_hof.style.format({
                        'tml_score': "{:.1f}",
                        'rs_score': "{:.1f}"
                    }),
                    column_config={
                        "Chart": st.column_config.LinkColumn("Chart", display_text="View 📈"),
                        "ticker": st.column_config.TextColumn("Ticker"),
                        "days_on_list": st.column_config.ProgressColumn("Days in Top 20 (90d)", format="%d", min_value=0, max_value=90),
                        "industry": st.column_config.TextColumn("Industry"),
                        "tml_score": st.column_config.NumberColumn("Latest TML Score", format="%.1f"),
                        "rs_score": st.column_config.NumberColumn("Latest RS Score", format="%.1f"),
                        "action_status": st.column_config.TextColumn("Setup Status"),
                        "last_seen": st.column_config.TextColumn("Last Seen Date")
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No historical data found yet. Run scans daily to build the Hall of Fame.")
                
            # Dynamic Roppel Quadrant
            st.markdown("### 🔭 The Roppel Quadrant System")
            st.caption("Apex Predators live in the top-right corner.")
            
            if asset_class == "ETFs":
                y_axis_choice = st.selectbox(
                    "Select Technical Y-Axis (ETF Mode):",
                    ["3M Historical Volatility (Annualized %)", "Extension from 50D SMA %", "Clenow R² (Trend Smoothness)", "3-Month Return %", "Relative Measured Volatility (15D)"],
                    index=0
                )
                y_col_map = {
                    "3M Historical Volatility (Annualized %)": ("vol_3m", "3M Annualized Volatility (%)"),
                    "Extension from 50D SMA %": ("ext_50d", "Extension from 50D SMA (%)"),
                    "Clenow R² (Trend Smoothness)": ("clenow_r2", "Trend Smoothness (R²)"),
                    "3-Month Return %": ("ret_3m", "3-Month Return (%)"),
                    "Relative Measured Volatility (15D)": ("rmv_15d", "Relative Measured Volatility (15D) [Lower = Tighter]")
                }
            else:
                y_axis_choice = st.selectbox(
                    "Select Fundamental Y-Axis:",
                    ["Sales Growth %", "EPS Growth %", "Return on Equity (ROE) %", "Relative Measured Volatility (15D)"],
                    index=0
                )
                y_col_map = {
                    "Sales Growth %": ("sales_growth", "YoY Sales Growth (%)"),
                    "EPS Growth %": ("eps_growth", "YoY EPS Growth (%)"),
                    "Return on Equity (ROE) %": ("roe", "Return on Equity (%)"),
                    "Relative Measured Volatility (15D)": ("rmv_15d", "Relative Measured Volatility (15D) [Lower = Tighter]")
                }
            y_col, y_label = y_col_map[y_axis_choice]
            
            if y_axis_choice == "Relative Measured Volatility (15D)":
                st.info("💡 **Apex Target Zone Changed**: When plotting RMV, the ideal setups are located in the **Bottom-Right Corner** (High RS > 90, Low RMV < 15).")
            
            plot_df = pd.DataFrame(final_leaders)
            # Removed > 0 filter so missing or negative turnaround fundamentals still map visually
            
            if not plot_df.empty:
                if 'mcap_b' not in plot_df.columns:
                    plot_df['mcap_b'] = 0.0
                    
                if asset_class != "ETFs":
                    st.markdown("#### Segment by Market Cap")
                    mcap_filter = st.radio(
                        "Filter scatter plot by company size:",
                        ["All", "Small Cap (< $2B)", "Mid Cap ($2B - $10B)", "Large Cap (> $10B)"],
                        horizontal=True,
                        label_visibility="collapsed"
                    )
                    
                    if mcap_filter == "Small Cap (< $2B)":
                        plot_df = plot_df[plot_df['mcap_b'] < 2]
                    elif mcap_filter == "Mid Cap ($2B - $10B)":
                        plot_df = plot_df[(plot_df['mcap_b'] >= 2) & (plot_df['mcap_b'] <= 10)]
                    elif mcap_filter == "Large Cap (> $10B)":
                        plot_df = plot_df[plot_df['mcap_b'] > 10]
                        
                plot_df['clean_ticker'] = plot_df['ticker']
                
                # --- INSTITUTIONAL TABS ---
                st.markdown("### 🔭 Institutional Visualizations")
                tab_roppel, tab_power, tab_vcp, tab_flow, tab_risk, tab_tq, tab_treemap, tab_fundas = st.tabs([
                    "Roppel Quadrant", 
                    "Inst Footprint",
                    "VCP Coiling", 
                    "Institutional Flow", 
                    "Extension Risk",
                    "Trend Quality",
                    "Industry Treemap",
                    "Fundamentals CAN SLIM"
                ])
                
                with tab_roppel:
                    st.caption("Apex Predators live in the top-right corner. High RS + Elite Fundamentals.")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        color_by = st.radio("Color Bubbles By:", ["TML Score", "Industry"], horizontal=True, key="color_roppel_us")
                    with col2:
                        show_trails = st.toggle("☄️ Show RS Momentum Trails (Last 10 Days)", value=False, key="show_trails_us")
                        
                    color_col = 'tml_score' if color_by == "TML Score" else 'industry'
                    color_scale = 'RdYlGn' if color_by == "TML Score" else px.colors.qualitative.Bold
                    
                    fig = px.scatter(
                        plot_df,
                        x='rs_score',
                        y=y_col,
                        size='adtv_m',
                        color=color_col,
                        hover_name='clean_ticker',
                        text='clean_ticker',
                        custom_data=['clean_ticker'],
                        color_continuous_scale=color_scale if color_by == "TML Score" else None,
                        color_discrete_sequence=color_scale if color_by == "Industry" else None,
                        labels={
                            'rs_score': 'Relative Strength (0-99)',
                            y_col: y_label,
                            'tml_score': 'TML Composite Score',
                            'industry': 'Industry'
                        }
                    )
                    fig.update_traces(textposition='top center')
                    fig.update_layout(height=600, template='plotly_dark')
                    
                    if show_trails and not plot_df.empty:
                        from database import get_historical_rs_for_tickers
                        tickers_list = plot_df['ticker'].tolist()
                        history_dict = get_historical_rs_for_tickers('US', tickers_list, days=10)
                        
                        for idx, row in plot_df.iterrows():
                            t = row['ticker']
                            hist_rs = history_dict.get(t, [])
                            if hist_rs:
                                x_vals = hist_rs + [row['rs_score']]
                                y_vals = [row[y_col]] * len(x_vals)
                                
                                # UI/UX: Determine velocity color (Red for losing RS, Green for gaining)
                                delta = row['rs_score'] - hist_rs[0]
                                if delta > 0:
                                    trail_color = 'rgba(50, 205, 50, 0.25)' # Softer Green
                                    origin_color = 'rgba(50, 205, 50, 0.6)'
                                else:
                                    trail_color = 'rgba(255, 69, 0, 0.25)' # Softer Red
                                    origin_color = 'rgba(255, 69, 0, 0.6)'
                                    
                                # Create an anchor dot at the start of the tail (oldest point)
                                marker_sizes = [4] + [0] * (len(x_vals) - 1)
                                
                                fig.add_trace(go.Scatter(
                                    x=x_vals,
                                    y=y_vals,
                                    mode='lines+markers',
                                    line=dict(color=trail_color, width=1.0),
                                    marker=dict(size=marker_sizes, color=origin_color),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))

                    fig.add_hline(y=40, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                    fig.add_vline(x=90, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                    
                    # Elite Apex Predator Zone Shading
                    max_y = plot_df[y_col].max() if not plot_df.empty else 100
                    zone_top = max(max_y, 50) * 1.1
                    fig.add_shape(
                        type="rect",
                        x0=90, y0=40, x1=100, y1=zone_top,
                        line=dict(color="rgba(0,0,0,0)"),
                        fillcolor="rgba(50, 205, 50, 0.1)",
                        layer="below"
                    )
                    fig.add_annotation(
                        x=95, y=zone_top * 0.95,
                        text="👑 Apex Predators",
                        showarrow=False,
                        font=dict(color="#32cd32", size=14, weight="bold")
                    )
                    
                    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
                    
                    selected_tickers = []
                    if event and 'selection' in event and 'points' in event['selection'] and event['selection']['points']:
                        selected_tickers = [p['customdata'][0] for p in event['selection']['points'] if 'customdata' in p]
                        
                    if not selected_tickers:
                        selected_tickers = plot_df['clean_ticker'].tolist()
                        
                    tv_list_dynamic = ",".join(selected_tickers)
                    with st.expander(f"📋 Copy Quadrant Tickers for TradingView ({len(selected_tickers)})"):
                        st.code(tv_list_dynamic, language="text")
                        st.caption("Use the 'Lasso Select' or 'Box Select' tool from the chart toolbar (top right) to select points and generate a custom list.")
                    
                    st.info('''
                    **Chart Legend:**
                    - **Circle Size:** Institutional Liquidity (Average Daily Trading Value in Millions). Larger circles highlight mega/large-caps capable of absorbing massive mutual fund volume.
                    - **Circle Color:** Overall 120-Point TML Composite Score OR Industry grouping.
                    - **Apex Predators:** Look for large bubbles isolated in the top-right quadrant.
                    ''')
                    
                with tab_power:
                    st.caption("Institutional Footprints. Look for high RS stocks with massive clusters of 4% Power Days over the last 3 months (Top-Right).")
                    
                    if 'Power_Days_3M' not in plot_df.columns:
                        plot_df['Power_Days_3M'] = 0
                        
                    fig_power = px.scatter(
                        plot_df,
                        x='rs_score',
                        y='Power_Days_3M',
                        size='adtv_m',
                        color='tml_score',
                        hover_name='clean_ticker',
                        text='clean_ticker',
                        custom_data=['clean_ticker'],
                        color_continuous_scale='RdYlGn',
                        labels={
                            'rs_score': 'Relative Strength (0-99)',
                            'Power_Days_3M': 'Number of 4% Up Days (Last 3M)',
                            'tml_score': 'TML Composite Score'
                        }
                    )
                    fig_power.update_traces(textposition='top center')
                    fig_power.update_layout(height=600, template='plotly_dark')
                    
                    fig_power.add_hline(y=3, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                    fig_power.add_vline(x=90, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                    
                    max_power = plot_df['Power_Days_3M'].max() if not plot_df.empty else 10
                    max_power = max(max_power * 1.1, 5)
                    
                    fig_power.add_shape(
                        type="rect",
                        x0=90, y0=3, x1=100, y1=max_power,
                        line=dict(color="rgba(0,0,0,0)"),
                        fillcolor="rgba(50, 205, 50, 0.15)",
                        layer="below"
                    )
                    fig_power.add_annotation(
                        x=95, y=max_power * 0.9,
                        text="🐘 Massive Accumulation",
                        showarrow=False,
                        font=dict(color="#32cd32", size=14, weight="bold")
                    )
                    st.plotly_chart(fig_power, use_container_width=True)
                    
                    st.info('''
                    **Chart Legend:**
                    - **Y-Axis:** Count of days where the stock closed >= 4% up in the last 3 months (approx 65 trading days).
                    - **X-Axis:** CANSLIM Relative Strength percentile rank (0-99).
                    - **Circle Size:** Average Daily Trading Value (ADTV) in Millions.
                    - **Why 4%?** A retail trader cannot move a stock by 4% in a day. A cluster of these 4% up days is an undeniable institutional footprint showing hedge funds and mutual funds are aggressively building a position.
                    ''')
                    
                with tab_vcp:
                    st.caption("Volatility Contraction (VCP) Actionability Matrix. Holy Grail setups are tightly coiled near 52-week highs (Top-Right).")
                    # Reversed axes so best is top right
                    fig_vcp = px.scatter(
                        plot_df,
                        x='dist_high',
                        y='rmv_15d',
                        size='adtv_m',
                        color='rs_score',
                        hover_name='clean_ticker',
                        text='clean_ticker',
                        custom_data=['clean_ticker'],
                        color_continuous_scale='RdYlGn',
                        labels={
                            'dist_high': 'Distance from 52W High (%) [0% is best]',
                            'rmv_15d': 'Relative Measured Volatility 15D [0 is tightest]',
                            'rs_score': 'Relative Strength'
                        }
                    )
                    fig_vcp.update_traces(textposition='top center')
                    # Reverse axes
                    fig_vcp.update_layout(height=600, template='plotly_dark')
                    fig_vcp.update_xaxes(autorange="reversed")
                    fig_vcp.update_yaxes(autorange="reversed")
                    
                    fig_vcp.add_hline(y=15, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                    fig_vcp.add_vline(x=5, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                    
                    fig_vcp.add_shape(
                        type="rect",
                        x0=5, y0=15, x1=0, y1=0,
                        line=dict(color="rgba(0,0,0,0)"),
                        fillcolor="rgba(99, 102, 241, 0.15)",
                        layer="below"
                    )
                    fig_vcp.add_annotation(
                        x=2.5, y=7.5,
                        text="🎯 Coiled & Ready",
                        showarrow=False,
                        font=dict(color="#818cf8", size=14, weight="bold")
                    )
                    st.plotly_chart(fig_vcp, use_container_width=True)
                    
                with tab_flow:
                    st.caption("Stealth Accumulation. Look for high RS stocks with massive Up/Down Volume Ratios (Top-Right).")
                    fig_flow = px.scatter(
                        plot_df,
                        x='ud_ratio',
                        y='rs_score',
                        size='adtv_m',
                        color='tml_score',
                        hover_name='clean_ticker',
                        text='clean_ticker',
                        custom_data=['clean_ticker'],
                        color_continuous_scale='RdYlGn',
                        labels={
                            'ud_ratio': '50-Day Up/Down Volume Ratio',
                            'rs_score': 'Relative Strength',
                            'tml_score': 'TML Score'
                        }
                    )
                    fig_flow.update_traces(textposition='top center')
                    fig_flow.update_layout(height=600, template='plotly_dark')
                    
                    fig_flow.add_hline(y=90, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                    fig_flow.add_vline(x=1.5, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                    
                    fig_flow.add_shape(
                        type="rect",
                        x0=1.5, y0=90, x1=max(plot_df['ud_ratio'].max() * 1.1, 2.0) if not plot_df.empty else 3.0, y1=100,
                        line=dict(color="rgba(0,0,0,0)"),
                        fillcolor="rgba(245, 158, 11, 0.15)",
                        layer="below"
                    )
                    fig_flow.add_annotation(
                        x=1.75, y=95,
                        text="🐋 Heavy Accumulation",
                        showarrow=False,
                        font=dict(color="#f59e0b", size=14, weight="bold")
                    )
                    st.plotly_chart(fig_flow, use_container_width=True)
                    
                with tab_risk:
                    st.caption("Rubber Band Extension Risk. Identifies high RS stocks that are too extended from their 50-Day SMA to buy safely.")
                    fig_risk = px.scatter(
                        plot_df,
                        x='ext_50d',
                        y='rs_score',
                        size='adtv_m',
                        color='rmv_15d',
                        hover_name='clean_ticker',
                        text='clean_ticker',
                        custom_data=['clean_ticker'],
                        color_continuous_scale='RdYlGn_r', # Red is high RMV (loose)
                        labels={
                            'ext_50d': 'Extension from 50D SMA (%)',
                            'rs_score': 'Relative Strength',
                            'rmv_15d': 'Volatility (RMV)'
                        }
                    )
                    fig_risk.update_traces(textposition='top center')
                    fig_risk.update_layout(height=600, template='plotly_dark')
                    
                    fig_risk.add_hline(y=90, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                    fig_risk.add_vline(x=5, line_dash="dash", line_color="rgba(50, 205, 50, 0.5)")
                    fig_risk.add_vline(x=20, line_dash="dash", line_color="rgba(220, 38, 38, 0.5)")
                    
                    # Danger Zone
                    fig_risk.add_shape(
                        type="rect",
                        x0=20, y0=80, x1=max(plot_df['ext_50d'].max() * 1.1, 30.0) if not plot_df.empty else 40.0, y1=100,
                        line=dict(color="rgba(0,0,0,0)"),
                        fillcolor="rgba(220, 38, 38, 0.15)",
                        layer="below"
                    )
                    fig_risk.add_annotation(
                        x=25, y=95,
                        text="⚠️ Chasing Danger Zone",
                        showarrow=False,
                        font=dict(color="#ef4444", size=14, weight="bold")
                    )
                    
                    # Safe Zone
                    fig_risk.add_shape(
                        type="rect",
                        x0=0, y0=90, x1=5, y1=100,
                        line=dict(color="rgba(0,0,0,0)"),
                        fillcolor="rgba(50, 205, 50, 0.15)",
                        layer="below"
                    )
                    fig_risk.add_annotation(
                        x=2.5, y=95,
                        text="✅ Safe Base",
                        showarrow=False,
                        font=dict(color="#22c55e", size=14, weight="bold")
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)

                with tab_tq:
                    st.caption("Cross-referencing systematic trend smoothness (R²) against relative strength. Top-right = the smoothest, strongest uptrends — where both systems agree.")
                    
                    fig_tq = px.scatter(
                        plot_df,
                        x='rs_score',
                        y='clenow_r2',
                        size='adtv_m',
                        color='tml_score',
                        hover_name='clean_ticker',
                        text='clean_ticker',
                        custom_data=['clean_ticker'],
                        color_continuous_scale='RdYlGn',
                        labels={
                            'rs_score': 'Relative Strength (0-99)',
                            'clenow_r2': 'R² Trend Smoothness (0.0 – 1.0)',
                            'tml_score': 'TML Composite Score'
                        }
                    )
                    fig_tq.update_traces(textposition='top center')
                    fig_tq.update_layout(height=600, template='plotly_dark')
                    fig_tq.add_hline(y=0.65, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                    fig_tq.add_vline(x=90, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                    
                    # High-Conviction Zone: RS > 90 & R² > 0.65
                    fig_tq.add_shape(
                        type="rect",
                        x0=90, y0=0.65, x1=100, y1=1.0,
                        line=dict(color="rgba(0,0,0,0)"),
                        fillcolor="rgba(99, 102, 241, 0.12)",
                        layer="below"
                    )
                    fig_tq.add_annotation(
                        x=95, y=0.97,
                        text="🎯 Smoothest Leaders",
                        showarrow=False,
                        font=dict(color="#818cf8", size=14, weight="bold")
                    )
                    
                    # Choppy Zone Label: Low R²
                    fig_tq.add_annotation(
                        x=84, y=0.15,
                        text="⚠️ Erratic / Choppy",
                        showarrow=False,
                        font=dict(color="rgba(255,255,255,0.3)", size=11)
                    )
                    
                    tq_event = st.plotly_chart(fig_tq, use_container_width=True, on_select="rerun")
                    
                    tq_selected = []
                    if tq_event and 'selection' in tq_event and 'points' in tq_event['selection'] and tq_event['selection']['points']:
                        tq_selected = [p['customdata'][0] for p in tq_event['selection']['points'] if 'customdata' in p]
                        
                    if not tq_selected:
                        tq_selected = plot_df['clean_ticker'].tolist()
                        
                    tv_tq_list = ",".join(tq_selected)
                    with st.expander(f"📋 Copy Trend Quality Tickers for TradingView ({len(tq_selected)})"):
                        st.code(tv_tq_list, language="text")
                        st.caption("Lasso/Box select to filter. High R² + High RS = the highest-conviction entries where Clenow's systematic math and CANSLIM's discretionary method converge.")
                    
                    st.info('''
                    **Trend Quality Chart Legend:**
                    - **Y-Axis (R²):** How smoothly the stock's price follows its 90-day exponential trendline. 0.80+ = staircase-like trend, 0.30 = erratic/volatile.
                    - **X-Axis (RS):** CANSLIM Relative Strength percentile rank (0-99).
                    - **Circle Size:** Average Daily Trading Value (ADTV) in $ Millions.
                    - **Circle Color:** TML Composite Score (120 pts).
                    - **🎯 Smoothest Leaders Zone:** RS > 90 AND R² > 0.65 — these stocks are rising fast AND doing it with institutional-grade consistency. Ideal for both systematic and discretionary traders.
                    ''')

                with tab_treemap:
                    st.caption("Sector Breadth & Dominance. Larger blocks mean more highly liquid stocks breaking out from that industry.")
                    if 'industry' in plot_df.columns:
                        plot_df['industry_clean'] = plot_df['industry'].fillna('Unknown')
                        
                        fig_tree = px.treemap(
                            plot_df, 
                            path=[px.Constant("True Market Leaders"), 'industry_clean', 'clean_ticker'],
                            values='adtv_m',
                            color='tml_score',
                            hover_data=['rs_score', 'adtv_m'],
                            color_continuous_scale='RdYlGn',
                            labels={'tml_score': 'Avg TML Score'}
                        )
                        fig_tree.update_traces(root_color="rgba(0,0,0,0)")
                        fig_tree.update_layout(height=650, template='plotly_dark', margin=dict(t=30, l=10, r=10, b=10))
                        st.plotly_chart(fig_tree, use_container_width=True)
                    else:
                        st.info("Industry data unavailable.")

                with tab_fundas:
                    if asset_class == "ETFs":
                        st.info("Fundamentals quadrant is not applicable for ETFs.")
                    else:
                        st.caption("CAN SLIM Core: Identifying the fastest-growing true market leaders. Top-right zone is explosive.")
                        
                        # Ensure columns exist, default to 0 if not
                        for col in ['sales_growth', 'eps_growth']:
                            if col not in plot_df.columns:
                                plot_df[col] = 0.0
                        
                        fig_fundas = px.scatter(
                            plot_df,
                            x='sales_growth',
                            y='eps_growth',
                            size='adtv_m',
                            color='tml_score',
                            hover_name='clean_ticker',
                            text='clean_ticker',
                            custom_data=['clean_ticker'],
                            color_continuous_scale='RdYlGn',
                            labels={
                                'sales_growth': 'Sales Growth YoY (%)',
                                'eps_growth': 'EPS Growth YoY (%)',
                                'tml_score': 'TML Composite Score'
                            }
                        )
                        fig_fundas.update_traces(textposition='top center')
                        fig_fundas.update_layout(height=600, template='plotly_dark')
                        
                        # Determine dynamic maxes for plot bounds
                        max_x = max(plot_df['sales_growth'].max() * 1.1, 50.0) if not plot_df.empty else 50.0
                        max_y = max(plot_df['eps_growth'].max() * 1.1, 50.0) if not plot_df.empty else 50.0
                        
                        fig_fundas.add_hline(y=25, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                        fig_fundas.add_vline(x=25, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                        
                        # Highlight Sweet Spot Zone (>25% Sales & EPS)
                        fig_fundas.add_shape(
                            type="rect",
                            x0=25, y0=25, x1=max_x, y1=max_y,
                            line=dict(color="rgba(0,0,0,0)"),
                            fillcolor="rgba(50, 205, 50, 0.15)",
                            layer="below"
                        )
                        fig_fundas.add_annotation(
                            x=35, y=35,
                            text="⭐ CAN SLIM Zone",
                            showarrow=False,
                            font=dict(color="#32cd32", size=14, weight="bold")
                        )
                        
                        st.plotly_chart(fig_fundas, use_container_width=True)



def render_sector_heat_rankings(history_df, tickers, db_cache, rs_scores, sort_by="RS Rating", min_rs=0, min_breadth=0, weight_mode="Equal"):
    """Renders the CSS Grid for Sector Heat Rankings."""
    import streamlit as st
    import industry_matrix
    
    with st.spinner("Calculating Sector Velocity and Breadth..."):
        industry_map = {t: db_cache.get(t, {}).get('industry', 'Unknown') for t in tickers}
        heat_df = industry_matrix.get_sector_heat_rankings_data(history_df, tickers, industry_map, rs_scores, weight_mode=weight_mode, db_cache=db_cache)
        
    if heat_df.empty:
        st.warning("Not enough sector data to generate heat rankings.")
        return
        
    if min_rs > 0:
        heat_df = heat_df[heat_df['Score'] >= min_rs]
        
    if min_breadth > 0:
        heat_df = heat_df[heat_df['Participation_%'] >= min_breadth]
        
    if heat_df.empty:
        st.info(f"No sectors found matching your filters (Min RS: {min_rs}, Min Breadth: {min_breadth}%).")
        return
        
    if sort_by == "1M Velocity":
        heat_df = heat_df.sort_values(by='Score_Change_1M', ascending=False).reset_index(drop=True)
        
    with st.expander("📊 Advanced Analytics: Relative Rotation Graph (RRG)", expanded=False):
        st.info("""
        **How to read the Relative Rotation Graph (RRG):**
        * **The Goal:** Spot capital rotation. Look for sectors migrating from the bottom-left (**Lagging**) into the top-left (**Improving**), heading towards the top-right (**Leading**).
        * **Bubble Size (Breadth):** Represents the percentage of stocks in the sector currently in Stage 2 Uptrends. A larger bubble indicates broad, healthy participation.
        * **Bubble Color (Momentum Phase):** 
          * 🟢 **Green:** Outperforming (Strong RS & Accelerating Velocity)
          * 🔵 **Blue:** Accumulating (Weak RS but Accelerating Velocity)
          * 🟡 **Yellow:** Consolidating (Strong RS but Decelerating Velocity)
          * 🔴 **Red:** Underperforming (Weak RS & Decelerating Velocity)
        """)
        import plotly.express as px
        fig_rrg = px.scatter(
            heat_df,
            x="Score",
            y="Score_Change_1M",
            hover_name="Industry",
            text="Industry",
            color="Status",
            color_discrete_map={
                "OUTPERFORMING": "#34d399", 
                "ACCUMULATING": "#3b82f6", 
                "CONSOLIDATING": "#fcd34d", 
                "UNDERPERFORMING": "#f87171"
            },
            size="Participation_%",
            size_max=30
        )
        # Format text to only show first two words of industry to prevent overlap
        fig_rrg.update_traces(
            textposition='top center',
            textfont=dict(size=11),
            marker=dict(line=dict(width=1, color='rgba(150,150,150,0.5)'))
        )
        
        # Add Quadrant Crosshairs
        fig_rrg.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_rrg.add_vline(x=60, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add Quadrant Labels
        fig_rrg.add_annotation(x=85, y=max(heat_df['Score_Change_1M'].max(), 5), text="LEADING", showarrow=False, font=dict(color="#34d399", size=18, weight="bold"), opacity=0.4)
        fig_rrg.add_annotation(x=25, y=max(heat_df['Score_Change_1M'].max(), 5), text="IMPROVING", showarrow=False, font=dict(color="#3b82f6", size=18, weight="bold"), opacity=0.4)
        fig_rrg.add_annotation(x=85, y=min(heat_df['Score_Change_1M'].min(), -5), text="WEAKENING", showarrow=False, font=dict(color="#fcd34d", size=18, weight="bold"), opacity=0.4)
        fig_rrg.add_annotation(x=25, y=min(heat_df['Score_Change_1M'].min(), -5), text="LAGGING", showarrow=False, font=dict(color="#f87171", size=18, weight="bold"), opacity=0.4)
        
        fig_rrg.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Relative Strength (RS Rating)",
            yaxis_title="1M Velocity (Rate of Change)",
            xaxis=dict(range=[0, 100], showgrid=False),
            yaxis=dict(showgrid=False),
            height=650,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_rrg, use_container_width=True, theme="streamlit")
        
    with st.expander("📖 How to interpret Sector Heat Rankings", expanded=False):
        st.markdown("""
        **1. RS Rating (1-99):**  
        *How it's calculated:* We generate a synthetic "Sector Index" by aggregating the daily price action of all stocks belonging to that sector. We then measure the performance of that synthetic index over 3 timeframes (1 Week, 1 Month, 3 Months). These returns are heavily weighted toward recent price action (20% 1W, 40% 1M, 40% 3M) and then percentile ranked against all other sectors from 1 to 99. An RS of 99 means the sector's blended momentum is outperforming 99% of the market.
        
        **2. 1M Velocity:**  
        *How it's calculated:* This measures the absolute point change in the sector's raw Momentum Score over the exact last 21 trading days (1 Month). If a sector had a raw score of 50 last month and is now at 65, the velocity is +15.0. It perfectly highlights institutional rotation—sectors where smart money is *currently* flowing.  
        *Why it might be 0.0:* If a sector has a velocity of 0.0, it means either the sector has been perfectly flat, or (more commonly) the underlying stocks are too newly listed to have a full 6-month price history, preventing the engine from establishing a baseline score 21 days ago.
        
        **3. Apex Predators:**  
        *What they are:* The absolute strongest Top 2 individual stocks within this specific industry.  
        *How they are chosen:* The engine scans every single stock inside the sector, calculates their individual stock-level RS Ratings using the same weighted math, and extracts the 2 stocks with the highest ratings. These are the true market leaders pulling the entire sector upwards.
        
        **4. Stage 2 Breadth:**  
        *What it means:* The exact percentage of stocks within the industry that have an individual RS Rating > 80. If an industry has 10 stocks and 8 of them have RS > 80, the breadth is 80%. High breadth confirms massive, broad-based institutional accumulation, whereas low breadth warns that a sector is being propped up by just one or two mega-cap stocks.
        """)
        
    def get_industry_icon(name):
        n = name.lower()
        if 'semi' in n or 'chip' in n or 'electronic' in n: return '🔬'
        if 'software' in n or 'it ' in n or 'technology' in n or 'computer' in n or 'digital' in n or 'hardware' in n: return '💻'
        if 'bank' in n or 'financial' in n or 'capital' in n or 'credit' in n or 'insurance' in n or 'exchange' in n: return '🏦'
        if 'pharma' in n or 'health' in n or 'medical' in n or 'biotech' in n or 'diagnostic' in n or 'hospital' in n: return '💊'
        if 'auto' in n or 'vehicle' in n or 'motor' in n or 'tyre' in n: return '🚗'
        if 'energy' in n or 'oil' in n or 'gas' in n or 'power' in n or 'petroleum' in n or 'solar' in n or 'renew' in n: return '⚡'
        if 'telecom' in n or 'communication' in n or 'network' in n: return '📡'
        if 'metal' in n or 'mining' in n or 'steel' in n or 'copper' in n: return '⛏️'
        if 'chem' in n or 'fertilizer' in n or 'plastic' in n: return '🧪'
        if 'consumer' in n or 'retail' in n or 'fmcg' in n or 'food' in n or 'beverage' in n or 'sugar' in n or 'tea' in n: return '🛒'
        if 'real estate' in n or 'realty' in n or 'construction' in n or 'infra' in n or 'cement' in n: return '🏗️'
        if 'textile' in n or 'apparel' in n or 'garment' in n: return '👕'
        if 'media' in n or 'entertainment' in n or 'cinema' in n: return '🎬'
        if 'transport' in n or 'logistics' in n or 'shipping' in n or 'airline' in n or 'marine' in n: return '✈️'
        if 'equipment' in n or 'machinery' in n or 'industrial' in n or 'capital goods' in n or 'engineering' in n or 'cable' in n: return '⚙️'
        if 'utility' in n or 'utilities' in n: return '🔌'
        if 'defense' in n or 'aerospace' in n: return '🛡️'
        if 'paper' in n or 'wood' in n or 'packaging' in n or 'glass' in n: return '📦'
        if 'hotel' in n or 'leisure' in n or 'tourism' in n or 'hospitality' in n: return '🏨'
        if 'agriculture' in n or 'farming' in n or 'agro' in n or 'pesticide' in n: return '🌾'
        return '📊'
        
    sector_constituents = {}
    for t in tickers:
        clean_t = t.replace('.NS', '')
        rs = rs_scores.get(clean_t) or rs_scores.get(t)
        ind = industry_map.get(t, "Unknown")
        if ind == "Unknown" or pd.isna(ind) or rs is None: continue
        if ind not in sector_constituents: sector_constituents[ind] = []
        sector_constituents[ind].append((clean_t, rs))
        
    for ind in sector_constituents:
        sector_constituents[ind].sort(key=lambda x: x[1], reverse=True)
        
    def generate_sparkline_svg(scores, width=280, height=35, color="#34d399"):
        if not hasattr(scores, '__len__') or len(scores) < 2: return ""
        min_y, max_y = min(scores), max(scores)
        if max_y == min_y: max_y = min_y + 1
        
        pts = []
        for i, val in enumerate(scores):
            x = (i / (len(scores) - 1)) * width
            y = height - ((val - min_y) / (max_y - min_y) * height)
            pts.append(f"{x:.1f},{y:.1f}")
            
        path_d = "M " + " L ".join(pts)
        fill_d = path_d + f" L {width},{height} L 0,{height} Z"
        
        return f"""<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" preserveAspectRatio="none" style="margin-bottom: 12px; border-radius: 4px; overflow: visible;">
            <defs>
                <linearGradient id="grad_{color.replace('#', '')}" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stop-color="{color}" stop-opacity="0.25" />
                    <stop offset="100%" stop-color="{color}" stop-opacity="0.0" />
                </linearGradient>
            </defs>
            <path d="{fill_d}" fill="url(#grad_{color.replace('#', '')})" stroke="none"/>
            <path d="{path_d}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>"""
        
    html_payload = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.sector-dashboard { font-family: 'Inter', sans-serif; background: transparent; padding: 10px 0; }
.heat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; }
.heat-card {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px; padding: 20px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex; flex-direction: column; justify-content: space-between;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.2);
    color: #f8fafc; position: relative; overflow: hidden;
}
.heat-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: transparent;
}
.heat-card:hover {
    transform: translateY(-5px); box-shadow: 0 12px 30px -5px rgba(0, 0, 0, 0.3);
    border-color: rgba(255, 255, 255, 0.15);
}
.glow-hot::before { background: linear-gradient(90deg, #10b981, #34d399); }
.glow-cold::before { background: linear-gradient(90deg, #ef4444, #f87171); }
.glow-warm::before { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.heat-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.heat-rank {
    font-size: 0.85rem; font-weight: 800; background: rgba(255,255,255,0.1);
    padding: 4px 10px; border-radius: 20px; color: #f1f5f9; letter-spacing: 0.5px;
}
.heat-status {
    font-size: 0.75rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 1px; padding: 4px 10px; border-radius: 8px;
}
.status-hot { background: rgba(16, 185, 129, 0.15); color: #34d399; }
.status-cold { background: rgba(239, 68, 68, 0.15); color: #f87171; }
.status-warm { background: rgba(59, 130, 246, 0.15); color: #60a5fa; }
.heat-title {
    font-size: 1.25rem; font-weight: 800; color: #ffffff;
    margin-bottom: 16px; line-height: 1.3; letter-spacing: -0.5px;
}
.heat-metrics-container {
    display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;
    background: rgba(0,0,0,0.2); padding: 12px; border-radius: 12px;
}
.metric-box { display: flex; flex-direction: column; }
.metric-label { 
    color: #94a3b8; font-size: 0.75rem; font-weight: 600; 
    text-transform: uppercase; margin-bottom: 4px;
}
.metric-val { font-size: 1.4rem; font-weight: 800; line-height: 1; }
.heat-apex { font-size: 0.85rem; color: #cbd5e1; margin-bottom: 16px; line-height: 1.4; }
.apex-label {
    font-size: 0.75rem; color: #64748b; text-transform: uppercase;
    font-weight: 700; margin-bottom: 4px; display: block;
}
.apex-tickers { color: #e2e8f0; font-weight: 600; }
.breadth-section { margin-top: auto; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.05); }
.breadth-header { display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; font-weight: 600; margin-bottom: 8px; }
.breadth-bar-bg {
    width: 100%; height: 8px; background: rgba(0,0,0,0.3);
    border-radius: 4px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
}
.breadth-bar-fill { height: 100%; border-radius: 4px; transition: width 1s ease-out; }
.heat-drilldown { margin-top: 12px; background: rgba(0,0,0,0.2); border-radius: 8px; overflow: hidden; }
.heat-drilldown summary {
    padding: 8px 12px; font-size: 0.75rem; font-weight: 600; color: #94a3b8;
    cursor: pointer; text-transform: uppercase; list-style: none; user-select: none;
}
.heat-drilldown summary::-webkit-details-marker { display: none; }
.heat-drilldown summary:hover { color: #f1f5f9; background: rgba(255,255,255,0.05); }
.drilldown-content {
    padding: 10px; display: flex; flex-wrap: wrap; gap: 6px;
    max-height: 120px; overflow-y: auto; border-top: 1px solid rgba(255,255,255,0.05);
}
.drilldown-content::-webkit-scrollbar { width: 4px; }
.drilldown-content::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
.constit-badge { font-size: 0.7rem; padding: 2px 6px; border: 1px solid; border-radius: 4px; background: rgba(0,0,0,0.3); transition: all 0.2s; }
.constit-badge:hover { background: rgba(255,255,255,0.1); cursor: pointer; border-color: rgba(255,255,255,0.3) !important; }
</style>
<div class="sector-dashboard">
<div class="heat-grid">
"""
    for idx, row in heat_df.iterrows():
        rank = row['Rank']
        industry = row['Industry']
        score = row['Score']
        mom = row['Score_Change_1M']
        status = row['Status']
        apex = row['Apex_Predators']
        breadth = min(100.0, max(0.0, row['Participation_%']))
        icon = get_industry_icon(industry)
        
        if mom >= 5.0 and score >= 70:
            status_class = "status-hot"
            status_text = f"🔥 ACCELERATING"
            glow_class = "glow-hot"
            bar_color = "linear-gradient(90deg, #10b981, #34d399)"
        elif mom < 0 and score < 50:
            status_class = "status-cold"
            status_text = f"❄️ DISTRIBUTING"
            glow_class = "glow-cold"
            bar_color = "linear-gradient(90deg, #ef4444, #f87171)"
        else:
            status_class = "status-warm"
            status_text = f"⚡ {status}"
            glow_class = "glow-warm"
            bar_color = "linear-gradient(90deg, #3b82f6, #60a5fa)"
            
        mom_sign = "+" if mom > 0 else ""
        mom_color = "#34d399" if mom > 0 else "#f87171"
        
        scores_arr = row.get('Scores_Array', [])
        sparkline_html = generate_sparkline_svg(scores_arr, color=mom_color)
        
        constits = sector_constituents.get(industry, [])
        constits_html = ""
        for t, crs in constits:
            ccolor = "#34d399" if crs >= 80 else "#fcd34d" if crs >= 50 else "#f87171"
            tv_url = f"https://www.tradingview.com/chart/?symbol={t}"
            constits_html += f'<a href="{tv_url}" target="_blank" class="constit-badge" style="border-color: {ccolor}33; color: {ccolor}; text-decoration: none;">{t} <b>{int(crs)}</b></a>'
        
        card_content = f"""
<div class="heat-card {glow_class}">
<div class="heat-header">
<span class="heat-rank">#{int(rank)}</span>
<span class="heat-status {status_class}">{status_text}</span>
</div>
<div class="heat-title">{icon} {industry}</div>
{sparkline_html}
<div class="heat-metrics-container">
<div class="metric-box">
<span class="metric-label">RS Rating</span>
<span class="metric-val" style="color: #ffffff;">{int(score)}</span>
</div>
<div class="metric-box">
<span class="metric-label">1M Velocity</span>
<span class="metric-val" style="color: {mom_color};">{mom_sign}{mom:.1f}</span>
</div>
</div>
<div class="heat-apex">
<span class="apex-label">Apex Predators</span>
<span class="apex-tickers">{apex if apex else 'None'}</span>
</div>
<div class="breadth-section">
<div class="breadth-header">
<span>Stage 2 Breadth</span>
<span style="color: #ffffff;">{breadth:.1f}%</span>
</div>
<div class="breadth-bar-bg">
<div class="breadth-bar-fill" style="width: {breadth}%; background: {bar_color};"></div>
</div>
<details class="heat-drilldown">
<summary>🔍 View Constituents ({len(constits)})</summary>
<div class="drilldown-content">{constits_html}</div>
</details>
</div>
</div>
"""
        html_payload += card_content.replace('\\n', '')
        
    html_payload += "</div></div>"
    st.markdown(html_payload, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
