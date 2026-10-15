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
import requests
import io
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import concurrent.futures

from config import BENCHMARK_TICKER, DATA_SETTINGS, RS_WEIGHTS
from macro_regime_engine import (
    calculate_distribution_days, detect_follow_through_day, 
    get_market_regime_label, aggregate_sector_strength, standardize_columns,
    calculate_leadership_metrics, calculate_breadth_thrust,
    fetch_major_indices_daily_technicals, check_haber_21sma_rule,
    evaluate_qualitative_leadership_health, get_progressive_exposure_plan,
    fetch_nifty500_telemetry, fetch_nifty50_telemetry
)
from breadth_engine import get_latest_volume_and_ad_distribution
from relative_strength import calculate_stock_returns, calculate_relative_returns, calculate_weighted_rs_raw, normalize_rs_against_nifty500
from trend_analyzer import determine_trend_state
from technical_indicators import add_technical_indicators

# Page Config
# st.set_page_config(page_title="🏛️ Market Regime & Sector RS", layout="wide")

# CSS Styling (Shared with main app)


@st.cache_data(ttl=3600*24)
def fetch_universe_with_industry(cache_buster=2):
    """Fetch 750 universe with industry mapping."""
    valid_tickers = []
    industry_map = {}
    try:
        from market_data import fetch_nifty_total_market_tickers
        valid_tickers, industry_map = fetch_nifty_total_market_tickers(show_progress=False, return_industry_map=True)
    except Exception as e:
        st.error(f"Failed to fetch universe: {e}")
        
    if not valid_tickers and os.path.exists('tickers.txt'):
        with open('tickers.txt', 'r') as f:
            valid_tickers = [line.strip().upper() for line in f if line.strip()]
            valid_tickers = [t + '.NS' if not t.endswith('.NS') else t for t in valid_tickers]
            industry_map = {t: "Unknown" for t in valid_tickers}
            
    return pd.DataFrame({'Symbol': [t.replace('.NS', '') for t in valid_tickers], 'Industry': [industry_map.get(t, "Unknown") for t in valid_tickers]})

def fetch_yfinance_batch(tickers, days=500):
    """Download historical data for a list of tickers. 500 days ensures enough history for backward-looking metrics."""
    yf_tickers = [f"{t}.NS" for t in tickers]
    data = yf.download(yf_tickers, period=f"{days}d", group_by='ticker', progress=False)
    return data

def process_macro_data(history_df, universe_df, index_n50, index_n500):
    """Computes all macro and sector metrics with full vectorization."""
    # 1. Index Regime Analysis
    n50_dd_count, _, n50_dd_details = calculate_distribution_days(index_n50, return_details=True)
    n500_dd_count, _, n500_dd_details = calculate_distribution_days(index_n500, return_details=True)
    ftd_detected, ftd_date, ftd_details = detect_follow_through_day(index_n500, return_details=True)
    haber_21_n500 = check_haber_21sma_rule(index_n500)
    haber_21_n50 = check_haber_21sma_rule(index_n50)
    
    index_n500_cols = standardize_columns(index_n500)
    if index_n500_cols.empty or 'close' not in index_n500_cols.columns:
        curr_n500, n500_sma50, bench_r = 0, 1, {'R1':0.0, 'R3':0.0, 'R6':0.0}
    else:
        n500_cl = index_n500_cols['close']
        n500_sma50 = n500_cl.rolling(50).mean().iloc[-1]
        curr_n500 = n500_cl.iloc[-1]
        bench_r = calculate_stock_returns(index_n500_cols)
        # Fallback to 0.0 if not enough data for certain periods
        for key in ['R1', 'R3', 'R6']:
            if bench_r.get(key) is None: 
                bench_r[key] = 0.0

    # 2. Vectorized Stock Analysis
    # Get all tickers (with .NS)
    all_tickers = [f"{t}.NS" for t in universe_df['Symbol'].tolist()]
    existing_tickers = [t for t in all_tickers if t in history_df.columns.levels[0]]
    
    # Extract 'Close' prices for all existing tickers
    close_prices = history_df.xs('Close', level=1, axis=1)[existing_tickers].fillna(method='ffill')
    
    # Calculate Returns (1M=21d, 3M=63d, 6M=126d)
    r1 = (close_prices.iloc[-1] / close_prices.iloc[-(DATA_SETTINGS['TRADING_DAYS_1M']+1)]) - 1
    r3 = (close_prices.iloc[-1] / close_prices.iloc[-(DATA_SETTINGS['TRADING_DAYS_3M']+1)]) - 1
    r6 = (close_prices.iloc[-1] / close_prices.iloc[-(DATA_SETTINGS['TRADING_DAYS_6M']+1)]) - 1
    
    # Historical Returns (T-21)
    r1_t21 = (close_prices.iloc[-22] / close_prices.iloc[-(DATA_SETTINGS['TRADING_DAYS_1M']+22)]) - 1
    r3_t21 = (close_prices.iloc[-22] / close_prices.iloc[-(DATA_SETTINGS['TRADING_DAYS_3M']+22)]) - 1
    r6_t21 = (close_prices.iloc[-22] / close_prices.iloc[-(DATA_SETTINGS['TRADING_DAYS_6M']+22)]) - 1
    
    # Relative Returns vs Benchmark
    rr1, rr3, rr6 = r1 - bench_r['R1'], r3 - bench_r['R3'], r6 - bench_r['R6']
    # Approximate bench_r_t21 as r21d lag (close enough for ranking)
    rr1_t21, rr3_t21, rr6_t21 = r1_t21 - bench_r['R1'], r3_t21 - bench_r['R3'], r6_t21 - bench_r['R6']

    # RS Raw & Normalized Score (0-100 percentile rank within the 750 universe)
    rs_raw = (rr1 * RS_WEIGHTS['1M']) + (rr3 * RS_WEIGHTS['3M']) + (rr6 * RS_WEIGHTS['6M'])
    rs_score = rs_raw.rank(pct=True) * 100
    
    rs_raw_t21 = (rr1_t21 * RS_WEIGHTS['1M']) + (rr3_t21 * RS_WEIGHTS['3M']) + (rr6_t21 * RS_WEIGHTS['6M'])
    rs_score_t21 = rs_raw_t21.rank(pct=True) * 100

    # Trend Calculations (Vectorized)
    ema8 = close_prices.ewm(span=8, adjust=False).mean().iloc[-1]
    ema21 = close_prices.ewm(span=21, adjust=False).mean().iloc[-1]
    sma50 = close_prices.rolling(window=50).mean().iloc[-1]
    last_close = close_prices.iloc[-1]
    prev_close = close_prices.iloc[-2]
    
    # 52W High Proximity (Vectorized)
    high_52w = close_prices.tail(252).max()
    is_near_high = (last_close >= (high_52w * 0.9)).astype(int)

    # Assemble Results
    processed_results = []
    rs_dict = {}
    
    ticker_industry_map = universe_df.set_index('Symbol')['Industry'].to_dict()
    
    for t in existing_tickers:
        clean_t = t.replace('.NS', '')
        score = rs_score[t]
        rs_dict[clean_t] = score
        
        # Determine Trend State (Vectorized logic check)
        lc, e8, e21, s50 = last_close[t], ema8[t], ema21[t], sma50[t]
        if lc > e8 and e8 > e21 and e21 > s50: trend = '🟢 Strong'
        elif lc > s50: trend = '🟡 Pullback'
        elif lc > (s50 * 0.95): trend = '🟠 Warning'
        else: trend = '🔴 Broken'
        
        processed_results.append({
            'Ticker': clean_t,
            'Industry': ticker_industry_map.get(clean_t, 'Other'),
            'Today %': ((lc - prev_close[t]) / prev_close[t]) * 100,
            'RS Score': score,
            'RS_Score_T21': rs_score_t21[t],
            'Trend State': trend,
            'Near_High': is_near_high[t]
        })
        
    results_df = pd.DataFrame(processed_results)
    sector_df = aggregate_sector_strength(results_df)

    leadership_metrics = calculate_leadership_metrics(history_df, existing_tickers, rs_dict)
    breadth_metrics = calculate_breadth_thrust(history_df, existing_tickers)
    
    leadership_qual = evaluate_qualitative_leadership_health(results_df)
    
    dd_cluster = n500_dd_details.get('cluster_detected', False)
    status_label, status_color, status_emoji = get_market_regime_label(
        n500_dd_count, 
        curr_n500, 
        n500_sma50, 
        dd_cluster=dd_cluster
    )
    
    exposure_plan = get_progressive_exposure_plan(
        status_label, 
        n500_dd_count, 
        dd_cluster=dd_cluster, 
        ftd_detected=ftd_detected, 
        leadership_grade=leadership_qual.get('leadership_grade', 'B')
    )
    
    return {
        'n50_dd': n50_dd_count,
        'n50_dd_details': n50_dd_details,
        'n500_dd': n500_dd_count,
        'n500_dd_details': n500_dd_details,
        'n500_curr': curr_n500,
        'n500_sma50': n500_sma50,
        'ftd': ftd_date,
        'ftd_detected': ftd_detected,
        'ftd_details': ftd_details,
        'sector_df': sector_df,
        'leadership': leadership_metrics,
        'breadth': breadth_metrics,
        'haber_21_n500': haber_21_n500,
        'haber_21_n50': haber_21_n50,
        'leadership_qual': leadership_qual,
        'exposure_plan': exposure_plan,
        'status_label': status_label,
        'status_color': status_color,
        'status_emoji': status_emoji,
        'results_df': results_df
    }

def main():
    from components import render_header
    render_header("🏛️ Market Regime & Sector RS", "Automated macro intelligence based on William O'Neil & Mark Minervini principles.")
    
    if st.button("🔄 Refresh Macro Analysis", type="primary"):
        st.session_state['macro_refresh'] = True

    # Auto-run if no results exist yet
    if 'macro_results' not in st.session_state:
        st.session_state['macro_refresh'] = True

    if st.session_state.get('macro_refresh', False):
        st.session_state['macro_refresh'] = False
        
        with st.spinner("Analyzing Market Indices & 750 Stocks..."):
            universe_df = fetch_universe_with_industry()
            tickers = universe_df['Symbol'].tolist()
            
            # Index data expects at least 250d to calculate 6M return + T-21 offsets safely
            index_n50 = yf.download('^NSEI', period="250d", progress=False)
            index_n500 = yf.download('^CRSLDX', period="250d", progress=False)
            
            # Stock data
            history_df = fetch_yfinance_batch(tickers, days=500)
            
            # Process
            macro_results = process_macro_data(history_df, universe_df, index_n50, index_n500)
            st.session_state['macro_results'] = macro_results
            st.session_state['macro_last_updated'] = datetime.now().strftime('%H:%M:%S')

    macro_results = st.session_state.get('macro_results')
    
    # Defensive Check: If data is from an older version (missing new Power Metrics), force a refresh
    if macro_results and 'sector_df' in macro_results:
        s_df = macro_results['sector_df']
        required_cols = ['Leadership Density', 'RS Momentum 21d', 'Near High Breadth']
        if not all(col in s_df.columns for col in required_cols) or 'exposure_plan' not in macro_results or 'status_label' not in macro_results:
            st.warning("⚠️ Data version mismatch detected. Upgrading sector metrics...")
            st.session_state['macro_refresh'] = True
            st.rerun()

    if not macro_results:
        st.info("Click Refresh to analyze the current market regime.")
        return

    # --- RENDER UI ---
    
    # 1. TOP HEADER: ROSS HABER PROGRESSIVE EXPOSURE & REGIME MASTER HUD
    # Single source of truth: authoritative telemetry from fetch_nifty500_telemetry & fetch_nifty50_telemetry
    n500_telem = fetch_nifty500_telemetry()
    n50_telem = fetch_nifty50_telemetry()
    
    if n500_telem:
        status_label = n500_telem.get('status_label', "Market in Correction")
        status_color = n500_telem.get('status_color', "red")
        status_emoji = n500_telem.get('status_emoji', "🔴")
        exposure_plan = n500_telem.get('exposure_plan') or (macro_results.get('exposure_plan', {}) if macro_results else {})
        haber_21 = n500_telem.get('haber_21') or (macro_results.get('haber_21_n500', {}) if macro_results else {})
        n500_details = n500_telem.get('dd_details') or (macro_results.get('n500_dd_details', {}) if macro_results else {})
        n500_dd = n500_telem.get('dd_count', macro_results.get('n500_dd', 0) if macro_results else 0)
    else:
        status_label = macro_results.get('status_label', "Market in Correction") if macro_results else "Market in Correction"
        status_color = macro_results.get('status_color', "red") if macro_results else "red"
        status_emoji = macro_results.get('status_emoji', "🔴") if macro_results else "🔴"
        exposure_plan = macro_results.get('exposure_plan', {}) if macro_results else {}
        haber_21 = macro_results.get('haber_21_n500', {}) if macro_results else {}
        n500_details = macro_results.get('n500_dd_details', {}) if macro_results else {}
        n500_dd = macro_results.get('n500_dd', 0) if macro_results else 0

    if n50_telem:
        n50_details = n50_telem.get('dd_details') or (macro_results.get('n50_dd_details', {}) if macro_results else {})
        n50_dd = n50_telem.get('dd_count', macro_results.get('n50_dd', 0) if macro_results else 0)
    else:
        n50_details = macro_results.get('n50_dd_details', {}) if macro_results else {}
        n50_dd = macro_results.get('n50_dd', 0) if macro_results else 0

    leadership_qual = macro_results.get('leadership_qual', {}) if macro_results else {}
    
    deck_state = exposure_plan.get('deck_state', "🧊 FROZEN DECK (Market in Correction)")
    deck_color = exposure_plan.get('accent_color', "#ef4444")
    exposure_range = exposure_plan.get('exposure_range', "0% – 10% (Maximum Cash)")
    pilot_size = exposure_plan.get('pilot_size', "0% (No New Buys)")
    deck_rule = exposure_plan.get('rule', "Do not fight the market. Swimming upstream. Enforce 21 SMA double-close exits and protect capital.")
    
    # Haber 21 SMA Buffer Calculation & Formatting
    buf_pct = haber_21.get('pct_diff_sma21', 0.0)
    consecutive_21 = haber_21.get('consecutive_closes', 0)
    buf_color = '#10b981' if buf_pct >= 0 else '#ef4444'
    if haber_21.get('triggered', False) and consecutive_21 >= 2:
        haber_badge_text = f"🚨 {consecutive_21} Closes Below 21 SMA"
    elif buf_pct >= 0:
        haber_badge_text = "🟢 Above 21 SMA"
    else:
        haber_badge_text = "⚠️ Testing 21 SMA"

    # Render Master Progressive Exposure Banner
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(2, 6, 23, 0.98) 100%); border: 1px solid rgba(255, 255, 255, 0.12); border-left: 5px solid {deck_color}; border-radius: 14px; padding: 20px 24px; margin-bottom: 18px; box-shadow: 0 14px 32px rgba(0,0,0,0.5);">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px; margin-bottom: 12px;">
            <div>
                <span style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; font-weight: 700;">Haber / O'Neil Progressive Exposure Engine</span>
                <div style="font-size: 1.45rem; font-weight: 800; color: #f8fafc; margin-top: 3px; display: flex; align-items: center; gap: 10px;">
                    {status_emoji} {status_label} <span style="font-size: 0.85rem; padding: 3px 10px; border-radius: 6px; background: rgba(255,255,255,0.08); color: {deck_color}; border: 1px solid {deck_color}44;">{deck_state}</span>
                </div>
            </div>
            <div style="display: flex; gap: 18px; align-items: center; flex-wrap: wrap;">
                <div style="text-align: right;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.08em; font-weight: 600;">Max Recommended Exposure</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 800; color: {deck_color};">{exposure_range}</div>
                </div>
                <div style="height: 32px; width: 1px; background: rgba(255,255,255,0.12);"></div>
                <div style="text-align: right;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.08em; font-weight: 600;">Pilot Position Sizing</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 800; color: #38bdf8;">{pilot_size}</div>
                </div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; padding-top: 12px; border-top: 1px solid rgba(255, 255, 255, 0.08); font-size: 0.84rem; color: #cbd5e1;">
            <div>
                <strong>Execution Mandate:</strong> {deck_rule}
            </div>
            <div>
                <strong>Index 21 SMA Buffer:</strong> <span style="font-family: 'JetBrains Mono', monospace; color: {buf_color}; font-weight: 700;">{'+' if buf_pct >= 0 else ''}{buf_pct:.1f}%</span> ({haber_badge_text})
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for Acute Distribution Cluster
    if n500_details.get('cluster_detected', False):
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.4); border-left: 5px solid #ef4444; border-radius: 10px; padding: 14px 18px; margin-bottom: 18px; color: #fca5a5;">
            <div style="font-weight: 800; font-size: 0.95rem; display: flex; align-items: center; gap: 8px;">
                🚨 ACUTE DISTRIBUTION CLUSTER DETECTED ({n500_details.get('cluster_count_7d', 3)} Distribution Days in Last 7 Sessions)
            </div>
            <div style="font-size: 0.82rem; margin-top: 4px; color: #e2e8f0;">
                Ross Haber Warning Rule: Three or more distribution days clustered within a 5 to 7 day span signals institutional topping pressure or high-volume liquidation. Do not initiate new pilot buys. Tighten trailing stops to breakeven or 21 SMA immediately.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    from components import render_metric_card
    vol_breadth = get_latest_volume_and_ad_distribution('IN')
    ud_ratio = vol_breadth.get('up_down_volume_ratio', 1.0) if vol_breadth else 1.0
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    ftd = macro_results.get('ftd') if macro_results else None
    dcr_val = n500_details.get('latest_dcr', 0.5)
    dcr_verdict = n500_details.get('latest_midpoint_verdict', "Neutral")
    
    with m_col1:
        render_metric_card("Nifty 50 DD Count", f"{n50_dd}/25", color_class="red-text" if n50_dd >= 4 else "green-text")
        st.caption(f"{n50_details.get('standard_dd_count', 0)} Down-Vol | {n50_details.get('churning_dd_count', 0)} Churning")
    with m_col2:
        render_metric_card("Nifty 500 DD Count", f"{n500_dd}/25", color_class="red-text" if n500_dd >= 4 else "green-text")
        st.caption(f"{n500_details.get('cluster_count_7d', 0)} in last 7d | {n500_details.get('hammer_acc_count', 0)} Hammers")
    with m_col3:
        dcr_color = "green-text" if dcr_val >= 0.50 else "red-text"
        render_metric_card("Daily Closing Range (DCR)", f"{dcr_val:.2f}", color_class=dcr_color)
        st.caption(f"{'🟢 Upper Half (Accumulative)' if dcr_val >= 0.50 else '🔴 Lower Half (Distributive)'}")
    with m_col4:
        ftd_display = f"Confirmed ({ftd})" if ftd else "Watching"
        l_grade = leadership_qual.get('leadership_grade', 'B')
        render_metric_card("FTD & Leadership", ftd_display, color_class="green-text" if ftd else "yellow-text")
        st.caption(f"Grade {l_grade} Leaders | {leadership_qual.get('near_high_pct', 0)}% Near 52W High")

    # Dr. Martin Zweig Breadth Thrust & 9:1 Tape Telemetry
    try:
        from breadth_engine import get_zweig_tape_summary
        zweig_tape = get_zweig_tape_summary(market='IN')
    except Exception:
        zweig_tape = {}

    zbt_val = zweig_tape.get('zbt_10_ema', 0.50)
    zbt_active = zweig_tape.get('zbt_active', False)
    double_9to1 = zweig_tape.get('double_9_to_1_active', False)
    count_9to1 = zweig_tape.get('count_9_to_1_65d', 0)
    days_since_zbt = zweig_tape.get('days_since_zbt')
    
    zbt_status_text = "🚀 ACTIVE THRUST" if zbt_active else ("IGNITION ZONE (<0.40)" if zbt_val < 0.40 else "Neutral")
    zbt_color = "#10b981" if zbt_active else ("#38bdf8" if zbt_val < 0.40 else "#94a3b8")
    d9to1_text = "🔥 BLAST-OFF ACTIVE" if double_9to1 else f"{count_9to1} Days (Needs 2)"
    d9to1_color = "#ec4899" if double_9to1 else ("#38bdf8" if count_9to1 >= 1 else "#94a3b8")

    st.markdown(f"""
    <div style="background: linear-gradient(145deg, rgba(15, 23, 42, 0.85) 0%, rgba(2, 6, 23, 0.95) 100%); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 14px 18px; margin: 12px 0 20px 0;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.15rem;">🏛️</span>
                <span style="color: #38bdf8; font-weight: 700; font-size: 0.95rem; letter-spacing: 0.3px;">DR. MARTIN ZWEIG QUANTITATIVE BREADTH & TAPE BAROMETER</span>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.06); padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 700; color: {zbt_color}; border: 1px solid {zbt_color}44;">
                    ZBT 10-EMA: {zbt_val:.3f} &nbsp;|&nbsp; {zbt_status_text}
                </span>
                <span style="background: rgba(255,255,255,0.06); padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 700; color: {d9to1_color}; border: 1px solid {d9to1_color}44;">
                    9:1 Up-Volume: {d9to1_text}
                </span>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-top: 12px; font-size: 0.82rem; color: #94a3b8;">
            <div style="background: rgba(255,255,255,0.02); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.04);">
                <strong style="color: #cbd5e1;">Zweig Breadth Thrust (ZBT):</strong> Surges from &lt;0.40 to &ge;0.615 in &le;10 days. 
                <span style="color: {zbt_color}; font-weight: 600;">{'Active within last ' + str(days_since_zbt) + ' days' if days_since_zbt is not None else 'No active thrust'}.</span>
            </div>
            <div style="background: rgba(255,255,255,0.02); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.04);">
                <strong style="color: #cbd5e1;">Double 9:1 Blast-Off:</strong> Two 9:1 up-volume days in 65 sessions unlocks 100% Super-Thrust allocation.
                <span style="color: {d9to1_color}; font-weight: 600;">{count_9to1} occurrences active.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("📖 William O'Neil & Ross Haber Institutional Playbook (Market Timing & Risk Rules)", expanded=False):
        st.markdown(r"""
        #### 🎓 Core Rules from Ross Haber (Former William O'Neil PM) & Richard Moglen
        
        1. **The 'M' in CANSLIM (Market Trend is the Master Filter)**:
           - **3 out of 4 stocks follow the general market trend.** Trying to buy growth setups in a downtrend is *"swimming upstream"* or playing with a *"cold deck"*.
           - Russian Nesting Dolls: The General Market is the outer doll $\rightarrow$ Industry Group is the middle doll $\rightarrow$ Individual Leader is the inner doll.
        
        2. **Where It Closes Matters Far More Than Up or Down (The Midpoint Rule)**:
           - $$\text{DCR} = \frac{\text{Close} - \text{Low}}{\text{High} - \text{Low}}$$
           - **Upper Half ($\text{DCR} \ge 0.50$, Accumulative)**:
             - *Strong Close*: Closed up near highs on volume.
             - *Hammer Bar (Absorption)*: Closed DOWN on the day, but finished in the upper half on volume. Institutional buyers stepped in and absorbed selling. **Excluded from Distribution Days!**
           - **Lower Half ($\text{DCR} < 0.50$, Distributive)**:
             - *Weak Close*: Closed down near lows.
             - *Churning / Stalling Day*: Closed UP ($0\% \le \Delta \text{Close} \le +0.4\%$) or flat on volume $>$ prior session, but in the lower half ($\text{DCR} < 0.50$). Huge volume with no upside progress = smart money selling into retail buying. **Counts as a Distribution Day!**
             
        3. **Follow-Through Day (FTD) — The Two Halves**:
           - **Quantitative Half**: Day 4+ surge of $\ge +1.5\%$ to $+1.75\%$ on volume above prior day off a rally attempt low.
           - **Qualitative Half ("Watch the Leaders")**: The deciding factor! If leaders are coiling within $10\%\text{–}15\%$ of 52-week highs and breaking out with group confirmation ($\ge 3\text{–}5$ stocks per group), the FTD succeeds. If no leaders emerge, it's a false rally.
           
        4. **Distribution Day Clusters (Impending Top Warning)**:
           - When **$\ge 3$ distribution days occur within 5 to 7 trading days**, it forms an acute **Cluster**.
           - Accompanied by wide-and-loose spreads and volume skyscrapers, indicating an impending market correction.
           
        5. **Ross Haber 21-Day Moving Average Double-Close Rule (Stock vs Index)**:
           - **Individual Stock Level**: If a stock posts **2 consecutive closes below its 21-day SMA**, trim $1/3$ to $1/2$ position (if Stage 1/2 base) or liquidate $100\%$ (if Stage 3/4 base).
           - **Index Level**: If the benchmark closes 2x below its 21-day SMA, downshift portfolio exposure (Hot Deck $\rightarrow$ Cold Deck) and halt aggressive new buys.
        """)

    st.markdown("")

    # --- MAJOR BENCHMARK TECHNICALS & A/D COCKPIT (IBD GMI STANDARD) ---
    st.markdown("### 🏛️ Major Benchmark Technicals & A/D Cockpit (IBD GMI Standard)")
    st.caption("Multi-index assessment of price action, distance from 52-week highs, YTD return, 50/200 SMA buffers, and trailing 13-week Accumulation/Distribution grades.")

    with st.expander("ℹ️ What do A, B, C, D, E mean in A/D and how is it calculated?", expanded=False):
        st.markdown(r"""
        #### 🏷️ What Do the A/D Grades Mean?
        Investor's Business Daily (IBD) assigns an **Accumulation/Distribution (A/D) Grade** to major indices to measure whether institutional money ("Smart Money") is net buying or net selling over a trailing 13-week (65 trading day) cycle:

        - 🟢 **Grade A (Heavy Institutional Accumulation)**: Large mutual funds, pension funds, and sovereign funds are aggressively buying on above-average volume. Index pullbacks are shallow and quickly bought up.
        - 🟢 **Grade B (Moderate Institutional Accumulation)**: Institutional buying pressure consistently outpaces selling. Positive institutional sponsorship supporting the trend.
        - 🟡 **Grade C+ (Mild Accumulation)**: Leaning slightly bullish with modest net institutional inflows.
        - ⚪ **Grade C (Neutral / Equilibrium)**: Buying and selling volume are balanced. Indecisive market with no clear institutional accumulation or distribution bias.
        - 🟠 **Grade D (Moderate Institutional Distribution)**: Institutions are net sellers, liquidating shares into bounces on elevated volume. Rallies tend to fail quickly.
        - 🔴 **Grade E (Heavy Institutional Distribution)**: Severe institutional distribution and capital exit. High risk of extended market corrections and failed breakouts.

        ---

        #### 🧮 How Is the A/D Grade Calculated?
        The algorithm evaluates the trailing **65 trading sessions (13 weeks)** using price action and volume dynamics:

        1. **Intraday Close Location Value (CLV)**:
           $$\text{CLV} = \frac{(\text{Close} - \text{Low}) - (\text{High} - \text{Close})}{\text{High} - \text{Low}} = \frac{2 \times \text{Close} - (\text{High} + \text{Low})}{\text{High} - \text{Low}}$$
           Measures *where* the index closed within its daily high-low range: $+1.0$ if closed at the exact high of the day (aggressive buying), $-1.0$ if closed at the low (aggressive selling).

        2. **Volume Ratio Scaling**:
           $$\text{Volume Ratio} = \frac{\text{Daily Volume}}{\text{50-Day Moving Average Volume}}$$
           High-volume sessions carry greater weighting; quiet low-volume days are dampened.

        3. **Daily Institutional Flow**:
           $$\text{Daily Flow} = \Big(0.6 \times \text{sign}(\Delta \text{Close}) + 0.4 \times \text{CLV}\Big) \times \text{Volume Ratio}$$

        4. **Time-Decay Front-Weighting**:
           Recent trading sessions are assigned up to **$2\times$ the weight** of older sessions from 13 weeks ago using a linear ramp, ensuring the score rapidly reflects current institutional accumulation or distribution shifts.

        5. **Canonical IBD Grade Thresholds**:
           - **A**: Score $\ge +12.0$
           - **B**: $+4.0 \le \text{Score} < +12.0$
           - **C+**: $0.0 \le \text{Score} < +4.0$
           - **C**: $-4.0 \le \text{Score} < 0.0$
           - **D**: $-12.0 \le \text{Score} < -4.0$
           - **E**: Score $< -12.0$
        """)
    
    try:
        idx_technicals = fetch_major_indices_daily_technicals('IN')
        if idx_technicals:
            c_cols = st.columns(len(idx_technicals))
            for col, idx in zip(c_cols, idx_technicals):
                with col:
                    ad_g = idx.get('ad_grade', 'C')
                    grade_bg = "rgba(16,185,129,0.18)" if ad_g in ['A', 'B'] else ("rgba(245,158,11,0.18)" if ad_g == 'C+' else ("rgba(148,163,184,0.18)" if ad_g == 'C' else "rgba(239,68,68,0.18)"))
                    grade_color = "#10b981" if ad_g in ['A', 'B'] else ("#f59e0b" if ad_g == 'C+' else ("#94a3b8" if ad_g == 'C' else "#ef4444"))
                    chg_color = "#10b981" if idx.get('chg_pct', 0) >= 0 else "#ef4444"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(145deg, #0f172a 0%, #020617 100%); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.4);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-weight: 700; color: #f8fafc; font-size: 1.05rem;">{idx['name']}</span>
                            <span style="background: {grade_bg}; color: {grade_color}; border: 1px solid {grade_color}; padding: 2px 8px; border-radius: 6px; font-weight: 800; font-size: 0.8rem;">
                                A/D: {ad_g}
                            </span>
                        </div>
                        <div style="display: flex; align-items: baseline; gap: 8px; margin-bottom: 12px;">
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 800; color: #f8fafc;">{idx['close']:,.2f}</span>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 700; color: {chg_color};">
                                {idx['chg_pts']:+,.1f} ({idx['chg_pct']:+.2f}%)
                            </span>
                        </div>
                        <div style="grid-template-columns: 1fr 1fr; display: grid; gap: 8px; font-size: 0.78rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 10px;">
                            <div>
                                <span style="color: #64748b;">Off 52W High:</span><br>
                                <strong style="color: {'#10b981' if idx['dist_52w_high'] >= -3.0 else '#f59e0b' if idx['dist_52w_high'] >= -8.0 else '#ef4444'}; font-family: 'JetBrains Mono';">
                                    {idx['dist_52w_high']:.1f}%
                                </strong>
                            </div>
                            <div>
                                <span style="color: #64748b;">Since Jan 1 (YTD):</span><br>
                                <strong style="color: {'#10b981' if idx['ytd_pct'] >= 0 else '#ef4444'}; font-family: 'JetBrains Mono';">
                                    {idx['ytd_pct']:+.1f}%
                                </strong>
                            </div>
                            <div>
                                <span style="color: #64748b;">vs 50 SMA:</span><br>
                                <strong style="color: {'#10b981' if idx['dist_sma50'] >= 0 else '#ef4444'}; font-family: 'JetBrains Mono';">
                                    {idx['dist_sma50']:+.1f}%
                                </strong>
                            </div>
                            <div>
                                <span style="color: #64748b;">vs 200 SMA:</span><br>
                                <strong style="color: {'#10b981' if idx['dist_sma200'] >= 0 else '#ef4444'}; font-family: 'JetBrains Mono';">
                                    {idx['dist_sma200']:+.1f}%
                                </strong>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load major benchmark cockpit: {e}")

    st.markdown("")
    
    # -------------------------------------------------------------------------
    # 2. MARKET LEADERSHIP & BREADTH INTELLIGENCE PANEL
    # -------------------------------------------------------------------------
    st.header("⚡ Market Leadership & Breadth Intelligence")
    st.caption("Professional indicators tracking the health of market participation and leadership emergence.")
    
    leadership = macro_results.get('leadership')
    breadth = macro_results.get('breadth')
    
    if leadership and breadth:
        ler = leadership['ler_current']
        lac = leadership['lac_current']
        lt = leadership['lt_current']
        bt_ma = breadth['breadth_ma10_current']
        bt_trigger = breadth['thrust_triggered']
        
        # --- Interpretation Logic ---
        
        # LER Interpretation
        if ler < 0.5: ler_status = "Dead Market"
        elif ler < 1.5: ler_status = "Early Forming"
        elif ler < 3.0: ler_status = "Strong"
        else: ler_status = "Explosive Phase"
            
        # LAC Interpretation
        if lac < 0: lac_status = "Shrinking"
        elif lac < 2: lac_status = "Stagnant/Slow"
        elif lac < 6: lac_status = "Early Cycle"
        else: lac_status = "Powerful Expansion"
            
        # LT Interpretation
        if lt < 1.0: lt_status = "Weak"
        elif lt < 3.0: lt_status = "Normal"
        elif lt < 5.0: lt_status = "Expanding"
        else: lt_status = "Strong Bull"
            
        # Market Regime Engine
        regime_title = "Neutral/Mixed Environment"
        regime_desc = "Market is lacking clear unified direction. Trade carefully."
        
        if ler > 3.0 and lac > 0 and bt_trigger:
            regime_title = "🔥 Strong Momentum Phase"
            regime_desc = "High probability environment for breakouts. Aggressive exposure warranted."
        elif ler > 1.5 and lac > 0 and lt > 3.0:
            regime_title = "🌱 Early Leadership Expansion"
            regime_desc = "Environment strongly supportive for trend trading. Leaders are emerging."
        elif ler < 0.5 and lac < 0 and lt < 1.0:
            regime_title = "❄️ Weak Market Conditions"
            regime_desc = "Avoid aggressive exposure. Leadership is deteriorating rapidly."
            
        # Save regime state
        from database import save_market_regime
        save_market_regime(regime_title)
            
        # --- UI Rendering ---
        
        # Regime Interpretation Card
        st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid #38bdf8; border-radius: 10px; padding: 1.5rem; margin-bottom: 2rem;">
            <h3 style="margin-top:0; color: #38bdf8;">Overall Signal: {regime_title}</h3>
            <p style="margin-bottom:0; font-size: 1.1rem;">{regime_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Leader Emergence Rate (LER)", 
                value=f"{ler:.1f}%", 
                delta=ler_status, 
                delta_color="normal" if ler >= 1.5 else "off" if ler >= 0.5 else "inverse",
                help="Measures how many strong stocks are forming in the market. >1.5% signals strong leader environment."
            )
        with col2:
            st.metric(
                label="Leadership Acceleration (LAC)", 
                value=f"{lac:+.1f}/wk", 
                delta=lac_status,
                delta_color="normal" if lac > 2 else "off" if lac >= 0 else "inverse",
                help="Measures how quickly new leaders are appearing based on 3-week EMA of growth."
            )
        with col3:
            st.metric(
                label="Leadership Thrust (LT)", 
                value=f"{lt:.1f}%", 
                delta=lt_status,
                delta_color="normal" if lt > 3 else "off" if lt >= 1 else "inverse",
                help="Measures expansion in new highs (smoothed 5d). >3% signals expansion."
            )
        with col4:
            st.metric(
                label="Breadth Thrust (BT) MA10", 
                value=f"{bt_ma:.2f}",
                delta="🟢 Triggered!" if bt_trigger else ("Strong" if bt_ma > 0.6 else "Weak" if bt_ma < 0.4 else "Neutral"),
                delta_color="normal" if bt_trigger or bt_ma > 0.6 else "inverse" if bt_ma < 0.4 else "off",
                help="10-day MA of Advancers ratio. Thrust triggers when rising from <0.4 to >0.6 in 10 days."
            )
            
        st.write("")
        chart_layout = st.radio(
            "Leadership Charts Display Layout:",
            ["📑 Modular Stacked (Full Width)", "🔲 2x2 Multi-Grid", "🔍 Focus Tabs"],
            horizontal=True,
            key="regime_chart_layout"
        )
        
        ler_series = leadership['ler_series']
        fig_ler = px.line(ler_series, title="Leader Emergence Rate (%) - 24 Weeks", markers=True)
        fig_ler.update_layout(xaxis_title="", yaxis_title="LER %", showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
        fig_ler.add_hline(y=1.5, line_dash="dash", line_color="green", opacity=0.5)

        lac_series = leadership['lac_series']
        fig_lac = px.bar(lac_series, title="Leadership Acceleration Curve (New/Wk) - 24 Weeks")
        fig_lac.update_traces(marker_color=['#10b981' if val > 0 else '#ef4444' for val in lac_series])
        fig_lac.update_layout(xaxis_title="", yaxis_title="Acceleration", showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')

        lt_series = leadership['lt_series']
        fig_lt = px.line(lt_series, title="Leadership Thrust (100 Days)")
        fig_lt.update_layout(xaxis_title="", yaxis_title="LT %", showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
        fig_lt.add_hline(y=3.0, line_dash="dash", line_color="orange", opacity=0.5)

        bt_series = breadth['breadth_ma10_series']
        fig_bt = px.line(bt_series, title="Dr. Martin Zweig Breadth Thrust MA10 (100 Days)")
        fig_bt.update_layout(xaxis_title="", yaxis_title="Advancers Ratio", showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
        fig_bt.add_hline(y=0.615, line_dash="dash", line_color="#10b981", annotation_text="Thrust Target (0.615)")
        fig_bt.add_hline(y=0.40, line_dash="dash", line_color="#ef4444", annotation_text="Ignition Floor (0.40)")

        if chart_layout == "📑 Modular Stacked (Full Width)":
            fig_ler.update_layout(height=420)
            st.plotly_chart(fig_ler, use_container_width=True)
            fig_lac.update_layout(height=400)
            st.plotly_chart(fig_lac, use_container_width=True)
            fig_lt.update_layout(height=420)
            st.plotly_chart(fig_lt, use_container_width=True)
            fig_bt.update_layout(height=440)
            st.plotly_chart(fig_bt, use_container_width=True)
        elif chart_layout == "🔲 2x2 Multi-Grid":
            ch1, ch2 = st.columns(2)
            with ch1:
                fig_ler.update_layout(height=360)
                st.plotly_chart(fig_ler, use_container_width=True)
            with ch2:
                fig_lac.update_layout(height=360)
                st.plotly_chart(fig_lac, use_container_width=True)
            ch3, ch4 = st.columns(2)
            with ch3:
                fig_lt.update_layout(height=360)
                st.plotly_chart(fig_lt, use_container_width=True)
            with ch4:
                fig_bt.update_layout(height=360)
                st.plotly_chart(fig_bt, use_container_width=True)
        else: # Focus Tabs
            t_ler, t_lac, t_lt, t_bt = st.tabs([
                "🌱 Leader Emergence (LER)", 
                "🚀 Acceleration (LAC)", 
                "⚡ Leadership Thrust (LT)", 
                "🏛️ Zweig Breadth Thrust (BT)"
            ])
            with t_ler:
                fig_ler.update_layout(height=480)
                st.plotly_chart(fig_ler, use_container_width=True)
            with t_lac:
                fig_lac.update_layout(height=480)
                st.plotly_chart(fig_lac, use_container_width=True)
            with t_lt:
                fig_lt.update_layout(height=480)
                st.plotly_chart(fig_lt, use_container_width=True)
            with t_bt:
                fig_bt.update_layout(height=480)
                st.plotly_chart(fig_bt, use_container_width=True)
            
        # User Education Section
        with st.expander("📚 How to Use Market Leadership Signals"):
            st.markdown("""
            These four signals help detect early bull cycles, strong trend phases, and deteriorating market conditions before price indices break down.

            - **Leader Emergence Rate (LER):** Shows how many strong stocks are forming in the market (RS > 85, >50SMA/200SMA, near 52W High, +25% in 8wks). When this rises after a correction, it often signals the beginning of a new leadership cycle.
            - **Leadership Acceleration Curve (LAC):** Shows how quickly these new leaders are appearing. Sharp increases often occur during the early stages of rapid bull markets.
            - **Leadership Thrust (LT):** Measures expansion in new 52-week highs. When this expands rapidly after corrections (>3%), it signals deep institutional buying across the board.
            - **Breadth Thrust (BT):** Shows broad market participation (Advancers vs Decliners smoothed over 10 days). A thrust occurs when buying expands rapidly (MA10 swings from below 0.40 to above 0.60), historically preceding strong macro upward phases.
            """)
            
    else:
        st.info("Market Leadership metrics require at least 252 days of continuous historical data to compute properly.")

    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # 3. SECTOR LEADERBOARD
    # -------------------------------------------------------------------------
    st.header("📈 Industry Group Leadership (Total Market)")
    st.caption("Aggregated Relative Strength and Momentum metrics across 750 stocks.")
    
    sector_df = macro_results['sector_df']
    
    # Filter out small industries
    sector_df = sector_df[sector_df['Stocks'] >= 3]
    
    # --- VISUAL: SECTOR HEATMAP (GRID) ---
    heatmap_cols = ['Avg_RS', 'Leadership Density', 'Bullish %', 'Near High Breadth']
    
    # Check if we have the columns
    if all(col in sector_df.columns for col in heatmap_cols):
        # Prepare data for heatmap
        heatmap_data = sector_df.set_index('Industry')[heatmap_cols].copy()
        heatmap_data = heatmap_data.rename(columns={'Bullish %': 'Trend Health'})
        
        # Sort by Leadership Density (institutional tracking)
        heatmap_data = heatmap_data.sort_values('Leadership Density', ascending=False)
        
        # Normalize each column to 0-1 range for independent coloring
        heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
        # Handle case where max == min (prevents NaNs)
        heatmap_norm = heatmap_norm.fillna(0.5)
        
        # Create a red-yellow-green colorscale
        color_scale = [
            [0.0, "#ef4444"],    # Red
            [0.5, "#eab308"],    # Yellow
            [1.0, "#10b981"]     # Green
        ]
        
        # Draw heatmap using normalized values for color, but showing raw text
        fig_heatmap = px.imshow(
            heatmap_norm,
            labels=dict(x="Metrics", y="Industry Group", color="Relative Score"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale=color_scale,
            aspect="auto",
            title="Sector Performance Heatmap (Sorted by Leadership Density, Colored by Column Rank)"
        )
        
        # Overlay actual raw numbers as text
        fig_heatmap.update_traces(
            text=heatmap_data.values.round(1),
            texttemplate="%{text}",
            hovertemplate="Industry: %{y}<br>Metric: %{x}<br>Value: %{text}<extra></extra>"
        )
        
        # Adjust height based on number of industries
        fig_heatmap.update_layout(height=max(500, len(heatmap_data) * 35), coloraxis_showscale=False)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.warning("Please refresh data to view the new Heatmap.")
    
    st.markdown("---")
    
    # --- DETAILED INDUSTRY POWER TABLE ---
    st.subheader("📊 Industry Power Metrics")
    st.dataframe(
        sector_df,
        column_config={
            "Industry": st.column_config.TextColumn("Industry Group", width="large"),
            "Group Confirmation": st.column_config.TextColumn("Group Confirmation", width="medium"),
            "Avg_RS": st.column_config.NumberColumn("Avg RS (T-0)", format="%.1f"),
            "RS Momentum 21d": st.column_config.NumberColumn("RS Mom (21d)", format="%+.1f"),
            "Leadership Density": st.column_config.ProgressColumn("Leadership Density", min_value=0, max_value=100, format="%d%%"),
            "Bullish %": st.column_config.ProgressColumn("Trend Health (🟢)", min_value=0, max_value=100, format="%d%%"),
            "Near High Breadth": st.column_config.ProgressColumn("Near 52W High", min_value=0, max_value=100, format="%d%%"),
            "Stocks": st.column_config.NumberColumn("Stocks"),
            "Leaders_Count": st.column_config.NumberColumn("RS>85"),
            "Avg_Today_Pct": st.column_config.NumberColumn("Today %", format="%+.2f%%")
        },
        hide_index=True,
        use_container_width=True
    )

    with st.expander("ℹ️ Column Definitions & Methodology"):
        st.markdown(r"""
| Column | What It Represents | How It's Calculated |
|---|---|---|
| **Industry Group** | The NSE-defined industry classification for all ~750 Nifty Total Market stocks. | Groups with fewer than 3 stocks are filtered out. |
| **Group Confirmation** | Ross Haber Institutional Pack-Hunting Confirmation. | ✅ Confirmed when the group has $\ge 3$ leading stocks (RS $\ge 85$) coiling/breaking out together; ⚪ Unconfirmed otherwise. |
| **Avg RS (T-0)** | The current average Weighted Relative Strength score of the group. | Percentage-ranked (0-100) across all 750 stocks. |
| **RS Mom (21d)** | The change in the group's Average RS over the last month (21 trading days). | `Current Avg_RS - Historical (T-21) Avg_RS`. Positive values indicate the sector is actively gaining leadership. |
| **Leadership Density**| The percentage of stocks in the group that are "Elite Leaders" (RS > 85). | `(Stocks with RS > 85 / Total Stocks) × 100`. High density signals a true institutional theme. |
| **Trend Health (🟢)** | Percentage of stocks in a 🟢 Strong technical uptrend. | `(Stocks above 8EMA, 21EMA, and 50SMA / Total stocks) × 100`. |
| **Near 52W High** | Percentage of stocks within 10% of their 52-week high. | `(Stocks within 10% of 52W high / Total stocks) × 100`. Higher participation near highs precedes major breakouts. |
| **Stocks** | Total number of evaluated stocks belonging to this industry. | Count of valid data constituents in the universe. |
| **RS>85** | Raw count of stocks with RS score above 85. | Identify top-tier momentum names. |
| **Today %** | The average intraday percentage change for all stocks in the group today. | Arithmetic mean of the daily %. |

**Strategy: Spotting Institutional Rotation**
- **Leadership Density > 15%** indicates a primary leading sector.
- **RS Mom (21d) > +5.0** suggests a new group is appearing on the institutional radar.
- **Trend Health > 70%** indicates the move is stable and broad-based.
        """)


if __name__ == "__main__":
    main()
