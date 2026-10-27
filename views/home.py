# Kush Tracker Lite - Home Dashboard View (Synced from app.py)
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Kush Tracker - Streamlit Dashboard
# Professional Portfolio Execution & Risk Management App

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import math
import json
import re
import html

# Local imports
from portfolio_fetcher import fetch_portfolio_data
from market_data import fetch_all_holdings_data, fetch_nifty_total_market_tickers, fetch_full_india_universe_tickers, get_ticker_symbol
from technical_indicators import add_technical_indicators, get_atr_state, calculate_webster_sell_signals
from relative_strength import calculate_all_rs_scores, get_rs_rating
from signal_engine import calculate_weighted_rs_for_universe
from price_history_manager import fetch_incremental_history
from trend_analyzer import determine_trend_state, get_trend_details
from decision_engine import make_decision, ACTION_EXIT, ACTION_TRIM, ACTION_HOLD, ACTION_ADD
from database import init_database, save_snapshot, get_signal_change_days, save_sector_leadership, get_sector_leadership_history, get_journal_entry, get_all_fundamentals_cache, get_current_tml_leaders, get_hat_stocks, get_top_rs_leaders
from climax_exhaustion import detect_climax_exhaustion
from macro_regime_engine import calculate_distribution_days, get_market_regime_label, detect_change_of_character
from daily_insights_engine import generate_exposure_guide, get_sector_clusters, generate_macro_health_score
try:
    from views.market_regime import process_macro_data, fetch_universe_with_industry, fetch_yfinance_batch
except ImportError:
    from pages.market_regime import process_macro_data, fetch_universe_with_industry, fetch_yfinance_batch
from safe_harbor_engine import evaluate_safe_harbor
from portfolio_earnings_engine import update_portfolio_earnings, get_portfolio_new_results_alerts, get_pre_earnings_risk_tickers
from risk_metrics import calculate_sortino_ratio, calculate_smooth_rs_atr


# Page config
# st.set_page_config handled in Lite app.py router

from styles import load_css
load_css()

import threading
import subprocess
import sys
import os

@st.cache_resource
def start_background_backfill():
    """Starts the TML history backfill script in the background once per server start."""
    def run_backfill():
        print("[System] Starting background TML history backfill...")
        try:
            # We use python executable from sys to ensure we stay in the same environment
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            script_path = os.path.join(root_dir, "backfill_tml_history.py")
            subprocess.run([sys.executable, script_path], check=True)
            print("[System] Background TML backfill completed or verified.")
        except Exception as e:
            print(f"[System] Error in background backfill: {e}")
            
    thread = threading.Thread(target=run_backfill, daemon=True)
    thread.start()
    return True

# Trigger the background backfill on app load
start_background_backfill()

def check_daily_beach_ball_sync():
    """Automatically runs Beach Ball accumulation sync daily in the background when breadth is low (<35%)."""
    from datetime import datetime
    today_str = datetime.now().strftime('%Y-%m-%d')
    if st.session_state.get('last_beach_ball_sync_check') != today_str:
        st.session_state['last_beach_ball_sync_check'] = today_str
        def run_sync():
            try:
                from beach_ball_engine import auto_sync_beach_ball_if_low_breadth
                auto_sync_beach_ball_if_low_breadth()
            except Exception as e:
                print(f"[System] Beach Ball background sync error: {e}")

        thread = threading.Thread(target=run_sync, daemon=True)
        thread.start()

check_daily_beach_ball_sync()

def check_daily_market_persona_sync():
    """Pre-computes and caches 2020 vs 2021 Market Execution Persona daily in the background."""
    from datetime import datetime
    today_str = datetime.now().strftime('%Y-%m-%d')
    if st.session_state.get('last_market_persona_sync_check') != today_str:
        st.session_state['last_market_persona_sync_check'] = today_str
        def run_sync():
            try:
                from market_persona_engine import sync_market_persona_daily
                sync_market_persona_daily('INDIA')
            except Exception as e:
                print(f"[System] Market persona background sync error: {e}")

        thread = threading.Thread(target=run_sync, daemon=True)
        thread.start()

check_daily_market_persona_sync()
def get_action_color(action: str) -> str:
    """Get color for action badge."""
    colors = {
        ACTION_EXIT: '#ef4444',
        ACTION_TRIM: '#f59e0b',
        ACTION_HOLD: '#6b7280',
        ACTION_ADD: '#10b981',
    }
    return colors.get(action, '#6b7280')


def get_trend_emoji(trend_state: str) -> str:
    """Get emoji for trend state."""
    if '🟢' in str(trend_state) or 'Strong' in str(trend_state):
        return '🟢'
    elif '🟡' in str(trend_state) or 'Pullback' in str(trend_state):
        return '🟡'
    elif '🟠' in str(trend_state) or 'Warning' in str(trend_state):
        return '🟠'
    else:
        return '🔴'


def calculate_total_portfolio_value(portfolio_df, market_data: dict) -> float:
    """Calculate total portfolio value from current prices."""
    total = 0.0
    for _, row in portfolio_df.iterrows():
        ticker = row['ticker']
        quantity = row['Qty']
        
        if ticker in market_data and not market_data[ticker].empty:
            current_price = market_data[ticker]['close'].iloc[-1]
            if pd.isna(current_price) or math.isnan(current_price):
                val = row['Avg Buy Price'] * quantity
                if not pd.isna(val):
                    total += val
            else:
                total += current_price * quantity
        else:
            val = row['Avg Buy Price'] * quantity
            if not pd.isna(val):
                total += val
    
    return total


@st.cache_data(ttl=3600*24)
def get_industry_mapping():
    """Cache industry map separately so it doesn't refresh constantly."""
    try:
        df = fetch_universe_with_industry()
        if not df.empty:
            return dict(zip(df['Symbol'], df['Industry']))
    except:
        pass
    return {}

@st.cache_data(ttl=3600*24)
def get_yfinance_industry(ticker):
    """Fallback to yfinance to fetch industry for stocks not in the Nifty Total Market."""
    try:
        yf_ticker = f"{ticker}.NS" if not ticker.endswith('.NS') and not ticker.endswith('.BO') and not ticker.startswith('^') else ticker
        stock = yf.Ticker(yf_ticker)
        info = stock.info
        return info.get('industry') or info.get('sector') or 'Other / ETF'
    except:
        return 'Other / ETF'

# Known ETF classification map — yfinance returns no sector/industry for Indian ETFs
# Format: {TICKER: (Sector, Industry)}
KNOWN_ETF_CLASSIFICATIONS = {
    'LIQUIDBEES': ('Cash / Liquid', 'Government Securities Fund'),
    'LIQUIDCASE': ('Cash / Liquid', 'Government Securities Fund'),
    'LIQUIDETF':  ('Cash / Liquid', 'Government Securities Fund'),
    'GOLDBEES':   ('Gold / Commodity', 'Gold ETF'),
    'GOLDCASE':   ('Gold / Commodity', 'Gold ETF'),
    'SILVERBEES': ('Gold / Commodity', 'Silver ETF'),
    'MOM30IETF':  ('Equity / Diversified ETF', 'Momentum Strategy ETF'),
    'MOM50':      ('Equity / Diversified ETF', 'Momentum Strategy ETF'),
    'GOLDBEES':   ('Cash / Commodity', 'Gold ETF'),
    'GOLDCASE':   ('Cash / Commodity', 'Gold ETF'),
    'SILVERBEES': ('Cash / Commodity', 'Silver ETF'),
    'MOM30IETF':  ('Other / ETF', 'Momentum Strategy ETF'),
    'MOM50':      ('Other / ETF', 'Momentum Strategy ETF'),
    'MOMENTUM50': ('Other / ETF', 'Momentum Strategy ETF'),
    'ALPHA':      ('Other / ETF', 'Alpha Strategy ETF'),
    'NIFTYBEES':  ('Equity / Diversified ETF', 'Nifty 50 Index ETF'),
    'JUNIORBEES': ('Equity / Diversified ETF', 'Nifty Next 50 Index ETF'),
}

# Tickers that are cash-equivalent and should be excluded from equity concentration risk
CASH_EQUIVALENT_TICKERS = {'LIQUIDBEES', 'LIQUIDCASE', 'LIQUIDETF', 'GOLDBEES', 'SILVERBEES'}

@st.cache_data(ttl=3600*24)
def get_yfinance_sector_and_industry(ticker):
    """Fetch both broad sector and granular industry from yfinance for concentration analysis.
    Uses a known ETF map for Indian ETFs where yfinance returns no classification."""
    # Check known ETF map first
    upper_ticker = ticker.upper()
    if upper_ticker in KNOWN_ETF_CLASSIFICATIONS:
        return KNOWN_ETF_CLASSIFICATIONS[upper_ticker]
    
    try:
        yf_ticker = f"{ticker}.NS" if not ticker.endswith('.NS') and not ticker.endswith('.BO') and not ticker.startswith('^') else ticker
        stock = yf.Ticker(yf_ticker)
        info = stock.info
        sector = info.get('sector') or 'Other / ETF'
        industry = info.get('industry') or sector
        return sector, industry
    except:
        return 'Other / ETF', 'Other / ETF'

@st.cache_data(ttl=3600*12, show_spinner=False)  # Cache for 12 hours — RS percentiles are stable intraday
def compute_universe_rs_scores(portfolio_tickers=None):
    """
    Compute RS percentile scores for the entire NIFTY Total Market universe + portfolio tickers.
    Uses the disk-cached price history (same source as Algo Terminal / Intraday Monitor)
    so this is fast after first run.
    
    Args:
        portfolio_tickers: list of portfolio tickers to ensure they get ranked even if they are ETFs.
        
    Returns:
        dict mapping clean ticker -> RS percentile (0-100), or empty dict on failure.
    """
    if portfolio_tickers is None:
        portfolio_tickers = []
        
    try:
        # Rank across the full 3-Tier Indian Equity Universe (~2,750+ stocks + all SME/BSE portfolio holdings)
        all_target_tickers = fetch_full_india_universe_tickers(portfolio_tickers=portfolio_tickers, show_progress=False)
        if not all_target_tickers or len(all_target_tickers) < 100:
            return {}
        
        # Use the incremental disk cache — avoids re-downloading 2,750+ stocks
        history_df = fetch_incremental_history(all_target_tickers, days=252)
        if history_df.empty:
            return {}
        
        # Same RS calculation used by the Algo Terminal (now adaptive for >=25 bar SMEs/IPOs)
        rs_scores = calculate_weighted_rs_for_universe(history_df, all_target_tickers)
        return rs_scores  # {clean_ticker: percentile_0_to_100}
    except Exception as e:
        print(f"[!] Universe RS computation failed: {e}")
        return {}


def _build_rs_data_from_universe(portfolio_tickers, universe_rs, market_data):
    """
    Convert universe-level RS percentile scores into the full rs_data format
    expected by the decision engine and UI.
    """
    rs_data = {}
    
    for ticker in portfolio_tickers:
        if ticker == 'BENCHMARK':
            continue
        
        # Look up the universe RS for this ticker
        rs_score = universe_rs.get(ticker)
        
        # Fallback to yf_ticker (for BSE stocks like ZELIO.BO and SME stocks like ANONDITA-SM -> ANONDITA.NS)
        if rs_score is None:
            yf_ticker = get_ticker_symbol(ticker)
            rs_score = universe_rs.get(yf_ticker)
            
            # Additional fallback in case signal_engine stripped .NS or .BO
            if rs_score is None:
                stripped = yf_ticker.replace('.NS', '').replace('.BO', '')
                rs_score = universe_rs.get(stripped)
            if rs_score is None:
                clean_raw = str(ticker).replace('-SM', '').replace('-ST', '').replace('.NS', '').replace('.BO', '').strip()
                rs_score = universe_rs.get(clean_raw)
        
        if rs_score is not None:
            rs_data[ticker] = {
                'rs_score': rs_score,
                'rs_raw': None,  # Raw not computed in this path
                'rs_rating': get_rs_rating(rs_score),
                'rs_source': 'universe',
                'details': {},
            }
        else:
            # Ticker not in universe (could be a micro-cap, ETF, etc.)
            rs_data[ticker] = {
                'rs_score': None,
                'rs_raw': None,
                'rs_rating': 'Unknown',
                'rs_source': 'not_in_universe',
                'details': {},
            }
    
    return rs_data


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(cache_v: int = 2):
    """Load and process all data."""
    # Initialize database
    init_database()
    
    # Fetch portfolio
    portfolio_df = fetch_portfolio_data()
    if portfolio_df.empty:
        return None, None, None, None, None
    
    # Fetch market data
    tickers = portfolio_df['ticker'].tolist()
    market_data = fetch_all_holdings_data(tickers)
    
    # Add technical indicators
    benchmark_df = market_data.get('BENCHMARK')
    for ticker in market_data:
        if ticker != 'BENCHMARK':
            market_data[ticker] = add_technical_indicators(market_data[ticker], benchmark_df)
        else:
            market_data[ticker] = add_technical_indicators(market_data[ticker])
    
    # Calculate RS scores using UNIVERSE-LEVEL ranking (not intra-portfolio)
    universe_rs = compute_universe_rs_scores(tickers)
    
    if universe_rs and len(universe_rs) >= 100:
        # Universe RS available — use it (correct method)
        rs_data = _build_rs_data_from_universe(tickers, universe_rs, market_data)
        print(f"[RS] Using universe-level RS ({len(universe_rs)} stocks ranked)")
    else:
        # Fallback to old method if universe data unavailable
        rs_data = calculate_all_rs_scores(market_data)
        print(f"[RS] ⚠️ Falling back to portfolio-level RS (universe unavailable)")
    
    # Calculate total portfolio value
    total_value = calculate_total_portfolio_value(portfolio_df, market_data)
    
    # Get Industry mapping
    industry_map = get_industry_mapping()
    
    # Update/Fetch portfolio earnings cache
    try:
        earnings_cache = update_portfolio_earnings(tickers)
    except Exception as e:
        print(f"[Earnings Engine] Error in load_data: {e}")
        earnings_cache = {}
    
    return portfolio_df, market_data, rs_data, total_value, industry_map, earnings_cache


def main():
    # Header
    from components import render_header
    render_header("📊 KUSH TRACKER", "Portfolio Execution & Risk Management Dashboard")
    
    # Journal Notification
    today_str = datetime.now().strftime("%Y-%m-%d")
    if not get_journal_entry(today_str):
        st.warning("🚨 **Daily Journal Pending:** You have not logged your institutional bias today. Go to the **daily journal** page to review your analytics.")
    
    # Sidebar Telemetry & System Controls
    with st.sidebar:
        st.markdown("### 🌐 Market Sessions")
        from components import render_market_telemetry_sidebar
        render_market_telemetry_sidebar()
        
        st.markdown("### ⚙️ Terminal Controls")
        if st.button("🔄 Refresh Data & Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    
    # Load data with spinner
    with st.spinner("Loading portfolio data..."):
        portfolio_df, market_data, rs_data, total_value, industry_map, earnings_cache = load_data()
    
    if portfolio_df is None or portfolio_df.empty:
        st.error("❌ No portfolio data found. Check your Google Sheets connection.")
        return

    t_col = 'ticker' if 'ticker' in portfolio_df.columns else 'NSE TICKER' if 'NSE TICKER' in portfolio_df.columns else portfolio_df.columns[0]
    active_portfolio_tickers = set(portfolio_df[t_col].astype(str).str.replace('.NS', '').str.replace('.BO', '').str.strip())


    # Check for newly reported portfolio earnings (reported within last 7 days)
    new_results = get_portfolio_new_results_alerts(earnings_cache, active_tickers=active_portfolio_tickers)

    if new_results:
        summary_items = []
        for r in new_results:
            e_val = r.get('eps_yoy_pct')
            s_val = r.get('sales_yoy_pct')
            e_str = f"+{e_val:.1f}%" if e_val is not None and e_val > 0 else f"{e_val:.1f}%" if e_val is not None else "N/A"
            s_str = f"+{s_val:.1f}%" if s_val is not None and s_val > 0 else f"{s_val:.1f}%" if s_val is not None else "N/A"
            summary_items.append(f"**{r['ticker']}** ({r['latest_quarter']}: EPS {e_str} YoY, Sales {s_str} YoY {r['verdict']})")
        
        st.info(f"🚨 **PORTFOLIO EARNINGS SEASON ALERT:** {len(new_results)} holdings reported results in the last 7 days:\n\n" + " • " + "\n • ".join(summary_items))
        
    # Check cache staleness
    from market_data import get_cache_staleness
    staleness = get_cache_staleness()
    if staleness > 0:
        st.warning(f"⚠️ Index data is {staleness} days old. The NSE fetch might be failing, so the system is securely falling back to the last known cache to keep the app running.", icon="⚠️")
    
    # Check if macro data exists in session state (from Market Regime page or cached here)
    if 'macro_results' not in st.session_state:
        st.session_state['macro_results'] = None
        
    @st.cache_data(ttl=7200)
    def fetch_market_navigator_data(cache_buster=2):
        universe = fetch_universe_with_industry()
        if universe is None or universe.empty or 'Symbol' not in universe.columns:
            st.error("Failed to fetch universe data from NSE. Please try again later.")
            return None
        tks = universe['Symbol'].tolist()
        idx_n50 = yf.download('^NSEI', period="250d", progress=False)
        idx_n500 = yf.download('^CRSLDX', period="250d", progress=False)
        hist_df = fetch_yfinance_batch(tks, days=500)
        return process_macro_data(hist_df, universe, idx_n50, idx_n500)

    # Add a top-level insights trigger if missing
    if st.session_state['macro_results'] is None:
        with st.spinner("🚀 Loading Professional Market Navigator (Fetching 750 stocks for elite market intelligence... ~15s)"):
            result = fetch_market_navigator_data()
            if result is None:
                st.session_state['macro_results'] = {} # Prevent infinite loop
                st.warning("Market Navigator disabled due to NSE fetch failure.")
            else:
                st.session_state['macro_results'] = result
                st.rerun()

    # =========================================================================
    # ELITE DAILY INSIGHTS COCKPIT
    # =========================================================================
    macro_res = st.session_state.get('macro_results')
    benchmark_df = market_data.get('BENCHMARK')
    regime = "Unknown"
    regime_color = "#64748b"
    exposure_guide = generate_exposure_guide("Unknown", None)
    
    dd_count = 0
    current_idx_price = 0
    sma50 = 1 # Avoid division by zero later
    
    if benchmark_df is not None and not benchmark_df.empty:
        dd_count, _ = calculate_distribution_days(benchmark_df)
        current_idx_price = benchmark_df['close'].iloc[-1]
        sma50 = benchmark_df['sma_50'].iloc[-1] if 'sma_50' in benchmark_df.columns else 0
        regime, color_name, r_emoji = get_market_regime_label(dd_count, current_idx_price, sma50)
        
        # Color mapping
        if color_name == 'green': regime_color = '#10b981'
        elif color_name == 'yellow': regime_color = '#f59e0b'
        elif color_name == 'red': regime_color = '#ef4444'
            
        # Wyckoff Change of Character
        choch = detect_change_of_character(benchmark_df)
    else:
        choch = None
    
    # Fetch live Nifty 500 benchmark telemetry directly for 100% precision
    from macro_regime_engine import fetch_nifty500_telemetry
    n500_telem = fetch_nifty500_telemetry()

    if macro_res or n500_telem:
        # ---------------------------------------------------------------------
        # WORLD-CLASS EXECUTIVE HEADER & TELEMETRY INDICATOR
        # ---------------------------------------------------------------------
        st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 16px; padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.06); flex-wrap: wrap; gap: 10px;">
    <div>
        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
            <span style="font-size: 1.5rem;">🦅</span>
            <span style="font-size: 1.6rem; font-weight: 800; letter-spacing: -0.5px; color: #f8fafc;">
                Institutional Market Commander Cockpit
            </span>
            <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); padding: 3px 10px; border-radius: 9999px; font-size: 0.68rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; display: inline-flex; align-items: center; gap: 6px;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981;"></span> LIVE TELEMETRY
            </span>
        </div>
        <div style="font-size: 0.80rem; color: #94a3b8; margin-top: 5px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
            <span>Nifty 500 Benchmark</span>
            <span style="color: #475569;">•</span>
            <span>Ross Haber Progressive Exposure</span>
            <span style="color: #475569;">•</span>
            <span>O'Neil Distribution Matrix</span>
            <span style="color: #475569;">•</span>
            <span>750 Universe Breadth</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 1. EXTRACT TELEMETRY & PROGRESSIVE EXPOSURE
        # ---------------------------------------------------------------------
        if n500_telem and n500_telem.get('exposure_plan'):
            exp_plan = n500_telem['exposure_plan']
        elif macro_res and macro_res.get('exposure_plan'):
            exp_plan = macro_res.get('exposure_plan')
        else:
            from macro_regime_engine import get_progressive_exposure_plan
            exp_plan = get_progressive_exposure_plan(regime, dd_count)

        deck_state = exp_plan.get('deck_state', 'COLD DECK')
        deck_tone = exp_plan.get('deck_tone', 'Defensive')
        exp_range = exp_plan.get('exposure_range', '0% – 10% (Maximum Cash)')
        pilot_size = exp_plan.get('pilot_size', '0% (No New Buys)')
        exp_rule = exp_plan.get('rule', 'Capital preservation mode.')
        accent_color = exp_plan.get('accent_color', '#ef4444')

        # Haber 21 SMA Buffer
        if n500_telem:
            buf_pct = n500_telem.get('pct_diff_sma21', 0.0)
            consecutive_closes_21 = n500_telem.get('consecutive_closes_below_21', 0)
            haber_triggered = n500_telem.get('haber_triggered', False)
        else:
            haber_21 = macro_res.get('haber_21_n500', {}) if macro_res else {}
            buf_pct = haber_21.get('pct_diff_sma21', 0.0)
            consecutive_closes_21 = haber_21.get('consecutive_closes', 0)
            haber_triggered = haber_21.get('triggered', False)
        buf_color = "#10b981" if buf_pct >= 0 else "#ef4444"

        if haber_triggered and consecutive_closes_21 >= 2:
            haber_badge = f'<div style="color: #f87171; font-size: 0.65rem; font-weight: 800; text-transform: uppercase; margin-top: 3px; letter-spacing: 0.4px;">🚨 {consecutive_closes_21} Closes Below</div>'
        elif buf_pct >= 0:
            haber_badge = '<div style="color: #34d399; font-size: 0.65rem; font-weight: 800; text-transform: uppercase; margin-top: 3px; letter-spacing: 0.4px;">🟢 Above 21 SMA</div>'
        else:
            haber_badge = '<div style="color: #f59e0b; font-size: 0.65rem; font-weight: 800; text-transform: uppercase; margin-top: 3px; letter-spacing: 0.4px;">⚠️ Testing 21 SMA</div>'

        # Distribution Details & DCR
        if n500_telem:
            cluster_detected = n500_telem.get('cluster_detected', False)
            cluster_count_7d = n500_telem.get('cluster_count_7d', 0)
            n500_dd = n500_telem.get('dd_count', dd_count)
            std_dd = n500_telem.get('standard_dd_count', 0)
            churn_dd = n500_telem.get('churning_dd_count', 0)
            hammer_dd = n500_telem.get('hammer_acc_count', 0)
            latest_dcr = n500_telem.get('latest_dcr', 0.5)
            latest_dcr_verdict = n500_telem.get('latest_midpoint_verdict', 'Neutral')
            status_label = n500_telem.get('status_label', regime)
            status_color_raw = n500_telem.get('status_color', regime_color)
            status_emoji = n500_telem.get('status_emoji', '🔴')
            dist_sma50 = n500_telem.get('dist_sma50', 0.0)
        else:
            n500_details = macro_res.get('n500_dd_details', {}) if macro_res else {}
            cluster_detected = n500_details.get('cluster_detected', False)
            cluster_count_7d = n500_details.get('cluster_count_7d', 0)
            n500_dd = macro_res.get('n500_dd', dd_count) if macro_res else dd_count
            std_dd = n500_details.get('standard_dd_count', 0)
            churn_dd = n500_details.get('churning_dd_count', 0)
            hammer_dd = n500_details.get('hammer_acc_count', 0)
            latest_dcr = n500_details.get('latest_dcr', 0.5)
            latest_dcr_verdict = n500_details.get('latest_midpoint_verdict', 'Neutral')
            status_label = macro_res.get('status_label', regime) if macro_res else regime
            status_color_raw = macro_res.get('status_color', regime_color) if macro_res else regime_color
            status_emoji = macro_res.get('status_emoji', '🔴' if 'Correction' in regime else '🟡' if 'Pressure' in regime else '🟢') if macro_res else '🔴'
            curr_price = macro_res.get('n500_curr', current_idx_price) if macro_res else current_idx_price
            curr_sma50 = macro_res.get('n500_sma50', sma50) if macro_res else sma50
            dist_sma50 = ((curr_price - curr_sma50) / curr_sma50 * 100.0) if curr_sma50 > 0 else 0.0

        if status_color_raw == 'red' or 'Correction' in status_label:
            status_color = '#ef4444'
        elif status_color_raw == 'yellow' or 'Pressure' in status_label:
            status_color = '#f59e0b'
        elif status_color_raw == 'green' or 'Confirmed' in status_label:
            status_color = '#10b981'
        else:
            status_color = status_color_raw if str(status_color_raw).startswith('#') else '#ef4444'

        dist_sma50_color = '#10b981' if dist_sma50 >= 0 else '#ef4444'
        sma50_stance = "Above 50 SMA" if dist_sma50 >= 0 else "Below 50 SMA"

        # Volume Flow & Smart Money
        try:
            from breadth_engine import get_latest_volume_and_ad_distribution
            vol_flow = get_latest_volume_and_ad_distribution('IN')
            ud_ratio = vol_flow.get('up_down_volume_ratio', 1.0)
            up_vol = vol_flow.get('up_volume', 0.0)
            down_vol = vol_flow.get('down_volume', 0.0)
            tot_vol = up_vol + down_vol
            adv_vol_pct = (up_vol / tot_vol * 100.0) if tot_vol > 0 else 50.0
            smart_money = vol_flow.get('smart_money_ratio', 1.0)
        except Exception:
            ud_ratio, adv_vol_pct, smart_money = 1.0, 50.0, 1.0

        ud_color = '#10b981' if ud_ratio >= 1.2 else '#ef4444' if ud_ratio <= 0.8 else '#f59e0b'

        # Super-Leader Breadth (Top 100 RS)
        try:
            from breadth_engine import compute_top_rs_breadth_history
            rs_hist = compute_top_rs_breadth_history('IN', lookback_days=5)
            if not rs_hist.empty:
                latest_rs_row = rs_hist.iloc[-1]
                top100_21_ema = float(latest_rs_row.get('top100_above_21_ema', 0.0))
                top100_50_sma = float(latest_rs_row.get('top100_above_50_sma', 0.0))
                broad_50_sma = float(latest_rs_row.get('broad_above_50_sma', 0.0))
                broad_20_ema = float(latest_rs_row.get('broad_above_20_ema', 0.0))
                rs_spread = float(latest_rs_row.get('leadership_spread', 0.0))
            else:
                top100_21_ema, top100_50_sma, broad_50_sma, broad_20_ema, rs_spread = 0.0, 0.0, 0.0, 0.0, 0.0
        except Exception:
            top100_21_ema, top100_50_sma, broad_50_sma, broad_20_ema, rs_spread = 0.0, 0.0, 0.0, 0.0, 0.0

        # Qualitative Leadership & FTD
        leadership_qual = macro_res.get('leadership_qual', {}) if macro_res else {}
        lead_grade = leadership_qual.get('leadership_grade', 'B')
        near_high_pct = leadership_qual.get('near_high_pct', 0.0)
        ftd_detected = macro_res.get('ftd_detected', False) if macro_res else False
        ftd_date = macro_res.get('ftd') if macro_res else None
        ftd_status = f"Confirmed ({ftd_date})" if ftd_detected and ftd_date else "Watching (No Active FTD)"

        def _to_rgb(hex_code):
            hc = str(hex_code).lstrip('#')
            if len(hc) == 6:
                try:
                    return f"{int(hc[0:2], 16)}, {int(hc[2:4], 16)}, {int(hc[4:6], 16)}"
                except Exception:
                    pass
            return "100, 116, 139"

        deck_rgb = _to_rgb(accent_color)
        status_rgb = _to_rgb(status_color)

        # ---------------------------------------------------------------------
        # TIER 1: MASTER PROGRESSIVE EXPOSURE HUD (FULL WIDTH)
        # ---------------------------------------------------------------------
        st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.88)); border: 1px solid rgba(255, 255, 255, 0.08); border-left: 6px solid {accent_color}; border-radius: 14px; padding: 18px 24px; margin-bottom: 14px; box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.6), 0 0 20px rgba({deck_rgb}, 0.12);">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
<div style="flex: 1; min-width: 300px;">
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
<span style="font-size: 1.5rem; font-weight: 900; color: {accent_color}; letter-spacing: 0.3px; text-shadow: 0 0 20px rgba({deck_rgb}, 0.35);">{deck_state}</span>
<span style="background: rgba({deck_rgb}, 0.2); color: {accent_color}; border: 1px solid rgba({deck_rgb}, 0.5); padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px;">{deck_tone}</span>
</div>
<div style="background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.05); padding: 7px 14px; border-radius: 6px; font-size: 0.84rem; color: #cbd5e1; line-height: 1.4;">
<strong style="color: #f8fafc;">Exposure Rule:</strong> {exp_rule}
</div>
</div>
<div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
<div style="text-align: right; background: rgba(0,0,0,0.32); padding: 10px 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); min-width: 170px;">
<div style="color: #94a3b8; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 3px;">Max Recommended Exposure</div>
<div style="color: #f8fafc; font-size: 1.35rem; font-weight: 900; font-family: 'JetBrains Mono', monospace;">{exp_range}</div>
</div>
<div style="text-align: right; background: rgba(0,0,0,0.32); padding: 10px 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); min-width: 140px;">
<div style="color: #94a3b8; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 3px;">Pilot Sizing</div>
<div style="color: {accent_color}; font-size: 1.35rem; font-weight: 900; font-family: 'JetBrains Mono', monospace;">{pilot_size}</div>
</div>
<div style="text-align: right; background: rgba(0,0,0,0.32); padding: 10px 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); min-width: 170px;">
<div style="color: #94a3b8; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 3px;">Nifty 500 vs 21 SMA</div>
<div style="color: {buf_color}; font-size: 1.35rem; font-weight: 900; font-family: 'JetBrains Mono', monospace;">{buf_pct:+.1f}%</div>
{haber_badge}
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

        # Acute Distribution Cluster Warning (Conditional)
        if cluster_detected:
            st.markdown(f"""
<div style="background: linear-gradient(90deg, rgba(239, 68, 68, 0.16) 0%, rgba(15, 23, 42, 0.85) 100%); border: 1px solid rgba(239, 68, 68, 0.45); border-left: 6px solid #ef4444; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px; box-shadow: 0 4px 20px rgba(239, 68, 68, 0.15);">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
<div style="display: flex; align-items: center; gap: 14px;">
<span style="font-size: 1.4rem;">🚨</span>
<div>
<strong style="color: #f87171; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.5px;">Acute Distribution Cluster Alert: {cluster_count_7d} Distribution Days in Last 7 Sessions</strong>
<p style="margin: 3px 0 0 0; color: #cbd5e1; font-size: 0.82rem; line-height: 1.4;">
<b>Ross Haber Rule:</b> Multiple distribution/churning days tightly compressed within 5–7 sessions indicate heavy institutional unloading. Halt new pilot positions; enforce strict defensive trailing stops.
</p>
</div>
</div>
<span style="background: rgba(239, 68, 68, 0.25); color: #fca5a5; border: 1px solid #ef4444; padding: 5px 14px; border-radius: 6px; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; white-space: nowrap; box-shadow: 0 0 10px rgba(239, 68, 68, 0.3);">DEFENSIVE LOCKDOWN</span>
</div>
</div>
""", unsafe_allow_html=True)

        # Low Breadth Beach Ball Alert (Conditional on 50-SMA breadth < 35%)
        try:
            from database import get_connection, get_beach_ball_leaderboard
            bb_conn = get_connection()
            bb_cursor = bb_conn.cursor()
            bb_cursor.execute("SELECT above_50_pct FROM market_breadth_daily WHERE market = 'IN' ORDER BY date DESC LIMIT 1")
            bb_row = bb_cursor.fetchone()
            bb_conn.close()
            cur_breadth_50 = float(bb_row[0]) if bb_row and bb_row[0] is not None else broad_50_sma
            if cur_breadth_50 < 35.0:
                bb_leaders = get_beach_ball_leaderboard(market='INDIA', breadth_threshold=35.0, min_power_plays_3m=2)
                top_3_str = ", ".join([f"<b>{x['ticker'].replace('.NS','')}</b> ({x['power_plays_3m']}x)" for x in bb_leaders[:4]]) if bb_leaders else "Submerged leaders detected"
                st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(15, 23, 42, 0.95)); border: 1px solid rgba(245, 158, 11, 0.35); border-left: 6px solid #f59e0b; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px; box-shadow: 0 4px 20px rgba(245, 158, 11, 0.12);">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
<div style="display: flex; align-items: center; gap: 14px;">
<span style="font-size: 1.5rem;">🏖️</span>
<div>
<div style="display: flex; align-items: center; gap: 8px;">
<span style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); padding: 1px 7px; border-radius: 4px; font-size: 0.68rem; font-weight: 800; text-transform: uppercase;">LOW BREADTH REGIME: {cur_breadth_50:.1f}% &lt; 35%</span>
<strong style="color: #fbbf24; font-size: 0.92rem; text-transform: uppercase; letter-spacing: 0.5px;">Institutional Beach Ball Accumulation in Progress</strong>
</div>
<p style="margin: 3px 0 0 0; color: #cbd5e1; font-size: 0.82rem; line-height: 1.4;">
Broad market 50-SMA breadth is depressed, but smart money is aggressively absorbing select leaders on heavy volume. 
Top Submerged Accumulators: {top_3_str}
</p>
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)
                st.page_link("pages/beach_ball_accumulation.py", label="🏖️ Open Beach Ball Accumulation Radar (Ranked by 3M & 1M Power Plays) ➔", icon="🏖️")
                st.markdown("<br/>", unsafe_allow_html=True)
        except Exception as e:
            print(f"[app.py] Low breadth banner error: {e}")

        # ---------------------------------------------------------------------
        # 🧭 MARKET EXECUTION PERSONA HUD (TREND-PERSISTENCE VS HIGH-VELOCITY ROTATION)
        # ---------------------------------------------------------------------
        try:
            from market_persona_engine import calculate_composite_market_persona
            persona_data = calculate_composite_market_persona(market='INDIA', force_recompute=False)
            
            p_score = persona_data.get('composite_score', 50.0)
            p_label = persona_data.get('persona_label', 'Selective Consolidation Regime')
            p_mandate = persona_data.get('trading_mandate', 'Hybrid Trading')
            p_details = persona_data.get('details', {})
            p_summary = p_details.get('mandate_summary', '')
            badge_color = p_details.get('badge_color', '#f59e0b')
            badge_bg = p_details.get('badge_bg', 'rgba(245, 158, 11, 0.15)')
            p_icon = p_details.get('icon', '🧭')
            regime_mode = p_details.get('regime_mode', 'Rotational Digestion Market')
            
            p1_info = p_details.get('pillar1', {})
            p2_info = p_details.get('pillar2', {})
            p3_info = p_details.get('pillar3', {})
            p4_info = p_details.get('pillar4', {})
            
            # Pillar 1 stats & chips
            tenure_pct = p1_info.get('tenure_pct', 0.0)
            retained_cnt = p1_info.get('retained_count', 0)
            p1_chip = f'<span style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 700;">🔴 Violent Whiplash</span>' if retained_cnt <= 1 else (f'<span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 700;">⚠️ Moderate Rotation</span>' if retained_cnt == 2 else f'<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 700;">🟢 Secular Persistence</span>')
            
            # Pillar 2 stats & chips
            bo_health_pct = p2_info.get('breakout_health_pct', 0.0)
            total_bo = p2_info.get('total_breakouts', 0)
            squat_cnt = p2_info.get('squat_count', 0)
            squat_pct = (squat_cnt / total_bo * 100) if total_bo > 0 else 0.0
            p2_chip = f'<span style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 700;">⚠️ {squat_pct:.0f}% Squats</span>' if squat_pct >= 35 else (f'<span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 700;">⚠️ Selective ({bo_health_pct:.0f}% Holds)</span>' if bo_health_pct < 60 else f'<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 700;">🟢 {bo_health_pct:.0f}% Continuation</span>')
            
            # Pillar 3 stats & chips
            lead_21 = p3_info.get('leader_21ema_pct', 0.0)
            p3_chip = f'<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 700;">🟢 Respecting 21-EMA</span>' if lead_21 >= 70 else (f'<span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 700;">⚠️ Normal Digestion</span>' if lead_21 >= 50 else f'<span style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 700;">🔴 Leaders Slicing 21</span>')
            
            # Pillar 4 stats & chips
            mc_spread = p4_info.get('mega_cap_spread', 0.0)
            is_trap = p4_info.get('is_divergence_trap', False)
            p4_chip = f'<span style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 700;">⚠️ Illusion Trap</span>' if is_trap else (f'<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 700;">🟢 Broad Harmony</span>' if mc_spread >= 0 else f'<span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 700;">⚠️ Narrow Dispersion</span>')

            hud_html = f"""<div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.92)); border: 1px solid rgba(255, 255, 255, 0.08); border-left: 6px solid {badge_color}; border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.55);">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; margin-bottom: 14px;">
<div style="display: flex; align-items: center; gap: 14px;">
<span style="font-size: 1.85rem; line-height: 1;">{p_icon}</span>
<div>
<div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
<span style="color: #94a3b8; font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.9px;">MARKET EXECUTION REGIME</span>
<span style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_color}60; padding: 2px 10px; border-radius: 9999px; font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.6px; font-family: 'JetBrains Mono', monospace;">SCORE: {p_score} / 100</span>
</div>
<div style="font-size: 1.45rem; font-weight: 900; color: #f8fafc; margin-top: 3px; letter-spacing: 0.2px;">{p_label}</div>
<div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px; font-weight: 500;">Dynamic Environment: <span style="color: #cbd5e1; font-weight: 600;">{regime_mode}</span></div>
</div>
</div>
<div style="text-align: right;">
<div style="color: #94a3b8; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px;">MANDATED EXECUTION PLAYBOOK</div>
<span style="background: {badge_bg}; color: {badge_color}; border: 1.5px solid {badge_color}; padding: 7px 18px; border-radius: 8px; font-size: 0.94rem; font-weight: 900; letter-spacing: 0.8px; display: inline-block;">{p_mandate.upper()}</span>
</div>
</div>
<div style="background: rgba(0, 0, 0, 0.42); border-left: 3px solid {badge_color}; border-top: 1px solid rgba(255, 255, 255, 0.04); border-right: 1px solid rgba(255, 255, 255, 0.04); border-bottom: 1px solid rgba(255, 255, 255, 0.04); border-radius: 8px; padding: 12px 16px; margin-bottom: 14px;">
<div style="display: flex; align-items: baseline; gap: 8px; margin-bottom: 4px;">
<span style="color: {badge_color}; font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px;">⚡ TACTICAL DIRECTIVE</span>
</div>
<div style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.5;">{p_summary}</div>
</div>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr)); gap: 12px;">
<div style="background: rgba(0, 0, 0, 0.28); border: 1px solid rgba(255, 255, 255, 0.06); padding: 12px 14px; border-radius: 8px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
<span style="color: #94a3b8; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">P1: SECTOR TENURE</span>
{p1_chip}
</div>
<div style="font-size: 1.25rem; font-weight: 900; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{tenure_pct:.0f}% <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 500;">({retained_cnt}/5 Retained)</span></div>
<div style="font-size: 0.72rem; color: #94a3b8; margin-top: 3px;">Top 5 group persistence over 40 sessions</div>
</div>
<div style="background: rgba(0, 0, 0, 0.28); border: 1px solid rgba(255, 255, 255, 0.06); padding: 12px 14px; border-radius: 8px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
<span style="color: #94a3b8; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">P2: BREAKOUT HEALTH</span>
{p2_chip}
</div>
<div style="font-size: 1.25rem; font-weight: 900; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{bo_health_pct:.1f}% <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 500;">(T+10 Holds)</span></div>
<div style="font-size: 0.72rem; color: #94a3b8; margin-top: 3px;">{squat_cnt} of {total_bo} breakouts squatted back into base</div>
</div>
<div style="background: rgba(0, 0, 0, 0.28); border: 1px solid rgba(255, 255, 255, 0.06); padding: 12px 14px; border-radius: 8px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
<span style="color: #94a3b8; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">P3: LEADER 21-EMA</span>
{p3_chip}
</div>
<div style="font-size: 1.25rem; font-weight: 900; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{lead_21:.1f}% <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 500;">(Top 100 RS)</span></div>
<div style="font-size: 0.72rem; color: #94a3b8; margin-top: 3px;">Leading momentum stocks holding 21 EMA</div>
</div>
<div style="background: rgba(0, 0, 0, 0.28); border: 1px solid rgba(255, 255, 255, 0.06); padding: 12px 14px; border-radius: 8px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
<span style="color: #94a3b8; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">P4: MEGA-CAP SPREAD</span>
{p4_chip}
</div>
<div style="font-size: 1.25rem; font-weight: 900; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{mc_spread:+.1f}% <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 500;">Spread</span></div>
<div style="font-size: 0.72rem; color: #94a3b8; margin-top: 3px;">Index dist to 50 SMA vs broad 750 breadth</div>
</div>
</div>
</div>"""
            st.markdown(hud_html, unsafe_allow_html=True)

            with st.expander("🔬 Why this score? View Mathematical Audit & Raw Data Behind Every Pillar"):
                d_tab1, d_tab2, d_tab3, d_tab4 = st.tabs([
                    "🏛️ Pillar 1: Sector Tenure Audit",
                    "🎯 Pillar 2: Breakout T+10 Ledger",
                    "📈 Pillar 3: Top 100 RS Adherence",
                    "⚖️ Pillar 4: Mega-Cap Divergence"
                ])
                
                with d_tab1:
                    st.markdown(f"**Comparison Window:** {p1_info.get('today_date', 'Today')} vs. **{p1_info.get('past_date', '40 Sessions Ago')}** (~40 trading sessions / 2 months)")
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.markdown("**Top 5 Industry Groups Today:**")
                        t5 = p1_info.get('today_top5', [])
                        if t5:
                            for idx, s in enumerate(t5, 1):
                                ret_icon = "🟢 **Retained Leader**" if s.get('retained') else "🆕 *New entrant*"
                                st.markdown(f"{idx}. **{s.get('industry')}** (Avg RS: {s.get('avg_rs', 0):.1f}) — {ret_icon}")
                        else:
                            st.info("No current sector ranking data.")
                    with col_t2:
                        st.markdown(f"**Top 5 Industry Groups 40 Sessions Ago ({p1_info.get('past_date', '')}):**")
                        p5 = p1_info.get('past_top5', [])
                        if p5:
                            for idx, s in enumerate(p5, 1):
                                st.markdown(f"{idx}. **{s.get('industry')}** (Avg RS: {s.get('avg_rs', 0):.1f})")
                        else:
                            st.info("No historical sector ranking data.")
                    st.caption(f"**Tenure Calculation:** {retained_cnt} of 5 sectors retained across 40 sessions = {tenure_pct:.0f}% tenure rate.")

                with d_tab2:
                    st.markdown(f"**Recent Stage 2 Breakouts Audited ({total_bo} Total):** Evaluated strictly at T+10 trading sessions forward.")
                    st.markdown(f"• **Continuation Successes (Held >= +4%):** {p2_info.get('success_count', 0)} ({bo_health_pct:.1f}%)\n"
                                f"• **Squats / Base Failures (< 0%):** {p2_info.get('squat_count', 0)} ({squat_pct:.1f}%)\n"
                                f"• **Choppy / Flat (0% - 4%):** {p2_info.get('choppy_count', 0)}")
                    
                    bo_list = p2_info.get('breakout_list', [])
                    if bo_list:
                        bo_df = pd.DataFrame(bo_list)
                        bo_df = bo_df.rename(columns={
                            'date': 'Breakout Date',
                            'eval_date': 'T+10 Date',
                            'ticker': 'Ticker',
                            'pivot': 'Pivot Price',
                            'price_t10': 'Price at T+10',
                            'return_t10_pct': 'T+10 Return %',
                            'status': 'Verdict'
                        })
                        st.dataframe(bo_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No breakouts recorded in the lookback window.")

                with d_tab3:
                    st.markdown("**Top 100 Relative Strength Institutional Leaders:**")
                    st.markdown(f"• **% of Top 100 RS Leaders Above 21-Day EMA:** **{p3_info.get('top100_above_21_ema', 0.0):.1f}%**\n"
                                f"• **% of Top 100 RS Leaders Above 50-Day SMA:** **{p3_info.get('top100_above_50_sma', 0.0):.1f}%**\n"
                                f"• **Institutional Leadership Spread vs Universe:** **{p3_info.get('leadership_spread', 0.0):+.1f}%**")
                    st.caption("When over 70% of leading stocks stay above their 21-day EMA, it confirms institutional accumulation with stair-step price trends. Below 50%, leaders are frequently undercutting support.")

                with d_tab4:
                    st.markdown(f"**Mega-Cap Benchmark ({p4_info.get('index_name', 'Nifty 50')}) vs. Broad 750-Stock Breadth:**")
                    st.markdown(f"• **{p4_info.get('index_name', 'Nifty 50')} Distance from 50 SMA:** **{p4_info.get('index_dist_50sma', 0.0):+.2f}%**\n"
                                f"• **Broad Market 750 Universe % Above 50 SMA:** **{p4_info.get('broad_breadth_50sma', 0.0):.1f}%**\n"
                                f"• **Net Breadth Spread:** **{mc_spread:+.1f}%**\n"
                                f"• **Divergence Trap Status:** **{p4_info.get('verdict', 'Neutral')}**")
                    if is_trap:
                        st.warning("⚠️ **Mega-Cap Illusion Trap Warning:** The headline benchmark index is elevated near highs, but broader mid/small-cap market breadth is severely damaged (< 45% > 50 SMA). This signals narrow index dispersion where headline advances mask systematic distribution under the surface.")
                    else:
                        st.success("✅ **Healthy Alignment:** Benchmark and broad market breadth are in structural harmony without deceptive divergence.")

        except Exception as e:
            print(f"[app.py] Market Persona HUD error: {e}")

        # ---------------------------------------------------------------------
        # 🧠 INSTITUTIONAL ARTIFICIAL BRAIN: 5-STAGE TIDE-TURN & 7-PILLAR CONFLUENCE FUNNEL
        # ---------------------------------------------------------------------
        try:
            from brain_synthesis_engine import run_artificial_brain_synthesis, compute_portfolio_traction_mirror
            force_brain_sync = st.session_state.pop('force_brain_recompute', False)
            brain_res = run_artificial_brain_synthesis(
                benchmark_df=benchmark_df,
                portfolio_tickers=active_portfolio_tickers,
                industry_map=industry_map,
                market='INDIA',
                force_recompute=force_brain_sync
            )
            port_mirror = compute_portfolio_traction_mirror(portfolio_df, market_data)
            tt = brain_res.get('tide_turn', {}) or {}
            conf_list = brain_res.get('confluence', []) or []
            bar_dt_str = brain_res.get('matrix_bar_date', today_str)
            univ_cnt = int(brain_res.get('universe_count', 0) or 0)
            st_num = int(tt.get('stage_num', 3) or 3)
            st_color = tt.get('stage_color', '#f59e0b')

            # Build 5-Stage Progression Pills
            stage_steps_html = ""
            for s_item in tt.get('stages_bar', []):
                s_n = int(s_item.get('num', 1))
                is_cur = (s_n == st_num)
                is_done = (s_n < st_num)
                bg_s = f"linear-gradient(135deg, {st_color}33, {st_color}15)" if is_cur else ("rgba(16, 185, 129, 0.10)" if is_done else "rgba(15, 23, 42, 0.55)")
                bdr_s = f"2px solid {st_color}" if is_cur else ("1px solid rgba(16, 185, 129, 0.30)" if is_done else "1px solid rgba(255,255,255,0.06)")
                txt_c = "#f8fafc" if is_cur else ("#34d399" if is_done else "#64748b")
                dot_icon = "🟢" if is_done else ("🔥" if is_cur else "⚪")
                stage_steps_html += (
                    f'<div style="flex:1; min-width:135px; background:{bg_s}; border:{bdr_s}; border-radius:8px; padding:8px 10px; text-align:center;">'
                    f'<div style="font-size:0.73rem; font-weight:800; color:{txt_c};">{dot_icon} {s_item.get("name")}</div>'
                    f'<div style="font-size:0.66rem; color:#94a3b8; margin-top:2px; font-family:\'JetBrains Mono\',monospace;">{s_item.get("sub")}</div>'
                    f'</div>'
                )

            brain_hud_html = f"""<div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(23, 37, 84, 0.90)); border: 1px solid rgba(56, 189, 248, 0.25); border-left: 6px solid {st_color}; border-radius: 12px; padding: 18px 22px; margin-bottom: 16px; box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.6);">
<div style="display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 14px;">
<div style="min-width: 0; flex: 1;">
<div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
<span style="font-size: 1.35rem;">🧠</span>
<span style="color: #38bdf8; font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;">ARTIFICIAL BRAIN: AUTOMATED 5-STAGE TIDE-TURN STATE MACHINE</span>
<span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); padding: 2px 9px; border-radius: 999px; font-size: 0.66rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;">🟢 SYNCED TO BAR: {bar_dt_str} • {univ_cnt:,} STOCKS + {len(active_portfolio_tickers)}/{len(active_portfolio_tickers)} HOLDINGS</span>
</div>
<div style="font-size: 1.25rem; font-weight: 900; color: #f8fafc; margin-top: 4px;">{tt.get('stage_title', '')}</div>
</div>
<div style="background: {st_color}1f; border: 1.5px solid {st_color}; border-radius: 8px; padding: 7px 16px; text-align: right; flex-shrink: 0; white-space: nowrap;">
<div style="color: #cbd5e1; font-size: 0.62rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px;">BRAIN PROGRESSIVE EXPOSURE</div>
<div style="color: {st_color}; font-size: 1.05rem; font-weight: 900; font-family: 'JetBrains Mono', monospace; margin-top: 1px;">{tt.get('target_exposure', '')}</div>
</div>
</div>
<div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;">
{stage_steps_html}
</div>
<div style="background: rgba(0,0,0,0.38); border-left: 3px solid {st_color}; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; color: #e2e8f0; font-size: 0.84rem; line-height: 1.45;">
<strong style="color: {st_color};">🎯 Brain Execution Mandate:</strong> {tt.get('directive', '')}
</div>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
<div style="background: rgba(0,0,0,0.28); border: 1px solid rgba(255,255,255,0.05); border-radius: 7px; padding: 8px 12px;">
<div style="color:#94a3b8; font-size:0.65rem; font-weight:700; text-transform:uppercase;">🏖️ Ironclad Beach Balls</div>
<div style="color:#38bdf8; font-size:1.1rem; font-weight:900; font-family:'JetBrains Mono',monospace;">{tt.get('ironclad_count', 0)} Leaders</div>
<div style="color:#64748b; font-size:0.66rem;">Defying last 10D down-days</div>
</div>
<div style="background: rgba(0,0,0,0.28); border: 1px solid rgba(255,255,255,0.05); border-radius: 7px; padding: 8px 12px;">
<div style="color:#94a3b8; font-size:0.65rem; font-weight:700; text-transform:uppercase;">🌊 5D Panic / Surge (Bonde)</div>
<div style="color:#f8fafc; font-size:1.1rem; font-weight:900; font-family:'JetBrains Mono',monospace;"><span style="color:#ef4444;">{tt.get('panic_5d_count', 0)}</span> / <span style="color:#10b981;">{tt.get('surge_5d_count', 0)}</span></div>
<div style="color:#64748b; font-size:0.66rem;">FOMO (&gt;5 EMA): {tt.get('fomo_5ema_pct', 0):.0f}%</div>
</div>
<div style="background: rgba(0,0,0,0.28); border: 1px solid rgba(255,255,255,0.05); border-radius: 7px; padding: 8px 12px;">
<div style="color:#94a3b8; font-size:0.65rem; font-weight:700; text-transform:uppercase;">⚡ Up-Vol Thrust (Zweig 9:1)</div>
<div style="color:{'#10b981' if tt.get('is_9to1_thrust') else '#f8fafc'}; font-size:1.1rem; font-weight:900; font-family:'JetBrains Mono',monospace;">{tt.get('up_vol_ratio_pct', 50):.1f}% {'🔥 9:1' if tt.get('is_9to1_thrust') else ''}</div>
<div style="color:#64748b; font-size:0.66rem;">20-EMA Sequence: Step {tt.get('recapture_step', 0)}/5</div>
</div>
<div style="background: rgba(0,0,0,0.28); border: 1px solid rgba(255,255,255,0.05); border-radius: 7px; padding: 8px 12px;">
<div style="color:#94a3b8; font-size:0.65rem; font-weight:700; text-transform:uppercase;">🚀 O'Neil FTD &amp; Squat Audit</div>
<div style="color:{'#f59e0b' if tt.get('ftd_suspect_warning') else ('#10b981' if tt.get('ftd_active') else '#94a3b8')}; font-size:0.92rem; font-weight:800; font-family:'JetBrains Mono',monospace;">{tt.get('ftd_status_text', 'No Active FTD')}</div>
<div style="color:#64748b; font-size:0.66rem;">Breakout Health: {tt.get('breakout_health_pct', 50):.0f}%</div>
</div>
<div style="background: rgba(0,0,0,0.28); border: 1px solid rgba(255,255,255,0.05); border-radius: 7px; padding: 8px 12px;">
<div style="color:#94a3b8; font-size:0.65rem; font-weight:700; text-transform:uppercase;">🪞 My Portfolio Traction</div>
<div style="color:{port_mirror.get('color', '#f59e0b')}; font-size:0.92rem; font-weight:900; font-family:'JetBrains Mono',monospace;">{port_mirror.get('above_21_low_count', 0)}/{port_mirror.get('total_holdings', 0)} &gt; 21-Low ({port_mirror.get('above_21_low_pct', 0):.0f}%)</div>
<div style="color:#64748b; font-size:0.66rem;">{port_mirror.get('verdict', '')} • 🌸 {port_mirror.get('house_money_count', 0)} Cushion | 🛫 Pilots {port_mirror.get('pilot_green_count', 0)}/{port_mirror.get('pilot_total', 5)}</div>
</div>
</div>
</div>"""
            st.markdown(brain_hud_html, unsafe_allow_html=True)

            # Render Top 5 Omni-Page Confluence Funnel + Interactive 2,750+ Universe Expander
            if conf_list:
                with st.expander("🎯 **Artificial Brain: Today's Top Cross-Platform Confluence Funnel (7-Pillar Synthesis Across 2,750+ Stocks & All SME/BSE Holdings)**", expanded=True):
                    f_col1, f_col2 = st.columns([3.2, 1.0])
                    with f_col1:
                        funnel_scope = st.radio(
                            "Filter Confluence Funnel:",
                            [
                                f"🔥 Top 5 Actionable Leaders (All {univ_cnt:,} Stocks)",
                                "💼 My Portfolio Holdings in Funnel",
                                "🔒 HV1 Float-Lockup (>= 6.0x Volume Dry-Up)",
                                "🛡️ Ironclad Beach Balls (>= 60% Down-Day Hit Rate)",
                                "📋 Full Top 25 Institutional Funnel"
                            ],
                            horizontal=True,
                            key="brain_funnel_scope_radio",
                            label_visibility="collapsed"
                        )
                    with f_col2:
                        if st.button("⚡ Re-Scan 2,750+ Matrix Now", key="btn_rescan_brain_matrix", use_container_width=True):
                            st.session_state['force_brain_recompute'] = True
                            st.rerun()

                    if "My Portfolio" in funnel_scope:
                        filtered_conf = [c for c in conf_list if c.get('is_holding')]
                    elif "HV1 Float-Lockup" in funnel_scope:
                        filtered_conf = [c for c in conf_list if float(c.get('hv1_lockup_ratio', 0) or 0) >= 6.0]
                    elif "Ironclad" in funnel_scope:
                        filtered_conf = [c for c in conf_list if float(c.get('dd_hit_rate', 0) or 0) >= 60.0]
                    elif "Top 5" in funnel_scope:
                        filtered_conf = conf_list[:5]
                    else:
                        filtered_conf = conf_list[:25]

                    if not filtered_conf:
                        filtered_conf = conf_list[:5]
                        st.caption("Showing Top 5 overall leaders across the 2,750+ matrix:")

                    funnel_rows_html = []
                    for c in filtered_conf:
                        clean_tk = str(c.get('clean_ticker', ''))
                        tv_sym = clean_tk.replace('-SM', '').replace('-ST', '').replace('-BE', '')
                        tv_url = f"https://in.tradingview.com/chart/?symbol=NSE:{tv_sym}"
                        hold_pill = (
                            '<span style="background:rgba(16,185,129,0.2); color:#34d399; border:1px solid rgba(16,185,129,0.4); '
                            'padding:1px 6px; border-radius:4px; font-size:0.62rem; font-weight:800; margin-left:5px;">💼 HELD</span>'
                        ) if c.get('is_holding') else ''
                        b_score = float(c.get('brain_score', 0) or 0)
                        b_col = "#10b981" if b_score >= 70 else ("#38bdf8" if b_score >= 55 else "#f59e0b")
                        ret_1d = float(c.get('return_1d', 0) or 0)
                        ret_col = "#10b981" if ret_1d >= 0 else "#ef4444"
                        hv1_r = float(c.get('hv1_lockup_ratio', 0) or 0)
                        if hv1_r >= 6.0:
                            hv1_badge = f'<span style="background:rgba(16,185,129,0.18); color:#34d399; border:1px solid rgba(16,185,129,0.4); padding:2px 7px; border-radius:6px; font-weight:800; font-family:\'JetBrains Mono\',monospace;">🔒 {hv1_r:.1f}x</span>'
                        elif hv1_r > 0:
                            hv1_badge = f'<span style="color:#94a3b8; font-family:\'JetBrains Mono\',monospace;">📦 {hv1_r:.1f}x</span>'
                        else:
                            hv1_badge = '<span style="color:#64748b;">-</span>'

                        raw_badges = [b.strip() for b in str(c.get('badges_str', '')).split('•') if b.strip()]
                        badge_chips = "".join([
                            f'<span style="display:inline-block; background:rgba(30,41,59,0.85); color:#e2e8f0; border:1px solid rgba(148,163,184,0.25); padding:2px 8px; border-radius:999px; font-size:0.70rem; font-weight:600; margin:2px 4px 2px 0; white-space:nowrap;">{b_item}</span>'
                            for b_item in raw_badges
                        ])
                        exec_str = str(c.get('execution_plan', ''))
                        funnel_rows_html.append(
                            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.06);">'
                            f'<td style="padding:10px 10px; white-space:nowrap;"><a href="{tv_url}" target="_blank" style="color:#38bdf8; font-weight:800; text-decoration:none; font-family:\'JetBrains Mono\',monospace; font-size:0.84rem;">{clean_tk} ↗</a>{hold_pill}</td>'
                            f'<td style="padding:10px 8px; white-space:nowrap;"><span style="background:{b_col}22; color:{b_col}; border:1px solid {b_col}66; padding:3px 8px; border-radius:6px; font-weight:900; font-family:\'JetBrains Mono\',monospace; font-size:0.8rem;">{b_score:.0f} / 100</span></td>'
                            f'<td style="padding:10px 8px; white-space:nowrap; font-family:\'JetBrains Mono\',monospace; font-size:0.8rem; color:#f8fafc;"><b>{c.get("rs_score", 0):.0f}</b> <span style="color:#94a3b8;">({c.get("smooth_rs", 0):.1f}x)</span></td>'
                            f'<td style="padding:10px 8px; white-space:nowrap; font-family:\'JetBrains Mono\',monospace; font-size:0.8rem; color:#f8fafc;">₹{c.get("close", 0):,.1f} <span style="color:{ret_col}; font-weight:700;">({ret_1d:+.1f}%)</span></td>'
                            f'<td style="padding:10px 8px; white-space:nowrap; font-family:\'JetBrains Mono\',monospace; font-size:0.8rem; color:#34d399;">₹{c.get("ema_21_low", 0):,.1f}</td>'
                            f'<td style="padding:10px 8px; white-space:nowrap; font-size:0.78rem;">{hv1_badge}</td>'
                            f'<td style="padding:8px 10px; min-width:250px; line-height:1.4;">{badge_chips}</td>'
                            f'<td style="padding:8px 10px; min-width:290px; color:#cbd5e1; font-size:0.76rem; line-height:1.45; font-family:\'JetBrains Mono\',monospace;">{exec_str}</td>'
                            f'</tr>'
                        )

                    funnel_table_html = (
                        '<div style="overflow-x:auto; border:1px solid rgba(56,189,248,0.2); border-radius:10px; background:rgba(15,23,42,0.75); margin-bottom:8px;">'
                        '<table style="width:100%; border-collapse:collapse; text-align:left;">'
                        '<thead><tr style="background:rgba(30,41,59,0.9); color:#94a3b8; font-size:0.68rem; font-weight:800; text-transform:uppercase; letter-spacing:0.6px; border-bottom:1px solid rgba(255,255,255,0.1);">'
                        '<th style="padding:9px 10px;">Ticker</th>'
                        '<th style="padding:9px 8px;">Brain Score</th>'
                        '<th style="padding:9px 8px;">RS / Smooth</th>'
                        '<th style="padding:9px 8px;">Price (1D%)</th>'
                        '<th style="padding:9px 8px;">21-EMA Low</th>'
                        '<th style="padding:9px 8px;">HV1 Lockup</th>'
                        '<th style="padding:9px 10px;">7-Pillar Confluence Badges</th>'
                        '<th style="padding:9px 10px;">Regime-Gated Execution Plan (Trigger • Stop • Target)</th>'
                        '</tr></thead>'
                        f'<tbody>{"".join(funnel_rows_html)}</tbody></table></div>'
                    )
                    st.markdown(funnel_table_html, unsafe_allow_html=True)
                    st.caption(
                        "💡 **How the 7-Pillar Score (0–100) Works:** Combines **P1: Down-Day Resilience (20)** + **P2: HV1 Float-Lockup ≥6x (20)** + "
                        "**P3: Universe RS & RSNH Blue Dot (15)** + **P4: Today's Trigger (Oops/PEB/Inside/21-EMA Low) (15)** + **P5: Livermore Sister Tandem (10)** + "
                        "**P6: 100% Club Doubler & 0-DD Tape (10)** + **P7: Sales/EPS Growth (10)** directly from the 2,750+ stock matrix."
                    )
        except Exception as e:
            print(f"[app.py] Artificial Brain HUD error: {e}")

        # ---------------------------------------------------------------------
        # TIER 2: TACTICAL COMMAND GRID (3-COLUMN LAYOUT)
        # ---------------------------------------------------------------------
        c_col1, c_col2, c_col3 = st.columns(3)

        # --- COLUMN 1: MACRO REGIME & FLOW ---
        with c_col1:
            dcr_badge = f'<span style="background: rgba(239,68,68,0.2); color: #ef4444; border: 1px solid rgba(239,68,68,0.4); padding: 3px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 800;">🔴 Lower Half · Distributive Liquidation</span>' if latest_dcr < 0.50 else f'<span style="background: rgba(16,185,129,0.2); color: #10b981; border: 1px solid rgba(16,185,129,0.4); padding: 3px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 800;">🟢 Upper Half · Institutional Absorption</span>'
            
            dd_color = '#ef4444' if n500_dd >= 6 else '#f59e0b' if n500_dd >= 4 else '#10b981'
            dcr_pct_marker = max(4, min(96, int(latest_dcr * 100)))
            
            st.markdown(f"""
<div class="metric-card" style="border-top: 4px solid {status_color}; height: 100%; display: flex; flex-direction: column; justify-content: space-between; padding: 18px 20px;">
<div>
<div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
<p style="color: #94a3b8; font-size: 0.80rem; margin: 0; text-transform: uppercase; font-weight: 800; letter-spacing: 1px;">Macro Regime & Flow</p>
<span style="font-size: 0.68rem; color: #64748b; font-weight: 700; text-transform: uppercase;">NIFTY 500 TELEMETRY</span>
</div>

<div style="background: rgba({status_rgb}, 0.12); border: 1px solid rgba({status_rgb}, 0.35); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
<div style="display: flex; align-items: center; gap: 8px;">
<span style="font-size: 1.25rem;">{status_emoji}</span>
<span style="font-size: 1.15rem; font-weight: 800; color: {status_color}; letter-spacing: 0.3px;">{status_label}</span>
</div>
<span style="font-size: 0.68rem; font-weight: 800; color: {status_color}; background: rgba({status_rgb}, 0.25); border: 1px solid rgba({status_rgb}, 0.5); padding: 2px 8px; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.5px;">ACTIVE REGIME</span>
</div>

<div style="background: rgba(0,0,0,0.28); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 10px 12px; margin-bottom: 10px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
<span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Nifty 500 Dist Days</span>
<strong style="font-size: 1.15rem; font-family: 'JetBrains Mono', monospace; color: {dd_color};">{n500_dd} / 25</strong>
</div>
<div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px;">
<span style="background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); padding: 1px 7px; border-radius: 4px; font-size: 0.68rem; font-weight: 700;">{std_dd} Standard</span>
<span style="background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); padding: 1px 7px; border-radius: 4px; font-size: 0.68rem; font-weight: 700;">{churn_dd} Churning</span>
<span style="background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); padding: 1px 7px; border-radius: 4px; font-size: 0.68rem; font-weight: 700;">{hammer_dd} Hammer Absorbed</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 6px;">
<span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">vs 50 SMA</span>
<span style="font-size: 0.85rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: {dist_sma50_color};">{dist_sma50:+.1f}% ({sma50_stance})</span>
</div>
</div>

<div style="background: rgba(0,0,0,0.28); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 10px 12px; margin-bottom: 10px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
<span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">DCR Midpoint (O'Neil)</span>
<strong style="font-size: 1.05rem; font-family: 'JetBrains Mono', monospace; color: #f8fafc;">{latest_dcr:.2f}</strong>
</div>
<div style="height: 6px; border-radius: 3px; background: rgba(255,255,255,0.08); position: relative; margin: 6px 0 6px 0;">
<div style="width: 50%; height: 100%; border-right: 1px dashed rgba(255,255,255,0.25);"></div>
<div style="position: absolute; left: {dcr_pct_marker}%; top: -3px; width: 12px; height: 12px; border-radius: 50%; background: {'#10b981' if latest_dcr >= 0.50 else '#ef4444'}; box-shadow: 0 0 8px {'#10b981' if latest_dcr >= 0.50 else '#ef4444'};"></div>
</div>
<div style="margin-bottom: 2px;">{dcr_badge}</div>
</div>

<div style="background: rgba(0,0,0,0.28); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 10px 12px;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Up/Down Vol Ratio</span>
<strong style="font-size: 1.05rem; font-family: 'JetBrains Mono', monospace; color: {ud_color};">{ud_ratio:.2f}x</strong>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px; font-size: 0.72rem; color: #94a3b8;">
<span>Adv Vol: <strong style="color: #f8fafc; font-family: monospace;">{adv_vol_pct:.0f}%</strong></span>
<span>Smart Money: <strong style="color: #f8fafc; font-family: monospace;">{smart_money:.2f}x</strong></span>
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

        # --- COLUMN 2: GLOBAL INDEX MOMENTUM & 52W HIGH FRESHNESS (KEPT & ELEVATED) ---
        with c_col2:
            highs_html = ""
            highs_data = []
            try:
                from macro_regime_engine import fetch_index_new_highs
                highs_data = fetch_index_new_highs()
            except Exception as e:
                print(f"Error fetching index highs: {e}")

            if not highs_data:
                cache_file = os.path.join(os.path.dirname(__file__), 'index_highs_cache.json')
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            highs_data = json.load(f)
                    except Exception:
                        pass

            if highs_data:
                highs_html = "<div style='display: flex; flex-direction: column; gap: 7px; margin-bottom: 10px;'>"
                for hd in highs_data:
                    c1 = '#10b981' if hd.get('1w', 0) > 0 else '#475569'
                    c2 = '#10b981' if hd.get('1m', 0) > 0 else '#475569'
                    c3 = '#10b981' if hd.get('3m', 0) > 0 else '#475569'
                    rgb_str = _to_rgb(hd.get('color', '#94a3b8'))
                    rgb_parts = [p.strip() for p in rgb_str.split(',')]
                    bg_r = rgb_parts[0] if len(rgb_parts) > 0 else "100"
                    bg_g = rgb_parts[1] if len(rgb_parts) > 1 else "116"
                    bg_b = rgb_parts[2] if len(rgb_parts) > 2 else "139"
                    flag = '<span style="background: rgba(56,189,248,0.18); color: #38bdf8; border: 1px solid rgba(56,189,248,0.35); padding: 1px 5px; border-radius: 4px; font-size: 0.65rem; font-weight: 800; font-family: monospace;">IN</span>' if "NIFTY" in hd.get('name', '') else '<span style="background: rgba(168,85,247,0.18); color: #c084fc; border: 1px solid rgba(168,85,247,0.35); padding: 1px 5px; border-radius: 4px; font-size: 0.65rem; font-weight: 800; font-family: monospace;">US</span>'
                    
                    highs_html += f"""
<div style="display: flex; justify-content: space-between; align-items: center; background: rgba(15,23,42,0.85); border: 1px solid rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 7px;">
<div style="display: flex; align-items: center; gap: 8px;">
{flag}
<span style="font-size: 0.82rem; font-weight: 700; color: #f8fafc; width: 85px;">{hd.get('name', 'INDEX')}</span>
<span style="font-size: 0.70rem; font-weight: 800; background: rgba({bg_r},{bg_g},{bg_b},0.15); color: {hd.get('color', '#94a3b8')}; border: 1px solid rgba({bg_r},{bg_g},{bg_b},0.35); padding: 2px 7px; border-radius: 5px;">{hd.get('last_high_label', 'Recent')}</span>
</div>
<div style="font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; font-weight: 700; display: flex; gap: 6px;">
<span style="color:{c1};">{hd.get('1w', 0)}</span><span style="color:#334155;">|</span><span style="color:{c2};">{hd.get('1m', 0)}</span><span style="color:#334155;">|</span><span style="color:{c3};">{hd.get('3m', 0)}</span>
</div>
</div>
"""
                highs_html += "</div>"
            else:
                highs_html = "<div style='color: #94a3b8; font-size: 0.75rem; padding: 10px; text-align: center; background: rgba(15,23,42,0.6); border-radius: 6px;'>Benchmark feeds refreshing...</div>"

            choch_html = ""
            if choch and choch.get('detected'):
                ch_color = choch.get('color', '#f59e0b')
                ch_rgb = _to_rgb(ch_color)
                choch_html = f"""
<div style="background: rgba({ch_rgb}, 0.12); border-left: 3px solid {ch_color}; padding: 6px 10px; border-radius: 4px; margin-top: 8px;">
<p style="margin: 0; font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;">Wyckoff Alert ({choch.get('date', 'Recent')})</p>
<p style="margin: 0; color: {ch_color}; font-weight: bold; font-size: 0.85rem;">{choch.get('label')}</p>
</div>
"""

            st.markdown(f"""
<div class="metric-card" style="border-top: 4px solid #3b82f6; height: 100%; display: flex; flex-direction: column; justify-content: space-between; padding: 18px 20px;">
<div>
<div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
<p style="color: #94a3b8; font-size: 0.80rem; margin: 0; text-transform: uppercase; font-weight: 800; letter-spacing: 1px;">Global Momentum</p>
<span style="font-size: 0.68rem; color: #64748b; font-weight: 700; text-transform: uppercase;">1W | 1M | 3M NEW HIGHS</span>
</div>
<h3 style="color: #60a5fa; margin: 0.2rem 0 0.8rem 0; font-size: 1.35rem; font-weight: 800; text-align: left;">52W High Freshness</h3>

<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.68rem; color: #64748b; font-weight: 700; text-transform: uppercase; padding: 0 10px 4px 10px; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 6px;">
<span>Benchmark Index</span>
<span>1W · 1M · 3M Highs</span>
</div>

{highs_html}

<div style="background: rgba(15,23,42,0.65); border: 1px solid rgba(245,158,11,0.25); border-left: 3px solid #f59e0b; padding: 8px 12px; border-radius: 6px; text-align: left; margin-top: 8px;">
<p style="margin: 0; font-size: 0.72rem; color: #cbd5e1; line-height: 1.4;">
<b style="color: #fbbf24;">💡 Divergence Insight:</b> Indian benchmarks are digesting 6-month overhead supply (zero 52W highs since April), while US tech/broad benchmarks are printing fresh blue-sky highs.
</p>
</div>
{choch_html}
</div>
</div>
""", unsafe_allow_html=True)

        # --- COLUMN 3: LEADER BREADTH & CONFIRMED THEMES ---
        with c_col3:
            rs21_color = '#10b981' if top100_21_ema >= 70 else '#f59e0b' if top100_21_ema >= 50 else '#ef4444'
            rs50_color = '#10b981' if top100_50_sma >= 70 else '#f59e0b' if top100_50_sma >= 50 else '#ef4444'
            broad50_color = '#10b981' if broad_50_sma >= 50 else '#ef4444' if broad_50_sma < 30 else '#f59e0b'
            grade_color = '#10b981' if lead_grade == 'A' else '#f59e0b' if lead_grade == 'B' else '#ef4444'

            # Sector Clusters with Group Confirmation (Polished 2-line layout prevents wrapping)
            sector_df_data = macro_res.get('sector_df') if macro_res else None
            clusters = get_sector_clusters(sector_df_data, top_n=3)
            cluster_html = ""
            for i, c in enumerate(clusters):
                conf_badge = c.get('confirmation', '')
                leaders_cnt = c.get('leaders_count', 0)
                if not conf_badge or conf_badge == '⚪ Unconfirmed':
                    if leaders_cnt >= 3:
                        conf_badge = f"✅ Confirmed ({int(leaders_cnt)} Leaders)"
                    else:
                        conf_badge = f"⚪ {int(leaders_cnt)} Leaders"
                
                is_conf = "Confirmed" in conf_badge or "✅" in conf_badge
                badge_color = "#38bdf8" if is_conf else "#94a3b8"
                badge_bg = "rgba(56,189,248,0.15)" if is_conf else "rgba(148,163,184,0.1)"
                badge_border = "rgba(56,189,248,0.35)" if is_conf else "rgba(148,163,184,0.2)"
                
                badge_str = f"<span style='font-size: 0.68rem; background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_border}; padding: 2px 7px; border-radius: 4px; font-weight: 700; white-space: nowrap;'>{conf_badge}</span>"
                cluster_html += f"""
<div style="background: rgba(15,23,42,0.6); border: 1px solid rgba(255,255,255,0.04); padding: 7px 10px; border-radius: 6px; margin-bottom: 5px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
<span style="color: #f8fafc; font-weight: 700; font-size: 0.82rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 190px;">{i+1}. {c['name']}</span>
<span style="color: #10b981; font-size: 0.80rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;">{c['bullish_pct']:.0f}% Bullish</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center;">
{badge_str}
<span style="font-size: 0.68rem; color: #64748b;">Avg RS: <strong style="color: #cbd5e1;">{c['avg_rs']:.0f}</strong></span>
</div>
</div>
"""

            st.markdown(f"""
<div class="metric-card" style="border-top: 4px solid #8b5cf6; height: 100%; display: flex; flex-direction: column; justify-content: space-between; padding: 18px 20px;">
<div>
<div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
<p style="color: #94a3b8; font-size: 0.80rem; margin: 0; text-transform: uppercase; font-weight: 800; letter-spacing: 1px;">Super-Leader Health</p>
<span style="font-size: 0.68rem; color: #64748b; font-weight: 700; text-transform: uppercase;">TOP 100 BASKET</span>
</div>
<h3 style="color: #a78bfa; margin: 0.2rem 0 0.8rem 0; font-size: 1.35rem; font-weight: 800; text-align: left;">Breadth & Leading Themes</h3>

<div style="background: rgba(0,0,0,0.28); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Top 100 RS > 21 EMA (Pullbacks)</span>
<strong style="font-size: 1.05rem; font-family: 'JetBrains Mono', monospace; color: {rs21_color};">{top100_21_ema:.0f}%</strong>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Top 100 RS > 50 SMA (Structure)</span>
<strong style="font-size: 1.05rem; font-family: 'JetBrains Mono', monospace; color: {rs50_color};">{top100_50_sma:.0f}%</strong>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 4px; padding-top: 4px;">
<span style="font-size: 0.70rem; color: #94a3b8; text-transform: uppercase;">Broad Market > 50 SMA</span>
<strong style="font-size: 0.88rem; font-family: 'JetBrains Mono', monospace; color: {broad50_color};">{broad_50_sma:.0f}%</strong>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 2px;">
<span style="font-size: 0.70rem; color: #64748b; text-transform: uppercase;">Leadership Spread vs Market</span>
<strong style="font-size: 0.95rem; font-family: 'JetBrains Mono', monospace; color: #10b981;">+{rs_spread:.0f}%</strong>
</div>
<div style="margin-top: 6px; background: rgba(0,0,0,0.4); border-radius: 4px; height: 5px; overflow: hidden; position: relative;">
<div style="width: 100%; height: 100%; background: rgba(239, 68, 68, 0.45);"></div>
<div style="position: absolute; left: 0; top: 0; width: {int(top100_50_sma)}%; height: 100%; background: linear-gradient(90deg, #10b981, #34d399); border-radius: 4px;"></div>
</div>
<div style="display: flex; justify-content: space-between; font-size: 0.65rem; color: #64748b; margin-top: 3px;">
<span>Leaders ({top100_50_sma:.0f}%)</span>
<span>Broad Market ({broad_50_sma:.0f}%)</span>
</div>
</div>

<div style="background: rgba(0,0,0,0.28); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Leadership Quality</span>
<span style="background: rgba(139,92,246,0.18); color: {grade_color}; border: 1px solid rgba(139,92,246,0.35); padding: 2px 8px; border-radius: 4px; font-weight: 800; font-size: 0.75rem;">Grade {lead_grade}</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px; font-size: 0.72rem; color: #cbd5e1;">
<span>Near 52W High: <strong style="color: #f8fafc; font-family: monospace;">{near_high_pct:.0f}%</strong></span>
<span style="color: #94a3b8;">FTD: <strong style="color: {'#10b981' if ftd_detected else '#94a3b8'};">{ftd_status}</strong></span>
</div>
</div>

<div style="background: rgba(0,0,0,0.28); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 8px 10px;">
<div style="color: #94a3b8; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">Confirmed Leading Themes (O'Neil)</div>
{cluster_html if cluster_html else "<p style='color: #64748b; font-size: 0.8rem; margin: 0;'>Gathering themes from 750-stock universe...</p>"}
</div>
</div>
</div>
""", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # TIER 3: SAFE HARBOR CAPITAL ROTATION (WORLD-CLASS HEDGE MATRIX)
        # ---------------------------------------------------------------------
        if 'Correction' in regime or 'Pressure' in regime or (n500_telem and 'Correction' in n500_telem.get('status_label', '')):
            import importlib
            import safe_harbor_engine
            importlib.reload(safe_harbor_engine)
            safe_assets = safe_harbor_engine.evaluate_safe_harbor(cache_buster=3)
            if safe_assets:
                asset_cards_html = ""
                for asset in safe_assets:
                    st_val = asset.get('Status', 'Wait')
                    if "Rotation" in st_val or "Strong" in st_val or "Excellent" in st_val:
                        bg_c = "rgba(16, 185, 129, 0.15)"
                        bdr_c = "rgba(16, 185, 129, 0.4)"
                        txt_c = "#34d399"
                    elif "Hold" in st_val or "Accumulating" in st_val:
                        bg_c = "rgba(245, 158, 11, 0.15)"
                        bdr_c = "rgba(245, 158, 11, 0.4)"
                        txt_c = "#fbbf24"
                    else:
                        bg_c = "rgba(100, 116, 139, 0.15)"
                        bdr_c = "rgba(100, 116, 139, 0.3)"
                        txt_c = "#94a3b8"
                        
                    ret1m = asset.get('1M_Return', 0.0)
                    if ret1m is None or pd.isna(ret1m) or (isinstance(ret1m, (int, float)) and math.isnan(ret1m)):
                        ret1m_str = "0.0%"
                        ret1m_color = "#94a3b8"
                    else:
                        ret1m_str = f"{ret1m:+.1f}%"
                        ret1m_color = "#10b981" if ret1m > 0 else "#ef4444"

                    rel_rs = asset.get('Rel_Strength_Score', 0.0)
                    if rel_rs is None or pd.isna(rel_rs) or (isinstance(rel_rs, (int, float)) and math.isnan(rel_rs)):
                        rel_rs_str = "+0.0"
                    else:
                        rel_rs_str = f"{rel_rs:+.1f}"
                    
                    asset_cards_html += f"""<div style="flex: 1 1 200px; min-width: 200px; background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 10px 14px; display: flex; flex-direction: column; justify-content: space-between;">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
<div>
<span style="font-size: 0.85rem; font-weight: 800; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{asset['Ticker']}</span>
<div style="font-size: 0.70rem; color: #94a3b8; margin-top: 1px;">{asset['Asset']}</div>
</div>
<span style="font-size: 0.82rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: {ret1m_color};">{ret1m_str} 1M</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(255,255,255,0.04); padding-top: 6px; margin-top: 4px;">
<span style="font-size: 0.68rem; background: {bg_c}; color: {txt_c}; border: 1px solid {bdr_c}; padding: 2px 7px; border-radius: 4px; font-weight: 700; white-space: nowrap;">{st_val}</span>
<span style="font-size: 0.68rem; color: #64748b;">Rel RS: <b style="color: #cbd5e1;">{rel_rs_str}</b></span>
</div>
</div>"""

                st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.8)); border: 1px solid rgba(245, 158, 11, 0.35); border-left: 5px solid #f59e0b; padding: 16px 20px; border-radius: 12px; margin-top: 14px; margin-bottom: 14px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
<div style="display: flex; align-items: center; gap: 10px;">
<span style="font-size: 1.3rem;">🛡️</span>
<div>
<strong style="color: #fbbf24; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.6px;">Safe Harbor Capital Rotation</strong>
<span style="font-size: 0.78rem; color: #94a3b8; margin-left: 8px;">Non-correlated asset allocation during active correction</span>
</div>
</div>
<span style="background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.45); padding: 3px 12px; border-radius: 6px; font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.6px;">DEFENSIVE SHIELD ACTIVE</span>
</div>
<div style="display: flex; gap: 10px; flex-wrap: wrap;">
{asset_cards_html}
</div>
</div>
""", unsafe_allow_html=True)

        # Scraper / Engine Actions
        with st.expander("🐺 StockScans Engine Actions & Setup"):
            sc_c1, sc_c2 = st.columns(2)
            app_dir = os.path.dirname(os.path.abspath(__file__))
            scraper_path = os.path.join(app_dir, "stockscans_scraper.py")
            with sc_c1:
                if st.button("⚙️ Setup Login (One-Time)", use_container_width=True):
                    if sys.platform == "win32":
                        subprocess.Popen(f'cmd.exe /k python "{scraper_path}" --setup', cwd=app_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
                    else:
                        subprocess.Popen([sys.executable, scraper_path, "--setup"], cwd=app_dir)
                    st.toast("Opening visible browser for setup...", icon="⚙️")
            with sc_c2:
                if st.button("🚀 Run Confluence Engine", use_container_width=True, type="primary"):
                    if sys.platform == "win32":
                        subprocess.Popen(f'cmd.exe /c python "{scraper_path}"', cwd=app_dir, creationflags=0x08000000)
                    else:
                        subprocess.Popen([sys.executable, scraper_path], cwd=app_dir)
                    st.toast("Scraper launched! Wait 30 seconds for the popup.", icon="🚀")

        # Deep Indicators Expander
        with st.expander("📚 Institutional Playbook: How to Read the Market Commander Cockpit"):
            st.markdown(r"""
**The Institutional Market Commander Cockpit** consolidates the rules and telemetry of legendary market operators: **Ross Haber**, **William O'Neil**, and **Mark Minervini**.

1. **Master Progressive Exposure Decks**:
   - **🔥 Hot Deck (100–120% Exposure · 15–20% Pilots)**: Benchmark above 21 SMA + $\le 2$ Distribution Days. Aggressive trend acceleration mode.
   - **🌤️ Warming Deck (50–75% Exposure · 8–12% Pilots)**: Confirmed FTD with Grade A leadership or 3–4 Distribution Days. Test market with disciplined pilots.
   - **❄️ Cold Deck (25–40% Exposure · 5% Pilots)**: Uptrend under pressure (5–6 Distribution Days) or below 21 SMA. Capital preservation, raise stops.
   - **🧊 Frozen Deck (0–10% Exposure · 0–2% Pilots)**: Correction regime, $\ge 7$ Distribution Days, or Haber 21-day SMA double close broken. Cash is king.

2. **Acute Distribution Cluster Warning (Haber Rule)**:
   - When **$\ge 3$ Distribution Days occur within 5 to 7 sessions**, institutions are actively liquidating holdings into every bounce. Halt buying immediately.

3. **Daily Closing Range (DCR) Midpoint Rule**:
   - $\\text{DCR} = \\frac{\\text{Close} - \\text{Low}}{\\text{High} - \\text{Low}}$
   - **Upper Half ($\ge 0.50$)**: Institutional Accumulation. Even if the index finishes red, closing near the high proves buyers absorbed supply.
   - **Lower Half ($< 0.50$)**: Distribution / Stalling. The index closed under seller control.

4. **Global 52-Week High Freshness Velocity**:
   - Compares the recency of 52-week highs across NIFTY 50, NIFTY 500, NASDAQ 100, and S&P 500.
   - Highlights whether domestic markets are digesting overhead supply or leading international breakouts into blue-sky territory.

5. **Super-Leader Breadth & O'Neil Pack-Hunting**:
   - Tracks the percentage of top momentum leaders holding above their 21 EMA (short-term pullbacks) and 50 SMA (institutional backbone).
   - Only buy stocks belonging to **Confirmed Leading Themes** where at least 2 other peers are printing fresh multi-month highs.
            """)

        st.markdown("---")

    # =========================================================================
    # SECTOR ROTATION ALERTS (DB-backed history)
    # =========================================================================
    if macro_res:
        sector_df = macro_res.get('sector_df')
        if sector_df is not None and not sector_df.empty:
            # Current top 5 by Avg RS
            sorted_sectors = sector_df.sort_values('Avg_RS', ascending=False).head(5)
            current_top5 = sorted_sectors[['Industry', 'Avg_RS']].to_dict('records') if 'Industry' in sorted_sectors.columns else []

            if not current_top5:
                # Try alternate column names
                if 'Sector' in sorted_sectors.columns:
                    sorted_sectors = sorted_sectors.rename(columns={'Sector': 'Industry'})
                    current_top5 = sorted_sectors[['Industry', 'Avg_RS']].to_dict('records')

            if current_top5:
                # Save today's leadership to DB
                save_sector_leadership(current_top5)

                # Get historical data
                history = get_sector_leadership_history(days=30)
                current_names = [r['Industry'] for r in current_top5]

                # Find the most recent PREVIOUS date's top 5
                today_str = datetime.now().strftime('%Y-%m-%d')
                prev_dates = sorted(set(h['date'] for h in history if h['date'] < today_str), reverse=True)

                if prev_dates:
                    last_date = prev_dates[0]
                    prev_top5 = [h['industry'] for h in history if h['date'] == last_date]

                    new_entrants = [g for g in current_names if g not in prev_top5]
                    dropouts = [g for g in prev_top5 if g not in current_names]

                    if new_entrants or dropouts:

                        
                        st.markdown("### 🔄 Sector Rotation Alert")
                        st.caption(f"Comparing today's Top 5 industry groups vs last scan ({last_date})")

                        rot_col1, rot_col2 = st.columns(2)
                        with rot_col1:
                            if new_entrants:
                                for g in new_entrants:
                                    st.success(f"🟢 **NEW LEADER:** {g} has entered the Top 5. O'Neil says: *Focus new buys in leading groups.*")
                            else:
                                st.info("No new groups entered the Top 5.")

                        with rot_col2:
                            if dropouts:
                                for g in dropouts:
                                    st.warning(f"🟡 **DROPPED OUT:** {g} has fallen from Top 5. Tighten stops on positions in this group.")
                            else:
                                st.info("No groups dropped out of the Top 5.")

                        # Show historical leadership table
                        with st.expander("📊 Sector Leadership History (Last 30 Days)"):
                            if history:
                                hist_df = pd.DataFrame(history)
                                pivot = hist_df.pivot_table(index='date', columns='rank', values='industry', aggfunc='first')
                                pivot.columns = [f'Rank {c}' for c in pivot.columns]
                                pivot = pivot.sort_index(ascending=False)
                                st.dataframe(pivot, use_container_width=True)
                    else:
                        st.info("✅ **Sector Leadership Stable** — The same Top 5 industry groups are leading since the last scan.")

                    st.markdown("---")
                else:
                    # --- BACKFILL: Reconstruct T-21 leadership from RS Momentum column ---
                    # sector_df has 'RS Momentum 21d' = (Avg_RS_now - Avg_RS_21d_ago)
                    # So Avg_RS_21d_ago = Avg_RS - RS_Momentum_21d
                    backfill_done = False
                    if 'RS Momentum 21d' in sector_df.columns:
                        backfill_df = sector_df.copy()
                        backfill_df['Avg_RS_T21'] = backfill_df['Avg_RS'] - backfill_df['RS Momentum 21d']
                        backfill_sorted = backfill_df.sort_values('Avg_RS_T21', ascending=False).head(5)

                        if 'Industry' in backfill_sorted.columns:
                            backfill_date = (datetime.now() - timedelta(days=21)).strftime('%Y-%m-%d')
                            backfill_top5 = backfill_sorted[['Industry', 'Avg_RS_T21']].rename(
                                columns={'Avg_RS_T21': 'Avg_RS'}
                            ).to_dict('records')

                            # Save the reconstructed T-21 snapshot
                            from database import get_connection
                            conn = get_connection()
                            cursor = conn.cursor()
                            for i, ind in enumerate(backfill_top5):
                                cursor.execute('''
                                    INSERT OR REPLACE INTO sector_leadership_history
                                    (date, rank, industry, avg_rs)
                                    VALUES (?, ?, ?, ?)
                                ''', (backfill_date, i + 1, ind['Industry'], ind['Avg_RS']))
                            conn.commit()
                            conn.close()

                            prev_top5_names = [r['Industry'] for r in backfill_top5]
                            new_entrants = [g for g in current_names if g not in prev_top5_names]
                            dropouts = [g for g in prev_top5_names if g not in current_names]

                            if new_entrants or dropouts:
                                st.markdown("### 🔄 Sector Rotation Alert (vs ~21 Days Ago)")
                                st.caption(f"Backfilled comparison: today's Top 5 vs reconstructed T-21 leadership ({backfill_date})")

                                rot_col1, rot_col2 = st.columns(2)
                                with rot_col1:
                                    if new_entrants:
                                        for g in new_entrants:
                                            st.success(f"🟢 **NEW LEADER:** {g} has entered the Top 5 industry groups.")
                                    else:
                                        st.info("No new groups entered the Top 5.")
                                with rot_col2:
                                    if dropouts:
                                        for g in dropouts:
                                            st.warning(f"🟡 **DROPPED OUT:** {g} was in the Top 5 ~21 days ago but has fallen out.")
                                    else:
                                        st.info("No groups dropped out of the Top 5.")
                            else:
                                st.info("✅ **Sector Leadership Stable** — The same Top 5 industry groups have been leading for the past ~21 days.")

                            backfill_done = True

                    if not backfill_done:
                        st.info("📊 Sector leadership tracking initialized. Rotation alerts will appear after the next scan.")
                    st.markdown("---")
        
    # =========================================================================
    # PORTFOLIO ACTION DASHBOARD
    # =========================================================================
    
    # Calculate decisions first to aggregate for the cockpit
    decisions = []
    audit_warnings = []
    
    # Needs a quick pre-loop to extract sparklines and base decisions before the full dashboard
    for _, row in portfolio_df.iterrows():
        ticker = row['ticker']
        quantity = row['Qty']
        avg_buy_price = row['Avg Buy Price']
        
        df = market_data.get(ticker)
        if df is None or df.empty:
            continue
            
        if len(df) < 200:
            audit_warnings.append(f"⚠️ **{ticker}**: Only {len(df)} days history fetched. 200+ recommended for precise EMA calculations.")
        
        current_price = df['close'].iloc[-1]
        if pd.isna(current_price) or math.isnan(current_price):
            current_price = avg_buy_price
            
        price_1m_ago = df['close'].iloc[-21] if len(df) > 21 else df['close'].iloc[0]
        price_3m_ago = df['close'].iloc[-63] if len(df) > 63 else df['close'].iloc[0]
        ret_1m = ((current_price / price_1m_ago) - 1) * 100 if price_1m_ago > 0 else 0.0
        ret_3m = ((current_price / price_3m_ago) - 1) * 100 if price_3m_ago > 0 else 0.0
            
        rs_info = rs_data.get(ticker, {})
        rs_score = rs_info.get('rs_score')
        
        decision = make_decision(
            df=df,
            ticker=ticker,
            current_price=current_price,
            avg_buy_price=avg_buy_price,
            quantity=quantity,
            total_portfolio_value=total_value,
            rs_score=rs_score,
        )
        decision['ret_1m'] = ret_1m
        decision['ret_3m'] = ret_3m
        
        trend_details = get_trend_details(df)
        decision.update(trend_details)
        decision['close_price'] = current_price
        decision['avg_buy_price'] = avg_buy_price
        decision['quantity'] = quantity
        
        peak_since_buy = row.get('Peak Since Buy', current_price)
        if pd.isna(peak_since_buy): peak_since_buy = current_price
        
        # Sanity check: cap peak_since_buy to split-adjusted historical max to prevent split anomalies
        hist_max = df['high'].max() if 'high' in df.columns else df['close'].max()
        if peak_since_buy > hist_max:
            peak_since_buy = hist_max
            
        port_hit_pct = 0.0
        if total_value > 0 and peak_since_buy > current_price:
            port_hit_pct = ((peak_since_buy - current_price) * quantity) / total_value * 100.0
        decision['port_hit_pct'] = port_hit_pct
        
        # Institutional Stop Defense & Open Risk (Heat) Calculation
        from technical_indicators import calculate_atr_chandelier_stop
        hard_base_stop = avg_buy_price * 0.92  # O'Neil 8% Hard Stop floor
        gain_loss_pct = ((current_price / avg_buy_price) - 1.0) * 100.0 if avg_buy_price > 0 else 0.0
        cost_defense_stop = avg_buy_price if gain_loss_pct >= 8.0 else hard_base_stop
        chand_stop = calculate_atr_chandelier_stop(df, atr_multiplier=3.5, high_window=20, atr_period=21)
        
        active_stop = max(hard_base_stop, cost_defense_stop, chand_stop if chand_stop > 0 else hard_base_stop)
        stop_breached = current_price < active_stop
        
        # Proper institutional risk per share:
        # 1. If stock is above active stop: risk is distance down to active stop (current_price - active_stop)
        # 2. If stock has breached active stop: stop order has already triggered (Exit pending).
        #    Open risk to market exit is execution gap risk (1 ATR, capped at 8% maximum stop floor),
        #    NOT 100% of the stock price!
        if not stop_breached:
            risk_per_share = current_price - active_stop
        else:
            latest_atr = df['atr_14'].iloc[-1] if ('atr_14' in df.columns and not pd.isna(df['atr_14'].iloc[-1])) else (current_price * 0.03)
            risk_per_share = min(current_price * 0.08, float(latest_atr))
            
        dollar_risk = risk_per_share * quantity
        position_heat_pct = (dollar_risk / total_value * 100.0) if total_value > 0 else 0.0
        
        decision['active_stop'] = active_stop
        decision['stop_breached'] = stop_breached
        decision['dollar_risk'] = dollar_risk
        decision['position_heat_pct'] = position_heat_pct
        
        # Add new metrics
        last_row = df.iloc[-1]
        decision['rel_vol'] = last_row.get('rel_vol', 0)
        decision['rs_blue_dot'] = bool(last_row.get('rs_blue_dot', False))
        
        from technical_indicators import calculate_power_days
        p_days, d_days = calculate_power_days(df, 65, 4.0)
        decision['power_days_3m'] = p_days
        decision['dist_days_3m'] = d_days
        decision['adr_pct'] = last_row.get('adr_pct_20', 0)
        decision['sparkline'] = df['close'].tail(60).tolist()
        
        # Risk Metrics: Dual-Horizon Sortino Ratio & Smooth RS (Composite RS / 20D ATR%)
        decision['sortino_3m'] = calculate_sortino_ratio(df['close'], window=63, mar_annual=0.065)
        decision['sortino_6m'] = calculate_sortino_ratio(df['close'], window=126, mar_annual=0.065)
        decision['smooth_rs'] = calculate_smooth_rs_atr(
            df['close'],
            df['high'] if 'high' in df.columns else None,
            df['low'] if 'low' in df.columns else None
        )
        
        # Institutional Stock Distribution (Pillars 1-4)
        try:
            from stock_distribution_engine import analyze_stock_distribution
            decision['distribution_metrics'] = analyze_stock_distribution(df, lookback_cluster=15, lookback_ud=50)
        except Exception as e:
            decision['distribution_metrics'] = {
                'idvd_count_15d': 0, 'churning_count_15d': 0, 'total_dist_days_15d': 0,
                'cluster_detected': False, 'ud_volume_ratio_50d': 1.0,
                'consecutive_closes_below_21': 0, 'haber_21_breach': False,
                'oneil_50_breach': False, 'ids_score': 0, 'status': '🟢 Clean',
                'badge_label': '🟢 0 DD (1.0x)',
                'prescription': 'Hold core position; normal action.',
                'dist_sessions': []
            }
        
        if len(df) >= 20:
            avg_vol = df['volume'].tail(20).mean()
            avg_price = df['close'].tail(20).mean()
            decision['adtv_cr'] = (avg_vol * avg_price) / 10000000.0
        else:
            decision['adtv_cr'] = 1.0
            
        # Calculate Days to Liquidate (DTL) using 10% Max Participation
        position_val_raw = quantity * current_price
        max_daily_volume_rs = (decision['adtv_cr'] * 10000000.0) * 0.10
        decision['days_to_exit'] = position_val_raw / max_daily_volume_rs if max_daily_volume_rs > 0 else 999.0
        
        # Mike Webster Signals
        if 'BENCHMARK' in market_data and not market_data['BENCHMARK'].empty:
            web_sig = calculate_webster_sell_signals(df, market_data['BENCHMARK'])
            decision['webster_signal'] = web_sig['signal']
        else:
            decision['webster_signal'] = '⚪ Unknown'
        
        # Fetch sector, industry & canonical theme with deterministic 3-tier classification
        from theme_contagion_engine import get_deterministic_stock_industry, map_sub_industry_to_theme
        _, _, det_industry = get_deterministic_stock_industry(ticker)
        sector, yf_ind = get_yfinance_sector_and_industry(ticker)
        final_industry = det_industry if det_industry and det_industry not in ('Other / ETF', 'Unknown', 'Unclassified / Micro-Cap') else yf_ind
        decision['sector'] = sector
        decision['industry'] = final_industry
        decision['theme'] = map_sub_industry_to_theme(final_industry)

        # Attach quarterly earnings info (Part 1 & Gap 6)
        e_info = earnings_cache.get(ticker.replace('.NS', '').replace('.BO', '').strip(), {})
        if e_info and e_info.get('latest_quarter') != 'N/A':
            eps_val = e_info.get('eps_yoy_pct')
            sales_val = e_info.get('sales_yoy_pct')
            e_str = f"EPS +{eps_val:.0f}%" if eps_val and eps_val > 0 else f"EPS {eps_val:.0f}%" if eps_val is not None else "N/A"
            s_str = f"Sales +{sales_val:.0f}%" if sales_val and sales_val > 0 else f"Sales {sales_val:.0f}%" if sales_val is not None else ""
            
            badge_icon = "🟢" if e_info.get('is_new') else "⚪"
            decision['q_results_badge'] = f"{badge_icon} {e_info.get('latest_quarter')}: {e_str} | {s_str}"
            
            days_left = e_info.get('days_to_earnings')
            if days_left is not None and 0 <= days_left <= 5:
                decision['binary_risk'] = f"⚠️ Due in {days_left}d"
            else:
                decision['binary_risk'] = "OK"
        else:
            decision['q_results_badge'] = "⚪ N/A"
            decision['binary_risk'] = "OK"

        decision['industry'] = final_industry
        
        decisions.append(decision)
        
        
    # Aggregate Metrics for Cockpit
    exit_count = sum(1 for d in decisions if d['action'] == ACTION_EXIT)
    trim_count = sum(1 for d in decisions if d['action'] == ACTION_TRIM)
    hold_count = sum(1 for d in decisions if d['action'] == ACTION_HOLD)
    add_count = sum(1 for d in decisions if d['action'] == ACTION_ADD)
    
    invested_value = sum(d['close_price'] * d['quantity'] for d in decisions)
    liquid_funds_value = sum(d['close_price'] * d['quantity'] for d in decisions if str(d['ticker']).upper() in ['LIQUIDBEES', 'LIQUIDCASE'])
    equity_value = invested_value - liquid_funds_value
    risk_exposure = (equity_value / total_value * 100) if total_value > 0 else 0
    total_open_heat_dollars = sum(d.get('dollar_risk', 0.0) for d in decisions)
    total_open_heat_pct = (total_open_heat_dollars / total_value * 100.0) if total_value > 0 else 0.0

    # ---------------------------------------------------------
    # TELEGRAM PORTFOLIO HEALTH BRIEFING (TIER 2 DAILY LAUNCH GUARD)
    # ---------------------------------------------------------
    from portfolio_telegram_notifier import (
        is_telegram_configured,
        has_sent_today,
        mark_sent_today,
        send_telegram_portfolio_message,
        get_telegram_credentials
    )
    from portfolio_health_engine import (
        calculate_portfolio_health_summary,
        format_telegram_portfolio_briefing
    )

    current_regime_label = f"{r_emoji} {regime} ({dd_count} DD)" if ('regime' in locals() and 'r_emoji' in locals()) else "Active Momentum Scan"
    st.session_state['portfolio_health_summary'] = calculate_portfolio_health_summary(
        decisions=decisions,
        total_value=total_value,
        market_regime_label=current_regime_label
    )

    # Automatically dispatch once per day on first launch (weekdays & weekends)
    _, _, auto_alert_enabled = get_telegram_credentials()
    if is_telegram_configured() and auto_alert_enabled:
        if not has_sent_today():
            briefing_text = format_telegram_portfolio_briefing(st.session_state['portfolio_health_summary'])
            sent_ok, sent_msg = send_telegram_portfolio_message(briefing_text)
            if sent_ok:
                mark_sent_today()
                st.toast("📲 Daily Portfolio Briefing sent to Telegram!", icon="🚀")

    # Market Direction HUD
    # ---------------------------------------------------------
    # Shared Helper: RS Line New High (RSNH) Leaders
    # ---------------------------------------------------------
    @st.cache_data(ttl=600, show_spinner=False)
    def fetch_rsnh_leaders(market='INDIA', limit=15):
        from database import get_connection
        import yfinance as yf
        try:
            conn = get_connection()
            c = conn.cursor()
            c.execute("SELECT MAX(date) FROM tml_snapshot WHERE market = ?", (market,))
            row = c.fetchone()
            if not row or not row[0]:
                conn.close()
                return []
            latest_date = row[0]
            
            query = """
                SELECT ticker, rs_score, industry, tml_score 
                FROM tml_snapshot 
                WHERE market = ? AND date = ? AND rs_score IS NOT NULL
                ORDER BY rs_score DESC
            """
            df_candidates = pd.read_sql_query(query, conn, params=[market, latest_date])
            conn.close()
            
            if df_candidates.empty:
                return []
                
            tickers = df_candidates['ticker'].tolist()
            bench_ticker = '^CRSLDX' if market == 'INDIA' else '^GSPC'
            
            all_tickers = [bench_ticker] + tickers
            batch = yf.download(all_tickers, period='1y', group_by='ticker', progress=False, threads=True)
            
            if batch.empty:
                return []
                
            if isinstance(batch.columns, pd.MultiIndex):
                if 'Ticker' in batch.columns.names and bench_ticker in batch.columns.get_level_values('Ticker'):
                    bench_close = batch.xs(bench_ticker, axis=1, level='Ticker')['Close'].dropna()
                elif bench_ticker in batch.columns.get_level_values(0):
                    bench_close = batch.xs(bench_ticker, axis=1, level=0)['Close'].dropna()
                elif bench_ticker in batch.columns.get_level_values(1):
                    bench_close = batch.xs(bench_ticker, axis=1, level=1)['Close'].dropna()
                else:
                    bench_close = pd.Series()
            else:
                bench_close = batch['Close'].dropna() if 'Close' in batch else pd.Series()
                
            rsnh_list = []
            for _, row_c in df_candidates.iterrows():
                tckr = row_c['ticker']
                try:
                    s_high = None
                    s_low = None
                    if isinstance(batch.columns, pd.MultiIndex):
                        if 'Ticker' in batch.columns.names and tckr in batch.columns.get_level_values('Ticker'):
                            sub_b = batch.xs(tckr, axis=1, level='Ticker')
                            s_close = sub_b['Close'].dropna()
                            s_high = sub_b['High'].dropna() if 'High' in sub_b else None
                            s_low = sub_b['Low'].dropna() if 'Low' in sub_b else None
                        elif tckr in batch.columns.get_level_values(0):
                            sub_b = batch.xs(tckr, axis=1, level=0)
                            s_close = sub_b['Close'].dropna()
                            s_high = sub_b['High'].dropna() if 'High' in sub_b else None
                            s_low = sub_b['Low'].dropna() if 'Low' in sub_b else None
                        elif tckr in batch.columns.get_level_values(1):
                            sub_b = batch.xs(tckr, axis=1, level=1)
                            s_close = sub_b['Close'].dropna()
                            s_high = sub_b['High'].dropna() if 'High' in sub_b else None
                            s_low = sub_b['Low'].dropna() if 'Low' in sub_b else None
                        else:
                            continue
                    else:
                        s_close = batch['Close'].dropna() if 'Close' in batch else pd.Series()
                        s_high = batch['High'].dropna() if 'High' in batch else None
                        s_low = batch['Low'].dropna() if 'Low' in batch else None
                        
                    if len(s_close) < 50:
                        continue
                        
                    aligned = pd.DataFrame({'stock': s_close, 'bench': bench_close}).dropna()
                    if len(aligned) < 50:
                        continue
                        
                    rs_line = aligned['stock'] / aligned['bench']
                    cur_rs = rs_line.iloc[-1]
                    max_rs_52w = rs_line.tail(252).max()
                    
                    cur_price = aligned['stock'].iloc[-1]
                    max_price_52w = aligned['stock'].tail(252).max()
                    
                    if max_price_52w <= 0 or max_rs_52w <= 0:
                        continue
                        
                    dist_from_high_pct = ((max_price_52w - cur_price) / max_price_52w) * 100.0
                    
                    # Option A (True Divergence): RS Line at 52W High (within 2%), Price strictly COILING in base (1% to 15% below high)
                    if cur_rs >= (0.980 * max_rs_52w) and (1.0 <= dist_from_high_pct <= 15.0):
                        # Stage 2 check: price above 50 SMA
                        if len(s_close) >= 50 and cur_price < s_close.tail(50).mean():
                            continue
                            
                        status = "🎯 In Buy Zone" if dist_from_high_pct <= 5.0 else "⏳ Forming Base"
                        clean_tckr = tckr.replace('.NS', '').replace('.BO', '')
                        tv_url = f"https://in.tradingview.com/chart/?symbol=NSE:{clean_tckr}" if market == 'INDIA' else f"https://www.tradingview.com/chart/?symbol={clean_tckr}"
                        
                        from risk_metrics import calculate_sortino_ratio, calculate_smooth_rs_atr
                        s3m = calculate_sortino_ratio(s_close, window=63, mar_annual=0.065)
                        s6m = calculate_sortino_ratio(s_close, window=126, mar_annual=0.065)
                        sm_rs = calculate_smooth_rs_atr(s_close, s_high, s_low)
                        
                        rsnh_list.append({
                            'Ticker': clean_tckr,
                            'industry': row_c['industry'],
                            'TradingView': tv_url,
                            'rs_score': float(row_c['rs_score']),
                            'smooth_rs': sm_rs,
                            'sortino_3m': s3m,
                            'sortino_6m': s6m,
                            'dist_52w': f"-{dist_from_high_pct:.1f}%",
                            'status': status,
                            'tml_score': float(row_c['tml_score']) if pd.notna(row_c['tml_score']) else 0.0
                        })
                except Exception:
                    continue
                    
            if not rsnh_list:
                return []
                
            df_rsnh = pd.DataFrame(rsnh_list).sort_values('rs_score', ascending=False).head(limit).reset_index(drop=True)
            df_rsnh['Rank'] = df_rsnh.index + 1
            return df_rsnh.to_dict(orient='records')
        except Exception as e:
            print(f"RSNH Fetch Error: {e}")
            return []

    # ---------------------------------------------------------
    # NEW: Top RS Leaders (TraderLion / Deepvue Inspired)
    # ---------------------------------------------------------
    top_15_rs_leaders = get_top_rs_leaders(market='INDIA', limit=15)
    top_rs_leaders = top_15_rs_leaders[:5] if top_15_rs_leaders else []
    
    if top_rs_leaders:
        st.markdown("### 🏆 Deep Market Leaders (Top 5 RS out of 750)")
        
        # Inject Custom CSS for World-Class UI
        st.markdown("""
        <style>
        .rs-leader-card {
            background: linear-gradient(145deg, rgba(20,25,35,0.8) 0%, rgba(10,15,20,0.95) 100%);
            border: 1px solid rgba(0, 243, 255, 0.2);
            border-radius: 12px;
            padding: 16px 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            overflow: hidden;
            margin-bottom: 10px;
        }
        .rs-leader-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 243, 255, 0.3);
            border: 1px solid rgba(0, 243, 255, 0.6);
        }
        .rs-leader-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 50%;
            height: 100%;
            background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.05) 50%, rgba(255,255,255,0) 100%);
            transform: skewX(-25deg);
            transition: all 0.7s ease;
        }
        .rs-leader-card:hover::before {
            left: 200%;
        }
        .rs-ticker {
            margin: 0;
            font-size: 1.25rem;
            font-weight: 800;
            background: -webkit-linear-gradient(0deg, #00f3ff, #0077ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 0.5px;
        }
        .rs-score {
            margin: 8px 0;
            font-size: 1.8rem;
            font-weight: 900;
            color: #10b981;
            text-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
            letter-spacing: -0.5px;
        }
        .rs-industry {
            margin: 0;
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .rs-rank-badge {
            position: absolute;
            top: 0;
            left: 0;
            background: rgba(0, 243, 255, 0.2);
            color: #00f3ff;
            padding: 2px 8px;
            font-size: 0.7rem;
            font-weight: 800;
            border-bottom-right-radius: 8px;
            border: 1px solid rgba(0, 243, 255, 0.3);
            border-top: none;
            border-left: none;
            backdrop-filter: blur(4px);
        }
        .rs-leader-card-link {
            text-decoration: none !important;
            color: inherit !important;
            display: block;
        }
        </style>
        """, unsafe_allow_html=True)
        
        cols = st.columns(len(top_rs_leaders))
        for idx, leader in enumerate(top_rs_leaders):
            with cols[idx]:
                # Clean up the ticker extension for better visual appeal
                clean_ticker = leader['ticker'].replace('.NS', '').replace('.BO', '')
                tv_url = f"https://in.tradingview.com/chart/?symbol=NSE:{clean_ticker}"
                
                st.markdown(
                    f"""
                    <a href="{tv_url}" target="_blank" class="rs-leader-card-link">
                        <div class="rs-leader-card">
                            <div class="rs-rank-badge">#{idx+1}</div>
                            <h4 class="rs-ticker">{clean_ticker}</h4>
                            <p class="rs-score">{leader['rs_score']:.1f} RS</p>
                            <p class="rs-industry" title="{leader['industry']}">{leader['industry']}</p>
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True
                )
        
        # Expanded Top 15 Table
        with st.expander("ℹ️ How are the RS, Smooth RS (RS/20D ATR%) & TML Scores Calculated?"):
            st.markdown("""
            **1. Tactical 3-Month RS Formula (0 to 99)**
            This scanner isolates stocks moving the fastest *right now* using a short-term, front-weighted momentum calculation:
            `Raw RS = (1-Week Return * 20%) + (1-Month Return * 40%) + (3-Month Return * 40%)`
            
            This Raw RS is then mathematically percent-ranked across the entire liquid universe of 750 stocks into a final **0 to 99 score**. 
            An RS of 99 means the stock is mathematically outperforming 99% of the universe right now.
            
            **2. TML Composite Score (Out of 120 Points)**
            The **True Market Leader (TML)** score is a holistic CANSLIM ranking combining both fundamentals and technicals:
            - **Growth (Max 40 pts):** EPS YoY Growth (20) + Sales YoY Growth (20).
            - **Momentum (Max 40 pts):** Elite 3M RS Score (25) + Proximity to 52W High (15).
            - **Institution (Max 40 pts):** Profit Margins (10) + ROE (10) + Liquidity Profiling (20).
            
            **3. Smooth RS (`Composite RS / 20D ATR%`)**
            - Divides multi-horizon Composite RS (`40% 1M + 35% 3M + 25% 6M` return %) by the stock's **20-Day Average True Range % (`ATR%`)**.
            - Directly rewards tight VCP staircase compounders and penalizes wide intraday high-low wicks that trigger `21 EMA` shakeouts (`≥ 15x` Elite Smooth Leader, `10x–15x` Strong Institutional Trend).

            **4. Dual-Horizon Sortino Ratio (3M & 6M)**
            - **Sortino 3M (Tactical):** Downside-risk-adjusted alpha over 63 trading days (~3 months).
            - **Sortino 6M (Structural):** Downside-risk-adjusted alpha over 126 trading days (~6 months).
            Penalizes only downside volatility below the 6.5% risk-free rate while ignoring explosive upside breakouts.
            """)
            
        top_15_df = pd.DataFrame(top_15_rs_leaders)
        if not top_15_df.empty:
            top_15_df['Rank'] = top_15_df.index + 1
            top_15_df['Ticker'] = top_15_df['ticker'].str.replace('.NS', '').str.replace('.BO', '')
            top_15_df['TradingView'] = top_15_df['Ticker'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{x}")
            
            # Enrich with Smooth RS (RS / 20D ATR%), Sortino 3M & Sortino 6M
            try:
                tickers_to_fetch = top_15_df['ticker'].tolist()
                df_hist = fetch_incremental_history(tickers_to_fetch, days=252)
                
                s3m_list = []
                s6m_list = []
                smooth_rs_list = []
                for t in tickers_to_fetch:
                    try:
                        sh = None
                        sl = None
                        if isinstance(df_hist.columns, pd.MultiIndex):
                            if t in df_hist.columns.get_level_values(0):
                                sub_h = df_hist.xs(t, axis=1, level=0)
                                sc = sub_h['Close'].dropna()
                                sh = sub_h['High'].dropna() if 'High' in sub_h else None
                                sl = sub_h['Low'].dropna() if 'Low' in sub_h else None
                            elif t in df_hist.columns.get_level_values(1):
                                sub_h = df_hist.xs(t, axis=1, level=1)
                                sc = sub_h['Close'].dropna()
                                sh = sub_h['High'].dropna() if 'High' in sub_h else None
                                sl = sub_h['Low'].dropna() if 'Low' in sub_h else None
                            else:
                                sc = pd.Series()
                        else:
                            sc = df_hist['Close'].dropna() if 'Close' in df_hist else pd.Series()
                            sh = df_hist['High'].dropna() if 'High' in df_hist else None
                            sl = df_hist['Low'].dropna() if 'Low' in df_hist else None
                        
                        if len(sc) >= 30:
                            s3m_list.append(calculate_sortino_ratio(sc, window=63, mar_annual=0.065))
                            s6m_list.append(calculate_sortino_ratio(sc, window=126, mar_annual=0.065))
                            smooth_rs_list.append(calculate_smooth_rs_atr(sc, sh, sl))
                        else:
                            s3m_list.append(0.0)
                            s6m_list.append(0.0)
                            smooth_rs_list.append(0.0)
                    except Exception:
                        s3m_list.append(0.0)
                        s6m_list.append(0.0)
                        smooth_rs_list.append(0.0)
                top_15_df['smooth_rs'] = smooth_rs_list
                top_15_df['sortino_3m'] = s3m_list
                top_15_df['sortino_6m'] = s6m_list
            except Exception as e:
                print(f"Error computing Sortino/Smooth RS for Top 15 RS: {e}")
                top_15_df['smooth_rs'] = 0.0
                top_15_df['sortino_3m'] = 0.0
                top_15_df['sortino_6m'] = 0.0

            # Inline Filter Controls for Top Leaders & RSNH Tables
            f_col1, f_col2, f_col3 = st.columns([1.2, 1.2, 1.6])
            with f_col1:
                app_min_smooth_rs = st.slider(
                    "⚡ Min Smooth RS (RS / 20D ATR%)",
                    min_value=-5.0, max_value=25.0, value=0.0, step=1.0,
                    key="app_top_leaders_min_smooth_rs",
                    help="Filter top leaders by volatility-efficiency: Composite RS % divided by 20-Day ATR%. (≥10x Strong, ≥15x Elite)"
                )
            with f_col2:
                app_min_s3m = st.slider(
                    "🛡️ Min Sortino 3M",
                    min_value=-2.0, max_value=10.0, value=0.0, step=0.5,
                    key="app_top_leaders_min_sortino_3m",
                    help="Filter top leaders by minimum 3-Month downside-risk-adjusted Sortino ratio (≥3.0 Elite)."
                )
            with f_col3:
                disp_top_15_df = top_15_df[
                    (top_15_df['smooth_rs'] >= app_min_smooth_rs) &
                    (top_15_df['sortino_3m'] >= app_min_s3m)
                ]
                if app_min_smooth_rs > -5.0 or app_min_s3m > -2.0:
                    st.caption(f"🎯 Showing **{len(disp_top_15_df)} of {len(top_15_df)}** Top Leaders matching `Smooth RS ≥ {app_min_smooth_rs:.1f}x` & `Sortino 3M ≥ {app_min_s3m:.1f}`")
            
            st.dataframe(
                disp_top_15_df[['Rank', 'Ticker', 'industry', 'TradingView', 'rs_score', 'smooth_rs', 'sortino_3m', 'sortino_6m', 'tml_score']],
                column_config={
                    "Rank": st.column_config.NumberColumn("#", format="%d"),
                    "Ticker": st.column_config.TextColumn("Company"),
                    "industry": st.column_config.TextColumn("Theme/Industry"),
                    "TradingView": st.column_config.LinkColumn("Chart", display_text="Open 📈"),
                    "rs_score": st.column_config.NumberColumn("3M RS Rating", format="%.1f", help="0-99 Cross-sectional ranking."),
                    "smooth_rs": st.column_config.NumberColumn("Smooth RS (RS/ATR)", format="%.2fx", help="Composite RS % divided by 20-Day ATR% (≥10x Strong, ≥15x Elite Smooth Leader)."),
                    "sortino_3m": st.column_config.NumberColumn("Sortino 3M", format="%.2f", help="3-Month downside-risk-adjusted alpha."),
                    "sortino_6m": st.column_config.NumberColumn("Sortino 6M", format="%.2f", help="6-Month structural downside-risk-adjusted alpha."),
                    "tml_score": st.column_config.NumberColumn("TML Composite", format="%.0f", help="120-point Fundamental + Technical Score.")
                },
                hide_index=True,
                use_container_width=True
            )
            
            with st.expander("📺 Export Deep Market Leaders to TradingView Watchlist"):
                tv_tickers_dml = [f"NSE:{t}" for t in disp_top_15_df['Ticker'].tolist()]
                st.code(",".join(tv_tickers_dml), language="text")
            
        st.markdown("<br/>", unsafe_allow_html=True)
        
        # ---------------------------------------------------------
        # RS Line New High (RSNH) Leaders Table (Option A)
        # ---------------------------------------------------------
        rsnh_leaders = fetch_rsnh_leaders(market='INDIA', limit=15)
        if rsnh_leaders:
            st.markdown("### 🔵 RS Line New High (RSNH) — Pre-Breakout Coils")
            with st.expander("ℹ️ What is RS Line New High (RSNH) Divergence & How to Trade It?"):
                st.markdown("""
                **O'Neil Blue Dot Setup (RS Line at 52W High AHEAD of Price)**
                - **The Core Edge (True Divergence):** The stock's Relative Strength line (Price vs Nifty 500) has reached a **52-week High**, but the stock's price has **NOT broken out yet** (it is coiling strictly **1% to 15% below its 52W price high**).
                - **Why It's Different from the Table Above:**
                  - The table above shows runaway momentum leaders that have already broken out to price highs (`At High 🚀`).
                  - **This table isolates pre-breakout setups** where institutions are quietly absorbing supply inside a base *before* the price breaks out.
                - **Setup Status:**
                  - `🎯 In Buy Zone`: Price is within 1% to 5% of its 52W high pivot (ready to ignite).
                  - `⏳ Forming Base`: Price is 5% to 15% below high (building right side of handle/base).
                - **Smooth RS (`RS / 20D ATR%`) & Sortino (3M & 6M):** High Smooth RS (`≥ 10x`) and Sortino (`≥ 3.0`) confirm tight institutional volatility contraction without wide intraday distribution bars.
                """)
            
            rsnh_df = pd.DataFrame(rsnh_leaders)
            if 'smooth_rs' not in rsnh_df.columns:
                rsnh_df['smooth_rs'] = 0.0
            st.dataframe(
                rsnh_df[['Rank', 'Ticker', 'industry', 'TradingView', 'rs_score', 'smooth_rs', 'sortino_3m', 'sortino_6m', 'dist_52w', 'status', 'tml_score']],
                column_config={
                    "Rank": st.column_config.NumberColumn("#", format="%d"),
                    "Ticker": st.column_config.TextColumn("Company"),
                    "industry": st.column_config.TextColumn("Theme/Industry"),
                    "TradingView": st.column_config.LinkColumn("Chart", display_text="Open 📈"),
                    "rs_score": st.column_config.NumberColumn("3M RS Rating", format="%.1f", help="0-99 Cross-sectional momentum ranking."),
                    "smooth_rs": st.column_config.NumberColumn("Smooth RS (RS/ATR)", format="%.2fx", help="Composite RS % divided by 20-Day ATR% (≥10x Strong, ≥15x Elite Smooth Leader)."),
                    "sortino_3m": st.column_config.NumberColumn("Sortino 3M", format="%.2f", help="3-Month downside-risk-adjusted alpha."),
                    "sortino_6m": st.column_config.NumberColumn("Sortino 6M", format="%.2f", help="6-Month structural downside-risk-adjusted alpha."),
                    "dist_52w": st.column_config.TextColumn("52W High Dist", help="Distance from 52-week price high (1% to 15% below pivot)."),
                    "status": st.column_config.TextColumn("Setup Status", help="🎯 In Buy Zone (1-5% below high) | ⏳ Forming Base (5-15% below high)"),
                    "tml_score": st.column_config.NumberColumn("TML Composite", format="%.0f", help="120-point Fundamental + Technical Score.")
                },
                hide_index=True,
                use_container_width=True
            )
            
            with st.expander("📺 Export RS Line New High (RSNH) to TradingView Watchlist"):
                tv_tickers_rsnh = [f"NSE:{t}" for t in rsnh_df['Ticker'].tolist()]
                st.code(",".join(tv_tickers_rsnh), language="text")
            
            st.markdown("<br/>", unsafe_allow_html=True)
        else:
            st.info("ℹ️ No stocks currently exhibit RS Line New High divergence (all current momentum leaders have already broken out to price highs or are pulling back >15%).")
            st.markdown("<br/>", unsafe_allow_html=True)
            
    top_15_rs_leaders_us = get_top_rs_leaders(market='US', limit=15)
    top_rs_leaders_us = top_15_rs_leaders_us[:5] if top_15_rs_leaders_us else []
    if top_rs_leaders_us:
        st.markdown("### 🦅 US Market Leaders (Top 5 RS)")
        cols_us = st.columns(len(top_rs_leaders_us))
        for idx, leader in enumerate(top_rs_leaders_us):
            with cols_us[idx]:
                clean_ticker = leader['ticker']
                tv_url = f"https://www.tradingview.com/chart/?symbol={clean_ticker}"
                
                st.markdown(
                    f"""
                    <a href="{tv_url}" target="_blank" class="rs-leader-card-link">
                        <div class="rs-leader-card">
                            <div class="rs-rank-badge">#{idx+1}</div>
                            <h4 class="rs-ticker">{clean_ticker}</h4>
                            <p class="rs-score">{leader['rs_score']:.1f} RS</p>
                            <p class="rs-industry" title="{leader['industry']}">{leader['industry']}</p>
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True
                )
        
        # Expanded Top 15 US Table
        with st.expander("ℹ️ How are US RS, Smooth RS (RS/20D ATR%) & TML Scores Calculated?"):
            st.markdown("""
            **1. Tactical 3-Month US RS Formula (0 to 99)**
            Calculates front-weighted momentum relative to the S&P 500 (^GSPC):
            `Raw RS = (1-Week Return * 20%) + (1-Month Return * 40%) + (3-Month Return * 40%)`
            
            This Raw RS is then mathematically percent-ranked across the US tracked universe into a final **0 to 99 score**. 
            An RS of 99 means the stock is mathematically outperforming 99% of the universe right now.
            
            **2. TML Composite Score (Out of 120 Points)**
            The **True Market Leader (TML)** score is a holistic CANSLIM ranking combining both fundamentals and technicals:
            - **Growth (Max 40 pts):** EPS YoY Growth (20) + Sales YoY Growth (20).
            - **Momentum (Max 40 pts):** Elite 3M RS Score (25) + Proximity to 52W High (15).
            - **Institution (Max 40 pts):** Profit Margins (10) + ROE (10) + Liquidity Profiling (20).
            
            **3. Smooth RS (`Composite RS / 20D ATR%`)**
            - Divides multi-horizon Composite RS (`40% 1M + 35% 3M + 25% 6M` return %) by the stock's **20-Day Average True Range % (`ATR%`)**.

            **4. Dual-Horizon Sortino Ratio (3M & 6M)**
            - **Sortino 3M (Tactical):** Downside-risk-adjusted alpha over 63 trading days (~3 months).
            - **Sortino 6M (Structural):** Downside-risk-adjusted alpha over 126 trading days (~6 months).
            Penalizes only downside volatility below the 6.5% risk-free rate while ignoring explosive upside breakouts.
            """)
            
        top_15_us_df = pd.DataFrame(top_15_rs_leaders_us)
        if not top_15_us_df.empty:
            top_15_us_df['Rank'] = top_15_us_df.index + 1
            top_15_us_df['Ticker'] = top_15_us_df['ticker']
            top_15_us_df['TradingView'] = top_15_us_df['Ticker'].apply(lambda x: f"https://www.tradingview.com/chart/?symbol={x}")
            
            # Enrich with Smooth RS, Sortino 3M & Sortino 6M
            try:
                tickers_to_fetch = top_15_us_df['ticker'].tolist()
                df_hist_us = fetch_incremental_history(tickers_to_fetch, days=252)
                
                s3m_list = []
                s6m_list = []
                smooth_rs_us_list = []
                for t in tickers_to_fetch:
                    try:
                        sh = None
                        sl = None
                        if isinstance(df_hist_us.columns, pd.MultiIndex):
                            if 'Ticker' in df_hist_us.columns.names and t in df_hist_us.columns.get_level_values('Ticker'):
                                sub_h = df_hist_us.xs(t, axis=1, level='Ticker')
                                sc = sub_h['Close'].dropna()
                                sh = sub_h['High'].dropna() if 'High' in sub_h else None
                                sl = sub_h['Low'].dropna() if 'Low' in sub_h else None
                            elif t in df_hist_us.columns.get_level_values(0):
                                sub_h = df_hist_us.xs(t, axis=1, level=0)
                                sc = sub_h['Close'].dropna()
                                sh = sub_h['High'].dropna() if 'High' in sub_h else None
                                sl = sub_h['Low'].dropna() if 'Low' in sub_h else None
                            elif t in df_hist_us.columns.get_level_values(1):
                                sub_h = df_hist_us.xs(t, axis=1, level=1)
                                sc = sub_h['Close'].dropna()
                                sh = sub_h['High'].dropna() if 'High' in sub_h else None
                                sl = sub_h['Low'].dropna() if 'Low' in sub_h else None
                            else:
                                sc = pd.Series()
                        else:
                            sc = df_hist_us['Close'].dropna() if 'Close' in df_hist_us else pd.Series()
                            sh = df_hist_us['High'].dropna() if 'High' in df_hist_us else None
                            sl = df_hist_us['Low'].dropna() if 'Low' in df_hist_us else None
                        
                        if len(sc) >= 30:
                            s3m_list.append(calculate_sortino_ratio(sc, window=63, mar_annual=0.065))
                            s6m_list.append(calculate_sortino_ratio(sc, window=126, mar_annual=0.065))
                            smooth_rs_us_list.append(calculate_smooth_rs_atr(sc, sh, sl))
                        else:
                            s3m_list.append(0.0)
                            s6m_list.append(0.0)
                            smooth_rs_us_list.append(0.0)
                    except Exception:
                        s3m_list.append(0.0)
                        s6m_list.append(0.0)
                        smooth_rs_us_list.append(0.0)
                top_15_us_df['smooth_rs'] = smooth_rs_us_list
                top_15_us_df['sortino_3m'] = s3m_list
                top_15_us_df['sortino_6m'] = s6m_list
            except Exception as e:
                print(f"Error computing Sortino/Smooth RS for US Top 15 RS: {e}")
                top_15_us_df['smooth_rs'] = 0.0
                top_15_us_df['sortino_3m'] = 0.0
                top_15_us_df['sortino_6m'] = 0.0
            
            st.dataframe(
                top_15_us_df[['Rank', 'Ticker', 'industry', 'TradingView', 'rs_score', 'smooth_rs', 'sortino_3m', 'sortino_6m', 'tml_score']],
                column_config={
                    "Rank": st.column_config.NumberColumn("#", format="%d"),
                    "Ticker": st.column_config.TextColumn("Company"),
                    "industry": st.column_config.TextColumn("Theme/Industry"),
                    "TradingView": st.column_config.LinkColumn("Chart", display_text="Open 📈"),
                    "rs_score": st.column_config.NumberColumn("3M RS Rating", format="%.1f", help="0-99 Cross-sectional ranking."),
                    "smooth_rs": st.column_config.NumberColumn("Smooth RS (RS/ATR)", format="%.2fx", help="Composite RS % divided by 20-Day ATR% (≥10x Strong, ≥15x Elite Smooth Leader)."),
                    "sortino_3m": st.column_config.NumberColumn("Sortino 3M", format="%.2f", help="3-Month downside-risk-adjusted alpha."),
                    "sortino_6m": st.column_config.NumberColumn("Sortino 6M", format="%.2f", help="6-Month structural downside-risk-adjusted alpha."),
                    "tml_score": st.column_config.NumberColumn("TML Composite", format="%.0f", help="120-point Fundamental + Technical Score.")
                },
                hide_index=True,
                use_container_width=True
            )
            
            with st.expander("📺 Export Deep Market Leaders to TradingView Watchlist"):
                tv_tickers_dml_us = top_15_us_df['Ticker'].tolist()
                st.code(",".join(tv_tickers_dml_us), language="text")
            
        st.markdown("<br/>", unsafe_allow_html=True)
        
        # ---------------------------------------------------------
        # US RS Line New High (RSNH) Leaders Table
        # ---------------------------------------------------------
        rsnh_leaders_us = fetch_rsnh_leaders(market='US', limit=15)
        if rsnh_leaders_us:
            st.markdown("### 🔵 RS Line New High (RSNH) — Pre-Breakout Coils")
            with st.expander("ℹ️ What is RS Line New High (RSNH) Divergence & How to Trade It?"):
                st.markdown("""
                **O'Neil Blue Dot Setup (RS Line at 52W High AHEAD of Price vs S&P 500)**
                - **The Core Edge (True Divergence):** The stock's Relative Strength line (Price vs S&P 500 / ^GSPC) has reached a **52-week High**, but the stock's price has **NOT broken out yet** (it is coiling strictly **1% to 15% below its 52W price high**).
                - **Why It's Different from the Table Above:**
                  - The table above shows runaway momentum leaders that have already broken out to price highs (`At High 🚀`).
                  - **This table isolates pre-breakout setups** where institutions are quietly absorbing supply inside a base *before* the price breaks out.
                - **Setup Status:**
                  - `🎯 In Buy Zone`: Price is within 1% to 5% of its 52W high pivot (ready to ignite).
                  - `⏳ Forming Base`: Price is 5% to 15% below high (building right side of handle/base).
                - **Smooth RS (`RS / 20D ATR%`) & Sortino (3M & 6M):** Confirms the base is absorbing selloffs tightly without wide intraday volatility drag.
                """)
            
            rsnh_df_us = pd.DataFrame(rsnh_leaders_us)
            if 'smooth_rs' not in rsnh_df_us.columns:
                rsnh_df_us['smooth_rs'] = 0.0
            st.dataframe(
                rsnh_df_us[['Rank', 'Ticker', 'industry', 'TradingView', 'rs_score', 'smooth_rs', 'sortino_3m', 'sortino_6m', 'dist_52w', 'status', 'tml_score']],
                column_config={
                    "Rank": st.column_config.NumberColumn("#", format="%d"),
                    "Ticker": st.column_config.TextColumn("Company"),
                    "industry": st.column_config.TextColumn("Theme/Industry"),
                    "TradingView": st.column_config.LinkColumn("Chart", display_text="Open 📈"),
                    "rs_score": st.column_config.NumberColumn("3M RS Rating", format="%.1f", help="0-99 Cross-sectional momentum ranking."),
                    "smooth_rs": st.column_config.NumberColumn("Smooth RS (RS/ATR)", format="%.2fx", help="Composite RS % divided by 20-Day ATR% (≥10x Strong, ≥15x Elite Smooth Leader)."),
                    "sortino_3m": st.column_config.NumberColumn("Sortino 3M", format="%.2f", help="3-Month downside-risk-adjusted alpha."),
                    "sortino_6m": st.column_config.NumberColumn("Sortino 6M", format="%.2f", help="6-Month structural downside-risk-adjusted alpha."),
                    "dist_52w": st.column_config.TextColumn("52W High Dist", help="Distance from 52-week price high (1% to 15% below pivot)."),
                    "status": st.column_config.TextColumn("Setup Status", help="🎯 In Buy Zone (1-5% below high) | ⏳ Forming Base (5-15% below high)"),
                    "tml_score": st.column_config.NumberColumn("TML Composite", format="%.0f", help="120-point Fundamental + Technical Score.")
                },
                hide_index=True,
                use_container_width=True
            )
            
            with st.expander("📺 Export RS Line New High (RSNH) to TradingView Watchlist"):
                tv_tickers_rsnh_us = rsnh_df_us['Ticker'].tolist()
                st.code(",".join(tv_tickers_rsnh_us), language="text")
            
            st.markdown("<br/>", unsafe_allow_html=True)
        else:
            st.info("ℹ️ No US stocks currently exhibit RS Line New High divergence (all current momentum leaders have already broken out to price highs or are pulling back >15%).")
            st.markdown("<br/>", unsafe_allow_html=True)
        
    # =========================================================
    # NEW: Market FOMO / FEAR Indicator
    # =========================================================
    @st.cache_data(ttl=3600)
    def compute_fomo_score(days=90):
        import pandas as pd
        import os
        import requests
        
        valid_tickers = []
        try:
            from market_data import fetch_nifty_total_market_tickers
            valid_tickers = fetch_nifty_total_market_tickers(show_progress=False)
        except Exception:
            pass
            
        if not valid_tickers and os.path.exists('tickers.txt'):
            with open('tickers.txt', 'r') as f:
                valid_tickers = [line.strip().upper() for line in f if line.strip()]
                valid_tickers = [t + '.NS' if not t.endswith('.NS') else t for t in valid_tickers]
                
        matrix_path = 'historical_prices_matrix.pkl'
        if not os.path.exists(matrix_path) or not valid_tickers:
            return None, None
            
        try:
            df = pd.read_pickle(matrix_path)
            if df.empty: return None, None
            
            closes = df.xs('Close', level=1, axis=1).dropna(how='all')
            available_tickers = [t for t in valid_tickers if t in closes.columns]
            if not available_tickers: return None, None
            
            closes = closes[available_tickers]
            closes = closes.tail(150).ffill() # Use forward fill for missing recent days
            
            ema5 = closes.ewm(span=5, adjust=False).mean()
            
            above_ema = (closes > ema5).sum(axis=1)
            total_valid = closes.notna().sum(axis=1)
            
            fomo_series = (above_ema / total_valid) * 100
            fomo_series = fomo_series.tail(days)
            
            return fomo_series.iloc[-1], fomo_series
        except Exception as e:
            print(f"FOMO Error: {e}")
            return None, None

    fomo_current, fomo_series = compute_fomo_score(days=90)
    if fomo_current is not None and fomo_series is not None:
        st.markdown("### 🌡️ Market FOMO / FEAR Indicator")
        st.caption("Percentage of Nifty Total Market (750 stocks) trading above their 5-day EMA.")
        
        if fomo_current >= 80:
            status_text = "🔴 Red Zone: Market is stretched. Do not chase breakouts."
            status_color = "#ef4444"
        elif fomo_current <= 25:
            status_text = "🟢 Fear Zone: Market is washed out. Hunt for RS leaders."
            status_color = "#10b981"
        else:
            status_text = "🟡 Neutral Zone: Look for pullbacks and coils."
            status_color = "#f59e0b"
            
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f'''
            <div style="background: linear-gradient(145deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.8) 100%); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.05); border-left: 4px solid {status_color}; border-radius: 12px; padding: 20px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; min-height: 200px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);">
                <div style="font-size: 0.95rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 15px; font-weight: 600;">Current Score</div>
                <div style="font-size: 4rem; font-weight: 900; color: {status_color}; line-height: 1; text-shadow: 0 0 30px {status_color}66; font-family: 'JetBrains Mono', monospace;">{fomo_current:.1f}%</div>
                <div style="font-size: 0.95rem; color: #cbd5e1; margin-top: 20px; font-weight: 500; padding: 8px 12px; background: rgba(0,0,0,0.2); border-radius: 8px; display: inline-block; line-height: 1.4;">{status_text}</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with col2:
            fig = go.Figure()
            
            c_rgb = status_color.lstrip('#')
            if len(c_rgb) == 6:
                r, g, b = tuple(int(c_rgb[i:i+2], 16) for i in (0, 2, 4))
                fill_color = f"rgba({r}, {g}, {b}, 0.35)"
            else:
                fill_color = "rgba(59, 130, 246, 0.35)"
                
            fig.add_trace(go.Scatter(
                x=fomo_series.index, y=fomo_series.values,
                fill='tozeroy',
                mode='lines',
                line=dict(color=status_color, width=2.5, shape='spline', smoothing=1.3),
                fillcolor=fill_color,
                name='FOMO Score',
                hovertemplate='%{x}<br><b>%{y:.1f}%</b><extra></extra>'
            ))
            fig.add_hline(y=80, line_dash="dash", line_color="#ef4444", annotation_text="Overbought (80)", annotation_position="top left", annotation_font_color="#ef4444")
            fig.add_hline(y=25, line_dash="dash", line_color="#10b981", annotation_text="Oversold (25)", annotation_position="bottom left", annotation_font_color="#10b981")
            fig.add_hline(y=50, line_dash="solid", line_color="rgba(255,255,255,0.15)", line_width=1)
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=20, b=0),
                height=220,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, color='#94a3b8'),
                yaxis=dict(showgrid=False, color='#94a3b8', range=[0, 100], zeroline=False),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        st.markdown("<br/>", unsafe_allow_html=True)
        
    st.markdown("### 🔭 Market Direction HUD")
    
    @st.cache_data(ttl=300)
    def fetch_hud_data():
        import pandas as pd
        import yfinance as yf
        indices = {
            "Nifty 50": "^NSEI",
            "Nifty 500": "^CRSLDX", 
            "Nasdaq 100": "^NDX",
            "S&P 500": "^GSPC",
            "GOLDBEES (Gold)": "GOLDBEES.NS",
            "SILVERBEES (Silver)": "SILVERBEES.NS",
            "MON100 (Nasdaq 100)": "MON100.NS"
        }
        fetched_hud = {}
        try:
            tickers = list(indices.values())
            df_batch = yf.download(tickers, period="3mo", group_by="ticker", progress=False, threads=True)
            for name, tckr in indices.items():
                df_idx = pd.DataFrame()
                if isinstance(df_batch.columns, pd.MultiIndex):
                    if 'Ticker' in df_batch.columns.names:
                        df_idx = df_batch.xs(tckr, axis=1, level='Ticker').copy()
                    else:
                        if tckr in df_batch.columns.levels[0]:
                            df_idx = df_batch[tckr].copy()
                        elif tckr in df_batch.columns.levels[1]:
                            df_idx = df_batch.xs(tckr, axis=1, level=1).copy()
                else:
                    df_idx = df_batch.copy()
                    
                df_idx = df_idx.dropna(subset=['Close'])
                if not df_idx.empty:
                    df_idx['ema10'] = df_idx['Close'].ewm(span=10).mean()
                    df_idx['ema21'] = df_idx['Close'].ewm(span=21).mean()
                    
                    last = df_idx.iloc[-1]
                    last_5 = df_idx.iloc[-5] if len(df_idx) >= 5 else last
                    
                    fetched_hud[name] = {
                        'close': float(last['Close']),
                        'ema10': float(last['ema10']),
                        'ema21': float(last['ema21']),
                        'slope_up': float(last['ema21']) > float(last_5['ema21'])
                    }
        except Exception as e:
            print(f"HUD Error: {e}")
        return fetched_hud
        
    hud_data = fetch_hud_data()
    
    if hud_data:
        st.markdown("""
        <style>
        @keyframes pulse-green {
            0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        @keyframes pulse-red {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
        .hud-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.8) 100%);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px 15px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 15px;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            position: relative;
        }
        .hud-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.15);
        }
        .hud-title {
            color: #cbd5e1; /* Much brighter for readability */
            font-size: 0.9rem;
            font-weight: bold;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.85rem;
            margin-bottom: 12px;
        }
        .badge-power {
            background-color: rgba(16, 185, 129, 0.2);
            color: #10b981;
            border: 1px solid #10b981;
            animation: pulse-green 2s infinite;
        }
        .badge-pullback {
            background-color: rgba(245, 158, 11, 0.2);
            color: #f59e0b;
            border: 1px solid #f59e0b;
        }
        .badge-cash {
            background-color: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            border: 1px solid #ef4444;
            animation: pulse-red 2s infinite;
        }
        .hud-metrics {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: #cbd5e1;
            padding: 0 10px;
        }
        .hud-metric {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .hud-metric span:first-child {
            color: #cbd5e1; /* Made text brighter */
            font-size: 0.75rem; /* Slightly larger */
            margin-bottom: 2px;
            font-weight: 600;
        }
        </style>
        """, unsafe_allow_html=True)
        
        def render_hud_card(col, name, data):
            c = data['close']
            e10 = data['ema10']
            e21 = data['ema21']
            slope = data['slope_up']
            
            a10 = c > e10
            a21 = c > e21
            
            if a10 and a21 and slope:
                b_class = "badge-power"
                b_text = "🟢 POWER TREND"
                card_color = "#10b981"
            elif a21 and not a10:
                b_class = "badge-pullback"
                b_text = "🟡 PULLBACK"
                card_color = "#f59e0b"
            elif not a21:
                b_class = "badge-cash"
                b_text = "🔴 CASH ZONE"
                card_color = "#ef4444"
            else:
                b_class = "badge-pullback"
                b_text = "🟡 CHOPPY"
                card_color = "#f59e0b"
                
            html = f'''
            <div class="hud-card" style="border-top: 4px solid {card_color};">
                <div class="hud-title" style="letter-spacing: 1.5px; color: #e2e8f0; font-size: 0.95rem;">{name}</div>
                <div class="badge {b_class}" style="margin-top: 5px; margin-bottom: 25px; font-size: 0.8rem; letter-spacing: 0.5px;">{b_text}</div>
                <div class="hud-metrics">
                    <div class="hud-metric">
                        <span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 6px;">10 EMA</span>
                        <span style="font-size: 1.1rem; filter: drop-shadow(0 0 8px {'#10b981' if a10 else '#ef4444'});">{'✅' if a10 else '❌'}</span>
                    </div>
                    <div class="hud-metric">
                        <span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 6px;">21 EMA</span>
                        <span style="font-size: 1.1rem; filter: drop-shadow(0 0 8px {'#10b981' if a21 else '#ef4444'});">{'✅' if a21 else '❌'}</span>
                    </div>
                    <div class="hud-metric">
                        <span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 6px;">21 Slope</span>
                        <span style="font-size: 1.1rem; filter: drop-shadow(0 0 8px {'#10b981' if slope else '#ef4444'});">{'✅' if slope else '❌'}</span>
                    </div>
                </div>
            </div>
            '''
            col.markdown(html, unsafe_allow_html=True)
            
        # Row 1: Core Equity Indices
        equity_keys = ["Nifty 50", "Nifty 500", "Nasdaq 100", "S&P 500"]
        cols_eq = st.columns(4)
        for idx, name in enumerate(equity_keys):
            if name in hud_data:
                render_hud_card(cols_eq[idx], name, hud_data[name])
                
        # Row 2: Commodities & International ETFs
        st.markdown("""
        <div style="margin-top: 4px; margin-bottom: 12px; font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px; display: flex; align-items: center; gap: 8px;">
            <span>🪙 Commodities & Global ETFs</span>
            <div style="flex: 1; height: 1px; background: rgba(255,255,255,0.08);"></div>
        </div>
        """, unsafe_allow_html=True)
        
        etf_keys = ["GOLDBEES (Gold)", "SILVERBEES (Silver)", "MON100 (Nasdaq 100)"]
        cols_etf = st.columns(3)
        for idx, name in enumerate(etf_keys):
            if name in hud_data:
                render_hud_card(cols_etf[idx], name, hud_data[name])
            
        # -- BREADTH ENGINE --
        import breadth_engine
        st.markdown("### 📊 Total Market Breadth")
        
        tab_b_in, tab_b_us, tab_rotation = st.tabs(["🇮🇳 India (Nifty 500)", "🇺🇸 US Equities (S&P/Nasdaq)", "🌍 Global Asset Rotation"])
        
        def render_breadth_ui(market_code, tab, title_desc):
            breadth_data = breadth_engine.compute_daily_breadth(force=False, market=market_code)
            
            # Fetch 180 days to calculate accurate 50-day SMAs, then we'll truncate to 120 for Extreme Indicators
            # and 60 for the historical area charts.
            full_hist_df = breadth_engine.fetch_historical_breadth(market=market_code, days=180)
            hist_df = full_hist_df.tail(60) if not full_hist_df.empty else full_hist_df
            
            if not breadth_data:
                tab.info(f"No breadth data available yet for {market_code}. Please run the historical data updater.")
                return
                
            with tab:
                st.caption(title_desc)
                
                b_cols = st.columns(3)
                
                # SVG Gauge helper
                def create_gauge_svg(pct, color):
                    # Circumference of half circle with radius 40 is 125.66
                    dashoffset = 125.66 * (1 - (pct / 100))
                    return f'<svg viewBox="0 0 100 55" style="width: 100%; max-height: 80px; margin-top: 10px;"><path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#374151" stroke-width="12" stroke-linecap="round" /><path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round" stroke-dasharray="125.66" stroke-dashoffset="{dashoffset}" style="transition: stroke-dashoffset 1s ease-in-out; filter: drop-shadow(0 0 4px {color}88);" /><text x="50" y="45" font-family="monospace" font-size="20" font-weight="bold" fill="{color}" text-anchor="middle">{pct:.1f}%</text></svg>'
                
                # Gauge 1: % > 50 SMA
                with b_cols[0]:
                    pct_50 = breadth_data['above_50_pct']
                    c_50 = "#ef4444" if pct_50 > 80 else "#3b82f6" if pct_50 < 20 else "#10b981"
                    
                    status_50 = "Broad Participation"
                    if pct_50 > 80: status_50 = "Overbought"
                    elif pct_50 < 20: status_50 = "Capitulation (Buy Zone)"
                    
                    st.markdown(f'<div class="hud-card" style="padding: 15px 10px; border-bottom: 3px solid {c_50}; height: 220px; display: flex; flex-direction: column; justify-content: space-between;"><div class="hud-title" style="margin-bottom: 0;">% Above 50-Day SMA</div>{create_gauge_svg(pct_50, c_50)}<div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 0; padding: 4px; background: rgba(0,0,0,0.2); border-radius: 6px; display: inline-block; width: fit-content; margin-left: auto; margin-right: auto;">Status: <strong style="color: {c_50}; letter-spacing: 0.5px;">{status_50}</strong></div></div>', unsafe_allow_html=True)
                    
                # Gauge 2: % > 200 SMA
                with b_cols[1]:
                    pct_200 = breadth_data['above_200_pct']
                    c_200 = "#ef4444" if pct_200 > 80 else "#3b82f6" if pct_200 < 20 else "#10b981"
                    
                    status_200 = "Secular Bull"
                    if pct_200 > 80: status_200 = "Late Stage Rally"
                    elif pct_200 < 20: status_200 = "Secular Bear"
                    
                    st.markdown(f'<div class="hud-card" style="padding: 15px 10px; border-bottom: 3px solid {c_200}; height: 220px; display: flex; flex-direction: column; justify-content: space-between;"><div class="hud-title" style="margin-bottom: 0;">% Above 200-Day SMA</div>{create_gauge_svg(pct_200, c_200)}<div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 0; padding: 4px; background: rgba(0,0,0,0.2); border-radius: 6px; display: inline-block; width: fit-content; margin-left: auto; margin-right: auto;">Status: <strong style="color: {c_200}; letter-spacing: 0.5px;">{status_200}</strong></div></div>', unsafe_allow_html=True)
                    
                # Net New Highs
                with b_cols[2]:
                    nnh = breadth_data['net_new_highs']
                    nnh_color = "#10b981" if nnh >= 0 else "#ef4444"
                    nnh_sign = "↗ +" if nnh > 0 else "↘ "
                    
                    b_health = "Healthy Expansion"
                    if pct_50 > 80: b_health = "Warning: Overbought"
                    elif pct_50 < 20: b_health = "Capitulation (Buy Zone)"
                    elif nnh < 0 and hud_data.get('Nifty 50' if market_code == 'IN' else 'S&P 500', {}).get('slope_up', False): b_health = "Bearish Divergence"
                    elif nnh < 0: b_health = "Bearish (Contracting)"
                    
                    glow_shadow = f"drop-shadow(0px 0px 8px {nnh_color}55)"
                    st.markdown(f'<div class="hud-card" style="padding: 15px 10px; border-bottom: 3px solid {nnh_color}; height: 220px; display: flex; flex-direction: column; justify-content: space-between;"><div class="hud-title" style="margin-bottom: 0;">Net New Highs (52W)</div><div style="font-size: 2.2rem; color: {nnh_color}; font-weight: 800; margin: 0; font-family: \'JetBrains Mono\', monospace; filter: {glow_shadow};">{nnh_sign}{nnh}</div><div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 0; padding: 4px; background: rgba(0,0,0,0.2); border-radius: 6px; display: inline-block; width: fit-content; margin-left: auto; margin-right: auto;">Status: <strong style="color: {nnh_color}; letter-spacing: 0.5px;">{b_health}</strong></div></div>', unsafe_allow_html=True)
                    
                # -- HISTORICAL PLOTLY CHARTS --
                if not hist_df.empty:
                    st.markdown("<br/>", unsafe_allow_html=True)
                    st.markdown("#### 📉 Historical Trends (60 Days)")
                    
                    hist_cols = st.columns(3)
                    
                    def create_area_chart(series, name, color, y_min=0, y_max=100, thresh_high=80, thresh_low=20):
                        fig = go.Figure()
                        
                        # Convert hex to rgba for fill
                        c_rgb = color.lstrip('#')
                        if len(c_rgb) == 6:
                            r, g, b = tuple(int(c_rgb[i:i+2], 16) for i in (0, 2, 4))
                            fill_color = f"rgba({r}, {g}, {b}, 0.15)"
                        else:
                            fill_color = "rgba(59, 130, 246, 0.15)"
                        
                        fig.add_trace(go.Scatter(
                            x=series.index, y=series.values,
                            fill='tozeroy', mode='lines',
                            line=dict(color=color, width=2.5, shape='spline', smoothing=1.3),
                            fillcolor=fill_color,
                            name=name,
                            hovertemplate='%{x}<br><b>%{y:.1f}</b><extra></extra>'
                        ))
                        
                        fig.add_hline(y=thresh_high, line_dash="dash", line_color="#ef4444", line_width=1, opacity=0.5)
                        fig.add_hline(y=thresh_low, line_dash="dash", line_color="#10b981", line_width=1, opacity=0.5)
                        fig.add_hline(y=50, line_dash="solid", line_color="rgba(255,255,255,0.2)", line_width=1)
                        
                        fig.update_layout(
                            margin=dict(l=0, r=0, t=10, b=0),
                            height=150,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=False, visible=False),
                            yaxis=dict(showgrid=False, color='#94a3b8', range=[y_min, y_max], zeroline=False),
                            hovermode='x unified'
                        )
                        return fig

                    def create_bar_chart(series, name):
                        fig = go.Figure()
                        
                        colors = ['#10b981' if val >= 0 else '#ef4444' for val in series.values]
                        
                        fig.add_trace(go.Bar(
                            x=series.index, y=series.values,
                            marker_color=colors,
                            name=name,
                            hovertemplate='%{x}<br><b>%{y}</b><extra></extra>'
                        ))
                        
                        fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.3)", line_width=1)
                        
                        y_min = min(0, series.min())
                        y_max = max(0, series.max())
                        padding = max(abs(y_min), abs(y_max)) * 0.1
                        if padding == 0: padding = 10
                        
                        fig.update_layout(
                            margin=dict(l=0, r=0, t=10, b=0),
                            height=150,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=False, visible=False),
                            yaxis=dict(showgrid=False, color='#94a3b8', range=[y_min - padding, y_max + padding], zeroline=False),
                            hovermode='x unified',
                            bargap=0.2
                        )
                        return fig

                    with hist_cols[0]:
                        st.plotly_chart(create_area_chart(hist_df['above_50_pct'], '% > 50 SMA', c_50), use_container_width=True, config={'displayModeBar': False})
                        
                    with hist_cols[1]:
                        st.plotly_chart(create_area_chart(hist_df['above_200_pct'], '% > 200 SMA', c_200), use_container_width=True, config={'displayModeBar': False})
                        
                    with hist_cols[2]:
                        st.plotly_chart(create_bar_chart(hist_df['net_new_highs'], 'Net New Highs'), use_container_width=True, config={'displayModeBar': False})
                        
                    # Pradeep Bonde Extreme Indicators
                    if 'surge_extreme_5d' in hist_df.columns:
                        pct_thresh_txt = "12%" if market_code == "IN" else "20%"
                        st.markdown("<br/>", unsafe_allow_html=True)
                        st.markdown("#### ⚡ Market Extremes (120 Days)")
                        ex_cols = st.columns(2)
                        
                        def create_surge_chart(series, name, overlay_sma=True):
                            fig = go.Figure()
                            
                            sma50 = None
                            if overlay_sma and len(series) >= 50:
                                sma50 = series.rolling(50, min_periods=40).mean()
                                
                            # Truncate both series to just the last 120 days for the actual chart
                            series = series.tail(120)
                            if sma50 is not None:
                                sma50 = sma50.tail(120)
                            
                            # Bar chart for counts
                            fig.add_trace(go.Bar(
                                x=series.index, y=series.values,
                                marker_color='rgba(16, 185, 129, 0.8)',
                                marker_line_color='#10b981',
                                marker_line_width=1,
                                name=name,
                                hovertemplate='%{x}<br><b>%{y} Stocks</b><extra></extra>'
                            ))
                            
                            if sma50 is not None:
                                fig.add_trace(go.Scatter(
                                    x=sma50.index, y=sma50.values,
                                    mode='lines',
                                    fill='tozeroy',
                                    fillcolor='rgba(245, 158, 11, 0.1)',
                                    line=dict(color='#f59e0b', width=2.5, shape='spline', smoothing=1.3),
                                    name='50-Day SMA',
                                    hovertemplate='50 SMA: <b>%{y:.1f}</b><extra></extra>'
                                ))
                                
                            fig.update_layout(
                                margin=dict(l=0, r=0, t=5, b=20),
                                height=180,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                showlegend=False,
                                xaxis=dict(showgrid=False, visible=True, color='#64748b', tickfont=dict(size=10)),
                                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#94a3b8', zeroline=False),
                                hovermode='x unified',
                                bargap=0.2
                            )
                            return fig
                            
                        def create_panic_chart(series, name, overlay_sma=True):
                            fig = go.Figure()
                            
                            sma50 = None
                            if overlay_sma and len(series) >= 50:
                                sma50 = series.rolling(50, min_periods=40).mean()
                                
                            # Truncate both series to just the last 120 days for the actual chart
                            series = series.tail(120)
                            if sma50 is not None:
                                sma50 = sma50.tail(120)
                            
                            fig.add_trace(go.Bar(
                                x=series.index, y=series.values,
                                marker_color='rgba(239, 68, 68, 0.8)',
                                marker_line_color='#ef4444',
                                marker_line_width=1,
                                name=name,
                                hovertemplate='%{x}<br><b>%{y} Stocks</b><extra></extra>'
                            ))
                            
                            if sma50 is not None:
                                fig.add_trace(go.Scatter(
                                    x=sma50.index, y=sma50.values,
                                    mode='lines',
                                    fill='tozeroy',
                                    fillcolor='rgba(245, 158, 11, 0.1)',
                                    line=dict(color='#f59e0b', width=2.5, shape='spline', smoothing=1.3),
                                    name='50-Day SMA',
                                    hovertemplate='50 SMA: <b>%{y:.1f}</b><extra></extra>'
                                ))
                                
                            fig.update_layout(
                                margin=dict(l=0, r=0, t=5, b=20),
                                height=180,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                showlegend=False,
                                xaxis=dict(showgrid=False, visible=True, color='#64748b', tickfont=dict(size=10)),
                                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#94a3b8', zeroline=False),
                                hovermode='x unified',
                                bargap=0.2
                            )
                            return fig

                        with ex_cols[0]:
                            st.markdown(f"<div style='font-size: 0.85rem; font-weight: 700; color: #10b981; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.5px;'>Surge Breadth (Up >{pct_thresh_txt} in 5D) + 50 SMA</div>", unsafe_allow_html=True)
                            st.plotly_chart(create_surge_chart(full_hist_df['surge_extreme_5d'], 'Surge Count'), use_container_width=True, config={'displayModeBar': False})
                        with ex_cols[1]:
                            st.markdown(f"<div style='font-size: 0.85rem; font-weight: 700; color: #ef4444; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.5px;'>Panic Breadth (Down >{pct_thresh_txt} in 5D) + 50 SMA</div>", unsafe_allow_html=True)
                            st.plotly_chart(create_panic_chart(full_hist_df['panic_extreme_5d'], 'Panic Count', overlay_sma=True), use_container_width=True, config={'displayModeBar': False})
                    
        render_breadth_ui("IN", tab_b_in, "Internal breadth metrics using Nifty Total Market.")
        render_breadth_ui("US", tab_b_us, "Internal breadth metrics using US Equities.")
        
        # --- Sectoral Relative Strength Heatmap ---
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("### 🔭 Sectoral Relative Strength (Top-Down Filter)")
        st.markdown("<div style='color: #cbd5e1; font-size: 0.9rem; margin-top: -10px; margin-bottom: 20px;'>Institutional money flows: comparing Sector Returns vs Nifty 50 to find leaders and laggards.</div>", unsafe_allow_html=True)
        
        rs_period_col, _ = st.columns([1, 3])
        with rs_period_col:
            rs_lookback = st.selectbox("Lookback Period", options=["1 Month", "3 Months"], index=1, key="rs_lookback")
        
        lookback_months = 1 if rs_lookback == "1 Month" else 3
        
        from sector_rs_engine import fetch_sector_rs_data
        with st.spinner(f"Calculating Relative Strength for {lookback_months} Month(s)..."):
            rs_df, bench_ret = fetch_sector_rs_data(lookback_months=lookback_months)
            
        if not rs_df.empty:
            st.markdown(f"<div style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 15px;'>Benchmark (Nifty 50) Absolute Return: <b>{bench_ret:+.2f}%</b></div>", unsafe_allow_html=True)
            
            st.markdown("""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');
            
            .rs-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
                gap: 16px;
                margin-bottom: 30px;
                padding-top: 10px;
            }
            .rs-tile {
                border-radius: 12px;
                padding: 16px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                height: 135px;
                border: 1px solid rgba(255,255,255,0.08);
                transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
                position: relative;
                overflow: hidden;
                font-family: 'Outfit', sans-serif;
            }
            .rs-tile::before {
                content: '';
                position: absolute;
                top: 0; left: 0; right: 0; height: 100%;
                background: linear-gradient(180deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0) 40%);
                pointer-events: none;
            }
            .rs-tile::after {
                content: '';
                position: absolute;
                top: 0; right: 0; width: 60px; height: 60px;
                background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
                border-radius: 50%;
                transform: translate(20px, -20px);
                pointer-events: none;
            }
            .rs-tile:hover {
                transform: translateY(-6px);
                border: 1px solid rgba(255,255,255,0.25);
                z-index: 10;
            }
            .rs-tile-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .rs-tile-title {
                font-size: 0.9rem;
                font-weight: 600;
                color: rgba(255,255,255,0.95);
                line-height: 1.2;
                letter-spacing: 0.3px;
                text-shadow: 0 1px 2px rgba(0,0,0,0.5);
            }
            .rs-tile-abs {
                font-size: 2.2rem;
                font-weight: 800;
                color: #ffffff;
                margin-top: auto;
                margin-bottom: 8px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.4);
                line-height: 1;
                letter-spacing: -1px;
            }
            .rs-tile-rel {
                font-size: 0.75rem;
                font-weight: 600;
                padding: 4px 8px;
                border-radius: 20px;
                background: rgba(0,0,0,0.25);
                display: inline-flex;
                align-items: center;
                gap: 4px;
                backdrop-filter: blur(4px);
                box-shadow: inset 0 1px 1px rgba(255,255,255,0.1);
                color: rgba(255,255,255,0.9);
                width: fit-content;
            }
            </style>
            """, unsafe_allow_html=True)
            
            grid_html = "<div class='rs-grid'>"
            for _, row in rs_df.iterrows():
                sector = row['Sector']
                abs_ret = row['Absolute Return']
                rel_ret = row['Relative Return']
                
                # Determine color based on Relative Return
                if rel_ret >= 10:
                    bg_color = "linear-gradient(135deg, #064e3b, #022c22)"
                    hover_shadow = "rgba(6, 78, 59, 0.6)"
                elif rel_ret >= 5:
                    bg_color = "linear-gradient(135deg, #059669, #064e3b)"
                    hover_shadow = "rgba(5, 150, 105, 0.6)"
                elif rel_ret >= 0:
                    bg_color = "linear-gradient(135deg, #10b981, #059669)"
                    hover_shadow = "rgba(16, 185, 129, 0.6)"
                elif rel_ret > -5:
                    bg_color = "linear-gradient(135deg, #ef4444, #dc2626)"
                    hover_shadow = "rgba(239, 68, 68, 0.6)"
                elif rel_ret > -10:
                    bg_color = "linear-gradient(135deg, #b91c1c, #991b1b)"
                    hover_shadow = "rgba(185, 28, 28, 0.6)"
                else:
                    bg_color = "linear-gradient(135deg, #7f1d1d, #450a0a)"
                    hover_shadow = "rgba(127, 29, 29, 0.6)"
                    
                abs_str = f"+{abs_ret:.1f}%" if abs_ret > 0 else f"{abs_ret:.1f}%"
                
                # Pill logic
                arrow = "🔥" if rel_ret > 5 else "▲" if rel_ret > 0 else "▼" if rel_ret > -5 else "❄️"
                rel_str = f"<span>{arrow}</span> {abs(rel_ret):.1f}% vs NIFTY"
                
                grid_html += f"<div class='rs-tile' style='background: {bg_color}; box-shadow: 0 4px 15px rgba(0,0,0,0.3);' onmouseover=\"this.style.boxShadow='0 12px 30px {hover_shadow}'\" onmouseout=\"this.style.boxShadow='0 4px 15px rgba(0,0,0,0.3)'\"><div class='rs-tile-header'><div class='rs-tile-title'>{sector}</div></div><div class='rs-tile-abs'>{abs_str}</div><div class='rs-tile-rel'>{rel_str}</div></div>"
                
            grid_html += "</div>"
            st.markdown(grid_html, unsafe_allow_html=True)
        else:
            st.warning("Failed to fetch Sectoral Relative Strength data.")

        # --- Market Liquidity (Always Visible underneath Breadth) ---
        st.markdown("<br>", unsafe_allow_html=True)
        tab_l_in, tab_l_us = st.tabs(["🇮🇳 India Equities", "🇺🇸 US Equities (Broad Market)"])
        
        def render_liquidity_ui(market_code, subtitle, unit_label, context_help=None):
            from liquidity_engine import ensure_liquidity_data
            liq_df = ensure_liquidity_data(market_code=market_code, days=1000)
            
            if liq_df is None or liq_df.empty:
                st.info(f"No liquidity data available for {market_code}. Run liquidity_engine.py")
                return
                
            # Determine current regime
            last_turnover = float(liq_df.iloc[-1]['monthly_turnover_k_cr'])
            last_sma = float(liq_df.iloc[-1]['sma_200'])
            diff_pct = ((last_turnover / last_sma) - 1.0) * 100 if last_sma > 0 else 0.0
            
            is_bull = last_turnover > last_sma
            status_color = "#10b981" if is_bull else "#ef4444"
            status_text = "EXPANDING (BULL)" if is_bull else "CONTRACTING (BEAR)"
            
            # Render sleek seamless Header
            st.markdown(f'''
            <div style="display: flex; justify-content: space-between; align-items: flex-end; padding: 10px 5px; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: -15px; margin-top: 6px;">
                <div>
                    <h3 style="margin: 0; color: #f8fafc; font-weight: 600; font-size: 1.15rem; display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.2rem;">💧</span> Institutional Liquidity
                    </h3>
                    <p style="margin: 0; font-size: 0.8rem; color: #94a3b8; margin-top: 2px; font-weight: 400;">{subtitle}</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.65rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px;">Market Regime</div>
                    <div style="font-size: 0.9rem; font-weight: 700; color: {status_color}; letter-spacing: 0.5px; filter: drop-shadow(0 0 4px {status_color});">
                        {status_text} <span style="font-size: 0.75rem; font-weight: 500; opacity: 0.9;">({diff_pct:+.1f}%)</span>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            if context_help:
                st.markdown(f'<div style="font-size: 0.72rem; color: #64748b; margin-top: 18px; margin-bottom: -10px;">💡 {context_help}</div>', unsafe_allow_html=True)
            
            fig = go.Figure()
            
            fill_color = 'rgba(16, 185, 129, 0.04)' if is_bull else 'rgba(239, 68, 68, 0.03)'
            
            # Ambient fill
            fig.add_trace(go.Scatter(
                x=liq_df['date'], y=liq_df['monthly_turnover_k_cr'],
                mode='none',
                fill='tozeroy',
                fillcolor=fill_color,
                showlegend=False,
                hoverinfo='skip'
            ))
            
            is_eq = (unit_label == 'x ADTV')
            val_fmt = ".2f" if is_eq else ".1f"
            metric_name = 'Relative Tide' if is_eq else 'Monthly Turnover'
            
            # Actual Turnover Line (Smooth Spline)
            fig.add_trace(go.Scatter(
                x=liq_df['date'], y=liq_df['monthly_turnover_k_cr'],
                mode='lines',
                line=dict(color='#10b981', width=1.8, shape='spline', smoothing=0.5),
                name=metric_name,
                hovertemplate=f'{metric_name}: <b>%{{y:{val_fmt}}} {unit_label}</b><extra></extra>'
            ))
            
            # SMA Line (Dashed, Smooth, Subtle)
            fig.add_trace(go.Scatter(
                x=liq_df['date'], y=liq_df['sma_200'],
                mode='lines',
                line=dict(color='rgba(239, 68, 68, 0.65)', width=1.5, dash='dash', shape='spline', smoothing=1.3),
                name='200-Day SMA',
                hovertemplate=f'SMA 200: <b>%{{y:{val_fmt}}} {unit_label}</b><extra></extra>'
            ))
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=25, b=0),
                height=280,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True, 
                    gridcolor='rgba(255,255,255,0.02)', 
                    showline=False, 
                    color='#475569',
                    tickfont=dict(size=9),
                    dtick="M3",
                    tickformat="%b %Y",
                    hoverformat="%d %b %Y"
                ),
                yaxis=dict(
                    showgrid=True, 
                    gridcolor='rgba(255,255,255,0.02)', 
                    color='#475569', 
                    zeroline=False,
                    tickfont=dict(size=9)
                ),
                hovermode='x unified',
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1,
                    font=dict(size=9, color='#64748b'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                dragmode='pan'
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': True})
            
        with tab_l_in:
            st.markdown("<div style='margin-top: 6px; margin-bottom: 2px;'>", unsafe_allow_html=True)
            liq_in_mode = st.radio(
                "Market Flow Dimension:",
                [
                    "🚀 Mid & Small-Cap (400 Stocks)",
                    "🏛️ Large-Cap (Nifty 50)",
                    "⚖️ Equal-Weighted Tide",
                    "🌐 Broad Market (Nifty 500)"
                ],
                index=0,
                horizontal=True,
                label_visibility="collapsed",
                key="liq_in_mode_sel"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
            if "Mid & Small" in liq_in_mode:
                render_liquidity_ui(
                    'IN_MIDSMALL',
                    'Nifty Midcap 150 + Smallcap 250 (400 Stocks) Monthly Turnover vs 200-Day SMA',
                    'K cr',
                    context_help='Tracks domestic & growth institutional flows in Midcap 150 & Smallcap 250. Isolates momentum breakouts from large-cap index drag.'
                )
            elif "Large-Cap" in liq_in_mode:
                render_liquidity_ui(
                    'IN_LARGE',
                    'Nifty 50 (Top 50 Heavyweights) Monthly Turnover vs 200-Day SMA',
                    'K cr',
                    context_help='Tracks mega-cap rupee turnover (Reliance, HDFC Bank, ICICI, TCS, etc.). Reflects FII macro allocations and index futures hedging.'
                )
            elif "Equal-Weighted" in liq_in_mode:
                render_liquidity_ui(
                    'IN_EQ_WEIGHT',
                    'Equal-Weighted Relative Turnover Tide (Turnover / 50-day ADTV) vs 200-Day SMA',
                    'x ADTV',
                    context_help='Normalizes each stock against its own 50-day average daily turnover. Every stock gets an equal vote, eliminating market-cap rupee bias.'
                )
            else:
                render_liquidity_ui(
                    'IN',
                    'Nifty 500 Aggregate Monthly Turnover vs 200-Day SMA',
                    'K cr',
                    context_help='Traditional aggregate rupee turnover across all 500 stocks in Nifty 500.'
                )
            
        with tab_l_us:
            render_liquidity_ui(
                'US',
                'US Broad Market (1,300+ Stocks) Total Monthly Turnover vs 200-Day SMA',
                '$B',
                context_help='Broad institutional dollar turnover across US NYSE and NASDAQ equities.'
            )
        
        with tab_rotation:
            st.caption("Intermarket analysis: Tracking global capital rotation and relative strength across major asset classes.")
            
            @st.cache_data(ttl=3600)
            def fetch_rotation_data_v2():
                import asset_rotation
                return asset_rotation.fetch_asset_returns()
                
            with st.spinner("Fetching global asset returns (yfinance)..."):
                rot_df = fetch_rotation_data_v2()
                
            if not rot_df.empty:
                def apply_color(val):
                    if pd.isna(val): return ''
                    c = '#065f46' if val > 15 else '#16a34a' if val > 5 else '#22c55e' if val > 0 else '#991b1b' if val < -10 else '#dc2626' if val < -5 else '#ef4444'
                    return f'background-color: {c}; color: white; font-weight: bold; border-radius: 4px; padding: 2px;'
                    
                format_dict = {c: '{:.1f}%' for c in rot_df.columns if c != 'Asset'}
                styled_df = rot_df.style.format(format_dict, na_rep='-').map(apply_color, subset=rot_df.columns.drop('Asset'))
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
    st.markdown("---")
    
    # --- INSTITUTIONAL THEME TRACKER (O'NEIL NEW HIGHS) ---
    try:
        from theme_engine import calculate_theme_alignment, calculate_portfolio_alignment
        theme_data = calculate_theme_alignment()
        if theme_data and 'top_themes' in theme_data and theme_data['top_themes']:
            alignment_pct, theme_exposure = calculate_portfolio_alignment(theme_data, decisions)
            
            st.markdown("### 🔥 Institutional Theme Tracker (O'Neil New Highs)")
            
            with st.expander("📚 How is this calculated?"):
                st.markdown("""
                **The Philosophy:** Institutional money moves in massive themes. When mutual funds accumulate a sector, they don't just buy one stock; they buy the top 5 or 10 names in that group. By tracking the *density* of 52-week highs within a sector, we are tracking the footprint of massive institutional capital flow.
                
                **The Calculation (No Black Box):**
                1. **Universe:** We scan the Top 750 most liquid stocks on the NSE (The Nifty Total Market).
                2. **Net New Highs:** We analyze the last 5 trading days. For every stock, we check if its high price breached a 250-day (52-week) high. We also count new 52-week lows. We calculate **Net New Highs (New Highs minus New Lows)** to ensure the sector is genuinely trending up, not just volatile.
                3. **Ranking:** We group these 750 stocks into their official NSE Macro-Economic Sectors and rank the sectors by the highest density of Net New Highs.
                4. **Portfolio Alignment:** We scan your current holdings and calculate exactly what percentage of your capital is deployed in these Top 6 leading sectors. If this drops, it's an early warning that your portfolio is falling out of sync with the broader market.
                5. **The Leaders & RS Score:** Under each sector, we print the top 3 specific stocks that made a new high, ranked by their Relative Strength (RS). 
                *Note: The RS Score (0-99) is calculated by aggressively weighting recent momentum: (1M Return × 40%) + (3M Return × 35%) + (6M Return × 25%), and then percentile-ranking the stock against all 750 names in the universe.*
                """)
                
            all_leader_tickers = []
            for theme in theme_data.get('top_themes', [])[:6]:
                for l in theme.get('leaders', []):
                    clean = l['ticker'].replace('.NS', '').replace('.BO', '')
                    all_leader_tickers.append(f"NSE:{clean}")
                    
            if all_leader_tickers:
                tv_string = ",".join(all_leader_tickers)
                with st.expander("📋 Copy TradingView Watchlist"):
                    st.markdown("<span style='font-size: 0.85rem; color: #94a3b8;'>Copy these 18 theme leaders into your TradingView scanner:</span>", unsafe_allow_html=True)
                    st.code(tv_string, language="text")
            
            
            align_color = "#22c55e" if alignment_pct >= 50 else "#eab308" if alignment_pct >= 25 else "#ef4444"
            st.markdown(f"**Portfolio Alignment:** <span style='color: {align_color}; font-weight: bold;'>{alignment_pct:.1f}%</span> of your capital is deployed in the Top 6 Market Themes", unsafe_allow_html=True)
            st.markdown("""
            <style>
            .theme-card {
                background: linear-gradient(145deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
                border: 1px solid rgba(255,255,255,0.05);
                border-radius: 12px;
                padding: 16px;
                height: 100%;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            }
            .theme-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 24px rgba(0,0,0,0.6);
                border: 1px solid rgba(255,255,255,0.15);
            }
            .theme-title {
                font-size: 0.95rem;
                font-weight: 700;
                color: #f8fafc;
                margin-bottom: 4px;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
                overflow: hidden;
                line-height: 1.3;
            }
            .theme-metric {
                font-size: 0.75rem;
                color: #94a3b8;
                margin: 0;
            }
            .theme-exposure {
                font-size: 0.8rem;
                font-weight: 700;
                margin-top: 12px;
                margin-bottom: 12px;
                padding: 4px 8px;
                border-radius: 6px;
                display: inline-block;
            }
            .rs-badge {
                padding: 1px 6px;
                border-radius: 10px;
                font-size: 0.7rem;
                font-weight: bold;
                border: 1px solid;
            }
            .leader-row {
                font-size: 0.8rem;
                margin: 6px 0;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .leader-ticker {
                color: #e2e8f0;
                font-weight: 600;
            }
            </style>
            """, unsafe_allow_html=True)
            
            top_themes = theme_data['top_themes']
            cols = st.columns(6)
            
            rank_colors = ["#fbbf24", "#94a3b8", "#b45309", "#3b82f6", "#3b82f6", "#3b82f6"]
            
            for i, theme in enumerate(top_themes):
                if i < 6:
                    with cols[i]:
                        border_color = rank_colors[i]
                        exp_val = theme_exposure.get(theme['sector'], 0)
                        exp_bg = "rgba(16, 185, 129, 0.15)" if exp_val > 0 else "rgba(239, 68, 68, 0.15)"
                        exp_color = "#34d399" if exp_val > 0 else "#f87171"
                        
                        leaders_html = ""
                        for l in theme['leaders']:
                            rs_val = l['rs']
                            if rs_val >= 90:
                                rs_bg, rs_txt, rs_brd = "rgba(16, 185, 129, 0.2)", "#34d399", "rgba(16, 185, 129, 0.3)"
                            elif rs_val >= 80:
                                rs_bg, rs_txt, rs_brd = "rgba(59, 130, 246, 0.2)", "#60a5fa", "rgba(59, 130, 246, 0.3)"
                            else:
                                rs_bg, rs_txt, rs_brd = "rgba(234, 179, 8, 0.2)", "#facc15", "rgba(234, 179, 8, 0.3)"
                                
                            leaders_html += f"<div class='leader-row'><span class='leader-ticker'>✦ {l['ticker']}</span><span class='rs-badge' style='background: {rs_bg}; color: {rs_txt}; border-color: {rs_brd};'>{rs_val:.0f} RS</span></div>"
                            
                        st.markdown(f"""<div class='theme-card' style='border-top: 4px solid {border_color};'>
<div class='theme-title'>{i+1}. {theme['sector']}</div>
<p class='theme-metric'>{theme['net_highs']} Net New Highs</p>
<div class='theme-exposure' style='background: {exp_bg}; color: {exp_color};'>You own: {exp_val:.1f}%</div>
<hr style="margin: 8px 0; border: none; border-top: 1px solid rgba(255,255,255,0.05);">
<p style="font-size: 0.7rem; color: #64748b; margin-bottom: 8px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Theme Leaders</p>
{leaders_html}
</div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    except Exception as e:
        print(f"Failed to render Theme Tracker: {e}")

    st.markdown("### 🛩️ Command Cockpit")

    with st.expander("ℹ️ How to Interpret Portfolio Heat & Action Playbook", expanded=False):
        st.markdown(r"""
        #### 🔥 What is Total Open Portfolio Heat?
        **Portfolio Heat** measures the exact percentage of your total account NAV at risk if every active position were to fall to its active stop loss order simultaneously.
        
        Unlike **Invested Exposure** (which tells you *how much capital is deployed*), **Portfolio Heat** tells you *how much real capital you could lose* if the market suddenly reversed. A portfolio can be 100% invested, but if your winning positions have stops trailed into profit or at breakeven, your actual Open Heat can be low and safe!
        
        ---
        
        #### 🧭 The 3 Heat Regimes & Action Guide
        
        | Heat Level | Status | Market Stance | Immediate Action Required |
        |:---|:---:|:---:|:---|
        | **$\le$ 1.5% NAV** | 🟢 **Controlled** | Conservative / Safe | **Green light for fresh buys.** Capital defense is intact. Feel free to initiate high-conviction CANSLIM pivot breakouts or add to proven leaders. |
        | **1.5% – 2.5% NAV** | 🟡 **Sweet Spot** | Optimal Aggression | **Normal execution mode.** Ideal balance of offensive participation and downside protection. Maintain trailing stops along 21 EMA / 10 EMA. Only buy by replacing weaker positions. |
        | **> 2.5% NAV** | 🔴 **High Heat** | Dangerous Over-exposure | **🚨 FREEZE ALL NEW BUYS.** You are over-extended to simultaneous market gap-down risk. Follow the 3-step cool-down protocol below. |
        
        ---
        
        #### 🛡️ The 3-Step Heat Cool-Down Protocol (When Heat > 2.5%)
        1. **Execute All `EXIT` Signals**: Sell stocks that have broken their stops or key moving averages (flagged with `HIT` badges in the table). Do not hope, hesitate, or bargain.
        2. **Trim Trailing Laggers**: Trim positions showing `Port Hit > 1.0%` or losing momentum.
        3. **Move Stops to Breakeven**: On any position with $\ge 8\%$ gain, move stops to Cost Defense ($0\%$ capital risk).
        
        ---
        
        #### 📌 Individual Position Heat in the Holdings Table
        * **Target per trade**: A disciplined trade risks no more than **$0.40\% - 0.60\%$** of total portfolio NAV.
        * **Formula**: $\text{Position Heat} = \frac{(\text{Current Price} - \text{Active Stop}) \times \text{Shares}}{\text{Total NAV}} \times 100$
        * **Stop Floor**: Trailing maximum of:
          1. **8% Hard Base Stop** (O'Neil maximum capital loss limit)
          2. **Cost Defense** (Automatic breakeven floor once a stock rallies $\ge 8\%$)
          3. **Chandelier ATR Stop** (Trailing 3.5 $\times$ ATR(21) below recent 20-day high)
        * **`HIT` Badge**: If a stock trades below its active stop, it is flagged with a red `HIT` badge. It represents a pending market exit order.
        
        ---
        
        #### 📉 Account Drawdown, Sortino (63D) & Calmar Ratio
        
        *   **Account Drawdown (▼ %)**:
            *   Measures the current peak-to-trough decline of your total portfolio NAV from its all-time high.
            *   🟢 **< 5% (Controlled)**: Normal equity fluctuation in a healthy trending market. Normal execution.
            *   🟡 **5% – 10% (Caution)**: Moderate retracement. Slow down new breakout buys; tighten stops along 21 EMA / breakeven.
            *   🔴 **> 10% (Severe)**: Market regime shift or systematic portfolio weakness. Halt aggressive entries and prioritize capital preservation.

        *   **Sortino Ratio (63D Rolling / ~3 Months)**:
            *   **Formula**: $\frac{\text{Portfolio Annualized Return} - \text{Risk-Free Rate (6.5\%)}}{\text{Downside Deviation}}$
            *   Unlike Sharpe Ratio (which penalizes upside explosive gains), Sortino **only penalizes volatility on down days**.
            *   **How to interpret**:
                *   🏆 **$\ge$ 2.0 (Elite / Institutional Alpha)**: Outstanding asymmetric upside with negligible downside drag. Your winners are expanding and drawdowns on red days are tightly contained.
                *   🟢 **1.5 – 1.99 (Strong)**: Healthy momentum engine. Downside risk is well managed.
                *   🟡 **0.8 – 1.49 (Average)**: Adequate, but down days are cutting into your equity. Review trade sizing.
                *   🔴 **< 0.8 (High Downside Drag)**: Downside volatility is eating your alpha. Check if losers are being held too long.

        *   **Calmar Ratio (Lifetime)**:
            *   **Formula**: $\frac{\text{Annualized Return (CAGR)}}{\text{Maximum Lifetime Drawdown}}$
            *   Measures how many dollars of annual compound gain you generate per dollar of historical peak-to-trough pain.
            *   **How to interpret**:
                *   🏆 **$\ge$ 2.0 (Exceptional)**: The system makes more than 2x its worst drawdown each year.
                *   🟢 **1.0 – 1.99 (Solid)**: Good reward-to-risk payoff.
                *   🔴 **< 1.0 (Strained)**: Maximum drawdown is too deep relative to compound return; reduce sizing or tighten defense.
        """)
        
    # Show gap between current exposure and target exposure
    target_exp = int(exposure_guide['level'].split('%')[0].split('-')[-1].strip()) if '%' in exposure_guide['level'] else 100
    gap = target_exp - risk_exposure
    gap_color = '#10b981' if gap >= 0 else '#ef4444'
    gap_text = f"Can deploy {gap:.1f}% more capital" if gap > 0 else f"Over-exposed by {abs(gap):.1f}%" if gap < 0 else "Optimal sizing"
    urgent_color = '#ef4444' if (exit_count > 0 or trim_count > 0) else '#10b981'
    pulse_class = "pulse-animation" if exit_count > 0 else ""

    st.markdown(f"""
    <style>
    .pulse-animation h2 span.exit-text {{
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0% {{ opacity: 1; }}
        50% {{ opacity: 0.5; color: #fca5a5; }}
        100% {{ opacity: 1; }}
    }}
    </style>
    """, unsafe_allow_html=True)
    
    import equity_tracker
    equity_snap = equity_tracker.get_equity_drawdown(total_value)
    
    cockpit1, cockpit2, cockpit3, cockpit4, cockpit5 = st.columns(5)
    
    heat_color = "#10b981" if total_open_heat_pct <= 1.5 else "#f59e0b" if total_open_heat_pct <= 2.5 else "#ef4444"
    heat_badge_text = "🟢 Controlled" if total_open_heat_pct <= 1.5 else "🟡 Sweet Spot" if total_open_heat_pct <= 2.5 else "🔴 High Heat"
    heat_pulse = "pulse-animation" if total_open_heat_pct > 2.5 else ""
    
    with cockpit1:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #3b82f6; margin-bottom: 2rem;">
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">Risk Exposure</p>
            <h2 style="color: white; margin: 0.5rem 0; font-family: 'JetBrains Mono', monospace;">{risk_exposure:.1f}% Invested</h2>
            <p style="color: #cbd5e1; font-size: 0.8rem; margin: 0; font-family: 'Inter', sans-serif;">Total Value: ₹{total_value:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

    with cockpit2:
        st.markdown(f"""
        <div class="metric-card {heat_pulse}" style="border-top: 4px solid {heat_color}; margin-bottom: 2rem;">
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">Total Open Heat</p>
            <h2 style="color: {heat_color}; margin: 0.5rem 0; font-family: 'JetBrains Mono', monospace;">{total_open_heat_pct:.2f}% NAV</h2>
            <p style="color: #cbd5e1; font-size: 0.8rem; margin: 0; font-family: 'Inter', sans-serif;">Risk: ₹{total_open_heat_dollars:,.0f} | <span style="color: {heat_color}; font-weight: 600;">{heat_badge_text}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
    with cockpit3:
        st.markdown(f"""
        <div class="metric-card {pulse_class}" style="border-top: 4px solid {urgent_color}; margin-bottom: 2rem;">
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">Immediate Actions</p>
            <h2 style="color: white; margin: 0.5rem 0; font-family: 'JetBrains Mono', monospace;">
                <span class="exit-text" style="color: #ef4444;">{exit_count} Exit</span> <span style="color: #475569;">|</span> 
                <span style="color: #f59e0b;">{trim_count} Trim</span>
            </h2>
            <p style="color: #cbd5e1; font-size: 0.8rem; margin: 0; font-family: 'Inter', sans-serif;">Tracking {add_count} Add setups</p>
        </div>
        """, unsafe_allow_html=True)
        
    with cockpit4:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid {gap_color}; margin-bottom: 2rem;">
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">Exposure vs Target</p>
            <h2 style="color: {gap_color}; margin: 0.5rem 0; font-family: 'JetBrains Mono', monospace;">{gap:+.1f}% Delta</h2>
            <p style="color: #cbd5e1; font-size: 0.8rem; margin: 0; font-family: 'Inter', sans-serif;">{gap_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with cockpit5:
        sortino_val = equity_snap.get('sortino_63d', 0.0)
        calmar_val = equity_snap.get('calmar', 0.0)
        sort_col = "#10b981" if sortino_val >= 1.5 else "#f59e0b" if sortino_val >= 0.8 else "#ef4444"
        
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid {equity_snap['color']}; margin-bottom: 2rem;">
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">Account Drawdown</p>
            <h2 style="color: {equity_snap['color']}; margin: 0.5rem 0; font-family: 'JetBrains Mono', monospace;">▼ {equity_snap['drawdown_pct']:.1f}%</h2>
            <p style="color: #cbd5e1; font-size: 0.75rem; margin: 0; font-family: 'Inter', sans-serif; line-height: 1.3;">
                Sortino (63D): <span style="color: {sort_col}; font-weight: 700;">{sortino_val:.2f}</span> | Calmar: <span style="color: #38bdf8; font-weight: 700;">{calmar_val:.2f}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    @st.cache_data(ttl=3600)
    def fetch_macro_regime_cached_v2():
        import macro_regime
        return macro_regime.get_macro_snapshot()
    
    with st.spinner("Analyzing Macro Risk Regime..."):
        macro_snap = fetch_macro_regime_cached_v2()
        regime = macro_snap['regime']
        n50 = macro_snap.get('nifty50', {})
        n500 = macro_snap.get('nifty500', {})
        sp500 = macro_snap.get('sp500', {})
        nasdaq = macro_snap.get('nasdaq', {})
        
        def get_sma_span(close, sma):
            if close < sma:
                return f'<span class="blink-down">{sma:.0f}</span>'
            return f'<span class="safe-up">{sma:.0f}</span>'

        def build_index_row(title, data):
            if not data: return ""
            c = data.get('close', 0)
            dd = data.get("drawdown", 0)
            dd_color = "#ef4444" if dd < -4 else "#10b981"
            return (
                f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">'
                f'<td style="padding: 0.8rem 0; font-weight: bold; color: #f8fafc; text-transform: uppercase; font-size: 0.85rem;">{title}</td>'
                f'<td style="text-align: right; color: white;">{c:.0f}</td>'
                f'<td style="text-align: right;">{get_sma_span(c, data.get("ema_21", 0))}</td>'
                f'<td style="text-align: right;">{get_sma_span(c, data.get("sma_50", 0))}</td>'
                f'<td style="text-align: right;">{get_sma_span(c, data.get("sma_100", 0))}</td>'
                f'<td style="text-align: right;">{get_sma_span(c, data.get("sma_150", 0))}</td>'
                f'<td style="text-align: right;">{get_sma_span(c, data.get("sma_200", 0))}</td>'
                f'<td style="text-align: right; color: {dd_color}; font-weight: bold;">{dd:.1f}%</td>'
                f'</tr>'
            )

        rows_html = ""
        rows_html += build_index_row("Nifty 50", n50)
        rows_html += build_index_row("Nifty 500", n500)
        rows_html += build_index_row("S&P 500", sp500)
        rows_html += build_index_row("Nasdaq", nasdaq)
        
        table_html = (
            f'<div class="glass-panel" style="padding: 1.5rem; overflow-x: auto; min-width: 450px;">'
            f'<table style="width: 100%; border-collapse: collapse; font-family: \'JetBrains Mono\', monospace; font-size: 0.85rem;">'
            f'<thead>'
            f'<tr style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid rgba(255,255,255,0.1);">'
            f'<th style="text-align: left; padding-bottom: 0.8rem;">Index</th>'
            f'<th style="text-align: right; padding-bottom: 0.8rem;">Close</th>'
            f'<th style="text-align: right; padding-bottom: 0.8rem;">21 EMA</th>'
            f'<th style="text-align: right; padding-bottom: 0.8rem;">50 SMA</th>'
            f'<th style="text-align: right; padding-bottom: 0.8rem;">100 SMA</th>'
            f'<th style="text-align: right; padding-bottom: 0.8rem;">150 SMA</th>'
            f'<th style="text-align: right; padding-bottom: 0.8rem;">200 SMA</th>'
            f'<th style="text-align: right; padding-bottom: 0.8rem;">DD%</th>'
            f'</tr>'
            f'</thead>'
            f'<tbody>'
            f'{rows_html}'
            f'</tbody>'
            f'</table>'
            f'</div>'
        )
        
    st.markdown("### 🌍 Macro-Adjusted Risk Rules")
    
    css_animation = f"""
    <style>
    @keyframes blink-red-macro {{
        0% {{ color: #ef4444; text-shadow: 0 0 5px #ef4444; }}
        50% {{ color: #fca5a5; text-shadow: 0 0 15px #ef4444; }}
        100% {{ color: #ef4444; text-shadow: 0 0 5px #ef4444; }}
    }}
    .blink-down {{ animation: blink-red-macro 1.5s infinite; font-weight: bold; }}
    .safe-up {{ color: #10b981; font-weight: bold; text-shadow: 0 0 8px rgba(16, 185, 129, 0.6); }}
    
    .glass-panel {{
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: inset 0 0 20px rgba(255,255,255,0.02), 0 8px 32px rgba(0, 0, 0, 0.3);
        flex: 1;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .glass-panel:hover {{ 
        transform: translateY(-4px) scale(1.02); 
        border: 1px solid rgba(255, 255, 255, 0.2); 
        box-shadow: inset 0 0 20px rgba(255,255,255,0.05), 0 12px 40px rgba(0, 0, 0, 0.4);
    }}
    .macro-container {{
        position: relative;
        overflow: hidden;
        background: #0b1121;
        border: 1px solid rgba(255,255,255,0.05);
        border-left: 6px solid {regime['color']};
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        display: flex;
        gap: 3rem;
        flex-wrap: wrap;
        box-shadow: 0 15px 40px rgba(0,0,0,0.6);
    }}
    .macro-glow {{
        position: absolute;
        top: -50%;
        left: -20%;
        width: 150%;
        height: 150%;
        background: radial-gradient(circle at 80% 20%, {regime['color']}15 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
    }}
    .macro-content {{
        position: relative;
        z-index: 1;
        flex: 2;
        min-width: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .macro-data {{
        position: relative;
        z-index: 1;
        flex: 1.5;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 1.5rem;
        min-width: 450px;
    }}
    .stat-box {{
        background: rgba(0,0,0,0.3);
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.05);
        text-align: left;
        flex: 1;
        transition: transform 0.2s;
    }}
    .stat-box:hover {{
        transform: translateY(-2px);
        background: rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.1);
    }}
    </style>
    """
    
    st.markdown(f"""
    {css_animation}
    <div class="macro-container">
        <div class="macro-glow"></div>
        <div class="macro-content">
            <h2 style="color: {regime['color']}; margin: 0 0 1rem 0; font-family: 'JetBrains Mono', monospace; text-shadow: 0 0 20px {regime['color']}88; font-size: 2.5rem; letter-spacing: 3px; display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 2rem;">⚡</span> REGIME: {regime['level']}
            </h2>
            <p style="color: #94a3b8; margin: 0 0 2.5rem 0; font-size: 1.15rem; line-height: 1.6; font-weight: 300;">{regime['desc']}</p>
            <div style="display: flex; gap: 1.5rem; flex-wrap: wrap;">
                <div class="stat-box">
                    <span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 2px;">🎯 Base Stop</span>
                    <h3 style="color: white; margin: 0.8rem 0 0 0; font-family: 'JetBrains Mono', monospace; font-size: 1.8rem;">{regime['bsl']}</h3>
                </div>
                <div class="stat-box">
                    <span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 2px;">🛡️ Trailing Stop</span>
                    <h3 style="color: white; margin: 0.8rem 0 0 0; font-family: 'JetBrains Mono', monospace; font-size: 1.8rem;">{regime['tsl']}</h3>
                </div>
                <div class="stat-box">
                    <span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 2px;">💥 Max Port Hit</span>
                    <h3 style="color: white; margin: 0.8rem 0 0 0; font-family: 'JetBrains Mono', monospace; font-size: 1.8rem;">{regime['max_port_hit']}%</h3>
                </div>
            </div>
        </div>
        <div class="macro-data">
            {table_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # Remove old decision generation loop since it was moved above for the cockpit
    
    # 1. EMA Data Audit (Top Priority Warning)
    if audit_warnings:
        with st.expander("🔍 Data Quality Audit (Click to expand)", expanded=False):
            st.warning("Insufficient historical data detected for some symbols. EMA calculations (esp 50 SMA) may be less precise.")
            for warn in audit_warnings:
                st.markdown(warn)
    
    # 2. Urgent Actions Alert
    urgent = [d for d in decisions if d['action'] in [ACTION_EXIT, ACTION_TRIM]]
    if urgent:
        st.markdown("### ⚠️ Urgent Actions Required")
        st.markdown("""
        <style>
        .urgent-scroll-container::-webkit-scrollbar {
            width: 6px;
        }
        .urgent-scroll-container::-webkit-scrollbar-track {
            background: transparent;
        }
        .urgent-scroll-container::-webkit-scrollbar-thumb {
            background-color: #4b5563;
            border-radius: 10px;
        }
        .urgent-item {
            transition: all 0.2s ease;
        }
        .urgent-item:hover {
            transform: translateX(4px);
            box-shadow: -4px 0 15px rgba(239, 68, 68, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        urgent_html = ""
        for d in urgent:
            action_color = '#ef4444' if d['action'] == ACTION_EXIT else '#f59e0b'
            rgb_val = ','.join(str(int(action_color[i:i+2], 16)) for i in (1, 3, 5))
            urgent_html += f"""<div class="urgent-item" style="background: linear-gradient(90deg, rgba({rgb_val}, 0.15) 0%, rgba(15, 23, 42, 0.6) 100%); 
border-left: 4px solid {action_color}; 
border-top: 1px solid rgba(255,255,255,0.05);
border-right: 1px solid rgba(255,255,255,0.05);
border-bottom: 1px solid rgba(255,255,255,0.05);
padding: 1.2rem; 
border-radius: 8px; 
margin-bottom: 0.5rem;
display: flex;
flex-direction: column;
gap: 0.3rem;">
<div style="display: flex; align-items: center; justify-content: space-between;">
<strong style="font-size: 1.1rem; color: #f8fafc; letter-spacing: 1px;">{d['ticker']}</strong>
<span style="background: rgba({rgb_val}, 0.2); color: {action_color}; padding: 0.2rem 0.8rem; border-radius: 20px; font-weight: 700; font-size: 0.8rem; letter-spacing: 1px;">{d['action']}</span>
</div>
<span style="color: #94a3b8; font-size: 0.95rem; display: flex; align-items: center; gap: 0.5rem;">
<span style="color: {action_color};">↳</span> {d['reason']}
</span>
</div>"""
            
        container_html = f"""<div class="urgent-scroll-container" style="max-height: 280px; overflow-y: auto; padding-right: 12px; margin-bottom: 1.5rem; display: flex; flex-direction: column; border-radius: 8px; scrollbar-width: thin; scrollbar-color: #4b5563 transparent;">
{urgent_html}
</div>"""
        st.markdown(container_html, unsafe_allow_html=True)
        st.markdown("---")
        

    # 3. Averaging Up Opportunities & Tom Hougaard "Flowers vs. Weeds" Zero-New-Heat Reallocator
    add_opps = [d for d in decisions if d['add_on_ready']]
    try:
        from brain_synthesis_engine import compute_hougaard_swap_directives
        hougaard_plan = compute_hougaard_swap_directives(decisions, portfolio_df, market_data, total_value)
    except Exception as e:
        print(f"[app.py] Hougaard reallocator warning: {e}")
        hougaard_plan = {'flowers': [], 'weeds': [], 'swap_pairs': [], 'total_locked_cushion_inr': 0.0, 'total_weed_capital_inr': 0.0}

    if add_opps or hougaard_plan.get('flowers'):
        st.markdown("### 🎯 Averaging Up & Tom Hougaard 'Flowers vs. Weeds' Zero-New-Heat Reallocator")
        st.success(
            "🏆 **Best Loser Wins (Tom Hougaard) Protocol:** Never fund a loser; pyramid into coiling winners ('Flowers') "
            "using **locked-in profit cushion** above your active stop, funded by trimming stagnant 'Weeds' (`Dead Money` / lagging RS) "
            "so **Net New Portfolio Heat = +0.00% NAV**."
        )

        if add_opps:
            _bse_card_map = {}
            try:
                if os.path.exists('bse_mapping.json'):
                    with open('bse_mapping.json', 'r') as _f:
                        _bse_card_map.update({str(v): k for k, v in json.load(_f).items()})
                if os.path.exists('bse_yf_mapping.json'):
                    with open('bse_yf_mapping.json', 'r') as _f:
                        _bse_card_map.update({str(k): str(v).replace('.BO', '').replace('.NS', '') for k, v in json.load(_f).items()})
            except Exception:
                pass

            cols = st.columns(len(add_opps)) if len(add_opps) <= 3 else st.columns(3)
            for i, d in enumerate(add_opps):
                ma_d = d.get('ma_dna', {}) or {}
                dna_b = ma_d.get('dna_badge', '⚡ 21-EMA Low')
                e21_l = d.get('ema_21_low', 0.0) or ma_d.get('ema_21_low', 0.0)
                active_sp = float(ma_d.get('active_spine_price', 0.0) or e21_l)
                sp_lbl = ma_d.get('recommended_spine', '21-EMA Low')
                free_sh = int(d.get('free_pyramid_shares', 0) or 0)
                lock_inr = float(d.get('locked_profit_inr', 0.0) or 0.0)
                card_tk = _bse_card_map.get(str(d['ticker']), str(d['ticker']))
                houg_line = (
                    f"<div style='margin-top:6px; padding:5px 8px; background:rgba(16,185,129,0.18); border-radius:6px; font-size:0.78rem; color:#a7f3d0;'>"
                    f"🔥 <b>House-Money Cushion:</b> Up to +{free_sh} sh (₹{lock_inr:,.0f} locked | {sp_lbl} Stop ₹{active_sp:,.1f})</div>"
                ) if free_sh > 0 else (
                    f"<div style='margin-top:6px; font-size:0.76rem; color:#6ee7b7;'>{sp_lbl} Floor: ₹{active_sp:,.1f} • {dna_b}</div>"
                )
                with cols[i % 3]:
                    st.markdown(f"""
                    <div style="background: #022c22; border: 1px solid #10b981; border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h3 style="color: #10b981; margin:0;">{card_tk} 🟢</h3>
                            <span style="font-size:0.72rem; background:rgba(255,255,255,0.1); color:#d1fae5; padding:2px 8px; border-radius:999px;">{dna_b}</span>
                        </div>
                        <p style="color: #d1fae5; font-size: 0.88rem; margin: 6px 0 4px 0;">{d['reason']}</p>
                        {houg_line}
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #6ee7b7; margin-top: 0.5rem;">
                            <span>RS: {d.get('rs_score', 0) or 0:.0f}</span>
                            <span>ATR: {d.get('atr_state')}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        swap_pairs = hougaard_plan.get('swap_pairs', [])
        if swap_pairs:
            with st.expander("🔄 **View Exact Zero-New-Heat Weed-to-Flower Reallocation Orders (0.00% Net New NAV Risk & Cash-Matched)**", expanded=True):
                swap_rows_html = []
                for sp in swap_pairs:
                    fl = sp['flower']
                    wd = sp.get('weed')
                    swap_sh = int(fl.get('swap_add_shares', fl.get('financed_add_shares', 0)))
                    swap_cap = float(fl.get('swap_add_capital_inr', fl.get('add_capital_inr', 0.0)))
                    max_c_sh = int(fl.get('max_cushion_shares', fl.get('financed_add_shares', 0)))
                    sp_name = str(fl.get('spine_name', '21-EMA Low'))
                    cush_tag = "House Money" if fl.get('is_strict_house_money') else "50% Unrealized"
                    if wd:
                        weed_html = (
                            f'<div style="color:#fca5a5; font-weight:700; font-family:\'JetBrains Mono\',monospace; font-size:0.78rem;">'
                            f'✂️ Sell {wd["sell_shares"]} sh <b>{wd["ticker"]}</b> @ ₹{wd["current_price"]:,.1f} '
                            f'<span style="color:#34d399;">(Frees ₹{wd["freed_cash_inr"]:,.0f})</span></div>'
                            f'<div style="color:#94a3b8; font-size:0.71rem; margin-top:2px;">{wd["weed_reason"]}</div>'
                        )
                    else:
                        weed_html = '<span style="color:#94a3b8; font-size:0.76rem;">💵 Funded via Available Cash + House-Money Cushion</span>'

                    cap_sub = (
                        f'<div style="color:#64748b; font-size:0.68rem; margin-top:2px;">Max Cushion Cap: +{max_c_sh:,} sh</div>'
                        if max_c_sh > swap_sh else ''
                    )
                    swap_rows_html.append(
                        f'<tr style="border-bottom:1px solid rgba(255,255,255,0.06);">'
                        f'<td style="padding:10px 10px; white-space:nowrap;"><span style="color:#10b981; font-weight:800; font-family:\'JetBrains Mono\',monospace; font-size:0.84rem;">{fl["ticker"]}</span> '
                        f'<span style="color:#cbd5e1; font-size:0.75rem;">(+{fl["pnl_pct"]:.1f}% | RS {fl["rs_score"]:.0f})</span></td>'
                        f'<td style="padding:10px 8px; white-space:nowrap;"><span style="background:rgba(30,41,59,0.9); color:#e2e8f0; border:1px solid rgba(148,163,184,0.3); padding:2px 8px; border-radius:999px; font-size:0.72rem; font-weight:700;">{fl["dna_badge"]}</span></td>'
                        f'<td style="padding:10px 8px; white-space:nowrap; font-family:\'JetBrains Mono\',monospace; font-size:0.78rem; color:#f8fafc;">₹{fl["locked_cushion_inr"]:,.0f} <span style="color:#94a3b8; font-size:0.7rem;">({cush_tag})</span></td>'
                        f'<td style="padding:10px 8px; white-space:nowrap; font-family:\'JetBrains Mono\',monospace;"><div style="color:#34d399; font-weight:800; font-size:0.82rem;">+{swap_sh:,} sh (₹{swap_cap:,.0f})</div>{cap_sub}</td>'
                        f'<td style="padding:10px 8px; white-space:nowrap; font-family:\'JetBrains Mono\',monospace; font-size:0.78rem; color:#fbbf24;">{sp_name}: <b>₹{fl["new_stop_price"]:,.1f}</b></td>'
                        f'<td style="padding:8px 10px; min-width:280px;">{weed_html}</td>'
                        f'<td style="padding:10px 8px; white-space:nowrap;"><span style="background:rgba(16,185,129,0.18); color:#34d399; border:1px solid rgba(16,185,129,0.4); padding:3px 8px; border-radius:6px; font-weight:800; font-family:\'JetBrains Mono\',monospace; font-size:0.74rem;">0.00% NAV • ₹0 Cash</span></td>'
                        f'</tr>'
                    )

                swap_table_html = (
                    '<div style="overflow-x:auto; border:1px solid rgba(16,185,129,0.28); border-radius:10px; background:rgba(15,23,42,0.78); margin-bottom:6px;">'
                    '<table style="width:100%; border-collapse:collapse; text-align:left;">'
                    '<thead><tr style="background:rgba(2,44,34,0.75); color:#a7f3d0; font-size:0.68rem; font-weight:800; text-transform:uppercase; letter-spacing:0.6px; border-bottom:1px solid rgba(16,185,129,0.25);">'
                    '<th style="padding:9px 10px;">🌸 Winner to Pyramid (Flower)</th>'
                    '<th style="padding:9px 8px;">MA Ride Spine</th>'
                    '<th style="padding:9px 8px;">Locked Cushion</th>'
                    '<th style="padding:9px 8px;">➕ Cash-Matched Swap Add</th>'
                    '<th style="padding:9px 8px;">🛡️ Active Spine Stop</th>'
                    '<th style="padding:9px 10px;">🥀 Stagnant Holding to Trim (Weed Source)</th>'
                    '<th style="padding:9px 8px;">Net New Heat</th>'
                    '</tr></thead>'
                    f'<tbody>{"".join(swap_rows_html)}</tbody></table></div>'
                )
                st.markdown(swap_table_html, unsafe_allow_html=True)
        st.markdown("---")
        
    # 3.5 Sector Allocation & Concentration Risk
    st.markdown("### 📊 Sector Allocation & Concentration Risk")
    
    with st.spinner("Mapping portfolio sectors..."):
        fundas_cache = get_all_fundamentals_cache()
        sector_alloc = []
        for d in decisions:
            ticker = d['ticker']
            industry = industry_map.get(ticker)
            if not industry:
                industry = get_yfinance_industry(ticker)
            
            # Fetch broad sector classification for concentration analysis
            sector, industry_detail = get_yfinance_sector_and_industry(ticker)
            
            alloc = d['close_price'] * d['quantity']
            if alloc > 0:
                mc_val = fundas_cache.get(ticker, {}).get('market_cap', 0.0)
                if mc_val == 0.0 and not ticker.endswith('.NS'):
                    mc_val = fundas_cache.get(f"{ticker}.NS", {}).get('market_cap', 0.0)
                if mc_val == 0.0 and ticker.endswith('.NS'):
                    mc_val = fundas_cache.get(ticker, {}).get('market_cap', 0.0)
                if not mc_val:
                    mc_val = fundas_cache.get(f"{ticker}.NS", {}).get('market_cap', 0.0)
                if not mc_val:
                    mc_val = fundas_cache.get(ticker.replace('.NS', ''), {}).get('market_cap', 0.0)
                
                # Try handling SME tickers
                if not mc_val and ('-SM' in ticker or '-ST' in ticker):
                    base_ticker = ticker.split('-')[0]
                    mc_val = fundas_cache.get(f"{base_ticker}.NS", {}).get('market_cap', 0.0)
                
                # Try handling BSE codes (approximate missing by converting to NSE if possible)
                if not mc_val and ticker.isdigit():
                    try:
                        if os.path.exists('bse_mapping.json'):
                            with open('bse_mapping.json', 'r') as f:
                                bse_map = json.load(f)
                                rev_map = {str(v): k for k, v in bse_map.items()}
                                nse_t = rev_map.get(ticker)
                                if nse_t:
                                    mc_val = fundas_cache.get(f"{nse_t}.NS", {}).get('market_cap', 0.0)
                    except Exception:
                        pass
                    
                sector_alloc.append({
                    'Ticker': ticker, 
                    'Sector': sector,
                    'Industry': industry_detail if industry_detail != 'Other / ETF' else industry,
                    'Allocation': alloc,
                    'Market_Cap': (mc_val / 10000000.0) if mc_val else 0.0
                })
    
    alloc_df = pd.DataFrame(sector_alloc)
    if not alloc_df.empty:
        total_alloc = alloc_df['Allocation'].sum()
        alloc_df['Percent'] = (alloc_df['Allocation'] / total_alloc) * 100
        
        # Soft, premium jewel tones for dark mode
        premium_colors = ['#3b82f6', '#6366f1', '#14b8a6', '#f59e0b', '#ec4899', '#8b5cf6', '#10b981', '#f43f5e']
        
        # Calculate Sector summaries and generate matched color mapping
        sector_summary = alloc_df.groupby('Sector')['Allocation'].sum().reset_index()
        sector_summary['Percent'] = (sector_summary['Allocation'] / total_alloc) * 100
        sector_summary = sector_summary.sort_values(by='Percent', ascending=False)
        
        unique_sectors = alloc_df['Sector'].unique()
        color_map = {sec: premium_colors[i % len(premium_colors)] for i, sec in enumerate(unique_sectors)}
        color_map['(?)'] = '#334155' # Fallback for unknown
        
        cards_html = ""
        for idx, row in sector_summary.iterrows():
            color = color_map.get(row['Sector'], '#94a3b8')
            cards_html += f"""<div style="background: linear-gradient(135deg, rgba(15,23,42,0.4) 0%, rgba(15,23,42,0.7) 100%); border: 1px solid rgba(255,255,255,0.05); border-top: 3px solid {color}; border-radius: 8px; padding: 1rem; min-width: 180px; flex: 0 0 auto; display: flex; flex-direction: column; gap: 0.2rem; box-shadow: 0 8px 20px rgba(0,0,0,0.15); backdrop-filter: blur(10px);">
<span style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{row['Sector']}</span>
<span style="color: #f8fafc; font-size: 1.6rem; font-weight: 700; letter-spacing: -0.5px;">{row['Percent']:.1f}%</span>
<span style="color: {color}; font-size: 0.9rem; font-weight: 600;">₹{row['Allocation']:,.0f}</span>
</div>"""
            
        container_html = f"""<div style="display: flex; gap: 1rem; overflow-x: auto; padding-bottom: 1rem; margin-bottom: 1rem; scrollbar-width: thin; scrollbar-color: #4b5563 transparent; -webkit-overflow-scrolling: touch;">
{cards_html}
</div>"""
        st.markdown(container_html, unsafe_allow_html=True)
        
        fig_tree = px.treemap(
            alloc_df,
            path=['Sector', 'Industry', 'Ticker'],
            values='Allocation',
            color='Sector',
            color_discrete_map=color_map
        )
        
        fig_tree.update_traces(
            textinfo="label+percent parent",
            textfont=dict(family="system-ui, -apple-system, sans-serif", size=14, color="#ffffff"),
            marker=dict(line=dict(color='#0f172a', width=2)), # Thinner, cleaner dark borders
            tiling=dict(pad=4), # Adds beautiful spacing between boxes
            hovertemplate="<b style='font-size:16px;'>%{label}</b><br>Value: ₹%{value:,.2f}<br>Share: %{percentParent:.1%}<extra></extra>"
        )
        
        fig_tree.update_layout(
            height=500, 
            margin=dict(l=0, r=0, t=10, b=10), 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            showlegend=False
        )
        
        st.plotly_chart(fig_tree, use_container_width=True)

    if not alloc_df.empty:
        # Separate cash-equivalent holdings from equity holdings (Strip spaces just in case)
        alloc_df['is_cash'] = alloc_df['Ticker'].astype(str).str.strip().str.upper().isin(CASH_EQUIVALENT_TICKERS)
        cash_df = alloc_df[alloc_df['is_cash']]
        equity_df = alloc_df[~alloc_df['is_cash']]
        
        cash_pct = cash_df['Percent'].sum() if not cash_df.empty else 0.0
        equity_total = equity_df['Allocation'].sum()
        
        # --- 1. Group by SECTOR (broad classification) for HHI — EQUITY ONLY ---
        if not equity_df.empty and equity_total > 0:
            # Recalculate weights relative to equity-only allocation
            equity_df = equity_df.copy()
            equity_df['Equity_Pct'] = (equity_df['Allocation'] / equity_total) * 100
            sector_weights = equity_df.groupby('Sector')['Equity_Pct'].sum().sort_values(ascending=False)
        else:
            sector_weights = pd.Series(dtype=float)
        
        # Calculate HHI (Herfindahl-Hirschman Index) on EQUITY sectors only
        if not sector_weights.empty:
            sector_weight_fractions = sector_weights / 100.0
            hhi = (sector_weight_fractions ** 2).sum()
            effective_sectors = 1.0 / hhi if hhi > 0 else 0
        else:
            hhi = 0
            effective_sectors = 0
        
        # HHI Risk Classification
        if hhi < 0.15:
            hhi_label = "Well Diversified"
            hhi_color = "#10b981"
            hhi_emoji = "🟢"
        elif hhi < 0.25:
            hhi_label = "Moderately Concentrated"
            hhi_color = "#f59e0b"
            hhi_emoji = "🟡"
        else:
            hhi_label = "Highly Concentrated"
            hhi_color = "#ef4444"
            hhi_emoji = "🔴"
        
        # --- HHI Gauge Card ---
        cash_badge = f"<div style='margin-top: 0.5rem; font-size: 0.85rem; color: #6ee7b7; background: rgba(16,185,129,0.1); padding: 0.35rem 0.65rem; border-radius: 6px; display: inline-block;'>💵 Cash Reserve: <strong>{cash_pct:.1f}%</strong> of portfolio (excluded from HHI)</div>" if cash_pct > 0 else ""
        
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid {hhi_color}; margin-bottom: 1rem;">
            <p style="color: #94a3b8; font-size: 0.8rem; margin: 0; text-transform: uppercase; font-weight: bold;">Equity Concentration Index (HHI)</p>
            <div style="display: flex; align-items: baseline; justify-content: center; gap: 12px; margin: 0.25rem 0;">
                <h2 style="color: {hhi_color}; margin: 0;">{hhi_emoji} {hhi:.3f}</h2>
            </div>
            <p style="color: {hhi_color}; font-weight: 600; margin: 0 0 0.5rem 0;">{hhi_label}</p>
            <div style="display: flex; justify-content: space-around; font-size: 0.85rem; color: #cbd5e1; background: rgba(0,0,0,0.2); padding: 0.5rem; border-radius: 6px;">
                <div><span style="color: #94a3b8;">Effective Sectors:</span> <strong>{effective_sectors:.1f}</strong></div>
                <div><span style="color: #94a3b8;">Equity Sectors:</span> <strong>{len(sector_weights)}</strong></div>
            </div>
            {cash_badge}
        </div>
        """, unsafe_allow_html=True)
        
        # --- 2. Sector Breakdown Table ---
        sector_table_data = []
        for sector_name, weight_pct in sector_weights.items():
            holdings_in_sector = equity_df[equity_df['Sector'] == sector_name]
            n_holdings = len(holdings_in_sector)
            industries_in_sector = holdings_in_sector['Industry'].nunique()
            
            # Risk Status
            if weight_pct > 30:
                status = "🔴 Over"
            elif weight_pct > 25:
                status = "⚠️ Caution"
            else:
                status = "✅ OK"
            
            sector_table_data.append({
                'Sector': sector_name,
                'Holdings': n_holdings,
                'Weight %': weight_pct,
                'Industries': industries_in_sector,
                'Status': status
            })
        
        # Add Cash / Liquid row if present (always safe)
        if not cash_df.empty:
            sector_table_data.append({
                'Sector': '💵 Cash / Liquid',
                'Holdings': len(cash_df),
                'Weight %': cash_pct,
                'Industries': 1,
                'Status': '💵 Safe'
            })
        
        sector_table_df = pd.DataFrame(sector_table_data)
        
        # Style the Status column
        def color_concentration_status(val):
            if '🔴' in str(val):
                return 'background-color: #450a0a; color: #fca5a5; font-weight: bold'
            elif '⚠️' in str(val):
                return 'background-color: #451a03; color: #fdba74; font-weight: bold'
            elif '💵' in str(val):
                return 'background-color: #064e3b; color: #6ee7b7; font-weight: bold'
            return 'color: #6ee7b7'
        
        st.dataframe(
            sector_table_df.style.applymap(color_concentration_status, subset=['Status']),
            column_config={
                'Sector': st.column_config.TextColumn('Sector', width='medium'),
                'Holdings': st.column_config.NumberColumn('# Holds', format='%d', width='small'),
                'Weight %': st.column_config.ProgressColumn('Weight %', format='%.1f%%', min_value=0, max_value=100),
                'Industries': st.column_config.NumberColumn('Diversified', format='%d', width='small', help='Number of distinct industries within this sector'),
                'Status': st.column_config.TextColumn('Risk', width='small'),
            },
            use_container_width=True,
            hide_index=True,
            height=min(250, 35 * len(sector_table_df) + 38)
        )

        # --- 3. Concentration Risk Alerts (EQUITY ONLY) ---
        risk_alerts = []
        
        # Alert: Overweight sectors (> 25% of equity)
        overweight = sector_weights[sector_weights > 25]
        for sec_name, sec_wt in overweight.items():
            risk_alerts.append({
                'severity': 'high' if sec_wt > 30 else 'medium',
                'text': f"**{sec_name}** is {sec_wt:.1f}% of equity (O'Neil max: 25%)"
            })
        
        # Alert: Single-stock dominance (equity only)
        if not equity_df.empty:
            max_stock = equity_df.loc[equity_df['Equity_Pct'].idxmax()]
            if max_stock['Equity_Pct'] > 20:
                risk_alerts.append({
                    'severity': 'medium',
                    'text': f"**{max_stock['Ticker']}** is {max_stock['Equity_Pct']:.1f}% of equity — single-stock risk"
                })
        
        # Alert: Same-industry clustering (equity only)
        if not equity_df.empty:
            industry_groups = equity_df.groupby('Industry').agg({'Ticker': list, 'Equity_Pct': 'sum'})
            clustered = industry_groups[industry_groups['Ticker'].apply(len) >= 2]
            for ind_name, row in clustered.iterrows():
                tickers_str = ", ".join(row['Ticker'])
                risk_alerts.append({
                    'severity': 'low',
                    'text': f"**{tickers_str}** share industry (*{ind_name}*) — correlated risk {row['Equity_Pct']:.1f}%"
                })
        
        if risk_alerts:
            st.markdown("<p style='color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; font-weight: bold; margin: 0.75rem 0 0.4rem 0;'>Concentration Alerts</p>", unsafe_allow_html=True)
            for alert in risk_alerts:
                if alert['severity'] == 'high':
                    icon, border_color, bg = "🔴", "#ef4444", "rgba(239, 68, 68, 0.08)"
                elif alert['severity'] == 'medium':
                    icon, border_color, bg = "🟡", "#f59e0b", "rgba(245, 158, 11, 0.08)"
                else:
                    icon, border_color, bg = "🔵", "#3b82f6", "rgba(59, 130, 246, 0.08)"
                
                st.markdown(f"""
                <div style="background: {bg}; border-left: 3px solid {border_color}; padding: 0.4rem 0.75rem; border-radius: 4px; margin-bottom: 0.35rem; font-size: 0.85rem;">
                    {icon} {alert['text']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981; padding: 0.5rem 0.75rem; border-radius: 4px; font-size: 0.85rem;">
                ✅ <strong>No concentration alerts.</strong> Portfolio is well diversified across sectors.
            </div>
            """, unsafe_allow_html=True)
        
        # --- 4. How to interpret (collapsible) ---
        with st.expander("ℹ️ How Professionals Monitor Sector Risk"):
            st.markdown("""
            **Herfindahl-Hirschman Index (HHI)** is the gold standard for measuring portfolio concentration used by institutional portfolio managers, central banks, and regulatory bodies (e.g., the Federal Reserve, BIS).
            
            **How it works:**
            - HHI = Sum of squared sector weights (0.0 to 1.0)
            - **< 0.15**: Well diversified — no single sector dominates
            - **0.15 – 0.25**: Moderately concentrated — acceptable for conviction-based strategies 
            - **> 0.25**: Highly concentrated — significant sector-specific risk exposure
            
            **Effective Sectors** = `1 / HHI`. This translates the math into an intuitive number: *"Your portfolio behaves as if it's spread across N independent sectors."* A portfolio with 8 sectors but Effective Sectors of 3.2 means most of your risk is driven by just ~3 sectors.
            
            **O'Neil's 25% Rule:** William O'Neil recommends no single sector should exceed 25% of a growth portfolio. Beyond this, a sector rotation or adverse event can cause outsized damage to the entire portfolio.
            
            **Industry Clustering:** Two stocks in the same industry (e.g., "Software — Application") have much higher correlation than their sector alone suggests. They respond to the same regulatory, competitive, and demand cycle forces.
            """)
    else:
        st.info("No allocation data available to compute sector concentration.")
            
    # =====================================================================
    # MARKET CAP ALLOCATION & REBALANCING DASHBOARD
    # =====================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚖️ Market Cap Allocation & Rebalancing")
    
    if not alloc_df.empty and (equity_total > 0 or cash_pct > 0):
        def categorize_mcap(row):
            mc = row.get('Market_Cap', 0)
            ticker = str(row['Ticker']).strip().upper()
            
            # SME stocks usually have missing market cap data on Yahoo Finance but are always Micro Caps
            if ticker.endswith('-SM') or ticker.endswith('-ST') or ticker.endswith('-BZ'):
                return "Micro Cap"
                
            if mc > 20000: return "Large Cap"
            if mc >= 5000: return "Mid Cap"
            if mc >= 1000: return "Small Cap"
            if mc > 0: return "Micro Cap"
            
            # User overrides for large-cap focused ETFs
            if ticker in ['MOM30IETF', 'ALPHA', 'MOMENTUM50', 'MOM50']:
                return "Large Cap"
            
            return "Unknown"
            
        if not equity_df.empty:
            equity_df['MC_Category'] = equity_df.apply(categorize_mcap, axis=1)
            mc_totals = equity_df.groupby('MC_Category')['Allocation'].sum()
        else:
            mc_totals = pd.Series(dtype=float)
            
        large_alloc = mc_totals.get("Large Cap", 0.0)
        mid_alloc = mc_totals.get("Mid Cap", 0.0)
        small_alloc = mc_totals.get("Small Cap", 0.0)
        micro_alloc = mc_totals.get("Micro Cap", 0.0)
        cash_alloc = cash_df['Allocation'].sum() if not cash_df.empty else 0.0
        
        total_portfolio = large_alloc + mid_alloc + small_alloc + micro_alloc + cash_alloc
        # Any "Unknown" equity is excluded from the math to force targets to sum to 100% of tracked capital
        if total_portfolio == 0: total_portfolio = 1.0
        
        large_pct = (large_alloc / total_portfolio) * 100
        mid_pct = (mid_alloc / total_portfolio) * 100
        small_pct = (small_alloc / total_portfolio) * 100
        micro_pct = (micro_alloc / total_portfolio) * 100
        cash_pct_display = (cash_alloc / total_portfolio) * 100
        
        # --- Top Row: Visual Overview ---
        mc_col1, mc_col2, mc_col3, mc_col4, mc_col5 = st.columns(5)
        
        def render_mc_card(title, pct, val, color):
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {color}; margin-bottom: 1rem; padding: 1rem;">
                <p style="color: #94a3b8; font-size: 0.75rem; margin: 0; text-transform: uppercase; font-weight: bold;">{title}</p>
                <h3 style="color: {color}; margin: 0.25rem 0; font-size: 1.5rem;">{pct:.1f}%</h3>
                <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0;">₹{val:,.0f}</p>
                <div style="width: 100%; background-color: rgba(255,255,255,0.1); border-radius: 4px; height: 6px; margin-top: 8px;">
                    <div style="width: {min(100, pct)}%; background-color: {color}; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with mc_col1: render_mc_card("Large (>20k Cr)", large_pct, large_alloc, "#10b981") # Green
        with mc_col2: render_mc_card("Mid (5k-20k Cr)", mid_pct, mid_alloc, "#3b82f6") # Blue
        with mc_col3: render_mc_card("Small (1k-5k Cr)", small_pct, small_alloc, "#8b5cf6") # Purple
        with mc_col4: render_mc_card("Micro (<1k Cr)", micro_pct, micro_alloc, "#ec4899") # Pink
        with mc_col5: render_mc_card("Cash / Liquid / Gold", cash_pct_display, cash_alloc, "#f59e0b") # Amber

        # --- Middle Row: Target Config ---
        st.markdown("<p style='color: #94a3b8; font-size: 0.85rem; margin: 1rem 0 0.5rem 0; text-transform: uppercase; font-weight: bold;'>🎯 Set Target Allocations (Must sum to 100%)</p>", unsafe_allow_html=True)
        
        if 'mc_target_large' not in st.session_state: st.session_state['mc_target_large'] = 30
        if 'mc_target_mid' not in st.session_state: st.session_state['mc_target_mid'] = 30
        if 'mc_target_small' not in st.session_state: st.session_state['mc_target_small'] = 20
        if 'mc_target_micro' not in st.session_state: st.session_state['mc_target_micro'] = 20
        if 'mc_target_cash' not in st.session_state: st.session_state['mc_target_cash'] = 0
        
        t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
        with t_col1: target_large = st.number_input("Large Cap %", min_value=0, max_value=100, value=st.session_state['mc_target_large'], step=5, key="t_large")
        with t_col2: target_mid = st.number_input("Mid Cap %", min_value=0, max_value=100, value=st.session_state['mc_target_mid'], step=5, key="t_mid")
        with t_col3: target_small = st.number_input("Small Cap %", min_value=0, max_value=100, value=st.session_state.get('mc_target_small', 20), step=5, key="t_small")
        with t_col4: target_micro = st.number_input("Micro Cap %", min_value=0, max_value=100, value=st.session_state['mc_target_micro'], step=5, key="t_micro")
        with t_col5: target_cash = st.number_input("Cash %", min_value=0, max_value=100, value=st.session_state['mc_target_cash'], step=5, key="t_cash")
        
        st.session_state['mc_target_large'] = target_large
        st.session_state['mc_target_mid'] = target_mid
        st.session_state['mc_target_small'] = target_small
        st.session_state['mc_target_micro'] = target_micro
        st.session_state['mc_target_cash'] = target_cash
        
        target_sum = target_large + target_mid + target_small + target_micro + target_cash
        
        # --- Bottom Row: Rebalancing Directives ---
        if target_sum != 100:
            st.error(f"⚠️ Targets must sum to 100%. Currently: {target_sum}%")
        else:
            st.markdown("<p style='color: #94a3b8; font-size: 0.85rem; margin: 1rem 0 0.5rem 0; text-transform: uppercase; font-weight: bold;'>⚡ Rebalancing Directives</p>", unsafe_allow_html=True)
            
            def calc_directive(actual_val, target_pct_whole, total_eq):
                target_val = (target_pct_whole / 100.0) * total_eq
                delta = target_val - actual_val
                delta_pct = (target_pct_whole) - (actual_val / total_eq * 100) if total_eq > 0 else 0
                return delta, delta_pct
            
            delta_l, d_pct_l = calc_directive(large_alloc, target_large, total_portfolio)
            delta_m, d_pct_m = calc_directive(mid_alloc, target_mid, total_portfolio)
            delta_s, d_pct_s = calc_directive(small_alloc, target_small, total_portfolio)
            delta_mic, d_pct_mic = calc_directive(micro_alloc, target_micro, total_portfolio)
            delta_c, d_pct_c = calc_directive(cash_alloc, target_cash, total_portfolio)
            
            directives = [
                ("Large Cap", delta_l, d_pct_l),
                ("Mid Cap", delta_m, d_pct_m),
                ("Small Cap", delta_s, d_pct_s),
                ("Micro Cap", delta_mic, d_pct_mic),
                ("Cash/Gold", delta_c, d_pct_c)
            ]
            
            for name, delta_val, delta_pct in directives:
                if abs(delta_pct) < 1.0:
                    st.markdown(f"<div style='background: rgba(255,255,255,0.05); padding: 0.6rem; border-radius: 6px; margin-bottom: 0.4rem; color: #94a3b8;'>✅ <strong>{name}:</strong> perfectly balanced.</div>", unsafe_allow_html=True)
                elif delta_val > 0:
                    st.markdown(f"<div style='background: rgba(16,185,129,0.1); border-left: 4px solid #10b981; padding: 0.6rem; border-radius: 6px; margin-bottom: 0.4rem; color: #6ee7b7;'>🟢 <strong>DEPLOY into {name}:</strong> add ₹{abs(delta_val):,.0f} (+{abs(delta_pct):.1f}%)</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background: rgba(239,68,68,0.1); border-left: 4px solid #ef4444; padding: 0.6rem; border-radius: 6px; margin-bottom: 0.4rem; color: #fca5a5;'>🔴 <strong>TRIM from {name}:</strong> free up ₹{abs(delta_val):,.0f} (-{abs(delta_pct):.1f}%)</div>", unsafe_allow_html=True)
    else:
        st.info("No allocation data available for Market Cap analysis.")

    # --- Liquidity Risk Alerts ---
    if not equity_df.empty:
        st.markdown("<p style='color: #94a3b8; font-size: 0.85rem; margin: 1.5rem 0 0.5rem 0; text-transform: uppercase; font-weight: bold;'>⚠️ Liquidity Risk Alerts</p>", unsafe_allow_html=True)
        
        liquidity_alerts = []
        for idx, row in equity_df.iterrows():
            ticker = row['Ticker']
            alloc_val = row['Allocation']
            
            df = market_data.get(ticker)
            if df is not None and not df.empty and 'Volume' in df.columns and 'Close' in df.columns:
                vol_50 = df['Volume'].tail(50).mean()
                last_close = df['Close'].iloc[-1]
                adtv = vol_50 * last_close
                
                if adtv > 0:
                    pct_of_adtv = (alloc_val / adtv) * 100
                    if pct_of_adtv > 5.0:
                        liquidity_alerts.append({
                            'ticker': ticker,
                            'alloc': alloc_val,
                            'adtv': adtv,
                            'pct': pct_of_adtv
                        })
                        
        if liquidity_alerts:
            liquidity_alerts.sort(key=lambda x: x['pct'], reverse=True)
            for alert in liquidity_alerts:
                def fmt_val(v):
                    if v >= 10000000: return f"₹{v/10000000:.2f} Cr"
                    return f"₹{v/100000:.1f} L"
                st.markdown(f"<div style='background: rgba(245,158,11,0.1); border-left: 4px solid #f59e0b; padding: 0.6rem; border-radius: 6px; margin-bottom: 0.4rem; color: #fdba74;'>⚠️ <strong>{alert['ticker']} Illiquidity Risk:</strong> Holding size ({fmt_val(alert['alloc'])}) is <strong>{alert['pct']:.1f}%</strong> of ADTV ({fmt_val(alert['adtv'])}). <em>Exceeds 5% safe limit.</em></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background: rgba(16,185,129,0.1); border-left: 4px solid #10b981; padding: 0.6rem; border-radius: 6px; color: #6ee7b7;'>✅ <strong>No Liquidity Risk.</strong> All equity positions are &lt; 5% of their Average Daily Traded Value (ADTV).</div>", unsafe_allow_html=True)


    st.markdown("---")
    
    # =========================================================================
    # 🗓️ PORTFOLIO EARNINGS & RISK HUB
    # =========================================================================
    hub_h_c1, hub_h_c2 = st.columns([4, 1])
    with hub_h_c1:
        st.markdown("### 🗓️ Portfolio Earnings & Risk Hub")
    with hub_h_c2:
        if st.button("🔄 Sync Earnings Data", help="Fetch latest quarterly results for all active holdings from exchange"):
            with st.spinner("Syncing latest quarterly financials..."):
                active_t = portfolio_df['ticker'].tolist() if (portfolio_df is not None and not portfolio_df.empty) else []
                update_portfolio_earnings(active_t, force_refresh=True)
                st.cache_data.clear()
                st.toast("Earnings cache synced successfully!", icon="✅")
                st.rerun()
    
    tab_scorecard, tab_calendar, tab_risk = st.tabs([
        "📈 Dual EPS & Sales Scorecard",
        "📅 Earnings Calendar",
        "🛡️ Sector Exposure & Binary Risk"
    ])
    
    with tab_scorecard:
        st.markdown("#### 🚀 4-Quarter Dual Acceleration Matrix (CANSLIM 'C' & 'A')")
        st.caption("Top-line (Sales) and Bottom-line (EPS) dual acceleration confirms institutional delivery.")

        # Filter cache items to ONLY active portfolio holdings
        filtered_cache = {k: v for k, v in earnings_cache.items() if not active_portfolio_tickers or k in active_portfolio_tickers}

        # Find latest chronological results season quarter across non-ETF holdings
        quarters_found = [e['latest_quarter'] for e in filtered_cache.values() if e.get('latest_quarter') not in ['N/A', 'ETF', 'Annual (Screener)']]
        def _parse_q_date(q_str):
            try:
                from datetime import datetime
                clean_q = str(q_str).replace(' (Annual)', '').strip()
                return datetime.strptime(clean_q, '%b %Y')
            except Exception:
                return datetime(1970, 1, 1)

        max_q = max(set(quarters_found), key=_parse_q_date) if quarters_found else "Jun 2026"
        
        reported_count = sum(1 for e in filtered_cache.values() if (e.get('latest_quarter') == max_q or '(Annual)' in str(e.get('latest_quarter', ''))) and not e.get('is_etf'))
        awaiting_count = sum(1 for e in filtered_cache.values() if e.get('latest_quarter') != max_q and '(Annual)' not in str(e.get('latest_quarter', '')) and not e.get('is_etf'))
        code33_count = sum(1 for e in filtered_cache.values() if e.get('dual_acceleration') and not e.get('is_etf'))



        c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
        with c_kpi1:
            st.markdown(f"""<div class="metric-card" style="border-top: 3px solid #10b981; padding: 0.75rem;"><p style="color:#94a3b8; margin:0; font-size:0.75rem; font-weight:bold;">{max_q} / AUDITED REPORTED</p><h3 style="color:#10b981; margin:0.2rem 0;">{reported_count} Holdings</h3></div>""", unsafe_allow_html=True)
        with c_kpi2:
            st.markdown(f"""<div class="metric-card" style="border-top: 3px solid #f59e0b; padding: 0.75rem;"><p style="color:#94a3b8; margin:0; font-size:0.75rem; font-weight:bold;">AWAITING {max_q}</p><h3 style="color:#f59e0b; margin:0.2rem 0;">{awaiting_count} Holdings</h3></div>""", unsafe_allow_html=True)
        with c_kpi3:
            st.markdown(f"""<div class="metric-card" style="border-top: 3px solid #3b82f6; padding: 0.75rem;"><p style="color:#94a3b8; margin:0; font-size:0.75rem; font-weight:bold;">DUAL ACCEL WINNERS</p><h3 style="color:#3b82f6; margin:0.2rem 0;">{code33_count} Stocks</h3></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Controls Row: Pre-sort selector
        c_ctrl1, c_ctrl2 = st.columns([2, 3])
        with c_ctrl1:
            sort_choice = st.selectbox(
                "Sort Scorecard By",
                [
                    "🚀 Dual Accel Winners First",
                    "🔥 Sales Current (High to Low)",
                    "💰 EPS Current (High to Low)",
                    "📅 Latest Quarter Reported First",
                    "🔤 Ticker (A-Z)"
                ],
                index=0,
                key="scorecard_table_sort",
                help="Pre-sort table or click any column header directly in the table to sort dynamically."
            )

        scorecard_rows = []
        etf_tickers = []
        for ticker, e_data in filtered_cache.items():

            if e_data.get('is_etf', False):
                etf_tickers.append(ticker)
                continue

            eps_traj = e_data.get('eps_trajectory', [])
            sales_traj = e_data.get('sales_trajectory', [])
            
            # Helper to extract numeric value safely for mathematical sorting
            def get_traj_num(traj_list, idx):
                if traj_list and 0 <= idx < len(traj_list):
                    t_item = traj_list[idx]
                    if isinstance(t_item, dict):
                        val = t_item.get('val')
                        if val is not None and pd.notna(val) and not np.isnan(val):
                            return float(val)
                    elif isinstance(t_item, (int, float)) and pd.notna(t_item) and not np.isnan(t_item):
                        return float(t_item)
                return np.nan

            num_eps = len(eps_traj)
            q1_eps = get_traj_num(eps_traj, 0) if num_eps >= 4 else np.nan
            q2_eps = get_traj_num(eps_traj, num_eps - 3) if num_eps >= 3 else np.nan
            q3_eps = get_traj_num(eps_traj, num_eps - 2) if num_eps >= 2 else np.nan
            q4_eps = get_traj_num(eps_traj, num_eps - 1) if num_eps >= 1 else np.nan

            num_s = len(sales_traj)
            q4_s = get_traj_num(sales_traj, num_s - 1) if num_s >= 1 else np.nan

            l_q = e_data.get('latest_quarter', 'N/A')
            if l_q == max_q or e_data.get('is_new'):
                q_badge = f"🔥 {l_q} (NEW)"
            elif '(Annual)' in l_q:
                q_badge = f"📘 {l_q}"
            elif l_q != 'N/A':
                q_badge = f"⏳ {l_q} (Awaiting {max_q})"
            else:
                q_badge = "N/A"

            eps_val_score = e_data.get('eps_yoy_pct')
            dual_acc = "🚀 DUAL ACCEL" if e_data.get('dual_acceleration') else "In-Line" if (eps_val_score is not None and eps_val_score >= 0) else "Contraction" if eps_val_score is not None else "Neutral"
            
            scorecard_rows.append({
                'Ticker': ticker,
                'Latest Quarter': q_badge,
                'EPS (Q-3)': q1_eps,
                'EPS (Q-2)': q2_eps,
                'EPS (Q-1)': q3_eps,
                'EPS Current': q4_eps,
                'Sales Current': q4_s,
                'Dual Status': dual_acc,
                'Verdict': e_data.get('verdict', 'Neutral'),
                'is_dual': 1 if e_data.get('dual_acceleration') else 0,
            })
            
        if scorecard_rows:
            scorecard_df = pd.DataFrame(scorecard_rows)

            # Pre-sort according to user selection
            if sort_choice == "🔥 Sales Current (High to Low)":
                scorecard_df = scorecard_df.sort_values(by='Sales Current', ascending=False, na_position='last')
            elif sort_choice == "💰 EPS Current (High to Low)":
                scorecard_df = scorecard_df.sort_values(by='EPS Current', ascending=False, na_position='last')
            elif sort_choice == "🚀 Dual Accel Winners First":
                scorecard_df = scorecard_df.sort_values(by=['is_dual', 'Sales Current'], ascending=[False, False], na_position='last')
            elif sort_choice == "📅 Latest Quarter Reported First":
                scorecard_df = scorecard_df.sort_values(by='Latest Quarter', ascending=True)
            elif sort_choice == "🔤 Ticker (A-Z)":
                scorecard_df = scorecard_df.sort_values(by='Ticker', ascending=True)

            display_df = scorecard_df.drop(columns=['is_dual'])

            # Row styling: highlight stocks where Sales Current < 25% (below CANSLIM growth gate)
            def highlight_low_sales(row):
                s = row.get('Sales Current')
                if pd.notna(s) and s < 25.0:
                    return ['background-color: rgba(239, 68, 68, 0.22); color: #fca5a5; font-weight: bold'] * len(row)
                return [''] * len(row)

            styled_df = display_df.style.apply(highlight_low_sales, axis=1)

            column_config = {
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Latest Quarter": st.column_config.TextColumn("Latest Quarter", width="medium"),
                "EPS (Q-3)": st.column_config.NumberColumn("EPS (Q-3)", format="%+.1f%%", help="Quarterly EPS growth YoY (3 quarters ago)"),
                "EPS (Q-2)": st.column_config.NumberColumn("EPS (Q-2)", format="%+.1f%%", help="Quarterly EPS growth YoY (2 quarters ago)"),
                "EPS (Q-1)": st.column_config.NumberColumn("EPS (Q-1)", format="%+.1f%%", help="Quarterly EPS growth YoY (1 quarter ago)"),
                "EPS Current": st.column_config.NumberColumn("EPS Current", format="%+.1f%%", help="Latest reported quarterly EPS growth YoY"),
                "Sales Current": st.column_config.NumberColumn("Sales Current", format="%+.1f%%", help="Latest reported quarterly Sales/Revenue growth YoY (<25% highlighted in crimson)"),
                "Dual Status": st.column_config.TextColumn("Dual Status", width="small"),
                "Verdict": st.column_config.TextColumn("Verdict", width="medium")
            }

            st.caption("⚠️ **Vulnerability Watchlist:** Rows highlighted in **soft crimson** indicate holdings where latest quarterly Sales Growth is **< 25%** (below the CANSLIM 'C' threshold).")

            st.dataframe(
                styled_df,
                column_config=column_config,
                use_container_width=True,
                hide_index=True
            )
            
        if etf_tickers:
            with st.expander("💵 View ETFs / Non-Reporting Assets (Excluded from Earnings Matrix)", expanded=False):
                st.caption("ETFs, Gold funds, and Preference Shares do not report corporate quarterly earnings.")
                st.write(", ".join(etf_tickers))


    with tab_calendar:
        st.markdown("#### 📅 Portfolio Earnings Announcement Radar")
        st.caption("Upcoming quarterly results dates reported by yfinance and exchange filings.")
        
        cal_rows = []
        for ticker, e_data in filtered_cache.items():

            if e_data.get('is_etf', False):
                continue
            ed_date = e_data.get('upcoming_earnings_date', 'N/A')
            days_left = e_data.get('days_to_earnings')
            
            if ed_date != 'N/A' or days_left is not None:
                cal_rows.append({
                    'Ticker': ticker,
                    'Upcoming Date': ed_date,
                    'Days Remaining': f"{days_left} Days" if days_left is not None else "TBD",
                    'Status': "⚠️ IMMINENT" if (days_left is not None and days_left <= 5) else "Upcoming"
                })
                
        if cal_rows:
            cal_df = pd.DataFrame(cal_rows).sort_values(by='Upcoming Date')
            st.dataframe(cal_df, use_container_width=True, hide_index=True)
        else:
            st.info("🗓️ **No Imminent Exchange-Scheduled Dates:** No upcoming earnings announcement dates scheduled in the next 14 days by exchanges. Results will automatically update as companies file outcome reports.")


    with tab_risk:
        st.markdown("#### 🛡️ Sector Concentration Caps & Pre-Earnings Binary Risk")
        
        # Sector Concentration Alerts (Max 25% Cap)
        st.markdown("**1. Sector Concentration Cap Audit (Max 25% Cap):**")
        if not alloc_df.empty:
            overweight_secs = sector_summary[sector_summary['Percent'] > 25]
            if not overweight_secs.empty:
                for _, s_row in overweight_secs.iterrows():
                    st.warning(f"⚠️ **{s_row['Sector']}** represents **{s_row['Percent']:.1f}%** of total portfolio allocation — **Exceeds O'Neil 25% Max Sector Cap**. Consider trimming to reduce sector-specific shock risk.")
            else:
                st.success("✅ **Sector Diversification Healthy:** No single sector exceeds the 25% maximum concentration cap.")
        else:
            st.info("No portfolio allocation data available for sector cap audit.")

        # Pre-Earnings Binary Risk Flag (Gap 6)
        st.markdown("<br>**2. Pre-Earnings Binary Gap Risk Radar:**", unsafe_allow_html=True)
        binary_risks = get_pre_earnings_risk_tickers(earnings_cache, portfolio_df, max_days=5)
        if binary_risks:
            for b in binary_risks:
                st.error(f"🚨 **PRE-EARNINGS BINARY GAP RISK:** **{b['ticker']}** reports earnings in **{b['days_left']} days** ({b['upcoming_date']}). Holding large unhedged positions into binary earnings events carries overnight gap-down risk.")
        else:
            st.success("✅ **No Imminent Binary Gap Risk:** No portfolio holdings are reporting results in the next 5 days.")

        # Launchpad link to Live Market Earnings Calendar & PEAD Hub
        st.markdown("""
        <div style="margin-top: 10px; padding: 10px 14px; background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: #cbd5e1;">
                <span>📅</span>
                <span>Track nationwide earnings surprises, consensus forecasts & post-earnings drift setups:</span>
            </div>
            <a href="/pead_scanner" target="_self" style="padding: 4px 12px; background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 6px; text-decoration: none; font-size: 0.78rem; font-weight: 700;">
                ⚡ Live Earnings Hub & PEAD Scanner ↗
            </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # 4. Climax / Exhaustion Detector (CED)

    ced_results = []
    for _, row in portfolio_df.iterrows():
        ticker = row['ticker']
        df = market_data.get(ticker)
        if df is not None and not df.empty:
            result = detect_climax_exhaustion(df, ticker)
            ced_results.append(result)
            
    # Sort by Score descending
    ced_results.sort(key=lambda x: x['climax_score'], reverse=True)
    
    # Check if any have elevated risk
    high_risk_ced = [r for r in ced_results if r['climax_score'] >= 3]
    
    st.markdown("### 🔥 Climax / Exhaustion Detector")
    if high_risk_ced:
        st.warning(f"⚠️ **{len(high_risk_ced)} holdings** are showing signs of climax/exhaustion. Review carefully.")
    else:
        st.success("✅ No climax signals detected in portfolio (Score < 3).")
        
    # Prepare CED table
    ced_table_data = []
    for res in ced_results:
        # Determine status emoji
        status_emoji = "🟢"
        if res['climax_score'] >= 4:
            status_emoji = "🔥"
        elif res['climax_score'] == 3:
            status_emoji = "⚠️"
        elif res['climax_score'] == 2:
            status_emoji = "🟡"
            
        slope_10 = res.get('slope_10', 0)
        slope_30 = res.get('slope_30', 0)
        slope_ratio = slope_10 / slope_30 if slope_30 > 0 else 0
        
        ced_table_data.append({
            'Ticker': f"https://in.tradingview.com/chart/?symbol=NSE:{res['ticker']}",
            'Score': res['climax_score'],
            'Status': f"{status_emoji} {res['status']}",
            '15D Return': res['return_15d'],
            '% >10EMA': res['pct_above_10ema'],
            'Vol Spike': res['vol_spike_ratio'],
            'ATR Ratio': res['atr_ratio'],
            'Slope Ratio': slope_ratio,
        })
        
    columns = ['Ticker', 'Score', 'Status', '15D Return', '% >10EMA', 'Vol Spike', 'ATR Ratio', 'Slope Ratio']
    df_ced = pd.DataFrame(ced_table_data, columns=columns)
    
    def highlight_climax_row(row):
        score = row.get('Score', 0)
        if score >= 4:
            return ['background-color: #450a0a; color: #fca5a5; font-weight: bold'] * len(row)
        elif score == 3:
            return ['background-color: #451a03; color: #fdba74'] * len(row)
        return [''] * len(row)
        
    st.dataframe(
        df_ced.style.apply(highlight_climax_row, axis=1),
        column_config={
            "Ticker": st.column_config.LinkColumn(
                "Ticker", display_text=r"symbol=NSE:(.*)", width="small"
            ),
            "Score": st.column_config.ProgressColumn(
                "Climax Score", 
                min_value=0, 
                max_value=5, 
                format="%d/5",
                help="0-1: Normal, 2: Elevated, 3: Risk, 4-5: High Prob"
            ),
            "Status": st.column_config.TextColumn("Risk Status", width="medium"),
            "15D Return": st.column_config.NumberColumn("15D Ret", format="%.1f%%"),
            "% >10EMA": st.column_config.NumberColumn("% >10EMA", format="%.1f%%"),
            "Vol Spike": st.column_config.NumberColumn("Vol Spike", format="%.2fx"),
            "ATR Ratio": st.column_config.NumberColumn("ATR Ratio", format="%.2fx"),
            "Slope Ratio": st.column_config.NumberColumn("Slope Ratio", format="%.1fx"),
        },
        use_container_width=True,
        hide_index=True,
        height=300 if len(df_ced) > 5 else "content"
    )
    
    with st.expander("ℹ️ How to Interpret Climax Scores"):
        st.markdown("""
        **The Climax / Exhaustion Detector (CED)** identifies stocks entering a parabolic, late-stage acceleration phase which often precedes a sharp pullback or top.
        
        **Scoring System (0-5 Points)**
        *   **Condition 1 (+1)**: **3-Week Acceleration** (Price up ≥ 25% in last 15 days)
        *   **Condition 2 (+1)**: **Extreme Extension** (Price ≥ 15% above 10 EMA)
        *   **Condition 3 (+1)**: **Volume Expansion** (10-day Avg Vol ≥ 1.75x 50-day Avg Vol)
        *   **Condition 4 (+1)**: **Range Expansion** (ATR-10 ≥ 1.5x ATR-30)
        *   **Condition 5 (+1)**: **Slope Acceleration** (10-day Slope ≥ 1.8x 30-day Slope)
        
        **Status Labels**
        *   🟢 **Normal (0-1)**: No exhaustion signs.
        *   🟡 **Elevated Momentum (2)**: Watch closely.
        *   ⚠️ **Climax Risk (3)**: High probability of pullback. partial profits recommended.
        *   🔥 **High Climax Probability (4-5)**: Extreme blow-off action. Tighten stops or exit.
        """)
    
    def render_institutional_defense_radar(decisions, bse_reverse_map):
        st.markdown("### 🛡️ Institutional Defense & Liquidation Radar")
        st.caption("Quantitative surveillance tracking institutional exit footprints across William O'Neil, David Ryan, Mark Minervini, and Ross Haber selling rules.")

        if not decisions:
            st.info("No holdings found in portfolio.")
            return

        # -------------------------------------------------------------
        # 1. Compute Individual Stock Defense Ledger (Pillars 1-4)
        # -------------------------------------------------------------
        active_defense_data = []
        for d in decisions:
            p_weight = d.get('loss_metrics', {}).get('portfolio_weight', 0.0)
            raw_tick = str(d['ticker'])
            disp_tick = bse_reverse_map.get(raw_tick, raw_tick)
            exch = 'BSE' if raw_tick.isdigit() else 'NSE'
            dist_m = d.get('distribution_metrics', {})
            
            active_defense_data.append({
                'ticker': raw_tick,
                'display_ticker': disp_tick,
                'exchange': exch,
                'ticker_url': f"https://in.tradingview.com/chart/?symbol={exch}:{disp_tick}",
                'weight': p_weight,
                'metrics': dist_m,
                'ids_score': dist_m.get('ids_score', 0),
                'idvd_15d': dist_m.get('idvd_count_15d', 0),
                'churn_15d': dist_m.get('churning_count_15d', 0),
                'total_dd_15d': dist_m.get('total_dist_days_15d', 0),
                'cluster_detected': dist_m.get('cluster_detected', False),
                'ud_ratio': dist_m.get('ud_volume_ratio_50d', 1.0),
                'haber_21': dist_m.get('haber_21_breach', False),
                'oneil_50': dist_m.get('oneil_50_breach', False),
                'status': dist_m.get('status', '🟢 Clean'),
                'prescription': dist_m.get('prescription', 'Hold core position; institutional accumulation remains constructive.'),
                'sessions': dist_m.get('dist_sessions', [])
            })

        # Filter active positions (weight > 0) if weights exist
        active_holdings = [x for x in active_defense_data if x['weight'] > 0] if any(x['weight'] > 0 for x in active_defense_data) else active_defense_data
        
        # Calculate individual executive metrics
        acute_list = [x for x in active_holdings if x['cluster_detected'] or x['ids_score'] >= 5 or x['oneil_50']]
        stalling_list = [x for x in active_holdings if 3 <= x['ids_score'] < 5 and not x['cluster_detected'] and not x['oneil_50']]
        clean_list = [x for x in active_holdings if x['ids_score'] < 3 and not x['cluster_detected']]
        ma_breach_list = [x for x in active_holdings if x['haber_21'] or x['oneil_50']]
        avg_ud_ratio = float(np.mean([x['ud_ratio'] for x in active_holdings])) if active_holdings else 1.0

        # -------------------------------------------------------------
        # 2. Compute Thematic Contagion Matrix (Pillar 1 Pack Defense)
        # -------------------------------------------------------------
        from theme_contagion_engine import analyze_portfolio_theme_contagion
        contagion_data = analyze_portfolio_theme_contagion(decisions, bse_reverse_map)
        theme_matrix = contagion_data["theme_matrix"]
        theme_summary = contagion_data["summary"]

        # Executive KPI Cards
        from components import render_metric_card
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            acute_c = "red-text" if len(acute_list) > 0 else "green-text"
            render_metric_card("🚨 Acute Sell Clusters", f"{len(acute_list)}", subtext="≥ 3 DDs in 15d or 50 SMA break", color_class=acute_c)
        with c2:
            contagion_count = theme_summary['severe_contagion_count']
            contagion_c = "red-text" if contagion_count > 0 else "yellow-text" if theme_summary['elevated_risk_count'] > 0 else "green-text"
            contagion_sub = f"{contagion_count} theme(s) ≥ 50% liquidation" if contagion_count > 0 else "No systemic pack exits"
            render_metric_card("🔥 Thematic Contagions", f"{contagion_count}", subtext=contagion_sub, color_class=contagion_c)
        with c3:
            ma_c = "red-text" if len(ma_breach_list) > 0 else "green-text"
            render_metric_card("📉 MA Defense Breaches", f"{len(ma_breach_list)}", subtext="Violating 21 EMA or 50 SMA", color_class=ma_c)
        with c4:
            ud_c = "green-text" if avg_ud_ratio >= 1.1 else "yellow-text" if avg_ud_ratio >= 0.9 else "red-text"
            ud_sub = "Constructive Accumulation" if avg_ud_ratio >= 1.1 else "Supply Dominance" if avg_ud_ratio < 0.9 else "Neutral Volume Flow"
            render_metric_card("📊 Portfolio 50D U/D Vol", f"{avg_ud_ratio:.2f}x", subtext=ud_sub, color_class=ud_c)

        st.markdown("")

        # -------------------------------------------------------------
        # 3. High-Alert Surveillance Banners
        # -------------------------------------------------------------
        # Systemic Thematic Contagion Warning (O'Neil 50% Rule)
        if theme_summary["has_critical_alert"]:
            severe_themes_str = ", ".join([f"**{t}**" for t in theme_summary["severe_contagion_themes"]])
            st.error(
                f"🚨 **SYSTEMIC THEMATIC CONTAGION DETECTED (O'NEIL 50% RULE)**: Multiple stocks in {severe_themes_str} "
                f"are suffering institutional distribution simultaneously. O'Neil warns: *50% of a stock's move is dictated by its group.* "
                f"Mutual funds & FIIs are liquidating entire baskets. Halt new buys in these themes and enforce strict capital preservation."
            )
        
        # Individual Holding Liquidation Alerts
        if acute_list:
            acute_ticks = ", ".join([f"**{x['display_ticker']}** ({x['total_dd_15d']} DDs, IDS {x['ids_score']}/10)" for x in acute_list])
            st.warning(f"⚠️ **STOCK-LEVEL LIQUIDATION SESSIONS**: Acute distribution detected in {len(acute_list)} holding(s): {acute_ticks}. Review tactical mandates below.")
        elif not theme_summary["has_critical_alert"]:
            st.success("✅ **INSTITUTIONAL DEFENSE CLEAR**: All portfolio themes reflect healthy institutional accumulation or minimal supply absorption.")

        # -------------------------------------------------------------
        # 4. Surveillance Mode Switcher: Thematic Matrix vs Stock Ledger
        # -------------------------------------------------------------
        st.markdown("")
        col_view, col_info = st.columns([3, 1])
        with col_view:
            defense_mode = st.radio(
                "Defense Surveillance Mode",
                ["🏭 Thematic Contagion Matrix (Pillar 1)", "🔍 Individual Stock Defense Ledger (Pillars 1-4)"],
                horizontal=True,
                key="defense_surveillance_mode_radio",
                label_visibility="collapsed"
            )

        st.markdown("")

        # =============================================================
        # VIEW 1: THEMATIC CONTAGION MATRIX (PILLAR 1)
        # =============================================================
        if defense_mode == "🏭 Thematic Contagion Matrix (Pillar 1)":
            theme_tbl = [
                '<style>',
                '.theme-table-container { overflow-x: auto; width: 100%; margin-bottom: 20px; }',
                '.theme-table { width: 100%; border-collapse: separate; border-spacing: 0 8px; font-family: "Inter", sans-serif; }',
                '.theme-table th { color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; text-align: left; padding: 12px 14px; border-bottom: 1px solid rgba(255,255,255,0.05); white-space: nowrap; }',
                '.theme-row { background: rgba(255,255,255,0.03); transition: all 0.2s ease; border-radius: 8px; }',
                '.theme-row:hover { background: rgba(255,255,255,0.06); transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }',
                '.theme-row td { padding: 14px 14px; font-size: 0.85rem; color: #f8fafc; border-top: 1px solid rgba(255,255,255,0.02); border-bottom: 1px solid rgba(255,255,255,0.02); white-space: nowrap; }',
                '.theme-badge-red { background: rgba(239, 68, 68, 0.2); color: #f87171; padding: 3px 8px; border-radius: 8px; font-size: 0.75rem; font-weight: 800; border: 1px solid rgba(239, 68, 68, 0.4); box-shadow: 0 0 8px rgba(239, 68, 68, 0.3); white-space: nowrap; display: inline-block; }',
                '.theme-badge-yellow { background: rgba(234, 179, 8, 0.15); color: #facc15; padding: 3px 8px; border-radius: 8px; font-size: 0.75rem; font-weight: 700; border: 1px solid rgba(234, 179, 8, 0.3); white-space: nowrap; display: inline-block; }',
                '.theme-badge-green { background: rgba(34, 197, 94, 0.15); color: #4ade80; padding: 3px 8px; border-radius: 8px; font-size: 0.75rem; font-weight: 700; border: 1px solid rgba(34, 197, 94, 0.3); white-space: nowrap; display: inline-block; }',
                '.theme-badge-slate { background: rgba(148, 163, 184, 0.15); color: #cbd5e1; padding: 3px 8px; border-radius: 8px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(148, 163, 184, 0.3); white-space: nowrap; display: inline-block; }',
                '.theme-row-red { background: rgba(239, 68, 68, 0.07) !important; border-left: 3px solid #ef4444 !important; }',
                '.theme-row-yellow { background: rgba(234, 179, 8, 0.05) !important; border-left: 3px solid #facc15 !important; }',
                '</style>',
                '<div class="theme-table-container">',
                '<table class="theme-table">',
                '<thead><tr><th>Thematic Cluster</th><th>Capital %</th><th>Constituents (N)</th><th>Contagion Rate</th><th>50D U/D Vol</th><th>Contagion Status</th><th>Tactical CANSLIM Mandate</th></tr></thead>',
                '<tbody>'
            ]

            for item in theme_matrix:
                row_cls = "theme-row"
                if item["status_color"] == "red":
                    row_cls = "theme-row theme-row-red"
                    badge_cls = "theme-badge-red"
                elif item["status_color"] == "yellow":
                    row_cls = "theme-row theme-row-yellow"
                    badge_cls = "theme-badge-yellow"
                elif item["status_color"] == "green":
                    badge_cls = "theme-badge-green"
                else:
                    badge_cls = "theme-badge-slate"

                theme_name_cell = f'<strong style="color: #60a5fa; font-size: 0.90rem;">{item["theme"]}</strong>'
                wt_cell = f'<strong style="color: #f1f5f9;">{item["capital_weight"]:.1f}%</strong>'

                # Format constituents list: color red if distributive, green if clean
                const_pills = []
                for c in item["constituents"]:
                    c_sym = c["resolved_symbol"]
                    if c["is_distributive"]:
                        const_pills.append(f'<span style="color: #f87171; font-weight: 700;">{c_sym} 🔴</span>')
                    else:
                        const_pills.append(f'<span style="color: #94a3b8;">{c_sym}</span>')
                const_cell = f'<span><b>{item["constituent_count"]}</b> ({", ".join(const_pills[:4])}{"..." if len(const_pills) > 4 else ""})</span>'

                # Contagion Rate Pill
                if item["constituent_count"] >= 2 and item["contagion_rate"] >= 50.0:
                    rate_cell = f'<span style="color: #ef4444; font-weight: 800;">{item["contagion_rate"]:.1f}% ({item["distributive_count"]}/{item["constituent_count"]}) 🚨</span>'
                elif item["contagion_rate"] > 0:
                    rate_cell = f'<span style="color: #facc15; font-weight: 700;">{item["contagion_rate"]:.1f}% ({item["distributive_count"]}/{item["constituent_count"]})</span>'
                else:
                    rate_cell = f'<span style="color: #4ade80; font-weight: 600;">0.0% (0/{item["constituent_count"]})</span>'

                # U/D Vol
                ud = item["avg_ud_ratio"]
                if ud >= 1.20:
                    ud_cell = f'<span style="color: #22c55e; font-weight: 700;">{ud:.2f}x (Accum.)</span>'
                elif ud >= 1.0:
                    ud_cell = f'<span style="color: #86efac; font-weight: 600;">{ud:.2f}x</span>'
                elif ud >= 0.85:
                    ud_cell = f'<span style="color: #facc15; font-weight: 600;">{ud:.2f}x</span>'
                else:
                    ud_cell = f'<span style="color: #ef4444; font-weight: 700;">{ud:.2f}x (Dist.)</span>'

                status_cell = f'<span class="{badge_cls}">{item["status_badge"]}</span>'
                mandate_color = "#fca5a5" if item["status_color"] == "red" else "#fef08a" if item["status_color"] == "yellow" else "#cbd5e1"
                mandate_cell = f'<span style="color: {mandate_color}; font-size: 0.82rem; font-weight: 500;">{item["tactical_mandate"]}</span>'

                theme_tbl.append(f'<tr class="{row_cls}">')
                theme_tbl.append(f'<td>{theme_name_cell}</td>')
                theme_tbl.append(f'<td>{wt_cell}</td>')
                theme_tbl.append(f'<td>{const_cell}</td>')
                theme_tbl.append(f'<td>{rate_cell}</td>')
                theme_tbl.append(f'<td>{ud_cell}</td>')
                theme_tbl.append(f'<td>{status_cell}</td>')
                theme_tbl.append(f'<td>{mandate_cell}</td>')
                theme_tbl.append('</tr>')

            theme_tbl.append('</tbody></table></div>')
            st.markdown("".join(theme_tbl), unsafe_allow_html=True)

            # Interactive Theme Drill-Down Expander
            with st.expander("🔍 Thematic Pack Inspector: Deep-Dive into Cluster Constituents", expanded=False):
                theme_options = [x["theme"] for x in theme_matrix]
                if theme_options:
                    selected_theme = st.selectbox("Select Thematic Cluster to Audit:", theme_options, key="theme_inspect_select")
                    target_theme_data = next((x for x in theme_matrix if x["theme"] == selected_theme), None)
                    if target_theme_data:
                        st.markdown(f"**Cluster: `{selected_theme}`** · Total Capital Allocation: **{target_theme_data['capital_weight']:.1f}%** · Contagion Rate: **{target_theme_data['contagion_rate']:.1f}%**")
                        st.caption(f"Canonical Sub-Industries: {', '.join(target_theme_data['sub_industries'])}")
                        
                        detail_rows = []
                        for c in target_theme_data["constituents"]:
                            detail_rows.append({
                                "Holding": c["resolved_symbol"],
                                "Official Sub-Industry": c["sub_industry"],
                                "Weight %": f"{c['weight']:.1f}%",
                                "Inst. Health (IDS)": f"{c['ids_score']}/10",
                                "15D IDVD": f"{c['idvd_15d']}d",
                                "15D Churn": f"{c['total_dd_15d'] - c['idvd_15d']}d",
                                "50D U/D Vol": f"{c['ud_ratio']:.2f}x",
                                "MA Defense": "🚨 50 SMA Breakdown" if c["oneil_50"] else "⚠️ < 21 EMA" if c["haber_21"] else "✅ Healthy",
                                "Distributive Status": "🔴 Under Liquidation" if c["is_distributive"] else "🟢 Accumulating",
                                "Prescription": c["prescription"]
                            })
                        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

            # Educational Methodology Expander for Pillar 1
            with st.expander("ℹ️ The O'Neil 50% Rule & Thematic Contagion Methodology"):
                st.markdown(r"""
                ### William J. O'Neil's 50% Rule (Why Theme Contagion Matters)
                In *How to Make Money in Stocks*, William O'Neil uncovered that:
                > *"50% of the movement of any leading stock is directly driven by its industry group (37%) and broad sector (13%). You cannot separate a stock from its pack."*
                
                When multiple stocks in the same industry cluster show distribution days, it indicates that **institutional portfolio managers (mutual funds, FIIs) are liquidating the entire sector basket**. Even an elite stock with phenomenal quarterly fundamentals will struggle to swim against a sector-wide liquidity outflow.
                
                ---
                ### Mathematical Formulation:
                $$\text{Contagion Rate}(T) = \frac{\sum_{i \in T} \mathbb{I}(\text{Stock}_i \text{ is Distributive})}{N_T} \times 100\%$$
                
                Where a holding is categorized as **Distributive** if:
                1. $\text{IDS Score} \ge 5$ (Acute or elevated institutional distribution).
                2. $\text{Total Distribution Days (15d)} \ge 2$ or $\text{IDVD (15d)} \ge 2$.
                3. Broken key Moving Averages (Ross Haber 21 EMA rule breach or O'Neil 50 SMA breakdown).
                
                ---
                ### Deterministic 3-Tier Classification Engine:
                To guarantee zero human errors or hallucinations, stocks are categorized via a 3-tier pipeline:
                1. **Tier 1 (Master DB)**: Resolves official sub-industry strings from `industry_map.json` (3,542 mappings) and `fundamentals_cache` in `kush_tracker.db` (3,769 verified rows).
                2. **Tier 2 (Exchange API)**: Dynamic lookup via yfinance `stock.info` (`sector` and `industry`) with automatic SQLite caching.
                3. **Tier 3 (Sub-Industry Rule Engine)**: Maps official sub-industries deterministically into Canonical Themes (e.g. *Drug Manufacturers* $\rightarrow$ *🏥 Pharmaceuticals & Healthcare*, *Auto Manufacturers/Parts* $\rightarrow$ *🏭 Automotive & Mobility*).
                """)

        # =============================================================
        # VIEW 2: INDIVIDUAL STOCK DEFENSE LEDGER (PILLARS 1-4)
        # =============================================================
        else:
            sorted_defense = sorted(active_defense_data, key=lambda x: (x['ids_score'], x['total_dd_15d']), reverse=True)

            tbl_lines = [
                '<style>',
                '.kush-table-container { overflow-x: auto; width: 100%; margin-bottom: 20px; }',
                '.kush-table { width: 100%; border-collapse: separate; border-spacing: 0 8px; font-family: "Inter", sans-serif; }',
                '.kush-table th { color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; text-align: left; padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.05); white-space: nowrap; }',
                '.kush-row { background: rgba(255,255,255,0.03); transition: all 0.2s ease; border-radius: 8px; }',
                '.kush-row:hover { background: rgba(255,255,255,0.06); transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }',
                '.kush-row td { padding: 14px 16px; font-size: 0.85rem; color: #f8fafc; border-top: 1px solid rgba(255,255,255,0.02); border-bottom: 1px solid rgba(255,255,255,0.02); white-space: nowrap; }',
                '.badge-dist-green { background: rgba(34, 197, 94, 0.15); color: #4ade80; padding: 3px 8px; border-radius: 8px; font-size: 0.75rem; font-weight: 700; border: 1px solid rgba(34, 197, 94, 0.3); white-space: nowrap; display: inline-block; }',
                '.badge-dist-yellow { background: rgba(234, 179, 8, 0.15); color: #facc15; padding: 3px 8px; border-radius: 8px; font-size: 0.75rem; font-weight: 700; border: 1px solid rgba(234, 179, 8, 0.3); white-space: nowrap; display: inline-block; }',
                '.badge-dist-red { background: rgba(239, 68, 68, 0.2); color: #f87171; padding: 3px 8px; border-radius: 8px; font-size: 0.75rem; font-weight: 800; border: 1px solid rgba(239, 68, 68, 0.4); box-shadow: 0 0 8px rgba(239, 68, 68, 0.3); white-space: nowrap; display: inline-block; }',
                '.row-red { background: rgba(239, 68, 68, 0.06) !important; border-left: 3px solid #ef4444 !important; }',
                '.row-orange { background: rgba(249, 115, 22, 0.05) !important; border-left: 3px solid #f97316 !important; }',
                '</style>',
                '<div class="kush-table-container">',
                '<table class="kush-table">',
                '<thead><tr><th>Ticker</th><th>Weight</th><th>Inst. Health (IDS)</th><th>15D IDVD (P1)</th><th>15D Churn (P2)</th><th>50D U/D Vol (P3)</th><th>MA Defense (P4)</th><th>Tactical Sell Mandate</th></tr></thead>',
                '<tbody>'
            ]

            for x in sorted_defense:
                is_high_risk = x['ids_score'] >= 5 or x['cluster_detected'] or x['oneil_50']
                is_mod_risk = x['ids_score'] >= 3 or x['haber_21']
                row_cls = "kush-row row-red" if is_high_risk else "kush-row row-orange" if is_mod_risk else "kush-row"

                ticker_cell = f'<a href="{x["ticker_url"]}" target="_blank" style="color: #60a5fa; text-decoration: none; font-weight: 700;">{x["display_ticker"]}</a>'
                wt_cell = f'<strong style="color: #cbd5e1;">{x["weight"]:.1f}%</strong>'

                if is_high_risk:
                    badge_cls = "badge-dist-red"
                    status_txt = f"🔴 IDS {x['ids_score']}/10 · Acute Risk"
                elif is_mod_risk:
                    badge_cls = "badge-dist-yellow"
                    status_txt = f"🟡 IDS {x['ids_score']}/10 · Stalling"
                else:
                    badge_cls = "badge-dist-green"
                    status_txt = f"🟢 IDS {x['ids_score']}/10 · Clean"
                health_cell = f'<span class="{badge_cls}">{status_txt}</span>'

                if x['idvd_15d'] >= 2:
                    idvd_cell = f'<span style="color: #ef4444; font-weight: 700;">{x["idvd_15d"]} days</span>'
                elif x['idvd_15d'] == 1:
                    idvd_cell = f'<span style="color: #f59e0b; font-weight: 600;">1 day</span>'
                else:
                    idvd_cell = '<span style="color: #64748b;">0</span>'

                if x['churn_15d'] >= 2:
                    churn_cell = f'<span style="color: #ef4444; font-weight: 700;">{x["churn_15d"]} days</span>'
                elif x['churn_15d'] == 1:
                    churn_cell = f'<span style="color: #f59e0b; font-weight: 600;">1 day</span>'
                else:
                    churn_cell = '<span style="color: #64748b;">0</span>'

                ud = x['ud_ratio']
                if ud >= 1.20:
                    ud_cell = f'<span style="color: #22c55e; font-weight: 700;">{ud:.2f}x (Accum.)</span>'
                elif ud >= 1.0:
                    ud_cell = f'<span style="color: #86efac; font-weight: 600;">{ud:.2f}x (Solid)</span>'
                elif ud >= 0.85:
                    ud_cell = f'<span style="color: #facc15; font-weight: 600;">{ud:.2f}x (Neutral)</span>'
                else:
                    ud_cell = f'<span style="color: #ef4444; font-weight: 700;">{ud:.2f}x (Dist.)</span>'

                if x['oneil_50']:
                    ma_cell = '<span style="color: #ef4444; font-weight: 700;">🚨 50 SMA Breakdown (Heavy Vol)</span>'
                elif x['haber_21']:
                    cb21 = x['metrics'].get('consecutive_closes_below_21', 2)
                    ma_cell = f'<span style="color: #f59e0b; font-weight: 700;">⚠️ {cb21} Closes &lt; 21 EMA (Haber Rule)</span>'
                else:
                    ma_cell = '<span style="color: #22c55e; font-weight: 600;">✅ Above Key MAs</span>'

                mandate_color = "#fca5a5" if is_high_risk else "#fef08a" if is_mod_risk else "#cbd5e1"
                mandate_cell = f'<span style="color: {mandate_color}; font-size: 0.85rem; font-weight: 500;">{x["prescription"]}</span>'

                tbl_lines.append(f'<tr class="{row_cls}">')
                tbl_lines.append(f'<td>{ticker_cell}</td>')
                tbl_lines.append(f'<td>{wt_cell}</td>')
                tbl_lines.append(f'<td>{health_cell}</td>')
                tbl_lines.append(f'<td style="text-align: center;">{idvd_cell}</td>')
                tbl_lines.append(f'<td style="text-align: center;">{churn_cell}</td>')
                tbl_lines.append(f'<td>{ud_cell}</td>')
                tbl_lines.append(f'<td>{ma_cell}</td>')
                tbl_lines.append(f'<td>{mandate_cell}</td>')
                tbl_lines.append('</tr>')

            tbl_lines.append('</tbody></table></div>')

            st.markdown("".join(tbl_lines), unsafe_allow_html=True)
            st.markdown("")

            # Historical Session Inspector
            with st.expander("🔍 Deep Session Inspector: Historical Distribution Days Drill-Down", expanded=False):
                ticker_options = [x['display_ticker'] for x in sorted_defense]
                if ticker_options:
                    sel_inspect = st.selectbox("Select Holding to Audit Session Footprints:", ticker_options, key="defense_inspect_ticker")
                    target_item = next((x for x in sorted_defense if x['display_ticker'] == sel_inspect), None)
                    if target_item:
                        sessions = target_item.get('sessions', [])
                        if sessions:
                            st.markdown(f"**Found {len(sessions)} distribution / churning session(s) for `{sel_inspect}` in the rolling 15-day window:**")
                            sess_rows = []
                            for s in sessions:
                                sess_rows.append({
                                    "Date": s.get('date'),
                                    "Distribution Type": s.get('type'),
                                    "Price Change %": f"{s.get('chg_pct', 0.0):+.2f}%",
                                    "Volume vs 50 SMA": f"{s.get('vol_ratio', 1.0):.2f}x Average",
                                    "Daily Closing Range (DCR)": f"{s.get('dcr', 0.5):.2f} ({'Lower 40%' if s.get('dcr', 0.5) < 0.40 else 'Upper Range'})",
                                    "Institutional Interpretation": "Heavy Institutional Dumping" if "Down-Volume" in s.get('type', '') else "Supply Absorption / Churning"
                                })
                            st.dataframe(pd.DataFrame(sess_rows), use_container_width=True, hide_index=True)
                        else:
                            st.success(f"🎉 **{sel_inspect}** has zero institutional distribution or churning sessions recorded in the last 15 trading days. Institutional accumulation is pristine.")

            # Educational Guide for Stock Distribution
            with st.expander("ℹ️ The CANSLIM Institutional Distribution Framework (Methodology & Rules)"):
                st.markdown(r"""
                ### Why Track Distribution Days on Individual Stocks?
                While William O'Neil pioneered Distribution Day tracking on broad market indexes (NIFTY / NASDAQ), top CANSLIM portfolio managers including **David Ryan**, **Mark Minervini**, and **Ross Haber** apply these exact volume footprints directly to individual portfolio holdings. 
                
                Individual stock distribution reveals when institutional sponsors (mutual funds, FIIs, hedge funds) are quietly liquidating their inventory before a major trend breakdown occurs.
                
                ---
                ### The 4 Pillars Defined:
                1. **Pillar 1: Institutional Down-Volume Day (IDVD)**
                   * **Formula**: Daily Price Change $\le -1.5\%$ on Volume $>$ 50-day SMA Volume, with **Daily Closing Range (DCR)** $< 0.40$.
                   * **The DCR Absorption Filter**: $DCR = \frac{Close - Low}{High - Low}$. If a stock drops $2\%$ intraday but rallies to close in the upper half of its range ($DCR \ge 0.50$), it is an **institutional absorption hammer / shakeout**, NOT distribution. O'Neil and Minervini emphasize never selling on high-volume days where institutions step in to absorb shares.
                
                2. **Pillar 2: Churning / Stalling Day**
                   * **Formula**: Daily Price Change between $-0.5\%$ and $+0.5\%$ on Volume $\ge 1.25\times$ 50-day SMA Volume, with $DCR < 0.40$.
                   * **The Trap**: Massive trading volume with negligible upward price progress. As O'Neil warned, *effort without reward* indicates large funds are unloading heavy blocks into retail buying enthusiasm.
                
                3. **Pillar 3: 50-Day Up/Down Volume Ratio (U/D Ratio)**
                   * **Formula**: Total volume traded on up days divided by total volume traded on down days over the past 50 sessions:
                     $$\text{U/D Ratio} = \frac{\sum_{t=1}^{50} V_t \cdot \mathbb{I}(\Delta P_t > 0)}{\sum_{t=1}^{50} V_t \cdot \mathbb{I}(\Delta P_t < 0)}$$
                   * **Thresholds**:
                     * **$\ge 1.20\times$**: Elite Institutional Accumulation (A/D Rating A).
                     * **$1.00\times - 1.19\times$**: Constructive accumulation.
                     * **$< 0.90\times$**: Supply overhang; down days are consistently heavier than up days.
                
                4. **Pillar 4: Distribution Clustering & Moving Average Breaches**
                   * **Acute Cluster Rule**: $\ge 3$ distribution sessions (IDVD or Churning) within any rolling 15-day window indicates systematic fund liquidation.
                   * **Ross Haber 21 EMA Rule**: Two consecutive closes below the 21-day EMA signal institutional support failure. Mandate: trim $1/3$ to $1/2$ position.
                   * **O'Neil 50 SMA Breakdown Rule**: A decisive close below the 50-day SMA on volume $\ge 1.20\times$ 50 SMA volume indicates complete institutional surrender. Mandate: immediate exit or capital preservation trim.
                """)

    st.markdown("---")
    
    # 5. Main Holdings Table (Fixed Sorting)
    st.markdown("### 📋 Holdings Analysis")
    
    with st.expander("ℹ️ How to Interpret the Holdings Table Metrics"):
        st.markdown("""
        **Core Risk & Health Metrics:**
        *   **Port Hit:** The percentage of your TOTAL portfolio you have lost from the stock's highest peak since you bought it. (If you hold a 10% position that drops 20%, your Port Hit is 2.0%).
        *   **Days Exit:** The estimated number of days it will take to safely exit your entire position assuming you cannot be more than 10% of the stock's Average Daily Traded Value (ADTV). (<1d = Safe, >3d = Illiquidity Trap).
        *   **Trend:** The mechanical health of the stock's price action (Strong = above 8 & 21 EMA; Warning = below 21 EMA; Broken = below 50 SMA).
        *   **52W Dist:** How far the current price is below its 52-week high.
        *   **ADR% / Vol:** The Average Daily Range (volatility) and the recent Relative Volume (1.0x = normal).
        *   **RS Score:** Mansfield Relative Strength score (0-100) comparing the stock's performance against the NIFTY 500 benchmark. (80+ = Elite Leader).
        *   **Sortino (6M):** Downside-risk-adjusted alpha over 6 months (126 days). Penalizes only downside volatility while ignoring upside explosiveness. (🏆 ≥ 2.5 Elite, 🟢 ≥ 1.5 Solid, 🟡 0.8–1.4 Average, 🔴 < 0.8 High Downside Drag).
        *   **Inst. Health (Pillars 1–4):** Institutional distribution health based on O'Neil, Ryan, Minervini & Haber rules:
            - **Pillar 1 (IDVD):** Price drop ≤ -1.5% on volume > 50 SMA volume with Daily Closing Range (DCR) < 0.40. Excludes absorption/shakeout hammers (DCR ≥ 0.50).
            - **Pillar 2 (Churning):** Price stagnant (-0.5% to +0.5%) on volume ≥ 1.25x 50 SMA volume with DCR < 0.40 (heavy turnover with zero upward progress).
            - **Pillar 3 (50D U/D Vol Ratio):** Ratio of up-day volume to down-day volume over 50 sessions (>1.2x accumulation, <0.9x distribution dominance).
            - **Pillar 4 (Clusters & MA Breaches):** Acute sell clusters (≥ 3 DD in 15 sessions) or structural breaks (≥ 2 closes below 21 EMA or heavy-volume 50 SMA break).
            - **Badges:** `🟢 0 DD (1.4x)` (Clean), `🟡 2 DD (0.9x)` (Stalling), `🔴 3 DD 🚨` (Acute Cluster / Violation).
        *   **Action:** The mechanical, algorithmic action generated by the engine based on the rules.
        """)

    with st.expander("ℹ️ How to Interpret Mike Webster RS Signals"):
        st.markdown("""
        **Mike Webster's RS Signal** is an algorithmic portfolio management tool that tells you when a stock is losing its institutional sponsorship relative to the benchmark index (NIFTY 50).
        It tracks the moving averages of the **RS Line** (Stock Price / Index Price).

        *   🟢 **Hold (Strong)**: RS Line is above its 10 EMA. The stock is generating pure alpha.
        *   🟡 **Quick (Trim)**: RS Line fell below its 10 EMA. Momentum is stalling; trim if the stock is extended.
        *   🟠 **Quicksand (Warning)**: RS Line fell below its 21 EMA. The stock is losing alpha rapidly. Tighten stops.
        *   🔴 **Grateful Dead (Exit)**: RS Line fell below its 50 SMA. The relative trend is completely broken. Exit.
        """)
        
    # Prepare data for table - Use raw types for sorting!
    current_tml = get_current_tml_leaders("INDIA")
    hat_stocks = get_hat_stocks("INDIA")
    if not current_tml:
        current_tml = get_current_tml_leaders("NIFTY TOTAL MARKET")
        hat_stocks = get_hat_stocks("NIFTY TOTAL MARKET")
        
    from systematic_engine import get_momentum_badges
    try:
        super_compounders, rockets = get_momentum_badges()
    except Exception as e:
        print(f"Failed to fetch momentum badges: {e}")
        super_compounders, rockets = set(), set()
        
    bse_reverse_map = {}
    try:
        if os.path.exists('bse_mapping.json'):
            with open('bse_mapping.json', 'r') as f:
                bmap = json.load(f)
                bse_reverse_map.update({str(v): k for k, v in bmap.items()})
        if os.path.exists('bse_yf_mapping.json'):
            with open('bse_yf_mapping.json', 'r') as f:
                yfmap = json.load(f)
                for k, v in yfmap.items():
                    bse_reverse_map[str(k)] = str(v).replace('.BO', '').replace('.NS', '')
    except Exception:
        pass

    table_data = []
    tml_capital_pct = 0.0
    laggard_capital_pct = 0.0
    
    for d in decisions:
        loss_metrics = d.get('loss_metrics', {})
        port_weight = loss_metrics.get('portfolio_weight', 0)
        
        raw_ticker = str(d['ticker'])
        display_ticker = bse_reverse_map.get(raw_ticker, raw_ticker)
        exchange = 'BSE' if raw_ticker.isdigit() else 'NSE'
        
        is_tml = raw_ticker in current_tml
        is_hat = raw_ticker in hat_stocks
        
        if is_tml:
            tml_capital_pct += port_weight
            
        rs_score_val = d.get('rs_score', 0) or 0
        if rs_score_val < 50:
            laggard_capital_pct += port_weight

        action_val = d.get('action', 'HOLD')
        action_map = {
            'EXIT': '🔴 EXIT',
            'TRIM': '🟡 TRIM',
            'HOLD': '⚪ HOLD',
            'ADD': '🟢 ADD'
        }
        
        dist_m = d.get('distribution_metrics', {})
        ids_score_val = dist_m.get('ids_score', 0)
        ud_vol_val = dist_m.get('ud_volume_ratio_50d', 1.0)
        tot_dd_val = dist_m.get('total_dist_days_15d', 0)
        presc_val = dist_m.get('prescription', '')
        is_cluster_val = dist_m.get('cluster_detected', False)
        is_50_breach = dist_m.get('oneil_50_breach', False)
        is_21_breach = dist_m.get('haber_21_breach', False)
        
        presc_clean = str(presc_val).replace("'", "&#39;").replace('"', '&quot;')
        if ids_score_val >= 5 or is_cluster_val or is_50_breach:
            badge_inst = f"<span class='badge-dist-red' title='{presc_clean}'>🔴 {tot_dd_val} DD 🚨</span>"
        elif ids_score_val >= 3 or tot_dd_val >= 2 or is_21_breach:
            badge_inst = f"<span class='badge-dist-yellow' title='{presc_clean}'>🟡 {tot_dd_val} DD ({ud_vol_val:.2f}x)</span>"
        else:
            badge_inst = f"<span class='badge-dist-green' title='{presc_clean}'>🟢 {tot_dd_val} DD ({ud_vol_val:.2f}x)</span>"

        table_data.append({
            'TickerURL': f"https://in.tradingview.com/chart/?symbol={exchange}:{display_ticker}",
            'DisplayTicker': display_ticker,
            'Ticker': raw_ticker,
            '👑 TML': '👑 Yes' if is_tml else ('🎩 Yes' if is_hat else ''),
            'Chart (60D)': d.get('sparkline', []),
            'Sector': d.get('sector', 'N/A'),
            'Industry': d.get('industry', 'N/A'),
            'Portfolio %': loss_metrics.get('portfolio_weight', 0),
            'Port Hit %': d.get('port_hit_pct', 0.0),
            'Active Stop': d.get('active_stop', 0.0),
            'Stop Breached': d.get('stop_breached', False),
            'Heat %': d.get('position_heat_pct', 0.0),
            'Dollar Risk': d.get('dollar_risk', 0.0),
            'Trend': f"{get_trend_emoji(d.get('trend_state', ''))} {d.get('trend_state', 'N/A').replace('🟢 ', '').replace('🟡 ', '').replace('🟠 ', '').replace('🔴 ', '')}",
            '8 EMA': d.get('above_8', False),
            '21 EMA': d.get('above_21', False),
            '50 SMA': d.get('above_50', False),
            'MA DNA': (d.get('ma_dna') or {}).get('dna_badge', '⚡ 21-EMA Low Rider'),
            'Active Spine Price': (d.get('ma_dna') or {}).get('active_spine_price', 0.0),
            'EMA 21 Low': d.get('ema_21_low', 0.0) or (d.get('ma_dna') or {}).get('ema_21_low', 0.0),
            '52W Dist': d.get('distance_from_52w_high', 0),
            'Dist 21EMA': d.get('dist_from_21ema', 999),
            'Ext 50SMA': d.get('ext_50sma', None),
            'Ext 200SMA': d.get('ext_200sma', None),
            'ADR%': d.get('adr_pct', 0),
            'Rel Vol': d.get('rel_vol', 0),
            'Sales Growth': fundas_cache.get(f"{d['ticker']}.NS", {}).get('sales_growth', fundas_cache.get(d['ticker'], {}).get('sales_growth', 0.0)),
            'RS Score': rs_score_val,
            '1M Ret': d.get('ret_1m', 0.0),
            '3M Ret': d.get('ret_3m', 0.0),
            'Smooth RS': d.get('smooth_rs', 0.0),
            'Sortino 3M': d.get('sortino_3m', 0.0),
            'Sortino 6M': d.get('sortino_6m', 0.0),
            'rs_blue_dot': d.get('rs_blue_dot', False),
            'Power Days': f"{d.get('power_days_3m', 0)} / {d.get('dist_days_3m', 0)}",
            'Inst. Health': badge_inst,
            'IDS Score': ids_score_val,
            'Dist Metrics': dist_m,
            'Days to Exit': d.get('days_to_exit', 0.0),
            'Add Ready': d.get('add_on_ready', False),
            'Action': action_map.get(action_val, action_val),
            'Reason': d.get('reason', '')[:60],
            'Webster': d.get('webster_signal', '⚪ Unknown')
        })
        
    tab_ledger, tab_defense_radar = st.tabs([
        "📋 Master Execution Ledger",
        "🛡️ Institutional Defense & Liquidation Radar (Pillars 1–4)"
    ])
    
    with tab_ledger:
        # Render TML & Laggard Concentration Metrics
        from components import render_metric_card
        c1, c2, c3 = st.columns(3)
        with c1:
            tml_color = "green-text" if tml_capital_pct >= 50 else "yellow-text" if tml_capital_pct >= 25 else "red-text"
            render_metric_card("👑 TML Concentration", f"{tml_capital_pct:.1f}%", color_class=tml_color)
        with c2:
            laggard_color = "red-text" if laggard_capital_pct > 10 else "yellow-text" if laggard_capital_pct > 0 else "green-text"
            render_metric_card("🐢 Laggard Exposure (RS<50)", f"{laggard_capital_pct:.1f}%", color_class=laggard_color)
        
        st.markdown("")
        
        df_display = pd.DataFrame(table_data)
        if not df_display.empty:
            df_display = df_display.sort_values(by='RS Score', ascending=False)
            
        st.info("💡 **Color & Symbol Guide:** 🟧 **Orange Row** = Extended > 15% from 52W High | 🟦 **Blue Row** = Within 5% of 21 EMA (Add-on Opportunity) | 👑 **Apex Predator** | ⚡ **Super Compounder** | 🚀 **Rocket Rank** | 🔵 **RS Blue Dot**")
        
        # Sorting & Smooth RS / Sortino Filter Controls
        sort_col1, sort_col2, sort_col3 = st.columns([1.4, 1.3, 1.3])
        with sort_col1:
            sort_metric = st.selectbox(
                "Sort Table By:", 
                ["RS Score (High to Low)", "Smooth RS / 20D ATR% (High to Low)", "Distribution Risk (IDS High to Low)", "Sortino 6M (High to Low)", "Portfolio Heat % (High to Low)", "1M Ret (High to Low)", "1M Ret (Low to High)", "3M Ret (High to Low)", "3M Ret (Low to High)", "Sales YoY (High to Low)", "Days to Exit (High to Low)", "Portfolio % (High to Low)", "Distance to 52W High", "ADR% (Volatility)", "Ticker (A-Z)"]
            )
        with sort_col2:
            port_min_smooth_rs = st.slider(
                "⚡ Min Smooth RS (RS / 20D ATR%)",
                min_value=-10.0, max_value=25.0, value=-10.0, step=1.0,
                key="port_ledger_min_smooth_rs",
                help="Filter active portfolio holdings by minimum Smooth RS (Composite RS % / 20D ATR%). Default -10.0 shows all holdings."
            )
        with sort_col3:
            port_min_sortino = st.slider(
                "🛡️ Min Sortino (6M)",
                min_value=-5.0, max_value=10.0, value=-5.0, step=0.5,
                key="port_ledger_min_sortino_6m",
                help="Filter active portfolio holdings by minimum 6-Month Sortino ratio. Default -5.0 shows all holdings."
            )
            
        html_table = """
<style>
.kush-table-container {
    overflow-x: auto;
    width: 100%;
}
.kush-table-container::-webkit-scrollbar {
    height: 8px;
}
.kush-table-container::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 4px;
}
.kush-table-container::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 4px;
}
.kush-table-container::-webkit-scrollbar-thumb:hover {
    background: #475569;
}

.kush-table { width: 100%; border-collapse: separate; border-spacing: 0 8px; font-family: 'Inter', sans-serif; }
.kush-table th { color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; text-align: left; padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.05); white-space: nowrap; }
.kush-table th:first-child {
    position: sticky;
    left: 0;
    z-index: 30;
    background: #0f172a;
    min-width: 130px;
    box-shadow: 4px 0 10px rgba(0, 0, 0, 0.4);
}
.kush-row { background: rgba(255,255,255,0.03); transition: all 0.2s ease; border-radius: 8px; }
.kush-row:hover { background: rgba(255,255,255,0.06); transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
.kush-row td { padding: 14px 16px; font-size: 0.85rem; color: #f8fafc; border-top: 1px solid rgba(255,255,255,0.02); border-bottom: 1px solid rgba(255,255,255,0.02); white-space: nowrap; }
.kush-row td:first-child {
    position: sticky;
    left: 0;
    z-index: 20;
    background: #141d31;
    border-left: 1px solid rgba(255,255,255,0.02);
    border-top-left-radius: 8px;
    border-bottom-left-radius: 8px;
    font-weight: 600;
    min-width: 130px;
    box-shadow: 4px 0 10px rgba(0, 0, 0, 0.4);
}
.kush-row:hover td:first-child {
    background: #1e2a47;
    z-index: 25;
}
.kush-row td:last-child { border-right: 1px solid rgba(255,255,255,0.02); border-top-right-radius: 8px; border-bottom-right-radius: 8px; }
.badge-tml { background: rgba(234, 179, 8, 0.15); color: #fbbf24; padding: 4px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: 700; border: 1px solid rgba(234, 179, 8, 0.3); }
.badge-action-add { background: #22c55e; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; box-shadow: 0 0 10px rgba(34, 197, 94, 0.4); }
.badge-action-trim { background: #eab308; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; box-shadow: 0 0 10px rgba(234, 179, 8, 0.4); }
.badge-action-exit { background: #ef4444; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; box-shadow: 0 0 10px rgba(239, 68, 68, 0.4); }
.badge-action-hold { background: rgba(255, 255, 255, 0.1); color: #cbd5e1; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(255,255,255,0.2); }
.badge-dist-green { background: rgba(34, 197, 94, 0.15); color: #4ade80; padding: 3px 8px; border-radius: 8px; font-size: 0.75rem; font-weight: 700; border: 1px solid rgba(34, 197, 94, 0.3); white-space: nowrap; display: inline-block; }
.badge-dist-yellow { background: rgba(234, 179, 8, 0.15); color: #facc15; padding: 3px 8px; border-radius: 8px; font-size: 0.75rem; font-weight: 700; border: 1px solid rgba(234, 179, 8, 0.3); white-space: nowrap; display: inline-block; }
.badge-dist-red { background: rgba(239, 68, 68, 0.2); color: #f87171; padding: 3px 8px; border-radius: 8px; font-size: 0.75rem; font-weight: 800; border: 1px solid rgba(239, 68, 68, 0.4); box-shadow: 0 0 8px rgba(239, 68, 68, 0.3); white-space: nowrap; display: inline-block; }
.rs-bar-bg { background: rgba(255, 255, 255, 0.1); width: 60px; height: 6px; border-radius: 3px; overflow: hidden; display: inline-block; vertical-align: middle; margin-left: 8px; }
.rs-bar-fill { height: 100%; border-radius: 3px; }

/* Dynamic Row Animations */
@keyframes pulse-orange-row {
    0% { box-shadow: inset 4px 0 0 #f97316, inset 0 0 10px rgba(249, 115, 22, 0.05); }
    50% { box-shadow: inset 4px 0 0 #f97316, inset 0 0 30px rgba(249, 115, 22, 0.25); }
    100% { box-shadow: inset 4px 0 0 #f97316, inset 0 0 10px rgba(249, 115, 22, 0.05); }
}
@keyframes pulse-blue-row {
    0% { box-shadow: inset 4px 0 0 #38bdf8, inset 0 0 10px rgba(56, 189, 248, 0.05); }
    50% { box-shadow: inset 4px 0 0 #38bdf8, inset 0 0 30px rgba(56, 189, 248, 0.25); }
    100% { box-shadow: inset 4px 0 0 #38bdf8, inset 0 0 10px rgba(56, 189, 248, 0.05); }
}
@keyframes pulse-red-row {
    0% { box-shadow: inset 4px 0 0 #ef4444, inset 0 0 10px rgba(239, 68, 68, 0.05); }
    50% { box-shadow: inset 4px 0 0 #ef4444, inset 0 0 30px rgba(239, 68, 68, 0.25); }
    100% { box-shadow: inset 4px 0 0 #ef4444, inset 0 0 10px rgba(239, 68, 68, 0.05); }
}
.row-orange { background: rgba(249, 115, 22, 0.05) !important; animation: pulse-orange-row 2s infinite !important; }
.row-blue { background: rgba(56, 189, 248, 0.05) !important; animation: pulse-blue-row 2s infinite !important; }
.row-red { background: rgba(239, 68, 68, 0.05) !important; animation: pulse-red-row 1.5s infinite !important; }

/* Sticky column coloring for dynamic states */
.row-orange td:first-child {
    background: #1f1b26 !important;
    border-left: 4px solid #f97316 !important;
}
.row-orange:hover td:first-child {
    background: #2b2333 !important;
    z-index: 25;
}
.row-blue td:first-child {
    background: #112239 !important;
    border-left: 4px solid #38bdf8 !important;
}
.row-blue:hover td:first-child {
    background: #182e4d !important;
    z-index: 25;
}
.row-red td:first-child {
    background: #26161d !important;
    border-left: 4px solid #ef4444 !important;
}
.row-red:hover td:first-child {
    background: #361c27 !important;
    z-index: 25;
}

/* Dynamic Intelligent Banner */
.monitor-banner { text-align: center; font-weight: 800; font-size: 0.9rem; letter-spacing: 4px; padding: 12px; margin-bottom: 16px; border-radius: 8px; text-transform: uppercase; }
@keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); } 70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); } 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); } }
@keyframes pulse-orange { 0% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0.6); } 70% { box-shadow: 0 0 0 10px rgba(249, 115, 22, 0); } 100% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0); } }
@keyframes pulse-blue { 0% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.6); } 70% { box-shadow: 0 0 0 10px rgba(56, 189, 248, 0); } 100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); } }
@keyframes pulse-green { 0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.6); } 70% { box-shadow: 0 0 0 10px rgba(34, 197, 94, 0); } 100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); } }
.banner-red { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); animation: pulse-red 2s infinite; }
.banner-orange { background: rgba(249, 115, 22, 0.15); color: #f97316; border: 1px solid rgba(249, 115, 22, 0.3); animation: pulse-orange 2s infinite; }
.banner-blue { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); animation: pulse-blue 2s infinite; }
.banner-green { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); animation: pulse-green 2s infinite; }
</style>
"""

        # Filter table_data by Smooth RS & Sortino sliders
        filtered_table_data = [
            x for x in table_data
            if (x.get('Smooth RS', 0.0) or 0.0) >= port_min_smooth_rs and (x.get('Sortino 6M', 0.0) or 0.0) >= port_min_sortino
        ]

        # Dynamically sort based on User Selection
        if sort_metric == "Distribution Risk (IDS High to Low)":
            sorted_table_data = sorted(filtered_table_data, key=lambda x: x.get('IDS Score', 0) or 0, reverse=True)
        elif sort_metric == "Smooth RS / 20D ATR% (High to Low)":
            sorted_table_data = sorted(filtered_table_data, key=lambda x: x.get('Smooth RS', -999) or -999, reverse=True)
        elif sort_metric == "Sortino 6M (High to Low)":
            sorted_table_data = sorted(filtered_table_data, key=lambda x: x.get('Sortino 6M', -999) or -999, reverse=True)
        elif sort_metric == "Portfolio Heat % (High to Low)":
            sorted_table_data = sorted(filtered_table_data, key=lambda x: x.get('Heat %', 0) or 0, reverse=True)
        elif sort_metric == "Days to Exit (High to Low)":
            sorted_table_data = sorted(filtered_table_data, key=lambda x: x.get('Days to Exit', 0) or 0, reverse=True)
        elif sort_metric == "1M Ret (High to Low)":
            sorted_table_data = sorted(filtered_table_data, key=lambda x: x.get('1M Ret', 0) or 0, reverse=True)
        elif sort_metric == "1M Ret (Low to High)":
            sorted_table_data = sorted(filtered_table_data, key=lambda x: x.get('1M Ret', 0) or 0)
        elif sort_metric == "3M Ret (High to Low)":
            sorted_table_data = sorted(filtered_table_data, key=lambda x: x.get('3M Ret', 0) or 0, reverse=True)
        elif sort_metric == "3M Ret (Low to High)":
            sorted_table_data = sorted(filtered_table_data, key=lambda x: x.get('3M Ret', 0) or 0)
        elif sort_metric == "Portfolio % (High to Low)":
            sorted_table_data = sorted(filtered_table_data, key=lambda x: x.get('Portfolio %', 0) or 0, reverse=True)
        elif sort_metric == "Sales YoY (High to Low)":
            sorted_table_data = sorted(filtered_table_data, key=lambda x: x.get('Sales Growth', 0) or 0, reverse=True)
        elif sort_metric == "Distance to 52W High":
            sorted_table_data = sorted(filtered_table_data, key=lambda x: x.get('52W Dist', 999) or 999)
        elif sort_metric == "ADR% (Volatility)":
            sorted_table_data = sorted(filtered_table_data, key=lambda x: x.get('ADR%', 0) or 0, reverse=True)
        elif sort_metric == "Ticker (A-Z)":
            sorted_table_data = sorted(filtered_table_data, key=lambda x: x.get('Ticker', ''))
        else:
            # Default to RS Score
            sorted_table_data = sorted(filtered_table_data, key=lambda x: x.get('RS Score', 0) or 0, reverse=True)
        
        # Determine Banner Logic based on Portfolio State
        has_exits = any('EXIT' in str(r.get('Action', '')) for r in table_data)
        has_extended = any(r.get('52W Dist', 0) > 15.0 for r in table_data)
        has_opportunity = any(r.get('Dist 21EMA', 999) <= 5.0 for r in table_data)
        
        if has_exits:
            banner_cls = "banner-red"
            banner_text = "🚨 ACTION REQUIRED: EXIT SIGNALS DETECTED"
        elif has_extended:
            banner_cls = "banner-orange"
            banner_text = "⚠️ WARNING: EXTENDED POSITIONS DETECTED (>15% 52W Dist)"
        elif has_opportunity:
            banner_cls = "banner-blue"
            banner_text = "🎯 OPPORTUNITY: ADD-ON ZONES DETECTED (Near 21 EMA)"
        else:
            banner_cls = "banner-green"
            banner_text = "✅ PORTFOLIO HEALTHY: NO URGENT ACTIONS"
            
        html_lines = []
        html_lines.append(f'<div style="background-color: #0f172a; padding: 20px; border-radius: 16px; border: 1px solid #1e293b; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">')
        html_lines.append(f'<div class="monitor-banner {banner_cls}">{banner_text}</div>')
        html_lines.append('<div class="kush-table-container">')
        html_lines.append('<table class="kush-table">')
        html_lines.append('<thead><tr><th>Ticker</th><th>TML</th><th>Port %</th><th>Port Hit</th><th>Stop / Heat</th><th>Days Exit</th><th>Trend</th><th>8/21/50 EMA</th><th>52W Dist</th><th>Ext 50/200</th><th>Sales YoY</th><th>ADR% / Vol</th><th>RS Score</th><th>1M Ret</th><th>3M Ret</th><th>Smooth RS</th><th>Sortino (6M)</th><th>Inst. Health</th><th>Webster</th><th>Add</th><th>Action</th><th>Reason</th></tr></thead>')
        html_lines.append('<tbody>')
        
        for r in sorted_table_data:
            # Determine row class
            port_hit = r.get('Port Hit %', 0)
            row_cls = "kush-row"
            
            if port_hit >= 2.0:
                row_cls += " row-red"
            elif port_hit >= 1.0:
                row_cls += " row-orange"
            elif r.get('52W Dist', 0) > 15.0:
                row_cls += " row-orange"
            elif r.get('Dist 21EMA', 999) <= 5.0:
                row_cls += " row-blue"
                
            add_icon = "🟩" if r.get('Add Ready') else "<span style='color: #475569'>-</span>"
            e8 = "✅" if r.get('8 EMA') else "❌"
            e21 = "✅" if r.get('21 EMA') else "❌"
            e50 = "✅" if r.get('50 SMA') else "❌"
            
            act = r.get('Action', 'HOLD')
            if 'ADD' in act: act_html = f"<span class='badge-action-add'>{act}</span>"
            elif 'TRIM' in act: act_html = f"<span class='badge-action-trim'>{act}</span>"
            elif 'EXIT' in act: act_html = f"<span class='badge-action-exit'>{act}</span>"
            else: act_html = f"<span class='badge-action-hold'>{act}</span>"
            
            rs = r.get('RS Score', 0)
            rs_color = "#22c55e" if rs >= 80 else "#eab308" if rs >= 50 else "#ef4444"
            rs_html = f"<span style='color:{rs_color}; font-weight:bold;'>{rs:.0f}</span><div class='rs-bar-bg'><div class='rs-bar-fill' style='width:{rs}%; background:{rs_color};'></div></div>"
            
            tml_val = r.get('👑 TML', '')
            if '👑' in tml_val:
                tml_html = "<span class='badge-tml'>👑 TML</span>"
            elif '🎩' in tml_val:
                tml_html = "<span class='badge-tml' style='background: rgba(168, 85, 247, 0.15); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.3);'>🎩 NEXT 20</span>"
            else:
                tml_html = ""
            ticker_sym = r.get('DisplayTicker', r.get('Ticker', ''))
            
            ext50 = r.get("Ext 50SMA")
            ext200 = r.get("Ext 200SMA")
            ext50_str = f"{ext50:+.1f}%" if (ext50 is not None and not pd.isna(ext50)) else "N/A"
            ext200_str = f"{ext200:+.1f}%" if (ext200 is not None and not pd.isna(ext200)) else "N/A"
            ext50_color = "#ef4444" if (ext50 is not None and not pd.isna(ext50) and ext50 > 30) else "#f97316" if (ext50 is not None and not pd.isna(ext50) and ext50 > 15) else "#94a3b8"
            ext200_color = "#ef4444" if (ext200 is not None and not pd.isna(ext200) and ext200 > 60) else "#f97316" if (ext200 is not None and not pd.isna(ext200) and ext200 > 30) else "#94a3b8"
            
            fm_badges = ""
            if ticker_sym in super_compounders:
                fm_badges += " ⚡"
            if ticker_sym in rockets:
                fm_badges += " 🚀"
            
            html_lines.append(f'<tr class="{row_cls}">')
            blue_dot_html = " <span class='rs-blue-dot-glow'>🔵</span>" if r.get('rs_blue_dot', False) else ""
            ticker_url = r.get('TickerURL', '#')
            html_lines.append(f'<td><a href="{ticker_url}" target="_blank" style="color: #60a5fa; text-decoration: none; font-weight: 700;">{ticker_sym}</a>{blue_dot_html}{fm_badges}</td>')
            html_lines.append(f'<td>{tml_html}</td>')
            html_lines.append(f'<td><strong style="color: #e2e8f0;">{r.get("Portfolio %", 0):.1f}%</strong></td>')
            
            # Color Port Hit
            hit_color = "#ef4444" if port_hit >= 2.0 else "#f97316" if port_hit >= 1.0 else "#94a3b8"
            hit_str = f"▼ {port_hit:.2f}%" if port_hit > 0 else "-"
            html_lines.append(f'<td><span style="color: {hit_color}; font-weight: bold;">{hit_str}</span></td>')
            
            # Active Stop & Open Portfolio Heat
            act_stop = r.get("Active Stop", 0.0)
            p_heat = r.get("Heat %", 0.0)
            is_breached = r.get("Stop Breached", False)
            
            heat_color = "#ef4444" if p_heat > 0.75 else "#f59e0b" if p_heat > 0.4 else "#10b981"
            if is_breached:
                stop_str = f"₹{act_stop:,.1f}"
                heat_badge = f'<div style="color:#ef4444; font-weight:700; font-size:0.75rem;">{p_heat:.2f}% Heat <span style="font-size:0.65rem; background:rgba(239,68,68,0.25); color:#fca5a5; padding:1px 4px; border-radius:4px; font-weight:800;">HIT</span></div>'
            else:
                stop_str = f"₹{act_stop:,.1f}" if act_stop > 0 else "N/A"
                heat_badge = f'<div style="color:{heat_color}; font-weight:700; font-size:0.75rem;">{p_heat:.2f}% Heat</div>'
                
            html_lines.append(f'<td><div style="font-size:0.8rem; color:#cbd5e1;">{stop_str}</div>{heat_badge}</td>')
            
            # Days to Exit
            dtl = r.get("Days to Exit", 0)
            dtl_color = "#10b981" if dtl < 1.0 else "#f59e0b" if dtl <= 3.0 else "#ef4444"
            dtl_fw = "800" if dtl > 3.0 else "600"
            html_lines.append(f'<td><span style="color: {dtl_color}; font-weight: {dtl_fw};">{dtl:.1f}d</span></td>')
            
            
            html_lines.append(f'<td>{r.get("Trend", "")}</td>')
            dna_lbl = str(r.get("MA DNA", "⚡ 21-EMA Low Rider"))
            e21_low_v = float(r.get("EMA 21 Low", 0.0) or 0.0)
            spine_p = float(r.get("Active Spine Price", 0.0) or e21_low_v)
            sp_str = f" ₹{spine_p:,.0f}" if spine_p > 0 else ""
            if "50-SMA" in dna_lbl:
                dna_pill = f"<div style='font-size:0.63rem; color:#fbbf24; font-weight:700; letter-spacing:0px; margin-top:2px;' title='John Boik 50-SMA Monster Ride Spine'>🦖 50-SMA{sp_str}</div>"
            elif "10-EMA" in dna_lbl:
                dna_pill = f"<div style='font-size:0.63rem; color:#38bdf8; font-weight:700; letter-spacing:0px; margin-top:2px;' title='High-Velocity 10-EMA Sprinter'>🚀 10-EMA{sp_str}</div>"
            else:
                e21_str = f" ₹{e21_low_v:,.0f}" if e21_low_v > 0 else ""
                dna_pill = f"<div style='font-size:0.63rem; color:#34d399; font-weight:700; letter-spacing:0px; margin-top:2px;' title='21-EMA of Lows Anti-Wick Floor'>⚡ 21-Low{e21_str}</div>"
            html_lines.append(f'<td style="font-size:0.7rem; letter-spacing: 2px;">{e8}{e21}{e50}{dna_pill}</td>')
            
            dist52 = r.get("52W Dist", 0)
            dist52_str = f"{dist52:.1f}%" if (dist52 is not None and not pd.isna(dist52)) else "N/A"
            adr = r.get("ADR%", 0)
            adr_str = f"{adr:.1f}%" if (adr is not None and not pd.isna(adr)) else "N/A"
            rel_vol = r.get("Rel Vol", 0)
            rel_vol_str = f"{rel_vol:.1f}x" if (rel_vol is not None and not pd.isna(rel_vol)) else "N/A"
            sales = r.get("Sales Growth", 0)
            sales_str = f"+{sales:.1f}%" if sales > 0 else f"{sales:.1f}%" if sales < 0 else "N/A"
            sales_color = "#22c55e" if sales >= 20 else "#eab308" if sales > 0 else "#ef4444"
            
            html_lines.append(f'<td>{dist52_str}</td>')
            html_lines.append(f'<td><span style="color:{ext50_color}">{ext50_str}</span> <span style="color:#475569;">|</span> <span style="color:{ext200_color}">{ext200_str}</span></td>')
            html_lines.append(f'<td><strong style="color: {sales_color};">{sales_str}</strong></td>')
            html_lines.append(f'<td><span style="color:#93c5fd">{adr_str}</span> <span style="color:#475569;">|</span> <span style="color:#a7f3d0">{rel_vol_str}</span></td>')
            p_days = r.get('Power Days', '0 / 0')
            html_lines.append(f'<td>{rs_html}</td>')
            
            ret_1m = r.get("1M Ret", 0.0)
            ret_3m = r.get("3M Ret", 0.0)
            ret_1m_str = f"<span style='color: {'#22c55e' if ret_1m > 0 else '#ef4444' if ret_1m < 0 else '#94a3b8'}; font-weight: bold;'>{ret_1m:+.1f}%</span>"
            ret_3m_str = f"<span style='color: {'#22c55e' if ret_3m > 0 else '#ef4444' if ret_3m < 0 else '#94a3b8'}; font-weight: bold;'>{ret_3m:+.1f}%</span>"
            html_lines.append(f'<td>{ret_1m_str}</td>')
            html_lines.append(f'<td>{ret_3m_str}</td>')

            smooth_rs_val = r.get("Smooth RS", 0.0)
            if smooth_rs_val >= 15.0:
                sm_badge = f"<span style='background:rgba(56, 189, 248, 0.15); color:#38bdf8; padding:2px 8px; border-radius:6px; font-weight:700; font-size:0.75rem; border:1px solid rgba(56,189,248,0.3);' title='Composite RS / 20D ATR% (≥15x Elite Smooth Leader)'>💎 {smooth_rs_val:.1f}x</span>"
            elif smooth_rs_val >= 10.0:
                sm_badge = f"<span style='background:rgba(34, 197, 94, 0.15); color:#22c55e; padding:2px 8px; border-radius:6px; font-weight:700; font-size:0.75rem; border:1px solid rgba(34,197,94,0.3);' title='Composite RS / 20D ATR% (10-15x Strong Institutional Trend)'>🟢 {smooth_rs_val:.1f}x</span>"
            elif smooth_rs_val >= 5.0:
                sm_badge = f"<span style='background:rgba(234, 179, 8, 0.15); color:#eab308; padding:2px 8px; border-radius:6px; font-weight:600; font-size:0.75rem; border:1px solid rgba(234,179,8,0.3);' title='Composite RS / 20D ATR% (5-10x Moderate Trend)'>🟡 {smooth_rs_val:.1f}x</span>"
            else:
                sm_badge = f"<span style='background:rgba(239, 68, 68, 0.15); color:#ef4444; padding:2px 8px; border-radius:6px; font-weight:600; font-size:0.75rem; border:1px solid rgba(239,68,68,0.3);' title='Composite RS / 20D ATR% (<5x High ATR Volatility / Chop)'>🔴 {smooth_rs_val:.1f}x</span>"
            html_lines.append(f'<td style="text-align: center;">{sm_badge}</td>')
            
            sortino_6m = r.get("Sortino 6M", 0.0)
            if sortino_6m >= 2.5:
                sort_color = "#38bdf8"
                sort_bg = "rgba(56, 189, 248, 0.15)"
                sort_badge = f"<span style='background:{sort_bg}; color:{sort_color}; padding:2px 8px; border-radius:6px; font-weight:700; font-size:0.75rem; border:1px solid rgba(56,189,248,0.3);'>🏆 {sortino_6m:.2f}</span>"
            elif sortino_6m >= 1.5:
                sort_color = "#22c55e"
                sort_bg = "rgba(34, 197, 94, 0.15)"
                sort_badge = f"<span style='background:{sort_bg}; color:{sort_color}; padding:2px 8px; border-radius:6px; font-weight:700; font-size:0.75rem; border:1px solid rgba(34,197,94,0.3);'>🟢 {sortino_6m:.2f}</span>"
            elif sortino_6m >= 0.8:
                sort_color = "#eab308"
                sort_bg = "rgba(234, 179, 8, 0.15)"
                sort_badge = f"<span style='background:{sort_bg}; color:{sort_color}; padding:2px 8px; border-radius:6px; font-weight:600; font-size:0.75rem; border:1px solid rgba(234,179,8,0.3);'>🟡 {sortino_6m:.2f}</span>"
            else:
                sort_color = "#ef4444"
                sort_bg = "rgba(239, 68, 68, 0.15)"
                sort_badge = f"<span style='background:{sort_bg}; color:{sort_color}; padding:2px 8px; border-radius:6px; font-weight:600; font-size:0.75rem; border:1px solid rgba(239,68,68,0.3);'>🔴 {sortino_6m:.2f}</span>"
            html_lines.append(f'<td style="text-align: center;">{sort_badge}</td>')
            
            html_lines.append(f'<td style="text-align: center;">{r.get("Inst. Health", "")}</td>')
            html_lines.append(f'<td><span style="font-size:0.8rem;">{r.get("Webster", "")}</span></td>')
            html_lines.append(f'<td>{add_icon}</td>')
            html_lines.append(f'<td>{act_html}</td>')
            html_lines.append(f'<td><span style="color:#94a3b8; font-size:0.8rem;">{r.get("Reason", "")}</span></td>')
            html_lines.append('</tr>')
            
        html_table += "".join(html_lines) + "</tbody></table></div></div>"
        st.markdown(html_table, unsafe_allow_html=True)
        
    with tab_defense_radar:
        render_institutional_defense_radar(decisions, bse_reverse_map)
    
    # =========================================================================
    # PORTFOLIO INTELLIGENCE (NEWS RADAR)
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📡 Portfolio Intelligence")
    st.markdown("Real-time institutional narrative and catalysts for your active holdings (Last 48 hours).")
    
    # Extract unique tickers from the portfolio that have a >0 weight
    active_tickers = [d['ticker'] for d in decisions if d.get('loss_metrics', {}).get('portfolio_weight', 0) > 0]
    
    if active_tickers:
        try:
            from news_fetcher import fetch_portfolio_news
            with st.spinner("Scanning global feeds for portfolio catalysts..."):
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
    gap: 20px;
    width: max-content;
    animation: marquee-scroll 75s linear infinite;
}
.news-marquee-wrapper:hover .news-marquee-track {
    animation-play-state: paused !important;
}
@keyframes marquee-scroll {
    0% { transform: translateX(0); }
    100% { transform: translateX(calc(-50% - 10px)); }
}
.news-card {
    width: 320px;
    flex-shrink: 0;
    background: linear-gradient(145deg, rgba(30,41,59,0.4) 0%, rgba(15,23,42,0.8) 100%);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 16px;
    padding: 20px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    text-decoration: none !important;
    display: flex;
    flex-direction: column;
    min-height: 150px;
    position: relative;
    overflow: hidden;
}
.news-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.8), transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
}
.news-card:hover {
    transform: translateY(-4px);
    border-color: rgba(56, 189, 248, 0.3);
    box-shadow: 0 12px 24px -10px rgba(56, 189, 248, 0.2);
    background: linear-gradient(145deg, rgba(30,41,59,0.6) 0%, rgba(15,23,42,0.95) 100%);
    text-decoration: none !important;
}
.news-card.news-card-fresh:hover {
    border-color: rgba(245, 158, 11, 0.6);
    box-shadow: 0 12px 24px -10px rgba(245, 158, 11, 0.3);
}
.news-card:hover::before {
    opacity: 1;
}
.news-card.news-card-fresh {
    border-color: rgba(245, 158, 11, 0.3);
    box-shadow: 0 0 15px rgba(245, 158, 11, 0.1);
}
.news-card.news-card-fresh::before {
    background: linear-gradient(90deg, transparent, rgba(245, 158, 11, 0.8), transparent);
    opacity: 0.8;
}
.fresh-badge {
    color: #f59e0b;
    font-weight: 800;
    font-size: 0.7rem;
    display: flex;
    align-items: center;
    gap: 4px;
    animation: pulse-fresh 2s infinite;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
@keyframes pulse-fresh {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
.news-badge {
    display: inline-flex;
    align-items: center;
    background: rgba(56, 189, 248, 0.1);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.7rem;
    font-weight: 800;
    margin-bottom: 14px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.news-title {
    color: #f1f5f9 !important;
    font-size: 1rem;
    font-weight: 600;
    line-height: 1.5;
    margin-bottom: 16px;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-decoration: none !important;
}
.news-card:hover .news-title {
    color: #ffffff !important;
}
.news-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 500;
    border-top: 1px solid rgba(255,255,255,0.05);
    padding-top: 14px;
    margin-top: auto;
}
.news-publisher {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #cbd5e1;
}
.news-time {
    color: #64748b;
    display: flex;
    align-items: center;
    gap: 4px;
}
</style>
<div class="news-marquee-wrapper">
"""
                news_duration = max(15, len(portfolio_news) * 4)
                news_html += f"""    <div class="news-marquee-track" style="animation: marquee-scroll {news_duration}s linear infinite;">
"""
                cards_html = ""
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
                st.info("No major catalysts or news stories detected for your holdings in the last 48 hours.")
        except Exception as e:
            st.error("📡 News radar temporarily offline.")
    else:
        st.info("Add holdings to your portfolio to activate the News Radar.")
    
    st.markdown("---")
    
    # =========================================================================
    # YOUTUBE MEDIA & INTERVIEWS
    # =========================================================================
    try:
        from media_fetcher import extract_portfolio_interviews
        
        portfolio_tickers_clean = [t for t in portfolio_df['ticker'].tolist() if t != 'CASH']
        
        with st.spinner("Fetching latest management interviews..."):
            interviews = extract_portfolio_interviews(portfolio_tickers_clean, [])
            
        if interviews:
            st.markdown("### 🎙️ Media & Interviews")
            st.markdown("<p style='color: #94a3b8; font-size: 0.9rem;'>Recent management appearances on leading financial networks</p>", unsafe_allow_html=True)
            
            # CSS for cards
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
    animation-play-state: paused !important;
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
            css_and_html += f"""    <div class="media-marquee-track" style="animation: marquee-media {media_duration}s linear infinite;">
"""
            
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
            # Duplicate the cards exactly once to create the seamless infinite loop
            css_and_html += cards_html + cards_html
            css_and_html += """
    </div>
</div>
"""
            st.markdown(css_and_html, unsafe_allow_html=True)
        else:
            st.markdown("### 🎙️ Media & Interviews")
            st.info("No recent management interviews or updates were found for your portfolio companies on the major networks in the past 30 days.")
            
        st.markdown("---")
    except Exception as e:
        st.error(f"Error loading media fetcher: {e}")

    # =====================================================================
    # CORPORATE FILINGS RADAR
    # =====================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    hdr_col, sync_col = st.columns([4.2, 1.2], vertical_alignment="center")
    with hdr_col:
        st.markdown("<h3 style='margin:0;'>📡 Corporate Filings Radar (Last 7 Days)</h3>", unsafe_allow_html=True)
    with sync_col:
        if st.button("🔄 Sync Filings", key="btn_sync_announcements_radar", help="Trigger an immediate scan of NSE/BSE corporate filings for portfolio stocks"):
            with st.spinner("Syncing corporate filings from NSE/BSE..."):
                try:
                    import announcements_engine
                    announcements_engine.sync_announcements()
                    st.toast("✅ Corporate Filings Synced!", icon="📡")
                    st.rerun()
                except Exception as e:
                    st.error(f"Sync error: {e}")
    
    # Quick Filters
    col1, col2 = st.columns([5, 3])
    with col1:
        filter_opt = st.pills(
            "Filter by Type:",
            ["All Filings", "Insider Trading & SAST", "Financial Results", "Board Meetings", "Others"],
            default="All Filings",
            label_visibility="collapsed"
        )
        if not filter_opt:
            filter_opt = "All Filings"
            
    with col2:
        time_opt = st.pills(
            "Filter by Time:",
            ["Last 24H", "Last 48H", "Last 72H", "Last 7 Days"],
            default="Last 7 Days",
            label_visibility="collapsed"
        )
        if not time_opt:
            time_opt = "Last 7 Days"
    st.markdown("<br>", unsafe_allow_html=True)
    
    try:
        from database import get_recent_announcements
        
        recent_announcements = get_recent_announcements(days=7)
        if not recent_announcements.empty:
            
            # Apply Time Filter
            if time_opt != "Last 7 Days":
                cutoff_hours = 24 if time_opt == "Last 24H" else 48 if time_opt == "Last 48H" else 72
                cutoff = datetime.now() - timedelta(hours=cutoff_hours)
                # Ensure date_time is datetime object for comparison
                if 'date_time' in recent_announcements.columns:
                    # Some sqlite reads might return strings, force to datetime
                    recent_announcements['dt_parsed'] = pd.to_datetime(recent_announcements['date_time'])
                    recent_announcements = recent_announcements[recent_announcements['dt_parsed'] >= cutoff]
            
            
            # Apply Filter
            if filter_opt == "Insider Trading & SAST":
                recent_announcements = recent_announcements[recent_announcements['title'].str.contains(r'\[Insider Buy\]|\[Insider Sell\]|\[SAST', case=False, na=False)]
            elif filter_opt == "Financial Results":
                recent_announcements = recent_announcements[recent_announcements['title'].str.contains(r'Financial Result|Earnings', case=False, na=False)]
            elif filter_opt == "Board Meetings":
                recent_announcements = recent_announcements[recent_announcements['title'].str.contains(r'Board Meeting', case=False, na=False)]
            elif filter_opt == "Others":
                recent_announcements = recent_announcements[~recent_announcements['title'].str.contains(r'\[Insider Buy\]|\[Insider Sell\]|\[SAST|Financial Result|Earnings|Board Meeting', case=False, na=False)]
                
            if recent_announcements.empty:
                st.info(f"No {filter_opt.lower()} found for your portfolio in the last 7 days.")
            
            # CSS for hover effects
            custom_css = """
            <style>
            .radar-card {
                background: linear-gradient(135deg, rgba(30,41,59,0.7), rgba(15,23,42,0.9));
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255,255,255,0.05);
                border-left: 5px solid #3b82f6;
                padding: 24px;
                margin-bottom: 20px;
                border-radius: 12px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.25);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            .radar-card:hover {
                transform: translateY(-4px) scale(1.01);
                box-shadow: 0 20px 40px rgba(59,130,246,0.2);
                border-left-color: #60a5fa;
                border-top: 1px solid rgba(255,255,255,0.1);
            }
            .radar-btn {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 18px;
                background: rgba(56,189,248,0.1);
                border: 1px solid rgba(56,189,248,0.3);
                border-radius: 8px;
                text-decoration: none !important;
                color: #38bdf8 !important;
                font-size: 1rem;
                font-weight: 700;
                letter-spacing: 0.5px;
                transition: all 0.3s ease;
            }
            .radar-btn:hover {
                background: rgba(56,189,248,0.25);
                border-color: #38bdf8;
                box-shadow: 0 0 15px rgba(56,189,248,0.3);
                transform: translateX(4px);
            }
            /* Custom Scrollbar for the container */
            .radar-container::-webkit-scrollbar { width: 6px; }
            .radar-container::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); }
            .radar-container::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 4px; }
            .radar-container::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.4); }
            
            .radar-card.insider-buy {
                border-left-color: #10b981;
                background: linear-gradient(135deg, rgba(16,185,129,0.05), rgba(15,23,42,0.95));
            }
            .radar-card.insider-buy:hover {
                box-shadow: 0 20px 40px rgba(16,185,129,0.25);
                border-left-color: #34d399;
            }
            .radar-card.insider-sell {
                border-left-color: #ef4444;
                background: linear-gradient(135deg, rgba(239,68,68,0.05), rgba(15,23,42,0.95));
            }
            .radar-card.insider-sell:hover {
                box-shadow: 0 20px 40px rgba(239,68,68,0.25);
                border-left-color: #f87171;
            }
            </style>
            """
            st.markdown(custom_css, unsafe_allow_html=True)
            
            ui_html = "<div class='radar-container' style='max-height: 500px; overflow-y: auto; padding-right: 12px; margin-bottom: 20px;'>"
            for _, row in recent_announcements.iterrows():
                safe_title = html.escape(str(row['title']))
                safe_desc = html.escape(str(row['description']))
                
                # Format date nicely
                try:
                    dt_obj = datetime.strptime(row['date_time'], '%Y-%m-%d %H:%M:%S')
                    display_date = dt_obj.strftime('%d %b %Y • %I:%M %p')
                except:
                    display_date = row['date_time']
                
                card_class = "radar-card"
                badge_color = "#60a5fa"
                badge_bg = "rgba(59,130,246,0.15)"
                
                if "[Insider Buy]" in safe_title or "[SAST" in safe_title:
                    card_class = "radar-card insider-buy"
                    badge_color = "#34d399"
                    badge_bg = "rgba(16,185,129,0.15)"
                elif "[Insider Sell]" in safe_title:
                    card_class = "radar-card insider-sell"
                    badge_color = "#f87171"
                    badge_bg = "rgba(239,68,68,0.15)"
                
                ui_html += f"""<div class="{card_class}">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
<span style="font-weight: 800; font-size: 1.25rem; color: {badge_color}; letter-spacing: 0.8px; background: {badge_bg}; padding: 4px 12px; border-radius: 6px; border: 1px solid {badge_color}40;">{row['ticker']}</span>
<span style="font-size: 1.05rem; color: #cbd5e1; font-weight: 600; display: flex; align-items: center; gap: 8px; opacity: 0.9;">🕒 {display_date}</span>
</div>
<div style="font-weight: 800; font-size: 1.3rem; color: #ffffff; margin-bottom: 10px; letter-spacing: 0.3px; line-height: 1.4;">{safe_title}</div>
<div style="font-size: 1.1rem; color: #cbd5e1; margin-bottom: 20px; line-height: 1.7; opacity: 0.85;">{safe_desc}</div>
<div style="text-align: right; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 16px; margin-top: 8px;">
<a href="{row['pdf_link']}" target="_blank" class="radar-btn">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
    View Original PDF Filing
</a>
</div>
</div>"""
            ui_html += "</div>"
            st.markdown(ui_html, unsafe_allow_html=True)
        else:
            st.info("No corporate announcements found for your portfolio in the last 7 days.")
    except Exception as e:
        st.error(f"Failed to load corporate announcements: {e}")

    
    # Portfolio Distribution Heatmap
    st.markdown("### 🗺️ Performance Heatmap (Sea of Green)")
    st.info("💡 **How to read:** Box **size** represents the stock's weight in your portfolio. Box **color** represents your P/L (Bright Green = Big Gains, Red = Losses).")
    
    heatmap_data = pd.DataFrame([{
        'Ticker': d['ticker'],
        'Weight': d.get('loss_metrics', {}).get('portfolio_weight', 0),
        'GainLossPct': d.get('loss_metrics', {}).get('gain_loss_pct', 0),
        'Action': d['action']
    } for d in decisions if d.get('loss_metrics', {}).get('portfolio_weight', 0) > 0])
    
    if not heatmap_data.empty:
        fig_treemap = px.treemap(
            heatmap_data,
            path=[px.Constant("Portfolio"), 'Ticker'],
            values='Weight',
            color='GainLossPct',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            custom_data=['GainLossPct', 'Weight', 'Action']
        )
        
        fig_treemap.update_traces(
            hovertemplate="<br>".join([
                "<b>%{label}</b>",
                "Weight: %{customdata[1]:.1f}%",
                "P/L: %{customdata[0]:.2f}%",
                "Action: %{customdata[2]}"
            ]),
            texttemplate="<b>%{label}</b><br>%{customdata[0]:.1f}%",
            textfont=dict(size=14)
        )
        
        fig_treemap.update_layout(margin=dict(t=30, l=10, r=10, b=10), height=500)
        st.plotly_chart(fig_treemap, use_container_width=True)
    else:
        st.info("📊 **Heatmap requires active positions.** Add shares (Qty > 0) to your connected Google Sheet to generate the Performance Heatmap.")
    
    # ---------------------------------------------------------
    # TELEGRAM PORTFOLIO DISPATCH CONTROL & DAY 0 SETUP CARD
    # ---------------------------------------------------------
    if is_telegram_configured():
        tg_c1, tg_c2 = st.columns([3.2, 1.2])
        with tg_c1:
            st.markdown(
                "<div style='font-size: 0.84rem; color: #94a3b8; display: flex; align-items: center; gap: 8px; margin-top: 8px;'>"
                "<span style='color: #10b981; font-weight: 700; background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); padding: 2px 8px; border-radius: 6px;'>🟢 Telegram Alert Active</span>"
                "<span>Automated RS, Sortino & Corporate Filings Radar daily briefing sends once per day on launch</span>"
                "</div>",
                unsafe_allow_html=True
            )
        with tg_c2:
            if st.button("📲 Send Telegram Briefing Now", key="btn_send_telegram_now", help="Push an immediate live portfolio health update to your Telegram chat"):
                with st.spinner("Dispatching Telegram Briefing..."):
                    h_sum = st.session_state.get('portfolio_health_summary') or calculate_portfolio_health_summary(
                        decisions=decisions,
                        total_value=total_value,
                        market_regime_label=current_regime_label
                    )
                    b_html = format_telegram_portfolio_briefing(h_sum)
                    ok, resp = send_telegram_portfolio_message(b_html)
                    if ok:
                        mark_sent_today()
                        st.toast("🚀 Telegram Briefing Dispatched!", icon="✅")
                        st.success("✅ Telegram Portfolio Briefing sent successfully!")
                    else:
                        st.error(f"❌ Failed to send Telegram briefing: {resp}")
    else:
        with st.expander("📱 Telegram Daily Portfolio Alerts (Day 0 Setup Guide)", expanded=False):
            st.markdown("""
            **Receive daily RS, Sortino (3M / 6M), and position health briefings directly on your Telegram!**
            
            **Day 0 Setup Instructions (Takes 2 minutes):**
            1. Open Telegram, search for `@BotFather`, send `/newbot`, and copy your `TELEGRAM_BOT_TOKEN`.
            2. Send a message to your new bot (or add it to your group as Admin), then find your numeric `TELEGRAM_CHAT_ID` (you can message `@userinfobot` to find your personal ID).
            3. Open your local `.env` file (copy from `.env.example` if needed) and add:
            ```env
            TELEGRAM_BOT_TOKEN=your_bot_token_here
            TELEGRAM_CHAT_ID=your_numeric_chat_id_here
            ENABLE_TELEGRAM_DAILY_ALERT=true
            ```
            4. Restart Kush Tracker. The system runs 100% locally on your machine—no portfolio details ever touch external cloud servers.
            """)
    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # RS Score Distribution
    # Portfolio Roppel Quadrant
    st.markdown("### 🔭 The Portfolio Roppel Quadrant")
    
    tab_tech, tab_funda, tab_momentum, tab_sortino = st.tabs(["Technical (RS vs P/L)", "Fundamental (RS vs Sales Growth)", "Momentum (1M vs 3M)", "Risk Quality (RS vs Sortino 3M)"])
    
    with tab_tech:
        st.caption("Apex Predators live in the top-right corner (High RS + High Profit). Size of the bubble represents Portfolio Weight.")
        
        roppel_data = pd.DataFrame([{
            'Ticker': d['ticker'],
            'RS Score': d.get('rs_score', 0) or 0,
            'GainLossPct': d.get('loss_metrics', {}).get('gain_loss_pct', 0),
            'Weight': d.get('loss_metrics', {}).get('portfolio_weight', 0)
        } for d in decisions])
        
        if not roppel_data.empty:
            # Handle NaN values before Plotly tries to render them
            roppel_data['Weight'] = roppel_data['Weight'].fillna(0).apply(lambda x: max(x, 2))
            roppel_data['GainLossPct'] = roppel_data['GainLossPct'].fillna(0)
            roppel_data['RS Score'] = roppel_data['RS Score'].fillna(0)
            
            fig_roppel = px.scatter(
                roppel_data,
                x='RS Score',
                y='GainLossPct',
                size='Weight',
                color='RS Score',
                hover_name='Ticker',
                text='Ticker',
                custom_data=['Weight', 'GainLossPct', 'RS Score'],
                color_continuous_scale='RdYlGn',
                range_color=[0, 100],
                labels={
                    'RS Score': 'Relative Strength (0-99)',
                    'GainLossPct': 'Unrealized P/L (%)'
                }
            )
            fig_roppel.update_traces(
                textposition='top center', 
                marker=dict(line=dict(width=1, color='DarkSlateGrey')),
                hovertemplate="<b>%{hovertext}</b><br>RS Score: %{customdata[2]}<br>P/L: %{customdata[1]:.2f}%<br>Weight: %{customdata[0]:.1f}%"
            )
            
            # Quadrant Lines
            fig_roppel.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
            fig_roppel.add_vline(x=80, line_dash="dash", line_color="rgba(255,255,255,0.4)")
            
            # Elite Zone Shading
            max_y = roppel_data['GainLossPct'].max() if roppel_data['GainLossPct'].max() > 0 else 10
            min_y = roppel_data['GainLossPct'].min()
            
            fig_roppel.add_shape(
                type="rect",
                x0=80, y0=0, x1=100, y1=max_y * 1.2,
                line=dict(color="rgba(0,0,0,0)"),
                fillcolor="rgba(50, 205, 50, 0.1)",
                layer="below"
            )
            fig_roppel.add_annotation(
                x=90, y=max_y * 1.1,
                text="👑 Apex Predators",
                showarrow=False,
                font=dict(color="#32cd32", size=14, weight="bold")
            )
            
            fig_roppel.update_layout(height=500, template='plotly_dark')
            st.plotly_chart(fig_roppel, use_container_width=True)

    with tab_funda:
        st.caption("Apex Predators live in the top-right corner (High RS + High Sales Growth). Size of the bubble represents ADTV.")
        
        fundas_cache = get_all_fundamentals_cache()
        funda_data_list = []
        
        for d in decisions:
            ticker = d['ticker']
            rs = d.get('rs_score', 0) or 0
            
            # Try to get fundamentals from cache
            clean_ticker = ticker.replace('.NS', '')
            ns_ticker = f"{clean_ticker}.NS"
            
            sales_growth = 0.0
            if ns_ticker in fundas_cache:
                sales_growth = fundas_cache[ns_ticker].get('sales_growth', 0.0)
            elif ticker in fundas_cache:
                sales_growth = fundas_cache[ticker].get('sales_growth', 0.0)
            elif clean_ticker in fundas_cache:
                sales_growth = fundas_cache[clean_ticker].get('sales_growth', 0.0)
                
            # Calculate ADTV
            adtv_cr = d.get('adtv_cr', 1.0)
            if pd.isna(adtv_cr) or adtv_cr is None:
                adtv_cr = 1.0
                    
            funda_data_list.append({
                'Ticker': ticker,
                'RS Score': rs,
                'SalesGrowth': sales_growth,
                'ADTV': max(adtv_cr, 1.0)
            })
                
        funda_data = pd.DataFrame(funda_data_list)
        
        if not funda_data.empty:
            funda_data['ADTV'] = funda_data['ADTV'].fillna(1.0)
            funda_data['RS Score'] = funda_data['RS Score'].fillna(0)
            funda_data['SalesGrowth'] = funda_data['SalesGrowth'].fillna(0)
            
            fig_funda = px.scatter(
                funda_data,
                x='RS Score',
                y='SalesGrowth',
                size='ADTV',
                color='RS Score',
                hover_name='Ticker',
                text='Ticker',
                custom_data=['ADTV', 'SalesGrowth', 'RS Score'],
                color_continuous_scale='RdYlGn',
                range_color=[0, 100],
                labels={
                    'RS Score': 'Relative Strength (0-99)',
                    'SalesGrowth': 'Sales Growth YoY (%)',
                    'ADTV': 'ADTV (Cr)'
                }
            )
            fig_funda.update_traces(
                textposition='top center', 
                marker=dict(line=dict(width=1, color='DarkSlateGrey')),
                hovertemplate="<b>%{hovertext}</b><br>RS Score: %{customdata[2]}<br>Sales Growth: %{customdata[1]:.1f}%<br>ADTV: ₹%{customdata[0]:.1f} Cr"
            )
            
            # Quadrant Lines
            fig_funda.add_hline(y=40, line_dash="dash", line_color="rgba(255,255,255,0.4)")
            fig_funda.add_vline(x=80, line_dash="dash", line_color="rgba(255,255,255,0.4)")
            
            # Elite Zone Shading
            max_sf_y = funda_data['SalesGrowth'].max() if funda_data['SalesGrowth'].max() > 40 else 60
            
            fig_funda.add_shape(
                type="rect",
                x0=80, y0=40, x1=100, y1=max_sf_y * 1.2,
                line=dict(color="rgba(0,0,0,0)"),
                fillcolor="rgba(50, 205, 50, 0.1)",
                layer="below"
            )
            fig_funda.add_annotation(
                x=90, y=max_sf_y * 1.1,
                text="👑 Apex Predators",
                showarrow=False,
                font=dict(color="#32cd32", size=14, weight="bold")
            )
            
            fig_funda.update_layout(height=500, template='plotly_dark')
            st.plotly_chart(fig_funda, use_container_width=True)
            
    with tab_momentum:
        st.caption("Momentum Leaders live in the top-right corner (Positive 3M Return + Positive 1M Return). Size of the bubble represents ADTV.")
        
        momentum_data_list = []
        for d in decisions:
            adtv_cr = d.get('adtv_cr', 1.0)
            if pd.isna(adtv_cr) or adtv_cr is None:
                adtv_cr = 1.0
                
            momentum_data_list.append({
                'Ticker': d['ticker'],
                '1M Ret': d.get('ret_1m', 0.0),
                '3M Ret': d.get('ret_3m', 0.0),
                'ADTV': max(adtv_cr, 1.0)
            })
            
        mom_data = pd.DataFrame(momentum_data_list)
        
        if not mom_data.empty:
            mom_data['ADTV'] = mom_data['ADTV'].fillna(1.0)
            mom_data['1M Ret'] = mom_data['1M Ret'].fillna(0)
            mom_data['3M Ret'] = mom_data['3M Ret'].fillna(0)
            
            fig_mom = px.scatter(
                mom_data,
                x='3M Ret',
                y='1M Ret',
                size='ADTV',
                color='1M Ret',
                hover_name='Ticker',
                text='Ticker',
                custom_data=['ADTV', '1M Ret', '3M Ret'],
                color_continuous_scale='RdYlGn',
                labels={
                    '3M Ret': '3-Month Return (%)',
                    '1M Ret': '1-Month Return (%)',
                    'ADTV': 'ADTV (Cr)'
                }
            )
            fig_mom.update_traces(
                textposition='top center', 
                marker=dict(line=dict(width=1, color='DarkSlateGrey')),
                hovertemplate="<b>%{hovertext}</b><br>1M Ret: %{customdata[1]:.2f}%<br>3M Ret: %{customdata[2]:.2f}%<br>ADTV: ₹%{customdata[0]:.1f} Cr"
            )
            
            # Quadrant Lines
            fig_mom.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
            fig_mom.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
            
            # Elite Zone Shading (Top Right)
            max_x = mom_data['3M Ret'].max() if mom_data['3M Ret'].max() > 0 else 10
            max_y = mom_data['1M Ret'].max() if mom_data['1M Ret'].max() > 0 else 10
            
            fig_mom.add_shape(
                type="rect",
                x0=0, y0=0, x1=max_x * 1.2, y1=max_y * 1.2,
                line=dict(color="rgba(0,0,0,0)"),
                fillcolor="rgba(50, 205, 50, 0.1)",
                layer="below"
            )
            fig_mom.add_annotation(
                x=max_x * 1.1, y=max_y * 1.1,
                text="👑 Leaders",
                showarrow=False,
                font=dict(color="#32cd32", size=14, weight="bold")
            )
            
            fig_mom.update_layout(height=500, template='plotly_dark')
            st.plotly_chart(fig_mom, use_container_width=True)

    with tab_sortino:
        st.caption("Apex Compounders live in the top-right quadrant (High RS Score + High Smooth RS / Sortino). Size of the bubble represents Portfolio Weight (%).")
        
        ps_c1, ps_c2 = st.columns([2.4, 1.6])
        with ps_c1:
            port_sortino_horizon = st.radio(
                "Risk-Quality Metric & Horizon:",
                [
                    "Smooth RS (Composite RS / 20D ATR%)",
                    "Sortino 3M (Tactical Acceleration - 63D)",
                    "Sortino 6M (Structural Compounder - 126D)"
                ],
                horizontal=True,
                key="portfolio_roppel_sortino_horizon"
            )
        if "Smooth RS" in port_sortino_horizon:
            sort_key = "smooth_rs"
            sort_label = "Smooth RS (RS / 20D ATR%)"
            y_thresh_strong = 10.0
            y_thresh_weak = 5.0
            y_line_label = "Strong Smooth RS (≥10.0x)"
        elif "3M" in port_sortino_horizon:
            sort_key = "sortino_3m"
            sort_label = "Sortino 3M"
            y_thresh_strong = 3.0
            y_thresh_weak = 2.0
            y_line_label = "Strong Sortino (≥3.0)"
        else:
            sort_key = "sortino_6m"
            sort_label = "Sortino 6M"
            y_thresh_strong = 3.0
            y_thresh_weak = 2.0
            y_line_label = "Strong Sortino (≥3.0)"
        
        with ps_c2:
            st.markdown("<div style='padding-top: 5px; color: #94a3b8; font-size: 0.85rem;'>Cross-references your active holdings' <b>Relative Strength</b> against the NIFTY 500 with <b>Smooth RS (RS / 20D ATR%)</b> or <b>Downside Sortino Quality</b>.</div>", unsafe_allow_html=True)
            
        port_sort_list = []
        for d in decisions:
            weight = d.get('loss_metrics', {}).get('portfolio_weight', 0)
            if weight <= 0:
                continue
            tckr = str(d['ticker']).replace('.NS', '').replace('.BO', '')
            rs_val = float(d.get('rs_score', 0) or 0)
            s_val = float(d.get(sort_key, 0.0) or 0.0)
            pnl = float(d.get('loss_metrics', {}).get('gain_loss_pct', 0) or 0.0)
            act = str(d.get('action', 'HOLD'))
            px_val = float(d.get('close_price', 0) or 0.0)
            s3m = float(d.get('sortino_3m', 0.0) or 0.0)
            s6m = float(d.get('sortino_6m', 0.0) or 0.0)
            sm_rs_val = float(d.get('smooth_rs', 0.0) or 0.0)
            
            # Classify Quadrant
            if rs_val >= 80 and s_val >= y_thresh_strong:
                quad = "👑 Apex Compounder"
            elif rs_val >= 80 and s_val < y_thresh_weak:
                quad = "🎢 Wild Bronco"
            elif rs_val < 70 and s_val >= y_thresh_strong:
                quad = "🐢 Sleepy Defender"
            else:
                quad = "⚪ Normal / Core"
                
            port_sort_list.append({
                'Ticker': tckr,
                'RS Score': rs_val,
                sort_label: s_val,
                'Weight': weight,
                'BubbleWeight': max(weight, 2.5),
                'PnL': pnl,
                'Price': px_val,
                'Action': act,
                'Smooth RS': sm_rs_val,
                'Sortino 3M': s3m,
                'Sortino 6M': s6m,
                'Quadrant': quad
            })
            
        df_ps = pd.DataFrame(port_sort_list)
        if not df_ps.empty:
            fig_ps = px.scatter(
                df_ps,
                x='RS Score',
                y=sort_label,
                size='BubbleWeight',
                color=sort_label,
                hover_name='Ticker',
                text='Ticker',
                custom_data=['Weight', 'PnL', 'RS Score', 'Sortino 3M', 'Sortino 6M', 'Price', 'Action', 'Quadrant', 'Smooth RS'],
                color_continuous_scale='Viridis',
                labels={
                    'RS Score': 'Mansfield Relative Strength vs NIFTY 500 (0-100)',
                    sort_label: f'{sort_label} (Volatility-Efficiency Quality)'
                }
            )
            
            fig_ps.update_traces(
                textposition='top center',
                textfont=dict(size=11, color='rgba(255, 255, 255, 0.85)'),
                marker=dict(line=dict(width=1.5, color='rgba(255, 255, 255, 0.4)')),
                hovertemplate="<b style='font-size:15px'>%{hovertext}</b> (%{customdata[7]})<br><br><b>RS Score:</b> %{customdata[2]}<br><b>Smooth RS (RS/ATR):</b> %{customdata[8]:.2f}x<br><b>Sortino 3M / 6M:</b> %{customdata[3]:.2f} / %{customdata[4]:.2f}<br><b>Portfolio Weight:</b> %{customdata[0]:.1f}%<br><b>Unrealized P/L:</b> %{customdata[1]:+.1f}%<br><b>Price:</b> ₹%{customdata[5]:,.2f}<br><b>Action:</b> %{customdata[6]}<extra></extra>"
            )
            
            # Quadrant Lines
            fig_ps.add_vline(x=80, line_dash="dash", line_color="rgba(56, 189, 248, 0.5)", line_width=2, annotation_text="Elite RS Zone (≥80)", annotation_position="top left", annotation_font=dict(color="rgba(56, 189, 248, 0.8)", size=11))
            fig_ps.add_hline(y=y_thresh_strong, line_dash="dash", line_color="rgba(16, 185, 129, 0.5)", line_width=2, annotation_text=y_line_label, annotation_position="bottom right", annotation_font=dict(color="rgba(16, 185, 129, 0.8)", size=11))
            
            # Shaded Apex Zone Box
            max_y_ps = max(y_thresh_strong * 1.6, df_ps[sort_label].max() * 1.15)
            min_y_ps = min(-1.0, df_ps[sort_label].min())
            
            fig_ps.add_shape(
                type="rect",
                x0=80, y0=y_thresh_strong, x1=101, y1=max_y_ps,
                line=dict(color="rgba(0,0,0,0)"),
                fillcolor="rgba(16, 185, 129, 0.12)",
                layer="below"
            )
            fig_ps.add_annotation(
                x=90, y=max_y_ps * 0.95,
                text="👑 APEX COMPOUNDERS",
                showarrow=False,
                font=dict(color="#10b981", size=14, family="Inter")
            )
            fig_ps.add_annotation(
                x=90, y=y_thresh_weak * 0.6,
                text="🎢 WILD BRONCOS",
                showarrow=False,
                font=dict(color="#f59e0b", size=12, family="Inter")
            )
            fig_ps.add_annotation(
                x=40, y=max_y_ps * 0.95,
                text="🐢 SLEEPY DEFENDERS",
                showarrow=False,
                font=dict(color="#94a3b8", size=12, family="Inter")
            )
            
            fig_ps.update_layout(
                height=560,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.15)',
                font=dict(color='#cbd5e1'),
                margin=dict(l=20, r=20, t=30, b=20),
                hoverlabel=dict(bgcolor="#0f172a", font_size=13, bordercolor="rgba(255,255,255,0.15)")
            )
            fig_ps.update_xaxes(range=[0, 103], showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.06)', zeroline=False)
            fig_ps.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.06)', zeroline=False)
            
            st.plotly_chart(fig_ps, use_container_width=True)
            
            # Capital Allocation by Quadrant
            apex_weight = df_ps[df_ps['Quadrant'] == '👑 Apex Compounder']['Weight'].sum()
            bronco_weight = df_ps[df_ps['Quadrant'] == '🎢 Wild Bronco']['Weight'].sum()
            other_weight = df_ps[~df_ps['Quadrant'].isin(['👑 Apex Compounder', '🎢 Wild Bronco'])]['Weight'].sum()
            
            q_col1, q_col2, q_col3 = st.columns(3)
            with q_col1:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 12px; text-align: center;">
                    <p style="color: #94a3b8; font-size: 0.75rem; margin: 0; text-transform: uppercase; font-weight: 700;">👑 Apex Capital Exposure</p>
                    <h3 style="color: #10b981; margin: 4px 0; font-family: 'JetBrains Mono', monospace;">{apex_weight:.1f}%</h3>
                    <p style="color: #cbd5e1; font-size: 0.75rem; margin: 0;">High RS + Disciplined Downside</p>
                </div>
                """, unsafe_allow_html=True)
            with q_col2:
                st.markdown(f"""
                <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 10px; padding: 12px; text-align: center;">
                    <p style="color: #94a3b8; font-size: 0.75rem; margin: 0; text-transform: uppercase; font-weight: 700;">🎢 Wild Bronco Exposure</p>
                    <h3 style="color: #f59e0b; margin: 4px 0; font-family: 'JetBrains Mono', monospace;">{bronco_weight:.1f}%</h3>
                    <p style="color: #cbd5e1; font-size: 0.75rem; margin: 0;">High RS + Volatile Downside</p>
                </div>
                """, unsafe_allow_html=True)
            with q_col3:
                st.markdown(f"""
                <div style="background: rgba(148, 163, 184, 0.1); border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 10px; padding: 12px; text-align: center;">
                    <p style="color: #94a3b8; font-size: 0.75rem; margin: 0; text-transform: uppercase; font-weight: 700;">🛡️ Core / Other Exposure</p>
                    <h3 style="color: #cbd5e1; margin: 4px 0; font-family: 'JetBrains Mono', monospace;">{other_weight:.1f}%</h3>
                    <p style="color: #cbd5e1; font-size: 0.75rem; margin: 0;">Normal / Consolidating Holds</p>
                </div>
                """, unsafe_allow_html=True)
                
            # TradingView Watchlist Export for Apex Holdings
            apex_holdings = df_ps[df_ps['Quadrant'] == '👑 Apex Compounder']['Ticker'].tolist()
            if apex_holdings:
                with st.expander(f"📺 Export {len(apex_holdings)} Apex Portfolio Holdings to TradingView"):
                    st.code(",".join([f"NSE:{t}" for t in apex_holdings]), language="text")
        else:
            st.info("No active holdings with weight > 0 found.")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; padding: 1rem;">
        <p>Kush Tracker v1.2 | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p style="font-size: 0.85rem;">Inspired by Qullamaggie, Mark Minervini, William O'Neil, Paul Tudor Jones</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
