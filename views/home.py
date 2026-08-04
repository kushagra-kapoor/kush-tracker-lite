# Kush Tracker - Streamlit Dashboard
# Professional Portfolio Execution & Risk Management App

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import math

# Local imports
from portfolio_fetcher import fetch_portfolio_data
from market_data import fetch_all_holdings_data, fetch_nifty_total_market_tickers, get_ticker_symbol
from technical_indicators import add_technical_indicators, get_atr_state, calculate_webster_sell_signals
from relative_strength import calculate_all_rs_scores, get_rs_rating
from signal_engine import calculate_weighted_rs_for_universe
from price_history_manager import fetch_incremental_history
from trend_analyzer import determine_trend_state, get_trend_details
from decision_engine import make_decision, ACTION_EXIT, ACTION_TRIM, ACTION_HOLD, ACTION_ADD
from database import init_database, save_snapshot, get_signal_change_days, save_sector_leadership, get_sector_leadership_history, get_journal_entry, get_all_fundamentals_cache, get_current_tml_leaders, get_hat_stocks, get_top_5_rs_leaders
from climax_exhaustion import detect_climax_exhaustion
from macro_regime_engine import calculate_distribution_days, get_market_regime_label, detect_change_of_character
from daily_insights_engine import generate_exposure_guide, get_sector_clusters, generate_macro_health_score
from views.market_regime import process_macro_data, fetch_universe_with_industry, fetch_yfinance_batch
from safe_harbor_engine import evaluate_safe_harbor
from portfolio_earnings_engine import update_portfolio_earnings, get_portfolio_new_results_alerts, get_pre_earnings_risk_tickers


# Page config
# st.set_page_config removed for Lite routing

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
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backfill_tml_history.py")
            subprocess.run([sys.executable, script_path], check=True)
            print("[System] Background TML backfill completed or verified.")
        except Exception as e:
            print(f"[System] Error in background backfill: {e}")
            
    thread = threading.Thread(target=run_backfill, daemon=True)
    thread.start()
    return True

# Trigger the background backfill on app load
start_background_backfill()
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
        universe_tickers = fetch_nifty_total_market_tickers(show_progress=False)
        if not universe_tickers or len(universe_tickers) < 100:
            return {}
        
        # Ensure portfolio tickers are formatted correctly (.NS if needed)
        portfolio_yf_tickers = [get_ticker_symbol(t) for t in portfolio_tickers if t != 'BENCHMARK']
        
        # Combine universe with portfolio (removes duplicates)
        all_target_tickers = list(set(universe_tickers + portfolio_yf_tickers))
        
        # Use the incremental disk cache — avoids re-downloading 750 stocks
        history_df = fetch_incremental_history(all_target_tickers, days=252)
        if history_df.empty:
            return {}
        
        # Same RS calculation used by the Algo Terminal
        rs_scores = calculate_weighted_rs_for_universe(history_df, all_target_tickers)
        return rs_scores  # {clean_ticker: percentile_0_to_100}
    except Exception as e:
        print(f"[!] Universe RS computation failed: {e}")
        return {}


def _build_rs_data_from_universe(portfolio_tickers, universe_rs, market_data):
    """
    Convert universe-level RS percentile scores into the full rs_data format
    expected by the decision engine and UI.
    
    Args:
        portfolio_tickers: list of portfolio ticker symbols
        universe_rs: dict from compute_universe_rs_scores() {clean_ticker: percentile}
        market_data: dict of portfolio DataFrames (for stock return details)
    
    Returns:
        dict in the same format as calculate_all_rs_scores():
        {ticker: {rs_score, rs_raw, rs_rating, rs_source, details}}
    """
    rs_data = {}
    
    for ticker in portfolio_tickers:
        if ticker == 'BENCHMARK':
            continue
        
        # Look up the universe RS for this ticker
        rs_score = universe_rs.get(ticker)
        
        # Fallback to yf_ticker (for BSE stocks like ZELIO.BO)
        if rs_score is None:
            yf_ticker = get_ticker_symbol(ticker)
            rs_score = universe_rs.get(yf_ticker)
            
            # Additional fallback in case signal_engine stripped .NS
            if rs_score is None:
                stripped = yf_ticker.replace('.NS', '')
                rs_score = universe_rs.get(stripped)
        
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
def load_data():
    """Load and process all data."""
    # Initialize database
    init_database()
    
    # Fetch portfolio
    portfolio_df = fetch_portfolio_data()
    if portfolio_df.empty:
        pass # Continue loading macro data for Lite version
    
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
    
    # Nightly AI Briefing
    today_str = datetime.now().strftime("%Y-%m-%d")
    journal_entry = get_journal_entry(today_str)
    
    if journal_entry and journal_entry.get('journal_notes'):
        st.markdown("### 🤖 Nightly AI Battle Plan")
        st.info(journal_entry.get('journal_notes'))
    else:
        st.warning("🚨 **Daily Journal Pending:** No AI Briefing or Journal generated for today yet.")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📅 Session Info")
        st.markdown(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
        st.markdown(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Load data with spinner
    with st.spinner("Loading portfolio data..."):
        portfolio_df, market_data, rs_data, total_value, industry_map, earnings_cache = load_data()
    


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
        
    @st.cache_data(ttl=86400)
    def fetch_market_navigator_data(cache_buster=1):
        universe = fetch_universe_with_industry()
        if universe is None or universe.empty or 'Symbol' not in universe.columns:
            st.error("Failed to fetch universe data from NSE. Please try again later.")
            return None
        tks = universe['Symbol'].tolist()
        idx_n50 = yf.download('^NSEI', period="100d", progress=False)
        idx_n500 = yf.download('^CRSLDX', period="100d", progress=False)
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
    
    if macro_res:
        st.markdown("## 🦅 Daily Insights Cockpit")
        
        c_col1, c_col2, c_col3 = st.columns(3)
        
        # Panel 1: Macro Health & Exposure Guide
        with c_col1:
            ler = macro_res.get('leadership', {}).get('ler_current', 0)
            lac = macro_res.get('leadership', {}).get('lac_current', 0)
            lt = macro_res.get('leadership', {}).get('lt_current', 0)
            bt = macro_res.get('breadth', {}).get('breadth_ma10_current', 0.5)
            
            health = generate_macro_health_score(ler, lac, lt, bt)
            exposure_guide = generate_exposure_guide(regime, ler)
            
            st.markdown(f"""
<div class="metric-card" style="border-top: 4px solid {health['color']}; height: 100%;">
<p style="color: #94a3b8; font-size: 0.85rem; margin: 0; text-transform: uppercase; font-weight: bold;">Macro Health Index</p>
<div style="display: flex; align-items: baseline; justify-content: center; gap: 10px; margin: 0.5rem 0;">
<h1 style="color: {health['color']}; margin: 0;">{health['score']:.0f}</h1>
<span style="color: #94a3b8; font-size: 1.2rem;">/ 100</span>
</div>
<h4 style="color: {health['color']}; margin: 0 0 1rem 0;">{health['label']}</h4>
<div style="background: rgba(0,0,0,0.2); padding: 0.75rem; border-radius: 8px; text-align: left;">
<p style="margin: 0; font-size: 0.9rem; color: #cbd5e1;">Target Exposure: <strong style="color: {exposure_guide['color']};">{exposure_guide['level']} ({exposure_guide['stance']})</strong></p>
<p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #94a3b8;">💡 <i>{exposure_guide['advice']}</i></p>
</div>
</div>
""", unsafe_allow_html=True)
            
        # Panel 2: Regime & Change of Character
        with c_col2:
            choch_html = ""
            if choch and choch['detected']:
                choch_html = f"""
<div style="background: rgba({int(choch['color'].lstrip('#')[0:2], 16)}, {int(choch['color'].lstrip('#')[2:4], 16)}, {int(choch['color'].lstrip('#')[4:6], 16)}, 0.1); border-left: 3px solid {choch['color']}; padding: 0.5rem; border-radius: 4px; margin-top: 0.75rem; text-align: left;">
<p style="margin: 0; font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Wyckoff Alert ({choch['date']})</p>
<p style="margin: 0; color: {choch['color']}; font-weight: bold; font-size: 0.9rem;">{choch['label']}</p>
</div>
"""
                
            st.markdown(f"""
<div class="metric-card" style="border-top: 4px solid {regime_color}; height: 100%;">
<p style="color: #94a3b8; font-size: 0.85rem; margin: 0; text-transform: uppercase; font-weight: bold;">Price Regime Status</p>
<h3 style="color: {regime_color}; margin: 0.5rem 0;">{regime}</h3>
<div style="display: flex; justify-content: space-around; font-size: 0.85rem; color: #cbd5e1; margin-top: 1rem;">
<div><span style="color: #94a3b8;">Dist Days:</span> <strong>{dd_count}</strong></div>
<div><span style="color: #94a3b8;">vs 50SMA:</span> <strong>{'+' if current_idx_price > sma50 else '-'}{abs((current_idx_price/sma50-1)*100):.1f}%</strong></div>
</div>
{choch_html}
</div>
""", unsafe_allow_html=True)

            with st.expander("Explore Deep Indicators"):
                st.metric("Leader Emergence Rate", f"{ler:.1f}%")
                st.metric("Leadership Acceleration", f"{lac:.2f}")
                st.metric("Leadership Thrust", f"{lt:.1f}")
                st.metric("Breadth Over 10 SMA", f"{bt*100:.1f}%")
                
            # Safe Harbor Suggestion
            if 'Correction' in regime or 'Pressure' in regime:
                st.markdown("<br/>", unsafe_allow_html=True)
                safe_assets = evaluate_safe_harbor()
                if safe_assets:
                    st.markdown("""
                    <div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 8px;">
                        <h4 style="color: #f59e0b; margin-top: 0; margin-bottom: 0.5rem;">🛡️ Safe Harbor Rotation</h4>
                        <p style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 0.75rem;">Protective asset allocation for current regime.</p>
                    """, unsafe_allow_html=True)
                    
                    for asset in safe_assets[:2]: # Show top 2 alternatives
                        status = asset['Status']
                        color = "#10b981" if "Strong" in status or "Excellent" in status else "#f59e0b" if "Hold" in status else "#ef4444"
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; background: rgba(0,0,0,0.2); padding: 0.5rem; border-radius: 4px;">
                            <span style="font-size: 0.85rem; font-weight: bold;">{asset['Ticker']}</span>
                            <span style="font-size: 0.8rem; color: {color};">{status}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
        # Panel 3: Institutional Clusters & Scraper Actions
        with c_col3:
            clusters = get_sector_clusters(macro_res.get('sector_df'), top_n=3)
            cluster_html = ""
            for i, c in enumerate(clusters):
                cluster_html += f"""
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2d3748; padding: 0.4rem 0;">
<span style="color: #cbd5e1; font-weight: bold; font-size: 0.9rem;">{i+1}. {c['name']}</span>
<span style="color: #10b981; font-size: 0.85rem;">{c['bullish_pct']:.0f}% Bullish</span>
</div>
"""
            
            st.markdown(f"""
<div class="metric-card" style="border-top: 4px solid #3b82f6; height: 100%;">
<p style="color: #94a3b8; font-size: 0.85rem; margin: 0; text-transform: uppercase; font-weight: bold;">Leading Themes (O'Neil)</p>
<div style="text-align: left; margin-top: 0.5rem;">
{cluster_html if cluster_html else "<p style='color: #64748b; font-size: 0.9rem;'>Gathering data...</p>"}
</div>
</div>
""", unsafe_allow_html=True)
            st.markdown("### 🐺 StockScans Engine")
            import os
            app_dir = os.path.dirname(os.path.abspath(__file__))
            scraper_path = os.path.join(app_dir, "stockscans_scraper.py")
            
            if st.button("⚙️ Setup Login (One-Time)", use_container_width=True):
                if sys.platform == "win32":
                    subprocess.Popen(f'cmd.exe /k python "{scraper_path}" --setup', cwd=app_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:
                    subprocess.Popen([sys.executable, scraper_path, "--setup"], cwd=app_dir)
                st.toast("Opening visible browser for setup...", icon="⚙️")
            if st.button("🚀 Run Confluence Engine", use_container_width=True, type="primary"):
                # Use cmd.exe /c to inherit the environment variables (like PATH) properly on Windows
                if sys.platform == "win32":
                    subprocess.Popen(f'cmd.exe /c python "{scraper_path}"', cwd=app_dir, creationflags=0x08000000)
                else:
                    subprocess.Popen([sys.executable, scraper_path], cwd=app_dir)
                st.toast("Scraper launched! Wait 30 seconds for the popup.", icon="🚀")
            
        with st.expander("📚 How to read the Daily Insights Cockpit"):
            st.markdown("""
            **What is this?**
            This professional insights cockpit aggregates the health of the entire total market to help you determine your daily "Risk OFF" vs "Risk ON" setting before you look at individual stocks. It uses principles from legendary traders like William O'Neil, Mark Minervini, and Richard Wyckoff.
            
            1. **Macro Health Index (0-100)**: Consolidates the strength of market breadth and emerging momentum leaders into a single score. 
               * Uses Leader Emergence Rate (LER), Leadership Acceleration (LAC), Breadth Thrusts (BT), and Leadership Thrusts (LT).
               * **Score > 60**: Healthy Bull. Start or continue progressive exposure.
               * **Score < 40**: Defensive. Cash is a position. Stop initiating new buys.
               
            2. **Target Exposure**: Gives you a concrete capital allocation limit based on the alignment of the price trend and the internal leadership metrics.
            
            3. **Price Regime Status**: A classic Mark Minervini trend template applied to the benchmark index. 
               * **Confirmed Uptrend**: Market is acting strong above key MAs.
               * **Uptrend Under Pressure**: ≥ 4 Distribution Days (high volume sell-offs) have accumulated.
               * **Correction**: Price has fallen below the 50 SMA or ≥ 6 Distribution Days accumulated.
               
            4. **Wyckoff Alerts**: Monitors the benchmark for "Change of Character" events such as volume-climax capitulations at bottoms (Springs) or distribution at tops (Upthrusts).
            
            5. **Leading Themes (O'Neil)**: Shows the 3 fastest growing industry groups in the market right now based on an aggregation of the RS scores and price trends of their component stocks. 
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
        
        # Fetch sector & industry for holdings table
        sector, industry_detail = get_yfinance_sector_and_industry(ticker)
        decision['sector'] = sector

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

        decision['industry'] = industry_detail
        
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

    # Market Direction HUD
    # ---------------------------------------------------------
    # NEW: Top 5 RS Leaders (TraderLion / Deepvue Inspired)
    # ---------------------------------------------------------
    top_rs_leaders = get_top_5_rs_leaders(market='INDIA')
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
            font-size: 0.7rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .rs-rank-badge {
            position: absolute;
            top: -1px;
            left: 12px;
            background: rgba(0, 243, 255, 0.15);
            color: #00f3ff;
            padding: 3px 8px;
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
            font-size: 0.7rem;
            font-weight: 800;
            border: 1px solid rgba(0, 243, 255, 0.3);
            border-top: none;
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
        st.markdown("<br/>", unsafe_allow_html=True)
        
    top_rs_leaders_us = get_top_5_rs_leaders(market='US')
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
        
        import plotly.graph_objects as go
        
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
            "S&P 500": "^GSPC"
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
        
        cols = st.columns(4)
        for idx, (name, data) in enumerate(hud_data.items()):
            col = cols[idx % 4]
            
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
                
            i10 = "✅" if a10 else "❌"
            i21 = "✅" if a21 else "❌"
            islope = "✅" if slope else "❌"
            
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
                    import plotly.graph_objects as go
                    
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
        tab_l_in, tab_l_us = st.tabs(["🇮🇳 India (Nifty 500)", "🇺🇸 US Equities (Broad Market)"])
        
        def render_liquidity_ui(market_code, subtitle, unit_label):
            from database import get_market_liquidity
            liq_df = get_market_liquidity(days=1000, market=market_code)
            
            if liq_df.empty:
                st.info(f"No liquidity data available for {market_code}. Run liquidity_engine.py")
                return
                
            # Determine current regime
            last_turnover = liq_df.iloc[-1]['monthly_turnover_k_cr']
            last_sma = liq_df.iloc[-1]['sma_200']
            
            is_bull = last_turnover > last_sma
            status_color = "#10b981" if is_bull else "#ef4444"
            status_text = "EXPANDING (BULL)" if is_bull else "CONTRACTING (BEAR)"
            
            # Render sleek seamless Header
            st.markdown(f'''
            <div style="display: flex; justify-content: space-between; align-items: flex-end; padding: 10px 5px; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: -15px; margin-top: 10px;">
                <div>
                    <h3 style="margin: 0; color: #f8fafc; font-weight: 600; font-size: 1.15rem; display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.2rem;">💧</span> Institutional Liquidity
                    </h3>
                    <p style="margin: 0; font-size: 0.8rem; color: #64748b; margin-top: 2px; font-weight: 400;">{subtitle}</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.65rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px;">Market Regime</div>
                    <div style="font-size: 0.9rem; font-weight: 700; color: {status_color}; letter-spacing: 0.5px; filter: drop-shadow(0 0 4px {status_color});">
                        {status_text}
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            import plotly.graph_objects as go
            fig = go.Figure()
            
            # Ambient fill (Subtle Green)
            fig.add_trace(go.Scatter(
                x=liq_df['date'], y=liq_df['monthly_turnover_k_cr'],
                mode='none',
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.03)',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Actual Turnover Line (Smooth Spline)
            fig.add_trace(go.Scatter(
                x=liq_df['date'], y=liq_df['monthly_turnover_k_cr'],
                mode='lines',
                line=dict(color='#10b981', width=1.8, shape='spline', smoothing=0.5),
                name='Monthly Turnover',
                hovertemplate=f'Turnover: <b>%{{y:.1f}} {unit_label}</b><extra></extra>'
            ))
            
            # SMA Line (Dashed, Smooth, Subtle)
            fig.add_trace(go.Scatter(
                x=liq_df['date'], y=liq_df['sma_200'],
                mode='lines',
                line=dict(color='rgba(239, 68, 68, 0.6)', width=1.5, dash='dash', shape='spline', smoothing=1.3),
                name='200-Day SMA',
                hovertemplate=f'SMA 200: <b>%{{y:.1f}} {unit_label}</b><extra></extra>'
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
            render_liquidity_ui('IN', 'Nifty 500 Total Monthly Turnover vs 200-Day SMA', 'K cr')
            
        with tab_l_us:
            render_liquidity_ui('US', 'US Broad Market (1,300+ Stocks) Total Monthly Turnover vs 200-Day SMA', '$B')
        
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

main()

