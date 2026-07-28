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
    
    # Journal Notification
    today_str = datetime.now().strftime("%Y-%m-%d")
    if not get_journal_entry(today_str):
        st.warning("🚨 **Daily Journal Pending:** You have not logged your institutional bias today. Go to the **daily journal** page to review your analytics.")
    
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
    st.markdown("### 🛩️ Command Cockpit")
    
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
    
    cockpit1, cockpit2, cockpit3, cockpit4 = st.columns(4)
    
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
        <div class="metric-card {pulse_class}" style="border-top: 4px solid {urgent_color}; margin-bottom: 2rem;">
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">Immediate Actions</p>
            <h2 style="color: white; margin: 0.5rem 0; font-family: 'JetBrains Mono', monospace;">
                <span class="exit-text" style="color: #ef4444;">{exit_count} Exit</span> <span style="color: #475569;">|</span> 
                <span style="color: #f59e0b;">{trim_count} Trim</span>
            </h2>
            <p style="color: #cbd5e1; font-size: 0.8rem; margin: 0; font-family: 'Inter', sans-serif;">Tracking {add_count} Add setups</p>
        </div>
        """, unsafe_allow_html=True)
        
    with cockpit3:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid {gap_color}; margin-bottom: 2rem;">
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">Exposure vs Target</p>
            <h2 style="color: {gap_color}; margin: 0.5rem 0; font-family: 'JetBrains Mono', monospace;">{gap:+.1f}% Delta</h2>
            <p style="color: #cbd5e1; font-size: 0.8rem; margin: 0; font-family: 'Inter', sans-serif;">{gap_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with cockpit4:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid {equity_snap['color']}; margin-bottom: 2rem;">
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">Account Drawdown</p>
            <h2 style="color: {equity_snap['color']}; margin: 0.5rem 0; font-family: 'JetBrains Mono', monospace;">▼ {equity_snap['drawdown_pct']:.1f}%</h2>
            <p style="color: #cbd5e1; font-size: 0.75rem; margin: 0; font-family: 'Inter', sans-serif; line-height: 1.2;">{equity_snap['action']}</p>
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
        

    # 3. Averaging Up Opportunities (New Section)
    add_opps = [d for d in decisions if d['add_on_ready']]
    if add_opps:
        st.markdown("### 🎯 Averaging Up Opportunities")
        st.success("Strong accumulation candidates meeting all criteria (Trend, EMA, ATR, RS, Price).")
        
        cols = st.columns(len(add_opps)) if len(add_opps) <= 3 else st.columns(3)
        for i, d in enumerate(add_opps):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="background: #022c22; border: 1px solid #10b981; border-radius: 8px; padding: 1rem;">
                    <h3 style="color: #10b981; margin:0;">{d['ticker']} 🟢</h3>
                    <p style="color: #d1fae5; font-size: 0.9rem;">{d['reason']}</p>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #6ee7b7; margin-top: 0.5rem;">
                        <span>RS: {d.get('rs_score', 0):.0f}</span>
                        <span>ATR: {d.get('atr_state')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
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
                    import json
                    import os
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
    st.markdown("### 🗓️ Portfolio Earnings & Risk Hub")
    
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
                return datetime.strptime(q_str, '%b %Y')
            except Exception:
                return datetime(1970, 1, 1)

        max_q = max(set(quarters_found), key=_parse_q_date) if quarters_found else "Jun 2026"
        
        reported_count = sum(1 for e in filtered_cache.values() if e.get('latest_quarter') == max_q and not e.get('is_etf'))
        awaiting_count = sum(1 for e in filtered_cache.values() if e.get('latest_quarter') != max_q and not e.get('is_etf'))
        code33_count = sum(1 for e in filtered_cache.values() if e.get('dual_acceleration') and not e.get('is_etf'))



        c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
        with c_kpi1:
            st.markdown(f"""<div class="metric-card" style="border-top: 3px solid #10b981; padding: 0.75rem;"><p style="color:#94a3b8; margin:0; font-size:0.75rem; font-weight:bold;">{max_q} REPORTED</p><h3 style="color:#10b981; margin:0.2rem 0;">{reported_count} Holdings</h3></div>""", unsafe_allow_html=True)
        with c_kpi2:
            st.markdown(f"""<div class="metric-card" style="border-top: 3px solid #f59e0b; padding: 0.75rem;"><p style="color:#94a3b8; margin:0; font-size:0.75rem; font-weight:bold;">AWAITING {max_q}</p><h3 style="color:#f59e0b; margin:0.2rem 0;">{awaiting_count} Holdings</h3></div>""", unsafe_allow_html=True)
        with c_kpi3:
            st.markdown(f"""<div class="metric-card" style="border-top: 3px solid #3b82f6; padding: 0.75rem;"><p style="color:#94a3b8; margin:0; font-size:0.75rem; font-weight:bold;">DUAL ACCEL WINNERS</p><h3 style="color:#3b82f6; margin:0.2rem 0;">{code33_count} Stocks</h3></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        scorecard_rows = []
        etf_tickers = []
        for ticker, e_data in filtered_cache.items():

            if e_data.get('is_etf', False):
                etf_tickers.append(ticker)
                continue

            eps_traj = e_data.get('eps_trajectory', [])
            sales_traj = e_data.get('sales_trajectory', [])
            
            # Helper to extract label safely from dict
            def fmt_traj(traj_list, idx):
                if traj_list and 0 <= idx < len(traj_list):
                    t_item = traj_list[idx]
                    if isinstance(t_item, dict):
                        lbl = t_item.get('label', '-')
                        return lbl if lbl != '-' else '-'
                    elif isinstance(t_item, (int, float)):
                        return f"+{t_item:.1f}%" if t_item > 0 else f"{t_item:.1f}%"
                return "-"

            q1_eps = fmt_traj(eps_traj, 0)
            q2_eps = fmt_traj(eps_traj, 1)
            q3_eps = fmt_traj(eps_traj, 2)
            q4_eps = fmt_traj(eps_traj, 3) if len(eps_traj) >= 4 else fmt_traj(eps_traj, len(eps_traj) - 1) if len(eps_traj) > 0 else "-"
            
            q4_s = fmt_traj(sales_traj, 3) if len(sales_traj) >= 4 else fmt_traj(sales_traj, len(sales_traj) - 1) if len(sales_traj) > 0 else "-"

            
            l_q = e_data.get('latest_quarter', 'N/A')
            if l_q == max_q or e_data.get('is_new'):
                q_badge = f"🔥 {l_q} (NEW)"
                is_new_q = True
            elif l_q != 'N/A':
                q_badge = f"⏳ {l_q} (Awaiting {max_q})"
                is_new_q = False

            else:
                q_badge = "N/A"
                is_new_q = False

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
                'is_new_q': is_new_q
            })
            
        if scorecard_rows:
            scorecard_df = pd.DataFrame(scorecard_rows)
            
            # Row styling: highlight latest season reported stocks with subtle emerald tint
            def highlight_latest(row):
                if row.get('is_new_q'):
                    return ['background-color: rgba(16, 185, 129, 0.15); font-weight: bold'] * len(row)
                return [''] * len(row)
                
            display_df = scorecard_df.drop(columns=['is_new_q'])
            st.dataframe(display_df.style.apply(highlight_latest, axis=1), use_container_width=True, hide_index=True)
            
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
        *   **Footprint:** Institutional footprint. Ratio of Power Days (high volume up days) to Distribution Days (high volume down days) over the last 3 months.
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
        import json, os
        if os.path.exists('bse_mapping.json'):
            with open('bse_mapping.json', 'r') as f:
                bmap = json.load(f)
                bse_reverse_map = {str(v): k for k, v in bmap.items()}
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
            'Trend': f"{get_trend_emoji(d.get('trend_state', ''))} {d.get('trend_state', 'N/A').replace('🟢 ', '').replace('🟡 ', '').replace('🟠 ', '').replace('🔴 ', '')}",
            '8 EMA': d.get('above_8', False),
            '21 EMA': d.get('above_21', False),
            '50 SMA': d.get('above_50', False),
            '52W Dist': d.get('distance_from_52w_high', 0),
            'Dist 21EMA': d.get('dist_from_21ema', 999),
            'Ext 50SMA': d.get('ext_50sma', None),
            'Ext 200SMA': d.get('ext_200sma', None),
            'ADR%': d.get('adr_pct', 0),
            'Rel Vol': d.get('rel_vol', 0),
            'Sales Growth': fundas_cache.get(f"{d['ticker']}.NS", {}).get('sales_growth', fundas_cache.get(d['ticker'], {}).get('sales_growth', 0.0)),
            'RS Score': rs_score_val,
            'rs_blue_dot': d.get('rs_blue_dot', False),
            'Power Days': f"{d.get('power_days_3m', 0)} / {d.get('dist_days_3m', 0)}",
            'Days to Exit': d.get('days_to_exit', 0.0),
            'Add Ready': d.get('add_on_ready', False),
            'Action': action_map.get(action_val, action_val),
            'Reason': d.get('reason', '')[:60],
            'Webster': d.get('webster_signal', '⚪ Unknown')
        })
        
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
    
    # Restoring sorting capability via native Streamlit controls
    sort_col1, sort_col2 = st.columns([1, 3])
    with sort_col1:
        sort_metric = st.selectbox(
            "Sort Table By:", 
            ["RS Score (High to Low)", "Sales YoY (High to Low)", "Days to Exit (High to Low)", "Portfolio % (High to Low)", "Distance to 52W High", "ADR% (Volatility)", "Ticker (A-Z)"]
        )
        
    html_table = """
<style>
.kush-table { width: 100%; border-collapse: separate; border-spacing: 0 8px; font-family: 'Inter', sans-serif; }
.kush-table th { color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; text-align: left; padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.05); white-space: nowrap; }
.kush-row { background: rgba(255,255,255,0.03); transition: all 0.2s ease; border-radius: 8px; }
.kush-row:hover { background: rgba(255,255,255,0.06); transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
.kush-row td { padding: 14px 16px; font-size: 0.85rem; color: #f8fafc; border-top: 1px solid rgba(255,255,255,0.02); border-bottom: 1px solid rgba(255,255,255,0.02); white-space: nowrap; }
.kush-row td:first-child { border-left: 1px solid rgba(255,255,255,0.02); border-top-left-radius: 8px; border-bottom-left-radius: 8px; font-weight: 600; }
.kush-row td:last-child { border-right: 1px solid rgba(255,255,255,0.02); border-top-right-radius: 8px; border-bottom-right-radius: 8px; }
.badge-tml { background: rgba(234, 179, 8, 0.15); color: #fbbf24; padding: 4px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: 700; border: 1px solid rgba(234, 179, 8, 0.3); }
.badge-action-add { background: #22c55e; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; box-shadow: 0 0 10px rgba(34, 197, 94, 0.4); }
.badge-action-trim { background: #eab308; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; box-shadow: 0 0 10px rgba(234, 179, 8, 0.4); }
.badge-action-exit { background: #ef4444; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; box-shadow: 0 0 10px rgba(239, 68, 68, 0.4); }
.badge-action-hold { background: rgba(255, 255, 255, 0.1); color: #cbd5e1; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(255,255,255,0.2); }
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

    # Dynamically sort based on User Selection
    if sort_metric == "Days to Exit (High to Low)":
        sorted_table_data = sorted(table_data, key=lambda x: x.get('Days to Exit', 0) or 0, reverse=True)
    elif sort_metric == "Portfolio % (High to Low)":
        sorted_table_data = sorted(table_data, key=lambda x: x.get('Portfolio %', 0) or 0, reverse=True)
    elif sort_metric == "Sales YoY (High to Low)":
        sorted_table_data = sorted(table_data, key=lambda x: x.get('Sales Growth', 0) or 0, reverse=True)
    elif sort_metric == "Distance to 52W High":
        sorted_table_data = sorted(table_data, key=lambda x: x.get('52W Dist', 999) or 999)
    elif sort_metric == "ADR% (Volatility)":
        sorted_table_data = sorted(table_data, key=lambda x: x.get('ADR%', 0) or 0, reverse=True)
    elif sort_metric == "Ticker (A-Z)":
        sorted_table_data = sorted(table_data, key=lambda x: x.get('Ticker', ''))
    else:
        # Default to RS Score
        sorted_table_data = sorted(table_data, key=lambda x: x.get('RS Score', 0) or 0, reverse=True)
    
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
    html_lines.append(f'<div style="overflow-x: auto; background-color: #0f172a; padding: 20px; border-radius: 16px; border: 1px solid #1e293b; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">')
    html_lines.append(f'<div class="monitor-banner {banner_cls}">{banner_text}</div>')
    html_lines.append('<table class="kush-table">')
    html_lines.append('<thead><tr><th>Ticker</th><th>TML</th><th>Port %</th><th>Port Hit</th><th>Days Exit</th><th>Trend</th><th>8/21/50 EMA</th><th>52W Dist</th><th>Ext 50/200</th><th>Sales YoY</th><th>ADR% / Vol</th><th>RS Score</th><th>Footprint</th><th>Webster</th><th>Add</th><th>Action</th><th>Reason</th></tr></thead>')
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
        
        # Days to Exit
        dtl = r.get("Days to Exit", 0)
        dtl_color = "#10b981" if dtl < 1.0 else "#f59e0b" if dtl <= 3.0 else "#ef4444"
        dtl_fw = "800" if dtl > 3.0 else "600"
        html_lines.append(f'<td><span style="color: {dtl_color}; font-weight: {dtl_fw};">{dtl:.1f}d</span></td>')
        
        
        html_lines.append(f'<td>{r.get("Trend", "")}</td>')
        html_lines.append(f'<td style="font-size:0.7rem; letter-spacing: 2px;">{e8}{e21}{e50}</td>')
        
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
        html_lines.append(f'<td style="font-size:0.85rem; text-align: center;"><span style="color: #10b981; font-weight: bold;">{p_days.split(" / ")[0]}</span> <span style="color: #475569;">|</span> <span style="color: #ef4444; font-weight: bold;">{p_days.split(" / ")[1]}</span></td>')
        html_lines.append(f'<td><span style="font-size:0.8rem;">{r.get("Webster", "")}</span></td>')
        html_lines.append(f'<td>{add_icon}</td>')
        html_lines.append(f'<td>{act_html}</td>')
        html_lines.append(f'<td><span style="color:#94a3b8; font-size:0.8rem;">{r.get("Reason", "")}</span></td>')
        html_lines.append('</tr>')
        
    html_table += "".join(html_lines) + "</tbody></table></div>"
    st.markdown(html_table, unsafe_allow_html=True)
    
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
    st.markdown("### 📡 Corporate Filings Radar (Last 7 Days)")
    
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
            else:
                import html
            
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
    
    # RS Score Distribution
    # Portfolio Roppel Quadrant
    st.markdown("### 🔭 The Portfolio Roppel Quadrant")
    
    tab_tech, tab_funda = st.tabs(["Technical (RS vs P/L)", "Fundamental (RS vs Sales Growth)"])
    
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
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; padding: 1rem;">
        <p>Kush Tracker v1.2 | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p style="font-size: 0.85rem;">Inspired by Qullamaggie, Mark Minervini, William O'Neil, Paul Tudor Jones</p>
    </div>
    """, unsafe_allow_html=True)


main()
