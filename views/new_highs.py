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
import plotly.express as px
import plotly.graph_objects as go

from components import render_header, render_metric_card, apply_plotly_theme, render_disk_cache_sidebar
from market_data import fetch_nifty_total_market_tickers

import importlib
import new_highs_engine
try:
    importlib.reload(new_highs_engine)
except Exception:
    pass

from new_highs_engine import (
    load_price_matrix,
    load_industry_map,
    compute_new_highs_universe,
    get_industry_clustering_stats
)
try:
    from views.true_market_leader import get_cached_universe
except ImportError:
    get_cached_universe = None

# Page configuration (guarded for st.navigation routing in Lite)
try:
    st.set_page_config(
        page_title="New Highs & Blue Sky Terminal",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
except Exception:
    pass

render_header(
    title="New Highs & Blue Sky Terminal",
    subtitle="Institutional Breakout & Expansion Radar: Lifetime All-Time Highs (ATH), Multi-Year Breakouts, 52W Highs, Dual Momentum & Sortino Leadership",
    icon="🌟"
)

@st.cache_data(ttl=900, show_spinner="Syncing price history & scanning breakouts on the go... Please wait ~20-30s...")
def get_cached_new_highs_data(universe_mode: str = "NIFTY 750 (High Conviction)"):
    """
    Loads price matrix and runs full vectorized new highs quantitative engine.
    If historical_prices_matrix.pkl does not exist (e.g. Streamlit Cloud),
    downloads on the go using price_history_manager.
    """
    industry_map = load_industry_map()
    
    if universe_mode == "NIFTY 750 (High Conviction)":
        tickers, nifty_ind = fetch_nifty_total_market_tickers(return_industry_map=True)
        if nifty_ind:
            industry_map = {**nifty_ind, **industry_map}
    else:
        tickers_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tickers.txt")
        if os.path.exists(tickers_file):
            try:
                with open(tickers_file, "r") as f:
                    tickers = [line.strip().upper() for line in f if line.strip() and '-' not in line]
            except Exception:
                tickers = fetch_nifty_total_market_tickers(show_progress=False)
        else:
            tickers = fetch_nifty_total_market_tickers(show_progress=False)

    try:
        close_df, high_df, low_df, volume_df = load_price_matrix(tickers=tickers, days=252)
    except TypeError:
        import importlib
        import new_highs_engine
        importlib.reload(new_highs_engine)
        close_df, high_df, low_df, volume_df = new_highs_engine.load_price_matrix(tickers=tickers, days=252)
    
    if close_df.empty and tickers:
        try:
            from price_history_manager import fetch_incremental_history
            fetch_incremental_history(tickers, days=252)
            close_df, high_df, low_df, volume_df = new_highs_engine.load_price_matrix(tickers=tickers, days=252)
        except Exception as e:
            print(f"Incremental history download failed: {e}")
        
    df = compute_new_highs_universe(
        close_df=close_df,
        high_df=high_df,
        low_df=low_df,
        volume_df=volume_df,
        industry_map=industry_map,
        tickers=tickers,
        min_price=10.0,
        compute_sortino=True
    )
    return df

# -----------------------------------------------------------------------------
# SIDEBAR / CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Universe Configuration")
    universe_mode = st.radio(
        "Select Universe:",
        ["NIFTY 750 (High Conviction)", "ALL NSE Equities (2000+)"],
        index=0,
        help="NIFTY 750 filters for institutional liquidity and eliminates illiquid penny stocks."
    )
    if st.button("🔄 Refresh Data Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if get_cached_universe is not None:
        render_disk_cache_sidebar(get_cached_universe)

# Load data
df = get_cached_new_highs_data(universe_mode)

# Defensive check: If stale cache without 'Days Since ATH' is loaded, clear cache & reload
if not df.empty and 'Days Since ATH' not in df.columns:
    get_cached_new_highs_data.clear()
    df = get_cached_new_highs_data(universe_mode)

if df.empty:
    st.info("📡 Price history matrix is building on the go from Yahoo Finance. Please wait a moment or click below to initialize:")
    if st.button("🚀 Initialize & Sync Universe History Now", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.stop()

# -----------------------------------------------------------------------------
# TOP KPI SUMMARY METRIC CARDS
# -----------------------------------------------------------------------------
total_ath_today = int(df['Is ATH'].sum()) if 'Is ATH' in df.columns else 0
if 'Days Since ATH' in df.columns and 'Is ATH' in df.columns:
    total_ath_1m = int(((df['Days Since ATH'] <= 21) | df['Is ATH']).sum())
else:
    total_ath_1m = total_ath_today

total_buy_zone = int((df['Setup Tag'] == "🎯 Near 21 EMA (Buy Zone)").sum()) if 'Setup Tag' in df.columns else 0
total_52w_today = int(df['Is 52W High'].sum()) if 'Is 52W High' in df.columns else 0
total_coiling = int(df['Is Coiling'].sum()) if 'Is Coiling' in df.columns else 0

ind_stats_full = get_industry_clustering_stats(df)
top_ind_name = ind_stats_full['Industry'].iloc[0] if not ind_stats_full.empty else "N/A"
top_ind_count = int(ind_stats_full['Total_Highs'].iloc[0]) if not ind_stats_full.empty else 0

kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
with kpi_c1:
    render_metric_card("🚀 Lifetime ATH Today", f"{total_ath_today}", color_class="green-text", extra_style="text-align:center;")
with kpi_c2:
    render_metric_card("🏆 1-Month ATH Leaders", f"{total_ath_1m}", color_class="green-text", extra_style="text-align:center;")
with kpi_c3:
    render_metric_card("🎯 Near 21 EMA (Buy Zone)", f"{total_buy_zone}", color_class="cyan-text", extra_style="text-align:center;")
with kpi_c4:
    render_metric_card("🌟 52W Highs Today", f"{total_52w_today}", color_class="yellow-text", extra_style="text-align:center;")
with kpi_c5:
    render_metric_card("🔭 Coiling Near ATH (<5%)", f"{total_coiling}", color_class="cyan-text", extra_style="text-align:center;")

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HORIZONTAL FILTER BAR (JUST ABOVE TABLES)
# -----------------------------------------------------------------------------
st.markdown("#### 🎯 Leadership Roster & Breakout Filters")
f_row1_col1, f_row1_col2, f_row1_col3, f_row1_col4 = st.columns([1.5, 1.3, 1.4, 1.8])

with f_row1_col1:
    horizon_choice = st.selectbox(
        "Leadership Horizon:",
        [
            "🏆 1-Month Leadership Roster (Incubator)",
            "🔥 1-Week Leaders (Last 5D)",
            "⚡ Today's Breakouts Only",
            "All Active Stocks"
        ],
        index=0,
        help="1-Month keeps leaders on radar for 21 days so you can buy them on pullbacks to 21 EMA!"
    )

with f_row1_col2:
    tier_choice = st.selectbox(
        "High Tier Filter:",
        ["🌟 All Highs (ATH + 52W)", "🚀 Lifetime ATH Only", "💎 Multi-Year Breakouts", "🌟 52W Highs Only"],
        index=0
    )

with f_row1_col3:
    setup_choice = st.selectbox(
        "Setup / Buy Zone:",
        ["All Setups", "🎯 Near 21 EMA (Buy Zone)", "⚡ Breaking Out Today", "🛡️ Near 50 SMA / Shakeout", "⏳ Extended (>7%)"],
        index=0
    )

with f_row1_col4:
    search_query = st.text_input("🔍 Search Ticker / Industry:", value="", placeholder="e.g. TATA, Steel, MBECL").strip().upper()

f_row2_col1, f_row2_col2, f_row2_col3, f_row2_col4 = st.columns(4)

with f_row2_col1:
    min_rs = st.slider("Min RS Rating:", min_value=1, max_value=99, value=60, step=5, help="Custom 0.40*1M + 0.40*3M + 0.20*6M Percentile Rank (1-99)")

with f_row2_col2:
    min_sortino = st.slider("Min Sortino 3M:", min_value=0.0, max_value=8.0, value=0.0, step=0.5, help="Annualized downside risk ratio (MAR 6.5%)")

with f_row2_col3:
    min_adtv = st.slider("Min ADTV (Cr):", min_value=0.0, max_value=25.0, value=0.0, step=0.50, format="₹%.2f Cr", help="20-day Average Daily Turnover in Crores (filters out illiquid stocks)")

with f_row2_col4:
    min_vol_exp = st.slider("Min Vol Exp:", min_value=0.5, max_value=3.0, value=1.0, step=0.1, help="Today Volume / 20D Average Volume")

# Apply Filters
filtered_df = df.copy()

# Horizon filter
if 'Highs 1M' in filtered_df.columns:
    if horizon_choice == "🏆 1-Month Leadership Roster (Incubator)":
        filtered_df = filtered_df[filtered_df['Highs 1M'] >= 1]
    elif horizon_choice == "🔥 1-Week Leaders (Last 5D)":
        filtered_df = filtered_df[filtered_df['Highs 1W'] >= 1]
    elif horizon_choice == "⚡ Today's Breakouts Only":
        filtered_df = filtered_df[filtered_df['Days Since High'] == 0]

# Tier filtering
if tier_choice == "🌟 All Highs (ATH + 52W)" and 'Tier' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Tier'].str.contains("ATH|High", na=False)]
elif tier_choice == "🚀 Lifetime ATH Only" and 'Tier' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Tier'].str.contains("ATH", na=False)]
elif tier_choice == "💎 Multi-Year Breakouts" and 'Tier' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Tier'].str.contains("3-Year|2-Year", na=False)]
elif tier_choice == "🌟 52W Highs Only" and 'Tier' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Tier'].str.contains("52W", na=False)]

# Setup filtering
if 'Setup Tag' in filtered_df.columns:
    if setup_choice == "🎯 Near 21 EMA (Buy Zone)":
        filtered_df = filtered_df[filtered_df['Setup Tag'] == "🎯 Near 21 EMA (Buy Zone)"]
    elif setup_choice == "⚡ Breaking Out Today":
        filtered_df = filtered_df[filtered_df['Setup Tag'] == "⚡ Breaking Out Today"]
    elif setup_choice == "🛡️ Near 50 SMA / Shakeout":
        filtered_df = filtered_df[filtered_df['Setup Tag'].str.contains("50 SMA|Shakeout", na=False)]
    elif setup_choice == "⏳ Extended (>7%)":
        filtered_df = filtered_df[filtered_df['Setup Tag'] == "⏳ Extended (>7%)"]

# Slider filters
filtered_df = filtered_df[
    (filtered_df['RS Rating'] >= min_rs) &
    (filtered_df['Sortino 3M'] >= min_sortino)
]

if 'ADTV (Cr)' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['ADTV (Cr)'] >= min_adtv]
elif 'Turnover Cr' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Turnover Cr'] >= min_adtv]

if 'Vol Exp' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Vol Exp'] >= min_vol_exp]

# Search query
if search_query:
    filtered_df = filtered_df[
        filtered_df['Ticker'].str.contains(search_query, na=False) |
        filtered_df['Industry'].str.upper().str.contains(search_query, na=False)
    ]

# Coiling Setups filtering
coiling_df = df[df['Is Coiling']].copy()
coiling_df = coiling_df[
    (coiling_df['RS Rating'] >= min_rs) &
    (coiling_df['Sortino 3M'] >= min_sortino)
]

if 'ADTV (Cr)' in coiling_df.columns:
    coiling_df = coiling_df[coiling_df['ADTV (Cr)'] >= min_adtv]
elif 'Turnover Cr' in coiling_df.columns:
    coiling_df = coiling_df[coiling_df['Turnover Cr'] >= min_adtv]

if search_query:
    coiling_df = coiling_df[
        coiling_df['Ticker'].str.contains(search_query, na=False) |
        coiling_df['Industry'].str.upper().str.contains(search_query, na=False)
    ]

coiling_df = coiling_df.sort_values(['Dist ATH %', 'Dual Score'], ascending=[True, False]).reset_index(drop=True)

# -----------------------------------------------------------------------------
# TABS INTERFACE
# -----------------------------------------------------------------------------
tab_breakouts, tab_coiling, tab_industry, tab_methodology = st.tabs([
    f"🏆 Leadership Roster ({len(filtered_df)})",
    f"🔭 Pre-Breakout Radar ({len(coiling_df)})",
    "📊 Industry & Theme Clustering",
    "📘 Quantitative Methodology Guide"
])

# -----------------------------------------------------------------------------
# TAB 1: LEADERSHIP ROSTER (BREAKOUTS & PULLBACKS)
# -----------------------------------------------------------------------------
with tab_breakouts:
    st.markdown(f"### 🏆 Leadership Roster & Pullback Radar ({len(filtered_df)} stocks matched)")
    st.caption("Active CANSLIM leaders making new highs within the selected horizon. Identify fresh breakouts or low-risk 21 EMA pullbacks:")
    
    if filtered_df.empty:
        st.info("No stocks matched the active filters. Try switching Horizon to 'All Active Stocks' or lowering Min RS / Min ADTV sliders.")
    else:
        # TradingView Watchlist Export Box
        tv_tickers = [f"NSE:{t}" for t in filtered_df['Ticker'].tolist()]
        tv_export_str = ",".join(tv_tickers)
        with st.expander(f"📋 Copy TradingView Watchlist ({len(tv_tickers)} tickers)"):
            st.code(tv_export_str, language="text")
            st.caption("Paste directly into TradingView Watchlist: `+` button -> paste text -> Enter.")

        cols_to_disp = [
            'TradingView', 'Ticker', 'Tier', 'Price', 'Today %',
            'Days Since High Text', 'Setup Tag', 'Dist 21EMA %',
            'Highs 1W', 'Highs 1M', 'RS Rating', 'Dual Score', 'Dual Rank',
            'Sortino 3M', 'Sortino 6M'
        ]
        if 'ADTV (Cr)' in filtered_df.columns:
            cols_to_disp.append('ADTV (Cr)')
        elif 'Turnover Cr' in filtered_df.columns:
            cols_to_disp.append('Turnover Cr')
        cols_to_disp.extend(['Vol Exp', 'Close Range %', 'Industry'])

        display_df = filtered_df[cols_to_disp].copy()
        
        st.dataframe(
            display_df,
            column_config={
                "TradingView": st.column_config.LinkColumn("Chart", display_text="↗", width="small"),
                "Ticker": st.column_config.TextColumn("Ticker", width="medium"),
                "Tier": st.column_config.TextColumn("Breakout Tier", width="medium"),
                "Price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                "Today %": st.column_config.NumberColumn("Today %", format="%+.2f%%"),
                "Days Since High Text": st.column_config.TextColumn("Last High", width="small", help="When the stock last made a new high"),
                "Setup Tag": st.column_config.TextColumn("Setup / Buy Zone", width="medium", help="CANSLIM / Minervini low-risk pullback zone"),
                "Dist 21EMA %": st.column_config.NumberColumn("Dist 21 EMA", format="%+.1f%%", help="Distance from 21 EMA. 0% to 3.5% is prime buy zone"),
                "Highs 1W": st.column_config.NumberColumn("Highs 1W", format="%d", help="Count of new highs printed in the last 5 sessions (range 0-5)"),
                "Highs 1M": st.column_config.NumberColumn("Highs 1M", format="%d", help="Count of new highs printed in the last 21 sessions (range 0-21)"),
                "RS Rating": st.column_config.ProgressColumn("RS Rating", min_value=1, max_value=99, format="%d", help="Custom 0.40*1M + 0.40*3M + 0.20*6M Percentile Rank"),
                "Dual Score": st.column_config.NumberColumn("Dual Score", format="%.1f", help="Composite 50% RS + 50% Absolute Velocity"),
                "Dual Rank": st.column_config.NumberColumn("Dual Rank", format="#%d"),
                "Sortino 3M": st.column_config.NumberColumn("Sortino 3M", format="%.2f", help="Annualized downside risk ratio (63D) with MAR = 6.5%"),
                "Sortino 6M": st.column_config.NumberColumn("Sortino 6M", format="%.2f"),
                "ADTV (Cr)": st.column_config.NumberColumn("ADTV", format="₹%.2f Cr", help="20-day Average Daily Turnover in Crores"),
                "Turnover Cr": st.column_config.NumberColumn("ADTV", format="₹%.2f Cr"),
                "Vol Exp": st.column_config.NumberColumn("Vol Exp", format="%.1fx", help="Today's Volume vs 20D Average"),
                "Close Range %": st.column_config.ProgressColumn("Close Range %", min_value=0, max_value=100, format="%d%%", help="Close position within day's High-Low range (100% = Closed at high)"),
                "Industry": st.column_config.TextColumn("Industry Group", width="large")
            },
            hide_index=True,
            use_container_width=True
        )

# -----------------------------------------------------------------------------
# TAB 2: PRE-BREAKOUT RADAR (COILING 0-5% FROM ATH)
# -----------------------------------------------------------------------------
with tab_coiling:
    st.markdown(f"### 🔭 Pre-Breakout Radar: Tight Consolidations Coiling Near ATH ({len(coiling_df)} setups)")
    st.caption("Stocks trading within 0.1% to 5.0% below their Lifetime All-Time High. These represent high-conviction bases setting up right before entering Blue Sky territory:")
    
    if coiling_df.empty:
        st.info("No stocks currently coiling within 5% of ATH match the active filters.")
    else:
        coil_tv_tickers = [f"NSE:{t}" for t in coiling_df['Ticker'].tolist()]
        with st.expander(f"📋 Copy TradingView Watchlist ({len(coil_tv_tickers)} coiling setups)"):
            st.code(",".join(coil_tv_tickers), language="text")

        coil_cols = [
            'TradingView', 'Ticker', 'Price', 'Dist ATH %', 'Dist 52W %',
            'RS Rating', 'Dual Score', 'Dual Rank', 'Sortino 3M', 'Sortino 6M'
        ]
        if 'ADTV (Cr)' in coiling_df.columns:
            coil_cols.append('ADTV (Cr)')
        elif 'Turnover Cr' in coiling_df.columns:
            coil_cols.append('Turnover Cr')
        coil_cols.extend(['Dist 21EMA %', 'Dist 50SMA %', 'Industry'])

        disp_coil = coiling_df[coil_cols].copy()
        
        st.dataframe(
            disp_coil,
            column_config={
                "TradingView": st.column_config.LinkColumn("Chart", display_text="↗", width="small"),
                "Ticker": st.column_config.TextColumn("Ticker", width="medium"),
                "Price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                "Dist ATH %": st.column_config.NumberColumn("Dist to ATH %", format="%.2f%%", help="Distance below Lifetime ATH"),
                "Dist 52W %": st.column_config.NumberColumn("Dist to 52W %", format="%.2f%%"),
                "RS Rating": st.column_config.ProgressColumn("RS Rating", min_value=1, max_value=99, format="%d"),
                "Dual Score": st.column_config.NumberColumn("Dual Score", format="%.1f"),
                "Dual Rank": st.column_config.NumberColumn("Dual Rank", format="#%d"),
                "Sortino 3M": st.column_config.NumberColumn("Sortino 3M", format="%.2f"),
                "Sortino 6M": st.column_config.NumberColumn("Sortino 6M", format="%.2f"),
                "ADTV (Cr)": st.column_config.NumberColumn("ADTV", format="₹%.2f Cr", help="20-day Average Daily Turnover in Crores"),
                "Turnover Cr": st.column_config.NumberColumn("ADTV", format="₹%.2f Cr"),
                "Dist 21EMA %": st.column_config.NumberColumn("Dist 21 EMA %", format="%+.2f%%"),
                "Dist 50SMA %": st.column_config.NumberColumn("Dist 50 SMA %", format="%+.2f%%"),
                "Industry": st.column_config.TextColumn("Industry Group", width="large")
            },
            hide_index=True,
            use_container_width=True
        )

# -----------------------------------------------------------------------------
# TAB 3: INDUSTRY & THEME CLUSTERING
# -----------------------------------------------------------------------------
with tab_industry:
    st.markdown("### 📊 Industry & Theme Concentration Among New High Leaders")
    st.caption("Breakouts that cluster together in the same sector indicate powerful institutional industry sponsorship (leading theme of the cycle):")
    
    ind_stats = get_industry_clustering_stats(filtered_df if not filtered_df.empty else df, top_n=15)
    
    if ind_stats.empty:
        st.info("No industry clusters detected with active filters.")
    else:
        # Plotly Bar Chart
        fig = px.bar(
            ind_stats,
            x='Total_Highs',
            y='Industry',
            orientation='h',
            color='Avg_RS',
            color_continuous_scale='Greens',
            title="Leading Industry Groups by New High Count Today",
            labels={'Total_Highs': 'Total New Highs Today', 'Industry': 'Industry Group', 'Avg_RS': 'Average RS Rating'},
            hover_data={'Leaders': True, 'Avg_Sortino_3M': ':.2f'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=480)
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        # Breakdown Table
        st.dataframe(
            ind_stats,
            column_config={
                "Industry": st.column_config.TextColumn("Industry Group"),
                "Total_Highs": st.column_config.NumberColumn("New Highs Today", format="%d"),
                "Lifetime_ATH_Count": st.column_config.NumberColumn("Lifetime ATHs", format="%d"),
                "Avg_RS": st.column_config.ProgressColumn("Avg RS Rating", min_value=1, max_value=99, format="%.1f"),
                "Avg_Sortino_3M": st.column_config.NumberColumn("Avg Sortino 3M", format="%.2f"),
                "Leaders": st.column_config.TextColumn("Leading Breakout Tickers", width="large")
            },
            hide_index=True,
            use_container_width=True
        )

# -----------------------------------------------------------------------------
# TAB 4: METHODOLOGY & QUANTITATIVE GUIDE
# -----------------------------------------------------------------------------
with tab_methodology:
    st.markdown(r"""
    ### 📘 Institutional New Highs & Blue Sky Methodology
    
    #### 1. Why All-Time Highs (ATH) Outperform 52-Week Highs
    - **Overhead Supply Absorption**: A stock at a 52-week high may still be down $-40\%$ from an all-time peak reached 2 or 3 years ago. As it rallies, investors who bought at the peak sell to break even, creating strong resistance.
    - **Blue Sky Territory**: When a stock achieves an **All-Time High (ATH)**, **100% of existing shareholders are in profit**. There are zero bagholders looking to sell at breakeven. Institutional accumulation encounters zero natural overhead supply, allowing prices to expand rapidly.
    
    #### 2. Highs Frequency: 1-Week and 1-Month Counts
    - Rather than looking at a single day's high in isolation, our engine measures **Highs Frequency**:
      $$\text{Highs (1W)} = \sum_{t=1}^{5} \mathbb{I}(\text{High}_t \ge \text{Prior Peak})$$
      $$\text{Highs (1M)} = \sum_{t=1}^{21} \mathbb{I}(\text{High}_t \ge \text{Prior Peak})$$
    - A stock printing **3 to 5 new highs in a week** or **8+ new highs in a month** demonstrates unrelenting institutional demand that absorbs all selling liquidity day after day.
    
    #### 3. Custom RS Rating Formula
    $$\text{RS Raw} = (0.40 \times \text{Return}_{1\text{M}}) + (0.40 \times \text{Return}_{3\text{M}}) + (0.20 \times \text{Return}_{6\text{M}})$$
    $$\text{RS Rating} = \lfloor \text{PercentileRank}(\text{RS Raw}) \times 98 \rfloor + 1 \quad (1 \dots 99)$$
    - Heavily weights recent momentum (80% over 1M and 3M horizons) with 20% anchoring to 6-month intermediate trend stability.
    
    #### 4. Dual Momentum Composite (Antonacci & Dhawan)
    - **Relative Strength Percentile**: Cross-sectional outperformance vs all peers.
    - **Absolute Velocity Score**: Short-term momentum speed ($0.10 \times 1\text{W} + 0.40 \times 1\text{M} + 0.50 \times 3\text{M}$).
    - **Dual Score**:
      $$\text{Dual Score} = 0.50 \times \text{RS Percentile} + 0.50 \times \text{Velocity Percentile}$$
    - Filtered by the **Absolute Gate**: Price must be $> 200\text{ SMA}$ and Velocity must be $> 0$.
    
    #### 5. Dual-Horizon Sortino Ratios (3M & 6M)
    - Unlike Sharpe Ratio, which penalizes upside explosive moves, the **Sortino Ratio** only penalizes **downside volatility** below the annual risk-free hurdle rate (6.5% for Indian Treasuries):
      $$\text{Sortino} = \frac{R_p - R_f}{\sigma_d}$$
    - A Sortino $> 3.0$ indicates that the stock's climb is steady, institutional, and devoid of violent drop-offs.
    
    #### 6. The 1-Month Leadership Incubator & Low-Risk Pullback Entries
    - **Why We Keep Leaders on the Roster for 1 Month**: Professional growth traders do not chase stocks extended $+10\%$ to $+15\%$ above their breakout pivot. After printing a new 52W high or ATH, institutional winners typically digest gains over 5 to 15 trading days.
    - **The 21 EMA Buy Zone**: When a stock that printed a new high within the last month pulls back calmly to within **$0\% \dots 3.5\%$ of its rising 21-day EMA** on declining volume, it presents the **lowest-risk, highest-reward follow-on entry** of the entire move.
    """)
