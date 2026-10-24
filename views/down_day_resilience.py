"""
Down-Day Resilient Leaders & Historical Washout Radar
====================================================
Point-in-Time Alpha Discovery: Identify which stocks demonstrated massive institutional
accumulation, broke out to new RS highs, or held green like "Beach Balls under water"
on any past or live market correction day.
"""

import os
import sys
import pandas as pd
import streamlit as st
from datetime import datetime

# Path setup
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

try:
    from styles import load_css
    load_css()
except ImportError:
    pass

from components import render_header, render_metric_card
import down_day_engine as dde

try:
    st.set_page_config(
        page_title="Down-Day Resilient Leaders | Kush Tracker Lite",
        page_icon="🏖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    pass

def clean_html(html_str: str) -> str:
    return "\n".join(line.strip() for line in html_str.splitlines() if line.strip())

def main():
    render_header(
        "🏖️ Down-Day Resilient Leaders (Point-in-Time Alpha Radar)",
        "Discover the 'Beach Balls Held Under Water' & RS Divergence Leaders on Any Market Down Day"
    )

    with st.expander("📖 William J. O'Neil & David Ryan: The Law of Down-Day Resilience", expanded=False):
        st.markdown(r"""
        > *"During general market declines of 10% to 20%, 90% of all stocks decline with the averages. **The 10% that resist the decline, hold their ground, or actually advance to new highs are almost always the huge winners of the next market cycle.** Smart money uses market panic to quietly accumulate their highest-conviction ideas without pushing the price through the roof."*  
        > — **William J. O'Neil**, *How to Make Money in Stocks*
        
        ### The 3 Resilience Archetypes Detected by This Engine:
        1. 🏖️ **Beach Ball Accumulation (VB Power Play)**: Price thrusts up $\ge +3.8\%$ on volume $\ge 1.0\times$ 20D ADTV during severe market breadth compression ($< 35\%$ 50-SMA breadth).
        2. 🚀 **New 52-Week High on Down Day**: Stock closes within $0.5\%$ of its absolute 52-week peak while the broad market is tumbling.
        3. 🟢 **Green on Red (Relative Strength Anchor)**: Stock closes firmly green ($> 0\%$) and sits within $15\%$ of its 52-week high, refusing to roll over.
        """)

    # -------------------------------------------------------------
    # 1. CONTROLS & MARKET SELECTION
    # -------------------------------------------------------------
    c_mkt, c_lookback, c_filter_arch, c_min_price = st.columns([1.2, 1.2, 1.5, 1.1])
    with c_mkt:
        market_choice = st.selectbox(
            "Market Universe",
            ["🇮🇳 India (NSE)", "🇺🇸 US (S&P 500 / Nasdaq)"],
            index=0,
            key="ddr_market_select"
        )
    is_us = "US" in market_choice
    market_code = "US" if is_us else "INDIA"
    curr_symbol = "$" if is_us else "₹"

    with c_lookback:
        lookback_days = st.selectbox(
            "Historical Lookback",
            ["Last 90 Days", "Last 180 Days", "Last 365 Days", "All History (2 Years)"],
            index=2,
            key="ddr_lookback_select"
        )
        days_num = 730 if "All" in lookback_days else (365 if "365" in lookback_days else (180 if "180" in lookback_days else 90))

    with c_filter_arch:
        arch_filter = st.selectbox(
            "Filter Resilience Archetype",
            [
                "🌟 All Resilient Leaders",
                "🏖️ Beach Ball (VB Power Plays) Only",
                "🚀 New 52W Highs on Down Day Only",
                "⚡ Institutional Accumulation Only",
                "🟢 Green on Red (Relative Anchors)"
            ],
            index=0,
            key="ddr_archetype_filter"
        )

    with c_min_price:
        min_price = st.number_input(
            f"Min Price ({curr_symbol})",
            value=20.0 if not is_us else 5.0,
            step=5.0,
            key="ddr_min_price"
        )

    # -------------------------------------------------------------
    # 2. QUERY HISTORICAL WASHOUT & DOWN DATES
    # -------------------------------------------------------------
    with st.spinner("Analyzing historical market corrections & breadth washouts..."):
        df_washouts = dde.get_historical_market_down_dates(
            market=market_code,
            min_drop_pct=0.75,
            max_breadth_pct=35.0,
            lookback_days=days_num
        )

    if df_washouts.empty:
        st.warning(f"No qualifying market washout or severe down sessions found for {market_choice} in the selected lookback.")
        return

    # Date selector dropdown with rich metadata
    date_options = df_washouts['date'].tolist()
    
    def format_date_label(d_str):
        row = df_washouts[df_washouts['date'] == d_str].iloc[0]
        b_pct = row['above_50_pct']
        nnh = int(row['net_new_highs'])
        badge = row['severity_badge']
        adv = row['advances']
        dec = row['declines']
        return f"{d_str}  ·  {badge}  (Breadth: {b_pct:.1f}% > 50SMA • Net Highs: {nnh:+d} • A/D: {adv}/{dec})"

    st.markdown("#### 📅 Select Market Down Day / Breadth Washout Session")
    selected_date = st.selectbox(
        "Choose a historical correction date to inspect resilient leaders:",
        options=date_options,
        index=0,
        format_func=format_date_label,
        key="ddr_date_picker",
        help="Select any date when breadth broke down or market dropped hard to uncover which stocks held strong."
    )

    date_meta = df_washouts[df_washouts['date'] == selected_date].iloc[0]

    # Executive HUD Cards for the selected session
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Session Date", selected_date, subtitle="Historical Correction")
    with m2:
        b_val = date_meta['above_50_pct']
        b_class = "red-text" if b_val <= 20.0 else ("orange-text" if b_val <= 35.0 else "blue-text")
        render_metric_card("50-SMA Market Breadth", f"{b_val:.1f}%", subtitle="Washed Out Regime", color_class=b_class)
    with m3:
        nnh_val = int(date_meta['net_new_highs'])
        nnh_class = "red-text" if nnh_val < 0 else "green-text"
        render_metric_card("Net 52W Highs/Lows", f"{nnh_val:+d}", subtitle=f"New Highs vs New Lows", color_class=nnh_class)
    with m4:
        adv = date_meta['advances']
        dec = date_meta['declines']
        render_metric_card("Advance / Decline", f"{adv} / {dec}", subtitle=f"Ratio: {adv/max(1,dec):.2f}")

    # -------------------------------------------------------------
    # 3. COMPUTE OR FETCH POINT-IN-TIME RESILIENT STOCKS (FAST DB QUERY)
    # -------------------------------------------------------------
    with st.spinner(f"Extracting Point-in-Time resilience data for {selected_date}..."):
        df_resilient = dde.compute_down_day_resilience(
            target_date_str=selected_date,
            market=market_code,
            min_price=min_price,
            max_dist_high=15.0
        )

    if df_resilient.empty:
        st.info(f"ℹ️ No resilient stocks matched the criteria on {selected_date} (all stocks declined or sat >15% below 52W highs).")
        return

    # Filter by selected Archetype
    if "Beach Ball" in arch_filter:
        df_display = df_resilient[df_resilient['Archetype'].str.contains("Beach Ball")].copy()
    elif "New 52W High" in arch_filter:
        df_display = df_resilient[df_resilient['Archetype'].str.contains("New 52W High")].copy()
    elif "Institutional Accumulation" in arch_filter:
        df_display = df_resilient[df_resilient['Archetype'].str.contains("Institutional Accumulation")].copy()
    elif "Green on Red" in arch_filter:
        df_display = df_resilient[df_resilient['Archetype'].str.contains("Green on Red")].copy()
    else:
        df_display = df_resilient.copy()

    # -------------------------------------------------------------
    # 4. ACTIONABLE MULTI-PLATFORM EXPORT BAR
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📤 Multi-Platform Actionable Export")
    st.caption("Instantly bridge this intelligence to your phone, TradingView mobile app, Notion workspace, or Telegram lock-screen.")

    e_tv, e_notion, e_tg, e_csv = st.columns([1.3, 1.3, 1.3, 1.0])

    tv_string = dde.format_tradingview_watchlist(df_display['Clean_Ticker'].head(30).tolist(), market=market_code)
    notion_table = dde.format_notion_markdown_table(df_display, selected_date)
    tg_alert = dde.format_telegram_alert(
        selected_date,
        date_meta['above_50_pct'],
        int(date_meta['net_new_highs']),
        df_display,
        market=market_code
    )

    with e_tv:
        with st.popover(f"📋 TradingView Watchlist ({len(df_display)})"):
            st.markdown("**Copy & paste into TradingView Symbol Search / Watchlist:**")
            st.code(tv_string, language="text")

    with e_notion:
        with st.popover("📓 Copy for Notion"):
            st.markdown("**Copy this formatted Markdown table and paste directly into Notion:**")
            st.code(notion_table, language="markdown")

    with e_tg:
        with st.popover("📱 Telegram Lock-Screen Alert"):
            st.markdown("**Copy formatted institutional Telegram alert:**")
            st.code(tg_alert, language="text")

    with e_csv:
        csv_data = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name=f"down_day_resilience_{selected_date}_{market_code}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # -------------------------------------------------------------
    # 5. RESILIENT LEADERS DATA TERMINAL TABLE
    # -------------------------------------------------------------
    st.markdown(f"### 👑 Resilient Leaders on {selected_date} ({len(df_display)} Stocks Found)")

    table_cols = [
        'Clean_Ticker', 'Archetype', 'Resilience_Score', 'Return_1D', 'Dist_52W_High',
        'Volume_Mult', 'Close', 'Turnover_Cr', 'Industry', 'Ind_Grade', 'Sister_Role'
    ]
    disp_df = df_display[[c for c in table_cols if c in df_display.columns]].copy()

    col_config = {
        "Clean_Ticker": st.column_config.TextColumn("Symbol", width=95),
        "Archetype": st.column_config.TextColumn("Resilience Archetype", width=220),
        "Resilience_Score": st.column_config.ProgressColumn("Alpha Score", min_value=0, max_value=100, format="%.1f", width=110),
        "Return_1D": st.column_config.NumberColumn("1D Gain %", format="%+.2f%%", width=95),
        "Dist_52W_High": st.column_config.NumberColumn("Dist 52W High", format="%+.1f%%", width=110, help="Distance from 52-week high as of this date"),
        "Volume_Mult": st.column_config.NumberColumn("Volume (x ADTV)", format="%.2fx", width=120, help="Volume relative to 20-day Average Daily Volume"),
        "Close": st.column_config.NumberColumn(f"Close ({curr_symbol})", format=f"{curr_symbol}%.2f", width=110),
        "Turnover_Cr": st.column_config.NumberColumn(f"Turnover ({curr_symbol}M/Cr)", format="%.1f", width=110),
        "Industry": st.column_config.TextColumn("Industry Group", width=180),
        "Ind_Grade": st.column_config.TextColumn("IBD Grade", width=90),
        "Sister_Role": st.column_config.TextColumn("Sister Role", width=150)
    }

    st.dataframe(
        disp_df,
        column_config=col_config,
        use_container_width=True,
        hide_index=True,
        height=550
    )

if __name__ == "__main__":
    main()
