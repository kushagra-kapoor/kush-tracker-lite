# Kush Tracker - Unified Beach Ball & Down-Day Resilience Terminal
# Institutional Workstation for Submerged Accumulation, Point-in-Time Washouts, and Multi-Horizon Alpha

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st

# Path setup
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from styles import load_css
from components import render_header, render_metric_card
import database
if not hasattr(database, 'save_beach_ball_events') or not hasattr(database, 'save_down_day_resilience_events'):
    import importlib
    importlib.reload(database)

from database import get_connection, get_latest_beach_ball_sync_date
from beach_ball_engine import get_enriched_beach_ball_radar, get_low_breadth_dates
import down_day_engine as dde

# =========================================================================
# 1. PAGE CONFIGURATION & STYLES
# =========================================================================
try:
    st.set_page_config(
        page_title="Beach Ball & Down-Day Resilience Terminal | Kush Tracker",
        page_icon="🏖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    pass

load_css()

st.markdown("""
<style>
.block-container {
    padding-top: 2.75rem !important;
}
.bb-hud-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85));
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 10px 28px -6px rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}
.bb-hud-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 34px -6px rgba(56, 189, 248, 0.2);
}
.bb-hud-title {
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94a3b8;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.bb-hud-val {
    font-size: 1.85rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    margin: 0;
    letter-spacing: -0.5px;
}
.bb-hud-sub {
    font-size: 0.78rem;
    color: #cbd5e1;
    margin-top: 6px;
    line-height: 1.4;
}
@keyframes pulseGlowRed {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
    70% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}
@keyframes pulseGlowGreen {
    0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}
.regime-dot-submerged {
    width: 9px;
    height: 9px;
    background: #ef4444;
    border-radius: 50%;
    display: inline-block;
    animation: pulseGlowRed 1.8s infinite;
}
.regime-dot-normal {
    width: 9px;
    height: 9px;
    background: #10b981;
    border-radius: 50%;
    display: inline-block;
    animation: pulseGlowGreen 1.8s infinite;
}
.breadth-meter-bg {
    width: 100%;
    height: 7px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    margin-top: 8px;
    position: relative;
    overflow: hidden;
}
.breadth-meter-fill {
    height: 100%;
    border-radius: 4px;
}
.breadth-threshold-line {
    position: absolute;
    left: 35%;
    top: 0;
    bottom: 0;
    width: 2px;
    background: #eab308;
    z-index: 2;
}
.tab-question-banner {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(30, 41, 59, 0.78) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 18px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(8px);
}
.tqb-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}
.tqb-badge {
    font-size: 0.68rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    padding: 2px 8px;
    border-radius: 4px;
}
.tqb-question {
    font-size: 1.02rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.45;
    margin: 6px 0 8px 0;
}
.tqb-meta {
    font-size: 0.80rem;
    color: #94a3b8;
    line-height: 1.45;
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    padding-top: 6px;
}
.tqb-meta span b {
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# =========================================================================
# 2. HERO HEADER & MARKET UNIVERSE SELECTION
# =========================================================================
h_left, h_right = st.columns([3.2, 1.3])

with h_left:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-top: 6px; margin-bottom: 6px;">
        <h2 style="margin: 0; padding-top: 6px; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.5px; line-height: 1.35;">
            🏖️ Beach Ball & Down-Day Resilience Terminal
        </h2>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Institutional Submerged Accumulation, Point-in-Time Washout Alpha, and Multi-Horizon Counter-Trend Divergence Engine")

with h_right:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    selected_market = st.radio(
        "Select Market Universe:",
        ["🇮🇳 India (NSE)", "🇺🇸 US (Nasdaq & NYSE)"],
        horizontal=True,
        label_visibility="collapsed",
        key="bb_unified_market_select"
    )

is_us = "US" in selected_market
mkt_tag = 'US' if is_us else 'INDIA'
mkt_code = 'US' if is_us else 'IN'
cs = "$" if is_us else "₹"
tv_prefix = "" if is_us else "NSE:"

# =========================================================================
# 3. LIVE MARKET BREADTH & MASTER TELEMETRY HUD
# =========================================================================
conn = get_connection()
c = conn.cursor()
c.execute("""
    SELECT date, above_50_pct, above_20_pct, above_200_pct 
    FROM market_breadth_daily 
    WHERE market = ? 
    ORDER BY date DESC LIMIT 1
""", (mkt_code,))
row_b = c.fetchone()
conn.close()

cur_date = row_b[0] if row_b else "Latest"
cur_50_breadth = float(row_b[1]) if row_b and row_b[1] is not None else 50.0
cur_20_breadth = float(row_b[2]) if row_b and row_b[2] is not None else 50.0
cur_200_breadth = float(row_b[3]) if row_b and row_b[3] is not None else 50.0
is_submerged = (cur_50_breadth < 35.0)

# Pre-fetch Cumulative Radar Data
radar_df = get_enriched_beach_ball_radar(market=mkt_tag, breadth_threshold=35.0, min_power_plays_3m=1)
thrust_badge_map = dde.get_cumulative_thrust_map(market=mkt_tag, lookback_days=90)

# Master 4-Column HUD
hud1, hud2, hud3, hud4 = st.columns(4)

with hud1:
    b_color = "#ef4444" if cur_50_breadth < 25.0 else "#f59e0b" if cur_50_breadth < 35.0 else "#10b981"
    b_label = "🔴 SEVERE DOWNTREND" if cur_50_breadth < 25.0 else "🟡 SUBMERGED REGIME" if cur_50_breadth < 35.0 else "🟢 HEALTHY EXPANSION"
    fill_width = min(100.0, max(0.0, cur_50_breadth))
    st.markdown(f"""
    <div class="bb-hud-card" style="border-left: 5px solid {b_color};">
        <div class="bb-hud-title">
            <span class="{'regime-dot-submerged' if is_submerged else 'regime-dot-normal'}"></span>
            50-SMA Breadth ({mkt_tag})
        </div>
        <div class="bb-hud-val" style="color: {b_color};">{cur_50_breadth:.1f}%</div>
        <div class="breadth-meter-bg">
            <div class="breadth-threshold-line" title="35% Submergence Threshold"></div>
            <div class="breadth-meter-fill" style="width: {fill_width}%; background: {b_color};"></div>
        </div>
        <div class="bb-hud-sub">
            <b>{b_label}</b><br/>
            20-EMA: <b>{cur_20_breadth:.1f}%</b> | 200-SMA: <b>{cur_200_breadth:.1f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with hud2:
    if is_submerged:
        phase_title = "🏖️ ACTIVE SUBMERGENCE"
        phase_color = "#f59e0b"
        phase_sub = "Smart money buying in secret during panic. <b>Prime watchlists for the tide turn.</b>"
    else:
        phase_title = "🚀 TIDE TURN / LAUNCH"
        phase_color = "#10b981"
        phase_sub = "Market breadth above 35%. <b>Strike aggressively on Follow-Through Days.</b>"
    st.markdown(f"""
    <div class="bb-hud-card" style="border-left: 5px solid {phase_color};">
        <div class="bb-hud-title">Cycle Accumulation Phase</div>
        <div class="bb-hud-val" style="color: {phase_color}; font-size: 1.35rem; line-height: 1.2; margin-top: 4px;">{phase_title}</div>
        <div class="bb-hud-sub">{phase_sub}</div>
    </div>
    """, unsafe_allow_html=True)

with hud3:
    total_leaders = len(radar_df) if not radar_df.empty else 0
    apex_count = len(radar_df[radar_df['power_plays_3m'] >= 4]) if not radar_df.empty else 0
    multi_count = len(radar_df[radar_df['power_plays_3m'].between(2, 3)]) if not radar_df.empty else 0
    st.markdown(f"""
    <div class="bb-hud-card" style="border-left: 5px solid #38bdf8;">
        <div class="bb-hud-title">Submerged Leaders Tracked</div>
        <div class="bb-hud-val" style="color: #38bdf8;">{total_leaders}</div>
        <div class="bb-hud-sub">
            <b>{apex_count} Apex Leaders</b> (≥4x Power Plays)<br/>
            <b>{multi_count} Multi-Accumulators</b> (2-3x Power Plays)
        </div>
    </div>
    """, unsafe_allow_html=True)

with hud4:
    if not radar_df.empty:
        top_leader = radar_df.iloc[0]
        t_sym = top_leader['clean_ticker']
        t_p3 = top_leader['power_plays_3m']
        t_p1 = top_leader['power_plays_1m']
        t_ind = str(top_leader.get('industry', 'Broad Market'))[:22]
        top_txt = f"{t_sym} ({t_p3}x in 3M)"
        top_sub = f"⚡ <b>{t_p1}x in last 30d</b> | RS <b>{top_leader['rs_score']}</b><br/>Theme: <i>{t_ind}</i>"
    else:
        top_txt = "Scanning..."
        top_sub = "No power plays recorded yet"
    st.markdown(f"""
    <div class="bb-hud-card" style="border-left: 5px solid #c084fc;">
        <div class="bb-hud-title">#1 Most Accumulated Leader</div>
        <div class="bb-hud-val" style="color: #c084fc; font-size: 1.35rem; line-height: 1.2; margin-top: 4px;">{top_txt}</div>
        <div class="bb-hud-sub">{top_sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

# =========================================================================
# 4. MASTER 3-TAB WORKSTATION
# =========================================================================
tab_session, tab_leaderboard, tab_horizon, tab_playbook = st.tabs([
    "📅 Point-in-Time Session Radar",
    "🏆 Cumulative Accumulation Leaderboard",
    "🔬 Multi-Horizon Correction Analytics (1M / 3M)",
    "📖 Master Trader Playbook"
])

# -------------------------------------------------------------------------
# TAB 1: POINT-IN-TIME SESSION RADAR (SINGLE DOWN-DAY DEEP DIVE)
# -------------------------------------------------------------------------
with tab_session:
    st.markdown("""
    <div class="tab-question-banner" style="border-left: 4px solid #38bdf8;">
        <div class="tqb-header">
            <span class="tqb-badge" style="color: #38bdf8; background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.25);">
                🎯 CORE QUESTION ANSWERED
            </span>
        </div>
        <div class="tqb-question">
            "Today was a red day where 90% of stocks declined. Which 10% of stocks defied gravity, closed green, surged ≥ +4% on heavy volume, or broke out to new 52-week highs?"
        </div>
        <div class="tqb-meta">
            <span>⏱️ <b>When to use:</b> Every market down session at 3:35 PM IST (or 4:05 PM EST) to construct tomorrow's high-conviction focus list.</span>
            <span>⚡ <b>What to do:</b> Select the washout date, inspect the 3 resilience archetypes, verify cumulative accumulation footprint, and export symbols to TradingView or Notion.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📅 Single-Day Washout & Correction Session Analyzer")
    st.caption("Inspect exactly which equities printed Beach Ball thrusts, broke out to 52W highs, or closed green during severe market pullbacks.")

    c1_lb, c1_arch, c1_price = st.columns([1.2, 1.8, 1.0])
    with c1_lb:
        lookback_choice = st.selectbox(
            "Historical Correction Lookback:",
            ["Last 90 Days", "Last 180 Days", "Last 365 Days", "All History (2 Years)"],
            index=1,
            key="pit_lookback_select"
        )
        days_num = 730 if "All" in lookback_choice else (365 if "365" in lookback_choice else (180 if "180" in lookback_choice else 90))

    with c1_arch:
        arch_filter = st.selectbox(
            "Filter Resilience Archetype:",
            [
                "🌟 All Resilient Leaders",
                "🏖️ Beach Ball (VB Power Plays) Only",
                "🚀 New 52W Highs on Down Day Only",
                "⚡ Institutional Accumulation Only",
                "🟢 Green on Red (Relative Anchors)"
            ],
            index=0,
            key="pit_arch_filter"
        )

    with c1_price:
        min_price_pit = st.number_input(
            f"Min Price ({cs}):",
            value=20.0 if not is_us else 5.0,
            step=5.0,
            key="pit_min_price"
        )

    # Query Washout Sessions
    df_washouts = dde.get_historical_market_down_dates(
        market=mkt_tag,
        min_drop_pct=0.75,
        max_breadth_pct=35.0,
        lookback_days=days_num
    )

    if df_washouts.empty:
        st.warning(f"No qualifying market washout sessions found for {selected_market} in the selected lookback.")
    else:
        date_options = df_washouts['date'].tolist()
        def format_date_label(d_str):
            row = df_washouts[df_washouts['date'] == d_str].iloc[0]
            b_pct = row['above_50_pct']
            nnh = int(row['net_new_highs'])
            badge = row['severity_badge']
            adv = int(row.get('advances', 0) or 0)
            dec = int(row.get('declines', 0) or 0)
            if adv == 0 and dec == 0:
                adv, dec = dde.get_session_advance_decline(d_str, market=mkt_tag)
            ad_str = f" • A/D: {adv}/{dec}" if (adv > 0 or dec > 0) else ""
            return f"{d_str}  ·  {badge}  (Breadth: {b_pct:.1f}% > 50SMA • Net Highs: {nnh:+d}{ad_str})"

        selected_date = st.selectbox(
            "Select Market Down Day to Inspect:",
            options=date_options,
            index=0,
            format_func=format_date_label,
            key="pit_date_picker",
            help="Choose any date when market breadth washed out or index plummeted."
        )

        date_meta = df_washouts[df_washouts['date'] == selected_date].iloc[0]

        # Session Telemetry Cards
        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1:
            render_metric_card("Session Date", selected_date, subtitle="Historical Correction")
        with sm2:
            b_val = date_meta['above_50_pct']
            b_class = "red-text" if b_val <= 20.0 else ("orange-text" if b_val <= 35.0 else "blue-text")
            render_metric_card("50-SMA Market Breadth", f"{b_val:.1f}%", subtitle="Washed Out Regime", color_class=b_class)
        with sm3:
            nnh_val = int(date_meta['net_new_highs'])
            nnh_class = "red-text" if nnh_val < 0 else "green-text"
            render_metric_card("Net 52W Highs/Lows", f"{nnh_val:+d}", subtitle="New Highs vs New Lows", color_class=nnh_class)
        with sm4:
            adv = int(date_meta.get('advances', 0) or 0)
            dec = int(date_meta.get('declines', 0) or 0)
            if adv == 0 and dec == 0:
                adv, dec = dde.get_session_advance_decline(selected_date, market=mkt_tag)
            ratio_str = f"{adv/max(1,dec):.2f}" if (adv > 0 or dec > 0) else "—"
            render_metric_card("Advance / Decline", f"{adv} / {dec}", subtitle=f"Ratio: {ratio_str}")

        # Fetch Resilient Leaders for Date
        df_resilient = dde.compute_down_day_resilience(
            target_date_str=selected_date,
            market=mkt_tag,
            min_price=min_price_pit,
            max_dist_high=15.0
        )

        if df_resilient.empty:
            st.info(f"ℹ️ No resilient stocks matched the criteria on {selected_date}.")
        else:
            # Archetype Filter
            if "Beach Ball" in arch_filter:
                df_disp = df_resilient[df_resilient['Archetype'].str.contains("Beach Ball")].copy()
            elif "New 52W High" in arch_filter:
                df_disp = df_resilient[df_resilient['Archetype'].str.contains("New 52W High")].copy()
            elif "Institutional Accumulation" in arch_filter:
                df_disp = df_resilient[df_resilient['Archetype'].str.contains("Institutional Accumulation")].copy()
            elif "Green on Red" in arch_filter:
                df_disp = df_resilient[df_resilient['Archetype'].str.contains("Green on Red")].copy()
            else:
                df_disp = df_resilient.copy()

            # Enrich with Cumulative Power Play Badges
            df_disp['Cumulative_Accumulation'] = df_disp['Clean_Ticker'].apply(
                lambda t: thrust_badge_map.get(t, {}).get('badge', '1x (Single Session)')
            )

            # Export Bar
            st.markdown("---")
            e_tv, e_notion, e_tg, e_csv = st.columns([1.3, 1.3, 1.3, 1.0])
            tv_string = dde.format_tradingview_watchlist(df_disp['Clean_Ticker'].head(30).tolist(), market=mkt_tag)
            notion_table = dde.format_notion_markdown_table(df_disp, selected_date)
            tg_alert = dde.format_telegram_alert(selected_date, date_meta['above_50_pct'], int(date_meta['net_new_highs']), df_disp, market=mkt_tag)

            with e_tv:
                with st.popover(f"📋 TradingView Watchlist ({len(df_disp)})"):
                    st.code(tv_string, language="text")
            with e_notion:
                with st.popover("📓 Copy for Notion"):
                    st.code(notion_table, language="markdown")
            with e_tg:
                with st.popover("📱 Telegram Lock-Screen Alert"):
                    st.code(tg_alert, language="text")
            with e_csv:
                st.download_button(
                    label="📊 Download CSV",
                    data=df_disp.to_csv(index=False).encode('utf-8'),
                    file_name=f"down_day_resilience_{selected_date}_{mkt_tag}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            st.markdown(f"##### 👑 Resilient Leaders on {selected_date} ({len(df_disp)} Stocks)")
            table_cols = [
                'Clean_Ticker', 'Archetype', 'Cumulative_Accumulation', 'Resilience_Score',
                'Return_1D', 'Dist_52W_High', 'Volume_Mult', 'Close', 'Turnover_Cr',
                'Industry', 'Ind_Grade', 'Sister_Role'
            ]
            disp_view = df_disp[[c for c in table_cols if c in df_disp.columns]].copy()

            st.dataframe(
                disp_view,
                column_config={
                    "Clean_Ticker": st.column_config.TextColumn("Symbol", width=95),
                    "Archetype": st.column_config.TextColumn("Resilience Archetype", width=200),
                    "Cumulative_Accumulation": st.column_config.TextColumn("Cumulative Footprint", width=160),
                    "Resilience_Score": st.column_config.ProgressColumn("Alpha Score", min_value=0, max_value=100, format="%.1f", width=105),
                    "Return_1D": st.column_config.NumberColumn("1D Gain %", format="%+.2f%%", width=95),
                    "Dist_52W_High": st.column_config.NumberColumn("Dist 52W High", format="%+.1f%%", width=110),
                    "Volume_Mult": st.column_config.NumberColumn("Vol Ratio (x ADTV)", format="%.2fx", width=110),
                    "Close": st.column_config.NumberColumn(f"Close ({cs})", format=f"{cs}%.2f", width=105),
                    "Turnover_Cr": st.column_config.NumberColumn(f"Turnover ({cs}M/Cr)", format="%.1f", width=105),
                    "Industry": st.column_config.TextColumn("Industry Group", width=180),
                    "Ind_Grade": st.column_config.TextColumn("IBD Grade", width=90),
                    "Sister_Role": st.column_config.TextColumn("Sister Role", width=140)
                },
                use_container_width=True,
                hide_index=True,
                height=520
            )

# -------------------------------------------------------------------------
# TAB 2: CUMULATIVE BEACH BALL LEADERBOARD (MULTI-DAY ACCUMULATION)
# -------------------------------------------------------------------------
with tab_leaderboard:
    st.markdown("""
    <div class="tab-question-banner" style="border-left: 4px solid #c084fc;">
        <div class="tqb-header">
            <span class="tqb-badge" style="color: #c084fc; background: rgba(192, 132, 252, 0.12); border: 1px solid rgba(192, 132, 252, 0.25);">
                🎯 CORE QUESTION ANSWERED
            </span>
        </div>
        <div class="tqb-question">
            "Across all down days over the past 1 to 3 months, which stocks have institutions accumulated multiple times (2x, 3x, 4x+)? Who has the tightest coiled spring ready to launch on the market follow-through day?"
        </div>
        <div class="tqb-meta">
            <span>⏱️ <b>When to use:</b> Weekend review or after multi-week market pullbacks to isolate recurring institutional footprints and group contagion.</span>
            <span>⚡ <b>What to do:</b> Filter by Multi-Accumulators (≥2x) or Apex Power (≥4x), inspect Industry Group Contagion clusters, and set pivot buy alerts for the next confirmed FTD.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    p_col1, p_col2 = st.columns([2.8, 1.2])
    with p_col1:
        preset_choice = st.pills(
            "Quick Filter Presets:",
            options=[
                "⭐ All Submerged (≥1x)",
                "🔥 Multi-Accumulators (≥2x)",
                "⚡ Recent Inflow (1M ≥ 1x)",
                "👑 Apex Power (≥4x)",
                "🎯 Near 52W High (≤15% Gap)",
                "🔵 RS Blue-Chips (RS ≥ 85)"
            ],
            default="🔥 Multi-Accumulators (≥2x)",
            key="bb_preset_pills"
        )
    with p_col2:
        search_query = st.text_input(
            "🔍 Search Ticker / Industry:",
            placeholder="e.g. DPABHUSHAN, Software...",
            key="bb_search_input"
        )

    if preset_choice == "👑 Apex Power (≥4x)":
        def_min_3m, def_min_1m, def_min_rs, def_max_gap = 4, 0, 70, 25
    elif preset_choice == "⚡ Recent Inflow (1M ≥ 1x)":
        def_min_3m, def_min_1m, def_min_rs, def_max_gap = 1, 1, 65, 25
    elif preset_choice == "🎯 Near 52W High (≤15% Gap)":
        def_min_3m, def_min_1m, def_min_rs, def_max_gap = 1, 0, 70, 15
    elif preset_choice == "🔵 RS Blue-Chips (RS ≥ 85)":
        def_min_3m, def_min_1m, def_min_rs, def_max_gap = 1, 0, 85, 25
    elif preset_choice == "⭐ All Submerged (≥1x)":
        def_min_3m, def_min_1m, def_min_rs, def_max_gap = 1, 0, 50, 30
    else:
        def_min_3m, def_min_1m, def_min_rs, def_max_gap = 2, 0, 60, 25

    with st.expander("⚙️ Fine-Tune Radar Thresholds & Rescan", expanded=False):
        f_c1, f_c2, f_c3, f_c4 = st.columns(4)
        with f_c1:
            min_3m_filter = st.slider("Min 3M Power Plays:", 1, 6, def_min_3m, 1, key="f_3m_slider")
        with f_c2:
            min_1m_filter = st.slider("Min 1M Power Plays:", 0, 4, def_min_1m, 1, key="f_1m_slider")
        with f_c3:
            min_rs_filter = st.slider("Min Relative Strength (RS):", 50, 99, def_min_rs, 5, key="f_rs_slider")
        with f_c4:
            max_dist_52w = st.slider("Max Distance to 52W High (%):", 2, 35, def_max_gap, 1, key="f_dist_slider")

        sync_btn_col1, sync_btn_col2 = st.columns([3, 1])
        with sync_btn_col2:
            if st.button("🔄 Rescan All Thrusts", help="Forces incremental rescan for low-breadth power plays"):
                with st.spinner("Scanning universe for low-breadth power plays..."):
                    from beach_ball_engine import sync_beach_ball_events
                    n_new = sync_beach_ball_events(market=mkt_tag, force_rescan=True)
                    st.success(f"Synced {n_new} power plays!")
                    st.rerun()

    # Filter Radar DataFrame
    filtered_df = radar_df.copy()
    if not filtered_df.empty:
        filtered_df = filtered_df[filtered_df['power_plays_3m'] >= min_3m_filter]
        if min_1m_filter > 0:
            filtered_df = filtered_df[filtered_df['power_plays_1m'] >= min_1m_filter]
        filtered_df = filtered_df[filtered_df['rs_score'] >= min_rs_filter]
        filtered_df = filtered_df[filtered_df['dist_52w_high'] <= max_dist_52w]
        if search_query.strip():
            q = search_query.strip().lower()
            filtered_df = filtered_df[
                filtered_df['clean_ticker'].str.lower().str.contains(q) |
                filtered_df['industry'].str.lower().str.contains(q)
            ]

    st.markdown(f"##### 🏆 Cumulative Submerged Leaderboard ({len(filtered_df)} Stocks)")
    if not filtered_df.empty:
        # 1-Click TradingView Watchlist
        clean_tickers_list = filtered_df['clean_ticker'].tolist()
        tv_watchlist_str = ", ".join([f"{tv_prefix}{t}" for t in clean_tickers_list])
        with st.popover(f"📋 1-Click TradingView Import ({len(clean_tickers_list)} Symbols)"):
            st.code(tv_watchlist_str, language="text")

        # Table Display
        disp_rows = []
        for _, row in filtered_df.iterrows():
            disp_rows.append({
                "Rank": int(row['rank']),
                "Ticker": row['clean_ticker'],
                "Industry / Theme": str(row.get('industry', 'Broad Market'))[:28],
                "Accumulation Tier": row['accumulation_tier'],
                "3M Power Plays": f"🔥 {row['power_plays_3m']}x",
                "1M Power Plays": f"⚡ {row['power_plays_1m']}x",
                "Latest Thrust": f"+{row['max_gain_pct']:.1f}% ({row['last_power_play_date']})",
                "Vol Ratio": f"{row['avg_vol_multiple']:.1f}x",
                "RS Rating": round(float(row['rs_score']), 1),
                "52W High Gap": f"-{row['dist_52w_high']:.1f}%",
                "Price": f"{cs}{row['current_price']:.1f}",
                "Setup Status": row['setup_status']
            })

        st.dataframe(
            pd.DataFrame(disp_rows),
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Industry / Theme": st.column_config.TextColumn("Industry / Theme", width="medium"),
                "Accumulation Tier": st.column_config.TextColumn("Tier", width="small"),
                "3M Power Plays": st.column_config.TextColumn("3M Power Plays", width="small"),
                "1M Power Plays": st.column_config.TextColumn("1M Power Plays", width="small"),
                "Latest Thrust": st.column_config.TextColumn("Latest Thrust", width="medium"),
                "Vol Ratio": st.column_config.TextColumn("Avg Vol Ratio", width="small"),
                "RS Rating": st.column_config.ProgressColumn("RS Rating", min_value=0, max_value=99, format="%.1f", width="small"),
                "52W High Gap": st.column_config.TextColumn("52W Gap", width="small"),
                "Price": st.column_config.TextColumn("Price", width="small"),
                "Setup Status": st.column_config.TextColumn("Setup Status", width="medium")
            },
            use_container_width=True,
            hide_index=True,
            height=480
        )

        # Industry Clustering Bar Chart
        if 'industry' in filtered_df.columns:
            ind_counts = filtered_df['industry'].value_counts().head(10).reset_index()
            ind_counts.columns = ['Industry Group', 'Submerged Count']
            fig_ind = px.bar(
                ind_counts,
                x='Submerged Count',
                y='Industry Group',
                orientation='h',
                title="🏭 Industry Group Contagion (Cluster of Submerged Leaders)",
                color='Submerged Count',
                color_continuous_scale='Tealgrn'
            )
            fig_ind.update_layout(yaxis={'categoryorder': 'total ascending'}, height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_ind, use_container_width=True)
    else:
        st.info("⚠️ No leaders match the filter criteria. Lower the sliders to expand results.")

# -------------------------------------------------------------------------
# TAB 3: MULTI-HORIZON CORRECTION & RELATIVE ALPHA ANALYTICS (1M / 3M)
# -------------------------------------------------------------------------
with tab_horizon:
    st.markdown("""
    <div class="tab-question-banner" style="border-left: 4px solid #10b981;">
        <div class="tqb-header">
            <span class="tqb-badge" style="color: #10b981; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.25);">
                🎯 CORE QUESTION ANSWERED
            </span>
        </div>
        <div class="tqb-question">
            "Over the last 1 Month (21D) or 3 Months (63D), the broad market fell. Which stocks completely decoupled, generated positive absolute returns, and delivered massive Alpha Delta (α)?"
        </div>
        <div class="tqb-meta">
            <span>⏱️ <b>When to use:</b> During slow, choppy, or correcting markets to verify institutional sponsorship and true regime-decoupling market generals.</span>
            <span>⚡ <b>What to do:</b> Compare Benchmark Max Drawdown vs stock gain, prioritize high Down-Day Win Rates (>70%), and spot decoupled theme clusters.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🔬 Multi-Horizon Correction & Counter-Trend Divergence Matrix")
    st.caption("When the broad market is bad over 1 Month (21D) or 3 Months (63D), uncover the true leaders generating positive absolute returns and massive Alpha Delta.")

    hz_col1, hz_col2 = st.columns([1.5, 1.0])
    with hz_col1:
        horizon_preset = st.radio(
            "Select Correction Lookback Horizon:",
            ["⏱️ Last 1 Month (21 Trading Days)", "⏱️ Last 3 Months (63 Trading Days)", "⏱️ Last 6 Months (126 Trading Days)"],
            horizontal=True,
            index=0,
            key="horizon_lookback_radio"
        )
        h_days = 63 if "3 Months" in horizon_preset else (126 if "6 Months" in horizon_preset else 21)

    with hz_col2:
        min_price_hz = st.number_input(f"Min Price ({cs}):", value=20.0 if not is_us else 5.0, step=5.0, key="hz_min_price")

    with st.spinner(f"Computing multi-horizon alpha divergence across {selected_market}..."):
        df_divergent, bench_summary, df_group_alpha = dde.compute_multi_horizon_correction_analytics(
            market=mkt_tag,
            horizon_days=h_days,
            min_price=min_price_hz
        )

    if bench_summary:
        # Macro Benchmark Context HUD
        bm1, bm2, bm3, bm4 = st.columns(4)
        with bm1:
            b_ret = bench_summary['bench_return_pct']
            b_ret_color = "red-text" if b_ret < 0 else "green-text"
            render_metric_card("Benchmark Return", f"{b_ret:+.2f}%", subtitle=bench_summary['bench_name'], color_class=b_ret_color)
        with bm2:
            render_metric_card("Benchmark Max Drawdown", f"{bench_summary['bench_max_dd_pct']:.2f}%", subtitle="Worst Peak-to-Trough", color_class="red-text")
        with bm3:
            render_metric_card("Market Down Days", f"{bench_summary['total_down_days']} Days", subtitle=f"Out of {bench_summary['horizon_bars']} trading bars", color_class="orange-text")
        with bm4:
            pos_count = bench_summary['positive_absolute_count']
            tot_count = bench_summary['total_universe']
            render_metric_card("Decoupled Green Stocks", f"{pos_count} / {tot_count}", subtitle=f"{pos_count/max(1,tot_count)*100:.1f}% positive absolute return", color_class="green-text")

    if df_divergent.empty:
        st.info("ℹ️ No divergent leaders found for this horizon.")
    else:
        # Actionable Exports
        st.markdown("---")
        hz_tv, hz_notion, hz_csv = st.columns([1.5, 1.5, 1.0])
        tv_div_str = dde.format_tradingview_watchlist(df_divergent['Clean_Ticker'].head(30).tolist(), market=mkt_tag)
        
        with hz_tv:
            with st.popover(f"📋 TradingView Divergent Watchlist ({len(df_divergent)})"):
                st.code(tv_div_str, language="text")

        with hz_notion:
            with st.popover("📓 Copy for Notion (Divergence Table)"):
                lines = [
                    f"### 🔬 Multi-Horizon Counter-Trend Divergence ({horizon_preset})",
                    "| Symbol | Window Return % | Alpha Delta % | Down-Day Win Rate | Beach Ball Thrusts | Industry | Sister Role |",
                    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
                ]
                for _, r in df_divergent.head(25).iterrows():
                    lines.append(f"| **{r['Clean_Ticker']}** | {r['Return_Window']:+.1f}% | {r['Alpha_Delta']:+.1f}% | {r['Down_Day_Win_Rate']:.1f}% | {r['BB_Badge']} | {r['Industry']} | {r['Sister_Role']} |")
                st.code("\n".join(lines), language="markdown")

        with hz_csv:
            st.download_button(
                label="📊 Download CSV",
                data=df_divergent.to_csv(index=False).encode('utf-8'),
                file_name=f"multi_horizon_alpha_{mkt_tag}_{h_days}d.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown(f"##### 🚀 Counter-Trend Alpha Leaders ({len(df_divergent)} Stocks Decoupled from Benchmark)")
        
        div_table_cols = [
            'Clean_Ticker', 'Return_Window', 'Alpha_Delta', 'Down_Day_Win_Rate',
            'BB_Badge', 'Dist_52W_High', 'Volume_Mult', 'Close', 'Turnover_Cr',
            'Industry', 'Ind_Grade', 'Sister_Role'
        ]
        disp_div = df_divergent[[c for c in div_table_cols if c in df_divergent.columns]].copy()

        st.dataframe(
            disp_div,
            column_config={
                "Clean_Ticker": st.column_config.TextColumn("Symbol", width=95),
                "Return_Window": st.column_config.NumberColumn("Window Gain %", format="%+.1f%%", width=110),
                "Alpha_Delta": st.column_config.NumberColumn("Alpha Delta (α)", format="%+.1f%%", width=115, help="Stock Return minus Benchmark Return"),
                "Down_Day_Win_Rate": st.column_config.ProgressColumn("Down-Day Win Rate", min_value=0, max_value=100, format="%.1f%%", width=130, help="% of days stock closed green when market was red"),
                "BB_Badge": st.column_config.TextColumn("Beach Ball Footprint", width=140),
                "Dist_52W_High": st.column_config.NumberColumn("Dist 52W High", format="%+.1f%%", width=105),
                "Volume_Mult": st.column_config.NumberColumn("Vol Ratio", format="%.2fx", width=95),
                "Close": st.column_config.NumberColumn(f"Close ({cs})", format=f"{cs}%.2f", width=100),
                "Turnover_Cr": st.column_config.NumberColumn(f"Turnover ({cs}M/Cr)", format="%.1f", width=105),
                "Industry": st.column_config.TextColumn("Industry Group", width=180),
                "Ind_Grade": st.column_config.TextColumn("IBD Grade", width=90),
                "Sister_Role": st.column_config.TextColumn("Sister Role", width=140)
            },
            use_container_width=True,
            hide_index=True,
            height=460
        )

        # Group-Level Alpha Decoupling
        if not df_group_alpha.empty:
            st.markdown("##### 🌊 Decoupled Industry Groups (Top Alpha Clusters)")
            fig_grp = px.bar(
                df_group_alpha.head(10),
                x='Avg_Alpha',
                y='Industry',
                orientation='h',
                title=f"Decoupled Themes: Avg Excess Alpha (α) during {horizon_preset}",
                color='Avg_Alpha',
                color_continuous_scale='Viridis',
                hover_data=['Leader_Count', 'Top_Stock', 'Top_Stock_Return']
            )
            fig_grp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_grp, use_container_width=True)

# -------------------------------------------------------------------------
# TAB 4: MASTER TRADER PLAYBOOK
# -------------------------------------------------------------------------
with tab_playbook:
    st.markdown("""
    <div class="tab-question-banner" style="border-left: 4px solid #eab308;">
        <div class="tqb-header">
            <span class="tqb-badge" style="color: #eab308; background: rgba(234, 179, 8, 0.12); border: 1px solid rgba(234, 179, 8, 0.25);">
                🎯 CORE QUESTION ANSWERED
            </span>
        </div>
        <div class="tqb-question">
            "What are the strict William J. O'Neil, David Ryan, and Mark Minervini execution rules for trading Beach Balls and striking on Follow-Through Days?"
        </div>
        <div class="tqb-meta">
            <span>⏱️ <b>When to use:</b> Before pulling the trigger on trades, sizing exposure, or placing stops during market turnarounds.</span>
            <span>⚡ <b>What to do:</b> Memorize Rule 1 (Never chase on red days; build the watchlist), Rule 2 (Execute on FTD from sound bases), and Rule 3 (Follow 3+ stock group contagion).</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(r"""
    ### 📖 The Law of Down-Day Resilience & Submerged Leaders
    > *"During general market declines of 10% to 20%, 90% of all stocks decline with the averages. **The 10% that resist the decline, hold their ground, or actually advance to new highs are almost always the huge winners of the next market cycle.** Smart money uses market panic to quietly accumulate their highest-conviction ideas without pushing the price through the roof."*  
    > — **William J. O'Neil**, *How to Make Money in Stocks*

    ---

    ### The 3 Quantitative Resilience Archetypes:
    1. 🏖️ **Beach Ball Accumulation (VB Power Play)**:
       - **Criteria**: Daily Return $\ge +3.8\%$ on Volume $\ge 1.0\times$ 20D ADTV during severe breadth compression ($< 35\%$ 50-SMA breadth).
       - **Physics**: Systemic market selling holds the stock down like hands holding a beach ball underwater. When selling abates, it violently explodes upward.
    2. 🚀 **New 52-Week High on Down Day**:
       - **Criteria**: Stock closes within $0.5\%$ of its absolute 52-week peak while the broad market is tumbling.
       - **Physics**: True monster leaders absorb all supply and break into blue-sky territory while the averages roll over.
    3. 🟢 **Green on Red (Relative Strength Anchor)**:
       - **Criteria**: Stock closes firmly green ($> 0\%$) and sits within $15\%$ of its 52-week high.
       - **Physics**: Coiled spring leaders that refuse to give up ground during market panic.

    ---

    ### Master Execution Rules (David Ryan & Mark Minervini):
    - **Rule 1: Never buy on down days themselves.** Use market washouts to build your focus list. Let smart money do the heavy lifting of absorbing supply.
    - **Rule 2: Strike on the Follow-Through Day (FTD).** When the NIFTY or S&P confirms an institutional accumulation follow-through day (Day 4+ rally of $\ge 1.5\%$ on heavy volume), immediately execute buy stops on the top Beach Ball accumulators breaking out from tight bases.
    - **Rule 3: Follow Industry Contagion.** When 3 or 4 stocks in the same niche industry group show repeated Beach Ball power plays, that entire group will lead the next bull market.
    """)

# Cleanup & Footer
st.markdown("---")
st.caption("Kush Tracker • Institutional Beach Ball & Down-Day Resilience Workstation • Zero-Defect CANSLIM Engine")
