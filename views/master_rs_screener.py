# Kush Tracker - Master RS & Structure Screener (2,650+ All-Cap & 750 Core)
# Institutional Workstation for Relative Strength, 21-EMA / 50-SMA Adherence, Ameet Rai Persistence & ADTV Liquidity

import os
import sys
import pandas as pd
import streamlit as st

# Ensure root directory is on sys.path
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from styles import load_css
from breadth_engine import compute_leader_21ema_table

load_css()

st.markdown("""
<style>
.block-container {
    padding-top: 1.4rem !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================================
# 1. TOP CONTROL BAR: MARKET, UNIVERSE DOMAIN & LEADER POOL DEPTH
# =========================================================================
ctrl_top1, ctrl_top2, ctrl_top3 = st.columns([0.9, 1.55, 1.55])
with ctrl_top1:
    market_choice = st.radio(
        "Exchange Market:",
        ["🇮🇳 India (NSE)", "🇺🇸 US (S&P 500 / Nasdaq)"],
        index=0,
        horizontal=True,
        key="master_rs_mkt_choice"
    )
    is_us = "US" in market_choice
    mkt_code = "US" if is_us else "IN"
    curr_sym = "$" if is_us else "₹"
    adtv_unit = "$M" if is_us else "₹ Cr"

with ctrl_top2:
    if is_us:
        us_sub1, us_sub2 = st.columns([1.3, 1.0])
        with us_sub1:
            univ_choice = st.radio(
                "Universe Scope:",
                ["🌐 Full US Equities"],
                index=0,
                horizontal=True,
                key="master_rs_univ_us",
            )
            univ_scope = "ALL"
        with us_sub2:
            us_bench_choice = st.radio(
                "US Benchmark:",
                ["^IXIC (Nasdaq)", "^GSPC (S&P 500)"],
                index=0,
                horizontal=True,
                key="master_rs_us_benchmark",
                help="Aligns US RS Line & Rai RS Persistence against Nasdaq Composite (^IXIC, matching app.py US Cockpit) or S&P 500 (^GSPC).",
            )
            bench_sym = "^IXIC" if "IXIC" in us_bench_choice else "^GSPC"
    else:
        univ_choice = st.radio(
            "Universe Scope:",
            ["🌐 Full 2,650+ All-Cap Universe (Default)", "🏛️ Top 750 Core (Nifty Total Market)"],
            index=(0 if os.path.exists("historical_prices_matrix.pkl") else 1),
            horizontal=True,
            key="master_rs_univ_in",
            help="Full 2,650+ scans historical_prices_matrix.pkl across all active NSE equities; 750 Core restricts strictly to official NIFTY Total Market constituents."
        )
        univ_scope = "ALL" if "2,650+" in univ_choice else "750"
        bench_sym = None

with ctrl_top3:
    pool_depth_choice = st.radio(
        "Momentum Leader Cohort Cap:",
        ["🏆 Top 100 RS Leaders", "🔥 Top 250 RS Leaders", "🌐 All Qualifying Stocks"],
        index=0,
        horizontal=True,
        key="master_rs_pool_depth",
        help="Applied AFTER the Pre-Ranking ADTV Liquidity Gate so you always get a full cohort of liquid leaders."
    )
    if "100" in pool_depth_choice:
        cohort_cap = 100
    elif "250" in pool_depth_choice:
        cohort_cap = 250
    else:
        cohort_cap = 3000

# =========================================================================
# 2. LOAD CACHED CANDIDATE POOL & APPLY PRE-RANKING ADTV GATE
# =========================================================================
with st.spinner(f"Synchronizing {'US' if is_us else ('2,650+ All-Cap' if univ_scope == 'ALL' else 'Nifty 750 Core')} RS & Structure Matrix..."):
    raw_pool_df = compute_leader_21ema_table(
        market=mkt_code,
        top_n=5000,
        universe_scope=univ_scope,
        min_adtv_cr=0.10 if not is_us else 1.0,
        benchmark_symbol=bench_sym,
    )

# =========================================================================
# 3. COMPACT WORKBENCH HEADER + LEGEND POPOVER
# =========================================================================
univ_badge_txt = f"US EQUITIES ({bench_sym})" if is_us else ("2,650+ ALL-CAP" if univ_scope == "ALL" else "NIFTY 750 CORE")
hdr_col, leg_btn_col = st.columns([5.2, 1.0])
with hdr_col:
    st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(23, 37, 84, 0.88)); border: 1px solid rgba(56, 189, 248, 0.30); border-left: 5px solid #38bdf8; border-radius: 10px; padding: 10px 16px; margin-top: 4px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
        <span style="font-size: 1.15rem;">🏛️</span>
        <span style="color: #f8fafc; font-size: 1.02rem; font-weight: 800; letter-spacing: 0.2px;">Master RS &amp; Structure Workbench</span>
        <span style="background: rgba(56, 189, 248, 0.14); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.32); padding: 2px 9px; border-radius: 999px; font-size: 0.68rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;">{univ_badge_txt}</span>
        <span style="color: #94a3b8; font-size: 0.74rem;">20% 1W + 40% 1M + 40% 3M • ADTV &amp; Circuit-Lock Scrubbed</span>
    </div>
    <span style="background: rgba(16, 185, 129, 0.14); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.38); padding: 3px 11px; border-radius: 7px; font-size: 0.76rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; white-space: nowrap;">
        ● {len(raw_pool_df):,} QUALIFIED
    </span>
</div>
""", unsafe_allow_html=True)

if raw_pool_df.empty:
    st.warning("⚠️ Price matrix unavailable or empty for the selected universe.")
    st.stop()

# =========================================================================
# 3b. INSTITUTIONAL EXECUTION LEGEND (POPOVER WHEN AVAILABLE, EXPANDER FALLBACK)
# =========================================================================
def _render_master_rs_legend():
    leg_c1, leg_c2 = st.columns(2)
    with leg_c1:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 12px; height: 100%;">
            <div style="color: #38bdf8; font-size: 0.84rem; font-weight: 800; margin-bottom: 6px;">🌱 STAGE 2 LIFECYCLE (vs 200 SMA)</div>
            <div style="font-size: 0.76rem; color: #cbd5e1; line-height: 1.45;">
                Stan Weinstein &amp; Mark Minervini Stage Analysis:<br><br>
                <b style="color: #34d399;">🌱 Fresh S2 (0% to +60%):</b> Early Base 1/2 breakout; best asymmetric R:R.<br><br>
                <b style="color: #f59e0b;">🔥 Adv S2 (+60% to +100%):</b> Base 3/4 advancing uptrend; trail stops tightly.<br><br>
                <b style="color: #f87171;">⚠️ Climax Ext (&gt; +100%):</b> Stretched above 200 SMA; high pullback risk.<br><br>
                <b style="color: #94a3b8;">⚠️ S4 Below 200D:</b> Stage 4 downtrend under overhead supply.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with leg_c2:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(168, 85, 247, 0.25); border-radius: 8px; padding: 12px; height: 100%;">
            <div style="color: #c084fc; font-size: 0.84rem; font-weight: 800; margin-bottom: 6px;">🏛️ SUPPLY &amp; CAP TIERS (CANSLIM 'S')</div>
            <div style="font-size: 0.76rem; color: #cbd5e1; line-height: 1.45;">
                William O'Neil Supply &amp; Float Principle:<br><br>
                <b style="color: #c084fc;">🚀 Micro &amp; Small Caps:</b> India &lt; ₹10k Cr | US &lt; $2B. Constrained supply float.<br><br>
                <b style="color: #38bdf8;">🏛️ Mid Caps:</b> India ₹10k–40k Cr | US $2B–$15B. Institutional sweet spot.<br><br>
                <b style="color: #34d399;">💎 Large &amp; Mega Caps:</b> India &gt; ₹40k Cr | US &gt; $15B. Liquid institutional anchors.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    leg_c3, leg_c4 = st.columns(2)
    with leg_c3:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(234, 179, 8, 0.25); border-radius: 8px; padding: 12px; height: 100%;">
            <div style="color: #facc15; font-size: 0.84rem; font-weight: 800; margin-bottom: 6px;">🎯 MINERVINI VCP COIL SIGNALS</div>
            <div style="font-size: 0.76rem; color: #cbd5e1; line-height: 1.45;">
                Volatility Contraction Pattern (VCP):<br><br>
                <b style="color: #facc15;">🎯 Inside Day (ID):</b> Inside yesterday's bar; allows tight stop.<br><br>
                <b style="color: #38bdf8;">🎯 NR7 Tight:</b> Narrowest daily range of last 7 sessions.<br><br>
                <b style="color: #c084fc;">🌀 5D Coil (≤ 1.65x ADR):</b> 5-session contraction near MAs.<br><br>
                <b style="color: #34d399;">🧬 3Wks Tight (3WT):</b> 3 weekly closes within ≤ 1.5%.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with leg_c4:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 12px; height: 100%;">
            <div style="color: #34d399; font-size: 0.84rem; font-weight: 800; margin-bottom: 6px;">📊 BAR QUALITY &amp; SPEEDOMETER</div>
            <div style="font-size: 0.76rem; color: #cbd5e1; line-height: 1.45;">
                Daily Candlestick &amp; Velocity:<br><br>
                <b style="color: #34d399;">🟢 DCR (Daily Closing Range %):</b> <code>(Close-Low)/(High-Low)*100</code>. ≥ 70% = buyers in control.<br><br>
                <b style="color: #f59e0b;">🔥 RVOL:</b> Volume vs 50D avg (≥ 1.5x institutional surge).<br><br>
                <b style="color: #38bdf8;">⚡ 20D ADR %:</b> Average daily range speedometer (≥ 3.5% fast movers).
            </div>
        </div>
        """, unsafe_allow_html=True)

with leg_btn_col:
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    if hasattr(st, "popover"):
        with st.popover("📖 Legend", use_container_width=True):
            _render_master_rs_legend()
if not hasattr(st, "popover"):
    with st.expander("📖 Institutional Execution Legend: Stages, Cap Tiers, VCP Coils & Bar Quality", expanded=False):
        _render_master_rs_legend()

# =========================================================================
# 4. UNIFIED INSTITUTIONAL LEADER RADAR WORKBENCH
# =========================================================================
from rs_radar_ui import render_leader_radar_cockpit

placeholder_txt = "e.g. NVDA, MSFT, SEMIS, BIOTECH, SOFTWARE, ALPHA" if is_us else "e.g. PHARMA, TATA, HOTELS, LUXURY, AUTO, ALPHA"
render_leader_radar_cockpit(
    raw_pool_df,
    state_prefix=f"master_p3_{mkt_code}_{univ_scope}",
    is_us=is_us,
    curr_sym=curr_sym,
    price_col_label=f"Price ({curr_sym})",
    price_fmt=f"{curr_sym}%.2f",
    adtv_col_label=f"20D ADTV ({adtv_unit})",
    adtv_fmt=f"{curr_sym}%.1f {'Cr' if not is_us else 'M'}",
    search_ph=placeholder_txt,
    cohort_cap=cohort_cap,
    is_master=True,
    univ_badge_txt=univ_badge_txt,
    univ_scope=univ_scope,
    diagnostics=raw_pool_df.attrs.get("diagnostics", {}),
    table_height=680,
    benchmark_symbol=bench_sym,
)
