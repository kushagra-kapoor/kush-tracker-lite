"""
CANSLIM Industry Group Momentum Matrix (197 Themes) - Lite View
==============================================================
Institutional group rotation tracker with multi-timeframe relative strength rankings,
rank velocity deltas, pack-hunting breadth, and actionable anchor stock execution setups.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Path configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components import render_header, render_metric_card, render_disk_cache_sidebar
try:
    from views.true_market_leader import get_cached_universe
except ImportError:
    from pages.true_market_leader import get_cached_universe

from industry_group_engine import (
    compute_industry_group_matrix,
    get_group_deep_dive_data,
    INDIAN_ALPHA_THEMES
)

try:
    from styles import load_css
    load_css()
except ImportError:
    pass

# st.set_page_config removed for Lite subpage routing

render_disk_cache_sidebar(get_cached_universe)

# Custom CSS styling for Matrix badges and terminal look
st.markdown("""
<style>
.hud-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
@media (max-width: 1024px) {
    .hud-container {
        grid-template-columns: repeat(2, 1fr);
    }
}
.hud-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(2, 6, 23, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
}
.hud-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #10b981, #06b6d4);
}
.hud-card.orange::before {
    background: linear-gradient(90deg, #f59e0b, #ef4444);
}
.hud-card.purple::before {
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
}
.hud-card.blue::before {
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
.hud-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 6px;
    font-weight: 600;
}
.hud-val {
    font-size: 1.35rem;
    font-weight: 800;
    color: #f8fafc;
    line-height: 1.2;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.hud-sub {
    font-size: 0.8rem;
    color: #64748b;
    display: flex;
    align-items: center;
    gap: 6px;
}
.hud-sub b {
    color: #38bdf8;
}
.hud-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 700;
}
.badge-green { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
.badge-blue { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
.badge-red { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
.badge-amber { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
.filter-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
}
.tv-copy-area {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #38bdf8;
    word-break: break-all;
    user-select: all;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

def main():
    render_header(
        "🌊 Industry Group Momentum Matrix (197 Themes)",
        "Cross-Sectional Institutional Rotation, Multi-Timeframe Velocity & Live Execution Setups"
    )

    with st.expander("📖 Institutional Methodology: The 50% Rule & Rank Velocity", expanded=False):
        st.markdown(r"""
        ### The William J. O'Neil 50% Rule
        *Studies of the greatest stock market winners of the last 100 years prove that **over 50% of an individual stock's total price advance is directly driven by the strength of its underlying industry group**.*
        
        Trading high-RS stocks in lagging or distributing sectors is like swimming against a 50-knot rip current. Conversely, buying actionable leaders in the **Top 10-20 Industry Groups** gives you institutional tailwinds from sovereign wealth funds, mutual funds, and hedge funds buying the entire sector in size.

        ---

        ### The 5 Time Horizon Matrix
        1. **Today**: Current composite momentum rank (1 = Strongest).
        2. **1-Week Ago (5d)**: Short-term tactical positioning.
        3. **1-Month Ago (21d)**: Critical swing cycle anchor point.
        4. **3-Months Ago (63d)**: Intermediate institutional accumulation anchor.
        5. **6-Months Ago (126d)**: Secular structural trend.

        ---

        ### Understanding Rank Velocity Deltas
        - **Rank 1M $\Delta$ (Velocity)**: Computed as `Rank_1M_Ago - Rank_Today`.
          - A positive delta (e.g. `+30`) means the sector has jumped 30 positions up the leadership leaderboard in the last 21 trading days (stealth accumulation).
        - **Rotation States**:
          - 🚀 **Surging Leader**: Top 20 rank AND $\Delta \ge +5$ (Coordinated multi-quarter breakout).
          - 🔄 **Accumulating**: Rank 21 to 70 AND $\Delta \ge +15$ (Smart money aggressively rotating in early).
          - ⏸️ **Consolidating Leader**: Top 25 rank AND $-5 \le \Delta \le +5$ (Institutional leaders holding high ground).
          - ⚠️ **Institutional Distribution**: $\Delta \le -15$ (Smart money dumping and reallocating capital elsewhere).
          - 💤 **Lagging**: Rank $> 70$ (Underperforming the broad market).

        ---

        ### Pack-Hunting Confirmation & Live Actionable Badges
        - **🐺 Pack-Hunting (RS $\ge 80$)**: True institutional sponsorship never moves alone. When $\ge 3$ constituents boast RS scores above 80, the group has valid institutional breadth.
        - **Live Actionable Badges**:
          - `🟢 IN BUY ZONE`: Price is within $0\%$ to $+5.5\%$ above base pivot on volume.
          - `🔵 RETEST`: Low-risk shakeout retesting pivot support ($-2.0\%$ to $0\%$).
          - `⏳ COILING`: Tight consolidation within $5\%$ of breakout pivot.
          - `🟠 EXTENDED`: Price is $>5.5\%$ extended beyond pivot (Do not chase!).
          - `🔴 FAILED`: Price broke below 21 EMA or $>8\%$ below pivot (Stop triggered).
        """)

    # -------------------------------------------------------------
    # 1. TAXONOMY SELECTION & FILTERS
    # -------------------------------------------------------------
    st.markdown("<div class='filter-box'>", unsafe_allow_html=True)
    c_tax, c_rot, c_top, c_search = st.columns([1.6, 1.2, 0.9, 1.3])
    
    with c_tax:
        taxonomy_choice = st.radio(
            "Taxonomy System",
            ["Canonical 145 Sub-Industries", "Curated Indian Alpha Themes"],
            index=0,
            horizontal=True,
            help="Toggle between standard 145 O'Neil industry groups and high-conviction thematic clusters (Power T&D, Defense, Solar, EMS, Railways, etc.)"
        )
        
    tax_key = "thematic" if "Alpha" in taxonomy_choice else "canonical"
    
    with c_rot:
        status_filter = st.selectbox(
            "Rotation State Filter",
            ["All Groups", "🚀 Surging Leaders", "🔄 Accumulating Only", "⏸️ Consolidating Leaders", "🐺 High Pack Hunting (≥3)"],
            index=0
        )
        
    with c_top:
        limit_choice = st.selectbox(
            "Display Scope",
            ["Top 25 Groups", "Top 50 Groups", "Top 100 Groups", "All Groups"],
            index=1
        )
        
    with c_search:
        search_query = st.text_input(
            "🔍 Search Industry / Symbol",
            "",
            placeholder="e.g. Defense, Power, TRIL, HAL..."
        ).strip().lower()
        
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. COMPUTE ENGINE DATA (SUB-SECOND)
    # -------------------------------------------------------------
    with st.spinner("Analyzing cross-sectional industry rotation..."):
        df_matrix, benchmark_curve = compute_industry_group_matrix(taxonomy=tax_key)
        
    if df_matrix.empty:
        st.error("No industry group price data available. Please verify historical_prices_matrix.pkl.")
        return

    # -------------------------------------------------------------
    # 3. EXECUTIVE HUD CARDS
    # -------------------------------------------------------------
    top_leader = df_matrix.iloc[0]
    fastest_accel = df_matrix.sort_values("Delta_1M", ascending=False).iloc[0]
    top_pack = df_matrix.sort_values("Pack_Hunting_Count", ascending=False).iloc[0]
    coldest = df_matrix.sort_values("Delta_1M", ascending=True).iloc[0]

    hud_html = f"""
    <div class='hud-container'>
        <div class='hud-card'>
            <div class='hud-label'>👑 #1 Dominant Super-Leader</div>
            <div class='hud-val'>{top_leader['Industry_Group']}</div>
            <div class='hud-sub'>
                <span class='hud-badge badge-green'>Rank #{top_leader['Rank_Today']}</span>
                <span>1M: <b>{top_leader['Return_1M']:+.1f}%</b></span>
                <span>• {top_leader['Stock_Count']} Stocks</span>
            </div>
        </div>
        <div class='hud-card blue'>
            <div class='hud-label'>🚀 Fastest Velocity Accelerating</div>
            <div class='hud-val'>{fastest_accel['Industry_Group']}</div>
            <div class='hud-sub'>
                <span class='hud-badge badge-blue'>Δ 1M: {fastest_accel['Delta_1M']:+d} Spots</span>
                <span>Now: <b>#{fastest_accel['Rank_Today']}</b> (was #{fastest_accel['Rank_1M']})</span>
            </div>
        </div>
        <div class='hud-card purple'>
            <div class='hud-label'>🐺 Top Pack Hunting Concentration</div>
            <div class='hud-val'>{top_pack['Industry_Group']}</div>
            <div class='hud-sub'>
                <span class='hud-badge badge-green'>{top_pack['Pack_Hunting_Count']} Stocks RS ≥ 80</span>
                <span>Rank: <b>#{top_pack['Rank_Today']}</b></span>
            </div>
        </div>
        <div class='hud-card orange'>
            <div class='hud-label'>⚠️ Coldest Institutional Distribution</div>
            <div class='hud-val'>{coldest['Industry_Group']}</div>
            <div class='hud-sub'>
                <span class='hud-badge badge-red'>Δ 1M: {coldest['Delta_1M']:+d} Spots</span>
                <span>Now: <b>#{coldest['Rank_Today']}</b> (was #{coldest['Rank_1M']})</span>
            </div>
        </div>
    </div>
    """
    st.markdown(hud_html, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. FILTERING TABLE DATA
    # -------------------------------------------------------------
    filtered_df = df_matrix.copy()

    if status_filter == "🚀 Surging Leaders":
        filtered_df = filtered_df[filtered_df["Rotation_Status"].str.contains("Surging")]
    elif status_filter == "🔄 Accumulating Only":
        filtered_df = filtered_df[filtered_df["Rotation_Status"].str.contains("Accumulating")]
    elif status_filter == "⏸️ Consolidating Leaders":
        filtered_df = filtered_df[filtered_df["Rotation_Status"].str.contains("Consolidating")]
    elif status_filter == "🐺 High Pack Hunting (≥3)":
        filtered_df = filtered_df[filtered_df["Pack_Hunting_Count"] >= 3]

    if search_query:
        filtered_df = filtered_df[
            filtered_df["Industry_Group"].str.lower().str.contains(search_query) |
            filtered_df["Constituents"].apply(lambda clist: any(search_query in str(t).lower() for t in clist))
        ]

    if limit_choice == "Top 25 Groups":
        filtered_df = filtered_df.head(25)
    elif limit_choice == "Top 50 Groups":
        filtered_df = filtered_df.head(50)
    elif limit_choice == "Top 100 Groups":
        filtered_df = filtered_df.head(100)

    # -------------------------------------------------------------
    # 5. MASTER ROTATION MATRIX TABLE
    # -------------------------------------------------------------
    st.markdown(f"#### 📊 Multi-Timeframe Leadership Matrix ({len(filtered_df)} of {len(df_matrix)} Groups)")
    st.caption("Click on any row or select from the dropdown below to inspect constituent stocks, pivots, and TradingView charts.")

    display_df = filtered_df[[
        "Rank_Today", "Industry_Group", "Rotation_Status",
        "Rank_1W", "Delta_1W", "Rank_1M", "Delta_1M", "Rank_3M", "Rank_6M",
        "Pack_Hunting_Count", "Return_1M", "Return_3M", "Return_6M",
        "Top_Leaders_Display", "Sparkline_1M"
    ]].copy()

    col_cfg = {
        "Rank_Today": st.column_config.NumberColumn("Rank", format="%d", help="Current relative strength rank (1 = strongest)"),
        "Industry_Group": st.column_config.TextColumn("Industry Sub-Group", width="medium"),
        "Rotation_Status": st.column_config.TextColumn("Rotation Status", width="small"),
        "Rank_1W": st.column_config.NumberColumn("1W Ago", format="%d"),
        "Delta_1W": st.column_config.NumberColumn("Δ 1W", format="%+d", help="Change in rank over 1 week (positive = improving)"),
        "Rank_1M": st.column_config.NumberColumn("1M Ago", format="%d"),
        "Delta_1M": st.column_config.NumberColumn("Δ 1M", format="%+d", help="Change in rank over 1 month (positive = accelerating)"),
        "Rank_3M": st.column_config.NumberColumn("3M Ago", format="%d"),
        "Rank_6M": st.column_config.NumberColumn("6M Ago", format="%d"),
        "Pack_Hunting_Count": st.column_config.NumberColumn("🐺 Pack (RS≥80)", format="%d", help="Count of stocks in group with RS >= 80"),
        "Return_1M": st.column_config.NumberColumn("1M %", format="%.1f%%"),
        "Return_3M": st.column_config.NumberColumn("3M %", format="%.1f%%"),
        "Return_6M": st.column_config.NumberColumn("6M %", format="%.1f%%"),
        "Top_Leaders_Display": st.column_config.TextColumn("Top 3 Anchor Leaders & Live Action State", width="large"),
        "Sparkline_1M": st.column_config.LineChartColumn("1M Trend", help="Normalized daily price trend over trailing 21 trading days")
    }

    event = st.dataframe(
        display_df,
        column_config=col_cfg,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="matrix_table_selection"
    )

    selected_group_name = None
    if event and event.selection and event.selection.rows:
        sel_idx = event.selection.rows[0]
        if 0 <= sel_idx < len(display_df):
            selected_group_name = display_df.iloc[sel_idx]["Industry_Group"]

    # -------------------------------------------------------------
    # 6. CONSTITUENT DEEP-DIVE DRILLDOWN
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔬 Constituent Deep-Dive & Actionable Setup Inspector")
    
    group_options = df_matrix["Industry_Group"].tolist()
    default_index = 0
    if selected_group_name and selected_group_name in group_options:
        default_index = group_options.index(selected_group_name)

    chosen_group = st.selectbox(
        "Select Industry Group to Drill Down",
        options=group_options,
        index=default_index,
        help="Select any group to inspect its constituent stocks, pivots, and synthetic equity curve"
    )

    if chosen_group:
        df_constits, curve_df, tv_copy_box = get_group_deep_dive_data(chosen_group, taxonomy=tax_key)
        grp_row = df_matrix[df_matrix["Industry_Group"] == chosen_group].iloc[0]

        c_g1, c_g2, c_g3, c_g4 = st.columns(4)
        with c_g1:
            st.metric("Hierarchy Rank", f"#{grp_row['Rank_Today']}", delta=f"{grp_row['Delta_1M']:+d} spots in 1M")
        with c_g2:
            st.metric("Rotation Velocity", grp_row["Rotation_Status"], delta=f"1W: {grp_row['Delta_1W']:+d}")
        with c_g3:
            st.metric("Pack Hunting Breadth", f"{grp_row['Pack_Hunting_Count']} Stocks RS≥80", delta=f"Total {grp_row['Stock_Count']} stocks")
        with c_g4:
            st.metric("Trailing 6M Return", f"{grp_row['Return_6M']:+.1f}%", delta=f"3M: {grp_row['Return_3M']:+.1f}%")

        ch_col, tv_col = st.columns([2.2, 1.2])

        with ch_col:
            st.markdown(f"##### 📈 Synthetic Index: {chosen_group} vs Universe Benchmark (1-Year)")
            if not curve_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=curve_df.index,
                    y=(curve_df["Industry Group"] / curve_df["Industry Group"].iloc[0] * 100),
                    name=chosen_group,
                    mode="lines",
                    line=dict(color="#10b981", width=2.5)
                ))
                fig.add_trace(go.Scatter(
                    x=curve_df.index,
                    y=(curve_df["Universe Benchmark"] / curve_df["Universe Benchmark"].iloc[0] * 100),
                    name="Universe Benchmark",
                    mode="lines",
                    line=dict(color="#64748b", width=1.5, dash="dot")
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(15, 23, 42, 0.4)",
                    plot_bgcolor="rgba(15, 23, 42, 0.4)",
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=280,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

        with tv_col:
            st.markdown("##### 📋 TradingView 1-Click Watchlist")
            st.caption("Copy this comma-separated list directly into TradingView Watchlist Import:")
            st.text_area(
                "TradingView Symbols",
                value=tv_copy_box,
                height=180,
                help="Select all and paste directly into TradingView symbol search or watchlist modal."
            )
            st.markdown(f"**Total Constituents:** `{len(df_constits)}`")

        st.markdown(f"##### 🎯 Actionable Constituent Breakdown: {chosen_group}")
        
        disp_constits = df_constits[[
            "Symbol", "CMP (₹)", "1D %", "1W %", "1M %", "3M %", "1Y %",
            "RS Rating", "Pivot Price", "Dist Pivot %", "Dist 21 EMA %",
            "Execution Status", "TradingView_URL"
        ]].copy()

        c_constit_cfg = {
            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
            "CMP (₹)": st.column_config.NumberColumn("CMP (₹)", format="₹%.2f"),
            "1D %": st.column_config.NumberColumn("1D %", format="%.2f%%"),
            "1W %": st.column_config.NumberColumn("1W %", format="%.2f%%"),
            "1M %": st.column_config.NumberColumn("1M %", format="%.2f%%"),
            "3M %": st.column_config.NumberColumn("3M %", format="%.2f%%"),
            "1Y %": st.column_config.NumberColumn("1Y %", format="%.2f%%"),
            "RS Rating": st.column_config.ProgressColumn("RS Score", min_value=0, max_value=99, format="%d"),
            "Pivot Price": st.column_config.NumberColumn("25D Pivot", format="₹%.2f"),
            "Dist Pivot %": st.column_config.NumberColumn("Dist Pivot", format="%.1f%%", help="Distance from 25-day pivot"),
            "Dist 21 EMA %": st.column_config.NumberColumn("Dist 21 EMA", format="%.1f%%", help="Distance from 21-day EMA"),
            "Execution Status": st.column_config.TextColumn("CANSLIM Action State", width="medium"),
            "TradingView_URL": st.column_config.LinkColumn("Chart", display_text="Open ↗")
        }

        st.dataframe(
            disp_constits,
            column_config=c_constit_cfg,
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    main()