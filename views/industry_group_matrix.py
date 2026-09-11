"""
CANSLIM Industry Group Momentum Matrix (197 Themes) - Lite View
==============================================================
Institutional group rotation tracker with multi-timeframe relative strength rankings,
rank velocity deltas, pack-hunting breadth, actionable leader cards, and constituent drilldowns.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
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

# st.set_page_config removed for Lite router

render_disk_cache_sidebar(get_cached_universe)

# =============================================================================
# WORLD-CLASS TERMINAL DESIGN SYSTEM (BLOOMBERG / HEDGE-FUND GRADE)
# =============================================================================
st.markdown("""
<style>
/* Executive HUD Layout */
.hud-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
@media (max-width: 1200px) {
    .hud-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 640px) {
    .hud-grid { grid-template-columns: 1fr; }
}

.hud-panel {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(2, 6, 23, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(16px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.hud-panel:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.18);
}

.hud-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #10b981, #06b6d4);
}
.hud-panel.accel::before {
    background: linear-gradient(90deg, #38bdf8, #818cf8);
}
.hud-panel.pack::before {
    background: linear-gradient(90deg, #a855f7, #ec4899);
}
.hud-panel.dist::before {
    background: linear-gradient(90deg, #f59e0b, #ef4444);
}

.hud-panel-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.hud-panel-val {
    font-size: 1.25rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.hud-panel-sub {
    font-size: 0.80rem;
    color: #94a3b8;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

/* Status Badges */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.pill-emerald { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
.pill-cyan { background: rgba(6, 182, 212, 0.15); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.4); }
.pill-purple { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
.pill-rose { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
.pill-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }

/* Thematic Card Grid */
.group-card-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 16px;
}
@media (max-width: 1200px) {
    .group-card-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 768px) {
    .group-card-grid { grid-template-columns: 1fr; }
}

.terminal-group-card {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.85) 0%, rgba(2, 6, 23, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.5);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.terminal-group-card:hover {
    transform: translateY(-3px);
    border-color: rgba(56, 189, 248, 0.4);
    box-shadow: 0 16px 32px -8px rgba(0, 0, 0, 0.7), 0 0 20px rgba(56, 189, 248, 0.15);
}

.tg-rank-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 800;
    line-height: 1;
    color: #f8fafc;
}
.tg-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.tg-stock-count {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 500;
}

.tg-perf-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    background: rgba(30, 41, 59, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 8px 10px;
    margin: 12px 0;
    text-align: center;
}
.tg-perf-label {
    font-size: 0.65rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
}
.tg-perf-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
}
.val-pos { color: #34d399; }
.val-neg { color: #f87171; }

.anchor-chip-row {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 10px;
}
.anchor-chip {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 0.80rem;
    font-family: 'JetBrains Mono', monospace;
}
.anchor-sym {
    font-weight: 700;
    color: #e2e8f0;
}
.anchor-badge {
    font-size: 0.70rem;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 4px;
}
.ab-buy { background: rgba(16, 185, 129, 0.25); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.6); }
.ab-retest { background: rgba(59, 130, 246, 0.25); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.6); }
.ab-coil { background: rgba(234, 179, 8, 0.25); color: #fde047; border: 1px solid rgba(234, 179, 8, 0.6); }
.ab-ext { background: rgba(249, 115, 22, 0.25); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.6); }
.ab-fail { background: rgba(239, 68, 68, 0.25); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.6); }
.ab-base { background: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.4); }
</style>
""", unsafe_allow_html=True)

def main():
    render_header(
        "🌊 Industry Group Momentum Matrix (197 Themes)",
        "Cross-Sectional Institutional Rotation, Multi-Timeframe Velocity & Live Execution Setups"
    )

    # -------------------------------------------------------------
    # 1. TAXONOMY SELECTION & CONTROLS (UP FRONT)
    # -------------------------------------------------------------
    c_tax, c_rot, c_top, c_search = st.columns([1.6, 1.2, 0.9, 1.3])
    
    with c_tax:
        taxonomy_choice = st.radio(
            "Taxonomy Architecture",
            ["🏛️ Canonical 145 Sub-Industries", "⚡ Curated Indian Alpha Themes"],
            index=0,
            horizontal=True,
            help="Toggle between standard 145 O'Neil granular groups and high-conviction thematic clusters (Power T&D, Defense, Solar, EMS, Railways, Wealth Tech, etc.)"
        )
        
    tax_key = "thematic" if "Alpha" in taxonomy_choice else "canonical"
    
    with c_rot:
        status_filter = st.selectbox(
            "Rotation Velocity Filter",
            ["All Groups", "🚀 Surging Only", "🔄 Accumulating Only", "⏸️ Consolidating Only", "🐺 High Pack Hunting (≥3)"],
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
            "🔍 Quick Search Sub-Group / Symbol",
            "",
            placeholder="e.g. Defense, Power, TRIL, HAL, Dixon..."
        ).strip().lower()

    # -------------------------------------------------------------
    # 2. COMPUTE ENGINE DATA (SUB-SECOND)
    # -------------------------------------------------------------
    with st.spinner("Calculating cross-sectional rotation & velocity deltas..."):
        df_matrix, benchmark_curve = compute_industry_group_matrix(taxonomy=tax_key)
        
    if df_matrix.empty:
        st.error("⚠️ No industry group price data available. Please verify historical_prices_matrix.pkl.")
        return

    # -------------------------------------------------------------
    # 3. EXECUTIVE HUD PANELS (CLEAN GLASSMORPHISM)
    # -------------------------------------------------------------
    top_leader = df_matrix.iloc[0]
    fastest_accel = df_matrix.sort_values("Delta_1M", ascending=False).iloc[0]
    top_pack = df_matrix.sort_values("Pack_Hunting_Count", ascending=False).iloc[0]
    coldest = df_matrix.sort_values("Delta_1M", ascending=True).iloc[0]

    st.markdown(f"""
    <div class='hud-grid'>
        <div class='hud-panel'>
            <div class='hud-panel-title'>👑 #1 Dominant Super-Leader</div>
            <div class='hud-panel-val' title='{top_leader["Industry_Group"]}'>{top_leader["Industry_Group"]}</div>
            <div class='hud-panel-sub'>
                <span class='badge-pill pill-emerald'>Rank #1</span>
                <span>1M: <b style='color:#34d399;'>{top_leader["Return_1M"]:+.1f}%</b></span>
                <span>• {top_leader["Stock_Count"]} Stocks</span>
            </div>
        </div>
        <div class='hud-panel accel'>
            <div class='hud-panel-title'>🚀 Fastest Velocity Accelerator</div>
            <div class='hud-panel-val' title='{fastest_accel["Industry_Group"]}'>{fastest_accel["Industry_Group"]}</div>
            <div class='hud-panel-sub'>
                <span class='badge-pill pill-cyan'>Δ 1M: {fastest_accel["Delta_1M"]:+d} Spots</span>
                <span>Now <b>#{fastest_accel["Rank_Today"]}</b> <span style='color:#64748b;'>(was #{fastest_accel["Rank_1M"]})</span></span>
            </div>
        </div>
        <div class='hud-panel pack'>
            <div class='hud-panel-title'>🐺 Top Pack Hunting Concentration</div>
            <div class='hud-panel-val' title='{top_pack["Industry_Group"]}'>{top_pack["Industry_Group"]}</div>
            <div class='hud-panel-sub'>
                <span class='badge-pill pill-purple'>{top_pack["Pack_Hunting_Count"]} Stocks RS ≥ 80</span>
                <span>Rank <b>#{top_pack["Rank_Today"]}</b></span>
            </div>
        </div>
        <div class='hud-panel dist'>
            <div class='hud-panel-title'>⚠️ Coldest Institutional Distribution</div>
            <div class='hud-panel-val' title='{coldest["Industry_Group"]}'>{coldest["Industry_Group"]}</div>
            <div class='hud-panel-sub'>
                <span class='badge-pill pill-rose'>Δ 1M: {coldest["Delta_1M"]:+d} Spots</span>
                <span>Now <b>#{coldest["Rank_Today"]}</b> <span style='color:#64748b;'>(was #{coldest["Rank_1M"]})</span></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. FILTERING TABLE DATA
    # -------------------------------------------------------------
    filtered_df = df_matrix.copy()

    if status_filter == "🚀 Surging Only":
        filtered_df = filtered_df[filtered_df["Rotation_Status"].str.contains("Surging")]
    elif status_filter == "🔄 Accumulating Only":
        filtered_df = filtered_df[filtered_df["Rotation_Status"].str.contains("Accumulating")]
    elif status_filter == "⏸️ Consolidating Only":
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
    # 5. MULTI-TABBED WORLD-CLASS PRESENTATION
    # -------------------------------------------------------------
    tab_matrix, tab_cards, tab_quad, tab_guide = st.tabs([
        f"📑 Leadership Matrix Table ({len(filtered_df)})",
        f"🎴 Actionable Leader Cards",
        "🌪️ Rotation Velocity Quadrants",
        "📖 Institutional Methodology"
    ])

    # -------------------------------------------------------------
    # TAB 1: MASTER MATRIX TABLE
    # -------------------------------------------------------------
    with tab_matrix:
        st.caption("💡 *Click on any row in the table to immediately sync its constituent breakdown, pivots, and 1-year curve below.*")
        
        display_df = filtered_df[[
            "Rank_Today", "Industry_Group", "Rotation_Status",
            "Rank_1W", "Delta_1W", "Rank_1M", "Delta_1M", "Rank_3M", "Rank_6M",
            "Pack_Hunting_Count", "Return_1M", "Return_3M", "Return_6M",
            "Top_Leaders_Display", "Sparkline_1M"
        ]].copy()

        col_cfg = {
            "Rank_Today": st.column_config.NumberColumn("Rank", format="%d", width=70, help="Current Relative Strength Rank (1 = Market Leader)"),
            "Industry_Group": st.column_config.TextColumn("Industry Sub-Group", width=250),
            "Rotation_Status": st.column_config.TextColumn("Rotation State", width=120),
            "Rank_1W": st.column_config.NumberColumn("1W Ago", format="%d", width=80),
            "Delta_1W": st.column_config.NumberColumn("Δ 1W", format="%+d", width=75, help="Change in rank over 1 week (positive = improving)"),
            "Rank_1M": st.column_config.NumberColumn("1M Ago", format="%d", width=80),
            "Delta_1M": st.column_config.NumberColumn("Δ 1M", format="%+d", width=75, help="Change in rank over 1 month (positive = accelerating)"),
            "Rank_3M": st.column_config.NumberColumn("3M Ago", format="%d", width=80),
            "Rank_6M": st.column_config.NumberColumn("6M Ago", format="%d", width=80),
            "Pack_Hunting_Count": st.column_config.NumberColumn("🐺 Pack (RS≥80)", format="%d", width=110, help="Count of stocks in group with RS >= 80"),
            "Return_1M": st.column_config.NumberColumn("1M %", format="%.1f%%", width=85),
            "Return_3M": st.column_config.NumberColumn("3M %", format="%.1f%%", width=85),
            "Return_6M": st.column_config.NumberColumn("6M %", format="%.1f%%", width=85),
            "Top_Leaders_Display": st.column_config.TextColumn("Top 3 Anchor Leaders & Live Action State", width=380),
            "Sparkline_1M": st.column_config.LineChartColumn("1M Trend", width=120, help="Normalized daily price trend over trailing 21 trading days")
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

        selected_table_group = None
        if event and event.selection and event.selection.rows:
            sel_idx = event.selection.rows[0]
            if 0 <= sel_idx < len(display_df):
                selected_table_group = display_df.iloc[sel_idx]["Industry_Group"]

    # -------------------------------------------------------------
    # TAB 2: ACTIONABLE LEADER CARDS GRID
    # -------------------------------------------------------------
    with tab_cards:
        st.caption("⚡ *High-density visual terminal cards highlighting the top actionable anchor stocks and live execution badges for every group.*")
        
        cards_slice = filtered_df.head(18)
        cols_per_row = 3
        card_rows = [cards_slice.iloc[i:i+cols_per_row] for i in range(0, len(cards_slice), cols_per_row)]
        
        for c_row in card_rows:
            card_cols = st.columns(cols_per_row)
            for idx, (_, g_data) in enumerate(c_row.iterrows()):
                with card_cols[idx]:
                    r_1w = g_data['Return_1W']
                    r_1m = g_data['Return_1M']
                    r_3m = g_data['Return_3M']
                    r_6m = g_data['Return_6M']
                    
                    c_1w = "val-pos" if r_1w >= 0 else "val-neg"
                    c_1m = "val-pos" if r_1m >= 0 else "val-neg"
                    c_3m = "val-pos" if r_3m >= 0 else "val-neg"
                    c_6m = "val-pos" if r_6m >= 0 else "val-neg"
                    
                    rot_pill = "pill-emerald" if "Surging" in g_data['Rotation_Status'] else (
                        "pill-cyan" if "Accumulating" in g_data['Rotation_Status'] else (
                            "pill-amber" if "Consolidating" in g_data['Rotation_Status'] else "pill-rose"
                        )
                    )
                    
                    chips_html = ""
                    for ldr in g_data['Top_3_Leaders']:
                        st_lbl = ldr['status']
                        if "IN BUY ZONE" in st_lbl:
                            b_cls = "ab-buy"
                            b_text = f"🟢 BUY {ldr['dist_pivot_str']}"
                        elif "RETEST" in st_lbl:
                            b_cls = "ab-retest"
                            b_text = f"🔵 RETEST {ldr['dist_pivot_str']}"
                        elif "COILING" in st_lbl:
                            b_cls = "ab-coil"
                            b_text = f"⏳ COIL {ldr['dist_pivot_str']}"
                        elif "EXTENDED" in st_lbl:
                            b_cls = "ab-ext"
                            b_text = f"🟠 EXT {ldr['dist_pivot_str']}"
                        elif "FAILED" in st_lbl:
                            b_cls = "ab-fail"
                            b_text = "🔴 FAIL"
                        else:
                            b_cls = "ab-base"
                            b_text = "⏳ BASE"
                            
                        chips_html += f"""
                        <div class='anchor-chip'>
                            <span class='anchor-sym'>{ldr['ticker']} <span style='font-size:0.72rem; color:#94a3b8;'>RS {ldr['rs']}</span></span>
                            <span style='color:#f8fafc; font-weight:700;'>₹{ldr['cmp']:,.1f}</span>
                            <span class='anchor-badge {b_cls}'>{b_text}</span>
                        </div>
                        """
                        
                    card_html = f"""
                    <div class='terminal-group-card'>
                        <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                            <div>
                                <div class='tg-title' title='{g_data["Industry_Group"]}'>{g_data["Industry_Group"]}</div>
                                <div class='tg-stock-count'>{g_data["Stock_Count"]} Stocks • 🐺 {g_data["Pack_Hunting_Count"]} with RS ≥ 80</div>
                            </div>
                            <div style='text-align:right;'>
                                <div class='tg-rank-num'>#{g_data["Rank_Today"]}</div>
                                <span class='badge-pill {rot_pill}'>{g_data["Rotation_Status"]}</span>
                            </div>
                        </div>
                        <div class='tg-perf-row'>
                            <div>
                                <div class='tg-perf-label'>1W</div>
                                <div class='tg-perf-val {c_1w}'>{r_1w:+.1f}%</div>
                            </div>
                            <div>
                                <div class='tg-perf-label'>1M</div>
                                <div class='tg-perf-val {c_1m}'>{r_1m:+.1f}%</div>
                            </div>
                            <div>
                                <div class='tg-perf-label'>3M</div>
                                <div class='tg-perf-val {c_3m}'>{r_3m:+.1f}%</div>
                            </div>
                            <div>
                                <div class='tg-perf-label'>6M</div>
                                <div class='tg-perf-val {c_6m}'>{r_6m:+.1f}%</div>
                            </div>
                        </div>
                        <div style='font-size:0.72rem; font-weight:700; color:#64748b; text-transform:uppercase; margin-top:6px;'>
                            Top Anchor Leaders:
                        </div>
                        <div class='anchor-chip-row'>
                            {chips_html}
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                    st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 3: ROTATION VELOCITY QUADRANTS (SCATTER PLOT)
    # -------------------------------------------------------------
    with tab_quad:
        st.markdown("##### 🌪️ Institutional Rotation Quadrants (Strength vs 1M Velocity)")
        st.caption("Identifies groups breaking out with high momentum acceleration (Top-Right) vs groups undergoing institutional distribution (Left).")
        
        plot_df = filtered_df.copy()
        color_map = {
            "🚀 Surging": "#10b981",
            "🔄 Accumulating": "#38bdf8",
            "⏸️ Consolidating": "#fbbf24",
            "⚠️ Distributing": "#ef4444",
            "💤 Lagging": "#64748b",
            "⚪ Neutral": "#94a3b8"
        }
        
        fig_quad = px.scatter(
            plot_df,
            x="Delta_1M",
            y="Comp_RS",
            size="Stock_Count",
            color="Rotation_Status",
            color_discrete_map=color_map,
            hover_name="Industry_Group",
            hover_data={
                "Rank_Today": True,
                "Delta_1M": True,
                "Pack_Hunting_Count": True,
                "Return_1M": ":.1f%",
                "Return_6M": ":.1f%"
            },
            labels={
                "Delta_1M": "1-Month Rank Velocity (Δ Spots Gained/Lost)",
                "Comp_RS": "Composite Relative Strength (0 to 99 Percentile)",
                "Rotation_Status": "Rotation State",
                "Stock_Count": "Universe Size"
            }
        )
        
        fig_quad.add_vline(x=0, line_width=1, line_dash="dash", line_color="#475569")
        fig_quad.add_hline(y=50, line_width=1, line_dash="dash", line_color="#475569")
        
        fig_quad.add_annotation(x=plot_df["Delta_1M"].max()*0.75, y=92, text="👑 ACCELERATING LEADERS", showarrow=False, font=dict(color="#34d399", size=11, family="JetBrains Mono"))
        fig_quad.add_annotation(x=plot_df["Delta_1M"].max()*0.75, y=25, text="🔄 STEALTH ACCUMULATION", showarrow=False, font=dict(color="#38bdf8", size=11, family="JetBrains Mono"))
        fig_quad.add_annotation(x=plot_df["Delta_1M"].min()*0.75, y=92, text="⏸️ DIGESTING LEADERS", showarrow=False, font=dict(color="#fbbf24", size=11, family="JetBrains Mono"))
        fig_quad.add_annotation(x=plot_df["Delta_1M"].min()*0.75, y=25, text="⚠️ DISTRIBUTION VECTOR", showarrow=False, font=dict(color="#f87171", size=11, family="JetBrains Mono"))

        fig_quad.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            height=460,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_quad, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: METHODOLOGY EXPANDER
    # -------------------------------------------------------------
    with tab_guide:
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
          - 🚀 **Surging**: Top 20 rank AND $\Delta \ge +5$ (Coordinated multi-quarter breakout).
          - 🔄 **Accumulating**: Rank 21 to 70 AND $\Delta \ge +15$ (Smart money aggressively rotating in early).
          - ⏸️ **Consolidating**: Top 25 rank AND $-5 \le \Delta \le +5$ (Institutional leaders holding high ground).
          - ⚠️ **Distributing**: $\Delta \le -15$ (Smart money dumping and reallocating capital elsewhere).
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
    # 6. CONSTITUENT DEEP-DIVE DRILLDOWN SECTION
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔬 Constituent Deep-Dive & Actionable Setup Inspector")
    
    group_options = df_matrix["Industry_Group"].tolist()
    default_index = 0
    if selected_table_group and selected_table_group in group_options:
        default_index = group_options.index(selected_table_group)

    chosen_group = st.selectbox(
        "Select Industry Group to Drill Down",
        options=group_options,
        index=default_index,
        help="Select any group to inspect its constituent stocks, pivots, and synthetic equity curve"
    )

    if chosen_group:
        df_constits, curve_df, tv_copy_box = get_group_deep_dive_data(chosen_group, taxonomy=tax_key)
        grp_row = df_matrix[df_matrix["Industry_Group"] == chosen_group].iloc[0]

        # Top Group Metrics Strip
        c_g1, c_g2, c_g3, c_g4 = st.columns(4)
        with c_g1:
            st.metric("Hierarchy Rank", f"#{grp_row['Rank_Today']}", delta=f"{grp_row['Delta_1M']:+d} spots in 1M")
        with c_g2:
            st.metric("Rotation State", grp_row["Rotation_Status"], delta=f"1W: {grp_row['Delta_1W']:+d}")
        with c_g3:
            st.metric("Pack Hunting Breadth", f"{grp_row['Pack_Hunting_Count']} Stocks RS≥80", delta=f"Total {grp_row['Stock_Count']} stocks")
        with c_g4:
            st.metric("Trailing 6M Return", f"{grp_row['Return_6M']:+.1f}%", delta=f"3M: {grp_row['Return_3M']:+.1f}%")

        # Two columns: Chart on left, TV copy box on right
        ch_col, tv_col = st.columns([2.2, 1.2])

        with ch_col:
            st.markdown(f"##### 📈 Synthetic Index: {chosen_group} vs Universe Benchmark (1-Year)")
            if not curve_df.empty:
                fig = go.Figure()
                y_group = (curve_df["Industry Group"] / curve_df["Industry Group"].iloc[0] * 100)
                y_bench = (curve_df["Universe Benchmark"] / curve_df["Universe Benchmark"].iloc[0] * 100)
                
                fig.add_trace(go.Scatter(
                    x=curve_df.index,
                    y=y_group,
                    name=chosen_group,
                    mode="lines",
                    line=dict(color="#10b981", width=2.5),
                    fill="tonexty",
                    fillcolor="rgba(16, 185, 129, 0.08)"
                ))
                fig.add_trace(go.Scatter(
                    x=curve_df.index,
                    y=y_bench,
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
            st.caption("Paste directly into TradingView symbol search or watchlist import modal:")
            st.text_area(
                "TradingView Symbols",
                value=tv_copy_box,
                height=180,
                help="Select all and paste directly into TradingView symbol search or watchlist modal."
            )
            st.markdown(f"**Total Constituents:** `{len(df_constits)}`  |  **RS ≥ 80 Leaders:** `{grp_row['Pack_Hunting_Count']}`")

        # Actionable spotlight cards (if any stock is in buy zone or retest)
        actionable_stocks = df_constits[df_constits['Execution Status'].str.contains("BUY|RETEST|COILING")]
        if not actionable_stocks.empty:
            st.markdown("##### 🎯 Low-Risk CANSLIM Entry Setups in this Group:")
            act_cols = st.columns(min(4, len(actionable_stocks)))
            for a_idx, (_, a_row) in enumerate(actionable_stocks.head(4).iterrows()):
                with act_cols[a_idx % len(act_cols)]:
                    b_color = "#10b981" if "BUY" in a_row['Execution Status'] else (
                        "#38bdf8" if "RETEST" in a_row['Execution Status'] else "#fbbf24"
                    )
                    st.markdown(f"""
                    <div style='background:rgba(15,23,42,0.8); border:1px solid {b_color}; border-radius:8px; padding:10px 12px; margin-bottom:8px;'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <b style='font-family:JetBrains Mono; font-size:1.0rem; color:#f8fafc;'>{a_row["Symbol"]}</b>
                            <span style='font-size:0.75rem; font-weight:700; color:{b_color};'>{a_row["Execution Status"]}</span>
                        </div>
                        <div style='display:flex; justify-content:space-between; font-size:0.80rem; color:#94a3b8; margin-top:4px;'>
                            <span>CMP: <b>₹{a_row["CMP (₹)"]:,.1f}</b></span>
                            <span>Pivot: <b>₹{a_row["Pivot Price"]:,.1f}</b> ({a_row["Dist Pivot %"]:+.1f}%)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Full constituent table
        st.markdown(f"##### 📋 All {len(df_constits)} Constituents in {chosen_group}")
        
        disp_constits = df_constits[[
            "Symbol", "CMP (₹)", "1D %", "1W %", "1M %", "3M %", "1Y %",
            "RS Rating", "Pivot Price", "Dist Pivot %", "Dist 21 EMA %",
            "Execution Status", "TradingView_URL"
        ]].copy()

        c_constit_cfg = {
            "Symbol": st.column_config.TextColumn("Symbol", width=90),
            "CMP (₹)": st.column_config.NumberColumn("CMP (₹)", format="₹%.2f", width=100),
            "1D %": st.column_config.NumberColumn("1D %", format="%.2f%%", width=80),
            "1W %": st.column_config.NumberColumn("1W %", format="%.2f%%", width=80),
            "1M %": st.column_config.NumberColumn("1M %", format="%.2f%%", width=80),
            "3M %": st.column_config.NumberColumn("3M %", format="%.2f%%", width=80),
            "1Y %": st.column_config.NumberColumn("1Y %", format="%.2f%%", width=80),
            "RS Rating": st.column_config.ProgressColumn("RS Score", min_value=0, max_value=99, format="%d", width=100),
            "Pivot Price": st.column_config.NumberColumn("25D Pivot", format="₹%.2f", width=100),
            "Dist Pivot %": st.column_config.NumberColumn("Dist Pivot", format="%.1f%%", width=95, help="Distance from 25-day pivot"),
            "Dist 21 EMA %": st.column_config.NumberColumn("Dist 21 EMA", format="%.1f%%", width=95, help="Distance from 21-day EMA"),
            "Execution Status": st.column_config.TextColumn("CANSLIM Action State", width=140),
            "TradingView_URL": st.column_config.LinkColumn("Chart", display_text="Open ↗", width=80)
        }

        st.dataframe(
            disp_constits,
            column_config=c_constit_cfg,
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    main()