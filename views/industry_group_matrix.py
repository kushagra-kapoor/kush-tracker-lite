"""
CANSLIM Industry Group Momentum Matrix (197 Themes)
===================================================
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
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from components import render_header, render_metric_card, render_disk_cache_sidebar

try:
    from views.true_market_leader import get_cached_universe
except Exception:
    try:
        from pages.true_market_leader import get_cached_universe
    except Exception:
        def get_cached_universe(*args, **kwargs):
            return []

import importlib
try:
    import industry_group_engine
    if not hasattr(industry_group_engine, 'compute_dual_momentum_summary'):
        importlib.reload(industry_group_engine)
except Exception:
    pass

from industry_group_engine import (
    compute_industry_group_matrix,
    compute_participation,
    compute_dual_momentum_summary,
    get_group_deep_dive_data,
    get_top_and_worst_40_groups,
    INDIAN_ALPHA_THEMES
)

try:
    from styles import load_css
    load_css()
except ImportError:
    pass

# st.set_page_config removed for Lite router
render_disk_cache_sidebar(get_cached_universe)

def clean_html(html_str: str) -> str:
    """Strips leading whitespace from every line so Markdown never treats it as code."""
    return "\n".join(line.strip() for line in html_str.splitlines() if line.strip())

# =============================================================================
# WORLD-CLASS TERMINAL DESIGN SYSTEM (BLOOMBERG / HEDGE-FUND GRADE)
# =============================================================================
st.markdown(clean_html("""
<style>
/* Where Is The Strength? — World-Class Participation Barometer */
.participation-barometer {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.95) 0%, rgba(2, 6, 23, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 22px;
    box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.65), 0 0 0 1px rgba(255, 255, 255, 0.04);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
}
.participation-barometer.broad::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #10b981 0%, #06b6d4 50%, #3b82f6 100%);
}
.participation-barometer.selective::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 50%, #38bdf8 100%);
}
.participation-barometer.narrow::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #ef4444 0%, #f97316 50%, #fbbf24 100%);
}

.part-top-strip {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 14px;
    margin-bottom: 16px;
    flex-wrap: wrap;
    gap: 12px;
}
.part-title-group {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
.part-title-badge {
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 6px;
}
.part-methodology-hint {
    font-size: 0.76rem;
    color: #94a3b8;
    font-weight: 500;
}
.part-badges-group {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.regime-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
}
.regime-badge.broad {
    background: rgba(16, 185, 129, 0.20);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.55);
    box-shadow: 0 0 14px rgba(16, 185, 129, 0.25);
}
.regime-badge.selective {
    background: rgba(245, 158, 11, 0.20);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.55);
    box-shadow: 0 0 14px rgba(245, 158, 11, 0.25);
}
.regime-badge.narrow {
    background: rgba(239, 68, 68, 0.20);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.55);
    box-shadow: 0 0 14px rgba(239, 68, 68, 0.25);
}

.drift-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(6, 182, 212, 0.16);
    color: #38bdf8;
    border: 1px solid rgba(6, 182, 212, 0.45);
}

.part-body-grid {
    display: grid;
    grid-template-columns: 1.15fr 1.85fr;
    gap: 24px;
    align-items: center;
}
@media (max-width: 1024px) {
    .part-body-grid {
        grid-template-columns: 1fr;
        gap: 18px;
    }
}

.part-macro-col {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.part-stat-hero {
    display: flex;
    align-items: baseline;
    gap: 14px;
    flex-wrap: wrap;
}
.part-big-num {
    font-size: 2.3rem;
    font-weight: 900;
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.03em;
    line-height: 1;
}
.part-big-sub {
    font-size: 1.15rem;
    font-weight: 600;
    color: #64748b;
}
.part-hero-pct {
    font-size: 1.30rem;
    font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
    padding: 3px 12px;
    border-radius: 8px;
    line-height: 1.2;
}
.part-hero-pct.broad {
    background: rgba(16, 185, 129, 0.16);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.45);
}
.part-hero-pct.selective {
    background: rgba(245, 158, 11, 0.16);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.45);
}
.part-hero-pct.narrow {
    background: rgba(239, 68, 68, 0.16);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.45);
}

.part-sub-explainer {
    font-size: 0.88rem;
    font-weight: 500;
    color: #cbd5e1;
    line-height: 1.4;
}
.part-sub-explainer b {
    color: #f8fafc;
    font-weight: 700;
}

.breadth-gauge-wrap {
    margin-top: 4px;
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.breadth-track {
    position: relative;
    width: 100%;
    height: 10px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 9999px;
    overflow: hidden;
}
.breadth-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.4s ease;
}
.breadth-fill.broad {
    background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.55);
}
.breadth-fill.selective {
    background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%);
    box-shadow: 0 0 12px rgba(245, 158, 11, 0.55);
}
.breadth-fill.narrow {
    background: linear-gradient(90deg, #ef4444 0%, #f87171 100%);
    box-shadow: 0 0 12px rgba(239, 68, 68, 0.55);
}

.breadth-ticks {
    display: flex;
    justify-content: space-between;
    font-size: 0.70rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #94a3b8;
}

.part-cards-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}
@media (max-width: 768px) {
    .part-cards-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

.part-horizon-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 10px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.part-horizon-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.4);
}
.phc-label {
    font-size: 0.68rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 4px;
}
.phc-num {
    font-size: 1.45rem;
    font-weight: 900;
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
    margin-bottom: 4px;
}
.phc-num.pos { color: #34d399; }
.phc-num.mid { color: #38bdf8; }
.phc-num.neg { color: #f87171; }
.phc-sub {
    font-size: 0.75rem;
    font-weight: 600;
    color: #cbd5e1;
    white-space: nowrap;
}

.alignment-chips-col {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 2px;
}
.align-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.74rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

/* Gary Antonacci Dual Momentum Regime Radar */
.dual-momentum-barometer {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.95) 0%, rgba(2, 6, 23, 0.98) 100%);
    border: 1px solid rgba(139, 92, 246, 0.28);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 22px;
    box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.65), 0 0 0 1px rgba(139, 92, 246, 0.08);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
}
.dual-momentum-barometer::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #8b5cf6 0%, #06b6d4 50%, #10b981 100%);
}

.dual-mirage-warning-box {
    margin-top: 10px;
    padding: 8px 12px;
    border-radius: 8px;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.35);
    font-size: 0.76rem;
    color: #fde68a;
    line-height: 1.45;
}

.dual-mirage-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(245, 158, 11, 0.16);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.45);
}

.dual-quad-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 10px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.dual-quad-card:hover {
    border-color: rgba(255, 255, 255, 0.25);
    transform: translateY(-2px);
}
.dual-quad-card.alpha { border-left: 3px solid #10b981; }
.dual-quad-card.mirage { border-left: 3px solid #f59e0b; }
.dual-quad-card.bleed { border-left: 3px solid #ef4444; }
.dual-quad-card.abs { border-left: 3px solid #38bdf8; }

.dqc-label {
    font-size: 0.68rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 2px;
}
.dqc-val {
    font-size: 1.35rem;
    font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
    margin-bottom: 3px;
}
.dqc-val.alpha { color: #34d399; }
.dqc-val.mirage { color: #fbbf24; }
.dqc-val.bleed { color: #f87171; }
.dqc-val.abs { color: #38bdf8; }

.dqc-sub {
    font-size: 0.73rem;
    font-weight: 600;
    color: #cbd5e1;
    line-height: 1.3;
}
.dqc-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.68rem;
    font-weight: 700;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

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
.terminal-group-card {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.85) 0%, rgba(2, 6, 23, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.5);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    margin-bottom: 16px;
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

.info-banner-bubble {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}
</style>
"""), unsafe_allow_html=True)

def main():
    render_header(
        "🌊 Industry Group Momentum Matrix (197 Themes)",
        "Cross-Sectional Institutional Rotation, Multi-Timeframe Velocity & Live Execution Setups"
    )

    # -------------------------------------------------------------
    # 1. TAXONOMY SELECTION & CONTROLS (UP FRONT)
    # -------------------------------------------------------------
    c_tax, c_sort, c_rot, c_top, c_search = st.columns([1.4, 1.4, 1.1, 0.9, 1.2])
    
    with c_tax:
        taxonomy_choice = st.selectbox(
            "Taxonomy Architecture",
            ["🏛️ Canonical 145 Sub-Industries", "⚡ Curated Indian Alpha Themes"],
            index=0,
            help="Toggle between standard 145 O'Neil granular groups and high-conviction thematic clusters (Power T&D, Defense, Solar, EMS, Railways, Wealth Tech, etc.)"
        )
        
    tax_key = "thematic" if "Alpha" in taxonomy_choice else "canonical"
    
    with c_sort:
        sort_choice = st.selectbox(
            "Sort Groups By",
            [
                "🏆 Leadership Rank (1 to N)",
                "🌐 3M Excess Return vs Index (High to Low)",
                "🛡️ Sortino 3M (High to Low)",
                "👑 Apex Leaders Count (High to Low)",
                "🐺 Wolfpack Breadth (RS ≥ 80 Count)",
                "🐺 Wolfpack Concentration (% RS ≥ 80)",
                "🚀 1M Rank Velocity (Δ Spots)",
                "📈 1M Momentum Return %",
                "🏢 Total Constituent Count"
            ],
            index=0,
            help="Sort groups across tables and cards by leadership momentum rank, Sortino ratio, wolfpack pack size, or velocity."
        )

    with c_rot:
        status_filter = st.selectbox(
            "Rotation Velocity Filter",
            [
                "All Groups",
                "🚀 Dual Alpha Only (RS>0 & Abs>0)",
                "🛡️ Relative Mirage Only (RS>0 & Abs≤0)",
                "🟢 Confluence Only",
                "🌱 Emerging Turn Only",
                "⚠️ Fading Only",
                "🚀 Surging Only",
                "🔄 Accumulating Only",
                "⏸️ Consolidating Only",
                "🐺 High Pack Hunting (≥3)"
            ],
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

    part_stats = compute_participation(df_matrix)

    # -------------------------------------------------------------
    # 3. EXECUTIVE HUD PANELS & PARTICIPATION BAROMETER
    # -------------------------------------------------------------
    tone = part_stats.get("tone", "broad")
    drift_txt = part_stats.get("drift", "Stable")
    drift_pts = part_stats.get("drift_pts", 0.0)
    drift_icon = "↗" if drift_pts >= 5.0 else ("↘" if drift_pts <= -5.0 else "→")
    bench_name = part_stats.get("benchmark", "NIFTY 500")
    total_grps = part_stats.get("total", len(df_matrix))
    above_3m = part_stats.get("above_3M", 0)
    pct_3m = part_stats.get("pct_3M", 0.0)
    above_1m = part_stats.get("above_1M", 0)
    pct_1m = part_stats.get("pct_1M", 0.0)
    above_6m = part_stats.get("above_6M", 0)
    pct_6m = part_stats.get("pct_6M", 0.0)
    confl = part_stats.get("confluence", 0)
    emerg = part_stats.get("emerging", 0)
    fading = part_stats.get("fading", 0)

    gauge_width = min(max(pct_3m, 0.0), 100.0)
    c1m_cls = "pos" if pct_1m >= 55.0 else ("mid" if pct_1m >= 40.0 else "neg")
    c3m_cls = "pos" if pct_3m >= 55.0 else ("mid" if pct_3m >= 40.0 else "neg")
    c6m_cls = "pos" if pct_6m >= 55.0 else ("mid" if pct_6m >= 40.0 else "neg")

    part_html = f"""
    <div class='participation-barometer {tone}'>
        <div class='part-top-strip'>
            <div class='part-title-group'>
                <span class='part-title-badge'>🌐 WHERE IS THE STRENGTH? · PARTICIPATION RATIO</span>
                <span class='part-methodology-hint'>| Atul Suri / Marathon Trends Breadth Barometer</span>
            </div>
            <div class='part-badges-group'>
                <span class='regime-badge {tone}'>
                    {"🟢" if tone == "broad" else ("🟡" if tone == "selective" else "🔴")} {part_stats.get("reading", "Broad").upper()} PARTICIPATION ({pct_3m:.1f}%)
                </span>
                <span class='drift-badge'>
                    BREADTH DRIFT: {drift_txt.upper()} ({drift_pts:+.1f}%) {drift_icon}
                </span>
            </div>
        </div>
        <div class='part-body-grid'>
            <div class='part-macro-col'>
                <div class='part-stat-hero'>
                    <div class='part-big-num'>{above_3m} <span class='part-big-sub'>/ {total_grps}</span></div>
                    <div class='part-hero-pct {tone}'>{pct_3m:.1f}%</div>
                </div>
                <div class='part-sub-explainer'>
                    Industry Groups Outperforming <b>{bench_name}</b> on the <b>3M Medium-Horizon</b> anchor.
                </div>
                <div class='breadth-gauge-wrap'>
                    <div class='breadth-track'>
                        <div class='breadth-fill {tone}' style='width: {gauge_width:.1f}%;'></div>
                        <div class='gauge-marker-40' title='40% Selective Threshold'></div>
                        <div class='gauge-marker-55' title='55% Broad Threshold'></div>
                    </div>
                    <div class='breadth-ticks'>
                        <span style='color:#f87171;'>0% Narrow</span>
                        <span style='color:#fbbf24;'>40% Selective</span>
                        <span style='color:#34d399;'>55% Broad</span>
                        <span style='color:#94a3b8;'>100%</span>
                    </div>
                </div>
            </div>
            <div class='part-cards-grid'>
                <div class='part-horizon-card'>
                    <div class='phc-label'>Short-Term (1M)</div>
                    <div class='phc-num {c1m_cls}'>{pct_1m:.1f}%</div>
                    <div class='phc-sub'><b style='color:#f8fafc;'>{above_1m}</b> of {total_grps} Beating</div>
                </div>
                <div class='part-horizon-card'>
                    <div class='phc-label'>Medium-Term (3M)</div>
                    <div class='phc-num {c3m_cls}'>{pct_3m:.1f}%</div>
                    <div class='phc-sub'><b style='color:#f8fafc;'>{above_3m}</b> of {total_grps} Beating</div>
                </div>
                <div class='part-horizon-card'>
                    <div class='phc-label'>Long-Term (6M)</div>
                    <div class='phc-num {c6m_cls}'>{pct_6m:.1f}%</div>
                    <div class='phc-sub'><b style='color:#f8fafc;'>{above_6m}</b> of {total_grps} Beating</div>
                </div>
                <div class='part-horizon-card'>
                    <div class='phc-label'>3H Alignment</div>
                    <div class='alignment-chips-col'>
                        <span class='align-chip' style='color:#34d399;'>🟢 <b>{confl}</b> Confluence</span>
                        <span class='align-chip' style='color:#38bdf8;'>🌱 <b>{emerg}</b> Emerging</span>
                        <span class='align-chip' style='color:#fbbf24;'>⚠️ <b>{fading}</b> Fading</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(part_html), unsafe_allow_html=True)

    dual_stats = compute_dual_momentum_summary(df_matrix)
    d_tone = dual_stats.get("tone", "selective")
    d_reading = dual_stats.get("reading", "Selective Dual Momentum")
    d_alpha_cnt = dual_stats.get("dual_alpha_count", 0)
    d_alpha_pct = dual_stats.get("dual_alpha_pct", 0.0)
    d_mirage_cnt = dual_stats.get("mirage_count", 0)
    d_mirage_pct = dual_stats.get("mirage_pct", 0.0)
    d_bleed_cnt = dual_stats.get("bleed_count", 0)
    d_bleed_pct = dual_stats.get("bleed_pct", 0.0)
    d_bench_ret = dual_stats.get("bench_ret_3m", 0.0)
    d_spread = dual_stats.get("mirage_spread", 0.0)
    
    d_abs_1m = dual_stats.get("abs_1m_pct", 0.0)
    d_abs_3m = dual_stats.get("abs_3m_pct", 0.0)
    d_abs_6m = dual_stats.get("abs_6m_pct", 0.0)
    
    d_gauge_width = min(max(d_alpha_pct, 0.0), 100.0)
    
    mirage_warning_html = ""
    if d_mirage_cnt > 0:
        mirage_warning_html = f"""
        <div class='dual-mirage-warning-box'>
            ⚠️ <b>Relative Mirage Alert ({d_mirage_cnt} Groups · {d_mirage_pct:.1f}%):</b> Outperforming <b>{bench_name}</b> ({d_bench_ret:+.1f}%), but in <i>negative</i> absolute territory. Naive RS flags them green; Dual Momentum filters them out.
        </div>
        """
        
    dual_html = f"""
    <div class='dual-momentum-barometer'>
        <div class='part-top-strip'>
            <div class='part-title-group'>
                <span class='part-title-badge'>⚔️ DUAL MOMENTUM REGIME RADAR</span>
                <span class='part-methodology-hint'>| Gary Antonacci Absolute Gate + Relative Alpha</span>
            </div>
            <div class='part-badges-group'>
                <span class='regime-badge {d_tone}'>
                    {"🚀" if d_tone == "broad" else ("🟡" if d_tone == "selective" else "⚠️")} {d_reading.upper()} ({d_alpha_pct:.1f}%)
                </span>
                <span class='dual-mirage-pill'>
                    MIRAGE SPREAD: {d_mirage_cnt} GROUPS ({d_spread:.1f}% GAP)
                </span>
            </div>
        </div>
        <div class='part-body-grid'>
            <div class='part-macro-col'>
                <div class='part-stat-hero'>
                    <div class='part-big-num'>{d_alpha_cnt} <span class='part-big-sub'>/ {total_grps}</span></div>
                    <div class='part-hero-pct {d_tone}'>{d_alpha_pct:.1f}%</div>
                </div>
                <div class='part-sub-explainer'>
                    Industry Groups passing <b>BOTH</b> Relative Strength (> {bench_name}) <b>AND</b> Absolute Trend (3M Return > 0%).
                </div>
                <div class='breadth-gauge-wrap'>
                    <div class='breadth-track'>
                        <div class='breadth-fill {d_tone}' style='width: {d_gauge_width:.1f}%;'></div>
                        <div class='gauge-marker-40' title='40% Selective Dual Threshold'></div>
                        <div class='gauge-marker-55' title='55% Broad Dual Threshold'></div>
                    </div>
                    <div class='breadth-ticks'>
                        <span style='color:#f87171;'>0% Narrow</span>
                        <span style='color:#fbbf24;'>40% Selective</span>
                        <span style='color:#34d399;'>55% Broad Alpha</span>
                        <span style='color:#94a3b8;'>100%</span>
                    </div>
                </div>
                {mirage_warning_html}
            </div>
            <div class='part-cards-grid'>
                <div class='dual-quad-card alpha'>
                    <div>
                        <div class='dqc-label'>Quad I · True Alpha</div>
                        <div class='dqc-val alpha'>{d_alpha_cnt} <span style='font-size:0.8rem; color:#94a3b8;'>({d_alpha_pct:.1f}%)</span></div>
                        <div class='dqc-sub'>RS > 0 & Return > 0%</div>
                    </div>
                    <span class='dqc-chip' style='color:#34d399;'>🟢 Full Approval</span>
                </div>
                <div class='dual-quad-card mirage'>
                    <div>
                        <div class='dqc-label'>Quad II · Mirage Trap</div>
                        <div class='dqc-val mirage'>{d_mirage_cnt} <span style='font-size:0.8rem; color:#94a3b8;'>({d_mirage_pct:.1f}%)</span></div>
                        <div class='dqc-sub'>RS > 0 BUT Return ≤ 0%</div>
                    </div>
                    <span class='dqc-chip' style='color:#fbbf24;'>⚠️ Capital at Risk</span>
                </div>
                <div class='dual-quad-card bleed'>
                    <div>
                        <div class='dqc-label'>Quad III · Dual Bleed</div>
                        <div class='dqc-val bleed'>{d_bleed_cnt} <span style='font-size:0.8rem; color:#94a3b8;'>({d_bleed_pct:.1f}%)</span></div>
                        <div class='dqc-sub'>RS ≤ 0 & Return ≤ 0%</div>
                    </div>
                    <span class='dqc-chip' style='color:#f87171;'>🔴 Chronic Drag</span>
                </div>
                <div class='dual-quad-card abs'>
                    <div>
                        <div class='dqc-label'>Absolute Breadth</div>
                        <div class='dqc-val abs'>{d_abs_3m:.1f}% <span style='font-size:0.8rem; color:#94a3b8;'>(3M)</span></div>
                        <div class='dqc-sub'>1M: <b style='color:#f8fafc;'>{d_abs_1m:.0f}%</b> · 6M: <b style='color:#f8fafc;'>{d_abs_6m:.0f}%</b></div>
                    </div>
                    <span class='dqc-chip' style='color:#38bdf8;'>📈 Nominal Positive</span>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(dual_html), unsafe_allow_html=True)

    top_leader = df_matrix.iloc[0]
    fastest_accel = df_matrix.sort_values("Delta_1M", ascending=False).iloc[0]
    top_pack = df_matrix.sort_values("Pack_Hunting_Count", ascending=False).iloc[0]
    top_sortino = df_matrix.sort_values("Sortino_3M", ascending=False).iloc[0]

    hud_html = f"""
    <div class='hud-grid'>
        <div class='hud-panel'>
            <div class='hud-panel-title'>👑 #1 Dominant Super-Leader</div>
            <div class='hud-panel-val' title='{top_leader["Industry_Group"]}'>{top_leader["Industry_Group"]}</div>
            <div class='hud-panel-sub'>
                <span class='badge-pill pill-emerald'>Rank #1</span>
                <span>1M: <b style='color:#34d399;'>{top_leader["Return_1M"]:+.1f}%</b></span>
                <span style='color:#38bdf8; font-weight:700;'>• 🛡️ S(3M): {top_leader.get("Sortino_3M", 0.0):.2f}</span>
                <span style='color:#c084fc;'>• 👑 {top_leader.get("Apex_Leaders", 0)} Apex</span>
            </div>
        </div>
        <div class='hud-panel accel'>
            <div class='hud-panel-title'>🚀 Fastest Velocity Accelerator</div>
            <div class='hud-panel-val' title='{fastest_accel["Industry_Group"]}'>{fastest_accel["Industry_Group"]}</div>
            <div class='hud-panel-sub'>
                <span class='badge-pill pill-cyan'>Δ 1M: {fastest_accel["Delta_1M"]:+d} Spots</span>
                <span>Now <b>#{fastest_accel["Rank_Today"]}</b> <span style='color:#94a3b8; font-weight:600;'>(was #{fastest_accel["Rank_1M"]})</span></span>
                <span style='color:#38bdf8;'>• S(3M): {fastest_accel.get("Sortino_3M", 0.0):.2f}</span>
            </div>
        </div>
        <div class='hud-panel pack'>
            <div class='hud-panel-title'>🐺 Top Pack Hunting Concentration</div>
            <div class='hud-panel-val' title='{top_pack["Industry_Group"]}'>{top_pack["Industry_Group"]}</div>
            <div class='hud-panel-sub'>
                <span class='badge-pill pill-purple'>{top_pack["Pack_Hunting_Count"]} RS ≥ 80</span>
                <span class='badge-pill pill-emerald'>👑 {top_pack.get("Apex_Leaders", 0)} Apex</span>
                <span>Rank <b>#{top_pack["Rank_Today"]}</b></span>
            </div>
        </div>
        <div class='hud-panel dist' style='border-top:3px solid #06b6d4;'>
            <div class='hud-panel-title'>🛡️ Top Downside Quality (Sortino)</div>
            <div class='hud-panel-val' title='{top_sortino["Industry_Group"]}'>{top_sortino["Industry_Group"]}</div>
            <div class='hud-panel-sub'>
                <span class='badge-pill pill-cyan'>🛡️ Sortino {top_sortino["Sortino_3M"]:.2f}</span>
                <span class='badge-pill pill-emerald'>👑 {top_sortino.get("Apex_Leaders", 0)} Apex</span>
                <span>Rank <b>#{top_sortino["Rank_Today"]}</b></span>
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(hud_html), unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. FILTERING & SORTING DATA ACROSS ALL VIEWS
    # -------------------------------------------------------------
    filtered_df = df_matrix.copy()

    if status_filter == "🚀 Dual Alpha Only (RS>0 & Abs>0)":
        filtered_df = filtered_df[filtered_df["Is_Dual_Alpha"] == True]
    elif status_filter == "🛡️ Relative Mirage Only (RS>0 & Abs≤0)":
        filtered_df = filtered_df[filtered_df["Is_Relative_Mirage"] == True]
    elif status_filter == "🟢 Confluence Only":
        filtered_df = filtered_df[filtered_df["Horizon_State"].str.contains("Confluence", na=False)]
    elif status_filter == "🌱 Emerging Turn Only":
        filtered_df = filtered_df[filtered_df["Horizon_State"].str.contains("Emerging", na=False)]
    elif status_filter == "⚠️ Fading Only":
        filtered_df = filtered_df[filtered_df["Horizon_State"].str.contains("Fading", na=False)]
    elif status_filter == "🚀 Surging Only":
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

    # In-page Dual-Horizon Sortino & Apex Leader controls (Global across Table, Cards & Quadrants)
    f_c1, f_c2, f_c3 = st.columns([1.8, 1.1, 1.1])
    with f_c1:
        st.markdown("**🛡️ Downside-Adjusted Quality Filters** *(MAR = 6.5%)*")
        st.caption("Filters themes with high risk-adjusted consistency & institutional compounder density across all tabs.")
    with f_c2:
        igm_min_s3m = st.slider(
            "Min Sortino (3M)",
            min_value=-2.0,
            max_value=15.0,
            value=-2.0,
            step=0.5,
            help="Filter groups by minimum 3-month Sortino Ratio of the theme synthetic curve.",
            key="igm_min_s3m"
        )
    with f_c3:
        igm_min_apex = st.slider(
            "Min Apex Leaders",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Filter groups with minimum count of Apex Leaders (RS ≥ 80 AND 3M Sortino ≥ 3.0).",
            key="igm_min_apex"
        )

    if igm_min_s3m > -2.0:
        filtered_df = filtered_df[filtered_df["Sortino_3M"] >= igm_min_s3m]
    if igm_min_apex > 0:
        filtered_df = filtered_df[filtered_df["Apex_Leaders"] >= igm_min_apex]

    if igm_min_s3m > -2.0 or igm_min_apex > 0:
        st.info(f"🔍 **Quality Filter Active**: Showing **{len(filtered_df)} of {len(df_matrix)}** groups matching Min Sortino 3M ≥ {igm_min_s3m:.1f} and Min Apex Leaders ≥ {igm_min_apex}")

    # Apply Selected Sort Hierarchy
    if "Excess Return" in sort_choice:
        filtered_df = filtered_df.sort_values(["Excess_3M", "Rank_Today"], ascending=[False, True])
    elif "Sortino 3M" in sort_choice:
        filtered_df = filtered_df.sort_values(["Sortino_3M", "Rank_Today"], ascending=[False, True])
    elif "Apex Leaders" in sort_choice:
        filtered_df = filtered_df.sort_values(["Apex_Leaders", "Rank_Today"], ascending=[False, True])
    elif "Wolfpack Breadth" in sort_choice or "RS ≥ 80 Count" in sort_choice:
        filtered_df = filtered_df.sort_values(["Pack_Hunting_Count", "Rank_Today"], ascending=[False, True])
    elif "Concentration" in sort_choice:
        filtered_df["_pack_ratio"] = filtered_df["Pack_Hunting_Count"] / filtered_df["Stock_Count"].clip(lower=1)
        filtered_df = filtered_df.sort_values(["_pack_ratio", "Pack_Hunting_Count", "Rank_Today"], ascending=[False, False, True])
    elif "Velocity" in sort_choice:
        filtered_df = filtered_df.sort_values(["Delta_1M", "Rank_Today"], ascending=[False, True])
    elif "Return" in sort_choice:
        filtered_df = filtered_df.sort_values(["Return_1M", "Rank_Today"], ascending=[False, True])
    elif "Constituent" in sort_choice:
        filtered_df = filtered_df.sort_values(["Stock_Count", "Rank_Today"], ascending=[False, True])
    else:
        filtered_df = filtered_df.sort_values("Rank_Today", ascending=True)

    if limit_choice == "Top 25 Groups":
        filtered_df = filtered_df.head(25)
    elif limit_choice == "Top 50 Groups":
        filtered_df = filtered_df.head(50)
    elif limit_choice == "Top 100 Groups":
        filtered_df = filtered_df.head(100)

    # -------------------------------------------------------------
    # TRACK FILTER & SORT SIGNATURE FOR SYNCING DRILLDOWN
    # -------------------------------------------------------------
    filter_sig = f"{tax_key}|{sort_choice}|{status_filter}|{limit_choice}|{search_query}|{igm_min_s3m}|{igm_min_apex}"
    sel_widget_key = f"igm_drilldown_sel_{filter_sig}"
    
    if st.session_state.get("igm_last_filter_sig") != filter_sig:
        st.session_state["igm_last_filter_sig"] = filter_sig
        # Top-level sort or filter changed! Reset drilldown to top group of the newly sorted list
        if not filtered_df.empty:
            new_top_group = filtered_df.iloc[0]["Industry_Group"]
            st.session_state["igm_selected_group"] = new_top_group
            st.session_state[sel_widget_key] = new_top_group

    # -------------------------------------------------------------
    # 5. MULTI-TABBED WORLD-CLASS PRESENTATION
    # -------------------------------------------------------------
    tab_matrix, tab_top_worst, tab_cards, tab_quad, tab_guide = st.tabs([
        f"📑 Leadership Matrix Table ({len(filtered_df)})",
        "🏆 Top 40 Leaders vs Worst 40 Laggards (6-Week)",
        f"🎴 Actionable Leader Cards ({len(filtered_df)})",
        "🌪️ Rotation Velocity Quadrants",
        "📖 Institutional Methodology"
    ])

    # -------------------------------------------------------------
    # TAB 1: MASTER MATRIX TABLE
    # -------------------------------------------------------------
    with tab_matrix:
        tv_col1, tv_col2 = st.columns([1.6, 2.4])
        with tv_col1:
            st.caption("💡 *Click on any row in the table to immediately sync constituent breakdown below.*")
        with tv_col2:
            tv_mode = st.radio(
                "Table View:",
                ["📊 Executive Overview", "🛡️ Risk & Quality", "🌪️ Momentum Velocity", "📑 All Columns"],
                horizontal=True,
                key="igm_table_view_mode"
            )

        if "Executive" in tv_mode:
            disp_cols = [
                "Rank_Today", "Industry_Group", "Dual_Badge", "Horizon_Badge", "Rotation_Status",
                "Sortino_3M", "Apex_Leaders", "Pack_Hunting_Count",
                "Return_1M", "Top_Leaders_Display", "Sparkline_1M"
            ]
        elif "Risk" in tv_mode:
            disp_cols = [
                "Rank_Today", "Industry_Group", "Dual_Badge", "Horizon_Badge", "Horizon_State",
                "Excess_3M", "Sortino_3M", "Sortino_6M", "Apex_Leaders",
                "Pack_Hunting_Count", "Stock_Count", "Return_3M", "Return_6M",
                "Top_Leaders_Display"
            ]
        elif "Velocity" in tv_mode:
            disp_cols = [
                "Rank_Today", "Industry_Group", "Dual_Badge", "Horizon_Badge", "Horizon_State", "Rotation_Status",
                "Rank_1W", "Delta_1W", "Rank_1M", "Delta_1M", "Rank_3M", "Rank_6M",
                "Pack_Hunting_Count", "Sparkline_1M"
            ]
        else:
            disp_cols = [
                "Rank_Today", "Industry_Group", "Dual_Badge", "Horizon_Badge", "Horizon_State", "Rotation_Status",
                "Excess_1M", "Excess_3M", "Excess_6M",
                "Sortino_3M", "Sortino_6M", "Apex_Leaders",
                "Rank_1W", "Delta_1W", "Rank_1M", "Delta_1M", "Rank_3M", "Rank_6M",
                "Pack_Hunting_Count", "Return_1M", "Return_3M", "Return_6M",
                "Top_Leaders_Display", "Sparkline_1M"
            ]

        display_df = filtered_df[disp_cols].copy()

        master_col_cfg = {
            "Rank_Today": st.column_config.NumberColumn("Rank", format="%d", width=70, help="Current Relative Strength Rank (1 = Market Leader)"),
            "Industry_Group": st.column_config.TextColumn("Industry Sub-Group", width=250),
            "Dual_Badge": st.column_config.TextColumn("Dual Gate", width=95, help="Gary Antonacci Dual Momentum Gate: 🚀 Alpha (Excess>0 & Return>0), 🛡️ Mirage (Excess>0 but Return<=0), 📉 Bleed (Excess<=0 & Return<=0)"),
            "Horizon_Badge": st.column_config.TextColumn("3H Alignment", width=95, help="3-Horizon relative strength agreement: 1M (Short) · 3M (Med) · 6M (Long) vs Benchmark (🟢 Outperform · 🟡 In-Line · 🔴 Lag)"),
            "Horizon_State": st.column_config.TextColumn("Alignment State", width=140, help="Classified horizon state: 🟢 Confluence (all 3 leading), 🌱 Emerging Turn (short-term turning up), ⚠️ Fading (short-term losing momentum), 🔴 Chronic Laggard"),
            "Excess_1M": st.column_config.NumberColumn("Excess 1M", format="%+.1f%%", width=90, help="Group 1M return minus benchmark return (% pts)"),
            "Excess_3M": st.column_config.NumberColumn("Excess 3M", format="%+.1f%%", width=90, help="Group 3M return minus benchmark return (% pts)"),
            "Excess_6M": st.column_config.NumberColumn("Excess 6M", format="%+.1f%%", width=90, help="Group 6M return minus benchmark return (% pts)"),
            "Rotation_Status": st.column_config.TextColumn("Rotation State", width=120),
            "Sortino_3M": st.column_config.NumberColumn("Sortino 3M", format="%.2f", width=95, help="3-Month Downside-adjusted Sortino ratio (MAR=6.5%)"),
            "Sortino_6M": st.column_config.NumberColumn("Sortino 6M", format="%.2f", width=95, help="6-Month Downside-adjusted Sortino ratio (MAR=6.5%)"),
            "Apex_Leaders": st.column_config.NumberColumn("👑 Apex", format="%d", width=90, help="Count of constituents with RS >= 80 AND Sortino 3M >= 3.0"),
            "Rank_1W": st.column_config.NumberColumn("1W Ago", format="%d", width=80),
            "Delta_1W": st.column_config.NumberColumn("Δ 1W", format="%+d", width=75, help="Change in rank over 1 week (positive = improving)"),
            "Rank_1M": st.column_config.NumberColumn("1M Ago", format="%d", width=80),
            "Delta_1M": st.column_config.NumberColumn("Δ 1M", format="%+d", width=75, help="Change in rank over 1 month (positive = accelerating)"),
            "Rank_3M": st.column_config.NumberColumn("3M Ago", format="%d", width=80),
            "Rank_6M": st.column_config.NumberColumn("6M Ago", format="%d", width=80),
            "Pack_Hunting_Count": st.column_config.NumberColumn("🐺 Pack (RS≥80)", format="%d", width=110, help="Count of stocks in group with RS >= 80"),
            "Stock_Count": st.column_config.NumberColumn("Total Stocks", format="%d", width=90, help="Total active constituents in this group"),
            "Return_1M": st.column_config.NumberColumn("1M %", format="%.1f%%", width=85),
            "Return_3M": st.column_config.NumberColumn("3M %", format="%.1f%%", width=85),
            "Return_6M": st.column_config.NumberColumn("6M %", format="%.1f%%", width=85),
            "Top_Leaders_Display": st.column_config.TextColumn("Top 3 Anchor Leaders & Live Action State", width=380),
            "Sparkline_1M": st.column_config.LineChartColumn("1M Trend", width=120, help="Normalized daily price trend over trailing 21 trading days")
        }

        col_cfg = {k: v for k, v in master_col_cfg.items() if k in display_df.columns}

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
                st.session_state["igm_selected_group"] = selected_table_group
                st.session_state[sel_widget_key] = selected_table_group

    # -------------------------------------------------------------
    # TAB: TOP 40 LEADERS VS WORST 40 LAGGARDS (6-WEEK TRAJECTORY)
    # -------------------------------------------------------------
    with tab_top_worst:
        top_40_df, worst_40_df = get_top_and_worst_40_groups(df_matrix)
        
        st.markdown(clean_html("""
        <div style='background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(2, 6, 23, 0.95) 100%);
                    border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;'>
            <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;'>
                <div>
                    <h4 style='margin:0; font-size:1.15rem; font-weight:800; color:#f8fafc;'>
                        🏆 IBD 197 Industry Groups: 6-Week Momentum Trajectory (Daily GMI Standard)
                    </h4>
                    <p style='margin:4px 0 0 0; font-size:0.82rem; color:#94a3b8;'>
                        Institutional money rotates systematically into top-performing industries. The canonical CANSLIM rule dictates: 
                        <b>concentrate long exposure exclusively in the Top 40 leading groups</b> and strictly avoid or short the Bottom 40 laggards.
                    </p>
                </div>
                <div style='display:flex; gap:12px;'>
                    <span style='background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.4); color:#34d399; padding:4px 10px; border-radius:6px; font-size:0.78rem; font-weight:700;'>
                        Top 40 = Institutional Accumulation
                    </span>
                    <span style='background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.4); color:#f87171; padding:4px 10px; border-radius:6px; font-size:0.78rem; font-weight:700;'>
                        Worst 40 = Chronic Distribution
                    </span>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)
        
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            t40_rs_med = top_40_df['Comp_RS'].median() if not top_40_df.empty else 0
            st.metric("Top 40 Median RS", f"{t40_rs_med:.0f}", delta="Leadership Tier", delta_color="normal")
        with m2:
            t40_apex_sum = int(top_40_df['Apex_Leaders'].sum()) if not top_40_df.empty else 0
            st.metric("Top 40 Apex Leaders", f"{t40_apex_sum} 👑", help="Constituents with RS ≥ 80 & Sortino 3M ≥ 3.0")
        with m3:
            w40_rs_med = worst_40_df['Comp_RS'].median() if not worst_40_df.empty else 0
            st.metric("Worst 40 Median RS", f"{w40_rs_med:.0f}", delta="Laggard Tier", delta_color="inverse")
        with m4:
            w40_apex_sum = int(worst_40_df['Apex_Leaders'].sum()) if not worst_40_df.empty else 0
            st.metric("Worst 40 Apex Leaders", f"{w40_apex_sum} 👑", help="Laggard groups typically have near-zero Apex compounders")
            
        st.write("")
        
        # Columns configuration for IBD tables
        col_cfg_ibd = {
            "Rank_Today": st.column_config.NumberColumn("Rank", format="%d", width=65, help="Current Group RS Rank"),
            "Horizon_Badge": st.column_config.TextColumn("3H", width=75, help="3-Horizon Alignment (1M·3M·6M)"),
            "Rank_3W": st.column_config.NumberColumn("3W", format="%d", width=65, help="Rank 3 Weeks Ago"),
            "Rank_6W": st.column_config.NumberColumn("6W", format="%d", width=65, help="Rank 6 Weeks Ago"),
            "Delta_6W": st.column_config.NumberColumn("Δ 6W", format="%+d", width=65, help="6-Week Rank Velocity (positive = climbing ranks)"),
            "Industry_Group": st.column_config.TextColumn("Industry Group", width=220),
            "Comp_RS": st.column_config.NumberColumn("RS", format="%d", width=60, help="Group RS (0-99)"),
            "Return_1D": st.column_config.NumberColumn("1D %", format="%+.1f%%", width=75),
            "Return_YTD": st.column_config.NumberColumn("YTD %", format="%+.1f%%", width=75),
            "Apex_Leaders": st.column_config.NumberColumn("👑", format="%d", width=55, help="Apex Leaders Count"),
            "Top_Leaders_Display": st.column_config.TextColumn("Leading Anchor Stocks", width=260)
        }
        
        c_lead, c_lagg = st.columns(2)
        
        with c_lead:
            st.markdown("#### 🏆 Top 40 Leading Industry Groups")
            st.caption("Click any group to inspect its constituents and pivot chart below.")
            
            t40_disp_cols = [c for c in ["Rank_Today", "Horizon_Badge", "Rank_3W", "Rank_6W", "Delta_6W", "Industry_Group", "Comp_RS", "Return_1D", "Return_YTD", "Apex_Leaders", "Top_Leaders_Display"] if c in top_40_df.columns]
            t40_disp = top_40_df[t40_disp_cols].copy()
            
            event_top = st.dataframe(
                t40_disp,
                column_config={k: v for k, v in col_cfg_ibd.items() if k in t40_disp.columns},
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="ibd_top_40_table",
                height=650
            )
            if event_top and event_top.selection and event_top.selection.rows:
                sel_idx = event_top.selection.rows[0]
                if 0 <= sel_idx < len(t40_disp):
                    grp_name = t40_disp.iloc[sel_idx]["Industry_Group"]
                    st.session_state["igm_selected_group"] = grp_name
                    st.session_state[sel_widget_key] = grp_name

        with c_lagg:
            st.markdown("#### ⚠️ Bottom 40 Lagging Industry Groups")
            st.caption("Groups suffering persistent institutional selling and rank decay.")
            
            w40_disp_cols = [c for c in ["Rank_Today", "Horizon_Badge", "Rank_3W", "Rank_6W", "Delta_6W", "Industry_Group", "Comp_RS", "Return_1D", "Return_YTD", "Apex_Leaders", "Top_Leaders_Display"] if c in worst_40_df.columns]
            w40_disp = worst_40_df[w40_disp_cols].copy()
            
            event_worst = st.dataframe(
                w40_disp,
                column_config={k: v for k, v in col_cfg_ibd.items() if k in w40_disp.columns},
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="ibd_worst_40_table",
                height=650
            )
            if event_worst and event_worst.selection and event_worst.selection.rows:
                sel_idx = event_worst.selection.rows[0]
                if 0 <= sel_idx < len(w40_disp):
                    grp_name = w40_disp.iloc[sel_idx]["Industry_Group"]
                    st.session_state["igm_selected_group"] = grp_name
                    st.session_state[sel_widget_key] = grp_name

    # -------------------------------------------------------------
    # TAB 3: ACTIONABLE LEADER CARDS GRID
    # -------------------------------------------------------------
    with tab_cards:
        c_cinfo, c_csort, c_cscope = st.columns([1.6, 1.4, 1.0])
        with c_cinfo:
            st.caption("⚡ *High-density visual terminal cards highlighting the top actionable anchor stocks and live execution badges for every group.*")
        with c_csort:
            card_sort_choice = st.selectbox(
                "Sort Cards By",
                [
                    "Same as Top Filter",
                    "🏆 Hierarchy Rank (#1 to #N)",
                    "🛡️ Sortino 3M (High to Low)",
                    "👑 Apex Leaders Count (High to Low)",
                    "🐺 Wolfpack Breadth (RS ≥ 80 Count)",
                    "🐺 Wolfpack Concentration (% RS ≥ 80)",
                    "🚀 1M Rank Velocity (Δ Spots)",
                    "📈 1M Momentum Return %"
                ],
                index=0,
                key="actionable_cards_sort_choice",
                help="Sort cards by momentum rank, wolfpack breadth, or velocity"
            )
        with c_cscope:
            card_count_choice = st.selectbox(
                "Show Cards",
                ["Top 18 Cards", "Top 36 Cards", "All Filtered Groups"],
                index=1,
                key="actionable_cards_count_choice"
            )
            
        effective_card_sort = sort_choice if card_sort_choice == "Same as Top Filter" else card_sort_choice
        cards_df = filtered_df.copy()
        
        if "Sortino 3M" in effective_card_sort:
            cards_df = cards_df.sort_values(["Sortino_3M", "Rank_Today"], ascending=[False, True])
        elif "Apex Leaders" in effective_card_sort:
            cards_df = cards_df.sort_values(["Apex_Leaders", "Rank_Today"], ascending=[False, True])
        elif "Wolfpack Breadth" in effective_card_sort or "RS ≥ 80 Count" in effective_card_sort:
            cards_df = cards_df.sort_values(["Pack_Hunting_Count", "Rank_Today"], ascending=[False, True])
        elif "Concentration" in effective_card_sort:
            cards_df["_pack_ratio"] = cards_df["Pack_Hunting_Count"] / cards_df["Stock_Count"].clip(lower=1)
            cards_df = cards_df.sort_values(["_pack_ratio", "Pack_Hunting_Count", "Rank_Today"], ascending=[False, False, True])
        elif "Velocity" in effective_card_sort:
            cards_df = cards_df.sort_values(["Delta_1M", "Rank_Today"], ascending=[False, True])
        elif "Return" in effective_card_sort:
            cards_df = cards_df.sort_values(["Return_1M", "Rank_Today"], ascending=[False, True])
        else:
            cards_df = cards_df.sort_values("Rank_Today", ascending=True)

        if card_count_choice == "Top 18 Cards":
            cards_slice = cards_df.head(18)
        elif card_count_choice == "Top 36 Cards":
            cards_slice = cards_df.head(36)
        else:
            cards_slice = cards_df

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
                    dual_pill_cls = "pill-emerald" if g_data.get('Is_Dual_Alpha') else (
                        "pill-amber" if g_data.get('Is_Relative_Mirage') else "pill-rose"
                    )
                    
                    chips_html_list = []
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
                            
                        exchange = 'BSE' if str(ldr.get('raw_ticker', '')).endswith('.BO') else 'NSE'
                        tv_link = f"https://www.tradingview.com/chart/?symbol={exchange}%3A{ldr['ticker']}"
                        s_val = ldr.get('sortino_3m', 0.0)
                        s_color = "#34d399" if s_val >= 3.0 else ("#38bdf8" if s_val >= 1.5 else "#94a3b8")
                        
                        chips_html_list.append(f"""
                        <div class='anchor-chip'>
                            <span class='anchor-sym'>
                                <a href='{tv_link}' target='_blank' style='color:#38bdf8; text-decoration:none;'>{ldr['ticker']} ↗</a>
                                <span style='font-size:0.72rem; color:#94a3b8; margin-left:4px;'>RS {ldr['rs']}</span>
                                <span style='font-size:0.70rem; color:{s_color}; font-weight:700; margin-left:4px;' title='3M Sortino: {s_val:.2f}'>S {s_val:.1f}</span>
                            </span>
                            <span style='color:#f8fafc; font-weight:700;'>₹{ldr['cmp']:,.1f}</span>
                            <span class='anchor-badge {b_cls}'>{b_text}</span>
                        </div>
                        """)
                        
                    chips_combined = "".join(chips_html_list)
                    
                    card_html = f"""
                    <div class='terminal-group-card'>
                        <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                            <div>
                                <div class='tg-title' title='{g_data["Industry_Group"]}'>{g_data["Industry_Group"]}</div>
                                <div class='tg-stock-count'>{g_data["Stock_Count"]} Stocks • <b style='color:#c084fc;'>🐺 {g_data["Pack_Hunting_Count"]} RS≥80</b> • <b style='color:#38bdf8;'>🛡️ S(3M) {g_data.get("Sortino_3M", 0.0):.2f}</b> • <b style='color:#34d399;'>👑 {g_data.get("Apex_Leaders", 0)} Apex</b></div>
                            </div>
                            <div style='text-align:right;'>
                                <div class='tg-rank-num'>#{g_data["Rank_Today"]}</div>
                                <div style='display:flex; gap:4px; justify-content:flex-end; margin-top:2px; flex-wrap:wrap;'>
                                    <span class='badge-pill {dual_pill_cls}' title='Dual Momentum: {g_data.get("Dual_State", "")}'>{g_data.get("Dual_Badge", "")}</span>
                                    <span class='badge-pill' style='background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.15); font-family:monospace;' title='3-Horizon (1M · 3M · 6M): {g_data.get("Horizon_State", "")}'>{g_data.get("Horizon_Badge", "")}</span>
                                    <span class='badge-pill pill-purple'>🐺 {g_data["Pack_Hunting_Count"]} Pack</span>
                                    <span class='badge-pill {rot_pill}'>{g_data["Rotation_Status"]}</span>
                                </div>
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
                            {chips_combined}
                        </div>
                    </div>
                    """
                    st.markdown(clean_html(card_html), unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 3: ROTATION VELOCITY QUADRANTS (SCATTER PLOT)
    # -------------------------------------------------------------
    with tab_quad:
        st.markdown("##### 🌪️ Institutional Rotation Quadrants (Strength vs 1M Velocity)")
        
        # Explicit user control & banner answering: "what does the size of bubble mean on rotation quadrant"
        c_size_toggle, c_size_desc = st.columns([1.8, 2.2])
        with c_size_toggle:
            size_mode = st.radio(
                "Bubble Size Represents",
                [
                    "🐺 Pack-Hunting Breadth (Stocks RS ≥ 80)",
                    "👑 Apex Leaders (RS ≥ 80 & Sortino ≥ 3.0)",
                    "🏢 Universe Stock Count (Total Constituents)"
                ],
                index=0,
                horizontal=True,
                help="Choose what diameter/size represents on the chart."
            )
            
        if "Apex Leaders" in size_mode:
            size_col = "Apex_Leaders"
            size_metric_name = "👑 Apex Leaders (RS ≥ 80 & S3M ≥ 3.0)"
            size_desc_text = "💡 **Bubble Size = Number of Apex Leaders**. High-diameter bubbles reveal groups packed with low-drawdown structural compounders."
        elif "Pack-Hunting" in size_mode:
            size_col = "Pack_Hunting_Count"
            size_metric_name = "🐺 Pack-Hunting Breadth (RS ≥ 80)"
            size_desc_text = "💡 **Bubble Size = Number of Stocks with RS ≥ 80**. Larger bubbles highlight groups experiencing coordinated multi-stock institutional pack accumulation."
        else:
            size_col = "Stock_Count"
            size_metric_name = "🏢 Total Constituent Count"
            size_desc_text = "💡 **Bubble Size = Total Constituent Stock Count** (universe breadth of the industry group)."

        with c_size_desc:
            st.info(size_desc_text)

        plot_df = filtered_df.copy()
        plot_df["Bubble_Size"] = plot_df[size_col].clip(lower=1)
        
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
            size="Bubble_Size",
            size_max=36,
            color="Rotation_Status",
            color_discrete_map=color_map,
            hover_name="Industry_Group",
            hover_data={
                "Rank_Today": True,
                "Delta_1M": True,
                "Sortino_3M": ":.2f",
                "Sortino_6M": ":.2f",
                "Apex_Leaders": True,
                "Pack_Hunting_Count": True,
                "Stock_Count": True,
                "Return_1M": ":.1f%",
                "Return_6M": ":.1f%",
                "Bubble_Size": False
            },
            labels={
                "Delta_1M": "1-Month Rank Velocity (Δ Spots Gained/Lost)",
                "Comp_RS": "Composite Relative Strength (0 to 99 Percentile)",
                "Rotation_Status": "Rotation State",
                "Sortino_3M": "Sortino (3M)",
                "Sortino_6M": "Sortino (6M)",
                "Apex_Leaders": "👑 Apex Leaders",
                "Pack_Hunting_Count": "Pack (RS≥80)",
                "Stock_Count": "Total Stocks"
            }
        )
        
        # Zero line crosshairs
        fig_quad.add_vline(x=0, line_width=1, line_dash="dash", line_color="#475569")
        fig_quad.add_hline(y=50, line_width=1, line_dash="dash", line_color="#475569")
        
        max_x = max(abs(plot_df["Delta_1M"].max()), 10)
        min_x = min(plot_df["Delta_1M"].min(), -10)
        
        fig_quad.add_annotation(x=max_x*0.65, y=94, text="👑 ACCELERATING LEADERS", showarrow=False, font=dict(color="#34d399", size=11, family="JetBrains Mono"))
        fig_quad.add_annotation(x=max_x*0.65, y=20, text="🔄 STEALTH ACCUMULATION", showarrow=False, font=dict(color="#38bdf8", size=11, family="JetBrains Mono"))
        fig_quad.add_annotation(x=min_x*0.65, y=94, text="⏸️ DIGESTING LEADERS", showarrow=False, font=dict(color="#fbbf24", size=11, family="JetBrains Mono"))
        fig_quad.add_annotation(x=min_x*0.65, y=20, text="⚠️ DISTRIBUTION VECTOR", showarrow=False, font=dict(color="#f87171", size=11, family="JetBrains Mono"))

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

        ### 🧮 How Relative Strength (RS) is Calculated
        The platform computes **two distinct tiers of Relative Strength (RS)**:
        
        #### 1. Constituent Stock RS Rating (0 to 99 Percentile)
        Every stock in the ~2,500+ Indian equity universe is scored using William J. O'Neil's quarterly-weighted performance algorithm:
        
        $$\text{Stock Performance Score} = 0.40 \times R_{\text{3M}} + 0.30 \times R_{\text{6M}} + 0.30 \times R_{\text{1Y}}$$
        
        - **$R_{\text{3M}}$ (63-Day Return, 40% Weight)**: The most recent quarter carries the heaviest weighting because institutional accumulation manifests as rapid recent acceleration.
        - **$R_{\text{6M}}$ (126-Day Return, 30% Weight)**: Confirms intermediate momentum and trend sustainability.
        - **$R_{\text{1Y}}$ (250-Day Return, 30% Weight)**: Validates primary structural stage-2 uptrend over the trailing 12 months.
        
        This raw performance score is converted into a **Percentile Rank from 0 to 99** across the entire universe:
        $$\text{Stock RS Rating} = \text{Percentile}(\text{Stock Performance Score}) \times 99$$
        
        *Example*: A stock with an **RS Rating of 94** has outperformed 94% of all listed companies in the entire Indian market over the past year. In CANSLIM, we exclusively target true market leaders with **RS $\ge 80$** (the top 20% of the market).

        #### 2. Industry Group Composite RS & Momentum Score
        For each industry group and thematic cluster:
        1. **Synthetic Group Price Index**: We construct an equal-weighted daily synthetic equity curve by normalizing all constituent stock prices to a base of 1.0 at inception:
           $$\text{Index}_{\text{Group}}(t) = \frac{1}{N} \sum_{i=1}^N \frac{P_{i}(t)}{P_{i}(t_0)}$$
        2. **Multi-Horizon Momentum Score**: Calculated across 4 strategic cycle horizons:
           $$\text{Group Momentum Score} = 0.40 \times R_{\text{3M}} + 0.30 \times R_{\text{6M}} + 0.20 \times R_{\text{1M}} + 0.10 \times R_{\text{1W}}$$
           - **40% (3-Month / 63d)**: The core quarterly institutional allocation cycle.
           - **30% (6-Month / 126d)**: Secular trend anchor that filters short squeezes from genuine multi-quarter themes.
           - **20% (1-Month / 21d)**: Tactical swing acceleration and early rotation detection.
           - **10% (1-Week / 5d)**: Immediate short-term velocity.
        3. **Group Composite RS Percentile (0 to 99)**:
           $$\text{Group Composite RS} = \text{Percentile}(\text{Group Momentum Score}) \times 99$$
           This composite score is what appears on the Y-Axis of the **Rotation Velocity Quadrants** and powers group hierarchy ranking.

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

        ---

        ### 🛡️ Dual-Horizon Sortino Ratio & Apex Leader Framework
        While standard CANSLIM Relative Strength measures gross price appreciation, institutional hedge funds and quants evaluate **Downside-Adjusted Asymmetry**. The traditional Sharpe ratio penalizes explosive upside volatility; the **Sortino Ratio** isolates and penalizes only harmful downside variance below the hurdle rate:

        #### 1. Downside Deviation ($\delta_{\text{down}}$)
        $$\delta_{\text{down}} = \sqrt{\frac{1}{N} \sum_{t=1}^{N} \min(0, R_t - \text{MAR}_{\text{daily}})^2} \times \sqrt{252}$$
        - **Hurdle Rate ($\text{MAR}$)**: Set to $6.5\%$ for Indian equities, reflecting the risk-free overnight repo benchmark.

        #### 2. Dual-Horizon Deployment
        - **3-Month Tactical Sortino ($N = 63$ trading days)**: Detects immediate acceleration quality and identifies smooth momentum without deep pullbacks.
        - **6-Month Structural Sortino ($N = 126$ trading days)**: Evaluates durability across full base-building consolidations (~45–60 down days).

        #### 3. 👑 Apex Leaders Defined
        An **Apex Leader** is an elite constituent meeting both strict institutional criteria:
        $$\text{RS Rating} \ge 80 \quad \text{AND} \quad \text{Sortino 3M} \ge 3.0$$
        Groups with high Apex Leader density represent the highest-quality compounder sectors in the entire market.

        ---

        ### 🌐 Where Is The Strength? — Participation Ratio & 3-Horizon Alignment
        *Inspired by the institutional frameworks of Atul Suri (Marathon Trends) and long-horizon relative-momentum studies.*

        #### 1. The Participation Ratio (Market Breadth Barometer)
        Rather than looking solely at broad index levels, institutional trend-followers measure what percentage of industry groups are outperforming the broad benchmark (e.g. NIFTY 500 / S&P 500):
        - **Broad ($\ge 55\%$)**: Broad market participation. Multiple engines of growth are firing simultaneously. Heavy institutional risk-on environment.
        - **Selective ($40\%\text{--}54\%$)**: Divergence is occurring. Capital is rotating into a subset of leading themes while broader equities stall.
        - **Narrow ($< 40\%$)**: Warning regime. Only a handful of defensive or high-conviction clusters are holding up. In this regime, alpha requires strict concentration in Top Groups with strong confluence.
        - **Breadth Drift**: Compares Short-Term (1M) vs Medium-Term (3M) participation. A positive drift ($\ge +5\%$) signals an expanding rally (*Broadening*), while a negative drift ($\le -5\%$) signals deteriorating underlying breadth (*Narrowing*).

        #### 2. The 3-Horizon Alignment Badge (`Short · Medium · Long`)
        Every industry group is scored across three core institutional allocation windows:
        - **1M (21 Trading Days)**: Short-term swing momentum & early rotation detection.
        - **3M (63 Trading Days)**: Medium-term quarterly earnings & institutional positioning cycle.
        - **6M (126 Trading Days)**: Long-term structural trend anchor.

        Each horizon displays:
        - 🟢 = Outperforming Benchmark
        - 🟡 = In-Line with Benchmark (within flat band)
        - 🔴 = Lagging Benchmark

        **Classified Alignment States**:
        - **🟢 Confluence (`🟢🟢🟢`)**: Trend fully intact across all three time horizons. Prime hunting ground for Stage-2 base breakouts.
        - **🌱 Emerging Turn (`🟢🟡🔴` or `🟢🔴🔴`)**: Short-term turning up while intermediate/long-term are still lagging. Signals early rotation / bottoming theme.
        - **⚠️ Fading (`🔴🟢🟢` or `🔴🟡🟢`)**: Short-term rolling over while long-term was strong. Signals momentum exhaustion or healthy pullback.
        - **🔴 Chronic Laggard (`🔴🔴🔴`)**: Underperforming across all horizons. Strict avoid.
        - **🔀 Mixed**: Transitional divergence.
        """)

    # -------------------------------------------------------------
    # 6. CONSTITUENT DEEP-DIVE DRILLDOWN SECTION
    # -------------------------------------------------------------
    st.markdown("---")
    
    # Section Header
    st.markdown(clean_html("""
    <div style='margin-bottom:14px;'>
        <h3 style='margin:0; font-size:1.30rem; font-weight:800; color:#f8fafc; letter-spacing:-0.02em;'>
            🔬 Constituent Deep-Dive & Actionable Setup Inspector
        </h3>
        <p style='margin:4px 0 0 0; font-size:0.80rem; color:#94a3b8;'>
            Granular multi-stock execution terminal, pivot proximity radar, and benchmark alpha curve for the chosen industry group.
        </p>
    </div>
    """), unsafe_allow_html=True)
    
    # Options in the exact sorted and filtered order of the user's active view
    group_options = filtered_df["Industry_Group"].tolist() if not filtered_df.empty else df_matrix["Industry_Group"].tolist()

    # Determine currently selected group (prioritizing table clicks, top sort changes, or manual selectbox picks)
    target_group = st.session_state.get(sel_widget_key, st.session_state.get("igm_selected_group"))
    if not target_group or target_group not in group_options:
        target_group = group_options[0]
    
    st.session_state["igm_selected_group"] = target_group
    st.session_state[sel_widget_key] = target_group
    default_index = group_options.index(target_group)

    def format_group_label(grp_name):
        match = df_matrix[df_matrix["Industry_Group"] == grp_name]
        if not match.empty:
            r = match.iloc[0]
            s3_val = r.get('Sortino_3M', 0.0)
            apex_val = r.get('Apex_Leaders', 0)
            return f"#{r['Rank_Today']}  {grp_name}  (👑 {apex_val} Apex • 🛡️ S {s3_val:.2f} • {r['Stock_Count']} Stocks • {r['Rotation_Status']})"
        return grp_name

    chosen_group = st.selectbox(
        "Select Industry Group to Drill Down",
        options=group_options,
        index=default_index,
        key=sel_widget_key,
        format_func=format_group_label,
        help="Select any group to inspect its constituent stocks, pivots, and synthetic equity curve"
    )
    st.session_state["igm_selected_group"] = chosen_group

    if chosen_group:
        df_constits, curve_df, tv_copy_box = get_group_deep_dive_data(chosen_group, taxonomy=tax_key)
        grp_row = df_matrix[df_matrix["Industry_Group"] == chosen_group].iloc[0]

        # Top Group Metrics Strip using Terminal Glass HUD Panels
        r_delta_class = "pill-emerald" if grp_row['Delta_1M'] >= 0 else "pill-rose"
        r_delta_sign = "+" if grp_row['Delta_1M'] >= 0 else ""
        r_1w_sign = "+" if grp_row['Delta_1W'] >= 0 else ""
        
        rot_panel_class = "accel" if "Surging" in grp_row['Rotation_Status'] else (
            "pack" if "Accumulating" in grp_row['Rotation_Status'] else (
                "dist" if "Distributing" in grp_row['Rotation_Status'] else ""
            )
        )
        
        pack_pct = int((grp_row['Pack_Hunting_Count'] / max(1, grp_row['Stock_Count'])) * 100)
        c6m_class = "pill-emerald" if grp_row['Return_6M'] >= 0 else "pill-rose"
        
        hud_html = f"""
        <div class='hud-grid' style='margin-top:8px; margin-bottom:20px;'>
            <div class='hud-panel'>
                <div class='hud-panel-title'>🏆 Group Hierarchy Rank</div>
                <div class='hud-panel-val'>#{grp_row['Rank_Today']} <span style='font-size:0.85rem; color:#64748b; font-weight:500;'>of {len(df_matrix)}</span></div>
                <div class='hud-panel-sub'>
                    <span class='badge-pill {r_delta_class}'>▲ {r_delta_sign}{grp_row['Delta_1M']} in 1M</span>
                    <span style='color:#64748b;'>• 1W: {r_1w_sign}{grp_row['Delta_1W']}</span>
                </div>
            </div>
            <div class='hud-panel {rot_panel_class}'>
                <div class='hud-panel-title'>🌪️ Rotation & 3H Alignment</div>
                <div class='hud-panel-val' style='font-size:1.15rem; display:flex; align-items:center; gap:8px;'>
                    <span>{grp_row['Rotation_Status']}</span>
                    <span class='badge-pill' style='background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.18); font-size:0.75rem;'>{grp_row.get('Horizon_Badge', '')} {grp_row.get('Horizon_State', '')}</span>
                </div>
                <div class='hud-panel-sub'>
                    <span class='badge-pill pill-cyan'>Velocity: {grp_row['Delta_1M']:+d} spots</span>
                    <span style='color:#94a3b8;'>Composite RS: <b>{grp_row['Comp_RS']}</b></span>
                </div>
            </div>
            <div class='hud-panel pack'>
                <div class='hud-panel-title'>🐺 Pack Breadth & Apex Density</div>
                <div class='hud-panel-val'>{grp_row['Pack_Hunting_Count']} Stocks <span style='font-size:0.80rem; color:#c084fc; font-weight:600;'>RS ≥ 80</span></div>
                <div class='hud-panel-sub'>
                    <span class='badge-pill pill-purple'>{pack_pct}% of Group</span>
                    <span class='badge-pill pill-emerald'>👑 {grp_row.get('Apex_Leaders', 0)} Apex</span>
                    <span style='color:#94a3b8;'>• {grp_row['Stock_Count']} Stocks</span>
                </div>
            </div>
            <div class='hud-panel'>
                <div class='hud-panel-title'>📈 Excess vs {grp_row.get('Benchmark_Name', 'Index')} & Quality</div>
                <div class='hud-panel-val'>{grp_row.get('Excess_3M', 0.0):+.1f}% <span style='font-size:0.80rem; color:#64748b;'>3M Excess</span></div>
                <div class='hud-panel-sub'>
                    <span class='badge-pill {c6m_class}'>6M: {grp_row.get('Excess_6M', 0.0):+.1f}%</span>
                    <span class='badge-pill pill-cyan'>1M: {grp_row.get('Excess_1M', 0.0):+.1f}%</span>
                    <span style='color:#38bdf8; font-weight:700;'>🛡️ S(3M) {grp_row.get('Sortino_3M', 0.0):.2f}</span>
                </div>
            </div>
        </div>
        """
        st.markdown(clean_html(hud_html), unsafe_allow_html=True)

        # Two columns: Chart on left (60%), TV copy box on right (40%)
        ch_col, tv_col = st.columns([1.8, 1.2])

        with ch_col:
            if not curve_df.empty:
                y_grp_final = curve_df["Industry Group"].iloc[-1] - 100
                y_bnc_final = curve_df["Universe Benchmark"].iloc[-1] - 100
                alpha_val = y_grp_final - y_bnc_final
                alpha_sign = "+" if alpha_val >= 0 else ""
                alpha_color = "#10b981" if alpha_val >= 0 else "#ef4444"
                
                st.markdown(clean_html(f"""
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>
                    <div style='font-weight:700; color:#f8fafc; font-size:0.95rem;'>
                        📈 Synthetic Index: {chosen_group} vs Universe Benchmark (1-Year)
                    </div>
                    <div style='font-size:0.80rem; font-weight:700; font-family:JetBrains Mono; color:{alpha_color};'>
                        Alpha: {alpha_sign}{alpha_val:.1f}% vs Market
                    </div>
                </div>
                """), unsafe_allow_html=True)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=curve_df.index,
                    y=curve_df["Industry Group"],
                    name=chosen_group,
                    mode="lines",
                    line=dict(color="#10b981", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(16, 185, 129, 0.08)",
                    hovertemplate=f"<b>{chosen_group}</b>: %{{y:.1f}} (Base 100)<extra></extra>"
                ))
                fig.add_trace(go.Scatter(
                    x=curve_df.index,
                    y=curve_df["Universe Benchmark"],
                    name="Universe Benchmark",
                    mode="lines",
                    line=dict(color="#94a3b8", width=1.5, dash="dot"),
                    hovertemplate="<b>Universe Benchmark</b>: %{y:.1f}<extra></extra>"
                ))
                
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(15, 23, 42, 0.4)",
                    plot_bgcolor="rgba(15, 23, 42, 0.4)",
                    margin=dict(l=20, r=20, t=15, b=20),
                    height=290,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified",
                    xaxis=dict(
                        showgrid=False,
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1M", step="month", stepmode="backward"),
                                dict(count=3, label="3M", step="month", stepmode="backward"),
                                dict(count=6, label="6M", step="month", stepmode="backward"),
                                dict(count=1, label="1Y", step="year", stepmode="backward"),
                                dict(step="all", label="ALL")
                            ]),
                            bgcolor="rgba(30, 41, 59, 0.8)",
                            activecolor="#10b981",
                            font=dict(color="#f8fafc", size=10)
                        )
                    ),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
                )
                st.plotly_chart(fig, use_container_width=True)

        with tv_col:
            st.markdown(clean_html("""
            <div style='font-weight:700; color:#f8fafc; font-size:0.95rem; margin-bottom:4px;'>
                📋 TradingView 1-Click Watchlist
            </div>
            <div style='font-size:0.75rem; color:#94a3b8; margin-bottom:8px;'>
                Select all & copy into TradingView symbol search or watchlist import modal:
            </div>
            """), unsafe_allow_html=True)
            
            st.text_area(
                "TradingView Symbols",
                value=tv_copy_box,
                height=110,
                label_visibility="collapsed",
                help="Select all and paste directly into TradingView symbol search or watchlist modal."
            )
            
            # Interactive Clickable Quick-Launch Chips
            if not df_constits.empty:
                chip_links = []
                for _, s_row in df_constits.head(8).iterrows():
                    chip_links.append(f"<a href='{s_row['TradingView_URL']}' target='_blank' style='text-decoration:none;'><span class='badge-pill pill-cyan' style='margin:2px;'>{s_row['Symbol']} ↗</span></a>")
                chips_rendered = "".join(chip_links)
                st.markdown(clean_html(f"""
                <div style='margin-top:6px;'>
                    <div style='font-size:0.72rem; color:#64748b; font-weight:700; text-transform:uppercase; margin-bottom:4px;'>Quick Chart Launch:</div>
                    <div style='display:flex; flex-wrap:wrap; gap:4px;'>{chips_rendered}</div>
                </div>
                """), unsafe_allow_html=True)
                
            st.markdown(clean_html(f"""
            <div style='margin-top:10px; font-size:0.75rem; color:#94a3b8; display:flex; gap:12px;'>
                <span>Total Constituents: <b style='color:#f8fafc;'>{len(df_constits)}</b></span>
                <span>•</span>
                <span>🐺 RS ≥ 80 Leaders: <b style='color:#34d399;'>{grp_row['Pack_Hunting_Count']}</b></span>
            </div>
            """), unsafe_allow_html=True)

        # -------------------------------------------------------------
        # Actionable spotlight cards (Show top setups / anchors)
        # -------------------------------------------------------------
        # Prioritize actionable setups (BUY, RETEST, COILING), else show top RS anchors
        act_subset = df_constits[df_constits['Execution Status'].str.contains("BUY|RETEST|COILING")]
        if act_subset.empty:
            act_subset = df_constits.head(4)
            section_title = "🎯 Top Institutional Anchors & Setups in this Group:"
        else:
            act_subset = act_subset.head(4)
            section_title = "🎯 Low-Risk CANSLIM Entry Setups in this Group:"
            
        if not act_subset.empty:
            st.markdown(f"##### {section_title}")
            act_cols = st.columns(min(4, len(act_subset)))
            for a_idx, (_, a_row) in enumerate(act_subset.iterrows()):
                with act_cols[a_idx % len(act_cols)]:
                    b_color = "#10b981" if "BUY" in a_row['Execution Status'] else (
                        "#38bdf8" if "RETEST" in a_row['Execution Status'] else (
                            "#fbbf24" if "COIL" in a_row['Execution Status'] else (
                                "#f87171" if "FAIL" in a_row['Execution Status'] else "#94a3b8"
                            )
                        )
                    )
                    spotlight_html = f"""
                    <div style='background:linear-gradient(135deg, rgba(15,23,42,0.85) 0%, rgba(2,6,23,0.95) 100%); border:1px solid rgba(255,255,255,0.08); border-top:3px solid {b_color}; border-radius:10px; padding:12px 14px; box-shadow:0 6px 18px rgba(0,0,0,0.4);'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <div>
                                <a href='{a_row["TradingView_URL"]}' target='_blank' style='text-decoration:none; font-family:JetBrains Mono; font-size:1.05rem; font-weight:800; color:#f8fafc;'>{a_row["Symbol"]} ↗</a>
                                <span style='font-size:0.70rem; color:#94a3b8; margin-left:6px; font-weight:700;'>RS {a_row["RS Rating"]}</span>
                            </div>
                            <span style='font-size:0.72rem; font-weight:700; color:{b_color}; background:rgba(255,255,255,0.05); padding:2px 6px; border-radius:4px;'>{a_row["Execution Status"]}</span>
                        </div>
                        <div style='display:flex; justify-content:space-between; align-items:baseline; margin-top:8px;'>
                            <span style='font-size:1.15rem; font-weight:800; color:#f8fafc;'>₹{a_row["CMP (₹)"]:,.2f}</span>
                            <span style='font-size:0.75rem; color:#94a3b8;'>Pivot: <b style='color:#f8fafc;'>₹{a_row["Pivot Price"]:,.1f}</b> ({a_row["Dist Pivot %"]:+.1f}%)</span>
                        </div>
                        <div style='display:flex; justify-content:space-between; font-size:0.75rem; color:#64748b; margin-top:6px; border-top:1px solid rgba(255,255,255,0.05); padding-top:6px;'>
                            <span>S(3M): <b style='color:#38bdf8;'>{a_row.get("Sortino 3M", 0.0):.2f}</b></span>
                            <span>21 EMA: <b style='color:#94a3b8;'>{a_row["Dist 21 EMA %"]:+.1f}%</b></span>
                            <span>1M: <b style='color:{"#10b981" if a_row["1M %"]>=0 else "#f87171"};'>{a_row["1M %"]:+.1f}%</b></span>
                            <span>1Y: <b style='color:{"#10b981" if a_row["1Y %"]>=0 else "#f87171"};'>{a_row["1Y %"]:+.1f}%</b></span>
                        </div>
                    </div>
                    """
                    st.markdown(clean_html(spotlight_html), unsafe_allow_html=True)

        # Full constituent table with quick filter
        c_filter_col1, c_filter_col2 = st.columns([1.5, 2.5])
        with c_filter_col1:
            st.markdown(f"##### 📋 Constituents in {chosen_group}")
        with c_filter_col2:
            constit_filter = st.radio(
                "Filter Constituents:",
                ["All Stocks", "👑 Apex Leaders (RS≥80 & S≥3)", "🐺 RS ≥ 80 Only", "🟢 Buy Zone & Retest"],
                horizontal=True,
                key="igm_constit_filter"
            )

        disp_constits_df = df_constits.copy()
        if "Apex Leaders" in constit_filter:
            disp_constits_df = disp_constits_df[(disp_constits_df["RS Rating"] >= 80) & (disp_constits_df["Sortino 3M"] >= 3.0)]
        elif "RS ≥ 80" in constit_filter:
            disp_constits_df = disp_constits_df[disp_constits_df["RS Rating"] >= 80]
        elif "Buy Zone" in constit_filter:
            disp_constits_df = disp_constits_df[disp_constits_df["Execution Status"].str.contains("BUY|RETEST")]

        if len(disp_constits_df) < len(df_constits):
            st.caption(f"Showing **{len(disp_constits_df)} of {len(df_constits)}** constituents matching *{constit_filter}*.")

        disp_constits = disp_constits_df[[
            "Symbol", "CMP (₹)", "1D %", "1W %", "1M %", "3M %", "1Y %",
            "RS Rating", "Sortino 3M", "Sortino 6M", "Pivot Price", "Dist Pivot %", "Dist 21 EMA %",
            "Execution Status", "TradingView_URL"
        ]].copy()

        c_constit_cfg = {
            "Symbol": st.column_config.TextColumn("Symbol", width=100),
            "CMP (₹)": st.column_config.NumberColumn("CMP (₹)", format="₹%.2f", width=110),
            "1D %": st.column_config.NumberColumn("1D %", format="%.2f%%", width=85),
            "1W %": st.column_config.NumberColumn("1W %", format="%.2f%%", width=85),
            "1M %": st.column_config.NumberColumn("1M %", format="%.2f%%", width=85),
            "3M %": st.column_config.NumberColumn("3M %", format="%.2f%%", width=85),
            "1Y %": st.column_config.NumberColumn("1Y %", format="%.2f%%", width=85),
            "RS Rating": st.column_config.ProgressColumn("RS Score", min_value=0, max_value=99, format="%d", width=110),
            "Sortino 3M": st.column_config.NumberColumn("Sortino 3M", format="%.2f", width=95, help="3-Month Downside-adjusted Sortino ratio (MAR=6.5%)"),
            "Sortino 6M": st.column_config.NumberColumn("Sortino 6M", format="%.2f", width=95, help="6-Month Downside-adjusted Sortino ratio (MAR=6.5%)"),
            "Pivot Price": st.column_config.NumberColumn("25D Pivot", format="₹%.2f", width=110),
            "Dist Pivot %": st.column_config.NumberColumn("Dist Pivot", format="%.1f%%", width=100, help="Distance from 25-day pivot"),
            "Dist 21 EMA %": st.column_config.NumberColumn("Dist 21 EMA", format="%.1f%%", width=105, help="Distance from 21-day EMA"),
            "Execution Status": st.column_config.TextColumn("CANSLIM Action State", width=150),
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