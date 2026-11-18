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
    _ige_path = getattr(industry_group_engine, '__file__', None)
    _ige_mtime = os.path.getmtime(_ige_path) if _ige_path and os.path.exists(_ige_path) else None
    if (
        not hasattr(industry_group_engine, 'build_top_n_tv_watchlist')
        or not hasattr(industry_group_engine, '_compute_capped_adtv_and_tradables')
        or (_ige_mtime is not None and getattr(industry_group_engine, '_LOADED_IGE_MTIME', None) != _ige_mtime)
    ):
        importlib.reload(industry_group_engine)
        industry_group_engine._LOADED_IGE_MTIME = _ige_mtime
except Exception:
    pass

try:
    from rrg_engine import render_rrg_sector_matrix_view
except Exception:
    render_rrg_sector_matrix_view = None

from industry_group_engine import (
    compute_industry_group_matrix,
    compute_industry_group_matrix_with_niche,
    load_cached_niche_industry_groups,
    compute_participation,
    compute_dual_momentum_summary,
    get_group_deep_dive_data,
    get_top_and_worst_40_groups,
    INDIAN_ALPHA_THEMES,
    MIN_RANKED_GROUP_SIZE_CANONICAL,
    build_top_n_tv_watchlist,
    format_tradingview_symbol,
    load_exporter_enrichment_maps,
)


def _sort_groups_df(df: pd.DataFrame, sort_choice: str) -> pd.DataFrame:
    """Shared deterministic MarketSurge 2.0 group sorting helper for Table, Cards, and Deep Dive."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "Pure Price-RS" in sort_choice and "Pure_RS_Rank" in out.columns:
        return out.sort_values(["Pure_RS_Rank", "Rank_Today"], ascending=[True, True])
    if "Breadth % RS≥80" in sort_choice and "Breadth_RS80_Pct" in out.columns:
        return out.sort_values(["Breadth_RS80_Pct", "Rank_Today"], ascending=[False, True])
    if "50 SMA" in sort_choice and "Breadth_50SMA_Pct" in out.columns:
        return out.sort_values(["Breadth_50SMA_Pct", "Rank_Today"], ascending=[False, True])
    if "Near 52W High" in sort_choice and "Near_52WH_Pct" in out.columns:
        return out.sort_values(["Near_52WH_Pct", "Rank_Today"], ascending=[False, True])
    if "RS New High %" in sort_choice and "RSNH_Pct" in out.columns:
        tie_cols = ["RSNH_Pct", "RSNH_Count", "Rank_Today"] if "RSNH_Count" in out.columns else ["RSNH_Pct", "Rank_Today"]
        tie_asc = [False, False, True] if "RSNH_Count" in out.columns else [False, True]
        return out.sort_values(tie_cols, ascending=tie_asc)
    if "Excess Return" in sort_choice and "Excess_3M" in out.columns:
        return out.sort_values(["Excess_3M", "Rank_Today"], ascending=[False, True])
    if "Smooth RS" in sort_choice and "Smooth_RS" in out.columns:
        return out.sort_values(["Smooth_RS", "Rank_Today"], ascending=[False, True])
    if "Leader Sortino" in sort_choice and "Leader_Sortino_3M" in out.columns:
        return out.sort_values(["Leader_Sortino_3M", "Rank_Today"], ascending=[False, True])
    if "Sortino 3M" in sort_choice or "Grp Sortino" in sort_choice:
        return out.sort_values(["Sortino_3M", "Rank_Today"], ascending=[False, True])
    if "Apex Leaders" in sort_choice:
        return out.sort_values(["Apex_Leaders", "Rank_Today"], ascending=[False, True])
    if "Wolfpack Breadth" in sort_choice or "RS ≥ 80 Count" in sort_choice:
        return out.sort_values(["Pack_Hunting_Count", "Rank_Today"], ascending=[False, True])
    if "Concentration" in sort_choice:
        denom = out["Tradable_Count"] if "Tradable_Count" in out.columns else out["Stock_Count"]
        out["_pack_ratio"] = out["Pack_Hunting_Count"] / denom.clip(lower=1)
        return out.sort_values(["_pack_ratio", "Pack_Hunting_Count", "Rank_Today"], ascending=[False, False, True])
    if "Velocity" in sort_choice:
        return out.sort_values(["Delta_1M", "Rank_Today"], ascending=[False, True])
    if "Return" in sort_choice:
        return out.sort_values(["Return_1M", "Rank_Today"], ascending=[False, True])
    if "Constituent" in sort_choice:
        cnt_col = "Tradable_Count" if "Tradable_Count" in out.columns else "Stock_Count"
        return out.sort_values([cnt_col, "Rank_Today"], ascending=[False, True])
    return out.sort_values("Rank_Today", ascending=True)


def clean_html(html_str: str) -> str:
    """Strips leading whitespace from every line so Markdown never treats it as code."""
    return "\n".join(line.strip() for line in html_str.splitlines() if line.strip())


def _render_top40_tv_watchlist_exporter(
    df_matrix: pd.DataFrame,
    global_filtered_groups: set,
    u_pfx: str,
    is_us: bool,
    us_tv_map: dict,
    funda_map: dict,
) -> None:
    """Render the CANSLIM & Quant-Hardened Master Top-40 TradingView Batch Exporter."""
    if df_matrix is None or df_matrix.empty:
        return

    max_rank = max(40, int(df_matrix["Rank_Today"].max()))
    adtv_unit = "$M" if is_us else "₹ Cr"

    with st.expander(
        "📋 Master Top-40 TradingView Watchlist Exporter (Dynamic Rank + Leadership + RS + Sortino + Smooth RS)",
        expanded=True,
    ):
        st.caption(
            "All constituent RS scores here reflect the Smoothed 6-Month Multi-Checkpoint RS (21/42/63/95/126d). "
            "Industry Rank locks strictly to O'Neil Hierarchy Rank (#1–#N) independent of visual card sort."
        )

        # Row 1: Group Hierarchy & Cohort Selection
        g1, g2, g3 = st.columns([1.2, 1.3, 1.0])
        with g1:
            rank_range = st.slider(
                "Industry Rank Window (#1 to #N)",
                min_value=1,
                max_value=max_rank,
                value=(1, min(40, max_rank)),
                step=1,
                key=f"tv_batch_rank_rng_{u_pfx}",
                help="William O'Neil rule focuses long exposure strictly on Top 40 ranked industries.",
            )
        with g2:
            leadership_mode = st.selectbox(
                "Leadership Cohort Scope",
                [
                    "Top 3 Leaders Only",
                    "Top 3 + Wolf Pack",
                    "Apex Leaders Only (RS ≥ 80 & S ≥ 3.0)",
                    "Custom Top K per Group",
                    "All Qualifying Constituents",
                ],
                index=0,
                key=f"tv_batch_ldr_mode_{u_pfx}",
                help="Controls how many top tradable leaders per qualifying group are exported (Top 3, +Wolf Pack, Apex RS≥80 & Sortino≥3, Custom Top K, or All Qualifying).",
            )
        with g3:
            grade_filter = st.selectbox(
                "Group Grade Filter",
                ["All in Rank Range", "A+ / A / B Only", "A+ & A Only (Top 20%)"],
                index=0,
                key=f"tv_batch_grade_{u_pfx}",
                help="Filters groups by O'Neil percentile grade within your Rank Window: A+ (Top 10%), A (Top 20%), B (Top 40%).",
            )

        wolf_pack_floor = "Same as Min Stock RS"
        max_pack_extras = 5
        custom_top_k = 5
        if leadership_mode == "Top 3 + Wolf Pack":
            wp1, wp2, _ = st.columns([1.2, 1.2, 1.6])
            wolf_pack_floor = wp1.selectbox(
                "Wolf Pack Extras RS Floor",
                ["Same as Min Stock RS", "Classic Pack (RS ≥ 80)"],
                index=0,
                key=f"tv_batch_wp_floor_{u_pfx}",
                help="Keep extra Pack members at your Min RS floor (default 88) or allow classic RS 80+ pack peers.",
            )
            pack_cap = wp2.slider(
                "Max Wolf Pack Extras / Group (0 = Unlimited)",
                min_value=0,
                max_value=25,
                value=5,
                step=1,
                key=f"tv_batch_wp_cap_{u_pfx}",
                help="Caps additional Wolf Pack peers exported beyond the Top 3 leaders per group (0 = unlimited).",
            )
            max_pack_extras = None if pack_cap == 0 else int(pack_cap)
        elif leadership_mode == "Custom Top K per Group":
            ck1, _ = st.columns([1.2, 2.8])
            custom_top_k = ck1.slider(
                "Top K Qualifying Leaders / Group",
                min_value=1,
                max_value=15,
                value=5,
                step=1,
                key=f"tv_batch_custom_k_{u_pfx}",
                help="Maximum qualifying leaders exported from each industry group when using Custom Top K.",
            )

        # Row 2: Stock-Level Quant Thresholds
        q1, q2, q3, q4 = st.columns(4)
        min_stock_rs = q1.slider(
            "Min Stock RS Rating",
            min_value=0,
            max_value=99,
            value=88,
            step=1,
            key=f"tv_batch_min_rs_{u_pfx}",
            help="Minimum Industry Matrix composite RS rating (0–99) required for an individual constituent to export.",
        )
        min_stock_sortino = q2.slider(
            "Min Stock Sortino (3M)",
            min_value=-2.0,
            max_value=15.0,
            value=1.5,
            step=0.5,
            key=f"tv_batch_min_sortino_{u_pfx}",
            help="Minimum 3-month Sortino ratio (downside-risk-adjusted return) to favor smooth accumulation over choppy spikes.",
        )
        min_stock_smooth_rs = q3.slider(
            "Min Stock Smooth RS (RS/ATR)",
            min_value=-5.0,
            max_value=25.0,
            value=0.0,
            step=0.5,
            key=f"tv_batch_min_srs_{u_pfx}",
            help="Minimum volatility-normalized trend efficiency (RS/ATR). Values ≥ 0 filter for steady right-side uptrends.",
        )
        min_stock_adtv = q4.slider(
            f"Min Stock ADTV ({adtv_unit})",
            min_value=0.0,
            max_value=50.0 if is_us else 25.0,
            value=2.0 if is_us else 1.0,
            step=0.5 if not is_us else 1.0,
            key=f"tv_batch_min_adtv_{u_pfx}",
            help="Block-deal capped 20D turnover gate applied before leadership slicing.",
        )

        # Row 3: Setup, Extension & Fundamentals
        s1, s2, s3, s4 = st.columns([1.2, 0.9, 1.1, 1.1])
        setup_scope = s1.selectbox(
            "Setup Stage Scope",
            [
                "All Non-Failed (Incl. Extended)",
                "Buyable Bases & Coils Only (Hide Extended)",
                "Immediate Buy & Retest Only",
            ],
            index=0,
            key=f"tv_batch_setup_scope_{u_pfx}",
            help="Filters technical execution states: All Non-Failed includes extended leaders; tighter options isolate bases, coils, or immediate buy/retest pivots.",
        )
        max_ema_choice = s2.selectbox(
            "Max Dist 21-EMA %",
            ["No Cap", "≤ +10%", "≤ +15%", "≤ +25%"],
            index=0,
            key=f"tv_batch_max_ema_{u_pfx}",
            help="Caps how far price can sit above its 21-EMA so overly stretched stocks don't crowd actionable setups.",
        )
        max_dist_ema = None if max_ema_choice == "No Cap" else float(max_ema_choice.replace("≤ +", "").replace("%", ""))
        can_growth_gate = s3.selectbox(
            "CANSLIM Earnings / Sales",
            [
                "All (No Fundamental Gate)",
                "Positive EPS (> 0%)",
                "CANSLIM Growth (EPS ≥ 25% or Sales ≥ 25%)",
            ],
            index=0,
            key=f"tv_batch_funda_gate_{u_pfx}",
            help="Optional CANSLIM fundamental filter using cached quarterly growth: Positive EPS (>0%) or ≥25% EPS/Sales acceleration.",
        )
        sort_export_by = s4.selectbox(
            "Export Sort Order",
            [
                "Hierarchy Rank (#1 to #N)",
                "Actionable Setup Priority (BUY → RETEST → COIL)",
            ],
            index=0,
            key=f"tv_batch_sort_order_{u_pfx}",
            help="Orders exported symbols either by O'Neil Group Rank (#1 to #N) or Setup Priority (Buy Zone → Retest → Coiling → Base → Extended).",
        )

        # Row 4: CANSLIM & TradingView Hygiene Guards
        tcols = st.columns(6 if is_us else 5)
        exempt_ipo = tcols[0].checkbox(
            "Exempt IPOs (< 63D)",
            value=True,
            key=f"tv_batch_exempt_ipo_{u_pfx}",
            help="Prevents recent <63-bar high-RS IPOs (default Sortino=0) from being silently dropped.",
        )
        excl_solitary = tcols[1].checkbox(
            "Min ≥ 4 Tradable/Grp",
            value=True,
            key=f"tv_batch_excl_sol_{u_pfx}",
            help="Excludes niche <4-tradable-stock groups from taking Top-40 spots.",
        )
        hide_failed = tcols[2].checkbox(
            "Hide FAILED Setups",
            value=True,
            key=f"tv_batch_hide_fail_{u_pfx}",
            help="Excludes stocks tagged 🔴 FAILED — i.e., close > 2.0% below the 21-EMA OR > 8.0% below base pivot high. Uncheck only if stalking deep pullbacks.",
        )
        excl_circuit = tcols[3].checkbox(
            "Hide Circuit-Locked",
            value=True,
            key=f"tv_batch_excl_circ_{u_pfx}",
            help="Filters stocks with ≥ 4 locked low-range bars in trailing 20 days.",
        )
        sync_global = tcols[4].checkbox(
            "Sync Top Group Filters",
            value=True,
            key=f"tv_batch_sync_global_{u_pfx}",
            help="Intersect with active top-bar Group Smooth RS / Sortino / Apex / Rotation filters.",
        )
        prefix_us = True
        if is_us:
            prefix_us = tcols[5].checkbox(
                "Prefix US Exchanges",
                value=True,
                key=f"tv_batch_prefix_us_{u_pfx}",
                help="Prepends NASDAQ:/NYSE:/AMEX: from Security Master to eliminate TradingView crypto/foreign symbol collisions.",
            )

        a1, a2, _ = st.columns([1.3, 1.5, 2.2])
        dual_alpha_only = a1.checkbox(
            "Dual Alpha Groups Only (3M > 0% & Beat Index)",
            value=False,
            key=f"tv_batch_dual_alpha_{u_pfx}",
            help="Keeps only industry groups with positive 3-month return (> 0%) that also outperform the benchmark index.",
        )
        backfill_top_k = a2.checkbox(
            "Backfill deeper ranks if Top 3 fail quality/setup",
            value=False,
            key=f"tv_batch_backfill_{u_pfx}",
            help="Off (Recommended): strictly evaluates the top 3 tradable leaders so lower-ranked sympathy stocks don't masquerade as Top 3.",
        )

        res = build_top_n_tv_watchlist(
            df_source=df_matrix,
            rank_range=rank_range,
            grade_filter=grade_filter,
            exclude_solitary=excl_solitary,
            dual_alpha_only=dual_alpha_only,
            allowed_groups=global_filtered_groups if sync_global else None,
            leadership_mode=leadership_mode,
            backfill_top_k=backfill_top_k,
            wolf_pack_floor=wolf_pack_floor,
            max_pack_extras=max_pack_extras,
            custom_top_k=custom_top_k,
            min_rs=min_stock_rs,
            min_sortino=min_stock_sortino,
            exempt_short_history_ipo=exempt_ipo,
            min_smooth_rs=min_stock_smooth_rs,
            min_adtv=min_stock_adtv,
            exclude_circuit_locked=excl_circuit,
            hide_failed=hide_failed,
            setup_scope=setup_scope,
            max_dist_ema=max_dist_ema,
            can_growth_gate=can_growth_gate,
            sort_export_by=sort_export_by,
            is_us=is_us,
            prefix_us_exchanges=prefix_us,
            us_tv_map=us_tv_map,
            funda_map=funda_map,
        )

        # Output Telemetry + Export Boxes
        m1, m2, m3, m4 = st.columns(4)
        avg_per_grp = round(res["unique_count"] / max(1, res["groups_matched"]), 1) if res["groups_matched"] else 0.0
        m1.metric("Qualifying Industries", f"{res['groups_matched']} / {res['groups_scanned']}")
        m2.metric("Unique Exported Stocks", f"{res['unique_count']:,}")
        m3.metric("Avg Stocks / Group", f"{avg_per_grp:.1f}")
        m4.metric("Apex Count (RS≥80 & S≥3)", f"{res['apex_count']} 👑")

        if res["ipo_exempt_count"] > 0:
            st.caption(f"🆕 Includes **{res['ipo_exempt_count']}** recent IPO(s) (<63 trading days) protected via Short-History IPO exemption.")
        if res["unique_count"] > 500:
            st.warning(f"⚠️ Export contains {res['unique_count']} tickers. TradingView Free/Plus tiers cap single watchlists (up to 1,000 on Premium) — tighten ADTV or Leadership Scope if needed.")

        if not res["flat_csv"]:
            st.info("No stocks passed all selected Group & Constituent gates. Relax ADTV, Sortino 3M, RS, or Setup Scope slightly.")
            return

        exp_t1, exp_t2, exp_t3 = st.tabs([
            f"📋 Flat Watchlist ({res['unique_count']} Unique)",
            f"🗂️ Sectioned by Industry ({res['section_entry_count']} Entries)",
            f"🔍 Preview Qualifying Stocks ({len(res['rows'])})",
        ])

        with exp_t1:
            st.caption("Click the copy icon in the top-right corner, then in TradingView open any Watchlist → **+** (`Alt+I`) → **Ctrl+V**.")
            st.code(res["flat_csv"], language="text")

        with exp_t2:
            st.caption(
                "Formatted with **`### #Rank Industry`** collapsible section headers. Download `.txt` or copy and use TradingView Watchlist **... → Import list...**."
            )
            st.code(res["sectioned_paste"], language="text")
            mkt_tag = "US" if is_us else "IN"
            st.download_button(
                label=f"⬇️ Download {mkt_tag} Top-{rank_range[1]} TradingView Watchlist (.txt)",
                data=res["sectioned_txt"],
                file_name=f"kush_tracker_{mkt_tag}_top{rank_range[1]}_watchlist.txt",
                mime="text/plain",
                key=f"tv_batch_dl_btn_{u_pfx}",
            )

        with exp_t3:
            prev_df = pd.DataFrame(res["rows"])
            st.dataframe(
                prev_df,
                use_container_width=True,
                hide_index=True,
                height=min(420, 68 + 35 * len(prev_df)),
            )

# =============================================================================
# WORLD-CLASS TERMINAL DESIGN SYSTEM (BLOOMBERG / HEDGE-FUND GRADE)
# =============================================================================
def inject_industry_matrix_css():
    """Injects global styles and the complete Bloomberg-grade CSS design system for Industry Group Matrix."""
    try:
        from styles import load_css
        load_css()
    except Exception:
        pass

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
    max-height: 480px;
    overflow-y: auto;
}
.anchor-chip-row::-webkit-scrollbar {
    width: 4px;
}
.anchor-chip-row::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 4px;
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

def main(default_universe: str = "India (NSE/BSE)"):
    inject_industry_matrix_css()
    try:
        render_disk_cache_sidebar(get_cached_universe)
    except Exception:
        pass
    is_default_us = default_universe.startswith("US")
    u_pfx = "us" if is_default_us else "in"
    render_header(
        "🌊 US Industry Group Momentum Matrix (S&P 500 / Nasdaq)" if is_default_us else "🌊 Industry Group Momentum Matrix (197 Themes)",
        "Cross-Sectional Institutional Rotation, Multi-Timeframe Velocity & Live Execution Setups"
    )

    active_jump_payload = None
    try:
        from leadership_cluster_ui import resolve_igm_jump_payload
        active_jump_payload = resolve_igm_jump_payload(st.session_state, u_pfx)
    except Exception:
        active_jump_payload = None

    # -------------------------------------------------------------
    # 1. MARKET UNIVERSE & CONTROLS (UP FRONT)
    # -------------------------------------------------------------
    c_mkt, c_tax, c_sort, c_rot, c_top, c_search = st.columns([1.3, 1.3, 1.3, 1.1, 0.9, 1.2])

    with c_mkt:
        market_choice = st.selectbox(
            "Market Universe",
            ["🇮🇳 India (NSE/BSE)", "🇺🇸 US (S&P 500 / Nasdaq)"],
            index=1 if is_default_us else 0,
            key=f"igm_market_universe_select_{u_pfx}",
            help="Toggle between Indian Equities (NIFTY 500 Benchmark) and US Equities (S&P 500 Benchmark)."
        )
    universe_scope = "US (S&P 500 / Nasdaq)" if "US" in market_choice else "India (NSE/BSE)"
    is_us = universe_scope.startswith("US")
    curr_symbol = "$" if is_us else "₹"
    
    with c_tax:
        tax_options = ["🏛️ Canonical 145 Sub-Industries"] if is_us else ["🏛️ Canonical 145 Sub-Industries", "⚡ Curated Indian Alpha Themes"]
        taxonomy_choice = st.selectbox(
            "Taxonomy Architecture",
            tax_options,
            index=0,
            key=f"igm_tax_choice_{u_pfx}",
            help="Toggle between standard 145 O'Neil granular groups and high-conviction thematic clusters."
        )
        
    tax_key = "thematic" if "Alpha" in taxonomy_choice else "canonical"
    
    with c_sort:
        sort_choice = st.selectbox(
            "Sort Groups By",
            [
                "🏆 Blended Rank (75% RS / 25% Breadth)",
                "⚡ Pure Price-RS Rank (No Breadth)",
                "🔵 RS New High % (Haber)",
                "📊 Breadth % RS≥80 (High to Low)",
                "📈 >50 SMA Breadth % (High to Low)",
                "🏔️ Near 52W High % (Low Supply)",
                "🌐 3M Excess Return vs Index (High to Low)",
                "⚡ Smooth RS / ATR (High to Low)",
                "🏆 Leader Sortino 3M (High to Low)",
                "🛡️ Grp Sortino 3M (High to Low)",
                "👑 Apex Leaders Count (High to Low)",
                "🐺 Wolfpack Breadth (RS ≥ 80 Count)",
                "🐺 Wolfpack Concentration (% RS ≥ 80)",
                "🚀 1M Rank Velocity (Δ Spots)",
                "📈 1M Momentum Return %",
                "🏢 Total Constituent Count"
            ],
            index=0,
            help="Sort groups across tables and cards by Option B Blended Rank, Pure Price-RS, Haber RSNH%, Breadth %, Smooth RS, or Velocity.",
            key=f"igm_sort_choice_{u_pfx}"
        )

    with c_rot:
        status_filter = st.selectbox(
            "Rotation & Breadth Filter",
            [
                "All Groups",
                "⭐ Top 20% Leading Groups (A+/A) Only",
                "💪 Broad Pack (Breadth ≥ 50%)",
                "✅ Breadth Confirmed (Above Med Breadth & RS ≥ 80)",
                "🏔️ Low Overhead Supply (≥50% Near 52W High)",
                "🔵 RS New High Cluster (≥30% RSNH)",
                "🎯 Pre-Breakout RS Blue Dot (≥20%)",
                "🚫 Avoid D & E Laggards Only",
                "🚀 Dual Alpha Only (RS>0 & Abs>0)",
                "🛡️ Relative Mirage Only (RS>0 & Abs≤0)",
                "🟢 Confluence Only",
                "🌱 Emerging Turn Only",
                "⚠️ Fading Only",
                "🚀 Surging Only",
                "🔄 Accumulating Only",
                "⏸️ Consolidating Only",
                "🐺 Group Confirmed (≥3 Leaders)",
                "🐺 High Pack Hunting (≥3)"
            ],
            index=0,
            key=f"igm_status_filter_{u_pfx}"
        )

    with c_top:
        limit_choice = st.selectbox(
            "Display Scope",
            ["Top 25 Groups", "Top 50 Groups", "Top 100 Groups", "All Groups"],
            index=1,
            key=f"igm_limit_choice_{u_pfx}"
        )

    with c_search:
        search_query = st.text_input(
            "🔍 Quick Search Sub-Group / Symbol",
            "",
            placeholder="e.g. Defense, Power, TRIL, HAL, Dixon..." if not is_us else "e.g. Software, Energy, NVDA, AAPL...",
            key=f"igm_search_{u_pfx}"
        ).strip().lower()

    # Institutional Calibration Bar: Weighting Method & Min Tradable Peers
    c_wt, c_min_tr, _ = st.columns([1.8, 1.4, 2.8])
    with c_wt:
        weight_mode_choice = st.radio(
            "Index Weighting Method:",
            ["⚖️ Capped Market-Cap (MarketSurge 2.0)", "📏 Equal-Weighted"],
            horizontal=True,
            index=0,
            key=f"igm_weight_mode_{u_pfx}",
            help="MarketSurge 2.0 reconstructs split-adjusted daily market-cap weights with 25% capping; Equal-Weighted treats every tradable member equally."
        )
    weight_mode_key = "equal" if "Equal" in weight_mode_choice else "capped_mcap"

    with c_min_tr:
        min_trad_choice = st.selectbox(
            "Min Tradable Stocks / Group",
            ["4+ (Default)", "5+", "6+", "10+"],
            index=0,
            key=f"igm_min_tradable_{u_pfx}",
            help="Display-level filter on minimum tradable non-duplicate peers per ranked group."
        )
    min_trad_floor = int(min_trad_choice.replace("+", "").split()[0])

    # -------------------------------------------------------------
    # 2. COMPUTE ENGINE DATA (SUB-SECOND)
    # -------------------------------------------------------------
    with st.spinner("Calculating MarketSurge 2.0 rotation, breadth & RSNH diagnostics..."):
        df_matrix, benchmark_curve, df_niche = compute_industry_group_matrix_with_niche(
            taxonomy=tax_key,
            universe_scope=universe_scope,
            weight_mode=weight_mode_key,
        )
    if (df_niche is None or df_niche.empty) and tax_key == "canonical":
        df_niche = load_cached_niche_industry_groups(universe_scope)
        
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

    if min_trad_floor > MIN_RANKED_GROUP_SIZE_CANONICAL:
        cnt_col = "Tradable_Count" if "Tradable_Count" in filtered_df.columns else "Stock_Count"
        filtered_df = filtered_df[filtered_df[cnt_col] >= min_trad_floor]

    if status_filter == "⭐ Top 20% Leading Groups (A+/A) Only":
        filtered_df = filtered_df[filtered_df["Is_Top_20_Pct"] == True]
    elif status_filter == "💪 Broad Pack (Breadth ≥ 50%)" and "Breadth_RS80_Pct" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Breadth_RS80_Pct"] >= 50.0]
    elif status_filter == "✅ Breadth Confirmed (Above Med Breadth & RS ≥ 80)" and "Breadth_RS80_Pct" in filtered_df.columns:
        med_b = float(df_matrix["Breadth_RS80_Pct"].median()) if not df_matrix.empty else 0.0
        filtered_df = filtered_df[(filtered_df["Breadth_RS80_Pct"] >= med_b) & (filtered_df["Comp_RS"] >= 80)]
    elif status_filter == "🏔️ Low Overhead Supply (≥50% Near 52W High)" and "Near_52WH_Pct" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Near_52WH_Pct"] >= 50.0]
    elif status_filter == "🔵 RS New High Cluster (≥30% RSNH)" and "RSNH_Pct" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["RSNH_Pct"] >= 30.0]
    elif status_filter == "🎯 Pre-Breakout RS Blue Dot (≥20%)" and "RS_BlueDot_Pct" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["RS_BlueDot_Pct"] >= 20.0]
    elif status_filter == "🚫 Avoid D & E Laggards Only":
        filtered_df = filtered_df[filtered_df["Is_Danger_Group"] == True]
    elif status_filter == "🚀 Dual Alpha Only (RS>0 & Abs>0)":
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
    elif status_filter in ["🐺 High Pack Hunting (≥3)", "🐺 Group Confirmed (≥3 Leaders)"]:
        filtered_df = filtered_df[filtered_df["Is_Group_Confirmed"] == True]

    if search_query:
        filtered_df = filtered_df[
            filtered_df["Industry_Group"].str.lower().str.contains(search_query) |
            filtered_df["Constituents"].apply(lambda clist: any(search_query in str(t).lower() for t in clist))
        ]

    # In-page Smooth RS, Dual-Horizon Sortino & Apex Leader controls (Global across Table, Cards & Quadrants)
    f_c1, f_c_srs, f_c2, f_c_ls, f_c3 = st.columns([1.2, 0.8, 0.8, 0.8, 0.8])
    with f_c1:
        st.markdown("**🛡️ Smooth RS & Downside-Adjusted Quality Filters** *(MAR = 6.5%)*")
        st.caption("Filters themes with high risk-adjusted consistency & institutional compounder density across all tabs.")
    with f_c_srs:
        igm_min_smooth_rs = st.slider(
            "Min Smooth RS",
            min_value=-5.0,
            max_value=25.0,
            value=-5.0,
            step=0.5,
            help="Filter groups by minimum Smooth RS (Composite RS / 20D ATR%) of the theme synthetic curve.",
            key=f"igm_min_smooth_rs_{u_pfx}"
        )
    with f_c2:
        igm_min_s3m = st.slider(
            "Min Grp Sortino (3M)",
            min_value=-2.0,
            max_value=15.0,
            value=-2.0,
            step=0.5,
            help="Filter groups by minimum 3-month Sortino Ratio of the broad equal-weighted theme synthetic curve.",
            key=f"igm_min_s3m_{u_pfx}"
        )
    with f_c_ls:
        igm_min_ls3m = st.slider(
            "Min Leader Sortino (3M)",
            min_value=-2.0,
            max_value=15.0,
            value=-2.0,
            step=0.5,
            help="Filter groups by minimum 3-month Sortino Ratio of the top leading constituents (RS ≥ 80).",
            key=f"igm_min_ls3m_{u_pfx}"
        )
    with f_c3:
        igm_min_apex = st.slider(
            "Min Apex Leaders",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Filter groups with minimum count of Apex Leaders (RS ≥ 80 AND 3M Sortino ≥ 3.0).",
            key=f"igm_min_apex_{u_pfx}"
        )

    if igm_min_smooth_rs > -5.0 and "Smooth_RS" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Smooth_RS"] >= igm_min_smooth_rs]
    if igm_min_s3m > -2.0:
        filtered_df = filtered_df[filtered_df["Sortino_3M"] >= igm_min_s3m]
    if igm_min_ls3m > -2.0 and "Leader_Sortino_3M" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Leader_Sortino_3M"] >= igm_min_ls3m]
    if igm_min_apex > 0:
        filtered_df = filtered_df[filtered_df["Apex_Leaders"] >= igm_min_apex]

    if igm_min_smooth_rs > -5.0 or igm_min_s3m > -2.0 or igm_min_ls3m > -2.0 or igm_min_apex > 0:
        active_filters = []
        if igm_min_smooth_rs > -5.0:
            active_filters.append(f"Min Smooth RS ≥ {igm_min_smooth_rs:.1f}x")
        if igm_min_s3m > -2.0:
            active_filters.append(f"Min Grp Sortino 3M ≥ {igm_min_s3m:.1f}")
        if igm_min_ls3m > -2.0:
            active_filters.append(f"Min Leader Sortino 3M ≥ {igm_min_ls3m:.1f}")
        if igm_min_apex > 0:
            active_filters.append(f"Min Apex Leaders ≥ {igm_min_apex}")
        filter_str = ", ".join(active_filters)
        st.info(f"🔍 **Quality Filter Active**: Showing **{len(filtered_df)} of {len(df_matrix)}** groups matching {filter_str}")

    # Apply Selected Sort Hierarchy
    filtered_df = _sort_groups_df(filtered_df, sort_choice)

    global_filtered_groups = set(filtered_df["Industry_Group"].tolist()) if not filtered_df.empty else set()

    if limit_choice == "Top 25 Groups":
        filtered_df = filtered_df.head(25)
    elif limit_choice == "Top 50 Groups":
        filtered_df = filtered_df.head(50)
    elif limit_choice == "Top 100 Groups":
        filtered_df = filtered_df.head(100)

    filter_sig = f"{universe_scope}|{tax_key}|{weight_mode_key}|{min_trad_choice}|{sort_choice}|{status_filter}|{limit_choice}|{search_query}|{igm_min_smooth_rs}|{igm_min_s3m}|{igm_min_ls3m}|{igm_min_apex}|{len(df_matrix)}"
    _pinned_jump_grp = None
    try:
        from leadership_cluster_ui import apply_igm_jump_with_sticky
        filtered_df, _pinned_jump_grp, _jump_notice = apply_igm_jump_with_sticky(
            st.session_state,
            u_pfx,
            filtered_df,
            df_matrix,
            active_jump_payload,
            filter_sig,
            df_niche=df_niche if "df_niche" in locals() else None,
        )
        if _jump_notice:
            st.info(_jump_notice)
    except Exception:
        _pinned_jump_grp = None

    # -------------------------------------------------------------
    # TRACK FILTER & SORT SIGNATURE FOR SYNCING DRILLDOWN
    # -------------------------------------------------------------
    sel_widget_key = f"igm_drilldown_sel_{u_pfx}_{filter_sig}"
    last_sig_key = f"igm_last_filter_sig_{u_pfx}"
    sel_grp_key = f"igm_selected_group_{u_pfx}"

    if _pinned_jump_grp:
        st.session_state[last_sig_key] = filter_sig
        st.session_state[sel_grp_key] = _pinned_jump_grp
        st.session_state[sel_widget_key] = _pinned_jump_grp
    elif st.session_state.get(last_sig_key) != filter_sig:
        st.session_state[last_sig_key] = filter_sig
        # Top-level sort or filter changed! Reset drilldown to top group of the newly sorted list
        if not filtered_df.empty:
            new_top_group = filtered_df.iloc[0]["Industry_Group"]
            st.session_state[sel_grp_key] = new_top_group
            st.session_state[sel_widget_key] = new_top_group

    # -------------------------------------------------------------
    # 5. MULTI-TABBED WORLD-CLASS PRESENTATION
    # -------------------------------------------------------------
    niche_cnt = len(df_niche) if df_niche is not None and not df_niche.empty else 0
    tab_quad, tab_heatmap, tab_matrix, tab_top_worst, tab_tandem, tab_cards, tab_niche, tab_guide = st.tabs([
        "🌪️ Rotation Velocity Quadrants",
        "🗺️ Heatmap & Themes",
        f"📑 Leadership Matrix Table ({len(filtered_df)})",
        "🏆 Top vs Worst Groups (6-Week)",
        "👯 Sister Tandem Breakout Tracker",
        f"🎴 Actionable Leader Cards ({len(filtered_df)})",
        f"🔬 Niche & Excluded Groups (<4) ({niche_cnt})",
        "📖 Institutional Methodology"
    ])

    # -------------------------------------------------------------
    # TAB 3: ROTATION VELOCITY QUADRANTS (SCATTER PLOT)
    # -------------------------------------------------------------
    with tab_quad:
        # Segmented view toggle: Himanshu Sharma RRG Engine vs Original Industry Scatter
        c_mode_sel, _ = st.columns([2.6, 1.4])
        with c_mode_sel:
            quad_view_mode = st.radio(
                "Rotation Analytics Engine",
                [
                    "🧭 Himanshu Sharma (CMT) & Julius de Kempenaer RRG Sector Velocity Engine",
                    "📊 Original Industry Group Scatter (Comp RS vs 1M Rank Velocity)"
                ],
                index=0,
                horizontal=True,
                help="Choose between the 2D Julius de Kempenaer RRG engine with rotating vectors or the legacy 197 Industry Group Scatter."
            )

        if "RRG" in quad_view_mode:
            if render_rrg_sector_matrix_view is not None:
                render_rrg_sector_matrix_view()
            else:
                try:
                    import rrg_engine
                    rrg_engine.render_rrg_sector_matrix_view()
                except Exception as e:
                    st.error(f"Error loading RRG Sector Engine: {e}")
        else:
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
    # TAB: SECTOR TREEMAP HEATMAP & MULTI-HORIZON THEME LEADERBOARD
    # -------------------------------------------------------------
    with tab_heatmap:
        hm_c1, hm_c2 = st.columns([2.4, 1.6])
        with hm_c1:
            st.caption("🗺️ **Institutional Capital Flow Map:** Click any **Theme Bar** or **Treemap Tile** (or use the Theme Desk selector below) to inspect constituent leaders & CANSLIM charts right here.")
        with hm_c2:
            hm_horizon = st.radio(
                "Return Horizon",
                ["1W", "1M", "3M", "6M"],
                index=1,
                horizontal=True,
                key=f"igm_heatmap_horizon_{u_pfx}",
                label_visibility="collapsed",
            )

        if filtered_df.empty:
            st.info("No industry groups match current filters.")
        else:
            from components import (
                build_constituent_diverging_bar_fig,
                build_sector_treemap_fig,
                build_theme_drilldown_kpi_html,
                build_theme_rotation_bar_fig,
                extract_plotly_selection_signature,
                load_cached_ticker_series_with_mas,
                resolve_heatmap_selected_group,
            )

            treemap_col, bar_col = st.columns([1.45, 1.0])
            tm_event = None
            bar_event = None
            with treemap_col:
                tm_fig = build_sector_treemap_fig(filtered_df, horizon=hm_horizon, height=520)
                try:
                    tm_event = st.plotly_chart(
                        tm_fig,
                        use_container_width=True,
                        on_select="rerun",
                        selection_mode="points",
                        key=f"igm_hm_treemap_{u_pfx}",
                        config={"displayModeBar": False},
                    )
                except TypeError:
                    st.plotly_chart(tm_fig, use_container_width=True, config={"displayModeBar": False})

            with bar_col:
                st.markdown(f"**🏆 Top & Bottom Themes ({hm_horizon})** — *click any bar to load below*")
                bar_fig = build_theme_rotation_bar_fig(filtered_df, horizon=hm_horizon, top_n=18, height=485)
                try:
                    bar_event = st.plotly_chart(
                        bar_fig,
                        use_container_width=True,
                        on_select="rerun",
                        selection_mode="points",
                        key=f"igm_hm_bar_{u_pfx}",
                        config={"displayModeBar": False},
                    )
                except TypeError:
                    st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

            # Event-signature diffing prevents sticky chart clicks from overriding cross-tab selections
            valid_hm_groups = filtered_df["Industry_Group"].tolist()
            tm_sig = extract_plotly_selection_signature(tm_event)
            bar_sig = extract_plotly_selection_signature(bar_event)
            last_tm_sig_key = f"igm_hm_last_tm_sig_{u_pfx}"
            last_bar_sig_key = f"igm_hm_last_bar_sig_{u_pfx}"
            tm_changed = bool(tm_sig and tm_sig != st.session_state.get(last_tm_sig_key))
            bar_changed = bool(bar_sig and bar_sig != st.session_state.get(last_bar_sig_key))
            st.session_state[last_tm_sig_key] = tm_sig
            st.session_state[last_bar_sig_key] = bar_sig

            clicked_grp = None
            if bar_changed:
                clicked_grp = resolve_heatmap_selected_group(
                    tm_event, bar_event, valid_hm_groups, None, prefer_source="bar"
                )
            elif tm_changed:
                clicked_grp = resolve_heatmap_selected_group(
                    tm_event, bar_event, valid_hm_groups, None, prefer_source="treemap"
                )

            if clicked_grp and clicked_grp in valid_hm_groups:
                st.session_state[sel_grp_key] = clicked_grp
                st.session_state[sel_widget_key] = clicked_grp

            cur_hm_grp = st.session_state.get(sel_grp_key)
            active_hm_grp = cur_hm_grp if cur_hm_grp in valid_hm_groups else valid_hm_groups[0]

            # Pre-sync inline selector strictly BEFORE widget instantiation
            hm_quick_key = f"igm_hm_quick_grp_{u_pfx}_{filter_sig}"
            st.session_state[hm_quick_key] = active_hm_grp

            st.markdown("---")
            hm_pick_c1, hm_pick_c2 = st.columns([2.0, 1.0], vertical_alignment="center")
            with hm_pick_c1:
                chosen_hm_grp = st.selectbox(
                    "🎯 Active Theme Quick Drill-Down Desk",
                    options=valid_hm_groups,
                    key=hm_quick_key,
                    help="Synced automatically when you click any bar/treemap tile or pick a theme here.",
                )
                if chosen_hm_grp != active_hm_grp:
                    active_hm_grp = chosen_hm_grp
                    st.session_state[sel_grp_key] = active_hm_grp
                    st.session_state[sel_widget_key] = active_hm_grp
            with hm_pick_c2:
                st.caption("Click any constituent row below to preview its **90D CANSLIM (21E / 50S / 65E)** chart without leaving this tab.")

            df_constits_hm, _, tv_copy_hm = get_group_deep_dive_data(
                active_hm_grp,
                taxonomy=tax_key,
                universe_scope=universe_scope,
                weight_mode=weight_mode_key,
            )
            grp_match_hm = filtered_df[filtered_df["Industry_Group"] == active_hm_grp]
            if not grp_match_hm.empty:
                st.markdown(
                    clean_html(
                        build_theme_drilldown_kpi_html(
                            grp_match_hm.iloc[0],
                            horizon=hm_horizon,
                            n_constits=len(df_constits_hm) if df_constits_hm is not None else 0,
                        )
                    ),
                    unsafe_allow_html=True,
                )

            if df_constits_hm is None or df_constits_hm.empty:
                st.info(f"No constituent price history available for **{active_hm_grp}**.")
            else:
                cmp_col_hm = "CMP ($)" if is_us else "CMP (₹)"
                if cmp_col_hm not in df_constits_hm.columns and "CMP" in df_constits_hm.columns:
                    cmp_col_hm = "CMP"
                hm_disp_cols = [
                    c for c in [
                        "Symbol", cmp_col_hm, "1D %", "1M %", "RS Rating",
                        "Sortino 3M", "RS Line High", "Dist Pivot %", "Dist 21 EMA %",
                        "Execution Status", "TradingView_URL",
                    ]
                    if c in df_constits_hm.columns
                ]
                hm_tbl_df = df_constits_hm[hm_disp_cols].copy()
                hm_col_cfg = {
                    "Symbol": st.column_config.TextColumn("Symbol", width=90),
                    cmp_col_hm: st.column_config.NumberColumn(f"CMP ({curr_symbol})", format=f"{curr_symbol}%.2f", width=95),
                    "1D %": st.column_config.NumberColumn("1D %", format="%.2f%%", width=75),
                    "1M %": st.column_config.NumberColumn("1M %", format="%.2f%%", width=75),
                    "RS Rating": st.column_config.ProgressColumn("RS", min_value=0, max_value=99, format="%d", width=85),
                    "Sortino 3M": st.column_config.NumberColumn("S(3M)", format="%.2f", width=70),
                    "RS Line High": st.column_config.TextColumn("RS Line", width=90),
                    "Dist Pivot %": st.column_config.NumberColumn("Piv%", format="%.1f%%", width=70),
                    "Dist 21 EMA %": st.column_config.NumberColumn("21E%", format="%.1f%%", width=70),
                    "Execution Status": st.column_config.TextColumn("Action State", width=130),
                    "TradingView_URL": st.column_config.LinkColumn("TV", display_text="Open ↗", width=70),
                }
                hm_col_cfg = {k: v for k, v in hm_col_cfg.items() if k in hm_tbl_df.columns}

                desk_left, desk_right = st.columns([1.3, 1.0])
                sel_constit_idx = 0
                with desk_left:
                    st.markdown(f"**📋 Constituent Leaders — {active_hm_grp} ({len(hm_tbl_df)})**")
                    constit_tbl_key = f"igm_hm_constit_tbl_{u_pfx}_{tax_key}_{active_hm_grp}"
                    try:
                        c_ev = st.dataframe(
                            hm_tbl_df,
                            column_config=hm_col_cfg,
                            use_container_width=True,
                            hide_index=True,
                            height=385,
                            on_select="rerun",
                            selection_mode="single-row",
                            key=constit_tbl_key,
                        )
                        if c_ev and getattr(c_ev, "selection", None) and c_ev.selection.rows:
                            raw_idx = int(c_ev.selection.rows[0])
                            if 0 <= raw_idx < len(df_constits_hm):
                                sel_constit_idx = raw_idx
                    except TypeError:
                        st.dataframe(hm_tbl_df, column_config=hm_col_cfg, use_container_width=True, hide_index=True, height=385)

                with desk_right:
                    desk_mode = st.radio(
                        "Desk Panel Mode",
                        ["📈 Leader CANSLIM Chart", "📊 Constituent Spread Bars"],
                        horizontal=True,
                        key=f"igm_hm_desk_view_{u_pfx}",
                        label_visibility="collapsed",
                    )
                    if "CANSLIM" in desk_mode:
                        sel_row = df_constits_hm.iloc[sel_constit_idx]
                        clean_t = str(sel_row.get("Symbol", ""))
                        raw_t = str(sel_row.get("Raw_Ticker", clean_t))
                        piv_val = float(sel_row.get("Pivot Price", 0.0) or 0.0)
                        piv_arg = piv_val if piv_val > 0 else None
                        st.markdown(
                            clean_html(
                                f"<div style='font-size:0.80rem;color:#cbd5e1;margin:2px 0 6px 0;font-family:JetBrains Mono,monospace;'>"
                                f"<b style='color:#f8fafc;'>{clean_t}</b> · RS <b>{int(sel_row.get('RS Rating', 0))}</b> · "
                                f"Piv <b>{float(sel_row.get('Dist Pivot %', 0.0)):+.1f}%</b> · "
                                f"21E <b>{float(sel_row.get('Dist 21 EMA %', 0.0)):+.1f}%</b> · "
                                f"<span style='color:#38bdf8;'>{sel_row.get('Execution Status', '')}</span></div>"
                            ),
                            unsafe_allow_html=True,
                        )
                        ohlcv_hm = load_cached_ticker_series_with_mas(
                            raw_t, clean_t, market="US" if is_us else "IN"
                        )
                        if ohlcv_hm is not None and not ohlcv_hm.empty:
                            from rs_radar_ui import build_canslim_ohlcv_chart
                            st.plotly_chart(
                                build_canslim_ohlcv_chart(
                                    ohlcv_hm,
                                    ticker=clean_t,
                                    entry_trigger=piv_arg,
                                    height=330,
                                    bars=90,
                                ),
                                use_container_width=True,
                                config={"displayModeBar": False},
                            )
                        else:
                            st.info(f"No local cached OHLCV bars found for {clean_t}.")
                    else:
                        hm_to_bar = {"1W": "1W %", "1M": "1M %", "3M": "3M %", "6M": "3M %"}
                        last_hm_hor_key = f"igm_hm_last_hor_sync_{u_pfx}"
                        bar_metric_key = f"igm_hm_bar_metric_{u_pfx}"
                        if st.session_state.get(last_hm_hor_key) != hm_horizon:
                            st.session_state[last_hm_hor_key] = hm_horizon
                            st.session_state[bar_metric_key] = hm_to_bar.get(hm_horizon, "1M %")
                        bar_metric_hm = st.pills(
                            "Spread Axis",
                            ["1D %", "1W %", "1M %", "3M %"],
                            default=hm_to_bar.get(hm_horizon, "1M %"),
                            key=bar_metric_key,
                            label_visibility="collapsed",
                        ) or "1M %"
                        if hm_horizon == "6M":
                            st.caption("Showing 3M % constituent spread (6M % not stored per constituent).")
                        st.plotly_chart(
                            build_constituent_diverging_bar_fig(df_constits_hm, metric_col=bar_metric_hm, max_bars=16),
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )

                if tv_copy_hm:
                    with st.expander(f"📋 Copy {active_hm_grp} Watchlist for TradingView ({len(df_constits_hm)})", expanded=False):
                        st.code(tv_copy_hm, language="text")

    # -------------------------------------------------------------
    # TAB 1: MASTER MATRIX TABLE
    # -------------------------------------------------------------
    with tab_matrix:
        tv_col1, tv_col2 = st.columns([1.6, 2.4])
        with tv_col1:
            st.caption("💡 *Click on any row in the table to immediately sync constituent breakdown below.*")
        with tv_col2:
            _tv_options = ["⚡ Compact Leaders", "📊 Executive Overview", "🛡️ Risk & Quality", "🌪️ Momentum Velocity", "📑 All Columns"]
            _tv_key = f"igm_table_view_mode_{u_pfx}"
            if _tv_key in st.session_state and st.session_state[_tv_key] not in _tv_options:
                st.session_state[_tv_key] = _tv_options[0]
            tv_mode = st.radio(
                "Table View:",
                _tv_options,
                horizontal=True,
                key=_tv_key
            )

        if "Compact" in tv_mode:
            from components import build_compact_matrix_view
            display_df = build_compact_matrix_view(filtered_df)
        elif "Executive" in tv_mode:
            disp_cols = [
                "Rank_Today", "Group_Grade_Badge", "Industry_Group", "Blended_Score", "Pure_RS_Rank",
                "Breadth_RS80_Pct", "RSNH_Pct", "RSNH_Count", "Dual_Badge", "Group_Confirmation_Badge",
                "Rotation_Status", "Smooth_RS", "Leader_Sortino_3M", "Apex_Leaders",
                "Return_1M", "Top_Leaders_Display", "Sparkline_1M"
            ]
            disp_cols = [c for c in disp_cols if c in filtered_df.columns]
            display_df = filtered_df[disp_cols].copy()
        elif "Risk" in tv_mode:
            disp_cols = [
                "Rank_Today", "Group_Grade_Badge", "Industry_Group", "Blended_Score", "Pure_RS_Rank",
                "Breadth_RS80_Pct", "Breadth_50SMA_Pct", "Near_52WH_Pct", "RSNH_Pct", "RS_BlueDot_Pct", "CANSLIM_Funda_Pct",
                "Dual_Badge", "Group_Confirmation_Badge", "Horizon_Badge", "Horizon_State",
                "Excess_3M", "Smooth_RS", "Leader_Sortino_3M", "Sortino_3M", "Sortino_6M", "Apex_Leaders",
                "Pack_Hunting_Count", "Tradable_Count", "Stock_Count", "Return_3M", "Return_6M",
                "Top_Leaders_Display"
            ]
            disp_cols = [c for c in disp_cols if c in filtered_df.columns]
            display_df = filtered_df[disp_cols].copy()
        elif "Velocity" in tv_mode:
            disp_cols = [
                "Rank_Today", "Group_Grade_Badge", "Industry_Group", "Pure_RS_Rank", "RSNH_Pct",
                "Dual_Badge", "Group_Confirmation_Badge", "Horizon_Badge", "Horizon_State", "Rotation_Status",
                "Rank_1W", "Delta_1W", "Rank_1M", "Delta_1M", "Rank_3M", "Rank_6M",
                "Pack_Hunting_Count", "Sparkline_1M"
            ]
            disp_cols = [c for c in disp_cols if c in filtered_df.columns]
            display_df = filtered_df[disp_cols].copy()
        else:
            disp_cols = [
                "Rank_Today", "Group_Grade_Badge", "Industry_Group", "Blended_Score", "Pure_Price_RS", "Pure_RS_Rank",
                "Breadth_RS80_Pct", "Breadth_50SMA_Pct", "Near_52WH_Pct", "RSNH_Pct", "RSNH_Count", "RS_BlueDot_Pct", "CANSLIM_Funda_Pct",
                "Dual_Badge", "Group_Confirmation_Badge", "Horizon_Badge", "Horizon_State", "Rotation_Status",
                "Excess_1M", "Excess_3M", "Excess_6M",
                "Smooth_RS", "Leader_Sortino_3M", "Sortino_3M", "Sortino_6M", "Apex_Leaders",
                "Rank_1W", "Delta_1W", "Rank_1M", "Delta_1M", "Rank_3M", "Rank_6M",
                "Pack_Hunting_Count", "Tradable_Count", "Stock_Count", "Return_1M", "Return_3M", "Return_6M",
                "Top_Leaders_Display", "Sparkline_1M"
            ]
            disp_cols = [c for c in disp_cols if c in filtered_df.columns]
            display_df = filtered_df[disp_cols].copy()

        master_col_cfg = {
            "Rank_Today": st.column_config.NumberColumn("Rank", format="%d", width=70, help="Current Option B Blended Rank (75% Smoothed 6M RS + 25% Breadth)"),
            "Group_Grade_Badge": st.column_config.TextColumn("IBD Grade", width=95, help="Investor's Business Daily Letter Grade: A+ / A (Top 20% Leading), B (Favorable), C (Neutral), D / E (Bottom 40% Laggards — Avoid)"),
            "Industry_Group": st.column_config.TextColumn("Industry Sub-Group", width=250),
            "Blended_Score": st.column_config.NumberColumn("Blended", format="%.1f", width=85, help="MarketSurge 2.0 Option B Blended Score: 75% Smoothed 6M Price RS + 25% Breadth Score"),
            "Pure_Price_RS": st.column_config.NumberColumn("Pure RS%", format="%.1f", width=85, help="Pure Smoothed 6M Price-RS Percentile (0-100) before Breadth blending"),
            "Pure_RS_Rank": st.column_config.NumberColumn("Pure Rank", format="%d", width=85, help="Rank by Pure Price-RS alone (excludes Breadth adjustment)"),
            "Breadth_RS80_Pct": st.column_config.NumberColumn("RS≥80 %", format="%.0f%%", width=85, help="% of active tradable peers with Stock RS Rating >= 80"),
            "Breadth_50SMA_Pct": st.column_config.NumberColumn(">50 SMA %", format="%.0f%%", width=90, help="% of active tradable peers trading above 50-day SMA"),
            "Near_52WH_Pct": st.column_config.NumberColumn("Near 52WH %", format="%.0f%%", width=95, help="% of active tradable peers within 15% of 52-week high"),
            "RSNH_Pct": st.column_config.NumberColumn("🔵 RSNH %", format="%.1f%%", width=90, help="Ross Haber RS Line New High %: active tradable peers whose Stock/Benchmark RS line is within 1% of its 52-week high"),
            "RSNH_Count": st.column_config.NumberColumn("RSNH #", format="%d", width=75, help="Count of tradable constituents printing an RS Line New High"),
            "RS_BlueDot_Pct": st.column_config.NumberColumn("🎯 Blue Dot %", format="%.1f%%", width=95, help="O'Neil Pre-Breakout Blue Dot %: peers printing an RS New High while still >=5% below price 52W high"),
            "CANSLIM_Funda_Pct": st.column_config.NumberColumn("CANSLIM %", format="%.0f%%", width=90, help="Observational % of constituents with EPS or Sales growth >= 25%"),
            "Dual_Badge": st.column_config.TextColumn("Dual Gate", width=95, help="Gary Antonacci Dual Momentum Gate: 🚀 Alpha (Excess>0 & Return>0), 🛡️ Mirage (Excess>0 but Return<=0), 📉 Bleed (Excess<=0 & Return<=0)"),
            "Group_Confirmation_Badge": st.column_config.TextColumn("Group Confirmed", width=140, help="Ross Haber Pack-Hunting Rule: At least 3 liquid leaders with RS >= 80 breaking out or coiling together in early-stage bases."),
            "Horizon_Badge": st.column_config.TextColumn("3H Alignment", width=95, help="3-Horizon relative strength agreement: 1M (Short) · 3M (Med) · 6M (Long) vs Benchmark (🟢 Outperform · 🟡 In-Line · 🔴 Lag)"),
            "Horizon_State": st.column_config.TextColumn("Alignment State", width=140, help="Classified horizon state: 🟢 Confluence (all 3 leading), 🌱 Emerging Turn (short-term turning up), ⚠️ Fading (short-term losing momentum), 🔴 Chronic Laggard"),
            "Excess_1M": st.column_config.NumberColumn("Excess 1M", format="%+.1f%%", width=90, help="Group 1M return minus benchmark return (% pts)"),
            "Excess_3M": st.column_config.NumberColumn("Excess 3M", format="%+.1f%%", width=90, help="Group 3M return minus benchmark return (% pts)"),
            "Excess_6M": st.column_config.NumberColumn("Excess 6M", format="%+.1f%%", width=90, help="Group 6M return minus benchmark return (% pts)"),
            "Rotation_Status": st.column_config.TextColumn("Rotation State", width=120),
            "Smooth_RS": st.column_config.NumberColumn("Smooth RS (RS/ATR)", format="%.2fx", width=105, help="Group Synthetic Composite RS / 20D ATR%"),
            "Leader_Sortino_3M": st.column_config.NumberColumn("Ldr Sortino 3M", format="%.2f", width=105, help="Average 3-Month Sortino ratio of top leading constituents (RS >= 80)"),
            "Sortino_3M": st.column_config.NumberColumn("Grp Sortino 3M", format="%.2f", width=105, help="3-Month Downside-adjusted Sortino ratio of capped group curve (MAR=6.5%)"),
            "Sortino_6M": st.column_config.NumberColumn("Sortino 6M", format="%.2f", width=95, help="6-Month Downside-adjusted Sortino ratio (MAR=6.5%)"),
            "Apex_Leaders": st.column_config.NumberColumn("👑 Apex", format="%d", width=90, help="Count of constituents with RS >= 80 AND Sortino 3M >= 3.0"),
            "Rank_1W": st.column_config.NumberColumn("1W Ago", format="%d", width=80),
            "Delta_1W": st.column_config.NumberColumn("Δ 1W", format="%+d", width=75, help="Change in rank over 1 week (positive = improving)"),
            "Rank_1W_Compact": st.column_config.TextColumn("1W (Δ)", width=85, help="1-Week Ago Rank and rank delta (positive = improving)"),
            "Rank_1M": st.column_config.NumberColumn("1M Ago", format="%d", width=80),
            "Delta_1M": st.column_config.NumberColumn("Δ 1M", format="%+d", width=75, help="Change in rank over 1 month (positive = accelerating)"),
            "Rank_1M_Compact": st.column_config.TextColumn("1M (Δ)", width=85, help="1-Month Ago Rank and rank delta (positive = accelerating)"),
            "Rank_3M": st.column_config.NumberColumn("3M Ago", format="%d", width=80),
            "Rank_6M": st.column_config.NumberColumn("6M Ago", format="%d", width=80),
            "Pack_Hunting_Count": st.column_config.NumberColumn("🐺 Pack (RS≥80)", format="%d", width=110, help="Count of tradable stocks in group with RS >= 80"),
            "Tradable_Count": st.column_config.NumberColumn("Tradable", format="%d", width=85, help="Active tradable non-duplicate-class peers passing liquidity floor"),
            "Stock_Count": st.column_config.NumberColumn("Total Stocks", format="%d", width=90, help="Total constituents mapped to this group (deduplicated)"),
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
            key=f"matrix_table_selection_{u_pfx}"
        )

        selected_table_group = None
        if event and event.selection and event.selection.rows:
            sel_idx = event.selection.rows[0]
            if 0 <= sel_idx < len(display_df):
                selected_table_group = display_df.iloc[sel_idx]["Industry_Group"]
                st.session_state[sel_grp_key] = selected_table_group
                st.session_state[sel_widget_key] = selected_table_group

    # -------------------------------------------------------------
    # TAB: TOP 40 LEADERS VS WORST 40 LAGGARDS (6-WEEK TRAJECTORY)
    # -------------------------------------------------------------
    with tab_top_worst:
        top_40_df, worst_40_df = get_top_and_worst_40_groups(df_matrix)
        n_lead_slice = len(top_40_df)
        n_lag_slice = len(worst_40_df)

        st.markdown(clean_html(f"""
        <div style='background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(2, 6, 23, 0.95) 100%);
                    border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;'>
            <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;'>
                <div>
                    <h4 style='margin:0; font-size:1.15rem; font-weight:800; color:#f8fafc;'>
                        🏆 Top {n_lead_slice} Leaders vs Bottom {n_lag_slice} Laggards: 6-Week Momentum Trajectory
                    </h4>
                    <p style='margin:4px 0 0 0; font-size:0.82rem; color:#94a3b8;'>
                        Non-overlapping institutional slices (up to <b>Top 40</b> vs <b>Bottom 40</b> across <b>{len(df_matrix)} ranked industry groups</b> · ≥ 4 tradable peers).
                        <b>Concentrate long exposure in Leading groups</b> and strictly avoid Laggards.
                    </p>
                </div>
                <div style='display:flex; gap:12px;'>
                    <span style='background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.4); color:#34d399; padding:4px 10px; border-radius:6px; font-size:0.78rem; font-weight:700;'>
                        Top {n_lead_slice} = Institutional Accumulation
                    </span>
                    <span style='background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.4); color:#f87171; padding:4px 10px; border-radius:6px; font-size:0.78rem; font-weight:700;'>
                        Worst {n_lag_slice} = Chronic Distribution
                    </span>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            t40_rs_med = top_40_df['Comp_RS'].median() if not top_40_df.empty else 0
            st.metric(f"Top {n_lead_slice} Median RS", f"{t40_rs_med:.0f}", delta="Leadership Tier", delta_color="normal")
        with m2:
            t40_apex_sum = int(top_40_df['Apex_Leaders'].sum()) if not top_40_df.empty else 0
            st.metric(f"Top {n_lead_slice} Apex Leaders", f"{t40_apex_sum} 👑", help="Constituents with RS ≥ 80 & Sortino 3M ≥ 3.0")
        with m3:
            w40_rs_med = worst_40_df['Comp_RS'].median() if not worst_40_df.empty else 0
            st.metric(f"Worst {n_lag_slice} Median RS", f"{w40_rs_med:.0f}", delta="Laggard Tier", delta_color="inverse")
        with m4:
            w40_apex_sum = int(worst_40_df['Apex_Leaders'].sum()) if not worst_40_df.empty else 0
            st.metric(f"Worst {n_lag_slice} Apex Leaders", f"{w40_apex_sum} 👑", help="Laggard groups typically have near-zero Apex compounders")

        st.write("")

        # Columns configuration for IBD tables
        col_cfg_ibd = {
            "Rank_Today": st.column_config.NumberColumn("Rank", format="%d", width=65, help="Current Group Blended Rank"),
            "Group_Grade_Badge": st.column_config.TextColumn("IBD Grade", width=85, help="Investor's Business Daily Letter Grade (A+ to E)"),
            "Horizon_Badge": st.column_config.TextColumn("3H", width=75, help="3-Horizon Alignment (1M·3M·6M)"),
            "Rank_3W": st.column_config.NumberColumn("3W", format="%d", width=65, help="Rank 3 Weeks Ago"),
            "Rank_6W": st.column_config.NumberColumn("6W", format="%d", width=65, help="Rank 6 Weeks Ago"),
            "Delta_6W": st.column_config.NumberColumn("Δ 6W", format="%+d", width=65, help="6-Week Rank Velocity (positive = climbing ranks)"),
            "Industry_Group": st.column_config.TextColumn("Industry Group", width=210),
            "Comp_RS": st.column_config.NumberColumn("RS", format="%d", width=60, help="Blended Group RS (0-99)"),
            "RSNH_Pct": st.column_config.NumberColumn("RSNH%", format="%.0f%%", width=75, help="RS Line New High %"),
            "Return_1D": st.column_config.NumberColumn("1D %", format="%+.1f%%", width=75),
            "Return_YTD": st.column_config.NumberColumn("YTD %", format="%+.1f%%", width=75),
            "Apex_Leaders": st.column_config.NumberColumn("👑", format="%d", width=55, help="Apex Leaders Count"),
            "Top_Leaders_Display": st.column_config.TextColumn("Leading Anchor Stocks", width=260)
        }

        c_lead, c_lagg = st.columns(2)

        with c_lead:
            st.markdown(f"#### 🏆 Top {n_lead_slice} Leading Industry Groups")
            st.caption("Click any group to inspect its constituents and pivot chart below.")

            t40_disp_cols = [c for c in ["Rank_Today", "Group_Grade_Badge", "Horizon_Badge", "Rank_3W", "Rank_6W", "Delta_6W", "Industry_Group", "Comp_RS", "RSNH_Pct", "Return_1D", "Return_YTD", "Apex_Leaders", "Top_Leaders_Display"] if c in top_40_df.columns]
            t40_disp = top_40_df[t40_disp_cols].copy()

            event_top = st.dataframe(
                t40_disp,
                column_config={k: v for k, v in col_cfg_ibd.items() if k in t40_disp.columns},
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key=f"ibd_top_40_table_{u_pfx}",
                height=650
            )
            if event_top and event_top.selection and event_top.selection.rows:
                sel_idx = event_top.selection.rows[0]
                if 0 <= sel_idx < len(t40_disp):
                    grp_name = t40_disp.iloc[sel_idx]["Industry_Group"]
                    st.session_state[sel_grp_key] = grp_name
                    st.session_state[sel_widget_key] = grp_name

        with c_lagg:
            st.markdown(f"#### ⚠️ Bottom {n_lag_slice} Lagging Industry Groups")
            st.caption("Groups suffering persistent institutional selling and rank decay.")

            w40_disp_cols = [c for c in ["Rank_Today", "Group_Grade_Badge", "Horizon_Badge", "Rank_3W", "Rank_6W", "Delta_6W", "Industry_Group", "Comp_RS", "RSNH_Pct", "Return_1D", "Return_YTD", "Apex_Leaders", "Top_Leaders_Display"] if c in worst_40_df.columns]
            w40_disp = worst_40_df[w40_disp_cols].copy()
            
            event_worst = st.dataframe(
                w40_disp,
                column_config={k: v for k, v in col_cfg_ibd.items() if k in w40_disp.columns},
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key=f"ibd_worst_40_table_{u_pfx}",
                height=650
            )
            if event_worst and event_worst.selection and event_worst.selection.rows:
                sel_idx = event_worst.selection.rows[0]
                if 0 <= sel_idx < len(w40_disp):
                    grp_name = w40_disp.iloc[sel_idx]["Industry_Group"]
                    st.session_state[sel_grp_key] = grp_name
                    st.session_state[sel_widget_key] = grp_name

    # -------------------------------------------------------------
    # TAB: SISTER TANDEM BREAKOUT TRACKER
    # -------------------------------------------------------------
    with tab_tandem:
        st.markdown(clean_html("""
        <div style='background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(2, 6, 23, 0.95) 100%);
                    border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;'>
            <h4 style='margin:0; font-size:1.15rem; font-weight:800; color:#f8fafc;'>
                👯 Jesse Livermore & William O'Neil Sister Stock Tandem Tracker
            </h4>
            <p style='margin:4px 0 0 0; font-size:0.82rem; color:#94a3b8;'>
                In every leading industry group, institutional money concentrates in the <b>#1 Alpha Leader</b> and <b>#2 Sister Stock</b>.
                When Leader #1 breaks out, Sister #2 typically follows within 10 to 15 trading days. 
                <b style='color:#f87171;'>Strict Rule:</b> Never buy the #3+ sympathy laggards.
            </p>
        </div>
        """), unsafe_allow_html=True)
        
        tandem_rows = []
        for _, r in df_matrix.iterrows():
            if r.get('Is_Danger_Group', False) or str(r.get('Group_Grade', '')).startswith(('D', 'E')):
                continue
            ldr_pool = r.get('Top_Leaders_Full', []) or r.get('Top_3_Leaders', []) or []
            top_3 = [
                l for l in ldr_pool
                if l.get('is_tradable_leader', l.get('is_tradable', True))
            ]
            if len(top_3) >= 2:
                l1 = top_3[0]
                l2 = top_3[1]
                if int(l1.get('rs', 0) or 0) < 75 or int(l2.get('rs', 0) or 0) < 70:
                    continue
                grp_name = r['Industry_Group']
                rank_num = r['Rank_Today']
                grade_b = r.get('Group_Grade_Badge', f"#{rank_num}")

                t1_dist = l1.get('dist_pivot', 0.0)
                t2_dist = l2.get('dist_pivot', 0.0)
                t1_status = l1.get('status', '⏳ BASE')
                t2_status = l2.get('status', '⏳ BASE')

                is_t1_breakout = "BUY" in t1_status or "EXTENDED" in t1_status
                is_t2_coiling = "COILING" in t2_status or "RETEST" in t2_status or (-8.0 <= t2_dist <= 2.0)

                if is_t1_breakout and is_t2_coiling:
                    tandem_state = "🔥 Tandem Setup (Leader Broke Out, Sister Coiling)"
                elif "BUY" in t1_status and "BUY" in t2_status:
                    tandem_state = "🚀 Dual Breakout (Both in Buy Zone)"
                elif "FAIL" in t1_status or "FAIL" in t2_status:
                    tandem_state = "⚠️ Divergence Breakdown"
                else:
                    tandem_state = "👯 Active Tandem Pair"

                tandem_rows.append({
                    'Group_Rank': rank_num,
                    'IBD_Grade': grade_b,
                    'Industry_Group': grp_name,
                    'Tandem_Status': tandem_state,
                    'Alpha_Leader': f"{l1['ticker']} (RS {l1['rs']})",
                    'Alpha_Action': l1['status'],
                    'Alpha_Dist_Pivot': f"{l1.get('dist_pivot_str', '0.0%')}",
                    'Sister_Stock': f"{l2['ticker']} (RS {l2['rs']})",
                    'Sister_Action': l2['status'],
                    'Sister_Dist_Pivot': f"{l2.get('dist_pivot_str', '0.0%')}",
                    'Sympathy_Count': max(0, int(r.get('Tradable_Count', len(r.get('Constituents', [])))) - 2)
                })
                
        df_tandem = pd.DataFrame(tandem_rows)
        if not df_tandem.empty:
            st.dataframe(
                df_tandem,
                column_config={
                    "Group_Rank": st.column_config.NumberColumn("Rank", format="%d", width=65),
                    "IBD_Grade": st.column_config.TextColumn("Grade", width=85),
                    "Industry_Group": st.column_config.TextColumn("Industry Group", width=200),
                    "Tandem_Status": st.column_config.TextColumn("Tandem Confirmation State", width=280),
                    "Alpha_Leader": st.column_config.TextColumn("👑 #1 Alpha Leader", width=150),
                    "Alpha_Action": st.column_config.TextColumn("Leader Action", width=120),
                    "Alpha_Dist_Pivot": st.column_config.TextColumn("Leader Dist", width=90),
                    "Sister_Stock": st.column_config.TextColumn("👯 #2 Sister Stock", width=150),
                    "Sister_Action": st.column_config.TextColumn("Sister Action", width=120),
                    "Sister_Dist_Pivot": st.column_config.TextColumn("Sister Dist", width=90),
                    "Sympathy_Count": st.column_config.NumberColumn("Sympathy Laggards (Avoid)", format="%d", width=130)
                },
                use_container_width=True,
                hide_index=True,
                height=550
            )
        else:
            st.info("No tandem pairs detected.")

    # -------------------------------------------------------------
    # TAB 3: ACTIONABLE LEADER CARDS GRID
    # -------------------------------------------------------------
    with tab_cards:
        us_tv_map, funda_map = load_exporter_enrichment_maps(is_us=is_us)
        _render_top40_tv_watchlist_exporter(
            df_matrix=df_matrix,
            global_filtered_groups=global_filtered_groups,
            u_pfx=u_pfx,
            is_us=is_us,
            us_tv_map=us_tv_map,
            funda_map=funda_map,
        )

        # Apex Leader Institutional Definition Callout
        st.html("""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: #f1f5f9;">
                <span style="font-weight: 800; color: #34d399; font-size: 0.88rem; display: flex; align-items: center; gap: 4px;">👑 What is an Apex Leader?</span>
                <span style="color: #cbd5e1;">A constituent stock with <b>RS Rating ≥ 80</b> (top 20% momentum) <b>+ 3-Month Sortino Ratio ≥ 3.0</b> (exceptional downside-risk protection). Groups with high Apex counts represent institutional safe havens packed with low-drawdown structural compounders.</span>
            </div>
            <div style="font-size: 0.74rem; color: #94a3b8; font-family: monospace; display: flex; gap: 10px;">
                <span>👑 <b>Alpha #1</b></span>
                <span>👯 <b>Sister #2</b></span>
                <span>🥉 <b>Leader #3</b></span>
                <span>🐺 <b>Pack (RS≥80)</b></span>
            </div>
        </div>
        """)

        c_csort, c_cscope, c_cdepth, c_crs, c_cadtv = st.columns([1.1, 0.8, 0.9, 1.0, 1.2])
        with c_csort:
            card_sort_choice = st.selectbox(
                "Sort Cards By",
                [
                    "Same as Top Filter",
                    "🏆 Blended Rank (75% RS / 25% Breadth)",
                    "⚡ Pure Price-RS Rank (No Breadth)",
                    "🔵 RS New High % (Haber)",
                    "📊 Breadth % RS≥80 (High to Low)",
                    "🏆 Leader Sortino 3M (High to Low)",
                    "🛡️ Grp Sortino 3M (High to Low)",
                    "👑 Apex Leaders Count (High to Low)",
                    "🐺 Wolfpack Breadth (RS ≥ 80 Count)",
                    "🐺 Wolfpack Concentration (% RS ≥ 80)",
                    "🚀 1M Rank Velocity (Δ Spots)",
                    "📈 1M Momentum Return %"
                ],
                index=0,
                key=f"actionable_cards_sort_choice_{u_pfx}",
                help="Sort cards by Option B Blended Rank, Pure Price-RS, RSNH%, Breadth %, Leader Sortino, wolfpack breadth, or velocity"
            )
        with c_cscope:
            card_count_choice = st.selectbox(
                "Show Cards",
                ["Top 18 Cards", "Top 36 Cards", "All Filtered Groups"],
                index=1,
                key=f"actionable_cards_count_choice_{u_pfx}"
            )
        with c_cdepth:
            card_depth_choice = st.selectbox(
                "Leaders per Card",
                ["Top 10 Leaders", "Top 3 Leaders", "Top 5 Leaders", "Top 15 Leaders", "All Constituents"],
                index=0,
                key=f"actionable_cards_depth_choice_{u_pfx}",
                help="Choose how many ranked constituents to display inside each group card. Default is Top 10 Leaders."
            )
        with c_crs:
            card_min_rs_choice = st.selectbox(
                "Min Constituent RS",
                [
                    "All RS (0+)",
                    "RS ≥ 70 (Contenders)",
                    "RS ≥ 80 (True Leaders)",
                    "RS ≥ 85 (High Momentum)",
                    "RS ≥ 90 (Apex Elite)"
                ],
                index=0,
                key=f"actionable_cards_min_rs_{u_pfx}",
                help="Filter constituents inside cards to only display stocks with RS at or above this threshold."
            )
        adtv_unit_label = "₹ Cr" if not is_us else "$M"
        with c_cadtv:
            min_card_adtv = st.slider(
                f"Min Stock ADTV ({adtv_unit_label})",
                min_value=0.0,
                max_value=25.0 if not is_us else 50.0,
                value=0.0,
                step=0.5 if not is_us else 1.0,
                key=f"actionable_cards_min_adtv_{u_pfx}",
                help=f"Filter individual constituent stocks inside each card by minimum 20-day Average Daily Turnover in {adtv_unit_label}."
            )

        if "90" in card_min_rs_choice:
            min_card_rs = 90
        elif "85" in card_min_rs_choice:
            min_card_rs = 85
        elif "80" in card_min_rs_choice:
            min_card_rs = 80
        elif "70" in card_min_rs_choice:
            min_card_rs = 70
        else:
            min_card_rs = 0

        if "Top 3" in card_depth_choice:
            depth_limit = 3
        elif "Top 5" in card_depth_choice:
            depth_limit = 5
        elif "Top 10" in card_depth_choice:
            depth_limit = 10
        elif "Top 15" in card_depth_choice:
            depth_limit = 15
        else:
            depth_limit = 999
            
        effective_card_sort = sort_choice if card_sort_choice == "Same as Top Filter" else card_sort_choice
        cards_df = _sort_groups_df(filtered_df, effective_card_sort)

        with st.expander("🔍 Quick Constituent Drilldown: Inspect All Ranked Stocks in Any Group (#1 to #N)", expanded=False):
            grp_drill_options = cards_df["Industry_Group"].tolist()
            chosen_tab_group = st.selectbox(
                "Select Group to View Full Ranked Constituent Roster:",
                options=grp_drill_options,
                key=f"tab_cards_quick_drill_{u_pfx}",
                format_func=lambda g: f"#{cards_df[cards_df['Industry_Group']==g]['Rank_Today'].iloc[0]}  {g}  (👑 {cards_df[cards_df['Industry_Group']==g]['Apex_Leaders'].iloc[0]} Apex • {cards_df[cards_df['Industry_Group']==g]['Stock_Count'].iloc[0]} Stocks • Grade: {cards_df[cards_df['Industry_Group']==g]['Group_Grade'].iloc[0]})"
            )
            if chosen_tab_group:
                sel_grp_row = cards_df[cards_df["Industry_Group"] == chosen_tab_group].iloc[0]
                full_constits = sel_grp_row.get('Top_Leaders_Full') or sel_grp_row.get('Top_3_Leaders') or []
                filtered_drill = []
                for c in full_constits:
                    rs_ok = (c.get('rs', 0) >= min_card_rs) if min_card_rs > 0 else True
                    adtv_val = c.get('adtv_cr', 0.0)
                    adtv_ok = (adtv_val >= min_card_adtv) if min_card_adtv > 0.0 else True
                    if rs_ok and adtv_ok:
                        filtered_drill.append(c)
                if filtered_drill:
                    df_full_display = pd.DataFrame([
                        {
                            "Rank": f"#{c.get('rank', i+1)}",
                            "Role": f"{c.get('role_icon', '')} {c.get('role_title', '')}".strip(),
                            "Symbol": c.get('ticker', ''),
                            "RS Rating": c.get('rs', 0),
                            "Sortino (3M)": round(c.get('sortino_3m', 0.0), 2),
                            f"CMP ({curr_symbol})": f"{curr_symbol}{c.get('cmp', 0.0):,.2f}",
                            f"20D ADTV ({adtv_unit_label})": f"{curr_symbol}{c.get('adtv_cr', 0.0):.2f}Cr" if not is_us else f"${c.get('adtv_cr', 0.0):.2f}M",
                            "Pivot Dist %": c.get('dist_pivot_str', '0.0%'),
                            "Execution Status": c.get('status', '⏳ BASE')
                        }
                        for i, c in enumerate(filtered_drill)
                    ])
                    st.dataframe(
                        df_full_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "RS Rating": st.column_config.ProgressColumn("RS Rating", format="%d", min_value=0, max_value=99),
                            "Sortino (3M)": st.column_config.NumberColumn("Sortino (3M)", format="%.2f"),
                        }
                    )
                    tv_drill_list = [
                        format_tradingview_symbol(c.get('ticker', ''), c.get('raw_ticker', ''), is_us=is_us, us_tv_map=us_tv_map, prefix_us=True)
                        for c in filtered_drill if c.get('ticker')
                    ]
                    if tv_drill_list:
                        from components import render_tv_export_popover
                        render_tv_export_popover(
                            tv_drill_list,
                            title=f"TradingView Watchlist — {chosen_tab_group} ({len(tv_drill_list)} Stocks)",
                            key=f"igm_drill_tv_{u_pfx}",
                            btn_label=f"📋 Export TV ({len(tv_drill_list)} Stocks)",
                        )
                else:
                    st.info(f"No constituents found with RS ≥ {min_card_rs} in {chosen_tab_group}.")

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
                    
                    leaders_list = g_data.get('Top_Leaders_Full') or g_data.get('Top_3_Leaders') or []
                    filtered_leaders = []
                    for ldr in leaders_list:
                        rs_ok = (ldr.get('rs', 0) >= min_card_rs) if min_card_rs > 0 else True
                        adtv_val = ldr.get('adtv_cr', 0.0)
                        adtv_ok = (adtv_val >= min_card_adtv) if min_card_adtv > 0.0 else True
                        if rs_ok and adtv_ok:
                            filtered_leaders.append(ldr)
                    leaders_slice = filtered_leaders[:depth_limit]
                    
                    grp_s3m = g_data.get("Sortino_3M", 0.0)
                    ldr_s3m = g_data.get("Leader_Sortino_3M", grp_s3m)
                    
                    filter_tags = []
                    if min_card_rs > 0:
                        filter_tags.append(f"RS ≥ {min_card_rs}")
                    if min_card_adtv > 0:
                        adtv_tag_lbl = f"ADTV ≥ ₹{min_card_adtv:.1f}Cr" if not is_us else f"ADTV ≥ ${min_card_adtv:.1f}M"
                        filter_tags.append(adtv_tag_lbl)
                        
                    if filter_tags:
                        joined_tags = " & ".join(filter_tags)
                        header_leaders_info = f"Top Leaders ({len(leaders_slice)} with {joined_tags}):"
                        header_ranked_info = f"Filtered ({joined_tags})"
                    else:
                        header_leaders_info = f"Top Leaders ({len(leaders_slice)} of {g_data['Stock_Count']}):"
                        header_ranked_info = f"Ranked in Group"
                    
                    chips_html_list = []
                    for l_idx, ldr in enumerate(leaders_slice):
                        rank_num = ldr.get('rank', l_idx + 1)
                        st_lbl = ldr.get('status', '⏳ BASE')
                        if "IN BUY ZONE" in st_lbl:
                            b_cls = "ab-buy"
                            b_text = f"🟢 BUY {ldr.get('dist_pivot_str', '')}"
                        elif "RETEST" in st_lbl:
                            b_cls = "ab-retest"
                            b_text = f"🔵 RETEST {ldr.get('dist_pivot_str', '')}"
                        elif "COILING" in st_lbl:
                            b_cls = "ab-coil"
                            b_text = f"⏳ COIL {ldr.get('dist_pivot_str', '')}"
                        elif "EXTENDED" in st_lbl:
                            b_cls = "ab-ext"
                            b_text = f"🟠 EXT {ldr.get('dist_pivot_str', '')}"
                        elif "FAILED" in st_lbl:
                            b_cls = "ab-fail"
                            b_text = "🔴 FAIL"
                        else:
                            b_cls = "ab-base"
                            b_text = "⏳ BASE"
                            
                        if is_us:
                            tv_link = f"https://www.tradingview.com/chart/?symbol={ldr['ticker']}"
                        else:
                            exchange = 'BSE' if str(ldr.get('raw_ticker', '')).endswith('.BO') else 'NSE'
                            tv_link = f"https://www.tradingview.com/chart/?symbol={exchange}%3A{ldr['ticker']}"
                        s_val = ldr.get('sortino_3m', 0.0)
                        s_color = "#34d399" if s_val >= 3.0 else ("#38bdf8" if s_val >= 1.5 else "#94a3b8")
                        
                        role_icon = ldr.get('role_icon')
                        if not role_icon:
                            if rank_num == 1:
                                role_icon = "👑" if ldr.get('rs', 0) >= 70 else "⚠️"
                            elif rank_num == 2:
                                role_icon = "👯" if ldr.get('rs', 0) >= 70 else "⚠️"
                            elif rank_num == 3:
                                role_icon = "🥉" if ldr.get('rs', 0) >= 70 else "⚠️"
                            elif ldr.get('rs', 0) >= 80:
                                role_icon = "🐺"
                            else:
                                role_icon = "⚠️"
                                
                        cur_adtv_cr = ldr.get('adtv_cr', 0.0)
                        if is_us:
                            adtv_disp = f"${cur_adtv_cr:.1f}M"
                        else:
                            if cur_adtv_cr >= 1.0:
                                adtv_disp = f"₹{cur_adtv_cr:.1f}Cr"
                            else:
                                adtv_disp = f"₹{cur_adtv_cr*100:.0f}L"
                        
                        chips_html_list.append(f"""
                        <div class='anchor-chip'>
                            <span class='anchor-sym'>
                                <span style='font-size:0.72rem; color:#64748b; margin-right:4px;'>#{rank_num}</span>
                                <span style='margin-right:2px;'>{role_icon}</span>
                                <a href='{tv_link}' target='_blank' style='color:#38bdf8; text-decoration:none; font-weight:700;'>{ldr['ticker']} ↗</a>
                                <span style='font-size:0.72rem; color:#94a3b8; margin-left:4px;'>RS {ldr['rs']}</span>
                                <span style='font-size:0.70rem; color:{s_color}; font-weight:700; margin-left:4px;' title='3M Sortino: {s_val:.2f}'>S {s_val:.1f}</span>
                            </span>
                            <div style='display:flex; flex-direction:column; align-items:flex-end; line-height:1.2; margin:0 6px;'>
                                <span style='color:#f8fafc; font-weight:700; font-size:0.78rem;'>{curr_symbol}{ldr['cmp']:,.1f}</span>
                                <span style='color:#c084fc; font-size:0.68rem; font-weight:600;' title='20-Day Average Daily Turnover (ADTV)'>{adtv_disp}</span>
                            </div>
                            <span class='anchor-badge {b_cls}'>{b_text}</span>
                        </div>
                        """)
                        
                    if len(leaders_slice) == 0:
                        filter_desc = " & ".join(filter_tags) if filter_tags else "matching criteria"
                        chips_combined = f"<div style='font-size:0.75rem; color:#64748b; padding:16px 10px; text-align:center; font-style:italic; background:rgba(255,255,255,0.02); border-radius:6px; margin-top:6px; border:1px dashed rgba(255,255,255,0.08);'>No constituents with {filter_desc} in this group</div>"
                    else:
                        chips_combined = "".join(chips_html_list)
                    
                    rsnh_pct_v = float(g_data.get("RSNH_Pct", 0.0) or 0.0)
                    blue_pct_v = float(g_data.get("RS_BlueDot_Pct", 0.0) or 0.0)
                    breadth_80_v = float(g_data.get("Breadth_RS80_Pct", 0.0) or 0.0)
                    trad_cnt_v = int(g_data.get("Tradable_Count", g_data["Stock_Count"]))
                    card_html = f"""
                    <div class='terminal-group-card'>
                        <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                            <div>
                                <div class='tg-title' title='{g_data["Industry_Group"]}'>{g_data["Industry_Group"]}</div>
                                <div class='tg-stock-count'>{trad_cnt_v} Tradable ({g_data["Stock_Count"]} Tot) • <b style='color:#c084fc;'>🐺 {g_data["Pack_Hunting_Count"]} RS≥80 ({breadth_80_v:.0f}%)</b> • <b style='color:#38bdf8;' title='Full Group Capped-Mcap (tradable peers): {grp_s3m:.2f}'>🛡️ Grp S(3M) {grp_s3m:.2f}</b> • <b style='color:#34d399;' title='Average 3M Sortino of leading momentum constituents: {ldr_s3m:.2f}'>🏆 Ldr S {ldr_s3m:.1f}</b> • <b style='color:#facc15;'>👑 {g_data.get("Apex_Leaders", 0)} Apex</b> • <b style='color:#60a5fa;'>🔵 {rsnh_pct_v:.0f}% RSNH</b> • <b style='color:#f472b6;'>🎯 {blue_pct_v:.0f}% BlueDot</b></div>
                            </div>
                            <div style='text-align:right;'>
                                <div class='tg-rank-num'>#{g_data["Rank_Today"]}</div>
                                <div style='display:flex; gap:4px; justify-content:flex-end; margin-top:2px; flex-wrap:wrap;'>
                                    <span class='badge-pill {dual_pill_cls}' title='Dual Momentum: {g_data.get("Dual_State", "")}'>{g_data.get("Dual_Badge", "")}</span>
                                    <span class='badge-pill' style='background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.15); font-family:monospace;' title='3-Horizon (1M · 3M · 6M): {g_data.get("Horizon_State", "")}'>{g_data.get("Horizon_Badge", "")}</span>
                                    <span class='badge-pill {"pill-emerald" if g_data.get("Is_Group_Confirmed") else "pill-purple"}' title='Ross Haber Group Confirmation: {g_data.get("Group_Confirmation_Badge", "")}'>{g_data.get("Group_Confirmation_Badge", f"🐺 {g_data['Pack_Hunting_Count']} Pack")}</span>
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
                        <div style='font-size:0.72rem; font-weight:700; color:#64748b; text-transform:uppercase; margin-top:6px; display:flex; justify-content:space-between;'>
                            <span>{header_leaders_info}</span>
                            <span style='font-size:0.70rem; color:#94a3b8; text-transform:none;'>{header_ranked_info}</span>
                        </div>
                        <div class='anchor-chip-row'>
                            {chips_combined}
                        </div>
                    </div>
                    """
                    st.html(clean_html(card_html))
                    
                    # TradingView Watchlist Exporter for this specific Industry
                    all_card_constits = g_data.get('Top_Leaders_Full') or []
                    display_roster = []
                    for c in all_card_constits:
                        rs_ok = (c.get('rs', 0) >= min_card_rs) if min_card_rs > 0 else True
                        adtv_val = c.get('adtv_cr', 0.0)
                        adtv_ok = (adtv_val >= min_card_adtv) if min_card_adtv > 0.0 else True
                        if rs_ok and adtv_ok:
                            display_roster.append(c)
                        
                    tv_leaders_list = [
                        format_tradingview_symbol(l.get('ticker', ''), l.get('raw_ticker', ''), is_us=is_us, us_tv_map=us_tv_map, prefix_us=True)
                        for l in leaders_slice if l.get('ticker')
                    ]
                    tv_leaders_str = ",".join(tv_leaders_list)

                    tv_filtered_list = [
                        format_tradingview_symbol(l.get('ticker', ''), l.get('raw_ticker', ''), is_us=is_us, us_tv_map=us_tv_map, prefix_us=True)
                        for l in display_roster if l.get('ticker')
                    ]
                    tv_filtered_str = ",".join(tv_filtered_list)
                    
                    tv_all_list = [
                        format_tradingview_symbol(l.get('ticker', ''), l.get('raw_ticker', ''), is_us=is_us, us_tv_map=us_tv_map, prefix_us=True)
                        for l in all_card_constits if l.get('ticker')
                    ]
                    tv_all_str = ",".join(tv_all_list)
                    
                    if tv_leaders_str or tv_filtered_str:
                        if filter_tags and len(tv_filtered_list) > len(tv_leaders_list):
                            exp_tv_lbl = f"📋 TradingView ({len(tv_leaders_list)} Ldr / {len(tv_filtered_list)} Flt / {len(tv_all_list)} All)"
                        else:
                            exp_tv_lbl = f"📋 TradingView ({len(tv_leaders_list)} Ldr / {len(tv_all_list)} All)"
                        _tv_ctx = st.popover(exp_tv_lbl, use_container_width=True) if hasattr(st, "popover") else st.expander(exp_tv_lbl, expanded=False)
                        with _tv_ctx:
                            if tv_leaders_str:
                                st.caption(f"**Top {len(tv_leaders_list)} Leaders in {g_data['Industry_Group']}:**")
                                st.code(tv_leaders_str, language="text")
                            if filter_tags and len(tv_filtered_list) > len(tv_leaders_list):
                                st.caption(f"**All {len(tv_filtered_list)} Filtered Stocks ({joined_tags}) in {g_data['Industry_Group']}:**")
                                st.code(tv_filtered_str, language="text")
                            if len(tv_all_list) > max(len(tv_leaders_list), len(tv_filtered_list)):
                                st.caption(f"**All {len(tv_all_list)} Constituents in {g_data['Industry_Group']}:**")
                                st.code(tv_all_str, language="text")
                            st.caption("💡 Click the copy button in the top-right of the code box, then in TradingView press **Alt+I** or **+** in your Watchlist and paste (**Ctrl+V**).")
                    
                    # Individual Card Expander for Full Constituent Roster Table
                    if len(display_roster) > len(leaders_slice):
                        desc_text = " & ".join(filter_tags) if filter_tags else ""
                        exp_lbl = f"📊 View all {len(display_roster)} stocks ({desc_text}) in {g_data['Industry_Group']}" if desc_text else f"📊 View all {g_data['Stock_Count']} stocks in {g_data['Industry_Group']}"
                        with st.expander(exp_lbl, expanded=False):
                            df_card_roster = pd.DataFrame([
                                {
                                    "Rank": f"#{c.get('rank', i+1)}",
                                    "Role": f"{c.get('role_icon', '')} {c.get('role_title', '')}".strip(),
                                    "Symbol": c.get('ticker', ''),
                                    "RS Rating": c.get('rs', 0),
                                    "Sortino": round(c.get('sortino_3m', 0.0), 2),
                                    "Price": f"{curr_symbol}{c.get('cmp', 0.0):,.2f}",
                                    f"20D ADTV ({adtv_unit_label})": f"{curr_symbol}{c.get('adtv_cr', 0.0):.2f}Cr" if not is_us else f"${c.get('adtv_cr', 0.0):.2f}M",
                                    "Status": c.get('status', '⏳ BASE')
                                }
                                for i, c in enumerate(display_roster)
                            ])
                            st.dataframe(
                                df_card_roster,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "RS Rating": st.column_config.ProgressColumn("RS Rating", format="%d", min_value=0, max_value=99),
                                    "Sortino": st.column_config.NumberColumn("Sortino", format="%.2f")
                                }
                            )

    # -------------------------------------------------------------
    # TAB: NICHE & EXCLUDED GROUPS (<4 TRADABLE PEERS)
    # -------------------------------------------------------------
    with tab_niche:
        st.markdown(clean_html(f"""
        <div style='background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;'>
            <h4 style='margin:0; font-size:1.10rem; font-weight:800; color:#f8fafc;'>
                🔬 Niche, Micro & Excluded Sub-Groups (Grade: NR, Rank: 999)
            </h4>
            <p style='margin:4px 0 0 0; font-size:0.82rem; color:#94a3b8;'>
                Under William O'Neil / MarketSurge 2.0 standards, groups with <b>fewer than {MIN_RANKED_GROUP_SIZE_CANONICAL} tradable non-duplicate-class stocks</b> or unclassified/shell labels are excluded from the ranked leaderboard so thin 1-stock spikes cannot distort Top-20 ranks. Their constituent stocks remain viewable here and in individual stock lookups.
            </p>
        </div>
        """), unsafe_allow_html=True)

        if df_niche is not None and not df_niche.empty:
            niche_disp_cols = [c for c in [
                "Industry_Group", "Exclusion_Reason", "Group_Grade", "Tradable_Count",
                "Stock_Count", "RSNH_Pct", "Return_1M", "Top_Leaders_Display"
            ] if c in df_niche.columns]
            st.dataframe(
                df_niche[niche_disp_cols],
                column_config={
                    "Industry_Group": st.column_config.TextColumn("Niche / Excluded Group", width=230),
                    "Exclusion_Reason": st.column_config.TextColumn("Reason", width=180),
                    "Group_Grade": st.column_config.TextColumn("Grade", width=70),
                    "Tradable_Count": st.column_config.NumberColumn("Tradable", format="%d", width=85),
                    "Stock_Count": st.column_config.NumberColumn("Total", format="%d", width=80),
                    "RSNH_Pct": st.column_config.NumberColumn("RSNH %", format="%.1f%%", width=85),
                    "Return_1M": st.column_config.NumberColumn("1M %", format="%.1f%%", width=85),
                    "Top_Leaders_Display": st.column_config.TextColumn("Constituents & Action State", width=360),
                },
                use_container_width=True,
                hide_index=True,
                height=520,
            )
        else:
            st.info("No niche (<4 tradable peers) or excluded groups present in this view.")

    # -------------------------------------------------------------
    # TAB 4: METHODOLOGY EXPANDER
    # -------------------------------------------------------------
    with tab_guide:
        st.markdown(r"""
        ### MarketSurge 2.0 & The William J. O'Neil 50% Rule
        *Studies of the greatest stock market winners of the last 100 years prove that **over 50% of an individual stock's total price advance is directly driven by the strength of its underlying industry group**.*

        Trading high-RS stocks in lagging or distributing sectors is like swimming against a 50-knot rip current. Conversely, buying actionable leaders in the **Top 20% Industry Groups (A+/A)** gives you institutional tailwinds from sovereign wealth funds, mutual funds, and hedge funds buying the entire sector in size.

        ---

        ### 🧮 MarketSurge 2.0 Engine Architecture

        #### 1. Universe Hygiene, Share-Class Deduplication & $\ge 4$ Tradable Group Floor
        - **Closed Canonical Taxonomy (145 Groups)**: Every stock string (Yahoo Finance, TradingView, NSE/BSE, GICS) is normalized through exact case-insensitive canonical synonym resolution; `Shell Companies` and `General / Unclassified` are partitioned out.
        - **Liquidity Floors (Two-Stage Gate)**: Constituents must satisfy the minimum index inclusion floor (NIFTY Total Market / S&P 500 / Nasdaq-100 membership OR 20-day ADTV $\ge$ ₹80L (`₹0.8 Cr`) in India / $\$250\text{K}$ in US) to enter `Tradable_Count`. Institutional leadership slots (`👑 Alpha #1`, `👯 Sister #2`, `🥉 #3`) further prioritize **Tier $\ge 2$ sponsorship** ($\ge ₹2.5\text{ Cr}$ ADTV in India / $\ge \$1.0\text{M}$ in US) with $\ge 126$ trading bars (`6M`) so thin micro-caps or recent IPOs cannot hijack institutional bellwethers.
        - **Dual-Class Dedup**: Multiple share classes of one corporation (`GOOG`/`GOOGL`, `FOX`/`FOXA`, `.NS`/`.BO`, DVR classes) collapse to the single highest-ADTV class so no corporation is double-weighted.
        - **Statistical $\ge 4$ Peer Minimum**: Canonical groups with $< 4$ tradable peers (or `< 3` in curated Indian Alpha Themes) are quarantined to **Niche / Unranked (`NR`, Rank `999`)**.

        #### 2. Dynamic Split-Adjusted 25% Capped Market-Cap Index + Lagged Weights
        In **Capped Market-Cap (Default)** mode, daily constituent market caps are reconstructed without lookahead via:
        $$\text{Cap}_i(t) = \left(\frac{\text{MarketCap}_{i,\text{today}}}{\text{Close}_{i,\text{today}}}\right) \times \text{Close}_i(t)$$
        Each member's daily weight is capped iteratively at $\max(25\%, 1 / N_{\text{active}})$ using lagged closing weights $W_{i,t-1}$ and chain-linked strictly across active non-zero bars with corporate-action jump clipping.

        #### 3. Smoothed 6-Month Multi-Checkpoint RS (`21/42/63/95/126d`)
        To eliminate single-day base-effect jumps while honoring O'Neil's 6-month institutional cycle (double-weighting the recent quarter via nested accumulation), both stock and group RS use 3-bar averaged checkpoints at **21d (30%), 42d (25%), 63d (20%), 95d (15%), and 126d (10%)**, with partial-history shrinkage for `< 63`-bar stocks.
        Cross-sectional 0–99 percentiles are anchored strictly across **active tradable peers at each historical bar `t`** (zero survivorship/future leakage).

        #### 4. Option B Blended Ranking (`75% Price RS + 25% Breadth`)
        - **Price RS Pillar (75%)**: Smoothed 6M Capped Market-Cap Group RS Percentile (`Pure_Price_RS` / `Pure_RS_Rank` always shown separately for full transparency).
        - **Breadth Pillar (25%)**: `0.55 × Breadth (RS ≥ 80 %) + 0.25 × Breadth (>50 SMA %) + 0.20 × Near 52W High & Stage-2 (%)` evaluated across active tradable peers at every historical snapshot (`0, 5, 15, 21, 30, 63, 126` bars ago).

        #### 5. Ross Haber RSNH% & O'Neil Pre-Breakout Blue Dot Diagnostics
        - **🔵 `RSNH %` (RS Line New High %)**: Percentage of active tradable peers (requiring $\ge \min(126, \max(40, 0.6 \times W))$ bars) whose Stock/Benchmark RS line hit within $1\%$ of its 52-week high over the trailing 5 sessions, keeping total `Tradable_Count` as denominator so brand-new IPOs cannot game RSNH%.
        - **🎯 `Blue Dot %` (Pre-Breakout RS Leader %)**: Subset of mature peers coiling inside a constructive base (`75%–99.5%` of 52W high) whose RS line hits a 52-week or 63-day base high ahead of price — spotlighting institutional accumulation before price breaks out.

        ---

        #### 6. Julius de Kempenaer (CMT) & Himanshu Sharma (CMT) RRG Sector Velocity
        Relative Rotation Graphs (RRG) evaluate sector momentum as a **two-dimensional normalized vector** centered at (100.0, 100.0) rather than a single static 1-year percentile:
        - **JdK RS-Ratio (X-Axis)**: Measures medium-term relative trend outperformance vs the Nifty 500 benchmark (^CRSLDX), double-smoothed with a 10-period WMA.
        - **JdK RS-Momentum (Y-Axis)**: Measures the rate of change / acceleration of the RS-Ratio over a 10-period lookback.
        
        **The 4 Rotational Quadrants**:
        - **🟢 Leading ($X \ge 100, Y \ge 100$)**: Sustained relative trend and accelerating speed. Core institutional leaders.
        - **🟡 Weakening ($X \ge 100, Y < 100$)**: Trend is positive, but momentum has rolled over. Institutional distribution alert.
        - **🔴 Lagging ($X < 100, Y < 100$)**: Sustained downtrend and decelerating momentum. Zero capital allocation.
        - **🔵 Improving ($X < 100, Y \ge 100$)**: **The Alpha Sweet Spot**. Relative trend is still below 100, but momentum has inflected sharply upward. Catches stage-1 breakouts and pocket pivots weeks before 1-year RS reaches 80.

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

        #### 4. 🛡️ Group Sortino vs 🏆 Leader Sortino (Ldr S)
        To eliminate distortion caused by laggards in large groups:
        - **Group Sortino 3M (`🛡️ Grp S`)**: Evaluates the synthetic curve of **ALL constituents** in the industry group (equal-weighted). In large sectors (e.g. 93 pharma stocks), laggards or micro-cap drags can lower the broad group Sortino even when leading stocks are soaring.
        - **Leader Sortino 3M (`🏆 Ldr S`)**: Calculates the arithmetic mean 3-Month Sortino of **up to the Top 10 leading constituents** with **RS Rating $\ge 80$**. This provides a pure reading of downside-risk protection for the actual actionable compounders institutions are buying.
        - *Example*: In `Drug Manufacturers`, `Grp S(3M)` is `-0.03` (due to 57 laggard pharma stocks dragging the broad index), while `Ldr S(3M)` is **`+5.28`** across the top 10 momentum leaders (`AHCL`, `APLLTD`, `LAURUSLABS`, `DIVISLAB`, etc.).

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
    target_group = st.session_state.get(sel_widget_key, st.session_state.get(sel_grp_key))
    if not target_group or target_group not in group_options:
        target_group = group_options[0]
    
    st.session_state[sel_grp_key] = target_group
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
    prev_sel_grp = st.session_state.get(sel_grp_key)
    st.session_state[sel_grp_key] = chosen_group
    if prev_sel_grp and chosen_group and str(chosen_group) != str(prev_sel_grp):
        _sticky_pin = st.session_state.get(f"kt_igm_sticky_pin_{u_pfx}")
        if isinstance(_sticky_pin, dict) and str(_sticky_pin.get("group") or "") != str(chosen_group):
            st.session_state.pop(f"kt_igm_sticky_pin_{u_pfx}", None)

    if chosen_group:
        df_constits, curve_df, tv_copy_box = get_group_deep_dive_data(
            chosen_group,
            taxonomy=tax_key,
            universe_scope=universe_scope,
            weight_mode=weight_mode_key,
        )
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

        trad_cnt = int(grp_row.get('Tradable_Count', grp_row['Stock_Count']))
        breadth_80 = float(grp_row.get('Breadth_RS80_Pct', (grp_row['Pack_Hunting_Count'] / max(1, trad_cnt)) * 100.0))
        breadth_50sma = float(grp_row.get('Breadth_50SMA_Pct', 0.0) or 0.0)
        rsnh_pct_v = float(grp_row.get('RSNH_Pct', 0.0) or 0.0)
        blue_dot_v = float(grp_row.get('RS_BlueDot_Pct', 0.0) or 0.0)
        pure_rs_rank_v = int(grp_row.get('Pure_RS_Rank', grp_row['Rank_Today']))
        blended_v = float(grp_row.get('Blended_Score', grp_row['Comp_RS']))
        c6m_class = "pill-emerald" if grp_row['Return_6M'] >= 0 else "pill-rose"

        hud_html = f"""
        <div class='hud-grid' style='margin-top:8px; margin-bottom:20px;'>
            <div class='hud-panel'>
                <div class='hud-panel-title'>🏆 Group Hierarchy Rank (Blended / Pure)</div>
                <div class='hud-panel-val'>#{grp_row['Rank_Today']} <span style='font-size:0.85rem; color:#64748b; font-weight:500;'>of {len(df_matrix)} (Pure #{pure_rs_rank_v})</span></div>
                <div class='hud-panel-sub'>
                    <span class='badge-pill {r_delta_class}'>▲ {r_delta_sign}{grp_row['Delta_1M']} in 1M</span>
                    <span style='color:#64748b;'>• 1W: {r_1w_sign}{grp_row['Delta_1W']} • Score: {blended_v:.1f}</span>
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
                    <span style='color:#94a3b8;'>Comp RS: <b>{grp_row['Comp_RS']}</b> • >50SMA: <b>{breadth_50sma:.0f}%</b></span>
                </div>
            </div>
            <div class='hud-panel pack'>
                <div class='hud-panel-title'>🐺 Breadth & Ross Haber RSNH Diagnostics</div>
                <div class='hud-panel-val'>{grp_row['Pack_Hunting_Count']}/{trad_cnt} <span style='font-size:0.80rem; color:#c084fc; font-weight:600;'>RS ≥ 80 ({breadth_80:.0f}%)</span></div>
                <div class='hud-panel-sub'>
                    <span class='badge-pill pill-cyan'>🔵 RSNH {rsnh_pct_v:.0f}%</span>
                    <span class='badge-pill pill-purple'>🎯 Blue Dot {blue_dot_v:.0f}%</span>
                    <span class='badge-pill pill-emerald'>👑 {grp_row.get('Apex_Leaders', 0)} Apex</span>
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
                if "Group RS Line" in curve_df.columns:
                    fig.add_trace(go.Scatter(
                        x=curve_df.index,
                        y=curve_df["Group RS Line"],
                        name="Group RS Line",
                        mode="lines",
                        yaxis="y2",
                        line=dict(color="#38bdf8", width=1.8),
                        hovertemplate="<b>Group RS Line</b>: %{y:.1f}<extra></extra>"
                    ))

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(15, 23, 42, 0.4)",
                    plot_bgcolor="rgba(15, 23, 42, 0.4)",
                    margin=dict(l=20, r=40, t=15, b=20),
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
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                    yaxis2=dict(title="RS Line", overlaying="y", side="right", showgrid=False, tickfont=dict(color="#38bdf8", size=10))
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
                            <span style='font-size:1.15rem; font-weight:800; color:#f8fafc;'>{curr_symbol}{a_row.get("CMP", a_row.get("CMP (₹)", 0.0)):,.2f}</span>
                            <span style='font-size:0.75rem; color:#94a3b8;'>Pivot: <b style='color:#f8fafc;'>{curr_symbol}{a_row["Pivot Price"]:,.1f}</b> ({a_row["Dist Pivot %"]:+.1f}%)</span>
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
                ["All Stocks", "👑 Apex Leaders (RS≥80 & S≥3)", "🐺 RS ≥ 80 Only", "🔵 RS New Highs / 🎯 Blue Dots", "🟢 Buy Zone & Retest"],
                horizontal=True,
                key=f"igm_constit_filter_{u_pfx}"
            )

        disp_constits_df = df_constits.copy()
        if "Apex Leaders" in constit_filter:
            disp_constits_df = disp_constits_df[(disp_constits_df["RS Rating"] >= 80) & (disp_constits_df["Sortino 3M"] >= 3.0)]
        elif "RS ≥ 80" in constit_filter:
            disp_constits_df = disp_constits_df[disp_constits_df["RS Rating"] >= 80]
        elif "RS New Highs" in constit_filter and "RS Line High" in disp_constits_df.columns:
            disp_constits_df = disp_constits_df[disp_constits_df["RS Line High"].str.contains("Blue Dot|RS NH", na=False)]
        elif "Buy Zone" in constit_filter:
            disp_constits_df = disp_constits_df[disp_constits_df["Execution Status"].str.contains("BUY|RETEST")]

        if len(disp_constits_df) < len(df_constits):
            st.caption(f"Showing **{len(disp_constits_df)} of {len(df_constits)}** constituents matching *{constit_filter}*.")

        constit_cols = [
            "Symbol", "Tier / Liquidity", "ADTV", "RS Line High", "CMP (₹)", "1D %", "1W %", "1M %", "3M %", "1Y %",
            "RS Rating", "Smooth RS", "Sortino 3M", "Sortino 6M", "Pivot Price", "Dist Pivot %", "Dist 21 EMA %",
            "Execution Status", "TradingView_URL"
        ]
        constit_cols = [c for c in constit_cols if c in disp_constits_df.columns]
        disp_constits = disp_constits_df[constit_cols].copy()

        c_constit_cfg = {
            "Symbol": st.column_config.TextColumn("Symbol", width=100),
            "Tier / Liquidity": st.column_config.TextColumn("Tier", width=110),
            "ADTV": st.column_config.TextColumn("ADTV (20D)", width=90),
            "RS Line High": st.column_config.TextColumn("RS Line", width=100, help="Ross Haber RS Line New High (🔵) or O'Neil Pre-Breakout Blue Dot (🎯)"),
            "CMP (₹)": st.column_config.NumberColumn(f"CMP ({curr_symbol})", format=f"{curr_symbol}%.2f", width=110),
            "1D %": st.column_config.NumberColumn("1D %", format="%.2f%%", width=85),
            "1W %": st.column_config.NumberColumn("1W %", format="%.2f%%", width=85),
            "1M %": st.column_config.NumberColumn("1M %", format="%.2f%%", width=85),
            "3M %": st.column_config.NumberColumn("3M %", format="%.2f%%", width=85),
            "1Y %": st.column_config.NumberColumn("1Y %", format="%.2f%%", width=85),
            "RS Rating": st.column_config.ProgressColumn("RS Score", min_value=0, max_value=99, format="%d", width=110),
            "Smooth RS": st.column_config.NumberColumn("Smooth RS (RS/ATR)", format="%.2fx", width=105, help="Composite RS / 20D ATR% (Super-Leaders ≥ 10x)"),
            "Sortino 3M": st.column_config.NumberColumn("Sortino 3M", format="%.2f", width=95, help="3-Month Downside-adjusted Sortino ratio (MAR=6.5%)"),
            "Sortino 6M": st.column_config.NumberColumn("Sortino 6M", format="%.2f", width=95, help="6-Month Downside-adjusted Sortino ratio (MAR=6.5%)"),
            "Pivot Price": st.column_config.NumberColumn(f"25D Pivot ({curr_symbol})", format=f"{curr_symbol}%.2f", width=110),
            "Dist Pivot %": st.column_config.NumberColumn("Dist Pivot", format="%.1f%%", width=100, help="Distance from 25-day pivot"),
            "Dist 21 EMA %": st.column_config.NumberColumn("Dist 21 EMA", format="%.1f%%", width=105, help="Distance from 21-day EMA"),
            "Execution Status": st.column_config.TextColumn("CANSLIM Action State", width=150),
            "TradingView_URL": st.column_config.LinkColumn("Chart", display_text="Open ↗", width=80)
        }

        _cv_bar1, _cv_bar2, _cv_bar3 = st.columns([1.4, 1.3, 1.3])
        with _cv_bar1:
            _constit_view = st.radio(
                "Constituent View:",
                ["📋 Table", "📊 Diverging Bars (1D / 1M)"],
                horizontal=True,
                key=f"igm_constit_view_{u_pfx}",
            )
        with _cv_bar2:
            if "Bars" in _constit_view:
                _bar_metric = st.radio(
                    "Momentum Axis:",
                    ["1D %", "1M %"],
                    horizontal=True,
                    key=f"igm_constit_bar_metric_{u_pfx}",
                )
            else:
                _bar_metric = "1D %"
        with _cv_bar3:
            if not disp_constits.empty and "Symbol" in disp_constits.columns:
                _fl_opts = disp_constits["Symbol"].astype(str).tolist()
                _fl_pick_c1, _fl_pick_c2 = st.columns([1.6, 1.0])
                with _fl_pick_c1:
                    _fl_sym = st.selectbox(
                        "Quick Focus List:",
                        _fl_opts,
                        key=f"igm_fl_pick_{u_pfx}",
                        label_visibility="collapsed",
                    )
                with _fl_pick_c2:
                    if st.button("⭐ Focus", key=f"igm_fl_add_{u_pfx}", use_container_width=True):
                        from database import add_to_focus_list
                        _mkt_code = "US" if is_us else "IN"
                        if add_to_focus_list(str(_fl_sym).upper().strip(), _mkt_code):
                            st.toast(f"⭐ Added {_fl_sym} to {_mkt_code} Focus List!")

        if "Bars" in _constit_view:
            from components import build_constituent_diverging_bar_fig
            _div_fig = build_constituent_diverging_bar_fig(disp_constits_df, metric_col=_bar_metric, max_bars=20)
            if _div_fig.data:
                st.plotly_chart(_div_fig, use_container_width=True)
            else:
                st.info(f"No numeric {_bar_metric} data available for diverging bars.")
        else:
            st.dataframe(
                disp_constits,
                column_config=c_constit_cfg,
                use_container_width=True,
                hide_index=True
            )

if __name__ == "__main__":
    main()