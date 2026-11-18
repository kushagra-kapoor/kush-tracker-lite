"""
Shared Leader Radar Cockpit UI & Filtering Pipeline
===================================================
Unified Institutional Momentum Radar shared across `app.py` (India ^CRSLDX & US ^IXIC Top 100)
and `pages/01_Master_RS_Screener.py` (IN/US, 750/ALL, 100/250/3000 cohorts).

Supports:
  - Immutable bull 21-EMA support mode (preserving regime & Beach Ball parity)
  - Bear / Correction structure modes: Defending 50 SMA (>= -4%), Ross Haber 65 EMA (>= -2%),
    and Undercut & Rally Window (50S/65E Pocket)
  - Structure-aware VCP/Pivot filters, Sortino, Rai RS persistence, Blue Dot & CANSLIM confluence
  - One-click Bear Stalking Preset (pre-widget `on_click` callback) & TradingView Watchlist Export
"""

from __future__ import annotations

import hashlib
import json
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


DEFAULT_STRUCTURE_OPTION = "🟢 Defending 21 EMA Zone (≥ -2%) [Default]"
HOLDING_50_OPTION = "🛡️ Defending 50 SMA Zone (≥ -4%)"
HOLDING_65_OPTION = "📐 Ross Haber 65 EMA Floor (≥ -2%)"
BELOW_21_RESCUED_OPTION = "🩹 Below 21 EMA Pullbacks Only (Holding 50S / 65E)"
BEAR_POCKET_OPTION = "🎯 Undercut & Rally Window (50S/65E Pocket)"
HEALTHY_50_OPTION = "💪 Healthy Above 50 SMA (≥ 0%)"
ALL_LEADERS_OPTION = "🌐 All Qualifying Leaders (No Structure Gate)"

STRUCTURE_FILTER_OPTIONS = [
    DEFAULT_STRUCTURE_OPTION,
    HOLDING_50_OPTION,
    HOLDING_65_OPTION,
    BELOW_21_RESCUED_OPTION,
    BEAR_POCKET_OPTION,
    HEALTHY_50_OPTION,
    "🎯 Actionable Buy Zone (-2% to +5% vs 21 EMA)",
    "🎯 Tight Pullback Pocket (-2% to +3%)",
    "🧲 21 EMA Test / Shakeout (-2% to 0%)",
    "🛡️ 21 EMA Cushion (+3% to +5%)",
    "🚀 Power Extended (> +5%)",
    ALL_LEADERS_OPTION,
]

BEAR_STRUCTURE_OPTIONS = {
    HOLDING_50_OPTION,
    HOLDING_65_OPTION,
    BELOW_21_RESCUED_OPTION,
    BEAR_POCKET_OPTION,
}

VCP_ACTIVE_PIVOT_OPTION = "⚡ Low-Risk Pivot (Active Support)"
VCP_STRICT_21_PIVOT_OPTION = "🔒 Tight 21 EMA Pivot Only (Coil + Near 21 EMA)"
VCP_FILTER_OPTIONS = [
    "All Price Action",
    "🎯 All VCP Tight Coils (ID / NR7 / 5D Coil / 3WT)",
    "💎 Inside Day / NR7 Only (Ultra-Tight Stop)",
    VCP_ACTIVE_PIVOT_OPTION,
    VCP_STRICT_21_PIVOT_OPTION,
    "🟢 Strong Bar Close Only (DCR ≥ 70% Buyers in Control)",
    "🔥 High RVOL Expansion Today (Vol ≥ 1.5x Avg)",
]

STAGE2_OPTION = "📈 Stage 2 Only (Above 50 & 200 SMA)"
STAGE2_OR_65_OPTION = "📐 Stage 2 / 65-EMA Recovery (Inst & Floor Intact)"
CANSLIM_OPTIONS = [
    "All Fundamentals",
    "💎 True Market Leader (C+A Growth + ROE/Inst)",
    "🔥 Double-25% CANSLIM (EPS ≥ 25% & Sales ≥ 25%)",
    "🚀 High EPS or Sales Growth (≥ 25%)",
    "🏛️ Quality ROE (≥ 15%) & Positive EPS",
    STAGE2_OPTION,
    STAGE2_OR_65_OPTION,
]

SORT_CLOSEST_21 = "🎯 Closest to 21 EMA Pivot (Actionable Pullbacks First)"
SORT_CLOSEST_50 = "🛡️ Closest to 50 SMA (Bear Floor)"
SORT_CLOSEST_65 = "📐 Closest to 65 EMA (Haber Floor)"


def _build_sort_options(h_days: int, adtv_col_label: str) -> list[str]:
    return [
        "Smooth RS / ATR (High to Low)",
        "RS Rating (1-99) (High to Low)",
        "20D ADR % (Fastest Movers First)",
        adtv_col_label + " (High to Low)",
        SORT_CLOSEST_21,
        SORT_CLOSEST_50,
        SORT_CLOSEST_65,
        "vs 52W High % (Closest to High First)",
        "VCP Tightness (Tightest 5D Coil First)",
        "Today DCR % (Strongest Bar Close First)",
        "Today RVOL (Highest Volume Surge First)",
        f"Ameet Rai {h_days}D RS Days % (High to Low)",
        "50D Up/Down Volume Ratio (High to Low)",
        "Inst Ownership Δ QoQ (High to Low)",
        "EPS Growth % (High to Low)",
        "Sales Growth % (High to Low)",
        "Sortino 3M Ratio (High to Low)",
        "Top RS Rank (1 to N)",
        "Industry Group Rank (#1 to #N)",
        "vs 21 EMA % (High to Low)",
        "vs 50 SMA % (High to Low)",
        "vs 65 EMA % (High to Low)",
    ]


PRESET_SPECS = {
    "bull": {
        "label": "🐂 Bull Power Leaders",
        "help": "Confirmed Uptrend: Defending 21 EMA + A+/A/B Groups + Top-3 Group Roles + Min 20D RS Days 40% + Min RS 88 + Max Off 52W High 15% + Min ADR 3.0%.",
        "overrides": {
            "grade_multi": ["A+", "A", "B"],
            "rai_slider_20": 40.0,
            "min_rs": 88,
            "max_52w_slider": 15.0,
            "role_choice": "🏆 Top 3 Group Leaders",
            "min_adr": 3.0,
        },
        "empty_hint": "0 matches in this cohort under Bull Power filters (RS ≥ 88, ≤ 15% off 52W High, ADR ≥ 3%, A+/A/B Top-3 Roles). Try switching Cohort Cap to Top 250 / All Stocks in Master RS Screener, or switching to Sideways Tight Coils.",
    },
    "sideways": {
        "label": "⚖️ Sideways Tight Coils",
        "help": "Choppy / Rotational Tape: Defending 21 EMA + All VCP Tight Coils + High Sortino ≥ 1.5 + A+/A/B Groups + Min RS 85 + Max Off 52W High 20% + Closest to 21 EMA.",
        "overrides": {
            "sortino": "⚡ High Sortino (≥ 1.5)",
            "grade_multi": ["A+", "A", "B"],
            "min_rs": 85,
            "max_52w_slider": 20.0,
            "vcp_choice": "🎯 All VCP Tight Coils (ID / NR7 / 5D Coil / 3WT)",
            "sort": SORT_CLOSEST_21,
        },
        "empty_hint": "0 tight VCP coils matched Sideways filters (Sortino ≥ 1.5, RS ≥ 85, ≤ 20% off 52W High). Try relaxing Sortino or Group Grades, or switching Cohort Cap to Top 250 / All Stocks.",
    },
    "bear": {
        "label": "🐻 Bear Stalking (65 EMA)",
        "help": "Correction / FTD Hunt: Ross Haber 65 EMA Floor (≥ -2%) + 3M/52W RS Line Highs + Min RS 85 + Max Off 52W High 25% + Closest to 65 EMA.",
        "overrides": {
            "filter_choice": HOLDING_65_OPTION,
            "min_rs": 85,
            "max_52w_slider": 25.0,
            "rsline_choice": "🔷 All 3M & 52W RS Line Highs",
            "sort": SORT_CLOSEST_65,
        },
        "empty_hint": "0 leaders matched Bear Stalking (65 EMA Floor, RS ≥ 85, 3M/52W RS Line Highs, ≤ 25% off 52W High). Try widening Cohort Cap to Top 250 / All Qualifying Stocks or clearing RS Line Signal.",
    },
}


def _baseline_filter_defaults(is_master: bool = False) -> dict:
    """Canonical default values for every resettable Leader Radar filter widget (never resets ADTV gate)."""
    defaults = {
        "filter_choice": DEFAULT_STRUCTURE_OPTION,
        "sortino": "All Sortino Ratios",
        "bb": "All Leaders",
        "grade_multi": [],
        "rai_slider_15": 0.0,
        "rai_slider_20": 0.0,
        "rai_slider_40": 0.0,
        "min_rs": 0,
        "max_52w_slider": 100.0,
        "role_choice": "All Group Roles",
        "rsline_choice": "All RS Line States",
        "funda_choice": "All Fundamentals",
        "vcp_choice": "All Price Action",
        "min_adr": 0.0,
        "stage_choice": "All Stages & Caps",
        "sort": "Smooth RS / ATR (High to Low)",
        "search": "",
    }
    if is_master:
        defaults["rai_horizon"] = "20D Core"
    return defaults


def _reset_radar_filters_cb(prefix: str, is_master: bool = False) -> None:
    """Pre-widget callback restoring clean default bull radar filter state (including dynamic Rai sliders)."""
    for suffix, val in _baseline_filter_defaults(is_master=is_master).items():
        st.session_state[f"{prefix}_{suffix}"] = list(val) if isinstance(val, list) else val


def _apply_preset_cb(prefix: str, preset_id: str, is_master: bool = False) -> None:
    """Atomic pre-widget preset callback: resets all filter state first, then applies exact preset spec."""
    _reset_radar_filters_cb(prefix, is_master=is_master)
    spec = PRESET_SPECS.get(preset_id, {})
    for suffix, val in spec.get("overrides", {}).items():
        st.session_state[f"{prefix}_{suffix}"] = list(val) if isinstance(val, list) else val


def _apply_bear_preset_cb(prefix: str, is_master: bool = False) -> None:
    """Backwards-compatible alias applying the upgraded Bear Stalking (65 EMA) preset cleanly."""
    _apply_preset_cb(prefix=prefix, preset_id="bear", is_master=is_master)


def _sanitize_widget_state(key: str, valid_options: list, default_val, is_multi: bool = False) -> None:
    """Heals stale session_state selectbox/multiselect values before widget rendering to prevent ValueError."""
    if key not in st.session_state:
        return
    if is_multi:
        cur = st.session_state[key]
        if not isinstance(cur, list):
            st.session_state[key] = list(default_val)
        else:
            st.session_state[key] = [v for v in cur if v in valid_options]
    else:
        if st.session_state[key] not in valid_options:
            st.session_state[key] = default_val


def _detect_active_preset(live_values: dict, is_master: bool = False) -> str | None:
    """Returns 'bull' / 'sideways' / 'bear' if live filters match a preset, None for default, or 'custom'."""
    base = _baseline_filter_defaults(is_master=is_master)
    check_keys = [k for k in base.keys() if k not in ("rai_slider_15", "rai_slider_40", "search") or k in live_values]

    def _matches(expected_map: dict) -> bool:
        for k in check_keys:
            if k not in live_values:
                continue
            cur = live_values[k]
            exp = expected_map[k]
            if isinstance(exp, list):
                if set(cur or []) != set(exp):
                    return False
            elif isinstance(exp, float):
                if abs(float(cur) - exp) > 1e-6:
                    return False
            elif isinstance(exp, int):
                if int(cur) != exp:
                    return False
            else:
                if cur != exp:
                    return False
        return True

    for pid, spec in PRESET_SPECS.items():
        expected = dict(base)
        expected.update(spec["overrides"])
        if _matches(expected):
            return pid
    if _matches(base) and not str(live_values.get("search", "") or "").strip():
        return None
    return "custom"


def format_leader_tv_symbols(df: pd.DataFrame, is_us: bool) -> list[str]:
    """Extracts TradingView watchlist symbols cleanly without broken manual `.BO` / `BRK-B` mutations."""
    if df is None or df.empty:
        return []
    if 'TV_Symbol' in df.columns:
        out = [str(x).strip() for x in df['TV_Symbol'].tolist() if str(x).strip()]
        if out:
            return out
    tickers = df['Ticker'].tolist() if 'Ticker' in df.columns else []
    if is_us:
        return [str(t).strip().replace('-', '.') for t in tickers if str(t).strip()]
    raws = df['Raw_Ticker'].tolist() if 'Raw_Ticker' in df.columns else tickers
    res = []
    for t, r in zip(tickers, raws):
        ex = 'BSE' if str(r).endswith('.BO') else 'NSE'
        res.append(f"{ex}:{str(t).strip().replace('-SM', '').replace('-ST', '').replace('-BE', '')}")
    return res


def apply_leader_radar_filters(
    df_lead: pd.DataFrame,
    *,
    filter_choice: str = DEFAULT_STRUCTURE_OPTION,
    sortino_choice: str = "All Sortino Ratios",
    bb_choice: str = "All Leaders",
    selected_grades: list | None = None,
    min_rai: float = 0.0,
    rai_col: str = "RS_Days_20D_Pct",
    min_rs: int = 0,
    max_off_52w: float = 100.0,
    role_choice: str = "All Group Roles",
    rs_line_choice: str = "All RS Line States",
    funda_choice: str = "All Fundamentals",
    vcp_choice: str = "All Price Action",
    min_adr: float = 0.0,
    stage_choice: str = "All Stages & Caps",
    sort_by: str = "Smooth RS / ATR (High to Low)",
    search_q: str = "",
) -> pd.DataFrame:
    """
    Pure stateless filtering & sorting pipeline for Leader Radar DataFrames.
    Supports bear-mode structure anchors without breaking 21 EMA bull defaults.
    """
    if df_lead is None or df_lead.empty:
        return pd.DataFrame() if df_lead is None else df_lead.copy()

    filtered_df = df_lead.copy()
    is_bear_mode = filter_choice in BEAR_STRUCTURE_OPTIONS

    # 1. Structure Filter (Bull 21-EMA modes + Bear 50-SMA / 65-EMA / Undercut & Rally modes)
    if filter_choice == DEFAULT_STRUCTURE_OPTION:
        if 'Holding_21EMA_Zone' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Holding_21EMA_Zone']]
    elif filter_choice == HOLDING_50_OPTION:
        if 'Holding_50SMA_Zone' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Holding_50SMA_Zone']]
    elif filter_choice == HOLDING_65_OPTION:
        if 'Holding_65EMA_Zone' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Holding_65EMA_Zone']]
    elif filter_choice == BELOW_21_RESCUED_OPTION:
        h50_col = filtered_df['Holding_50SMA_Zone'] if 'Holding_50SMA_Zone' in filtered_df.columns else False
        h65_col = filtered_df['Holding_65EMA_Zone'] if 'Holding_65EMA_Zone' in filtered_df.columns else False
        h21_col = filtered_df['Holding_21EMA_Zone'] if 'Holding_21EMA_Zone' in filtered_df.columns else False
        filtered_df = filtered_df[(~h21_col) & (h50_col | h65_col)]
    elif filter_choice == BEAR_POCKET_OPTION:
        if 'Bear_Pivot_Pocket' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Bear_Pivot_Pocket']]
    elif filter_choice == HEALTHY_50_OPTION:
        if 'Dist_50SMA' in filtered_df.columns:
            mask_h50 = filtered_df['Dist_50SMA'].notna() & (filtered_df['Dist_50SMA'] >= 0.0)
            if 'Heavy_Dist_Break' in filtered_df.columns:
                mask_h50 = mask_h50 & (~filtered_df['Heavy_Dist_Break'])
            filtered_df = filtered_df[mask_h50]
    elif filter_choice == "🎯 Actionable Buy Zone (-2% to +5% vs 21 EMA)" and 'Holding_21EMA_Zone' in filtered_df.columns and 'Dist_21EMA' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Holding_21EMA_Zone'] & (filtered_df['Dist_21EMA'] >= -2.0) & (filtered_df['Dist_21EMA'] <= 5.0)]
    elif filter_choice == "🎯 Tight Pullback Pocket (-2% to +3%)" and 'Holding_21EMA_Zone' in filtered_df.columns and 'Dist_21EMA' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Holding_21EMA_Zone'] & (filtered_df['Dist_21EMA'] >= -2.0) & (filtered_df['Dist_21EMA'] <= 3.0)]
    elif filter_choice == "🧲 21 EMA Test / Shakeout (-2% to 0%)" and 'Holding_21EMA_Zone' in filtered_df.columns and 'Dist_21EMA' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Holding_21EMA_Zone'] & (filtered_df['Dist_21EMA'] >= -2.0) & (filtered_df['Dist_21EMA'] <= 0.0)]
    elif filter_choice == "🛡️ 21 EMA Cushion (+3% to +5%)" and 'Holding_21EMA_Zone' in filtered_df.columns and 'Dist_21EMA' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Holding_21EMA_Zone'] & (filtered_df['Dist_21EMA'] > 3.0) & (filtered_df['Dist_21EMA'] <= 5.0)]
    elif filter_choice == "🚀 Power Extended (> +5%)" and 'Holding_21EMA_Zone' in filtered_df.columns and 'Dist_21EMA' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Holding_21EMA_Zone'] & (filtered_df['Dist_21EMA'] > 5.0)]

    # 2. Sortino Filter
    if 'Sortino_3M' in filtered_df.columns:
        if sortino_choice == "🔥 Positive (> 0.0)":
            filtered_df = filtered_df[filtered_df['Sortino_3M'] > 0.0]
        elif sortino_choice == "⚡ High Sortino (≥ 1.5)":
            filtered_df = filtered_df[filtered_df['Sortino_3M'] >= 1.5]
        elif sortino_choice == "💎 Elite Sortino (≥ 3.0)":
            filtered_df = filtered_df[filtered_df['Sortino_3M'] >= 3.0]

    # 3. Beach Ball & Volume (VDU opens to Bear 65E VDU only in Bear/All modes — C5)
    if bb_choice == "👑 Beach Ball Active Only (Accumulating on Washouts)" and 'Has_Beach_Ball' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Has_Beach_Ball']]
    elif "High 50D Up/Down Vol" in bb_choice and 'UDVR_50D' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['UDVR_50D'] >= 1.25]
    elif "Pocket Pivot Today" in bb_choice and 'Is_Pocket_Pivot' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Is_Pocket_Pivot']]
    elif "Volume Dry-Up Today" in bb_choice and 'Is_VDU' in filtered_df.columns:
        if (is_bear_mode or filter_choice == ALL_LEADERS_OPTION) and 'Is_Bear_VDU' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Is_VDU'] | filtered_df['Is_Bear_VDU']]
        else:
            filtered_df = filtered_df[filtered_df['Is_VDU']]
    elif "Apex Accumulation" in bb_choice and 'UDVR_50D' in filtered_df.columns and 'Has_Beach_Ball' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Has_Beach_Ball'] & (filtered_df['UDVR_50D'] >= 1.50)]

    # 4. Group Grade
    if selected_grades and "All" not in selected_grades and 'Industry_Grade' in filtered_df.columns:
        target_grades = set()
        include_top_20 = False
        include_top_40 = False
        for g in selected_grades:
            if "Top 40%" in g:
                target_grades.update(["A+", "A", "B"])
                include_top_40 = True
            elif "Top 20%" in g:
                target_grades.update(["A+", "A"])
                include_top_20 = True
            else:
                target_grades.add(g)
        grade_mask = filtered_df['Industry_Grade'].astype(str).isin(target_grades)
        if include_top_20 and 'Is_Top_20_Group' in filtered_df.columns:
            grade_mask = grade_mask | filtered_df['Is_Top_20_Group'].fillna(False).astype(bool)
        if include_top_40 and 'Group_Rank' in filtered_df.columns:
            grade_mask = grade_mask | (pd.to_numeric(filtered_df['Group_Rank'], errors='coerce').fillna(999) <= 40)
        filtered_df = filtered_df[grade_mask]

    # 5. Quantitative Cutoffs
    if min_rai > 0.0 and rai_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[rai_col] >= min_rai]
    if min_rs > 0 and 'RS_Rating' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['RS_Rating'] >= min_rs]
    if max_off_52w < 100.0 and 'Dist_52W_High' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Dist_52W_High'] >= (-1.0 * float(max_off_52w))]

    # 6. Livermore Role
    if role_choice != "All Group Roles" and 'Role_Code' in filtered_df.columns:
        if "Alpha #1 & 👯 Sister #2" in role_choice:
            filtered_df = filtered_df[filtered_df['Role_Code'].isin(['ALPHA', 'SISTER'])]
        elif "Top 3 Group Leaders" in role_choice:
            filtered_df = filtered_df[filtered_df['Role_Code'].isin(['ALPHA', 'SISTER', 'LEADER_3'])]
        elif "Confirmed Pack Leaders" in role_choice:
            filtered_df = filtered_df[filtered_df['Role_Code'].isin(['ALPHA', 'SISTER', 'LEADER_3', 'PACK_LEADER'])]

    # 7. RS Line Signal
    if rs_line_choice != "All RS Line States":
        if "Blue Dot Only" in rs_line_choice and 'Is_RS_Blue_Dot' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Is_RS_Blue_Dot']]
        elif "All 52W RS Line Highs" in rs_line_choice and 'Is_RS_52W_High' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Is_RS_52W_High']]
        elif "All 3M & 52W RS Line Highs" in rs_line_choice and 'Is_RS_3M_High' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Is_RS_3M_High'] | filtered_df['Is_RS_52W_High']]

    # 8. CANSLIM Confluence (includes D6 65-EMA Recovery option)
    if funda_choice != "All Fundamentals":
        if "True Market Leader" in funda_choice and 'Is_TML_Confluence' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Is_TML_Confluence']]
        elif "Double-25% CANSLIM" in funda_choice and 'Has_Double_25' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Has_Double_25']]
        elif "High EPS or Sales Growth" in funda_choice and 'Has_High_Sales' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Has_High_Sales']]
        elif "Quality ROE" in funda_choice and 'Has_Quality_ROE' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Has_Quality_ROE']]
        elif funda_choice == STAGE2_OPTION and 'Is_Stage2' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Is_Stage2']]
        elif funda_choice == STAGE2_OR_65_OPTION:
            s2_mask = filtered_df['Is_Stage2'] if 'Is_Stage2' in filtered_df.columns else False
            h65_mask = (
                (filtered_df['Holding_65EMA_Zone'] & filtered_df['SMA200_Slope_OK'])
                if ('Holding_65EMA_Zone' in filtered_df.columns and 'SMA200_Slope_OK' in filtered_df.columns)
                else False
            )
            qual_mask = (
                filtered_df.get('Is_TML_Confluence', False)
                | filtered_df.get('Has_Double_25', False)
                | filtered_df.get('Has_High_Sales', False)
                | filtered_df.get('Has_Inst_Accum', False)
            )
            filtered_df = filtered_df[(s2_mask | h65_mask) & qual_mask]

    # 9. Structure-Aware VCP Tightness & Bar Filter (D6 + C5)
    if vcp_choice != "All Price Action":
        bear_coil_col = filtered_df['Is_Bear_Coil'] if 'Is_Bear_Coil' in filtered_df.columns else False
        vcp_active = (filtered_df['Is_VCP_Tight'] | bear_coil_col) if (is_bear_mode or filter_choice == ALL_LEADERS_OPTION) else filtered_df.get('Is_VCP_Tight', False)

        if "All VCP Tight Coils" in vcp_choice and 'Is_VCP_Tight' in filtered_df.columns:
            filtered_df = filtered_df[vcp_active]
        elif "Inside Day / NR7 Only" in vcp_choice and 'Is_ID_NR7' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Is_ID_NR7']]
        elif vcp_choice == VCP_ACTIVE_PIVOT_OPTION:
            if filter_choice == HOLDING_50_OPTION and 'Dist_50SMA' in filtered_df.columns:
                piv_mask = filtered_df['Dist_50SMA'].notna() & (filtered_df['Dist_50SMA'] >= -4.0) & (filtered_df['Dist_50SMA'] <= 3.0)
                if 'Holding_50SMA_Zone' in filtered_df.columns:
                    piv_mask = piv_mask & filtered_df['Holding_50SMA_Zone']
            elif filter_choice in (HOLDING_65_OPTION, BELOW_21_RESCUED_OPTION, BEAR_POCKET_OPTION) and 'Bear_Pivot_Pocket' in filtered_df.columns:
                piv_mask = filtered_df['Bear_Pivot_Pocket']
            else:
                piv_mask = (filtered_df['Dist_21EMA'] >= -2.0) & (filtered_df['Dist_21EMA'] <= 5.0)
            filtered_df = filtered_df[vcp_active & piv_mask]
        elif vcp_choice == VCP_STRICT_21_PIVOT_OPTION and 'Is_VCP_Tight' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Is_VCP_Tight'] & (filtered_df['Dist_21EMA'] >= -2.0) & (filtered_df['Dist_21EMA'] <= 5.0)]
        elif "Strong Bar Close Only" in vcp_choice and 'DCR_Pct' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['DCR_Pct'] >= 70.0]
        elif "High RVOL Expansion Today" in vcp_choice and 'RVOL_Today' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['RVOL_Today'] >= 1.50]

    # 10. Min 20D ADR %
    if min_adr > 0.0 and 'ADR_20D' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['ADR_20D'] >= min_adr]

    # 11. Stage & Cap Tier
    if stage_choice != "All Stages & Caps":
        if "Fresh Stage 2 Only" in stage_choice and 'Is_Fresh_Stage2' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Is_Fresh_Stage2']]
        elif "Exclude Climax Extended" in stage_choice and 'Is_Climax_Ext' in filtered_df.columns:
            filtered_df = filtered_df[~filtered_df['Is_Climax_Ext']]
        elif "Institutional QoQ Accumulation" in stage_choice and 'Inst_Change_QoQ' in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df['Inst_Change_QoQ'] > 0.0) | (filtered_df['Sponsorship_Grade'].isin(['A+', 'A']))]
        elif "Agile Small & Mid-Caps Only" in stage_choice and 'Cap_Tier' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Cap_Tier'].isin(['Small Cap', 'Mid Cap', 'Micro Cap'])]
        elif "Large & Mid-Cap Institutional Leaders" in stage_choice and 'Cap_Tier' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Cap_Tier'].isin(['Large Cap', 'Mid Cap', 'Mega Cap'])]

    # 12. Search Query (Null-Safe regex=False na=False fix — D7)
    if search_q and search_q.strip():
        q = search_q.strip().upper()
        search_cols = [
            'Ticker', 'Industry', 'Industry_Badge', 'Group_Role',
            'RS_Line_Status', 'Funda_Badge', 'Apex_Status', 'Exchange',
            'VCP_Action_Badge', 'Cap_Stage_Badge', 'Setup_Trigger_State', 'MA_Status',
        ]
        mask_q = pd.Series(False, index=filtered_df.index)
        for col in search_cols:
            if col in filtered_df.columns:
                mask_q = mask_q | filtered_df[col].fillna('').astype(str).str.upper().str.contains(q, regex=False, na=False)
        filtered_df = filtered_df[mask_q]

    # 13. Sorting (Includes Bear 50-SMA & Haber 65-EMA Floor sorts — D6)
    if filtered_df.empty:
        return filtered_df

    if sort_by == "Smooth RS / ATR (High to Low)" and 'Smooth_RS' in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by='Smooth_RS', ascending=False)
    elif sort_by == "RS Rating (1-99) (High to Low)" and "RS_Rating" in filtered_df.columns:
        by_cols = [c for c in ('RS_Rating', 'Smooth_RS') if c in filtered_df.columns]
        filtered_df = filtered_df.sort_values(by=by_cols, ascending=[False] * len(by_cols))
    elif sort_by == "20D ADR % (Fastest Movers First)" and "ADR_20D" in filtered_df.columns:
        by_cols = [c for c in ('ADR_20D', 'Smooth_RS') if c in filtered_df.columns]
        filtered_df = filtered_df.sort_values(by=by_cols, ascending=[False] * len(by_cols))
    elif sort_by.startswith("20D ADTV") and "ADTV_20D" in filtered_df.columns:
        by_cols = [c for c in ('ADTV_20D', 'Smooth_RS') if c in filtered_df.columns]
        filtered_df = filtered_df.sort_values(by=by_cols, ascending=[False] * len(by_cols))
    elif sort_by == SORT_CLOSEST_21 and 'Dist_21EMA' in filtered_df.columns:
        tie_col = 'Smooth_RS' if 'Smooth_RS' in filtered_df.columns else 'Rank'
        by_cols = (['Holding_21EMA_Zone'] if 'Holding_21EMA_Zone' in filtered_df.columns else []) + ['_abs_d21', tie_col]
        asc = ([False] if 'Holding_21EMA_Zone' in filtered_df.columns else []) + [True, False if tie_col == 'Smooth_RS' else True]
        filtered_df = filtered_df.assign(_abs_d21=filtered_df['Dist_21EMA'].abs()).sort_values(
            by=by_cols, ascending=asc
        ).drop(columns=['_abs_d21'])
    elif sort_by == SORT_CLOSEST_50 and "Dist_50SMA" in filtered_df.columns:
        tie_col = 'RS_Rating' if 'RS_Rating' in filtered_df.columns else 'Rank'
        by_cols = (['Holding_50SMA_Zone'] if 'Holding_50SMA_Zone' in filtered_df.columns else []) + ['_abs_d50', tie_col]
        asc = ([False] if 'Holding_50SMA_Zone' in filtered_df.columns else []) + [True, False if tie_col == 'RS_Rating' else True]
        filtered_df = filtered_df.assign(_abs_d50=filtered_df['Dist_50SMA'].abs().fillna(999.0)).sort_values(
            by=by_cols, ascending=asc
        ).drop(columns=['_abs_d50'])
    elif sort_by == SORT_CLOSEST_65 and "Dist_65EMA" in filtered_df.columns:
        tie_col = 'RS_Rating' if 'RS_Rating' in filtered_df.columns else 'Rank'
        by_cols = (['Holding_65EMA_Zone'] if 'Holding_65EMA_Zone' in filtered_df.columns else []) + ['_abs_d65', tie_col]
        asc = ([False] if 'Holding_65EMA_Zone' in filtered_df.columns else []) + [True, False if tie_col == 'RS_Rating' else True]
        filtered_df = filtered_df.assign(_abs_d65=filtered_df['Dist_65EMA'].abs().fillna(999.0)).sort_values(
            by=by_cols, ascending=asc
        ).drop(columns=['_abs_d65'])
    elif sort_by == "vs 52W High % (Closest to High First)" and "Dist_52W_High" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=['Dist_52W_High', 'Smooth_RS'], ascending=[False, False])
    elif sort_by == "VCP Tightness (Tightest 5D Coil First)" and "Range_5D_Pct" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=['Range_5D_Pct', 'Smooth_RS'], ascending=[True, False])
    elif sort_by == "Today DCR % (Strongest Bar Close First)" and "DCR_Pct" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=['DCR_Pct', 'Smooth_RS'], ascending=[False, False])
    elif sort_by == "Today RVOL (Highest Volume Surge First)" and "RVOL_Today" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=['RVOL_Today', 'Smooth_RS'], ascending=[False, False])
    elif "RS Days %" in sort_by and rai_col in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=[rai_col, 'Smooth_RS'], ascending=[False, False])
    elif sort_by == "50D Up/Down Volume Ratio (High to Low)" and "UDVR_50D" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=['UDVR_50D', 'Smooth_RS'], ascending=[False, False])
    elif sort_by == "Inst Ownership Δ QoQ (High to Low)" and "Inst_Change_QoQ" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=['Inst_Change_QoQ', 'Smooth_RS'], ascending=[False, False])
    elif sort_by == "EPS Growth % (High to Low)" and "EPS_Growth" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=['EPS_Growth', 'Smooth_RS'], ascending=[False, False])
    elif sort_by == "Sales Growth % (High to Low)" and "Sales_Growth" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=['Sales_Growth', 'Smooth_RS'], ascending=[False, False])
    elif sort_by == "Sortino 3M Ratio (High to Low)":
        filtered_df = filtered_df.sort_values(by='Sortino_3M', ascending=False)
    elif sort_by == "vs 21 EMA % (High to Low)":
        filtered_df = filtered_df.sort_values(by='Dist_21EMA', ascending=False)
    elif sort_by == "vs 50 SMA % (High to Low)" and "Dist_50SMA" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by='Dist_50SMA', ascending=False)
    elif sort_by == "vs 65 EMA % (High to Low)" and "Dist_65EMA" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by='Dist_65EMA', ascending=False)
    elif sort_by == "Industry Group Rank (#1 to #N)" and "Group_Rank" in filtered_df.columns:
        rk_num = pd.to_numeric(filtered_df['Group_Rank'], errors='coerce').fillna(9999)
        tie_rs = 'Smooth_RS' if 'Smooth_RS' in filtered_df.columns else ('RS_Rating' if 'RS_Rating' in filtered_df.columns else 'Rank')
        filtered_df = filtered_df.assign(_tmp_grp_rk=rk_num).sort_values(
            by=['_tmp_grp_rk', tie_rs],
            ascending=[True, False if tie_rs != 'Rank' else True],
        ).drop(columns=['_tmp_grp_rk'])
    elif 'Rank' in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by='Rank', ascending=True)

    return filtered_df


LENS_EXECUTION = "⚡ Execution"
LENS_STRUCTURE = "📐 Structure & Bear Floor"
LENS_CANSLIM = "💎 CANSLIM & Flow"
LENS_FULL = "🌐 Full 25-Col Matrix"
COLUMN_LENS_OPTIONS = [
    LENS_EXECUTION,
    LENS_STRUCTURE,
    LENS_CANSLIM,
    LENS_FULL,
]


def project_columns_for_lens(
    disp_cols: dict[str, str],
    lens_choice: str,
    h_days: int,
    price_col_label: str,
    adtv_col_label: str,
) -> dict[str, str]:
    """Return an ordered subset of `disp_cols` matching the active Column View Lens."""
    if lens_choice == LENS_FULL:
        return dict(disp_cols)

    lens_labels = {
        LENS_EXECUTION: [
            "RS Rank", "Ticker", "Chart", price_col_label, "RS Rating",
            "Smooth RS / ATR", "vs 21 EMA %", "Trigger State", "VCP & Bar Action",
            "Industry & Grade", "Group Role", adtv_col_label,
        ],
        LENS_STRUCTURE: [
            "RS Rank", "Ticker", "Chart", price_col_label, "vs 21 EMA %",
            "vs 50 SMA %", "vs 65 EMA %", "50D ATRx", "Structure Status",
            "vs 52W High %", "RS Line Signal", "VCP & Bar Action",
        ],
        LENS_CANSLIM: [
            "RS Rank", "Ticker", "Chart", "RS Rating", "CANSLIM Confluence",
            "Stage & Cap Tier", "Industry & Grade", "Group Role",
            f"Rai {h_days}D RS %", "50D U/D Vol", "Beach Ball & Vol", "Sortino 3M",
        ],
    }
    wanted = lens_labels.get(lens_choice)
    if not wanted:
        return dict(disp_cols)

    label_to_key = {lbl: k for k, lbl in disp_cols.items()}
    projected = {}
    for lbl in wanted:
        src_key = label_to_key.get(lbl)
        if src_key:
            projected[src_key] = lbl
    return projected if projected else dict(disp_cols)


def project_split_pane_columns(
    disp_cols: dict[str, str],
    *,
    price_col_label: str,
) -> dict[str, str]:
    """Project `disp_cols` down to a crisp 5-column left-rail projection for Split View."""
    wanted = ["RS Rank", "Ticker", price_col_label, "RS Rating", "vs 21 EMA %"]
    label_to_key = {lbl: k for k, lbl in disp_cols.items()}
    projected = {}
    for lbl in wanted:
        src_key = label_to_key.get(lbl)
        if src_key:
            projected[src_key] = lbl
    return projected if projected else dict(list(disp_cols.items())[:5])


def resolve_split_selection(
    tickers: list[str],
    *,
    clicked_rows: list[int] | None = None,
    last_clicked_rows: list[int] | None = None,
    stored_ticker: str | None = None,
    step_delta: int = 0,
) -> tuple[int, str | None, list[int]]:
    """Resolve active split-pane ticker index consistently across row clicks and Prev/Next buttons."""
    if not tickers:
        return 0, None, []

    prev_clicked = list(last_clicked_rows or [])
    cur_clicked = list(clicked_rows or [])
    click_changed = bool(cur_clicked) and cur_clicked != prev_clicked and 0 <= cur_clicked[0] < len(tickers)

    if click_changed:
        idx = cur_clicked[0]
    elif stored_ticker and stored_ticker in tickers:
        idx = tickers.index(stored_ticker)
    elif cur_clicked and 0 <= cur_clicked[0] < len(tickers):
        idx = cur_clicked[0]
    else:
        idx = 0

    if step_delta != 0:
        idx = max(0, min(len(tickers) - 1, idx + step_delta))

    return idx, tickers[idx], cur_clicked


def count_preset_matches(df_lead: pd.DataFrame) -> dict[str, int]:
    """Compute live match counts on `df_lead` for each regime preset without side effects."""
    if df_lead is None or df_lead.empty:
        return {pid: 0 for pid in PRESET_SPECS}

    base = _baseline_filter_defaults(is_master=True)
    counts: dict[str, int] = {}
    for pid, spec in PRESET_SPECS.items():
        merged = dict(base)
        merged.update(spec.get("overrides", {}))
        res = apply_leader_radar_filters(
            df_lead,
            filter_choice=merged["filter_choice"],
            sortino_choice=merged["sortino"],
            bb_choice=merged["bb"],
            selected_grades=merged["grade_multi"],
            min_rai=float(merged["rai_slider_20"]),
            rai_col="RS_Days_20D_Pct",
            min_rs=int(merged["min_rs"]),
            max_off_52w=float(merged["max_52w_slider"]),
            role_choice=merged["role_choice"],
            rs_line_choice=merged["rsline_choice"],
            funda_choice=merged["funda_choice"],
            vcp_choice=merged["vcp_choice"],
            min_adr=float(merged["min_adr"]),
            stage_choice=merged["stage_choice"],
            sort_by=merged["sort"],
            search_q="",
        )
        counts[pid] = int(len(res))
    return counts


def summarize_active_filters(
    live_values: dict,
    *,
    h_days: int,
    is_master: bool,
    min_adtv: float = 0.0,
    curr_sym: str = "₹",
    adtv_unit: str = "₹ Cr",
) -> dict:
    """Summarize non-default active filter chips and per-category counts for the Advanced drawer."""
    base = _baseline_filter_defaults(is_master=is_master)
    chips: list[str] = []

    fc = live_values.get("filter_choice", base["filter_choice"])
    if fc != base["filter_choice"]:
        short_fc = fc.split(" (")[0]
        chips.append(f"Structure: {short_fc}")

    vc = live_values.get("vcp_choice", base["vcp_choice"])
    if vc != base["vcp_choice"]:
        chips.append(f"Action: {vc.split(' (')[0]}")

    sq = str(live_values.get("search", "") or "").strip()
    if sq:
        chips.append(f"Search: \"{sq}\"")

    # Category 1: Momentum & Liquidity (inside Advanced drawer)
    mom_cnt = 0
    if int(live_values.get("min_rs", 0)) > 0:
        mom_cnt += 1
        chips.append(f"RS ≥ {int(live_values['min_rs'])}")
    if float(live_values.get("max_52w_slider", 100.0)) < 100.0:
        mom_cnt += 1
        chips.append(f"Off 52W ≤ -{float(live_values['max_52w_slider']):.0f}%")
    if float(live_values.get("min_rai", 0.0)) > 0.0:
        mom_cnt += 1
        chips.append(f"Rai {h_days}D ≥ {float(live_values['min_rai']):.0f}%")
    if float(live_values.get("min_adr", 0.0)) > 0.0:
        mom_cnt += 1
        chips.append(f"ADR ≥ {float(live_values['min_adr']):.1f}%")
    if is_master and float(min_adtv) > 0.0:
        mom_cnt += 1
        unit_clean = adtv_unit.replace("₹ ", "")
        chips.append(f"ADTV ≥ {curr_sym}{float(min_adtv):.1f} {unit_clean}")

    # Category 2: Group & Sponsorship (inside Advanced drawer)
    grp_cnt = 0
    grades = live_values.get("grade_multi") or []
    if grades:
        grp_cnt += 1
        chips.append(f"Grade: {', '.join(grades)}")
    rc = live_values.get("role_choice", base["role_choice"])
    if rc != base["role_choice"]:
        grp_cnt += 1
        chips.append(f"Role: {rc.split(' (')[0]}")
    rsl = live_values.get("rsline_choice", base["rsline_choice"])
    if rsl != base["rsline_choice"]:
        grp_cnt += 1
        chips.append(f"RS Line: {rsl.split(' (')[0]}")
    bb = live_values.get("bb", base["bb"])
    if bb != base["bb"]:
        grp_cnt += 1
        chips.append(f"Flow: {bb.split(' (')[0]}")

    # Category 3: CANSLIM & Stage (inside Advanced drawer)
    can_cnt = 0
    fc_funda = live_values.get("funda_choice", base["funda_choice"])
    if fc_funda != base["funda_choice"]:
        can_cnt += 1
        chips.append(f"CANSLIM: {fc_funda.split(' (')[0]}")
    sc = live_values.get("stage_choice", base["stage_choice"])
    if sc != base["stage_choice"]:
        can_cnt += 1
        chips.append(f"Stage: {sc.split(' (')[0]}")
    so = live_values.get("sortino", base["sortino"])
    if so != base["sortino"]:
        can_cnt += 1
        chips.append(f"Sortino: {so}")

    return {
        "chips": chips,
        "cat_counts": {"momentum": mom_cnt, "group": grp_cnt, "canslim": can_cnt},
        "adv_active_count": mom_cnt + grp_cnt + can_cnt,
    }


def _render_kpi_summary_strip(
    *,
    is_master: bool,
    n_tot: int,
    n_defending: int,
    pct_defending: float,
    pct50: float,
    pct65: float,
    n_vcp: int,
    n_id_nr7: int,
    n_bear_coil: int,
    n_rs_high: int,
    n_blue_dot: int,
    n_sortino: int,
    n_top_groups: int,
    n_tml: int,
    n_bb: int,
    n_rai_core: int,
    h_days: int,
    rai_core_thresh: float,
) -> None:
    """Render a compact 6-card responsive KPI summary strip without truncation."""
    vcp_sub = f"{n_id_nr7} ID/NR7" + (f" · +{n_bear_coil} 65E" if n_bear_coil > 0 else "")
    acc_sub = f"Rai {h_days}D ≥{rai_core_thresh:g}%: {n_rai_core}" if is_master else f"High Sortino: {n_sortino}"
    cards = [
        ("COHORT LEADERS" if is_master else "TOTAL LEADERS", f"{n_tot}", f"50S≥-4%: {pct50:.0f}%", "#f8fafc"),
        ("21-EMA SUPPORT", f"{n_defending} ({pct_defending:.0f}%)", f"65E Floor: {pct65:.0f}%", "#34d399"),
        ("VCP TIGHT COILS", f"{n_vcp}", vcp_sub, "#facc15"),
        ("RS 52W / BLUE DOT", f"{n_rs_high} ({n_blue_dot} 🔵)", "RS Line at New High", "#38bdf8"),
        ("TML & TOP GROUPS", f"{n_tml} 💎", f"Top 20% Grp: {n_top_groups}", "#c084fc"),
        ("BEACH BALL ACCUM", f"{n_bb} 🏖️", acc_sub, "#f59e0b"),
    ]
    cards_html = "".join([
        f'<div style="background:rgba(15,23,42,0.72); border:1px solid rgba(148,163,184,0.16); border-radius:8px; padding:8px 11px;">'
        f'<div style="color:#94a3b8; font-size:0.65rem; font-weight:800; letter-spacing:0.6px; text-transform:uppercase;">{lbl}</div>'
        f'<div style="color:{col}; font-size:1.08rem; font-weight:900; font-family:\'JetBrains Mono\',monospace; margin-top:2px;">{val}</div>'
        f'<div style="color:#cbd5e1; font-size:0.69rem; margin-top:2px; font-family:\'JetBrains Mono\',monospace;">{sub}</div>'
        f'</div>'
        for lbl, val, sub, col in cards
    ])
    st.markdown(
        f'<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(145px, 1fr)); gap:8px; margin-bottom:10px;">{cards_html}</div>',
        unsafe_allow_html=True,
    )


def _load_cached_ticker_ohlcv(
    raw_ticker: str,
    clean_ticker: str,
    univ_scope: str,
    *,
    is_us: bool = False,
    max_bars: int = 180,
    prefer_deepest: bool = False,
) -> pd.DataFrame | None:
    """Load local cached daily OHLCV for a single ticker without hitting network."""
    try:
        from price_history_manager import get_matrix_cache_signature, load_cached_price_matrix
    except Exception:
        return None

    scope_clean = str(univ_scope or "").upper().strip()
    if is_us:
        candidates = ["us_prices_matrix.pkl", "breadth_matrix.pkl", "historical_prices_matrix.pkl"]
        norm_dash = clean_ticker.replace(".", "-") if clean_ticker else ""
        sym_candidates = tuple(dict.fromkeys(x for x in (raw_ticker, clean_ticker, norm_dash) if x))
    else:
        candidates = (
            ["historical_prices_matrix.pkl", "breadth_matrix.pkl", "microcap_prices_matrix.pkl"]
            if scope_clean == "ALL"
            else ["breadth_matrix.pkl", "historical_prices_matrix.pkl", "microcap_prices_matrix.pkl"]
        )
        if prefer_deepest and "breadth_matrix.pkl" in candidates:
            candidates = ["breadth_matrix.pkl"] + [c for c in candidates if c != "breadth_matrix.pkl"]
        sym_candidates = tuple(
            dict.fromkeys(
                x
                for x in (
                    raw_ticker,
                    clean_ticker,
                    f"{clean_ticker}.NS" if clean_ticker and not clean_ticker.endswith((".NS", ".BO")) else "",
                    f"{clean_ticker}.BO" if clean_ticker and not clean_ticker.endswith((".NS", ".BO")) else "",
                )
                if x
            )
        )

    matrix_paths = [p for p in candidates if os.path.exists(p)]
    if not matrix_paths:
        return None

    best_sub: pd.DataFrame | None = None
    for matrix_path in matrix_paths:
        try:
            mat = load_cached_price_matrix(matrix_path, cache_sig=get_matrix_cache_signature(matrix_path))
            if mat is None or mat.empty or not isinstance(mat.columns, pd.MultiIndex):
                continue
            lvl0 = set(mat.columns.get_level_values(0))
            lookup = None
            for cand in sym_candidates:
                if cand and cand in lvl0:
                    lookup = cand
                    break
            if lookup is None:
                continue
            sub = mat[lookup].dropna(how="all").copy()
            if sub.empty or "Close" not in sub.columns:
                continue
            if getattr(sub.index, "tz", None) is not None:
                sub.index = sub.index.tz_localize(None)
            for col in ("Open", "High", "Low", "Close", "Volume"):
                if col in sub.columns:
                    sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub = sub.dropna(subset=["Close"])
            if sub.empty:
                continue
            if not prefer_deepest:
                return sub.tail(int(max_bars))
            if best_sub is None or len(sub) > len(best_sub):
                best_sub = sub
            if len(best_sub) >= int(max_bars):
                break
        except Exception:
            continue
    return best_sub.tail(int(max_bars)) if best_sub is not None else None


def _load_local_vol_proxy_series(is_us: bool = False, max_bars: int = 520) -> pd.Series | None:
    """Load local volume proxy series (NIFTYBEES.NS for India) when index volume is 0/NaN."""
    if is_us:
        return None
    sub = _load_cached_ticker_ohlcv("NIFTYBEES.NS", "NIFTYBEES", "ALL", is_us=False, max_bars=max_bars)
    if sub is not None and "Volume" in sub.columns:
        vs = pd.to_numeric(sub["Volume"], errors="coerce").dropna()
        if (vs > 0).any():
            return vs
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def _get_cached_kt_v3_benchmark(market: str = "IN", benchmark_symbol: str | None = None) -> tuple[pd.DataFrame | None, str]:
    """Fetch cached 520D full-OHLCV benchmark dataframe (Open/High/Low/Close/Volume) for KT v3.0."""
    mkt = str(market or "IN").upper().strip()
    is_us_mkt = mkt in ("US", "USA", "US_IXIC")
    if benchmark_symbol:
        sym_list = [str(benchmark_symbol).strip()]
        if sym_list[0] == "^IXIC":
            sym_list.append("QQQ")
        elif sym_list[0] == "^GSPC":
            sym_list.append("SPY")
    else:
        sym_list = ["^IXIC", "QQQ", "^GSPC", "SPY"] if is_us_mkt else ["^CRSLDX", "^NSEI", "NIFTYBEES.NS"]

    label_map = {
        "^CRSLDX": "Nifty 500",
        "^NSEI": "Nifty 50",
        "NIFTYBEES.NS": "Nifty ETF",
        "^IXIC": "Nasdaq Comp",
        "^GSPC": "S&P 500",
        "QQQ": "QQQ ETF",
        "SPY": "SPY ETF",
    }

    last_us_dt = None
    if is_us_mkt:
        try:
            from price_history_manager import get_latest_completed_session_date
            last_us_dt = pd.Timestamp(get_latest_completed_session_date("US"))
        except Exception:
            last_us_dt = None

    try:
        import yfinance as yf
    except Exception:
        yf = None

    if yf is not None:
        for sym in sym_list:
            try:
                raw = yf.download(sym, period="2y", progress=False)
                if raw is None or raw.empty:
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    lvl1 = raw.columns.get_level_values(1)
                    raw = raw.xs(sym, level=1, axis=1) if sym in lvl1 else raw.xs(raw.columns.get_level_values(0)[0], level=0, axis=1)
                col_map = {c: str(c).strip().capitalize() for c in raw.columns if str(c).strip().lower() in ("open", "high", "low", "close", "volume")}
                df = raw.rename(columns=col_map)
                if "Close" not in df.columns:
                    continue
                df.index = pd.to_datetime(df.index)
                if getattr(df.index, "tz", None) is not None:
                    df.index = df.index.tz_localize(None)
                df.index = df.index.normalize()
                if last_us_dt is not None:
                    settled = df.loc[df.index <= last_us_dt]
                    if len(settled) >= 30:
                        df = settled
                df = df.dropna(subset=["Close"]).tail(520)
                if len(df) >= 30:
                    return df, label_map.get(sym, sym)
            except Exception:
                continue

    if not is_us_mkt:
        nb = _load_cached_ticker_ohlcv("NIFTYBEES.NS", "NIFTYBEES", "ALL", is_us=False, max_bars=520)
        if nb is not None and len(nb) >= 30:
            return nb, "Nifty ETF (Local)"
    return None, (benchmark_symbol or ("Nasdaq/S&P" if is_us_mkt else "Nifty 500"))


def _extract_portfolio_cost_basis(portfolio_df: pd.DataFrame | None, clean_tk: str) -> tuple[bool, float | None]:
    """Safely check if symbol is held in Portfolio DataFrame and extract cost basis if present."""
    if portfolio_df is None or portfolio_df.empty or not clean_tk:
        return False, None
    try:
        t_col = (
            "ticker" if "ticker" in portfolio_df.columns
            else ("NSE TICKER" if "NSE TICKER" in portfolio_df.columns else portfolio_df.columns[0])
        )
        norm_target = str(clean_tk).replace(".NS", "").replace(".BO", "").strip().upper()
        norm_col = (
            portfolio_df[t_col]
            .astype(str)
            .str.replace(".NS", "", regex=False)
            .str.replace(".BO", "", regex=False)
            .str.strip()
            .str.upper()
        )
        matched = portfolio_df.loc[norm_col == norm_target]
        if matched.empty:
            return False, None
        for cost_col in ("Avg Buy Price", "avg_buy_price", "buy_price", "avg_price", "Avg_Buy_Price", "cost_basis"):
            if cost_col in matched.columns:
                val = pd.to_numeric(matched[cost_col].iloc[0], errors="coerce")
                if pd.notna(val) and float(val) > 0:
                    return True, float(val)
        return True, None
    except Exception:
        return False, None


def _get_pre_earnings_days(clean_tk: str) -> int | None:
    """Safely look up upcoming earnings proximity (days) from session_state earnings_cache."""
    try:
        ec = st.session_state.get("earnings_cache") or {}
        if not ec or not clean_tk:
            return None
        key_up = str(clean_tk).replace(".NS", "").replace(".BO", "").strip().upper()
        entry = ec.get(clean_tk) or ec.get(key_up) or ec.get(f"{key_up}.NS")
        if not isinstance(entry, dict):
            return None
        for k in ("days_to_earnings", "days_until", "days_away"):
            if k in entry and entry[k] is not None:
                return int(entry[k])
        dt_str = entry.get("next_earnings_date") or entry.get("earnings_date")
        if dt_str:
            delta = (pd.Timestamp(dt_str).normalize() - pd.Timestamp.now().normalize()).days
            if 0 <= delta <= 30:
                return int(delta)
    except Exception:
        pass
    return None


def build_canslim_ohlcv_chart(
    ohlcv: pd.DataFrame,
    *,
    ticker: str,
    entry_trigger: float | None = None,
    stop_loss: float | None = None,
    show_r_zones: bool = False,
    height: int = 330,
    bars: int = 90,
    kt_telemetry: dict | None = None,
    overlay_mode: str = "clean",
    show_rs_line: bool = False,
    signal_scope: str = "recent45",
) -> go.Figure:
    """Build a dark CANSLIM Candlestick + MAs + optional KushTracker v3.0 overlays + Volume subplot."""
    from plotly.subplots import make_subplots

    mode_clean = str(overlay_mode or "clean").lower().strip()
    use_kt = (kt_telemetry is not None) and (mode_clean != "clean") and not (kt_telemetry.get("series_df") is None or kt_telemetry["series_df"].empty)
    src_df = kt_telemetry["series_df"] if use_kt else ohlcv
    plot_df = src_df.tail(bars).copy() if src_df is not None and not src_df.empty else pd.DataFrame()

    has_ohlc = (
        not plot_df.empty
        and {"Open", "High", "Low", "Close"}.issubset(plot_df.columns)
        and plot_df[["Open", "High", "Low", "Close"]].notna().all(axis=1).sum() >= 5
    )
    has_vol = (
        not plot_df.empty
        and "Volume" in plot_df.columns
        and pd.to_numeric(plot_df["Volume"], errors="coerce").fillna(0).gt(0).any()
    )
    rs_active = use_kt and bool(show_rs_line) and ("rs_norm" in plot_df.columns) and plot_df["rs_norm"].notna().any()

    if has_vol:
        specs = [[{"secondary_y": rs_active}], [{"secondary_y": False}]]
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.74, 0.26],
            specs=specs,
        )
    elif rs_active:
        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    add_kw = {"row": 1, "col": 1} if (has_vol or rs_active) else {}
    add_sec_kw = ({"row": 1, "col": 1, "secondary_y": False} if (rs_active and has_vol) else ({"secondary_y": False} if rs_active else add_kw))

    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df["Open"],
                high=plot_df["High"],
                low=plot_df["Low"],
                close=plot_df["Close"],
                name=ticker or "Price",
                increasing_line_color="#10b981",
                decreasing_line_color="#f87171",
                increasing_fillcolor="rgba(16, 185, 129, 0.72)",
                decreasing_fillcolor="rgba(248, 113, 113, 0.72)",
            ),
            **add_sec_kw,
        )
    elif not plot_df.empty and "Close" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["Close"],
                mode="lines",
                name=ticker or "Close",
                line=dict(color="#f8fafc", width=2.1),
            ),
            **add_sec_kw,
        )

    # Optional 21E <-> 50S Bull Trend Ribbon in KT v3.0 mode (plotted with candles)
    if use_kt and mode_clean == "kt_v3" and {"EMA21", "SMA50"}.issubset(plot_df.columns) and plot_df["SMA50"].notna().any():
        e21_s = plot_df["EMA21"]
        s50_s = plot_df["SMA50"]
        is_bull_rib = bool(pd.notna(e21_s.iloc[-1]) and pd.notna(s50_s.iloc[-1]) and float(e21_s.iloc[-1]) >= float(s50_s.iloc[-1]))
        rib_fill = "rgba(16, 185, 129, 0.11)" if is_bull_rib else "rgba(245, 158, 11, 0.08)"
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=s50_s, mode="lines", name="50 SMA", line=dict(color="#f59e0b", width=1.45)),
            **add_sec_kw,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=e21_s,
                mode="lines",
                name="21 EMA",
                line=dict(color="#38bdf8", width=1.45),
                fill="tonexty",
                fillcolor=rib_fill,
            ),
            **add_sec_kw,
        )
        for col, lbl, col_hex, dash, w in (
            ("EMA10", "10 EMA", "#f44336", "dot", 1.15),
            ("EMA65", "65 EMA", "#a855f7", "dash", 1.15),
        ):
            if col in plot_df.columns and plot_df[col].notna().any():
                fig.add_trace(
                    go.Scatter(x=plot_df.index, y=plot_df[col], mode="lines", name=lbl, line=dict(color=col_hex, width=w, dash=dash)),
                    **add_sec_kw,
                )
        # Include 200 SMA only if within ±35% of visible price band so runaway leaders don't crush candles
        if "SMA200" in plot_df.columns and plot_df["SMA200"].notna().any():
            vis_low = float(plot_df["Low"].min() if has_ohlc else plot_df["Close"].min())
            vis_high = float(plot_df["High"].max() if has_ohlc else plot_df["Close"].max())
            s200_tail = plot_df["SMA200"].dropna()
            if not s200_tail.empty and vis_low * 0.65 <= float(s200_tail.iloc[-1]) <= vis_high * 1.35:
                fig.add_trace(
                    go.Scatter(x=plot_df.index, y=plot_df["SMA200"], mode="lines", name="200 SMA", line=dict(color="#64748b", width=1.25, dash="dashdot")),
                    **add_sec_kw,
                )
    else:
        for col, lbl, col_hex, dash in (
            ("EMA21", "21 EMA", "#38bdf8", "dot"),
            ("SMA50", "50 SMA", "#f59e0b", "solid"),
            ("EMA65", "65 EMA", "#a855f7", "dash"),
        ):
            if not plot_df.empty and col in plot_df.columns and plot_df[col].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=plot_df.index,
                        y=plot_df[col],
                        mode="lines",
                        name=lbl,
                        line=dict(color=col_hex, width=1.4, dash=dash),
                    ),
                    **add_sec_kw,
                )

    # Resolve KT v3.0 dynamic pivot / stop / R-zones if active
    hdr = kt_telemetry.get("header_summary", {}) if use_kt else {}
    eff_entry = hdr.get("active_pivot") if (use_kt and hdr.get("is_in_position") and hdr.get("active_pivot")) else entry_trigger
    eff_stop = hdr.get("dynamic_stop") if (use_kt and hdr.get("is_in_position") and hdr.get("dynamic_stop")) else stop_loss

    if use_kt:
        # Plot segmented 3-Phase Dynamic Trailing Stop step-line
        if "dynamic_stop" in plot_df.columns and plot_df["dynamic_stop"].notna().any():
            for ph_code, ph_name, ph_col in (
                (1, "P1 Hard Stop", "#ef5350"),
                (2, "P2 Breakeven", "#26a69a"),
                (3, "P3 MA Trail", "#facc15"),
                (4, "8-Wk Hold Stop", "#14b8a6"),
            ):
                mask = (plot_df["stop_phase"] == ph_code) & plot_df["dynamic_stop"].notna()
                if mask.any():
                    y_seg = plot_df["dynamic_stop"].where(mask, float("nan"))
                    fig.add_trace(
                        go.Scatter(
                            x=plot_df.index,
                            y=y_seg,
                            mode="lines",
                            name=ph_name,
                            line=dict(color=ph_col, width=2.0, shape="hv"),
                            connectgaps=False,
                        ),
                        **add_sec_kw,
                    )

        # Active Pivot + Buy Zone (0..5%) or Prospective Pivot when flat
        pane_kw = {"row": 1, "col": 1} if (has_vol or rs_active) else {}
        if hdr.get("is_in_position") and hdr.get("active_pivot"):
            ap = float(hdr["active_pivot"])
            bz = float(hdr.get("buy_zone_upper") or (ap * 1.05))
            t20 = hdr.get("target_20")
            fig.add_hline(
                y=ap,
                line_dash="dash",
                line_color="#00c853",
                line_width=1.4,
                annotation_text=f"Pivot {ap:,.2f}",
                annotation_position="top left",
                annotation_font_color="#34d399",
                **pane_kw,
            )
            if mode_clean == "kt_v3":
                fig.add_hrect(
                    y0=ap,
                    y1=bz,
                    fillcolor="rgba(0, 200, 83, 0.10)",
                    line_width=0,
                    layer="below",
                    **pane_kw,
                )
                if t20 and float(t20) > ap:
                    fig.add_hline(
                        y=float(t20),
                        line_dash="dot",
                        line_color="rgba(250, 204, 21, 0.65)",
                        line_width=1.1,
                        annotation_text=f"+20% Tgt {float(t20):,.2f}",
                        annotation_position="top right",
                        annotation_font_color="#facc15",
                        **pane_kw,
                    )
        elif hdr.get("prospective_pivot"):
            pp = float(hdr["prospective_pivot"])
            fig.add_hline(
                y=pp,
                line_dash="dot",
                line_color="#2dd4bf",
                line_width=1.25,
                annotation_text=f"Watch Pivot {pp:,.2f}",
                annotation_position="top left",
                annotation_font_color="#2dd4bf",
                **pane_kw,
            )

        if hdr.get("is_stopped_out"):
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.97,
                text="⛔ STOPPED OUT — Cooldown / Stalking New Constructive Base",
                showarrow=False,
                font=dict(size=10, color="#fca5a5"),
                bgcolor="rgba(127, 29, 29, 0.65)",
                bordercolor="rgba(248, 113, 113, 0.55)",
                borderpad=4,
            )

        # Van Tharp Fixed-1R Ladders when Risk & R-Zones mode is selected
        if mode_clean == "risk_zones" and hdr.get("is_in_position") and hdr.get("active_pivot") and float(hdr.get("init_risk_pts", 0) or 0) > 0:
            _et_v = float(hdr["active_pivot"])
            _r_pts = float(hdr["init_risk_pts"])
            _hard_sl = _et_v - _r_pts
            fig.add_hrect(y0=_hard_sl, y1=_et_v, fillcolor="rgba(239, 68, 68, 0.11)", line_width=0, layer="below", **pane_kw)
            fig.add_hrect(y0=_et_v, y1=_et_v + 3.0 * _r_pts, fillcolor="rgba(16, 185, 129, 0.08)", line_width=0, layer="below", **pane_kw)
            for _rl, _mult, _rc in (("-1R Stop", -1.0, "#f87171"), ("+1R", 1.0, "#6ee7b7"), ("+2R", 2.0, "#34d399"), ("+3R", 3.0, "#10b981")):
                _rv = _et_v + _mult * _r_pts
                fig.add_hline(
                    y=_rv,
                    line_dash="dot",
                    line_color=_rc,
                    line_width=1.1,
                    annotation_text=f"{_rl} {_rv:,.2f}",
                    annotation_position="top right",
                    annotation_font_color=_rc,
                    **pane_kw,
                )

        # Priority-gated Signal Markers (KT v3.0 mode)
        if mode_clean == "kt_v3" and {"below_label", "above_label"}.issubset(plot_df.columns):
            sig_df = plot_df.tail(45) if str(signal_scope) == "recent45" else plot_df
            use_text = len(plot_df) <= 110
            for col_lbl, col_hex, col_rsn, is_above in (
                ("below_label", "below_color", "below_reason", False),
                ("above_label", "above_color", "above_reason", True),
            ):
                sub_sig = sig_df[sig_df[col_lbl].astype(str).str.len() > 0]
                if sub_sig.empty:
                    continue
                y_anchor = (sub_sig["High"] * 1.018) if is_above else (sub_sig["Low"] * 0.982)
                hover_txt = [
                    f"<b>{lbl}</b><br>{pd.Timestamp(dt).strftime('%d %b %Y')} • Close: {cl:,.2f}<br>{rsn}"
                    for dt, lbl, cl, rsn in zip(sub_sig.index, sub_sig[col_lbl], sub_sig["Close"], sub_sig[col_rsn])
                ]
                fig.add_trace(
                    go.Scatter(
                        x=sub_sig.index,
                        y=y_anchor,
                        mode="markers+text" if use_text else "markers",
                        text=sub_sig[col_lbl].tolist() if use_text else None,
                        textposition="top center" if is_above else "bottom center",
                        textfont=dict(size=9, color="#f8fafc"),
                        marker=dict(
                            symbol="triangle-down" if is_above else "triangle-up",
                            size=8,
                            color=sub_sig[col_hex].tolist(),
                            line=dict(color="#020617", width=1),
                        ),
                        hovertext=hover_txt,
                        hoverinfo="text",
                        name="Sell / Warn Signals" if is_above else "Buy / Setup Signals",
                        showlegend=False,
                    ),
                    **add_sec_kw,
                )

        # Normalized MarketSurge Blue RS Line overlay (optional)
        if rs_active:
            sec_kw = {"row": 1, "col": 1, "secondary_y": True} if has_vol else {"secondary_y": True}
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df["rs_norm"],
                    mode="lines",
                    name="RS Line",
                    line=dict(color="rgba(56, 189, 248, 0.72)", width=1.45, dash="solid"),
                ),
                **sec_kw,
            )
    else:
        if entry_trigger is not None and float(entry_trigger) > 0:
            fig.add_hline(
                y=float(entry_trigger),
                line_dash="dash",
                line_color="#10b981",
                line_width=1.4,
                annotation_text=f"Entry {float(entry_trigger):.2f}",
                annotation_position="top left",
                annotation_font_color="#34d399",
                **({"row": 1, "col": 1} if has_vol else {}),
            )
        if stop_loss is not None and float(stop_loss) > 0:
            fig.add_hline(
                y=float(stop_loss),
                line_dash="dash",
                line_color="#ef4444",
                line_width=1.4,
                annotation_text=f"Stop {float(stop_loss):.2f}",
                annotation_position="bottom left",
                annotation_font_color="#f87171",
                **({"row": 1, "col": 1} if has_vol else {}),
            )

        if (
            show_r_zones
            and entry_trigger is not None
            and stop_loss is not None
            and float(entry_trigger) > float(stop_loss) > 0
        ):
            _et_v = float(entry_trigger)
            _sl_v = float(stop_loss)
            _r_pts = _et_v - _sl_v
            _r2_v = _et_v + 2.0 * _r_pts
            _r3_v = _et_v + 3.0 * _r_pts
            _pane_kw = {"row": 1, "col": 1} if has_vol else {}
            fig.add_hrect(y0=_sl_v, y1=_et_v, fillcolor="rgba(239, 68, 68, 0.10)", line_width=0, layer="below", **_pane_kw)
            fig.add_hrect(y0=_et_v, y1=_r3_v, fillcolor="rgba(16, 185, 129, 0.08)", line_width=0, layer="below", **_pane_kw)
            for _r_lbl, _r_val, _r_col in (("+2R", _r2_v, "#34d399"), ("+3R", _r3_v, "#10b981")):
                fig.add_hline(
                    y=_r_val,
                    line_dash="dot",
                    line_color=_r_col,
                    line_width=1.1,
                    annotation_text=f"{_r_lbl} {_r_val:.2f}",
                    annotation_position="top right",
                    annotation_font_color=_r_col,
                    **_pane_kw,
                )

    if has_vol:
        vols = pd.to_numeric(plot_df["Volume"], errors="coerce").fillna(0.0)
        if use_kt and "vol_role" in plot_df.columns:
            role_colors = {
                "accum": "rgba(16, 185, 129, 0.92)",
                "vdu": "rgba(56, 189, 248, 0.80)",
                "dist": "rgba(239, 68, 68, 0.85)",
                "neutral": "rgba(100, 116, 139, 0.45)",
            }
            vol_colors = [role_colors.get(str(r), "rgba(100, 116, 139, 0.45)") for r in plot_df["vol_role"]]
        else:
            if "Open" in plot_df.columns and plot_df["Open"].notna().all():
                is_up = plot_df["Close"] >= plot_df["Open"]
            else:
                is_up = plot_df["Close"].diff().fillna(0.0) >= 0.0
            vol_colors = ["rgba(16, 185, 129, 0.72)" if up else "rgba(248, 113, 113, 0.65)" for up in is_up]
        fig.add_trace(
            go.Bar(x=plot_df.index, y=vols, name="Volume", marker_color=vol_colors, showlegend=False),
            row=2,
            col=1,
        )
        if "Vol_SMA50" in plot_df.columns and plot_df["Vol_SMA50"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df["Vol_SMA50"],
                    mode="lines",
                    name="50D Vol Avg",
                    line=dict(color="#94a3b8", width=1.2),
                    showlegend=not use_kt,
                ),
                row=2,
                col=1,
            )

    try:
        from components import apply_plotly_theme
        apply_plotly_theme(fig)
    except Exception:
        pass
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=26 if use_kt else 22, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=9.5 if use_kt else 10), itemwidth=30),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )
    if has_vol:
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", row=2, col=1)
    if rs_active:
        fig.update_yaxes(showgrid=False, showticklabels=False, row=1, col=1, secondary_y=True)
    return fig


def build_canslim_stats_strip_html(row: pd.Series | dict, *, curr_sym: str = "₹") -> str:
    """Render a compact docked institutional CANSLIM quarterly & flow telemetry strip."""
    import html as _html

    get_val = row.get if isinstance(row, dict) else row.get

    def _fmt_pct(val, high_thresh: float = 25.0) -> tuple[str, str]:
        if val is None or pd.isna(val):
            return "—", "#64748b"
        fv = float(val)
        col = "#10b981" if fv >= high_thresh else ("#38bdf8" if fv > 0 else "#f43f5e")
        return f"{fv:+.1f}%", col

    eps_s, eps_c = _fmt_pct(get_val("EPS_Growth"), 25.0)
    sls_s, sls_c = _fmt_pct(get_val("Sales_Growth"), 25.0)

    roe_v = get_val("ROE")
    if roe_v is not None and pd.notna(roe_v) and float(roe_v) != 0:
        rf = float(roe_v)
        roe_s, roe_c = f"{rf:.1f}%", ("#10b981" if rf >= 17.0 else ("#f59e0b" if rf >= 10.0 else "#94a3b8"))
    else:
        roe_s, roe_c = "—", "#64748b"

    udv_v = get_val("UDVR_50D")
    if udv_v is not None and pd.notna(udv_v) and float(udv_v) > 0:
        uv = float(udv_v)
        udv_s, udv_c = f"{uv:.2f}x", ("#10b981" if uv >= 1.2 else ("#38bdf8" if uv >= 1.0 else "#f59e0b"))
    else:
        udv_s, udv_c = "—", "#64748b"

    d52_v = get_val("Dist_52W_High")
    d52_s = f"{float(d52_v):+.1f}%" if d52_v is not None and pd.notna(d52_v) else "—"
    d52_c = "#10b981" if (d52_v is not None and pd.notna(d52_v) and float(d52_v) >= -5.0) else "#cbd5e1"

    dcr_v = get_val("DCR_Pct")
    dcr_pill = ""
    try:
        from components import build_dcr_mini_gauge_html
        dcr_pill = build_dcr_mini_gauge_html(float(dcr_v) if dcr_v is not None and pd.notna(dcr_v) else None, label="DCR")
    except Exception:
        dcr_pill = ""

    funda_badge = _html.escape(str(get_val("Funda_Badge", "—") or "—"))
    rs_line = _html.escape(str(get_val("RS_Line_Status", "—") or "—"))

    cells = [
        ("C · EPS YoY", eps_s, eps_c),
        ("C · Sales YoY", sls_s, sls_c),
        ("A · ROE", roe_s, roe_c),
        ("I · 50D U/D Vol", udv_s, udv_c),
        ("Off 52W High", d52_s, d52_c),
    ]
    cell_html = "".join(
        f'<div style="background:rgba(2,6,23,0.45);border:1px solid rgba(255,255,255,0.06);border-radius:7px;padding:6px 10px;">'
        f'<div style="font-size:0.64rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;">{lbl}</div>'
        f'<div style="font-size:0.9rem;font-weight:800;color:{col};font-family:\'JetBrains Mono\',monospace;">{val}</div>'
        f"</div>"
        for lbl, val, col in cells
    )
    return (
        f'<div style="background:rgba(15,23,42,0.88);border:1px solid rgba(56,189,248,0.20);'
        f'border-radius:10px;padding:9px 12px;margin-top:8px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:7px;">'
        f'<span style="font-size:0.72rem;font-weight:800;color:#94a3b8;text-transform:uppercase;letter-spacing:0.6px;">'
        f'🧬 CANSLIM & Accumulation Strip · <span style="color:#e2e8f0;">{funda_badge}</span> · <span style="color:#38bdf8;">{rs_line}</span></span>'
        f'{dcr_pill}</div>'
        f'<div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;">{cell_html}</div>'
        f"</div>"
    )


def build_preset_rationale_html(preset_counts: dict[str, int] | None = None) -> str:
    """Build structured HTML explaining exact quantitative rules behind each regime preset."""
    import html as _html

    counts = preset_counts or {}
    blocks = []
    for pid in ("bull", "sideways", "bear"):
        spec = PRESET_SPECS.get(pid, {})
        cnt = int(counts.get(pid, 0))
        lbl = _html.escape(str(spec.get("label", pid)))
        hlp = _html.escape(str(spec.get("help", "")))
        blocks.append(
            f'<div style="padding:7px 9px;border-radius:7px;background:rgba(15,23,42,0.8);'
            f'border:1px solid rgba(255,255,255,0.08);margin-bottom:6px;">'
            f'<div style="display:flex;justify-content:space-between;font-size:0.78rem;font-weight:800;color:#f8fafc;">'
            f'<span>{lbl}</span><span style="color:#38bdf8;font-family:monospace;">{cnt} matched</span></div>'
            f'<div style="font-size:0.72rem;color:#cbd5e1;margin-top:2px;line-height:1.35;">{hlp}</div></div>'
        )
    return "".join(blocks)


def build_outlier_scatter_fig(
    filtered_df: pd.DataFrame,
    *,
    curr_sym: str = "₹",
    adtv_col_label: str = "20D ADTV",
    max_points: int = 250,
    height: int = 520,
) -> go.Figure:
    """Build an Outlier Launch-Pocket scatter plotting Dist vs 21 EMA (X) vs RS Rating (Y), sized by ADTV."""
    import math

    fig = go.Figure()
    if (
        filtered_df is None
        or filtered_df.empty
        or "Dist_21EMA" not in filtered_df.columns
        or "RS_Rating" not in filtered_df.columns
    ):
        try:
            from components import apply_plotly_theme
            apply_plotly_theme(fig)
        except Exception:
            pass
        return fig

    work = filtered_df.head(max_points).copy()
    work["Dist_21EMA"] = pd.to_numeric(work["Dist_21EMA"], errors="coerce")
    work["RS_Rating"] = pd.to_numeric(work["RS_Rating"], errors="coerce")
    work = work.dropna(subset=["Dist_21EMA", "RS_Rating"])
    if work.empty:
        return fig

    adtv_vals = (
        pd.to_numeric(work["ADTV_20D"], errors="coerce").fillna(1.0).clip(lower=0.1)
        if "ADTV_20D" in work.columns
        else pd.Series(1.0, index=work.index)
    )
    max_adtv = max(1.0, float(adtv_vals.max()))
    sizes = [8.0 + 20.0 * math.sqrt(float(v) / max_adtv) for v in adtv_vals.tolist()]

    in_pocket = (work["Dist_21EMA"] >= -2.0) & (work["Dist_21EMA"] <= 4.0) & (work["RS_Rating"] >= 80.0)
    high_rs = work["RS_Rating"] >= 80.0
    colors = [
        "#10b981" if pkt else ("#38bdf8" if hrs else "#64748b")
        for pkt, hrs in zip(in_pocket.tolist(), high_rs.tolist())
    ]

    tickers = work["Ticker"].astype(str).tolist() if "Ticker" in work.columns else [str(i) for i in work.index]
    show_labels = [
        tk if (pkt or idx < 18) else ""
        for idx, (tk, pkt) in enumerate(zip(tickers, in_pocket.tolist()))
    ]
    prices = pd.to_numeric(work["Price"], errors="coerce").fillna(0.0).tolist() if "Price" in work.columns else [0.0] * len(work)
    dcrs = pd.to_numeric(work["DCR_Pct"], errors="coerce").fillna(0.0).tolist() if "DCR_Pct" in work.columns else [0.0] * len(work)
    customdata = list(zip(tickers, prices, adtv_vals.tolist(), dcrs))

    fig.add_shape(
        type="rect",
        x0=-2.0,
        x1=4.0,
        y0=80.0,
        y1=100.5,
        fillcolor="rgba(16, 185, 129, 0.09)",
        line=dict(color="rgba(16, 185, 129, 0.55)", dash="dot", width=1.2),
        layer="below",
    )
    fig.add_annotation(
        x=1.0,
        y=99.5,
        text="🎯 OPTIMAL LAUNCH POCKET (-2%..+4% vs 21E & RS≥80)",
        showarrow=False,
        font=dict(size=10, color="#34d399", family="JetBrains Mono"),
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color="rgba(148,163,184,0.35)", line_width=1)

    fig.add_trace(
        go.Scatter(
            x=work["Dist_21EMA"].tolist(),
            y=work["RS_Rating"].tolist(),
            mode="markers+text",
            text=show_labels,
            textposition="top center",
            textfont=dict(size=10, color="#e2e8f0", family="JetBrains Mono"),
            customdata=customdata,
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.88,
                line=dict(width=1.0, color="#0f172a"),
            ),
            hovertemplate=(
                f"<b>%{{customdata[0]}}</b><br>"
                f"Price: {curr_sym}%{{customdata[1]:,.2f}}<br>"
                f"RS Rating: <b>%{{y:.0f}}</b> | vs 21 EMA: <b>%{{x:+.2f}}%</b><br>"
                f"{adtv_col_label}: %{{customdata[2]:,.1f}} | DCR: %{{customdata[3]:.0f}}%"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    try:
        from components import apply_plotly_theme
        apply_plotly_theme(fig)
    except Exception:
        pass
    fig.update_layout(
        height=int(height),
        margin=dict(l=16, r=16, t=28, b=18),
        xaxis=dict(title="Distance vs 21-Day EMA (%)", ticksuffix="%", zeroline=False),
        yaxis=dict(title="RS Rating (0-99)", range=[max(0.0, float(work["RS_Rating"].min()) - 4.0), 102.0]),
        dragmode="pan",
    )
    return fig


@st.cache_data(ttl=600, show_spinner=False)
def _get_cached_kt_v3_telemetry(
    raw_tk: str,
    clean_tk: str,
    *,
    is_us: bool,
    univ_scope: str,
    benchmark_symbol: str | None,
    cohort_rs_rating: float,
    dist_65ema_pct: float | None,
    cap_stage_badge: str,
    portfolio_cost_basis: float | None,
    use_portfolio_anchor: bool,
    pre_earnings_days: int | None,
    matrix_sig: tuple,
) -> dict | None:
    """Compute & cache KushTracker v3.0 telemetry for one ticker across its deepest local matrix."""
    _ = matrix_sig  # Cache-key invalidation token
    ohlcv = _load_cached_ticker_ohlcv(
        raw_tk,
        clean_tk,
        univ_scope,
        is_us=is_us,
        max_bars=520,
        prefer_deepest=True,
    )
    if ohlcv is None or len(ohlcv) < 15:
        return None
    from kt_v3_engine import compute_kt_v3_telemetry

    mkt_code = "US" if is_us else "IN"
    bench_df, bench_lbl = _get_cached_kt_v3_benchmark(mkt_code, benchmark_symbol=benchmark_symbol)
    vol_proxy = _load_local_vol_proxy_series(is_us=is_us, max_bars=520)
    return compute_kt_v3_telemetry(
        ohlcv,
        bench_ohlcv=bench_df,
        vol_proxy_ser=vol_proxy,
        is_us=is_us,
        cohort_rs_rating=cohort_rs_rating,
        dist_65ema_pct=dist_65ema_pct,
        cap_stage_badge=cap_stage_badge,
        portfolio_cost_basis=portfolio_cost_basis,
        use_portfolio_anchor=use_portfolio_anchor,
        pre_earnings_days=pre_earnings_days,
        bench_label=bench_lbl,
    )


def _render_kt_v3_cockpit_hud(
    kt_telemetry: dict,
    *,
    is_held: bool = False,
) -> None:
    """Render the KushTracker v3.0 Live 5-Bar Playbook Strip + Tabbed 16-Row Institutional HUD."""
    import html as _html
    from kt_v3_engine import get_kt_v3_playbook_reference

    live_sigs = kt_telemetry.get("live_signals_5d", []) or []
    hud_rows = kt_telemetry.get("hud_rows", []) or []
    hdr = kt_telemetry.get("header_summary", {}) or {}

    if hdr.get("has_split_warning"):
        st.warning("⚠️ Potential unadjusted split / data spike detected (>85% low-vol jump in cache). Verify split-adjusted chart before sizing.")

    # 1. Live Active Signals Strip (Last 5 Bars + Active Cycle Badges)
    if live_sigs:
        badges_html = " ".join(
            f'<span title="{_html.escape(s["action"])}" style="display:inline-flex; align-items:center; gap:4px; '
            f'background:rgba(15,23,42,0.92); color:#f8fafc; border:1px solid {s["color"]}; '
            f'box-shadow:inset 0 0 8px {s["color"]}33; padding:2px 8px; border-radius:999px; '
            f'font-size:0.70rem; font-weight:800; font-family:\'JetBrains Mono\',monospace;">{_html.escape(s["signal"])}</span>'
            for s in live_sigs
        )
        lead_action = _html.escape(f"{live_sigs[0]['signal']} → {live_sigs[0]['action']}")
        sub_extra = f" <span style='color:#64748b;'>(+{len(live_sigs) - 1} more in hover/tab)</span>" if len(live_sigs) > 1 else ""
        strip_body = f"<div style='display:flex; flex-wrap:wrap; gap:5px; margin-bottom:5px;'>{badges_html}</div><div style='color:#cbd5e1; font-size:0.73rem;'>{lead_action}{sub_extra}</div>"
    else:
        act_safe = _html.escape(str(hdr.get("action_text", "WAITING FOR PIVOT BASE")))
        strip_body = (
            f"<div style='color:#94a3b8; font-size:0.73rem;'>"
            f"<b style='color:#38bdf8;'>📡 SCANNING:</b> No new signal labels in past 5 bars — <b>{act_safe}</b></div>"
        )

    st.markdown(
        f'<div style="background:rgba(8,13,24,0.92); border:1px solid rgba(56,189,248,0.22); border-left:3px solid #38bdf8; '
        f'border-radius:8px; padding:7px 10px; margin:6px 0 8px 0;">{strip_body}</div>',
        unsafe_allow_html=True,
    )

    def _rows_html(row_items: list[dict]) -> str:
        trs = []
        for r in row_items:
            trs.append(
                f'<tr style="border-bottom:1px solid rgba(30,41,59,0.75);">'
                f'<td style="padding:4px 6px; width:26px; text-align:center; font-size:0.80rem;">{_html.escape(str(r["tone"]))}</td>'
                f'<td style="padding:4px 8px; color:#94a3b8; font-size:0.73rem; font-weight:600; width:38%; white-space:nowrap;" title="{_html.escape(str(r.get("tooltip", "")))}">{_html.escape(str(r["label"]))}</td>'
                f'<td style="padding:4px 8px; color:#f8fafc; font-size:0.74rem; font-weight:700; font-family:\'JetBrains Mono\',monospace;">{_html.escape(str(r["value"]))}</td>'
                f'</tr>'
            )
        return (
            '<div style="background:#070b14; border:1px solid #1e293b; border-radius:8px; overflow:hidden; margin-bottom:6px;">'
            '<table style="width:100%; border-collapse:collapse;">'
            + "".join(trs)
            + "</table></div>"
        )

    tab_setup, tab_flow, tab_full, tab_guide = st.tabs([
        "⚡ Setup & Stop",
        "🦁 Flow, RS & Template",
        "📋 Full 16-Row HUD",
        "📖 Playbook Guide",
    ])
    with tab_setup:
        st.markdown(_rows_html([r for r in hud_rows if r["id"] in (1, 2, 3, 4, 5, 6, 13)]), unsafe_allow_html=True)
    with tab_flow:
        st.markdown(_rows_html([r for r in hud_rows if r["id"] in (7, 8, 9, 10, 11, 12, 14, 15, 16)]), unsafe_allow_html=True)
    with tab_full:
        st.markdown(_rows_html(hud_rows), unsafe_allow_html=True)
    with tab_guide:
        guide = get_kt_v3_playbook_reference()
        g_rows = []
        for g in guide:
            badge_bg = "rgba(16,185,129,0.14)" if g["side"] == "ENTRY" else "rgba(239,68,68,0.14)"
            g_rows.append(
                f'<tr style="border-bottom:1px solid rgba(30,41,59,0.75);">'
                f'<td style="padding:4px 7px; background:{badge_bg}; color:#f8fafc; font-size:0.71rem; font-weight:800; font-family:monospace; white-space:nowrap;">{_html.escape(g["symbol"])}</td>'
                f'<td style="padding:4px 8px; color:#cbd5e1; font-size:0.72rem;">{_html.escape(g["action"])}</td>'
                f'</tr>'
            )
        st.markdown(
            '<div style="background:#070b14; border:1px solid #1e293b; border-radius:8px; max-height:260px; overflow-y:auto;">'
            '<table style="width:100%; border-collapse:collapse;">' + "".join(g_rows) + "</table></div>",
            unsafe_allow_html=True,
        )


def _render_quick_chart_inspector(
    row: pd.Series,
    *,
    is_us: bool,
    curr_sym: str,
    univ_scope: str,
    entry_trigger: float | None = None,
    stop_loss: float | None = None,
    compact_stats_below: bool = False,
    state_prefix: str = "insp",
    benchmark_symbol: str | None = None,
    enable_kt_v3: bool = False,
) -> None:
    """Render an inline Quick Chart & Execution Card for a selected row in the screener."""
    clean_tk = str(row.get("Ticker", "") or "").strip()
    raw_tk = str(row.get("Raw_Ticker", clean_tk) or clean_tk).strip()
    tv_url = str(row.get("TradingView_URL", "") or "")
    price_val = float(row.get("Price", 0) or 0)

    is_held, portfolio_cost_basis = _extract_portfolio_cost_basis(st.session_state.get("portfolio_df"), clean_tk)
    pre_earnings_days = _get_pre_earnings_days(clean_tk)

    held_pill = (
        '<span style="background:rgba(16,185,129,0.16); color:#34d399; border:1px solid rgba(16,185,129,0.4); '
        'padding:1px 7px; border-radius:999px; font-size:0.68rem; font-weight:800;">👜 HELD</span>'
        if is_held
        else ""
    )
    earn_pill = (
        f'<span style="background:rgba(245,158,11,0.16); color:#fbbf24; border:1px solid rgba(245,158,11,0.45); '
        f'padding:1px 7px; border-radius:999px; font-size:0.68rem; font-weight:800;">⚠️ Earn {pre_earnings_days}d</span>'
        if (pre_earnings_days is not None and 0 <= pre_earnings_days <= 7)
        else ""
    )

    kt_telemetry: dict | None = None
    overlay_mode = "clean"
    disp_bars = 90
    show_rs_line = False
    signal_scope = "recent45"
    use_portfolio_anchor = False

    if enable_kt_v3:
        if is_held and portfolio_cost_basis:
            c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns([2.1, 1.35, 1.55, 0.9, 1.15])
        else:
            c_m1, c_m2, c_m3, c_m4 = st.columns([2.2, 1.4, 1.65, 0.95])
            c_m5 = None
        with c_m1:
            mode_pick = st.radio(
                "Chart Layer:",
                ["✨ KT v3.0 Overlay", "🔴 Risk & R-Zones", "Clean"],
                index=0,
                horizontal=True,
                key=f"{state_prefix}_kt_layer",
                label_visibility="collapsed",
            )
            overlay_mode = "kt_v3" if "KT" in mode_pick else ("risk_zones" if "Risk" in mode_pick else "clean")
        with c_m2:
            win_pick = st.radio(
                "Window:",
                ["90D", "150D", "250D"],
                index=0,
                horizontal=True,
                key=f"{state_prefix}_kt_win",
                label_visibility="collapsed",
            )
            disp_bars = 90 if win_pick == "90D" else (150 if win_pick == "150D" else 250)
        with c_m3:
            sig_pick = st.radio(
                "Signal Scope:",
                ["45D Clean Labels", "Full Window Signals"],
                index=0,
                horizontal=True,
                key=f"{state_prefix}_kt_sig_scope",
                label_visibility="collapsed",
            )
            signal_scope = "recent45" if "45D" in sig_pick else "all"
        with c_m4:
            show_rs_line = st.checkbox("📈 RS Line", value=False, key=f"{state_prefix}_kt_rs_line")
        if c_m5 is not None:
            with c_m5:
                use_portfolio_anchor = st.checkbox(
                    f"💼 Cost Stop ({curr_sym}{portfolio_cost_basis:,.0f})",
                    value=False,
                    key=f"{state_prefix}_kt_cost_anchor",
                )

        try:
            from price_history_manager import get_matrix_cache_signature
            m_sig = (
                get_matrix_cache_signature("breadth_matrix.pkl") if os.path.exists("breadth_matrix.pkl") else (0, 0),
                get_matrix_cache_signature("historical_prices_matrix.pkl") if os.path.exists("historical_prices_matrix.pkl") else (0, 0),
            )
        except Exception:
            m_sig = (0, 0)

        d65_val = float(row["Dist_65EMA"]) if pd.notna(row.get("Dist_65EMA")) else None
        kt_telemetry = _get_cached_kt_v3_telemetry(
            raw_tk,
            clean_tk,
            is_us=is_us,
            univ_scope=univ_scope,
            benchmark_symbol=benchmark_symbol,
            cohort_rs_rating=float(row.get("RS_Rating", 50.0) or 50.0),
            dist_65ema_pct=d65_val,
            cap_stage_badge=str(row.get("Cap_Stage_Badge", "") or ""),
            portfolio_cost_basis=portfolio_cost_basis,
            use_portfolio_anchor=use_portfolio_anchor,
            pre_earnings_days=pre_earnings_days,
            matrix_sig=m_sig,
        )

    kt_hdr = (kt_telemetry or {}).get("header_summary", {}) if kt_telemetry else {}
    stage_pill = (
        f'<span style="background:rgba(56,189,248,0.14); color:#38bdf8; border:1px solid rgba(56,189,248,0.35); '
        f'padding:1px 7px; border-radius:999px; font-size:0.68rem; font-weight:800;">{kt_hdr.get("stage_text", "")} • B{kt_hdr.get("base_count", 1)}</span>'
        if kt_hdr
        else ""
    )
    act_pill = (
        f'<span style="background:rgba(16,185,129,0.14); color:#e2e8f0; border:1px solid rgba(148,163,184,0.35); '
        f'padding:1px 7px; border-radius:999px; font-size:0.68rem; font-weight:700;">{kt_hdr.get("action_text", "")}</span>'
        if kt_hdr
        else ""
    )

    hdr_l, hdr_r = st.columns([5.2, 1.0] if enable_kt_v3 else [3.8, 1.2])
    with hdr_l:
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap; margin:4px 0 6px 0;">'
            f'<span style="color:#38bdf8; font-size:1.02rem; font-weight:900; font-family:\'JetBrains Mono\',monospace;">📌 {clean_tk} Quick Inspect</span>'
            f'{held_pill}{earn_pill}{stage_pill}{act_pill}'
            f'<span style="color:#cbd5e1; font-size:0.78rem;">{row.get("Industry_Badge", "")}</span>'
            f'<span style="color:#94a3b8; font-size:0.76rem;">• {row.get("Group_Role", "")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with hdr_r:
        if tv_url:
            st.link_button("📈 Open TV ↗", tv_url, use_container_width=True)

    ohlcv = None if kt_telemetry is not None else _load_cached_ticker_series_with_mas(raw_tk, clean_tk, univ_scope, is_us=is_us)
    d21 = float(row.get("Dist_21EMA", 0) or 0)
    d50 = row.get("Dist_50SMA")
    d65 = row.get("Dist_65EMA")
    d50_str = f"{float(d50):+.2f}%" if pd.notna(d50) else "N/A"
    d65_str = f"{float(d65):+.2f}%" if pd.notna(d65) else "N/A"
    stats_html = (
        f'<div style="background:rgba(15,23,42,0.82); border:1px solid rgba(56,189,248,0.22); border-radius:10px; padding:10px 13px; font-size:0.77rem; line-height:1.5;">'
        f'<div style="display:flex; justify-content:space-between;"><span style="color:#94a3b8;">Price / RS</span><b style="color:#f8fafc; font-family:monospace;">{curr_sym}{price_val:,.2f} · RS {float(row.get("RS_Rating", 0) or 0):.0f}</b></div>'
        f'<div style="display:flex; justify-content:space-between;"><span style="color:#94a3b8;">21E / 50S / 65E</span><b style="color:#38bdf8; font-family:monospace;">{d21:+.1f}% / {d50_str} / {d65_str}</b></div>'
        f'<div style="display:flex; justify-content:space-between;"><span style="color:#94a3b8;">DCR% / RVOL / ADR</span><b style="color:#34d399; font-family:monospace;">{float(row.get("DCR_Pct", 0) or 0):.0f}% / {float(row.get("RVOL_Today", 0) or 0):.2f}x / {float(row.get("ADR_20D", 0) or 0):.1f}%</b></div>'
        f'<hr style="border-color:rgba(255,255,255,0.07); margin:6px 0;">'
        f'<div><span style="color:#94a3b8;">Trigger:</span> <span style="color:#f8fafc;">{row.get("Setup_Trigger_State", "-")}</span></div>'
        f'<div><span style="color:#94a3b8;">VCP/Bar:</span> <span style="color:#facc15;">{row.get("VCP_Action_Badge", "-")}</span></div>'
        f'<div><span style="color:#94a3b8;">CANSLIM:</span> <span style="color:#cbd5e1;">{row.get("Funda_Badge", "-")}</span></div>'
        f'<div><span style="color:#94a3b8;">Stage/Cap:</span> <span style="color:#cbd5e1;">{row.get("Cap_Stage_Badge", "-")}</span></div>'
        f'</div>'
    )

    chart_ready = (kt_telemetry is not None and not kt_telemetry["series_df"].empty) or (ohlcv is not None and len(ohlcv) >= 15)
    chart_height = 485 if (enable_kt_v3 and not compact_stats_below) else (385 if (enable_kt_v3 and compact_stats_below) else (340 if compact_stats_below else 330))
    chart_key = f"{state_prefix}_quick_chart"

    if enable_kt_v3 and not compact_stats_below:
        c_chart, c_hud = st.columns([1.58, 1.0])
        with c_chart:
            if chart_ready:
                fig = build_canslim_ohlcv_chart(
                    kt_telemetry["series_df"] if kt_telemetry is not None else ohlcv,
                    ticker=clean_tk,
                    entry_trigger=entry_trigger,
                    stop_loss=stop_loss,
                    height=chart_height,
                    bars=disp_bars,
                    kt_telemetry=kt_telemetry,
                    overlay_mode=overlay_mode,
                    show_rs_line=show_rs_line,
                    signal_scope=signal_scope,
                )
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
            else:
                st.info("Cached daily chart unavailable locally for this symbol; use Open TV ↗.")
            st.markdown(build_canslim_stats_strip_html(row, curr_sym=curr_sym), unsafe_allow_html=True)
        with c_hud:
            if kt_telemetry is not None and overlay_mode != "clean":
                _render_kt_v3_cockpit_hud(kt_telemetry, is_held=is_held)
            else:
                st.markdown(stats_html, unsafe_allow_html=True)
        return

    if compact_stats_below:
        if chart_ready:
            fig = build_canslim_ohlcv_chart(
                kt_telemetry["series_df"] if kt_telemetry is not None else ohlcv,
                ticker=clean_tk,
                entry_trigger=entry_trigger,
                stop_loss=stop_loss,
                height=chart_height,
                bars=disp_bars,
                kt_telemetry=kt_telemetry,
                overlay_mode=overlay_mode,
                show_rs_line=show_rs_line,
                signal_scope=signal_scope,
            )
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
        else:
            st.info("Cached daily chart unavailable locally for this symbol; use Open TV ↗.")

        if kt_telemetry is not None and overlay_mode != "clean":
            _render_kt_v3_cockpit_hud(kt_telemetry, is_held=is_held)
        else:
            st.markdown(stats_html, unsafe_allow_html=True)
    else:
        c_chart, c_stats = st.columns([2.1, 1.0])
        with c_chart:
            if chart_ready:
                fig = build_canslim_ohlcv_chart(
                    kt_telemetry["series_df"] if kt_telemetry is not None else ohlcv,
                    ticker=clean_tk,
                    entry_trigger=entry_trigger,
                    stop_loss=stop_loss,
                    height=chart_height,
                    bars=disp_bars,
                    kt_telemetry=kt_telemetry,
                    overlay_mode=overlay_mode,
                    show_rs_line=show_rs_line,
                    signal_scope=signal_scope,
                )
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
            else:
                st.info("Cached daily chart unavailable locally for this symbol; use Open TV ↗.")
        with c_stats:
            st.markdown(stats_html, unsafe_allow_html=True)
        if kt_telemetry is not None and overlay_mode != "clean":
            _render_kt_v3_cockpit_hud(kt_telemetry, is_held=is_held)
    st.markdown(build_canslim_stats_strip_html(row, curr_sym=curr_sym), unsafe_allow_html=True)


def _series_to_json_list(ser: pd.Series | None, digits: int = 2) -> list[float | None]:
    """Convert a numeric Series slice into JSON-safe rounded floats with None for NaN."""
    if ser is None:
        return []
    out: list[float | None] = []
    for val in ser.tolist():
        if pd.isna(val):
            out.append(None)
        else:
            fval = float(val)
            out.append(int(round(fval)) if digits == 0 else round(fval, digits))
    return out


def build_hover_deck_payload(
    filtered_df: pd.DataFrame,
    *,
    univ_scope: str,
    curr_sym: str,
    max_rows: int = 100,
    bars: int = 90,
) -> list[dict]:
    """Build a compact JSON-serializable payload of row metrics & cached OHLCV/MA bars for Hover Deck."""
    if filtered_df is None or filtered_df.empty:
        return []

    held_set: set[str] = set()
    try:
        pf = st.session_state.get("portfolio_df")
        if pf is not None and not pf.empty:
            t_col = "ticker" if "ticker" in pf.columns else ("NSE TICKER" if "NSE TICKER" in pf.columns else pf.columns[0])
            held_set = {str(x).replace(".NS", "").replace(".BO", "").strip().upper() for x in pf[t_col].tolist()}
    except Exception:
        held_set = set()

    rows_payload: list[dict] = []
    slice_df = filtered_df.head(max_rows)
    for _, row in slice_df.iterrows():
        clean_tk = str(row.get("Ticker", "") or "").strip()
        raw_tk = str(row.get("Raw_Ticker", clean_tk) or clean_tk).strip()
        price_val = float(row.get("Price", 0) or 0)
        d21_val = float(row.get("Dist_21EMA", 0) or 0) if pd.notna(row.get("Dist_21EMA")) else None
        d50_val = round(float(row["Dist_50SMA"]), 2) if pd.notna(row.get("Dist_50SMA")) else None
        d65_val = round(float(row["Dist_65EMA"]), 2) if pd.notna(row.get("Dist_65EMA")) else None
        smooth_val = round(float(row["Smooth_RS"]), 2) if pd.notna(row.get("Smooth_RS")) else None

        ohlcv = _load_cached_ticker_series_with_mas(raw_tk, clean_tk, univ_scope)
        series_dict: dict = {}
        if ohlcv is not None and len(ohlcv) >= 15:
            plot_df = ohlcv.tail(bars).copy()
            dates_str = [pd.Timestamp(idx).strftime("%d %b") for idx in plot_df.index]
            has_ohlc = all(c in plot_df.columns and plot_df[c].notna().any() for c in ("Open", "High", "Low", "Close"))
            series_dict = {
                "dates": dates_str,
                "has_ohlc": bool(has_ohlc),
                "o": _series_to_json_list(plot_df["Open"], 2) if has_ohlc else [],
                "h": _series_to_json_list(plot_df["High"], 2) if has_ohlc else [],
                "l": _series_to_json_list(plot_df["Low"], 2) if has_ohlc else [],
                "c": _series_to_json_list(plot_df["Close"], 2),
                "v": _series_to_json_list(plot_df["Volume"], 0) if "Volume" in plot_df.columns else [],
                "e21": _series_to_json_list(plot_df["EMA21"], 2) if "EMA21" in plot_df.columns else [],
                "s50": _series_to_json_list(plot_df["SMA50"], 2) if "SMA50" in plot_df.columns else [],
                "e65": _series_to_json_list(plot_df["EMA65"], 2) if "EMA65" in plot_df.columns else [],
                "vs50": _series_to_json_list(plot_df["Vol_SMA50"], 0) if "Vol_SMA50" in plot_df.columns else [],
            }

        rows_payload.append(
            {
                "rank": int(row.get("Rank", 0) or 0) if pd.notna(row.get("Rank")) else 0,
                "ticker": clean_tk,
                "tv_url": str(row.get("TradingView_URL", "") or ""),
                "price": round(price_val, 2),
                "price_str": f"{curr_sym}{price_val:,.2f}",
                "rs": int(round(float(row.get("RS_Rating", 0) or 0))),
                "smooth_rs": smooth_val,
                "d21": round(d21_val, 2) if d21_val is not None else None,
                "d50": d50_val,
                "d65": d65_val,
                "dcr": int(round(float(row.get("DCR_Pct", 0) or 0))) if pd.notna(row.get("DCR_Pct")) else None,
                "rvol": round(float(row.get("RVOL_Today", 0) or 0), 2) if pd.notna(row.get("RVOL_Today")) else None,
                "adr": round(float(row.get("ADR_20D", 0) or 0), 1) if pd.notna(row.get("ADR_20D")) else None,
                "trigger": str(row.get("Setup_Trigger_State", "—") or "—"),
                "vcp": str(row.get("VCP_Action_Badge", "—") or "—"),
                "industry": str(row.get("Industry_Badge", "—") or "—"),
                "role": str(row.get("Group_Role", "—") or "—"),
                "funda": str(row.get("Funda_Badge", "—") or "—"),
                "stage": str(row.get("Cap_Stage_Badge", "—") or "—"),
                "held": clean_tk.upper() in held_set,
                "series": series_dict,
            }
        )
    return rows_payload


def render_hover_deck_html(
    payload: list[dict],
    *,
    initial_ticker: str | None = None,
    curr_sym: str = "₹",
    height_px: int = 610,
) -> str:
    """Build self-contained zero-lag HTML/CSS/Canvas hover + keyboard split-pane chart deck."""
    init_idx = 0
    if initial_ticker:
        for idx, item in enumerate(payload):
            if item.get("ticker") == initial_ticker:
                init_idx = idx
                break

    safe_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    safe_curr = json.dumps(curr_sym).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  :root {{
    color-scheme: dark;
    --bg: #0b1120;
    --panel: #0f172a;
    --border: rgba(56, 189, 248, 0.22);
    --border-subtle: rgba(148, 163, 184, 0.14);
    --text: #f8fafc;
    --muted: #94a3b8;
    --cyan: #38bdf8;
    --green: #34d399;
    --red: #f87171;
    --amber: #f59e0b;
    --purple: #a855f7;
    --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    --sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0; padding: 0; height: 100%; width: 100%;
    background: transparent; color: var(--text); font-family: var(--sans);
    overflow: hidden;
  }}
  .hd-root {{
    display: grid;
    grid-template-columns: minmax(290px, 0.39fr) minmax(380px, 0.61fr);
    grid-template-rows: minmax(0, 1fr);
    gap: 12px;
    height: {height_px}px;
    width: 100%;
    outline: none;
    overflow: hidden;
  }}
  .hd-table-wrap {{
    background: rgba(15, 23, 42, 0.90);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: auto;
    height: 100%;
    min-height: 0;
    contain: layout paint;
  }}
  .hd-stage {{
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    min-width: 0;
    overflow: hidden;
    contain: layout style paint;
  }}
  :where(.hd-table-wrap, .hd-row, .hd-btn, .hd-link):focus-visible {{
    outline: 2px solid var(--cyan);
    outline-offset: -2px;
  }}
  table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.77rem;
    font-variant-numeric: tabular-nums;
  }}
  thead {{
    position: sticky;
    inset-block-start: 0;
    z-index: 2;
    background: #111c33;
  }}
  th {{
    text-align: left;
    padding: 7px 8px;
    color: var(--muted);
    font-size: 0.68rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  th.num, td.num {{ text-align: right; font-family: var(--mono); }}
  .hd-row {{
    cursor: pointer;
    transition: background-color 80ms ease-out;
  }}
  .hd-row td {{
    padding: 6px 8px;
    border-bottom: 1px solid var(--border-subtle);
    white-space: nowrap;
  }}
  .hd-row:hover {{
    background: rgba(56, 189, 248, 0.12);
  }}
  .hd-row.is-active {{
    background: rgba(56, 189, 248, 0.20);
  }}
  .hd-row.is-active td:first-child {{
    box-shadow: inset 3px 0 0 var(--cyan);
  }}
  .hd-row.is-pinned.is-active {{
    background: rgba(16, 185, 129, 0.16);
  }}
  .hd-row.is-pinned.is-active td:first-child {{
    box-shadow: inset 3px 0 0 var(--green);
  }}
  .tk-cell {{
    font-family: var(--mono);
    font-weight: 800;
    color: var(--text);
  }}
  .pos {{ color: var(--green); font-weight: 700; }}
  .neg {{ color: var(--red); font-weight: 700; }}
  .stage-hdr {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    flex-shrink: 0;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-subtle);
  }}
  .stage-title {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    min-width: 0;
  }}
  .stage-tk {{
    color: var(--cyan);
    font-family: var(--mono);
    font-size: 1.08rem;
    font-weight: 900;
  }}
  .pill {{
    font-size: 0.68rem;
    font-weight: 800;
    padding: 2px 8px;
    border-radius: 999px;
    font-family: var(--mono);
  }}
  .pill-held {{
    background: rgba(16, 185, 129, 0.16);
    color: var(--green);
    border: 1px solid rgba(16, 185, 129, 0.40);
  }}
  .pill-pin {{
    background: rgba(16, 185, 129, 0.15);
    color: var(--green);
    border: 1px solid rgba(16, 185, 129, 0.35);
    cursor: pointer;
  }}
  .pill-hover {{
    background: rgba(56, 189, 248, 0.12);
    color: var(--cyan);
    border: 1px solid rgba(56, 189, 248, 0.30);
    cursor: pointer;
  }}
  .stage-actions {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .hd-btn, .hd-link {{
    background: rgba(30, 41, 59, 0.9);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 9px;
    font-size: 0.72rem;
    font-weight: 700;
    cursor: pointer;
    text-decoration: none;
    line-height: 1.2;
  }}
  .hd-btn:hover, .hd-link:hover {{
    border-color: var(--cyan);
    background: rgba(56, 189, 248, 0.15);
  }}
  .legend-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    font-size: 0.69rem;
    color: var(--muted);
    flex-wrap: wrap;
    flex-shrink: 0;
  }}
  .legend-keys {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
  .lg-dot {{ display: inline-block; width: 10px; height: 2px; vertical-align: middle; margin-right: 4px; }}
  .canvas-box {{
    position: relative;
    flex: 1 1 0;
    min-height: 0;
    width: 100%;
    background: rgba(7, 11, 22, 0.80);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
  }}
  canvas {{
    position: absolute;
    inset: 0;
    display: block;
    width: 100%;
    height: 100%;
  }}
  .ohcl-hud {{
    position: absolute;
    top: 6px;
    left: 8px;
    font-family: var(--mono);
    font-size: 0.69rem;
    color: #cbd5e1;
    background: rgba(11, 17, 32, 0.78);
    padding: 2px 7px;
    border-radius: 4px;
    pointer-events: none;
  }}
  .stats-grid {{
    margin-top: 8px;
    flex-shrink: 0;
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 6px;
    font-size: 0.71rem;
  }}
  .stat-cell {{
    background: rgba(30, 41, 59, 0.50);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 5px 8px;
    min-width: 0;
  }}
  .stat-lbl {{ color: var(--muted); font-size: 0.63rem; text-transform: uppercase; font-weight: 700; }}
  .stat-val {{
    color: var(--text);
    font-family: var(--mono);
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-top: 1px;
  }}
  .foot-hint {{
    margin-top: 5px;
    flex-shrink: 0;
    font-size: 0.67rem;
    color: #64748b;
    display: flex;
    justify-content: space-between;
    gap: 8px;
  }}
</style>
</head>
<body>
<div class="hd-root" id="hdRoot" tabindex="0">
  <div class="hd-table-wrap" id="hdTableWrap" tabindex="0" role="region" aria-label="RS Screener Hover Deck Table">
    <table>
      <thead>
        <tr>
          <th class="num">#</th>
          <th>Ticker</th>
          <th class="num">Price</th>
          <th class="num">RS</th>
          <th class="num">vs 21E</th>
          <th class="num">SmRS</th>
        </tr>
      </thead>
      <tbody id="hdTbody"></tbody>
    </table>
  </div>
  <div class="hd-stage" id="hdStage">
    <div class="stage-hdr">
      <div class="stage-title">
        <span class="stage-tk" id="stTicker">—</span>
        <span class="pill pill-held" id="stHeld" style="display:none;">👜 HELD</span>
        <span style="color:#cbd5e1;font-size:0.74rem;" id="stInd"></span>
        <span style="color:#94a3b8;font-size:0.71rem;" id="stRole"></span>
      </div>
      <div class="stage-actions">
        <button type="button" class="pill pill-hover" id="btnLock" title="Click row or press L to pin/unpin">🔓 Hover Active</button>
        <button type="button" class="hd-btn" id="btnPrev" aria-label="Previous stock">◀</button>
        <span style="font-family:var(--mono);font-size:0.71rem;color:#94a3b8;" id="stCounter">0/0</span>
        <button type="button" class="hd-btn" id="btnNext" aria-label="Next stock">▶</button>
        <a class="hd-link" id="stTvLink" href="#" target="_blank" rel="noopener noreferrer" style="display:none;">📈 TV ↗</a>
      </div>
    </div>
    <div class="legend-bar">
      <div class="legend-keys">
        <span><i class="lg-dot" style="background:#38bdf8;"></i>21 EMA</span>
        <span><i class="lg-dot" style="background:#f59e0b;"></i>50 SMA</span>
        <span><i class="lg-dot" style="background:#a855f7;"></i>65 EMA</span>
        <span><i class="lg-dot" style="background:#94a3b8;"></i>50D Vol Avg</span>
      </div>
      <span id="stCounterNote" style="color:#64748b;"></span>
    </div>
    <div class="canvas-box" id="canvasBox">
      <canvas id="hdCanvas"></canvas>
      <div class="ohcl-hud" id="ohlcvHud"></div>
    </div>
    <div class="stats-grid">
      <div class="stat-cell"><div class="stat-lbl">Price / RS</div><div class="stat-val" id="sPriceRs">—</div></div>
      <div class="stat-cell"><div class="stat-lbl">21E / 50S / 65E</div><div class="stat-val" style="color:#38bdf8;" id="sMaDist">—</div></div>
      <div class="stat-cell"><div class="stat-lbl">DCR / RVOL / ADR</div><div class="stat-val" style="color:#34d399;" id="sBarQual">—</div></div>
      <div class="stat-cell"><div class="stat-lbl">Trigger</div><div class="stat-val" id="sTrigger">—</div></div>
      <div class="stat-cell"><div class="stat-lbl">VCP / Bar</div><div class="stat-val" style="color:#facc15;" id="sVcp">—</div></div>
      <div class="stat-cell"><div class="stat-lbl">CANSLIM</div><div class="stat-val" id="sFunda">—</div></div>
      <div class="stat-cell"><div class="stat-lbl">Stage & Cap</div><div class="stat-val" id="sStage">—</div></div>
      <div class="stat-cell"><div class="stat-lbl">Smooth RS</div><div class="stat-val" id="sSmooth">—</div></div>
    </div>
    <div class="foot-hint">
      <span>🖱️ Hover any row or press ↑ / ↓ keys to switch charts instantly</span>
      <span>Click a row (or L) to pin while inspecting candle crosshair</span>
    </div>
  </div>
</div>

<script>
(() => {{
  const DATA = {safe_json};
  const CURR = {safe_curr};
  let activeIdx = {init_idx};
  let isLocked = false;
  let hoverBarIdx = null;
  let rafId = null;

  const tbody = document.getElementById('hdTbody');
  const rootEl = document.getElementById('hdRoot');
  const wrapEl = document.getElementById('hdTableWrap');
  const canvasBox = document.getElementById('canvasBox');
  const canvas = document.getElementById('hdCanvas');
  const ctx = canvas.getContext('2d');
  const hud = document.getElementById('ohlcvHud');
  const btnLock = document.getElementById('btnLock');

  function fmtPct(v) {{
    if (v === null || v === undefined || Number.isNaN(v)) return 'N/A';
    return (v >= 0 ? '+' : '') + Number(v).toFixed(1) + '%';
  }}

  function fmtVol(v) {{
    if (!v && v !== 0) return 'N/A';
    if (v >= 1e7) return (v / 1e7).toFixed(2) + 'Cr';
    if (v >= 1e5) return (v / 1e5).toFixed(1) + 'L';
    if (v >= 1e3) return (v / 1e3).toFixed(1) + 'K';
    return String(Math.round(v));
  }}

  const rowEls = [];
  DATA.forEach((d, idx) => {{
    const tr = document.createElement('tr');
    tr.className = 'hd-row';
    tr.dataset.idx = String(idx);
    tr.tabIndex = 0;

    const tdRank = document.createElement('td');
    tdRank.className = 'num';
    tdRank.style.color = '#94a3b8';
    tdRank.textContent = d.rank || (idx + 1);

    const tdTk = document.createElement('td');
    tdTk.className = 'tk-cell';
    tdTk.textContent = (d.held ? '👜 ' : '') + d.ticker;

    const tdPr = document.createElement('td');
    tdPr.className = 'num';
    tdPr.textContent = d.price_str;

    const tdRs = document.createElement('td');
    tdRs.className = 'num';
    tdRs.style.color = d.rs >= 90 ? '#38bdf8' : '#cbd5e1';
    tdRs.textContent = d.rs;

    const td21 = document.createElement('td');
    td21.className = 'num ' + (d.d21 !== null && d.d21 >= 0 ? 'pos' : 'neg');
    td21.textContent = fmtPct(d.d21);

    const tdSm = document.createElement('td');
    tdSm.className = 'num';
    tdSm.style.color = '#cbd5e1';
    tdSm.textContent = d.smooth_rs !== null ? Number(d.smooth_rs).toFixed(2) : '—';

    tr.append(tdRank, tdTk, tdPr, tdRs, td21, tdSm);
    tbody.appendChild(tr);
    rowEls.push(tr);
  }});

  function updateLockBadge() {{
    if (isLocked) {{
      btnLock.className = 'pill pill-pin';
      btnLock.textContent = '📌 Pinned (Esc to unlock)';
    }} else {{
      btnLock.className = 'pill pill-hover';
      btnLock.textContent = '🔓 Hover Active';
    }}
    rowEls.forEach((el, i) => {{
      el.classList.toggle('is-pinned', isLocked && i === activeIdx);
    }});
  }}

  function setText(id, text) {{
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }}

  function renderMeta(item) {{
    setText('stTicker', '📌 ' + item.ticker);
    document.getElementById('stHeld').style.display = item.held ? 'inline-block' : 'none';
    setText('stInd', item.industry || '');
    setText('stRole', item.role ? '• ' + item.role : '');
    setText('stCounter', (activeIdx + 1) + '/' + DATA.length);

    const tvEl = document.getElementById('stTvLink');
    if (item.tv_url && (item.tv_url.startsWith('https://') || item.tv_url.startsWith('http://'))) {{
      tvEl.href = item.tv_url;
      tvEl.style.display = 'inline-block';
    }} else {{
      tvEl.style.display = 'none';
    }}

    setText('sPriceRs', item.price_str + ' · RS ' + item.rs);
    setText('sMaDist', fmtPct(item.d21) + ' / ' + fmtPct(item.d50) + ' / ' + fmtPct(item.d65));
    const dcrS = item.dcr !== null ? item.dcr + '%' : 'N/A';
    const rvS = item.rvol !== null ? item.rvol.toFixed(2) + 'x' : 'N/A';
    const adrS = item.adr !== null ? item.adr.toFixed(1) + '%' : 'N/A';
    setText('sBarQual', dcrS + ' / ' + rvS + ' / ' + adrS);
    setText('sTrigger', item.trigger || '—');
    setText('sVcp', item.vcp || '—');
    setText('sFunda', item.funda || '—');
    setText('sStage', item.stage || '—');
    setText('sSmooth', item.smooth_rs !== null ? item.smooth_rs.toFixed(2) + 'x' : '—');
  }}

  function drawLine(xs, vals, yFn, color, width, dash) {{
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < vals.length; i++) {{
      const v = vals[i];
      if (v === null || v === undefined) {{ started = false; continue; }}
      const x = xs[i], y = yFn(v);
      if (!started) {{ ctx.moveTo(x, y); started = true; }}
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();
    ctx.restore();
  }}

  function drawChart() {{
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(200, canvasBox.clientWidth || Math.floor(canvasBox.getBoundingClientRect().width));
    const h = Math.max(180, canvasBox.clientHeight || Math.floor(canvasBox.getBoundingClientRect().height));
    const targetW = Math.floor(w * dpr);
    const targetH = Math.floor(h * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {{
      canvas.width = targetW;
      canvas.height = targetH;
    }}
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    if (!DATA.length) return;
    const item = DATA[activeIdx];
    const ser = item.series || {{}};
    const cArr = ser.c || [];
    const n = cArr.length;
    if (n < 5) {{
      ctx.fillStyle = '#94a3b8';
      ctx.font = '13px sans-serif';
      ctx.fillText('Cached 90D series unavailable for ' + item.ticker + ' — click Open TV ↗', 18, 48);
      hud.textContent = '';
      return;
    }}

    const padL = 10, padR = 54, padT = 26, padB = 22;
    const hasVol = Array.isArray(ser.v) && ser.v.length === n && ser.v.some(x => x && x > 0);
    const plotW = w - padL - padR;
    const priceH = hasVol ? Math.floor((h - padT - padB) * 0.72) : (h - padT - padB);
    const volTop = padT + priceH + 10;
    const volH = hasVol ? Math.max(30, h - volTop - padB) : 0;

    const hasOhlc = Boolean(ser.has_ohlc && ser.h && ser.h.length === n && ser.l && ser.l.length === n && ser.o && ser.o.length === n);
    let pMin = Infinity, pMax = -Infinity;
    const collectMinMax = (arr) => {{
      if (!arr) return;
      for (const v of arr) {{
        if (v !== null && v !== undefined) {{
          if (v < pMin) pMin = v;
          if (v > pMax) pMax = v;
        }}
      }}
    }};
    if (hasOhlc) {{ collectMinMax(ser.h); collectMinMax(ser.l); }}
    else {{ collectMinMax(cArr); }}
    collectMinMax(ser.e21);
    collectMinMax(ser.s50);
    collectMinMax(ser.e65);
    if (!Number.isFinite(pMin) || !Number.isFinite(pMax)) return;
    if (pMax === pMin) {{ pMax += 1; pMin -= 1; }}
    const pPad = (pMax - pMin) * 0.06;
    pMin -= pPad; pMax += pPad;

    const yPrice = (v) => padT + ((pMax - v) / (pMax - pMin)) * priceH;
    const stepX = plotW / n;
    const xs = Array.from({{ length: n }}, (_, i) => padL + (i + 0.5) * stepX);

    // Grid & Y labels
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.10)';
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px ui-monospace, monospace';
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {{
      const frac = g / 4;
      const y = padT + frac * priceH;
      const pv = pMax - frac * (pMax - pMin);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.fillText(pv >= 1000 ? pv.toFixed(0) : pv.toFixed(1), w - padR + 5, y + 3);
    }}

    // Candles or Close Line
    if (hasOhlc) {{
      const bw = Math.max(2, Math.min(9, stepX * 0.64));
      for (let i = 0; i < n; i++) {{
        const o = ser.o[i], hi = ser.h[i], lo = ser.l[i], cl = cArr[i];
        if (o === null || hi === null || lo === null || cl === null) continue;
        const up = cl >= o;
        const col = up ? '#10b981' : '#f87171';
        const x = xs[i];
        ctx.strokeStyle = col;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, yPrice(hi)); ctx.lineTo(x, yPrice(lo)); ctx.stroke();
        const yO = yPrice(o), yC = yPrice(cl);
        const top = Math.min(yO, yC);
        const bh = Math.max(1.4, Math.abs(yC - yO));
        ctx.fillStyle = up ? 'rgba(16, 185, 129, 0.78)' : 'rgba(248, 113, 113, 0.78)';
        ctx.fillRect(x - bw / 2, top, bw, bh);
      }}
    }} else {{
      drawLine(xs, cArr, yPrice, '#f8fafc', 1.8, null);
    }}

    drawLine(xs, ser.e21 || [], yPrice, '#38bdf8', 1.35, [3, 3]);
    drawLine(xs, ser.s50 || [], yPrice, '#f59e0b', 1.45, null);
    drawLine(xs, ser.e65 || [], yPrice, '#a855f7', 1.35, [5, 3]);

    // Last price tag
    const lastC = cArr[n - 1];
    if (lastC !== null && lastC !== undefined) {{
      const ly = yPrice(lastC);
      ctx.fillStyle = '#0284c7';
      ctx.fillRect(w - padR + 2, ly - 8, padR - 4, 16);
      ctx.fillStyle = '#f8fafc';
      ctx.fillText(lastC >= 1000 ? lastC.toFixed(0) : lastC.toFixed(1), w - padR + 5, ly + 3);
    }}

    // Volume subplot
    if (hasVol) {{
      let vMax = 1;
      for (const vv of ser.v) if (vv && vv > vMax) vMax = vv;
      const yVol = (v) => volTop + volH - (Math.max(0, v || 0) / vMax) * volH;
      const bw = Math.max(1.5, Math.min(8, stepX * 0.64));
      for (let i = 0; i < n; i++) {{
        const vv = ser.v[i];
        if (!vv) continue;
        const cl = cArr[i], op = hasOhlc ? ser.o[i] : (i > 0 ? cArr[i - 1] : cl);
        const up = cl >= op;
        ctx.fillStyle = up ? 'rgba(16, 185, 129, 0.58)' : 'rgba(248, 113, 113, 0.55)';
        const yv = yVol(vv);
        ctx.fillRect(xs[i] - bw / 2, yv, bw, volTop + volH - yv);
      }}
      if (ser.vs50 && ser.vs50.length === n) {{
        drawLine(xs, ser.vs50, yVol, '#94a3b8', 1.15, null);
      }}
    }}

    // X date ticks
    if (ser.dates && ser.dates.length === n) {{
      ctx.fillStyle = '#64748b';
      const tickCount = Math.min(6, n);
      for (let t = 0; t < tickCount; t++) {{
        const idx = Math.min(n - 1, Math.round((t / Math.max(1, tickCount - 1)) * (n - 1)));
        ctx.fillText(ser.dates[idx], Math.max(padL, xs[idx] - 16), h - 6);
      }}
    }}

    // Crosshair / OHLC HUD
    const bi = (hoverBarIdx !== null && hoverBarIdx >= 0 && hoverBarIdx < n) ? hoverBarIdx : (n - 1);
    if (hoverBarIdx !== null && hoverBarIdx >= 0 && hoverBarIdx < n) {{
      ctx.save();
      ctx.strokeStyle = 'rgba(226, 232, 240, 0.35)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(xs[bi], padT); ctx.lineTo(xs[bi], h - padB); ctx.stroke();
      ctx.restore();
    }}
    const dtTxt = ser.dates ? ser.dates[bi] : '';
    if (hasOhlc) {{
      hud.textContent = `${{dtTxt}}  O:${{ser.o[bi]}} H:${{ser.h[bi]}} L:${{ser.l[bi]}} C:${{cArr[bi]}}  Vol:${{fmtVol((ser.v || [])[bi])}}`;
    }} else {{
      hud.textContent = `${{dtTxt}}  Close:${{CURR}}${{cArr[bi]}}  Vol:${{fmtVol((ser.v || [])[bi])}}`;
    }}
  }}

  function scheduleDraw() {{
    if (rafId !== null) cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(() => {{
      rafId = null;
      drawChart();
    }});
  }}

  function selectRow(idx, scrollIntoViewFlag) {{
    if (!DATA.length) return;
    const clamped = Math.max(0, Math.min(DATA.length - 1, idx));
    if (clamped !== activeIdx) {{
      hoverBarIdx = null;
    }}
    activeIdx = clamped;
    rowEls.forEach((el, i) => {{
      const isAct = i === activeIdx;
      el.classList.toggle('is-active', isAct);
      el.setAttribute('aria-selected', isAct ? 'true' : 'false');
    }});
    updateLockBadge();
    renderMeta(DATA[activeIdx]);
    if (scrollIntoViewFlag && rowEls[activeIdx]) {{
      rowEls[activeIdx].scrollIntoView({{ block: 'nearest' }});
    }}
    scheduleDraw();
  }}

  tbody.addEventListener('pointermove', (e) => {{
    if (isLocked) return;
    const tr = e.target.closest('tr.hd-row');
    if (!tr) return;
    const idx = Number(tr.dataset.idx);
    if (!Number.isNaN(idx) && idx !== activeIdx) {{
      selectRow(idx, false);
    }}
  }});

  tbody.addEventListener('click', (e) => {{
    const tr = e.target.closest('tr.hd-row');
    if (!tr) return;
    const idx = Number(tr.dataset.idx);
    if (Number.isNaN(idx)) return;
    if (isLocked && idx === activeIdx) {{
      isLocked = false;
    }} else {{
      isLocked = true;
      selectRow(idx, false);
    }}
    updateLockBadge();
    rootEl.focus({{ preventScroll: true }});
  }});

  btnLock.addEventListener('click', () => {{
    isLocked = !isLocked;
    updateLockBadge();
    rootEl.focus({{ preventScroll: true }});
  }});

  document.getElementById('btnPrev').addEventListener('click', () => {{
    selectRow(activeIdx - 1, true);
    rootEl.focus({{ preventScroll: true }});
  }});
  document.getElementById('btnNext').addEventListener('click', () => {{
    selectRow(activeIdx + 1, true);
    rootEl.focus({{ preventScroll: true }});
  }});

  document.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowDown') {{
      e.preventDefault();
      selectRow(activeIdx + 1, true);
    }} else if (e.key === 'ArrowUp') {{
      e.preventDefault();
      selectRow(activeIdx - 1, true);
    }} else if (e.key === 'Home') {{
      e.preventDefault();
      selectRow(0, true);
    }} else if (e.key === 'End') {{
      e.preventDefault();
      selectRow(DATA.length - 1, true);
    }} else if (e.key === 'Escape' && isLocked) {{
      isLocked = false;
      updateLockBadge();
    }} else if ((e.key === 'l' || e.key === 'L') && !e.metaKey && !e.ctrlKey) {{
      isLocked = !isLocked;
      updateLockBadge();
    }}
  }});

  canvas.addEventListener('pointermove', (e) => {{
    if (!DATA.length) return;
    const ser = DATA[activeIdx].series || {{}};
    const n = (ser.c || []).length;
    if (!n) return;
    const rect = canvasBox.getBoundingClientRect();
    const padL = 10, padR = 54;
    const plotW = Math.max(1, rect.width - padL - padR);
    const relX = e.clientX - rect.left - padL;
    if (relX < 0 || relX > plotW) {{ hoverBarIdx = null; }}
    else {{ hoverBarIdx = Math.max(0, Math.min(n - 1, Math.floor((relX / plotW) * n))); }}
    scheduleDraw();
  }});
  canvas.addEventListener('pointerleave', () => {{
    hoverBarIdx = null;
    scheduleDraw();
  }});

  if (typeof ResizeObserver !== 'undefined') {{
    new ResizeObserver(() => scheduleDraw()).observe(canvasBox);
  }}
  window.addEventListener('resize', () => scheduleDraw());

  selectRow(activeIdx, false);
}})();
</script>
</body>
</html>"""


def _load_cached_ticker_series_with_mas(
    raw_ticker: str,
    clean_ticker: str,
    univ_scope: str,
    *,
    is_us: bool = False,
) -> pd.DataFrame | None:
    """Load local cached OHLCV series (or Cloud single-ticker Yahoo fallback) with MAs & Vol SMA."""
    sub = _load_cached_ticker_ohlcv(raw_ticker, clean_ticker, univ_scope, is_us=is_us)
    if sub is None or sub.empty:
        try:
            from components import load_cached_ticker_series_with_mas
            yf_sym = raw_ticker or (clean_ticker if is_us else f"{clean_ticker}.NS")
            return load_cached_ticker_series_with_mas(yf_sym, market=("US" if is_us else "IN"))
        except Exception:
            return None
    keep_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in sub.columns]
    out = sub[keep_cols].copy()
    out["EMA21"] = out["Close"].ewm(span=21, adjust=False).mean()
    out["SMA50"] = out["Close"].rolling(50, min_periods=20).mean()
    out["EMA65"] = out["Close"].ewm(span=65, adjust=False, min_periods=45).mean()
    if "Volume" in out.columns:
        out["Vol_SMA50"] = out["Volume"].rolling(50, min_periods=10).mean()
    return out


def render_leader_radar_cockpit(
    raw_df: pd.DataFrame,
    *,
    state_prefix: str,
    is_us: bool,
    curr_sym: str,
    price_col_label: str,
    price_fmt: str,
    adtv_col_label: str,
    adtv_fmt: str,
    search_ph: str,
    cohort_cap: int = 100,
    is_master: bool = False,
    univ_badge_txt: str = "",
    univ_scope: str = "750",
    diagnostics: dict | None = None,
    table_height: int | None = None,
    benchmark_symbol: str | None = None,
) -> None:
    """
    Renders the unified institutional Leader Radar Cockpit:
      - Bear Stalking Preset / Reset Toolbar (safe `on_click` callbacks)
      - Pre-Slice ADTV Gate (when `is_master=True`) & Cohort Slicing
      - Non-Intrusive Bull/Bear KPI Strip (`Defending 21 EMA` w/ `50S≥-4% | 65E` delta)
      - 15+ Tactical Filters, Search, Sorting, TV Watchlist Exporter & Rich DataFrame
    """
    if raw_df is None or raw_df.empty:
        return

    diag = diagnostics if diagnostics is not None else raw_df.attrs.get('diagnostics', {})

    # Static option catalogs (preserved verbatim for preset & sanitize invariants)
    grade_options = ["🔥 Top 40% (A+ / A / B)", "🏆 Top 20% (A+ & A)", "A+", "A", "B", "C", "D", "E"] + (["NR"] if is_master else [])
    sortino_options = ["All Sortino Ratios", "🔥 Positive (> 0.0)", "⚡ High Sortino (≥ 1.5)", "💎 Elite Sortino (≥ 3.0)"]
    bb_options = [
        "All Leaders",
        "👑 Beach Ball Active Only (Accumulating on Washouts)",
        "🏦 High 50D Up/Down Vol (≥ 1.25x Accumulation)",
        "⚡ Pocket Pivot Today (Vol > 10D Max Down Vol)",
        "🔇 Volume Dry-Up Today (VDU ≤ 55% Avg Vol)",
        "💎 Apex Accumulation (Beach Ball + U/D Vol ≥ 1.5x)",
    ]
    rai_horizon_options = ["20D Core", "15D Tactical", "40D Institutional"]
    role_options = ["All Group Roles", "👑 Alpha #1 & 👯 Sister #2 Only", "🏆 Top 3 Group Leaders", "🐺 Confirmed Pack Leaders"]
    rs_line_options = ["All RS Line States", "🔵 Blue Dot Only (RS High Before Price)", "🟢 All 52W RS Line Highs", "🔷 All 3M & 52W RS Line Highs"]
    stage_options = [
        "All Stages & Caps",
        "🌱 Fresh Stage 2 Only (0% to +60% vs 200 SMA)",
        "🛡️ Exclude Climax Extended (≤ +100% vs 200 SMA)",
        "🏛️ Institutional QoQ Accumulation (Inst Δ > 0% or A+/A)",
        "🚀 Agile Small & Mid-Caps Only (High Velocity Supply)",
        "💎 Large & Mid-Cap Institutional Leaders",
    ]

    _sanitize_widget_state(f"{state_prefix}_filter_choice", STRUCTURE_FILTER_OPTIONS, DEFAULT_STRUCTURE_OPTION)
    _sanitize_widget_state(f"{state_prefix}_sortino", sortino_options, "All Sortino Ratios")
    _sanitize_widget_state(f"{state_prefix}_bb", bb_options, "All Leaders")
    _sanitize_widget_state(f"{state_prefix}_grade_multi", grade_options, [], is_multi=True)
    _sanitize_widget_state(f"{state_prefix}_role_choice", role_options, "All Group Roles")
    _sanitize_widget_state(f"{state_prefix}_rsline_choice", rs_line_options, "All RS Line States")
    _sanitize_widget_state(f"{state_prefix}_funda_choice", CANSLIM_OPTIONS, "All Fundamentals")
    _sanitize_widget_state(f"{state_prefix}_vcp_choice", VCP_FILTER_OPTIONS, "All Price Action")
    _sanitize_widget_state(f"{state_prefix}_stage_choice", stage_options, "All Stages & Caps")
    if is_master:
        _sanitize_widget_state(f"{state_prefix}_rai_horizon", rai_horizon_options, "20D Core")

    cur_rai_horizon = st.session_state.get(f"{state_prefix}_rai_horizon", "20D Core") if is_master else "20D Core"
    h_days = 20 if "20D" in cur_rai_horizon else (15 if "15D" in cur_rai_horizon else 40)
    rai_core_thresh = 66.7 if h_days == 15 else (62.5 if h_days == 20 else 55.0)
    rai_col = f"RS_Days_{h_days}D_Pct"
    rai_label_col = f"RS_Days_{h_days}D_Label"

    adtv_unit = "$M" if is_us else "₹ Cr"
    default_adtv = (5.0 if is_us else (2.0 if univ_scope == "ALL" else 0.0)) if is_master else 0.0
    pre_min_adtv = float(st.session_state.get(f"{state_prefix}_adtv_gate", default_adtv)) if is_master else 0.0

    # Pre-slice ADTV gate & Cohort Cap (strictly client-side, never re-runs engine!)
    if is_master:
        adtv_gated = raw_df[raw_df['ADTV_20D'] >= pre_min_adtv].copy()
        adtv_gated['Rank'] = range(1, len(adtv_gated) + 1)
        df_lead = adtv_gated.head(cohort_cap).copy()
    else:
        df_lead = raw_df.head(cohort_cap).copy()

    # 1. Compact 6-Card KPI Summary Strip
    n_tot = len(df_lead)
    n_defending = int(df_lead['Holding_21EMA_Zone'].sum()) if n_tot > 0 else 0
    pct_defending = (n_defending / n_tot * 100.0) if n_tot > 0 else 0.0
    n50 = int(df_lead['Holding_50SMA_Zone'].sum()) if (n_tot > 0 and 'Holding_50SMA_Zone' in df_lead.columns) else 0
    n65 = int(df_lead['Holding_65EMA_Zone'].sum()) if (n_tot > 0 and 'Holding_65EMA_Zone' in df_lead.columns) else 0
    pct50 = (n50 / n_tot * 100.0) if n_tot > 0 else 0.0
    pct65 = (n65 / n_tot * 100.0) if n_tot > 0 else 0.0
    n_vcp = int(df_lead['Is_VCP_Tight'].sum()) if (n_tot > 0 and 'Is_VCP_Tight' in df_lead.columns) else 0
    n_id_nr7 = int(df_lead['Is_ID_NR7'].sum()) if (n_tot > 0 and 'Is_ID_NR7' in df_lead.columns) else 0
    n_bear_coil = int(df_lead['Is_Bear_Coil'].sum()) if (n_tot > 0 and 'Is_Bear_Coil' in df_lead.columns) else 0
    n_sortino = int((df_lead['Sortino_3M'] >= 1.5).sum()) if n_tot > 0 else 0
    n_bb = int(df_lead['Has_Beach_Ball'].sum()) if (n_tot > 0 and 'Has_Beach_Ball' in df_lead.columns) else 0
    n_top_groups = int(df_lead['Is_Top_20_Group'].sum()) if (n_tot > 0 and 'Is_Top_20_Group' in df_lead.columns) else 0
    n_rs_high = int(df_lead['Is_RS_52W_High'].sum()) if (n_tot > 0 and 'Is_RS_52W_High' in df_lead.columns) else 0
    n_blue_dot = int(df_lead['Is_RS_Blue_Dot'].sum()) if (n_tot > 0 and 'Is_RS_Blue_Dot' in df_lead.columns) else 0
    n_tml = int(df_lead['Is_TML_Confluence'].sum()) if (n_tot > 0 and 'Is_TML_Confluence' in df_lead.columns) else 0
    n_rai_core = int((df_lead[rai_col] >= rai_core_thresh).sum()) if (is_master and n_tot > 0 and rai_col in df_lead.columns) else 0

    _render_kpi_summary_strip(
        is_master=is_master,
        n_tot=n_tot,
        n_defending=n_defending,
        pct_defending=pct_defending,
        pct50=pct50,
        pct65=pct65,
        n_vcp=n_vcp,
        n_id_nr7=n_id_nr7,
        n_bear_coil=n_bear_coil,
        n_rs_high=n_rs_high,
        n_blue_dot=n_blue_dot,
        n_sortino=n_sortino,
        n_top_groups=n_top_groups,
        n_tml=n_tml,
        n_bb=n_bb,
        n_rai_core=n_rai_core,
        h_days=h_days,
        rai_core_thresh=rai_core_thresh,
    )

    if isinstance(diag, dict) and int(diag.get('bench_stale_days', 0) or 0) >= 1:
        st.caption(
            f"⚠️ Benchmark index (`{diag.get('bench_last_dt', '')}`) lagged matrix close (`{diag.get('matrix_last_dt', '')}`); "
            "same-day synthetic Blue Dot promotions suppressed."
        )

    # 2. Primary Fast Bar — Row A: Live-Count Presets + Search
    preset_counts = count_preset_matches(df_lead)
    _pre_base = _baseline_filter_defaults(is_master=is_master)
    _pre_snap = {k: st.session_state.get(f"{state_prefix}_{k}", v) for k, v in _pre_base.items()}
    pre_active_preset = _detect_active_preset(_pre_snap, is_master=is_master)

    tb_c1, tb_c2, tb_c3, tb_c4, tb_info, tb_search = st.columns([1.0, 1.0, 1.05, 0.72, 0.45, 1.25])
    for col_w, pid in ((tb_c1, "bull"), (tb_c2, "sideways"), (tb_c3, "bear")):
        spec = PRESET_SPECS[pid]
        with col_w:
            st.button(
                f"{spec['label']} ({preset_counts.get(pid, 0)})",
                key=f"{state_prefix}_btn_preset_{pid}",
                type="primary" if pre_active_preset == pid else "secondary",
                on_click=_apply_preset_cb,
                kwargs={"prefix": state_prefix, "preset_id": pid, "is_master": is_master},
                use_container_width=True,
                help=spec["help"],
            )
    with tb_c4:
        st.button(
            "↺ Reset",
            key=f"{state_prefix}_btn_reset",
            on_click=_reset_radar_filters_cb,
            kwargs={"prefix": state_prefix, "is_master": is_master},
            use_container_width=True,
            help="Restore default 21-EMA bull momentum scan (preserves ADTV liquidity floor).",
        )
    with tb_info:
        if hasattr(st, "popover"):
            with st.popover("ℹ️", help="View exact quantitative filter rules behind each regime preset"):
                st.markdown(build_preset_rationale_html(preset_counts), unsafe_allow_html=True)
        else:
            with st.expander("ℹ️", expanded=False):
                st.markdown(build_preset_rationale_html(preset_counts), unsafe_allow_html=True)
    with tb_search:
        search_q = st.text_input(
            "Search:",
            key=f"{state_prefix}_search",
            placeholder=search_ph,
            label_visibility="collapsed",
        )

    # Primary Fast Bar — Row B: High-Frequency Structure, Action & Sort
    sort_options = _build_sort_options(h_days, adtv_col_label)
    _sanitize_widget_state(f"{state_prefix}_sort", sort_options, "Smooth RS / ATR (High to Low)")

    pb_c1, pb_c2, pb_c3 = st.columns([1.3, 1.15, 1.15])
    with pb_c1:
        filter_choice = st.selectbox(
            "Structure Support Zone:",
            STRUCTURE_FILTER_OPTIONS,
            index=0,
            key=f"{state_prefix}_filter_choice",
        )
    with pb_c2:
        vcp_choice = st.selectbox(
            "VCP Tightness & Bar:",
            VCP_FILTER_OPTIONS,
            index=0,
            key=f"{state_prefix}_vcp_choice",
        )
    with pb_c3:
        sort_by = st.selectbox(
            "Sort By:",
            sort_options,
            index=0,
            key=f"{state_prefix}_sort",
        )

    # 3. Pre-compute Drawer Active Counts for Expander Header (using persisted/sanitized session state)
    adv_snapshot = {
        "filter_choice": filter_choice,
        "vcp_choice": vcp_choice,
        "search": search_q,
        "min_rs": st.session_state.get(f"{state_prefix}_min_rs", 0),
        "max_52w_slider": st.session_state.get(f"{state_prefix}_max_52w_slider", 100.0),
        "min_rai": st.session_state.get(f"{state_prefix}_rai_slider_{h_days}", 0.0),
        "min_adr": st.session_state.get(f"{state_prefix}_min_adr", 0.0),
        "grade_multi": st.session_state.get(f"{state_prefix}_grade_multi", []),
        "role_choice": st.session_state.get(f"{state_prefix}_role_choice", "All Group Roles"),
        "rsline_choice": st.session_state.get(f"{state_prefix}_rsline_choice", "All RS Line States"),
        "bb": st.session_state.get(f"{state_prefix}_bb", "All Leaders"),
        "funda_choice": st.session_state.get(f"{state_prefix}_funda_choice", "All Fundamentals"),
        "stage_choice": st.session_state.get(f"{state_prefix}_stage_choice", "All Stages & Caps"),
        "sortino": st.session_state.get(f"{state_prefix}_sortino", "All Sortino Ratios"),
    }
    snap_summary = summarize_active_filters(
        adv_snapshot,
        h_days=h_days,
        is_master=is_master,
        min_adtv=pre_min_adtv,
        curr_sym=curr_sym,
        adtv_unit=adtv_unit,
    )
    cats = snap_summary["cat_counts"]

    # 4. Collapsible Advanced Filters Drawer (preserves exact session_state widget keys)
    adv_title = f"🎛️ Advanced Quantitative & Fundamental Filters ({snap_summary['adv_active_count']} active)"
    with st.expander(adv_title, expanded=False):
        st.caption(f"**Momentum & Liquidity ({cats['momentum']} active)**")
        m_cols = st.columns(5 if is_master else 3)
        with m_cols[0]:
            min_rs_slider = st.slider("Min RS Rating (1-99):", 0, 99, 0, step=1, key=f"{state_prefix}_min_rs")
        with m_cols[1]:
            max_off_52w_slider = st.slider("Max Off 52W High %:", 5.0, 100.0, 100.0, step=2.5, format="-%.1f%%", key=f"{state_prefix}_max_52w_slider")
        with m_cols[2]:
            min_adr_slider = st.slider("Min 20D ADR %:", 0.0, 8.0, 0.0, step=0.5, format="%.1f%%", key=f"{state_prefix}_min_adr")
        if is_master:
            with m_cols[3]:
                rai_horizon = st.selectbox("Rai Horizon:", rai_horizon_options, index=0, key=f"{state_prefix}_rai_horizon")
            with m_cols[4]:
                max_adtv_scale = 200.0 if is_us else 100.0
                min_adtv_slider = st.slider(
                    f"Min 20D ADTV ({adtv_unit}):",
                    min_value=0.0,
                    max_value=max_adtv_scale,
                    value=default_adtv,
                    step=0.5 if not is_us else 1.0,
                    format=f"%.1f {adtv_unit.replace('₹ ', '')}",
                    key=f"{state_prefix}_adtv_gate",
                    help="Client-side Pre-Ranking Liquidity Gate applied before slicing Cohort Cap.",
                )
        else:
            rai_horizon = "20D Core"
            min_adtv_slider = 0.0

        min_rai_slider = st.slider(
            f"Min {h_days}D RS Days % (Ameet Rai Persistence):",
            min_value=0.0,
            max_value=95.0,
            value=0.0,
            step=2.5,
            format="%.1f%%",
            key=f"{state_prefix}_rai_slider_{h_days}",
        )

        st.caption(f"**Group & Sponsorship ({cats['group']} active)**")
        g_c1, g_c2, g_c3, g_c4 = st.columns(4)
        with g_c1:
            selected_grades = st.multiselect(
                "Group Grade:",
                options=grade_options,
                default=[],
                key=f"{state_prefix}_grade_multi",
                placeholder="All Grades (e.g. A+, B)",
            )
        with g_c2:
            role_choice = st.selectbox("Livermore Group Role:", role_options, index=0, key=f"{state_prefix}_role_choice")
        with g_c3:
            rs_line_choice = st.selectbox("RS Line Signal (Blue Dot):", rs_line_options, index=0, key=f"{state_prefix}_rsline_choice")
        with g_c4:
            bb_choice = st.selectbox("Beach Ball & Volume:", bb_options, index=0, key=f"{state_prefix}_bb")

        st.caption(f"**CANSLIM & Stage ({cats['canslim']} active)**")
        q_c1, q_c2, q_c3 = st.columns(3)
        with q_c1:
            funda_choice = st.selectbox("CANSLIM Confluence:", CANSLIM_OPTIONS, index=0, key=f"{state_prefix}_funda_choice")
        with q_c2:
            stage_choice = st.selectbox("Stage & Cap Tier:", stage_options, index=0, key=f"{state_prefix}_stage_choice")
        with q_c3:
            sortino_choice = st.selectbox("Sortino 3M Ratio:", sortino_options, index=0, key=f"{state_prefix}_sortino")

    live_state = {
        "filter_choice": filter_choice,
        "sortino": sortino_choice,
        "bb": bb_choice,
        "grade_multi": selected_grades,
        "rai_slider_20": float(st.session_state.get(f"{state_prefix}_rai_slider_20", 0.0)) if h_days != 20 else float(min_rai_slider),
        "min_rai": float(min_rai_slider),
        "min_rs": int(min_rs_slider),
        "max_52w_slider": float(max_off_52w_slider),
        "role_choice": role_choice,
        "rsline_choice": rs_line_choice,
        "funda_choice": funda_choice,
        "vcp_choice": vcp_choice,
        "min_adr": float(min_adr_slider),
        "stage_choice": stage_choice,
        "sort": sort_by,
        "search": search_q,
    }
    if is_master:
        live_state["rai_horizon"] = rai_horizon
    active_preset = _detect_active_preset(live_state, is_master=is_master)

    live_summary = summarize_active_filters(
        live_state,
        h_days=h_days,
        is_master=is_master,
        min_adtv=min_adtv_slider,
        curr_sym=curr_sym,
        adtv_unit=adtv_unit,
    )
    if active_preset in PRESET_SPECS or live_summary["chips"]:
        chip_list = []
        if active_preset in PRESET_SPECS:
            chip_list.append(f"Preset: {PRESET_SPECS[active_preset]['label']}")
        chip_list.extend(live_summary["chips"])
        chips_html = "".join([
            f'<span style="background:rgba(56,189,248,0.14); color:#7dd3fc; border:1px solid rgba(56,189,248,0.32); '
            f'padding:2px 9px; border-radius:999px; font-size:0.71rem; font-weight:700; font-family:\'JetBrains Mono\',monospace;">{c}</span>'
            for c in chip_list
        ])
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap; margin:4px 0 8px 0;">'
            f'<span style="color:#94a3b8; font-size:0.72rem; font-weight:700;">Active:</span>{chips_html}</div>',
            unsafe_allow_html=True,
        )

    if is_master and cohort_cap >= 1000 and filter_choice == ALL_LEADERS_OPTION and min_rs_slider == 0:
        st.info("ℹ️ Scanning the full cohort without a structure or Min RS floor may include lower-RS pullbacks. Raise Min RS Rating (e.g. ≥ 80) or select 50 SMA / 65 EMA for tighter focus.")

    # Apply Filtering & Sorting on the full DataFrame before View Lens projection
    filtered_df = apply_leader_radar_filters(
        df_lead,
        filter_choice=filter_choice,
        sortino_choice=sortino_choice,
        bb_choice=bb_choice,
        selected_grades=selected_grades,
        min_rai=min_rai_slider,
        rai_col=rai_col,
        min_rs=min_rs_slider,
        max_off_52w=max_off_52w_slider,
        role_choice=role_choice,
        rs_line_choice=rs_line_choice,
        funda_choice=funda_choice,
        vcp_choice=vcp_choice,
        min_adr=min_adr_slider,
        stage_choice=stage_choice,
        sort_by=sort_by,
        search_q=search_q,
    )

    if filtered_df.empty:
        hint = (
            PRESET_SPECS[active_preset]["empty_hint"]
            if active_preset in PRESET_SPECS
            else "💡 0 leaders matched active filters. Try widening Structure Filter, Min RS, or Cohort Cap."
        )
        st.info(hint)

    # 5. Toolbar Row: Column View Lenses | Layout Toggle | Match Count | Inline TradingView Export
    _sanitize_widget_state(f"{state_prefix}_col_lens", COLUMN_LENS_OPTIONS, LENS_EXECUTION)
    _layout_options = ["📋 Table", "🪟 Split View", "🖱️ Hover Deck", "🎯 Outlier Scatter"]
    _sanitize_widget_state(f"{state_prefix}_layout_mode", _layout_options, "📋 Table")
    tv_symbols = format_leader_tv_symbols(filtered_df, is_us=is_us)
    tv_copy = ", ".join(tv_symbols)
    tv_digest = hashlib.md5(tv_copy.encode("utf-8")).hexdigest()[:10]
    exp_suffix = f" • {univ_badge_txt} • ADTV ≥ {curr_sym}{min_adtv_slider:.1f} {adtv_unit.replace('₹ ', '')}" if is_master else ""

    tbar_l, tbar_view, tbar_m, tbar_r = st.columns([2.15, 1.55, 0.60, 0.90])
    with tbar_l:
        lens_choice = st.radio(
            "View Lens:",
            COLUMN_LENS_OPTIONS,
            index=0,
            horizontal=True,
            key=f"{state_prefix}_col_lens",
            label_visibility="collapsed",
        )
    with tbar_view:
        layout_mode = st.radio(
            "Layout:",
            _layout_options,
            index=0,
            horizontal=True,
            key=f"{state_prefix}_layout_mode",
            label_visibility="collapsed",
        )
    with tbar_m:
        st.markdown(
            f"<div style='padding-top:6px; text-align:right; font-size:0.80rem; font-weight:800; color:#38bdf8;'>{len(filtered_df)} / {len(df_lead)}</div>",
            unsafe_allow_html=True,
        )
    with tbar_r:
        from components import render_tv_export_popover
        render_tv_export_popover(
            tv_symbols,
            title=f"TradingView Watchlist{exp_suffix}",
            key=f"{state_prefix}_tv_{len(filtered_df)}_{tv_digest}",
        )

    disp_cols = {
        'Rank': 'RS Rank',
        'Ticker': 'Ticker',
        'TradingView_URL': 'Chart',
        'Price': price_col_label,
        'RS_Rating': 'RS Rating',
        'ADR_20D': '20D ADR %',
        'Dist_21EMA': 'vs 21 EMA %',
        'Dist_50SMA': 'vs 50 SMA %',
        'Dist_65EMA': 'vs 65 EMA %',
        'Dist_50SMA_ATRx': '50D ATRx',
        'MA_Status': 'Structure Status',
        'Setup_Trigger_State': 'Trigger State',
        'VCP_Action_Badge': 'VCP & Bar Action',
        'Dist_52W_High': 'vs 52W High %',
        'RS_Line_Status': 'RS Line Signal',
        'Industry_Badge': 'Industry & Grade',
        'Group_Role': 'Group Role',
        'Funda_Badge': 'CANSLIM Confluence',
        'Cap_Stage_Badge': 'Stage & Cap Tier',
        rai_label_col: f'Rai {h_days}D RS %',
        'UDVR_50D': '50D U/D Vol',
        'Apex_Status': 'Beach Ball & Vol',
        'ADTV_20D': adtv_col_label,
        'Smooth_RS': 'Smooth RS / ATR',
        'Sortino_3M': 'Sortino 3M',
    }
    is_split = "Split" in layout_mode
    is_hover = "Hover" in layout_mode
    is_scatter = "Scatter" in layout_mode
    if is_split:
        active_disp_cols = project_split_pane_columns(disp_cols, price_col_label=price_col_label)
    else:
        active_disp_cols = project_columns_for_lens(
            disp_cols,
            lens_choice,
            h_days=h_days,
            price_col_label=price_col_label,
            adtv_col_label=adtv_col_label,
        )
    show_df = filtered_df[[c for c in active_disp_cols.keys() if c in filtered_df.columns]].rename(columns=active_disp_cols)

    all_column_config = {
        "RS Rank": st.column_config.NumberColumn("RS Rank", format="%d", pinned=True, width="small"),
        "Ticker": st.column_config.TextColumn("Ticker", pinned=True, width="small"),
        "Chart": st.column_config.LinkColumn("Chart", display_text="📈 TV", pinned=True, width="small"),
        price_col_label: st.column_config.NumberColumn(price_col_label, format=price_fmt),
        "RS Rating": st.column_config.ProgressColumn("RS Rating", min_value=0, max_value=99, format="%.0f", help="0-99 Cross-sectional momentum percentile."),
        "20D ADR %": st.column_config.NumberColumn("20D ADR %", format="%.2f%%"),
        "vs 21 EMA %": st.column_config.NumberColumn(format="%+.2f%%"),
        "vs 50 SMA %": st.column_config.NumberColumn(format="%+.2f%%"),
        "vs 65 EMA %": st.column_config.NumberColumn("vs 65 EMA %", format="%+.2f%%", help="Distance vs Ross Haber's 65-day EMA (13-week institutional floor)."),
        "50D ATRx": st.column_config.NumberColumn("50D ATRx", format="%+.2fx", help="Distance from 50 SMA expressed in multiples of 20D ADR/ATR."),
        "Structure Status": st.column_config.TextColumn("Structure Status"),
        "Trigger State": st.column_config.TextColumn("Trigger State", help="Separates Passive Support Watches vs Tight Coils vs Right-Side Reclaim/Pocket Pivot Triggers."),
        "VCP & Bar Action": st.column_config.TextColumn("VCP & Bar Action"),
        "vs 52W High %": st.column_config.NumberColumn(format="%+.2f%%"),
        "RS Line Signal": st.column_config.TextColumn("RS Line Signal"),
        "Industry & Grade": st.column_config.TextColumn("Industry & Grade"),
        "Group Role": st.column_config.TextColumn("Group Role"),
        "CANSLIM Confluence": st.column_config.TextColumn("CANSLIM Confluence"),
        "Stage & Cap Tier": st.column_config.TextColumn("Stage & Cap Tier"),
        f"Rai {h_days}D RS %": st.column_config.TextColumn(f"Rai {h_days}D RS %"),
        "50D U/D Vol": st.column_config.NumberColumn("50D U/D Vol", format="%.2fx"),
        "Beach Ball & Vol": st.column_config.TextColumn("Beach Ball & Vol"),
        adtv_col_label: st.column_config.NumberColumn(adtv_col_label, format=adtv_fmt),
        "Smooth RS / ATR": st.column_config.NumberColumn(format="%.2f"),
        "Sortino 3M": st.column_config.NumberColumn(format="%.2f"),
    }

    df_kwargs = {
        "use_container_width": True,
        "hide_index": True,
        "key": f"{state_prefix}_radar_table",
        "column_config": {k: v for k, v in all_column_config.items() if k in show_df.columns},
    }
    if is_split:
        df_kwargs["height"] = 560
    elif table_height is not None:
        df_kwargs["height"] = table_height

    sel_tk_key = f"{state_prefix}_sel_ticker"
    last_click_key = f"{state_prefix}_last_click_rows"
    tickers_list = filtered_df["Ticker"].astype(str).tolist() if not filtered_df.empty and "Ticker" in filtered_df.columns else []

    if is_hover and not filtered_df.empty:
        _hd_cap = 150
        _hd_payload = build_hover_deck_payload(
            filtered_df,
            univ_scope=univ_scope,
            curr_sym=curr_sym,
            max_rows=_hd_cap,
            bars=90,
        )
        if len(filtered_df) > _hd_cap:
            st.caption(f"🖱️ Hover Deck shows the top {_hd_cap} sorted rows ({len(filtered_df)} matched) for instant 0ms switching.")
        _hd_html = render_hover_deck_html(
            _hd_payload,
            initial_ticker=st.session_state.get(sel_tk_key),
            curr_sym=curr_sym,
            height_px=575,
        )
        components.html(_hd_html, height=583, scrolling=False)
    elif is_scatter and not filtered_df.empty:
        _sc_cap = 250
        _sc_fig = build_outlier_scatter_fig(
            filtered_df,
            curr_sym=curr_sym,
            adtv_col_label=adtv_col_label,
            max_points=_sc_cap,
            height=365,
        )
        _clicked_pts = []
        try:
            _sc_event = st.plotly_chart(
                _sc_fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode=("points",),
                key=f"{state_prefix}_outlier_scatter",
            )
            _clicked_pts = list(getattr(getattr(_sc_event, "selection", None), "points", []) or [])
        except TypeError:
            st.plotly_chart(_sc_fig, use_container_width=True, key=f"{state_prefix}_outlier_scatter")
        _sc_tickers = tickers_list[:_sc_cap]
        _last_pt_key = f"{state_prefix}_last_scatter_pt_idx"
        _cur_pt_idx = _clicked_pts[0].get("point_index") if _clicked_pts else None
        if not _clicked_pts:
            st.session_state.pop(_last_pt_key, None)
        elif isinstance(_cur_pt_idx, int) and 0 <= _cur_pt_idx < len(_sc_tickers):
            if st.session_state.get(_last_pt_key) != _cur_pt_idx:
                st.session_state[_last_pt_key] = _cur_pt_idx
                st.session_state[sel_tk_key] = _sc_tickers[_cur_pt_idx]
                st.session_state[f"{state_prefix}_scatter_pick"] = _sc_tickers[_cur_pt_idx]
        st.markdown(
            "<div style='border-top:1px solid rgba(56,189,248,0.18); margin:4px 0 8px 0;'></div>",
            unsafe_allow_html=True,
        )
        _cur_sc_tk = st.session_state.get(sel_tk_key)
        _def_sc_idx = _sc_tickers.index(_cur_sc_tk) if _cur_sc_tk in _sc_tickers else 0
        _pick_c1, _pick_c2 = st.columns([1.4, 3.6])
        with _pick_c1:
            _pick_sc_tk = st.selectbox(
                "Inspect Outlier Symbol:",
                _sc_tickers,
                index=_def_sc_idx,
                key=f"{state_prefix}_scatter_pick",
            )
            st.session_state[sel_tk_key] = _pick_sc_tk
        with _pick_c2:
            st.markdown(
                f"<div style='padding-top:26px; color:#94a3b8; font-size:0.78rem;'>"
                f"Click any scatter bubble above or choose from top <b style='color:#e2e8f0;'>{len(_sc_tickers)}</b> leaders "
                f"to inspect its full-width <b style='color:#38bdf8;'>KushTracker v3.0</b> Chart &amp; 16-Row Cockpit below."
                f"</div>",
                unsafe_allow_html=True,
            )
        _render_quick_chart_inspector(
            filtered_df.iloc[_sc_tickers.index(_pick_sc_tk)],
            is_us=is_us,
            curr_sym=curr_sym,
            univ_scope=univ_scope,
            compact_stats_below=False,
            state_prefix=f"{state_prefix}_scatter",
            benchmark_symbol=benchmark_symbol,
            enable_kt_v3=True,
        )
    elif is_split and not filtered_df.empty:
        left_col, right_col = st.columns([0.40, 0.60])
        with left_col:
            try:
                table_event = st.dataframe(show_df, selection_mode="single-row", on_select="rerun", **df_kwargs)
            except TypeError:
                table_event = st.dataframe(show_df, **df_kwargs)
        sel_rows = list(getattr(getattr(table_event, "selection", None), "rows", []) or []) if table_event is not None else []
        with right_col:
            nav_c1, nav_c2, nav_c3, nav_c4 = st.columns([0.85, 1.45, 0.85, 1.1])
            step_delta = 0
            with nav_c1:
                if st.button("◀ Prev", key=f"{state_prefix}_split_prev", use_container_width=True):
                    step_delta = -1
            with nav_c3:
                if st.button("Next ▶", key=f"{state_prefix}_split_next", use_container_width=True):
                    step_delta = 1
            sel_idx, active_tk, recorded_click = resolve_split_selection(
                tickers_list,
                clicked_rows=sel_rows,
                last_clicked_rows=st.session_state.get(last_click_key, []),
                stored_ticker=st.session_state.get(sel_tk_key),
                step_delta=step_delta,
            )
            st.session_state[sel_tk_key] = active_tk
            st.session_state[last_click_key] = recorded_click
            with nav_c2:
                st.markdown(
                    f"<div style='text-align:center; padding-top:6px; font-weight:800; color:#f8fafc; font-size:0.86rem;'>"
                    f"{active_tk} <span style='color:#64748b; font-size:0.76rem;'>({sel_idx + 1} / {len(tickers_list)})</span></div>",
                    unsafe_allow_html=True,
                )
            with nav_c4:
                if st.button("⭐ Focus", key=f"{state_prefix}_split_focus_{active_tk}", use_container_width=True):
                    from database import add_to_focus_list
                    _mkt_code = "US" if is_us else "IN"
                    if add_to_focus_list(str(active_tk).upper().strip(), _mkt_code):
                        st.toast(f"⭐ Added {active_tk} to {_mkt_code} Focus List!")
            _render_quick_chart_inspector(
                filtered_df.iloc[sel_idx],
                is_us=is_us,
                curr_sym=curr_sym,
                univ_scope=univ_scope,
                compact_stats_below=True,
            )
    else:
        try:
            table_event = st.dataframe(show_df, selection_mode="single-row", on_select="rerun", **df_kwargs)
        except TypeError:
            table_event = st.dataframe(show_df, **df_kwargs)

        sel_rows = []
        if table_event is not None and hasattr(table_event, "selection") and table_event.selection:
            sel_rows = list(getattr(table_event.selection, "rows", []) or [])
        if sel_rows and 0 <= sel_rows[0] < len(filtered_df):
            if tickers_list:
                st.session_state[sel_tk_key] = tickers_list[sel_rows[0]]
                st.session_state[last_click_key] = list(sel_rows)
            _render_quick_chart_inspector(
                filtered_df.iloc[sel_rows[0]],
                is_us=is_us,
                curr_sym=curr_sym,
                univ_scope=univ_scope,
            )
        elif not filtered_df.empty:
            st.caption("💡 Tip: Click any row in the table above (or toggle **🪟 Split View**) to preview its 90-session chart (21 EMA / 50 SMA / 65 EMA) & trigger summary inline.")
