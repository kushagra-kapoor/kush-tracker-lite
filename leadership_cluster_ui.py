"""
Institutional Cluster Leadership & Thrusting Industries UI Renderers
=====================================================================
Thin, purely presentational Streamlit renderers + testable IGM jump-handoff
helpers backed by `leadership_cluster_engine.py`.
"""

from __future__ import annotations

import hashlib
import html
import os
import time
from typing import Any

import pandas as pd
import streamlit as st

from leadership_cluster_engine import (
    compute_cluster_leadership_board,
    compute_stockbee_thrusting_industries,
)


def _strip_html_indent(raw: str) -> str:
    """Strip leading whitespace per line so Markdown never turns HTML into code blocks."""
    return "\n".join(line.lstrip() for line in str(raw or "").splitlines() if line.strip())


def _esc(val: Any) -> str:
    return html.escape(str(val if val is not None else ""), quote=True)


def _resolve_igm_target_page(is_us: bool) -> str:
    """Resolve views/ vs pages/ Industry Group Matrix path dynamically for KTL vs KT."""
    fname = "industry_group_matrix_us.py" if is_us else "industry_group_matrix.py"
    views_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "views")
    folder = "views" if os.path.isdir(views_dir) else "pages"
    return f"{folder}/{fname}"


def _grp_hash(name: str) -> str:
    return hashlib.md5(str(name or "").encode("utf-8")).hexdigest()[:8]


def _inject_clb_tooltip_css() -> None:
    """Inject lightweight CSS for instant on-hover explanation pills (zero JS)."""
    st.markdown(
        _strip_html_indent(
            """
            <style>
            .kt-tip-bar {
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 6px;
                margin: 4px 0 8px 0;
                padding: 6px 10px;
                background: rgba(15, 23, 42, 0.55);
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 8px;
                font-size: 11px;
                color: #94a3b8;
            }
            .kt-tip {
                position: relative;
                display: inline-flex;
                align-items: center;
                gap: 4px;
                padding: 2px 8px;
                border-radius: 999px;
                background: rgba(30, 41, 59, 0.85);
                border: 1px solid rgba(148, 163, 184, 0.25);
                color: #cbd5e1;
                font-size: 10.5px;
                font-weight: 600;
                cursor: help;
                transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease;
            }
            .kt-tip:hover {
                border-color: #38bdf8;
                color: #f8fafc;
                background: rgba(14, 116, 144, 0.35);
            }
            .kt-tip-active {
                border-color: rgba(56, 189, 248, 0.65);
                color: #e0f2fe;
                background: rgba(14, 116, 144, 0.28);
            }
            .kt-tip .kt-tip-box {
                opacity: 0;
                visibility: hidden;
                pointer-events: none;
                position: absolute;
                top: calc(100% + 7px);
                left: 0;
                z-index: 9999;
                width: min(320px, 78vw);
                padding: 9px 11px;
                border-radius: 9px;
                background: #0b1220;
                border: 1px solid rgba(56, 189, 248, 0.5);
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.65);
                color: #e2e8f0;
                font-size: 11px;
                font-weight: 400;
                line-height: 1.45;
                text-align: left;
                white-space: normal;
                transition: opacity 0.12s ease;
            }
            .kt-tip-r .kt-tip-box {
                left: auto;
                right: 0;
            }
            .kt-tip:hover .kt-tip-box,
            .kt-tip:focus-within .kt-tip-box {
                opacity: 1;
                visibility: visible;
            }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


def _hover_pill(label: str, title: str, body: str, active: bool = False, align_right: bool = False) -> str:
    cls = "kt-tip"
    if active:
        cls += " kt-tip-active"
    if align_right:
        cls += " kt-tip-r"
    return (
        f'<span class="{cls}" tabindex="0">'
        f'{_esc(label)}'
        f'<span class="kt-tip-box">'
        f'<div style="font-weight:700;color:#38bdf8;margin-bottom:3px;">{_esc(title)}</div>'
        f'<div>{_esc(body)}</div>'
        f'</span></span>'
    )


# -----------------------------------------------------------------------------
# TESTABLE IGM DRILLDOWN JUMP HANDOFF HELPERS
# -----------------------------------------------------------------------------
def build_igm_jump_payload(market: str, group_name: str, now_ts: float | None = None) -> dict:
    mkt = "US" if str(market or "IN").strip().upper().startswith("US") else "IN"
    return {
        "market": mkt,
        "group": str(group_name or "").strip(),
        "taxonomy": "canonical",
        "ts": float(time.time() if now_ts is None else now_ts),
    }


def resolve_igm_jump_payload(
    session_state: dict,
    u_pfx: str,
    now_ts: float | None = None,
    ttl_sec: float = 60.0,
) -> dict | None:
    """
    Consume & validate a pending IGM jump payload before Streamlit renders sidebar/top widgets.
    Returns payload only if valid and within TTL (<= 60s), while forcing canonical taxonomy
    and target market into session_state.
    """
    cur_ts = float(time.time() if now_ts is None else now_ts)
    pfx = "us" if str(u_pfx or "").lower().startswith("us") else "in"
    expected_mkt = "US" if pfx == "us" else "IN"

    jump_key = f"kt_igm_jump_{pfx}"
    payload = session_state.pop(jump_key, None) if isinstance(session_state.get(jump_key), dict) else None
    if payload is None and isinstance(session_state.get("kt_igm_jump"), dict):
        cand = session_state["kt_igm_jump"]
        cand_mkt = "US" if str(cand.get("market", "")).upper().startswith("US") else "IN"
        if cand_mkt == expected_mkt:
            payload = session_state.pop("kt_igm_jump", None)

    if not isinstance(payload, dict):
        return None
    ts_val = float(payload.get("ts", 0.0) or 0.0)
    if (cur_ts - ts_val) > float(ttl_sec):
        return None

    target_mkt = "US" if str(payload.get("market", expected_mkt)).upper().startswith("US") else "IN"
    session_state[f"igm_market_universe_select_{pfx}"] = (
        "🇺🇸 US (S&P 500 / Nasdaq)" if target_mkt == "US" else "🇮🇳 India (NSE/BSE)"
    )
    session_state[f"igm_tax_choice_{pfx}"] = "🏛️ Canonical 145 Sub-Industries"
    jump_grp = str(payload.get("group") or "").strip()
    if jump_grp:
        session_state[f"kt_igm_sticky_pin_{pfx}"] = {"group": jump_grp, "sig": None, "ts": cur_ts}
    return payload


def apply_igm_jump_to_filtered_df(
    filtered_df: pd.DataFrame,
    df_matrix: pd.DataFrame,
    jump_payload: dict | None,
    df_niche: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str | None, str | None]:
    """
    Pin the jumped Industry_Group into `filtered_df` if filters/Head(N) sliced it out,
    or emit a Niche notice when it belongs in the Niche (<4) tab.
    """
    if not isinstance(jump_payload, dict):
        return filtered_df, None, None
    jump_grp = str(jump_payload.get("group") or "").strip()
    if not jump_grp:
        return filtered_df, None, None

    if isinstance(df_matrix, pd.DataFrame) and not df_matrix.empty and "Industry_Group" in df_matrix.columns:
        in_matrix = df_matrix[df_matrix["Industry_Group"] == jump_grp]
        if not in_matrix.empty:
            has_col = isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty and "Industry_Group" in filtered_df.columns
            already_in = bool(has_col and (filtered_df["Industry_Group"] == jump_grp).any())
            if not already_in:
                out_df = pd.concat([filtered_df, in_matrix.head(1)], ignore_index=True)
                return out_df, jump_grp, f"🎯 Focused: **{jump_grp}** (pinned despite active filters)"
            return filtered_df, jump_grp, None

    if isinstance(df_niche, pd.DataFrame) and not df_niche.empty and "Industry_Group" in df_niche.columns:
        if (df_niche["Industry_Group"] == jump_grp).any():
            return (
                filtered_df,
                None,
                f"🔬 **{jump_grp}** is a Niche group (< 4 constituents) — view it in the **🔬 Niche & Excluded Groups (<4)** tab below.",
            )

    return filtered_df, None, None


def apply_igm_jump_with_sticky(
    session_state: dict,
    u_pfx: str,
    filtered_df: pd.DataFrame,
    df_matrix: pd.DataFrame,
    jump_payload: dict | None,
    filter_sig: str,
    df_niche: pd.DataFrame | None = None,
    now_ts: float | None = None,
    sticky_ttl_sec: float = 600.0,
) -> tuple[pd.DataFrame, str | None, str | None]:
    """
    Pin a jumped Industry_Group on Frame 1 AND retain it across same-filter Frame 2+
    interactions inside that drilled group so `Top 25 Groups` slicing does not snap back to #1.
    Clears automatically on filter change, user dropdown change, or TTL expiry.
    """
    pfx = "us" if str(u_pfx or "").lower().startswith("us") else "in"
    sticky_key = f"kt_igm_sticky_pin_{pfx}"
    sel_grp_key = f"igm_selected_group_{pfx}"
    cur_ts = float(time.time() if now_ts is None else now_ts)

    if isinstance(jump_payload, dict) and str(jump_payload.get("group") or "").strip():
        out_df, pinned_grp, notice = apply_igm_jump_to_filtered_df(
            filtered_df, df_matrix, jump_payload, df_niche=df_niche
        )
        if pinned_grp:
            session_state[sticky_key] = {"group": pinned_grp, "sig": filter_sig, "ts": cur_ts}
        else:
            session_state.pop(sticky_key, None)
        return out_df, pinned_grp, notice

    sticky = session_state.get(sticky_key)
    if not isinstance(sticky, dict):
        return filtered_df, None, None

    sticky_grp = str(sticky.get("group") or "").strip()
    sticky_ts = float(sticky.get("ts", 0.0) or 0.0)
    sticky_sig = sticky.get("sig")
    if (not sticky_grp) or ((cur_ts - sticky_ts) > float(sticky_ttl_sec)):
        session_state.pop(sticky_key, None)
        return filtered_df, None, None

    if sticky_sig is not None and sticky_sig != filter_sig:
        session_state.pop(sticky_key, None)
        return filtered_df, None, None

    cur_sel = session_state.get(sel_grp_key)
    if cur_sel and str(cur_sel) != sticky_grp:
        session_state.pop(sticky_key, None)
        return filtered_df, None, None

    out_df, pinned_grp, _ = apply_igm_jump_to_filtered_df(
        filtered_df, df_matrix, {"group": sticky_grp}, df_niche=df_niche
    )
    if pinned_grp:
        sticky["sig"] = filter_sig
        session_state[sticky_key] = sticky
        return out_df, pinned_grp, None

    session_state.pop(sticky_key, None)
    return filtered_df, None, None


# -----------------------------------------------------------------------------
# 1. STOCKBEE THRUSTING INDUSTRIES (MARKET BREADTH INDIA & US)
# -----------------------------------------------------------------------------
_THRUST_MODE_OPTIONS = [
    ("🚀 5D + Today", "5D_AND_TODAY"),
    ("⚡ Today ±4%", "TODAY_4PCT"),
    ("📈 +25% Qtr", "QTR_25PCT"),
    ("🏔️ New 52W Highs", "NEW_HIGHS"),
]


def _render_thrust_card_html(row: dict, idx_num: int) -> str:
    grp = _esc(row.get("Industry_Group", ""))
    grade_badge = _esc(row.get("Group_Grade_Badge", ""))
    rank_td = int(row.get("Rank_Today", 999) or 999)
    delta_1m = int(row.get("Delta_1M", 0) or 0)
    tag = _esc(row.get("Thrust_Status_Tag", ""))
    u_t0 = int(row.get("Up4_Today_Cnt", 0) or 0)
    u_t0_s2 = int(row.get("Up4_Today_S2_Cnt", 0) or 0)
    u_5d = int(row.get("Up4_5D_Unique", 0) or 0)
    u_5d_hits = int(row.get("Up4_5D_Hits", 0) or 0)
    nh_5d = int(row.get("NewHigh_5D_Unique", 0) or 0)
    q25 = int(row.get("Up25_Qtr_Cnt", 0) or 0)
    b50 = float(row.get("Breadth_50SMA_Pct", 0.0) or 0.0)
    score = float(row.get("Thrust_Score", 0.0) or 0.0)
    dens = float(row.get("Thrust_Density_Pct", 0.0) or 0.0)
    valid_c = int(row.get("Group_Valid_Cnt", 0) or 0)
    is_bounce = bool(row.get("Is_Counter_Trend_Bounce", False))

    border_col = "rgba(245,158,11,0.45)" if is_bounce else ("rgba(34,197,94,0.45)" if score >= 70 else "rgba(148,163,184,0.22)")
    score_col = "#fbbf24" if is_bounce else ("#34d399" if score >= 70 else "#93c5fd")
    d_sign = f"+{delta_1m}" if delta_1m > 0 else str(delta_1m)
    rank_txt = "Niche" if rank_td == 999 else f"#{rank_td} ({d_sign} 1M)"

    chips = []
    for t in (row.get("Thrusting_Tickers") or [])[:4]:
        tk = _esc(t.get("ticker", ""))
        url = _esc(t.get("tv_url", "#"))
        r1d = float(t.get("ret_1d_pct", 0.0) or 0.0)
        ipo_flag = " 🆕" if t.get("is_ipo") else ""
        vol_flag = " 🔊" if t.get("is_vol_confirmed") else ""
        pct_cls = "#34d399" if r1d >= 4.0 else "#cbd5e1"
        chips.append(
            f'<a href="{url}" target="_blank" style="text-decoration:none;background:rgba(15,23,42,0.85);'
            f'border:1px solid rgba(148,163,184,0.25);border-radius:6px;padding:2px 7px;font-size:11px;'
            f'color:#f8fafc;display:inline-flex;align-items:center;gap:4px;">'
            f'<b>{tk}</b><span style="color:{pct_cls};">{r1d:+.1f}%</span>{vol_flag}{ipo_flag}</a>'
        )
    chips_html = "".join(chips) if chips else '<span style="color:#64748b;font-size:11px;">No liquid movers</span>'

    return f"""
    <div style="background:linear-gradient(145deg,rgba(15,23,42,0.92),rgba(30,41,59,0.78));
                border:1px solid {border_col};border-radius:10px;padding:11px 13px;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:6px;">
        <div>
          <div style="font-size:13px;font-weight:700;color:#f8fafc;">
            #{idx_num} {grp}
            <span title="O'Neil Industry Group Grade &amp; Current Rank" style="font-size:10px;font-weight:600;color:#cbd5e1;background:rgba(51,65,85,0.8);padding:1px 6px;border-radius:99px;margin-left:4px;">{grade_badge}</span>
          </div>
          <div style="font-size:11px;color:#94a3b8;margin-top:2px;">{tag} · {rank_txt} · <span title="Scoped tradable members in active universe">Scoped: {valid_c}</span></div>
        </div>
        <div style="text-align:right;white-space:nowrap;" title="Size-normalized Stockbee Thrust Score (0-100); counter-trend &lt;50SMA bounces damped 0.55x">
          <div style="font-size:14px;font-weight:800;color:{score_col};">{score:.0f}</div>
          <div style="font-size:10px;color:#94a3b8;">{dens:.0f}% pack</div>
        </div>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:10px;font-size:11px;color:#cbd5e1;margin-bottom:7px;">
        <span title="Liquid stocks up >= +4% today (count already in true Stage 2 uptrend)"><b>Today +4%:</b> {u_t0} ({u_t0_s2} S2)</span>
        <span title="Unique stocks up >= +4% across trailing 5 sessions (total thrust hits)"><b>5D Uniq:</b> {u_5d} ({u_5d_hits} hits)</span>
        <span title="Stocks hitting fresh 52-Week Highs over last 5 days"><b>52W NH:</b> {nh_5d}</span>
        <span title="Stocks up >= +25% over trailing 63 trading days (excl. split spikes)"><b>+25Q:</b> {q25}</span>
        <span title="Percentage of group constituents closing above 50-day SMA"><b>&gt;50SMA:</b> {b50:.0f}%</span>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:5px;">{chips_html}</div>
    </div>
    """


def render_thrusting_industries_section(
    market: str,
    universe_scope: str,
    m_gate_regime: str | None = None,
) -> None:
    _inject_clb_tooltip_css()
    is_us = str(market or "IN").strip().upper().startswith("US")
    mkt_code = "US" if is_us else "IN"
    pfx = "mb_us" if is_us else "mb_in"
    target_page = _resolve_igm_target_page(is_us)
    jump_key = f"kt_igm_jump_{'us' if is_us else 'in'}"

    st.markdown(
        _strip_html_indent(
            f"""
            <div style="margin:14px 0 6px 0;padding:8px 12px;border-left:3px solid #38bdf8;
                        background:rgba(15,23,42,0.55);border-radius:0 8px 8px 0;">
              <span style="font-size:14px;font-weight:700;color:#f8fafc;">🏭 Stockbee Thrusting Industries</span>
              <span style="font-size:11px;color:#94a3b8;margin-left:8px;">Canonical Group Attribution · Scoped to {_esc(universe_scope)}</span>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    labels = [x[0] for x in _THRUST_MODE_OPTIONS]
    mode_map = dict(_THRUST_MODE_OPTIONS)
    c_t_mode, c_t_guide = st.columns([4.2, 0.9])
    with c_t_mode:
        sel_lbl = st.radio(
            "Thrust Attribution Lens",
            labels,
            horizontal=True,
            key=f"{pfx}_thrust_mode",
            label_visibility="collapsed",
            help="5D+Today: balanced short-term thrust score | Today ±4%: strongest Stage 2 bursts today | +25% Qtr: sustained 63D momentum leaders | New 52W Highs: institutional accumulation breakouts.",
        )
    mode_code = mode_map.get(sel_lbl, "5D_AND_TODAY")

    with c_t_guide:
        with st.popover("📖 Guide", use_container_width=True):
            st.markdown(
                """
**How Stockbee Thrusting Industries Works**

- **🚀 5D + Today:** Composite score blending trailing 5-day Stage-2 +4% thrusts, today's +4% bursts, 52W highs, +25% Qtr trends, volume-confirmed breakouts (`🔊 RVOL ≥ 1.4x`), and **pack density** (`÷ √N`).
- **⚡ Today ±4%:** Focuses strictly on today's Stage-2 ≥+4% movers (falls back to 5D leaders on quiet pullback days).
- **📈 +25% Qtr:** Groups housing the most institutional **63-day +25% trend winners** (unadjusted split spikes filtered out).
- **🏔️ New 52W Highs:** Groups pushing the most names to fresh 52-Week Highs over the last 5 sessions.
- **📊 Attribution Ledger:** Reconciles every stock in today's Stockbee `+4%` row into `attributed` (valid mapped), `illiquid` (`< ₹2.5 Cr` IN / `< $5M` US), `quarantined` (corporate actions `>35%` or India exact `+5%` upper-circuit locks), and `unmapped`.
- **🩹 Counter-Trend Bounce (`0.55x`):** If `<40%` of thrusts are in Stage 2 and group `>50SMA < 35%`, score is damped so broken downtrend dead-cat bounces don't outrank real Stage-2 leaders.
                """
            )

    t_pills = [
        _hover_pill("🚀 5D + Today", "5D + Today Thrust Score", "Blends 5-day unique +4% Stage-2 movers, today's bursts, 52W NH, +25Q trends & pack density (÷ √GroupSize).", active=(mode_code == "5D_AND_TODAY")),
        _hover_pill("⚡ Today ±4%", "Today's Immediate Impulse", "Sorts strictly by today's liquid Stage-2 +4% breakouts (gracefully falls back to 5D thrust on quiet down days).", active=(mode_code == "TODAY_4PCT")),
        _hover_pill("📈 +25% Qtr", "Quarterly Trend Leaders (+25% / 63D)", "Surfaces groups where the most liquid names have gained >= +25% over 63 sessions (splits excluded).", active=(mode_code == "QTR_25PCT")),
        _hover_pill("🏔️ New 52W Highs", "Fresh 52-Week High Clusters", "Ranks groups by unique names making fresh 252-day highs over the past 5 trading days.", active=(mode_code == "NEW_HIGHS")),
        _hover_pill("📊 Ledger & Guards", "Zero Silent Drops Reconciliation", "Attributed + Illiquid + Quarantined (splits >35% or India H=L=C circuits) + Unmapped always equals Grid +4% total.", align_right=True),
        _hover_pill("🩹 Bounce 0.55x", "Counter-Trend Bounce Damping", "Groups <35% above 50SMA with weak Stage-2 participation get 0.55x damped so oversold bounces don't fool you.", align_right=True),
    ]
    st.markdown(
        f'<div class="kt-tip-bar"><span style="color:#64748b;font-weight:700;">Hover for details:</span>{"".join(t_pills)}</div>',
        unsafe_allow_html=True,
    )

    res = compute_stockbee_thrusting_industries(
        market=mkt_code,
        universe_scope=universe_scope,
        mode=mode_code,
        top_n=6,
    )
    as_of = res.get("as_of") or {}
    ledger = res.get("ledger") or {}
    summary = res.get("summary") or {}

    if as_of.get("is_provisional"):
        st.warning(
            f"⏱ **Provisional Intraday Bar ({as_of.get('prices_date', '—')}):** Final Stockbee thrust verdict waits for session close."
        )
    if as_of.get("igm_stale"):
        st.caption(
            f"⚠️ Industry Group Matrix cache (`{as_of.get('groups_date', '—')}`) lags price matrix (`{as_of.get('prices_date', '—')}`)."
        )

    mg_badge = f" · M-Gate: {_esc(m_gate_regime)}" if m_gate_regime else ""
    st.markdown(
        _strip_html_indent(
            f"""
            <div style="font-size:12px;color:#e2e8f0;background:rgba(30,41,59,0.6);border:1px solid rgba(148,163,184,0.2);
                        border-radius:8px;padding:7px 11px;margin-bottom:8px;">
              <div><b>{_esc(summary.get('headline_msg', ''))}</b></div>
              <div style="font-size:11px;color:#94a3b8;margin-top:3px;">
                Ledger ({_esc(as_of.get('prices_date', '—'))}): <b>{int(ledger.get('grid_up4_total', 0))}</b> ≥+4% ·
                <b>{int(ledger.get('attributed', 0))}</b> attributed ·
                {int(ledger.get('illiquid', 0))} illiquid ·
                {int(ledger.get('quarantined', 0))} quarantined ·
                {int(ledger.get('unmapped', 0))} unmapped
                · T2108: {_esc(summary.get('t2108') if summary.get('t2108') is not None else 'N/A')}%{mg_badge}
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    if summary.get("used_fallback_note"):
        st.caption(f"ℹ️ {summary['used_fallback_note']}")

    thrust_df = res.get("thrust_df")
    if not isinstance(thrust_df, pd.DataFrame) or thrust_df.empty:
        st.info(summary.get("headline_msg") or "No thrusting industries detected for this window.")
        return

    cards_html = [
        _render_thrust_card_html(row.to_dict(), idx_i)
        for idx_i, (_, row) in enumerate(thrust_df.iterrows(), 1)
    ]
    st.markdown(
        _strip_html_indent(
            f"""
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:10px;margin-bottom:8px;">
              {''.join(cards_html)}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    # 1-Click Drill Buttons in Main Flow (never inside on_click!)
    btn_cols = st.columns(min(6, max(1, len(thrust_df))))
    for idx_i, (_, row) in enumerate(thrust_df.iterrows()):
        grp_name = str(row["Industry_Group"])
        short_lbl = grp_name if len(grp_name) <= 18 else f"{grp_name[:16]}…"
        with btn_cols[idx_i % len(btn_cols)]:
            if st.button(f"🔍 Drill {short_lbl}", key=f"{pfx}_drill_{_grp_hash(grp_name)}", use_container_width=True):
                st.session_state[jump_key] = build_igm_jump_payload(mkt_code, grp_name)
                st.switch_page(target_page)

    dist_df = res.get("distribution_df")
    if isinstance(dist_df, pd.DataFrame) and not dist_df.empty:
        dist_items = []
        for _, drow in dist_df.iterrows():
            sells = " ".join(
                f'<a href="{_esc(s.get("tv_url", "#"))}" target="_blank" style="color:#fda4af;text-decoration:none;margin-right:6px;">'
                f'<b>{_esc(s.get("ticker", ""))}</b> ({float(s.get("ret_1d_pct", 0.0)):+.1f}%)</a>'
                for s in (drow.get("Selling_Tickers") or [])[:3]
            )
            dist_items.append(
                f'<span style="margin-right:14px;"><b>{_esc(drow.get("Industry_Group", ""))}</b> '
                f'({int(drow.get("Dn4_Today_Cnt", 0))} ≤-4%): {sells}</span>'
            )
        st.markdown(
            _strip_html_indent(
                f"""
                <div style="margin-top:6px;padding:6px 10px;border-radius:8px;font-size:11px;
                            background:rgba(127,29,29,0.22);border:1px solid rgba(248,113,113,0.3);color:#fecaca;">
                  🩸 <b>Distribution Clusters:</b> {''.join(dist_items)}
                </div>
                """
            ),
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# 2. CLUSTER-FIRST LEADERSHIP BOARD (APP.PY SECTION 0 INDIA & US)
# -----------------------------------------------------------------------------
_LENS_OPTIONS = [
    ("⚖️ Auto (M-Gate)", "AUTO"),
    ("🛡️ Holding Up Best", "DEFENSE"),
    ("👑 CANSLIM Balanced", "CANSLIM"),
    ("⚡ Thrust Surgers", "THRUST"),
    ("💎 O'Neil Rank #1-20", "ONEIL"),
]


def _render_cluster_card_html(cluster: dict) -> str:
    b_rank = int(cluster.get("board_rank", 1))
    grp = _esc(cluster.get("industry_group", ""))
    grade_badge = _esc(cluster.get("grade_badge", ""))
    r_path = _esc(cluster.get("rank_path", ""))
    p_tag = _esc(cluster.get("persistence_tag", ""))
    score = float(cluster.get("cluster_score", 0.0) or 0.0)
    action_cnt = int(cluster.get("actionable_cnt", 0) or 0)
    is_ext = bool(cluster.get("is_extended_group", False))

    status_pill = (
        '<span style="color:#fbbf24;background:rgba(120,53,15,0.45);padding:1px 6px;border-radius:99px;font-size:10px;">⏳ Extended Group</span>'
        if is_ext and action_cnt == 0
        else f'<span style="color:#34d399;background:rgba(6,78,59,0.5);padding:1px 6px;border-radius:99px;font-size:10px;">🎯 Actionable ({action_cnt})</span>'
    )
    alpha_pill = ' <span style="color:#38bdf8;font-size:10px;">⚡ Dual Alpha</span>' if cluster.get("is_dual_alpha") else ""
    mirage_pill = ' <span style="color:#fca5a5;font-size:10px;">🪞 Mirage</span>' if cluster.get("is_relative_mirage") else ""

    inst_p = cluster.get("inst_accum_pct")
    inst_txt = f" · Inst: {float(inst_p):.0f}%" if inst_p is not None else ""
    funda_cov = float(cluster.get("funda_cov", 0.0) or 0.0)
    funda_pct_v = float(cluster.get("funda_pct", 0.0) or 0.0)
    funda_chip = "N/A" if (funda_cov == 0.0 and funda_pct_v == 0.0) else f"{funda_pct_v:.0f}%"
    why = cluster.get("why_ranked") or {}
    w_rs = int(why.get("rs", 0))
    w_br = int(why.get("breadth", 0))
    w_fn = int(why.get("funda", 0))
    w_th = int(why.get("thrust", 0))
    why_txt = f"Why ranked: RS {w_rs} · Br {w_br} · Fn {w_fn} · Th {w_th}"
    b50_v = float(cluster.get("breadth_50sma", 0.0) or 0.0)
    pen_note = " (−8 weak <50SMA penalty applied because >50SMA <45%)" if b50_v < 45.0 else ""
    score_tip_body = (
        f"0–100 CANSLIM Cluster Composite for {cluster.get('industry_group', '')}: "
        f"35% Pure Price-RS ({w_rs}) + 25% Internal Breadth ({w_br}) + "
        f"20% Fundamentals/Inst ({w_fn}) + 20% Stockbee 5D Thrust ({w_th}){pen_note}. "
        f"(Re-weighted 42/33/25 if fundamentals cache is sparse.)"
    )

    leader_rows = []
    for l in (cluster.get("leaders") or []):
        url = _esc(l.get("tv_url", "#"))
        tk = _esc(l.get("ticker", ""))
        t_badge = _esc(l.get("tier_badge", ""))
        rs_v = int(l.get("rs", 0) or 0)
        if l.get("has_funda"):
            funda_txt = f"EPS {float(l.get('eps_growth', 0)):+.0f}% · Sales {float(l.get('sales_growth', 0)):+.0f}% · ROE {float(l.get('roe', 0)):.0f}%"
        else:
            funda_txt = "Funda: N/A"
        badges = []
        if l.get("is_blue_dot"):
            badges.append("🔵")
        if l.get("is_ipo"):
            badges.append("🆕 IPO")
        if l.get("is_held"):
            badges.append("💼 HELD")
        if l.get("ern_days") is not None:
            badges.append(f"⚠️ ERN {int(l['ern_days'])}D")
        if l.get("brain_score") is not None:
            badges.append(f"🧠 {int(l['brain_score'])}")
        b_str = f' <span style="font-size:10px;color:#e2e8f0;">{" ".join(badges)}</span>' if badges else ""
        st_str = _esc(l.get("status", ""))
        d_ema = float(l.get("dist_ema", 0.0) or 0.0)

        leader_rows.append(
            f'<div style="display:flex;justify-content:space-between;align-items:center;gap:6px;'
            f'padding:4px 7px;background:rgba(15,23,42,0.72);border-radius:6px;margin-top:4px;font-size:11px;">'
            f'  <div style="min-width:0;">'
            f'    <a href="{url}" target="_blank" title="Open TradingView chart" style="color:#f8fafc;font-weight:700;text-decoration:none;">{tk}</a>'
            f'    <span title="O&#39;Neil Universe Relative Strength Rating (1-99)" style="color:#93c5fd;margin-left:5px;">RS {rs_v}</span>'
            f'    <span title="CANSLIM Tier: TML (earnings+sales+inst accum) &gt; Growth RS &gt; Pure RS &gt; Neg EPS" style="color:#cbd5e1;margin-left:5px;">{t_badge}</span>{b_str}'
            f'    <div style="font-size:10px;color:#94a3b8;">{funda_txt}</div>'
            f'  </div>'
            f'  <div style="text-align:right;white-space:nowrap;font-size:10px;color:#cbd5e1;">'
            f'    <div title="Setup Actionability vs Base Pivot">{st_str}</div>'
            f'    <div title="Distance above/below 21-day EMA" style="color:#94a3b8;">21E: {d_ema:+.1f}%</div>'
            f'  </div>'
            f'</div>'
        )
    sup_lag = int(cluster.get("suppressed_laggards", 0) or 0)
    sup_fail = int(cluster.get("suppressed_failed", 0) or 0)
    sup_sales = int(cluster.get("suppressed_low_sales", 0) or 0)
    min_sales_thr = int(cluster.get("min_sales_pct", 0) or 0)
    hidden_parts = []
    if sup_lag > 0:
        hidden_parts.append(f"{sup_lag} <RS80")
    if sup_fail > 0:
        hidden_parts.append(f"{sup_fail} failed/damaged")
    if sup_sales > 0:
        hidden_parts.append(f"{sup_sales} <{min_sales_thr}% sales" if min_sales_thr > 0 else f"{sup_sales} low sales")

    if leader_rows:
        leaders_html = "".join(leader_rows)
        if len(leader_rows) < 3 and hidden_parts:
            leaders_html += (
                f'<div style="font-size:10px;color:#64748b;margin-top:4px;font-style:italic;">'
                f'Only {len(leader_rows)} RS≥80 healthy leader(s) · {_esc(" · ".join(hidden_parts))} hidden'
                f'</div>'
            )
    else:
        hidden_suffix = f" ({_esc(' · '.join(hidden_parts))} hidden)" if hidden_parts else ""
        leaders_html = f'<div style="font-size:11px;color:#64748b;margin-top:6px;">No healthy RS≥80 leaders in group{hidden_suffix}</div>'

    return _strip_html_indent(
        f"""
    <div style="background:linear-gradient(155deg,rgba(15,23,42,0.95),rgba(30,41,59,0.82));
                border:1px solid rgba(148,163,184,0.22);border-radius:11px;padding:11px 13px;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
        <div>
          <div style="font-size:13px;font-weight:700;color:#f8fafc;">
            #{b_rank} {grp}
            <span title="O&#39;Neil Group Grade &amp; Rank (#1-145)" style="font-size:10px;background:rgba(51,65,85,0.85);color:#e2e8f0;padding:1px 6px;border-radius:99px;margin-left:4px;">{grade_badge}</span>
            {alpha_pill}{mirage_pill}
          </div>
          <div style="font-size:10px;color:#94a3b8;margin-top:2px;"><span title="Rank trajectory: 6W -&gt; 3W -&gt; 1W -&gt; Today">{r_path}</span> · {p_tag} · {status_pill}</div>
        </div>
        <div class="kt-tip kt-tip-r" tabindex="0" style="text-align:right;white-space:nowrap;flex-direction:column;align-items:flex-end;gap:0;padding:3px 7px;border-radius:8px;background:rgba(15,23,42,0.55);">
          <div style="font-size:15px;font-weight:800;color:#38bdf8;line-height:1.1;">{score:.1f}</div>
          <div style="font-size:9px;color:#94a3b8;letter-spacing:0.04em;">SCORE ⓘ</div>
          <span class="kt-tip-box">
            <div style="font-weight:700;color:#38bdf8;margin-bottom:3px;">Cluster Score: {score:.1f} / 100</div>
            <div>{_esc(score_tip_body)}</div>
          </span>
        </div>
      </div>
      <div style="font-size:10.5px;color:#cbd5e1;margin-top:6px;display:flex;flex-wrap:wrap;gap:8px;">
        <span title="% of group constituents with RS &gt;= 80"><b>RS≥80:</b> {float(cluster.get('breadth_rs80', 0)):.0f}%</span>
        <span title="% of group constituents above 50-day SMA"><b>&gt;50SMA:</b> {float(cluster.get('breadth_50sma', 0)):.0f}%</span>
        <span title="% of group making RS Line New Highs"><b>RSNH:</b> {float(cluster.get('rsnh_pct', 0)):.0f}%</span>
        <span title="% of group with CANSLIM fundamentals / institutional sponsorship"><b>Funda:</b> {funda_chip}{inst_txt}</span>
      </div>
      <div style="font-size:10.5px;color:#94a3b8;margin-top:3px;display:flex;flex-wrap:wrap;gap:8px;">
        <span title="Stocks up &gt;= +4% today">T0 +4%: <b>{int(cluster.get('up4_today', 0))}</b></span>
        <span title="Unique stocks up &gt;= +4% in trailing 5 days">5D Uniq: <b>{int(cluster.get('up4_5d_unique', 0))}</b></span>
        <span title="Fresh 52W Highs in trailing 5 days">52W NH: <b>{int(cluster.get('nh_5d', 0))}</b></span>
        <span title="Stocks up &gt;= +25% in 63 trading days">+25Q: <b>{int(cluster.get('up25_qtr', 0))}</b></span>
        <span title="Tradable stocks in scope for this group">Scoped: {int(cluster.get('scoped_valid_cnt', 0))}</span>
      </div>
      <div title="0-100 percentile sub-scores contributing to cluster rank: Pure Price-RS, Internal Breadth, CANSLIM Fundamentals, Stockbee Thrust" style="font-size:10px;color:#64748b;margin-top:3px;">{_esc(why_txt)}</div>
      {leaders_html}
    </div>
    """
    )


_CLB_ACTIVE_LENS_HELP = {
    "DEFENSE": "🛡️ Holding Up Best: Sorting defensively by RS New Highs %, Near 52W Highs %, >50SMA % & Blue Dots to stalk names refusing to break.",
    "CANSLIM": "👑 CANSLIM Balanced: Blending Pure Price-RS (35%), Internal Breadth (25%), Fundamentals/Sponsorship (20%) & 5D Thrust (20%).",
    "THRUST": "⚡ Thrust Surgers: Prioritizing groups with the highest 5-day unique +4% bursts, today's +4% moves & 1-month rank velocity.",
    "ONEIL": "💎 O'Neil Rank #1-20: Classic IBD rank order sorted strictly by Industry Group Rank #1-145 fastest momentum first.",
}


def _extract_unique_de_drill_groups(
    de_holdings: list[dict],
    surfaced_clusters: list[dict],
    pfx: str,
    max_groups: int = 4,
) -> list[dict]:
    """Deduplicate D/E danger drilldown buttons by unique group (excluding top surfaced cards)."""
    surfaced_names = {str(c.get("industry_group") or "") for c in (surfaced_clusters or [])}
    seen_groups = set()
    out = []
    for item in (de_holdings or []):
        grp = str(item.get("industry_group") or "").strip()
        if not grp or grp in surfaced_names or grp in seen_groups:
            continue
        seen_groups.add(grp)
        out.append(
            {
                "industry_group": grp,
                "grade_badge": str(item.get("grade_badge") or ""),
                "rank_today": int(item.get("rank_today") or 0),
                "button_key": f"{pfx}_de_drill_{_grp_hash(grp)}",
            }
        )
        if len(out) >= int(max_groups):
            break
    return out


def _render_triage_chip_link(item: dict, is_danger: bool = False) -> str:
    url = _esc(item.get("tv_url", "#"))
    tk = _esc(item.get("ticker", ""))
    rs_val = item.get("rs")
    rs_txt = f"RS {int(rs_val)}" if rs_val is not None else "RS —"
    if is_danger:
        gb = _esc(item.get("grade_badge") or "D/E")
        bg = "rgba(127,29,29,0.5)"
        bd = "rgba(248,113,113,0.45)"
        fg = "#fecaca"
        meta_txt = f"{gb} · {rs_txt}"
    else:
        br = item.get("board_rank")
        rk_txt = f"#{int(br)}" if br else ""
        alerts = [a for a in (item.get("alert_reasons") or []) if not a.startswith("⚠️ ERN")]
        flag_txt = f" · {_esc(alerts[0])}" if alerts else ""
        bg = "rgba(6,78,59,0.42)" if not alerts else "rgba(120,53,15,0.48)"
        bd = "rgba(52,211,153,0.4)" if not alerts else "rgba(251,191,36,0.45)"
        fg = "#d1fae5" if not alerts else "#fde68a"
        meta_txt = f"{rk_txt} · {rs_txt}{flag_txt}".lstrip(" ·")
    return (
        f'<span style="display:inline-flex;align-items:center;gap:4px;padding:2px 7px;border-radius:999px;'
        f'background:{bg};border:1px solid {bd};font-size:11px;color:{fg};">'
        f'<a href="{url}" target="_blank" style="color:#f8fafc;font-weight:700;text-decoration:none;">{tk}</a>'
        f'<span style="opacity:0.9;">({meta_txt})</span></span>'
    )


def _render_portfolio_triage_table_html(de_holdings: list[dict], weak_alerts: list[dict]) -> str:
    rows_html = []
    for item in (de_holdings or []) + (weak_alerts or []):
        url = _esc(item.get("tv_url", "#"))
        tk = _esc(item.get("ticker", ""))
        reasons = " · ".join(_esc(r) for r in (item.get("alert_reasons") or [])) or "—"
        grp = _esc(item.get("industry_group", ""))
        gb = _esc(item.get("grade_badge") or item.get("grade_str") or "—")
        b50 = float(item.get("breadth_50sma", 0.0) or 0.0)
        d1m = int(item.get("delta_1m", 0) or 0)
        rs_val = item.get("rs")
        rs_txt = str(int(rs_val)) if rs_val is not None else "—"
        st_txt = _esc(item.get("status") or "—")
        d_ema = item.get("dist_ema")
        ema_txt = f"{float(d_ema):+.1f}%" if d_ema is not None else "—"
        ern_d = item.get("ern_days")
        ern_txt = f"⚠️ {int(ern_d)}D" if ern_d is not None else "—"
        row_bg = "rgba(127,29,29,0.22)" if item.get("is_danger") else "rgba(120,53,15,0.16)"
        rows_html.append(
            f'<tr style="background:{row_bg};border-bottom:1px solid rgba(148,163,184,0.15);">'
            f'<td style="padding:6px 8px;white-space:nowrap;"><a href="{url}" target="_blank" style="color:#f8fafc;font-weight:700;text-decoration:none;">{tk}</a></td>'
            f'<td style="padding:6px 8px;color:#fecaca;">{reasons}</td>'
            f'<td style="padding:6px 8px;color:#e2e8f0;">{grp} <span style="color:#94a3b8;">({gb})</span></td>'
            f'<td style="padding:6px 8px;white-space:nowrap;color:#cbd5e1;">{b50:.0f}% / {d1m:+d}</td>'
            f'<td style="padding:6px 8px;white-space:nowrap;color:#93c5fd;font-weight:700;">{rs_txt}</td>'
            f'<td style="padding:6px 8px;white-space:nowrap;color:#cbd5e1;">{st_txt} · {ema_txt}</td>'
            f'<td style="padding:6px 8px;white-space:nowrap;color:#fbbf24;">{ern_txt}</td>'
            f'</tr>'
        )
    return _strip_html_indent(
        f"""
        <div style="overflow-x:auto;margin-top:6px;margin-bottom:8px;">
          <table style="width:100%;border-collapse:collapse;font-size:11.5px;text-align:left;">
            <thead>
              <tr style="border-bottom:1px solid rgba(148,163,184,0.3);color:#94a3b8;font-size:10.5px;">
                <th style="padding:5px 8px;">Ticker</th>
                <th style="padding:5px 8px;">Alert Reasons</th>
                <th style="padding:5px 8px;">Group (Grade #Rank)</th>
                <th style="padding:5px 8px;">Grp &gt;50SMA / Δ1M</th>
                <th style="padding:5px 8px;">Stock RS</th>
                <th style="padding:5px 8px;">Status &amp; 21E</th>
                <th style="padding:5px 8px;">ERN</th>
              </tr>
            </thead>
            <tbody>{''.join(rows_html)}</tbody>
          </table>
        </div>
        """
    )


def render_cluster_leadership_board_section(
    market: str,
    portfolio_tickers: list[str] | set[str] | None = None,
    earnings_cache: dict | None = None,
) -> None:
    _inject_clb_tooltip_css()
    is_us = str(market or "INDIA").strip().upper().startswith("US")
    pfx = "clb_us" if is_us else "clb_in"
    mkt_code = "US" if is_us else "IN"
    target_page = _resolve_igm_target_page(is_us)
    jump_key = f"kt_igm_jump_{'us' if is_us else 'in'}"

    st.markdown(
        _strip_html_indent(
            f"""
            <div style="margin:6px 0 8px 0;padding:8px 12px;border-left:3px solid #22c55e;
                        background:rgba(15,23,42,0.65);border-radius:0 8px 8px 0;
                        display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
              <div>
                <span style="font-size:15px;font-weight:800;color:#f8fafc;">🌐 0. Where Leadership Lives Today</span>
                <span style="font-size:11px;color:#94a3b8;margin-left:8px;">
                  Cluster-First CANSLIM Desk · {'US Equities' if is_us else 'India NSE/BSE'}
                </span>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    c_lens, c_top, c_sales, c_conf, c_guide, c_exp = st.columns([2.15, 0.72, 1.0, 0.85, 0.62, 0.68])
    with c_lens:
        lens_lbl = st.radio(
            "Ranking Lens",
            [x[0] for x in _LENS_OPTIONS],
            horizontal=True,
            key=f"{pfx}_lens",
            label_visibility="collapsed",
            help="Auto (M-Gate) matches the market regime (Holding Up Best in RED; CANSLIM Balanced in YELLOW/GREEN). Or override manually with Holding Up Best, CANSLIM Balanced, Thrust Surgers, or O'Neil Rank #1-20.",
        )
    lens_code = dict(_LENS_OPTIONS).get(lens_lbl, "AUTO")
    with c_top:
        top_lbl = st.radio(
            "Count",
            ["Top 8", "Top 10"],
            horizontal=True,
            key=f"{pfx}_topn",
            label_visibility="collapsed",
            help="Top 8 shows 2 tight rows of top clusters for rapid execution focus. Switch to Top 10 for a wider bench when breadth is expanding.",
        )
    top_n_val = 10 if "10" in top_lbl else 8
    with c_sales:
        min_sales_val = st.slider(
            "Min Sales %",
            min_value=0,
            max_value=50,
            value=20,
            step=5,
            format="%d%%",
            key=f"{pfx}_min_sales_pct",
            help="Minimum Sales Growth % for Top-3 card slots & Export TV (default 20%). Stocks with missing fundamentals (N/A) or 0% sales growth stay visible in case sales data isn't populated; drag to 0% to turn off.",
        )
    with c_conf:
        conf_only = st.toggle(
            "Confirmed Pack",
            value=True,
            key=f"{pfx}_confirmed_only",
            help="ON (Recommended): Excludes D/E Danger Groups & filters to confirmed/grade A+–B packs (backfills only if fewer than N qualify). OFF: Inspect all canonical groups including weak/early packs.",
        )

    pt_list = sorted(str(t) for t in portfolio_tickers if t) if portfolio_tickers else None
    board = compute_cluster_leadership_board(
        market=market,
        lens=lens_code,
        top_n=top_n_val,
        confirmed_only=conf_only,
        portfolio_tickers=pt_list,
        earnings_cache=earnings_cache,
        min_sales_pct=float(min_sales_val),
    )

    with c_guide:
        with st.popover("📖 Guide", use_container_width=True):
            st.markdown(
                """
**How to Read Where Leadership Lives Today**

### 1. Ranking Lenses
- **⚖️ Auto (M-Gate):** Reads Market Tide + Stockbee T2108 automatically. **RED Market:** defaults to `🛡️ Holding Up Best` (stalk mode). **YELLOW / Stretched (T2108 >72%):** selective `👑 CANSLIM Balanced`. **GREEN:** full `👑 CANSLIM Balanced`.
- **🛡️ Holding Up Best:** Defensive RS filter (`35% RSNH% + 25% Near 52WH% + 25% >50SMA% + 15% Blue Dots`). Use during corrections to build your stalk list.
- **👑 CANSLIM Balanced:** `35% Pure Price-RS + 25% Internal Breadth + 20% CANSLIM Funda/Inst + 20% 5D Thrust`. Re-weights cleanly if fundamentals are sparse.
- **⚡ Thrust Surgers:** Sorts by trailing `5D +4% Unique`, `Today +4%`, and `Δ 1M Velocity` to catch fresh money rotation early.
- **💎 O'Neil Rank #1-20:** Pure IBD/O'Neil Industry Group rank order (`Rank_Today` `#1` to `#145`).

### 2. Controls & Filters
- **Top 8 / Top 10:** Number of leading industry cards displayed (8 = focused execution, 10 = wider rotation view).
- **Min Sales % (Slider, default 20%):** Excludes Top-3 buy slots & `📋 Export TV` names whose reported non-zero Sales Growth is below this threshold (e.g., `< 20%` or negative). Keeps `0%` and `Funda: N/A` stocks visible in case sales data isn't populated; set slider to `0%` to turn off.
- **Confirmed Pack (ON/OFF):** When **ON**, drops `D/E` Danger Groups and prioritizes Grade `A+/A/B` or pack-confirmed groups (`≥2 RS≥80 leaders`). Turn **OFF** to scan lower-ranked turning groups.
- **📋 Export TV:** Generates `###Industry_Group,SYM1,SYM2...` TradingView section watchlist ready to paste.

### 3. Card Metrics & Leader Tiers
- **Cluster `SCORE` (0–100):** Overall leadership composite for the industry group: `35% Pure Price-RS + 25% Internal Breadth + 20% Fundamentals/Inst + 20% 5D Stockbee Thrust` (with `-8` points if `<45%` of constituents are above 50 SMA, or `42/33/25` re-weighting when fundamentals cache is sparse). Hover `SCORE ⓘ` on any card for that group's exact breakdown.
- **Rank Path (`#54 → #17 → #14 → #7`):** 6W → 3W → 1W → Today industry rank climb (`🆕 New Leader` / `🏛️ Established`).
- **`Why ranked` (`RS · Br · Fn · Th`):** 0–100 percentile contribution of Pure Price-RS, Internal Breadth, Fundamentals, and Stockbee Thrust.
- **Tiers & Buy-Slot Gate:** Top-3 card slots & `📋 Export TV` strictly require **`RS ≥ 80`, non-`FAILED`, and `21E ≥ -5%`** (zero weak backfill; omitted laggards/broken charts are noted per card). Tier order: `💎 TML` (strong CANSLIM EPS/Sales/ROE + inst. accumulation) > `🚀 Growth RS` > `📈 RS` (clean breakout even if `Funda: N/A`) > `⚠️ Neg EPS/Sales`.
- **Status & Portfolio Risk Triage:** `🎯 Actionable (N)` = setups in Buy Zone / Retest / Coiling / `🔵` Blue Dot. `⏳ Extended Group` = ≥60% top leaders extended (>8% above 21E). `💼 HELD` = in your portfolio. `⚠️ ERN ≤5D` = earnings due within 5 days on held names. `🧠` = cached Artificial Brain confluence score. Bottom **Portfolio Alignment** & **🚨 Portfolio Risk Triage** flag every held stock in `D/E` Danger groups, failed/laggard charts, or `🚫 Tier-0 Illiquid` / `🔒 Circuit Risk` traps.
                """
            )

    with c_exp:
        with st.popover("📋 Export TV", use_container_width=True):
            tv_txt = board.get("tv_watchlist") or ""
            if tv_txt:
                st.code(tv_txt, language=None)
            else:
                st.caption("No symbols available for watchlist export.")

    active_l = str(board.get("active_lens", "CANSLIM"))
    clb_pills = [
        _hover_pill("⚖️ Auto (M-Gate)", "Auto Regime Switching", "Automatically switches to Holding Up Best when Market Tide is RED, and CANSLIM Balanced in YELLOW/GREEN uptrends.", active=(lens_code == "AUTO")),
        _hover_pill("🛡️ Holding Up Best", "Defense / Bear-Stalk Lens", "Ranks groups by RS New Highs %, Near 52W High %, >50SMA % & Blue Dots — finds leaders refusing to break.", active=(active_l == "DEFENSE")),
        _hover_pill("👑 CANSLIM Balanced", "Core Multibagger Composite", "35% Pure Price-RS + 25% Internal Breadth + 20% CANSLIM Fundamentals/Inst + 20% 5D Stockbee Thrust.", active=(active_l == "CANSLIM")),
        _hover_pill("⚡ Thrust Surgers", "Early Rotation / Momentum Burst", "Prioritizes groups with the highest 5D unique +4% movers, today's +4% bursts & 1M Rank Velocity.", active=(active_l == "THRUST")),
        _hover_pill("💎 O'Neil #1-20", "Classic Industry Group Rank", "Strict O'Neil Rank_Today ordering (#1 fastest group momentum first), using Cluster Score for ties.", active=(active_l == "ONEIL")),
        _hover_pill("🔢 Top 8 / 10", "Display Breadth Depth", "Top 8 fits 2 compact 4-col rows for highest-conviction focus. Top 10 expands the bench during broad rallies."),
        _hover_pill(
            f"📈 Min Sales: {min_sales_val}%" if min_sales_val > 0 else "📈 Min Sales: OFF",
            "Minimum Sales Growth Filter (Default 20%)",
            "Hides Top-3 card & Export TV stocks with non-zero Sales Growth below the slider threshold. Keeps 0% and Funda: N/A stocks visible in case sales data isn't populated.",
            active=(min_sales_val > 0),
        ),
        _hover_pill(f"✅ Confirmed Pack: {'ON' if conf_only else 'OFF'}", "Confirmed Pack Gate (ON / OFF)", "ON hides D/E Danger Groups & keeps Grade A+/A/B confirmed packs (backfills only if < N qualify). Turn OFF to view all canonical groups.", align_right=True),
        _hover_pill("🏷️ Tiers & Badges", "Leader Quality, Buy-Slot Gate & Risk Badges", "Top-3 slots & Export TV strictly require RS≥80, non-FAILED, and 21E≥-5%. Tiers: 💎 TML > 🚀 Growth RS > 📈 RS (Funda N/A) > ⚠️ Neg EPS. 🔵=RS New High | 🆕=IPO (<200d) | 💼 HELD | ⚠️ ERN≤5D (held) | 🧠=Brain.", align_right=True),
    ]
    st.markdown(
        f'<div class="kt-tip-bar"><span style="color:#64748b;font-weight:700;">Hover for definitions:</span>{"".join(clb_pills)}</div>',
        unsafe_allow_html=True,
    )

    m_gate = board.get("m_gate") or {}
    tone = m_gate.get("tone", "slate")
    bg_map = {
        "rose": "rgba(127,29,29,0.28)",
        "amber": "rgba(120,53,15,0.28)",
        "emerald": "rgba(6,78,59,0.30)",
        "slate": "rgba(30,41,59,0.55)",
    }
    bd_map = {
        "rose": "rgba(248,113,113,0.45)",
        "amber": "rgba(251,191,36,0.45)",
        "emerald": "rgba(52,211,153,0.45)",
        "slate": "rgba(148,163,184,0.25)",
    }
    lens_expl = _CLB_ACTIVE_LENS_HELP.get(active_l, "")
    sales_state_txt = f" · Min Sales ≥{min_sales_val}% (0%/N/A kept)" if min_sales_val > 0 else " · Min Sales OFF"
    pack_state_txt = f"{'Confirmed Pack ON (Danger D/E excluded)' if conf_only else 'Confirmed Pack OFF (All canonical ≥4-stock groups)'}{sales_state_txt}"
    st.markdown(
        _strip_html_indent(
            f"""
            <div style="background:{bg_map.get(tone, bg_map['slate'])};border:1px solid {bd_map.get(tone, bd_map['slate'])};
                        border-radius:8px;padding:7px 12px;margin-bottom:6px;font-size:12px;color:#f8fafc;">
              <div>
                <b>{_esc(m_gate.get('label', ''))}</b>
                <span style="font-size:11px;color:#cbd5e1;margin-left:10px;">Active Lens: <b>{_esc(active_l)}</b> · {pack_state_txt}</span>
              </div>
              <div style="font-size:10.5px;color:#cbd5e1;margin-top:3px;opacity:0.9;">{_esc(lens_expl)}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    if m_gate.get("early_rally"):
        st.caption("⚡ Stage 2–3 Early-Rally Regime: check Section 2 below for **Bull Market Initiators** (new leaders stepping out first).")

    meta = board.get("meta") or {}
    t2108_txt = f"{meta['t2108']}%" if meta.get("t2108") is not None else "N/A"
    partial_note = " · ⚠️ Sparse funda cache (Price/Breadth/Thrust re-weighted)" if meta.get("partial_funda_cache") else ""
    stale_note = " · ⚠️ Groups cache lagging prices" if meta.get("igm_stale") else ""
    st.caption(
        f"Prices: {meta.get('prices_date', '—')} · Groups: {meta.get('groups_date', '—')} · Funda: {meta.get('funda_as_of', 'N/A')} · "
        f"Ratio 5D: {meta.get('ratio_5d', 1.0):.2f} · T2108: {t2108_txt} · Dual Alpha Top-40: {meta.get('dual_alpha_top40', 0)}{partial_note}{stale_note}"
    )

    clusters = board.get("clusters") or []
    if not clusters:
        st.info("Cluster Leadership Board awaiting local Industry Group Matrix cache.")
        return

    grid_cards = [_render_cluster_card_html(c) for c in clusters]
    st.markdown(
        _strip_html_indent(
            f"""
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:10px;margin-bottom:8px;">
              {''.join(grid_cards)}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    # Drilldown buttons cleanly in rows of 4 (main-body switch_page only)
    for start in range(0, len(clusters), 4):
        chunk = clusters[start : start + 4]
        cols = st.columns(len(chunk))
        for j, cl in enumerate(chunk):
            grp_name = str(cl["industry_group"])
            short_nm = grp_name if len(grp_name) <= 20 else f"{grp_name[:18]}…"
            with cols[j]:
                if st.button(
                    f"🔍 #{cl['board_rank']} {short_nm}",
                    key=f"{pfx}_drill_{_grp_hash(grp_name)}",
                    use_container_width=True,
                ):
                    st.session_state[jump_key] = build_igm_jump_payload(mkt_code, grp_name)
                    st.switch_page(target_page)

    # Portfolio Alignment Bar & Collapsible Risk Triage Drawer (rendered only when holdings exist)
    overlap = board.get("overlap") or {}
    held_total = int(overlap.get("held_total", 0) or 0)
    if portfolio_tickers is not None and held_total > 0:
        in_cluster_items = overlap.get("held_in_clusters_items") or []
        de_holdings = overlap.get("de_holdings") or []
        weak_alerts = overlap.get("held_weak_alerts") or []
        ledger = overlap.get("ledger") or {}

        tot_surf = int(overlap.get("total_surfaced", 0) or 0)
        in_cl_chips = [_render_triage_chip_link(it, is_danger=False) for it in in_cluster_items[:8]]
        if len(in_cluster_items) > 8:
            in_cl_chips.append(f'<span style="font-size:11px;color:#94a3b8;">+{len(in_cluster_items) - 8} more</span>')
        in_cl_str = " ".join(in_cl_chips) if in_cl_chips else '<span style="color:#94a3b8;">None surfaced in Top clusters</span>'

        if de_holdings:
            de_chips = [_render_triage_chip_link(it, is_danger=True) for it in de_holdings[:8]]
            if len(de_holdings) > 8:
                de_chips.append(f'<span style="font-size:11px;color:#fca5a5;">+{len(de_holdings) - 8} more</span>')
            de_str = f'<span style="color:#fca5a5;font-weight:700;">🚨 D/E Danger ({len(de_holdings)}):</span> ' + " ".join(de_chips)
        else:
            de_str = '<span style="color:#34d399;font-weight:600;">✅ 0 in D/E Danger Groups</span>'

        weak_pill = (
            f' · <span style="color:#fbbf24;font-weight:700;">⚠️ {len(weak_alerts)} Weak/Failed Alert(s)</span>'
            if weak_alerts
            else ""
        )

        st.markdown(
            _strip_html_indent(
                f"""
                <div style="margin:6px 0 6px 0;padding:8px 11px;border-radius:8px;
                            background:rgba(15,23,42,0.72);border:1px solid rgba(148,163,184,0.22);
                            font-size:11.5px;color:#cbd5e1;line-height:1.6;">
                  <div>
                    💼 <b>Portfolio Alignment ({held_total} held · {tot_surf} TMLs shown):</b>
                    <b>{len(in_cluster_items)}</b> in Top {len(clusters)} clusters: {in_cl_str}
                  </div>
                  <div style="margin-top:4px;">{de_str}{weak_pill}</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        if de_holdings or weak_alerts:
            with st.expander(
                f"🚨 Portfolio Risk Triage — {len(de_holdings)} in D/E Danger · {len(weak_alerts)} Weak/Failed Held",
                expanded=False,
            ):
                l_tot = int(ledger.get("held_total", held_total) or held_total)
                l_top = int(ledger.get("top_clusters", 0) or 0)
                l_ab = int(ledger.get("ab_other", 0) or 0)
                l_c = int(ledger.get("c_groups", 0) or 0)
                l_de = int(ledger.get("de_danger", 0) or 0)
                l_ni = int(ledger.get("niche", 0) or 0)
                l_un = int(ledger.get("unmapped", 0) or 0)
                st.caption(
                    f"Held Map ({l_tot}): {l_top} in Top Clusters · {l_ab} in A+/A/B · "
                    f"{l_c} in C-Lagging · {l_de} in D/E Danger · {l_ni} Niche · {l_un} Unmapped/ETF"
                )
                st.markdown(
                    _render_portfolio_triage_table_html(de_holdings, weak_alerts),
                    unsafe_allow_html=True,
                )
                de_drill_groups = _extract_unique_de_drill_groups(de_holdings, clusters, pfx=pfx, max_groups=4)
                if de_drill_groups:
                    dcols = st.columns(len(de_drill_groups))
                    for j, dg in enumerate(de_drill_groups):
                        gname = dg["industry_group"]
                        short_g = gname if len(gname) <= 22 else f"{gname[:20]}…"
                        with dcols[j]:
                            if st.button(
                                f"🔍 Inspect {short_g}",
                                key=dg["button_key"],
                                use_container_width=True,
                            ):
                                st.session_state[jump_key] = build_igm_jump_payload(mkt_code, gname)
                                st.switch_page(target_page)
