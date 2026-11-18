"""
Institutional Cluster Leadership & Stockbee Thrust Attribution Engine
=====================================================================
Strictly non-blocking (zero network I/O): reads local cached OHLCV matrices,
`.cache/industry_group_*.json`, cached Artificial Brain snapshots, and read-only SQLite.

Shared Single-Source-of-Truth powering:
  - Stockbee Thrusting Industries panel (pages/market_breadth.py & _us.py)
  - Cluster-First Leadership Board ("Where Leadership Lives Today" on app.py)
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from canslim_rules import (
    classify_leader_tier,
    classify_stage2_series,
    has_valid_fundamentals,
    is_tml_confluence,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_IN_SUFFIX_STRIP_RE = re.compile(r"-(?:SM|ST|SZ|BE|BZ)$", re.IGNORECASE)
_EXCLUDED_GROUPS = {"General / Unclassified", "Shell Companies", "General", "Other", "Unknown", "None", ""}


# -----------------------------------------------------------------------------
# NORMALIZATION & READ-ONLY SQLITE HELPERS
# -----------------------------------------------------------------------------
def _normalize_ticker_key(raw_sym: str) -> str:
    s = str(raw_sym or "").strip().upper()
    s = s.replace(".NS", "").replace(".BO", "")
    return _IN_SUFFIX_STRIP_RE.sub("", s)


def _ticker_lookup_candidates(raw_sym: str) -> list[str]:
    s = str(raw_sym or "").strip()
    su = s.upper()
    clean = _normalize_ticker_key(su)
    cands = [s, su, clean]
    if "-" in clean:
        cands.append(clean.replace("-", "."))
    if "." in clean:
        cands.append(clean.replace(".", "-"))
    seen = set()
    out = []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _resolve_intel_lookup(stock_intel: dict, raw_t: str) -> dict | None:
    if not isinstance(stock_intel, dict) or not stock_intel:
        return None
    for k in _ticker_lookup_candidates(raw_t):
        hit = stock_intel.get(k)
        if isinstance(hit, dict):
            grp = str(hit.get("group_name") or "").strip()
            if grp and grp not in _EXCLUDED_GROUPS:
                return hit
    return None


def _lookup_map(mapping: dict, raw_t: str) -> dict | None:
    if not isinstance(mapping, dict) or not mapping:
        return None
    for k in _ticker_lookup_candidates(raw_t):
        if k in mapping:
            return mapping[k]
    return None


def _get_near_earnings_days(earnings_cache: dict | None, sym: str, raw_t: str = "") -> int | None:
    """Return integer days to earnings if 0 <= days <= 5, else None."""
    if not isinstance(earnings_cache, dict) or not earnings_cache:
        return None
    e_info = _lookup_map(earnings_cache, sym)
    if e_info is None and raw_t:
        e_info = _lookup_map(earnings_cache, raw_t)
    if isinstance(e_info, dict) and e_info.get("days_to_earnings") is not None:
        try:
            d_val = int(e_info["days_to_earnings"])
            if 0 <= d_val <= 5:
                return d_val
        except (TypeError, ValueError):
            pass
    return None


def _get_db_freshness_sig() -> tuple[str, str]:
    """Read freshness timestamps from Turso/SQLite via database.read_sql_df (zero raw sqlite3)."""
    funda_ts, spons_ts = "none", "none"
    try:
        from database import read_sql_df

        df_f = read_sql_df("SELECT MAX(updated_at) AS ts FROM fundamentals_cache")
        if df_f is not None and not df_f.empty and pd.notna(df_f.iloc[0].get("ts")):
            funda_ts = str(df_f.iloc[0]["ts"])
    except Exception:
        pass
    try:
        from database import read_sql_df

        df_s = read_sql_df("SELECT MAX(updated_at) AS ts FROM institutional_sponsorship")
        if df_s is not None and not df_s.empty and pd.notna(df_s.iloc[0].get("ts")):
            spons_ts = str(df_s.iloc[0]["ts"])
    except Exception:
        pass
    return (funda_ts, spons_ts)


@st.cache_data(ttl=1800, show_spinner=False)
def _load_fundamentals_and_sponsorship_cached(market: str, db_sig: tuple[str, str]) -> dict:
    """Scalar-keyed Turso/SQLite loader for fundamentals_cache + optional institutional_sponsorship."""
    is_us = str(market or "IN").strip().upper().startswith("US")
    mkt_code = "US" if is_us else "IN"
    funda_map: dict[str, dict] = {}
    spons_map: dict[str, dict] = {}
    funda_as_of = str(db_sig[0])[:10] if db_sig and db_sig[0] not in ("none", "", "None") else "N/A"

    try:
        from database import get_fundamentals_map_for_engine, read_sql_df

        raw_funda, latest_f = get_fundamentals_map_for_engine(mkt_code)
        if latest_f and latest_f not in ("none", "None", "N/A"):
            funda_as_of = latest_f
        for raw_tk, info in (raw_funda or {}).items():
            if not raw_tk or not isinstance(info, dict):
                continue
            sym = _normalize_ticker_key(raw_tk)
            payload = {
                "eps_growth": float(info.get("eps_growth", 0.0) or 0.0),
                "sales_growth": float(info.get("sales_growth", 0.0) or 0.0),
                "roe": float(info.get("roe", 0.0) or 0.0),
                "market_cap": float(info.get("market_cap", 0.0) or 0.0),
            }
            if sym not in funda_map or (has_valid_fundamentals(payload) and not has_valid_fundamentals(funda_map[sym])):
                funda_map[sym] = payload

        if not is_us:
            try:
                df_s = read_sql_df(
                    "SELECT symbol, sponsorship_grade, inst_change_qoq, total_inst_pct FROM institutional_sponsorship"
                )
                if df_s is not None and not df_s.empty:
                    for row_d in df_s.to_dict(orient="records"):
                        sym = _normalize_ticker_key(row_d.get("symbol"))
                        if sym:
                            spons_map[sym] = {
                                "grade": str(row_d.get("sponsorship_grade") or "NR") or "NR",
                                "inst_change": float(row_d.get("inst_change_qoq") or 0.0),
                                "inst_pct": float(row_d.get("total_inst_pct") or 0.0),
                            }
            except Exception:
                pass
    except Exception:
        pass

    return {"funda_map": funda_map, "spons_map": spons_map, "funda_as_of": funda_as_of}


def _compute_50d_udvr_series(close_df: pd.DataFrame, vol_df: pd.DataFrame) -> pd.Series:
    """Vectorized 50-session Up/Down Volume Ratio across columns (D9)."""
    if close_df is None or close_df.empty or vol_df is None or vol_df.empty:
        return pd.Series(dtype=float)
    common_cols = [c for c in close_df.columns if c in vol_df.columns]
    if not common_cols:
        return pd.Series(dtype=float)
    c_sub = close_df[common_cols].tail(51)
    v_sub = vol_df[common_cols].reindex(c_sub.index).fillna(0.0)
    if len(c_sub) < 2:
        return pd.Series(1.0, index=common_cols, dtype=float)
    diff = c_sub.diff().iloc[1:]
    v_tail = v_sub.iloc[1:]
    up_v = v_tail.where(diff > 0, 0.0).sum(axis=0)
    dn_v = v_tail.where(diff < 0, 0.0).sum(axis=0)
    udvr = up_v / dn_v.replace(0.0, np.nan)
    fallback = np.where(up_v > 0, 2.0, 1.0)
    return udvr.fillna(pd.Series(fallback, index=common_cols)).round(2)


def _format_tv_for_market(ticker: str, raw_ticker: str, is_us: bool, us_tv_map: dict | None = None) -> tuple[str, str]:
    try:
        from industry_group_engine import format_tradingview_symbol
        sym = format_tradingview_symbol(
            ticker=ticker,
            raw_ticker=raw_ticker,
            is_us=is_us,
            us_tv_map=us_tv_map,
            prefix_us=True,
        )
    except Exception:
        clean = _normalize_ticker_key(ticker or raw_ticker)
        if is_us:
            sym = clean.replace("-", ".")
        else:
            ex = "BSE" if str(raw_ticker).upper().endswith(".BO") else "NSE"
            sym = f"{ex}:{clean}"
    base = "https://www.tradingview.com/chart/?symbol=" if is_us else "https://in.tradingview.com/chart/?symbol="
    return sym, f"{base}{sym}"


@st.cache_data(ttl=3600, show_spinner=False)
def _load_us_tv_exchange_map_cached(db_sig: tuple[str, str]) -> dict:
    out = {}
    try:
        from database import read_sql_df

        df_sm = read_sql_df(
            "SELECT yf_ticker, exchange FROM security_master WHERE market='US' AND status='ACTIVE' AND exchange IS NOT NULL"
        )
        if df_sm is not None and not df_sm.empty:
            for sym, exch in zip(df_sm["yf_ticker"].tolist(), df_sm["exchange"].tolist()):
                su = str(sym or "").strip().upper()
                ex = str(exch or "").strip().upper()
                if su and ex:
                    tv_clean = su.replace("-", ".")
                    out[su] = f"{ex}:{tv_clean}"
                    out[tv_clean] = f"{ex}:{tv_clean}"
    except Exception:
        pass
    return out


# -----------------------------------------------------------------------------
# THRUSTING INDUSTRIES ENGINE (SHARED CORE + CACHED WRAPPER)
# -----------------------------------------------------------------------------
def _empty_thrust_payload(note: str = "") -> dict:
    return {
        "thrust_df": pd.DataFrame(),
        "all_thrust_df": pd.DataFrame(),
        "distribution_df": pd.DataFrame(),
        "ledger": {"grid_up4_total": 0, "attributed": 0, "illiquid": 0, "quarantined": 0, "unmapped": 0},
        "summary": {
            "total_up4_today": 0,
            "total_dn4_today": 0,
            "valid_universe_cnt": 0,
            "top3_share_pct": 0.0,
            "top3_names": [],
            "headline_msg": note or "No local Stockbee matrix history available.",
            "used_fallback_note": "",
            "t2108": None,
            "pulse": "⚖️ BALANCED",
        },
        "as_of": {"prices_date": "—", "groups_date": "—", "igm_stale": False, "is_provisional": False},
    }


def _project_thrust_bundle_for_mode(
    bundle: dict,
    mode: str = "5D_AND_TODAY",
    top_n: int = 6,
) -> dict:
    """Re-sort/slice a Stockbee thrust bundle in-memory for the active mode pill (<1ms)."""
    if not isinstance(bundle, dict):
        return _empty_thrust_payload()
    full_df = bundle.get("all_thrust_df")
    if not isinstance(full_df, pd.DataFrame) or full_df.empty:
        return _empty_thrust_payload((bundle.get("summary") or {}).get("headline_msg", ""))

    work_df = full_df.copy()
    dist_df = bundle.get("distribution_df")
    dist_copy = dist_df.copy() if isinstance(dist_df, pd.DataFrame) else pd.DataFrame()
    ledger = dict(bundle.get("ledger") or {})
    summary_in = dict(bundle.get("summary") or {})
    as_of_in = dict(bundle.get("as_of") or {})

    t0_str = str(as_of_in.get("prices_date") or "—")
    is_provisional = bool(as_of_in.get("is_provisional", False))
    grid_up4_total = int(summary_in.get("total_up4_today", ledger.get("grid_up4_total", 0)) or 0)
    attributed_cnt = int(ledger.get("attributed", 0) or 0)

    used_fallback_note = ""
    mode_key = str(mode or "5D_AND_TODAY").upper().strip()
    if mode_key == "TODAY_4PCT":
        today_sub = work_df[work_df["Up4_Today_Cnt"] > 0].copy()
        if not today_sub.empty:
            sorted_df = today_sub.sort_values(
                by=["Up4_Today_S2_Cnt", "Up4_Today_Cnt", "Thrust_Score"], ascending=[False, False, False]
            )
        else:
            used_fallback_note = "0 attributed stocks up >= 4% today — showing Trailing 5D Thrust Clusters instead."
            sorted_df = work_df.sort_values(by=["Thrust_Score", "Up4_5D_Unique"], ascending=[False, False])
    elif mode_key == "QTR_25PCT":
        q_sub = work_df[work_df["Up25_Qtr_Cnt"] > 0].copy()
        sorted_df = (q_sub if not q_sub.empty else work_df).sort_values(
            by=["Up25_Qtr_Cnt", "Breadth_RS80_Pct", "Thrust_Score"], ascending=[False, False, False]
        )
    elif mode_key == "NEW_HIGHS":
        nh_sub = work_df[work_df["NewHigh_5D_Unique"] > 0].copy()
        sorted_df = (nh_sub if not nh_sub.empty else work_df).sort_values(
            by=["NewHigh_5D_Unique", "NewHigh_Today_Cnt", "Thrust_Score"], ascending=[False, False, False]
        )
    else:
        sorted_df = work_df.sort_values(by=["Thrust_Score", "Up4_5D_Unique", "NewHigh_5D_Unique"], ascending=[False, False, False])

    top_df = sorted_df.head(max(1, int(top_n))).reset_index(drop=True)
    top3_df = top_df.head(3)
    top3_names = top3_df["Industry_Group"].tolist()
    top3_share = round(float(top3_df["Up4_Today_Share_Pct"].sum()), 1) if attributed_cnt > 0 else 0.0
    if is_provisional:
        headline = f"⏱ INTRADAY PROVISIONAL ({t0_str}): thrust verdicts suppressed until market close; active clusters: {', '.join(top3_names[:3])}."
    elif attributed_cnt > 0:
        headline = f"Today's +{grid_up4_total} buying impulse ({attributed_cnt} attributed) concentrates {top3_share:.0f}% in top 3 groups: {', '.join(top3_names[:3])}."
    else:
        headline = f"Calm/pullback session ({t0_str}): 5D thrust & new-high leadership anchors in {', '.join(top3_names[:3])}."

    new_summary = dict(summary_in)
    new_summary.update(
        {
            "total_up4_today": grid_up4_total,
            "top3_share_pct": top3_share,
            "top3_names": top3_names,
            "headline_msg": headline,
            "used_fallback_note": used_fallback_note,
        }
    )

    return {
        "thrust_df": top_df,
        "all_thrust_df": work_df.sort_values("Thrust_Score", ascending=False).reset_index(drop=True),
        "distribution_df": dist_copy,
        "ledger": ledger,
        "summary": new_summary,
        "as_of": as_of_in,
    }


def _normalize_thrust_scope_slug(mkt: str, scope_key: str) -> list[str]:
    """Return ordered list of cloud_json_cache keys for Stockbee thrust attribution."""
    m_low = "us" if str(mkt or "IN").upper().startswith("US") else "in"
    sc_u = str(scope_key or "DEFAULT").upper().strip()
    raw_slug = re.sub(r"[^a-z0-9]+", "_", sc_u.lower()).strip("_")
    keys = []
    if m_low == "us":
        if "NASDAQ" in sc_u or "NDX" in sc_u:
            keys.append("stockbee_thrust_us_nasdaq_100")
        elif "BROAD" in sc_u or "ALL" in sc_u or "IXIC" in sc_u:
            keys.append("stockbee_thrust_us_broad")
        elif "SP500" in sc_u or "S&P" in sc_u or "GSPC" in sc_u:
            keys.append("stockbee_thrust_us_sp500")
        if raw_slug:
            keys.append(f"stockbee_thrust_us_{raw_slug}")
        keys.extend(["stockbee_thrust_us_broad", "stockbee_thrust_us_sp500", "stockbee_thrust_us_nasdaq_100"])
    else:
        if "FULL" in sc_u or "ALL" in sc_u:
            keys.append("stockbee_thrust_in_full")
        elif "NTM" in sc_u or "750" in sc_u:
            keys.append("stockbee_thrust_in_ntm750")
        if raw_slug:
            keys.append(f"stockbee_thrust_in_{raw_slug}")
        keys.extend(["stockbee_thrust_in_ntm750", "stockbee_thrust_in_full"])
    dedup = []
    for k in keys:
        if k not in dedup:
            dedup.append(k)
    return dedup


def _compute_thrusting_industries_core(
    *,
    close_df: pd.DataFrame,
    high_df: pd.DataFrame | None,
    low_df: pd.DataFrame | None,
    vol_df: pd.DataFrame | None,
    stock_intel: dict,
    matrix_df: pd.DataFrame | None,
    niche_df: pd.DataFrame | None = None,
    market: str = "IN",
    mode: str = "5D_AND_TODAY",
    top_n: int = 6,
    lookback_days: int = 20,
    completed_session_date: str | None = None,
    igm_stale: bool = False,
    groups_as_of: str = "—",
    us_tv_map: dict | None = None,
) -> dict:
    from components import _prepare_stockbee_work_frame

    is_us = str(market or "IN").strip().upper().startswith("US")
    work, daily_ret, valid_ret_mask, _ = _prepare_stockbee_work_frame(
        close_df, lookback_days=lookback_days
    )
    if work.empty or not bool(valid_ret_mask.any()):
        return _empty_thrust_payload("Insufficient cached OHLCV sessions to attribute Stockbee thrusts.")

    valid_work = work.loc[valid_ret_mask]
    ret_valid = daily_ret.loc[valid_ret_mask]
    if len(ret_valid) < 1:
        return _empty_thrust_payload()

    cols = list(valid_work.columns)
    idx = valid_work.index
    t0_ts = idx[-1]
    t0_str = t0_ts.strftime("%Y-%m-%d")

    is_provisional = False
    if completed_session_date and t0_str > str(completed_session_date):
        is_provisional = True

    # Align auxiliary panels strictly onto valid_work index/cols
    def _align_sub(panel: pd.DataFrame | None, fallback: pd.DataFrame) -> pd.DataFrame:
        if panel is None or not isinstance(panel, pd.DataFrame) or panel.empty:
            return fallback
        p = panel.copy()
        p.index = pd.to_datetime(p.index, errors="coerce")
        if getattr(p.index, "tz", None) is not None:
            p.index = p.index.tz_localize(None)
        p = p.loc[p.index.notna()]
        p = p[~p.index.duplicated(keep="last")]
        return p.reindex(index=idx, columns=cols).apply(pd.to_numeric, errors="coerce")

    high_w = _align_sub(high_df, valid_work)
    low_w = _align_sub(low_df, valid_work)
    has_vol_panel = (
        vol_df is not None
        and isinstance(vol_df, pd.DataFrame)
        and not vol_df.empty
        and bool(vol_df.reindex(index=idx, columns=cols).notna().any().any())
    )
    vol_w = _align_sub(vol_df, pd.DataFrame(np.nan, index=idx, columns=cols)) if has_vol_panel else pd.DataFrame(np.nan, index=idx, columns=cols)

    # Latest Stockbee summary scalars (matching components.compute_stockbee_monitor_table)
    ret_t0 = ret_valid.iloc[-1]
    valid_today_mask = ret_t0.notna()
    valid_universe_cnt = int(valid_today_mask.sum())
    sma40 = valid_work.rolling(40, min_periods=20).mean().iloc[-1]
    abv40_elig = (valid_work.iloc[-1].notna() & sma40.notna()).sum()
    t2108_val = round(float((valid_work.iloc[-1] > sma40).sum() / abv40_elig * 100.0), 1) if abv40_elig > 0 else None

    qtr_ret = valid_work.pct_change(63, fill_method=None)
    qtr_t0 = qtr_ret.iloc[-1] if len(qtr_ret) >= 1 else pd.Series(np.nan, index=cols)

    # Quarantine & noise guards (D6)
    win_5 = min(5, len(ret_valid))
    ret_5d = ret_valid.iloc[-win_5:]
    ca_mask_all = ret_valid.abs() > 0.35
    ca_t0 = ret_t0.abs() > 0.35
    ca_63d = ca_mask_all.tail(min(63, len(ca_mask_all))).any(axis=0)

    if not is_us and high_df is not None and low_df is not None and not high_w.empty and not low_w.empty:
        circ_mask_all = (high_w == low_w) & (high_w == valid_work) & (ret_valid >= 0.048)
    else:
        circ_mask_all = pd.DataFrame(False, index=idx, columns=cols)
    circ_t0 = circ_mask_all.iloc[-1].fillna(False)
    quarantine_t0 = (ca_t0 | circ_t0).fillna(False)
    quarantine_5d = (ca_mask_all.iloc[-win_5:] | circ_mask_all.iloc[-win_5:]).fillna(False)

    # Liquidity & RVOL (D6 + C3)
    adtv_floor = 5.0 if is_us else 2.5
    if has_vol_panel:
        dollar_vol = (valid_work * vol_w).tail(20).mean(axis=0)
        adtv_20 = dollar_vol / (1e6 if is_us else 1e7)
        vol_sma50 = vol_w.rolling(50, min_periods=20).mean().iloc[-1]
        rvol_t0 = (vol_w.iloc[-1] / vol_sma50.replace(0.0, np.nan)).fillna(1.0)
        is_illiquid = (adtv_20 < adtv_floor) | adtv_20.isna()
    else:
        adtv_20 = pd.Series(np.nan, index=cols)
        rvol_t0 = pd.Series(1.0, index=cols)
        is_illiquid = pd.Series(False, index=cols)

    # Stage 2 + 52W Highs (D11 + D12)
    is_s2_s, abv50_s = classify_stage2_series(valid_work)
    bars_valid_s = valid_work.notna().sum(axis=0)

    win_nh = min(252, len(high_w))
    high_252 = high_w.rolling(win_nh, min_periods=min(126, win_nh)).max()
    nh_raw_5d = (valid_work.iloc[-win_5:] >= (high_252.iloc[-win_5:] * 0.998)) & (~quarantine_5d)
    nh_t0 = (valid_work.iloc[-1] >= (high_252.iloc[-1] * 0.998)) & (~quarantine_t0)
    nh_5d_any = nh_raw_5d.any(axis=0)

    # Exact Stockbee Grid Up +4% movers at T0
    grid_up4_cols = list(ret_t0[ret_t0 >= 0.04].index)
    grid_up4_total = len(grid_up4_cols)
    ledger = {"grid_up4_total": grid_up4_total, "attributed": 0, "illiquid": 0, "quarantined": 0, "unmapped": 0}

    # Pre-index matrix & niche metadata
    matrix_lookup: dict[str, dict] = {}
    if matrix_df is not None and not matrix_df.empty and "Industry_Group" in matrix_df.columns:
        for _, mrow in matrix_df.iterrows():
            matrix_lookup[str(mrow["Industry_Group"])] = mrow.to_dict()

    niche_set: set[str] = set()
    if niche_df is not None and not niche_df.empty and "Industry_Group" in niche_df.columns:
        niche_set = {str(x) for x in niche_df["Industry_Group"].dropna().tolist()}

    # Partition grid_up4_cols into exact mutually exclusive ledger buckets (D3)
    col_intel: dict[str, dict | None] = {c: _resolve_intel_lookup(stock_intel, str(c)) for c in cols}
    for c in grid_up4_cols:
        if bool(quarantine_t0.get(c, False)):
            ledger["quarantined"] += 1
        elif bool(is_illiquid.get(c, False)):
            ledger["illiquid"] += 1
        elif col_intel.get(c) is None:
            ledger["unmapped"] += 1
        else:
            ledger["attributed"] += 1

    # Scoped deduplication per canonical symbol (prevents .NS/.BO or share-class twin double-count)
    group_buckets: dict[str, dict[str, Any]] = {}
    seen_scoped_norm: dict[str, set[str]] = {}
    seen_up4_norm: dict[str, set[str]] = {}
    seen_5d_norm: dict[str, set[str]] = {}
    seen_nh_t0_norm: dict[str, set[str]] = {}
    seen_nh_norm: dict[str, set[str]] = {}
    seen_qtr_norm: dict[str, set[str]] = {}
    seen_dn4_norm: dict[str, set[str]] = {}

    clean_5d_up4 = (ret_5d >= 0.04) & (~quarantine_5d)
    hits_5d_s = clean_5d_up4.sum(axis=0)

    for c in cols:
        intel = col_intel.get(c)
        if intel is None:
            continue
        grp = str(intel.get("group_name") or "").strip()
        if not grp or grp in _EXCLUDED_GROUPS:
            continue

        norm_tk = _normalize_ticker_key(str(c))
        bkt = group_buckets.setdefault(
            grp,
            {
                "valid_cnt": 0,
                "up4_today": 0,
                "up4_today_s2": 0,
                "up4_5d_unique": 0,
                "up4_5d_s2_unique": 0,
                "up4_5d_hits": 0,
                "up25_qtr": 0,
                "nh_today": 0,
                "nh_5d_unique": 0,
                "vol_conf": 0,
                "dn4_today": 0,
                "tickers": [],
                "sell_tickers": [],
            },
        )

        # Scoped valid denominator (D2)
        if bool(valid_today_mask.get(c, False)):
            ss = seen_scoped_norm.setdefault(grp, set())
            if norm_tk not in ss:
                ss.add(norm_tk)
                bkt["valid_cnt"] += 1

        # Skip illiquid or T0-quarantined from thrust signals
        if bool(is_illiquid.get(c, False)):
            continue

        r1d_val = float(ret_t0.get(c, np.nan))
        is_up4_t0 = bool(pd.notna(r1d_val) and r1d_val >= 0.04 and not bool(quarantine_t0.get(c, False)))
        is_dn4_t0 = bool(pd.notna(r1d_val) and r1d_val <= -0.04 and not bool(ca_t0.get(c, False)))
        h5d = int(hits_5d_s.get(c, 0))
        qtr_val = float(qtr_t0.get(c, np.nan))
        is_q25 = bool(pd.notna(qtr_val) and qtr_val >= 0.25 and not bool(ca_63d.get(c, False)))
        is_nh_t0_c = bool(nh_t0.get(c, False))
        is_nh_5d_c = bool(nh_5d_any.get(c, False))
        is_s2_c = bool(is_s2_s.get(c, False))
        is_ipo_c = int(bars_valid_s.get(c, 0)) < 200
        rv_val = float(rvol_t0.get(c, 1.0) or 1.0)
        is_vconf = bool(is_up4_t0 and has_vol_panel and rv_val >= 1.4)

        if is_up4_t0:
            su = seen_up4_norm.setdefault(grp, set())
            if norm_tk not in su:
                su.add(norm_tk)
                bkt["up4_today"] += 1
                if is_s2_c:
                    bkt["up4_today_s2"] += 1
                if is_vconf:
                    bkt["vol_conf"] += 1

        if h5d > 0:
            s5 = seen_5d_norm.setdefault(grp, set())
            if norm_tk not in s5:
                s5.add(norm_tk)
                bkt["up4_5d_unique"] += 1
                if is_s2_c:
                    bkt["up4_5d_s2_unique"] += 1
                bkt["up4_5d_hits"] += h5d

        if is_q25:
            sq = seen_qtr_norm.setdefault(grp, set())
            if norm_tk not in sq:
                sq.add(norm_tk)
                bkt["up25_qtr"] += 1

        if is_nh_t0_c:
            sn0 = seen_nh_t0_norm.setdefault(grp, set())
            if norm_tk not in sn0:
                sn0.add(norm_tk)
                bkt["nh_today"] += 1
        if is_nh_5d_c:
            sn = seen_nh_norm.setdefault(grp, set())
            if norm_tk not in sn:
                sn.add(norm_tk)
                bkt["nh_5d_unique"] += 1

        if is_dn4_t0:
            sd = seen_dn4_norm.setdefault(grp, set())
            if norm_tk not in sd:
                sd.add(norm_tk)
                bkt["dn4_today"] += 1
                tv_sym, tv_url = _format_tv_for_market(norm_tk, str(c), is_us=is_us, us_tv_map=us_tv_map)
                bkt["sell_tickers"].append(
                    {"ticker": norm_tk, "ret_1d_pct": round(r1d_val * 100.0, 1), "tv_symbol": tv_sym, "tv_url": tv_url}
                )

        if is_up4_t0 or h5d > 0 or is_nh_5d_c or is_q25:
            tv_sym, tv_url = _format_tv_for_market(norm_tk, str(c), is_us=is_us, us_tv_map=us_tv_map)
            bkt["tickers"].append(
                {
                    "ticker": norm_tk,
                    "raw_ticker": str(c),
                    "ret_1d_pct": round(r1d_val * 100.0, 1) if pd.notna(r1d_val) else 0.0,
                    "hits_5d": h5d,
                    "ret_qtr_pct": round(qtr_val * 100.0, 1) if pd.notna(qtr_val) else 0.0,
                    "is_today_up4": is_up4_t0,
                    "is_qtr_up25": is_q25,
                    "is_nh_5d": is_nh_5d_c,
                    "is_stage2": is_s2_c,
                    "is_ipo": is_ipo_c,
                    "rvol": round(rv_val, 2),
                    "is_vol_confirmed": is_vconf,
                    "tv_symbol": tv_sym,
                    "tv_url": tv_url,
                }
            )

    min_unique_req = 1 if valid_universe_cnt <= 180 else 2
    total_attr_up4 = max(1, ledger["attributed"])

    rows = []
    dist_rows = []
    for grp, bkt in group_buckets.items():
        g_valid = max(1, bkt["valid_cnt"])
        mrow = matrix_lookup.get(grp, {})
        rank_td = int(mrow.get("Rank_Today", 999) or 999)
        is_niche = grp in niche_set or rank_td == 999
        grade_badge = f"Niche (<4)" if is_niche else str(mrow.get("Group_Grade_Badge") or f"#{rank_td}")
        delta_1m = int(mrow.get("Delta_1M", 0) or 0)
        b_sma50 = float(mrow.get("Breadth_50SMA_Pct", 0.0) or 0.0)
        b_rs80 = float(mrow.get("Breadth_RS80_Pct", 0.0) or 0.0)
        rsnh_pct = float(mrow.get("RSNH_Pct", 0.0) or 0.0)
        funda_pct = float(mrow.get("CANSLIM_Funda_Pct", 0.0) or 0.0)
        is_top20 = bool(mrow.get("Is_Top_20_Pct", False))
        is_dual_alpha = bool(mrow.get("Is_Dual_Alpha", False))
        is_grp_conf = bool(mrow.get("Is_Group_Confirmed", False))
        full_uni_cnt = int(mrow.get("Tradable_Count", bkt["valid_cnt"]) or bkt["valid_cnt"])

        if bkt["dn4_today"] >= max(1, min_unique_req):
            bkt["sell_tickers"].sort(key=lambda x: x["ret_1d_pct"])
            dist_rows.append(
                {
                    "Industry_Group": grp,
                    "Rank_Today": rank_td,
                    "Group_Grade_Badge": grade_badge,
                    "Dn4_Today_Cnt": bkt["dn4_today"],
                    "Group_Valid_Cnt": bkt["valid_cnt"],
                    "Selling_Tickers": bkt["sell_tickers"][:4],
                }
            )

        has_activity = (
            bkt["up4_today"] >= 1
            or bkt["up4_5d_unique"] >= min_unique_req
            or bkt["nh_5d_unique"] >= min_unique_req
            or bkt["up25_qtr"] >= min_unique_req
        )
        if not has_activity:
            continue

        up4_t_nons2 = max(0, bkt["up4_today"] - bkt["up4_today_s2"])
        raw_energy = (
            3.5 * bkt["up4_today_s2"]
            + 1.0 * up4_t_nons2
            + 2.2 * bkt["up4_5d_s2_unique"]
            + 0.35 * bkt["up4_5d_hits"]
            + 0.9 * bkt["up25_qtr"]
            + 2.0 * bkt["nh_5d_unique"]
            + 0.8 * bkt["vol_conf"]
        )
        scaled_energy = raw_energy / math.sqrt(g_valid)
        density_5d = bkt["up4_5d_unique"] / float(g_valid)

        # Deduplicate & sort tickers cleanly
        seen_t_card = set()
        uniq_tickers = []
        for t_item in sorted(
            bkt["tickers"],
            key=lambda x: (x["is_today_up4"], x["ret_1d_pct"], x["hits_5d"], x["is_nh_5d"], x["ret_qtr_pct"]),
            reverse=True,
        ):
            if t_item["ticker"] not in seen_t_card:
                seen_t_card.add(t_item["ticker"])
                uniq_tickers.append(t_item)

        if bkt["up4_5d_unique"] > 0:
            s2_share = bkt["up4_5d_s2_unique"] / float(bkt["up4_5d_unique"])
        elif bkt["up4_today"] > 0:
            s2_share = bkt["up4_today_s2"] / float(bkt["up4_today"])
        else:
            s2_share = b_sma50 / 100.0

        is_bounce = bool(s2_share < 0.40 and b_sma50 < 35.0)
        if is_bounce:
            status_tag = "🩹 Counter-Trend Bounce (<50SMA)"
        elif (not is_niche) and rank_td <= 40 and s2_share >= 0.50 and (bkt["up4_today"] >= 2 or bkt["up4_5d_unique"] >= 3):
            status_tag = "👑 Top-40 S2 Thrust"
        elif rank_td > 40 and delta_1m >= 10 and s2_share >= 0.50:
            status_tag = "🌱 Emerging S2 Rotation"
        elif density_5d >= 0.25 and bkt["up4_5d_unique"] >= 2 and s2_share >= 0.50:
            status_tag = "🚀 Pack Surge"
        else:
            status_tag = "⚡ Active"

        rows.append(
            {
                "Industry_Group": grp,
                "Rank_Today": rank_td,
                "Group_Grade_Badge": grade_badge,
                "Is_Niche": is_niche,
                "Is_Top_20_Pct": is_top20,
                "Is_Dual_Alpha": is_dual_alpha,
                "Is_Group_Confirmed": is_grp_conf,
                "Delta_1M": delta_1m,
                "Group_Valid_Cnt": bkt["valid_cnt"],
                "Full_Universe_Cnt": full_uni_cnt,
                "Up4_Today_Cnt": bkt["up4_today"],
                "Up4_Today_S2_Cnt": bkt["up4_today_s2"],
                "Up4_Today_Share_Pct": round(bkt["up4_today"] / float(total_attr_up4) * 100.0, 1) if ledger["attributed"] > 0 else 0.0,
                "Up4_5D_Unique": bkt["up4_5d_unique"],
                "Up4_5D_S2_Unique": bkt["up4_5d_s2_unique"],
                "Up4_5D_Hits": bkt["up4_5d_hits"],
                "Up25_Qtr_Cnt": bkt["up25_qtr"],
                "NewHigh_Today_Cnt": bkt["nh_today"],
                "NewHigh_5D_Unique": bkt["nh_5d_unique"],
                "Vol_Confirmed_Cnt": bkt["vol_conf"],
                "Dn4_Today_Cnt": bkt["dn4_today"],
                "Breadth_50SMA_Pct": b_sma50,
                "Breadth_RS80_Pct": b_rs80,
                "RSNH_Pct": rsnh_pct,
                "CANSLIM_Funda_Pct": funda_pct,
                "Thrust_Density_Pct": round(density_5d * 100.0, 1),
                "S2_Share": round(s2_share, 2),
                "Is_Counter_Trend_Bounce": is_bounce,
                "Thrust_Status_Tag": status_tag,
                "_scaled_energy": scaled_energy,
                "_density_5d": density_5d,
                "Thrusting_Tickers": uniq_tickers[:4],
                "All_Thrusting_Tickers": uniq_tickers,
            }
        )

    full_df = pd.DataFrame(rows)
    if full_df.empty:
        return _empty_thrust_payload("No canonical industry groups met the minimum thrust threshold.")

    # Size-normalized percentile Thrust Score (D13) + Counter-trend damping (C2)
    pct_energy = full_df["_scaled_energy"].rank(pct=True) * 100.0
    pct_density = full_df["_density_5d"].rank(pct=True) * 100.0
    raw_score = 0.5 * pct_energy + 0.5 * pct_density
    mult = np.where(full_df["Is_Counter_Trend_Bounce"], 0.55, np.where(full_df["Is_Top_20_Pct"], 1.10, 1.0))
    full_df["Thrust_Score"] = (raw_score * mult).round(1)
    full_df = full_df.drop(columns=["_scaled_energy", "_density_5d"])

    return _project_thrust_bundle_for_mode(
        {
            "all_thrust_df": full_df.sort_values("Thrust_Score", ascending=False).reset_index(drop=True),
            "distribution_df": pd.DataFrame(dist_rows).sort_values("Dn4_Today_Cnt", ascending=False).head(4).reset_index(drop=True) if dist_rows else pd.DataFrame(),
            "ledger": ledger,
            "summary": {
                "total_up4_today": grid_up4_total,
                "total_dn4_today": int((ret_t0 <= -0.04).sum()),
                "valid_universe_cnt": valid_universe_cnt,
                "t2108": t2108_val,
            },
            "as_of": {
                "prices_date": t0_str,
                "groups_date": groups_as_of,
                "igm_stale": bool(igm_stale),
                "is_provisional": is_provisional,
            },
        },
        mode=mode,
        top_n=top_n,
    )


def _load_panels_for_stockbee(market: str, scope_key: str) -> tuple[dict, str]:
    from components import _stage_matrix_path_for_market, _stockbee_allowed_tickers_for_scope
    from price_history_manager import extract_market_ohlcv_panels, get_matrix_cache_signature, load_cached_price_matrix

    mkt = "US" if str(market).upper().startswith("US") else "IN"
    matrix_path = _stage_matrix_path_for_market(mkt)
    if not os.path.exists(matrix_path):
        return {}, ""
    sig = get_matrix_cache_signature(matrix_path)
    raw_df = load_cached_price_matrix(matrix_path, cache_sig=sig)
    if raw_df is None or raw_df.empty:
        return {}, ""
    allowed = _stockbee_allowed_tickers_for_scope(mkt, scope_key)
    panels = extract_market_ohlcv_panels(
        raw_df,
        market=mkt,
        allowed_tickers=allowed,
        include_etfs=False,
        deduplicate_dual_in=(mkt != "US"),
        min_coverage_ratio=0.35,
        min_abs=20,
    )
    return panels, matrix_path


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_thrusting_industries_loader(
    market: str,
    scope_key: str,
    mode: str,
    top_n: int,
    lookback_days: int,
    matrix_sig: tuple,
    igm_matrix_sig: tuple,
    igm_intel_sig: tuple,
    db_sig: tuple[str, str],
) -> dict:
    from industry_group_engine import (
        CACHE_MATRIX_INDIA_PATH,
        CACHE_MATRIX_US_PATH,
        get_stock_group_intelligence,
        is_industry_cache_stale,
        load_cached_industry_group_matrix,
        load_cached_niche_industry_groups,
    )
    from price_history_manager import get_latest_completed_session_date

    is_us = str(market).upper().startswith("US")
    igm_scope = "US (S&P 500 / Nasdaq)" if is_us else "India (NSE/BSE)"
    panels, _ = _load_panels_for_stockbee(market, scope_key)
    if not panels or panels.get("close") is None or panels["close"].empty:
        return _empty_thrust_payload("Price matrix cache unavailable for Stockbee thrust attribution.")

    stock_intel = get_stock_group_intelligence(universe_scope=igm_scope)
    mat_df = load_cached_industry_group_matrix(universe_scope=igm_scope)
    mat_df = mat_df.copy() if isinstance(mat_df, pd.DataFrame) and not mat_df.empty else pd.DataFrame()
    niche_df = load_cached_niche_industry_groups(universe_scope=igm_scope)
    niche_df = niche_df.copy() if isinstance(niche_df, pd.DataFrame) and not niche_df.empty else pd.DataFrame()

    mpath = CACHE_MATRIX_US_PATH if is_us else CACHE_MATRIX_INDIA_PATH
    groups_dt = datetime.fromtimestamp(os.path.getmtime(mpath)).strftime("%Y-%m-%d") if os.path.exists(mpath) else "—"
    igm_stale = is_industry_cache_stale(igm_scope)
    completed_dt = get_latest_completed_session_date("US" if is_us else "IN")
    us_tv_map = _load_us_tv_exchange_map_cached(db_sig) if is_us else None

    return _compute_thrusting_industries_core(
        close_df=panels["close"],
        high_df=panels.get("high"),
        low_df=panels.get("low"),
        vol_df=panels.get("volume"),
        stock_intel=stock_intel,
        matrix_df=mat_df,
        niche_df=niche_df,
        market="US" if is_us else "IN",
        mode=mode,
        top_n=top_n,
        lookback_days=lookback_days,
        completed_session_date=completed_dt,
        igm_stale=igm_stale,
        groups_as_of=groups_dt,
        us_tv_map=us_tv_map,
    )


def compute_stockbee_thrusting_industries(
    market: str = "IN",
    universe_scope: str = "NTM750",
    mode: str = "5D_AND_TODAY",
    top_n: int = 6,
    lookback_days: int = 20,
) -> dict:
    """Public cached entry point for Stockbee Thrusting Industries (India & US)."""
    from components import _stage_matrix_path_for_market
    from industry_group_engine import (
        CACHE_INTEL_INDIA_PATH,
        CACHE_INTEL_US_PATH,
        CACHE_MATRIX_INDIA_PATH,
        CACHE_MATRIX_US_PATH,
        _get_file_signature,
    )
    from price_history_manager import get_matrix_cache_signature

    mkt = "US" if str(market or "IN").upper().startswith("US") else "IN"
    scope_key = str(universe_scope or "DEFAULT").upper().strip()
    m_path = _stage_matrix_path_for_market(mkt)
    m_sig = get_matrix_cache_signature(m_path) if os.path.exists(m_path) else (0, 0)
    igm_mat_path = CACHE_MATRIX_US_PATH if mkt == "US" else CACHE_MATRIX_INDIA_PATH
    igm_int_path = CACHE_INTEL_US_PATH if mkt == "US" else CACHE_INTEL_INDIA_PATH
    db_sig = _get_db_freshness_sig()

    if os.path.exists(m_path):
        res = _cached_thrusting_industries_loader(
            market=mkt,
            scope_key=scope_key,
            mode=str(mode or "5D_AND_TODAY"),
            top_n=int(top_n),
            lookback_days=int(lookback_days),
            matrix_sig=m_sig,
            igm_matrix_sig=_get_file_signature(igm_mat_path),
            igm_intel_sig=_get_file_signature(igm_int_path),
            db_sig=db_sig,
        )
    else:
        res = _empty_thrust_payload()

    all_df_chk = res.get("all_thrust_df") if isinstance(res, dict) else None
    if all_df_chk is None or not isinstance(all_df_chk, pd.DataFrame) or all_df_chk.empty:
        try:
            from database import load_cloud_json_cache_cached
            for ck in _normalize_thrust_scope_slug(mkt, scope_key):
                cloud_bundle, _ = load_cloud_json_cache_cached(ck)
                if (
                    isinstance(cloud_bundle, dict)
                    and isinstance(cloud_bundle.get("all_thrust_df"), pd.DataFrame)
                    and not cloud_bundle["all_thrust_df"].empty
                ):
                    return _project_thrust_bundle_for_mode(
                        cloud_bundle,
                        mode=str(mode or "5D_AND_TODAY"),
                        top_n=int(top_n),
                    )
        except Exception:
            pass

    # Return shallow copy of dict with copied DataFrames so callers never mutate cached state (E2)
    out = dict(res)
    if isinstance(out.get("thrust_df"), pd.DataFrame):
        out["thrust_df"] = out["thrust_df"].copy()
    if isinstance(out.get("all_thrust_df"), pd.DataFrame):
        out["all_thrust_df"] = out["all_thrust_df"].copy()
    if isinstance(out.get("distribution_df"), pd.DataFrame):
        out["distribution_df"] = out["distribution_df"].copy()
    return out


# -----------------------------------------------------------------------------
# CLUSTER-FIRST LEADERSHIP BOARD (APP.PY INDIA & US DESKS)
# -----------------------------------------------------------------------------
def _resolve_m_gate(stage_num: int, pulse: str, t2108: float | None, is_provisional: bool) -> dict:
    pulse_u = str(pulse or "").upper()
    if is_provisional:
        return {
            "code": "PROVISIONAL",
            "tone": "amber",
            "default_lens": "CANSLIM",
            "label": "⏱ M-GATE: INTRADAY SESSION IN PROGRESS — Wait Close Before Sizing",
            "early_rally": False,
        }
    if stage_num in (1, 2) or "SELL" in pulse_u:
        return {
            "code": "RED",
            "tone": "rose",
            "default_lens": "DEFENSE",
            "label": "🛑 M-GATE: WATCHLIST & STALK MODE ONLY — No New Breakout Buys (Track 🔵 Blue Dots)",
            "early_rally": stage_num == 2,
        }
    if stage_num == 3 or (t2108 is not None and t2108 > 72.0):
        extra = " · T2108 Stretched (>72%)" if (t2108 is not None and t2108 > 72.0) else ""
        return {
            "code": "YELLOW",
            "tone": "amber",
            "default_lens": "CANSLIM",
            "label": f"🟡 M-GATE: SELECTIVE MODE{extra} — Pullbacks to 21 EMA / Tight Pivots Only",
            "early_rally": stage_num == 3,
        }
    if stage_num in (4, 5):
        return {
            "code": "GREEN",
            "tone": "emerald",
            "default_lens": "CANSLIM",
            "label": "🟢 M-GATE: PRESS CONFIRMED LEADERS — Target Top-40 Clusters & Tight Entries",
            "early_rally": False,
        }
    return {
        "code": "PENDING",
        "tone": "slate",
        "default_lens": "CANSLIM",
        "label": "⚪ M-GATE: Brain Cache Pending — Run EOD Sync for Regime Gating",
        "early_rally": False,
    }


def _select_cluster_tml_leaders(
    *,
    group_row: dict,
    is_us: bool,
    funda_map: dict,
    spons_map: dict,
    udvr_map: dict,
    thrust_ticker_map: dict,
    held_set: set[str],
    earnings_cache: dict | None,
    brain_score_map: dict,
    us_tv_map: dict | None,
    max_leaders: int = 3,
    min_leader_rs: float = 80.0,
    min_sales_pct: float = 0.0,
) -> tuple[list[dict], int, bool, dict]:
    """
    Scan ALL deduplicated constituents in `Top_Leaders_Full` (T2) so elite TMLs
    deeper than rank 10 are never missed, ranking strict CANSLIM tiers cleanly
    while excluding sympathy laggards (<min_leader_rs), failed/damaged charts,
    and non-zero low-sales stocks (<min_sales_pct when > 0).
    Returns (selected_top3, actionable_cnt, is_extended_group, leader_stats).
    """
    pool = group_row.get("Top_Leaders_Full") or group_row.get("Top_3_Leaders") or []
    candidates = []
    seen_sym = set()
    suppressed_laggards = 0
    suppressed_failed = 0
    suppressed_low_sales = 0

    for item in pool:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("is_tradable_leader", True)):
            continue
        if int(item.get("circuit_days_20d", 0) or 0) >= 3:
            continue

        raw_t = str(item.get("raw_ticker") or item.get("ticker") or "")
        sym = _normalize_ticker_key(str(item.get("ticker") or raw_t))
        if not sym or sym in seen_sym:
            continue
        seen_sym.add(sym)

        rs_v = float(item.get("rs", 0) or 0.0)
        dist_ema = float(item.get("dist_ema", 0.0) or 0.0)
        dist_piv = float(item.get("dist_pivot", 0.0) or 0.0)
        status_str = str(item.get("status") or "⏳ BASE")

        is_failed_chart = ("FAILED" in status_str.upper()) or (dist_ema < -5.0)
        if is_failed_chart:
            suppressed_failed += 1
            continue
        if rs_v < float(min_leader_rs):
            suppressed_laggards += 1
            continue

        is_blue_dot = bool(item.get("is_rs_blue_dot", False))
        is_ipo = bool(item.get("is_short_history", False)) or int(item.get("bars_count", 252) or 252) < 200

        f_row = _lookup_map(funda_map, sym)
        has_funda = has_valid_fundamentals(f_row)
        eps_g = float((f_row or {}).get("eps_growth", 0.0) or 0.0)
        sales_g = float((f_row or {}).get("sales_growth", 0.0) or 0.0)
        roe_v = float((f_row or {}).get("roe", 0.0) or 0.0)

        # Keep unavailable ("Funda: N/A") or explicit 0% sales growth (possible missing/stale feed);
        # hide only when non-zero reported sales growth falls strictly below min_sales_pct.
        if float(min_sales_pct) > 0.0 and has_funda and abs(sales_g) > 1e-6 and sales_g < float(min_sales_pct):
            suppressed_low_sales += 1
            continue

        s_row = _lookup_map(spons_map, sym) if not is_us else None
        sp_grade = str((s_row or {}).get("grade", "NR") or "NR")
        sp_chg = float((s_row or {}).get("inst_change", 0.0) or 0.0)
        udvr_v = float(_lookup_map(udvr_map, raw_t) or _lookup_map(udvr_map, sym) or 1.0)

        is_tml = has_funda and is_tml_confluence(
            eps_growth=eps_g,
            sales_growth=sales_g,
            roe=roe_v,
            sponsorship_grade=sp_grade,
            inst_change_qoq=sp_chg,
            udvr_50d=udvr_v,
        )
        tier_num, tier_badge, tier_tone = classify_leader_tier(rs_v, eps_g, sales_g, roe_v, has_funda, is_tml)
        actionable = bool(
            any(k in status_str for k in ("IN BUY ZONE", "RETEST", "COILING")) or is_blue_dot
        )

        th_info = thrust_ticker_map.get(sym, {})
        held_cands = {c.upper() for c in (_ticker_lookup_candidates(sym) + _ticker_lookup_candidates(raw_t))}
        is_held = bool(held_set and (sym.upper() in held_set or bool(held_cands & held_set)))
        ern_days = _get_near_earnings_days(earnings_cache, sym, raw_t) if is_held else None

        brain_sc = brain_score_map.get(sym)
        tv_sym, tv_url = _format_tv_for_market(sym, raw_t, is_us=is_us, us_tv_map=us_tv_map)

        candidates.append(
            {
                "ticker": sym,
                "raw_ticker": raw_t,
                "role_icon": str(item.get("role_icon") or "⭐"),
                "role_code": str(item.get("role_code") or ""),
                "rs": int(round(rs_v)),
                "dist_ema": round(dist_ema, 1),
                "dist_pivot": round(dist_piv, 1),
                "status": status_str,
                "is_blue_dot": is_blue_dot,
                "is_ipo": is_ipo,
                "actionable": actionable,
                "has_funda": has_funda,
                "eps_growth": round(eps_g, 1),
                "sales_growth": round(sales_g, 1),
                "roe": round(roe_v, 1),
                "sp_grade": sp_grade,
                "sp_chg": round(sp_chg, 2),
                "udvr": udvr_v,
                "tier_num": tier_num,
                "tier_badge": tier_badge,
                "tier_tone": tier_tone,
                "is_up4_today": bool(th_info.get("is_today_up4", False)),
                "ret_1d_pct": float(th_info.get("ret_1d_pct", 0.0) or 0.0),
                "rvol": float(th_info.get("rvol", 1.0) or 1.0),
                "is_held": is_held,
                "ern_days": ern_days,
                "brain_score": int(round(float(brain_sc))) if brain_sc is not None else None,
                "tv_symbol": tv_sym,
                "tv_url": tv_url,
            }
        )

    candidates.sort(
        key=lambda c: (
            c["tier_num"],
            c["actionable"],
            not c["is_ipo"],
            c["rs"],
            -abs(c["dist_ema"]),
        ),
        reverse=True,
    )

    top_5 = candidates[:5]
    ext_cnt = sum(1 for c in top_5 if "EXTENDED" in c["status"])
    is_extended_group = bool(len(top_5) > 0 and (ext_cnt / float(len(top_5))) >= 0.60)
    actionable_cnt = sum(1 for c in candidates[:6] if c["actionable"])

    leader_stats = {
        "suppressed_laggards": suppressed_laggards,
        "suppressed_failed": suppressed_failed,
        "suppressed_low_sales": suppressed_low_sales,
        "qualifying_total": len(candidates),
    }
    return candidates[:max_leaders], actionable_cnt, is_extended_group, leader_stats


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    return int(round(_safe_float(val, float(default))))


def _safe_bool(val: Any, default: bool = False) -> bool:
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    return bool(val)


def _dedup_held_aliases(held_raw_tickers: list[str] | None) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Build canonical order + alias lookup across share-class/exchange variants (.NS/.BO/./-)."""
    canonical_order: list[str] = []
    canon_raw_map: dict[str, str] = {}
    alias_to_held: dict[str, str] = {}
    for raw_h in (held_raw_tickers or []):
        raw_s = str(raw_h or "").strip()
        if not raw_s:
            continue
        existing = None
        for cand in _ticker_lookup_candidates(raw_s):
            cu = cand.upper()
            cn = _normalize_ticker_key(cu)
            if cu in alias_to_held:
                existing = alias_to_held[cu]
                break
            if cn in alias_to_held:
                existing = alias_to_held[cn]
                break
        if existing:
            for cand in _ticker_lookup_candidates(raw_s):
                alias_to_held[cand.upper()] = existing
                alias_to_held[_normalize_ticker_key(cand)] = existing
            continue
        h_norm = _normalize_ticker_key(raw_s)
        if not h_norm:
            continue
        canonical_order.append(h_norm)
        canon_raw_map[h_norm] = raw_s
        for cand in _ticker_lookup_candidates(raw_s):
            alias_to_held[cand.upper()] = h_norm
            alias_to_held[_normalize_ticker_key(cand)] = h_norm
    return canonical_order, canon_raw_map, alias_to_held


def _build_portfolio_triage(
    *,
    matrix_df: pd.DataFrame,
    niche_df: pd.DataFrame | None,
    clusters: list[dict],
    held_raw_tickers: list[str] | None,
    earnings_cache: dict | None,
    is_us: bool,
    us_tv_map: dict | None,
) -> dict:
    """
    Reconcile 100% of held portfolio tickers across Top-N clusters, A+/A/B/C/D/E groups,
    Niche (<4 stock) groups, and Unmapped/ETF tickers with zero network I/O.
    Never hides Tier-0 Illiquid or Circuit-Locked held holdings.
    """
    empty_ledger = {
        "held_total": 0,
        "top_clusters": 0,
        "ab_other": 0,
        "c_groups": 0,
        "de_danger": 0,
        "niche": 0,
        "unmapped": 0,
    }
    canonical_order, canon_raw_map, alias_to_held = _dedup_held_aliases(held_raw_tickers)
    if not canonical_order:
        return {
            "held_total": 0,
            "held_in_clusters": 0,
            "holdings_in_de": 0,
            "de_holdings": [],
            "held_in_clusters_items": [],
            "held_weak_alerts": [],
            "ledger": empty_ledger,
        }

    top_cluster_rank = {str(c["industry_group"]): int(c["board_rank"]) for c in (clusters or [])}
    surfaced_leader_keys = {
        (str(c["industry_group"]), str(l.get("ticker", "")))
        for c in (clusters or [])
        for l in (c.get("leaders") or [])
    }

    def _match_held(sym_a: str, sym_b: str = "") -> str | None:
        for src in (sym_a, sym_b):
            if not src:
                continue
            for cand in _ticker_lookup_candidates(src):
                hit = alias_to_held.get(cand.upper()) or alias_to_held.get(_normalize_ticker_key(cand))
                if hit:
                    return hit
        return None

    # Record best industry hit per canonical held ticker (prefer matrix > niche, rich Top_Leaders_Full > bare Constituents)
    held_hits: dict[str, dict] = {}
    for df_src, src_kind in ((matrix_df, "matrix"), (niche_df, "niche")):
        if not isinstance(df_src, pd.DataFrame) or df_src.empty:
            continue
        for _, r in df_src.iterrows():
            grp = str(r.get("Industry_Group") or "").strip()
            if not grp:
                continue
            rank_td = _safe_int(r.get("Rank_Today"), 0)
            grade_str = str(r.get("Group_Grade") or "").strip()
            if grade_str.lower() == "nan":
                grade_str = ""
            raw_gb = r.get("Group_Grade_Badge")
            grade_badge = str(raw_gb if raw_gb is not None and not (isinstance(raw_gb, float) and math.isnan(raw_gb)) else (grade_str or (f"#{rank_td}" if rank_td > 0 else "Niche")))
            is_danger = _safe_bool(r.get("Is_Danger_Group"), False) or (grade_str in ("D", "E"))
            delta_1m = _safe_int(r.get("Delta_1M"), 0)
            b50 = round(_safe_float(r.get("Breadth_50SMA_Pct"), 0.0), 1)
            src_prio = 2 if src_kind == "matrix" else 1

            raw_leaders_list = r.get("Top_Leaders_Full")
            if not isinstance(raw_leaders_list, (list, tuple)):
                raw_leaders_list = r.get("Top_3_Leaders")
            if not isinstance(raw_leaders_list, (list, tuple)):
                raw_leaders_list = []

            for item in raw_leaders_list:
                if not isinstance(item, dict):
                    continue
                raw_t = str(item.get("raw_ticker") or item.get("ticker") or "")
                sym_t = str(item.get("ticker") or raw_t)
                h_norm = _match_held(sym_t, raw_t)
                if not h_norm:
                    continue
                cand_prio = (src_prio, 2, -rank_td)
                prev = held_hits.get(h_norm)
                if prev is not None and prev["_prio"] >= cand_prio:
                    continue
                rs_raw = item.get("rs")
                ema_raw = item.get("dist_ema")
                held_hits[h_norm] = {
                    "_prio": cand_prio,
                    "source": src_kind,
                    "industry_group": grp,
                    "rank_today": rank_td,
                    "grade_str": grade_str,
                    "grade_badge": grade_badge,
                    "is_danger": is_danger,
                    "delta_1m": delta_1m,
                    "breadth_50sma": b50,
                    "raw_ticker": raw_t or canon_raw_map.get(h_norm, h_norm),
                    "rs": _safe_int(rs_raw) if rs_raw is not None and not (isinstance(rs_raw, float) and math.isnan(rs_raw)) else None,
                    "status": str(item.get("status") or "—"),
                    "dist_ema": round(_safe_float(ema_raw), 1) if ema_raw is not None and not (isinstance(ema_raw, float) and math.isnan(ema_raw)) else None,
                    "is_tradable_leader": _safe_bool(item.get("is_tradable_leader"), True),
                    "circuit_days_20d": _safe_int(item.get("circuit_days_20d"), 0),
                }

            constits_list = r.get("Constituents")
            if not isinstance(constits_list, (list, tuple)):
                constits_list = []
            for t in constits_list:
                t_str = str(t or "")
                h_norm = _match_held(t_str)
                if not h_norm:
                    continue
                cand_prio = (src_prio, 1, -rank_td)
                prev = held_hits.get(h_norm)
                if prev is not None and prev["_prio"] >= cand_prio:
                    continue
                held_hits[h_norm] = {
                    "_prio": cand_prio,
                    "source": src_kind,
                    "industry_group": grp,
                    "rank_today": rank_td,
                    "grade_str": grade_str,
                    "grade_badge": grade_badge,
                    "is_danger": is_danger,
                    "delta_1m": delta_1m,
                    "breadth_50sma": b50,
                    "raw_ticker": t_str or canon_raw_map.get(h_norm, h_norm),
                    "rs": None,
                    "status": "—",
                    "dist_ema": None,
                    "is_tradable_leader": True,
                    "circuit_days_20d": 0,
                }

    de_holdings: list[dict] = []
    held_in_clusters_items: list[dict] = []
    held_weak_alerts: list[dict] = []
    ledger = dict(empty_ledger)
    ledger["held_total"] = len(canonical_order)

    for h_norm in canonical_order:
        hit = held_hits.get(h_norm)
        raw_sym = (hit or {}).get("raw_ticker") or canon_raw_map.get(h_norm, h_norm)
        ern_days = _get_near_earnings_days(earnings_cache, h_norm, raw_sym)
        tv_sym, tv_url = _format_tv_for_market(h_norm, raw_sym, is_us=is_us, us_tv_map=us_tv_map)

        if hit is None:
            ledger["unmapped"] += 1
            if ern_days is not None:
                held_weak_alerts.append(
                    {
                        "ticker": h_norm,
                        "tv_symbol": tv_sym,
                        "tv_url": tv_url,
                        "industry_group": "Unmapped / ETF",
                        "rank_today": 0,
                        "grade_str": "—",
                        "grade_badge": "Unmapped",
                        "delta_1m": 0,
                        "breadth_50sma": 0.0,
                        "rs": None,
                        "status": "—",
                        "dist_ema": None,
                        "is_tradable_leader": True,
                        "circuit_days_20d": 0,
                        "is_danger": False,
                        "ern_days": ern_days,
                        "alert_reasons": [f"⚠️ ERN {ern_days}D"],
                        "_sev": 4,
                    }
                )
            continue

        grp = hit["industry_group"]
        in_top_clusters = grp in top_cluster_rank
        is_danger = bool(hit["is_danger"])
        rs_val = hit["rs"]
        status_str = hit["status"]
        dist_ema = hit["dist_ema"]
        is_tradable = bool(hit["is_tradable_leader"])
        circ_days = int(hit["circuit_days_20d"] or 0)

        reasons: list[str] = []
        if is_danger:
            reasons.append(f"🚨 {hit['grade_badge'] or 'D/E Danger'}")
        if "FAILED" in status_str.upper():
            reasons.append("🛑 FAILED")
        elif dist_ema is not None and dist_ema < -5.0:
            reasons.append(f"📉 21E {dist_ema:+.1f}%")
        if not is_tradable:
            reasons.append("🚫 Tier-0 Illiquid")
        if circ_days >= 3:
            reasons.append("🔒 Circuit Risk")
        if rs_val is not None and rs_val < 70:
            reasons.append(f"📉 Laggard RS {rs_val}")
        elif in_top_clusters and rs_val is not None and rs_val < 80:
            reasons.append(f"📉 <RS80 ({rs_val})")
        if ern_days is not None:
            reasons.append(f"⚠️ ERN {ern_days}D")

        sev = 4
        if ("FAILED" in status_str.upper()) or (not is_tradable) or (circ_days >= 3):
            sev = 0
        elif rs_val is not None and rs_val < 70:
            sev = 1
        elif dist_ema is not None and dist_ema < -5.0:
            sev = 2
        elif in_top_clusters and rs_val is not None and rs_val < 80:
            sev = 3

        rec = {
            "ticker": h_norm,
            "tv_symbol": tv_sym,
            "tv_url": tv_url,
            "industry_group": grp,
            "board_rank": top_cluster_rank.get(grp),
            "rank_today": hit["rank_today"],
            "grade_str": hit["grade_str"],
            "grade_badge": hit["grade_badge"],
            "delta_1m": hit["delta_1m"],
            "breadth_50sma": hit["breadth_50sma"],
            "rs": rs_val,
            "status": status_str,
            "dist_ema": dist_ema,
            "is_tradable_leader": is_tradable,
            "circuit_days_20d": circ_days,
            "is_danger": is_danger,
            "ern_days": ern_days,
            "surfaced_on_card": bool((grp, h_norm) in surfaced_leader_keys),
            "alert_reasons": reasons,
            "_sev": sev,
        }

        if is_danger:
            de_holdings.append(rec)
        if in_top_clusters:
            held_in_clusters_items.append(rec)
        if (not is_danger) and reasons:
            held_weak_alerts.append(rec)

        # Mutually exclusive 6-bucket ledger (D/E > Top-N > Niche > A+/A/B > C > Unmapped)
        if is_danger:
            ledger["de_danger"] += 1
        elif in_top_clusters:
            ledger["top_clusters"] += 1
        elif hit["source"] == "niche":
            ledger["niche"] += 1
        elif hit["grade_str"] in ("A+", "A", "B"):
            ledger["ab_other"] += 1
        elif hit["grade_str"] == "C":
            ledger["c_groups"] += 1
        else:
            ledger["unmapped"] += 1

    de_holdings.sort(
        key=lambda x: (
            -int(x.get("rank_today") or 0),
            x["rs"] if x.get("rs") is not None else 999,
            x["ticker"],
        )
    )
    held_weak_alerts.sort(
        key=lambda x: (
            int(x.get("_sev", 4)),
            -int(x.get("rank_today") or 0),
            x["rs"] if x.get("rs") is not None else 999,
            x["ticker"],
        )
    )
    for lst in (de_holdings, held_in_clusters_items, held_weak_alerts):
        for r in lst:
            r.pop("_sev", None)

    return {
        "held_total": len(canonical_order),
        "held_in_clusters": len(held_in_clusters_items),
        "holdings_in_de": len(de_holdings),
        "de_holdings": de_holdings,
        "held_in_clusters_items": held_in_clusters_items,
        "held_weak_alerts": held_weak_alerts,
        "ledger": ledger,
    }


def _compute_cluster_board_core(
    *,
    matrix_df: pd.DataFrame,
    thrust_res: dict,
    funda_map: dict,
    spons_map: dict,
    udvr_map: dict,
    funda_as_of: str,
    market: str = "INDIA",
    lens: str = "AUTO",
    top_n: int = 8,
    confirmed_only: bool = True,
    portfolio_tickers: list[str] | None = None,
    earnings_cache: dict | None = None,
    brain_cache: dict | None = None,
    stockbee_pulse: str = "⚖️ BALANCED",
    stockbee_ratio_5d: float = 1.0,
    us_tv_map: dict | None = None,
    niche_df: pd.DataFrame | None = None,
    min_sales_pct: float = 0.0,
) -> dict:
    is_us = str(market or "INDIA").strip().upper().startswith("US")
    held_set = {
        c.upper()
        for t in (portfolio_tickers or [])
        if t
        for c in (_normalize_ticker_key(t), *_ticker_lookup_candidates(t))
        if c
    }

    tt = (brain_cache or {}).get("tide_turn") or {}
    stage_num = int(tt.get("stage_num", 0) or 0)
    conf_list = (brain_cache or {}).get("confluence") or []
    brain_score_map = {}
    for ci in conf_list:
        if isinstance(ci, dict):
            tk = _normalize_ticker_key(ci.get("clean_ticker") or ci.get("ticker") or "")
            if tk:
                brain_score_map[tk] = ci.get("brain_score", 0)

    t2108_val = (thrust_res.get("summary") or {}).get("t2108")
    is_provisional = bool((thrust_res.get("as_of") or {}).get("is_provisional", False))
    m_gate = _resolve_m_gate(stage_num, stockbee_pulse, t2108_val, is_provisional)
    active_lens = m_gate["default_lens"] if str(lens or "AUTO").upper() == "AUTO" else str(lens).upper()

    if matrix_df is None or matrix_df.empty:
        return {
            "status": "no_cache",
            "m_gate": m_gate,
            "active_lens": active_lens,
            "clusters": [],
            "overlap": {},
            "meta": {"funda_as_of": funda_as_of, "ratio_5d": stockbee_ratio_5d, "t2108": t2108_val},
            "tv_watchlist": "",
        }

    all_thrust_df = thrust_res.get("all_thrust_df")
    thrust_by_grp: dict[str, dict] = {}
    if isinstance(all_thrust_df, pd.DataFrame) and not all_thrust_df.empty:
        for _, tr in all_thrust_df.iterrows():
            thrust_by_grp[str(tr["Industry_Group"])] = tr.to_dict()

    # Eligibility + backfill (always keep non-empty board)
    base_df = matrix_df[matrix_df.get("Tradable_Count", pd.Series(4, index=matrix_df.index)).fillna(0) >= 4].copy()
    if base_df.empty:
        base_df = matrix_df.copy()

    if confirmed_only:
        danger_s = (
            base_df["Is_Danger_Group"].fillna(False).astype(bool)
            if "Is_Danger_Group" in base_df.columns
            else pd.Series(False, index=base_df.index, dtype=bool)
        )
        conf_s = (
            base_df["Is_Group_Confirmed"].fillna(False).astype(bool)
            if "Is_Group_Confirmed" in base_df.columns
            else pd.Series(False, index=base_df.index, dtype=bool)
        )
        grade_s = (
            base_df["Group_Grade"].fillna("C").astype(str)
            if "Group_Grade" in base_df.columns
            else pd.Series("C", index=base_df.index, dtype=str)
        )
        elig_df = base_df[
            (~danger_s) & (conf_s | grade_s.isin(["A+", "A", "B"]))
        ].copy()
        elig_df["_is_backfill"] = False
        if len(elig_df) < top_n:
            rem = base_df[~base_df["Industry_Group"].isin(elig_df["Industry_Group"])].copy()
            rem["_is_backfill"] = True
            elig_df = pd.concat([elig_df, rem], ignore_index=True)
    else:
        elig_df = base_df.copy()
        elig_df["_is_backfill"] = False

    if elig_df.empty:
        return {
            "status": "empty",
            "m_gate": m_gate,
            "active_lens": active_lens,
            "clusters": [],
            "overlap": {},
            "meta": {"funda_as_of": funda_as_of},
            "tv_watchlist": "",
        }

    def _col_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
        if col not in df.columns:
            return pd.Series(default, index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce").fillna(default)

    def _col_bool(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
        if col not in df.columns:
            return pd.Series(default, index=df.index, dtype=bool)
        return df[col].fillna(default).astype(bool)

    # Compute independent group pillars (NO double breadth: Pure_Price_RS only - D1!)
    rs_raw_col = "Pure_Price_RS" if "Pure_Price_RS" in elig_df.columns else "Comp_RS"
    pure_rs_vals = _col_num(elig_df, rs_raw_col, 0.0)
    rs_pct = pure_rs_vals.rank(pct=True) * 100.0
    alpha_adj = np.where(_col_bool(elig_df, "Is_Dual_Alpha"), 5.0, np.where(_col_bool(elig_df, "Is_Relative_Mirage"), -10.0, 0.0))
    elig_df["_rs_comp"] = (rs_pct + alpha_adj).clip(0.0, 100.0)

    b_rs80 = _col_num(elig_df, "Breadth_RS80_Pct", 0.0)
    b_sma50 = _col_num(elig_df, "Breadth_50SMA_Pct", 0.0)
    b_rsnh = _col_num(elig_df, "RSNH_Pct", 0.0)
    raw_breadth = 0.45 * b_rs80 + 0.30 * b_sma50 + 0.25 * b_rsnh
    elig_df["_breadth_comp"] = raw_breadth.rank(pct=True) * 100.0

    # Group Funda Coverage + Inst Accum % (IN)
    funda_cov_list = []
    funda_scores_list = []
    inst_accum_list = []
    thrust_score_list = []

    has_spons_data = bool(spons_map) and (not is_us)
    for _, erow in elig_df.iterrows():
        grp = str(erow["Industry_Group"])
        constits = erow.get("Constituents") or []
        denom = max(1, len(constits))
        f_valid_cnt = 0
        s_acc_cnt = 0
        for t in constits:
            sym = _normalize_ticker_key(t)
            if has_valid_fundamentals(_lookup_map(funda_map, sym)):
                f_valid_cnt += 1
            if has_spons_data:
                s_row = _lookup_map(spons_map, sym) or {}
                if str(s_row.get("grade", "NR")) in ("A+", "A") or float(s_row.get("inst_change", 0.0) or 0.0) >= 0.5:
                    s_acc_cnt += 1
        cov = (f_valid_cnt / float(denom)) * 100.0
        funda_cov_list.append(round(cov, 1))
        inst_pct = round((s_acc_cnt / float(denom)) * 100.0, 1) if has_spons_data else None
        inst_accum_list.append(inst_pct)

        can_f = float(erow.get("CANSLIM_Funda_Pct", 0.0) or 0.0)
        funda_scores_list.append((0.70 * can_f + 0.30 * float(inst_pct or 0.0)) if has_spons_data else can_f)
        thrust_score_list.append(float((thrust_by_grp.get(grp) or {}).get("Thrust_Score", 0.0) or 0.0))

    elig_df["_funda_cov"] = funda_cov_list
    elig_df["_inst_accum_pct"] = inst_accum_list
    elig_df["_funda_comp"] = pd.Series(funda_scores_list, index=elig_df.index).rank(pct=True) * 100.0
    elig_df["_thrust_comp"] = pd.Series(thrust_score_list, index=elig_df.index).rank(pct=True) * 100.0

    median_cov = float(np.median(funda_cov_list)) if funda_cov_list else 0.0
    partial_funda_cache = median_cov < 15.0

    if partial_funda_cache:
        base_score = 0.42 * elig_df["_rs_comp"] + 0.33 * elig_df["_breadth_comp"] + 0.25 * elig_df["_thrust_comp"]
    else:
        base_score = (
            0.35 * elig_df["_rs_comp"]
            + 0.25 * elig_df["_breadth_comp"]
            + 0.20 * elig_df["_funda_comp"]
            + 0.20 * elig_df["_thrust_comp"]
        )
    # Weak internal trend dampener
    elig_df["Cluster_Score"] = np.where(b_sma50 < 45.0, (base_score - 8.0).clip(lower=0.0), base_score).round(1)

    # Holding-up-best defense score for RED markets (P1)
    near52 = _col_num(elig_df, "Near_52WH_Pct", 0.0)
    bluedot_p = _col_num(elig_df, "RS_BlueDot_Pct", 0.0)
    elig_df["_defense_score"] = (0.35 * b_rsnh + 0.25 * near52 + 0.25 * b_sma50 + 0.15 * bluedot_p).round(1)

    elig_df["_u_5d"] = [int((thrust_by_grp.get(str(g)) or {}).get("Up4_5D_Unique", 0)) for g in elig_df["Industry_Group"]]
    elig_df["_u_t0"] = [int((thrust_by_grp.get(str(g)) or {}).get("Up4_Today_Cnt", 0)) for g in elig_df["Industry_Group"]]
    elig_df["_rank_td"] = _col_num(elig_df, "Rank_Today", 999.0)
    elig_df["_d1m"] = _col_num(elig_df, "Delta_1M", 0.0)

    if active_lens == "DEFENSE":
        ranked_df = elig_df.sort_values(by=["_defense_score", "_rs_comp", "Cluster_Score"], ascending=[False, False, False])
    elif active_lens == "THRUST":
        ranked_df = elig_df.sort_values(by=["_u_5d", "_u_t0", "_d1m", "Cluster_Score"], ascending=[False, False, False, False])
    elif active_lens == "ONEIL":
        ranked_df = elig_df.sort_values(by=["_rank_td", "Cluster_Score"], ascending=[True, False])
    else:
        ranked_df = elig_df.sort_values(by=["Cluster_Score", "_rank_td"], ascending=[False, True])

    top_clusters_df = ranked_df.head(max(1, int(top_n)))

    clusters = []
    tv_sections = []
    total_surfaced = 0
    total_tml = 0
    held_surfaced = 0
    dual_alpha_top40 = int(
        len(matrix_df[(_col_num(matrix_df, "Rank_Today", 999.0) <= 40) & _col_bool(matrix_df, "Is_Dual_Alpha")])
    )

    for idx_pos, (_, crow) in enumerate(top_clusters_df.iterrows(), 1):
        grp = str(crow["Industry_Group"])
        th_info = thrust_by_grp.get(grp, {})
        th_tmap = {t["ticker"]: t for t in (th_info.get("All_Thrusting_Tickers") or [])}

        leaders, action_cnt, is_ext_grp, leader_stats = _select_cluster_tml_leaders(
            group_row=crow.to_dict(),
            is_us=is_us,
            funda_map=funda_map,
            spons_map=spons_map,
            udvr_map=udvr_map,
            thrust_ticker_map=th_tmap,
            held_set=held_set,
            earnings_cache=earnings_cache,
            brain_score_map=brain_score_map,
            us_tv_map=us_tv_map,
            max_leaders=3,
            min_sales_pct=float(min_sales_pct),
        )

        total_surfaced += len(leaders)
        total_tml += sum(1 for l in leaders if l["tier_num"] == 4)
        held_surfaced += sum(1 for l in leaders if l["is_held"])

        if leaders:
            clean_sec = grp.replace(",", "")
            tv_sections.append(",".join([f"###{clean_sec}"] + [l["tv_symbol"] for l in leaders]))

        r_6w = _safe_int(crow.get("Rank_6W"), 0)
        r_3w = _safe_int(crow.get("Rank_3W"), 0)
        r_1w = _safe_int(crow.get("Rank_1W"), 0)
        r_td = _safe_int(crow.get("Rank_Today"), 0)
        persistence_tag = "🆕 New Leader" if (r_6w > 40 and r_td <= 25) else ("🏛️ Established" if (0 < r_6w <= 25 and r_td <= 25) else str(crow.get("Rotation_Status") or ""))

        clusters.append(
            {
                "board_rank": idx_pos,
                "industry_group": grp,
                "rank_today": r_td,
                "rank_path": f"#{r_6w} → #{r_3w} → #{r_1w} → #{r_td}",
                "persistence_tag": persistence_tag,
                "grade_badge": str(crow.get("Group_Grade_Badge") or f"#{r_td}"),
                "delta_1m": _safe_int(crow.get("Delta_1M"), 0),
                "is_dual_alpha": _safe_bool(crow.get("Is_Dual_Alpha"), False),
                "is_relative_mirage": _safe_bool(crow.get("Is_Relative_Mirage"), False),
                "is_confirmed": _safe_bool(crow.get("Is_Group_Confirmed"), False),
                "is_backfill": _safe_bool(crow.get("_is_backfill"), False),
                "cluster_score": _safe_float(crow.get("Cluster_Score"), 0.0),
                "why_ranked": {
                    "rs": round(float(crow.get("_rs_comp", 0.0)), 0),
                    "breadth": round(float(crow.get("_breadth_comp", 0.0)), 0),
                    "funda": round(float(crow.get("_funda_comp", 0.0)), 0),
                    "thrust": round(float(crow.get("_thrust_comp", 0.0)), 0),
                },
                "breadth_rs80": round(float(crow.get("Breadth_RS80_Pct", 0.0) or 0.0), 1),
                "breadth_50sma": round(float(crow.get("Breadth_50SMA_Pct", 0.0) or 0.0), 1),
                "rsnh_pct": round(float(crow.get("RSNH_Pct", 0.0) or 0.0), 1),
                "funda_pct": round(float(crow.get("CANSLIM_Funda_Pct", 0.0) or 0.0), 1),
                "funda_cov": float(crow.get("_funda_cov", 0.0)),
                "inst_accum_pct": crow.get("_inst_accum_pct"),
                "up4_today": int(th_info.get("Up4_Today_Cnt", 0) or 0),
                "up4_5d_unique": int(th_info.get("Up4_5D_Unique", 0) or 0),
                "up25_qtr": int(th_info.get("Up25_Qtr_Cnt", 0) or 0),
                "nh_5d": int(th_info.get("NewHigh_5D_Unique", 0) or 0),
                "scoped_valid_cnt": int(th_info.get("Group_Valid_Cnt", crow.get("Tradable_Count", 0)) or 0),
                "actionable_cnt": action_cnt,
                "is_extended_group": is_ext_grp,
                "qualifying_leader_cnt": int(leader_stats.get("qualifying_total", len(leaders))),
                "suppressed_laggards": int(leader_stats.get("suppressed_laggards", 0)),
                "suppressed_failed": int(leader_stats.get("suppressed_failed", 0)),
                "suppressed_low_sales": int(leader_stats.get("suppressed_low_sales", 0)),
                "min_sales_pct": int(round(float(min_sales_pct))),
                "leaders": leaders,
            }
        )

    triage = _build_portfolio_triage(
        matrix_df=matrix_df,
        niche_df=niche_df,
        clusters=clusters,
        held_raw_tickers=portfolio_tickers,
        earnings_cache=earnings_cache,
        is_us=is_us,
        us_tv_map=us_tv_map,
    )

    as_of_info = thrust_res.get("as_of") or {}
    return {
        "status": "ok",
        "m_gate": m_gate,
        "active_lens": active_lens,
        "clusters": clusters,
        "overlap": {
            "held_total": triage["held_total"],
            "held_in_clusters": triage["held_in_clusters"],
            "total_surfaced": total_surfaced,
            "holdings_in_de": triage["holdings_in_de"],
            "de_holdings": triage["de_holdings"],
            "held_in_clusters_items": triage["held_in_clusters_items"],
            "held_weak_alerts": triage["held_weak_alerts"],
            "ledger": triage["ledger"],
        },
        "meta": {
            "prices_date": as_of_info.get("prices_date", "—"),
            "groups_date": as_of_info.get("groups_date", "—"),
            "igm_stale": bool(as_of_info.get("igm_stale", False)),
            "funda_as_of": funda_as_of,
            "partial_funda_cache": partial_funda_cache,
            "ratio_5d": stockbee_ratio_5d,
            "pulse": stockbee_pulse,
            "t2108": t2108_val,
            "dual_alpha_top40": dual_alpha_top40,
            "total_tml_surfaced": total_tml,
        },
        "tv_watchlist": "\n".join(tv_sections),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def _load_udvr_map_cached(market: str, scope_key: str, matrix_sig: tuple) -> dict:
    panels, _ = _load_panels_for_stockbee(market, scope_key)
    if panels and panels.get("close") is not None and panels.get("volume") is not None:
        return _compute_50d_udvr_series(panels["close"], panels["volume"]).to_dict()
    try:
        from database import load_cloud_json_cache_cached
        is_us = str(market or "IN").strip().upper().startswith("US")
        cloud_map, _ = load_cloud_json_cache_cached("udvr_map_us" if is_us else "udvr_map_in")
        if isinstance(cloud_map, dict) and cloud_map:
            return {str(k): float(v) for k, v in cloud_map.items() if v is not None}
    except Exception:
        pass
    return {}


def compute_cluster_leadership_board(
    market: str = "INDIA",
    lens: str = "AUTO",
    top_n: int = 8,
    confirmed_only: bool = True,
    portfolio_tickers: list[str] | None = None,
    earnings_cache: dict | None = None,
    min_sales_pct: float = 0.0,
) -> dict:
    """Public non-blocking orchestrator for app.py Section 0 Cluster Leadership Board."""
    from components import _stage_matrix_path_for_market, load_stockbee_monitor_df
    from database import get_brain_synthesis_cache
    from industry_group_engine import load_cached_industry_group_matrix, load_cached_niche_industry_groups
    from price_history_manager import get_matrix_cache_signature

    is_us = str(market or "INDIA").strip().upper().startswith("US")
    mkt_code = "US" if is_us else "IN"
    igm_scope = "US (S&P 500 / Nasdaq)" if is_us else "India (NSE/BSE)"
    sb_scope = "BROAD" if is_us else "FULL"

    mat_df = load_cached_industry_group_matrix(universe_scope=igm_scope)
    mat_df = mat_df.copy() if isinstance(mat_df, pd.DataFrame) and not mat_df.empty else pd.DataFrame()
    niche_df = None
    try:
        nd = load_cached_niche_industry_groups(universe_scope=igm_scope)
        if isinstance(nd, pd.DataFrame) and not nd.empty:
            niche_df = nd.copy()
    except Exception:
        niche_df = None

    thrust_res = compute_stockbee_thrusting_industries(
        market=mkt_code, universe_scope=sb_scope, mode="5D_AND_TODAY", top_n=200
    )

    # Read latest Stockbee row for Ratio_5D + Pulse pill
    sb_df = load_stockbee_monitor_df(market=mkt_code, universe_scope=("SP500" if is_us else "NTM750"), lookback_days=15)
    sb_pulse = str(sb_df.iloc[0]["Pulse"]) if isinstance(sb_df, pd.DataFrame) and not sb_df.empty else "⚖️ BALANCED"
    sb_ratio5 = float(sb_df.iloc[0]["Ratio_5D"]) if isinstance(sb_df, pd.DataFrame) and not sb_df.empty else 1.0

    db_sig = _get_db_freshness_sig()
    db_data = _load_fundamentals_and_sponsorship_cached(mkt_code, db_sig)
    us_tv_map = _load_us_tv_exchange_map_cached(db_sig) if is_us else None

    m_path = _stage_matrix_path_for_market(mkt_code)
    m_sig = get_matrix_cache_signature(m_path) if os.path.exists(m_path) else (0, 0)
    udvr_map = _load_udvr_map_cached(mkt_code, sb_scope, m_sig)

    brain_cache = get_brain_synthesis_cache("US_IXIC" if is_us else "INDIA")

    return _compute_cluster_board_core(
        matrix_df=mat_df,
        thrust_res=thrust_res,
        funda_map=db_data["funda_map"],
        spons_map=db_data["spons_map"],
        udvr_map=udvr_map,
        funda_as_of=db_data["funda_as_of"],
        market="US_IXIC" if is_us else "INDIA",
        lens=lens,
        top_n=top_n,
        confirmed_only=confirmed_only,
        portfolio_tickers=portfolio_tickers,
        earnings_cache=earnings_cache,
        brain_cache=brain_cache,
        stockbee_pulse=sb_pulse,
        stockbee_ratio_5d=sb_ratio5,
        us_tv_map=us_tv_map,
        niche_df=niche_df,
        min_sales_pct=float(min_sales_pct),
    )
