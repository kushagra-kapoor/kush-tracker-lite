"""
Institutional Artificial Brain Synthesis Engine for Kush Tracker
================================================================
Single-Source-of-Truth (SSOT) Dual-Scope Vectorized Quantitative Engine:
  - Scope A (Nifty 750 Liquid Core):
      Automated 5-Stage Bear-to-Bull Tide-Turn State Machine, 5D Panic/Surge Breadth,
      9:1 Zweig Up-Volume Thrust, 20-EMA Recapture Sequence, and Non-Squat FTD Validation.
  - Scope B (Full 2,750+ Universe Matrix + All Active Portfolio SME/BSE Holdings):
      7-Pillar Omni-Page Confluence Funnel (Ironclad Beach Balls + HV1 Float-Lockup Ratio
      + RSNH/Smooth RS + Live Triggers + Livermore Sister Tandem + 100% Doublers + Earnings).
  - Tom Hougaard ("Best Loser Wins") Zero-New-Heat Reallocator:
      Pairs stagnant "Weeds" (Dead Money / Lagging RS) with coiling "Flowers" (Add-Ready
      Winners) financed 100% by locked-in house-money profit cushion (0.00% net new NAV heat).
"""

import os
import pickle
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from database import (
    get_connection,
    get_all_fundamentals_cache,
    get_latest_market_persona,
    save_brain_synthesis_cache,
    get_brain_synthesis_cache,
)
from market_data import (
    fetch_nifty_total_market_tickers,
    fetch_full_india_universe_tickers,
    get_ticker_symbol,
)
from macro_regime_engine import (
    detect_follow_through_day,
    detect_20ema_recapture_and_sequence,
    calculate_distribution_days,
)

# Canonical India cash / defensive ETFs excluded from active equity traction & funnel
CASH_EQUIVALENT_TICKERS = {'LIQUIDBEES', 'LIQUIDCASE', 'LIQUIDETF', 'GOLDBEES', 'SILVERBEES'}
BRAIN_CACHE_SCHEMA_VERSION = 'v5_dual_market_us_ixic'

MARKET_BRAIN_PROFILES = {
    'INDIA': {
        'profile_key': 'INDIA',
        'breadth_market': 'IN',
        'is_us': False,
        'benchmark_sym': '^CRSLDX',
        'benchmark_label': 'Nifty 500 (^CRSLDX)',
        'curr_sym': '₹',
        'price_decimals': 1,
        'adtv_scale': 1e7,
        'adtv_unit': '₹ Cr',
        'funnel_min_adtv': 2.5,
        'min_price': 20.0,
        'bonde_pct': 0.12,
        'min_cache_universe': 1500,
        'include_portfolio': True,
    },
    'US_IXIC': {
        'profile_key': 'US_IXIC',
        'breadth_market': 'US_IXIC',
        'is_us': True,
        'benchmark_sym': '^IXIC',
        'benchmark_label': 'Nasdaq Composite (^IXIC)',
        'curr_sym': '$',
        'price_decimals': 2,
        'adtv_scale': 1e6,
        'adtv_unit': '$M',
        'funnel_min_adtv': 10.0,
        'min_price': 5.0,
        'bonde_pct': 0.20,
        'min_cache_universe': 800,
        'include_portfolio': False,
    },
}


def get_market_brain_profile(market: str = 'INDIA') -> dict:
    """Returns canonical dual-market Brain & Funnel profile config ('INDIA' or 'US_IXIC')."""
    m = str(market or 'INDIA').strip().upper()
    if m in ('US', 'USA', 'US_IXIC'):
        return dict(MARKET_BRAIN_PROFILES['US_IXIC'])
    return dict(MARKET_BRAIN_PROFILES['INDIA'])


def _resolve_market_session_date(market: str = 'INDIA') -> str:
    try:
        from price_history_manager import get_latest_completed_session_date
        return get_latest_completed_session_date(market)
    except Exception:
        return datetime.now().strftime('%Y-%m-%d')

# Deterministic CANSLIM 5-Stage Progressive Exposure & Per-Trade Risk Policy
STAGE_SIZING_POLICY = {
    1: {
        'target_min_pct': 0.0,
        'target_max_pct': 0.0,
        'risk_per_trade_pct': 0.0,
        'risk_label': '0%',
        'max_pos_pct': 0.0,
        'max_new_positions_label': '0 New Buys (Hold Cash)',
    },
    2: {
        'target_min_pct': 0.0,
        'target_max_pct': 10.0,
        'risk_per_trade_pct': 0.25,
        'risk_label': '0.25% NAV',
        'max_pos_pct': 5.0,
        'max_new_positions_label': 'Max 1–2 Test Pilots (≤ 5% Pos Cap)',
    },
    3: {
        'target_min_pct': 15.0,
        'target_max_pct': 30.0,
        'risk_per_trade_pct': 0.35,
        'risk_label': '0.35% NAV',
        'max_pos_pct': 8.0,
        'max_new_positions_label': 'Max 2–3 Pilots (≤ 8% Pos Cap)',
    },
    4: {
        'target_min_pct': 50.0,
        'target_max_pct': 75.0,
        'risk_per_trade_pct': 0.60,
        'risk_label': '0.50%–0.75% NAV',
        'max_pos_pct': 12.0,
        'max_new_positions_label': 'Progressive Scale (≤ 12% Pos Cap)',
    },
    5: {
        'target_min_pct': 85.0,
        'target_max_pct': 100.0,
        'risk_per_trade_pct': 0.85,
        'risk_label': '0.75%–1.00% NAV',
        'max_pos_pct': 15.0,
        'max_new_positions_label': 'Full Scale + Pyramid (≤ 15% Pos Cap)',
    },
}


def _evaluate_zweig_9to1_thrust(
    close_core: pd.DataFrame = None,
    vol_core: pd.DataFrame = None,
    breadth_df: pd.DataFrame = None,
    lookback_sessions: int = 7,
) -> dict:
    """
    Evaluates Zweig 9:1 Up-Volume Thrust (>= 90% up-volume) with canonical breadth_df
    priority and partial/thin-session protection (total volume >= 40% of 20D median).
    """
    default_out = {
        'up_vol_ratio_pct': 50.0,
        'is_9to1_thrust': False,
        'last_9to1_date': None,
        'last_9to1_ratio': 0.0,
    }
    daily_rows = []

    if breadth_df is not None and not breadth_df.empty and {'up_volume', 'down_volume'}.issubset(breadth_df.columns):
        b_sub = breadth_df.dropna(subset=['up_volume', 'down_volume']).tail(25)
        if len(b_sub) >= lookback_sessions:
            for idx_val, row in b_sub.iterrows():
                uv = float(row.get('up_volume') or 0.0)
                dv = float(row.get('down_volume') or 0.0)
                dt_s = str(row.get('date', idx_val))[:10]
                if (uv + dv) > 0:
                    daily_rows.append((dt_s, uv, dv))

    if len(daily_rows) < lookback_sessions and close_core is not None and vol_core is not None and not close_core.empty and not vol_core.empty:
        daily_rows = []
        rets_w = close_core.pct_change(fill_method=None).iloc[-25:]
        vols_w = vol_core.iloc[-25:]
        for pos in range(len(rets_w)):
            r_row = rets_w.iloc[pos]
            v_row = vols_w.iloc[pos]
            uv = float(v_row[r_row > 0].sum())
            dv = float(v_row[r_row < 0].sum())
            dt_s = str(rets_w.index[pos])[:10]
            if (uv + dv) > 0:
                daily_rows.append((dt_s, uv, dv))

    if not daily_rows:
        return default_out

    tot_vols = [uv + dv for _, uv, dv in daily_rows if (uv + dv) > 0]
    med_vol = float(np.median(tot_vols)) if tot_vols else 0.0

    last_dt, last_uv, last_dv = daily_rows[-1]
    last_tot = last_uv + last_dv
    if last_tot <= 0 and len(daily_rows) >= 2:
        for _, uv_p, dv_p in reversed(daily_rows[:-1]):
            if (uv_p + dv_p) > 0:
                last_uv, last_dv, last_tot = uv_p, dv_p, uv_p + dv_p
                break
    up_vol_pct = round((last_uv / last_tot) * 100.0, 1) if last_tot > 0 else 50.0

    # Check trailing 7 valid (non-thin) sessions for true >= 0.90 Zweig 9:1 thrust
    min_valid_vol = 0.40 * med_vol if med_vol > 0 else 0.0
    last_9to1_date = None
    last_9to1_ratio = 0.0
    is_thrust = False
    for dt_s, uv, dv in reversed(daily_rows[-lookback_sessions:]):
        tot = uv + dv
        if tot <= 0 or (min_valid_vol > 0 and tot < min_valid_vol):
            continue
        pct = uv / tot
        ratio = round(uv / dv, 2) if dv > 0 else 99.0
        if pct >= 0.90 or ratio >= 9.0:
            is_thrust = True
            last_9to1_date = dt_s
            last_9to1_ratio = ratio
            break

    return {
        'up_vol_ratio_pct': up_vol_pct,
        'is_9to1_thrust': is_thrust,
        'last_9to1_date': last_9to1_date,
        'last_9to1_ratio': last_9to1_ratio,
    }


def _filter_market_confluence_columns(cols, market: str = 'INDIA', portfolio_clean_set: set = None) -> list:
    """
    Normalizes dual-market matrix columns for the Confluence Funnel:
    - India ('INDIA'): excludes cash/commodity ETFs, deduplicates .BO vs .NS (preferring .NS),
      restricts broad matrices (>= 300) to active non-ETF Security Master tickers while
      exempting active portfolio holdings.
    - US ('US' / 'US_IXIC'): excludes .NS/.BO/index '^'/ETF symbols, intersects broad matrices
      (>= 300) strictly with active non-ETF US equities across NASDAQ, NYSE & AMEX.
    """
    prof = get_market_brain_profile(market)
    if prof['is_us']:
        raw_us = []
        try:
            from down_day_engine import is_etf_or_non_equity
        except Exception:
            is_etf_or_non_equity = lambda _x: False

        for c in cols:
            c_str = str(c).strip()
            if not c_str or c_str.endswith(('.NS', '.BO')) or c_str.startswith('^'):
                continue
            if is_etf_or_non_equity(c_str):
                continue
            raw_us.append((c, c_str.upper()))

        allowed_us = None
        if len(raw_us) >= 300:
            try:
                from universe_security_master import load_universe_tickers
                sec_us = load_universe_tickers(market='US', include_etfs=False)
                if sec_us:
                    allowed_us = {str(t).strip().upper() for t in sec_us}
            except Exception:
                allowed_us = None

        return [
            col_obj for col_obj, sym_up in raw_us
            if allowed_us is None or sym_up in allowed_us
        ]

    p_set = portfolio_clean_set or set()
    raw_ind = []
    for c in cols:
        c_str = str(c).strip()
        if not (c_str.endswith('.NS') or c_str.endswith('.BO')):
            continue
        base = c_str[:-3].upper()
        if base in CASH_EQUIVALENT_TICKERS:
            continue
        raw_ind.append((c, c_str, base, 0 if c_str.endswith('.NS') else 1))

    # Deduplicate .BO vs .NS keeping .NS first
    best_by_base = {}
    for col_obj, c_str, base, prio in raw_ind:
        norm_base = base.replace('-SM', '').replace('-ST', '').strip()
        prev = best_by_base.get(norm_base)
        if prev is None or prio < prev[3]:
            best_by_base[norm_base] = (col_obj, c_str, norm_base, prio)

    allowed_clean = None
    if len(best_by_base) >= 300:
        try:
            from universe_security_master import load_universe_tickers
            sec_tickers = load_universe_tickers(market='IN', include_etfs=False)
            if sec_tickers:
                allowed_clean = {
                    str(t).replace('.NS', '').replace('.BO', '').replace('-SM', '').replace('-ST', '').strip().upper()
                    for t in sec_tickers
                }
        except Exception:
            allowed_clean = None

    filtered = []
    for col_obj, c_str, norm_base, _ in best_by_base.values():
        if allowed_clean is not None and norm_base not in allowed_clean and norm_base not in p_set and c_str[:-3].upper() not in p_set:
            continue
        filtered.append(col_obj)
    return filtered


def _filter_india_confluence_columns(cols, portfolio_clean_set: set = None) -> list:
    """Backward-compatible India alias for _filter_market_confluence_columns."""
    return _filter_market_confluence_columns(cols, market='INDIA', portfolio_clean_set=portfolio_clean_set)


def apply_stage_sizing_to_confluence_list(
    candidates: list,
    tide_stage_num: int = 3,
    market: str = 'INDIA',
    bh_is_pre_ftd_lag: bool = False,
    ftd_suspect_warning: bool = False,
) -> list:
    """
    Stamps risk-normalized Allocation %, Stage Risk Budget, and currency-aware execution plans
    onto Confluence Funnel rows without duplicating formatting across passes.
    """
    prof = get_market_brain_profile(market)
    st_num = max(1, min(5, int(tide_stage_num or 3)))
    policy = STAGE_SIZING_POLICY[st_num]
    risk_budget = float(policy['risk_per_trade_pct'])
    pos_cap = float(policy['max_pos_pct'])
    curr_sym = prof['curr_sym']
    fmt = f",.{prof['price_decimals']}f"

    for item in candidates or []:
        trig_name = item.get('trigger_name', '')
        trig_p = float(item.get('trigger_price', 0.0) or 0.0)
        stop_p = float(item.get('stop_price', 0.0) or 0.0)
        risk_pct = max(0.5, float(item.get('risk_pct', 3.5) or 3.5))
        t25 = float(item.get('target_2_5r', 0.0) or 0.0)
        is_ext = bool(item.get('is_extended_setup', False))
        ext_pct = float(item.get('ext_above_21l_pct', 0.0) or 0.0)

        alloc_pct = 0.0 if st_num <= 1 else round(min(pos_cap, (risk_budget / risk_pct) * 100.0), 1)
        item['alloc_pct'] = alloc_pct
        item['risk_budget_pct'] = risk_budget
        item['pos_cap_pct'] = pos_cap
        item['curr_sym'] = curr_sym

        tp_s = format(trig_p, fmt)
        sp_s = format(stop_p, fmt)
        t25_s = format(t25, fmt)

        if st_num <= 1:
            item['execution_plan'] = (
                f"🌊 CASH HOLD (Stage 1): Track {trig_name} @ {curr_sym}{tp_s} | "
                f"Stop {curr_sym}{sp_s} | Alloc 0%"
            )
        elif is_ext:
            item['execution_plan'] = (
                f"⏳ EXTENDED (+{ext_pct:.1f}% > 21L): Alloc ~{alloc_pct:.1f}% ({risk_budget:.2f}% Risk) "
                f"on Pullback {curr_sym}{tp_s} • Stop {curr_sym}{sp_s} ({risk_pct:.1f}%)"
            )
        elif st_num == 2:
            item['execution_plan'] = (
                f"🛡️ TEST PILOT ({trig_name}): Alloc ~{alloc_pct:.1f}% ({risk_budget:.2f}% Risk) • "
                f"Buy {curr_sym}{tp_s} • Stop {curr_sym}{sp_s}"
            )
        elif st_num == 3:
            item['execution_plan'] = (
                f"⚡ PILOT ({trig_name}): Alloc ~{alloc_pct:.1f}% ({risk_budget:.2f}% Risk) • "
                f"Buy {curr_sym}{tp_s} • Stop {curr_sym}{sp_s} • +2.5R {curr_sym}{t25_s}"
            )
        else:
            mode_tag = "🚀 PILOT-SCALE" if (st_num == 4 and (bh_is_pre_ftd_lag or ftd_suspect_warning)) else "🚀 SCALE"
            item['execution_plan'] = (
                f"{mode_tag} ({trig_name}): Alloc ~{alloc_pct:.1f}% ({risk_budget:.2f}% Risk) • "
                f"Buy {curr_sym}{tp_s} • Stop {curr_sym}{sp_s} • Trail 21L/50S"
            )
    return candidates


def compute_buy_budget_verdict(
    tide_turn: dict,
    portfolio_mirror: dict = None,
    include_portfolio: bool = True,
) -> dict:
    """
    Compares actual portfolio Deployed Equity % vs Brain Stage Target Ceiling ('INDIA'),
    or emits pure Stage Target Model Risk Guidelines ('US_IXIC', include_portfolio=False).
    """
    st_num = max(1, min(5, int((tide_turn or {}).get('stage_num', 3) or 3)))
    policy = STAGE_SIZING_POLICY[st_num]
    t_min = float((tide_turn or {}).get('target_min_pct', policy['target_min_pct']))
    t_max = float((tide_turn or {}).get('target_max_pct', policy['target_max_pct']))
    target_range = f"{t_min:.0f}%" if t_min == t_max else f"{t_min:.0f}%–{t_max:.0f}%"

    if not include_portfolio:
        if st_num <= 1:
            mode = 'CASH_HOLD'
            verdict = "Nasdaq (^IXIC) Stage 1 Model Target: 0% Cash Defense — Protect USD Capital (No New Equity Buys)"
        else:
            mode = 'US_STAGE_MODEL'
            verdict = (
                f"Nasdaq (^IXIC) Stage {st_num} Model Target: {target_range} Equity • "
                f"Deploy ≤{policy['max_pos_pct']:.0f}% Positions ({policy['risk_label']} Risk)"
            )
        return {
            'stage_num': st_num,
            'deployed_equity_pct': None,
            'cash_reserve_pct': None,
            'target_min_pct': t_min,
            'target_max_pct': t_max,
            'target_range_label': target_range,
            'risk_per_trade_pct': policy['risk_per_trade_pct'],
            'risk_label': policy['risk_label'],
            'max_pos_pct': policy['max_pos_pct'],
            'max_new_positions_label': policy['max_new_positions_label'],
            'funding_mode': mode,
            'verdict_text': verdict,
        }

    deployed = float((portfolio_mirror or {}).get('deployed_equity_pct', 0.0) or 0.0)
    cash_pct = float((portfolio_mirror or {}).get('cash_reserve_pct', max(0.0, 100.0 - deployed)) or 0.0)

    if st_num <= 1:
        mode = 'CASH_HOLD'
        verdict = f"Deployed {deployed:.1f}% vs Target 0% → Hold Cash / Trail Tight (No New Equity Deployments)"
    elif deployed >= t_max:
        mode = 'SWAP_ONLY'
        over_pct = round(deployed - t_max, 1)
        verdict = (
            f"Deployed {deployed:.1f}% vs Target {target_range} (+{over_pct:.1f}% Above Ceiling) → "
            f"Swap-Only: Fund New Top-5 Buys by Trimming Hougaard Dead Money"
        )
    else:
        mode = 'NEW_CASH_OR_SWAP'
        hd_max = round(max(0.0, t_max - deployed), 1)
        hd_min = round(max(0.0, t_min - deployed), 1)
        hd_str = f"+{hd_min:.0f}% to +{hd_max:.0f}%" if hd_min > 0 else f"Up to +{hd_max:.0f}%"
        verdict = (
            f"Deployed {deployed:.1f}% (Cash/Gold {cash_pct:.1f}%) vs Target {target_range} → "
            f"{hd_str} Cash Headroom: Deploy ≤{policy['max_pos_pct']:.0f}% Positions ({policy['risk_label']} Risk)"
        )

    return {
        'stage_num': st_num,
        'deployed_equity_pct': round(deployed, 1),
        'cash_reserve_pct': round(cash_pct, 1),
        'target_min_pct': t_min,
        'target_max_pct': t_max,
        'target_range_label': target_range,
        'risk_per_trade_pct': policy['risk_per_trade_pct'],
        'risk_label': policy['risk_label'],
        'max_pos_pct': policy['max_pos_pct'],
        'max_new_positions_label': policy['max_new_positions_label'],
        'funding_mode': mode,
        'verdict_text': verdict,
    }


def _extract_ohlcv_panels(matrix_df: pd.DataFrame):
    """Safely extracts 2D (Date x Ticker) Close, Open, High, Low, Volume DataFrames from MultiIndex matrix."""
    if matrix_df is None or matrix_df.empty or not isinstance(matrix_df.columns, pd.MultiIndex):
        return None, None, None, None, None

    lvl1_vals = [str(v).lower() for v in matrix_df.columns.get_level_values(1)]
    ohlcv_lvl = 1 if ('close' in lvl1_vals or 'adj close' in lvl1_vals) else 0

    def _get_panel(attr_name: str):
        vals = [str(v).lower() for v in matrix_df.columns.get_level_values(ohlcv_lvl)]
        if attr_name.lower() in vals:
            orig_key = matrix_df.columns.get_level_values(ohlcv_lvl)[vals.index(attr_name.lower())]
            return matrix_df.xs(orig_key, level=ohlcv_lvl, axis=1)
        return pd.DataFrame()

    close_p = _get_panel('close')
    open_p = _get_panel('open')
    high_p = _get_panel('high')
    low_p = _get_panel('low')
    vol_p = _get_panel('volume')
    return close_p, open_p, high_p, low_p, vol_p


def _trim_panels_to_valid_market_bars(
    close_p: pd.DataFrame,
    open_p: pd.DataFrame,
    high_p: pd.DataFrame,
    low_p: pd.DataFrame,
    vol_p: pd.DataFrame,
    market: str = 'INDIA',
):
    """
    Removes BOTH interior AND trailing closed/holiday rows where the target market did not
    trade (e.g., Sunday 2026-11-08 0 US bars, US-only trailing rows vs India, or 15-stock
    partial US rows). Prevents ffill(limit=5) from fabricating phantom flat sessions.
    """
    if close_p is None or close_p.empty:
        return close_p, open_p, high_p, low_p, vol_p, ''

    prof = get_market_brain_profile(market)
    if not prof['is_us']:
        mkt_cols = [
            c for c in close_p.columns
            if (str(c).endswith('.NS') or str(c).endswith('.BO'))
            and str(c)[:-3].upper() not in CASH_EQUIVALENT_TICKERS
        ]
    else:
        mkt_cols = [
            c for c in close_p.columns
            if not str(c).endswith('.NS') and not str(c).endswith('.BO') and not str(c).startswith('^')
        ]

    if not mkt_cols:
        mkt_cols = list(close_p.columns)

    valid_counts = close_p[mkt_cols].notna().sum(axis=1)
    min_req = max(50, int(0.08 * len(mkt_cols))) if len(mkt_cols) >= 300 else 1
    valid_index = valid_counts[valid_counts >= min_req].index
    if len(valid_index) == 0:
        last_dt = _extract_last_date_str(close_p)
        return close_p, open_p, high_p, low_p, vol_p, last_dt

    if len(valid_index) != len(close_p.index) or valid_index[-1] != close_p.index[-1]:
        close_p = close_p.loc[valid_index]
        if open_p is not None and not open_p.empty:
            open_p = open_p.reindex(valid_index)
        if high_p is not None and not high_p.empty:
            high_p = high_p.reindex(valid_index)
        if low_p is not None and not low_p.empty:
            low_p = low_p.reindex(valid_index)
        if vol_p is not None and not vol_p.empty:
            vol_p = vol_p.reindex(valid_index)

    return close_p, open_p, high_p, low_p, vol_p, _extract_last_date_str(close_p)


def _backfill_market_breadth_from_matrix(
    close_core: pd.DataFrame,
    high_core: pd.DataFrame,
    low_core: pd.DataFrame,
    vol_core: pd.DataFrame = None,
    market: str = 'US_IXIC',
    lookback_days: int = 252,
) -> int:
    """
    Cold-start vectorized backfill for `market_breadth_daily` when a breadth cohort
    (such as `US_IXIC`) has fewer than 60 rows. Ensures Day-1 Breadth Velocity,
    Stealth Top decay, Cumulative A/D, and Zweig 7D 9:1 work cleanly from session 1.
    """
    if close_core is None or close_core.empty or len(close_core) < 55:
        return 0
    try:
        conn = get_connection()
        c = conn.cursor()
        cnt = c.execute("SELECT COUNT(*) FROM market_breadth_daily WHERE market = ?", (market,)).fetchone()[0]
        if int(cnt or 0) >= 60:
            conn.close()
            return 0

        rets = close_core.pct_change(fill_method=None)
        ret_5d = (close_core / close_core.shift(5)) - 1.0
        sma20 = close_core.rolling(20, min_periods=15).mean()
        sma50 = close_core.rolling(50, min_periods=30).mean()
        sma200 = close_core.rolling(200, min_periods=100).mean()
        roll_hi = high_core.rolling(252, min_periods=60).max().shift(1)
        roll_lo = low_core.rolling(252, min_periods=60).min().shift(1)

        s_thresh = 0.12 if market == 'IN' else 0.20
        p_thresh = -0.12 if market == 'IN' else -0.20
        has_vol = vol_core is not None and not vol_core.empty

        # Mask extreme single-day downward/synthetic split anomalies from 5D Bonde counts
        clean_5d_ok = (rets.abs() <= 0.85) | ((rets > 0.85) & (vol_core >= vol_core.rolling(20, min_periods=5).mean() * 2.0)) if has_vol else (rets.abs() <= 0.85)
        split_5d_bad = (~clean_5d_ok).rolling(5, min_periods=1).max().fillna(0).astype(bool)

        start_pos = max(5, len(close_core) - int(lookback_days))
        ad_cum = 0
        vol_cum = 0.0
        records = []

        for pos in range(start_pos, len(close_core)):
            dt_idx = close_core.index[pos]
            dt_str = pd.to_datetime(dt_idx).strftime('%Y-%m-%d')
            last_c = close_core.iloc[pos]
            vm = last_c.notna()
            n_valid = int(vm.sum())
            if n_valid < 50:
                continue

            lr = rets.iloc[pos][vm]
            adv = int((lr > 0).sum())
            dec = int((lr < 0).sum())
            ad_cum += (adv - dec)

            abv20 = float((last_c[vm] > sma20.iloc[pos][vm]).mean() * 100.0)
            abv50 = float((last_c[vm] > sma50.iloc[pos][vm]).mean() * 100.0)
            abv200 = float((last_c[vm] > sma200.iloc[pos][vm]).mean() * 100.0)

            nh = int((high_core.iloc[pos][vm] >= (roll_hi.iloc[pos][vm] * 0.96)).sum())
            nl = int((low_core.iloc[pos][vm] <= (roll_lo.iloc[pos][vm] * 1.04)).sum())
            nnh = nh - nl
            d52 = int((last_c[vm] >= (roll_lo.iloc[pos][vm] * 2.0)).sum())
            q52 = int((last_c[vm] >= (roll_lo.iloc[pos][vm] * 4.0)).sum())

            r5_v = ret_5d.iloc[pos][vm]
            r5_clean = r5_v[~split_5d_bad.iloc[pos][vm]]
            surge_cnt = int((r5_clean >= s_thresh).sum())
            panic_cnt = int((r5_clean <= p_thresh).sum())

            uv, dv = 0.0, 0.0
            if has_vol and pos < len(vol_core):
                lv = vol_core.iloc[pos][vm].fillna(0.0)
                uv = float(lv[lr > 0].sum())
                dv = float(lv[lr < 0].sum())
            ud_r = round(float(uv / dv), 2) if dv > 0 else 1.0
            vol_cum += (uv - dv)

            records.append((
                dt_str, market, round(abv20, 2), round(abv50, 2), round(abv200, 2),
                nnh, nh, nl, adv, dec, ad_cum, d52, q52, n_valid, surge_cnt, panic_cnt,
                uv, dv, ud_r, vol_cum, 0, 0, 0, 0, 0
            ))

        if records:
            c.executemany("""
                INSERT OR REPLACE INTO market_breadth_daily
                (date, market, above_20_pct, above_50_pct, above_200_pct, net_new_highs,
                 new_highs_count, new_lows_count, advances, declines, ad_line_cumulative,
                 double_52w_count, quadruple_52w_count, total_stocks, surge_extreme_5d, panic_extreme_5d,
                 up_volume, down_volume, up_down_volume_ratio, cumulative_volume_ad_line,
                 ad_grade_a_count, ad_grade_b_count, ad_grade_c_count, ad_grade_d_count, ad_grade_e_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()
        conn.close()
        return len(records)
    except Exception as e:
        print(f"[BrainEngine] Cold-start breadth backfill warning ({market}): {e}")
        return 0


def _self_heal_market_breadth_row(
    bar_date_str: str,
    close_core: pd.DataFrame,
    high_core: pd.DataFrame,
    low_core: pd.DataFrame,
    vol_core: pd.DataFrame = None,
    market: str = 'IN'
):
    """
    Ensures market_breadth_daily has today's row with complete institutional breadth,
    exchange volume, and IBD GMI 13-week A-E money flow distribution metrics.
    Automatically replaces partial/intraday rows when fuller universe/volume data arrives.
    """
    if close_core is None or close_core.empty or len(close_core) < 55:
        return
    try:
        rets = close_core.pct_change(fill_method=None)
        last_c = close_core.iloc[-1]
        valid_mask = last_c.notna()
        n_valid = int(valid_mask.sum())
        if n_valid < 50:
            return

        last_r = rets.iloc[-1][valid_mask]
        fresh_up_vol = 0.0
        fresh_dn_vol = 0.0
        if vol_core is not None and not vol_core.empty and len(vol_core) >= 1:
            last_v = vol_core.iloc[-1][valid_mask].fillna(0)
            fresh_up_vol = float(last_v[last_r > 0].sum())
            fresh_dn_vol = float(last_v[last_r < 0].sum())
        fresh_tot_vol = fresh_up_vol + fresh_dn_vol

        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            SELECT total_stocks, up_volume, down_volume, ad_grade_a_count
            FROM market_breadth_daily
            WHERE date = ? AND market = ?
        """, (bar_date_str, market))
        existing = c.fetchone()
        # Skip only when existing row already reflects >=95% of active universe AND >=70% of volume
        if existing is not None:
            ex_n = int(existing[0] or 0)
            ex_uv = float(existing[1] or 0.0)
            ex_dv = float(existing[2] or 0.0)
            ex_ga = int(existing[3] or 0)
            has_full_universe = (ex_n >= int(n_valid * 0.95))
            has_full_volume = (fresh_tot_vol <= 0) or ((ex_uv + ex_dv) >= fresh_tot_vol * 0.70)
            if ex_uv > 0 and ex_ga > 0 and has_full_universe and has_full_volume:
                conn.close()
                return

        sma20 = close_core.rolling(20, min_periods=15).mean()
        sma50 = close_core.rolling(50, min_periods=30).mean()
        sma200 = close_core.rolling(200, min_periods=100).mean()

        abv20 = float((last_c[valid_mask] > sma20.iloc[-1][valid_mask]).mean() * 100.0)
        abv50 = float((last_c[valid_mask] > sma50.iloc[-1][valid_mask]).mean() * 100.0)
        abv200 = float((last_c[valid_mask] > sma200.iloc[-1][valid_mask]).mean() * 100.0)

        adv = int((last_r > 0).sum())
        dec = int((last_r < 0).sum())

        roll_hi = high_core.rolling(252, min_periods=60).max().shift(1)
        roll_lo = low_core.rolling(252, min_periods=60).min().shift(1)
        nh = int((high_core.iloc[-1][valid_mask] >= (roll_hi.iloc[-1][valid_mask] * (1 - 0.04))).sum())
        nl = int((low_core.iloc[-1][valid_mask] <= (roll_lo.iloc[-1][valid_mask] * (1 + 0.04))).sum())
        nnh = nh - nl

        double_52w = int((last_c[valid_mask] >= (roll_lo.iloc[-1][valid_mask] * 2)).sum())
        quad_52w = int((last_c[valid_mask] >= (roll_lo.iloc[-1][valid_mask] * 4)).sum())

        ret_5d = close_core / close_core.shift(5) - 1
        s_thresh = 0.12 if market == 'IN' else 0.20
        p_thresh = -0.12 if market == 'IN' else -0.20
        surge_count = int((ret_5d.iloc[-1][valid_mask] >= s_thresh).sum())
        panic_count = int((ret_5d.iloc[-1][valid_mask] <= p_thresh).sum())

        # Up / Down Volume Breadth
        up_vol = fresh_up_vol
        down_vol = fresh_dn_vol
        ud_ratio = round(float(up_vol / down_vol), 2) if down_vol > 0 else 1.0

        # Continuous Cumulative A/D and Volume A/D Lines
        c.execute("""
            SELECT ad_line_cumulative, cumulative_volume_ad_line 
            FROM market_breadth_daily 
            WHERE market = ? AND date < ? 
            ORDER BY date DESC LIMIT 1
        """, (market, bar_date_str))
        prev_cum = c.fetchone()
        prev_ad_cum = int(prev_cum[0]) if (prev_cum and prev_cum[0] is not None) else 0
        prev_vol_cum = float(prev_cum[1]) if (prev_cum and len(prev_cum) > 1 and prev_cum[1] is not None) else 0.0
        ad_line_cumulative = prev_ad_cum + (adv - dec)
        cumulative_volume_ad_line = prev_vol_cum + (up_vol - down_vol)

        # Universe A/B/C/D/E Distribution Counts (IBD GMI Trailing 65-Day Money Flow)
        a_cnt, b_cnt, c_cnt, d_cnt, e_cnt = 0, 0, 0, 0, 0
        if vol_core is not None and not vol_core.empty and len(close_core) >= 65 and len(vol_core) >= 65:
            vol_sma50 = vol_core.rolling(50, min_periods=20).mean()
            vol_ratio_matrix = (vol_core / vol_sma50).clip(0.2, 3.5)
            hl_range = (high_core - low_core).replace(0, np.nan)
            clv_matrix = (((close_core - low_core) - (high_core - close_core)) / hl_range).fillna(0).clip(-1, 1)
            daily_flow_matrix = (0.6 * np.sign(rets).fillna(0) + 0.4 * clv_matrix) * vol_ratio_matrix
            weights_65 = np.linspace(1.0, 2.0, 65)
            weights_norm_65 = weights_65 / weights_65.sum()
            sub_flow = daily_flow_matrix.iloc[-65:]
            w_scores = np.dot(weights_norm_65, sub_flow.fillna(0).values) * 20.0
            a_cnt = int((w_scores >= 2.0).sum())
            b_cnt = int(((w_scores >= 0.6) & (w_scores < 2.0)).sum())
            c_cnt = int(((w_scores >= -0.6) & (w_scores < 0.6)).sum())
            d_cnt = int(((w_scores >= -2.0) & (w_scores < -0.6)).sum())
            e_cnt = int((w_scores < -2.0).sum())

        c.execute("""
            INSERT OR REPLACE INTO market_breadth_daily
            (date, market, above_20_pct, above_50_pct, above_200_pct, net_new_highs, 
             new_highs_count, new_lows_count, advances, declines, ad_line_cumulative, 
             double_52w_count, quadruple_52w_count, total_stocks, surge_extreme_5d, panic_extreme_5d, 
             up_volume, down_volume, up_down_volume_ratio, cumulative_volume_ad_line, 
             ad_grade_a_count, ad_grade_b_count, ad_grade_c_count, ad_grade_d_count, ad_grade_e_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            bar_date_str, market, round(abv20, 2), round(abv50, 2), round(abv200, 2),
            nnh, nh, nl, adv, dec, ad_line_cumulative, double_52w, quad_52w,
            n_valid, surge_count, panic_count,
            up_vol, down_vol, ud_ratio, cumulative_volume_ad_line,
            a_cnt, b_cnt, c_cnt, d_cnt, e_cnt
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[BrainEngine] Breadth self-heal warning: {e}")


def compute_breadth_velocity(close_core: pd.DataFrame = None, breadth_df: pd.DataFrame = None) -> dict:
    """
    Computes rolling 5-day and 10-day Rate of Change (Delta ROC in percentage points) for:
      - % of stocks > 50-day SMA
      - % of stocks > 20-day EMA
    Distinguishes institutional thrusts (high acceleration) from lazy dead-cat bounce traps.
    Accepts either a cross-sectional close matrix (close_core) or a historical breadth DataFrame (breadth_df).
    """
    default_res = {
        'roc_5d_50sma': 0.0,
        'roc_10d_50sma': 0.0,
        'roc_5d_20ema': 0.0,
        'roc_10d_20ema': 0.0,
        'curr_abv50': 50.0,
        'curr_abv20': 50.0,
        'is_thrust': False,
        'velocity_status': 'NEUTRAL_COMPRESSION',
        'velocity_label': '⚪ Neutral Breadth Compression',
        'velocity_badge': '⚪ BREADTH COMPRESSION',
        'velocity_color': '#94a3b8',
        'velocity_bg': 'rgba(148, 163, 184, 0.15)',
        'velocity_border': 'rgba(148, 163, 184, 0.35)',
    }

    # Support passing breadth_df as first argument
    if breadth_df is None and close_core is not None and ('above_50_pct' in close_core.columns or 'above_20_pct' in close_core.columns):
        breadth_df = close_core
        close_core = None

    if breadth_df is None and (close_core is None or close_core.empty or len(close_core) < 15):
        return default_res

    try:
        if breadth_df is not None and not breadth_df.empty and 'above_50_pct' in breadth_df.columns:
            abv50_series = pd.to_numeric(breadth_df['above_50_pct'], errors='coerce').dropna()
            if len(abv50_series) >= 25:
                if 'above_20_pct' in breadth_df.columns:
                    abv20_series = pd.to_numeric(breadth_df['above_20_pct'], errors='coerce').dropna()
                else:
                    abv20_series = abv50_series
            else:
                abv50_series = pd.Series(dtype=float)
        else:
            abv50_series = pd.Series(dtype=float)

        if abv50_series.empty and close_core is not None and not close_core.empty and len(close_core) >= 15:
            num_core = close_core.select_dtypes(include=[np.number])
            if num_core.empty or len(num_core) < 15:
                return default_res
            tail_len = min(60, len(num_core))
            c_sub = num_core.iloc[-tail_len:]
            sma50_df = num_core.rolling(50, min_periods=25).mean().iloc[-tail_len:]
            ema20_df = num_core.ewm(span=20, adjust=False).mean().iloc[-tail_len:]
            abv50_series = (c_sub > sma50_df).mean(axis=1) * 100.0
            abv20_series = (c_sub > ema20_df).mean(axis=1) * 100.0

        if abv50_series.empty:
            return default_res

        curr_abv50 = float(abv50_series.iloc[-1])
        curr_abv20 = float(abv20_series.iloc[-1]) if not abv20_series.empty else curr_abv50

        idx_5d = -6 if len(abv50_series) >= 6 else 0
        roc_5d_50 = float(curr_abv50 - abv50_series.iloc[idx_5d])
        idx_5d_20 = -6 if len(abv20_series) >= 6 else 0
        roc_5d_20 = float(curr_abv20 - abv20_series.iloc[idx_5d_20]) if not abv20_series.empty else 0.0

        idx_10d = -11 if len(abv50_series) >= 11 else 0
        roc_10d_50 = float(curr_abv50 - abv50_series.iloc[idx_10d])
        idx_10d_20 = -11 if len(abv20_series) >= 11 else 0
        roc_10d_20 = float(curr_abv20 - abv20_series.iloc[idx_10d_20]) if not abv20_series.empty else 0.0

        if roc_5d_50 >= 5.0 or roc_5d_20 >= 20.0:
            is_thrust = True
            v_status = 'THRUST_CONFIRMED'
            v_label = '⚡ Institutional Thrust (High Acceleration)'
            v_badge = '⚡ THRUST CONFIRMED'
            v_color = '#10b981'
            v_bg = 'rgba(16, 185, 129, 0.15)'
            v_border = 'rgba(16, 185, 129, 0.35)'
        elif roc_5d_50 >= 1.0 or roc_5d_20 >= 5.0:
            is_thrust = False
            v_status = 'LAZY_DRIFT'
            v_label = '⚠️ Lazy Drift (Lacks Institutional Impulse)'
            v_badge = '⚠️ LAZY DRIFT TRAP'
            v_color = '#f59e0b'
            v_bg = 'rgba(245, 158, 11, 0.15)'
            v_border = 'rgba(245, 158, 11, 0.35)'
        elif roc_5d_50 <= -3.0 or roc_5d_20 <= -10.0:
            is_thrust = False
            v_status = 'ACCELERATING_LIQUIDATION'
            v_label = '🔻 Accelerating Liquidation (Supply Expanding)'
            v_badge = '🔻 LIQUIDATION EXPANDING'
            v_color = '#ef4444'
            v_bg = 'rgba(239, 68, 68, 0.15)'
            v_border = 'rgba(239, 68, 68, 0.35)'
        else:
            is_thrust = False
            v_status = 'NEUTRAL_COMPRESSION'
            v_label = '⚪ Neutral Breadth Compression'
            v_badge = '⚪ BREADTH COMPRESSION'
            v_color = '#94a3b8'
            v_bg = 'rgba(148, 163, 184, 0.15)'
            v_border = 'rgba(148, 163, 184, 0.35)'

        return {
            'roc_5d_50sma': round(roc_5d_50, 1),
            'roc_10d_50sma': round(roc_10d_50, 1),
            'roc_5d_20ema': round(roc_5d_20, 1),
            'roc_10d_20ema': round(roc_10d_20, 1),
            'curr_abv50': round(curr_abv50, 1),
            'curr_abv20': round(curr_abv20, 1),
            'is_thrust': is_thrust,
            'velocity_status': v_status,
            'velocity_label': v_label,
            'velocity_badge': v_badge,
            'velocity_color': v_color,
            'velocity_bg': v_bg,
            'velocity_border': v_border,
        }
    except Exception as e:
        print(f"[BrainEngine] Breadth velocity calculation warning: {e}")
        return default_res


def compute_stealth_top_state_machine(
    benchmark_df: pd.DataFrame = None,
    close_core: pd.DataFrame = None,
    high_core: pd.DataFrame = None,
    low_core: pd.DataFrame = None,
    breakout_health: float = 50.0,
    dd_count: int = 0,
    breadth_df: pd.DataFrame = None
) -> dict:
    """
    Evaluates the 4-Level Automated Stealth Top & Distribution State Machine.
    Identifies multi-week breadth rot, mega-cap masking, and distribution day clusters.
    Accepts either cross-sectional matrices or a historical breadth DataFrame.
    """
    default_res = {
        'level_num': 0,
        'level_code': 'HEALTHY_BULL',
        'level_title': 'Level 0: Healthy Bull Expansion',
        'level_badge': '🟢 TOPPING RISK: LEVEL 0 (HEALTHY)',
        'level_color': '#10b981',
        'level_bg': 'rgba(16, 185, 129, 0.15)',
        'level_border': 'rgba(16, 185, 129, 0.35)',
        'directive': 'Full Institutional Tailwind (80%–100% Exposure). No breadth divergence, distribution contained.',
        'breadth_decay_pct': 0.0,
        'max_breadth_60d': 50.0,
        'curr_abv50': 50.0,
        'dist_52w_high_pct': 0.0,
        'net_new_highs': 0,
        'new_highs_count': 0,
        'new_lows_count': 0,
        'is_near_highs': False,
        'is_stealth_rot': False,
    }

    # Handle Series benchmark
    if isinstance(benchmark_df, pd.Series):
        benchmark_df = benchmark_df.to_frame(name='close')

    # Support passing breadth_df as close_core
    if breadth_df is None and close_core is not None and ('above_50_pct' in close_core.columns or 'new_highs_count' in close_core.columns):
        breadth_df = close_core
        close_core = None

    if (benchmark_df is None or benchmark_df.empty or len(benchmark_df) < 15) and (breadth_df is None or breadth_df.empty):
        return default_res

    try:
        curr_idx = 100.0
        curr_50 = 100.0
        idx_below_50 = False
        dist_52w_high_pct = 0.0
        is_near_highs = False

        if benchmark_df is not None and not benchmark_df.empty and len(benchmark_df) >= 15:
            b_df = benchmark_df.copy()
            c_col = 'close' if 'close' in b_df.columns else ('Close' if 'Close' in b_df.columns else b_df.columns[0])
            b_close = pd.to_numeric(b_df[c_col], errors='coerce').dropna()
            if not b_close.empty:
                curr_idx = float(b_close.iloc[-1])
                b_sma50 = b_close.rolling(50, min_periods=min(15, len(b_close))).mean()
                curr_50 = float(b_sma50.iloc[-1])
                idx_below_50 = curr_idx < curr_50
                roll_span = min(252, len(b_close))
                high_52w = float(b_close.rolling(roll_span, min_periods=min(30, roll_span)).max().iloc[-1])
                dist_52w_high_pct = round(((high_52w - curr_idx) / high_52w) * 100.0, 1) if high_52w > 0 else 0.0
                is_near_highs = dist_52w_high_pct <= 2.5

        curr_abv50 = 50.0
        max_breadth_60d = 50.0
        breadth_decay = 0.0

        abv50_db = pd.Series(dtype=float)
        if breadth_df is not None and not breadth_df.empty and 'above_50_pct' in breadth_df.columns:
            tail_len = min(60, len(breadth_df))
            abv50_db = pd.to_numeric(breadth_df['above_50_pct'], errors='coerce').dropna().tail(tail_len)

        if len(abv50_db) >= 25:
            curr_abv50 = float(abv50_db.iloc[-1])
            max_breadth_60d = float(abv50_db.max())
            breadth_decay = round(max(0.0, max_breadth_60d - curr_abv50), 1)
        elif close_core is not None and not close_core.empty and len(close_core) >= 25:
            num_core = close_core.select_dtypes(include=[np.number])
            if not num_core.empty and len(num_core) >= 25:
                tail_len = min(60, len(num_core))
                c_sub = num_core.iloc[-tail_len:]
                sma50_df = num_core.rolling(50, min_periods=25).mean().iloc[-tail_len:]
                abv50_s = (c_sub > sma50_df).mean(axis=1) * 100.0
                curr_abv50 = float(abv50_s.iloc[-1])
                max_breadth_60d = float(abv50_s.max())
                breadth_decay = round(max(0.0, max_breadth_60d - curr_abv50), 1)
        elif not abv50_db.empty:
            curr_abv50 = float(abv50_db.iloc[-1])
            max_breadth_60d = float(abv50_db.max())
            breadth_decay = round(max(0.0, max_breadth_60d - curr_abv50), 1)

        is_stealth_rot = is_near_highs and (breadth_decay >= 15.0)

        nh, nl, nnh = 0, 0, 0
        core_n = 750
        if breadth_df is not None and not breadth_df.empty and 'total_stocks' in breadth_df.columns and pd.notna(breadth_df['total_stocks'].iloc[-1]):
            core_n = max(100, int(breadth_df['total_stocks'].iloc[-1] or 750))
        elif close_core is not None and not close_core.empty:
            core_n = max(100, int(close_core.iloc[-1].notna().sum()))

        if breadth_df is not None and not breadth_df.empty and 'new_highs_count' in breadth_df.columns and 'new_lows_count' in breadth_df.columns:
            nh = int(pd.to_numeric(breadth_df['new_highs_count'], errors='coerce').fillna(0).iloc[-1])
            nl = int(pd.to_numeric(breadth_df['new_lows_count'], errors='coerce').fillna(0).iloc[-1])
            nnh = nh - nl
        elif high_core is not None and low_core is not None and not high_core.empty and not low_core.empty and len(high_core) >= 60:
            num_hi = high_core.select_dtypes(include=[np.number])
            num_lo = low_core.select_dtypes(include=[np.number])
            roll_hi = num_hi.rolling(252, min_periods=40).max().shift(1)
            roll_lo = num_lo.rolling(252, min_periods=40).min().shift(1)
            last_mask = num_hi.iloc[-1].notna()
            nh = int((num_hi.iloc[-1][last_mask] >= roll_hi.iloc[-1][last_mask]).sum())
            nl = int((num_lo.iloc[-1][last_mask] <= roll_lo.iloc[-1][last_mask]).sum())
            nnh = nh - nl

        nl_dom_thresh = max(30, int(round(core_n * 0.04)))
        net_lows_dominant = (nl > nh) and (nl >= nl_dom_thresh)

        if idx_below_50 or curr_abv50 < 30.0:
            lvl_num = 3
            lvl_code = 'FROZEN_DECK'
            lvl_title = 'Level 3: Frozen Deck / Structural Breakdown'
            lvl_badge = '❄️ TOPPING RISK: LEVEL 3 (FROZEN DECK)'
            lvl_color = '#ef4444'
            lvl_bg = 'rgba(239, 68, 68, 0.15)'
            lvl_border = 'rgba(239, 68, 68, 0.35)'
            directive = 'Cash Fortress (0%–10% Exposure). Index below 50-SMA or broad breadth broken. Enforce strict risk discipline.'
        elif dd_count >= 5 or net_lows_dominant or breakout_health < 35.0:
            lvl_num = 2
            lvl_code = 'DISTRIBUTION_INVERSION'
            lvl_title = 'Level 2: Distribution Inversion (Heavy Supply Overhang)'
            lvl_badge = '🔴 TOPPING RISK: LEVEL 2 (DISTRIBUTION INVERSION)'
            lvl_color = '#f87171'
            lvl_bg = 'rgba(248, 113, 113, 0.15)'
            lvl_border = 'rgba(248, 113, 113, 0.35)'
            directive = f"Defensive Pivot (20%–30% Exposure). Heavy distribution ({dd_count} DDs, {nl} New Lows, Breakout Health {breakout_health:.0f}%). Raise cash into strength."
        elif is_stealth_rot:
            lvl_num = 1
            lvl_code = 'STEALTH_ROT_WARNING'
            lvl_title = 'Level 1: Stealth Rot Warning (Mega-Cap Masking Alert)'
            lvl_badge = '🟡 TOPPING RISK: LEVEL 1 (STEALTH ROT WARNING)'
            lvl_color = '#f59e0b'
            lvl_bg = 'rgba(245, 158, 11, 0.15)'
            lvl_border = 'rgba(245, 158, 11, 0.35)'
            directive = f"Mega-Cap Masking Alert (40%–50% Max Exposure). Breadth down -{breadth_decay:.1f}% from 60D peak while index is within {dist_52w_high_pct:.1f}% of highs! Tighten stops to 10-EMA."
        else:
            lvl_num = 0
            lvl_code = 'HEALTHY_BULL'
            lvl_title = 'Level 0: Healthy Bull Expansion'
            lvl_badge = '🟢 TOPPING RISK: LEVEL 0 (HEALTHY)'
            lvl_color = '#10b981'
            lvl_bg = 'rgba(16, 185, 129, 0.15)'
            lvl_border = 'rgba(16, 185, 129, 0.35)'
            directive = 'Full Institutional Tailwind (80%–100% Exposure). No breadth divergence, distribution contained.'

        return {
            'level_num': lvl_num,
            'level_code': lvl_code,
            'level_title': lvl_title,
            'level_badge': lvl_badge,
            'level_color': lvl_color,
            'level_bg': lvl_bg,
            'level_border': lvl_border,
            'directive': directive,
            'breadth_decay_pct': breadth_decay,
            'max_breadth_60d': round(max_breadth_60d, 1),
            'curr_abv50': round(curr_abv50, 1),
            'dist_52w_high_pct': dist_52w_high_pct,
            'net_new_highs': nnh,
            'new_highs_count': nh,
            'new_lows_count': nl,
            'is_near_highs': is_near_highs,
            'is_stealth_rot': is_stealth_rot,
        }
    except Exception as e:
        print(f"[BrainEngine] Stealth top calculation warning: {e}")
        return default_res


def compute_tide_turn_state_machine(
    benchmark_df: pd.DataFrame,
    close_core: pd.DataFrame,
    high_core: pd.DataFrame,
    low_core: pd.DataFrame,
    vol_core: pd.DataFrame,
    ironclad_count: int = 0,
    persona_data: dict = None,
    breadth_df: pd.DataFrame = None,
    market: str = 'INDIA',
) -> dict:
    """
    Evaluates the 5-Stage Automated Bear-to-Bull Tide-Turn State Machine on the market's
    Liquid Core (Nifty 750 or Liquid NASDAQ) & official Benchmark (^CRSLDX or ^IXIC),
    enhanced with Breadth Velocity (Delta ROC), Zweig 9:1, and Stealth Top Guardrails.
    """
    prof = get_market_brain_profile(market)
    bonde_thresh = float(prof['bonde_pct'])
    # Default telemetry
    res = {
        'stage_num': 3,
        'stage_code': 'TACTICAL_PILOT',
        'stage_title': 'Stage 3: Tactical Tide-Turn / Pilot Deck',
        'stage_badge': '⚡ STAGE 3: TACTICAL PILOT DECK',
        'stage_color': '#f59e0b',
        'target_exposure': '15% – 30% Equity (0.35% Risk Pilots)',
        'directive': 'Deploy progressive pilot positions in Top Confluence Leaders holding 21-EMA Low.',
        'above_50sma_pct': 50.0,
        'above_20ema_pct': 50.0,
        'fomo_5ema_pct': 50.0,
        'panic_5d_count': 0,
        'surge_5d_count': 0,
        'up_vol_ratio_pct': 50.0,
        'is_9to1_thrust': False,
        'last_9to1_date': None,
        'last_9to1_ratio': 0.0,
        'recapture_step': 0,
        'recapture_label': 'Monitoring',
        'recapture_summary': 'Step 0/7: Monitoring',
        'ftd_active': False,
        'ftd_date': None,
        'ftd_gain_pct': 0.0,
        'ftd_vol_ratio': 1.0,
        'ftd_rally_day': 0,
        'ftd_rally_low': 0.0,
        'ftd_sessions_ago': -1,
        'recent_ftd_events': [],
        'ftd_suspect_warning': False,
        'ftd_status_text': 'No Active FTD',
        'dd_count': 0,
        'ironclad_count': int(ironclad_count),
        'breakout_health_pct': 50.0,
        'bh_is_pre_ftd_lag': False,
        'bh_audit_label': '50%',
        'top100_above_21_pct': None,
        'stages_bar': [
            {'num': 1, 'name': '1. Washout', 'sub': '0% Cash'},
            {'num': 2, 'name': '2. Beach Ball Div', 'sub': '0-10% Stalk'},
            {'num': 3, 'name': '3. 20EMA / 9:1 Pilot', 'sub': '15-30% Pilots'},
            {'num': 4, 'name': '4. Validated FTD', 'sub': '50-75% Scale'},
            {'num': 5, 'name': '5. Power Trend', 'sub': '85-100% Full'},
        ]
    }

    # 1. Compute Breadth Velocity (5D & 10D ROC) using canonical breadth_df
    vel = compute_breadth_velocity(close_core=close_core, breadth_df=breadth_df)
    res['velocity'] = vel

    if close_core is not None and not close_core.empty and len(close_core) >= 25:
        last_c = close_core.iloc[-1]
        valid = last_c.notna()
        if valid.sum() >= 30:
            ema5 = close_core.ewm(span=5, adjust=False).mean().iloc[-1]
            ema20 = close_core.ewm(span=20, adjust=False).mean().iloc[-1]
            sma50 = close_core.rolling(50, min_periods=25).mean().iloc[-1]

            res['fomo_5ema_pct'] = round(float((last_c[valid] > ema5[valid]).mean() * 100.0), 1)
            res['above_20ema_pct'] = round(float((last_c[valid] > ema20[valid]).mean() * 100.0), 1)
            res['above_50sma_pct'] = round(float((last_c[valid] > sma50[valid]).mean() * 100.0), 1)

            # Pradeep Bonde 5D Panic & 5D Surge (India ±12%, US ±20%)
            if len(close_core) >= 6:
                ret_5d = (close_core.iloc[-1] / close_core.iloc[-6]) - 1.0
                res['panic_5d_count'] = int((ret_5d[valid] <= -bonde_thresh).sum())
                res['surge_5d_count'] = int((ret_5d[valid] >= bonde_thresh).sum())

    # Override breadth & Bonde counts with canonical breadth_df for 100% parity with Market Breadth page
    if breadth_df is not None and not breadth_df.empty:
        b_latest = breadth_df.iloc[-1]
        if 'above_50_pct' in breadth_df.columns and pd.notna(b_latest.get('above_50_pct')):
            res['above_50sma_pct'] = round(float(b_latest['above_50_pct']), 1)
        if 'above_20_pct' in breadth_df.columns and pd.notna(b_latest.get('above_20_pct')):
            res['above_20ema_pct'] = round(float(b_latest['above_20_pct']), 1)
        if 'panic_extreme_5d' in breadth_df.columns and pd.notna(b_latest.get('panic_extreme_5d')) and int(b_latest.get('panic_extreme_5d') or 0) >= 0:
            res['panic_5d_count'] = int(b_latest['panic_extreme_5d'])
        if 'surge_extreme_5d' in breadth_df.columns and pd.notna(b_latest.get('surge_extreme_5d')) and int(b_latest.get('surge_extreme_5d') or 0) >= 0:
            res['surge_5d_count'] = int(b_latest['surge_extreme_5d'])

    # Clean Zweig >= 90% (9:1) Up-Volume Thrust detection (canonical breadth_df first, thin-session guarded)
    zweig_res = _evaluate_zweig_9to1_thrust(close_core=close_core, vol_core=vol_core, breadth_df=breadth_df)
    res['up_vol_ratio_pct'] = zweig_res['up_vol_ratio_pct']
    res['is_9to1_thrust'] = zweig_res['is_9to1_thrust']
    res['last_9to1_date'] = zweig_res['last_9to1_date']
    res['last_9to1_ratio'] = zweig_res['last_9to1_ratio']

    # Benchmark checks
    idx_above_50 = False
    idx_above_21 = False
    power_trend_aligned = False
    if benchmark_df is not None and not benchmark_df.empty and len(benchmark_df) >= 20:
        b_df = benchmark_df.copy()
        c_col = 'close' if 'close' in b_df.columns else 'Close'
        b_close = pd.to_numeric(b_df[c_col], errors='coerce')
        b_ema10 = b_close.ewm(span=10, adjust=False).mean()
        b_ema21 = b_close.ewm(span=21, adjust=False).mean()
        b_sma50 = b_close.rolling(50, min_periods=20).mean()

        curr_idx = float(b_close.iloc[-1])
        curr_21 = float(b_ema21.iloc[-1])
        curr_50 = float(b_sma50.iloc[-1])
        idx_above_21 = curr_idx >= curr_21
        idx_above_50 = curr_idx >= curr_50
        ema21_prev5 = float(b_ema21.iloc[-5]) if len(b_ema21) >= 5 else float(b_ema21.iloc[0])
        power_trend_aligned = (float(b_ema10.iloc[-1]) >= curr_21 >= curr_50) and (curr_21 > ema21_prev5)

        dd_cnt, _ = calculate_distribution_days(b_df)
        res['dd_count'] = int(dd_cnt)

        try:
            recap = detect_20ema_recapture_and_sequence(b_df)
            res['recapture_step'] = int(recap.get('current_step_num', 0) or 0)
            res['recapture_label'] = str(recap.get('current_step_name', 'Monitoring'))
            res['recapture_summary'] = str(recap.get('sequence_summary', f"Step {res['recapture_step']}/7"))
        except Exception as e:
            print(f"[BrainEngine] 20-EMA sequence warning: {e}")

        try:
            ftd_det, ftd_dt_str, ftd_details = detect_follow_through_day(b_df, return_details=True)
            res['ftd_active'] = bool(ftd_det)
            res['ftd_date'] = str(ftd_dt_str)[:10] if ftd_dt_str else None
            if isinstance(ftd_details, dict):
                res['ftd_gain_pct'] = float(ftd_details.get('ftd_gain_pct', 0.0) or 0.0)
                res['ftd_vol_ratio'] = float(ftd_details.get('ftd_vol_ratio', 1.0) or 1.0)
                res['ftd_rally_day'] = int(ftd_details.get('rally_day', 0) or 0)
                res['ftd_rally_low'] = float(ftd_details.get('rally_low', 0.0) or 0.0)
                s_ago = ftd_details.get('ftd_sessions_ago', -1)
                res['ftd_sessions_ago'] = int(s_ago) if s_ago is not None else -1
                res['recent_ftd_events'] = list(ftd_details.get('recent_ftd_events', []) or [])
                if ftd_det and res['ftd_date']:
                    day_sfx = f", Day {res['ftd_rally_day']}" if res['ftd_rally_day'] > 0 else ""
                    res['ftd_status_text'] = f"Confirmed: {res['ftd_date']} (+{res['ftd_gain_pct']:.1f}%{day_sfx})"
                elif ftd_details.get('rally_low_undercut'):
                    f_dt = str(ftd_details.get('ftd_date') or '')[:10]
                    res['ftd_status_text'] = f"Failed ({f_dt} Undercut < {res['ftd_rally_low']:,.0f})"
                elif res['ftd_rally_day'] > 0:
                    res['ftd_status_text'] = f"Rally Attempt Day {res['ftd_rally_day']} (Guard > {res['ftd_rally_low']:,.0f})"
        except Exception as e:
            print(f"[BrainEngine] FTD detection warning: {e}")

    persona_score = 50.0
    breakout_health = 55.0
    top100_above_21 = None
    if persona_data:
        persona_score = float(persona_data.get('composite_score', 50.0) or 50.0)
        breakout_health = float(persona_data.get('breakout_health_pct', 55.0) or 55.0)
        raw_l21 = persona_data.get('leader_21ema_pct')
        if raw_l21 is None and isinstance(persona_data.get('details'), dict):
            raw_l21 = persona_data['details'].get('pillar3', {}).get('leader_21ema_pct')
        if raw_l21 is not None:
            top100_above_21 = float(raw_l21)
    res['breakout_health_pct'] = round(breakout_health, 1)
    res['top100_above_21_pct'] = round(top100_above_21, 1) if top100_above_21 is not None else None

    # 2. Compute Stealth Top State Machine & Wire Net New Highs/Lows using canonical breadth_df
    stop_matrix = compute_stealth_top_state_machine(
        benchmark_df=benchmark_df,
        close_core=close_core,
        high_core=high_core,
        low_core=low_core,
        breakout_health=breakout_health,
        dd_count=res['dd_count'],
        breadth_df=breadth_df
    )
    res['stealth_top'] = stop_matrix
    res['new_highs_count'] = stop_matrix.get('new_highs_count', 0)
    res['new_lows_count'] = stop_matrix.get('new_lows_count', 0)
    res['net_new_highs'] = stop_matrix.get('net_new_highs', 0)

    # Cohort-aware Suspect FTD & Bull-Trap Guardrail:
    # Separate lagging pre-FTD T+10 Breakout Health (< 10 sessions post-FTD) from real post-FTD failure.
    is_early_post_ftd = res['ftd_active'] and (0 <= res['ftd_sessions_ago'] < 10)
    res['bh_is_pre_ftd_lag'] = bool(is_early_post_ftd and breakout_health < 45.0)
    res['bh_audit_label'] = (
        f"Pre-FTD T+10 Lag ({breakout_health:.0f}%)"
        if res['bh_is_pre_ftd_lag']
        else f"{breakout_health:.0f}%"
    )

    leaders_failing = (top100_above_21 is not None and top100_above_21 < 50.0)
    post_ftd_squat_fail = (not is_early_post_ftd) and (breakout_health < 40.0)
    weak_breadth_or_stealth = (stop_matrix.get('level_num', 0) >= 2) or (res['above_50sma_pct'] < 45.0) or (res['dd_count'] >= 4)

    if res['ftd_active'] and (res['dd_count'] >= 5 or leaders_failing or (post_ftd_squat_fail and weak_breadth_or_stealth)):
        res['ftd_suspect_warning'] = True
        reasons = []
        if res['dd_count'] >= 5:
            reasons.append(f"DD={res['dd_count']}")
        if leaders_failing:
            reasons.append("Leaders <21E")
        if post_ftd_squat_fail and weak_breadth_or_stealth:
            reasons.append("High Squat")
        res['ftd_status_text'] += f" ⚠️ SUSPECT ({' / '.join(reasons)})"

    # Velocity, Supply Exhaustion (scaled to core universe size ~8%), and Stage 4 Structural Confirmation Floor
    core_n_st = 750
    if breadth_df is not None and not breadth_df.empty and 'total_stocks' in breadth_df.columns and pd.notna(breadth_df['total_stocks'].iloc[-1]):
        core_n_st = max(100, int(breadth_df['total_stocks'].iloc[-1] or 750))
    elif close_core is not None and not close_core.empty:
        core_n_st = max(100, int(close_core.iloc[-1].notna().sum()))
    nl_exhausted_cap = max(40, int(round(core_n_st * 0.08)))

    is_thrust_velocity = vel.get('is_thrust', False) or res['is_9to1_thrust'] or (res['above_50sma_pct'] >= 30.0)
    supply_exhausted = (res['new_lows_count'] <= nl_exhausted_cap)
    stage4_structural_ok = (
        idx_above_21
        and (res['above_50sma_pct'] >= 35.0)
        and (idx_above_50 or res['above_50sma_pct'] >= 42.0 or res['is_9to1_thrust'])
        and (res['dd_count'] <= 4)
        and (not res['ftd_suspect_warning'])
    )

    # Evaluate Strictly 5-Stage State Machine
    if (
        power_trend_aligned
        and res['above_50sma_pct'] >= 55.0
        and res['dd_count'] <= 3
        and not res['ftd_suspect_warning']
        and (persona_score >= 52.0 or res['is_9to1_thrust'])
    ):
        res.update({
            'stage_num': 5,
            'stage_code': 'POWER_TREND',
            'stage_title': 'Stage 5: Super-Thrust / Power Trend (10 EMA > 21 EMA > 50 SMA)',
            'stage_badge': '🔥 STAGE 5: POWER TREND (FULL PEDAL)',
            'stage_color': '#10b981',
            'target_exposure': '85% – 100%+ Equity (Pyramiding Enabled)',
            'directive': (
                f"Full institutional tailwind ({res['above_50sma_pct']:.0f}% > 50 SMA, {res['dd_count']} DDs). "
                "Hold 🦖 50-SMA Monsters for big swings and pyramid Flowers using Hougaard house-money cushion."
            ),
        })
    elif stage4_structural_ok and (res['ftd_active'] or (idx_above_50 and res['above_50sma_pct'] >= 45.0)):
        bh_note = (
            f" | BH Note: Pre-FTD T+10 Lag ({breakout_health:.0f}%) — Verify New Pilots Show +4% Cushion Before Scaling Full"
            if res['bh_is_pre_ftd_lag']
            else f"Breakout Health {breakout_health:.0f}%"
        )
        res.update({
            'stage_num': 4,
            'stage_code': 'VALIDATED_FTD',
            'stage_title': 'Stage 4: Validated Follow-Through / Broadening Uptrend',
            'stage_badge': '🚀 STAGE 4: VALIDATED FTD (PROGRESSIVE BULL)',
            'stage_color': '#34d399',
            'target_exposure': '50% – 75% Equity (0.50%–0.75% Risk)',
            'directive': (
                f"Broadening uptrend validated ({res['above_50sma_pct']:.0f}% > 50 SMA, {res['dd_count']} DDs"
                f"{bh_note if res['bh_is_pre_ftd_lag'] else ', ' + bh_note}). "
                "Scale progressively into Top 5 Omni-Confluence Leaders (≤ 12% pos cap) as pilots prove profitable."
            ),
        })
    elif (
        (idx_above_21 or res['recapture_step'] >= 5 or res['is_9to1_thrust'] or res['ftd_active'])
        and (is_thrust_velocity or res['ftd_active'])
        and supply_exhausted
    ):
        if res['ftd_suspect_warning']:
            warn_prefix = "⚠️ BULL-TRAP FTD GUARDRAIL: "
        elif res['ftd_active'] and not stage4_structural_ok:
            warn_prefix = "🛡️ EARLY BEAR-TURN FTD CAP (Awaiting Breadth / 21-EMA Confirm): "
        else:
            warn_prefix = ""
        res.update({
            'stage_num': 3,
            'stage_code': 'TACTICAL_PILOT',
            'stage_title': 'Stage 3: Tactical Tide-Turn (Velocity Thrust / 9:1 Pilot Trigger)',
            'stage_badge': '⚡ STAGE 3: TACTICAL PILOT DECK',
            'stage_color': '#f59e0b',
            'target_exposure': '15% – 30% Equity (0.35% Risk Pilots Only)',
            'directive': (
                f"{warn_prefix}Tactical turn active ({vel.get('velocity_badge', '')}, Up-Vol {res['up_vol_ratio_pct']:.0f}%, {res['ironclad_count']} Ironclad Beach Balls). "
                "Test the waters with 2–3 Pilot positions (≤ 8% position cap, 0.35% NAV risk) on 21-EMA Low / Oops / PEB triggers."
            ),
        })
    elif res['ironclad_count'] >= 10 and res['fomo_5ema_pct'] > 18.0:
        drift_note = ""
        if res['above_50sma_pct'] < 25.0 and not is_thrust_velocity and vel.get('roc_5d_50sma', 0) > 0:
            drift_note = f" ⚠️ LAZY DRIFT GUARDRAIL: Breadth curling (+{vel.get('roc_5d_50sma', 0):+.1f}% in 5D) but lacks institutional thrust velocity. Pilot sizing gated until velocity >= +5%."
        res.update({
            'stage_num': 2,
            'stage_code': 'BEACH_BALL_DIVERGENCE',
            'stage_title': 'Stage 2: Submerged Beach Ball Divergence (Smart Money Accumulating)',
            'stage_badge': '🏖️ STAGE 2: BEACH BALL DIVERGENCE',
            'stage_color': '#38bdf8',
            'target_exposure': '0% – 10% Equity (Stalk List + 1 Test Pilot Max)',
            'directive': (
                f"Index remains below trend ({res['above_50sma_pct']:.0f}% > 50 SMA), BUT {res['ironclad_count']} Ironclad Leaders "
                f"are defying the down-days under water!{drift_note} Do NOT buy extended breakouts; limit to 1–2 test pilots (≤ 5% cap, 0.25% risk)."
            ),
        })
    else:
        res.update({
            'stage_num': 1,
            'stage_code': 'CAPITULATION_WASHOUT',
            'stage_title': 'Stage 1: Capitulation Washout / Heavy Correction',
            'stage_badge': '🌊 STAGE 1: CAPITULATION WASHOUT (DEFENSIVE CASH)',
            'stage_color': '#ef4444',
            'target_exposure': '0% New Buys (Protect Cash & Mental Capital)',
            'directive': (
                f"Heavy washout regime ({res['above_50sma_pct']:.0f}% > 50 SMA, 5D Panic: {res['panic_5d_count']} stocks, New Lows: {res['new_lows_count']}). "
                "Stay out of new breakouts; let the Artificial Brain track which Ironclad Beach Balls refuse to break."
            ),
        })

    # Attach deterministic "How Much to Buy" sizing policy parameters
    res.update(STAGE_SIZING_POLICY[int(res['stage_num'])])
    return res


def compute_omni_confluence_funnel(
    close_p: pd.DataFrame,
    open_p: pd.DataFrame,
    high_p: pd.DataFrame,
    low_p: pd.DataFrame,
    vol_p: pd.DataFrame,
    benchmark_df: pd.DataFrame = None,
    portfolio_tickers: set = None,
    industry_map: dict = None,
    fundamentals_map: dict = None,
    tide_stage_num: int = 3,
    persona_score: float = 50.0,
    top_n: int = 25,
    market: str = 'INDIA',
) -> tuple:
    """
    Vectorized 7-Pillar Cross-Platform Confluence Scanner across all 2,750+ Indian stocks
    plus 100% of active Portfolio SME/BSE holdings.
    Returns (top_confluence_list, total_ironclad_count, universe_count).
    """
    if close_p is None or close_p.empty or len(close_p) < 30:
        return [], 0, 0

    prof = get_market_brain_profile(market)
    is_us_mkt = bool(prof['is_us'])
    curr_sym = prof['curr_sym']
    p_dec = int(prof['price_decimals'])

    portfolio_clean_set = set()
    if (not is_us_mkt) and portfolio_tickers:
        for pt in portfolio_tickers:
            c_pt = str(pt).replace('.NS', '').replace('.BO', '').replace('-SM', '').replace('-ST', '').strip().upper()
            portfolio_clean_set.add(c_pt)
            yf_pt = get_ticker_symbol(str(pt)).replace('.NS', '').replace('.BO', '').strip().upper()
            portfolio_clean_set.add(yf_pt)

    # Filter & deduplicate market columns strictly by profile (India .NS/.BO or US active NASDAQ/NYSE/AMEX)
    ind_cols = _filter_market_confluence_columns(close_p.columns, market=market, portfolio_clean_set=portfolio_clean_set)
    if not ind_cols:
        return [], 0, 0

    close_p, open_p, high_p, low_p, vol_p, _ = _trim_panels_to_valid_market_bars(
        close_p, open_p, high_p, low_p, vol_p, market=market
    )
    if close_p is None or close_p.empty or len(close_p) < 30:
        return [], 0, 0

    raw_last_valid = close_p[ind_cols].iloc[-1].notna()
    c_raw = close_p[ind_cols].ffill(limit=5)
    # Require >= 25 valid bars AND active latest-session close (or portfolio override)
    last_valid_mask = (raw_last_valid | c_raw.iloc[-1].notna()) & (close_p[ind_cols].notna().sum() >= 25)
    valid_cols = last_valid_mask[last_valid_mask].index.tolist()
    if not valid_cols:
        return [], 0, 0

    has_real_vol = vol_p is not None and not vol_p.empty
    c_df = c_raw[valid_cols]
    o_df = open_p[valid_cols].ffill(limit=5) if (open_p is not None and not open_p.empty) else c_df
    h_df = high_p[valid_cols].ffill(limit=5) if (high_p is not None and not high_p.empty) else c_df
    l_df = low_p[valid_cols].ffill(limit=5) if (low_p is not None and not low_p.empty) else c_df
    v_df = vol_p[valid_cols].fillna(0.0) if has_real_vol else pd.DataFrame(100000.0, index=c_df.index, columns=valid_cols)

    universe_count = len(valid_cols)
    curr_close = c_df.iloc[-1]
    curr_open = o_df.iloc[-1]
    curr_high = h_df.iloc[-1]
    curr_low = l_df.iloc[-1]
    curr_vol = v_df.iloc[-1]

    prev_close = c_df.iloc[-2]
    prev_open = o_df.iloc[-2]
    prev_high = h_df.iloc[-2]
    prev_low = l_df.iloc[-2]
    prev_vol = v_df.iloc[-2]

    # Moving averages, liquidity & active-trading hygiene
    ema10 = c_df.ewm(span=10, adjust=False).mean().iloc[-1]
    ema21_c = c_df.ewm(span=21, adjust=False).mean().iloc[-1]
    ema21_l = l_df.ewm(span=21, adjust=False).mean().iloc[-1]
    sma50 = c_df.rolling(50, min_periods=25).mean().iloc[-1]
    vol_ma20 = v_df.rolling(20, min_periods=5).mean().iloc[-1].replace(0, 1.0)
    turnover_val = (curr_close * vol_ma20) / float(prof['adtv_scale'])
    active_vol_days = (v_df.tail(20) > 0).sum(axis=0)
    halted_now = (~raw_last_valid.reindex(valid_cols).fillna(False)) | (has_real_vol & (curr_vol <= 0)) | (active_vol_days < min(16, len(v_df)))

    # 52W High & 252D Low (min_periods=30 for SMEs/IPOs)
    high_52w = h_df.rolling(252, min_periods=30).max().iloc[-1]
    low_52w = l_df.rolling(252, min_periods=30).min().iloc[-1]
    dist_52w_pct = ((high_52w - curr_close) / high_52w.replace(0, np.nan)) * 100.0

    # ADR% (20D)
    daily_rng_pct = ((h_df - l_df) / c_df.replace(0, np.nan)) * 100.0
    adr_20 = daily_rng_pct.rolling(20, min_periods=5).mean().iloc[-1].fillna(3.0).clip(lower=0.8)

    # -------------------------------------------------------------------------
    # PILLAR 3: Vectorized O'Neil Universe RS & Smooth RS & RSNH Blue Dot (15 pts)
    # -------------------------------------------------------------------------
    def _ret_n(n_bars):
        if len(c_df) > n_bars:
            return (curr_close / c_df.iloc[-(n_bars + 1)].bfill()) - 1.0
        return pd.Series(0.0, index=valid_cols)

    r_1m = _ret_n(21).fillna(0.0)
    r_3m = _ret_n(63).fillna(r_1m)
    r_6m = _ret_n(126).fillna(r_3m)
    r_12m = _ret_n(min(250, len(c_df) - 1)).fillna(r_6m)

    raw_rs = (0.40 * r_3m) + (0.20 * r_6m) + (0.20 * r_12m) + (0.20 * r_1m)
    rs_pct = (raw_rs.rank(pct=True) * 99.0).round(1)
    smooth_rs = (rs_pct / adr_20).round(1)

    # RSNH Blue Dot (Stock relative line at 252D high)
    rsnh_active = pd.Series(False, index=valid_cols)
    if benchmark_df is not None and not benchmark_df.empty:
        try:
            b_c = benchmark_df.set_index('date')['close'] if 'date' in benchmark_df.columns else benchmark_df['close']
            if b_c.index.tz is not None:
                b_c.index = b_c.index.tz_localize(None)
            b_c.index = b_c.index.normalize()
            b_aligned = b_c.reindex(c_df.index).ffill()
            rs_line = c_df.div(b_aligned, axis=0)
            rs_line_max = rs_line.rolling(252, min_periods=40).max().iloc[-1]
            rsnh_active = (rs_line.iloc[-1] >= rs_line_max * 0.995) & (dist_52w_pct <= 18.0)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # PILLAR 1: Down-Day Resilience & Ironclad Beach Balls (Last 10 Sessions) (20 pts)
    # -------------------------------------------------------------------------
    rets_all = c_df.pct_change(fill_method=None) * 100.0
    mkt_median_ret = rets_all.median(axis=1)
    tail_15 = rets_all.iloc[-15:]
    mkt_tail_15 = mkt_median_ret.iloc[-15:]
    down_dates = mkt_tail_15[mkt_tail_15 <= -0.25].index.tolist()
    if len(down_dates) < 2:
        # Take the 4 weakest median sessions in the last 12 trading days
        down_dates = mkt_tail_15.nsmallest(4).index.tolist()

    dd_rets = rets_all.loc[down_dates]
    mkt_dd_rets = mkt_median_ret.loc[down_dates]
    alpha_matrix = dd_rets.sub(mkt_dd_rets, axis=0)

    # Resilient on a down day if stock closed green (>0%) OR beat market by >= +1.25%
    resilient_hits = (dd_rets > 0.0) | (alpha_matrix >= 1.25)
    dd_hit_rate = (resilient_hits.mean(axis=0) * 100.0).fillna(0.0)
    dd_alpha_spread = alpha_matrix.mean(axis=0).fillna(0.0)

    # Power Plays in last 60 sessions (up >= 3.8% on vol >= 1.0x 20D avg with upper range close)
    tail_60 = min(60, len(c_df))
    r60 = rets_all.iloc[-tail_60:]
    v60 = v_df.iloc[-tail_60:]
    vma60 = v_df.rolling(20, min_periods=5).mean().iloc[-tail_60:].replace(0, 1.0)
    clv60 = ((c_df.iloc[-tail_60:] - l_df.iloc[-tail_60:]) / (h_df.iloc[-tail_60:] - l_df.iloc[-tail_60:]).replace(0, np.nan)).fillna(0.5)
    power_plays_60d = ((r60 >= 3.8) & (v60 >= vma60) & (clv60 >= 0.60)).sum(axis=0)

    is_ironclad = (
        (dd_hit_rate >= 60.0)
        & (dd_alpha_spread >= 1.0)
        & (rs_pct >= 80.0)
        & (curr_close >= sma50 * 0.97)
        & (dist_52w_pct <= 25.0)
    )

    # -------------------------------------------------------------------------
    # PILLAR 2: HV1 Defended & Post-HV1 Float-Lockup Ratio >= 6.0x (20 pts) — Pure NumPy Vectorized
    # -------------------------------------------------------------------------
    tail_252 = min(252, len(c_df))
    v_252_arr = np.nan_to_num(v_df.iloc[-tail_252:].to_numpy(dtype=float), nan=-1.0)
    c_252_arr = np.nan_to_num(c_df.iloc[-tail_252:].to_numpy(dtype=float), nan=0.0)
    o_252_arr = np.nan_to_num(o_df.iloc[-tail_252:].to_numpy(dtype=float), nan=0.0)
    h_252_arr = np.nan_to_num(h_df.iloc[-tail_252:].to_numpy(dtype=float), nan=0.0)
    l_252_arr = np.nan_to_num(l_df.iloc[-tail_252:].to_numpy(dtype=float), nan=0.0)

    n_cols = len(valid_cols)
    col_indices = np.arange(n_cols)
    hv1_row_idx = np.argmax(v_252_arr, axis=0)

    hv1_vol_arr = v_252_arr[hv1_row_idx, col_indices]
    hv1_low_arr = l_252_arr[hv1_row_idx, col_indices]
    hv1_high_arr = h_252_arr[hv1_row_idx, col_indices]
    hv1_close_arr = c_252_arr[hv1_row_idx, col_indices]
    hv1_open_arr = o_252_arr[hv1_row_idx, col_indices]

    hv1_rng_arr = np.where((hv1_high_arr - hv1_low_arr) > 0, hv1_high_arr - hv1_low_arr, 1.0)
    hv1_clv_arr = (hv1_close_arr - hv1_low_arr) / hv1_rng_arr
    hv1_const_arr = (hv1_close_arr >= hv1_open_arr) | (hv1_clv_arr >= 0.50)

    hv1_vol = pd.Series(hv1_vol_arr, index=valid_cols)
    hv1_low = pd.Series(hv1_low_arr, index=valid_cols)
    hv1_constructive = pd.Series(hv1_const_arr, index=valid_cols)

    hv1_defended = hv1_constructive & (curr_close >= hv1_low) & (hv1_low > 0)
    min_vol_10d = v_df.iloc[-10:].replace(0, np.nan).min(axis=0)
    hv1_lockup_ratio = (hv1_vol / min_vol_10d).replace([np.inf, -np.inf], np.nan).fillna(1.0).round(1)
    dist_hv1_low_pct = (((curr_close - hv1_low) / hv1_low.replace(0, np.nan)) * 100.0).fillna(99.0)
    is_hv1_lockup = hv1_defended & (hv1_lockup_ratio >= 6.0) & (dist_hv1_low_pct <= 35.0)

    # -------------------------------------------------------------------------
    # PILLAR 4: Live Actionable Trigger Today (Oops, PEB, Inside Day, 21-EMA Low Bounce, Breakout) (15 pts)
    # -------------------------------------------------------------------------
    today_ret = rets_all.iloc[-1].fillna(0.0)
    yest_ret = rets_all.iloc[-2].fillna(0.0)
    today_clv = ((curr_close - curr_low) / (curr_high - curr_low).replace(0, np.nan)).fillna(0.5)
    rel_vol_today = (curr_vol / vol_ma20).fillna(1.0)

    # 1. Oops Reversal
    is_oops = (~halted_now) & (curr_open < prev_low) & (curr_close > prev_low) & ((curr_close >= curr_open) | (today_clv >= 0.45))
    # 2. Positive Expectation Breaker (PEB)
    is_peb = (~halted_now) & (prev_close < prev_open) & (yest_ret <= -1.3) & (curr_close > curr_open) & (curr_close > prev_close)
    # 3. Inside / Double-Inside Day Priming (must actually trade with valid range > 0!)
    is_inside = (~halted_now) & (curr_high > curr_low) & (curr_high <= prev_high) & (curr_low >= prev_low) & (rel_vol_today <= 0.90) & (dist_52w_pct <= 18.0)
    # 4. Jim Roppel 21-EMA Low Channel Bounce
    dist_to_21l = ((curr_low - ema21_l) / ema21_l.replace(0, np.nan)) * 100.0
    is_21ema_low_bounce = (~halted_now) & (dist_to_21l >= -1.2) & (dist_to_21l <= 1.5) & (curr_close >= ema21_l) & (curr_close >= curr_open) & (curr_close >= sma50)
    # 5. High-Volume Pocket Pivot / Breakout Thrust
    is_breakout_thrust = (~halted_now) & (today_ret >= 3.0) & (rel_vol_today >= 1.4) & (today_clv >= 0.65) & (dist_52w_pct <= 12.0)

    # Split artifact filter (US preserves high-volume Episodic Pivots like MRNA/KOD/SLN!)
    rets_raw_65 = rets_all.tail(65) / 100.0
    if is_us_mkt and has_real_vol:
        vma65 = v_df.rolling(20, min_periods=5).mean().tail(65)
        v_raw65 = v_df.tail(65)
        has_split_jump = ((rets_raw_65.abs() > 0.85) & (v_raw65 < 2.0 * vma65)).any(axis=0).fillna(False)
    else:
        has_split_jump = (rets_raw_65.abs() > 0.85).any(axis=0).fillna(False)

    # -------------------------------------------------------------------------
    # PILLAR 6: 100% Club Doubler & Low Distribution Tape (10 pts)
    # -------------------------------------------------------------------------
    is_doubler = (curr_close >= low_52w * 2.0) & (curr_close >= sma50) & (dist_52w_pct <= 25.0)
    # Distribution days in last 15 sessions (down <= -0.5% on higher volume than prior day)
    r15 = rets_all.iloc[-15:]
    v15 = v_df.iloc[-15:]
    v15_prev = v_df.shift(1).iloc[-15:]
    stock_dd_15d = ((r15 <= -0.5) & (v15 > v15_prev)).sum(axis=0)
    clean_tape = stock_dd_15d <= 2

    # -------------------------------------------------------------------------
    # PILLAR 5: Livermore Sister Tandem & Top Industry Rank (10 pts)
    # Strictly partitioned by market suffix so HAL/META/CCL/PTC never collide!
    # -------------------------------------------------------------------------
    ind_lookup = {}
    canon_intel = {}
    mkt_label_ig = "US (S&P 500 / Nasdaq)" if is_us_mkt else "India (NSE/BSE)"
    try:
        from industry_group_engine import get_stock_group_intelligence
        loaded_intel = get_stock_group_intelligence(taxonomy="canonical", universe_scope=mkt_label_ig)
        if isinstance(loaded_intel, dict):
            canon_intel = loaded_intel
            for k, info in canon_intel.items():
                if isinstance(info, dict):
                    grp = info.get('group_name') or info.get('industry')
                    if grp and str(grp) not in ['Unknown', 'None', 'nan', 'Other / ETF', 'General']:
                        k_up = str(k).replace('.NS', '').replace('.BO', '').strip().upper()
                        ind_lookup[k_up] = str(grp)
                        if '.' in k_up:
                            ind_lookup.setdefault(k_up.replace('.', '-'), str(grp))
                        elif '-' in k_up:
                            ind_lookup.setdefault(k_up.replace('-', '.'), str(grp))
    except Exception:
        canon_intel = {}
    if industry_map:
        for k, val in industry_map.items():
            k_str = str(k).strip()
            k_is_in = k_str.endswith(('.NS', '.BO'))
            if is_us_mkt and k_is_in:
                continue
            if (not is_us_mkt) and k_str and ('.' not in k_str) and not k_is_in:
                # India dict keys may already be stripped NSE symbols only if caller passed India industry_map
                pass
            clean_k = k_str.replace('.NS', '').replace('.BO', '').upper()
            val_str = val.get('group_name') if isinstance(val, dict) else val
            if clean_k not in ind_lookup and val_str and str(val_str) not in ['Unknown', 'None', 'nan']:
                ind_lookup[clean_k] = str(val_str)
    if fundamentals_map:
        for k, f_dict in fundamentals_map.items():
            k_str = str(k).strip()
            k_is_in = k_str.endswith(('.NS', '.BO'))
            if is_us_mkt and k_is_in:
                continue
            if (not is_us_mkt) and (not k_is_in):
                continue
            clean_k = k_str.replace('.NS', '').replace('.BO', '').upper()
            if clean_k not in ind_lookup and isinstance(f_dict, dict):
                ind_val = f_dict.get('industry')
                if ind_val and str(ind_val) not in ['Unknown', 'None', 'nan']:
                    ind_lookup[clean_k] = str(ind_val)

    us_tv_map = {}
    format_tradingview_symbol = None
    if is_us_mkt:
        try:
            from industry_group_engine import load_exporter_enrichment_maps, format_tradingview_symbol
            us_tv_map, _ = load_exporter_enrichment_maps(is_us=True)
        except Exception:
            format_tradingview_symbol = None
            us_tv_map = {}

    # Group stocks by canonical MarketSurge 2.0 industry (>= 4 peers) to find Top RS industries & Sister Tandems
    from industry_group_engine import (
        EXCLUDED_CANONICAL_GROUPS,
        MIN_RANKED_GROUP_SIZE_CANONICAL,
        normalize_taxonomy,
        load_cached_industry_group_matrix,
    )

    ind_members = {}
    for col in valid_cols:
        clean_t = str(col).replace('.NS', '').replace('.BO', '').strip().upper()
        raw_ind = ind_lookup.get(clean_t, 'General')
        ind_name = normalize_taxonomy(raw_ind)
        if ind_name not in EXCLUDED_CANONICAL_GROUPS:
            ind_members.setdefault(ind_name, []).append((col, float(rs_pct[col]), float(r_1m[col])))

    cached_mat = load_cached_industry_group_matrix(mkt_label_ig)
    cached_top_groups = set()
    if cached_mat is not None and not cached_mat.empty and 'Industry_Group' in cached_mat.columns:
        top_cut = max(10, int(round(len(cached_mat) * 0.20)))
        cached_top_groups = set(cached_mat.nsmallest(top_cut, 'Rank_Today')['Industry_Group'].astype(str).tolist())

    ind_scores = {}
    sister_confirmed_tickers = set()
    sister_partner_map = {}
    for ind_name, members in ind_members.items():
        if len(members) >= MIN_RANKED_GROUP_SIZE_CANONICAL:
            members_sorted = sorted(members, key=lambda x: x[1], reverse=True)
            rs_vals = [m[1] for m in members_sorted]
            avg_rs = float(np.mean(rs_vals))
            breadth_80 = (sum(1 for r in rs_vals if r >= 80.0) / float(len(rs_vals))) * 100.0
            ind_scores[ind_name] = (0.75 * avg_rs) + (0.25 * breadth_80)
            # Check Livermore Sister Tandem: #1 and #2 both have RS >= 80 and positive 1M return
            m1, m2 = members_sorted[0], members_sorted[1]
            if m1[1] >= 80.0 and m2[1] >= 78.0 and m1[2] > 0 and m2[2] > 0:
                c1 = str(m1[0]).replace('.NS', '').replace('.BO', '')
                c2 = str(m2[0]).replace('.NS', '').replace('.BO', '')
                for m_col, m_rs, m_r1m in members_sorted[:3]:
                    if m_rs >= 78.0:
                        sister_confirmed_tickers.add(m_col)
                        mc = str(m_col).replace('.NS', '').replace('.BO', '')
                        sister_partner_map[m_col] = c2 if mc == c1 else c1

    if canon_intel:
        for col_s in valid_cols:
            raw_t = str(col_s)
            clean_t = raw_t.replace('.NS', '').replace('.BO', '').strip().upper()
            info_c = (
                canon_intel.get(raw_t)
                or canon_intel.get(clean_t)
                or canon_intel.get(clean_t.replace('.', '-'))
                or canon_intel.get(clean_t.replace('-', '.'))
                or {}
            )
            if not isinstance(info_c, dict) or not info_c:
                continue
            rc_c = str(info_c.get("role_code", "")).upper()
            gg_c = str(info_c.get("group_grade", "NR")).upper()
            is_conf_c = bool(info_c.get("is_group_confirmed", False)) or (gg_c in ("A+", "A", "B"))
            is_dng_c = bool(info_c.get("is_danger_bottom_20", False)) or (gg_c in ("D", "E"))
            a_ldr = str(info_c.get("alpha_leader", "None"))
            s_ldr = str(info_c.get("sister_stock", "None"))
            if is_conf_c and (not is_dng_c) and rc_c in ("ALPHA", "SISTER") and a_ldr != "None" and s_ldr != "None":
                sister_confirmed_tickers.add(col_s)
                if col_s not in sister_partner_map:
                    sister_partner_map[col_s] = s_ldr if rc_c == "ALPHA" else a_ldr

    ranked_local_top = set(
        k for k, _ in sorted(ind_scores.items(), key=lambda item: item[1], reverse=True)[:10]
    )
    top_industries = (ranked_local_top & cached_top_groups) if (cached_top_groups & set(ind_scores.keys())) else ranked_local_top
    if len(top_industries) < 5:
        top_industries = ranked_local_top | (cached_top_groups & set(ind_scores.keys()))

    # -------------------------------------------------------------------------
    # SCORE & ASSEMBLE CANDIDATES (WITH 100% PORTFOLIO SME EXEMPTION)
    # -------------------------------------------------------------------------
    total_ironclad_count = 0
    candidates = []

    min_adtv_gate = float(prof['funnel_min_adtv'])
    min_p_gate = float(prof['min_price'])

    for col in valid_cols:
        clean_t = str(col).replace('.NS', '').replace('.BO', '').strip().upper()
        is_in_portfolio = (not is_us_mkt) and (clean_t in portfolio_clean_set)
        t_val = float(turnover_val[col]) if pd.notna(turnover_val[col]) else 0.0
        c_price = float(curr_close[col])

        if (not is_in_portfolio) and bool(has_split_jump.get(col, False)):
            continue
        if (not is_in_portfolio) and bool(halted_now.get(col, False)):
            continue
        if (not is_in_portfolio) and c_price < min_p_gate:
            continue

        # Liquidity Gate: >= ₹2.5 Cr (India) or >= $10M (US), exempt for active India portfolio holdings
        if not is_in_portfolio and t_val < min_adtv_gate:
            continue

        if bool(is_ironclad[col]) and (t_val >= min_adtv_gate or is_in_portfolio):
            total_ironclad_count += 1

        rs_val = float(rs_pct[col]) if pd.notna(rs_pct[col]) else 50.0
        if rs_val < 70.0 and not is_in_portfolio:
            continue

        s50_val = float(sma50[col]) if pd.notna(sma50[col]) else c_price
        e21l_val = float(ema21_l[col]) if pd.notna(ema21_l[col]) else c_price
        e10_val = float(ema10[col]) if pd.notna(ema10[col]) else c_price

        if c_price < s50_val * 0.95 and not is_in_portfolio:
            continue

        score = 0.0
        badges = []

        # Pillar 1: Down-Day Resilience (20 pts)
        hit_r = float(dd_hit_rate[col])
        alp_s = float(dd_alpha_spread[col])
        pp_cnt = int(power_plays_60d[col])
        if bool(is_ironclad[col]):
            score += 20.0
            badges.append(f"🛡️ Ironclad ({hit_r:.0f}% DD Hit)")
        elif hit_r >= 50.0 and alp_s >= 0.5:
            score += 12.0
            badges.append(f"🏖️ Resilient ({hit_r:.0f}% DD)")
        elif pp_cnt >= 2:
            score += 8.0
            badges.append(f"⚡ {pp_cnt} Power Plays")

        # Pillar 2: HV1 & Float-Lockup (20 pts)
        lock_r = float(hv1_lockup_ratio[col])
        if bool(is_hv1_lockup[col]):
            score += 20.0
            badges.append(f"🔒 HV1 Lockup ({lock_r:.1f}x)")
        elif bool(hv1_defended[col]):
            score += 12.0
            badges.append("📦 HV1 Defended")

        # Pillar 3: RS & Smooth RS & RSNH Blue Dot (15 pts)
        sm_rs = float(smooth_rs[col])
        if rs_val >= 90.0:
            score += 8.0
        elif rs_val >= 82.0:
            score += 5.0
        if sm_rs >= 18.0:
            score += 4.0
        if bool(rsnh_active[col]):
            score += 3.0
            badges.append("🔵 RSNH Blue Dot")

        # Pillar 4: Today's Live Trigger (15 pts)
        trigger_name = "Stalk 21-EMA Low"
        trigger_price = round(max(c_price, float(curr_high[col])), 2)
        stop_price = round(min(float(curr_low[col]), e21l_val), 2)

        if bool(is_oops[col]):
            score += 15.0
            trigger_name = "🎯 Oops Reversal"
            trigger_price = round(c_price, 2)
            stop_price = round(float(curr_low[col]), 2)
            badges.append("🎯 Oops")
        elif bool(is_peb[col]):
            score += 15.0
            trigger_name = "⚡ PEB Reversal"
            trigger_price = round(c_price, 2)
            stop_price = round(min(float(curr_low[col]), float(prev_low[col])), 2)
            badges.append("⚡ PEB")
        elif bool(is_inside[col]):
            score += 14.0
            trigger_name = "📐 Inside Day Coil"
            trigger_price = round(float(curr_high[col]), 2)
            stop_price = round(float(curr_low[col]), 2)
            badges.append("📐 Inside Day")
        elif bool(is_21ema_low_bounce[col]):
            score += 14.0
            trigger_name = "🏄 21-EMA Low Bounce"
            trigger_price = round(c_price, 2)
            stop_price = round(min(float(curr_low[col]), e21l_val * 0.99), 2)
            badges.append("🏄 21-EMA Low Bounce")
        elif bool(is_breakout_thrust[col]):
            score += 13.0 if persona_score >= 52.0 else 9.0
            trigger_name = "🚀 Volume Thrust"
            trigger_price = round(c_price, 2)
            stop_price = round(max(float(curr_low[col]), e21l_val), 2)
            badges.append("🚀 Vol Thrust")
        else:
            # Check proximity to 21-EMA Low or 10 EMA for stalk entry
            ext_21l = abs(c_price - e21l_val) / e21l_val * 100.0 if e21l_val > 0 else 99.0
            if ext_21l <= 3.5 and c_price >= e21l_val:
                score += 7.0
                fmt_21l = format(e21l_val, f".{p_dec}f")
                trigger_name = f"👀 Tight near 21-EMA Low ({curr_sym}{fmt_21l})"

        # Pillar 5: Livermore Sister Tandem & Top Industry (10 pts)
        ind_name = ind_lookup.get(clean_t, 'General')
        if col in sister_confirmed_tickers:
            score += 10.0
            partner = sister_partner_map.get(col, '')
            badges.append(f"👯 Sister ({partner})" if partner else "👯 Sister Tandem")
        elif ind_name in top_industries:
            score += 6.0
            badges.append("🏆 Top Group")

        # Pillar 6: 100% Doubler & Clean Tape (10 pts)
        if bool(is_doubler[col]):
            score += 6.0
            badges.append("🚀 100% Club")
        if bool(clean_tape[col]):
            score += 4.0

        # Pillar 7: Fundamentals / Earnings Growth (10 pts, strictly market-keyed)
        if fundamentals_map:
            if is_us_mkt:
                f_info = fundamentals_map.get(str(col).upper()) or fundamentals_map.get(clean_t) or {}
            else:
                f_info = fundamentals_map.get(str(col)) or fundamentals_map.get(f"{clean_t}.NS") or fundamentals_map.get(f"{clean_t}.BO") or {}
        else:
            f_info = {}
        sales_g = float(f_info.get('sales_growth', 0.0) or 0.0)
        eps_g = float(f_info.get('eps_growth', 0.0) or 0.0)
        if sales_g >= 25.0 or eps_g >= 25.0:
            score += 10.0
            badges.append(f"💰 Sales/EPS +{max(sales_g, eps_g):.0f}%")
        elif sales_g >= 15.0 or eps_g >= 15.0:
            score += 5.0

        if score < 38.0 and not is_in_portfolio:
            continue

        ext_above_21l_pct = ((c_price - e21l_val) / e21l_val * 100.0) if e21l_val > 0 else 0.0
        raw_risk_pct = ((trigger_price - stop_price) / trigger_price) * 100.0 if trigger_price > stop_price else 3.5

        # O'Neil / Minervini 7.5% Max Stop Risk & Extension Rule:
        # Never chase a breakout where structural stop risk > 7.5% or stock is extended > 7.5% above 21-EMA Low without a tight pivot
        is_extended_setup = False
        if raw_risk_pct > 7.5 or (ext_above_21l_pct > 7.5 and not bool(is_inside[col]) and raw_risk_pct > 6.5):
            is_extended_setup = True
            if ext_above_21l_pct > 12.0:
                score -= 12.0
            elif ext_above_21l_pct > 7.5:
                score -= 8.0
            else:
                score -= 5.0
            # Calculate low-risk Pullback Buy Zone (+3.0% above 21-EMA Low, with stop 1.0% below 21-EMA Low = ~3.9% risk)
            pb_entry = round(e21l_val * 1.03, p_dec) if e21l_val > 0 else round(c_price * 0.94, p_dec)
            pb_stop = round(e21l_val * 0.99, p_dec) if e21l_val > 0 else round(pb_entry * 0.96, p_dec)
            pb_risk_pct = round(((pb_entry - pb_stop) / pb_entry) * 100.0, 1) if pb_entry > pb_stop else 3.9
            trigger_name = f"⏳ Pullback Zone (+{ext_above_21l_pct:.0f}% > 21L)"
            trigger_price = pb_entry
            stop_price = pb_stop
            risk_pct = pb_risk_pct
            badges.append(f"⏳ Ext +{ext_above_21l_pct:.0f}% > 21L")
        else:
            risk_pct = max(1.0, min(raw_risk_pct, 7.5))
            if 0.0 <= ext_above_21l_pct <= 4.5:
                score += 3.0  # Bonus for tight low-risk coil right on the 21-EMA Low floor

        if score < 38.0 and not is_in_portfolio:
            continue

        swing_target_2_5r = round(trigger_price * (1.0 + (risk_pct * 2.5 / 100.0)), 2)
        if is_us_mkt:
            if format_tradingview_symbol and us_tv_map:
                tv_sym = format_tradingview_symbol(ticker=clean_t, raw_ticker=str(col), is_us=True, us_tv_map=us_tv_map, prefix_us=True)
                exch_name = tv_sym.split(':', 1)[0] if ':' in tv_sym else 'US'
            else:
                mapped_tv = us_tv_map.get(str(col).upper()) or us_tv_map.get(clean_t)
                if mapped_tv and ':' in str(mapped_tv):
                    tv_sym = str(mapped_tv).upper()
                    exch_name = tv_sym.split(':', 1)[0]
                else:
                    tv_sym = clean_t.replace('-', '.')
                    exch_name = 'US'
        else:
            base_tv = clean_t.replace('-SM', '').replace('-ST', '').replace('-BE', '')
            tv_sym = f"BSE:{base_tv}" if str(col).endswith('.BO') else f"NSE:{base_tv}"
            exch_name = 'BSE' if str(col).endswith('.BO') else 'NSE'

        candidates.append({
            'ticker': col,
            'clean_ticker': clean_t,
            'tv_symbol': tv_sym,
            'exchange': exch_name,
            'is_holding': is_in_portfolio,
            'brain_score': round(min(100.0, score), 1),
            'rs_score': rs_val,
            'smooth_rs': sm_rs,
            'close': round(c_price, 2),
            'return_1d': round(float(today_ret[col]), 2),
            'ema_21_low': round(e21l_val, 2),
            'sma_50': round(s50_val, 2),
            'dd_hit_rate': round(hit_r, 0),
            'dd_alpha_spread': round(alp_s, 2),
            'hv1_lockup_ratio': lock_r if bool(hv1_defended[col]) else 0.0,
            'trigger_name': trigger_name,
            'trigger_price': round(trigger_price, p_dec),
            'stop_price': round(stop_price, p_dec),
            'risk_pct': round(risk_pct, 2),
            'is_extended_setup': is_extended_setup,
            'ext_above_21l_pct': round(ext_above_21l_pct, 1),
            'target_2_5r': swing_target_2_5r,
            'industry': ind_name,
            'turnover_cr': round(t_val, 1),
            'adtv_unit': prof['adtv_unit'],
            'badges_str': " • ".join(badges[:5]),
            'execution_plan': '',
        })

    candidates.sort(key=lambda x: (x['brain_score'], x['rs_score'], x['dd_hit_rate']), reverse=True)
    top_candidates = candidates[:top_n]
    apply_stage_sizing_to_confluence_list(top_candidates, tide_stage_num=tide_stage_num, market=market)
    return top_candidates, total_ironclad_count, universe_count


def compute_hougaard_swap_directives(
    decisions: list,
    portfolio_df: pd.DataFrame,
    market_data: dict,
    total_portfolio_value: float
) -> dict:
    """
    Tom Hougaard ("Best Loser Wins") Zero-New-Heat Portfolio Reallocator.
    Identifies:
      1. Flowers (Add-Ready or High-RS Winners with Locked-In Profit Cushion > Avg Cost)
      2. Weeds (Dead Money, Trim/Exit, or Stagnant Lagging Holdings < 8% gain & RS < 72)
    Computes exact share counts to add to Flowers financed 100% by House-Money Profit Cushion
    and funded by trimming Weeds so Net New Portfolio Heat = 0.00% NAV.
    """
    flowers = []
    weeds = []

    if not decisions or portfolio_df is None or portfolio_df.empty:
        return {'flowers': [], 'weeds': [], 'swap_pairs': [], 'total_locked_cushion_inr': 0.0, 'total_weed_capital_inr': 0.0}

    bse_display_map = {}
    try:
        if os.path.exists('bse_mapping.json'):
            with open('bse_mapping.json', 'r') as f:
                bm = json.load(f)
                for k, v in bm.items():
                    bse_display_map[str(v)] = k
        if os.path.exists('bse_yf_mapping.json'):
            with open('bse_yf_mapping.json', 'r') as f:
                bym = json.load(f)
                for k, v in bym.items():
                    bse_display_map[str(k)] = str(v).replace('.BO', '').replace('.NS', '')
    except Exception:
        pass

    p_map = {}
    for _, row in portfolio_df.iterrows():
        tk = str(row.get('ticker') or row.get('NSE TICKER') or '').strip()
        raw_qty = row.get('quantity') if row.get('quantity') is not None else row.get('Qty', 0)
        raw_buy = row.get('avg_buy_price') if row.get('avg_buy_price') is not None else row.get('Avg Buy Price', 0)
        qty_val = float(str(raw_qty or 0).replace(',', '').strip() or 0)
        buy_val = float(str(raw_buy or 0).replace(',', '').replace('₹', '').strip() or 0)
        p_map[tk] = {'qty': qty_val, 'avg_buy': buy_val}
        nse_tk = str(row.get('NSE TICKER') or '').strip()
        if nse_tk:
            p_map[nse_tk] = {'qty': qty_val, 'avg_buy': buy_val}

    for d in decisions:
        tk = d.get('ticker', '')
        if tk.upper() in {'LIQUIDBEES', 'LIQUIDCASE', 'LIQUIDETF', 'GOLDBEES', 'SILVERBEES', 'BENCHMARK'}:
            continue
        loss_m = d.get('loss_metrics', {}) or {}
        pnl_pct = float(loss_m.get('gain_loss_pct', 0.0) or 0.0)
        pos_val = float(loss_m.get('position_value', 0.0) or 0.0)
        weight_pct = float(loss_m.get('portfolio_weight', 0.0) or 0.0)
        rs_val = float(d.get('rs_score') or 0.0)
        action = d.get('action', 'HOLD')
        layer = d.get('layer', '')
        reason = d.get('reason', '')
        ma_dna = d.get('ma_dna', {}) or {}

        p_info = p_map.get(tk, {'qty': 0.0, 'avg_buy': 0.0})
        qty = p_info['qty']
        avg_buy = p_info['avg_buy']
        df_t = (market_data or {}).get(tk)
        curr_p = float(df_t['close'].iloc[-1]) if (df_t is not None and not df_t.empty and 'close' in df_t.columns) else avg_buy

        locked_inr = float(d.get('locked_profit_inr', 0.0) or 0.0)
        free_shares = int(d.get('free_pyramid_shares', 0) or 0)
        spine_name = ma_dna.get('active_spine_name', '21 EMA Low')
        spine_price = float(ma_dna.get('active_spine_price', 0.0) or ma_dna.get('ema_21_low', 0.0) or 0.0)
        raw_stop = float(d.get('new_share_stop', 0.0) or spine_price or (curr_p * 0.94))
        new_stop = spine_price if (spine_price > 0 and spine_price < curr_p) else (raw_stop if (0 < raw_stop < curr_p) else round(curr_p * 0.94, 2))
        display_tk = bse_display_map.get(str(tk), str(tk))

        # Check if Flower: Add-on ready OR strong winner (P/L >= +10%, RS >= 82, holding spine)
        if (d.get('add_on_ready') or (pnl_pct >= 10.0 and rs_val >= 82.0 and action in ['ADD', 'HOLD'])) and curr_p > new_stop and qty > 0:
            # If locked_inr is 0 because avg_buy is slightly above spine, compute unrealized cushion available at breakeven stop
            unrealized_inr = max(0.0, (curr_p - avg_buy) * qty)
            effective_cushion_inr = locked_inr if locked_inr > 0 else (unrealized_inr * 0.35 if pnl_pct >= 12.0 else 0.0)
            risk_per_sh = max(curr_p * 0.03, curr_p - new_stop)
            sh_taper_cap = max(1, int(qty * 0.50))
            raw_financed = free_shares if free_shares > 0 else int((effective_cushion_inr * 0.35) // risk_per_sh)
            financed_sh = min(raw_financed, sh_taper_cap)
            add_capital_req = round(financed_sh * curr_p, 0)

            if financed_sh >= 1:
                flowers.append({
                    'ticker': display_tk,
                    'raw_ticker': tk,
                    'current_price': round(curr_p, 2),
                    'avg_buy_price': round(avg_buy, 2),
                    'qty': int(qty),
                    'pnl_pct': round(pnl_pct, 1),
                    'rs_score': round(rs_val, 1),
                    'dna_badge': ma_dna.get('dna_badge', '⚡ 21-EMA Low Rider'),
                    'spine_name': spine_name,
                    'locked_cushion_inr': round(effective_cushion_inr, 0),
                    'is_strict_house_money': locked_inr > 0,
                    'financed_add_shares': financed_sh,
                    'add_capital_inr': add_capital_req,
                    'new_stop_price': round(new_stop, 2),
                    'setup_reason': reason,
                    'weight_pct': round(weight_pct, 1),
                })

        # Check if Weed: Dead Money, Trim, Exit, or Laggard (P/L < 5% and RS < 70)
        is_weed = (
            layer == 'L4-DeadMoney'
            or action in ['EXIT', 'TRIM']
            or (pnl_pct < 4.0 and 0 < rs_val < 68.0)
        )
        if is_weed and qty > 0 and pos_val > 0:
            trim_fraction = 1.0 if (action == 'EXIT' or layer == 'L4-DeadMoney') else 0.5
            sell_shares = max(1, int(round(qty * trim_fraction)))
            freed_cash_inr = round(sell_shares * curr_p, 0)
            weeds.append({
                'ticker': display_tk,
                'raw_ticker': tk,
                'current_price': round(curr_p, 2),
                'qty': int(qty),
                'sell_shares': sell_shares,
                'freed_cash_inr': freed_cash_inr,
                'pnl_pct': round(pnl_pct, 1),
                'rs_score': round(rs_val, 1),
                'action': action,
                'weed_reason': reason if reason else f"Lagging RS ({rs_val:.0f}) with {pnl_pct:+.1f}% return",
            })

    flowers.sort(key=lambda x: (x['is_strict_house_money'], x['rs_score'], x['pnl_pct']), reverse=True)
    weeds.sort(key=lambda x: (x['action'] == 'EXIT', -x['rs_score'], x['pnl_pct']), reverse=True)

    # Pair Top Flowers with Top Weeds for 1-Click Zero-New-Heat & Self-Funded Swap Directives
    swap_pairs = []
    for idx, fl in enumerate(flowers[:5]):
        matched_weed = weeds[idx] if idx < len(weeds) else (weeds[0] if weeds else None)
        if matched_weed and fl['current_price'] > 0:
            weed_funded_sh = max(1, int(matched_weed['freed_cash_inr'] // fl['current_price']))
            swap_sh = min(fl['financed_add_shares'], weed_funded_sh)
        else:
            swap_sh = fl['financed_add_shares']
        swap_cap_inr = round(swap_sh * fl['current_price'], 0)
        swap_pairs.append({
            'flower': fl,
            'weed': matched_weed,
            'swap_add_shares': swap_sh,
            'swap_add_capital_inr': swap_cap_inr,
            'max_cushion_shares': fl['financed_add_shares'],
            'max_cushion_capital_inr': fl['add_capital_inr'],
        })

    return {
        'flowers': flowers,
        'weeds': weeds,
        'swap_pairs': swap_pairs,
        'total_locked_cushion_inr': sum(f['locked_cushion_inr'] for f in flowers),
        'total_weed_capital_inr': sum(w['freed_cash_inr'] for w in weeds),
    }


def compute_pig_at_the_trough_directives(
    confluence_candidates: list,
    decisions: list,
    portfolio_df: pd.DataFrame,
    market_data: dict,
    total_portfolio_value: float,
    min_leader_score: float = 70.0
) -> dict:
    """
    Delta 3: The 'Pig-at-the-Trough' Fresh Buy Capital Allocator.
    Derived from Steve Watson (Friess Associates / Brandywine Fund), Michael Lauer, and Tom Hougaard.

    Identifies which existing holding is the weakest 'pig at the trough' (consuming capital with
    low velocity or negative efficiency) and creates an automated 1-click swap directive to sacrifice
    it in order to fund a brand new top-tier (Score >= 70) 7-Pillar Confluence Leader.
    """
    if not decisions or portfolio_df is None or portfolio_df.empty:
        return {'pigs': [], 'fresh_leaders': [], 'swap_directives': [], 'total_stagnant_capital_inr': 0.0, 'dead_money_count': 0}

    portfolio_tickers = set(
        str(row.get('ticker') or row.get('NSE TICKER') or '').strip().replace('.NS', '').replace('.BO', '')
        for _, row in portfolio_df.iterrows()
    )

    # 1. Filter Fresh Confluence Leaders (not already held in portfolio)
    fresh_leaders = []
    for cand in (confluence_candidates or []):
        tk = str(cand.get('clean_ticker') or cand.get('ticker', '')).replace('.NS', '').replace('.BO', '').strip()
        score = float(cand.get('brain_score') or cand.get('score', 0.0) or 0.0)
        curr_p = float(cand.get('close') or cand.get('current_price', 0.0) or cand.get('trigger_price', 0.0) or 0.0)
        is_held = bool(cand.get('is_holding', False)) or (tk in portfolio_tickers)
        if tk and (not is_held) and score >= min_leader_score and curr_p > 0:
            badges = cand.get('badges', [])
            if not badges and cand.get('badges_str'):
                badges = [cand.get('badges_str')]
            fresh_leaders.append({
                'ticker': tk,
                'score': round(score, 1),
                'current_price': round(curr_p, 2),
                'trigger_price': round(float(cand.get('trigger_price', curr_p) or curr_p), 2),
                'stop_price': round(float(cand.get('stop_price', curr_p * 0.94) or (curr_p * 0.94)), 2),
                'trigger_name': cand.get('trigger_name', 'Breakout'),
                'badges': badges,
                'industry': cand.get('industry', ''),
                'rs_score': round(float(cand.get('rs_score', 80.0) or 80.0), 1),
            })

    fresh_leaders.sort(key=lambda x: x['score'], reverse=True)

    # 2. Evaluate All Active Holdings for Opportunity Efficiency
    pigs = []
    tot_stagnant_cap = 0.0
    dead_money_cnt = 0

    for d in decisions:
        tk = str(d.get('ticker', '')).replace('.NS', '').replace('.BO', '').strip()
        if not tk or tk.upper() in ('LIQUIDBEES', 'LIQUIDCASE', 'LIQUIDETF', 'GOLDBEES', 'SILVERBEES'):
            continue

        loss_m = d.get('loss_metrics', {}) or {}
        pnl_pct = float(loss_m.get('gain_loss_pct', 0.0) or 0.0)
        curr_p = float(d.get('close_price', d.get('current_price', 100.0)) or 100.0)
        qty = int(d.get('quantity', 0) or 0)
        pos_val = float(loss_m.get('position_value', 0.0) or (curr_p * qty if qty > 0 else 0.0))
        if qty <= 0 and curr_p > 0 and pos_val > 0:
            qty = max(1, int(pos_val // curr_p))

        weight_pct = float(loss_m.get('portfolio_weight', 0.0) or ((pos_val / total_portfolio_value * 100.0) if total_portfolio_value > 0 else 0.0))
        rs_val = float(d.get('rs_score', 50.0) or 50.0)
        action = d.get('action', 'HOLD')
        layer = d.get('layer', '')

        ts = d.get('time_stop', {}) or {}
        d_held = int(ts.get('days_held', 15) or 15)
        atr_prog = float(ts.get('atr_progress', 0.0) or 0.0)
        is_dead = bool(ts.get('is_dead_money', False)) or (layer == 'L4-DeadMoney')

        if qty <= 0 or pos_val <= 0:
            continue

        if is_dead:
            dead_money_cnt += 1
            tot_stagnant_cap += pos_val

        # Immunity check: Large winners holding spine should never be sacrificed for pilots
        is_protected_flower = (pnl_pct >= 10.0 and rs_val >= 80.0 and action != 'EXIT')
        if is_protected_flower:
            continue

        # Opportunity Efficiency Formula:
        # OE = (RS * Cushion) / (Weight * Days Held * Velocity Penalty)
        cushion_factor = max(0.5, 1.0 + (pnl_pct / 10.0)) if pnl_pct > 0 else 0.4
        velocity_penalty = max(0.8, 1.0 + max(0.0, 1.5 - atr_prog))

        oe = (rs_val * cushion_factor) / (max(0.3, weight_pct) * max(5, d_held) * velocity_penalty)
        if is_dead:
            oe *= 0.4
        if pnl_pct < 0:
            oe *= 0.6
        if action in ('EXIT', 'TRIM'):
            oe *= 0.5

        # Eviction sizing: Exit 100% if severe dead money or exit signal; else trim 50%
        full_evict = is_dead or action == 'EXIT' or (pnl_pct < 0 and d_held >= 15)
        sell_shares = qty if full_evict else max(1, qty // 2)
        freed_cap = round(sell_shares * curr_p, 0)

        pigs.append({
            'ticker': tk,
            'current_price': round(curr_p, 2),
            'qty': qty,
            'sell_shares': sell_shares,
            'freed_cash_inr': freed_cap,
            'pnl_pct': round(pnl_pct, 1),
            'rs_score': round(rs_val, 1),
            'days_held': d_held,
            'atr_progress': round(atr_prog, 1),
            'is_dead_money': is_dead,
            'efficiency': round(oe, 2),
            'action': action,
            'reason': ts.get('reason') or d.get('reason') or f"Low efficiency ({oe:.1f}) laggard",
        })

    # Sort pigs from lowest efficiency to highest (weakest pigs first)
    pigs.sort(key=lambda x: (not x['is_dead_money'], x['efficiency'], x['rs_score'], x['pnl_pct']))

    # 3. Pair Fresh Leaders with Weakest Pigs
    swap_directives = []
    for idx, leader in enumerate(fresh_leaders[:5]):
        matched_pig = pigs[idx] if idx < len(pigs) else (pigs[0] if pigs else None)
        if matched_pig and leader['current_price'] > 0:
            buy_shares = max(1, int(matched_pig['freed_cash_inr'] // leader['current_price']))
            swap_cap = round(buy_shares * leader['current_price'], 0)
        else:
            default_pilot_cap = min(250000.0, total_portfolio_value * 0.05) if total_portfolio_value > 0 else 200000.0
            buy_shares = max(1, int(default_pilot_cap // leader['current_price'])) if leader['current_price'] > 0 else 0
            swap_cap = round(buy_shares * leader['current_price'], 0)

        directive_text = (
            f"To initiate ₹{swap_cap:,.0f} of new 7-Pillar Leader {leader['ticker']} (Score {leader['score']:.0f}), "
            f"sacrifice weakest pig {matched_pig['ticker']} (OE {matched_pig['efficiency']:.1f}, Held {matched_pig['days_held']}d, P/L {matched_pig['pnl_pct']:+.1f}%)."
            if matched_pig else f"Initiate pilot in {leader['ticker']} (Score {leader['score']:.0f})."
        )

        swap_directives.append({
            'leader': leader,
            'sacrificed_pig': matched_pig,
            'buy_shares': buy_shares,
            'swap_capital_inr': swap_cap,
            'directive_text': directive_text,
        })

    return {
        'fresh_leaders': fresh_leaders,
        'pigs': pigs,
        'swap_directives': swap_directives,
        'total_stagnant_capital_inr': tot_stagnant_cap,
        'dead_money_count': dead_money_cnt,
    }


def compute_portfolio_traction_mirror(
    portfolio_df: pd.DataFrame,
    market_data: dict,
    total_portfolio_value: float = None,
) -> dict:
    """
    Mark Minervini / Oliver Kell 'Trader's Mirror' (Internal Portfolio Feedback Loop).
    Separates all CASH_EQUIVALENT_TICKERS (LIQUIDBEES, LIQUIDCASE, LIQUIDETF, GOLDBEES, SILVERBEES)
    cleanly into cash reserves so equity traction & Deployed Equity % are 100% undistorted.
    """
    empty_res = {
        'total_holdings': 0,
        'cash_tickers_count': 0,
        'above_21_low_count': 0,
        'above_21_low_pct': 0.0,
        'green_count': 0,
        'house_money_count': 0,
        'broken_count': 0,
        'dead_money_count': 0,
        'dead_money_pct': 0.0,
        'pilot_green_count': 0,
        'pilot_total': 0,
        'equity_val': 0.0,
        'cash_reserve_val': 0.0,
        'total_nav': float(total_portfolio_value or 0.0),
        'deployed_equity_pct': 0.0,
        'cash_reserve_pct': 100.0 if (total_portfolio_value or 0.0) > 0 else 0.0,
        'verdict': 'Neutral',
        'color': '#94a3b8',
    }
    if portfolio_df is None or portfolio_df.empty:
        return empty_res

    m_data = market_data or {}
    records = []
    equity_val = 0.0
    cash_reserve_val = 0.0
    cash_tickers_count = 0

    for idx_r, row in portfolio_df.iterrows():
        raw_t = str(row.get('ticker', '')).strip()
        if not raw_t:
            continue
        clean_sym = raw_t.replace('.NS', '').replace('.BO', '').strip().upper()
        try:
            avg_buy = float(row.get('Avg Buy Price', 0.0) or 0.0)
            qty = float(row.get('Qty', 0.0) or 0.0)
        except Exception:
            continue

        df = m_data.get(raw_t)
        if df is not None and not df.empty:
            c_col = 'close' if 'close' in df.columns else 'Close'
            curr_p_raw = df[c_col].iloc[-1] if c_col in df.columns else np.nan
            curr_p = float(curr_p_raw) if pd.notna(curr_p_raw) and np.isfinite(float(curr_p_raw)) else avg_buy
        else:
            row_cp = row.get('Current Price', 0.0)
            curr_p = float(row_cp) if pd.notna(row_cp) and float(row_cp or 0.0) > 0 else avg_buy

        pos_val = max(0.0, curr_p * qty)

        if clean_sym in CASH_EQUIVALENT_TICKERS:
            cash_reserve_val += pos_val
            cash_tickers_count += 1
            continue

        equity_val += pos_val
        if df is None or df.empty:
            continue

        try:
            c_col = 'close' if 'close' in df.columns else 'Close'
            l_col = 'low' if 'low' in df.columns else 'Low'
            if avg_buy <= 0:
                avg_buy = curr_p
            pnl_pct = ((curr_p / avg_buy) - 1.0) * 100.0 if avg_buy > 0 else 0.0

            if 'ema_21_low' in df.columns and pd.notna(df['ema_21_low'].iloc[-1]):
                e21_low = float(df['ema_21_low'].iloc[-1])
            elif l_col in df.columns:
                e21_low = float(df[l_col].ewm(span=21, adjust=False).mean().iloc[-1])
            else:
                e21_low = curr_p * 0.95

            s50 = float(df['sma_50'].iloc[-1]) if ('sma_50' in df.columns and pd.notna(df['sma_50'].iloc[-1])) else curr_p
            above_21l = curr_p >= e21_low
            holding_spine = above_21l or (curr_p >= s50)
            is_green = pnl_pct > 0.0
            is_hm = (pnl_pct >= 8.0) and holding_spine
            is_broken = (pnl_pct <= -7.5) or (pnl_pct < -2.0 and not holding_spine)

            # Delta 1: Catalytic ATR Velocity calculation
            h_col = 'high' if 'high' in df.columns else 'High'
            high = df[h_col]
            low = df[l_col]
            close_prev = df[c_col].shift(1)
            tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
            atr_14 = float(tr.rolling(14, min_periods=5).mean().iloc[-1])
            atr_prog = round((curr_p - avg_buy) / atr_14, 1) if atr_14 > 0 else 0.0
            is_dead = (pnl_pct < 5.0) and (atr_prog < 1.5) and (len(df) >= 12)

            records.append({
                'ticker': raw_t,
                'pos_val': pos_val,
                'pnl_pct': pnl_pct,
                'above_21l': above_21l,
                'is_green': is_green,
                'is_hm': is_hm,
                'is_broken': is_broken,
                'is_dead_money': is_dead,
                'atr_progress': atr_prog,
                'row_order': idx_r,
            })
        except Exception:
            continue

    tracked_total = equity_val + cash_reserve_val
    total_nav = max(tracked_total, float(total_portfolio_value or 0.0))
    if total_nav > tracked_total:
        cash_reserve_val += (total_nav - tracked_total)

    deployed_equity_pct = round((equity_val / total_nav) * 100.0, 1) if total_nav > 0 else 0.0
    cash_reserve_pct = round((cash_reserve_val / total_nav) * 100.0, 1) if total_nav > 0 else 0.0

    total_h = len(records)
    if total_h == 0:
        return {
            **empty_res,
            'cash_tickers_count': cash_tickers_count,
            'equity_val': round(equity_val, 2),
            'cash_reserve_val': round(cash_reserve_val, 2),
            'total_nav': round(total_nav, 2),
            'deployed_equity_pct': deployed_equity_pct,
            'cash_reserve_pct': cash_reserve_pct,
        }

    above_21l_cnt = sum(1 for r in records if r['above_21l'])
    above_21l_pct = (above_21l_cnt / total_h) * 100.0
    green_cnt = sum(1 for r in records if r['is_green'])
    hm_cnt = sum(1 for r in records if r['is_hm'])
    broken_cnt = sum(1 for r in records if r['is_broken'])

    # Pilot Traction: evaluate the last 5 added positions (bottom 5 rows of sheet)
    pilots = records[-min(5, total_h):]
    pilot_green_cnt = sum(1 for p in pilots if p['is_green'] or p['above_21l'])
    pilot_total = len(pilots)

    if above_21l_pct >= 55.0 and pilot_green_cnt >= 3:
        verdict = "🟢 Positive Traction"
        color = "#10b981"
    elif above_21l_pct >= 35.0 or hm_cnt >= 8:
        verdict = "🟡 Selective Traction"
        color = "#f59e0b"
    else:
        verdict = "🔴 Under Pressure"
        color = "#ef4444"

    dead_money_cnt = sum(1 for r in records if r.get('is_dead_money', False))
    dead_money_pct = (dead_money_cnt / total_h) * 100.0 if total_h > 0 else 0.0

    return {
        'total_holdings': total_h,
        'cash_tickers_count': cash_tickers_count,
        'above_21_low_count': above_21l_cnt,
        'above_21_low_pct': round(above_21l_pct, 0),
        'green_count': green_cnt,
        'house_money_count': hm_cnt,
        'broken_count': broken_cnt,
        'dead_money_count': dead_money_cnt,
        'dead_money_pct': round(dead_money_pct, 0),
        'pilot_green_count': pilot_green_cnt,
        'pilot_total': pilot_total,
        'equity_val': round(equity_val, 2),
        'cash_reserve_val': round(cash_reserve_val, 2),
        'total_nav': round(total_nav, 2),
        'deployed_equity_pct': deployed_equity_pct,
        'cash_reserve_pct': cash_reserve_pct,
        'verdict': verdict,
        'color': color,
    }


def _extract_last_date_str(df: pd.DataFrame) -> str:
    """Safely extracts the latest YYYY-MM-DD bar date string from an OHLCV or breadth DataFrame."""
    if df is None or df.empty:
        return ''
    try:
        for col_name in ('date', 'Date'):
            if col_name in df.columns:
                val = df[col_name].dropna().iloc[-1]
                return pd.to_datetime(val).strftime('%Y-%m-%d')
        val = df.index[-1]
        return pd.to_datetime(val).strftime('%Y-%m-%d')
    except Exception:
        return str(df.index[-1])[:10] if len(df.index) > 0 else ''


def _query_latest_breadth_date(b_market: str = 'IN') -> str:
    """O(1) indexed lookup for latest non-empty market_breadth_daily date without loading full breadth history."""
    conn = None
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT MAX(date) FROM market_breadth_daily WHERE market = ?", (b_market,))
        row = c.fetchone()
        return str(row[0])[:10] if (row and row[0]) else ''
    except Exception:
        return ''
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _select_scope_a_core_columns(
    close_p: pd.DataFrame,
    vol_p: pd.DataFrame,
    prof: dict,
) -> list:
    """
    Selects Scope A liquid core columns matching each benchmark index:
    - 'INDIA': Official NIFTY Total Market (750) constituents present in matrix.
    - 'US_IXIC': Active non-ETF NASDAQ stocks meeting institutional liquidity
                 (price >= $5.0 & 20D ADTV >= $2.0M) matching ^IXIC.
    """
    if close_p is None or close_p.empty:
        return []

    if not prof['is_us']:
        n750_tickers = set(fetch_nifty_total_market_tickers(show_progress=False))
        core_cols = [c for c in close_p.columns if c in n750_tickers]
        if len(core_cols) < 100:
            core_cols = [
                c for c in close_p.columns
                if str(c).endswith('.NS') and str(c)[:-3].upper() not in CASH_EQUIVALENT_TICKERS
            ][:750]
        return core_cols

    # US_IXIC Core: active NASDAQ equities with institutional price/liquidity filter
    raw_us = [
        c for c in close_p.columns
        if not str(c).endswith(('.NS', '.BO')) and not str(c).startswith('^')
    ]
    nasdaq_set = set()
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT yf_ticker FROM security_master WHERE market='US' AND status='ACTIVE' AND exchange LIKE 'NASDAQ%'"
        ).fetchall()
        conn.close()
        nasdaq_set = {str(r[0]).strip().upper() for r in rows}
    except Exception:
        nasdaq_set = set()

    if nasdaq_set:
        nq_cols = [c for c in raw_us if str(c).strip().upper() in nasdaq_set]
    else:
        nq_cols = []

    pool = nq_cols if len(nq_cols) >= 100 else raw_us
    if len(pool) >= 200 and not close_p.empty:
        sub_c = close_p[pool]
        last_c = sub_c.iloc[-1].fillna(0.0)
        if vol_p is not None and not vol_p.empty:
            sub_v = vol_p[[c for c in pool if c in vol_p.columns]]
            adtv_m = ((sub_c.tail(20) * sub_v.tail(20)).mean() / 1e6).fillna(0.0)
            liq = [c for c in pool if (last_c.get(c, 0.0) >= 5.0 and adtv_m.get(c, 0.0) >= 2.0)]
        else:
            liq = [c for c in pool if last_c.get(c, 0.0) >= 5.0]
        if len(liq) >= 100:
            return liq
    return pool


def _build_cache_fingerprint(
    market: str,
    persona_data: dict = None,
    breadth_latest_date: str = '',
    benchmark_df: pd.DataFrame = None,
) -> dict:
    """Constructs lightweight metadata fingerprint stored inside tide_turn_json['cache_meta']."""
    prof = get_market_brain_profile(market)
    p = persona_data or {}
    l21_raw = p.get('leader_21ema_pct')
    bm_last = _extract_last_date_str(benchmark_df)
    if benchmark_df is not None and prof['is_us'] and 'ticker' in benchmark_df.columns:
        tk_vals = set(str(x).upper() for x in benchmark_df['ticker'].dropna().unique())
        if 'BENCHMARK' in tk_vals or '^CRSLDX' in tk_vals:
            bm_last = ''
    return {
        'schema_v': BRAIN_CACHE_SCHEMA_VERSION,
        'profile_key': prof['profile_key'],
        'benchmark_sym': prof['benchmark_sym'],
        'session_date': _resolve_market_session_date(prof['profile_key']),
        'persona_date': str(p.get('date', '') or '')[:10],
        'persona_score': round(float(p.get('composite_score', 50.0) or 50.0), 2),
        'breakout_health': round(float(p.get('breakout_health_pct', 55.0) or 55.0), 2),
        'leader_21ema_pct': round(float(l21_raw), 2) if l21_raw is not None else -1.0,
        'breadth_latest_date': str(breadth_latest_date or '')[:10],
        'benchmark_last_date': bm_last,
    }


def _is_brain_cache_valid(
    cached: dict,
    expected_fp: dict,
    matrix_mtime: float = None,
    require_full_universe: bool = True,
    market: str = 'INDIA',
) -> bool:
    """Verifies cached Brain payload against profile schema, matrix freshness, session, persona, breadth, and benchmark."""
    if not cached or not cached.get('tide_turn') or cached.get('confluence') is None:
        return False
    tt_c = cached.get('tide_turn') or {}
    if tt_c.get('velocity') is None or tt_c.get('stealth_top') is None:
        return False
    prof = get_market_brain_profile(expected_fp.get('profile_key') or market)
    min_univ = int(prof['min_cache_universe'])
    if require_full_universe and int(cached.get('universe_count', 0) or 0) < min_univ:
        return False

    if matrix_mtime is not None:
        upd_str = cached.get('updated_at', '')
        if not upd_str:
            return False
        try:
            upd_ts = datetime.strptime(upd_str[:19], '%Y-%m-%d %H:%M:%S').timestamp()
        except Exception:
            return False
        if upd_ts < matrix_mtime:
            return False

    saved_meta = tt_c.get('cache_meta') or {}
    required_keys = (
        'schema_v',
        'session_date',
        'benchmark_sym',
        'persona_date',
        'persona_score',
        'breakout_health',
        'leader_21ema_pct',
        'breadth_latest_date',
    )
    for k in required_keys:
        if saved_meta.get(k) != expected_fp.get(k):
            return False

    exp_bm = expected_fp.get('benchmark_last_date', '')
    if exp_bm and saved_meta.get('benchmark_last_date', '') != exp_bm:
        return False
    return True


def run_artificial_brain_synthesis(
    benchmark_df: pd.DataFrame = None,
    portfolio_tickers: set = None,
    industry_map: dict = None,
    fundamentals_map: dict = None,
    market: str = 'INDIA',
    force_recompute: bool = False
) -> dict:
    """
    Master entrypoint called by app.py & eod_pipeline_sync.py for both 'INDIA' (^CRSLDX)
    and 'US_IXIC' (^IXIC). Verifies lightweight cache_meta before unpickling the large
    price matrix, auto-heals missing US Breadth history / Persona on cold start, runs
    dual-scope vectorized synthesis, stamps currency-aware sizing, and persists cache.
    """
    prof = get_market_brain_profile(market)
    p_market = prof['profile_key']
    b_market = prof['breadth_market']
    is_us_mkt = bool(prof['is_us'])

    # Guard against accidental India benchmark injection when computing US Nasdaq Brain
    if is_us_mkt and benchmark_df is not None and not benchmark_df.empty and 'ticker' in benchmark_df.columns:
        tk_vals = set(str(x).upper() for x in benchmark_df['ticker'].dropna().unique())
        if 'BENCHMARK' in tk_vals or '^CRSLDX' in tk_vals:
            benchmark_df = None

    matrix_path = 'historical_prices_matrix.pkl'
    matrix_df = None
    session_date = _resolve_market_session_date(p_market)
    matrix_bar_date = session_date

    persona_data = get_latest_market_persona(market=p_market)
    breadth_latest_date = _query_latest_breadth_date(b_market=b_market)

    # Cloud-first fast return (<15ms) when running without local price matrix .pkl
    if not force_recompute and not os.path.exists(matrix_path):
        try:
            cached_cloud = get_brain_synthesis_cache(market=p_market)
            if isinstance(cached_cloud, dict) and cached_cloud.get('tide_turn'):
                return {
                    'matrix_bar_date': cached_cloud.get('matrix_bar_date', matrix_bar_date),
                    'universe_count': int(cached_cloud.get('universe_count', 0) or 0),
                    'tide_turn': cached_cloud['tide_turn'],
                    'confluence': cached_cloud.get('confluence') or [],
                    'from_cache': True,
                    'updated_at': cached_cloud.get('updated_at', ''),
                }
        except Exception:
            pass

    # Ultra-fast cache hit path (<5ms): skip ONLY if persona exists & cache_meta matches
    if not force_recompute and os.path.exists(matrix_path) and persona_data is not None:
        try:
            cached_fast = get_brain_synthesis_cache(market=p_market)
            mat_mtime = os.path.getmtime(matrix_path)
            expected_fp_fast = _build_cache_fingerprint(
                market=p_market,
                persona_data=persona_data,
                breadth_latest_date=breadth_latest_date,
                benchmark_df=benchmark_df,
            )
            if _is_brain_cache_valid(cached_fast, expected_fp_fast, matrix_mtime=mat_mtime, require_full_universe=True, market=p_market):
                return {
                    'matrix_bar_date': cached_fast.get('matrix_bar_date', matrix_bar_date),
                    'universe_count': int(cached_fast.get('universe_count', 0)),
                    'tide_turn': cached_fast['tide_turn'],
                    'confluence': cached_fast['confluence'],
                    'from_cache': True,
                    'updated_at': cached_fast.get('updated_at', ''),
                }
        except Exception:
            pass

    close_p, open_p, high_p, low_p, vol_p = None, None, None, None, None
    if os.path.exists(matrix_path):
        try:
            with open(matrix_path, 'rb') as f:
                matrix_df = pickle.load(f)
            if isinstance(matrix_df, pd.DataFrame) and not matrix_df.empty:
                close_p, open_p, high_p, low_p, vol_p = _extract_ohlcv_panels(matrix_df)
                close_p, open_p, high_p, low_p, vol_p, eff_dt = _trim_panels_to_valid_market_bars(
                    close_p, open_p, high_p, low_p, vol_p, market=p_market
                )
                matrix_bar_date = eff_dt or session_date
        except Exception as e:
            print(f"[BrainEngine] Matrix load warning: {e}")

    # Secondary cache hit check (verified against trimmed market bar_date & session_date)
    if not force_recompute and persona_data is not None:
        cached = get_brain_synthesis_cache(market=p_market, required_matrix_bar_date=matrix_bar_date)
        expected_fp = _build_cache_fingerprint(
            market=p_market,
            persona_data=persona_data,
            breadth_latest_date=breadth_latest_date,
            benchmark_df=benchmark_df,
        )
        if _is_brain_cache_valid(cached, expected_fp, matrix_mtime=None, require_full_universe=(matrix_df is not None), market=p_market):
            return {
                'matrix_bar_date': cached.get('matrix_bar_date', matrix_bar_date),
                'universe_count': int(cached.get('universe_count', 0)),
                'tide_turn': cached['tide_turn'],
                'confluence': cached['confluence'],
                'from_cache': True,
                'updated_at': cached.get('updated_at', ''),
            }

    if fundamentals_map is None:
        try:
            fundamentals_map = get_all_fundamentals_cache()
        except Exception:
            fundamentals_map = {}

    if close_p is None and matrix_df is not None:
        close_p, open_p, high_p, low_p, vol_p = _extract_ohlcv_panels(matrix_df)

    # Scope A: Filter Liquid Core (India Nifty 750 or US Liquid NASDAQ matching ^IXIC)
    core_cols = _select_scope_a_core_columns(close_p, vol_p, prof)
    if close_p is not None and not close_p.empty and core_cols:
        close_core = close_p[core_cols]
        high_core = high_p[core_cols] if (high_p is not None and not high_p.empty) else close_core
        low_core = low_p[core_cols] if (low_p is not None and not low_p.empty) else close_core
        vol_core = vol_p[core_cols] if (vol_p is not None and not vol_p.empty) else None

        # Cold-start 252-day backfill if cohort has <60 rows (e.g. Day-1 US_IXIC), then self-heal latest bar
        _backfill_market_breadth_from_matrix(close_core, high_core, low_core, vol_core=vol_core, market=b_market, lookback_days=252)
        _self_heal_market_breadth_row(matrix_bar_date, close_core, high_core, low_core, vol_core=vol_core, market=b_market)
    else:
        close_core, high_core, low_core, vol_core = None, None, None, None

    # Day-1 Persona Auto-Heal: never evaluate/cache dummy 50/55 defaults when Persona row missing/stale
    if persona_data is None or (str(persona_data.get('date', '') or '')[:10] < matrix_bar_date):
        try:
            from market_persona_engine import calculate_composite_market_persona
            persona_data = calculate_composite_market_persona(market=p_market, force_recompute=True) or persona_data
        except Exception as e:
            print(f"[BrainEngine] Persona self-heal notice ({p_market}): {e}")

    persona_score = float((persona_data or {}).get('composite_score', 50.0) or 50.0)

    # Fallback benchmark fetch if benchmark_df is None (400d window ensures 252+ bars for RS Line)
    if benchmark_df is None or benchmark_df.empty:
        try:
            from market_data import fetch_benchmark_data
            benchmark_df = fetch_benchmark_data(days=400, ticker=prof['benchmark_sym'])
        except Exception as e:
            print(f"[BrainEngine] Benchmark fetch warning ({prof['benchmark_sym']}): {e}")

    # Scope B: Compute 7-Pillar Omni-Confluence candidates across Full Active Market Universe
    confluence_list, ironclad_count, universe_count = compute_omni_confluence_funnel(
        close_p=close_p,
        open_p=open_p,
        high_p=high_p,
        low_p=low_p,
        vol_p=vol_p,
        benchmark_df=benchmark_df,
        portfolio_tickers=None if is_us_mkt else portfolio_tickers,
        industry_map=None if is_us_mkt else industry_map,
        fundamentals_map=fundamentals_map,
        tide_stage_num=3,
        persona_score=persona_score,
        top_n=25,
        market=p_market,
    )

    # Fetch canonical breadth history from database for exact Breadth Velocity & Stealth Top parity
    breadth_df = None
    try:
        from breadth_engine import fetch_full_breadth_history
        breadth_df = fetch_full_breadth_history(market=b_market, days=1500)
    except Exception as e:
        print(f"[BrainEngine] Breadth history fetch warning: {e}")

    # Evaluate Authoritative 5-Stage Tide-Turn State Machine BEFORE stamping Funnel execution plans
    tide_turn = compute_tide_turn_state_machine(
        benchmark_df=benchmark_df,
        close_core=close_core,
        high_core=high_core,
        low_core=low_core,
        vol_core=vol_core,
        ironclad_count=ironclad_count,
        persona_data=persona_data,
        breadth_df=breadth_df,
        market=p_market,
    )

    # Stamp live stage-normalized Alloc %, Risk %, and currency-aware execution plans onto every Funnel candidate
    apply_stage_sizing_to_confluence_list(
        confluence_list,
        tide_stage_num=int(tide_turn.get('stage_num', 3)),
        market=p_market,
        bh_is_pre_ftd_lag=bool(tide_turn.get('bh_is_pre_ftd_lag', False)),
        ftd_suspect_warning=bool(tide_turn.get('ftd_suspect_warning', False)),
    )

    final_breadth_date = _extract_last_date_str(breadth_df) or _query_latest_breadth_date(b_market=b_market)
    tide_turn['cache_meta'] = _build_cache_fingerprint(
        market=p_market,
        persona_data=persona_data,
        breadth_latest_date=final_breadth_date,
        benchmark_df=benchmark_df,
    )

    save_brain_synthesis_cache(
        date_str=session_date,
        market=p_market,
        matrix_bar_date=matrix_bar_date,
        universe_count=universe_count,
        tide_turn_data=tide_turn,
        confluence_list=confluence_list
    )

    return {
        'matrix_bar_date': matrix_bar_date,
        'universe_count': universe_count,
        'tide_turn': tide_turn,
        'confluence': confluence_list,
        'from_cache': False,
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
