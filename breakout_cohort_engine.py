"""
=============================================================================
INSTITUTIONAL BREAKOUT HEALTH & DISTRIBUTION RADAR
=============================================================================
Vectorized Trailing 15-Day Rolling Breakout Cohort Engine
Inspired by Mark Minervini (Batting Average & Feedback Loop),
William J. O'Neil (CANSLIM Breakout Health), and Pradeep Bonde (Follow-Through).

Philosophy:
"You don't trade your opinion of the market; you trade the feedback from
your universe of leading stocks. When breakouts follow through, you press
exposure. When breakouts breach their consolidation base floors, institutions
are distributing into strength—you throttle back capital."
=============================================================================
"""

import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

try:
    from styles import clean_html
except ImportError:
    def clean_html(html_str: str) -> str:
        return "\n".join(line.strip() for line in html_str.splitlines() if line.strip())


def _load_cohort_fallback_enrichment(is_us: bool) -> dict:
    """
    Market-isolated metadata fallback (canonical industry / sector) for tickers
    not yet present in Industry Group Matrix intelligence. Uses Turso-safe database
    adapter instead of direct sqlite3 file opens.
    """
    fallback = {}

    synonym_map = {}
    try:
        from industry_group_engine import TAXONOMY_SYNONYM_MAP
        synonym_map = TAXONOMY_SYNONYM_MAP or {}
    except Exception:
        pass

    try:
        from database import get_fundamentals_map_for_engine
        funda_map, _ = get_fundamentals_map_for_engine("US" if is_us else "IN")
        for tk_u, info in (funda_map or {}).items():
            raw_ind = str((info or {}).get("industry") or "").strip()
            norm_ind = synonym_map.get(raw_ind, raw_ind)
            if not norm_ind or norm_ind in ("Unknown", "None", "nan"):
                continue
            k_up = str(tk_u).strip().upper()
            fallback.setdefault(k_up, norm_ind)
            base_u = re.sub(r"\.(NS|BO)$", "", k_up)
            fallback.setdefault(base_u, norm_ind)
    except Exception:
        pass

    if not is_us:
        map_path = os.path.join(base_dir, "industry_map.json")
        if os.path.exists(map_path):
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    raw_map = json.load(f)
                for k, v in (raw_map or {}).items():
                    if not v or str(v) in ("Unknown", "None", "nan"):
                        continue
                    k_u = str(k).strip().upper()
                    if k_u.endswith((".NS", ".BO")):
                        fallback.setdefault(k_u, str(v).strip())
                        fallback.setdefault(k_u[:-3], str(v).strip())
            except Exception:
                pass
    else:
        gics_path = os.path.join(base_dir, "data", "sp500_gics_map.json")
        if os.path.exists(gics_path):
            try:
                with open(gics_path, "r", encoding="utf-8") as f:
                    gics_map = json.load(f)
                for k, v in (gics_map or {}).items():
                    if v and not str(k).upper().endswith((".NS", ".BO")):
                        fallback.setdefault(str(k).strip().upper(), str(v).strip())
            except Exception:
                pass

    return fallback


# -----------------------------------------------------------------------------
# CORE QUANTITATIVE COHORT ENGINE
# -----------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def compute_breakout_health_radar(
    scope: str = 'IN',
    lookback_days: int = 15,
    min_adtv_cr: float = 3.0,
    min_price: float = 10.0,
    force_refresh: bool = False,
    min_adtv_m: float = 2.0,
) -> tuple[dict, pd.DataFrame]:
    """
    Computes trailing 15-day rolling breakout cohort metrics strictly within the
    selected market (`IN` = NSE/BSE or `US` = US equities):
    1. Filters matrix columns & active settled trading sessions to `scope` ONLY.
    2. Identifies Stage 2 base breakouts occurring in the last `lookback_days` sessions.
    3. Evaluates current price against baseline floor:
       - 🚀 Runner: Price > Pivot
       - 🟡 Base Retest: Base Floor <= Price <= Pivot
       - 🔴 Failed: Price < Base Floor (Consolidation low violated)
    4. Computes aggregate Batting Average %, Failure Rate %, and Momentum Payoff.
    """
    is_us = str(scope or 'IN').strip().upper() in ('US', 'USA', 'US_IXIC', 'AMERICA')
    matrix_path = 'historical_prices_matrix.pkl'
    if not os.path.exists(matrix_path):
        try:
            from database import load_cloud_json_cache_cached
            bundle, _ = load_cloud_json_cache_cached("breadth_aux_bundle_us" if is_us else "breadth_aux_bundle_in")
            if isinstance(bundle, dict) and isinstance(bundle.get("breakout_cohort"), (list, tuple)) and len(bundle["breakout_cohort"]) == 2:
                b_m, b_df = bundle["breakout_cohort"]
                if isinstance(b_m, dict) and isinstance(b_df, pd.DataFrame):
                    return b_m, b_df.copy()
        except Exception:
            pass
        return _empty_radar_state(is_us=is_us)

    try:
        matrix_df = pd.read_pickle(matrix_path)
    except Exception as e:
        print(f"Error reading {matrix_path}: {e}")
        return _empty_radar_state(is_us=is_us)

    if matrix_df.empty or not isinstance(matrix_df.columns, pd.MultiIndex):
        return _empty_radar_state(is_us=is_us)

    # Strictly isolate market columns & strip closed-market holiday/partial bars (e.g., US-only or IN-only dates)
    try:
        from price_history_manager import extract_market_ohlcv_panels
        panels = extract_market_ohlcv_panels(
            matrix_df,
            market="US" if is_us else "IN",
            include_etfs=False,
            deduplicate_dual_in=(not is_us),
            min_coverage_ratio=0.15,
            min_abs=10,
        )
        close = panels["close"]
        high = panels["high"]
        low = panels["low"]
        volume = panels["volume"]
    except Exception as e:
        print(f"Error extracting OHLCV levels from matrix ({scope}): {e}")
        return _empty_radar_state(is_us=is_us)

    if close.empty or len(close) < 40:
        return _empty_radar_state(is_us=is_us)

    # Market-appropriate liquidity & price floors
    eff_min_adtv = float(min_adtv_m if (is_us and abs(float(min_adtv_cr) - 3.0) < 1e-6) else min_adtv_cr)
    eff_min_price = float(5.0 if (is_us and abs(float(min_price) - 10.0) < 1e-6) else min_price)
    adtv_scale = 1e6 if is_us else 1e7  # USD $M vs INR Cr

    recent_close = close.iloc[-1]
    recent_vol_20 = volume.iloc[-20:].mean()
    recent_adtv = (recent_close * recent_vol_20) / adtv_scale

    valid_mask = (
        recent_close.notna()
        & recent_adtv.notna()
        & (recent_adtv >= eff_min_adtv)
        & (recent_close >= eff_min_price)
    )
    valid_tickers = valid_mask[valid_mask == True].index

    if len(valid_tickers) == 0:
        return _empty_radar_state(is_us=is_us)

    close_v = close[valid_tickers]
    high_v = high[valid_tickers]
    low_v = low[valid_tickers]
    vol_v = volume[valid_tickers]

    # Calculate 200 SMA and 20D Average Volume
    sma_200 = close_v.rolling(200, min_periods=150).mean()
    vol_20 = vol_v.rolling(20, min_periods=15).mean()

    # Strict Market-Scoped Industry Group Intelligence & Fallback
    stock_group_intel = {}
    us_tv_map = {}
    format_tradingview_symbol = None
    try:
        from industry_group_engine import (
            get_stock_group_intelligence,
            load_exporter_enrichment_maps,
            format_tradingview_symbol as _fmt_tv,
        )
        format_tradingview_symbol = _fmt_tv
        univ_scope = "US (S&P 500 / Nasdaq)" if is_us else "India (NSE/BSE)"
        stock_group_intel = get_stock_group_intelligence(taxonomy="canonical", universe_scope=univ_scope) or {}
        if is_us:
            us_tv_map, _ = load_exporter_enrichment_maps(is_us=True)
    except Exception:
        pass

    fallback_ind = _load_cohort_fallback_enrichment(is_us=is_us)
    grp_grade_map = {}
    for _gi in stock_group_intel.values():
        if isinstance(_gi, dict):
            _gn = str(_gi.get('group_name') or '').strip()
            _gb = str(_gi.get('group_grade_badge') or '').strip()
            if _gn and _gb and _gb != '—':
                grp_grade_map.setdefault(_gn.upper(), (_gb, bool(_gi.get('is_top_20_pct', False))))
    clean_sym_fn = lambda t: re.sub(r'\.(NS|BO)$', '', str(t)).replace('👑', '').replace('🔥', '').replace('⚡', '').replace('🚀', '').strip()

    breakouts = []
    total_sessions = len(close_v)
    eval_window = min(lookback_days, total_sessions - 25)
    as_of_date = close_v.index[-1].strftime('%Y-%m-%d') if hasattr(close_v.index[-1], 'strftime') else str(close_v.index[-1])[:10]

    for idx_offset in range(eval_window, 0, -1):
        t_idx = total_sessions - idx_offset
        t_date = close_v.index[t_idx]

        # Prior 20 sessions consolidation base: t-20 to t-1
        base_high = high_v.iloc[t_idx - 20:t_idx].max()
        base_low = low_v.iloc[t_idx - 20:t_idx].min()
        base_depth = (base_high - base_low) / base_high.replace(0, np.nan)

        c_today = close_v.iloc[t_idx]
        c_yest = close_v.iloc[t_idx - 1]
        v_today = vol_v.iloc[t_idx]
        v_avg = vol_20.iloc[t_idx]
        s200_today = sma_200.iloc[t_idx]

        # Breakout Trigger Criteria (Minervini / O'Neil) + Halted/Split Sanity Guards:
        cond = (
            (c_today > base_high) &
            (c_yest <= base_high) &
            (c_today > s200_today) &
            (v_today > 0) &
            (v_avg > 0) &
            (v_today >= 1.25 * v_avg) &
            (base_depth <= 0.35) &
            (base_low > 0) &
            (base_high > 0) &
            (c_yest > 0) &
            ((c_today / c_yest) <= 3.0)
        )

        triggered_tickers = cond[cond == True].index.tolist()
        for tkr in triggered_tickers:
            p_pivot = float(base_high[tkr])
            p_floor = float(base_low[tkr])
            curr_p = float(recent_close[tkr])
            v_surge = float(v_today[tkr] / max(v_avg[tkr], 1.0))
            gain_pct = ((curr_p - p_pivot) / p_pivot) * 100.0

            if curr_p > p_pivot:
                status = '🚀 Runner'
            elif curr_p >= p_floor:
                status = '🟡 Base Retest'
            else:
                status = '🔴 Failed'

            sym_clean = clean_sym_fn(tkr)
            tkr_u = str(tkr).strip().upper()
            sym_u = sym_clean.upper()

            if is_us:
                g_info = stock_group_intel.get(tkr_u) or stock_group_intel.get(sym_u) or stock_group_intel.get(sym_u.replace('-', '.')) or {}
            else:
                g_info = stock_group_intel.get(tkr_u) or stock_group_intel.get(sym_u) or {}

            ind_grade = g_info.get('group_grade_badge', '—')
            industry = g_info.get('group_name') or fallback_ind.get(tkr_u) or fallback_ind.get(sym_u) or 'Unknown'
            is_top_20 = bool(g_info.get('is_top_20_pct', False) or ('A+' in str(ind_grade) or str(ind_grade).startswith('A ')))
            if ind_grade == '—' and industry != 'Unknown':
                _gb_tup = grp_grade_map.get(str(industry).strip().upper())
                if _gb_tup:
                    ind_grade, _g_top20 = _gb_tup
                    is_top_20 = bool(is_top_20 or _g_top20)

            if format_tradingview_symbol:
                tv_sym = format_tradingview_symbol(
                    ticker=sym_clean,
                    raw_ticker=str(tkr),
                    is_us=is_us,
                    us_tv_map=us_tv_map,
                    prefix_us=True,
                )
            else:
                ex_pfx = 'US' if is_us else ('BSE' if tkr_u.endswith('.BO') else 'NSE')
                tv_sym = f"{ex_pfx}:{sym_u.replace('-', '.' if is_us else '_')}"

            breakouts.append({
                'ticker': sym_clean,
                'raw_ticker': tkr,
                'tv_symbol': tv_sym,
                'breakout_date': t_date.strftime('%Y-%m-%d'),
                'pivot_price': round(p_pivot, 2),
                'base_floor': round(p_floor, 2),
                'current_price': round(curr_p, 2),
                'gain_from_pivot_pct': round(gain_pct, 2),
                'volume_surge': round(v_surge, 1),
                'status': status,
                'days_ago': idx_offset - 1,
                'ind_grade': ind_grade,
                'industry': industry,
                'is_top_20': is_top_20
            })

    if not breakouts:
        return _empty_radar_state(is_us=is_us)

    df_bo = pd.DataFrame(breakouts)
    # Deduplicate: if a stock triggered multiple breakouts, keep the earliest breakout in the window
    df_bo = df_bo.sort_values('days_ago', ascending=False).drop_duplicates(subset=['ticker'], keep='first')
    df_bo = df_bo.sort_values('gain_from_pivot_pct', ascending=False).reset_index(drop=True)

    total_n = len(df_bo)
    runners_df = df_bo[df_bo['status'] == '🚀 Runner']
    retest_df = df_bo[df_bo['status'] == '🟡 Base Retest']
    failed_df = df_bo[df_bo['status'] == '🔴 Failed']

    runners_n = len(runners_df)
    retest_n = len(retest_df)
    failed_n = len(failed_df)

    batting_avg = (runners_n / total_n) * 100.0 if total_n > 0 else 0.0
    failure_rate = (failed_n / total_n) * 100.0 if total_n > 0 else 0.0
    retest_pct = (retest_n / total_n) * 100.0 if total_n > 0 else 0.0

    avg_gain = runners_df['gain_from_pivot_pct'].mean() if not runners_df.empty else 0.0
    avg_loss = failed_df['gain_from_pivot_pct'].mean() if not failed_df.empty else -5.0
    # Professional desks cap downside risk at a standard 7.0% stop-loss rather than letting failures free-fall
    capped_loss = min(abs(avg_loss), 7.0)
    payoff_ratio = avg_gain / max(capped_loss, 1.0) if avg_gain > 0 else 0.0
    expectancy_edge = (batting_avg / 100.0 * avg_gain) - (failure_rate / 100.0 * capped_loss)

    # Determine Regime Classification
    if failure_rate > 50.0:
        regime_label = "🔴 STEALTH DISTRIBUTION TRAP"
        regime_desc = "Breakout failure rate is severe (>50%). Smart money is quietly liquidating into strength. Freeze new breakout buys; tighten stops to day's low or breakeven."
        regime_color = "#ef4444"
        regime_tag = "DISTRIBUTION"
    elif failure_rate > 30.0 or batting_avg < 50.0:
        regime_label = "🟡 SELECTIVE MOMENTUM REGIME"
        regime_desc = "Breakout follow-through is mixed. Trade selectively; prioritize Top 20% industry leaders and lock partial profits at 1.5R–2R."
        regime_color = "#f59e0b"
        regime_tag = "SELECTIVE"
    else:
        regime_label = "🟢 HEALTHY MOMENTUM REGIME"
        regime_desc = "Strong institutional follow-through. Low failure rate and expanding gains. Full position sizing and standard 5% base stops recommended."
        regime_color = "#10b981"
        regime_tag = "HEALTHY"

    # Market-accurate TradingView copy string & chart URLs
    tv_runners = ", ".join([str(s) for s in runners_df['tv_symbol'].tolist() if s])
    tv_base_url = "https://www.tradingview.com/chart/?symbol=" if is_us else "https://in.tradingview.com/chart/?symbol="

    display_df = df_bo.copy()
    display_df['TickerLink'] = display_df['tv_symbol'].apply(lambda sym: f"{tv_base_url}{sym}")

    metrics = {
        'scope': 'US' if is_us else 'IN',
        'currency_sym': '$' if is_us else '₹',
        'as_of_date': as_of_date,
        'total_breakouts': total_n,
        'runners_count': runners_n,
        'retest_count': retest_n,
        'failed_count': failed_n,
        'batting_avg': round(batting_avg, 1),
        'failure_rate': round(failure_rate, 1),
        'retest_pct': round(retest_pct, 1),
        'avg_gain': round(avg_gain, 2),
        'avg_loss': round(avg_loss, 2),
        'payoff_ratio': round(payoff_ratio, 2),
        'expectancy_edge': round(expectancy_edge, 2),
        'regime_label': regime_label,
        'regime_desc': regime_desc,
        'regime_color': regime_color,
        'regime_tag': regime_tag,
        'tv_runners_export': tv_runners,
        'eval_window_days': eval_window
    }

    return metrics, display_df


def _empty_radar_state(is_us: bool = False) -> tuple[dict, pd.DataFrame]:
    """Returns an empty state if data cannot be processed."""
    metrics = {
        'scope': 'US' if is_us else 'IN',
        'currency_sym': '$' if is_us else '₹',
        'as_of_date': '',
        'total_breakouts': 0,
        'runners_count': 0,
        'retest_count': 0,
        'failed_count': 0,
        'batting_avg': 0.0,
        'failure_rate': 0.0,
        'retest_pct': 0.0,
        'avg_gain': 0.0,
        'avg_loss': 0.0,
        'payoff_ratio': 0.0,
        'expectancy_edge': 0.0,
        'regime_label': "⚪ RADAR INITIALIZING",
        'regime_desc': "Calculating 15-day rolling breakout cohort metrics...",
        'regime_color': "#94a3b8",
        'regime_tag': "INITIALIZING",
        'tv_runners_export': "",
        'eval_window_days': 15
    }
    return metrics, pd.DataFrame()


# -----------------------------------------------------------------------------
# HIGH-FIDELITY UI RENDERER
# -----------------------------------------------------------------------------

def render_breakout_health_radar_section(is_us: bool = False, key_prefix: str = "breadth_radar"):
    """
    Renders the photorealistic Institutional Breakout Health & Failure Radar widget.
    Designed for inclusion in pages/market_breadth.py (IN) or pages/market_breadth_us.py (US).
    """
    metrics, df_cohort = compute_breakout_health_radar(
        scope="US" if is_us else "IN",
        min_adtv_cr=3.0,
        min_price=5.0 if is_us else 10.0,
        min_adtv_m=2.0,
    )

    if metrics['total_breakouts'] == 0:
        st.info("Breakout Health Radar: Initializing historical price matrix or no qualifying breakouts in trailing 15 days.")
        return

    curr_sym = "$" if is_us else "₹"
    mkt_pill = f"{'🇺🇸 US Equities ($)' if is_us else '🇮🇳 India NSE/BSE (₹)'} • Bar {metrics.get('as_of_date', '')}".strip()

    # Institutional HUD Header & Status Banner
    header_html = f'''
    <div style="
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(2, 6, 23, 0.98) 100%);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-left: 5px solid {metrics['regime_color']};
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                    <span style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.3px;">
                        🚨 Institutional Breakout Health &amp; Distribution Radar
                    </span>
                    <span style="
                        background: rgba(255, 255, 255, 0.06);
                        border: 1px solid rgba(255, 255, 255, 0.12);
                        padding: 3px 10px;
                        border-radius: 6px;
                        font-size: 0.72rem;
                        font-weight: 700;
                        color: #94a3b8;
                    ">Trailing {metrics['eval_window_days']}-Day Cohort</span>
                    <span style="
                        background: rgba(56, 189, 248, 0.10);
                        border: 1px solid rgba(56, 189, 248, 0.28);
                        padding: 3px 10px;
                        border-radius: 6px;
                        font-size: 0.72rem;
                        font-weight: 700;
                        color: #bae6fd;
                    ">{mkt_pill}</span>
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">
                    Mark Minervini Batting Average &amp; True False Breakout Floor Tracker &bull; <b>{metrics['total_breakouts']} Total Breakouts Evaluated</b>
                </div>
            </div>
            <div>
                <span style="
                    background: {metrics['regime_color']}20;
                    border: 1px solid {metrics['regime_color']}60;
                    color: {metrics['regime_color']};
                    padding: 6px 14px;
                    border-radius: 8px;
                    font-weight: 800;
                    font-size: 0.85rem;
                    letter-spacing: 0.5px;
                    display: inline-block;
                ">
                    {metrics['regime_label']}
                </span>
            </div>
        </div>
        <div style="margin-top: 10px; font-size: 0.80rem; color: #cbd5e1; background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 6px; border-left: 2px solid {metrics['regime_color']};">
            💡 <b>Macro Action Directive:</b> {metrics['regime_desc']}
        </div>
    </div>
    '''
    st.html(clean_html(header_html))

    # 4 Metric KPI Cards
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    with col_kpi1:
        st.markdown(clean_html(f'''
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 12px 14px;">
            <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px;">🎯 Batting Avg (Win %)</div>
            <div style="font-size: 1.70rem; font-weight: 900; color: #10b981; margin: 4px 0;">{metrics['batting_avg']:.1f}%</div>
            <div style="font-size: 0.72rem; color: #64748b;">
                <b style="color: #6ee7b7;">{metrics['runners_count']}</b> of {metrics['total_breakouts']} holding above pivot
            </div>
        </div>
        '''), unsafe_allow_html=True)

    with col_kpi2:
        fail_color = "#10b981" if metrics['failure_rate'] < 30 else ("#f59e0b" if metrics['failure_rate'] <= 50 else "#ef4444")
        st.markdown(clean_html(f'''
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid {fail_color}50; border-radius: 10px; padding: 12px 14px;">
            <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px;">🛡️ Base Floor Violation</div>
            <div style="font-size: 1.70rem; font-weight: 900; color: {fail_color}; margin: 4px 0;">{metrics['failure_rate']:.1f}%</div>
            <div style="font-size: 0.72rem; color: #64748b;">
                <b style="color: {fail_color};">{metrics['failed_count']}</b> true floor breaches (&gt;50% alert)
            </div>
        </div>
        '''), unsafe_allow_html=True)

    with col_kpi3:
        st.markdown(clean_html(f'''
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 12px 14px;">
            <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px;">🚀 Avg Follow-Through</div>
            <div style="font-size: 1.70rem; font-weight: 900; color: #38bdf8; margin: 4px 0;">{metrics['avg_gain']:+.2f}%</div>
            <div style="font-size: 0.72rem; color: #64748b;">
                Mean gain of active working runners
            </div>
        </div>
        '''), unsafe_allow_html=True)

    with col_kpi4:
        st.markdown(clean_html(f'''
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(234, 179, 8, 0.3); border-radius: 10px; padding: 12px 14px;">
            <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px;">⚡ Momentum Expectancy</div>
            <div style="font-size: 1.70rem; font-weight: 900; color: #fde047; margin: 4px 0;">{metrics.get('expectancy_edge', 5.1):+.2f}%</div>
            <div style="font-size: 0.72rem; color: #64748b;">
                Payoff: <b>{metrics['payoff_ratio']:.2f}x</b> (7% Stop-Loss Capped)
            </div>
        </div>
        '''), unsafe_allow_html=True)

    # Segmented Visual Progress Bar for Cohort Distribution
    prog_html = f'''
    <div style="margin: 16px 0 12px 0;">
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 700; margin-bottom: 6px;">
            <span style="color: #6ee7b7;">🚀 Active Runners: {metrics['runners_count']} ({metrics['batting_avg']:.1f}%)</span>
            <span style="color: #fde68a;">🟡 Base Retests: {metrics['retest_count']} ({metrics['retest_pct']:.1f}%)</span>
            <span style="color: #f87171;">🔴 Base Floor Violations: {metrics['failed_count']} ({metrics['failure_rate']:.1f}%)</span>
        </div>
        <div style="display: flex; height: 10px; border-radius: 6px; overflow: hidden; background: rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.08);">
            <div style="width: {metrics['batting_avg']}%; background: #10b981;" title="Active Runners ({metrics['batting_avg']:.1f}%)"></div>
            <div style="width: {metrics['retest_pct']}%; background: #f59e0b;" title="Base Retests ({metrics['retest_pct']:.1f}%)"></div>
            <div style="width: {metrics['failure_rate']}%; background: #ef4444;" title="Base Floor Violations ({metrics['failure_rate']:.1f}%)"></div>
        </div>
    </div>
    '''
    st.html(clean_html(prog_html))

    pivot_col = f"Pivot {curr_sym}"
    curr_col = f"Current {curr_sym}"
    floor_col = f"Base Floor {curr_sym}"

    # Rename for user-friendly table headers
    display_rename = {
        'ticker': 'Ticker',
        'breakout_date': 'Breakout Date',
        'pivot_price': pivot_col,
        'current_price': curr_col,
        'base_floor': floor_col,
        'gain_from_pivot_pct': 'Gain from Pivot %',
        'volume_surge': 'Vol Surge',
        'status': 'Status',
        'days_ago': 'Days Held',
        'ind_grade': 'Ind Grade',
        'industry': 'Industry'
    }
    renamed_df = df_cohort.rename(columns=display_rename)

    # Filter Controls for Cohort Table (Status, Top 20% Group & Search)
    f_c1, f_c2, f_c3 = st.columns([1.8, 1.2, 1.0], vertical_alignment="bottom")
    with f_c1:
        status_filter = st.multiselect(
            "Filter Cohort by Status:",
            options=["🚀 Runner", "🟡 Base Retest", "🔴 Failed"],
            default=["🚀 Runner", "🟡 Base Retest"],
            key=f"{key_prefix}_status_filter"
        )
    with f_c2:
        top_20_only = st.checkbox(
            "🏆 Top 20% Group (A+/A) Only",
            value=False,
            key=f"{key_prefix}_top20_filter",
            help="William O'Neil CANSLIM Rule: Filters for breakouts belonging to the top 20% industry groups (Grades A+ and A)."
        )
    with f_c3:
        search_query = st.text_input(
            "Search Ticker / Theme:",
            placeholder="e.g. NVDA or Semi" if is_us else "e.g. TATAPOWER or Drug",
            key=f"{key_prefix}_search"
        ).strip().upper()

    # Row 2: Dual-Handle Gain % Range Slider
    c_gain_slider, c_presets = st.columns([2.5, 1.5], vertical_alignment="center")
    with c_gain_slider:
        gain_range = st.slider(
            "Filter by Gain from Pivot % (Min to Max):",
            min_value=-25.0,
            max_value=50.0,
            value=(-25.0, 50.0),
            step=0.5,
            format="%+.1f%%",
            key=f"{key_prefix}_gain_range",
            help="Dual-handle range filter: Drag lower handle up to find explosive momentum runners (e.g. +5% to +50%), or set 0% to +5% to isolate stocks currently resting in the low-risk buy zone!"
        )
    with c_presets:
        st.markdown(
            f"<div style='font-size: 0.72rem; color: #94a3b8; margin-top: 14px;'>"
            f"Active Range: <b style='color: #38bdf8;'>{gain_range[0]:+.1f}% to {gain_range[1]:+.1f}%</b><br>"
            f"<span style='color: #64748b;'>Set 0% to +5% for Buy Zone consolidations</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    filtered_table = renamed_df[renamed_df['Status'].isin(status_filter)]
    if top_20_only and 'is_top_20' in filtered_table.columns:
        filtered_table = filtered_table[filtered_table['is_top_20'] == True]
    if search_query:
        filtered_table = filtered_table[
            filtered_table['Ticker'].str.contains(search_query, na=False) |
            filtered_table['Industry'].str.upper().str.contains(search_query, na=False)
        ]
    if 'Gain from Pivot %' in filtered_table.columns:
        filtered_table = filtered_table[
            (filtered_table['Gain from Pivot %'] >= gain_range[0]) &
            (filtered_table['Gain from Pivot %'] <= gain_range[1])
        ]

    # Dynamic TradingView Watchlist Export based on filtered table (preserves NYSE:/NASDAQ:/BSE:/NSE:)
    if 'tv_symbol' in filtered_table.columns:
        tv_export_str = ", ".join([str(s) for s in filtered_table['tv_symbol'].tolist() if s])
    else:
        filtered_tickers = filtered_table['Ticker'].tolist()
        tv_export_str = ", ".join([f"{'US' if is_us else 'NSE'}:{t}" for t in filtered_tickers if t])

    with st.expander(f"📋 Copy {len(filtered_table)} Filtered Breakouts for TradingView", expanded=False):
        if tv_export_str:
            st.code(tv_export_str, language="text")
            st.caption("💡 **Tip:** Paste this filtered watchlist into TradingView to chart the exact breakout setups matching your criteria.")
        else:
            st.info("No tickers match the active filters.")

    # Interactive Table
    table_cols = [
        'TickerLink', 'Breakout Date', pivot_col, curr_col,
        floor_col, 'Gain from Pivot %', 'Vol Surge', 'Status',
        'Days Held', 'Ind Grade', 'Industry'
    ]
    valid_cols = [c for c in table_cols if c in filtered_table.columns]

    price_fmt = "$%.2f" if is_us else "₹%.2f"
    column_config = {
        "TickerLink": st.column_config.LinkColumn(
            "Ticker",
            display_text=r"symbol=(?:NSE:|BSE:|NASDAQ:|NYSE:|AMEX:)?(.*)",
            width="small",
        ),
        "Breakout Date": st.column_config.TextColumn("BO Date", width="small"),
        pivot_col: st.column_config.NumberColumn(pivot_col, format=price_fmt),
        curr_col: st.column_config.NumberColumn(curr_col, format=price_fmt),
        floor_col: st.column_config.NumberColumn(floor_col, format=price_fmt, help="Consolidation Low (Base floor). Violation of this level triggers true failure."),
        "Gain from Pivot %": st.column_config.NumberColumn("Gain %", format="%+.2f%%", help="Gain or loss since breaking out above the pivot"),
        "Vol Surge": st.column_config.NumberColumn("Vol Surge", format="%.1fx", help="Volume expansion on breakout day vs 20-day average"),
        "Status": st.column_config.TextColumn("Status", width="medium"),
        "Days Held": st.column_config.NumberColumn("Days", format="%d d"),
        "Ind Grade": st.column_config.TextColumn("Group Grade", width="small", help="IBD Industry Group Letter Grade and Rank"),
        "Industry": st.column_config.TextColumn("Sector / Theme", width="medium"),
    }

    st.dataframe(
        filtered_table[valid_cols],
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=380
    )

    idx_guide_label = "S&amp;P 500 / Nasdaq index corrections" if is_us else "Nifty index corrections"
    # Educational Guide Expander
    with st.expander("📖 Institutional Breakout Health Guide (How Pros Read This)", expanded=False):
        st.markdown(clean_html(f"""
        <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
            <p><b>1. Batting Average (Win Rate):</b> In a healthy market, &gt;60% of breakouts should hold their pivots and expand into profits. When the batting average drops below 40%, Minervini reduces position sizing by 50% and raises cash.</p>
            <p><b>2. Healthy Digestion vs True Failure:</b> When a stock breaks out and pulls back into the base, retail traders falsely call it a 'failed breakout'. As long as the stock holds above the <b>Base Floor (Consolidation Low)</b>, it is merely digesting supply and forming a handle or retest.</p>
            <p><b>3. The Stealth Distribution Signal:</b> True failure occurs ONLY when the base floor is breached. When the <b>Base Floor Violation Rate crosses &gt;50%</b>, it signals that institutions are actively selling into breakout liquidity. This warning routinely precedes major {idx_guide_label} by 1 to 3 weeks.</p>
        </div>
        """), unsafe_allow_html=True)


def get_breakout_health_summary_chip(is_us: bool = False) -> str:
    """
    Returns a compact HTML chip summarizing breakout health for telemetry bars.
    """
    try:
        metrics, _ = compute_breakout_health_radar(scope="US" if is_us else "IN")
        if metrics['total_breakouts'] == 0:
            return ""
        
        tag = metrics.get('regime_tag', 'HEALTHY')
        color = metrics.get('regime_color', '#10b981')
        batting = metrics.get('batting_avg', 0.0)
        failure = metrics.get('failure_rate', 0.0)
        
        chip = f'''
        <div style="
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid {color}50;
            border-left: 3px solid {color};
            border-radius: 6px;
            padding: 3px 10px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.72rem;
            color: #cbd5e1;
        ">
            <span style="font-weight: 800; color: #f8fafc;">⚡ Breakout Health:</span>
            <span style="color: #10b981; font-weight: 700;">{batting:.0f}% Batting</span>
            <span style="color: #64748b;">|</span>
            <span style="color: {color}; font-weight: 700;">{failure:.1f}% Failure</span>
            <span style="background: {color}20; color: {color}; padding: 1px 6px; border-radius: 4px; font-weight: 800; font-size: 0.68rem;">{tag}</span>
        </div>
        '''
        return clean_html(chip)
    except Exception:
        return ""

