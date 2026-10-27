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


def _self_heal_market_breadth_row(
    bar_date_str: str,
    close_core: pd.DataFrame,
    high_core: pd.DataFrame,
    low_core: pd.DataFrame,
    market: str = 'IN'
):
    """
    Ensures market_breadth_daily has today's row so app.py and Beach Ball engines
    never suffer from unvisited-page breadth starvation.
    """
    if close_core is None or close_core.empty or len(close_core) < 55:
        return
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT 1 FROM market_breadth_daily WHERE date = ? AND market = ?", (bar_date_str, market))
        exists = c.fetchone()
        if exists:
            conn.close()
            return

        sma20 = close_core.rolling(20, min_periods=15).mean()
        sma50 = close_core.rolling(50, min_periods=30).mean()
        sma200 = close_core.rolling(200, min_periods=100).mean()
        rets = close_core.pct_change(fill_method=None)

        last_c = close_core.iloc[-1]
        valid_mask = last_c.notna()
        n_valid = int(valid_mask.sum())
        if n_valid < 50:
            conn.close()
            return

        abv20 = float((last_c[valid_mask] > sma20.iloc[-1][valid_mask]).mean() * 100.0)
        abv50 = float((last_c[valid_mask] > sma50.iloc[-1][valid_mask]).mean() * 100.0)
        abv200 = float((last_c[valid_mask] > sma200.iloc[-1][valid_mask]).mean() * 100.0)

        last_r = rets.iloc[-1][valid_mask]
        adv = int((last_r > 0).sum())
        dec = int((last_r < 0).sum())

        roll_hi = high_core.rolling(252, min_periods=60).max().shift(1)
        roll_lo = low_core.rolling(252, min_periods=60).min().shift(1)
        nh = int((high_core.iloc[-1][valid_mask] >= roll_hi.iloc[-1][valid_mask]).sum())
        nl = int((low_core.iloc[-1][valid_mask] <= roll_lo.iloc[-1][valid_mask]).sum())
        nnh = nh - nl

        c.execute('''
            INSERT OR IGNORE INTO market_breadth_daily
            (date, market, above_20_pct, above_50_pct, above_200_pct, advances, declines, new_highs, new_lows, net_new_highs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (bar_date_str, market, round(abv20, 2), round(abv50, 2), round(abv200, 2), adv, dec, nh, nl, nnh))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[BrainEngine] Breadth self-heal warning: {e}")


def compute_tide_turn_state_machine(
    benchmark_df: pd.DataFrame,
    close_core: pd.DataFrame,
    high_core: pd.DataFrame,
    low_core: pd.DataFrame,
    vol_core: pd.DataFrame,
    ironclad_count: int = 0,
    persona_data: dict = None
) -> dict:
    """
    Evaluates the 5-Stage Automated Bear-to-Bull Tide-Turn State Machine on the Liquid Core (Nifty 750)
    and Nifty 500 Benchmark.
    """
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
        'recapture_step': 0,
        'recapture_label': 'Monitoring',
        'ftd_active': False,
        'ftd_suspect_warning': False,
        'ftd_status_text': 'No Active FTD',
        'dd_count': 0,
        'ironclad_count': int(ironclad_count),
        'breakout_health_pct': 50.0,
        'stages_bar': [
            {'num': 1, 'name': '1. Washout', 'sub': '0% Cash'},
            {'num': 2, 'name': '2. Beach Ball Div', 'sub': '0-10% Stalk'},
            {'num': 3, 'name': '3. 20EMA / 9:1 Pilot', 'sub': '15-30% Pilots'},
            {'num': 4, 'name': '4. Validated FTD', 'sub': '50-75% Scale'},
            {'num': 5, 'name': '5. Power Trend', 'sub': '85-100% Full'},
        ]
    }

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

            # Pradeep Bonde 5D Panic (down >= 12% in 5D) & 5D Surge (up >= 12% in 5D)
            if len(close_core) >= 6:
                ret_5d = (close_core.iloc[-1] / close_core.iloc[-6]) - 1.0
                res['panic_5d_count'] = int((ret_5d[valid] <= -0.12).sum())
                res['surge_5d_count'] = int((ret_5d[valid] >= 0.12).sum())

            # Zweig 9-to-1 Up-Volume Thrust (today or in last 5 sessions)
            if vol_core is not None and not vol_core.empty:
                rets_5 = close_core.pct_change(fill_method=None).iloc[-5:]
                vols_5 = vol_core.iloc[-5:]
                up_v_today = float(vols_5.iloc[-1][rets_5.iloc[-1] > 0].sum())
                dn_v_today = float(vols_5.iloc[-1][rets_5.iloc[-1] < 0].sum())
                tot_v_today = up_v_today + dn_v_today
                res['up_vol_ratio_pct'] = round((up_v_today / tot_v_today) * 100.0, 1) if tot_v_today > 0 else 50.0

                # Check if any day in last 5 sessions hit >= 88% up-volume (9:1 thrust)
                for idx_d in range(len(rets_5)):
                    uv = float(vols_5.iloc[idx_d][rets_5.iloc[idx_d] > 0].sum())
                    dv = float(vols_5.iloc[idx_d][rets_5.iloc[idx_d] < 0].sum())
                    if (uv + dv) > 0 and (uv / (uv + dv)) >= 0.88:
                        res['is_9to1_thrust'] = True
                        break

    # Benchmark checks
    idx_above_50 = False
    idx_above_21 = False
    power_trend_aligned = False
    if benchmark_df is not None and not benchmark_df.empty and len(benchmark_df) >= 30:
        b_df = benchmark_df.copy()
        c_col = 'close' if 'close' in b_df.columns else 'Close'
        b_close = pd.to_numeric(b_df[c_col], errors='coerce')
        b_ema10 = b_close.ewm(span=10, adjust=False).mean()
        b_ema21 = b_close.ewm(span=21, adjust=False).mean()
        b_sma50 = b_close.rolling(50, min_periods=25).mean()

        curr_idx = float(b_close.iloc[-1])
        curr_21 = float(b_ema21.iloc[-1])
        curr_50 = float(b_sma50.iloc[-1])
        idx_above_21 = curr_idx >= curr_21
        idx_above_50 = curr_idx >= curr_50
        power_trend_aligned = (float(b_ema10.iloc[-1]) >= curr_21 >= curr_50) and (curr_21 > float(b_ema21.iloc[-5]))

        dd_cnt, dd_list = calculate_distribution_days(b_df)
        res['dd_count'] = int(dd_cnt)

        try:
            recap = detect_20ema_recapture_and_sequence(b_df)
            res['recapture_step'] = int(recap.get('current_step', 0) or 0)
            res['recapture_label'] = str(recap.get('status_label', 'Monitoring'))
        except Exception:
            pass

        try:
            ftd_info = detect_follow_through_day(b_df)
            ftd_det = bool(ftd_info.get('ftd_detected', False))
            res['ftd_active'] = ftd_det
            if ftd_det:
                ftd_dt = str(ftd_info.get('ftd_date', ''))[:10]
                res['ftd_status_text'] = f"Confirmed ({ftd_dt}, +{float(ftd_info.get('ftd_gain_pct', 0.0)):.1f}%)"
        except Exception:
            pass

    persona_score = 50.0
    breakout_health = 55.0
    if persona_data:
        persona_score = float(persona_data.get('composite_score', 50.0) or 50.0)
        breakout_health = float(persona_data.get('breakout_health_pct', 55.0) or 55.0)
    res['breakout_health_pct'] = round(breakout_health, 1)

    # Check Bull-Trap / Suspect FTD (high squat rate -> low breakout health < 45% or heavy DDs >= 5)
    if res['ftd_active'] and (breakout_health < 45.0 or res['dd_count'] >= 5):
        res['ftd_suspect_warning'] = True
        res['ftd_status_text'] += " ⚠️ SUSPECT (High Squat / DD Cluster)"

    # Evaluate 5-Stage State Machine
    if (
        power_trend_aligned
        and res['above_50sma_pct'] >= 55.0
        and res['dd_count'] <= 3
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
    elif (res['ftd_active'] and not res['ftd_suspect_warning']) or (idx_above_50 and res['above_50sma_pct'] >= 45.0 and res['dd_count'] <= 4):
        res.update({
            'stage_num': 4,
            'stage_code': 'VALIDATED_FTD',
            'stage_title': 'Stage 4: Validated Follow-Through / Broadening Uptrend',
            'stage_badge': '🚀 STAGE 4: VALIDATED FTD (PROGRESSIVE BULL)',
            'stage_color': '#34d399',
            'target_exposure': '50% – 75% Equity (0.50%–0.75% Risk)',
            'directive': (
                f"Broadening uptrend validated (Breakout Health {breakout_health:.0f}%, {res['above_50sma_pct']:.0f}% > 50 SMA). "
                "Scale progressively into Top 5 Omni-Confluence Leaders as pilots prove profitable."
            ),
        })
    elif (
        idx_above_21
        or res['recapture_step'] >= 5
        or res['is_9to1_thrust']
        or (res['ftd_active'] and res['ftd_suspect_warning'])
    ):
        warn_prefix = "⚠️ BULL-TRAP FTD GUARDRAIL: " if res['ftd_suspect_warning'] else ""
        res.update({
            'stage_num': 3,
            'stage_code': 'TACTICAL_PILOT',
            'stage_title': 'Stage 3: Tactical Tide-Turn (20-EMA Recapture / 9:1 Pilot Trigger)',
            'stage_badge': '⚡ STAGE 3: TACTICAL PILOT DECK',
            'stage_color': '#f59e0b',
            'target_exposure': '15% – 30% Equity (0.35% Risk Pilots Only)',
            'directive': (
                f"{warn_prefix}Tactical turn active (Up-Vol {res['up_vol_ratio_pct']:.0f}%, {res['ironclad_count']} Ironclad Beach Balls). "
                "Test the waters with 2–3 Pilot positions (5–8% position size, 0.35% NAV risk) on 21-EMA Low / Oops / PEB triggers."
            ),
        })
    elif res['ironclad_count'] >= 10 and res['fomo_5ema_pct'] > 18.0:
        res.update({
            'stage_num': 2,
            'stage_code': 'BEACH_BALL_DIVERGENCE',
            'stage_title': 'Stage 2: Submerged Beach Ball Divergence (Smart Money Accumulating)',
            'stage_badge': '🏖️ STAGE 2: BEACH BALL DIVERGENCE',
            'stage_color': '#38bdf8',
            'target_exposure': '0% – 10% Equity (Stalk List + 1 Test Pilot Max)',
            'directive': (
                f"Index remains below trend ({res['above_50sma_pct']:.0f}% > 50 SMA), BUT {res['ironclad_count']} Ironclad Leaders "
                "are defying the down-days under water! Do NOT buy extended breakouts; stalk 10/21-EMA Low reclaims on the Top 5 Funnel."
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
                f"Heavy washout regime ({res['above_50sma_pct']:.0f}% > 50 SMA, 5D Panic: {res['panic_5d_count']} stocks). "
                "Stay out of new breakouts; let the Artificial Brain track which Ironclad Beach Balls refuse to break."
            ),
        })

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
    top_n: int = 25
) -> tuple:
    """
    Vectorized 7-Pillar Cross-Platform Confluence Scanner across all 2,750+ Indian stocks
    plus 100% of active Portfolio SME/BSE holdings.
    Returns (top_confluence_list, total_ironclad_count, universe_count).
    """
    if close_p is None or close_p.empty or len(close_p) < 30:
        return [], 0, 0

    portfolio_clean_set = set()
    if portfolio_tickers:
        for pt in portfolio_tickers:
            c_pt = str(pt).replace('.NS', '').replace('.BO', '').replace('-SM', '').replace('-ST', '').strip().upper()
            portfolio_clean_set.add(c_pt)
            yf_pt = get_ticker_symbol(str(pt)).replace('.NS', '').replace('.BO', '').strip().upper()
            portfolio_clean_set.add(yf_pt)

    # Filter Indian columns (.NS and .BO) with at least 25 valid bars (using ffill(limit=5) for SME lot-traded stocks)
    ind_cols = [
        c for c in close_p.columns
        if (str(c).endswith('.NS') or str(c).endswith('.BO')) and str(c) != 'GOLDBEES.NS'
    ]
    if not ind_cols:
        return [], 0, 0

    c_raw = close_p[ind_cols].ffill(limit=5)
    last_valid_mask = c_raw.iloc[-1].notna() & (close_p[ind_cols].notna().sum() >= 25)
    valid_cols = last_valid_mask[last_valid_mask].index.tolist()
    if not valid_cols:
        return [], 0, 0

    c_df = c_raw[valid_cols]
    o_df = open_p[valid_cols].ffill(limit=5) if (open_p is not None and not open_p.empty) else c_df
    h_df = high_p[valid_cols].ffill(limit=5) if (high_p is not None and not high_p.empty) else c_df
    l_df = low_p[valid_cols].ffill(limit=5) if (low_p is not None and not low_p.empty) else c_df
    v_df = vol_p[valid_cols].fillna(0.0) if (vol_p is not None and not vol_p.empty) else pd.DataFrame(100000.0, index=c_df.index, columns=valid_cols)

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

    # Moving averages & liquidity
    ema10 = c_df.ewm(span=10, adjust=False).mean().iloc[-1]
    ema21_c = c_df.ewm(span=21, adjust=False).mean().iloc[-1]
    ema21_l = l_df.ewm(span=21, adjust=False).mean().iloc[-1]
    sma50 = c_df.rolling(50, min_periods=25).mean().iloc[-1]
    vol_ma20 = v_df.rolling(20, min_periods=5).mean().iloc[-1].replace(0, 1.0)
    turnover_cr = (curr_close * vol_ma20) / 1e7

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
    is_oops = (curr_open < prev_low) & (curr_close > prev_low) & ((curr_close >= curr_open) | (today_clv >= 0.45))
    # 2. Positive Expectation Breaker (PEB)
    is_peb = (prev_close < prev_open) & (yest_ret <= -1.3) & (curr_close > curr_open) & (curr_close > prev_close)
    # 3. Inside / Double-Inside Day Priming
    is_inside = (curr_high <= prev_high) & (curr_low >= prev_low) & (rel_vol_today <= 0.90) & (dist_52w_pct <= 18.0)
    # 4. Jim Roppel 21-EMA Low Channel Bounce
    dist_to_21l = ((curr_low - ema21_l) / ema21_l.replace(0, np.nan)) * 100.0
    is_21ema_low_bounce = (dist_to_21l >= -1.2) & (dist_to_21l <= 1.5) & (curr_close >= ema21_l) & (curr_close >= curr_open) & (curr_close >= sma50)
    # 5. High-Volume Pocket Pivot / Breakout Thrust
    is_breakout_thrust = (today_ret >= 3.0) & (rel_vol_today >= 1.4) & (today_clv >= 0.65) & (dist_52w_pct <= 12.0)

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
    # -------------------------------------------------------------------------
    ind_lookup = {}
    if industry_map:
        for k, val in industry_map.items():
            clean_k = str(k).replace('.NS', '').replace('.BO', '').strip().upper()
            if val and str(val) not in ['Unknown', 'None', 'nan']:
                ind_lookup[clean_k] = str(val)
    if fundamentals_map:
        for k, f_dict in fundamentals_map.items():
            clean_k = str(k).replace('.NS', '').replace('.BO', '').strip().upper()
            if clean_k not in ind_lookup and isinstance(f_dict, dict):
                ind_val = f_dict.get('industry')
                if ind_val and str(ind_val) not in ['Unknown', 'None', 'nan']:
                    ind_lookup[clean_k] = str(ind_val)

    # Group stocks by industry to find Top 8 RS industries and Sister Tandems
    ind_members = {}
    for col in valid_cols:
        clean_t = str(col).replace('.NS', '').replace('.BO', '').strip().upper()
        ind_name = ind_lookup.get(clean_t, 'General')
        if ind_name not in ['General', 'Unknown', 'Other / ETF']:
            ind_members.setdefault(ind_name, []).append((col, float(rs_pct[col]), float(r_1m[col])))

    ind_scores = {}
    sister_confirmed_tickers = set()
    sister_partner_map = {}
    for ind_name, members in ind_members.items():
        if len(members) >= 2:
            members_sorted = sorted(members, key=lambda x: x[1], reverse=True)
            top_slice = members_sorted[:min(5, len(members_sorted))]
            ind_scores[ind_name] = np.mean([m[1] for m in top_slice])
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

    top_industries = set(
        k for k, _ in sorted(ind_scores.items(), key=lambda item: item[1], reverse=True)[:10]
    )

    # -------------------------------------------------------------------------
    # SCORE & ASSEMBLE CANDIDATES (WITH 100% PORTFOLIO SME EXEMPTION)
    # -------------------------------------------------------------------------
    total_ironclad_count = 0
    candidates = []

    for col in valid_cols:
        clean_t = str(col).replace('.NS', '').replace('.BO', '').strip().upper()
        is_in_portfolio = clean_t in portfolio_clean_set
        t_cr = float(turnover_cr[col]) if pd.notna(turnover_cr[col]) else 0.0

        # Liquidity Gate: >= ₹2.5 Cr for general universe, 100% exempt for active portfolio holdings
        if not is_in_portfolio and t_cr < 2.5:
            continue

        if bool(is_ironclad[col]) and (t_cr >= 2.5 or is_in_portfolio):
            total_ironclad_count += 1

        rs_val = float(rs_pct[col]) if pd.notna(rs_pct[col]) else 50.0
        if rs_val < 70.0 and not is_in_portfolio:
            continue

        c_price = float(curr_close[col])
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
                trigger_name = f"👀 Tight near 21-EMA Low (₹{e21l_val:.1f})"

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

        # Pillar 7: Fundamentals / Earnings Growth (10 pts)
        f_info = (fundamentals_map or {}).get(col) or (fundamentals_map or {}).get(clean_t) or {}
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
            pb_entry = round(e21l_val * 1.03, 1) if e21l_val > 0 else round(c_price * 0.94, 1)
            pb_stop = round(e21l_val * 0.99, 1) if e21l_val > 0 else round(pb_entry * 0.96, 1)
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

        # Regime-adaptive execution plan
        if is_extended_setup:
            exec_plan = (
                f"⏳ EXTENDED (+{ext_above_21l_pct:.1f}% > 21-Low): Stalk Pullback to ₹{trigger_price:,.1f} | "
                f"Stop ₹{stop_price:,.1f} ({risk_pct:.1f}% risk)"
            )
        elif tide_stage_num <= 2:
            exec_plan = f"🛡️ STALK ({trigger_name}): Set Alert @ ₹{trigger_price:,.1f} | Stop ₹{stop_price:,.1f} ({risk_pct:.1f}% risk)"
        elif persona_score < 55.0:
            exec_plan = f"⚡ SWING ({trigger_name}): Buy ₹{trigger_price:,.1f} | Stop ₹{stop_price:,.1f} | Trim 1/2 @ ₹{swing_target_2_5r:,.1f} (+2.5R)"
        else:
            exec_plan = f"🦖 POSITIONAL ({trigger_name}): Buy ₹{trigger_price:,.1f} | Stop ₹{stop_price:,.1f} | Trail 21-EMA Low / 50 SMA"

        candidates.append({
            'ticker': col,
            'clean_ticker': clean_t,
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
            'trigger_price': trigger_price,
            'stop_price': stop_price,
            'risk_pct': round(risk_pct, 2),
            'is_extended_setup': is_extended_setup,
            'ext_above_21l_pct': round(ext_above_21l_pct, 1),
            'target_2_5r': swing_target_2_5r,
            'industry': ind_name,
            'turnover_cr': round(t_cr, 1),
            'badges_str': " • ".join(badges[:5]),
            'execution_plan': exec_plan,
        })

    candidates.sort(key=lambda x: (x['brain_score'], x['rs_score'], x['dd_hit_rate']), reverse=True)
    return candidates[:top_n], total_ironclad_count, universe_count


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


def compute_portfolio_traction_mirror(
    portfolio_df: pd.DataFrame,
    market_data: dict
) -> dict:
    """
    Mark Minervini / Oliver Kell 'Trader's Mirror' (Internal Portfolio Feedback Loop).
    Evaluates:
      1. % of active holdings holding above their 21-EMA Low anti-wick floor.
      2. House-Money Cushion Winners (>= +8% gain & holding spine) vs Broken/Underwater Laggards.
      3. Pilot Traction: Hit rate of the 5 newest / smallest tactical pilot positions.
    """
    if portfolio_df is None or portfolio_df.empty or not market_data:
        return {
            'total_holdings': 0,
            'above_21_low_count': 0,
            'above_21_low_pct': 0.0,
            'green_count': 0,
            'house_money_count': 0,
            'broken_count': 0,
            'pilot_green_count': 0,
            'pilot_total': 0,
            'verdict': 'Neutral',
            'color': '#94a3b8',
        }

    records = []
    for idx_r, row in portfolio_df.iterrows():
        raw_t = str(row.get('ticker', '')).strip()
        if not raw_t or raw_t.upper() in ('LIQUIDBEES', 'LIQUIDCASE'):
            continue
        df = market_data.get(raw_t)
        if df is None or df.empty:
            continue
        try:
            c_col = 'close' if 'close' in df.columns else 'Close'
            l_col = 'low' if 'low' in df.columns else 'Low'
            curr_p = float(df[c_col].iloc[-1])
            avg_buy = float(row.get('Avg Buy Price', curr_p) or curr_p)
            qty = float(row.get('Qty', 0) or 0)
            pos_val = curr_p * qty
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

            records.append({
                'ticker': raw_t,
                'pos_val': pos_val,
                'pnl_pct': pnl_pct,
                'above_21l': above_21l,
                'is_green': is_green,
                'is_hm': is_hm,
                'is_broken': is_broken,
                'row_order': idx_r,
            })
        except Exception:
            continue

    total_h = len(records)
    if total_h == 0:
        return {
            'total_holdings': 0,
            'above_21_low_count': 0,
            'above_21_low_pct': 0.0,
            'green_count': 0,
            'house_money_count': 0,
            'broken_count': 0,
            'pilot_green_count': 0,
            'pilot_total': 0,
            'verdict': 'Neutral',
            'color': '#94a3b8',
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

    return {
        'total_holdings': total_h,
        'above_21_low_count': above_21l_cnt,
        'above_21_low_pct': round(above_21l_pct, 0),
        'green_count': green_cnt,
        'house_money_count': hm_cnt,
        'broken_count': broken_cnt,
        'pilot_green_count': pilot_green_cnt,
        'pilot_total': pilot_total,
        'verdict': verdict,
        'color': color,
    }


def run_artificial_brain_synthesis(
    benchmark_df: pd.DataFrame = None,
    portfolio_tickers: set = None,
    industry_map: dict = None,
    fundamentals_map: dict = None,
    market: str = 'INDIA',
    force_recompute: bool = False
) -> dict:
    """
    Master entrypoint called by app.py.
    1. Checks historical_prices_matrix.pkl max bar date.
    2. If SQLite brain_daily_synthesis_cache matches matrix_bar_date (and not force_recompute),
       returns cached synthesis in <10ms.
    3. Otherwise runs the Dual-Scope Vectorized Pass (<0.85s) across Nifty 750 (Macro Breadth)
       and Full 2,750+ Universe + Portfolio SME/BSE Holdings (7-Pillar Confluence),
       self-heals market_breadth_daily, and caches the result.
    """
    matrix_path = 'historical_prices_matrix.pkl'
    matrix_df = None
    matrix_bar_date = datetime.now().strftime('%Y-%m-%d')

    # Ultra-fast cache hit path (<2ms): if SQLite cache is newer than matrix file mtime on disk, skip pickle load!
    if not force_recompute and os.path.exists(matrix_path):
        try:
            cached_fast = get_brain_synthesis_cache(market=market)
            if cached_fast and cached_fast.get('tide_turn') and cached_fast.get('confluence') is not None:
                upd_str = cached_fast.get('updated_at', '')
                if upd_str and int(cached_fast.get('universe_count', 0)) >= 1500:
                    upd_ts = datetime.strptime(upd_str[:19], '%Y-%m-%d %H:%M:%S').timestamp()
                    mat_mtime = os.path.getmtime(matrix_path)
                    if upd_ts >= mat_mtime:
                        return {
                            'matrix_bar_date': cached_fast.get('matrix_bar_date', matrix_bar_date),
                            'universe_count': int(cached_fast.get('universe_count', 0)),
                            'tide_turn': cached_fast['tide_turn'],
                            'confluence': cached_fast['confluence'],
                            'from_cache': True,
                            'updated_at': upd_str,
                        }
        except Exception:
            pass

    if os.path.exists(matrix_path):
        try:
            with open(matrix_path, 'rb') as f:
                matrix_df = pickle.load(f)
            if isinstance(matrix_df, pd.DataFrame) and not matrix_df.empty:
                matrix_bar_date = matrix_df.index.max().strftime('%Y-%m-%d')
        except Exception as e:
            print(f"[BrainEngine] Matrix load warning: {e}")

    persona_data = get_latest_market_persona(market=market)
    persona_score = float((persona_data or {}).get('composite_score', 50.0) or 50.0)

    # Fast cache hit check (strictly verified against matrix_bar_date)
    if not force_recompute:
        cached = get_brain_synthesis_cache(market=market, required_matrix_bar_date=matrix_bar_date)
        if cached and cached.get('tide_turn') and cached.get('confluence') is not None:
            # Verify cached universe count includes full 2,000+ matrix if matrix is present
            if matrix_df is None or int(cached.get('universe_count', 0)) >= 1500:
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

    close_p, open_p, high_p, low_p, vol_p = _extract_ohlcv_panels(matrix_df)

    # Scope A: Filter Nifty 750 Liquid Core for Macro Breadth & 5-Stage Tide-Turn
    n750_tickers = set(fetch_nifty_total_market_tickers(show_progress=False))
    if close_p is not None and not close_p.empty:
        core_cols = [c for c in close_p.columns if c in n750_tickers]
        if len(core_cols) < 100:
            core_cols = [c for c in close_p.columns if str(c).endswith('.NS')][:750]
        close_core = close_p[core_cols]
        high_core = high_p[core_cols] if (high_p is not None and not high_p.empty) else close_core
        low_core = low_p[core_cols] if (low_p is not None and not low_p.empty) else close_core
        vol_core = vol_p[core_cols] if (vol_p is not None and not vol_p.empty) else None

        # Self-heal market_breadth_daily so app.py never reads stale breadth
        _self_heal_market_breadth_row(matrix_bar_date, close_core, high_core, low_core, market='IN')
    else:
        close_core, high_core, low_core, vol_core = None, None, None, None

    # Scope B: Run 7-Pillar Omni-Confluence Funnel across FULL 2,750+ Universe + All Portfolio Holdings
    confluence_list, ironclad_count, universe_count = compute_omni_confluence_funnel(
        close_p=close_p,
        open_p=open_p,
        high_p=high_p,
        low_p=low_p,
        vol_p=vol_p,
        benchmark_df=benchmark_df,
        portfolio_tickers=portfolio_tickers,
        industry_map=industry_map,
        fundamentals_map=fundamentals_map,
        tide_stage_num=3,
        persona_score=persona_score,
        top_n=25
    )

    # Evaluate 5-Stage Tide-Turn State Machine with live Ironclad Beach Ball count
    tide_turn = compute_tide_turn_state_machine(
        benchmark_df=benchmark_df,
        close_core=close_core,
        high_core=high_core,
        low_core=low_core,
        vol_core=vol_core,
        ironclad_count=ironclad_count,
        persona_data=persona_data
    )

    # Update execution plan labels in confluence_list if tide_turn stage is 1 or 2
    st_num = int(tide_turn.get('stage_num', 3))
    for item in confluence_list:
        if item.get('is_extended_setup'):
            item['execution_plan'] = (
                f"⏳ EXTENDED (+{item.get('ext_above_21l_pct', 0):.1f}% > 21-Low): Stalk Pullback to ₹{item['trigger_price']:,.1f} "
                f"| Stop ₹{item['stop_price']:,.1f} ({item['risk_pct']:.1f}% risk)"
            )
        elif st_num <= 2:
            item['execution_plan'] = (
                f"🛡️ STALK ({item['trigger_name']}): Set Alert @ ₹{item['trigger_price']:,.1f} "
                f"| Stop ₹{item['stop_price']:,.1f} ({item['risk_pct']:.1f}% risk)"
            )

    save_brain_synthesis_cache(
        date_str=datetime.now().strftime('%Y-%m-%d'),
        market=market,
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
