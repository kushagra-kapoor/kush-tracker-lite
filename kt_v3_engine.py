"""
KushTracker v3.0 (KT-CANSLIM) Pure Quantitative Telemetry & Signal Engine.

Ports the full state machine, priority-gated chart markers, 3-Phase Dynamic Stop
ratchet, Van Tharp R-Multiple math, O'Neil FTD / Distribution Day benchmark engine,
Webster IBD Power Trend, Disjoint 3-Wave VCP, Minervini 8-Point Trend Template,
and 16-Row Institutional Cockpit HUD from `tradingview/kush_tracker_v3_0.pine`
without any Streamlit or Plotly runtime coupling.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


_PLAYBOOK_GUIDE: list[dict[str, str]] = [
    {"symbol": "BUY [B1-B2] 🎯", "side": "ENTRY", "title": "Primary Base 1-2 Breakout", "action": "BUY 100% UNIT: Fresh breakout >=1.4x vol. Buy <=5% pivot; -7% initial stop.", "color": "#00c853"},
    {"symbol": "BUY [B3] 🎯", "side": "ENTRY", "title": "Mature Base 3 Breakout", "action": "BUY 50% UNIT: Mature 3rd-stage base. Half size; tighter -5% stop.", "color": "#26a69a"},
    {"symbol": "BUY [B4+ ⚠️] 🎯", "side": "ENTRY", "title": "Late-Stage Base 4+ Breakout", "action": "PILOT 25% ONLY: Late-stage base risk. Trade small; bank fast +10-15%.", "color": "#e65100"},
    {"symbol": "GAP-BUY 🚀", "side": "ENTRY", "title": "Power Gap Breakout", "action": "BUY POWER GAP: Open > pivot >=1.75x vol. Buy now; stop @ Gap Day Low.", "color": "#00e676"},
    {"symbol": "HTF-BUY 🚀🚀", "side": "ENTRY", "title": "High-Tight Flag Power Play", "action": "BUY POWER PLAY: +90% surge <=8w, <=25% flag. Stop @ Flag Low / 10 EMA.", "color": "#00bcd4"},
    {"symbol": "EP 💥", "side": "ENTRY", "title": "Episodic Pivot Catalyst Gap", "action": "BUY CATALYST GAP: >=8% gap on >=3x vol. Stop @ EP Day Low; trail 10/21.", "color": "#76ff03"},
    {"symbol": "WEDGE ⚡", "side": "ENTRY", "title": "Oliver Kell Wedge Pop", "action": "STARTER 25-30%: Early handle break inside base. Stop @ Handle Low.", "color": "#00e5ff"},
    {"symbol": "RETEST 💧", "side": "ENTRY", "title": "Low-Vol Kiss & Go Retest", "action": "ADD PULLBACK: Low-vol <=70% test holding pivot. Enter/add; stop -2% < pivot.", "color": "#4fc3f7"},
    {"symbol": "ADD ➕", "side": "ENTRY", "title": "Pyramid Add-On Entry", "action": "PYRAMID +20-30%: Advancing +2-3.5% > pivot on vol. Add smaller lot.", "color": "#2e7d32"},
    {"symbol": "PP ⚡", "side": "ENTRY", "title": "Morales/Kacher Pocket Pivot", "action": "EARLY ACCUM: >10d max down vol off 10/21/50 MA. Buy starter; stop < MA.", "color": "#00acc1"},
    {"symbol": "3WT 🧱", "side": "ENTRY", "title": "Three-Weeks-Tight Add-On", "action": "ADD TIGHT BREAK: 3 weekly closes <=1.5%. Buy break above 3WT high on vol.", "color": "#fb8c00"},
    {"symbol": "VDU 💧", "side": "ENTRY", "title": "Volume Dry-Up Inside Base", "action": "SET PIVOT ALERT: Supply dead (<=50% avg vol) in base. Stage stop-buy order.", "color": "#1e88e5"},
    {"symbol": "RS-NH 🌟", "side": "ENTRY", "title": "RS 52W High Ahead of Price", "action": "FOCUS LEADER: RS hit 52w high pre-price. Highest odds setup.", "color": "#ffd700"},
    {"symbol": "TENNIS 🎾", "side": "ENTRY", "title": "Tennis Ball Resilience", "action": "TOP #1 LEADER: Holds >=70% of recent market drops. Prioritize size.", "color": "#ffd700"},
    {"symbol": "STOP 🛑", "side": "EXIT", "title": "Dynamic Trailing Stop Hit", "action": "SELL 100% NOW: 3-Phase stop breached. Exit immediately, zero exceptions.", "color": "#c62828"},
    {"symbol": "FAILED BO ❌", "side": "EXIT", "title": "Intraday Squat / Days 1-5 Fail", "action": "EXIT / CUT NOW: Breakout squatted back below pivot on heavy vol.", "color": "#b71c1c"},
    {"symbol": "EXP-BRK ⚠️", "side": "EXIT", "title": "Expectation Breaker (Days 1-5)", "action": "SCRATCH 50-100%: Closed < pivot & BO Day Low on vol. Cut before -7%.", "color": "#e65100"},
    {"symbol": "CLIMAX 🚨", "side": "EXIT", "title": "Climax Top Blow-Off", "action": "SELL 75-100%: Exhaustion blow-off at new high. Lock major gains now.", "color": "#7b1fa2"},
    {"symbol": "EXT-EXHAUST 🚨", "side": "EXIT", "title": "Kell Extension Exhaustion", "action": "TRIM 25-50%: >=3x ATR > 10 EMA on vol. Sell partial; trail tight.", "color": "#aa00ff"},
    {"symbol": "PROFIT 💰", "side": "EXIT", "title": "+20% Standard Profit Target", "action": "BOOK 25-50%: +20% target hit (non-8Wk). Bank partial; trail 21 MA.", "color": "#ffd700"},
    {"symbol": "LPD ⚠️", "side": "EXIT", "title": "Largest Point Drop Warning", "action": "TRIM 33-50%: Largest drop after >=15% run on vol. Tighten stop fast.", "color": "#ff1744"},
    {"symbol": "50d-BRK ⚠️", "side": "EXIT", "title": "Institutional 50-SMA Breakdown", "action": "SELL CORE: 50 SMA broken on >=1.25x vol. Institutional line lost.", "color": "#d32f2f"},
    {"symbol": "21d-WARN ⚡", "side": "EXIT", "title": "Webster 3 Closes < 21 MA", "action": "HALT ADDS: 3 closes < 21 MA. Sell if 3rd bar's low is undercut.", "color": "#f57c00"},
    {"symbol": "STK-DIST ⚠️", "side": "EXIT", "title": "Stock Distribution Cluster", "action": "TIGHTEN STOPS: 3+ heavy down days in 15d. No new buys until VDU.", "color": "#d84315"},
    {"symbol": "STALLING ⚠️", "side": "EXIT", "title": "Wyckoff Churning at Pivot", "action": "DO NOT CHASE: Heavy vol with weak close / narrow range at pivot.", "color": "#ff6f00"},
    {"symbol": "WEDGE TRAP ❌", "side": "EXIT", "title": "Low-Vol Wedging Handle Fail", "action": "PASS / SKIP: Up-wedging low-vol handle failed vol surge (80% fail rate).", "color": "#b71c1c"},
    {"symbol": "BASE RESET 🔄", "side": "EXIT", "title": "Base Cycle Reset to B1", "action": "RESET TO B1: Undercut prior base low or broke 200 SMA. Wait for Stage 2.", "color": "#5c6bc0"},
]


def get_kt_v3_playbook_reference() -> list[dict[str, str]]:
    """Return static 26-signal institutional Actionable Playbook definitions."""
    return [dict(item) for item in _PLAYBOOK_GUIDE]


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_c).abs(),
            (low - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / float(period), adjust=False, min_periods=max(2, period // 2)).mean()


def normalize_circuit_close_pos(
    high_s: pd.Series,
    low_s: pd.Series,
    close_s: pd.Series,
) -> pd.Series:
    """Compute Daily Closing Range (0..1) with NSE upper/lower circuit lock protection."""
    rng = high_s - low_s
    prev_c = close_s.shift(1)
    locked_up = (rng <= 0) & (close_s > prev_c.fillna(close_s))
    raw = (close_s - low_s) / rng.replace(0.0, np.nan)
    raw = raw.clip(lower=0.0, upper=1.0)
    fallback = pd.Series(np.where(locked_up, 1.0, 0.0), index=close_s.index, dtype=float)
    return raw.fillna(fallback)


def _standardize_ohlcv_df(df: pd.DataFrame | None) -> tuple[pd.DataFrame, bool, bool]:
    """Normalize arbitrary OHLCV dataframe into sorted tz-naive Title-Case frame."""
    if df is None or df.empty:
        return pd.DataFrame(), False, False
    work = df.copy()
    col_map = {}
    for c in work.columns:
        low_c = str(c).strip().lower()
        if low_c in ("open", "high", "low", "close", "volume"):
            col_map[c] = low_c.capitalize()
    work = work.rename(columns=col_map)
    if "Close" not in work.columns:
        return pd.DataFrame(), False, False
    work.index = pd.to_datetime(work.index)
    if getattr(work.index, "tz", None) is not None:
        work.index = work.index.tz_localize(None)
    work.index = work.index.normalize()
    work = work[~work.index.duplicated(keep="last")].sort_index()
    work["Close"] = pd.to_numeric(work["Close"], errors="coerce")
    work = work.dropna(subset=["Close"])
    if work.empty:
        return pd.DataFrame(), False, False

    has_ohlc = all(c in work.columns and pd.to_numeric(work[c], errors="coerce").notna().any() for c in ("Open", "High", "Low"))
    if has_ohlc:
        for c in ("Open", "High", "Low"):
            work[c] = pd.to_numeric(work[c], errors="coerce").fillna(work["Close"])
    else:
        work["Open"] = work["Close"]
        work["High"] = work["Close"]
        work["Low"] = work["Close"]

    has_volume = "Volume" in work.columns and (pd.to_numeric(work["Volume"], errors="coerce").fillna(0) > 0).any()
    if "Volume" in work.columns:
        work["Volume"] = pd.to_numeric(work["Volume"], errors="coerce").fillna(0.0)
    else:
        work["Volume"] = 0.0
    return work[["Open", "High", "Low", "Close", "Volume"]], bool(has_ohlc), bool(has_volume)


def evaluate_benchmark_regime_series(
    bench_ohlcv: pd.DataFrame | None,
    vol_proxy_ser: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Evaluate O'Neil FTD, 25-day Expiring Distribution Days, Webster IBD Power Trend,
    and lookback levels cleanly on the benchmark's NATIVE trading calendar.
    """
    work, has_ohlc, has_vol = _standardize_ohlcv_df(bench_ohlcv)
    if work.empty:
        return pd.DataFrame()

    vol_proxy_used = False
    tail60 = work["Volume"].tail(min(60, len(work)))
    zero_frac = float((tail60 <= 0).mean()) if len(tail60) else 1.0
    if (not has_vol or zero_frac > 0.50) and vol_proxy_ser is not None and not vol_proxy_ser.empty:
        vp = pd.to_numeric(vol_proxy_ser, errors="coerce").copy()
        vp.index = pd.to_datetime(vp.index)
        if getattr(vp.index, "tz", None) is not None:
            vp.index = vp.index.tz_localize(None)
        vp.index = vp.index.normalize()
        vp = vp[~vp.index.duplicated(keep="last")].sort_index()
        aligned_vp = vp.reindex(work.index).ffill()
        if (aligned_vp.fillna(0) > 0).any():
            work["Volume"] = np.where(work["Volume"] > 0, work["Volume"], aligned_vp.fillna(0.0))
            has_vol = True
            vol_proxy_used = True

    c = work["Close"]
    h = work["High"]
    l = work["Low"]
    v = work["Volume"]
    c1 = c.shift(1)
    v1 = v.shift(1)

    pct_chg = ((c - c1) / c1.replace(0.0, np.nan)) * 100.0
    pct_chg = pct_chg.fillna(0.0)
    rng = (h - l)
    dcr = normalize_circuit_close_pos(h, l, c)
    if not has_ohlc:
        dcr = pd.Series(np.where(pct_chg > 0.3, 0.70, np.where(pct_chg < -0.3, 0.30, 0.50)), index=work.index)

    e21 = c.ewm(span=21, adjust=False).mean()
    s50 = c.rolling(50, min_periods=20).mean()
    s50_prev5 = s50.shift(5)
    vol_up = (v > v1) if has_vol else (pct_chg.abs() >= 1.25)

    n = len(work)
    c_arr = c.to_numpy(float)
    l_arr = l.to_numpy(float)
    pct_arr = pct_chg.to_numpy(float)
    dcr_arr = dcr.to_numpy(float)
    vol_up_arr = vol_up.to_numpy(bool)
    e21_arr = e21.to_numpy(float)
    s50_arr = s50.to_numpy(float)
    s50_p_arr = s50_prev5.to_numpy(float)

    # Power Trend vectorized runs matching Pine L280-298
    above_21_10d = (pd.Series(l_arr > e21_arr, index=work.index).rolling(10, min_periods=10).sum() == 10).to_numpy(bool)
    e21_gt_50_raw = (e21_arr > s50_arr) & (~np.isnan(s50_arr))
    e21_gt_50_5d = (pd.Series(e21_gt_50_raw, index=work.index).rolling(5, min_periods=5).sum() == 5).to_numpy(bool)
    s50_rising = (~np.isnan(s50_arr)) & (~np.isnan(s50_p_arr)) & (s50_arr > s50_p_arr)

    regime_out = np.empty(n, dtype=object)
    status_out = np.empty(n, dtype=object)
    tl_out = np.empty(n, dtype=object)
    dd_cnt_out = np.zeros(n, dtype=int)
    rally_day_out = np.zeros(n, dtype=int)
    pt_out = np.zeros(n, dtype=bool)
    ftd_wave_out = np.zeros(n, dtype=bool)
    ftd_sig_out = np.zeros(n, dtype=bool)

    bench_regime = "CONFIRMED UPTREND"
    rally_day = 0
    rally_low = np.nan
    last_ftd_bar = -999
    ftd_below_50ma = False
    ftd_low_anchor = np.nan
    dd_bars: list[tuple[int, float]] = []
    power_trend_active = False

    for i in range(n):
        ci = c_arr[i]
        li = l_arr[i]
        pi = pct_arr[i]
        dcri = dcr_arr[i]
        vup = bool(vol_up_arr[i])
        e21i = e21_arr[i]
        s50i = s50_arr[i]
        s50pi = s50_p_arr[i]

        has_s50 = not np.isnan(s50i)
        has_e21 = not np.isnan(e21i)
        bench_under_50ma = has_s50 and ci < (s50i * 0.985)

        if ftd_below_50ma and has_s50 and ci >= s50i:
            ftd_below_50ma = False

        if bench_regime != "IN CORRECTION":
            is_std_dd = (pi <= -0.20) and vup and (dcri < 0.50)
            is_churn_dd = (0.0 <= pi <= 0.40) and vup and (dcri < 0.40) and ((not has_s50) or ci >= s50i)
            if is_std_dd or is_churn_dd:
                dd_bars.append((i, ci))

            # Prune expired by 25 bars or +5% index rally above DD close
            dd_bars = [
                (b_idx, d_pr)
                for (b_idx, d_pr) in dd_bars
                if (i - b_idx) <= 25 and ci < (d_pr * 1.05)
            ]
            active_dds = len(dd_bars)

            ftd_grace_fail = ftd_below_50ma and (
                ((not np.isnan(rally_low)) and li < rally_low)
                or ((not np.isnan(ftd_low_anchor)) and ci < ftd_low_anchor * 0.99)
            )
            s50_cmp = s50pi if not np.isnan(s50pi) else s50i
            norm_50_fail = (not ftd_below_50ma) and (
                bench_under_50ma
                or (has_e21 and has_s50 and ci < e21i and ci < s50i and s50i <= s50_cmp)
            )
            if ftd_grace_fail or norm_50_fail or active_dds >= 6:
                bench_regime = "IN CORRECTION"
                rally_day = 0
                rally_low = li
                ftd_below_50ma = False
                ftd_low_anchor = np.nan
            elif active_dds >= 5 or (active_dds >= 3 and has_e21 and ci < e21i):
                bench_regime = "UNDER PRESSURE"
            else:
                bench_regime = "CONFIRMED UPTREND"
        else:
            rally_low = li if np.isnan(rally_low) else min(rally_low, li)
            if rally_day == 0:
                if pi > 0.0 or dcri >= 0.55:
                    rally_day = 1
                    rally_low = li
            else:
                if li < rally_low:
                    rally_day = 0
                    rally_low = li
                else:
                    rally_day += 1
                    if rally_day >= 4 and pi >= 1.25 and vup and dcri >= 0.50:
                        bench_regime = "CONFIRMED UPTREND"
                        last_ftd_bar = i
                        ftd_sig_out[i] = True
                        ftd_below_50ma = has_s50 and ci < s50i
                        ftd_low_anchor = li
                        rally_day = 0
                        dd_bars.clear()

        if above_21_10d[i] and e21_gt_50_5d[i] and s50_rising[i] and bench_regime == "CONFIRMED UPTREND":
            power_trend_active = True
        if (has_s50 and ci < s50i * 0.995) or (not e21_gt_50_raw[i]) or bench_regime == "IN CORRECTION":
            power_trend_active = False

        active_dds = len(dd_bars)
        is_ftd_wave = (i - last_ftd_bar) <= 20
        if bench_regime == "IN CORRECTION":
            if rally_day > 0:
                mkt_status = f"RALLY ATTEMPT (Day {rally_day}) ⏳"
                tl = "🟡"
            else:
                mkt_status = "IN CORRECTION 🔴 (Defend Cash)"
                tl = "🔴"
        elif bench_regime == "UNDER PRESSURE":
            mkt_status = f"UNDER PRESSURE ({active_dds} DDs) ⚠️"
            tl = "🟡"
        else:
            ftd_tag = " • FTD Wave 🏆" if is_ftd_wave else ""
            mkt_status = f"CONFIRMED UPTREND ({active_dds} DDs){ftd_tag}"
            tl = "🟢"

        regime_out[i] = bench_regime
        status_out[i] = mkt_status
        tl_out[i] = tl
        dd_cnt_out[i] = active_dds
        rally_day_out[i] = rally_day
        pt_out[i] = power_trend_active
        ftd_wave_out[i] = is_ftd_wave

    out = pd.DataFrame(
        {
            "bench_c": c_arr,
            "bench_pct_chg": pct_arr,
            "bench_regime": regime_out,
            "mkt_status_str": status_out,
            "tl_mkt": tl_out,
            "active_dd_count": dd_cnt_out,
            "rally_day": rally_day_out,
            "power_trend_active": pt_out,
            "ftd_wave": ftd_wave_out,
            "ftd_signal": ftd_sig_out,
            "b_1w": c.shift(5).to_numpy(float),
            "b_1m": c.shift(21).to_numpy(float),
            "b_3m": c.shift(63).to_numpy(float),
            "b_6m": c.shift(126).to_numpy(float),
            "b_12m": c.shift(250).to_numpy(float),
            "vol_proxy_used": bool(vol_proxy_used),
        },
        index=work.index,
    )
    return out


def _safe_val(arr: np.ndarray, idx: int, default: float = np.nan) -> float:
    if idx < 0 or idx >= len(arr):
        return default
    v = float(arr[idx])
    return default if np.isnan(v) else v


def compute_kt_v3_telemetry(
    ohlcv: pd.DataFrame,
    bench_ohlcv: pd.DataFrame | None = None,
    vol_proxy_ser: pd.Series | None = None,
    *,
    is_us: bool = False,
    cohort_rs_rating: float = 50.0,
    dist_65ema_pct: float | None = None,
    cap_stage_badge: str = "",
    portfolio_cost_basis: float | None = None,
    use_portfolio_anchor: bool = False,
    trail_ma_type: str = "21 MA",
    use_8wk_rule: bool = True,
    use_smart_stop: bool = True,
    init_stop_pct: float = 7.0,
    breakeven_pct: float = 8.0,
    trailing_pct: float = 15.0,
    buy_zone_pct: float = 5.0,
    profit_t1_pct: float = 20.0,
    profit_t2_pct: float = 25.0,
    pivot_lookback: int = 20,
    max_base_depth: float = 35.0,
    vol_mult: float = 1.4,
    enable_base_reset: bool = True,
    pre_earnings_days: int | None = None,
    bench_label: str | None = None,
) -> dict[str, Any]:
    """
    Run full KushTracker v3.0 technical simulation over OHLCV history and return:
      - `series_df`: bar-by-bar MAs, Dynamic Stop, Pivots, Buy Zone, priority-gated labels
      - `hud_rows`: exact 16-row institutional cockpit metrics matching Pine L1261-1360
      - `live_signals_5d`: trailing 5-bar triggers & live state action badges
      - `header_summary`: scalar cockpit & trade-defense metrics
    """
    stock_df, has_ohlc, has_volume = _standardize_ohlcv_df(ohlcv)
    if stock_df.empty:
        return {"series_df": pd.DataFrame(), "hud_rows": [], "live_signals_5d": [], "header_summary": {}}

    n = len(stock_df)
    idx = stock_df.index
    o_s = stock_df["Open"]
    h_s = stock_df["High"]
    l_s = stock_df["Low"]
    c_s = stock_df["Close"]
    v_s = stock_df["Volume"]

    # Indicators & MAs (Pine L100-147)
    ema10 = c_s.ewm(span=10, adjust=False, min_periods=5).mean()
    ema21 = c_s.ewm(span=21, adjust=False, min_periods=10).mean()
    sma21 = c_s.rolling(21, min_periods=10).mean()
    sma50 = c_s.rolling(50, min_periods=20).mean()
    ema65 = c_s.ewm(span=65, adjust=False, min_periods=35).mean()
    sma150 = c_s.rolling(150, min_periods=120).mean()
    sma200 = c_s.rolling(200, min_periods=160).mean()
    vol_sma50 = v_s.rolling(50, min_periods=10).mean()
    vol_ma50_prev = vol_sma50.shift(1).fillna(vol_sma50)
    atr14 = _wilder_atr(h_s, l_s, c_s, 14)
    atr20 = _wilder_atr(h_s, l_s, c_s, 20)

    c_prev = c_s.shift(1)
    pct_chg = ((c_s - c_prev) / c_prev.replace(0.0, np.nan) * 100.0).fillna(0.0)
    range_day = (h_s - l_s).clip(lower=0.0)
    close_pos = normalize_circuit_close_pos(h_s, l_s, c_s)
    dcr_pct = close_pos * 100.0
    vol_ratio = (v_s / vol_ma50_prev.replace(0.0, np.nan)).fillna(1.0 if not has_volume else 0.0)
    natr_s = ((atr14 / c_s.replace(0.0, np.nan)) * 100.0).fillna(0.0)

    # Split / corrupt price spike detection (breadth_engine heuristic)
    low_vol_spike = ((pct_chg.abs() > 85.0) & (vol_ratio < 2.0)).iloc[1:] if n > 1 else pd.Series(dtype=bool)
    has_split_warning = bool(low_vol_spike.any())

    # Liquidity Tier (Pine L140-146)
    avg_turn_raw = np.where(vol_sma50.notna(), vol_sma50 * c_s, v_s * c_s)
    turn_val_s = pd.Series(avg_turn_raw / (1e6 if is_us else 1e7), index=idx).fillna(0.0)

    # Native Benchmark Regime Alignment (Fix Gap #4: bench metrics computed on native calendar first!)
    bench_native = evaluate_benchmark_regime_series(bench_ohlcv, vol_proxy_ser=vol_proxy_ser)
    has_bench = not bench_native.empty
    if has_bench:
        bench_al = bench_native.reindex(idx, method="ffill")
        bench_al["bench_regime"] = bench_al["bench_regime"].ffill().fillna("CONFIRMED UPTREND")
        bench_al["mkt_status_str"] = bench_al["mkt_status_str"].ffill().fillna("CONFIRMED UPTREND")
        bench_al["tl_mkt"] = bench_al["tl_mkt"].ffill().fillna("🟢")
        bench_al["power_trend_active"] = bench_al["power_trend_active"].fillna(False).astype(bool)
        bench_al["active_dd_count"] = bench_al["active_dd_count"].fillna(0).astype(int)
        bench_al["bench_pct_chg"] = bench_al["bench_pct_chg"].fillna(0.0)
    else:
        bench_al = pd.DataFrame(
            {
                "bench_c": np.nan,
                "bench_pct_chg": 0.0,
                "bench_regime": "CONFIRMED UPTREND",
                "mkt_status_str": "NO BENCHMARK FEED (Stock-Only)",
                "tl_mkt": "⚪",
                "active_dd_count": 0,
                "rally_day": 0,
                "power_trend_active": False,
                "b_1w": np.nan,
                "b_1m": np.nan,
                "b_3m": np.nan,
                "b_6m": np.nan,
                "b_12m": np.nan,
                "vol_proxy_used": False,
            },
            index=idx,
        )

    bench_c_s = pd.to_numeric(bench_al["bench_c"], errors="coerce")
    rs_raw = np.where(bench_c_s.notna() & (bench_c_s > 0), c_s / bench_c_s, 1.0)
    rs_raw_s = pd.Series(rs_raw, index=idx)
    rs_sma21 = rs_raw_s.rolling(21, min_periods=10).mean()
    # Normalized RS line scaled comfortably for chart overlay (base 100 at visible/series start)
    first_valid_rs = float(rs_raw_s.dropna().iloc[0]) if not rs_raw_s.dropna().empty else 1.0
    rs_norm_s = (rs_raw_s / (first_valid_rs if first_valid_rs > 0 else 1.0)) * 100.0

    # Alpha & Rolling Historical Pine RS Proxy (strictly used on historical bars i < n - 1)
    c_1w, c_1m, c_3m, c_6m = c_s.shift(5), c_s.shift(21), c_s.shift(63), c_s.shift(126)
    b_1w, b_1m, b_3m, b_6m = (bench_al[col] for col in ("b_1w", "b_1m", "b_3m", "b_6m"))
    has_1w = (c_1w > 0) & (b_1w > 0)
    has_1m = (c_1m > 0) & (b_1m > 0)
    has_3m = (c_3m > 0) & (b_3m > 0)
    has_6m = (c_6m > 0) & (b_6m > 0)

    stk_1w = np.where(has_1w, (c_s - c_1w) / c_1w * 100.0, pct_chg)
    stk_1m = np.where(has_1m, (c_s - c_1m) / c_1m * 100.0, 0.0)
    stk_3m = np.where(has_3m, (c_s - c_3m) / c_3m * 100.0, 0.0)
    stk_6m = np.where(has_6m, (c_s - c_6m) / c_6m * 100.0, 0.0)

    bnch_1w = np.where(has_1w, (bench_c_s - b_1w) / b_1w * 100.0, bench_al["bench_pct_chg"])
    bnch_1m = np.where(has_1m, (bench_c_s - b_1m) / b_1m * 100.0, 0.0)
    bnch_3m = np.where(has_3m, (bench_c_s - b_3m) / b_3m * 100.0, 0.0)
    bnch_6m = np.where(has_6m, (bench_c_s - b_6m) / b_6m * 100.0, 0.0)

    a_1w = stk_1w - bnch_1w
    a_1m = np.where(has_1m, stk_1m - bnch_1m, a_1w)
    a_3m = np.where(has_3m, stk_3m - bnch_3m, a_1m)
    a_6m = np.where(has_6m, stk_6m - bnch_6m, a_3m)
    comp_alpha = (0.40 * a_1m) + (0.40 * a_3m) + (0.20 * a_6m)
    z_sc = (comp_alpha - 3.0) / 18.0
    hist_rs_proxy = np.clip(np.round((1.0 / (1.0 + np.exp(-1.702 * z_sc))) * 100.0), 1, 99).astype(float)
    # Lock canonical Screener Percentile RS on latest bar (user mandate!)
    canonical_rs = float(np.clip(round(float(cohort_rs_rating or 50.0)), 1, 99))

    # 52W & RS rolling highs
    rs_prev_yr = rs_raw_s.shift(1).rolling(250, min_periods=20).max()
    rs_all_max = rs_raw_s.shift(1).expanding(min_periods=20).max()
    rs_prev_high = rs_prev_yr.fillna(rs_all_max)

    high_52w = h_s.rolling(250, min_periods=1).max()
    low_52w = l_s.rolling(250, min_periods=1).min()

    # Pre-compute rolling arrays for O(N) loop
    highest_prev_20 = h_s.shift(1).rolling(pivot_lookback, min_periods=5).max()
    lowest_prev_20 = l_s.shift(1).rolling(pivot_lookback, min_periods=5).min()
    lowest_prior_60 = l_s.shift(20).rolling(60, min_periods=15).min()
    highest_prev_50 = h_s.shift(1).rolling(50, min_periods=15).max()
    highest_5d = h_s.shift(1).rolling(5, min_periods=3).max()
    vol_ma5_prev = v_s.shift(1).rolling(5, min_periods=3).mean()
    vol_ma15 = v_s.rolling(15, min_periods=5).mean()
    vcp_vol_10 = v_s.rolling(10, min_periods=5).mean()
    vcp_vol_30 = v_s.rolling(30, min_periods=10).mean()
    max_drop_prev10 = range_day.shift(1).rolling(10, min_periods=3).max()

    # Exact Pine Disjoint 3-Wave VCP (L599-614): w1=[25..36], w2=[12..24], w3=[1..11]
    w1_h = h_s.shift(25).rolling(12, min_periods=8).max()
    w1_l = l_s.shift(25).rolling(12, min_periods=8).min()
    w2_h = h_s.shift(12).rolling(13, min_periods=8).max()
    w2_l = l_s.shift(12).rolling(13, min_periods=8).min()
    w3_h = h_s.shift(1).rolling(11, min_periods=6).max()
    w3_l = l_s.shift(1).rolling(11, min_periods=6).min()

    depth_t1 = ((w1_h - w1_l) / w1_h.replace(0.0, np.nan) * 100.0).fillna(0.0)
    depth_t2 = ((w2_h - w2_l) / w2_h.replace(0.0, np.nan) * 100.0).fillna(0.0)
    depth_t3 = ((w3_h - w3_l) / w3_h.replace(0.0, np.nan) * 100.0).fillna(0.0)
    vcp_contracting_s = (
        (depth_t1 > depth_t2)
        & (depth_t2 > depth_t3)
        & (depth_t3 > 0)
        & (depth_t3 <= 12.0)
        & (w3_l >= w2_l * 0.985)
    )
    is_vcp_confirmed_s = vcp_contracting_s & (vcp_vol_10 < vcp_vol_30) & sma50.notna() & (c_s > sma50)

    # High-Tight Flag pre-computed window (Pine L572-577)
    low_flagpole = l_s.shift(10).rolling(45, min_periods=15).min()
    flag_h15 = h_s.shift(1).rolling(15, min_periods=10).max()
    flag_l15 = l_s.shift(1).rolling(15, min_periods=10).min()

    # Beat Bench, Tennis Ball, U/D Ratio, Ants, Stock DDs
    beat_bench = ((bench_c_s.shift(1).notna()) & (pct_chg > bench_al["bench_pct_chg"])).astype(float)
    rs_persist_20d_s = (beat_bench.rolling(20, min_periods=5).sum() / 20.0) * 100.0

    is_bench_down = (bench_al["bench_pct_chg"] <= -0.35) & has_bench
    stock_resilient = is_bench_down & ((c_s > c_prev) | ((close_pos >= 0.65) & (v_s < vol_ma50_prev)))
    bench_down_cnt_20_s = is_bench_down.astype(float).rolling(20, min_periods=1).sum()
    s_res_sum_20_s = stock_resilient.astype(float).rolling(20, min_periods=1).sum()

    up_vol_s = np.where(c_s > c_prev, v_s, 0.0)
    down_vol_s = np.where(c_s < c_prev, v_s, 0.0)
    sum_up_v = pd.Series(up_vol_s, index=idx).rolling(50, min_periods=10).sum()
    sum_dn_v = pd.Series(down_vol_s, index=idx).rolling(50, min_periods=10).sum()
    ud_ratio_s = np.where(sum_dn_v > 0, sum_up_v / sum_dn_v, 1.0)

    up_days_15_s = (c_s > c_prev).astype(float).rolling(15, min_periods=10).sum()
    gain_15d_s = ((c_s - c_s.shift(15)) / c_s.shift(15).replace(0.0, np.nan) * 100.0).fillna(0.0)
    is_ryan_ants_s = (up_days_15_s >= 12) & (gain_15d_s >= 15.0) & (vol_ma15 >= vol_ma50_prev * 1.15) & sma50.notna() & (c_s > sma50)

    stk_dd_s = (pct_chg <= -0.20) & (v_s > vol_ma50_prev) & (close_pos < 0.40)
    stk_dd_cnt_15_s = stk_dd_s.astype(float).rolling(15, min_periods=1).sum()

    # Pocket Pivot max down vol in prev 10 bars (L855-857)
    down_vol_prev = pd.Series(np.where(c_prev < c_s.shift(2), v_s.shift(1), 0.0), index=idx)
    max_dn_vol_10 = down_vol_prev.rolling(10, min_periods=3).max()
    fb_down_vol = np.where(max_dn_vol_10.isna() | (max_dn_vol_10 <= 0), vol_ma50_prev, max_dn_vol_10)

    # ISO Calendar Week change for zero-lookahead 3WT (Fix Gap #14)
    iso = idx.isocalendar()
    week_ids = (iso.year.to_numpy(int) * 100) + iso.week.to_numpy(int)

    # Arrays for fast chronological loop
    o_a = o_s.to_numpy(float)
    h_a = h_s.to_numpy(float)
    l_a = l_s.to_numpy(float)
    c_a = c_s.to_numpy(float)
    c1_a = c_prev.to_numpy(float)
    v_a = v_s.to_numpy(float)
    pct_a = pct_chg.to_numpy(float)
    rng_a = range_day.to_numpy(float)
    cp_a = close_pos.to_numpy(float)
    vr_a = vol_ratio.to_numpy(float)

    e10_a = ema10.to_numpy(float)
    e21_a = ema21.to_numpy(float)
    s21_a = sma21.to_numpy(float)
    s50_a = sma50.to_numpy(float)
    s150_a = sma150.to_numpy(float)
    s200_a = sma200.to_numpy(float)
    v50p_a = vol_ma50_prev.to_numpy(float)
    v5p_a = vol_ma5_prev.to_numpy(float)
    atr14_a = atr14.to_numpy(float)
    atr20_a = atr20.to_numpy(float)
    fb_dv_a = pd.Series(fb_down_vol, index=idx).to_numpy(float)

    h20_a = highest_prev_20.to_numpy(float)
    l20_a = lowest_prev_20.to_numpy(float)
    lp60_a = lowest_prior_60.to_numpy(float)
    h50_a = highest_prev_50.to_numpy(float)
    h5_a = highest_5d.to_numpy(float)
    md10_a = max_drop_prev10.to_numpy(float)
    lfp_a = low_flagpole.to_numpy(float)
    fh15_a = flag_h15.to_numpy(float)
    fl15_a = flag_l15.to_numpy(float)
    h52_a = high_52w.to_numpy(float)
    l52_a = low_52w.to_numpy(float)
    rs_a = rs_raw_s.to_numpy(float)
    rs_ph_a = rs_prev_high.to_numpy(float)
    bdown20_a = bench_down_cnt_20_s.to_numpy(float)
    sres20_a = s_res_sum_20_s.to_numpy(float)
    stk_dd_a = stk_dd_s.to_numpy(bool)
    stk_dd_cnt_a = stk_dd_cnt_15_s.to_numpy(float)
    mkt_reg_a = bench_al["bench_regime"].to_numpy(object)
    vcp_conf_a = is_vcp_confirmed_s.to_numpy(bool)

    # Output series arrays
    dyn_stop_out = np.full(n, np.nan)
    stop_phase_out = np.zeros(n, dtype=int)  # 0=flat, 1=hard, 2=BE, 3=trail, 4=8wk
    act_piv_out = np.full(n, np.nan)
    prosp_piv_out = np.full(n, np.nan)
    bz_up_out = np.full(n, np.nan)
    t20_out = np.full(n, np.nan)
    in_pos_out = np.zeros(n, dtype=bool)
    stopped_out_arr = np.zeros(n, dtype=bool)
    base_cnt_out = np.ones(n, dtype=int)

    below_lbl = [""] * n
    below_col = [""] * n
    below_rsn = [""] * n
    above_lbl = [""] * n
    above_col = [""] * n
    above_rsn = [""] * n
    vol_role = ["neutral"] * n

    # Rolling ring for recent 5-bar signal tracking
    sig_history: dict[str, list[bool]] = {k: [False] * n for k in (
        "bo", "gap", "htf", "ep", "wedge", "retest", "add", "pp", "vdu", "3wt",
        "rsnh", "tennis", "stop", "fbo", "exp", "clmx", "ext", "prof", "lpd",
        "50brk", "21w", "stkdd", "stall", "wtrap", "reset",
    )}

    # State machine variables (Pine L473-497)
    is_in_position = False
    is_stopped_out = False
    stop_bar_index = -999
    active_pivot_price = np.nan
    last_base_pivot = np.nan
    buy_zone_upper = np.nan
    hard_stop_price = np.nan
    target_20_price = np.nan
    target_25_price = np.nan
    dynamic_stop = np.nan
    pivot_bar = -999
    base_count = 1
    current_base_low = np.nan
    prev_base_low = np.nan
    bars_since_prev_bo = 999
    has_added_pos = False
    profit_taken = False
    exp_breaker_fired = False
    base_was_reset = False
    last_reset_bar = -999
    highest_since_bo = np.nan
    eight_week_start = -999
    eight_week_active = False
    eight_week_abort = False
    eight_week_done = False

    last_3wt_bar = -999
    last_stall_bar = -999
    last_fbo_bar = -999
    last_stkdd_bar = -999
    last_rsnh_bar = -999
    last_tennis_bar = -999
    last_pp_bar = -999
    last_vdu_bar = -999
    last_lpd_was = False
    last_ext_was = False
    last_vdu_was = False
    last_stall_was = False
    last_tennis_was = False
    last_rs_ahead_prev = False
    consec_below_21 = 0
    stop_phase_str = f"Phase 1: -{init_stop_pct:.1f}% Hard Stop 🛑"
    last_bo_type_str = ""
    completed_weeks: list[float] = []

    final_state: dict[str, Any] = {}

    for i in range(n):
        ci = c_a[i]
        oi = o_a[i]
        hi = h_a[i]
        li = l_a[i]
        vi = v_a[i]
        c1i = c1_a[i]
        pii = pct_a[i]
        cpi = cp_a[i]
        vri = vr_a[i]
        rngi = rng_a[i]

        e10i = e10_a[i]
        ma21i = s21_a[i] if "SMA" in trail_ma_type.upper() else e21_a[i]
        e21i = e21_a[i]
        s50i = s50_a[i]
        s150i = s150_a[i]
        s200i = s200_a[i]
        v50pi = v50p_a[i]
        atr14i = atr14_a[i]
        atr20i = atr20_a[i]

        has_s50 = not np.isnan(s50i)
        has_s200 = not np.isnan(s200i)
        is_young_ipo = not has_s200
        h20i = h20_a[i]
        l20i = l20_a[i]
        h52i = h52_a[i]
        l52i = l52_a[i]

        rs_bar = canonical_rs if (i == n - 1 or not has_bench) else float(hist_rs_proxy[i])
        s200_prev20 = _safe_val(s200_a, i - 20)
        s50_prev20 = _safe_val(s50_a, i - 20)
        h20_prev10 = _safe_val(h20_a, i - 10)

        # Minervini 8-Point Score (Pine L453-468)
        c_gt_150_200 = (not np.isnan(s150i)) and has_s200 and ci > s150i and ci > s200i
        s150_gt_200 = (not np.isnan(s150i)) and has_s200 and s150i > s200i
        s200_rising = has_s200 and (not np.isnan(s200_prev20)) and s200i > s200_prev20
        s50_gt_all = has_s50 and (not np.isnan(s150i)) and has_s200 and s50i > s150i and s50i > s200i
        c_gt_50 = has_s50 and ci > s50i
        c_above_low = (l52i > 0) and (((ci - l52i) / l52i) * 100.0 >= 30.0)
        c_near_high = (h52i > 0) and (ci >= h52i * 0.75)
        rs_qual = rs_bar >= 70.0
        minervini_score = sum((c_gt_150_200, s150_gt_200, s200_rising, s50_gt_all, c_gt_50, c_above_low, c_near_high, rs_qual))
        ipo_stage2_ok = (
            (ci > s50i and (not np.isnan(ma21i) and s50i > ma21i) and c_above_low and c_near_high and rs_bar >= 70.0)
            if has_s50
            else ((not np.isnan(ma21i) and ci > ma21i) and (not np.isnan(e10i) and ci > e10i) and rs_bar >= 75.0)
        )
        minervini_template = ipo_stage2_ok if is_young_ipo else (minervini_score == 8)

        # Base validity & stepping-base (Pine L501-517)
        base_depth_pct = ((h20i - l20i) / h20i * 100.0) if (not np.isnan(h20i) and h20i > 0 and not np.isnan(l20i)) else 0.0
        is_valid_base = 4.0 <= base_depth_pct <= max_base_depth
        pivot_flat_enough = (
            (not np.isnan(h20i))
            and (not np.isnan(h20_prev10))
            and (h20i <= h20_prev10 * 1.02)
            and (not np.isnan(c1i))
            and (c1i <= h20i)
        )
        is_stepping_base = is_in_position and (bars_since_prev_bo >= 25) and pivot_flat_enough and is_valid_base
        can_breakout_now = (not is_in_position) or is_stepping_base

        if (not is_in_position or is_stepping_base) and is_valid_base and has_s50 and ci >= s50i * 0.90 and not np.isnan(l20i):
            current_base_low = l20i if np.isnan(current_base_low) else min(current_base_low, li)

        lp60i = lp60_a[i]
        prior_run_pct = ((h20i - lp60i) / lp60i * 100.0) if (not np.isnan(lp60i) and lp60i > 0 and not np.isnan(h20i)) else 0.0
        stage2_trend_ok = (
            has_s50
            and ci > s50i
            and c_near_high
            and rs_bar >= 70.0
            and (
                (not np.isnan(ma21i) and (ci > ma21i or s50i > ma21i))
                if is_young_ipo
                else (has_s200 and ci > s200i and s50i > s200i and s200i >= (s200_prev20 if not np.isnan(s200_prev20) else s200i) * 0.998)
            )
        )
        has_prior_uptrend = stage2_trend_ok and (prior_run_pct >= 20.0 or minervini_template or (l52i > 0 and ci >= l52i * 1.25))

        # Dormant Reset & Base Reset (Pine L520-558)
        s50_cmp_reset = s50_prev20 if not np.isnan(s50_prev20) else s50i
        is_dormant_reset = (
            (base_count > 1)
            and (bars_since_prev_bo >= 95)
            and (not is_young_ipo)
            and has_s200
            and has_s50
            and (s50i <= s200i * 1.02)
            and (s50i <= s50_cmp_reset)
        )
        if is_dormant_reset:
            base_count = 1
            base_was_reset = True
            prev_base_low = l20i
            current_base_low = l20i

        undercut_prior = (not np.isnan(prev_base_low)) and (ci < prev_base_low * 0.985) and (base_count > 1 or is_in_position)
        deep_stage4 = (not is_young_ipo) and has_s200 and has_s50 and (ci < s200i) and (s50i < s200i)
        mkt_bear_reset = (mkt_reg_a[i] == "IN CORRECTION") and has_s50 and (ci < s50i * 0.92)
        c1_s200_prev = _safe_val(s200_a, i - 1)
        ma200_crossunder = has_s200 and (not np.isnan(c1_s200_prev)) and (not np.isnan(c1i)) and (ci < s200i) and (c1i >= c1_s200_prev) and (base_count > 1 or is_in_position)
        deep_corr_reset = (not np.isnan(last_base_pivot)) and (ci < last_base_pivot * 0.80)
        needs_base_reset = enable_base_reset and (not is_young_ipo) and (undercut_prior or deep_stage4 or mkt_bear_reset or ma200_crossunder or deep_corr_reset)
        is_base_reset_event = False
        if needs_base_reset and (not np.isnan(last_base_pivot) or base_count > 1 or is_in_position):
            if (i - last_reset_bar) >= 20:
                is_base_reset_event = True
                last_reset_bar = i
            base_count = 1
            base_was_reset = True
            prev_base_low = current_base_low if not np.isnan(current_base_low) else l20i
            current_base_low = np.nan
            active_pivot_price = np.nan
            last_base_pivot = np.nan
            is_in_position = False
            is_stopped_out = False

        # Wedging Handle Trap, HTF, EP, Kell Wedge Pop, Stalling (Pine L560-597)
        rising_lows_3d = (
            i >= 4
            and l_a[i - 1] > l_a[i - 2]
            and l_a[i - 2] > l_a[i - 3]
            and l_a[i - 3] > l_a[i - 4]
        )
        v5pi = v5p_a[i]
        handle_vol_dec = (not np.isnan(v5pi)) and (not np.isnan(v50pi)) and (v5pi < v50pi * 0.85) and (i >= 1 and v_a[i - 1] < v50pi)
        is_wedging_handle = rising_lows_3d and (i >= 4 and c_a[i - 1] > c_a[i - 4]) and handle_vol_dec and has_s50 and ci > s50i and base_depth_pct > 6.0
        is_power_clearance = (vri >= 1.75) and (cpi >= 0.65)
        is_pivot_cross = (not np.isnan(h20i)) and (not np.isnan(c1i)) and (ci > h20i) and (c1i <= h20i)
        is_wedge_trap_event = is_pivot_cross and is_wedging_handle and (not is_power_clearance) and can_breakout_now

        lfpi, fh15i, fl15i = lfp_a[i], fh15_a[i], fl15_a[i]
        flag_depth = ((fh15i - fl15i) / fh15i * 100.0) if (not np.isnan(fh15i) and fh15i > 0 and not np.isnan(fl15i)) else 0.0
        htf_run_pct = ((fh15i - lfpi) / lfpi * 100.0) if (not np.isnan(lfpi) and lfpi > 0 and not np.isnan(fh15i)) else 0.0
        is_htf_candidate = (htf_run_pct >= 90.0) and (4.0 <= flag_depth <= 25.0) and stage2_trend_ok
        is_htf_breakout = (
            ((not is_in_position) or bars_since_prev_bo >= 15)
            and is_htf_candidate
            and (not np.isnan(fh15i))
            and ci > fh15i
            and (not np.isnan(c1i) and c1i <= fh15i)
            and (not np.isnan(v50pi) and vi >= v50pi * 1.50)
            and ci > oi
            and cpi >= 0.50
        )

        is_ep_gap = (
            (not np.isnan(c1i))
            and oi >= (c1i * 1.08)
            and (not np.isnan(v50pi) and vi >= v50pi * 3.0)
            and (ci >= oi)
            and (cpi >= 0.50)
            and has_s50
            and ci > s50i
        )
        ep_ref = h20i if not np.isnan(h20i) else (c1i if not np.isnan(c1i) else oi)
        is_ep_breakout = ((not is_in_position) or bars_since_prev_bo >= 12) and is_ep_gap and (oi >= ep_ref * 0.99) and (ci > ep_ref)

        h5i = h5_a[i]
        is_wedge_pop = (
            (not is_in_position)
            and has_s50
            and ci > s50i
            and (not np.isnan(ma21i) and ci > ma21i)
            and is_valid_base
            and (not np.isnan(h5i))
            and ci > h5i
            and (not np.isnan(c1i) and c1i <= h5i)
            and (not np.isnan(h20i) and h20i * 0.94 <= ci < h20i)
            and (not np.isnan(v50pi) and vi > v50pi * 1.20)
            and ci > oi
        )

        # Wyckoff Stalling (guarded against NSE locked upper-circuit bars where H == L & pct >= 4%)
        is_locked_uc = (hi <= li) and (pii >= 4.0)
        stall_heavy = (not np.isnan(v50pi)) and (vi >= v50pi * 1.30)
        stall_narrow = (not np.isnan(atr20i)) and (rngi < atr20i * 0.65)
        stall_weak = cpi < 0.45
        is_stalling = (
            (not is_locked_uc)
            and (not np.isnan(h20i))
            and (h20i * 0.99 <= ci <= h20i * 1.04)
            and stall_heavy
            and (stall_narrow or stall_weak)
            and (not is_wedge_pop)
            and (not is_ep_breakout)
        )
        is_stall_event = is_stalling and (not last_stall_was) and ((i - last_stall_bar) >= 8)
        last_stall_was = is_stalling
        if is_stall_event:
            last_stall_bar = i

        # Primary Breakout Triggers (Pine L616-678)
        vol_surge = (not np.isnan(v50pi)) and v50pi > 0 and (vi >= v50pi * vol_mult)
        wedge_block = is_wedging_handle and (not is_power_clearance)
        is_breakout_raw = (
            can_breakout_now
            and bars_since_prev_bo >= 15
            and (not np.isnan(h20i))
            and ci > h20i
            and (not np.isnan(c1i) and c1i <= h20i)
            and vol_surge
            and ci > oi
            and cpi >= 0.50
            and is_valid_base
            and has_prior_uptrend
            and (not wedge_block)
        )
        is_gap_breakout = (
            can_breakout_now
            and bars_since_prev_bo >= 15
            and (not np.isnan(h20i))
            and oi >= h20i * 1.005
            and pii >= 2.0
            and (not np.isnan(v50pi) and v50pi > 0 and vi >= v50pi * max(1.75, vol_mult))
            and ci > oi
            and cpi >= 0.50
            and is_valid_base
            and has_prior_uptrend
            and (not is_ep_breakout)
        )
        is_reg_breakout = is_breakout_raw and (not is_gap_breakout) and (not is_ep_breakout) and (not is_htf_breakout)
        is_breakout_bar = is_reg_breakout or is_gap_breakout or is_htf_breakout or is_ep_breakout

        if is_breakout_bar:
            is_turnaround_ep = is_ep_breakout and (not is_young_ipo) and has_s200 and (ci < s200i)
            if is_turnaround_ep:
                base_count = 1
                base_was_reset = True
            elif base_was_reset:
                base_count = 1
                base_was_reset = False
            else:
                had_base_run = (not np.isnan(last_base_pivot)) and (bars_since_prev_bo >= 22) and (
                    ((not np.isnan(highest_since_bo)) and highest_since_bo >= last_base_pivot * 1.18)
                    or (ci >= last_base_pivot * 1.10 and bars_since_prev_bo >= 25)
                )
                if had_base_run:
                    base_count = min(4, base_count + 1)

            prev_base_low = current_base_low if not np.isnan(current_base_low) else l20i
            current_base_low = l20i
            if is_ep_breakout or is_gap_breakout:
                active_pivot_price = oi
            elif is_htf_breakout and (not is_breakout_raw):
                active_pivot_price = fh15i
            else:
                active_pivot_price = h20i
            last_base_pivot = active_pivot_price if (np.isnan(last_base_pivot) or base_count == 1) else max(last_base_pivot, active_pivot_price)

            pct_stop_floor = active_pivot_price * (1.0 - (init_stop_pct / 100.0))
            if is_ep_breakout or is_gap_breakout:
                hard_stop_price = max(li * 0.995, pct_stop_floor)
                last_bo_type_str = "EP 💥" if is_ep_breakout else "GAP-BUY 🚀"
            elif is_htf_breakout and (not is_breakout_raw):
                fl_ref = fl15i if not np.isnan(fl15i) else pct_stop_floor
                hard_stop_price = max(fl_ref, pct_stop_floor)
                last_bo_type_str = "HTF-BUY 🚀🚀"
            else:
                hard_stop_price = pct_stop_floor
                last_bo_type_str = f"BUY [B{base_count}] 🎯"

            highest_since_bo = hi
            is_in_position = True
            is_stopped_out = False
            stop_bar_index = -999
            pivot_bar = i
            buy_zone_upper = active_pivot_price * (1.0 + (buy_zone_pct / 100.0))
            target_20_price = active_pivot_price * (1.0 + (profit_t1_pct / 100.0))
            target_25_price = active_pivot_price * (1.0 + (profit_t2_pct / 100.0))
            dynamic_stop = hard_stop_price
            bars_since_prev_bo = 0
            has_added_pos = False
            profit_taken = False
            exp_breaker_fired = False
            eight_week_active = False
            eight_week_abort = False
            eight_week_done = False
            eight_week_start = -999
        else:
            bars_since_prev_bo += 1

        if is_in_position or (not np.isnan(last_base_pivot)):
            highest_since_bo = hi if np.isnan(highest_since_bo) else max(highest_since_bo, hi)

        # Post-BO Re-tests, Pyramid Adds & Failed BO (Pine L684-704)
        is_pivot_retest = (
            is_in_position
            and (not is_breakout_bar)
            and (not np.isnan(active_pivot_price))
            and (2 <= (i - pivot_bar) <= 10)
            and (active_pivot_price * 0.985 <= li <= active_pivot_price * 1.02)
            and (ci >= active_pivot_price)
            and (not np.isnan(v50pi) and vi <= v50pi * 0.70)
            and cpi >= 0.50
        )
        is_add_on_bar = (
            is_in_position
            and (not is_breakout_bar)
            and (not has_added_pos)
            and (not np.isnan(active_pivot_price))
            and (active_pivot_price * 1.02 < ci <= active_pivot_price * 1.035)
            and (ci >= (highest_since_bo if not np.isnan(highest_since_bo) else hi) * 0.985)
            and (not np.isnan(v50pi) and vi > v50pi)
            and ci > oi
            and cpi >= 0.50
        )
        if is_add_on_bar:
            has_added_pos = True

        is_intraday_bo_squat = (
            (not is_breakout_bar)
            and can_breakout_now
            and bars_since_prev_bo >= 15
            and is_valid_base
            and pivot_flat_enough
            and has_prior_uptrend
            and (not np.isnan(h20i))
            and (i >= 1 and h_a[i - 1] < h20i)
            and (hi >= h20i * 1.004)
            and (ci < h20i)
            and (not np.isnan(v50pi) and vi >= v50pi * 1.20)
            and cpi <= 0.35
            and ci < oi
        )
        is_post_bo_fail = (
            is_in_position
            and (not is_breakout_bar)
            and (1 <= (i - pivot_bar) <= 5)
            and (not np.isnan(active_pivot_price))
            and (ci < active_pivot_price * 0.985)
            and (not np.isnan(v50pi) and vi >= v50pi * 1.25)
            and cpi <= 0.35
            and ci < oi
        )
        is_failed_bo = (is_intraday_bo_squat or is_post_bo_fail) and ((i - last_fbo_bar) >= 10)
        if is_failed_bo:
            last_fbo_bar = i
            if is_in_position:
                is_in_position = False
                is_stopped_out = True
                stop_bar_index = i
                current_base_low = np.nan
                active_pivot_price = np.nan

        # Non-lookahead Weekly 3WT via ISO Week Boundary (Pine L705-723)
        is_3wt = False
        if i > 0 and week_ids[i] != week_ids[i - 1]:
            completed_weeks.append(c_a[i - 1])
            if len(completed_weeks) >= 3 and (i - last_3wt_bar) >= 12:
                w3_slice = completed_weeks[-3:]
                max_wc, min_wc = max(w3_slice), min(w3_slice)
                if (
                    min_wc > 0
                    and ((max_wc - min_wc) / min_wc) * 100.0 <= 1.50
                    and has_s50
                    and ci > s50i
                    and (is_young_ipo or ci > s200i)
                    and ci >= min_wc * 0.985
                    and (is_in_position or has_prior_uptrend)
                ):
                    is_3wt = True
                    last_3wt_bar = i

        # Position Defense, 8-Wk Hold, 3-Phase Dynamic Stop Ratchet (Pine L732-808)
        days_since_bo = (i - pivot_bar) if (is_in_position and pivot_bar >= 0) else 0
        max_gain_reached = ((highest_since_bo - active_pivot_price) / active_pivot_price * 100.0) if (is_in_position and not np.isnan(active_pivot_price) and active_pivot_price > 0 and not np.isnan(highest_since_bo)) else 0.0
        gain_from_pivot = ((ci - active_pivot_price) / active_pivot_price * 100.0) if (is_in_position and not np.isnan(active_pivot_price) and active_pivot_price > 0) else 0.0

        if use_8wk_rule and is_in_position and (not eight_week_active) and (not eight_week_abort) and (not eight_week_done) and (days_since_bo <= 15) and (max_gain_reached >= 20.0):
            eight_week_active = True
            eight_week_start = pivot_bar

        if eight_week_active:
            held_bars = i - eight_week_start
            if has_s50 and ci < (s50i * 0.98) and (not np.isnan(v50pi) and vi > v50pi * 1.40):
                eight_week_active = False
                eight_week_abort = True
            elif held_bars >= 40:
                eight_week_active = False
                eight_week_done = True

        is_profit_target_hit = (
            is_in_position
            and (not eight_week_active)
            and (not profit_taken)
            and (not np.isnan(target_20_price))
            and (ci >= target_20_price or hi >= target_20_price)
        )
        if is_profit_target_hit:
            profit_taken = True

        phase_code = 0
        if is_in_position and not np.isnan(active_pivot_price):
            if eight_week_active:
                ew_floor = max(hard_stop_price if not np.isnan(hard_stop_price) else s50i * 0.98, s50i * 0.98) if has_s50 else (hard_stop_price if not np.isnan(hard_stop_price) else active_pivot_price * 0.93)
                dynamic_stop = ew_floor
                stop_phase_str = "8-Wk Defense (50 SMA) 🏆"
                phase_code = 4
            elif max_gain_reached >= trailing_pct:
                p3_line = s50i if "50" in trail_ma_type else ma21i
                trail_floor = max(p3_line if not np.isnan(p3_line) else active_pivot_price, active_pivot_price)
                dynamic_stop = trail_floor if np.isnan(dynamic_stop) else max(dynamic_stop, trail_floor)
                ma_above_piv = (not np.isnan(p3_line)) and (p3_line > active_pivot_price)
                if ma_above_piv:
                    stop_phase_str = "Phase 3: 50 SMA Trail 🟠" if "50" in trail_ma_type else "Phase 3: 21 MA Trail 🟡"
                else:
                    stop_phase_str = "Phase 3: Breakeven Floor 🛡️"
                phase_code = 3
            elif max_gain_reached >= breakeven_pct:
                dynamic_stop = active_pivot_price if np.isnan(dynamic_stop) else max(dynamic_stop, active_pivot_price)
                stop_phase_str = "Phase 2: Breakeven (0%) 🛡️"
                phase_code = 2
            else:
                dynamic_stop = hard_stop_price
                stop_phase_str = f"Phase 1: -{init_stop_pct:.1f}% Hard Stop 🛑"
                phase_code = 1

        # Smart Stop Shakeout & Exit (Pine L785-806)
        is_phase3_trail = is_in_position and (not eight_week_active) and (max_gain_reached >= trailing_pct)
        prev_dyn = dyn_stop_out[i - 1] if i > 0 else np.nan
        prev_below_stop = (i > 0) and (not np.isnan(prev_dyn)) and (c_a[i - 1] < prev_dyn)
        phase3_smart_hit = (not np.isnan(dynamic_stop)) and (ci < dynamic_stop) and (
            cpi < 0.45 or (not np.isnan(v50pi) and vi > v50pi) or prev_below_stop or li < dynamic_stop * 0.975
        )
        stop_hit_cond = phase3_smart_hit if (is_phase3_trail and use_smart_stop) else ((not np.isnan(dynamic_stop)) and (ci < dynamic_stop or li < dynamic_stop * 0.985))
        is_stop_event = is_in_position and (not is_breakout_bar) and (not np.isnan(dynamic_stop)) and (i > pivot_bar) and stop_hit_cond
        if is_stop_event:
            is_in_position = False
            is_stopped_out = True
            eight_week_active = False
            current_base_low = np.nan
            active_pivot_price = np.nan
            stop_bar_index = i
            phase_code = 0

        if is_stopped_out and (not is_in_position) and (not is_breakout_bar):
            if ((i - stop_bar_index) >= 5) or (is_valid_base and has_s50 and ci >= s50i):
                is_stopped_out = False

        # Warnings & Secondary Signals (Pine L813-873)
        if is_in_position and (not np.isnan(ma21i)) and ci < ma21i:
            consec_below_21 += 1
        else:
            consec_below_21 = 0
        is_21ma_warn = is_in_position and (not eight_week_active) and (consec_below_21 == 3)

        bo_day_low = l_a[pivot_bar] if (is_in_position and 1 <= days_since_bo <= 5 and pivot_bar >= 0) else np.nan
        is_exp_breaker = (
            is_in_position
            and (not exp_breaker_fired)
            and (1 <= days_since_bo <= 5)
            and (not np.isnan(active_pivot_price))
            and (ci <= active_pivot_price * 0.99)
            and (np.isnan(bo_day_low) or ci < bo_day_low)
            and ((not np.isnan(v50pi) and vi >= v50pi * 1.10) or (pii <= -2.5 and cpi <= 0.35))
            and ci < oi
            and (not is_failed_bo)
        )
        if is_exp_breaker:
            exp_breaker_fired = True

        md10i = md10_a[i]
        is_lpd = (
            is_in_position
            and max_gain_reached >= 15.0
            and (not np.isnan(md10i))
            and (rngi > md10i * 1.30)
            and (not np.isnan(c1i) and ci < c1i)
            and pii <= -2.5
            and (not np.isnan(v50pi) and vi > v50pi * 1.25)
            and cpi < 0.45
        )
        is_lpd_edge = is_lpd and (not last_lpd_was)
        last_lpd_was = is_lpd

        c1_s50 = _safe_val(s50_a, i - 1)
        is_50ma_break = (
            has_s50
            and (not np.isnan(c1_s50))
            and (not np.isnan(c1i))
            and (ci < s50i)
            and (c1i >= c1_s50)
            and pii <= -1.2
            and (not np.isnan(v50pi) and vi > v50pi * 1.25)
            and ci < oi
        )
        if is_in_position and (not eight_week_active) and is_50ma_break:
            is_in_position = False
            is_stopped_out = True
            stop_bar_index = i
            current_base_low = np.nan
            active_pivot_price = np.nan

        # Climax Top & Kell Exhaustion (Pine L841-850)
        dist_50 = ((ci - s50i) / s50i * 100.0) if (has_s50 and s50i > 0) else 0.0
        dist_200 = ((ci - s200i) / s200i * 100.0) if (has_s200 and s200i > 0) else 0.0
        atr_dist_10 = ((ci - e10i) / atr14i) if ((not np.isnan(e10i)) and (not np.isnan(atr14i)) and atr14i > 0) else 0.0
        h50i = h50_a[i]
        at_50d_high = (not np.isnan(h50i)) and (hi >= h50i or (i > 0 and h_a[i - 1] >= h50i))
        had_clmx_run = (dist_50 >= 25.0) and (dist_200 >= 50.0 or max_gain_reached >= 40.0 or (is_young_ipo and dist_50 >= 28.75))
        is_climax_rev = (
            at_50d_high
            and had_clmx_run
            and (not np.isnan(atr14i))
            and (rngi >= atr14i * 1.8)
            and (not np.isnan(v50pi) and vi >= v50pi * 2.0)
            and cpi <= 0.45
            and (not is_htf_breakout)
            and (not is_ep_breakout)
        )
        is_ext_exhaust = (
            is_in_position
            and gain_from_pivot >= 20.0
            and (atr_dist_10 >= 3.0)
            and (not np.isnan(v50pi) and vi > v50pi * 1.50)
            and ci < hi
            and ci > oi
        )
        is_ext_exhaust_edge = is_ext_exhaust and (not last_ext_was)
        last_ext_was = is_ext_exhaust

        # Pocket Pivot & VDU (Pine L854-873)
        pp_base = has_s50 and (li <= s50i * 1.02) and (s50i * 0.995 <= ci <= s50i * 1.06)
        pp_cont = has_s50 and (ci > s50i) and (
            ((not np.isnan(e10i)) and li <= e10i * 1.02 and e10i <= ci <= e10i * 1.06)
            or ((not np.isnan(ma21i)) and li <= ma21i * 1.02 and ma21i <= ci <= ma21i * 1.06)
        )
        pp_stage_ok = is_young_ipo or (has_s200 and ci > s200i)
        is_pp_raw = (
            (not np.isnan(c1i) and ci > c1i)
            and (not np.isnan(fb_dv_a[i]) and vi > fb_dv_a[i])
            and (not np.isnan(v50pi) and vi >= v50pi * 0.85)
            and (pp_base or pp_cont)
            and pp_stage_ok
            and ci > oi
            and cpi >= 0.50
        )
        is_pocket_pivot = is_pp_raw and ((i - last_pp_bar) >= 5)
        if is_pocket_pivot:
            last_pp_bar = i

        is_vdu = (
            has_volume
            and (not np.isnan(v50pi))
            and (vi <= v50pi * 0.50)
            and (not np.isnan(atr14i) and rngi < atr14i * 0.70)
            and has_s50
            and (ci >= s50i * 0.95)
            and pp_stage_ok
            and (not np.isnan(h20i) and hi < h20i)
            and ((abs(pii) <= 1.75) or cpi >= 0.40)
        )
        is_vdu_start = is_vdu and (not last_vdu_was) and ((i - last_vdu_bar) >= 7)
        last_vdu_was = is_vdu
        if is_vdu_start:
            last_vdu_bar = i

        # RS-NH & Tennis Ball events
        rs_ph_i = rs_ph_a[i]
        is_rs_new_high_raw = i >= 20 and (not np.isnan(rs_ph_i)) and (rs_a[i] > rs_ph_i) and has_s50 and (ci > s50i) and has_bench
        is_rs_ahead_raw = is_rs_new_high_raw and (hi < h52i * 0.97) and (not last_rs_ahead_prev)
        last_rs_ahead_prev = is_rs_new_high_raw
        is_rs_ahead_event = is_rs_ahead_raw and ((i - last_rsnh_bar) >= 10)
        if is_rs_ahead_event:
            last_rsnh_bar = i

        bd20 = int(bdown20_a[i])
        sr20 = int(sres20_a[i])
        tennis_ratio = (float(sr20) / float(bd20) * 100.0) if (bd20 >= 3) else np.nan
        is_tennis_now = (not np.isnan(tennis_ratio)) and (tennis_ratio >= 70.0)
        tennis_stage_ok = has_s50 and ci > s50i and (is_young_ipo or (has_s200 and ci > s200i))
        is_tennis_first = is_tennis_now and (not last_tennis_was) and tennis_stage_ok and ((i - last_tennis_bar) >= 15)
        last_tennis_was = is_tennis_now
        if is_tennis_first:
            last_tennis_bar = i

        # Individual stock DD cluster
        stkdd_stage_ok = has_s50 and (ci >= s50i * 0.92) and (is_young_ipo or (has_s200 and ci >= s200i * 0.95))
        is_stkdd_event = bool(stk_dd_a[i]) and (stk_dd_cnt_a[i] >= 3) and stkdd_stage_ok and ((i - last_stkdd_bar) >= 10)
        if is_stkdd_event:
            last_stkdd_bar = i

        # Zero-Collision Priority Gate (<=1 below & <=1 above per bar, Pine L948-983)
        if is_ep_breakout:
            below_lbl[i], below_col[i], below_rsn[i] = "EP 💥", "#76ff03", f"Episodic Pivot gap >=8% on {vri:.1f}x vol"
        elif is_htf_breakout and (not is_reg_breakout) and (not is_gap_breakout):
            below_lbl[i], below_col[i], below_rsn[i] = "HTF-BUY 🚀🚀", "#00bcd4", f"High-Tight Flag (+{htf_run_pct:.0f}% pole, {flag_depth:.1f}% flag)"
        elif is_gap_breakout:
            below_lbl[i], below_col[i], below_rsn[i] = f"GAP [B{base_count}] 🚀", "#00e676", f"Power Gap above pivot on {vri:.1f}x vol"
        elif is_reg_breakout:
            bcol = "#00c853" if base_count <= 2 else ("#26a69a" if base_count == 3 else "#e65100")
            below_lbl[i], below_col[i], below_rsn[i] = f"BUY [B{base_count}] 🎯", bcol, f"Base {base_count} Pivot Breakout on {vri:.1f}x vol"
        elif is_add_on_bar:
            below_lbl[i], below_col[i], below_rsn[i] = f"ADD [B{base_count}] ➕", "#2e7d32", "Pyramid +2-3.5% above pivot on volume"
        elif is_pivot_retest:
            below_lbl[i], below_col[i], below_rsn[i] = "RETEST 💧", "#4fc3f7", "Low-vol (<=70% avg) kiss holding active pivot"
        elif is_wedge_pop:
            below_lbl[i], below_col[i], below_rsn[i] = "WEDGE ⚡", "#00e5ff", "Oliver Kell early 5D handle trendline pop"
        elif is_pocket_pivot:
            below_lbl[i], below_col[i], below_rsn[i] = "PP ⚡", "#00acc1", "Pocket Pivot off 10/21/50 MA support"
        elif is_rs_ahead_event:
            below_lbl[i], below_col[i], below_rsn[i] = "RS-NH 🌟", "#ffd700", "RS line hit 52W high ahead of price"
        elif is_vdu_start:
            below_lbl[i], below_col[i], below_rsn[i] = "VDU 💧", "#1e88e5", "Volume Dry-Up <=50% avg vol inside tight range"
        elif is_tennis_first:
            below_lbl[i], below_col[i], below_rsn[i] = "TENNIS 🎾", "#ffd700", f"Resilient on {sr20}/{bd20} index drops ({tennis_ratio:.0f}%)"

        if is_stop_event:
            above_lbl[i], above_col[i], above_rsn[i] = "STOP 🛑", "#c62828", "Dynamic Trailing Stop breached — 100% exit rule"
        elif is_failed_bo:
            above_lbl[i], above_col[i], above_rsn[i] = "FAILED BO ❌", "#b71c1c", "Pivot squat / Days 1-5 heavy-volume reversal"
        elif is_climax_rev:
            above_lbl[i], above_col[i], above_rsn[i] = "CLIMAX 🚨", "#7b1fa2", "Climax exhaustion wide-spread reversal at new highs"
        elif is_50ma_break:
            above_lbl[i], above_col[i], above_rsn[i] = "50d-BRK ⚠️", "#d32f2f", "50 SMA broken on >=1.25x heavy institutional selling"
        elif is_ext_exhaust_edge:
            above_lbl[i], above_col[i], above_rsn[i] = "EXT-EXH 🚨", "#aa00ff", f"Kell Exhaustion >=3 ATR above 10 EMA ({atr_dist_10:.1f} ATR)"
        elif is_lpd_edge:
            above_lbl[i], above_col[i], above_rsn[i] = "LPD ⚠️", "#ff1744", "Largest Point Drop (>1.3x 10D max range) on heavy vol"
        elif is_exp_breaker:
            above_lbl[i], above_col[i], above_rsn[i] = "EXP-BRK ⚠️", "#e65100", "Expectation Breaker: < pivot & BO Day Low in Days 1-5"
        elif is_profit_target_hit:
            above_lbl[i], above_col[i], above_rsn[i] = "PROFIT 💰", "#ffd700", f"+{profit_t1_pct:.0f}% standard profit target achieved"
        elif is_base_reset_event:
            above_lbl[i], above_col[i], above_rsn[i] = "RESET 🔄", "#5c6bc0", "Base Count reset to B1 (undercut / 200 SMA breach)"
        elif is_wedge_trap_event:
            above_lbl[i], above_col[i], above_rsn[i] = "W-TRAP ❌", "#b71c1c", "Up-wedging low-vol handle failed power clearance"
        elif is_stall_event:
            above_lbl[i], above_col[i], above_rsn[i] = "STALL ⚠️", "#ff6f00", "Wyckoff churning at pivot (heavy vol, narrow/weak close)"
        elif is_21ma_warn:
            above_lbl[i], above_col[i], above_rsn[i] = "21d-WARN ⚡", "#f57c00", "3 consecutive closes below 21 MA"
        elif is_stkdd_event:
            above_lbl[i], above_col[i], above_rsn[i] = "STK-DIST ⚠️", "#d84315", "3+ heavy distribution bars in last 15 sessions"
        elif is_3wt:
            above_lbl[i], above_col[i], above_rsn[i] = "3WT 🧱", "#fb8c00", "3 completed weekly closes within <=1.5%"

        # Volume conviction role
        if is_breakout_bar or (ci >= c1i and vri >= 1.40 and cpi >= 0.55):
            vol_role[i] = "accum"
        elif is_vdu:
            vol_role[i] = "vdu"
        elif (ci < c1i and vri >= 1.15 and cpi <= 0.45) or stk_dd_a[i]:
            vol_role[i] = "dist"

        # Record series outputs & 5-bar signal flags
        dyn_stop_out[i] = dynamic_stop if (is_in_position and not is_stopped_out) else np.nan
        stop_phase_out[i] = phase_code
        act_piv_out[i] = active_pivot_price if (is_in_position and not is_stopped_out) else np.nan
        prosp_piv_out[i] = h20i if (not is_in_position) else np.nan
        bz_up_out[i] = buy_zone_upper if (is_in_position and not is_stopped_out) else np.nan
        t20_out[i] = target_20_price if (is_in_position and not is_stopped_out) else np.nan
        in_pos_out[i] = is_in_position
        stopped_out_arr[i] = is_stopped_out
        base_cnt_out[i] = base_count

        sig_history["bo"][i] = is_reg_breakout
        sig_history["gap"][i] = is_gap_breakout
        sig_history["htf"][i] = is_htf_breakout
        sig_history["ep"][i] = is_ep_breakout
        sig_history["wedge"][i] = is_wedge_pop
        sig_history["retest"][i] = is_pivot_retest
        sig_history["add"][i] = is_add_on_bar
        sig_history["pp"][i] = is_pocket_pivot
        sig_history["vdu"][i] = is_vdu_start
        sig_history["3wt"][i] = is_3wt
        sig_history["rsnh"][i] = is_rs_ahead_event
        sig_history["tennis"][i] = is_tennis_first
        sig_history["stop"][i] = is_stop_event
        sig_history["fbo"][i] = is_failed_bo
        sig_history["exp"][i] = is_exp_breaker
        sig_history["clmx"][i] = is_climax_rev
        sig_history["ext"][i] = is_ext_exhaust_edge
        sig_history["prof"][i] = is_profit_target_hit
        sig_history["lpd"][i] = is_lpd_edge
        sig_history["50brk"][i] = is_50ma_break
        sig_history["21w"][i] = is_21ma_warn
        sig_history["stkdd"][i] = is_stkdd_event
        sig_history["stall"][i] = is_stall_event
        sig_history["wtrap"][i] = is_wedge_trap_event
        sig_history["reset"][i] = is_base_reset_event

        if i == n - 1:
            final_state = {
                "is_young_ipo": is_young_ipo,
                "minervini_score": minervini_score,
                "minervini_template": minervini_template,
                "stage2_trend_ok": stage2_trend_ok,
                "base_depth_pct": base_depth_pct,
                "is_valid_base": is_valid_base,
                "dist_50": dist_50,
                "dist_200": dist_200,
                "atr_dist_10": atr_dist_10,
                "dist_10": ((ci - e10i) / e10i * 100.0) if (not np.isnan(e10i) and e10i > 0) else 0.0,
                "tennis_pts": sr20,
                "bench_down_cnt": bd20,
                "tennis_ratio": tennis_ratio,
                "is_tennis_now": is_tennis_now,
                "is_rs_ahead_now": is_rs_new_high_raw and (hi < h52i * 0.97),
                "is_rs_at_high": i >= 20 and (not np.isnan(rs_ph_i)) and (rs_a[i] >= rs_ph_i * 0.998),
                "is_breakout_bar": is_breakout_bar,
                "is_pocket_pivot": is_pocket_pivot,
                "is_vdu": is_vdu,
                "is_climax_rev": is_climax_rev,
                "is_ext_exhaust": is_ext_exhaust,
                "is_profit_target_hit": is_profit_target_hit,
                "is_exp_breaker": is_exp_breaker,
                "is_stop_event": is_stop_event,
                "last_bo_type_str": last_bo_type_str,
                "highest_prev_20": h20i,
                "days_since_bo": days_since_bo,
            }

    # Build enriched series_df
    series_df = stock_df.copy()
    series_df["EMA10"] = ema10
    series_df["EMA21"] = ema21
    series_df["SMA50"] = sma50
    series_df["EMA65"] = ema65
    series_df["SMA150"] = sma150
    series_df["SMA200"] = sma200
    series_df["Vol_SMA50"] = vol_sma50
    series_df["ATR14"] = atr14
    series_df["dynamic_stop"] = dyn_stop_out
    series_df["stop_phase"] = stop_phase_out
    series_df["active_pivot"] = act_piv_out
    series_df["prospective_pivot"] = prosp_piv_out
    series_df["buy_zone_upper"] = bz_up_out
    series_df["target_20"] = t20_out
    series_df["is_in_position"] = in_pos_out
    series_df["is_stopped_out"] = stopped_out_arr
    series_df["base_count"] = base_cnt_out
    series_df["is_3wt_bull"] = sig_history["3wt"]
    series_df["rs_norm"] = rs_norm_s
    series_df["below_label"] = below_lbl
    series_df["below_color"] = below_col
    series_df["below_reason"] = below_rsn
    series_df["above_label"] = above_lbl
    series_df["above_color"] = above_col
    series_df["above_reason"] = above_rsn
    series_df["vol_role"] = vol_role

    # Final-bar synthesis & optional Portfolio Cost-Basis anchor (Fix Gap #12)
    last_c = float(c_a[-1])
    last_s50 = float(s50_a[-1]) if not np.isnan(s50_a[-1]) else np.nan
    last_s200 = float(s200_a[-1]) if not np.isnan(s200_a[-1]) else np.nan
    s200_p20_end = _safe_val(s200_a, n - 21)
    is_young_end = final_state["is_young_ipo"]
    is_stage4_now = (not is_young_end) and (not np.isnan(last_s200)) and (last_c < last_s200)
    is_below_50now = (not np.isnan(last_s50)) and (last_c < last_s50)

    # Effective Anchor (Simulated BO Pivot vs Held Portfolio Cost Basis)
    anchor_mode = "Simulated BO Cycle"
    eff_in_pos = bool(is_in_position)
    eff_pivot = float(active_pivot_price) if not np.isnan(active_pivot_price) else np.nan
    eff_hard_stop = float(hard_stop_price) if not np.isnan(hard_stop_price) else np.nan
    eff_dyn_stop = float(dynamic_stop) if not np.isnan(dynamic_stop) else np.nan
    eff_stop_phase = stop_phase_str
    eff_bz_upper = float(buy_zone_upper) if not np.isnan(buy_zone_upper) else np.nan
    eff_t20 = float(target_20_price) if not np.isnan(target_20_price) else np.nan
    eff_t25 = float(target_25_price) if not np.isnan(target_25_price) else np.nan

    if use_portfolio_anchor and portfolio_cost_basis is not None and portfolio_cost_basis > 0:
        anchor_mode = "Portfolio Cost Basis"
        eff_in_pos = True
        eff_pivot = float(portfolio_cost_basis)
        eff_hard_stop = eff_pivot * (1.0 - (init_stop_pct / 100.0))
        eff_bz_upper = eff_pivot * (1.0 + (buy_zone_pct / 100.0))
        eff_t20 = eff_pivot * (1.0 + (profit_t1_pct / 100.0))
        eff_t25 = eff_pivot * (1.0 + (profit_t2_pct / 100.0))
        gain_cost = ((last_c - eff_pivot) / eff_pivot) * 100.0
        trail_ref = s50_a[-1] if "50" in trail_ma_type else (s21_a[-1] if "SMA" in trail_ma_type.upper() else e21_a[-1])
        if gain_cost >= trailing_pct:
            eff_dyn_stop = max(eff_pivot, trail_ref if not np.isnan(trail_ref) else eff_pivot)
            eff_stop_phase = f"Phase 3 ({trail_ma_type} Trail vs Cost)"
        elif gain_cost >= breakeven_pct:
            eff_dyn_stop = eff_pivot
            eff_stop_phase = "Phase 2: Breakeven (Cost) 🛡️"
        else:
            eff_dyn_stop = eff_hard_stop
            eff_stop_phase = f"Phase 1: -{init_stop_pct:.1f}% Hard Stop 🛑"

    # Fixed Van Tharp 1R denominator at entry (Fix Gap #10)
    if eff_in_pos and not np.isnan(eff_pivot) and eff_pivot > 0:
        init_risk_pts = (eff_pivot - eff_hard_stop) if (not np.isnan(eff_hard_stop) and eff_pivot > eff_hard_stop) else (eff_pivot * (init_stop_pct / 100.0))
        live_r_mult = ((last_c - eff_pivot) / init_risk_pts) if init_risk_pts > 0 else 0.0
        eff_gain_pct = ((last_c - eff_pivot) / eff_pivot) * 100.0
    else:
        init_risk_pts = 0.0
        live_r_mult = 0.0
        eff_gain_pct = 0.0

    pct_to_stop = ((last_c - eff_dyn_stop) / last_c * 100.0) if (eff_in_pos and not np.isnan(eff_dyn_stop) and last_c > 0) else None

    # Latest Benchmark & RS diagnostics
    mkt_regime_now = str(bench_al["bench_regime"].iloc[-1])
    mkt_status_now = str(bench_al["mkt_status_str"].iloc[-1])
    tl_mkt_now = str(bench_al["tl_mkt"].iloc[-1])
    pt_now = bool(bench_al["power_trend_active"].iloc[-1])
    ud_ratio_now = float(ud_ratio_s[-1])
    rs_persist_now = float(rs_persist_20d_s.iloc[-1]) if pd.notna(rs_persist_20d_s.iloc[-1]) else 50.0
    up_days_15_now = int(up_days_15_s.iloc[-1]) if pd.notna(up_days_15_s.iloc[-1]) else 0
    is_ants_now = bool(is_ryan_ants_s.iloc[-1]) if pd.notna(is_ryan_ants_s.iloc[-1]) else False
    stk_dd_15_now = int(stk_dd_cnt_15_s.iloc[-1]) if pd.notna(stk_dd_cnt_15_s.iloc[-1]) else 0
    turn_now = float(turn_val_s.iloc[-1])
    is_thin_liq = turn_now < 5.0
    natr_now = float(natr_s.iloc[-1])
    atr14_now = float(atr14_a[-1]) if not np.isnan(atr14_a[-1]) else 0.0
    is_vol_expensive = natr_now >= (6.5 if is_us else 6.0)
    is_vcp_now = bool(vcp_conf_a[-1])
    t1_now, t2_now, t3_now = float(depth_t1.iloc[-1]), float(depth_t2.iloc[-1]), float(depth_t3.iloc[-1])

    # Weinstein Stage Label (Pine L875-917)
    if is_young_end:
        if final_state["minervini_template"]:
            stage_text = "IPO STAGE 2 🟢 (Young Leader)"
        elif is_below_50now:
            stage_text = "IPO PULLBACK 🔴 (<50 SMA)"
        else:
            stage_text = "IPO BASING 🟡"
    else:
        s200_rising_now = (not np.isnan(last_s200)) and (not np.isnan(s200_p20_end)) and (last_s200 > s200_p20_end)
        if (not is_below_50now) and (not np.isnan(last_s200)) and (last_s50 > last_s200) and s200_rising_now:
            stage_text = "STAGE 2 🟢 (Confirmed Uptrend)" if final_state["stage2_trend_ok"] else "STAGE 1/2 🟡 (Low RS / Early)"
        elif (not np.isnan(last_s200)) and (last_c < last_s200) and (last_s200 <= (s200_p20_end if not np.isnan(s200_p20_end) else last_s200)):
            stage_text = "STAGE 4 🔴 (Decline / Avoid)"
        elif (not np.isnan(last_s200)) and (last_c >= last_s200) and (last_s50 <= last_s200):
            stage_text = "STAGE 1 🟡 (Basing / Transition)"
        else:
            stage_text = "STAGE 3 ⚠️ (Topping / Volatile)"

    if is_stage4_now:
        base_quality_str = "Stage 4 Decline 🔴 (Reset to B1)"
    elif is_below_50now:
        base_quality_str = f"Base {base_count} Pullback 🟡 (<50 SMA)"
    elif not final_state["stage2_trend_ok"] and not eff_in_pos:
        base_quality_str = f"Base {base_count} (Await Stage 2 / RS>=70)"
    elif base_count <= 2:
        base_quality_str = f"Base {base_count} ({'Fresh Leader 🎯' if eff_in_pos else 'Forming 🎯'})"
    elif base_count == 3:
        base_quality_str = "Base 3 (Mature ⚠️)"
    else:
        base_quality_str = f"Base {base_count}+ (Late Stage 🚨)"

    # Pullback Health Diagnostic (Pine L919-941)
    is_dn_now = n >= 2 and last_c < c_a[-2]
    is_near_supp = ((not np.isnan(e21_a[-1])) and e21_a[-1] * 0.98 <= last_c <= e21_a[-1] * 1.03) or (
        (not np.isnan(last_s50)) and last_s50 * 0.98 <= last_c <= last_s50 * 1.03
    )
    if stk_dd_15_now >= 3:
        pb_health = f"HEAVY DIST ⚠️ ({stk_dd_15_now} DDs/15d)"
    elif is_below_50now and (not np.isnan(v50p_a[-1])) and v_a[-1] > v50p_a[-1]:
        pb_health = "INST. SELLING 🔴 (50 SMA Broken)"
    elif is_dn_now and (not np.isnan(v50p_a[-1])) and v_a[-1] > v50p_a[-1]:
        pb_health = "DISTRIBUTION BAR ⚠️"
    elif is_dn_now and (not np.isnan(v50p_a[-1])) and v_a[-1] < v50p_a[-1]:
        pb_health = "HEALTHY PB ☕ (Low Vol @ Supp)" if is_near_supp else "CONSTRUCTIVE PB ☕"
    elif is_near_supp:
        pb_health = "TESTING SUPPORT ✅ (Near 21/50 MA)"
    else:
        pb_health = "ADVANCING NORMALLY ✅"

    # Sizing & Action Directive (Pine L1168-1226)
    if mkt_regime_now == "IN CORRECTION":
        sizing_str = "25% Pilot (Mkt Red Caution)" if (sig_history["ep"][-1] or sig_history["htf"][-1]) else "0% Cash (Preserve Capital)"
    elif is_stage4_now:
        sizing_str = "0% Cash (Stage 4 — Avoid)"
    elif (not final_state["stage2_trend_ok"]) and (not eff_in_pos) and (not sig_history["ep"][-1]) and (not sig_history["htf"][-1]):
        sizing_str = "0% Cash (Await Stage 2 / RS>=70)"
    elif base_count >= 4 or is_thin_liq:
        sizing_str = f"25% Pilot ({'Thin Liq' if is_thin_liq else 'Late B4+'})"
    elif mkt_regime_now == "UNDER PRESSURE" or base_count == 3 or canonical_rs < 70:
        sizing_str = "50% Half Unit (Caution)"
    elif pt_now and base_count <= 2 and canonical_rs >= 85:
        sizing_str = "125% Max (Power Trend 🚀)"
    elif mkt_regime_now == "CONFIRMED UPTREND" and base_count <= 2 and canonical_rs >= 80 and final_state["minervini_template"]:
        sizing_str = "100% Full Unit (A+ Leader)"
    else:
        sizing_str = "50–75% Standard Unit"

    if pre_earnings_days is not None and 0 <= pre_earnings_days <= 7:
        sizing_str += f" • ⚠️ Earnings {pre_earnings_days}d"

    if final_state["is_stop_event"] or is_stopped_out:
        action_text, action_tone = "🛑 STOPPED OUT (Exit / Cooldown)", "red"
    elif final_state["is_climax_rev"] or final_state["is_ext_exhaust"]:
        action_text = "🚨 CLIMAX EXHAUSTION (Sell 75-100%)" if final_state["is_climax_rev"] else "🚨 EXT-EXHAUST (Trim 25-50%)"
        action_tone = "red"
    elif final_state["is_profit_target_hit"] or (profit_taken and eff_in_pos and not eight_week_active):
        action_text, action_tone = f"💰 +{profit_t1_pct:.0f}% TARGET HIT (Book Partial / Trail)", "green"
    elif final_state["is_exp_breaker"]:
        action_text, action_tone = "⚠️ EXP-BRK (Pivot Lost on Vol — Scratch)", "amber"
    elif eight_week_active:
        action_text, action_tone = "🏆 8-WEEK HOLD (Hold via 50 SMA Defense)", "teal"
    elif eff_in_pos and not np.isnan(eff_pivot):
        if eff_pivot <= last_c <= eff_bz_upper:
            action_text = f"⚠️ BUY ZONE IN MKT RED ({sizing_str})" if mkt_regime_now == "IN CORRECTION" else f"🟢 IN BUY ZONE (0 to +{buy_zone_pct:.1f}% • {sizing_str})"
            action_tone = "amber" if mkt_regime_now == "IN CORRECTION" else "green"
        elif last_c > eff_bz_upper:
            action_text, action_tone = f"⚠️ EXTENDED (+{eff_gain_pct:.1f}% > Pivot — Do Not Chase)", "amber"
        else:
            action_text, action_tone = f"HOLDING TRADE ({eff_stop_phase})", "teal"
    elif final_state["is_breakout_bar"]:
        action_text, action_tone = f"🚀 {final_state['last_bo_type_str']} ({sizing_str})", ("amber" if mkt_regime_now == "IN CORRECTION" else "green")
    elif final_state["is_pocket_pivot"]:
        action_text, action_tone = f"⚡ POCKET PIVOT (Early Accum • {sizing_str})", "teal"
    elif final_state["is_vdu"]:
        piv_ref = final_state["highest_prev_20"]
        action_text, action_tone = f"💧 VOLUME DRY-UP (Alert @ {piv_ref:,.2f})", "teal"
    elif is_stage4_now:
        action_text, action_tone = "🔴 STAGE 4 DECLINE (Avoid — Cash)", "red"
    else:
        action_text, action_tone = "WAITING FOR PIVOT BASE", "slate"

    # Traffic lights & formatted metrics
    curr_sym = "$" if is_us else "₹"
    bench_name_lbl = bench_label or ("Nasdaq/S&P" if is_us else "Nifty 500")
    ud_grade = (
        "A (Heavy Accum)" if ud_ratio_now >= 1.25 else
        "B (Accumulation)" if ud_ratio_now >= 1.10 else
        "C (Neutral)" if ud_ratio_now >= 0.90 else
        "D (Distribution)" if ud_ratio_now >= 0.75 else "E (Heavy Selling)"
    )
    rs_grade = (
        "A+ (Leader)" if canonical_rs >= 90 else
        "A (Leading)" if canonical_rs >= 80 else
        "B (Above Avg)" if canonical_rs >= 70 else
        "C (Neutral)" if canonical_rs >= 50 else "D/E (Laggard)"
    )

    tr = final_state["tennis_ratio"]
    bd = final_state["bench_down_cnt"]
    sp = final_state["tennis_pts"]
    tennis_str = f"{sp}/{bd} ({tr:.0f}%)" if (not np.isnan(tr)) else "Scanning Index Drops..."
    tennis_prefix = f"TENNIS 🎾 ({tennis_str})" if final_state["is_tennis_now"] else (f"EGG 🥚 ({tennis_str})" if (not np.isnan(tr) and tr < 40.0) else tennis_str)
    tape_str = f"{tennis_prefix} | RVOL: {vr_a[-1]:.2f}x • DCR: {dcr_pct.iloc[-1]:.0f}%"

    d10, ad10, d50 = final_state["dist_10"], final_state["atr_dist_10"], final_state["dist_50"]
    stretch_tag = " 🚨 EXHAUST" if ad10 >= 3.0 else (" ⚠️ CLIMAX ZONE" if (d50 > 25.0 and (is_young_end or final_state["dist_200"] >= 0)) else "")
    d65_note = f" | 65E: {dist_65ema_pct:+.1f}%" if dist_65ema_pct is not None and pd.notna(dist_65ema_pct) else ""
    stretch_str = f"10E: {ad10:.1f} ATR ({d10:+.1f}%) | 50S: {d50:+.1f}%{d65_note}{stretch_tag}"

    vcp_lbl = f"VCP ⚡ ({t1_now:.0f}→{t2_now:.0f}→{t3_now:.1f}%)" if is_vcp_now else (f"Contracting ({t3_now:.1f}%)" if vcp_contracting_s.iloc[-1] else "No VCP")
    if eight_week_active:
        ew_wk = ((n - 1 - eight_week_start) // 5) + 1
        ew_str = f"ACTIVE 🏆 (Wk {ew_wk}/8 — 50 SMA Defense)"
    elif eight_week_abort:
        ew_str = "ABORTED 🚨 (50 SMA Broken on Vol)"
    elif eight_week_done:
        ew_str = "COMPLETED ✅ (8 Weeks Held)"
    elif eff_in_pos and not np.isnan(eff_t20):
        ew_str = f"T1: {curr_sym}{eff_t20:,.2f} (+{profit_t1_pct:.0f}%) | T2: {curr_sym}{eff_t25:,.2f} (+{profit_t2_pct:.0f}%)"
    else:
        ew_str = f"Inactive (<+{profit_t1_pct:.0f}% in 3 wks)"

    liq_tier = (
        ("Inst. Liquid 🟢" if turn_now >= 20.0 else ("Mid-Tier 🟡" if turn_now >= 5.0 else "Thin 🔴"))
        if is_us else
        ("Inst. Liquid 🟢" if turn_now >= 25.0 else ("Mid-Tier 🟡" if turn_now >= 5.0 else "Thin 🔴"))
    )
    turn_str = f"{'$' if is_us else '₹'}{turn_now:,.2f}{'M/d' if is_us else ' Cr/d'} ({liq_tier})"

    a1_s = f"{a_1m[-1]:+.1f}%" if has_1m.iloc[-1] and has_bench else "N/A"
    a3_s = f"{(stk_3m[-1] - bnch_3m[-1]):+.1f}%" if has_3m.iloc[-1] and has_bench else "N/A"
    a6_s = f"{(stk_6m[-1] - bnch_6m[-1]):+.1f}%" if has_6m.iloc[-1] and has_bench else "N/A"

    if eff_in_pos and not np.isnan(eff_pivot):
        piv_str = f"{curr_sym}{eff_pivot:,.2f} (Buy Zone: {curr_sym}{eff_pivot:,.2f} – {curr_sym}{eff_bz_upper:,.2f})"
        stop_dist_txt = f" ({pct_to_stop:.1f}% away)" if pct_to_stop is not None else ""
        pl_r_str = f"{eff_gain_pct:+.1f}% ({live_r_mult:+.2f}R) | Stop: {curr_sym}{eff_dyn_stop:,.2f}{stop_dist_txt} [{eff_stop_phase}]"
    else:
        h20_now = final_state["highest_prev_20"]
        h20_txt = f"{curr_sym}{h20_now:,.2f}" if not np.isnan(h20_now) else "N/A"
        piv_str = f"Watch Pivot: {h20_txt} (Depth: {final_state['base_depth_pct']:.1f}%)"
        pl_r_str = "CASH / SCANNING ⚪ (Waiting Constructive Setup)"

    tl_stage = "🔴" if is_stage4_now else ("🟡" if is_below_50now else ("🟠" if base_count >= 4 else ("🟢" if final_state["minervini_template"] else "🟡")))
    tl_action = "🔴" if action_tone == "red" else ("🟢" if action_tone in ("green", "teal") else ("🟡" if action_tone == "amber" else "⚪"))
    tl_piv = "🔴" if is_stopped_out else ("🟢" if (eff_in_pos and not np.isnan(eff_bz_upper) and eff_pivot <= last_c <= eff_bz_upper) else ("🟠" if (eff_in_pos and not np.isnan(eff_bz_upper) and last_c > eff_bz_upper) else ("🟡" if final_state["is_valid_base"] else "⚪")))
    tl_pl = "⚪" if not eff_in_pos else ("🟢" if eff_gain_pct >= 0 else ("🟡" if eff_gain_pct >= -4.0 else "🔴"))
    tl_rs = "🟢" if canonical_rs >= 85 else ("🟡" if canonical_rs >= 70 else "🔴")
    tl_ud = "🐜" if is_ants_now else ("🟠" if stk_dd_15_now >= 3 else ("🟢" if ud_ratio_now >= 1.10 else ("⚪" if ud_ratio_now >= 0.90 else "🔴")))
    tl_stretch = "🚨" if ad10 >= 3.0 else ("🟠" if d50 > 25.0 else ("🟢" if d50 >= 0 else "🔴"))
    hist_label = f"{n}D ({'4Y+ Cycle' if n >= 750 else ('2Y Cycle' if n >= 450 else '1.2Y Window')})"

    hud_rows = [
        {"id": 1, "group": "setup", "tone": tl_mkt_now, "label": f"Market Direction ({bench_name_lbl})", "value": mkt_status_now, "tooltip": "O'Neil FTD & expiring 25-day Distribution Day engine on benchmark index."},
        {"id": 2, "group": "setup", "tone": "🚀" if pt_now else "⚪", "label": "IBD Power Trend (Webster)", "value": "ACTIVE 🚀 (Max Long Exposure)" if pt_now else "OFF ⚪ (Standard Risk Defense)", "tooltip": "Benchmark lows > 21 EMA (10d) & 21 EMA > 50 SMA (5d) with rising 50 SMA."},
        {"id": 3, "group": "setup", "tone": tl_action, "label": f"Action & Sizing ({anchor_mode})", "value": f"{action_text} | {sizing_str}", "tooltip": "Combines Market Regime, Base Stage, Canonical RS, Liquidity & Earnings buffer."},
        {"id": 4, "group": "setup", "tone": tl_stage, "label": f"Stage & Base Cycle [{hist_label}]", "value": f"{stage_text} • {base_quality_str}" + (f" • {cap_stage_badge}" if cap_stage_badge else ""), "tooltip": "David Ryan Base Counting + Weinstein Stage across loaded OHLCV history."},
        {"id": 5, "group": "setup", "tone": tl_piv, "label": "Active Pivot & Buy Zone" if eff_in_pos else "Prospective Pivot & Depth", "value": piv_str, "tooltip": "Breakout pivot reference and +5% institutional Buy Zone ceiling."},
        {"id": 6, "group": "setup", "tone": tl_pl, "label": "P&L / R-Mult / Stop Phase", "value": pl_r_str, "tooltip": "Van Tharp R-Multiple vs fixed entry 1R risk + 3-Phase Dynamic Trailing Stop."},
        {"id": 7, "group": "flow", "tone": tl_rs, "label": "RS Rating & Persistence (L)", "value": f"{canonical_rs:.0f} — {rs_grade} | {rs_persist_now:.0f}% 20D Beat" + (" 🌟" if final_state["is_rs_ahead_now"] else (" 🟢" if final_state["is_rs_at_high"] else "")), "tooltip": "Canonical Screener Universe Percentile RS (0-99) + 20D daily beat persistence."},
        {"id": 8, "group": "flow", "tone": tl_ud, "label": "Sponsorship / U-D & Ants (S)", "value": f"{ud_ratio_now:.2f} — {ud_grade}" + (f" • ANTS 🐜 ({up_days_15_now}/15d)" if is_ants_now else "") + (f" | {stk_dd_15_now} DDs" if stk_dd_15_now > 0 else ""), "tooltip": "Symmetric 50D Up/Down Volume Ratio + David Ryan Ants (12+/15 up days, +15% on vol)."},
        {"id": 9, "group": "flow", "tone": "🎾" if final_state["is_tennis_now"] else "⚪", "label": "Tennis Ball & Tape (RVOL/DCR)", "value": tape_str, "tooltip": "Resilience on benchmark down days (>= -0.35%) + Today's RVOL & DCR%."},
        {"id": 10, "group": "flow", "tone": tl_stretch, "label": "Kell 10 EMA / 50 SMA Stretch", "value": stretch_str, "tooltip": "Oliver Kell ATR extension above 10 EMA (>=3.0 ATR warns exhaustion) + 50 SMA / 65 EMA distance."},
        {"id": 11, "group": "flow", "tone": "🟢" if final_state["minervini_template"] else "⚪", "label": "Minervini Trend Template", "value": ("IPO STAGE 2 🟢 (Young Leader)" if (is_young_end and final_state["minervini_template"]) else ("IPO DEVELOPING 🟡" if is_young_end else (f"CONFIRMED STAGE 2 🟢 (8/8)" if final_state["minervini_template"] else f"INCOMPLETE ⚪ ({final_state['minervini_score']}/8)"))), "tooltip": "Minervini 8-Point Stage 2 Trend Template (with Young IPO grace when <200 bars)."},
        {"id": 12, "group": "flow", "tone": "⚡" if is_vcp_now else "☕", "label": "VCP Waves & Pullback Health", "value": f"{vcp_lbl} | {pb_health}", "tooltip": "Disjoint 3-Wave VCP ([25..36], [12..24], [1..11]) + pullback volume character."},
        {"id": 13, "group": "setup", "tone": "🏆" if eight_week_active else ("🚨" if eight_week_abort else "⚪"), "label": "8-Week Hold & Targets", "value": ew_str, "tooltip": "+20% within 3 weeks triggers O'Neil 8-week hold via 50 SMA defense."},
        {"id": 14, "group": "flow", "tone": "🟢" if not is_thin_liq else "🔴", "label": "Institutional Liquidity (50d)", "value": turn_str, "tooltip": "50-day average daily rupee/dollar turnover tier."},
        {"id": 15, "group": "flow", "tone": "🔴" if is_vol_expensive else "🟢", "label": "Cost of Business (NATR %)", "value": f"{natr_now:.2f}% (ATR: {curr_sym}{atr14_now:,.2f}) — {'HIGH VOL ⚠️' if is_vol_expensive else 'CONTROLLED ✅'}", "tooltip": "14-day Normalized Average True Range (volatility drag on stop width)."},
        {"id": 16, "group": "flow", "tone": "🟢" if comp_alpha[-1] >= 0 else "🔴", "label": "Alpha vs Bench (1M/3M/6M)", "value": f"{a1_s} / {a3_s} / {a6_s}" + (" (Vol Proxy)" if bool(bench_al["vol_proxy_used"].iloc[-1]) else ""), "tooltip": "Stock return minus benchmark return across 1M (21d), 3M (63d), and 6M (126d)."},
    ]

    # Trailing 5-bar Live Signals Strip (Pine L1561-1687)
    def rec5(key: str) -> bool:
        arr = sig_history[key]
        return any(arr[max(0, n - 5):])

    live_signals_5d: list[dict[str, str]] = []
    if eight_week_active:
        live_signals_5d.append({"signal": "8-WK HOLD 🏆", "action": "HOLD CORE: +20% in <=3 wks. Hold for 8 weeks via 50 SMA stop.", "color": "#ffd700"})
    if eff_in_pos and not np.isnan(eff_pivot) and eff_pivot <= last_c <= eff_bz_upper and (not rec5("bo")) and (not rec5("gap")):
        live_signals_5d.append({"signal": "BUY ZONE 🟢", "action": f"ACTIONABLE ENTRY: Within +{buy_zone_pct:.0f}% of {curr_sym}{eff_pivot:,.2f} pivot ({sizing_str}). Stop @ {curr_sym}{eff_dyn_stop:,.2f}.", "color": "#00c853"})
    if eff_in_pos and not np.isnan(eff_bz_upper) and last_c > eff_bz_upper and (not eight_week_active):
        live_signals_5d.append({"signal": "EXTENDED ⚠️", "action": f"DO NOT CHASE: Extended {eff_gain_pct:+.1f}% > pivot. Trail stop @ {curr_sym}{eff_dyn_stop:,.2f}.", "color": "#fb8c00"})
    if is_vcp_now and not eff_in_pos:
        h20_vcp = final_state["highest_prev_20"]
        live_signals_5d.append({"signal": "VCP READY ⚡", "action": f"3-WAVE VCP ({t1_now:.0f}→{t2_now:.0f}→{t3_now:.1f}%): Stage buy-stop @ {curr_sym}{h20_vcp:,.2f}.", "color": "#00e5ff"})
    if is_ants_now:
        live_signals_5d.append({"signal": "ANTS 🐜", "action": f"INST. ACCUMULATION: {up_days_15_now}/15 up days (+15% on vol). Hold winners / add on 10-21 EMA.", "color": "#26a69a"})

    rec5_map = [
        ("ep", "EP 💥", None, "#76ff03"),
        ("htf", "HTF-BUY 🚀🚀", None, "#00bcd4"),
        ("gap", f"GAP-BUY [B{base_count}] 🚀", None, "#00e676"),
        ("bo", f"BUY [B{base_count}] 🎯", None, "#00c853"),
        ("add", "ADD ➕", None, "#2e7d32"),
        ("retest", "RETEST 💧", None, "#4fc3f7"),
        ("wedge", "WEDGE ⚡", None, "#00e5ff"),
        ("pp", "PP ⚡", None, "#00acc1"),
        ("rsnh", "RS-NH 🌟", None, "#ffd700"),
        ("vdu", "VDU 💧", None, "#1e88e5"),
        ("3wt", "3WT 🧱", None, "#fb8c00"),
        ("tennis", "TENNIS 🎾", None, "#ffd700"),
        ("prof", "PROFIT 💰", None, "#ffd700"),
        ("stop", "STOP 🛑", None, "#c62828"),
        ("fbo", "FAILED BO ❌", None, "#b71c1c"),
        ("exp", "EXP-BRK ⚠️", None, "#e65100"),
        ("clmx", "CLIMAX 🚨", None, "#7b1fa2"),
        ("ext", "EXT-EXHAUST 🚨", None, "#aa00ff"),
        ("lpd", "LPD ⚠️", None, "#ff1744"),
        ("50brk", "50d-BRK ⚠️", None, "#d32f2f"),
        ("21w", "21d-WARN ⚡", None, "#f57c00"),
        ("stkdd", "STK-DIST ⚠️", None, "#d84315"),
        ("stall", "STALLING ⚠️", None, "#ff6f00"),
        ("wtrap", "WEDGE TRAP ❌", None, "#b71c1c"),
        ("reset", "BASE RESET 🔄", None, "#5c6bc0"),
    ]
    playbook_by_sym = {g["symbol"]: g["action"] for g in _PLAYBOOK_GUIDE}
    for k, disp_lbl, _, col in rec5_map:
        if rec5(k):
            lookup_key = (
                "BUY [B1-B2] 🎯" if (k == "bo" and base_count <= 2) else
                "BUY [B3] 🎯" if (k == "bo" and base_count == 3) else
                "BUY [B4+ ⚠️] 🎯" if k == "bo" else
                ("GAP-BUY 🚀" if k == "gap" else disp_lbl)
            )
            live_signals_5d.append({"signal": disp_lbl, "action": playbook_by_sym.get(lookup_key, ""), "color": col})

    return {
        "series_df": series_df,
        "hud_rows": hud_rows,
        "live_signals_5d": live_signals_5d,
        "header_summary": {
            "stage_text": stage_text,
            "base_count": int(base_count),
            "base_quality_str": base_quality_str,
            "action_text": action_text,
            "action_tone": action_tone,
            "sizing_str": sizing_str,
            "stop_phase_str": eff_stop_phase,
            "is_in_position": eff_in_pos,
            "is_stopped_out": bool(is_stopped_out),
            "active_pivot": None if np.isnan(eff_pivot) else float(eff_pivot),
            "prospective_pivot": None if np.isnan(final_state["highest_prev_20"]) else float(final_state["highest_prev_20"]),
            "dynamic_stop": None if np.isnan(eff_dyn_stop) else float(eff_dyn_stop),
            "hard_stop_price": None if np.isnan(eff_hard_stop) else float(eff_hard_stop),
            "init_risk_pts": float(init_risk_pts),
            "live_r_mult": float(live_r_mult),
            "pct_to_stop": pct_to_stop,
            "buy_zone_upper": None if np.isnan(eff_bz_upper) else float(eff_bz_upper),
            "target_20": None if np.isnan(eff_t20) else float(eff_t20),
            "target_25": None if np.isnan(eff_t25) else float(eff_t25),
            "history_bars": int(n),
            "hist_label": hist_label,
            "anchor_mode": anchor_mode,
            "has_split_warning": has_split_warning,
            "has_ohlc": has_ohlc,
            "has_volume": has_volume,
            "canonical_rs": canonical_rs,
            "eight_week_active": bool(eight_week_active),
        },
    }
