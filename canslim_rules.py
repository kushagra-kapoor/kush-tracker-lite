"""
Canonical CANSLIM Quantitative Rules
====================================
Pure stateless helper functions (no Streamlit, SQLite, or network I/O) shared
across Market Breadth, Thrusting Industries, and the Cluster Leadership Board.
"""

from __future__ import annotations

import pandas as pd


def has_valid_fundamentals(f_row: dict | None) -> bool:
    """Return True only when a fundamentals row exists with at least one non-zero core metric."""
    if not isinstance(f_row, dict) or not f_row:
        return False
    eps_g = float(f_row.get("eps_growth", 0.0) or 0.0)
    sales_g = float(f_row.get("sales_growth", 0.0) or 0.0)
    roe_v = float(f_row.get("roe", 0.0) or 0.0)
    return abs(eps_g) > 1e-9 or abs(sales_g) > 1e-9 or abs(roe_v) > 1e-9


def is_tml_confluence(
    eps_growth: float,
    sales_growth: float,
    roe: float,
    sponsorship_grade: str = "NR",
    inst_change_qoq: float = 0.0,
    udvr_50d: float = 1.0,
    has_beach_ball: bool = False,
) -> bool:
    """
    Canonical True Market Leader (TML) fundamental + institutional confluence gate.
    Matches breadth_engine.py L1383-L1391 verbatim.
    """
    eps_g = float(eps_growth or 0.0)
    sales_g = float(sales_growth or 0.0)
    roe_val = float(roe or 0.0)
    sp_grade = str(sponsorship_grade or "NR").upper().strip()
    sp_change = float(inst_change_qoq or 0.0)
    udvr = float(udvr_50d or 0.0)

    has_inst_accum = bool(
        sp_grade in ("A+", "A")
        or sp_change >= 0.5
        or (udvr >= 1.25 and has_beach_ball)
        or udvr >= 1.50
    )
    is_c_and_a_strong = bool(
        (eps_g >= 25.0 and sales_g >= 20.0)
        or (sales_g >= 35.0 and eps_g >= 15.0)
        or (eps_g >= 50.0 and sales_g >= 15.0)
    )
    return bool(is_c_and_a_strong and (roe_val >= 15.0 or has_inst_accum) and roe_val >= 0.0)


def classify_leader_tier(
    rs: float,
    eps_growth: float,
    sales_growth: float,
    roe: float,
    has_funda: bool,
    is_tml: bool,
) -> tuple[int, str, str]:
    """
    Assign deterministic qualification tier for display inside Cluster Cards.
    Returns (tier_num, badge_label, tone).
    """
    rs_v = float(rs or 0.0)
    eps_g = float(eps_growth or 0.0)
    sales_g = float(sales_growth or 0.0)
    roe_v = float(roe or 0.0)
    is_neg = bool(has_funda and (eps_g < 0.0 or sales_g < -5.0 or roe_v < 0.0))

    if rs_v >= 80.0 and is_tml:
        return (4, "💎 TML", "purple")
    if rs_v >= 80.0 and has_funda and not is_neg and (eps_g >= 15.0 or sales_g >= 15.0):
        return (3, "🚀 Growth RS", "emerald")
    if rs_v >= 80.0 and not is_neg:
        return (2, "📈 Pure RS" if has_funda else "📈 RS (Funda N/A)", "cyan")
    if is_neg:
        return (1, "⚠️ Neg EPS/Sales", "amber")
    return (1, "⚪ Below RS80", "slate")


def classify_stage2_series(close_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Classify latest-row Stage 2 uptrend across columns:
      - Mature (>=200 valid bars): Close > 50 SMA > 200 SMA AND 200 SMA > 200 SMA (20 sessions ago)
      - IPO (<200 valid bars): Close > 50 SMA (handled cleanly without Stage-4 penalty)
    Returns (is_stage2_or_ipo_uptrend, above_50sma).
    """
    if close_df is None or close_df.empty:
        empty = pd.Series(dtype=bool)
        return empty, empty

    c_now = close_df.iloc[-1]
    sma50 = close_df.rolling(50, min_periods=35).mean().iloc[-1]
    sma200_roll = close_df.rolling(200, min_periods=180).mean()
    sma200_now = sma200_roll.iloc[-1]
    sma200_prev = sma200_roll.iloc[-21] if len(sma200_roll) >= 21 else sma200_roll.iloc[0]

    bars_valid = close_df.notna().sum(axis=0)
    above_50 = (c_now > sma50).fillna(False)

    is_s2_mature = (
        (bars_valid >= 200)
        & (c_now > sma50)
        & (sma50 > sma200_now)
        & (sma200_now > sma200_prev)
    ).fillna(False)
    is_s2_ipo = ((bars_valid < 200) & above_50).fillna(False)

    return (is_s2_mature | is_s2_ipo), above_50
