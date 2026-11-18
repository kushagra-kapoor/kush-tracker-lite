import numpy as np
import pandas as pd
import pytest

from canslim_rules import (
    classify_leader_tier,
    classify_stage2_series,
    has_valid_fundamentals,
    is_tml_confluence,
)
from components import compute_stockbee_monitor_table


def _ref_breadth_engine_tml(eps_g, sales_g, roe_val, sp_grade, sp_change, udvr_val, has_bb=False):
    has_inst_accum = bool(
        sp_grade in ["A+", "A"]
        or sp_change >= 0.5
        or (udvr_val >= 1.25 and has_bb)
        or udvr_val >= 1.50
    )
    is_c_and_a_strong = bool(
        (eps_g >= 25.0 and sales_g >= 20.0)
        or (sales_g >= 35.0 and eps_g >= 15.0)
        or (eps_g >= 50.0 and sales_g >= 15.0)
    )
    return bool(is_c_and_a_strong and (roe_val >= 15.0 or has_inst_accum) and roe_val >= 0.0)


@pytest.mark.parametrize(
    "eps,sales,roe,grade,chg,udvr,bb",
    [
        (25.0, 20.0, 15.0, "C", 0.0, 1.0, False),
        (24.9, 20.0, 20.0, "A+", 1.0, 2.0, True),
        (15.0, 35.0, 16.0, "NR", 0.0, 1.0, False),
        (14.9, 35.0, 16.0, "A", 1.0, 2.0, True),
        (50.0, 15.0, 10.0, "A", 0.0, 1.0, False),
        (50.0, 15.0, 10.0, "C", 0.4, 1.49, False),
        (50.0, 15.0, 10.0, "C", 0.5, 1.0, False),
        (50.0, 15.0, 10.0, "C", 0.0, 1.50, False),
        (50.0, 15.0, 10.0, "C", 0.0, 1.25, True),
        (50.0, 14.9, 25.0, "A+", 2.0, 2.0, True),
        (40.0, 30.0, -0.1, "A+", 2.0, 2.0, True),
        (0.0, 0.0, 0.0, "NR", 0.0, 1.0, False),
    ],
)
def test_tml_confluence_parity_with_breadth_engine(eps, sales, roe, grade, chg, udvr, bb):
    assert is_tml_confluence(eps, sales, roe, grade, chg, udvr, bb) == _ref_breadth_engine_tml(
        eps, sales, roe, grade, chg, udvr, bb
    )


def test_has_valid_fundamentals_distinguishes_missing_zeroes():
    assert has_valid_fundamentals(None) is False
    assert has_valid_fundamentals({}) is False
    assert has_valid_fundamentals({"eps_growth": 0.0, "sales_growth": 0.0, "roe": 0.0}) is False
    assert has_valid_fundamentals({"eps_growth": -5.0, "sales_growth": 0.0, "roe": 0.0}) is True
    assert has_valid_fundamentals({"eps_growth": 0.0, "sales_growth": 0.0, "roe": 18.0}) is True


def test_classify_leader_tier():
    assert classify_leader_tier(92, 40, 30, 20, True, True) == (4, "💎 TML", "purple")
    assert classify_leader_tier(92, 18, 12, 10, True, False) == (3, "🚀 Growth RS", "emerald")
    assert classify_leader_tier(97, 0, 0, 0, False, False) == (2, "📈 RS (Funda N/A)", "cyan")
    assert classify_leader_tier(98, -12, 5, 4, True, False) == (1, "⚠️ Neg EPS/Sales", "amber")


def test_classify_stage2_series_rising_vs_falling_200_and_ipo():
    dates = pd.bdate_range("2025-01-01", periods=250)
    # Uptrend: 200 SMA rising strictly
    s_up = pd.Series(np.linspace(100, 200, 250), index=dates)
    # Falling 200 SMA even though last close popped above 50 SMA
    vals_fall = np.concatenate([np.linspace(300, 120, 230), np.linspace(120, 145, 20)])
    s_fall = pd.Series(vals_fall, index=dates)
    # IPO (< 200 bars) above 50 SMA
    vals_ipo = np.concatenate([np.full(140, np.nan), np.linspace(50, 90, 110)])
    s_ipo = pd.Series(vals_ipo, index=dates)

    df = pd.DataFrame({"UP": s_up, "FALL200": s_fall, "IPO": s_ipo})
    is_s2, abv50 = classify_stage2_series(df)

    assert bool(is_s2["UP"]) is True
    assert bool(is_s2["FALL200"]) is False
    assert bool(abv50["IPO"]) is True
    assert bool(is_s2["IPO"]) is True


def test_stockbee_monitor_table_determinism():
    dates = pd.bdate_range("2025-06-01", periods=150)
    rng = np.random.default_rng(7)
    data = {f"STK{i}": 100 * np.cumprod(1 + rng.normal(0.002, 0.02, len(dates))) for i in range(12)}
    close_df = pd.DataFrame(data, index=dates)
    # Force clear +5% burst on last day
    close_df.iloc[-1, :4] = close_df.iloc[-2, :4] * 1.055
    tbl = compute_stockbee_monitor_table(close_df, lookback_days=20)
    assert not tbl.empty
    assert int(tbl.iloc[0]["Up_4_Pct"]) >= 4
