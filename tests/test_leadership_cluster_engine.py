import time
import numpy as np
import pandas as pd
import pytest

from components import _stockbee_allowed_tickers_for_scope, compute_stockbee_monitor_table
from leadership_cluster_engine import (
    _compute_cluster_board_core,
    _compute_thrusting_industries_core,
    _select_cluster_tml_leaders,
)
from leadership_cluster_ui import (
    _extract_unique_de_drill_groups,
    _render_cluster_card_html,
    _render_portfolio_triage_table_html,
    apply_igm_jump_to_filtered_df,
    apply_igm_jump_with_sticky,
    build_igm_jump_payload,
    resolve_igm_jump_payload,
)


def _make_base_panels(
    tickers: list[str],
    n_days: int = 230,
    end_date: str = "2026-10-30",
    daily_drift: float = 0.001,
    volume_val: float = 1_000_000.0,
):
    dates = pd.bdate_range(end=end_date, periods=n_days)
    close_data = {}
    for i, t in enumerate(tickers):
        base = 200.0 + (i % 5) * 10.0
        close_data[t] = base * np.cumprod(np.full(n_days, 1.0 + daily_drift))
    close_df = pd.DataFrame(close_data, index=dates)
    high_df = close_df * 1.005
    low_df = close_df * 0.995
    vol_df = pd.DataFrame(volume_val, index=dates, columns=tickers, dtype=float)
    return close_df, high_df, low_df, vol_df


def test_1_ledger_reconciliation_exact():
    tickers = [
        "ATTR1.NS",
        "ATTR2.NS",
        "ILLIQ.NS",
        "CA_JMP.NS",
        "CIRC.NS",
        "UNMAP.NS",
        "FLAT1.NS",
        "FLAT2.NS",
    ]
    close_df, high_df, low_df, vol_df = _make_base_panels(tickers, n_days=160)
    # All 6 test names spike >= +4% on T0 cleanly, except ILLIQ gets tiny volume,
    # CA_JMP spikes +100%, CIRC hits exact +5% upper circuit (H == L == C),
    # and UNMAP has no industry group mapping.
    for tk in ["ATTR1.NS", "ATTR2.NS", "ILLIQ.NS", "UNMAP.NS"]:
        close_df.loc[close_df.index[-1], tk] = close_df[tk].iloc[-2] * 1.06
        high_df.loc[high_df.index[-1], tk] = close_df[tk].iloc[-1] * 1.01
        low_df.loc[low_df.index[-1], tk] = close_df[tk].iloc[-1] * 0.99

    vol_df["ILLIQ.NS"] = 100.0  # tiny ADTV < 2.5 Cr
    close_df.loc[close_df.index[-1], "CA_JMP.NS"] = close_df["CA_JMP.NS"].iloc[-2] * 2.00  # +100% corporate action
    c_val = close_df["CIRC.NS"].iloc[-2] * 1.05
    close_df.loc[close_df.index[-1], "CIRC.NS"] = c_val
    high_df.loc[high_df.index[-1], "CIRC.NS"] = c_val
    low_df.loc[low_df.index[-1], "CIRC.NS"] = c_val  # H == L == C & ret >= 4.8%

    intel = {
        k: {"group_name": "Defense Electronics"}
        for k in ["ATTR1", "ATTR2", "ILLIQ", "CA_JMP", "CIRC", "FLAT1", "FLAT2"]
    }
    res = _compute_thrusting_industries_core(
        close_df=close_df,
        high_df=high_df,
        low_df=low_df,
        vol_df=vol_df,
        stock_intel=intel,
        matrix_df=None,
        market="IN",
    )
    led = res["ledger"]
    assert led["grid_up4_total"] == 6
    assert led["attributed"] == 2
    assert led["illiquid"] == 1
    assert led["quarantined"] == 2
    assert led["unmapped"] == 1
    assert led["attributed"] + led["illiquid"] + led["quarantined"] + led["unmapped"] == led["grid_up4_total"]


def test_2_shared_t0_with_compute_stockbee_monitor_table():
    tickers = [f"STK{i}.NS" for i in range(10)]
    close_df, high_df, low_df, vol_df = _make_base_panels(tickers, n_days=140)
    # Final day has sparse coverage (only 2/10 non-NaN -> skipped as shared T0!)
    close_df.iloc[-1, 2:] = np.nan
    close_df.iloc[-2, :4] = close_df.iloc[-3, :4] * 1.06

    tbl = compute_stockbee_monitor_table(close_df, lookback_days=20)
    expected_t0 = str(tbl.iloc[0]["Date"])

    intel = {f"STK{i}": {"group_name": "Semiconductors"} for i in range(10)}
    res = _compute_thrusting_industries_core(
        close_df=close_df,
        high_df=high_df,
        low_df=low_df,
        vol_df=vol_df,
        stock_intel=intel,
        matrix_df=None,
        market="IN",
    )
    assert res["as_of"]["prices_date"] == expected_t0[:10]
    assert res["ledger"]["grid_up4_total"] == int(tbl.iloc[0]["Up_4_Pct"])


def test_3_5d_hits_vs_unique_repeat_spikes():
    tickers = ["RPT.NS"] + [f"S{i}.NS" for i in range(190)]
    close_df, high_df, low_df, vol_df = _make_base_panels(tickers, n_days=140)
    # RPT spikes +5% on three separate days inside trailing 5 days
    for back in [-5, -3, -1]:
        close_df.iloc[back, 0] = close_df.iloc[back - 1, 0] * 1.055
    intel = {"RPT": {"group_name": "Solar Gear"}}
    res = _compute_thrusting_industries_core(
        close_df=close_df,
        high_df=high_df,
        low_df=low_df,
        vol_df=vol_df,
        stock_intel=intel,
        matrix_df=None,
        market="IN",
    )
    row = res["all_thrust_df"].iloc[0]
    assert int(row["Up4_5D_Hits"]) == 3
    assert int(row["Up4_5D_Unique"]) == 1


def test_4_dual_listing_ns_bo_deduplication():
    tickers = ["RELIANCE.NS", "RELIANCE.BO", "ONGC.NS", "ONGC.BO"]
    close_df, high_df, low_df, vol_df = _make_base_panels(tickers, n_days=140)
    for col in ["RELIANCE.NS", "RELIANCE.BO", "ONGC.NS", "ONGC.BO"]:
        close_df.loc[close_df.index[-1], col] = close_df[col].iloc[-2] * 1.06

    intel = {"RELIANCE": {"group_name": "Energy Refining"}, "ONGC": {"group_name": "Energy Refining"}}
    res = _compute_thrusting_industries_core(
        close_df=close_df,
        high_df=high_df,
        low_df=low_df,
        vol_df=vol_df,
        stock_intel=intel,
        matrix_df=None,
        market="IN",
    )
    row = res["all_thrust_df"].iloc[0]
    assert int(row["Group_Valid_Cnt"]) == 2
    assert int(row["Up4_Today_Cnt"]) == 2
    assert int(row["Up4_5D_Unique"]) == 2
    assert len(row["All_Thrusting_Tickers"]) == 2


def test_5_corporate_action_and_upper_circuit_quarantined():
    tickers = ["SPLIT.NS", "UC.NS", "NORM.NS", "BASE.NS"]
    close_df, high_df, low_df, vol_df = _make_base_panels(tickers, n_days=140)
    close_df.loc[close_df.index[-1], "SPLIT.NS"] = close_df["SPLIT.NS"].iloc[-2] * 2.1
    uc_p = close_df["UC.NS"].iloc[-2] * 1.05
    close_df.loc[close_df.index[-1], "UC.NS"] = uc_p
    high_df.loc[high_df.index[-1], "UC.NS"] = uc_p
    low_df.loc[low_df.index[-1], "UC.NS"] = uc_p
    close_df.loc[close_df.index[-1], "NORM.NS"] = close_df["NORM.NS"].iloc[-2] * 1.055
    high_df.loc[high_df.index[-1], "NORM.NS"] = close_df["NORM.NS"].iloc[-1] * 1.01
    low_df.loc[low_df.index[-1], "NORM.NS"] = close_df["NORM.NS"].iloc[-1] * 0.99

    intel = {t.replace(".NS", ""): {"group_name": "Chemicals"} for t in tickers}
    res = _compute_thrusting_industries_core(
        close_df=close_df,
        high_df=high_df,
        low_df=low_df,
        vol_df=vol_df,
        stock_intel=intel,
        matrix_df=None,
        market="IN",
    )
    assert res["ledger"]["quarantined"] == 2
    row = res["all_thrust_df"].iloc[0]
    assert int(row["Up4_Today_Cnt"]) == 1
    assert [t["ticker"] for t in row["All_Thrusting_Tickers"]] == ["NORM"]


def test_6_counter_trend_bounce_damping_055x():
    tickers_s2 = [f"S2_{i}" for i in range(6)]
    tickers_dn = [f"DN_{i}" for i in range(6)]
    c_s2, h_s2, l_s2, v_s2 = _make_base_panels(tickers_s2, n_days=220, daily_drift=0.002)
    c_dn, h_dn, l_dn, v_dn = _make_base_panels(tickers_dn, n_days=220, daily_drift=-0.003)

    # Identical thrust counts (3 stocks up +6% today), one Stage 2 group, one broken Stage 4 bounce group
    for c_df, h_df, l_df, grp in [(c_s2, h_s2, l_s2, tickers_s2), (c_dn, h_dn, l_dn, tickers_dn)]:
        for t in grp[:3]:
            c_df.loc[c_df.index[-1], t] = c_df[t].iloc[-2] * 1.06
            h_df.loc[h_df.index[-1], t] = c_df[t].iloc[-1] * 1.01
            l_df.loc[l_df.index[-1], t] = c_df[t].iloc[-1] * 0.99

    close_df = pd.concat([c_s2, c_dn], axis=1)
    high_df = pd.concat([h_s2, h_dn], axis=1)
    low_df = pd.concat([l_s2, l_dn], axis=1)
    vol_df = pd.concat([v_s2, v_dn], axis=1)

    intel = {t: {"group_name": "Strong S2"} for t in tickers_s2}
    intel.update({t: {"group_name": "Broken Bounce"} for t in tickers_dn})
    mat_df = pd.DataFrame(
        [
            {"Industry_Group": "Strong S2", "Rank_Today": 15, "Breadth_50SMA_Pct": 85.0, "Is_Top_20_Pct": False},
            {"Industry_Group": "Broken Bounce", "Rank_Today": 110, "Breadth_50SMA_Pct": 12.0, "Is_Top_20_Pct": False},
        ]
    )
    res = _compute_thrusting_industries_core(
        close_df=close_df,
        high_df=high_df,
        low_df=low_df,
        vol_df=vol_df,
        stock_intel=intel,
        matrix_df=mat_df,
        market="US",
    )
    by_grp = res["all_thrust_df"].set_index("Industry_Group")
    assert bool(by_grp.loc["Broken Bounce", "Is_Counter_Trend_Bounce"]) is True
    assert "Counter-Trend Bounce" in by_grp.loc["Broken Bounce", "Thrust_Status_Tag"]
    assert float(by_grp.loc["Strong S2", "Thrust_Score"]) > float(by_grp.loc["Broken Bounce", "Thrust_Score"])


def test_7_size_normalized_density_small_vs_large_group():
    small_t = [f"SM{i}" for i in range(6)]
    large_t = [f"LG{i}" for i in range(40)]
    c_all, h_all, l_all, v_all = _make_base_panels(small_t + large_t, n_days=220, daily_drift=0.0015)

    # 3/6 (50% density) vs 4/40 (10% density)
    for t in small_t[:3] + large_t[:4]:
        c_all.loc[c_all.index[-1], t] = c_all[t].iloc[-2] * 1.06
        h_all.loc[h_all.index[-1], t] = c_all[t].iloc[-1] * 1.01
        l_all.loc[l_all.index[-1], t] = c_all[t].iloc[-1] * 0.99

    intel = {t: {"group_name": "Focused Niche 6"} for t in small_t}
    intel.update({t: {"group_name": "Bloated Sector 40"} for t in large_t})

    res = _compute_thrusting_industries_core(
        close_df=c_all,
        high_df=h_all,
        low_df=l_all,
        vol_df=v_all,
        stock_intel=intel,
        matrix_df=None,
        market="US",
    )
    by_grp = res["all_thrust_df"].set_index("Industry_Group")
    assert float(by_grp.loc["Focused Niche 6", "Thrust_Score"]) > float(by_grp.loc["Bloated Sector 40", "Thrust_Score"])


def test_8_scoped_denominator_ntm750_vs_full_tradable_count():
    scoped_t = ["T1.NS", "T2.NS", "T3.NS", "T4.NS"]
    c_df, h_df, l_df, v_df = _make_base_panels(scoped_t, n_days=140)
    c_df.loc[c_df.index[-1], "T1.NS"] = c_df["T1.NS"].iloc[-2] * 1.06
    h_df.loc[h_df.index[-1], "T1.NS"] = c_df["T1.NS"].iloc[-1] * 1.01
    l_df.loc[l_df.index[-1], "T1.NS"] = c_df["T1.NS"].iloc[-1] * 0.99

    intel = {f"T{i}": {"group_name": "Auto Ancillaries"} for i in range(1, 5)}
    mat_df = pd.DataFrame([{"Industry_Group": "Auto Ancillaries", "Rank_Today": 12, "Tradable_Count": 52}])
    res = _compute_thrusting_industries_core(
        close_df=c_df,
        high_df=h_df,
        low_df=l_df,
        vol_df=v_df,
        stock_intel=intel,
        matrix_df=mat_df,
        market="IN",
    )
    row = res["all_thrust_df"].iloc[0]
    assert int(row["Group_Valid_Cnt"]) == 4
    assert int(row["Full_Universe_Cnt"]) == 52


def test_9_no_double_breadth_pure_price_rs_only():
    mat_a = pd.DataFrame(
        [
            {"Industry_Group": "Grp A", "Rank_Today": 1, "Pure_Price_RS": 92.0, "Comp_RS": 50.0, "Breadth_Pct_Rank": 10.0, "Breadth_RS80_Pct": 60.0, "Breadth_50SMA_Pct": 70.0, "RSNH_Pct": 40.0, "Tradable_Count": 10, "Is_Group_Confirmed": True, "Group_Grade": "A", "Constituents": ["A1", "A2"]},
            {"Industry_Group": "Grp B", "Rank_Today": 2, "Pure_Price_RS": 80.0, "Comp_RS": 95.0, "Breadth_Pct_Rank": 99.0, "Breadth_RS80_Pct": 60.0, "Breadth_50SMA_Pct": 70.0, "RSNH_Pct": 40.0, "Tradable_Count": 10, "Is_Group_Confirmed": True, "Group_Grade": "A", "Constituents": ["B1", "B2"]},
        ]
    )
    mat_mutated = mat_a.copy()
    mat_mutated["Breadth_Pct_Rank"] = [99.0, 1.0]
    mat_mutated["Comp_RS"] = [10.0, 99.0]

    board1 = _compute_cluster_board_core(
        matrix_df=mat_a, thrust_res={}, funda_map={}, spons_map={}, udvr_map={}, funda_as_of="2026-10-30", market="INDIA"
    )
    board2 = _compute_cluster_board_core(
        matrix_df=mat_mutated, thrust_res={}, funda_map={}, spons_map={}, udvr_map={}, funda_as_of="2026-10-30", market="INDIA"
    )
    assert board1["clusters"][0]["why_ranked"]["rs"] == board2["clusters"][0]["why_ranked"]["rs"]
    assert board1["clusters"][0]["industry_group"] == "Grp A"


def test_10_tier_ordering_and_missing_funda_never_tml():
    group_row = {
        "Top_Leaders_Full": [
            {"ticker": "NEG_EPS", "rs": 98, "is_tradable_leader": True, "dist_ema": 1.0, "status": "🟢 IN BUY ZONE"},
            {"ticker": "MISS_FN", "rs": 97, "is_tradable_leader": True, "dist_ema": 1.0, "status": "🟢 IN BUY ZONE"},
            {"ticker": "TML_GEM", "rs": 91, "is_tradable_leader": True, "dist_ema": 1.0, "status": "🟢 IN BUY ZONE"},
        ]
    }
    funda_map = {
        "TML_GEM": {"eps_growth": 42.0, "sales_growth": 32.0, "roe": 22.0},
        "MISS_FN": {"eps_growth": 0.0, "sales_growth": 0.0, "roe": 0.0},
        "NEG_EPS": {"eps_growth": -18.0, "sales_growth": 10.0, "roe": 8.0},
    }
    selected, _, _, _ = _select_cluster_tml_leaders(
        group_row=group_row,
        is_us=False,
        funda_map=funda_map,
        spons_map={"TML_GEM": {"grade": "A+", "inst_change": 1.2}},
        udvr_map={"TML_GEM": 1.8},
        thrust_ticker_map={},
        held_set=set(),
        earnings_cache=None,
        brain_score_map={},
        us_tv_map=None,
    )
    assert [s["ticker"] for s in selected] == ["TML_GEM", "MISS_FN", "NEG_EPS"]
    assert selected[0]["tier_badge"] == "💎 TML"
    assert selected[1]["has_funda"] is False
    assert selected[1]["tier_badge"] != "💎 TML"
    assert selected[2]["tier_badge"] == "⚠️ Neg EPS/Sales"


def test_11_short_history_ipo_kept_ranked_below_mature_equal_tier():
    group_row = {
        "Top_Leaders_Full": [
            {"ticker": "NEW_IPO", "rs": 95, "is_tradable_leader": True, "is_short_history": True, "bars_count": 90, "dist_ema": 1.0, "status": "🟢 IN BUY ZONE"},
            {"ticker": "MATURE", "rs": 92, "is_tradable_leader": True, "is_short_history": False, "bars_count": 252, "dist_ema": 1.0, "status": "🟢 IN BUY ZONE"},
        ]
    }
    selected, _, _, _ = _select_cluster_tml_leaders(
        group_row=group_row,
        is_us=True,
        funda_map={},
        spons_map={},
        udvr_map={},
        thrust_ticker_map={},
        held_set=set(),
        earnings_cache=None,
        brain_score_map={},
        us_tv_map=None,
    )
    assert [s["ticker"] for s in selected] == ["MATURE", "NEW_IPO"]
    assert selected[1]["is_ipo"] is True


def test_12_m_gate_red_defaults_auto_lens_to_defense():
    mat_df = pd.DataFrame(
        [
            {"Industry_Group": "Defensive Pharma", "Rank_Today": 12, "Pure_Price_RS": 82.0, "Breadth_RS80_Pct": 55.0, "Breadth_50SMA_Pct": 78.0, "RSNH_Pct": 65.0, "Near_52WH_Pct": 70.0, "RS_BlueDot_Pct": 40.0, "Tradable_Count": 12, "Is_Group_Confirmed": True, "Group_Grade": "A", "Constituents": ["PH1"]},
            {"Industry_Group": "Speculative Infra", "Rank_Today": 4, "Pure_Price_RS": 94.0, "Breadth_RS80_Pct": 70.0, "Breadth_50SMA_Pct": 30.0, "RSNH_Pct": 10.0, "Near_52WH_Pct": 15.0, "RS_BlueDot_Pct": 5.0, "Tradable_Count": 12, "Is_Group_Confirmed": True, "Group_Grade": "A", "Constituents": ["IN1"]},
        ]
    )
    board = _compute_cluster_board_core(
        matrix_df=mat_df,
        thrust_res={},
        funda_map={},
        spons_map={},
        udvr_map={},
        funda_as_of="N/A",
        market="INDIA",
        lens="AUTO",
        brain_cache={"tide_turn": {"stage_num": 1}},
    )
    assert board["m_gate"]["code"] == "RED"
    assert board["active_lens"] == "DEFENSE"
    assert "WATCHLIST & STALK" in board["m_gate"]["label"]
    assert board["clusters"][0]["industry_group"] == "Defensive Pharma"


def test_13_provisional_intraday_bar_detection():
    tickers = ["S1", "S2", "S3"]
    c_df, h_df, l_df, v_df = _make_base_panels(tickers, n_days=100, end_date="2026-10-30")
    c_df.loc[c_df.index[-1], "S1"] = c_df["S1"].iloc[-2] * 1.06
    intel = {t: {"group_name": "Software"} for t in tickers}

    res = _compute_thrusting_industries_core(
        close_df=c_df,
        high_df=h_df,
        low_df=l_df,
        vol_df=v_df,
        stock_intel=intel,
        matrix_df=None,
        market="US",
        completed_session_date="2026-10-29",
    )
    assert res["as_of"]["is_provisional"] is True
    assert "INTRADAY PROVISIONAL" in res["summary"]["headline_msg"]


def test_14_us_scope_resolution_and_small_universe_threshold():
    broad = set(_stockbee_allowed_tickers_for_scope("US", "BROAD") or [])
    all_u = set(_stockbee_allowed_tickers_for_scope("US", "ALL") or [])
    ixic_u = set(_stockbee_allowed_tickers_for_scope("US", "US_IXIC") or [])
    sp500 = set(_stockbee_allowed_tickers_for_scope("US", "SP500") or [])
    ndx100 = set(_stockbee_allowed_tickers_for_scope("US", "Nasdaq 100") or [])

    assert broad and all_u == broad and ixic_u == broad
    assert len(broad) > len(sp500)
    assert len(ndx100) <= 120

    # Small <= 180 universe allows min_unique = 1 on trailing 5D even if 0 today
    tickers = ["Q1"] + [f"N{i}" for i in range(99)]
    c_df, h_df, l_df, v_df = _make_base_panels(tickers, n_days=140)
    c_df.iloc[-3, 0] = c_df.iloc[-4, 0] * 1.06  # 1 stock spiked 3 days ago, 0 today
    intel = {"Q1": {"group_name": "Quantum Computing"}}
    res = _compute_thrusting_industries_core(
        close_df=c_df,
        high_df=h_df,
        low_df=l_df,
        vol_df=v_df,
        stock_intel=intel,
        matrix_df=None,
        market="US",
    )
    assert not res["all_thrust_df"].empty
    assert res["all_thrust_df"].iloc[0]["Industry_Group"] == "Quantum Computing"


def test_15_empty_caches_degrade_gracefully():
    res = _compute_thrusting_industries_core(
        close_df=pd.DataFrame(),
        high_df=None,
        low_df=None,
        vol_df=None,
        stock_intel={},
        matrix_df=pd.DataFrame(),
        market="IN",
    )
    assert res["thrust_df"].empty
    assert res["ledger"]["grid_up4_total"] == 0

    board = _compute_cluster_board_core(
        matrix_df=pd.DataFrame(),
        thrust_res=res,
        funda_map={},
        spons_map={},
        udvr_map={},
        funda_as_of="N/A",
        brain_cache=None,
    )
    assert board["status"] == "no_cache"
    assert board["clusters"] == []


def test_16_jump_payload_canonical_override_append_missing_and_ttl_expiry():
    now_t = 1_700_000_000.0
    sess = {"kt_igm_jump_us": build_igm_jump_payload("US", "Biotechnology", now_ts=now_t - 10.0)}
    active = resolve_igm_jump_payload(sess, "us", now_ts=now_t, ttl_sec=60.0)
    assert active is not None
    assert sess["igm_tax_choice_us"] == "🏛️ Canonical 145 Sub-Industries"
    assert sess["igm_market_universe_select_us"] == "🇺🇸 US (S&P 500 / Nasdaq)"
    assert "kt_igm_jump_us" not in sess

    df_matrix = pd.DataFrame([{"Industry_Group": "Software", "Rank_Today": 1}, {"Industry_Group": "Biotechnology", "Rank_Today": 32}])
    filtered_df = df_matrix.head(1).copy()  # Top 25 sliced out Biotechnology
    updated_df, pinned, notice = apply_igm_jump_to_filtered_df(filtered_df, df_matrix, active)
    assert pinned == "Biotechnology"
    assert "Biotechnology" in updated_df["Industry_Group"].values
    assert notice and "pinned despite active filters" in notice

    # Stale > 60s payload must be discarded silently without overriding taxonomy
    stale_sess = {
        "kt_igm_jump_in": build_igm_jump_payload("IN", "Defense Electronics", now_ts=now_t - 95.0),
        "igm_tax_choice_in": "⚡ Curated Indian Alpha Themes",
    }
    assert resolve_igm_jump_payload(stale_sess, "in", now_ts=now_t, ttl_sec=60.0) is None
    assert stale_sess["igm_tax_choice_in"] == "⚡ Curated Indian Alpha Themes"


def test_17_sympathy_laggard_and_failed_gate():
    group_row = {
        "Top_Leaders_Full": [
            {"ticker": "LEAD1", "rs": 98, "is_tradable_leader": True, "dist_ema": 2.0, "status": "🟢 IN BUY ZONE"},
            {"ticker": "LEAD2", "rs": 97, "is_tradable_leader": True, "dist_ema": 1.5, "status": "🔵 RETEST"},
            {"ticker": "FAIL1", "rs": 85, "is_tradable_leader": True, "dist_ema": -4.9, "status": "🔴 FAILED"},
            {"ticker": "DEEP_EMA", "rs": 88, "is_tradable_leader": True, "dist_ema": -7.0, "status": "⏳ BASE"},
            {"ticker": "LAG77", "rs": 77, "is_tradable_leader": True, "dist_ema": 2.4, "status": "⏳ COILING"},
            {"ticker": "LAG68", "rs": 68, "is_tradable_leader": True, "dist_ema": -1.6, "status": "⏳ BASE"},
            {"ticker": "LAG62", "rs": 62, "is_tradable_leader": True, "dist_ema": 4.3, "status": "⏳ BASE"},
        ]
    }
    selected, act_cnt, _, stats = _select_cluster_tml_leaders(
        group_row=group_row,
        is_us=False,
        funda_map={},
        spons_map={},
        udvr_map={},
        thrust_ticker_map={},
        held_set=set(),
        earnings_cache=None,
        brain_score_map={},
        us_tv_map=None,
    )
    assert [s["ticker"] for s in selected] == ["LEAD1", "LEAD2"]
    assert act_cnt == 2
    assert stats["suppressed_laggards"] == 3
    assert stats["suppressed_failed"] == 2
    assert stats["qualifying_total"] == 2

    # Board + tv_watchlist integration (plus 0-leader group omitted from tv_watchlist)
    mat_df = pd.DataFrame(
        [
            {
                "Industry_Group": "Infrastructure Operations",
                "Rank_Today": 3,
                "Pure_Price_RS": 90.0,
                "Breadth_RS80_Pct": 40.0,
                "Breadth_50SMA_Pct": 80.0,
                "RSNH_Pct": 20.0,
                "Tradable_Count": 10,
                "Is_Group_Confirmed": True,
                "Group_Grade": "A+",
                "Group_Grade_Badge": "A+ #3",
                "Top_Leaders_Full": group_row["Top_Leaders_Full"],
                "Constituents": [x["ticker"] for x in group_row["Top_Leaders_Full"]],
            },
            {
                "Industry_Group": "Hollow Group",
                "Rank_Today": 4,
                "Pure_Price_RS": 85.0,
                "Breadth_RS80_Pct": 20.0,
                "Breadth_50SMA_Pct": 60.0,
                "RSNH_Pct": 10.0,
                "Tradable_Count": 5,
                "Is_Group_Confirmed": True,
                "Group_Grade": "A",
                "Group_Grade_Badge": "A #4",
                "Top_Leaders_Full": [
                    {"ticker": "HOLLOW70", "rs": 70, "is_tradable_leader": True, "dist_ema": 1.0, "status": "⏳ BASE"},
                ],
                "Constituents": ["HOLLOW70"],
            },
        ]
    )
    board = _compute_cluster_board_core(
        matrix_df=mat_df, thrust_res={}, funda_map={}, spons_map={}, udvr_map={}, funda_as_of="N/A", market="INDIA"
    )
    c0, c1 = board["clusters"][0], board["clusters"][1]
    assert [l["ticker"] for l in c0["leaders"]] == ["LEAD1", "LEAD2"]
    assert c0["suppressed_laggards"] == 3 and c0["suppressed_failed"] == 2
    assert c1["leaders"] == [] and c1["suppressed_laggards"] == 1
    assert "###Hollow Group" not in board["tv_watchlist"]
    for bad in ("FAIL1", "DEEP_EMA", "LAG77", "LAG68", "LAG62", "HOLLOW70"):
        assert bad not in board["tv_watchlist"]


def test_18_de_danger_holdings_dedup_and_sort():
    mat_df = pd.DataFrame(
        [
            {
                "Industry_Group": "Banks - Top",
                "Rank_Today": 2,
                "Pure_Price_RS": 92.0,
                "Breadth_RS80_Pct": 60.0,
                "Breadth_50SMA_Pct": 85.0,
                "RSNH_Pct": 30.0,
                "Tradable_Count": 12,
                "Is_Group_Confirmed": True,
                "Group_Grade": "A+",
                "Group_Grade_Badge": "A+ #2",
                "Top_Leaders_Full": [{"ticker": "TOPB", "rs": 95, "is_tradable_leader": True, "dist_ema": 2.0, "status": "🟢 IN BUY ZONE"}],
                "Constituents": ["TOPB"],
            },
            {
                "Industry_Group": "Mid-Tier D Grp",
                "Rank_Today": 110,
                "Delta_1M": -25,
                "Breadth_50SMA_Pct": 32.0,
                "Tradable_Count": 8,
                "Is_Group_Confirmed": False,
                "Is_Danger_Group": True,
                "Group_Grade": "D",
                "Group_Grade_Badge": "D #110",
                "Top_Leaders_Full": [{"ticker": "DHOLD", "raw_ticker": "DHOLD.NS", "rs": 45, "is_tradable_leader": True, "dist_ema": -4.0, "status": "⏳ BASE"}],
                "Constituents": ["DHOLD.NS"],
            },
            {
                "Industry_Group": "Deep E Grp",
                "Rank_Today": 138,
                "Delta_1M": -52,
                "Breadth_50SMA_Pct": 14.0,
                "Tradable_Count": 10,
                "Is_Group_Confirmed": False,
                "Is_Danger_Group": True,
                "Group_Grade": "E",
                "Group_Grade_Badge": "E #138",
                "Top_Leaders_Full": [{"ticker": "EHOLD", "raw_ticker": "EHOLD.NS", "rs": 28, "is_tradable_leader": True, "dist_ema": -6.5, "status": "🔴 FAILED"}],
                "Constituents": ["EHOLD.NS", "EHOLD.BO"],
            },
        ]
    )
    ern = {
        "EHOLD": {"days_to_earnings": 2},
        "DHOLD": {"days_to_earnings": 12},
    }
    board = _compute_cluster_board_core(
        matrix_df=mat_df,
        thrust_res={},
        funda_map={},
        spons_map={},
        udvr_map={},
        funda_as_of="N/A",
        market="INDIA",
        portfolio_tickers=["EHOLD.NS", "EHOLD.BO", "DHOLD.NS"],
        earnings_cache=ern,
    )
    ov = board["overlap"]
    assert ov["held_total"] == 2
    assert ov["holdings_in_de"] == 2
    de_list = ov["de_holdings"]
    assert [x["ticker"] for x in de_list] == ["EHOLD", "DHOLD"]
    assert de_list[0]["ern_days"] == 2
    assert de_list[1]["ern_days"] is None
    assert "⚠️ ERN 2D" in de_list[0]["alert_reasons"]


def test_19_portfolio_triage_6_bucket_ledger_and_backfill_overlap():
    mat_df = pd.DataFrame(
        [
            {
                "Industry_Group": "Medical Top",
                "Rank_Today": 5,
                "Pure_Price_RS": 91.0,
                "Breadth_RS80_Pct": 55.0,
                "Breadth_50SMA_Pct": 82.0,
                "RSNH_Pct": 25.0,
                "Tradable_Count": 10,
                "Is_Group_Confirmed": True,
                "Group_Grade": "A+",
                "Group_Grade_Badge": "A+ #5",
                "Top_Leaders_Full": [
                    {"ticker": "TOPFAIL", "rs": 83, "is_tradable_leader": True, "dist_ema": -4.9, "status": "🔴 FAILED"},
                    {"ticker": "TOPLAG", "rs": 68, "is_tradable_leader": True, "dist_ema": 1.0, "status": "⏳ BASE"},
                    {"ticker": "TOPOK", "rs": 94, "is_tradable_leader": True, "dist_ema": 1.2, "status": "🟢 IN BUY ZONE"},
                ],
                "Constituents": ["TOPFAIL", "TOPLAG", "TOPOK"],
            },
            {
                "Industry_Group": "Backfilled D Pack",
                "Rank_Today": 120,
                "Pure_Price_RS": 89.0,
                "Breadth_RS80_Pct": 50.0,
                "Breadth_50SMA_Pct": 80.0,
                "RSNH_Pct": 22.0,
                "Tradable_Count": 6,
                "Is_Group_Confirmed": False,
                "Is_Danger_Group": True,
                "Group_Grade": "D",
                "Group_Grade_Badge": "D #120",
                "Top_Leaders_Full": [{"ticker": "BFDE", "rs": 60, "is_tradable_leader": True, "dist_ema": -2.0, "status": "⏳ BASE"}],
                "Constituents": ["BFDE"],
            },
            {
                "Industry_Group": "Strong Other AB",
                "Rank_Today": 18,
                "Pure_Price_RS": 65.0,
                "Breadth_RS80_Pct": 30.0,
                "Breadth_50SMA_Pct": 65.0,
                "RSNH_Pct": 12.0,
                "Tradable_Count": 8,
                "Is_Group_Confirmed": True,
                "Group_Grade": "B",
                "Group_Grade_Badge": "B #18",
                "Top_Leaders_Full": [{"ticker": "ABH", "rs": 84, "is_tradable_leader": True, "dist_ema": 1.0, "status": "⏳ BASE"}],
                "Constituents": ["ABH"],
            },
            {
                "Industry_Group": "Lagging C",
                "Rank_Today": 82,
                "Pure_Price_RS": 50.0,
                "Breadth_RS80_Pct": 20.0,
                "Breadth_50SMA_Pct": 45.0,
                "RSNH_Pct": 5.0,
                "Tradable_Count": 8,
                "Is_Group_Confirmed": False,
                "Group_Grade": "C",
                "Group_Grade_Badge": "C #82",
                "Top_Leaders_Full": [{"ticker": "CHOLD", "rs": 75, "is_tradable_leader": True, "dist_ema": 0.5, "status": "⏳ BASE"}],
                "Constituents": ["CHOLD"],
            },
        ]
    )
    niche_df = pd.DataFrame(
        [
            {
                "Industry_Group": "Small Niche Grp",
                "Group_Grade": "NR",
                "Top_Leaders_Full": [{"ticker": "NICHE1", "rs": 82, "is_tradable_leader": True, "dist_ema": 1.0, "status": "⏳ BASE"}],
                "Constituents": ["NICHE1"],
            }
        ]
    )
    board = _compute_cluster_board_core(
        matrix_df=mat_df,
        niche_df=niche_df,
        thrust_res={},
        funda_map={},
        spons_map={},
        udvr_map={},
        funda_as_of="N/A",
        market="INDIA",
        top_n=2,  # Includes "Medical Top" and "Backfilled D Pack" with confirmed_only=False while "Strong Other AB" stays outside Top-2
        confirmed_only=False,
        portfolio_tickers=["TOPOK", "TOPFAIL", "TOPLAG", "BFDE", "ABH", "CHOLD", "NICHE1", "GOLDETF"],
    )
    ov = board["overlap"]
    led = ov["ledger"]
    assert led["held_total"] == 8
    assert (
        led["top_clusters"]
        + led["ab_other"]
        + led["c_groups"]
        + led["de_danger"]
        + led["niche"]
        + led["unmapped"]
    ) == led["held_total"]
    # BFDE counts in de_danger (priority over top_clusters) even though its group was backfilled onto board
    assert led["de_danger"] == 1
    assert led["top_clusters"] == 3
    assert led["ab_other"] == 1
    assert led["c_groups"] == 1
    assert led["niche"] == 1
    assert led["unmapped"] == 1
    assert [x["ticker"] for x in ov["de_holdings"]] == ["BFDE"]
    weak_map = {x["ticker"]: x for x in ov["held_weak_alerts"]}
    assert "TOPFAIL" in weak_map and "🛑 FAILED" in weak_map["TOPFAIL"]["alert_reasons"]
    assert "TOPLAG" in weak_map and any("RS 68" in r for r in weak_map["TOPLAG"]["alert_reasons"])


def test_20_tier0_illiquid_and_circuit_held_never_hidden_from_triage():
    mat_df = pd.DataFrame(
        [
            {
                "Industry_Group": "Chemicals",
                "Rank_Today": 6,
                "Pure_Price_RS": 88.0,
                "Breadth_RS80_Pct": 50.0,
                "Breadth_50SMA_Pct": 80.0,
                "RSNH_Pct": 25.0,
                "Tradable_Count": 10,
                "Is_Group_Confirmed": True,
                "Group_Grade": "A+",
                "Group_Grade_Badge": "A+ #6",
                "Top_Leaders_Full": [
                    {"ticker": "GOODL", "rs": 94, "is_tradable_leader": True, "circuit_days_20d": 0, "dist_ema": 1.0, "status": "🟢 IN BUY ZONE"},
                    {"ticker": "THIN_SME", "rs": 92, "is_tradable_leader": False, "circuit_days_20d": 0, "dist_ema": 1.0, "status": "⏳ BASE"},
                    {"ticker": "CIRC_TRAP", "rs": 91, "is_tradable_leader": True, "circuit_days_20d": 4, "dist_ema": 2.0, "status": "⏳ BASE"},
                ],
                "Constituents": ["GOODL", "THIN_SME", "CIRC_TRAP"],
            }
        ]
    )
    board = _compute_cluster_board_core(
        matrix_df=mat_df,
        thrust_res={},
        funda_map={},
        spons_map={},
        udvr_map={},
        funda_as_of="N/A",
        market="INDIA",
        portfolio_tickers=["THIN_SME", "CIRC_TRAP"],
    )
    assert [l["ticker"] for l in board["clusters"][0]["leaders"]] == ["GOODL"]
    weak_map = {x["ticker"]: x for x in board["overlap"]["held_weak_alerts"]}
    assert "🚫 Tier-0 Illiquid" in weak_map["THIN_SME"]["alert_reasons"]
    assert "🔒 Circuit Risk" in weak_map["CIRC_TRAP"]["alert_reasons"]


def test_21_constituents_fallback_and_us_dot_hyphen_aliases():
    mat_df = pd.DataFrame(
        [
            {
                "Industry_Group": "Insurance - Legacy",
                "Rank_Today": 125,
                "Breadth_50SMA_Pct": 25.0,
                "Delta_1M": -30,
                "Is_Danger_Group": True,
                "Group_Grade": "E",
                "Group_Grade_Badge": "E #125",
                "Top_Leaders_Full": [],
                "Constituents": ["BRK-B", "FALLBACK1"],
            }
        ]
    )
    board = _compute_cluster_board_core(
        matrix_df=mat_df,
        thrust_res={},
        funda_map={},
        spons_map={},
        udvr_map={},
        funda_as_of="N/A",
        market="US_IXIC",
        confirmed_only=False,
        portfolio_tickers=["BRK.B", "FALLBACK1"],
    )
    ov = board["overlap"]
    assert ov["held_total"] == 2
    assert ov["holdings_in_de"] == 2
    de_map = {x["ticker"]: x for x in ov["de_holdings"]}
    assert "BRK.B" in de_map and de_map["BRK.B"]["rs"] is None
    assert "FALLBACK1" in de_map and "E #125" in de_map["FALLBACK1"]["alert_reasons"][0]


def test_22_sticky_igm_pin_retains_de_group_across_frame2_and_resets():
    now_t = 1_700_000_000.0
    sess = {"kt_igm_jump_in": build_igm_jump_payload("IN", "Airlines D-Graded", now_ts=now_t - 5.0)}
    jump1 = resolve_igm_jump_payload(sess, "in", now_ts=now_t)
    assert jump1 is not None and "kt_igm_jump_in" not in sess

    df_matrix = pd.DataFrame(
        [
            {"Industry_Group": "Banks - Top", "Rank_Today": 1},
            {"Industry_Group": "Airlines D-Graded", "Rank_Today": 130},
        ]
    )
    filtered_top = df_matrix.head(1).copy()

    # Frame 1 pins Airlines D-Graded despite Top-N filter
    f1, p1, n1 = apply_igm_jump_with_sticky(sess, "in", filtered_top, df_matrix, jump1, "sig_A", now_ts=now_t)
    assert p1 == "Airlines D-Graded"
    assert "Airlines D-Graded" in f1["Industry_Group"].values
    assert n1 and "pinned" in n1
    sess["igm_selected_group_in"] = p1

    # Frame 2 (interaction inside drilled D/E group -> jump=None, identical sig_A) stays pinned
    f2, p2, n2 = apply_igm_jump_with_sticky(sess, "in", filtered_top, df_matrix, None, "sig_A", now_ts=now_t + 30.0)
    assert p2 == "Airlines D-Graded"
    assert "Airlines D-Graded" in f2["Industry_Group"].values
    assert n2 is None

    # Frame 3 with changed filter_sig clears sticky pin cleanly
    f3, p3, _ = apply_igm_jump_with_sticky(sess, "in", filtered_top, df_matrix, None, "sig_B", now_ts=now_t + 40.0)
    assert p3 is None and "Airlines D-Graded" not in f3["Industry_Group"].values
    assert "kt_igm_sticky_pin_in" not in sess


def test_23_ui_card_html_escaping_and_de_drill_key_dedup():
    card_partial = _render_cluster_card_html(
        {
            "board_rank": 2,
            "industry_group": "Electronics & Computer Distribution",
            "grade_badge": "A+ #4",
            "rank_path": "#1 -> #4",
            "persistence_tag": "Established",
            "cluster_score": 81.4,
            "actionable_cnt": 1,
            "suppressed_laggards": 2,
            "suppressed_failed": 1,
            "leaders": [{"ticker": "CML", "tv_url": "#", "tier_badge": "🚀 Growth RS", "rs": 98, "has_funda": False, "status": "⏳ BASE", "dist_ema": 1.0}],
        }
    )
    assert "Electronics &amp; Computer Distribution" in card_partial
    assert "Only 1 RS≥80 healthy leader(s)" in card_partial
    assert "2 &lt;RS80 · 1 failed/damaged hidden" in card_partial
    assert not any(ln.startswith("    ") for ln in card_partial.splitlines() if ln.strip())

    card_empty = _render_cluster_card_html(
        {
            "board_rank": 3,
            "industry_group": "Hollow Pack",
            "grade_badge": "A #5",
            "suppressed_laggards": 3,
            "suppressed_failed": 0,
            "leaders": [],
        }
    )
    assert "No healthy RS≥80 leaders in group" in card_empty

    de_items = [
        {"industry_group": "Auto & Parts <Legacy>", "grade_badge": "E #140", "rank_today": 140, "ticker": "A1", "tv_url": "#", "alert_reasons": ["🚨 E #140"]},
        {"industry_group": "Auto & Parts <Legacy>", "grade_badge": "E #140", "rank_today": 140, "ticker": "A2", "tv_url": "#", "alert_reasons": ["🚨 E #140"]},
        {"industry_group": "Auto & Parts <Legacy>", "grade_badge": "E #140", "rank_today": 140, "ticker": "A3", "tv_url": "#", "alert_reasons": ["🚨 E #140"]},
        {"industry_group": "Already On Board", "grade_badge": "D #115", "rank_today": 115, "ticker": "B1", "tv_url": "#", "alert_reasons": ["🚨 D #115"]},
    ]
    drills = _extract_unique_de_drill_groups(de_items, [{"industry_group": "Already On Board"}], pfx="clb_in", max_groups=4)
    assert len(drills) == 1
    assert drills[0]["industry_group"] == "Auto & Parts <Legacy>"
    assert drills[0]["button_key"].startswith("clb_in_de_drill_")
    tbl_html = _render_portfolio_triage_table_html(de_items[:1], [])
    assert "Auto &amp; Parts &lt;Legacy&gt;" in tbl_html


def test_24_min_sales_filter_exempts_zero_and_missing_funda():
    group_row = {
        "Top_Leaders_Full": [
            {"ticker": "HIGH_SALES", "rs": 95, "is_tradable_leader": True, "dist_ema": 1.5, "status": "🟢 IN BUY ZONE"},
            {"ticker": "EXACT_20", "rs": 92, "is_tradable_leader": True, "dist_ema": 1.0, "status": "🔵 RETEST"},
            {"ticker": "ZERO_SALES", "rs": 90, "is_tradable_leader": True, "dist_ema": 2.0, "status": "⏳ COILING"},
            {"ticker": "MISS_FUNDA", "rs": 88, "is_tradable_leader": True, "dist_ema": 1.2, "status": "⏳ BASE"},
            {"ticker": "LOW_SALES_12", "rs": 96, "is_tradable_leader": True, "dist_ema": 1.0, "status": "🟢 IN BUY ZONE"},
            {"ticker": "NEG_SALES", "rs": 94, "is_tradable_leader": True, "dist_ema": 1.0, "status": "🟢 IN BUY ZONE"},
        ]
    }
    funda_map = {
        "HIGH_SALES": {"eps_growth": 40.0, "sales_growth": 35.0, "roe": 20.0},
        "EXACT_20": {"eps_growth": 30.0, "sales_growth": 20.0, "roe": 18.0},
        "ZERO_SALES": {"eps_growth": 25.0, "sales_growth": 0.0, "roe": 18.0},
        "MISS_FUNDA": {"eps_growth": 0.0, "sales_growth": 0.0, "roe": 0.0},
        "LOW_SALES_12": {"eps_growth": 23.0, "sales_growth": 12.0, "roe": 14.0},
        "NEG_SALES": {"eps_growth": 30.0, "sales_growth": -8.0, "roe": 15.0},
    }

    selected, _, _, stats = _select_cluster_tml_leaders(
        group_row=group_row,
        is_us=False,
        funda_map=funda_map,
        spons_map={"HIGH_SALES": {"grade": "A+", "inst_change": 1.0}, "EXACT_20": {"grade": "A", "inst_change": 0.8}},
        udvr_map={"HIGH_SALES": 1.6, "EXACT_20": 1.6},
        thrust_ticker_map={},
        held_set=set(),
        earnings_cache=None,
        brain_score_map={},
        us_tv_map=None,
        max_leaders=6,
        min_sales_pct=20.0,
    )
    kept_syms = {s["ticker"] for s in selected}
    assert kept_syms == {"HIGH_SALES", "EXACT_20", "ZERO_SALES", "MISS_FUNDA"}
    assert stats["suppressed_low_sales"] == 2

    # When min_sales_pct=0.0 (filter OFF), all 6 return
    all_sel, _, _, all_stats = _select_cluster_tml_leaders(
        group_row=group_row,
        is_us=False,
        funda_map=funda_map,
        spons_map={},
        udvr_map={},
        thrust_ticker_map={},
        held_set=set(),
        earnings_cache=None,
        brain_score_map={},
        us_tv_map=None,
        max_leaders=6,
        min_sales_pct=0.0,
    )
    assert len(all_sel) == 6
    assert all_stats["suppressed_low_sales"] == 0

    # End-to-end board + tv_watchlist + UI omitted-count formatting
    mat_df = pd.DataFrame(
        [
            {
                "Industry_Group": "Mixed Sales Pack",
                "Rank_Today": 4,
                "Pure_Price_RS": 90.0,
                "Breadth_RS80_Pct": 50.0,
                "Breadth_50SMA_Pct": 80.0,
                "RSNH_Pct": 25.0,
                "Tradable_Count": 6,
                "Is_Group_Confirmed": True,
                "Group_Grade": "A+",
                "Group_Grade_Badge": "A+ #4",
                "Top_Leaders_Full": group_row["Top_Leaders_Full"],
                "Constituents": [x["ticker"] for x in group_row["Top_Leaders_Full"]],
            }
        ]
    )
    board = _compute_cluster_board_core(
        matrix_df=mat_df,
        thrust_res={},
        funda_map=funda_map,
        spons_map={},
        udvr_map={},
        funda_as_of="N/A",
        market="INDIA",
        min_sales_pct=20.0,
    )
    cl = board["clusters"][0]
    assert cl["suppressed_low_sales"] == 2
    assert "LOW_SALES_12" not in board["tv_watchlist"] and "NEG_SALES" not in board["tv_watchlist"]
    html = _render_cluster_card_html(cl | {"leaders": cl["leaders"][:2]})
    assert "2 &lt;20% sales hidden" in html
