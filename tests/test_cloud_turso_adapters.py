"""
Cloud Turso Adapter, Cold-Boot Hydration & Memory Audit Tests (Gates 3, 4, 6).
All filesystem and SQLite state uses pytest `tmp_path` + `monkeypatch` so real
workspace `.cache/`, `.pkl`, and `.db` files are never mutated.
"""

from __future__ import annotations

import os
import sqlite3
import time

import numpy as np
import pandas as pd
import pytest

import breakout_cohort_engine
import breadth_engine
import brain_synthesis_engine
import components
import database
import industry_group_engine
import leadership_cluster_engine
import leadership_cluster_ui


def _get_process_rss_mb() -> float:
    """Return current process resident set size (MB) using psutil or Windows API."""
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        get_cur = ctypes.windll.kernel32.GetCurrentProcess
        get_cur.restype = wintypes.HANDLE
        hproc = get_cur()
        fn = getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None) or ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        if fn(hproc, ctypes.byref(counters), counters.cb):
            return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        pass
    return 0.0


def _clear_streamlit_caches() -> None:
    for fn in (
        getattr(database, "load_cloud_json_cache_cached", None),
        getattr(industry_group_engine, "_load_json_payload_cached", None),
        getattr(industry_group_engine, "_get_group_constituents_map_cached", None),
        getattr(industry_group_engine, "compute_industry_group_matrix_with_niche", None),
        getattr(leadership_cluster_engine, "_load_fundamentals_and_sponsorship_cached", None),
        getattr(leadership_cluster_engine, "_load_udvr_map_cached", None),
        getattr(leadership_cluster_engine, "_cached_thrusting_industries_loader", None),
        getattr(breadth_engine, "compute_leader_21ema_table", None),
        getattr(breakout_cohort_engine, "compute_breakout_health_radar", None),
        getattr(components, "_load_cached_stage_distribution_inner", None),
    ):
        clear_fn = getattr(fn, "clear", None)
        if callable(clear_fn):
            try:
                clear_fn()
            except Exception:
                pass


@pytest.fixture
def isolated_cloud_env(tmp_path, monkeypatch):
    """Set up an isolated SQLite DB and empty Cloud working directory inside `tmp_path`."""
    db_file = str(tmp_path / "cloud_test.db")
    monkeypatch.setattr(database, "DATABASE_PATH", db_file)
    monkeypatch.setattr(database, "get_connection", lambda: sqlite3.connect(db_file))

    cache_dir = str(tmp_path / ".cache")
    monkeypatch.setattr(industry_group_engine, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(industry_group_engine, "CACHE_INTEL_INDIA_PATH", str(tmp_path / ".cache" / "industry_group_intelligence_india.json"))
    monkeypatch.setattr(industry_group_engine, "CACHE_INTEL_US_PATH", str(tmp_path / ".cache" / "industry_group_intelligence_us.json"))
    monkeypatch.setattr(industry_group_engine, "CACHE_MATRIX_INDIA_PATH", str(tmp_path / ".cache" / "industry_group_matrix_india.json"))
    monkeypatch.setattr(industry_group_engine, "CACHE_MATRIX_US_PATH", str(tmp_path / ".cache" / "industry_group_matrix_us.json"))
    monkeypatch.setattr(industry_group_engine, "CACHE_NICHE_INDIA_PATH", str(tmp_path / ".cache" / "industry_group_niche_india.json"))
    monkeypatch.setattr(industry_group_engine, "CACHE_NICHE_US_PATH", str(tmp_path / ".cache" / "industry_group_niche_us.json"))
    monkeypatch.setattr(industry_group_engine, "MATRIX_PKL_PATH", str(tmp_path / "missing_historical_prices_matrix.pkl"))

    monkeypatch.chdir(tmp_path)
    _clear_streamlit_caches()
    database.init_database()
    yield tmp_path
    _clear_streamlit_caches()


def _sample_igm_bundle(is_us: bool) -> dict:
    group_name = "Semiconductors" if is_us else "Power Infrastructure"
    sym_prefix = "" if is_us else ".NS"
    leaders = []
    # Include >50 stock aliases so get_stock_group_intelligence (>50 keys) recognizes hydrated snapshot
    constituents = []
    for idx in range(1, 20):
        tk = f"{'NV' if is_us else 'PWR'}{idx:02d}"
        raw_tk = f"{tk}{sym_prefix}"
        constituents.append(raw_tk)
        leaders.append(
            {
                "ticker": tk,
                "raw_ticker": raw_tk,
                "rs": 95 - idx if idx <= 10 else 68,
                "adtv": 50.0,
                "dist_ema": 1.2,
                "dist_pivot": -0.5,
                "price": 500.0,
                "status": "ACTIONABLE" if idx <= 3 else "EXTENDED",
                "bars": 250,
                "is_tradable_leader": True,
                "is_short_history": False,
                "is_fresh_stage2": True,
                "stage_code": "S2_FRESH",
                "is_near_52wh": True,
                "is_blue_dot": idx == 1,
                "is_rsnh": idx <= 3,
                "role_code": "ALPHA" if idx == 1 else ("SISTER" if idx == 2 else "PACK_LEADER"),
                "role_title": "Alpha Leader" if idx == 1 else "Pack Leader",
                "role_icon": "👑",
            }
        )
    df_m = pd.DataFrame(
        [
            {
                "Cache_Schema_Version": industry_group_engine.CACHE_SCHEMA_VERSION,
                "Industry_Group": group_name,
                "Rank_Today": 1,
                "Rank_1W": 3,
                "Rank_3W": 8,
                "Rank_6W": 15,
                "Rank_1M": 9,
                "Rank_3M": 22,
                "Delta_1W": 2,
                "Delta_1M": 8,
                "Delta_3M": 21,
                "Total_Groups": 140,
                "Group_Grade": "A+",
                "Group_Grade_Style": "elite",
                "Group_Grade_Badge": "A+",
                "Is_Top_20_Pct": True,
                "Is_Danger_Group": False,
                "Is_Group_Confirmed": True,
                "Is_Dual_Alpha": True,
                "Group_Confirmation_Badge": "Confirmed",
                "Group_Rotation_Status": "Surging",
                "Rotation_Badge": "Surging",
                "Breadth_RS80_Pct": 72.0,
                "Breadth_50SMA_Pct": 84.0,
                "RSNH_Pct": 45.0,
                "Near_52WH_Pct": 60.0,
                "RS_BlueDot_Pct": 25.0,
                "CANSLIM_Funda_Pct": 82.5,
                "Tradable_Count": len(constituents),
                "Constituents": constituents,
                "Top_Leaders_Full": leaders,
                "Alpha_Leader": leaders[0]["ticker"],
                "Sister_Stock": leaders[1]["ticker"],
            }
        ]
    )
    df_n = pd.DataFrame(
        [
            {
                "Cache_Schema_Version": industry_group_engine.CACHE_SCHEMA_VERSION,
                "Industry_Group": "Specialty Niche Group",
                "Rank_Today": 0,
                "Group_Grade": "B",
                "Group_Grade_Style": "neutral",
                "Group_Grade_Badge": "NR (3 Stocks)",
                "Is_Top_20_Pct": False,
                "Is_Danger_Group": False,
                "Is_Group_Confirmed": False,
                "Group_Confirmation_Badge": "Unconfirmed",
                "Group_Rotation_Status": "Watch",
                "Delta_1M": 0,
                "Constituents": [f"{'NC' if is_us else 'NCH'}{i}{sym_prefix}" for i in range(1, 15)],
                "Top_Leaders_Full": [
                    {
                        "ticker": f"{'NC' if is_us else 'NCH'}{i}",
                        "raw_ticker": f"{'NC' if is_us else 'NCH'}{i}{sym_prefix}",
                        "rs": 81,
                        "bars": 200,
                        "is_tradable_leader": True,
                        "role_code": "ALPHA" if i == 1 else "PACK_LEADER",
                        "role_title": "Niche Leader",
                        "role_icon": "⭐",
                    }
                    for i in range(1, 15)
                ],
            }
        ]
    )
    bench = pd.Series(
        [100.0, 102.5, 105.0, 107.2],
        index=pd.date_range("2026-11-10", periods=4, freq="D"),
        name="Benchmark",
    )
    return {"matrix_df": df_m, "benchmark_curve": bench, "niche_df": df_n}


def _sample_thrust_bundle(is_us: bool) -> dict:
    grp = "Semiconductors" if is_us else "Power Infrastructure"
    sym1 = "NV01" if is_us else "PWR01"
    sym2 = "NV02" if is_us else "PWR02"
    all_df = pd.DataFrame(
        [
            {
                "Industry_Group": grp,
                "Rank_Today": 1,
                "Group_Grade": "A+",
                "Is_Top_20_Pct": True,
                "Is_Counter_Trend_Bounce": False,
                "Breadth_RS80_Pct": 72.0,
                "Up4_Today_Cnt": 2,
                "Up4_Today_S2_Cnt": 2,
                "Up4_Today_Share_Pct": 40.0,
                "Up4_5D_Unique": 5,
                "Up25_Qtr_Cnt": 4,
                "NewHigh_5D_Unique": 6,
                "NewHigh_Today_Cnt": 3,
                "Thrust_Score": 94.0,
                "All_Thrusting_Tickers": [
                    {"ticker": sym1, "up4_today": True, "up4_5d_cnt": 2, "dist_21ema": 1.2},
                    {"ticker": sym2, "up4_today": True, "up4_5d_cnt": 1, "dist_21ema": 0.4},
                ],
                "Thrusting_Tickers": [
                    {"ticker": sym1, "up4_today": True},
                    {"ticker": sym2, "up4_today": True},
                ],
            },
            {
                "Industry_Group": "Specialty Software" if is_us else "Defense Electronics",
                "Rank_Today": 8,
                "Group_Grade": "A",
                "Is_Top_20_Pct": True,
                "Is_Counter_Trend_Bounce": False,
                "Breadth_RS80_Pct": 60.0,
                "Up4_Today_Cnt": 1,
                "Up4_Today_S2_Cnt": 1,
                "Up4_Today_Share_Pct": 20.0,
                "Up4_5D_Unique": 3,
                "Up25_Qtr_Cnt": 2,
                "NewHigh_5D_Unique": 2,
                "NewHigh_Today_Cnt": 1,
                "Thrust_Score": 76.0,
                "All_Thrusting_Tickers": [],
                "Thrusting_Tickers": [],
            },
        ]
    )
    return {
        "thrust_df": all_df.copy(),
        "all_thrust_df": all_df,
        "distribution_df": pd.DataFrame([{"Industry_Group": "Lagging Utilities", "Dn4_Today_Cnt": 2}]),
        "ledger": {"grid_up4_total": 5, "attributed": 3, "illiquid": 0, "quarantined": 0, "unmapped": 2},
        "summary": {
            "total_up4_today": 5,
            "total_dn4_today": 1,
            "valid_universe_cnt": 500,
            "top3_share_pct": 60.0,
            "top3_names": [grp],
            "headline_msg": "Thrust test",
            "used_fallback_note": "",
            "t2108": 58.0,
        },
        "as_of": {"prices_date": "2026-11-18", "groups_date": "2026-11-18", "igm_stale": False, "is_provisional": False},
    }


def test_gate3_dual_schema_fundamentals_coalescing(isolated_cloud_env):
    conn = database.get_connection()
    cur = conn.cursor()
    # Row 1: populated via legacy KTL columns (eps_yoy, sales_yoy)
    cur.execute(
        """
        INSERT INTO fundamentals_cache
        (ticker, eps_yoy, sales_yoy, eps_growth, sales_growth, roe, market_cap, industry, updated_at)
        VALUES (?, ?, ?, NULL, NULL, ?, ?, ?, ?)
        """,
        ("PWR01.NS", 42.5, 31.0, 21.0, 5e10, "Power Infrastructure", "2026-11-18T18:00:00"),
    )
    # Row 2: populated via KT/sync columns (eps_growth, sales_growth)
    cur.execute(
        """
        INSERT INTO fundamentals_cache
        (ticker, eps_yoy, sales_yoy, eps_growth, sales_growth, roe, market_cap, industry, updated_at)
        VALUES (?, NULL, NULL, ?, ?, ?, ?, ?, ?)
        """,
        ("NV01", 42.5, 31.0, 24.0, 9e11, "Semiconductors", "2026-11-18T21:00:00"),
    )
    conn.commit()
    conn.close()

    in_map, latest_in = database.get_fundamentals_map_for_engine("IN")
    us_map, latest_us = database.get_fundamentals_map_for_engine("US")
    assert latest_in == "2026-11-18"
    assert latest_us == "2026-11-18"

    assert in_map["PWR01.NS"]["eps_growth"] == pytest.approx(42.5)
    assert in_map["PWR01"]["sales_growth"] == pytest.approx(31.0)
    assert us_map["NV01"]["eps_growth"] == pytest.approx(42.5)
    assert us_map["NV01"]["sales_growth"] == pytest.approx(31.0)
    assert "NV01" not in in_map
    assert "PWR01.NS" not in us_map


def test_gate3_cloud_json_cache_roundtrip_df_attrs_and_series(isolated_cloud_env):
    df_orig = pd.DataFrame(
        [
            {"Ticker": "NVDA", "RS_Rating": 98, "Meta": [{"k": 1, "v": "alpha"}]},
            {"Ticker": "AVGO", "RS_Rating": 94, "Meta": [{"k": 2, "v": "beta"}]},
        ]
    )
    df_orig.attrs["diagnostics"] = {
        "matrix_last_dt": "2026-11-18",
        "bench_stale_days": 0,
        "universe_scope": "ALL",
    }
    series_orig = pd.Series(
        [100.0, 101.25, 104.75],
        index=pd.date_range("2026-11-15", periods=3, freq="D"),
        name="^IXIC",
    )
    payload = {
        "matrix_df": df_orig,
        "benchmark_curve": series_orig,
        "nested_list": [{"ticker": "NVDA", "rs": 98}],
    }

    assert database.save_cloud_json_cache("roundtrip_test", "US", payload, as_of_date="2026-11-18") is True
    loaded, as_of = database.load_cloud_json_cache("roundtrip_test")
    assert as_of == "2026-11-18"
    assert isinstance(loaded, dict)
    assert isinstance(loaded["matrix_df"], pd.DataFrame)
    assert loaded["matrix_df"].attrs.get("diagnostics") == df_orig.attrs["diagnostics"]
    assert loaded["matrix_df"].iloc[0]["Meta"] == [{"k": 1, "v": "alpha"}]
    assert isinstance(loaded["benchmark_curve"], pd.Series)
    assert loaded["benchmark_curve"].name == "^IXIC"
    assert np.allclose(loaded["benchmark_curve"].values, series_orig.values)


def test_gate4_and_gate6_empty_disk_cloud_cold_boot_and_rss(isolated_cloud_env):
    # Populate temporary Turso-emulated tables
    in_bundle = _sample_igm_bundle(is_us=False)
    us_bundle = _sample_igm_bundle(is_us=True)
    database.save_cloud_json_cache("igm_bundle_in", "IN", in_bundle, "2026-11-18")
    database.save_cloud_json_cache("igm_thematic_in", "IN", in_bundle, "2026-11-18")
    database.save_cloud_json_cache("igm_bundle_us", "US", us_bundle, "2026-11-18")

    th_in = _sample_thrust_bundle(is_us=False)
    th_us = _sample_thrust_bundle(is_us=True)
    database.save_cloud_json_cache("stockbee_thrust_in_full", "IN", th_in, "2026-11-18")
    database.save_cloud_json_cache("stockbee_thrust_in_ntm750", "IN", th_in, "2026-11-18")
    database.save_cloud_json_cache("stockbee_thrust_us_broad", "US", th_us, "2026-11-18")
    database.save_cloud_json_cache("stockbee_thrust_us_sp500", "US", th_us, "2026-11-18")

    database.save_cloud_json_cache("udvr_map_in", "IN", {"PWR01.NS": 1.85, "PWR02.NS": 1.62}, "2026-11-18")
    database.save_cloud_json_cache("udvr_map_us", "US", {"NV01": 1.92, "NV02": 1.55}, "2026-11-18")

    sb_mon_in = pd.DataFrame([{"Date": "2026-11-18", "Pulse": "🚀 5D THRUST", "Ratio_5D": 2.45}])
    sb_mon_us = pd.DataFrame([{"Date": "2026-11-18", "Pulse": "🚀 5D THRUST", "Ratio_5D": 2.15}])
    database.save_cloud_json_cache("stockbee_monitor_in_ntm750", "IN", sb_mon_in, "2026-11-18")
    database.save_cloud_json_cache("stockbee_monitor_us_sp500", "US", sb_mon_us, "2026-11-18")

    conn = database.get_connection()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO fundamentals_cache (ticker, eps_growth, sales_growth, roe, market_cap, industry, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("PWR01.NS", 38.0, 29.0, 22.0, 4e10, "Power Infrastructure", "2026-11-18"),
            ("PWR02.NS", 30.0, 12.0, 19.0, 3e10, "Power Infrastructure", "2026-11-18"),
            ("NV01", 55.0, 40.0, 28.0, 8e11, "Semiconductors", "2026-11-18"),
        ],
    )
    conn.commit()
    conn.close()

    radar_df = pd.DataFrame(
        [{"Ticker": f"PWR{i:02d}", "RS_Rating": 99 - i} for i in range(1, 301)]
    )
    radar_df.attrs["diagnostics"] = {"matrix_last_dt": "2026-11-18", "market": "IN"}
    database.save_cloud_json_cache("leader_radar_in_750", "IN", radar_df, "2026-11-18")

    aux_in = {
        "alpha": {"ret_n500_1m": 3.5},
        "top_rs_breadth": pd.DataFrame([{"date": "2026-11-18", "top100_above_50_sma": 74.0}]),
        "sector_breadth": pd.DataFrame([{"industry": "Power Infrastructure", "above_50_pct": 85.0}]),
        "breakout_cohort": [{"health_pct": 66.0, "total_breakouts": 12}, pd.DataFrame([{"Ticker": "PWR01", "Status": "Runner"}])],
        "stage_dist": {"s1_count": 100, "s2_count": 250, "s3_count": 80, "s4_count": 70, "total": 500, "s1_pct": 20.0, "s2_pct": 50.0, "s3_pct": 16.0, "s4_pct": 14.0},
    }
    aux_us = {
        "alpha": {"ret_sp500_1m": 4.2},
        "top_rs_breadth": pd.DataFrame([{"date": "2026-11-18", "top100_above_50_sma": 78.0}]),
        "sector_breadth": pd.DataFrame([{"sector": "Technology", "above_50_pct": 80.0}]),
        "breakout_cohort": [{"health_pct": 70.0, "total_breakouts": 15}, pd.DataFrame([{"Ticker": "NV01", "Status": "Runner"}])],
        "stage_dist": {"s1_count": 90, "s2_count": 260, "s3_count": 75, "s4_count": 75, "total": 500, "s1_pct": 18.0, "s2_pct": 52.0, "s3_pct": 15.0, "s4_pct": 15.0},
        "us_scopes": {"sp500": pd.DataFrame([{"above_50_pct": 71.0}]), "ndx100": pd.DataFrame([{"above_50_pct": 79.0}])},
    }
    database.save_cloud_json_cache("breadth_aux_bundle_in", "IN", aux_in, "2026-11-18")
    database.save_cloud_json_cache("breadth_aux_bundle_us", "US", aux_us, "2026-11-18")

    database.save_brain_synthesis_cache(
        date_str="2026-11-18",
        market="INDIA",
        matrix_bar_date="2026-11-18",
        universe_count=750,
        tide_turn_data={"stage_num": 5, "stage_badge": "Stage 5: Confirmed Bull"},
        confluence_list=[{"clean_ticker": "PWR01", "brain_score": 92}],
    )

    # 1. Cold-boot hydration speed (< 100ms per market)
    t0 = time.perf_counter()
    industry_group_engine._hydrate_igm_cache_from_turso_if_missing("India (NSE/BSE)")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[Gate 4] Hydration elapsed: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 250.0  # local SQLite hydration should easily complete well under 100ms
    assert os.path.exists(industry_group_engine.CACHE_MATRIX_INDIA_PATH)
    assert os.path.exists(industry_group_engine.CACHE_INTEL_INDIA_PATH)
    assert len(industry_group_engine.get_stock_group_intelligence("India (NSE/BSE)")) >= 50

    # 2. Matrix & Thematic fallback
    df_in, b_in, n_in = industry_group_engine.compute_industry_group_matrix_with_niche("canonical", "India (NSE/BSE)")
    df_th, b_th, _ = industry_group_engine.compute_industry_group_matrix_with_niche("thematic", "India (NSE/BSE)")
    df_us, b_us, n_us = industry_group_engine.compute_industry_group_matrix_with_niche("canonical", "US (S&P 500 / Nasdaq)")
    assert not df_in.empty and isinstance(b_in, pd.Series) and not n_in.empty
    assert not df_th.empty and not b_th.empty
    assert not df_us.empty and isinstance(b_us, pd.Series) and not n_us.empty

    # 3. Stockbee Thrusting Industries across all 4 modes
    for mode in ("5D_AND_TODAY", "TODAY_4PCT", "QTR_25PCT", "NEW_HIGHS"):
        res_m = leadership_cluster_engine.compute_stockbee_thrusting_industries(
            market="IN", universe_scope="NTM750", mode=mode, top_n=6
        )
        assert not res_m["thrust_df"].empty
        assert res_m["summary"]["headline_msg"]

    # 4. Cluster Leadership Board (empty sponsorship => inst_accum_pct is None & 100% can_f, RS >= 80, 21E >= -5%, Min Sales %)
    board_in = leadership_cluster_engine.compute_cluster_leadership_board("INDIA", min_sales_pct=20.0)
    board_us = leadership_cluster_engine.compute_cluster_leadership_board("US_IXIC", min_sales_pct=20.0)
    assert board_in["status"] == "ok" and len(board_in["clusters"]) >= 1
    assert board_us["status"] == "ok" and len(board_us["clusters"]) >= 1
    assert board_in["clusters"][0]["inst_accum_pct"] is None
    in_leader_tks = [ldr["ticker"] for ldr in board_in["clusters"][0]["leaders"]]
    assert "PWR01" in in_leader_tks
    assert "PWR02" not in in_leader_tks  # filtered out by Min Sales % >= 20 gate

    # 5. Master RS Leader Radar Top-300 + attrs["diagnostics"]
    ldr_df = breadth_engine.compute_leader_21ema_table(market="IN", top_n=300, universe_scope="750")
    assert len(ldr_df) == 300
    assert ldr_df.attrs.get("diagnostics", {}).get("matrix_last_dt") == "2026-11-18"

    # 6. Breadth aux sub-panels & stage dist
    assert not breadth_engine.compute_top_rs_breadth_history(market="IN").empty
    assert not breadth_engine.compute_sector_breadth_matrix_in().empty
    assert not breadth_engine.compute_sector_breadth_matrix_us().empty
    bh_m, bh_df = breakout_cohort_engine.compute_breakout_health_radar(scope="IN")
    assert bh_m["health_pct"] == 66.0
    assert len(bh_df) == 1
    assert components.load_cached_stage_distribution_summary("IN")["total"] == 500
    assert components.load_cached_stage_distribution_summary("US")["s2_count"] == 260

    # 7. UI target page resolver (points to KTL views/ directory)
    views_dir = os.path.join(os.path.dirname(os.path.abspath(leadership_cluster_ui.__file__)), "views")
    expected_in = "views/industry_group_matrix.py" if os.path.isdir(views_dir) else "pages/industry_group_matrix.py"
    expected_us = "views/industry_group_matrix_us.py" if os.path.isdir(views_dir) else "pages/industry_group_matrix_us.py"
    assert leadership_cluster_ui._resolve_igm_target_page(False) == expected_in
    assert leadership_cluster_ui._resolve_igm_target_page(True) == expected_us

    # 8. Brain synthesis cloud fast return without .pkl
    br = brain_synthesis_engine.run_artificial_brain_synthesis(market="INDIA")
    assert br.get("from_cache") is True
    assert br["tide_turn"]["stage_num"] == 5

    # Gate 6: Resident memory footprint stays well below Streamlit Community Cloud 1 GB limit
    rss_mb = _get_process_rss_mb()
    print(f"[Gate 6] Cold-boot simulation RSS: {rss_mb:.1f} MB")
    assert rss_mb > 0.0, "Process RSS measurement failed (returned 0.0 MB)"
    assert rss_mb < 400.0, f"Process RSS too high on cold-boot simulation: {rss_mb:.1f} MB"
