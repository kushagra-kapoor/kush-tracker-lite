import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime
from database import (
    init_database, save_global_regime, 
    save_global_etf_momentum, save_global_sovereign_returns,
    save_cloud_json_cache
)
from market_data import fetch_nifty_total_market_tickers
from price_history_manager import fetch_incremental_history
from liquidity_engine import compute_liquidity
from breadth_engine import sync_breadth_history
from macro_regime_engine import (
    standardize_columns, calculate_distribution_days, 
    detect_follow_through_day, detect_change_of_character, get_market_regime_label
)

def get_tickers_from_file(filename):
    if not os.path.exists(filename):
        return []
    with open(filename, 'r') as f:
        return [line.strip().upper() for line in f if line.strip()]

def sync_all_data():
    print("Starting Global Macro Sync Engine...")
    
    # 1. Initialize Database
    init_database()
    
    # 2. Gather All Universes
    print("Gathering ticker universes...")
    try:
        in_tickers = fetch_nifty_total_market_tickers(show_progress=False)
        in_tickers = [t + '.NS' if not t.endswith('.NS') else t for t in in_tickers]
    except Exception as e:
        print(f"Failed to fetch IN tickers: {e}")
        in_tickers = get_tickers_from_file('tickers.txt')
        in_tickers = [t + '.NS' if not t.endswith('.NS') else t for t in in_tickers]
        
    us_tickers = get_tickers_from_file('tickers_us.txt')
    try:
        from us_universe_manager import get_sp500_constituents, get_nasdaq100_constituents
        us_tickers = list(set(us_tickers) | set(get_sp500_constituents() or []) | set(get_nasdaq100_constituents() or []))
    except Exception as e_us_uni:
        print(f"US constituent union notice: {e_us_uni}")
    us_etfs = get_tickers_from_file('tickers_us_etf.txt')
    in_etfs = get_tickers_from_file('tickers_etf.txt')
    in_etfs = [t + '.NS' if not t.endswith('.NS') else t for t in in_etfs]
    
    all_tickers = list(set(in_tickers + us_tickers + us_etfs + in_etfs))
    
    # 3. Fetch Matrix History
    print(f"Fetching history for {len(all_tickers)} global assets...")
    def print_prog(msg):
        print(f"  > {msg}")
    fetch_incremental_history(all_tickers, days=252, period="2y", progress_callback=print_prog, force_today_refresh=True)
    
    # 4. Update Engine Tables
    print("Updating Breadth and Liquidity for IN Market...")
    compute_liquidity(market='IN')
    sync_breadth_history(market='IN')
    
    print("Updating Breadth and Liquidity for US Market...")
    compute_liquidity(market='US')
    sync_breadth_history(market='US')
    
    # 5. Global Benchmark Regime
    print("Calculating Global Regimes...")
    regime_data = []
    # Using Nifty 500 for India and S&P 500 for US
    for market, benchmark in [('IN', '^CRSLDX'), ('US', '^GSPC')]:
        try:
            idx = yf.download(benchmark, period="250d", progress=False)
            idx_std = standardize_columns(idx)
            if not idx_std.empty and 'close' in idx_std.columns:
                dd_count, _ = calculate_distribution_days(idx)
                ftd_detected, _ = detect_follow_through_day(idx)
                choch = detect_change_of_character(idx)
                
                curr = idx_std['close'].iloc[-1]
                sma50 = idx_std['close'].rolling(50).mean().iloc[-1]
                sma200 = idx_std['close'].rolling(200).mean().iloc[-1] if len(idx_std) >= 200 else curr
                regime_label, _, _ = get_market_regime_label(dd_count, curr, sma50)
                choch_label = choch['label'] if choch and choch['detected'] else "None"
                
                date_val = idx_std.index[-1].strftime('%Y-%m-%d')
                
                regime_data.append({
                    'date': date_val,
                    'market': market,
                    'benchmark_ticker': benchmark,
                    'close': float(curr),
                    'sma50': float(sma50),
                    'sma200': float(sma200),
                    'dd_count': int(dd_count),
                    'ftd_detected': bool(ftd_detected),
                    'regime_label': regime_label,
                    'choch_label': choch_label
                })
        except Exception as e:
            print(f"Failed regime calc for {benchmark}: {e}")
            
    if regime_data:
        save_global_regime(regime_data)
        
    # 6. Global ETF Momentum
    print("Calculating ETF Momentum...")
    all_etfs = list(set(us_etfs + in_etfs))
    if all_etfs:
        try:
            data = yf.download(all_etfs, period="1y", progress=False)
            close_prices = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
            etf_results = []
            for etf in all_etfs:
                if etf in close_prices.columns:
                    series = close_prices[etf].dropna()
                    if len(series) >= 20:
                        curr_close = float(series.iloc[-1])
                        high_52w = float(series.max())
                        dist_52w_high = float(((curr_close / high_52w) - 1.0) * 100.0) if high_52w > 0 else 0.0

                        r1 = (curr_close / series.iloc[-21]) - 1 if len(series) > 21 else 0
                        r3 = (curr_close / series.iloc[-63]) - 1 if len(series) > 63 else 0
                        r6 = (curr_close / series.iloc[-126]) - 1 if len(series) > 126 else 0
                        
                        etf_results.append({
                            'date': series.index[-1].strftime('%Y-%m-%d'),
                            'ticker': etf,
                            'name': etf.replace('.NS', ''),
                            'return_1m': float(r1) * 100,
                            'return_3m': float(r3) * 100,
                            'return_6m': float(r6) * 100,
                            'close': curr_close,
                            'high_52w': high_52w,
                            'dist_52w_high': dist_52w_high
                        })
            if etf_results:
                save_global_etf_momentum(etf_results)
                print(f"Saved momentum for {len(etf_results)} global ETFs.")
        except Exception as e:
            print(f"Failed ETF momentum calc: {e}")
                
    # 7. Update NIFTY 500 RS Reference
    print("Computing NIFTY 500 RS Reference...")
    try:
        from nifty500_universe import compute_nifty500_rs_reference
        compute_nifty500_rs_reference(show_progress=True)
    except Exception as e:
        print(f"Failed to compute RS Reference: {e}")

    # 8. Global Sovereign Normalized USD Returns
    print("Calculating Global Sovereign Normalized USD Returns...")
    sync_global_sovereign_returns()

    today_iso = datetime.now().strftime("%Y-%m-%d")

    # 9. Industry Group Matrix v2.0 Cloud Bundles (IN Canonical, IN Thematic, US Canonical)
    print("Publishing Industry Group Matrix v2.0 bundles to Turso cloud_json_cache...")
    try:
        from industry_group_engine import (
            refresh_industry_group_intelligence_cache,
            compute_industry_group_matrix_with_niche,
        )
        for _u_scope in ("India (NSE/BSE)", "US (S&P 500 / Nasdaq)"):
            try:
                refresh_industry_group_intelligence_cache(
                    universe_scope=_u_scope, force=False, skip_enrichment=True
                )
            except Exception as _e_int:
                print(f"[Cloud Publish] IGM intelligence refresh notice ({_u_scope}): {_e_int}")

        for _tax, _scope, _mkt_tag, _ckey in (
            ("canonical", "India (NSE/BSE)", "IN", "igm_bundle_in"),
            ("thematic", "India (NSE/BSE)", "IN", "igm_thematic_in"),
            ("canonical", "US (S&P 500 / Nasdaq)", "US", "igm_bundle_us"),
        ):
            try:
                df_m, bench_c, df_n = compute_industry_group_matrix_with_niche(
                    taxonomy=_tax, universe_scope=_scope
                )
                if isinstance(df_m, pd.DataFrame) and not df_m.empty:
                    save_cloud_json_cache(
                        _ckey,
                        _mkt_tag,
                        {"matrix_df": df_m, "benchmark_curve": bench_c, "niche_df": df_n},
                        as_of_date=today_iso,
                    )
                    print(f"  > Published {_ckey} ({len(df_m)} groups)")
            except Exception as _e_igm:
                print(f"[Cloud Publish] Failed {_ckey}: {_e_igm}")
    except Exception as _e_step9:
        print(f"[Step 9] Industry Group Matrix cloud publish warning: {_e_step9}")

    # 10. Stockbee Monitor + Thrusting Industries + UDVR Maps across IN & US scopes
    print("Publishing Stockbee Monitor, Thrusting Industries & UDVR maps to Turso...")
    try:
        from components import load_stockbee_monitor_df, _stage_matrix_path_for_market
        from leadership_cluster_engine import compute_stockbee_thrusting_industries, _load_udvr_map_cached
        from price_history_manager import get_matrix_cache_signature

        for _mkt_c, _scope_arg, _slug in (
            ("IN", "NTM750", "ntm750"),
            ("IN", "FULL", "full"),
            ("US", "SP500", "sp500"),
            ("US", "Nasdaq 100", "nasdaq_100"),
            ("US", "BROAD", "broad"),
        ):
            try:
                sb_df = load_stockbee_monitor_df(market=_mkt_c, universe_scope=_scope_arg, lookback_days=40)
                if isinstance(sb_df, pd.DataFrame) and not sb_df.empty:
                    sb_dt = str(sb_df.iloc[0].get("Date") or today_iso)[:10]
                    save_cloud_json_cache(
                        f"stockbee_monitor_{_mkt_c.lower()}_{_slug}",
                        _mkt_c,
                        sb_df,
                        as_of_date=sb_dt,
                    )
                    print(f"  > Published stockbee_monitor_{_mkt_c.lower()}_{_slug} ({len(sb_df)} rows)")
            except Exception as _e_sbm:
                print(f"[Cloud Publish] stockbee_monitor_{_mkt_c.lower()}_{_slug} warning: {_e_sbm}")

            try:
                th_res = compute_stockbee_thrusting_industries(
                    market=_mkt_c, universe_scope=_scope_arg, mode="5D_AND_TODAY", top_n=200
                )
                if isinstance(th_res, dict) and isinstance(th_res.get("all_thrust_df"), pd.DataFrame) and not th_res["all_thrust_df"].empty:
                    th_dt = str((th_res.get("as_of") or {}).get("prices_date") or today_iso)[:10]
                    save_cloud_json_cache(
                        f"stockbee_thrust_{_mkt_c.lower()}_{_slug}",
                        _mkt_c,
                        th_res,
                        as_of_date=th_dt,
                    )
                    print(f"  > Published stockbee_thrust_{_mkt_c.lower()}_{_slug} ({len(th_res['all_thrust_df'])} clusters)")
            except Exception as _e_sbt:
                print(f"[Cloud Publish] stockbee_thrust_{_mkt_c.lower()}_{_slug} warning: {_e_sbt}")

        for _mkt_u, _scope_u, _ukey in (
            ("IN", "FULL", "udvr_map_in"),
            ("US", "BROAD", "udvr_map_us"),
        ):
            try:
                _mp = _stage_matrix_path_for_market(_mkt_u)
                _msig = get_matrix_cache_signature(_mp) if os.path.exists(_mp) else (0, 0)
                _umap = _load_udvr_map_cached(_mkt_u, _scope_u, _msig)
                if isinstance(_umap, dict) and _umap:
                    save_cloud_json_cache(_ukey, _mkt_u, _umap, as_of_date=today_iso)
                    print(f"  > Published {_ukey} ({len(_umap)} tickers)")
            except Exception as _e_udvr:
                print(f"[Cloud Publish] {_ukey} warning: {_e_udvr}")
    except Exception as _e_step10:
        print(f"[Step 10] Stockbee/UDVR cloud publish warning: {_e_step10}")

    # 11. Auxiliary Breadth Bundles (IN & US)
    print("Publishing Breadth Auxiliary Bundles to Turso...")
    try:
        from breadth_engine import (
            compute_institutional_breadth_alpha_in_core,
            compute_institutional_breadth_alpha_us_core,
            compute_sector_breadth_matrix_in,
            compute_sector_breadth_matrix_us,
            compute_top_rs_breadth_history,
            slice_custom_scope_breadth_us_core,
        )
        from breakout_cohort_engine import compute_breakout_health_radar
        from components import load_cached_stage_distribution_summary

        _bench_closes = {}
        for _idx_sym in ("^CRSLDX", "^NSEI", "^GSPC", "^IXIC"):
            try:
                _idx_raw = yf.download(_idx_sym, period="1y", progress=False, auto_adjust=True)
                if isinstance(_idx_raw, pd.DataFrame) and not _idx_raw.empty and "Close" in _idx_raw.columns:
                    _cs = _idx_raw["Close"]
                    if isinstance(_cs, pd.DataFrame):
                        _cs = _cs.iloc[:, 0]
                    _cs = _cs.dropna()
                    if getattr(_cs.index, "tz", None) is not None:
                        _cs.index = _cs.index.tz_localize(None)
                    _cs.index = pd.to_datetime(_cs.index).normalize()
                    _bench_closes[_idx_sym] = _cs
            except Exception:
                pass

        try:
            _alpha_in = (
                compute_institutional_breadth_alpha_in_core(_bench_closes["^CRSLDX"], _bench_closes["^NSEI"])
                if ("^CRSLDX" in _bench_closes and "^NSEI" in _bench_closes)
                else None
            )
            _in_aux = {
                "alpha": _alpha_in,
                "top_rs_breadth": compute_top_rs_breadth_history(market="IN", lookback_days=300),
                "sector_breadth": compute_sector_breadth_matrix_in(min_stocks=5),
                "breakout_cohort": list(compute_breakout_health_radar(scope="IN")),
                "stage_dist": load_cached_stage_distribution_summary(market="IN"),
            }
            save_cloud_json_cache("breadth_aux_bundle_in", "IN", _in_aux, as_of_date=today_iso)
            print("  > Published breadth_aux_bundle_in")
        except Exception as _e_bin:
            print(f"[Cloud Publish] breadth_aux_bundle_in warning: {_e_bin}")

        try:
            _alpha_us = (
                compute_institutional_breadth_alpha_us_core(_bench_closes["^GSPC"], _bench_closes["^IXIC"])
                if ("^GSPC" in _bench_closes and "^IXIC" in _bench_closes)
                else None
            )
            _us_aux = {
                "alpha": _alpha_us,
                "top_rs_breadth": compute_top_rs_breadth_history(market="US", lookback_days=300),
                "sector_breadth": compute_sector_breadth_matrix_us(),
                "breakout_cohort": list(compute_breakout_health_radar(scope="US")),
                "stage_dist": load_cached_stage_distribution_summary(market="US"),
                "us_scopes": {
                    "sp500": slice_custom_scope_breadth_us_core("S&P 500"),
                    "ndx100": slice_custom_scope_breadth_us_core("Nasdaq 100"),
                },
            }
            save_cloud_json_cache("breadth_aux_bundle_us", "US", _us_aux, as_of_date=today_iso)
            print("  > Published breadth_aux_bundle_us")
        except Exception as _e_bus:
            print(f"[Cloud Publish] breadth_aux_bundle_us warning: {_e_bus}")
    except Exception as _e_step11:
        print(f"[Step 11] Breadth Aux cloud publish warning: {_e_step11}")

    # 12. Master RS Leader Radar Top-300 Snapshots
    print("Publishing Master RS Leader Radar Top-300 snapshots to Turso...")
    try:
        from breadth_engine import compute_leader_21ema_table

        for _lkey, _lmkt, _lscope, _ladtv, _lbench in (
            ("leader_radar_in_750", "IN", "750", 0.10, None),
            ("leader_radar_in_all", "IN", "ALL", 0.10, None),
            ("leader_radar_us_ixic", "US", "ALL", 1.0, "^IXIC"),
            ("leader_radar_us_gspc", "US", "ALL", 1.0, "^GSPC"),
        ):
            try:
                _ldf = compute_leader_21ema_table(
                    market=_lmkt,
                    top_n=300,
                    universe_scope=_lscope,
                    min_adtv_cr=_ladtv,
                    benchmark_symbol=_lbench,
                )
                if isinstance(_ldf, pd.DataFrame) and not _ldf.empty:
                    _ldt = str((_ldf.attrs.get("diagnostics") or {}).get("matrix_last_dt") or today_iso)[:10]
                    save_cloud_json_cache(_lkey, _lmkt, _ldf, as_of_date=_ldt)
                    print(f"  > Published {_lkey} ({len(_ldf)} leaders)")
            except Exception as _e_ldr:
                print(f"[Cloud Publish] {_lkey} warning: {_e_ldr}")
    except Exception as _e_step12:
        print(f"[Step 12] Leader Radar cloud publish warning: {_e_step12}")

    # 13. Dual-Market Persona & Artificial Brain Synthesis (INDIA + US_IXIC)
    print("Computing Dual-Market Persona & Artificial Brain Synthesis...")
    try:
        from market_persona_engine import sync_market_persona_daily
        from brain_synthesis_engine import run_artificial_brain_synthesis

        for _bmkt in ("INDIA", "US_IXIC"):
            try:
                sync_market_persona_daily(market=_bmkt)
            except Exception as _e_pers:
                print(f"[Cloud Publish] Persona notice ({_bmkt}): {_e_pers}")
            try:
                _bres = run_artificial_brain_synthesis(market=_bmkt, force_recompute=True)
                _cnt_c = len((_bres or {}).get("confluence") or []) if isinstance(_bres, dict) else 0
                print(f"  > Published Brain Synthesis ({_bmkt}, {_cnt_c} confluence setups)")
            except Exception as _e_brn:
                print(f"[Cloud Publish] Brain Synthesis warning ({_bmkt}): {_e_brn}")
    except Exception as _e_step13:
        print(f"[Step 13] Artificial Brain publish warning: {_e_step13}")

    print("Global Macro Sync Completed!")

SOVEREIGN_MARKETS = [
    {'country': 'United States', 'flag': '🇺🇸', 'index_ticker': '^GSPC', 'etf_ticker': 'SPY', 'index_name': 'S&P 500 Core', 'currency': 'USD', 'fx_ticker': None, 'inverse_fx': False},
    {'country': 'United States', 'flag': '🇺🇸', 'index_ticker': '^NDX', 'etf_ticker': 'QQQ', 'index_name': 'Nasdaq 100 Growth', 'currency': 'USD', 'fx_ticker': None, 'inverse_fx': False},
    {'country': 'India', 'flag': '🇮🇳', 'index_ticker': '^NSEI', 'etf_ticker': 'INDY', 'index_name': 'NIFTY 50 LargeCap', 'currency': 'INR', 'fx_ticker': 'USDINR=X', 'inverse_fx': True},
    {'country': 'India', 'flag': '🇮🇳', 'index_ticker': '^CRSLDX', 'etf_ticker': 'INDA', 'index_name': 'NIFTY 500 BroadMarket', 'currency': 'INR', 'fx_ticker': 'USDINR=X', 'inverse_fx': True},
    {'country': 'Japan', 'flag': '🇯🇵', 'index_ticker': '^N225', 'etf_ticker': 'EWJ', 'index_name': 'Nikkei 225', 'currency': 'JPY', 'fx_ticker': 'JPY=X', 'inverse_fx': True},
    {'country': 'Germany', 'flag': '🇩🇪', 'index_ticker': '^GDAXI', 'etf_ticker': 'EWG', 'index_name': 'DAX 40', 'currency': 'EUR', 'fx_ticker': 'EURUSD=X', 'inverse_fx': False},
    {'country': 'United Kingdom', 'flag': '🇬🇧', 'index_ticker': '^FTSE', 'etf_ticker': 'EWU', 'index_name': 'FTSE 100', 'currency': 'GBP', 'fx_ticker': 'GBPUSD=X', 'inverse_fx': False},
    {'country': 'China', 'flag': '🇨🇳', 'index_ticker': '^HSI', 'etf_ticker': 'MCHI', 'index_name': 'Hang Seng Index', 'currency': 'HKD', 'fx_ticker': 'HKD=X', 'inverse_fx': True},
    {'country': 'France', 'flag': '🇫🇷', 'index_ticker': '^FCHI', 'etf_ticker': 'EWQ', 'index_name': 'CAC 40', 'currency': 'EUR', 'fx_ticker': 'EURUSD=X', 'inverse_fx': False},
    {'country': 'Australia', 'flag': '🇦🇺', 'index_ticker': '^AXJO', 'etf_ticker': 'EWA', 'index_name': 'S&P/ASX 200', 'currency': 'AUD', 'fx_ticker': 'AUDUSD=X', 'inverse_fx': False},
    {'country': 'Canada', 'flag': '🇨🇦', 'index_ticker': '^GSPTSE', 'etf_ticker': 'EWC', 'index_name': 'S&P/TSX Composite', 'currency': 'CAD', 'fx_ticker': 'CAD=X', 'inverse_fx': True},
    {'country': 'Brazil', 'flag': '🇧🇷', 'index_ticker': '^BVSP', 'etf_ticker': 'EWZ', 'index_name': 'Ibovespa', 'currency': 'BRL', 'fx_ticker': 'BRL=X', 'inverse_fx': True},
    {'country': 'South Korea', 'flag': '🇰🇷', 'index_ticker': '^KS11', 'etf_ticker': 'EWY', 'index_name': 'KOSPI', 'currency': 'KRW', 'fx_ticker': 'KRW=X', 'inverse_fx': True},
    {'country': 'Taiwan', 'flag': '🇹🇼', 'index_ticker': '^TWII', 'etf_ticker': 'EWT', 'index_name': 'TAIEX', 'currency': 'TWD', 'fx_ticker': 'TWD=X', 'inverse_fx': True},
]

def sync_global_sovereign_returns():
    """Fetches global sovereign indices and FX pairs, normalizing returns into USD across 1M, 3M, 6M, 1Y."""
    init_database()
    idx_tickers = [m['index_ticker'] for m in SOVEREIGN_MARKETS]
    fx_tickers = [m['fx_ticker'] for m in SOVEREIGN_MARKETS if m['fx_ticker']]
    all_syms = list(set(idx_tickers + fx_tickers))
    
    try:
        data = yf.download(all_syms, period="15mo", progress=False)
        close_prices = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
        close_prices = close_prices.ffill().bfill()
        
        parsed_markets = []
        sp500_usd_returns = {'1m': 0.0, '3m': 0.0, '6m': 0.0, '1y': 0.0}
        
        for m in SOVEREIGN_MARKETS:
            t = m['index_ticker']
            if t not in close_prices.columns:
                continue
            idx_series = close_prices[t].dropna()
            if len(idx_series) < 25:
                continue
                
            fx_sym = m['fx_ticker']
            if fx_sym and fx_sym in close_prices.columns:
                fx_raw = close_prices[fx_sym].reindex(idx_series.index).ffill().bfill()
                if m['inverse_fx']:
                    fx_mult = 1.0 / fx_raw
                else:
                    fx_mult = fx_raw
                fx_latest = float(fx_raw.iloc[-1])
            else:
                fx_mult = pd.Series(1.0, index=idx_series.index)
                fx_latest = 1.0
                
            usd_series = idx_series * fx_mult
            curr_usd = float(usd_series.iloc[-1])
            curr_local = float(idx_series.iloc[-1])
            high_52w_usd = float(usd_series.iloc[-252:].max()) if len(usd_series) >= 252 else float(usd_series.max())
            dist_52w = ((curr_usd / high_52w_usd) - 1.0) * 100.0 if high_52w_usd > 0 else 0.0
            
            def calc_metrics(h_offset):
                target_dt = idx_series.index[-1] - h_offset
                past_idx = idx_series.loc[:target_dt]
                if len(past_idx) > 0:
                    past_local = float(past_idx.iloc[-1])
                    past_mult = float(fx_mult.loc[:target_dt].iloc[-1])
                    past_usd = float(usd_series.loc[:target_dt].iloc[-1])
                    ret_local = ((curr_local / past_local) - 1.0) * 100.0
                    ret_fx = ((float(fx_mult.iloc[-1]) / past_mult) - 1.0) * 100.0
                    ret_usd = ((curr_usd / past_usd) - 1.0) * 100.0
                else:
                    ret_local, ret_fx, ret_usd = 0.0, 0.0, 0.0
                return ret_local, ret_fx, ret_usd
                
            r1_loc, r1_fx, r1_usd = calc_metrics(pd.DateOffset(months=1))
            r3_loc, r3_fx, r3_usd = calc_metrics(pd.DateOffset(months=3))
            r6_loc, r6_fx, r6_usd = calc_metrics(pd.DateOffset(months=6))
            ry_loc, ry_fx, ry_usd = calc_metrics(pd.DateOffset(years=1))
            
            if t == '^GSPC':
                sp500_usd_returns = {'1m': r1_usd, '3m': r3_usd, '6m': r6_usd, '1y': ry_usd}
                
            date_str = idx_series.index[-1].strftime('%Y-%m-%d')
            parsed_markets.append({
                'date': date_str,
                'country': m['country'],
                'index_ticker': t,
                'etf_ticker': m.get('etf_ticker', ''),
                'index_name': m['index_name'],
                'currency': m['currency'],
                'fx_ticker': fx_sym if fx_sym else 'USD',
                'local_close': curr_local,
                'fx_rate': fx_latest,
                'return_1m_local': r1_loc,
                'return_1m_fx': r1_fx,
                'return_1m_usd': r1_usd,
                'return_3m_local': r3_loc,
                'return_3m_fx': r3_fx,
                'return_3m_usd': r3_usd,
                'return_6m_local': r6_loc,
                'return_6m_fx': r6_fx,
                'return_6m_usd': r6_usd,
                'return_1y_local': ry_loc,
                'return_1y_fx': ry_fx,
                'return_1y_usd': ry_usd,
                'dist_52w_high': dist_52w
            })
            
        # Calculate USD Alpha vs S&P 500
        for p in parsed_markets:
            p['alpha_1m_usd'] = p['return_1m_usd'] - sp500_usd_returns['1m']
            p['alpha_3m_usd'] = p['return_3m_usd'] - sp500_usd_returns['3m']
            p['alpha_6m_usd'] = p['return_6m_usd'] - sp500_usd_returns['6m']
            p['alpha_1y_usd'] = p['return_1y_usd'] - sp500_usd_returns['1y']
            
        if parsed_markets:
            save_global_sovereign_returns(parsed_markets)
            print(f"[Macro Engine] Successfully saved {len(parsed_markets)} sovereign USD return records.")
        return parsed_markets
    except Exception as e:
        print(f"[Macro Engine] Failed sovereign USD calculation: {e}")
        return []

if __name__ == "__main__":
    sync_all_data()
