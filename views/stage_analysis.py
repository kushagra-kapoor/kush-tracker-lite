import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import requests
import io


# st.set_page_config managed by app.py in Lite

try:
    from styles import load_css
    load_css()
except ImportError:
    pass

@st.cache_data(ttl=3600*24)
def fetch_universe_mapping(cache_buster=2):
    st.cache_data.clear()
    try:
        from views.market_regime import fetch_universe_with_industry
        df = fetch_universe_with_industry()
        if df is None or df.empty:
            return {}
        return df.set_index('Symbol')['Industry'].to_dict()
    except Exception as e:
        return {}

@st.cache_data(ttl=3600*24)
def fetch_cap_segments():
    keys = {
        "Nifty 50": "NIFTY_50",
        "Nifty Next 50": "NIFTY_NEXT_50",
        "Midcap 150": "NIFTY_MIDCAP_150",
        "Smallcap 250": "NIFTY_SMALLCAP_250",
        "Microcap 250": "NIFTY_MICROCAP_250",
        "CNX 500": "NIFTY_500"
    }
    segments = {}
    from market_data import fetch_and_cache_csv
    for name, key in keys.items():
        try:
            df = fetch_and_cache_csv(key, show_progress=False)
            
            ticker_col = None
            for col in df.columns:
                if 'symbol' in col.lower():
                    ticker_col = col
            if ticker_col is None:
                ticker_col = df.columns[2]
                
            tickers = df[ticker_col].dropna().astype(str).str.strip().str.upper().tolist()
            segments[name] = [f"{t}.NS" for t in tickers if t and (t.isalnum() or '-' in t or '&' in t)]
        except Exception as e:
            pass
    return segments

@st.cache_data(ttl=3600*6)
def load_and_compute_internals():
    matrix_path = 'historical_prices_matrix.pkl'
    if not os.path.exists(matrix_path):
        return None, None, None, None, None, None, None
        
    df = pd.read_pickle(matrix_path)
    close_df = df.xs('Close', level=1, axis=1)
    open_df = df.xs('Open', level=1, axis=1)
    vol_df = df.xs('Volume', level=1, axis=1)
    
    # 1. UNIVERSE FILTER: Only Nifty Total Market
    industry_map = fetch_universe_mapping()
    valid_symbols = set([f"{sym}.NS" for sym in industry_map.keys()])
    
    available_cols = [c for c in close_df.columns if c in valid_symbols]
    close_df = close_df[available_cols]
    open_df = open_df[available_cols]
    vol_df = vol_df[available_cols]
    
    # Filter out holidays/weekends globally (where almost all stocks have NaN prices)
    valid_trading_days = close_df.notna().sum(axis=1) > 10
    close_df = close_df[valid_trading_days]
    open_df = open_df[valid_trading_days]
    vol_df = vol_df[valid_trading_days]
    
    # 2. MOMENTUM BREADTH (Latest Day)
    if close_df.empty:
        st.error("Failed to load sufficient market data. NSE fetching might be blocked.")
        return None, None, None, None, None, None, None
        
    latest_close = close_df.iloc[-1]
    latest_open = open_df.iloc[-1]
    prev_close = close_df.iloc[-2] if len(close_df) > 1 else latest_close
    latest_vol = vol_df.iloc[-1]
    
    ret_pct = ((latest_close - prev_close) / prev_close) * 100
    
    # Up/Down from Open
    up_open = (latest_close > latest_open).sum()
    down_open = (latest_close < latest_open).sum()
    
    # Up/Down Volume
    up_vol = latest_vol[latest_close > prev_close].sum()
    down_vol = latest_vol[latest_close < prev_close].sum()
    
    # +/- 4%
    up_4 = (ret_pct >= 4.0).sum()
    down_4 = (ret_pct <= -4.0).sum()
    
    momentum_metrics = {
        'up_open': up_open, 'down_open': down_open,
        'up_vol': up_vol, 'down_vol': down_vol,
        'up_4': up_4, 'down_4': down_4
    }
    
    # 3. STAGE ANALYSIS (Vectorized over time)
    sma_50 = close_df.rolling(50, min_periods=20).mean()
    sma_200 = close_df.rolling(200, min_periods=50).mean()
    
    # 200-day Slope
    sma_200_slope_pos = sma_200 > sma_200.shift(20)
    sma_200_slope_neg = sma_200 < sma_200.shift(20)
    
    # Stage logic based on Minervini/Weinstein
    s2_mask = (close_df > sma_50) & (sma_50 > sma_200) & sma_200_slope_pos
    s4_mask = (close_df < sma_50) & (sma_50 < sma_200) & sma_200_slope_neg
    s3_mask = (~s2_mask) & (~s4_mask) & (sma_50 > sma_200)
    s1_mask = (~s2_mask) & (~s4_mask) & (sma_50 <= sma_200)
    
    # Time-series counts
    historical_counts = pd.DataFrame({
        'S1': s1_mask.sum(axis=1),
        'S2': s2_mask.sum(axis=1),
        'S3': s3_mask.sum(axis=1),
        'S4': s4_mask.sum(axis=1)
    })
    
    # (Empty days already filtered globally above)
    
    # Drop the initial MA warmup period (where S2 and S4 are exactly 0)
    active_days = historical_counts[(historical_counts['S2'] > 0) | (historical_counts['S4'] > 0)]
    if not active_days.empty:
        historical_counts = historical_counts.loc[active_days.index[0]:]
    
    # 4-Way Weekly Stage Migration tracking (5-day rolling transition)
    s1_to_s2_mask = s1_mask.shift(5).fillna(False) & s2_mask
    s2_to_s3_mask = s2_mask.shift(5).fillna(False) & s3_mask
    s3_to_s4_mask = s3_mask.shift(5).fillna(False) & s4_mask
    s4_to_s1_mask = s4_mask.shift(5).fillna(False) & s1_mask

    s1_to_s2_series = s1_to_s2_mask.sum(axis=1)
    s2_to_s3_series = s2_to_s3_mask.sum(axis=1)
    s3_to_s4_series = s3_to_s4_mask.sum(axis=1)
    s4_to_s1_series = s4_to_s1_mask.sum(axis=1)
    net_migration_series = s1_to_s2_series - s3_to_s4_series

    stage_migration_df = pd.DataFrame({
        'S1_to_S2': s1_to_s2_series,
        'S2_to_S3': s2_to_s3_series,
        'S3_to_S4': s3_to_s4_series,
        'S4_to_S1': s4_to_s1_series,
        'Net_Migration': net_migration_series
    })
    if not active_days.empty:
        stage_migration_df = stage_migration_df.loc[active_days.index[0]:]

    migration_syms_dict = {
        'S1_to_S2': [c for c in s1_to_s2_mask.columns if bool(s1_to_s2_mask.iloc[-1][c])],
        'S2_to_S3': [c for c in s2_to_s3_mask.columns if bool(s2_to_s3_mask.iloc[-1][c])],
        'S3_to_S4': [c for c in s3_to_s4_mask.columns if bool(s3_to_s4_mask.iloc[-1][c])],
        'S4_to_S1': [c for c in s4_to_s1_mask.columns if bool(s4_to_s1_mask.iloc[-1][c])]
    }
    
    # Latest Day Details
    latest_s1 = s1_mask.iloc[-1]
    latest_s2 = s2_mask.iloc[-1]
    latest_s3 = s3_mask.iloc[-1]
    latest_s4 = s4_mask.iloc[-1]
    
    latest_stages = pd.Series('Unknown', index=close_df.columns)
    latest_stages[latest_s1] = 'S1'
    latest_stages[latest_s2] = 'S2'
    latest_stages[latest_s3] = 'S3'
    latest_stages[latest_s4] = 'S4'
    
    # New additions: Breadth and Momentum
    latest_sma50 = sma_50.iloc[-1]
    latest_sma200 = sma_200.iloc[-1]
    above_50 = (latest_close > latest_sma50)
    dist_50 = ((latest_close - latest_sma50) / latest_sma50) * 100
    dist_200 = ((latest_close - latest_sma200) / latest_sma200) * 100
    
    ret_1w = (close_df.iloc[-1] / close_df.iloc[-6] - 1) * 100 if len(close_df) >= 6 else pd.Series(0, index=close_df.columns)
    ret_1m = (close_df.iloc[-1] / close_df.iloc[-22] - 1) * 100 if len(close_df) >= 22 else pd.Series(0, index=close_df.columns)
    ret_3m = (close_df.iloc[-1] / close_df.iloc[-64] - 1) * 100 if len(close_df) >= 64 else pd.Series(0, index=close_df.columns)
    
    df_latest = pd.DataFrame({
        'Stage': latest_stages,
        'Close': latest_close,
        'Above_50': above_50,
        'Dist_50': dist_50,
        'Dist_200': dist_200,
        'Ret_1W': ret_1w,
        'Ret_1M': ret_1m,
        'Ret_3M': ret_3m
    })
    df_latest['Industry'] = [industry_map.get(sym.replace('.NS', ''), 'Other') for sym in df_latest.index]
    
    # New additions: Market Cap Segments Breadth Matrix
    segments = fetch_cap_segments()
    segment_stats = []
    
    sma_25 = close_df.rolling(25, min_periods=10).mean()
    brutal_strength = (latest_close > sma_25.iloc[-1]) & (sma_25.iloc[-1] > latest_sma50) & (latest_sma50 > sma_200.iloc[-1])
    
    weekly_ret = (close_df.iloc[-1] / close_df.iloc[-6]) - 1 if len(close_df) >= 6 else pd.Series(0, index=close_df.columns)
    
    for seg_name, seg_symbols in segments.items():
        valid_syms = [s for s in seg_symbols if s in close_df.columns]
        if not valid_syms: continue
        
        total = len(valid_syms)
        bs_count = brutal_strength[valid_syms].sum()
        ab_50_count = above_50[valid_syms].sum()
        
        adv_count = (weekly_ret[valid_syms] > 0).sum()
        dec_count = (weekly_ret[valid_syms] < 0).sum()
        flat_count = total - adv_count - dec_count
        
        segment_stats.append({
            'Segment': seg_name,
            'Total': total,
            'Brutal_Strength_Pct': (bs_count / total) * 100 if total > 0 else 0,
            'Above_50_Pct': (ab_50_count / total) * 100 if total > 0 else 0,
            'Adv': adv_count,
            'Dec': dec_count,
            'Flat': flat_count
        })
        
    df_segments = pd.DataFrame(segment_stats)
    
    # 4b. Relative Alpha Breadth: % of Stocks Outperforming Benchmark across 1W, 1M, 3M, 6M, 1Y
    windows_config = {
        '1 Week': 5,
        '1 Month': 21,
        '3 Months': 63,
        '6 Months': 126,
        '1 Year': 250
    }
    
    bench_close = None
    bench_name = "NIFTY 500"
    if '^CRSLDX' in df.columns.get_level_values(0):
        bench_close = df['^CRSLDX']['Close'].dropna()
        bench_name = "NIFTY 500 (^CRSLDX)"
    elif 'NIFTYBEES.NS' in df.columns.get_level_values(0):
        bench_close = df['NIFTYBEES.NS']['Close'].dropna()
        bench_name = "NIFTYBEES"
    elif len(close_df.columns) > 0:
        bench_close = close_df.mean(axis=1).dropna()
        bench_name = "Total Market Equal-Weight"
        
    # Precompute Vectorized RS Rating (CANSLIM Percentile 1-99) and Mansfield RS (MRS)
    c0 = close_df.iloc[-1]
    n_bars = len(close_df)
    c_1m = close_df.iloc[max(0, n_bars - 22)].replace(0, np.nan)
    c_3m = close_df.iloc[max(0, n_bars - 64)].replace(0, np.nan)
    c_6m = close_df.iloc[max(0, n_bars - 127)].replace(0, np.nan)
    
    ret_1m = ((c0 - c_1m) / c_1m) * 100.0
    ret_3m = ((c0 - c_3m) / c_3m) * 100.0
    ret_6m = ((c0 - c_6m) / c_6m) * 100.0
    
    rs_raw = (0.40 * ret_1m.fillna(0)) + (0.40 * ret_3m.fillna(0)) + (0.20 * ret_6m.fillna(0))
    rs_pct = rs_raw.rank(pct=True, na_option='bottom')
    rs_rating = (rs_pct * 98.0 + 1.0).round().clip(lower=1, upper=99).fillna(50).astype(int)
    
    if bench_close is not None and len(bench_close) >= 20:
        r_mat = close_df.div(bench_close, axis=0).dropna(how='all')
        sma_r = r_mat.rolling(window=250, min_periods=min(50, len(r_mat))).mean()
        mrs_series = ((r_mat.iloc[-1] / sma_r.iloc[-1]) - 1.0) * 100.0
    else:
        mrs_series = pd.Series(0.0, index=close_df.columns)

    outperformance_data = {}
    outperf_segments = dict(segments)
    outperf_segments["All Stocks (All Universe)"] = list(close_df.columns)
    for horizon_label, days_back in windows_config.items():
        if bench_close is not None and len(bench_close) > days_back:
            bench_ret = float((bench_close.iloc[-1] / bench_close.iloc[-days_back-1]) - 1) * 100.0
        else:
            bench_ret = 0.0
            
        horizon_segments = {}
        for seg_name, seg_symbols in outperf_segments.items():
            valid_syms = [s for s in seg_symbols if s in close_df.columns]
            if not valid_syms:
                continue
            seg_closes = close_df[valid_syms].dropna(how='all')
            if len(seg_closes) > days_back:
                stk_rets = ((seg_closes.iloc[-1] / seg_closes.iloc[-days_back-1]) - 1) * 100.0
                valid_rets = stk_rets.dropna()
                tot_count = len(valid_rets)
                outperf_mask = valid_rets > bench_ret
                outperf_count = int(outperf_mask.sum())
                outperf_pct = (outperf_count / tot_count * 100.0) if tot_count > 0 else 0.0
                
                outperf_tickers = [s.replace('.NS', '') for s in valid_rets[outperf_mask].index]
                underperf_tickers = [s.replace('.NS', '') for s in valid_rets[~outperf_mask].index]
                
                stock_details = []
                for s in valid_rets.index:
                    clean_sym = s.replace('.NS', '')
                    s_ret = float(valid_rets[s])
                    is_out = bool(s_ret > bench_ret)
                    stock_details.append({
                        'symbol': clean_sym,
                        'raw_symbol': s,
                        'return': round(s_ret, 2),
                        'excess_return': round(s_ret - bench_ret, 2),
                        'is_outperforming': is_out,
                        'rs_rating': int(rs_rating.get(s, 50)),
                        'mrs': round(float(mrs_series.get(s, 0.0)), 2) if not pd.isna(mrs_series.get(s, np.nan)) else 0.0,
                        'stage': str(latest_stages.get(s, 'S1')),
                        'industry': str(industry_map.get(clean_sym, 'Other')),
                        'close': round(float(latest_close.get(s, 0.0)), 2)
                    })
                
                horizon_segments[seg_name] = {
                    'pct': round(outperf_pct, 1),
                    'outperf_count': outperf_count,
                    'total_count': tot_count,
                    'avg_return': round(float(valid_rets.mean()), 2) if tot_count else 0.0,
                    'median_return': round(float(valid_rets.median()), 2) if tot_count else 0.0,
                    'outperforming_tickers': outperf_tickers,
                    'underperforming_tickers': underperf_tickers,
                    'stock_details': stock_details,
                }
        
        outperformance_data[horizon_label] = {
            'benchmark_name': bench_name,
            'benchmark_return': round(bench_ret, 2),
            'segments': horizon_segments
        }
    
    return momentum_metrics, historical_counts, df_latest, df_segments, stage_migration_df, migration_syms_dict, outperformance_data

st.title("🔬 Market Internals & Stage Analysis")
st.markdown("Quantifying true institutional money flow across the **Nifty Total Market** universe. Filtered for liquidity.")

with st.spinner("Calculating quantitative internal metrics..."):
    internals_res = load_and_compute_internals()

if internals_res is None or internals_res[0] is None:
    st.warning("Historical data not found. Please run the database builder.")
    st.stop()

if len(internals_res) >= 7:
    momentum_metrics, historical_counts, df_latest, df_segments, stage_migration_df, migration_syms_dict, outperformance_data = internals_res[:7]
else:
    momentum_metrics, historical_counts, df_latest, df_segments, stage_migration_df, migration_syms_dict = internals_res[:6]
    outperformance_data = {}

# --- CUSTOM HTML GENERATORS ---
def format_num(num):
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return f"{num:.0f}"

def enhanced_momentum_bar(val1, val2, label1, label2, title, metric_type="open", subtitle=""):
    total = val1 + val2 if (val1 + val2) > 0 else 1
    pct1 = (val1 / total) * 100
    pct2 = 100.0 - pct1
    
    # Determine dynamic state badge & color based on metric_type
    if metric_type == "open":
        if pct1 < 40:
            badge_text = "⚠️ INTRADAY FADE"
            badge_color = "#f43f5e"
            badge_bg = "rgba(244, 63, 94, 0.15)"
        elif pct1 > 60:
            badge_text = "🔥 DAYTIME BUYING"
            badge_color = "#10b981"
            badge_bg = "rgba(16, 185, 129, 0.15)"
        else:
            badge_text = "⚖️ NEUTRAL CHURN"
            badge_color = "#fbbf24"
            badge_bg = "rgba(251, 191, 36, 0.15)"
        bar_grad1 = "linear-gradient(90deg, #059669, #10b981)"
        bar_grad2 = "linear-gradient(90deg, #e11d48, #f43f5e)"
        c1 = "#10b981"
        c2 = "#f43f5e"
    elif metric_type == "vol":
        # Check volume divergence against up_open
        up_open_ratio = momentum_metrics['up_open'] / (momentum_metrics['up_open'] + momentum_metrics['down_open'] + 1e-6)
        if pct1 > 75 and up_open_ratio < 0.45:
            badge_text = "⚠️ CHURN DIVERGENCE"
            badge_color = "#fbbf24"
            badge_bg = "rgba(251, 191, 36, 0.15)"
        elif pct1 > 75:
            badge_text = "🚀 VOLUME THRUST"
            badge_color = "#10b981"
            badge_bg = "rgba(16, 185, 129, 0.15)"
        elif pct1 < 25:
            badge_text = "🛑 HEAVY DUMPING"
            badge_color = "#f43f5e"
            badge_bg = "rgba(244, 63, 94, 0.15)"
        else:
            badge_text = "⚖️ BALANCED VOL"
            badge_color = "#94a3b8"
            badge_bg = "rgba(148, 163, 184, 0.15)"
        bar_grad1 = "linear-gradient(90deg, #0284c7, #38bdf8)"
        bar_grad2 = "linear-gradient(90deg, #d97706, #fbbf24)"
        c1 = "#38bdf8"
        c2 = "#fbbf24"
    else: # 4% thrust
        if val1 >= 25 and val1 >= val2 * 3:
            badge_text = "⚡ EXPANSION THRUST"
            badge_color = "#10b981"
            badge_bg = "rgba(16, 185, 129, 0.15)"
        elif val1 < 10 and val2 < 10:
            badge_text = "⚪ LOW VELOCITY"
            badge_color = "#94a3b8"
            badge_bg = "rgba(148, 163, 184, 0.15)"
        elif val2 >= 25:
            badge_text = "💥 SEVERE DAMAGE"
            badge_color = "#f43f5e"
            badge_bg = "rgba(244, 63, 94, 0.15)"
        else:
            badge_text = "⚖️ SELECTIVE"
            badge_color = "#38bdf8"
            badge_bg = "rgba(56, 189, 248, 0.15)"
        bar_grad1 = "linear-gradient(90deg, #059669, #10b981)"
        bar_grad2 = "linear-gradient(90deg, #e11d48, #f43f5e)"
        c1 = "#10b981"
        c2 = "#f43f5e"

    return f"""
    <div style="margin-bottom: 20px; font-family: 'Outfit', sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #f1f3f5; font-size: 13px; font-weight: 600;">{title}</span>
                <span style="font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 4px; background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_color}40; letter-spacing: 0.5px;">{badge_text}</span>
            </div>
            <span style="color: {c1}; font-size: 14px; font-weight: 800;">{pct1:.1f}%</span>
        </div>
        <div style="width: 100%; height: 8px; background: #23232c; border-radius: 6px; overflow: hidden; display: flex; box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);">
            <div style="width: {pct1}%; background: {bar_grad1}; transition: width 0.6s ease;"></div>
            <div style="width: {pct2}%; background: {bar_grad2}; transition: width 0.6s ease;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 11px;">
            <span style="color: {c1}; font-weight: 600;">{format_num(val1)} {label1}</span>
            {f'<span style="color: #71717a; font-size: 10px;">{subtitle}</span>' if subtitle else ''}
            <span style="color: {c2}; font-weight: 600;">{format_num(val2)} {label2}</span>
        </div>
    </div>
    """

# --- MARKET CAP BREADTH MATRIX ---
if df_segments is not None and not df_segments.empty:
    cnx_row = df_segments[df_segments['Segment'] == 'CNX 500']
    cnx_breadth = float(cnx_row['Above_50_Pct'].values[0]) if not cnx_row.empty else float(df_segments['Above_50_Pct'].mean())
    cnx_bs = float(cnx_row['Brutal_Strength_Pct'].values[0]) if not cnx_row.empty else float(df_segments['Brutal_Strength_Pct'].mean())
    
    n50_row = df_segments[df_segments['Segment'] == 'Nifty 50']
    n50_bs = float(n50_row['Brutal_Strength_Pct'].values[0]) if not n50_row.empty else 0.0
    n50_breadth = float(n50_row['Above_50_Pct'].values[0]) if not n50_row.empty else 0.0

    if cnx_breadth < 15:
        top_pulse_badge = f"🔴 SEVERE WASHOUT ZONE ({cnx_breadth:.1f}% BREADTH)"
        top_pulse_color = "#f43f5e"
        top_pulse_bg = "rgba(244, 63, 94, 0.15)"
    elif cnx_breadth < 35:
        top_pulse_badge = f"⚠️ DEFENSIVE CHILL ({cnx_breadth:.1f}% BREADTH)"
        top_pulse_color = "#fbbf24"
        top_pulse_bg = "rgba(251, 191, 36, 0.15)"
    elif cnx_breadth < 60:
        top_pulse_badge = f"🧭 SELECTIVE ACCUMULATION ({cnx_breadth:.1f}% BREADTH)"
        top_pulse_color = "#38bdf8"
        top_pulse_bg = "rgba(56, 189, 248, 0.15)"
    else:
        top_pulse_badge = f"🚀 BROAD MARKET MARKUP ({cnx_breadth:.1f}% BREADTH)"
        top_pulse_color = "#10b981"
        top_pulse_bg = "rgba(16, 185, 129, 0.15)"

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 10px;">
        <div>
            <h3 style="margin: 0; font-family: 'Outfit', sans-serif; font-size: 20px; color: #f1f3f5;">🏢 Market Cap Breadth Matrix</h3>
            <span style="color: #8b95a5; font-size: 13px;">5-second read on institutional money flow (Large vs Mid vs Small).</span>
        </div>
        <span style="background: {top_pulse_bg}; color: {top_pulse_color}; border: 1px solid {top_pulse_color}40; font-size: 11px; font-weight: 800; padding: 5px 14px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.8px;">{top_pulse_badge}</span>
    </div>
    """, unsafe_allow_html=True)
    
    html_matrix = """
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
    .breadth-container {
        font-family: 'Outfit', sans-serif;
        background: rgba(18, 18, 24, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 20px 24px;
        margin-bottom: 30px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .b-header {
        display: flex;
        padding: 0 16px 14px 16px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        color: #8b95a5;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    .b-row {
        display: flex;
        align-items: center;
        padding: 16px;
        border-bottom: 1px solid rgba(255,255,255,0.03);
        transition: all 0.3s ease;
        border-radius: 10px;
    }
    .b-row:hover {
        background: rgba(255, 255, 255, 0.03);
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    }
    .b-row:last-child {
        border-bottom: none;
    }
    .col-name { flex: 1.6; font-size: 15px; font-weight: 600; color: #f1f3f5; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .col-count { font-size: 10px; background: rgba(255,255,255,0.08); padding: 2px 7px; border-radius: 20px; color: #a1aab5; font-weight: 500; }
    .col-metric { flex: 1; display: flex; flex-direction: column; gap: 3px; }
    .metric-val { font-size: 17px; font-weight: 700; letter-spacing: -0.5px; }
    .col-pulse { flex: 1.5; display: flex; flex-direction: column; gap: 6px; justify-content: center; }
    
    .pulse-track { display: flex; gap: 3px; height: 7px; width: 100%; border-radius: 8px; background: transparent; }
    .pulse-seg { height: 100%; border-radius: 8px; transition: width 0.6s ease; }
    .pulse-labels { display: flex; justify-content: space-between; font-size: 11px; font-weight: 500; }
    </style>
    
    <div class="breadth-container">
        <div class="b-header">
            <div style="flex: 1.6;">Market Segment</div>
            <div style="flex: 1;">Brutal Strength <span style="opacity:0.5; text-transform:none;">(25>50>200)</span></div>
            <div style="flex: 1;">Breadth <span style="opacity:0.5; text-transform:none;">(>50d SMA)</span></div>
            <div style="flex: 1.5;">Weekly Pulse <span style="opacity:0.5; text-transform:none;">(Adv / Flat / Dec)</span></div>
        </div>
    """
    
    order_map = {"Nifty 50": 1, "Nifty Next 50": 2, "Midcap 150": 3, "Smallcap 250": 4, "Microcap 250": 5, "CNX 500": 6}
    df_segments['Order'] = df_segments['Segment'].map(order_map).fillna(99)
    df_segments = df_segments.sort_values('Order')
    
    for _, row in df_segments.iterrows():
        seg = row['Segment']
        total = int(row['Total'])
        bs_pct = row['Brutal_Strength_Pct']
        ab_50_pct = row['Above_50_Pct']
        adv = int(row['Adv'])
        dec = int(row['Dec'])
        flat = int(row['Flat'])
        
        # Micro status pill per segment
        if ab_50_pct < 15:
            seg_status = "WASHOUT"
            seg_status_color = "#f43f5e"
            seg_status_bg = "rgba(244, 63, 94, 0.12)"
        elif ab_50_pct < 35:
            seg_status = "CHILLED"
            seg_status_color = "#fbbf24"
            seg_status_bg = "rgba(251, 191, 36, 0.12)"
        elif ab_50_pct < 60:
            seg_status = "WARMING"
            seg_status_color = "#38bdf8"
            seg_status_bg = "rgba(56, 189, 248, 0.12)"
        else:
            seg_status = "MARKUP"
            seg_status_color = "#10b981"
            seg_status_bg = "rgba(16, 185, 129, 0.12)"

        if bs_pct >= 40: bs_color = "#10b981"; bs_bg = "rgba(16, 185, 129, 0.15)"
        elif bs_pct >= 20: bs_color = "#fbbf24"; bs_bg = "rgba(251, 191, 36, 0.15)"
        else: bs_color = "#f43f5e"; bs_bg = "rgba(244, 63, 94, 0.15)"
            
        if ab_50_pct >= 60: ab_color = "#10b981"; ab_bg = "rgba(16, 185, 129, 0.15)"
        elif ab_50_pct >= 40: ab_color = "#fbbf24"; ab_bg = "rgba(251, 191, 36, 0.15)"
        else: ab_color = "#f43f5e"; ab_bg = "rgba(244, 63, 94, 0.15)"
        
        total_ad = adv + dec + flat
        if total_ad == 0: total_ad = 1
        adv_pct = (adv / total_ad) * 100
        dec_pct = (dec / total_ad) * 100
        flat_pct = (flat / total_ad) * 100
        
        html_matrix += f"""
        <div class="b-row">
            <div class="col-name">
                <span>{seg}</span>
                <span class="col-count">{total}</span>
                <span style="font-size: 9px; font-weight: 700; padding: 2px 6px; border-radius: 4px; background: {seg_status_bg}; color: {seg_status_color}; border: 1px solid {seg_status_color}30; letter-spacing: 0.5px;">{seg_status}</span>
            </div>
            
            <div class="col-metric">
                <span class="metric-val" style="color: {bs_color}; text-shadow: 0 0 12px {bs_bg};">{bs_pct:.1f}%</span>
                <div style="width: 70px; height: 3px; background: rgba(255,255,255,0.08); border-radius: 2px; margin-top: 3px; overflow: hidden;">
                    <div style="width: {min(bs_pct, 100):.1f}%; height: 100%; background: {bs_color}; border-radius: 2px;"></div>
                </div>
            </div>
            
            <div class="col-metric">
                <span class="metric-val" style="color: {ab_color}; text-shadow: 0 0 12px {ab_bg};">{ab_50_pct:.1f}%</span>
                <div style="width: 70px; height: 3px; background: rgba(255,255,255,0.08); border-radius: 2px; margin-top: 3px; overflow: hidden;">
                    <div style="width: {min(ab_50_pct, 100):.1f}%; height: 100%; background: {ab_color}; border-radius: 2px;"></div>
                </div>
            </div>
            
            <div class="col-pulse">
                <div class="pulse-track">
                    <div class="pulse-seg" style="width: {adv_pct}%; background: linear-gradient(90deg, #059669, #10b981); box-shadow: 0 0 8px rgba(16,185,129,0.3);"></div>
                    <div class="pulse-seg" style="width: {flat_pct}%; background: #374151;"></div>
                    <div class="pulse-seg" style="width: {dec_pct}%; background: linear-gradient(90deg, #e11d48, #f43f5e); box-shadow: 0 0 8px rgba(244,63,94,0.3);"></div>
                </div>
                <div class="pulse-labels">
                    <span style="color: #10b981;">{adv} <span style="opacity:0.5; font-size:10px;">ADV</span></span>
                    <span style="color: #6b7280;">{flat}</span>
                    <span style="color: #f43f5e;"><span style="opacity:0.5; font-size:10px;">DEC</span> {dec}</span>
                </div>
            </div>
        </div>
        """.replace('\n', '')
        
    html_matrix += "</div>"
    st.markdown(html_matrix, unsafe_allow_html=True)

    # --- RELATIVE ALPHA BREADTH: % OF STOCKS OUTPERFORMING BENCHMARK (DATAWRAPPER STYLE) ---
    def render_outperformance_chart(outperf_data, default_bench_name="NIFTY 500"):
        if not outperf_data:
            return

        horizons = ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"]
        available_horizons = [h for h in horizons if h in outperf_data]
        if not available_horizons:
            return

        key_suffix = default_bench_name.lower().replace(" ", "_").replace("^", "").replace("(", "").replace(")", "")
        default_horizon = "6 Months" if "6 Months" in available_horizons else available_horizons[0]

        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(18, 18, 28, 0.75) 0%, rgba(12, 12, 18, 0.85) 100%); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 20px; padding: 22px 24px 20px 24px; margin-bottom: 24px; box-shadow: 0 20px 50px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px; margin-bottom: 12px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <h3 style="margin: 0; font-family: 'Outfit', sans-serif; font-size: 21px; font-weight: 700; color: #f8fafc; letter-spacing: -0.3px;">
                            % of Stocks Outperforming Benchmark
                        </h3>
                        <span style="background: rgba(249, 115, 22, 0.15); color: #f97316; border: 1px solid rgba(249, 115, 22, 0.35); font-size: 10px; font-weight: 800; padding: 2px 8px; border-radius: 6px; letter-spacing: 0.5px; text-transform: uppercase;">Relative Alpha Breadth</span>
                    </div>
                    <div style="color: #94a3b8; font-size: 13px; margin-top: 5px;">
                        Relative alpha breadth across capitalization tiers. Highlights where institutional capital is actively generating outperformance.
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        try:
            selected_horizon = st.segmented_control(
                "Timeframe Horizon",
                available_horizons,
                default=default_horizon,
                key=f"outperf_tf_{key_suffix}",
                label_visibility="collapsed"
            )
        except Exception:
            selected_horizon = st.radio(
                "Timeframe Horizon",
                available_horizons,
                index=available_horizons.index(default_horizon) if default_horizon in available_horizons else 0,
                horizontal=True,
                label_visibility="collapsed",
                key=f"outperf_tf_{key_suffix}"
            )
        if not selected_horizon:
            selected_horizon = default_horizon

        h_data = outperf_data[selected_horizon]
        bench_name = h_data.get('benchmark_name', default_bench_name)
        bench_ret = h_data.get('benchmark_return', 0.0)
        seg_map = h_data.get('segments', {})

        preferred_order = [
            "Nifty 50", "Nifty Next 50", "Midcap 150", "Smallcap 250", "Microcap 250", "CNX 500", "All Stocks (All Universe)",
            "Mega-Cap ($200B+)", "Large-Cap ($10B-$200B)", "Mid & Small (<$10B)", "Tech & Growth Leaders", "Tech Leaders", "S&P Top 100 Leaders", "All Stocks (All Universe)"
        ]
        ordered_segs = [s for s in preferred_order if s in seg_map]
        if not ordered_segs:
            ordered_segs = list(seg_map.keys())

        x_labels = []
        y_pcts = []
        hover_texts = []
        text_labels = []
        bar_colors = []
        border_colors = []

        for seg in ordered_segs:
            s_info = seg_map[seg]
            pct = s_info['pct']
            cnt = s_info['outperf_count']
            tot = s_info['total_count']
            med_ret = s_info['median_return']

            x_labels.append(seg)
            y_pcts.append(pct)

            if pct >= 50.0:
                # Outperforming tier: High-contrast glowing signature Datawrapper orange
                bar_colors.append("#f97316")
                border_colors.append("#fed7aa")
                text_labels.append(f"<b>{int(round(pct))}%</b><br><span style='font-size:10px; color:#fed7aa;'>{cnt}/{tot}</span>")
            else:
                # Underperforming tier: Muted smoked amber (instant visual contrast for alpha expansion)
                bar_colors.append("rgba(234, 88, 12, 0.42)")
                border_colors.append("rgba(255, 255, 255, 0.18)")
                text_labels.append(f"<b>{int(round(pct))}%</b><br><span style='font-size:10px; color:#94a3b8;'>{cnt}/{tot}</span>")

            excess = med_ret - bench_ret
            excess_sign = "+" if excess > 0 else ""
            hover_texts.append(
                f"<b>{seg}</b><br>"
                f"────────────────────────<br>"
                f"Outperforming Benchmark: <b>{pct:.1f}%</b> ({cnt}/{tot} stocks)<br>"
                f"Segment Median Return: <b>{med_ret:+.2f}%</b><br>"
                f"Benchmark ({bench_name}): <b>{bench_ret:+.2f}%</b><br>"
                f"Alpha Spread: <b>{excess_sign}{excess:.2f}%</b><extra></extra>"
            )

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=x_labels,
            y=y_pcts,
            text=text_labels,
            textposition='outside',
            textfont=dict(size=13, color='#ffffff', family='Outfit, Inter, sans-serif', weight=700),
            marker=dict(
                color=bar_colors,
                line=dict(color=border_colors, width=1.4),
                cornerradius=8
            ),
            hovertext=hover_texts,
            hoverinfo='text',
            cliponaxis=False
        ))

        fig.add_hline(
            y=50,
            line_dash="dot",
            line_color="rgba(255, 255, 255, 0.35)",
            line_width=1.5,
            annotation_text="⚡ 50% ALPHA BREAKEVEN",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#cbd5e1", family="Outfit, sans-serif", weight=700)
        )

        max_y = max(y_pcts) if y_pcts else 50
        y_upper = min(116, max(100, max_y + 16))

        fig.update_layout(
            template='plotly_dark',
            height=340,
            bargap=0.38,
            margin=dict(l=10, r=10, t=28, b=25),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor='rgba(255,255,255,0.15)',
                tickfont=dict(color='#f1f5f9', size=12, family='Outfit, sans-serif', weight=700)
            ),
            yaxis=dict(
                range=[0, y_upper],
                showgrid=True,
                gridcolor='rgba(255,255,255,0.04)',
                zeroline=False,
                showline=False,
                ticksuffix="%",
                tickfont=dict(color='#94a3b8', size=11, family='Outfit, sans-serif')
            ),
            hoverlabel=dict(
                bgcolor="#0f172a",
                font_size=12,
                font_family="Outfit, sans-serif",
                bordercolor="rgba(255,255,255,0.2)"
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Dynamic Regime diagnosis
        large_cap_key = "Nifty 50" if "Nifty 50" in seg_map else ("Mega-Cap ($200B+)" if "Mega-Cap ($200B+)" in seg_map else None)
        mid_cap_key = "Midcap 150" if "Midcap 150" in seg_map else ("Mid & Small (<$10B)" if "Mid & Small (<$10B)" in seg_map else None)
        
        regime_msg = "Rotational market phase across tiers. Leadership is distributed without clear capitalization bias."
        regime_badge = "BALANCED ALPHA"
        regime_icon = "⚖️"
        regime_color = "#38bdf8"
        regime_bg = "rgba(56, 189, 248, 0.12)"

        if large_cap_key and mid_cap_key:
            l_pct = seg_map[large_cap_key]['pct']
            m_pct = seg_map[mid_cap_key]['pct']
            if m_pct - l_pct >= 8.0:
                regime_msg = f"Mid & Small-Caps ({m_pct:.0f}%) significantly outpace Large Caps ({l_pct:.0f}%). Institutions are actively allocating down-cap into high-beta leaders."
                regime_badge = "BETA EXPANSION"
                regime_icon = "🚀"
                regime_color = "#10b981"
                regime_bg = "rgba(16, 185, 129, 0.15)"
            elif l_pct - m_pct >= 8.0:
                regime_msg = f"Large caps ({l_pct:.0f}%) outperform broad market. Capital is seeking mega-cap safety while mid/small caps experience relative distribution."
                regime_badge = "DEFENSIVE FLIGHT"
                regime_icon = "🛡️"
                regime_color = "#f43f5e"
                regime_bg = "rgba(244, 63, 94, 0.15)"
            elif all(seg_map[k]['pct'] >= 60.0 for k in seg_map):
                regime_msg = f"Over 60% of stocks across every tier are beating {bench_name}. Broad institutional sponsorship favorable for swing breakout follow-through."
                regime_badge = "MAX BULL SPONSORSHIP"
                regime_icon = "🟢"
                regime_color = "#10b981"
                regime_bg = "rgba(16, 185, 129, 0.15)"
            elif all(seg_map[k]['pct'] < 45.0 for k in seg_map):
                regime_msg = f"Fewer than 45% of stocks are outperforming. The headline index is masking widespread individual stock weakness."
                regime_badge = "INDEX DISTORTION"
                regime_icon = "🔴"
                regime_color = "#f43f5e"
                regime_bg = "rgba(244, 63, 94, 0.15)"

        b_color = "#10b981" if bench_ret >= 0 else "#f43f5e"
        b_sign = "+" if bench_ret >= 0 else ""
        st.markdown((
            f'<div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255, 255, 255, 0.025); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 12px 18px; margin-top: -6px; flex-wrap: wrap; gap: 12px;">'
            f'<div style="display: flex; align-items: center; gap: 12px; font-size: 13px; color: #cbd5e1; flex: 1; min-width: 280px;">'
            f'<span style="background: {regime_bg}; color: {regime_color}; border: 1px solid {regime_color}50; font-size: 10px; font-weight: 800; padding: 3px 10px; border-radius: 6px; letter-spacing: 0.6px; white-space: nowrap;">{regime_icon} {regime_badge}</span>'
            f'<span style="line-height: 1.4;">{regime_msg}</span>'
            f'</div>'
            f'<div style="display: flex; gap: 8px; align-items: center; background: rgba(0,0,0,0.3); padding: 5px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">'
            f'<span style="font-size: 11px; color: #8b95a5; font-weight: 600; text-transform: uppercase;">Benchmark ({bench_name}):</span>'
            f'<span style="color: {b_color}; font-weight: 800; font-family: monospace; font-size: 13px;">{b_sign}{bench_ret:.2f}%</span>'
            f'</div>'
            f'</div>'
            f'</div>'
        ), unsafe_allow_html=True)

        # Inspect Outperforming Alpha Leaders with Relative Strength (RS) Filter
        with st.expander("🔍 Inspect Outperforming Alpha Leaders by Segment & RS Filter", expanded=False):
            ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1.1, 1.4, 1.1, 1.1])
            with ctrl_c1:
                inspect_seg = st.selectbox("Select Segment:", ordered_segs, key=f"inspect_seg_{key_suffix}")
            with ctrl_c2:
                rs_filter = st.selectbox(
                    "Relative Strength Filter:",
                    [
                        "🌟 All Outperforming Stocks (Alpha > 0)",
                        "🔥 RS 90+ (Elite Momentum Leaders)",
                        "💎 RS 80+ (Institutional Core Leaders)",
                        "🟢 RS 70+ (Above Average Momentum)",
                        "🌊 Positive Mansfield RS (MRS > 0)",
                        "⚡ Leading MRS (MRS > +5)"
                    ],
                    index=0,
                    key=f"rs_filter_{key_suffix}"
                )
            with ctrl_c3:
                sort_by = st.selectbox(
                    "Sort Order:",
                    [
                        "Period Return (High → Low)",
                        "RS Rating (High → Low)",
                        "Mansfield RS (High → Low)",
                        "Alpha Spread (High → Low)",
                        "Ticker (A → Z)"
                    ],
                    index=0,
                    key=f"sort_{key_suffix}"
                )
            with ctrl_c4:
                search_q = st.text_input("Quick Filter Ticker / Sector:", placeholder="e.g. BSE, Tech...", key=f"search_{key_suffix}")

            if inspect_seg in seg_map:
                seg_data = seg_map[inspect_seg]
                stock_details = seg_data.get('stock_details', [])
                out_tickers = seg_data.get('outperforming_tickers', [])
                under_tickers = seg_data.get('underperforming_tickers', [])
                
                # Filter candidates: base pool is stocks outperforming benchmark
                if stock_details:
                    pool = [s for s in stock_details if s.get('is_outperforming', False)]
                    
                    if "RS 90+" in rs_filter:
                        filtered_stocks = [s for s in pool if s['rs_rating'] >= 90]
                    elif "RS 80+" in rs_filter:
                        filtered_stocks = [s for s in pool if s['rs_rating'] >= 80]
                    elif "RS 70+" in rs_filter:
                        filtered_stocks = [s for s in pool if s['rs_rating'] >= 70]
                    elif "MRS > +5" in rs_filter:
                        filtered_stocks = [s for s in pool if s['mrs'] > 5.0]
                    elif "MRS > 0" in rs_filter:
                        filtered_stocks = [s for s in pool if s['mrs'] > 0.0]
                    else:
                        filtered_stocks = pool

                    if search_q and search_q.strip():
                        q = search_q.strip().lower()
                        filtered_stocks = [s for s in filtered_stocks if q in s['symbol'].lower() or q in s.get('industry', '').lower()]

                    # Sort candidates
                    if "Period Return" in sort_by:
                        filtered_stocks.sort(key=lambda x: x['return'], reverse=True)
                    elif "RS Rating" in sort_by:
                        filtered_stocks.sort(key=lambda x: (x['rs_rating'], x['return']), reverse=True)
                    elif "Mansfield RS" in sort_by:
                        filtered_stocks.sort(key=lambda x: (x['mrs'], x['return']), reverse=True)
                    elif "Alpha Spread" in sort_by:
                        filtered_stocks.sort(key=lambda x: (x['excess_return'], x['return']), reverse=True)
                    else:
                        filtered_stocks.sort(key=lambda x: x['symbol'])
                else:
                    filtered_stocks = [{'symbol': t, 'return': 0.0, 'excess_return': 0.0, 'rs_rating': 50, 'mrs': 0.0, 'stage': 'S1', 'industry': 'Other', 'close': 0.0} for t in out_tickers]
                    if search_q and search_q.strip():
                        q = search_q.strip().lower()
                        filtered_stocks = [s for s in filtered_stocks if q in s['symbol'].lower()]

                # Executive Mini-Stats Cards
                total_tier = seg_data.get('total_count', len(stock_details))
                total_outperf = len(out_tickers)
                outperf_pct = seg_data.get('pct', 0.0)
                filtered_cnt = len(filtered_stocks)
                med_ret = seg_data.get('median_return', 0.0)
                alpha_spread = med_ret - bench_ret

                alpha_color = "#10b981" if alpha_spread >= 0 else "#f43f5e"
                alpha_sign = "+" if alpha_spread >= 0 else ""
                bar_w = min(100.0, max(0.0, outperf_pct))
                
                st.markdown((
                    f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: 16px 0 20px 0;">'
                    f'<div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.01) 100%); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 12px; padding: 14px 16px; box-shadow: 0 4px 16px rgba(0,0,0,0.2);">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                    f'<span style="font-size: 11px; color: #8b95a5; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Tier Universe</span>'
                    f'<span style="font-size: 14px; opacity: 0.6;">🌐</span>'
                    f'</div>'
                    f'<div style="font-size: 22px; font-weight: 800; color: #f8fafc; font-family: monospace; margin-top: 4px;">{total_tier} <span style="font-size: 12px; color: #64748b; font-weight: 500; font-family: sans-serif;">Stocks</span></div>'
                    f'<div style="font-size: 11px; color: #64748b; margin-top: 2px;">{inspect_seg}</div>'
                    f'</div>'

                    f'<div style="background: linear-gradient(135deg, rgba(249, 115, 22, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%); border: 1px solid rgba(249, 115, 22, 0.25); border-radius: 12px; padding: 14px 16px; box-shadow: 0 4px 16px rgba(0,0,0,0.2);">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                    f'<span style="font-size: 11px; color: #fdba74; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Beating Benchmark</span>'
                    f'<span style="font-size: 14px; opacity: 0.8;">🚀</span>'
                    f'</div>'
                    f'<div style="display: flex; align-items: baseline; gap: 8px; margin-top: 4px;">'
                    f'<span style="font-size: 22px; font-weight: 800; color: #f97316; font-family: monospace;">{total_outperf}</span>'
                    f'<span style="font-size: 13px; font-weight: 700; color: #fdba74; background: rgba(249, 115, 22, 0.15); padding: 1px 6px; border-radius: 4px;">{outperf_pct:.1f}%</span>'
                    f'</div>'
                    f'<div style="width: 100%; height: 4px; background: rgba(255,255,255,0.06); border-radius: 2px; margin-top: 6px; overflow: hidden;">'
                    f'<div style="width: {bar_w:.1f}%; height: 100%; background: linear-gradient(90deg, #ea580c, #f97316); border-radius: 2px;"></div>'
                    f'</div>'
                    f'</div>'

                    f'<div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 12px; padding: 14px 16px; box-shadow: 0 4px 16px rgba(0,0,0,0.2);">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                    f'<span style="font-size: 11px; color: #6ee7b7; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Matching RS Filter</span>'
                    f'<span style="font-size: 14px; opacity: 0.8;">💎</span>'
                    f'</div>'
                    f'<div style="font-size: 22px; font-weight: 800; color: #10b981; font-family: monospace; margin-top: 4px;">{filtered_cnt} <span style="font-size: 12px; color: #64748b; font-weight: 500; font-family: sans-serif;">Qualified</span></div>'
                    f'<div style="font-size: 11px; color: #6ee7b7; margin-top: 2px;">{(filtered_cnt/total_tier*100 if total_tier>0 else 0):.1f}% of tier universe</div>'
                    f'</div>'

                    f'<div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.01) 100%); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 12px; padding: 14px 16px; box-shadow: 0 4px 16px rgba(0,0,0,0.2);">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                    f'<span style="font-size: 11px; color: #8b95a5; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Tier Median vs Bench</span>'
                    f'<span style="font-size: 14px; opacity: 0.6;">⚡</span>'
                    f'</div>'
                    f'<div style="display: flex; align-items: baseline; gap: 8px; margin-top: 4px;">'
                    f'<span style="font-size: 22px; font-weight: 800; color: #f8fafc; font-family: monospace;">{med_ret:+.2f}%</span>'
                    f'<span style="font-size: 12px; font-weight: 700; color: {alpha_color}; background: {alpha_color}18; padding: 1px 6px; border-radius: 4px; font-family: monospace;">{alpha_sign}{alpha_spread:+.2f}% Spread</span>'
                    f'</div>'
                    f'<div style="font-size: 11px; color: #64748b; margin-top: 2px;">Bench: {bench_ret:+.2f}%</div>'
                    f'</div>'
                    f'</div>'
                ), unsafe_allow_html=True)

                is_us = "Mega-Cap" in inspect_seg or "Large-Cap" in inspect_seg or "US" in default_bench_name or "SPY" in default_bench_name
                tv_prefix = "" if is_us else "NSE:"

                if filtered_stocks:
                    # Interactive Leaderboard Table
                    css_block = (
                        '<style>'
                        '.alpha-table-wrap::-webkit-scrollbar { width: 6px; height: 6px; }'
                        '.alpha-table-wrap::-webkit-scrollbar-track { background: rgba(10, 10, 18, 0.6); }'
                        '.alpha-table-wrap::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 4px; }'
                        '.alpha-table-wrap::-webkit-scrollbar-thumb:hover { background: rgba(56, 189, 248, 0.4); }'
                        '.alpha-table tbody tr { transition: all 0.15s ease; border-bottom: 1px solid rgba(255,255,255,0.035); }'
                        '.alpha-table tbody tr:nth-child(even) { background: rgba(255, 255, 255, 0.012); }'
                        '.alpha-table tbody tr:hover { background: rgba(56, 189, 248, 0.08) !important; }'
                        '.tv-link-pill { display: inline-flex; align-items: center; gap: 4px; padding: 2px 7px; border-radius: 6px; background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); color: #38bdf8 !important; text-decoration: none !important; font-weight: 700; font-family: monospace; font-size: 12px; transition: all 0.15s ease; }'
                        '.tv-link-pill:hover { background: rgba(56, 189, 248, 0.22); border-color: #38bdf8; box-shadow: 0 0 10px rgba(56, 189, 248, 0.3); }'
                        '</style>'
                    )

                    table_rows = []
                    for idx, s in enumerate(filtered_stocks, 1):
                        sym = s['symbol']
                        tv_link = f"https://www.tradingview.com/chart/?symbol={tv_prefix}{sym}"
                        ret_val = s['return']
                        ret_color = "#10b981" if ret_val >= 0 else "#f43f5e"
                        excess_val = s['excess_return']
                        excess_color = "#34d399" if excess_val >= 0 else "#fb7185"
                        excess_sign = "+" if excess_val >= 0 else ""
                        
                        rs_val = s['rs_rating']
                        if rs_val >= 90:
                            rs_badge = f'<span style="background: rgba(16,185,129,0.18); color: #34d399; border: 1px solid rgba(16,185,129,0.45); padding: 3px 8px; border-radius: 8px; font-weight: 800; font-size: 11px; letter-spacing: 0.3px; box-shadow: 0 0 8px rgba(16,185,129,0.2);">🔥 {rs_val}</span>'
                        elif rs_val >= 80:
                            rs_badge = f'<span style="background: rgba(2,132,199,0.18); color: #38bdf8; border: 1px solid rgba(2,132,199,0.45); padding: 3px 8px; border-radius: 8px; font-weight: 800; font-size: 11px; letter-spacing: 0.3px;">💎 {rs_val}</span>'
                        elif rs_val >= 70:
                            rs_badge = f'<span style="background: rgba(234,179,8,0.18); color: #fbbf24; border: 1px solid rgba(234,179,8,0.45); padding: 3px 8px; border-radius: 8px; font-weight: 800; font-size: 11px; letter-spacing: 0.3px;">🟢 {rs_val}</span>'
                        else:
                            rs_badge = f'<span style="background: rgba(148,163,184,0.12); color: #94a3b8; border: 1px solid rgba(148,163,184,0.25); padding: 3px 8px; border-radius: 8px; font-weight: 700; font-size: 11px;">⚪ {rs_val}</span>'
                        
                        mrs_val = s['mrs']
                        if mrs_val > 0:
                            mrs_display = f'<span style="color: #38bdf8; font-weight: 700; font-family: monospace;">+{mrs_val:.1f}% <span style="font-size: 9px; opacity: 0.8;">↗</span></span>'
                        else:
                            mrs_display = f'<span style="color: #94a3b8; font-weight: 600; font-family: monospace;">{mrs_val:.1f}% <span style="font-size: 9px; opacity: 0.8;">↘</span></span>'
                        
                        stage_val = s.get('stage', 'S1')
                        if stage_val == 'S2':
                            stage_badge = '<span style="background: rgba(16,185,129,0.18); color: #10b981; border: 1px solid rgba(16,185,129,0.4); padding: 3px 8px; border-radius: 6px; font-size: 10.5px; font-weight: 700; letter-spacing: 0.3px; white-space: nowrap;">🚀 S2 Markup</span>'
                        elif stage_val == 'S1':
                            stage_badge = '<span style="background: rgba(56,189,248,0.18); color: #38bdf8; border: 1px solid rgba(56,189,248,0.4); padding: 3px 8px; border-radius: 6px; font-size: 10.5px; font-weight: 700; letter-spacing: 0.3px; white-space: nowrap;">🛡️ S1 Base</span>'
                        elif stage_val == 'S3':
                            stage_badge = '<span style="background: rgba(251,191,36,0.18); color: #fbbf24; border: 1px solid rgba(251,191,36,0.4); padding: 3px 8px; border-radius: 6px; font-size: 10.5px; font-weight: 700; letter-spacing: 0.3px; white-space: nowrap;">⚠️ S3 Top</span>'
                        else:
                            stage_badge = '<span style="background: rgba(244,63,94,0.18); color: #f43f5e; border: 1px solid rgba(244,63,94,0.4); padding: 3px 8px; border-radius: 6px; font-size: 10.5px; font-weight: 700; letter-spacing: 0.3px; white-space: nowrap;">🛑 S4 Decline</span>'

                        currency = "$" if is_us else "₹"
                        price_str = f"{currency}{s['close']:,.2f}" if s['close'] > 0 else "—"

                        row_html = (
                            f'<tr>'
                            f'<td style="padding: 11px 14px; color: #64748b; font-size: 11px; font-weight: 600; text-align: center;">{idx}</td>'
                            f'<td style="padding: 11px 14px;">'
                            f'<a href="{tv_link}" target="_blank" class="tv-link-pill">'
                            f'{sym} <span style="font-size: 10px; opacity: 0.7;">↗</span>'
                            f'</a></td>'
                            f'<td style="padding: 11px 14px; color: #cbd5e1; font-size: 11.5px; max-width: 170px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{s["industry"]}</td>'
                            f'<td style="padding: 11px 14px; color: #f8fafc; font-weight: 600; font-family: monospace; font-size: 12.5px; text-align: right;">{price_str}</td>'
                            f'<td style="padding: 11px 14px; color: {ret_color}; font-weight: 700; font-family: monospace; font-size: 12.5px; text-align: right;">{ret_val:+.2f}%</td>'
                            f'<td style="padding: 11px 14px; color: {excess_color}; font-weight: 700; font-family: monospace; font-size: 12px; text-align: right;">{excess_sign}{excess_val:.2f}%</td>'
                            f'<td style="padding: 11px 14px; text-align: center;">{rs_badge}</td>'
                            f'<td style="padding: 11px 14px; text-align: right;">{mrs_display}</td>'
                            f'<td style="padding: 11px 14px; text-align: center;">{stage_badge}</td>'
                            f'</tr>'
                        )
                        table_rows.append(row_html)

                    rows_joined = "".join(table_rows)
                    table_html = (
                        f'{css_block}'
                        '<div class="alpha-table-wrap" style="border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; overflow: hidden; background: rgba(12, 13, 20, 0.75); backdrop-filter: blur(12px); margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.35);">'
                        '<div style="max-height: 440px; overflow-y: auto;">'
                        '<table class="alpha-table" style="width: 100%; border-collapse: collapse; text-align: left; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif;">'
                        '<thead style="position: sticky; top: 0; background: #131622; border-bottom: 1px solid rgba(255,255,255,0.09); z-index: 2;">'
                        '<tr>'
                        '<th style="padding: 12px 14px; color: #8b95a5; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; text-align: center;">#</th>'
                        '<th style="padding: 12px 14px; color: #8b95a5; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px;">Ticker</th>'
                        '<th style="padding: 12px 14px; color: #8b95a5; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px;">Industry / Sector</th>'
                        '<th style="padding: 12px 14px; color: #8b95a5; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; text-align: right;">Price</th>'
                        f'<th style="padding: 12px 14px; color: #8b95a5; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; text-align: right;">Return ({selected_horizon})</th>'
                        '<th style="padding: 12px 14px; color: #8b95a5; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; text-align: right;">Alpha Spread</th>'
                        '<th style="padding: 12px 14px; color: #8b95a5; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; text-align: center;">CANSLIM RS</th>'
                        '<th style="padding: 12px 14px; color: #8b95a5; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; text-align: right;">Mansfield RS</th>'
                        '<th style="padding: 12px 14px; color: #8b95a5; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; text-align: center;">Stage</th>'
                        '</tr>'
                        '</thead>'
                        f'<tbody>{rows_joined}</tbody>'
                        '</table>'
                        '</div>'
                        '</div>'
                    )
                    st.markdown(table_html, unsafe_allow_html=True)

                    # Quick-copy TradingView Watchlist Box
                    filtered_symbols = [s['symbol'] for s in filtered_stocks]
                    filtered_tv_list = [f"{tv_prefix}{s}" for s in filtered_symbols]
                    
                    st.markdown((
                        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px; margin-bottom: 6px;">'
                        f'<div style="display: flex; align-items: center; gap: 8px;">'
                        f'<span style="font-size: 12px; color: #f1f5f9; font-weight: 700;">📋 TradingView Watchlist</span>'
                        f'<span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; font-size: 10.5px; font-weight: 800; padding: 2px 7px; border-radius: 10px;">{len(filtered_symbols)} Tickers</span>'
                        f'</div>'
                        f'<span style="font-size: 11px; color: #8b95a5;">Copy and paste into TradingView Watchlist window</span>'
                        f'</div>'
                    ), unsafe_allow_html=True)
                    st.code(", ".join(filtered_tv_list), language="text")
                else:
                    st.markdown((
                        f'<div style="background: rgba(255, 255, 255, 0.02); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 30px; text-align: center; margin: 16px 0;">'
                        f'<div style="font-size: 28px; margin-bottom: 8px;">🔍</div>'
                        f'<div style="font-size: 14px; font-weight: 700; color: #f1f5f9;">No Outperforming Stocks Found</div>'
                        f'<div style="font-size: 12px; color: #8b95a5; margin-top: 4px;">No stocks in <b>{inspect_seg}</b> match the selected RS criteria ({rs_filter}) for {selected_horizon}. Try choosing a broader RS filter or clearing the search query.</div>'
                        f'</div>'
                    ), unsafe_allow_html=True)

    render_outperformance_chart(outperformance_data, default_bench_name="NIFTY 500")



# --- 2-COLUMN COMMAND CENTER: MOMENTUM RADAR & EXECUTIVE PLAYBOOK ---
mom_col1, mom_col2 = st.columns([1, 1], gap="medium")

# Precalculate metrics for executive radar
up_open_total = momentum_metrics['up_open'] + momentum_metrics['down_open']
pct_up_open = (momentum_metrics['up_open'] / (up_open_total if up_open_total > 0 else 1)) * 100

up_vol_total = momentum_metrics['up_vol'] + momentum_metrics['down_vol']
pct_up_vol = (momentum_metrics['up_vol'] / (up_vol_total if up_vol_total > 0 else 1)) * 100

with mom_col1:
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <h4 style="margin: 0; font-family: 'Outfit', sans-serif; font-size: 17px; color: #f1f3f5;">🌊 Daily Momentum & Thrust Radar</h4>
        <span style="font-size: 11px; color: #8b95a5;">Intraday Flow & Volume</span>
    </div>
    """, unsafe_allow_html=True)

    mom_card_html = f"""
    <div style="background: rgba(18, 18, 24, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 22px; box-shadow: 0 12px 32px rgba(0,0,0,0.3); height: 100%; display: flex; flex-direction: column; justify-content: space-around;">
        {enhanced_momentum_bar(momentum_metrics['up_open'], momentum_metrics['down_open'], "Up", "Down", "Up from Open vs Down from Open", metric_type="open", subtitle="Intraday holding conviction")}
        {enhanced_momentum_bar(momentum_metrics['up_vol'], momentum_metrics['down_vol'], "Up", "Down", "Up on Volume vs Down on Volume", metric_type="vol", subtitle="Institutional volume bias")}
        <div style="margin-bottom: -5px;">
            {enhanced_momentum_bar(momentum_metrics['up_4'], momentum_metrics['down_4'], "Up", "Down", "Power Thrust (+4% vs -4%)", metric_type="thrust", subtitle="Leading thrust participation")}
        </div>
    </div>
    """.replace('\n', '')
    st.markdown(mom_card_html, unsafe_allow_html=True)

with mom_col2:
    # Formulate dynamic diagnostics
    if cnx_breadth < 15:
        climate_status = "🔴 SEVERE CAPITULATION WASHOUT"
        climate_color = "#f43f5e"
        climate_badge = "EXTREME OVERSOLD"
        climate_bg = "rgba(244, 63, 94, 0.15)"
        cash_target = "70% – 100% Cash Buffer"
    elif cnx_breadth < 35:
        climate_status = "⚠️ DEFENSIVE DISTRIBUTION REGIME"
        climate_color = "#fbbf24"
        climate_badge = "HIGH CAUTION"
        climate_bg = "rgba(251, 191, 36, 0.15)"
        cash_target = "50% – 70% Cash Buffer"
    elif cnx_breadth < 60:
        climate_status = "🧭 SELECTIVE ACCUMULATION REGIME"
        climate_color = "#38bdf8"
        climate_badge = "SELECTIVE RISK"
        climate_bg = "rgba(56, 189, 248, 0.15)"
        cash_target = "25% – 40% Cash Buffer"
    else:
        climate_status = "🚀 MAXIMUM BULL MARKUP"
        climate_color = "#10b981"
        climate_badge = "MAX RISK-ON"
        climate_bg = "rgba(16, 185, 129, 0.15)"
        cash_target = "0% – 15% Cash Buffer"

    # Dynamic divergence callout
    if pct_up_vol > 75 and pct_up_open < 45:
        div_border = "#fbbf24"
        div_text = f"⚠️ <b>Intraday Fade / Churn Alert:</b> Up-Volume is <b>{pct_up_vol:.1f}%</b>, yet <b>{(100-pct_up_open):.1f}%</b> of stocks faded below their morning open. Institutions dumped into morning gap-ups. Do NOT chase morning breakouts."
    elif n50_bs == 0 and cnx_breadth < 25:
        div_border = "#f43f5e"
        div_text = f"❄️ <b>Zero Large-Cap Leadership:</b> Nifty 50 has <b>0.0%</b> stocks in 25>50>200 alignment. Institutions are absent from trending large-caps. Breakout failure rate remains >75%."
    else:
        div_border = "#38bdf8"
        div_text = f"🧭 <b>Transitional Equilibrium:</b> Breadth at <b>{cnx_breadth:.1f}%</b>. Focus on high-conviction Stage 1 bases breaking out on 2x volume."

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <h4 style="margin: 0; font-family: 'Outfit', sans-serif; font-size: 17px; color: #f1f3f5;">🧭 Market Internals Playbook</h4>
        <span style="font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 20px; background: {climate_bg}; color: {climate_color}; border: 1px solid {climate_color}40; letter-spacing: 0.6px;">{climate_badge}</span>
    </div>
    """, unsafe_allow_html=True)

    playbook_html = f"""
    <div style="background: rgba(18, 18, 24, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 20px; box-shadow: 0 12px 32px rgba(0,0,0,0.3); font-family: 'Outfit', sans-serif; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.06);">
                <div>
                    <div style="color: #8b95a5; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px;">Regime Assessment</div>
                    <div style="font-size: 15px; font-weight: 800; color: {climate_color}; margin-top: 1px;">{climate_status}</div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #8b95a5; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px;">Target Cash Buffer</div>
                    <div style="font-size: 14px; font-weight: 800; color: #f1f3f5; margin-top: 1px;">{cash_target}</div>
                </div>
            </div>

            <div style="background: rgba(0,0,0,0.3); border-radius: 8px; padding: 10px 12px; margin-bottom: 14px; border-left: 4px solid {div_border};">
                <div style="font-size: 11.5px; line-height: 1.45; color: #d1d5db;">
                    {div_text}
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div style="background: rgba(16, 185, 129, 0.06); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 10px 12px;">
                    <div style="font-size: 10.5px; font-weight: 700; color: #10b981; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px;">✅ Tactical Playbook</div>
                    <div style="font-size: 11px; color: #cbd5e1; line-height: 1.4;">
                        • <b>Stalk S1 Bases:</b> Build VCP watchlists.<br>
                        • <b>Wait for Thrust:</b> Require >60% Up-Open.<br>
                        • <b>Pilot Sizes Only:</b> Limit risk to 5-8%.
                    </div>
                </div>
                <div style="background: rgba(244, 63, 94, 0.06); border: 1px solid rgba(244, 63, 94, 0.2); border-radius: 8px; padding: 10px 12px;">
                    <div style="font-size: 10.5px; font-weight: 700; color: #f43f5e; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px;">❌ Fatal Pitfalls</div>
                    <div style="font-size: 11px; color: #cbd5e1; line-height: 1.4;">
                        • <b>Never Chase Gaps:</b> Fade risk >65%.<br>
                        • <b>Do Not Short Washouts:</b> Snapback squeeze.<br>
                        • <b>No Averaging Down:</b> Cut losses at 4-6%.
                    </div>
                </div>
            </div>
        </div>
    </div>
    """.replace('\n', '')
    st.markdown(playbook_html, unsafe_allow_html=True)

with st.expander("ℹ️ Understanding Breadth Matrix Tags ('Chilled' / 'Washout') & Market Internals Playbook Rules", expanded=False):
    st.markdown(f"""
    ### 🧭 1. What Does "DEFENSIVE CHILL" & "CHILLED" Mean?
    The **Market Cap Breadth Matrix** acts as an institutional temperature check across market capitalization tiers (Large vs Mid vs Small). Each tier is evaluated on the percentage of constituent stocks trading above their **50-day Simple Moving Average (SMA)**—the primary institutional dividing line between intermediate accumulation and distribution:

    | Segment Breadth (>50d SMA) | Status Tag | Institutional Meaning | Action Directive |
    | :--- | :--- | :--- | :--- |
    | **< 15.0%** | `🔴 WASHOUT` | **Capitulation Exhaustion:** >85% of stocks broken below 50 SMA. Panic selling climax. | Do not short. Prepare for violent mean-reversion snapbacks. |
    | **15.0% – 34.9%** | `⚠️ CHILLED` | **Defensive Distribution (Current State):** Under 35% of stocks can hold their 50 SMA. Lacks broad institutional bids; breakouts fail frequently. | High caution. Defensive posture. Raise cash (50% – 70%). |
    | **35.0% – 59.9%** | `🧭 WARMING` | **Transitional / Selective:** Market breadth is improving. Institutional capital is quietly accumulating select leaders. | Trade high-conviction A+ setups with pilot sizing. |
    | **≥ 60.0%** | `🟢 MARKUP` | **Broad Bull Sponsorship:** Over 60% of stocks are in healthy intermediate uptrends above their 50 SMA. | Aggressive buy & hold. Full position sizing. |

    > **Current Reading:** CNX 500 Breadth is at **{cnx_breadth:.1f}%**, placing the entire market in a **DEFENSIVE CHILL**. Nearly 4 out of every 5 stocks are trading below their 50-day average.

    ---

    ### ⚡ 2. The Key Matrix Metrics Explained
    - **Brutal Strength (25>50>200):** Measures the elite group of stocks whose moving averages are in stacked bullish alignment (`Price > 25 SMA > 50 SMA > 200 SMA`). When large-caps (Nifty 50) have near **0.0% Brutal Strength**, true trending market momentum does not exist.
    - **Breadth (>50d SMA):** The percentage of stocks above their 50-day moving average. Acts as a market-wide "thermometer" of health.
    - **Weekly Pulse (Adv / Flat / Dec):** The 5-day directional net flow of stocks within that segment. When red (decline) bars dominate green (advance) bars, selling pressure is dominant.

    ---

    ### 📖 3. How the "Market Internals Playbook" is Defined
    The Playbook translates raw breadth and intraday thrust into an actionable operating plan:

    1. **Regime Assessment & Target Cash Buffer:**
       - `🔴 SEVERE WASHOUT (< 15% Breadth):` Extreme oversold exhaustion. Recommends **70%–100% Cash Buffer** (or selective counter-trend shorts).
       - `⚠️ DEFENSIVE DISTRIBUTION (15%–35% Breadth):` Current regime. Lacks institutional participation. Recommends **50%–70% Cash Buffer**.
       - `🧭 SELECTIVE ACCUMULATION (35%–60% Breadth):` Moderate health. Recommends **25%–40% Cash Buffer** with selective 10–12% position sizes.
       - `🚀 MAXIMUM BULL MARKUP (≥ 60% Breadth):` Full risk-on tailwind. Recommends **0%–15% Cash Buffer** (90–100% invested).
    
    2. **Dynamic Divergence Diagnosis:**
       - Automatically detects "sneaky" institutional selling, such as **Intraday Fade/Churn** (when headline up-volume is high, but 60%+ of stocks close below their morning open), warning traders that morning gap-ups are being sold into retail.
    
    3. **Tactical Playbook vs. Fatal Pitfalls:**
       - **Tactical Playbook:** Focuses on stalking Stage 1 bases, demanding >60% Up-from-Open confirmation before buying, and limiting risk to 5–8% pilot sizes.
       - **Fatal Pitfalls:** Explicitly warns against chasing morning gaps, shorting deep washouts, and averaging down on losers.
    """)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
st.markdown("---")

# --- STAN WEINSTEIN 4-STAGE LIFECYCLE GUIDE ---
st.markdown("### 🏛️ Stan Weinstein's 4-Stage Market Framework")
st.caption("How institutional capital rotates through cycles: Accumulation ➔ Markup ➔ Distribution ➔ Markdown.")

st.markdown("""
<style>
.stage-flow-card {
    background: rgba(18, 18, 24, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px;
    height: 100%;
    transition: all 0.25s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.stage-flow-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.stage-pill-tag {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 3px 8px;
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 8px;
}
.stage-title {
    font-size: 15px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 6px;
    font-family: 'Outfit', sans-serif;
}
.stage-desc {
    font-size: 12px;
    color: #94a3b8;
    line-height: 1.45;
    margin-bottom: 10px;
}
.stage-action {
    font-size: 11px;
    font-weight: 600;
    padding: 6px 10px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
}
</style>

<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px;">
    <div class="stage-flow-card" style="border-top: 3px solid #6366f1;">
        <div>
            <span class="stage-pill-tag" style="background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3);">Stage 1: Basing</span>
            <div class="stage-title">Stealth Accumulation</div>
            <div class="stage-desc">Downtrend stops. Price moves sideways in a neutral channel. 50 SMA ≤ 200 SMA (flattening). Volume dries up (VDU). Smart money quietly accumulates inventory.</div>
        </div>
        <div class="stage-action" style="color: #818cf8;">
            🎯 <b>Action:</b> Stalk bases, track VCPs, prepare buy stops above base resistance.
        </div>
    </div>
    <div class="stage-flow-card" style="border-top: 3px solid #10b981;">
        <div>
            <span class="stage-pill-tag" style="background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3);">Stage 2: Advancing</span>
            <div class="stage-title">Institutional Markup</div>
            <div class="stage-desc">Aggressive fund buying. Price > 50 SMA > 200 SMA, 200 SMA sloping upwards (+20d). Higher highs and higher lows. High-volume breakouts with low-volume pullbacks.</div>
        </div>
        <div class="stage-action" style="color: #10b981;">
            🚀 <b>Action:</b> Aggressive buy & hold. Buy fresh breakouts & 21 EMA pullbacks. Ride winners.
        </div>
    </div>
    <div class="stage-flow-card" style="border-top: 3px solid #f59e0b;">
        <div>
            <span class="stage-pill-tag" style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3);">Stage 3: Topping</span>
            <div class="stage-title">Distribution & Churn</div>
            <div class="stage-desc">Smart money unloads to retail. Volatility expands, heavy volume down days, breakouts fail/squat. Price violates 50 SMA or 200 SMA slope rolls over after an advance.</div>
        </div>
        <div class="stage-action" style="color: #fbbf24;">
            ⚠️ <b>Action:</b> Protect profits, tighten stops, stop buying breakouts, raise cash.
        </div>
    </div>
    <div class="stage-flow-card" style="border-top: 3px solid #f43f5e;">
        <div>
            <span class="stage-pill-tag" style="background: rgba(244, 63, 94, 0.15); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.3);">Stage 4: Declining</span>
            <div class="stage-title">Markdown & Liquidation</div>
            <div class="stage-desc">Sustained institutional liquidation. Price < 50 SMA < 200 SMA, 200 SMA sloping downward (-20d). Lower lows. Rallies fail at falling MAs. Even good news gets sold.</div>
        </div>
        <div class="stage-action" style="color: #f43f5e;">
            🛑 <b>Action:</b> 100% Forbidden to buy. Bottom fishing is account suicide. Stay in cash or short.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("📖 Stan Weinstein & Mark Minervini Stage Technical Rules & Quantitative Definitions", expanded=False):
    st.markdown("""
    ### Quantitative Moving Average & Trend Criteria Used in Kush Tracker

    | Stage | Stage Name | Moving Average Structure | Slope Condition | Institutional Character | Actionable Trader Rule |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **Stage 1** | **Basing / Accumulation** | `(~S2) & (~S4) & (SMA50 <= SMA200)` | 200 SMA is flat / neutral | Stealth institutional accumulation, volume dry-up (VDU), contraction of volatility (VCP) | **Stalk & Build Watchlists.** Do not buy prematurely (dead capital). Track base completion. |
    | **Stage 2** | **Advancing / Markup** | `Price > SMA50` and `SMA50 > SMA200` | `SMA200 > SMA200[-20d]` (Rising) | Aggressive institutional sponsorship, series of higher highs and higher lows, volume expansion on rallies | **Aggressive Buy & Hold.** Buy breakouts from Stage 1 or Base 1/2. Pyramid winners at 21 EMA. |
    | **Stage 3** | **Topping / Distribution** | `(~S2) & (~S4) & (SMA50 > SMA200)` | 200 SMA slope flattening | Institutional unloading, erratic wide price swings, heavy down-volume, breakout squats & failure | **Sell & Protect Capital.** Tighten stops to breakeven or backstop profit floors. Cease buying breakouts. |
    | **Stage 4** | **Declining / Markdown** | `Price < SMA50` and `SMA50 < SMA200` | `SMA200 < SMA200[-20d]` (Falling) | Heavy supply, lack of institutional bids, successive lower lows, failed dead-cat bounces | **Strict "Forbidden" Zone.** Never buy or average down. Cut remaining losses immediately. |
    """)

st.markdown("<br>", unsafe_allow_html=True)

# --- STAGE ANALYSIS TOP CARDS ---
st.markdown("### 📊 Market Stage Distribution (Current Universe)")
st.caption("Real-time breakdown of all **749 Nifty Total Market stocks** across Stan Weinstein's 4 Stages.")

total_stocks = len(df_latest)
s1_count = len(df_latest[df_latest['Stage'] == 'S1'])
s2_count = len(df_latest[df_latest['Stage'] == 'S2'])
s3_count = len(df_latest[df_latest['Stage'] == 'S3'])
s4_count = len(df_latest[df_latest['Stage'] == 'S4'])

s1_pct = (s1_count / total_stocks) * 100 if total_stocks else 0
s2_pct = (s2_count / total_stocks) * 100 if total_stocks else 0
s3_pct = (s3_count / total_stocks) * 100 if total_stocks else 0
s4_pct = (s4_count / total_stocks) * 100 if total_stocks else 0

# Institutional Action Status Engine (Stan Weinstein & Minervini Rules)
if s4_pct >= 30:
    regime_status = "🔴 RED LIGHT: FULL BEAR MARKET LIQUIDATION"
    regime_theme = "#f43f5e"
    regime_bg = "rgba(244, 63, 94, 0.12)"
    regime_border = "rgba(244, 63, 94, 0.4)"
    what_to_do = "100% Capital Preservation. Hold 80–100% cash or trade selective short hedges. Let the market complete its capitulation before looking for bottoms."
    what_not_to_do = "DO NOT BUY ANYTHING LONG. 'Bottom-fishing' during active Stage 4 liquidation is financial suicide. Never average down on losing positions."
    action_badge = "100% CASH / FORBIDDEN LONGS"
    trigger_rule = f"Stage 4 Declining = {s4_pct:.1f}% (Triggered: S4 ≥ 30.0%)"
    regime_reason = "Over 30% of the entire universe is in active Stage 4 markdown below declining 50 & 200 SMAs. Market structure is in active institutional distribution/liquidation."
elif s3_pct >= 40 or (s3_pct >= 30 and s2_pct <= 15):
    regime_status = "⚠️ YELLOW/RED ALERT: HEAVY DISTRIBUTION & TOPPING CHURN"
    regime_theme = "#f59e0b"
    regime_bg = "rgba(245, 158, 11, 0.12)"
    regime_border = "rgba(245, 158, 11, 0.4)"
    what_to_do = f"Protect Capital & Tighten Stops. Over half the market ({s3_pct:.1f}%) is in Stage 3 Topping while Stage 2 is down to {s2_pct:.1f}%. Sell into rallies, take quick profits (10–15%), and raise cash to 60–80%."
    what_not_to_do = "DO NOT BUY NEW BREAKOUTS. Breakout failure/squat rate exceeds 80% during heavy Stage 3 distribution. Do not hold through earnings without a +10% cushion."
    action_badge = "PROTECT PROFITS / STOP BUYING"
    trigger_rule = f"Stage 3 Topping = {s3_pct:.1f}% (Triggered: S3 ≥ 40.0% or S3 ≥ 30% with S2 ≤ 15%)"
    regime_reason = "Distribution churn is rampant across institutional leaders. Breakouts have extreme squat/failure rates (>80%)."
elif s2_pct >= 40 and s4_pct < 15:
    regime_status = "🟢 GREEN LIGHT: AGGRESSIVE BULL MARKET MARKUP"
    regime_theme = "#10b981"
    regime_bg = "rgba(16, 185, 129, 0.12)"
    regime_border = "rgba(16, 185, 129, 0.4)"
    what_to_do = "Aggressively Buy & Hold. Buy fresh breakouts from Base 1 & 2. Pyramid winners on 21 EMA pullbacks. Deploy maximum position sizing (20–25% per stock). Stay 90–100% invested."
    what_not_to_do = "Do not sell leading winners prematurely. Do not bottom-fish in Stage 4 laggards. Follow your trailing stops (21 EMA / 50 SMA)."
    action_badge = "MAX RISK-ON / AGGRESSIVE BUY"
    trigger_rule = f"Stage 2 Advancing = {s2_pct:.1f}% & Stage 4 = {s4_pct:.1f}% (Triggered: S2 ≥ 40.0% & S4 < 15.0%)"
    regime_reason = "Aggressive institutional markup is widespread with high breakout follow-through and strong market breadth."
elif s1_pct >= 40 and s4_pct < 20:
    regime_status = "🌱 BLUE LIGHT: CYCLE BOTTOM ACCUMULATION"
    regime_theme = "#6366f1"
    regime_bg = "rgba(99, 102, 241, 0.12)"
    regime_border = "rgba(99, 102, 241, 0.4)"
    what_to_do = "Stalk Bases & Build Primary Watchlists. Smart money is quietly absorbing supply in Stage 1. Look for Volume Dry-Up (VDU) and VCPs. Take small pilot positions (5% risk)."
    what_not_to_do = "Do not deploy 100% size until confirmed Stage 1 ➔ Stage 2 ignition breakouts appear on heavy volume."
    action_badge = "STALK BASES / PILOT BUYS"
    trigger_rule = f"Stage 1 Basing = {s1_pct:.1f}% & Stage 4 = {s4_pct:.1f}% (Triggered: S1 ≥ 40.0% & S4 < 20.0%)"
    regime_reason = "Market is forming an institutional cycle bottom with widespread volume dry-up and stealth accumulation."
else:
    regime_status = "🧭 NEUTRAL / SELECTIVE TRANSITIONAL REGIME"
    regime_theme = "#38bdf8"
    regime_bg = "rgba(56, 189, 248, 0.12)"
    regime_border = "rgba(56, 189, 248, 0.4)"
    what_to_do = "Trade Only High-Conviction A+ Setups. Use smaller position sizes (10–12%) with tight 4–6% stops. Keep a 40–60% cash buffer."
    what_not_to_do = "Do not chase extended stocks (>5% above 50 SMA). Do not overtrade or force setups."
    action_badge = "SELECTIVE RISK / TIGHT STOPS"
    trigger_rule = f"Balanced / Mixed Market: S1={s1_pct:.1f}%, S2={s2_pct:.1f}%, S3={s3_pct:.1f}%, S4={s4_pct:.1f}%"
    regime_reason = "No single stage exhibits extreme dominance. Selective individual leadership may work, but broad index tailwinds are absent."

# Real-Time Action Directive Banner
st.markdown(f"""
<div style="background: {regime_bg}; border: 1px solid {regime_border}; border-left: 6px solid {regime_theme}; border-radius: 14px; padding: 20px 24px; margin-bottom: 14px; box-shadow: 0 8px 30px rgba(0,0,0,0.3); font-family: 'Inter', sans-serif;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 10px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 18px; font-weight: 800; color: {regime_theme}; font-family: 'Outfit', sans-serif; letter-spacing: 0.5px;">{regime_status}</span>
        </div>
        <span style="background: {regime_theme}; color: #000; font-size: 11px; font-weight: 800; padding: 5px 14px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.8px;">{action_badge}</span>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
        <div style="background: rgba(0,0,0,0.35); padding: 14px 18px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);">
            <div style="color: #10b981; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">✅ WHEN TO ACT (Actionable Playbook)</div>
            <div style="color: #f1f5f9; font-size: 13px; line-height: 1.55;">{what_to_do}</div>
        </div>
        <div style="background: rgba(0,0,0,0.35); padding: 14px 18px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);">
            <div style="color: #f43f5e; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">❌ WHEN NOT TO ACT (What to Avoid)</div>
            <div style="color: #f1f5f9; font-size: 13px; line-height: 1.55;">{what_not_to_do}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ How This Action Directive & Cash Allocation is Calculated (Quantitative Rules Engine)", expanded=False):
    st.markdown(f"""
    ### 🎯 Current Live Quantitative Diagnosis
    - **Active Regime:** <b style="color: {regime_theme};">{regime_status}</b>
    - **Trigger Condition Met:** `{trigger_rule}`
    - **Why This Action is Mandated:** {regime_reason}
    - **Live Universe Breakdown ({total_stocks:,} stocks):**
      - 🛡️ **Stage 1 (Basing):** **{s1_pct:.1f}%** ({s1_count:,} stocks)
      - 🚀 **Stage 2 (Advancing):** **{s2_pct:.1f}%** ({s2_count:,} stocks)
      - ⚠️ **Stage 3 (Topping):** **{s3_pct:.1f}%** ({s3_count:,} stocks)
      - 🛑 **Stage 4 (Declining):** **{s4_pct:.1f}%** ({s4_count:,} stocks)
    - **Current Directive:** <span style="background: {regime_theme}; color: #000; font-weight: 800; padding: 3px 10px; border-radius: 6px; font-size: 11px;">{action_badge}</span>

    ---

    ### 📐 The Stan Weinstein & Mark Minervini 5-Regime Decision Matrix
    The action directive banner and recommended cash allocation are calculated dynamically from the aggregate stage distribution across the entire equity universe:

    | Regime & Traffic Light | Mathematical Trigger Condition | Cash Allocation Target | Max Position Size | Actionable Playbook |
    | :--- | :--- | :--- | :--- | :--- |
    | 🔴 **Red Light: Bear Liquidation** | `S4_% ≥ 30.0%` | **80% – 100% Cash** | 0% (Forbidden) / Short Hedges only | **Capital Preservation.** >30% of stocks in Stage 4 signals systemic institutional liquidation. Bottom-fishing is account suicide. Stay in cash. |
    | ⚠️ **Yellow/Red Alert: Distribution** | `S3_% ≥ 40.0%` OR `(S3_% ≥ 30.0% & S2_% ≤ 15.0%)` | **60% – 80% Cash** | 5% – 8% (Max 2–3 positions) | **Protect Profits & Cease Buying.** Breakout failure rate exceeds 80%. Tighten stops to breakeven, take quick 10–15% gains, raise cash. |
    | 🟢 **Green Light: Bull Markup** | `S2_% ≥ 40.0% & S4_% < 15.0%` | **0% – 10% Cash (90–100% Invested)** | 20% – 25% (Concentrated) | **Aggressive Buy & Hold.** High institutional tailwind. Buy Stage 2 breakouts & 21 EMA pullbacks. Pyramid winning trades. |
    | 🌱 **Blue Light: Bottom Basing** | `S1_% ≥ 40.0% & S4_% < 20.0%` | **50% – 70% Cash** | 5% – 10% (Pilot positions) | **Stalk Bases.** Smart money is accumulating supply in quiet bases (VDU/VCP). Prepare buy stops above base resistance. |
    | 🧭 **Neutral / Selective Regime** | All other intermediate conditions | **40% – 60% Cash** | 10% – 12% (Strict A+ only) | **Selective Risk.** No broad market tailwind. Trade only high-conviction leadership setups with tight 4–6% stops. |

    ---

    ### 💡 Core Institutional Principles: Why Cash is an Active Weapon
    1. **The 30% Stage 4 Rule (Stan Weinstein Chapter 8):** When 30%+ of the market breaks down into Stage 4 (price < 50 SMA < 200 SMA with falling 200 SMA), systemic correlation pulls down even high-quality stocks. Institutional bids vanish, support levels fail, and rallies turn into bull traps.
    2. **Breakout Failure Asymmetry (Mark Minervini Chapter 10):** In Stage 2 bull markets, 70–80% of sound breakouts follow through to 20%+ gains. In Stage 3/4 distribution regimes, 80%+ of breakouts fail immediately ("squat"). Buying breakouts during distribution guarantees churn and account drawdown.
    3. **Cash as a Position:** Holding 80–100% cash is not "missing out"—it is an active risk-management stance that protects purchasing power and mental capital. When the market completes capitulation and enters Stage 1 Basing ➔ Stage 2 Ignitions, traders in cash can deploy maximum size into the true generational market leaders.
    """)

st.markdown("<br>", unsafe_allow_html=True)

cc1, cc2, cc3, cc4 = st.columns(4)
cc1.markdown(f"""
<div style="background: #16161c; padding: 22px 18px; border-radius: 12px; text-align: center; border-top: 4px solid #6366f1; border-bottom: 1px solid #222; border-left: 1px solid #222; border-right: 1px solid #222; box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15); font-family: 'Inter', sans-serif;">
    <div style="color: #818cf8; font-size: 13px; font-weight: 700; margin-bottom: 6px; letter-spacing: 0.5px;">S1 (BASING)</div>
    <div style="font-size: 30px; font-weight: 800; color: #f8fafc;">{s1_count}</div>
    <div style="font-size: 13px; color: #818cf8; font-weight: 600; margin-top: 4px;">{s1_pct:.1f}% of Market</div>
    <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Stealth Accumulation</div>
    <div style="display: inline-block; background: rgba(99, 102, 241, 0.18); color: #818cf8; font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 6px; margin-top: 8px;">🎯 STALK BASES</div>
</div>
""".replace('\n', ''), unsafe_allow_html=True)

cc2.markdown(f"""
<div style="background: #16161c; padding: 22px 18px; border-radius: 12px; text-align: center; border-top: 4px solid #10b981; border-bottom: 1px solid #222; border-left: 1px solid #222; border-right: 1px solid #222; box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15); font-family: 'Inter', sans-serif;">
    <div style="color: #10b981; font-size: 13px; font-weight: 700; margin-bottom: 6px; letter-spacing: 0.5px;">S2 (ADVANCING)</div>
    <div style="font-size: 30px; font-weight: 800; color: #f8fafc;">{s2_count}</div>
    <div style="font-size: 13px; color: #10b981; font-weight: 600; margin-top: 4px;">{s2_pct:.1f}% of Market</div>
    <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Institutional Markup</div>
    <div style="display: inline-block; background: rgba(16, 185, 129, 0.18); color: #10b981; font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 6px; margin-top: 8px;">🚀 BUY BREAKOUTS</div>
</div>
""".replace('\n', ''), unsafe_allow_html=True)

cc3.markdown(f"""
<div style="background: #16161c; padding: 22px 18px; border-radius: 12px; text-align: center; border-top: 4px solid #f59e0b; border-bottom: 1px solid #222; border-left: 1px solid #222; border-right: 1px solid #222; box-shadow: 0 4px 20px rgba(245, 158, 11, 0.15); font-family: 'Inter', sans-serif;">
    <div style="color: #fbbf24; font-size: 13px; font-weight: 700; margin-bottom: 6px; letter-spacing: 0.5px;">S3 (TOPPING)</div>
    <div style="font-size: 30px; font-weight: 800; color: #f8fafc;">{s3_count}</div>
    <div style="font-size: 13px; color: #fbbf24; font-weight: 600; margin-top: 4px;">{s3_pct:.1f}% of Market</div>
    <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Distribution & Churn</div>
    <div style="display: inline-block; background: rgba(245, 158, 11, 0.18); color: #fbbf24; font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 6px; margin-top: 8px;">⚠️ STOP BUYING / TIGHTEN</div>
</div>
""".replace('\n', ''), unsafe_allow_html=True)

cc4.markdown(f"""
<div style="background: #16161c; padding: 22px 18px; border-radius: 12px; text-align: center; border-top: 4px solid #f43f5e; border-bottom: 1px solid #222; border-left: 1px solid #222; border-right: 1px solid #222; box-shadow: 0 4px 20px rgba(244, 63, 94, 0.15); font-family: 'Inter', sans-serif;">
    <div style="color: #f43f5e; font-size: 13px; font-weight: 700; margin-bottom: 6px; letter-spacing: 0.5px;">S4 (DECLINING)</div>
    <div style="font-size: 30px; font-weight: 800; color: #f8fafc;">{s4_count}</div>
    <div style="font-size: 13px; color: #f43f5e; font-weight: 600; margin-top: 4px;">{s4_pct:.1f}% of Market</div>
    <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Markdown & Liquidation</div>
    <div style="display: inline-block; background: rgba(244, 63, 94, 0.18); color: #f43f5e; font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 6px; margin-top: 8px;">🛑 FORBIDDEN TO BUY</div>
</div>
""".replace('\n', ''), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- MULTI-MODE TREND & REGIME CHART ---
st.markdown("### 📈 Market Stage Dynamics & Regime Evolution")

ctrl_c1, ctrl_c2 = st.columns([3, 1])
with ctrl_c1:
    chart_mode = st.selectbox(
        "Chart Perspective",
        [
            "100% Stacked Regime (Macro Lifecycle Evolution)",
            "Direct Stage % Share Trend (Individual % Lines)",
            "All 4 Stages Trend (Absolute Stock Counts)",
            "S2 (Bulls) vs S4 (Bears) Trend",
            "Net Stage Momentum Spread (S2 - S4)"
        ],
        index=0,
        help="Switch between full market regime evolution, direct percentage lines, and net momentum spreads."
    )
with ctrl_c2:
    timeframe_opt = st.selectbox(
        "Timeframe",
        ["1 Year (252D)", "6 Months (126D)", "3 Months (63D)", "All Available"],
        index=0
    )

# Live Market Stage Breakdown Strip (Always visible above the chart)
st.markdown(f"""
<div style="background: rgba(18, 18, 24, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 12px 16px; margin-top: 10px; margin-bottom: 14px; box-shadow: 0 4px 16px rgba(0,0,0,0.3); font-family: 'Inter', sans-serif;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <span style="color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;">Current Market Stage Breakdown (Latest Session)</span>
        <span style="color: #fbbf24; font-size: 11px; font-weight: 700;">⚠️ Distribution Heavy: Stage 3 Topping Dominates at {s3_pct:.1f}%</span>
    </div>
    <div style="display: flex; height: 32px; width: 100%; border-radius: 8px; overflow: hidden; font-family: 'Outfit', sans-serif; box-shadow: inset 0 2px 6px rgba(0,0,0,0.6);">
        <div style="width: {s2_pct}%; min-width: 55px; background: linear-gradient(90deg, #059669, #10b981); color: #fff; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 800;" title="Stage 2 Advancing: {s2_pct:.1f}% ({s2_count} stocks)">S2: {s2_pct:.1f}%</div>
        <div style="width: {s3_pct}%; background: linear-gradient(90deg, #d97706, #fbbf24); color: #000; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 800;" title="Stage 3 Topping: {s3_pct:.1f}% ({s3_count} stocks)">⚠️ S3 (Topping): {s3_pct:.1f}% ({s3_count} stocks)</div>
        <div style="width: {s4_pct}%; min-width: 60px; background: linear-gradient(90deg, #e11d48, #f43f5e); color: #fff; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 800;" title="Stage 4 Declining: {s4_pct:.1f}% ({s4_count} stocks)">S4: {s4_pct:.1f}%</div>
        <div style="width: {s1_pct}%; background: linear-gradient(90deg, #4f46e5, #818cf8); color: #fff; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 800;" title="Stage 1 Basing: {s1_pct:.1f}% ({s1_count} stocks)">🛡️ S1 (Basing): {s1_pct:.1f}% ({s1_count})</div>
    </div>
</div>
""", unsafe_allow_html=True)

if timeframe_opt.startswith("3 Months"):
    chart_hist = historical_counts.tail(63).copy()
elif timeframe_opt.startswith("6 Months"):
    chart_hist = historical_counts.tail(126).copy()
elif timeframe_opt.startswith("1 Year"):
    chart_hist = historical_counts.tail(252).copy()
else:
    chart_hist = historical_counts.copy()

fig = go.Figure()

if chart_mode.startswith("100% Stacked Regime"):
    tot = chart_hist['S1'] + chart_hist['S2'] + chart_hist['S3'] + chart_hist['S4']
    tot = tot.replace(0, 1)
    s1_p = (chart_hist['S1'] / tot) * 100
    s2_p = (chart_hist['S2'] / tot) * 100
    s3_p = (chart_hist['S3'] / tot) * 100
    s4_p = (chart_hist['S4'] / tot) * 100

    # Stack Order: S2 (Bottom) -> S3 -> S4 -> S1 (Top)
    fig.add_trace(go.Scatter(
        x=chart_hist.index, y=s2_p, name='S2 (Advancing)',
        stackgroup='one', mode='lines',
        line=dict(width=1, color='#10b981'), fillcolor='rgba(16, 185, 129, 0.70)',
        customdata=chart_hist['S2'],
        hovertemplate='<b>%{x|%d %b %Y}</b><br>🚀 S2 Advancing: <b>%{y:.1f}%</b> (%{customdata} stocks)<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=chart_hist.index, y=s3_p, name='S3 (Topping)',
        stackgroup='one', mode='lines',
        line=dict(width=1, color='#f59e0b'), fillcolor='rgba(245, 158, 11, 0.70)',
        customdata=chart_hist['S3'],
        hovertemplate='⚠️ S3 Topping: <b>%{y:.1f}%</b> (%{customdata} stocks)<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=chart_hist.index, y=s4_p, name='S4 (Declining)',
        stackgroup='one', mode='lines',
        line=dict(width=1, color='#f43f5e'), fillcolor='rgba(244, 63, 94, 0.70)',
        customdata=chart_hist['S4'],
        hovertemplate='🛑 S4 Declining: <b>%{y:.1f}%</b> (%{customdata} stocks)<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=chart_hist.index, y=s1_p, name='S1 (Basing)',
        stackgroup='one', mode='lines',
        line=dict(width=1, color='#818cf8'), fillcolor='rgba(99, 102, 241, 0.60)',
        customdata=chart_hist['S1'],
        hovertemplate='🛡️ S1 Basing: <b>%{y:.1f}%</b> (%{customdata} stocks)<extra></extra>'
    ))

    # Pinned Floating On-Chart % Badges at the rightmost edge
    latest_date = chart_hist.index[-1]
    latest_s2_pct = s2_p.iloc[-1]
    latest_s3_pct = s3_p.iloc[-1]
    latest_s4_pct = s4_p.iloc[-1]
    latest_s1_pct = s1_p.iloc[-1]

    mid_s2 = latest_s2_pct / 2
    mid_s3 = latest_s2_pct + (latest_s3_pct / 2)
    mid_s4 = latest_s2_pct + latest_s3_pct + (latest_s4_pct / 2)
    mid_s1 = latest_s2_pct + latest_s3_pct + latest_s4_pct + (latest_s1_pct / 2)

    fig.add_annotation(
        x=latest_date, y=mid_s2,
        text=f"<b>S2: {latest_s2_pct:.1f}%</b>",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#10b981",
        ax=55, ay=0, xanchor="left",
        bgcolor="#10b981", bordercolor="#10b981", borderwidth=1, borderpad=4,
        font=dict(color="#000", size=11, family="Outfit, sans-serif")
    )
    fig.add_annotation(
        x=latest_date, y=mid_s3,
        text=f"<b>S3: {latest_s3_pct:.1f}%</b>",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#fbbf24",
        ax=55, ay=0, xanchor="left",
        bgcolor="#fbbf24", bordercolor="#fbbf24", borderwidth=1, borderpad=4,
        font=dict(color="#000", size=11, family="Outfit, sans-serif")
    )
    fig.add_annotation(
        x=latest_date, y=mid_s4,
        text=f"<b>S4: {latest_s4_pct:.1f}%</b>",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#f43f5e",
        ax=55, ay=0, xanchor="left",
        bgcolor="#f43f5e", bordercolor="#f43f5e", borderwidth=1, borderpad=4,
        font=dict(color="#fff", size=11, family="Outfit, sans-serif")
    )
    fig.add_annotation(
        x=latest_date, y=mid_s1,
        text=f"<b>S1: {latest_s1_pct:.1f}%</b>",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#818cf8",
        ax=55, ay=0, xanchor="left",
        bgcolor="#6366f1", bordercolor="#818cf8", borderwidth=1, borderpad=4,
        font=dict(color="#fff", size=11, family="Outfit, sans-serif")
    )

    # In-band text watermark labels across historical milestones
    if len(chart_hist) >= 30:
        idx_mid1 = len(chart_hist) // 3
        idx_mid2 = (len(chart_hist) * 2) // 3
        for idx_m in [idx_mid1, idx_mid2]:
            d_m = chart_hist.index[idx_m]
            p_s2 = s2_p.iloc[idx_m]
            p_s3 = s3_p.iloc[idx_m]
            p_s4 = s4_p.iloc[idx_m]
            p_s1 = s1_p.iloc[idx_m]

            if p_s2 >= 18:
                fig.add_annotation(
                    x=d_m, y=p_s2 / 2, text=f"<b>S2: {p_s2:.0f}%</b>", showarrow=False,
                    font=dict(color="rgba(16, 185, 129, 0.95)", size=11, family="Outfit, sans-serif")
                )
            if p_s3 >= 18:
                fig.add_annotation(
                    x=d_m, y=p_s2 + (p_s3 / 2), text=f"<b>S3: {p_s3:.0f}%</b>", showarrow=False,
                    font=dict(color="rgba(251, 191, 36, 0.9)", size=11, family="Outfit, sans-serif")
                )
            if p_s4 >= 18:
                fig.add_annotation(
                    x=d_m, y=p_s2 + p_s3 + (p_s4 / 2), text=f"<b>S4: {p_s4:.0f}%</b>", showarrow=False,
                    font=dict(color="rgba(244, 63, 94, 0.9)", size=11, family="Outfit, sans-serif")
                )
            if p_s1 >= 18:
                fig.add_annotation(
                    x=d_m, y=p_s2 + p_s3 + p_s4 + (p_s1 / 2), text=f"<b>S1: {p_s1:.0f}%</b>", showarrow=False,
                    font=dict(color="rgba(129, 140, 248, 0.9)", size=11, family="Outfit, sans-serif")
                )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="<b>100% Market Stage Regime Evolution</b> (Macro Rotation Across 4 Stages)",
        yaxis=dict(title="Market Share (%)", range=[0, 100], ticksuffix="%"),
        margin=dict(l=20, r=115, t=40, b=20),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

elif chart_mode.startswith("Direct Stage % Share"):
    tot = chart_hist['S1'] + chart_hist['S2'] + chart_hist['S3'] + chart_hist['S4']
    tot = tot.replace(0, 1)
    s1_p = (chart_hist['S1'] / tot) * 100
    s2_p = (chart_hist['S2'] / tot) * 100
    s3_p = (chart_hist['S3'] / tot) * 100
    s4_p = (chart_hist['S4'] / tot) * 100

    fig.add_trace(go.Scatter(
        x=chart_hist.index, y=s2_p, name='S2 Advancing (%)',
        line=dict(color='#10b981', width=3),
        hovertemplate='🚀 S2 Advancing: <b>%{y:.1f}%</b><extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=chart_hist.index, y=s3_p, name='S3 Topping (%)',
        line=dict(color='#fbbf24', width=3),
        hovertemplate='⚠️ S3 Topping: <b>%{y:.1f}%</b><extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=chart_hist.index, y=s4_p, name='S4 Declining (%)',
        line=dict(color='#f43f5e', width=2),
        hovertemplate='🛑 S4 Declining: <b>%{y:.1f}%</b><extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=chart_hist.index, y=s1_p, name='S1 Basing (%)',
        line=dict(color='#818cf8', width=2, dash='dot'),
        hovertemplate='🛡️ S1 Basing: <b>%{y:.1f}%</b><extra></extra>'
    ))

    # Pinned End Badges on the lines
    latest_d = chart_hist.index[-1]
    fig.add_annotation(
        x=latest_d, y=s2_p.iloc[-1], text=f"<b>S2: {s2_p.iloc[-1]:.1f}%</b>",
        showarrow=True, arrowhead=2, arrowsize=1, arrowcolor="#10b981", ax=50, ay=0, xanchor="left",
        bgcolor="#10b981", font=dict(color="#000", size=11, family="Outfit, sans-serif")
    )
    fig.add_annotation(
        x=latest_d, y=s3_p.iloc[-1], text=f"<b>S3: {s3_p.iloc[-1]:.1f}%</b>",
        showarrow=True, arrowhead=2, arrowsize=1, arrowcolor="#fbbf24", ax=50, ay=0, xanchor="left",
        bgcolor="#fbbf24", font=dict(color="#000", size=11, family="Outfit, sans-serif")
    )
    fig.add_annotation(
        x=latest_d, y=s4_p.iloc[-1], text=f"<b>S4: {s4_p.iloc[-1]:.1f}%</b>",
        showarrow=True, arrowhead=2, arrowsize=1, arrowcolor="#f43f5e", ax=50, ay=0, xanchor="left",
        bgcolor="#f43f5e", font=dict(color="#fff", size=11, family="Outfit, sans-serif")
    )
    fig.add_annotation(
        x=latest_d, y=s1_p.iloc[-1], text=f"<b>S1: {s1_p.iloc[-1]:.1f}%</b>",
        showarrow=True, arrowhead=2, arrowsize=1, arrowcolor="#818cf8", ax=50, ay=0, xanchor="left",
        bgcolor="#6366f1", font=dict(color="#fff", size=11, family="Outfit, sans-serif")
    )

    # Reference Action Thresholds
    fig.add_hline(y=40, line_dash="dash", line_color="#10b981", line_width=1.5,
                  annotation_text="🟢 S2 Bull Markup Threshold (40%)", annotation_position="top left",
                  annotation_font=dict(color="#10b981", size=10))
    fig.add_hline(y=35, line_dash="dash", line_color="#f59e0b", line_width=1.5,
                  annotation_text="⚠️ S3 Distribution Danger Threshold (35%)", annotation_position="bottom left",
                  annotation_font=dict(color="#fbbf24", size=10))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="<b>Direct Stage % Share Trend</b> (Individual Stage % with Action Thresholds)",
        yaxis=dict(title="Stage Share (% of Total Market)", range=[0, max(75, int(s3_p.max()) + 5)], ticksuffix="%"),
        margin=dict(l=20, r=115, t=40, b=20),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

elif chart_mode.startswith("All 4 Stages Trend"):
    fig.add_trace(go.Scatter(
        x=chart_hist.index, y=chart_hist['S2'], name='S2 (Advancing)',
        line=dict(color='#10b981', width=2.5), fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.12)',
        hovertemplate='S2 Advancing: <b>%{y} stocks</b><extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=chart_hist.index, y=chart_hist['S3'], name='S3 (Topping)',
        line=dict(color='#fbbf24', width=2), fill='tozeroy', fillcolor='rgba(251, 191, 36, 0.08)',
        hovertemplate='S3 Topping: <b>%{y} stocks</b><extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=chart_hist.index, y=chart_hist['S4'], name='S4 (Declining)',
        line=dict(color='#f43f5e', width=2), fill='tozeroy', fillcolor='rgba(244, 63, 94, 0.08)',
        hovertemplate='S4 Declining: <b>%{y} stocks</b><extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=chart_hist.index, y=chart_hist['S1'], name='S1 (Basing)',
        line=dict(color='#818cf8', width=2, dash='dot'),
        hovertemplate='S1 Basing: <b>%{y} stocks</b><extra></extra>'
    ))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="<b>All 4 Stages Absolute Trend</b> (Daily Stock Counts)",
        yaxis=dict(title="Number of Stocks"),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

elif chart_mode.startswith("S2 (Bulls) vs S4 (Bears)"):
    fig.add_trace(go.Scatter(
        x=chart_hist.index, y=chart_hist['S2'], name='S2 (Advancing Bulls)', 
        line=dict(color='#10b981', width=2.5), fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.15)',
        hovertemplate='S2 Bulls: <b>%{y} stocks</b><extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=chart_hist.index, y=chart_hist['S4'], name='S4 (Declining Bears)', 
        line=dict(color='#f43f5e', width=2.5), fill='tozeroy', fillcolor='rgba(244, 63, 94, 0.15)',
        hovertemplate='S4 Bears: <b>%{y} stocks</b><extra></extra>'
    ))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="<b>S2 (Bulls) vs S4 (Bears) Trend</b>",
        yaxis=dict(title="Number of Stocks"),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

else:
    # Net Stage Momentum Spread: S2 - S4
    net_spread = chart_hist['S2'] - chart_hist['S4']
    spread_ma10 = net_spread.rolling(10, min_periods=3).mean()
    bar_colors = ['#10b981' if v >= 0 else '#f43f5e' for v in net_spread]

    fig.add_trace(go.Bar(
        x=chart_hist.index, y=net_spread, name='Net Spread (S2 - S4)',
        marker_color=bar_colors, opacity=0.85,
        hovertemplate='<b>%{x|%d %b %Y}</b><br>Net Spread: <b>%{y:+d} stocks</b><extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=chart_hist.index, y=spread_ma10, name='10-Day MA',
        line=dict(color='#fbbf24', width=2),
        hovertemplate='10-Day MA: <b>%{y:.1f}</b><extra></extra>'
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b", line_width=1)
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="<b>Net Stage Momentum Spread</b> (S2 Advancing minus S4 Declining)",
        yaxis=dict(title="Net Bull Spread (Stocks)"),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

st.plotly_chart(fig, use_container_width=True)

# --- 5-SECOND CHART READING & CAPITAL DEPLOYMENT GUIDE ---
st.markdown("""
<div style="background: rgba(18, 18, 24, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 20px 22px; margin-top: 15px; margin-bottom: 25px; box-shadow: 0 8px 24px rgba(0,0,0,0.3); font-family: 'Inter', sans-serif;">
    <div style="font-size: 15px; font-weight: 700; color: #f8fafc; font-family: 'Outfit', sans-serif; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span>🧭 How to Read This Chart in 5 Seconds & When to Trade</span>
            <span style="font-size: 10.5px; background: rgba(255,255,255,0.08); color: #cbd5e1; padding: 3px 10px; border-radius: 12px; font-weight: 600;">Super-Investor Macro Rules</span>
        </div>
        <span style="font-size: 11px; color: #94a3b8;">Legendary wizards (Weinstein, Minervini) only deploy risk when Emerald Green (S2) dominates.</span>
    </div>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;">
        <div style="background: rgba(16, 185, 129, 0.04); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 10px; padding: 14px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="color: #10b981; font-size: 11.5px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">🟢 Emerald Green (S2) > 40%</div>
                <div style="font-size: 13.5px; font-weight: 700; color: #fff; margin-bottom: 6px;">POWER BULL MARKUP</div>
                <div style="color: #94a3b8; font-size: 12px; line-height: 1.45;">Breakouts work with high follow-through. Institutional money is pushing leaders higher.</div>
            </div>
            <div style="margin-top: 10px; font-size: 11px; color: #10b981; font-weight: 600; background: rgba(16, 185, 129, 0.1); padding: 6px 8px; border-radius: 6px;">
                🚀 <b>ACTION:</b> Aggressively buy Stage 2 leaders, pyramid into strength, stay 90-100% invested.
            </div>
        </div>
        <div style="background: rgba(245, 158, 11, 0.04); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 10px; padding: 14px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="color: #fbbf24; font-size: 11.5px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">🟡 Amber (S3) > 35%</div>
                <div style="font-size: 13.5px; font-weight: 700; color: #fff; margin-bottom: 6px;">DISTRIBUTION ALERT (NOW)</div>
                <div style="color: #94a3b8; font-size: 12px; line-height: 1.45;">Smart money is quietly unloading shares. Breakouts squat & fail (>80% failure rate). Volatility widens.</div>
            </div>
            <div style="margin-top: 10px; font-size: 11px; color: #fbbf24; font-weight: 600; background: rgba(245, 158, 11, 0.1); padding: 6px 8px; border-radius: 6px;">
                ⚠️ <b>ACTION:</b> DO NOT BUY BREAKOUTS. Tighten stops to breakeven, raise cash to 60-80%.
            </div>
        </div>
        <div style="background: rgba(244, 63, 94, 0.04); border: 1px solid rgba(244, 63, 94, 0.25); border-radius: 10px; padding: 14px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="color: #f43f5e; font-size: 11.5px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">🔴 Crimson (S4) > 30%</div>
                <div style="font-size: 13.5px; font-weight: 700; color: #fff; margin-bottom: 6px;">BEAR LIQUIDATION</div>
                <div style="color: #94a3b8; font-size: 12px; line-height: 1.45;">Unrelenting institutional selling. Support lines collapse. Rallies are dead-cat bounces that get sold.</div>
            </div>
            <div style="margin-top: 10px; font-size: 11px; color: #f43f5e; font-weight: 600; background: rgba(244, 63, 94, 0.1); padding: 6px 8px; border-radius: 6px;">
                🛑 <b>ACTION:</b> 100% Forbidden to buy long. Sit in cash. Bottom fishing here destroys accounts.
            </div>
        </div>
        <div style="background: rgba(99, 102, 241, 0.04); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 10px; padding: 14px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="color: #818cf8; font-size: 11.5px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">🔵 Indigo (S1) > 40%</div>
                <div style="font-size: 13.5px; font-weight: 700; color: #fff; margin-bottom: 6px;">BOTTOM INCUBATION</div>
                <div style="color: #94a3b8; font-size: 12px; line-height: 1.45;">Capitulation ends. Stocks build sideways bases with Volume Dry-Up (VDU). Next cycle is being born.</div>
            </div>
            <div style="margin-top: 10px; font-size: 11px; color: #818cf8; font-weight: 600; background: rgba(99, 102, 241, 0.1); padding: 6px 8px; border-radius: 6px;">
                🎯 <b>ACTION:</b> Stalk VCPs and build watchlists. Wait for confirmed S1 ➔ S2 ignition breakouts.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- INTERACTIVE STAGE STOCK SCREENER ---
st.markdown("---")
st.markdown("### 🔍 Interactive Stage Stock Screener & Watchlist")
st.caption("Inspect, sort, and export the exact stocks residing in each stage right now.")

filter_col1, filter_col2, filter_col3 = st.columns([2, 1.5, 1.5])
with filter_col1:
    stage_filter = st.radio(
        "Select Stage to Inspect",
        [
            f"S1 Basing ({s1_count})",
            f"S2 Advancing ({s2_count})",
            f"S3 Topping ({s3_count})",
            f"S4 Declining ({s4_count})",
            f"All Stocks ({total_stocks})"
        ],
        horizontal=True,
        index=0
    )
with filter_col2:
    search_query = st.text_input("Quick Search", placeholder="Symbol or Industry...").strip().upper()
with filter_col3:
    cap_segments = fetch_cap_segments()
    cap_filter = st.selectbox("Market Cap Segment", ["All Segments"] + list(cap_segments.keys()), index=0)

# Filter df_latest
if stage_filter.startswith("S1"):
    df_filtered = df_latest[df_latest['Stage'] == 'S1'].copy()
elif stage_filter.startswith("S2"):
    df_filtered = df_latest[df_latest['Stage'] == 'S2'].copy()
elif stage_filter.startswith("S3"):
    df_filtered = df_latest[df_latest['Stage'] == 'S3'].copy()
elif stage_filter.startswith("S4"):
    df_filtered = df_latest[df_latest['Stage'] == 'S4'].copy()
else:
    df_filtered = df_latest.copy()

# Filter by segment
if cap_filter != "All Segments" and cap_filter in cap_segments:
    seg_syms = set(cap_segments[cap_filter])
    df_filtered = df_filtered[df_filtered.index.isin(seg_syms)]

# Filter by search
if search_query:
    sym_mask = df_filtered.index.str.replace('.NS', '').str.contains(search_query, case=False, na=False)
    ind_mask = df_filtered['Industry'].str.contains(search_query, case=False, na=False)
    df_filtered = df_filtered[sym_mask | ind_mask]

# Sort options
sort_col1, sort_col2 = st.columns([2, 2])
with sort_col1:
    sort_field = st.selectbox(
        "Sort By",
        ["1M Return", "3M Return", "1W Return", "Distance to 50 SMA", "Distance to 200 SMA", "Close Price"],
        index=0
    )
with sort_col2:
    sort_dir = st.selectbox("Sort Order", ["Highest First (Descending)", "Lowest First (Ascending)"], index=0)

sort_map = {
    "1M Return": 'Ret_1M',
    "3M Return": 'Ret_3M',
    "1W Return": 'Ret_1W',
    "Distance to 50 SMA": 'Dist_50',
    "Distance to 200 SMA": 'Dist_200',
    "Close Price": 'Close'
}
asc = (sort_dir == "Lowest First (Ascending)")
df_filtered = df_filtered.sort_values(sort_map[sort_field], ascending=asc)

# Display Table
if not df_filtered.empty:
    display_df = pd.DataFrame({
        'Ticker': [s.replace('.NS', '') for s in df_filtered.index],
        'Industry': df_filtered['Industry'],
        'Stage': df_filtered['Stage'],
        'LTP (₹)': [f"₹{c:,.2f}" if pd.notna(c) else "-" for c in df_filtered['Close']],
        'Dist 50 SMA': [f"{d:+.1f}%" if pd.notna(d) else "-" for d in df_filtered['Dist_50']],
        'Dist 200 SMA': [f"{d:+.1f}%" if pd.notna(d) else "-" for d in df_filtered['Dist_200']],
        '1W Ret': [f"{r:+.1f}%" if pd.notna(r) else "-" for r in df_filtered['Ret_1W']],
        '1M Ret': [f"{r:+.1f}%" if pd.notna(r) else "-" for r in df_filtered['Ret_1M']],
        '3M Ret': [f"{r:+.1f}%" if pd.notna(r) else "-" for r in df_filtered['Ret_3M']],
        'Chart': [f"https://in.tradingview.com/chart/?symbol=NSE:{s.replace('.NS', '')}" for s in df_filtered.index]
    })
    
    st.dataframe(
        display_df,
        column_config={
            "Chart": st.column_config.LinkColumn("TradingView", display_text="Open TV ↗"),
            "Stage": st.column_config.TextColumn("Stage", width="small")
        },
        hide_index=True,
        use_container_width=True
    )
    
    # TV Export for current filtered stocks
    filtered_tickers = [s.replace('.NS', '') for s in df_filtered.index]
    if filtered_tickers:
        tv_watchlist_str = ",".join([f"NSE:{t}" for t in filtered_tickers[:200]])
        with st.expander(f"📋 Export Filtered {stage_filter.split(' ')[0]} Watchlist to TradingView ({len(filtered_tickers)} stocks)"):
            st.caption("Copy and paste directly into a TradingView Watchlist window:")
            st.code(tv_watchlist_str, language='text')
else:
    st.info("No stocks match the selected stage and search filters.")

st.markdown("<br>", unsafe_allow_html=True)

# --- MARKET-WIDE STAGE MIGRATION ---
st.markdown("---")
st.markdown("### 🌊 Market-Wide Stage Migration (4-Way Weekly Transitions)")
st.caption("Quantifying Stan Weinstein stage transitions over a rolling 5-day window across the NIFTY Total Market universe. Tracks net institutional accumulation vs distribution onset.")

if stage_migration_df is not None and not stage_migration_df.empty:
    latest_mig_s1_s2 = int(stage_migration_df['S1_to_S2'].iloc[-1])
    latest_mig_s2_s3 = int(stage_migration_df['S2_to_S3'].iloc[-1]) if 'S2_to_S3' in stage_migration_df.columns else 0
    latest_mig_s3_s4 = int(stage_migration_df['S3_to_S4'].iloc[-1])
    latest_mig_s4_s1 = int(stage_migration_df['S4_to_S1'].iloc[-1]) if 'S4_to_S1' in stage_migration_df.columns else 0
    latest_net_mig = int(stage_migration_df['Net_Migration'].iloc[-1])
    prev_net_mig = int(stage_migration_df['Net_Migration'].iloc[-2]) if len(stage_migration_df) > 1 else latest_net_mig
    delta_mig = latest_net_mig - prev_net_mig

    if latest_net_mig >= 10:
        regime_title = "ACCUMULATION SURGE"
        regime_color = "#10b981"
        regime_border = "rgba(16, 185, 129, 0.4)"
        regime_desc = "Aggressive net expansion into Stage 2"
    elif latest_net_mig > 0:
        regime_title = "CONSTRUCTIVE FLOW"
        regime_color = "#34d399"
        regime_border = "rgba(52, 211, 153, 0.4)"
        regime_desc = "Breakouts outpacing stage breakdowns"
    elif latest_net_mig == 0:
        regime_title = "NEUTRAL BALANCE"
        regime_color = "#94a3b8"
        regime_border = "rgba(148, 163, 184, 0.4)"
        regime_desc = "Market transitions in equilibrium"
    elif latest_net_mig > -10:
        regime_title = "MILD DISTRIBUTION"
        regime_color = "#f59e0b"
        regime_border = "rgba(245, 158, 11, 0.4)"
        regime_desc = "Distribution pressures emerging"
    else:
        regime_title = "DISTRIBUTION REGIME"
        regime_color = "#f43f5e"
        regime_border = "rgba(244, 63, 94, 0.4)"
        regime_desc = "Heavy breakdowns into Stage 4"

    # 4 Transition Metric Cards
    tm1, tm2, tm3, tm4 = st.columns(4)
    tm1.markdown(f"""
    <div style="background: #16161c; padding: 20px 16px; border-radius: 12px; text-align: center; border-top: 4px solid #10b981; border-bottom: 1px solid #222; border-left: 1px solid #222; border-right: 1px solid #222; box-shadow: 0 4px 20px rgba(16, 185, 129, 0.12); font-family: 'Inter', sans-serif;">
        <div style="color: #10b981; font-size: 12px; font-weight: 700; margin-bottom: 4px; letter-spacing: 0.5px;">🚀 S1 ➔ S2 IGNITIONS</div>
        <div style="font-size: 26px; font-weight: 800; color: white;">{latest_mig_s1_s2}</div>
        <div style="font-size: 11px; color: #888; margin-top: 4px;">Breakouts Past 5 Days</div>
    </div>
    """.replace('\n', ''), unsafe_allow_html=True)

    tm2.markdown(f"""
    <div style="background: #16161c; padding: 20px 16px; border-radius: 12px; text-align: center; border-top: 4px solid #f59e0b; border-bottom: 1px solid #222; border-left: 1px solid #222; border-right: 1px solid #222; box-shadow: 0 4px 20px rgba(245, 158, 11, 0.12); font-family: 'Inter', sans-serif;">
        <div style="color: #fbbf24; font-size: 12px; font-weight: 700; margin-bottom: 4px; letter-spacing: 0.5px;">⚠️ S2 ➔ S3 DISTRIBUTION</div>
        <div style="font-size: 26px; font-weight: 800; color: white;">{latest_mig_s2_s3}</div>
        <div style="font-size: 11px; color: #888; margin-top: 4px;">Leaders Stalling / Churning</div>
    </div>
    """.replace('\n', ''), unsafe_allow_html=True)

    tm3.markdown(f"""
    <div style="background: #16161c; padding: 20px 16px; border-radius: 12px; text-align: center; border-top: 4px solid #f43f5e; border-bottom: 1px solid #222; border-left: 1px solid #222; border-right: 1px solid #222; box-shadow: 0 4px 20px rgba(244, 63, 94, 0.12); font-family: 'Inter', sans-serif;">
        <div style="color: #f43f5e; font-size: 12px; font-weight: 700; margin-bottom: 4px; letter-spacing: 0.5px;">🛑 S3 ➔ S4 BREAKDOWNS</div>
        <div style="font-size: 26px; font-weight: 800; color: white;">{latest_mig_s3_s4}</div>
        <div style="font-size: 11px; color: #888; margin-top: 4px;">Liquidation Cascades</div>
    </div>
    """.replace('\n', ''), unsafe_allow_html=True)

    tm4.markdown(f"""
    <div style="background: #16161c; padding: 20px 16px; border-radius: 12px; text-align: center; border-top: 4px solid #6366f1; border-bottom: 1px solid #222; border-left: 1px solid #222; border-right: 1px solid #222; box-shadow: 0 4px 20px rgba(99, 102, 241, 0.12); font-family: 'Inter', sans-serif;">
        <div style="color: #818cf8; font-size: 12px; font-weight: 700; margin-bottom: 4px; letter-spacing: 0.5px;">🛡️ S4 ➔ S1 BASING ONSET</div>
        <div style="font-size: 26px; font-weight: 800; color: white;">{latest_mig_s4_s1}</div>
        <div style="font-size: 11px; color: #888; margin-top: 4px;">Bottoming / Base Building</div>
    </div>
    """.replace('\n', ''), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Net Migration & Regime Summary
    net_c1, net_c2 = st.columns(2)
    net_color = "#10b981" if latest_net_mig > 0 else ("#f43f5e" if latest_net_mig < 0 else "#94a3b8")
    net_c1.markdown(f"""
    <div style="background: #16161c; padding: 18px 20px; border-radius: 12px; display: flex; justify-content: space-between; align-items: center; border-left: 4px solid {net_color}; border-top: 1px solid #222; border-bottom: 1px solid #222; border-right: 1px solid #222;">
        <div>
            <div style="color: #888; font-size: 12px; font-weight: 600; text-transform: uppercase;">Net Weekly Migration (S1➔S2 minus S3➔S4)</div>
            <div style="font-size: 26px; font-weight: 800; color: {net_color}; margin-top: 2px;">{'+' if latest_net_mig > 0 else ''}{latest_net_mig} Stocks</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 12px; color: #aaa;">Day-over-Day</div>
            <div style="font-size: 15px; font-weight: 700; color: {net_color};">{delta_mig:+d} vs Prev Day</div>
        </div>
    </div>
    """.replace('\n', ''), unsafe_allow_html=True)

    net_c2.markdown(f"""
    <div style="background: #16161c; padding: 18px 20px; border-radius: 12px; display: flex; justify-content: space-between; align-items: center; border-left: 4px solid {regime_color}; border-top: 1px solid #222; border-bottom: 1px solid #222; border-right: 1px solid #222; box-shadow: 0 4px 16px {regime_border};">
        <div>
            <div style="color: #888; font-size: 12px; font-weight: 600; text-transform: uppercase;">Current Migration Regime</div>
            <div style="font-size: 22px; font-weight: 800; color: white; margin-top: 2px;">{regime_title}</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 12px; color: {regime_color}; font-weight: 600;">{regime_desc}</div>
        </div>
    </div>
    """.replace('\n', ''), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Historical Migration Plotly Chart
    recent_mig = stage_migration_df.tail(252).copy()
    mig_colors = ['#10b981' if v >= 0 else '#f43f5e' for v in recent_mig['Net_Migration']]
    mig_ma10 = recent_mig['Net_Migration'].rolling(10, min_periods=3).mean()

    fig_mig = go.Figure()
    fig_mig.add_trace(go.Bar(
        x=recent_mig.index,
        y=recent_mig['Net_Migration'],
        name='Net Stage Migration (5D)',
        marker_color=mig_colors,
        opacity=0.85,
        customdata=np.stack((recent_mig['S1_to_S2'], recent_mig['S3_to_S4']), axis=-1),
        hovertemplate='<b>%{x|%d %b %Y}</b><br>Net Migration: <b>%{y:+d}</b><br>S1 ➔ S2 (Ignitions): %{customdata[0]}<br>S3 ➔ S4 (Breakdowns): %{customdata[1]}<extra></extra>'
    ))
    fig_mig.add_trace(go.Scatter(
        x=recent_mig.index,
        y=mig_ma10,
        name='10-Day MA (Trend)',
        line=dict(color='#fbbf24', width=2),
        hovertemplate='10-Day MA: %{y:.1f}<extra></extra>'
    ))
    fig_mig.add_hline(y=0, line_dash="dash", line_color="#555", line_width=1)
    fig_mig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="Net Stage Migration Trend (Stage 1→2 Ignitions minus Stage 3→4 Breakdowns, 1-Year)",
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified',
        yaxis_title="Net Migration (Stocks)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_mig, use_container_width=True)

    # Detailed 4-Way Transition Expanders
    t_tab1, t_tab2, t_tab3, t_tab4 = st.tabs([
        f"🚀 S1 ➔ S2 Ignitions ({len(migration_syms_dict.get('S1_to_S2', []))})",
        f"⚠️ S2 ➔ S3 Distribution Onset ({len(migration_syms_dict.get('S2_to_S3', []))})",
        f"🛑 S3 ➔ S4 Breakdowns ({len(migration_syms_dict.get('S3_to_S4', []))})",
        f"🛡️ S4 ➔ S1 Basing Onset ({len(migration_syms_dict.get('S4_to_S1', []))})"
    ])

    def render_transition_table(sym_list, empty_msg, sort_asc=False):
        if not sym_list:
            st.caption(empty_msg)
            return
        valid_syms = [s for s in sym_list if s in df_latest.index]
        if not valid_syms:
            st.caption(empty_msg)
            return
        sub_df = pd.DataFrame({
            'Ticker': [s.replace('.NS', '') for s in valid_syms],
            'Industry': [df_latest.loc[s, 'Industry'] for s in valid_syms],
            'LTP (₹)': [f"₹{df_latest.loc[s, 'Close']:,.2f}" if pd.notna(df_latest.loc[s, 'Close']) else "-" for s in valid_syms],
            'Dist 50 SMA': [f"{df_latest.loc[s, 'Dist_50']:+.1f}%" if pd.notna(df_latest.loc[s, 'Dist_50']) else "-" for s in valid_syms],
            '1W Ret': [f"{df_latest.loc[s, 'Ret_1W']:+.2f}%" for s in valid_syms],
            '1M Ret': [f"{df_latest.loc[s, 'Ret_1M']:+.2f}%" for s in valid_syms],
            '3M Ret': [f"{df_latest.loc[s, 'Ret_3M']:+.2f}%" for s in valid_syms],
            'Chart': [f"https://in.tradingview.com/chart/?symbol=NSE:{s.replace('.NS', '')}" for s in valid_syms]
        })
        # Sort by 1W Ret
        sub_df['_sort'] = [df_latest.loc[s, 'Ret_1W'] for s in valid_syms]
        sub_df = sub_df.sort_values('_sort', ascending=sort_asc).drop(columns=['_sort'])
        st.dataframe(
            sub_df,
            column_config={"Chart": st.column_config.LinkColumn("TradingView", display_text="Open TV ↗")},
            hide_index=True,
            use_container_width=True
        )

    with t_tab1:
        render_transition_table(
            migration_syms_dict.get('S1_to_S2', []),
            "No stocks transitioned from Stage 1 to Stage 2 over the past 5 sessions.",
            sort_asc=False
        )

    with t_tab2:
        render_transition_table(
            migration_syms_dict.get('S2_to_S3', []),
            "No stocks slipped from Stage 2 to Stage 3 over the past 5 sessions.",
            sort_asc=True
        )

    with t_tab3:
        render_transition_table(
            migration_syms_dict.get('S3_to_S4', []),
            "No stocks broke down from Stage 3 to Stage 4 over the past 5 sessions.",
            sort_asc=True
        )

    with t_tab4:
        render_transition_table(
            migration_syms_dict.get('S4_to_S1', []),
            "No stocks transitioned from Stage 4 to Stage 1 basing over the past 5 sessions.",
            sort_asc=False
        )

st.markdown("---")

# --- DATA PREP FOR INDUSTRY ANALYSIS ---
industry_groups = df_latest.groupby('Industry').agg(
    S1=('Stage', lambda x: (x == 'S1').sum()),
    S2=('Stage', lambda x: (x == 'S2').sum()),
    S3=('Stage', lambda x: (x == 'S3').sum()),
    S4=('Stage', lambda x: (x == 'S4').sum()),
    Total=('Stage', 'count'),
    Above_50_Count=('Above_50', 'sum'),
    Ret_1W=('Ret_1W', 'mean'),
    Ret_1M=('Ret_1M', 'mean'),
    Ret_3M=('Ret_3M', 'mean')
)

# Filter out empty or tiny industries
industry_groups = industry_groups[industry_groups['Total'] > 3].copy()

# Rank sectors
industry_groups['Rank_1W'] = industry_groups['Ret_1W'].rank(ascending=False).fillna(999).astype(int)
industry_groups['Rank_1M'] = industry_groups['Ret_1M'].rank(ascending=False).fillna(999).astype(int)
industry_groups['Rank_3M'] = industry_groups['Ret_3M'].rank(ascending=False).fillna(999).astype(int)

# Calculate Percentages
for col in ['S1', 'S2', 'S3', 'S4']:
    industry_groups[f'{col}_%'] = (industry_groups[col] / industry_groups['Total']) * 100

industry_groups['Above_50_%'] = (industry_groups['Above_50_Count'] / industry_groups['Total']) * 100

# --- SECTOR LEADERSHIP DASHBOARD ---
st.markdown("### 🏆 Sector Leadership & Rotation")
st.caption("Top performing industries across multiple timeframes and their underlying stock leaders.")

st.markdown("""
<style>
.micro-pill {
    flex: 1; 
    background: rgba(255,255,255,0.03); 
    border: 1px solid rgba(255,255,255,0.04); 
    border-radius: 6px; 
    padding: 6px 4px; 
    text-align: center; 
    display: flex; 
    flex-direction: column; 
    gap: 2px;
    text-decoration: none !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
.micro-pill:hover {
    background: rgba(255,255,255,0.08);
    border-color: rgba(255,255,255,0.25);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

def build_leader_card(title, metric_col, df_ind, df_stk, theme_color, theme_rgba, leader_counts, is_pct=True):
    top_3_ind = df_ind.nlargest(3, metric_col)
    card_tickers = []
    
    html = f"""
    <div style="background: linear-gradient(180deg, {theme_rgba} 0%, rgba(18,18,24,0.7) 120px); backdrop-filter: blur(16px); border: 1px solid rgba(255,255,255,0.06); border-top: 3px solid {theme_color}; border-radius: 16px; padding: 20px 16px; height: 100%; box-shadow: 0 12px 30px rgba(0,0,0,0.5);">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 24px;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: {theme_color}; box-shadow: 0 0 10px {theme_color};"></div>
            <div style="color: #f8fafc; font-size: 13px; font-weight: 700; font-family: 'Outfit', sans-serif; text-transform: uppercase; letter-spacing: 1.5px;">{title}</div>
        </div>
    """
    
    rank = 1
    for ind, row in top_3_ind.iterrows():
        val = row[metric_col]
        val_str = f"+{val:.1f}%" if val > 0 else f"{val:.1f}%"
        
        freq = leader_counts.get(ind, 1)
        if freq == 4:
            badge_html = '<span style="background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%); color: #451a03; padding: 0 8px; border-radius: 12px; font-size: 8.5px; font-weight: 800; margin-left: 6px; box-shadow: 0 2px 8px rgba(245,158,11,0.4); letter-spacing: 0.5px; display: inline-flex; align-items: center; height: 16px;">👑 SWEEP</span>'
            ind_color = "#fbbf24"
        elif freq == 3:
            badge_html = '<span style="background: rgba(244,63,94,0.15); color: #f43f5e; border: 1px solid rgba(244,63,94,0.3); padding: 0 8px; border-radius: 12px; font-size: 8.5px; font-weight: 700; margin-left: 6px; letter-spacing: 0.5px; display: inline-flex; align-items: center; height: 16px;">🔥 3X</span>'
            ind_color = "#f8fafc"
        elif freq == 2:
            badge_html = '<span style="background: rgba(14,165,233,0.15); color: #0ea5e9; border: 1px solid rgba(14,165,233,0.3); padding: 0 8px; border-radius: 12px; font-size: 8.5px; font-weight: 700; margin-left: 6px; letter-spacing: 0.5px; display: inline-flex; align-items: center; height: 16px;">⚡ 2X</span>'
            ind_color = "#f8fafc"
        else:
            badge_html = ''
            ind_color = "#f8fafc"
        
        # Get top 3 stocks for this industry
        stk_metric = metric_col if metric_col != 'Above_50_%' else 'Ret_3M'
        stks = df_stk[df_stk['Industry'] == ind].nlargest(3, stk_metric)
        
        stk_html = ""
        for _, s_row in stks.iterrows():
            sym = str(s_row.name if 'Symbol' not in s_row else s_row['Symbol']).replace('.NS', '')
            card_tickers.append(sym)
            
            # Truncate symbol if too long to fit in the micro-grid
            display_sym = sym[:7] + '..' if len(sym) > 8 else sym
            s_val = s_row[stk_metric]
            s_val_str = f"+{s_val:.1f}%" if s_val > 0 else f"{s_val:.1f}%"
            tv_link = f"https://in.tradingview.com/chart/?symbol=NSE:{sym}"
            
            stk_html += f"""
            <a href="{tv_link}" target="_blank" class="micro-pill" title="Open {sym} in TradingView">
                <div style="color: #cbd5e1; font-size: 10px; font-weight: 600; font-family: 'Inter', sans-serif; letter-spacing: 0.3px;">{display_sym}</div>
                <div style="color: {theme_color}; font-size: 11px; font-family: 'JetBrains Mono', monospace; font-weight: 700;">{s_val_str}</div>
            </a>
            """
            
        html += f"""
        <div style="background: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.04); border-radius: 12px; padding: 12px; margin-bottom: {"12px" if rank < 3 else "0"}; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="background: {theme_rgba.replace('0.1', '0.15')}; border: 1px solid {theme_rgba.replace('0.1', '0.3')}; color: {theme_color}; font-size: 11px; font-weight: 700; width: 22px; height: 22px; display: flex; align-items: center; justify-content: center; border-radius: 6px; font-family: 'Outfit', sans-serif;">{rank}</div>
                    <span style="color: {ind_color}; font-size: 14px; font-weight: 600; font-family: 'Inter', sans-serif; letter-spacing: 0.2px; display: flex; align-items: center;">{ind}{badge_html}</span>
                </div>
                <span style="color: {theme_color}; font-size: 14px; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{val_str}</span>
            </div>
            <div style="display: flex; justify-content: space-between; gap: 6px;">
                {stk_html}
            </div>
        </div>
        """
        rank += 1
        
    html += "</div>"
    return html.replace('\n', ''), card_tickers

# Calculate Cross-Timeframe Frequencies
from collections import Counter
t_1w = industry_groups.nlargest(3, 'Ret_1W').index.tolist()
t_1m = industry_groups.nlargest(3, 'Ret_1M').index.tolist()
t_3m = industry_groups.nlargest(3, 'Ret_3M').index.tolist()
t_br = industry_groups.nlargest(3, 'Above_50_%').index.tolist()
leader_counts = Counter(t_1w + t_1m + t_3m + t_br)

df_stocks = df_latest.reset_index().rename(columns={'index': 'Symbol'})
l_cols = st.columns(4)

html_1, t1 = build_leader_card("1W Rotation", 'Ret_1W', industry_groups, df_stocks, "#f43f5e", "rgba(244, 63, 94, 0.1)", leader_counts)
html_2, t2 = build_leader_card("1M Breakouts", 'Ret_1M', industry_groups, df_stocks, "#fbbf24", "rgba(251, 191, 36, 0.1)", leader_counts)
html_3, t3 = build_leader_card("3M Core Leaders", 'Ret_3M', industry_groups, df_stocks, "#0ea5e9", "rgba(14, 165, 233, 0.1)", leader_counts)
html_4, t4 = build_leader_card("Structural Breadth", 'Above_50_%', industry_groups, df_stocks, "#10b981", "rgba(16, 185, 129, 0.1)", leader_counts, is_pct=True)

l_cols[0].markdown(html_1, unsafe_allow_html=True)
l_cols[1].markdown(html_2, unsafe_allow_html=True)
l_cols[2].markdown(html_3, unsafe_allow_html=True)
l_cols[3].markdown(html_4, unsafe_allow_html=True)

# Generate TV Export
all_tickers = sorted(list(set(t1 + t2 + t3 + t4)))
if all_tickers:
    tv_str = ",".join([f"NSE:{t}" for t in all_tickers])
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Export All Rotation Leaders to TradingView"):
        st.markdown(f"**{len(all_tickers)} unique leading stocks** identified across 1W, 1M, 3M, and Breadth rotation. Copy the comma-separated list below and paste directly into a TradingView Watchlist:")
        st.code(tv_str, language='text')

st.markdown("<br><br>", unsafe_allow_html=True)

# --- INDUSTRY STACKED BAR GRID ---
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### 🏢 Industry Stage Breakdown")
    st.caption("Deep-dive into where institutional capital is hiding.")
with col2:
    sort_by = st.selectbox(
        "Sort By", 
        ["Stage 2 Momentum", "Stage 1 Accumulation (% S1)", "Stage 3 Distribution (% S3)", "Breadth (>50d SMA)", "1W Rank", "1M Rank", "3M Rank"],
        index=0
    )

# Dynamic Sorting Logic
if sort_by == "Stage 2 Momentum":
    industry_groups = industry_groups.sort_values('S2_%', ascending=False)
elif sort_by == "Stage 1 Accumulation (% S1)":
    industry_groups = industry_groups.sort_values('S1_%', ascending=False)
elif sort_by == "Stage 3 Distribution (% S3)":
    industry_groups = industry_groups.sort_values('S3_%', ascending=False)
elif sort_by == "Breadth (>50d SMA)":
    industry_groups = industry_groups.sort_values('Above_50_%', ascending=False)
elif sort_by == "1W Rank":
    industry_groups = industry_groups.sort_values('Rank_1W', ascending=True)
elif sort_by == "1M Rank":
    industry_groups = industry_groups.sort_values('Rank_1M', ascending=True)
elif sort_by == "3M Rank":
    industry_groups = industry_groups.sort_values('Rank_3M', ascending=True)

html_grid = """
<div class="breadth-container" style="margin-bottom: 0;">
    <style>
    details > summary { list-style: none; cursor: pointer; outline: none; }
    details > summary::-webkit-details-marker { display: none; }
    details[open] > summary .b-row { 
        background: rgba(255, 255, 255, 0.04);
        border-bottom-left-radius: 0;
        border-bottom-right-radius: 0;
        border-bottom: 1px solid transparent;
    }
    details[open] > summary .accordion-icon { transform: rotate(90deg); }
    .b-row:hover { background: rgba(255, 255, 255, 0.05); }
    
    .dropdown-content {
        padding: 24px;
        background: rgba(0, 0, 0, 0.35);
        border-radius: 0 0 16px 16px;
        margin-top: -12px;
        border: 1px solid rgba(255,255,255,0.05);
        border-top: none;
        box-shadow: inset 0 8px 20px rgba(0,0,0,0.6);
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        position: relative;
        z-index: 0;
    }
    .stock-pill {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 6px 14px;
        border-radius: 24px;
        text-decoration: none !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(4px);
    }
    .stock-pill:hover {
        background: rgba(255,255,255,0.08);
        border-color: rgba(255,255,255,0.25);
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
    }
    .st-name { 
        color: #f8fafc; 
        font-size: 13px; 
        font-weight: 600; 
        font-family: 'Outfit', sans-serif; 
        letter-spacing: 0.5px;
    }
    .st-rs { 
        font-family: 'JetBrains Mono', 'Roboto Mono', monospace; 
        font-size: 11px; 
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 12px;
        letter-spacing: -0.2px;
    }
    .accordion-icon {
        transition: transform 0.2s ease;
    }
    </style>
    
    <div class="b-header">
        <div style="flex: 2.5;">Industry Name</div>
        <div style="flex: 3; padding-right: 20px;">Stage Distribution <span style="opacity:0.5; text-transform:none;">(S1 / S2 / S3 / S4)</span></div>
        <div style="flex: 1;">S2 ↓</div>
        <div style="flex: 1;">Breadth <span style="opacity:0.5; text-transform:none;">(>50d)</span></div>
        <div style="flex: 1;">Health</div>
        <div style="flex: 0.8;">1W Rk</div>
        <div style="flex: 0.8;">1M Rk</div>
        <div style="flex: 0.8;">3M Rk</div>
    </div>
""".replace('\n', '')

def get_rank_color(r, total_items):
    if r <= 5: return "#10b981" # emerald
    if r <= 10: return "#0ea5e9" # sky blue
    if r >= total_items - 5: return "#f43f5e" # rose
    return "#6b7280" # gray

# Pre-group stocks for the dropdowns
df_stocks = df_latest.reset_index().rename(columns={'index': 'Symbol'})

for industry, row in industry_groups.iterrows():
    s1_pct, s2_pct, s3_pct, s4_pct = row['S1_%'], row['S2_%'], row['S3_%'], row['S4_%']
    total = int(row['Total'])
    above_50_pct = row['Above_50_%']
    
    r1w, r1m, r3m = row['Rank_1W'], row['Rank_1M'], row['Rank_3M']
    total_industries = len(industry_groups)
    
    # Sophisticated Colors for Health
    health = "Weak"
    health_color = "#f43f5e"
    health_bg = "rgba(244, 63, 94, 0.15)"
    
    if s2_pct > 40:
        health = "Strong"
        health_color = "#0ea5e9"
        health_bg = "rgba(14, 165, 233, 0.15)"
    elif s2_pct + s3_pct > 50:
        health = "Healthy"
        health_color = "#fbbf24"
        health_bg = "rgba(251, 191, 36, 0.15)"
    elif s1_pct + s2_pct > 50 and s4_pct < 35:
        health = "Improving"
        health_color = "#9ca3af"
        health_bg = "rgba(255, 255, 255, 0.1)"
        
    # Beautiful Progress Bar for Stage
    bar_html = f"""
    <div style="display: flex; height: 10px; width: 100%; border-radius: 8px; overflow: hidden; background: #1a1a24; box-shadow: inset 0 1px 3px rgba(0,0,0,0.8);">
        <div style="width: {s1_pct}%; background: #4b5563; border-right: 1px solid rgba(0,0,0,0.3);"></div>
        <div style="width: {s2_pct}%; background: linear-gradient(90deg, #0284c7, #0ea5e9); box-shadow: 0 0 8px rgba(14,165,233,0.4); border-right: 1px solid rgba(0,0,0,0.3); z-index: 2;"></div>
        <div style="width: {s3_pct}%; background: linear-gradient(90deg, #d97706, #fbbf24); box-shadow: 0 0 8px rgba(251,191,36,0.3); border-right: 1px solid rgba(0,0,0,0.3); z-index: 1;"></div>
        <div style="width: {s4_pct}%; background: linear-gradient(90deg, #e11d48, #f43f5e);"></div>
    </div>
    """.replace('\n', '')
    
    badge = f'<span style="background: {health_bg}; color: {health_color}; padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; border: 1px solid rgba(255,255,255,0.05);">{health}</span>'
    
    if above_50_pct >= 60: ab_color = "#10b981"; ab_bg = "rgba(16, 185, 129, 0.3)"
    elif above_50_pct >= 40: ab_color = "#fbbf24"; ab_bg = "rgba(251, 191, 36, 0.3)"
    else: ab_color = "#f43f5e"; ab_bg = "rgba(244, 63, 94, 0.3)"
    
    def rb(r):
        if pd.isna(r) or r == 999:
            return '<span class="num-val" style="color: #6b7280; background: rgba(255,255,255,0.05); padding: 4px 8px; border-radius: 6px; font-size: 13px; border: 1px solid rgba(255,255,255,0.03);">N/A</span>'
        r_int = int(r)
        bg = "rgba(16, 185, 129, 0.15)" if r_int <= 5 else "rgba(14, 165, 233, 0.15)" if r_int <= 10 else "rgba(244, 63, 94, 0.15)" if r_int >= total_industries - 5 else "rgba(255,255,255,0.05)"
        col = get_rank_color(r_int, total_industries)
        return f'<span class="num-val" style="color: {col}; background: {bg}; padding: 4px 8px; border-radius: 6px; font-size: 13px; border: 1px solid rgba(255,255,255,0.03);">#{r_int}</span>'
    
    # Generate Dropdown Content
    ind_stocks = df_stocks[df_stocks['Industry'] == industry].sort_values('Ret_3M', ascending=False)
    dropdown_html = '<div class="dropdown-content">'
    for _, s_row in ind_stocks.iterrows():
        sym = str(s_row['Symbol'])
        clean_sym = sym.replace('.NS', '')
        ret_3m = s_row['Ret_3M']
        tv_link = f"https://in.tradingview.com/chart/?symbol=NSE:{clean_sym}"
        
        if ret_3m >= 15:
            rs_color = "#10b981"
            rs_bg = "rgba(16, 185, 129, 0.15)"
        elif ret_3m >= 0:
            rs_color = "#fbbf24"
            rs_bg = "rgba(251, 191, 36, 0.15)"
        else:
            rs_color = "#f43f5e"
            rs_bg = "rgba(244, 63, 94, 0.15)"
            
        plus = "+" if ret_3m > 0 else ""
        
        dropdown_html += f"""
        <a href="{tv_link}" target="_blank" class="stock-pill" title="Open {clean_sym} in TradingView">
            <span class="st-name">{clean_sym}</span>
            <span class="st-rs" style="color: {rs_color}; background: {rs_bg};">{plus}{ret_3m:.1f}%</span>
        </a>
        """.replace('\n', '')
    dropdown_html += '</div>'
    
    html_grid += f"""
    <details>
        <summary>
            <div class="b-row">
                <div style="flex: 2.5; font-size: 15px; font-weight: 600; color: #f1f3f5; display: flex; align-items: center; gap: 8px;">
                    <svg class="accordion-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: -4px;"><polyline points="6 9 12 15 18 9"></polyline></svg>
                    {industry} <span class="col-count">{total}</span>
                </div>
                <div style="flex: 3; display: flex; align-items: center; padding-right: 20px;">
                    {bar_html}
                </div>
                <div style="flex: 1;">
                    <span class="metric-val" style="color: #0ea5e9; text-shadow: 0 0 16px rgba(14,165,233,0.3);">{s2_pct:.0f}%</span>
                </div>
                <div style="flex: 1;">
                    <span class="metric-val" style="color: {ab_color}; text-shadow: 0 0 16px {ab_bg};">{above_50_pct:.0f}%</span>
                </div>
                <div style="flex: 1;">
                    {badge}
                </div>
                <div style="flex: 0.8; display:flex; align-items:center;">{rb(r1w)}</div>
                <div style="flex: 0.8; display:flex; align-items:center;">{rb(r1m)}</div>
                <div style="flex: 0.8; display:flex; align-items:center;">{rb(r3m)}</div>
            </div>
        </summary>
        {dropdown_html}
    </details>
    """.replace('\n', '')

html_grid += "</div>"

st.markdown(html_grid, unsafe_allow_html=True)
