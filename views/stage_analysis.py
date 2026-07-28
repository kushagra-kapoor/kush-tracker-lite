import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import requests
import io




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
        return None, None, None
        
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
        return None, None, None, None
        
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
    above_50 = (latest_close > latest_sma50)
    
    ret_1w = (close_df.iloc[-1] / close_df.iloc[-6] - 1) * 100 if len(close_df) >= 6 else pd.Series(0, index=close_df.columns)
    ret_1m = (close_df.iloc[-1] / close_df.iloc[-22] - 1) * 100 if len(close_df) >= 22 else pd.Series(0, index=close_df.columns)
    ret_3m = (close_df.iloc[-1] / close_df.iloc[-64] - 1) * 100 if len(close_df) >= 64 else pd.Series(0, index=close_df.columns)
    
    df_latest = pd.DataFrame({
        'Stage': latest_stages,
        'Above_50': above_50,
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
    
    return momentum_metrics, historical_counts, df_latest, df_segments

st.title("🔬 Market Internals & Stage Analysis")
st.markdown("Quantifying true institutional money flow across the **Nifty Total Market** universe. Filtered for liquidity.")

with st.spinner("Calculating quantitative internal metrics..."):
    momentum_metrics, historical_counts, df_latest, df_segments = load_and_compute_internals()

if momentum_metrics is None:
    st.warning("Historical data not found. Please run the database builder.")
    st.stop()

# --- CUSTOM HTML GENERATORS ---
def format_num(num):
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return f"{num:.0f}"

def dual_bar(val1, val2, label1, label2, title):
    total = val1 + val2 if (val1 + val2) > 0 else 1
    pct1 = (val1 / total) * 100
    
    return f"""
    <div style="margin-bottom: 25px; font-family: 'Inter', sans-serif;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: #eee; font-size: 13px; font-weight: 500;">{title}</span>
            <span style="color: #0088ff; font-size: 13px; font-weight: bold;">{pct1:.1f}%</span>
        </div>
        <div style="width: 100%; height: 16px; background: #2b2b36; border-radius: 8px; overflow: hidden; display: flex;">
            <div style="width: {pct1}%; background: #0088ff;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 6px;">
            <span style="color: #888; font-size: 12px;">{format_num(val1)} {label1}</span>
            <span style="color: #888; font-size: 12px;">{format_num(val2)} {label2}</span>
        </div>
    </div>
    """

# --- MARKET CAP BREADTH MATRIX ---
# --- MARKET CAP BREADTH MATRIX ---
if df_segments is not None and not df_segments.empty:
    st.markdown("### 🏢 Market Cap Breadth Matrix")
    st.caption("5-second read on institutional money flow (Large vs Mid vs Small).")
    
    html_matrix = """
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
    .breadth-container {
        font-family: 'Outfit', sans-serif;
        background: rgba(18, 18, 24, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 40px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .b-header {
        display: flex;
        padding: 0 16px 16px 16px;
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
        padding: 20px 16px;
        border-bottom: 1px solid rgba(255,255,255,0.03);
        transition: all 0.3s ease;
        border-radius: 12px;
    }
    .b-row:hover {
        background: rgba(255, 255, 255, 0.03);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }
    .b-row:last-child {
        border-bottom: none;
    }
    .col-name { flex: 1.5; font-size: 16px; font-weight: 600; color: #f1f3f5; display: flex; align-items: center; gap: 10px; }
    .col-count { font-size: 11px; background: rgba(255,255,255,0.08); padding: 3px 8px; border-radius: 20px; color: #a1aab5; font-weight: 500; }
    .col-metric { flex: 1; display: flex; flex-direction: column; gap: 4px; }
    .metric-val { font-size: 18px; font-weight: 700; letter-spacing: -0.5px; }
    .col-pulse { flex: 1.5; display: flex; flex-direction: column; gap: 8px; justify-content: center; }
    
    .pulse-track { display: flex; gap: 4px; height: 8px; width: 100%; border-radius: 10px; background: transparent; }
    .pulse-seg { height: 100%; border-radius: 10px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); }
    .pulse-labels { display: flex; justify-content: space-between; font-size: 12px; font-weight: 500; }
    </style>
    
    <div class="breadth-container">
        <div class="b-header">
            <div style="flex: 1.5;">Market Segment</div>
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
        
        # Sophisticated HSL Colors
        # Green: #10b981, Yellow: #fbbf24, Red: #f43f5e
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
                {seg} <span class="col-count">{total}</span>
            </div>
            
            <div class="col-metric">
                <span class="metric-val" style="color: {bs_color}; text-shadow: 0 0 16px {bs_bg};">{bs_pct:.1f}%</span>
            </div>
            
            <div class="col-metric">
                <span class="metric-val" style="color: {ab_color}; text-shadow: 0 0 16px {ab_bg};">{ab_50_pct:.1f}%</span>
            </div>
            
            <div class="col-pulse">
                <div class="pulse-track">
                    <div class="pulse-seg" style="width: {adv_pct}%; background: linear-gradient(90deg, #059669, #10b981); box-shadow: 0 0 10px rgba(16,185,129,0.3);"></div>
                    <div class="pulse-seg" style="width: {flat_pct}%; background: #374151;"></div>
                    <div class="pulse-seg" style="width: {dec_pct}%; background: linear-gradient(90deg, #e11d48, #f43f5e); box-shadow: 0 0 10px rgba(244,63,94,0.3);"></div>
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


# --- MOMENTUM BREADTH UI ---
st.markdown("### 🌊 Daily Momentum Breadth")
_, center, _ = st.columns([1, 2, 1])
with center:
    card_html = f"""
    <div style="background: #111115; padding: 30px; border-radius: 12px; border: 1px solid #222;">
        {dual_bar(momentum_metrics['up_open'], momentum_metrics['down_open'], "Up", "Down", "Up from Open vs Down from Open")}
        {dual_bar(momentum_metrics['up_vol'], momentum_metrics['down_vol'], "Up", "Down", "Up on Volume vs Down on Volume")}
        <div style="margin-bottom: -25px;">
            {dual_bar(momentum_metrics['up_4'], momentum_metrics['down_4'], "Up", "Down", "Up 4% vs Down 4%")}
        </div>
    </div>
    """.replace('\n', '')
    st.markdown(card_html, unsafe_allow_html=True)

st.markdown("---")

# --- STAGE ANALYSIS TOP CARDS ---
st.markdown("### 📊 Market Stage Analysis")
st.caption("Categorizing the Nifty Total Market into Stan Weinstein's 4 Stages.")

total_stocks = len(df_latest)
s1_count = len(df_latest[df_latest['Stage'] == 'S1'])
s2_count = len(df_latest[df_latest['Stage'] == 'S2'])
s3_count = len(df_latest[df_latest['Stage'] == 'S3'])
s4_count = len(df_latest[df_latest['Stage'] == 'S4'])

cc1, cc2, cc3, cc4 = st.columns(4)
cc1.markdown(f"""
<div style="background: #16161c; padding: 25px 20px; border-radius: 12px; text-align: center; border-top: 4px solid #555; border-bottom: 1px solid #222; border-left: 1px solid #222; border-right: 1px solid #222; box-shadow: 0 8px 24px rgba(0,0,0,0.4); font-family: 'Inter', sans-serif;">
    <div style="color: #888; font-size: 13px; font-weight: 700; margin-bottom: 8px; letter-spacing: 0.5px;">S1 (BASING)</div>
    <div style="font-size: 28px; font-weight: 800; color: #aaa;">{s1_count}</div>
    <div style="font-size: 13px; color: #666; font-weight: 500; margin-top: 4px;">{(s1_count/total_stocks)*100:.1f}%</div>
</div>
""".replace('\n', ''), unsafe_allow_html=True)

cc2.markdown(f"""
<div style="background: #16161c; padding: 25px 20px; border-radius: 12px; text-align: center; border-top: 4px solid #0088ff; border-bottom: 1px solid #222; border-left: 1px solid #222; border-right: 1px solid #222; box-shadow: 0 4px 20px rgba(0, 136, 255, 0.12); font-family: 'Inter', sans-serif;">
    <div style="color: #0088ff; font-size: 13px; font-weight: 700; margin-bottom: 8px; letter-spacing: 0.5px;">S2 (ADVANCING)</div>
    <div style="font-size: 28px; font-weight: 800; color: white;">{s2_count}</div>
    <div style="font-size: 13px; color: #0088ff; font-weight: 500; margin-top: 4px;">{(s2_count/total_stocks)*100:.1f}%</div>
</div>
""".replace('\n', ''), unsafe_allow_html=True)

cc3.markdown(f"""
<div style="background: #16161c; padding: 25px 20px; border-radius: 12px; text-align: center; border-top: 4px solid #ffaa00; border-bottom: 1px solid #222; border-left: 1px solid #222; border-right: 1px solid #222; box-shadow: 0 4px 20px rgba(255, 170, 0, 0.1); font-family: 'Inter', sans-serif;">
    <div style="color: #ffaa00; font-size: 13px; font-weight: 700; margin-bottom: 8px; letter-spacing: 0.5px;">S3 (TOPPING)</div>
    <div style="font-size: 28px; font-weight: 800; color: white;">{s3_count}</div>
    <div style="font-size: 13px; color: #ffaa00; font-weight: 500; margin-top: 4px;">{(s3_count/total_stocks)*100:.1f}%</div>
</div>
""".replace('\n', ''), unsafe_allow_html=True)

cc4.markdown(f"""
<div style="background: #16161c; padding: 25px 20px; border-radius: 12px; text-align: center; border-top: 4px solid #ff007f; border-bottom: 1px solid #222; border-left: 1px solid #222; border-right: 1px solid #222; box-shadow: 0 4px 20px rgba(255, 0, 127, 0.15); font-family: 'Inter', sans-serif;">
    <div style="color: #ff007f; font-size: 13px; font-weight: 700; margin-bottom: 8px; letter-spacing: 0.5px;">S4 (DECLINING)</div>
    <div style="font-size: 28px; font-weight: 800; color: white;">{s4_count}</div>
    <div style="font-size: 13px; color: #ff007f; font-weight: 500; margin-top: 4px;">{(s4_count/total_stocks)*100:.1f}%</div>
</div>
""".replace('\n', ''), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- TREND CHART ---
recent_hist = historical_counts.tail(252).copy()
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=recent_hist.index, y=recent_hist['S2'], name='S2 (Advancing)', 
    line=dict(color='#0088ff', width=2), fill='tozeroy', fillcolor='rgba(0,136,255,0.15)'
))
fig.add_trace(go.Scatter(
    x=recent_hist.index, y=recent_hist['S4'], name='S4 (Declining)', 
    line=dict(color='#ff007f', width=2), fill='tozeroy', fillcolor='rgba(255,0,127,0.15)'
))
fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title="S2 (Bulls) vs S4 (Bears) Trend (1-Year)",
    margin=dict(l=20, r=20, t=40, b=20),
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

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
        ["Stage 2 Momentum", "Breadth (>50d SMA)", "1W Rank", "1M Rank", "3M Rank"],
        index=0
    )

# Dynamic Sorting Logic
if sort_by == "Stage 2 Momentum":
    industry_groups = industry_groups.sort_values('S2_%', ascending=False)
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

