import streamlit as st

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from styles import load_css
    load_css()
except ImportError:
    pass

import pandas as pd
import yfinance as yf
import requests
import io
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import concurrent.futures

from config import BENCHMARK_TICKER, DATA_SETTINGS, RS_WEIGHTS
from macro_regime_engine import (
    calculate_distribution_days, detect_follow_through_day, 
    get_market_regime_label, aggregate_sector_strength, standardize_columns,
    calculate_leadership_metrics, calculate_breadth_thrust
)
from relative_strength import calculate_stock_returns, calculate_relative_returns, calculate_weighted_rs_raw, normalize_rs_against_nifty500
from trend_analyzer import determine_trend_state
from technical_indicators import add_technical_indicators

# Page Config
st.set_page_config(page_title="🏛️ Market Regime & Sector RS", layout="wide")

# CSS Styling (Shared with main app)


@st.cache_data(ttl=3600*24)
def fetch_universe_with_industry(cache_buster=2):
    """Fetch 750 universe with industry mapping."""
    valid_tickers = []
    industry_map = {}
    try:
        from market_data import fetch_nifty_total_market_tickers
        valid_tickers, industry_map = fetch_nifty_total_market_tickers(show_progress=False, return_industry_map=True)
    except Exception as e:
        st.error(f"Failed to fetch universe: {e}")
        
    if not valid_tickers and os.path.exists('tickers.txt'):
        with open('tickers.txt', 'r') as f:
            valid_tickers = [line.strip().upper() for line in f if line.strip()]
            valid_tickers = [t + '.NS' if not t.endswith('.NS') else t for t in valid_tickers]
            industry_map = {t: "Unknown" for t in valid_tickers}
            
    return pd.DataFrame({'Symbol': [t.replace('.NS', '') for t in valid_tickers], 'Industry': [industry_map.get(t, "Unknown") for t in valid_tickers]})

def fetch_yfinance_batch(tickers, days=500):
    """Download historical data for a list of tickers. 500 days ensures enough history for backward-looking metrics."""
    yf_tickers = [f"{t}.NS" for t in tickers]
    data = yf.download(yf_tickers, period=f"{days}d", group_by='ticker', progress=False)
    return data

def process_macro_data(history_df, universe_df, index_n50, index_n500):
    """Computes all macro and sector metrics with full vectorization."""
    # 1. Index Regime Analysis
    n50_dd_count, _ = calculate_distribution_days(index_n50)
    n500_dd_count, _ = calculate_distribution_days(index_n500)
    ftd_detected, ftd_date = detect_follow_through_day(index_n500)
    
    index_n500_cols = standardize_columns(index_n500)
    if index_n500_cols.empty or 'close' not in index_n500_cols.columns:
        curr_n500, n500_sma50, bench_r = 0, 1, {'R1':0.0, 'R3':0.0, 'R6':0.0}
    else:
        n500_cl = index_n500_cols['close']
        n500_sma50 = n500_cl.rolling(50).mean().iloc[-1]
        curr_n500 = n500_cl.iloc[-1]
        bench_r = calculate_stock_returns(index_n500_cols)
        # Fallback to 0.0 if not enough data for certain periods
        for key in ['R1', 'R3', 'R6']:
            if bench_r.get(key) is None: 
                bench_r[key] = 0.0

    # 2. Vectorized Stock Analysis
    # Get all tickers (with .NS)
    all_tickers = [f"{t}.NS" for t in universe_df['Symbol'].tolist()]
    existing_tickers = [t for t in all_tickers if t in history_df.columns.levels[0]]
    
    # Extract 'Close' prices for all existing tickers
    close_prices = history_df.xs('Close', level=1, axis=1)[existing_tickers].fillna(method='ffill')
    
    # Calculate Returns (1M=21d, 3M=63d, 6M=126d)
    r1 = (close_prices.iloc[-1] / close_prices.iloc[-(DATA_SETTINGS['TRADING_DAYS_1M']+1)]) - 1
    r3 = (close_prices.iloc[-1] / close_prices.iloc[-(DATA_SETTINGS['TRADING_DAYS_3M']+1)]) - 1
    r6 = (close_prices.iloc[-1] / close_prices.iloc[-(DATA_SETTINGS['TRADING_DAYS_6M']+1)]) - 1
    
    # Historical Returns (T-21)
    r1_t21 = (close_prices.iloc[-22] / close_prices.iloc[-(DATA_SETTINGS['TRADING_DAYS_1M']+22)]) - 1
    r3_t21 = (close_prices.iloc[-22] / close_prices.iloc[-(DATA_SETTINGS['TRADING_DAYS_3M']+22)]) - 1
    r6_t21 = (close_prices.iloc[-22] / close_prices.iloc[-(DATA_SETTINGS['TRADING_DAYS_6M']+22)]) - 1
    
    # Relative Returns vs Benchmark
    rr1, rr3, rr6 = r1 - bench_r['R1'], r3 - bench_r['R3'], r6 - bench_r['R6']
    # Approximate bench_r_t21 as r21d lag (close enough for ranking)
    rr1_t21, rr3_t21, rr6_t21 = r1_t21 - bench_r['R1'], r3_t21 - bench_r['R3'], r6_t21 - bench_r['R6']

    # RS Raw & Normalized Score (0-100 percentile rank within the 750 universe)
    rs_raw = (rr1 * RS_WEIGHTS['1M']) + (rr3 * RS_WEIGHTS['3M']) + (rr6 * RS_WEIGHTS['6M'])
    rs_score = rs_raw.rank(pct=True) * 100
    
    rs_raw_t21 = (rr1_t21 * RS_WEIGHTS['1M']) + (rr3_t21 * RS_WEIGHTS['3M']) + (rr6_t21 * RS_WEIGHTS['6M'])
    rs_score_t21 = rs_raw_t21.rank(pct=True) * 100

    # Trend Calculations (Vectorized)
    ema8 = close_prices.ewm(span=8, adjust=False).mean().iloc[-1]
    ema21 = close_prices.ewm(span=21, adjust=False).mean().iloc[-1]
    sma50 = close_prices.rolling(window=50).mean().iloc[-1]
    last_close = close_prices.iloc[-1]
    prev_close = close_prices.iloc[-2]
    
    # 52W High Proximity (Vectorized)
    high_52w = close_prices.tail(252).max()
    is_near_high = (last_close >= (high_52w * 0.9)).astype(int)

    # Assemble Results
    processed_results = []
    rs_dict = {}
    
    ticker_industry_map = universe_df.set_index('Symbol')['Industry'].to_dict()
    
    for t in existing_tickers:
        clean_t = t.replace('.NS', '')
        score = rs_score[t]
        rs_dict[clean_t] = score
        
        # Determine Trend State (Vectorized logic check)
        lc, e8, e21, s50 = last_close[t], ema8[t], ema21[t], sma50[t]
        if lc > e8 and e8 > e21 and e21 > s50: trend = '🟢 Strong'
        elif lc > s50: trend = '🟡 Pullback'
        elif lc > (s50 * 0.95): trend = '🟠 Warning'
        else: trend = '🔴 Broken'
        
        processed_results.append({
            'Ticker': clean_t,
            'Industry': ticker_industry_map.get(clean_t, 'Other'),
            'Today %': ((lc - prev_close[t]) / prev_close[t]) * 100,
            'RS Score': score,
            'RS_Score_T21': rs_score_t21[t],
            'Trend State': trend,
            'Near_High': is_near_high[t]
        })
        
    results_df = pd.DataFrame(processed_results)
    sector_df = aggregate_sector_strength(results_df)

    leadership_metrics = calculate_leadership_metrics(history_df, existing_tickers, rs_dict)
    breadth_metrics = calculate_breadth_thrust(history_df, existing_tickers)
    
    return {
        'n50_dd': n50_dd_count,
        'n500_dd': n500_dd_count,
        'n500_curr': curr_n500,
        'n500_sma50': n500_sma50,
        'ftd': ftd_date,
        'sector_df': sector_df,
        'leadership': leadership_metrics,
        'breadth': breadth_metrics
    }

def main():
    from components import render_header
    render_header("🏛️ Market Regime & Sector RS", "Automated macro intelligence based on William O'Neil & Mark Minervini principles.")
    
    if st.button("🔄 Refresh Macro Analysis", type="primary"):
        st.session_state['macro_refresh'] = True

    # Auto-run if no results exist yet
    if 'macro_results' not in st.session_state:
        st.session_state['macro_refresh'] = True

    if st.session_state.get('macro_refresh', False):
        st.session_state['macro_refresh'] = False
        
        with st.spinner("Analyzing Market Indices & 750 Stocks..."):
            universe_df = fetch_universe_with_industry()
            tickers = universe_df['Symbol'].tolist()
            
            # Index data expects at least 250d to calculate 6M return + T-21 offsets safely
            index_n50 = yf.download('^NSEI', period="250d", progress=False)
            index_n500 = yf.download('^CRSLDX', period="250d", progress=False)
            
            # Stock data
            history_df = fetch_yfinance_batch(tickers, days=500)
            
            # Process
            macro_results = process_macro_data(history_df, universe_df, index_n50, index_n500)
            st.session_state['macro_results'] = macro_results
            st.session_state['macro_last_updated'] = datetime.now().strftime('%H:%M:%S')

    macro_results = st.session_state.get('macro_results')
    
    # Defensive Check: If data is from an older version (missing new Power Metrics), force a refresh
    if macro_results and 'sector_df' in macro_results:
        s_df = macro_results['sector_df']
        required_cols = ['Leadership Density', 'RS Momentum 21d', 'Near High Breadth']
        if not all(col in s_df.columns for col in required_cols):
            st.warning("⚠️ Data version mismatch detected. Upgrading sector metrics...")
            st.session_state['macro_refresh'] = True
            st.rerun()

    if not macro_results:
        st.info("Click Refresh to analyze the current market regime.")
        return

    # --- RENDER UI ---
    
    # 1. TOP HEADER: MARKET STATUS
    dd_count = macro_results['n500_dd']
    status_label, status_color, status_emoji = get_market_regime_label(
        dd_count, 
        macro_results['n500_curr'], 
        macro_results['n500_sma50']
    )
    
    st.markdown(f"""
    <div class="status-box {status_color}-status">
        <div class="status-title">{status_emoji} {status_label}</div>
        <div class="status-subtitle">
            Nifty 500 Distribution Days: <strong>{dd_count}</strong> (Rolling 25 Sessions) | 
            Price vs 50 SMA: <strong>{'+' if macro_results['n500_curr'] > macro_results['n500_sma50'] else '-'}{abs((macro_results['n500_curr']/macro_results['n500_sma50']-1)*100):.1f}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    from components import render_metric_card
    m_col1, m_col2, m_col3 = st.columns(3)
    n50_dd = macro_results['n50_dd']
    n500_dd = macro_results['n500_dd']
    ftd = macro_results['ftd']
    with m_col1:
        render_metric_card("Nifty 50 DD Count", f"{n50_dd}/25", color_class="red-text" if n50_dd >= 4 else "green-text")
        st.caption("Bearish" if n50_dd >= 4 else "Safe")
    with m_col2:
        render_metric_card("Nifty 500 DD Count", f"{n500_dd}/25", color_class="red-text" if n500_dd >= 4 else "green-text")
        st.caption("Bearish" if n500_dd >= 4 else "Safe")
    with m_col3:
        render_metric_card("Follow-Through Day", ftd if ftd else "None", color_class="green-text" if ftd else "yellow-text")

    st.markdown("")
    
    # -------------------------------------------------------------------------
    # 2. MARKET LEADERSHIP & BREADTH INTELLIGENCE PANEL
    # -------------------------------------------------------------------------
    st.header("⚡ Market Leadership & Breadth Intelligence")
    st.caption("Professional indicators tracking the health of market participation and leadership emergence.")
    
    leadership = macro_results.get('leadership')
    breadth = macro_results.get('breadth')
    
    if leadership and breadth:
        ler = leadership['ler_current']
        lac = leadership['lac_current']
        lt = leadership['lt_current']
        bt_ma = breadth['breadth_ma10_current']
        bt_trigger = breadth['thrust_triggered']
        
        # --- Interpretation Logic ---
        
        # LER Interpretation
        if ler < 0.5: ler_status = "Dead Market"
        elif ler < 1.5: ler_status = "Early Forming"
        elif ler < 3.0: ler_status = "Strong"
        else: ler_status = "Explosive Phase"
            
        # LAC Interpretation
        if lac < 0: lac_status = "Shrinking"
        elif lac < 2: lac_status = "Stagnant/Slow"
        elif lac < 6: lac_status = "Early Cycle"
        else: lac_status = "Powerful Expansion"
            
        # LT Interpretation
        if lt < 1.0: lt_status = "Weak"
        elif lt < 3.0: lt_status = "Normal"
        elif lt < 5.0: lt_status = "Expanding"
        else: lt_status = "Strong Bull"
            
        # Market Regime Engine
        regime_title = "Neutral/Mixed Environment"
        regime_desc = "Market is lacking clear unified direction. Trade carefully."
        
        if ler > 3.0 and lac > 0 and bt_trigger:
            regime_title = "🔥 Strong Momentum Phase"
            regime_desc = "High probability environment for breakouts. Aggressive exposure warranted."
        elif ler > 1.5 and lac > 0 and lt > 3.0:
            regime_title = "🌱 Early Leadership Expansion"
            regime_desc = "Environment strongly supportive for trend trading. Leaders are emerging."
        elif ler < 0.5 and lac < 0 and lt < 1.0:
            regime_title = "❄️ Weak Market Conditions"
            regime_desc = "Avoid aggressive exposure. Leadership is deteriorating rapidly."
            
        # Save regime state
        from database import save_market_regime
        save_market_regime(regime_title)
            
        # --- UI Rendering ---
        
        # Regime Interpretation Card
        st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid #38bdf8; border-radius: 10px; padding: 1.5rem; margin-bottom: 2rem;">
            <h3 style="margin-top:0; color: #38bdf8;">Overall Signal: {regime_title}</h3>
            <p style="margin-bottom:0; font-size: 1.1rem;">{regime_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Leader Emergence Rate (LER)", 
                value=f"{ler:.1f}%", 
                delta=ler_status, 
                delta_color="normal" if ler >= 1.5 else "off" if ler >= 0.5 else "inverse",
                help="Measures how many strong stocks are forming in the market. >1.5% signals strong leader environment."
            )
        with col2:
            st.metric(
                label="Leadership Acceleration (LAC)", 
                value=f"{lac:+.1f}/wk", 
                delta=lac_status,
                delta_color="normal" if lac > 2 else "off" if lac >= 0 else "inverse",
                help="Measures how quickly new leaders are appearing based on 3-week EMA of growth."
            )
        with col3:
            st.metric(
                label="Leadership Thrust (LT)", 
                value=f"{lt:.1f}%", 
                delta=lt_status,
                delta_color="normal" if lt > 3 else "off" if lt >= 1 else "inverse",
                help="Measures expansion in new highs (smoothed 5d). >3% signals expansion."
            )
        with col4:
            st.metric(
                label="Breadth Thrust (BT) MA10", 
                value=f"{bt_ma:.2f}",
                delta="🟢 Triggered!" if bt_trigger else ("Strong" if bt_ma > 0.6 else "Weak" if bt_ma < 0.4 else "Neutral"),
                delta_color="normal" if bt_trigger or bt_ma > 0.6 else "inverse" if bt_ma < 0.4 else "off",
                help="10-day MA of Advancers ratio. Thrust triggers when rising from <0.4 to >0.6 in 10 days."
            )
            
        st.write("")
        
        # Charts Row 1
        import plotly.graph_objects as go
        ch1, ch2 = st.columns(2)
        
        with ch1:
            ler_series = leadership['ler_series']
            fig_ler = px.line(ler_series, title="Leader Emergence Rate (%) - 24 Weeks", markers=True)
            fig_ler.update_layout(xaxis_title="", yaxis_title="LER %", showlegend=False, height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
            fig_ler.add_hline(y=1.5, line_dash="dash", line_color="green", opacity=0.5)
            st.plotly_chart(fig_ler, use_container_width=True)
            
        with ch2:
            lac_series = leadership['lac_series']
            fig_lac = px.bar(lac_series, title="Leadership Acceleration Curve (New/Wk) - 24 Weeks")
            fig_lac.update_traces(marker_color=['#10b981' if val > 0 else '#ef4444' for val in lac_series])
            fig_lac.update_layout(xaxis_title="", yaxis_title="Acceleration", showlegend=False, height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
            st.plotly_chart(fig_lac, use_container_width=True)
            
        # Charts Row 2
        ch3, ch4 = st.columns(2)
        
        with ch3:
            lt_series = leadership['lt_series']
            fig_lt = px.line(lt_series, title="Leadership Thrust (100 Days)")
            fig_lt.update_layout(xaxis_title="", yaxis_title="LT %", showlegend=False, height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
            fig_lt.add_hline(y=3.0, line_dash="dash", line_color="orange", opacity=0.5)
            st.plotly_chart(fig_lt, use_container_width=True)
            
        with ch4:
            bt_series = breadth['breadth_ma10_series']
            fig_bt = px.line(bt_series, title="Breadth Thrust MA10 (100 Days)")
            fig_bt.update_layout(xaxis_title="", yaxis_title="Advancers Ratio", showlegend=False, height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
            fig_bt.add_hline(y=0.60, line_dash="dash", line_color="green", annotation_text="Strong Buy")
            fig_bt.add_hline(y=0.40, line_dash="dash", line_color="red", annotation_text="Weak")
            st.plotly_chart(fig_bt, use_container_width=True)
            
        # User Education Section
        with st.expander("📚 How to Use Market Leadership Signals"):
            st.markdown("""
            These four signals help detect early bull cycles, strong trend phases, and deteriorating market conditions before price indices break down.

            - **Leader Emergence Rate (LER):** Shows how many strong stocks are forming in the market (RS > 85, >50SMA/200SMA, near 52W High, +25% in 8wks). When this rises after a correction, it often signals the beginning of a new leadership cycle.
            - **Leadership Acceleration Curve (LAC):** Shows how quickly these new leaders are appearing. Sharp increases often occur during the early stages of rapid bull markets.
            - **Leadership Thrust (LT):** Measures expansion in new 52-week highs. When this expands rapidly after corrections (>3%), it signals deep institutional buying across the board.
            - **Breadth Thrust (BT):** Shows broad market participation (Advancers vs Decliners smoothed over 10 days). A thrust occurs when buying expands rapidly (MA10 swings from below 0.40 to above 0.60), historically preceding strong macro upward phases.
            """)
            
    else:
        st.info("Market Leadership metrics require at least 252 days of continuous historical data to compute properly.")

    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # 3. SECTOR LEADERBOARD
    # -------------------------------------------------------------------------
    st.header("📈 Industry Group Leadership (Total Market)")
    st.caption("Aggregated Relative Strength and Momentum metrics across 750 stocks.")
    
    sector_df = macro_results['sector_df']
    
    # Filter out small industries
    sector_df = sector_df[sector_df['Stocks'] >= 3]
    
    # --- VISUAL: SECTOR HEATMAP (GRID) ---
    heatmap_cols = ['Avg_RS', 'Leadership Density', 'Bullish %', 'Near High Breadth']
    
    # Check if we have the columns
    if all(col in sector_df.columns for col in heatmap_cols):
        # Prepare data for heatmap
        heatmap_data = sector_df.set_index('Industry')[heatmap_cols].copy()
        heatmap_data = heatmap_data.rename(columns={'Bullish %': 'Trend Health'})
        
        # Sort by Leadership Density (institutional tracking)
        heatmap_data = heatmap_data.sort_values('Leadership Density', ascending=False)
        
        # Normalize each column to 0-1 range for independent coloring
        heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
        # Handle case where max == min (prevents NaNs)
        heatmap_norm = heatmap_norm.fillna(0.5)
        
        # Create a red-yellow-green colorscale
        color_scale = [
            [0.0, "#ef4444"],    # Red
            [0.5, "#eab308"],    # Yellow
            [1.0, "#10b981"]     # Green
        ]
        
        # Draw heatmap using normalized values for color, but showing raw text
        fig_heatmap = px.imshow(
            heatmap_norm,
            labels=dict(x="Metrics", y="Industry Group", color="Relative Score"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale=color_scale,
            aspect="auto",
            title="Sector Performance Heatmap (Sorted by Leadership Density, Colored by Column Rank)"
        )
        
        # Overlay actual raw numbers as text
        fig_heatmap.update_traces(
            text=heatmap_data.values.round(1),
            texttemplate="%{text}",
            hovertemplate="Industry: %{y}<br>Metric: %{x}<br>Value: %{text}<extra></extra>"
        )
        
        # Adjust height based on number of industries
        fig_heatmap.update_layout(height=max(500, len(heatmap_data) * 35), coloraxis_showscale=False)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.warning("Please refresh data to view the new Heatmap.")
    
    st.markdown("---")
    
    # --- DETAILED INDUSTRY POWER TABLE ---
    st.subheader("📊 Industry Power Metrics")
    st.dataframe(
        sector_df,
        column_config={
            "Industry": st.column_config.TextColumn("Industry Group", width="large"),
            "Avg_RS": st.column_config.NumberColumn("Avg RS (T-0)", format="%.1f"),
            "RS Momentum 21d": st.column_config.NumberColumn("RS Mom (21d)", format="%+.1f"),
            "Leadership Density": st.column_config.ProgressColumn("Leadership Density", min_value=0, max_value=100, format="%d%%"),
            "Bullish %": st.column_config.ProgressColumn("Trend Health (🟢)", min_value=0, max_value=100, format="%d%%"),
            "Near High Breadth": st.column_config.ProgressColumn("Near 52W High", min_value=0, max_value=100, format="%d%%"),
            "Stocks": st.column_config.NumberColumn("Stocks"),
            "Leaders_Count": st.column_config.NumberColumn("RS>85"),
            "Avg_Today_Pct": st.column_config.NumberColumn("Today %", format="%+.2f%%")
        },
        hide_index=True,
        use_container_width=True
    )

    with st.expander("ℹ️ Column Definitions & Methodology"):
        st.markdown("""
| Column | What It Represents | How It's Calculated |
|---|---|---|
| **Industry Group** | The NSE-defined industry classification for all ~750 Nifty Total Market stocks. | Groups with fewer than 3 stocks are filtered out. |
| **Avg RS (T-0)** | The current average Weighted Relative Strength score of the group. | Percentage-ranked (0-100) across all 750 stocks. |
| **RS Mom (21d)** | The change in the group's Average RS over the last month (21 trading days). | `Current Avg_RS - Historical (T-21) Avg_RS`. Positive values indicate the sector is actively gaining leadership. |
| **Leadership Density**| The percentage of stocks in the group that are "Elite Leaders" (RS > 85). | `(Stocks with RS > 85 / Total Stocks) × 100`. High density signals a true institutional theme. |
| **Trend Health (🟢)** | Percentage of stocks in a 🟢 Strong technical uptrend. | `(Stocks above 8EMA, 21EMA, and 50SMA / Total stocks) × 100`. |
| **Near 52W High** | Percentage of stocks within 10% of their 52-week high. | `(Stocks within 10% of 52W high / Total stocks) × 100`. Higher participation near highs precedes major breakouts. |
| **Stocks** | Total number of evaluated stocks belonging to this industry. | Count of valid data constituents in the universe. |
| **RS>85** | Raw count of stocks with RS score above 85. | Identify top-tier momentum names. |
| **Today %** | The average intraday percentage change for all stocks in the group today. | Arithmetic mean of the daily %. |

**Strategy: Spotting Institutional Rotation**
- **Leadership Density > 15%** indicates a primary leading sector.
- **RS Mom (21d) > +5.0** suggests a new group is appearing on the institutional radar.
- **Trend Health > 70%** indicates the move is stable and broad-based.
        """)


if __name__ == "__main__":
    main()
