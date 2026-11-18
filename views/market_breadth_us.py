import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import yfinance as yf
import numpy as np
from scipy.signal import find_peaks

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from breadth_engine import (
    fetch_full_breadth_history,
    compute_top_rs_breadth_history,
    get_latest_volume_and_ad_distribution,
    get_zweig_tape_summary,
    evaluate_comprehensive_breadth_checklist,
    compute_sector_breadth_matrix_us,
    evaluate_early_inflection_radar,
    render_early_inflection_radar_html,
    render_early_inflection_cheat_sheet_html,
    compute_rolling_ad_divergence,
    render_stealth_decay_banner_html
)
from macro_regime_engine import fetch_sp500_telemetry
from us_universe_manager import (
    get_sp500_constituents,
    get_nasdaq100_constituents,
    get_broad_us_universe
)
from components import render_header
from breakout_cohort_engine import render_breakout_health_radar_section
import styles

styles.load_css()

def clean_html(html_str: str) -> str:
    """Strips leading whitespace from every line so Markdown never treats it as code."""
    return "\n".join(line.strip() for line in html_str.splitlines() if line.strip())

# =============================================================================
# WORLD-CLASS TERMINAL DESIGN SYSTEM (BLOOMBERG / IBD GMI GRADE)
# =============================================================================
st.markdown(clean_html("""
<style>
/* Header Emoji Transparency Override */
.st-app-header h1 {
    -webkit-text-fill-color: initial !important;
    color: #ffffff !important;
    background: none !important;
}

/* Executive 6-Card HUD Layout */
.mb-hud-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
@media (max-width: 1400px) {
    .mb-hud-grid { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 768px) {
    .mb-hud-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 480px) {
    .mb-hud-grid { grid-template-columns: 1fr; }
}

/* Executive 4-Card Institutional Alpha HUD */
.mb-alpha-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
@media (max-width: 1200px) {
    .mb-alpha-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 600px) {
    .mb-alpha-grid { grid-template-columns: 1fr; }
}

.mb-hud-card {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.85) 0%, rgba(2, 6, 23, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(16px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.mb-hud-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.18);
}
.mb-hud-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #10b981, #06b6d4);
}
.mb-hud-card.red::before {
    background: linear-gradient(90deg, #ef4444, #f97316);
}
.mb-hud-card.amber::before {
    background: linear-gradient(90deg, #f59e0b, #eab308);
}
.mb-hud-card.cyan::before {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
}
.mb-hud-card.emerald::before {
    background: linear-gradient(90deg, #10b981, #34d399);
}
.mb-hud-card.purple::before {
    background: linear-gradient(90deg, #818cf8, #c084fc);
}

.mb-hud-title {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.mb-hud-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.22rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.mb-hud-sub {
    font-size: 0.74rem;
    color: #64748b;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Institutional Dual Cockpit Modules */
.mb-panel {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.85) 0%, rgba(2, 6, 23, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 12px 30px -5px rgba(0, 0, 0, 0.6);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(16px);
    margin-bottom: 20px;
}
.mb-panel-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    padding-bottom: 12px;
}
.mb-panel-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0;
}
.mb-panel-desc {
    font-size: 0.76rem;
    color: #94a3b8;
    margin: 3px 0 0 0;
}

/* Progress split flow bar */
.mb-flow-bar {
    height: 14px;
    border-radius: 7px;
    width: 100%;
    overflow: hidden;
    display: flex;
    background: rgba(255, 255, 255, 0.05);
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4);
    margin: 14px 0 6px 0;
}
.mb-flow-adv {
    background: linear-gradient(90deg, #059669, #10b981);
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
    transition: width 0.3s ease;
}
.mb-flow-dec {
    background: linear-gradient(90deg, #ef4444, #dc2626);
    box-shadow: 0 0 10px rgba(239, 68, 68, 0.4);
    transition: width 0.3s ease;
}

/* Methodology Grid */
.mb-guide-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-top: 10px;
}
@media (max-width: 900px) {
    .mb-guide-grid { grid-template-columns: 1fr; }
}
.mb-guide-card {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 14px;
}
.mb-guide-card h5 {
    margin: 0 0 6px 0;
    font-size: 0.88rem;
    font-weight: 700;
    color: #f1f5f9;
}
.mb-guide-card p {
    margin: 0;
    font-size: 0.80rem;
    color: #94a3b8;
    line-height: 1.45;
}

/* Automated Breadth Action Cockpit */
.mb-action-cockpit {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.95) 0%, rgba(2, 6, 23, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 16px 36px -5px rgba(0, 0, 0, 0.65);
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
}
.mb-hero-card {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 16px;
}
.mb-hero-badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border: 1px solid transparent;
}
.mb-hero-meta {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    align-items: center;
}
.mb-hero-kpi {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 8px;
    padding: 8px 12px;
    text-align: right;
}
.mb-hero-kpi-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #94a3b8;
    letter-spacing: 0.05em;
}
.mb-hero-kpi-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 800;
    color: #f8fafc;
}
.mb-directives-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 18px;
}
@media (max-width: 1200px) {
    .mb-directives-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 600px) {
    .mb-directives-grid { grid-template-columns: 1fr; }
}
.mb-directive-card {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 12px 14px;
}
.mb-directive-title {
    font-size: 0.70rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #94a3b8;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.mb-directive-text {
    font-size: 0.82rem;
    color: #f1f5f9;
    line-height: 1.45;
    font-weight: 500;
}
.mb-checklist-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 10px;
}
.mb-check-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.mb-check-card:hover {
    border-color: rgba(255, 255, 255, 0.15);
    transform: translateY(-1px);
}
.mb-check-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 8px;
    margin-bottom: 8px;
}
.mb-check-name {
    font-size: 0.78rem;
    font-weight: 700;
    color: #cbd5e1;
    display: flex;
    align-items: center;
    gap: 6px;
}
.mb-check-status-pill {
    font-size: 0.68rem;
    font-weight: 800;
    padding: 2px 7px;
    border-radius: 12px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.mb-check-body {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 4px 8px;
    margin-bottom: 6px;
}
.mb-check-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 800;
    color: #f8fafc;
    white-space: nowrap;
}
.mb-check-thresh {
    font-size: 0.70rem;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
    text-align: right;
}
.mb-check-guidance {
    font-size: 0.72rem;
    color: #94a3b8;
    line-height: 1.35;
    border-top: 1px solid rgba(255, 255, 255, 0.04);
    padding-top: 6px;
}

/* Early Inflection Radar Dual Grid & Cards */
.mb-radar-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}
@media (max-width: 900px) {
    .mb-radar-grid {
        grid-template-columns: 1fr;
    }
}
.mb-radar-col {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.mb-radar-col-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    padding-bottom: 10px;
    margin-bottom: 4px;
    flex-wrap: wrap;
    gap: 8px;
}
.mb-radar-card {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.mb-radar-card:hover {
    border-color: rgba(255, 255, 255, 0.14);
    transform: translateY(-1px);
}
.mb-radar-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
}
.mb-radar-card-name {
    font-size: 0.80rem;
    font-weight: 700;
    color: #f1f5f9;
    display: flex;
    align-items: center;
    gap: 6px;
}
.mb-radar-lead-pill {
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 6px;
    background: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
    white-space: nowrap;
}
.mb-radar-card-body {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
}
.mb-radar-reading {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 800;
    color: #f8fafc;
}
.mb-radar-status-pill {
    font-size: 0.76rem;
    font-weight: 800;
    padding: 4px 10px;
    border-radius: 12px;
    letter-spacing: 0.02em;
    white-space: nowrap;
}
.mb-radar-directive {
    font-size: 0.72rem;
    color: #94a3b8;
    line-height: 1.35;
    border-top: 1px solid rgba(255, 255, 255, 0.04);
    padding-top: 6px;
}
</style>
"""), unsafe_allow_html=True)

# =============================================================================
# INSTITUTIONAL PLOTLY VISUALIZERS (BELL CURVE & MULTI-TIMEFRAME)
# =============================================================================
def create_ad_distribution_fig(vol_dist):
    """
    Renders an institutional IBD bell-curve bar chart for Universe A-E Accumulation Grades.
    """
    grades = ['Grade A', 'Grade B', 'Grade C', 'Grade D', 'Grade E']
    counts = [
        vol_dist.get('ad_grade_a', 0),
        vol_dist.get('ad_grade_b', 0),
        vol_dist.get('ad_grade_c', 0),
        vol_dist.get('ad_grade_d', 0),
        vol_dist.get('ad_grade_e', 0)
    ]
    deltas = [
        vol_dist.get('delta_a', 0),
        vol_dist.get('delta_b', 0),
        vol_dist.get('delta_c', 0),
        vol_dist.get('delta_d', 0),
        vol_dist.get('delta_e', 0)
    ]
    colors = ['#10b981', '#34d399', '#64748b', '#f59e0b', '#ef4444']
    total = sum(counts) if sum(counts) > 0 else 1
    pcts = [(c / total) * 100 for c in counts]
    
    labels = []
    for c, d, p in zip(counts, deltas, pcts):
        d_color = "#34d399" if d > 0 else ("#f87171" if d < 0 else "#94a3b8")
        d_str = f" <span style='color:{d_color};'>({d:+d})</span>" if d != 0 else ""
        labels.append(f"<b>{c}</b>{d_str}<br><span style='font-size:10px;color:#94a3b8;'>{p:.1f}%</span>")

    descriptions = [
        "Heavy Institutional Accumulation (Strong Net Buying)",
        "Moderate Institutional Accumulation (Steady Inflows)",
        "Neutral / Equilibrium (Balanced Volume)",
        "Moderate Institutional Distribution (Selling into Rallies)",
        "Heavy Institutional Distribution (Severe Liquidation)"
    ]

    customdata = list(zip(deltas, pcts, descriptions))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grades,
        y=counts,
        text=labels,
        textposition='outside',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.15)', width=1)
        ),
        customdata=customdata,
        hovertemplate="<b>%{x}</b><br>Count: <b>%{y} stocks</b> (%{customdata[1]:.1f}%)<br>1-Day Delta: <b>%{customdata[0]:+d}</b><br><i>%{customdata[2]}</i><extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=grades,
        y=counts,
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.4)', width=2, shape='spline'),
        hoverinfo='skip'
    ))

    max_c = max(counts) if counts else 100
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        font=dict(family="JetBrains Mono, monospace", color="#94a3b8", size=10),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#cbd5e1', size=11, family="Inter, sans-serif")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.03)',
            zeroline=False,
            showticklabels=False,
            range=[0, max_c * 1.45]
        )
    )
    return fig

def create_multi_timeframe_hl_fig(multi_hl):
    """
    Renders an institutional grouped bar chart comparing New Highs vs New Lows across 1W, 1M, 3M, and 52W (1Y).
    """
    horizons = list(multi_hl.keys())
    highs = [multi_hl[h]['Highs'] for h in horizons]
    lows = [multi_hl[h]['Lows'] for h in horizons]
    nets = [multi_hl[h]['Net'] for h in horizons]

    fig = go.Figure()

    # Trace 1: New Highs (Green)
    fig.add_trace(go.Bar(
        x=horizons,
        y=highs,
        name='New Highs (within 2%)',
        marker=dict(
            color='#10b981',
            line=dict(color='rgba(255,255,255,0.15)', width=1)
        ),
        text=[f"<b>+{h}</b>" for h in highs],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>New Highs: <b>%{y} stocks</b><extra></extra>"
    ))

    # Trace 2: New Lows (Red)
    fig.add_trace(go.Bar(
        x=horizons,
        y=lows,
        name='New Lows (within 2%)',
        marker=dict(
            color='#ef4444',
            line=dict(color='rgba(255,255,255,0.15)', width=1)
        ),
        text=[f"<b>{l}</b>" for l in lows],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>New Lows: <b>%{y} stocks</b><extra></extra>"
    ))

    # Trace 3: Net (Scatter with lines & markers)
    net_colors = ['#34d399' if n >= 0 else '#f87171' for n in nets]
    fig.add_trace(go.Scatter(
        x=horizons,
        y=nets,
        mode='lines+markers+text',
        name='Net Highs - Lows',
        line=dict(color='#38bdf8', width=2.5),
        marker=dict(color=net_colors, size=10, line=dict(color='#ffffff', width=1.5)),
        text=[f"<b>Net: {n:+d}</b>" for n in nets],
        textposition='top center',
        hovertemplate="<b>%{x}</b><br>Net Highs - Lows: <b>%{y:+d}</b><extra></extra>"
    ))

    max_val = max(max(highs) if highs else 10, max(lows) if lows else 10, max(abs(n) for n in nets) if nets else 10)
    min_val = min(min(nets) if nets else -10, 0)
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        barmode='group',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=11, color='#94a3b8')
        ),
        font=dict(family="JetBrains Mono, monospace", color="#94a3b8", size=10),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#cbd5e1', size=11, family="Inter, sans-serif")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.03)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.1)',
            range=[min_val - 15, max_val * 1.35]
        )
    )
    return fig

@st.cache_data(ttl=600)
def compute_institutional_breadth_alpha_us(df_sp500_close, df_nasdaq_close):
    """
    Computes Dual-Benchmark Outperformance and Multi-Timeframe Highs/Lows for US Equities
    (or loads from Turso breadth_aux_bundle_us on Cloud).
    """
    from breadth_engine import compute_institutional_breadth_alpha_us_core
    matrix_path = 'historical_prices_matrix.pkl' if os.path.exists('historical_prices_matrix.pkl') else 'breadth_matrix.pkl'
    if not os.path.exists(matrix_path):
        try:
            from database import load_cloud_json_cache_cached
            bundle, _ = load_cloud_json_cache_cached("breadth_aux_bundle_us")
            if isinstance(bundle, dict) and bundle.get("alpha"):
                return bundle["alpha"]
        except Exception:
            pass
        return None
    return compute_institutional_breadth_alpha_us_core(df_sp500_close, df_nasdaq_close)

def detect_divergences(df_idx_close, df_ad_line):
    bearish_divs = []
    bullish_divs = []
    
    if len(df_idx_close) < 60:
        return bearish_divs, bullish_divs
        
    df_aligned = pd.concat([df_idx_close, df_ad_line], axis=1, join='inner').dropna()
    if df_aligned.empty:
        return bearish_divs, bullish_divs
        
    prices = df_aligned.iloc[:, 0].values
    ad_vals = df_aligned.iloc[:, 1].values
    dates = df_aligned.index
    
    smooth_price = pd.Series(prices).rolling(window=5, min_periods=1).mean().values
    sma_200 = pd.Series(prices).rolling(window=200, min_periods=1).mean().values
    
    # 1. Bearish Divergences
    peaks, _ = find_peaks(smooth_price, distance=20, prominence=0.03 * np.mean(smooth_price))
    for i in range(1, len(peaks)):
        idx2 = peaks[i]
        for j in range(max(0, i-3), i):
            idx1 = peaks[j]
            if prices[idx2] >= (prices[idx1] * 0.992) and ad_vals[idx2] < (ad_vals[idx1] - 100) and prices[idx2] > sma_200[idx2]:
                if (dates[idx2] - dates[idx1]).days > 25:
                    bearish_divs.append((dates[idx2], ad_vals[idx2]))
                    break
                
    # 2. Bullish Divergences
    troughs, _ = find_peaks(-smooth_price, distance=20, prominence=0.03 * np.mean(smooth_price))
    for i in range(1, len(troughs)):
        idx2 = troughs[i]
        for j in range(max(0, i-3), i):
            idx1 = troughs[j]
            if prices[idx2] <= (prices[idx1] * 1.008) and ad_vals[idx2] > (ad_vals[idx1] + 100):
                if (dates[idx2] - dates[idx1]).days > 25:
                    bullish_divs.append((dates[idx2], ad_vals[idx2]))
                    break
                
    return bearish_divs, bullish_divs

def detect_tactical_signals(df_idx_close, df_breadth):
    hindenburg_alerts = []
    froth_peaks = []
    breadth_thrusts = []
    
    if len(df_breadth) < 50:
        return hindenburg_alerts, froth_peaks, breadth_thrusts, [], []
        
    df_aligned = pd.concat([df_idx_close, df_breadth], axis=1, join='inner').dropna()
    if df_aligned.empty:
        return hindenburg_alerts, froth_peaks, breadth_thrusts, [], []
        
    idx_close = df_aligned.iloc[:, 0]
    df_b = df_aligned.iloc[:, 1:]
    dates = df_aligned.index
    
    # 1. Hindenburg Warnings
    sma_50 = idx_close.rolling(window=50, min_periods=1).mean()
    net_hl = df_b['new_highs_count'] - df_b['new_lows_count']
    
    for i in range(50, len(df_b)):
        if idx_close.iloc[i] > sma_50.iloc[i] and net_hl.iloc[i] < -25:
            if not hindenburg_alerts or (dates[i] - hindenburg_alerts[-1][0]).days > 15:
                hindenburg_alerts.append((dates[i], net_hl.iloc[i]))
                
    # 2. Froth Peaks
    froth_sma = df_b['double_52w_count'].rolling(window=20).mean()
    f_peaks, _ = find_peaks(froth_sma.dropna().values, height=120, distance=90)
    for p in f_peaks:
        peak_date = froth_sma.dropna().index[p]
        peak_val = froth_sma.dropna().iloc[p]
        froth_peaks.append((peak_date, peak_val))
        
    # 3. Breadth Thrusts
    if 'zbt_signal' in df_b.columns and df_b['zbt_signal'].sum() > 0:
        for i in range(len(df_b)):
            if df_b['zbt_signal'].iloc[i] == 1:
                end_date = dates[i]
                start_date = dates[max(0, i-10)]
                if not breadth_thrusts or (start_date - breadth_thrusts[-1][1]).days > 0:
                    breadth_thrusts.append((start_date, end_date))
    else:
        above_20 = df_b['above_20_pct']
        for i in range(15, len(above_20)):
            if above_20.iloc[i] > 60 and above_20.iloc[i-1] <= 60:
                window = above_20.iloc[i-15:i]
                if window.min() < 20:
                    start_date = window.idxmin()
                    end_date = dates[i]
                    if not breadth_thrusts or (start_date - breadth_thrusts[-1][1]).days > 0:
                        breadth_thrusts.append((start_date, end_date))

    # 4. MCO Swing Signals
    mco_alerts = []
    if 'advances' in df_b.columns and 'declines' in df_b.columns:
        total_issues = (df_b['advances'] + df_b['declines']).replace(0, 1)
        rna = ((df_b['advances'] - df_b['declines']) / total_issues) * 1000
        mco = rna.ewm(span=19, adjust=False).mean() - rna.ewm(span=39, adjust=False).mean()
        for i in range(1, len(mco)):
            if mco.iloc[i-1] < -100 and mco.iloc[i] > mco.iloc[i-1]:
                if not mco_alerts or (dates[i] - mco_alerts[-1][0]).days > 15:
                    mco_alerts.append((dates[i], mco.iloc[i], "buy"))
            elif mco.iloc[i-1] > 100 and mco.iloc[i] < mco.iloc[i-1]:
                if not mco_alerts or (dates[i] - mco_alerts[-1][0]).days > 15:
                    mco_alerts.append((dates[i], mco.iloc[i], "sell"))

    # 5. Climax Extremes
    climax_alerts = []
    if 'panic_extreme_5d' in df_b.columns and 'surge_extreme_5d' in df_b.columns:
        panic = df_b['panic_extreme_5d']
        surge = df_b['surge_extreme_5d']
        p_peaks, _ = find_peaks(panic.fillna(0).values, height=40, distance=30)
        for p in p_peaks:
            climax_alerts.append((dates[p], panic.iloc[p], "capitulation"))
        s_peaks, _ = find_peaks(surge.fillna(0).values, height=40, distance=30)
        for p in s_peaks:
            climax_alerts.append((dates[p], surge.iloc[p], "climax"))

    return hindenburg_alerts, froth_peaks, breadth_thrusts, mco_alerts, climax_alerts

def compute_canslim_status(df_idx_close, df_idx_vol, df_breadth):
    from macro_regime_engine import calculate_distribution_days
    if df_idx_close.empty or len(df_idx_close) < 25:
        return 0, "Market in Correction", "#ef4444", "0% – 10%", False
    df_temp = pd.DataFrame({'close': df_idx_close, 'volume': df_idx_vol})
    dist_days, _ = calculate_distribution_days(df_temp)
    curr_p = df_idx_close.iloc[-1]
    sma_50 = df_idx_close.rolling(window=50, min_periods=1).mean().iloc[-1]
    above_50 = curr_p > sma_50

    if dist_days >= 6 or (not above_50 and dist_days >= 4):
        status = "Market in Correction"
        color = "#ef4444"
        exposure = "0% – 20%"
    elif dist_days >= 4 or not above_50:
        status = "Uptrend Under Pressure"
        color = "#f59e0b"
        exposure = "20% – 50%"
    else:
        status = "Confirmed Uptrend"
        color = "#10b981"
        exposure = "80% – 100%"

    thrust_alert = False
    if 'above_20_pct' in df_breadth.columns and len(df_breadth) >= 20:
        last_20 = df_breadth['above_20_pct'].iloc[-20:]
        if last_20.min() < 20 and last_20.max() > 60:
            min_idx = last_20.idxmin()
            max_idx = last_20.idxmax()
            if min_idx < max_idx and (max_idx - min_idx).days <= 15:
                thrust_alert = True
    return dist_days, status, color, exposure, thrust_alert

# =============================================================================
# HEADER & EXECUTIVE UNIVERSE SCOPE SELECTOR
# =============================================================================
col_hdr, col_sync = st.columns([3.2, 0.8], vertical_alignment="center")
with col_hdr:
    render_header(
        "🗽 Deep Market Breadth (US Equities)",
        "Institutional Multi-Horizon Breadth, True Exchange Volume Flow & Accumulation Curve (IBD GMI Standard)"
    )
with col_sync:
    if st.button("🔄 Sync Today's Breadth", key="sync_us_breadth_btn", use_container_width=True, help="Fetch latest US EOD closing data and update all breadth metrics"):
        with st.spinner("Syncing US EOD Breadth Matrix..."):
            from breadth_engine import sync_breadth_history
            sync_breadth_history('US')
            st.cache_data.clear()
            st.success("✅ US breadth matrix updated!")
            st.rerun()

# -----------------------------------------------------------------------------
# EXECUTIVE UNIVERSE SCOPE SELECTOR (S&P 500 | NASDAQ 100 | BROAD US)
# -----------------------------------------------------------------------------
scope_col1, scope_col2 = st.columns([2.5, 1.5], vertical_alignment="center")
with scope_col1:
    selected_scope = st.segmented_control(
        "Market Scope / Universe Filter:",
        [
            "🏛️ S&P 500 (Large-Cap Core)",
            "⚡ Nasdaq 100 (Growth Tech)",
            "🌐 Broad US Universe (1,300+ Equities)"
        ],
        default="🏛️ S&P 500 (Large-Cap Core)",
        key="us_breadth_scope"
    )
with scope_col2:
    if "S&P 500" in selected_scope:
        st.caption("🏛️ **S&P 500 Mode**: Pure institutional breadth (503 constituents via State Street SPY). Eliminates low-cap noise.")
    elif "Nasdaq 100" in selected_scope:
        st.caption("⚡ **Nasdaq 100 Mode**: Growth & tech momentum leaders (101 constituents via official Nasdaq API).")
    else:
        st.caption("🌐 **Broad Mode**: Total liquid US market (1,350+ equities). Statistically essential for Zweig Thrusts & Froth.")

# =============================================================================
# DATA INGESTION & US MARKET SESSION STATUS
# =============================================================================
with st.spinner("Loading US Breadth History..."):
    df_raw = fetch_full_breadth_history('US', 1500)

if df_raw.empty:
    st.warning("No US breadth history found in database. Please click '🔄 Sync Today's Breadth' above to build history.")
    st.stop()

# Check US Market Session Timing (Eastern Time)
try:
    from zoneinfo import ZoneInfo
    from datetime import datetime as dt_now, time
    est = ZoneInfo('America/New_York')
    now_est = dt_now.now(est)
    is_weekday = now_est.weekday() < 5
    t_est = now_est.time()
    is_market_hours = is_weekday and (time(9, 30) <= t_est <= time(16, 0))
    latest_session_str = df_raw.index.max()
    if hasattr(latest_session_str, 'strftime'):
        latest_session_str = latest_session_str.strftime('%Y-%m-%d')
    today_est_str = now_est.strftime('%Y-%m-%d')

    if is_market_hours:
        st.info(
            f"🟡 **LIVE US MARKET SESSION IN PROGRESS (NYSE / NASDAQ)** • Showing finalized EOD session: **{latest_session_str}**. "
            f"Official exchange volume flow, A-E curves, and closing MCO finalize post-market after 4:00 PM EST. "
            f"For live tick-by-tick monitoring during market hours, see **📡 Intraday Monitor (US)**."
        )
    elif is_weekday and today_est_str != latest_session_str and now_est.hour >= 16:
        st.warning(
            f"🔔 **POST-MARKET SYNC AVAILABLE**: Today's US session (**{today_est_str}**) has concluded, but current charts display **{latest_session_str}**. "
            f"Click **'🔄 Sync Today's Breadth'** above to compute official closing metrics."
        )
except Exception:
    pass

@st.cache_data(ttl=300)
def load_cached_top_rs_breadth_us(cache_buster=4):
    return compute_top_rs_breadth_history('US', 1000)

with st.spinner("Calculating Top 100 US RS Stock Breadth..."):
    df_top_rs = load_cached_top_rs_breadth_us(cache_buster=4)
    if df_top_rs is not None and not df_top_rs.empty:
        df_top_rs.index = pd.to_datetime(df_top_rs.index)
        if hasattr(df_top_rs.index, 'tz') and df_top_rs.index.tz is not None:
            df_top_rs.index = df_top_rs.index.tz_localize(None)

# Convert index to datetime for Plotly
df = df_raw.copy()
df.index = pd.to_datetime(df.index)
if hasattr(df.index, 'tz') and df.index.tz is not None:
    df.index = df.index.tz_localize(None)

# -----------------------------------------------------------------------------
# DYNAMIC UNIVERSE SLICING FOR CHOSEN SCOPE (S&P 500 / NASDAQ 100 / BROAD)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def slice_custom_scope_breadth(scope_name: str):
    """Slices breadth statistics from breadth_matrix.pkl (or Turso breadth_aux_bundle_us on Cloud)."""
    from breadth_engine import slice_custom_scope_breadth_us_core
    matrix_path = 'breadth_matrix.pkl'
    if not os.path.exists(matrix_path):
        try:
            from database import load_cloud_json_cache_cached
            bundle, _ = load_cloud_json_cache_cached("breadth_aux_bundle_us")
            scopes = bundle.get("us_scopes") if isinstance(bundle, dict) else None
            if isinstance(scopes, dict):
                if "S&P 500" in scope_name and isinstance(scopes.get("sp500"), pd.DataFrame):
                    return scopes["sp500"].copy()
                if "Nasdaq 100" in scope_name and isinstance(scopes.get("ndx100"), pd.DataFrame):
                    return scopes["ndx100"].copy()
        except Exception:
            pass
        return None
    return slice_custom_scope_breadth_us_core(scope_name)

sliced_df = slice_custom_scope_breadth(selected_scope)
if sliced_df is not None and not sliced_df.empty:
    df_active = df.copy()
    overlap_idx = df_active.index.intersection(sliced_df.index)
    df_active.loc[overlap_idx, 'above_20_pct'] = sliced_df.loc[overlap_idx, 'above_20_pct']
    df_active.loc[overlap_idx, 'above_50_pct'] = sliced_df.loc[overlap_idx, 'above_50_pct']
    df_active.loc[overlap_idx, 'above_200_pct'] = sliced_df.loc[overlap_idx, 'above_200_pct']
    df_active.loc[overlap_idx, 'net_new_highs'] = sliced_df.loc[overlap_idx, 'net_new_highs']
    df_active.loc[overlap_idx, 'new_highs_count'] = sliced_df.loc[overlap_idx, 'new_highs_count']
    df_active.loc[overlap_idx, 'new_lows_count'] = sliced_df.loc[overlap_idx, 'new_lows_count']
    df_active.loc[overlap_idx, 'ad_line_cumulative'] = sliced_df.loc[overlap_idx, 'ad_line_cumulative']
else:
    df_active = df.copy()

with st.spinner("Fetching US Benchmark Overlays & Telemetry..."):
    if df.empty or pd.isna(df.index.min()):
        st.warning("US breadth database has no rows yet; run Sync US Breadth.")
        st.stop()
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = (df.index.max() + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    df_idx = yf.download(['^GSPC', '^IXIC'], start=start_date, end=end_date, progress=False)
    
    # 1. Authoritative S&P 500 Benchmark Telemetry
    sp500_telem = fetch_sp500_telemetry()

    if not df_idx.empty:
        if hasattr(df_idx.index, 'tz') and df_idx.index.tz is not None:
            df_idx.index = df_idx.index.tz_localize(None)
        df_sp500_close = df_idx['Close']['^GSPC'] if isinstance(df_idx.columns, pd.MultiIndex) else df_idx['Close']
        df_nasdaq_close = df_idx['Close']['^IXIC'] if isinstance(df_idx.columns, pd.MultiIndex) else df_idx['Close']
        df_sp500_vol = df_idx['Volume']['^GSPC'] if isinstance(df_idx.columns, pd.MultiIndex) else df_idx['Volume']
        
        _, _, _, _, thrust_alert = compute_canslim_status(df_sp500_close, df_sp500_vol, df_active)
        bearish_divs, bullish_divs = detect_divergences(df_sp500_close, df_active['ad_line_cumulative'])
        rolling_ad_div = compute_rolling_ad_divergence(df_sp500_close, df_active['ad_line_cumulative'], lookback=20)
        hindenburg_alerts, froth_peaks, breadth_thrusts, mco_alerts, climax_alerts = detect_tactical_signals(df_sp500_close, df)
    else:
        df_sp500_close = pd.Series(index=df.index, data=0)
        df_nasdaq_close = pd.Series(index=df.index, data=0)
        thrust_alert = False
        bearish_divs, bullish_divs = [], []
        rolling_ad_div = compute_rolling_ad_divergence(None, None)
        hindenburg_alerts, froth_peaks, breadth_thrusts, mco_alerts, climax_alerts = [], [], [], [], []

    if sp500_telem:
        dist_days = sp500_telem.get('dd_count', 0)
        std_dd = sp500_telem.get('standard_dd_count', 0)
        churn_dd = sp500_telem.get('churning_dd_count', 0)
        hammer_dd = sp500_telem.get('hammer_acc_count', 0)
        mkt_status = sp500_telem.get('status_label', 'Market in Correction')
        exp_plan = sp500_telem.get('exposure_plan', {})
        exposure = exp_plan.get('exposure_range', '0% – 10%')
        exp_rule = exp_plan.get('rule', 'Capital preservation mode.')
        cluster_detected = sp500_telem.get('cluster_detected', False)
        cluster_count_7d = sp500_telem.get('cluster_count_7d', 0)
    else:
        dist_days, mkt_status, _, exposure, thrust_alert = compute_canslim_status(df_sp500_close, df_sp500_vol, df_active)
        std_dd, churn_dd, hammer_dd = dist_days, 0, 0
        exp_rule = 'Capital Preservation'
        cluster_detected = False
        cluster_count_7d = 0

vol_dist = get_latest_volume_and_ad_distribution('US')
with st.spinner("Calculating Institutional Breadth Alpha & Multi-Horizon Telemetry..."):
    breadth_alpha = compute_institutional_breadth_alpha_us(df_sp500_close, df_nasdaq_close)

status_color = '#ef4444' if 'Correction' in mkt_status else ('#f59e0b' if 'Pressure' in mkt_status else '#10b981')
latest_top_rs_21 = df_top_rs['top100_above_21_ema'].iloc[-1] if (df_top_rs is not None and not df_top_rs.empty) else 0.0
latest_top_rs_50 = df_top_rs['top100_above_50_sma'].iloc[-1] if (df_top_rs is not None and not df_top_rs.empty) else 0.0
c21_color = '#10b981' if latest_top_rs_21 >= 70 else '#f59e0b' if latest_top_rs_21 >= 50 else '#ef4444'
c50_color = '#10b981' if latest_top_rs_50 >= 70 else '#f59e0b' if latest_top_rs_50 >= 50 else '#ef4444'
dist_color = '#ef4444' if dist_days >= 5 else '#10b981'

# Fetch Dr. Martin Zweig Breadth Thrust & Tape Telemetry
zweig_tape = get_zweig_tape_summary('US')
zbt_val = float(zweig_tape.get('zbt_10_ema', 0.50))
zbt_active = bool(zweig_tape.get('zbt_active', False))
zbt_signal = bool(zweig_tape.get('zbt_signal', False))
double_91_active = bool(zweig_tape.get('double_9_to_1_active', False))
count_91 = int(zweig_tape.get('count_9_to_1_65d', 0))
days_since_zbt = zweig_tape.get('days_since_zbt')

# Macro Alert / ZBT Card Display
if cluster_detected:
    alert_card_class = "red"
    alert_title_html = '<span style="color: #ef4444; font-size: 1.05rem; font-weight: 800;">🚨 ACUTE CLUSTER</span>'
    alert_sub_html = f'<span style="color: #f87171; font-weight: 700;">{cluster_count_7d} DDs · ZBT: {zbt_val:.3f}</span>'
elif thrust_alert or zbt_active:
    alert_card_class = "emerald"
    alert_title_html = '<span style="color: #10b981; font-size: 1.05rem; font-weight: 800;">ZBT ACTIVE 🔥</span>'
    alert_sub_html = f'<span>ZBT 10-EMA: {zbt_val:.3f} (&ge;0.615)</span>'
else:
    alert_card_class = "cyan" if zbt_val < 0.40 else "purple"
    alert_title_html = f'<span style="color: {"#38bdf8" if zbt_val < 0.40 else "#f8fafc"}; font-size: 1.05rem; font-weight: 800;">ZBT: {zbt_val:.3f}</span>'
    alert_sub_html = f'<span>{"🟡 Coiling (<0.40)" if zbt_val < 0.40 else "Target: ≥0.615 in 10D"}</span>'

# =============================================================================
# Delta 5: Cumulative Advance/Decline (A/D) Stealth Decay Top-Level Urgent Banner
stealth_banner_html = render_stealth_decay_banner_html(rolling_ad_div)
if stealth_banner_html:
    st.markdown(clean_html(stealth_banner_html), unsafe_allow_html=True)

# 1. CANSLIM COMMAND CENTER HUD (EXECUTIVE 6-CARD GRID - 1:1 WITH ORIGINAL PAGE)
# =============================================================================
hud_html = f"""
<div class="mb-hud-grid">
    <div class="mb-hud-card {'red' if 'Correction' in mkt_status else ('amber' if 'Pressure' in mkt_status else 'emerald')}">
        <div class="mb-hud-title">🏛️ Market Regime</div>
        <div class="mb-hud-val" style="color: {status_color}; font-size: 1.10rem;">{mkt_status}</div>
        <div class="mb-hud-sub">
            <span>Action: <b style="color:{status_color};">{'Cash / Defensive' if 'Correction' in mkt_status else ('Selective Pilots' if 'Pressure' in mkt_status else 'Active Execution')}</b></span>
        </div>
    </div>
    <div class="mb-hud-card {'red' if dist_days >= 5 else 'emerald'}">
        <div class="mb-hud-title">🛑 Distribution Days (S&P 500)</div>
        <div class="mb-hud-val" style="color: {dist_color};">{dist_days} <span style="font-size: 0.85rem; font-weight: 500; color: #94a3b8;">/ 25 Sessions</span></div>
        <div class="mb-hud-sub">
            <span>{'⚠️' if dist_days >= 5 else '🟢'} {std_dd} Std · {churn_dd} Churn · {hammer_dd} Hammer</span>
        </div>
    </div>
    <div class="mb-hud-card {'red' if '0%' in exposure else ('amber' if '25%' in exposure else 'emerald')}">
        <div class="mb-hud-title">🎯 Target Exposure</div>
        <div class="mb-hud-val" style="color: {status_color}; font-size: 1.10rem;">{exposure}</div>
        <div class="mb-hud-sub">
            <span>{'Capital Preservation' if '0%' in exposure else 'Disciplined Pilots'}</span>
        </div>
    </div>
    <div class="mb-hud-card {alert_card_class}">
        <div class="mb-hud-title">⚡ Zweig Thrust (ZBT)</div>
        <div class="mb-hud-val">{alert_title_html}</div>
        <div class="mb-hud-sub">{alert_sub_html}</div>
    </div>
    <div class="mb-hud-card cyan">
        <div class="mb-hud-title">👑 Top 100 RS > 21 EMA</div>
        <div class="mb-hud-val" style="color: {c21_color};">{latest_top_rs_21:.1f}%</div>
        <div class="mb-hud-sub">
            <span>{'🔥 Super-Leader Strength' if latest_top_rs_21 >= 70 else ('⚠️ Pullback Phase' if latest_top_rs_21 >= 50 else '🔴 Breakdown')}</span>
        </div>
    </div>
    <div class="mb-hud-card emerald">
        <div class="mb-hud-title">👑 Top 100 RS > 50 SMA</div>
        <div class="mb-hud-val" style="color: {c50_color};">{latest_top_rs_50:.1f}%</div>
        <div class="mb-hud-sub">
            <span>{'🛡️ Structural Base Safe' if latest_top_rs_50 >= 70 else '⚠️ Violating 50 SMA'}</span>
        </div>
    </div>
</div>
"""
st.markdown(clean_html(hud_html), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DR. MARTIN ZWEIG BREADTH THRUST & 9:1 TAPE TELEMETRY STRIP
# -----------------------------------------------------------------------------
if zbt_active:
    zbt_status_text = f"🟢 ACTIVE THRUST ({days_since_zbt}d elapsed)" if days_since_zbt is not None else "🟢 ACTIVE THRUST"
    zbt_status_color = "#10b981"
elif zbt_val < 0.40:
    zbt_status_text = "🟡 Oversold Coiling (<0.400)"
    zbt_status_color = "#f59e0b"
elif zbt_val >= 0.55:
    zbt_status_text = "🟢 Bullish Accumulation"
    zbt_status_color = "#34d399"
else:
    zbt_status_text = "⚪ Neutral Tape (No Thrust)"
    zbt_status_color = "#94a3b8"

d91_status_text = "🚀 Double 9:1 Blast-Off Active" if double_91_active else f"{count_91} / 2 Days (65D Window)"
d91_color = "#ec4899" if double_91_active else ("#38bdf8" if count_91 > 0 else "#94a3b8")
tape_alloc = "🚀 100% Super-Thrust Deck" if (double_91_active or zbt_active) else "CANSLIM Standard Exposure"
tape_alloc_color = "#ec4899" if (double_91_active or zbt_active) else "#94a3b8"

zweig_strip_html = f"""
<div style="background: linear-gradient(145deg, rgba(15, 23, 42, 0.85) 0%, rgba(2, 6, 23, 0.95) 100%); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 14px 18px; margin-bottom: 20px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 8px; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.15rem;">⚡</span>
            <strong style="color: #f8fafc; font-size: 0.95rem; letter-spacing: 0.3px;">Dr. Martin Zweig Breadth Thrust (ZBT) & 9:1 Volume Engine</strong>
            <span style="color: #64748b; font-size: 0.78rem;">• Winning on Wall Street Quantitative Model (US Equities)</span>
        </div>
        <div style="font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; color: #94a3b8;">
            Rule: Advance Ratio 10-EMA surges from <b style="color: #f59e0b;">&lt;0.400</b> to <b style="color: #10b981;">&ge;0.615</b> in &le;10 Days
        </div>
    </div>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 14px;">
            <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; margin-bottom: 3px;">ZBT 10-Day EMA</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 800; color: {zbt_status_color};">{zbt_val:.3f}</div>
            <div style="font-size: 0.72rem; color: #64748b; margin-top: 2px;">Advance Ratio: Adv / (Adv + Dec)</div>
        </div>
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 14px;">
            <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; margin-bottom: 3px;">ZBT Signal Status</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 800; color: {zbt_status_color};">{zbt_status_text}</div>
            <div style="font-size: 0.72rem; color: #64748b; margin-top: 2px;">Rare institutional buying stampede edge</div>
        </div>
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 14px;">
            <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; margin-bottom: 3px;">Zweig 9-to-1 Up-Volume</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 800; color: {d91_color};">{d91_status_text}</div>
            <div style="font-size: 0.72rem; color: #64748b; margin-top: 2px;">Up-Vol &ge;90% of total volume</div>
        </div>
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 14px;">
            <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; margin-bottom: 3px;">Zweig Tape Exposure</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 800; color: {tape_alloc_color};">{tape_alloc}</div>
            <div style="font-size: 0.72rem; color: #64748b; margin-top: 2px;">Historical +30.2% 1-yr forward return</div>
        </div>
    </div>
</div>
"""
st.markdown(clean_html(zweig_strip_html), unsafe_allow_html=True)

# Acute Distribution Cluster Alert (Ross Haber Rule)
if cluster_detected:
    st.markdown(clean_html(f"""
    <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.4); border-left: 5px solid #ef4444; border-radius: 8px; padding: 12px 18px; margin-top: 14px; margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 1.4rem;">🚨</span>
                <div>
                    <strong style="color: #f87171; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.5px;">Acute Distribution Cluster Alert: {cluster_count_7d} Distribution Days in Last 7 Sessions (S&P 500)</strong>
                    <p style="margin: 2px 0 0 0; color: #cbd5e1; font-size: 0.82rem;">
                        <b>Ross Haber Rule:</b> Multiple distribution/churning days tightly compressed within 5–7 sessions indicate heavy institutional unloading. Halt new pilot positions; enforce strict defensive trailing stops.
                    </p>
                </div>
            </div>
            <span style="background: rgba(239, 68, 68, 0.25); color: #fca5a5; border: 1px solid #ef4444; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 800; text-transform: uppercase;">DEFENSIVE LOCKDOWN</span>
        </div>
    </div>
    """), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INSTITUTIONAL BREADTH ALPHA & BENCHMARK PARTICIPATION HUD (4-CARD GRID)
# -----------------------------------------------------------------------------
if breadth_alpha:
    p_sp_1m = breadth_alpha['pct_beat_sp_1m']
    p_sp_3m = breadth_alpha['pct_beat_sp_3m']
    ret_sp_1m = breadth_alpha['ret_sp500_1m']

    p_ndx_1m = breadth_alpha['pct_beat_ndx_1m']
    p_ndx_3m = breadth_alpha['pct_beat_ndx_3m']
    ret_ndx_1m = breadth_alpha['ret_ndx_1m']

    sp_spx = breadth_alpha['spread_sp_1m']
    sp_ndx = breadth_alpha['spread_ndx_1m']

    c_sp = '#10b981' if p_sp_1m >= 50 else ('#f59e0b' if p_sp_1m >= 45 else '#ef4444')
    c_ndx = '#10b981' if p_ndx_1m >= 50 else ('#f59e0b' if p_ndx_1m >= 45 else '#ef4444')
    c_spread = '#10b981' if sp_spx >= 0.5 else ('#f59e0b' if sp_spx >= -0.5 else '#ef4444')

    card1_cls = 'emerald' if p_sp_1m >= 50 else ('amber' if p_sp_1m >= 45 else 'red')
    card2_cls = 'emerald' if p_ndx_1m >= 50 else ('amber' if p_ndx_1m >= 45 else 'red')
    card3_cls = 'emerald' if sp_spx >= 0.5 else ('amber' if sp_spx >= -0.5 else 'red')

    if vol_dist:
        ud_ratio = vol_dist.get('up_down_volume_ratio', 1.0)
        up_v = vol_dist.get('up_volume', 0.0)
        down_v = vol_dist.get('down_volume', 0.0)
        tot_v = up_v + down_v if (up_v + down_v) > 0 else 1.0
        up_pct = (up_v / tot_v) * 100.0
        down_pct = (down_v / tot_v) * 100.0
        up_v_fmt = f"{up_v / 1e9:.2f}B" if up_v >= 1e9 else f"{up_v / 1e6:.0f}M"
        down_v_fmt = f"{down_v / 1e9:.2f}B" if down_v >= 1e9 else f"{down_v / 1e6:.0f}M"
    else:
        ud_ratio = 1.0
        up_pct, down_pct = 50.0, 50.0
        up_v_fmt, down_v_fmt = "0.0B", "0.0B"

    if ud_ratio >= 1.5:
        ud_card_cls = 'emerald'
        ud_badge = "🟢 Heavy Accumulation"
        ud_c = "#34d399"
    elif ud_ratio >= 1.0:
        ud_card_cls = 'cyan'
        ud_badge = "🟢 Net Accumulation"
        ud_c = "#10b981"
    elif ud_ratio >= 0.8:
        ud_card_cls = 'amber'
        ud_badge = "🟡 Neutral Volume"
        ud_c = "#f59e0b"
    else:
        ud_card_cls = 'red'
        ud_badge = "🔴 Distribution Flow"
        ud_c = "#ef4444"

    alpha_hud_html = f"""
    <div style="margin-top: 10px; margin-bottom: 8px; font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; display: flex; align-items: center; gap: 8px;">
        <span>📊 Institutional Breadth Alpha & Benchmark Participation (TradingView Standard)</span>
        <div style="flex: 1; height: 1px; background: rgba(255,255,255,0.08);"></div>
    </div>
    <div class="mb-alpha-grid">
        <div class="mb-hud-card {card1_cls}">
            <div class="mb-hud-title">🏛️ Broad Participation (vs S&P 500)</div>
            <div class="mb-hud-val" style="color: {c_sp};">{p_sp_1m:.1f}% <span style="font-size: 0.76rem; font-weight: 500; color: #94a3b8;">(1M)</span></div>
            <div class="mb-hud-sub">
                <span>3M: <b style="color:#f8fafc;">{p_sp_3m:.1f}%</b> · S&P 500 1M: <b style="color:{'#34d399' if ret_sp_1m>=0 else '#f87171'};">{ret_sp_1m:+.1f}%</b></span>
            </div>
            <div style="margin-top: 6px; font-size: 0.70rem; color: {c_sp}; font-weight: 700;">
                {'🟢 Broad Expansion (>50%)' if p_sp_1m >= 50 else ('🟡 Selective Picking' if p_sp_1m >= 45 else '🔴 Heavyweight Drag (<45%)')}
            </div>
        </div>

        <div class="mb-hud-card {card2_cls}">
            <div class="mb-hud-title">⚡ Growth Alpha (vs Nasdaq 100)</div>
            <div class="mb-hud-val" style="color: {c_ndx};">{p_ndx_1m:.1f}% <span style="font-size: 0.76rem; font-weight: 500; color: #94a3b8;">(1M)</span></div>
            <div class="mb-hud-sub">
                <span>3M: <b style="color:#f8fafc;">{p_ndx_3m:.1f}%</b> · Nasdaq 1M: <b style="color:{'#34d399' if ret_ndx_1m>=0 else '#f87171'};">{ret_ndx_1m:+.1f}%</b></span>
            </div>
            <div style="margin-top: 6px; font-size: 0.70rem; color: {c_ndx}; font-weight: 700;">
                {'🟢 Momentum Beating Tech Giants' if p_ndx_1m >= 50 else '🔴 Tech Mega-Cap Domination'}
            </div>
        </div>

        <div class="mb-hud-card {card3_cls}">
            <div class="mb-hud-title">⚖️ Equal-Weight Spread (1M)</div>
            <div class="mb-hud-val" style="color: {c_spread};">{sp_spx:+.2f}% <span style="font-size: 0.72rem; font-weight: 500; color: #94a3b8;">(vs S&P 500)</span></div>
            <div class="mb-hud-sub">
                <span>NDX EW Spread: <b style="color:#f8fafc;">{sp_ndx:+.2f}%</b></span>
            </div>
            <div style="margin-top: 6px; font-size: 0.70rem; color: {c_spread}; font-weight: 700;">
                {'🟢 Broad Participation (>+0.5%)' if sp_spx >= 0.5 else ('🔴 Mega-Cap Divergence (<-0.5%)' if sp_spx < -0.5 else '🟡 Neutral Equal-Weight Spread')}
            </div>
        </div>

        <div class="mb-hud-card {ud_card_cls}">
            <div class="mb-hud-title">🌊 Up / Down Volume Ratio</div>
            <div class="mb-hud-val" style="color: {ud_c};">{ud_ratio:.2f}x</div>
            <div class="mb-hud-sub">
                <span>Adv: <b style="color:#34d399;">{up_pct:.0f}%</b> ({up_v_fmt}) · Dec: <b style="color:#f87171;">{down_pct:.0f}%</b> ({down_v_fmt})</span>
            </div>
            <div style="margin-top: 6px; font-size: 0.70rem; color: {ud_c}; font-weight: 700;">
                {ud_badge}
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(alpha_hud_html), unsafe_allow_html=True)

# =============================================================================
# 2. INSTITUTIONAL METHODOLOGY GUIDE (STRUCTURED 6-CARD EXPANDER)
# =============================================================================
with st.expander("📖 Institutional Methodology & Breadth Calculation Engine (US Edition)", expanded=False):
    guide_html = """
    <div class="mb-guide-grid">
        <div class="mb-guide-card">
            <h5>🚦 Market Regimes & Exposure Rules</h5>
            <p>
                • <b>Confirmed Uptrend</b>: S&P 500 trades above its 50 SMA and Distribution Days &lt; 5. Recommended exposure: <b>80%–100%</b>.<br>
                • <b>Uptrend Under Pressure</b>: Distribution Days &ge; 5 or moving averages unaligned. Recommended exposure: <b>25%–50%</b>.<br>
                • <b>Market in Correction</b>: S&P 500 breaks below 50 SMA. Recommended exposure: <b>0%–20% (Cash is an Active Position)</b>.
            </p>
        </div>
        <div class="mb-guide-card">
            <h5>⚖️ True Exchange Volume Breadth & Smart Money Ratio</h5>
            <p>
                • <b>Volume Ratio</b>: Aggregates total shares traded in advancing vs declining equities. Ratio &gt; 1.5x confirms institutional accumulation; &lt; 0.8x indicates systematic supply liquidation.<br>
                • <b>Smart Money Pressure Ratio</b>: <code>(Grade A + B) / (Grade D + E)</code>. Measures institutional balance across ~1,350 stocks. Values &gt; 1.2x confirm broad institutional demand.
            </p>
        </div>
        <div class="mb-guide-card">
            <h5>👑 Top 100 US RS Super-Leader Breadth</h5>
            <p>
                • Tracks the percentage of the 100 highest Relative Strength market leaders trading above their 21 EMA and 50 SMA.<br>
                • <b>Bullish Ignition (&gt;70%)</b>: When leaders hold their 21 EMA while the index corrects, fresh breakout ignition is imminent.<br>
                • <b>Distribution Warning (&lt;50%)</b>: Leaders breaking below 50 SMA while the index trades near highs is the earliest warning of smart money distribution.
            </p>
        </div>
        <div class="mb-guide-card">
            <h5>⚠️ Tactical Divergences & Macro Warnings</h5>
            <p>
                • <b>Bearish Divergence (▼)</b>: S&P 500 sets a new macro high while the cumulative Net A/D line makes a lower high.<br>
                • <b>Bullish Divergence (▲)</b>: S&P 500 sets a lower low while the A/D line holds a higher low (stealth accumulation).<br>
                • <b>Hindenburg Warning (!)</b>: Index in uptrend, but Net Highs/Lows drops deeply negative (&lt;-25).<br>
                • <b>McClellan Oscillator (MCO)</b>: Ratio-adjusted momentum flagging overbought (&gt;+100) and oversold (&lt;-100) swing turns.
            </p>
        </div>
        <div class="mb-guide-card">
            <h5>📊 Dual-Benchmark Outperformance & Equal-Weight Alpha</h5>
            <p>
                • <b>Broad Participation (vs S&P 500)</b>: Measures % of the US universe generating positive alpha over the broad market. Values &gt; 50% confirm broad-based participation.<br>
                • <b>Growth Alpha (vs Nasdaq 100)</b>: Compares active momentum against tech giants (NVDA, MSFT, AAPL, AMZN, META).<br>
                • <b>Equal-Weight Spread (1M)</b>: <code>(US Equal-Weight 1M) - (S&P 500 1M)</code>. When positive (&gt;+0.5%), average stocks are beating mega-caps (Stock Picker's Market).
            </p>
        </div>
        <div class="mb-guide-card">
            <h5>⚡ Dr. Martin Zweig Breadth Thrust (ZBT) & 9:1 Volume Model</h5>
            <p>
                • <b>Dr. Martin Zweig Breadth Thrust (ZBT)</b>: A rare, explosive quantitative thrust where the 10-day EMA of <code>Advances / (Advances + Declines)</code> surges from deeply oversold (&lt;0.400) to buying stampede (&ge;0.615) within &le;10 consecutive trading sessions. In 70 years of US equity history, an active ZBT has never failed to produce positive 12-month forward returns.<br>
                • <b>9-to-1 Up-Volume Days</b>: Days where advancing volume accounts for &ge;90% of total volume.<br>
                • <b>Double 9:1 Blast-Off</b>: Two 9-to-1 up-volume days occurring within a 65 trading-day window triggers maximum 100% portfolio allocation with a historical +30.2% 1-year forward return edge.
            </p>
        </div>
    </div>
    """
    st.markdown(clean_html(guide_html), unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# =============================================================================
# 3. DUAL INSTITUTIONAL COCKPIT MODULES (VOLUME BREADTH & ACCUMULATION CURVE)
# =============================================================================
if vol_dist:
    v_col1, v_col2 = st.columns([1, 1])

    # -------------------------------------------------------------
    # MODULE A: EXCHANGE VOLUME BREADTH (LEFT)
    # -------------------------------------------------------------
    with v_col1:
        up_v = vol_dist.get('up_volume', 0.0)
        down_v = vol_dist.get('down_volume', 0.0)
        tot_v = up_v + down_v if (up_v + down_v) > 0 else 1.0
        up_pct = (up_v / tot_v) * 100.0
        down_pct = (down_v / tot_v) * 100.0
        net_flow_b = (up_v - down_v) / 1e9

        if up_v >= 1e9 or down_v >= 1e9:
            up_str = f"{up_v / 1e9:.2f}B"
            down_str = f"{down_v / 1e9:.2f}B"
            vol_unit = "Billions of Shares"
        else:
            up_str = f"{up_v / 1e6:.0f}M"
            down_str = f"{down_v / 1e6:.0f}M"
            vol_unit = "Millions of Shares"

        ud_ratio = vol_dist.get('up_down_volume_ratio', 1.0)
        if ud_ratio >= 1.5:
            ratio_label = "🟢 Heavy Accumulation"
            ratio_pill_style = "background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.6);"
            ud_color = "#34d399"
        elif ud_ratio >= 1.0:
            ratio_label = "🟢 Net Accumulation"
            ratio_pill_style = "background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4);"
            ud_color = "#10b981"
        elif ud_ratio >= 0.8:
            ratio_label = "🟡 Neutral Volume"
            ratio_pill_style = "background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.4);"
            ud_color = "#f59e0b"
        else:
            ratio_label = "🔴 Heavy Distribution Pressure"
            ratio_pill_style = "background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.6);"
            ud_color = "#ef4444"

        vol_panel_html = f"""
        <div class="mb-panel">
            <div class="mb-panel-header">
                <div>
                    <h4 class="mb-panel-title">⚖️ Exchange Volume Breadth (Advancers vs Decliners)</h4>
                    <p class="mb-panel-desc">Session: <b>{vol_dist.get('date', 'Latest')}</b> • Pure institutional volume flow across US equities ({vol_unit})</p>
                </div>
                <span style="{ratio_pill_style}; padding: 3px 10px; border-radius: 6px; font-size: 0.76rem; font-weight: 800; letter-spacing: 0.02em;">
                    {ratio_label}
                </span>
            </div>

            <!-- 3 High Density Metric Pillars -->
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 12px;">
                <div style="background: rgba(16, 185, 129, 0.06); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 10px; padding: 12px;">
                    <div style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; color: #34d399; margin-bottom: 4px;">Up-Volume (Advancers)</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.45rem; font-weight: 800; color: #f8fafc;">{up_str}</div>
                    <div style="font-size: 0.75rem; color: #34d399; font-weight: 600; margin-top: 3px;">
                        🟢 <b>{vol_dist.get('advances', 0)}</b> Stocks ({up_pct:.1f}%)
                    </div>
                </div>
                
                <div style="background: rgba(239, 68, 68, 0.06); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 10px; padding: 12px;">
                    <div style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; color: #f87171; margin-bottom: 4px;">Down-Volume (Decliners)</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.45rem; font-weight: 800; color: #f8fafc;">{down_str}</div>
                    <div style="font-size: 0.75rem; color: #f87171; font-weight: 600; margin-top: 3px;">
                        🔴 <b>{vol_dist.get('declines', 0)}</b> Stocks ({down_pct:.1f}%)
                    </div>
                </div>
                
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 12px;">
                    <div style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;">Up / Down Vol Ratio</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.45rem; font-weight: 800; color: {ud_color};">{ud_ratio:.2f}x</div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 3px;">
                        Net Flow: <b style="color: {'#34d399' if net_flow_b >= 0 else '#f87171'}; font-family: 'JetBrains Mono';">{net_flow_b:+.2f}B</b>
                    </div>
                </div>
            </div>

            <!-- Flow Bar Visualizer -->
            <div class="mb-flow-bar">
                <div class="mb-flow-adv" style="width: {up_pct:.1f}%;" title="Advancing Volume: {up_pct:.1f}%"></div>
                <div class="mb-flow-dec" style="width: {down_pct:.1f}%;" title="Declining Volume: {down_pct:.1f}%"></div>
            </div>
            
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">
                <span>🟢 Advancing Vol: <b>{up_pct:.1f}%</b> ({up_str})</span>
                <span>🔴 Declining Vol: <b>{down_pct:.1f}%</b> ({down_str})</span>
            </div>
        </div>
        """
        st.markdown(clean_html(vol_panel_html), unsafe_allow_html=True)

    # -------------------------------------------------------------
    # MODULE B: UNIVERSE ACCUMULATION VS DISTRIBUTION CURVE (RIGHT)
    # -------------------------------------------------------------
    with v_col2:
        sm_ratio = vol_dist.get('smart_money_ratio', 1.0)
        sm_color = "#34d399" if sm_ratio >= 1.2 else ("#f59e0b" if sm_ratio >= 0.8 else "#ef4444")
        
        a_cnt, b_cnt = vol_dist.get('ad_grade_a', 0), vol_dist.get('ad_grade_b', 0)
        c_cnt = vol_dist.get('ad_grade_c', 0)
        d_cnt, e_cnt = vol_dist.get('ad_grade_d', 0), vol_dist.get('ad_grade_e', 0)
        
        tot_graded = a_cnt + b_cnt + c_cnt + d_cnt + e_cnt
        if tot_graded == 0: tot_graded = 1
        acc_count = a_cnt + b_cnt
        dist_count = d_cnt + e_cnt
        acc_pct = (acc_count / tot_graded) * 100
        c_pct = (c_cnt / tot_graded) * 100
        dist_pct = (dist_count / tot_graded) * 100

        curve_header_html = f"""
        <div class="mb-panel" style="padding-bottom: 12px; margin-bottom: 0;">
            <div class="mb-panel-header" style="margin-bottom: 4px; padding-bottom: 8px;">
                <div>
                    <h4 class="mb-panel-title">🌊 Universe Accumulation vs Distribution Curve (IBD GMI)</h4>
                    <p class="mb-panel-desc">Count of ~1,350 US equities classified into trailing 13-week institutional money flow tiers (A to E)</p>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Smart Money Ratio</span><br>
                    <span style="font-family: 'JetBrains Mono', monospace; font-weight: 800; font-size: 1.15rem; color: {sm_color};">
                        {sm_ratio:.2f}x
                    </span>
                </div>
            </div>
        """
        st.markdown(clean_html(curve_header_html), unsafe_allow_html=True)
        
        # Plotly Interactive Distribution Bell-Curve Chart
        fig_ad_curve = create_ad_distribution_fig(vol_dist)
        st.plotly_chart(fig_ad_curve, use_container_width=True, config={'displayModeBar': False})
        
        # 3-Tier Money Flow Summary Footer
        curve_footer_html = f"""
            <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 6px 12px; font-size: 0.77rem; font-family: 'JetBrains Mono', monospace; margin-top: -6px;">
                <span style="color: #34d399;">🟢 <b>Accumulating (A+B):</b> {acc_count} ({acc_pct:.1f}%)</span>
                <span style="color: #94a3b8;">⚪ <b>Neutral (C):</b> {c_cnt} ({c_pct:.1f}%)</span>
                <span style="color: #f87171;">🔴 <b>Distributing (D+E):</b> {dist_count} ({dist_pct:.1f}%)</span>
            </div>
        </div>
        """
        st.markdown(clean_html(curve_footer_html), unsafe_allow_html=True)

# -------------------------------------------------------------
# MODULE 1B: INSTITUTIONAL BREAKOUT HEALTH & FAILURE RADAR (US EQUITIES)
# -------------------------------------------------------------
try:
    render_breakout_health_radar_section(is_us=True, key_prefix="us_breadth_radar")
except Exception as e:
    st.error(f"Error rendering US Breakout Health Radar: {e}")

# -------------------------------------------------------------
# MODULE C: MULTI-TIMEFRAME EXPANSION (NEW HIGHS VS LOWS: 1W · 1M · 3M · 52W)
# -------------------------------------------------------------
if breadth_alpha and 'multi_hl' in breadth_alpha and breadth_alpha['multi_hl']:
    hl_1w = breadth_alpha['multi_hl'].get('1W (5D)', {})
    hl_1m = breadth_alpha['multi_hl'].get('1M (21D)', {})
    hl_3m = breadth_alpha['multi_hl'].get('3M (63D)', {})
    hl_52w = breadth_alpha['multi_hl'].get('1Y (52W)', {})

    net_1w = hl_1w.get('Net', 0)
    net_1m = hl_1m.get('Net', 0)
    net_3m = hl_3m.get('Net', 0)
    net_52w = hl_52w.get('Net', 0)

    if net_1w > 0 and net_1m > 0 and net_52w > 0:
        trend_summary = "🔥 Synchronized expansion across all horizons (Bullish Super-Cycle)."
    elif net_1w > 0 and net_1m <= 0:
        trend_summary = "🟡 Tactical counter-trend bounce inside intermediate consolidation."
    elif net_1w < 0 and net_1m < 0 and net_52w < 0:
        trend_summary = "🚨 Synchronized distribution across all horizons (Severe Correction Phase)."
    else:
        trend_summary = "⚖️ Mixed multi-horizon flow: selective leadership defending key bases."

    multi_hl_panel_html = f"""
    <div class="mb-panel" style="margin-bottom: 20px;">
        <div class="mb-panel-header" style="margin-bottom: 4px; padding-bottom: 8px;">
            <div>
                <h4 class="mb-panel-title">🎯 Multi-Horizon Expansion: New Highs vs Lows (1W · 1M · 3M · 52W)</h4>
                <p class="mb-panel-desc">Leading cycle transition gauge: tracks expanding leadership vs breakdown pressure across short, intermediate, and annual horizons before 52W extremes register</p>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Cycle State</span><br>
                <span style="font-family: 'Inter', sans-serif; font-weight: 700; font-size: 0.85rem; color: #38bdf8;">
                    {trend_summary}
                </span>
            </div>
        </div>
    """
    st.markdown(clean_html(multi_hl_panel_html), unsafe_allow_html=True)

    fig_multi_hl = create_multi_timeframe_hl_fig(breadth_alpha['multi_hl'])
    st.plotly_chart(fig_multi_hl, use_container_width=True, config={'displayModeBar': False})

    multi_hl_footer_html = f"""
        <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 6px 14px; font-size: 0.77rem; font-family: 'JetBrains Mono', monospace; margin-top: -6px;">
            <span>1W: <b style="color:{'#34d399' if net_1w >= 0 else '#f87171'};">{net_1w:+d}</b> ({hl_1w.get('Highs', 0)}H · {hl_1w.get('Lows', 0)}L) | 1M: <b style="color:{'#34d399' if net_1m >= 0 else '#f87171'};">{net_1m:+d}</b> ({hl_1m.get('Highs', 0)}H · {hl_1m.get('Lows', 0)}L) | 3M: <b style="color:{'#34d399' if net_3m >= 0 else '#f87171'};">{net_3m:+d}</b> ({hl_3m.get('Highs', 0)}H · {hl_3m.get('Lows', 0)}L) | 52W: <b style="color:{'#34d399' if net_52w >= 0 else '#f87171'};">{net_52w:+d}</b> ({hl_52w.get('Highs', 0)}H · {hl_52w.get('Lows', 0)}L)</span>
            <span style="color: #cbd5e1;">🟢 Highs: within 2% of peak | 🔴 Lows: within 2% of trough | 🔵 Net: Highs minus Lows</span>
        </div>
    </div>
    """
    st.markdown(clean_html(multi_hl_footer_html), unsafe_allow_html=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# =============================================================================
# 3.5. AUTOMATED BREADTH REGIME & DAILY ACTION COCKPIT (THE "TRAFFIC LIGHT" ENGINE)
# =============================================================================

@st.cache_data(ttl=600)
def load_cached_sector_breadth_us():
    return compute_sector_breadth_matrix_us()

with st.spinner("Calculating S&P 500 GICS Sector Breadth Matrix..."):
    sector_breadth_df = load_cached_sector_breadth_us()

breadth_eval = evaluate_comprehensive_breadth_checklist(
    df_active, df_top_rs, vol_dist, sp500_telem, zweig_tape,
    benchmark_name="S&P 500",
    breadth_alpha=breadth_alpha,
    sector_breadth_df=sector_breadth_df
)

cockpit_cards_html = ""
for it in breadth_eval['checklist_items']:
    status_bg = "rgba(16, 185, 129, 0.15)" if it['status'] == 'PASS' else ("rgba(245, 158, 11, 0.15)" if it['status'] == 'CAUTION' else "rgba(239, 68, 68, 0.15)")
    cockpit_cards_html += f"""
        <div class="mb-check-card" style="border-left: 3px solid {it['status_color']};">
            <div>
                <div class="mb-check-header">
                    <span class="mb-check-name">{it['icon']} {it['name']}</span>
                    <span class="mb-check-status-pill" style="background: {status_bg}; color: {it['status_color']}; border: 1px solid {it['status_color']}44;">
                        {it['status_icon']} {it['status']}
                    </span>
                </div>
                <div class="mb-check-body">
                    <span class="mb-check-val" style="color: {it['status_color']};">{it['reading_str']}</span>
                    <span class="mb-check-thresh">{it['threshold_str']}</span>
                </div>
            </div>
            <div class="mb-check-guidance">
                {it['guidance']}
            </div>
        </div>
    """

session_date_str = df_active.index.max().strftime('%Y-%m-%d') if not df_active.empty else 'Latest'

# Inflection guardrail alert badges in Cockpit Hero
inflection_pills_html = ""
stats = breadth_eval.get('summary_stats', {})
if stats.get('thrust_fast_track'):
    inflection_pills_html += '<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: 800;">🚀 BREADTH THRUST ACTIVE</span>'
if stats.get('is_bear_rally'):
    inflection_pills_html += '<span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: 800;">⚠️ BEAR TRAP GUARD ACTIVE</span>'
if stats.get('mega_cap_divergence'):
    inflection_pills_html += '<span style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: 800;">🚨 MEGA-CAP DIVERGENCE</span>'
if stats.get('decoupled_sectors'):
    n_dec = len(stats['decoupled_sectors'])
    inflection_pills_html += f'<span style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: 800;">👑 {n_dec} DECOUPLED GICS SECTORS</span>'

two_tiered_html = ""
if 'two_tiered_exception' in breadth_eval.get('action_directives', {}):
    two_tiered_html = f"""
    <div style="margin-top: 10px; background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 8px; padding: 10px 14px; font-size: 0.81rem; color: #e0f2fe; line-height: 1.5;">
        <strong>{breadth_eval['action_directives']['two_tiered_exception']}</strong>
    </div>
    """

cockpit_html = f"""
<div class="mb-action-cockpit">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 10px; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.25rem;">🗽</span>
            <div>
                <h3 style="margin: 0; font-size: 1.12rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.01em;">
                    S&P 500 Breadth Regime & Execution Cockpit (The "Traffic Light" System)
                </h3>
                <p style="margin: 2px 0 0 0; font-size: 0.74rem; color: #94a3b8;">
                    Live Non-Discretionary Allocation Engine • Verified Across 2022 US Bear Market & 2023–2024 AI Secular Advance
                </p>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 0.74rem; font-family: 'JetBrains Mono', monospace; color: #cbd5e1; background: rgba(255,255,255,0.05); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08);">
                Audit Score: <b style="color: #10b981;">{breadth_eval['summary_stats']['pass_count']} PASS</b> · <b style="color: #f59e0b;">{breadth_eval['summary_stats']['caution_count']} CAUTION</b> · <b style="color: #ef4444;">{breadth_eval['summary_stats']['fail_count']} FAIL</b>
            </span>
        </div>
    </div>

    <!-- Hero Regime Banner -->
    <div class="mb-hero-card" style="border-left: 4px solid {breadth_eval['badge_color']}; padding-left: 14px; background: {breadth_eval['badge_bg']}; border-radius: 10px; padding: 14px 16px;">
        <div style="flex: 1.8; min-width: 280px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px; flex-wrap: wrap;">
                <span class="mb-hero-badge-pill" style="background: {breadth_eval['badge_color']}; color: #0f172a;">
                    ● ACTIVE REGIME
                </span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: #cbd5e1;">
                    Last Session: <b>{session_date_str}</b>
                </span>
                {inflection_pills_html}
            </div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #f8fafc; margin-bottom: 6px; letter-spacing: -0.01em;">
                {breadth_eval['regime_title']}
            </div>
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.45;">
                {breadth_eval['regime_desc']}
            </div>
            {two_tiered_html}
        </div>
        <div class="mb-hero-meta">
            <div class="mb-hero-kpi" style="border-color: {breadth_eval['badge_color']}44;">
                <div class="mb-hero-kpi-label">🎯 Target Exposure</div>
                <div class="mb-hero-kpi-val" style="color: {breadth_eval['badge_color']};">{breadth_eval['target_exposure']}</div>
            </div>
            <div class="mb-hero-kpi">
                <div class="mb-hero-kpi-label">💰 Cash Target</div>
                <div class="mb-hero-kpi-val" style="color: {'#10b981' if '0%' in breadth_eval['cash_target'] else ('#f59e0b' if '35%' in breadth_eval['cash_target'] else '#ef4444')};">{breadth_eval['cash_target']}</div>
            </div>
            <div class="mb-hero-kpi">
                <div class="mb-hero-kpi-label">👑 Portfolio Sizing</div>
                <div class="mb-hero-kpi-val" style="font-size: 0.82rem; color: #38bdf8; max-width: 170px;">{breadth_eval['portfolio_sizing']}</div>
            </div>
        </div>
    </div>

    <!-- 4-Card Action Directives -->
    <div class="mb-directives-grid">
        <div class="mb-directive-card">
            <div class="mb-directive-title">🛒 New Breakout Buys</div>
            <div class="mb-directive-text">{breadth_eval['action_directives']['new_buys']}</div>
        </div>
        <div class="mb-directive-card">
            <div class="mb-directive-title">🛡️ Trailing Stop Rule</div>
            <div class="mb-directive-text">{breadth_eval['action_directives']['trailing_stops']}</div>
        </div>
        <div class="mb-directive-card">
            <div class="mb-directive-title">⚡ Cash Redeployment</div>
            <div class="mb-directive-text">{breadth_eval['action_directives']['cash_strategy']}</div>
        </div>
        <div class="mb-directive-card">
            <div class="mb-directive-title">🎯 Tape Risk Posture</div>
            <div class="mb-directive-text">{breadth_eval['action_directives']['risk_posture']}</div>
        </div>
    </div>

    <!-- 7-Pillar Live Checklist Grid Header -->
    <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
        <span>📋 S&P 500 7-Pillar Institutional Breadth Audit Checklist (Live Quantitative Telemetry)</span>
        <div style="flex: 1; height: 1px; background: rgba(255,255,255,0.06);"></div>
    </div>

    <!-- 7-Pillar Cards -->
    <div class="mb-checklist-grid">
        {cockpit_cards_html}
    </div>
</div>
"""
st.markdown(clean_html(cockpit_html), unsafe_allow_html=True)

# =============================================================================
# 3.7. INSTITUTIONAL EARLY INFLECTION RADAR (LIVE TRAFFIC LIGHTS & CHEAT SHEET)
# =============================================================================
early_radar_data = evaluate_early_inflection_radar(
    breadth_eval=breadth_eval,
    zweig_tape=zweig_tape,
    breadth_alpha=breadth_alpha,
    telem=sp500_telem,
    top_rs_21=latest_top_rs_21,
    bullish_divs=bullish_divs,
    bearish_divs=bearish_divs,
    latest_date=df_active.index[-1] if not df_active.empty else None,
    market="US",
    rolling_ad_div=rolling_ad_div
)

st.markdown(clean_html(render_early_inflection_radar_html(early_radar_data)), unsafe_allow_html=True)

with st.expander("📚 Early Inflection Signal Guide & Historical Lead-Time Cheat Sheet", expanded=False):
    st.markdown(clean_html(render_early_inflection_cheat_sheet_html()), unsafe_allow_html=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# =============================================================================
# 3.8. INSTITUTIONAL SECTOR BREADTH & LEADERSHIP RADAR (11 OFFICIAL S&P 500 GICS)
# =============================================================================
if sector_breadth_df is not None and not sector_breadth_df.empty:
    sec_df = sector_breadth_df.copy()
    tot_sec = len(sec_df)
    n_dec = int(sec_df['is_decoupled'].sum()) if 'is_decoupled' in sec_df.columns else 0
    n_bull = int((sec_df['above_50_pct'] >= 50.0).sum())
    n_chop = int(((sec_df['above_50_pct'] >= 40.0) & (sec_df['above_50_pct'] < 50.0)).sum())
    n_weak = int((sec_df['above_50_pct'] < 40.0).sum())

    st.markdown(clean_html(f"""
    <div style="background: linear-gradient(145deg, rgba(15, 23, 42, 0.90) 0%, rgba(2, 6, 23, 0.95) 100%); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px 20px; margin-top: 14px; margin-bottom: 18px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 12px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.3rem;">🗽</span>
                <div>
                    <h3 style="margin: 0; font-size: 1.08rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.01em;">
                        Institutional S&P 500 GICS Sector Breadth Radar
                    </h3>
                    <p style="margin: 2px 0 0 0; font-size: 0.75rem; color: #94a3b8;">
                        Multi-Horizon Moving Average Health across the 11 Official S&P 500 GICS Sectors • Surfaces Leading Clusters with Direct TradingView Charts
                    </p>
                </div>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span style="background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 3px 10px; border-radius: 6px; font-size: 0.74rem; font-family: 'JetBrains Mono', monospace;">
                    👑 Decoupled: <b>{n_dec}</b>
                </span>
                <span style="background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 3px 10px; border-radius: 6px; font-size: 0.74rem; font-family: 'JetBrains Mono', monospace;">
                    🟢 Bull (≥50%): <b>{n_bull}</b>
                </span>
                <span style="background: rgba(245, 158, 11, 0.12); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); padding: 3px 10px; border-radius: 6px; font-size: 0.74rem; font-family: 'JetBrains Mono', monospace;">
                    🟡 Chop (40-49%): <b>{n_chop}</b>
                </span>
                <span style="background: rgba(239, 68, 68, 0.12); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); padding: 3px 10px; border-radius: 6px; font-size: 0.74rem; font-family: 'JetBrains Mono', monospace;">
                    🔴 Weak (&lt;40%): <b>{n_weak}</b>
                </span>
            </div>
        </div>
    """), unsafe_allow_html=True)

    c_rf1, c_rf2 = st.columns([2, 1], vertical_alignment="center")
    with c_rf1:
        f_choice = st.segmented_control(
            "Filter GICS Sectors:",
            ["🟢 Bull Sectors (≥50%)", "👑 Decoupled Leaders", "🟡 Consolidating (40-49%)", "🔴 Liquidating (<40%)", "All 11 Sectors"],
            default="🟢 Bull Sectors (≥50%)",
            key="sec_radar_filter_us"
        )
    with c_rf2:
        search_kw = st.text_input("🔍 Search Sector or Ticker:", "", key="sec_radar_search_us", placeholder="e.g. Technology, NVDA, Energy")

    display_sec = sec_df.copy()
    if f_choice == "👑 Decoupled Leaders":
        display_sec = display_sec[display_sec['is_decoupled']]
    elif f_choice == "🟢 Bull Sectors (≥50%)":
        display_sec = display_sec[display_sec['above_50_pct'] >= 50.0]
    elif f_choice == "🟡 Consolidating (40-49%)":
        display_sec = display_sec[(display_sec['above_50_pct'] >= 40.0) & (display_sec['above_50_pct'] < 50.0)]
    elif f_choice == "🔴 Liquidating (<40%)":
        display_sec = display_sec[display_sec['above_50_pct'] < 40.0]

    if search_kw:
        kw = search_kw.strip().lower()
        def match_row(row):
            if kw in row['sector'].lower():
                return True
            for s in row.get('stocks', []):
                if kw in s['ticker'].lower() or kw in s.get('company', '').lower():
                    return True
            return False
        display_sec = display_sec[display_sec.apply(match_row, axis=1)]

    if display_sec.empty:
        st.info(f"No GICS sectors match the current filter: '{f_choice}' ({search_kw}).")
    else:
        for _, srow in display_sec.iterrows():
            sec_name = srow['sector']
            tot_stk = srow['total_stocks']
            p50 = srow['above_50_pct']
            p200 = srow['above_200_pct']
            p20 = srow['above_20_pct']
            status_lbl = srow['status']
            stk_list = srow.get('stocks', [])

            bar_color = "#38bdf8" if srow['is_decoupled'] else ("#10b981" if p50 >= 50.0 else ("#f59e0b" if p50 >= 40.0 else "#ef4444"))
            badge_bg = "rgba(56, 189, 248, 0.15)" if srow['is_decoupled'] else ("rgba(16, 185, 129, 0.15)" if p50 >= 50.0 else ("rgba(245, 158, 11, 0.15)" if p50 >= 40.0 else "rgba(239, 68, 68, 0.15)"))

            pills_html = ""
            for s in stk_list[:8]:
                tick = s['ticker']
                cname = s.get('company', tick)
                short_cname = '' if str(cname).strip().upper() == str(tick).strip().upper() else ((cname[:18] + '..') if len(cname) > 18 else cname)
                p_dist = s.get('pct_above_50', 0.0)
                stk_color = "#34d399" if s.get('above_50') else "#f87171"
                stk_bg = "rgba(16, 185, 129, 0.10)" if s.get('above_50') else "rgba(239, 68, 68, 0.10)"
                stk_border = "rgba(16, 185, 129, 0.25)" if s.get('above_50') else "rgba(239, 68, 68, 0.25)"
                tv_url = f"https://www.tradingview.com/chart/?symbol={tick}"

                pills_html += f"""
                <a href="{tv_url}" target="_blank" style="text-decoration: none; margin: 2px;">
                    <span style="display: inline-flex; align-items: center; gap: 4px; background: {stk_bg}; color: {stk_color}; border: 1px solid {stk_border}; border-radius: 6px; padding: 2px 7px; font-size: 0.73rem; font-family: 'JetBrains Mono', monospace;">
                        <b>{tick}</b> <span style="color: #94a3b8; font-size: 0.70rem;">{short_cname}</span> <span style="font-weight: 700;">{p_dist:+0.1f}%</span>
                    </span>
                </a>
                """

            card_html = f"""
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 6px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 0.88rem; font-weight: 700; color: #f8fafc;">{sec_name}</span>
                        <span style="color: #64748b; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;">({tot_stk} S&P 500 stocks)</span>
                        <span style="background: {badge_bg}; color: {bar_color}; border: 1px solid {bar_color}44; padding: 1px 7px; border-radius: 10px; font-size: 0.70rem; font-weight: 800;">
                            {status_lbl}
                        </span>
                    </div>
                    <div style="display: flex; gap: 12px; font-size: 0.74rem; font-family: 'JetBrains Mono', monospace; color: #cbd5e1;">
                        <span>>50 SMA: <b style="color: {bar_color};">{p50:.1f}%</b> ({srow['above_50_count']}/{tot_stk})</span>
                        <span>>200 SMA: <b style="color: {'#34d399' if p200 >= 50 else '#f87171'};">{p200:.1f}%</b></span>
                        <span>>20 EMA: <b style="color: {'#34d399' if p20 >= 50 else '#f87171'};">{p20:.1f}%</b></span>
                    </div>
                </div>
                <div style="height: 4px; background: rgba(255,255,255,0.06); border-radius: 2px; overflow: hidden; margin-bottom: 8px;">
                    <div style="width: {min(100, max(0, p50))}%; height: 100%; background: {bar_color}; border-radius: 2px;"></div>
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 4px; align-items: center;">
                    <span style="font-size: 0.70rem; color: #64748b; margin-right: 4px;">Top Constituents:</span>
                    {pills_html}
                </div>
            </div>
            """
            st.markdown(clean_html(card_html), unsafe_allow_html=True)
            if len(stk_list) > 8:
                with st.expander(f"View all {len(stk_list)} constituent stocks in {sec_name}", expanded=False):
                    extra_pills = ""
                    for s in stk_list:
                        tick = s['ticker']
                        cname = s.get('company', tick)
                        p_dist = s.get('pct_above_50', 0.0)
                        stk_color = "#34d399" if s.get('above_50') else "#f87171"
                        stk_bg = "rgba(16, 185, 129, 0.10)" if s.get('above_50') else "rgba(239, 68, 68, 0.10)"
                        stk_border = "rgba(16, 185, 129, 0.25)" if s.get('above_50') else "rgba(239, 68, 68, 0.25)"
                        tv_url = f"https://www.tradingview.com/chart/?symbol={tick}"
                        extra_pills += f"""
                        <a href="{tv_url}" target="_blank" style="text-decoration: none; margin: 2px;">
                            <span style="display: inline-flex; align-items: center; gap: 4px; background: {stk_bg}; color: {stk_color}; border: 1px solid {stk_border}; border-radius: 6px; padding: 3px 8px; font-size: 0.74rem; font-family: 'JetBrains Mono', monospace;">
                                <b>{tick}</b> <span style="color: #cbd5e1;">{cname}</span> <span style="font-weight: 700;">({p_dist:+0.1f}%)</span>
                            </span>
                        </a>
                        """
                    st.markdown(clean_html(f"<div style='display: flex; flex-wrap: wrap; gap: 4px; padding: 6px 0;'>{extra_pills}</div>"), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 4. TACTICAL TIME-SERIES MULTI-PANEL BREADTH VISUALIZER
# =============================================================================

# Ensure data types are correct
df_active['above_20_pct'] = pd.to_numeric(df_active['above_20_pct'], errors='coerce')
df_active['above_50_pct'] = pd.to_numeric(df_active['above_50_pct'], errors='coerce')
df_active['above_200_pct'] = pd.to_numeric(df_active['above_200_pct'], errors='coerce')
df_active['ad_line_cumulative'] = pd.to_numeric(df_active['ad_line_cumulative'], errors='coerce')
df_active['new_highs_count'] = pd.to_numeric(df_active['new_highs_count'], errors='coerce')
df_active['new_lows_count'] = pd.to_numeric(df_active['new_lows_count'], errors='coerce')
df_active['double_52w_count'] = pd.to_numeric(df_active['double_52w_count'], errors='coerce')
if 'quadruple_52w_count' in df_active.columns:
    df_active['quadruple_52w_count'] = pd.to_numeric(df_active['quadruple_52w_count'], errors='coerce')
if 'advances' in df_active.columns:
    df_active['advances'] = pd.to_numeric(df_active['advances'], errors='coerce')
    df_active['declines'] = pd.to_numeric(df_active['declines'], errors='coerce')
if 'panic_extreme_5d' in df_active.columns:
    df_active['panic_extreme_5d'] = pd.to_numeric(df_active['panic_extreme_5d'], errors='coerce')
    df_active['surge_extreme_5d'] = pd.to_numeric(df_active['surge_extreme_5d'], errors='coerce')

# Trim warmup days
if len(df_active) > 200:
    df_calc = df_active.iloc[200:]
else:
    df_calc = df_active

# Calculate SMAs across full history before slicing
ad_sma_20 = df_calc['ad_line_cumulative'].rolling(window=20, min_periods=1).mean()
net_hl = df_calc['new_highs_count'] - df_calc['new_lows_count']
net_hl_sma = net_hl.rolling(window=20).mean()
froth_sma = df_calc['double_52w_count'].rolling(window=20).mean()
quad_sma = df_calc['quadruple_52w_count'].rolling(window=20).mean() if 'quadruple_52w_count' in df_calc.columns else pd.Series(index=df_calc.index, data=0)

if 'advances' in df_calc.columns and 'declines' in df_calc.columns:
    total_issues = (df_calc['advances'] + df_calc['declines']).replace(0, 1)
    rna = ((df_calc['advances'] - df_calc['declines']) / total_issues) * 1000
    mco = rna.ewm(span=19, adjust=False).mean() - rna.ewm(span=39, adjust=False).mean()
else:
    mco = pd.Series(index=df_calc.index, data=0)

panic_ext = df_calc['panic_extreme_5d'] if 'panic_extreme_5d' in df_calc.columns else pd.Series(index=df_calc.index, data=0)
surge_ext = df_calc['surge_extreme_5d'] if 'surge_extreme_5d' in df_calc.columns else pd.Series(index=df_calc.index, data=0)

# =============================================================================
# 3.9 STOCKBEE MARKET MONITOR (US) — DYNAMIC BY SELECTED UNIVERSE SCOPE
# =============================================================================
from components import load_stockbee_monitor_df, render_stockbee_monitor_grid_html

_sb_us_label = (
    "Nasdaq 100"
    if "Nasdaq 100" in selected_scope
    else ("Broad US Equities" if "Broad" in selected_scope else "S&P 500")
)
_sb_uh1, _sb_uh2 = st.columns([2.2, 1.0], vertical_alignment="center")
with _sb_uh1:
    st.markdown(
        f"<h3 style='margin:0; font-size:1.22rem; font-weight:800; color:#f8fafc;'>📊 Stockbee Market Monitor — {_sb_us_label}</h3>"
        "<p style='margin:2px 0 0 0; font-size:0.78rem; color:#94a3b8;'>Synced to your active top Market Scope filter: daily &plusmn;4% movers, 5D/10D thrust ratios (&ge; 2.0), &plusmn;25% Qtr leaders, and T2108 (% &gt; 40 SMA).</p>",
        unsafe_allow_html=True,
    )
with _sb_uh2:
    _sb_us_sessions = st.pills(
        "Sessions:",
        [15, 20, 30],
        format_func=lambda x: f"{x} Sessions",
        default=20,
        key="mb_us_stockbee_sessions",
    ) or 20

_sb_df_us = load_stockbee_monitor_df(
    market="US",
    universe_scope=selected_scope,
    lookback_days=int(_sb_us_sessions),
)
if not _sb_df_us.empty:
    st.markdown(
        clean_html(render_stockbee_monitor_grid_html(_sb_df_us, universe_label=_sb_us_label)),
        unsafe_allow_html=True,
    )
else:
    st.info(f"Stockbee Market Monitor has insufficient local history for {_sb_us_label}. Click '🔄 Sync Today's Breadth' to populate.")

try:
    from leadership_cluster_ui import render_thrusting_industries_section
    _m_reg_us = breadth_eval.get("regime") if isinstance(locals().get("breadth_eval"), dict) else None
    render_thrusting_industries_section(market="US", universe_scope=selected_scope, m_gate_regime=_m_reg_us)
except Exception as _ti_us_err:
    print(f"[market_breadth_us] Thrusting Industries warning: {_ti_us_err}")
    st.caption("Thrusting Industries panel temporarily unavailable.")

# Time Horizon & View Layout Control Bar
c_ch_hdr, c_ch_ctrl, c_ch_view, c_ch_bm = st.columns([1.4, 1.1, 1.1, 0.9], vertical_alignment="center")
with c_ch_hdr:
    st.markdown("<h3 style='margin:0; font-size:1.25rem; font-weight:800; color:#f8fafc;'>📈 Technical Breadth Analytics</h3><p style='margin:2px 0 0 0; font-size:0.78rem; color:#94a3b8;'>Time-series synchronization across Net A/D Line, Net 52W Highs/Lows, Froth, Leadership MA Breadth, and MCO.</p>", unsafe_allow_html=True)
with c_ch_ctrl:
    time_horizon = st.pills(
        "Zoom Horizon:",
        ["6M (Tactical)", "1Y (Swing)", "2Y (Cycle)", "All History (1000D)"],
        default="1Y (Swing)",
        key="us_time_horizon"
    )
with c_ch_view:
    view_mode = st.pills(
        "Display Layout:",
        ["📑 Modular Stacked (Full Width)", "🔲 Multi-Grid (Synchronized)", "🔍 Focus Tabs"],
        default="📑 Modular Stacked (Full Width)",
        key="us_view_mode"
    )
with c_ch_bm:
    benchmark_choice = st.pills(
        "Overlay:",
        ["S&P 500", "Nasdaq"],
        default="S&P 500",
        key="us_bm_choice"
    )

if not time_horizon:
    time_horizon = "1Y (Swing)"
if not view_mode:
    view_mode = "📑 Modular Stacked (Full Width)"
if not benchmark_choice:
    benchmark_choice = "S&P 500"

if "6M" in time_horizon:
    df_plot = df_calc.tail(126).copy()
elif "1Y" in time_horizon:
    df_plot = df_calc.tail(252).copy()
elif "2Y" in time_horizon:
    df_plot = df_calc.tail(504).copy()
else:
    df_plot = df_calc.copy()

min_chart_dt = df_plot.index.min()
colors_hl = ['#ef4444' if val < 0 else '#3b82f6' for val in net_hl.reindex(df_plot.index)]

df_active_overlay = df_sp500_close if "S&P 500" in benchmark_choice else df_nasdaq_close
overlay_name = "S&P 500 Index" if "S&P 500" in benchmark_choice else "Nasdaq Composite"

# -------------------------------------------------------------
# STANDALONE CHART BUILDERS FOR MODULAR STACK & TABS
# -------------------------------------------------------------
def render_ad_line_signal_guide_html(overlay_label: str = "Benchmark") -> str:
    """
    Renders compact institutional UI legend explaining chart signals,
    Stealth Decay, Bearish/Bullish divergences, and 20-Day SMA breadth filter.
    """
    return clean_html(f"""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 9px 12px; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; padding-bottom: 4px; border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
            <span style="font-size: 0.68rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; color: #94a3b8; display: flex; align-items: center; gap: 5px;">
                <span>🧭</span> Chart Signals & Divergence Field Guide
            </span>
            <span style="font-size: 0.65rem; color: #64748b; font-style: italic;">
                Institutional Breadth Mechanics (Weinstein, Ryan & Zweig)
            </span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 8px;">
            <!-- Stealth Decay -->
            <div style="background: rgba(239, 68, 68, 0.07); border: 1px solid rgba(239, 68, 68, 0.22); border-radius: 6px; padding: 6px 9px;">
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 2px;">
                    <span style="background: rgba(239, 68, 68, 0.25); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.45); padding: 1px 5px; border-radius: 3px; font-size: 0.66rem; font-weight: 700; white-space: nowrap;">
                        ⚠️ Stealth Decay
                    </span>
                    <span style="font-size: 0.68rem; color: #fca5a5; font-weight: 700;">Mega-Cap Masking</span>
                </div>
                <div style="font-size: 0.70rem; color: #cbd5e1; line-height: 1.3;">
                    {overlay_label} trades within &le;1.5% of highs, but Cumulative A/D drops &ge;1.0&sigma;. Narrow heavyweights conceal widespread institutional distribution (2–6 week advance warning).
                </div>
            </div>
            <!-- Bearish Divergence -->
            <div style="background: rgba(239, 68, 68, 0.04); border: 1px solid rgba(239, 68, 68, 0.15); border-radius: 6px; padding: 6px 9px;">
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 2px;">
                    <span style="background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.35); padding: 1px 5px; border-radius: 3px; font-size: 0.66rem; font-weight: 700; white-space: nowrap;">
                        ▼ Bearish Div
                    </span>
                    <span style="font-size: 0.68rem; color: #f87171; font-weight: 700;">Topping Warning</span>
                </div>
                <div style="font-size: 0.70rem; color: #cbd5e1; line-height: 1.3;">
                    {overlay_label} prints equal/higher swing peaks while Cumulative A/D prints lower highs (&gt;100 issue loss over &gt;25 days). Signals narrowing rally participation before a correction.
                </div>
            </div>
            <!-- Bullish Divergence -->
            <div style="background: rgba(16, 185, 129, 0.04); border: 1px solid rgba(16, 185, 129, 0.15); border-radius: 6px; padding: 6px 9px;">
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 2px;">
                    <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); padding: 1px 5px; border-radius: 3px; font-size: 0.66rem; font-weight: 700; white-space: nowrap;">
                        ▲ Bullish Div
                    </span>
                    <span style="font-size: 0.68rem; color: #6ee7b7; font-weight: 700;">Bottoming Accumulation</span>
                </div>
                <div style="font-size: 0.70rem; color: #cbd5e1; line-height: 1.3;">
                    {overlay_label} prints lower swing lows while Cumulative A/D prints higher lows. Smart money secretly accumulates broader issues before the headline index rebounds.
                </div>
            </div>
            <!-- Trend Line & SMA -->
            <div style="background: rgba(59, 130, 246, 0.04); border: 1px solid rgba(59, 130, 246, 0.15); border-radius: 6px; padding: 6px 9px;">
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 2px;">
                    <span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); padding: 1px 5px; border-radius: 3px; font-size: 0.66rem; font-weight: 700; white-space: nowrap;">
                        〰️ 20-Day SMA
                    </span>
                    <span style="font-size: 0.68rem; color: #93c5fd; font-weight: 700;">Breadth Momentum</span>
                </div>
                <div style="font-size: 0.70rem; color: #cbd5e1; line-height: 1.3;">
                    Net A/D (blue) above 20 SMA (gold) confirms expanding participation. Slipping below 20 SMA signals short-term breadth exhaustion and tactical chop.
                </div>
            </div>
        </div>
    </div>
    """)

def build_ad_line_fig(height=460):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_active_overlay.reindex(df_plot.index), mode='lines', name=overlay_name, line=dict(color='rgba(255,255,255,0.4)', width=1.5), fill='tozeroy', fillcolor='rgba(255,255,255,0.06)'), secondary_y=True)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ad_line_cumulative'], mode='lines', name='Cumulative Net A/D Line', line=dict(color='#3b82f6', width=2.5), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.05)'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_plot.index, y=ad_sma_20.reindex(df_plot.index), mode='lines', name='20-Day SMA', line=dict(color='#f59e0b', width=1.5, dash='dot')), secondary_y=False)

    # Delta 5: Active Rolling Divergence Shading
    if rolling_ad_div and rolling_ad_div.get('sessions_count', 0) >= 2:
        div_mask = rolling_ad_div.get('div_mask', pd.Series(dtype=bool))
        if not div_mask.empty:
            act_dts = div_mask[div_mask].index
            if len(act_dts) > 0 and pd.to_datetime(act_dts[-1]) >= min_chart_dt:
                fig.add_vrect(
                    x0=act_dts[max(0, len(act_dts) - rolling_ad_div['sessions_count'])],
                    x1=act_dts[-1],
                    fillcolor="rgba(239, 68, 68, 0.14)",
                    layer="below", line_width=1, line_color="rgba(239, 68, 68, 0.4)", line_dash="dash",
                    annotation_text=f"<b>Stealth Decay ({rolling_ad_div['sessions_count']}d)</b>",
                    annotation_position="top left",
                    annotation_font=dict(color="#f87171", size=10)
                )
    
    for div_date, div_ad_val in bearish_divs:
        if pd.to_datetime(div_date) >= min_chart_dt:
            fig.add_annotation(
                x=div_date, y=div_ad_val,
                text="<b>▼ Bearish Div</b>",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#ef4444",
                ax=0, ay=-35,
                font=dict(color="#ef4444", size=10),
                bgcolor="rgba(15,23,42,0.85)", bordercolor="#ef4444", borderwidth=1,
                secondary_y=False
            )
    for div_date, div_ad_val in bullish_divs:
        if pd.to_datetime(div_date) >= min_chart_dt:
            fig.add_annotation(
                x=div_date, y=div_ad_val,
                text="<b>▲ Bullish Div</b>",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#10b981",
                ax=0, ay=35,
                font=dict(color="#10b981", size=10),
                bgcolor="rgba(15,23,42,0.85)", bordercolor="#10b981", borderwidth=1,
                secondary_y=False
            )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        hovermode='x unified',
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=True, zerolinecolor='rgba(255,255,255,0.1)', secondary_y=False)
    fig.update_yaxes(showgrid=False, zeroline=False, secondary_y=True)
    return fig

def build_ma_breadth_fig(height=480):
    fig = go.Figure()
    fig.add_hrect(y0=40, y1=60, line_width=0, fillcolor="rgba(255,255,255,0.03)", layer="below")
    if df_top_rs is not None and not df_top_rs.empty:
        if not isinstance(df_top_rs.index, pd.DatetimeIndex):
            df_top_rs.index = pd.to_datetime(df_top_rs.index)
        if hasattr(df_top_rs.index, 'tz') and df_top_rs.index.tz is not None:
            df_top_rs.index = df_top_rs.index.tz_localize(None)
        sub_top_rs = df_top_rs[df_top_rs.index >= min_chart_dt]
        if not sub_top_rs.empty:
            fig.add_trace(go.Scatter(x=sub_top_rs.index, y=sub_top_rs['top100_above_21_ema'], mode='lines', name='👑 Top 100 RS > 21 EMA', line=dict(color='#06b6d4', width=2), hovertemplate="👑 Top 100 > 21 EMA: %{y:.1f}%<extra></extra>"))
            fig.add_trace(go.Scatter(x=sub_top_rs.index, y=sub_top_rs['top100_above_50_sma'], mode='lines', name='👑 Top 100 RS > 50 SMA', line=dict(color='#10b981', width=2.5), hovertemplate="👑 Top 100 > 50 SMA: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['above_50_pct'], mode='lines', name='Broad % Above 50 SMA', line=dict(color='#f59e0b', width=1.5, dash='dot'), fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.04)', hovertemplate="Broad > 50 SMA: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['above_200_pct'], mode='lines', name='Broad % Above 200 SMA', line=dict(color='#3b82f6', width=1.3), hovertemplate="Broad > 200 SMA: %{y:.1f}%<extra></extra>"))

    for t_start, t_end in breadth_thrusts:
        dt_end = pd.to_datetime(t_end)
        dt_start = pd.to_datetime(t_start)
        if dt_end >= min_chart_dt:
            fig.add_vrect(
                x0=max(dt_start, min_chart_dt), x1=dt_end,
                fillcolor="rgba(16, 185, 129, 0.18)", layer="below", line_width=1,
                line_color="rgba(16, 185, 129, 0.5)", line_dash="dot",
                annotation_text="⚡ Zweig Thrust (ZBT)", annotation_position="top left",
                annotation_font_size=10, annotation_font_color="#10b981"
            )

    fig.add_hline(y=60, line_dash="dash", line_color="rgba(16,185,129,0.5)", annotation_text="60% (Bullish Zone)", annotation_position="top left", annotation_font_size=10, annotation_font_color="rgba(16,185,129,0.8)")
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text="50% (Neutral)", annotation_position="top left", annotation_font_size=10, annotation_font_color="rgba(255,255,255,0.5)")
    fig.add_hline(y=40, line_dash="dash", line_color="rgba(239,68,68,0.5)", annotation_text="40% (Bearish Zone)", annotation_position="bottom left", annotation_font_size=10, annotation_font_color="rgba(239,68,68,0.8)")

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        hovermode='x unified',
        height=height,
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor='rgba(255,255,255,0.03)'),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def build_net_hl_fig(height=420):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_plot.index, y=net_hl.reindex(df_plot.index), name='Net High-Low', marker_color=colors_hl, marker_line_width=0, opacity=0.8))
    fig.add_trace(go.Scatter(x=df_plot.index, y=net_hl_sma.reindex(df_plot.index), mode='lines', name='20-Day SMA', line=dict(color='#f8fafc', width=1.6)))
    
    for h_date, h_val in hindenburg_alerts:
        if pd.to_datetime(h_date) >= min_chart_dt:
            fig.add_annotation(
                x=h_date, y=h_val, text="<b>! Warning</b>",
                showarrow=True, arrowhead=1, arrowsize=1, arrowwidth=1.5, arrowcolor="#ef4444",
                ax=0, ay=30, font=dict(color="#ef4444", size=9),
                bgcolor="rgba(15,23,42,0.85)", bordercolor="#ef4444", borderwidth=1
            )
            
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        hovermode='x unified',
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=True, zerolinecolor='rgba(255,255,255,0.15)')
    return fig

def build_mco_fig(height=420):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=mco.reindex(df_plot.index), mode='lines', name='MCO', line=dict(color='#3b82f6', width=1.6), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.05)'))
    fig.add_hline(y=100, line_dash="dash", line_color="rgba(239,68,68,0.5)", annotation_text="Overbought (+100)", annotation_position="top left", annotation_font_size=9, annotation_font_color="rgba(239,68,68,0.7)")
    fig.add_hline(y=-100, line_dash="dash", line_color="rgba(16,185,129,0.5)", annotation_text="Oversold (-100)", annotation_position="bottom left", annotation_font_size=9, annotation_font_color="rgba(16,185,129,0.7)")
    
    for a_date, a_val, a_type in mco_alerts:
        if pd.to_datetime(a_date) >= min_chart_dt:
            if a_type == "buy":
                fig.add_annotation(x=a_date, y=a_val, text="<b>▲ Swing Buy</b>", showarrow=True, arrowhead=1, arrowsize=1, arrowcolor="#10b981", ax=0, ay=30, font=dict(color="#10b981", size=9), bgcolor="rgba(15,23,42,0.85)", bordercolor="#10b981", borderwidth=1)
            else:
                fig.add_annotation(x=a_date, y=a_val, text="<b>▼ Swing Sell</b>", showarrow=True, arrowhead=1, arrowsize=1, arrowcolor="#ef4444", ax=0, ay=-30, font=dict(color="#ef4444", size=9), bgcolor="rgba(15,23,42,0.85)", bordercolor="#ef4444", borderwidth=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        hovermode='x unified',
        height=height,
        showlegend=False
    )
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=True, zerolinecolor='rgba(255,255,255,0.15)')
    return fig

def build_froth_fig(height=420):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['double_52w_count'], name='Doubled in 52W', marker_color='#f59e0b', marker_line_width=0, opacity=0.75))
    fig.add_trace(go.Scatter(x=df_plot.index, y=froth_sma.reindex(df_plot.index), mode='lines', name='Doubler 20-Day SMA', line=dict(color='#f8fafc', width=1.5)))
    fig.add_trace(go.Scatter(x=df_plot.index, y=quad_sma.reindex(df_plot.index), mode='lines', name='Quadrupled (4X) 20-Day SMA', line=dict(color='#ef4444', width=1.5), fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.3)'))
    
    for f_date, f_val in froth_peaks:
        if pd.to_datetime(f_date) >= min_chart_dt:
            fig.add_annotation(x=f_date, y=f_val, text="<b>▼ Peak Froth</b>", showarrow=True, arrowhead=1, arrowsize=1, arrowwidth=1.5, arrowcolor="#ef4444", ax=0, ay=-30, font=dict(color="#ef4444", size=9), bgcolor="rgba(15,23,42,0.85)", bordercolor="#ef4444", borderwidth=1)
            
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        hovermode='x unified',
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=True, zerolinecolor='rgba(255,255,255,0.15)')
    return fig

def build_climax_fig(height=420):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_plot.index, y=-panic_ext.reindex(df_plot.index), name='Panic (>20% Drop in 5D)', marker_color='#ef4444', marker_line_width=0, opacity=0.8))
    fig.add_trace(go.Bar(x=df_plot.index, y=surge_ext.reindex(df_plot.index), name='Surge (>20% Up in 5D)', marker_color='#10b981', marker_line_width=0, opacity=0.8))
    
    for c_date, c_val, c_type in climax_alerts:
        if pd.to_datetime(c_date) >= min_chart_dt:
            if c_type == "capitulation":
                fig.add_annotation(x=c_date, y=-c_val, text="<b>! Capitulation</b>", showarrow=True, arrowhead=1, arrowsize=1, arrowcolor="#ef4444", ax=0, ay=30, font=dict(color="#ef4444", size=9), bgcolor="rgba(15,23,42,0.85)", bordercolor="#ef4444", borderwidth=1)
            else:
                fig.add_annotation(x=c_date, y=c_val, text="<b>! Climax</b>", showarrow=True, arrowhead=1, arrowsize=1, arrowcolor="#10b981", ax=0, ay=-30, font=dict(color="#10b981", size=9), bgcolor="rgba(15,23,42,0.85)", bordercolor="#10b981", borderwidth=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        hovermode='x unified',
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=True, zerolinecolor='rgba(255,255,255,0.15)')
    return fig

# -------------------------------------------------------------
# RENDER ACCORDING TO SELECTED DISPLAY MODE
# -------------------------------------------------------------
if view_mode == "📑 Modular Stacked (Full Width)":
    # 1. Net A/D Line
    st.markdown(f"""
    <div class="mb-panel" style="margin-bottom: 20px;">
        <div class="mb-panel-header" style="margin-bottom: 8px; padding-bottom: 8px;">
            <div>
                <h4 class="mb-panel-title">📈 1. Cumulative Net Advance/Decline Line vs {overlay_name}</h4>
                <p class="mb-panel-desc">Leading indicator of broad market trend. Bearish divergences form when {overlay_name} makes higher highs while the A/D line makes lower highs.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(render_ad_line_signal_guide_html(overlay_name), unsafe_allow_html=True)
    st.plotly_chart(build_ad_line_fig(height=460), use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Moving Average Breadth & ZBT
    st.markdown("""
    <div class="mb-panel" style="margin-bottom: 20px;">
        <div class="mb-panel-header" style="margin-bottom: 8px; padding-bottom: 8px;">
            <div>
                <h4 class="mb-panel-title">⚡ 2. Moving Average Breadth & Dr. Martin Zweig Breadth Thrust (ZBT)</h4>
                <p class="mb-panel-desc">Cyan: Top 100 RS > 21 EMA | Green: Top 100 RS > 50 SMA | Yellow Dotted: Universe > 50 SMA | Shaded Green Bands: Dr. Martin Zweig Breadth Thrusts.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(build_ma_breadth_fig(height=480), use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Net High-Lows
    st.markdown("""
    <div class="mb-panel" style="margin-bottom: 20px;">
        <div class="mb-panel-header" style="margin-bottom: 8px; padding-bottom: 8px;">
            <div>
                <h4 class="mb-panel-title">🎯 3. Net New 52-Week Highs minus Lows</h4>
                <p class="mb-panel-desc">True expansion metric: Net differential with 20-Day SMA trendline. Red cluster warnings indicate expanding internal liquidation.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(build_net_hl_fig(height=420), use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    # 4. McClellan Oscillator
    st.markdown("""
    <div class="mb-panel" style="margin-bottom: 20px;">
        <div class="mb-panel-header" style="margin-bottom: 8px; padding-bottom: 8px;">
            <div>
                <h4 class="mb-panel-title">🌊 4. McClellan Oscillator (MCO)</h4>
                <p class="mb-panel-desc">Short-term momentum oscillator. Highlighted signals denote swing buy turn inflections (< -100) and swing sell reversals (> +100).</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(build_mco_fig(height=420), use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    # 5. Dual Column: Speculative Froth & Capitulation Climax
    f_c1, f_c2 = st.columns(2)
    with f_c1:
        st.markdown("""
        <div class="mb-panel" style="margin-bottom: 20px;">
            <div class="mb-panel-header" style="margin-bottom: 8px; padding-bottom: 8px;">
                <div>
                    <h4 class="mb-panel-title">🔥 5. Speculative Froth Tracker</h4>
                    <p class="mb-panel-desc">Stocks up 100% (2X) and 300% (4X) in 52 weeks.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(build_froth_fig(height=420), use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    with f_c2:
        st.markdown("""
        <div class="mb-panel" style="margin-bottom: 20px;">
            <div class="mb-panel-header" style="margin-bottom: 8px; padding-bottom: 8px;">
                <div>
                    <h4 class="mb-panel-title">🚨 6. 5-Day Capitulation & Climax Extremes</h4>
                    <p class="mb-panel-desc">Counts stocks crashing &gt;20% (Panic) or surging &gt;20% (Climax) in 5 days.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(build_climax_fig(height=420), use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

elif view_mode == "🔍 Focus Tabs":
    t_ad, t_ma, t_nh, t_mco, t_froth, t_climax = st.tabs([
        f"📈 Net A/D Line vs {overlay_name}",
        "⚡ MA Breadth & Zweig Thrust (ZBT)",
        "🎯 Net 52W Highs / Lows",
        "🌊 McClellan Oscillator (MCO)",
        "🔥 Speculative Froth Tracker",
        "🚨 Panic / Climax Extremes"
    ])
    with t_ad:
        st.caption(f"Cumulative Net Advancers minus Decliners vs {overlay_name}. Watch for divergence turns.")
        st.markdown(render_ad_line_signal_guide_html(overlay_name), unsafe_allow_html=True)
        st.plotly_chart(build_ad_line_fig(height=520), use_container_width=True, config={'displayModeBar': False})
    with t_ma:
        st.caption("Leading Growth RS Leaders (21 EMA / 50 SMA) vs Broad Market. Shaded bands highlight Zweig Breadth Thrust windows.")
        st.plotly_chart(build_ma_breadth_fig(height=520), use_container_width=True, config={'displayModeBar': False})
    with t_nh:
        st.caption("Net 52-Week Highs minus Lows with 20-Day SMA trendline and Hindenburg warnings.")
        st.plotly_chart(build_net_hl_fig(height=500), use_container_width=True, config={'displayModeBar': False})
    with t_mco:
        st.caption("McClellan Oscillator ratio-adjusted momentum. Highlights tactical swing overbought/oversold inflection points.")
        st.plotly_chart(build_mco_fig(height=500), use_container_width=True, config={'displayModeBar': False})
    with t_froth:
        st.caption("Speculative euphoria gauge: Count of stocks doubled (2X) or quadrupled (4X) within 52 weeks.")
        st.plotly_chart(build_froth_fig(height=500), use_container_width=True, config={'displayModeBar': False})
    with t_climax:
        st.caption("5-Day capitulation liquidation (>20% drop) vs runaway buying climax (>20% surge).")
        st.plotly_chart(build_climax_fig(height=500), use_container_width=True, config={'displayModeBar': False})

else:
    # 🔲 Multi-Grid (Synchronized) with User-Controlled Height
    g_ctrl1, g_ctrl2 = st.columns([2, 1], vertical_alignment="center")
    with g_ctrl1:
        st.caption("Synchronized crosshair across all 6 indicators. Select your display scale below to adjust vertical breathing room.")
    with g_ctrl2:
        grid_scale = st.segmented_control(
            "Grid Height Scale:",
            ["Standard (1800px)", "Large (2200px)", "Ultra (2600px)"],
            default="Large (2200px)",
            key="us_grid_scale"
        )
    if not grid_scale or "Large" in str(grid_scale):
        grid_height = 2200
    elif "Standard" in str(grid_scale):
        grid_height = 1800
    else:
        grid_height = 2600

    fig_grid = make_subplots(
        rows=4, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
        row_heights=[0.28, 0.22, 0.26, 0.24],
        specs=[[{"colspan": 2, "secondary_y": True}, None], [{}, {}], [{"colspan": 2}, None], [{}, {}]],
        subplot_titles=(
            f"<b>Cumulative Advance/Decline (A/D) Line vs {overlay_name} Overlay</b><br><span style='font-size:11px;color:#64748b;'>Leading indicator. Watch for divergences when the index makes new highs without breadth confirmation.</span>",
            "<b>Net 52-Week Highs minus Lows</b><br><span style='font-size:11px;color:#64748b;'>Blue bars indicate expanding leadership highs. Red clusters warn of supply.</span>",
            "<b>Speculative Froth Tracker (Stocks up 100% in 52W)</b><br><span style='font-size:11px;color:#64748b;'>Tracks extreme speculation phases. When the white line rolls over, easy money has ended.</span>",
            "<b>Leading vs Broad Market Moving Average Breadth (Top 100 RS vs Broad Market) & ⚡ Zweig Breadth Thrust</b><br><span style='font-size:11px;color:#64748b;'>Cyan: Top 100 RS > 21 EMA | Green: Top 100 RS > 50 SMA | Yellow Dotted: Universe > 50 SMA | Shaded Bands: ZBT</span>",
            "<b>McClellan Oscillator (MCO)</b><br><span style='font-size:11px;color:#64748b;'>Short-term momentum. Highlights swing inflection turns across ±100 boundaries.</span>",
            "<b>Capitulation / Buying Climax Tracker</b><br><span style='font-size:11px;color:#64748b;'>Counts stocks crashing >20% (Panic) or surging >20% (Climax) in 5 days.</span>"
        )
    )

    # 1. A/D Line (Row 1)
    fig_grid.add_trace(go.Scatter(x=df_plot.index, y=df_active_overlay.reindex(df_plot.index), mode='lines', name=overlay_name, line=dict(color='rgba(255,255,255,0.4)', width=1.5), fill='tozeroy', fillcolor='rgba(255,255,255,0.06)', legendgroup='1'), row=1, col=1, secondary_y=True)
    fig_grid.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ad_line_cumulative'], mode='lines', name='Net A/D Line', line=dict(color='#3b82f6', width=2.5), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.05)', legendgroup='1'), row=1, col=1, secondary_y=False)
    fig_grid.add_trace(go.Scatter(x=df_plot.index, y=ad_sma_20.reindex(df_plot.index), mode='lines', name='20-Day SMA', line=dict(color='#f59e0b', width=1.5, dash='dot'), legendgroup='1'), row=1, col=1, secondary_y=False)

    for div_date, div_ad_val in bearish_divs:
        if pd.to_datetime(div_date) >= min_chart_dt:
            fig_grid.add_annotation(
                x=div_date, y=div_ad_val,
                text="<b>▼ Bearish Div</b>",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#ef4444",
                ax=0, ay=-35, font=dict(color="#ef4444", size=10),
                bgcolor="rgba(15,23,42,0.85)", bordercolor="#ef4444", borderwidth=1,
                row=1, col=1, secondary_y=False
            )
    for div_date, div_ad_val in bullish_divs:
        if pd.to_datetime(div_date) >= min_chart_dt:
            fig_grid.add_annotation(
                x=div_date, y=div_ad_val,
                text="<b>▲ Bullish Div</b>",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#10b981",
                ax=0, ay=35, font=dict(color="#10b981", size=10),
                bgcolor="rgba(15,23,42,0.85)", bordercolor="#10b981", borderwidth=1,
                row=1, col=1, secondary_y=False
            )

    # 2. Net High-Lows (Row 2, Col 1)
    fig_grid.add_trace(go.Bar(x=df_plot.index, y=net_hl.reindex(df_plot.index), name='Net High-Low', marker_color=colors_hl, marker_line_width=0, opacity=0.8, legendgroup='2'), row=2, col=1)
    fig_grid.add_trace(go.Scatter(x=df_plot.index, y=net_hl_sma.reindex(df_plot.index), mode='lines', name='20-Day SMA', line=dict(color='#f8fafc', width=1.6), legendgroup='2'), row=2, col=1)
    for h_date, h_val in hindenburg_alerts:
        if pd.to_datetime(h_date) >= min_chart_dt:
            fig_grid.add_annotation(
                x=h_date, y=h_val, text="<b>! Warning</b>",
                showarrow=True, arrowhead=1, arrowsize=1, arrowwidth=1.5, arrowcolor="#ef4444",
                ax=0, ay=30, font=dict(color="#ef4444", size=9),
                bgcolor="rgba(15,23,42,0.85)", bordercolor="#ef4444", borderwidth=1,
                row=2, col=1
            )

    # 3. Speculative Froth (Row 2, Col 2)
    fig_grid.add_trace(go.Bar(x=df_plot.index, y=df_plot['double_52w_count'], name='Doubled in 52W', marker_color='#f59e0b', marker_line_width=0, opacity=0.75, legendgroup='3'), row=2, col=2)
    fig_grid.add_trace(go.Scatter(x=df_plot.index, y=froth_sma.reindex(df_plot.index), mode='lines', name='Doubler 20-Day SMA', line=dict(color='#f8fafc', width=1.5), legendgroup='3'), row=2, col=2)
    fig_grid.add_trace(go.Scatter(x=df_plot.index, y=quad_sma.reindex(df_plot.index), mode='lines', name='Quadrupled (4X) 20-Day SMA', line=dict(color='#ef4444', width=1.5), fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.3)', legendgroup='3'), row=2, col=2)
    for f_date, f_val in froth_peaks:
        if pd.to_datetime(f_date) >= min_chart_dt:
            fig_grid.add_annotation(
                x=f_date, y=f_val, text="<b>▼ Peak Froth</b>",
                showarrow=True, arrowhead=1, arrowsize=1, arrowwidth=1.5, arrowcolor="#ef4444",
                ax=0, ay=-30, font=dict(color="#ef4444", size=9),
                bgcolor="rgba(15,23,42,0.85)", bordercolor="#ef4444", borderwidth=1,
                row=2, col=2
            )

    # 4. Moving Average Breadth & ZBT (Row 3)
    fig_grid.add_hrect(y0=40, y1=60, line_width=0, fillcolor="rgba(255,255,255,0.03)", layer="below", row=3, col=1)
    if df_top_rs is not None and not df_top_rs.empty:
        if not isinstance(df_top_rs.index, pd.DatetimeIndex):
            df_top_rs.index = pd.to_datetime(df_top_rs.index)
        if hasattr(df_top_rs.index, 'tz') and df_top_rs.index.tz is not None:
            df_top_rs.index = df_top_rs.index.tz_localize(None)
        sub_top_rs = df_top_rs[df_top_rs.index >= min_chart_dt]
        if not sub_top_rs.empty:
            fig_grid.add_trace(go.Scatter(x=sub_top_rs.index, y=sub_top_rs['top100_above_21_ema'], mode='lines', name='👑 Top 100 RS > 21 EMA', line=dict(color='#06b6d4', width=2), legendgroup='4'), row=3, col=1)
            fig_grid.add_trace(go.Scatter(x=sub_top_rs.index, y=sub_top_rs['top100_above_50_sma'], mode='lines', name='👑 Top 100 RS > 50 SMA', line=dict(color='#10b981', width=2.5), legendgroup='4'), row=3, col=1)

    fig_grid.add_trace(go.Scatter(x=df_plot.index, y=df_plot['above_50_pct'], mode='lines', name='Broad % Above 50 SMA', line=dict(color='#f59e0b', width=1.5, dash='dot'), fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.04)', legendgroup='4'), row=3, col=1)
    fig_grid.add_trace(go.Scatter(x=df_plot.index, y=df_plot['above_200_pct'], mode='lines', name='Broad % Above 200 SMA', line=dict(color='#3b82f6', width=1.3), legendgroup='4'), row=3, col=1)

    for t_start, t_end in breadth_thrusts:
        dt_end = pd.to_datetime(t_end)
        dt_start = pd.to_datetime(t_start)
        if dt_end >= min_chart_dt:
            fig_grid.add_vrect(
                x0=max(dt_start, min_chart_dt), x1=dt_end,
                fillcolor="rgba(16, 185, 129, 0.18)", layer="below", line_width=1,
                line_color="rgba(16, 185, 129, 0.5)", line_dash="dot",
                annotation_text="⚡ Zweig Thrust (ZBT)", annotation_position="top left",
                annotation_font_size=10, annotation_font_color="#10b981",
                row=3, col=1
            )

    fig_grid.add_hline(y=60, line_dash="dash", line_color="rgba(16,185,129,0.5)", annotation_text="60% (Bullish Zone)", annotation_position="top left", annotation_font_size=10, annotation_font_color="rgba(16,185,129,0.8)", row=3, col=1)
    fig_grid.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text="50% (Neutral)", annotation_position="top left", annotation_font_size=10, annotation_font_color="rgba(255,255,255,0.5)", row=3, col=1)
    fig_grid.add_hline(y=40, line_dash="dash", line_color="rgba(239,68,68,0.5)", annotation_text="40% (Bearish Zone)", annotation_position="bottom left", annotation_font_size=10, annotation_font_color="rgba(239,68,68,0.8)", row=3, col=1)

    # 5. McClellan Oscillator (Row 4, Col 1)
    fig_grid.add_trace(go.Scatter(x=df_plot.index, y=mco.reindex(df_plot.index), mode='lines', name='MCO', line=dict(color='#3b82f6', width=1.6), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.05)', legendgroup='5'), row=4, col=1)
    fig_grid.add_hline(y=100, line_dash="dash", line_color="rgba(239,68,68,0.5)", annotation_text="Overbought (+100)", annotation_position="top left", annotation_font_size=9, annotation_font_color="rgba(239,68,68,0.7)", row=4, col=1)
    fig_grid.add_hline(y=-100, line_dash="dash", line_color="rgba(16,185,129,0.5)", annotation_text="Oversold (-100)", annotation_position="bottom left", annotation_font_size=9, annotation_font_color="rgba(16,185,129,0.7)", row=4, col=1)

    for a_date, a_val, a_type in mco_alerts:
        if pd.to_datetime(a_date) >= min_chart_dt:
            if a_type == "buy":
                fig_grid.add_annotation(x=a_date, y=a_val, text="<b>▲ Swing Buy</b>", showarrow=True, arrowhead=1, arrowsize=1, arrowcolor="#10b981", ax=0, ay=30, font=dict(color="#10b981", size=9), bgcolor="rgba(15,23,42,0.85)", bordercolor="#10b981", borderwidth=1, row=4, col=1)
            else:
                fig_grid.add_annotation(x=a_date, y=a_val, text="<b>▼ Swing Sell</b>", showarrow=True, arrowhead=1, arrowsize=1, arrowcolor="#ef4444", ax=0, ay=-30, font=dict(color="#ef4444", size=9), bgcolor="rgba(15,23,42,0.85)", bordercolor="#ef4444", borderwidth=1, row=4, col=1)

    # 6. Climax Extremes (Row 4, Col 2)
    fig_grid.add_trace(go.Bar(x=df_plot.index, y=-panic_ext.reindex(df_plot.index), name='Panic (>20% Drop)', marker_color='#ef4444', marker_line_width=0, opacity=0.8, legendgroup='6'), row=4, col=2)
    fig_grid.add_trace(go.Bar(x=df_plot.index, y=surge_ext.reindex(df_plot.index), name='Surge (>20% Up)', marker_color='#10b981', marker_line_width=0, opacity=0.8, legendgroup='6'), row=4, col=2)

    for c_date, c_val, c_type in climax_alerts:
        if pd.to_datetime(c_date) >= min_chart_dt:
            if c_type == "capitulation":
                fig_grid.add_annotation(x=c_date, y=-c_val, text="<b>! Capitulation</b>", showarrow=True, arrowhead=1, arrowsize=1, arrowcolor="#ef4444", ax=0, ay=30, font=dict(color="#ef4444", size=9), bgcolor="rgba(15,23,42,0.85)", bordercolor="#ef4444", borderwidth=1, row=4, col=2)
            else:
                fig_grid.add_annotation(x=c_date, y=c_val, text="<b>! Climax</b>", showarrow=True, arrowhead=1, arrowsize=1, arrowcolor="#10b981", ax=0, ay=-30, font=dict(color="#10b981", size=9), bgcolor="rgba(15,23,42,0.85)", bordercolor="#10b981", borderwidth=1, row=4, col=2)

    fig_grid.update_xaxes(showticklabels=True)
    fig_grid.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=55, b=20),
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        hovermode='x unified',
        height=grid_height,
        showlegend=False
    )
    fig_grid.update_xaxes(showgrid=False, zeroline=False, linecolor='rgba(255,255,255,0.1)')
    fig_grid.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=True, zerolinecolor='rgba(255,255,255,0.1)')
    fig_grid.update_yaxes(showticklabels=False, row=1, col=1, secondary_y=False)
    fig_grid.update_yaxes(showticklabels=False, row=1, col=1, secondary_y=True)
    fig_grid.update_yaxes(range=[0, 100], row=3, col=1)

    st.plotly_chart(fig_grid, use_container_width=True, config={'displayModeBar': False})

st.markdown("<br><hr style='border-color: rgba(255,255,255,0.05);'><p style='text-align: center; color: #64748b; font-size: 0.85rem; letter-spacing: 1px;'>DEEP MARKET BREADTH ENGINE (US) • KUSH TRACKER</p>", unsafe_allow_html=True)
