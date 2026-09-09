import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systematic_engine import compute_live_fast_momentum_matrix, get_target_portfolio
from canslim_swing_engine import generate_daily_canslim_swing_state

# st.set_page_config removed for Lite routing

try:
    from styles import load_css
    load_css()
except ImportError:
    pass

from components import render_header, render_metric_card, render_empty_state, render_disk_cache_sidebar
try:
    from views.true_market_leader import get_cached_universe
except ImportError:
    from pages.true_market_leader import get_cached_universe

render_disk_cache_sidebar(get_cached_universe)

# =============================================================================
# OPERATING MODE TOGGLE
# =============================================================================

st.markdown("""
<style>
div.row-widget.stRadio > div {
    flex-direction: row;
    background: rgba(30, 41, 59, 0.5);
    padding: 8px 16px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    display: inline-flex;
    margin-bottom: 12px;
}

/* =============================================================================
   WORLD-CLASS TERMINAL DESIGN SYSTEM (BLOOMBERG / TRADINGVIEW INSPIRED)
   ============================================================================= */

.terminal-card {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.85) 0%, rgba(2, 6, 23, 0.95) 100%);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 12px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(12px);
    cursor: pointer;
}

.terminal-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.7);
}

/* Card variant accents */
.card-green {
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-top: 2px solid #10b981;
}
.card-green:hover {
    border-color: rgba(16, 185, 129, 0.6);
    box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.7), 0 0 15px rgba(16, 185, 129, 0.2);
}

.card-amber {
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-top: 2px solid #f59e0b;
}
.card-amber:hover {
    border-color: rgba(245, 158, 11, 0.6);
    box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.7), 0 0 15px rgba(245, 158, 11, 0.2);
}

.card-red {
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-top: 2px solid #ef4444;
}
.card-red:hover {
    border-color: rgba(239, 68, 68, 0.6);
    box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.7), 0 0 15px rgba(239, 68, 68, 0.2);
}

.card-orange {
    border: 1px solid rgba(249, 115, 22, 0.3);
    border-top: 2px solid #f97316;
}
.card-orange:hover {
    border-color: rgba(249, 115, 22, 0.6);
    box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.7), 0 0 15px rgba(249, 115, 22, 0.2);
}

.card-blue {
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-top: 2px solid #3b82f6;
}
.card-blue:hover {
    border-color: rgba(59, 130, 246, 0.6);
    box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.7), 0 0 15px rgba(59, 130, 246, 0.2);
}

/* Card Header Components */
.card-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
}

.ticker-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.12rem;
    font-weight: 800;
    letter-spacing: 0.5px;
}

/* Z-Score Tiered Pills */
.z-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem;
    font-weight: 700;
    padding: 1px 7px;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
    gap: 3px;
    letter-spacing: 0.3px;
}

/* Tier 4: Z >= 4.0 (Ultra High / Monster Alpha) - Electric Neon Fuchsia */
.z-pill-super {
    background: rgba(217, 70, 239, 0.2);
    border: 1px solid rgba(217, 70, 239, 0.7);
    color: #f0abfc;
    box-shadow: 0 0 10px rgba(217, 70, 239, 0.35);
    text-shadow: 0 0 8px rgba(217, 70, 239, 0.6);
}

/* Tier 3: 3.0 <= Z < 4.0 (Elite Market Leader) - Vivid Cyan */
.z-pill-tier3 {
    background: rgba(14, 165, 233, 0.18);
    border: 1px solid rgba(56, 189, 248, 0.6);
    color: #38bdf8;
    box-shadow: 0 0 8px rgba(56, 189, 248, 0.25);
    text-shadow: 0 0 6px rgba(56, 189, 248, 0.5);
}

/* Tier 2: 2.0 <= Z < 3.0 (Strong Breakout Potential) - Mint Emerald */
.z-pill-tier2 {
    background: rgba(16, 185, 129, 0.14);
    border: 1px solid rgba(16, 185, 129, 0.5);
    color: #6ee7b7;
    box-shadow: 0 0 6px rgba(16, 185, 129, 0.2);
    text-shadow: 0 0 5px rgba(16, 185, 129, 0.4);
}

/* Tier 1: Z < 2.0 (Developing / Moderate / Fading) - Greyed Muted Slate */
.z-pill-tier1 {
    background: rgba(100, 116, 139, 0.12);
    border: 1px solid rgba(100, 116, 139, 0.35);
    color: #94a3b8;
}

/* Unranked / SME */
.z-pill-unranked {
    background: rgba(51, 65, 85, 0.2);
    border: 1px dashed rgba(100, 116, 139, 0.35);
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
}

.rsnh-pill {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.5);
    color: #fbbf24;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 4px;
    letter-spacing: 0.3px;
}

/* Action Badges */
.action-pill {
    font-size: 0.68rem;
    font-weight: 800;
    padding: 2px 7px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    white-space: nowrap;
}

.action-pill-in-zone {
    background: #10b981;
    color: #022c22;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.3);
}

.action-pill-retest {
    background: #3b82f6;
    color: #ffffff;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
}

.action-pill-extended {
    background: #f59e0b;
    color: #451a03;
}

.action-pill-failed {
    background: #ef4444;
    color: #ffffff;
}

.action-pill-amber {
    background: #f59e0b;
    color: #451a03;
    box-shadow: 0 0 10px rgba(245, 158, 11, 0.3);
}

.action-pill-exit {
    background: #ef4444;
    color: #ffffff;
    box-shadow: 0 0 10px rgba(239, 68, 68, 0.3);
}

.action-pill-trim {
    background: #f97316;
    color: #ffffff;
    box-shadow: 0 0 10px rgba(249, 115, 22, 0.3);
}

.action-pill-blue {
    background: rgba(59, 130, 246, 0.2);
    border: 1px solid rgba(59, 130, 246, 0.5);
    color: #60a5fa;
}

/* 2x2 Metrics Micro-Grid */
.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 10px;
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 8px;
    padding: 7px 9px;
    margin: 6px 0;
}

.metric-cell {
    display: flex;
    flex-direction: column;
}

.metric-cell-lbl {
    font-size: 0.64rem;
    text-transform: uppercase;
    color: #64748b;
    font-weight: 700;
    letter-spacing: 0.4px;
}

.metric-cell-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    color: #f1f5f9;
}

/* Card Footer */
.card-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    padding-top: 5px;
    margin-top: 4px;
    font-size: 0.70rem;
    color: #94a3b8;
}

.vol-tag {
    font-family: 'JetBrains Mono', monospace;
    color: #cbd5e1;
}

/* Clickable Terminal Card Links (TradingView Integration) */
a.terminal-card-link {
    text-decoration: none !important;
    color: inherit !important;
    display: block;
    cursor: pointer;
}
a.terminal-card-link:hover,
a.terminal-card-link:focus,
a.terminal-card-link:active,
a.terminal-card-link:visited {
    text-decoration: none !important;
    color: inherit !important;
}

/* TradingView Pill Badge */
.tv-link-badge {
    font-family: 'JetBrains Mono', -apple-system, sans-serif;
    font-size: 0.64rem;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 4px;
    background: rgba(37, 99, 235, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.35);
    color: #60a5fa;
    display: inline-flex;
    align-items: center;
    letter-spacing: 0.3px;
    transition: all 0.2s ease;
}

a.terminal-card-link:hover .tv-link-badge {
    background: rgba(37, 99, 235, 0.4);
    border-color: #60a5fa;
    color: #bfdbfe;
    box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
}

/* ADTV Liquidity Badges */
.adtv-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 4px;
    letter-spacing: 0.3px;
    display: inline-flex;
    align-items: center;
}
.adtv-inst {
    background: rgba(234, 179, 8, 0.15);
    border: 1px solid rgba(234, 179, 8, 0.45);
    color: #facc15;
}
.adtv-liquid {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
}
.adtv-tradable {
    background: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.35);
    color: #93c5fd;
}
.adtv-low {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.35);
    color: #fca5a5;
}

/* Industry & Thematic Cluster Badges */
.theme-pill {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 4px;
    background: rgba(148, 163, 184, 0.12);
    border: 1px solid rgba(148, 163, 184, 0.25);
    color: #cbd5e1;
    display: inline-flex;
    align-items: center;
    max-width: 170px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.theme-cluster {
    background: rgba(249, 115, 22, 0.18);
    border: 1px solid rgba(249, 115, 22, 0.5);
    color: #fb923c;
    font-weight: 700;
}

/* Theme Topper Header Cards */
.theme-leader-card {
    background: rgba(15, 23, 42, 0.75);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
}
.theme-leader-card:hover {
    transform: translateY(-2px);
}
.theme-card-super {
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-top: 3px solid #f59e0b;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.6), 0 0 16px rgba(245, 158, 11, 0.15);
}
.theme-card-super:hover {
    border-color: rgba(245, 158, 11, 0.7);
    box-shadow: 0 12px 28px -4px rgba(0, 0, 0, 0.7), 0 0 22px rgba(245, 158, 11, 0.25);
}
.theme-card-cluster {
    border: 1px solid rgba(14, 165, 233, 0.4);
    border-top: 3px solid #38bdf8;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.6), 0 0 16px rgba(14, 165, 233, 0.12);
}
.theme-card-cluster:hover {
    border-color: rgba(14, 165, 233, 0.7);
    box-shadow: 0 12px 28px -4px rgba(0, 0, 0, 0.7), 0 0 22px rgba(14, 165, 233, 0.2);
}
.theme-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 8px;
    margin-bottom: 8px;
}
.theme-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.3;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.theme-badge-super {
    background: rgba(245, 158, 11, 0.2);
    border: 1px solid rgba(245, 158, 11, 0.6);
    color: #fbbf24;
    font-size: 0.66rem;
    font-weight: 800;
    padding: 2px 7px;
    border-radius: 9999px;
    white-space: nowrap;
    letter-spacing: 0.4px;
}
.theme-badge-cluster {
    background: rgba(14, 165, 233, 0.18);
    border: 1px solid rgba(56, 189, 248, 0.5);
    color: #38bdf8;
    font-size: 0.66rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 9999px;
    white-space: nowrap;
    letter-spacing: 0.3px;
}
.theme-metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 10px;
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 8px;
    padding: 8px 10px;
    margin: 8px 0;
}
.theme-stocks-row {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 6px;
    align-items: center;
}
.theme-stock-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.25);
    color: #e2e8f0;
    text-decoration: none !important;
    transition: all 0.15s ease;
}
.theme-stock-chip:hover {
    background: rgba(37, 99, 235, 0.35);
    border-color: #60a5fa;
    color: #ffffff;
    box-shadow: 0 0 6px rgba(59, 130, 246, 0.4);
}
.theme-stock-chip-buyable {
    border-color: rgba(16, 185, 129, 0.6);
    background: rgba(16, 185, 129, 0.18);
    color: #6ee7b7;
}
</style>
""", unsafe_allow_html=True)

def format_z_badge(z_val, ticker=None):
    if z_val is not None and pd.notna(z_val):
        try:
            zf = float(z_val)
            sign = "+" if zf >= 0 else ""
            if zf >= 4.0:
                return f"<span class='z-pill z-pill-super'>🔥 Z: {sign}{zf:.2f}</span>"
            elif zf >= 3.0:
                return f"<span class='z-pill z-pill-tier3'>⚡ Z: {sign}{zf:.2f}</span>"
            elif zf >= 2.0:
                return f"<span class='z-pill z-pill-tier2'>⚡ Z: {sign}{zf:.2f}</span>"
            else:
                return f"<span class='z-pill z-pill-tier1'>Z: {sign}{zf:.2f}</span>"
        except Exception:
            pass
    if ticker and ("-SM" in str(ticker).upper() or "-ST" in str(ticker).upper()):
        return "<span class='z-pill-unranked'>SME</span>"
    return "<span class='z-pill-unranked'>Unranked</span>"

def format_adtv_badge(adtv_val):
    if adtv_val is not None and pd.notna(adtv_val):
        try:
            val = float(adtv_val)
            if val >= 10.0:
                return f"<span class='adtv-pill adtv-inst' title='Institutional Liquidity: 20d ADTV ≥ ₹10 Cr'>🏛️ ₹{val:.1f} Cr</span>"
            elif val >= 3.0:
                return f"<span class='adtv-pill adtv-liquid' title='Liquid Growth: 20d ADTV ≥ ₹3 Cr'>💧 ₹{val:.1f} Cr</span>"
            elif val >= 1.0:
                return f"<span class='adtv-pill adtv-tradable' title='Tradable Smallcap: 20d ADTV ≥ ₹1 Cr'>🪙 ₹{val:.1f} Cr</span>"
            else:
                return f"<span class='adtv-pill adtv-low' title='Illiquid Microcap: 20d ADTV < ₹1 Cr'>⚠️ ₹{val:.2f} Cr</span>"
        except Exception:
            pass
    return "<span class='adtv-pill adtv-low'>ADTV N/A</span>"

def format_industry_badge(industry, is_cluster=False, cluster_count=0):
    ind = str(industry or "").strip()
    if not ind or ind in ["Unknown", "None", "nan"]:
        return ""
    if is_cluster and cluster_count >= 3:
        return f"<span class='theme-pill theme-cluster' title='Thematic Cluster: {cluster_count} leaders in {ind}'>🔥 {ind} ({cluster_count})</span>"
    return f"<span class='theme-pill' title='Industry: {ind}'>🏭 {ind}</span>"

def get_tradingview_url(ticker: str) -> str:
    clean = str(ticker).replace('.NS', '').replace('.BO', '').strip()
    if '-' in clean:
        clean = clean.split('-')[0]
    if clean.isdigit():
        return f"https://in.tradingview.com/chart/?symbol=BSE:{clean}"
    return f"https://in.tradingview.com/chart/?symbol=NSE:{clean}"



terminal_mode = st.radio(
    "Select Operating Mode:",
    [
        "⚡ Active CANSLIM Swing Trader (Daily Signals & 6-8 Leaders)",
        "📅 Monthly Passive Robo-Advisor (Top 15 Rebalance)"
    ],
    index=0
)

# =============================================================================
# MODE 1: ACTIVE CANSLIM SWING TRADER
# =============================================================================

if "Active CANSLIM Swing Trader" in terminal_mode:
    render_header(
        "⚡ Fast Momentum: CANSLIM Swing Terminal",
        "High-octane active execution engine for the NIFTY 750 universe. Generates precision Buy breakouts, Pyramiding add-ons, and 2-close 21 EMA exits based on William O'Neil, David Ryan, and Jim Roppel."
    )

    with st.expander("📖 CANSLIM Swing Trading Rules & Playbook (O'Neil / Ryan / Roppel)", expanded=False):
        st.markdown(r"""
        ### 🏆 The 5-Layer Execution Playbook
        
        1. **Market Direction (The 'M' in CANSLIM):**
           - We track Distribution Days (DD) and Follow-Through Days (FTD) on the benchmark (Nifty 500 / Nifty 50).
           - When Distribution Days $\ge 5$, all new breakout buying is **halted** and exposure is dialed down to protect capital.
           
        2. **How the Top 50 Momentum Leaders Are Selected (The Math & Algorithm):**
           - **Universe:** Strictly filters to the **NIFTY 750** (Nifty Total Market covering 750 liquid large, mid, small, and microcaps).
           - **Step A: Volatility-Adjusted Momentum Ratios:**
             For every stock in the 750 universe, we compute trailing returns divided by rolling 1-year (252-day) annualized volatility:
             $$\text{Momentum Ratio} = \frac{\text{Return}}{\text{Annualized Volatility}_{252d}}$$
             Calculated across 3 distinct swing horizons:
             * **1-Month Momentum (21 days):** $\text{MR}_{1M} = R_{1M} / \sigma_{252d}$
             * **3-Month Momentum (63 days):** $\text{MR}_{3M} = R_{3M} / \sigma_{252d}$
             * **6-Month Momentum (126 days):** $\text{MR}_{6M} = R_{6M} / \sigma_{252d}$
             *(Dividing by volatility filters out noisy, low-liquidity spikes and heavily rewards smooth, persistent institutional accumulation).*
           - **Step B: Cross-Sectional Z-Score Standardization:**
             Each day across all 750 stocks, the momentum ratios are standardized into normal distributions ($Z = \frac{X - \mu_{750}}{\sigma_{750}}$), producing cross-sectional scores: $Z_{1M}$, $Z_{3M}$, and $Z_{6M}$.
           - **Step C: Balanced Time-Weighting:**
             To capture responsive breakout momentum while maintaining robust intermediate trend stability, the weighting scale is balanced:
             $$\text{Weighted } Z = (40\% \times Z_{1M}) + (40\% \times Z_{3M}) + (20\% \times Z_{6M})$$
           - **Step D: Top 50 Elite Radar:**
             All 750 stocks are ranked descending by their Weighted Z-Score. **The Top 50 highest-scoring stocks form the active CANSLIM pool** that feeds into the Sales Growth gate ($\ge 25\%$), chart base setups, buy breakout triggers, and pyramiding scans.
           
        3. **Fundamental Gate (C & A):**
           - Minimum **+25% Sales Growth YoY** (Top-line revenue expansion gate; EPS and ROE requirements removed).
           
        4. **Precision Entry Tactics & 5% Buy Zone:**
           - **Stage 1/2 Base Breakout:** Breaking out above a valid consolidation on volume $\ge 1.4\times$ average.
           - **David Ryan RS Line New High (RSNH):** RS Line breaking out to a 52-week high *before* the stock breaks out.
           - **Jim Roppel 21 EMA Support Bounce:** Pullback to the rising 21 EMA with Volume Dry-Up (VDU) followed by a reversal candle.
           - **Strict Buy Zone:** Never chase a stock $>5\%$ above its pivot.
           
        5. **Disciplined Exits & Pyramiding (Averaging Up):**
           - **Averaging Up (Pyramiding):** When a position is up $\ge +2.5\%$ and testing the 21 EMA on low volume, add a 30% secondary tranche and **elevate the stop-loss to a higher low** to protect profits.
           - **Trailing Swing Stop:** **Two consecutive daily closes below the 21 EMA** signals definitive loss of short-term momentum and triggers a full exit.
           - **O'Neil 8-Week Hold Rule:** If a breakout generates $+20\%$ in $<3$ weeks, hold for a minimum of 8 weeks (protecting through 21 EMA dips, using the 50 SMA as the floor).
           - **Initial Stop-Loss:** Low of the breakout day (3% to 5%, hard emergency ceiling at 7.5%).
        """)

    # Universe Selector Toggle
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 12px;
    ">
        <span style="color: rgba(255,255,255,0.5); font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px;">
            📡 Scan Universe
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    u_col1, u_col2 = st.columns([4, 1])
    with u_col1:
        universe_choice = st.radio(
            "Select stock universe for momentum scoring:",
            [
                "⚡ NIFTY 750 (Default — Large, Mid & Small Caps)",
                "🌐 All India Deep Market (2500+ NSE Stocks)"
            ],
            index=0,
            horizontal=True,
            key="mode1_universe"
        )
    universe_mode = "deep_market" if "2500+" in universe_choice else "nifty_750"
    
    if universe_mode == "deep_market":
        st.info("🌐 **Deep Market Mode:** Computing Z-scores across 2500+ NSE stocks. First load may take 30–60 seconds while the momentum matrix is built. Z-scores are cross-sectional against the entire Indian equity universe.", icon="⏳")

    # Portfolio Source Toggle
    t_col1, t_col2 = st.columns([4, 1])
    with t_col1:
        portfolio_source = st.radio(
            "Holdings Source to Evaluate for Pyramiding & Exits:",
            [
                "💼 My Personal Google Sheet Portfolio (Live Holdings)",
                "🤖 Autonomous Swing Model Portfolio (Top 6–8 Leaders)"
            ],
            index=0,
            horizontal=True
        )
    with t_col2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Force Refresh", help="Clear cached computations and recalculate fresh swing signals"):
            st.cache_data.clear()
            st.rerun()

    use_personal = "Personal Google Sheet" in portfolio_source

    # Cache CANSLIM swing state computation with cache-busting key
    @st.cache_data(ttl=600, show_spinner=False)
    def load_canslim_swing_state_cached(is_personal: bool, cache_key: str = "v5_40_40_20_z", _universe_mode: str = "nifty_750"):
        try:
            positions = None
            if is_personal:
                from portfolio_fetcher import fetch_portfolio_data
                df = fetch_portfolio_data()
                if df is not None and not df.empty:
                    positions = []
                    for _, row in df.iterrows():
                        t = str(row['ticker']).strip().upper()
                        if not t or t in ['GOLDBEES', 'LIQUIDBEES', 'LIQUIDCASE', 'LIQUIDETF']:
                            continue
                        try:
                            positions.append({
                                'ticker': t,
                                'avg_buy_price': float(row['Avg Buy Price']),
                                'quantity': float(row['Qty']),
                                'buy_date': 'Google Sheet',
                                'is_8_week_hold': False
                            })
                        except Exception:
                            continue
            return generate_daily_canslim_swing_state(portfolio_positions=positions, universe_mode=_universe_mode)
        except Exception as e:
            st.error(f"Error computing CANSLIM swing state: {e}")
            return None

    universe_label = "Deep Market (2500+ NSE)" if universe_mode == "deep_market" else "NIFTY 750"
    spinner_msg = f"⚡ Running CANSLIM momentum filters across {universe_label}..."
    with st.spinner(spinner_msg):
        swing_state = load_canslim_swing_state_cached(use_personal, cache_key=f"v6_{universe_mode}", _universe_mode=universe_mode)

    if not swing_state:
        st.error("Failed to generate swing trading state. Please check network connection or market data cache.")
        st.stop()

    regime = swing_state['market_regime']
    buys = swing_state['buys_today']
    pyramids = swing_state['pyramids_today']
    exits = swing_state['exits_today']
    radar = swing_state['radar_setups']
    leaderboard = swing_state['leaderboard']

    # --- 1. MARKET REGIME BANNER ---
    regime_color = "#10b981" if "Uptrend" in regime['label'] and regime['distribution_days'] < 5 else ("#f59e0b" if regime['distribution_days'] < 5 else "#ef4444")
    regime_icon = "🚀" if "Uptrend" in regime['label'] and regime['distribution_days'] < 4 else ("⚠️" if regime['distribution_days'] < 5 else "🔻")

    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-left: 6px solid {regime_color};
        padding: 16px 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    ">
        <div>
            <div style="font-size: 0.85rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                CANSLIM Market Direction ('M')
                <span style="background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(59, 130, 246, 0.4); color: #60a5fa; padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; text-transform: none; font-weight: normal;">📡 {universe_label}</span>
            </div>
            <div style="font-size: 1.4rem; font-weight: bold; color: #f8fafc; margin-top: 4px;">{regime_icon} {regime['label']}</div>
        </div>
        <div style="display: flex; gap: 24px; align-items: center;">
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; color: #94a3b8;">Distribution Days</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: {'#ef4444' if regime['distribution_days'] >= 5 else ('#f59e0b' if regime['distribution_days'] >= 3 else '#10b981')};">{regime['distribution_days']} Days</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; color: #94a3b8;">Follow-Through Day</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: {'#10b981' if regime['ftd_active'] else '#94a3b8'};">{'🟢 Active' if regime['ftd_active'] else '⚪ Neutral'}</div>
            </div>
            <div style="text-align: right; border-left: 1px solid rgba(255,255,255,0.1); padding-left: 20px;">
                <div style="font-size: 0.8rem; color: #94a3b8;">Allowed Exposure</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: {regime_color};">{regime['recommended_exposure_pct']}% Risk-On</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- 2. LIVE ACTION BOARD (4 PANELS) ---
    st.markdown("## 🎯 Live Action Board for Today")
    st.caption(f"Daily dynamic swing trading signals based on the {universe_label} cross-sectional momentum leaders.")

    # Breakout Trigger Controls Bar (Aligned above 4 action columns)
    f_c1, f_c2, f_c3, f_c4, f_c5 = st.columns([1.8, 1.9, 1.8, 1.4, 3.1])
    with f_c1:
        buy_lookback = st.selectbox(
            "Trigger Lookback:",
            ["Past 10 Days (2 Weeks)", "Past 5 Days (1 Week)", "Today Only"],
            index=0,
            key="buy_lookback_sel"
        )
    with f_c2:
        card_sort_order = st.selectbox(
            "Sort Cards By:",
            ["🔥 Highest Z-Score First", "📅 Trigger Date (Recent First)", "🎯 Closest to Base Pivot"],
            index=0,
            key="card_sort_order_sel",
            help="Sort candidate tiles across all panels by Momentum Z-Score, Recency, or Pivot Proximity."
        )
    with f_c3:
        adtv_filter = st.selectbox(
            "Liquidity Gate (ADTV):",
            [
                "🪙 All Tradable (≥ ₹1 Cr)",
                "💧 Liquid Growth (≥ ₹3 Cr)",
                "🏛️ Institutional (≥ ₹10 Cr)",
                "⚠️ Unfiltered (Show All)"
            ],
            index=0,
            key="adtv_filter_sel",
            help="Filters out illiquid stocks based on 20-Day Average Daily Turnover in ₹ Crores."
        )
    with f_c4:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        only_actionable = st.checkbox(
            "In Buy Zone",
            value=True,
            key="buy_act_filter",
            help="Filters to stocks currently sitting within the 5% Buy Zone or Retest Zone."
        )
    with f_c5:
        st.markdown("""
        <div style="display: flex; gap: 5px; align-items: center; justify-content: flex-end; margin-top: 30px; flex-wrap: wrap;">
            <span style="font-size: 0.68rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Tiers:</span>
            <span class="z-pill z-pill-super" style="font-size: 0.62rem;">🔥 Z≥4</span>
            <span class="z-pill z-pill-tier3" style="font-size: 0.62rem;">⚡ Z≥3</span>
            <span class="adtv-pill adtv-inst" style="font-size: 0.62rem;">🏛️ ≥10Cr</span>
            <span class="adtv-pill adtv-liquid" style="font-size: 0.62rem;">💧 ≥3Cr</span>
            <span class="theme-pill theme-cluster" style="font-size: 0.62rem;">🔥 Cluster</span>
        </div>
        """, unsafe_allow_html=True)

    if "≥ ₹10 Cr" in adtv_filter:
        min_adtv = 10.0
    elif "≥ ₹3 Cr" in adtv_filter:
        min_adtv = 3.0
    elif "≥ ₹1 Cr" in adtv_filter:
        min_adtv = 1.0
    else:
        min_adtv = 0.0

    def _passes_adtv(item):
        if min_adtv <= 0.0:
            return True
        adtv = item.get('adtv_cr', 0.0)
        try:
            return float(adtv or 0.0) >= min_adtv
        except Exception:
            return False

    recent_buys = swing_state.get('recent_buy_triggers', [])
    max_days = 10 if "10" in buy_lookback else (5 if "5" in buy_lookback else 0)
    display_buys = [b for b in recent_buys if b['days_ago'] <= max_days and _passes_adtv(b)]
    if only_actionable:
        display_buys = [b for b in display_buys if b['actionable']]

    pyramids = [p for p in swing_state['pyramids_today'] if _passes_adtv(p)]
    exits = [e for e in swing_state['exits_today'] if _passes_adtv(e)]
    radar = [r for r in swing_state['radar_setups'] if _passes_adtv(r)]

    def _safe_sort_z(item):
        z = item.get('z_score')
        if z is not None and pd.notna(z):
            try:
                return float(z)
            except Exception:
                return -999.0
        return -999.0

    if card_sort_order == "🔥 Highest Z-Score First":
        display_buys = sorted(display_buys, key=_safe_sort_z, reverse=True)
        pyramids = sorted(pyramids, key=_safe_sort_z, reverse=True)
        exits = sorted(exits, key=_safe_sort_z, reverse=True)
        radar = sorted(radar, key=_safe_sort_z, reverse=True)
    elif card_sort_order == "🎯 Closest to Base Pivot":
        display_buys = sorted(display_buys, key=lambda b: abs(float(b.get('dist_now_pct', 999.0))))
        radar = sorted(radar, key=lambda r: abs(float(r.get('dist_from_pivot_pct', 999.0))))
    else:  # "📅 Trigger Date (Recent First)"
        display_buys = sorted(display_buys, key=lambda b: (b.get('days_ago', 99), -_safe_sort_z(b)))

    col1, col2, col3, col4 = st.columns(4)

    # PANEL 1: BUY SIGNALS & MULTI-DAY BREAKOUT TRIGGER RADAR
    with col1:
        st.markdown(f"### 🟢 Buy Triggers ({len(display_buys)})")
        st.caption(fr"Breakouts & 21 EMA bounces with Sales $\ge 25\%$.")
        
        if display_buys:
            for b in display_buys:
                is_act = b['actionable']
                is_retest = "Retesting" in b['status']
                is_ext = "Extended" in b['status']
                
                border_c = "#10b981" if is_act and not is_retest else ("#3b82f6" if is_retest else ("#f59e0b" if is_ext else "#ef4444"))
                card_style_class = "card-green" if (is_act and not is_retest) else ("card-blue" if is_retest else ("card-amber" if is_ext else "card-red"))
                badge_lbl = "IN BUY ZONE" if (is_act and not is_retest) else ("RETEST" if is_retest else ("EXTENDED" if is_ext else "FAILED"))
                badge_class = "in-zone" if (is_act and not is_retest) else ("retest" if is_retest else ("extended" if is_ext else "failed"))
                
                timing_lbl = "Today" if b['days_ago'] == 0 else f"{b['days_ago']}d ago ({b['trigger_date']})"
                z_badge = format_z_badge(b.get('z_score'), b.get('clean_ticker'))
                adtv_badge = format_adtv_badge(b.get('adtv_cr'))
                theme_badge = format_industry_badge(b.get('industry'), b.get('is_thematic_cluster'), b.get('cluster_count'))
                theme_html = f"<div style='margin-top: 3px; margin-bottom: 6px;'>{theme_badge}</div>" if theme_badge else ""
                dist_color = "#34d399" if b['dist_now_pct'] >= 0 else "#f87171"
                tv_url = get_tradingview_url(b['clean_ticker'])
                
                card_html = f"""<a href="{tv_url}" target="_blank" rel="noopener noreferrer" class="terminal-card-link" title="Open {b['clean_ticker']} interactive chart on TradingView">
<div class="terminal-card {card_style_class}">
<div class="card-header-row">
<div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
<span class="ticker-title" style="color: {border_c};">{b['clean_ticker']}</span>
<span class="tv-link-badge">TV ↗</span>
{z_badge}
{adtv_badge}
</div>
<span class="action-pill action-pill-{badge_class}">{badge_lbl}</span>
</div>
{theme_html}
<div class="metrics-grid">
<div class="metric-cell">
<span class="metric-cell-lbl">CMP (vs Pivot)</span>
<span class="metric-cell-val">₹{b['current_price']:.2f} <span style="font-size: 0.72rem; color: {dist_color};">({b['dist_now_pct']:+.1f}%)</span></span>
</div>
<div class="metric-cell">
<span class="metric-cell-lbl">Base Pivot</span>
<span class="metric-cell-val">₹{b['pivot_price']:.2f}</span>
</div>
<div class="metric-cell">
<span class="metric-cell-lbl">Stop-Loss (Risk)</span>
<span class="metric-cell-val" style="color: #f87171;">₹{b['stop_loss']:.2f} <span style="font-size: 0.72rem; color: #fca5a5;">(-{b['risk_pct']}%)</span></span>
</div>
<div class="metric-cell">
<span class="metric-cell-lbl">Sales Growth YoY</span>
<span class="metric-cell-val" style="color: #34d399;">+{b['sales_growth']:.1f}%</span>
</div>
</div>
<div class="card-footer">
<span>⏱️ <b>Triggered:</b> {timing_lbl}</span>
<span class="vol-tag">📊 <b>Vol:</b> {b['volume_ratio']}x</span>
</div>
</div>
</a>"""
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            render_empty_state(f"No triggers matching {buy_lookback}.", icon="🛡️")

    # PANEL 2: PYRAMIDING / AVERAGING UP
    with col2:
        st.markdown(f"### ⚡ Averaging Up ({len(pyramids)})")
        p_caption = r"Holdings from your Google Sheet (gain $\ge +2.5\%$) coiling near 21 EMA on VDU." if use_personal else r"Working winners ($\ge +2.5\%$) coiling near 21 EMA. Raised stop protects gains."
        st.caption(p_caption)
        if pyramids:
            for p in pyramids:
                z_badge = format_z_badge(p.get('z_score'), p.get('ticker'))
                adtv_badge = format_adtv_badge(p.get('adtv_cr'))
                theme_badge = format_industry_badge(p.get('industry'))
                theme_html = f"<div style='margin-top: 3px; margin-bottom: 6px;'>{theme_badge}</div>" if theme_badge else ""
                tv_url = get_tradingview_url(p['ticker'])
                card_html = f"""<a href="{tv_url}" target="_blank" rel="noopener noreferrer" class="terminal-card-link" title="Open {p['ticker']} interactive chart on TradingView">
<div class="terminal-card card-amber">
<div class="card-header-row">
<div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
<span class="ticker-title" style="color: #fbbf24;">{p['ticker']}</span>
<span class="tv-link-badge">TV ↗</span>
{z_badge}
{adtv_badge}
</div>
<span class="action-pill action-pill-amber">ADD +30%</span>
</div>
{theme_html}
<div class="metrics-grid">
<div class="metric-cell">
<span class="metric-cell-lbl">CMP (Gain)</span>
<span class="metric-cell-val">₹{p['current_price']:.2f} <span style="font-size: 0.72rem; color: #34d399;">(+{p['current_gain_pct']}%)</span></span>
</div>
<div class="metric-cell">
<span class="metric-cell-lbl">Avg Buy Cost</span>
<span class="metric-cell-val">₹{p['avg_cost']:.2f}</span>
</div>
<div class="metric-cell">
<span class="metric-cell-lbl">Raised Stop (Locked)</span>
<span class="metric-cell-val" style="color: #34d399;">₹{p['new_raised_stop']:.2f} <span style="font-size: 0.72rem;">(+{p['risk_locked_in']}%)</span></span>
</div>
<div class="metric-cell">
<span class="metric-cell-lbl">Dist vs 21 EMA</span>
<span class="metric-cell-val">+{p['extension_from_21ema']}%</span>
</div>
</div>
<div class="card-footer">
<span>⚡ <b>Trigger:</b> {p['trigger_reason']}</span>
<span class="vol-tag">🎯 2nd Tranche</span>
</div>
</div>
</a>"""
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            render_empty_state("No open positions currently qualifying for secondary pyramiding adds.", icon="⏳")

    # PANEL 3: SELL & STOP ALERTS
    with col3:
        st.markdown(f"### 🔴 Sell / Stop Alerts ({len(exits)})")
        e_caption = "Holdings from your Google Sheet with 2 closes below 21 EMA or stop-loss breaches." if use_personal else "2 Closes Below 21 EMA, initial stop breaches, or Climax blow-offs."
        st.caption(e_caption)
        if exits:
            for e in exits:
                is_stop = "STOP" in e['badge']
                card_style_class = "card-red" if is_stop else "card-orange"
                pill_class = "exit" if is_stop else "trim"
                title_col = "#f87171" if is_stop else "#fb923c"
                z_badge = format_z_badge(e.get('z_score'), e.get('ticker'))
                adtv_badge = format_adtv_badge(e.get('adtv_cr'))
                theme_badge = format_industry_badge(e.get('industry'))
                theme_html = f"<div style='margin-top: 3px; margin-bottom: 6px;'>{theme_badge}</div>" if theme_badge else ""
                pnl_col = "#34d399" if e['pnl_pct'] >= 0 else "#f87171"
                tv_url = get_tradingview_url(e['ticker'])
                card_html = f"""<a href="{tv_url}" target="_blank" rel="noopener noreferrer" class="terminal-card-link" title="Open {e['ticker']} interactive chart on TradingView">
<div class="terminal-card {card_style_class}">
<div class="card-header-row">
<div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
<span class="ticker-title" style="color: {title_col};">{e['ticker']}</span>
<span class="tv-link-badge">TV ↗</span>
{z_badge}
{adtv_badge}
</div>
<span class="action-pill action-pill-{pill_class}">{e['action']}</span>
</div>
{theme_html}
<div style="font-size: 0.74rem; font-weight: 700; color: {'#fca5a5' if is_stop else '#fdba74'}; margin-bottom: 4px;">{e['badge']}</div>
<div class="metrics-grid">
<div class="metric-cell">
<span class="metric-cell-lbl">Current Price</span>
<span class="metric-cell-val">₹{e['current_price']:.2f}</span>
</div>
<div class="metric-cell">
<span class="metric-cell-lbl">P&L</span>
<span class="metric-cell-val" style="color: {pnl_col};">{e['pnl_pct']:+.1f}%</span>
</div>
<div class="metric-cell">
<span class="metric-cell-lbl">Current Stop</span>
<span class="metric-cell-val">₹{e['current_stop']:.2f}</span>
</div>
<div class="metric-cell">
<span class="metric-cell-lbl">Closes &lt; 21 EMA</span>
<span class="metric-cell-val" style="color: {'#ef4444' if e['consecutive_below_21ema']>=2 else '#f59e0b'};">{e['consecutive_below_21ema']} Days</span>
</div>
</div>
<div class="card-footer">
<span>⚠️ <b>Reason:</b> {e['reason']}</span>
</div>
</div>
</a>"""
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            render_empty_state("All active positions are healthy and holding above their 21 EMA / stop levels.", icon="✅")

    # PANEL 4: SWING RADAR (DEVELOPING BASES)
    with col4:
        st.markdown(f"### 🔵 Swing Radar ({len(radar)})")
        st.caption(r"Pre-breakout setups coiling within $\le 4\%$ of pivot holding 21 EMA.")
        if radar:
            for r in radar:
                rsnh_html = f"<span class='rsnh-pill'>👑 RSNH</span> " if r['rsnh']['rsnh_active'] else ""
                z_badge = format_z_badge(r.get('z_score'), r.get('clean_ticker'))
                adtv_badge = format_adtv_badge(r.get('adtv_cr'))
                theme_badge = format_industry_badge(r.get('industry'), r.get('is_thematic_cluster'), r.get('cluster_count'))
                theme_html = f"<div style='margin-top: 3px; margin-bottom: 6px;'>{theme_badge}</div>" if theme_badge else ""
                sales_g = r['fundamentals'].get('sales_growth', 0.0)
                tv_url = get_tradingview_url(r['clean_ticker'])
                card_html = f"""<a href="{tv_url}" target="_blank" rel="noopener noreferrer" class="terminal-card-link" title="Open {r['clean_ticker']} interactive chart on TradingView">
<div class="terminal-card card-blue">
<div class="card-header-row">
<div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
<span class="ticker-title" style="color: #60a5fa;">{r['clean_ticker']}</span>
<span class="tv-link-badge">TV ↗</span>
{z_badge}
{adtv_badge}
</div>
<div style="display: flex; gap: 4px; align-items: center;">{rsnh_html}<span class="action-pill action-pill-blue">COILING</span></div>
</div>
{theme_html}
<div class="metrics-grid">
<div class="metric-cell">
<span class="metric-cell-lbl">CMP</span>
<span class="metric-cell-val">₹{r['current_price']:.2f}</span>
</div>
<div class="metric-cell">
<span class="metric-cell-lbl">Base Pivot (Dist)</span>
<span class="metric-cell-val">₹{r['pivot_price']:.2f} <span style="font-size: 0.72rem; color: #94a3b8;">({r['dist_from_pivot_pct']:+.1f}%)</span></span>
</div>
<div class="metric-cell">
<span class="metric-cell-lbl">Base Stage</span>
<span class="metric-cell-val">{r['base_stage']}</span>
</div>
<div class="metric-cell">
<span class="metric-cell-lbl">Sales Growth YoY</span>
<span class="metric-cell-val" style="color: #34d399;">+{sales_g:.1f}%</span>
</div>
</div>
<div class="card-footer">
<span>⏳ <b>Volume:</b> {'VDU (Low Vol)' if r['vdu_detected'] else 'Normal'}</span>
<span class="vol-tag">📡 In Base (≤4% Pivot)</span>
</div>
</div>
</a>"""
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            render_empty_state("No setups currently coiling inside base thresholds.", icon="🔍")

    st.markdown("---")

    # --- 2.5 2-WEEK BREAKOUT ARCHIVE & SIGNAL JOURNAL ---
    with st.expander("📅 2-Week Breakout Journal & Trigger Archive (Past 10 Trading Sessions)", expanded=False):
        st.caption("Complete chronological record of all institutional base breakouts and 21 EMA bounces detected across the NIFTY 750 universe over the last 10 trading sessions (2 calendar weeks). Breakouts are also permanently archived in SQLite (`signal_history`).")
        recent_all = swing_state.get('recent_buy_triggers', [])
        if min_adtv > 0.0:
            recent_all = [b for b in recent_all if _passes_adtv(b)]
        if recent_all:
            if card_sort_order == "🔥 Highest Z-Score First":
                recent_all = sorted(recent_all, key=_safe_sort_z, reverse=True)
            elif card_sort_order == "🎯 Closest to Base Pivot":
                recent_all = sorted(recent_all, key=lambda b: abs(float(b.get('dist_now_pct', 999.0))))
            else:
                recent_all = sorted(recent_all, key=lambda b: (b.get('days_ago', 99), -_safe_sort_z(b)))
            archive_rows = []
            for b in recent_all:
                z_val = b.get('z_score')
                z_str = f"+{float(z_val):.2f}" if (z_val is not None and pd.notna(z_val)) else "SME / Unranked"
                ind = b.get('industry', 'Unknown')
                if b.get('is_thematic_cluster'):
                    ind_display = f"🔥 {ind} ({b.get('cluster_count', 3)})"
                else:
                    ind_display = ind
                archive_rows.append({
                    'Ticker': b['clean_ticker'],
                    'Chart': get_tradingview_url(b['clean_ticker']),
                    'Industry': ind_display,
                    'ADTV (₹ Cr)': f"₹{b.get('adtv_cr', 0.0):.1f} Cr",
                    'Z-Score': z_str,
                    'Trigger Date': b['trigger_date'],
                    'Days Ago': f"{b['days_ago']}d ago" if b['days_ago'] > 0 else "Today",
                    'Setup Type': b['setup_type'],
                    'Pivot Price': f"₹{b['pivot_price']:.2f}",
                    'Current Price (CMP)': f"₹{b['current_price']:.2f}",
                    'Dist from Pivot': f"{b['dist_now_pct']:+.2f}%",
                    'Breakout Vol Ratio': f"{b['volume_ratio']:.1f}x",
                    'Stop-Loss': f"₹{b['stop_loss']:.2f}",
                    'Sales YoY': f"+{b['sales_growth']:.1f}%",
                    'Current Status': b['status'],
                    'Actionable Now': "✅ YES (Buyable)" if b['actionable'] else "❌ No"
                })
            df_archive = pd.DataFrame(archive_rows)
            st.dataframe(
                df_archive,
                column_config={
                    "Chart": st.column_config.LinkColumn("Chart", display_text="Open TV ↗"),
                    "Industry": st.column_config.TextColumn("Industry / Theme")
                },
                use_container_width=True
            )
        else:
            st.info("No recent breakout signals recorded in the last 10 trading sessions.")

    st.markdown("---")

    # --- 3. ACTIVE PORTFOLIO HOLDINGS STATUS & SWING DECISIONS ---
    active_status = swing_state.get('active_portfolio_status', [])
    pyramid_tickers = {p['ticker'] for p in pyramids}
    
    if use_personal:
        st.markdown(f"## 💼 My Personal Portfolio Holdings ({len(active_status)} Stocks)")
        st.caption("Live holdings loaded from your Google Sheet (`RISK MANAGEMENT`). Evaluated against CANSLIM swing rules: 21 EMA institutional support, Volume Dry-Up (VDU) pyramiding, and 2-close 21 EMA trailing stops.")
    else:
        st.markdown("## 💼 CANSLIM Swing Model Portfolio (Top 6–8 Leaders)")
        st.caption("Autonomous systematic model tracking the top 6–8 qualifying CANSLIM leaders from the NIFTY 750 universe (12.5% to 16.6% target allocation). Exits triggered when 2 consecutive daily closes occur below the 21 EMA.")

    if active_status:
        portfolio_rows = []
        plot_holdings = []
        z_map = swing_state.get('z_score_map', {})
        for s in active_status:
            t = s['ticker']
            z_val = s.get('z_score', z_map.get(t, None))
            try:
                numeric_z = float(z_val) if (z_val is not None and pd.notna(z_val)) else 0.0
            except Exception:
                numeric_z = 0.0
            z_str = f"+{numeric_z:.2f}" if (z_val is not None and pd.notna(z_val)) else "SME / Unranked"
            
            if t in pyramid_tickers:
                act_display = "⚡ ADD +30%"
            elif s['action'] == 'EXIT':
                act_display = "🔴 EXIT"
            elif s['action'] == 'TRIM':
                act_display = "🟠 TRIM"
            elif s['consecutive_below_21ema'] == 1:
                act_display = "⚠️ WATCH (1 Close < 21)"
            else:
                act_display = "🟢 HOLD"
                
            entry_lbl = "Avg Buy Price" if use_personal else "Model Pivot"
            pnl_lbl = "P&L % (vs Cost)" if use_personal else "P&L % (vs Pivot)"
            pnl_val = float(s.get('pnl_pct', 0.0))
            clean_sym = str(t).replace('.NS', '').replace('.BO', '').split('-')[0].strip()
            
            portfolio_rows.append({
                'Ticker': t,
                'Chart': get_tradingview_url(t),
                'Industry': s.get('industry', 'Unknown'),
                'ADTV (₹ Cr)': f"₹{s.get('adtv_cr', 0.0):.1f} Cr" if s.get('adtv_cr') else "N/A",
                'Z-Score': z_str,
                'Current Price': f"₹{s['current_price']:.2f}",
                entry_lbl: f"₹{s.get('avg_buy_price', s['current_price']):.2f}",
                pnl_lbl: f"{pnl_val:+.2f}%",
                'Trailing Stop': f"₹{s['current_stop']:.2f}",
                'Status vs 21 EMA': s['badge'],
                'Closes < 21 EMA': s['consecutive_below_21ema'],
                'Action Today': act_display
            })
            
            plot_holdings.append({
                'clean_ticker': clean_sym,
                'ticker': t,
                'industry': s.get('industry', 'Unknown'),
                'adtv_cr': float(s.get('adtv_cr', 0.0) or 0.0),
                'z_score': numeric_z,
                'pnl_pct': pnl_val,
                'current_price': float(s.get('current_price', 0.0)),
                'current_stop': float(s.get('current_stop', 0.0)),
                'action': act_display,
                'status': s.get('badge', ''),
                'bubble_size': max(min(abs(pnl_val) * 1.2, 25.0), 6.0)
            })
        df_active = pd.DataFrame(portfolio_rows)
        df_h_plot = pd.DataFrame(plot_holdings)
        
        tab_h_quadrant, tab_h_table = st.tabs(["🔭 Roppel Holdings Matrix", "📋 Position Ledger Table"])
        
        with tab_h_quadrant:
            st.caption("Visualizing active portfolio holdings: **X-Axis = Momentum Z-Score**, **Y-Axis = Unrealized P&L (%)**. Apex compounders live in the top-right quadrant ($Z \\ge 2.0$, $\\text{P&L} > 0\\%$). Bubble size reflects P&L magnitude.")
            
            if not df_h_plot.empty:
                max_z = float(df_h_plot['z_score'].max())
                min_z = float(df_h_plot['z_score'].min())
                max_pnl = float(df_h_plot['pnl_pct'].max())
                min_pnl = float(df_h_plot['pnl_pct'].min())
                
                fig_h = px.scatter(
                    df_h_plot,
                    x='z_score',
                    y='pnl_pct',
                    color='action',
                    size='bubble_size',
                    hover_name='clean_ticker',
                    text='clean_ticker',
                    custom_data=['clean_ticker', 'action', 'current_price', 'current_stop', 'pnl_pct', 'z_score', 'industry', 'adtv_cr'],
                    color_discrete_map={
                        '⚡ ADD +30%': '#fbbf24',
                        '🟢 HOLD': '#10b981',
                        '⚠️ WATCH (1 Close < 21)': '#f59e0b',
                        '🟠 TRIM': '#f97316',
                        '🔴 EXIT': '#ef4444'
                    },
                    labels={
                        'z_score': 'Momentum Z-Score (40% 1M, 40% 3M, 20% 6M)',
                        'pnl_pct': 'Unrealized P&L (%)',
                        'action': 'Action Today'
                    }
                )
                fig_h.update_traces(
                    textposition='top center',
                    marker=dict(line=dict(width=1.5, color='rgba(255, 255, 255, 0.4)')),
                    textfont=dict(color='#f1f5f9', size=11, family="JetBrains Mono, Inter"),
                    hovertemplate="<b>%{hovertext}</b><br>Industry: %{customdata[6]}<br>ADTV: ₹%{customdata[7]:.1f} Cr<br>Z-Score: %{customdata[5]:+.2f}<br>P&L: %{customdata[4]:+.2f}%<br>Action: %{customdata[1]}<br>CMP: ₹%{customdata[2]:.2f}<br>Stop: ₹%{customdata[3]:.2f}"
                )
                
                # Crosshairs
                fig_h.add_hline(y=0, line_dash="dash", line_color="rgba(255, 255, 255, 0.35)", line_width=1.5)
                fig_h.add_vline(x=2.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.35)", line_width=1.5)
                
                # Shaded Apex Zone (Z >= 2.0, P&L > 0)
                zone_x1 = max(max_z * 1.15, 4.5)
                zone_y1 = max(max_pnl * 1.2, 15.0)
                zone_y0_neg = min(min_pnl * 1.2, -10.0)
                zone_x0_neg = min(min_z - 0.5, -0.5)
                
                fig_h.add_shape(
                    type="rect",
                    x0=2.0, y0=0, x1=zone_x1, y1=zone_y1,
                    fillcolor="rgba(16, 185, 129, 0.08)",
                    line=dict(color="rgba(16, 185, 129, 0.35)", width=1, dash="dot"),
                    layer="below"
                )
                fig_h.add_annotation(
                    x=(2.0 + zone_x1) / 2, y=zone_y1 * 0.95,
                    text="👑 Apex Compounders (Pyramid Zone)",
                    showarrow=False,
                    font=dict(color="#34d399", size=13, weight="bold")
                )
                fig_h.add_annotation(
                    x=(zone_x0_neg + 2.0) / 2, y=zone_y1 * 0.95,
                    text="🛡️ Fading Winners (Watch 21 EMA)",
                    showarrow=False,
                    font=dict(color="#94a3b8", size=11)
                )
                fig_h.add_annotation(
                    x=(2.0 + zone_x1) / 2, y=zone_y0_neg * 0.85,
                    text="⏳ Coiling / Fresh Breakouts",
                    showarrow=False,
                    font=dict(color="#60a5fa", size=11)
                )
                fig_h.add_annotation(
                    x=(zone_x0_neg + 2.0) / 2, y=zone_y0_neg * 0.85,
                    text="🔴 Cut Loss Zone (Low Z + Loss)",
                    showarrow=False,
                    font=dict(color="#f87171", size=11)
                )
                
                fig_h.update_layout(
                    template="plotly_dark",
                    height=560,
                    margin=dict(l=40, r=40, t=40, b=40),
                    paper_bgcolor='rgba(15, 23, 42, 0.6)',
                    plot_bgcolor='rgba(15, 23, 42, 0.6)',
                    hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.95)", font_size=12, font_family="JetBrains Mono")
                )
                
                event_h = st.plotly_chart(fig_h, use_container_width=True, on_select="rerun", key="holdings_roppel_chart")
                
                selected_h = []
                if event_h and 'selection' in event_h and 'points' in event_h['selection'] and event_h['selection']['points']:
                    selected_h = [p['customdata'][0] for p in event_h['selection']['points'] if 'customdata' in p]
                if not selected_h:
                    selected_h = df_h_plot['clean_ticker'].tolist()
                    
                tv_list_h = ",".join(["NSE:" + t for t in selected_h])
                has_h_sel = bool(event_h and 'selection' in event_h and event_h['selection'].get('points'))
                with st.expander(f"📋 Copy Selected Holdings for TradingView ({len(selected_h)})", expanded=has_h_sel):
                    st.code(tv_list_h, language="text")
                    st.caption("💡 **Box Select Tool:** Click **Box Select** or **Lasso Select** in the chart toolbar (top-right of plot) to drag across any points and export directly to TradingView.")
        
        with tab_h_table:
            # Add filter for holdings view if more than 8
            if len(df_active) > 8:
                hf1, hf2 = st.columns([1, 3])
                with hf1:
                    h_filter = st.selectbox(
                        "Filter Holdings:", 
                        ["All Holdings", "⚡ Averaging Up Only", "🔴 Exit / Stop Alerts Only", "🟢 Healthy Holdings Only"],
                        key="holdings_filter_select"
                    )
                if h_filter == "⚡ Averaging Up Only":
                    df_active = df_active[df_active['Action Today'].str.contains('ADD')]
                elif h_filter == "🔴 Exit / Stop Alerts Only":
                    df_active = df_active[df_active['Action Today'].str.contains('EXIT|TRIM')]
                elif h_filter == "🟢 Healthy Holdings Only":
                    df_active = df_active[df_active['Action Today'].str.contains('HOLD')]

            st.dataframe(
                df_active,
                column_config={
                    "Chart": st.column_config.LinkColumn("Chart", display_text="Open TV ↗"),
                    "Industry": st.column_config.TextColumn("Industry"),
                    "ADTV (₹ Cr)": st.column_config.TextColumn("ADTV (₹ Cr)")
                },
                use_container_width=True
            )
    else:
        st.info("No active holdings data available.")

    st.markdown("---")

    # --- 4. NIFTY 750 CANSLIM LEADERBOARD (TOP 50) ---
    st.markdown("## 📊 NIFTY 750 Cross-Sectional Leaderboard (Top 50 Momentum Pool)")
    st.caption(r"Cross-referencing Volatility-Adjusted Z-Scores with CANSLIM Sales Growth ($\ge 25\%$), Base Stages, and David Ryan RSNH signals.")

    # 4.0 TOP THEMATIC CLUSTERS & SECTOR LEADERSHIP SUMMARY
    df_lead = leaderboard.copy() if isinstance(leaderboard, pd.DataFrame) else pd.DataFrame(leaderboard)
    if min_adtv > 0.0 and 'ADTV_Cr' in df_lead.columns:
        df_lead = df_lead[df_lead['ADTV_Cr'] >= min_adtv]

    theme_group_stats = []
    if not df_lead.empty and 'Industry' in df_lead.columns:
        for ind, g in df_lead.groupby('Industry'):
            if ind and ind != 'Unknown':
                stock_list = g.sort_values(by='Z_Score', ascending=False)
                theme_group_stats.append({
                    'industry': ind,
                    'count': len(g),
                    'stocks': stock_list.to_dict('records'),
                    'avg_z': float(g['Z_Score'].mean()),
                    'buyable_count': int(len(g[g['In_Buy_Zone'] == True])),
                    'avg_sales': float(g['Sales_YoY'].mean()),
                    'is_super': len(g) >= 5,
                    'is_cluster': len(g) >= 3
                })
        theme_group_stats.sort(key=lambda x: (x['count'], x['avg_z']), reverse=True)

    topper_themes = [t for t in theme_group_stats if t['count'] >= 3]
    if not topper_themes and theme_group_stats:
        topper_themes = theme_group_stats[:3]

    if topper_themes:
        st.markdown("### 👑 Leading Institutional Industry Themes (Top 50 Concentrations)")
        st.caption(r"William O'Neil CANSLIM Rule: **37% of a stock's price move is directly propelled by its industry group**. Themes with $\ge 3$ leaders (or $\ge 5$ super-clusters) indicate massive institutional accumulation across that sector.")
        st.markdown(
            "<div style='font-size: 0.80rem; color: #94a3b8; margin-top: -6px; margin-bottom: 12px;'>"
            "🎯 <b style='color: #34d399;'>In Buy Zone (🟢)</b>: Stock is at a low-risk CANSLIM entry point — trading within <b>0% to +5% above base/VCP pivot</b> on volume, or <b>bouncing off rising 21 EMA</b> (stocks &gt;5% above pivot are extended)."
            "</div>",
            unsafe_allow_html=True
        )
        
        display_toppers = topper_themes[:4]
        cols = st.columns(len(display_toppers))
        for idx, t in enumerate(display_toppers):
            with cols[idx]:
                card_type = "theme-card-super" if t['is_super'] else "theme-card-cluster"
                badge_html = f"<span class='theme-badge-super'>🔥 SUPER-CLUSTER ({t['count']})</span>" if t['is_super'] else f"<span class='theme-badge-cluster'>⚡ THEME CLUSTER ({t['count']})</span>"
                
                chips_html = []
                for s in t['stocks']:
                    c_sym = str(s['Ticker']).replace('.NS', '').strip()
                    tv_url = get_tradingview_url(c_sym)
                    is_b = s.get('In_Buy_Zone', False)
                    cls_b = " theme-stock-chip-buyable" if is_b else ""
                    b_icon = "🟢 " if is_b else ""
                    s_z = float(s.get('Z_Score', 0.0))
                    s_p = float(s.get('Price', 0.0))
                    chips_html.append(f"<a href='{tv_url}' target='_blank' class='theme-stock-chip{cls_b}' title='Z: {s_z:+.2f} | CMP: ₹{s_p:.2f}'>{b_icon}{c_sym} (+{s_z:.1f})</a>")
                chips_str = "".join(chips_html)
                
                pct_share = (t['count'] / len(df_lead)) * 100 if len(df_lead) > 0 else 0
                buyable_str = f"<span style='color:#34d399; font-weight:700;'>{t['buyable_count']} Buyable</span>" if t['buyable_count'] > 0 else "<span style='color:#94a3b8;'>0 in zone</span>"
                t_ind = t['industry']
                
                st.markdown(f"""
                <div class='theme-leader-card {card_type}'>
                    <div class='theme-card-header'>
                        <div class='theme-title' title='{t_ind}'>{t_ind}</div>
                        {badge_html}
                    </div>
                    <div class='theme-metrics-grid'>
                        <div class='metric-cell'>
                            <span class='metric-cell-lbl'>Concentration</span>
                            <span class='metric-cell-val'>{t['count']} Stocks <span style='font-size:0.68rem; color:#94a3b8;'>({pct_share:.0f}%)</span></span>
                        </div>
                        <div class='metric-cell'>
                            <span class='metric-cell-lbl'>In Buy Zone</span>
                            <span class='metric-cell-val'>{buyable_str}</span>
                        </div>
                        <div class='metric-cell'>
                            <span class='metric-cell-lbl'>Avg Momentum Z</span>
                            <span class='metric-cell-val' style='color:#38bdf8;'>+{t['avg_z']:.2f}</span>
                        </div>
                        <div class='metric-cell'>
                            <span class='metric-cell-lbl'>Avg Sales YoY</span>
                            <span class='metric-cell-val' style='color:#facc15;'>+{t['avg_sales']:.1f}%</span>
                        </div>
                    </div>
                    <div>
                        <div style='font-size:0.64rem; text-transform:uppercase; color:#64748b; font-weight:700; margin-top:2px;'>Top Leaders in Theme</div>
                        <div class='theme-stocks-row'>{chips_str}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    tab_l_quadrant, tab_l_table = st.tabs(["🔭 Roppel CANSLIM Quadrant", "📋 Top 50 Leaderboard Table"])

    with tab_l_quadrant:
        st.caption("Visualizing the Top 50 Momentum Pool: **X-Axis = Momentum Z-Score (40/40/20)**, **Y-Axis = Selected Fundamental/Technical Indicator**. Apex CANSLIM leaders live in the top-right quadrant ($Z \\ge 2.5$, Sales $\\ge 25\\%$). Bubble size reflects Breakout Volume Multiple.")
        
        q_c1, q_c2 = st.columns([2, 2])
        with q_c1:
            y_choice = st.selectbox(
                "Select Y-Axis Indicator:",
                ["YoY Sales Growth (%) [Canonical Roppel]", "Distance from Base Pivot (%)", "Breakout Volume Ratio (x)", "YoY EPS Growth (%)"],
                key="lead_roppel_y_choice"
            )
        with q_c2:
            color_choice = st.radio(
                "Color Bubbles By:",
                ["In Buy Zone / Actionable", "RS Line New High (RSNH)", "Setup Type", "Industry / Theme"],
                horizontal=True,
                key="lead_roppel_color_choice"
            )
        
        df_l_plot = leaderboard.copy() if isinstance(leaderboard, pd.DataFrame) else pd.DataFrame(leaderboard)
        if not df_l_plot.empty:
            if min_adtv > 0.0 and 'ADTV_Cr' in df_l_plot.columns:
                df_l_plot = df_l_plot[df_l_plot['ADTV_Cr'] >= min_adtv]
            df_l_plot['clean_ticker'] = df_l_plot['Ticker'].apply(lambda x: str(x).replace('.NS', '').split('-')[0].strip())
            df_l_plot['plot_sales'] = df_l_plot['Sales_YoY'].clip(lower=-50, upper=300)
            df_l_plot['plot_eps'] = df_l_plot['EPS_YoY'].clip(lower=-50, upper=300)
            df_l_plot['bubble_size'] = df_l_plot['Vol_Ratio'].clip(lower=1.0, upper=5.0) * 5.0
            
            if "Sales Growth" in y_choice:
                y_var = 'plot_sales'
                y_title = "YoY Sales Growth (%)"
                cross_y = 25.0
                apex_title = "👑 Apex CANSLIM Leaders (Sales ≥ 25%, Z ≥ 2.5)"
            elif "Pivot" in y_choice:
                y_var = 'Dist_Pivot'
                y_title = "Distance from Pivot (%)"
                cross_y = 5.0
                apex_title = "🎯 Sweet Spot: ≤ 5% Above Base Pivot"
            elif "Volume Ratio" in y_choice:
                y_var = 'Vol_Ratio'
                y_title = "Breakout Volume Ratio (x)"
                cross_y = 1.4
                apex_title = "🚀 Institutional Volume Thrust (≥ 1.4x)"
            else:
                y_var = 'plot_eps'
                y_title = "YoY EPS Growth (%)"
                cross_y = 25.0
                apex_title = "💰 High EPS Growth Leaders (≥ 25%)"
                
            if color_choice == "In Buy Zone / Actionable":
                df_l_plot['Buy_Zone_Status'] = df_l_plot['In_Buy_Zone'].map({True: '🟢 In Buy Zone', False: '⚪ Extended / Setting Up'})
                color_var = 'Buy_Zone_Status'
                color_map = {'🟢 In Buy Zone': '#10b981', '⚪ Extended / Setting Up': '#64748b'}
                sorted_theme_order = None
            elif color_choice == "RS Line New High (RSNH)":
                df_l_plot['RSNH_Status'] = df_l_plot['RSNH'].map({True: '👑 RS New High', False: '⚪ Normal RS'})
                color_var = 'RSNH_Status'
                color_map = {'👑 RS New High': '#fbbf24', '⚪ Normal RS': '#64748b'}
                sorted_theme_order = None
            elif color_choice == "Industry / Theme":
                ind_counts = df_l_plot['Industry'].value_counts().to_dict()
                
                def _get_theme_label(row):
                    ind = str(row.get('Industry', 'Unknown'))
                    cnt = ind_counts.get(ind, 1)
                    if cnt >= 5:
                        return f"🔥 {ind} ({cnt})"
                    elif cnt >= 3:
                        return f"⚡ {ind} ({cnt})"
                    elif cnt > 1:
                        return f"🏢 {ind} ({cnt})"
                    else:
                        return f"🏢 {ind} (1)"

                df_l_plot['Theme_Group'] = df_l_plot.apply(_get_theme_label, axis=1)
                color_var = 'Theme_Group'
                color_map = None
                
                # Explicitly sort theme groups by stock count descending, then avg Z-Score descending
                theme_order_df = df_l_plot.groupby('Theme_Group').agg(
                    cnt=('clean_ticker', 'count'),
                    avg_z=('Z_Score', 'mean')
                ).sort_values(by=['cnt', 'avg_z'], ascending=[False, False])
                
                sorted_theme_order = theme_order_df.index.tolist()
            else:
                color_var = 'Setup'
                color_map = None
                sorted_theme_order = None
                
            cat_orders = {color_var: sorted_theme_order} if (color_choice == "Industry / Theme" and sorted_theme_order) else None

            fig_l = px.scatter(
                df_l_plot,
                x='Z_Score',
                y=y_var,
                size='bubble_size',
                color=color_var,
                hover_name='clean_ticker',
                text='clean_ticker',
                custom_data=['clean_ticker', 'Price', 'Setup', 'Sales_YoY', 'Vol_Ratio', 'Z_Score', 'Dist_Pivot', 'Industry', 'ADTV_Cr'],
                color_discrete_map=color_map,
                category_orders=cat_orders,
                labels={
                    'Z_Score': 'Cross-Sectional Momentum Z-Score (40% 1M, 40% 3M, 20% 6M)',
                    y_var: y_title
                }
            )
            fig_l.update_traces(
                textposition='top center',
                marker=dict(line=dict(width=1.5, color='rgba(255, 255, 255, 0.4)')),
                textfont=dict(color='#f1f5f9', size=11, family="JetBrains Mono, Inter"),
                hovertemplate="<b>%{hovertext}</b><br>Industry: %{customdata[7]}<br>ADTV: ₹%{customdata[8]:.1f} Cr<br>Z-Score: %{customdata[5]:+.2f}<br>CMP: ₹%{customdata[1]:.2f}<br>Setup: %{customdata[2]}<br>Sales YoY: %{customdata[3]:+.1f}%<br>Vol: %{customdata[4]:.1f}x<br>Dist Pivot: %{customdata[6]:+.1f}%"
            )
            
            # Crosshairs
            fig_l.add_hline(y=cross_y, line_dash="dash", line_color="rgba(255, 255, 255, 0.35)", line_width=1.5)
            fig_l.add_vline(x=2.5, line_dash="dash", line_color="rgba(255, 255, 255, 0.35)", line_width=1.5)
            
            max_lz = float(df_l_plot['Z_Score'].max()) if not df_l_plot.empty else 5.0
            max_ly = float(df_l_plot[y_var].max()) if not df_l_plot.empty else cross_y * 2
            min_ly = float(df_l_plot[y_var].min()) if not df_l_plot.empty else 0.0
            
            zone_lx1 = max(max_lz * 1.15, 5.0)
            zone_ly1 = max(max_ly * 1.2, cross_y * 1.5)
            
            fig_l.add_shape(
                type="rect",
                x0=2.5, y0=cross_y, x1=zone_lx1, y1=zone_ly1,
                fillcolor="rgba(16, 185, 129, 0.08)",
                line=dict(color="rgba(16, 185, 129, 0.35)", width=1, dash="dot"),
                layer="below"
            )
            fig_l.add_annotation(
                x=(2.5 + zone_lx1) / 2, y=zone_ly1 * 0.95,
                text=apex_title,
                showarrow=False,
                font=dict(color="#34d399", size=13, weight="bold")
            )
            
            fig_l.update_layout(
                template="plotly_dark",
                height=620,
                margin=dict(l=40, r=40, t=40, b=40),
                paper_bgcolor='rgba(15, 23, 42, 0.6)',
                plot_bgcolor='rgba(15, 23, 42, 0.6)',
                hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.95)", font_size=12, font_family="JetBrains Mono"),
                legend=dict(
                    traceorder="normal",
                    font=dict(size=10.5, family="JetBrains Mono, Inter")
                )
            )
            
            event_l = st.plotly_chart(fig_l, use_container_width=True, on_select="rerun", key="leaderboard_roppel_chart")
            
            selected_l = []
            if event_l and 'selection' in event_l and 'points' in event_l['selection'] and event_l['selection']['points']:
                selected_l = [p['customdata'][0] for p in event_l['selection']['points'] if 'customdata' in p]
            if not selected_l:
                selected_l = df_l_plot['clean_ticker'].tolist()
                
            tv_list_l = ",".join(["NSE:" + t for t in selected_l])
            has_l_sel = bool(event_l and 'selection' in event_l and event_l['selection'].get('points'))
            with st.expander(f"📋 Copy Selected Leaders for TradingView ({len(selected_l)})", expanded=has_l_sel):
                st.code(tv_list_l, language="text")
                st.caption("💡 **Box Select Tool:** Click **Box Select** or **Lasso Select** in the chart toolbar (top-right of plot) to drag across any cluster and generate a custom TradingView watchlist.")

    with tab_l_table:
        # Leaderboard filters
        filter_options = [
            "All Top 50",
            "In Buy Zone Only",
            "All Thematic Clusters (≥3 Leaders)",
            "Sales Growth >= 25% Only",
            "RS Line New High (RSNH) Only"
        ]
        if topper_themes:
            for t in topper_themes:
                filter_options.append(f"🔥 Theme: {t['industry']} ({t['count']})")

        f_col1, f_col2 = st.columns([1.5, 3.5])
        with f_col1:
            view_filter = st.selectbox(
                "Filter Leaderboard:",
                filter_options
            )

        df_filtered = leaderboard.copy()
        if view_filter == "In Buy Zone Only":
            df_filtered = df_filtered[df_filtered['In_Buy_Zone'] == True]
        elif view_filter == "All Thematic Clusters (≥3 Leaders)":
            df_filtered = df_filtered[df_filtered['Is_Cluster'] == True]
        elif view_filter.startswith("🔥 Theme: "):
            chosen_ind = view_filter.replace("🔥 Theme: ", "").rsplit(" (", 1)[0].strip()
            df_filtered = df_filtered[df_filtered['Industry'] == chosen_ind]
        elif view_filter == "Sales Growth >= 25% Only":
            df_filtered = df_filtered[df_filtered['Fund_Pass'] == True]
        elif view_filter == "RS Line New High (RSNH) Only":
            df_filtered = df_filtered[df_filtered['RSNH'] == True]

        if min_adtv > 0.0 and 'ADTV_Cr' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['ADTV_Cr'] >= min_adtv]

        df_filtered_view = df_filtered.copy()
        df_filtered_view['Chart'] = df_filtered_view['Ticker'].apply(get_tradingview_url)
        df_filtered_view['Industry_Theme'] = df_filtered_view.apply(
            lambda r: f"🔥 {r['Industry']} ({r['Cluster_Count']})" if r.get('Is_Cluster') else str(r.get('Industry', 'Unknown')),
            axis=1
        )
        leader_cols = [
            'Rank', 'Ticker', 'Chart', 'Industry_Theme', 'ADTV_Cr', 'Z_Score', 'Price', 'Setup', 'In_Buy_Zone',
            'Dist_Pivot', 'Vol_Ratio', 'EPS_YoY', 'Sales_YoY', 'ROE', 'Base_Stage', 'RSNH'
        ]
        st.dataframe(
            df_filtered_view[leader_cols],
            column_config={
                "Chart": st.column_config.LinkColumn("Chart", display_text="Open TV ↗"),
                "Industry_Theme": st.column_config.TextColumn("Industry / Theme"),
                "ADTV_Cr": st.column_config.NumberColumn("ADTV (₹ Cr)", format="₹%.1f Cr")
            },
            use_container_width=True
        )

        with st.expander("📋 Copy Filtered Leaders for TradingView"):
            tv_list = ",".join(["NSE:" + t for t in df_filtered['Ticker'].tolist()])
            st.code(tv_list, language="text")


# =============================================================================
# MODE 2: MONTHLY PASSIVE ROBO-ADVISOR (ORIGINAL BASELINE)
# =============================================================================

else:
    render_header(
        "⚡ Fast Momentum Robo-Advisor",
        "Autonomous execution engine for the Nifty 750 Top 15 Equal-Weight strategy. Tracks the optimized momentum portfolio on a daily basis and calculates exact start-of-month rebalance orders."
    )

    with st.expander("📖 Fast Momentum Scoring Methodology (No Black Box)", expanded=False):
        st.markdown("""
        **What makes this systematic engine work?**
        Unlike simple price-return momentum, this algorithm implements an institutional-grade **Volatility-Adjusted Cross-Sectional Z-Score** model:
        
        🥇 **1. The Universe (The Alpha Zone)**
        - Strictly filters to the **NIFTY 750** (Nifty Total Market).
        
        📊 **2. Volatility-Adjusted Returns (The Math)**
        - Absolute returns across 1-Month (21d), 3-Month (63d), and 6-Month (126d) divided by **Rolling 1-Year Annualized Volatility**.
        
        ⚖️ **3. Cross-Sectional Z-Score (The Ranking)**
        - Standardized Z-Score across the entire 750 stock cross-section.
        
        🎛️ **4. The Balanced Weighting Scale**
        - 1-Month Z-Score: 40% Weight | 3-Month Z-Score: 40% Weight | 6-Month Z-Score: 20% Weight.
        
        🔄 **5. Rebalancing & Exits**
        - Locks on the last trading day of the month; executes on the 1st trading day of the new month.
        """)

    # Universe Selector Toggle
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 12px;
    ">
        <span style="color: rgba(255,255,255,0.5); font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px;">
            📡 Scan Universe
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    u_col1, u_col2 = st.columns([4, 1])
    with u_col1:
        universe_choice_m2 = st.radio(
            "Select stock universe for Robo-Advisor momentum scoring:",
            [
                "⚡ NIFTY 750 (Default — Large, Mid & Small Caps)",
                "🌐 All India Deep Market (2500+ NSE Stocks)"
            ],
            index=0,
            horizontal=True,
            key="mode2_universe"
        )
    with u_col2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Data", key="refresh_m2", help="Clear cache and recompute"):
            st.cache_data.clear()
            st.rerun()

    universe_mode_m2 = "deep_market" if "2500+" in universe_choice_m2 else "nifty_750"
    universe_label_m2 = "Deep Market (2500+ NSE)" if universe_mode_m2 == "deep_market" else "NIFTY 750"
    
    if universe_mode_m2 == "deep_market":
        st.info("🌐 **Deep Market Mode:** Computing Z-scores across 2500+ NSE stocks. First load may take 30–60 seconds while the momentum matrix is built.", icon="⏳")

    @st.cache_data(ttl=3600)
    def load_momentum_state(_universe_mode="nifty_750"):
        score_matrix, close_df = compute_live_fast_momentum_matrix(universe_mode=_universe_mode)
        if score_matrix is None or score_matrix.empty:
            return None, None, None, None, None, None
            
        dates = pd.Series(score_matrix.index, index=score_matrix.index)
        today = dates.iloc[-1]
        first_day_this_month = today.replace(day=1)
        past_dates = dates[dates < first_day_this_month]
        
        if not past_dates.empty:
            last_month_rebalance_date = past_dates.iloc[-1]
        else:
            last_month_rebalance_date = dates.iloc[0]
            
        locked_portfolio = get_target_portfolio(last_month_rebalance_date, score_matrix, close_df, n=15)
        live_portfolio = get_target_portfolio(today, score_matrix, close_df, n=15)
        
        is_month_end = dates.dt.month != dates.shift(-1).dt.month
        month_end_dates = dates[is_month_end]
        valid_historical_ends = month_end_dates[month_end_dates <= last_month_rebalance_date]
        hist_ends = valid_historical_ends.tail(6)
        
        hist_portfolios = {}
        for d in hist_ends:
            hist_portfolios[d] = get_target_portfolio(d, score_matrix, close_df, n=50)
        
        return last_month_rebalance_date, today, locked_portfolio, live_portfolio, hist_portfolios, score_matrix

    with st.spinner(f"Crunching massive {universe_label_m2} volatility and momentum vectors..."):
        last_month_date, today_date, locked_portfolio, live_portfolio, hist_portfolios, score_matrix = load_momentum_state(_universe_mode=universe_mode_m2)

    if not locked_portfolio:
        st.error("Failed to load historical cache. Please ensure Price History Manager has downloaded initial data.")
        st.stop()

    locked_tickers = set(locked_portfolio.keys())
    live_tickers = set(live_portfolio.keys())

    hist_dates_sorted = sorted(hist_portfolios.keys(), reverse=True)
    streaks = {}
    base_tickers = list(hist_portfolios[hist_dates_sorted[0]].keys())
    for ticker in base_tickers:
        months_in = []
        for hd in hist_dates_sorted:
            if ticker in hist_portfolios[hd]:
                months_in.append(hd.strftime('%b'))
        months_in.reverse()
        streaks[ticker] = {
            'count': len(months_in),
            'months': ", ".join(months_in)
        }

    rank_velocity = {}
    if score_matrix is not None:
        scores_last = score_matrix.loc[last_month_date].dropna()
        scores_today = score_matrix.loc[today_date].dropna()
        ranks_last = scores_last.rank(ascending=False, method='first')
        ranks_today = scores_today.rank(ascending=False, method='first')
        common_tickers = ranks_last.index.intersection(ranks_today.index)
        for t in common_tickers:
            velocity = int(ranks_last[t] - ranks_today[t])
            current_rank = int(ranks_today[t])
            prev_rank = int(ranks_last[t])
            rank_velocity[t] = {
                'velocity': velocity,
                'current_rank': current_rank,
                'prev_rank': prev_rank
            }

    sells = locked_tickers - live_tickers
    buys = live_tickers - locked_tickers
    holds = locked_tickers.intersection(live_tickers)

    st.header("📋 Rebalance Action Plan")
    if today_date.month != last_month_date.month and today_date.day <= 5:
        st.info(f"🚨 **REBALANCE WINDOW OPEN** (Based on {last_month_date.strftime('%b %d')} closing scores)")
    else:
        st.warning("⏳ Rebalance Window is currently **CLOSED**. Next rebalance occurs on the 1st trading day of next month. The below orders are *projections*.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🔴 DROPS (Sell to Cash)")
        if sells:
            for t in sells:
                clean_sym = t.replace('.NS', '')
                tv_url = get_tradingview_url(clean_sym)
                st.markdown(f"- [**{clean_sym}**]({tv_url}) ↗ *(Failed Momentum cut)*")
        else:
            st.success("No Sell Signals.")

    with col2:
        st.subheader("🟢 ADDS (Buy Equal Weight)")
        if buys:
            for t in buys:
                clean_sym = t.replace('.NS', '')
                tv_url = get_tradingview_url(clean_sym)
                st.markdown(f"- [**{clean_sym}**]({tv_url}) ↗")
        else:
            st.success("No New Buy Signals.")

    with col3:
        st.subheader("🔵 HOLDS")
        for t in holds:
            score = live_portfolio[t]['Score']
            clean_sym = t.replace('.NS', '')
            tv_url = get_tradingview_url(clean_sym)
            st.markdown(f"- [{clean_sym}]({tv_url}) ↗ *(Score: {score:.1f})*")

    st.markdown("---")
    st.header("📊 Systematic Allocation States")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Current Enforced Portfolio")
        st.caption(f"Locked on the close of **{last_month_date.strftime('%A, %b %d, %Y')}**")
        locked_data = []
        for t, data in locked_portfolio.items():
            clean_t = t.replace(".NS", "")
            locked_data.append({
                "Ticker": clean_t,
                "Chart": get_tradingview_url(clean_t),
                "Locked Score (Z)": round(data['Score'], 2),
                "Locked Price": round(data['Price'], 2) if pd.notna(data['Price']) else "N/A",
                "Months Held": streaks.get(t, {}).get('count', 1)
            })
        df_locked = pd.DataFrame(locked_data)
        df_locked.index = df_locked.index + 1
        st.dataframe(
            df_locked,
            column_config={
                "Chart": st.column_config.LinkColumn("Chart", display_text="Open TV ↗")
            },
            use_container_width=True
        )

        with st.expander("📋 Copy Portfolio for TradingView"):
            tv_locked_list = ",".join(["NSE:" + d['Ticker'] for d in locked_data])
            st.code(tv_locked_list, language="text")

    with col_r:
        st.markdown("### Live Forward Target (Next Month)")
        st.caption(f"Based on today's close **{today_date.strftime('%A, %b %d, %Y')}**")
        live_data = []
        for t, data in live_portfolio.items():
            status = "⭐ NEW" if t in buys else "✅ HOLD"
            clean_t = t.replace(".NS", "")
            live_data.append({
                "Ticker": clean_t,
                "Chart": get_tradingview_url(clean_t),
                "Live Score (Z)": round(data['Score'], 2),
                "Current Price": round(data['Price'], 2) if pd.notna(data['Price']) else "N/A",
                "Status": status
            })
        df_live = pd.DataFrame(live_data)
        df_live.index = df_live.index + 1
        st.dataframe(
            df_live,
            column_config={
                "Chart": st.column_config.LinkColumn("Chart", display_text="Open TV ↗")
            },
            use_container_width=True
        )

        with st.expander("📋 Copy Forward Target for TradingView"):
            tv_live_list = ",".join(["NSE:" + d['Ticker'] for d in live_data])
            st.code(tv_live_list, language="text")

    # Mode 2: Momentum Allocation Frontier Scatter
    with st.expander("🔭 The Robo-Advisor Momentum Frontier (Live vs Locked Comparison)", expanded=False):
        st.caption("Visualizing the Top 15 systematic portfolio allocation: **X-Axis = Momentum Score**, **Y-Axis = Months in Top 50**. Box-select any cluster to export directly to TradingView.")
        all_robo_tickers = list(locked_tickers.union(live_tickers))
        robo_rows = []
        for t in all_robo_tickers:
            clean_t = t.replace(".NS", "")
            is_locked = t in locked_tickers
            is_live = t in live_tickers
            if is_live and not is_locked:
                status_lbl = "🟢 NEW ADD"
            elif is_locked and not is_live:
                status_lbl = "🔴 DROP"
            else:
                status_lbl = "🔵 PERSISTENT HOLD"
            
            sc = live_portfolio.get(t, {}).get('Score', locked_portfolio.get(t, {}).get('Score', 0.0))
            pr = live_portfolio.get(t, {}).get('Price', locked_portfolio.get(t, {}).get('Price', 0.0))
            m_count = streaks.get(t, {}).get('count', 1)
            
            robo_rows.append({
                'clean_ticker': clean_t,
                'Score': float(sc),
                'Price': float(pr) if pd.notna(pr) else 0.0,
                'Months_Held': int(m_count),
                'Status': status_lbl,
                'bubble_size': max(m_count * 5.0, 10.0)
            })
        df_robo_plot = pd.DataFrame(robo_rows)
        if not df_robo_plot.empty:
            fig_r = px.scatter(
                df_robo_plot,
                x='Score',
                y='Months_Held',
                color='Status',
                size='bubble_size',
                hover_name='clean_ticker',
                text='clean_ticker',
                custom_data=['clean_ticker', 'Score', 'Months_Held', 'Status'],
                color_discrete_map={
                    '🟢 NEW ADD': '#10b981',
                    '🔵 PERSISTENT HOLD': '#3b82f6',
                    '🔴 DROP': '#ef4444'
                },
                labels={
                    'Score': 'Momentum Score (NSE 1 + Weighted Z)',
                    'Months_Held': 'Months in Top 50 (Continuity)'
                }
            )
            fig_r.update_traces(
                textposition='top center',
                marker=dict(line=dict(width=1.5, color='rgba(255, 255, 255, 0.4)')),
                textfont=dict(color='#f1f5f9', size=11, family="JetBrains Mono, Inter"),
                hovertemplate="<b>%{hovertext}</b><br>Score: %{customdata[1]:.2f}<br>Months: %{customdata[2]}M<br>Status: %{customdata[3]}"
            )
            fig_r.add_hline(y=2, line_dash="dash", line_color="rgba(255, 255, 255, 0.35)", line_width=1.5)
            fig_r.add_vline(x=float(df_robo_plot['Score'].median()), line_dash="dash", line_color="rgba(255, 255, 255, 0.35)", line_width=1.5)
            
            fig_r.update_layout(
                template="plotly_dark",
                height=480,
                margin=dict(l=40, r=40, t=40, b=40),
                paper_bgcolor='rgba(15, 23, 42, 0.6)',
                plot_bgcolor='rgba(15, 23, 42, 0.6)',
                hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.95)", font_size=12, font_family="JetBrains Mono")
            )
            event_r = st.plotly_chart(fig_r, use_container_width=True, on_select="rerun", key="robo_roppel_chart")
            
            selected_r = []
            if event_r and 'selection' in event_r and 'points' in event_r['selection'] and event_r['selection']['points']:
                selected_r = [p['customdata'][0] for p in event_r['selection']['points'] if 'customdata' in p]
            if not selected_r:
                selected_r = df_robo_plot['clean_ticker'].tolist()
                
            tv_list_r = ",".join(["NSE:" + t for t in selected_r])
            has_r_sel = bool(event_r and 'selection' in event_r and event_r['selection'].get('points'))
            st.code(tv_list_r, language="text")
            st.caption("💡 **Tip:** Box-select any cluster in the chart above to isolate tickers for TradingView.")

    st.markdown("---")
    st.markdown("### 🔥 Super Compounders (Top 50 Continuity Buffer)")
    st.caption("All stocks mathematically maintaining elite status (Top 50 out of 750) for at least 2 out of the last 6 months.")

    super_compounders = {t: details for t, details in streaks.items() if details['count'] >= 2}
    if super_compounders:
        sorted_compounders = sorted(super_compounders.items(), key=lambda x: x[1]['count'], reverse=True)
        cols = st.columns(4)
        for idx, (t, details) in enumerate(sorted_compounders):
            clean_t = t.replace(".NS", "")
            streak_count = details['count']
            months_str = details['months']
            tv_url = get_tradingview_url(clean_t)
            with cols[idx % 4]:
                render_metric_card(
                    label="Months in Top 50",
                    value=f"{streak_count} M",
                    color_class="green-text" if streak_count >= 4 else "yellow-text",
                    extra_style="border-color: #10b981; border-width: 2px;" if streak_count >= 4 else ""
                )
                st.markdown(f"<div style='text-align: center; margin-top: 5px;'><a href='{tv_url}' target='_blank' rel='noopener noreferrer' style='font-weight: 800; font-size: 1.1rem; color: #60a5fa; text-decoration: none;'>{clean_t} <span style='font-size: 0.75rem;'>↗</span></a></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; font-size: 0.85rem; color: #64748b; margin-top: -5px;'>({months_str})</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Copy Super Compounders for TradingView"):
            tv_sc_list = ",".join(["NSE:" + t.replace(".NS", "") for t, _ in sorted_compounders])
            st.code(tv_sc_list, language="text")
    else:
        st.info("No Super Compounders currently exist.")

    st.markdown("---")
    st.markdown("### 🚀 Rocket Rank (Highest Inter-Month Velocity)")
    if rank_velocity:
        rockets = {t: v for t, v in rank_velocity.items() if v['current_rank'] <= 50 and v['velocity'] > 0}
        sorted_rockets = sorted(rockets.items(), key=lambda x: x[1]['velocity'], reverse=True)[:20]
        if sorted_rockets:
            rocket_cols = st.columns(4)
            for idx, (t, v) in enumerate(sorted_rockets):
                clean_t = t.replace(".NS", "")
                tv_url = get_tradingview_url(clean_t)
                with rocket_cols[idx % 4]:
                    render_metric_card(
                        label=f"Rank {v['prev_rank']} → {v['current_rank']}",
                        value=f"+{v['velocity']}",
                        color_class="green-text",
                        extra_style="border-color: #f59e0b; border-width: 2px;"
                    )
                    st.markdown(f"<div style='text-align: center; margin-top: 5px;'><a href='{tv_url}' target='_blank' rel='noopener noreferrer' style='font-weight: 800; font-size: 1.1rem; color: #60a5fa; text-decoration: none;'>{clean_t} <span style='font-size: 0.75rem;'>↗</span></a></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📋 Copy Rocket Rank for TradingView"):
                tv_rocket_list = ",".join(["NSE:" + t.replace(".NS", "") for t, _ in sorted_rockets])
                st.code(tv_rocket_list, language="text")
        else:
            st.info("No significant velocity thrusts detected this cycle.")
    else:
        st.info("Rank velocity data unavailable.")
