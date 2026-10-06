import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import re
import yfinance as yf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from styles import load_css
    load_css()
except ImportError:
    pass

from components import render_header, apply_plotly_theme
from database import (
    get_connection, get_latest_global_regime, 
    get_latest_global_etf_momentum, classify_etf_asset_class
)
try:
    from sync_macro import sync_all_data
except ImportError:
    def sync_all_data():
        pass


def render_html(html_str: str):
    """
    Renders custom HTML cleanly without Markdown code-block indentation interpretation.
    Strips comments and leading whitespace on each line.
    """
    clean_html = re.sub(r'<!--.*?-->', '', html_str, flags=re.DOTALL)
    clean_html = re.sub(r'^[ \t]+', '', clean_html.strip(), flags=re.MULTILINE)
    st.markdown(clean_html, unsafe_allow_html=True)


def format_display_label(ticker: str, display_name: str) -> tuple:
    """Cleans up ticker and asset names for institutional readability."""
    clean_ticker = str(ticker).replace('.NS', '').replace('.BO', '').strip()
    clean_name = str(display_name).strip()
    for drop_word in ['Select Sector', 'Select Secto', 'Select Sect', 'Trust', 'Fund', 'Index Fund', 'ETF']:
        clean_name = clean_name.replace(drop_word, '').strip()
    clean_name = re.sub(r'[\(\)]', '', clean_name).strip()
    if len(clean_name) > 28:
        clean_name = clean_name[:26].rstrip() + '…'
    return clean_ticker, clean_name


# =============================================================================
# DATA FETCHING HELPERS WITH CACHING
# =============================================================================

@st.cache_data(ttl=900)
def fetch_macro_barometers():
    """
    Fetches live / delayed inter-market macro barometers:
    1. Currencies: USD/INR ('USDINR=X') & US Dollar Index DXY ('DX-Y.NYB')
    2. Yields: US 10-Year Treasury Yield ('^TNX')
    3. Volatility: India VIX ('^INDIAVIX') & US VIX ('^VIX')
    4. Energy: Brent Crude Oil ('BZ=F')
    """
    symbols = ['USDINR=X', 'DX-Y.NYB', '^TNX', '^INDIAVIX', '^VIX', 'BZ=F']
    try:
        data = yf.download(symbols, period="1mo", progress=False)
        close_df = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
        close_df = close_df.ffill().bfill()
        
        results = {}
        for s in symbols:
            if s in close_df.columns:
                series = close_df[s].dropna()
                if len(series) >= 2:
                    curr = float(series.iloc[-1])
                    prev = float(series.iloc[-2])
                    m1 = float(series.iloc[0])
                    chg_1d = ((curr / prev) - 1.0) * 100.0
                    chg_1m = ((curr / m1) - 1.0) * 100.0
                    results[s] = {
                        'latest': curr,
                        'change_1d': chg_1d,
                        'change_1m': chg_1m
                    }
        return results
    except Exception as e:
        print(f"[Macro Barometers] Error fetching data: {e}")
        return {}


@st.cache_data(ttl=900)
def fetch_benchmark_returns():
    """Fetches benchmark returns for Nifty 500 (^CRSLDX) and S&P 500 (^GSPC)."""
    try:
        bm = yf.download(['^CRSLDX', '^GSPC'], period="7mo", progress=False)
        close_df = bm['Close'] if isinstance(bm.columns, pd.MultiIndex) else bm
        close_df = close_df.ffill().bfill()
        
        benchmarks = {}
        for t in ['^CRSLDX', '^GSPC']:
            if t in close_df.columns:
                s = close_df[t].dropna()
                if len(s) >= 20:
                    r1 = ((s.iloc[-1] / s.iloc[-21]) - 1.0) * 100.0 if len(s) > 21 else 0.0
                    r3 = ((s.iloc[-1] / s.iloc[-63]) - 1.0) * 100.0 if len(s) > 63 else 0.0
                    r6 = ((s.iloc[-1] / s.iloc[-126]) - 1.0) * 100.0 if len(s) > 126 else 0.0
                    benchmarks[t] = {'return_1m': float(r1), 'return_3m': float(r3), 'return_6m': float(r6)}
        return benchmarks
    except Exception as e:
        print(f"[Benchmark Returns] Error: {e}")
        return {
            '^CRSLDX': {'return_1m': -3.91, 'return_3m': -4.91, 'return_6m': 0.11},
            '^GSPC': {'return_1m': 1.31, 'return_3m': 2.62, 'return_6m': 11.58}
        }


def fetch_latest_breadth(market):
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM market_breadth_daily WHERE market=? ORDER BY date DESC LIMIT 100", conn, params=(market,))
    conn.close()
    return df


def fetch_latest_liquidity(market):
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM market_liquidity_daily WHERE market=? ORDER BY date DESC LIMIT 252", conn, params=(market,))
    conn.close()
    return df


# =============================================================================
# DUAL MOMENTUM CLASSIFIER HELPER
# =============================================================================

def compute_dual_momentum_status(ret_val, bm_ret):
    """
    Gary Antonacci Dual Momentum evaluation:
    - Absolute Momentum: Is the asset gaining purchasing power (ret_val > 0)?
    - Relative Momentum: Is the asset outperforming the opportunity cost benchmark?
    """
    if ret_val > 0 and ret_val >= bm_ret:
        return "🚀 Dual Alpha", "#10b981", "rgba(16,185,129,0.15)", "Positive return & outperforming benchmark"
    elif ret_val <= 0 and ret_val >= bm_ret:
        return "🛡️ Relative Defense", "#38bdf8", "rgba(56,189,248,0.15)", "Beating benchmark despite negative nominal return"
    elif ret_val > 0 and ret_val < bm_ret:
        return "⚠️ Nominal Gain", "#f59e0b", "rgba(245,158,11,0.15)", "Positive return but lagging benchmark"
    else:
        return "📉 Dual Bleed", "#ef4444", "rgba(239,68,68,0.15)", "Negative nominal return and lagging benchmark"


# =============================================================================
# UI SUB-COMPONENTS
# =============================================================================

def render_canslim_exposure_dial(in_regime, us_regime):
    """Actionable CANSLIM Exposure Dial & Tactical Allocation Directives."""
    in_label = in_regime[0]['regime_label'] if in_regime else "Unknown"
    us_label = us_regime[0]['regime_label'] if us_regime else "Unknown"
    in_dd = in_regime[0]['dd_count'] if in_regime else 0
    us_dd = us_regime[0]['dd_count'] if us_regime else 0

    # Determine Synthesized Posture
    if "Uptrend" in in_label and "Uptrend" in us_label:
        posture_title = "🔥 CONFIRMED UPTREND — FULL RISK-ON"
        posture_badge = "MAXIMUM OFFENSIVE"
        posture_color = "#10b981"
        exposure_range = "80% – 100% Active Equity"
        equity_pct = 90
        safe_haven_pct = 5
        cash_pct = 5
        quote_text = "Three out of four stocks track the market trend. Market conditions are optimal for aggressive growth. Full-size positions on sound base breakouts."
        d1_title, d1_sub = "🚀 Aggressive Breakout Buys", "Deploy full position sizes into leading pivot breakouts with 40%+ above-average volume."
        d2_title, d2_sub = "📈 Pyramid Winning Leaders", "Add 30%–50% on initial pullbacks to 10-day / 21-day EMA; cut losers strictly at 7%–8%."
        d3_title, d3_sub = "💎 Concentrate in Top RS", "Hold True Market Leaders with RS Rating > 85; let compounders run while market expands."
        d4_title, d4_sub = "✅ Confirmed Bull Regime", "Both domestic & global indices holding firmly above rising 50-day moving averages."
    elif "Correction" in in_label or "Correction" in us_label:
        posture_title = "❄️ MARKET IN CORRECTION — CAPITAL PRESERVATION"
        posture_badge = "DEFENSIVE POSTURE"
        posture_color = "#ef4444"
        exposure_range = "0% – 20% Active Equity"
        equity_pct = 15
        safe_haven_pct = 25
        cash_pct = 60
        quote_text = "Protecting principal is rule #1. Breakouts have a 75%+ failure rate in correction. Preserve cash dry powder and wait for a Day 4+ Follow-Through Day."
        d1_title, d1_sub = "🛑 Stand Down on Breakout Buys", "Pivot breakouts suffer heavy shakeouts and reversals. Cease initiating fresh equity entries."
        d2_title, d2_sub = "✂️ 7%–8% Hard Stop Discipline", "Execute non-negotiable stop losses without emotion; absolutely zero averaging down on laggards."
        d3_title, d3_sub = "🔍 Build Relative Strength Watchlist", "Screen daily for resilient stocks holding above 50-day SMA while benchmark indices drop."
        d4_title, d4_sub = "⏳ Awaiting Day 4+ FTD Confirmation", "Preserve dry powder until a high-volume index rally (+1.25%+) confirms institutional accumulation."
    else:
        posture_title = "⚠️ UPTREND UNDER PRESSURE — SELECTIVE CAUTION"
        posture_badge = "ELEVATED CAUTION"
        posture_color = "#f59e0b"
        exposure_range = "40% – 60% Active Equity"
        equity_pct = 50
        safe_haven_pct = 15
        cash_pct = 35
        quote_text = "Market showing distribution symptoms. Reduce commitment size by half. Take quicker profits (+20%–25%) and demand exceptional volume."
        d1_title, d1_sub = "⚠️ Selective Entries Only", "Restrict new purchases exclusively to top 1-2% RS leaders emerging from sound, deep bases."
        d2_title, d2_sub = "✂️ Lock in Partial Profits", "Trim gains at +20% to +25%; raise stops to break-even immediately after initial headway."
        d3_title, d3_sub = "🔍 Eliminate Weak Laggards", "Prune secondary names violating 21-day EMA to elevate cash buffer."
        d4_title, d4_sub = "👀 Monitor Distribution Clusters", "Watch closely for additional institutional distribution days that could trigger full correction."

    dial_html = f"""<div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(20, 30, 48, 0.90) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08); border-left: 6px solid {posture_color};
            border-radius: 14px; padding: 22px 26px; margin-bottom: 25px;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.45);">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 18px;">
<div>
<div style="display: flex; align-items: center; gap: 8px;">
<span style="font-size: 0.76rem; text-transform: uppercase; letter-spacing: 1.5px; color: #94a3b8; font-weight: 700;">
CANSLIM Market Direction & Portfolio Dial
</span>
<span style="background: {posture_color}25; color: {posture_color}; font-size: 0.70rem; padding: 2px 8px; border-radius: 9999px; font-weight: 800; border: 1px solid {posture_color}40;">
{posture_badge}
</span>
</div>
<h2 style="margin: 4px 0 0 0; color: {posture_color}; font-size: 1.6rem; font-weight: 800; letter-spacing: 0.3px;">
{posture_title}
</h2>
</div>
<div style="background: rgba(0, 0, 0, 0.4); border: 1px solid rgba(255,255,255,0.08); padding: 8px 16px; border-radius: 10px; text-align: right;">
<div style="font-size: 0.70rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px;">Recommended Allocation</div>
<div style="font-size: 1.25rem; font-weight: 800; color: #f8fafc;">{exposure_range}</div>
</div>
</div>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; align-items: stretch;">
<div style="background: rgba(15, 23, 42, 0.65); padding: 18px 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); display: flex; flex-direction: column; justify-content: space-between;">
<div>
<div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 10px;">
<div>
<div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Active Equity Ceiling</div>
<div style="font-size: 1.8rem; font-weight: 800; color: {posture_color}; line-height: 1.1;">{equity_pct}%</div>
</div>
<div style="text-align: right;">
<div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Defensive Dry Powder</div>
<div style="font-size: 1.8rem; font-weight: 800; color: #38bdf8; line-height: 1.1;">{safe_haven_pct + cash_pct}%</div>
</div>
</div>
<div style="width: 100%; height: 20px; background: rgba(51, 65, 85, 0.6); border-radius: 9999px; overflow: hidden; display: flex; box-shadow: inset 0 2px 4px rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.05);">
<div style="width: {equity_pct}%; background: {posture_color}; transition: width 0.5s ease;" title="Active Equity: {equity_pct}%"></div>
<div style="width: {safe_haven_pct}%; background: #eab308;" title="Safe Havens (Gold/Silver): {safe_haven_pct}%"></div>
<div style="width: {cash_pct}%; background: #475569;" title="Cash / Liquid: {cash_pct}%"></div>
</div>
<div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #cbd5e1; margin-top: 10px;">
<span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{posture_color};margin-right:5px;"></span>Active Equity: <strong>{equity_pct}%</strong></span>
<span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#eab308;margin-right:5px;"></span>Safe Havens: <strong>{safe_haven_pct}%</strong></span>
<span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#475569;margin-right:5px;"></span>Cash: <strong>{cash_pct}%</strong></span>
</div>
</div>
<div style="margin-top: 14px; font-size: 0.80rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 10px; font-style: italic; line-height: 1.4;">
"{quote_text}"
</div>
</div>
<div style="display: flex; flex-direction: column; gap: 8px;">
<div style="background: rgba(15, 23, 42, 0.65); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
<div style="font-weight: 700; color: #f8fafc; font-size: 0.84rem;">{d1_title}</div>
<div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">{d1_sub}</div>
</div>
<div style="background: rgba(15, 23, 42, 0.65); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
<div style="font-weight: 700; color: #f8fafc; font-size: 0.84rem;">{d2_title}</div>
<div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">{d2_sub}</div>
</div>
<div style="background: rgba(15, 23, 42, 0.65); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
<div style="font-weight: 700; color: #f8fafc; font-size: 0.84rem;">{d3_title}</div>
<div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">{d3_sub}</div>
</div>
<div style="background: rgba(15, 23, 42, 0.65); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
<div style="font-weight: 700; color: #f8fafc; font-size: 0.84rem;">{d4_title}</div>
<div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">{d4_sub}</div>
</div>
</div>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-top: 16px; font-size: 0.78rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 12px;">
<div style="display: flex; gap: 16px;">
<span>🇮🇳 <strong>India (Nifty 500):</strong> <span style="color: {'#10b981' if 'Uptrend' in in_label else '#ef4444'}; font-weight: 700;">{in_label}</span> ({in_dd} Dist Days)</span>
<span>•</span>
<span>🇺🇸 <strong>United States (S&P 500):</strong> <span style="color: {'#10b981' if 'Uptrend' in us_label else '#ef4444'}; font-weight: 700;">{us_label}</span> ({us_dd} Dist Days)</span>
</div>
<div style="color: #64748b; font-size: 0.74rem;">
Last Telemetry Check: {datetime.now().strftime('%d %b %Y %H:%M IST')}
</div>
</div>
</div>"""
    render_html(dial_html)


def render_macro_barometers(baro_dict):
    """Renders 4 Institutional Inter-Market Leading Barometer Cards."""
    st.markdown("### 📡 Inter-Market Macro Barometers (Tripwires)")
    st.caption("Live cross-market signals governing liquidity, institutional risk appetite, currency flight, and breakout odds.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # 1. Currencies (USD/INR & DXY)
    usdinr = baro_dict.get('USDINR=X', {'latest': 96.41, 'change_1d': 0.09, 'change_1m': 2.10})
    dxy = baro_dict.get('DX-Y.NYB', {'latest': 101.87, 'change_1d': 0.0, 'change_1m': 3.10})
    dxy_status = "Dollar Strong (EM Drain)" if dxy['latest'] >= 100 else "Dollar Soft (EM Inflow)"
    dxy_color = "#ef4444" if dxy['latest'] >= 100 else "#10b981"
    
    with col1:
        render_html(f"""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); 
                    border-radius: 12px; padding: 18px; min-height: 185px; backdrop-filter: blur(8px);
                    transition: border-color 0.2s ease, transform 0.2s ease;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.78rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">💵 FX & Dollar Flow</span>
                <span style="background: rgba(239, 68, 68, 0.15); color: {dxy_color}; font-size: 0.70rem; padding: 2px 7px; border-radius: 6px; font-weight: 700;">{dxy_status}</span>
            </div>
            <div style="font-size: 1.55rem; font-weight: 800; color: #f8fafc; margin-bottom: 2px;">
                USD/INR: ₹{usdinr['latest']:.2f}
            </div>
            <div style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 10px;">
                1D: <span style="color: {'#10b981' if usdinr['change_1d'] < 0 else '#ef4444'}; font-weight: 700;">{usdinr['change_1d']:+.2f}%</span> | 
                1M: <span style="color: {'#10b981' if usdinr['change_1m'] < 0 else '#ef4444'}; font-weight: 700;">{usdinr['change_1m']:+.2f}%</span>
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
                DXY: <strong style="color: #f8fafc;">{dxy['latest']:.2f}</strong> ({dxy['change_1m']:+.1f}% 1M)<br>
                <span style="color: #64748b;">Signals FII capital flight pressure on Indian equities</span>
            </div>
        </div>
        """)

    # 2. Yields (US 10Y Yield ^TNX)
    tnx = baro_dict.get('^TNX', {'latest': 5.27, 'change_1d': -0.72, 'change_1m': 9.72})
    tnx_val = tnx['latest']
    tnx_status = "Valuation Drag (>4.5%)" if tnx_val >= 4.5 else "Accommodative (<4.0%)"
    tnx_color = "#ef4444" if tnx_val >= 4.5 else "#10b981"
    
    with col2:
        render_html(f"""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); 
                    border-radius: 12px; padding: 18px; min-height: 185px; backdrop-filter: blur(8px);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.78rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">📈 10Y US Treasury</span>
                <span style="background: rgba(239, 68, 68, 0.15); color: {tnx_color}; font-size: 0.70rem; padding: 2px 7px; border-radius: 6px; font-weight: 700;">{tnx_status}</span>
            </div>
            <div style="font-size: 1.55rem; font-weight: 800; color: #f8fafc; margin-bottom: 2px;">
                ^TNX: {tnx_val:.2f}%
            </div>
            <div style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 10px;">
                1D: <span style="color: {'#10b981' if tnx['change_1d'] < 0 else '#ef4444'}; font-weight: 700;">{tnx['change_1d']:+.2f}%</span> | 
                1M: <span style="color: {'#10b981' if tnx['change_1m'] < 0 else '#ef4444'}; font-weight: 700;">{tnx['change_1m']:+.2f}%</span>
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
                Equity Multiple Compression<br>
                <span style="color: #64748b;">Rising discount rate pressures high-growth P/E multiples</span>
            </div>
        </div>
        """)

    # 3. Volatility & Breakout Odds (India VIX & US VIX)
    invix = baro_dict.get('^INDIAVIX', {'latest': 13.61, 'change_1d': 0.0, 'change_1m': 21.9})
    usvix = baro_dict.get('^VIX', {'latest': 15.3, 'change_1d': 0.0, 'change_1m': 1.44})
    
    if invix['latest'] < 13.0:
        odds_label = "High Odds (>75%)"
        odds_color = "#10b981"
        odds_desc = "Calm market; sound pivots follow through"
    elif invix['latest'] <= 17.0:
        odds_label = "Neutral (~50%)"
        odds_color = "#f59e0b"
        odds_desc = "Choppy regime; demand high volume"
    else:
        odds_label = "Whipsaw Risk (<30%)"
        odds_color = "#ef4444"
        odds_desc = "High failure rate; avoid pivot buys"

    with col3:
        render_html(f"""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); 
                    border-radius: 12px; padding: 18px; min-height: 185px; backdrop-filter: blur(8px);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.78rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">🌪️ Volatility & Odds</span>
                <span style="background: rgba(245, 158, 11, 0.15); color: {odds_color}; font-size: 0.70rem; padding: 2px 7px; border-radius: 6px; font-weight: 700;">{odds_label}</span>
            </div>
            <div style="font-size: 1.55rem; font-weight: 800; color: #f8fafc; margin-bottom: 2px;">
                India VIX: {invix['latest']:.2f}
            </div>
            <div style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 10px;">
                1M Change: <span style="color: {'#ef4444' if invix['change_1m'] > 0 else '#10b981'}; font-weight: 700;">{invix['change_1m']:+.1f}%</span> | 
                US VIX: <strong style="color: #f8fafc;">{usvix['latest']:.1f}</strong>
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
                CANSLIM Breakout Probability<br>
                <span style="color: #64748b;">{odds_desc}</span>
            </div>
        </div>
        """)

    # 4. Energy (Brent Crude BZ=F)
    brent = baro_dict.get('BZ=F', {'latest': 97.27, 'change_1d': -3.04, 'change_1m': -0.66})
    brent_val = brent['latest']
    brent_status = "Inflation Drag (>$85)" if brent_val >= 85 else "Favorable (<$80)"
    brent_color = "#ef4444" if brent_val >= 85 else "#10b981"
    
    with col4:
        render_html(f"""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); 
                    border-radius: 12px; padding: 18px; min-height: 185px; backdrop-filter: blur(8px);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.78rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">🛢️ Brent Crude Oil</span>
                <span style="background: rgba(239, 68, 68, 0.15); color: {brent_color}; font-size: 0.70rem; padding: 2px 7px; border-radius: 6px; font-weight: 700;">{brent_status}</span>
            </div>
            <div style="font-size: 1.55rem; font-weight: 800; color: #f8fafc; margin-bottom: 2px;">
                ${brent_val:.2f} <span style="font-size: 0.82rem; font-weight: 500; color: #94a3b8;">/bbl</span>
            </div>
            <div style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 10px;">
                1D: <span style="color: {'#ef4444' if brent['change_1d'] > 0 else '#10b981'}; font-weight: 700;">{brent['change_1d']:+.2f}%</span> | 
                1M: <span style="color: {'#ef4444' if brent['change_1m'] > 0 else '#10b981'}; font-weight: 700;">{brent['change_1m']:+.2f}%</span>
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
                Operating Margin Stress<br>
                <span style="color: #64748b;">High crude increases India CAD and input inflation</span>
            </div>
        </div>
        """)


def render_market_column(market_code, market_name):
    """Renders side-by-side market regime, breadth and liquidity."""
    st.subheader(f"{'🇮🇳' if market_code=='IN' else '🇺🇸'} {market_name}")
    
    # 1. Regime
    regimes = get_latest_global_regime(market=market_code)
    if not regimes:
        st.warning(f"No regime data for {market_code}. Please run macro sync.")
        return
        
    reg = regimes[0]
    is_up = "Uptrend" in reg['regime_label']
    is_corr = "Correction" in reg['regime_label']
    color = "#10b981" if is_up else "#ef4444" if is_corr else "#f59e0b"
    bg_color = "rgba(16, 185, 129, 0.1)" if is_up else "rgba(239, 68, 68, 0.1)" if is_corr else "rgba(245, 158, 11, 0.1)"
    
    sma200 = float(reg.get('sma200', reg['close']))
    vs_50 = ((reg['close'] / reg['sma50']) - 1.0) * 100.0
    vs_200 = ((reg['close'] / sma200) - 1.0) * 100.0 if sma200 > 0 else 0.0

    vs_50_color = "#10b981" if vs_50 > 0 else "#ef4444"
    vs_200_color = "#10b981" if vs_200 > 0 else "#ef4444"

    ftd_val = reg.get('ftd_detected', False)
    ftd_badge = '<span style="color: #10b981; font-weight: 700;">🟢 Confirmed</span>' if ftd_val else '<span style="color: #94a3b8;">Awaiting Day 4+</span>'
    
    render_html(f"""
    <div style="background: {bg_color}; border: 1px solid {color}40; border-left: 5px solid {color};
                padding: 14px 18px; border-radius: 10px; margin-bottom: 18px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-weight: 800; color: {color}; font-size: 1.12rem;">{reg['regime_label']}</span>
                <span style="font-size: 0.74rem; margin-left: 8px; padding: 2px 7px; border-radius: 4px; background: rgba(255,255,255,0.06); color: #cbd5e1;">
                    FTD: {ftd_badge}
                </span>
            </div>
            <span style="font-size: 0.75rem; color: #94a3b8;">As of {reg['date']}</span>
        </div>
        <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 6px;">
            {reg['benchmark_ticker']} Close: <strong>{reg['close']:,.2f}</strong> | 
            Dist Days: <strong style="color: {'#ef4444' if reg['dd_count'] >= 5 else '#10b981'};">{reg['dd_count']}</strong> | 
            vs 50SMA: <strong style="color: {vs_50_color};">{vs_50:+.1f}%</strong> | 
            vs 200SMA: <strong style="color: {vs_200_color};">{vs_200:+.1f}%</strong>
        </div>
    </div>
    """)
    
    # 2. Breadth Telemetry
    breadth_df = fetch_latest_breadth(market_code)
    if not breadth_df.empty:
        latest_b = breadth_df.iloc[0]
        nnh = int(latest_b.get('net_new_highs', 0))
        nh_count = int(latest_b.get('new_highs_count', 0))
        nl_count = int(latest_b.get('new_lows_count', 0))

        above_50 = float(latest_b.get('above_50_pct', 0.0))
        above_200 = float(latest_b.get('above_200_pct', 0.0))

        adv = int(latest_b.get('advances', 0))
        dec = int(latest_b.get('declines', 0))
        ad_ratio = (adv / max(1, dec))
        up_vol_ratio = float(latest_b.get('up_down_volume_ratio', 1.0))
        
        nnh_color = "#10b981" if nnh > 0 else "#ef4444"
        nnh_bg = "rgba(16,185,129,0.15)" if nnh > 0 else "rgba(239,68,68,0.15)"
        nnh_label = "Expansion" if nnh > 50 else "Severe Contraction" if nnh < -50 else "Contraction" if nnh < 0 else "Neutral"
        
        a50_color = "#10b981" if above_50 >= 60 else "#f59e0b" if above_50 >= 40 else "#ef4444"
        a50_bg = "rgba(16,185,129,0.15)" if above_50 >= 60 else "rgba(245,158,11,0.15)" if above_50 >= 40 else "rgba(239,68,68,0.15)"
        a50_label = "Bullish" if above_50 >= 60 else "Divergent" if above_50 >= 40 else "Washed Out"

        a200_color = "#10b981" if above_200 >= 60 else "#f59e0b" if above_200 >= 45 else "#ef4444"
        a200_bg = "rgba(16,185,129,0.15)" if above_200 >= 60 else "rgba(245,158,11,0.15)" if above_200 >= 45 else "rgba(239,68,68,0.15)"
        a200_label = "Secular Bull" if above_200 >= 60 else "Neutral Base" if above_200 >= 45 else "Secular Bear"

        ad_color = "#10b981" if ad_ratio >= 1.5 else "#f59e0b" if ad_ratio >= 0.8 else "#ef4444"
        ad_bg = "rgba(16,185,129,0.15)" if ad_ratio >= 1.5 else "rgba(245,158,11,0.15)" if ad_ratio >= 0.8 else "rgba(239,68,68,0.15)"
        ad_badge = "Strong Inflow" if ad_ratio >= 1.5 else "Balanced" if ad_ratio >= 0.8 else "Distribution"
        
        # Row 1: Net New Highs & Stocks > 50 SMA
        col1, col2 = st.columns(2)
        with col1:
            render_html(f"""
            <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 8px;">
                <div style="font-size: 0.70rem; color: #94a3b8; text-transform: uppercase;">Net New Highs</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: {nnh_color}; margin: 2px 0;">
                    {nnh:+d} 
                    <span style="font-size: 0.68rem; font-weight: 700; padding: 2px 6px; border-radius: 4px; background: {nnh_bg}; vertical-align: middle;">{nnh_label}</span>
                </div>
                <div style="font-size: 0.70rem; color: #64748b;">Highs: {nh_count} | Lows: {nl_count}</div>
            </div>
            """)
        with col2:
            render_html(f"""
            <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 8px;">
                <div style="font-size: 0.70rem; color: #94a3b8; text-transform: uppercase;">Stocks > 50 SMA</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: {a50_color}; margin: 2px 0;">
                    {above_50:.1f}% 
                    <span style="font-size: 0.68rem; font-weight: 700; padding: 2px 6px; border-radius: 4px; background: {a50_bg}; vertical-align: middle;">{a50_label}</span>
                </div>
                <div style="font-size: 0.70rem; color: #64748b;">Intermediate Tactical Breadth</div>
            </div>
            """)

        # Row 2: Stocks > 200 SMA & Advance / Decline Volume Ratio
        col3, col4 = st.columns(2)
        with col3:
            render_html(f"""
            <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 12px;">
                <div style="font-size: 0.70rem; color: #94a3b8; text-transform: uppercase;">Stocks > 200 SMA</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: {a200_color}; margin: 2px 0;">
                    {above_200:.1f}% 
                    <span style="font-size: 0.68rem; font-weight: 700; padding: 2px 6px; border-radius: 4px; background: {a200_bg}; vertical-align: middle;">{a200_label}</span>
                </div>
                <div style="font-size: 0.70rem; color: #64748b;">Secular Multi-Year Health</div>
            </div>
            """)
        with col4:
            render_html(f"""
            <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 12px;">
                <div style="font-size: 0.70rem; color: #94a3b8; text-transform: uppercase;">A/D & Up/Down Volume</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: {ad_color}; margin: 2px 0;">
                    {ad_ratio:.2f}x 
                    <span style="font-size: 0.68rem; font-weight: 700; padding: 2px 6px; border-radius: 4px; background: {ad_bg}; vertical-align: middle;">{ad_badge}</span>
                </div>
                <div style="font-size: 0.70rem; color: #64748b;">Adv: {adv} | Dec: {dec} | Vol: {up_vol_ratio:.1f}x</div>
            </div>
            """)
        
        # Breadth Bar Chart
        breadth_df['date'] = pd.to_datetime(breadth_df['date'])
        breadth_df = breadth_df.sort_values('date')
        fig_b = px.bar(breadth_df.tail(60), x='date', y='net_new_highs', title=f"{market_name} Net New Highs (60D)")
        fig_b.update_traces(marker_color=['#10b981' if val > 0 else '#ef4444' for val in breadth_df.tail(60)['net_new_highs']])
        fig_b = apply_plotly_theme(fig_b)
        fig_b.update_layout(
            height=230, 
            margin=dict(l=0, r=0, t=32, b=0),
            xaxis_title=None,
            yaxis_title=None
        )
        st.plotly_chart(fig_b, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info(f"No daily breadth history recorded for {market_code}.")
        
    # 3. Liquidity
    liq_df = fetch_latest_liquidity(market_code)
    if not liq_df.empty:
        liq_df['date'] = pd.to_datetime(liq_df['date'])
        liq_df = liq_df.sort_values('date')
        
        unit_label = "₹ Cr" if market_code == 'IN' else "Turnover Index"
        fig_l = px.line(liq_df, x='date', y='monthly_turnover_k_cr', title=f"{market_name} Monthly Turnover ({unit_label}) & 200SMA")
        fig_l.update_traces(line_color='#38bdf8', line_width=2)
        if 'sma_200' in liq_df.columns:
            fig_l.add_scatter(x=liq_df['date'], y=liq_df['sma_200'], mode='lines', name='200 SMA', line=dict(color='#f59e0b', dash='dot', width=1.5))
        fig_l = apply_plotly_theme(fig_l)
        fig_l.update_layout(
            height=230, 
            margin=dict(l=0, r=0, t=32, b=0), 
            showlegend=False,
            xaxis_title=None,
            yaxis_title=None
        )
        st.plotly_chart(fig_l, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info(f"No liquidity turnover data recorded for {market_code}.")


def render_asset_rotation_radar(etf_data, benchmarks):
    """
    Renders Segmented 4-Category Asset Rotation Radar with Dual Momentum Classification:
    1. Precious Metals & Safe Havens
    2. Indian Sectoral & Factor ETFs
    3. US Thematics & Sectors
    4. Global Country Flows
    5. Master Cross-Asset Table
    """
    st.markdown("### 🔄 Where is the Momentum? — Asset Rotation Radar")
    st.caption("Cross-asset capital allocation across Precious Metals, Indian Sectors, US Thematics, and Global Country Indices with Gary Antonacci Dual Momentum classification.")
    
    if not etf_data:
        st.warning("No ETF Momentum data found. Please run sync_macro.py.")
        return

    df = pd.DataFrame(etf_data)
    
    # Ensure numeric columns for price telemetry
    df['dist_52w_high'] = pd.to_numeric(df.get('dist_52w_high', 0.0), errors='coerce').fillna(0.0)
    df['close'] = pd.to_numeric(df.get('close', 0.0), errors='coerce').fillna(0.0)
    df['high_52w'] = pd.to_numeric(df.get('high_52w', 0.0), errors='coerce').fillna(0.0)

    # Enrich with Dual Momentum Status & RS Alpha Spread
    in_bm_1m = benchmarks.get('^CRSLDX', {}).get('return_1m', -3.91)
    in_bm_3m = benchmarks.get('^CRSLDX', {}).get('return_3m', -4.91)
    in_bm_6m = benchmarks.get('^CRSLDX', {}).get('return_6m', 0.11)

    us_bm_1m = benchmarks.get('^GSPC', {}).get('return_1m', 1.31)
    us_bm_3m = benchmarks.get('^GSPC', {}).get('return_3m', 2.62)
    us_bm_6m = benchmarks.get('^GSPC', {}).get('return_6m', 11.58)

    def assign_dual_mom_and_alpha(row):
        ctry = row.get('country', 'US')
        bm_1m = in_bm_1m if ctry == 'IN' else us_bm_1m
        bm_3m = in_bm_3m if ctry == 'IN' else us_bm_3m
        bm_6m = in_bm_6m if ctry == 'IN' else us_bm_6m
        
        s1, c1, bg1, _ = compute_dual_momentum_status(row['return_1m'], bm_1m)
        s3, c3, bg3, _ = compute_dual_momentum_status(row['return_3m'], bm_3m)
        s6, c6, bg6, _ = compute_dual_momentum_status(row['return_6m'], bm_6m)
        
        a1 = row['return_1m'] - bm_1m
        a3 = row['return_3m'] - bm_3m
        a6 = row['return_6m'] - bm_6m
        
        return pd.Series([s1, s3, s6, c1, c3, c6, bg1, bg3, bg6, a1, a3, a6], 
                         index=['status_1m', 'status_3m', 'status_6m', 
                                'color_1m', 'color_3m', 'color_6m', 
                                'bg_1m', 'bg_3m', 'bg_6m',
                                'alpha_1m', 'alpha_3m', 'alpha_6m'])

    df[['status_1m', 'status_3m', 'status_6m', 
        'color_1m', 'color_3m', 'color_6m', 
        'bg_1m', 'bg_3m', 'bg_6m',
        'alpha_1m', 'alpha_3m', 'alpha_6m']] = df.apply(assign_dual_mom_and_alpha, axis=1)

    # CANSLIM 52-Week High Proximity Classifier
    def classify_52w_proximity(dist_val):
        if dist_val >= -5.0:
            return "🔥 Coiling (<5%)", "#10b981", "rgba(16,185,129,0.15)"
        elif dist_val >= -15.0:
            return "🛡️ Base Depth (5-15%)", "#38bdf8", "rgba(56,189,248,0.15)"
        elif dist_val >= -25.0:
            return "⚠️ Deep Pullback", "#f59e0b", "rgba(245,158,11,0.15)"
        else:
            return "📉 Severe Downtrend", "#ef4444", "rgba(239,68,68,0.15)"

    prox_series = df['dist_52w_high'].apply(lambda d: pd.Series(classify_52w_proximity(d), index=['prox_label', 'prox_color', 'prox_bg']))
    df[['prox_label', 'prox_color', 'prox_bg']] = prox_series

    # 4 Category Tabs + Master Table
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🥇 Precious Metals & Safe Havens",
        "🇮🇳 Indian Sectoral & Factor ETFs",
        "🇺🇸 US Thematic & Mega-Trends",
        "🌐 Global Country Capital Flows",
        "📑 Master Cross-Asset Matrix"
    ])

    def render_category_view(cat_name, cat_desc, default_bm_name, default_bm_val):
        cat_df = df[df['category'] == cat_name].copy()
        if cat_df.empty:
            st.info(f"No ETFs currently categorized under {cat_name}.")
            return
            
        st.markdown(f"**{cat_desc}**")
        
        # Horizon Selector
        col_ctrl1, col_ctrl2 = st.columns([2, 3])
        with col_ctrl1:
            horizon = st.radio(
                "Rotation Horizon",
                options=["1-Month Tactical", "3-Month Intermediate", "6-Month Structural"],
                horizontal=True,
                key=f"rad_lite_{cat_name}"
            )
        
        h_col = "return_1m" if "1-Month" in horizon else "return_3m" if "3-Month" in horizon else "return_6m"
        a_col = "alpha_1m" if "1-Month" in horizon else "alpha_3m" if "3-Month" in horizon else "alpha_6m"
        s_col = "status_1m" if "1-Month" in horizon else "status_3m" if "3-Month" in horizon else "status_6m"
        c_col = "color_1m" if "1-Month" in horizon else "color_3m" if "3-Month" in horizon else "color_6m"
        bg_col = "bg_1m" if "1-Month" in horizon else "bg_3m" if "3-Month" in horizon else "bg_6m"
        
        # Determine Benchmark for this category and horizon
        if cat_name == "Indian Sectoral & Factor":
            bm_val = in_bm_1m if "1-Month" in horizon else in_bm_3m if "3-Month" in horizon else in_bm_6m
            bm_label = f"Nifty 500 ({bm_val:+.1f}%)"
        else:
            bm_val = us_bm_1m if "1-Month" in horizon else us_bm_3m if "3-Month" in horizon else us_bm_6m
            bm_label = f"S&P 500 ({bm_val:+.1f}%)"

        cat_df = cat_df.sort_values(h_col, ascending=False)

        # Leader Spotlight Cards
        leaders = cat_df.head(3)
        cols = st.columns(min(3, len(leaders)))
        medals = ["🥇", "🥈", "🥉"]
        for idx, (_, r) in enumerate(leaders.iterrows()):
            with cols[idx]:
                ret = r[h_col]
                ret_color = r[c_col]
                stat_badge = r[s_col]
                badge_bg = r[bg_col]
                clean_t, clean_n = format_display_label(r['ticker'], r['display_name'])
                
                alpha_val = r[a_col]
                alpha_color = "#10b981" if alpha_val > 0 else "#ef4444"
                alpha_bg = "rgba(16,185,129,0.15)" if alpha_val > 0 else "rgba(239,68,68,0.15)"
                
                dist_val = r['dist_52w_high']
                prox_lbl = r['prox_label']
                prox_col = r['prox_color']
                prox_bg = r['prox_bg']

                render_html(f"""
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); 
                            border-top: 3px solid {ret_color};
                            border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.74rem; color: #94a3b8; font-weight: 800; text-transform: uppercase;">
                            {medals[idx]} #{idx+1} Leader
                        </span>
                        <span style="font-size: 0.72rem; padding: 2px 7px; border-radius: 6px; background: {badge_bg}; color: {ret_color}; font-weight: 700; border: 1px solid {ret_color}40;">
                            {stat_badge}
                        </span>
                    </div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: #f8fafc; margin: 4px 0 2px 0;">
                        {clean_t}
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                        {clean_n}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 8px;">
                        <span style="font-size: 1.45rem; font-weight: 800; color: {ret_color};">
                            {ret:+.2f}%
                        </span>
                        <span style="font-size: 0.78rem; font-weight: 700; color: {alpha_color}; background: {alpha_bg}; padding: 3px 8px; border-radius: 5px; border: 1px solid {alpha_color}40;">
                            {alpha_val:+.2f}% α vs BM
                        </span>
                    </div>
                    <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.06); display: flex; justify-content: space-between; align-items: center; font-size: 0.76rem;">
                        <span style="color: #94a3b8;">52W Proximity:</span>
                        <span style="font-weight: 800; color: {prox_col};">
                            {dist_val:+.1f}% <span style="font-size: 0.68rem; font-weight: 700; padding: 1px 6px; border-radius: 4px; background: {prox_bg};">{prox_lbl}</span>
                        </span>
                    </div>
                </div>
                """)

        # Horizontal Bar Chart with Benchmark Reference Line
        fig = go.Figure()
        
        formatted_labels = []
        for t, n in zip(cat_df['ticker'], cat_df['display_name']):
            ct, cn = format_display_label(t, n)
            formatted_labels.append(f"<b>{ct}</b> <span style='font-size:11px;color:#94a3b8;'>({cn})</span>")

        fig.add_trace(go.Bar(
            y=formatted_labels,
            x=cat_df[h_col],
            orientation='h',
            marker=dict(
                color=cat_df[c_col], 
                line=dict(color='rgba(255,255,255,0.1)', width=0.5)
            ),
            text=[f"{v:+.1f}% (α {a:+.1f}%)" for v, a in zip(cat_df[h_col], cat_df[a_col])],
            textposition='auto',
            hovertext=[
                f"<b>{t}</b> — {n}<br>"
                f"<b>{horizon} Return:</b> {v:+.2f}%<br>"
                f"<b>RS Alpha vs Benchmark ({bm_label}):</b> {a:+.2f}%<br>"
                f"<b>Distance from 52-Week High:</b> {d:+.2f}% ({pl})<br>"
                f"<b>Dual Momentum:</b> {s}<br>"
                f"<span style='color:#94a3b8;'>1M: {r1:+.1f}% (α {a1:+.1f}%) | 3M: {r3:+.1f}% (α {a3:+.1f}%) | 6M: {r6:+.1f}% (α {a6:+.1f}%)</span>"
                for t, n, v, a, d, pl, s, r1, a1, r3, a3, r6, a6 in zip(
                    cat_df['ticker'], cat_df['display_name'], cat_df[h_col], cat_df[a_col],
                    cat_df['dist_52w_high'], cat_df['prox_label'], cat_df[s_col],
                    cat_df['return_1m'], cat_df['alpha_1m'],
                    cat_df['return_3m'], cat_df['alpha_3m'],
                    cat_df['return_6m'], cat_df['alpha_6m']
                )
            ],
            hoverinfo='text'
        ))
        
        # Add Benchmark Reference Line
        fig.add_vline(
            x=bm_val, 
            line_dash="dash", 
            line_color="#f59e0b", 
            line_width=2,
            annotation_text=f"Benchmark: {bm_label}",
            annotation_position="top left" if bm_val < 0 else "top right",
            annotation_font=dict(color="#f59e0b", size=11, family="Inter")
        )
        
        # Add Zero Reference Line
        fig.add_vline(
            x=0, 
            line_color="rgba(255, 255, 255, 0.25)", 
            line_width=1
        )
        
        fig = apply_plotly_theme(fig)
        fig.update_layout(
            height=max(340, len(cat_df) * 28 + 60),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(
                title=f"{horizon} Return (%)", 
                zeroline=False,
                gridcolor="rgba(255,255,255,0.05)"
            ),
            margin=dict(l=10, r=30, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Dynamic CANSLIM Institutional Insight Callout
        top_lead = cat_df.iloc[0]
        defenders = cat_df[cat_df[h_col] >= bm_val]
        coiling = cat_df[cat_df['dist_52w_high'] >= -10.0]
        ct_lead, cn_lead = format_display_label(top_lead['ticker'], top_lead['display_name'])
        
        render_html(f"""
        <div style="background: rgba(30, 41, 59, 0.5); border-left: 4px solid #38bdf8; 
                    padding: 10px 16px; border-radius: 6px; font-size: 0.82rem; color: #cbd5e1; margin-bottom: 15px;">
            💡 <strong>CANSLIM Leadership Insight:</strong> 
            <strong>{len(defenders)} of {len(cat_df)}</strong> assets are outperforming the {bm_label} benchmark.
            <strong>{len(coiling)} assets</strong> are coiling within 10% of their 52-Week High.
            Top alpha leader is <strong>{ct_lead} ({cn_lead}: {top_lead[h_col]:+.2f}%, {top_lead[a_col]:+.2f}% α)</strong> holding <strong>{top_lead['dist_52w_high']:+.1f}% from 52W High ({top_lead['prox_label']})</strong>.
        </div>
        """)

        # Data Table
        with st.expander(f"📋 View Full {cat_name} Table ({len(cat_df)} Assets)"):
            disp_df = cat_df[['ticker', 'display_name', 'return_1m', 'alpha_1m', 'return_3m', 'alpha_3m', 'return_6m', 'alpha_6m', 'dist_52w_high', 'prox_label', s_col, 'country']].copy()
            st.dataframe(
                disp_df,
                column_config={
                    "ticker": "Ticker",
                    "display_name": "Asset Name",
                    "return_1m": st.column_config.NumberColumn("1M Ret (%)", format="%.2f"),
                    "alpha_1m": st.column_config.NumberColumn("1M Alpha (%)", format="%+.2f"),
                    "return_3m": st.column_config.NumberColumn("3M Ret (%)", format="%.2f"),
                    "alpha_3m": st.column_config.NumberColumn("3M Alpha (%)", format="%+.2f"),
                    "return_6m": st.column_config.NumberColumn("6M Ret (%)", format="%.2f"),
                    "alpha_6m": st.column_config.NumberColumn("6M Alpha (%)", format="%+.2f"),
                    "dist_52w_high": st.column_config.NumberColumn("% Off 52W High", format="%.2f%%"),
                    "prox_label": "52W Base Proximity",
                    s_col: "Dual Momentum Status",
                    "country": "Region"
                },
                use_container_width=True,
                hide_index=True
            )

    # 1. Precious Metals & Commodities
    with tab1:
        render_category_view(
            "Precious Metals & Commodities", 
            "Safe haven capital preservation vehicles (Gold, Silver, Commodities) protecting capital during broad equity corrections.",
            "S&P 500", us_bm_1m
        )

    # 2. Indian Sectoral & Factor ETFs
    with tab2:
        render_category_view(
            "Indian Sectoral & Factor", 
            "Domestic Indian institutional sector rotation & smart-beta factor ETFs (Bank, IT, Pharma, Auto, CPSE, Defence, Alpha, Momentum).",
            "Nifty 500", in_bm_1m
        )

    # 3. US Thematics & Sectors
    with tab3:
        render_category_view(
            "US Thematic & Sector", 
            "Wall Street mega-thematics, leading industry groups, and high-beta innovation leaders (Semiconductors, Bitcoin, Tech, Defense, Clean Energy).",
            "S&P 500", us_bm_1m
        )

    # 4. Global Country Flows
    with tab4:
        render_category_view(
            "Global Country Flows", 
            "Cross-border sovereign equity allocation tracking foreign institutional capital rotation across US, India, China, Brazil, Japan, and Europe.",
            "S&P 500", us_bm_1m
        )

    # 5. Master Cross-Asset Matrix
    with tab5:
        st.markdown("**Complete Global Cross-Asset Matrix (80+ Global Instruments)**")
        
        col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 2, 2])
        with col_f1:
            search_q = st.text_input("🔍 Search Asset or Ticker", placeholder="e.g. Gold, Silver, SOXX, Bank...")
        with col_f2:
            sel_cats = st.multiselect("Filter Asset Classes", options=sorted(df['category'].unique()), default=sorted(df['category'].unique()))
        with col_f3:
            sel_stats = st.multiselect("Filter Dual Momentum", options=sorted(df['status_1m'].unique()), default=sorted(df['status_1m'].unique()))
        with col_f4:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            only_leaders = st.checkbox("🎯 Leading Bases Only (≤ 15% off High)", value=False)
            
        filtered_df = df[df['category'].isin(sel_cats) & df['status_1m'].isin(sel_stats)].copy()
        if only_leaders:
            filtered_df = filtered_df[filtered_df['dist_52w_high'] >= -15.0]
        if search_q:
            filtered_df = filtered_df[
                filtered_df['ticker'].str.contains(search_q, case=False, na=False) |
                filtered_df['display_name'].str.contains(search_q, case=False, na=False)
            ]
            
        st.dataframe(
            filtered_df[['ticker', 'display_name', 'category', 'return_1m', 'alpha_1m', 'return_3m', 'alpha_3m', 'return_6m', 'alpha_6m', 'dist_52w_high', 'prox_label', 'status_1m', 'country']].sort_values('return_1m', ascending=False),
            column_config={
                "ticker": "Ticker",
                "display_name": "Asset Name",
                "category": "Asset Class",
                "return_1m": st.column_config.NumberColumn("1M Ret (%)", format="%.2f"),
                "alpha_1m": st.column_config.NumberColumn("1M Alpha (%)", format="%+.2f"),
                "return_3m": st.column_config.NumberColumn("3M Ret (%)", format="%.2f"),
                "alpha_3m": st.column_config.NumberColumn("3M Alpha (%)", format="%+.2f"),
                "return_6m": st.column_config.NumberColumn("6M Ret (%)", format="%.2f"),
                "alpha_6m": st.column_config.NumberColumn("6M Alpha (%)", format="%+.2f"),
                "dist_52w_high": st.column_config.NumberColumn("% Off 52W High", format="%.2f%%"),
                "prox_label": "52W Base Proximity",
                "status_1m": "1M Dual Momentum",
                "country": "Region"
            },
            use_container_width=True,
            hide_index=True
        )


# =============================================================================
# MAIN CONTROLLER
# =============================================================================

def main():
    # Modern Action Toolbar & Header
    col_hdr, col_btn = st.columns([4, 1.2])
    with col_hdr:
        render_header(
            "Global Macro Terminal", 
            "Cross-Asset Rotation, Inter-Market Barometers, and CANSLIM Allocation Playbook.", 
            icon="🌍"
        )
    with col_btn:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Sync Macro Data", help="Fetch fresh benchmark and ETF momentum prices", use_container_width=True):
            with st.spinner("Syncing Global Macro Telemetry & Barometers..."):
                sync_all_data()
                st.cache_data.clear()
                st.success("Global Macro Data Synchronized!")
                st.rerun()
    
    # 1. Fetch Regimes and Barometers
    in_regime = get_latest_global_regime(market='IN')
    us_regime = get_latest_global_regime(market='US')
    etf_data = get_latest_global_etf_momentum()
    benchmarks = fetch_benchmark_returns()

    # 2. CANSLIM Exposure Dial
    render_canslim_exposure_dial(in_regime, us_regime)

    # 3. Inter-Market Macro Barometers (Tripwires)
    baro_dict = fetch_macro_barometers()
    render_macro_barometers(baro_dict)
    
    render_html("<hr style='border: 0; height: 1px; background: rgba(255,255,255,0.08); margin: 30px 0;'>")

    # 4. Side-by-Side Market Telemetry (Nifty 500 vs S&P 500)
    col_in, col_us = st.columns(2)
    with col_in:
        render_market_column('IN', 'India (Nifty 500)')
    with col_us:
        render_market_column('US', 'United States (S&P 500)')

    render_html("<hr style='border: 0; height: 1px; background: rgba(255,255,255,0.08); margin: 30px 0;'>")

    # 5. Asset Rotation Radar ("Where is the Momentum?")
    render_asset_rotation_radar(etf_data, benchmarks)


if __name__ == "__main__":
    main()
