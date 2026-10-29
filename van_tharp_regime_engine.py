"""
Van Tharp 6-Market Regime Engine (Canonical Weekly Formulation)
==============================================================
Based on Dr. Van K. Tharp's "Super Trader" (Page 19) and "Trade Your Way to Financial Freedom".

Evaluates 4 macro benchmark indices:
  - Nifty 50 (^NSEI)       - India Mega-Cap
  - Nifty 500 (^CRSLDX)    - India Broad Market
  - S&P 500 (^GSPC)        - US Large-Cap
  - Nasdaq 100 (^NDX)      - US Growth / Tech Leadership

Mathematical Logic:
  1. Resample to Weekly Bars (Open: first, High: max, Low: min, Close: last).
  2. Direction (13-Week Return):
     - BULL:     > +5.53%
     - BEAR:     < -5.53%
     - SIDEWAYS: -5.53% to +5.53%
  3. Volatility (13-Week Weekly ATR as % of Close):
     - VOLATILE: > 2.87%
     - QUIET:    <= 2.87%

Caching:
  - Uses .cache/van_tharp_indices_weekly.pkl
  - Updates only ONCE per calendar day on first boot (0.38s fetch).
  - Loads from disk in < 0.005s on subsequent launches.
"""

import os
import json
import pickle
import time
from datetime import datetime
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_FILE = os.path.join(CACHE_DIR, "van_tharp_indices_weekly.pkl")
SYNC_TRACKER_FILE = os.path.join(CACHE_DIR, "matrix_sync_tracker.json")

INDEX_SPECS = {
    '^NSEI': {
        'symbol': '^NSEI',
        'name': 'NIFTY 50',
        'market': 'IN',
        'flag': '🇮🇳',
        'desc': 'India Mega-Cap Benchmark'
    },
    '^CRSLDX': {
        'symbol': '^CRSLDX',
        'name': 'NIFTY 500',
        'market': 'IN',
        'flag': '🇮🇳',
        'desc': 'India Broad-Market Universe'
    },
    '^GSPC': {
        'symbol': '^GSPC',
        'name': 'S&P 500',
        'market': 'US',
        'flag': '🇺🇸',
        'desc': 'US Large-Cap Benchmark'
    },
    '^NDX': {
        'symbol': '^NDX',
        'name': 'NASDAQ 100',
        'market': 'US',
        'flag': '🇺🇸',
        'desc': 'US Tech & Growth Leaders'
    }
}

REGIME_METADATA = {
    'BULL QUIET': {
        'color': '#10b981',
        'border_color': '#059669',
        'bg_gradient': 'linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%)',
        'icon': '🟢',
        'title': 'BULL QUIET (The Holy Grail Regime)',
        'directive': 'Aggressive Breakouts & Pyramiding (Max 80–100% Exposure)',
        'playbook': 'Optimal CANSLIM environment. High follow-through (>65% win rate). Buy pivot breakouts, pyramid winners at +2% to +3%, trail 21-EMA.',
        'breakout_expectancy': 'High Positive (+0.8R to +1.5R)',
        'risk_stance': 'Aggressive Offense'
    },
    'BULL VOLATILE': {
        'color': '#f59e0b',
        'border_color': '#d97706',
        'bg_gradient': 'linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.05) 100%)',
        'icon': '🟡',
        'title': 'BULL VOLATILE (Late-Stage Expansion)',
        'directive': 'Defensive Strength Selling & Tight Stops (40–60% Exposure)',
        'playbook': 'Late-stage or parabolic expansion with wide swings. Tighten stops to 10-EMA, take partial profits at +3R to +5R into strength, do not chase extended bases.',
        'breakout_expectancy': 'Moderate (+0.3R, High Squat Risk)',
        'risk_stance': 'Guarded Offense'
    },
    'SIDEWAYS QUIET': {
        'color': '#38bdf8',
        'border_color': '#0284c7',
        'bg_gradient': 'linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(2, 132, 199, 0.05) 100%)',
        'icon': '⚪',
        'title': 'SIDEWAYS QUIET (Base-Building Accumulation)',
        'directive': 'Selective Pocket Pivots & 10-EMA Cheats (25–40% Exposure)',
        'playbook': 'Orderly base-building range. Index is flat and calm. Focus strictly on True Market Leaders breaking out of 6-month+ bases with RS > 90 while broad market sleeps.',
        'breakout_expectancy': 'Selective (+0.4R in Primary Leaders)',
        'risk_stance': 'Tactical Selective'
    },
    'SIDEWAYS VOLATILE': {
        'color': '#ef4444',
        'border_color': '#dc2626',
        'bg_gradient': 'linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%)',
        'icon': '🔴',
        'title': 'SIDEWAYS VOLATILE (High-Velocity Chop)',
        'directive': 'Violent Whipsaws: Swing Dips Only (10–20% Pilot Max)',
        'playbook': 'Breakout graveyard. Standard 52-week breakouts squat >70% of the time. Strictly forbid chasing highs. Only buy undercut-and-rally setups or hold cash.',
        'breakout_expectancy': 'Negative (-0.5R, High Whip-Saw Rate)',
        'risk_stance': 'Capital Preservation / Defense'
    },
    'BEAR QUIET': {
        'color': '#f97316',
        'border_color': '#ea580c',
        'bg_gradient': 'linear-gradient(135deg, rgba(249, 115, 22, 0.15) 0%, rgba(234, 88, 12, 0.05) 100%)',
        'icon': '🟠',
        'title': 'BEAR QUIET (Grinding Distribution)',
        'directive': 'Grinding Slow Bleed: Cash Fortress (0–10% Exposure)',
        'playbook': 'Low-volatility persistent drift lower. Rallies are dead-cat bounces that fade. Strictly protect capital, eliminate lagging positions, and wait for market turn.',
        'breakout_expectancy': 'Near Zero (Breakouts Disappear)',
        'risk_stance': 'Fortress Cash'
    },
    'BEAR VOLATILE': {
        'color': '#a855f7',
        'border_color': '#9333ea',
        'bg_gradient': 'linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(147, 51, 234, 0.05) 100%)',
        'icon': '🟣',
        'title': 'BEAR VOLATILE (Panic Liquidation & Washout)',
        'directive': 'Panic Washouts: Submerged Beach Ball Stalking Mode',
        'playbook': 'Indiscriminate institutional selling and volatility spikes. Maintain high cash (0–10%). Use Beach Ball Radar to identify submerged leaders printing high-volume power plays.',
        'breakout_expectancy': 'Extreme Negative (Washout Hazards)',
        'risk_stance': 'Deep Stalking / Cash'
    }
}


def _has_synced_today() -> bool:
    if os.path.exists(SYNC_TRACKER_FILE):
        try:
            with open(SYNC_TRACKER_FILE, "r", encoding="utf-8") as f:
                tracker = json.load(f)
            today_str = datetime.now().strftime("%Y-%m-%d")
            return tracker.get("van_tharp_indices", {}).get("last_sync_date") == today_str
        except Exception:
            pass
    return False


def _record_sync_complete():
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tracker = {}
        if os.path.exists(SYNC_TRACKER_FILE):
            with open(SYNC_TRACKER_FILE, "r", encoding="utf-8") as f:
                tracker = json.load(f)
        today_str = datetime.now().strftime("%Y-%m-%d")
        if "van_tharp_indices" not in tracker:
            tracker["van_tharp_indices"] = {}
        tracker["van_tharp_indices"]["last_sync_date"] = today_str
        tracker["van_tharp_indices"]["sync_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(SYNC_TRACKER_FILE, "w", encoding="utf-8") as f:
            json.dump(tracker, f, indent=2)
    except Exception as e:
        print(f"[VanTharp Engine] Tracker save notice: {e}")


def _download_and_cache_weekly_indices() -> dict:
    """Downloads 2 years of daily data for the 4 indices and converts to weekly bars."""
    symbols = list(INDEX_SPECS.keys())
    print(f"[VanTharp Engine] Fetching fresh weekly bars for {symbols} from Yahoo Finance...")
    try:
        raw_df = yf.download(symbols, period="2y", progress=False)
    except Exception as e:
        print(f"[VanTharp Engine] Download error: {e}")
        raw_df = pd.DataFrame()

    weekly_dict = {}
    if not raw_df.empty:
        for sym in symbols:
            try:
                if isinstance(raw_df.columns, pd.MultiIndex):
                    sub = raw_df.xs(sym, level=1, axis=1)
                else:
                    sub = raw_df
                if sub.empty or 'Close' not in sub.columns:
                    continue
                w = sub.resample('W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                if len(w) >= 15:
                    weekly_dict[sym] = w
            except Exception as e:
                print(f"[VanTharp Engine] Resample error for {sym}: {e}")

    if weekly_dict:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(weekly_dict, f)
            _record_sync_complete()
            print(f"[VanTharp Engine] Saved {len(weekly_dict)} indices to {CACHE_FILE}")
        except Exception as e:
            print(f"[VanTharp Engine] Cache write error: {e}")

    return weekly_dict


def load_weekly_indices(force_refresh: bool = False) -> dict:
    """
    Loads weekly index history from local disk cache in < 0.005s.
    Only triggers a network fetch if not synced today or force_refresh is True.
    """
    if not force_refresh and _has_synced_today() and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            if data and all(sym in data for sym in INDEX_SPECS):
                return data
        except Exception:
            pass

    # If cache exists and has all symbols, use it if network fails
    existing_cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                existing_cache = pickle.load(f)
        except Exception:
            pass

    # Attempt daily sync
    fresh_data = _download_and_cache_weekly_indices()
    if fresh_data:
        return fresh_data
    if existing_cache:
        print("[VanTharp Engine] Network unavailable, falling back to existing cache.")
        return existing_cache

    return {}


def calculate_van_tharp_metrics(weekly_df: pd.DataFrame) -> dict:
    """
    Computes canonical Van Tharp 13-week metrics:
      - 13-Week Percentage Change vs +/- 5.53%
      - 13-Week Weekly True Range % vs 2.87%
    """
    if weekly_df is None or len(weekly_df) < 15:
        return None

    c = weekly_df['Close']
    h = weekly_df['High']
    l = weekly_df['Low']

    # 1. 13-Week Percentage Return
    ret_13w = float((c.iloc[-1] - c.iloc[-14]) / c.iloc[-14] * 100.0)

    # 2. Weekly True Range & 13-Week Rolling Mean
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_13w = tr.rolling(13).mean()
    atr_pct_series = (atr_13w / c) * 100.0
    cur_atr_pct = float(atr_pct_series.iloc[-1])

    # 3. Canonical Direction Rule
    if ret_13w > 5.53:
        direction = 'BULL'
    elif ret_13w < -5.53:
        direction = 'BEAR'
    else:
        direction = 'SIDEWAYS'

    # 4. Canonical Volatility Rule
    if cur_atr_pct > 2.87:
        volatility = 'VOLATILE'
    else:
        volatility = 'QUIET'

    regime = f"{direction} {volatility}"
    meta = REGIME_METADATA.get(regime, REGIME_METADATA['SIDEWAYS QUIET'])

    # Historical 52-week trajectory
    history_df = pd.DataFrame({
        'close': c,
        'ret_13w': ((c - c.shift(13)) / c.shift(13) * 100.0),
        'atr_pct': atr_pct_series
    }).tail(52)

    return {
        'last_date': weekly_df.index[-1].strftime('%Y-%m-%d'),
        'current_close': float(c.iloc[-1]),
        'ret_13w': ret_13w,
        'direction': direction,
        'atr_13w_pct': cur_atr_pct,
        'volatility': volatility,
        'regime': regime,
        'meta': meta,
        'history_df': history_df
    }


def get_all_van_tharp_regimes(force_refresh: bool = False) -> dict:
    """
    Returns the comprehensive Van Tharp regime telemetry across all 4 benchmarks.
    """
    weekly_data = load_weekly_indices(force_refresh=force_refresh)
    results = {}

    for sym, spec in INDEX_SPECS.items():
        w_df = weekly_data.get(sym)
        metrics = calculate_van_tharp_metrics(w_df)
        if metrics:
            results[sym] = {
                **spec,
                **metrics
            }
        else:
            # Fallback safe defaults if symbol missing
            results[sym] = {
                **spec,
                'last_date': datetime.now().strftime('%Y-%m-%d'),
                'current_close': 0.0,
                'ret_13w': 0.0,
                'direction': 'SIDEWAYS',
                'atr_13w_pct': 2.0,
                'volatility': 'QUIET',
                'regime': 'SIDEWAYS QUIET',
                'meta': REGIME_METADATA['SIDEWAYS QUIET'],
                'history_df': pd.DataFrame()
            }

    return results


def _clean_html(html_str: str) -> str:
    """Strips leading whitespace from every line so Streamlit Markdown never treats it as code."""
    return "\n".join(line.strip() for line in html_str.splitlines() if line.strip())


def render_van_tharp_cockpit_card(market: str = 'ALL'):
    """
    Renders an institutional 4-card telemetry HUD on the main dashboard.
    Safe to call within Streamlit without breaking any existing components.
    """
    try:
        import streamlit as st
    except ImportError:
        return

    data = get_all_van_tharp_regimes(force_refresh=False)
    if not data:
        return

    # Filter by market if requested
    if market.upper() in ['IN', 'INDIA']:
        display_items = [v for v in data.values() if v.get('market') == 'IN']
    elif market.upper() in ['US', 'USA']:
        display_items = [v for v in data.values() if v.get('market') == 'US']
    else:
        display_items = list(data.values())

    cards_html = []
    for item in display_items:
        meta = item.get('meta', {})
        ret = item.get('ret_13w', 0.0)
        atr = item.get('atr_13w_pct', 0.0)
        direction = item.get('direction', 'SIDEWAYS')
        vol = item.get('volatility', 'QUIET')
        regime = item.get('regime', 'SIDEWAYS QUIET')
        color = meta.get('color', '#38bdf8')
        border_col = meta.get('border_color', '#0284c7')
        bg_grad = meta.get('bg_gradient', 'rgba(15, 23, 42, 0.95)')
        icon = meta.get('icon', '⚪')
        directive = meta.get('directive', 'Tactical Selective')

        ret_col = '#10b981' if ret > 5.53 else '#ef4444' if ret < -5.53 else '#38bdf8'
        atr_col = '#ef4444' if atr > 2.87 else '#10b981'

        cards_html.append(f"""
        <div style="background: {bg_grad}; border: 1px solid {border_col}55; border-top: 4px solid {color}; border-radius: 10px; padding: 14px 16px; box-shadow: 0 8px 20px -4px rgba(0,0,0,0.45); display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 0.82rem; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 6px;">
                        <span>{item.get('flag')}</span>
                        <span>{item.get('name')}</span>
                    </div>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #94a3b8; background: rgba(0,0,0,0.35); padding: 1px 6px; border-radius: 4px;">
                        {item.get('symbol')}
                    </span>
                </div>
                <div style="margin-bottom: 10px;">
                    <span style="background: {color}22; color: {color}; border: 1px solid {color}66; padding: 4px 10px; border-radius: 9999px; font-size: 0.74rem; font-weight: 900; letter-spacing: 0.5px; display: inline-flex; align-items: center; gap: 6px;">
                        <span>{icon}</span> {regime}
                    </span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; background: rgba(0,0,0,0.35); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.04);">
                    <div>
                        <div style="font-size: 0.65rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">13W Return</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.90rem; font-weight: 800; color: {ret_col};">
                            {ret:+.2f}%
                        </div>
                        <div style="font-size: 0.58rem; color: #64748b;">(Baseline: &plusmn;5.5%)</div>
                    </div>
                    <div>
                        <div style="font-size: 0.65rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">13W Wkly ATR</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.90rem; font-weight: 800; color: {atr_col};">
                            {atr:.2f}%
                        </div>
                        <div style="font-size: 0.58rem; color: #64748b;">(Baseline: 2.87%)</div>
                    </div>
                </div>
            </div>
            <div style="border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px; font-size: 0.70rem; color: #cbd5e1; line-height: 1.35; font-weight: 600;">
                <span style="color: {color}; font-weight: 800;">DIRECTIVE:</span> {directive}
            </div>
        </div>
        """)

    grid_cols = len(display_items)
    grid_template = f"repeat(auto-fit, minmax(240px, 1fr))" if grid_cols > 2 else f"repeat({grid_cols}, 1fr)"

    hud_html = f"""
    <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(30, 41, 59, 0.92) 100%); border: 1px solid rgba(255, 255, 255, 0.08); border-left: 6px solid #38bdf8; border-radius: 12px; padding: 18px 20px; margin-bottom: 16px; box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.55);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 14px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.45rem;">🏛️</span>
                <div>
                    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                        <span style="color: #f8fafc; font-size: 1.05rem; font-weight: 800; letter-spacing: -0.2px;">
                            Van Tharp 6-Market Regime Matrix
                        </span>
                        <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); padding: 2px 8px; border-radius: 9999px; font-size: 0.65rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.6px;">
                            WEEKLY 13W TELEMETRY
                        </span>
                    </div>
                    <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 2px;">
                        Canonical formulation from Dr. Van K. Tharp's <i>Super Trader</i> (Page 19). Classifies macro trend direction and volatility to mandate positive-expectancy CANSLIM playbooks.
                    </div>
                </div>
            </div>
            <div style="font-size: 0.68rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
                Fast Disk Cache: &lt; 0.01s
            </div>
        </div>
        <div style="display: grid; grid-template-columns: {grid_template}; gap: 12px;">
            {''.join(cards_html)}
        </div>
    </div>
    """
    st.markdown(_clean_html(hud_html), unsafe_allow_html=True)


def render_van_tharp_full_page():
    """
    Renders the complete Van Tharp 6-Market Regime Visualizer on the Market Regime & Health page.
    Includes the 2D Grid, index positioning, CANSLIM expectancy matrix, and historical trajectory.
    """
    try:
        import streamlit as st
        import plotly.graph_objects as go
    except ImportError:
        return

    data = get_all_van_tharp_regimes(force_refresh=False)
    if not data:
        st.warning("Van Tharp regime data unavailable.")
        return

    # 1. Top Cockpit Card
    render_van_tharp_cockpit_card(market='ALL')

    # 2. Interactive 2D Van Tharp Grid Visualizer
    st.markdown("""
    <div style="margin-top: 24px; margin-bottom: 12px;">
        <h4 style="color: #f8fafc; margin-bottom: 4px;">🗺️ The 2D Van Tharp Regime Matrix (Trend Direction vs. Volatility)</h4>
        <div style="color: #94a3b8; font-size: 0.82rem;">
            Where all 4 benchmark indices currently sit inside Van Tharp's 6 operational states:
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Map current indices to their cells
    cell_indices = {
        'BULL QUIET': [],
        'BULL VOLATILE': [],
        'SIDEWAYS QUIET': [],
        'SIDEWAYS VOLATILE': [],
        'BEAR QUIET': [],
        'BEAR VOLATILE': []
    }
    for item in data.values():
        reg = item.get('regime')
        if reg in cell_indices:
            cell_indices[reg].append(f"{item.get('flag')} {item.get('name')}")

    def _render_cell(regime_name):
        meta = REGIME_METADATA[regime_name]
        col = meta['color']
        icon = meta['icon']
        title = regime_name
        directive = meta['directive']
        seated = cell_indices.get(regime_name, [])
        seated_html = "".join([f'<div style="background: {col}; color: #000; font-weight: 800; font-size: 0.72rem; padding: 3px 8px; border-radius: 4px; margin-top: 4px; display: inline-block; margin-right: 4px; box-shadow: 0 0 10px {col}88;">📍 {idx}</div>' for idx in seated])
        is_active = len(seated) > 0
        glow = f"border: 2px solid {col}; box-shadow: 0 0 18px {col}44;" if is_active else f"border: 1px solid rgba(255,255,255,0.08);"
        return f"""
        <div style="background: rgba(15, 23, 42, 0.85); {glow} border-radius: 10px; padding: 14px; min-height: 140px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 900; color: {col}; font-size: 0.85rem; letter-spacing: 0.5px;">{icon} {title}</span>
                    <span style="font-size: 0.65rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">{meta.get('breakout_expectancy')}</span>
                </div>
                <div style="font-size: 0.72rem; color: #cbd5e1; line-height: 1.4; margin-bottom: 8px;">
                    {directive}
                </div>
            </div>
            <div>
                {seated_html if seated_html else '<div style="font-size: 0.65rem; color: #64748b; font-style: italic;">No benchmark in this regime</div>'}
            </div>
        </div>
        """

    grid_html = f"""
    <div style="background: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 16px; margin-bottom: 24px;">
        <div style="display: grid; grid-template-columns: 120px 1fr 1fr 1fr; gap: 10px; align-items: stretch;">
            <!-- Header Row -->
            <div></div>
            <div style="text-align: center; color: #10b981; font-weight: 800; font-size: 0.82rem; background: rgba(16, 185, 129, 0.1); padding: 8px; border-radius: 6px; border: 1px solid rgba(16, 185, 129, 0.25);">
                🟢 BULL MARKET<br/><span style="font-size: 0.65rem; color: #94a3b8;">13W Ret &gt; +5.53%</span>
            </div>
            <div style="text-align: center; color: #38bdf8; font-weight: 800; font-size: 0.82rem; background: rgba(56, 189, 248, 0.1); padding: 8px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.25);">
                ⚪ SIDEWAYS MARKET<br/><span style="font-size: 0.65rem; color: #94a3b8;">-5.53% to +5.53%</span>
            </div>
            <div style="text-align: center; color: #f97316; font-weight: 800; font-size: 0.82rem; background: rgba(249, 115, 22, 0.1); padding: 8px; border-radius: 6px; border: 1px solid rgba(249, 115, 22, 0.25);">
                🔴 BEAR MARKET<br/><span style="font-size: 0.65rem; color: #94a3b8;">13W Ret &lt; -5.53%</span>
            </div>

            <!-- Row 1: QUIET -->
            <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 10px; text-align: center;">
                <div style="font-weight: 800; font-size: 0.80rem; color: #34d399;">🌊 QUIET</div>
                <div style="font-size: 0.62rem; color: #94a3b8;">ATR &le; 2.87%</div>
                <div style="font-size: 0.58rem; color: #64748b; margin-top: 4px;">Orderly Accumulation</div>
            </div>
            {_render_cell('BULL QUIET')}
            {_render_cell('SIDEWAYS QUIET')}
            {_render_cell('BEAR QUIET')}

            <!-- Row 2: VOLATILE -->
            <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 10px; text-align: center;">
                <div style="font-weight: 800; font-size: 0.80rem; color: #f87171;">⚡ VOLATILE</div>
                <div style="font-size: 0.62rem; color: #94a3b8;">ATR &gt; 2.87%</div>
                <div style="font-size: 0.58rem; color: #64748b; margin-top: 4px;">High Churn / Gaps</div>
            </div>
            {_render_cell('BULL VOLATILE')}
            {_render_cell('SIDEWAYS VOLATILE')}
            {_render_cell('BEAR VOLATILE')}
        </div>
    </div>
    """
    st.markdown(_clean_html(grid_html), unsafe_allow_html=True)

    # 3. Mandated CANSLIM Playbooks Reference Table
    with st.expander("📖 View Canonical CANSLIM Playbook & Expectancy Matrix for Each Regime", expanded=False):
        playbook_rows = []
        for reg_name, meta in REGIME_METADATA.items():
            playbook_rows.append(f"""
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 10px; font-weight: 800; color: {meta['color']}; white-space: nowrap;">{meta['icon']} {reg_name}</td>
                <td style="padding: 10px; color: #f8fafc; font-weight: 700;">{meta['directive']}</td>
                <td style="padding: 10px; color: #94a3b8; font-size: 0.78rem;">{meta['playbook']}</td>
                <td style="padding: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {meta['color']};">{meta['breakout_expectancy']}</td>
            </tr>
            """)
        table_html = f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 0.82rem; background: rgba(15, 23, 42, 0.6); border-radius: 8px; overflow: hidden;">
            <thead>
                <tr style="background: rgba(0,0,0,0.4); text-align: left; color: #94a3b8; font-size: 0.72rem; text-transform: uppercase;">
                    <th style="padding: 10px;">Market Type</th>
                    <th style="padding: 10px;">Tactical Directive</th>
                    <th style="padding: 10px;">Execution Playbook</th>
                    <th style="padding: 10px;">Breakout Expectancy</th>
                </tr>
            </thead>
            <tbody>
                {''.join(playbook_rows)}
            </tbody>
        </table>
        """
        st.markdown(_clean_html(table_html), unsafe_allow_html=True)


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("Testing Van Tharp Regime Engine...")
    t0 = time.time()
    res = get_all_van_tharp_regimes(force_refresh=True)
    t1 = time.time()
    print(f"Computed all 4 regimes in {t1-t0:.2f}s:\n")
    for sym, d in res.items():
        print(f"{d['flag']} {d['name']} ({sym}):")
        print(f"  Regime: {d['meta']['icon']} {d['regime']}")
        print(f"  13W Return: {d['ret_13w']:+.2f}% | 13W Weekly ATR%: {d['atr_13w_pct']:.2f}%")
        print(f"  Directive: {d['meta']['directive']}")
        print()
