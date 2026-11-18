"""
Relative Rotation Graph (RRG) & Sector Velocity Engine
======================================================
Theoretical Framework: Julius de Kempenaer (CMT, RRG Creator) & Himanshu Sharma (CMT)
Implementation: Kush Tracker Quantitative Terminal (Hedge-Fund Grade Cockpit)

Quantifies 2-Dimensional Sector Relative Strength:
  - X-Axis (JdK RS-Ratio): Double-smoothed trend outperformance vs Nifty 500 (^CRSLDX) centered at 100.0.
  - Y-Axis (JdK RS-Momentum): Rate of change / acceleration of RS-Ratio centered at 100.0.

Provides rotational lifecycle analysis across 4 quadrants:
  - Leading   (X >= 100, Y >= 100): High relative trend + accelerating momentum.
  - Weakening (X >= 100, Y < 100) : High relative trend, but decelerating velocity (early distribution).
  - Lagging   (X < 100,  Y < 100) : Underperforming trend + decelerating downward (avoid capital).
  - Improving (X < 100,  Y >= 100): Turnaround alpha sweet spot (smart money accumulation before 1-yr RS reaches 80).
"""

import os
import sys
import time
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# HELPER: HTML CLEANER (PREVENTS MARKDOWN TREATING INDENTED HTML AS CODE)
# =============================================================================

def clean_html(html_str: str) -> str:
    """Strips leading whitespace from every line so Markdown never treats indented HTML as a code block."""
    return "\n".join(line.strip() for line in html_str.splitlines() if line.strip())


# =============================================================================
# UNIVERSE DEFINITIONS (INDIAN SECTORAL INDICES & THEMATIC ETFS)
# =============================================================================

DEFAULT_BENCHMARK = "^CRSLDX"  # Nifty 500 Broad Market Index
ALT_BENCHMARK = "^NSEI"        # Nifty 50 Large Cap Index

SECTOR_UNIVERSE = {
    "Nifty Auto": "^CNXAUTO",
    "Nifty IT": "^CNXIT",
    "Nifty Realty": "^CNXREALTY",
    "Nifty Metal": "^CNXMETAL",
    "Nifty FMCG": "^CNXFMCG",
    "Nifty Energy": "^CNXENERGY",
    "Nifty Infrastructure": "^CNXINFRA",
    "Nifty Pharma": "^CNXPHARMA",
    "Nifty Bank": "^NSEBANK",
    "Nifty Financial Services": "NIFTY_FIN_SERVICE.NS",
    "Nifty PSE": "^CNXPSE",
    "Nifty Defence (ETF)": "MODEFENCE.NS",
    "CPSE Central PSE (ETF)": "CPSEETF.NS",
    "India Manufacturing (ETF)": "MAKEINDIA.NS",
    "Oil & Gas (ETF)": "OILIETF.NS",
    "Nifty Consumption (ETF)": "CONSUMBEES.NS",
    "Momentum 30 (ETF)": "MOM30IETF.NS",
    "Healthcare (ETF)": "HEALTHY.NS"
}

CORE_10_SECTORS = [
    "Nifty Auto", "Nifty IT", "Nifty Realty", "Nifty Metal", "Nifty FMCG",
    "Nifty Energy", "Nifty Infrastructure", "Nifty Pharma", "Nifty Bank",
    "Nifty Financial Services", "Nifty PSE"
]

THEMATIC_ETFS = [
    "Nifty Defence (ETF)", "CPSE Central PSE (ETF)", "India Manufacturing (ETF)",
    "Oil & Gas (ETF)", "Nifty Consumption (ETF)", "Momentum 30 (ETF)", "Healthcare (ETF)"
]

SECTOR_CONSTITUENTS_MAP = {
    "Nifty Auto": ["M&M", "TATAMOTORS", "BAJAJ-AUTO", "MARUTI", "TVSMOTOR"],
    "Nifty IT": ["TCS", "INFY", "HCLTECH", "PERSISTENT", "COFORGE"],
    "Nifty Realty": ["DLF", "GODREJPROP", "LODHA", "OBEROIRLTY", "BRIGADE"],
    "Nifty Metal": ["TATASTEEL", "JINDALSTEL", "HINDALCO", "VEDL", "NMDC"],
    "Nifty FMCG": ["ITC", "HINDUNILVR", "NESTLEIND", "TATACONSUM", "VBL"],
    "Nifty Energy": ["RELIANCE", "NTPC", "POWERGRID", "ONGC", "BPCL"],
    "Nifty Infrastructure": ["LT", "ADANIENT", "BHARTIARTL", "ULTRACEMCO", "GRASIM"],
    "Nifty Pharma": ["SUNPHARMA", "CIPLA", "DIVISLAB", "DRREDDY", "LUPIN"],
    "Nifty Bank": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"],
    "Nifty Financial Services": ["BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "MUTHOOTFIN"],
    "Nifty PSE": ["BEL", "HAL", "COALINDIA", "NTPC", "PFC"],
    "Nifty Defence (ETF)": ["HAL", "BEL", "MAZDOCK", "COCHINSHIP", "BDL"],
    "CPSE Central PSE (ETF)": ["COALINDIA", "NTPC", "ONGC", "BEL", "OIL"],
    "India Manufacturing (ETF)": ["SIEMENS", "ABB", "BHEL", "CUMMINSIND", "DIXON"],
    "Oil & Gas (ETF)": ["ONGC", "IOC", "BPCL", "RELIANCE", "GAIL"],
    "Nifty Consumption (ETF)": ["TITAN", "ITC", "ASIANPAINT", "TRENT", "TATACONSUM"],
    "Momentum 30 (ETF)": ["TRENT", "BHARTIARTL", "BAJAJ-AUTO", "BEL", "HAL"],
    "Healthcare (ETF)": ["APOLLOHOSP", "FORTIS", "MAXHEALTH", "SUNPHARMA", "LUPIN"]
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_FILE = os.path.join(CACHE_DIR, "rrg_sector_prices.parquet")
CACHE_PKL_FILE = os.path.join(CACHE_DIR, "rrg_sector_prices.pkl")


# =============================================================================
# DATA FETCHING & DUAL-TIER DISK CACHING
# =============================================================================

def get_disk_cached_prices(max_age_hours: int = 4) -> pd.DataFrame:
    """Loads historical prices from disk cache if fresh, otherwise returns empty DataFrame."""
    try:
        if os.path.exists(CACHE_FILE):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
            if datetime.now() - file_mtime < timedelta(hours=max_age_hours):
                df = pd.read_parquet(CACHE_FILE)
                if not df.empty and df.shape[0] > 60:
                    return df
    except Exception:
        pass
    try:
        if os.path.exists(CACHE_PKL_FILE):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_PKL_FILE))
            if datetime.now() - file_mtime < timedelta(hours=max_age_hours):
                df = pd.read_pickle(CACHE_PKL_FILE)
                if not df.empty and df.shape[0] > 60:
                    return df
    except Exception:
        pass
    return pd.DataFrame()


def save_disk_cached_prices(df: pd.DataFrame) -> None:
    """Saves downloaded historical prices to disk cache."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(CACHE_FILE)
        return
    except Exception:
        pass
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_pickle(CACHE_PKL_FILE)
    except Exception:
        pass


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rrg_historical_prices(benchmark: str = DEFAULT_BENCHMARK, period: str = "2y") -> pd.DataFrame:
    """
    Fetches daily closing prices for all sector indices/ETFs and the benchmark.
    Uses disk caching to guarantee sub-second page loads.
    """
    cached_df = get_disk_cached_prices(max_age_hours=4)
    all_symbols = list(SECTOR_UNIVERSE.values()) + [benchmark, ALT_BENCHMARK]
    
    if not cached_df.empty:
        present_cols = [c for c in all_symbols if c in cached_df.columns]
        if benchmark in cached_df.columns and len(present_cols) >= len(all_symbols) - 2:
            return cached_df

    try:
        raw = yf.download(all_symbols, period=period, interval="1d", progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.levels[0]:
                close_df = raw["Close"].copy()
            else:
                close_df = raw.xs("Close", axis=1, level=0, drop_level=True)
        else:
            close_df = raw.copy()

        close_df = close_df.ffill(limit=3).dropna(how="all")
        if not close_df.empty and close_df.shape[0] > 60:
            save_disk_cached_prices(close_df)
            return close_df
    except Exception as e:
        if not cached_df.empty:
            return cached_df
        print(f"RRG Data Fetch Error: {e}")

    return pd.DataFrame()


# =============================================================================
# MATHEMATICAL RRG ENGINE (JULIUS DE KEMPENAER / HIMANSHU SHARMA)
# =============================================================================

def calc_wma(series: pd.Series, window: int = 10) -> pd.Series:
    """Computes Weighted Moving Average (WMA) with linear weights 1 to window."""
    weights = np.arange(1, window + 1)
    weight_sum = weights.sum()
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weight_sum, raw=True)


def compute_rrg_vectors(
    sector_series: pd.Series,
    benchmark_series: pd.Series,
    window: int = 10,
    momentum_window: int = 10,
    norm_window: int = 20
) -> pd.DataFrame:
    """
    Computes JdK RS-Ratio and JdK RS-Momentum series for a single sector vs benchmark.
    """
    common_idx = sector_series.dropna().index.intersection(benchmark_series.dropna().index)
    if len(common_idx) < (window * 2 + norm_window):
        return pd.DataFrame()

    sec = sector_series.loc[common_idx]
    bench = benchmark_series.loc[common_idx]

    # Step 1: Raw RS
    rs = (sec / bench) * 100.0

    # Step 2: JdK RS-Ratio (X-Axis)
    rs_smooth = calc_wma(rs, window)
    rs_base = calc_wma(rs_smooth, window)
    std_smooth = rs_smooth.rolling(window).std().replace(0, np.nan).bfill()
    rs_ratio = 100.0 + ((rs_smooth - rs_base) / std_smooth) * 1.5

    # Step 3: JdK RS-Momentum (Y-Axis)
    roc = ((rs_ratio - rs_ratio.shift(momentum_window)) / rs_ratio.shift(momentum_window)) * 100.0
    roc_mean = roc.rolling(norm_window).mean()
    roc_std = roc.rolling(norm_window).std().replace(0, np.nan).bfill()
    rs_momentum = 100.0 + ((roc - roc_mean) / roc_std) * 2.0

    res = pd.DataFrame({
        "RS": rs,
        "RS_Ratio": rs_ratio,
        "RS_Momentum": rs_momentum
    }, index=common_idx).dropna()

    return res


def classify_quadrant(x: float, y: float) -> str:
    """Classifies (X, Y) coordinate pair into Julius de Kempenaer's 4 quadrants."""
    if x >= 100.0 and y >= 100.0:
        return "Leading"
    elif x >= 100.0 and y < 100.0:
        return "Weakening"
    elif x < 100.0 and y < 100.0:
        return "Lagging"
    else:
        return "Improving"


def calculate_heading_details(curr_x: float, curr_y: float, prev_x: float, prev_y: float) -> tuple:
    """
    Computes heading angle in degrees (0° to 360°) and categorical trajectory dynamics.
    """
    dx = curr_x - prev_x
    dy = curr_y - prev_y
    velocity = math.sqrt(dx**2 + dy**2)
    
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360.0

    if 0 <= angle_deg < 90:
        heading_desc = "North-East (Expanding Outperformance)"
        heading_short = "↗️ Accelerating NE"
        heading_code = "NE"
        color = "#10b981"
    elif 90 <= angle_deg < 180:
        heading_desc = "North-West (Bottom Reversal / Accumulation)"
        heading_short = "↖️ Bottoming NW"
        heading_code = "NW"
        color = "#38bdf8"
    elif 180 <= angle_deg < 270:
        heading_desc = "South-West (Decelerating Underperformance)"
        heading_short = "↙️ Distributing SW"
        heading_code = "SW"
        color = "#ef4444"
    else:
        heading_desc = "South-East (Topping / Profit-Taking)"
        heading_short = "↘️ Topping SE"
        heading_code = "SE"
        color = "#f59e0b"

    return angle_deg, heading_desc, heading_short, heading_code, velocity, color


def get_canslim_tactical_playbook(quadrant: str, heading_code: str) -> str:
    """Returns institutional action recommendation based on quadrant and heading."""
    if quadrant == "Leading":
        if heading_code in ["NE", "NW"]:
            return "🔥 Core Institutional Driver. Ride secular trend; use 10/20 EMA pullbacks for add-on entries."
        else:
            return "⚠️ Mature Leader. Watch for exhaustion gap-ups; trail stops tight to 21 EMA."
    elif quadrant == "Improving":
        if heading_code in ["NE", "NW"]:
            return "🎯 Prime CANSLIM Alpha Sweetspot. Scan for fresh stage-1 base breakouts & VCP pivots."
        else:
            return "🔄 Early Base Building. Watchlist top liquid names; wait for institutional pocket pivot."
    elif quadrant == "Weakening":
        if heading_code in ["SW", "SE"]:
            return "🛑 Distribution Alert. Momentum rolling over; lock in partial gains, do NOT buy new breakouts."
        else:
            return "⏳ Mean-Reversion Testing. Wait for 50-day SMA stabilization before redeploying."
    else:  # Lagging
        if heading_code in ["NW", "NE"]:
            return "👀 Early Stage-1 Bottom Formation. Observe volume accumulation footprints."
        else:
            return "❌ Dead Capital / Institutional Liquidation. Zero portfolio allocation."


# =============================================================================
# MULTI-SECTOR ROTATION SNAPSHOT GENERATION
# =============================================================================

def generate_rrg_snapshot(
    prices_df: pd.DataFrame,
    benchmark_symbol: str = DEFAULT_BENCHMARK,
    timeframe: str = "Weekly",
    trail_length: int = 5,
    window: int = 10,
    momentum_window: int = 10
) -> tuple:
    """
    Computes RRG metrics for all sectors in the universe.
    """
    if prices_df.empty or benchmark_symbol not in prices_df.columns:
        return pd.DataFrame(), {}

    if timeframe == "Weekly":
        data = prices_df.resample("W-FRI").last().dropna(how="all")
    else:
        data = prices_df.copy()

    bench = data[benchmark_symbol]
    snapshot_records = []
    trails_dict = {}

    for sector_name, sym in SECTOR_UNIVERSE.items():
        if sym not in data.columns:
            continue
        sec_prices = data[sym]
        
        vectors = compute_rrg_vectors(
            sector_series=sec_prices,
            benchmark_series=bench,
            window=window,
            momentum_window=momentum_window,
            norm_window=20
        )
        
        if vectors.empty or len(vectors) < trail_length + 2:
            continue

        curr_row = vectors.iloc[-1]
        prev_row = vectors.iloc[-2]
        lookback_bar = vectors.iloc[-trail_length] if len(vectors) >= trail_length else vectors.iloc[0]

        curr_x = float(curr_row["RS_Ratio"])
        curr_y = float(curr_row["RS_Momentum"])
        prev_x = float(prev_row["RS_Ratio"])
        prev_y = float(prev_row["RS_Momentum"])
        
        hist_x = float(lookback_bar["RS_Ratio"])
        hist_y = float(lookback_bar["RS_Momentum"])

        quadrant = classify_quadrant(curr_x, curr_y)
        prev_quad = classify_quadrant(hist_x, hist_y)
        
        angle_deg, heading_desc, heading_short, heading_code, velocity, heading_color = calculate_heading_details(
            curr_x, curr_y, prev_x, prev_y
        )
        
        playbook = get_canslim_tactical_playbook(quadrant, heading_code)

        trail_slice = vectors.tail(trail_length)
        trail_coords = [
            {
                "date": dt.strftime("%Y-%m-%d"),
                "x": float(row["RS_Ratio"]),
                "y": float(row["RS_Momentum"])
            }
            for dt, row in trail_slice.iterrows()
        ]
        trails_dict[sector_name] = {
            "coords": trail_coords,
            "quadrant": quadrant,
            "color": heading_color,
            "symbol": sym,
            "angle": angle_deg,
            "velocity": velocity
        }

        if quadrant != prev_quad:
            transition = f"{prev_quad} ➔ {quadrant}"
        else:
            transition = f"Sustained in {quadrant}"

        snapshot_records.append({
            "Sector": sector_name,
            "Symbol": sym,
            "Quadrant": quadrant,
            "RS_Ratio": curr_x,
            "RS_Momentum": curr_y,
            "Delta_X": curr_x - hist_x,
            "Delta_Y": curr_y - hist_y,
            "Heading_Angle": angle_deg,
            "Heading_Desc": heading_desc,
            "Heading_Short": heading_short,
            "Heading_Code": heading_code,
            "Heading_Color": heading_color,
            "Velocity": velocity,
            "Transition": transition,
            "Playbook": playbook,
            "Top_Stocks": ", ".join(SECTOR_CONSTITUENTS_MAP.get(sector_name, [])[:4])
        })

    snapshot_df = pd.DataFrame(snapshot_records)
    if not snapshot_df.empty:
        quad_order = {"Leading": 1, "Improving": 2, "Weakening": 3, "Lagging": 4}
        snapshot_df["Quad_Rank"] = snapshot_df["Quadrant"].map(quad_order)
        snapshot_df = snapshot_df.sort_values(
            by=["Quad_Rank", "RS_Ratio"],
            ascending=[True, False]
        ).drop(columns=["Quad_Rank"])

    return snapshot_df, trails_dict


# =============================================================================
# HEDGE-FUND GRADE PLOTLY RRG CANVAS BUILDER (WITH SMART DE-COLLISION)
# =============================================================================

def build_world_class_rrg_figure(
    snapshot_df: pd.DataFrame,
    trails_dict: dict,
    trail_length: int = 5,
    selected_sectors: list = None,
    focused_sector: str = None,
    label_mode: str = "Smart (De-Collided)"
) -> go.Figure:
    """
    Constructs a true hedge-fund grade RRG radar canvas:
      - Quadrant zones with high-contrast subtle fills.
      - Concentric velocity radar rings (isolines at 2.5, 5.0, 7.5).
      - Fluid rectangular column fitting without awkward squished black bands.
      - Intelligent proximity de-collision to eliminate center text blobs.
      - Highlighting halo for the currently selected sector.
    """
    fig = go.Figure()

    if snapshot_df.empty:
        fig.add_annotation(
            text="No RRG price data available. Please verify network connection.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94a3b8")
        )
        return fig

    # Calculate symmetrical domain boundaries centered at (100.0, 100.0)
    all_x = snapshot_df["RS_Ratio"].tolist()
    all_y = snapshot_df["RS_Momentum"].tolist()
    
    for sec, t_data in trails_dict.items():
        if selected_sectors and sec not in selected_sectors:
            continue
        for c in t_data["coords"]:
            all_x.append(c["x"])
            all_y.append(c["y"])

    max_dist_x = max(abs(max(all_x) - 100.0), abs(min(all_x) - 100.0), 3.5)
    max_dist_y = max(abs(max(all_y) - 100.0), abs(min(all_y) - 100.0), 3.5)
    
    span = max(max_dist_x, max_dist_y) * 1.15
    bound_min, bound_max = 100.0 - span, 100.0 + span

    # 1. Background Quadrant Zones
    fig.add_shape(type="rect", x0=100.0, y0=100.0, x1=bound_max, y1=bound_max,
                  fillcolor="rgba(16, 185, 129, 0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=100.0, y0=bound_min, x1=bound_max, y1=100.0,
                  fillcolor="rgba(245, 158, 11, 0.07)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=bound_min, y0=bound_min, x1=100.0, y1=100.0,
                  fillcolor="rgba(239, 68, 68, 0.07)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=bound_min, y0=100.0, x1=100.0, y1=bound_max,
                  fillcolor="rgba(56, 189, 248, 0.08)", line_width=0, layer="below")

    # 2. Concentric Velocity Radar Rings (Isolines)
    for r in [2.5, 5.0, 7.5]:
        if r < span:
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=100.0 - r, y0=100.0 - r, x1=100.0 + r, y1=100.0 + r,
                line=dict(color="rgba(148, 163, 184, 0.16)", width=1, dash="dot"),
                layer="below"
            )

    # 3. High-Contrast Center Crosshairs at (100.0, 100.0)
    fig.add_vline(x=100.0, line_width=1.5, line_dash="dash", line_color="#64748b")
    fig.add_hline(y=100.0, line_width=1.5, line_dash="dash", line_color="#64748b")

    # 4. Corner Watermark Labels
    font_lead = dict(family="Inter, sans-serif", size=11, color="rgba(16, 185, 129, 0.85)")
    font_imp  = dict(family="Inter, sans-serif", size=11, color="rgba(56, 189, 248, 0.85)")
    font_weak = dict(family="Inter, sans-serif", size=11, color="rgba(245, 158, 11, 0.85)")
    font_lag  = dict(family="Inter, sans-serif", size=11, color="rgba(239, 68, 68, 0.85)")

    fig.add_annotation(
        x=bound_max - span*0.04, y=bound_max - span*0.04,
        text="<b>LEADING</b><br><span style='font-size:9px;'>Outperformance + Speed</span>",
        showarrow=False, xanchor="right", yanchor="top", font=font_lead
    )
    fig.add_annotation(
        x=bound_min + span*0.04, y=bound_max - span*0.04,
        text="<b>IMPROVING</b><br><span style='font-size:9px;'>Turnaround Alpha</span>",
        showarrow=False, xanchor="left", yanchor="top", font=font_imp
    )
    fig.add_annotation(
        x=bound_max - span*0.04, y=bound_min + span*0.04,
        text="<b>WEAKENING</b><br><span style='font-size:9px;'>Distribution Alert</span>",
        showarrow=False, xanchor="right", yanchor="bottom", font=font_weak
    )
    fig.add_annotation(
        x=bound_min + span*0.04, y=bound_min + span*0.04,
        text="<b>LAGGING</b><br><span style='font-size:9px;'>Underperformance</span>",
        showarrow=False, xanchor="left", yanchor="bottom", font=font_lag
    )

    palette = {
        "Leading": "#10b981",    # Emerald
        "Weakening": "#fbbf24",  # Amber
        "Lagging": "#ef4444",    # Rose Red
        "Improving": "#38bdf8"   # Cyan Blue
    }

    active_df = snapshot_df.copy()
    if selected_sectors:
        active_df = active_df[active_df["Sector"].isin(selected_sectors)]

    # 5. Smart Label De-Collision Logic (Solves Center Blob)
    labeled_sectors = set()
    if focused_sector and focused_sector in active_df["Sector"].values:
        labeled_sectors.add(focused_sector)

    if label_mode == "All Labels":
        labeled_sectors = set(active_df["Sector"].tolist())
    elif label_mode == "Selected Sector Only":
        labeled_sectors = set([focused_sector]) if focused_sector else set()
    elif label_mode == "Hover Only":
        labeled_sectors = set()
    else:  # "Smart (De-Collided)" default
        # Add focused sector coordinate first
        labeled_coords = []
        if focused_sector and focused_sector in active_df["Sector"].values:
            f_row = active_df[active_df["Sector"] == focused_sector].iloc[0]
            labeled_coords.append((f_row["RS_Ratio"], f_row["RS_Momentum"]))

        # Sort remaining sectors by kinetic velocity (outliers get priority)
        unlabeled = active_df[active_df["Sector"] != focused_sector].sort_values(by="Velocity", ascending=False)
        for _, r in unlabeled.iterrows():
            pt = (r["RS_Ratio"], r["RS_Momentum"])
            # If coordinates list is empty, add first
            if not labeled_coords:
                labeled_sectors.add(r["Sector"])
                labeled_coords.append(pt)
                continue
            
            # Check minimum distance to any already labeled sector
            min_dist = min([math.sqrt((pt[0]-c[0])**2 + (pt[1]-c[1])**2) for c in labeled_coords])
            # Only label if spaced at least 1.1 units away, or if it is a high-speed outlier
            if min_dist >= 1.1 or r["Velocity"] >= 1.5:
                labeled_sectors.add(r["Sector"])
                labeled_coords.append(pt)

    # 6. Plot Rotating Trails & Endpoint Markers
    for _, row in active_df.iterrows():
        sec_name = row["Sector"]
        quad = row["Quadrant"]
        base_color = palette.get(quad, "#94a3b8")
        is_focused = (focused_sector and sec_name == focused_sector)
        
        t_info = trails_dict.get(sec_name, {})
        coords = t_info.get("coords", [])
        
        line_opacity = 0.95 if is_focused else 0.55
        line_w = 3.2 if is_focused else 1.8
        bubble_size = 18 if is_focused else 12

        if coords:
            tx = [c["x"] for c in coords]
            ty = [c["y"] for c in coords]
            
            fig.add_trace(go.Scatter(
                x=tx,
                y=ty,
                mode="lines+markers",
                line=dict(color=base_color, width=line_w),
                opacity=line_opacity,
                marker=dict(
                    size=[3 + i * 1.4 for i in range(len(tx))],
                    color=base_color,
                    opacity=0.65
                ),
                hoverinfo="skip",
                showlegend=False
            ))

        # Directional position to push labels outward from center
        if quad == "Leading":
            pos = "top right"
        elif quad == "Improving":
            pos = "top left"
        elif quad == "Lagging":
            pos = "bottom left"
        else:
            pos = "bottom right"

        show_text = (sec_name in labeled_sectors)
        label_display = f" <b>{sec_name}</b>" if show_text else ""

        hover_html = (
            f"<b>{sec_name}</b> ({row['Symbol']})<br>"
            f"─────────────────────────────<br>"
            f"<b>Quadrant:</b> {quad.upper()}<br>"
            f"<b>RS-Ratio (X):</b> {row['RS_Ratio']:.2f} ({row['Delta_X']:+.2f})<br>"
            f"<b>RS-Mom (Y):</b> {row['RS_Momentum']:.2f} ({row['Delta_Y']:+.2f})<br>"
            f"<b>Heading:</b> {row['Heading_Angle']:.0f}° {row['Heading_Short']}<br>"
            f"<b>Kinetic Speed:</b> {row['Velocity']:.2f}<br>"
            f"<b>Key Leaders:</b> {row['Top_Stocks']}<br>"
            f"<b>Playbook:</b><br>{row['Playbook'][:85]}..."
        )

        marker_border = "#ffffff" if is_focused else "rgba(255,255,255,0.7)"
        marker_border_w = 2.8 if is_focused else 1.2

        fig.add_trace(go.Scatter(
            x=[row["RS_Ratio"]],
            y=[row["RS_Momentum"]],
            mode="markers+text" if show_text else "markers",
            name=sec_name,
            marker=dict(
                size=bubble_size,
                color=base_color,
                line=dict(color=marker_border, width=marker_border_w),
                symbol="circle"
            ),
            text=[label_display],
            textposition=pos,
            textfont=dict(color="#f8fafc", size=10, family="Inter, sans-serif"),
            hovertext=[hover_html],
            hoverinfo="text",
            showlegend=False
        ))

    # 7. Symmetrical Plotly Layout (Fluid Rectangular Fit)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        height=590,
        margin=dict(l=20, r=20, t=25, b=25),
        showlegend=False,
        xaxis=dict(
            title="<b>JdK RS-Ratio (Relative Trend vs Benchmark)</b> [Center = 100.0]",
            range=[bound_min, bound_max],
            zeroline=False,
            gridcolor="rgba(51, 65, 85, 0.35)",
            tickfont=dict(color="#cbd5e1", family="JetBrains Mono", size=10)
        ),
        yaxis=dict(
            title="<b>JdK RS-Momentum (Velocity / Rate of Change)</b> [Center = 100.0]",
            range=[bound_min, bound_max],
            zeroline=False,
            gridcolor="rgba(51, 65, 85, 0.35)",
            tickfont=dict(color="#cbd5e1", family="JetBrains Mono", size=10)
        ),
        hoverlabel=dict(
            bgcolor="#0f172a",
            font_size=11,
            font_family="JetBrains Mono, monospace"
        )
    )

    return fig


# =============================================================================
# WORLD-CLASS STREAMLIT UI COCKPIT
# =============================================================================

def render_rrg_sector_matrix_view():
    """
    Renders the Bloomberg/Koyfin-grade RRG Cockpit:
      - Clean custom CSS for modern segmented pills (zero ugly radio dots).
      - Weekly Timeframe as default.
      - 1-Click Universe Presets (All Themes, Core 10, Thematic ETFs, Outperformers).
      - Interactive Quadrant Filter Tabs.
      - 2-Column Cockpit Layout: Radar on Left, Live Focus Inspector HUD on Right with 100% clean HTML.
      - Sortable Institutional Matrix Data Table.
    """
    # -------------------------------------------------------------------------
    # INJECT CUSTOM SEGMENTED PILL STYLES
    # -------------------------------------------------------------------------
    st.markdown(clean_html("""
    <style>
    /* Sleek Segmented Pill Tabs for RRG Controls */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        background: transparent;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 6px !important;
        padding: 5px 12px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: rgba(51, 65, 85, 0.8) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
        background: rgba(56, 189, 248, 0.16) !important;
        border-color: #38bdf8 !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label p {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        margin: 0 !important;
    }
    </style>
    """), unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 1. TOP SINGLE-ROW CONTROL BAR
    # -------------------------------------------------------------------------
    c_tf, c_trail, c_bench, c_labels = st.columns([1.6, 1.8, 1.8, 1.8])
    
    with c_tf:
        timeframe = st.selectbox(
            "Timeframe",
            ["Weekly (Institutional Trend)", "Daily (Tactical Swing)"],
            index=0,  # Weekly as requested
            help="Weekly is the institutional benchmark (Himanshu Sharma rule: filters noise for multi-week moves). Daily is for tactical 5-15 day swing timing."
        )
        is_weekly = "Weekly" in timeframe
        tf_clean = "Weekly" if is_weekly else "Daily"

    with c_trail:
        trail_unit = "Weeks" if is_weekly else "Trading Days"
        trail_length = st.select_slider(
            f"Trail History ({trail_unit})",
            options=[3, 5, 8, 12],
            value=5,
            help=f"Number of historical {trail_unit.lower()} plotted as rotating tail vectors (5 {trail_unit.lower()} = {('~1 month of positioning' if is_weekly else '1 full trading week')})."
        )

    with c_bench:
        benchmark_choice = st.selectbox(
            "Benchmark",
            ["Nifty 500 (^CRSLDX) [Broad Market]", "Nifty 50 (^NSEI) [LargeCap]"],
            index=0,
            help="Standard relative strength benchmark. Nifty 500 represents the total institutional Indian equity universe."
        )
        bench_sym = DEFAULT_BENCHMARK if "500" in benchmark_choice else ALT_BENCHMARK

    with c_labels:
        label_mode = st.selectbox(
            "Canvas Labels",
            ["Smart (De-Collided)", "Selected Sector Only", "Hover Only", "All Labels"],
            index=0,
            help="'Smart (De-Collided)' automatically prevents overlapping white text blobs in crowded center clusters."
        )

    # Fetch data and generate calculations
    prices_df = fetch_rrg_historical_prices(benchmark=bench_sym)
    if prices_df.empty:
        st.error("⚠️ Unable to load historical market data for sector universe. Please verify network access.")
        return

    snapshot_df, trails_dict = generate_rrg_snapshot(
        prices_df=prices_df,
        benchmark_symbol=bench_sym,
        timeframe=tf_clean,
        trail_length=trail_length
    )

    if snapshot_df.empty:
        st.warning("No sector rotation data computed.")
        return

    # -------------------------------------------------------------------------
    # 2. 1-CLICK UNIVERSE PRESET PILLS
    # -------------------------------------------------------------------------
    p_col1, p_col2 = st.columns([3.6, 1.4])
    with p_col1:
        preset_choice = st.radio(
            "Quick Sector Filter Preset",
            [
                "⚡ All Themes (18)",
                "🏛️ Core 10 Sectors",
                "🛡️ Thematic ETFs (7)",
                "🚀 Outperformers (Leading + Improving)"
            ],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )

    if "Core 10" in preset_choice:
        active_sectors = [s for s in CORE_10_SECTORS if s in snapshot_df["Sector"].values]
    elif "Thematic" in preset_choice:
        active_sectors = [s for s in THEMATIC_ETFS if s in snapshot_df["Sector"].values]
    elif "Outperformers" in preset_choice:
        active_sectors = snapshot_df[snapshot_df["Quadrant"].isin(["Leading", "Improving"])]["Sector"].tolist()
    else:
        active_sectors = snapshot_df["Sector"].tolist()

    with p_col2:
        with st.expander("🔍 Custom Pick..."):
            custom_picks = st.multiselect(
                "Filter specific sectors",
                options=snapshot_df["Sector"].tolist(),
                default=active_sectors,
                label_visibility="collapsed"
            )
            if custom_picks:
                active_sectors = custom_picks

    # -------------------------------------------------------------------------
    # 3. INTERACTIVE QUADRANT FILTER TABS
    # -------------------------------------------------------------------------
    lead_cnt = len(snapshot_df[snapshot_df["Quadrant"] == "Leading"])
    imp_cnt  = len(snapshot_df[snapshot_df["Quadrant"] == "Improving"])
    weak_cnt = len(snapshot_df[snapshot_df["Quadrant"] == "Weakening"])
    lag_cnt  = len(snapshot_df[snapshot_df["Quadrant"] == "Lagging"])

    net_bullish = (lead_cnt + imp_cnt) - (weak_cnt + lag_cnt)
    if net_bullish >= 4:
        macro_bias = "🟢 Broad-Market Institutional Expansion"
        macro_color = "#10b981"
    elif net_bullish > 0:
        macro_bias = "🟢 Selective Sector Accumulation"
        macro_color = "#34d399"
    elif net_bullish == 0:
        macro_bias = "🟡 Neutral Rotational Churn"
        macro_color = "#fbbf24"
    else:
        macro_bias = "🔴 Defensive Capital Contraction"
        macro_color = "#f87171"

    q_filter_choice = st.radio(
        "Isolate Quadrant",
        [
            f"🌐 All Active ({len(active_sectors)})",
            f"🟢 Leading ({lead_cnt})",
            f"🔵 Improving ({imp_cnt})",
            f"🟡 Weakening ({weak_cnt})",
            f"🔴 Lagging ({lag_cnt})"
        ],
        index=0,
        horizontal=True
    )

    filtered_sectors = active_sectors.copy()
    if "Leading" in q_filter_choice:
        filtered_sectors = [s for s in filtered_sectors if s in snapshot_df[snapshot_df["Quadrant"] == "Leading"]["Sector"].values]
    elif "Improving" in q_filter_choice:
        filtered_sectors = [s for s in filtered_sectors if s in snapshot_df[snapshot_df["Quadrant"] == "Improving"]["Sector"].values]
    elif "Weakening" in q_filter_choice:
        filtered_sectors = [s for s in filtered_sectors if s in snapshot_df[snapshot_df["Quadrant"] == "Weakening"]["Sector"].values]
    elif "Lagging" in q_filter_choice:
        filtered_sectors = [s for s in filtered_sectors if s in snapshot_df[snapshot_df["Quadrant"] == "Lagging"]["Sector"].values]

    # -------------------------------------------------------------------------
    # 4. 2-COLUMN COCKPIT LAYOUT (CHART + LIVE FOCUS INSPECTOR HUD)
    # -------------------------------------------------------------------------
    c_chart, c_hud = st.columns([2.0, 1.1])

    available_focus_list = filtered_sectors if filtered_sectors else snapshot_df["Sector"].tolist()
    default_focus_sec = available_focus_list[0] if available_focus_list else snapshot_df["Sector"].iloc[0]

    with c_hud:
        st.markdown(clean_html("""
        <div style="font-size: 13px; font-weight: 700; color: #f8fafc; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
            🎯 Live Sector Focus Inspector
        </div>
        """), unsafe_allow_html=True)

        focused_sec_name = st.selectbox(
            "Select Sector to Inspect",
            options=available_focus_list,
            index=0,
            label_visibility="collapsed"
        )

        focus_row = snapshot_df[snapshot_df["Sector"] == focused_sec_name].iloc[0]
        f_quad = focus_row["Quadrant"]
        f_color = focus_row["Heading_Color"]

        quad_badge_bg = (
            "rgba(16, 185, 129, 0.15)" if f_quad == "Leading" else (
            "rgba(56, 189, 248, 0.15)" if f_quad == "Improving" else (
            "rgba(245, 158, 11, 0.15)" if f_quad == "Weakening" else "rgba(239, 68, 68, 0.15)"))
        )
        quad_badge_border = (
            "#10b981" if f_quad == "Leading" else (
            "#38bdf8" if f_quad == "Improving" else (
            "#f59e0b" if f_quad == "Weakening" else "#ef4444"))
        )

        # High-Contrast Glassmorphism HUD Card (Clean HTML with zero leading indentation)
        hud_card_html = clean_html(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 10px; padding: 14px; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.06); padding-bottom: 8px;">
                <div>
                    <div style="font-size: 15px; font-weight: 800; color: #f8fafc;">{focused_sec_name}</div>
                    <div style="font-size: 11px; color: #94a3b8; font-family: monospace;">{focus_row['Symbol']}</div>
                </div>
                <div style="background: {quad_badge_bg}; border: 1px solid {quad_badge_border};
                            color: {quad_badge_border}; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 6px;">
                    {f_quad.upper()}
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px;">
                <div style="background: rgba(30, 41, 59, 0.5); padding: 8px 10px; border-radius: 6px;">
                    <div style="font-size: 10px; color: #94a3b8; text-transform: uppercase;">RS-Ratio (X)</div>
                    <div style="font-size: 16px; font-weight: 700; color: #f8fafc;">{focus_row['RS_Ratio']:.2f}</div>
                    <div style="font-size: 10px; color: {'#34d399' if focus_row['Delta_X'] >= 0 else '#f87171'}; font-weight: 600;">
                        {focus_row['Delta_X']:+.2f} ({trail_length} {trail_unit.lower()})
                    </div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.5); padding: 8px 10px; border-radius: 6px;">
                    <div style="font-size: 10px; color: #94a3b8; text-transform: uppercase;">RS-Momentum (Y)</div>
                    <div style="font-size: 16px; font-weight: 700; color: #f8fafc;">{focus_row['RS_Momentum']:.2f}</div>
                    <div style="font-size: 10px; color: {'#34d399' if focus_row['Delta_Y'] >= 0 else '#f87171'}; font-weight: 600;">
                        {focus_row['Delta_Y']:+.2f} ({trail_length} {trail_unit.lower()})
                    </div>
                </div>
            </div>

            <div style="margin-top: 10px; background: rgba(30, 41, 59, 0.4); padding: 8px 10px; border-radius: 6px; font-size: 11px;">
                <div style="color: #94a3b8;">Heading Compass: <b style="color: {f_color};">{focus_row['Heading_Angle']:.0f}° {focus_row['Heading_Short']}</b></div>
                <div style="color: #94a3b8; margin-top: 3px;">Kinetic Speed: <b style="color: #f8fafc;">{focus_row['Velocity']:.2f}</b> | Transition: <b style="color: #e2e8f0;">{focus_row['Transition']}</b></div>
            </div>

            <div style="margin-top: 12px; border-left: 3px solid {quad_badge_border}; background: rgba(30, 41, 59, 0.6); padding: 10px; border-radius: 0 6px 6px 0;">
                <div style="font-size: 10px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">CANSLIM Tactical Playbook</div>
                <div style="font-size: 12px; color: #e2e8f0; margin-top: 4px; line-height: 1.4;">{focus_row['Playbook']}</div>
            </div>

            <div style="margin-top: 10px; font-size: 11px; color: #94a3b8;">
                👑 <b>Key Constituents:</b> <span style="color: #38bdf8; font-weight: 600;">{focus_row['Top_Stocks']}</span>
            </div>
        </div>
        """)
        st.markdown(hud_card_html, unsafe_allow_html=True)

        regime_card_html = clean_html(f"""
        <div style="margin-top: 10px; background: rgba(30, 41, 59, 0.5); padding: 8px 12px; border-radius: 8px; border-left: 3px solid {macro_color};">
            <div style="font-size: 10px; color: #94a3b8; text-transform: uppercase;">Institutional Market Regime</div>
            <div style="font-size: 12px; font-weight: 700; color: {macro_color}; margin-top: 2px;">{macro_bias}</div>
        </div>
        """)
        st.markdown(regime_card_html, unsafe_allow_html=True)

    with c_chart:
        fig = build_world_class_rrg_figure(
            snapshot_df=snapshot_df,
            trails_dict=trails_dict,
            trail_length=trail_length,
            selected_sectors=filtered_sectors,
            focused_sector=focused_sec_name,
            label_mode=label_mode
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # 5. SORTABLE INSTITUTIONAL DATA MATRIX & ACTION PLAYBOOK
    # -------------------------------------------------------------------------
    st.markdown("##### 📋 Institutional Sector Velocity & Action Matrix")
    
    display_df = snapshot_df[snapshot_df["Sector"].isin(filtered_sectors)].copy()

    table_records = []
    for _, r in display_df.iterrows():
        quad = r["Quadrant"]
        quad_badge = (
            f"🟢 Leading" if quad == "Leading" else (
            f"🔵 Improving" if quad == "Improving" else (
            f"🟡 Weakening" if quad == "Weakening" else f"🔴 Lagging"))
        )
        
        table_records.append({
            "Sector Theme": r["Sector"],
            "Symbol": r["Symbol"],
            "Quadrant": quad_badge,
            "RS-Ratio (X)": f"{r['RS_Ratio']:.2f}",
            f"{trail_length} {trail_unit[:1]} Δ X": f"{r['Delta_X']:+.2f}",
            "RS-Mom (Y)": f"{r['RS_Momentum']:.2f}",
            f"{trail_length} {trail_unit[:1]} Δ Y": f"{r['Delta_Y']:+.2f}",
            "Heading Vector": f"{r['Heading_Angle']:.0f}° {r['Heading_Short']}",
            "Kinetic Velocity": f"{r['Velocity']:.2f}",
            "Top Leaders": r["Top_Stocks"],
            "CANSLIM Action Playbook": r["Playbook"]
        })

    view_table_df = pd.DataFrame(table_records)
    st.dataframe(
        view_table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sector Theme": st.column_config.TextColumn("Sector Theme", width="medium"),
            "Symbol": st.column_config.TextColumn("Ticker", width="small"),
            "Quadrant": st.column_config.TextColumn("Quadrant", width="small"),
            "RS-Ratio (X)": st.column_config.TextColumn("RS-Ratio (X)", width="small"),
            f"{trail_length} {trail_unit[:1]} Δ X": st.column_config.TextColumn("Δ X", width="small"),
            "RS-Mom (Y)": st.column_config.TextColumn("RS-Mom (Y)", width="small"),
            f"{trail_length} {trail_unit[:1]} Δ Y": st.column_config.TextColumn("Δ Y", width="small"),
            "Heading Vector": st.column_config.TextColumn("Heading & Angle", width="medium"),
            "Kinetic Velocity": st.column_config.TextColumn("Kinetic Speed", width="small"),
            "Top Leaders": st.column_config.TextColumn("Key Leaders", width="medium"),
            "CANSLIM Action Playbook": st.column_config.TextColumn("CANSLIM Tactical Action Playbook", width="large")
        }
    )
