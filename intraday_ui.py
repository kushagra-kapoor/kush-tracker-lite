"""
Intraday UI/UX Cockpit Engine for Kush Tracker
Provides institutional HUD status bars, setup matrix ribbons, TradingView export utilities,
and unified column formatting for both India and US intraday monitors.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import plotly.express as px
from components import render_empty_state


def clean_html(html_str: str) -> str:
    """Strips leading whitespace from every line so Markdown never treats it as code."""
    if not html_str:
        return ""
    return "\n".join(line.strip() for line in html_str.splitlines() if line.strip())


def apply_institutional_theme(fig: go.Figure) -> go.Figure:
    """Applies high-contrast dark institutional styling to Plotly figures."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        margin=dict(l=0, r=0, t=35, b=0),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(15,23,42,0.95)", font_size=12, font_family="JetBrains Mono")
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="rgba(255,255,255,0.1)",
        tickfont=dict(color="#64748b")
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.03)",
        zeroline=True,
        zerolinecolor="rgba(255,255,255,0.1)",
        showline=False,
        tickfont=dict(color="#64748b")
    )
    return fig


def generate_tv_export(df: pd.DataFrame, ticker_col: str = 'TickerLink', is_us: bool = False) -> str:
    """Extracts clean base tickers and produces a TradingView-ready comma-separated watchlist."""
    if df is None or df.empty:
        return ""
    if 'Ticker' in df.columns:
        tickers = [re.sub(r'[^\w\.\-\_]', '', str(t)).strip() for t in df['Ticker']]
    elif ticker_col in df.columns:
        pattern = r"symbol=([^&]*)" if is_us else r"symbol=NSE:([^&]*)"
        extracted = df[ticker_col].astype(str).str.extract(pattern)[0]
        tickers = [re.sub(r'[^\w\.\-\_]', '', str(t)).strip() for t in extracted if pd.notna(t)]
    else:
        return ""

    if is_us:
        return ",".join([t for t in tickers if t])
    else:
        return ",".join([f"NSE:{t}" for t in tickers if t])


def render_tv_copy_box(df: pd.DataFrame, label: str = "Watchlist", is_us: bool = False):
    """Renders a collapsible copy block with full TradingView watchlist string."""
    count = len(df) if df is not None else 0
    tv_str = generate_tv_export(df, is_us=is_us)
    with st.expander(f"📋 Copy TradingView List ({count} {label})", expanded=False):
        if tv_str:
            st.code(tv_str, language="text")
        else:
            st.caption("No tickers available to export.")


def style_dataframe(df: pd.DataFrame):
    """Applies standardized negative-red / positive-green and institutional row highlights."""
    pct_cols = [c for c in df.columns if '%' in c or 'Ret' in c or 'Breakout' in c or 'Intraday' in c]

    def color_negative_red(val):
        try:
            val_f = float(val)
            if val_f < 0:
                return 'color: #ef4444'
            if val_f > 0:
                return 'color: #10b981'
        except Exception:
            pass
        return ''

    def highlight_intraday_rows(row):
        is_tml = '👑' in str(row.get('Ticker', row.get('TickerLink', '')))
        vol_mult = row.get('Volume Expansion', row.get('Vol Multiplier', 0))
        is_bo = False
        trigger = str(row.get('Setup Trigger', ''))
        if 'BO' in trigger or 'Breakout' in trigger:
            is_bo = True

        try:
            vol_f = float(vol_mult) if pd.notna(vol_mult) else 0.0
        except Exception:
            vol_f = 0.0

        if vol_f >= 5.0:
            return ['background-color: rgba(217, 70, 239, 0.12); color: #f0abfc'] * len(row)
        elif is_bo and vol_f >= 2.0:
            return ['background-color: rgba(56, 189, 248, 0.12); color: #bae6fd'] * len(row)
        elif is_tml:
            return ['background-color: rgba(234, 179, 8, 0.12); color: #fde047'] * len(row)
        return [''] * len(row)

    styler = df.style
    if pct_cols:
        existing_pct_cols = [c for c in pct_cols if c in df.columns]
        if existing_pct_cols:
            if hasattr(styler, 'map'):
                styler = styler.map(color_negative_red, subset=existing_pct_cols)
            else:
                styler = styler.applymap(color_negative_red, subset=existing_pct_cols)

    return styler.apply(highlight_intraday_rows, axis=1)


def get_intraday_column_config(is_us: bool = False):
    """Unified column configurations with tooltips, number formats, and progress bars."""
    turnover_name = "Turnover $M" if is_us else "Turnover ₹Cr"
    turnover_col = "Dollar Volume ($M)" if is_us else "Dollar Volume (Cr)"

    cfg = {
        "TickerLink": st.column_config.LinkColumn("Ticker", display_text=r"name=(.*)", width="small"),
        "RS Rating": st.column_config.ProgressColumn(
            "RS Score",
            min_value=0,
            max_value=99,
            format="%d",
            help="Market-Normalized Alpha Relative Strength Rating (0 to 99)"
        ),
        "Today %": st.column_config.NumberColumn("Today %", format="%+.2f%%"),
        "Volume Expansion": st.column_config.NumberColumn(
            "Vol Surge",
            format="%.2fx",
            help="Today's Volume vs 20-Day Average Volume"
        ),
        turnover_col: st.column_config.NumberColumn(turnover_name, format="%.1f"),
        "Close Range %": st.column_config.ProgressColumn(
            "Close Range",
            min_value=0,
            max_value=100,
            format="%d%%",
            help="Where stock is trading within its high-low range (100% = at high of day)"
        ),
        "Dist from 10 EMA %": st.column_config.NumberColumn("Dist 10 EMA", format="%+.2f%%"),
        "Dist from 50 SMA %": st.column_config.NumberColumn("Dist 50 SMA", format="%+.2f%%"),
        "Dist from Day High %": st.column_config.NumberColumn("Dist High", format="%.2f%%"),
        "Dist from 52W High %": st.column_config.NumberColumn("Dist 52W High", format="%.2f%%", help="Distance below 52-Week High (%)"),
        "ADR %": st.column_config.NumberColumn("ADR %", format="%.2f%%", help="20-Day Average Daily Range %"),
        "1M Return": st.column_config.NumberColumn("1M Ret", format="%+.2f%%"),
        "1M Ret Pctile": st.column_config.NumberColumn("1M Pctile", format="%.0fth"),
        "3M Return": st.column_config.NumberColumn("3M Ret", format="%+.2f%%"),
        "3M Ret Pctile": st.column_config.NumberColumn("3M Pctile", format="%.0fth"),
        "Industry": st.column_config.TextColumn("Sector / Theme", width="medium"),
        "Setup Trigger": st.column_config.TextColumn("Setup Trigger", width="medium"),
        "Active Setups": st.column_config.TextColumn("Active Setups", width="medium"),
        "12D Trend Score": st.column_config.NumberColumn(
            "12D Score",
            format="%d/12",
            help="Count of green days in the last 12 sessions"
        ),
        "12D Heatmap": st.column_config.TextColumn("12D Heatmap", help="Oldest (Left) -> Newest (Right)"),
        "Gap %": st.column_config.NumberColumn("Gap %", format="%+.2f%%"),
        "Launchpad Compression %": st.column_config.NumberColumn("Compression", format="%.2f%%"),
        "Vol vs 1Y Max": st.column_config.NumberColumn("Vol vs 1Y Max", format="%.2fx"),
        "HTF Thrust %": st.column_config.NumberColumn("Pole Thrust %", format="%+.1f%%"),
        "HTF Drawdown %": st.column_config.NumberColumn("Flag Drawdown %", format="%+.1f%%"),
        "Is Power Trend": st.column_config.CheckboxColumn("⚡ Power Trend", help="Strict multi-timeframe MA uptrend"),
        "Is Ants": st.column_config.CheckboxColumn("🐜 Ants", help="12+ up days in last 15"),
        "HV1 Defended": st.column_config.CheckboxColumn("🛡️ HV1 Defender", help="Within 5% of 1Y HV1 Anchored VWAP"),
        "Beach Ball Score": st.column_config.ProgressColumn("BB Score", min_value=0, max_value=100, format="%d", help="Beach Ball Divergence Strength Score (0 to 100)"),
        "Is Beach Ball": st.column_config.CheckboxColumn("🏖️ Beach Ball", help="Steve Cohen / Shake Pryzby Beach Ball Divergence Leader"),
        "Sister Stock Tandem": st.column_config.TextColumn(
            "Sister Tandem",
            width="medium",
            help="Jesse Livermore Sister Stock Tandem: Tracks the #1 and #2 RS leaders in the industry group to confirm group accumulation vs solitary false move"
        ),
        "Support Bounce Type": st.column_config.TextColumn(
            "Bounce MA",
            width="small",
            help="Jim Roppel Institutional Support Bounce: Stage 2 leader touching rising 21 EMA or 50 SMA on quiet volume and rebounding"
        ),
        "Bounce Dist %": st.column_config.NumberColumn(
            "Touch Dist",
            format="%+.2f%%",
            help="Distance between today's intraday low and the moving average"
        ),
        "Is Roppel Bounce": st.column_config.CheckboxColumn("🎯 Roppel Bounce", help="Jim Roppel 21 EMA / 50 SMA Institutional Support Bounce"),
    }
    return cfg


def render_setup_table(
    df: pd.DataFrame,
    columns: list,
    is_us: bool = False,
    empty_msg: str = "No setups detected matching criteria today.",
    empty_icon: str = "🧘‍♂️",
    height: int = 420
):
    """Renders a clean, full-width, styled dataframe with TradingView export."""
    if df is None or df.empty:
        render_empty_state(empty_msg, empty_icon)
        return

    # Filter columns to only those actually present in df
    valid_cols = [c for c in columns if c in df.columns]
    display_df = df[valid_cols].copy()

    styled_df = style_dataframe(display_df)
    cfg = get_intraday_column_config(is_us=is_us)

    st.dataframe(
        styled_df,
        column_config=cfg,
        hide_index=True,
        use_container_width=True,
        height=height
    )


def get_active_setups_tags(row) -> str:
    """Returns a comma-separated list of active bullish setup tags for a stock."""
    setups = []
    if row.get('Is EP', False):
        setups.append("💥 EP")
    if row.get('Is Elite Breakout', False):
        setups.append("🏆 Stage 2 BO")
    elif row.get('Is Breakout', False):
        setups.append("⚡ BO")
    if row.get('Is Roppel Bounce', False):
        b_type = str(row.get('Support Bounce Type', '21/50'))
        setups.append(f"🎯 {b_type} Bounce")
    if row.get('Is 50 Reclaim', False):
        setups.append("🎣 50 SMA")
    if row.get('Is Launchpad', False):
        setups.append("📦 VCP")
    if row.get('Is GLB', False):
        setups.append("🟢 GLB")
    if row.get('Is HTF', False):
        setups.append("🚩 HTF")
    if row.get('Is EMA Crossback', False):
        setups.append("🏹 Kell Cross")
    if row.get('Is Reversal Ext', False):
        setups.append("🪃 Rev Ext")
    if row.get('Is 3WT', False):
        setups.append("⏳ 3WT")
    if row.get('Is HV1', False):
        setups.append("🔊 HV1")
    if row.get('Is Beach Ball', False):
        setups.append("🏖️ Beach Ball")
    if row.get('Is New High', False):
        setups.append("🌟 52W High")
    if row.get('Is Focus', False):
        setups.append("⭐ Focus")
    return ", ".join(setups) if setups else "None"


def detect_roppel_support_bounce(
    current_price: float,
    today_low: float,
    sma_21: float,
    sma_50: float,
    sma_200: float,
    vol_expansion: float = 1.0,
    close_range_pct: float = 50.0,
    today_pct: float = 0.0
) -> dict:
    """
    Jim Roppel & John Boik Institutional Support Bounce Detector (Monster Stocks Step 4).
    
    Identifies Stage 2 leaders quietly testing their rising 21-day EMA or 50-day SMA on
    contracting volume and reversing off support intraday.
    
    Conditions:
      1. Stage 2 Trend: Price > SMA 50 and SMA 50 > SMA 200 (or Price > SMA 200).
      2. Key MA Touch: Today's low came within -1.8% to +1.0% of 21 EMA or 50 SMA.
      3. Constructive Bounce: Current price >= MA and closing range >= 35% (or positive return).
      4. Volume Dry-Up: Volume expansion <= 1.40x (quiet institutional absorption, not heavy distribution).
    """
    res = {
        'is_bounce': False,
        'bounce_type': None,
        'bounce_dist_pct': 0.0,
        'badge': ''
    }
    
    if current_price <= 0 or sma_50 <= 0:
        return res
        
    is_stage_2 = (current_price > sma_50) and (sma_50 > sma_200 or sma_200 == 0)
    if not is_stage_2:
        if not (current_price > sma_200 and sma_200 > 0):
            return res
            
    # Check constructive action (not dumping at lows of day)
    constructive = (close_range_pct >= 30.0 or today_pct >= 0.0)
    quiet_vol = (vol_expansion <= 1.5)
    
    # 1. 21 EMA check
    if sma_21 > 0:
        dist_low_21 = ((today_low - sma_21) / sma_21) * 100.0
        dist_curr_21 = ((current_price - sma_21) / sma_21) * 100.0
        
        if -1.8 <= dist_low_21 <= 1.0 and dist_curr_21 >= -0.3 and constructive and quiet_vol:
            res['is_bounce'] = True
            res['bounce_type'] = '21 EMA'
            res['bounce_dist_pct'] = round(dist_low_21, 2)
            res['badge'] = '🟢 21 EMA Support'
            return res
                
    # 2. 50 SMA check (Institutional Line in the Sand)
    if sma_50 > 0:
        dist_low_50 = ((today_low - sma_50) / sma_50) * 100.0
        dist_curr_50 = ((current_price - sma_50) / sma_50) * 100.0
        
        if -1.8 <= dist_low_50 <= 1.2 and dist_curr_50 >= -0.3 and constructive and quiet_vol:
            res['is_bounce'] = True
            res['bounce_type'] = '50 SMA'
            res['bounce_dist_pct'] = round(dist_low_50, 2)
            res['badge'] = '🔵 50 SMA Institutional'
            return res
                
    return res


def compute_sister_stock_tandems(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Jesse Livermore 'Sister Stock' Tandem Confirmation Engine (Reminiscences of a Stock Operator).
    
    For each industry group, identifies the top 2 RS Rating stocks (Leader 1 and Leader 2).
    Determines whether the sister stock confirms or diverges:
      - Confirmed: Sister is above 20 EMA, near 52W high, and not dumping.
      - Divergence: Sister is breaking down below key MAs or dropping sharply.
    """
    if results_df is None or results_df.empty or 'Industry' not in results_df.columns:
        return results_df
        
    df = results_df.copy()
    
    sister_tags = {}
    sister_tickers = {}
    
    for ind, group in df.groupby('Industry'):
        if pd.isna(ind) or ind == 'Unknown' or len(group) == 0:
            continue
            
        sorted_grp = group.sort_values(by='RS Rating', ascending=False)
        
        if len(sorted_grp) == 1:
            t1 = sorted_grp.iloc[0]['Ticker']
            sister_tags[t1] = '— Solitary Leader'
            sister_tickers[t1] = 'None'
            continue
            
        row1 = sorted_grp.iloc[0]
        row2 = sorted_grp.iloc[1]
        t1 = row1['Ticker']
        t2 = row2['Ticker']
        clean_t1 = str(t1).replace('👑', '').replace('🔥', '').replace('⚡', '').replace('🚀', '').strip()
        clean_t2 = str(t2).replace('👑', '').replace('🔥', '').replace('⚡', '').replace('🚀', '').strip()
        
        ret1 = float(row1.get('Today %', 0.0))
        ret2 = float(row2.get('Today %', 0.0))
        rs1 = int(row1.get('RS Rating', 50))
        rs2 = int(row2.get('RS Rating', 50))
        dist52_1 = float(row1.get('Dist from 52W High %', 0.0))
        dist52_2 = float(row2.get('Dist from 52W High %', 0.0))
        
        t1_healthy = (ret1 >= -1.5) and (dist52_1 <= 22.0)
        t2_healthy = (ret2 >= -1.5) and (dist52_2 <= 22.0)
        both_healthy = t1_healthy and t2_healthy
        tag1 = f"👯 Confirmed ({clean_t2} {ret2:+.1f}%, RS{rs2})" if both_healthy else f"⚠️ Divergence ({clean_t2} {ret2:+.1f}%, RS{rs2})"
        tag2 = f"👯 Confirmed ({clean_t1} {ret1:+.1f}%, RS{rs1})" if both_healthy else f"⚠️ Divergence ({clean_t1} {ret1:+.1f}%, RS{rs1})"
            
        sister_tags[t1] = tag1
        sister_tickers[t1] = clean_t2
        sister_tags[t2] = tag2
        sister_tickers[t2] = clean_t1
        
        for k in range(2, len(sorted_grp)):
            tk = sorted_grp.iloc[k]['Ticker']
            both_confirmed = t1_healthy and t2_healthy
            if both_confirmed:
                sister_tags[tk] = f"🛡️ Group Confirmed ({clean_t1}/{clean_t2})"
            else:
                sister_tags[tk] = f"⚠️ Group Mixed ({clean_t1}/{clean_t2})"
            sister_tickers[tk] = f"{clean_t1}, {clean_t2}"
            
    df['Sister Stock Tandem'] = df['Ticker'].map(sister_tags).fillna('— Solitary')
    df['Sister Ticker'] = df['Ticker'].map(sister_tickers).fillna('None')
    return df


def detect_beach_ball_divergence(results_df: pd.DataFrame, index_state: dict = None, is_us: bool = False) -> pd.DataFrame:
    """
    Steve Cohen (Market Wizards) & Shake Pryzby (Pinpoint Trading p. 106)
    'Beach Ball Under Water' Relative Strength Divergence Detector.
    
    Identifies resilient institutional leaders that refuse to drop while the benchmark index
    is undergoing a correction or pullback.
    """
    if results_df is None or results_df.empty:
        return results_df
        
    df = results_df.copy()
    
    # 1. Price above key moving averages
    dist_50 = df.get('Dist from 50 SMA %', pd.Series(0.0, index=df.index)).fillna(0.0)
    dist_10 = df.get('Dist from 10 EMA %', pd.Series(0.0, index=df.index)).fillna(0.0)
    above_mas = (dist_50 >= -1.0) & (dist_10 >= -3.5)
    
    # 2. Resilient price action: Today % >= 0.0 or 1M Return >= 0.0
    today_pct = df.get('Today %', pd.Series(0.0, index=df.index)).fillna(0.0)
    ret_1m = df.get('1M Return', pd.Series(0.0, index=df.index)).fillna(0.0)
    positive_action = (today_pct >= 0.0) | (ret_1m >= 0.0)
    
    # 3. 12D Trend Score >= 6 (Constructive accumulation candles in last 12 sessions)
    score_12d = df.get('12D Trend Score', pd.Series(6, index=df.index)).fillna(6)
    trend_solid = score_12d >= 6
    
    # 4. Near 52W High (within 18%)
    dist_high = df.get('Dist from 52W High %', df.get('Dist from Day High %', pd.Series(0.0, index=df.index))).fillna(0.0)
    is_nh = df.get('Is New High', pd.Series(False, index=df.index)).fillna(False)
    is_focus = df.get('Is Focus', pd.Series(False, index=df.index)).fillna(False)
    near_high = (dist_high <= 18.0) | is_nh | is_focus
    
    # 5. Volume supply exhaustion / dry-up or constructive expansion
    vol_exp = df.get('Volume Expansion', pd.Series(1.0, index=df.index)).fillna(1.0)
    vol_ok = (vol_exp <= 4.0)
    
    # 6. High Relative Strength (RS Rating >= 70 or 👑 TML)
    rs = df.get('RS Rating', pd.Series(70, index=df.index)).fillna(70)
    is_tml = df['Ticker'].astype(str).str.contains('👑') if 'Ticker' in df.columns else pd.Series(False, index=df.index)
    rs_ok = (rs >= 70) | is_tml
    
    # Combined Beach Ball mask
    bb_mask = above_mas & positive_action & trend_solid & near_high & vol_ok & rs_ok
    df['Is Beach Ball'] = bb_mask
    
    # Compute Beach Ball Divergence Score (0-100)
    bb_score = (
        np.clip(rs, 0, 99) * 0.40 +
        np.clip(score_12d / 12.0 * 100.0, 0, 100) * 0.30 +
        np.where(dist_50 > 0, 15.0, 0.0) +
        np.where(today_pct > 0, 15.0, 0.0)
    ).round(0).astype(int)
    df['Beach Ball Score'] = np.where(bb_mask, bb_score, 0)
    
    return df


def get_beach_ball_timing_state(market_state: dict, is_us: bool = False) -> dict:
    """
    Evaluates whether today's benchmark index environment is prime for Beach Ball divergence.
    Returns timing classification, badge text, styling colors, and executive instructions.
    """
    if not market_state:
        return {
            'state': 'NEUTRAL',
            'badge': '',
            'title': 'MONITORING MARKET REGIME',
            'color': '#94a3b8',
            'bg_color': 'rgba(148, 163, 184, 0.1)',
            'border_color': '#94a3b855',
            'tag': 'OBSERVE',
            'summary': 'Benchmark index regime data is calculating.',
            'action': 'Observe benchmark moving averages and price action.'
        }
        
    qull_green = market_state.get('green_light', False)
    recap = market_state.get('recapture_sequence', {}) or market_state.get('recapture_data', {})
    
    recapture_active = recap.get('recapture_active', False)
    current_step_num = recap.get('current_step_num', 4)
    secular_bull = recap.get('secular_bull', True)
    dist_200 = recap.get('dist_200_sma_pct', 0.0)
    index_pct = market_state.get('index_pct', 0.0)
    index_name = "S&P 500" if is_us else "NIFTY 500"
    
    # 1. 🚀 Triggered Recapture (Water pressure released - Buy now)
    if recapture_active or current_step_num == 5:
        return {
            'state': 'IGNITION',
            'badge': '🚀 BUY IGNITION',
            'title': f'🚀 MAXIMUM EXPLOSION TIMING ({index_name} 20 EMA Recaptured)',
            'color': '#10b981',
            'bg_color': 'rgba(16, 185, 129, 0.14)',
            'border_color': '#10b98188',
            'tag': 'RELEASE STAGE',
            'summary': f"{index_name} has crossed and closed above its 20 EMA after an extended correction. The downward index headwind is vanishing.",
            'action': "ACTION: EXECUTE PILOT BUYS. The hand holding the beach ball underwater has been released! Top Beach Ball leaders clearing resistance pivots on expanding volume are prime buy candidates today (2.5%–5.0% risk)."
        }
    
    # 2. 🎯 Prime Stalking Environment (Water pressure active - Market in correction/chop)
    elif not qull_green or not secular_bull or current_step_num in [2, 3, 4] or index_pct < -0.2:
        step_str = f"Step {current_step_num}/7: {recap.get('current_step_name', 'Consolidating')}"
        reason = (
            f"{index_name} is in Secular Bear ({dist_200:+.1f}% from 200 SMA)" if not secular_bull else
            (f"{index_name} in {step_str}" if current_step_num in [2, 3, 4] else f"{index_name} MA slopes unaligned")
        )
        return {
            'state': 'PRIME_STALK',
            'badge': '🎯 PRIME TIMING',
            'title': f'🏖️ PRIME BEACH BALL STALKING ENVIRONMENT ({reason})',
            'color': '#38bdf8',
            'bg_color': 'rgba(56, 189, 248, 0.12)',
            'border_color': '#38bdf888',
            'tag': 'COILING UNDER PRESSURE',
            'summary': f"The broader market ({index_name}) is currently under pressure or correcting, but the stocks listed below refuse to sink and are holding near 52-week highs.",
            'action': "ACTION: STALK PIVOTS & PREPARE ALERTS. Do not chase blindly while the index is weak. Put these resilient leaders on your primary TradingView focus list and set price alerts at flag/52W highs. Be ready to pounce the instant the index recaptures 20 EMA."
        }
        
    # 3. 🟢 Broad Uptrend Active (Normal Breakout Mode)
    else:
        return {
            'state': 'BULL_NORMAL',
            'badge': '🟢 BROAD BULL',
            'title': f'🟢 BROAD UPTREND ACTIVE ({index_name} Green Light)',
            'color': '#10b981',
            'bg_color': 'rgba(16, 185, 129, 0.08)',
            'border_color': '#10b98144',
            'tag': 'NORMAL BREAKOUT MODE',
            'summary': f"{index_name} is in a healthy uptrend with 10 EMA > 20 EMA both sloping upward.",
            'action': "ACTION: TRADE STANDARD BREAKOUTS. Normal Stage 2 base breakouts and Episodic Pivots are working across sectors. Use Beach Balls primarily to spot the top relative strength (RS 90+) market generals."
        }


def render_beach_ball_timing_banner(timing_info: dict):
    """Renders the executive institutional Beach Ball timing alert box."""
    if not timing_info or not timing_info.get('title'):
        return
        
    color = timing_info.get('color', '#38bdf8')
    bg = timing_info.get('bg_color', 'rgba(56, 189, 248, 0.12)')
    border = timing_info.get('border_color', '#38bdf888')
    title = timing_info.get('title', '')
    tag = timing_info.get('tag', '')
    summary = timing_info.get('summary', '')
    action = timing_info.get('action', '')
    
    html = f"""
    <div style="background: {bg}; border: 1px solid {border}; border-left: 4px solid {color}; border-radius: 10px; padding: 12px 18px; margin: 12px 0 16px 0;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: {color}; font-size: 0.96rem; font-weight: 700; letter-spacing: 0.3px;">{title}</span>
            </div>
            <span style="background: rgba(255,255,255,0.06); color: {color}; font-weight: 700; font-size: 0.74rem; padding: 3px 10px; border-radius: 6px; border: 1px solid {color}55;">{tag}</span>
        </div>
        <div style="color: #cbd5e1; font-size: 0.84rem; margin: 6px 0 4px 0; line-height: 1.4;">
            {summary}
        </div>
        <div style="color: #f8fafc; font-size: 0.85rem; font-weight: 600; margin-top: 4px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.06);">
            👉 <span style="color: {color};">{action}</span>
        </div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)


def render_intraday_hud(
    market_state: dict,
    sentiment_val,
    sentiment_label: str,
    vix_val,
    vix_label: str,
    total_stocks: int,
    pos_count: int,
    neg_count: int,
    avg_ret: float,
    highs_count: int,
    is_us: bool = False,
    recapture_data: dict = None
):
    """Renders the executive institutional Heads-Up Display (HUD) status bar."""
    qull_green = market_state.get('green_light', False)
    ema_10_sl = market_state.get('ema_10_slope', 0.0)
    ema_20_sl = market_state.get('ema_20_slope', 0.0)

    # Extract 20 EMA Recapture & 200 SMA Data
    recap = recapture_data or market_state.get('recapture_sequence', {}) or market_state.get('recapture_data', {})
    secular_label = recap.get('secular_label', "Secular Bull (Above 200 SMA)")
    secular_color = recap.get('secular_color', "#10b981")
    dist_200 = recap.get('dist_200_sma_pct', 0.0)
    recapture_active = recap.get('recapture_active', False)
    seq_summary = recap.get('sequence_summary', "")
    step_badge_color = recap.get('step_badge_color', "#38bdf8")

    index_name = "S&P 500" if is_us else "NIFTY 500"

    if qull_green:
        banner_color = "#10b981"
        banner_icon = "🟢"
        banner_title = f"{index_name} GREEN LIGHT: CLEAR TO TRADE BREAKOUTS"
        banner_desc = "10 EMA > 20 EMA and both moving averages are sloping upward."
    elif ema_10_sl < 0 and ema_20_sl < 0:
        banner_color = "#ef4444"
        banner_icon = "🔴"
        banner_title = f"{index_name} RED LIGHT: DEFENSIVE MODE (AVOID LONGS)"
        banner_desc = "Both moving averages are sloping downward. Protect capital."
    else:
        banner_color = "#f59e0b"
        banner_icon = "🟡"
        banner_title = f"{index_name} CAUTION: TREND IS WEAKENING OR MIXED"
        banner_desc = "Moving average slopes are unaligned. Reduce position sizes."

    # Sentiment styling
    sent_title = "CNN Fear & Greed" if is_us else "Tickertape MMI"
    if sentiment_val is not None:
        try:
            val_f = float(sentiment_val)
            sent_str = f"{val_f:.0f}"
            if val_f < 30:
                sent_color = "#10b981"
            elif val_f > 70:
                sent_color = "#ef4444"
            else:
                sent_color = "#f59e0b"
        except Exception:
            sent_str = str(sentiment_val)
            sent_color = "#94a3b8"
    else:
        sent_str = "N/A"
        sent_color = "#64748b"

    # VIX styling
    vix_title = "US VIX" if is_us else "India VIX"
    if vix_val is not None:
        try:
            v_val_f = float(vix_val)
            vix_str = f"{v_val_f:.1f}"
            vix_threshold = 22 if not is_us else 25
            if v_val_f < 15:
                vix_color = "#10b981"
            elif v_val_f > vix_threshold:
                vix_color = "#ef4444"
            else:
                vix_color = "#e2e8f0"
        except Exception:
            vix_str = str(vix_val)
            vix_color = "#94a3b8"
    else:
        vix_str = "N/A"
        vix_color = "#64748b"

    pos_pct = (pos_count / total_stocks * 100) if total_stocks > 0 else 0.0

    # 20 EMA Recapture Tactical Alert Ribbon (Paul Tudor Jones & Shake Pryzby)
    recap_ribbon_html = ""
    if recapture_active:
        recap_ribbon_html = f"""
        <div style="background: linear-gradient(90deg, rgba(56, 189, 248, 0.22) 0%, rgba(16, 185, 129, 0.22) 100%); border: 1px solid #38bdf888; border-radius: 10px; padding: 10px 16px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.15rem;">🚀</span>
                <strong style="color: #38bdf8; font-size: 0.92rem; letter-spacing: 0.3px;">20 EMA RECAPTURE BUY SIGNAL ACTIVE:</strong>
                <span style="color: #e2e8f0; font-size: 0.84rem;">Index crossed and closed above 20 EMA after correction. Tactical green light for pilot risk (2.5%–5.0%) in leading breakouts.</span>
            </div>
            <span style="background: rgba(56, 189, 248, 0.18); color: #38bdf8; font-weight: 700; font-size: 0.74rem; padding: 3px 9px; border-radius: 6px; border: 1px solid #38bdf855;">PTJ / PRYZBY SIGNAL</span>
        </div>
        """

    # Tooltip text for hover guidance
    is_secular_bear = dist_200 < 0
    secular_tooltip = (
        f"Paul Tudor Jones 200 SMA Rule ({secular_label}):&#10;"
        f"{index_name} is {'below' if is_secular_bear else 'above'} its 200-day simple moving average by {dist_200:+.1f}%.&#10;"
        + ("⚠️ SECULAR BEAR: Major multi-month market crashes occur almost exclusively below the 200 SMA. Protect capital, avoid aggressive long margin, and keep stops tight."
           if is_secular_bear else
           "🟢 SECULAR BULL: Structural long-term uptrend intact. Stage 2 breakouts enjoy institutional tailwinds and higher follow-through.")
    )

    action_guidance = recap.get('action_guidance', "Monitor market posture and manage open risk.")
    current_step_num = recap.get('current_step_num', 4)
    current_step_name = recap.get('current_step_name', "Moving Average Clustering")

    seq_tooltip = (
        f"Shake Pryzby 7-Step Recovery Cycle (Pinpoint Trading):&#10;"
        f"Active Phase: {seq_summary}&#10;"
        f"Tactical Action: {action_guidance}&#10;"
        f"Click the Regime Guide below for full breakdown."
    )

    slope_tooltip = (
        f"Kristjan Qullamaggie 10/20 EMA Slopes:&#10;"
        f"10 EMA Slope: {ema_10_sl:+.2f} (Short-term momentum)&#10;"
        f"20 EMA Slope: {ema_20_sl:+.2f} (Intermediate swing trend)&#10;"
        f"Both positive = Green light for breakouts. Both negative = Red light / defense."
    )

    seq_badge_html = f'<span title="{seq_tooltip}" style="background: rgba(255,255,255,0.06); padding: 3px 9px; border-radius: 6px; color: {step_badge_color}; font-size: 0.78rem; font-weight: 700; border: 1px solid {step_badge_color}44; cursor: help;">📍 {seq_summary}</span>' if seq_summary else ''

    hud_html = f"""
    <div class="intraday-hud-container">
        {recap_ribbon_html}
        <div class="intraday-regime-bar" style="background: {banner_color}18; border: 1px solid {banner_color}55;">
            <div style="display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;">
                <span style="font-size: 1.15rem;">{banner_icon}</span>
                <span style="color: {banner_color}; font-size: 0.95rem; font-weight: 700; letter-spacing: 0.4px;">{banner_title}</span>
                <span style="color: #94a3b8; font-size: 0.82rem; margin-left: 0.4rem;">{banner_desc}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                <span title="{secular_tooltip}" style="background: rgba(255,255,255,0.06); padding: 3px 9px; border-radius: 6px; color: {secular_color}; font-size: 0.78rem; font-weight: 700; border: 1px solid {secular_color}44; cursor: help;">
                    🏛️ {secular_label} ({'+' if dist_200 >= 0 else ''}{dist_200:.1f}%)
                </span>
                {seq_badge_html}
                <div title="{slope_tooltip}" style="color: #94a3b8; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; margin-left: 4px; cursor: help;">
                    10 EMA: <strong style="color: {'#10b981' if ema_10_sl > 0 else '#ef4444'};">{ema_10_sl:+.2f}</strong> &nbsp;|&nbsp; 
                    20 EMA: <strong style="color: {'#10b981' if ema_20_sl > 0 else '#ef4444'};">{ema_20_sl:+.2f}</strong>
                </div>
            </div>
        </div>
        <div class="intraday-hud-grid">
            <div class="intraday-metric-card">
                <div class="intraday-metric-label">{sent_title}</div>
                <div class="intraday-metric-val" style="color: {sent_color};">{sent_str}</div>
                <div class="intraday-metric-sub">{sentiment_label}</div>
            </div>
            <div class="intraday-metric-card">
                <div class="intraday-metric-label">{vix_title}</div>
                <div class="intraday-metric-val" style="color: {vix_color};">{vix_str}</div>
                <div class="intraday-metric-sub">{vix_label}</div>
            </div>
            <div class="intraday-metric-card">
                <div class="intraday-metric-label">Market Breadth</div>
                <div class="intraday-metric-val" style="color: {'#10b981' if pos_pct >= 50 else '#ef4444'};">{pos_pct:.1f}% Up</div>
                <div class="intraday-metric-sub">🟢 {pos_count} Adv &nbsp;|&nbsp; 🔴 {neg_count} Dec</div>
            </div>
            <div class="intraday-metric-card">
                <div class="intraday-metric-label">Universe Average</div>
                <div class="intraday-metric-val" style="color: {'#10b981' if avg_ret >= 0 else '#ef4444'};">{avg_ret:+.2f}%</div>
                <div class="intraday-metric-sub">Across {total_stocks} stocks</div>
            </div>
            <div class="intraday-metric-card">
                <div class="intraday-metric-label">Expansion Radar</div>
                <div class="intraday-metric-val" style="color: #f59e0b;">{highs_count} Highs</div>
                <div class="intraday-metric-sub">New 52-Week Highs Today</div>
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(hud_html), unsafe_allow_html=True)

    # Informational Guide on Regime Badges & Capitulation Sequence
    with st.expander(f"ℹ️ What Do These Mean? (Paul Tudor Jones 200 SMA & Shake Pryzby 7-Step Cycle Explained)", expanded=False):
        c_ptj, c_seq = st.columns([1, 1.4])
        with c_ptj:
            st.markdown(f"""
            #### 🏛️ Paul Tudor Jones 200 SMA Rule:
            > *"My metric for anything I trade is whether it is above or below the 200-day simple moving average. If it is below, you are playing defense."* — Paul Tudor Jones (*Market Wizards*)
            
            - **Current Status**: **{secular_label}** (`{dist_200:+.2f}%` from 200 SMA).
            - **🟢 Secular Bull (Above 200 SMA)**: Structural multi-year bull trend. Institutional liquidity supports growth breakouts; pullbacks to 20/50 EMA are buyable.
            - **🔴 Secular Bear (Below 200 SMA)**: Institutional headwinds. Statistically, >90% of deep market corrections and crashes occur below the 200 SMA. **Action**: Reduce position sizing (e.g. 5% max risk), do not use aggressive long margin, and respect stops instantly.
            
            #### 📐 Kristjan Qullamaggie 10/20 EMA Slopes:
            - **10 EMA Slope ({ema_10_sl:+.2f})**: Short-term 2-5 day momentum vector.
            - **20 EMA Slope ({ema_20_sl:+.2f})**: Intermediate 2-4 week swing trend vector.
            - When **both slopes are positive**, you have a green light to step on the gas. When unaligned or negative, tighten stops and play defense.
            """)
        with c_seq:
            st.markdown(f"""
            #### 📍 Shake Pryzby 7-Step Capitulation & Recovery Sequence (*Pinpoint Trading*):
            Institutional turnaround sequence from selling climax to confirmed bull trend:
            
            1. **Step 1: Capitulation Liquidation** — High-volume panic selling and climactic breakdown.
            2. **Step 2: Turnaround Attempt** — Initial sharp bounce off intraday lows (Daily Close Range >60%). Do not chase.
            3. **Step 3: Volatility Contraction** — Short-term ATR drops below 20-day ATR; panic selling subsides.
            4. **Step 4: Moving Average Clustering** {'👈 **[ACTIVE TODAY]**' if current_step_num == 4 else ''} — 10 & 20 EMA flatten and pinch together (<= 1.5% distance). Volatility coils tightly. **Action: Prepare buy stop orders on top watchlist leaders.**
            5. **Step 5: 20 EMA Recapture (Tactical Buy)** {'👈 **[ACTIVE TODAY]**' if current_step_num == 5 else ''} — Index closes decisively above 20 EMA. **Action: Tactical green light for 2.5%–5.0% pilot risk in leaders.**
            6. **Step 6: 10/20 EMA Crossback** {'👈 **[ACTIVE TODAY]**' if current_step_num == 6 else ''} — Fast 10 EMA crosses back above 20 EMA. Short-term trend flips bullish.
            7. **Step 7: Stress Test Held (Confirmed Trend)** {'👈 **[ACTIVE TODAY]**' if current_step_num == 7 else ''} — First pullback to rising 20 EMA is defended. Full offensive sizing permitted.
            
            **👉 Current Tactical Directive**: `{action_guidance}`
            """)


def standardize_columns(df: pd.DataFrame):
    """Standardizes dataframe columns to lowercase and handles MultiIndex flattening."""
    if df.empty:
        return df
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        found_level = -1
        for i in range(df.columns.nlevels):
            level_vals = [str(c).lower() for c in df.columns.get_level_values(i)]
            if 'close' in level_vals:
                found_level = i
                break
        df.columns = df.columns.get_level_values(found_level)
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df


def detect_20ema_recapture_and_sequence(df: pd.DataFrame) -> dict:
    """
    Paul Tudor Jones 200 SMA rule and Shake Pryzby 7-step bottom progression sequence.
    """
    default_res = {
        'secular_bull': True,
        'dist_200_sma_pct': 0.0,
        'secular_label': "Secular Bull (Above 200 SMA)",
        'secular_color': "#10b981",
        'recapture_active': False,
        'recapture_fresh': False,
        'recapture_sessions_ago': -1,
        'current_step_num': 1,
        'current_step_name': "Base Foundation",
        'step_badge_color': "#94a3b8",
        'sequence_summary': "Step 1/7: Initial Base Construction",
        'action_guidance': "Monitor market internals.",
        'steps_detail': {}
    }
    
    if df is None or df.empty or len(df) < 20:
        return default_res
        
    df_std = standardize_columns(df).copy()
    if 'close' not in df_std.columns:
        return default_res
        
    close = df_std['close'].dropna()
    if len(close) < 20:
        return default_res
        
    # Moving Averages
    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    sma50 = close.rolling(50, min_periods=10).mean()
    sma200 = close.rolling(200, min_periods=25).mean()
    
    c_now = float(close.iloc[-1])
    sma200_now = float(sma200.iloc[-1])
    ema20_now = float(ema20.iloc[-1])
    ema10_now = float(ema10.iloc[-1])
    sma50_now = float(sma50.iloc[-1])
    
    # 1. Secular Regime (PTJ 200 SMA)
    secular_bull = c_now >= sma200_now
    dist_200 = ((c_now - sma200_now) / sma200_now) * 100.0 if sma200_now > 0 else 0.0
    secular_label = "Secular Bull (Above 200 SMA)" if secular_bull else "Secular Bear (Below 200 SMA)"
    secular_color = "#10b981" if secular_bull else "#ef4444"
    
    # 2. Tactical 20 EMA Recapture
    below_mask = close < ema20
    above_mask = close >= ema20
    
    recapture_active = False
    recapture_fresh = False
    recapture_sessions_ago = -1
    
    if c_now >= ema20_now:
        prior_below = below_mask.iloc[-12:-1]
        if prior_below.sum() >= 2:
            for i in range(1, min(6, len(close))):
                if close.iloc[-i] >= ema20.iloc[-i] and close.iloc[-i - 1] < ema20.iloc[-i - 1]:
                    recapture_active = True
                    recapture_sessions_ago = i - 1
                    recapture_fresh = (recapture_sessions_ago <= 1)
                    break
            if not recapture_active and prior_below.iloc[-1]:
                recapture_active = True
                recapture_fresh = True
                recapture_sessions_ago = 0
                
    # 3. 7-Step Bottom Progression
    sub_df = df_std.tail(30).copy()
    sub_close = sub_df['close']
    sub_pct = sub_close.pct_change() * 100.0
    sub_vol = sub_df['volume'] if 'volume' in sub_df.columns else pd.Series(1, index=sub_df.index)
    
    if 'high' in sub_df.columns and 'low' in sub_df.columns:
        high = sub_df['high']
        low = sub_df['low']
        prev_close = sub_close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr_3 = float(tr.tail(3).mean()) if len(tr) >= 3 else 1.0
        atr_20 = float(tr.tail(20).mean()) if len(tr) >= 20 else 1.0
        hl_spread = (high - low).replace(0, np.nan)
        dcr = ((sub_close - low) / hl_spread).fillna(0.5).clip(0.0, 1.0)
    else:
        atr_3 = 1.0
        atr_20 = 1.0
        dcr = pd.Series(0.5, index=sub_df.index)
        
    avg_vol = sub_vol.rolling(20, min_periods=5).mean()
    
    step1_capitulation = bool(((sub_pct <= -1.2) & (sub_vol > avg_vol * 1.25)).any())
    step2_turnaround = bool((dcr.tail(15) >= 0.60).any() and (sub_pct.tail(15) > 0).any())
    step3_contraction = bool(atr_3 < atr_20)
    
    dist_10 = abs((c_now - ema10_now) / ema10_now) * 100.0 if ema10_now > 0 else 99.0
    dist_20 = abs((c_now - ema20_now) / ema20_now) * 100.0 if ema20_now > 0 else 99.0
    step4_clustering = bool(dist_10 <= 1.5 or dist_20 <= 1.5)
    
    step5_recaptured = bool(c_now >= ema20_now)
    step6_golden_cross = bool(ema10_now > ema20_now)
    
    ema20_slope = float((ema20.iloc[-1] - ema20.iloc[-3]) / 2.0) if len(ema20) >= 3 else 0.0
    step7_stress_test = bool(step6_golden_cross and step5_recaptured and ema20_slope > 0)
    
    if step7_stress_test:
        current_step_num = 7
        current_step_name = "Stress Test Held (Confirmed Trend)"
        step_badge_color = "#10b981"
        action_guidance = "Trend confirmed. Full pilot & sizing allowed on leading Stage 1/2 breakouts."
    elif step6_golden_cross:
        current_step_num = 6
        current_step_name = "10/20 EMA Crossback"
        step_badge_color = "#34d399"
        action_guidance = "Short-term momentum turned bullish. Look for high-volume base breakouts."
    elif step5_recaptured:
        current_step_num = 5
        current_step_name = "20 EMA Recapture (Tactical Buy)"
        step_badge_color = "#38bdf8"
        action_guidance = "Tactical supply-demand flip. Deploy initial 2.5%-5% pilot risk in top leaders."
    elif step4_clustering:
        current_step_num = 4
        current_step_name = "Moving Average Clustering"
        step_badge_color = "#f59e0b"
        action_guidance = "Price coiling near MAs. Prepare buy stop orders; wait for 20 EMA reclaim."
    elif step3_contraction:
        current_step_num = 3
        current_step_name = "Volatility Contraction"
        step_badge_color = "#fbbf24"
        action_guidance = "Range contracting. Watch for volume dry-up and subsequent expansion."
    elif step2_turnaround:
        current_step_num = 2
        current_step_name = "Turnaround Attempt"
        step_badge_color = "#fb923c"
        action_guidance = "First bounce off lows. Do not chase; wait for constructive digestion."
    elif step1_capitulation:
        current_step_num = 1
        current_step_name = "Capitulation Liquidation"
        step_badge_color = "#ef4444"
        action_guidance = "Selling climax in progress. Maintain 100% defense; observe potential support."
    else:
        current_step_num = 5 if step5_recaptured else 4
        current_step_name = "20 EMA Recaptured" if step5_recaptured else "Consolidating"
        step_badge_color = "#38bdf8" if step5_recaptured else "#94a3b8"
        action_guidance = "Monitor market posture."
        
    return {
        'secular_bull': secular_bull,
        'dist_200_sma_pct': round(dist_200, 2),
        'secular_label': secular_label,
        'secular_color': secular_color,
        'recapture_active': recapture_active,
        'recapture_fresh': recapture_fresh,
        'recapture_sessions_ago': recapture_sessions_ago,
        'current_step_num': current_step_num,
        'current_step_name': current_step_name,
        'step_badge_color': step_badge_color,
        'sequence_summary': f"Step {current_step_num}/7: {current_step_name}",
        'action_guidance': action_guidance,
        'steps_detail': {
            '1_capitulation': step1_capitulation,
            '2_turnaround': step2_turnaround,
            '3_contraction': step3_contraction,
            '4_clustering': step4_clustering,
            '5_recaptured': step5_recaptured,
            '6_golden_cross': step6_golden_cross,
            '7_stress_test': step7_stress_test
        }
    }
