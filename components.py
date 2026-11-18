from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go

def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Applies a professional, dark-mode transparent theme to Plotly figures."""
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)", 
            zerolinecolor="rgba(255,255,255,0.1)",
            color="#94a3b8"
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)", 
            zerolinecolor="rgba(255,255,255,0.1)",
            color="#94a3b8"
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_size=13,
            font_family="Inter, sans-serif"
        )
    )
    return fig

def render_header(title: str, subtitle: str, icon: str = ""):
    """Renders a standardized, universally styled modern app header."""
    
    # If standard streamlit emoji icon is provided, merge it into title
    display_title = f"{icon} {title}" if icon else title
    
    st.markdown(f"""
    <div class="st-app-header">
        <h1>{display_title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
    
def render_metric_card(label: str, value: str, color_class: str = "", extra_style: str = "", subtext: str = "", **kwargs):
    """Renders a standardized metric card."""
    # color_class e.g., 'green-text', 'red-text', 'yellow-text'
    st_text = subtext or kwargs.get('subtitle', '') or kwargs.get('sub_label', '') or kwargs.get('help', '')
    subtext_html = f'<div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">{st_text}</div>' if st_text else ''
    st.markdown(f"""
    <div class="metric-card" style="{extra_style}">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
        {subtext_html}
    </div>
    """, unsafe_allow_html=True)

def render_score_card(score: str, label: str, state: str = "neutral"):
    """
    Renders score cards (Ready, Developing, Error, Neutral). 
    state must be one of: 'ready', 'developing', 'notready', 'error', 'neutral'
    """
    state_classes = {
        'ready': 'score-ready',
        'developing': 'score-developing',
        'notready': 'score-notready',
        'error': 'score-error',
        'neutral': 'score-neutral'
    }
    
    cls = state_classes.get(state, 'score-neutral')
    
    st.markdown(f"""
    <div class="score-card {cls}">
        <h2>{score}</h2>
        <p>{label}</p>
    </div>
    """, unsafe_allow_html=True)

def render_empty_state(message: str, icon: str = "📭"):
    """Renders a beautiful, premium empty state for when no stocks meet criteria."""
    st.markdown(f"""
    <div style="
        display: flex; 
        flex-direction: column; 
        align-items: center; 
        justify-content: center; 
        padding: 3rem 2rem; 
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px dashed rgba(100, 116, 139, 0.5);
        border-radius: 16px;
        margin: 1rem 0;
        text-align: center;
    ">
        <span style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.8;">{icon}</span>
        <h3 style="color: #cbd5e1; font-family: 'Inter', sans-serif; font-weight: 500; font-size: 1.1rem; margin: 0;">No Setups Found</h3>
        <p style="color: #94a3b8; font-family: 'Inter', sans-serif; font-size: 0.9rem; margin-top: 0.5rem; max-width: 400px;">
            {message}
        </p>
    </div>
    """, unsafe_allow_html=True)

def _refresh_matrix_after_price_sync():
    """Rebuild Industry Group Matrix JSONs in local fast mode (<5s) after EOD PKL updates."""
    try:
        from industry_group_engine import refresh_industry_group_intelligence_cache
        refresh_industry_group_intelligence_cache(universe_scope="India (NSE/BSE)", force=False, skip_enrichment=True)
    except Exception:
        pass


def render_disk_cache_sidebar(universe_fetcher_func=None):
    """Renders the local persistent NVMe cache status in the Streamlit Sidebar."""
    from price_history_manager import get_cache_status, fetch_incremental_history
    import streamlit as st
    
    with st.sidebar:
        st.markdown("---")
        tickers_count, max_date = get_cache_status()
        
        st.markdown("<h3 style='color: #f8fafc; margin-bottom: 0px;'>🗄️ Master Price Cache</h3>", unsafe_allow_html=True)
        if tickers_count > 0:
            st.markdown(f"<p style='color: #cbd5e1; font-size: 0.85em; margin-bottom: 2px;'><b>Persistent Universe:</b> {tickers_count} tickers</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #cbd5e1; font-size: 0.85em; margin-bottom: 15px;'><b>EOD Sync Date:</b> {max_date.strftime('%Y-%m-%d') if max_date else 'N/A'}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #cbd5e1; font-size: 0.85em; margin-bottom: 15px;'>Cache Empty. Run an Apex Scan to initialize.</p>", unsafe_allow_html=True)
            
        if universe_fetcher_func is not None:
            if st.button("🔄 Force EOD Delta Sync", use_container_width=True):
                tickers = universe_fetcher_func("Deep Market (2500+ NSE Stocks)")
                # Show an interactive progress bar for the background sync
                progress_text = "Incrementally fetching missing days... Please wait."
                my_bar = st.progress(0.0, text=progress_text)
                
                def update_progress(msg):
                    my_bar.progress(0.5, text=msg)

                fetch_incremental_history(tickers, days=252, progress_callback=update_progress)
                
                my_bar.progress(0.95, text="Refreshing Industry Group Matrix cache...")
                _refresh_matrix_after_price_sync()
                my_bar.progress(1.0, text="Sync Complete!")
                st.success("Universal EOD Sync Complete!")
                st.rerun()
                
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🧹 Scrub & Rebuild Database", type="secondary", use_container_width=True, help="Wipe cache and fetch 252-days backwards to safely adjust for Stock Splits."):
                tickers = universe_fetcher_func("Deep Market (2500+ NSE Stocks)")
                progress_text = "Destroying DB & Downloading Pristine History..."
                my_bar = st.progress(0.0, text=progress_text)
                
                def update_progress(msg):
                    my_bar.progress(0.5, text=msg)

                from price_history_manager import rebuild_full_history
                rebuild_full_history(tickers, days=252, progress_callback=update_progress)
                
                my_bar.progress(0.95, text="Refreshing Industry Group Matrix cache...")
                _refresh_matrix_after_price_sync()
                my_bar.progress(1.0, text="Rebuild Complete!")
                st.success("Database scrubbed & matrix perfectly rebuilt!")
                st.rerun()


def render_market_telemetry_sidebar():
    """Renders a live institutional market session telemetry widget in the sidebar."""
    from zoneinfo import ZoneInfo
    from datetime import datetime, time
    
    try:
        ist_now = datetime.now(ZoneInfo('Asia/Kolkata'))
        est_now = datetime.now(ZoneInfo('America/New_York'))
        
        # NSE session calculation
        if ist_now.weekday() < 5:
            t = ist_now.time()
            if time(9, 15) <= t <= time(15, 30):
                nse_label, nse_bg, nse_color = "ACTIVE", "rgba(16,185,129,0.18)", "#34d399"
            elif time(9, 0) <= t < time(9, 15):
                nse_label, nse_bg, nse_color = "PRE-MKT", "rgba(245,158,11,0.18)", "#fbbf24"
            else:
                nse_label, nse_bg, nse_color = "CLOSED", "rgba(100,116,139,0.2)", "#94a3b8"
        else:
            nse_label, nse_bg, nse_color = "WEEKEND", "rgba(100,116,139,0.2)", "#94a3b8"
            
        # US session calculation
        if est_now.weekday() < 5:
            t = est_now.time()
            if time(9, 30) <= t <= time(16, 0):
                us_label, us_bg, us_color = "OPEN", "rgba(16,185,129,0.18)", "#34d399"
            elif time(4, 0) <= t < time(9, 30):
                us_label, us_bg, us_color = "PRE-MKT", "rgba(245,158,11,0.18)", "#fbbf24"
            elif time(16, 0) < t <= time(20, 0):
                us_label, us_bg, us_color = "POST-MKT", "rgba(249,115,22,0.18)", "#fb923c"
            else:
                us_label, us_bg, us_color = "CLOSED", "rgba(100,116,139,0.2)", "#94a3b8"
        else:
            us_label, us_bg, us_color = "WEEKEND", "rgba(100,116,139,0.2)", "#94a3b8"

        st.sidebar.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(15,23,42,0.85) 0%, rgba(2,6,23,0.98) 100%);
                    border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 10px 12px; margin-bottom: 12px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.73rem; font-weight: 700; color: #94a3b8; letter-spacing: 0.8px;">
                    🇮🇳 NSE (IST)
                </span>
                <span style="background: {nse_bg}; color: {nse_color}; border: 1px solid {nse_color}55;
                             padding: 2px 7px; border-radius: 9999px; font-size: 0.67rem; font-weight: 800; letter-spacing: 0.5px;">
                    ● {nse_label}
                </span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 0.73rem; font-weight: 700; color: #94a3b8; letter-spacing: 0.8px;">
                    🇺🇸 US (EST)
                </span>
                <span style="background: {us_bg}; color: {us_color}; border: 1px solid {us_color}55;
                             padding: 2px 7px; border-radius: 9999px; font-size: 0.67rem; font-weight: 800; letter-spacing: 0.5px;">
                    ● {us_label}
                </span>
            </div>
            <div style="border-top: 1px solid rgba(255,255,255,0.06); padding-top: 6px; display: flex; justify-content: space-between; font-size: 0.68rem; color: #64748b; font-family: monospace;">
                <span>IST {ist_now.strftime('%H:%M')}</span>
                <span>EDT {est_now.strftime('%H:%M')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass



def render_model_book_badge_legend(expanded: bool = False):
    """
    Renders a sleek, educational Badge Legend on top of setup and leader tables.
    Explains the model book setup meaning, exact entry trigger, and stop-loss placement.
    """
    import streamlit as st
    with st.expander("📖 Model Book Tactical Badge Guide & Execution Rules", expanded=expanded):
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; font-size: 0.82rem; line-height: 1.5;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px;">
                <div style="background: rgba(254, 240, 138, 0.08); border-left: 3px solid #eab308; padding: 8px 12px; border-radius: 4px;">
                    <b style="color: #fde047;">⚡ Oops Reversal</b> (Larry Williams / TraderLion)
                    <br><span style="color: #cbd5e1;">Price undercut yesterday's low (shaking out stops), then reversed & closed back above it.</span>
                    <br><b style="color: #94a3b8;">Trigger:</b> Reclaim of prior low | <b style="color: #ef4444;">Stop:</b> Low of the Day (LOD).
                </div>
                <div style="background: rgba(103, 232, 249, 0.08); border-left: 3px solid #06b6d4; padding: 8px 12px; border-radius: 4px;">
                    <b style="color: #67e8f9;">🛡️ Pos Expectation Breaker (PEB)</b> (TraderLion)
                    <br><span style="color: #cbd5e1;">Yesterday was a sharp selloff (Extension Down). Today immediately breaks downward momentum by surging green.</span>
                    <br><b style="color: #94a3b8;">Trigger:</b> Green close at support | <b style="color: #ef4444;">Stop:</b> Low of reversal bar.
                </div>
                <div style="background: rgba(192, 132, 252, 0.08); border-left: 3px solid #a855f7; padding: 8px 12px; border-radius: 4px;">
                    <b style="color: #d8b4fe;">🎯 Primed (Inside Day / Double Inside)</b> (TraderLion)
                    <br><span style="color: #cbd5e1;">Extreme volatility coil right at pivot with Volume Dry-Up (≤75%). Double inside (🎯🎯) indicates maximum energy.</span>
                    <br><b style="color: #94a3b8;">Trigger:</b> Break of inside high | <b style="color: #ef4444;">Stop:</b> Inside low.
                </div>
                <div style="background: rgba(52, 211, 153, 0.08); border-left: 3px solid #10b981; padding: 8px 12px; border-radius: 4px;">
                    <b style="color: #6ee7b7;">💎 Ross Haber 65-EMA Floor</b> (William O'Neil PM Rule)
                    <br><span style="color: #cbd5e1;">Testing & holding 13-week institutional line. Haber Containment: 23-EMA contained by 50-SMA; 50-SMA contained by 65-EMA.</span>
                    <br><b style="color: #ef4444;">Institutional Exit (🚨):</b> High-volume close below 65-EMA.
                </div>
                <div style="background: rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; padding: 8px 12px; border-radius: 4px;">
                    <b style="color: #38bdf8;">🏖️ Beach Ball (Nx)</b> (Institutional Accumulation Under Water)
                    <br><span style="color: #cbd5e1;">Stock surged +4% on heavy volume (≥1.0x 20D ADTV) while Market Breadth 50-SMA is submerged &lt;35%. <i>N</i> = power plays in last 3 months.</span>
                    <br><b style="color: #94a3b8;">Significance:</b> Coiled spring ready to launch the moment broad market selling exhausts.
                </div>
                <div style="background: rgba(244, 114, 182, 0.08); border-left: 3px solid #ec4899; padding: 8px 12px; border-radius: 4px; grid-column: 1 / -1;">
                    <b style="color: #f472b6;">👑 True Leader vs 🚀 Performance Enhancer</b>
                    <span style="color: #cbd5e1;"><b>True Market Leaders</b> (pristine CANSLIM earnings) trail 21-EMA / 50-SMA for multi-quarter compounding. <b>Performance Enhancers</b> (high-beta / story runners) must trail strictly along the <b>10-EMA</b> to lock in gains before the bust phase.</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_tv_export_popover(
    tv_symbols: list[str],
    *,
    title: str,
    key: str,
    btn_label: str | None = None,
) -> None:
    """Render an inline TradingView watchlist copy popover with an expander fallback."""
    tv_copy = ", ".join(tv_symbols) if tv_symbols else ""
    label = btn_label or f"📋 Export TV ({len(tv_symbols)})"
    if hasattr(st, "popover"):
        with st.popover(label, use_container_width=True):
            st.caption(title)
            st.code(tv_copy or "No tickers", language="text")
            st.text_area(
                "Copy String (Ctrl+A, Ctrl+C):",
                value=tv_copy,
                height=75,
                key=key,
            )
    else:
        with st.expander(label, expanded=False):
            st.caption(title)
            st.code(tv_copy or "No tickers", language="text")
            st.text_area(
                "TradingView Watchlist String (Click box, Ctrl+A, Ctrl+C):",
                value=tv_copy,
                height=75,
                key=key,
            )


def render_compact_workbench_banner(
    *,
    icon: str,
    title: str,
    badge_text: str = "",
    subtitle: str = "",
    right_pill_text: str = "",
    accent_color: str = "#38bdf8",
) -> None:
    """Render a compact 1-line institutional workbench header banner without vertical bloat."""
    import html as _html

    badge_html = (
        f'<span style="background: rgba(56, 189, 248, 0.14); color: {accent_color}; '
        f'border: 1px solid rgba(56, 189, 248, 0.32); padding: 2px 9px; border-radius: 999px; '
        f'font-size: 0.68rem; font-weight: 800; font-family: \'JetBrains Mono\', monospace;">'
        f'{_html.escape(badge_text)}</span>'
        if badge_text
        else ""
    )
    sub_html = (
        f'<span style="color: #94a3b8; font-size: 0.74rem;">{_html.escape(subtitle)}</span>'
        if subtitle
        else ""
    )
    right_html = (
        f'<span style="background: rgba(16, 185, 129, 0.14); color: #34d399; '
        f'border: 1px solid rgba(16, 185, 129, 0.38); padding: 3px 11px; border-radius: 7px; '
        f'font-size: 0.76rem; font-weight: 800; font-family: \'JetBrains Mono\', monospace; white-space: nowrap;">'
        f'{_html.escape(right_pill_text)}</span>'
        if right_pill_text
        else ""
    )
    st.markdown(
        f'<div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(23, 37, 84, 0.88)); '
        f'border: 1px solid rgba(56, 189, 248, 0.30); border-left: 5px solid {accent_color}; '
        f'border-radius: 10px; padding: 10px 16px; margin-top: 4px; margin-bottom: 10px; '
        f'display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">'
        f'<div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">'
        f'<span style="font-size: 1.15rem;">{icon}</span>'
        f'<span style="color: #f8fafc; font-size: 1.02rem; font-weight: 800; letter-spacing: 0.2px;">{_html.escape(title)}</span>'
        f'{badge_html}{sub_html}'
        f'</div>{right_html}</div>',
        unsafe_allow_html=True,
    )


def summarize_portfolio_action_counts(decisions: list[dict]) -> dict[str, int]:
    """Summarize ADD / TRIM / EXIT / HOLD portfolio decision counts deterministically."""
    counts = {"ADD": 0, "TRIM": 0, "EXIT": 0, "HOLD": 0, "TOTAL": 0}
    for d in decisions or []:
        if not isinstance(d, dict):
            continue
        raw = str(d.get("action") or d.get("Action") or "").upper()
        counts["TOTAL"] += 1
        if "EXIT" in raw or "SELL" in raw:
            counts["EXIT"] += 1
        elif "TRIM" in raw:
            counts["TRIM"] += 1
        elif "ADD" in raw:
            counts["ADD"] += 1
        elif "HOLD" in raw:
            counts["HOLD"] += 1
        else:
            counts["HOLD"] += 1
    return counts


def build_earnings_calendar_grid(cal_rows: list[dict]) -> list[dict]:
    """Group upcoming earnings rows chronologically by date with day-name and urgency metadata."""
    import datetime as _dt

    buckets: dict[str, list[dict]] = {}
    for row in cal_rows or []:
        if not isinstance(row, dict):
            continue
        dt_str = str(row.get("Upcoming Date") or "TBD").strip()
        day_key = dt_str[:10] if len(dt_str) >= 10 else dt_str
        buckets.setdefault(day_key, []).append(row)

    ordered: list[dict] = []
    for day_key in sorted(buckets.keys()):
        try:
            parsed = _dt.date.fromisoformat(day_key)
            day_label = parsed.strftime("%a %d %b")
            sort_ord = parsed.toordinal()
        except Exception:
            day_label = day_key
            sort_ord = 9_999_999
        items = buckets[day_key]
        imminent = any("IMMINENT" in str(it.get("Status", "")).upper() for it in items)
        ordered.append(
            {
                "date": day_key,
                "label": day_label,
                "sort_ord": sort_ord,
                "imminent": imminent,
                "items": items,
            }
        )
    ordered.sort(key=lambda x: (x["sort_ord"], x["date"]))
    return ordered


def render_earnings_calendar_grid(cal_rows: list[dict]) -> None:
    """Render a compact Mon-Fri CSS grid agenda of upcoming portfolio earnings above the dataframe."""
    import html as _html

    days = build_earnings_calendar_grid(cal_rows)
    if not days:
        return
    cards_html = []
    for day in days[:10]:
        border_col = "rgba(245, 158, 11, 0.55)" if day["imminent"] else "rgba(56, 189, 248, 0.25)"
        hdr_col = "#fbbf24" if day["imminent"] else "#38bdf8"
        badges = []
        for it in day["items"][:6]:
            tkr = _html.escape(str(it.get("Ticker", "")))
            d_left = _html.escape(str(it.get("Days Remaining", "")))
            is_imm = "IMMINENT" in str(it.get("Status", "")).upper()
            bg = "rgba(245, 158, 11, 0.16)" if is_imm else "rgba(30, 41, 59, 0.85)"
            fg = "#fcd34d" if is_imm else "#e2e8f0"
            badges.append(
                f'<div style="background:{bg}; color:{fg}; border-radius:5px; padding:3px 7px; '
                f'font-size:0.72rem; font-weight:700; display:flex; justify-content:space-between; gap:6px; margin-top:4px;">'
                f'<span>{tkr}</span><span style="opacity:0.8; font-weight:600;">{d_left}</span></div>'
            )
        extra = len(day["items"]) - 6
        if extra > 0:
            badges.append(f'<div style="font-size:0.68rem; color:#94a3b8; margin-top:4px;">+{extra} more</div>')
        cards_html.append(
            f'<div style="background:rgba(15, 23, 42, 0.78); border:1px solid {border_col}; '
            f'border-radius:8px; padding:8px 10px; min-width:0;">'
            f'<div style="color:{hdr_col}; font-size:0.74rem; font-weight:800; letter-spacing:0.3px; '
            f'border-bottom:1px solid rgba(148,163,184,0.14); padding-bottom:4px; margin-bottom:4px;">'
            f'{_html.escape(day["label"])} ({len(day["items"])})</div>'
            f'{"".join(badges)}</div>'
        )
    st.markdown(
        f'<div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(175px, 1fr)); '
        f'gap:8px; margin-bottom:12px;">{"".join(cards_html)}</div>',
        unsafe_allow_html=True,
    )


def _format_rank_delta(rank_val, delta_val) -> str:
    """Format compact rank + superscript-style delta string (e.g. '1 (+20)')."""
    import pandas as _pd

    if _pd.isna(rank_val):
        return "—"
    r_int = int(round(float(rank_val)))
    if _pd.isna(delta_val) or int(round(float(delta_val))) == 0:
        return f"{r_int}"
    d_int = int(round(float(delta_val)))
    return f"{r_int} ({d_int:+d})"


def build_compact_matrix_view(df):
    """Project an Industry Group Matrix DataFrame to a compact DeepVue-style view without mutating input."""
    import pandas as _pd

    if df is None or not isinstance(df, _pd.DataFrame) or df.empty:
        return _pd.DataFrame()
    out = df.copy()
    if "Rank_1W" in out.columns and "Delta_1W" in out.columns:
        out["Rank_1W_Compact"] = [
            _format_rank_delta(r, d) for r, d in zip(out["Rank_1W"], out["Delta_1W"])
        ]
    elif "Rank_1W" in out.columns:
        out["Rank_1W_Compact"] = [_format_rank_delta(r, None) for r in out["Rank_1W"]]

    if "Rank_1M" in out.columns and "Delta_1M" in out.columns:
        out["Rank_1M_Compact"] = [
            _format_rank_delta(r, d) for r, d in zip(out["Rank_1M"], out["Delta_1M"])
        ]
    elif "Rank_1M" in out.columns:
        out["Rank_1M_Compact"] = [_format_rank_delta(r, None) for r in out["Rank_1M"]]

    compact_cols = [
        "Rank_Today",
        "Group_Grade_Badge",
        "Industry_Group",
        "Rank_1W_Compact",
        "Rank_1M_Compact",
        "Rank_3M",
        "Blended_Score",
        "Breadth_RS80_Pct",
        "Dual_Badge",
        "Group_Confirmation_Badge",
        "Top_Leaders_Display",
        "Sparkline_1M",
    ]
    present = [c for c in compact_cols if c in out.columns]
    return out[present].copy()


def build_constituent_diverging_bar_fig(
    df_constits,
    *,
    metric_col: str = "1D %",
    max_bars: int = 20,
) -> go.Figure:
    """Build a horizontal diverging bar chart comparing group constituents on 1D% or 1M%."""
    import pandas as _pd

    fig = go.Figure()
    if (
        df_constits is None
        or not isinstance(df_constits, _pd.DataFrame)
        or df_constits.empty
        or "Symbol" not in df_constits.columns
        or metric_col not in df_constits.columns
    ):
        return fig

    work = df_constits[["Symbol", metric_col]].copy()
    work[metric_col] = _pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=[metric_col])
    if work.empty:
        return fig

    work = work.sort_values(metric_col, ascending=False)
    if len(work) > max_bars and max_bars >= 4:
        half = max_bars // 2
        tail_n = max_bars - half
        work = _pd.concat([work.head(half), work.tail(tail_n)]).drop_duplicates(subset=["Symbol"])
        work = work.sort_values(metric_col, ascending=False)

    vals = work[metric_col].tolist()
    syms = work["Symbol"].astype(str).tolist()
    colors = ["#10b981" if v >= 0 else "#f43f5e" for v in vals]
    texts = [f"{v:+.2f}%" for v in vals]

    fig.add_trace(
        go.Bar(
            x=vals,
            y=syms,
            orientation="h",
            marker_color=colors,
            text=texts,
            textposition="outside",
            textfont=dict(size=11, color="#e2e8f0"),
            hovertemplate="<b>%{y}</b>: %{x:+.2f}%<extra></extra>",
            showlegend=False,
        )
    )
    apply_plotly_theme(fig)
    chart_h = max(250, min(520, len(work) * 24 + 65))
    fig.update_layout(
        height=chart_h,
        margin=dict(l=12, r=55, t=24, b=16),
        xaxis=dict(
            zeroline=True,
            zerolinecolor="rgba(148, 163, 184, 0.45)",
            zerolinewidth=1.4,
            title=None,
            ticksuffix="%",
        ),
        yaxis=dict(autorange="reversed", title=None),
    )
    return fig


def compute_focus_trade_metrics(
    live_price: float | None,
    entry_trigger: float | None,
    stop_loss: float | None,
) -> dict[str, float | None]:
    """Compute Dist-to-Pivot %, Stop Risk %, and 2R/3R price targets for Focus List trade plans."""
    lp = float(live_price or 0.0)
    et = float(entry_trigger or 0.0)
    sl = float(stop_loss or 0.0)

    dist_to_pivot = ((et - lp) / lp) * 100.0 if lp > 0.0 and et > 0.0 else None
    valid_risk = et > 0.0 and sl > 0.0 and et > sl
    risk_pct = ((et - sl) / et) * 100.0 if valid_risk else None
    risk_pts = (et - sl) if valid_risk else 0.0
    r2_target = (et + 2.0 * risk_pts) if valid_risk else None
    r3_target = (et + 3.0 * risk_pts) if valid_risk else None
    return {
        "dist_to_pivot_pct": round(dist_to_pivot, 2) if dist_to_pivot is not None else None,
        "risk_pct": round(risk_pct, 2) if risk_pct is not None else None,
        "r2_target": round(r2_target, 2) if r2_target is not None else None,
        "r3_target": round(r3_target, 2) if r3_target is not None else None,
    }


def compute_stage_distribution_from_close_df(close_df) -> dict[str, float | int]:
    """Compute Stage 1-4 stock counts and percentages on the latest settled bar."""
    import pandas as _pd

    empty = {
        "s1_count": 0,
        "s2_count": 0,
        "s3_count": 0,
        "s4_count": 0,
        "total": 0,
        "s1_pct": 0.0,
        "s2_pct": 0.0,
        "s3_pct": 0.0,
        "s4_pct": 0.0,
    }
    if close_df is None or not isinstance(close_df, _pd.DataFrame) or close_df.empty or len(close_df) < 25:
        return empty

    sma50 = close_df.rolling(50, min_periods=20).mean()
    sma200 = close_df.rolling(200, min_periods=50).mean()
    slope_pos = sma200 > sma200.shift(20)
    slope_neg = sma200 < sma200.shift(20)

    c_last = close_df.iloc[-1]
    s50_last = sma50.iloc[-1]
    s200_last = sma200.iloc[-1]
    valid = c_last.notna() & s50_last.notna() & s200_last.notna()
    if not bool(valid.any()):
        return empty

    c_v = c_last[valid]
    s50_v = s50_last[valid]
    s200_v = s200_last[valid]
    sp_pos = slope_pos.iloc[-1][valid].fillna(False)
    sp_neg = slope_neg.iloc[-1][valid].fillna(False)

    s2_m = (c_v > s50_v) & (s50_v > s200_v) & sp_pos
    s4_m = (c_v < s50_v) & (s50_v < s200_v) & sp_neg
    s3_m = (~s2_m) & (~s4_m) & (s50_v > s200_v)
    s1_m = (~s2_m) & (~s4_m) & (s50_v <= s200_v)

    s1_n, s2_n, s3_n, s4_n = int(s1_m.sum()), int(s2_m.sum()), int(s3_m.sum()), int(s4_m.sum())
    tot = s1_n + s2_n + s3_n + s4_n
    if tot <= 0:
        return empty

    return {
        "s1_count": s1_n,
        "s2_count": s2_n,
        "s3_count": s3_n,
        "s4_count": s4_n,
        "total": tot,
        "s1_pct": round(s1_n * 100.0 / tot, 1),
        "s2_pct": round(s2_n * 100.0 / tot, 1),
        "s3_pct": round(s3_n * 100.0 / tot, 1),
        "s4_pct": round(s4_n * 100.0 / tot, 1),
    }


def build_stage_distribution_bar_html(
    s1_count: int,
    s2_count: int,
    s3_count: int,
    s4_count: int,
    *,
    label: str = "Stage 1–4 Health",
) -> str:
    """Build a compact single-line horizontal segmented Stage 1-4 health bar HUD."""
    import html as _html

    counts = [max(0, int(s1_count or 0)), max(0, int(s2_count or 0)), max(0, int(s3_count or 0)), max(0, int(s4_count or 0))]
    total = sum(counts)
    if total <= 0:
        return ""

    stages = [
        ("S1", "Basing", counts[0], "#6366f1"),
        ("S2", "Advancing", counts[1], "#10b981"),
        ("S3", "Topping", counts[2], "#f59e0b"),
        ("S4", "Declining", counts[3], "#ef4444"),
    ]
    pcts = [c * 100.0 / total for c in counts]
    s1_p, s2_p, s3_p, s4_p = pcts
    if s4_p >= 30.0:
        verdict, v_color = "Stage 4 Bear Defense", "#ef4444"
    elif s2_p >= 40.0 and s4_p < 15.0:
        verdict, v_color = "Stage 2 Bull Markup", "#10b981"
    elif s3_p >= 35.0:
        verdict, v_color = "Stage 3 Distribution Caution", "#f59e0b"
    elif s1_p >= 35.0:
        verdict, v_color = "Stage 1 Basing Accumulation", "#818cf8"
    else:
        verdict, v_color = "Mixed Rotational", "#38bdf8"

    bar_segments = "".join(
        f'<div title="{code} ({name}): {cnt:,} stocks ({pct:.1f}%)" '
        f'style="width: {pct:.2f}%; background: {col}; height: 100%;"></div>'
        for (code, name, cnt, col), pct in zip(stages, pcts)
        if pct > 0.0
    )
    legend_chips = "".join(
        f'<span style="display: inline-flex; align-items: center; gap: 5px; font-size: 0.72rem; color: #cbd5e1; font-family: \'JetBrains Mono\', monospace;">'
        f'<span style="width: 8px; height: 8px; border-radius: 50%; background: {col}; display: inline-block;"></span>'
        f'<b style="color: #f8fafc;">{code}</b> {cnt:,} ({pct:.1f}%)'
        f"</span>"
        for (code, _name, cnt, col), pct in zip(stages, pcts)
    )
    return (
        f'<div style="background: rgba(15, 23, 42, 0.88); border: 1px solid rgba(255, 255, 255, 0.08); '
        f'border-radius: 10px; padding: 10px 14px; margin: 6px 0 12px 0;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 6px;">'
        f'<span style="font-size: 0.74rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.7px;">'
        f'{_html.escape(label)} · {total:,} Stocks</span>'
        f'<span style="font-size: 0.74rem; font-weight: 800; color: {v_color};">{_html.escape(verdict)}</span>'
        f"</div>"
        f'<div style="display: flex; width: 100%; height: 10px; border-radius: 999px; overflow: hidden; background: rgba(255,255,255,0.06);">'
        f"{bar_segments}</div>"
        f'<div style="display: flex; flex-wrap: wrap; gap: 14px; margin-top: 7px;">{legend_chips}</div>'
        f"</div>"
    )


def _stage_matrix_path_for_market(market: str) -> str:
    import os

    mkt = str(market or "IN").upper().strip()
    if mkt == "US":
        if os.path.exists("us_prices_matrix.pkl"):
            return "us_prices_matrix.pkl"
        if os.path.exists("breadth_matrix.pkl"):
            return "breadth_matrix.pkl"
        return "historical_prices_matrix.pkl"
    if os.path.exists("historical_prices_matrix.pkl"):
        return "historical_prices_matrix.pkl"
    return "breadth_matrix.pkl"


def _stage_allowed_tickers_for_market(market: str) -> list[str] | None:
    mkt = str(market or "IN").upper().strip()
    if mkt == "US":
        try:
            from us_universe_manager import get_sp500_constituents
            sp500 = get_sp500_constituents()
            if sp500:
                return list(sp500)
        except Exception:
            pass
        return None
    try:
        from market_data import fetch_nifty_total_market_tickers
        ntm = fetch_nifty_total_market_tickers(show_progress=False)
        if ntm:
            return list(ntm)
    except Exception:
        pass
    try:
        from universe_security_master import load_universe_tickers
        uni = load_universe_tickers("IN", include_etfs=False)
        if uni:
            return list(uni)
    except Exception:
        pass
    try:
        import os
        if os.path.exists("tickers.txt"):
            with open("tickers.txt", "r", encoding="utf-8") as _f:
                file_tickers = [
                    line.strip()
                    for line in _f
                    if line.strip() and not line.strip().startswith("#")
                ]
            if file_tickers:
                return file_tickers
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def _load_cached_stage_distribution_inner(market: str, cache_sig: tuple) -> dict[str, float | int]:
    import os
    try:
        from price_history_manager import extract_market_ohlcv_panels, load_cached_price_matrix
    except Exception:
        return compute_stage_distribution_from_close_df(None)

    mkt = str(market or "IN").upper().strip()
    matrix_path = _stage_matrix_path_for_market(mkt)
    if not os.path.exists(matrix_path):
        return compute_stage_distribution_from_close_df(None)

    df = load_cached_price_matrix(matrix_path, cache_sig=cache_sig)
    if df is None or df.empty:
        return compute_stage_distribution_from_close_df(None)

    allowed = _stage_allowed_tickers_for_market(mkt)
    panels = extract_market_ohlcv_panels(
        df,
        market="US" if mkt == "US" else "IN",
        allowed_tickers=allowed,
        include_etfs=False,
        deduplicate_dual_in=(mkt != "US"),
        min_coverage_ratio=0.35,
        min_abs=20,
    )
    return compute_stage_distribution_from_close_df(panels.get("close"))


def load_cached_stage_distribution_summary(market: str = "IN") -> dict[str, float | int]:
    """Read local cached price matrix without network calls to summarize Stage 1-4 health."""
    mkt = str(market or "IN").upper().strip()
    try:
        from price_history_manager import get_matrix_cache_signature
        matrix_path = _stage_matrix_path_for_market(mkt)
        sig = get_matrix_cache_signature(matrix_path)
        res = _load_cached_stage_distribution_inner(mkt, sig)
        if res and int(res.get("total", 0) or 0) > 0:
            return res
    except Exception:
        pass
    try:
        from database import load_cloud_json_cache_cached
        key = "breadth_aux_bundle_us" if mkt == "US" else "breadth_aux_bundle_in"
        bundle, _ = load_cloud_json_cache_cached(key)
        if (
            isinstance(bundle, dict)
            and isinstance(bundle.get("stage_dist"), dict)
            and int(bundle["stage_dist"].get("total", 0) or 0) > 0
        ):
            return bundle["stage_dist"]
    except Exception:
        pass
    return compute_stage_distribution_from_close_df(None)


def load_local_benchmark_df(market: str = "IN"):
    """Load local benchmark OHLCV/Close series from disk cache without hitting network."""
    import os
    import pandas as _pd
    try:
        from price_history_manager import get_matrix_cache_signature, load_cached_price_matrix
    except Exception:
        return None

    mkt = str(market or "IN").upper().strip()
    sym_candidates = ("^CRSLDX", "^NSEI") if mkt != "US" else ("^IXIC", "^GSPC", "SPY", "QQQ")
    matrix_candidates = (
        ["historical_prices_matrix.pkl", "breadth_matrix.pkl"]
        if mkt != "US"
        else ["us_prices_matrix.pkl", "breadth_matrix.pkl", "historical_prices_matrix.pkl"]
    )
    for mp in matrix_candidates:
        if not os.path.exists(mp):
            continue
        try:
            mat = load_cached_price_matrix(mp, cache_sig=get_matrix_cache_signature(mp))
            if mat is None or mat.empty or not isinstance(mat.columns, _pd.MultiIndex):
                continue
            lvl0 = set(mat.columns.get_level_values(0))
            for sym in sym_candidates:
                if sym in lvl0:
                    sub = mat[sym].dropna(how="all").copy()
                    if "Close" in sub.columns:
                        sub["Close"] = _pd.to_numeric(sub["Close"], errors="coerce")
                        sub = sub.dropna(subset=["Close"]).tail(180)
                        if not sub.empty:
                            if getattr(sub.index, "tz", None) is not None:
                                sub.index = sub.index.tz_localize(None)
                            return sub
        except Exception:
            continue
    return None


def build_synced_breadth_stack_fig(
    hist_df,
    bench_df=None,
    *,
    bench_label: str = "Benchmark",
    height: int = 540,
) -> go.Figure:
    """Build a synced vertical breadth stack with unified x-axis crosshair across panes."""
    import pandas as _pd
    from plotly.subplots import make_subplots

    fig = go.Figure()
    if hist_df is None or not isinstance(hist_df, _pd.DataFrame) or hist_df.empty:
        apply_plotly_theme(fig)
        return fig

    work = hist_df.copy()
    if "date" in work.columns and not isinstance(work.index, _pd.DatetimeIndex):
        work["date"] = _pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"]).set_index("date")
    else:
        work.index = _pd.to_datetime(work.index, errors="coerce")
        work = work[~work.index.isna()]
    work = work.sort_index()
    if work.empty:
        apply_plotly_theme(fig)
        return fig

    has_bench = (
        bench_df is not None
        and isinstance(bench_df, _pd.DataFrame)
        and not bench_df.empty
        and "Close" in bench_df.columns
    )
    bench_aligned = None
    if has_bench:
        b = bench_df.copy()
        b.index = _pd.to_datetime(b.index, errors="coerce")
        b = b[~b.index.isna()].sort_index()
        b["Close"] = _pd.to_numeric(b["Close"], errors="coerce")
        b = b.dropna(subset=["Close"])
        if not b.empty:
            b["EMA21"] = b["Close"].ewm(span=21, adjust=False).mean()
            b["SMA50"] = b["Close"].rolling(50, min_periods=10).mean()
            b = b.loc[b.index >= work.index.min()]
            if not b.empty:
                bench_aligned = b
            else:
                has_bench = False
        else:
            has_bench = False

    n_rows = 3 if has_bench else 2
    row_heights = [0.44, 0.26, 0.30] if has_bench else [0.46, 0.54]
    titles = (
        [f"{bench_label} (21 EMA / 50 SMA)", "Net 52W Highs / Lows", "% Stocks Above Moving Averages"]
        if has_bench
        else ["Net 52W Highs / Lows", "% Stocks Above Moving Averages"]
    )

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=titles,
    )

    bar_row = 2 if has_bench else 1
    pct_row = 3 if has_bench else 2

    if has_bench and bench_aligned is not None:
        fig.add_trace(
            go.Scatter(
                x=bench_aligned.index,
                y=bench_aligned["Close"],
                name=bench_label,
                mode="lines",
                line=dict(color="#f8fafc", width=2.0),
                hovertemplate="%{x|%b %d}<br>Close: <b>%{y:,.1f}</b><extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bench_aligned.index,
                y=bench_aligned["EMA21"],
                name="21 EMA",
                mode="lines",
                line=dict(color="#38bdf8", width=1.3, dash="dot"),
                hovertemplate="21 EMA: %{y:,.1f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bench_aligned.index,
                y=bench_aligned["SMA50"],
                name="50 SMA",
                mode="lines",
                line=dict(color="#f59e0b", width=1.4),
                hovertemplate="50 SMA: %{y:,.1f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if "net_new_highs" in work.columns:
        nnh = _pd.to_numeric(work["net_new_highs"], errors="coerce").fillna(0)
        nnh_colors = ["#10b981" if v >= 0 else "#f43f5e" for v in nnh.values]
        fig.add_trace(
            go.Bar(
                x=work.index,
                y=nnh.values,
                name="Net New Highs",
                marker_color=nnh_colors,
                hovertemplate="%{x|%b %d}<br>Net 52W H/L: <b>%{y:+.0f}</b><extra></extra>",
            ),
            row=bar_row,
            col=1,
        )
        fig.add_hline(y=0, line_color="rgba(148,163,184,0.35)", line_width=1, row=bar_row, col=1)

    pct_specs = [
        ("above_20_pct", "% > 20 SMA", "#38bdf8", "dot", None),
        ("above_50_pct", "% > 50 SMA", "#10b981", "solid", "tozeroy"),
        ("above_200_pct", "% > 200 SMA", "#a78bfa", "dash", None),
    ]
    for col_name, trace_name, color, dash_style, fill_mode in pct_specs:
        if col_name not in work.columns:
            continue
        series = _pd.to_numeric(work[col_name], errors="coerce")
        if series.dropna().empty:
            continue
        trace_kw = dict(
            x=work.index,
            y=series.values,
            name=trace_name,
            mode="lines",
            line=dict(color=color, width=2.0 if col_name == "above_50_pct" else 1.5, dash=dash_style),
            hovertemplate=f"{trace_name}: <b>%{{y:.1f}}%</b><extra></extra>",
        )
        if fill_mode:
            trace_kw["fill"] = fill_mode
            trace_kw["fillcolor"] = "rgba(16, 185, 129, 0.14)"
        fig.add_trace(go.Scatter(**trace_kw), row=pct_row, col=1)

    fig.add_hline(y=80, line_dash="dash", line_color="rgba(239,68,68,0.45)", line_width=1, row=pct_row, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="rgba(16,185,129,0.45)", line_width=1, row=pct_row, col=1)
    fig.update_yaxes(range=[0, 100], ticksuffix="%", row=pct_row, col=1)

    apply_plotly_theme(fig)
    fig.update_layout(
        height=int(height),
        margin=dict(l=12, r=16, t=34, b=16),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(size=11)),
        bargap=0.22,
    )
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikethickness=1,
        spikecolor="rgba(148, 163, 184, 0.55)",
    )
    return fig


_HORIZON_RET_COL = {
    "1W": "Return_1W",
    "1M": "Return_1M",
    "3M": "Return_3M",
    "6M": "Return_6M",
}


def build_theme_rotation_bar_fig(
    df,
    *,
    horizon: str = "1M",
    top_n: int = 18,
    height: int = 520,
) -> go.Figure:
    """Build horizontal theme rotation leaderboard bars for a selected return horizon."""
    import pandas as _pd

    fig = go.Figure()
    h_key = str(horizon or "1M").upper().strip()
    metric_col = _HORIZON_RET_COL.get(h_key, "Return_1M")

    if (
        df is None
        or not isinstance(df, _pd.DataFrame)
        or df.empty
        or "Industry_Group" not in df.columns
        or metric_col not in df.columns
    ):
        apply_plotly_theme(fig)
        return fig

    work = df[["Industry_Group", metric_col]].copy()
    work[metric_col] = _pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=[metric_col]).sort_values(metric_col, ascending=False)
    if work.empty:
        apply_plotly_theme(fig)
        return fig

    if len(work) > top_n and top_n >= 4:
        has_neg = bool((work[metric_col] < 0).any())
        if has_neg:
            n_top = max(1, (top_n * 2) // 3)
            n_bot = max(1, top_n - n_top)
            work = _pd.concat([work.head(n_top), work.tail(n_bot)]).drop_duplicates(subset=["Industry_Group"])
            work = work.sort_values(metric_col, ascending=False)
        else:
            work = work.head(top_n)

    vals = work[metric_col].tolist()
    labels = work["Industry_Group"].astype(str).tolist()
    colors = ["#10b981" if v >= 0 else "#f43f5e" for v in vals]
    texts = [f"{v:+.1f}%" for v in vals]

    fig.add_trace(
        go.Bar(
            x=vals,
            y=labels,
            customdata=labels,
            orientation="h",
            marker_color=colors,
            text=texts,
            textposition="outside",
            textfont=dict(size=11, color="#e2e8f0"),
            hovertemplate=f"<b>%{{y}}</b><br>{h_key} Return: <b>%{{x:+.2f}}%</b><extra></extra>",
            showlegend=False,
        )
    )
    apply_plotly_theme(fig)
    fig.update_layout(
        height=int(height),
        clickmode="event+select",
        margin=dict(l=12, r=58, t=22, b=16),
        xaxis=dict(
            zeroline=True,
            zerolinecolor="rgba(148, 163, 184, 0.45)",
            zerolinewidth=1.4,
            ticksuffix="%",
            title=None,
        ),
        yaxis=dict(autorange="reversed", title=None, tickfont=dict(size=11, color="#e2e8f0")),
    )
    return fig


def build_sector_treemap_fig(
    df,
    *,
    horizon: str = "1M",
    group_by: str = "Rotation_Status",
    height: int = 520,
) -> go.Figure:
    """Build a DeepVue-style Industry Group treemap colored by horizon return and sized by breadth."""
    import pandas as _pd

    fig = go.Figure()
    h_key = str(horizon or "1M").upper().strip()
    metric_col = _HORIZON_RET_COL.get(h_key, "Return_1M")

    if (
        df is None
        or not isinstance(df, _pd.DataFrame)
        or df.empty
        or "Industry_Group" not in df.columns
        or metric_col not in df.columns
    ):
        apply_plotly_theme(fig)
        return fig

    work = df.copy()
    work[metric_col] = _pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=[metric_col])
    if work.empty:
        apply_plotly_theme(fig)
        return fig

    size_col = "Tradable_Count" if "Tradable_Count" in work.columns else ("Stock_Count" if "Stock_Count" in work.columns else None)
    if size_col:
        work["_tile_size"] = _pd.to_numeric(work[size_col], errors="coerce").fillna(1).clip(lower=1)
    else:
        work["_tile_size"] = 1.0

    parent_col = group_by if group_by in work.columns else None

    ids: list[str] = ["All Groups"]
    labels: list[str] = [f"Market ({h_key})"]
    parents: list[str] = [""]
    values: list[float] = [float(work["_tile_size"].sum())]
    color_vals: list[float] = [float(work[metric_col].mean())]
    custom_texts: list[str] = [f"{len(work)} Groups"]

    if parent_col:
        for grp_name, sub in work.groupby(parent_col, dropna=False):
            g_label = str(grp_name) if _pd.notna(grp_name) and str(grp_name).strip() else "Other"
            g_id = f"grp::{g_label}"
            ids.append(g_id)
            labels.append(g_label)
            parents.append("All Groups")
            values.append(float(sub["_tile_size"].sum()))
            color_vals.append(float(sub[metric_col].mean()))
            custom_texts.append(f"{len(sub)} groups · Avg {sub[metric_col].mean():+.1f}%")

            for _, row in sub.iterrows():
                ind = str(row["Industry_Group"])
                ret = float(row[metric_col])
                cnt = int(row.get("Tradable_Count", row.get("Stock_Count", row["_tile_size"])) or 1)
                rs_str = f"RS {float(row['Comp_RS']):.0f}" if "Comp_RS" in row and _pd.notna(row["Comp_RS"]) else ""
                ids.append(f"ind::{ind}")
                labels.append(ind)
                parents.append(g_id)
                values.append(float(row["_tile_size"]))
                color_vals.append(ret)
                custom_texts.append(f"{cnt} stocks {rs_str}".strip())
    else:
        for _, row in work.iterrows():
            ind = str(row["Industry_Group"])
            ret = float(row[metric_col])
            cnt = int(row.get("Tradable_Count", row.get("Stock_Count", row["_tile_size"])) or 1)
            ids.append(f"ind::{ind}")
            labels.append(ind)
            parents.append("All Groups")
            values.append(float(row["_tile_size"]))
            color_vals.append(ret)
            custom_texts.append(f"{cnt} stocks")

    max_abs = max(5.0, float(work[metric_col].abs().quantile(0.95)) if not work.empty else 10.0)
    colorscale = [
        [0.0, "#9f1239"],
        [0.35, "#f43f5e"],
        [0.5, "#1e293b"],
        [0.65, "#059669"],
        [1.0, "#10b981"],
    ]

    fig.add_trace(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            customdata=ids,
            branchvalues="total",
            text=custom_texts,
            texttemplate="<b>%{label}</b><br>%{color:+.1f}%",
            hovertemplate=(
                f"<b>%{{label}}</b><br>{h_key} Return: <b>%{{color:+.2f}}%</b>"
                "<br>%{text}<extra></extra>"
            ),
            marker=dict(
                colors=color_vals,
                colorscale=colorscale,
                cmin=-max_abs,
                cmax=max_abs,
                cmid=0.0,
                line=dict(width=1.2, color="#090d16"),
            ),
            pathbar=dict(visible=True, thickness=20),
            textfont=dict(family="Inter, sans-serif", size=12, color="#f8fafc"),
        )
    )
    apply_plotly_theme(fig)
    fig.update_layout(height=int(height), clickmode="event+select", margin=dict(l=6, r=6, t=12, b=6))
    return fig


def build_position_sizer_plan(
    equity: float | None,
    risk_pct_budget: float | None,
    entry_trigger: float | None,
    stop_loss: float | None,
    *,
    elder_cb: dict | None = None,
) -> dict[str, float | int | str | None]:
    """Compute Elder-guardrail-aware position sizing metrics without mutating risk state."""
    import math

    eq = max(0.0, float(equity or 0.0))
    req_risk = max(0.0, float(risk_pct_budget or 0.0))
    et = float(entry_trigger or 0.0)
    sl = float(stop_loss or 0.0)

    cb = elder_cb or {}
    cb_fired = bool(cb.get("circuit_breaker_fired", False))
    caution = bool(cb.get("caution_active", False))

    if cb_fired:
        eff_risk = 0.0
        guardrail = "HALTED (Elder 6% CB)"
        tone = "#ef4444"
    elif caution:
        eff_risk = min(req_risk, 0.20)
        guardrail = "THROTTLED <=0.20% (Elder 4%)"
        tone = "#f59e0b"
    else:
        eff_risk = req_risk
        guardrail = "ACTIVE (Within Guardrails)"
        tone = "#10b981"

    valid_levels = et > 0.0 and sl > 0.0 and et > sl
    stop_pts = (et - sl) if valid_levels else 0.0
    stop_pct = (stop_pts / et * 100.0) if valid_levels else None

    if valid_levels and eq > 0.0 and eff_risk > 0.0:
        dollar_risk = eq * (eff_risk / 100.0)
        max_shares = int(math.floor(dollar_risk / stop_pts))
        pos_value = max_shares * et
        pos_weight_pct = (pos_value / eq * 100.0) if eq > 0.0 else 0.0
    else:
        dollar_risk = 0.0
        max_shares = 0
        pos_value = 0.0
        pos_weight_pct = 0.0

    return {
        "effective_risk_pct": round(eff_risk, 2),
        "requested_risk_pct": round(req_risk, 2),
        "guardrail_status": guardrail,
        "guardrail_tone": tone,
        "stop_risk_pct": round(stop_pct, 2) if stop_pct is not None else None,
        "dollar_risk": round(dollar_risk, 2),
        "max_shares": max_shares,
        "position_value": round(pos_value, 2),
        "position_weight_pct": round(pos_weight_pct, 2),
        "valid_levels": valid_levels,
    }


def build_dcr_mini_gauge_html(dcr_pct: float | None, *, label: str = "DCR") -> str:
    """Render a compact horizontal Daily Closing Range (0-100%) capsule meter."""
    import html as _html

    if dcr_pct is None or dcr_pct != dcr_pct:
        return (
            f'<span style="display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border-radius:6px;'
            f'background:rgba(15,23,42,0.75);border:1px solid rgba(148,163,184,0.22);'
            f'font-size:0.72rem;color:#94a3b8;font-family:\'JetBrains Mono\',monospace;">'
            f'{_html.escape(label)}: N/A</span>'
        )
    clamped = max(0.0, min(100.0, float(dcr_pct)))
    fill_col = "#10b981" if clamped >= 70.0 else ("#f59e0b" if clamped >= 40.0 else "#f43f5e")
    return (
        f'<span style="display:inline-flex;align-items:center;gap:7px;padding:3px 9px;border-radius:6px;'
        f'background:rgba(15,23,42,0.85);border:1px solid rgba(255,255,255,0.08);'
        f'font-size:0.73rem;font-family:\'JetBrains Mono\',monospace;color:#e2e8f0;" '
        f'title="Daily Closing Range: {clamped:.0f}% (Close position within High-Low range)">'
        f'<span style="color:#94a3b8;font-weight:700;">{_html.escape(label)}</span>'
        f'<span style="width:46px;height:7px;border-radius:999px;background:rgba(255,255,255,0.10);overflow:hidden;display:inline-block;">'
        f'<span style="display:block;width:{clamped:.1f}%;height:100%;background:{fill_col};"></span></span>'
        f'<b style="color:{fill_col};">{clamped:.0f}%</b></span>'
    )


def render_position_sizer_strip_html(
    plan: dict,
    *,
    currency_symbol: str = "₹",
    dcr_pct: float | None = None,
) -> str:
    """Render a compact 1-line institutional Position Sizer summary card for Focus List."""
    import html as _html

    max_sh = int(plan.get("max_shares", 0) or 0)
    pos_val = float(plan.get("position_value", 0.0) or 0.0)
    pos_wt = float(plan.get("position_weight_pct", 0.0) or 0.0)
    eff_r = float(plan.get("effective_risk_pct", 0.0) or 0.0)
    d_risk = float(plan.get("dollar_risk", 0.0) or 0.0)
    status = str(plan.get("guardrail_status", "ACTIVE"))
    tone = str(plan.get("guardrail_tone", "#10b981"))
    valid_lvls = bool(plan.get("valid_levels", False))

    dcr_badge = build_dcr_mini_gauge_html(dcr_pct, label="Last Bar DCR") if dcr_pct is not None else ""
    shares_txt = f"{max_sh:,} sh" if valid_lvls else "Set Entry > Stop"
    pos_txt = f"{currency_symbol}{pos_val:,.0f} ({pos_wt:.1f}%)" if valid_lvls else "—"

    return (
        f'<div style="background:rgba(15,23,42,0.88);border:1px solid rgba(255,255,255,0.08);'
        f'border-left:4px solid {tone};border-radius:10px;padding:9px 13px;margin:6px 0 10px 0;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:6px;">'
        f'<span style="font-size:0.72rem;font-weight:800;color:#94a3b8;text-transform:uppercase;letter-spacing:0.6px;">'
        f'⚖️ Elder-Aware Position Sizer</span>'
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
        f'{dcr_badge}'
        f'<span style="color:{tone};font-size:0.72rem;font-weight:800;font-family:\'JetBrains Mono\',monospace;">'
        f'{_html.escape(status)}</span></div></div>'
        f'<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;font-family:\'JetBrains Mono\',monospace;">'
        f'<div><div style="font-size:0.66rem;color:#94a3b8;text-transform:uppercase;">Max Position</div>'
        f'<div style="font-size:0.95rem;font-weight:800;color:#f8fafc;">{shares_txt}</div></div>'
        f'<div><div style="font-size:0.66rem;color:#94a3b8;text-transform:uppercase;">Capital Alloc</div>'
        f'<div style="font-size:0.95rem;font-weight:800;color:#38bdf8;">{pos_txt}</div></div>'
        f'<div><div style="font-size:0.66rem;color:#94a3b8;text-transform:uppercase;">Allowed Risk ({eff_r:.2f}%)</div>'
        f'<div style="font-size:0.95rem;font-weight:800;color:{tone};">{currency_symbol}{d_risk:,.0f}</div></div>'
        f'</div></div>'
    )





def _extract_plotly_points(event) -> list:
    """Safely extract selection points from Streamlit PlotlyState objects or dicts."""
    if not event:
        return []
    sel = event.get("selection") if isinstance(event, dict) else getattr(event, "selection", None)
    if not sel:
        return []
    pts = sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
    return list(pts) if pts else []


def _point_field(pt, key: str):
    if isinstance(pt, dict):
        return pt.get(key)
    return getattr(pt, key, None)


def extract_plotly_selection_signature(event) -> str:
    """Return a deterministic signature for Plotly point selections, or '' when empty."""
    pts = _extract_plotly_points(event)
    if not pts:
        return ""
    tokens = []
    for p in pts:
        cdata = _point_field(p, "customdata")
        if isinstance(cdata, (list, tuple)):
            cdata = "|".join(str(x) for x in cdata)
        tokens.append(
            f"{_point_field(p, 'point_index') or _point_field(p, 'point_number') or ''}"
            f":{cdata or ''}:{_point_field(p, 'id') or ''}:{_point_field(p, 'y') or ''}:{_point_field(p, 'label') or ''}"
        )
    return ";".join(sorted(tokens))


def resolve_heatmap_selected_group(
    treemap_event,
    bar_event,
    valid_groups,
    current_group: str | None = None,
    *,
    prefer_source: str = "bar",
) -> str | None:
    """Resolve a clicked Industry_Group leaf from treemap or bar Plotly events."""
    valid_set = {str(g).strip() for g in (valid_groups or []) if g is not None and str(g).strip()}
    if not valid_set:
        return None

    def _from_event(ev) -> str | None:
        for pt in _extract_plotly_points(ev):
            candidates = []
            cdata = _point_field(pt, "customdata")
            if isinstance(cdata, (list, tuple)) and cdata:
                candidates.append(cdata[0])
            elif cdata is not None:
                candidates.append(cdata)
            for k in ("id", "y", "label"):
                v = _point_field(pt, k)
                if v is not None:
                    candidates.append(v)

            for raw_val in candidates:
                s = str(raw_val).strip()
                if not s or s == "All Groups" or s.startswith("grp::"):
                    continue
                if s.startswith("ind::"):
                    leaf = s[5:].strip()
                    if leaf in valid_set:
                        return leaf
                elif s in valid_set:
                    return s
        return None

    sources = (bar_event, treemap_event) if prefer_source == "bar" else (treemap_event, bar_event)
    for src in sources:
        picked = _from_event(src)
        if picked:
            return picked

    cur = str(current_group).strip() if current_group else None
    return cur if (cur and cur in valid_set) else None


def build_theme_drilldown_kpi_html(
    grp_row,
    *,
    horizon: str = "1M",
    n_constits: int = 0,
) -> str:
    """Build a compact zero-indent dark KPI strip for the Heatmap inline Theme Desk."""
    import html as _html

    get_v = grp_row.get if isinstance(grp_row, dict) else grp_row.get
    h_key = str(horizon or "1M").upper().strip()
    ret_col = _HORIZON_RET_COL.get(h_key, "Return_1M")

    grp_name = _html.escape(str(get_v("Industry_Group", "Unknown Theme")))
    rot_status = _html.escape(str(get_v("Rotation_Status", "Neutral")))
    rank_val = int(float(get_v("Rank_Today", 0) or 0))
    comp_rs = float(get_v("Comp_RS", 0.0) or 0.0)
    ret_val = float(get_v(ret_col, 0.0) or 0.0)
    sortino_val = float(get_v("Sortino_3M", 0.0) or 0.0)
    apex_cnt = int(float(get_v("Apex_Leaders", 0) or 0))

    ret_col_hex = "#34d399" if ret_val >= 0 else "#f87171"
    rs_col_hex = "#34d399" if comp_rs >= 70 else ("#38bdf8" if comp_rs >= 50 else "#f87171")
    rot_bg = "rgba(16,185,129,0.16)" if "Surging" in rot_status or "Accum" in rot_status else (
        "rgba(239,68,68,0.16)" if "Distribut" in rot_status or "Lag" in rot_status else "rgba(56,189,248,0.14)"
    )
    rot_fg = "#34d399" if "Surging" in rot_status or "Accum" in rot_status else (
        "#f87171" if "Distribut" in rot_status or "Lag" in rot_status else "#38bdf8"
    )

    return (
        '<div style="background:linear-gradient(135deg,rgba(15,23,42,0.92),rgba(2,6,23,0.96));'
        'border:1px solid rgba(56,189,248,0.28);border-radius:10px;padding:10px 14px;margin:8px 0 10px 0;'
        'display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">'
        '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.82rem;font-weight:800;color:#38bdf8;background:rgba(56,189,248,0.12);padding:3px 8px;border-radius:6px;">#{rank_val}</span>'
        f'<span style="font-size:1.02rem;font-weight:800;color:#f8fafc;">{grp_name}</span>'
        f'<span style="font-size:0.72rem;font-weight:700;color:{rot_fg};background:{rot_bg};padding:2px 8px;border-radius:999px;">{rot_status}</span>'
        '</div>'
        '<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;font-family:\'JetBrains Mono\',monospace;font-size:0.80rem;">'
        f'<span>Comp RS: <b style="color:{rs_col_hex};">{comp_rs:.0f}</b></span>'
        f'<span>{h_key} Ret: <b style="color:{ret_col_hex};">{ret_val:+.2f}%</b></span>'
        f'<span>3M Sortino: <b style="color:#38bdf8;">{sortino_val:.2f}</b></span>'
        f'<span>👑 Apex: <b style="color:#fbbf24;">{apex_cnt}</b></span>'
        f'<span style="color:#94a3b8;">Peers: <b style="color:#f8fafc;">{int(n_constits)}</b></span>'
        '</div></div>'
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_cloud_single_ticker_series_with_mas(yf_sym: str):
    """Cloud on-demand single-ticker fallback when local .pkl matrix is absent."""
    import pandas as _pd
    try:
        import yfinance as _yf
        hist = _yf.Ticker(str(yf_sym)).history(period="1y", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        if getattr(hist.index, "tz", None) is not None:
            hist.index = hist.index.tz_localize(None)
        keep_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in hist.columns]
        out = hist[keep_cols].copy()
        for col in keep_cols:
            out[col] = _pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["Close"]).tail(180)
        if out.empty:
            return None
        out["EMA21"] = out["Close"].ewm(span=21, adjust=False).mean()
        out["SMA50"] = out["Close"].rolling(50, min_periods=20).mean()
        out["EMA65"] = out["Close"].ewm(span=65, adjust=False, min_periods=45).mean()
        if "Volume" in out.columns:
            out["Vol_SMA50"] = out["Volume"].rolling(50, min_periods=10).mean()
        return out
    except Exception:
        return None


def load_cached_ticker_series_with_mas(
    raw_ticker: str,
    clean_ticker: str,
    market: str = "IN",
):
    """Load local cached OHLCV + 21EMA/50SMA/65EMA for IN or US, falling back to single-ticker yfinance on Cloud."""
    import os
    import pandas as _pd

    is_us = str(market or "IN").upper().strip().startswith("US")
    raw_s = str(raw_ticker or "").strip()
    clean_s = str(clean_ticker or raw_s).strip()
    lookup_cands = [raw_s, clean_s]
    if not is_us:
        lookup_cands.extend([f"{clean_s}.NS", f"{clean_s}.BO"])

    try:
        from price_history_manager import get_matrix_cache_signature, load_cached_price_matrix
        candidates = (
            ["us_prices_matrix.pkl", "breadth_matrix.pkl", "historical_prices_matrix.pkl"]
            if is_us
            else ["historical_prices_matrix.pkl", "breadth_matrix.pkl", "microcap_prices_matrix.pkl"]
        )
        matrix_paths = [p for p in candidates if os.path.exists(p)]
    except Exception:
        matrix_paths = []

    for mp in matrix_paths:
        try:
            mat = load_cached_price_matrix(mp, cache_sig=get_matrix_cache_signature(mp))
            if mat is None or mat.empty or not isinstance(mat.columns, _pd.MultiIndex):
                continue
            lvl0 = set(mat.columns.get_level_values(0))
            lookup = next((c for c in lookup_cands if c and c in lvl0), None)
            if lookup is None:
                continue
            sub = mat[lookup].dropna(how="all").copy()
            if sub.empty or "Close" not in sub.columns:
                continue
            if getattr(sub.index, "tz", None) is not None:
                sub.index = sub.index.tz_localize(None)
            keep_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in sub.columns]
            out = sub[keep_cols].copy()
            for col in keep_cols:
                out[col] = _pd.to_numeric(out[col], errors="coerce")
            out = out.dropna(subset=["Close"]).tail(180)
            if out.empty:
                continue
            out["EMA21"] = out["Close"].ewm(span=21, adjust=False).mean()
            out["SMA50"] = out["Close"].rolling(50, min_periods=20).mean()
            out["EMA65"] = out["Close"].ewm(span=65, adjust=False, min_periods=45).mean()
            if "Volume" in out.columns:
                out["Vol_SMA50"] = out["Volume"].rolling(50, min_periods=10).mean()
            return out
        except Exception:
            continue

    yf_sym = raw_s if (is_us or raw_s.upper().endswith((".NS", ".BO"))) else f"{clean_s}.NS"
    if yf_sym:
        res = _fetch_cloud_single_ticker_series_with_mas(yf_sym)
        if res is not None:
            return res
    return None


_STOCKBEE_COLS = [
    "Date",
    "Valid_Stocks",
    "Up_4_Pct",
    "Down_4_Pct",
    "Ratio_5D",
    "Ratio_10D",
    "Up_5D_Sum",
    "Down_5D_Sum",
    "Up_10D_Sum",
    "Down_10D_Sum",
    "Up_25_Qtr",
    "Down_25_Qtr",
    "T2108_Pct",
    "Pulse",
]


def _prepare_stockbee_work_frame(
    close_df,
    *,
    lookback_days: int = 20,
    min_valid_stocks: int | None = None,
):
    """
    Shared clean-panel & T0 selector for Stockbee Market Monitor grid and
    Thrusting Industries attribution so both compute against identical bars.
    Returns (work, daily_ret, valid_ret_mask, n_sessions).
    """
    import pandas as _pd

    n_sessions = max(1, int(lookback_days))
    empty = _pd.DataFrame()
    empty_s = _pd.Series(dtype=bool)
    if close_df is None or not isinstance(close_df, _pd.DataFrame) or close_df.empty or len(close_df) < 2:
        return empty, empty, empty_s, n_sessions

    work = close_df.apply(_pd.to_numeric, errors="coerce")
    work = work.where(work > 0)
    idx = _pd.to_datetime(work.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    work.index = idx
    work = work.loc[work.index.notna()].sort_index()
    work = work.loc[~work.index.duplicated(keep="last")]
    if len(work) < 2:
        return empty, empty, empty_s, n_sessions

    warmup_rows = max(n_sessions + 95, 140)
    work = work.tail(warmup_rows)

    thresh = (
        int(min_valid_stocks)
        if min_valid_stocks is not None
        else max(min(5, max(1, work.shape[1])), int(round(0.35 * max(1, work.shape[1]))))
    )
    work = work.loc[work.notna().sum(axis=1) >= thresh]
    if len(work) < 2:
        return empty, empty, empty_s, n_sessions

    daily_ret = work.pct_change(fill_method=None)
    valid_ret_mask = daily_ret.notna().sum(axis=1) > 0
    if not bool(valid_ret_mask.any()):
        return empty, empty, empty_s, n_sessions

    return work, daily_ret, valid_ret_mask, n_sessions


def compute_stockbee_monitor_table(
    close_df,
    *,
    lookback_days: int = 20,
    min_valid_stocks: int | None = None,
):
    """Compute Stockbee Market Monitor daily grid cleanly with warmup & noise guards."""
    import numpy as _np
    import pandas as _pd

    work, daily_ret, valid_ret_mask, n_sessions = _prepare_stockbee_work_frame(
        close_df,
        lookback_days=lookback_days,
        min_valid_stocks=min_valid_stocks,
    )
    if work.empty:
        return _pd.DataFrame(columns=_STOCKBEE_COLS)

    qtr_ret = work.pct_change(63, fill_method=None)
    sma40 = work.rolling(40, min_periods=20).mean()

    up4 = (daily_ret >= 0.04).sum(axis=1).astype(int)
    dn4 = (daily_ret <= -0.04).sum(axis=1).astype(int)
    valid_cnt = daily_ret.notna().sum(axis=1).astype(int)

    up_5d = up4.rolling(5, min_periods=1).sum().astype(int)
    dn_5d = dn4.rolling(5, min_periods=1).sum().astype(int)
    up_10d = up4.rolling(10, min_periods=1).sum().astype(int)
    dn_10d = dn4.rolling(10, min_periods=1).sum().astype(int)

    ratio_5d = (up_5d / dn_5d.clip(lower=1)).round(2)
    ratio_10d = (up_10d / dn_10d.clip(lower=1)).round(2)

    up_25_qtr = (qtr_ret >= 0.25).sum(axis=1).astype(int)
    dn_25_qtr = (qtr_ret <= -0.25).sum(axis=1).astype(int)

    abv_40_elig = (work.notna() & sma40.notna()).sum(axis=1)
    abv_40_cnt = (work > sma40).sum(axis=1)
    t2108 = _pd.Series(
        _np.where(abv_40_elig > 0, (abv_40_cnt / abv_40_elig.replace(0, _np.nan)) * 100.0, _np.nan),
        index=work.index,
    ).round(1)

    out = _pd.DataFrame(
        {
            "Date": work.index,
            "Valid_Stocks": valid_cnt,
            "Up_4_Pct": up4,
            "Down_4_Pct": dn4,
            "Ratio_5D": ratio_5d,
            "Ratio_10D": ratio_10d,
            "Up_5D_Sum": up_5d,
            "Down_5D_Sum": dn_5d,
            "Up_10D_Sum": up_10d,
            "Down_10D_Sum": dn_10d,
            "Up_25_Qtr": up_25_qtr,
            "Down_25_Qtr": dn_25_qtr,
            "T2108_Pct": t2108,
        },
        index=work.index,
    )
    out = out.loc[valid_ret_mask].tail(n_sessions).copy()
    if out.empty:
        return _pd.DataFrame(columns=_STOCKBEE_COLS)

    pulses = []
    for _, r in out.iterrows():
        v_cnt = max(1, int(r["Valid_Stocks"]))
        u4 = int(r["Up_4_Pct"])
        d4 = int(r["Down_4_Pct"])
        u5 = int(r["Up_5D_Sum"])
        d5 = int(r["Down_5D_Sum"])
        r5 = float(r["Ratio_5D"])

        min_day_surge = max(3, int(round(0.035 * v_cnt)))
        min_5d_impulse = max(2, int(round(0.025 * v_cnt)))

        if r5 >= 2.0 and u5 >= min_5d_impulse and u4 >= d4:
            pulses.append("🚀 5D THRUST")
        elif u4 >= min_day_surge and u4 >= max(d4 * 2, 1):
            pulses.append("⚡ DAY THRUST")
        elif d5 >= min_5d_impulse and r5 <= 0.5 and d4 >= u4:
            pulses.append("🔴 5D SELLING")
        elif d4 >= min_day_surge and d4 >= max(u4 * 2, 1):
            pulses.append("🔻 DAY SELLOFF")
        else:
            pulses.append("⚖️ BALANCED")

    out["Pulse"] = pulses
    out = out.iloc[::-1].reset_index(drop=True)
    return out[_STOCKBEE_COLS]


def _stockbee_allowed_tickers_for_scope(market: str, universe_scope: str = ""):
    """Resolve constituent ticker list for IN (NTM 750) or US (S&P 500 / Nasdaq 100 / Broad)."""
    mkt = str(market or "IN").upper().strip()
    sc = str(universe_scope or "").upper()
    if mkt == "US":
        try:
            from us_universe_manager import (
                get_broad_us_universe,
                get_nasdaq100_constituents,
                get_sp500_constituents,
            )

            if "NASDAQ" in sc:
                nq = get_nasdaq100_constituents()
                if nq:
                    return list(nq)
            if any(k in sc for k in ("BROAD", "ALL", "US_IXIC", "IXIC", "FULL")):
                br = get_broad_us_universe()
                if br:
                    return list(br)
                try:
                    from universe_security_master import load_universe_tickers
                    uni = load_universe_tickers("US", include_etfs=False)
                    if uni:
                        return list(uni)
                except Exception:
                    pass
            sp = get_sp500_constituents()
            if sp:
                return list(sp)
        except Exception:
            pass
        return _stage_allowed_tickers_for_market("US")
    return _stage_allowed_tickers_for_market("IN")


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def _load_stockbee_monitor_inner(
    market: str,
    scope_key: str,
    cache_sig: tuple,
    lookback_days: int,
):
    import os
    import pandas as _pd

    try:
        from price_history_manager import extract_market_ohlcv_panels, load_cached_price_matrix
    except Exception:
        return _pd.DataFrame(columns=_STOCKBEE_COLS)

    mkt = str(market or "IN").upper().strip()
    matrix_path = _stage_matrix_path_for_market(mkt)
    if not os.path.exists(matrix_path):
        return _pd.DataFrame(columns=_STOCKBEE_COLS)

    df = load_cached_price_matrix(matrix_path, cache_sig=cache_sig)
    if df is None or df.empty:
        return _pd.DataFrame(columns=_STOCKBEE_COLS)

    allowed = _stockbee_allowed_tickers_for_scope(mkt, scope_key)
    panels = extract_market_ohlcv_panels(
        df,
        market="US" if mkt == "US" else "IN",
        allowed_tickers=allowed,
        include_etfs=False,
        deduplicate_dual_in=(mkt != "US"),
        min_coverage_ratio=0.35,
        min_abs=20,
    )
    return compute_stockbee_monitor_table(panels.get("close"), lookback_days=lookback_days)


def load_stockbee_monitor_df(
    market: str = "IN",
    universe_scope: str = "DEFAULT",
    lookback_days: int = 30,
):
    """Read local cached price matrix or Turso cloud_json_cache to build the Stockbee Market Monitor grid."""
    import os
    import re
    import pandas as _pd

    try:
        from price_history_manager import get_matrix_cache_signature

        mkt = str(market or "IN").upper().strip()
        matrix_path = _stage_matrix_path_for_market(mkt)
        scope_key = str(universe_scope or "DEFAULT").upper().strip()
        if os.path.exists(matrix_path):
            sig = get_matrix_cache_signature(matrix_path)
            df_loc = _load_stockbee_monitor_inner(mkt, scope_key, sig, int(lookback_days))
            if df_loc is not None and not df_loc.empty:
                return df_loc

        from database import load_cloud_json_cache_cached
        sc_u = str(universe_scope or "DEFAULT").upper().strip()
        raw_slug = re.sub(r"[^a-z0-9]+", "_", sc_u.lower()).strip("_")
        keys_to_try = []
        if mkt == "US":
            if "NASDAQ" in sc_u or "NDX" in sc_u:
                keys_to_try.append("stockbee_monitor_us_nasdaq_100")
            elif "BROAD" in sc_u or "ALL" in sc_u or "IXIC" in sc_u:
                keys_to_try.append("stockbee_monitor_us_broad")
            elif "SP500" in sc_u or "S&P" in sc_u or "GSPC" in sc_u:
                keys_to_try.append("stockbee_monitor_us_sp500")
            if raw_slug:
                keys_to_try.append(f"stockbee_monitor_us_{raw_slug}")
            keys_to_try.extend(["stockbee_monitor_us_sp500", "stockbee_monitor_us_broad"])
        else:
            if "FULL" in sc_u or "ALL" in sc_u:
                keys_to_try.append("stockbee_monitor_in_full")
            elif "NTM" in sc_u or "750" in sc_u:
                keys_to_try.append("stockbee_monitor_in_ntm750")
            if raw_slug:
                keys_to_try.append(f"stockbee_monitor_in_{raw_slug}")
            keys_to_try.extend(["stockbee_monitor_in_ntm750", "stockbee_monitor_in_full"])
        for ck in dict.fromkeys(keys_to_try):
            cached_df, _ = load_cloud_json_cache_cached(ck)
            if isinstance(cached_df, _pd.DataFrame) and not cached_df.empty:
                return cached_df.head(int(lookback_days)).copy()
        return _pd.DataFrame(columns=_STOCKBEE_COLS)
    except Exception:
        return _pd.DataFrame(columns=_STOCKBEE_COLS)


def _format_stockbee_ratio_cell(ratio_val: float, up_sum: int, dn_sum: int) -> tuple[str, str, str]:
    """Return (display_text, text_color, bg_color) guarding dead-week 0/0 vs Inf."""
    if up_sum == 0 and dn_sum == 0:
        return "0.00", "#94a3b8", "rgba(148,163,184,0.06)"
    if dn_sum == 0 and up_sum > 0:
        return f"∞ ({up_sum})", "#34d399", "rgba(16,185,129,0.26)"
    if ratio_val >= 2.0:
        return f"{ratio_val:.2f}", "#34d399", "rgba(16,185,129,0.24)"
    if ratio_val <= 0.5:
        return f"{ratio_val:.2f}", "#f87171", "rgba(239,68,68,0.24)"
    return f"{ratio_val:.2f}", "#e2e8f0", "transparent"


def analyze_stockbee_regime_insight(df_grid) -> dict:
    """Synthesize O'Neil / Minervini / Stockbee actionable regime insight from monitor rows."""
    import pandas as _pd

    if df_grid is None or not isinstance(df_grid, _pd.DataFrame) or df_grid.empty:
        return {}

    n_rows = len(df_grid)
    r0 = df_grid.iloc[0]
    u4 = int(r0.get("Up_4_Pct", 0))
    d4 = int(r0.get("Down_4_Pct", 0))
    u5 = int(r0.get("Up_5D_Sum", 0))
    d5 = int(r0.get("Down_5D_Sum", 0))
    r5 = float(r0.get("Ratio_5D", 0.0))
    r10 = float(r0.get("Ratio_10D", 0.0))
    qu = int(r0.get("Up_25_Qtr", 0))
    qd = int(r0.get("Down_25_Qtr", 0))
    q_spread = qu - qd
    pulse = str(r0.get("Pulse", "⚖️ BALANCED"))
    t21_raw = r0.get("T2108_Pct", float("nan"))
    t2108 = float(t21_raw) if _pd.notna(t21_raw) else None

    # 1. Recent 2-day impulse flow
    u2d = int(df_grid["Up_4_Pct"].iloc[: min(2, n_rows)].sum())
    d2d = int(df_grid["Down_4_Pct"].iloc[: min(2, n_rows)].sum())
    two_day_neg = (d2d > u2d) and (d2d > 0)
    today_paused = (u4 <= int(round(d4 * 1.35))) and ((u4 + d4) > 0)

    # 2. Rolling 5D window drop-off risk (row index 4 drops off next session)
    drop_dt, drop_u, drop_d, ex_drop_ratio, rolloff_fragile = "", 0, 0, r5, False
    if n_rows >= 5:
        r_drop = df_grid.iloc[4]
        dt_obj = r_drop.get("Date")
        drop_dt = dt_obj.strftime("%b %d") if hasattr(dt_obj, "strftime") else str(dt_obj)[:10]
        drop_u = int(r_drop.get("Up_4_Pct", 0))
        drop_d = int(r_drop.get("Down_4_Pct", 0))
        rem_u = max(0, u5 - drop_u)
        rem_d = max(0, d5 - drop_d)
        ex_drop_ratio = round(rem_u / max(1, rem_d), 2)
        rolloff_fragile = (r5 >= 1.8) and (u5 > 0) and (drop_u >= max(1, int(round(0.45 * u5)))) and (ex_drop_ratio < 1.5)

    # 3. Quarterly 25% leader crossover / contraction
    prior_q = [
        int(df_grid.iloc[i].get("Up_25_Qtr", 0)) - int(df_grid.iloc[i].get("Down_25_Qtr", 0))
        for i in range(1, min(10, n_rows))
    ]
    fresh_q_cross = (q_spread >= 0) and any(s < 0 for s in prior_q[:6])
    q_fading = (q_spread < 0) and (len(prior_q) >= 4 and q_spread < prior_q[min(4, len(prior_q) - 1)])

    # 4. T2108 level & 10D velocity
    t2108_delta = None
    if t2108 is not None and n_rows >= 2:
        ref_t21 = df_grid.iloc[min(9, n_rows - 1)].get("T2108_Pct", float("nan"))
        if _pd.notna(ref_t21):
            t2108_delta = round(t2108 - float(ref_t21), 1)

    t2108_oversold = (t2108 is not None) and (t2108 < 22.0)
    t2108_stretched = (t2108 is not None) and (t2108 >= 72.0)
    t2108_fast_surge = (t2108_delta is not None) and (t2108_delta >= 22.0)
    t21_txt = f"{t2108:.1f}%" if t2108 is not None else "—"

    # 5. Hedge Fund Regime Synthesis
    if t2108_oversold and r5 < 2.0 and "DAY THRUST" not in pulse:
        verdict = "🔵 OVERSOLD BOUNCE ZONE"
        posture = "10–30% Pilot Watch"
        tone = "cyan"
        playbook = (
            f"T2108 ({t21_txt}) sits in washed-out oversold territory (<22%). "
            "Halt aggressive shorting, build a Stage 2 RS watchlist, and wait for a Day-1 +4% thrust."
        )
    elif (r5 >= 2.0 or "DAY THRUST" in pulse) and (t2108 is not None and t2108 < 45.0):
        verdict = "🟢 IGNITION OFF LOWS"
        posture = "50–75% Ramp Exposure"
        tone = "emerald"
        playbook = (
            f"Fresh buying impulse off depressed T2108 ({t21_txt}; 5D {r5:.2f}). "
            "Pilot high-RS Stage 2 pivots early with defined initial stops."
        )
    elif r5 >= 2.0 and (two_day_neg or rolloff_fragile or (t2108_stretched and r10 < 1.8) or (r10 < 1.5 and q_spread < 0)):
        verdict = "⚠️ THRUST DECAY / EXTENDED"
        posture = "40–60% Selective · Raise Stops"
        tone = "amber"
        echo_part = f"{drop_dt} (+{drop_u}/-{drop_d}) exits 5D next (dropping ex-ratio to {ex_drop_ratio:.2f})" if rolloff_fragile else f"10D Ratio ({r10:.2f}) hasn't confirmed"
        playbook = (
            f"5D Ratio ({r5:.2f}) masks short-term narrowing (2D flow +{u2d}/-{d2d}, Qtr Spread {q_spread:+d}, T2108 {t21_txt}); "
            f"{echo_part}. Protect open winners, tighten stops, and avoid chasing."
        )
    elif r5 >= 2.0 and (r10 >= 1.8 or q_spread >= 0):
        if t2108_stretched:
            verdict = "🟠 STRETCHED BULL THRUST"
            posture = "60–80% Hold / Trail"
            tone = "amber"
            playbook = (
                f"Dual thrust active (5D {r5:.2f} · 10D {r10:.2f}), but T2108 ({t21_txt}) is short-term extended (>72%). "
                "Let winners run on trailing stops; avoid initiating extended entries."
            )
        elif t2108_fast_surge or today_paused:
            verdict = "🚀 CONFIRMED BULL THRUST"
            posture = "75–100% Buy Pullbacks"
            tone = "emerald"
            surge_note = f"T2108 surged {t2108_delta:+.1f} pts to {t21_txt}" if t2108_fast_surge else f"today paused (+{u4}/-{d4}) with T2108 at {t21_txt}"
            q_note = f"fresh Qtr crossover ({q_spread:+d})" if fresh_q_cross else f"Qtr Spread {q_spread:+d}"
            playbook = (
                f"Dual 5D ({r5:.2f}) + 10D ({r10:.2f}) accumulation confirmed with {q_note}; {surge_note}. "
                "Aggressive long bias in Stage 2 leaders — stalk tight 10E/21E pullbacks & VCP pivots rather than chasing."
            )
        else:
            verdict = "🚀 CONFIRMED BULL THRUST"
            posture = "80–100% Press Leaders"
            tone = "emerald"
            playbook = (
                f"Broad multi-horizon expansion (5D {r5:.2f} · 10D {r10:.2f} · Qtr {q_spread:+d} · T2108 {t21_txt}). "
                "Press exposure in leading Industry Groups breaking out of sound bases."
            )
    elif (r5 <= 0.65 and r10 <= 0.85) or ("SELL" in pulse and q_spread < 0):
        verdict = "🔴 DISTRIBUTION / CORRECTION"
        posture = "0–25% Cash Heavy"
        tone = "rose"
        playbook = (
            f"Heavy liquidation pressure (5D {r5:.2f} · 10D {r10:.2f} · Qtr {q_spread:+d} · T2108 {t21_txt}). "
            "Preserve capital, cut laggards fast, and avoid catching falling knives."
        )
    elif r5 >= 1.4 and q_spread >= 0:
        verdict = "🟢 ACCUMULATION SHIFT"
        posture = "60–80% Selective Longs"
        tone = "emerald"
        playbook = (
            f"Constructive breadth rotation (5D {r5:.2f} · Qtr {q_spread:+d} · T2108 {t21_txt}). "
            "Focus on RS New Highs / Blue-Dot pivots in top-ranked themes."
        )
    else:
        verdict = "⚖️ MIXED ROTATIONAL CHOP"
        posture = "30–50% A+ Setups Only"
        tone = "slate"
        playbook = (
            f"Cross-currents (5D {r5:.2f} · 10D {r10:.2f} · 2D +{u2d}/-{d2d} · Qtr {q_spread:+d} · T2108 {t21_txt}). "
            "Stay picky: require tight pivots and quick partial profits."
        )

    # 6. Four Diagnostic Pills
    if r5 >= 2.0 and r10 >= 1.8 and not two_day_neg:
        p1 = {"label": "5D vs 10D", "value": f"✅ Dual Thrust ({r5:.2f} / {r10:.2f})", "tone": "emerald"}
    elif r5 >= 2.0 and (two_day_neg or r10 < 1.8):
        p1 = {"label": "5D vs 10D", "value": f"⚠️ Stale Echo (2D +{u2d}/-{d2d} · 10D {r10:.2f})", "tone": "amber"}
    elif r5 <= 0.65:
        p1 = {"label": "5D vs 10D", "value": f"🔴 Weak Impulse ({r5:.2f} / {r10:.2f})", "tone": "rose"}
    else:
        p1 = {"label": "5D vs 10D", "value": f"⚖️ Mixed ({r5:.2f} / {r10:.2f})", "tone": "slate"}

    if fresh_q_cross:
        p2 = {"label": "25% Qtr Flow", "value": f"🟢 Bullish Crossover ({qu} vs {qd} · {q_spread:+d})", "tone": "emerald"}
    elif q_spread >= 0:
        p2 = {"label": "25% Qtr Flow", "value": f"🟢 Positive Core ({qu} vs {qd} · {q_spread:+d})", "tone": "emerald"}
    elif q_fading:
        p2 = {"label": "25% Qtr Flow", "value": f"🔴 Contracting ({qu} vs {qd} · {q_spread:+d})", "tone": "rose"}
    else:
        p2 = {"label": "25% Qtr Flow", "value": f"🟠 Negative ({qu} vs {qd} · {q_spread:+d})", "tone": "amber"}

    d_str = f" ({t2108_delta:+.1f} pts/10D)" if t2108_delta is not None else ""
    if t2108_oversold:
        p3 = {"label": "T2108 Zone", "value": f"🔵 Oversold {t21_txt}{d_str}", "tone": "cyan"}
    elif t2108_stretched:
        p3 = {"label": "T2108 Zone", "value": f"🟠 Near Ceiling {t21_txt}{d_str}", "tone": "amber"}
    elif t2108_fast_surge:
        p3 = {"label": "T2108 Zone", "value": f"⚡ Fast Surge {t21_txt}{d_str}", "tone": "emerald"}
    else:
        p3 = {"label": "T2108 Zone", "value": f"🟢 Bull Sweet Spot {t21_txt}{d_str}" if (t2108 and t2108 >= 45) else f"⚖️ Mid-Range {t21_txt}{d_str}", "tone": "emerald" if (t2108 and t2108 >= 45) else "slate"}

    if n_rows >= 5:
        if rolloff_fragile:
            p4 = {"label": "Next 5D Roll-Off", "value": f"⏳ Risk: {drop_dt} (+{drop_u}/-{drop_d}) → {ex_drop_ratio:.2f}", "tone": "amber"}
        elif ex_drop_ratio >= 2.0:
            p4 = {"label": "Next 5D Roll-Off", "value": f"🧱 Durable ({ex_drop_ratio:.2f} ex-{drop_dt})", "tone": "emerald"}
        else:
            p4 = {"label": "Next 5D Roll-Off", "value": f"📅 {drop_dt} (+{drop_u}/-{drop_d}) → {ex_drop_ratio:.2f}", "tone": "slate"}
    else:
        p4 = {"label": "Next 5D Roll-Off", "value": "Insufficient 5D history", "tone": "slate"}

    return {
        "verdict": verdict,
        "posture": posture,
        "tone": tone,
        "playbook": playbook,
        "pills": [p1, p2, p3, p4],
    }


def render_stockbee_monitor_grid_html(
    df_grid,
    *,
    universe_label: str = "Nifty Total Market (750)",
    show_insight: bool = True,
) -> str:
    """Render an institutional zero-indent Stockbee Market Monitor HTML heatmap matrix."""
    import html as _html
    import pandas as _pd

    if df_grid is None or not isinstance(df_grid, _pd.DataFrame) or df_grid.empty:
        return ""

    u_lbl = _html.escape(str(universe_label or "Universe"))
    latest = df_grid.iloc[0]
    lat_u4 = int(latest.get("Up_4_Pct", 0))
    lat_d4 = int(latest.get("Down_4_Pct", 0))
    lat_r5_txt, lat_r5_col, _ = _format_stockbee_ratio_cell(
        float(latest.get("Ratio_5D", 0.0)), int(latest.get("Up_5D_Sum", 0)), int(latest.get("Down_5D_Sum", 0))
    )
    lat_r10_txt, lat_r10_col, _ = _format_stockbee_ratio_cell(
        float(latest.get("Ratio_10D", 0.0)), int(latest.get("Up_10D_Sum", 0)), int(latest.get("Down_10D_Sum", 0))
    )
    lat_q_spread = int(latest.get("Up_25_Qtr", 0)) - int(latest.get("Down_25_Qtr", 0))
    lat_t2108 = latest.get("T2108_Pct", float("nan"))
    lat_t2108_txt = f"{float(lat_t2108):.1f}%" if _pd.notna(lat_t2108) else "—"
    lat_pulse = _html.escape(str(latest.get("Pulse", "⚖️ BALANCED")))
    lat_valid = int(latest.get("Valid_Stocks", 0))

    tone_styles = {
        "emerald": ("#34d399", "rgba(16,185,129,0.14)", "rgba(16,185,129,0.40)"),
        "amber": ("#fbbf24", "rgba(245,158,11,0.14)", "rgba(245,158,11,0.40)"),
        "rose": ("#f87171", "rgba(239,68,68,0.14)", "rgba(239,68,68,0.40)"),
        "cyan": ("#38bdf8", "rgba(56,189,248,0.14)", "rgba(56,189,248,0.40)"),
        "slate": ("#cbd5e1", "rgba(148,163,184,0.10)", "rgba(148,163,184,0.28)"),
    }
    insight_html = ""
    if show_insight:
        ins = analyze_stockbee_regime_insight(df_grid)
        if ins:
            v_fg, v_bg, v_bd = tone_styles.get(ins["tone"], tone_styles["slate"])
            pills_markup = []
            for pl in ins["pills"]:
                p_fg, p_bg, p_bd = tone_styles.get(pl["tone"], tone_styles["slate"])
                pills_markup.append(
                    f'<span style="display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:6px;'
                    f'background:{p_bg};border:1px solid {p_bd};font-size:0.72rem;color:#cbd5e1;">'
                    f'<span style="color:#94a3b8;font-weight:600;">{_html.escape(pl["label"])}:</span> '
                    f'<b style="color:{p_fg};font-family:\'JetBrains Mono\',monospace;">{_html.escape(pl["value"])}</b></span>'
                )
            insight_html = (
                f'<div style="background:rgba(15,23,42,0.72);border:1px solid {v_bd};border-left:4px solid {v_fg};'
                'border-radius:9px;padding:10px 12px;margin-bottom:10px;">'
                '<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:6px;">'
                '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
                f'<span style="padding:3px 10px;border-radius:999px;background:{v_bg};border:1px solid {v_bd};color:{v_fg};font-size:0.76rem;font-weight:800;letter-spacing:0.03em;">{_html.escape(ins["verdict"])}</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.74rem;font-weight:700;color:#f8fafc;background:rgba(255,255,255,0.06);padding:2px 8px;border-radius:5px;">Posture: {_html.escape(ins["posture"])}</span>'
                '</div></div>'
                f'<div style="font-size:0.78rem;color:#e2e8f0;line-height:1.45;margin-bottom:8px;">{_html.escape(ins["playbook"])}</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:6px;">{"".join(pills_markup)}</div>'
                '</div>'
            )

    rows_html = []
    for _, r in df_grid.iterrows():
        dt_val = r["Date"]
        dt_str = dt_val.strftime("%b %d") if hasattr(dt_val, "strftime") else _html.escape(str(dt_val)[:10])
        v_cnt = max(1, int(r.get("Valid_Stocks", 1)))
        u4 = int(r.get("Up_4_Pct", 0))
        d4 = int(r.get("Down_4_Pct", 0))
        r5_txt, r5_col, r5_bg = _format_stockbee_ratio_cell(
            float(r.get("Ratio_5D", 0.0)), int(r.get("Up_5D_Sum", 0)), int(r.get("Down_5D_Sum", 0))
        )
        r10_txt, r10_col, r10_bg = _format_stockbee_ratio_cell(
            float(r.get("Ratio_10D", 0.0)), int(r.get("Up_10D_Sum", 0)), int(r.get("Down_10D_Sum", 0))
        )
        qu = int(r.get("Up_25_Qtr", 0))
        qd = int(r.get("Down_25_Qtr", 0))
        t21 = r.get("T2108_Pct", float("nan"))
        pulse = str(r.get("Pulse", "⚖️ BALANCED"))

        u4_ratio = u4 / v_cnt
        d4_ratio = d4 / v_cnt
        u4_bg = "rgba(16,185,129,0.28)" if u4_ratio >= 0.04 else ("rgba(16,185,129,0.13)" if u4_ratio >= 0.02 else "transparent")
        u4_fg = "#34d399" if u4_ratio >= 0.02 else "#e2e8f0"
        d4_bg = "rgba(239,68,68,0.28)" if d4_ratio >= 0.04 else ("rgba(239,68,68,0.13)" if d4_ratio >= 0.02 else "transparent")
        d4_fg = "#f87171" if d4_ratio >= 0.02 else "#cbd5e1"

        qu_fg = "#34d399" if qu > qd * 1.25 else "#e2e8f0"
        qu_bg = "rgba(16,185,129,0.12)" if qu > qd * 1.25 else "transparent"
        qd_fg = "#f87171" if qd > qu else "#cbd5e1"
        qd_bg = "rgba(239,68,68,0.14)" if qd > qu else "transparent"

        if _pd.notna(t21):
            t21_f = float(t21)
            t21_str = f"{t21_f:.1f}%"
            if t21_f < 20.0:
                t21_fg, t21_bg = "#38bdf8", "rgba(56,189,248,0.16)"
            elif t21_f > 75.0:
                t21_fg, t21_bg = "#fbbf24", "rgba(245,158,11,0.18)"
            elif t21_f >= 50.0:
                t21_fg, t21_bg = "#34d399", "rgba(16,185,129,0.13)"
            else:
                t21_fg, t21_bg = "#fbbf24", "transparent"
        else:
            t21_str, t21_fg, t21_bg = "—", "#64748b", "transparent"

        if "THRUST" in pulse:
            p_fg, p_bg = "#34d399", "rgba(16,185,129,0.20)"
        elif "SELL" in pulse:
            p_fg, p_bg = "#f87171", "rgba(239,68,68,0.20)"
        else:
            p_fg, p_bg = "#94a3b8", "rgba(148,163,184,0.10)"

        td_base = "padding:6px 10px;border-bottom:1px solid rgba(255,255,255,0.05);font-family:'JetBrains Mono',monospace;text-align:right;"
        rows_html.append(
            '<tr style="background:rgba(15,23,42,0.35);">'
            f'<td style="{td_base}text-align:left;color:#cbd5e1;font-weight:700;">{dt_str}</td>'
            f'<td style="{td_base}color:{u4_fg};background:{u4_bg};font-weight:700;">{u4}</td>'
            f'<td style="{td_base}color:{d4_fg};background:{d4_bg};font-weight:700;">{d4}</td>'
            f'<td style="{td_base}color:{r5_col};background:{r5_bg};font-weight:800;">{r5_txt}</td>'
            f'<td style="{td_base}color:{r10_col};background:{r10_bg};font-weight:800;">{r10_txt}</td>'
            f'<td style="{td_base}color:{qu_fg};background:{qu_bg};">{qu}</td>'
            f'<td style="{td_base}color:{qd_fg};background:{qd_bg};">{qd}</td>'
            f'<td style="{td_base}color:{t21_fg};background:{t21_bg};font-weight:700;">{t21_str}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid rgba(255,255,255,0.05);text-align:center;">'
            f'<span style="display:inline-block;padding:2px 7px;border-radius:4px;font-size:0.70rem;font-weight:800;color:{p_fg};background:{p_bg};white-space:nowrap;">{_html.escape(pulse)}</span>'
            '</td></tr>'
        )

    th_base = "padding:8px 10px;font-weight:700;color:#94a3b8;font-size:0.70rem;text-transform:uppercase;letter-spacing:0.04em;border-bottom:1px solid rgba(255,255,255,0.12);position:sticky;top:0;background:#0b1120;z-index:2;text-align:right;white-space:nowrap;"
    q_col = "#34d399" if lat_q_spread >= 0 else "#f87171"

    return (
        '<div style="background:linear-gradient(135deg,rgba(15,23,42,0.90),rgba(2,6,23,0.96));'
        'border:1px solid rgba(56,189,248,0.22);border-radius:12px;padding:12px 14px;margin:8px 0 18px 0;">'
        '<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px;">'
        f'<div style="font-size:0.78rem;color:#94a3b8;">Universe: <b style="color:#f8fafc;">{u_lbl}</b> ({lat_valid} valid) · Trailing <b style="color:#38bdf8;">{len(df_grid)} sessions</b></div>'
        '<div style="display:flex;gap:14px;flex-wrap:wrap;font-family:\'JetBrains Mono\',monospace;font-size:0.76rem;">'
        f'<span>Today 4%: <b style="color:#34d399;">+{lat_u4}</b> / <b style="color:#f87171;">-{lat_d4}</b></span>'
        f'<span>5D Ratio: <b style="color:{lat_r5_col};">{lat_r5_txt}</b></span>'
        f'<span>10D Ratio: <b style="color:{lat_r10_col};">{lat_r10_txt}</b></span>'
        f'<span>Qtr Spread: <b style="color:{q_col};">{lat_q_spread:+d}</b></span>'
        f'<span>T2108: <b style="color:#38bdf8;">{lat_t2108_txt}</b></span>'
        f'<span>Pulse: <b style="color:#f8fafc;">{lat_pulse}</b></span>'
        '</div></div>'
        f'{insight_html}'
        '<div style="overflow-x:auto;max-height:520px;overflow-y:auto;border:1px solid rgba(255,255,255,0.08);border-radius:8px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:0.78rem;">'
        '<thead><tr>'
        f'<th style="{th_base}text-align:left;">Date</th>'
        f'<th style="{th_base}">Up 4%</th>'
        f'<th style="{th_base}">Down 4%</th>'
        f'<th style="{th_base}">5D Ratio</th>'
        f'<th style="{th_base}">10D Ratio</th>'
        f'<th style="{th_base}">Up 25% Qtr</th>'
        f'<th style="{th_base}">Dn 25% Qtr</th>'
        f'<th style="{th_base}">T2108 (&gt;40 SMA)</th>'
        f'<th style="{th_base}text-align:center;">Stockbee Pulse</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table></div>'
        '<div style="margin-top:7px;font-size:0.71rem;color:#64748b;display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;">'
        '<span>📌 <b>Thrust Rule:</b> 5D Ratio &ge; 2.00 = accumulation regime; Day Thrust &ge; 3.5% of universe up &ge; 4% (2:1+ vs bears).</span>'
        '<span>📌 <b>T2108 (% &gt; 40 SMA):</b> &lt;20% oversold bounce zone · 50–75% bull sweet spot · &gt;75% short-term stretched.</span>'
        '</div></div>'
    )
