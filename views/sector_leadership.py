import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Ensure root path is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from styles import load_css
    load_css()
except ImportError:
    pass

from components import render_header, render_metric_card
from database import get_all_fundamentals_cache
from industry_matrix import calculate_industry_leadership, calculate_industry_strength_cycle
from sync_sectors import sync_industries_ui
from views.true_market_leader import get_cached_universe, get_cached_history, compute_rs_scores_fast
from database import save_sector_leadership



from components import render_disk_cache_sidebar
render_disk_cache_sidebar(get_cached_universe)

def main():
    render_header("🌊 Sector & Theme Leadership", "Tracking systemic smart money flows and institutional accumulation across industry groups.")
    
    st.markdown("""
    *Elite growth traders know that 50% of an individual stock's move is dictated by its industry. This module measures the True Market Relative Strength of the entire universe, normalizes the data, and ranks industries by their underlying momentum. Look for sectors displaying both high Average RS and a large concentration of Breakout Leaders (RS > 80).*
    """)
    
    with st.expander("📖 Methodology: How the Sector Engine Works"):
        st.markdown("""
        **1. Cross-Sectional Relative Strength (RS):**
        We calculate a composite momentum score for every single stock in the 2,500+ ticker universe. This is driven by trailing structural returns across specific timeframes, weighted toward recent action:
        
        <code style="color: #6ee7b7; background-color:#1e293b; padding: 4px; display:block; text-align:center; font-size: 1.1em; border-radius: 5px; border: 1px solid #475569;">
            Raw RS = (1-Month Return * 40%) + (3-Month Return * 35%) + (6-Month Return * 25%)
        </code><br>
        
        Instead of plotting raw percentages, we strictly normalize these `Raw RS` returns cross-sectionally into a **0 to 99 percentile rank** (matching William O'Neil's proprietary CAN SLIM mechanism). 
        *Formula: `Percentile_Rank(Raw RS) * 100`*. An RS of 99 means the stock is mathematically outperforming 99% of the universe right now.
        
        **2. Average Sector RS:**
        The engine identifies the specific institutional *Yahoo Finance* categorizations for all 2,500+ stocks. It then maps the stocks to their respective industries and calculates the **mean Relative Strength** (Avg RS) across all constituents. This reveals where the broadest "market tide" is lifting boats.
        
        **3. Stage 2 Breakout Count (Institutional Accumulation):**
        Average RS is great, but it can be skewed by one micro-cap exploding 500%. 
        To confirm true institutional participation, we run a David Ryan-style filter: We count exactly how many individual stocks within that sector possess the elite **RS > 80** threshold. If an industry has 10+ stocks boasting an RS strictly above 80, it confirms massive, coordinated hedge fund and mutual fund accumulation ("Peak Accumulation").
        """, unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # DATABASE BREADTH HEALTH
    # -------------------------------------------------------------------------
    db_cache = get_all_fundamentals_cache()
    
    # We load the universe purely to count here
    universe_tickers = get_cached_universe("Deep Market (2500+ NSE Stocks)")
    universe_size = len(universe_tickers)
    universe_set = set(universe_tickers)
    
    # Only count industries for tickers actually in the current universe
    known_industries = len([k for k, v in db_cache.items() if k in universe_set and v.get('industry') and v.get('industry') != 'Unknown'])
    
    with st.expander(f"🗄️ Database Breadth Health ({known_industries}/{universe_size} Stocks Mapped)", expanded=(known_industries < 1000)):
        st.write(f"The Sector Engine currently has precise Yahoo Finance industry classifications for **{known_industries}** stocks out of a theoretical **{universe_size}** universe.")
        st.caption("Yahoo Finance strictly rate-limits automated data mapping. By clicking the button below, the app will incrementally fetch the missing industries in the background and permanently cache them. It usually successfully adds ~350 stocks at a time before Yahoo enforces a cool-down block.")
        
        if known_industries < universe_size:
            if st.button("📥 Incrementally Fetch Missing Industries", type="secondary"):
                try:
                    sync_industries_ui()
                    st.success("Incremental sync completed. Your breadth tracking capability has expanded!")
                except Exception as e:
                    st.error(f"Industry sync encountered an error: {e}. Try again later — Yahoo Finance may be rate-limiting your IP.")
                st.rerun()
        else:
            st.success("You have achieved 100% Sector Mapping Coverage!")
            
    c1, c2 = st.columns([1, 4])
    with c1:
        run_scan = st.button("🚀 Analyze Sector Breadth", type="primary", use_container_width=True)
    with c2:
        if st.button("🔄 Force Refresh Cache"):
            st.cache_data.clear()
            st.rerun()

    if run_scan:
        with st.status("Aggregating Universal Data Matrix...", expanded=True) as status:
            st.write("1. Loading comprehensive market universe...")
            tickers = get_cached_universe("Deep Market (2500+ NSE Stocks)")
            
            st.write("2. Downloading fast technical history for structural momentum calculation...")
            # We need ~250 bars to reliably calculate full (1M, 3M, 6M) RS composites natively
            history_df = get_cached_history(tickers, days=250)
            
            st.write("3. Ranking absolute Cross-Sectional Relative Strength (0-99)...")
            rs_scores = compute_rs_scores_fast(history_df, tickers)
            
            st.write("4. Retrieving dynamic local Institutional Mapping Table...")
            db_cache = get_all_fundamentals_cache()
            
            # Build the industry mapping dictionary natively from the fundamentals cache
            # This ensures we strictly retain Yahoo Finance's hyper-specific categorizations
            industry_map = {}
            for t, data in db_cache.items():
                ind = data.get('industry', 'Unknown')
                if ind and ind != 'Unknown':
                    industry_map[t] = ind
            
            # Some tickers might be stripped of .NS, map them too safely
            for t, data in db_cache.items():
                ind = data.get('industry', 'Unknown')
                if ind and ind != 'Unknown':
                    industry_map[t.replace('.NS', '')] = ind
                    
            st.write("5. Distilling Sector Vectors & Identifying Thematic Rotation...")
            industry_df = calculate_industry_leadership(history_df, tickers, industry_map, rs_scores)
            
            st.write("6. Synthesizing Institutional Strength Cycle...")
            try:
                cycle_df = calculate_industry_strength_cycle(history_df, tickers, industry_map)
            except Exception as e:
                cycle_df = pd.DataFrame()
            
            st.session_state.sector_matrix = industry_df
            st.session_state.sector_cycle_df = cycle_df
            st.session_state.sector_rs_scores = rs_scores
            st.session_state.sector_map = industry_map
            st.session_state.sector_history_df = history_df
            
            # Save the Top 5 sectors to the database for Intraday Monitor badge cross-referencing
            if not industry_df.empty:
                top_5_sectors = industry_df.head(5).to_dict('records')
                save_sector_leadership(top_5_sectors)
            try:
                st.session_state.latest_market_date = history_df.index.max()
            except Exception:
                st.session_state.latest_market_date = None
            
            status.update(label="Sector Breadth Analysis Complete!", state="complete", expanded=False)
            
    # -------------------------------------------------------------------------
    # RENDER RESULTS
    # -------------------------------------------------------------------------
    if 'sector_matrix' in st.session_state and not st.session_state.sector_matrix.empty:
        df = st.session_state.sector_matrix
        cycle_df = st.session_state.get('sector_cycle_df')
        rs_scores = st.session_state.sector_rs_scores
        ind_map = st.session_state.sector_map
        history_df = st.session_state.get('sector_history_df')
        
        # Safe fallback block on hot-reload
        if history_df is None or history_df.empty:
            st.warning("Historical data matrix dropped from RAM cache. Please press 'Analyze Sector Breadth' to rebuild the memory payload.")
            return

        # ---------------------------------------------------------------------
        # FOCUS MODE VIEW SWAP (Drill-Down)
        # ---------------------------------------------------------------------
        if st.session_state.get('focus_mode', False) and st.session_state.get('focus_industry'):
            def exit_focus():
                st.session_state.focus_mode = False
                
            st.button("◀ Back to Market Scans", on_click=exit_focus)
            st.markdown(f"## {st.session_state.focus_industry}")
            st.caption("Sector constituents ranked against the entire market universe by Intermediate Structural Momentum.")
            
            # --- Filters inside Focus Mode ---
            col_foc1, col_foc2, col_foc3 = st.columns([1, 1, 1])
            with col_foc1:
                search_query = st.text_input("🔍 Search for a company...", key="focus_search")
            with col_foc2:
                # We will populate the options dynamically after generating the dataframe
                status_placeholder = st.empty()
            with col_foc3:
                min_score = st.slider(
                    "Minimum RS Score",
                    min_value=0.0,
                    max_value=99.0,
                    value=0.0,
                    step=5.0,
                    key="focus_score_sl"
                )
            
            # Reverse map
            rev_map = {}
            for t, ind in ind_map.items():
                if ind not in rev_map: rev_map[ind] = []
                rev_map[ind].append(t)
                
            focus_ind = st.session_state.focus_industry
            if focus_ind in rev_map:
                industry_tickers = rev_map[focus_ind]
                
                with st.spinner("Scoring constituents against broader market..."):
                    from industry_matrix import calculate_constituent_strength_cycle
                    focus_df = calculate_constituent_strength_cycle(history_df, industry_tickers)
                    
                if not focus_df.empty:
                    # Apply Status Filter
                    status_options = sorted(focus_df['Status'].unique().tolist())
                    selected_statuses = status_placeholder.multiselect(
                        "Filter by Cycle Status",
                        options=status_options,
                        default=[],
                        key="focus_status_sel"
                    )
                    
                    if selected_statuses:
                        focus_df = focus_df[focus_df['Status'].isin(selected_statuses)]
                        
                    focus_df = focus_df[focus_df['Score'] >= min_score]
                    
                    if search_query:
                        focus_df = focus_df[focus_df['Ticker'].str.contains(search_query, case=False)]
                        
                    db_cache = get_all_fundamentals_cache()
                    names = []
                    urls = []
                    
                    for t in focus_df['Ticker']:
                        clean_t = t.replace('.NS', '')
                        info = db_cache.get(t, db_cache.get(clean_t, {}))
                        comp_name = info.get('shortName', info.get('longName', clean_t))
                        
                        tv_url = f"https://in.tradingview.com/chart/?symbol=NSE:{clean_t}"
                        names.append(comp_name)
                        urls.append(tv_url)
                        
                    focus_df['Company'] = names
                    focus_df['Chart'] = urls
                    focus_df = focus_df.reset_index(drop=True)
                    focus_df.index += 1
                    focus_df['#'] = focus_df.index
                    
                    render_df = focus_df[['#', 'Company', 'Chart', 'Score', 'Status', 'Score_Change_1M', 'Scores_Array']]
                    
                    def color_status(val):
                        color = 'green'
                        if val == 'OUTPERFORMING': color = '#16a34a'
                        elif val == 'ACCUMULATING': color = '#2563eb'
                        elif val == 'CONSOLIDATING': color = '#d97706'
                        elif val == 'UNDERPERFORMING': color = '#dc2626'
                        return f'color: {color}; font-weight: bold;'
                        
                    st.dataframe(
                        render_df.style.format({
                            'Score': "{:.2f}",
                            'Score_Change_1M': "{:+.2f}"
                        }).map(color_status, subset=['Status']),
                        column_config={
                            "#": st.column_config.NumberColumn("#", format="%d"),
                            "Company": st.column_config.TextColumn("Company"),
                            "Chart": st.column_config.LinkColumn("TradingView", display_text="Open 📈"),
                            "Score": st.column_config.NumberColumn("Score", format="%.2f"),
                            "Status": st.column_config.TextColumn("Status"),
                            "Score_Change_1M": st.column_config.NumberColumn("Score Change 1M", format="%+.2f"),
                            "Scores_Array": st.column_config.LineChartColumn("1M Scores Sparkline", y_min=0.0, y_max=99.0)
                        },
                        use_container_width=True,
                        hide_index=True,
                        height=600
                    )
                else:
                    st.warning("Not enough data to calculate component cycle scores.")
                    
            # STOP execution of the main dashboard UI
            return

        st.markdown("---")
        
        if 'latest_market_date' in st.session_state and st.session_state.latest_market_date is not None:
            try:
                formatted_date = st.session_state.latest_market_date.strftime('%A, %B %d, %Y')
                st.caption(f"🕒 **Market Data Current As Of:** `{formatted_date}`")
            except Exception:
                pass
                
        # 1. Executive Summary Cards
        m1, m2, m3, m4 = st.columns(4)
        top_sector_rs = df.iloc[0]['Industry']
        top_rs_value = df.iloc[0]['Avg_RS']
        
        top_breakout_df = df.sort_values(by='Leaders_80_Plus', ascending=False)
        top_breakout_sector = top_breakout_df.iloc[0]['Industry']
        top_breakout_count = top_breakout_df.iloc[0]['Leaders_80_Plus']
        
        total_sectors = len(df)
        
        # Determine Coldest Sector that has at least 5 constituents
        cold_df = df[df['Constituent_Count'] >= 5]
        coldest_sector = cold_df.iloc[-1]['Industry'] if len(cold_df) > 0 else "N/A"
        
        with m1:
            render_metric_card("Dominant Momentum Theme", top_sector_rs, color_class="green-text")
        with m2:
            render_metric_card("Highest Average RS", f"{top_rs_value:.1f}", color_class="green-text")
        with m3:
            render_metric_card("Peak Accumulation", f"{top_breakout_sector} ({top_breakout_count} Leaders)", color_class="green-text")
        with m4:
            render_metric_card("Distribution Vector", coldest_sector, color_class="red-text")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        if cycle_df is not None and not cycle_df.empty:
            st.markdown("### 🔄 Market Scans: Industry Strength Cycle")
            st.caption("Categorizes industries based on 1-Month score velocity to immediately spot OUTPERFORMING vs ACCUMULATING rotation themes.")
            
            with st.expander("📖 Methodology: Composite Scores and Cycle Status"):
                st.markdown("""
                **1. The Synthetic Score (0-99)**
                Instead of using simple moving averages, the engine dynamically calculates a *Synthetic Index* for every industry by dollar-weighting its constituents. We then measure short-term and structural momentum across this index, and **cross-sectionally rank** it against all other industries into a 0 to 99 percentile score. 
                *A score of 99 means the industry is outperforming 99% of other sectors on a risk-adjusted momentum basis.*
                
                **2. The 1M Velocity and Cycle Status**
                Absolute scores tell us where a sector is *today*, but the **Score Change 1M** tells us where it's *going*. The algorithm tracks the exact score from 21-trading days ago to compute velocity. This generates the cycle statuses:
                
                - 🟢 **OUTPERFORMING:** High Current Score (>= 60) AND positive momentum velocity (Change > 5). Primary Stage 2 structural leaders absorbing market liquidity.
                - 🔵 **ACCUMULATING:** Low Current Score (< 60) BUT experiencing violent positive velocity (Change > 10). Occurs at major market bottoms or deep base building phases. It isolates exactly where "Smart Money" is quietly rotating into a beaten-down sector *before* it triggers widespread breakout alerts.
                - 🟠 **CONSOLIDATING:** High Current Score (>= 60) BUT stalling or losing velocity (Change <= 5). The primary trend is intact but the sector is resting or forming flags.
                - 🔴 **UNDERPERFORMING:** Low Score and decaying velocity. Avoid these distribution zones entirely.
                """)
                
            col_cycl_f1, col_cycl_f2, col_cycl_f3 = st.columns([1, 1, 1])
            with col_cycl_f1:
                max_cycl = len(cycle_df)
                top_n_cycl = st.slider(
                    "Filter Scans (Top 'N')", 
                    min_value=5, 
                    max_value=max_cycl, 
                    value=min(40, max_cycl), 
                    step=5,
                    key="cycl_slider"
                )
            with col_cycl_f2:
                status_options = sorted(cycle_df['Status'].unique().tolist())
                selected_statuses = st.multiselect(
                    "Filter by Cycle Status",
                    options=status_options,
                    default=[],
                    key="cycl_status_sel"
                )
            with col_cycl_f3:
                min_score = st.slider(
                    "Minimum Target Score",
                    min_value=0.0,
                    max_value=99.0,
                    value=0.0,
                    step=5.0,
                    key="cycl_score_sl"
                )
                
            disp_cycle = cycle_df.copy()
            if selected_statuses:
                disp_cycle = disp_cycle[disp_cycle['Status'].isin(selected_statuses)]
            disp_cycle = disp_cycle[disp_cycle['Score'] >= min_score]
            disp_cycle = disp_cycle.head(top_n_cycl)
            
            # Helper to style the status dynamically
            def color_status(val):
                color = 'green'
                if val == 'OUTPERFORMING': color = '#16a34a'
                elif val == 'ACCUMULATING': color = '#2563eb'
                elif val == 'CONSOLIDATING': color = '#d97706'
                elif val == 'UNDERPERFORMING': color = '#dc2626'
                return f'color: {color}; font-weight: bold;'
                
            def highlight_cycle_rows(row):
                change = row.get('Score_Change_1M', 0)
                if pd.notna(change):
                    if change >= 10.0:
                        return ['background-color: rgba(22, 163, 74, 0.15); color: #86efac'] * len(row)
                    elif change <= -10.0:
                        return ['background-color: rgba(220, 38, 38, 0.15); color: #fca5a5'] * len(row)
                return [''] * len(row)
            
            st.info("💡 **Color Guide:** 🟩 **Green Row** = Surging Velocity (Score +10) | 🟥 **Red Row** = Collapsing Velocity (Score -10)")
            
            import streamlit as st_module
            st_ver = getattr(st_module, '__version__', '1.0.0').split('.')
            is_modern_st = False
            try:
                if int(st_ver[0]) > 1 or (int(st_ver[0]) == 1 and int(st_ver[1]) >= 35):
                    is_modern_st = True
            except:
                pass
                
            dt_kwargs = {
                "use_container_width": True,
                "hide_index": True,
                "height": 450,
                "column_order": ['Rank', 'Industry', 'Constituents', 'Score', 'Status', 'Score_Change_1M', 'Scores_Array']
            }
            if is_modern_st:
                dt_kwargs["selection_mode"] = "single-row"
                dt_kwargs["on_select"] = "rerun"
                dt_kwargs["key"] = "macro_scan_dt"
                
            event = st.dataframe(
                disp_cycle.style.format({
                    'Score': "{:.2f}",
                    'Score_Change_1M': "{:+.2f}"
                }).map(color_status, subset=['Status']).apply(highlight_cycle_rows, axis=1),
                column_config={
                    "Rank": st.column_config.NumberColumn("Hierarchy", format="%d"),
                    "Industry": st.column_config.TextColumn("Industry"),
                    "Constituents": st.column_config.NumberColumn("Companies Count", format="%d"),
                    "Score": st.column_config.NumberColumn("Composite Score (0-99)", format="%.2f"),
                    "Status": st.column_config.TextColumn("Cycle Status"),
                    "Score_Change_1M": st.column_config.NumberColumn("Score Change 1M", format="%.2f"),
                    "Scores_Array": st.column_config.LineChartColumn("1M Scores Sparkline", y_min=0.0, y_max=99.0)
                },
                **dt_kwargs
            )
            
            if is_modern_st and isinstance(event, dict) and 'selection' in event:
                rows = event['selection'].get('rows', [])
                if rows:
                    selected_row = rows[0]
                    ind_name = disp_cycle.iloc[selected_row]['Industry']
                    if st.session_state.get('focus_industry') != ind_name or not st.session_state.get('focus_mode', False):
                        st.session_state.focus_mode = True
                        st.session_state.focus_industry = ind_name
                        st.rerun()
                        
            st.markdown("<br><hr>", unsafe_allow_html=True)
        
        # 2. Main Sector Hierarchy Table
        st.markdown("### 🏆 Industry Group Power Rankings")
        
        display_df = df.copy()
        
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            max_sectors = len(display_df)
            default_val = min(30, max_sectors)
            top_n = st.slider(
                "Filter Hierarchy (Top 'N')", 
                min_value=5, 
                max_value=max_sectors, 
                value=default_val, 
                step=5
            )
            
        display_df = display_df.head(top_n)
        
        # We cap progress bars safely
        max_rs = 100.0
        max_leaders = float(display_df['Leaders_80_Plus'].max()) if not display_df.empty else 10.0
        
        st.dataframe(
            display_df.style.format({
                'Avg_RS': "{:.1f}",
                'Max_RS': "{:.1f}",
                'Participation_%': "{:.1f}%"
            }),
            column_config={
                "Rank": st.column_config.NumberColumn("Hierarchy", format="%d"),
                "Industry": st.column_config.TextColumn("Industry/Sector Theme"),
                "Avg_RS": st.column_config.ProgressColumn("Average Sector RS", format="%.1f", min_value=0.0, max_value=max_rs),
                "Leaders_80_Plus": st.column_config.NumberColumn("Stage 2 Count", format="%d"),
                "Participation_%": st.column_config.ProgressColumn("Breadth Participation %", format="%.1f%%", min_value=0.0, max_value=100.0),
                "Constituent_Count": st.column_config.NumberColumn("Total Traded Stocks", format="%d"),
                "Max_RS": st.column_config.NumberColumn("Highest Ticker RS", format="%.1f"),
            },
            use_container_width=True,
            hide_index=True,
            height=450
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 3. Sector Distribution Quadrant (Swarm Plot)
        st.markdown("### 🔭 The Minervini Sector Quadrant")
        st.caption("Visually maps Breakout Clusters across top industries. The elite sectors possess dense vertical clusters (high count) of large (liquid) green bubbles above the Stage-2 threshold (RS > 80).")
        
        try:
            # Fast Vectorized Liquidity Calculation for all 2500 tickers simultaneously
            vol_panel = history_df.xs('Volume', level=1, axis=1)
            close_panel = history_df.xs('Close', level=1, axis=1)
            vol_21 = vol_panel.tail(21).mean()
            # Handle potential empty series gracefully
            close_1 = close_panel.iloc[-1] if not close_panel.empty else pd.Series()
            adtv_s = (vol_21 * close_1) / 10000000.0
            
            swarm_data = []
            top_industries = display_df['Industry'].tolist()
            
            for t, industry in ind_map.items():
                if industry in top_industries:
                    clean_t = t.replace('.NS', '')
                    lookup_t = f"{clean_t}.NS"
                    rs = rs_scores.get(lookup_t, rs_scores.get(clean_t, rs_scores.get(t, -1)))
                    
                    if rs >= 0:
                        liq = adtv_s.get(lookup_t, adtv_s.get(clean_t, 0.0))
                        if pd.notna(liq) and liq >= 0:
                            swarm_data.append({
                                'Ticker': f"NSE:{clean_t}",
                                'Industry': industry,
                                'RS_Score': round(rs, 1),
                                'ADTV_Cr': float(liq)
                            })
                            
            if swarm_data:
                plot_df = pd.DataFrame(swarm_data)
                # Max bubble size cap to prevent a 10,000 CR ADTV stock eclipsing the screen
                plot_df['Size_Capped'] = plot_df['ADTV_Cr'].clip(upper=500) 
                
                # Enforce X-Axis categorical order strictly matching the Power Rankings (Left-to-Right)
                plot_df['Industry'] = pd.Categorical(plot_df['Industry'], categories=top_industries, ordered=True)
                
                fig = px.scatter(
                    plot_df,
                    x='Industry',
                    y='RS_Score',
                    size='Size_Capped',
                    color='RS_Score',
                    hover_name='Ticker',
                    hover_data={'ADTV_Cr': ':.1f', 'Size_Capped': False, 'Industry': True},
                    color_continuous_scale='RdYlGn',
                    labels={
                        'RS_Score': 'Relative Strength (0-99)',
                        'ADTV_Cr': 'ADTV (₹ Crores)',
                        'Industry': 'Sector / Theme'
                    }
                )
                fig.update_layout(
                    height=550, 
                    template='plotly_dark',
                    xaxis_title="", 
                    xaxis={'categoryorder':'array', 'categoryarray':top_industries, 'tickangle': -45},
                    margin=dict(b=120)
                )
                # Paint the Stage 2 "Breakout Zone" above RS 80
                fig.add_shape(
                    type="rect",
                    x0=-0.5, y0=80, x1=len(top_industries)-0.5, y1=100,
                    line=dict(color="rgba(0,0,0,0)"),
                    fillcolor="rgba(50, 205, 50, 0.05)",
                    layer="below"
                )
                fig.add_hline(y=80, line_dash="dash", line_color="rgba(50, 205, 50, 0.4)", annotation_text="Elite Stage 2 Zone (RS > 80)", annotation_position="top right")
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📖 How to Interpret The Swarm Plot Signature"):
                    st.markdown("""
                    **The Institutional Footprint:**
                    Elite momentum traders (like Mark Minervini) don't look for strong sectors using basic averages. They visually hunt for **Breakout Clusters**. 
                    
                    * **The Bubbles:** Every vertical column is an Industry Group. Every single bubble is a stock within that group.
                    * **Y-Axis (Height):** This is the stock's Relative Strength. You want industries where the entire 'stack' of bubbles is shifted upwards.
                    * **Bubble Size (Liquidity):** The size of the bubble represents the stock's Average Daily Trading Value (ADTV). Massive bubbles equal massive institutional liquidity.
                    
                    **🎯 The Golden Setup:**
                    You are hunting for columns on the far left that exhibit a dense canopy of **massive green bubbles** securely penetrating the shaded "Elite Stage 2 Zone" (RS > 80). 
                    If you see 10 giant green bubbles grouped tightly at the top of an industry stack, that is undeniable proof that hedge funds are systematically accumulating the most important megacaps in that thematic group.
                    
                    **⚠️ The Warning Sign:**
                    Watch out for 'Fake' sectors. If an industry ranks highly on the table, but its upper canopy in this chart consists entirely of tiny, illiquid micro-cap dots, it means retail traders are illegally pumping penny stocks, while the large institutional bubbles in that same column are languishing at the bottom in the red zone. Avoid these sectors completely!
                    """)
        except Exception as e:
            st.error(f"Could not render swarm plot: {e}")
            
        st.markdown("<br><hr>", unsafe_allow_html=True)
        
        # 4. Global Master Scans Array
        st.markdown("### 🌍 Global Master Scans: True Market Leaders")
        st.caption("Cross-sectional momentum rankings for all tracked constituents across every sector simultaneously. Discover true market leaders independent of industry silos.")
        
        with st.spinner("Scoring global equity universe..."):
            from industry_matrix import calculate_constituent_strength_cycle
            # Pass None to retrieve all tickers natively ranked 0-99 across the whole market
            global_df = calculate_constituent_strength_cycle(history_df, industry_tickers=None)
            
        if not global_df.empty:
            # Map industries
            global_df['Industry'] = global_df['Ticker'].apply(lambda x: ind_map.get(x, "Unknown"))
            
            # Vectorized ADTV for all stocks natively
            try:
                vol_panel = history_df.xs('Volume', level=1, axis=1)
                close_panel = history_df.xs('Close', level=1, axis=1)
                vol_21 = vol_panel.tail(21).mean()
                close_1 = close_panel.iloc[-1]
                adtv_all = (vol_21 * close_1) / 10000000.0
                global_df['ADTV_Cr'] = global_df['Ticker'].map(adtv_all).fillna(0.0)
            except:
                global_df['ADTV_Cr'] = 0.0
            
            # Use columns for filters - Row 1
            r1c1, r1c2, r1c3 = st.columns([1.5, 1.5, 1])
            with r1c1:
                g_search = st.text_input("🔍 Search Company or Industry...", key="global_search")
            with r1c2:
                g_status = st.multiselect("Filter by Cycle Status", options=["OUTPERFORMING", "ACCUMULATING", "CONSOLIDATING", "UNDERPERFORMING"], default=[], key="global_status")
            with r1c3:
                max_ind = 0
                if cycle_df is not None and not cycle_df.empty:
                    max_ind = len(cycle_df)
                
                g_top_ind = st.slider(
                    "Top 'N' Ranked Industries", 
                    min_value=0, 
                    max_value=max_ind if max_ind > 0 else 100, 
                    value=max_ind if max_ind > 0 else 100, 
                    step=5, 
                    help="Filter table to only show stocks from the top N performing sectors. Set to Max to show all.",
                    key="global_ind_limit"
                )
                
            # Row 2 Filters
            r2c1, r2c2, r2c3 = st.columns([1, 1, 1])
            with r2c1:
                g_min = st.slider("Minimum RS Score", 0.0, 99.0, 80.0, 5.0, key="global_min")
            with r2c2:
                g_adtv_min = st.number_input("Minimum ADTV (₹ Crores)", min_value=0.0, max_value=5000.0, value=0.0, step=5.0, help="Filter out low liquidity (penny) stocks.", key="global_adtv_min")
            with r2c3:
                g_r2_min = st.slider("Minimum R² (Trend Quality)", 0.0, 1.0, 0.0, 0.05, help="Filter out choppy, volatile trends. 0.8+ is generally a smooth, persistent trend.", key="global_r2_min")
                
            # Execute Vectorized Filtering
            if cycle_df is not None and not cycle_df.empty and g_top_ind > 0 and g_top_ind < max_ind:
                top_industries = cycle_df.sort_values(by='Score', ascending=False).head(g_top_ind)['Industry'].tolist()
                global_df = global_df[global_df['Industry'].isin(top_industries)]

            if g_status:
                global_df = global_df[global_df['Status'].isin(g_status)]
            global_df = global_df[global_df['Score'] >= g_min]
            global_df = global_df[global_df['ADTV_Cr'] >= g_adtv_min]
            if g_search:
                mask = global_df['Ticker'].str.contains(g_search, case=False) | global_df['Industry'].str.contains(g_search, case=False)
                global_df = global_df[mask]
                
            # Slice early to save computation time on R2 loop
            global_df = global_df.head(300)
            
            # Clean names, URLS, compute Clenow
            db_cache = get_all_fundamentals_cache()
            g_names = []
            g_urls = []
            g_mom = []
            g_r2 = []
            
            from clenow_math import calculate_adjusted_slope
            
            for t in global_df['Ticker']:
                clean_t = t.replace('.NS', '')
                info = db_cache.get(t, db_cache.get(clean_t, {}))
                comp_name = info.get('shortName', info.get('longName', clean_t))
                tv_url = f"https://in.tradingview.com/chart/?symbol=NSE:{clean_t}"
                    
                score = 0.0
                r2 = 0.0
                try:
                    c_series = history_df.xs('Close', level=1, axis=1)[t].dropna()
                    if len(c_series) >= 60:
                        tmp_df = pd.DataFrame({'Close': c_series})
                        c_res = calculate_adjusted_slope(tmp_df, window=90)
                        score = c_res.get('score', 0.0)
                        r2 = c_res.get('r_squared', 0.0)
                except Exception:
                    pass
                
                g_names.append(comp_name)
                g_urls.append(tv_url)
                g_mom.append(score)
                g_r2.append(r2)
                
            global_df['Company'] = g_names
            global_df['Chart'] = g_urls
            global_df['Clenow_Mom'] = g_mom
            global_df['R2'] = g_r2
            
            # Final R2 filter and UI clamp
            global_df = global_df[global_df['R2'] >= g_r2_min]
            global_df = global_df.head(250)
            
            global_df = global_df.reset_index(drop=True)
            global_df.index += 1
            global_df['#'] = global_df.index
            
            render_g_df = global_df[['#', 'Company', 'Industry', 'Chart', 'ADTV_Cr', 'Clenow_Mom', 'R2', 'Score', 'Status', 'Score_Change_1M', 'Scores_Array']]
            
            def color_status_g(val):
                color = 'green'
                if val == 'OUTPERFORMING': color = '#16a34a'
                elif val == 'ACCUMULATING': color = '#2563eb'
                elif val == 'CONSOLIDATING': color = '#d97706'
                elif val == 'UNDERPERFORMING': color = '#dc2626'
                return f'color: {color}; font-weight: bold;'
                
            st.dataframe(
                render_g_df.style.format({
                    'Score': "{:.2f}",
                    'Score_Change_1M': "{:+.2f}",
                    'ADTV_Cr': "{:.1f}",
                    'Clenow_Mom': "{:.1f}",
                    'R2': "{:.2f}"
                }).map(color_status_g, subset=['Status']),
                column_config={
                    "#": st.column_config.NumberColumn("#", format="%d"),
                    "Company": st.column_config.TextColumn("Company"),
                    "Industry": st.column_config.TextColumn("Theme/Industry"),
                    "Chart": st.column_config.LinkColumn("TradingView", display_text="Open 📈"),
                    "ADTV_Cr": st.column_config.NumberColumn("ADTV (Cr)"),
                    "Clenow_Mom": st.column_config.NumberColumn("Clenow SCORE", help="Adjusted Momentum measured via annualized slope * R² over 90 days."),
                    "R2": st.column_config.NumberColumn("R² (Trend Quality)", help="0 to 1 measure of how smooth the 90 day trend is."),
                    "Score": st.column_config.NumberColumn("Comp RS", format="%.2f"),
                    "Status": st.column_config.TextColumn("Status"),
                    "Score_Change_1M": st.column_config.NumberColumn("Score Change 1M", format="%+.2f"),
                    "Scores_Array": st.column_config.LineChartColumn("1M Scores Sparkline", y_min=0.0, y_max=99.0)
                },
                use_container_width=True,
                hide_index=True,
                height=650
            )
        else:
            st.warning("Insufficient data to generate global constituency array.")

if __name__ == "__main__":
    main()

