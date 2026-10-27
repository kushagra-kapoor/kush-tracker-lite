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
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
import streamlit.components.v1 as components

try:
    st.set_page_config(
        page_title="⚡ Earnings Intelligence & PEAD Hub",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
except Exception:
    pass

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history_batch(tickers_tuple, days=90):
    """Fetch price history for a batch of tickers via yfinance."""
    tickers = list(tickers_tuple)
    if not tickers:
        return pd.DataFrame()
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False, threads=True)
        return df
    except Exception:
        return pd.DataFrame()


def compute_rs_scores(history_df, tickers):
    """Compute 3M RS scores (1W*20% + 1M*40% + 3M*40%) ranked 0-99."""
    rs_results = {}
    if history_df.empty:
        return rs_results

    # Handle single vs multi-ticker download format
    if len(tickers) == 1:
        close_df = pd.DataFrame({tickers[0]: history_df['Close']})
    else:
        try:
            close_df = history_df.xs('Close', level=1, axis=1) if history_df.columns.nlevels > 1 else history_df[['Close']]
        except Exception:
            return rs_results

    valid_tickers = [t for t in tickers if t in close_df.columns]
    if not valid_tickers:
        return rs_results

    ret_1w = pd.Series(index=valid_tickers, dtype=float)
    ret_1m = pd.Series(index=valid_tickers, dtype=float)
    ret_3m = pd.Series(index=valid_tickers, dtype=float)

    for t in valid_tickers:
        s = close_df[t].dropna()
        ret_1w[t] = (s.iloc[-1] / s.iloc[-6] - 1) if len(s) >= 6 else 0
        ret_1m[t] = (s.iloc[-1] / s.iloc[-22] - 1) if len(s) >= 22 else 0
        ret_3m[t] = (s.iloc[-1] / s.iloc[-64] - 1) if len(s) >= 64 else 0

    composite = (ret_1w * 0.2) + (ret_1m * 0.4) + (ret_3m * 0.4)
    ranks = composite.rank(pct=True) * 99

    for t in valid_tickers:
        rs_results[t] = round(ranks[t], 1)
    return rs_results


def calculate_drift_from_history(history_df, ticker, days_from_result):
    """Calculate the exact return from the earnings date to today using our own data."""
    try:
        if len(history_df.columns.names) > 1 or history_df.columns.nlevels > 1:
            close_series = history_df.xs('Close', level=1, axis=1)[ticker].dropna()
        else:
            close_series = history_df['Close'].dropna()

        # days_from_result is negative (e.g., -26 means 26 calendar days ago)
        abs_days = abs(int(days_from_result))
        # Find the closest trading day to that date
        target_date = datetime.now() - timedelta(days=abs_days)
        
        # Find the trading day closest to (but not after) the target date
        # We search for the last close ON or BEFORE the target date
        close_on_earnings = None
        for i in range(len(close_series) - 1, -1, -1):
            idx_date = close_series.index[i]
            if hasattr(idx_date, 'tz_localize'):
                idx_date = idx_date.tz_localize(None)
            elif hasattr(idx_date, 'tz_convert'):
                idx_date = idx_date.tz_convert(None)
            if idx_date.date() <= target_date.date():
                close_on_earnings = close_series.iloc[i]
                break
        
        if close_on_earnings is None or close_on_earnings == 0:
            return None
        
        current_price = close_series.iloc[-1]
        drift_pct = ((current_price - close_on_earnings) / close_on_earnings) * 100
        return round(drift_pct, 2)
    except Exception:
        return None


def render_metric_card(label, value, color_class="green-text"):
    """Render a styled metric card."""
    color = "#10b981" if color_class == "green-text" else "#ef4444" if color_class == "red-text" else "#3b82f6"
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, rgba(20,25,35,0.8), rgba(10,15,20,0.95));
                border: 1px solid rgba(0,243,255,0.15); border-radius: 12px; padding: 16px; text-align: center;">
        <p style="margin:0; color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">{label}</p>
        <p style="margin: 8px 0 0 0; color: {color}; font-size: 1.6rem; font-weight: 900;">{value}</p>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 2: HISTORICAL PEAD CSV SCANNER
# -----------------------------------------------------------------------------
def render_pead_csv_scanner():
    st.markdown("#### 📁 Historical PEAD CSV Drift Scanner")
    st.caption("Upload a StockScans.in PEAD CSV export. Kush Tracker will enrich it with proprietary RS scores and independently verify the drift using its own price data.")

    # --- Info Panel ---
    with st.expander("📖 Methodology: How This Scanner Works"):
        st.markdown("""
        **1. What is PEAD?**
        Post-Earnings Announcement Drift is a proven market anomaly: stocks that report massive earnings beats continue to "drift" higher for 4-8 weeks as institutions slowly accumulate positions. This scanner helps you find and rank these setups.
        
        **2. Data Pipeline**
        - **StockScans.in** provides the earnings dates and fundamental growth numbers (Revenue/PAT Growth).
        - **Kush Tracker** independently downloads price history via yfinance and calculates its own **Verified Drift %** from the exact earnings date.
        
        **3. RS Enrichment Formula (3M RS, 0-99)**
        `Raw RS = (1-Week Return × 20%) + (1-Month Return × 40%) + (3-Month Return × 40%)`
        This is percent-ranked cross-sectionally against all uploaded tickers.
        
        **4. The Holy Grail Setup**
        Look for stocks where: **Verified Drift > +15%** AND **3M RS > 90** AND **PAT Growth > 50%**.
        This means a stock that was *already* a momentum leader just got a massive fundamental catalyst and institutions are aggressively buying the new data.
        """)

    # --- CSV Upload ---
    uploaded_file = st.file_uploader("📁 Upload StockScans PEAD CSV", type=["csv"], help="Export the CSV from your StockScans.in PEAD scan and upload it here.")

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)

        # Validate required columns
        required_cols = ['companyId', 'Name', 'Industry', 'Days From Result']
        missing = [c for c in required_cols if c not in raw_df.columns]
        if missing:
            st.error(f"CSV is missing required columns: {missing}. Please upload a valid StockScans export.")
            return

        st.success(f"✅ Loaded **{len(raw_df)} stocks** from CSV.")

        # Clean tickers for yfinance
        raw_df['Clean_Ticker'] = raw_df['companyId'].str.replace('NSE:', '').str.replace('BSE:', '')
        raw_df['YF_Ticker'] = raw_df['Clean_Ticker'] + '.NS'
        yf_tickers = raw_df['YF_Ticker'].tolist()

        # --- Fetch Price History & Calculate ---
        with st.spinner("📡 Fetching price history & calculating RS scores..."):
            history_df = fetch_price_history_batch(tuple(yf_tickers), days=90)

            # Calculate RS scores
            rs_scores = compute_rs_scores(history_df, yf_tickers)

            # Calculate independent drift for each stock
            drifts = []
            for _, row in raw_df.iterrows():
                drift = calculate_drift_from_history(history_df, row['YF_Ticker'], row['Days From Result'])
                drifts.append(drift)
            raw_df['Kush_Drift'] = drifts

            # Add RS scores
            raw_df['RS_3M'] = raw_df['YF_Ticker'].map(rs_scores).fillna(0).round(1)
            
            # Convert Days From Result to positive for display (e.g., -26 → 26)
            raw_df['Days_Abs'] = raw_df['Days From Result'].abs().astype(int)

        # --- Summary Metrics ---
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("Stocks Uploaded", f"{len(raw_df)}", "blue-text")
        with m2:
            avg_drift = raw_df['Kush_Drift'].dropna().mean()
            render_metric_card("Avg Verified Drift", f"+{avg_drift:.1f}%" if avg_drift else "N/A", "green-text")
        with m3:
            avg_pat = raw_df['PAT Growth YoY'].dropna().mean()
            render_metric_card("Avg PAT Growth YoY", f"+{avg_pat:.0f}%" if not pd.isna(avg_pat) else "N/A", "green-text")
        with m4:
            avg_rs = raw_df['RS_3M'].mean()
            render_metric_card("Avg 3M RS Score", f"{avg_rs:.1f}", "blue-text")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Filters ---
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            min_drift = st.slider("Min Verified Drift %", -50.0, 200.0, 0.0, 5.0, key="pead_min_drift")
        with f2:
            min_rs = st.slider("Min 3M RS Score", 0.0, 99.0, 0.0, 5.0, key="pead_min_rs")
        with f3:
            min_pat = st.number_input("Min PAT Growth YoY %", min_value=-100.0, max_value=10000.0, value=0.0, step=10.0, key="pead_min_pat")
        with f4:
            min_rev = st.number_input("Min Rev Growth YoY %", min_value=-100.0, max_value=10000.0, value=0.0, step=10.0, key="pead_min_rev")

        # Apply filters
        filtered = raw_df.copy()
        
        # Check for missing data
        missing_data_mask = filtered['Kush_Drift'].isna()
        if missing_data_mask.any():
            missing_tickers = filtered.loc[missing_data_mask, 'Clean_Ticker'].tolist()
            st.warning(f"⚠️ **Missing Data:** Could not calculate drift for {len(missing_tickers)} stocks (usually due to recent IPOs, SME listings, or ticker changes on Yahoo Finance): {', '.join(missing_tickers)}")
            
        filtered = filtered[~missing_data_mask]
        filtered = filtered[filtered['Kush_Drift'] >= min_drift]
        filtered = filtered[filtered['RS_3M'] >= min_rs]
        if 'PAT Growth YoY' in filtered.columns:
            pat_mask = filtered['PAT Growth YoY'].fillna(0) >= min_pat
            filtered = filtered[pat_mask]
        if 'Revenue Growth YoY' in filtered.columns:
            rev_mask = filtered['Revenue Growth YoY'].fillna(0) >= min_rev
            filtered = filtered[rev_mask]

        # Sort by drift descending
        filtered = filtered.sort_values('Kush_Drift', ascending=False).reset_index(drop=True)
        filtered.index += 1
        filtered['#'] = filtered.index
        filtered['TradingView'] = filtered['Clean_Ticker'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{x}")

        # --- TradingView Export ---
        tv_list = ",".join(["NSE:" + t for t in filtered['Clean_Ticker']])
        with st.expander(f"📋 Copy Filtered Tickers for TradingView ({len(filtered)})"):
            st.code(tv_list, language="text")
            st.caption("Copy and paste into TradingView watchlist import.")

        # --- Build Display Table ---
        display_cols = ['#', 'Clean_Ticker', 'Name', 'Industry', 'Days_Abs', 'Kush_Drift',
                        'RS_3M', 'TradingView']

        # Add optional StockScans columns if they exist
        optional_cols = ['PAT Growth YoY', 'Revenue Growth YoY', '52WH Distance',
                         'Returns Vs Nifty 500 1M', 'Market Capitalization']
        for col in optional_cols:
            if col in filtered.columns:
                display_cols.append(col)

        render_df = filtered[display_cols]

        # Format styling
        format_dict = {
            'Kush_Drift': "{:+.1f}%",
            'RS_3M': "{:.1f}",
        }
        for col in optional_cols:
            if col in render_df.columns:
                if col == 'Market Capitalization':
                    format_dict[col] = "{:,.0f}"
                else:
                    format_dict[col] = "{:.1f}"

        # Color status for drift
        def color_drift(val):
            try:
                v = float(str(val).replace('%', '').replace('+', ''))
                if v >= 30: return 'color: #10b981; font-weight: bold;'
                elif v >= 15: return 'color: #22d3ee; font-weight: bold;'
                elif v >= 0: return 'color: #94a3b8;'
                else: return 'color: #ef4444; font-weight: bold;'
            except:
                return ''

        st.dataframe(
            render_df,
            column_config={
                "#": st.column_config.NumberColumn("#", format="%d"),
                "Clean_Ticker": st.column_config.TextColumn("Ticker"),
                "Name": st.column_config.TextColumn("Company"),
                "Industry": st.column_config.TextColumn("Industry"),
                "Days_Abs": st.column_config.NumberColumn("Days Since Result", format="%d"),
                "Kush_Drift": st.column_config.NumberColumn("Verified Drift %", format="%+.1f%%", help="Independently calculated from yfinance price data."),
                "RS_3M": st.column_config.NumberColumn("3M RS", format="%.1f", help="0-99 cross-sectional momentum score."),
                "TradingView": st.column_config.LinkColumn("Chart", display_text="Open 📈"),
                "PAT Growth YoY": st.column_config.NumberColumn("PAT Growth %", format="%.1f%%"),
                "Revenue Growth YoY": st.column_config.NumberColumn("Rev Growth %", format="%.1f%%"),
                "52WH Distance": st.column_config.NumberColumn("52WH Dist %", format="%.1f%%", help="Distance from 52-week high."),
                "Returns Vs Nifty 500 1M": st.column_config.NumberColumn("vs Nifty 1M", format="%+.1f%%"),
                "Market Capitalization": st.column_config.NumberColumn("MCap (Cr)", format="₹%,.0f"),
            },
            hide_index=True,
            use_container_width=True,
            height=650
        )

        # --- Analytics & Visualizations ---
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("### 📊 PEAD Analytics Engine")
        
        tab_roppel, tab_sector = st.tabs(["🫧 The Roppel Quadrant", "🏭 Sector Clustering"])
        
        with tab_roppel:
            st.caption("Apex Predators live in the top-right corner: Elite Momentum (High RS) combined with massive Post-Earnings Drift. Larger bubbles = Larger Market Cap.")
            
            if not filtered.empty:
                # Prepare data for plotting (handle NaNs for sizing)
                plot_df = filtered.copy()
                plot_df['Market Capitalization'] = plot_df.get('Market Capitalization', 1000).fillna(1000)
                # Cap bubble sizes for visual balance
                plot_df['Bubble_Size'] = np.clip(plot_df['Market Capitalization'], 500, 100000)
                
                fig_roppel = px.scatter(
                    plot_df,
                    x='RS_3M',
                    y='Kush_Drift',
                    size='Bubble_Size',
                    color='Industry',
                    hover_name='Clean_Ticker',
                    text='Clean_Ticker',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    labels={
                        'RS_3M': '3M Relative Strength (0-99)',
                        'Kush_Drift': 'Verified Post-Earnings Drift (%)'
                    }
                )
                
                fig_roppel.update_traces(
                    textposition='top center', 
                    marker=dict(line=dict(width=1, color='rgba(255,255,255,0.5)'), opacity=0.8),
                    textfont=dict(size=10)
                )
                
                # Add quadrant crosshairs (assume 80 RS and 10% drift are the magic thresholds)
                fig_roppel.add_hline(y=10.0, line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text="Breakout Threshold (10%)")
                fig_roppel.add_vline(x=80.0, line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text="Elite RS (80+)")
                
                fig_roppel.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=600,
                    margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=False
                )
                st.plotly_chart(fig_roppel, use_container_width=True)
            else:
                st.warning("No data available for plotting. Adjust your filters.")
                
        with tab_sector:
            st.caption("Industries with the most PEAD drifters signal coordinated institutional accumulation.")
            
            if not filtered.empty:
                sector_counts = filtered['Industry'].value_counts().head(15).sort_values(ascending=True)
                fig_sector = px.bar(
                    x=sector_counts.values,
                    y=sector_counts.index,
                    orientation='h',
                    labels={'x': 'Number of PEAD Stocks', 'y': ''},
                    color=sector_counts.values,
                    color_continuous_scale='Tealgrn'
                )
                fig_sector.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=500,
                    showlegend=False,
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_sector, use_container_width=True)

    else:
        st.info("👆 Upload a CSV from StockScans.in to begin. Go to StockScans → Run your PEAD scan → Click 'Download' → Upload the CSV here.")


# -----------------------------------------------------------------------------
# TAB 1: LIVE CORPORATE EARNINGS CALENDAR & CONSENSUS ESTIMATES HUB
# -----------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def _get_company_names_map():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ind_niftytotalmarket_list.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'Symbol' in df.columns and 'Company Name' in df.columns:
                return df.set_index('Symbol')['Company Name'].to_dict()
        except Exception:
            pass
    return {}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_india_declared_earnings():
    """
    Fetch recently declared corporate results for Indian equities directly from
    TradingView's institutional scanner API.
    Guarantees fresh, real-time results from the active earnings season with exact consensus surprises.
    Falls back to cached JSON if offline.
    """
    cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "india_declared_earnings.json")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Origin': 'https://www.tradingview.com',
            'Referer': 'https://www.tradingview.com/'
        }
        url = 'https://scanner.tradingview.com/india/scan'
        cols = [
            'name', 'description', 'earnings_release_date', 'fiscal_period_end_fq',
            'earnings_per_share_fq', 'earnings_per_share_forecast_fq',
            'revenue_fq', 'revenue_forecast_fq', 'revenue_surprise_percent_fq',
            'close', 'change', 'relative_volume_10d_calc', 'volume',
            'Perf.W', 'Perf.1M', 'Perf.3M', 'sector', 'industry',
            'earnings_per_share_diluted_yoy_growth_fq', 'total_revenue_yoy_growth_fq',
            'market_cap_basic',
            'average_volume_10d_calc', 'average_volume_30d_calc', 'Value.Traded'
        ]

        # Channel 1: Declared results with official earnings release date on institutional calendar
        payload_rel = {
            'filter': [
                {'left': 'earnings_release_date', 'operation': 'nempty'},
                {'left': 'type', 'operation': 'equal', 'right': 'stock'}
            ],
            'options': {'lang': 'en'},
            'symbols': {'query': {'types': []}, 'tickers': []},
            'columns': cols,
            'sort': {'sortBy': 'earnings_release_date', 'sortOrder': 'desc'},
            'range': [0, 2000]
        }

        # Channel 2: All companies filing quarterly reports for current reporting quarter (Sep 30, 2026),
        # capturing small/microcap equities uncovered by sell-side consensus analysts
        payload_period = {
            'filter': [
                {'left': 'fiscal_period_end_fq', 'operation': 'in_range', 'right': [1790700000, 1790850000]},
                {'left': 'type', 'operation': 'equal', 'right': 'stock'}
            ],
            'options': {'lang': 'en'},
            'symbols': {'query': {'types': []}, 'tickers': []},
            'columns': cols,
            'sort': {'sortBy': 'market_cap_basic', 'sortOrder': 'desc'},
            'range': [0, 2000]
        }

        r_rel = requests.post(url, json=payload_rel, headers=headers, timeout=10)
        r_period = requests.post(url, json=payload_period, headers=headers, timeout=10)

        raw_rows = []
        if r_period.status_code == 200:
            raw_rows.extend(r_period.json().get('data', []))
        if r_rel.status_code == 200:
            raw_rows.extend(r_rel.json().get('data', []))

        if raw_rows:
            # Prefer NSE quotes over BSE so primary volume and ADTV are authentic
            raw_rows.sort(key=lambda r: (0 if str(r.get('s', '')).startswith('NSE:') else 1))
            seen = set()
            results = []
            for row in raw_rows:
                s = row['s']
                name = row['d'][0]
                if name in seen or '/' in name:
                    continue
                seen.add(name)
                d = row['d']
                dt = datetime.fromtimestamp(d[2]).strftime('%Y-%m-%d') if d[2] else None
                f_end = d[3]
                is_sep_qtr = bool(f_end and 1790700000 <= f_end <= 1790850000)

                # If company filed for Sep 2026 quarter without a sell-side calendar date or has an old historical date,
                # anchor date into the active October Q2 reporting window
                if is_sep_qtr and (not dt or dt < '2026-10-01'):
                    dt = '2026-10-15'

                act = d[4]
                est = d[5]
                surp = None
                if act is not None and est is not None:
                    if est != 0:
                        surp = round(((act - est) / abs(est)) * 100, 2)
                    else:
                        surp = round((act - est) * 100, 2)

                act_rev = d[6]
                est_rev = d[7]
                surp_rev = round(d[8], 2) if d[8] is not None else None

                close_val = round(d[9], 2) if len(d) > 9 and d[9] is not None else None
                change_val = round(d[10], 2) if len(d) > 10 and d[10] is not None else None
                rel_vol = round(d[11], 2) if len(d) > 11 and d[11] is not None else None
                vol_val = d[12] if len(d) > 12 else None
                perf_w = round(d[13], 2) if len(d) > 13 and d[13] is not None else None
                perf_1m = round(d[14], 2) if len(d) > 14 and d[14] is not None else None
                perf_3m = round(d[15], 2) if len(d) > 15 and d[15] is not None else None
                sector_val = d[16] if len(d) > 16 and d[16] else "Other"
                industry_val = d[17] if len(d) > 17 and d[17] else "Other"
                eps_yoy = round(d[18], 2) if len(d) > 18 and d[18] is not None else None
                rev_yoy = round(d[19], 2) if len(d) > 19 and d[19] is not None else None
                mc_cr = round(d[20] / 1e7, 1) if len(d) > 20 and d[20] is not None else None

                # 20-Day Average Daily Traded Value (ADTV) in ₹ Crores
                v10 = d[21] if len(d) > 21 and d[21] is not None else (vol_val or 0)
                v30 = d[22] if len(d) > 22 and d[22] is not None else v10
                v20 = (v10 + v30) / 2.0
                adtv_cr = round((v20 * close_val) / 1e7, 2) if close_val and v20 else None

                results.append({
                    'symbol': name,
                    'ticker': s,
                    'company': d[1],
                    'date': dt,
                    'est': est,
                    'act': act,
                    'surprise': surp,
                    'rev_est': est_rev,
                    'rev_act': act_rev,
                    'rev_surprise': surp_rev,
                    'close': close_val,
                    'change': change_val,
                    'rel_vol': rel_vol,
                    'volume': vol_val,
                    'perf_w': perf_w,
                    'perf_1m': perf_1m,
                    'perf_3m': perf_3m,
                    'sector': sector_val,
                    'industry': industry_val,
                    'eps_yoy': eps_yoy,
                    'rev_yoy': rev_yoy,
                    'market_cap': mc_cr,
                    'adtv_cr': adtv_cr
                })
            if results:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                return pd.DataFrame(results)
    except Exception:
        pass

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except Exception:
            pass

    return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_us_declared_earnings():
    """
    Fetch recently declared corporate results for US equities directly from
    TradingView's institutional scanner API.
    Guarantees exact parity with TradingView charts (including standardized consensus & surprises).
    Falls back to cached JSON if offline.
    """
    cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "us_declared_earnings.json")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Origin': 'https://www.tradingview.com',
            'Referer': 'https://www.tradingview.com/'
        }
        url = 'https://scanner.tradingview.com/america/scan'
        payload = {
            'filter': [
                {'left': 'earnings_release_date', 'operation': 'nempty'},
                {'left': 'type', 'operation': 'equal', 'right': 'stock'},
                {'left': 'subtype', 'operation': 'equal', 'right': 'common'}
            ],
            'options': {'lang': 'en'},
            'symbols': {'query': {'types': []}, 'tickers': []},
            'columns': [
                'name', 'description', 'earnings_release_date', 'earnings_release_next_date',
                'earnings_per_share_fq', 'earnings_per_share_forecast_fq',
                'revenue_fq', 'revenue_forecast_fq', 'revenue_surprise_percent_fq',
                'close', 'change', 'relative_volume_10d_calc', 'volume',
                'Perf.W', 'Perf.1M', 'Perf.3M', 'sector', 'industry',
                'earnings_per_share_diluted_yoy_growth_fq', 'total_revenue_yoy_growth_fq',
                'market_cap_basic',
                'average_volume_10d_calc', 'average_volume_30d_calc', 'Value.Traded'
            ],
            'sort': {'sortBy': 'earnings_release_date', 'sortOrder': 'desc'},
            'range': [0, 3000]
        }
        r = requests.post(url, json=payload, headers=headers, timeout=8)
        if r.status_code == 200:
            seen = set()
            results = []
            for row in r.json().get('data', []):
                s = row['s']
                name = row['d'][0]
                if name in seen or '/' in name:
                    continue
                seen.add(name)
                d = row['d']
                dt = datetime.fromtimestamp(d[2]).strftime('%Y-%m-%d %H:%M') if d[2] else None
                act = d[4]
                est = d[5]
                surp = None
                if act is not None and est is not None:
                    if est != 0:
                        surp = round(((act - est) / abs(est)) * 100, 2)
                    else:
                        surp = round((act - est) * 100, 2)

                act_rev = d[6]
                est_rev = d[7]
                surp_rev = round(d[8], 2) if d[8] is not None else None

                close_val = round(d[9], 2) if len(d) > 9 and d[9] is not None else None
                change_val = round(d[10], 2) if len(d) > 10 and d[10] is not None else None
                rel_vol = round(d[11], 2) if len(d) > 11 and d[11] is not None else None
                vol_val = d[12] if len(d) > 12 else None
                perf_w = round(d[13], 2) if len(d) > 13 and d[13] is not None else None
                perf_1m = round(d[14], 2) if len(d) > 14 and d[14] is not None else None
                perf_3m = round(d[15], 2) if len(d) > 15 and d[15] is not None else None
                sector_val = d[16] if len(d) > 16 and d[16] else "Other"
                industry_val = d[17] if len(d) > 17 and d[17] else "Other"
                eps_yoy = round(d[18], 2) if len(d) > 18 and d[18] is not None else None
                rev_yoy = round(d[19], 2) if len(d) > 19 and d[19] is not None else None
                mc_m = round(d[20] / 1e6, 1) if len(d) > 20 and d[20] is not None else None

                # 20-Day Average Daily Traded Value (ADTV) in $ Millions
                v10 = d[21] if len(d) > 21 and d[21] is not None else (vol_val or 0)
                v30 = d[22] if len(d) > 22 and d[22] is not None else v10
                v20 = (v10 + v30) / 2.0
                adtv_m = round((v20 * close_val) / 1e6, 2) if close_val and v20 else None

                results.append({
                    'symbol': name,
                    'ticker': s,
                    'company': d[1],
                    'date': dt,
                    'est': est,
                    'act': act,
                    'surprise': surp,
                    'rev_est': est_rev,
                    'rev_act': act_rev,
                    'rev_surprise': surp_rev,
                    'close': close_val,
                    'change': change_val,
                    'rel_vol': rel_vol,
                    'volume': vol_val,
                    'perf_w': perf_w,
                    'perf_1m': perf_1m,
                    'perf_3m': perf_3m,
                    'sector': sector_val,
                    'industry': industry_val,
                    'eps_yoy': eps_yoy,
                    'rev_yoy': rev_yoy,
                    'market_cap': mc_m,
                    'adtv_m': adtv_m
                })
            if results:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                return pd.DataFrame(results)
    except Exception:
        pass

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except Exception:
            pass

    return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_nse_corporate_earnings_calendar():
    """
    Fetch corporate earnings board meetings calendar from NSE India.
    Filters for 'Financial Results' events.
    Falls back to cached JSON if network or NSE session fails.
    """
    cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "nse_earnings_calendar.json")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-event-calendar"
        }
        s = requests.Session()
        s.headers.update(headers)
        s.get("https://www.nseindia.com", timeout=6)
        r = s.get("https://www.nseindia.com/api/event-calendar", timeout=8)
        if r.status_code == 200:
            raw_data = r.json()
            results = []
            for item in raw_data:
                purpose = str(item.get("purpose", ""))
                desc = str(item.get("bm_desc", ""))
                if "Financial Results" in purpose or "Financial Results" in desc:
                    results.append({
                        "symbol": item.get("symbol", "").strip(),
                        "company": item.get("company", "").strip(),
                        "purpose": purpose,
                        "bm_desc": desc,
                        "date": item.get("date", "").strip()
                    })
            if results:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                return pd.DataFrame(results)
    except Exception:
        pass

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except Exception:
            pass

    return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_us_corporate_earnings_calendar():
    """
    Fetch corporate earnings calendar for US Equities via yfinance Calendars.
    Falls back to cached JSON if yfinance API throttles.
    """
    cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "us_earnings_calendar.json")
    try:
        c = yf.Calendars()
        cal = c.get_earnings_calendar(filter_most_active=False, limit=100)
        if cal is not None and not cal.empty:
            records = cal.to_dict(orient="records")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(records, f, default=str, indent=2)
            return pd.DataFrame(records)
    except Exception:
        pass

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except Exception:
            pass

    return pd.DataFrame()


def _format_days_badge(days):
    if pd.isna(days):
        return "Unknown"
    d = int(days)
    if d < -1:
        return f"✅ Reported ({abs(d)}d ago)"
    elif d == -1:
        return "✅ Reported Yesterday"
    elif d == 0:
        return "⚡ Today"
    elif d == 1:
        return "⚠️ Tomorrow (1d)"
    elif d <= 3:
        return f"🚨 In {d} Days"
    elif d <= 7:
        return f"⏳ In {d} Days"
    else:
        return f"📅 In {d} Days"


# -----------------------------------------------------------------------------
# DUAL SURPRISE (EPS + REVENUE) HELPERS
# -----------------------------------------------------------------------------
def _format_revenue_val(val, currency="$"):
    if val is None or pd.isna(val):
        return "-"
    try:
        v = float(val)
        if currency == "₹":
            if abs(v) >= 1e7:
                return f"₹{v / 1e7:,.1f} Cr"
            elif abs(v) >= 1e5:
                return f"₹{v / 1e5:,.1f} L"
            return f"₹{v:,.0f}"
        else:
            if abs(v) >= 1e9:
                return f"${v / 1e9:,.2f}B"
            elif abs(v) >= 1e6:
                return f"${v / 1e6:,.2f}M"
            return f"${v:,.2f}"
    except Exception:
        return "-"


def _format_dual_verdict(eps_surp, rev_surp):
    has_eps = eps_surp is not None and not pd.isna(eps_surp)
    has_rev = rev_surp is not None and not pd.isna(rev_surp)
    eps_val = float(eps_surp) if has_eps else None
    rev_val = float(rev_surp) if has_rev else None

    if has_eps and has_rev and eps_val > 0 and rev_val > 0:
        return f"🔥 DUAL BEAT (EPS +{eps_val:.1f}%, Rev +{rev_val:.1f}%)"
    elif has_eps and eps_val >= 5.0:
        rev_txt = f", Rev +{rev_val:.1f}%" if has_rev and rev_val > 0 else (f", Rev {rev_val:.1f}%" if has_rev else "")
        return f"🚀 Rocket EPS Beat (+{eps_val:.1f}%{rev_txt})"
    elif has_eps and eps_val > 0:
        rev_txt = f", Rev +{rev_val:.1f}%" if has_rev and rev_val > 0 else (f", Rev {rev_val:.1f}%" if has_rev else "")
        return f"🟢 EPS Beat (+{eps_val:.1f}%{rev_txt})"
    elif has_rev and rev_val > 0 and (not has_eps or eps_val <= 0):
        eps_txt = f", EPS {eps_val:.1f}%" if has_eps else ""
        return f"⚡ Sales Beat (Rev +{rev_val:.1f}%{eps_txt})"
    elif has_eps and has_rev and eps_val < 0 and rev_val < 0:
        return f"🔴 Double Miss (EPS {eps_val:.1f}%, Rev {rev_val:.1f}%)"
    elif has_eps and eps_val < 0:
        return f"🔴 EPS Miss ({eps_val:.1f}%)"
    elif not has_eps and not has_rev:
        return "📊 Uncovered (Pure Growth)"
    else:
        return "⚡ In-Line"


def _format_dual_pead_setup(eps_surp, rev_surp):
    has_eps = eps_surp is not None and not pd.isna(eps_surp)
    has_rev = rev_surp is not None and not pd.isna(rev_surp)
    eps_val = float(eps_surp) if has_eps else None
    rev_val = float(rev_surp) if has_rev else None

    if has_eps and has_rev and eps_val > 0 and rev_val > 0:
        if eps_val >= 15.0 or rev_val >= 15.0:
            return "🔥 Monster Dual Beat (Top Drift)"
        return "🚀 Prime Dual Beat Leader"
    elif has_eps and eps_val >= 5.0:
        return "🚀 Prime EPS Drift"
    elif has_eps and eps_val > 0:
        return "🟢 Moderate EPS Drift"
    elif has_rev and rev_val >= 5.0:
        return "⚡ Top-Line Sales Runner"
    elif has_rev and rev_val > 0:
        return "⚡ Top-Line Drift Watch"
    elif not has_eps and not has_rev:
        return "🌱 Uncovered Growth Setup"
    else:
        return "⚠️ Post-Earnings Drag"


# -----------------------------------------------------------------------------
# CANSLIM QUANTITATIVE METRICS & THEMATIC CONTAGION ENGINE
# -----------------------------------------------------------------------------
def _compute_relative_strength_ranks(df):
    """
    Computes William O'Neil CANSLIM 3-Month Composite Relative Strength:
    RS Composite = (0.20 * Perf.W) + (0.40 * Perf.1M) + (0.40 * Perf.3M)
    Percentile ranked from 1 to 99 across the universe.
    """
    if df.empty:
        return df
    res = df.copy()
    pw = pd.to_numeric(res.get("perf_w", 0), errors="coerce").fillna(0)
    p1m = pd.to_numeric(res.get("perf_1m", 0), errors="coerce").fillna(0)
    p3m = pd.to_numeric(res.get("perf_3m", 0), errors="coerce").fillna(0)
    composite = (0.20 * pw) + (0.40 * p1m) + (0.40 * p3m)
    res["rs_composite"] = composite
    valid_mask = res["perf_3m"].notna() | res["perf_1m"].notna()
    if valid_mask.sum() > 1:
        ranks = pd.Series(index=res.index, dtype=float)
        ranks[valid_mask] = res.loc[valid_mask, "rs_composite"].rank(pct=True) * 98 + 1
        res["rs_raw_pct"] = ranks.round().fillna(50).astype(int)
    else:
        res["rs_raw_pct"] = 50
    return res


def _format_rs_badge(rs_val):
    if pd.isna(rs_val) or rs_val is None:
        return "-"
    try:
        val = int(rs_val)
        if val >= 80:
            return f"🟢 {val} (Leader)"
        elif val >= 60:
            return f"🔵 {val} (Strong)"
        elif val >= 40:
            return f"⚪ {val} (Avg)"
        else:
            return f"🔴 {val} (Lag)"
    except Exception:
        return "-"


def _format_growth_val(val):
    """Format YoY percentage with directional color indicator and fire icon for >= 25%."""
    if pd.isna(val) or val is None:
        return "-"
    try:
        v = float(val)
        if v >= 25.0:
            return f"🟢 +{v:.1f}% 🔥"
        elif v > 0:
            return f"🟢 +{v:.1f}%"
        elif v == 0:
            return "⚪ 0.0%"
        else:
            return f"🔴 {v:.1f}%"
    except Exception:
        return "-"


def _format_growth_badge(eps_yoy, rev_yoy, eps_surp=None):
    """
    CANSLIM Pillar 'C' Classifier:
    - 💎 CANSLIM Super-Growth: EPS YoY >= 25% AND Sales YoY >= 20%
    - 🚀 High EPS Growth: EPS YoY >= 25%
    - ⚡ High Sales Growth: Sales YoY >= 20%
    - ⚠️ Surprise Mirage: Positive analyst surprise, but negative absolute YoY growth!
    - 🟢 Dual Growth: Both EPS and Sales > 0%
    - 🔴 Contraction: Both EPS and Sales < 0%
    """
    try:
        has_eps = pd.notna(eps_yoy) and eps_yoy is not None
        has_rev = pd.notna(rev_yoy) and rev_yoy is not None

        ey = float(eps_yoy) if has_eps else None
        ry = float(rev_yoy) if has_rev else None
        surp = float(eps_surp) if pd.notna(eps_surp) and eps_surp is not None else None

        # Check for Surprise Mirage first (Beats estimate but shrinking business)
        if surp is not None and surp > 0 and ((ey is not None and ey < 0) or (ry is not None and ry < 0)):
            return "⚠️ Surprise Mirage"

        if ey is not None and ry is not None:
            if ey >= 25.0 and ry >= 20.0:
                return "💎 Super-Growth"
            elif ey >= 25.0:
                return "🚀 High EPS"
            elif ry >= 20.0:
                return "⚡ High Sales"
            elif ey > 0 and ry > 0:
                return "🟢 Dual Growth"
            elif ey < 0 and ry < 0:
                return "🔴 Contraction"
            elif ey < 0:
                return "⚠️ EPS Decline"
            elif ry < 0:
                return "⚠️ Sales Decline"
        elif ey is not None:
            if ey >= 25.0:
                return "🚀 High EPS"
            elif ey > 0:
                return "🟢 EPS Growth"
            else:
                return "🔴 EPS Decline"
        elif ry is not None:
            if ry >= 20.0:
                return "⚡ High Sales"
            elif ry > 0:
                return "🟢 Sales Growth"
            else:
                return "🔴 Sales Decline"

        return "-"
    except Exception:
        return "-"


def _format_closing_range_badge(cr_val):
    """
    Format intraday Closing Range badge:
    CR = (Close - Low) / (High - Low) * 100
    - 🔥 >= 75%: Power Close (Institutional Net Accumulation)
    - 🟢 >= 50%: Constructive Close
    - 🟡 >= 35%: Mid-Range Indecision
    - 🔴 < 35%: Gap & Crap Trap (Distribution into Strength)
    """
    if cr_val is None or pd.isna(cr_val):
        return "-"
    try:
        cr = float(cr_val)
        if cr >= 75.0:
            return f"🔥 {cr:.0f}% (Power Close)"
        elif cr >= 50.0:
            return f"🟢 {cr:.0f}% (Constructive)"
        elif cr >= 35.0:
            return f"🟡 {cr:.0f}% (Mid-Range)"
        else:
            return f"🔴 {cr:.0f}% (Gap & Crap)"
    except Exception:
        return "-"


def _classify_episodic_pivot(day0_rx, rel_vol, closing_range, drift):
    """
    Classifies Episodic Pivot (EP) / Institutional Accumulation Quality:
    - 🏆 True Episodic Pivot: Day0 Rx >= +3.0%, Rel Vol >= 2.0x, Closing Range >= 70%, Drift >= 0%
    - 🔥 Power Close: Closing Range >= 75%
    - ⚡ Momentum Gap: Day0 Rx >= +3.0%, Closing Range >= 50%
    - 🔴 Gap & Crap Trap: Rel Vol >= 1.5x and Closing Range < 35%
    """
    try:
        rx = float(day0_rx) if pd.notna(day0_rx) and day0_rx is not None else 0.0
        rv = float(rel_vol) if pd.notna(rel_vol) and rel_vol is not None else 1.0
        cr = float(closing_range) if pd.notna(closing_range) and closing_range is not None else 50.0
        df = float(drift) if pd.notna(drift) and drift is not None else 0.0

        if rv >= 1.5 and cr < 35.0:
            return "🔴 Gap & Crap Trap"
        if rx >= 3.0 and rv >= 2.0 and cr >= 70.0 and df >= 0.0:
            return "🏆 True Episodic Pivot (EP)"
        if cr >= 75.0 and rx > 0:
            return "🔥 Power Close"
        if rx >= 3.0 and cr >= 50.0:
            return "⚡ Momentum Gap"
        if cr >= 50.0:
            return "🟢 Constructive"
        return "Normal"
    except Exception:
        return "Normal"


def _format_reaction_badge(change, rel_vol, closing_range=None):
    """Format color-coded reaction candle badge with volume surge factor and optional closing range."""
    if pd.isna(change) or change is None:
        return "-"
    try:
        c = float(change)
        rv_str = f" ({rel_vol:.1f}x Vol)" if pd.notna(rel_vol) and rel_vol is not None else ""
        cr_str = f" • CR: {closing_range:.0f}%" if pd.notna(closing_range) and closing_range is not None else ""
        if c > 0:
            flag = " 🚀" if pd.notna(rel_vol) and rel_vol and float(rel_vol) >= 2.0 else ""
            if pd.notna(closing_range) and closing_range is not None and float(closing_range) >= 75.0:
                flag += " 🔥"
            return f"🟢 +{c:.2f}%{rv_str}{cr_str}{flag}"
        elif c < 0:
            fade_flag = " ⚠️ Gap Fade" if pd.notna(rel_vol) and rel_vol and float(rel_vol) >= 2.0 else ""
            return f"🔴{fade_flag} {c:.2f}%{rv_str}{cr_str}"
        else:
            return f"⚪ 0.00%{rv_str}{cr_str}"
    except Exception:
        return "-"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_batch_pead_drift(symbol_date_tuples, market="IN"):
    """
    CANSLIM & PEAD Dual Momentum Engine:
    Computes for each (symbol, announcement_date):
      1. Day 0 Announcement Reaction %:
         Price return specifically on the day of earnings announcement (Close on result day vs Close on prior day).
      2. PEAD Drift %:
         Cumulative total return from earnings announcement date to current date:
         (Current Close - Close on Result Day) / (Close on Result Day) * 100
    Vectorized across all symbols using yf.download() with 30-minute caching.
    """
    results = {}
    if not symbol_date_tuples:
        return results

    yf_map = {}
    valid_dates = []
    for s, d in symbol_date_tuples:
        s_clean = str(s).strip()
        if market == "IN":
            clean_s = s_clean.replace("NSE:", "").replace("BSE:", "").strip()
            yf_ticker = f"{clean_s}.NS"
        else:
            yf_ticker = s_clean
        yf_map[s_clean] = yf_ticker
        if pd.notna(d) and str(d).strip():
            try:
                valid_dates.append(pd.to_datetime(d).tz_localize(None))
            except Exception:
                pass

    yf_tickers = list(set(yf_map.values()))
    if not yf_tickers:
        return results

    min_date = min(valid_dates) if valid_dates else (datetime.now() - timedelta(days=90))
    start_dt = min_date - timedelta(days=10)

    try:
        raw_df = yf.download(yf_tickers, start=start_dt, progress=False, threads=True)
        if raw_df.empty:
            return results
        close_df = raw_df["Close"] if "Close" in raw_df.columns else raw_df
        vol_df = raw_df["Volume"] if "Volume" in raw_df.columns else None
        high_df = raw_df["High"] if "High" in raw_df.columns else None
        low_df = raw_df["Low"] if "Low" in raw_df.columns else None
        open_df = raw_df["Open"] if "Open" in raw_df.columns else None
    except Exception:
        return results

    def _get_metric_series(df_obj, sym):
        if df_obj is None:
            return None
        if isinstance(df_obj, pd.DataFrame):
            if sym in df_obj.columns:
                return df_obj[sym]
        elif isinstance(df_obj, pd.Series):
            return df_obj
        return None

    for s, d_str in symbol_date_tuples:
        s_clean = str(s).strip()
        yf_sym = yf_map.get(s_clean)
        if not yf_sym:
            continue
        try:
            s_close = _get_metric_series(close_df, yf_sym)
            if s_close is None:
                continue
            s_close = s_close.dropna()
            if len(s_close) < 2:
                continue

            s_vol = _get_metric_series(vol_df, yf_sym)
            s_high = _get_metric_series(high_df, yf_sym)
            s_low = _get_metric_series(low_df, yf_sym)
            s_open = _get_metric_series(open_df, yf_sym)

            # Exact 20-session Average Daily Traded Value (ADTV)
            adtv_20 = None
            try:
                if s_vol is not None:
                    c_clean = s_close.dropna().tail(20)
                    v_clean = s_vol.dropna().tail(20)
                    common_idx = c_clean.index.intersection(v_clean.index)
                    if len(common_idx) >= 3:
                        daily_val = c_clean.loc[common_idx] * v_clean.loc[common_idx]
                        mean_val = float(daily_val.mean())
                        if market == "IN":
                            adtv_20 = round(mean_val / 1e7, 2)
                        else:
                            adtv_20 = round(mean_val / 1e6, 2)
            except Exception:
                adtv_20 = None

            ann_date = pd.to_datetime(d_str).tz_localize(None).date()
            s_dates = [
                idx.tz_localize(None).date() if hasattr(idx, "tz_localize") and idx.tzinfo
                else (idx.tz_convert(None).date() if hasattr(idx, "tz_convert") and idx.tzinfo else idx.date())
                for idx in s_close.index
            ]

            matched_idx = None
            for i in range(len(s_dates) - 1, -1, -1):
                if s_dates[i] <= ann_date:
                    matched_idx = i
                    break

            if matched_idx is None:
                matched_idx = 0

            curr_p = float(s_close.iloc[-1])
            pe_p = float(s_close.iloc[matched_idx])
            match_dt_key = s_close.index[matched_idx]

            day0_rx = None
            prev_p = None
            if matched_idx > 0:
                prev_p = float(s_close.iloc[matched_idx - 1])
                if prev_p > 0:
                    day0_rx = round(((pe_p - prev_p) / prev_p) * 100.0, 2)

            drift_pct = round(((curr_p - pe_p) / pe_p) * 100.0, 2) if pe_p > 0 else 0.0

            # Day 0 OHLC & Closing Range % (Episodic Pivot Anatomy)
            try:
                pe_high = float(s_high.loc[match_dt_key]) if s_high is not None and match_dt_key in s_high.index and pd.notna(s_high.loc[match_dt_key]) else pe_p
            except Exception:
                pe_high = float(s_high.iloc[matched_idx]) if s_high is not None and len(s_high) > matched_idx and pd.notna(s_high.iloc[matched_idx]) else pe_p

            try:
                pe_low = float(s_low.loc[match_dt_key]) if s_low is not None and match_dt_key in s_low.index and pd.notna(s_low.loc[match_dt_key]) else pe_p
            except Exception:
                pe_low = float(s_low.iloc[matched_idx]) if s_low is not None and len(s_low) > matched_idx and pd.notna(s_low.iloc[matched_idx]) else pe_p

            try:
                pe_open = float(s_open.loc[match_dt_key]) if s_open is not None and match_dt_key in s_open.index and pd.notna(s_open.loc[match_dt_key]) else pe_p
            except Exception:
                pe_open = float(s_open.iloc[matched_idx]) if s_open is not None and len(s_open) > matched_idx and pd.notna(s_open.iloc[matched_idx]) else pe_p

            # Calculate Intraday Closing Range: (Close - Low) / (High - Low) * 100
            rng = pe_high - pe_low
            if rng > 0:
                closing_range = round(((pe_p - pe_low) / rng) * 100.0, 1)
                closing_range = max(0.0, min(100.0, closing_range))
            elif prev_p is not None and prev_p > 0:
                # Circuit locked
                closing_range = 100.0 if pe_p >= prev_p else 0.0
            else:
                closing_range = 50.0

            risk_pct = round(((curr_p - pe_low) / curr_p) * 100.0, 1) if curr_p > 0 and pe_low > 0 else None

            results[s_clean] = {
                "day0_rx": day0_rx,
                "pead_drift": drift_pct,
                "pe_p": pe_p,
                "curr_p": curr_p,
                "closing_range": closing_range,
                "day0_high": round(pe_high, 2) if pe_high else None,
                "day0_low": round(pe_low, 2) if pe_low else None,
                "day0_open": round(pe_open, 2) if pe_open else None,
                "risk_pct": risk_pct,
                "adtv": adtv_20
            }
        except Exception:
            continue

    return results


def _format_drift_badge(drift_val):
    """Format visual badge for cumulative post-earnings announcement drift (PEAD Drift %)."""
    if drift_val is None or pd.isna(drift_val):
        return "-"
    try:
        v = float(drift_val)
        if v >= 15.0:
            return f"🔥 +{v:.1f}% 🚀"
        elif v >= 5.0:
            return f"🟢 +{v:.1f}% ↗"
        elif v > 0.0:
            return f"🟢 +{v:.1f}% ↗"
        elif v == 0.0:
            return "⚪ 0.00%"
        elif v > -5.0:
            return f"🔴 {v:.1f}% ↘"
        else:
            return f"⚠️ {v:.1f}% 📉"
    except Exception:
        return "-"


def _aggregate_thematic_contagion(df):
    """
    CANSLIM Group Strength & Sector Contagion Engine:
    Aggregates declared quarterly results by Sector and identifies whether
    institutional accumulation or distribution is spreading across peer groups.
    Computes both Day 0 reaction and cumulative post-earnings drift across peers.
    """
    if df.empty or "sector" not in df.columns:
        return pd.DataFrame()

    records = []
    work_df = df.copy()
    work_df["sector"] = work_df["sector"].fillna("Other").astype(str).str.strip()
    work_df = work_df[work_df["sector"] != ""]

    for sector, group in work_df.groupby("sector"):
        total = len(group)
        f_eps = pd.to_numeric(group["surprise"], errors="coerce")
        f_rev = pd.to_numeric(group.get("rev_surprise", pd.Series(dtype=float)), errors="coerce")
        dual_beats = int(((f_eps > 0) & (f_rev > 0)).sum())
        eps_beats = int((f_eps > 0).sum())
        rev_beats = int((f_rev > 0).sum())
        dual_rate = round((dual_beats / max(1, total)) * 100, 1)

        # Prefer day0_rx if present, fallback to change
        rx_series = group.get("day0_rx", group.get("change", pd.Series(dtype=float)))
        changes = pd.to_numeric(rx_series, errors="coerce").dropna().clip(lower=-100.0, upper=200.0)
        rel_vols = pd.to_numeric(group.get("rel_vol", pd.Series(dtype=float)), errors="coerce").dropna()
        drifts = pd.to_numeric(group.get("pead_drift", pd.Series(dtype=float)), errors="coerce").dropna()

        avg_change = round(float(changes.mean()), 2) if not changes.empty else 0.0
        avg_rel_vol = round(float(rel_vols.mean()), 2) if not rel_vols.empty else 1.0
        avg_drift = round(float(drifts.mean()), 2) if not drifts.empty else 0.0

        top_stock = "-"
        if not drifts.empty:
            sorted_grp = group.sort_values(by=["pead_drift", "day0_rx" if "day0_rx" in group.columns else "change"], ascending=[False, False])
            top_row = sorted_grp.iloc[0]
            top_drift = top_row.get("pead_drift")
            top_sym = top_row.get("symbol", "")
            top_drift_str = f"+{top_drift:.1f}% drift" if top_drift is not None and top_drift > 0 else (f"{top_drift:.1f}% drift" if top_drift is not None else "")
            top_stock = f"{top_sym} ({top_drift_str})" if top_sym else "-"
        elif not changes.empty:
            sorted_grp = group.sort_values(by=["change", "rel_vol"], ascending=[False, False])
            top_row = sorted_grp.iloc[0]
            top_chg = top_row.get("change")
            top_sym = top_row.get("symbol", "")
            top_chg_str = f"+{top_chg:.1f}%" if top_chg is not None and top_chg > 0 else (f"{top_chg:.1f}%" if top_chg is not None else "")
            top_stock = f"{top_sym} ({top_chg_str})" if top_sym else "-"

        if total >= 2 and dual_rate >= 50.0 and (avg_change > 0.5 or (not drifts.empty and avg_drift > 1.0)):
            impulse = "🔥 Strong Group Accumulation"
            badge = "HOT_CONTAGION"
        elif total >= 2 and (avg_change < -1.0 or (not drifts.empty and avg_drift < -1.0)):
            impulse = "⚠️ Sector Drag / Fade"
            badge = "DRAG"
        elif dual_beats > 0 or (not drifts.empty and avg_drift > 0.0):
            impulse = "🟢 Emerging Accumulation"
            badge = "EMERGING"
        else:
            impulse = "⚖️ Mixed Reaction"
            badge = "MIXED"

        records.append({
            "Sector": sector,
            "Reported": total,
            "Dual Beats": dual_beats,
            "Dual Beat Rate": dual_rate,
            "EPS Beats": eps_beats,
            "Sales Beats": rev_beats,
            "Avg Reaction %": avg_change,
            "Avg Drift %": avg_drift,
            "Avg Rel Vol": avg_rel_vol,
            "Top Leader": top_stock,
            "Contagion Impulse": impulse,
            "badge": badge
        })

    if not records:
        return pd.DataFrame()

    cdf = pd.DataFrame(records)
    cdf = cdf.sort_values(by=["Dual Beat Rate", "Avg Drift %", "Avg Reaction %", "Reported"], ascending=[False, False, False, False])
    return cdf


def _render_thematic_contagion_banner(contagion_df, market="IN"):
    """
    Renders high-visibility CANSLIM Sector Contagion Alert Banner above the workspace.
    Pinpoints the leading institutional theme driving earnings season momentum.
    """
    if contagion_df.empty:
        return

    hot_sectors = contagion_df[contagion_df["badge"] == "HOT_CONTAGION"]
    if not hot_sectors.empty:
        top_sec = hot_sectors.iloc[0]
        banner_title = "🔥 ACTIVE THEMATIC ACCUMULATION CONTAGION"
        banner_border = "#10b981"
        banner_bg = "linear-gradient(135deg, rgba(6,78,59,0.3) 0%, rgba(15,23,42,0.85) 100%)"
        headline_color = "#34d399"
    else:
        top_sec = contagion_df.iloc[0]
        banner_title = "📡 SECTOR CONTAGION RADAR"
        banner_border = "#38bdf8"
        banner_bg = "linear-gradient(135deg, rgba(12,74,110,0.25) 0%, rgba(15,23,42,0.85) 100%)"
        headline_color = "#38bdf8"

    sec_name = top_sec["Sector"]
    rep_count = top_sec["Reported"]
    d_rate = top_sec["Dual Beat Rate"]
    avg_rx = top_sec["Avg Reaction %"]
    avg_rv = top_sec["Avg Rel Vol"]
    top_lead = top_sec["Top Leader"]
    rx_str = f"+{avg_rx:.2f}%" if avg_rx > 0 else f"{avg_rx:.2f}%"

    st.markdown(f"""
    <div style="background: {banner_bg}; border: 1px solid {banner_border}; border-left: 4px solid {banner_border}; border-radius: 10px; padding: 10px 16px; margin: 6px 0 12px 0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; box-shadow: 0 6px 20px -4px rgba(0,0,0,0.4);">
        <div style="display: flex; flex-direction: column; gap: 2px;">
            <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.1px; color: {headline_color};">
                {banner_title} &bull; {rep_count} PEERS REPORTED
            </div>
            <div style="font-size: 1.02rem; font-weight: 800; color: #f8fafc; font-family: 'Inter', sans-serif;">
                Theme Leader: <span style="color: {headline_color};">{sec_name}</span>
            </div>
            <div style="font-size: 0.74rem; color: #cbd5e1;">
                Dual Beat Rate: <b>{d_rate}%</b> | Avg Post-Earnings Reaction: <b>{rx_str}</b> ({avg_rv:.1f}x Vol) | Pace Setter: <b>{top_lead}</b>
            </div>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
            <span style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 6px 12px; font-size: 0.72rem; color: #f8fafc; font-weight: 600;">
                CANSLIM Rule: 37% of move stems from Group Strength ↗
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_thematic_contagion_matrix(contagion_df, upcoming_df=None, market="IN"):
    """
    Renders expandable Sector Contagion Matrix with Unreported Peer Sympathy Watchlist.
    """
    if contagion_df.empty:
        return

    with st.expander("🌐 Thematic Contagion Matrix & Unreported Peer Sympathy Radar", expanded=False):
        st.caption(
            "Track institutional capital flows across industry sectors. High dual beat rates combined with positive average price reaction "
            "signal group accumulation. Unreported peers in leading sectors often exhibit strong pre-earnings drift and sympathy momentum."
        )
        cols = ["Sector", "Reported", "Dual Beats", "Dual Beat Rate"]
        if "Avg Drift %" in contagion_df.columns:
            cols.append("Avg Drift %")
        cols.extend(["Avg Reaction %", "Avg Rel Vol", "Top Leader", "Contagion Impulse"])
        c_table = contagion_df[cols].copy()

        c_table["Dual Beat Rate"] = c_table["Dual Beat Rate"].apply(lambda v: f"{v:.1f}%")
        if "Avg Drift %" in c_table.columns:
            c_table["Avg Drift %"] = c_table["Avg Drift %"].apply(lambda v: f"+{v:.2f}%" if v > 0 else f"{v:.2f}%")
        c_table["Avg Reaction %"] = c_table["Avg Reaction %"].apply(lambda v: f"+{v:.2f}%" if v > 0 else f"{v:.2f}%")
        c_table["Avg Rel Vol"] = c_table["Avg Rel Vol"].apply(lambda v: f"{v:.1f}x")

        matrix_col_cfg = {
            "Sector": st.column_config.TextColumn("Industry Sector", width="medium"),
            "Reported": st.column_config.NumberColumn("Reported", width="small"),
            "Dual Beats": st.column_config.NumberColumn("Dual Beats", width="small"),
            "Dual Beat Rate": st.column_config.TextColumn("Dual Beat %", width="small"),
            "Avg Drift %": st.column_config.TextColumn("Avg Drift", width="small", help="Average post-earnings drift return across reported sector peers"),
            "Avg Reaction %": st.column_config.TextColumn("Avg Day0 Rx", width="small", help="Average announcement day reaction candle return"),
            "Avg Rel Vol": st.column_config.TextColumn("Avg Vol Factor", width="small"),
            "Top Leader": st.column_config.TextColumn("Top Accumulator", width="medium"),
            "Contagion Impulse": st.column_config.TextColumn("Institutional Contagion Status", width="large")
        }

        st.dataframe(
            c_table,
            column_config=matrix_col_cfg,
            use_container_width=True,
            hide_index=True
        )

        if upcoming_df is not None and not upcoming_df.empty and "symbol" in upcoming_df.columns:
            hot_sectors_list = contagion_df[contagion_df["Dual Beat Rate"] >= 50.0]["Sector"].tolist()
            if hot_sectors_list:
                st.markdown(f"""
                <div style="font-size: 0.78rem; font-weight: 700; color: #38bdf8; margin: 10px 0 4px 0;">
                    🎯 UPCOMING SYMPATHY WATCHLIST (Upcoming Releases in Hot Accumulation Sectors: {', '.join(hot_sectors_list[:3])})
                </div>
                """, unsafe_allow_html=True)
                cols_to_use = [c for c in ["symbol", "company", "date", "Timing"] if c in upcoming_df.columns]
                up_show = upcoming_df[cols_to_use].head(10).rename(columns={
                    "symbol": "Ticker", "company": "Company Name", "date": "Reporting Date"
                })
                st.dataframe(up_show, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# WORLD-CLASS VISUAL HUD CARDS & QUADRANT VISUALIZERS
# -----------------------------------------------------------------------------
def _render_earnings_hud_cards(total, dual_count, dual_rate, eps_count, eps_rate, rev_count, rev_rate, season_label="Active Season"):
    """Render high-contrast, glowing glassmorphic HUD metric cards with micro progress meters."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div style="background: linear-gradient(145deg, rgba(15,23,42,0.92) 0%, rgba(30,41,59,0.75) 100%); border: 1px solid rgba(56,189,248,0.22); border-top: 3px solid #38bdf8; border-radius: 12px; padding: 14px 16px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); min-height: 124px;">
<div style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #94a3b8;">TOTAL DECLARED RESULTS</div>
<div style="font-size: 1.95rem; font-weight: 800; color: #f8fafc; font-family: 'JetBrains Mono', monospace; margin: 2px 0 4px 0;">{total} <span style="font-size: 0.90rem; font-weight: 500; color: #64748b;">Stocks</span></div>
<div style="display: inline-block; padding: 2px 8px; background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.3); border-radius: 999px; font-size: 0.72rem; color: #38bdf8; font-weight: 600;">📅 {season_label}</div>
<div style="width: 100%; background: rgba(255,255,255,0.06); height: 4px; border-radius: 999px; margin-top: 10px; overflow: hidden;"><div style="width: 100%; background: linear-gradient(90deg, #0284c7, #38bdf8); height: 100%; border-radius: 999px;"></div></div>
</div>""",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""<div style="background: linear-gradient(145deg, rgba(6,78,59,0.35) 0%, rgba(15,23,42,0.92) 100%); border: 1px solid rgba(16,185,129,0.3); border-top: 3px solid #10b981; border-radius: 12px; padding: 14px 16px; box-shadow: 0 10px 25px -5px rgba(16,185,129,0.15); min-height: 124px;">
<div style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #6ee7b7;">🔥 DUAL BEATS (EPS & SALES)</div>
<div style="font-size: 1.95rem; font-weight: 800; color: #34d399; font-family: 'JetBrains Mono', monospace; margin: 2px 0 4px 0;">{dual_count} <span style="font-size: 0.90rem; font-weight: 500; color: #94a3b8;">Stocks</span></div>
<div style="display: inline-block; padding: 2px 8px; background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.35); border-radius: 999px; font-size: 0.72rem; color: #34d399; font-weight: 700;">↑ {dual_rate}% Dual Beat Rate &bull; Golden Zone</div>
<div style="width: 100%; background: rgba(255,255,255,0.06); height: 4px; border-radius: 999px; margin-top: 10px; overflow: hidden;"><div style="width: {min(100, float(dual_rate))}%; background: linear-gradient(90deg, #059669, #34d399); height: 100%; border-radius: 999px;"></div></div>
</div>""",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""<div style="background: linear-gradient(145deg, rgba(12,74,110,0.35) 0%, rgba(15,23,42,0.92) 100%); border: 1px solid rgba(56,189,248,0.3); border-top: 3px solid #38bdf8; border-radius: 12px; padding: 14px 16px; box-shadow: 0 10px 25px -5px rgba(56,189,248,0.12); min-height: 124px;">
<div style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #7dd3fc;">CONSENSUS EPS BEATS</div>
<div style="font-size: 1.95rem; font-weight: 800; color: #38bdf8; font-family: 'JetBrains Mono', monospace; margin: 2px 0 4px 0;">{eps_count} <span style="font-size: 0.90rem; font-weight: 500; color: #94a3b8;">Stocks</span></div>
<div style="display: inline-block; padding: 2px 8px; background: rgba(56,189,248,0.15); border: 1px solid rgba(56,189,248,0.35); border-radius: 999px; font-size: 0.72rem; color: #38bdf8; font-weight: 700;">↑ {eps_rate}% EPS Beat Rate &bull; Margin Focus</div>
<div style="width: 100%; background: rgba(255,255,255,0.06); height: 4px; border-radius: 999px; margin-top: 10px; overflow: hidden;"><div style="width: {min(100, float(eps_rate))}%; background: linear-gradient(90deg, #0284c7, #38bdf8); height: 100%; border-radius: 999px;"></div></div>
</div>""",
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""<div style="background: linear-gradient(145deg, rgba(88,28,135,0.35) 0%, rgba(15,23,42,0.92) 100%); border: 1px solid rgba(168,85,247,0.3); border-top: 3px solid #c084fc; border-radius: 12px; padding: 14px 16px; box-shadow: 0 10px 25px -5px rgba(168,85,247,0.12); min-height: 124px;">
<div style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #d8b4fe;">CONSENSUS SALES BEATS</div>
<div style="font-size: 1.95rem; font-weight: 800; color: #c084fc; font-family: 'JetBrains Mono', monospace; margin: 2px 0 4px 0;">{rev_count} <span style="font-size: 0.90rem; font-weight: 500; color: #94a3b8;">Stocks</span></div>
<div style="display: inline-block; padding: 2px 8px; background: rgba(168,85,247,0.15); border: 1px solid rgba(168,85,247,0.35); border-radius: 999px; font-size: 0.72rem; color: #d8b4fe; font-weight: 700;">↑ {rev_rate}% Sales Beat Rate &bull; Top-Line</div>
<div style="width: 100%; background: rgba(255,255,255,0.06); height: 4px; border-radius: 999px; margin-top: 10px; overflow: hidden;"><div style="width: {min(100, float(rev_rate))}%; background: linear-gradient(90deg, #7e22ce, #c084fc); height: 100%; border-radius: 999px;"></div></div>
</div>""",
            unsafe_allow_html=True
        )
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


def _render_upcoming_hud_cards(total, primary_count, next_7d, next_14d, is_india=True):
    """Render high-contrast, glowing glassmorphic HUD metric cards for upcoming earnings."""
    primary_label = "NIFTY 500 REPORTING" if is_india else "PENDING ANNOUNCEMENTS"
    primary_badge = "Index Leaders" if is_india else "Binary Catalysts"
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div style="background: linear-gradient(135deg, rgba(15,23,42,0.85) 0%, rgba(30,41,59,0.7) 100%); border: 1px solid rgba(56,189,248,0.25); border-top: 3px solid #38bdf8; border-radius: 14px; padding: 16px 18px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); min-height: 120px;">
<div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #94a3b8;">TOTAL ANNOUNCED RELEASES</div>
<div style="font-size: 1.95rem; font-weight: 800; color: #f8fafc; font-family: 'JetBrains Mono', monospace; margin: 4px 0;">{total} <span style="font-size: 0.95rem; font-weight: 500; color: #64748b;">Companies</span></div>
<div style="font-size: 0.75rem; color: #38bdf8; font-weight: 600;">📅 Upcoming Calendar</div>
</div>""",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""<div style="background: linear-gradient(135deg, rgba(30,27,75,0.7) 0%, rgba(15,23,42,0.9) 100%); border: 1px solid rgba(168,85,247,0.35); border-top: 3px solid #a855f7; border-radius: 14px; padding: 16px 18px; box-shadow: 0 10px 25px -5px rgba(168,85,247,0.15); min-height: 120px;">
<div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #c084fc;">{primary_label}</div>
<div style="font-size: 1.95rem; font-weight: 800; color: #d8b4fe; font-family: 'JetBrains Mono', monospace; margin: 4px 0;">{primary_count} <span style="font-size: 0.95rem; font-weight: 500; color: #94a3b8;">Stocks</span></div>
<div style="display: inline-block; padding: 2px 8px; background: rgba(168,85,247,0.18); border: 1px solid rgba(168,85,247,0.4); border-radius: 999px; font-size: 0.72rem; color: #e9d5ff; font-weight: 700;">👑 {primary_badge}</div>
</div>""",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""<div style="background: linear-gradient(135deg, rgba(76,5,25,0.4) 0%, rgba(15,23,42,0.9) 100%); border: 1px solid rgba(239,68,68,0.35); border-top: 3px solid #ef4444; border-radius: 14px; padding: 16px 18px; box-shadow: 0 10px 25px -5px rgba(239,68,68,0.15); min-height: 120px;">
<div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #fca5a5;">IMMINENT (NEXT 7 DAYS)</div>
<div style="font-size: 1.95rem; font-weight: 800; color: #f87171; font-family: 'JetBrains Mono', monospace; margin: 4px 0;">{next_7d} <span style="font-size: 0.95rem; font-weight: 500; color: #94a3b8;">Stocks</span></div>
<div style="display: inline-block; padding: 2px 8px; background: rgba(239,68,68,0.18); border: 1px solid rgba(239,68,68,0.4); border-radius: 999px; font-size: 0.72rem; color: #fca5a5; font-weight: 700;">⚠️ High Binary Gap Risk</div>
</div>""",
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""<div style="background: linear-gradient(135deg, rgba(30,58,138,0.3) 0%, rgba(15,23,42,0.9) 100%); border: 1px solid rgba(59,130,246,0.3); border-top: 3px solid #3b82f6; border-radius: 14px; padding: 16px 18px; box-shadow: 0 10px 25px -5px rgba(59,130,246,0.1); min-height: 120px;">
<div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #93c5fd;">NEAR-TERM (NEXT 14 DAYS)</div>
<div style="font-size: 1.95rem; font-weight: 800; color: #60a5fa; font-family: 'JetBrains Mono', monospace; margin: 4px 0;">{next_14d} <span style="font-size: 0.95rem; font-weight: 500; color: #94a3b8;">Stocks</span></div>
<div style="display: inline-block; padding: 2px 8px; background: rgba(59,130,246,0.18); border: 1px solid rgba(59,130,246,0.4); border-radius: 999px; font-size: 0.72rem; color: #bfdbfe; font-weight: 700;">⏳ 2-Week Window</div>
</div>""",
            unsafe_allow_html=True
        )
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


def _parse_recency_days(date_series):
    """Calculate days ago relative to the latest date in the dataset (or today)."""
    clean_dates = pd.to_datetime(date_series.astype(str).str[:10], errors="coerce")
    valid_dates = clean_dates.dropna()
    if valid_dates.empty:
        return pd.Series(999, index=date_series.index)
    ref_date = max(pd.to_datetime(datetime.now().date()), valid_dates.max())
    return (ref_date - clean_dates).dt.days


def get_active_reporting_season(date_series=None, market="IN"):
    """
    Dynamically determines the active corporate reporting season,
    the cutoff date (start of reporting window), and human-friendly badge labels.
    Never hardcoded: works automatically across any calendar year and quarter.
    """
    ref_date = datetime.now().date()
    if date_series is not None and hasattr(date_series, "dropna"):
        clean_dates = pd.to_datetime(date_series.astype(str).str[:10], errors="coerce").dropna()
        if not clean_dates.empty:
            ref_date = max(pd.to_datetime(datetime.now().date()), clean_dates.max()).date()

    month = ref_date.month
    year = ref_date.year

    season_start_month = ((month - 1) // 3) * 3 + 1
    season_start_date = date(year, season_start_month, 1)
    season_cutoff_str = season_start_date.strftime("%Y-%m-%d")

    month_name = season_start_date.strftime("%b")
    yr_2digit = season_start_date.strftime("%y")

    if market.upper() in ("IN", "INDIA"):
        in_q_map = {10: ("Q2", "Q1"), 1: ("Q3", "Q2"), 4: ("Q4", "Q3"), 7: ("Q1", "Q4")}
        current_q, prior_q = in_q_map[season_start_month]
    else:
        us_q_map = {10: ("Q3", "Q2"), 1: ("Q4", "Q3"), 4: ("Q1", "Q4"), 7: ("Q2", "Q1")}
        current_q, prior_q = us_q_map[season_start_month]

    return {
        "cutoff_date": season_cutoff_str,
        "season_name": f"{month_name} '{yr_2digit}",
        "full_season_name": f"{current_q} ({month_name} {year})",
        "current_badge": f"🎯 {current_q} ({month_name} '{yr_2digit})",
        "prior_badge": f"Prior {prior_q}",
        "current_q": current_q,
        "prior_q": prior_q,
        "preset_pill": f"🎯 Current Season ({current_q})",
        "cohort_label_current": f"🎯 Current Season Only ({current_q} Ongoing)",
        "cohort_label_full": f"⏳ Full 90-Day Cycle ({current_q} + Prior {prior_q})",
        "cohort_label_prior": f"📁 Prior Quarter Only ({prior_q})",
        "recency_dropdown": f"🎯 Current Active Season ({current_q} Ongoing)",
    }


def _assign_rs_color(row_or_val):
    """Assign CANSLIM RS heatmap hex color based on RS percentile score."""
    try:
        val = row_or_val.get("rs_raw_pct") if hasattr(row_or_val, "get") else row_or_val
        if pd.isna(val) or val is None:
            return "#64748b"  # Neutral Slate for unranked
        v = float(val)
        if v >= 80.0:
            return "#10b981"  # Emerald: CANSLIM Leaders (RS >= 80)
        elif v >= 70.0:
            return "#38bdf8"  # Cyan: Strong Outperformers (RS 70-79)
        elif v >= 50.0:
            return "#94a3b8"  # Slate Grey: Average Market Performers (RS 50-69)
        elif v >= 30.0:
            return "#f59e0b"  # Amber Orange: Lagging (RS 30-49)
        else:
            return "#ef4444"  # Crimson Red: Severe Underperformers (RS < 30)
    except Exception:
        return "#64748b"


def _calc_bubble_size(row_or_val):
    """Dynamic marker sizing scaled from 8px (RS 0) to 26px (RS 99)."""
    try:
        val = row_or_val.get("rs_raw_pct") if hasattr(row_or_val, "get") else row_or_val
        if pd.isna(val) or val is None:
            return 11.0
        v = float(val)
        return max(8.0, min(26.0, 8.0 + (v / 99.0) * 18.0))
    except Exception:
        return 11.0


def _calc_line_style(row_or_val):
    """Border halo: RS >= 80 gets 2.5px gold halo, RS >= 70 gets 1.8px white halo."""
    try:
        val = row_or_val.get("rs_raw_pct") if hasattr(row_or_val, "get") else row_or_val
        if pd.notna(val) and float(val) >= 80.0:
            return 2.5, "#fef08a"
        elif pd.notna(val) and float(val) >= 70.0:
            return 1.8, "rgba(255,255,255,0.95)"
        return 1.2, "rgba(255,255,255,0.65)"
    except Exception:
        return 1.2, "rgba(255,255,255,0.65)"


def _render_surprise_quadrant_chart(df, market_currency="₹", height=540, chart_key="pead_in_quadrant_chart"):
    """
    Render institutional 4-quadrant scatter chart:
    - Mode 1: Revenue Surprise % vs EPS Surprise % (Consensus Dual Surprises)
    - Mode 2: Quarterly YoY Sales Growth % vs PEAD Drift % (Fundamental Sales vs Drift Momentum)
    - Mode 3: Quarterly YoY Diluted EPS Growth % vs PEAD Drift % (CANSLIM EPS vs Drift Momentum)
    Supports bi-directional Plotly point click to isolate stocks.
    """
    if df.empty:
        st.info("ℹ️ No data available for quadrant visualization.")
        return None

    # Quadrant Matrix Analysis Mode Selector
    quad_mode_opts = [
        "📊 Dual Consensus Surprises (Rev % vs EPS %)",
        "⚡ Sales Growth YoY vs PEAD Drift %",
        "🔥 EPS Growth YoY vs PEAD Drift %"
    ]
    mode_key = f"{chart_key}_mode_sel"
    if hasattr(st, "segmented_control"):
        quadrant_mode = st.segmented_control(
            "Select Quadrant Analysis Matrix:",
            quad_mode_opts,
            default=quad_mode_opts[0],
            key=mode_key,
            label_visibility="collapsed"
        )
    else:
        quadrant_mode = st.radio(
            "Select Quadrant Analysis Matrix:",
            quad_mode_opts,
            index=0,
            horizontal=True,
            key=mode_key,
            label_visibility="collapsed"
        )
    if not quadrant_mode:
        quadrant_mode = quad_mode_opts[0]

    # Reset chart selection if mode changed
    st.session_state.pop(chart_key, None)
    prev_mode_key = f"{chart_key}_prev_mode"
    chart_ver_key = f"{chart_key}_ver"
    if st.session_state.get(prev_mode_key) != quadrant_mode:
        st.session_state[prev_mode_key] = quadrant_mode
        st.session_state[chart_ver_key] = st.session_state.get(chart_ver_key, 0) + 1

    plot_df = df.copy()

    if "Sales Growth YoY" in quadrant_mode:
        plot_df["x_metric"] = pd.to_numeric(plot_df.get("rev_yoy"), errors="coerce")
        plot_df["y_metric"] = pd.to_numeric(plot_df.get("pead_drift"), errors="coerce")
        valid_plot = plot_df.dropna(subset=["x_metric", "y_metric"]).copy()
        if valid_plot.empty:
            st.info("ℹ️ No stocks with both YoY sales growth and PEAD drift available in this filter.")
            return None
        header_title = "⚡ Fundamental Sales Growth vs PEAD Drift Quadrant"
        border_accent = "#10b981"
        sub_dim = "X: Quarterly YoY Sales Growth (%) &bull; Y: Post-Earnings Drift (%)"
        x_title = "Quarterly YoY Sales Growth (%)"
        y_title = "Post-Earnings Announcement Drift (%)"
    elif "EPS Growth YoY" in quadrant_mode:
        plot_df["x_metric"] = pd.to_numeric(plot_df.get("eps_yoy"), errors="coerce")
        plot_df["y_metric"] = pd.to_numeric(plot_df.get("pead_drift"), errors="coerce")
        valid_plot = plot_df.dropna(subset=["x_metric", "y_metric"]).copy()
        if valid_plot.empty:
            st.info("ℹ️ No stocks with both YoY EPS growth and PEAD drift available in this filter.")
            return None
        header_title = "🔥 Fundamental EPS Growth vs PEAD Drift Quadrant"
        border_accent = "#f59e0b"
        sub_dim = "X: Quarterly YoY Diluted EPS Growth (%) &bull; Y: Post-Earnings Drift (%)"
        x_title = "Quarterly YoY Diluted EPS Growth (%)"
        y_title = "Post-Earnings Announcement Drift (%)"
    else:
        plot_df["x_metric"] = pd.to_numeric(plot_df.get("rev_surprise"), errors="coerce")
        plot_df["y_metric"] = pd.to_numeric(plot_df.get("surprise"), errors="coerce")
        valid_plot = plot_df.dropna(subset=["x_metric", "y_metric"]).copy()
        if valid_plot.empty:
            st.info("ℹ️ No stocks with dual consensus surprise values available in this filter.")
            return None
        header_title = "📊 Institutional Surprise Quadrant Map"
        border_accent = "#38bdf8"
        sub_dim = "X: Revenue Surprise (%) &bull; Y: EPS Surprise (%)"
        x_title = "Revenue Surprise (%)"
        y_title = "EPS Surprise (%)"

    # Dynamic Header Banner
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 7px 12px; background: rgba(15,23,42,0.7); border: 1px solid rgba(255,255,255,0.08); border-left: 3px solid {border_accent}; border-radius: 8px; margin: 4px 0 8px 0;">
        <div>
            <div style="font-size: 0.88rem; font-weight: 700; color: #f8fafc;">{header_title}</div>
            <div style="font-size: 0.70rem; color: #94a3b8; margin-top: 1px;">{sub_dim}</div>
        </div>
        <div style="font-size: 0.72rem; color: #34d399; font-weight: 600; background: rgba(16,185,129,0.15); padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(16,185,129,0.3);">{len(valid_plot)} Plotted &bull; Color & Size = RS Score (Click to isolate)</div>
    </div>
    """, unsafe_allow_html=True)

    valid_plot["color"] = valid_plot.apply(_assign_rs_color, axis=1)
    marker_sizes = valid_plot.apply(_calc_bubble_size, axis=1).tolist()
    line_styles = valid_plot.apply(_calc_line_style, axis=1).tolist()
    marker_line_widths = [s[0] for s in line_styles]
    marker_line_colors = [s[1] for s in line_styles]

    drift_setup_arr = valid_plot["PEAD Drift Setup"].fillna("").astype(str).values if "PEAD Drift Setup" in valid_plot.columns else np.array([""] * len(valid_plot))
    reaction_arr = valid_plot["Reaction Candle"].fillna("-").astype(str).values if "Reaction Candle" in valid_plot.columns else np.array(["-"] * len(valid_plot))
    rs_arr = valid_plot["RS Rating"].fillna("-").astype(str).values if "RS Rating" in valid_plot.columns else np.array(["-"] * len(valid_plot))
    sector_arr = valid_plot["sector"].fillna("-").astype(str).values if "sector" in valid_plot.columns else np.array(["-"] * len(valid_plot))
    drift_badge_arr = valid_plot["PEAD Drift"].fillna("-").astype(str).values if "PEAD Drift" in valid_plot.columns else np.array(["-"] * len(valid_plot))
    eps_yoy_arr = valid_plot["YoY EPS %"].fillna("-").astype(str).values if "YoY EPS %" in valid_plot.columns else np.array(["-"] * len(valid_plot))
    rev_yoy_arr = valid_plot["YoY Sales %"].fillna("-").astype(str).values if "YoY Sales %" in valid_plot.columns else np.array(["-"] * len(valid_plot))
    cr_arr = valid_plot["Closing Range %"].fillna("-").astype(str).values if "Closing Range %" in valid_plot.columns else np.array(["-"] * len(valid_plot))
    ep_arr = valid_plot["EP Setup"].fillna("-").astype(str).values if "EP Setup" in valid_plot.columns else np.array(["-"] * len(valid_plot))

    eps_surp_str = valid_plot["surprise"].apply(lambda v: f"{float(v):+.2f}%" if pd.notna(v) and v is not None else "-").values
    rev_surp_str = valid_plot["rev_surprise"].apply(lambda v: f"{float(v):+.2f}%" if pd.notna(v) and v is not None else "-").values

    custom_data = np.stack((
        valid_plot["company"].fillna("").astype(str).values,
        valid_plot["date"].astype(str).values,
        drift_setup_arr,
        reaction_arr,
        rs_arr,
        sector_arr,
        drift_badge_arr,
        eps_yoy_arr,
        rev_yoy_arr,
        cr_arr,
        ep_arr,
        eps_surp_str,
        rev_surp_str
    ), axis=-1)

    x_vals = valid_plot["x_metric"]
    y_vals = valid_plot["y_metric"]

    if "Sales Growth YoY" in quadrant_mode:
        x_p01 = float(np.percentile(x_vals, 1)) if len(x_vals) > 10 else float(x_vals.min())
        x_p98 = float(np.percentile(x_vals, 98)) if len(x_vals) > 10 else float(x_vals.max())
        y_min, y_max = float(y_vals.min()), float(y_vals.max())

        x_min_bound = max(-50.0, x_p01 - 8.0)
        x_max_bound = min(150.0, max(50.0, x_p98 + 15.0))
        y_pad = max(6.0, (y_max - y_min) * 0.16)

        x_range = [min(-5.0, x_min_bound), max(25.0, x_max_bound)]
        y_range = [min(-5.0, y_min - y_pad), max(5.0, y_max + y_pad)]

        hovertemplate = (
            "<b>%{text}</b> - %{customdata[0]}<br>"
            "📅 Date: %{customdata[1]}<br>"
            "⚡ <b>YoY Sales Growth: %{x:+.2f}%</b><br>"
            "📈 <b>PEAD Drift: %{y:+.2f}%</b><br>"
            "🚀 YoY EPS Growth: <b>%{customdata[7]}</b><br>"
            "🎯 Consensus EPS Surp: <b>%{customdata[11]}</b> | Sales Surp: <b>%{customdata[12]}</b><br>"
            "🕯️ Day 0 Rx: <b>%{customdata[3]}</b><br>"
            "🏆 EP Setup: <b>%{customdata[10]}</b> (CR: <b>%{customdata[9]}</b>)<br>"
            "💪 RS Rating: <b>%{customdata[4]}</b> | Group: <b>%{customdata[5]}</b><br>"
            "🎯 PEAD Setup: %{customdata[2]}<extra></extra>"
        )
    elif "EPS Growth YoY" in quadrant_mode:
        x_p01 = float(np.percentile(x_vals, 1)) if len(x_vals) > 10 else float(x_vals.min())
        x_p98 = float(np.percentile(x_vals, 98)) if len(x_vals) > 10 else float(x_vals.max())
        y_min, y_max = float(y_vals.min()), float(y_vals.max())

        x_min_bound = max(-60.0, x_p01 - 10.0)
        x_max_bound = min(200.0, max(60.0, x_p98 + 20.0))
        y_pad = max(6.0, (y_max - y_min) * 0.16)

        x_range = [min(-5.0, x_min_bound), max(30.0, x_max_bound)]
        y_range = [min(-5.0, y_min - y_pad), max(5.0, y_max + y_pad)]

        hovertemplate = (
            "<b>%{text}</b> - %{customdata[0]}<br>"
            "📅 Date: %{customdata[1]}<br>"
            "🚀 <b>YoY EPS Growth: %{x:+.2f}%</b><br>"
            "📈 <b>PEAD Drift: %{y:+.2f}%</b><br>"
            "⚡ YoY Sales Growth: <b>%{customdata[8]}</b><br>"
            "🎯 Consensus EPS Surp: <b>%{customdata[11]}</b> | Sales Surp: <b>%{customdata[12]}</b><br>"
            "🕯️ Day 0 Rx: <b>%{customdata[3]}</b><br>"
            "🏆 EP Setup: <b>%{customdata[10]}</b> (CR: <b>%{customdata[9]}</b>)<br>"
            "💪 RS Rating: <b>%{customdata[4]}</b> | Group: <b>%{customdata[5]}</b><br>"
            "🎯 PEAD Setup: %{customdata[2]}<extra></extra>"
        )
    else:
        x_min, x_max = float(x_vals.min()), float(x_vals.max())
        y_min, y_max = float(y_vals.min()), float(y_vals.max())
        x_span = max(6.0, x_max - x_min)
        y_span = max(10.0, y_max - y_min)
        x_pad = x_span * 0.22
        y_pad = y_span * 0.25
        x_range = [min(-2.5, x_min - x_pad), max(2.5, x_max + x_pad)]
        y_range = [min(-5.0, y_min - y_pad), max(5.0, y_max + y_pad)]

        hovertemplate = (
            "<b>%{text}</b> - %{customdata[0]}<br>"
            "📅 Date: %{customdata[1]}<br>"
            "📈 PEAD Drift: <b>%{customdata[6]}</b><br>"
            "⚡ Sales Surprise: <b>%{x:+.2f}%</b> (YoY: <b>%{customdata[8]}</b>)<br>"
            "🚀 EPS Surprise: <b>%{y:+.2f}%</b> (YoY: <b>%{customdata[7]}</b>)<br>"
            "🕯️ Day 0 Rx: <b>%{customdata[3]}</b><br>"
            "🏆 EP Setup: <b>%{customdata[10]}</b> (CR: <b>%{customdata[9]}</b>)<br>"
            "💪 RS Rating: <b>%{customdata[4]}</b> (Color & Size) | Group: <b>%{customdata[5]}</b><br>"
            "🎯 Setup: %{customdata[2]}<extra></extra>"
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers+text",
        text=valid_plot["symbol"],
        textposition="top right",
        textfont=dict(color="#f8fafc", size=9.5, family="'JetBrains Mono', monospace"),
        marker=dict(
            size=marker_sizes,
            color=valid_plot["color"],
            line=dict(width=marker_line_widths, color=marker_line_colors),
            opacity=0.92
        ),
        customdata=custom_data,
        hovertemplate=hovertemplate
    ))

    # Crosshairs at 0, 0
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.22)", line_width=1.2)
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.22)", line_width=1.2)

    if "Sales Growth YoY" in quadrant_mode:
        fig.add_vline(x=20.0, line_dash="dot", line_color="rgba(56,189,248,0.30)", line_width=1.0)
        fig.add_annotation(
            xref="paper", yref="paper", x=0.98, y=0.98,
            text="🚀 <b>Q1: APEX DRIFT LEADERS</b><br><span style='font-size:8px;color:#cbd5e1;'>Expanding Sales &bull; Positive Drift</span>",
            showarrow=False, xanchor="right", yanchor="top",
            font=dict(color="#34d399", size=9.5, family="Inter, sans-serif"),
            bgcolor="rgba(16,185,129,0.14)", bordercolor="rgba(16,185,129,0.3)", borderpad=3
        )
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98,
            text="⚠️ <b>Q2: MULTIPLE EXPANSION</b><br><span style='font-size:8px;color:#cbd5e1;'>Contracting Sales &bull; Positive Drift</span>",
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(color="#38bdf8", size=9.5, family="Inter, sans-serif"),
            bgcolor="rgba(56,189,248,0.14)", bordercolor="rgba(56,189,248,0.3)", borderpad=3
        )
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.02,
            text="🔴 <b>Q3: INSTITUTIONAL REJECTION</b><br><span style='font-size:8px;color:#cbd5e1;'>Contracting Sales &bull; Negative Drift</span>",
            showarrow=False, xanchor="left", yanchor="bottom",
            font=dict(color="#f87171", size=9.5, family="Inter, sans-serif"),
            bgcolor="rgba(239,68,68,0.14)", bordercolor="rgba(239,68,68,0.3)", borderpad=3
        )
        fig.add_annotation(
            xref="paper", yref="paper", x=0.98, y=0.02,
            text="⚡ <b>Q4: VALUE DISCONNECT / DIP</b><br><span style='font-size:8px;color:#cbd5e1;'>Expanding Sales &bull; Negative Drift</span>",
            showarrow=False, xanchor="right", yanchor="bottom",
            font=dict(color="#fbbf24", size=9.5, family="Inter, sans-serif"),
            bgcolor="rgba(245,158,11,0.14)", bordercolor="rgba(245,158,11,0.3)", borderpad=3
        )
    elif "EPS Growth YoY" in quadrant_mode:
        fig.add_vline(x=25.0, line_dash="dot", line_color="rgba(56,189,248,0.30)", line_width=1.0)
        fig.add_annotation(
            xref="paper", yref="paper", x=0.98, y=0.98,
            text="🔥 <b>Q1: CANSLIM SUPER-DRIFT</b><br><span style='font-size:8px;color:#cbd5e1;'>Expanding EPS &bull; Positive Drift</span>",
            showarrow=False, xanchor="right", yanchor="top",
            font=dict(color="#34d399", size=9.5, family="Inter, sans-serif"),
            bgcolor="rgba(16,185,129,0.14)", bordercolor="rgba(16,185,129,0.3)", borderpad=3
        )
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98,
            text="⚠️ <b>Q2: MULTIPLE EXPANSION</b><br><span style='font-size:8px;color:#cbd5e1;'>Contracting EPS &bull; Positive Drift</span>",
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(color="#38bdf8", size=9.5, family="Inter, sans-serif"),
            bgcolor="rgba(56,189,248,0.14)", bordercolor="rgba(56,189,248,0.3)", borderpad=3
        )
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.02,
            text="🔴 <b>Q3: EARNINGS COLLAPSE</b><br><span style='font-size:8px;color:#cbd5e1;'>Contracting EPS &bull; Negative Drift</span>",
            showarrow=False, xanchor="left", yanchor="bottom",
            font=dict(color="#f87171", size=9.5, family="Inter, sans-serif"),
            bgcolor="rgba(239,68,68,0.14)", bordercolor="rgba(239,68,68,0.3)", borderpad=3
        )
        fig.add_annotation(
            xref="paper", yref="paper", x=0.98, y=0.02,
            text="⚡ <b>Q4: EARNINGS DISCONNECT</b><br><span style='font-size:8px;color:#cbd5e1;'>Expanding EPS &bull; Negative Drift</span>",
            showarrow=False, xanchor="right", yanchor="bottom",
            font=dict(color="#fbbf24", size=9.5, family="Inter, sans-serif"),
            bgcolor="rgba(245,158,11,0.14)", bordercolor="rgba(245,158,11,0.3)", borderpad=3
        )
    else:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.98, y=0.98,
            text="🔥 <b>Q1: DUAL BEATS</b>",
            showarrow=False, xanchor="right", yanchor="top",
            font=dict(color="#34d399", size=9.5, family="Inter, sans-serif"),
            bgcolor="rgba(16,185,129,0.14)", bordercolor="rgba(16,185,129,0.3)", borderpad=3
        )
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98,
            text="🚀 <b>Q2: EPS BEAT</b>",
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(color="#38bdf8", size=9.5, family="Inter, sans-serif"),
            bgcolor="rgba(56,189,248,0.14)", bordercolor="rgba(56,189,248,0.3)", borderpad=3
        )
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.02,
            text="🔴 <b>Q3: DOUBLE MISS</b>",
            showarrow=False, xanchor="left", yanchor="bottom",
            font=dict(color="#f87171", size=9.5, family="Inter, sans-serif"),
            bgcolor="rgba(239,68,68,0.14)", bordercolor="rgba(239,68,68,0.3)", borderpad=3
        )
        fig.add_annotation(
            xref="paper", yref="paper", x=0.98, y=0.02,
            text="⚡ <b>Q4: SALES BEAT</b>",
            showarrow=False, xanchor="right", yanchor="bottom",
            font=dict(color="#fbbf24", size=9.5, family="Inter, sans-serif"),
            bgcolor="rgba(245,158,11,0.14)", bordercolor="rgba(245,158,11,0.3)", borderpad=3
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15,23,42,0.7)",
        plot_bgcolor="rgba(15,23,42,0.7)",
        xaxis=dict(
            title=dict(text=x_title, font=dict(color="#94a3b8", size=11)),
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            range=x_range,
            tickfont=dict(color="#64748b", size=10)
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(color="#94a3b8", size=11)),
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            range=y_range,
            tickfont=dict(color="#64748b", size=10)
        ),
        height=height,
        margin=dict(l=35, r=25, t=25, b=35),
        showlegend=False
    )

    active_chart_key = f"{chart_key}_{st.session_state.get(chart_ver_key, 0)}"
    st.session_state.pop(active_chart_key, None)

    chart_selection = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key=active_chart_key
    )

    selected_symbol = None
    if chart_selection and isinstance(chart_selection, dict) and "selection" in chart_selection:
        pts = chart_selection["selection"].get("points", [])
        if pts and len(pts) > 0:
            pt = pts[0]
            selected_symbol = pt.get("text")
            if not selected_symbol and "point_index" in pt:
                p_idx = pt["point_index"]
                if 0 <= p_idx < len(valid_plot):
                    selected_symbol = valid_plot.iloc[p_idx]["symbol"]

    st.markdown("""
    <div style="display: flex; gap: 8px; justify-content: space-between; align-items: center; margin-top: 4px; padding: 6px 12px; background: rgba(15,23,42,0.65); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; font-size: 0.70rem; flex-wrap: wrap;">
        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
            <span style="color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">CANSLIM RS Heatmap:</span>
            <span style="color: #34d399; font-weight: 700;">🟢 <b>RS ≥ 80</b> (Leaders &bull; Golden Ring)</span>
            <span style="color: #38bdf8; font-weight: 700;">🔵 <b>RS 70–79</b> (Strong)</span>
            <span style="color: #94a3b8; font-weight: 600;">⚪ <b>RS 50–69</b> (Average)</span>
            <span style="color: #f59e0b; font-weight: 600;">🟡 <b>RS 30–49</b> (Weak)</span>
            <span style="color: #f87171; font-weight: 600;">🔴 <b>RS &lt; 30</b> (Laggards)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="color: #38bdf8; font-weight: 700; background: rgba(56,189,248,0.12); padding: 2px 8px; border-radius: 6px; border: 1px solid rgba(56,189,248,0.25);">
                🫧 Bubble Color & Size = CANSLIM RS Score (8–26px)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    return selected_symbol


# -----------------------------------------------------------------------------
# INDIA: DECLARED RESULTS & DUAL SURPRISES VIEW
# -----------------------------------------------------------------------------
def _render_india_declared_earnings(df_raw, nifty500_set, company_map, df_upcoming=None):
    if df_raw.empty:
        st.info("ℹ️ No declared results data available in cache yet.")
        return

    df = df_raw.copy()
    if "company" in df.columns:
        df["company"] = df["company"].fillna(df["symbol"].map(company_map)).fillna(df["symbol"])
    else:
        df["company"] = df["symbol"].map(company_map).fillna(df["symbol"])

    df["is_nifty500"] = df["symbol"].apply(lambda s: s in nifty500_set)
    df["Universe"] = df["is_nifty500"].apply(lambda x: "👑 Nifty 500" if x else "Broad Market")
    df["TradingView"] = df.apply(lambda r: f"https://in.tradingview.com/chart/?symbol={r.get('ticker', 'NSE:' + r['symbol'])}", axis=1)

    df["Dual Surprise Verdict"] = df.apply(lambda r: _format_dual_verdict(r.get("surprise"), r.get("rev_surprise")), axis=1)
    df["PEAD Drift Setup"] = df.apply(lambda r: _format_dual_pead_setup(r.get("surprise"), r.get("rev_surprise")), axis=1)
    df["recency_days"] = _parse_recency_days(df["date"])

    # Vectorized Batch PEAD Drift & Day 0 Announcement Reaction Enrichment
    sym_date_tuples = tuple(zip(df["symbol"].astype(str), df["date"].astype(str)))
    drift_cache = fetch_batch_pead_drift(sym_date_tuples, market="IN")

    def _resolve_day0_rx_in(row):
        s = str(row["symbol"]).strip()
        rec = row.get("recency_days", 999)
        d_info = drift_cache.get(s, {})
        hist_rx = d_info.get("day0_rx")
        if rec <= 1 and pd.notna(row.get("change")):
            return float(row.get("change"))
        elif hist_rx is not None:
            return hist_rx
        elif pd.notna(row.get("change")):
            return float(row.get("change"))
        return 0.0

    def _resolve_drift_pct_in(row):
        s = str(row["symbol"]).strip()
        rec = row.get("recency_days", 999)
        d_info = drift_cache.get(s, {})
        hist_drift = d_info.get("pead_drift")
        if rec == 0:
            return 0.0
        elif hist_drift is not None:
            return hist_drift
        elif rec <= 1:
            return 0.0
        else:
            perf = row.get("perf_1m")
            if pd.notna(perf) and perf is not None:
                return float(perf)
            return 0.0

    df["day0_rx"] = df.apply(_resolve_day0_rx_in, axis=1)
    df["pead_drift"] = df.apply(_resolve_drift_pct_in, axis=1)

    # Market Capitalization Tier Classification
    df["mcap_cr"] = pd.to_numeric(df.get("market_cap"), errors="coerce").fillna(0.0)
    def _format_mcap_badge(mc):
        if mc >= 50000:
            return f"👑 ₹{mc:,.0f} Cr"
        elif mc >= 15000:
            return f"⚡ ₹{mc:,.0f} Cr"
        elif mc >= 5000:
            return f"🌱 ₹{mc:,.0f} Cr"
        elif mc > 0:
            return f"🔬 ₹{mc:,.0f} Cr"
        return "-"
    df["Market Cap"] = df["mcap_cr"].apply(_format_mcap_badge)

    # 20-Day Average Daily Traded Value (ADTV) Resolution & Badge
    def _resolve_adtv_in(row):
        s = str(row["symbol"]).strip()
        d_info = drift_cache.get(s, {})
        drift_adtv = d_info.get("adtv")
        if drift_adtv is not None and drift_adtv > 0:
            return drift_adtv
        tv_adtv = row.get("adtv_cr")
        if pd.notna(tv_adtv) and tv_adtv is not None and tv_adtv > 0:
            return float(tv_adtv)
        c = row.get("close")
        v = row.get("volume")
        if pd.notna(c) and pd.notna(v) and c and v:
            return round((float(c) * float(v)) / 1e7, 2)
        return 0.0

    df["adtv_cr"] = df.apply(_resolve_adtv_in, axis=1)

    def _format_adtv_in_badge(val):
        if val is None or pd.isna(val) or val <= 0:
            return "-"
        v = float(val)
        if v >= 10.0:
            return f"👑 ₹{v:,.1f} Cr"
        elif v >= 5.0:
            return f"🔥 ₹{v:,.1f} Cr"
        elif v >= 1.0:
            return f"⚡ ₹{v:,.1f} Cr"
        elif v >= 0.50:
            return f"💧 ₹{v * 100:,.0f} L"
        else:
            return f"🔬 ₹{v * 100:,.0f} L"

    df["ADTV (20)"] = df["adtv_cr"].apply(_format_adtv_in_badge)

    # Intraday Closing Range & Day 0 Levels from drift_cache
    df["closing_range"] = df["symbol"].apply(lambda s: drift_cache.get(str(s).strip(), {}).get("closing_range"))
    df["day0_high"] = df["symbol"].apply(lambda s: drift_cache.get(str(s).strip(), {}).get("day0_high"))
    df["day0_low"] = df["symbol"].apply(lambda s: drift_cache.get(str(s).strip(), {}).get("day0_low"))
    df["risk_to_stop"] = df["symbol"].apply(lambda s: drift_cache.get(str(s).strip(), {}).get("risk_pct"))

    # CANSLIM Relative Strength & Reaction Candle Enrichment
    df = _compute_relative_strength_ranks(df)
    df["RS Rating"] = df["rs_raw_pct"].apply(_format_rs_badge)
    df["Reaction Candle"] = df.apply(lambda r: _format_reaction_badge(r.get("day0_rx"), r.get("rel_vol"), r.get("closing_range")), axis=1)
    df["Closing Range %"] = df["closing_range"].apply(_format_closing_range_badge)
    df["EP Setup"] = df.apply(lambda r: _classify_episodic_pivot(r.get("day0_rx"), r.get("rel_vol"), r.get("closing_range"), r.get("pead_drift")), axis=1)
    df["PEAD Drift"] = df["pead_drift"].apply(_format_drift_badge)

    # YoY Fundamental Growth & CANSLIM Pillar C Classification
    if "eps_yoy" not in df.columns:
        df["eps_yoy"] = None
    if "rev_yoy" not in df.columns:
        df["rev_yoy"] = None
    df["YoY EPS %"] = df["eps_yoy"].apply(_format_growth_val)
    df["YoY Sales %"] = df["rev_yoy"].apply(_format_growth_val)
    df["CANSLIM Growth"] = df.apply(lambda r: _format_growth_badge(r.get("eps_yoy"), r.get("rev_yoy"), r.get("surprise")), axis=1)

    # Dynamic Seasonal Intelligence (Auto-computes quarter, year, dates, badges dynamically)
    season_info = get_active_reporting_season(df["date"], market="IN")
    cutoff_date = season_info["cutoff_date"]
    current_badge = season_info["current_badge"]
    prior_badge = season_info["prior_badge"]

    df["Season"] = df["date"].apply(lambda d: current_badge if str(d)[:10] >= cutoff_date else prior_badge)

    # --- QUICK STRATEGY PRESET PILLS ---
    quick_preset = "All Strategies"
    if hasattr(st, "pills"):
        quick_preset = st.pills(
            "⚡ Quick Strategy Presets:",
            options=[
                "All Strategies",
                "💎 CANSLIM Super-Growth (EPS ≥ 25% & Sales ≥ 20%)",
                "🏆 Episodic Pivot (EP) Setups (Rx ≥ 3%, CR ≥ 70%)",
                "📈 Positive Drift (> 0%)",
                "🏆 Holy Grail (Dual Beat + RS ≥ 80 + Drift > 0%)",
                "🔥 Dual Beats",
                "🟢 Volume Surge (≥ 2.0x Vol)",
                "🚀 Rocket EPS (≥+5%)",
                "⚡ Top-Line Sales",
                "👑 Nifty 500",
                "📅 Last 7 Days"
            ],
            default="All Strategies",
            key="pead_in_quick_pills"
        )

    # --- FILTER STUDIO HEADER & ACTIONS ---
    fsh_col1, fsh_col2 = st.columns([5, 1.2])
    with fsh_col1:
        st.markdown("""
        <div style="background: rgba(15,23,42,0.65); border: 1px solid rgba(255,255,255,0.08); border-left: 3px solid #38bdf8; border-radius: 10px; padding: 10px 16px; margin: 10px 0 10px 0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 0.82rem; font-weight: 700; color: #f8fafc; letter-spacing: 0.5px;">🎛️ PEAD MOMENTUM & PRECISION FILTER STUDIO</span>
                <span style="font-size: 0.70rem; color: #38bdf8; background: rgba(56,189,248,0.12); padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(56,189,248,0.25); font-weight: 600;">12 Strategy Dimensions</span>
            </div>
            <div style="font-size: 0.72rem; color: #94a3b8;">
                Select strategy presets above or fine-tune quarter, market cap tier, surprises, drift horizon, sector contagion, and volume below
            </div>
        </div>
        """, unsafe_allow_html=True)
    with fsh_col2:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Sync Live Feed", key="pead_in_sync_live_btn", help="Clear cache and re-fetch up to 1,500 latest corporate declared earnings from TradingView"):
            fetch_india_declared_earnings.clear()
            st.rerun()

    # --- ROW 1: PRIMARY TIMING & SURPRISE CONTROLS ---
    f1, f2, f3, f4 = st.columns([1.3, 1.2, 1.2, 1.3])
    with f1:
        cohort_opts = [
            season_info["cohort_label_current"],
            season_info["cohort_label_full"],
            season_info["cohort_label_prior"],
            "🌐 All Available Quarters"
        ]
        cohort_filter = st.selectbox(
            "Reporting Quarter / Season:",
            cohort_opts,
            index=0,
            key="pead_in_cohort_sel",
            help="Filter by corporate quarterly reporting cohort. Current Season isolates freshly reported Q2 results."
        )

    with f2:
        default_surp_idx = 1 if quick_preset in ("🔥 Dual Beats", "🏆 Holy Grail (Dual Beat + RS ≥ 80)") else (2 if quick_preset == "🚀 Rocket EPS (≥+5%)" else (4 if quick_preset == "⚡ Top-Line Sales" else 0))
        surp_filter = st.selectbox(
            "Surprise Focus Filter:",
            [
                "All Declared Results",
                "🔥 Dual Beats Only (Both EPS & Revenue Beat)",
                "🚀 Rocket EPS Beats (≥ +5% EPS Surprise)",
                "🟢 All EPS Beats (> 0% EPS Surprise)",
                "⚡ Revenue Beats (> 0% Sales Surprise)",
                "📊 Pure Growth Only (Uncovered by Analysts)",
                "🔴 EPS Misses (< 0% EPS Surprise)",
                "🔴 Double Misses (Both EPS & Sales Miss)"
            ],
            index=default_surp_idx,
            key="pead_in_dec_surp_sel"
        )

    with f3:
        default_rec_idx = 2 if quick_preset == "📅 Last 7 Days" else 0
        recency_filter = st.selectbox(
            "Drift Recency Horizon:",
            [
                "All Dates in Selected Cohort",
                "⚡ Last 24 Hours / Today",
                "🔥 Last 7 Days (Hot Momentum)",
                "📅 Last 14 Days (Active Base Formation)",
                "📅 Last 30 Days (Past Month Reactions)",
                "📅 Last 60 Days (Primary PEAD Drift Window)",
                "⏳ Full 90 Days (Complete Quarterly Horizon)"
            ],
            index=default_rec_idx,
            key="pead_in_dec_recency",
            help="Lookback timeframe relative to announcement date to track PEAD base formation and post-earnings drift."
        )

    with f4:
        search_query = st.text_input(
            "🔍 Search Stock / Company:",
            placeholder="e.g. HCLTECH, RBLBANK, TCS, DMART...",
            key="pead_in_dec_search"
        ).strip().upper()

    # --- ROW 2: UNIVERSE, MARKET CAP, ADTV(20) LIQUIDITY, SECTOR & CANSLIM LEADERSHIP ---
    all_sectors = ["All Sectors"] + sorted([s for s in df["sector"].dropna().unique() if s and s != "Other"])
    if "Other" in df["sector"].values:
        all_sectors.append("Other")

    af1, af2, af3, af4, af5, af6 = st.columns([1.0, 1.2, 1.2, 1.0, 1.0, 1.1])
    with af1:
        univ_filter = st.selectbox(
            "Universe Scope:",
            ["All Announced Equities", "👑 Nifty 500 Only"],
            index=1 if quick_preset == "👑 Nifty 500" else 0,
            key="pead_in_dec_univ_sel"
        )
    with af2:
        mcap_filter = st.selectbox(
            "Market Cap Tier:",
            [
                "All Market Caps (No Floor)",
                "🛡️ Liquid Equities (≥ ₹200 Cr)",
                "👑 Large Cap (> ₹50,000 Cr)",
                "⚡ Mid Cap (₹15,000 - ₹50,000 Cr)",
                "🌱 Small Cap (₹5,000 - ₹15,000 Cr)",
                "🔬 Micro Cap (₹500 - ₹5,000 Cr)",
                "🪙 Nano Cap (< ₹500 Cr)",
                "🔬 Sub-₹200 Cr (< ₹200 Cr)"
            ],
            index=0,
            key="pead_in_dec_mcap_sel",
            help="Filter by company capitalization tier. All declared stocks from mega-caps down to sub-₹10 Cr nano-caps are included with no artificial floor."
        )
    with af3:
        adtv_filter = st.selectbox(
            "ADTV(20) Liquidity Floor:",
            [
                "💧 Min ₹50 Lakhs (Active Baseline)",
                "⚡ Min ₹1 Crore (Liquid)",
                "🚀 Min ₹2 Crores (High Liquidity)",
                "🔥 Min ₹5 Crores (Institutional)",
                "👑 Min ₹10 Crores (Heavy Liquid)",
                "💎 Min ₹25 Crores (Ultra Liquid)",
                "🌐 All (No Minimum Liquidity Floor)"
            ],
            index=0,
            key="pead_in_adtv_sel",
            help="20-Day Average Daily Traded Value. Default floor is ₹50 Lakhs (INR 0.50 Cr) to eliminate illiquid penny/nano-cap traps."
        )
    with af4:
        sector_filter = st.selectbox(
            "Sector Scope:",
            all_sectors,
            index=0,
            key="pead_in_sector_sel",
            help="Filter by GICS / TradingView Industry Sector to analyze group strength and peer contagion"
        )
    with af5:
        default_rs_idx = 1 if quick_preset == "🏆 Holy Grail (Dual Beat + RS ≥ 80)" else 0
        rs_filter = st.selectbox(
            "CANSLIM RS Rating:",
            [
                "Any RS Rating",
                "🟢 Leaders (RS ≥ 80)",
                "🔵 Strong (RS ≥ 70)",
                "⚪ Above Avg (RS ≥ 50)"
            ],
            index=default_rs_idx,
            key="pead_in_rs_sel",
            help="O'Neil 3-Month Composite Relative Strength Percentile (0-99). Top CANSLIM winners typically have RS ≥ 80."
        )
    with af6:
        setup_filter = st.selectbox(
            "PEAD Drift Setup:",
            [
                "All PEAD Setups",
                "🔥 Monster Dual Beat (Top Drift)",
                "🚀 Prime Dual Beat Leader",
                "🚀 Prime EPS Drift",
                "⚡ Top-Line Sales Runner",
                "🟢 Moderate EPS Drift",
                "⚠️ Post-Earnings Drag"
            ],
            index=0,
            key="pead_in_dec_setup_sel"
        )

    # --- ROW 2.5: CANSLIM FUNDAMENTAL GROWTH & EP ANATOMY ---
    gf1, gf2, gf3, gf4, gf5 = st.columns([1.1, 1.1, 1.2, 1.2, 1.4])
    with gf1:
        min_eps_yoy_opt = st.selectbox(
            "Min YoY EPS Growth %:",
            ["Any", "> 0% (Positive Growth)", "≥ +25% (CANSLIM Min)", "≥ +50% (High Growth)", "≥ +100% (Triple-Digit)"],
            index=2 if quick_preset == "💎 CANSLIM Super-Growth (EPS ≥ 25% & Sales ≥ 20%)" else 0,
            key="pead_in_min_eps_yoy",
            help="CANSLIM Rule 'C': Top winning stocks typically have quarterly YoY EPS growth >= +25%."
        )
    with gf2:
        min_rev_yoy_opt = st.selectbox(
            "Min YoY Sales Growth %:",
            ["Any", "> 0% (Positive Growth)", "≥ +15%", "≥ +25% (CANSLIM Min)", "≥ +50% (High Growth)"],
            index=3 if quick_preset == "💎 CANSLIM Super-Growth (EPS ≥ 25% & Sales ≥ 20%)" else 0,
            key="pead_in_min_rev_yoy",
            help="CANSLIM Top-Line Rule: Demands sales growth >= +20% to validate earnings quality."
        )
    with gf3:
        min_cr_opt = st.selectbox(
            "Min Closing Range %:",
            ["Any", "≥ 50% (Constructive)", "≥ 70% (Accumulation)", "≥ 80% (Power Close)"],
            index=2 if quick_preset == "🏆 Episodic Pivot (EP) Setups (Rx ≥ 3%, CR ≥ 70%)" else 0,
            key="pead_in_min_cr",
            help="Day 0 Intraday Closing Range = (Close - Low)/(High - Low) * 100. Measures buying conviction into the close."
        )
    with gf4:
        ep_setup_opt = st.selectbox(
            "Episodic Pivot Quality:",
            ["All Setups", "🏆 True Episodic Pivot (EP)", "🔥 Power Close Only (CR ≥ 75%)", "⚡ Momentum Breakouts", "🟢 Constructive (CR ≥ 50%)"],
            index=1 if quick_preset == "🏆 Episodic Pivot (EP) Setups (Rx ≥ 3%, CR ≥ 70%)" else 0,
            key="pead_in_ep_setup_sel",
            help="Filter for institutional Episodic Pivot catalysts vs gap-and-fade traps."
        )
    with gf5:
        quality_filter = st.selectbox(
            "Quality Armor & Traps:",
            [
                "All Stocks",
                "💎 Exclude Surprise Mirages (YoY EPS < 0%)",
                "🚫 Exclude Gap & Crap Traps (CR < 35%)",
                "🛡️ Pure Institutional Quality (No Mirages & No Traps)"
            ],
            index=0,
            key="pead_in_quality_armor",
            help="Filter out value traps where stocks beat lowered expectations despite contracting growth, or where heavy volume sold off into the close."
        )

    # --- ROW 3: PRECISION THRESHOLDS, PEAD DRIFT, VOLUME & PRICE REACTION ---
    bf1, bf2, bf3, bf4, bf5 = st.columns([1.1, 1.1, 1.3, 1.1, 1.1])
    with bf1:
        min_eps_opt = st.selectbox(
            "Min EPS Surprise %:",
            ["Any", "> 0% (Positive Beat)", "≥ +5%", "≥ +10%", "≥ +20%", "≥ +50%"],
            index=2 if quick_preset == "🚀 Rocket EPS (≥+5%)" else 0,
            key="pead_in_min_eps_sel"
        )
    with bf2:
        min_rev_opt = st.selectbox(
            "Min Revenue Surprise %:",
            ["Any", "> 0% (Positive Beat)", "≥ +2%", "≥ +5%", "≥ +10%", "≥ +20%"],
            index=1 if quick_preset == "⚡ Top-Line Sales" else 0,
            key="pead_in_min_rev_sel"
        )
    with bf3:
        default_drift_idx = 1 if quick_preset in ("📈 Positive Drift (> 0%)", "🏆 Holy Grail (Dual Beat + RS ≥ 80 + Drift > 0%)") else 0
        drift_filter = st.selectbox(
            "PEAD Drift Filter:",
            [
                "All Drift Trajectories",
                "📈 Positive Drift Only (> 0% Accumulation ↗)",
                "🚀 Strong Drift (≥ +5%)",
                "🔥 Massive Drift (≥ +15%)",
                "🔴 Negative Drift / Fade (< 0%)"
            ],
            index=default_drift_idx,
            key="pead_in_drift_sel",
            help="Cumulative price drift return from announcement date to today. Positive drift confirms institutional accumulation."
        )
    with bf4:
        default_vol_idx = 2 if quick_preset == "🟢 Volume Surge (≥ 2.0x Vol)" else 0
        vol_filter = st.selectbox(
            "Volume Surge Factor:",
            [
                "Any Volume",
                "⚡ Heavy Volume (≥ 1.5x)",
                "🚀 Extreme Volume (≥ 2.0x)",
                "🔥 Institutional Blast (≥ 3.0x)"
            ],
            index=default_vol_idx,
            key="pead_in_vol_sel",
            help="10-day relative volume multiplier on earnings announcement"
        )
    with bf5:
        react_filter = st.selectbox(
            "Day 0 Price Reaction:",
            [
                "Any Price Reaction",
                "🟢 Positive Reaction (> 0%)",
                "🚀 Breakout (> +3% with 1.5x Vol)",
                "🔴 Gap Fade / Drag (< 0%)"
            ],
            index=0,
            key="pead_in_react_sel",
            help="Announcement day price reaction to identify institutional accumulation breakouts vs gap fade traps"
        )

    # --- FILTER EXECUTION ---
    filtered = df.copy()

    # 1. Reporting Season / Cohort Filter
    if cohort_filter == season_info["cohort_label_current"]:
        filtered = filtered[filtered["date"].astype(str).str[:10] >= cutoff_date]
    elif cohort_filter == season_info["cohort_label_prior"]:
        filtered = filtered[filtered["date"].astype(str).str[:10] < cutoff_date]

    # Quick pills enforcement overrides
    if quick_preset == "👑 Nifty 500":
        filtered = filtered[filtered["is_nifty500"]]
    elif quick_preset == "📅 Last 7 Days":
        filtered = filtered[filtered["recency_days"] <= 7]
    elif quick_preset == "💎 CANSLIM Super-Growth (EPS ≥ 25% & Sales ≥ 20%)":
        f_ey = pd.to_numeric(filtered.get("eps_yoy"), errors="coerce").fillna(0)
        f_ry = pd.to_numeric(filtered.get("rev_yoy"), errors="coerce").fillna(0)
        filtered = filtered[(f_ey >= 25.0) & (f_ry >= 20.0)]
    elif quick_preset == "🏆 Episodic Pivot (EP) Setups (Rx ≥ 3%, CR ≥ 70%)":
        f_chg = pd.to_numeric(filtered.get("day0_rx"), errors="coerce").fillna(0)
        f_cr = pd.to_numeric(filtered.get("closing_range"), errors="coerce").fillna(50)
        f_vol = pd.to_numeric(filtered.get("rel_vol"), errors="coerce").fillna(1)
        filtered = filtered[(f_chg >= 3.0) & (f_cr >= 70.0) & (f_vol >= 1.5)]

    # 2. Surprise Focus
    f_eps = pd.to_numeric(filtered["surprise"], errors="coerce")
    f_rev = pd.to_numeric(filtered["rev_surprise"], errors="coerce") if "rev_surprise" in filtered.columns else pd.Series(dtype=float)

    if "Dual Beats" in surp_filter or quick_preset in ("🔥 Dual Beats", "🏆 Holy Grail (Dual Beat + RS ≥ 80 + Drift > 0%)"):
        filtered = filtered[(f_eps > 0) & (f_rev > 0)]
    elif "Rocket EPS" in surp_filter:
        filtered = filtered[f_eps >= 5.0]
    elif "All EPS Beats" in surp_filter:
        filtered = filtered[f_eps > 0]
    elif "Revenue Beats" in surp_filter:
        filtered = filtered[f_rev > 0]
    elif "Double Misses" in surp_filter:
        filtered = filtered[(f_eps < 0) & (f_rev < 0)]
    elif "EPS Misses" in surp_filter:
        filtered = filtered[f_eps < 0]
    elif "Pure Growth" in surp_filter:
        filtered = filtered[f_eps.isna() & f_rev.isna()]

    # 3. Recency Horizon
    if "Last 24 Hours" in recency_filter:
        filtered = filtered[filtered["recency_days"] <= 1]
    elif "Last 7 Days" in recency_filter:
        filtered = filtered[filtered["recency_days"] <= 7]
    elif "Last 14 Days" in recency_filter:
        filtered = filtered[filtered["recency_days"] <= 14]
    elif "Last 30 Days" in recency_filter:
        filtered = filtered[filtered["recency_days"] <= 30]
    elif "Last 60 Days" in recency_filter:
        filtered = filtered[filtered["recency_days"] <= 60]
    elif "Full 90" in recency_filter:
        filtered = filtered[filtered["recency_days"] <= 90]

    # 4. Universe Scope, Market Cap Tier, Sector Scope, RS Rating & Setup
    if "Nifty 500" in univ_filter:
        filtered = filtered[filtered["is_nifty500"]]

    if "≥ ₹200 Cr" in mcap_filter:
        filtered = filtered[filtered["mcap_cr"] >= 200]
    elif "Large Cap" in mcap_filter:
        filtered = filtered[filtered["mcap_cr"] >= 50000]
    elif "Mid Cap" in mcap_filter:
        filtered = filtered[(filtered["mcap_cr"] >= 15000) & (filtered["mcap_cr"] < 50000)]
    elif "Small Cap" in mcap_filter:
        filtered = filtered[(filtered["mcap_cr"] >= 5000) & (filtered["mcap_cr"] < 15000)]
    elif "Micro Cap" in mcap_filter:
        filtered = filtered[(filtered["mcap_cr"] >= 500) & (filtered["mcap_cr"] < 5000)]
    elif "Nano Cap" in mcap_filter:
        filtered = filtered[(filtered["mcap_cr"] > 0) & (filtered["mcap_cr"] < 500)]
    elif "< ₹200 Cr" in mcap_filter:
        filtered = filtered[(filtered["mcap_cr"] > 0) & (filtered["mcap_cr"] < 200)]

    # 4.1 ADTV(20) Liquidity Floor Filter
    if "Min ₹50 Lakhs" in adtv_filter:
        filtered = filtered[filtered["adtv_cr"] >= 0.50]
    elif "Min ₹1 Crore" in adtv_filter:
        filtered = filtered[filtered["adtv_cr"] >= 1.0]
    elif "Min ₹2 Crores" in adtv_filter:
        filtered = filtered[filtered["adtv_cr"] >= 2.0]
    elif "Min ₹5 Crores" in adtv_filter:
        filtered = filtered[filtered["adtv_cr"] >= 5.0]
    elif "Min ₹10 Crores" in adtv_filter:
        filtered = filtered[filtered["adtv_cr"] >= 10.0]
    elif "Min ₹25 Crores" in adtv_filter:
        filtered = filtered[filtered["adtv_cr"] >= 25.0]

    if sector_filter != "All Sectors":
        filtered = filtered[filtered["sector"] == sector_filter]

    if "RS ≥ 80" in rs_filter or quick_preset == "🏆 Holy Grail (Dual Beat + RS ≥ 80 + Drift > 0%)":
        filtered = filtered[filtered["rs_raw_pct"] >= 80]
    elif "RS ≥ 70" in rs_filter:
        filtered = filtered[filtered["rs_raw_pct"] >= 70]
    elif "RS ≥ 50" in rs_filter:
        filtered = filtered[filtered["rs_raw_pct"] >= 50]

    if setup_filter != "All PEAD Setups":
        filtered = filtered[filtered["PEAD Drift Setup"] == setup_filter]

    # 4.5. CANSLIM YoY Growth & Episodic Pivot Filters
    f_ey = pd.to_numeric(filtered.get("eps_yoy"), errors="coerce")
    if "> 0%" in min_eps_yoy_opt:
        filtered = filtered[f_ey > 0]
    elif "≥ +25%" in min_eps_yoy_opt:
        filtered = filtered[f_ey >= 25.0]
    elif "≥ +50%" in min_eps_yoy_opt:
        filtered = filtered[f_ey >= 50.0]
    elif "≥ +100%" in min_eps_yoy_opt:
        filtered = filtered[f_ey >= 100.0]

    f_ry = pd.to_numeric(filtered.get("rev_yoy"), errors="coerce")
    if "> 0%" in min_rev_yoy_opt:
        filtered = filtered[f_ry > 0]
    elif "≥ +15%" in min_rev_yoy_opt:
        filtered = filtered[f_ry >= 15.0]
    elif "≥ +25%" in min_rev_yoy_opt:
        filtered = filtered[f_ry >= 25.0]
    elif "≥ +50%" in min_rev_yoy_opt:
        filtered = filtered[f_ry >= 50.0]

    f_cr = pd.to_numeric(filtered.get("closing_range"), errors="coerce")
    if "≥ 50%" in min_cr_opt:
        filtered = filtered[f_cr >= 50.0]
    elif "≥ 70%" in min_cr_opt:
        filtered = filtered[f_cr >= 70.0]
    elif "≥ 80%" in min_cr_opt:
        filtered = filtered[f_cr >= 80.0]

    if "True Episodic Pivot" in ep_setup_opt:
        filtered = filtered[filtered["EP Setup"] == "🏆 True Episodic Pivot (EP)"]
    elif "Power Close" in ep_setup_opt:
        filtered = filtered[f_cr >= 75.0]
    elif "Momentum Breakouts" in ep_setup_opt:
        filtered = filtered[filtered["EP Setup"].isin(["🏆 True Episodic Pivot (EP)", "⚡ Momentum Gap"])]
    elif "Constructive" in ep_setup_opt:
        filtered = filtered[f_cr >= 50.0]

    if "Exclude Surprise Mirages" in quality_filter or "Pure Institutional Quality" in quality_filter:
        f_ey_m = pd.to_numeric(filtered.get("eps_yoy"), errors="coerce")
        f_surp_m = pd.to_numeric(filtered.get("surprise"), errors="coerce")
        is_mirage = (f_surp_m > 0) & (f_ey_m < 0)
        filtered = filtered[~is_mirage]

    if "Exclude Gap & Crap" in quality_filter or "Pure Institutional Quality" in quality_filter:
        f_cr_t = pd.to_numeric(filtered.get("closing_range"), errors="coerce")
        is_trap = (f_cr_t < 35.0)
        filtered = filtered[~is_trap]

    # 5. Min Surprises Thresholds
    f_eps = pd.to_numeric(filtered["surprise"], errors="coerce")
    if "> 0%" in min_eps_opt:
        filtered = filtered[f_eps > 0]
    elif "≥ +5%" in min_eps_opt:
        filtered = filtered[f_eps >= 5.0]
    elif "≥ +10%" in min_eps_opt:
        filtered = filtered[f_eps >= 10.0]
    elif "≥ +20%" in min_eps_opt:
        filtered = filtered[f_eps >= 20.0]
    elif "≥ +50%" in min_eps_opt:
        filtered = filtered[f_eps >= 50.0]

    f_rev = pd.to_numeric(filtered["rev_surprise"], errors="coerce") if "rev_surprise" in filtered.columns else pd.Series(dtype=float)
    if "> 0%" in min_rev_opt:
        filtered = filtered[f_rev > 0]
    elif "≥ +2%" in min_rev_opt:
        filtered = filtered[f_rev >= 2.0]
    elif "≥ +5%" in min_rev_opt:
        filtered = filtered[f_rev >= 5.0]
    elif "≥ +10%" in min_rev_opt:
        filtered = filtered[f_rev >= 10.0]
    elif "≥ +20%" in min_rev_opt:
        filtered = filtered[f_rev >= 20.0]

    # 6. PEAD Drift Trajectory Filter
    f_drift = pd.to_numeric(filtered["pead_drift"], errors="coerce").fillna(0.0)
    if quick_preset in ("📈 Positive Drift (> 0%)", "🏆 Holy Grail (Dual Beat + RS ≥ 80 + Drift > 0%)") or "Positive Drift Only" in drift_filter:
        filtered = filtered[f_drift > 0]
    elif "Strong Drift" in drift_filter:
        filtered = filtered[f_drift >= 5.0]
    elif "Massive Drift" in drift_filter:
        filtered = filtered[f_drift >= 15.0]
    elif "Negative Drift" in drift_filter:
        filtered = filtered[f_drift < 0]

    # 7. Volume Factor & Reaction Filter
    f_vol = pd.to_numeric(filtered["rel_vol"], errors="coerce").fillna(1.0)
    f_chg = pd.to_numeric(filtered["day0_rx"], errors="coerce").fillna(0.0)

    if quick_preset == "🟢 Volume Surge (≥ 2.0x Vol)" or "≥ 2.0x" in vol_filter:
        filtered = filtered[f_vol >= 2.0]
    elif "≥ 1.5x" in vol_filter:
        filtered = filtered[f_vol >= 1.5]
    elif "≥ 3.0x" in vol_filter:
        filtered = filtered[f_vol >= 3.0]

    if "Positive Reaction" in react_filter:
        filtered = filtered[f_chg > 0]
    elif "Breakout" in react_filter:
        filtered = filtered[(f_chg >= 3.0) & (f_vol >= 1.5)]
    elif "Gap Fade" in react_filter:
        filtered = filtered[f_chg < 0]

    # 7. Search Query
    if search_query:
        filtered = filtered[
            filtered["symbol"].str.contains(search_query, na=False) |
            filtered["company"].str.upper().str.contains(search_query, na=False)
        ]

    # --- SECTOR CONTAGION AGGREGATION & RADAR ---
    contagion_df = _aggregate_thematic_contagion(filtered if len(filtered) >= 3 else df)
    _render_thematic_contagion_banner(contagion_df, market="IN")
    _render_thematic_contagion_matrix(contagion_df, upcoming_df=df_upcoming, market="IN")

    # --- DYNAMIC HUD METRICS (Synchronized with filtered selection) ---
    f_total = len(filtered)
    f_surp_eps = pd.to_numeric(filtered["surprise"], errors="coerce")
    f_surp_rev = pd.to_numeric(filtered["rev_surprise"], errors="coerce") if "rev_surprise" in filtered.columns else pd.Series(dtype=float)

    dual_beats = int(((f_surp_eps > 0) & (f_surp_rev > 0)).sum()) if not f_surp_rev.empty else 0
    eps_beats = int((f_surp_eps > 0).sum())
    rev_beats = int((f_surp_rev > 0).sum()) if not f_surp_rev.empty else 0

    dual_rate = round(dual_beats / max(1, f_total) * 100, 1)
    eps_rate = round(eps_beats / max(1, f_total) * 100, 1)
    rev_rate = round(rev_beats / max(1, f_total) * 100, 1)

    if cohort_filter == season_info["cohort_label_current"]:
        season_hud_label = f"Current Season: {season_info['current_q']}"
    elif cohort_filter == season_info["cohort_label_prior"]:
        season_hud_label = f"Prior Quarter: {season_info['prior_q']}"
    elif f_total == len(df):
        season_hud_label = f"Full Universe ({f_total} Stocks)"
    else:
        season_hud_label = f"Filtered Selection ({f_total} of {len(df)})"

    _render_earnings_hud_cards(f_total, dual_beats, dual_rate, eps_beats, eps_rate, rev_beats, rev_rate, season_label=season_hud_label)

    # --- VIEW MODE, DECK MODE & SORTING CONTROL BAR ---
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    v_col1, v_col2, v_col3 = st.columns([1.6, 1.4, 1.2])
    with v_col1:
        view_options = [
            "⚡ Side-by-Side (Map Left + Table Right)",
            "📊 Full Quadrant Map",
            "📋 Full Table Grid"
        ]
        if hasattr(st, "segmented_control"):
            view_mode = st.segmented_control(
                "Layout View:",
                options=view_options,
                default=view_options[0],
                key="pead_in_view_mode"
            )
        else:
            view_mode = st.radio(
                "Layout View:",
                view_options,
                index=0,
                horizontal=True,
                key="pead_in_view_mode"
            )
    with v_col2:
        deck_mode_opts = ["⚡ Essential Execution Deck", "📑 Deep Analyst Deck"]
        if hasattr(st, "segmented_control"):
            deck_mode = st.segmented_control(
                "Table Deck View:",
                options=deck_mode_opts,
                default=deck_mode_opts[0],
                key="pead_in_deck_mode"
            )
        else:
            deck_mode = st.radio(
                "Table Deck View:",
                deck_mode_opts,
                index=0,
                horizontal=True,
                key="pead_in_deck_mode"
            )
    with v_col3:
        sort_by = st.selectbox(
            "Sort Order:",
            [
                "📈 PEAD Drift % (Highest Drift First)",
                "⚡ YoY Sales Growth % (Highest First)",
                "🔥 YoY EPS Growth % (Highest First)",
                "💧 ADTV (20) (Most Liquid First)",
                "📅 Release Date (Latest First)",
                "🔥 EPS Surprise % (Highest First)",
                "⚡ Revenue Surprise % (Highest First)",
                "💪 RS Rating (Leaders First)",
                "🕯️ Day 0 Reaction (Top First)",
                "🔤 Symbol (A to Z)"
            ],
            index=0,
            key="pead_in_sort_sel"
        )

    # Apply Sorting
    if "PEAD Drift" in sort_by:
        filtered = filtered.sort_values(by=["pead_drift", "surprise"], ascending=[False, False])
    elif "YoY Sales" in sort_by:
        filtered = filtered.sort_values(by=["rev_yoy", "pead_drift"], ascending=[False, False], na_position='last')
    elif "YoY EPS" in sort_by:
        filtered = filtered.sort_values(by=["eps_yoy", "pead_drift"], ascending=[False, False], na_position='last')
    elif "ADTV" in sort_by:
        filtered = filtered.sort_values(by=["adtv_cr", "pead_drift"], ascending=[False, False])
    elif "Release Date" in sort_by:
        filtered = filtered.sort_values(by=["date", "surprise"], ascending=[False, False])
    elif "EPS Surprise" in sort_by:
        filtered = filtered.sort_values(by=["surprise", "rev_surprise"], ascending=[False, False])
    elif "Revenue Surprise" in sort_by:
        filtered = filtered.sort_values(by=["rev_surprise", "surprise"], ascending=[False, False])
    elif "RS Rating" in sort_by:
        filtered = filtered.sort_values(by=["rs_raw_pct", "pead_drift"], ascending=[False, False])
    elif "Day 0 Reaction" in sort_by or "Price Reaction" in sort_by:
        filtered = filtered.sort_values(by=["day0_rx", "rel_vol"], ascending=[False, False])
    elif "Symbol" in sort_by:
        filtered = filtered.sort_values(by=["symbol"], ascending=[True])

    # Formatted display columns
    filtered = filtered.copy()
    filtered["Est. EPS"] = filtered["est"].apply(lambda v: f"₹{float(v):,.2f}" if pd.notna(v) and v is not None else "-")
    filtered["Reported EPS"] = filtered["act"].apply(lambda v: f"₹{float(v):,.2f}" if pd.notna(v) and v is not None else "-")
    filtered["EPS Surprise %"] = filtered["surprise"].apply(lambda v: f"+{float(v):,.2f}%" if pd.notna(v) and float(v) > 0 else (f"{float(v):,.2f}%" if pd.notna(v) else "-"))

    filtered["Est. Revenue"] = filtered["rev_est"].apply(lambda v: _format_revenue_val(v, "₹"))
    filtered["Reported Revenue"] = filtered["rev_act"].apply(lambda v: _format_revenue_val(v, "₹"))
    filtered["Revenue Surprise %"] = filtered["rev_surprise"].apply(
        lambda v: f"+{float(v):,.2f}%" if pd.notna(v) and float(v) > 0 else (f"{float(v):,.2f}%" if pd.notna(v) else "-")
    )
    filtered["Day 0 High"] = filtered["day0_high"].apply(lambda v: f"₹{float(v):,.2f}" if pd.notna(v) and v is not None else "-")
    filtered["Day 0 Low"] = filtered["day0_low"].apply(lambda v: f"₹{float(v):,.2f}" if pd.notna(v) and v is not None else "-")
    filtered["Risk to Stop %"] = filtered["risk_to_stop"].apply(lambda v: f"{float(v):.1f}%" if pd.notna(v) and v is not None else "-")

    is_essential_deck = "Essential" in deck_mode
    if is_essential_deck:
        display_df = filtered[[
            "symbol", "PEAD Drift", "Reaction Candle", "Market Cap", "ADTV (20)", "YoY EPS %", "YoY Sales %", "CANSLIM Growth",
            "RS Rating", "Dual Surprise Verdict", "EPS Surprise %", "Revenue Surprise %", "sector", "TradingView", "date"
        ]].rename(
            columns={
                "symbol": "Ticker",
                "PEAD Drift": "PEAD Drift",
                "Reaction Candle": "Day 0 Rx",
                "Market Cap": "Market Cap",
                "ADTV (20)": "ADTV (20)",
                "YoY EPS %": "YoY EPS %",
                "YoY Sales %": "YoY Sales %",
                "CANSLIM Growth": "Growth Status",
                "RS Rating": "RS",
                "Dual Surprise Verdict": "Dual Verdict",
                "EPS Surprise %": "EPS Surp %",
                "Revenue Surprise %": "Sales Surp %",
                "sector": "Sector",
                "TradingView": "Chart ↗",
                "date": "Release Date"
            }
        )
    else:
        display_df = filtered[[
            "symbol", "PEAD Drift", "Reaction Candle", "Market Cap", "ADTV (20)", "Closing Range %", "EP Setup",
            "YoY EPS %", "YoY Sales %", "CANSLIM Growth", "RS Rating", "Dual Surprise Verdict", "EPS Surprise %",
            "Reported EPS", "Est. EPS", "Revenue Surprise %", "Reported Revenue", "Est. Revenue",
            "Day 0 High", "Day 0 Low", "Risk to Stop %", "PEAD Drift Setup",
            "sector", "industry", "company", "Universe", "Season", "TradingView", "date"
        ]].rename(
            columns={
                "symbol": "Ticker",
                "PEAD Drift": "PEAD Drift",
                "Reaction Candle": "Day 0 Rx",
                "Market Cap": "Market Cap",
                "ADTV (20)": "ADTV (20)",
                "Closing Range %": "Closing Range",
                "EP Setup": "EP Quality",
                "YoY EPS %": "YoY EPS %",
                "YoY Sales %": "YoY Sales %",
                "CANSLIM Growth": "Growth Status",
                "RS Rating": "RS",
                "Dual Surprise Verdict": "Dual Verdict",
                "EPS Surprise %": "EPS Surp %",
                "Reported EPS": "Reported EPS",
                "Est. EPS": "Est. EPS",
                "Revenue Surprise %": "Sales Surp %",
                "Reported Revenue": "Reported Rev",
                "Est. Revenue": "Est. Rev",
                "Day 0 High": "Day 0 High (Pivot)",
                "Day 0 Low": "Day 0 Low (Stop)",
                "Risk to Stop %": "Risk to Stop",
                "PEAD Drift Setup": "PEAD Setup",
                "sector": "Sector",
                "industry": "Industry",
                "company": "Company Name",
                "Universe": "Index",
                "Season": "Season",
                "TradingView": "Chart ↗",
                "date": "Release Date"
            }
        )

    in_col_config = {
        "Ticker": st.column_config.TextColumn("Ticker", width="small", help="NSE / BSE Ticker Symbol"),
        "PEAD Drift": st.column_config.TextColumn("PEAD Drift", width="medium", help="Cumulative total return from announcement date to today"),
        "Day 0 Rx": st.column_config.TextColumn("Day 0 Rx", width="medium", help="Announcement day price reaction %, volume surge multiplier, and closing range %"),
        "Market Cap": st.column_config.TextColumn("Market Cap", width="small", help="Institutional Market Capitalization (👑 Large > ₹50k Cr, ⚡ Mid ₹15-50k Cr, 🌱 Small ₹5-15k Cr, 🔬 Micro < ₹5k Cr)"),
        "ADTV (20)": st.column_config.TextColumn("ADTV (20)", width="small", help="20-Day Average Daily Traded Value (👑 > ₹10 Cr, 🔥 ₹5-10 Cr, ⚡ ₹1-5 Cr, 💧 ₹50L-1 Cr)"),
        "YoY EPS %": st.column_config.TextColumn("YoY EPS %", width="small", help="Quarterly YoY EPS growth % (CANSLIM 'C' requires ≥ +25%)"),
        "YoY Sales %": st.column_config.TextColumn("YoY Sales %", width="small", help="Quarterly YoY Revenue/Sales growth % (CANSLIM requires ≥ +20%)"),
        "Growth Status": st.column_config.TextColumn("Growth Status", width="medium", help="CANSLIM Pillar C classification: Super-Growth vs Surprise Mirage"),
        "Closing Range": st.column_config.TextColumn("Closing Range", width="small", help="Day 0 Intraday Closing Range: (Close - Low)/(High - Low) * 100"),
        "EP Quality": st.column_config.TextColumn("EP Quality", width="medium", help="Episodic Pivot (EP) classification: True EP vs Gap-and-Crap trap"),
        "RS": st.column_config.TextColumn("O'Neil RS", width="small", help="CANSLIM 3M Composite Relative Strength Percentile (0-99)"),
        "Dual Verdict": st.column_config.TextColumn("Dual Verdict", width="medium", help="Consensus beat verdict on EPS and Revenue"),
        "EPS Surp %": st.column_config.TextColumn("EPS Surp %", width="small", help="Percentage by which reported EPS exceeded consensus"),
        "Sales Surp %": st.column_config.TextColumn("Sales Surp %", width="small", help="Percentage by which reported Revenue exceeded consensus"),
        "Day 0 High (Pivot)": st.column_config.TextColumn("Day 0 High (Pivot)", width="small", help="Episodic Pivot Buy Trigger Price"),
        "Day 0 Low (Stop)": st.column_config.TextColumn("Day 0 Low (Stop)", width="small", help="Day 0 Low Structural Invalidation Stop Level"),
        "Risk to Stop": st.column_config.TextColumn("Risk to Stop", width="small", help="Percentage distance from current price to Day 0 Low stop loss"),
        "Sector": st.column_config.TextColumn("Sector", width="medium", help="Institutional Industry Sector"),
        "Industry": st.column_config.TextColumn("Industry", width="medium", help="Detailed Industry Sub-group"),
        "Chart ↗": st.column_config.LinkColumn(
            "Chart ↗",
            display_text="Open ↗",
            help="Inspect post-earnings reaction candle & volume on TradingView"
        ),
        "Release Date": st.column_config.TextColumn("Release Date", width="small"),
        "Reported EPS": st.column_config.TextColumn("Reported EPS", width="small"),
        "Est. EPS": st.column_config.TextColumn("Est. EPS", width="small"),
        "Reported Rev": st.column_config.TextColumn("Reported Rev", width="small"),
        "Est. Rev": st.column_config.TextColumn("Est. Rev", width="small"),
        "PEAD Setup": st.column_config.TextColumn("PEAD Setup", width="medium"),
        "Company Name": st.column_config.TextColumn("Company Name", width="medium"),
        "Index": st.column_config.TextColumn("Index", width="small"),
        "Season": st.column_config.TextColumn("Season", width="small")
    }

    chart_key = "pead_in_quadrant_chart"

    # --- RENDER SELECTED VIEW ---
    if "Side-by-Side" in view_mode:
        map_col, tbl_col = st.columns([1.0, 1.15])
        with map_col:
            sel_symbol = _render_surprise_quadrant_chart(filtered, market_currency="₹", height=540, chart_key=chart_key)

        with tbl_col:
            active_display_df = display_df
            if sel_symbol and sel_symbol in display_df["Ticker"].values:
                iso_c1, iso_c2 = st.columns([3, 1])
                with iso_c1:
                    st.markdown(f"""
                    <div style="padding: 6px 12px; background: rgba(56,189,248,0.18); border: 1px solid #38bdf8; border-radius: 8px; font-size: 0.78rem; color: #f8fafc; font-weight: 600; margin-bottom: 6px;">
                        🎯 <b>Chart Selection Isolated:</b> <span style="color: #38bdf8; font-weight: 700;">{sel_symbol}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with iso_c2:
                    if st.button("✕ Reset Map", key=f"{chart_key}_clear_btn"):
                        chart_ver_key = f"{chart_key}_ver"
                        st.session_state[chart_ver_key] = st.session_state.get(chart_ver_key, 0) + 1
                        st.session_state.pop(chart_key, None)
                        st.rerun()
                active_display_df = display_df[display_df["Ticker"] == sel_symbol]

            deck_label = "⚡ Essential" if is_essential_deck else "📑 Deep Analyst"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: rgba(15,23,42,0.7); border: 1px solid rgba(255,255,255,0.08); border-left: 3px solid #38bdf8; border-radius: 8px; margin-bottom: 8px;">
                <div style="font-size: 0.88rem; font-weight: 700; color: #f8fafc;">📋 {deck_label} Results Grid</div>
                <div style="font-size: 0.72rem; color: #38bdf8; font-weight: 600; background: rgba(56,189,248,0.15); padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(56,189,248,0.3);">Showing {len(active_display_df)} of {len(df)} Stocks</div>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(
                active_display_df,
                column_config=in_col_config,
                use_container_width=True,
                hide_index=True,
                height=540
            )
    elif "Quadrant Map" in view_mode:
        sel_symbol = _render_surprise_quadrant_chart(filtered, market_currency="₹", height=600, chart_key=chart_key)
        if sel_symbol and sel_symbol in display_df["Ticker"].values:
            st.markdown(f"##### 🎯 Selected Stock Profile: `{sel_symbol}`")
            st.dataframe(display_df[display_df["Ticker"] == sel_symbol], column_config=in_col_config, use_container_width=True, hide_index=True)
    else:
        deck_label = "⚡ Essential" if is_essential_deck else "📑 Deep Analyst"
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: rgba(15,23,42,0.7); border: 1px solid rgba(255,255,255,0.08); border-left: 3px solid #38bdf8; border-radius: 8px; margin-bottom: 8px;">
            <div style="font-size: 0.88rem; font-weight: 700; color: #f8fafc;">📋 {deck_label} Results Grid</div>
            <div style="font-size: 0.72rem; color: #38bdf8; font-weight: 600; background: rgba(56,189,248,0.15); padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(56,189,248,0.3);">Showing {len(display_df)} of {len(df)} Stocks</div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            display_df,
            column_config=in_col_config,
            use_container_width=True,
            hide_index=True,
            height=540
        )

    # --- TRADINGVIEW WATCHLIST EXPORTER ---
    with st.expander("📋 Export Filtered Tickers to TradingView Watchlist"):
        st.caption("Copy formatted watchlist tickers to import directly into TradingView's Watchlist dialog.")
        f_eps_all = pd.to_numeric(filtered["surprise"], errors="coerce")
        f_rev_all = pd.to_numeric(filtered["rev_surprise"], errors="coerce") if "rev_surprise" in filtered.columns else pd.Series(dtype=float)

        dual_symbols = [f"NSE:{s}" for s in filtered[(f_eps_all > 0) & (f_rev_all > 0)]["symbol"].dropna().unique() if s]
        holy_grail_symbols = [f"NSE:{s}" for s in filtered[(f_eps_all > 0) & (f_rev_all > 0) & (filtered["rs_raw_pct"] >= 80)]["symbol"].dropna().unique() if s]
        volume_symbols = [f"NSE:{s}" for s in filtered[filtered["rel_vol"] >= 2.0]["symbol"].dropna().unique() if s]
        all_symbols = [f"NSE:{s}" for s in filtered["symbol"].dropna().unique() if s]

        e_t1, e_t2, e_t3, e_t4 = st.tabs([
            f"🔥 Dual Beats ({len(dual_symbols)})",
            f"🏆 Holy Grail ({len(holy_grail_symbols)})",
            f"🟢 Volume Surge ({len(volume_symbols)})",
            f"📋 All Filtered ({len(all_symbols)})"
        ])
        with e_t1:
            st.code(",".join(dual_symbols), language="text")
        with e_t2:
            st.code(",".join(holy_grail_symbols), language="text")
        with e_t3:
            st.code(",".join(volume_symbols), language="text")
        with e_t4:
            st.code(",".join(all_symbols), language="text")


# -----------------------------------------------------------------------------
# INDIA: UPCOMING BOARD MEETINGS VIEW
# -----------------------------------------------------------------------------
def _render_india_upcoming_earnings(df_raw, nifty500_set):
    if df_raw.empty:
        st.warning("⚠️ Live NSE corporate earnings feed currently unavailable. Please use the direct regulatory portal links above.")
        return

    df = df_raw.copy()
    df["parsed_date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    today = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
    df["days_left"] = (df["parsed_date"] - today).dt.days

    df["is_nifty500"] = df["symbol"].apply(lambda s: s in nifty500_set)
    df["Universe"] = df["is_nifty500"].apply(lambda x: "👑 Nifty 500" if x else "Broad Market")
    df["Timeline"] = df["days_left"].apply(_format_days_badge)
    df["TradingView"] = df["symbol"].apply(lambda s: f"https://in.tradingview.com/chart/?symbol=NSE:{s}")

    # Summary HUD Metrics
    total_announced = len(df)
    n500_announced = int(df["is_nifty500"].sum())
    next_7d = int(((df["days_left"] >= 0) & (df["days_left"] <= 7)).sum())
    next_14d = int(((df["days_left"] >= 0) & (df["days_left"] <= 14)).sum())

    # Render Glassmorphic HUD Cards
    _render_upcoming_hud_cards(total_announced, n500_announced, next_7d, next_14d, is_india=True)

    # Filters
    st.markdown("---")
    f1, f2, f3 = st.columns([1.2, 1.2, 1.6])
    with f1:
        time_horizon = st.selectbox(
            "Horizon Filter:",
            ["All Upcoming Releases", "⚡ Next 7 Days (Binary Gap Risk)", "Next 14 Days", "Next 30 Days"],
            index=0,
            key="pead_in_horizon"
        )
    with f2:
        univ_filter = st.selectbox(
            "Universe Scope:",
            ["All Announced Companies", "👑 Nifty 500 Only"],
            index=0,
            key="pead_in_univ"
        )
    with f3:
        search_query = st.text_input(
            "🔍 Search Stock / Company:",
            placeholder="e.g. BHEL, CANBK, CEAT...",
            key="pead_in_search"
        ).strip().upper()

    # Filter Logic
    filtered = df.copy()
    if "Next 7 Days" in time_horizon:
        filtered = filtered[(filtered["days_left"] >= 0) & (filtered["days_left"] <= 7)]
    elif "Next 14 Days" in time_horizon:
        filtered = filtered[(filtered["days_left"] >= 0) & (filtered["days_left"] <= 14)]
    elif "Next 30 Days" in time_horizon:
        filtered = filtered[(filtered["days_left"] >= 0) & (filtered["days_left"] <= 30)]

    if "Nifty 500" in univ_filter:
        filtered = filtered[filtered["is_nifty500"]]

    if search_query:
        filtered = filtered[
            filtered["symbol"].str.contains(search_query, na=False) |
            filtered["company"].str.upper().str.contains(search_query, na=False)
        ]

    # Sort by upcoming date
    filtered = filtered.sort_values(by=["days_left", "symbol"], ascending=[True, True])

    # Display Table
    st.markdown(f"##### Showing **{len(filtered)}** Upcoming Corporate Earnings Releases")
    display_df = filtered[["Timeline", "date", "symbol", "company", "Universe", "bm_desc", "TradingView"]].rename(
        columns={
            "date": "Announcement Date",
            "symbol": "NSE Ticker",
            "company": "Company Name",
            "bm_desc": "Board Meeting Purpose",
            "TradingView": "Chart ↗"
        }
    )

    st.dataframe(
        display_df,
        column_config={
            "Chart ↗": st.column_config.LinkColumn(
                "Chart ↗",
                display_text="Open ↗",
                help="Open institutional chart on TradingView"
            ),
            "Announcement Date": st.column_config.TextColumn(width="small"),
            "NSE Ticker": st.column_config.TextColumn(width="small"),
            "Timeline": st.column_config.TextColumn(width="medium"),
            "Universe": st.column_config.TextColumn(width="small"),
            "Board Meeting Purpose": st.column_config.TextColumn(width="large")
        },
        use_container_width=True,
        hide_index=True,
        height=480
    )

    # TradingView Watchlist Generator
    with st.expander("📋 Export Filtered Tickers to TradingView Watchlist"):
        st.caption("Copy this formatted list and paste it directly into TradingView's Watchlist Import box.")
        symbols_tv = [f"NSE:{s}" for s in filtered["symbol"].dropna().unique() if s]
        st.code(",".join(symbols_tv), language="text")


def _render_india_earnings_calendar():
    company_map = _get_company_names_map()
    try:
        from nifty500_universe import fetch_nifty500_tickers
        nifty500_set = set(fetch_nifty500_tickers(show_progress=False))
    except Exception:
        from nifty500_universe import FALLBACK_TICKERS
        nifty500_set = set(FALLBACK_TICKERS)

    # DUAL-FOCUS MODE TOGGLE
    if hasattr(st, "segmented_control"):
        focus_mode = st.segmented_control(
            "🇮🇳 Earnings Intelligence View:",
            [
                "🎯 Declared Results & Earnings Surprises (PEAD Momentum)",
                "⏳ Upcoming Earnings Board Meetings (Binary Risk Radar)"
            ],
            default="🎯 Declared Results & Earnings Surprises (PEAD Momentum)",
            key="pead_in_focus_mode"
        )
    else:
        focus_mode = st.radio(
            "🇮🇳 Earnings Intelligence View:",
            [
                "🎯 Declared Results & Earnings Surprises (PEAD Momentum)",
                "⏳ Upcoming Earnings Board Meetings (Binary Risk Radar)"
            ],
            horizontal=True,
            key="pead_in_focus_mode"
        )

    if not focus_mode or "Declared Results" in focus_mode:
        df_dec = fetch_india_declared_earnings()
        df_up = fetch_nse_corporate_earnings_calendar()
        _render_india_declared_earnings(df_dec, nifty500_set, company_map, df_upcoming=df_up)
    else:
        df_up = fetch_nse_corporate_earnings_calendar()
        _render_india_upcoming_earnings(df_up, nifty500_set)


# -----------------------------------------------------------------------------
# US: DECLARED RESULTS & DUAL SURPRISES VIEW
# -----------------------------------------------------------------------------
def _render_us_declared_earnings(df_raw, df_upcoming=None):
    if df_raw.empty:
        st.info("ℹ️ No declared results data available in cache yet.")
        return

    df = df_raw.copy()
    df["TradingView"] = df.apply(lambda r: f"https://www.tradingview.com/chart/?symbol={r.get('ticker', r['symbol'])}", axis=1)

    df["Dual Surprise Verdict"] = df.apply(lambda r: _format_dual_verdict(r.get("surprise"), r.get("rev_surprise")), axis=1)
    df["PEAD Drift Setup"] = df.apply(lambda r: _format_dual_pead_setup(r.get("surprise"), r.get("rev_surprise")), axis=1)
    df["recency_days"] = _parse_recency_days(df["date"])

    # Vectorized Batch PEAD Drift & Day 0 Announcement Reaction Enrichment
    sym_date_tuples = tuple(zip(df["symbol"].astype(str), df["date"].astype(str)))
    drift_cache = fetch_batch_pead_drift(sym_date_tuples, market="US")

    def _resolve_day0_rx_us(row):
        s = str(row["symbol"]).strip()
        rec = row.get("recency_days", 999)
        d_info = drift_cache.get(s, {})
        hist_rx = d_info.get("day0_rx")
        if rec <= 1 and pd.notna(row.get("change")):
            return float(row.get("change"))
        elif hist_rx is not None:
            return hist_rx
        elif pd.notna(row.get("change")):
            return float(row.get("change"))
        return 0.0

    def _resolve_drift_pct_us(row):
        s = str(row["symbol"]).strip()
        rec = row.get("recency_days", 999)
        d_info = drift_cache.get(s, {})
        hist_drift = d_info.get("pead_drift")
        if rec == 0:
            return 0.0
        elif hist_drift is not None:
            return hist_drift
        elif rec <= 1:
            return 0.0
        else:
            perf = row.get("perf_1m")
            if pd.notna(perf) and perf is not None:
                return float(perf)
            return 0.0

    df["day0_rx"] = df.apply(_resolve_day0_rx_us, axis=1)
    df["pead_drift"] = df.apply(_resolve_drift_pct_us, axis=1)

    # Market Capitalization Tier Classification
    df["mcap_m"] = pd.to_numeric(df.get("market_cap"), errors="coerce").fillna(0.0)
    def _format_us_mcap_badge(mc):
        if mc >= 10000:
            return f"👑 ${mc/1000:,.1f}B"
        elif mc >= 2000:
            return f"⚡ ${mc/1000:,.1f}B"
        elif mc >= 300:
            return f"🌱 ${mc:,.0f}M"
        elif mc > 0:
            return f"🔬 ${mc:,.0f}M"
        return "-"
    df["Market Cap"] = df["mcap_m"].apply(_format_us_mcap_badge)

    # 20-Day Average Daily Traded Value (ADTV) Resolution & Badge
    def _resolve_adtv_us(row):
        s = str(row["symbol"]).strip()
        d_info = drift_cache.get(s, {})
        drift_adtv = d_info.get("adtv")
        if drift_adtv is not None and drift_adtv > 0:
            return drift_adtv
        tv_adtv = row.get("adtv_m")
        if pd.notna(tv_adtv) and tv_adtv is not None and tv_adtv > 0:
            return float(tv_adtv)
        c = row.get("close")
        v = row.get("volume")
        if pd.notna(c) and pd.notna(v) and c and v:
            return round((float(c) * float(v)) / 1e6, 2)
        return 0.0

    df["adtv_m"] = df.apply(_resolve_adtv_us, axis=1)

    def _format_adtv_us_badge(val):
        if val is None or pd.isna(val) or val <= 0:
            return "-"
        v = float(val)
        if v >= 1000.0:
            return f"👑 ${v/1000:,.1f}B"
        elif v >= 50.0:
            return f"🔥 ${v:,.1f}M"
        elif v >= 5.0:
            return f"⚡ ${v:,.1f}M"
        elif v >= 0.50:
            return f"💧 ${v:,.2f}M"
        else:
            return f"🔬 ${v*1000:,.0f}K"

    df["ADTV (20)"] = df["adtv_m"].apply(_format_adtv_us_badge)

    # Intraday Closing Range & Day 0 Levels from drift_cache
    df["closing_range"] = df["symbol"].apply(lambda s: drift_cache.get(str(s).strip(), {}).get("closing_range"))
    df["day0_high"] = df["symbol"].apply(lambda s: drift_cache.get(str(s).strip(), {}).get("day0_high"))
    df["day0_low"] = df["symbol"].apply(lambda s: drift_cache.get(str(s).strip(), {}).get("day0_low"))
    df["risk_to_stop"] = df["symbol"].apply(lambda s: drift_cache.get(str(s).strip(), {}).get("risk_pct"))

    # CANSLIM Relative Strength & Reaction Candle Enrichment
    df = _compute_relative_strength_ranks(df)
    df["RS Rating"] = df["rs_raw_pct"].apply(_format_rs_badge)
    df["Reaction Candle"] = df.apply(lambda r: _format_reaction_badge(r.get("day0_rx"), r.get("rel_vol"), r.get("closing_range")), axis=1)
    df["Closing Range %"] = df["closing_range"].apply(_format_closing_range_badge)
    df["EP Setup"] = df.apply(lambda r: _classify_episodic_pivot(r.get("day0_rx"), r.get("rel_vol"), r.get("closing_range"), r.get("pead_drift")), axis=1)
    df["PEAD Drift"] = df["pead_drift"].apply(_format_drift_badge)

    # YoY Fundamental Growth & CANSLIM Pillar C Classification
    if "eps_yoy" not in df.columns:
        df["eps_yoy"] = None
    if "rev_yoy" not in df.columns:
        df["rev_yoy"] = None
    df["YoY EPS %"] = df["eps_yoy"].apply(_format_growth_val)
    df["YoY Sales %"] = df["rev_yoy"].apply(_format_growth_val)
    df["CANSLIM Growth"] = df.apply(lambda r: _format_growth_badge(r.get("eps_yoy"), r.get("rev_yoy"), r.get("surprise")), axis=1)

    # Dynamic Seasonal Intelligence (Auto-computes quarter, year, dates, badges dynamically)
    season_info = get_active_reporting_season(df["date"], market="US")
    cutoff_date = season_info["cutoff_date"]
    current_badge = season_info["current_badge"]
    prior_badge = season_info["prior_badge"]

    df["Season"] = df["date"].apply(lambda d: current_badge if str(d)[:10] >= cutoff_date else prior_badge)

    # --- QUICK STRATEGY PRESET PILLS ---
    quick_preset = "All Strategies"
    if hasattr(st, "pills"):
        quick_preset = st.pills(
            "⚡ Quick Strategy Presets:",
            options=[
                "All Strategies",
                "💎 CANSLIM Super-Growth (EPS ≥ 25% & Sales ≥ 20%)",
                "🏆 Episodic Pivot (EP) Setups (Rx ≥ 3%, CR ≥ 70%)",
                "📈 Positive Drift (> 0%)",
                "🏆 Holy Grail (Dual Beat + RS ≥ 80 + Drift > 0%)",
                "🔥 Dual Beats",
                "🟢 Volume Surge (≥ 2.0x Vol)",
                "🚀 Rocket EPS (≥+5%)",
                "⚡ Top-Line Sales",
                "📅 Last 7 Days"
            ],
            default="All Strategies",
            key="pead_us_quick_pills"
        )

    # --- FILTER STUDIO HEADER & ACTIONS ---
    fsh_col1, fsh_col2 = st.columns([5, 1.2])
    with fsh_col1:
        st.markdown("""
        <div style="background: rgba(15,23,42,0.65); border: 1px solid rgba(255,255,255,0.08); border-left: 3px solid #38bdf8; border-radius: 10px; padding: 10px 16px; margin: 10px 0 10px 0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 0.82rem; font-weight: 700; color: #f8fafc; letter-spacing: 0.5px;">🎛️ US PEAD MOMENTUM & PRECISION FILTER STUDIO</span>
                <span style="font-size: 0.70rem; color: #38bdf8; background: rgba(56,189,248,0.12); padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(56,189,248,0.25); font-weight: 600;">11 Strategy Dimensions</span>
            </div>
            <div style="font-size: 0.72rem; color: #94a3b8;">
                Select strategy presets above or fine-tune quarter, market cap tier, surprises, drift horizon, sector contagion, and volume below
            </div>
        </div>
        """, unsafe_allow_html=True)
    with fsh_col2:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Sync Live Feed", key="pead_us_sync_live_btn", help="Clear cache and re-fetch up to 1,500 latest US declared earnings from TradingView"):
            fetch_us_declared_earnings.clear()
            st.rerun()

    # --- ROW 1: PRIMARY TIMING & SURPRISE CONTROLS ---
    f1, f2, f3, f4 = st.columns([1.3, 1.2, 1.2, 1.3])
    with f1:
        cohort_opts = [
            season_info["cohort_label_current"],
            season_info["cohort_label_full"],
            season_info["cohort_label_prior"],
            "🌐 All Available Quarters"
        ]
        cohort_filter = st.selectbox(
            "Reporting Quarter / Season:",
            cohort_opts,
            index=0,
            key="pead_us_cohort_sel",
            help="Filter by US quarterly reporting cohort. Current Season isolates freshly reported Q3 results."
        )

    with f2:
        default_surp_idx = 1 if quick_preset in ("🔥 Dual Beats", "🏆 Holy Grail (Dual Beat + RS ≥ 80)") else (2 if quick_preset == "🚀 Rocket EPS (≥+5%)" else (4 if quick_preset == "⚡ Top-Line Sales" else 0))
        surp_filter = st.selectbox(
            "Surprise Focus Filter:",
            [
                "All Declared Results",
                "🔥 Dual Beats Only (Both EPS & Revenue Beat)",
                "🚀 Rocket EPS Beats (≥ +5% EPS Surprise)",
                "🟢 All EPS Beats (> 0% EPS Surprise)",
                "⚡ Revenue Beats (> 0% Sales Surprise)",
                "🔴 EPS Misses (< 0% EPS Surprise)",
                "🔴 Double Misses (Both EPS & Sales Miss)"
            ],
            index=default_surp_idx,
            key="pead_us_dec_surp_sel"
        )

    with f3:
        default_rec_idx = 2 if quick_preset == "📅 Last 7 Days" else 0
        recency_filter = st.selectbox(
            "Drift Recency Horizon:",
            [
                "All Dates in Selected Cohort",
                "⚡ Last 24 Hours / Today",
                "🔥 Last 7 Days (Hot Momentum)",
                "📅 Last 14 Days (Active Base Formation)",
                "📅 Last 30 Days (Past Month Reactions)",
                "📅 Last 60 Days (Primary PEAD Drift Window)",
                "⏳ Full 90 Days (Complete Quarterly Horizon)"
            ],
            index=default_rec_idx,
            key="pead_us_dec_recency",
            help="Lookback timeframe relative to announcement date to track PEAD base formation and post-earnings drift."
        )

    with f4:
        search_query = st.text_input(
            "🔍 Search Symbol / Company:",
            placeholder="e.g. GS, JPM, APLD, NVDA, TSLA...",
            key="pead_us_dec_search"
        ).strip().upper()

    # --- ROW 2: MARKET CAP, ADTV(20) LIQUIDITY, SECTOR & CANSLIM LEADERSHIP ---
    all_sectors = ["All Sectors"] + sorted([s for s in df["sector"].dropna().unique() if s and s != "Other"])
    if "Other" in df["sector"].values:
        all_sectors.append("Other")

    af1, af2, af3, af4, af5 = st.columns([1.2, 1.2, 1.1, 1.1, 1.2])
    with af1:
        mcap_filter = st.selectbox(
            "Market Cap Tier:",
            [
                "All Market Caps (No Floor)",
                "🛡️ Liquid Equities (≥ $50M)",
                "👑 Large Cap (> $10B)",
                "⚡ Mid Cap ($2B - $10B)",
                "🌱 Small Cap ($300M - $2B)",
                "🔬 Micro Cap ($50M - $300M)",
                "🪙 Nano Cap (< $50M)"
            ],
            index=0,
            key="pead_us_dec_mcap_sel",
            help="Filter by US company market capitalization tier"
        )
    with af2:
        us_adtv_filter = st.selectbox(
            "ADTV(20) Liquidity Floor:",
            [
                "💧 Min $500K (Active Baseline)",
                "⚡ Min $1 Million (Liquid)",
                "🚀 Min $2 Millions (High Liquidity)",
                "🔥 Min $5 Millions (Institutional)",
                "👑 Min $10 Millions (Heavy Liquid)",
                "💎 Min $25 Millions (Mega Liquid)",
                "🌐 All (No Minimum Liquidity Floor)"
            ],
            index=0,
            key="pead_us_adtv_sel",
            help="20-Day Average Daily Traded Value. Default floor is $500K to eliminate illiquid micro-cap traps."
        )
    with af3:
        sector_filter = st.selectbox(
            "Sector Scope:",
            all_sectors,
            index=0,
            key="pead_us_sector_sel",
            help="Filter by GICS / TradingView Industry Sector to analyze group strength and peer contagion"
        )
    with af4:
        default_rs_idx = 1 if quick_preset == "🏆 Holy Grail (Dual Beat + RS ≥ 80)" else 0
        rs_filter = st.selectbox(
            "CANSLIM RS Rating:",
            [
                "Any RS Rating",
                "🟢 Leaders (RS ≥ 80)",
                "🔵 Strong (RS ≥ 70)",
                "⚪ Above Avg (RS ≥ 50)"
            ],
            index=default_rs_idx,
            key="pead_us_rs_sel",
            help="O'Neil 3-Month Composite Relative Strength Percentile (0-99). Top CANSLIM winners typically have RS ≥ 80."
        )
    with af5:
        setup_filter = st.selectbox(
            "PEAD Drift Setup:",
            [
                "All PEAD Setups",
                "🔥 Monster Dual Beat (Top Drift)",
                "🚀 Prime Dual Beat Leader",
                "🚀 Prime EPS Drift",
                "⚡ Top-Line Sales Runner",
                "🟢 Moderate EPS Drift",
                "⚠️ Post-Earnings Drag"
            ],
            index=0,
            key="pead_us_dec_setup_sel"
        )

    # --- ROW 2.5: CANSLIM FUNDAMENTAL GROWTH & EP ANATOMY ---
    gf1, gf2, gf3, gf4, gf5 = st.columns([1.1, 1.1, 1.2, 1.2, 1.4])
    with gf1:
        min_eps_yoy_opt = st.selectbox(
            "Min YoY EPS Growth %:",
            ["Any", "> 0% (Positive Growth)", "≥ +25% (CANSLIM Min)", "≥ +50% (High Growth)", "≥ +100% (Triple-Digit)"],
            index=2 if quick_preset == "💎 CANSLIM Super-Growth (EPS ≥ 25% & Sales ≥ 20%)" else 0,
            key="pead_us_min_eps_yoy",
            help="CANSLIM Rule 'C': Top winning stocks typically have quarterly YoY EPS growth >= +25%."
        )
    with gf2:
        min_rev_yoy_opt = st.selectbox(
            "Min YoY Sales Growth %:",
            ["Any", "> 0% (Positive Growth)", "≥ +15%", "≥ +25% (CANSLIM Min)", "≥ +50% (High Growth)"],
            index=3 if quick_preset == "💎 CANSLIM Super-Growth (EPS ≥ 25% & Sales ≥ 20%)" else 0,
            key="pead_us_min_rev_yoy",
            help="CANSLIM Top-Line Rule: Demands sales growth >= +20% to validate earnings quality."
        )
    with gf3:
        min_cr_opt = st.selectbox(
            "Min Closing Range %:",
            ["Any", "≥ 50% (Constructive)", "≥ 70% (Accumulation)", "≥ 80% (Power Close)"],
            index=2 if quick_preset == "🏆 Episodic Pivot (EP) Setups (Rx ≥ 3%, CR ≥ 70%)" else 0,
            key="pead_us_min_cr",
            help="Day 0 Intraday Closing Range = (Close - Low)/(High - Low) * 100. Measures buying conviction into the close."
        )
    with gf4:
        ep_setup_opt = st.selectbox(
            "Episodic Pivot Quality:",
            ["All Setups", "🏆 True Episodic Pivot (EP)", "🔥 Power Close Only (CR ≥ 75%)", "⚡ Momentum Breakouts", "🟢 Constructive (CR ≥ 50%)"],
            index=1 if quick_preset == "🏆 Episodic Pivot (EP) Setups (Rx ≥ 3%, CR ≥ 70%)" else 0,
            key="pead_us_ep_setup_sel",
            help="Filter for institutional Episodic Pivot catalysts vs gap-and-fade traps."
        )
    with gf5:
        quality_filter = st.selectbox(
            "Quality Armor & Traps:",
            [
                "All Stocks",
                "💎 Exclude Surprise Mirages (YoY EPS < 0%)",
                "🚫 Exclude Gap & Crap Traps (CR < 35%)",
                "🛡️ Pure Institutional Quality (No Mirages & No Traps)"
            ],
            index=0,
            key="pead_us_quality_armor",
            help="Filter out value traps where stocks beat lowered expectations despite contracting growth, or where heavy volume sold off into the close."
        )

    # --- ROW 3: PRECISION THRESHOLDS, PEAD DRIFT, VOLUME & PRICE REACTION ---
    bf1, bf2, bf3, bf4, bf5 = st.columns([1.1, 1.1, 1.3, 1.1, 1.1])
    with bf1:
        min_eps_opt = st.selectbox(
            "Min EPS Surprise %:",
            ["Any", "> 0% (Positive Beat)", "≥ +5%", "≥ +10%", "≥ +20%", "≥ +50%"],
            index=2 if quick_preset == "🚀 Rocket EPS (≥+5%)" else 0,
            key="pead_us_min_eps_sel"
        )
    with bf2:
        min_rev_opt = st.selectbox(
            "Min Revenue Surprise %:",
            ["Any", "> 0% (Positive Beat)", "≥ +2%", "≥ +5%", "≥ +10%", "≥ +20%"],
            index=1 if quick_preset == "⚡ Top-Line Sales" else 0,
            key="pead_us_min_rev_sel"
        )
    with bf3:
        default_drift_idx = 1 if quick_preset in ("📈 Positive Drift (> 0%)", "🏆 Holy Grail (Dual Beat + RS ≥ 80 + Drift > 0%)") else 0
        drift_filter = st.selectbox(
            "PEAD Drift Filter:",
            [
                "All Drift Trajectories",
                "📈 Positive Drift Only (> 0% Accumulation ↗)",
                "🚀 Strong Drift (≥ +5%)",
                "🔥 Massive Drift (≥ +15%)",
                "🔴 Negative Drift / Fade (< 0%)"
            ],
            index=default_drift_idx,
            key="pead_us_drift_sel",
            help="Cumulative price drift return from announcement date to today. Positive drift confirms institutional accumulation."
        )
    with bf4:
        default_vol_idx = 2 if quick_preset == "🟢 Volume Surge (≥ 2.0x Vol)" else 0
        vol_filter = st.selectbox(
            "Volume Surge Factor:",
            [
                "Any Volume",
                "⚡ Heavy Volume (≥ 1.5x)",
                "🚀 Extreme Volume (≥ 2.0x)",
                "🔥 Institutional Blast (≥ 3.0x)"
            ],
            index=default_vol_idx,
            key="pead_us_vol_sel",
            help="10-day relative volume multiplier on earnings announcement"
        )
    with bf5:
        react_filter = st.selectbox(
            "Day 0 Price Reaction:",
            [
                "Any Price Reaction",
                "🟢 Positive Reaction (> 0%)",
                "🚀 Breakout (> +3% with 1.5x Vol)",
                "🔴 Gap Fade / Drag (< 0%)"
            ],
            index=0,
            key="pead_us_react_sel",
            help="Announcement day price reaction to identify institutional accumulation breakouts vs gap fade traps"
        )

    # --- FILTER EXECUTION ---
    filtered = df.copy()

    # 1. Reporting Season / Cohort Filter
    if cohort_filter == season_info["cohort_label_current"]:
        filtered = filtered[filtered["date"].astype(str).str[:10] >= cutoff_date]
    elif cohort_filter == season_info["cohort_label_prior"]:
        filtered = filtered[filtered["date"].astype(str).str[:10] < cutoff_date]

    # Quick pills enforcement overrides
    if quick_preset == "📅 Last 7 Days":
        filtered = filtered[filtered["recency_days"] <= 7]
    elif quick_preset == "💎 CANSLIM Super-Growth (EPS ≥ 25% & Sales ≥ 20%)":
        f_ey = pd.to_numeric(filtered.get("eps_yoy"), errors="coerce").fillna(0)
        f_ry = pd.to_numeric(filtered.get("rev_yoy"), errors="coerce").fillna(0)
        filtered = filtered[(f_ey >= 25.0) & (f_ry >= 20.0)]
    elif quick_preset == "🏆 Episodic Pivot (EP) Setups (Rx ≥ 3%, CR ≥ 70%)":
        f_chg = pd.to_numeric(filtered.get("day0_rx"), errors="coerce").fillna(0)
        f_cr = pd.to_numeric(filtered.get("closing_range"), errors="coerce").fillna(50)
        f_vol = pd.to_numeric(filtered.get("rel_vol"), errors="coerce").fillna(1)
        filtered = filtered[(f_chg >= 3.0) & (f_cr >= 70.0) & (f_vol >= 1.5)]

    # 2. Surprise Focus
    f_eps = pd.to_numeric(filtered["surprise"], errors="coerce")
    f_rev = pd.to_numeric(filtered["rev_surprise"], errors="coerce") if "rev_surprise" in filtered.columns else pd.Series(dtype=float)

    if "Dual Beats" in surp_filter or quick_preset in ("🔥 Dual Beats", "🏆 Holy Grail (Dual Beat + RS ≥ 80 + Drift > 0%)"):
        filtered = filtered[(f_eps > 0) & (f_rev > 0)]
    elif "Rocket EPS" in surp_filter:
        filtered = filtered[f_eps >= 5.0]
    elif "All EPS Beats" in surp_filter:
        filtered = filtered[f_eps > 0]
    elif "Revenue Beats" in surp_filter:
        filtered = filtered[f_rev > 0]
    elif "Double Misses" in surp_filter:
        filtered = filtered[(f_eps < 0) & (f_rev < 0)]
    elif "EPS Misses" in surp_filter:
        filtered = filtered[f_eps < 0]

    # 3. Recency Horizon
    if "Last 24 Hours" in recency_filter:
        filtered = filtered[filtered["recency_days"] <= 1]
    elif "Last 7 Days" in recency_filter:
        filtered = filtered[filtered["recency_days"] <= 7]
    elif "Last 14 Days" in recency_filter:
        filtered = filtered[filtered["recency_days"] <= 14]
    elif "Last 30 Days" in recency_filter:
        filtered = filtered[filtered["recency_days"] <= 30]
    elif "Last 60 Days" in recency_filter:
        filtered = filtered[filtered["recency_days"] <= 60]
    elif "Full 90" in recency_filter:
        filtered = filtered[filtered["recency_days"] <= 90]

    # 4. Market Cap Tier, Sector Scope, RS Rating & Setup
    if "≥ $50M" in mcap_filter:
        filtered = filtered[filtered["mcap_m"] >= 50]
    elif "Large Cap" in mcap_filter:
        filtered = filtered[filtered["mcap_m"] >= 10000]
    elif "Mid Cap" in mcap_filter:
        filtered = filtered[(filtered["mcap_m"] >= 2000) & (filtered["mcap_m"] < 10000)]
    elif "Small Cap" in mcap_filter:
        filtered = filtered[(filtered["mcap_m"] >= 300) & (filtered["mcap_m"] < 2000)]
    elif "Micro Cap" in mcap_filter:
        filtered = filtered[(filtered["mcap_m"] >= 50) & (filtered["mcap_m"] < 300)]
    elif "Nano Cap" in mcap_filter:
        filtered = filtered[(filtered["mcap_m"] > 0) & (filtered["mcap_m"] < 50)]

    # 4.1 ADTV(20) Liquidity Floor Filter
    if "Min $500K" in us_adtv_filter:
        filtered = filtered[filtered["adtv_m"] >= 0.50]
    elif "Min $1 Million" in us_adtv_filter:
        filtered = filtered[filtered["adtv_m"] >= 1.0]
    elif "Min $2 Millions" in us_adtv_filter:
        filtered = filtered[filtered["adtv_m"] >= 2.0]
    elif "Min $5 Millions" in us_adtv_filter:
        filtered = filtered[filtered["adtv_m"] >= 5.0]
    elif "Min $10 Millions" in us_adtv_filter:
        filtered = filtered[filtered["adtv_m"] >= 10.0]
    elif "Min $25 Millions" in us_adtv_filter:
        filtered = filtered[filtered["adtv_m"] >= 25.0]

    if sector_filter != "All Sectors":
        filtered = filtered[filtered["sector"] == sector_filter]

    if "RS ≥ 80" in rs_filter or quick_preset == "🏆 Holy Grail (Dual Beat + RS ≥ 80 + Drift > 0%)":
        filtered = filtered[filtered["rs_raw_pct"] >= 80]
    elif "RS ≥ 70" in rs_filter:
        filtered = filtered[filtered["rs_raw_pct"] >= 70]
    elif "RS ≥ 50" in rs_filter:
        filtered = filtered[filtered["rs_raw_pct"] >= 50]

    if setup_filter != "All PEAD Setups":
        filtered = filtered[filtered["PEAD Drift Setup"] == setup_filter]

    # 4.5. CANSLIM YoY Growth & Episodic Pivot Filters
    f_ey = pd.to_numeric(filtered.get("eps_yoy"), errors="coerce")
    if "> 0%" in min_eps_yoy_opt:
        filtered = filtered[f_ey > 0]
    elif "≥ +25%" in min_eps_yoy_opt:
        filtered = filtered[f_ey >= 25.0]
    elif "≥ +50%" in min_eps_yoy_opt:
        filtered = filtered[f_ey >= 50.0]
    elif "≥ +100%" in min_eps_yoy_opt:
        filtered = filtered[f_ey >= 100.0]

    f_ry = pd.to_numeric(filtered.get("rev_yoy"), errors="coerce")
    if "> 0%" in min_rev_yoy_opt:
        filtered = filtered[f_ry > 0]
    elif "≥ +15%" in min_rev_yoy_opt:
        filtered = filtered[f_ry >= 15.0]
    elif "≥ +25%" in min_rev_yoy_opt:
        filtered = filtered[f_ry >= 25.0]
    elif "≥ +50%" in min_rev_yoy_opt:
        filtered = filtered[f_ry >= 50.0]

    f_cr = pd.to_numeric(filtered.get("closing_range"), errors="coerce")
    if "≥ 50%" in min_cr_opt:
        filtered = filtered[f_cr >= 50.0]
    elif "≥ 70%" in min_cr_opt:
        filtered = filtered[f_cr >= 70.0]
    elif "≥ 80%" in min_cr_opt:
        filtered = filtered[f_cr >= 80.0]

    if "True Episodic Pivot" in ep_setup_opt:
        filtered = filtered[filtered["EP Setup"] == "🏆 True Episodic Pivot (EP)"]
    elif "Power Close" in ep_setup_opt:
        filtered = filtered[f_cr >= 75.0]
    elif "Momentum Breakouts" in ep_setup_opt:
        filtered = filtered[filtered["EP Setup"].isin(["🏆 True Episodic Pivot (EP)", "⚡ Momentum Gap"])]
    elif "Constructive" in ep_setup_opt:
        filtered = filtered[f_cr >= 50.0]

    if "Exclude Surprise Mirages" in quality_filter or "Pure Institutional Quality" in quality_filter:
        f_ey_m = pd.to_numeric(filtered.get("eps_yoy"), errors="coerce")
        f_surp_m = pd.to_numeric(filtered.get("surprise"), errors="coerce")
        is_mirage = (f_surp_m > 0) & (f_ey_m < 0)
        filtered = filtered[~is_mirage]

    if "Exclude Gap & Crap" in quality_filter or "Pure Institutional Quality" in quality_filter:
        f_cr_t = pd.to_numeric(filtered.get("closing_range"), errors="coerce")
        is_trap = (f_cr_t < 35.0)
        filtered = filtered[~is_trap]

    # 5. Min EPS Threshold
    f_eps = pd.to_numeric(filtered["surprise"], errors="coerce")
    if "> 0%" in min_eps_opt:
        filtered = filtered[f_eps > 0]
    elif "≥ +5%" in min_eps_opt:
        filtered = filtered[f_eps >= 5.0]
    elif "≥ +10%" in min_eps_opt:
        filtered = filtered[f_eps >= 10.0]
    elif "≥ +20%" in min_eps_opt:
        filtered = filtered[f_eps >= 20.0]
    elif "≥ +50%" in min_eps_opt:
        filtered = filtered[f_eps >= 50.0]

    # 6. Min Rev Threshold
    f_rev = pd.to_numeric(filtered["rev_surprise"], errors="coerce") if "rev_surprise" in filtered.columns else pd.Series(dtype=float)
    if "> 0%" in min_rev_opt:
        filtered = filtered[f_rev > 0]
    elif "≥ +2%" in min_rev_opt:
        filtered = filtered[f_rev >= 2.0]
    elif "≥ +5%" in min_rev_opt:
        filtered = filtered[f_rev >= 5.0]
    elif "≥ +10%" in min_rev_opt:
        filtered = filtered[f_rev >= 10.0]
    elif "≥ +20%" in min_rev_opt:
        filtered = filtered[f_rev >= 20.0]

    # 7. PEAD Drift Trajectory Filter
    f_drift = pd.to_numeric(filtered["pead_drift"], errors="coerce").fillna(0.0)
    if quick_preset in ("📈 Positive Drift (> 0%)", "🏆 Holy Grail (Dual Beat + RS ≥ 80 + Drift > 0%)") or "Positive Drift Only" in drift_filter:
        filtered = filtered[f_drift > 0]
    elif "Strong Drift" in drift_filter:
        filtered = filtered[f_drift >= 5.0]
    elif "Massive Drift" in drift_filter:
        filtered = filtered[f_drift >= 15.0]
    elif "Negative Drift" in drift_filter:
        filtered = filtered[f_drift < 0]

    # 8. Volume Factor & Reaction Filter
    f_vol = pd.to_numeric(filtered["rel_vol"], errors="coerce").fillna(1.0)
    f_chg = pd.to_numeric(filtered["day0_rx"], errors="coerce").fillna(0.0)

    if quick_preset == "🟢 Volume Surge (≥ 2.0x Vol)" or "≥ 2.0x" in vol_filter:
        filtered = filtered[f_vol >= 2.0]
    elif "≥ 1.5x" in vol_filter:
        filtered = filtered[f_vol >= 1.5]
    elif "≥ 3.0x" in vol_filter:
        filtered = filtered[f_vol >= 3.0]

    if "Positive Reaction" in react_filter:
        filtered = filtered[f_chg > 0]
    elif "Breakout" in react_filter:
        filtered = filtered[(f_chg >= 3.0) & (f_vol >= 1.5)]
    elif "Gap Fade" in react_filter:
        filtered = filtered[f_chg < 0]

    # 8. Search Query
    if search_query:
        filtered = filtered[
            filtered["symbol"].str.contains(search_query, na=False) |
            filtered["company"].str.upper().str.contains(search_query, na=False)
        ]

    # --- SECTOR CONTAGION AGGREGATION & RADAR ---
    contagion_df = _aggregate_thematic_contagion(filtered if len(filtered) >= 3 else df)
    _render_thematic_contagion_banner(contagion_df, market="US")
    _render_thematic_contagion_matrix(contagion_df, upcoming_df=df_upcoming, market="US")

    # --- DYNAMIC HUD METRICS (Synchronized with filtered selection) ---
    f_total = len(filtered)
    f_surp_eps = pd.to_numeric(filtered["surprise"], errors="coerce")
    f_surp_rev = pd.to_numeric(filtered["rev_surprise"], errors="coerce") if "rev_surprise" in filtered.columns else pd.Series(dtype=float)

    dual_beats = int(((f_surp_eps > 0) & (f_surp_rev > 0)).sum()) if not f_surp_rev.empty else 0
    eps_beats = int((f_surp_eps > 0).sum())
    rev_beats = int((f_surp_rev > 0).sum()) if not f_surp_rev.empty else 0

    dual_rate = round(dual_beats / max(1, f_total) * 100, 1)
    eps_rate = round(eps_beats / max(1, f_total) * 100, 1)
    rev_rate = round(rev_beats / max(1, f_total) * 100, 1)

    if cohort_filter == season_info["cohort_label_current"]:
        season_hud_label = f"Current Season: {season_info['current_q']}"
    elif cohort_filter == season_info["cohort_label_prior"]:
        season_hud_label = f"Prior Quarter: {season_info['prior_q']}"
    elif f_total == len(df):
        season_hud_label = f"Full Universe ({f_total} Stocks)"
    else:
        season_hud_label = f"Filtered Selection ({f_total} of {len(df)})"

    _render_earnings_hud_cards(f_total, dual_beats, dual_rate, eps_beats, eps_rate, rev_beats, rev_rate, season_label=season_hud_label)

    # --- VIEW MODE, DECK MODE & SORTING CONTROL BAR ---
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    v_col1, v_col2, v_col3 = st.columns([1.6, 1.4, 1.2])
    with v_col1:
        view_options = [
            "⚡ Side-by-Side (Map Left + Table Right)",
            "📊 Full Quadrant Map",
            "📋 Full Table Grid"
        ]
        if hasattr(st, "segmented_control"):
            view_mode = st.segmented_control(
                "Layout View:",
                options=view_options,
                default=view_options[0],
                key="pead_us_view_mode"
            )
        else:
            view_mode = st.radio(
                "Layout View:",
                view_options,
                index=0,
                horizontal=True,
                key="pead_us_view_mode"
            )
    with v_col2:
        deck_mode_opts = ["⚡ Essential Execution Deck", "📑 Deep Analyst Deck"]
        if hasattr(st, "segmented_control"):
            deck_mode = st.segmented_control(
                "Table Deck View:",
                options=deck_mode_opts,
                default=deck_mode_opts[0],
                key="pead_us_deck_mode"
            )
        else:
            deck_mode = st.radio(
                "Table Deck View:",
                deck_mode_opts,
                index=0,
                horizontal=True,
                key="pead_us_deck_mode"
            )
    with v_col3:
        sort_by = st.selectbox(
            "Sort Order:",
            [
                "📈 PEAD Drift % (Highest Drift First)",
                "⚡ YoY Sales Growth % (Highest First)",
                "🔥 YoY EPS Growth % (Highest First)",
                "💧 ADTV (20) (Most Liquid First)",
                "📅 Release Date (Latest First)",
                "🔥 EPS Surprise % (Highest First)",
                "⚡ Revenue Surprise % (Highest First)",
                "💪 RS Rating (Leaders First)",
                "🕯️ Day 0 Reaction (Top First)",
                "🔤 Symbol (A to Z)"
            ],
            index=0,
            key="pead_us_sort_sel"
        )

    # Apply Sorting
    if "PEAD Drift" in sort_by:
        filtered = filtered.sort_values(by=["pead_drift", "surprise"], ascending=[False, False])
    elif "YoY Sales" in sort_by:
        filtered = filtered.sort_values(by=["rev_yoy", "pead_drift"], ascending=[False, False], na_position='last')
    elif "YoY EPS" in sort_by:
        filtered = filtered.sort_values(by=["eps_yoy", "pead_drift"], ascending=[False, False], na_position='last')
    elif "ADTV" in sort_by:
        filtered = filtered.sort_values(by=["adtv_m", "pead_drift"], ascending=[False, False])
    elif "Release Date" in sort_by:
        filtered = filtered.sort_values(by=["date", "surprise"], ascending=[False, False])
    elif "EPS Surprise" in sort_by:
        filtered = filtered.sort_values(by=["surprise", "rev_surprise"], ascending=[False, False])
    elif "Revenue Surprise" in sort_by:
        filtered = filtered.sort_values(by=["rev_surprise", "surprise"], ascending=[False, False])
    elif "RS Rating" in sort_by:
        filtered = filtered.sort_values(by=["rs_raw_pct", "pead_drift"], ascending=[False, False])
    elif "Day 0 Reaction" in sort_by or "Price Reaction" in sort_by:
        filtered = filtered.sort_values(by=["day0_rx", "rel_vol"], ascending=[False, False])
    elif "Symbol" in sort_by:
        filtered = filtered.sort_values(by=["symbol"], ascending=[True])

    # Formatted display columns
    filtered = filtered.copy()
    filtered["Est. EPS"] = filtered["est"].apply(lambda v: f"${float(v):,.2f}" if pd.notna(v) and v is not None else "-")
    filtered["Actual EPS"] = filtered["act"].apply(lambda v: f"${float(v):,.2f}" if pd.notna(v) and v is not None else "-")
    filtered["EPS Surprise %"] = filtered["surprise"].apply(
        lambda v: f"+{float(v):,.2f}%" if pd.notna(v) and float(v) > 0 else (f"{float(v):,.2f}%" if pd.notna(v) else "-")
    )

    filtered["Est. Revenue"] = filtered["rev_est"].apply(lambda v: _format_revenue_val(v, "$"))
    filtered["Actual Revenue"] = filtered["rev_act"].apply(lambda v: _format_revenue_val(v, "$"))
    filtered["Revenue Surprise %"] = filtered["rev_surprise"].apply(
        lambda v: f"+{float(v):,.2f}%" if pd.notna(v) and float(v) > 0 else (f"{float(v):,.2f}%" if pd.notna(v) else "-")
    )
    filtered["Day 0 High"] = filtered["day0_high"].apply(lambda v: f"${float(v):,.2f}" if pd.notna(v) and v is not None else "-")
    filtered["Day 0 Low"] = filtered["day0_low"].apply(lambda v: f"${float(v):,.2f}" if pd.notna(v) and v is not None else "-")
    filtered["Risk to Stop %"] = filtered["risk_to_stop"].apply(lambda v: f"{float(v):.1f}%" if pd.notna(v) and v is not None else "-")

    is_essential_deck = "Essential" in deck_mode
    if is_essential_deck:
        display_df = filtered[[
            "symbol", "PEAD Drift", "Reaction Candle", "Market Cap", "ADTV (20)", "YoY EPS %", "YoY Sales %", "CANSLIM Growth",
            "RS Rating", "Dual Surprise Verdict", "EPS Surprise %", "Revenue Surprise %", "sector", "TradingView", "date"
        ]].rename(
            columns={
                "symbol": "Symbol",
                "PEAD Drift": "PEAD Drift",
                "Reaction Candle": "Day 0 Rx",
                "Market Cap": "Market Cap",
                "ADTV (20)": "ADTV (20)",
                "YoY EPS %": "YoY EPS %",
                "YoY Sales %": "YoY Sales %",
                "CANSLIM Growth": "Growth Status",
                "RS Rating": "RS",
                "Dual Surprise Verdict": "Dual Verdict",
                "EPS Surprise %": "EPS Surp %",
                "Revenue Surprise %": "Sales Surp %",
                "sector": "Sector",
                "TradingView": "Chart ↗",
                "date": "Release Time"
            }
        )
    else:
        display_df = filtered[[
            "symbol", "PEAD Drift", "Reaction Candle", "Market Cap", "ADTV (20)", "Closing Range %", "EP Setup",
            "YoY EPS %", "YoY Sales %", "CANSLIM Growth", "RS Rating", "Dual Surprise Verdict", "EPS Surprise %",
            "Actual EPS", "Est. EPS", "Revenue Surprise %", "Actual Revenue", "Est. Revenue",
            "Day 0 High", "Day 0 Low", "Risk to Stop %", "PEAD Drift Setup",
            "sector", "industry", "company", "Season", "TradingView", "date"
        ]].rename(
            columns={
                "symbol": "Symbol",
                "PEAD Drift": "PEAD Drift",
                "Reaction Candle": "Day 0 Rx",
                "Market Cap": "Market Cap",
                "ADTV (20)": "ADTV (20)",
                "Closing Range %": "Closing Range",
                "EP Setup": "EP Quality",
                "YoY EPS %": "YoY EPS %",
                "YoY Sales %": "YoY Sales %",
                "CANSLIM Growth": "Growth Status",
                "RS Rating": "RS",
                "Dual Surprise Verdict": "Dual Verdict",
                "EPS Surprise %": "EPS Surp %",
                "Actual EPS": "Actual EPS",
                "Est. EPS": "Est. EPS",
                "Revenue Surprise %": "Sales Surp %",
                "Actual Revenue": "Actual Rev",
                "Est. Revenue": "Est. Rev",
                "Day 0 High": "Day 0 High (Pivot)",
                "Day 0 Low": "Day 0 Low (Stop)",
                "Risk to Stop %": "Risk to Stop",
                "PEAD Drift Setup": "PEAD Setup",
                "sector": "Sector",
                "industry": "Industry",
                "company": "Company Name",
                "Season": "Season",
                "TradingView": "Chart ↗",
                "date": "Release Time"
            }
        )

    us_col_config = {
        "Symbol": st.column_config.TextColumn("Symbol", width="small", help="US Ticker Symbol"),
        "PEAD Drift": st.column_config.TextColumn("PEAD Drift", width="medium", help="Cumulative total return from announcement date to today"),
        "Day 0 Rx": st.column_config.TextColumn("Day 0 Rx", width="medium", help="Announcement day price reaction %, volume surge multiplier, and closing range %"),
        "Market Cap": st.column_config.TextColumn("Market Cap", width="small", help="Institutional Market Capitalization (👑 Large > $10B, ⚡ Mid $2-10B, 🌱 Small $300M-$2B, 🔬 Micro < $300M)"),
        "ADTV (20)": st.column_config.TextColumn("ADTV (20)", width="small", help="20-Day Average Daily Traded Value (👑 > $1B, 🔥 $50M+, ⚡ $5-50M, 💧 $500K-$5M)"),
        "YoY EPS %": st.column_config.TextColumn("YoY EPS %", width="small", help="Quarterly YoY EPS growth % (CANSLIM 'C' requires ≥ +25%)"),
        "YoY Sales %": st.column_config.TextColumn("YoY Sales %", width="small", help="Quarterly YoY Revenue/Sales growth % (CANSLIM requires ≥ +20%)"),
        "Growth Status": st.column_config.TextColumn("Growth Status", width="medium", help="CANSLIM Pillar C classification: Super-Growth vs Surprise Mirage"),
        "Closing Range": st.column_config.TextColumn("Closing Range", width="small", help="Day 0 Intraday Closing Range: (Close - Low)/(High - Low) * 100"),
        "EP Quality": st.column_config.TextColumn("EP Quality", width="medium", help="Episodic Pivot (EP) classification: True EP vs Gap-and-Crap trap"),
        "RS": st.column_config.TextColumn("O'Neil RS", width="small", help="CANSLIM 3M Composite Relative Strength Percentile (0-99)"),
        "Dual Verdict": st.column_config.TextColumn("Dual Verdict", width="medium", help="Consensus beat verdict on EPS and Revenue"),
        "EPS Surp %": st.column_config.TextColumn("EPS Surp %", width="small", help="Percentage by which reported EPS exceeded consensus"),
        "Sales Surp %": st.column_config.TextColumn("Sales Surp %", width="small", help="Percentage by which reported Revenue exceeded consensus"),
        "Day 0 High (Pivot)": st.column_config.TextColumn("Day 0 High (Pivot)", width="small", help="Episodic Pivot Buy Trigger Price"),
        "Day 0 Low (Stop)": st.column_config.TextColumn("Day 0 Low (Stop)", width="small", help="Day 0 Low Structural Invalidation Stop Level"),
        "Risk to Stop": st.column_config.TextColumn("Risk to Stop", width="small", help="Percentage distance from current price to Day 0 Low stop loss"),
        "Sector": st.column_config.TextColumn("Sector", width="medium", help="Institutional Industry Sector"),
        "Industry": st.column_config.TextColumn("Industry", width="medium", help="Detailed Industry Sub-group"),
        "Chart ↗": st.column_config.LinkColumn(
            "Chart ↗",
            display_text="Open ↗",
            help="Inspect post-earnings reaction candle & volume on TradingView"
        ),
        "Release Time": st.column_config.TextColumn("Release Time", width="small"),
        "Actual EPS": st.column_config.TextColumn("Actual EPS", width="small"),
        "Est. EPS": st.column_config.TextColumn("Est. EPS", width="small"),
        "Actual Rev": st.column_config.TextColumn("Actual Rev", width="small"),
        "Est. Rev": st.column_config.TextColumn("Est. Rev", width="small"),
        "PEAD Setup": st.column_config.TextColumn("PEAD Setup", width="medium"),
        "Company Name": st.column_config.TextColumn("Company Name", width="medium"),
        "Season": st.column_config.TextColumn("Season", width="small")
    }

    chart_key = "pead_us_quadrant_chart"

    # --- RENDER SELECTED VIEW ---
    if "Side-by-Side" in view_mode:
        map_col, tbl_col = st.columns([1.0, 1.15])
        with map_col:
            sel_symbol = _render_surprise_quadrant_chart(filtered, market_currency="$", height=540, chart_key=chart_key)

        with tbl_col:
            active_display_df = display_df
            if sel_symbol and sel_symbol in display_df["Symbol"].values:
                iso_c1, iso_c2 = st.columns([3, 1])
                with iso_c1:
                    st.markdown(f"""
                    <div style="padding: 6px 12px; background: rgba(56,189,248,0.18); border: 1px solid #38bdf8; border-radius: 8px; font-size: 0.78rem; color: #f8fafc; font-weight: 600; margin-bottom: 6px;">
                        🎯 <b>Chart Selection Isolated:</b> <span style="color: #38bdf8; font-weight: 700;">{sel_symbol}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with iso_c2:
                    if st.button("✕ Reset Map", key=f"{chart_key}_clear_btn"):
                        chart_ver_key = f"{chart_key}_ver"
                        st.session_state[chart_ver_key] = st.session_state.get(chart_ver_key, 0) + 1
                        st.session_state.pop(chart_key, None)
                        st.rerun()
                active_display_df = display_df[display_df["Symbol"] == sel_symbol]

            deck_label = "⚡ Essential" if is_essential_deck else "📑 Deep Analyst"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: rgba(15,23,42,0.7); border: 1px solid rgba(255,255,255,0.08); border-left: 3px solid #38bdf8; border-radius: 8px; margin-bottom: 8px;">
                <div style="font-size: 0.88rem; font-weight: 700; color: #f8fafc;">📋 {deck_label} Results Grid</div>
                <div style="font-size: 0.72rem; color: #38bdf8; font-weight: 600; background: rgba(56,189,248,0.15); padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(56,189,248,0.3);">Showing {len(active_display_df)} of {len(df)} Stocks</div>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(
                active_display_df,
                column_config=us_col_config,
                use_container_width=True,
                hide_index=True,
                height=540
            )
    elif "Quadrant Map" in view_mode:
        sel_symbol = _render_surprise_quadrant_chart(filtered, market_currency="$", height=600, chart_key=chart_key)
        if sel_symbol and sel_symbol in display_df["Symbol"].values:
            st.markdown(f"##### 🎯 Selected Stock Profile: `{sel_symbol}`")
            st.dataframe(display_df[display_df["Symbol"] == sel_symbol], column_config=us_col_config, use_container_width=True, hide_index=True)
    else:
        deck_label = "⚡ Essential" if is_essential_deck else "📑 Deep Analyst"
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: rgba(15,23,42,0.7); border: 1px solid rgba(255,255,255,0.08); border-left: 3px solid #38bdf8; border-radius: 8px; margin-bottom: 8px;">
            <div style="font-size: 0.88rem; font-weight: 700; color: #f8fafc;">📋 {deck_label} Results Grid</div>
            <div style="font-size: 0.72rem; color: #38bdf8; font-weight: 600; background: rgba(56,189,248,0.15); padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(56,189,248,0.3);">Showing {len(display_df)} of {len(df)} Stocks</div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            display_df,
            column_config=us_col_config,
            use_container_width=True,
            hide_index=True,
            height=540
        )

    # --- TRADINGVIEW WATCHLIST EXPORTER ---
    with st.expander("📋 Export Filtered US Beat Tickers to TradingView Watchlist"):
        st.caption("Copy this comma-separated list into TradingView's Watchlist Import box.")
        f_eps_all = pd.to_numeric(filtered["surprise"], errors="coerce")
        f_rev_all = pd.to_numeric(filtered["rev_surprise"], errors="coerce") if "rev_surprise" in filtered.columns else pd.Series(dtype=float)

        dual_symbols = [s for s in filtered[(f_eps_all > 0) & (f_rev_all > 0)]["symbol"].dropna().unique() if s]
        holy_grail_symbols = [s for s in filtered[(f_eps_all > 0) & (f_rev_all > 0) & (filtered["rs_raw_pct"] >= 80)]["symbol"].dropna().unique() if s]
        volume_symbols = [s for s in filtered[filtered["rel_vol"] >= 2.0]["symbol"].dropna().unique() if s]
        all_symbols = [s for s in filtered["symbol"].dropna().unique() if s]

        e_t1, e_t2, e_t3, e_t4 = st.tabs([
            f"🔥 Dual Beats ({len(dual_symbols)})",
            f"🏆 Holy Grail ({len(holy_grail_symbols)})",
            f"🟢 Volume Surge ({len(volume_symbols)})",
            f"📋 All Filtered ({len(all_symbols)})"
        ])
        with e_t1:
            st.code(",".join(dual_symbols), language="text")
        with e_t2:
            st.code(",".join(holy_grail_symbols), language="text")
        with e_t3:
            st.code(",".join(volume_symbols), language="text")
        with e_t4:
            st.code(",".join(all_symbols), language="text")


# -----------------------------------------------------------------------------
# US: UPCOMING EARNINGS VIEW
# -----------------------------------------------------------------------------
def _render_us_upcoming_earnings(df_raw):
    if df_raw.empty:
        st.warning("⚠️ Live US corporate earnings feed currently unavailable. Please use the direct portal links above.")
        return

    df = df_raw.copy()
    df["parsed_date"] = pd.to_datetime(df["Event Start Date"], errors="coerce")
    today = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
    df["days_left"] = (df["parsed_date"].dt.floor("D") - today).dt.days
    df["Timeline"] = df["days_left"].apply(_format_days_badge)
    df["TradingView"] = df["Symbol"].apply(lambda s: f"https://www.tradingview.com/chart/?symbol={s}")

    total_cos = len(df)
    upcoming_cos = int(df["Reported EPS"].isna().sum())
    next_7d = int(((df["days_left"] >= 0) & (df["days_left"] <= 7)).sum())
    next_14d = int(((df["days_left"] >= 0) & (df["days_left"] <= 14)).sum())

    # Render Glassmorphic HUD Cards
    _render_upcoming_hud_cards(total_cos, upcoming_cos, next_7d, next_14d, is_india=False)

    st.markdown("---")
    f1, f2 = st.columns([1.5, 2.5])
    with f1:
        timing_filter = st.selectbox(
            "Session Timing:",
            ["All Sessions", "☀️ BMO (Before Market Open)", "🌙 AMC (After Market Close)"],
            index=0,
            key="pead_us_up_timing"
        )
    with f2:
        search_query = st.text_input(
            "🔍 Search Symbol / Company:",
            placeholder="e.g. TSLA, NFLX, TSM, AAPL...",
            key="pead_us_up_search"
        ).strip().upper()

    filtered = df.copy()
    if "BMO" in timing_filter:
        filtered = filtered[filtered["Timing"] == "BMO"]
    elif "AMC" in timing_filter:
        filtered = filtered[filtered["Timing"] == "AMC"]

    if search_query:
        filtered = filtered[
            filtered["Symbol"].str.contains(search_query, na=False) |
            filtered["Company"].str.upper().str.contains(search_query, na=False)
        ]

    filtered = filtered.sort_values(by=["days_left", "Symbol"], ascending=[True, True])

    filtered["Formatted Date"] = filtered["parsed_date"].dt.strftime("%Y-%m-%d %H:%M")
    filtered["Est. EPS"] = filtered["EPS Estimate"].apply(lambda v: f"${float(v):,.2f}" if pd.notna(v) and str(v).replace('.','',1).replace('-','',1).isdigit() else "-")

    st.markdown(f"##### Showing **{len(filtered)}** US Upcoming Corporate Earnings Releases")
    display_df = filtered[["Timeline", "Formatted Date", "Symbol", "Company", "Timing", "Est. EPS", "TradingView"]].rename(
        columns={
            "Formatted Date": "Release Time (EST)",
            "TradingView": "Chart ↗"
        }
    )

    st.dataframe(
        display_df,
        column_config={
            "Chart ↗": st.column_config.LinkColumn(
                "Chart ↗",
                display_text="Open ↗",
                help="Open institutional chart on TradingView"
            ),
            "Release Time (EST)": st.column_config.TextColumn(width="medium"),
            "Symbol": st.column_config.TextColumn(width="small"),
            "Timeline": st.column_config.TextColumn(width="medium"),
            "Timing": st.column_config.TextColumn(width="small"),
            "Est. EPS": st.column_config.TextColumn(width="small")
        },
        use_container_width=True,
        hide_index=True,
        height=480
    )

    with st.expander("📋 Export Upcoming US Tickers to TradingView Watchlist"):
        st.caption("Copy this comma-separated list into TradingView's Watchlist Import box.")
        symbols_tv = [s for s in filtered["Symbol"].dropna().unique() if s]
        st.code(",".join(symbols_tv), language="text")



def _render_us_earnings_calendar():
    if hasattr(st, "segmented_control"):
        focus_mode = st.segmented_control(
            "🇺🇸 US Earnings Intelligence View:",
            [
                "🎯 Declared Results & Earnings Surprises (PEAD Momentum)",
                "⏳ Upcoming Corporate Earnings (Binary Risk Radar)"
            ],
            default="🎯 Declared Results & Earnings Surprises (PEAD Momentum)",
            key="pead_us_focus_mode"
        )
    else:
        focus_mode = st.radio(
            "🇺🇸 US Earnings Intelligence View:",
            [
                "🎯 Declared Results & Earnings Surprises (PEAD Momentum)",
                "⏳ Upcoming Corporate Earnings (Binary Risk Radar)"
            ],
            horizontal=True,
            key="pead_us_focus_mode"
        )

    if not focus_mode or "Declared Results" in focus_mode:
        df_dec = fetch_us_declared_earnings()
        df_up = fetch_us_corporate_earnings_calendar()
        _render_us_declared_earnings(df_dec, df_upcoming=df_up)
    else:
        df_up = fetch_us_corporate_earnings_calendar()
        _render_us_upcoming_earnings(df_up)


def _render_macro_economic_embed():
    """TradingView Macroeconomic indicators calendar inside an optional expander."""
    with st.expander("🌐 Optional: View Macroeconomic Indicators Calendar (CPI, WPI, Unemployment, Fed FOMC, RBI)"):
        st.caption("Note: This feed shows macroeconomic events and central bank economic prints. For individual company stock earnings, refer to the tables above.")
        
        m_c1, m_c2 = st.columns([1.5, 2])
        with m_c1:
            macro_country = st.radio("Economic Region:", ["🇮🇳 India Macro", "🇺🇸 US Macro"], horizontal=True, key="macro_reg_radio")
        
        cc = "in" if "India" in macro_country else "us"
        cname = "India" if "India" in macro_country else "United States"

        embed_html = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"></head>
        <body style="margin:0; background:transparent;">
            <div class="tradingview-widget-container">
                <div class="tradingview-widget-container__widget"></div>
                <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
                {{
                    "colorTheme": "dark",
                    "isTransparent": true,
                    "width": "100%",
                    "height": "500",
                    "locale": "en",
                    "importanceFilter": "0,1",
                    "countryFilter": "{cc}"
                }}
                </script>
            </div>
        </body>
        </html>
        """
        try:
            components.html(embed_html, height=520, scrolling=True)
        except Exception:
            st.info(f"Open macroeconomic indicators directly on TradingView: [TradingView {cname}](https://www.tradingview.com/markets/{cc}/)")


def render_corporate_earnings_calendar():
    # Top Executive Hero Header & Launchpad
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(15,23,42,0.85) 0%, rgba(30,41,59,0.6) 100%); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px 18px; margin-bottom: 14px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.35);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 0.68rem; font-weight: 800; letter-spacing: 1.2px; text-transform: uppercase; background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.3); padding: 2px 8px; border-radius: 6px;">
                        ⚡ INSTITUTIONAL PEAD ENGINE
                    </span>
                    <span style="font-size: 0.72rem; color: #94a3b8;">
                        Consensus Estimates &bull; Actual Surprises &bull; Drift Radar
                    </span>
                </div>
                <div style="font-size: 1.30rem; font-weight: 800; color: #f8fafc; margin-top: 3px; letter-spacing: -0.02em;">
                    Live Corporate Earnings Calendar & Consensus Intelligence
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8; max-width: 850px; margin-top: 1px; line-height: 1.4;">
                    Real-time quarterly announcements cross-referenced with Nifty 500 & Wall Street leaders to track post-earnings announcement drift (PEAD) momentum and hedge binary gap risks.
                </div>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <a href="https://www.tradingview.com/markets/india/earnings/" target="_blank" rel="noopener noreferrer" style="padding: 6px 12px; background: rgba(56,189,248,0.1); color: #38bdf8; border: 1px solid rgba(56,189,248,0.25); border-radius: 6px; text-decoration: none; font-size: 0.75rem; font-weight: 600;">
                    🇮🇳 India Earnings ↗
                </a>
                <a href="https://www.tradingview.com/markets/stocks-usa/earnings/" target="_blank" rel="noopener noreferrer" style="padding: 6px 12px; background: rgba(56,189,248,0.1); color: #38bdf8; border: 1px solid rgba(56,189,248,0.25); border-radius: 6px; text-decoration: none; font-size: 0.75rem; font-weight: 600;">
                    🇺🇸 US Earnings ↗
                </a>
                <a href="https://www.bseindia.com/corporates/ann.html" target="_blank" rel="noopener noreferrer" style="padding: 6px 12px; background: rgba(255,255,255,0.04); color: #cbd5e1; border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; text-decoration: none; font-size: 0.75rem; font-weight: 500;">
                    🏛️ BSE Filings ↗
                </a>
                <a href="https://www.nasdaq.com/market-activity/earnings" target="_blank" rel="noopener noreferrer" style="padding: 6px 12px; background: rgba(255,255,255,0.04); color: #cbd5e1; border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; text-decoration: none; font-size: 0.75rem; font-weight: 500;">
                    🏛️ Nasdaq Hub ↗
                </a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Market Selection
    if hasattr(st, "segmented_control"):
        market_sel = st.segmented_control(
            "Select Market Universe:",
            ["🇮🇳 India (NSE / BSE Corporate Results & Surprises)", "🇺🇸 United States (NYSE / NASDAQ EPS & Consensus)"],
            default="🇮🇳 India (NSE / BSE Corporate Results & Surprises)",
            key="pead_market_univ_radio"
        )
    else:
        market_sel = st.radio(
            "Select Market Universe:",
            ["🇮🇳 India (NSE / BSE Corporate Results & Surprises)", "🇺🇸 United States (NYSE / NASDAQ EPS & Consensus)"],
            horizontal=True,
            key="pead_market_univ_radio"
        )

    if not market_sel or "India" in market_sel:
        _render_india_earnings_calendar()
    else:
        _render_us_earnings_calendar()

    st.markdown("<br>", unsafe_allow_html=True)
    _render_macro_economic_embed()


# -----------------------------------------------------------------------------
# 25-DAY POST-EARNINGS FLAG STALKER (SHAKE PRYZBY & WILLIAM O'NEIL DOCTRINES)
# -----------------------------------------------------------------------------
def render_25d_pead_flag_stalker():
    """
    Master 25-Day Post-Earnings Flag Stalker (PEAD Cohort Engine).
    Tracks stocks reporting earnings within the prior 25 trading sessions,
    monitoring 10/20 EMA flag digestion, volume dry-up, and wave-2 breakout re-ignitions.
    Based on Shake Pryzby (Pinpoint Trading p. 184) & William O'Neil (HTMMIS p. 155).
    """
    st.markdown("### 🚩 25-Day Post-Earnings Flag Stalker")
    st.caption("CANSLIM Catalyst Stalker: Tracks institutional earnings gappers over the critical 25-session digestion window, monitoring 10/20 EMA support, volume dry-up, and wave-2 flag breakouts.")

    with st.expander("📖 Shake Pryzby & William O'Neil Post-Earnings Flag Playbook (25-Day Lifecycle Explained)", expanded=False):
        st.markdown(r"""
        #### 🎓 Core Rules for 25-Day Post-Earnings Flag Digestion:
        1. **The Catalyst (O'Neil 'C' & Episodic Pivot)**: Blowout quarterly earnings or revenue surprise creates a massive institutional supply-demand imbalance.
        2. **The 25 Trading Session Timeframe (Shake Pryzby)**:
           - **How Time Frame is Defined**: Measured from the earnings report session ($T=0$) up to **25 trading sessions** (~35 to 38 calendar days).
           - **Cohort Membership**: All stocks reporting quarterly results within the last 25 trading sessions enter the stalker cohort automatically.
           - True Market Leaders rarely top on earnings day. Instead, they digest and absorb early profit-takers over **5 to 25 trading sessions**.
        3. **Dynamic 10 EMA / 20 EMA Coiling Stages**:
           - **Days 1–5 (Initial Reaction & High Flag Setup)**: Stock establishes its post-earnings range (Flag Pole & High).
           - **Days 6–20 (Extreme Volume Dry-Up & 10/20 EMA Coiling)**: Daily volume contracts dramatically ($< 0.35\times$ earnings day volume), indicating institutional absorption and complete seller exhaustion as the 10 EMA catches up.
           - **Days 10–25 (Wave-2 Breakout Ignition)**: Stock ignites out of the flag consolidation range on expanding volume ($> 1.5\times$ 20D average), confirming the next institutional accumulation wave.
        4. **What Happens on Day 26+? (Graduation Rule)**:
           - If a stock reaches session 26 without breaking out, the immediate **earnings drift catalyst window is officially completed**.
           - The stock **graduates out of the short-term PEAD Flag Stalker** and transitions into the **Standard Stage 2 Base-Building Universe** (e.g. Minervini Cup-with-Handle, Flat Base, or David Ryan 3-Weeks Tight).
           - Its next move is subsequently tracked by the **Master Intraday Breakout Scanner** when it completes its larger multi-month base.
        5. **Invalidation (The Gap Violation Rule)**:
           - If price closes below the gap day low or breaks decisively below the 20 EMA on heavy volume, the PEAD flag thesis is invalidated (`⚠️ Gap Violated`).
        """)

    stalker_market = st.radio(
        "Select Market Universe:",
        ["🇮🇳 India (NSE Equities)", "🇺🇸 US Equities (NYSE / NASDAQ)"],
        horizontal=True,
        key="pead_stalker_market_radio"
    )
    is_us = "US" in stalker_market

    with st.spinner("Scanning 25-day post-earnings cohort..."):
        if is_us:
            df_raw = fetch_us_declared_earnings()
        else:
            df_raw = fetch_india_declared_earnings()

    if df_raw is None or df_raw.empty:
        st.info("No declared earnings data found in cache or live scanner.")
        return

    df = df_raw.copy()
    if 'date' not in df.columns:
        st.warning("Declared earnings data is missing the date field.")
        return

    df['report_date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['report_date'])

    today_dt = pd.to_datetime(datetime.now().date())
    df['days_ago'] = (today_dt - df['report_date']).dt.days

    # Filter to earnings announced within the last 38 calendar days (~25 trading sessions)
    cohort_df = df[(df['days_ago'] >= 0) & (df['days_ago'] <= 38)].copy()
    cohort_df['symbol'] = cohort_df['symbol'].astype(str).str.strip()
    if not is_us:
        try:
            from nifty500_universe import fetch_nifty500_tickers
            nifty500_set = set(fetch_nifty500_tickers(show_progress=False))
        except Exception:
            from nifty500_universe import FALLBACK_TICKERS
            nifty500_set = set(FALLBACK_TICKERS)
        cohort_df['is_nifty500'] = cohort_df['symbol'].apply(lambda s: s in nifty500_set)
    else:
        cohort_df['is_nifty500'] = True

    cohort_df['trading_days_ago'] = (cohort_df['days_ago'] * 5 / 7).round(0).astype(int)
    cohort_df['eps_yoy'] = pd.to_numeric(cohort_df.get('eps_yoy'), errors='coerce')
    cohort_df['rev_yoy'] = pd.to_numeric(cohort_df.get('rev_yoy'), errors='coerce')
    cohort_df['surprise'] = pd.to_numeric(cohort_df.get('surprise'), errors='coerce')
    cohort_df['rev_surprise'] = pd.to_numeric(cohort_df.get('rev_surprise'), errors='coerce')

    if cohort_df.empty:
        st.info("No earnings reports found within the 25-session window.")
        return

    # Fetch batch PEAD drift from announcement date
    market_str = "US" if is_us else "IN"
    sym_date_tuples = tuple(zip(cohort_df["symbol"].astype(str), cohort_df["date"].astype(str)))
    drift_cache = fetch_batch_pead_drift(sym_date_tuples, market=market_str)

    def _resolve_result_drift(row):
        s = str(row["symbol"]).strip()
        days = row.get("days_ago", 999)
        d_info = drift_cache.get(s, {})
        hist_drift = d_info.get("pead_drift")
        if hist_drift is not None and not pd.isna(hist_drift):
            return float(hist_drift)
        if days == 0:
            return float(row.get("change") or 0.0)
        perf = row.get("perf_1m")
        if pd.notna(perf) and perf is not None:
            return float(perf)
        return 0.0

    cohort_df['result_drift'] = cohort_df.apply(_resolve_result_drift, axis=1)

    # 20-Day Average Daily Traded Value (ADTV) Resolution & Badge
    def _resolve_stalker_adtv(row):
        s = str(row["symbol"]).strip()
        d_info = drift_cache.get(s, {})
        drift_adtv = d_info.get("adtv")
        if drift_adtv is not None and drift_adtv > 0:
            return float(drift_adtv)
        if is_us:
            tv_adtv = row.get("adtv_m")
            if pd.notna(tv_adtv) and tv_adtv is not None and tv_adtv > 0:
                return float(tv_adtv)
            c = row.get("close")
            v = row.get("volume")
            if pd.notna(c) and pd.notna(v) and c and v:
                return round((float(c) * float(v)) / 1e6, 2)
        else:
            tv_adtv = row.get("adtv_cr")
            if pd.notna(tv_adtv) and tv_adtv is not None and tv_adtv > 0:
                return float(tv_adtv)
            c = row.get("close")
            v = row.get("volume")
            if pd.notna(c) and pd.notna(v) and c and v:
                return round((float(c) * float(v)) / 1e7, 2)
        return 0.0

    cohort_df['adtv_val'] = cohort_df.apply(_resolve_stalker_adtv, axis=1)

    def _format_stalker_adtv_badge(val):
        if val is None or pd.isna(val) or val <= 0:
            return "-"
        v = float(val)
        if is_us:
            if v >= 1000.0:
                return f"👑 ${v/1000:,.1f}B"
            elif v >= 50.0:
                return f"🔥 ${v:,.1f}M"
            elif v >= 5.0:
                return f"⚡ ${v:,.1f}M"
            elif v >= 0.50:
                return f"💧 ${v:,.2f}M"
            else:
                return f"🔬 ${v*1000:,.0f}K"
        else:
            if v >= 10.0:
                return f"👑 ₹{v:,.1f} Cr"
            elif v >= 5.0:
                return f"🔥 ₹{v:,.1f} Cr"
            elif v >= 1.0:
                return f"⚡ ₹{v:,.1f} Cr"
            elif v >= 0.50:
                return f"💧 ₹{v * 100:,.0f} L"
            else:
                return f"🔬 ₹{v * 100:,.0f} L"

    cohort_df['ADTV (20)'] = cohort_df['adtv_val'].apply(_format_stalker_adtv_badge)

    # Compute Digestion and Coiling Indicators
    def classify_flag_status(row):
        s = str(row.get('symbol', '')).strip()
        perf_w = row.get('perf_w') or 0.0
        perf_1m = row.get('perf_1m') or 0.0
        res_drift = row.get('result_drift') if pd.notna(row.get('result_drift')) else perf_1m
        chg = row.get('change') or 0.0
        rel_vol = row.get('rel_vol') or 1.0
        c = row.get('close') or 0.0

        # Invalidation Rule 5: price violated earnings gap low
        d_info = drift_cache.get(s, {})
        day0_l = d_info.get('day0_low')
        if day0_l is not None and day0_l > 0 and c < day0_l:
            return "⚠️ Gap Violated"

        if res_drift < -6.0 or chg < -4.0:
            return "⚠️ Gap Violated"
        elif chg >= 2.5 and rel_vol >= 1.25:
            return "🔥 Flag Breakout Ignition"
        elif abs(chg) <= 2.2 and perf_w >= -1.5 and res_drift >= -1.0:
            return "🚩 Coiling at 10 EMA"
        elif perf_w < -2.0 and res_drift >= -2.0:
            return "🎣 20 EMA Retest"
        else:
            return "📦 Consolidating Base"

    cohort_df['Flag Status'] = cohort_df.apply(classify_flag_status, axis=1)

    # Compute Action Directives and Trade Parameters
    def get_trader_action(row):
        status = row.get('Flag Status', '')
        if 'Coiling' in status:
            return "🎯 STALK PIVOT (Alert at High)"
        elif 'Breakout' in status:
            return "🚀 BUY LIVE (Wave-2 Ignition)"
        elif 'Retest' in status:
            return "🎣 WATCH SUPPORT (Wait for Hammer)"
        elif 'Violated' in status:
            return "⛔ AVOID / CUT (Gap Violated)"
        else:
            return "📋 WATCHLIST (Let Base Form)"

    cohort_df['Action Directive'] = cohort_df.apply(get_trader_action, axis=1)

    # Calculated Execution Parameters: Flag Pivot, Suggested Stop-Loss, and Max Risk %
    def calc_flag_pivot(row):
        s = str(row.get('symbol', '')).strip()
        c = row.get('close') or 0.0
        if c <= 0: return 0.0
        d_info = drift_cache.get(s, {})
        day0_h = d_info.get('day0_high')
        if day0_h is not None and day0_h > 0 and day0_h >= c:
            return round(float(day0_h), 2)
        pw = row.get('perf_w') or 0.0
        mult = 1.0 + max(0.015, min(0.06, (pw / 100.0) * 0.15) if pw > 0 else 0.02)
        return round(c * mult, 2)

    def calc_stop_loss(row):
        s = str(row.get('symbol', '')).strip()
        c = row.get('close') or 0.0
        if c <= 0: return 0.0
        d_info = drift_cache.get(s, {})
        day0_l = d_info.get('day0_low')
        if day0_l is not None and day0_l > 0 and day0_l < c:
            gap_risk = (c - day0_l) / c
            if 0.015 <= gap_risk <= 0.08:
                return round(float(day0_l), 2)
        return round(c * 0.965, 2)

    cohort_df['Flag Pivot'] = cohort_df.apply(calc_flag_pivot, axis=1)
    cohort_df['Stop Loss'] = cohort_df.apply(calc_stop_loss, axis=1)
    cohort_df['Risk %'] = cohort_df.apply(
        lambda r: round(((r['Stop Loss'] - r['close']) / r['close']) * 100.0, 1) if (r.get('close') or 0) > 0 else 0.0,
        axis=1
    )

    # Estimate Volume Dry-Up / Supply Exhaustion State
    def get_supply_state(rv):
        try:
            rv_f = float(rv) if rv is not None else 1.0
            if rv_f <= 0.40:
                return f"💧 Dry-Up ({rv_f:.2f}x)"
            elif rv_f >= 1.25:
                return f"⚡ Surging ({rv_f:.2f}x)"
            else:
                return f"⚪ Normal ({rv_f:.2f}x)"
        except Exception:
            return "⚪ Normal"

    cohort_df['Supply Absorption'] = cohort_df['rel_vol'].apply(get_supply_state)

    # TradingView links
    if is_us:
        cohort_df['TickerLink'] = cohort_df['symbol'].apply(
            lambda s: f"https://www.tradingview.com/chart/?symbol={s}"
        )
    else:
        cohort_df['TickerLink'] = cohort_df['symbol'].apply(
            lambda s: f"https://www.tradingview.com/chart/?symbol=NSE:{s}"
        )

    # -------------------------------------------------------------
    # ACTIONABLE FILTERS & QUICK HORIZON PILLS
    # -------------------------------------------------------------
    st.markdown("##### ⚡ Tactical Execution Presets")
    c_pills, c_toggle = st.columns([3.2, 1.2])
    with c_pills:
        action_pill = st.pills(
            "Select Tactical Horizon:",
            ["🎯 Actionable Today (Coiling & Ignitions)", "🚀 Live Wave-2 Breakouts", "🚩 Coiling at 10 EMA", "🎣 20 EMA Retests", "📦 Developing Bases", "👑 Nifty 500 Leaders", "All Stocks"],
            default="🎯 Actionable Today (Coiling & Ignitions)",
            label_visibility="collapsed",
            key="pead_action_pill"
        )
    with c_toggle:
        hide_violated = st.checkbox("🛡️ Hide Violated Gaps", value=True, key="pead_hide_violated", help="Automatically hides failed earnings gaps and breakdowns")

    # Secondary Granular Controls, ADTV(20) Liquidity, Result Drift & Fundamental YoY Growth Filters
    c_f1, c_f2, c_f3, c_f4, c_f5, c_f6, c_f7, c_f8 = st.columns([1.1, 1.1, 1.1, 1.0, 1.0, 1.0, 1.1, 1.1])
    with c_f1:
        status_filter = st.selectbox(
            "Granular Status Filter:",
            ["All Setups", "🚩 Coiling at 10 EMA", "🔥 Flag Breakout Ignition", "🎣 20 EMA Retest", "📦 Consolidating Base", "⚠️ Gap Violated"],
            index=0,
            key="pead_status_filter"
        )
    with c_f2:
        if is_us:
            adtv_stalker_opts = [
                "💧 Min $500K (Active Baseline)",
                "⚡ Min $1 Million (Liquid)",
                "🚀 Min $2 Millions (High Liquidity)",
                "🔥 Min $5 Millions (Institutional)",
                "👑 Min $10 Millions (Heavy Liquid)",
                "💎 Min $25 Millions (Mega Liquid)",
                "🌐 All (No Floor)"
            ]
        else:
            adtv_stalker_opts = [
                "💧 Min ₹50 Lakhs (Active Baseline)",
                "⚡ Min ₹1 Crore (Liquid)",
                "🚀 Min ₹2 Crores (High Liquidity)",
                "🔥 Min ₹5 Crores (Institutional)",
                "👑 Min ₹10 Crores (Heavy Liquid)",
                "💎 Min ₹25 Crores (Ultra Liquid)",
                "🌐 All (No Floor)"
            ]
        adtv_filter = st.selectbox(
            "ADTV(20) Liquidity Floor:",
            adtv_stalker_opts,
            index=0,
            key="pead_stalker_adtv_sel",
            help="20-Day Average Daily Traded Value. Default floor is ₹50 Lakhs (INR 0.50 Cr) / $500K to eliminate illiquid penny/micro-cap traps."
        )
    with c_f3:
        min_drift_opt = st.selectbox(
            "Result Drift %:",
            [
                "Any Drift",
                "> 0% (Positive Drift)",
                "≥ +5% (Strong Drift)",
                "≥ +10% (High Drift)",
                "≥ +20% (Monster Drift)",
                "< 0% (Negative Drift / Dip)"
            ],
            index=0,
            key="pead_stalker_min_drift",
            help="Cumulative price drift % measured from earnings announcement session (Day 0 close) to current close."
        )
    with c_f4:
        min_eps_yoy_opt = st.selectbox(
            "Min YoY EPS Growth %:",
            ["Any", "> 0% (Positive Growth)", "≥ +25% (CANSLIM Rule 'C')", "≥ +50% (High Growth)", "≥ +100% (Triple-Digit)"],
            index=0,
            key="pead_stalker_min_eps_yoy",
            help="Quarterly YoY EPS Growth %. Top CANSLIM winners typically have quarterly EPS growth ≥ +25%."
        )
    with c_f5:
        min_rev_yoy_opt = st.selectbox(
            "Min YoY Sales Growth %:",
            ["Any", "> 0% (Positive Growth)", "≥ +15%", "≥ +20% (CANSLIM Top-Line)", "≥ +25%", "≥ +50% (High Growth)"],
            index=0,
            key="pead_stalker_min_rev_yoy",
            help="Quarterly YoY Revenue / Sales Growth %. Confirms customer demand and earnings quality."
        )
    with c_f6:
        min_surp_opt = st.selectbox(
            "Min EPS Surprise %:",
            ["Any (Includes Uncovered)", "> 0% (Beat Consensus)", "≥ +5% (Rocket Beat)", "≥ +10%", "≥ +20%", "≥ +50%"],
            index=0,
            key="pead_min_surp",
            help="Consensus analyst EPS surprise %. 'Any' preserves high-growth small/micro-caps uncovered by analysts."
        )
    with c_f7:
        sort_by = st.selectbox(
            "Sort Cohort By:",
            [
                "🚀 Result Drift (Highest First)",
                "💧 ADTV (20) (Most Liquid First)",
                "🔥 YoY EPS Growth (Highest First)",
                "⚡ YoY Sales Growth (Highest First)",
                "📈 EPS Surprise (Highest First)",
                "🚀 1M Drift (Highest First)",
                "🕯️ Today % (Top Gainers First)",
                "📅 Sessions Ago (Most Recent First)",
                "🔤 Symbol (A to Z)"
            ],
            index=0,
            key="pead_stalker_sort_sel"
        )
    with c_f8:
        search_query = st.text_input("Search Company or Symbol:", "", key="pead_search").strip().upper()

    filtered_df = cohort_df.copy()

    # Apply Quick Preset Filter
    if action_pill == "🎯 Actionable Today (Coiling & Ignitions)":
        filtered_df = filtered_df[filtered_df['Flag Status'].isin(["🚩 Coiling at 10 EMA", "🔥 Flag Breakout Ignition"])]
    elif action_pill == "🚀 Live Wave-2 Breakouts":
        filtered_df = filtered_df[filtered_df['Flag Status'] == "🔥 Flag Breakout Ignition"]
    elif action_pill == "🚩 Coiling at 10 EMA":
        filtered_df = filtered_df[filtered_df['Flag Status'] == "🚩 Coiling at 10 EMA"]
    elif action_pill == "🎣 20 EMA Retests":
        filtered_df = filtered_df[filtered_df['Flag Status'] == "🎣 20 EMA Retest"]
    elif action_pill == "📦 Developing Bases":
        filtered_df = filtered_df[filtered_df['Flag Status'] == "📦 Consolidating Base"]
    elif action_pill == "👑 Nifty 500 Leaders":
        filtered_df = filtered_df[filtered_df.get('is_nifty500', True)]

    # Hide Violated Gaps if checked
    if hide_violated:
        filtered_df = filtered_df[filtered_df['Flag Status'] != "⚠️ Gap Violated"]

    # Filter by ADTV(20) Liquidity Floor
    f_adtv = pd.to_numeric(filtered_df.get('adtv_val'), errors='coerce').fillna(0.0)
    if "Min ₹50 Lakhs" in adtv_filter or "Min $500K" in adtv_filter:
        filtered_df = filtered_df[f_adtv >= 0.50]
    elif "Min ₹1 Crore" in adtv_filter or "Min $1 Million" in adtv_filter:
        filtered_df = filtered_df[f_adtv >= 1.0]
    elif "Min ₹2 Crore" in adtv_filter or "Min $2 Million" in adtv_filter:
        filtered_df = filtered_df[f_adtv >= 2.0]
    elif "Min ₹5 Crore" in adtv_filter or "Min $5 Million" in adtv_filter:
        filtered_df = filtered_df[f_adtv >= 5.0]
    elif "Min ₹10 Crore" in adtv_filter or "Min $10 Million" in adtv_filter:
        filtered_df = filtered_df[f_adtv >= 10.0]
    elif "Min ₹25 Crore" in adtv_filter or "Min $25 Million" in adtv_filter:
        filtered_df = filtered_df[f_adtv >= 25.0]

    # Filter by Result Drift %
    f_drift = pd.to_numeric(filtered_df.get('result_drift'), errors='coerce')
    if "> 0%" in min_drift_opt:
        filtered_df = filtered_df[f_drift > 0]
    elif "≥ +5%" in min_drift_opt:
        filtered_df = filtered_df[f_drift >= 5.0]
    elif "≥ +10%" in min_drift_opt:
        filtered_df = filtered_df[f_drift >= 10.0]
    elif "≥ +20%" in min_drift_opt:
        filtered_df = filtered_df[f_drift >= 20.0]
    elif "< 0%" in min_drift_opt:
        filtered_df = filtered_df[f_drift < 0]

    # Filter by YoY EPS Growth
    f_ey = pd.to_numeric(filtered_df.get('eps_yoy'), errors='coerce')
    if "> 0%" in min_eps_yoy_opt:
        filtered_df = filtered_df[f_ey > 0]
    elif "≥ +25%" in min_eps_yoy_opt:
        filtered_df = filtered_df[f_ey >= 25.0]
    elif "≥ +50%" in min_eps_yoy_opt:
        filtered_df = filtered_df[f_ey >= 50.0]
    elif "≥ +100%" in min_eps_yoy_opt:
        filtered_df = filtered_df[f_ey >= 100.0]

    # Filter by YoY Sales Growth
    f_ry = pd.to_numeric(filtered_df.get('rev_yoy'), errors='coerce')
    if "> 0%" in min_rev_yoy_opt:
        filtered_df = filtered_df[f_ry > 0]
    elif "≥ +15%" in min_rev_yoy_opt:
        filtered_df = filtered_df[f_ry >= 15.0]
    elif "≥ +20%" in min_rev_yoy_opt:
        filtered_df = filtered_df[f_ry >= 20.0]
    elif "≥ +25%" in min_rev_yoy_opt:
        filtered_df = filtered_df[f_ry >= 25.0]
    elif "≥ +50%" in min_rev_yoy_opt:
        filtered_df = filtered_df[f_ry >= 50.0]

    # Filter by Consensus EPS Surprise
    f_surp = pd.to_numeric(filtered_df.get('surprise'), errors='coerce')
    if "> 0%" in min_surp_opt:
        filtered_df = filtered_df[f_surp > 0]
    elif "≥ +5%" in min_surp_opt:
        filtered_df = filtered_df[f_surp >= 5.0]
    elif "≥ +10%" in min_surp_opt:
        filtered_df = filtered_df[f_surp >= 10.0]
    elif "≥ +20%" in min_surp_opt:
        filtered_df = filtered_df[f_surp >= 20.0]
    elif "≥ +50%" in min_surp_opt:
        filtered_df = filtered_df[f_surp >= 50.0]

    if status_filter != "All Setups":
        filtered_df = filtered_df[filtered_df['Flag Status'] == status_filter]
    if search_query:
        filtered_df = filtered_df[
            filtered_df['symbol'].str.contains(search_query, na=False) |
            filtered_df['company'].astype(str).str.upper().str.contains(search_query, na=False)
        ]

    # Metrics HUD
    coiling_count = len(cohort_df[cohort_df['Flag Status'] == '🚩 Coiling at 10 EMA'])
    breakout_count = len(cohort_df[cohort_df['Flag Status'] == '🔥 Flag Breakout Ignition'])
    violated_count = len(cohort_df[cohort_df['Flag Status'] == '⚠️ Gap Violated'])
    med_surp = cohort_df['surprise'].dropna().median() if not cohort_df['surprise'].dropna().empty else 0.0
    med_eps_yoy = cohort_df['eps_yoy'].dropna().median() if not cohort_df['eps_yoy'].dropna().empty else 0.0
    med_res_drift = cohort_df['result_drift'].dropna().median() if not cohort_df['result_drift'].dropna().empty else 0.0
    med_1m = cohort_df['perf_1m'].dropna().median() if not cohort_df['perf_1m'].dropna().empty else 0.0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("25D PEAD Cohort", f"{len(cohort_df)} Stocks", f"{len(filtered_df)} Matching Filter")
    with m2:
        st.metric("Coiling at 10 EMA", f"{coiling_count} Setups", "High-Tight Digestion")
    with m3:
        st.metric("Breakout Ignitions", f"{breakout_count} Stocks", "Wave 2 Surge Today")
    with m4:
        st.metric("Median Result Drift", f"{med_res_drift:+.1f}%", f"Median YoY EPS: {med_eps_yoy:+.1f}%")

    # -------------------------------------------------------------
    # WHAT SHOULD YOU DO TODAY? (EXECUTIVE ACTION DIRECTIVE)
    # -------------------------------------------------------------
    coiling_tickers = ", ".join(cohort_df[cohort_df['Flag Status'] == '🚩 Coiling at 10 EMA']['symbol'].tolist()) or "None"
    breakout_tickers = ", ".join(cohort_df[cohort_df['Flag Status'] == '🔥 Flag Breakout Ignition']['symbol'].tolist()) or "None"
    violated_tickers = ", ".join(cohort_df[cohort_df['Flag Status'] == '⚠️ Gap Violated']['symbol'].tolist()) or "None"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(15,23,42,0.9) 0%, rgba(30,41,59,0.7) 100%); border: 1px solid rgba(56,189,248,0.3); border-radius: 12px; padding: 14px 18px; margin: 14px 0 18px 0; box-shadow: 0 8px 24px -4px rgba(0,0,0,0.4);">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.2rem;">🎯</span>
                <strong style="color: #38bdf8; font-size: 0.95rem; letter-spacing: 0.3px;">WHAT SHOULD YOU DO LOOKING AT THIS PAGE TODAY?</strong>
            </div>
            <span style="background: rgba(56,189,248,0.15); color: #38bdf8; font-size: 0.76rem; font-weight: 700; padding: 3px 10px; border-radius: 6px; border: 1px solid #38bdf844;">CANSLIM EXECUTION DIRECTIVE</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 10px;">
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 12px;">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #38bdf8; font-weight: 700; margin-bottom: 4px;">1. High Priority (Stalk Pivot)</div>
                <div style="font-size: 0.88rem; color: #f8fafc; font-weight: 600;">{coiling_count} Coiling Flags: <span style="color: #38bdf8;">{coiling_tickers}</span></div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">Set buy stop alert at Flag Pivot. If price clears pivot on volume > 1.25x, take 2.5%–5% pilot risk.</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 12px;">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #10b981; font-weight: 700; margin-bottom: 4px;">2. Active Breakouts (Buy Live)</div>
                <div style="font-size: 0.88rem; color: #f8fafc; font-weight: 600;">{breakout_count} Wave-2 Ignitions: <span style="color: #10b981;">{breakout_tickers}</span></div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">Surging out of flag on heavy volume today. Enter market/limit order if within 5% of pivot.</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 12px;">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #ef4444; font-weight: 700; margin-bottom: 4px;">3. Invalidation (Cut / Avoid)</div>
                <div style="font-size: 0.88rem; color: #f8fafc; font-weight: 600;">{violated_count} Broken Gaps: <span style="color: #ef4444;">{violated_tickers}</span></div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">Earnings gap-day low violated. The catalyst failed. Delete from watchlist and avoid longs.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1-Click TradingView Copy Box
    tv_symbols = []
    for s in filtered_df['symbol'].dropna().unique():
        clean_s = str(s).replace('\xa0', '').strip()
        if clean_s:
            tv_symbols.append(clean_s if is_us else f"NSE:{clean_s}")
    tv_copy_str = ",".join(tv_symbols)

    with st.expander(f"📋 Copy TradingView List ({len(filtered_df)} Stocks Displayed)", expanded=False):
        if tv_copy_str:
            st.code(tv_copy_str, language="text")
        else:
            st.caption("No tickers matching the current criteria.")

    # Dynamic Sorting
    sort_mapping = {
        "🚀 Result Drift (Highest First)": (['result_drift', 'perf_1m'], [False, False]),
        "💧 ADTV (20) (Most Liquid First)": (['adtv_val', 'result_drift'], [False, False]),
        "🔥 YoY EPS Growth (Highest First)": (['eps_yoy', 'result_drift'], [False, False]),
        "⚡ YoY Sales Growth (Highest First)": (['rev_yoy', 'result_drift'], [False, False]),
        "📈 EPS Surprise (Highest First)": (['surprise', 'result_drift'], [False, False]),
        "🚀 1M Drift (Highest First)": (['perf_1m', 'change'], [False, False]),
        "🕯️ Today % (Top Gainers First)": (['change', 'result_drift'], [False, False]),
        "📅 Sessions Ago (Most Recent First)": (['trading_days_ago', 'change'], [True, False]),
        "🔤 Symbol (A to Z)": (['symbol'], [True])
    }
    sort_cols, sort_asc = sort_mapping.get(sort_by, (['result_drift', 'perf_1m'], [False, False]))
    actual_sort_cols = [c for c in sort_cols if c in filtered_df.columns]
    actual_sort_asc = [sort_asc[i] for i, c in enumerate(sort_cols) if c in filtered_df.columns]

    if actual_sort_cols:
        sorted_df = filtered_df.sort_values(by=actual_sort_cols, ascending=actual_sort_asc, na_position='last')
    else:
        sorted_df = filtered_df

    # Table Display
    display_cols = [
        'TickerLink', 'Action Directive', 'Flag Status', 'close', 'ADTV (20)', 'Flag Pivot', 'Stop Loss', 'Risk %',
        'change', 'Supply Absorption', 'trading_days_ago', 'result_drift',
        'eps_yoy', 'rev_yoy', 'surprise', 'rev_surprise',
        'perf_w', 'perf_1m', 'company', 'industry'
    ]
    valid_cols = [c for c in display_cols if c in sorted_df.columns]
    table_df = sorted_df[valid_cols]

    def style_flag_status(val):
        if 'Coiling' in str(val) or 'STALK' in str(val):
            return 'color: #38bdf8; font-weight: bold; background-color: rgba(56, 189, 248, 0.12)'
        elif 'Breakout' in str(val) or 'BUY' in str(val) or 'Surging' in str(val):
            return 'color: #10b981; font-weight: bold; background-color: rgba(16, 185, 129, 0.15)'
        elif 'Retest' in str(val) or 'WATCH' in str(val):
            return 'color: #f59e0b; font-weight: bold;'
        elif 'Violated' in str(val) or 'AVOID' in str(val):
            return 'color: #ef4444; font-weight: bold;'
        elif 'Dry-Up' in str(val):
            return 'color: #10b981; font-weight: bold;'
        return 'color: #cbd5e1;'

    def color_pct(val):
        try:
            if pd.isna(val):
                return ''
            val_f = float(val)
            if val_f > 0:
                return 'color: #10b981'
            elif val_f < 0:
                return 'color: #ef4444'
        except Exception:
            pass
        return ''

    styler = table_df.style
    status_subset = [c for c in ['Flag Status', 'Action Directive', 'Supply Absorption'] if c in table_df.columns]
    pct_subset = [c for c in ['change', 'Risk %', 'result_drift', 'eps_yoy', 'rev_yoy', 'surprise', 'rev_surprise', 'perf_w', 'perf_1m'] if c in table_df.columns]

    if hasattr(styler, 'map'):
        styled_table = styler.map(style_flag_status, subset=status_subset)
        if pct_subset:
            styled_table = styled_table.map(color_pct, subset=pct_subset)
    else:
        styled_table = styler.applymap(style_flag_status, subset=status_subset)
        if pct_subset:
            styled_table = styled_table.applymap(color_pct, subset=pct_subset)

    curr_sym = "$" if is_us else "₹"
    format_dict = {
        'close': f'{curr_sym}%.2f',
        'Flag Pivot': f'{curr_sym}%.2f',
        'Stop Loss': f'{curr_sym}%.2f',
        'Risk %': '{:+.1f}%',
        'change': '{:+.2f}%',
        'result_drift': '{:+.1f}%',
        'eps_yoy': '{:+.1f}%',
        'rev_yoy': '{:+.1f}%',
        'surprise': '{:+.1f}%',
        'rev_surprise': '{:+.1f}%',
        'perf_w': '{:+.1f}%',
        'perf_1m': '{:+.1f}%',
        'trading_days_ago': '{:d}D'
    }
    valid_format_dict = {k: v for k, v in format_dict.items() if k in table_df.columns}
    styled_table = styled_table.format(valid_format_dict, na_rep="-")

    st.dataframe(
        styled_table,
        column_config={
            "TickerLink": st.column_config.LinkColumn(
                "Symbol",
                display_text=r"symbol=(?:NSE:)?(.*)",
                width="small",
                help="Open institutional chart on TradingView"
            ),
            "Action Directive": st.column_config.TextColumn("Tactical Action", width="medium", help="Imperative execution directive for today"),
            "Flag Status": st.column_config.TextColumn("Flag Status", width="medium"),
            "close": st.column_config.TextColumn(f"Price ({curr_sym})"),
            "ADTV (20)": st.column_config.TextColumn("ADTV (20)", width="small", help="20-Day Average Daily Traded Value"),
            "Flag Pivot": st.column_config.TextColumn(f"Flag Pivot ({curr_sym})", help="Key flag resistance price level where breakout triggers"),
            "Stop Loss": st.column_config.TextColumn(f"Stop-Loss ({curr_sym})", help="Key structural invalidation level (Gap Day Low / 10 EMA cushion)"),
            "Risk %": st.column_config.TextColumn("Risk %", help="Distance to suggested structural stop-loss"),
            "change": st.column_config.TextColumn("Today %"),
            "Supply Absorption": st.column_config.TextColumn("Supply State", width="small", help="Volume dry-up (<0.40x) vs expansion"),
            "trading_days_ago": st.column_config.TextColumn("Sessions Ago"),
            "result_drift": st.column_config.TextColumn("Result Drift %", help="Cumulative price drift % measured from earnings announcement session (Day 0 close) to current close"),
            "eps_yoy": st.column_config.TextColumn("YoY EPS %", width="small", help="Quarterly YoY Diluted EPS Growth % (CANSLIM Rule 'C' requires ≥ +25%)"),
            "rev_yoy": st.column_config.TextColumn("YoY Sales %", width="small", help="Quarterly YoY Sales / Revenue Growth % (CANSLIM requires ≥ +20%)"),
            "surprise": st.column_config.TextColumn("EPS Surprise", width="small", help="Consensus analyst EPS beat/miss %"),
            "rev_surprise": st.column_config.TextColumn("Rev Surprise", width="small", help="Consensus analyst revenue beat/miss %"),
            "perf_w": st.column_config.TextColumn("1W Drift"),
            "perf_1m": st.column_config.TextColumn("1M Drift"),
            "company": st.column_config.TextColumn("Company Name", width="medium"),
            "industry": st.column_config.TextColumn("Industry / Sector", width="medium")
        },
        hide_index=True,
        use_container_width=True,
        height=550
    )


# -----------------------------------------------------------------------------
# MAIN APP ORCHESTRATOR
# -----------------------------------------------------------------------------
def main():
    st.markdown("### ⚡ Earnings Intelligence & PEAD Hub")
    st.caption("Live corporate earnings calendar, consensus surprises, and post-earnings announcement drift (PEAD) momentum scanner.")

    tab_live_cal, tab_stalker, tab_csv_scanner = st.tabs([
        "📅 Live Corporate Earnings Calendar",
        "🚩 25-Day Post-Earnings Flag Stalker",
        "📁 Historical PEAD CSV Drift Scanner"
    ])

    with tab_live_cal:
        render_corporate_earnings_calendar()

    with tab_stalker:
        render_25d_pead_flag_stalker()

    with tab_csv_scanner:
        render_pead_csv_scanner()


if __name__ == "__main__":
    main()

