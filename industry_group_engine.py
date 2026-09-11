"""
CANSLIM Industry Group Momentum Matrix Engine
=============================================
Calculates multi-timeframe relative strength rankings, rank velocity deltas,
pack hunting breadth, and actionable CANSLIM execution badges for 100+ granular
sub-industries and curated Indian Alpha thematic clusters.
"""

import os
import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kush_tracker.db")
MATRIX_PKL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_prices_matrix.pkl")
MAP_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "industry_map.json")

# =============================================================================
# CURATED INDIAN ALPHA THEMATIC CLUSTERS
# High-conviction growth themes dominating the current Indian capex & bull cycle
# =============================================================================
INDIAN_ALPHA_THEMES = {
    "⚡ Power T&D & Transformers": [
        "TRIL.NS", "VOLTAMP.NS", "APARINDS.NS", "SCHNEIDER.NS", "BHEL.NS",
        "GEPIL.NS", "POWERINDIA.NS", "KEC.NS", "KALPATPOWR.NS", "TRANSFORME.BO"
    ],
    "🛡️ Defense & Aerospace Systems": [
        "BEL.NS", "HAL.NS", "DATAPATTNS.NS", "ZENTEC.NS", "ASTRAMICRO.NS",
        "PARAS.NS", "SOLARINDS.NS", "BDL.NS", "COCHINSHIP.NS", "MAZDOCK.NS",
        "GRSE.NS", "SIGMAADV.NS", "SWANDEF.NS"
    ],
    "☀️ Solar, Wind & Renewable EPC": [
        "WAAREEENER.NS", "PREMIERENE.NS", "SUZLON.NS", "INOXWIND.NS", "SWENERGY.NS",
        "KPIGREEN.NS", "BORORENEW.NS", "GENSOL.NS", "ADANIGREEN.NS"
    ],
    "🔌 EMS & Electronics Hardware": [
        "DIXON.NS", "KAYNES.NS", "SYRMA.NS", "CYIENTDLM.NS", "AMBER.NS",
        "PGEL.NS", "AVALON.NS", "CENTRUM.NS", "DCXINDIA.NS"
    ],
    "🚆 Railways & Transit Infra": [
        "TITAGARH.NS", "JWL.NS", "RVNL.NS", "IRFC.NS", "TEXRAIL.NS",
        "RITES.NS", "RAILTEL.NS", "CONCOR.NS", "BEML.NS", "IRCTC.NS"
    ],
    "📈 Capital Markets & Wealth Platforms": [
        "BSE.NS", "MCX.NS", "CDSL.NS", "CAMS.NS", "ANGELONE.NS",
        "360ONE.NS", "MOTILALOFS.NS", "NUVAMA.NS", "GEOJITFSL.NS", "ANANDRATHI.NS", "KFINTECH.NS"
    ],
    "🔌 Wires, Cables & EPC": [
        "POLYCAB.NS", "RRKABEL.NS", "KEI.NS", "FINCABLES.NS", "STERTOOLS.NS", "DYNAMIC.NS", "APARINDS.NS"
    ],
    "🏥 Hospitals & Specialty Healthcare": [
        "APOLLOHOSP.NS", "MAXHEALTH.NS", "FORTIS.NS", "RAINBOW.NS", "NH.NS",
        "KIMS.NS", "MEDANTA.NS", "YATHARTH.NS", "ASTERDM.NS"
    ],
    "💎 Retail Luxury, Gold & Jewelry": [
        "TITAN.NS", "KALYANKJIL.NS", "SENCO.NS", "TBZ.NS", "PNGSREVA.NS", "THANGAMAYL.NS", "ETHOSLTD.NS"
    ],
    "🏗️ Real Estate Developers & Infra": [
        "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "MACROTECH.NS", "PRESTIGE.NS",
        "BRIGADE.NS", "SOBHA.NS", "PURVA.NS"
    ],
    "🧪 Specialty Chemicals & Agro": [
        "PIIND.NS", "DEEPAKNTR.NS", "NAVINFLUOR.NS", "AARTIIND.NS", "TATACHEM.NS",
        "SRF.NS", "FINEORG.NS", "NEOGEN.NS", "COROMANDEL.NS", "SUMICHEM.NS"
    ],
    "🏨 Hotels & Premium Hospitality": [
        "INDHOTEL.NS", "EIHOTEL.NS", "CHALET.NS", "LEMONONTRE.NS", "TAJGVK.NS", "ORIENTHOT.NS"
    ],
    "💻 IT Midcap Growth & Digital": [
        "PERSISTENT.NS", "COFORGE.NS", "KPITTECH.NS", "LTTS.NS", "TATAELXSI.NS",
        "CYIENT.NS", "BSOFT.NS", "SONATSOFTW.NS", "NETWEB.NS"
    ],
    "📦 Logistics, Ports & Cold Chain": [
        "ADANIPORTS.NS", "ALLCARGO.NS", "MAHLOG.NS", "TCIEXP.NS", "BLUEDART.NS",
        "GATEWAY.NS", "VRL.NS", "DELHIVERY.NS", "SCI.NS"
    ],
    "🏭 Auto Ancillaries & Precision Machining": [
        "BHARATFORG.NS", "SONACOMS.NS", "CIEINDIA.NS", "CRAFTSMAN.NS", "SUPRAJIT.NS",
        "RAMKRISNA.NS", "UNOMINDA.NS", "MOTHERSON.NS"
    ],
    "💰 High-Growth NBFCs & Retail Lenders": [
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS", "MUTHOOTFIN.NS",
        "MANAPPURAM.NS", "POONAWALLA.NS"
    ],
    "🌾 Sugar, Distilleries & Biofuels": [
        "BALRAMCHIN.NS", "PRAJSIND.NS", "TRIVENI.NS", "EIDPARRY.NS", "DWARKESH.NS", "RENUKA.NS"
    ],
    "🧱 Cement & Building Materials": [
        "ULTRACEMCO.NS", "AMBUJACEM.NS", "SHREECEM.NS", "DALBHARAT.NS", "JKCEMENT.NS",
        "SAGCEM.NS", "RAMCOCEM.NS"
    ],
    "🔋 Battery & EV Ecosystem": [
        "EXIDEIND.NS", "AMARAJABAT.NS", "TATACHEM.NS", "HBLPOWER.NS", "SERVOTECH.NS",
        "OLECTRA.NS", "JBMMA.NS"
    ],
    "🚢 Shipbuilders & Heavy Marine": [
        "COCHINSHIP.NS", "MAZDOCK.NS", "GRSE.NS"
    ],
    "💧 Water Pipes & Infrastructure": [
        "VASTECH.NS", "ASTRAL.NS", "SUPREMEIND.NS", "PRINCEPIPE.NS", "JISLJALEQS.NS",
        "FINPIPE.NS", "SHAKTIPUMP.NS"
    ],
    "🧬 CDMO & Diagnostics": [
        "DIVISLAB.NS", "SUVENPHAR.NS", "LAURUSLABS.NS", "LALPATHLAB.NS", "METROPOLIS.NS",
        "VIJAYA.NS", "SYNGENE.NS"
    ],
    "🏬 Consumer Durables & Appliances": [
        "VOLTAS.NS", "BLUESTARCO.NS", "HAVELLS.NS", "CROMPTON.NS", "WHIRLPOOL.NS", "SYMPHONY.NS"
    ]
}

@st.cache_data(ttl=3600, show_spinner=False)
def load_price_history_matrix():
    """
    Loads historical_prices_matrix.pkl and returns close_df and high_df.
    Cached for fast sub-second repeated calls.
    """
    if not os.path.exists(MATRIX_PKL_PATH):
        return pd.DataFrame(), pd.DataFrame()
        
    with open(MATRIX_PKL_PATH, "rb") as f:
        matrix = pickle.load(f)
        
    if isinstance(matrix.columns, pd.MultiIndex):
        close_df = matrix.xs("Close", level=1, axis=1)
        high_df = matrix.xs("High", level=1, axis=1) if "High" in matrix.columns.levels[1] else close_df
    else:
        close_df = matrix
        high_df = matrix
        
    return close_df, high_df

@st.cache_data(ttl=3600, show_spinner=False)
def get_group_constituents_map(taxonomy: str = "canonical"):
    """
    Returns a mapping of group_name -> list of tickers.
    - taxonomy='canonical': Granular 145 sub-industries from industry_map.json / database.
    - taxonomy='thematic': Curated Indian Alpha Thematic Clusters.
    """
    if taxonomy == "thematic":
        return INDIAN_ALPHA_THEMES
        
    group_map = {}
    if os.path.exists(MAP_JSON_PATH):
        try:
            with open(MAP_JSON_PATH, "r", encoding="utf-8") as f:
                ind_map = json.load(f)
            for t, ind in ind_map.items():
                if ind and ind != 'Unknown':
                    group_map.setdefault(ind, []).append(t)
            return {k: v for k, v in group_map.items() if len(v) >= 3}
        except Exception:
            pass

    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        try:
            df_fund = pd.read_sql_query(
                "SELECT ticker, industry FROM fundamentals_cache WHERE industry IS NOT NULL AND industry != '' AND industry != 'Unknown'",
                conn
            )
            for ind, sub_df in df_fund.groupby("industry"):
                t_list = sub_df["ticker"].dropna().tolist()
                if len(t_list) >= 3:
                    group_map[ind] = t_list
            return group_map
        finally:
            conn.close()
            
    try:
        from database import get_all_fundamentals_cache
        cache = get_all_fundamentals_cache()
        for t, d in cache.items():
            ind = d.get('industry')
            if ind and ind != 'Unknown':
                sym = t if t.endswith('.NS') or t.endswith('.BO') else t + '.NS'
                group_map.setdefault(ind, []).append(sym)
        return {k: v for k, v in group_map.items() if len(v) >= 3}
    except Exception:
        pass
        
    return group_map

def classify_stock_action_state(dist_pivot: float, dist_ema: float) -> tuple[str, str, int]:
    """
    Assigns CANSLIM execution status badge and priority rank:
    - 🟢 IN BUY ZONE (0 to +5.5% from pivot): 1
    - 🔵 RETEST (-2.0% to 0% from pivot): 2
    - ⏳ COILING (-5.0% to -2.0% from pivot): 3
    - 🟠 EXTENDED (>5.5% above pivot): 4
    - ⏳ FORMING BASE (rest): 5
    - 🔴 FAILED (< 21 EMA or > -8% below pivot): 6
    """
    if np.isnan(dist_pivot) or np.isnan(dist_ema):
        return "⏳ BASE", "base", 5
    if dist_ema < -2.0 or dist_pivot < -8.0:
        return "🔴 FAILED", "failed", 6
    elif dist_pivot > 5.5:
        return "🟠 EXTENDED", "extended", 4
    elif -2.0 <= dist_pivot < 0.0:
        return "🔵 RETEST", "retest", 2
    elif 0.0 <= dist_pivot <= 5.5:
        return "🟢 IN BUY ZONE", "buy_zone", 1
    elif -5.0 <= dist_pivot < -2.0:
        return "⏳ COILING", "coiling", 3
    else:
        return "⏳ BASE", "base", 5

def classify_rotation_status(cur_rank: int, delta_1m: int) -> tuple[str, str]:
    """
    Classifies the institutional rotation velocity of an industry group:
    - 🚀 Surging: Top 20 rank AND delta_1m >= +5
    - 🔄 Accumulating: Rank 21-70 AND delta_1m >= +15
    - ⏸️ Consolidating: Top 25 rank AND -5 <= delta_1m <= +5
    - ⚠️ Distributing: delta_1m <= -15
    - 💤 Lagging: Rank > 70
    """
    if cur_rank <= 20 and delta_1m >= 5:
        return "🚀 Surging", "surging"
    elif 21 <= cur_rank <= 70 and delta_1m >= 15:
        return "🔄 Accumulating", "accumulating"
    elif cur_rank <= 25 and -5 <= delta_1m <= 5:
        return "⏸️ Consolidating", "consolidating"
    elif delta_1m <= -15:
        return "⚠️ Distributing", "distributing"
    elif cur_rank > 70:
        return "💤 Lagging", "lagging"
    else:
        return "⚪ Neutral", "neutral"

@st.cache_data(ttl=600, show_spinner=False)
def compute_industry_group_matrix(taxonomy: str = "canonical", universe_scope: str = "India (NSE/BSE)"):
    """
    Computes complete industry group relative strength matrix across 5 time horizons:
    Today, 1W, 1M, 3M, 6M.
    Returns:
      - df_matrix: Processed DataFrame ready for rendering
      - benchmark_curve: Series of benchmark index
    """
    close_df, high_df = load_price_history_matrix()
    if close_df.empty:
        return pd.DataFrame(), pd.Series()
        
    if universe_scope.startswith("India"):
        valid_cols = [c for c in close_df.columns if c.endswith(".NS") or c.endswith(".BO")]
        close_df = close_df[valid_cols]
        high_df = high_df[[c for c in valid_cols if c in high_df.columns]]
        
    group_map = get_group_constituents_map(taxonomy=taxonomy)
    if not group_map:
        return pd.DataFrame(), pd.Series()
        
    n_bars = len(close_df)
    c0 = close_df.iloc[-1]
    idx_1y = max(0, n_bars - 250)
    idx_6m = max(0, n_bars - 126)
    idx_3m = max(0, n_bars - 63)
    
    c_1y = close_df.iloc[idx_1y]
    c_6m = close_df.iloc[idx_6m]
    c_3m = close_df.iloc[idx_3m]
    
    r3m = (c0 / c_3m - 1) * 100
    r6m = (c0 / c_6m - 1) * 100
    r1y = (c0 / c_1y - 1) * 100
    
    composite_stock = 0.4 * r3m.fillna(0) + 0.3 * r6m.fillna(0) + 0.3 * r1y.fillna(0)
    rs_percentile = (composite_stock.rank(pct=True) * 99).fillna(0).round(0).astype(int)
    
    pivot_25 = high_df.iloc[-26:-1].max() if len(high_df) >= 27 else high_df.max()
    ema_21 = close_df.ewm(span=21, adjust=False).mean().iloc[-1]
    
    dist_pivot_s = ((c0 - pivot_25) / pivot_25) * 100
    dist_ema_s = ((c0 - ema_21) / ema_21) * 100
    
    stock_status_dict = {}
    for t in close_df.columns:
        p_dist = dist_pivot_s.get(t, np.nan)
        e_dist = dist_ema_s.get(t, np.nan)
        lbl, b_style, prio = classify_stock_action_state(p_dist, e_dist)
        stock_status_dict[t] = {
            'status': lbl,
            'style': b_style,
            'priority': prio,
            'cmp': round(float(c0.get(t, 0)), 2),
            'dist_pivot': round(float(p_dist), 1) if not np.isnan(p_dist) else 0.0,
            'dist_ema': round(float(e_dist), 1) if not np.isnan(e_dist) else 0.0,
            'rs': int(rs_percentile.get(t, 0))
        }

    curves = {}
    group_clean_constituents = {}
    
    for grp, tickers in group_map.items():
        avail = [t for t in tickers if t in close_df.columns]
        if len(avail) < (2 if taxonomy == "thematic" else 3):
            continue
        sub = close_df[avail].dropna(how='all')
        if sub.empty or len(sub) < 130:
            continue
        norm = sub.div(sub.bfill().iloc[0], axis=1)
        mean_s = norm.mean(axis=1).dropna()
        if len(mean_s) >= 130:
            curves[grp] = mean_s
            group_clean_constituents[grp] = avail

    curves_df = pd.DataFrame(curves).dropna(how='all')
    if curves_df.empty or len(curves_df) < 130:
        return pd.DataFrame(), pd.Series()
        
    benchmark_curve = curves_df.mean(axis=1)
    
    def get_score_series(df, offset=0):
        end_idx = len(df) - 1 - offset
        if end_idx < 126:
            return pd.Series(index=df.columns, dtype=float)
        c_cur = df.iloc[end_idx]
        c_w1 = df.iloc[end_idx - 5]
        c_m1 = df.iloc[end_idx - 21]
        c_m3 = df.iloc[end_idx - 63]
        c_m6 = df.iloc[end_idx - 126]
        
        r_w1 = (c_cur / c_w1 - 1) * 100
        r_m1 = (c_cur / c_m1 - 1) * 100
        r_m3 = (c_cur / c_m3 - 1) * 100
        r_m6 = (c_cur / c_m6 - 1) * 100
        
        return 0.4 * r_m3 + 0.3 * r_m6 + 0.2 * r_m1 + 0.1 * r_w1

    score_today = get_score_series(curves_df, 0).fillna(-999.0)
    score_1w = get_score_series(curves_df, 5).fillna(-999.0)
    score_1m = get_score_series(curves_df, 21).fillna(-999.0)
    score_3m = get_score_series(curves_df, 63).fillna(-999.0)
    score_6m = get_score_series(curves_df, 126).fillna(-999.0)
    
    ranks_today = score_today.rank(ascending=False, method='min').fillna(99).astype(int)
    ranks_1w = score_1w.rank(ascending=False, method='min').fillna(99).astype(int)
    ranks_1m = score_1m.rank(ascending=False, method='min').fillna(99).astype(int)
    ranks_3m = score_3m.rank(ascending=False, method='min').fillna(99).astype(int)
    ranks_6m = score_6m.rank(ascending=False, method='min').fillna(99).astype(int)
    
    composite_rs_pct = (score_today.rank(pct=True) * 99).fillna(0).round(0).astype(int)

    rows = []
    end_idx = len(curves_df) - 1
    
    for grp in curves_df.columns:
        c_series = curves_df[grp]
        c_now = c_series.iloc[-1]
        
        r_1d = round(float((c_now / c_series.iloc[-2] - 1) * 100), 2) if len(c_series) > 1 else 0.0
        r_1w = round(float((c_now / c_series.iloc[-6] - 1) * 100), 2) if len(c_series) > 5 else 0.0
        r_1m = round(float((c_now / c_series.iloc[-22] - 1) * 100), 2) if len(c_series) > 21 else 0.0
        r_3m = round(float((c_now / c_series.iloc[-64] - 1) * 100), 2) if len(c_series) > 63 else 0.0
        r_6m = round(float((c_now / c_series.iloc[-127] - 1) * 100), 2) if len(c_series) > 126 else 0.0
        
        rank_td = int(ranks_today[grp])
        rank_w1 = int(ranks_1w[grp])
        rank_m1 = int(ranks_1m[grp])
        rank_m3 = int(ranks_3m[grp])
        rank_m6 = int(ranks_6m[grp])
        
        delta_1w = rank_w1 - rank_td
        delta_1m = rank_m1 - rank_td
        
        rot_status, rot_style = classify_rotation_status(rank_td, delta_1m)
        
        constits = group_clean_constituents.get(grp, [])
        pack_count = sum(1 for t in constits if stock_status_dict.get(t, {}).get('rs', 0) >= 80)
        
        sorted_constits = sorted(
            constits,
            key=lambda t: stock_status_dict.get(t, {}).get('rs', 0),
            reverse=True
        )
        top_3 = []
        for t in sorted_constits[:3]:
            st_info = stock_status_dict.get(t, {})
            clean_t = t.replace('.NS', '').replace('.BO', '')
            dist_p = st_info.get('dist_pivot', 0.0)
            sign = "+" if dist_p > 0 else ""
            dist_str = f"{sign}{dist_p:.1f}%"
            
            top_3.append({
                'ticker': clean_t,
                'raw_ticker': t,
                'rs': st_info.get('rs', 0),
                'cmp': st_info.get('cmp', 0.0),
                'dist_pivot': dist_p,
                'dist_pivot_str': dist_str,
                'dist_ema': st_info.get('dist_ema', 0.0),
                'status': st_info.get('status', '⏳ BASE'),
                'style': st_info.get('style', 'base')
            })
            
        spark_vals = c_series.iloc[-21:]
        if len(spark_vals) > 0 and spark_vals.iloc[0] > 0:
            spark_norm = (spark_vals / spark_vals.iloc[0] * 100).round(1).tolist()
        else:
            spark_norm = []
            
        leader_badges = []
        for ldr in top_3:
            st_lbl = ldr['status']
            if "IN BUY ZONE" in st_lbl:
                badge_str = f"{ldr['ticker']} [🟢 BUY {ldr['dist_pivot_str']}]"
            elif "RETEST" in st_lbl:
                badge_str = f"{ldr['ticker']} [🔵 RETEST {ldr['dist_pivot_str']}]"
            elif "COILING" in st_lbl:
                badge_str = f"{ldr['ticker']} [⏳ COIL {ldr['dist_pivot_str']}]"
            elif "EXTENDED" in st_lbl:
                badge_str = f"{ldr['ticker']} [🟠 EXT {ldr['dist_pivot_str']}]"
            elif "FAILED" in st_lbl:
                badge_str = f"{ldr['ticker']} [🔴 FAIL]"
            else:
                badge_str = f"{ldr['ticker']} [⏳ BASE]"
            leader_badges.append(badge_str)
            
        leader_display = "  •  ".join(leader_badges)
        
        rows.append({
            'Industry_Group': grp,
            'Rank_Today': rank_td,
            'Rank_1W': rank_w1,
            'Rank_1M': rank_m1,
            'Rank_3M': rank_m3,
            'Rank_6M': rank_m6,
            'Delta_1W': delta_1w,
            'Delta_1M': delta_1m,
            'Rotation_Status': rot_status,
            'Rotation_Style': rot_style,
            'Comp_RS': int(composite_rs_pct[grp]),
            'Pack_Hunting_Count': pack_count,
            'Stock_Count': len(constits),
            'Return_1D': r_1d,
            'Return_1W': r_1w,
            'Return_1M': r_1m,
            'Return_3M': r_3m,
            'Return_6M': r_6m,
            'Top_Leaders_Display': leader_display,
            'Top_3_Leaders': top_3,
            'Sparkline_1M': spark_norm,
            'Constituents': constits
        })
        
    df_result = pd.DataFrame(rows).sort_values('Rank_Today', ascending=True).reset_index(drop=True)
    return df_result, benchmark_curve

def get_group_deep_dive_data(group_name: str, taxonomy: str = "canonical"):
    """
    Extracts deep-dive constituent table and historical comparison curve for a chosen group.
    """
    close_df, high_df = load_price_history_matrix()
    if close_df.empty:
        return pd.DataFrame(), pd.Series(), ""
        
    group_map = get_group_constituents_map(taxonomy=taxonomy)
    tickers = group_map.get(group_name, [])
    avail = [t for t in tickers if t in close_df.columns]
    if not avail:
        return pd.DataFrame(), pd.Series(), ""
        
    sub_close = close_df[avail]
    sub_high = high_df[[t for t in avail if t in high_df.columns]]
    
    n_bars = len(close_df)
    c0 = sub_close.iloc[-1]
    c_1d = sub_close.iloc[-2] if n_bars > 1 else c0
    c_1w = sub_close.iloc[-6] if n_bars > 5 else c0
    c_1m = sub_close.iloc[-22] if n_bars > 21 else c0
    c_3m = sub_close.iloc[-64] if n_bars > 63 else c0
    c_1y = sub_close.iloc[max(0, n_bars - 250)]
    
    pivot_25 = sub_high.iloc[-26:-1].max() if len(sub_high) >= 27 else sub_high.max()
    ema_21 = sub_close.ewm(span=21, adjust=False).mean().iloc[-1]
    
    r1d = ((c0 / c_1d - 1) * 100).round(2)
    r1w = ((c0 / c_1w - 1) * 100).round(2)
    r1m = ((c0 / c_1m - 1) * 100).round(2)
    r3m = ((c0 / c_3m - 1) * 100).round(2)
    r1y = ((c0 / c_1y - 1) * 100).round(2)
    
    comp = 0.4 * r3m.fillna(0) + 0.3 * ((c0 / sub_close.iloc[max(0, n_bars - 126)] - 1) * 100).fillna(0) + 0.3 * r1y.fillna(0)
    rs_pct = (comp.rank(pct=True) * 99).fillna(0).round(0).astype(int)
    
    dist_piv = (((c0 - pivot_25) / pivot_25) * 100).round(2)
    dist_ema = (((c0 - ema_21) / ema_21) * 100).round(2)
    
    constit_rows = []
    tv_tickers = []
    
    for t in avail:
        clean_t = t.replace('.NS', '').replace('.BO', '')
        exchange = 'BSE' if t.endswith('.BO') else 'NSE'
        tv_tickers.append(f"{exchange}:{clean_t}")
        
        p_dist = float(dist_piv.get(t, 0.0))
        e_dist = float(dist_ema.get(t, 0.0))
        status_lbl, badge_style, prio = classify_stock_action_state(p_dist, e_dist)
        
        constit_rows.append({
            'Symbol': clean_t,
            'Raw_Ticker': t,
            'CMP (₹)': float(c0.get(t, 0.0)),
            '1D %': float(r1d.get(t, 0.0)),
            '1W %': float(r1w.get(t, 0.0)),
            '1M %': float(r1m.get(t, 0.0)),
            '3M %': float(r3m.get(t, 0.0)),
            '1Y %': float(r1y.get(t, 0.0)),
            'RS Rating': int(rs_pct.get(t, 0)),
            'Pivot Price': round(float(pivot_25.get(t, 0.0)), 2),
            'Dist Pivot %': p_dist,
            'Dist 21 EMA %': e_dist,
            'Execution Status': status_lbl,
            'Status_Style': badge_style,
            'Status_Priority': prio,
            'TradingView_URL': f"https://www.tradingview.com/chart/?symbol={exchange}%3A{clean_t}"
        })
        
    df_constits = pd.DataFrame(constit_rows).sort_values(['Status_Priority', 'RS Rating'], ascending=[True, False]).reset_index(drop=True)
    
    norm_group = sub_close.div(sub_close.bfill().iloc[0], axis=1).mean(axis=1)
    norm_bench = close_df.div(close_df.bfill().iloc[0], axis=1).mean(axis=1)
    
    curve_df = pd.DataFrame({
        'Industry Group': norm_group.iloc[-250:],
        'Universe Benchmark': norm_bench.iloc[-250:]
    })
    
    tv_copy_box = ",".join(tv_tickers)
    return df_constits, curve_df, tv_copy_box