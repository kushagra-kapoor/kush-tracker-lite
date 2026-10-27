"""
Market Execution Persona Classifier Engine
Quantitatively distinguishes between:
  - Secular Trend-Persistence Regime: Positional Trading (Hold for Monster Runs, pyramid, trail 50 SMA / 21 EMA)
  - High-Velocity Churn Regime: Swing Trading (Hit and run, sell into strength at +12% to +18%, trail tight 10 EMA)

Based on quantitative frameworks from:
  - John Boik: "Monster Stocks" & "Lessons from the Greatest Stock Traders of All Time"
  - Mark Minervini: "Think and Trade Like a Champion" & "Trade Like a Stock Market Wizard"
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import config
import database
from database import (
    get_connection,
    save_market_persona,
    get_latest_market_persona,
    get_market_persona_history
)


# =============================================================================
# PILLAR 1: SECTOR LEADERSHIP TENURE / PERSISTENCE INDEX (WEIGHT: 30%)
# =============================================================================

def compute_sector_tenure(market: str = 'INDIA', lookback_sessions: int = 40) -> dict:
    """
    Computes Sector Leadership Tenure:
    Finds the Top 5 industry groups today by RS and checks how many were in the
    Top 5 exactly lookback_sessions ago (~40 trading days / 2 calendar months).

    Dual-source architecture:
      1. Primary: SQLite `sector_leadership_history` table.
      2. Fallback / US: Point-in-time sector RS calculation from `historical_prices_matrix.pkl`.
    """
    conn = get_connection()
    c = conn.cursor()
    try:
        # Check SQLite sector leadership records
        c.execute('''
            SELECT DISTINCT date FROM sector_leadership_history
            ORDER BY date DESC
        ''')
        dates = [r[0] for r in c.fetchall()]
        
        if len(dates) >= 10:
            today_date = dates[0]
            past_idx = min(lookback_sessions, len(dates) - 1)
            past_date = dates[past_idx]
            
            c.execute('''
                SELECT industry, avg_rs FROM sector_leadership_history
                WHERE date = ? ORDER BY rank ASC LIMIT 5
            ''', (today_date,))
            today_top5_raw = c.fetchall()
            
            c.execute('''
                SELECT industry, avg_rs FROM sector_leadership_history
                WHERE date = ? ORDER BY rank ASC LIMIT 5
            ''', (past_date,))
            past_top5_raw = c.fetchall()
            
            today_industries = [r[0] for r in today_top5_raw]
            past_industries = [r[0] for r in past_top5_raw]
            past_set = set(past_industries)
            
            retained_sectors = [ind for ind in today_industries if ind in past_set]
            retained_count = len(retained_sectors)
            tenure_pct = (retained_count / 5.0) * 100.0
            
            # Map tenure to 0-100 score:
            # 5 retained (100%) -> 100
            # 4 retained (80%)  -> 95
            # 3 retained (60%)  -> 80
            # 2 retained (40%)  -> 60
            # 1 retained (20%)  -> 35
            # 0 retained (0%)   -> 15
            score_map = {5: 100.0, 4: 95.0, 3: 80.0, 2: 60.0, 1: 35.0, 0: 15.0}
            pillar_score = score_map.get(retained_count, 50.0)
            
            today_top5_detail = [
                {'industry': r[0], 'avg_rs': round(r[1], 1) if r[1] else 0.0, 'retained': r[0] in past_set}
                for r in today_top5_raw
            ]
            past_top5_detail = [
                {'industry': r[0], 'avg_rs': round(r[1], 1) if r[1] else 0.0}
                for r in past_top5_raw
            ]
            
            return {
                'pillar_score': pillar_score,
                'tenure_pct': round(tenure_pct, 1),
                'retained_count': retained_count,
                'retained_sectors': retained_sectors,
                'today_date': today_date,
                'past_date': past_date,
                'today_top5': today_top5_detail,
                'past_top5': past_top5_detail,
                'source': 'sqlite_history'
            }
    except Exception as e:
        print(f"[MarketPersona] Error querying sector history: {e}")
    finally:
        conn.close()
        
    # Fallback / Matrix calculation
    return _compute_sector_tenure_from_matrix(market, lookback_sessions)


def _compute_sector_tenure_from_matrix(market: str = 'INDIA', lookback_sessions: int = 40) -> dict:
    """Fallback calculation using historical prices matrix and industry mapping."""
    matrix_path = 'historical_prices_matrix.pkl' if os.path.exists('historical_prices_matrix.pkl') else 'breadth_matrix.pkl'
    if not os.path.exists(matrix_path):
        return {
            'pillar_score': 50.0,
            'tenure_pct': 40.0,
            'retained_count': 2,
            'retained_sectors': [],
            'today_date': datetime.now().strftime('%Y-%m-%d'),
            'past_date': '',
            'today_top5': [],
            'past_top5': [],
            'source': 'default_fallback'
        }
        
    try:
        df = pd.read_pickle(matrix_path)
        closes = df.xs('Close', level=1, axis=1) if isinstance(df.columns, pd.MultiIndex) else df
        
        # Load industry mapping
        industry_map = {}
        if market.upper() in ['INDIA', 'IN']:
            if os.path.exists('industry_map.json'):
                with open('industry_map.json', 'r', encoding='utf-8') as f:
                    industry_map = json.load(f)
            in_cols = [c for c in closes.columns if str(c).endswith('.NS')]
            closes = closes[in_cols]
        else:
            if os.path.exists('data/sp500_gics_map.json'):
                with open('data/sp500_gics_map.json', 'r', encoding='utf-8') as f:
                    industry_map = json.load(f)
            us_cols = [c for c in closes.columns if not str(c).endswith('.NS') and not str(c).endswith('.BO')]
            closes = closes[us_cols]
            
        if len(closes) < lookback_sessions + 65:
            return {'pillar_score': 50.0, 'tenure_pct': 40.0, 'retained_count': 2, 'source': 'insufficient_history'}
            
        # Calculate RS momentum at T and T-lookback_sessions
        # RS = 0.40 * 1M_ret + 0.60 * 3M_ret
        ret_1m = closes.pct_change(21)
        ret_3m = closes.pct_change(63)
        rs_matrix = (ret_1m * 0.40) + (ret_3m * 0.60)
        
        idx_now = len(closes) - 1
        idx_past = max(0, idx_now - lookback_sessions)
        
        date_now = closes.index[idx_now].strftime('%Y-%m-%d')
        date_past = closes.index[idx_past].strftime('%Y-%m-%d')
        
        rs_now = rs_matrix.iloc[idx_now].dropna()
        rs_past = rs_matrix.iloc[idx_past].dropna()
        
        # Aggregate by industry
        def get_top5_industries(rs_series):
            sec_scores = {}
            sec_counts = {}
            for t, val in rs_series.items():
                clean_t = str(t).replace('.NS', '')
                ind = industry_map.get(clean_t) or industry_map.get(t)
                if ind and ind != 'Unknown':
                    sec_scores[ind] = sec_scores.get(ind, 0.0) + val
                    sec_counts[ind] = sec_counts.get(ind, 0) + 1
            avg_sec = {ind: sec_scores[ind] / sec_counts[ind] for ind in sec_scores if sec_counts[ind] >= 3}
            sorted_sec = sorted(avg_sec.items(), key=lambda x: x[1], reverse=True)[:5]
            return sorted_sec

        top5_now = get_top5_industries(rs_now)
        top5_past = get_top5_industries(rs_past)
        
        now_names = [s[0] for s in top5_now]
        past_names = [s[0] for s in top5_past]
        past_set = set(past_names)
        
        retained = [ind for ind in now_names if ind in past_set]
        retained_count = len(retained)
        tenure_pct = (retained_count / 5.0) * 100.0 if top5_now else 40.0
        
        score_map = {5: 100.0, 4: 95.0, 3: 80.0, 2: 60.0, 1: 35.0, 0: 15.0}
        pillar_score = score_map.get(retained_count, 50.0)
        
        return {
            'pillar_score': pillar_score,
            'tenure_pct': round(tenure_pct, 1),
            'retained_count': retained_count,
            'retained_sectors': retained,
            'today_date': date_now,
            'past_date': date_past,
            'today_top5': [{'industry': s[0], 'avg_rs': round(s[1] * 100, 1), 'retained': s[0] in past_set} for s in top5_now],
            'past_top5': [{'industry': s[0], 'avg_rs': round(s[1] * 100, 1)} for s in top5_past],
            'source': 'matrix_point_in_time'
        }
    except Exception as e:
        print(f"[MarketPersona] Error computing matrix sector tenure: {e}")
        return {'pillar_score': 50.0, 'tenure_pct': 40.0, 'retained_count': 2, 'source': 'error_fallback'}


# =============================================================================
# PILLAR 2: STAGE 2 BREAKOUT T+10 FOLLOW-THROUGH HEALTH (WEIGHT: 30%)
# =============================================================================

def compute_breakout_health(market: str = 'INDIA', lookback_window: int = 20) -> dict:
    """
    Computes Stage 2 Breakout T+10 Follow-Through Health:
    1. Scans qualified Stage 2 base breakouts occurring between T-35 and T-10
       (giving every breakout exactly 10 sessions of forward evaluation).
    2. Measures return at T+10 relative to breakout pivot price:
       - Success / Continuation: Return >= +4.0%
       - Squat / Failure: Return < 0.0% (slipped back below pivot into base)
       - Choppy: 0.0% <= Return < +4.0%
    3. Breakout Health % = (Success Breakouts / Total Breakouts) * 100
    """
    matrix_path = 'historical_prices_matrix.pkl' if os.path.exists('historical_prices_matrix.pkl') else 'breadth_matrix.pkl'
    if not os.path.exists(matrix_path):
        return {
            'pillar_score': 50.0,
            'breakout_health_pct': 50.0,
            'total_breakouts': 0,
            'success_count': 0,
            'squat_count': 0,
            'choppy_count': 0,
            'breakout_list': []
        }
        
    try:
        df = pd.read_pickle(matrix_path)
        closes = df.xs('Close', level=1, axis=1) if isinstance(df.columns, pd.MultiIndex) else df
        highs = df.xs('High', level=1, axis=1) if isinstance(df.columns, pd.MultiIndex) else df
        lows = df.xs('Low', level=1, axis=1) if isinstance(df.columns, pd.MultiIndex) else df
        volumes = df.xs('Volume', level=1, axis=1) if isinstance(df.columns, pd.MultiIndex) else pd.DataFrame()
        
        if market.upper() in ['INDIA', 'IN']:
            valid_cols = [c for c in closes.columns if str(c).endswith('.NS')]
        else:
            valid_cols = [c for c in closes.columns if not str(c).endswith('.NS') and not str(c).endswith('.BO')]
            
        if not valid_cols or len(closes) < 100:
            return {'pillar_score': 50.0, 'breakout_health_pct': 50.0, 'total_breakouts': 0, 'breakout_list': []}
            
        # Select top liquid universe (~600 stocks by dollar volume) for high institutional relevance
        if not volumes.empty:
            dv = (closes[valid_cols].iloc[-20:] * volumes[valid_cols].iloc[-20:]).mean()
            liquid_cols = dv.nlargest(min(600, len(valid_cols))).index
        else:
            liquid_cols = valid_cols[:min(600, len(valid_cols))]
            
        c_sub = closes[liquid_cols]
        h_sub = highs[liquid_cols]
        l_sub = lows[liquid_cols]
        v_sub = volumes[liquid_cols] if not volumes.empty else pd.DataFrame(index=closes.index, columns=liquid_cols).fillna(1.0)
        
        sma50 = c_sub.rolling(50, min_periods=40).mean()
        sma200 = c_sub.rolling(200, min_periods=80).mean()
        vol_sma20 = v_sub.rolling(20, min_periods=15).mean()
        high_252 = h_sub.rolling(252, min_periods=80).max()
        low_252 = l_sub.rolling(252, min_periods=80).min()
        
        N = len(c_sub)
        start_idx = max(50, N - 35)
        end_idx = N - 10
        
        breakouts = []
        
        for t_idx in range(start_idx, end_idx):
            date_t = c_sub.index[t_idx].strftime('%Y-%m-%d')
            date_t10 = c_sub.index[t_idx + 10].strftime('%Y-%m-%d')
            
            # Base consolidation over prior 15-20 days [t_idx-20 : t_idx]
            base_high = h_sub.iloc[t_idx - 20 : t_idx].max()
            base_low = l_sub.iloc[t_idx - 20 : t_idx].min()
            base_depth = (base_high - base_low) / base_high
            
            c_t = c_sub.iloc[t_idx]
            v_t = v_sub.iloc[t_idx]
            s50_t = sma50.iloc[t_idx]
            s200_t = sma200.iloc[t_idx]
            s200_slope = s200_t >= sma200.iloc[max(0, t_idx - 20)] * 0.99
            h252_t = high_252.iloc[t_idx]
            l252_t = low_252.iloc[t_idx]
            v_avg = vol_sma20.iloc[t_idx - 1]
            
            # Minervini Stage 2 criteria + Base + Volume breakout
            is_stage2 = (c_t > s50_t) & (s50_t > s200_t) & s200_slope & (c_t >= 0.75 * h252_t) & (c_t >= 1.25 * l252_t)
            valid_base = (base_depth >= 0.08) & (base_depth <= 0.38)
            vol_expansion = (v_t >= 1.20 * v_avg) if not volumes.empty else pd.Series(True, index=c_sub.columns)
            is_bo = (c_t > base_high) & vol_expansion
            
            qualified = is_stage2 & valid_base & is_bo
            matching = qualified[qualified].index.tolist()
            
            for ticker in matching:
                pivot = float(base_high[ticker])
                p_t10 = float(c_sub[ticker].iloc[t_idx + 10])
                if pivot <= 0 or np.isnan(p_t10):
                    continue
                ret_t10 = ((p_t10 - pivot) / pivot) * 100.0
                status = 'Success' if ret_t10 >= 4.0 else ('Squat' if ret_t10 < 0.0 else 'Choppy')
                clean_t = str(ticker).replace('.NS', '')
                breakouts.append({
                    'date': date_t,
                    'eval_date': date_t10,
                    'ticker': clean_t,
                    'pivot': round(pivot, 2),
                    'price_t10': round(p_t10, 2),
                    'return_t10_pct': round(ret_t10, 2),
                    'status': status
                })
                
        total_bo = len(breakouts)
        if total_bo == 0:
            return {
                'pillar_score': 50.0,
                'breakout_health_pct': 50.0,
                'total_breakouts': 0,
                'success_count': 0,
                'squat_count': 0,
                'choppy_count': 0,
                'breakout_list': []
            }
            
        success_cnt = sum(1 for b in breakouts if b['status'] == 'Success')
        squat_cnt = sum(1 for b in breakouts if b['status'] == 'Squat')
        choppy_cnt = sum(1 for b in breakouts if b['status'] == 'Choppy')
        
        health_pct = (success_cnt / total_bo) * 100.0
        
        # Scale health % (typically ranges 20% to 75%) to 0-100 pillar score:
        # 60%+ success -> 90-100 (Exceptional 2020 follow-through)
        # 45%-59%      -> 65-85  (Balanced)
        # 30%-44%      -> 35-60  (2021 High Churn / Squats)
        # < 30%        -> 15-30  (Severe failure rate)
        pillar_score = float(np.clip(15.0 + (health_pct - 20.0) * (85.0 / 45.0), 10.0, 100.0))
        
        # Sort recent breakouts descending by date
        breakouts.sort(key=lambda x: x['date'], reverse=True)
        
        return {
            'pillar_score': round(pillar_score, 1),
            'breakout_health_pct': round(health_pct, 1),
            'total_breakouts': total_bo,
            'success_count': success_cnt,
            'squat_count': squat_cnt,
            'choppy_count': choppy_cnt,
            'breakout_list': breakouts[:35]  # Top 35 recent audited breakouts for drilldown
        }
    except Exception as e:
        print(f"[MarketPersona] Error computing breakout health: {e}")
        return {
            'pillar_score': 50.0,
            'breakout_health_pct': 50.0,
            'total_breakouts': 0,
            'success_count': 0,
            'squat_count': 0,
            'choppy_count': 0,
            'breakout_list': []
        }


# =============================================================================
# PILLAR 3: SUPER-LEADER 21-EMA ADHERENCE RATE (WEIGHT: 25%)
# =============================================================================

def compute_leader_21ema_adherence(market: str = 'INDIA') -> dict:
    """
    Computes Super-Leader 21-EMA Adherence Rate:
    % of Top 100 RS institutional momentum leaders holding above their 21-day EMA.
    Uses existing `compute_top_rs_breadth_history` from `breadth_engine.py`.
    """
    try:
        from breadth_engine import compute_top_rs_breadth_history
        mkt_code = 'US' if market.upper() in ['US', 'USA'] else 'IN'
        df_rs = compute_top_rs_breadth_history(market=mkt_code, lookback_days=10)
        
        if df_rs is not None and not df_rs.empty:
            latest_row = df_rs.iloc[-1]
            above_21 = float(latest_row.get('top100_above_21_ema', 50.0))
            above_50 = float(latest_row.get('top100_above_50_sma', 50.0))
            lead_spread = float(latest_row.get('leadership_spread', 0.0))
            
            # Leader adherence scoring:
            # >= 75%: 90 - 100 (Strong secular adherence -> Positional)
            # 50% - 74%: 60 - 85 (Normal pullback digestion)
            # < 50%: 20 - 50 (Leaders repeatedly breaking short-term support -> Swing)
            pillar_score = float(np.clip(above_21, 10.0, 100.0))
            
            return {
                'pillar_score': round(pillar_score, 1),
                'leader_21ema_pct': round(above_21, 1),
                'top100_above_21_ema': round(above_21, 1),
                'top100_above_50_sma': round(above_50, 1),
                'leadership_spread': round(lead_spread, 1)
            }
    except Exception as e:
        print(f"[MarketPersona] Error computing leader 21-EMA adherence: {e}")
        
    return {
        'pillar_score': 50.0,
        'leader_21ema_pct': 50.0,
        'top100_above_21_ema': 50.0,
        'top100_above_50_sma': 50.0,
        'leadership_spread': 0.0
    }


# =============================================================================
# PILLAR 4: MEGA-CAP VS BROAD MARKET DIVERGENCE SPREAD (WEIGHT: 15%)
# =============================================================================

def compute_mega_cap_spread(market: str = 'INDIA') -> dict:
    """
    Detects the classic 2021 Mega-Cap Illusion:
    Compares Benchmark Index (Nifty 50 or S&P 500) distance from its 50 SMA
    directly against Broad Market Breadth (% of universe > 50 SMA).
    
    If the Index is propped up while broad breadth is severely depressed (< 40%),
    a major divergence penalty is triggered.
    """
    index_name = "Nifty 50" if market.upper() in ['INDIA', 'IN'] else "S&P 500"
    dist_sma50 = 0.0
    broad_50 = 50.0
    
    # 1. Fetch benchmark telemetry
    try:
        from macro_regime_engine import fetch_nifty50_telemetry, fetch_sp500_telemetry
        if market.upper() in ['INDIA', 'IN']:
            tel = fetch_nifty50_telemetry()
            if tel:
                dist_sma50 = float(tel.get('dist_sma50', 0.0))
        else:
            tel = fetch_sp500_telemetry()
            if tel:
                dist_sma50 = float(tel.get('dist_sma50', 0.0))
    except Exception as e:
        print(f"[MarketPersona] Error fetching benchmark telemetry: {e}")
        
    # 2. Fetch Broad Market Breadth (% > 50 SMA)
    conn = get_connection()
    c = conn.cursor()
    try:
        mkt_code = 'US' if market.upper() in ['US', 'USA'] else 'IN'
        c.execute('''
            SELECT above_50_pct FROM market_breadth_daily
            WHERE market = ? ORDER BY date DESC LIMIT 1
        ''', (mkt_code,))
        row = c.fetchone()
        if row and row[0] is not None:
            broad_50 = float(row[0])
    except Exception as e:
        print(f"[MarketPersona] Error fetching broad breadth: {e}")
    finally:
        conn.close()
        
    # 3. Calculate Divergence Trap & Harmonized Pillar Score
    # Spread = (Broad % > 50 SMA) - (Index Dist from 50 SMA + 50.0)
    # Normalized so positive = broad market leading or healthy; negative = mega-caps propping up decay
    spread_pct = round(broad_50 - (dist_sma50 + 50.0), 1)
    
    # Divergence Trap: Index is positive (> 0%) but broad breadth is low (< 45%)
    is_trap = (dist_sma50 >= 0.0) and (broad_50 < 45.0)
    trap_magnitude = max(0.0, dist_sma50) - max(0.0, broad_50 - 45.0) if is_trap else 0.0
    
    # Base score tracks broad breadth, penalized heavily if trap is active
    base_score = broad_50
    penalty = trap_magnitude * 2.0
    pillar_score = float(np.clip(base_score - penalty, 10.0, 100.0))
    
    verdict = "⚠️ Mega-Cap Illusion Active" if is_trap else ("✅ Broad Market In Sync" if broad_50 >= 50.0 else "Neutral Digestion")
    
    return {
        'pillar_score': round(pillar_score, 1),
        'mega_cap_spread': spread_pct,
        'index_name': index_name,
        'index_dist_50sma': round(dist_sma50, 2),
        'broad_breadth_50sma': round(broad_50, 1),
        'is_divergence_trap': is_trap,
        'trap_magnitude': round(trap_magnitude, 1),
        'verdict': verdict
    }


# =============================================================================
# COMPOSITE MARKET PERSONA CLASSIFIER
# =============================================================================

def calculate_composite_market_persona(market: str = 'INDIA', force_recompute: bool = False) -> dict:
    """
    Computes or retrieves the daily Market Execution Persona:
      Composite Score = 0.30 * P1 + 0.30 * P2 + 0.25 * P3 + 0.15 * P4

    Returns dictionary with:
      - composite_score (0 to 100)
      - persona_label ('🚀 2020 Trend-Persistence', '⚖️ Balanced Regime', '🌪️ 2021 High-Velocity Rotation')
      - trading_mandate ('Positional Trading', 'Hybrid Trading', 'Swing Trading')
      - mandate_summary (Actionable playbook guidelines)
      - pillar details & raw audit records for transparent drilldown
    """
    today_str = datetime.now().strftime('%Y-%m-%d')
    mkt_key = 'INDIA' if market.upper() in ['INDIA', 'IN'] else 'US'
    
    # Check SQLite cache first if not forced
    if not force_recompute:
        latest = get_latest_market_persona(mkt_key)
        if latest and latest.get('date') == today_str and latest.get('composite_score'):
            return latest

    # Compute 4 Pillars
    p1 = compute_sector_tenure(mkt_key, lookback_sessions=40)
    p2 = compute_breakout_health(mkt_key, lookback_window=20)
    p3 = compute_leader_21ema_adherence(mkt_key)
    p4 = compute_mega_cap_spread(mkt_key)
    
    # Calculate Composite Score (0 to 100)
    composite = (
        0.30 * p1['pillar_score'] +
        0.30 * p2['pillar_score'] +
        0.25 * p3['pillar_score'] +
        0.15 * p4['pillar_score']
    )
    composite_score = round(float(np.clip(composite, 0.0, 100.0)), 1)
    
    # Determine Persona Label & Trading Mandate
    if composite_score >= 65.0:
        persona_label = "🚀 Secular Trend-Persistence Regime"
        regime_mode = "High-Persistence Monster Run Market"
        trading_mandate = "Positional Trading"
        mandate_summary = (
            "Sector leadership is persistent and breakouts are powering through. "
            "Hold high-conviction leaders for monster campaign runs (+40% to +100%+). "
            "Trail the 50-day SMA or 21-day EMA. Avoid premature profit taking."
        )
        badge_color = "#10b981"
        badge_bg = "rgba(16, 185, 129, 0.15)"
        icon = "🚀"
    elif composite_score >= 45.0:
        persona_label = "⚖️ Selective Consolidation Regime"
        regime_mode = "Rotational Digestion Market"
        trading_mandate = "Hybrid Trading"
        mandate_summary = (
            "Selective environment with moderate sector rotation. "
            "Deploy Minervini's Free-Roll: Lock in partial gains (+15% to +20%) to finance risk, "
            "and trail the remainder with a breakeven stop to catch potential runners."
        )
        badge_color = "#f59e0b"
        badge_bg = "rgba(245, 158, 11, 0.15)"
        icon = "⚖️"
    else:
        persona_label = "🌪️ High-Velocity Churn Regime"
        regime_mode = "High-Velocity Sector Churn & Squat Market"
        trading_mandate = "Swing Trading Only"
        mandate_summary = (
            "Sectors are rotating rapidly and breakout squats are frequent. "
            "Strict swing trading: Take quick profits into strength at +12% to +18%. "
            "Trail a tight 10-day EMA, keep stops tight (3%-5%), and do NOT marry positions."
        )
        badge_color = "#ef4444"
        badge_bg = "rgba(239, 68, 68, 0.15)"
        icon = "🌪️"
        
    details = {
        'pillar1': p1,
        'pillar2': p2,
        'pillar3': p3,
        'pillar4': p4,
        'regime_mode': regime_mode,
        'mandate_summary': mandate_summary,
        'badge_color': badge_color,
        'badge_bg': badge_bg,
        'icon': icon
    }
    
    record = {
        'date': today_str,
        'market': mkt_key,
        'composite_score': composite_score,
        'persona_label': persona_label,
        'trading_mandate': trading_mandate,
        'sector_tenure_pct': p1.get('tenure_pct', 0.0),
        'breakout_health_pct': p2.get('breakout_health_pct', 0.0),
        'leader_21ema_pct': p3.get('leader_21ema_pct', 0.0),
        'mega_cap_spread': p4.get('mega_cap_spread', 0.0),
        'details_json': details,
        'details': details
    }
    
    # Save to SQLite
    save_market_persona(record)
    return record


def sync_market_persona_daily(market: str = 'INDIA') -> dict:
    """
    Autonomous daily sync entry point.
    Pre-computes and caches today's market persona in SQLite so app.py renders in < 2ms.
    """
    mkt_key = 'INDIA' if market.upper() in ['INDIA', 'IN'] else 'US'
    return calculate_composite_market_persona(market=mkt_key, force_recompute=True)
