"""
Portfolio Health Engine
=======================
Institutional-grade factor analytics and briefing generator for active portfolio holdings:
- Weighted Portfolio Relative Strength (RS Rating 0-99, accurately ignoring unranked SME IPOs)
- Dual-Horizon Sortino Ratios (3M Tactical vs 6M Structural)
- Net Unrealized P&L in Rupees and Percentage
- Equity Exposure vs Liquid Reserve % and Total Open Heat
- Jim Roppel CANSLIM Quadrants + Dedicated Recent IPOs/SMEs Category
- Action Trigger Reason Badging (e.g. 2 Closes < 21 EMA, ATR Stop Hit, 21 EMA Retest)
- Smart BSE vs NSE TradingView Deep-Links (numeric scrips open BSE charts)
- One-Tap Copyable Watchlists for Actionable Exits and Adds
"""

import html
import logging
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def clean_ticker(ticker: str) -> str:
    """Strips exchange suffixes (.NS, .BO) and cleans ticker string."""
    return str(ticker).replace('.NS', '').replace('.BO', '').split('-')[0].strip().upper()


def get_tradingview_url(sym: str, raw_ticker: Optional[str] = None) -> str:
    """Generates TradingView chart link. Uses BSE for numeric scrips or .BO, NSE otherwise."""
    is_bse = sym.isdigit() or (raw_ticker is not None and str(raw_ticker).upper().endswith('.BO'))
    exchange = "BSE" if is_bse else "NSE"
    return f"https://in.tradingview.com/chart/?symbol={exchange}:{sym}"


def get_tradingview_symbol(sym: str, raw_ticker: Optional[str] = None) -> str:
    """Returns properly prefixed symbol for TradingView watchlists (e.g. BSE:544563 or NSE:SETL)."""
    is_bse = sym.isdigit() or (raw_ticker is not None and str(raw_ticker).upper().endswith('.BO'))
    exchange = "BSE" if is_bse else "NSE"
    return f"{exchange}:{sym}"


def _parse_action_reason(act: str, raw_reason: str, stop_breached: bool = False) -> str:
    """Extracts a concise, institutional 2-4 word trigger reason."""
    act_upper = act.upper()
    r_lower = str(raw_reason).lower()

    if "EXIT" in act_upper:
        if "2 close" in r_lower or "2 consecutive close" in r_lower or "below 21" in r_lower:
            return "2 Closes < 21 EMA"
        elif "50 sma" in r_lower:
            return "50 SMA Breakdown"
        elif "chandelier" in r_lower or "atr" in r_lower:
            return "ATR Stop"
        elif "climax" in r_lower:
            return "Climax Blow-Off"
        elif "hard stop" in r_lower or "loss floor" in r_lower:
            return "8% Hard Stop Floor"
        elif "max loss" in r_lower or "portfolio loss" in r_lower or "l1" in r_lower:
            return "Max Risk Ceiling"
        elif stop_breached:
            return "Trailing Stop Hit"
        else:
            return "Loss of Momentum"
    elif "TRIM" in act_upper:
        if "haber" in r_lower or "2 close" in r_lower or "below 21" in r_lower:
            return "Haber 21 EMA (Trim ½)"
        elif "1 close" in r_lower:
            return "1 Close < 21 EMA"
        elif "climax" in r_lower:
            return "Climax Warning"
        elif "chandelier" in r_lower or "atr" in r_lower:
            return "ATR Warning"
        else:
            return "Defensive Trim"
    elif "ADD" in act_upper:
        return "21 EMA Retest"
    else:
        return ""


def classify_corporate_announcement(title: str, desc: str) -> Tuple[Optional[str], int, str]:
    """
    Classifies corporate announcements into high-signal institutional catalyst categories.
    Returns (category_badge, priority_score, summary_headline).
    Returns (None, 0, "") if the announcement is routine administrative noise.
    """
    import re
    text = (str(title) + " " + str(desc)).lower()

    # 1. Noise Filter: Omit routine administrative / compliance filings
    NOISE_PATTERNS = [
        'scrutinizer', 'voting result', 'shareholders meeting', 'proceedings of the agm',
        'annual general meeting held on', 'loss of share certificate', 'duplicate share',
        'trading window', 'newspaper publication', 'analyst / investor meet - intimation',
        'investor meet - intimation', 'allotment of esop', 'clarification sought',
        'loss of certificate', 'closure of trading', 'confirmation certificate under reg',
        'unclaimed dividend'
    ]
    if any(p in text for p in NOISE_PATTERNS):
        return None, 0, ""

    # Clean boilerplate prefixes
    raw_title = str(title).strip()
    raw_desc = str(desc).strip()
    clean_t = re.sub(r'(?i)^announcement\s+under\s+(?:reg|regulation)\s*\d*\s*(?:\([^)]*\))?\s*[-:]\s*', '', raw_title).strip()
    clean_t = re.sub(r'(?i)^board\s+meeting\s+(?:intimation|outcome)\s+(?:for|under)?\s*[-:]?\s*', '', clean_t).strip()
    clean_t = clean_t.replace('_', ' ')

    boilerplate = ['intimation under regulation', 'enclosed herewith', 'has informed the exchange', 'pursuant to regulation']
    has_boilerplate = any(b in raw_desc.lower() for b in boilerplate)

    GENERIC_TITLES = {'press release', 'press release / media release', 'media release', 'updates', 'general updates', 'announcement', 'outcome of board meeting', 'outcome'}

    quoted_match = re.search(r'titled\s+["\']([^"\']+)["\']', raw_desc, re.IGNORECASE)
    if quoted_match:
        headline = quoted_match.group(1).strip()
    elif clean_t.lower() in GENERIC_TITLES and raw_desc and not has_boilerplate:
        headline = raw_desc[:85] + '...' if len(raw_desc) > 85 else raw_desc
    elif raw_desc and not has_boilerplate and len(raw_desc) > 10 and len(raw_desc) < 95:
        headline = raw_desc
    elif len(clean_t) > 5 and clean_t.lower() not in GENERIC_TITLES:
        headline = clean_t[:85] + '...' if len(clean_t) > 85 else clean_t
    else:
        headline = raw_desc[:85] + '...' if len(raw_desc) > 85 else raw_desc

    # 2. Insider Trading (PIT)
    if '[insider buy]' in text or 'insider buy' in text:
        return '🟢 INSIDER BUY', 1, headline
    if '[insider sell]' in text or 'insider sell' in text:
        return '🔴 INSIDER SELL', 1, headline

    # 3. Preferential Issue, Warrants & QIP
    if 'preferential issue' in text or 'preferential allotment' in text or 'warrant' in text:
        return '⚡ PREFERENTIAL ISSUE', 2, headline
    if 'qualified institution' in text or 'qip' in text:
        return '💼 QIP / CAPITAL RAISE', 2, headline
    if 'rights issue' in text:
        return '⚡ RIGHTS ISSUE', 2, headline
    if 'buyback' in text:
        return '💰 BUYBACK', 2, headline
    if 'bonus' in text or 'stock split' in text or 'sub-division' in text:
        return '🎁 BONUS / SPLIT', 2, headline

    # 4. SAST & Ownership Disclosures
    if 'sast' in text or 'substantial acquisition' in text or 'reg. 29' in text or 'reg 29' in text or 'reg. 31' in text or 'reg 31' in text or 'encumbrance' in text or 'pledge' in text:
        return '🛡️ SAST / OWNERSHIP', 3, headline

    # 5. Major Capital & Share Allotments
    if 'allotment' in text and ('securities' in text or 'equity' in text) and 'esop' not in text:
        return '💼 SHARE ALLOTMENT', 4, headline

    # 6. Business Catalysts & Expansion
    if 'acquisition' in text or 'amalgamation' in text or 'merger' in text or 'takeover of business' in text:
        return '🚀 ACQUISITION', 5, headline
    if 'award_of_order' in text or 'receipt_of_order' in text or 'order win' in text or 'order receipt' in text or 'contract' in text:
        return '📦 ORDER WIN', 5, headline
    if 'commencement of' in text or 'commercial production' in text or 'commercial operation' in text or 'new manufacturing' in text:
        return '🏭 CAPACITY EXPANSION', 5, headline
    if 'financial result' in text or 'earnings' in text:
        return '📊 FINANCIAL RESULTS', 6, headline

    # 7. Governance & Credit
    if 'change in management' in text or 'cessation' in text or 'resignation' in text or 'appointment' in text:
        return '🏛️ MGMT CHANGE', 7, headline
    if 'credit rating' in text:
        return '🏷️ CREDIT RATING', 8, headline

    return None, 0, ""


def get_material_portfolio_announcements(
    lookback_hours: Optional[int] = None,
    active_tickers: Optional[List[str]] = None,
    max_items: int = 8
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Fetches and filters high-signal corporate announcements for active portfolio holdings.
    Applies adaptive lookback (72h on weekends/Mondays, 36h on weekdays), smart deduplication,
    and priority ranking.
    """
    now = datetime.now()
    if lookback_hours is None:
        # Weekend or Monday morning: expand to 72 hours
        lookback_hours = 72 if now.weekday() in (0, 5, 6) else 36

    lookback_str = f"Last {lookback_hours}H"
    cutoff = now - timedelta(hours=lookback_hours)
    cutoff_str = cutoff.strftime('%Y-%m-%d %H:%M:%S')

    try:
        import database
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, date_time, title, description, pdf_link FROM corporate_announcements WHERE date_time >= ? ORDER BY date_time DESC",
            (cutoff_str,)
        )
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        logger.warning(f"[Portfolio Health Engine] Error loading announcements: {e}")
        return [], lookback_str

    if not rows:
        return [], lookback_str

    cleaned_active_set = None
    if active_tickers:
        cleaned_active_set = {clean_ticker(t) for t in active_tickers if t}

    material_items = []
    seen_dedup = set()

    for r in rows:
        raw_ticker = str(r[0] or '')
        sym = clean_ticker(raw_ticker)
        if cleaned_active_set is not None and sym not in cleaned_active_set:
            continue

        raw_dt = str(r[1] or '')
        raw_title = str(r[2] or '')
        raw_desc = str(r[3] or '')
        pdf_link = str(r[4] or '')

        cat_badge, priority, headline = classify_corporate_announcement(raw_title, raw_desc)
        if not cat_badge:
            continue

        dt_day = raw_dt[:10]
        dedup_key = (sym, cat_badge, dt_day)
        if dedup_key in seen_dedup:
            continue
        seen_dedup.add(dedup_key)

        # Format relative timestamp
        try:
            dt_obj = datetime.strptime(raw_dt[:19], '%Y-%m-%d %H:%M:%S')
            if dt_obj.date() == now.date():
                time_str = dt_obj.strftime('Today %H:%M')
            elif dt_obj.date() == (now - timedelta(days=1)).date():
                time_str = dt_obj.strftime('Yesterday %H:%M')
            else:
                time_str = dt_obj.strftime('%b %d, %H:%M')
        except Exception:
            time_str = raw_dt[:16]

        tv_url = get_tradingview_url(sym, raw_ticker=raw_ticker)
        valid_pdf = pdf_link if (pdf_link and pdf_link.startswith('http')) else ""

        material_items.append({
            'ticker': sym,
            'category': cat_badge,
            'priority': priority,
            'date_time': raw_dt,
            'time_str': time_str,
            'title': headline,
            'pdf_link': valid_pdf,
            'tv_url': tv_url
        })

    # Sort by datetime descending
    material_items.sort(key=lambda x: x['date_time'], reverse=True)
    capped_items = material_items[:max_items]

    return capped_items, lookback_str


def get_latest_breadth_pulse(market: str = 'IN') -> Optional[Dict[str, Any]]:
    """
    Fetches the latest institutional market breadth snapshot from SQLite (market_breadth_daily).
    Takes ~1ms and does not require running heavy calculations.
    """
    try:
        import database, sqlite3
        conn = database.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM market_breadth_daily WHERE market = ? ORDER BY date DESC LIMIT 1",
            (market,)
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None

        d = dict(row)
        a_cnt = int(d.get('ad_grade_a_count', 0) or 0)
        b_cnt = int(d.get('ad_grade_b_count', 0) or 0)
        d_cnt = int(d.get('ad_grade_d_count', 0) or 0)
        e_cnt = int(d.get('ad_grade_e_count', 0) or 0)
        dist_tot = d_cnt + e_cnt
        sm_ratio = round((a_cnt + b_cnt) / dist_tot, 2) if dist_tot > 0 else 1.0

        raw_date = d.get('date', '')
        try:
            dt_obj = datetime.strptime(raw_date, '%Y-%m-%d')
            date_formatted = dt_obj.strftime('%b %d, %Y')
        except Exception:
            date_formatted = raw_date

        total_stocks = int(d.get('total_stocks', 750) or 750)
        up_down_ratio = float(d.get('up_down_volume_ratio', 1.0) or 1.0)
        vol_badge = "Demand" if up_down_ratio >= 1.2 else ("Supply" if up_down_ratio <= 0.8 else "Neutral")

        net_nh = int(d.get('net_new_highs', 0) or 0)
        nh_cnt = int(d.get('new_highs_count', 0) or 0)
        nl_cnt = int(d.get('new_lows_count', 0) or 0)
        nh_sign = "+" if net_nh >= 0 else ""

        above_50 = float(d.get('above_50_pct', 0.0) or 0.0)

        return {
            'date': raw_date,
            'date_formatted': date_formatted,
            'total_stocks': total_stocks,
            'up_down_volume_ratio': up_down_ratio,
            'vol_badge': vol_badge,
            'smart_money_ratio': sm_ratio,
            'net_new_highs': net_nh,
            'nh_sign': nh_sign,
            'new_highs_count': nh_cnt,
            'new_lows_count': nl_cnt,
            'above_50_pct': round(above_50, 1)
        }
    except Exception as e:
        logger.warning(f"[Portfolio Health Engine] Error reading breadth pulse: {e}")
        return None


def calculate_portfolio_health_summary(
    decisions: List[Dict[str, Any]],
    total_value: float,
    market_regime_label: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes institutional mathematical aggregates across active holdings.
    """
    if not decisions or total_value <= 0:
        return {
            'has_positions': False,
            'total_positions': 0,
            'weighted_rs': 0.0,
            'weighted_sortino_3m': 0.0,
            'weighted_sortino_6m': 0.0,
            'weighted_smooth_rs': 0.0,
            'total_value': total_value,
            'equity_value': 0.0,
            'liquid_value': 0.0,
            'total_cost_basis': 0.0,
            'net_unrealized_pnl_dollars': 0.0,
            'net_unrealized_pnl_pct': 0.0,
            'equity_pct': 0.0,
            'liquid_pct': 100.0,
            'open_heat_pct': 0.0,
            'open_heat_dollars': 0.0,
            'ranked_count': 0,
            'unranked_sme_count': 0,
            'action_counts': {'ADD': 0, 'HOLD': 0, 'TRIM': 0, 'EXIT': 0},
            'quadrants': {'apex': [], 'core': [], 'bronco': [], 'defender': [], 'laggard': [], 'ipo_sme': []},
            'material_announcements': [],
            'announcements_window_str': 'Last 36-48H',
            'breadth_pulse': get_latest_breadth_pulse('IN')
        }

    CASH_EQUIVALENTS = {'LIQUIDBEES', 'LIQUIDCASE', 'LIQUIDETF', 'GOLDBEES', 'SILVERBEES'}

    equity_items = []
    liquid_value = 0.0
    total_dollar_risk = 0.0
    total_cost_basis = 0.0
    action_counts = {'ADD': 0, 'HOLD': 0, 'TRIM': 0, 'EXIT': 0}

    for d in decisions:
        raw_t = str(d.get('ticker', '')).upper()
        clean_t = clean_ticker(raw_t)
        px = float(d.get('close_price', 0.0) or 0.0)
        qty = float(d.get('quantity', 0.0) or 0.0)
        buy_px = float(d.get('avg_buy_price', px) or px)
        pos_val = px * qty
        cost_val = buy_px * qty

        if clean_t in CASH_EQUIVALENTS:
            liquid_value += pos_val
            continue

        if pos_val <= 0:
            continue

        total_cost_basis += cost_val
        dollar_risk = float(d.get('dollar_risk', 0.0) or 0.0)
        total_dollar_risk += dollar_risk

        act = str(d.get('action', 'HOLD')).upper()
        if 'EXIT' in act:
            action_counts['EXIT'] += 1
            canonical_act = 'EXIT'
        elif 'TRIM' in act:
            action_counts['TRIM'] += 1
            canonical_act = 'TRIM'
        elif 'ADD' in act:
            action_counts['ADD'] += 1
            canonical_act = 'ADD'
        else:
            action_counts['HOLD'] += 1
            canonical_act = 'HOLD'

        rs_raw = d.get('rs_score')
        try:
            rs = float(rs_raw) if (rs_raw is not None and not pd.isna(rs_raw)) else 0.0
        except Exception:
            rs = 0.0

        s3m = float(d.get('sortino_3m', 0.0) or 0.0)
        s6m = float(d.get('sortino_6m', 0.0) or 0.0)
        smooth_rs = float(d.get('smooth_rs', 0.0) or 0.0)
        
        # P&L calculation
        loss_metrics = d.get('loss_metrics', {})
        pnl_pct = float(loss_metrics.get('gain_loss_pct', 0.0) or 0.0)
        if pnl_pct == 0.0 and buy_px > 0 and px > 0:
            pnl_pct = ((px / buy_px) - 1.0) * 100.0

        # Trigger reason
        raw_reason = str(d.get('reason', ''))
        stop_breached = bool(d.get('stop_breached', False))
        trigger_reason = _parse_action_reason(canonical_act, raw_reason, stop_breached)

        # Classify Quadrants:
        # Detect recent IPOs / SMEs (<63d history where RS is 0 and Sortino is 0.00)
        is_fresh_ipo_or_sme = (rs <= 0.0) and (abs(s3m) < 0.001 and abs(s6m) < 0.001)

        if is_fresh_ipo_or_sme:
            quadrant = 'ipo_sme'
            quad_label = "🆕 Recent IPO / SME"
        elif rs >= 80.0 and s3m >= 3.0:
            quadrant = 'apex'
            quad_label = "👑 Apex Compounder"
        elif rs >= 80.0 and s3m < 2.0:
            quadrant = 'bronco'
            quad_label = "🎢 Wild Bronco"
        elif rs < 70.0 and s3m >= 3.0:
            quadrant = 'defender'
            quad_label = "🐢 Sleepy Defender"
        elif rs < 70.0 and s3m < 1.5:
            quadrant = 'laggard'
            quad_label = "🔴 Laggard / Trap"
        else:
            quadrant = 'core'
            quad_label = "⚪ Core Momentum"

        equity_items.append({
            'raw_ticker': raw_t,
            'clean_ticker': clean_t,
            'tv_symbol': get_tradingview_symbol(clean_t, raw_t),
            'position_value': pos_val,
            'cost_value': cost_val,
            'price': px,
            'avg_buy_price': buy_px,
            'quantity': qty,
            'rs_score': rs,
            'sortino_3m': s3m,
            'sortino_6m': s6m,
            'smooth_rs': smooth_rs,
            'pnl_pct': pnl_pct,
            'dollar_risk': dollar_risk,
            'action': canonical_act,
            'trigger_reason': trigger_reason,
            'industry': d.get('industry', 'Unknown'),
            'stop_breached': stop_breached,
            'is_unranked': is_fresh_ipo_or_sme,
            'quadrant': quadrant,
            'quad_label': quad_label,
            'chart_url': get_tradingview_url(clean_t, raw_t)
        })

    total_equity_val = sum(item['position_value'] for item in equity_items)
    
    # Calculate position weights vs total portfolio NAV
    for item in equity_items:
        item['weight_pct'] = (item['position_value'] / total_value * 100.0) if total_value > 0 else 0.0

    # Calculate Weighted Metrics across RANKED equities (excluding 0 RS unranked IPOs)
    ranked_items = [it for it in equity_items if not it['is_unranked'] and it['rs_score'] > 0]
    total_ranked_val = sum(it['position_value'] for it in ranked_items)

    if total_ranked_val > 0:
        weighted_rs = sum(it['rs_score'] * (it['position_value'] / total_ranked_val) for it in ranked_items)
        weighted_s3m = sum(it['sortino_3m'] * (it['position_value'] / total_ranked_val) for it in ranked_items)
        weighted_s6m = sum(it['sortino_6m'] * (it['position_value'] / total_ranked_val) for it in ranked_items)
        weighted_smooth_rs = sum(it['smooth_rs'] * (it['position_value'] / total_ranked_val) for it in ranked_items)
    elif total_equity_val > 0:
        weighted_rs = sum(it['rs_score'] * (it['position_value'] / total_equity_val) for it in equity_items)
        weighted_s3m = sum(it['sortino_3m'] * (it['position_value'] / total_equity_val) for it in equity_items)
        weighted_s6m = sum(it['sortino_6m'] * (it['position_value'] / total_equity_val) for it in equity_items)
        weighted_smooth_rs = sum(it['smooth_rs'] * (it['position_value'] / total_equity_val) for it in equity_items)
    else:
        weighted_rs = 0.0
        weighted_s3m = 0.0
        weighted_s6m = 0.0
        weighted_smooth_rs = 0.0

    # Net Unrealized Portfolio Gain
    net_pnl_dollars = total_equity_val - total_cost_basis
    net_pnl_pct = (net_pnl_dollars / total_cost_basis * 100.0) if total_cost_basis > 0 else 0.0

    equity_pct = (total_equity_val / total_value * 100.0) if total_value > 0 else 0.0
    liquid_pct = ((total_value - total_equity_val) / total_value * 100.0) if total_value > 0 else 0.0
    open_heat_pct = (total_dollar_risk / total_value * 100.0) if total_value > 0 else 0.0

    # Group into Quadrants
    quadrants = {'apex': [], 'core': [], 'bronco': [], 'defender': [], 'laggard': [], 'ipo_sme': []}
    for item in equity_items:
        quadrants[item['quadrant']].append(item)

    for q_key in quadrants:
        quadrants[q_key].sort(key=lambda x: x['weight_pct'], reverse=True)

    # Health evaluation badge
    if weighted_rs >= 85.0:
        rs_status = "🟢 Elite Alpha Machine (Top 15%)"
    elif weighted_rs >= 75.0:
        rs_status = "🟢 Healthy Institutional Momentum"
    elif weighted_rs >= 65.0:
        rs_status = "🟡 Developing / Core"
    else:
        rs_status = "🔴 Underperforming Laggard Zone"

    # Corporate Announcements Radar & Market Breadth Pulse
    active_tickers = [d.get('ticker') for d in decisions if d.get('ticker')]
    material_announcements, ann_window_str = get_material_portfolio_announcements(active_tickers=active_tickers)
    breadth_pulse = get_latest_breadth_pulse('IN')

    return {
        'has_positions': len(equity_items) > 0,
        'total_positions': len(equity_items),
        'ranked_count': len(ranked_items),
        'unranked_sme_count': len(equity_items) - len(ranked_items),
        'weighted_rs': round(weighted_rs, 1),
        'rs_status': rs_status,
        'weighted_sortino_3m': round(weighted_s3m, 2),
        'weighted_sortino_6m': round(weighted_s6m, 2),
        'weighted_smooth_rs': round(weighted_smooth_rs, 2),
        'total_value': total_value,
        'equity_value': total_equity_val,
        'liquid_value': liquid_value,
        'total_cost_basis': total_cost_basis,
        'net_unrealized_pnl_dollars': round(net_pnl_dollars, 2),
        'net_unrealized_pnl_pct': round(net_pnl_pct, 1),
        'equity_pct': round(equity_pct, 1),
        'liquid_pct': round(liquid_pct, 1),
        'open_heat_pct': round(open_heat_pct, 2),
        'open_heat_dollars': round(total_dollar_risk, 2),
        'action_counts': action_counts,
        'quadrants': quadrants,
        'all_items': sorted(equity_items, key=lambda x: x['weight_pct'], reverse=True),
        'market_regime': market_regime_label or "Active Trend Scan",
        'material_announcements': material_announcements,
        'announcements_window_str': ann_window_str,
        'breadth_pulse': breadth_pulse
    }


def format_telegram_portfolio_briefing(summary: Dict[str, Any], date_str: Optional[str] = None) -> str:
    """
    Assembles a high-density, institutional CANSLIM Telegram briefing.
    Features:
    - Net P&L in ₹ and % alongside Open Heat and Factor Barometer
    - Unranked SME runners isolated so they don't distort weighted RS
    - Trigger reasons on all actions (e.g. 2 Closes < 21, ATR Stop Hit, Retest)
    - Compressed 2-line format fitting full portfolios into 1-2 Telegram parts
    - One-tap copyable TradingView watchlists for Actionable Exits and Adds
    """
    if date_str is None:
        date_str = date.today().strftime("%A, %b %d, %Y")

    pulse = summary.get('breadth_pulse')
    if not summary.get('has_positions'):
        pulse_text = ""
        if pulse:
            p_date = pulse['date_formatted']
            p_stocks = pulse.get('total_stocks', 750)
            p_vr = pulse['up_down_volume_ratio']
            p_badge = pulse['vol_badge']
            p_sm = pulse['smart_money_ratio']
            p_net_nh = pulse['net_new_highs']
            p_sign = pulse['nh_sign']
            p_nh = pulse['new_highs_count']
            p_nl = pulse['new_lows_count']
            p_sma50 = pulse['above_50_pct']
            pulse_text = (
                f"\n🌊 <b>MARKET BREADTH PULSE ({p_date} • Top ~750 Equities [{p_stocks} Scanned])</b>\n"
                f"• ⚖️ <b>Exchange Vol Flow:</b> <code>{p_vr:.2f}x ({p_badge})</code> | 🏛️ <b>Smart Money:</b> <code>{p_sm:.2f}x</code>\n"
                f"• 🎯 <b>Net 52W Highs:</b> <code>{p_sign}{p_net_nh}</code> ({p_nh} NH / {p_nl} NL) | 📈 <b>>50 SMA:</b> <code>{p_sma50:.1f}%</code>\n"
            )
        return (
            f"📊 <b>KUSH TRACKER — DAILY PORTFOLIO BRIEFING</b>\n"
            f"📅 <i>{date_str}</i>\n"
            f"{pulse_text}\n"
            f"💼 <b>Status:</b> 100% Cash / Liquid Reserves. No active equity positions held."
        )

    w_rs = summary['weighted_rs']
    rs_stat = summary['rs_status']
    s3m = summary['weighted_sortino_3m']
    s6m = summary['weighted_sortino_6m']
    eq_pct = summary['equity_pct']
    liq_pct = summary['liquid_pct']
    heat_pct = summary['open_heat_pct']
    net_pnl_pct = summary['net_unrealized_pnl_pct']
    net_pnl_dlr = summary['net_unrealized_pnl_dollars']
    pnl_sign = "+" if net_pnl_pct >= 0 else ""
    regime = html.escape(str(summary.get('market_regime', 'Active Trend Scan')))
    tot_pos = summary['total_positions']
    ranked_cnt = summary.get('ranked_count', tot_pos)
    sme_cnt = summary.get('unranked_sme_count', 0)

    lines = []

    # 1. Header
    lines.append(f"📊 <b>KUSH TRACKER — PORTFOLIO HEALTH BRIEFING</b>")
    lines.append(f"📅 <i>{date_str}</i> | 🎯 {regime}")
    lines.append("")

    # Market Breadth Pulse (IBD GMI Standard)
    if pulse:
        p_date = pulse['date_formatted']
        p_stocks = pulse.get('total_stocks', 750)
        p_vr = pulse['up_down_volume_ratio']
        p_badge = pulse['vol_badge']
        p_sm = pulse['smart_money_ratio']
        p_net_nh = pulse['net_new_highs']
        p_sign = pulse['nh_sign']
        p_nh = pulse['new_highs_count']
        p_nl = pulse['new_lows_count']
        p_sma50 = pulse['above_50_pct']

        lines.append(f"🌊 <b>MARKET BREADTH PULSE ({p_date} • Top ~750 Equities [{p_stocks} Scanned])</b>")
        lines.append(f"• ⚖️ <b>Exchange Vol Flow:</b> <code>{p_vr:.2f}x ({p_badge})</code> | 🏛️ <b>Smart Money:</b> <code>{p_sm:.2f}x</code>")
        lines.append(f"• 🎯 <b>Net 52W Highs:</b> <code>{p_sign}{p_net_nh}</code> ({p_nh} NH / {p_nl} NL) | 📈 <b>>50 SMA:</b> <code>{p_sma50:.1f}%</code>")
        lines.append("")

    # 2. Executive Health Cockpit
    sme_note = f" <i>[{ranked_cnt} Ranked | {sme_cnt} Fresh IPOs]</i>" if sme_cnt > 0 else ""
    lines.append("💼 <b>PORTFOLIO HEALTH COCKPIT</b>")
    lines.append(f"• 📈 <b>Net Unrealized P&L:</b> <code>{pnl_sign}{net_pnl_pct:.1f}%</code> (<code>₹{net_pnl_dlr:+,.0f}</code>)")
    lines.append(f"• ⚡ <b>Weighted RS Score:</b> <code>{w_rs:.1f} / 99</code> ({rs_stat}){sme_note}")
    lines.append(f"• 🛡️ <b>Tactical Sortino (3M):</b> <code>{s3m:+.2f}</code> | 🏛️ <b>Structural (6M):</b> <code>{s6m:+.2f}</code>")
    lines.append(f"• 💵 <b>Asset Allocation:</b> <code>{eq_pct:.1f}% Equities</code> | <code>{liq_pct:.1f}% Liquid/Cash</code>")
    lines.append(f"• ⚠️ <b>Portfolio Open Heat:</b> <code>{heat_pct:.2f}% NAV Risk</code>")
    lines.append("")
    lines.append("")

    # High-Density Single-Line Holding Formatter (Institutional Compactness)
    def _format_compact_row(item: Dict[str, Any]) -> str:
        sym = item['clean_ticker']
        url = item['chart_url']
        wt = item['weight_pct']
        rs_val = item['rs_score']
        s3 = item['sortino_3m']
        s6 = item['sortino_6m']
        pnl = item['pnl_pct']
        p_sign = "+" if pnl >= 0 else ""
        act = item['action']
        reason = item.get('trigger_reason', '')

        if "ADD" in act:
            act_tag = f"⚡ <b>ADD</b>: {reason}" if reason else "⚡ <b>ADD</b>"
        elif "EXIT" in act:
            act_tag = f"🔴 <b>EXIT</b>: {reason}" if reason else "🔴 <b>EXIT</b>"
        elif "TRIM" in act:
            act_tag = f"🟠 <b>TRIM</b>: {reason}" if reason else "🟠 <b>TRIM</b>"
        else:
            act_tag = "🟢 <b>HOLD</b>"

        rs_str = f"{rs_val:.0f}" if rs_val > 0 else "SME"
        s3_str = f"{s3:+.1f}" if (abs(s3) > 0.001) else "N/A"
        s6_str = f"{s6:+.1f}" if (abs(s6) > 0.001) else "N/A"

        if item['is_unranked']:
            return f"• <a href=\"{url}\"><b>{sym}</b></a> ({act_tag}) | Wt: {wt:.1f}% | P&L: {p_sign}{pnl:.1f}% | RS: SME"
        else:
            return f"• <a href=\"{url}\"><b>{sym}</b></a> ({act_tag}) | Wt: {wt:.1f}% | P&L: {p_sign}{pnl:.1f}% | RS: {rs_str} | S: {s3_str}/{s6_str}"

    quads = summary.get('quadrants', {})

    # 3. Apex Compounders (Sorted descending by Portfolio %)
    if quads.get('apex'):
        lines.append(f"👑 <b>APEX COMPOUNDERS ({len(quads['apex'])}) — RS ≥ 80 | S(3M) ≥ 3.0</b>")
        lines.append("<i>Untouchable institutional leaders. Let them compound.</i>")
        for item in quads['apex']:
            lines.append(_format_compact_row(item))
        lines.append("")

    # 4. Core Growth Leaders (Sorted descending by Portfolio %)
    if quads.get('core'):
        lines.append(f"⚪ <b>CORE GROWTH & BASES ({len(quads['core'])}) — RS ≥ 70</b>")
        for item in quads['core']:
            lines.append(_format_compact_row(item))
        lines.append("")

    # 5. Wild Broncos (Sorted descending by Portfolio %)
    if quads.get('bronco'):
        lines.append(f"🎢 <b>WILD BRONCOS ({len(quads['bronco'])}) — RS ≥ 80 | High Pullback Volatility</b>")
        for item in quads['bronco']:
            lines.append(_format_compact_row(item))
        lines.append("")

    # 6. Sleepy Defenders (Sorted descending by Portfolio %)
    if quads.get('defender'):
        lines.append(f"🐢 <b>SLEEPY DEFENDERS ({len(quads['defender'])}) — Low Drawdown | RS &lt; 70</b>")
        for item in quads['defender']:
            lines.append(_format_compact_row(item))
        lines.append("")

    # 7. Recent IPOs & SMEs (Sorted descending by Portfolio %)
    if quads.get('ipo_sme'):
        lines.append(f"🆕 <b>RECENT IPOS & SMES ({len(quads['ipo_sme'])}) — &lt;63D History (Unranked)</b>")
        lines.append("<i>Fresh momentum runners without 63-day Sortino/RS history.</i>")
        for item in quads['ipo_sme']:
            lines.append(_format_compact_row(item))
        lines.append("")

    # 8. Laggards & Distribution Risks (Sorted descending by Portfolio %)
    if quads.get('laggard'):
        lines.append(f"🔴 <b>LAGGARDS & DISTRIBUTION RISKS ({len(quads['laggard'])}) — RS &lt; 70 | S(3M) &lt; 1.5</b>")
        for item in quads['laggard']:
            lines.append(_format_compact_row(item))
        lines.append("")

    # 9. Execution Summary
    acts = summary['action_counts']
    lines.append(f"🎯 <b>EXECUTION & ACTION SUMMARY ({tot_pos} Holdings)</b>")
    lines.append(f"• 🟢 <b>HOLD:</b> {acts['HOLD']} positions intact")
    if acts['ADD'] > 0:
        lines.append(f"• ⚡ <b>ADD:</b> {acts['ADD']} qualifying for +30% pyramiding")
    if acts['TRIM'] > 0:
        lines.append(f"• 🟠 <b>TRIM:</b> {acts['TRIM']} warning of momentum loss")
    if acts['EXIT'] > 0:
        lines.append(f"• 🔴 <b>EXIT:</b> {acts['EXIT']} breached stop-loss / trailing floor")
    lines.append("")

    # 10. Corporate Filings & Material Catalysts Radar
    material_announcements = summary.get('material_announcements', [])
    ann_window_str = summary.get('announcements_window_str', 'Last 36-48H')
    lines.append(f"📡 <b>PORTFOLIO CORPORATE FILINGS & CATALYSTS ({ann_window_str})</b>")
    if material_announcements:
        lines.append("<i>High-impact filings for your active portfolio holdings:</i>")
        for ann in material_announcements:
            sym = ann['ticker']
            cat = ann['category']
            t_str = ann['time_str']
            tv_url = ann['tv_url']
            title_text = html.escape(ann['title'])
            pdf_url = ann.get('pdf_link', '')
            pdf_tag = f" [<a href=\"{pdf_url}\">PDF</a>]" if pdf_url else ""
            lines.append(f"• <a href=\"{tv_url}\"><b>{sym}</b></a> <b>{cat}</b>")
            lines.append(f"  {title_text} <i>({t_str})</i>{pdf_tag}")
    else:
        lines.append("• 🟢 <i>No material insider trades, SAST, or preferential filings reported. All clear.</i>")
    lines.append("")

    # 11. One-Tap Copyable Watchlist Exports for TradingView
    exit_symbols = [it['tv_symbol'] for it in summary['all_items'] if it['action'] == 'EXIT']
    add_symbols = [it['tv_symbol'] for it in summary['all_items'] if it['action'] == 'ADD']
    
    lines.append("📋 <b>ONE-TAP TRADINGVIEW WATCHLIST EXPORTS:</b>")
    if exit_symbols:
        lines.append(f"🔴 <b>Actionable Exits ({len(exit_symbols)}) — Tap to copy:</b>")
        lines.append(f"<code>{', '.join(exit_symbols)}</code>")
    if add_symbols:
        lines.append(f"⚡ <b>Actionable Adds ({len(add_symbols)}) — Tap to copy:</b>")
        lines.append(f"<code>{', '.join(add_symbols)}</code>")

    lines.append("")
    lines.append(f"🔗 <a href=\"https://in.tradingview.com\">Open TradingView Web</a>")

    return "\n".join(lines)
