# Multibagger Analyzer Engine
# Comprehensive technical analysis for buy-readiness scoring
# Inspired by Mark Minervini's Trend Template, VCP, and momentum analysis

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from market_data import fetch_stock_data, get_ticker_symbol
from technical_indicators import (
    calculate_ema, calculate_sma, calculate_atr, calculate_returns,
    add_technical_indicators, get_atr_state
)
from relative_strength import calculate_all_rs_scores
from trend_analyzer import determine_trend_state


# =============================================================================
# INDIVIDUAL METRIC CALCULATORS
# =============================================================================

def count_52w_high_hits(df: pd.DataFrame, lookback_days: int) -> int:
    """
    Count days where the stock made a new 52-week high within a lookback period.

    Args:
        df: DataFrame with 'high' column, full history.
        lookback_days: Number of trading days to look back (21 = 1M, 63 = 3M).

    Returns:
        Number of days hitting 52-week high in the lookback window.
    """
    if df.empty or 'high' not in df.columns or len(df) < 252:
        return 0

    # Rolling 252-day max of high, computed for full history
    rolling_52w_high = df['high'].rolling(window=252, min_periods=252).max()

    # Only look at the last `lookback_days` trading days
    recent = df.tail(lookback_days)
    recent_highs = rolling_52w_high.tail(lookback_days)

    # A "hit" = daily high >= 99% of rolling 52-week high (within 1% tolerance)
    hits = (recent['high'].values >= 0.99 * recent_highs.values)
    return int(np.nansum(hits))


def calculate_period_returns(df: pd.DataFrame) -> dict:
    """
    Calculate returns over 1W, 1M, 3M, 6M, and 1Y.

    Returns:
        Dict with keys: ret_1w, ret_1m, ret_3m, ret_6m, ret_1y (as percentages).
    """
    if df.empty or 'close' not in df.columns:
        return {k: None for k in ['ret_1w', 'ret_1m', 'ret_3m', 'ret_6m', 'ret_1y']}

    close = df['close']
    current = close.iloc[-1]

    periods = {
        'ret_1w': 5,
        'ret_1m': 21,
        'ret_3m': 63,
        'ret_6m': 126,
        'ret_1y': 252,
    }

    results = {}
    for key, days in periods.items():
        if len(close) > days:
            past_price = close.iloc[-(days + 1)]
            results[key] = ((current - past_price) / past_price) * 100
        else:
            results[key] = None

    return results


def check_minervini_trend_template(df: pd.DataFrame, rs_score: float = None) -> dict:
    """
    Check all 8 conditions of Minervini's Trend Template.

    Conditions:
      1. Close > 150 SMA
      2. Close > 200 SMA
      3. 150 SMA > 200 SMA
      4. 200 SMA trending up for at least 1 month (22 trading days)
      5. 50 SMA > 150 SMA
      6. Close > 50 SMA
      7. Close within 25% of 52-week high
      8. Close at least 30% above 52-week low
      (Bonus: RS >= 70)

    Returns:
        Dict with pass/fail per condition, overall pass count, and boolean.
    """
    result = {
        'conditions': {},
        'pass_count': 0,
        'total': 8,
        'passed': False,
    }

    if df.empty or len(df) < 200:
        result['conditions'] = {f'C{i}': False for i in range(1, 9)}
        return result

    last = df.iloc[-1]
    close = last['close']

    # Calculate longer SMAs needed for Minervini
    sma_150 = df['close'].rolling(window=150, min_periods=150).mean()
    sma_200 = df['close'].rolling(window=200, min_periods=200).mean()
    sma_50_val = last.get('sma_50', df['close'].rolling(50).mean().iloc[-1])

    if sma_150.iloc[-1] is None or pd.isna(sma_150.iloc[-1]):
        result['conditions'] = {f'C{i}': False for i in range(1, 9)}
        return result

    sma_150_val = sma_150.iloc[-1]
    sma_200_val = sma_200.iloc[-1]

    # 52-week high and low
    high_52w = df['high'].tail(252).max()
    low_52w = df['low'].tail(252).min()

    # Condition checks
    c1 = close > sma_150_val                              # Close > 150 SMA
    c2 = close > sma_200_val                              # Close > 200 SMA
    c3 = sma_150_val > sma_200_val                        # 150 SMA > 200 SMA
    c4 = False                                             # 200 SMA rising for 1 month
    if len(sma_200) >= 22 and not pd.isna(sma_200.iloc[-22]):
        c4 = sma_200.iloc[-1] > sma_200.iloc[-22]
    c5 = sma_50_val > sma_150_val                         # 50 SMA > 150 SMA
    c6 = close > sma_50_val                               # Close > 50 SMA
    c7 = close >= high_52w * 0.75                         # Within 25% of 52W high
    c8 = close >= low_52w * 1.30 if low_52w > 0 else False  # 30% above 52W low

    conditions = {
        'C1: Close > 150 SMA': bool(c1),
        'C2: Close > 200 SMA': bool(c2),
        'C3: 150 SMA > 200 SMA': bool(c3),
        'C4: 200 SMA Rising (1M)': bool(c4),
        'C5: 50 SMA > 150 SMA': bool(c5),
        'C6: Close > 50 SMA': bool(c6),
        'C7: Within 25% of 52W High': bool(c7),
        'C8: 30%+ Above 52W Low': bool(c8),
    }

    pass_count = sum(conditions.values())
    result['conditions'] = conditions
    result['pass_count'] = pass_count
    result['total'] = 8
    result['passed'] = pass_count >= 7  # Allow 1 miss as near-pass

    return result


def detect_vcp_pattern(df: pd.DataFrame) -> dict:
    """
    Detect Volatility Contraction Pattern (VCP).

    VCP requires successive contractions in price range. We look for 2-4 contractions
    over the last 35-50 trading days (7-10 weeks), where each contraction is ≥30%
    smaller than the previous one.

    Returns:
        Dict with detected, num_contractions, contractions list.
    """
    result = {
        'detected': False,
        'num_contractions': 0,
        'contractions': [],
        'depth_shrinking': False,
    }

    if df.empty or len(df) < 50:
        return result

    # Look at the last 50 trading days
    recent = df.tail(50).copy()
    close = recent['close'].values
    high = recent['high'].values
    low = recent['low'].values

    # Break into ~weekly segments (5 trading days each) for 10 weeks
    segment_size = 5
    num_segments = len(close) // segment_size
    if num_segments < 4:
        return result

    # Calculate range (high-low) for each weekly segment as % of mid price
    ranges = []
    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size
        seg_high = high[start_idx:end_idx].max()
        seg_low = low[start_idx:end_idx].min()
        seg_mid = (seg_high + seg_low) / 2
        if seg_mid > 0:
            range_pct = ((seg_high - seg_low) / seg_mid) * 100
            ranges.append(range_pct)

    if len(ranges) < 4:
        return result

    # Find successive contractions (each range smaller than previous)
    contractions = []
    for i in range(1, len(ranges)):
        if ranges[i] < ranges[i - 1]:
            contraction_pct = (1 - ranges[i] / ranges[i - 1]) * 100
            contractions.append({
                'week': i + 1,
                'range_pct': ranges[i],
                'contraction_from_prior': contraction_pct,
            })

    # Check for 2+ contractions with meaningful shrinkage
    significant = [c for c in contractions if c['contraction_from_prior'] >= 20]

    # Also check if the most recent 2 weeks have tighter ranges than the first 2
    early_avg = np.mean(ranges[:3])
    late_avg = np.mean(ranges[-3:])
    depth_shrinking = late_avg < early_avg * 0.70  # Last 3 weeks 30% tighter

    result['contractions'] = contractions
    result['num_contractions'] = len(significant)
    result['depth_shrinking'] = depth_shrinking
    result['detected'] = len(significant) >= 2 and depth_shrinking

    return result


def check_volume_dryup(df: pd.DataFrame) -> bool:
    """
    Detect volume dry-up: last 5-day avg volume < 50% of 50-day avg volume.
    This signals institutional selling is drying up (favorable for breakout).
    """
    if df.empty or 'volume' not in df.columns or len(df) < 50:
        return False

    vol_5d = df['volume'].tail(5).mean()
    vol_50d = df['volume'].tail(50).mean()

    if vol_50d <= 0:
        return False

    return vol_5d < (vol_50d * 0.50)


def check_ema_stack(df: pd.DataFrame) -> dict:
    """
    Check EMA stack alignment: 8 EMA > 21 EMA > 50 SMA > 150 SMA > 200 SMA.

    Returns:
        Dict with each stack level and whether perfect stack exists.
    """
    result = {
        'ema8_gt_ema21': False,
        'ema21_gt_sma50': False,
        'sma50_gt_sma150': False,
        'sma150_gt_sma200': False,
        'perfect_stack': False,
    }

    if df.empty or len(df) < 200:
        return result

    last = df.iloc[-1]
    ema8 = last.get('ema_8')
    ema21 = last.get('ema_21')
    sma50 = last.get('sma_50')
    sma150 = df['close'].rolling(150).mean().iloc[-1]
    sma200 = df['close'].rolling(200).mean().iloc[-1]

    if any(pd.isna(x) for x in [ema8, ema21, sma50, sma150, sma200]):
        return result

    result['ema8_gt_ema21'] = bool(ema8 > ema21)
    result['ema21_gt_sma50'] = bool(ema21 > sma50)
    result['sma50_gt_sma150'] = bool(sma50 > sma150)
    result['sma150_gt_sma200'] = bool(sma150 > sma200)
    result['perfect_stack'] = all([
        result['ema8_gt_ema21'],
        result['ema21_gt_sma50'],
        result['sma50_gt_sma150'],
        result['sma150_gt_sma200'],
    ])

    return result


# =============================================================================
# BUY READINESS SCORING
# =============================================================================

def calculate_fundamental_score(fundamentals: dict) -> tuple:
    """
    Calculate Fundamental Catalyst Score (0-5) based on O'Neil CANSLIM logic.
    
    Criteria:
      - EPS Growth > 20%: +1
      - Sales Growth > 20%: +1
      - ROE > 15%: +1
      - Operating Profit Margin (OPM) > 15%: +1
      - Debt to Equity < 1.0: +1
      
    Returns:
        (score, breakdown_dict)
    """
    score = 0
    breakdown = {}
    
    # EPS Growth
    eps = fundamentals.get('eps_growth', 0) or 0
    if eps > 20:
        score += 1
        breakdown['EPS Growth > 20%'] = f'+1 ✅ ({eps:.1f}%)'
    else:
        breakdown['EPS Growth > 20%'] = f'+0 ❌ ({eps:.1f}%)'
        
    # Sales Growth
    sales = fundamentals.get('sales_growth', 0) or 0
    if sales > 20:
        score += 1
        breakdown['Sales Growth > 20%'] = f'+1 ✅ ({sales:.1f}%)'
    else:
        breakdown['Sales Growth > 20%'] = f'+0 ❌ ({sales:.1f}%)'
        
    # ROE
    roe = fundamentals.get('roe', 0) or 0
    if roe > 15:
        score += 1
        breakdown['ROE > 15%'] = f'+1 ✅ ({roe:.1f}%)'
    else:
        breakdown['ROE > 15%'] = f'+0 ❌ ({roe:.1f}%)'
        
    # OPM
    opm = fundamentals.get('opm', 0) or 0
    if opm > 15:
        score += 1
        breakdown['OPM > 15%'] = f'+1 ✅ ({opm:.1f}%)'
    else:
        breakdown['OPM > 15%'] = f'+0 ❌ ({opm:.1f}%)'
        
    # Debt to Equity
    de = fundamentals.get('debt_to_equity', 0)
    # Handle missing or 0 effectively (0 is good for debt)
    if pd.isna(de):
        breakdown['Debt/Equity < 1.0'] = '+0 ❌ (N/A)'
    elif de < 1.0:
        score += 1
        breakdown['Debt/Equity < 1.0'] = f'+1 ✅ ({de:.2f})'
    else:
        breakdown['Debt/Equity < 1.0'] = f'+0 ❌ ({de:.2f})'
        
    return score, breakdown


def calculate_buy_readiness(
    minervini: dict,
    vcp: dict,
    atr_state: str,
    rs_score: float,
    volume_dryup: bool,
    dist_from_52w_high: float,
    ema_stack: dict,
    fundamental_score: int = 0,
    fund_breakdown: dict = None,
) -> dict:
    """
    Composite Buy Readiness Score (0-15).

    Scoring (Technical - max 10):
      - Minervini Trend Template pass (7-8/8): +3
      - VCP detected: +2
      - ATR Tight (< 0.80): +1
      - RS Score >= 80: +1
      - Volume Dry-Up: +1
      - Within 15% of 52W High: +1
      - Perfect EMA Stack: +1
      
    Scoring (Fundamental - max 5):
      - Added from fundamental_score

    Labels:
      - 🟢 Ready (11-15)
      - 🟡 Developing (6-10)
      - ⚪ Not Ready (0-5)
    """
    score = 0
    breakdown = {}

    # Minervini (+3)
    if minervini.get('passed', False):
        score += 3
        breakdown['Minervini Template'] = '+3 ✅'
    elif minervini.get('pass_count', 0) >= 6:
        score += 1
        breakdown['Minervini Template'] = '+1 (6/8)'
    else:
        breakdown['Minervini Template'] = '+0 ❌'

    # VCP (+2)
    if vcp.get('detected', False):
        score += 2
        breakdown['VCP Pattern'] = '+2 ✅'
    elif vcp.get('num_contractions', 0) >= 1 and vcp.get('depth_shrinking', False):
        score += 1
        breakdown['VCP Pattern'] = '+1 (forming)'
    else:
        breakdown['VCP Pattern'] = '+0 ❌'

    # ATR Tight (+1)
    if atr_state == 'Tight':
        score += 1
        breakdown['ATR Contraction'] = '+1 ✅'
    else:
        breakdown['ATR Contraction'] = f'+0 ({atr_state})'

    # RS (+1)
    if rs_score is not None and rs_score >= 80:
        score += 1
        breakdown['RS Score ≥ 80'] = '+1 ✅'
    else:
        breakdown['RS Score ≥ 80'] = f'+0 (RS: {rs_score:.0f})' if rs_score else '+0 (N/A)'

    # Volume Dry-Up (+1)
    if volume_dryup:
        score += 1
        breakdown['Volume Dry-Up'] = '+1 ✅'
    else:
        breakdown['Volume Dry-Up'] = '+0 ❌'

    # Within 15% of 52W High (+1)
    if dist_from_52w_high is not None and dist_from_52w_high <= 15:
        score += 1
        breakdown['Within 15% of 52W High'] = '+1 ✅'
    else:
        d = f'{dist_from_52w_high:.1f}%' if dist_from_52w_high else 'N/A'
        breakdown['Within 15% of 52W High'] = f'+0 ({d} away)'

    # EMA Stack (+1)
    if ema_stack.get('perfect_stack', False):
        score += 1
        breakdown['Perfect EMA Stack'] = '+1 ✅'
    else:
        breakdown['Perfect EMA Stack'] = '+0 ❌'

    # Fundamental Score (+5 max)
    score += fundamental_score
    if fund_breakdown:
        breakdown.update(fund_breakdown)

    # Label
    if score >= 11:
        label = '🟢 Ready'
    elif score >= 6:
        label = '🟡 Developing'
    else:
        label = '⚪ Not Ready'

    return {
        'score': score,
        'max_score': 15,
        'label': label,
        'breakdown': breakdown,
    }


# =============================================================================
# MAIN ANALYSIS ORCHESTRATOR
# =============================================================================

def analyze_multibagger_stocks(tickers: list, progress_callback=None, pre_fetched_data=None) -> list:
    """
    Run comprehensive analysis on a list of tickers.

    Args:
        tickers: List of raw ticker symbols (e.g., ['RELIANCE', 'TCS'])
        progress_callback: Optional callable(current, total, ticker) for UI progress.
        pre_fetched_data: Optional dict mapping ticker to historical DataFrame to bypass yfinance.

    Returns:
        List of analysis result dicts, one per ticker.
    """
    results = []
    
    # Build exchange map: if tickers is a dict {ticker: exchange}, use it; otherwise default to NSE
    if isinstance(tickers, dict):
        exchange_map = tickers
        ticker_list = list(tickers.keys())
    else:
        exchange_map = {t: 'NSE' for t in tickers}
        ticker_list = list(tickers)

    total = len(ticker_list)

    # Fetch market data for all tickers
    market_data = {}
    pre_fetched_data = pre_fetched_data or {}
    
    for i, ticker in enumerate(ticker_list):
        if progress_callback:
            progress_callback(i, total, f"Fetching {ticker}...")

        # Bypass yfinance if data is already provided
        if ticker in pre_fetched_data:
            df = pre_fetched_data[ticker].copy()
            df.columns = df.columns.str.lower()
            if 'date' not in df.columns and df.index.name != 'Date' and df.index.name != 'date':
                pass # Usually pre-fetched data has Date index
            df = df.reset_index(names='date') if 'date' not in df.columns else df
            df['ticker'] = ticker
            try:
                df = add_technical_indicators(df)
                market_data[ticker] = df
            except Exception as e:
                print(f"❌ Error processing pre-fetched data for {ticker}: {e}")
            continue

        exchange = exchange_map.get(ticker, 'NSE')
        exchange_val = exchange
        company_name = ''
        if isinstance(exchange, dict):
            company_name = exchange.get('name', '')
            exchange_val = exchange.get('exchange', 'NSE')

        yf_suffix = '.BO' if exchange_val in ['BSE', 'BOM'] else '.NS'
        yf_ticker = f"{ticker}{yf_suffix}"

        try:
            import yfinance as yf
            from datetime import timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=430)
            
            def try_fetch(t):
                s = yf.Ticker(t)
                d = s.history(start=start_date, end=end_date)
                return d if not d.empty else None

            df = try_fetch(yf_ticker)

            # If df is empty or has suspiciously few rows (YFinance -SM.NS glitch returns 1 row)
            if df is None or len(df) < 5:
                print(f"⚠️ Poor/No data for {yf_ticker} (Rows: {len(df) if df is not None else 0}), attempting advanced resolution...")
                
                candidates = []
                
                # 0. Clean ticker (strip -SM, -ST)
                clean_ticker = str(ticker).replace('-SM', '').replace('-ST', '')
                
                # 1. Clean primary exchange
                candidates.append(f"{clean_ticker}{'.NS' if yf_suffix == '.BO' else '.BO'}")
                candidates.append(f"{clean_ticker}{yf_suffix}")
                
                # 2. SME/Emerge suffixes
                candidates.extend([f"{clean_ticker}-SM.NS", f"{clean_ticker}-ST.NS"])
                
                # 3. Numeric BSE resolution (search by company name)
                if str(ticker).isdigit() and company_name:
                    search_res = yf.Search(company_name, max_results=3).quotes
                    for res in search_res:
                        sym = res.get('symbol', '')
                        if sym.endswith('.BO') or sym.endswith('.NS'):
                            candidates.append(sym)
                    
                    # Try first word fallback
                    first_word = company_name.split()[0]
                    search_res_fw = yf.Search(first_word, max_results=3).quotes
                    for res in search_res_fw:
                        sym = res.get('symbol', '')
                        if sym.endswith('.BO') or sym.endswith('.NS'):
                            candidates.append(sym)
                            
                # 4. Text ticker resolution
                if not str(ticker).isdigit():
                    search_res = yf.Search(clean_ticker, max_results=3).quotes
                    for res in search_res:
                        sym = res.get('symbol', '')
                        if sym.endswith('.NS') or sym.endswith('.BO'):
                            candidates.append(sym)

                # De-duplicate while preserving order
                seen = set()
                candidates = [x for x in candidates if not (x in seen or seen.add(x))]
                
                # Try candidates and pick the one with the most data
                best_df = df
                best_ticker = yf_ticker
                
                for alt_ticker in candidates:
                    alt_df = try_fetch(alt_ticker)
                    if alt_df is not None:
                        # If we find one with > 5 days of data, use it immediately
                        if len(alt_df) > 5:
                            print(f"  ✅ Found robust data using {alt_ticker} (Rows: {len(alt_df)})")
                            best_df = alt_df
                            best_ticker = alt_ticker
                            break
                        # Otherwise keep track of it if it's better than what we have
                        elif best_df is None or len(alt_df) > len(best_df):
                            best_df = alt_df
                            best_ticker = alt_ticker
                            
                df = best_df
                yf_ticker = best_ticker
                
            # Final validation: If even the best candidate has less than 5 days of data, 
            # we reject it entirely so it goes to the "Failed to Load" banner rather than showing 0s.
            if df is None or df.empty or len(df) < 5:
                print(f"  ❌ Still no robust data for {ticker} (Too few rows)")
                continue

            df.columns = df.columns.str.lower()
            df = df.reset_index()
            if 'close' in df.columns:
                df = df.dropna(subset=['close'])
            df['ticker'] = ticker
            if not df.empty:
                df = add_technical_indicators(df)
            market_data[ticker] = df
            market_data[ticker] = df
        except Exception as e:
            print(f"❌ Error fetching {yf_ticker}: {e}")
            continue

    # Calculate RS scores for all tickers at once
    if progress_callback:
        progress_callback(total - 1, total, "Calculating RS scores...")
    rs_data = calculate_all_rs_scores(market_data)

    # Analyze each ticker
    for i, ticker in enumerate(ticker_list):
        if progress_callback:
            progress_callback(i, total, f"Analyzing {ticker}...")

        df = market_data.get(ticker)
        if df is None or df.empty:
            results.append({
                'ticker': ticker,
                'error': 'No data available',
                'buy_readiness': {'score': 0, 'max_score': 10, 'label': '⚪ Not Ready', 'breakdown': {}},
            })
            continue

        # RS Score
        rs_info = rs_data.get(ticker, {})
        rs_score = rs_info.get('rs_score', 0) or 0

        # Trend State
        trend_state = determine_trend_state(df)

        # 52-Week High Hits
        hits_1m = count_52w_high_hits(df, 21)
        hits_3m = count_52w_high_hits(df, 63)

        # Distance from 52W High
        last_row = df.iloc[-1]
        dist_from_high = last_row.get('dist_from_52w_high', None)
        if pd.isna(dist_from_high):
            dist_from_high = None

        # Returns
        returns = calculate_period_returns(df)

        # Minervini Trend Template
        minervini = check_minervini_trend_template(df, rs_score)

        # VCP
        vcp = detect_vcp_pattern(df)

        # ATR State
        atr_ratio = last_row.get('atr_ratio', 1.0)
        if pd.isna(atr_ratio):
            atr_ratio = 1.0
        atr_state = get_atr_state(atr_ratio)

        # Volume Dry-Up
        vol_dryup = check_volume_dryup(df)

        # EMA Stack
        ema_stack = check_ema_stack(df)

        # Fundamental data
        fundamental_score = 0
        fund_breakdown = {'Fundamental Data': 'Unavailable'}
        
        last_row_dict = df.iloc[-1].to_dict()
        if 'eps_growth' in last_row_dict and pd.notna(last_row_dict.get('eps_growth')):
            fundamental_score, fund_breakdown = calculate_fundamental_score(last_row_dict)

        # Buy Readiness Score
        buy_readiness = calculate_buy_readiness(
            minervini=minervini,
            vcp=vcp,
            atr_state=atr_state,
            rs_score=rs_score,
            volume_dryup=vol_dryup,
            dist_from_52w_high=dist_from_high,
            ema_stack=ema_stack,
            fundamental_score=fundamental_score,
            fund_breakdown=fund_breakdown
        )

        # Current price and ADTV
        close_price = last_row.get('close', 0)
        avg_vol_50d = df['volume'].tail(50).mean() if 'volume' in df.columns else 0
        adtv_cr = (avg_vol_50d * close_price) / 10000000 if not pd.isna(avg_vol_50d) else 0

        results.append({
            'ticker': ticker,
            'close_price': close_price,
            'adtv_cr': adtv_cr,
            'rs_score': rs_score,
            'trend_state': trend_state,
            'hits_52w_1m': hits_1m,
            'hits_52w_3m': hits_3m,
            'dist_from_high': dist_from_high,
            'returns': returns,
            'minervini': minervini,
            'vcp': vcp,
            'atr_state': atr_state,
            'atr_ratio': atr_ratio,
            'vol_dryup': vol_dryup,
            'ema_stack': ema_stack,
            'buy_readiness': buy_readiness,
        })

    # Sort by buy readiness score descending
    results.sort(key=lambda x: x.get('buy_readiness', {}).get('score', 0), reverse=True)

    return results


if __name__ == '__main__':
    # Quick test with a couple tickers
    test_results = analyze_multibagger_stocks(['RELIANCE', 'TCS'])
    for r in test_results:
        print(f"\n{r['ticker']}: Score {r['buy_readiness']['score']}/10 — {r['buy_readiness']['label']}")
        for k, v in r['buy_readiness']['breakdown'].items():
            print(f"  {k}: {v}")
