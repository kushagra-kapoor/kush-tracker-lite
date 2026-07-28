# Screener Analytics Engine
# CSV parser, ticker resolver, and analysis orchestrator for screener.in data
# Reuses Minervini, VCP, RS, EMA stack analysis from multibagger_analyzer

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from io import StringIO

from technical_indicators import add_technical_indicators, get_atr_state
from relative_strength import calculate_all_rs_scores
from trend_analyzer import determine_trend_state
from multibagger_analyzer import (
    count_52w_high_hits,
    calculate_period_returns,
    check_minervini_trend_template,
    detect_vcp_pattern,
    check_volume_dryup,
    check_ema_stack,
    calculate_buy_readiness,
    calculate_fundamental_score,
)


# =============================================================================
# CSV PARSING
# =============================================================================

# Expected CSV columns from screener.in (mapped to internal names)
SCREENER_COLUMNS_MAP = {
    'Name': 'name',
    'BSE Code': 'bse_code',
    'NSE Code': 'nse_code',
    'ISIN Code': 'isin_code',
    'Industry Group': 'industry_group',
    'Industry': 'industry',
    'Current Price': 'current_price',
    'Market Capitalization': 'market_cap',
    'Return over 1day': 'ret_1d',
    'Return over 1week': 'ret_1w_csv',
    'Return over 1month': 'ret_1m_csv',
    'Month Momentum': 'month_momentum',
    'Qtr Momentum': 'qtr_momentum',
    'QoQ Sales': 'qoq_sales',
    'YOY Quarterly sales growth': 'yoy_qtr_sales_growth',
    'KUSH Momentum': 'kush_momentum',
    '50 Days Momentum': 'momentum_50d',
    'Relative Vol 1W1Y': 'rel_vol_1w1y',
    'RVol1Mto1Y': 'rvol_1m_1y',
    'UP From 200 DMA': 'up_from_200dma',
    'Price to Earning': 'pe_ratio',
    'Historical PE 3Years': 'hist_pe_3y',
    'Market Cap to Sales': 'mcap_to_sales',
    'CMP to FCF': 'cmp_to_fcf',
    'EV TO SALES': 'ev_to_sales',
    'PEG Ratio': 'peg_ratio',
    'PE to profit growth': 'pe_to_profit_growth',
    'Down All Time High': 'down_ath',
    'Down from 52w high': 'down_52w_high',
    'Dividend yield': 'dividend_yield',
    'EVEBITDA': 'ev_ebitda',
    'Debt to equity': 'debt_to_equity',
    'EPS growth': 'eps_growth',
    'EPS growth 3Years': 'eps_growth_3y',
    'YOY Quarterly profit growth': 'yoy_qtr_profit_growth',
    'Profit growth': 'profit_growth',
    'Profit growth 3Years': 'profit_growth_3y',
    'Profit growth 5Years': 'profit_growth_5y',
    'Sales growth': 'sales_growth',
    'Sales growth 3Years': 'sales_growth_3y',
    'CFO to EBITDA last yr': 'cfo_to_ebitda_last',
    'CFO to EBIDTA preceding Year': 'cfo_to_ebitda_prev',
    'Cfo 5 yr to ebit': 'cfo_5y_to_ebit',
    'Return on invested capital': 'roic',
    'Return on capital employed': 'roce',
    'Average return on capital employed 3Years': 'avg_roce_3y',
    'Return on equity': 'roe',
    'Average return on equity 3Years': 'avg_roe_3y',
    'GPM latest quarter': 'gpm_latest_qtr',
    'OPM': 'opm',
    'Return over 3months': 'ret_3m_csv',
    'Return over 6months': 'ret_6m_csv',
    'Return over 1year': 'ret_1y_csv',
    'Return over 3years': 'ret_3y_csv',
    'Cwip to net block': 'cwip_to_net_block',
    'FII holding': 'fii_holding',
    'DII holding': 'dii_holding',
}


def parse_screener_csv(uploaded_file, cap_category: str) -> pd.DataFrame:
    """
    Parse a screener.in CSV file and return a cleaned DataFrame.

    Args:
        uploaded_file: Streamlit UploadedFile object or file-like
        cap_category: 'Large Cap' or 'Small Cap'

    Returns:
        Cleaned DataFrame with standardized columns.
    """
    try:
        # Read CSV — screener.in uses tab-separated values
        raw_text = uploaded_file.read().decode('utf-8') if hasattr(uploaded_file, 'read') else uploaded_file
        
        # Try tab-separated first, then comma
        if '\t' in raw_text[:500]:
            df = pd.read_csv(StringIO(raw_text), sep='\t')
        else:
            df = pd.read_csv(StringIO(raw_text))

        if df.empty:
            return pd.DataFrame()

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Rename columns to internal names where found
        rename_map = {}
        for orig, internal in SCREENER_COLUMNS_MAP.items():
            if orig in df.columns:
                rename_map[orig] = internal
        df = df.rename(columns=rename_map)

        # Ensure critical columns exist
        for col in ['name', 'nse_code', 'bse_code', 'isin_code', 'market_cap', 'industry_group', 'industry']:
            if col not in df.columns:
                df[col] = ''

        # Clean string columns
        for col in ['name', 'nse_code', 'bse_code', 'isin_code', 'industry_group', 'industry']:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'None', 'NaN', ''], '')

        # Clean numeric columns
        numeric_cols = [v for v in SCREENER_COLUMNS_MAP.values()
                        if v not in ['name', 'bse_code', 'nse_code', 'isin_code', 'industry_group', 'industry']]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('%', '').str.replace(',', '').str.strip(),
                    errors='coerce'
                )

        # Add cap category
        df['cap_category'] = cap_category

        # Remove rows without a name
        df = df[df['name'] != '']

        print(f"[OK] Parsed {len(df)} stocks from {cap_category} CSV")
        return df

    except Exception as e:
        print(f"[ERROR] Error parsing screener CSV: {e}")
        return pd.DataFrame()


# =============================================================================
# TICKER RESOLUTION
# =============================================================================

def resolve_ticker(nse_code: str, bse_code: str, isin_code: str, company_name: str = "") -> tuple:
    """
    Resolve a screener.in row to a yfinance-compatible ticker.

    Priority:
      1. NSE Code → {NSE_CODE}.NS
      2. BSE Code (numeric) → {BSE_CODE}.BO (w/ yfinance search for string name if available)
      3. ISIN Code → search via yfinance

    Returns:
        (yf_ticker, exchange, resolved_name)
        yf_ticker: e.g. 'RELIANCE.NS' or 'VALIANT.BO'
        exchange: 'NSE' or 'BSE'
        resolved_name: The ticker portion without suffix
    """
    # 1. NSE Code
    if nse_code and nse_code not in ['', 'nan', 'None']:
        clean = nse_code.strip().upper()
        return f"{clean}.NS", 'NSE', clean

    # 2. BSE Code (numeric)
    if bse_code and bse_code not in ['', 'nan', 'None']:
        clean_num = bse_code.strip()
        
        # Try to find the string ticker (e.g. VALIANT.BO) using yfinance search with company name
        if company_name and company_name not in ['', 'nan', 'None']:
            try:
                import yfinance as yf
                
                # Try full name first
                search_res = yf.Search(company_name, max_results=3).quotes
                for res in search_res:
                    symbol = res.get('symbol', '')
                    if symbol.endswith('.BO'):
                        clean_str = symbol.replace('.BO', '')
                        return symbol, 'BSE', clean_str
                        
                # If full name fails, try the first word (e.g. 'Kilburn Engg' -> 'Kilburn')
                first_word = company_name.split()[0]
                search_res_fw = yf.Search(first_word, max_results=3).quotes
                for res in search_res_fw:
                    symbol = res.get('symbol', '')
                    if symbol.endswith('.BO'):
                        clean_str = symbol.replace('.BO', '')
                        return symbol, 'BSE', clean_str
            except Exception:
                pass
                
            # If all searching fails, generate a fallback string name for UI and TradingView export
            # Example: "Kilburn Engg." -> "KILBURNENG"
            import re
            fallback_name = company_name.upper().replace(' LTD.', '').replace(' LTD', '')
            fallback_name = re.sub(r'[^A-Z0-9]', '', fallback_name)
            
            # Key trick: return numeric yf_ticker for data fetching, but string fallback for UI display
            return f"{clean_num}.BO", 'BSE', fallback_name[:12]

        # Standard fallback to numeric if no company name was provided
        return f"{clean_num}.BO", 'BSE', clean_num

    # 3. ISIN fallback — try searching with yfinance
    if isin_code and isin_code not in ['', 'nan', 'None']:
        clean = isin_code.strip().upper()
        try:
            # yfinance can look up by ISIN in some cases
            stock = yf.Ticker(clean)
            info = stock.info
            symbol = info.get('symbol', '')
            if symbol:
                return symbol, 'ISIN', clean
        except Exception:
            pass
        # Return ISIN as identifier — will show as unresolved
        return None, 'UNRESOLVED', clean

    return None, 'UNRESOLVED', ''


# =============================================================================
# ADDITIONAL METRICS
# =============================================================================

def compute_atv21(df: pd.DataFrame) -> float:
    """
    Calculate Average Trading Value over the last 21 trading days.
    ATV = sum(Close * Volume) for the last 21 days / 21
    Converted to Crores (1 Crore = 10,000,000).

    Args:
        df: DataFrame with 'close' and 'volume' columns

    Returns:
        ATV21 in Crores
    """
    if df.empty or 'close' not in df.columns or 'volume' not in df.columns:
        return 0.0

    recent = df.tail(21)
    if len(recent) == 0:
        return 0.0

    daily_value = recent['close'] * recent['volume']
    avg_value_cr = (daily_value.sum() / len(recent)) / 10000000.0
    
    return avg_value_cr


def compute_pct_above_ema(df: pd.DataFrame, ema_col: str = 'ema_21', lookback: int = 30) -> float:
    """
    Calculate what percentage of the last `lookback` trading days the close
    was above the specified EMA.

    Args:
        df: DataFrame with 'close' and ema_col columns
        ema_col: EMA column name (default: 'ema_21')
        lookback: Number of trading days to check

    Returns:
        Percentage (0-100) of days above EMA.
    """
    if df.empty or ema_col not in df.columns or 'close' not in df.columns:
        return 0.0

    recent = df.tail(lookback)
    if len(recent) == 0:
        return 0.0

    above = (recent['close'] > recent[ema_col]).sum()
    return (above / len(recent)) * 100


def classify_oneil_stage(df: pd.DataFrame) -> tuple:
    """
    Classify a stock into O'Neil/Weinstein Stage (1-4).

    Stage 1 — Basing: Price consolidating around flat 200 SMA
    Stage 2 — Advancing: Price > rising 200 SMA, 50 SMA > 200 SMA
    Stage 3 — Topping: Price starts failing 50 SMA, 200 SMA flattening
    Stage 4 — Declining: Price < falling 200 SMA

    Returns:
        (stage_number, stage_label, stage_emoji)
    """
    if df.empty or len(df) < 200:
        return (0, 'Insufficient Data', '⚫')

    close = df['close']
    last_close = close.iloc[-1]

    # Calculate SMAs
    sma_50 = close.rolling(50, min_periods=50).mean()
    sma_150 = close.rolling(150, min_periods=150).mean()
    sma_200 = close.rolling(200, min_periods=200).mean()

    if pd.isna(sma_200.iloc[-1]):
        return (0, 'Insufficient Data', '⚫')

    sma_50_now = sma_50.iloc[-1]
    sma_150_now = sma_150.iloc[-1]
    sma_200_now = sma_200.iloc[-1]

    # 200 SMA slope (is it rising or falling over last 22 trading days?)
    sma_200_slope_up = sma_200.iloc[-1] > sma_200.iloc[-22] if len(sma_200) >= 22 else False
    sma_200_slope_down = sma_200.iloc[-1] < sma_200.iloc[-22] if len(sma_200) >= 22 else False

    # 200 SMA relatively flat (changed < 2% in 22 days)
    if len(sma_200) >= 22 and not pd.isna(sma_200.iloc[-22]):
        sma_200_change = abs(sma_200.iloc[-1] - sma_200.iloc[-22]) / sma_200.iloc[-22] * 100
        sma_200_flat = sma_200_change < 2.0
    else:
        sma_200_flat = False

    # Stage 2 — Advancing (most important for buying)
    # Price > 200 SMA, 200 SMA rising, 50 SMA > 150 SMA > 200 SMA
    if (last_close > sma_200_now and
            sma_200_slope_up and
            sma_50_now > sma_150_now and
            sma_150_now > sma_200_now):
        return (2, 'Stage 2 — Advancing', '🟢')

    # Stage 4 — Declining
    # Price < 200 SMA, 200 SMA falling
    if last_close < sma_200_now and sma_200_slope_down:
        return (4, 'Stage 4 — Declining', '🔴')

    # Stage 3 — Topping
    # Price near or below 50 SMA, 200 SMA flattening or starting to roll, price was above 200 SMA recently
    if (last_close < sma_50_now and
            last_close > sma_200_now * 0.95 and
            (sma_200_flat or not sma_200_slope_up)):
        return (3, 'Stage 3 — Topping', '🟠')

    # Stage 1 — Basing
    # Price near flat 200 SMA, choppy, not clearly advancing or declining
    if sma_200_flat or (last_close > sma_200_now * 0.90 and last_close < sma_200_now * 1.10):
        return (1, 'Stage 1 — Basing', '🟡')

    # Transitional — doesn't clearly fit
    if last_close > sma_200_now:
        return (2, 'Stage 2 — Advancing', '🟢')
    else:
        return (4, 'Stage 4 — Declining', '🔴')


def compute_market_breadth(analysis_results: list, market_data: dict) -> dict:
    """
    Compute market breadth metrics across all analyzed stocks.

    Returns:
        Dict with pct_above_200sma, pct_above_50sma, pct_above_21ema, total
    """
    total = 0
    above_200 = 0
    above_50 = 0
    above_21 = 0

    for r in analysis_results:
        ticker = r.get('resolved_name', '')
        df = market_data.get(ticker)
        if df is None or df.empty:
            continue

        total += 1
        last = df.iloc[-1]

        # Check 200 SMA
        sma_200 = df['close'].rolling(200, min_periods=200).mean().iloc[-1]
        if not pd.isna(sma_200) and last['close'] > sma_200:
            above_200 += 1

        # Check 50 SMA
        sma_50 = last.get('sma_50')
        if sma_50 is not None and not pd.isna(sma_50) and last['close'] > sma_50:
            above_50 += 1

        # Check 21 EMA
        ema_21 = last.get('ema_21')
        if ema_21 is not None and not pd.isna(ema_21) and last['close'] > ema_21:
            above_21 += 1

    if total == 0:
        return {'pct_above_200sma': 0, 'pct_above_50sma': 0, 'pct_above_21ema': 0, 'total': 0}

    return {
        'pct_above_200sma': (above_200 / total) * 100,
        'pct_above_50sma': (above_50 / total) * 100,
        'pct_above_21ema': (above_21 / total) * 100,
        'total': total,
    }


def aggregate_industry_data(parsed_df: pd.DataFrame, analysis_results: list) -> pd.DataFrame:
    """
    Aggregate industry-level analytics.

    Returns:
        DataFrame with industry_group, count, avg_momentum, avg_rs, avg_qoq_sales
    """
    # Merge analysis results with parsed CSV data
    result_map = {r.get('csv_name', ''): r for r in analysis_results}

    rows = []
    for _, row in parsed_df.iterrows():
        name = row.get('name', '')
        ig = row.get('industry_group', '')
        ind = row.get('industry', '')
        kush_mom = row.get('kush_momentum', 0) or 0
        qoq_sales = row.get('qoq_sales', 0) or 0

        ar = result_map.get(name, {})
        rs = ar.get('rs_score', 0) or 0

        rows.append({
            'industry_group': ig,
            'industry': ind,
            'kush_momentum': kush_mom,
            'qoq_sales': qoq_sales,
            'rs_score': rs,
        })

    if not rows:
        return pd.DataFrame()

    df_ind = pd.DataFrame(rows)
    agg = df_ind.groupby('industry_group').agg(
        count=('industry_group', 'size'),
        avg_momentum=('kush_momentum', 'mean'),
        avg_rs=('rs_score', 'mean'),
        avg_qoq_sales=('qoq_sales', 'mean'),
    ).reset_index().sort_values('avg_rs', ascending=False)
    
    # Add Group RS Rank (1 is best)
    agg['group_rs_rank'] = agg['avg_rs'].rank(method='min', ascending=False)
    # Add Percentile Rank (100 is best)
    if len(agg) > 1:
        agg['group_rs_percentile'] = agg['avg_rs'].rank(pct=True) * 100
    else:
        agg['group_rs_percentile'] = 100

    return agg


# =============================================================================
# MAIN ANALYSIS ORCHESTRATOR
# =============================================================================

def analyze_screener_stocks(parsed_df: pd.DataFrame, progress_callback=None) -> tuple:
    """
    Run comprehensive analysis on screener.in CSV data.

    Args:
        parsed_df: Parsed DataFrame from parse_screener_csv
        progress_callback: Optional callable(current, total, message)

    Returns:
        (analysis_results, market_data_dict)
        analysis_results: List of enriched result dicts
        market_data_dict: Dict mapping resolved_name → DataFrame
    """
    results = []
    market_data = {}
    total = len(parsed_df)

    if total == 0:
        return results, market_data

    # Phase 1: Resolve tickers and fetch market data
    ticker_map = {}  # resolved_name → yf_ticker
    for i, (_, row) in enumerate(parsed_df.iterrows()):
        name = row.get('name', f'Stock_{i}')
        nse_code = row.get('nse_code', '')
        bse_code = row.get('bse_code', '')
        isin_code = row.get('isin_code', '')

        if progress_callback:
            progress_callback(i, total * 2, f"Resolving {name}...")

        yf_ticker, exchange, resolved_name = resolve_ticker(nse_code, bse_code, isin_code, name)

        if yf_ticker is None:
            # Cannot resolve — store with CSV data only
            results.append({
                'csv_name': name,
                'ticker': name,
                'resolved_name': name,
                'exchange': exchange,
                'error': 'Ticker unresolved',
                'cap_category': row.get('cap_category', ''),
                'industry_group': row.get('industry_group', ''),
                'industry': row.get('industry', ''),
                'market_cap': row.get('market_cap', 0),
                'current_price_csv': row.get('current_price', 0),
                'buy_readiness': {'score': 0, 'max_score': 10, 'label': '⚪ Not Ready', 'breakdown': {}},
            })
            continue

        ticker_map[resolved_name] = yf_ticker

        # Fetch market data
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=430)
            stock = yf.Ticker(yf_ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                # Try alternate exchange
                if yf_ticker.endswith('.NS'):
                    alt_ticker = yf_ticker.replace('.NS', '.BO')
                elif yf_ticker.endswith('.BO'):
                    alt_ticker = yf_ticker.replace('.BO', '.NS')
                else:
                    alt_ticker = None

                if alt_ticker:
                    stock = yf.Ticker(alt_ticker)
                    df = stock.history(start=start_date, end=end_date)

            if df.empty:
                results.append({
                    'csv_name': name,
                    'ticker': resolved_name,
                    'resolved_name': resolved_name,
                    'exchange': exchange,
                    'error': 'No market data',
                    'cap_category': row.get('cap_category', ''),
                    'industry_group': row.get('industry_group', ''),
                    'industry': row.get('industry', ''),
                    'market_cap': row.get('market_cap', 0),
                    'current_price_csv': row.get('current_price', 0),
                    'buy_readiness': {'score': 0, 'max_score': 10, 'label': '⚪ Not Ready', 'breakdown': {}},
                })
                continue

            df.columns = df.columns.str.lower()
            df = df.reset_index()
            df['ticker'] = resolved_name
            df = add_technical_indicators(df)
            
            # Extract Sparkline Data (Last 60 Days Close)
            df['sparkline'] = df['close'].rolling(1).min() # placeholder, actual extraction below
            
            market_data[resolved_name] = df

        except Exception as e:
            print(f"[ERROR] Error fetching {yf_ticker}: {e}")
            results.append({
                'csv_name': name,
                'ticker': resolved_name,
                'resolved_name': resolved_name,
                'exchange': exchange,
                'error': str(e),
                'cap_category': row.get('cap_category', ''),
                'industry_group': row.get('industry_group', ''),
                'industry': row.get('industry', ''),
                'market_cap': row.get('market_cap', 0),
                'current_price_csv': row.get('current_price', 0),
                'buy_readiness': {'score': 0, 'max_score': 10, 'label': '⚪ Not Ready', 'breakdown': {}},
            })
            continue

    # Phase 2: Calculate RS scores for all tickers at once
    if progress_callback:
        progress_callback(total, total * 2, "Calculating RS scores...")
    rs_data = calculate_all_rs_scores(market_data) if market_data else {}

    # Phase 3: Analyze each resolved ticker
    for i, (_, row) in enumerate(parsed_df.iterrows()):
        name = row.get('name', f'Stock_{i}')
        nse_code = row.get('nse_code', '')
        bse_code = row.get('bse_code', '')

        _, exchange, resolved_name = resolve_ticker(
            nse_code, bse_code, row.get('isin_code', ''), name
        )

        # Skip if already added as error
        if any(r.get('csv_name') == name and 'error' in r for r in results):
            continue

        df = market_data.get(resolved_name)
        if df is None or df.empty:
            continue

        if progress_callback:
            progress_callback(total + i, total * 2, f"Analyzing {name}...")

        # RS Score
        rs_info = rs_data.get(resolved_name, {})
        rs_score = rs_info.get('rs_score', 0) or 0

        # Trend State
        trend_state = determine_trend_state(df)

        # 52-Week High Hits
        hits_5d = count_52w_high_hits(df, 5)
        hits_1m = count_52w_high_hits(df, 21)
        hits_3m = count_52w_high_hits(df, 63)

        # Distance from 52W High
        last_row = df.iloc[-1]
        dist_from_high = last_row.get('dist_from_52w_high', None)
        if pd.isna(dist_from_high):
            dist_from_high = None
            
        # Extract Sparkline
        sparkline = df['close'].tail(60).tolist()

        # Returns
        returns = calculate_period_returns(df)

        # Minervini Trend Template
        minervini = check_minervini_trend_template(df, rs_score)

        # VCP
        vcp = detect_vcp_pattern(df)

        # High Tight Flag (Power Play)
        from technical_indicators import detect_high_tight_flag, detect_ants_momentum, calculate_hv1_avwap, detect_ema_crossback, detect_reversal_extension
        htf_data = detect_high_tight_flag(df, min_thrust_pct=70.0)
        is_htf = htf_data['is_htf']
        
        # Additional Deepvue/Kell Setups
        # Stage condition check must be available for crossback
        stage_num, stage_label, stage_emoji = classify_oneil_stage(df)
        
        is_ants = detect_ants_momentum(df, lookback=15, min_up_days=12)
        hv1_avwap = calculate_hv1_avwap(df, lookback=252)
        close_price_raw = last_row.get('close', 0)
        is_hv1_defended = bool((hv1_avwap > 0) and (close_price_raw > hv1_avwap) and (abs(close_price_raw - hv1_avwap)/hv1_avwap < 0.05))
        is_crossback = (stage_num == 2) and detect_ema_crossback(df, ema_period=10, max_days_below=5)
        is_reversal_ext = detect_reversal_extension(df, ema_period=10, extension_threshold=12.0)

        # ATR State
        atr_ratio = last_row.get('atr_ratio', 1.0)
        if pd.isna(atr_ratio):
            atr_ratio = 1.0
        atr_state = get_atr_state(atr_ratio)

        # Volume Dry-Up
        vol_dryup = check_volume_dryup(df)

        # EMA Stack
        ema_stack = check_ema_stack(df)

        # % Time Above 21 EMA in Last 30 Days
        pct_above_21ema = compute_pct_above_ema(df, 'ema_21', 30)

        # Average Trading Value (21 Days) and Safe Position
        atv21_cr = compute_atv21(df)
        safe_position_cr = atv21_cr * 0.05  # Max 5% of ATV21

        # Fundamental Data for Scoring
        fundamentals = {
            'eps_growth': row.get('eps_growth', 0),
            'sales_growth': row.get('sales_growth', 0),
            'roe': row.get('roe', 0),
            'opm': row.get('opm', 0),
            'debt_to_equity': row.get('debt_to_equity', 0),
        }
        
        fundamental_score, fund_breakdown = calculate_fundamental_score(fundamentals)

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

        # Current price from yfinance
        close_price = last_row.get('close', 0)

        results.append({
            'csv_name': name,
            'ticker': resolved_name,
            'resolved_name': resolved_name,
            'exchange': exchange,
            'cap_category': row.get('cap_category', ''),
            'industry_group': row.get('industry_group', ''),
            'industry': row.get('industry', ''),
            'market_cap': row.get('market_cap', 0),
            'current_price_csv': row.get('current_price', 0),
            'close_price': close_price,
            'rs_score': rs_score,
            'trend_state': trend_state,
            'atv21_cr': atv21_cr,
            'safe_position_cr': safe_position_cr,
            'stage_num': stage_num,
            'stage_label': stage_label,
            'stage_emoji': stage_emoji,
            'hits_52w_5d': hits_5d,
            'hits_52w_1m': hits_1m,
            'hits_52w_3m': hits_3m,
            'dist_from_high': dist_from_high,
            'returns': returns,
            'sparkline': sparkline,
            'minervini': minervini,
            'vcp': vcp,
            'htf': is_htf,
            'atr_state': atr_state,
            'atr_ratio': atr_ratio,
            'vol_dryup': vol_dryup,
            'ema_stack': ema_stack,
            'pct_above_21ema_30d': pct_above_21ema,
            'is_ants': is_ants,
            'is_hv1_defended': is_hv1_defended,
            'is_crossback': is_crossback,
            'is_reversal_ext': is_reversal_ext,
            'buy_readiness': buy_readiness,
            # CSV metrics passthrough
            'kush_momentum': row.get('kush_momentum', 0),
            'qoq_sales': row.get('qoq_sales', 0),
            'yoy_qtr_sales_growth': row.get('yoy_qtr_sales_growth', 0),
            'roe': row.get('roe', 0),
            'opm': row.get('opm', 0),
            'debt_to_equity': row.get('debt_to_equity', 0),
            'fii_holding': row.get('fii_holding', 0),
            'dii_holding': row.get('dii_holding', 0),
            'down_52w_high_csv': row.get('down_52w_high', 0),
            'pe_ratio': row.get('pe_ratio', 0),
            'peg_ratio': row.get('peg_ratio', 0),
            'profit_growth': row.get('profit_growth', 0),
            'sales_growth': row.get('sales_growth', 0),
        })

    # Map Group RS Rank to results
    industry_agg = aggregate_industry_data(parsed_df, results)
    group_rs_map = dict(zip(industry_agg['industry_group'], industry_agg['group_rs_percentile']))
    
    for r in results:
        ig = r.get('industry_group', '')
        r['group_rs_percentile'] = group_rs_map.get(ig, 0)

    # Sort by buy readiness score descending
    results.sort(key=lambda x: x.get('buy_readiness', {}).get('score', 0), reverse=True)

    return results, market_data


if __name__ == '__main__':
    print("Screener Engine — run via pages/screener_analytics.py")
