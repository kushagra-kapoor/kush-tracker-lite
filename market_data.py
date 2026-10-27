# Market Data Fetcher for Kush Tracker

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from config import DATA_SETTINGS, BENCHMARK_TICKER


def get_ticker_symbol(ticker: str) -> str:
    """
    Convert ticker to yfinance format for Indian stocks.
    
    Args:
        ticker: Raw ticker symbol (e.g., 'RELIANCE')
    
    Returns:
        yfinance compatible ticker (e.g., 'RELIANCE.NS')
    """
    import re
    ticker = ticker.strip().upper()
    
    # Symbol Sanitization: Strip known suffixes that break Yahoo Finance
    ticker = re.sub(r'-(ST|SM|BE|EQ|P1|P2|P3)$', '', ticker)
    
    # Auto-resolve BSE numerical codes to Yahoo Finance alphabetical symbols via Screener
    if ticker.isdigit():
        import json, os
        cache_file = 'bse_yf_mapping.json'
        
        # Check cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    mapping = json.load(f)
                    if ticker in mapping:
                        return mapping[ticker]
            except:
                pass
                
        # Resolve via Screener
        try:
            import requests
            from bs4 import BeautifulSoup
            url = f"https://www.screener.in/company/{ticker}/"
            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                links = soup.find_all('a', href=True)
                for l in links:
                    if 'bseindia.com/stock-share-price' in l['href']:
                        # Example format: https://www.bseindia.com/stock-share-price/zelio-e-mobility-ltd/ZELIO/544563/
                        parts = [p for p in l['href'].split('/') if p]
                        if ticker in parts:
                            idx = parts.index(ticker)
                            if idx > 0:
                                yf_ticker = f"{parts[idx-1]}.BO"
                                
                                # Save to cache
                                mapping = {}
                                if os.path.exists(cache_file):
                                    with open(cache_file, 'r') as f:
                                        mapping = json.load(f)
                                mapping[ticker] = yf_ticker
                                with open(cache_file, 'w') as f:
                                    json.dump(mapping, f)
                                    
                                return yf_ticker
        except Exception as e:
            print(f"Error auto-resolving BSE code {ticker}: {e}")
        
    if not ticker.endswith('.NS') and not ticker.startswith('^') and not ticker.endswith('.BO'):
        return f"{ticker}.NS"
    return ticker


def fetch_stock_data(ticker: str, days: int = None) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a stock across .NS -> .BO -> nselib SME waterfall.
    Requires >= 5 trading bars from Yahoo Finance so 1-row SME quote stubs
    (like ANONDITA-SM.NS) automatically fall through to nselib full history.
    """
    if days is None:
        days = DATA_SETTINGS['HISTORY_DAYS']
    
    yf_ticker_primary = get_ticker_symbol(ticker)
    raw_ticker = ticker.strip().upper()
    
    # Generate candidate tickers to try in order
    candidates = [
        yf_ticker_primary,                                                # Sanitized .NS
        f"{raw_ticker}.NS" if not raw_ticker.endswith('.NS') else raw_ticker, # Original .NS
        yf_ticker_primary.replace('.NS', '.BO'),                          # Sanitized .BO
        f"{raw_ticker}.BO" if not raw_ticker.endswith('.BO') else raw_ticker  # Original .BO
    ]
    
    # Remove duplicates while preserving order
    candidates = list(dict.fromkeys(candidates))
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 30)  # Extra buffer for trading days
    
    df = pd.DataFrame()
    best_stub_df = pd.DataFrame()
    successful_ticker = None
    
    for cand in candidates:
        try:
            stock = yf.Ticker(cand)
            temp_df = stock.history(start=start_date, end=end_date)
            
            if not temp_df.empty and 'Close' in temp_df.columns:
                temp_df = temp_df.dropna(subset=['Close'])
                if len(temp_df) >= 5:
                    df = temp_df
                    successful_ticker = cand
                    break # Found valid multi-bar history, stop trying candidates
                elif not temp_df.empty and best_stub_df.empty:
                    best_stub_df = temp_df
        except Exception:
            continue
            
    if df.empty:
        print(f"[Fallback] <5 bars on YF for {ticker}. Trying nselib SME/Equities fallback...")
        try:
            from nselib import capital_market
            clean_sym = ticker.replace('-SM', '').replace('-ST', '').replace('.NS', '').replace('.BO', '').strip()
            from_str = start_date.strftime("%d-%m-%Y")
            to_str = end_date.strftime("%d-%m-%Y")
            
            nselib_df = capital_market.price_volume_and_deliverable_position_data(
                symbol=clean_sym, 
                from_date=from_str, 
                to_date=to_str
            )
            
            if nselib_df is not None and not nselib_df.empty:
                nselib_df = nselib_df.rename(columns={
                    'Date': 'Date',
                    'OpenPrice': 'Open',
                    'HighPrice': 'High',
                    'LowPrice': 'Low',
                    'ClosePrice': 'Close',
                    'TotalTradedQuantity': 'Volume'
                })
                
                # Convert Date to datetime, setting as index to mimic yfinance
                nselib_df['Date'] = pd.to_datetime(nselib_df['Date'])
                nselib_df = nselib_df.set_index('Date')
                if nselib_df.index.tz is not None:
                    nselib_df.index = nselib_df.index.tz_localize(None)
                nselib_df.index = nselib_df.index.normalize()
                
                # Convert string numbers to float
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in nselib_df.columns:
                        nselib_df[col] = pd.to_numeric(nselib_df[col].astype(str).str.replace(',', ''), errors='coerce')
                
                nselib_df = nselib_df.dropna(subset=['Close'])
                nselib_df = nselib_df[~nselib_df.index.duplicated(keep='last')]
                nselib_df = nselib_df.sort_index()
                
                if not nselib_df.empty:
                    df = nselib_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    print(f"[OK] Fetched {len(df)} bars for {ticker} via nselib.")
        except Exception as e:
            print(f"[WARN] nselib fallback failed for {ticker}: {e}")

    if df.empty and not best_stub_df.empty:
        df = best_stub_df
            
    if df.empty:
        print(f"[WARN] No data found for {ticker} across all fallbacks.")
        return pd.DataFrame()
    
    # Clean column names
    df.columns = df.columns.str.lower()
    if 'date' not in df.columns:
        df = df.reset_index()
        if 'Date' in df.columns and 'date' not in df.columns:
            df = df.rename(columns={'Date': 'date'})
    df['ticker'] = ticker # Keep original ticker in df for app consistency
    
    return df


def fetch_screener_fallback(ticker: str) -> tuple:
    import requests
    from bs4 import BeautifulSoup
    import json
    
    # Clean ticker (remove -SM, -ST, .NS, etc)
    search_q = ticker.replace('-SM', '').replace('-ST', '').replace('.NS', '').replace('.BO', '')
    if search_q.isdigit():
        import os
        try:
            if os.path.exists('bse_mapping.json'):
                with open('bse_mapping.json', 'r') as f:
                    bse_map = json.load(f)
                    rev_map = {str(v): k for k, v in bse_map.items()}
                    if search_q in rev_map:
                        search_q = rev_map[search_q]
        except Exception:
            pass

    search_url = f"https://www.screener.in/api/company/search/?q={search_q}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    mcap = None
    price = None
    
    try:
        res = requests.get(search_url, headers=headers, timeout=5)
        if res.status_code == 200 and res.json():
            company_url = res.json()[0]['url']
            details_url = f"https://www.screener.in{company_url}"
            det_res = requests.get(details_url, headers=headers, timeout=5)
            if det_res.status_code == 200:
                soup = BeautifulSoup(det_res.content, 'html.parser')
                for li in soup.find_all('li', class_='flex'):
                    name_span = li.find('span', class_='name')
                    if name_span:
                        if 'Market Cap' in name_span.text:
                            val_span = li.find('span', class_='number')
                            if val_span:
                                mcap = float(val_span.text.replace(',', ''))
                        elif 'Current Price' in name_span.text:
                            val_span = li.find('span', class_='number')
                            if val_span:
                                price = float(val_span.text.replace(',', ''))
    except Exception as e:
        print(f"    Screener error for {ticker}: {e}")
        
    df = pd.DataFrame()
    if price:
        from datetime import datetime
        df = pd.DataFrame({
            'open': [price],
            'high': [price],
            'low': [price],
            'close': [price],
            'volume': [0],
            'ticker': [ticker], 
            'date': [datetime.now()]
        })
    
    return df, mcap


_cached_total_market_tickers = None
_cached_total_market_industries = None
_total_market_cache_date = None


def get_cache_staleness(key: str = "NIFTY_TOTAL_MARKET") -> int:
    """Returns number of days old the newest cache file is. 0 means fresh today. -1 means no cache."""
    import os, glob
    from datetime import datetime
    
    cache_dir = '.cache'
    files = glob.glob(os.path.join(cache_dir, f"{key}_*.csv"))
    if not files:
        return -1
        
    newest_date = None
    for f in files:
        basename = os.path.basename(f)
        try:
            date_str = basename.replace(f"{key}_", "").replace(".csv", "")
            file_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            if newest_date is None or file_date > newest_date:
                newest_date = file_date
        except Exception:
            continue
            
    if newest_date is None:
        return -1
        
    today = datetime.now().date()
    return (today - newest_date).days

def fetch_and_cache_csv(key: str, show_progress: bool = True):
    import os
    import json
    import time
    import glob
    import requests
    import pandas as pd
    from io import StringIO
    from datetime import datetime

    cache_dir = '.cache'
    os.makedirs(cache_dir, exist_ok=True)
    try:
        now = time.time()
        for f in glob.glob(os.path.join(cache_dir, '*.csv')):
            if os.path.isfile(f) and os.stat(f).st_mtime < now - 3 * 86400:
                os.remove(f)
    except Exception:
        pass
        
    try:
        with open('index_urls.json', 'r') as f:
            urls = json.load(f)
    except Exception:
        urls = {}
    url = urls.get(key)
    if not url:
        raise ValueError(f"Key {key} not found in index_urls.json")

    today_str = datetime.now().strftime('%Y-%m-%d')
    cache_file = os.path.join(cache_dir, f"{key}_{today_str}.csv")
    
    if os.path.exists(cache_file):
        try:
            return pd.read_csv(cache_file)
        except Exception:
            pass

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Referer': 'https://www.nseindia.com/'
    })
    try:
        session.get('https://www.nseindia.com', timeout=10)
        time.sleep(1)
    except Exception:
        pass
        
    try:
        response = session.get(url, timeout=30)
        if response.status_code != 200 or 'Company Name' not in response.text:
            raise Exception("Invalid response from NSE or blocked.")
            
        df = pd.read_csv(StringIO(response.text))
        if df.empty or len(df.columns) < 2:
            raise Exception("Downloaded CSV is empty or malformed.")
            
        temp_file = f"{cache_file}.tmp.{os.getpid()}"
        df.to_csv(temp_file, index=False)
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except Exception:
                pass
        os.rename(temp_file, cache_file)
        return df
        
    except Exception as e:
        if show_progress:
            print(f"  [ERROR] Network fetch failed for {key}: {e}")
        try:
            available_caches = glob.glob(os.path.join(cache_dir, f"{key}_*.csv"))
            if available_caches:
                available_caches.sort(reverse=True)
                if show_progress:
                    print(f"  [INFO] Falling back to previous cache: {available_caches[0]}")
                return pd.read_csv(available_caches[0])
        except Exception:
            pass
        raise e

def _process_ticker_df(df, return_industry_map=False):
    ticker_col = None
    industry_col = None
    for col in df.columns:
        if 'symbol' in col.lower():
            ticker_col = col
        if 'industry' in col.lower():
            industry_col = col
    
    if ticker_col is None:
        ticker_col = df.columns[2]
        
    tickers = df[ticker_col].dropna().astype(str).str.strip().str.upper().tolist()
    tickers = [f"{t}.NS" for t in tickers if t and len(t) <= 20 and (t.isalnum() or '-' in t or '&' in t)]
    
    if not return_industry_map:
        return tickers
        
    industry_map = {}
    if industry_col is not None:
        for _, row in df.iterrows():
            raw_t = str(row[ticker_col]).strip().upper()
            if raw_t and len(raw_t) <= 20 and (raw_t.isalnum() or '-' in raw_t or '&' in raw_t):
                t_ns = f"{raw_t}.NS"
                industry_map[t_ns] = str(row[industry_col]).strip()
    return tickers, industry_map

def fetch_nifty_total_market_tickers(show_progress: bool = True, return_industry_map: bool = False):
    global _cached_total_market_tickers, _cached_total_market_industries, _total_market_cache_date
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    if _cached_total_market_tickers and _total_market_cache_date == today:
        if return_industry_map:
            return _cached_total_market_tickers, _cached_total_market_industries
        return _cached_total_market_tickers
    
    if show_progress:
        print(f"  -> Fetching NIFTY Total Market constituents from NSE...")
    
    try:
        df = fetch_and_cache_csv("NIFTY_TOTAL_MARKET", show_progress)
        result = _process_ticker_df(df, return_industry_map=True)
        tickers, industry_map = result
        
        if len(tickers) >= 500:
            if show_progress:
                print(f"  [OK] Fetched {len(tickers)} Total Market constituents")
            _cached_total_market_tickers = tickers
            _cached_total_market_industries = industry_map
            _total_market_cache_date = today
            
            if return_industry_map:
                return tickers, industry_map
            return tickers
        else:
            raise ValueError(f"Only {len(tickers)} tickers found, expected 700+")
            
    except Exception as e:
        if show_progress:
            print(f"  [ERROR] Failed to fetch Nifty Total Market: {e}")
        
    if return_industry_map:
        return _cached_total_market_tickers or [], _cached_total_market_industries or {}
    return _cached_total_market_tickers or []

def fetch_full_india_universe_tickers(portfolio_tickers: list = None, show_progress: bool = False) -> list:
    """
    Returns the complete 3-Tier Indian Equity Universe (~2,750+ stocks):
      Tier 1: NIFTY Total Market (~750 Liquid Core)
      Tier 2: Broad NSE Small/Micro-Cap Universe from tickers.txt & historical_prices_matrix.pkl (~2,000+ stocks)
      Tier 3: Active Portfolio & Watchlist SME (-SM/-ST) & BSE (.BO) holdings
    """
    import os
    universe = set(fetch_nifty_total_market_tickers(show_progress=show_progress))
    
    if os.path.exists('tickers.txt'):
        try:
            with open('tickers.txt', 'r') as f:
                for line in f:
                    t = line.strip().upper()
                    if t and not t.startswith('#'):
                        if not t.endswith('.NS') and not t.endswith('.BO') and not t.startswith('^'):
                            universe.add(f"{t}.NS")
                        else:
                            universe.add(t)
        except Exception:
            pass

    if os.path.exists('historical_prices_matrix.pkl'):
        try:
            import pickle
            with open('historical_prices_matrix.pkl', 'rb') as f:
                mat = pickle.load(f)
            if isinstance(mat, pd.DataFrame) and isinstance(mat.columns, pd.MultiIndex):
                for col_t in mat.columns.get_level_values(0).unique():
                    s_t = str(col_t)
                    if s_t.endswith('.NS') or s_t.endswith('.BO'):
                        universe.add(s_t)
        except Exception:
            pass

    if portfolio_tickers:
        for pt in portfolio_tickers:
            if pt and pt != 'BENCHMARK':
                universe.add(get_ticker_symbol(str(pt)))

    return sorted(list(universe))

def fetch_nifty500_tickers(show_progress: bool = True):
    try:
        df = fetch_and_cache_csv("NIFTY_500", show_progress)
        return _process_ticker_df(df, return_industry_map=False)
    except Exception as e:
        if show_progress:
            print(f"  [ERROR] Failed to fetch Nifty 500: {e}")
        return []


def fetch_benchmark_data(days: int = None) -> pd.DataFrame:
    """
    Fetch historical data for the benchmark (Nifty 500).
    
    Args:
        days: Number of days of history
    
    Returns:
        DataFrame with benchmark OHLCV data
    """
    if days is None:
        days = DATA_SETTINGS['HISTORY_DAYS']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 30)
    
    try:
        benchmark = yf.Ticker(BENCHMARK_TICKER)
        df = benchmark.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"[WARN] No data found for benchmark {BENCHMARK_TICKER}")
            return pd.DataFrame()
            
        if 'Close' in df.columns:
            df = df.dropna(subset=['Close'])
            
        if df.empty:
            return pd.DataFrame()
        
        df.columns = df.columns.str.lower()
        df = df.reset_index()
        df['ticker'] = 'BENCHMARK'
        
        return df
    except Exception as e:
        print(f"[ERROR] Error fetching benchmark data: {e}")
        return pd.DataFrame()


def fetch_all_holdings_data(tickers: list) -> dict:
    """
    Fetch market data for all holdings using batched yfinance downloads,
    with automatic waterfall fallback to .BO and nselib SME history (for -SM/-ST/BSE stocks)
    and write-through injection into historical_prices_matrix.pkl.
    """
    print(f"[Holdings] Batch fetching market data for {len(tickers)} holdings...")
    data = {}
    if not tickers:
        return data
        
    yf_tickers = [get_ticker_symbol(t) for t in tickers]
    days = DATA_SETTINGS.get('HISTORY_DAYS', 252)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 30)
    matrix_injections = {}
    local_matrix_df = None

    def _extract_from_local_matrix(yf_sym: str, raw_ticker: str) -> pd.DataFrame:
        nonlocal local_matrix_df
        import os, pickle
        if local_matrix_df is None and os.path.exists('historical_prices_matrix.pkl'):
            try:
                with open('historical_prices_matrix.pkl', 'rb') as f:
                    local_matrix_df = pickle.load(f)
            except Exception:
                local_matrix_df = pd.DataFrame()
        if local_matrix_df is None or local_matrix_df.empty or not isinstance(local_matrix_df.columns, pd.MultiIndex):
            return pd.DataFrame()
        lvl0 = local_matrix_df.columns.get_level_values(0).unique()
        clean_base = raw_ticker.replace('-SM', '').replace('-ST', '').replace('.NS', '').replace('.BO', '').strip()
        for cand in [yf_sym, f"{clean_base}.NS", f"{clean_base}.BO", raw_ticker]:
            if cand in lvl0:
                sub = local_matrix_df[cand].dropna(how='all').copy()
                if len(sub) >= 5:
                    sub.columns = [str(c).lower() for c in sub.columns]
                    sub = sub.dropna(subset=['close']).reset_index()
                    if 'Date' in sub.columns and 'date' not in sub.columns:
                        sub = sub.rename(columns={'Date': 'date'})
                    elif 'index' in sub.columns and 'date' not in sub.columns:
                        sub = sub.rename(columns={'index': 'date'})
                    sub['ticker'] = raw_ticker
                    return sub
        return pd.DataFrame()
    
    try:
        import yfinance as yf
        df_batch = yf.download(
            yf_tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            auto_adjust=True,
            threads=True,
            progress=False
        )
        
        for i, ticker in enumerate(tickers):
            yf_t = yf_tickers[i]
            stock_df = pd.DataFrame()
            
            try:
                if isinstance(df_batch.columns, pd.MultiIndex):
                    if 'Ticker' in df_batch.columns.names:
                        stock_df = df_batch.xs(yf_t, axis=1, level='Ticker').copy()
                    else:
                        if yf_t in df_batch.columns.levels[0]:
                            stock_df = df_batch[yf_t].copy()
                        elif yf_t in df_batch.columns.levels[1]:
                            stock_df = df_batch.xs(yf_t, axis=1, level=1).copy()
                else:
                    stock_df = df_batch.copy()
            except Exception:
                pass
                
            if not stock_df.empty:
                if 'Close' in stock_df.columns:
                    stock_df = stock_df.dropna(subset=['Close'])
                elif 'close' in stock_df.columns:
                    stock_df = stock_df.dropna(subset=['close'])
                    
            if not stock_df.empty and len(stock_df) >= 5:
                stock_df.columns = stock_df.columns.str.lower()
                stock_df = stock_df.reset_index()
                if 'Date' in stock_df.columns and 'date' not in stock_df.columns:
                    stock_df = stock_df.rename(columns={'Date': 'date'})
                stock_df['ticker'] = ticker
                data[ticker] = stock_df
                matrix_injections[yf_t] = stock_df
                print(f"  -> {ticker} OK ({len(stock_df)} days)")
            else:
                # Waterfall Step 2: Check historical_prices_matrix.pkl first (instant 250+ bar SME/BSE history), then .BO / nselib
                matrix_fallback_df = _extract_from_local_matrix(yf_t, ticker)
                if not matrix_fallback_df.empty and len(matrix_fallback_df) >= 20:
                    data[ticker] = matrix_fallback_df
                    print(f"  -> {ticker} OK via local matrix ({len(matrix_fallback_df)} days)")
                else:
                    print(f"  -> {ticker} (<5 bars in YF .NS batch) - running .BO / nselib full-history waterfall...")
                    waterfall_df = fetch_stock_data(ticker, days=days)
                    if not waterfall_df.empty and len(waterfall_df) >= 5:
                        data[ticker] = waterfall_df
                        matrix_injections[yf_t] = waterfall_df
                        print(f"  -> {ticker} OK via waterfall ({len(waterfall_df)} days)")
                    elif not matrix_fallback_df.empty:
                        data[ticker] = matrix_fallback_df
                        print(f"  -> {ticker} OK via local matrix ({len(matrix_fallback_df)} days)")
                    else:
                        print(f"  -> {ticker} attempting Screener fallback...")
                        sme_df, mcap = fetch_screener_fallback(ticker)
                        data[ticker] = sme_df if not sme_df.empty else waterfall_df
                        if mcap:
                            from database import get_all_fundamentals_cache, save_fundamentals_cache
                            fundas = get_all_fundamentals_cache()
                            industry = fundas.get(ticker, {}).get('industry', 'Unknown')
                            save_fundamentals_cache([(ticker, 0.0, 0.0, 0.0, industry, mcap * 10000000.0)])
                
    except Exception as e:
        print(f"[ERROR] Error in batch fetch: {e}")
        for ticker in tickers:
            yf_t = get_ticker_symbol(ticker)
            s_df = fetch_stock_data(ticker, days=days)
            data[ticker] = s_df
            if not s_df.empty and len(s_df) >= 5:
                matrix_injections[yf_t] = s_df

    # Persist all resolved portfolio histories (including SME nselib & .BO) into historical_prices_matrix.pkl
    if matrix_injections:
        try:
            from price_history_manager import inject_external_history_into_matrix
            inject_external_history_into_matrix(matrix_injections)
        except Exception as e:
            print(f"[WARN] Matrix injection warning: {e}")
            
    # Also fetch benchmark
    print(f"  -> Fetching benchmark ({BENCHMARK_TICKER})...", end=" ")
    benchmark_df = fetch_benchmark_data()
    if not benchmark_df.empty:
        data['BENCHMARK'] = benchmark_df
        print(f"OK ({len(benchmark_df)} days)")
    else:
        print("EMPTY (no data)")
        
    return data


def get_latest_price(ticker: str) -> float:
    """
    Get the latest close price for a ticker.
    
    Args:
        ticker: Stock ticker
    
    Returns:
        Latest close price or None
    """
    yf_ticker = get_ticker_symbol(ticker)
    try:
        stock = yf.Ticker(yf_ticker)
        info = stock.info
        return info.get('previousClose') or info.get('regularMarketPrice')
    except:
        return None


def get_52_week_high(df: pd.DataFrame) -> float:
    """
    Calculate 52-week high from historical data.
    
    Args:
        df: DataFrame with 'high' column
    
    Returns:
        52-week high price
    """
    if df.empty or 'high' not in df.columns:
        return None
    
    # Take last 252 trading days (approximately 1 year)
    recent = df.tail(252)
    return recent['high'].max()


if __name__ == '__main__':
    # Test with a sample ticker
    print("Testing market data fetcher...")
    df = fetch_stock_data('RELIANCE')
    print(f"Fetched {len(df)} rows for RELIANCE")
    print(df.tail())
