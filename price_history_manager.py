import pandas as pd
import yfinance as yf
import os
import json
from datetime import datetime, timedelta
import time

CACHE_FILE = "historical_prices_matrix.pkl"
DELISTED_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "delisted_tickers.json")
_LAGGING_CHECKED_TODAY = set()

def load_delisted_tickers() -> set:
    """Loads permanently delisted or invalid tickers to avoid spamming Yahoo Finance on boot."""
    if os.path.exists(DELISTED_CACHE_FILE):
        try:
            with open(DELISTED_CACHE_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()

def save_delisted_tickers(delisted_set: set):
    """Persists delisted tickers to JSON cache."""
    try:
        os.makedirs(os.path.dirname(DELISTED_CACHE_FILE), exist_ok=True)
        with open(DELISTED_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(list(delisted_set)), f, indent=2)
    except Exception:
        pass

SYNC_TRACKER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matrix_sync_tracker.json")

def load_sync_tracker() -> dict:
    if os.path.exists(SYNC_TRACKER_FILE):
        try:
            with open(SYNC_TRACKER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def record_sync_complete(cache_file: str, market: str):
    tracker = load_sync_tracker()
    base_key = os.path.basename(cache_file)
    if base_key not in tracker:
        tracker[base_key] = {}
    today_str = datetime.now().strftime("%Y-%m-%d")
    tracker[base_key][market.upper()] = {
        "last_sync_date": today_str,
        "sync_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        os.makedirs(os.path.dirname(SYNC_TRACKER_FILE), exist_ok=True)
        with open(SYNC_TRACKER_FILE, "w", encoding="utf-8") as f:
            json.dump(tracker, f, indent=2)
    except Exception as e:
        print(f"[PriceHistoryManager] Sync tracker notice: {e}")

def has_synced_today(cache_file: str, market: str) -> bool:
    tracker = load_sync_tracker()
    base_key = os.path.basename(cache_file)
    mkt_entry = tracker.get(base_key, {}).get(market.upper(), {})
    last_date = mkt_entry.get("last_sync_date", "")
    today_str = datetime.now().strftime("%Y-%m-%d")
    return last_date == today_str

def get_effective_last_date(cached_df: pd.DataFrame, target_tickers: list = None, market: str = 'IN') -> pd.Timestamp:
    """
    Finds the newest date where the requested market's tickers actually have data.
    Decouples US stocks from Indian stocks even in the same matrix file.
    """
    if cached_df is None or cached_df.empty:
        return None
    try:
        if isinstance(cached_df.columns, pd.MultiIndex) and 'Close' in cached_df.columns.get_level_values(1):
            close_df = cached_df.xs('Close', level=1, axis=1)
        else:
            close_df = cached_df
    except Exception:
        close_df = pd.DataFrame()
        
    if close_df.empty:
        return cached_df.index.max()
        
    valid_existing = [t for t in target_tickers if t in close_df.columns] if target_tickers else []
    if not valid_existing:
        if market.upper() in ['IN', 'INDIA']:
            valid_existing = [c for c in close_df.columns if str(c).endswith('.NS') or str(c).endswith('.BO')]
        else:
            valid_existing = [c for c in close_df.columns if not (str(c).endswith('.NS') or str(c).endswith('.BO'))]
            
    if not valid_existing:
        return cached_df.index.max()
        
    counts = close_df[valid_existing].notna().sum(axis=1)
    threshold = max(1, int(len(valid_existing) * 0.15))
    valid_dates = counts[counts >= threshold].index
    if len(valid_dates) > 0:
        return valid_dates.max()
    return cached_df.index.max()

def get_matrix_last_date(cache_file=CACHE_FILE, market=None) -> str:
    """
    Returns the newest date string (YYYY-MM-DD) found in the cache matrix for the given market.
    Runs in < 0.05s from disk.
    """
    if os.path.exists(cache_file):
        try:
            df = pd.read_pickle(cache_file)
            if not df.empty and df.index is not None:
                if market:
                    eff_d = get_effective_last_date(df, market=market)
                    if eff_d is not None:
                        return eff_d.strftime('%Y-%m-%d')
                return df.index.max().strftime('%Y-%m-%d')
        except Exception:
            pass
    return None

def is_new_session_possible(last_date=None, now=None, market='IN') -> bool:
    """Simplified compatibility wrapper: returns True if today != last_synced_date."""
    return not has_synced_today(CACHE_FILE, market)

def _safe_yf_download(tickers, max_retries=3, chunk_size=500, **kwargs):
    """
    yfinance multi-threading throws 'RuntimeError: dictionary changed size' occasionally.
    This safely chunks the download and retries if the race condition occurs, keeping it fast.
    Aborts gracefully on Yahoo Finance rate limits (HTTP 429) to prevent UI freezes.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
        
    kwargs['threads'] = True  # Re-enable multi-threading for speed!
    all_dfs = []
    
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        for attempt in range(max_retries):
            try:
                df = yf.download(chunk, **kwargs)
                if not df.empty:
                    all_dfs.append(df)
                break
            except Exception as e:
                err_str = str(e).lower()
                if "rate limit" in err_str or "too many requests" in err_str:
                    print(f"[PriceHistoryManager] ⚠️ Yahoo Finance rate limit active: {e}. Aborting network sync and using cached matrix.")
                    return pd.concat(all_dfs, axis=1) if all_dfs else pd.DataFrame()
                if "dictionary changed size" in err_str and attempt < max_retries - 1:
                    time.sleep(1.5)
                    continue
                break
                
    if all_dfs:
        if len(all_dfs) == 1:
            return all_dfs[0]
        # Concat horizontally (columns) since different tickers
        return pd.concat(all_dfs, axis=1)
    return pd.DataFrame()

def _ensure_multiindex(df, tickers):
    """
    yfinance drops the MultiIndex if only 1 ticker is downloaded.
    This ensures it always matches the (Ticker, PriceType) format used globally.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
        
    if df.empty:
        return df
        
    if not isinstance(df.columns, pd.MultiIndex):
        if len(tickers) == 1:
            # Single ticker case
            ticker = list(tickers)[0]
            # yfinance returns flat columns like 'Open', 'High', 'Close'
            new_cols = pd.MultiIndex.from_tuples([(ticker, c) for c in df.columns])
            df.columns = new_cols
            
    # Guarantee index is timezone-naive and strictly truncated to Date (hours/mins stripped)
    if not df.empty:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        
    # Pandas will throw "ValueError: cannot join with no overlapping index names"
    # if the new df has ['Ticker', 'Price'] and the cached df has [None, None].
    # So we force it to [None, None] to match the cache.
    df.columns.names = [None, None]
        
    return df

def fetch_incremental_history(tickers, days=252, progress_callback=None, force_today_refresh=False, cache_file=CACHE_FILE, period="1y"):
    """
    Loads historical prices from local disk to avoid rate limiting.
    Only incrementally downloads the exact delta of missing dates from Yahoo Finance,
    including per-ticker catch-up backfills if a partial-subset sync left NaN holes.
    """
    target_tickers = set(tickers)
    
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_pickle(cache_file)
            cached_tickers = set(cached_df.columns.get_level_values(0))
        except Exception as e:
            print("Error loading cache, rebuilding:", e)
            cached_df = pd.DataFrame()
            cached_tickers = set()
    else:
        cached_df = pd.DataFrame()
        cached_tickers = set()

    delisted_tickers = load_delisted_tickers()
    missing_tickers = [t for t in list(target_tickers - cached_tickers) if t not in delisted_tickers]
    existing_tickers = list(target_tickers.intersection(cached_tickers))
    
    new_dataframes = []
    
    # 1. Fetch completely missing tickers (Full History)
    if missing_tickers:
        if progress_callback:
            progress_callback(f"Downloading {period} history for {len(missing_tickers)} uncached tickers...")
        
        missing_df = _safe_yf_download(
            list(missing_tickers), 
            period=period, 
            group_by='ticker',
            progress=False,
            auto_adjust=False
        )
        missing_df = _ensure_multiindex(missing_df, missing_tickers)
        if not missing_df.empty:
            new_dataframes.append(missing_df)
            downloaded = set(missing_df.columns.get_level_values(0))
        else:
            downloaded = set()
            
        # Cache persistently failed / delisted tickers so we never spam Yahoo Finance again
        failed_tickers = set(missing_tickers) - downloaded
        if failed_tickers:
            delisted_tickers.update(failed_tickers)
            save_delisted_tickers(delisted_tickers)

    # 2. Fetch incremental deltas for existing tickers (once per calendar day per market)
    if existing_tickers and not cached_df.empty:
        is_in_market = any(isinstance(t, str) and (t.endswith('.NS') or t.endswith('.BO')) for t in existing_tickers)
        mkt = 'IN' if is_in_market or not existing_tickers else 'US'
        
        last_date = get_effective_last_date(cached_df, existing_tickers, market=mkt)
        if last_date is None:
            last_date = cached_df.index.max()
            
        already_synced = has_synced_today(cache_file, mkt)
        need_sync = (not already_synced) or force_today_refresh
        
        if not need_sync:
            print(f"[PriceHistoryManager] Already synced today ({mkt}) for {os.path.basename(cache_file)} (last date: {last_date.date() if hasattr(last_date, 'date') else last_date}). Fast loading from disk cache.")
        else:
            print(f"[PriceHistoryManager] First sync of the day for {mkt} (last date: {last_date.date() if hasattr(last_date, 'date') else last_date}). Fetching delta from Yahoo Finance...")
            if progress_callback:
                progress_callback(f"Syncing daily incremental delta for {len(existing_tickers)} {mkt} stocks since {last_date.date()}...")
            delta_df = _safe_yf_download(
                list(existing_tickers),
                start=last_date.strftime("%Y-%m-%d"),
                group_by='ticker',
                progress=False,
                auto_adjust=False
            )
            delta_df = _ensure_multiindex(delta_df, existing_tickers)
            if not delta_df.empty:
                new_dataframes.append(delta_df)
                
            record_sync_complete(cache_file, mkt)

    # 3. Merge everything together
    master_df = cached_df.copy()
    
    if new_dataframes:
        if progress_callback:
            progress_callback("Merging memory matrix and saving to NVMe cache...")
            
        for new_df in new_dataframes:
            master_df = new_df.combine_first(master_df)
            
        if not master_df.empty:
            
            master_df.sort_index(inplace=True)
            
            try:
                if 'Close' in master_df.columns.get_level_values(1):
                    close_df = master_df.xs('Close', level=1, axis=1)
                    valid_counts = close_df.notna().sum()
                    valid_tickers = valid_counts[valid_counts >= 5].index.tolist()
                    
                    master_df = master_df.loc[:, master_df.columns.get_level_values(0).isin(valid_tickers)]
            except Exception as e:
                pass
            
            temp_file = cache_file + ".tmp"
            master_df.to_pickle(temp_file)
            
            for attempt in range(10):
                try:
                    os.replace(temp_file, cache_file)
                    break
                except Exception:
                    time.sleep(0.2)
            else:
                if os.path.exists(cache_file):
                    try: os.remove(cache_file)
                    except: pass
                try: os.rename(temp_file, cache_file)
                except: pass
                
            cached_df = master_df
            
    if not cached_df.empty and cached_df.index.tz is not None:
        cached_df.index = cached_df.index.tz_localize(None)

    if not cached_df.empty:
        available_tickers = [t for t in target_tickers if t in cached_df.columns.get_level_values(0)]
        return cached_df[available_tickers].copy()
        
    return cached_df

def get_cache_status():
    """Returns number of tickers cached, and the latest date available."""
    if not os.path.exists(CACHE_FILE):
        return 0, None
    try:
        df = pd.read_pickle(CACHE_FILE)
        tickers = len(set(df.columns.get_level_values(0)))
        max_date = df.index.max()
        return tickers, max_date
    except:
        return 0, None

def rebuild_full_history(tickers, days=252, progress_callback=None, cache_file=CACHE_FILE, period="1y"):
    """
    Forcefully wipes the disk cache and downloads pristine history.
    """
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
        except Exception as e:
            if progress_callback:
                progress_callback(f"Failed to delete DB. Ensure it's not locked. {e}")
            pass
            
    return fetch_incremental_history(tickers, days=days, progress_callback=progress_callback, cache_file=cache_file, period=period)

def ensure_history_range(tickers, required_start_date, required_end_date=None, progress_callback=None):
    """
    Ensures that the cache contains complete data from required_start_date to required_end_date.
    If the cache is missing data or has huge gaps, it dynamically fetches the missing historical block.
    """
    target_tickers = set(tickers)
    
    if os.path.exists(CACHE_FILE):
        try:
            cached_df = pd.read_pickle(CACHE_FILE)
        except Exception as e:
            print("Error loading cache in ensure_history_range:", e)
            cached_df = pd.DataFrame()
    else:
        cached_df = pd.DataFrame()

    req_start_dt = pd.to_datetime(required_start_date)
    if req_start_dt.tz is not None:
        req_start_dt = req_start_dt.tz_localize(None)
        
    req_end_dt = pd.to_datetime(required_end_date) if required_end_date else datetime.now()
    if req_end_dt.tz is not None:
        req_end_dt = req_end_dt.tz_localize(None)
        
    needs_download = False
    
    if cached_df.empty:
        needs_download = True
    else:
        # Check if the requested range has sufficient data density
        mask = (cached_df.index >= req_start_dt) & (cached_df.index <= req_end_dt)
        actual_days = cached_df[mask].shape[0]
        
        # Expect roughly 252 trading days per year (approx 5 days a week minus holidays)
        calendar_days = (req_end_dt - req_start_dt).days
        expected_trading_days = calendar_days * (252.0 / 365.0) * 0.85 # 85% tolerance for holidays/missing
        
        if actual_days < expected_trading_days:
            needs_download = True
            
    if not needs_download:
        return
        
    if progress_callback:
        progress_callback(f"Downloading missing data from {req_start_dt.strftime('%Y-%m-%d')} to {req_end_dt.strftime('%Y-%m-%d')}... This may take a minute.")
        
    # We add 1 day to end_date because yfinance 'end' is exclusive
    fetch_end_dt = req_end_dt + timedelta(days=1)
    
    delta_df = _safe_yf_download(
        list(target_tickers),
        start=req_start_dt.strftime("%Y-%m-%d"),
        end=fetch_end_dt.strftime("%Y-%m-%d"),
        group_by='ticker',
        progress=False,
        auto_adjust=False
    )
    
    delta_df = _ensure_multiindex(delta_df, target_tickers)
    
    if not delta_df.empty:
        if progress_callback:
            progress_callback("Merging historical data to cache...")
        master_df = cached_df.copy()
        master_df = delta_df.combine_first(master_df)
        master_df.sort_index(inplace=True)
        
        # Save it permanently
        temp_file = CACHE_FILE + ".tmp"
        master_df.to_pickle(temp_file)
        
        for attempt in range(10):
            try:
                os.replace(temp_file, CACHE_FILE)
                break
            except Exception:
                time.sleep(0.2)
        else:
            if os.path.exists(CACHE_FILE):
                try: os.remove(CACHE_FILE)
                except: pass
            try: os.rename(temp_file, CACHE_FILE)
            except: pass


def inject_external_history_into_matrix(ticker_df_map: dict, cache_file=CACHE_FILE) -> bool:
    """
    Injects externally resolved OHLCV DataFrames (e.g., from nselib SME or .BO fallback)
    directly into historical_prices_matrix.pkl so small/micro/SME portfolio holdings
    are 100% present in universe RS rankings and Brain synthesis scans.
    """
    if not ticker_df_map:
        return False

    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_pickle(cache_file)
        except Exception:
            cached_df = pd.DataFrame()
    else:
        cached_df = pd.DataFrame()

    existing_close = None
    if not cached_df.empty and isinstance(cached_df.columns, pd.MultiIndex):
        if 'Close' in cached_df.columns.get_level_values(1):
            existing_close = cached_df.xs('Close', level=1, axis=1)

    frames_to_merge = []
    for matrix_ticker, raw_df in ticker_df_map.items():
        if raw_df is None or raw_df.empty or len(raw_df) < 5:
            continue

        df = raw_df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        elif not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                continue

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        df = df[~df.index.duplicated(keep='last')].sort_index()

        col_map = {c.lower(): c for c in df.columns}
        if 'close' not in col_map:
            continue

        max_new_dt = df.index.max()
        if existing_close is not None and matrix_ticker in existing_close.columns:
            cur_lvi = existing_close[matrix_ticker].last_valid_index()
            cur_cnt = int(existing_close[matrix_ticker].notna().sum())
            if cur_lvi is not None and cur_lvi >= max_new_dt and cur_cnt >= len(df):
                continue

        sub_data = {}
        for std_col, low_key in [('Open', 'open'), ('High', 'high'), ('Low', 'low'), ('Close', 'close'), ('Volume', 'volume')]:
            if low_key in col_map:
                sub_data[(matrix_ticker, std_col)] = pd.to_numeric(df[col_map[low_key]], errors='coerce')
            elif std_col != 'Volume':
                sub_data[(matrix_ticker, std_col)] = pd.to_numeric(df[col_map['close']], errors='coerce')
            else:
                sub_data[(matrix_ticker, std_col)] = 0.0
        sub_data[(matrix_ticker, 'Adj Close')] = sub_data[(matrix_ticker, 'Close')]

        multi_df = pd.DataFrame(sub_data, index=df.index)
        multi_df.columns = pd.MultiIndex.from_tuples(multi_df.columns)
        multi_df.columns.names = [None, None]
        frames_to_merge.append(multi_df)

    if not frames_to_merge:
        return False

    merged_ext = pd.concat(frames_to_merge, axis=1)
    merged_ext.columns.names = [None, None]

    if cached_df.empty:
        master_df = merged_ext
    else:
        if cached_df.index.tz is not None:
            cached_df.index = cached_df.index.tz_localize(None)
        cached_df.columns.names = [None, None]
        master_df = merged_ext.combine_first(cached_df)

    master_df.sort_index(inplace=True)
    temp_file = cache_file + ".tmp"
    master_df.to_pickle(temp_file)
    for attempt in range(10):
        try:
            os.replace(temp_file, cache_file)
            break
        except Exception:
            time.sleep(0.2)
    else:
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except Exception:
                pass
        try:
            os.rename(temp_file, cache_file)
        except Exception:
            pass
    return True
