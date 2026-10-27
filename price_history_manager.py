import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta
import time

CACHE_FILE = "historical_prices_matrix.pkl"
_LAGGING_CHECKED_TODAY = set()

def _safe_yf_download(tickers, max_retries=3, chunk_size=500, **kwargs):
    """
    yfinance multi-threading throws 'RuntimeError: dictionary changed size' occasionally.
    This safely chunks the download and retries if the race condition occurs, keeping it fast.
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
            except RuntimeError as e:
                if "dictionary changed size" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(1.5)
                    continue
                raise
                
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

    missing_tickers = list(target_tickers - cached_tickers)
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

    # 2. Fetch incremental deltas for existing tickers (including per-ticker NaN hole catch-up)
    if existing_tickers and not cached_df.empty:
        last_date = cached_df.index.max()
        now = datetime.now()
        today_key = now.strftime("%Y-%m-%d")

        # Detect partial-subset NaN holes: existing tickers whose last valid Close trails last_date
        lagging_tickers = []
        min_lag_date = last_date
        try:
            if 'Close' in cached_df.columns.get_level_values(1):
                close_sub = cached_df.xs('Close', level=1, axis=1)
                valid_existing = [t for t in existing_tickers if t in close_sub.columns]
                if valid_existing:
                    last_row = close_sub.loc[last_date, valid_existing]
                    nan_on_last = last_row[last_row.isna()].index.tolist()
                    cutoff_dt = last_date - timedelta(days=45)
                    for t in nan_on_last:
                        if (t, today_key) in _LAGGING_CHECKED_TODAY and not force_today_refresh:
                            continue
                        lvi = close_sub[t].last_valid_index()
                        if lvi is not None and cutoff_dt <= lvi < last_date:
                            lagging_tickers.append(t)
                            if lvi < min_lag_date:
                                min_lag_date = lvi
                        _LAGGING_CHECKED_TODAY.add((t, today_key))
        except Exception as e:
            print(f"[PriceHistoryManager] Lagging ticker check warning: {e}")

        need_global_sync = (last_date.date() < now.date()) or force_today_refresh

        if need_global_sync:
            non_lagging = [t for t in existing_tickers if t not in set(lagging_tickers)]
            if non_lagging:
                if progress_callback:
                    progress_callback(f"Syncing incremental bars since {last_date.date()}...")
                delta_df = _safe_yf_download(
                    list(non_lagging),
                    start=last_date.strftime("%Y-%m-%d"),
                    group_by='ticker',
                    progress=False,
                    auto_adjust=False
                )
                delta_df = _ensure_multiindex(delta_df, non_lagging)
                if not delta_df.empty:
                    new_dataframes.append(delta_df)

        if lagging_tickers:
            if progress_callback:
                progress_callback(f"Backfilling {len(lagging_tickers)} partial-subset lagging tickers since {min_lag_date.date()}...")
            lag_df = _safe_yf_download(
                list(lagging_tickers),
                start=min_lag_date.strftime("%Y-%m-%d"),
                group_by='ticker',
                progress=False,
                auto_adjust=False
            )
            lag_df = _ensure_multiindex(lag_df, lagging_tickers)
            if not lag_df.empty:
                new_dataframes.append(lag_df)

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
