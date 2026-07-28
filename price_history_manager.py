import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta
import time

import config
CACHE_FILE = config.PRICE_HISTORY_CACHE_PATH

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

def fetch_incremental_history(tickers, days=252, progress_callback=None, force_today_refresh=False):
    """
    Loads historical prices from local disk to avoid rate limiting.
    Only incrementally downloads the exact delta of missing dates from Yahoo Finance.
    """
    target_tickers = set(tickers)
    
    if os.path.exists(CACHE_FILE):
        try:
            cached_df = pd.read_pickle(CACHE_FILE)
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
    
    # 1. Fetch completely missing tickers (Full 1 Year History)
    if missing_tickers:
        if progress_callback:
            progress_callback(f"Downloading 1Y history for {len(missing_tickers)} uncached tickers...")
        
        missing_df = _safe_yf_download(
            list(missing_tickers), 
            period="1y", 
            group_by='ticker',
            progress=False,
            auto_adjust=False
        )
        missing_df = _ensure_multiindex(missing_df, missing_tickers)
        if not missing_df.empty:
            new_dataframes.append(missing_df)

    # 2. Fetch incremental deltas for existing tickers
    if existing_tickers and not cached_df.empty:
        last_date = cached_df.index.max()
        # If the last_date is older than today, fetch delta
        now = datetime.now()
        
        # We fetch starting from the last known date
        # yfinance `start` date is inclusive, so we'll get crossover, which we drop later
        if last_date.date() < now.date() or force_today_refresh:
            if progress_callback:
                progress_callback(f"Syncing incremental bars since {last_date.date()}...")
                
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

    # 3. Merge everything together
    master_df = cached_df.copy()
    
    if new_dataframes:
        if progress_callback:
            progress_callback("Merging memory matrix and saving to NVMe cache...")
            
        # .combine_first() is structurally bulletproof. 
        # If new_df has new dates, it appends them. 
        # If new_df has new columns (tickers), it appends them.
        # If new_df has overlapping dates with valid data, it overrides cached_df.
        # If new_df has overlapping dates with YFinance NaNs (rate limit), it IGNORES them and keeps cached_df's valid data!
        for new_df in new_dataframes:
            master_df = new_df.combine_first(master_df)
            
        if not master_df.empty:
            
            master_df.sort_index(inplace=True)
            
            # -- VALIDATION STEP --
            # yfinance creates NaN columns even if a ticker fails to download.
            # If we save NaNs, the cache thinks it succeeded and won't retry.
            # We strip any ticker that has fewer than 5 valid close prices (5 protects IPOs while punishing dead drops).
            try:
                if 'Close' in master_df.columns.get_level_values(1):
                    close_df = master_df.xs('Close', level=1, axis=1)
                    # Count valid non-NaN bars per ticker
                    valid_counts = close_df.notna().sum()
                    valid_tickers = valid_counts[valid_counts >= 5].index.tolist()
                    
                    # Contract the master_df to strictly valid tickers
                    master_df = master_df.loc[:, master_df.columns.get_level_values(0).isin(valid_tickers)]
                    
                    # Scrub Orphaned Tickers (companies removed from the universe)
                    # DISABLED: Because we run multiple universes (US and IN) on the same cache,
                    # dropping 'orphans' based on a single target_tickers list will delete the other market's data!
                    # all_cached = set(master_df.columns.get_level_values(0))
                    # orphans = list(all_cached - target_tickers)
                    # if orphans:
                    #     master_df.drop(columns=orphans, level=0, inplace=True, errors='ignore')
                        
            except Exception as e:
                pass
            
            # Save it permanently (Atomic Write to prevent corruption on Streamlit reload)
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
                
            cached_df = master_df
            
    # Remove any timezone awareness dynamically to avoid merge crashes
    if not cached_df.empty and cached_df.index.tz is not None:
        cached_df.index = cached_df.index.tz_localize(None)

    # In case the user requested specific tickers out of a massive overall cache,
    # we filter out everything else so their RAM isn't loaded with non-requested tickers
    # Get available level 0 columns (tickers)
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

def rebuild_full_history(tickers, days=252, progress_callback=None):
    """
    Forcefully wipes the disk cache and downloads pristine 252-day history.
    Used to retroactively adjust the dataset for Stock Splits and Corporate Actions.
    """
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
        except Exception as e:
            if progress_callback:
                progress_callback(f"Failed to delete DB. Ensure it's not locked. {e}")
            pass
            
    return fetch_incremental_history(tickers, days=days, progress_callback=progress_callback)

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
