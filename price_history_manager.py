import pandas as pd
import yfinance as yf
import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = "historical_prices_matrix.pkl"
DELISTED_CACHE_FILE = os.path.join(_BASE_DIR, ".cache", "delisted_tickers.json")
QUARANTINE_META_FILE = os.path.join(_BASE_DIR, ".cache", "ticker_quarantine.json")
_LAGGING_CHECKED_TODAY = set()
_LAST_DOWNLOAD_RATE_LIMITED = False


def _resolve_cache_path(cache_file: str) -> str:
    """Resolve relative cache filenames cleanly (preferring CWD if file exists for tests, else project root)."""
    if not cache_file:
        return os.path.join(_BASE_DIR, CACHE_FILE)
    if os.path.isabs(cache_file):
        return cache_file
    if os.path.exists(cache_file):
        return os.path.abspath(cache_file)
    return os.path.join(_BASE_DIR, cache_file)



def _clamp_df_to_settled_eod(df: pd.DataFrame, resolved_cache: str, allow_intraday: bool = False) -> pd.DataFrame:
    """
    Normalizes datetimes (tz-naive, midnight) and strips/masks any unclosed intraday bar
    (date > get_latest_completed_session_date(mkt)) on settled EOD matrices.
    Intraday caches (filename starting with `intraday_`) or `allow_intraday=True` bypass clamping.
    """
    if df is None or df.empty:
        return df
    base_name = os.path.basename(str(resolved_cache or "")).lower()
    if allow_intraday or base_name.startswith("intraday_"):
        return df
    out = df.copy()
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out.index = pd.to_datetime(out.index).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()

    in_cutoff = pd.Timestamp(get_latest_completed_session_date("IN")).normalize()
    us_cutoff = pd.Timestamp(get_latest_completed_session_date("US")).normalize()

    if isinstance(out.columns, pd.MultiIndex):
        lvl0 = out.columns.get_level_values(0)
        is_in_mask = lvl0.map(lambda c: str(c).upper().endswith((".NS", ".BO"))).to_numpy(dtype=bool)
        if is_in_mask.all():
            return out.loc[out.index <= in_cutoff]
        if (~is_in_mask).all():
            return out.loc[out.index <= us_cutoff]
        if is_in_mask.any():
            bad_in = out.index > in_cutoff
            if bad_in.any():
                out.loc[bad_in, is_in_mask] = float("nan")
        if (~is_in_mask).any():
            bad_us = out.index > us_cutoff
            if bad_us.any():
                out.loc[bad_us, ~is_in_mask] = float("nan")
        return out.dropna(how="all")
    return out


def _atomic_write_pickle(df: pd.DataFrame, cache_path: str) -> None:
    """Safely write pickle to disk using temp file + atomic replace."""
    temp_file = cache_path + ".tmp"
    df.to_pickle(temp_file)
    for _ in range(10):
        try:
            os.replace(temp_file, cache_path)
            return
        except Exception:
            time.sleep(0.2)
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
        except Exception:
            pass
    try:
        os.rename(temp_file, cache_path)
    except Exception:
        pass


def _load_quarantine_meta() -> dict:
    """Loads strike counts & cooldown timestamps for failed tickers."""
    if os.path.exists(QUARANTINE_META_FILE):
        try:
            with open(QUARANTINE_META_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def _save_quarantine_meta(meta: dict) -> None:
    try:
        os.makedirs(os.path.dirname(QUARANTINE_META_FILE), exist_ok=True)
        with open(QUARANTINE_META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
    except Exception:
        pass


def load_delisted_tickers() -> set:
    """
    Loads active quarantined tickers whose cooldown has not expired.
    Legacy plain entries in delisted_tickers.json are honored until sanitized or expired.
    """
    delisted = set()
    if os.path.exists(DELISTED_CACHE_FILE):
        try:
            with open(DELISTED_CACHE_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
                if isinstance(raw, list):
                    delisted = set(str(x).strip().upper() for x in raw if x)
        except Exception:
            pass

    meta = _load_quarantine_meta()
    today_str = datetime.now().strftime("%Y-%m-%d")
    expired = []
    for sym, info in meta.items():
        if not isinstance(info, dict):
            continue
        cooldown = info.get("cooldown_until", "")
        if cooldown and today_str >= cooldown:
            expired.append(sym)

    if expired:
        for sym in expired:
            meta.pop(sym, None)
            delisted.discard(sym)
        _save_quarantine_meta(meta)
        save_delisted_tickers(delisted)

    return delisted


def save_delisted_tickers(delisted_set: set):
    """Persists active quarantined tickers to JSON cache."""
    try:
        os.makedirs(os.path.dirname(DELISTED_CACHE_FILE), exist_ok=True)
        with open(DELISTED_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(list(delisted_set)), f, indent=2)
    except Exception:
        pass


def record_failed_tickers(failed_tickers: set, max_strikes: int = 3, cooldown_days: int = 7) -> set:
    """
    3-Strike Quarantine Guard:
    Only quarantines a ticker in delisted_tickers.json after `max_strikes` consecutive
    download failures, with automatic `cooldown_days` expiration.
    """
    if not failed_tickers:
        return load_delisted_tickers()

    delisted = load_delisted_tickers()
    meta = _load_quarantine_meta()
    today_dt = datetime.now().date()
    today_str = today_dt.strftime("%Y-%m-%d")
    cooldown_str = (today_dt + timedelta(days=cooldown_days)).strftime("%Y-%m-%d")

    for t in failed_tickers:
        key = str(t).strip().upper()
        if not key:
            continue
        entry = meta.get(key, {"strikes": 0, "first_failed": today_str})
        entry["strikes"] = int(entry.get("strikes", 0)) + 1
        entry["last_failed"] = today_str
        if entry["strikes"] >= max_strikes:
            entry["cooldown_until"] = cooldown_str
            delisted.add(key)
        meta[key] = entry

    _save_quarantine_meta(meta)
    save_delisted_tickers(delisted)
    return delisted


def clear_quarantine_strikes(succeeded_tickers: set) -> None:
    """Resets strike counters for tickers that downloaded cleanly."""
    if not succeeded_tickers:
        return
    meta = _load_quarantine_meta()
    delisted = load_delisted_tickers()
    changed_meta = False
    changed_del = False
    for t in succeeded_tickers:
        key = str(t).strip().upper()
        if key in meta:
            meta.pop(key, None)
            changed_meta = True
        if key in delisted:
            delisted.discard(key)
            changed_del = True
    if changed_meta:
        _save_quarantine_meta(meta)
    if changed_del:
        save_delisted_tickers(delisted)


def get_quarantined_tickers_info() -> list:
    """Returns detailed metadata for quarantined tickers for admin diagnostics."""
    delisted = load_delisted_tickers()
    meta = _load_quarantine_meta()
    items = []
    for sym in sorted(delisted.union(set(meta.keys()))):
        m = meta.get(sym, {})
        items.append({
            "ticker": sym,
            "quarantined": sym in delisted,
            "strikes": int(m.get("strikes", 3 if sym in delisted else 0)),
            "first_failed": m.get("first_failed", "legacy"),
            "last_failed": m.get("last_failed", "legacy"),
            "cooldown_until": m.get("cooldown_until", "unbounded (legacy)"),
        })
    return items


def sanitize_delisted_tickers(active_clean_symbols: set) -> int:
    """
    Purges false-positive SME/IPO/active tickers from .cache/delisted_tickers.json
    and .cache/ticker_quarantine.json when verified active on official exchange master lists.
    Supports both clean symbols ('RELIANCE') and full yf tickers ('TATAMOTORS.NS', 'BRK-B').
    """
    if not active_clean_symbols:
        return 0
    current_delisted = load_delisted_tickers()
    meta = _load_quarantine_meta()
    if not current_delisted and not meta:
        return 0

    upper_active = {str(s).strip().upper() for s in active_clean_symbols if s}
    cleaned = set()
    removed = 0
    for item in current_delisted:
        full_up = str(item).strip().upper()
        base = full_up.replace(".NS", "").replace(".BO", "").replace("-SM", "").replace("-ST", "")
        if full_up in upper_active or base in upper_active:
            removed += 1
        else:
            cleaned.add(item)

    meta_removed = 0
    for k in list(meta.keys()):
        full_up = str(k).strip().upper()
        base = full_up.replace(".NS", "").replace(".BO", "").replace("-SM", "").replace("-ST", "")
        if full_up in upper_active or base in upper_active:
            meta.pop(k, None)
            meta_removed += 1

    if removed > 0:
        save_delisted_tickers(cleaned)
    if meta_removed > 0:
        _save_quarantine_meta(meta)
    return max(removed, meta_removed)

SYNC_TRACKER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matrix_sync_tracker.json")

_MARKET_TIMEZONE_SPECS = {
    "IN": ("Asia/Kolkata", 15, 45),
    "INDIA": ("Asia/Kolkata", 15, 45),
    "US": ("America/New_York", 16, 15),
    "USA": ("America/New_York", 16, 15),
    "US_IXIC": ("America/New_York", 16, 15),
}


def normalize_tracker_market(market: str = "IN") -> str:
    """Normalizes market identifiers into canonical tracker keys ('IN' or 'US')."""
    m = str(market or "IN").strip().upper()
    return "IN" if m in ("IN", "INDIA") else "US"


def is_market_after_close(market: str = "IN", now_utc: datetime = None) -> bool:
    """
    Returns True if the target exchange is currently settled after close on a weekday
    or it is a weekend (previous Friday settled). Correctly converts from Singapore/India
    local clocks to Asia/Kolkata (15:45) or America/New_York (16:15 EST/EDT).
    """
    m_key = str(market or "IN").strip().upper()
    tz_name, close_h, close_m = _MARKET_TIMEZONE_SPECS.get(m_key, ("Asia/Kolkata", 15, 45))
    tz = ZoneInfo(tz_name)
    if now_utc is None:
        now_local = datetime.now(tz)
    elif now_utc.tzinfo is None:
        now_local = now_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
    else:
        now_local = now_utc.astimezone(tz)

    if now_local.weekday() >= 5:
        return True
    return (now_local.hour, now_local.minute) >= (close_h, close_m)


def get_latest_completed_session_date(market: str = "IN", now_utc: datetime = None) -> str:
    """
    Returns YYYY-MM-DD of the most recent completed exchange session in the market's
    own timezone (IST 15:45 for India; NY 16:15 EST/EDT for US -> ~05:15/04:15 SGT next day).
    Prevents SGT/IST calendar-date drift across midnight.
    """
    m_key = str(market or "IN").strip().upper()
    tz_name, close_h, close_m = _MARKET_TIMEZONE_SPECS.get(m_key, ("Asia/Kolkata", 15, 45))
    tz = ZoneInfo(tz_name)
    if now_utc is None:
        now_local = datetime.now(tz)
    elif now_utc.tzinfo is None:
        now_local = now_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
    else:
        now_local = now_utc.astimezone(tz)

    wd = now_local.weekday()
    after_close = (now_local.hour, now_local.minute) >= (close_h, close_m)
    if wd == 5:
        target = now_local - timedelta(days=1)
    elif wd == 6:
        target = now_local - timedelta(days=2)
    elif not after_close:
        target = now_local - timedelta(days=3 if wd == 0 else 1)
    else:
        target = now_local
    return target.strftime("%Y-%m-%d")


def load_sync_tracker() -> dict:
    if os.path.exists(SYNC_TRACKER_FILE):
        try:
            with open(SYNC_TRACKER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def _atomic_write_json(path: str, payload: dict) -> None:
    """Safely writes JSON atomically so concurrent keys like van_tharp_indices are preserved."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def record_sync_complete(cache_file: str, market: str, session_date: str = None, after_close: bool = None):
    tracker = load_sync_tracker()
    base_key = os.path.basename(cache_file)
    if base_key not in tracker or not isinstance(tracker.get(base_key), dict):
        tracker[base_key] = {}
    mkt_key = normalize_tracker_market(market)
    eff_session = session_date or get_latest_completed_session_date(market)
    eff_after = is_market_after_close(market) if after_close is None else bool(after_close)
    tracker[base_key][mkt_key] = {
        "last_sync_date": eff_session,
        "session_date": eff_session,
        "after_close": eff_after,
        "sync_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        _atomic_write_json(SYNC_TRACKER_FILE, tracker)
    except Exception as e:
        print(f"[PriceHistoryManager] Sync tracker notice: {e}")


def has_synced_today(cache_file: str, market: str) -> bool:
    tracker = load_sync_tracker()
    base_key = os.path.basename(cache_file)
    mkt_key = normalize_tracker_market(market)
    mkt_entry = tracker.get(base_key, {}).get(mkt_key, {})
    if not isinstance(mkt_entry, dict):
        return False
    last_session = mkt_entry.get("session_date") or mkt_entry.get("last_sync_date", "")
    expected_session = get_latest_completed_session_date(market)
    if last_session != expected_session:
        return False
    # If prior sync ran intraday before official exchange close, allow post-close re-sync once settled
    if mkt_entry.get("after_close") is False and is_market_after_close(market):
        return False
    return True


def get_effective_last_date(cached_df: pd.DataFrame, target_tickers: list = None, market: str = 'IN') -> pd.Timestamp:
    """
    Finds the newest date where a true majority of the requested market's tickers have data.
    Decouples US stocks from Indian stocks even in the same matrix file and ensures
    partial 15-stock runs never mask stale full-universe bars.
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
            valid_existing = [
                c for c in close_df.columns
                if not (str(c).endswith('.NS') or str(c).endswith('.BO')) and not str(c).startswith('^')
            ]
            
    if not valid_existing:
        return cached_df.index.max()
        
    counts = close_df[valid_existing].notna().sum(axis=1)
    # Require >= 50% coverage on multi-stock panels so small partial runs self-heal
    ratio = 0.50 if len(valid_existing) >= 50 else 0.15
    threshold = max(1, int(len(valid_existing) * ratio))
    valid_dates = counts[counts >= threshold].index
    if len(valid_dates) > 0:
        return valid_dates.max()
    return cached_df.index.max()

def get_matrix_last_date(cache_file=CACHE_FILE, market=None) -> str:
    """
    Returns the newest date string (YYYY-MM-DD) found in the cache matrix for the given market.
    Runs in < 0.05s from disk.
    """
    resolved_cache = _resolve_cache_path(cache_file)
    if os.path.exists(resolved_cache):
        try:
            df = pd.read_pickle(resolved_cache)
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

def _safe_yf_download(tickers, max_retries=3, chunk_size=500, chunk_sleep=0.0, **kwargs):
    """
    yfinance multi-threading throws 'RuntimeError: dictionary changed size' occasionally.
    This safely chunks the download and retries if the race condition occurs, keeping it fast.
    Aborts gracefully on Yahoo Finance rate limits (HTTP 429) to prevent UI freezes.
    """
    global _LAST_DOWNLOAD_RATE_LIMITED
    _LAST_DOWNLOAD_RATE_LIMITED = False

    if isinstance(tickers, str):
        tickers = [tickers]
        
    kwargs['threads'] = True  # Re-enable multi-threading for speed!
    all_dfs = []
    
    total_chunks = (len(tickers) + chunk_size - 1) // chunk_size
    for idx, i in enumerate(range(0, len(tickers), chunk_size)):
        chunk = tickers[i:i+chunk_size]
        for attempt in range(max_retries):
            try:
                df = yf.download(chunk, **kwargs)
                if not df.empty:
                    all_dfs.append(df)
                break
            except Exception as e:
                err_str = str(e).lower()
                if "rate limit" in err_str or "too many requests" in err_str or "429" in err_str:
                    _LAST_DOWNLOAD_RATE_LIMITED = True
                    print(f"[PriceHistoryManager] ⚠️ Yahoo Finance rate limit active: {e}. Aborting network sync and preserving cache.")
                    return pd.concat(all_dfs, axis=1) if all_dfs else pd.DataFrame()
                if "dictionary changed size" in err_str and attempt < max_retries - 1:
                    time.sleep(1.5)
                    continue
                break
        if chunk_sleep > 0 and idx < total_chunks - 1:
            time.sleep(chunk_sleep)
                
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

def fetch_incremental_history(tickers, days=252, progress_callback=None, force_today_refresh=False, cache_file=CACHE_FILE, period="1y", chunk_size=500, chunk_sleep=0.0):
    """
    Loads historical prices from local disk to avoid rate limiting.
    Only incrementally downloads the exact delta of missing dates from Yahoo Finance,
    including per-ticker catch-up backfills if a partial-subset sync left NaN holes.
    """
    resolved_cache = _resolve_cache_path(cache_file)
    target_tickers = set(tickers)
    
    if os.path.exists(resolved_cache):
        try:
            cached_df = pd.read_pickle(resolved_cache)
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
            chunk_size=chunk_size,
            chunk_sleep=chunk_sleep,
            group_by='ticker',
            progress=False,
            auto_adjust=False
        )
        missing_df = _ensure_multiindex(missing_df, missing_tickers)
        missing_df = _clamp_df_to_settled_eod(missing_df, resolved_cache)
        if not missing_df.empty:
            non_empty_cols = missing_df.dropna(how='all', axis=1)
            if not non_empty_cols.empty:
                new_dataframes.append(non_empty_cols)
                downloaded = set(non_empty_cols.columns.get_level_values(0))
            else:
                downloaded = set()
        else:
            downloaded = set()

        if downloaded:
            clear_quarantine_strikes(downloaded)

        # Only count failed downloads toward 3-strike quarantine if Yahoo did NOT throttle us
        if not _LAST_DOWNLOAD_RATE_LIMITED:
            failed_tickers = set(missing_tickers) - downloaded
            if failed_tickers:
                record_failed_tickers(failed_tickers, max_strikes=3, cooldown_days=7)

    # 2. Fetch incremental deltas for existing tickers (once per completed market session)
    _pending_sync_record = None
    if existing_tickers and not cached_df.empty:
        is_in_market = any(isinstance(t, str) and (t.endswith('.NS') or t.endswith('.BO')) for t in existing_tickers)
        mkt = 'IN' if is_in_market or not existing_tickers else 'US'
        target_session = get_latest_completed_session_date(mkt)
        after_close = is_market_after_close(mkt)

        last_date = get_effective_last_date(cached_df, existing_tickers, market=mkt)
        if last_date is None:
            last_date = cached_df.index.max()

        already_synced = has_synced_today(resolved_cache, mkt)
        eff_last_str = last_date.strftime("%Y-%m-%d") if hasattr(last_date, "strftime") else str(last_date)[:10]
        # Force sync even if previously skipped if matrix lags completed exchange session on broad request
        lags_session = (len(existing_tickers) >= 300) and (eff_last_str < target_session)
        need_sync = (not already_synced) or force_today_refresh or lags_session

        if not need_sync:
            print(f"[PriceHistoryManager] Already synced session ({mkt} {target_session}) for {os.path.basename(resolved_cache)} (last date: {eff_last_str}). Fast loading from disk cache.")
        else:
            print(f"[PriceHistoryManager] Syncing {mkt} incremental delta since {eff_last_str} (target session: {target_session})...")
            if progress_callback:
                progress_callback(f"Syncing daily incremental delta for {len(existing_tickers)} {mkt} stocks since {eff_last_str}...")
            eff_sleep = chunk_sleep if chunk_sleep > 0 else (0.35 if len(existing_tickers) >= 800 else 0.0)
            delta_df = _safe_yf_download(
                list(existing_tickers),
                start=last_date.strftime("%Y-%m-%d"),
                chunk_size=chunk_size,
                chunk_sleep=eff_sleep,
                group_by='ticker',
                progress=False,
                auto_adjust=False
            )
            delta_df = _ensure_multiindex(delta_df, existing_tickers)
            delta_df = _clamp_df_to_settled_eod(delta_df, resolved_cache)
            if not delta_df.empty:
                new_dataframes.append(delta_df)

            if not _LAST_DOWNLOAD_RATE_LIMITED and len(existing_tickers) >= 300:
                cov_ok = False
                if not delta_df.empty and isinstance(delta_df.columns, pd.MultiIndex) and 'Close' in delta_df.columns.get_level_values(1):
                    d_close = delta_df.xs('Close', level=1, axis=1)
                    max_d_cov = int(d_close.notna().sum(axis=1).max() or 0)
                    cov_ok = max_d_cov >= int(len(existing_tickers) * 0.70)
                elif eff_last_str >= target_session:
                    cov_ok = True
                if cov_ok:
                    _pending_sync_record = (resolved_cache, mkt, target_session, after_close)

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
                    # Keep >= 1 valid trading bar so newly listed Day 1-4 IPOs persist cleanly
                    valid_tickers = valid_counts[valid_counts >= 1].index.tolist()
                    master_df = master_df.loc[:, master_df.columns.get_level_values(0).isin(valid_tickers)]
            except Exception:
                pass

            _atomic_write_pickle(master_df, resolved_cache)
            cached_df = master_df

    if _pending_sync_record is not None:
        rec_cache, rec_mkt, rec_sess, rec_after = _pending_sync_record
        record_sync_complete(rec_cache, rec_mkt, session_date=rec_sess, after_close=rec_after)
            
    if not cached_df.empty and cached_df.index.tz is not None:
        cached_df.index = cached_df.index.tz_localize(None)

    if not cached_df.empty:
        available_tickers = [t for t in target_tickers if t in cached_df.columns.get_level_values(0)]
        return cached_df[available_tickers].copy()
        
    return cached_df

def get_cache_status():
    """Returns number of tickers cached, and the latest date available."""
    resolved_cache = _resolve_cache_path(CACHE_FILE)
    if not os.path.exists(resolved_cache):
        return 0, None
    try:
        df = pd.read_pickle(resolved_cache)
        tickers = len(set(df.columns.get_level_values(0)))
        max_date = df.index.max()
        return tickers, max_date
    except Exception:
        return 0, None

def rebuild_full_history(tickers, days=252, progress_callback=None, cache_file=CACHE_FILE, period="1y"):
    """
    Forcefully wipes the disk cache and downloads pristine history.
    """
    resolved_cache = _resolve_cache_path(cache_file)
    if os.path.exists(resolved_cache):
        try:
            os.remove(resolved_cache)
        except Exception as e:
            if progress_callback:
                progress_callback(f"Failed to delete DB. Ensure it's not locked. {e}")
            
    return fetch_incremental_history(tickers, days=days, progress_callback=progress_callback, cache_file=cache_file, period=period)

def ensure_history_range(tickers, required_start_date, required_end_date=None, progress_callback=None):
    """
    Ensures that the cache contains complete data from required_start_date to required_end_date.
    If the cache is missing data or has huge gaps, it dynamically fetches the missing historical block.
    """
    resolved_cache = _resolve_cache_path(CACHE_FILE)
    target_tickers = set(tickers)
    
    if os.path.exists(resolved_cache):
        try:
            cached_df = pd.read_pickle(resolved_cache)
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
    delta_df = _clamp_df_to_settled_eod(delta_df, resolved_cache)

    if not delta_df.empty:
        if progress_callback:
            progress_callback("Merging historical data to cache...")
        master_df = cached_df.copy()
        master_df = delta_df.combine_first(master_df)
        master_df.sort_index(inplace=True)
        _atomic_write_pickle(master_df, resolved_cache)


def inject_external_history_into_matrix(ticker_df_map: dict, cache_file=CACHE_FILE, allow_intraday: bool = False) -> bool:
    """
    Injects externally resolved OHLCV DataFrames (e.g., from nselib SME or .BO fallback)
    directly into historical_prices_matrix.pkl so small/micro/SME portfolio holdings
    are present in universe RS rankings and Brain synthesis scans.
    Never writes unsettled intraday rows (> settled market session date) unless allow_intraday=True.
    """
    if not ticker_df_map:
        return False

    resolved_cache = _resolve_cache_path(cache_file)
    if os.path.exists(resolved_cache):
        try:
            cached_df = pd.read_pickle(resolved_cache)
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
        if raw_df is None or raw_df.empty or len(raw_df) < 1:
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

        if not allow_intraday:
            tk_mkt = "IN" if str(matrix_ticker).upper().endswith((".NS", ".BO")) else "US"
            max_settled = pd.Timestamp(get_latest_completed_session_date(tk_mkt))
            df = df[df.index <= max_settled]
            if df.empty:
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
    _atomic_write_pickle(master_df, resolved_cache)
    return True


def prune_matrix_columns(tickers_to_remove: set, cache_file=CACHE_FILE) -> int:
    """
    Removes archived/delisted/garbage columns from the price matrix cleanly without
    wiping or re-downloading remaining valid history.
    Returns number of distinct ticker columns removed.
    """
    if not tickers_to_remove:
        return 0
    resolved_cache = _resolve_cache_path(cache_file)
    if not os.path.exists(resolved_cache):
        return 0
    try:
        df = pd.read_pickle(resolved_cache)
    except Exception:
        return 0
    if df is None or df.empty or not isinstance(df.columns, pd.MultiIndex):
        return 0

    remove_upper = {str(t).strip().upper() for t in tickers_to_remove if t}
    col_tickers = df.columns.get_level_values(0)
    mask_keep = ~col_tickers.astype(str).str.upper().isin(remove_upper)
    removed_tickers = set(col_tickers[~mask_keep])
    if not removed_tickers:
        return 0

    pruned_df = df.loc[:, mask_keep].copy()
    _atomic_write_pickle(pruned_df, resolved_cache)
    return len(removed_tickers)


def fetch_missing_tickers_throttled(
    missing_tickers: list,
    period: str = "1y",
    chunk_size: int = 120,
    chunk_sleep: float = 1.8,
    checkpoint_every_chunks: int = 3,
    force_overwrite: bool = False,
    progress_callback=None,
    cache_file: str = CACHE_FILE,
) -> dict:
    """
    Rate-limit-safe chunked historical backfill for newly added or renamed tickers.
    Checkpoints intermediate chunks atomically so large Day-1 commits survive interruptions.
    """
    resolved_cache = _resolve_cache_path(cache_file)
    targets = list(dict.fromkeys([str(t).strip().upper() for t in (missing_tickers or []) if t]))
    if not targets:
        return {"requested": 0, "downloaded": [], "failed": [], "rate_limited": False}

    if os.path.exists(resolved_cache):
        try:
            cached_df = pd.read_pickle(resolved_cache)
        except Exception:
            cached_df = pd.DataFrame()
    else:
        cached_df = pd.DataFrame()

    if not force_overwrite and not cached_df.empty and isinstance(cached_df.columns, pd.MultiIndex):
        close_sub = cached_df.xs("Close", level=1, axis=1) if "Close" in cached_df.columns.get_level_values(1) else pd.DataFrame()
        already_cols = set(cached_df.columns.get_level_values(0))
        max_dt = pd.to_datetime(cached_df.index.max()).tz_localize(None) if len(cached_df.index) else None
        stale_cutoff = (max_dt - pd.Timedelta(days=3)) if max_dt is not None else None
        refreshed_targets = []
        for t in targets:
            if t not in already_cols:
                refreshed_targets.append(t)
            elif stale_cutoff is not None and t in close_sub.columns:
                lvi = close_sub[t].last_valid_index()
                if lvi is None or pd.to_datetime(lvi).tz_localize(None) < stale_cutoff:
                    refreshed_targets.append(t)
        targets = refreshed_targets
        if not targets:
            return {"requested": 0, "downloaded": [], "failed": [], "rate_limited": False}

    downloaded_all = set()
    rate_limited = False
    buffer_frames = []
    total_chunks = (len(targets) + chunk_size - 1) // chunk_size

    def _flush_buffer(base_df, frames):
        if not frames:
            return base_df
        combined = pd.concat(frames, axis=1)
        combined.columns.names = [None, None]
        if base_df is None or base_df.empty:
            out = combined
        else:
            if base_df.index.tz is not None:
                base_df.index = base_df.index.tz_localize(None)
            base_df.columns.names = [None, None]
            out = combined.combine_first(base_df)
        out.sort_index(inplace=True)
        _atomic_write_pickle(out, resolved_cache)
        return out

    for idx, i in enumerate(range(0, len(targets), chunk_size)):
        chunk = targets[i:i + chunk_size]
        if progress_callback:
            progress_callback((idx / max(1, total_chunks)), f"Fetching OHLCV batch {idx + 1}/{total_chunks} ({len(chunk)} tickers)...")

        df_chunk = _safe_yf_download(
            chunk,
            period=period,
            chunk_size=chunk_size,
            chunk_sleep=0.0,
            group_by="ticker",
            progress=False,
            auto_adjust=False,
        )
        if _LAST_DOWNLOAD_RATE_LIMITED:
            rate_limited = True
            break

        df_chunk = _ensure_multiindex(df_chunk, chunk)
        if not df_chunk.empty:
            non_empty = df_chunk.dropna(how="all", axis=1)
            if not non_empty.empty:
                buffer_frames.append(non_empty)
                downloaded_all.update(set(non_empty.columns.get_level_values(0)))

        if (idx + 1) % checkpoint_every_chunks == 0 and buffer_frames:
            cached_df = _flush_buffer(cached_df, buffer_frames)
            buffer_frames = []

        if chunk_sleep > 0 and idx < total_chunks - 1:
            time.sleep(chunk_sleep)

    if buffer_frames:
        cached_df = _flush_buffer(cached_df, buffer_frames)

    if downloaded_all:
        clear_quarantine_strikes(downloaded_all)

    failed = [t for t in targets if t not in downloaded_all]
    return {
        "requested": len(targets),
        "downloaded": sorted(downloaded_all),
        "failed": sorted(failed),
        "rate_limited": rate_limited,
    }


def filter_panel_active_market_dates(
    panel_df: pd.DataFrame,
    min_coverage_ratio: float = 0.15,
    min_abs: int = 10,
) -> pd.DataFrame:
    """
    Normalizes datetimes (tz-naive, date-normalized, deduped, sorted) and drops
    rows where the market represented by `panel_df` was closed (all-NaN or
    sparse cross-market holiday / glitch bars).
    """
    if panel_df is None or panel_df.empty:
        return panel_df
    out = panel_df.copy()
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out.index = pd.to_datetime(out.index).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    n_cols = len(out.columns)
    if n_cols == 0:
        return out
    min_req = (
        max(1, int(n_cols * min_coverage_ratio))
        if n_cols < min_abs
        else min(n_cols, max(min_abs, int(n_cols * min_coverage_ratio)))
    )
    counts = out.notna().sum(axis=1)
    return out[counts >= min_req]


def _extract_matrix_level(matrix_df: pd.DataFrame, target_field: str) -> pd.DataFrame:
    if matrix_df is None or matrix_df.empty:
        return pd.DataFrame()
    if not isinstance(matrix_df.columns, pd.MultiIndex):
        return matrix_df.copy()
    lvl1 = matrix_df.columns.get_level_values(1)
    lvl0 = matrix_df.columns.get_level_values(0)
    for cand in (target_field, target_field.capitalize(), target_field.upper(), target_field.lower()):
        if cand in lvl1:
            return matrix_df.xs(cand, level=1, axis=1)
        if cand in lvl0:
            return matrix_df.xs(cand, level=0, axis=1)
    return pd.DataFrame()


_CACHED_FILE_ETFS = None


def load_etf_exclusion_set() -> set:
    """Returns uppercase base & suffixed ETF symbols from repo ETF lists."""
    global _CACHED_FILE_ETFS
    if _CACHED_FILE_ETFS is not None:
        return _CACHED_FILE_ETFS
    etfs = {"SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "TLT", "GLD", "SLV", "SHY", "SGOV"}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for fn in ("tickers_etf.txt", "tickers_us_etf.txt"):
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for ln in f:
                        t = ln.strip().upper()
                        if t and not t.startswith("#"):
                            etfs.add(t)
                            etfs.add(t.replace(".NS", "").replace(".BO", ""))
            except Exception:
                pass
    _CACHED_FILE_ETFS = etfs
    return etfs


def is_excluded_etf_or_index(ticker: str) -> bool:
    """True for `^` indices and India/US ETFs (preserving synthetic test tickers cleanly)."""
    s_up = str(ticker or "").strip().upper()
    if not s_up or s_up.startswith("^"):
        return True
    base = s_up.replace(".NS", "").replace(".BO", "")
    if s_up in load_etf_exclusion_set() or base in load_etf_exclusion_set():
        return True
    try:
        from down_day_engine import is_etf_or_non_equity
        if is_etf_or_non_equity(base):
            return True
    except Exception:
        pass
    return False


def get_matrix_cache_signature(cache_file: str = CACHE_FILE) -> tuple[int, int]:
    """Returns `(st_mtime_ns, st_size)` so @st.cache_data auto-invalidates on disk updates."""
    resolved = _resolve_cache_path(cache_file)
    if os.path.exists(resolved):
        try:
            st_res = os.stat(resolved)
            return (int(st_res.st_mtime_ns), int(st_res.st_size))
        except Exception:
            pass
    return (0, 0)


try:
    import streamlit as _st_mod
    _st_matrix_cache = _st_mod.cache_data(ttl=3600, show_spinner=False)
except Exception:
    _st_matrix_cache = lambda fn: fn


@_st_matrix_cache
def _read_pickle_cached(resolved_path: str, cache_sig: tuple[int, int]) -> pd.DataFrame:
    if not os.path.exists(resolved_path):
        return pd.DataFrame()
    return pd.read_pickle(resolved_path)


def load_cached_price_matrix(cache_file: str = CACHE_FILE, cache_sig: tuple = None) -> pd.DataFrame:
    """Mtime-aware shared pickle reader reducing duplicated multi-page RAM loads."""
    resolved = _resolve_cache_path(cache_file)
    sig = cache_sig if cache_sig is not None else get_matrix_cache_signature(resolved)
    if sig == (0, 0):
        return pd.DataFrame()
    try:
        return _read_pickle_cached(resolved, sig)
    except Exception:
        try:
            return pd.read_pickle(resolved)
        except Exception:
            return pd.DataFrame()


def repair_sawtooth_glitches(
    cache_file: str = CACHE_FILE,
    max_jump: float = 1.0,
    snapback_drop: float = -0.45,
    tickers: list = None,
) -> dict:
    """
    Detects split-adjustment sawtooth corruptions (e.g. +100%+ spike snapping back <= -45%
    within 1-5 trading days, like KMEW.NS / A1L.BO), re-downloads clean 2y OHLCV from Yahoo
    with auto_adjust=False, and atomically replaces verified clean columns.
    """
    resolved = _resolve_cache_path(cache_file)
    if not os.path.exists(resolved):
        return {"scanned": 0, "corrupt": [], "repaired": [], "failed": []}
    df = pd.read_pickle(resolved)
    if df is None or df.empty or not isinstance(df.columns, pd.MultiIndex):
        return {"scanned": 0, "corrupt": [], "repaired": [], "failed": []}

    close_df = _extract_matrix_level(df, "Close")
    if close_df.empty:
        return {"scanned": 0, "corrupt": [], "repaired": [], "failed": []}

    scan_cols = [c for c in close_df.columns if (not tickers or c in tickers)]
    corrupt_cols = []
    for col in scan_cols:
        s = pd.to_numeric(close_df[col], errors="coerce").dropna()
        if len(s) < 10:
            continue
        def _has_sawtooth(ret_ser: pd.Series) -> bool:
            for dt in ret_ser[ret_ser > max_jump].index:
                pos = ret_ser.index.get_loc(dt)
                if (ret_ser.iloc[pos + 1:min(len(ret_ser), pos + 6)] < snapback_drop).any():
                    return True
            for dt in ret_ser[ret_ser < snapback_drop].index:
                pos = ret_ser.index.get_loc(dt)
                if (ret_ser.iloc[pos + 1:min(len(ret_ser), pos + 6)] > max_jump).any():
                    return True
            return False

        r = s.pct_change()
        is_saw = _has_sawtooth(r)
        is_split_suspect = (not is_saw) and ((r > 1.5).any() or (r < -0.60).any())
        if is_saw or is_split_suspect:
            corrupt_cols.append(col)

    if not corrupt_cols:
        return {"scanned": len(scan_cols), "corrupt": [], "repaired": [], "failed": []}

    repaired, failed = [], []
    fresh_frames = []
    in_max = pd.Timestamp(get_latest_completed_session_date("IN"))
    us_max = pd.Timestamp(get_latest_completed_session_date("US"))

    for col in corrupt_cols:
        try:
            dl = _safe_yf_download([col], period="2y", chunk_size=1, chunk_sleep=0.0, group_by="ticker", progress=False, auto_adjust=False)
            if dl is None or dl.empty:
                failed.append(col)
                continue
            dl = _ensure_multiindex(dl, [col])
            if dl.empty:
                failed.append(col)
                continue
            if getattr(dl.index, "tz", None) is not None:
                dl.index = dl.index.tz_localize(None)
            dl.index = pd.to_datetime(dl.index).normalize()
            dl = dl[~dl.index.duplicated(keep="last")].sort_index()
            cutoff = in_max if str(col).upper().endswith((".NS", ".BO")) else us_max
            dl = dl[dl.index <= cutoff]
            c_new = _extract_matrix_level(dl, "Close")
            if col not in c_new.columns or c_new[col].dropna().empty:
                failed.append(col)
                continue
            s_new = pd.to_numeric(c_new[col], errors="coerce").dropna()
            r_new = s_new.pct_change()
            still_bad = False
            for dt in r_new[r_new > max_jump].index:
                pos = r_new.index.get_loc(dt)
                if (r_new.iloc[pos + 1:min(len(r_new), pos + 6)] < snapback_drop).any():
                    still_bad = True
                    break
            if not still_bad:
                for dt in r_new[r_new < snapback_drop].index:
                    pos = r_new.index.get_loc(dt)
                    if (r_new.iloc[pos + 1:min(len(r_new), pos + 6)] > max_jump).any():
                        still_bad = True
                        break
            s_old = pd.to_numeric(close_df[col], errors="coerce").dropna()
            r_old = s_old.pct_change().dropna()
            was_sawtooth = _has_sawtooth(r_old)
            overlap = s_new.index.intersection(s_old.index)
            # For one-way split suspects (non-sawtooth), only replace if Yahoo live smoothed away the extreme split discontinuity
            if not was_sawtooth:
                has_extreme_new = bool((r_new > 1.5).any() or (r_new < -0.60).any())
                has_extreme_old = bool((r_old > 1.5).any() or (r_old < -0.60).any())
                # Also compare first bar in common overlap: if ratio differs by >1.5x, Yahoo retroactively split-adjusted it!
                ratio_changed = False
                if len(overlap) >= 20:
                    v_old = float(s_old.loc[overlap[0]])
                    v_new = float(s_new.loc[overlap[0]])
                    if v_old > 0 and v_new > 0 and max(v_old / v_new, v_new / v_old) > 1.4:
                        ratio_changed = True
                if has_extreme_new and not ratio_changed:
                    still_bad = True
                elif (not ratio_changed) and (has_extreme_new == has_extreme_old):
                    continue
            if still_bad:
                failed.append(col)
                continue
            fresh_frames.append(dl[[c for c in dl.columns if c[0] == col]])
            repaired.append(col)
        except Exception:
            failed.append(col)

    if fresh_frames:
        keep_mask = ~df.columns.get_level_values(0).isin(repaired)
        df_kept = df.loc[:, keep_mask].copy()
        df_kept.columns.names = [None, None]
        if getattr(df_kept.index, "tz", None) is not None:
            df_kept.index = df_kept.index.tz_localize(None)
        df_kept.index = pd.to_datetime(df_kept.index).normalize()
        fresh_all = pd.concat(fresh_frames, axis=1)
        fresh_all.columns.names = [None, None]
        fresh_all = fresh_all.reindex(df_kept.index)
        out = pd.concat([df_kept, fresh_all], axis=1).sort_index()
        _atomic_write_pickle(out, resolved)

    return {"scanned": len(scan_cols), "corrupt": corrupt_cols, "repaired": repaired, "failed": failed}


def sanitize_equity_matrix(
    cache_file: str = CACHE_FILE,
    prune_dead_older_days: int = 45,
) -> dict:
    """
    Enforces settled EOD equity hygiene on `historical_prices_matrix.pkl`:
      1) Drops unsettled partial intraday tail dates (`date > settled` with < 10% market coverage).
      2) Strips `^` index tickers, ETFs, and duplicate `.BO` where `.NS` exists.
      3) Strips truly dead stale `<20d` columns whose last valid bar is > `prune_dead_older_days` old.
    """
    resolved = _resolve_cache_path(cache_file)
    if not os.path.exists(resolved):
        return {"status": "missing"}
    df = pd.read_pickle(resolved)
    if df is None or df.empty or not isinstance(df.columns, pd.MultiIndex):
        return {"status": "empty"}

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()

    close_df = _extract_matrix_level(df, "Close")

    all_cols = list(close_df.columns)
    in_cols = [c for c in all_cols if str(c).upper().endswith((".NS", ".BO"))]
    ns_bases = {str(c).upper()[:-3] for c in in_cols if str(c).upper().endswith(".NS")}

    cols_to_drop = set()
    max_dt = df.index.max()
    stale_cutoff = max_dt - pd.Timedelta(days=prune_dead_older_days)
    for col in all_cols:
        c_u = str(col).upper()
        if is_excluded_etf_or_index(c_u):
            cols_to_drop.add(col)
            continue
        if c_u.endswith(".BO") and c_u[:-3] in ns_bases:
            cols_to_drop.add(col)
            continue
        s = close_df[col].dropna()
        if len(s) < 20 and (s.empty or pd.Timestamp(s.index.max()) < stale_cutoff):
            cols_to_drop.add(col)

    if cols_to_drop:
        keep_mask = ~df.columns.get_level_values(0).isin(cols_to_drop)
        df = df.loc[:, keep_mask].copy()
        close_df = _extract_matrix_level(df, "Close")

    # Enforce settled EOD session date fence per market AND minimum cross-market coverage
    in_settled = pd.Timestamp(get_latest_completed_session_date("IN")).normalize()
    us_settled = pd.Timestamp(get_latest_completed_session_date("US")).normalize()
    in_rem = [c for c in close_df.columns if str(c).upper().endswith((".NS", ".BO"))]
    us_rem = [c for c in close_df.columns if not str(c).upper().endswith((".NS", ".BO"))]
    in_cnt = close_df[in_rem].notna().sum(axis=1) if in_rem else pd.Series(0, index=close_df.index)
    us_cnt = close_df[us_rem].notna().sum(axis=1) if us_rem else pd.Series(0, index=close_df.index)
    in_min = max(25, int(len(in_rem) * 0.10)) if in_rem else 1
    us_min = max(25, int(len(us_rem) * 0.10)) if us_rem else 1
    row_keep = ((in_cnt >= in_min) & (close_df.index <= in_settled)) | (
        (us_cnt >= us_min) & (close_df.index <= us_settled)
    )
    dropped_rows = [d.strftime("%Y-%m-%d") for d in close_df.index[~row_keep]]
    if not row_keep.all():
        df = df.loc[row_keep].copy()

    df.columns.names = [None, None]
    _atomic_write_pickle(df, resolved)
    return {
        "dropped_cols": len(cols_to_drop),
        "dropped_rows": dropped_rows,
        "remaining_tickers": len(df.columns.get_level_values(0).unique()),
        "rows": len(df),
    }


def extract_market_ohlcv_panels(
    matrix_df: pd.DataFrame,
    market: str = "IN",
    allowed_tickers: list = None,
    include_etfs: bool = False,
    deduplicate_dual_in: bool = False,
    min_coverage_ratio: float = 0.15,
    min_abs: int = 10,
) -> dict:
    """
    Safely slices a shared multi-market OHLCV MultiIndex matrix (`IN` + `US`)
    down to strictly the requested market's active columns and settled trading dates.
    Eliminates cross-market holiday NaN trailing bars and opposite-market ticker leaks.
    """
    empty = {
        "close": pd.DataFrame(),
        "high": pd.DataFrame(),
        "low": pd.DataFrame(),
        "volume": pd.DataFrame(),
        "open": pd.DataFrame(),
    }
    if matrix_df is None or matrix_df.empty:
        return empty

    mkt = normalize_tracker_market(market)
    close_df = _extract_matrix_level(matrix_df, "Close")
    if close_df.empty:
        return empty

    all_cols = list(close_df.columns)
    if mkt == "IN":
        mkt_cols = [c for c in all_cols if str(c).upper().endswith((".NS", ".BO"))]
        if not include_etfs and mkt_cols:
            mkt_cols = [c for c in mkt_cols if not is_excluded_etf_or_index(c)]
        if deduplicate_dual_in and mkt_cols:
            ns_bases = {str(c).upper()[:-3] for c in mkt_cols if str(c).upper().endswith(".NS")}
            mkt_cols = [
                c for c in mkt_cols
                if not (str(c).upper().endswith(".BO") and str(c).upper()[:-3] in ns_bases)
            ]
    else:
        us_raw = [
            c for c in all_cols
            if not str(c).upper().endswith((".NS", ".BO"))
            and not str(c).startswith("^")
            and (include_etfs or not is_excluded_etf_or_index(c))
        ]
        mkt_cols = us_raw
        if not include_etfs and us_raw:
            try:
                from universe_security_master import load_universe_tickers
                active_eq = {str(t).strip().upper() for t in (load_universe_tickers("US", include_etfs=False) or [])}
                inter = [c for c in us_raw if str(c).strip().upper() in active_eq]
                if len(inter) >= max(20, int(len(us_raw) * 0.15)):
                    mkt_cols = inter
            except Exception:
                pass

    if allowed_tickers:
        req_up = {str(t).strip().upper() for t in allowed_tickers if t}
        matched = [c for c in mkt_cols if str(c).strip().upper() in req_up]
        if not matched and mkt == "IN":
            req_base = {r.replace(".NS", "").replace(".BO", "") for r in req_up}
            matched = [
                c for c in mkt_cols
                if str(c).strip().upper().replace(".NS", "").replace(".BO", "") in req_base
            ]
        mkt_cols = matched

    if not mkt_cols:
        return empty

    close_clean = filter_panel_active_market_dates(
        close_df[mkt_cols],
        min_coverage_ratio=min_coverage_ratio,
        min_abs=min_abs,
    )
    if close_clean.empty:
        return empty

    target_idx = close_clean.index
    target_cols = list(close_clean.columns)

    def _align(field_name: str) -> pd.DataFrame:
        sub = _extract_matrix_level(matrix_df, field_name)
        if sub.empty:
            return pd.DataFrame(index=target_idx, columns=target_cols, dtype=float)
        if getattr(sub.index, "tz", None) is not None:
            sub.index = sub.index.tz_localize(None)
        sub.index = pd.to_datetime(sub.index).normalize()
        sub = sub[~sub.index.duplicated(keep="last")]
        keep = [c for c in target_cols if c in sub.columns]
        if not keep:
            return pd.DataFrame(index=target_idx, columns=target_cols, dtype=float)
        return sub[keep].reindex(index=target_idx, columns=target_cols)

    return {
        "close": close_clean,
        "high": _align("High"),
        "low": _align("Low"),
        "volume": _align("Volume"),
        "open": _align("Open"),
    }

