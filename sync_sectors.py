import sqlite3
import yfinance as yf
import streamlit as st
import concurrent.futures
import time
from database import get_connection, get_all_fundamentals_cache
from views.true_market_leader import get_cached_universe


# Maximum consecutive failures before we assume Yahoo has IP-banned us
_MAX_CONSECUTIVE_FAILURES = 30


def sync_industries_ui():
    """
    Called from the Streamlit UI to populate the database with Yahoo Finance 
    industries while providing a visual progress bar.
    
    Handles HTTP 401 / rate-limit errors gracefully:
      • Limits concurrency to 5 threads.
      • Pauses 2 s every 25 completions.
      • Aborts early after 30 consecutive failures (likely an IP ban).
    """
    tickers = get_cached_universe("Deep Market (2500+ NSE Stocks)")
    db_cache = get_all_fundamentals_cache()
    
    # Filter tickers that already have an industry mapped
    missing_tickers = [t for t in tickers if db_cache.get(t, {}).get('industry', 'Unknown') == 'Unknown']
    
    if not missing_tickers:
        return "Complete"

    st.info(f"Preparing to fetch industries for {len(missing_tickers)} unknown tickers. "
            f"Using conservative rate-limiting (5 threads, 2 s cooldown every 25 requests) to avoid IP bans.")
    progress_text = "Downloading Yahoo Finance Industry categorizations..."
    my_bar = st.progress(0.0, text=progress_text)

    def fetch_industry(ticker):
        """Fetch a single ticker's industry from Yahoo Finance."""
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if info:
                ind = info.get('industry', 'Unknown')
                return ticker, ind, None
        except Exception as e:
            err_msg = str(e)
            # Print concise error to console for debugging
            if "401" in err_msg or "Rate" in err_msg:
                print(f"HTTP Error 401: {ticker}")
            else:
                print(f"Error fetching {ticker}: {err_msg[:120]}")
            return ticker, "Unknown", err_msg
        return ticker, "Unknown", None

    successful_maps = []
    total = len(missing_tickers)
    consecutive_failures = 0
    aborted = False
    
    # Use fewer workers to reduce the blast radius of rate-limiting
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(fetch_industry, t): t for t in missing_tickers}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_ticker):
            completed += 1
            try:
                ticker, industry, error = future.result()
            except Exception as exc:
                print(f"Unexpected worker exception: {exc}")
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    aborted = True
                    break
                continue
            
            if industry != "Unknown":
                # Save into cache with 0.0 for financials so TML fundamental fetching is preserved
                successful_maps.append((ticker, 0.0, 0.0, 0.0, industry))
                consecutive_failures = 0  # reset on success
            else:
                consecutive_failures += 1
            
            # Check for early abort (Yahoo has likely IP-banned us)
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                aborted = True
                break
            
            if completed % 10 == 0:
                pct = min(completed / total, 1.0)
                status_msg = f"{progress_text} ({completed}/{total}) — {len(successful_maps)} mapped"
                my_bar.progress(pct, text=status_msg)
            
            # Conservative rate-limit protection
            if completed % 25 == 0:
                time.sleep(2)

    if aborted:
        st.warning(
            f"⚠️ Yahoo Finance appears to be rate-limiting / blocking requests "
            f"({_MAX_CONSECUTIVE_FAILURES} consecutive failures). "
            f"Stopping early. Successfully mapped **{len(successful_maps)}** industries this run. "
            f"Try again later to continue where you left off."
        )

    my_bar.progress(1.0, text=f"Saving {len(successful_maps)} discovered sectors to local database...")
    
    if successful_maps:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.executemany('''
            INSERT OR IGNORE INTO fundamentals_cache 
            (ticker, eps_growth, sales_growth, roe, industry) 
            VALUES (?, ?, ?, ?, ?)
        ''', successful_maps)
        
        for row in successful_maps:
            cursor.execute('''
                UPDATE fundamentals_cache 
                SET industry = ?
                WHERE ticker = ? AND industry = 'Unknown'
            ''', (row[4], row[0]))
            
        conn.commit()
        conn.close()
    
    my_bar.empty()
    return "Complete"

def sync_industries_cli():
    print("Starting background Yahoo Finance Industry Sync...")
    # ... previous CLI logic (omitted since we just need the UI logic now, but could be added if needed)
    pass

if __name__ == "__main__":
    sync_industries_cli()


