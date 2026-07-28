"""
Hybrid Database Layer for Kush Tracker Lite.
Supports Turso Cloud DB (LibSQL) with automatic local SQLite fallback.
Includes automated retention purging (purge_stale_data) & Day 0 / Gap Backfilling (backfill_history_if_needed).
"""
import os
import sqlite3
from datetime import datetime, timedelta
import config

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

def get_turso_credentials():
    url = os.environ.get("TURSO_DATABASE_URL")
    token = os.environ.get("TURSO_AUTH_TOKEN")
    
    if (not url or not token) and HAS_STREAMLIT:
        try:
            if "database" in st.secrets:
                url = st.secrets["database"].get("turso_url", url)
                token = st.secrets["database"].get("turso_token", token)
        except Exception:
            pass
            
    return url, token

def get_connection():
    url, token = get_turso_credentials()
    
    if url and token:
        try:
            import libsql_experimental as libsql
            conn = libsql.connect(url, auth_token=token)
            return conn
        except Exception as e:
            print(f"[DB] Failed to connect to Turso Cloud DB ({e}). Falling back to local SQLite.")

    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_database():
    """Initializes tables, runs automated purging, and backfills Day 0 / stale gaps."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Focus List Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS focus_list (
            ticker TEXT PRIMARY KEY,
            market TEXT DEFAULT 'IN',
            entry_trigger REAL,
            stop_loss REAL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Institutional Footprints Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS institutional_footprints (
            ticker TEXT PRIMARY KEY,
            shock_date TEXT,
            shock_vol_multiple REAL,
            shock_close REAL,
            shock_high REAL,
            shock_low REAL,
            market TEXT DEFAULT 'IN',
            status TEXT DEFAULT 'Active'
        )
    """)

    # 3. Fundamentals Cache Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals_cache (
            ticker TEXT PRIMARY KEY,
            eps_yoy REAL,
            sales_yoy REAL,
            roe REAL,
            market_cap REAL,
            sector TEXT,
            industry TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 4. TML Snapshot Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tml_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            market TEXT,
            ticker TEXT,
            rank INTEGER,
            rs_score REAL,
            clenow_slope REAL,
            eps_yoy REAL,
            sales_yoy REAL
        )
    """)

    # 5. Intraday Signals History Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS intraday_signals_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ticker TEXT,
            signal_type TEXT,
            details TEXT,
            market TEXT
        )
    """)

    conn.commit()
    conn.close()
    
    # Run storage retention purging
    purge_stale_data()
    
    # Run Day 0 / Gap Backfilling
    backfill_history_if_needed()

def purge_stale_data():
    """Automated Storage Cleanup Routine."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cutoff_90d = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        cursor.execute("DELETE FROM tml_snapshot WHERE date < ?", (cutoff_90d,))
        
        cutoff_30d = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        cursor.execute("DELETE FROM institutional_footprints WHERE status LIKE 'Failed%' AND shock_date < ?", (cutoff_30d,))
        
        cutoff_14d = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("DELETE FROM intraday_signals_history WHERE timestamp < ?", (cutoff_14d,))
        
        cutoff_180d = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("DELETE FROM fundamentals_cache WHERE updated_at < ?", (cutoff_180d,))
        
        conn.commit()
        print("[DB Engine] Purge routine completed cleanly.")
    except Exception as e:
        print(f"[DB Engine] Purge routine warning: {e}")
    finally:
        conn.close()

def backfill_history_if_needed():
    """
    Day 0 & Long Gap Automated Backfill Engine:
    Detects if database is empty (Day 0) or if last footprint/snapshot is older than 2 days.
    Backfills 30-day historical volume shock breakouts into institutional_footprints.
    """
    conn = get_connection()
    cursor = conn.cursor()
    need_backfill = False
    
    try:
        cursor.execute("SELECT COUNT(*) FROM institutional_footprints")
        shock_count = cursor.fetchone()[0]
        
        if shock_count == 0:
            print("[DB Engine] Day 0 detected (empty DB). Triggering historical backfill...")
            need_backfill = True
        else:
            cursor.execute("SELECT MAX(shock_date) FROM institutional_footprints")
            max_date = cursor.fetchone()[0]
            if max_date:
                try:
                    last_dt = datetime.strptime(max_date, "%Y-%m-%d")
                    if (datetime.now() - last_dt).days >= 2:
                        print(f"[DB Engine] Gap detected (last scan {max_date} was {(datetime.now() - last_dt).days} days ago). Triggering backfill...")
                        need_backfill = True
                except Exception:
                    pass
    except Exception as e:
        print(f"[DB Engine] Check gap error: {e}")
    finally:
        conn.close()
        
    if need_backfill:
        run_day_zero_backfill()

def run_day_zero_backfill():
    """Scans past 30 days of market history for volume shock breakouts and populates DB."""
    try:
        from market_data import fetch_nifty_total_market_tickers, fetch_us_tickers
        from price_history_manager import fetch_incremental_history
        
        # 1. Backfill India Market
        in_tickers = fetch_nifty_total_market_tickers()[:60]
        yf_in = [t if t.endswith(".NS") or t.endswith(".BO") else f"{t}.NS" for t in in_tickers]
        hist_in = fetch_incremental_history(yf_in, days=40)
        
        for orig_t, yf_t in zip(in_tickers, yf_in):
            df = hist_in.get(yf_t)
            if df is not None and not df.empty and len(df) >= 20:
                for idx in range(len(df) - 1, max(len(df) - 30, 20), -1):
                    row = df.iloc[idx]
                    prev_row = df.iloc[idx - 1]
                    avg_vol = df['Volume'].iloc[max(0, idx - 20):idx].mean()
                    
                    if avg_vol > 0:
                        vol_mult = row['Volume'] / avg_vol
                        pct_change = ((row['Close'] - prev_row['Close']) / prev_row['Close']) * 100
                        
                        if vol_mult >= 2.0 and pct_change >= 1.5:
                            shock_date = df.index[idx].strftime("%Y-%m-%d")
                            log_volume_shock(orig_t, shock_date, vol_mult, float(row['Close']), float(row['High']), float(row['Low']), market="IN")
                            break
                            
        # 2. Backfill US Market
        us_tickers = fetch_us_tickers()[:40]
        hist_us = fetch_incremental_history(us_tickers, days=40)
        for t in us_tickers:
            df = hist_us.get(t)
            if df is not None and not df.empty and len(df) >= 20:
                for idx in range(len(df) - 1, max(len(df) - 30, 20), -1):
                    row = df.iloc[idx]
                    prev_row = df.iloc[idx - 1]
                    avg_vol = df['Volume'].iloc[max(0, idx - 20):idx].mean()
                    
                    if avg_vol > 0:
                        vol_mult = row['Volume'] / avg_vol
                        pct_change = ((row['Close'] - prev_row['Close']) / prev_row['Close']) * 100
                        
                        if vol_mult >= 2.0 and pct_change >= 1.5:
                            shock_date = df.index[idx].strftime("%Y-%m-%d")
                            log_volume_shock(t, shock_date, vol_mult, float(row['Close']), float(row['High']), float(row['Low']), market="US")
                            break
                            
        print("[DB Engine] Day 0 / Gap backfill completed successfully!")
    except Exception as e:
        print(f"[DB Engine] Error during backfill: {e}")

# --- FOCUS LIST CRUD ---

def add_to_focus_list(ticker: str, market: str = "IN", entry: float = 0.0, stop: float = 0.0, notes: str = "") -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO focus_list (ticker, market, entry_trigger, stop_loss, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (ticker.upper().strip(), market, entry, stop, notes))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error adding to focus list: {e}")
        return False
    finally:
        conn.close()

def remove_from_focus_list(ticker: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM focus_list WHERE ticker = ?", (ticker.upper().strip(),))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error removing from focus list: {e}")
        return False
    finally:
        conn.close()

def get_focus_list(market: str = None) -> list:
    conn = get_connection()
    conn.row_factory = sqlite3.Row if hasattr(conn, "row_factory") else None
    cursor = conn.cursor()
    try:
        query = "SELECT * FROM focus_list"
        params = []
        if market:
            query += " WHERE market = ?"
            params.append(market)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if conn.row_factory:
            return [dict(r) for r in rows]
        else:
            cols = [desc[0] for desc in cursor.description]
            return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        print(f"Error fetching focus list: {e}")
        return []
    finally:
        conn.close()

def update_focus_list_trade_plan(ticker: str, entry: float, stop: float, notes: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE focus_list
            SET entry_trigger = ?, stop_loss = ?, notes = ?
            WHERE ticker = ?
        """, (entry, stop, notes, ticker.upper().strip()))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating focus list trade plan: {e}")
        return False
    finally:
        conn.close()

# --- INSTITUTIONAL FOOTPRINTS (VOLUME SHOCKS) ---

def log_volume_shock(ticker: str, shock_date: str, vol_mult: float, close_price: float, high_price: float, low_price: float, market: str = "IN") -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO institutional_footprints
            (ticker, shock_date, shock_vol_multiple, shock_close, shock_high, shock_low, market, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'Active')
        """, (ticker.upper().strip(), shock_date, vol_mult, close_price, high_price, low_price, market))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error logging volume shock: {e}")
        return False
    finally:
        conn.close()

def get_active_volume_shocks(market: str = None) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        query = "SELECT * FROM institutional_footprints WHERE status = 'Active'"
        params = []
        if market:
            query += " AND market = ?"
            params.append(market)
        cursor.execute(query, params)
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, r)) for r in cursor.fetchall()]
    except Exception as e:
        print(f"Error fetching volume shocks: {e}")
        return []
    finally:
        conn.close()

def mark_shock_failed(ticker: str, reason: str):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE institutional_footprints
            SET status = ?
            WHERE ticker = ?
        """, (f"Failed: {reason}", ticker.upper().strip()))
        conn.commit()
    except Exception as e:
        print(f"Error marking shock failed: {e}")
    finally:
        conn.close()
