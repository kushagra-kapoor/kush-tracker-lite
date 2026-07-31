# Database module for Kush Tracker

import sqlite3
from datetime import datetime, timedelta
from config import DATABASE_PATH


def get_connection():
    """Get database connection (Turso if configured, otherwise local SQLite)."""
    # 1. Try Streamlit Secrets First (when running in app)
    try:
        import streamlit as st
        if "database" in st.secrets and st.secrets.database.get("db_type") == "turso":
            url = st.secrets.database.get("turso_url")
            token = st.secrets.database.get("turso_token")
            if url and token:
                import libsql_experimental as libsql
                return libsql.connect(database=url, auth_token=token)
    except Exception:
        pass
        
    # 2. Try Environment Variables (for GitHub Actions)
    import os
    env_url = os.environ.get("TURSO_URL")
    env_token = os.environ.get("TURSO_TOKEN")
    if env_url and env_token:
        try:
            import libsql_experimental as libsql
            return libsql.connect(database=env_url, auth_token=env_token)
        except Exception:
            pass
            
    # 3. Try manual TOML parsing (when running via python sync_macro.py)
    try:
        import toml
        import os
        secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            secrets = toml.load(secrets_path)
            if "database" in secrets and secrets["database"].get("db_type") == "turso":
                url = secrets["database"].get("turso_url")
                token = secrets["database"].get("turso_token")
                if url and token:
                    import libsql_experimental as libsql
                    return libsql.connect(database=url, auth_token=token)
    except Exception:
        pass
        
    # 3. Fallback to Local SQLite
    return sqlite3.connect(DATABASE_PATH)



def _fetch_all_dicts(cursor):
    rows = cursor.fetchall()
    if not rows: return []
    try:
        # Check if it already acts like a dict (e.g. sqlite3.Row)
        return [dict(r) for r in rows]
    except Exception:
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, r)) for r in rows]

def _fetch_one_dict(cursor):
    row = cursor.fetchone()
    if not row: return None
    try:
        return dict(row)
    except Exception:
        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))



def safe_execute(cursor, query, *args):
    try:
        if args:
            cursor.execute(query, *args)
        else:
            cursor.execute(query)
    except Exception:
        pass
def init_database():
    """Initialize the database with required tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Daily snapshot table
    safe_execute(cursor, '''
        CREATE TABLE IF NOT EXISTS daily_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            trend_state TEXT,
            rs_score REAL,
            action TEXT,
            reason TEXT,
            portfolio_risk_pct REAL,
            close_price REAL,
            distance_from_52w_high REAL,
            atr_state TEXT,
            add_on_ready TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, ticker)
        )
    ''')
    
    # Create index for faster queries
    safe_execute(cursor, '''
        CREATE INDEX IF NOT EXISTS idx_snapshot_date 
        ON daily_snapshot(date)
    ''')
    
    safe_execute(cursor, '''
        CREATE INDEX IF NOT EXISTS idx_snapshot_ticker 
        ON daily_snapshot(ticker)
    ''')
    
    # Signal history table for algo terminal
    safe_execute(cursor, '''
        CREATE TABLE IF NOT EXISTS signal_history (
            signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_generated TEXT NOT NULL,
            ticker TEXT NOT NULL,
            setup_type TEXT NOT NULL,
            regime TEXT,
            entry_price REAL,
            stop_price REAL,
            risk_percent REAL,
            holding_bias TEXT,
            confidence_score REAL,
            status TEXT DEFAULT 'Open',
            exit_price REAL,
            exit_date TEXT,
            R_multiple REAL,
            mfe REAL,
            mae REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date_generated, ticker, setup_type)
        )
    ''')
    
    safe_execute(cursor, '''
        CREATE INDEX IF NOT EXISTS idx_signal_status
        ON signal_history(status)
    ''')
    
    safe_execute(cursor, '''
        CREATE INDEX IF NOT EXISTS idx_signal_ticker
        ON signal_history(ticker)
    ''')
    
    # Add squat_alert column if it doesn't exist
    try:
        safe_execute(cursor, "ALTER TABLE signal_history ADD COLUMN squat_alert BOOLEAN DEFAULT 0")
    except Exception:
        pass

    # Sector leadership history table
    safe_execute(cursor, '''
        CREATE TABLE IF NOT EXISTS sector_leadership_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            rank INTEGER NOT NULL,
            industry TEXT NOT NULL,
            avg_rs REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, rank)
        )
    ''')
    
    # True Market Leader Snapshot Table
    safe_execute(cursor, '''
        CREATE TABLE IF NOT EXISTS tml_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            market TEXT NOT NULL,
            rank INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            tml_score REAL,
            rs_score REAL,
            action_status TEXT,
            industry TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, market, ticker)
        )
    ''')
    
    safe_execute(cursor, '''
        CREATE INDEX IF NOT EXISTS idx_tml_date ON tml_snapshot(date)
    ''')
    
    safe_execute(cursor, '''
        CREATE INDEX IF NOT EXISTS idx_tml_ticker ON tml_snapshot(ticker)
    ''')

    safe_execute(cursor, '''
        CREATE INDEX IF NOT EXISTS idx_sector_leadership_date
        ON sector_leadership_history(date)
    ''')
    
    # Daily Journal table
    safe_execute(cursor, '''
        CREATE TABLE IF NOT EXISTS daily_journal (
            date TEXT PRIMARY KEY,
            market_bias TEXT,
            newsflow TEXT,
            net_new_highs INTEGER,
            net_new_lows INTEGER,
            leaders_behavior TEXT,
            journal_notes TEXT,
            automated_filled BOOLEAN DEFAULT 1,
            is_distribution_day BOOLEAN DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Add is_distribution_day column if it doesn't exist (Migration)
    try:
        safe_execute(cursor, "ALTER TABLE daily_journal ADD COLUMN is_distribution_day BOOLEAN DEFAULT 0")
    except Exception:
        pass

    # Market Cap cache table
    safe_execute(cursor, '''
        CREATE TABLE IF NOT EXISTS market_cap_cache (
            ticker TEXT PRIMARY KEY,
            market_cap REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Intraday signals history table
    safe_execute(cursor, '''
        CREATE TABLE IF NOT EXISTS intraday_signals_history (
            market TEXT,
            ticker TEXT,
            signal_name TEXT,
            date DATE,
            PRIMARY KEY (market, ticker, signal_name, date)
        )
    ''')
    
    # Indexes for fast querying
    safe_execute(cursor, '''
        CREATE INDEX IF NOT EXISTS idx_intraday_signals_date 
        ON intraday_signals_history(date)
    ''')

    # Fundamentals Cache Table
    safe_execute(cursor, '''
        CREATE TABLE IF NOT EXISTS fundamentals_cache (
            ticker TEXT PRIMARY KEY,
            eps_yoy REAL,
            sales_yoy REAL,
            roe REAL,
            industry TEXT,
            market_cap REAL DEFAULT 0.0,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    try:
        safe_execute(cursor, "ALTER TABLE fundamentals_cache ADD COLUMN market_cap REAL DEFAULT 0.0")
    except Exception:
        pass

    # Focus List table
    safe_execute(cursor, '''
        CREATE TABLE IF NOT EXISTS focus_list (
            ticker TEXT PRIMARY KEY,
            market TEXT,
            entry_trigger REAL DEFAULT 0.0,
            stop_loss REAL DEFAULT 0.0,
            notes TEXT DEFAULT '',
            added_date TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Institutional Footprints (Volume Shocks)
    safe_execute(cursor, '''
        CREATE TABLE IF NOT EXISTS institutional_footprints (
            ticker TEXT PRIMARY KEY,
            shock_date TEXT NOT NULL,
            shock_vol_multiple REAL,
            shock_close REAL,
            shock_high REAL,
            shock_low REAL,
            market TEXT,
            status TEXT DEFAULT 'Active',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    safe_execute(cursor, '''
        CREATE TABLE IF NOT EXISTS market_liquidity_daily (
            date TEXT,
            market TEXT DEFAULT 'IN',
            monthly_turnover_k_cr REAL,
            sma_200 REAL,
            PRIMARY KEY (date, market)
        )
    ''')

    safe_execute(cursor, '''
        CREATE TABLE IF NOT EXISTS market_breadth_daily (
            date TEXT,
            market TEXT DEFAULT 'IN',
            net_new_highs INTEGER,
            above_50_pct REAL,
            PRIMARY KEY (date, market)
        )
    ''')

    safe_execute(cursor, '''
        CREATE TABLE IF NOT EXISTS corporate_announcements (
            id TEXT PRIMARY KEY,
            ticker TEXT,
            date_time TEXT,
            title TEXT,
            description TEXT,
            pdf_link TEXT
        )
    ''')

    safe_execute(cursor, '''
        CREATE TABLE IF NOT EXISTS global_regime_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            market TEXT NOT NULL,
            benchmark_ticker TEXT NOT NULL,
            close REAL,
            sma50 REAL,
            sma200 REAL,
            dd_count INTEGER,
            ftd_detected BOOLEAN,
            regime_label TEXT,
            choch_label TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, market, benchmark_ticker)
        )
    ''')

    safe_execute(cursor, '''
        CREATE TABLE IF NOT EXISTS global_etf_momentum (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            name TEXT,
            return_1m REAL,
            return_3m REAL,
            return_6m REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, ticker)
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized successfully")

def save_snapshot(snapshot_data: list):
    """
    Save daily snapshot data to database.
    
    Args:
        snapshot_data: List of dictionaries containing snapshot data
    """
    conn = get_connection()
    cursor = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    
    for row in snapshot_data:
        safe_execute(cursor, '''
            INSERT OR REPLACE INTO daily_snapshot 
            (date, ticker, trend_state, rs_score, action, reason, 
             portfolio_risk_pct, close_price, distance_from_52w_high, 
             atr_state, add_on_ready)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            today,
            row.get('ticker'),
            row.get('trend_state'),
            row.get('rs_score'),
            row.get('action'),
            row.get('reason'),
            row.get('portfolio_risk_pct'),
            row.get('close_price'),
            row.get('distance_from_52w_high'),
            row.get('atr_state'),
            row.get('add_on_ready'),
        ))
    
    conn.commit()
    conn.close()
    print(f"✅ Saved {len(snapshot_data)} snapshots for {today}")


def get_recent_snapshots(ticker: str = None, days: int = 30):
    """
    Get recent snapshots from database.
    
    Args:
        ticker: Optional ticker to filter by
        days: Number of days to look back
    
    Returns:
        List of snapshot records
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if ticker:
        safe_execute(cursor, '''
            SELECT * FROM daily_snapshot 
            WHERE ticker = ? 
            ORDER BY date DESC 
            LIMIT ?
        ''', (ticker, days))
    else:
        safe_execute(cursor, '''
            SELECT * FROM daily_snapshot 
            ORDER BY date DESC 
            LIMIT ?
        ''', (days * 20,))  # Assume max 20 holdings
    
    results = cursor.fetchall()
    conn.close()
    return results


def get_signal_change_days(ticker: str) -> int:
    """
    Get the number of days since the last signal change for a ticker.
    
    Args:
        ticker: Stock ticker
    
    Returns:
        Number of days since signal changed
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    safe_execute(cursor, '''
        SELECT date, action FROM daily_snapshot 
        WHERE ticker = ? 
        ORDER BY date DESC
    ''', (ticker,))
    
    results = cursor.fetchall()
    conn.close()
    
    if len(results) < 2:
        return 0
    
    current_action = results[0][1]
    days = 0
    
    for date, action in results:
        if action == current_action:
            days += 1
        else:
            break
    
    return days


# =============================================================================
# SIGNAL HISTORY FUNCTIONS
# =============================================================================

def save_signal(signal: dict):
    """Save a new signal to signal_history. Skips duplicates (same day + ticker + setup)."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        safe_execute(cursor, '''
            INSERT OR IGNORE INTO signal_history
            (date_generated, ticker, setup_type, regime, entry_price, stop_price,
             risk_percent, holding_bias, confidence_score, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Open')
        ''', (
            signal.get('date_generated', datetime.now().strftime('%Y-%m-%d')),
            signal['ticker'],
            signal['setup_type'],
            signal.get('regime'),
            signal.get('entry_price'),
            signal.get('stop_price'),
            signal.get('risk_percent'),
            signal.get('holding_bias'),
            signal.get('confidence_score'),
        ))
        conn.commit()
        inserted = cursor.rowcount > 0
    except Exception as e:
        print(f"Error saving signal: {e}")
        inserted = False
    finally:
        conn.close()
    return inserted


def get_open_signals():
    """Get all signals with status = 'Open'."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM signal_history WHERE status = 'Open'")
    rows = _fetch_all_dicts(cursor)
    conn.close()
    return rows


def update_signal_outcome(signal_id: int, status: str, exit_price: float,
                           exit_date: str, r_multiple: float,
                           mfe: float = None, mae: float = None):
    """Update a signal's outcome after tracking."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE signal_history
        SET status = ?, exit_price = ?, exit_date = ?,
            R_multiple = ?, mfe = ?, mae = ?
        WHERE signal_id = ?
    ''', (status, exit_price, exit_date, r_multiple, mfe, mae, signal_id))
    conn.commit()
    conn.close()

def mark_squat_alert(signal_id: int):
    """Mark a signal as having a squat failure on Day 1."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE signal_history
        SET squat_alert = 1
        WHERE signal_id = ?
    ''', (signal_id,))
    conn.commit()
    conn.close()


def get_all_signals():
    """Get all signals (for edge performance analytics)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM signal_history ORDER BY date_generated DESC")
    rows = _fetch_all_dicts(cursor)
    conn.close()
    return rows


if __name__ == '__main__':
    init_database()


def save_sector_leadership(top_industries: list):
    """
    Save today's top-N industry groups to the database.

    Args:
        top_industries: List of dicts with 'Industry' and 'Avg_RS' keys,
                        ordered by rank (index 0 = rank 1).
    """
    conn = get_connection()
    cursor = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')

    for i, ind in enumerate(top_industries):
        name = ind.get('Industry', ind) if isinstance(ind, dict) else str(ind)
        avg_rs = ind.get('Avg_RS', 0) if isinstance(ind, dict) else 0
        cursor.execute('''
            INSERT OR REPLACE INTO sector_leadership_history
            (date, rank, industry, avg_rs)
            VALUES (?, ?, ?, ?)
        ''', (today, i + 1, name, avg_rs))

    conn.commit()
    conn.close()


def get_sector_leadership_history(days: int = 30) -> list:
    """
    Get sector leadership snapshots over the last N days.

    Returns:
        List of dicts with date, rank, industry, avg_rs.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT date, rank, industry, avg_rs
        FROM sector_leadership_history
        WHERE date >= date('now', ? || ' days')
        ORDER BY date DESC, rank ASC
    ''', (f'-{days}',))
    rows = _fetch_all_dicts(cursor)
    conn.close()
    return rows


def get_latest_top_sectors(limit: int = 5) -> list:
    """
    Returns the names of the most recently cached top N sectors.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get the max date
    cursor.execute('SELECT MAX(date) FROM sector_leadership_history')
    latest_date_row = cursor.fetchone()
    if not latest_date_row or not latest_date_row[0]:
        conn.close()
        return []
        
    latest_date = latest_date_row[0]
    cursor.execute('''
        SELECT industry FROM sector_leadership_history
        WHERE date = ?
        ORDER BY rank ASC
        LIMIT ?
    ''', (latest_date, limit))
    
    results = [row[0] for row in cursor.fetchall()]
    conn.close()
    return results


# =============================================================================
# INTRADAY SIGNALS FUNCTIONS
# =============================================================================

def save_intraday_signals(market: str, signals_list: list):
    """
    Save a list of triggered intraday signals.
    signals_list: list of dicts {'ticker': str, 'signal_name': str}
    """
    conn = get_connection()
    cursor = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    
    for sig in signals_list:
        ticker = sig['ticker'].replace('.NS', '')  # Ensure clean ticker
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO intraday_signals_history
                (market, ticker, signal_name, date)
                VALUES (?, ?, ?, ?)
            ''', (market, ticker, sig['signal_name'], today))
        except Exception:
            pass
        
    conn.commit()
    conn.close()

def get_recent_intraday_signals(market: str, days: int = 3) -> dict:
    """
    Returns a dictionary mapping tickers to a list of tuples: (signal_name, date_str).
    Only includes signals from the last `days` days.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Calculate cutoff date
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    results = {}
    try:
        cursor.execute('''
            SELECT ticker, signal_name, date FROM intraday_signals_history
            WHERE market = ? AND date >= ?
            ORDER BY date DESC
        ''', (market, cutoff_date))
        
        for row in cursor.fetchall():
            ticker = row[0]
            signal = row[1]
            date_val = row[2]
            
            if ticker not in results:
                results[ticker] = []
            results[ticker].append((signal, date_val))
    except Exception as e:
        print(f"Error fetching recent intraday signals: {e}")
        
    conn.close()
    return results


# =============================================================================
# MARKET REGIME FUNCTIONS
# =============================================================================

def save_market_regime(regime_title: str):
    """
    Save the current market regime title.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_regime_cache (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            regime_title TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        INSERT OR REPLACE INTO market_regime_cache (id, regime_title, last_updated)
        VALUES (1, ?, CURRENT_TIMESTAMP)
    ''', (regime_title,))
    conn.commit()
    conn.close()

def get_market_regime() -> str:
    """
    Get the cached market regime title.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT regime_title FROM market_regime_cache WHERE id = 1')
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else "Neutral/Mixed Environment"
    except sqlite3.OperationalError:
        conn.close()
        return "Neutral/Mixed Environment"

def save_market_liquidity(records):
    """
    Save historical market liquidity records to database.
    records is a list of tuples: (date, market, monthly_turnover_k_cr, sma_200)
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.executemany('''
        INSERT OR REPLACE INTO market_liquidity_daily 
        (date, market, monthly_turnover_k_cr, sma_200)
        VALUES (?, ?, ?, ?)
    ''', records)
    conn.commit()
    conn.close()

def get_market_liquidity(days=300, market='IN'):
    """
    Fetch market liquidity data for plotting.
    """
    import pandas as pd
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM market_liquidity_daily WHERE market = ? ORDER BY date DESC LIMIT ?", conn, params=(market, days,))
    conn.close()
    if not df.empty:
        df = df.sort_values('date').reset_index(drop=True)
    return df

# =============================================================================
# JOURNAL FUNCTIONS
# =============================================================================

def save_journal_entry(journal_data: dict):
    """
    Save or update a daily journal entry.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO daily_journal
            (date, market_bias, newsflow, net_new_highs, net_new_lows, 
             leaders_behavior, journal_notes, automated_filled, is_distribution_day, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            journal_data['date'],
            journal_data.get('market_bias', ''),
            journal_data.get('newsflow', ''),
            journal_data.get('net_new_highs', 0),
            journal_data.get('net_new_lows', 0),
            journal_data.get('leaders_behavior', ''),
            journal_data.get('journal_notes', ''),
            journal_data.get('automated_filled', True),
            int(journal_data.get('is_distribution_day', False))
        ))
        conn.commit()
        success = True
    except Exception as e:
        print(f"Error saving journal: {e}")
        success = False
    finally:
        conn.close()
    
    return success

def get_journal_history(days: int = 30) -> list:
    """
    Get journal history for the last N days.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT *
        FROM daily_journal
        WHERE date >= date('now', ? || ' days')
        ORDER BY date DESC
    ''', (f'-{days}',))
    rows = _fetch_all_dicts(cursor)
    conn.close()
    return rows

def get_journal_entry(date_str: str) -> dict:
    """
    Get a specific journal entry by date.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM daily_journal WHERE date = ?", (date_str,))
    result = _fetch_one_dict(cursor)
    conn.close()
    return result


# =============================================================================
# FUNDAMENTALS CACHE
# =============================================================================

def _safe_float(val, default=0.0):
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

def get_all_fundamentals_cache() -> dict:
    """
    Load the entire fundamentals cache into memory to avoid SQLite DB Locks
    while multithreading.
    Returns: {ticker: {'eps_growth': x, 'sales_growth': y, 'roe': z, 'industry': w}}
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM fundamentals_cache")
        rows = _fetch_all_dicts(cursor)
        cache = {}
        for r in rows:
            # Strip .NS to match the UI pipeline which strips suffixes for presentation
            clean_ticker = r['ticker'].replace('.NS', '')
            # We map DB columns (eps_yoy, sales_yoy) to app dictionary keys (eps_growth, sales_growth)
            cache[clean_ticker] = {
                'eps_growth': _safe_float(r['eps_yoy'] if 'eps_yoy' in r.keys() else r.get('eps_growth', 0.0)),
                'sales_growth': _safe_float(r['sales_yoy'] if 'sales_yoy' in r.keys() else r.get('sales_growth', 0.0)),
                'roe': _safe_float(r['roe']),
                'industry': r['industry'] if r['industry'] is not None else 'Unknown',
                'market_cap': _safe_float(r['market_cap']) if 'market_cap' in r.keys() else 0.0,
                'updated_at': r['updated_at']
            }
        return cache
    except Exception as e:
        print(f"Error reading fundamentals cache: {e}")
        return {}
    finally:
        conn.close()

def save_tml_snapshot(market: str, leaders: list, top_n: int = 20):
    """
    Save the top N true market leaders for a given market to the snapshot table.
    """
    if not leaders:
        return
        
    date_str = datetime.now().strftime('%Y-%m-%d')
    conn = get_connection()
    
    # Ensure leaders are sorted by TML score just in case
    sorted_leaders = sorted(leaders, key=lambda x: float(x.get('tml_score', 0)), reverse=True)
    top_leaders = sorted_leaders[:top_n]
    
    try:
        cursor = conn.cursor()
        for i, leader in enumerate(top_leaders):
            ticker = str(leader.get('ticker', ''))
            tml_score = float(leader.get('tml_score', 0.0))
            rs_score = float(leader.get('rs_score', 0.0))
            action_status = str(leader.get('Action_Status', 'Unknown'))
            industry = str(leader.get('industry', 'Unknown'))
            
            cursor.execute('''
                INSERT OR REPLACE INTO tml_snapshot 
                (date, market, rank, ticker, tml_score, rs_score, action_status, industry)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (date_str, market, i + 1, ticker, tml_score, rs_score, action_status, industry))
            
        conn.commit()
    except Exception as e:
        print(f"Error saving TML snapshot: {e}")
        try:
            import streamlit as st
            st.error(f"Database Error (save_tml_snapshot): {e}")
        except Exception:
            pass
    finally:
        conn.close()

def get_current_tml_leaders(market: str) -> set:
    """
    Returns a set of the raw tickers currently in the Top 20 True Market Leaders for the given market.
    Used for cross-referencing in the Intraday Monitor.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Get the most recent date available for this market
        cursor.execute('''
            SELECT MAX(date) FROM tml_snapshot WHERE market = ?
        ''', (market,))
        
        latest_date_result = cursor.fetchone()
        if not latest_date_result or not latest_date_result[0]:
            return set()
            
        latest_date = latest_date_result[0]
        
        cursor.execute('''
            SELECT ticker FROM tml_snapshot
            WHERE market = ? AND date = ? AND rank <= 20
        ''', (market, latest_date))
        
        results = cursor.fetchall()
        return {row[0].replace('.NS', '') for row in results}
    except Exception as e:
        print(f"Error getting current TML leaders: {e}")
        return set()
    finally:
        conn.close()

def get_hat_stocks(market: str) -> set:
    """
    Returns a set of the raw tickers ranked 21-40 with RS > 85.
    Used for marking next-tier momentum leaders.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT MAX(date) FROM tml_snapshot WHERE market = ?
        ''', (market,))
        
        latest_date_result = cursor.fetchone()
        if not latest_date_result or not latest_date_result[0]:
            return set()
            
        latest_date = latest_date_result[0]
        
        cursor.execute('''
            SELECT ticker FROM tml_snapshot
            WHERE market = ? AND date = ? AND rank > 20 AND rank <= 40 AND rs_score > 40
        ''', (market, latest_date))
        
        results = cursor.fetchall()
        return {row[0].replace('.NS', '') for row in results}
    except Exception as e:
        print(f"Error getting hat stocks: {e}")
        return set()
    finally:
        conn.close()

def get_top_5_rs_leaders(market: str = 'INDIA') -> list:
    """
    Returns the top 5 highest RS stocks from the most recent TML snapshot.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT MAX(date) FROM tml_snapshot WHERE market = ?
        ''', (market,))
        
        latest_date_result = cursor.fetchone()
        if not latest_date_result or not latest_date_result[0]:
            return []
            
        latest_date = latest_date_result[0]
        
        cursor.execute('''
            SELECT ticker, rs_score, industry, tml_score
            FROM tml_snapshot
            WHERE market = ? AND date = ? AND rs_score IS NOT NULL
            ORDER BY rs_score DESC
            LIMIT 5
        ''', (market, latest_date))
        
        results = cursor.fetchall()
        return [{'ticker': row[0], 'rs_score': row[1], 'industry': row[2], 'tml_score': row[3]} for row in results]
    except Exception as e:
        print(f"Error getting top 5 RS leaders: {e}")
        return []
    finally:
        conn.close()

def get_tml_persistence(market: str, days: int = 90):
    """
    Returns a dictionary of {ticker: days_on_list} for the given market 
    over the last `days` distinct snapshot dates.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT date FROM tml_snapshot 
            WHERE market = ? 
            ORDER BY date DESC LIMIT ?
        ''', (market, days))
        
        dates_result = cursor.fetchall()
        if not dates_result:
            return {}
            
        oldest_date = dates_result[-1][0]
        
        cursor.execute('''
            SELECT ticker, COUNT(*) as days_on_list
            FROM tml_snapshot
            WHERE market = ? AND date >= ?
            GROUP BY ticker
        ''', (market, oldest_date))
        
        results = cursor.fetchall()
        return {row[0]: row[1] for row in results}
    except Exception as e:
        print(f"Error getting TML persistence: {e}")
        return {}
    finally:
        conn.close()

def get_tml_hall_of_fame(market: str, days: int = 90, limit: int = 20):
    """
    Returns the most persistent stocks in the top 20 over the last `days`.
    Includes basic info about their most recent appearance.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # We find the last `days` distinct dates the scan was run
        cursor.execute('''
            SELECT DISTINCT date FROM tml_snapshot 
            WHERE market = ? 
            ORDER BY date DESC LIMIT ?
        ''', (market, days))
        
        dates_result = cursor.fetchall()
        if not dates_result:
            return []
            
        oldest_date = dates_result[-1][0]
        
        # Get count and join with the most recent row for that ticker to get latest industry and score
        cursor.execute('''
            WITH Counts AS (
                SELECT ticker, COUNT(*) as days_on_list, MAX(date) as last_seen
                FROM tml_snapshot
                WHERE market = ? AND date >= ?
                GROUP BY ticker
            )
            SELECT c.ticker, c.days_on_list, c.last_seen, t.industry, t.tml_score, t.rs_score, t.action_status
            FROM Counts c
            JOIN tml_snapshot t ON c.ticker = t.ticker AND c.last_seen = t.date AND t.market = ?
            ORDER BY c.days_on_list DESC, t.tml_score DESC
            LIMIT ?
        ''', (market, oldest_date, market, limit))
        
        results = cursor.fetchall()
        
        hof_list = []
        for row in results:
            hof_list.append({
                'ticker': row[0],
                'days_on_list': row[1],
                'last_seen': row[2],
                'industry': row[3],
                'tml_score': row[4],
                'rs_score': row[5],
                'action_status': row[6]
            })
            
        return hof_list
    except Exception as e:
        print(f"Error getting TML hall of fame: {e}")
        return []
    finally:
        conn.close()

def get_historical_rs_for_tickers(market: str, tickers: list, days: int = 10) -> dict:
    """
    Fetches the historical RS scores for a specific list of tickers over the last N snapshot days.
    Returns a dictionary mapping ticker -> list of historical RS scores (oldest to newest).
    """
    if not tickers:
        return {}
        
    try:
        with get_connection() as conn:
            # 1. Get the last N unique dates for this market
            date_query = """
            SELECT DISTINCT date FROM tml_snapshot 
            WHERE market = ? 
            ORDER BY date DESC LIMIT ?
            """
            recent_dates = [row[0] for row in conn.execute(date_query, (market, days)).fetchall()]
            
            if not recent_dates:
                return {}
                
            recent_dates.sort() # Sort oldest to newest
            
            # 2. Build the query to fetch RS scores for the specific tickers
            placeholders = ','.join(['?'] * len(tickers))
            query = f"""
                SELECT ticker, date, rs_score
                FROM tml_snapshot
                WHERE market = ? 
                  AND date >= ? 
                  AND ticker IN ({placeholders})
                ORDER BY ticker, date ASC
            """
            
            # Parameter list: market, oldest_date, *tickers
            params = [market, recent_dates[0]] + tickers
            
            cursor = conn.execute(query, params)
            
            # 3. Group the results by ticker
            history = {ticker: [] for ticker in tickers}
            for row in cursor:
                ticker, date, rs = row
                if rs is not None:
                    history[ticker].append(rs)
                    
            return history
    except Exception as e:
        print(f"Error fetching historical RS for tails: {e}")
        return {}

def save_fundamentals_cache(data_list: list):
    """
    Batch save successful fundamentals to bypass Yahoo Finance rate limits securely.
    Args:
        data_list: List of dicts [{'ticker': 'AAPL', 'eps_growth': 10.0, ...}]
    """
    if not data_list:
        return
        
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        for item in data_list:
            cursor.execute('''
                INSERT OR REPLACE INTO fundamentals_cache 
                (ticker, eps_growth, sales_growth, roe, industry, market_cap, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                item['ticker'],
                item.get('eps_growth', 0.0),
                item.get('sales_growth', 0.0),
                item.get('roe', 0.0),
                item.get('industry', 'Unknown'),
                item.get('market_cap', 0.0)
            ))
        conn.commit()
        print(f"Securely cached {len(data_list)} stock fundamentals to local database.")
    except Exception as e:
        print(f"Error saving fundamentals cache: {e}")
    finally:
        conn.close()


# ==========================================
# Focus List Functions
# ==========================================

def add_to_focus_list(ticker: str, market: str) -> bool:
    """Add a stock to the Focus List."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO focus_list (ticker, market, entry_trigger, stop_loss, notes)
                VALUES (?, ?, 0.0, 0.0, '')
            ''', (ticker, market))
        except Exception:
            pass
        conn.commit()
        return True
    except Exception as e:
        print(f"Error adding to focus list: {e}")
        return False
    finally:
        conn.close()

def remove_from_focus_list(ticker: str) -> bool:
    """Remove a stock from the Focus List."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM focus_list WHERE ticker = ?', (ticker,))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error removing from focus list: {e}")
        return False
    finally:
        conn.close()

def get_focus_list(market: str = None) -> list:
    """Get all stocks on the Focus List, optionally filtered by market."""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = 'SELECT * FROM focus_list'
    params = ()
    
    if market:
        query += ' WHERE market = ?'
        params = (market,)
        
    query += ' ORDER BY added_date DESC'
    
    try:
        cursor.execute(query, params)
        return _fetch_all_dicts(cursor)
    except Exception as e:
        print(f"Error fetching focus list: {e}")
        return []
    finally:
        conn.close()

def update_focus_list_trade_plan(ticker: str, entry: float, stop: float, notes: str) -> bool:
    """Update the trading plan parameters for a stock on the Focus List."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE focus_list 
            SET entry_trigger = ?, stop_loss = ?, notes = ?
            WHERE ticker = ?
        ''', (entry, stop, notes, ticker))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating focus list plan: {e}")
        return False
    finally:
        conn.close()

def log_volume_shock(ticker: str, shock_date: str, vol_mult: float, close_price: float, high_price: float, low_price: float, market: str) -> bool:
    """Log an institutional footprint (Volume Shock) to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Check if an active shock already exists for this ticker
        cursor.execute("SELECT shock_date, shock_vol_multiple FROM institutional_footprints WHERE ticker = ? AND status = 'Active'", (ticker,))
        row = cursor.fetchone()
        
        if row:
            # Only overwrite if the new shock is significantly bigger (e.g., higher volume multiple)
            # or if it's been more than a few days. For simplicity, we just update it.
            pass
            
        cursor.execute('''
            INSERT OR REPLACE INTO institutional_footprints 
            (ticker, shock_date, shock_vol_multiple, shock_close, shock_high, shock_low, market, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'Active')
        ''', (ticker, shock_date, vol_mult, close_price, high_price, low_price, market))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error logging volume shock: {e}")
        return False
    finally:
        conn.close()

def get_active_volume_shocks(market: str = None):
    """Get all active volume shocks."""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM institutional_footprints WHERE status = 'Active'"
    params = []
    if market:
        query += " AND market = ?"
        params.append(market)
        
    try:
        cursor.execute(query, params)
        return _fetch_all_dicts(cursor)
    except Exception as e:
        print(f"Error fetching active volume shocks: {e}")
        return []
    finally:
        conn.close()

def mark_shock_failed(ticker: str, reason: str):
    """Mark a volume shock as Failed."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE institutional_footprints 
            SET status = ? 
            WHERE ticker = ?
        ''', (f"Failed: {reason}", ticker))
        conn.commit()
    except Exception as e:
        print(f"Error updating shock status: {e}")
    finally:
        conn.close()

def backfill_volume_shock(ticker: str, shock_date: str, vol_mult: float, close_price: float, high_price: float, low_price: float, market: str) -> bool:
    """Only inserts a volume shock if the ticker has absolutely no history in the footprint table."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT 1 FROM institutional_footprints WHERE ticker = ?", (ticker,))
        if cursor.fetchone():
            return False # Already exists (Active or Failed), don't resurrect or overwrite it!
            
        cursor.execute('''
            INSERT INTO institutional_footprints 
            (ticker, shock_date, shock_vol_multiple, shock_close, shock_high, shock_low, market, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'Active')
        ''', (ticker, shock_date, vol_mult, close_price, high_price, low_price, market))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()

def auto_backfill_footprints(ticker_data: dict, market: str, lookback_days: int = 30) -> int:
    """
    Scans a dictionary of dataframes for historic volume shocks 
    and logs them into the database automatically if missing.
    """
    count = 0
    if not ticker_data: return count
        
    for t, df_original in ticker_data.items():
        try:
            df = df_original.copy()
            df = df.dropna(subset=['Close'])
            if len(df) < 22: continue
                
            df = df.iloc[-(lookback_days + 21):].copy()
            if df.empty: continue
            
            df['AvgVol20'] = df['Volume'].shift(1).rolling(window=20).mean()
            df['PrevClose'] = df['Close'].shift(1)
            
            valid_rows = (df['AvgVol20'] > 0) & (df['PrevClose'] > 0)
            df = df[valid_rows].copy()
            
            df['VolExp'] = df['Volume'] / df['AvgVol20']
            df['DailyPct'] = ((df['Close'] - df['PrevClose']) / df['PrevClose']) * 100
            
            if market == "INDIA":
                df['ADTV'] = df['AvgVol20'] * df['Close']
                adtv_threshold = 50000000  # 5 Cr
            else:
                df['ADTV'] = (df['AvgVol20'] * df['Close']) / 1000000
                adtv_threshold = 5.0       # $5M
                
            df_lookback = df.iloc[-lookback_days:]
            shocks = df_lookback[(df_lookback['VolExp'] >= 5.0) & (df_lookback['DailyPct'] >= 4.8) & (df_lookback['ADTV'] >= adtv_threshold)]
            
            for date_idx, row in shocks.iterrows():
                shock_date_str = date_idx.strftime('%Y-%m-%d')
                clean_t = str(t).replace('.NS', '') if market == "INDIA" else str(t)
                if backfill_volume_shock(clean_t, shock_date_str, float(row['VolExp']), float(row['Close']), float(row['High']), float(row['Low']), market):
                    count += 1
        except Exception:
            continue
    return count

def save_corporate_announcements(records):
    """
    Save corporate announcements.
    records is a list of tuples: (id, ticker, date_time, title, description, pdf_link)
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.executemany('''
        INSERT OR REPLACE INTO corporate_announcements 
        (id, ticker, date_time, title, description, pdf_link)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', records)
    conn.commit()
    conn.close()

def cleanup_old_announcements(days=7):
    """
    Deletes announcements older than N days to save disk space.
    """
    from datetime import datetime, timedelta
    conn = get_connection()
    cursor = conn.cursor()
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        DELETE FROM corporate_announcements 
        WHERE date_time < ?
    ''', (cutoff_date,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted

def get_recent_announcements(days=7):
    """
    Fetch corporate announcements for the last N days.
    """
    import pandas as pd
    from datetime import datetime, timedelta
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d 00:00:00')
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM corporate_announcements WHERE date_time >= ? ORDER BY date_time DESC", conn, params=(cutoff_date,))
    conn.close()
    return df


# =============================================================================
# GLOBAL MACRO & ETF FUNCTIONS
# =============================================================================

def save_global_regime(regime_data: list):
    import sqlite3
    conn = get_connection()
    cursor = conn.cursor()
    for r in regime_data:
        cursor.execute('''
            INSERT OR REPLACE INTO global_regime_history
            (date, market, benchmark_ticker, close, sma50, sma200, dd_count, ftd_detected, regime_label, choch_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            r['date'], r['market'], r['benchmark_ticker'], r['close'],
            r['sma50'], r['sma200'], r['dd_count'], r['ftd_detected'],
            r['regime_label'], r['choch_label']
        ))
    conn.commit()
    conn.close()

def get_latest_global_regime(market: str = None) -> list:
    import sqlite3
    conn = get_connection()
    cursor = conn.cursor()
    
    if market:
        cursor.execute('''
            SELECT * FROM global_regime_history 
            WHERE market = ? 
            ORDER BY date DESC LIMIT 1
        ''', (market,))
    else:
        cursor.execute("SELECT MAX(date) FROM global_regime_history")
        row = cursor.fetchone()
        latest = row[0] if row else None
        if not latest:
            conn.close()
            return []
        cursor.execute("SELECT * FROM global_regime_history WHERE date = ?", (latest,))
    
    results = _fetch_all_dicts(cursor)
    conn.close()
    return results

def save_global_etf_momentum(etf_data: list):
    import sqlite3
    conn = get_connection()
    cursor = conn.cursor()
    for e in etf_data:
        cursor.execute('''
            INSERT OR REPLACE INTO global_etf_momentum
            (date, ticker, name, return_1m, return_3m, return_6m)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            e['date'], e['ticker'], e.get('name', ''), e['return_1m'], e['return_3m'], e['return_6m']
        ))
    conn.commit()
    conn.close()

def get_latest_global_etf_momentum() -> list:
    import sqlite3
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM global_etf_momentum")
    row = cursor.fetchone()
    latest = row[0] if row else None
    if not latest:
        conn.close()
        return []
        
    cursor.execute("SELECT * FROM global_etf_momentum WHERE date = ?", (latest,))
    results = _fetch_all_dicts(cursor)
    conn.close()
    return results
