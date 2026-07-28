# Screener Analytics — Database Module
# SQLite persistence for historical tracking and industry analytics

import sqlite3
from datetime import datetime
from config import DATABASE_PATH


def get_connection():
    """Get database connection to the main kush_tracker.db."""
    return sqlite3.connect(DATABASE_PATH)


def init_screener_db():
    """Create screener-specific tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Upload history — track total stock counts over time
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS screener_upload_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            cap_category TEXT NOT NULL,
            total_stocks INTEGER,
            stocks_above_200sma INTEGER DEFAULT 0,
            stocks_above_50sma INTEGER DEFAULT 0,
            stocks_above_21ema INTEGER DEFAULT 0,
            avg_rs_score REAL DEFAULT 0,
            ready_count INTEGER DEFAULT 0,
            developing_count INTEGER DEFAULT 0,
            not_ready_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, cap_category)
        )
    ''')

    # Industry snapshot — per-industry group counts per upload
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS screener_industry_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            cap_category TEXT NOT NULL,
            industry_group TEXT NOT NULL,
            stock_count INTEGER DEFAULT 0,
            avg_momentum REAL DEFAULT 0,
            avg_rs_score REAL DEFAULT 0,
            avg_qoq_sales REAL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, cap_category, industry_group)
        )
    ''')

    # Indexes for fast queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_screener_upload_date
        ON screener_upload_history(date)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_screener_industry_date
        ON screener_industry_snapshot(date)
    ''')

    conn.commit()
    conn.close()
    print("[OK] Screener database tables initialized")


def save_upload_snapshot(date: str, cap_category: str, stats: dict):
    """
    Save upload-level statistics.

    Args:
        date: Date string (YYYY-MM-DD)
        cap_category: 'Large Cap' or 'Small Cap'
        stats: Dict with total_stocks, stocks_above_200sma, stocks_above_50sma,
               stocks_above_21ema, avg_rs_score, ready_count, developing_count, not_ready_count
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT OR REPLACE INTO screener_upload_history
        (date, cap_category, total_stocks, stocks_above_200sma, stocks_above_50sma,
         stocks_above_21ema, avg_rs_score, ready_count, developing_count, not_ready_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        date,
        cap_category,
        stats.get('total_stocks', 0),
        stats.get('stocks_above_200sma', 0),
        stats.get('stocks_above_50sma', 0),
        stats.get('stocks_above_21ema', 0),
        stats.get('avg_rs_score', 0),
        stats.get('ready_count', 0),
        stats.get('developing_count', 0),
        stats.get('not_ready_count', 0),
    ))

    conn.commit()
    conn.close()
    print(f"[OK] Saved screener upload snapshot: {cap_category} on {date}")


def save_industry_snapshot(date: str, cap_category: str, industry_data: list):
    """
    Save per-industry-group snapshot data.

    Args:
        date: Date string (YYYY-MM-DD)
        cap_category: 'Large Cap' or 'Small Cap'
        industry_data: List of dicts with industry_group, stock_count, avg_momentum, avg_rs, avg_qoq_sales
    """
    conn = get_connection()
    cursor = conn.cursor()

    for row in industry_data:
        cursor.execute('''
            INSERT OR REPLACE INTO screener_industry_snapshot
            (date, cap_category, industry_group, stock_count, avg_momentum, avg_rs_score, avg_qoq_sales)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            date,
            cap_category,
            row.get('industry_group', ''),
            row.get('count', 0),
            row.get('avg_momentum', 0),
            row.get('avg_rs', 0),
            row.get('avg_qoq_sales', 0),
        ))

    conn.commit()
    conn.close()
    print(f"[OK] Saved {len(industry_data)} industry snapshots: {cap_category} on {date}")


def get_upload_history(cap_category: str = None, days: int = 180) -> list:
    """
    Retrieve historical upload data.

    Args:
        cap_category: Optional filter ('Large Cap' or 'Small Cap')
        days: Number of days to look back

    Returns:
        List of dicts with upload history records
    """
    conn = get_connection()
    cursor = conn.cursor()

    if cap_category:
        cursor.execute('''
            SELECT date, cap_category, total_stocks, stocks_above_200sma,
                   stocks_above_50sma, stocks_above_21ema, avg_rs_score,
                   ready_count, developing_count, not_ready_count
            FROM screener_upload_history
            WHERE cap_category = ?
            ORDER BY date DESC
            LIMIT ?
        ''', (cap_category, days))
    else:
        cursor.execute('''
            SELECT date, cap_category, total_stocks, stocks_above_200sma,
                   stocks_above_50sma, stocks_above_21ema, avg_rs_score,
                   ready_count, developing_count, not_ready_count
            FROM screener_upload_history
            ORDER BY date DESC
            LIMIT ?
        ''', (days * 2,))

    columns = ['date', 'cap_category', 'total_stocks', 'stocks_above_200sma',
               'stocks_above_50sma', 'stocks_above_21ema', 'avg_rs_score',
               'ready_count', 'developing_count', 'not_ready_count']

    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return results


def get_industry_history(cap_category: str = None, days: int = 90) -> list:
    """
    Retrieve historical industry snapshot data.

    Args:
        cap_category: Optional filter
        days: Number of days to look back

    Returns:
        List of dicts with industry history records
    """
    conn = get_connection()
    cursor = conn.cursor()

    if cap_category:
        cursor.execute('''
            SELECT date, cap_category, industry_group, stock_count,
                   avg_momentum, avg_rs_score, avg_qoq_sales
            FROM screener_industry_snapshot
            WHERE cap_category = ?
            ORDER BY date DESC
            LIMIT ?
        ''', (cap_category, days * 50))
    else:
        cursor.execute('''
            SELECT date, cap_category, industry_group, stock_count,
                   avg_momentum, avg_rs_score, avg_qoq_sales
            FROM screener_industry_snapshot
            ORDER BY date DESC
            LIMIT ?
        ''', (days * 100,))

    columns = ['date', 'cap_category', 'industry_group', 'stock_count',
               'avg_momentum', 'avg_rs_score', 'avg_qoq_sales']

    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return results


if __name__ == '__main__':
    init_screener_db()
    print("Tables created. Run via pages/screener_analytics.py")
