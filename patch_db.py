import re

with open('database.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add schema to init_database
schema_to_insert = """
    cursor.execute('''
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

    cursor.execute('''
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
"""

if 'global_regime_history' not in content:
    content = content.replace('    print("Database initialized successfully")', schema_to_insert + '\n    print("Database initialized successfully")')

# 2. Add functions to end of file
functions_to_append = """
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
    conn.row_factory = sqlite3.Row
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
    
    results = [dict(r) for r in cursor.fetchall()]
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
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM global_etf_momentum")
    row = cursor.fetchone()
    latest = row[0] if row else None
    if not latest:
        conn.close()
        return []
        
    cursor.execute("SELECT * FROM global_etf_momentum WHERE date = ?", (latest,))
    results = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return results
"""

if 'save_global_regime' not in content:
    content += "\n" + functions_to_append

with open('database.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("database.py successfully patched!")
