import libsql_experimental as libsql

try:
    conn = libsql.connect("file:.cache/kush_tracker_lite.db")
    cursor = conn.cursor()
    
    # Try the exact query that failed
    market = "INDIA"
    ticker = "TCS"
    signal_name = "Test Signal"
    today = "2026-07-29"
    
    print("Executing query...")
    cursor.execute('''
        INSERT OR IGNORE INTO intraday_signals_history
        (market, ticker, signal_name, date)
        VALUES (?, ?, ?, ?)
    ''', (market, ticker, signal_name, today))
    
    print("Success!")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
