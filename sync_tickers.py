import urllib.request
import json
import csv
import io
import os
import sys
from datetime import datetime

# --- Constants ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
NSE_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
NASDAQ_API_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=15000&offset=0&download=true"

def sync_nse_tickers():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching NSE Equity Master List...")
    req = urllib.request.Request(NSE_URL, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        response = urllib.request.urlopen(req)
        csv_data = response.read().decode('utf-8')
        
        reader = csv.DictReader(io.StringIO(csv_data))
        valid_tickers = []
        for row in reader:
            # Only track main equities (EQ series)
            if row.get(' SERIES', '').strip() == 'EQ':
                symbol = row.get('SYMBOL', '').strip()
                if symbol:
                    valid_tickers.append(f"{symbol}.NS")
                    
        valid_tickers = sorted(list(set(valid_tickers)))
        if len(valid_tickers) > 1000:
            with open(os.path.join(ROOT_DIR, "tickers.txt"), "w") as f:
                f.write("\n".join(valid_tickers))
            print(f"SUCCESS: Saved {len(valid_tickers)} NSE tickers to tickers.txt")
        else:
            print(f"ERROR: Found suspiciously low number of NSE tickers ({len(valid_tickers)}). Aborting write.")
            
    except Exception as e:
        print(f"ERROR: Failed to fetch or process NSE tickers: {e}")

def sync_us_tickers():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching US Equity Master List from Nasdaq Screener API...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.nasdaq.com/'
    }
    req = urllib.request.Request(NASDAQ_API_URL, headers=headers)
    try:
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode('utf-8'))
        
        rows = data.get('data', {}).get('rows', [])
        valid_tickers = []
        
        for row in rows:
            symbol = row.get('symbol', '').strip()
            # Contains ^ or / usually indicates warrants or preferred stock
            if not symbol or '^' in symbol or '/' in symbol:
                continue
                
            # Parse Market Cap (e.g. "2,500,000,000")
            mcap_str = str(row.get('marketCap', '')).replace(',', '')
            try:
                mcap = float(mcap_str)
            except ValueError:
                mcap = 0.0
                
            # Filter for Market Cap >= $2 Billion
            if mcap >= 2_000_000_000:
                valid_tickers.append(symbol)
                
        valid_tickers = sorted(list(set(valid_tickers)))
        if len(valid_tickers) > 1000:
            with open(os.path.join(ROOT_DIR, "tickers_us.txt"), "w") as f:
                f.write("\n".join(valid_tickers))
            print(f"SUCCESS: Saved {len(valid_tickers)} US tickers (> $2B Market Cap) to tickers_us.txt")
        else:
            print(f"ERROR: Found suspiciously low number of US tickers ({len(valid_tickers)}). Aborting write.")
            
    except Exception as e:
        print(f"ERROR: Failed to fetch or process US tickers: {e}")

if __name__ == "__main__":
    print("--- STARTING TICKER SYNC ENGINE ---")
    sync_nse_tickers()
    sync_us_tickers()
    print("--- SYNC COMPLETE ---")
