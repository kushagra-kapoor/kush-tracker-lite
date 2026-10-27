import os
import sqlite3
import requests
from datetime import datetime, timedelta
import pandas as pd
import database

NOISE_KEYWORDS = [
    'trading window',
    'closure of trading window',
    'loss of share certificate',
    'duplicate share certificate',
    'compliance certificate',
    'newspaper publication'
]

def is_noise(title, description):
    combined = (str(title) + " " + str(description)).lower()
    for kw in NOISE_KEYWORDS:
        if kw in combined:
            return True
    return False

def get_active_portfolio_tickers():
    import portfolio_fetcher
    df = portfolio_fetcher.fetch_portfolio_data()
    return df['ticker'].tolist() if not df.empty else []

def clean_ticker(ticker):
    # Remove exchange suffixes
    t = ticker.replace('.NS', '').replace('.BO', '')
    # Remove SME suffixes
    t = t.replace('-SM', '').replace('-ST', '').replace('-BE', '')
    return t

def fetch_announcements_for_ticker(session, base_ticker):
    # Try equities index first
    url_eq = f'https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={base_ticker}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        res = session.get(url_eq, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if len(data) > 0:
                return data
            
        # If equities fails or returns 0, try sme
        url_sme = f'https://www.nseindia.com/api/corporate-announcements?index=sme&symbol={base_ticker}'
        res_sme = session.get(url_sme, headers=headers, timeout=10)
        if res_sme.status_code == 200:
            data_sme = res_sme.json()
            return data_sme
            
    except Exception as e:
        print(f"Error fetching for {base_ticker}: {e}")
        
    return []

def fetch_pit(session, base_ticker):
    url = f'https://www.nseindia.com/api/corporates-pit?index=equities&symbol={base_ticker}'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        res = session.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, dict):
                return data.get('data', [])
            elif isinstance(data, list):
                return data
    except Exception as e:
        print(f"Error fetching PIT for {base_ticker}: {e}")
    return []

def fetch_financials(session, base_ticker):
    url = f'https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol={base_ticker}'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        res = session.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get('data', [])
    except Exception as e:
        print(f"Error fetching Financials for {base_ticker}: {e}")
    return []

def fetch_bse_announcements(scrip_cd):
    if not scrip_cd:
        return []
    
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    url = f'https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?pageno=1&strCat=-1&strPrevDate={start_str}&strScrip={scrip_cd}&strSearch=P&strToDate={end_str}&strType=C'
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': 'https://www.bseindia.com/'
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return data.get('Table', [])
    except Exception as e:
        print(f"Error fetching BSE for {scrip_cd}: {e}")
    return []

def sync_announcements():
    database.init_database()
    tickers = get_active_portfolio_tickers()
    if not tickers:
        print("No active portfolio tickers found.")
        return
        
    print(f"Found {len(tickers)} tickers in portfolio. Syncing announcements...")
    
    # Establish session to bypass bot protection
    session = requests.Session()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        session.get('https://www.nseindia.com', headers=headers, timeout=10)
    except:
        print("Warning: Could not hit NSE homepage to establish cookie.")
        
    seven_days_ago_date = datetime.now() - timedelta(days=7)
    
    import json
    try:
        with open('bse_mapping.json', 'r') as f:
            bse_map = json.load(f)
    except:
        bse_map = {}
        
    all_records = []
    for ticker in tickers:
        base_ticker = clean_ticker(ticker)
        
        is_bse_only = base_ticker.isdigit()
        
        data = []
        pit_data = []
        fin_data = []
        
        if is_bse_only:
            scrip_cd = base_ticker
        else:
            scrip_cd = bse_map.get(base_ticker)
            data = fetch_announcements_for_ticker(session, base_ticker)
            pit_data = fetch_pit(session, base_ticker)
            fin_data = fetch_financials(session, base_ticker)
            
        bse_data = fetch_bse_announcements(scrip_cd)
        
        count = 0
        for item in data:
            # Parse sort_date, e.g. "2026-06-25 12:27:50"
            sort_date_str = item.get('sort_date')
            if not sort_date_str:
                continue
                
            try:
                dt_obj = datetime.strptime(sort_date_str, '%Y-%m-%d %H:%M:%S')
                if dt_obj >= seven_days_ago_date:
                    seq_id = str(item.get('seq_id'))
                    desc = item.get('desc', '')
                    attchmntText = item.get('attchmntText', '')
                    pdf_link = item.get('attchmntFile', '')
                    
                    if not is_noise(desc, attchmntText):
                        all_records.append((seq_id, ticker, sort_date_str, desc, attchmntText, pdf_link))
                        count += 1
            except ValueError:
                continue

        for item in pit_data:
            date_raw = item.get('date', '') # e.g. "25-Mar-2026 17:05"
            try:
                dt_obj = datetime.strptime(date_raw, "%d-%b-%Y %H:%M")
                sort_date_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
            except: continue
            
            if dt_obj >= seven_days_ago_date:
                seq_id = str(item.get('did', '')) + "_pit"
                trans_type = item.get('tdpTransactionType', 'Unknown')
                is_buy = 'Buy' in trans_type
                title_prefix = "[Insider Buy]" if is_buy else "[Insider Sell]"
                qty = item.get('secAcq', '0')
                mode = item.get('acqMode', 'Market')
                desc = f"{title_prefix} {trans_type} of {qty} shares"
                
                name = item.get('acqName', 'Unknown')
                val = item.get('secVal', '0')
                attchmntText = f"{name} executed a {trans_type} transaction of {qty} shares (Value: ₹{val}) via {mode}."
                pdf_link = item.get('xbrl', '')
                
                if not is_noise(desc, attchmntText):
                    all_records.append((seq_id, ticker, sort_date_str, desc, attchmntText, pdf_link))
                    count += 1

        for item in fin_data:
            date_raw = item.get('broadCastDate', '') # e.g. "30-Oct-2010 12:53:08"
            try:
                dt_obj = datetime.strptime(date_raw, "%d-%b-%Y %H:%M:%S")
                sort_date_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
            except: continue
            
            if dt_obj >= seven_days_ago_date:
                seq_id = str(item.get('seqNumber', '')) + "_fin"
                period = item.get('period', '')
                desc = f"[Financial Results] {period} Earnings Report"
                attchmntText = f"Financial results published for period ending {item.get('toDate', '')}."
                pdf_link = item.get('resultDetailedDataLink', '')
                if pdf_link and not pdf_link.startswith('http'):
                    pdf_link = 'https://www.nseindia.com' + pdf_link
                    
                if not is_noise(desc, attchmntText):
                    all_records.append((seq_id, ticker, sort_date_str, desc, attchmntText, pdf_link))
                    count += 1
                
        for item in bse_data:
            date_raw = item.get('NEWS_DT', '') # e.g. "2026-06-27T17:22:16.603"
            try:
                dt_obj = datetime.strptime(date_raw.split('.')[0], "%Y-%m-%dT%H:%M:%S")
                sort_date_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
            except: continue
            
            if dt_obj >= seven_days_ago_date:
                seq_id = str(item.get('NEWSID', '')) + "_bse"
                headline = item.get('HEADLINE', '')
                sub_cat = item.get('NEWSSUB', '')
                
                title = sub_cat if sub_cat else "Update"
                if sub_cat and ('Takeover' in sub_cat or 'SAST' in sub_cat):
                    title = f"[SAST / Takeover] {sub_cat}"
                    
                desc = headline
                
                file_name = item.get('ATTACHMENTNAME', '')
                if file_name and file_name.strip():
                    pdf_link = f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{file_name}"
                else:
                    pdf_link = ""
                    
                if not is_noise(title, desc):
                    all_records.append((seq_id, ticker, sort_date_str, title, desc, pdf_link))
                    count += 1
                
        print(f"[{ticker}] -> Found {count} announcements in last 7 days.")
        
    if all_records:
        database.save_corporate_announcements(all_records)
        print(f"Successfully saved {len(all_records)} announcements to database.")
    else:
        print("No new announcements in the last 7 days.")
        
    # Auto-cleanup old data to save disk space
    deleted_count = database.cleanup_old_announcements(days=7)
    print(f"Purged {deleted_count} old announcements from the database.")

if __name__ == '__main__':
    sync_announcements()
