"""
Catalyst & Corporate Filings Engine
----------------------------------
Fetches, cleans, and caches real-time catalysts for momentum breakout candidates:
1. Google News RSS headlines (last 24-48 hours) with source & relative time.
2. NSE and BSE Corporate Announcements / Filings with direct exchange PDF links.
3. Multi-threaded batch prefetching (ThreadPoolExecutor) with 2-hour SQLite & memory caching.
"""

import os
import re
import json
import html
import sqlite3
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import email.utils
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
from config import DATABASE_PATH

# -----------------------------------------------------------------------------
# CONSTANTS & NOISE FILTERS
# -----------------------------------------------------------------------------

NOISE_KEYWORDS = [
    'trading window',
    'closure of trading window',
    'loss of share certificate',
    'duplicate share certificate',
    'compliance certificate',
    'newspaper publication',
    'confirmation certificate',
    'reconciliation of share capital',
    'certificate under reg 74(5)',
    'allotment of shares under esop',
    'allotment of equity shares under esop',
    'esop allotment',
    'grant of stock options'
]

BSE_MAP_CACHE = None


def get_bse_mapping() -> dict:
    """Loads and caches BSE mapping table."""
    global BSE_MAP_CACHE
    if BSE_MAP_CACHE is not None:
        return BSE_MAP_CACHE
    
    mapping_path = os.path.join(PROJECT_ROOT, 'bse_mapping.json')
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                BSE_MAP_CACHE = json.load(f)
                return BSE_MAP_CACHE
        except Exception:
            pass
    BSE_MAP_CACHE = {}
    return BSE_MAP_CACHE


def clean_ticker(ticker: str) -> str:
    """Normalize ticker symbol removing exchange/series suffixes."""
    if not ticker:
        return ""
    t = str(ticker).strip().upper()
    t = t.replace('.NS', '').replace('.BO', '')
    t = t.replace('-SM', '').replace('-ST', '').replace('-BE', '')
    return t


def is_filing_noise(title: str, description: str) -> bool:
    """Identifies routine administrative filings that do not represent trading catalysts."""
    combined = (str(title) + " " + str(description)).lower()
    for kw in NOISE_KEYWORDS:
        if kw in combined:
            return True
    return False


def categorize_filing(title: str, description: str) -> tuple:
    """
    Categorizes announcement and returns (CategoryName, ColorHex, Icon).
    """
    combined = (str(title) + " " + str(description)).lower()
    if 'press release' in combined or 'media release' in combined:
        return ('Press Release', '#a855f7', '📢')
    elif 'financial result' in combined or 'quarterly' in combined or 'earnings' in combined:
        return ('Earnings', '#10b981', '💰')
    elif 'order' in combined or 'contract' in combined or 'award' in combined or 'bagged' in combined or 'commercial' in combined:
        return ('Order / Contract', '#06b6d4', '📑')
    elif 'usfda' in combined or 'fda' in combined or 'regulatory' in combined or 'patent' in combined or 'eir' in combined:
        return ('FDA / Regulatory', '#6366f1', '🔬')
    elif 'board meeting' in combined:
        return ('Board Meeting', '#f59e0b', '🏛️')
    elif 'acquisition' in combined or 'takeover' in combined or 'sast' in combined or 'merger' in combined or 'insider' in combined:
        return ('M&A / Insider', '#ec4899', '🤝')
    elif 'dividend' in combined or 'bonus' in combined or 'split' in combined:
        return ('Corporate Action', '#8b5cf6', '🎁')
    elif 'investor meet' in combined or 'concall' in combined or 'conference' in combined or 'analyst' in combined:
        return ('Investor Meet', '#38bdf8', '🎙️')
    else:
        return ('Corporate Filing', '#64748b', '📄')


def format_relative_time(dt: datetime) -> tuple:
    """
    Returns (human_time_ago, is_within_48h).
    """
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        # Assume UTC or local time
        dt = dt.replace(tzinfo=timezone.utc)
    
    diff = now - dt
    total_seconds = max(0, int(diff.total_seconds()))
    is_recent = total_seconds <= (48 * 3600)
    
    if total_seconds < 3600:
        mins = max(1, total_seconds // 60)
        return (f"{mins}m ago", is_recent)
    elif total_seconds < 86400:
        hours = total_seconds // 3600
        return (f"{hours}h ago", is_recent)
    elif total_seconds < 172800:
        return ("1d ago", is_recent)
    else:
        days = total_seconds // 86400
        if days <= 7:
            return (f"{days}d ago", False)
        else:
            return (dt.strftime("%d %b"), False)


# -----------------------------------------------------------------------------
# GOOGLE NEWS RSS FETCHER
# -----------------------------------------------------------------------------

def fetch_google_news(ticker: str, max_items: int = 2) -> list:
    """
    Fetches latest news from Google News RSS for the given ticker.
    Returns list of dicts: [{'title', 'source', 'published_at', 'time_ago', 'url', 'is_recent'}]
    """
    base_sym = clean_ticker(ticker)
    if not base_sym:
        return []
    
    # Primary search with when:7d, fallback to general share query
    queries = [
        f"{base_sym} when:7d",
        f"{base_sym} share OR {base_sym} stock"
    ]
    
    parsed_items = []
    seen_titles = set()
    
    for q in queries:
        encoded_q = urllib.parse.quote(q)
        url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-IN&gl=IN&ceid=IN:en"
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            items = root.findall('./channel/item')
            
            for it in items:
                title_elem = it.find('title')
                link_elem = it.find('link')
                pub_elem = it.find('pubDate')
                source_elem = it.find('source')
                
                raw_title = title_elem.text if title_elem is not None and title_elem.text else ''
                link = link_elem.text if link_elem is not None and link_elem.text else ''
                pub_str = pub_elem.text if pub_elem is not None and pub_elem.text else ''
                source = source_elem.text if source_elem is not None and source_elem.text else ''
                
                if not raw_title or not link or not pub_str:
                    continue
                
                # Clean title: Google News appends " - SourceName"
                clean_title = raw_title
                if source and clean_title.endswith(f" - {source}"):
                    clean_title = clean_title[:-len(f" - {source}")].strip()
                elif ' - ' in clean_title:
                    parts = clean_title.rsplit(' - ', 1)
                    clean_title = parts[0].strip()
                    if not source:
                        source = parts[1].strip()
                
                # Deduplicate by title normalization
                t_key = re.sub(r'[^a-zA-Z0-9]', '', clean_title.lower())[:40]
                if t_key in seen_titles:
                    continue
                seen_titles.add(t_key)
                
                try:
                    dt = email.utils.parsedate_to_datetime(pub_str)
                except Exception:
                    continue
                
                time_ago, is_recent = format_relative_time(dt)
                
                parsed_items.append({
                    'title': clean_title,
                    'source': source or 'News',
                    'published_at': pub_str,
                    'dt': dt,
                    'time_ago': time_ago,
                    'url': link,
                    'is_recent': is_recent
                })
                
            if len(parsed_items) >= max_items:
                break
                
        except Exception:
            continue
            
    # Sort parsed items by datetime descending
    parsed_items.sort(key=lambda x: x['dt'], reverse=True)
    
    # Strip dt object before returning to ensure clean JSON serialization
    results = []
    for item in parsed_items[:max_items]:
        results.append({
            'title': item['title'],
            'source': item['source'],
            'published_at': item['published_at'],
            'time_ago': item['time_ago'],
            'url': item['url'],
            'is_recent': item['is_recent']
        })
        
    return results


# -----------------------------------------------------------------------------
# NSE & BSE CORPORATE FILINGS FETCHER
# -----------------------------------------------------------------------------

def fetch_corporate_filings(ticker: str, session: requests.Session = None, max_items: int = 2) -> list:
    """
    Fetches latest corporate filings from NSE (primary) and BSE (fallback/supplement).
    Returns list of dicts: [{'title', 'category', 'category_color', 'icon', 'source', 'date_time', 'time_ago', 'pdf_url'}]
    """
    base_sym = clean_ticker(ticker)
    if not base_sym:
        return []
        
    bse_map = get_bse_mapping()
    scrip_cd = bse_map.get(base_sym)
    
    if session is None:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*'
        })
        try:
            session.get('https://www.nseindia.com', timeout=5)
        except Exception:
            pass
            
    announcements = []
    seen_texts = set()
    
    # 1. FETCH FROM NSE
    try:
        url_nse = f'https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={base_sym}'
        r = session.get(url_nse, timeout=6)
        if r.status_code == 200:
            nse_items = r.json()
            if isinstance(nse_items, list):
                for item in nse_items:
                    desc = item.get('desc', '') or ''
                    text = item.get('attchmntText', '') or ''
                    sort_date = item.get('sort_date', '') or ''
                    pdf_link = item.get('attchmntFile', '') or ''
                    
                    if is_filing_noise(desc, text):
                        continue
                    
                    # Deduplication key
                    k = re.sub(r'[^a-zA-Z0-9]', '', (desc + text).lower())[:40]
                    if k in seen_texts:
                        continue
                    seen_texts.add(k)
                    
                    # Parse timestamp (e.g. 2026-09-10 17:00:15)
                    try:
                        dt = datetime.strptime(sort_date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                    except Exception:
                        dt = datetime.now(timezone.utc)
                    
                    time_ago, is_recent = format_relative_time(dt)
                    cat_name, cat_color, cat_icon = categorize_filing(desc, text)
                    
                    # Display headline
                    display_title = text.strip() if (text and len(text) > 15) else desc.strip()
                    # Clean redundant ticker prefix if present
                    if display_title.startswith(f"{base_sym}:"):
                        display_title = display_title[len(f"{base_sym}:"):].strip()
                    
                    if pdf_link and not pdf_link.startswith('http'):
                        pdf_link = f"https://nsearchives.nseindia.com/corporate/{pdf_link}"
                        
                    announcements.append({
                        'title': display_title,
                        'desc': desc,
                        'category': cat_name,
                        'category_color': cat_color,
                        'icon': cat_icon,
                        'source': 'NSE',
                        'date_time': sort_date,
                        'dt': dt,
                        'time_ago': time_ago,
                        'pdf_url': pdf_link,
                        'is_recent': is_recent
                    })
    except Exception:
        pass
        
    # 2. FETCH FROM BSE IF WE NEED MORE ANNOUNCEMENTS
    if scrip_cd and len(announcements) < max_items:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            
            url_bse = f'https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?pageno=1&strCat=-1&strPrevDate={start_str}&strScrip={scrip_cd}&strSearch=P&strToDate={end_str}&strType=C'
            headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.bseindia.com/'}
            r_bse = requests.get(url_bse, headers=headers, timeout=6)
            if r_bse.status_code == 200:
                bse_items = r_bse.json().get('Table', [])
                for b in bse_items:
                    headline = b.get('HEADLINE', '') or ''
                    sub_cat = b.get('NEWSSUB', '') or ''
                    news_dt = b.get('NEWS_DT', '') or ''
                    att_file = b.get('ATTACHMENTNAME', '') or ''
                    
                    if is_filing_noise(sub_cat, headline):
                        continue
                        
                    k = re.sub(r'[^a-zA-Z0-9]', '', (sub_cat + headline).lower())[:40]
                    if k in seen_texts:
                        continue
                    seen_texts.add(k)
                    
                    try:
                        dt = datetime.strptime(news_dt.split('.')[0], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                    except Exception:
                        dt = datetime.now(timezone.utc)
                        
                    time_ago, is_recent = format_relative_time(dt)
                    cat_name, cat_color, cat_icon = categorize_filing(sub_cat, headline)
                    
                    display_title = headline.strip() if headline else sub_cat.strip()
                    pdf_url = f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{att_file}" if att_file else ""
                    
                    announcements.append({
                        'title': display_title,
                        'desc': sub_cat,
                        'category': cat_name,
                        'category_color': cat_color,
                        'icon': cat_icon,
                        'source': 'BSE',
                        'date_time': news_dt,
                        'dt': dt,
                        'time_ago': time_ago,
                        'pdf_url': pdf_url,
                        'is_recent': is_recent
                    })
        except Exception:
            pass
            
    # Sort announcements by date descending
    announcements.sort(key=lambda x: x['dt'], reverse=True)
    
    results = []
    for a in announcements[:max_items]:
        results.append({
            'title': a['title'],
            'category': a['category'],
            'category_color': a['category_color'],
            'icon': a['icon'],
            'source': a['source'],
            'date_time': a['date_time'],
            'time_ago': a['time_ago'],
            'pdf_url': a['pdf_url'],
            'is_recent': a['is_recent']
        })
        
    return results


# -----------------------------------------------------------------------------
# DATABASE CACHING (SQLITE)
# -----------------------------------------------------------------------------

def init_catalyst_cache_table():
    """Ensure catalyst_feed_cache table exists in the database."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS catalyst_feed_cache (
                ticker TEXT PRIMARY KEY,
                news_json TEXT,
                filings_json TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_cached_catalysts_db(tickers: list, max_age_hours: int = 2) -> dict:
    """
    Retrieves catalysts from SQLite if updated within max_age_hours.
    """
    cached = {}
    if not tickers:
        return cached
        
    try:
        init_catalyst_cache_table()
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        
        placeholders = ','.join('?' for _ in tickers)
        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).strftime('%Y-%m-%d %H:%M:%S')
        
        query = f"""
            SELECT ticker, news_json, filings_json, updated_at 
            FROM catalyst_feed_cache 
            WHERE ticker IN ({placeholders}) AND updated_at >= ?
        """
        c.execute(query, tickers + [cutoff])
        rows = c.fetchall()
        for r in rows:
            t = r[0]
            try:
                n = json.loads(r[1]) if r[1] else []
                f = json.loads(r[2]) if r[2] else []
                cached[t] = {'news': n, 'filings': f, 'cached': True}
            except Exception:
                continue
        conn.close()
    except Exception:
        pass
        
    return cached


def save_catalysts_db(ticker_catalysts: dict):
    """
    Saves batch of ticker catalysts into SQLite cache.
    """
    if not ticker_catalysts:
        return
        
    try:
        init_catalyst_cache_table()
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        records = []
        for t, data in ticker_catalysts.items():
            news_json = json.dumps(data.get('news', []))
            filings_json = json.dumps(data.get('filings', []))
            records.append((t, news_json, filings_json, now_str))
            
        c.executemany("""
            INSERT OR REPLACE INTO catalyst_feed_cache (ticker, news_json, filings_json, updated_at)
            VALUES (?, ?, ?, ?)
        """, records)
        conn.commit()
        conn.close()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# CONCURRENT BATCH PREFETCHER
# -----------------------------------------------------------------------------

def fetch_single_ticker(ticker: str, session: requests.Session) -> tuple:
    """Worker task to fetch news and filings for one ticker."""
    clean_t = clean_ticker(ticker)
    news = fetch_google_news(clean_t, max_items=2)
    filings = fetch_corporate_filings(clean_t, session=session, max_items=2)
    return clean_t, {'news': news, 'filings': filings, 'cached': False}


def batch_fetch_catalysts(tickers: list, max_workers: int = 8) -> dict:
    """
    High-performance multi-threaded catalyst prefetcher.
    1. Checks local SQLite cache for all tickers (returns instantly if fresh).
    2. Runs ThreadPoolExecutor only for missing/stale tickers.
    3. Saves fresh results to SQLite cache.
    """
    if not tickers:
        return {}
        
    clean_tickers = list(dict.fromkeys([clean_ticker(t) for t in tickers if t]))
    results = get_cached_catalysts_db(clean_tickers, max_age_hours=2)
    
    missing_tickers = [t for t in clean_tickers if t not in results]
    if not missing_tickers:
        return results
        
    # Initialize a single session with NSE cookies for parallel workers
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*'
    })
    try:
        session.get('https://www.nseindia.com', timeout=4)
    except Exception:
        pass
        
    fresh_data = {}
    with ThreadPoolExecutor(max_workers=min(len(missing_tickers), max_workers)) as executor:
        future_map = {
            executor.submit(fetch_single_ticker, ticker, session): ticker 
            for ticker in missing_tickers
        }
        for future in as_completed(future_map):
            try:
                t, data = future.result()
                results[t] = data
                fresh_data[t] = data
            except Exception:
                orig_t = future_map[future]
                results[orig_t] = {'news': [], 'filings': [], 'cached': False}
                
    # Save newly fetched data to database
    if fresh_data:
        save_catalysts_db(fresh_data)
        
    return results


# -----------------------------------------------------------------------------
# STREAMLIT CACHED INTERFACE
# -----------------------------------------------------------------------------

def get_catalysts_for_buy_triggers(tickers: list) -> dict:
    """
    Streamlit-friendly wrapper to fetch catalysts for Buy Triggers.
    Cached in Streamlit if st is present, else calls batch_fetch_catalysts directly.
    """
    try:
        import streamlit as st
        @st.cache_data(ttl=1800, show_spinner=False)
        def _cached_batch(ticker_tuple: tuple) -> dict:
            return batch_fetch_catalysts(list(ticker_tuple))
            
        return _cached_batch(tuple(tickers))
    except Exception:
        return batch_fetch_catalysts(tickers)
