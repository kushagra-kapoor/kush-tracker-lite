import requests
import urllib.parse
import xml.etree.ElementTree as ET
import concurrent.futures
from datetime import datetime
import email.utils
import time
import streamlit as st

def _parse_pubdate(pubdate_str):
    """Convert RFC-2822 date string from RSS into a relative human-readable string."""
    try:
        dt_tuple = email.utils.parsedate_tz(pubdate_str)
        if dt_tuple:
            timestamp = email.utils.mktime_tz(dt_tuple)
            dt = datetime.fromtimestamp(timestamp)
            now = datetime.now()
            diff = now - dt
            
            hours = int(diff.total_seconds() / 3600)
            if hours < 1:
                mins = int(diff.total_seconds() / 60)
                return f"{mins} mins ago" if mins > 0 else "Just now"
            elif hours < 24:
                return f"{hours} hours ago"
            else:
                days = int(hours / 24)
                return f"{days} days ago"
    except Exception:
        pass
    return "Recent"

def _fetch_ticker_news(ticker, max_results=2):
    """
    Fetch news for a single ticker via Google News RSS.
    Target strictly to last 48 hours.
    """
    # Clean ticker (remove .NS, .BO, etc., and NSE:, BSE:)
    clean_ticker = str(ticker).replace('NSE:', '').replace('BSE:', '').split('.')[0].strip()
    
    if not clean_ticker:
        return []
        
    query = f"{clean_ticker} stock India when:2d"
    q_encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={q_encoded}&hl=en-IN&gl=IN&ceid=IN:en"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    results = []
    try:
        response = requests.get(url, headers=headers, timeout=3.5)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        items = root.findall('.//item')
        
        for item in items:
            title_elem = item.find('title')
            link_elem = item.find('link')
            pubdate_elem = item.find('pubDate')
            source_elem = item.find('source')
            
            if title_elem is not None and link_elem is not None:
                title_text = title_elem.text or ""
                # Strip publisher from the end of title if it exists (Google News appends it)
                if " - " in title_text:
                    title_text = " - ".join(title_text.split(" - ")[:-1])
                    
                publisher = source_elem.text if source_elem is not None else "News"
                time_ago = _parse_pubdate(pubdate_elem.text) if pubdate_elem is not None else "Recent"
                
                results.append({
                    'ticker': ticker,
                    'clean_ticker': clean_ticker,
                    'title': title_text.strip(),
                    'link': link_elem.text,
                    'publisher': publisher,
                    'time_ago': time_ago,
                    'raw_title': title_elem.text
                })
                
            if len(results) >= max_results:
                break
                
    except Exception as e:
        # Silently fail for individual tickers to ensure app resilience
        pass
        
    return results

import time
_TICKER_NEWS_CACHE = {}

def fetch_portfolio_news(tickers, max_per_ticker=2):
    """
    Fetch news concurrently for a list of tickers.
    Uses an internal dictionary cache per-ticker to prevent Streamlit list-hashing issues.
    """
    if not tickers:
        return []
        
    all_news = []
    tickers_to_fetch = []
    now = time.time()
    
    # 1. Pull from cache where available
    for t in tickers:
        if t in _TICKER_NEWS_CACHE and (now - _TICKER_NEWS_CACHE[t]['time']) < 1800:
            if _TICKER_NEWS_CACHE[t]['data']:
                all_news.extend(_TICKER_NEWS_CACHE[t]['data'])
        else:
            tickers_to_fetch.append(t)
            
    # 2. Fetch missing tickers concurrently
    if tickers_to_fetch:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(_fetch_ticker_news, t, max_per_ticker): t for t in tickers_to_fetch}
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                t = future_to_ticker[future]
                try:
                    news_items = future.result()
                    if news_items:
                        _TICKER_NEWS_CACHE[t] = {'time': now, 'data': news_items}
                        all_news.extend(news_items)
                    else:
                        # Cache empty results for only 3 minutes to self-heal from network drops
                        _TICKER_NEWS_CACHE[t] = {'time': now - 1620, 'data': []}
                except Exception:
                    # Cache failure briefly
                    _TICKER_NEWS_CACHE[t] = {'time': now - 1620, 'data': []}
                    pass
                    
    # 3. Deduplicate
    deduped_news = []
    seen_links = set()
    seen_titles = set()
    
    for item in all_news:
        if item['link'] not in seen_links and item['title'] not in seen_titles:
            seen_links.add(item['link'])
            seen_titles.add(item['title'])
            deduped_news.append(item)
            
    return deduped_news
