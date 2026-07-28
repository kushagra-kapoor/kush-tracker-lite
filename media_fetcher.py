import os
import json
import time
import re
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Try to import googleapiclient, gracefully handle if not available
try:
    from googleapiclient.discovery import build
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

load_dotenv()

CACHE_FILE = ".cache/youtube_cache.json"
CACHE_EXPIRY_HOURS = 6

# Whitelist Channels (Financial News)
# Key: Channel Name, Value: Channel ID
CHANNELS_IN = {
    "ET Now": "UCI_mwTKUhicNzFrhm33MzBQ",
    "CNBC-TV18": "UCmRbHAgG2k2vDUvb3xsEunQ",
    "NDTV Profit": "UC3uJIdRFTGgLWrUziaHbzrg",
    "Zee Business": "UCkXopQ3ubd-rnXnStZqCl2w",
    "Moneycontrol": "UChftTVI0QJmyXkajQYt2tiQ"
}

CHANNELS_US = {
    "Investor's Business Daily": "UC5fZv7bPcF5j2RsfO-9OiLA",
    "CNBC Television": "UCrp_UI8XtuYfpiqluWLD7Lw",
    "Bloomberg Television": "UCIALMKvObZNtJ6AmdCLP7Lg",
    "TraderLion": "UCMgOSW62URSxzPCcNXyohLw"
}

def get_youtube_client():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key or not GOOGLE_API_AVAILABLE:
        return None
    return build('youtube', 'v3', developerKey=api_key)

def fetch_recent_channel_videos(youtube, channel_id):
    """
    Fetches the 50 most recent videos from a channel using the search endpoint.
    Costs 100 quota units per call, well within the 10,000 daily limit with caching.
    """
    videos = []
    
    # Calculate timestamp for 30 days ago
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
    
    # The uploads playlist ID is typically the channel ID with 'UU' instead of 'UC'
    if channel_id.startswith('UC'):
        playlist_id = 'UU' + channel_id[2:]
    else:
        playlist_id = channel_id
        
    try:
        next_page = None
        for _ in range(20): # Fetch up to 1000 videos per channel
            request = youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page
            )
            response = request.execute()
            
            items = response.get("items", [])
            cutoff_reached = False
            
            for item in items:
                snippet = item.get("snippet", {})
                published_at_str = snippet.get("publishedAt")
                if not published_at_str:
                    continue
                    
                pub_dt = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
                if pub_dt < cutoff_date:
                    cutoff_reached = True
                    break
                    
                videos.append({
                    "video_id": snippet.get("resourceId", {}).get("videoId"),
                    "title": snippet.get("title", "").replace("&quot;", '"').replace("&#39;", "'"),
                    "channel": snippet.get("channelTitle", ""),
                    "published_at": published_at_str,
                    "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", "")
                })
                
            if cutoff_reached:
                break
                
            next_page = response.get('nextPageToken')
            if not next_page:
                break
            
    except Exception as e:
        print(f"[MediaFetcher] Error fetching channel {channel_id}: {e}")
        
    return videos

def update_cache(region="IN"):
    """Fetches all recent videos from all whitelist channels and caches them."""
    youtube = get_youtube_client()
    if not youtube:
        print("[MediaFetcher] YouTube API key not found or google-api-client missing.")
        return []
        
    all_videos = []
    channels = CHANNELS_US if region == "US" else CHANNELS_IN
    cache_file = f".cache/youtube_cache_{region}.json"
    
    print(f"[MediaFetcher] Updating YouTube cache from {region} whitelist channels...")
    for channel_name, channel_id in channels.items():
        vids = fetch_recent_channel_videos(youtube, channel_id)
        all_videos.extend(vids)
        
    # Sort by newest first
    all_videos.sort(key=lambda x: x['published_at'], reverse=True)
    
    os.makedirs(".cache", exist_ok=True)
    cache_data = {
        "timestamp": time.time(),
        "videos": all_videos
    }
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
    return all_videos

def get_all_recent_videos(region="IN"):
    """Gets recent videos from cache, or fetches if cache is expired/missing."""
    cache_file = f".cache/youtube_cache_{region}.json"
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            cache_time = cache_data.get("timestamp", 0)
            if (time.time() - cache_time) < (CACHE_EXPIRY_HOURS * 3600):
                return cache_data.get("videos", [])
        except Exception as e:
            print(f"[MediaFetcher] Cache read error: {e}")
            
    # Cache expired or missing, fetch new
    return update_cache(region)

def extract_portfolio_interviews(portfolio_tickers, company_names=None, region="IN"):
    """
    Returns a list of videos matching the given portfolio tickers and names.
    Uses smart fuzzy matching to catch company name variations.
    """
    if company_names is None:
        company_names = []
        
    all_vids = get_all_recent_videos(region)
    matched = []
    seen = set()
    
    # Pre-process search terms
    search_terms = set()
    
    # Custom Aliases for companies whose tickers don't match their spoken names
    ALIASES = {
        'PNGSREVA': 'GADGIL',
        '544563': 'GADGIL',
        'SUDEEPPHRM': 'SUDEEP',
        'PRIZOR-ST': 'PRIZOR',
        'ANONDITA-SM': 'ANONDITA'
    }
    
    # Ignore commodity/index ETFs to prevent false positives (like 'Gold' matching every gold news)
    IGNORE_TICKERS = {'GOLDBEES', 'SILVERBEES', 'LIQUIDBEES', 'LIQUIDCASE', 'MOM30IETF', 'MOMENTUM50', 'ALPHA', 'GVT&D'}
    
    # Build reverse BSE map if possible
    bse_to_nse = {}
    try:
        if os.path.exists('bse_mapping.json'):
            import json
            with open('bse_mapping.json', 'r') as f:
                nse_to_bse = json.load(f)
                bse_to_nse = {str(v): k for k, v in nse_to_bse.items()}
    except Exception:
        pass

    for t in portfolio_tickers:
        clean_t = t.split(':')[-1] if ':' in t else t
        clean_t = clean_t.replace('.NS', '').replace('.BO', '')
        
        if clean_t.upper() in IGNORE_TICKERS:
            continue
            
        search_terms.add(clean_t.upper())
        # If it's a BSE code, also add the NSE ticker to match against names
        if clean_t.isdigit() and clean_t in bse_to_nse:
            search_terms.add(bse_to_nse[clean_t].upper())
            
        # Add Custom Aliases
        if clean_t.upper() in ALIASES:
            search_terms.add(ALIASES[clean_t.upper()])
            
    for c in company_names:
        if c:
            clean_c = c.split(':')[-1] if ':' in c else c
            clean_c = clean_c.replace('.NS', '').replace('.BO', '')
            search_terms.add(clean_c.upper())
    
    for v in all_vids:
        title_upper = v['title'].upper()
        # Create a compressed version of the title for fuzzy matching (removes spaces and punctuation)
        title_compressed = title_upper.replace(' ', '').replace('-', '').replace('&', '').replace("'", "")
        
        is_match = False
        for term in search_terms:
            # Clean up suffixes from tickers (e.g. GOLDBEES -> GOLD, ANONDITA-SM -> ANONDITA)
            clean_term = re.sub(r'(-SM|-ST|BEES|IETF|-P1)$', '', term).upper()
            
            # 1. Standard Word Boundary Match (for exact tickers/names)
            if re.search(r'\b' + re.escape(term) + r'\b', title_upper):
                is_match = True
                break
                
            # 2. Fuzzy Substring Match (only for substantial terms >= 6 chars to avoid false positives like GLAND in ENGLAND)
            if len(clean_term) >= 6 and clean_term in title_compressed:
                is_match = True
                break
                
        if is_match and v['video_id'] not in seen:
            seen.add(v['video_id'])
            # Formatting Date
            dt = datetime.fromisoformat(v['published_at'].replace('Z', '+00:00'))
            diff = datetime.now(timezone.utc) - dt
            
            if diff.total_seconds() < 3600:
                mins = int(diff.total_seconds() / 60)
                display_date = f"{mins} mins ago"
            elif diff.total_seconds() < 86400:
                hrs = int(diff.total_seconds() / 3600)
                display_date = f"{hrs} hours ago"
            else:
                days = int(diff.total_seconds() / 86400)
                display_date = f"{days} days ago"
                
            matched.append({
                "video_id": v['video_id'],
                "title": v['title'],
                "channel": v['channel'],
                "display_date": display_date,
                "thumbnail": v['thumbnail'],
                "dt": dt
            })
            
    # Sort matches newest first
    matched.sort(key=lambda x: x['dt'], reverse=True)
    return matched

if __name__ == "__main__":
    print("Updating media cache...")
    update_cache("IN")
    update_cache("US")
    print("Done!")
