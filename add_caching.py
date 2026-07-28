import re

def add_caching(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Add caching to fetch_yfinance_batch
    content = re.sub(r'(?m)^def fetch_yfinance_batch', "@st.cache_data(ttl=300, show_spinner=False)\ndef fetch_yfinance_batch", content)

    # Add caching to process_intraday_data
    content = re.sub(r'(?m)^def process_intraday_data', "@st.cache_data(ttl=300, show_spinner=False)\ndef process_intraday_data", content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

add_caching('C:/projects/Kush Tracker Lite/views/intraday_monitor.py')
add_caching('C:/projects/Kush Tracker Lite/views/intraday_monitor_us.py')

print("Added intraday caching!")
