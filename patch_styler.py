import re

# PATCH INTRADAY MONITOR IN
with open('C:/projects/Kush Tracker Lite/views/intraday_monitor.py', 'r', encoding='utf-8') as f:
    in_content = f.read()

# 1. Neutralize style_dataframe so it returns raw dataframe (enabling column_config to work!)
in_content = re.sub(r'def style_dataframe\(df\):\s+df = df\.copy\(\)[\s\S]*?return df\.style\.format[\s\S]*?highlight_intraday_rows, axis=1\)', 'def style_dataframe(df):\n        return df', in_content)

# 2. Fix TickerLink regex
in_content = in_content.replace(r'display_text=r"name=(.*)"', r'display_text=r".*NSE:(.*)"')

with open('C:/projects/Kush Tracker Lite/views/intraday_monitor.py', 'w', encoding='utf-8') as f:
    f.write(in_content)

# PATCH INTRADAY MONITOR US
with open('C:/projects/Kush Tracker Lite/views/intraday_monitor_us.py', 'r', encoding='utf-8') as f:
    us_content = f.read()

us_content = re.sub(r'def style_dataframe\(df\):\s+df = df\.copy\(\)[\s\S]*?return df\.style\.format[\s\S]*?highlight_intraday_rows, axis=1\)', 'def style_dataframe(df):\n        return df', us_content)
us_content = us_content.replace(r'display_text=r"name=(.*)"', r'display_text=r"symbol=(.*)"')

with open('C:/projects/Kush Tracker Lite/views/intraday_monitor_us.py', 'w', encoding='utf-8') as f:
    f.write(us_content)

print("Successfully patched intraday_monitor files!")
