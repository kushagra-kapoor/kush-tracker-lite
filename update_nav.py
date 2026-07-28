with open('C:/projects/Kush Tracker Lite/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re
content = re.sub(
    r'(st\.Page\("views/market_regime\.py".*?\))',
    r'\1,\n        st.Page("views/stage_analysis.py", title="Stage Analysis", icon="📊"),\n        st.Page("views/sector_leadership.py", title="Sector Leadership", icon="🔥")',
    content
)

with open('C:/projects/Kush Tracker Lite/app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated app.py navigation!")
