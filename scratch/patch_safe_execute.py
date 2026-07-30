import re
import os

filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database.py')

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# We only want to replace cursor.execute in init_database()
# So we slice the content.
start_idx = content.find('def init_database():')
end_idx = content.find('def get_open_signals(', start_idx)

init_db_content = content[start_idx:end_idx]

# Define safe_execute above init_database
safe_execute_def = """
def safe_execute(cursor, query):
    try:
        cursor.execute(query)
    except Exception:
        pass
"""

# Replace all cursor.execute in init_database with safe_execute
# but we have some try/except blocks already around cursor.execute in init_db_content
# To be safe, just replacing cursor.execute with safe_execute(cursor, ) is fine.
init_db_content = re.sub(r'cursor\.execute\((.*?)\)', r'safe_execute(cursor, \1)', init_db_content, flags=re.DOTALL)

# Re-assemble
content = content[:start_idx] + safe_execute_def + init_db_content + content[end_idx:]

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("init_database successfully patched with safe_execute!")
