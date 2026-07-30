import re
import os

filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database.py')

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

helpers = """
def _fetch_all_dicts(cursor):
    rows = cursor.fetchall()
    if not rows: return []
    try:
        # Check if it already acts like a dict (e.g. sqlite3.Row)
        return [dict(r) for r in rows]
    except Exception:
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, r)) for r in rows]

def _fetch_one_dict(cursor):
    row = cursor.fetchone()
    if not row: return None
    try:
        return dict(row)
    except Exception:
        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))
"""

if '_fetch_all_dicts' not in content:
    # insert after get_connection
    content = content.replace('def init_database():', helpers + '\n\ndef init_database():')

# Remove row_factory lines
content = re.sub(r'^[ \t]*conn\.row_factory\s*=\s*sqlite3\.Row.*?\n', '', content, flags=re.MULTILINE)

# Replace fetch logic for the specific functions
replacements = [
    ("return [dict(row) for row in rows]", "return _fetch_all_dicts(cursor)"),
    ("return [dict(r) for r in cursor.fetchall()]", "return _fetch_all_dicts(cursor)"),
    ("return dict(row) if row else None", "return _fetch_one_dict(cursor)"),
    ("results = [dict(r) for r in cursor.fetchall()]", "results = _fetch_all_dicts(cursor)"),
]

for old, new in replacements:
    content = content.replace(old, new)

# Special fix for get_focus_list
content = content.replace(
    "rows = cursor.fetchall()\n        return [dict(row) for row in rows]", 
    "return _fetch_all_dicts(cursor)"
)

# Special fix for get_active_volume_shocks
content = content.replace(
    "rows = cursor.fetchall()\n        return [dict(row) for row in rows]",
    "return _fetch_all_dicts(cursor)"
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("database.py successfully patched for libsql compatibility!")
