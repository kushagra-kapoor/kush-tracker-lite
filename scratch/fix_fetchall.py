import re
import os

filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database.py')

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix pattern: rows = cursor.fetchall() followed by return _fetch_all_dicts(cursor)
content = re.sub(r'^[ \t]*rows\s*=\s*cursor\.fetchall\(\)\s*\n([ \t]*return\s*_fetch_all_dicts\(cursor\))', r'\1', content, flags=re.MULTILINE)

# Fix pattern: rows = [dict(r) for r in cursor.fetchall()]
content = content.replace('rows = [dict(r) for r in cursor.fetchall()]', 'rows = _fetch_all_dicts(cursor)')

# Verify get_journal_entry is fine
# We already fixed it in previous step. Let's do a sanity check to ensure no cursor.fetchone() followed by _fetch_one_dict(cursor)
content = re.sub(r'^[ \t]*row\s*=\s*cursor\.fetchone\(\)\s*\n([ \t]*return\s*_fetch_one_dict\(cursor\))', r'\1', content, flags=re.MULTILINE)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Double fetches cleaned up successfully!")
