import os, re
d = r"C:\projects\Kush Tracker Lite\pages"
for f in os.listdir(d):
    if f.endswith('.py'):
        path = os.path.join(d, f)
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Comment out st.set_page_config
        new_content = re.sub(
            r'(st\.set_page_config\([^)]*\))', 
            lambda m: '\n'.join('# ' + line for line in m.group(1).split('\n')), 
            content, 
            flags=re.DOTALL
        )
        
        if new_content != content:
            with open(path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            print(f"Fixed {f}")
