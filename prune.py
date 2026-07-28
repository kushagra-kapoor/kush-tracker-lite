import re

with open('C:/projects/Kush Tracker Lite/views/home.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip_mode = False

for i, line in enumerate(lines):
    # Stop processing completely when we hit the Portfolio Command Cockpit
    if 'Command Cockpit' in line:
        break
        
    # Remove load_data function entirely
    if line.startswith('@st.cache_data'):
        if i+1 < len(lines) and 'def load_data' in lines[i+1]:
            skip_mode = True
    
    if skip_mode and line.startswith('def main():'):
        skip_mode = False
        
    # Remove portfolio validation from main
    if 'portfolio_df, market_data' in line and 'load_data()' in line:
        continue
    if 'if portfolio_df is None or portfolio_df.empty:' in line:
        skip_mode = 'portfolio_check'
        continue
        
    if skip_mode == 'portfolio_check':
        if 'return' in line:
            skip_mode = False
        continue
        
    # Remove the massive pre-computation loop
    if 'decisions = []' in line:
        skip_mode = 'pre_compute'
        continue
        
    if skip_mode == 'pre_compute':
        if 'col1, col2, col3, col4 = st.columns(4)' in line or 'Deep Market Leaders' in line or 'Market FOMO' in line:
            skip_mode = False
        else:
            continue
            
    if not skip_mode:
        new_lines.append(line)

with open('C:/projects/Kush Tracker Lite/views/home.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
    
print("Pruned views/home.py successfully!")
