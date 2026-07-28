with open('C:/projects/Kush Tracker Lite/views/home.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('C:/projects/Kush Tracker Lite/views/home.py', 'w', encoding='utf-8') as f:
    f.writelines(lines[:1790])

print("Successfully truncated home.py at line 1790!")
