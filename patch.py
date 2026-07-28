import re

with open('C:/projects/Kush Tracker Lite/views/home.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove set_page_config
content = re.sub(r'(?m)^st\.set_page_config\([\s\S]*?\)', '# st.set_page_config removed for Lite routing', content)

# 2. Modify load_data() to not return early when portfolio is empty
content = content.replace('''    if portfolio_df.empty:
        return None, None, None, None, None''', '''    if portfolio_df.empty:
        pass # Continue loading macro data for Lite version''')

# 3. Remove portfolio early exit from main()
content = content.replace('''    if portfolio_df is None or portfolio_df.empty:
        st.error("? Failed to load portfolio data. Please check portfolio.csv.")
        return''', '''    # Portfolio exit removed for Lite version''')

# 4. Add return before Command Cockpit to stop rendering portfolio sections
content = content.replace('st.markdown("### ??? Command Cockpit")', 'return\n    st.markdown("### ??? Command Cockpit")')

# 5. Remove __main__ execution block
content = content.replace("if __name__ == '__main__':\n    main()", "main()")
content = content.replace('if __name__ == "__main__":\n    main()', "main()")

# Also update the pages.market_regime import
content = content.replace("from views.market_regime import", "from views.market_regime import")

with open('C:/projects/Kush Tracker Lite/views/home.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Restored and surgically patched views/home.py")

