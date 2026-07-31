import streamlit as st
import traceback

st.title("Debug Database connection")

try:
    from database import get_connection, _fetch_all_dicts, get_all_fundamentals_cache
    
    st.subheader("1. Testing get_all_fundamentals_cache()")
    cache = get_all_fundamentals_cache()
    st.write(f"Cache length: {len(cache)}")
    if len(cache) > 0:
        first_key = list(cache.keys())[0]
        st.write(f"First element: {first_key} -> {cache[first_key]}")
    else:
        st.error("Cache is completely empty!")
        
    st.subheader("2. Testing RAW cursor fetch")
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM fundamentals_cache LIMIT 5")
    st.write("Description:")
    st.write(c.description)
    
    st.write("Fetchall (Raw):")
    raw_rows = c.fetchall()
    st.write(raw_rows)
    
    st.write("Fetch all dicts:")
    cols = [desc[0] for desc in c.description]
    st.write(f"Cols: {cols}")
    dicts = [dict(zip(cols, r)) for r in raw_rows]
    st.write(dicts)
    
except Exception as e:
    st.error(f"Error: {e}")
    st.text(traceback.format_exc())
