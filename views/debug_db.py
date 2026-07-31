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
    
    st.write("Fetch all dicts (Limit 5):")
    cols = [desc[0] for desc in c.description]
    st.write(f"Cols: {cols}")
    dicts = [dict(zip(cols, r)) for r in raw_rows]
    st.write(dicts)
    
    st.subheader("2.5 Look up specific tickers")
    c.execute("SELECT * FROM fundamentals_cache WHERE ticker IN ('ADANIENT.NS', 'ADANIENT', 'SMLISUZU.NS', 'SMLISUZU')")
    spec_rows = c.fetchall()
    spec_dicts = [dict(zip(cols, r)) for r in spec_rows]
    st.write(spec_dicts)

    
    st.subheader("3. Test TML Snapshot Insertion")
    if st.button("Test Insert to tml_snapshot"):
        try:
            from database import save_tml_snapshot
            dummy_leaders = [{
                'ticker': 'TEST.NS',
                'tml_score': 99.9,
                'rs_score': 99,
                'Action_Status': 'Test Status',
                'industry': 'Test Industry'
            }]
            save_tml_snapshot("TEST_MARKET", dummy_leaders, top_n=1)
            st.success("Successfully executed save_tml_snapshot!")
            
            # Verify insertion
            c.execute("SELECT * FROM tml_snapshot WHERE ticker = 'TEST.NS'")
            test_row = c.fetchall()
            st.write("Verification Query Result:")
            st.write(test_row)
            
            # Clean up
            c.execute("DELETE FROM tml_snapshot WHERE ticker = 'TEST.NS'")
            conn.commit()
        except Exception as e:
            st.error(f"Insertion failed: {e}")
            st.text(traceback.format_exc())

    st.subheader("4. Emergency Turso Schema Patch")
    if st.button("🚨 Patch tml_snapshot Schema"):
        try:
            conn = get_connection()
            c = conn.cursor()
            c.execute("DROP TABLE IF EXISTS tml_snapshot")
            
            # Recreate with correct schema
            from database import init_database
            init_database()
            
            st.success("Successfully dropped and recreated tml_snapshot in Turso!")
        except Exception as e:
            st.error(f"Failed to patch schema: {e}")
            st.text(traceback.format_exc())
            
except Exception as e:
    st.error(f"Error: {e}")
    st.text(traceback.format_exc())
