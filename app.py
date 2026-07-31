"""
Kush Tracker Lite - Main Application Shell, Navigation & Authentication Entrypoint.
Cloud-Ready for Streamlit Community Cloud with Secure Login Gate.
"""
import streamlit as st
import sys
import os

# Ensure app directory is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import init_database
from styles import load_css

# Page Configuration
st.set_page_config(
    page_title="Kush Tracker Lite | CANSLIM Trading Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS & Init DB
try:
    load_css()
except Exception:
    pass

try:
    init_database()
except Exception as e:
    print(f"[App Init] Database initialization warning: {e}")

# --- AUTHENTICATION ENGINE ---
def check_auth_credentials(user_input, pass_input):
    """Check input credentials against Streamlit secrets or default fallback."""
    valid_user = "admin"
    valid_pass = "admin123"
    
    try:
        if "auth" in st.secrets:
            valid_user = st.secrets["auth"].get("username", valid_user)
            valid_pass = st.secrets["auth"].get("password", valid_pass)
    except Exception:
        pass
        
    return user_input.strip() == valid_user and pass_input.strip() == valid_pass

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Render Login Screen if not authenticated
if not st.session_state.authenticated:
    st.markdown("<br/><br/>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    
    with col_l2:
        st.markdown("""
        <div style='background: linear-gradient(145deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.9)); border: 1px solid rgba(0, 243, 255, 0.3); border-radius: 16px; padding: 32px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); text-align: center;'>
            <h1 style='color: #00F3FF; margin-bottom: 4px;'>🔒 Kush Tracker <span style='color: #FFD700;'>Lite</span></h1>
            <p style='color: #94a3b8; font-size: 14px; margin-bottom: 24px;'>Private CANSLIM & Minervini Execution Terminal</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter username")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter password")
            submit_login = st.form_submit_button("🔐 Sign In", use_container_width=True, type="primary")
            
            if submit_login:
                if check_auth_credentials(username, password):
                    st.session_state.authenticated = True
                    st.success("Authentication successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password. Access Denied.")
                    
        st.caption("🔒 Configured for private execution via Streamlit Secrets. Default: `admin / admin123`")
    st.stop()

# --- MAIN AUTHENTICATED APP SHELL ---

# Sidebar Brand Header
st.sidebar.markdown("""
<div style='text-align: center; padding: 10px 0;'>
    <h2 style='color: #00F3FF; margin:0;'>⚡ Kush Tracker <span style='color: #FFD700;'>Lite</span></h2>
    <p style='color: #94a3b8; font-size: 12px; margin-top:4px;'>CANSLIM & Minervini Execution Hub</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation Definition - Exact Original Kush Tracker Pages
pages = {
    "🏠 Main Dashboard": [
        st.Page("views/home.py", title="Kush Tracker Home", icon="🏠")
    ],
    "⚡ Intraday Monitors": [
        st.Page("views/intraday_monitor.py", title="Intraday Monitor (India)", icon="🇮🇳"),
        st.Page("views/intraday_monitor_us.py", title="Intraday Monitor (US)", icon="🇺🇸")
    ],
    "👑 Institutional Leaders": [
        st.Page("views/true_market_leader.py", title="True Market Leaders (India)", icon="👑"),
        st.Page("views/true_market_leader_us.py", title="True Market Leaders (US)", icon="🦅")
    ],
    "🛡️ Market Direction": [
        st.Page("views/global_macro.py", title="Global Macro & ETF Flows", icon="🌍"),
        st.Page("views/market_regime.py", title="Market Regime & Health", icon="🛡️"),
        st.Page("views/stage_analysis.py", title="Stage Analysis", icon="📊"),
        st.Page("views/sector_leadership.py", title="Sector Leadership", icon="🔥")
    ],
    "⭐ Execution Workspace": [
        st.Page("views/focus_list.py", title="Focus List & Breakouts", icon="⭐"),
        st.Page("views/debug_db.py", title="Debug DB internals", icon="🐛")
    ]
}

# Sign Out Button in Sidebar Footer
# Database Connection Status
st.sidebar.markdown("---")
try:
    from database import get_connection
    conn = get_connection()
    conn_type = type(conn).__name__
    if 'libsql' in str(type(conn)).lower() or 'Connection' in str(type(conn)) and 'sqlite3' not in str(type(conn)):
        st.sidebar.success(f"🟢 Database: Turso Cloud")
    else:
        st.sidebar.warning(f"🟡 Database: Local SQLite")
        st.sidebar.caption("⚠️ If you expect Cloud DB, check Streamlit Secrets.")
except Exception as e:
    st.sidebar.error("🔴 Database: Disconnected")

st.sidebar.markdown("---")
st.sidebar.caption("Kush Tracker Lite v2.0")
if st.sidebar.button("🚪 Sign Out", use_container_width=True):
    st.session_state.authenticated = False
    st.rerun()

# Run Modern Streamlit Router
pg = st.navigation(pages)
pg.run()
