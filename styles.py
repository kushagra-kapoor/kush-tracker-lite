import streamlit as st

def load_css():
    st.markdown("""
    <style>
        /* =========================================
           FUTURISTIC GLOW / ELITE TERMINAL THEME
           Matches the premium aesthetic of app.py
           ========================================= */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

        /* Global Reset */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, sans-serif !important;
            background-color: #020617 !important; /* Deepest Slate */
            color: #f8fafc !important;
        }

        /* Top Padding Fix for edge-to-edge immersion */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 3rem !important;
            max-width: 95% !important;
        }

        /* -------------------------------------
           App Header (Glowing, Majestic)
           ------------------------------------- */
        .st-app-header {
            background: linear-gradient(145deg, rgba(15,23,42,0.8) 0%, rgba(2,6,23,0.95) 100%) !important;
            padding: 2.5rem !important;
            margin-bottom: 2.5rem !important;
            border-radius: 20px !important;
            border: 1px solid rgba(255,255,255,0.05) !important;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.8), inset 0 1px 0 rgba(255,255,255,0.1) !important;
            position: relative;
            overflow: hidden;
            text-align: center;
        }
        
        /* Subtle glow light leak */
        .st-app-header::before {
            content: '';
            position: absolute;
            top: -50%; left: 20%; width: 60%; height: 200%;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, transparent 70%);
            transform: rotate(-15deg);
            pointer-events: none;
        }

        .st-app-header h1 {
            color: #ffffff !important;
            font-size: 3rem !important;
            font-weight: 800 !important;
            margin: 0 !important;
            letter-spacing: -1px !important;
            text-shadow: 0 0 30px rgba(59,130,246,0.5) !important;
            background: linear-gradient(to right, #ffffff, #93c5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .st-app-header p {
            color: #94a3b8 !important;
            font-size: 1.15rem !important;
            margin-top: 0.75rem !important;
            font-weight: 300 !important;
            letter-spacing: 0.5px;
        }

        /* -------------------------------------
           Metric Cards (Deep Shadows, Neon Text)
           ------------------------------------- */
        .metric-card, [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(145deg, rgba(15,23,42,0.9) 0%, rgba(2,6,23,1) 100%) !important;
            border-radius: 20px !important;
            padding: 2rem !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-top: 3px solid #3b82f6 !important;
            box-shadow: 0 15px 30px -5px rgba(0,0,0,0.8), inset 0 1px 0 rgba(255,255,255,0.1) !important;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative;
            overflow: hidden;
        }
        .metric-card:hover, [data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-5px) scale(1.02);
            border-color: rgba(59,130,246,0.3) !important;
            box-shadow: 0 20px 40px -10px rgba(0,0,0,0.7), 0 0 20px rgba(59,130,246,0.15) !important;
        }
        
        .metric-label, .metric-card div:first-child {
            color: #94a3b8 !important;
            font-size: 0.85rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 1.5px !important;
            margin-bottom: 0.75rem !important;
        }
        .metric-value {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            line-height: 1 !important;
            color: #ffffff !important;
        }

        /* Glowing Text Colors */
        .green-text { color: #10b981 !important; text-shadow: 0 0 20px rgba(16,185,129,0.4) !important; }
        .red-text { color: #ef4444 !important; text-shadow: 0 0 20px rgba(239,68,68,0.4) !important; }
        .yellow-text { color: #f59e0b !important; text-shadow: 0 0 20px rgba(245,158,11,0.4) !important; }
        .blue-text { color: #3b82f6 !important; text-shadow: 0 0 20px rgba(59,130,246,0.4) !important; }
        
        .rs-blue-dot-glow {
            color: #60a5fa !important;
            text-shadow: 0 0 15px rgba(96,165,250,0.6) !important;
            font-weight: 800 !important;
        }

        /* -------------------------------------
           Expanders (Sleek Panels)
           ------------------------------------- */
        [data-testid="stExpander"] {
            background: rgba(15,23,42,0.4) !important;
            border: 1px solid rgba(255,255,255,0.05) !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3) !important;
            backdrop-filter: blur(10px);
        }
        [data-testid="stExpander"] summary {
            font-weight: 600 !important;
            color: #f1f5f9 !important;
            padding: 1rem !important;
            letter-spacing: 0.5px;
        }
        [data-testid="stExpander"]:hover {
            border-color: rgba(255,255,255,0.15) !important;
        }

        /* -------------------------------------
           Dataframes (Premium Institutional)
           ------------------------------------- */
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(255,255,255,0.05) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
            background: rgba(15,23,42,0.4) !important;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.4) !important;
        }
        [data-testid="stDataFrame"] th {
            background: rgba(2,6,23,0.8) !important;
            color: #94a3b8 !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.8rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            border-bottom: 1px solid rgba(255,255,255,0.05) !important;
        }
        [data-testid="stDataFrame"] td {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.9rem !important;
            color: #e2e8f0 !important;
            border-bottom: 1px solid rgba(255,255,255,0.02) !important;
        }

        /* -------------------------------------
           Buttons (Glowing Interactions)
           ------------------------------------- */
        [data-testid="stBaseButton-primary"] button {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            border: 1px solid rgba(59,130,246,0.4) !important;
            border-radius: 8px !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            padding: 0.5rem 1.5rem !important;
            box-shadow: 0 0 15px rgba(37,99,235,0.4) !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stBaseButton-primary"] button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 0 25px rgba(59,130,246,0.6) !important;
            border-color: rgba(96,165,250,0.8) !important;
        }
        
        [data-testid="stBaseButton-secondary"] button {
            background: rgba(30,41,59,0.5) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            color: #f1f5f9 !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stBaseButton-secondary"] button:hover {
            background: rgba(51,65,85,0.8) !important;
            border-color: rgba(255,255,255,0.2) !important;
            color: #ffffff !important;
        }
        /* -------------------------------------
           Institutional Containers (st.container(border=True))
           ------------------------------------- */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(145deg, rgba(30,41,59,0.2) 0%, rgba(15,23,42,0.5) 100%) !important;
            border-radius: 16px !important;
            padding: 1rem !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.02) !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(59,130,246,0.3) !important;
            box-shadow: 0 15px 35px -10px rgba(0,0,0,0.6), 0 0 20px rgba(59,130,246,0.1) !important;
        }



        
        /* -------------------------------------
           Badges & Alerts
           ------------------------------------- */
        .badge-exit { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 700; font-size: 0.75rem; text-shadow: 0 0 10px rgba(239,68,68,0.3); }
        .badge-trim { background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 700; font-size: 0.75rem; text-shadow: 0 0 10px rgba(245,158,11,0.3); }
        .badge-hold { background: rgba(100,116,139,0.2); color: #cbd5e1; border: 1px solid rgba(100,116,139,0.4); padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 700; font-size: 0.75rem; }
        .badge-add  { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 700; font-size: 0.75rem; text-shadow: 0 0 10px rgba(16,185,129,0.3); }

        .urgent-alert {
            background: linear-gradient(135deg, rgba(127,29,29,0.4) 0%, rgba(69,10,10,0.8) 100%) !important;
            border-left: 4px solid #ef4444 !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            margin: 1rem 0 !important;
            border: 1px solid rgba(239,68,68,0.2) !important;
            box-shadow: 0 10px 20px rgba(0,0,0,0.5) !important;
        }
        
        /* -------------------------------------
           Custom Scrollbars
           ------------------------------------- */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #020617; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #334155; }
    </style>
    """, unsafe_allow_html=True)
