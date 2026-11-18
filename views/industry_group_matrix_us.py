"""
US Industry Group Momentum Matrix (S&P 500 / Nasdaq)
===================================================
Institutional group rotation tracker for US equities with multi-timeframe relative strength rankings,
rank velocity deltas, pack-hunting breadth, actionable leader cards, and constituent drilldowns.
"""
import os
import sys
import importlib
import streamlit as st

# Add parent directory to path to import app modules
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

import views.industry_group_matrix as igm
importlib.reload(igm)

# Execute directly on every Streamlit rerun for seamless styling & rendering
igm.inject_industry_matrix_css()
igm.main(default_universe="US (S&P 500 / Nasdaq)")
