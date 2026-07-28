import streamlit as st
import yfinance as yf
import pandas as pd
import urllib.request
import json
import requests
import io

@st.cache_data(ttl=300)
def get_us_vix():
    try:
        vix = yf.Ticker('^VIX').history(period='5d')
        if not vix.empty:
            return float(vix['Close'].dropna().iloc[-1])
    except Exception:
        pass
    return None

@st.cache_data(ttl=300)
def get_india_vix():
    try:
        vix = yf.Ticker('^INDIAVIX').history(period='5d')
        if not vix.empty:
            return float(vix['Close'].dropna().iloc[-1])
    except Exception:
        pass
    return None

@st.cache_data(ttl=300)
def get_tickertape_mmi():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }
        r = requests.get("https://api.tickertape.in/mmi/now", headers=headers, timeout=5)
        if r.status_code == 200:
            return float(r.json()['data']['currentValue'])
    except Exception:
        pass
    return None

@st.cache_data(ttl=300)
def fetch_fear_and_greed():
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Referer": "https://edition.cnn.com/",
            "Origin": "https://edition.cnn.com"
        }
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            score = data['fear_and_greed']['score']
            rating = data['fear_and_greed']['rating']
            return float(score), rating.title()
    except Exception:
        pass
    return None, None

@st.cache_data(ttl=3600)
def get_us_pcr():
    try:
        url = "https://cdn.cboe.com/data/us/options/market_statistics/daily/volume_and_put_call_ratios.csv"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/csv'
        }
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text), skiprows=2)
            if 'EQUITY P/C RATIO' in df.columns:
                return float(df['EQUITY P/C RATIO'].dropna().iloc[-1])
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def get_india_pcr():
    try:
        from nselib import derivatives
        df = derivatives.nse_live_option_chain('NIFTY')
        if not df.empty and 'PE_Open_Interest' in df.columns and 'CE_Open_Interest' in df.columns:
            tot_pe = df['PE_Open_Interest'].sum()
            tot_ce = df['CE_Open_Interest'].sum()
            if tot_ce > 0:
                return float(tot_pe / tot_ce)
    except Exception:
        pass
    return None

import plotly.graph_objects as go

def create_gauge(value, title, min_val, max_val, thresholds, inverse_colors=False, valueformat=".2f"):
    if value is None:
        fig = go.Figure()
        fig.add_annotation(text="Data Unavailable", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=18, color="#64748b"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(l=10, r=10, t=50, b=10))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig
    
    steps = []
    if len(thresholds) == 2:
        green_color = "rgba(16, 185, 129, 0.9)"
        amber_color = "rgba(245, 158, 11, 0.9)"
        red_color = "rgba(239, 68, 68, 0.9)"
        
        steps = [
            {'range': [min_val, thresholds[0]], 'color': green_color if not inverse_colors else red_color},
            {'range': [thresholds[0], thresholds[1]], 'color': amber_color},
            {'range': [thresholds[1], max_val], 'color': red_color if not inverse_colors else green_color}
        ]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16, 'color': '#f8fafc', 'family': 'Inter', 'weight': 'bold'}},
        number={'font': {'size': 38, 'color': '#ffffff', 'family': 'JetBrains Mono', 'weight': 'bold'}, 'valueformat': valueformat},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 2, 'tickcolor': "#475569", 'tickfont': {'size': 12, 'color': '#94a3b8'}, 'dtick': (max_val - min_val) / 4},
            'bar': {'color': "rgba(255,255,255,0.9)", 'thickness': 0.15},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': steps
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig
