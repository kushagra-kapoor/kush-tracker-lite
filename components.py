import streamlit as st
import plotly.graph_objects as go

def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Applies a professional, dark-mode transparent theme to Plotly figures."""
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)", 
            zerolinecolor="rgba(255,255,255,0.1)",
            color="#94a3b8"
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)", 
            zerolinecolor="rgba(255,255,255,0.1)",
            color="#94a3b8"
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_size=13,
            font_family="Inter, sans-serif"
        )
    )
    return fig

def render_header(title: str, subtitle: str, icon: str = ""):
    """Renders a standardized, universally styled modern app header."""
    
    # If standard streamlit emoji icon is provided, merge it into title
    display_title = f"{icon} {title}" if icon else title
    
    st.markdown(f"""
    <div class="st-app-header">
        <h1>{display_title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
    
def render_metric_card(label: str, value: str, color_class: str = "", extra_style: str = ""):
    """Renders a standardized metric card."""
    # color_class e.g., 'green-text', 'red-text', 'yellow-text'
    st.markdown(f"""
    <div class="metric-card" style="{extra_style}">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def render_score_card(score: str, label: str, state: str = "neutral"):
    """
    Renders score cards (Ready, Developing, Error, Neutral). 
    state must be one of: 'ready', 'developing', 'notready', 'error', 'neutral'
    """
    state_classes = {
        'ready': 'score-ready',
        'developing': 'score-developing',
        'notready': 'score-notready',
        'error': 'score-error',
        'neutral': 'score-neutral'
    }
    
    cls = state_classes.get(state, 'score-neutral')
    
    st.markdown(f"""
    <div class="score-card {cls}">
        <h2>{score}</h2>
        <p>{label}</p>
    </div>
    """, unsafe_allow_html=True)

def render_empty_state(message: str, icon: str = "📭"):
    """Renders a beautiful, premium empty state for when no stocks meet criteria."""
    st.markdown(f"""
    <div style="
        display: flex; 
        flex-direction: column; 
        align-items: center; 
        justify-content: center; 
        padding: 3rem 2rem; 
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px dashed rgba(100, 116, 139, 0.5);
        border-radius: 16px;
        margin: 1rem 0;
        text-align: center;
    ">
        <span style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.8;">{icon}</span>
        <h3 style="color: #cbd5e1; font-family: 'Inter', sans-serif; font-weight: 500; font-size: 1.1rem; margin: 0;">No Setups Found</h3>
        <p style="color: #94a3b8; font-family: 'Inter', sans-serif; font-size: 0.9rem; margin-top: 0.5rem; max-width: 400px;">
            {message}
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_disk_cache_sidebar(universe_fetcher_func=None):
    """Renders the local persistent NVMe cache status in the Streamlit Sidebar."""
    from price_history_manager import get_cache_status, fetch_incremental_history
    import streamlit as st
    
    with st.sidebar:
        st.markdown("---")
        tickers_count, max_date = get_cache_status()
        
        st.markdown("<h3 style='color: #f8fafc; margin-bottom: 0px;'>🗄️ Master Price Cache</h3>", unsafe_allow_html=True)
        if tickers_count > 0:
            st.markdown(f"<p style='color: #cbd5e1; font-size: 0.85em; margin-bottom: 2px;'><b>Persistent Universe:</b> {tickers_count} tickers</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #cbd5e1; font-size: 0.85em; margin-bottom: 15px;'><b>EOD Sync Date:</b> {max_date.strftime('%Y-%m-%d') if max_date else 'N/A'}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #cbd5e1; font-size: 0.85em; margin-bottom: 15px;'>Cache Empty. Run an Apex Scan to initialize.</p>", unsafe_allow_html=True)
            
        if universe_fetcher_func is not None:
            if st.button("🔄 Force EOD Delta Sync", use_container_width=True):
                tickers = universe_fetcher_func("Deep Market (2500+ NSE Stocks)")
                # Show an interactive progress bar for the background sync
                progress_text = "Incrementally fetching missing days... Please wait."
                my_bar = st.progress(0.0, text=progress_text)
                
                def update_progress(msg):
                    my_bar.progress(0.5, text=msg)

                fetch_incremental_history(tickers, days=252, progress_callback=update_progress)
                
                my_bar.progress(1.0, text="Sync Complete!")
                st.success("Universal EOD Sync Complete!")
                st.rerun()
                
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🧹 Scrub & Rebuild Database", type="secondary", use_container_width=True, help="Wipe cache and fetch 252-days backwards to safely adjust for Stock Splits."):
                tickers = universe_fetcher_func("Deep Market (2500+ NSE Stocks)")
                progress_text = "Destroying DB & Downloading Pristine History..."
                my_bar = st.progress(0.0, text=progress_text)
                
                def update_progress(msg):
                    my_bar.progress(0.5, text=msg)

                from price_history_manager import rebuild_full_history
                rebuild_full_history(tickers, days=252, progress_callback=update_progress)
                
                my_bar.progress(1.0, text="Rebuild Complete!")
                st.success("Database scrubbed & matrix perfectly rebuilt!")
                st.rerun()
