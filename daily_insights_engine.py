import pandas as pd
import numpy as np

def generate_exposure_guide(regime_label: str, ler_current: float = None) -> dict:
    """
    Generates a progressive exposure guide based on the macro regime and
    underlying leadership health.
    """
    exposure = {
        'level': '0%',
        'stance': 'Risk OFF',
        'color': '#ef4444',
        'advice': 'Protect capital. Stay in cash or LiquidBees.'
    }
    
    if "Confirmed Uptrend" in regime_label:
        if ler_current is not None and ler_current >= 15:
            # Strong bull, broad leadership
            exposure = {
                'level': '100%',
                'stance': 'Aggressive Bull',
                'color': '#10b981',
                'advice': 'Full exposure. Press winners aggressively and use margin if comfortable.'
            }
        else:
            # Uptrend, but leadership is thin or just starting
            exposure = {
                'level': '75%',
                'stance': 'Progressive Bull',
                'color': '#34d399',
                'advice': 'Scale in. Increase size only as current positions work and show profit cushions.'
            }
            
    elif "Uptrend Under Pressure" in regime_label:
        exposure = {
            'level': '50%',
            'stance': 'Risk CAUTION',
            'color': '#f59e0b',
            'advice': 'Raise stops. Trim weaker holdings. Do not initiate new full-size positions.'
        }
        
    elif "Correction" in regime_label:
        exposure = {
            'level': '0 - 25%',
            'stance': 'Risk OFF',
            'color': '#ef4444',
            'advice': 'Heavy defense. Cash is a position. Only hold absolute strongest elite leaders with deep profit cushions.'
        }
        
    return exposure

def get_sector_clusters(sector_strength_df: pd.DataFrame, top_n: int = 3) -> list:
    """
    Identifies the leading sector clusters based on the sector strength aggregation.
    Expects the output of aggregate_sector_strength from macro_regime_engine.
    """
    clusters = []
    if sector_strength_df is None or sector_strength_df.empty:
        return clusters
        
    # We want sectors that have a decent number of stocks and high RS/Bullish %
    # Filter out tiny sectors to avoid noise
    valid_sectors = sector_strength_df[sector_strength_df['Stocks'] >= 5].copy()
    
    # Sort by a combination of RS and Bullish %
    if not valid_sectors.empty:
        # Create a composite score
        valid_sectors['Cluster_Score'] = valid_sectors['Avg_RS'] * 0.6 + valid_sectors['Bullish %'] * 0.4
        top_sectors = valid_sectors.sort_values('Cluster_Score', ascending=False).head(top_n)
        
        for _, row in top_sectors.iterrows():
            clusters.append({
                'name': row['Industry'],
                'avg_rs': row.get('Avg_RS', 0),
                'bullish_pct': row.get('Bullish %', 0),
                'count': row.get('Stocks', 0)
            })
            
    return clusters

def generate_macro_health_score(ler_current: float, lac_current: float, lt_current: float, bt_current: float) -> dict:
    """
    Synthesizes the 4 key metrics into a single Macro Health Score (0-100).
    """
    # Defensive programming against None
    ler = ler_current or 0
    lac = lac_current or 0
    lt = lt_current or 0
    bt = bt_current or 0.5 # Neutral for BT is 0.5
    
    # 1. LER (0-40 points) - 20% is considered very strong, 30%+ is elite
    ler_score = min(ler / 25.0 * 40, 40)
    
    # 2. LAC (0-20 points) - Rate of change. +2% growth per week is excellent
    lac_score = max(0, min((lac + 5) / 10.0 * 20, 20)) # Shifted to reward positive acceleration
    
    # 3. LT (0-20 points) - 5% hitting new highs is very strong
    lt_score = min(lt / 5.0 * 20, 20)
    
    # 4. BT (0-20 points) - Above 0.5 is good
    bt_score = max(0, (bt - 0.3) / 0.4 * 20)
    bt_score = min(bt_score, 20)
    
    total_score = ler_score + lac_score + lt_score + bt_score
    total_score = min(max(total_score, 0), 100) # Clamp 0-100
    
    if total_score >= 80:
        label = "Euphoric / Extremely Strong"
        color = "#10b981"
    elif total_score >= 60:
        label = "Healthy Bull"
        color = "#34d399"
    elif total_score >= 40:
        label = "Constructive / Improving"
        color = "#fcd34d"
    elif total_score >= 20:
        label = "Weak / Deteriorating"
        color = "#f97316"
    else:
        label = "Severe Bear / Cash Regime"
        color = "#ef4444"
        
    return {
        'score': total_score,
        'label': label,
        'color': color,
        'components': {
            'LER': ler,
            'LAC': lac,
            'LT': lt,
            'BT': bt
        }
    }
