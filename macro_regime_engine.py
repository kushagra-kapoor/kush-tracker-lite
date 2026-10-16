import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def standardize_columns(df: pd.DataFrame):
    """
    Standardizes dataframe columns to lowercase and handles MultiIndex flattening.
    """
    if df.empty:
        return df
    
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # Find the level that contains OHLCV data
        found_level = -1
        for i in range(df.columns.nlevels):
            level_vals = [str(c).lower() for c in df.columns.get_level_values(i)]
            if 'close' in level_vals:
                found_level = i
                break
        df.columns = df.columns.get_level_values(found_level)
    
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df

def calculate_distribution_days(df: pd.DataFrame, window: int = 25):
    """
    Count distribution days in the specified window for a major index.
    A Distribution Day (DD) is: 
    1. Index close < -0.2%
    2. Volume > Previous Day's Volume
    
    Returns:
        (dd_count, dd_dates_list)
    """
    if df.empty or len(df) < window + 1:
        return 0, []
    
    # Standardize
    df = standardize_columns(df)
    
    # Safely find close and volume
    if 'close' not in df.columns or 'volume' not in df.columns:
        return 0, []
    
    # Calculate price change
    df['pct_change'] = df['close'].pct_change() * 100
    
    # Check for DD condition
    df['is_dd'] = (df['pct_change'] <= -0.2) & (df['volume'] > df['volume'].shift(1))
    
    # Get the last 25 trading days
    recent = df.tail(window)
    dd_rows = recent[recent['is_dd']]
    
    if 'date' in df.columns:
        dd_dates_list = pd.to_datetime(dd_rows['date']).dt.strftime('%Y-%m-%d').tolist()
    else:
        dd_dates_list = dd_rows.index.strftime('%Y-%m-%d').tolist()
    
    return len(dd_dates_list), dd_dates_list

def detect_follow_through_day(df: pd.DataFrame):
    """
    Detects a Follow-Through Day (FTD) which signals the start of a new uptrend.
    FTD = Day 4 to 10 of a rally attempt where index gains > 1.5% on higher volume.
    """
    if df.empty or len(df) < 20:
        return False, None
        
    # Standardize
    recent = standardize_columns(df.tail(15))
    
    if 'close' not in recent.columns or 'volume' not in recent.columns:
        return False, None

    recent['pct_change'] = recent['close'].pct_change() * 100
    ftd = recent[(recent['pct_change'] >= 1.5) & (recent['volume'] > recent['volume'].shift(1))]
    
    if not ftd.empty:
        last_row = ftd.iloc[-1]
        if 'date' in recent.columns:
            dt_str = pd.to_datetime(last_row['date']).strftime('%Y-%m-%d')
        else:
            dt_str = ftd.index[-1].strftime('%Y-%m-%d')
        return True, dt_str
    return False, None

def get_market_regime_label(dd_count, current_price, sma50):
    """
    Classifies the market into one of three regimes.
    """
    if current_price < sma50:
        return "Market in Correction", "red", "🔴"
        
    if dd_count >= 6:
        return "Market in Correction", "red", "🔴"
    elif dd_count >= 4:
        return "Uptrend Under Pressure", "yellow", "🟡"
    else:
        return "Confirmed Uptrend", "green", "🟢"

def detect_change_of_character(df: pd.DataFrame, window: int = 20):
    """
    Wyckoff-style "Change of Character" (ChoCh) detector for the index.
    Looks for climactic volume on wide spreads or sudden reversal days.
    """
    if df.empty or len(df) < window + 10:
        return None
        
    df = standardize_columns(df).tail(window).copy()
    if 'close' not in df.columns or 'volume' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
        return None
        
    df['range'] = df['high'] - df['low']
    df['pct_change'] = df['close'].pct_change() * 100
    avg_vol = df['volume'].mean()
    avg_range = df['range'].mean()
    
    last_day = df.iloc[-1]
    prev_day = df.iloc[-2]
    
    # 1. Volume Climax (Bottom or Top)
    is_climax_vol = last_day['volume'] > (avg_vol * 1.5)
    
    # 2. Bullish Reversal (Spring)
    # Undercut previous low but closed higher
    is_spring = last_day['low'] < prev_day['low'] and last_day['close'] > prev_day['close'] and is_climax_vol
    
    # 3. Bearish Reversal (Upthrust)
    # Pierced previous high but closed lower
    is_upthrust = last_day['high'] > prev_day['high'] and last_day['close'] < prev_day['close'] and is_climax_vol
    
    # 4. Wide Spread Down (Sign of Weakness)
    is_sow = last_day['range'] > (avg_range * 1.5) and last_day['pct_change'] < -1.5 and is_climax_vol
    
    # 5. Wide Spread Up (Sign of Strength)
    is_sos = last_day['range'] > (avg_range * 1.5) and last_day['pct_change'] > 1.5 and is_climax_vol
    
    choch_label = "Neutral"
    choch_type = "None"
    color = "gray"
    
    if is_spring:
        choch_label = "Bullish Spring (Accumulation)"
        choch_type = "Bullish"
        color = "#10b981"
    elif is_sos:
        choch_label = "Sign of Strength (Demand)"
        choch_type = "Bullish"
        color = "#34d399"
    elif is_upthrust:
        choch_label = "Bearish Upthrust (Distribution)"
        choch_type = "Bearish"
        color = "#ef4444"
    elif is_sow:
        choch_label = "Sign of Weakness (Supply)"
        choch_type = "Bearish"
        color = "#b91c1c"
        
    return {
        'detected': choch_type != "None",
        'type': choch_type,
        'label': choch_label,
        'color': color,
        'date': last_day.name.strftime('%Y-%m-%d') if hasattr(last_day, 'name') and hasattr(last_day.name, 'strftime') else "Today"
    }

def aggregate_sector_strength(results_df: pd.DataFrame):
    """
    results_df expects: ['Ticker', 'Industry', 'Today %', 'RS Score', 'Trend State', 'Near_High', 'RS_Score_T21']
    """
    if results_df.empty:
        return pd.DataFrame()
        
    # Group by Industry
    sector_agg = results_df.groupby('Industry').agg(
        Stocks=('Ticker', 'count'),
        Avg_RS=('RS Score', 'mean'),
        Avg_RS_T21=('RS_Score_T21', 'mean'),
        Bullish_Count=('Trend State', lambda x: (x == '🟢 Strong').sum()),
        Avg_Today_Pct=('Today %', 'mean'),
        Leaders_Count=('RS Score', lambda x: (x >= 85).sum()),
        Near_High_Count=('Near_High', 'sum')
    )
    
    # Derivations
    sector_agg['Bullish %'] = (sector_agg['Bullish_Count'] / sector_agg['Stocks']) * 100
    sector_agg['Leadership Density'] = (sector_agg['Leaders_Count'] / sector_agg['Stocks']) * 100
    sector_agg['Near High Breadth'] = (sector_agg['Near_High_Count'] / sector_agg['Stocks']) * 100
    sector_agg['RS Momentum 21d'] = sector_agg['Avg_RS'] - sector_agg['Avg_RS_T21']
    
    # Sort by Avg RS mainly
    sector_agg = sector_agg.sort_values('Avg_RS', ascending=False).reset_index()
    
    return sector_agg

# =============================================================================
# MARKET LEADERSHIP & BREADTH SIGNALS (LER, LAC, LT, BT)
# =============================================================================

def calculate_leadership_metrics(history_df, tickers, rs_scores):
    """
    Computes Leader Emergence Rate (LER), Leadership Acceleration Curve (LAC),
    and Leadership Thrust (LT) using historical universe data.
    
    Optimized version using vectorized operations over the provided history.
    """
    import numpy as np

    is_multi = isinstance(history_df.columns, pd.MultiIndex)
    if not is_multi or len(tickers) <= 1:
        return None
        
    # Clean tickers and RS scores
    rs_clean = {str(k).replace('.NS', '').upper(): v for k, v in rs_scores.items()}
    
    # Extract close, high, volume panels for the whole universe
    # This is MUCH faster than looping stocks and slicing
    try:
        # Identify OHLCV levels
        l0 = [str(x).lower() for x in history_df.columns.get_level_values(0)]
        l1 = [str(x).lower() for x in history_df.columns.get_level_values(1)]
        
        if 'close' in l0: ticker_level, ohlcv_level = 1, 0
        else: ticker_level, ohlcv_level = 0, 1
        
        # Helper to get a panel
        def get_panel(col_name):
            col_key = history_df.columns.get_level_values(ohlcv_level)[
                [str(x).lower() for x in history_df.columns.get_level_values(ohlcv_level)].index(col_name.lower())
            ]
            return history_df.xs(col_key, level=ohlcv_level, axis=1)

        close_panel = get_panel('close')
        high_panel = get_panel('high')
        
    except Exception as e:
        print(f"Error extracting panels: {e}")
        return None

    # Calculate indicators for the whole panel (column-wise)
    sma50_panel = close_panel.rolling(window=50, min_periods=50).mean()
    sma200_panel = close_panel.rolling(window=200, min_periods=200).mean()
    high52w_panel = high_panel.rolling(window=252, min_periods=1).max()
    ret8w_panel = close_panel.pct_change(periods=40, fill_method=None) * 100
    
    # Map RS scores to a row compatible with the panel columns
    current_rs_row = pd.Series(index=close_panel.columns, dtype=float)
    for col in close_panel.columns:
        symbol = str(col).replace('.NS', '').upper()
        current_rs_row[col] = rs_clean.get(symbol, 0)
        
    # Condition 1: Leadership Thrust (LT) - Daily (Stocks near 52W High)
    lt_hits = (close_panel >= high52w_panel * 0.99).sum(axis=1)
    lt_series = (lt_hits / len(tickers)) * 100
    lt_ma5 = lt_series.rolling(window=5, min_periods=1).mean()
    
    # Condition 2: Leader Emergence Rate (LER)
    # Conditions: RS >= 85 (current proxy), Close >= 90% High, Close > 50SMA, Close > 200SMA, Return_8W >= 25%
    c_high = (close_panel >= 0.90 * high52w_panel)
    c_sma50 = (close_panel > sma50_panel)
    c_sma200 = (close_panel > sma200_panel)
    c_ret8w = (ret8w_panel >= 25)
    
    # We omit the 'Static RS' check for historical dates to avoid survivorship bias.
    # Today's LER will still feel the effect of current RS in the interpreted metrics if desired,
    # but for the chart we prioritize price momentum signals.
    leader_panel = c_high & c_sma50 & c_sma200 & c_ret8w
    
    # Weekly sampling for LER/LAC
    weekly_leaders = leader_panel.resample('W-FRI').last().sum(axis=1)
    ler_series = (weekly_leaders / len(tickers)) * 100
    
    # LAC (Leadership Acceleration Curve)
    leader_growth = weekly_leaders.diff().fillna(0)
    lac_series = leader_growth.ewm(span=3, adjust=False).mean()
    
    # Final cleanup
    # Only return last 100 days for daily, 24 weeks for weekly
    return {
        'lt_series': lt_ma5.tail(100),
        'lt_current': lt_ma5.iloc[-1] if not lt_ma5.empty else 0,
        'ler_series': ler_series.tail(24),
        'ler_current': ler_series.iloc[-1] if not ler_series.empty else 0,
        'lac_series': lac_series.tail(24),
        'lac_current': lac_series.iloc[-1] if not lac_series.empty else 0,
        'leader_counts': weekly_leaders.tail(24),
        'universe_size': len(tickers)
    }

def calculate_breadth_thrust(history_df, tickers):
    """
    Computes the Breadth Thrust (BT) signal based on the 10-day MA of Ad/Dec breadth.
    """
    is_multi = isinstance(history_df.columns, pd.MultiIndex) and len(tickers) > 1
    
    try:
        if is_multi:
            ohlcv_level = -1
            for i in range(history_df.columns.nlevels):
                l_vals = [str(v).lower() for v in history_df.columns.get_level_values(i)]
                if 'close' in l_vals:
                    ohlcv_level = i
                    break
            if ohlcv_level == -1:
                return None
                
            key = history_df.columns.get_level_values(ohlcv_level)[l_vals.index('close')]
            close_panel = history_df.xs(key, level=ohlcv_level, axis=1)
        else:
             return None
             
        # Compute daily returns
        daily_rets = close_panel.pct_change()
        
        # Count advancers and decliners per day (last 100 days)
        daily_rets = daily_rets.tail(100)
        advancers = (daily_rets > 0).sum(axis=1)
        decliners = (daily_rets < 0).sum(axis=1)
        total_active = advancers + decliners
        
        # Avoid division by zero
        total_active = total_active.replace(0, 1)
        
        breadth_ratio = advancers / total_active
        breadth_ma10 = breadth_ratio.rolling(window=10, min_periods=1).mean()
        
        # Check for Thrust trigger: Rises from < 0.40 to > 0.60 within 10 days
        thrust_triggered = False
        if len(breadth_ma10) >= 10:
            last_10 = breadth_ma10.tail(10)
            if last_10.min() < 0.40 and last_10.max() > 0.60:
                # Ensure the low happened before the high
                idx_min = last_10.idxmin()
                idx_max = last_10.idxmax()
                if idx_min < idx_max and last_10.iloc[-1] > 0.60:
                    thrust_triggered = True
                    
        return {
            'breadth_ma10_series': breadth_ma10,
            'breadth_ma10_current': breadth_ma10.iloc[-1] if not breadth_ma10.empty else 0.5,
            'thrust_triggered': thrust_triggered
        }
    except Exception as e:
        print(f"Breadth erro: {e}")
        return None

def get_current_regime():
    """Determine the current market regime from Nifty 500."""
    try:
        from config import BENCHMARK_TICKER
        import yfinance as yf
        idx = yf.download(BENCHMARK_TICKER, period="100d", progress=False)
        if idx.empty:
            return 'Market in Correction', 'red', "Failed to download NIFTY 500 data."

        idx_std = standardize_columns(idx)
        if idx_std.empty or 'close' not in idx_std.columns:
            return 'Market in Correction', 'red', "Failed to standardize NIFTY 500 columns."

        dd_count, _ = calculate_distribution_days(idx)
        curr = idx_std['close'].iloc[-1]
        sma50 = idx_std['close'].rolling(50).mean().iloc[-1] if len(idx_std) >= 50 else curr

        regime, color_name, _ = get_market_regime_label(dd_count, curr, sma50)

        reason = []
        if curr < sma50:
            reason.append(f"The benchmark ({BENCHMARK_TICKER}) is trading **below** its 50-day moving average (Current: {curr:.0f} vs 50SMA: {sma50:.0f}).")
        else:
            reason.append(f"The benchmark ({BENCHMARK_TICKER}) is trading **above** its 50-day moving average (Current: {curr:.0f} vs 50SMA: {sma50:.0f}).")
        
        reason.append(f"It has registered {dd_count} Distribution Days over the last 25 trading sessions.")

        return regime, color_name, " ".join(reason)
    except Exception as e:
        return 'Market in Correction', 'red', f"Error computing regime: {e}"

def get_clenow_regime(benchmark_df=None):
    """
    Evaluates Clenow's Market Regime.
    Returns:
        (is_bull_regime: bool, label: str, explanation: str)
    """
    if benchmark_df is None or benchmark_df.empty:
        try:
            from config import BENCHMARK_TICKER
            import yfinance as yf
            idx = yf.download(BENCHMARK_TICKER, period="250d", progress=False)
            if not idx.empty:
                benchmark_df = standardize_columns(idx)
        except:
            pass
            
    if benchmark_df is None or benchmark_df.empty:
        return False, "Unknown Regime", "Failed to fetch benchmark data."
        
    close_col = 'close'
    if close_col not in benchmark_df.columns:
        return False, "Unknown Regime", "No close data found."
        
    current_price = benchmark_df[close_col].iloc[-1]
    
    if len(benchmark_df) < 200:
        sma200 = benchmark_df[close_col].mean()
    else:
        sma200 = benchmark_df[close_col].rolling(200).mean().iloc[-1]
        
    if current_price > sma200:
        return True, "Bull Regime", f"Benchmark is {current_price:.0f}, above its 200-day SMA of {sma200:.0f}. **System is allowed to buy.**"
    else:
        return False, "Bear Regime", f"Benchmark is {current_price:.0f}, below its 200-day SMA of {sma200:.0f}. **System is prohibited from buying.**"


def detect_20ema_recapture_and_sequence(df: pd.DataFrame) -> dict:
    """
    Paul Tudor Jones (Market Wizards p. 120) & Shake Pryzby (Pinpoint Trading p. 46):
    
    1. Secular Macro Regime (200-day Simple Moving Average):
       - 'My metric for everything to give me what direction the market is in is the 200-day moving average.
          I've seen too many things go to zero, stocks and commodities. The whole trick in investing is:
          How do I keep from losing everything? If you use the 200-day moving average rule, then you get out.'
          (Paul Tudor Jones)
       - Secular Bull: Price >= 200 SMA
       - Secular Bear: Price < 200 SMA
       
    2. Tactical 20 EMA Recapture Buy Signal:
       - Triggers when price crosses and closes above the 20 EMA after residing below it for >= 3 sessions,
         or tests and bounces cleanly off a rising 20 EMA.
       - Signals institutional supply-demand flip and tactical risk-on pilot buy clearance.
       
    3. The 7-Step Bottom Progression Sequence (Shake Pryzby / Minervini / O'Neil):
       - Step 1: Capitulation (Heavy volume liquidation day, drop <= -1.2%)
       - Step 2: Turnaround (Hammer / Reversal candle with DCR >= 55%)
       - Step 3: Volatility Contraction (3-day ATR compresses below 20-day ATR)
       - Step 4: MA Clustering (Price coils within 1.5% of 10/20 EMA)
       - Step 5: 20 EMA Recapture (Price closes decisively above 20 EMA) -> TACTICAL BUY
       - Step 6: 10/20 EMA Golden Cross (10 EMA crosses above 20 EMA)
       - Step 7: Stress Test Confirmed (Pullback tests 20 EMA and holds on light volume)
    """
    default_res = {
        'secular_bull': True,
        'dist_200_sma_pct': 0.0,
        'secular_label': "Secular Bull (Above 200 SMA)",
        'secular_color': "#10b981",
        'recapture_active': False,
        'recapture_fresh': False,
        'recapture_sessions_ago': -1,
        'current_step_num': 1,
        'current_step_name': "Base Foundation",
        'step_badge_color': "#94a3b8",
        'sequence_summary': "Step 1/7: Initial Base Construction",
        'action_guidance': "Monitor market internals.",
        'steps_detail': {}
    }
    
    if df.empty or len(df) < 20:
        return default_res
        
    df_std = standardize_columns(df).copy()
    if 'close' not in df_std.columns:
        return default_res
        
    close = df_std['close'].dropna()
    if len(close) < 20:
        return default_res
        
    # Moving Averages
    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    sma50 = close.rolling(50, min_periods=10).mean()
    sma200 = close.rolling(200, min_periods=25).mean()
    
    c_now = float(close.iloc[-1])
    sma200_now = float(sma200.iloc[-1])
    ema20_now = float(ema20.iloc[-1])
    ema10_now = float(ema10.iloc[-1])
    sma50_now = float(sma50.iloc[-1])
    
    # 1. Secular Regime (PTJ 200 SMA)
    secular_bull = c_now >= sma200_now
    dist_200 = ((c_now - sma200_now) / sma200_now) * 100.0 if sma200_now > 0 else 0.0
    secular_label = "Secular Bull (Above 200 SMA)" if secular_bull else "Secular Bear (Below 200 SMA)"
    secular_color = "#10b981" if secular_bull else "#ef4444"
    
    # 2. Tactical 20 EMA Recapture
    below_mask = close < ema20
    above_mask = close >= ema20
    
    recapture_active = False
    recapture_fresh = False
    recapture_sessions_ago = -1
    
    if c_now >= ema20_now:
        prior_below = below_mask.iloc[-12:-1]
        if prior_below.sum() >= 2:
            for i in range(1, min(6, len(close))):
                if close.iloc[-i] >= ema20.iloc[-i] and close.iloc[-i - 1] < ema20.iloc[-i - 1]:
                    recapture_active = True
                    recapture_sessions_ago = i - 1
                    recapture_fresh = (recapture_sessions_ago <= 1)
                    break
            if not recapture_active and prior_below.iloc[-1]:
                recapture_active = True
                recapture_fresh = True
                recapture_sessions_ago = 0
                
    # 3. 7-Step Bottom Progression
    sub_df = df_std.tail(30).copy()
    sub_close = sub_df['close']
    sub_pct = sub_close.pct_change() * 100.0
    sub_vol = sub_df['volume'] if 'volume' in sub_df.columns else pd.Series(1, index=sub_df.index)
    
    if 'high' in sub_df.columns and 'low' in sub_df.columns:
        high = sub_df['high']
        low = sub_df['low']
        prev_close = sub_close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr_3 = float(tr.tail(3).mean()) if len(tr) >= 3 else 1.0
        atr_20 = float(tr.tail(20).mean()) if len(tr) >= 20 else 1.0
        hl_spread = (high - low).replace(0, np.nan)
        dcr = ((sub_close - low) / hl_spread).fillna(0.5).clip(0.0, 1.0)
    else:
        atr_3 = 1.0
        atr_20 = 1.0
        dcr = pd.Series(0.5, index=sub_df.index)
        
    avg_vol = sub_vol.rolling(20, min_periods=5).mean()
    
    step1_capitulation = bool(((sub_pct <= -1.2) & (sub_vol > avg_vol * 1.25)).any())
    step2_turnaround = bool((dcr.tail(15) >= 0.60).any() and (sub_pct.tail(15) > 0).any())
    step3_contraction = bool(atr_3 < atr_20)
    
    dist_10 = abs((c_now - ema10_now) / ema10_now) * 100.0 if ema10_now > 0 else 99.0
    dist_20 = abs((c_now - ema20_now) / ema20_now) * 100.0 if ema20_now > 0 else 99.0
    step4_clustering = bool(dist_10 <= 1.5 or dist_20 <= 1.5)
    
    step5_recaptured = bool(c_now >= ema20_now)
    step6_golden_cross = bool(ema10_now > ema20_now)
    
    ema20_slope = float((ema20.iloc[-1] - ema20.iloc[-3]) / 2.0) if len(ema20) >= 3 else 0.0
    step7_stress_test = bool(step6_golden_cross and step5_recaptured and ema20_slope > 0)
    
    if step7_stress_test:
        current_step_num = 7
        current_step_name = "Stress Test Held (Confirmed Trend)"
        step_badge_color = "#10b981"
        action_guidance = "Trend confirmed. Full pilot & sizing allowed on leading Stage 1/2 breakouts."
    elif step6_golden_cross:
        current_step_num = 6
        current_step_name = "10/20 EMA Crossback"
        step_badge_color = "#34d399"
        action_guidance = "Short-term momentum turned bullish. Look for high-volume base breakouts."
    elif step5_recaptured:
        current_step_num = 5
        current_step_name = "20 EMA Recapture (Tactical Buy)"
        step_badge_color = "#38bdf8"
        action_guidance = "Tactical supply-demand flip. Deploy initial 2.5%-5% pilot risk in top leaders."
    elif step4_clustering:
        current_step_num = 4
        current_step_name = "Moving Average Clustering"
        step_badge_color = "#f59e0b"
        action_guidance = "Price coiling near MAs. Prepare buy stop orders; wait for 20 EMA reclaim."
    elif step3_contraction:
        current_step_num = 3
        current_step_name = "Volatility Contraction"
        step_badge_color = "#fbbf24"
        action_guidance = "Range contracting. Watch for volume dry-up and subsequent expansion."
    elif step2_turnaround:
        current_step_num = 2
        current_step_name = "Turnaround Attempt"
        step_badge_color = "#fb923c"
        action_guidance = "First bounce off lows. Do not chase; wait for constructive digestion."
    elif step1_capitulation:
        current_step_num = 1
        current_step_name = "Capitulation Liquidation"
        step_badge_color = "#ef4444"
        action_guidance = "Selling climax in progress. Maintain 100% defense; observe potential support."
    else:
        current_step_num = 5 if step5_recaptured else 4
        current_step_name = "20 EMA Recaptured" if step5_recaptured else "Consolidating"
        step_badge_color = "#38bdf8" if step5_recaptured else "#94a3b8"
        action_guidance = "Monitor market posture."
        
    return {
        'secular_bull': secular_bull,
        'dist_200_sma_pct': round(dist_200, 2),
        'secular_label': secular_label,
        'secular_color': secular_color,
        'recapture_active': recapture_active,
        'recapture_fresh': recapture_fresh,
        'recapture_sessions_ago': recapture_sessions_ago,
        'current_step_num': current_step_num,
        'current_step_name': current_step_name,
        'step_badge_color': step_badge_color,
        'sequence_summary': f"Step {current_step_num}/7: {current_step_name}",
        'action_guidance': action_guidance,
        'steps_detail': {
            '1_capitulation': step1_capitulation,
            '2_turnaround': step2_turnaround,
            '3_contraction': step3_contraction,
            '4_clustering': step4_clustering,
            '5_recaptured': step5_recaptured,
            '6_golden_cross': step6_golden_cross,
            '7_stress_test': step7_stress_test
        }
    }

