import os
import pandas as pd
from datetime import datetime
from risk_metrics import calculate_rolling_portfolio_metrics

DATA_FILE = 'data/equity_curve.csv'

def initialize_equity_log():
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=['Date', 'TotalValue'])
        df.to_csv(DATA_FILE, index=False)

def log_equity(total_value):
    """Log today's equity, updating if already exists for today."""
    initialize_equity_log()
    df = pd.read_csv(DATA_FILE)
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    if not df.empty and df.iloc[-1]['Date'] == today_str:
        # Update today's value
        df.loc[df.index[-1], 'TotalValue'] = total_value
    else:
        # Append new day
        new_row = pd.DataFrame([{'Date': today_str, 'TotalValue': total_value}])
        df = pd.concat([df, new_row], ignore_index=True)
        
    df.to_csv(DATA_FILE, index=False)
    return df

def get_equity_drawdown(total_value):
    """
    Log the current value and calculate the peak drawdown, rolling Sortino (63D, 126D), and Calmar ratio.
    Returns: dict with peak, drawdown_pct, cash_target, directive, Sortino, and Calmar metrics.
    """
    df = log_equity(total_value)
    
    if df.empty:
        peak = total_value
    else:
        peak = max(df['TotalValue'].max(), total_value)
        
    drawdown_pct = 0.0
    if peak > 0:
        drawdown_pct = ((peak - total_value) / peak) * 100.0
        
    # Determine Cash Target & Action based on institutional drawdown bands
    if drawdown_pct >= 15.0:
        target_cash = 100
        action = "100% Cash / Capital Preservation"
        color = "#ef4444"
    elif drawdown_pct >= 12.0:
        target_cash = 80
        action = "80% Cash / Extreme Defense"
        color = "#ea580c"
    elif drawdown_pct >= 10.0:
        target_cash = 50
        action = "50% Cash / Cut Losers Heavily"
        color = "#f97316"
    elif drawdown_pct >= 5.0:
        target_cash = 5
        action = "No New Buys / Cut Losers"
        color = "#eab308"
    else:
        target_cash = 0
        action = "Progressive Exposure (Business as usual)"
        color = "#10b981"
        
    # Calculate rolling risk-adjusted metrics from equity curve history
    port_risk = calculate_rolling_portfolio_metrics(df)
        
    return {
        'current': total_value,
        'peak': peak,
        'drawdown_pct': drawdown_pct,
        'target_cash': target_cash,
        'action': action,
        'color': color,
        'sortino_63d': port_risk.get('sortino_63d', 0.0),
        'sortino_126d': port_risk.get('sortino_126d', 0.0),
        'calmar': port_risk.get('calmar', 0.0),
        'downside_dev_63d': port_risk.get('downside_dev_63d', 0.0),
        'cagr': port_risk.get('cagr', 0.0),
        'history_count': len(df)
    }

def get_elder_circuit_breaker(current_equity: float = None) -> dict:
    """
    Dr. Alexander Elder's 6% Monthly Drawdown Circuit Breaker & 2% Rule (The New Trading for a Living p. 248):
    
    1. Measures Month-to-Date (MTD) Drawdown relative to the highest peak achieved during the current calendar month
       and starting equity on the 1st of the month.
    2. Thresholds:
       - Normal (< 4.0% Drawdown): Full risk permitted (0.50% - 1.0% per trade).
       - Caution (4.0% <= Drawdown < 6.0%): Elder Warning. Reduce risk to 0.20% per trade.
       - Circuit Breaker Active (>= 6.0% Drawdown): Mandatory halt on all new buy executions for the remainder of the month.
    """
    initialize_equity_log()
    df = pd.read_csv(DATA_FILE)
    
    now = datetime.now()
    cur_year = now.year
    cur_month = now.month
    
    if df.empty:
        curr = current_equity if current_equity is not None else 1000000.0
        return {
            'status': "NORMAL",
            'trading_permitted': True,
            'caution_active': False,
            'circuit_breaker_fired': False,
            'current_equity': curr,
            'month_start_equity': curr,
            'mtd_peak_equity': curr,
            'mtd_drawdown_pct': 0.0,
            'mtd_pnl_pct': 0.0,
            'max_allowed_loss_dollars': curr * 0.06,
            'remaining_cushion_dollars': curr * 0.06,
            'action': "Normal Trading Permitted (Standard 0.5%–1.0% Risk per trade)",
            'color': "#10b981",
            'badge': "🟢 Normal Risk"
        }
        
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Current equity
    if current_equity is not None:
        curr = float(current_equity)
    else:
        curr = float(df['TotalValue'].iloc[-1])
        
    # Month to Date rows
    mtd_mask = (df['Date'].dt.year == cur_year) & (df['Date'].dt.month == cur_month)
    mtd_df = df[mtd_mask]
    
    # Month starting balance
    prior_rows = df[~mtd_mask & (df['Date'] < datetime(cur_year, cur_month, 1))]
    if not prior_rows.empty:
        month_start_equity = float(prior_rows['TotalValue'].iloc[-1])
    elif not mtd_df.empty:
        month_start_equity = float(mtd_df['TotalValue'].iloc[0])
    else:
        month_start_equity = curr
        
    # MTD peak equity
    if not mtd_df.empty:
        mtd_peak_equity = max(float(mtd_df['TotalValue'].max()), curr, month_start_equity)
    else:
        mtd_peak_equity = max(curr, month_start_equity)
        
    # MTD Drawdown % from MTD Peak
    if mtd_peak_equity > 0:
        mtd_drawdown_pct = ((mtd_peak_equity - curr) / mtd_peak_equity) * 100.0
    else:
        mtd_drawdown_pct = 0.0
        
    # MTD Net PnL %
    if month_start_equity > 0:
        mtd_pnl_pct = ((curr - month_start_equity) / month_start_equity) * 100.0
    else:
        mtd_pnl_pct = 0.0
        
    # Circuit breaker levels (Elder 6% limit is 6% of month peak)
    max_loss_budget = mtd_peak_equity * 0.06
    current_loss_dollars = max(0.0, mtd_peak_equity - curr)
    remaining_cushion = max(0.0, max_loss_budget - current_loss_dollars)
    
    if mtd_drawdown_pct >= 6.0:
        status = "HALTED"
        trading_permitted = False
        caution_active = True
        circuit_breaker_fired = True
        action = "🚨 ELDER 6% CIRCUIT BREAKER TRIGGERED: Trading Halted for Remainder of Month. Move to 100% Cash / No New Positions."
        color = "#ef4444"
        badge = "🔴 Trading Halted (6% Monthly Freeze)"
    elif mtd_drawdown_pct >= 4.0:
        status = "CAUTION"
        trading_permitted = True
        caution_active = True
        circuit_breaker_fired = False
        action = "⚠️ ELDER 4% CAUTION TRIGGERED: Drawdown exceeds 4%. Scale risk per trade down to 0.20% maximum."
        color = "#f59e0b"
        badge = "🟡 Caution (Scale to 0.2% Risk)"
    else:
        status = "NORMAL"
        trading_permitted = True
        caution_active = False
        circuit_breaker_fired = False
        action = "🟢 Normal Risk Permitted (Elder 2% Rule: Risk 0.5%–1.0% NAV per trade)."
        color = "#10b981"
        badge = "🟢 Normal Risk"
        
    return {
        'status': status,
        'trading_permitted': trading_permitted,
        'caution_active': caution_active,
        'circuit_breaker_fired': circuit_breaker_fired,
        'current_equity': round(curr, 2),
        'month_start_equity': round(month_start_equity, 2),
        'mtd_peak_equity': round(mtd_peak_equity, 2),
        'mtd_drawdown_pct': round(mtd_drawdown_pct, 2),
        'mtd_pnl_pct': round(mtd_pnl_pct, 2),
        'max_allowed_loss_dollars': round(max_loss_budget, 2),
        'remaining_cushion_dollars': round(remaining_cushion, 2),
        'action': action,
        'color': color,
        'badge': badge
    }

if __name__ == '__main__':
    # Test
    res = get_equity_drawdown(1000000)
    print("Equity Drawdown calculated successfully")
    elder = get_elder_circuit_breaker(940000)
    print("Elder Status:", elder['status'], "MTD Drawdown:", elder['mtd_drawdown_pct'], "%")
