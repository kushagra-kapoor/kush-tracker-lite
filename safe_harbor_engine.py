import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from config import BENCHMARK_TICKER, RS_WEIGHTS

SAFE_HARBOR_TICKERS = {
    'GOLDBEES.NS': 'Gold (Safe Haven)',
    'SILVERBEES.NS': 'Silver (Precious Metal)',
    'MON100.NS': 'Nasdaq 100 (Geographical Diversification)',
    'LIQUIDCASE.NS': 'Liquid Fund (Cash Parking)'
}

def calculate_simple_returns(close_prices: pd.Series, periods: int) -> float:
    if len(close_prices) < periods + 1:
        return None
    current_price = close_prices.iloc[-1]
    past_price = close_prices.iloc[-(periods + 1)]
    if pd.isna(past_price) or past_price == 0:
        return None
    return (current_price / past_price) - 1

def evaluate_safe_harbor(days=150):
    """
    Evaluates the momentum of Safe Harbor assets against the Nifty 500 benchmark.
    Returns recommendations for capital rotation during weak market regimes.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # 1. Fetch Benchmark
    try:
        benchmark = yf.Ticker(BENCHMARK_TICKER)
        bench_df = benchmark.history(start=start_date, end=end_date)
        b1 = calculate_simple_returns(bench_df['Close'], 21)
        b3 = calculate_simple_returns(bench_df['Close'], 63)
    except Exception:
        return []

    if b1 is None or b3 is None:
        return []

    # 2. Fetch Safe Harbor Assets
    results = []
    
    for ticker, name in SAFE_HARBOR_TICKERS.items():
        try:
            asset = yf.Ticker(ticker)
            df = asset.history(start=start_date, end=end_date)
            r1 = calculate_simple_returns(df['Close'], 21)
            r3 = calculate_simple_returns(df['Close'], 63)
            
            if r1 is None or r3 is None:
                continue
                
            # Calculate absolute and relative performance
            rel_1m = r1 - b1
            rel_3m = r3 - b3
            
            # Simple weighted momentum score prioritizing 1M over 3M for nimble rotation
            score = (r1 * 0.7) + (r3 * 0.3)
            rel_score = (rel_1m * 0.7) + (rel_3m * 0.3)
            
            # Trend Check (Is it currently above 21 and 50 EMA?)
            close = df['Close'].iloc[-1]
            ema21 = df['Close'].ewm(span=21, adjust=False).mean().iloc[-1]
            sma50 = df['Close'].rolling(window=50).mean().iloc[-1]
            uptrend = close > ema21 and close > sma50
            
            status = "Wait"
            # Recommendations logic
            if ticker == 'LIQUIDCASE.NS':
                status = "Excellent Parking" # Liquid case is always cash equivalent
            elif uptrend and rel_score > 0 and score > 0:
                status = "Strong Rotation Candidate"
            elif uptrend and score > 0:
                status = "Defensive Hold"
            else:
                status = "Weak (Avoid)"
                
            results.append({
                'Ticker': ticker.replace('.NS', ''),
                'Asset': name,
                'Status': status,
                '1M_Return': round(r1 * 100, 2),
                '3M_Return': round(r3 * 100, 2),
                'Rel_Strength_Score': round(rel_score * 100, 2),
                'Trend': 'Up' if uptrend else 'Down'
            })
            
        except Exception as e:
            continue
            
    # Sort by Relative Strength Score descending (excluding Liquid fund which artificially skews)
    results = sorted(results, key=lambda x: x['Rel_Strength_Score'], reverse=True)
    return results

