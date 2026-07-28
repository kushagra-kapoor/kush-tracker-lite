import yfinance as yf
import pandas as pd
import numpy as np

# Global Asset Classes mapping
ASSETS = {
    'Nifty 50 (IN Large)': '^NSEI',
    'Nifty 500 (IN Broad)': '^CRSLDX',
    'Nasdaq 100 (US Tech)': 'QQQ',
    'S&P 500 (US Broad)': '^GSPC',
    'Hang Seng (China)': '^HSI',
    'Nifty200Mom30 (Factor)': 'MOM30IETF.NS',
    'Gold (Safe Haven)': 'GOLDBEES.NS',
    'Silver (Precious)': 'SILVERBEES.NS',
    'FANG+ (Global Tech)': 'MAFANG.NS'
}

def fetch_asset_returns(benchmark='Nifty 50 (IN Large)'):
    """
    Fetches 2 years of daily data for global assets to ensure 1Y return works,
    calculates absolute returns, and relative strength (vs Benchmark).
    """
    results = []
    
    # Pre-fetch all data to optimize
    tickers = list(ASSETS.values())
    try:
        data = yf.download(tickers, period='2y', group_by='ticker', threads=True, progress=False)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
        
    for name, ticker in ASSETS.items():
        if len(tickers) == 1:
            df = data
        else:
            df = data[ticker] if ticker in data else None
            
        if df is None or df.empty or 'Close' not in df:
            continue
            
        df = df.dropna(subset=['Close'])
        if len(df) < 5:
            continue
            
        # Absolute Returns
        c = df['Close'].iloc[-1]
        
        def safe_ret(days):
            if len(df) >= days:
                past = df['Close'].iloc[-days]
                return ((c / past) - 1) * 100
            return np.nan
            
        ret_1w = safe_ret(5)
        ret_2w = safe_ret(10)
        ret_1m = safe_ret(21)
        ret_3m = safe_ret(63)
        ret_6m = safe_ret(126)
        ret_1y = safe_ret(252)
        
        display_name = f"{name} [{ticker}]"
        
        results.append({
            'Asset': display_name,
            'Ret_1W': ret_1w,
            'Ret_2W': ret_2w,
            'Ret_1M': ret_1m,
            'Ret_3M': ret_3m,
            'Ret_6M': ret_6m,
            'Ret_1Y': ret_1y
        })
        
    # Convert to DataFrame
    res_df = pd.DataFrame(results)
    if res_df.empty:
        return res_df
        
    # Calculate Relative Strength vs Benchmark (Outperformance in %)
    bench_data = res_df[res_df['Asset'].str.startswith(benchmark)]
    if not bench_data.empty:
        b_1m = bench_data.iloc[0]['Ret_1M']
        b_3m = bench_data.iloc[0]['Ret_3M']
        b_6m = bench_data.iloc[0]['Ret_6M']
        
        res_df['RS_1M'] = res_df['Ret_1M'] - b_1m
        res_df['RS_3M'] = res_df['Ret_3M'] - b_3m
        res_df['RS_6M'] = res_df['Ret_6M'] - b_6m
    else:
        res_df['RS_1M'] = np.nan
        res_df['RS_3M'] = np.nan
        res_df['RS_6M'] = np.nan
        
    # Sort by 3M Return by default (momentum window)
    res_df = res_df.sort_values('Ret_3M', ascending=False)
    
    return res_df

if __name__ == '__main__':
    df = fetch_asset_returns()
    print(df.to_string())
