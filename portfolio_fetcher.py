import pandas as pd

def fetch_portfolio_data():
    return pd.DataFrame(columns=['ticker', 'Qty', 'Avg Buy Price', 'Total Value'])
