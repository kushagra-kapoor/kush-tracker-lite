# portfolio_fetcher
import os
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
SERVICE_ACCOUNT_FILE = 'gspread_service_account.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
# Get URL from environment variable
SPREADSHEET_URL = os.getenv('SPREADSHEET_URL')
SHEET_NAME = 'RISK MANAGEMENT'
HEADER_ROW = 4  # header at this row (1-based index)


def fetch_portfolio_data():
    if not SPREADSHEET_URL:
        print("Error: SPREADSHEET_URL not found in .env file")
        return pd.DataFrame()
        
    import time
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"Authenticating with Google Sheets... (Attempt {attempt + 1}/{max_retries})")
            creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
            gc = gspread.authorize(creds)
        
            print(f"Opening spreadsheet: {SPREADSHEET_URL}")
            spreadsheet = gc.open_by_url(SPREADSHEET_URL)
            sheet = spreadsheet.worksheet(SHEET_NAME)
            print(f"Opened sheet: {sheet.title}")
        
            # Fetch raw values
            raw = sheet.get_all_values()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                print(f"Google Sheets API Error: {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Failed to fetch portfolio data after {max_retries} attempts. Google Sheets API might be down.")
                raise e
    headers = raw[HEADER_ROW - 1]
    data = raw[HEADER_ROW:]

    # Build DataFrame from row immediately after header row
    df = pd.DataFrame(data, columns=headers)


    # Select relevant columns
    cols_needed = ['NSE TICKER', 'Qty', 'Avg Buy Price', 'Peak Since Buy']
    df = df[cols_needed]

    # Trim whitespace and drop empty tickers
    df['NSE TICKER'] = df['NSE TICKER'].astype(str).str.strip()
    # Include all non-blank tickers now, whether they contain ':' or not
    df = df[df['NSE TICKER'] != '']

    # Extract raw ticker by removing any exchange prefix
    df['ticker'] = df['NSE TICKER'].str.replace(r'.*?:', '', regex=True).str.strip().str.upper()

    # Convert numeric columns
    df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
    df['Avg Buy Price'] = pd.to_numeric(df['Avg Buy Price'].str.replace('₹','').str.replace(',',''), errors='coerce')
    df['Peak Since Buy'] = pd.to_numeric(df['Peak Since Buy'].str.replace('₹','').str.replace(',',''), errors='coerce')

    # Drop rows missing essential data
    df = df.dropna(subset=['ticker', 'Qty', 'Avg Buy Price'])

    print("Cleaned tickers:", df['ticker'].tolist())
    print(f"Total tickers after cleaning: {len(df)}")
    return df


if __name__ == '__main__':
    fetch_portfolio_data()