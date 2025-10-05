import os
import requests
import pandas as pd
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import yfinance as yf 

# Load environment variables
load_dotenv()
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")

# --- Configuration ---
COMPANIES_CSV_PATH = "companies.csv"
FINANCIALS_DIR = "data/structured/financials"
PRICES_DIR = "data/structured/prices"
FILINGS_10K_DIR = "data/unstructured/10k"

# Create directories if they don't exist
os.makedirs(FINANCIALS_DIR, exist_ok=True)
os.makedirs(PRICES_DIR, exist_ok=True)
os.makedirs(FILINGS_10K_DIR, exist_ok=True)


def fetch_financial_statements(ticker: str):
    """Fetches annual income statements using yfinance."""
    print(f"Fetching financial statements for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        income_stmt = stock.income_stmt

        if income_stmt.empty:
            print(f" -> No financial data found for {ticker}")
            return

        # Convert DataFrame to a JSON format similar to the original API
        # Reset index to make 'Year' a column, then transpose and format
        data = income_stmt.transpose()
        data.index.name = 'date'
        data = data.reset_index()
        data['date'] = data['date'].astype(str) # Convert timestamp to string
        records = data.to_dict('records')

        with open(os.path.join(FINANCIALS_DIR, f"{ticker}_financials.json"), 'w') as f:
            json.dump(records, f, indent=4)
        print(f" -> Saved financials for {ticker}")
    except Exception as e:
        print(f"Error fetching financials for {ticker}: {e}")

def fetch_stock_prices(ticker: str):
    """Fetches the last 5 years of daily stock prices using yfinance."""
    print(f"Fetching stock prices for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        # Get 5 years of historical market data
        hist = stock.history(period="5y")

        if hist.empty:
            print(f" -> No price data found for {ticker}")
            return

        hist.to_csv(os.path.join(PRICES_DIR, f"{ticker}_prices.csv"))
        print(f" -> Saved prices for {ticker}")
    except Exception as e:
        print(f"Error fetching prices for {ticker}: {e}")

# --- SEC function (remains the same) ---

def fetch_10k_filings(ticker: str, cik: str):
    """Fetches the last 5 annual 10-K filings from the SEC EDGAR database."""
    print(f"Fetching 10-K filings for {ticker} (CIK: {cik})...")
    headers = {'User-Agent': SEC_USER_AGENT}

    submissions_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    try:
        response = requests.get(submissions_url, headers=headers)
        response.raise_for_status()
        submissions = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching submission history for {ticker}: {e}")
        return

    filing_count = 0
    recent_filings = submissions['filings']['recent']

    for i in range(len(recent_filings['form'])):
        if filing_count >= 5:
            break
        if recent_filings['form'][i] == '10-K':
            accession_no = recent_filings['accessionNumber'][i].replace('-', '')
            primary_doc_name = recent_filings['primaryDocument'][i]
            filing_date = recent_filings['filingDate'][i]
            year = filing_date.split('-')[0]

            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no}/{primary_doc_name}"

            print(f"  -> Downloading 10-K for {year}...")
            try:
                time.sleep(0.2)
                doc_response = requests.get(doc_url, headers=headers)
                doc_response.raise_for_status()

                file_path = os.path.join(FILINGS_10K_DIR, f"{ticker}_10K_{year}.html")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(doc_response.text)

                filing_count += 1
            except requests.exceptions.RequestException as e:
                print(f"    Error downloading filing {doc_url}: {e}")

    print(f" -> Finished fetching filings for {ticker}")


if __name__ == "__main__":
    companies_df = pd.read_csv(COMPANIES_CSV_PATH)

    # Filter out tickers with dots or dashes that yfinance might not handle well
    companies_df = companies_df[~companies_df['ticker'].str.contains('\.|\-')]

    for index, row in tqdm(companies_df.iterrows(), total=companies_df.shape[0], desc="Processing Companies"):
        ticker = row['ticker']
        cik = str(row['cik'])

        # --- Fetch and Save Data ---
        fetch_financial_statements(ticker)
        fetch_stock_prices(ticker)
        fetch_10k_filings(ticker, cik)

        # Rate limit to be respectful to APIs
        time.sleep(0.5)

    print("\nData fetching complete. Check the 'data' directory.")