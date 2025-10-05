import pandas as pd
import requests
import os

# --- Configuration ---
# You must set a User-Agent header for SEC requests
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "Your Name your.email@provider.com")
OUTPUT_CSV_PATH = "companies.csv"
# List of tickers you are interested in for your project
TARGET_TICKERS = [
    # Information Technology
    'MSFT', 'AAPL', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'AMD', 'QCOM', 'ADBE', 'ACN', 'INTC', 'IBM', 'CSCO', 'TXN', 'AMAT', 'ADP', 'NOW', 'INTU', 'ADI', 'LRCX', 'MU', 'PANW', 'SNPS', 'CDNS', 'ROP', 'KLAC', 'APH', 'TEL', 'MSI', 'FTNT', 'PAYX', 'ADSK', 'ANET', 'IT', 'CTSH', 'GLW', 'HPQ', 'KEYS', 'MCHP', 'CDW', 'FICO', 'HPE', 'PAYC', 'STX', 'ANSS', 'TER', 'TRMB', 'WDC', 'ZBRA', 'ENPH', 'DXC', 'JNPR', 'AKAM', 'NTAP', 'PTC', 'SEDG', 'TDY', 'FFIV', 'SWKS', 'QRVO', 'VRSN', 'FLT', 'JBL', 'JKHY',

    # Health Care
    'LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'ABT', 'PFE', 'DHR', 'AMGN', 'MDT', 'CI', 'ISRG', 'SYK', 'GILD', 'REGN', 'BSX', 'VRTX', 'MCK', 'BDX', 'CVS', 'HCA', 'HUM', 'ZTS', 'ELV', 'BIIB', 'IDXX', 'EW', 'COR', 'IQV', 'CZR', 'COO', 'CNC', 'MTD', 'A', 'WST', 'DXCM', 'HOLX', 'ALGN', 'CAH', 'BAX', 'STE', 'TECH', 'DGX', 'CME', 'CRL', 'BMRN', 'CTLT', 'WAT', 'RVTY', 'RMD', 'UHS', 'XRAY', 'BIO', 'CZR', 'DVA', 'MOH', 'HSIC', 'INCY', 'LH', 'OGN',

    # Financials
    'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'BLK', 'MS', 'AXP', 'SCHW', 'SPGI', 'C', 'CB', 'MMC', 'PNC', 'AON', 'ICE', 'USB', 'AIG', 'TRV', 'COF', 'TFC', 'MET', 'PRU', 'CME', 'PYPL', 'PGR', 'ALL', 'MCO', 'NDAQ', 'AJG', 'DFS', 'FITB', 'BK', 'WLTW', 'KEY', 'RF', 'HIG', 'L', 'HBAN', 'AMP', 'IVZ', 'WRB', 'CINF', 'NTRS', 'RJF', 'MTB', 'ETFC', 'STT', 'BEN', 'ZION', 'PFG', 'CBOE',

    # Consumer Discretionary
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'BKNG', 'SBUX', 'TJX', 'CMG', 'MAR', 'ORLY', 'HLT', 'GM', 'F', 'AZO', 'ROST', 'YUM', 'LEN', 'DHI', 'APTV', 'ULTA', 'PHM', 'CPRT', 'LVS', 'RCL', 'DRI', 'GRMN', 'KDP', 'EBAY', 'CCL', 'EXPE', 'POOL', 'WHR', 'WYNN', 'KMX', 'DPZ', 'BBY', 'GPC', 'Has', 'MGM', 'TGT', 'NVR', 'VFC', 'TSCO', 'WBA', 'LKQ', 'TPR', 'RL', 'NWL',

    # Communication Services
    'GOOGL', 'GOOG', 'META', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'WBD', 'EA', 'TTWO', 'OMC', 'IPG', 'LYV', 'FOXA', 'FOX', 'NWSA', 'NWS', 'DISCA', 'DISCK', 'PARA',

    # Industrials
    'CAT', 'UNP', 'UPS', 'RTX', 'BA', 'GE', 'HON', 'DE', 'LMT', 'ETN', 'ADP', 'WM', 'NSC', 'CSX', 'ITW', 'EMR', 'PCAR', 'ROP', 'PAYX', 'PH', 'JCI', 'FDX', 'TDG', 'CTAS', 'FAST', 'OTIS', 'AAL', 'DAL', 'LUV', 'UAL', 'CARR', 'GWW', 'RSG', 'IR', 'VRSK', 'XYL', 'DOV', 'LHX', 'IEX', 'AME', 'PWR', 'KSU', 'TXT', 'MAS', 'SNA', 'SWK', 'TT', 'WAB',

    # Consumer Staples
    'PG', 'COST', 'WMT', 'KO', 'PEP', 'MDLZ', 'MO', 'PM', 'CL', 'EL', 'GIS', 'ADM', 'SYY', 'KMB', 'MNST', 'TGT', 'DG', 'KR', 'KHC', 'STZ', 'CPB', 'HSY', 'CAG', 'SJM', 'CHD', 'MKC', 'HRL', 'TSN', 'LW', 'TAP',

    # Energy
    'XOM', 'CVX', 'SLB', 'COP', 'EOG', 'PXD', 'OXY', 'MPC', 'VLO', 'WMB', 'KMI', 'HES', 'PSX', 'HAL', 'DVN', 'OKE', 'BKR', 'APA', 'CTRA', 'FANG', 'MRO', 'TRGP', 'PPL',

    # Real Estate
    'PLD', 'AMT', 'EQIX', 'CCI', 'SPG', 'PSA', 'O', 'WELL', 'DLR', 'AVB', 'EQR', 'SBAC', 'VICI', 'WY', 'ARE', 'PEAK', 'EXR', 'ESS', 'MAA', 'VTR', 'KIM', 'REG', 'BXP', 'FRT', 'UDR', 'HST',

    # Materials
    'LIN', 'APD', 'SHW', 'ECL', 'NEM', 'DD', 'FCX', 'PPG', 'DOW', 'LYB', 'VMC', 'MLM', 'ALB', 'IFF', 'AVY', 'BALL', 'CF', 'IP', 'MOS', 'NUE', 'STLD', 'VMC', 'WRK', 'CE',

    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'PEG', 'ED', 'WEC', 'EIX', 'ES', 'AWK', 'PPL', 'FE', 'AEE', 'ETR', 'PCG', 'CNP', 'DTE', 'LNT', 'NRG', 'PNW'
]

def create_company_list_from_sec():
    """
    Downloads the official ticker-to-CIK mapping from the SEC
    and creates a filtered CSV file for the target companies.
    """
    print("Downloading the latest company ticker data from the SEC...")

    # 1. Download the JSON data
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {'User-Agent': SEC_USER_AGENT}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        company_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return

    # 2. Process the data into a list of dictionaries
    # The JSON is a dictionary of {cik: {ticker, title, ...}}, so we iterate over the values
    all_companies = [
        {
            "cik": str(details['cik_str']),
            "ticker": details['ticker'],
            "company_name": details['title']
        }
        for details in company_data.values()
    ]

    # 3. Convert to a Pandas DataFrame
    df = pd.DataFrame(all_companies)

    # 4. Filter for only the tickers you care about
    filtered_df = df[df['ticker'].isin(TARGET_TICKERS)]

    # Ensure the columns are in the desired order
    final_df = filtered_df[['ticker', 'company_name', 'cik']]

    # 5. Save to CSV
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"Successfully created '{OUTPUT_CSV_PATH}' with {len(final_df)} companies.")
    print("File content:")
    print(final_df)

if __name__ == "__main__":
    create_company_list_from_sec()