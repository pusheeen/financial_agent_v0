#!/usr/bin/env python3
"""
Data fetching script for the financial agent project.
Fetches:
1. Company information from SEC (ticker, company name, CIK)
2. CEO profiles and information
3. Financial statements
4. Stock prices
5. 10-K filings
"""

import os
import requests
import pandas as pd
import json
import time
import csv
import re
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import yfinance as yf
from bs4 import BeautifulSoup
from typing import Dict, List, Any
import logging

# Optional imports for Reddit
try:
    import praw
    import feedparser
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False
    print("Warning: Reddit scraping not available. Install praw, feedparser, and vaderSentiment to enable.")

# Load environment variables
load_dotenv()
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "Your Name your.email@provider.com")

# Import target tickers
from target_tickers import TARGET_TICKERS

# --- Configuration ---
DATA_DIR = "data"
COMPANIES_CSV_PATH = "data/companies.csv"
FINANCIALS_DIR = "data/structured/financials"
PRICES_DIR = "data/structured/prices"
FILINGS_10K_DIR = "data/unstructured/10k"
CEO_REPORTS_DIR = "data/reports"
REDDIT_DATA_DIR = "data/unstructured/reddit"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FINANCIALS_DIR, exist_ok=True)
os.makedirs(PRICES_DIR, exist_ok=True)
os.makedirs(FILINGS_10K_DIR, exist_ok=True)
os.makedirs(CEO_REPORTS_DIR, exist_ok=True)
os.makedirs(REDDIT_DATA_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Static CEO database cache (to avoid repeated web searches for known CEOs)
# This will be populated from data/reports/ceo_summary_*.csv if available
_CEO_CACHE = {}


def fetch_company_info_from_sec() -> pd.DataFrame:
    """
    Fetches company information (ticker, company name, CIK) from SEC
    for all tickers in TARGET_TICKERS.
    """
    logger.info("Fetching company information from SEC...")

    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {'User-Agent': SEC_USER_AGENT}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        company_data = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading SEC data: {e}")
        return pd.DataFrame()

    # Process the data into a list of dictionaries
    all_companies = [
        {
            "cik": str(details['cik_str']),
            "ticker": details['ticker'],
            "company_name": details['title']
        }
        for details in company_data.values()
    ]

    # Convert to DataFrame
    df = pd.DataFrame(all_companies)

    # Filter for only the tickers in TARGET_TICKERS
    filtered_df = df[df['ticker'].isin(TARGET_TICKERS)]

    # Ensure the columns are in the desired order
    final_df = filtered_df[['ticker', 'company_name', 'cik']]

    # Save to CSV
    final_df.to_csv(COMPANIES_CSV_PATH, index=False)

    logger.info(f"Successfully fetched {len(final_df)} companies from SEC")
    logger.info(f"Saved to {COMPANIES_CSV_PATH}")

    return final_df


def search_linkedin_profile(ceo_name: str, company_name: str) -> Dict[str, Any]:
    """
    Search for LinkedIn profile URL and extract profile data using Google search.
    """
    linkedin_data = {
        "linkedin_url": "Not found",
        "education": "Not found",
        "past_experience": [],
        "career_highlights": [],
        "start_date": "Not found",
        "tenure_duration": "Not found"
    }

    if ceo_name == "Not found" or not ceo_name:
        return linkedin_data

    try:
        # Clean CEO name (remove titles like Mr., Dr., etc.)
        clean_name = re.sub(r'\b(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s*', '', ceo_name).strip()

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Search for LinkedIn profile
        search_query = f"{clean_name} {company_name} CEO LinkedIn"
        search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"

        response = requests.get(search_url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find LinkedIn URL in search results
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'linkedin.com/in/' in href:
                    # Extract clean LinkedIn URL
                    match = re.search(r'(https://[a-z]+\.linkedin\.com/in/[^&\s]+)', href)
                    if match:
                        linkedin_url = match.group(1)
                        # Remove trailing garbage
                        linkedin_url = re.sub(r'["\'].*$', '', linkedin_url)
                        linkedin_data["linkedin_url"] = linkedin_url
                        logger.info(f"Found LinkedIn URL: {linkedin_url}")
                        break

        # If LinkedIn URL found, try to scrape profile page for additional info
        if linkedin_data["linkedin_url"] != "Not found":
            time.sleep(2)  # Be respectful
            try:
                profile_response = requests.get(linkedin_data["linkedin_url"], headers=headers, timeout=15)
                if profile_response.status_code == 200:
                    profile_soup = BeautifulSoup(profile_response.content, 'html.parser')
                    page_text = profile_soup.get_text()

                    # Try to extract education (basic pattern matching)
                    education_patterns = [
                        r'(University|College|Institute|School) of [A-Z][a-z\s]+',
                        r'(Harvard|Stanford|MIT|Yale|Princeton|Berkeley|Cambridge|Oxford)[^,\.]*',
                        r'(Bachelor|Master|MBA|PhD|B\.S\.|M\.S\.|B\.A\.|M\.A\.)[^,\.]{0,50}'
                    ]

                    for pattern in education_patterns:
                        matches = re.findall(pattern, page_text, re.IGNORECASE)
                        if matches:
                            # Take first 2 matches to avoid too much data
                            linkedin_data["education"] = ', '.join(matches[:2])
                            break

                    # Try to extract experience (look for company names and positions)
                    experience_pattern = r'(Chief|Senior|Vice President|Director|Manager|Head of)[^\.]{0,100}'
                    experience_matches = re.findall(experience_pattern, page_text)
                    if experience_matches:
                        linkedin_data["past_experience"] = experience_matches[:5]  # Top 5

                    # Try to find start date / tenure
                    year_patterns = re.findall(r'(20\d{2})\s*-\s*Present', page_text, re.IGNORECASE)
                    if year_patterns:
                        start_year = int(year_patterns[0])
                        current_year = datetime.now().year
                        years = current_year - start_year
                        linkedin_data["start_date"] = str(start_year)
                        linkedin_data["tenure_duration"] = f"{years} year{'s' if years != 1 else ''}"

            except Exception as e:
                logger.warning(f"Could not scrape LinkedIn profile: {e}")

        time.sleep(2)  # Rate limiting

    except Exception as e:
        logger.warning(f"LinkedIn search failed: {e}")

    return linkedin_data


def query_ceo_info_by_ticker(ticker: str, company_name: str) -> Dict[str, Any]:
    """
    Queries CEO information by company ticker using yfinance and LinkedIn.
    """
    try:
        if not ticker or not isinstance(ticker, str):
            return {"error": "Invalid ticker"}

        ticker = ticker.upper().strip()
        logger.info(f"Fetching CEO data for {ticker}")

        ceo_data = {
            "ticker": ticker,
            "company_name": company_name,
            "ceo_name": "Not found",
            "ceo_title": "Not found",
            "tenure_duration": "Not found",
            "start_date": "Not found",
            "linkedin_url": "Not found",
            "source": "yfinance + LinkedIn",
            "past_experience": [],
            "education": "Not found",
            "career_highlights": [],
            "fetch_timestamp": datetime.now().isoformat()
        }

        # Try to get officer information from yfinance
        try:
            stock = yf.Ticker(ticker)

            # Get company info which includes officers
            info = stock.info

            # Try to get officers data
            if hasattr(stock, 'get_officers') and callable(stock.get_officers):
                officers = stock.get_officers()
                if officers is not None and not officers.empty:
                    # Look for CEO in the officers list
                    for idx, officer in officers.iterrows():
                        title = officer.get('title', '').lower()
                        if 'chief executive officer' in title or 'ceo' in title or title == 'ceo':
                            ceo_data["ceo_name"] = officer.get('name', 'Not found')
                            ceo_data["ceo_title"] = officer.get('title', 'Chief Executive Officer')

                            # Try to get age/tenure info
                            if 'age' in officer:
                                ceo_data["age"] = officer.get('age')

                            # Try to get pay info
                            if 'totalPay' in officer:
                                ceo_data["total_pay"] = officer.get('totalPay')

                            logger.info(f"Found CEO {ceo_data['ceo_name']} via yfinance officers")
                            break

            # Fallback: Try to get CEO from company info
            if ceo_data["ceo_name"] == "Not found":
                # Some companies have companyOfficers in info
                if 'companyOfficers' in info and info['companyOfficers']:
                    for officer in info['companyOfficers']:
                        title = officer.get('title', '').lower()
                        if 'chief executive officer' in title or 'ceo' in title:
                            ceo_data["ceo_name"] = officer.get('name', 'Not found')
                            ceo_data["ceo_title"] = officer.get('title', 'Chief Executive Officer')

                            # Get additional info if available
                            if 'age' in officer:
                                ceo_data["age"] = officer['age']
                            if 'yearBorn' in officer:
                                ceo_data["year_born"] = officer['yearBorn']
                            if 'totalPay' in officer:
                                ceo_data["total_pay"] = officer['totalPay']

                            logger.info(f"Found CEO {ceo_data['ceo_name']} via yfinance info")
                            break

            # If still not found, try alternative field names
            if ceo_data["ceo_name"] == "Not found":
                # Check for CEO in various info fields
                for field in ['ceo', 'CEO', 'chiefExecutiveOfficer']:
                    if field in info and info[field]:
                        ceo_data["ceo_name"] = info[field]
                        ceo_data["ceo_title"] = "Chief Executive Officer"
                        logger.info(f"Found CEO {ceo_data['ceo_name']} via yfinance {field} field")
                        break

        except Exception as e:
            logger.warning(f"yfinance lookup failed for {ticker}: {e}")

        # Enrich with LinkedIn data
        if ceo_data["ceo_name"] != "Not found":
            logger.info(f"Searching LinkedIn for {ceo_data['ceo_name']}")
            linkedin_data = search_linkedin_profile(ceo_data["ceo_name"], company_name)

            # Update CEO data with LinkedIn info (only if found)
            if linkedin_data["linkedin_url"] != "Not found":
                ceo_data["linkedin_url"] = linkedin_data["linkedin_url"]

            if linkedin_data["education"] != "Not found":
                ceo_data["education"] = linkedin_data["education"]

            if linkedin_data["past_experience"]:
                ceo_data["past_experience"] = linkedin_data["past_experience"]

            if linkedin_data["start_date"] != "Not found":
                ceo_data["start_date"] = linkedin_data["start_date"]
                ceo_data["tenure_duration"] = linkedin_data["tenure_duration"]

        return {
            "success": True,
            "ceo_data": ceo_data,
            "note": "CEO information from yfinance and LinkedIn"
        }

    except Exception as e:
        logger.error(f"Error fetching CEO for {ticker}: {e}")
        return {"error": f"Error: {str(e)}"}


def fetch_ceo_profiles(companies_df: pd.DataFrame):
    """
    Fetches CEO profiles for all companies in the dataframe.
    """
    logger.info("=" * 60)
    logger.info("Starting CEO profile batch processing")
    logger.info("=" * 60)

    results = []
    successful_count = 0
    failed_count = 0

    for index, row in companies_df.iterrows():
        ticker = row['ticker'].upper()
        company_name = row['company_name']

        logger.info(f"Processing CEO for {index+1}/{len(companies_df)}: {ticker} ({company_name})")

        try:
            # Fetch CEO data
            result = query_ceo_info_by_ticker(ticker, company_name)
            results.append(result)

            if result.get('success', False):
                successful_count += 1
                ceo_name = result.get('ceo_data', {}).get('ceo_name', 'Not found')
                logger.info(f"✅ {ticker}: Found CEO {ceo_name}")
            else:
                failed_count += 1
                error = result.get('error', 'Unknown error')
                logger.warning(f"❌ {ticker}: {error}")

        except Exception as e:
            logger.error(f"❌ {ticker}: Unexpected error - {e}")
            results.append({
                'success': False,
                'ticker': ticker,
                'error': f"Unexpected error: {str(e)}"
            })
            failed_count += 1

        # Delay between requests (longer due to Google search + LinkedIn scraping)
        if index < len(companies_df) - 1:
            time.sleep(5)

    # Create summary reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON report
    json_report = {
        "batch_info": {
            "timestamp": datetime.now().isoformat(),
            "total_companies": len(companies_df),
            "successful_fetches": successful_count,
            "failed_fetches": failed_count
        },
        "results": results
    }

    json_filename = os.path.join(CEO_REPORTS_DIR, f"ceo_batch_report_{timestamp}.json")
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)

    # CSV report
    csv_filename = os.path.join(CEO_REPORTS_DIR, f"ceo_summary_{timestamp}.csv")
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['ticker', 'company_name', 'ceo_name', 'ceo_title', 'tenure_duration',
                     'start_date', 'education', 'num_past_roles', 'num_highlights',
                     'linkedin_url', 'source', 'fetch_timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            if result.get('success', False):
                ceo_data = result.get('ceo_data', {})
                writer.writerow({
                    'ticker': ceo_data.get('ticker', 'Unknown'),
                    'company_name': ceo_data.get('company_name', 'Unknown'),
                    'ceo_name': ceo_data.get('ceo_name', 'Not found'),
                    'ceo_title': ceo_data.get('ceo_title', 'Not found'),
                    'tenure_duration': ceo_data.get('tenure_duration', 'Not found'),
                    'start_date': ceo_data.get('start_date', 'Not found'),
                    'education': ceo_data.get('education', 'Not found'),
                    'num_past_roles': len(ceo_data.get('past_experience', [])),
                    'num_highlights': len(ceo_data.get('career_highlights', [])),
                    'linkedin_url': ceo_data.get('linkedin_url', 'Not found'),
                    'source': ceo_data.get('source', 'Unknown'),
                    'fetch_timestamp': ceo_data.get('fetch_timestamp', 'Unknown')
                })
            else:
                writer.writerow({
                    'ticker': result.get('ticker', 'Unknown'),
                    'company_name': 'Error',
                    'ceo_name': 'Error',
                    'ceo_title': 'Error',
                    'tenure_duration': 'Error',
                    'start_date': 'Error',
                    'education': 'Error',
                    'num_past_roles': 0,
                    'num_highlights': 0,
                    'linkedin_url': 'Error',
                    'source': 'Error',
                    'fetch_timestamp': datetime.now().isoformat()
                })

    # Final summary
    logger.info("=" * 60)
    logger.info("CEO BATCH PROCESSING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total companies: {len(companies_df)}")
    logger.info(f"Successful: {successful_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Success rate: {successful_count/len(companies_df)*100:.1f}%")
    logger.info(f"JSON report: {json_filename}")
    logger.info(f"CSV report: {csv_filename}")


def fetch_financial_statements(ticker: str):
    """Fetches annual income statements using yfinance."""
    logger.info(f"Fetching financial statements for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        income_stmt = stock.income_stmt

        if income_stmt.empty:
            logger.warning(f"No financial data found for {ticker}")
            return

        # Convert DataFrame to JSON format
        data = income_stmt.transpose()
        data.index.name = 'date'
        data = data.reset_index()
        data['date'] = data['date'].astype(str)
        records = data.to_dict('records')

        with open(os.path.join(FINANCIALS_DIR, f"{ticker}_financials.json"), 'w') as f:
            json.dump(records, f, indent=4)
        logger.info(f"✅ Saved financials for {ticker}")
    except Exception as e:
        logger.error(f"Error fetching financials for {ticker}: {e}")


def fetch_stock_prices(ticker: str):
    """Fetches the last 5 years of daily stock prices using yfinance."""
    logger.info(f"Fetching stock prices for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")

        if hist.empty:
            logger.warning(f"No price data found for {ticker}")
            return

        hist.to_csv(os.path.join(PRICES_DIR, f"{ticker}_prices.csv"))
        logger.info(f"✅ Saved prices for {ticker}")
    except Exception as e:
        logger.error(f"Error fetching prices for {ticker}: {e}")


def fetch_10k_filings(ticker: str, cik: str):
    """Fetches the last 5 annual 10-K filings from the SEC EDGAR database."""
    logger.info(f"Fetching 10-K filings for {ticker} (CIK: {cik})...")
    headers = {'User-Agent': SEC_USER_AGENT}

    submissions_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    try:
        response = requests.get(submissions_url, headers=headers)
        response.raise_for_status()
        submissions = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching submission history for {ticker}: {e}")
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

            logger.info(f"  Downloading 10-K for {year}...")
            try:
                time.sleep(0.2)
                doc_response = requests.get(doc_url, headers=headers)
                doc_response.raise_for_status()

                file_path = os.path.join(FILINGS_10K_DIR, f"{ticker}_10K_{year}.html")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(doc_response.text)

                filing_count += 1
            except requests.exceptions.RequestException as e:
                logger.error(f"    Error downloading filing {doc_url}: {e}")

    logger.info(f"✅ Finished fetching filings for {ticker}")


# ========== REDDIT SCRAPING FUNCTIONS (PRAW API) ==========

# Subreddits to scrape
SUBREDDITS = [
    'stocks', 'investing', 'wallstreetbets', 'SecurityAnalysis', 'ValueInvesting',
    'UraniumSqueeze', 'uraniumstocks', 'renewableenergy', 'cryptomining',
    'BitcoinMining', 'NVDA', 'CryptoCurrency', 'gpumining', 'NiceHash',
    'EtherMining', 'CryptoMiningTalk', 'BitcoinMiningStock', 'CryptoMarkets'
]

# Time range: past 1 month
DAYS_BACK = 30


def contains_ticker(text: str) -> List[str]:
    """Check if text contains any of our target tickers or company names."""
    text_upper = text.upper()
    found_tickers = []

    # Map of tickers to alternative names/variations
    ticker_aliases = {
        'IREN': ['IREN', 'IRIS ENERGY', '$IREN'],
        'RIOT': ['RIOT', 'RIOT PLATFORMS', 'RIOT BLOCKCHAIN', '$RIOT'],
        'CIFR': ['CIFR', 'CIPHER MINING', '$CIFR'],
        'OKLO': ['OKLO', '$OKLO'],
        'SMR': ['SMR', 'NUSCALE', 'NUSCALE POWER', '$SMR'],
        'INOD': ['INOD', 'INNODATA', '$INOD'],
        'EOSE': ['EOSE', 'EOS ENERGY', '$EOSE'],
        'QS': ['QS', 'QUANTUMSCAPE', '$QS'],
        'VRT': ['VRT', 'VERTIV', '$VRT'],
        'VST': ['VST', 'VISTRA', '$VST'],
        'CCJ': ['CCJ', 'CAMECO', '$CCJ'],
        'NXE': ['NXE', 'NEXGEN', 'NEXGEN ENERGY', '$NXE'],
        'NVDA': ['NVDA', 'NVIDIA', '$NVDA'],
        'MU': ['MU', 'MICRON', '$MU'],
        'AVGO': ['AVGO', 'BROADCOM', '$AVGO'],
        'TSM': ['TSM', 'TSMC', 'TAIWAN SEMI', '$TSM'],
        'SMCI': ['SMCI', 'SUPER MICRO', 'SUPERMICRO', '$SMCI'],
        'RR': ['$RR']  # Too common, only match with $
    }

    for ticker in TARGET_TICKERS:
        aliases = ticker_aliases.get(ticker, [ticker])
        for alias in aliases:
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, text_upper):
                if ticker not in found_tickers:
                    found_tickers.append(ticker)
                break

    return found_tickers


def analyze_sentiment(text: str, analyzer) -> Dict[str, Any]:
    """Analyze sentiment of text using VADER."""
    scores = analyzer.polarity_scores(text)

    if scores['compound'] >= 0.05:
        sentiment = 'bullish'
    elif scores['compound'] <= -0.05:
        sentiment = 'bearish'
    else:
        sentiment = 'neutral'

    return {
        'sentiment': sentiment,
        'compound_score': scores['compound'],
        'positive_score': scores['pos'],
        'negative_score': scores['neg'],
        'neutral_score': scores['neu']
    }


def extract_topics(text: str) -> List[str]:
    """Extract key topics from text using simple keyword matching."""
    text_lower = text.lower()
    topics = []

    topic_keywords = {
        'earnings': ['earnings', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'revenue', 'profit'],
        'technical_analysis': ['chart', 'ta', 'technical', 'support', 'resistance', 'breakout'],
        'fundamentals': ['pe ratio', 'p/e', 'valuation', 'book value', 'debt', 'cash flow'],
        'news': ['news', 'announcement', 'press release', 'ceo', 'management'],
        'partnerships': ['partnership', 'deal', 'acquisition', 'merger', 'collaboration'],
        'regulatory': ['sec', 'fda', 'approval', 'regulation', 'compliance'],
        'market_sentiment': ['bullish', 'bearish', 'optimistic', 'pessimistic', 'hype'],
        'AI': ['ai', 'artificial intelligence', 'machine learning', 'gpu', 'data center'],
        'crypto': ['crypto', 'bitcoin', 'ethereum', 'mining', 'blockchain'],
        'uranium': ['uranium', 'nuclear', 'reactor', 'fuel'],
        'renewable': ['renewable', 'solar', 'wind', 'clean energy', 'green']
    }

    for topic, keywords in topic_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            topics.append(topic)

    return topics


def scrape_subreddit_with_praw(reddit_client, subreddit_name: str, analyzer, since_date, limit: int = 100) -> List[Dict[str, Any]]:
    """Scrape posts from a specific subreddit using PRAW."""
    logger.info(f"Scraping r/{subreddit_name}...")

    subreddit = reddit_client.subreddit(subreddit_name)
    posts_data = []
    seen_post_ids = set()

    try:
        # Get both hot and new posts for better coverage
        for post_type, post_generator in [('hot', subreddit.hot(limit=limit)),
                                          ('new', subreddit.new(limit=limit))]:
            for post in post_generator:
                # Skip duplicates
                if post.id in seen_post_ids:
                    continue
                seen_post_ids.add(post.id)

                # Check if post is from the past month
                post_date = datetime.fromtimestamp(post.created_utc)
                if post_date < since_date:
                    continue

                # Combine title and selftext
                full_text = f"{post.title}\n{post.selftext or ''}"

                # Check if post mentions any of our tickers
                mentioned_tickers = contains_ticker(full_text)
                if not mentioned_tickers:
                    continue

                # Analyze sentiment
                sentiment_data = analyze_sentiment(full_text, analyzer)

                # Extract topics
                topics = extract_topics(full_text)

                # Get top comments (up to 5)
                comments_data = []
                post.comments.replace_more(limit=0)  # Don't load more comments
                for comment in post.comments[:5]:
                    if hasattr(comment, 'body') and comment.body != '[deleted]':
                        comment_tickers = contains_ticker(comment.body)
                        if comment_tickers:
                            comment_sentiment = analyze_sentiment(comment.body, analyzer)
                            comments_data.append({
                                'id': comment.id,
                                'body': comment.body[:500],  # Limit length
                                'score': comment.score,
                                'created_utc': comment.created_utc,
                                'mentioned_tickers': comment_tickers,
                                'sentiment': comment_sentiment['sentiment'],
                                'compound_score': comment_sentiment['compound_score']
                            })

                # Store post data
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'selftext': post.selftext or '',
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'subreddit': subreddit_name,
                    'url': f"https://reddit.com{post.permalink}",
                    'mentioned_tickers': mentioned_tickers,
                    'sentiment': sentiment_data['sentiment'],
                    'compound_score': sentiment_data['compound_score'],
                    'positive_score': sentiment_data['positive_score'],
                    'negative_score': sentiment_data['negative_score'],
                    'topics': topics,
                    'comments': comments_data
                }

                posts_data.append(post_data)
                logger.info(f"  Found post about {mentioned_tickers}: {post.title[:50]}...")

                # Rate limiting
                time.sleep(1)

    except Exception as e:
        logger.error(f"Error scraping r/{subreddit_name}: {e}")

    return posts_data


def scrape_reddit_with_praw() -> List[Dict[str, Any]]:
    """Scrape Reddit posts using PRAW API with authentication."""
    if not REDDIT_AVAILABLE:
        logger.warning("Reddit scraping not available (praw not installed). Skipping...")
        return []

    logger.info("=" * 60)
    logger.info("SCRAPING REDDIT WITH PRAW API")
    logger.info("=" * 60)
    logger.info(f"Target tickers: {', '.join(sorted(TARGET_TICKERS))}")
    logger.info(f"Subreddits: {', '.join(SUBREDDITS)}")
    logger.info(f"Time range: Past {DAYS_BACK} days")

    # Initialize Reddit API client
    try:
        reddit_client = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID', "9RrzkLg9kN06g-kpti2ncw"),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET', "OH0pyFbl8T2ykN0IeAC1m5uNUu287A"),
            user_agent=os.getenv('REDDIT_USER_AGENT', "FinancialAgent/1.0 by u/Feeling-Berry5335")
        )

        # Test authentication
        logger.info(f"Authenticated as: {reddit_client.user.me() if reddit_client.user.me() else 'Anonymous (Read-only)'}")
    except Exception as e:
        logger.error(f"Failed to initialize Reddit client: {e}")
        logger.info("Falling back to RSS scraping...")
        return []

    analyzer = SentimentIntensityAnalyzer()
    since_date = datetime.now() - timedelta(days=DAYS_BACK)
    all_posts = []

    for subreddit_name in SUBREDDITS:
        try:
            posts = scrape_subreddit_with_praw(reddit_client, subreddit_name, analyzer, since_date)
            all_posts.extend(posts)
            logger.info(f"  Scraped {len(posts)} posts from r/{subreddit_name}")
        except Exception as e:
            logger.error(f"Failed to scrape r/{subreddit_name}: {e}")
            continue

    return all_posts


def fetch_reddit_data():
    """Fetch Reddit data and save to files."""
    posts_data = scrape_reddit_with_praw()

    if not posts_data:
        logger.warning("No Reddit posts found mentioning target tickers.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save all posts
    all_posts_file = os.path.join(REDDIT_DATA_DIR, f"reddit_posts_{timestamp}.json")
    with open(all_posts_file, 'w') as f:
        json.dump(posts_data, f, indent=2)

    # Save summary by ticker
    ticker_summary = {}
    for post in posts_data:
        for ticker in post['mentioned_tickers']:
            if ticker not in ticker_summary:
                ticker_summary[ticker] = {
                    'total_posts': 0,
                    'bullish_posts': 0,
                    'bearish_posts': 0,
                    'neutral_posts': 0,
                    'avg_sentiment': 0,
                    'subreddits': set()
                }

            ticker_summary[ticker]['total_posts'] += 1
            ticker_summary[ticker]['subreddits'].add(post['subreddit'])

            if post['sentiment'] == 'bullish':
                ticker_summary[ticker]['bullish_posts'] += 1
            elif post['sentiment'] == 'bearish':
                ticker_summary[ticker]['bearish_posts'] += 1
            else:
                ticker_summary[ticker]['neutral_posts'] += 1

    # Convert sets to lists for JSON serialization
    for ticker_data in ticker_summary.values():
        ticker_data['subreddits'] = list(ticker_data['subreddits'])

    summary_file = os.path.join(REDDIT_DATA_DIR, f"reddit_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(ticker_summary, f, indent=2)

    logger.info(f"✅ Saved {len(posts_data)} Reddit posts to {all_posts_file}")
    logger.info(f"✅ Saved summary to {summary_file}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("FINANCIAL DATA FETCH PIPELINE STARTED")
    logger.info("=" * 60)
    logger.info(f"Target tickers: {', '.join(TARGET_TICKERS)}")
    logger.info("")

    # Step 1: Fetch company information from SEC
    companies_df = fetch_company_info_from_sec()

    if companies_df.empty:
        logger.error("Failed to fetch company information. Exiting.")
        exit(1)

    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPANY INFORMATION FETCHED")
    logger.info("=" * 60)
    logger.info(f"Total companies: {len(companies_df)}")
    logger.info("")

    # Step 2: Fetch CEO profiles
    fetch_ceo_profiles(companies_df)

    logger.info("")
    logger.info("=" * 60)
    logger.info("FETCHING FINANCIAL DATA")
    logger.info("=" * 60)

    # Step 3: Fetch financial statements, prices, and filings
    # Filter out tickers with dots or dashes that yfinance might not handle well
    companies_df = companies_df[~companies_df['ticker'].str.contains(r'\.|\-')]

    for index, row in tqdm(companies_df.iterrows(), total=companies_df.shape[0], desc="Processing Companies"):
        ticker = row['ticker']
        cik = str(row['cik'])

        # Fetch and Save Data
        fetch_financial_statements(ticker)
        fetch_stock_prices(ticker)
        fetch_10k_filings(ticker, cik)

        # Rate limit to be respectful to APIs
        time.sleep(0.5)

    logger.info("")
    logger.info("=" * 60)
    logger.info("FETCHING REDDIT DATA")
    logger.info("=" * 60)

    # Step 4: Fetch Reddit data
    fetch_reddit_data()

    logger.info("")
    logger.info("=" * 60)
    logger.info("DATA FETCHING PIPELINE COMPLETED")
    logger.info("=" * 60)
    logger.info("Check the 'data' directory for all fetched data.")
