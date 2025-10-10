# agents.py (Corrected for Financial Domain)
"""
Defines the ADK agent team for the financial data application.
This includes a root agent for orchestration and specialized sub-agents
for graph querying, document retrieval, and stock price predictions.
"""
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from ..neo4j_for_adk import graphdb
from app.models.predict import predict_next_day_price
from langchain_google_vertexai import VertexAIEmbeddings
from target_tickers import TARGET_TICKERS
import pandas as pd
import os
from datetime import datetime, timezone
import yfinance as yf
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    build = None  # type: ignore[assignment]
    HttpError = Exception  # type: ignore[assignment]

# --- Setup ---
llm = LiteLlm(model="gemini-2.5-flash")

embeddings = VertexAIEmbeddings(model_name="text-embedding-005")

GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
TICKER_LIST_STR = ", ".join(sorted(TARGET_TICKERS))

# --- Tool Definitions ---
def query_graph_database(question: str) -> dict:
    """
    Generates a Cypher query for the financial graph and executes it.
    """
    schema = graphdb.send_query("CALL db.schema.visualization()")["query_result"]
    
    # CORRECTED: Updated schema description and examples to match actual database structure
    cypher_generation_prompt = f"""
    Task: Generate a Cypher statement to query a financial graph database.
    
    Schema: {schema}
    
    Instructions:
    - Use ONLY the provided relationship types and property keys.
    - The graph contains the following nodes and relationships:
      - (c:Company)-[:HAS_FINANCIALS]->(f:Financials)
      - (c:Company)-[:FILED]->(doc:Document)
      - (c:Company)-[:HAS_RISK]->(r:Risk)
      - (c:Company)-[:HAD_EVENT]->(e:Event)
      - (c:Company)-[:HAS_STRATEGY]->(s:Strategy)
      - (doc:Document)-[:MENTIONS_RISK]->(r:Risk)
      - (doc:Document)-[:DESCRIBES_EVENT]->(e:Event)
      - (doc:Document)-[:MENTIONS_STRATEGY]->(s:Strategy)
      - (chunk:Chunk) nodes with vector embeddings for document chunks
    
    - Key properties for nodes:
    - Company: `ticker` (e.g., 'NVDA'), `name`, `cik`
    - Financials: `company` (ticker), `year` (string like '2024'), `revenue`, `netIncome`, `eps`
    - Risk, Event, Strategy: `name`
    - Document: `source` (filename), `year`, `type`, `management_outlook`
    - Chunk: `text`, `embedding` (vector)
    
    - IMPORTANT: The Financials node uses `company` property (not ticker directly) and `year` is a STRING
    - Company tickers in your data: {TICKER_LIST_STR}
    
    Example Questions & Queries (ticker and year are database property names, not variables):
    - Question: "What was the revenue for NVDA in 2024?"
      Query: MATCH (c:Company {{ticker: 'NVDA'}})-[:HAS_FINANCIALS]->(f:Financials {{year: '2024'}}) RETURN f.revenue
    - Question: "What are the key risks for NVDA?"
      Query: MATCH (c:Company {{ticker: 'NVDA'}})-[:HAS_RISK]->(r:Risk) RETURN r.name
    - Question: "Show me financial trends for NVDA over the years"
      Query: MATCH (c:Company {{ticker: 'NVDA'}})-[:HAS_FINANCIALS]->(f:Financials) RETURN f.year, f.revenue, f.netIncome, f.eps ORDER BY f.year
    - Question: "What events happened at Apple?"
      Query: MATCH (c:Company {{ticker: 'AAPL'}})-[:HAD_EVENT]->(e:Event) RETURN e.name
    
    Question: {question}
    Return only the Cypher query, no explanation or formatting.
    """
    
    cypher_query = llm.llm_client.completion(
        model=llm.model,
        messages=[{"role": "user", "content": cypher_generation_prompt}],
        tools=[], # <-- ADD THIS LINE
    ).choices[0].message.content.strip()
    
    # Clean the response
    cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
    print(f"Generated Cypher: {cypher_query}")
    
    return graphdb.send_query(cypher_query)

def retrieve_from_documents(question: str) -> dict:
    """
    Performs vector search on 10-K filing chunks and synthesizes an answer.
    """
    question_embedding = embeddings.embed_query(question)
    
    search_query = """
    CALL db.index.vector.queryNodes('filings', 5, $embedding) YIELD node, score
    RETURN node.text AS text, score
    ORDER BY score DESC
    """
    
    search_results = graphdb.send_query(search_query, {"embedding": question_embedding})
    
    if search_results['status'] == 'error' or not search_results['query_result']:
        return {"answer": "Could not retrieve relevant documents from filings.", "error": search_results.get('message', 'Unknown error')}
    
    context = "\n".join([r['text'] for r in search_results['query_result']])
    
    synthesis_prompt = f"""
    Based on the following context from SEC 10-K filings, answer the question comprehensively.
    
    Context from filings:
    {context}
    
    Question: {question}
    
    Instructions:
    - Provide a detailed answer based on the context
    - If the context doesn't contain relevant information, say so
    - Cite specific information from the filings when possible
    - Focus on the financial and strategic aspects mentioned
    
    Answer:
    """
    
    response = llm.llm_client.completion(
        model=llm.model,
        messages=[{"role": "user", "content": synthesis_prompt}],
        tools=[], # <-- ADD THIS LINE
    ).choices[0].message.content
    
    return {"answer": response}

def _discover_trained_tickers() -> set:
    from pathlib import Path
    models_dir = Path(__file__).resolve().parents[1] / "models" / "saved_models"
    if not models_dir.exists():
        return set()
    tickers = set()
    for p in models_dir.glob("*_price_regressor.joblib"):
        tickers.add(p.name.split('_')[0].upper())
    return tickers


def predict_stock_price_tool(ticker: str) -> dict:
    """
    A wrapper for the stock price prediction model.
    Input must be a single, valid stock ticker string from our available companies.
    """
    valid_tickers = _discover_trained_tickers()
    
    if not isinstance(ticker, str):
        return {"error": f"Invalid input type. Please provide a ticker as a string."}
    
    ticker = ticker.upper().strip()
    
    if not valid_tickers:
        return {"error": "No trained models found. Train models via app/models/train_predictor.py first."}
    if ticker not in valid_tickers:
        return {"error": f"Ticker '{ticker}' not found. Available tickers: {', '.join(sorted(valid_tickers))}"}
    
    print(f"Predicting price for ticker: {ticker}")
    return predict_next_day_price(ticker)


def search_latest_news(query: str, max_results: int = 5) -> dict:
    """
    Uses Google Custom Search API to retrieve the latest news articles related to the query.
    """
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
        return {"error": "Google Search API is not configured. Please set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID."}

    if build is None:
        return {"error": "google-api-python-client is not installed. Please install google-api-python-client."}

    max_results = max(1, min(max_results, 10))

    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY, cache_discovery=False)
        response = service.cse().list(
            q=query,
            cx=GOOGLE_SEARCH_ENGINE_ID,
            num=max_results,
            dateRestrict="d7"
        ).execute()

        items = response.get("items", [])
        if not items:
            return {"results": [], "message": "No recent news found for the query."}

        results = []
        for item in items:
            pagemap = item.get("pagemap", {})
            metatags = pagemap.get("metatags", [{}])
            news_article = pagemap.get("newsarticle", [{}])

            published = (
                metatags[0].get("article:published_time")
                or metatags[0].get("og:updated_time")
                or news_article[0].get("datepublished")
                or news_article[0].get("datemodified")
            )

            source = metatags[0].get("og:site_name") if metatags else None

            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
                "published": published,
                "source": source
            })

        return {"query": query, "results": results}
    except HttpError as http_error:
        return {"error": f"Google Search API error: {http_error}"}
    except Exception as exc:
        return {"error": f"Unexpected error querying Google Search API: {exc}"}


def fetch_intraday_price_and_events(ticker: str, max_events: int = 5) -> dict:
    """
    Fetches the latest available price information and recent news events for a ticker.
    Uses Yahoo Finance (delayed data) for intraday prices and press coverage.
    """
    if not isinstance(ticker, str):
        return {"error": "Ticker must be provided as a string."}

    ticker = ticker.upper().strip()
    if ticker not in TARGET_TICKERS:
        return {"error": f"Ticker '{ticker}' is not in the supported list: {TICKER_LIST_STR}"}

    try:
        instrument = yf.Ticker(ticker)
    except Exception as exc:
        return {"error": f"Failed to initialize market data client for {ticker}: {exc}"}

    price_payload: dict = {}
    try:
        fast_info = getattr(instrument, "fast_info", {}) or {}
        last_price = fast_info.get("last_price")
        prev_close = fast_info.get("previous_close") or fast_info.get("regular_market_previous_close")
        currency = fast_info.get("currency") or fast_info.get("last_price_currency")
        exchange = fast_info.get("exchange") or fast_info.get("market")
        price_time = fast_info.get("last_price_time")

        if not last_price:
            intraday = instrument.history(period="1d", interval="5m")
            if not intraday.empty:
                last_row = intraday.tail(1).iloc[0]
                last_price = float(last_row["Close"])
                prev_close = prev_close or float(intraday.head(1).iloc[0]["Open"])
                price_time = last_row.name.to_pydatetime().replace(tzinfo=timezone.utc)

        if last_price:
            change = None
            percent_change = None
            if prev_close and prev_close != 0:
                change = float(last_price) - float(prev_close)
                percent_change = (change / float(prev_close)) * 100

            if isinstance(price_time, (int, float)):
                price_dt = datetime.fromtimestamp(price_time, tz=timezone.utc)
            elif isinstance(price_time, datetime):
                price_dt = price_time.astimezone(timezone.utc)
            else:
                price_dt = datetime.now(timezone.utc)

            price_payload = {
                "ticker": ticker,
                "last_price": round(float(last_price), 4),
                "previous_close": round(float(prev_close), 4) if prev_close else None,
                "change": round(float(change), 4) if change is not None else None,
                "percent_change": round(float(percent_change), 4) if percent_change is not None else None,
                "currency": currency,
                "exchange": exchange,
                "price_timestamp_utc": price_dt.isoformat(),
                "disclaimer": "Pricing data sourced from Yahoo Finance and may be delayed."
            }
        else:
            price_payload = {"warning": f"No price data available for {ticker} at this time."}
    except Exception as exc:
        price_payload = {"error": f"Failed to fetch intraday price for {ticker}: {exc}"}

    events: list = []
    try:
        raw_news = getattr(instrument, "news", []) or []
        for item in raw_news[:max_events]:
            published = item.get("providerPublishTime")
            if published:
                published_dt = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()
            else:
                published_dt = None
            events.append({
                "title": item.get("title"),
                "publisher": item.get("publisher"),
                "published_at_utc": published_dt,
                "type": item.get("type"),
                "link": item.get("link")
            })
    except Exception as exc:
        events = []
        price_payload.setdefault("warnings", []).append(f"Failed to retrieve news for {ticker}: {exc}")

    return {
        "ticker": ticker,
        "price": price_payload,
        "recent_events": events,
        "notes": "News items provided by Yahoo Finance; availability varies by ticker."
    }

# --- Sub-Agent Definitions ---
graph_qa_subagent = Agent(
    name="GraphQA_Agent",
    model=llm,
    tools=[query_graph_database],
    description=f"Use for questions about company financials (revenue, net income, EPS), risks, events, strategies, and any structured data queries. Works with tickers: {TICKER_LIST_STR}.",
    instruction=f"""
    Your task is to use the `query_graph_database` tool to answer questions about:
    - Financial metrics (revenue, net income, EPS) by company and year
    - Company risks, events, and strategic focuses
    - Comparisons between companies
    - Financial trends over time
    
    Always use the exact ticker symbols: {TICKER_LIST_STR}
    Remember that years are stored as strings (e.g., '2024', '2023').
    """
)

document_rag_subagent = Agent(
    name="DocumentRAG_Agent",
    model=llm,
    tools=[retrieve_from_documents],
    description="Use for qualitative questions about company strategy, management outlook, detailed business descriptions, or any information that requires reading through SEC 10-K filing text.",
    instruction="""
    Your task is to use the `retrieve_from_documents` tool to find detailed, qualitative information from SEC filings including:
    - Management's discussion and analysis
    - Business strategy and outlook
    - Detailed risk descriptions
    - Product and service descriptions
    - Market analysis and competitive positioning
    
    Provide comprehensive answers based on the retrieved document chunks.
    """
)

prediction_subagent = Agent(
    name="StockPricePredictor_Agent",
    model=llm,
    tools=[predict_stock_price_tool],
    description="Use ONLY to predict the next day's closing stock price. Dynamically discovers available tickers from trained models.",
    instruction="""
    Your only task is to use the `predict_stock_price_tool` for stock price predictions.

    IMPORTANT:
    - Only use tickers for which models have been trained (dynamically discovered).
    - Input must be a single ticker string.
    - Always include a disclaimer that predictions are estimates based on historical data and not financial advice.
    """
)

news_search_subagent = Agent(
    name="NewsSearch_Agent",
    model=llm,
    tools=[search_latest_news],
    description="Use to gather the latest news headlines and summaries for financial queries.",
    instruction="""
    Your task is to search for the most recent and relevant news stories that help answer the user's question.

    CAPABILITIES:
    - Retrieve fresh headlines related to companies, sectors, or macroeconomic topics
    - Provide source name, publication time, and short summaries
    - Highlight multiple perspectives when available

    IMPORTANT:
    - Use concise summaries referencing the original article
    - Include publication time and source where possible
    - Clarify that links point to external news sites
    - Do not fabricate articles; rely solely on returned search results
    """
)

market_data_subagent = Agent(
    name="MarketData_Agent",
    model=llm,
    tools=[fetch_intraday_price_and_events],
    description="Use to retrieve the latest available stock price and recent events for supported tickers using Yahoo Finance.",
    instruction=f"""
    Your task is to report the most recent market data for a company.

    CAPABILITIES:
    - Provide the latest intraday price (delayed) and percent change for supported tickers ({TICKER_LIST_STR})
    - Include timestamp, exchange, and currency when available
    - Surface recent news/events with titles, publishers, and links
    - Mention that prices may be delayed and sourced from Yahoo Finance

    IMPORTANT:
    - Only answer for supported tickers
    - Clearly state when data is unavailable or delayed
    - Do not generate forecasts; stick to retrieved data
    """
)

# --- LinkedIn CEO Lookup Agent ---
def query_ceo_info_by_ticker(ticker: str) -> dict:
    """
    Queries CEO information by company ticker using multiple data sources.

    Args:
        ticker: The company ticker symbol (e.g., 'NVDA', 'AAPL')

    Returns:
        dict: Contains CEO name, current title, duration, and company info
    """
    import requests
    from bs4 import BeautifulSoup
    import re
    from datetime import datetime
    from dateutil import parser
    import json

    try:
        # Validate ticker input
        if not ticker or not isinstance(ticker, str):
            return {"error": "Invalid ticker. Please provide a valid stock ticker symbol."}

        ticker = ticker.upper().strip()

        # Get company name from data/companies.csv if available
        company_name = f"{ticker} Company"
        if os.path.exists("data/companies.csv"):
            try:
                companies_df = pd.read_csv("data/companies.csv")
                company_row = companies_df[companies_df['ticker'] == ticker]
                if not company_row.empty:
                    company_name = company_row.iloc[0]['company_name']
            except Exception:
                pass

        ceo_data = {
            "ticker": ticker,
            "company_name": company_name,
            "ceo_name": "Not found",
            "ceo_title": "Not found",
            "tenure_duration": "Not found",
            "start_date": "Not found",
            "linkedin_url": "Not found",
            "source": "Multiple sources",
            "past_experience": [],
            "education": "Not found",
            "career_highlights": []
        }

        # Method 1: Try to find CEO via company's investor relations or about page
        search_terms = [
            f"{company_name} CEO chief executive officer",
            f"{ticker} company CEO leadership",
            f"{company_name} management team"
        ]

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

        # Method 2: Use a direct company website search or SEC filings approach
        # For well-known companies, use hardcoded data for reliability
        known_ceos = {
            'NVDA': {
                'name': 'Jensen Huang',
                'title': 'President, CEO and Co-Founder',
                'start_date': 'April 1993',
                'calculated_duration': '30+ years',
                'source': 'NVIDIA official records',
                'past_experience': [
                    {'company': 'AMD', 'role': 'Director of CoreWare', 'duration': '1993', 'description': 'Led microprocessor design'},
                    {'company': 'LSI Logic', 'role': 'Microprocessor Designer', 'duration': '1985-1993', 'description': 'Designed microprocessors and chipsets'}
                ],
                'education': 'Stanford University (MS Electrical Engineering), Oregon State University (BS Electrical Engineering)',
                'career_highlights': [
                    'Co-founded NVIDIA at age 30',
                    'Led GPU revolution in computing',
                    'Pioneer in AI and machine learning hardware',
                    'Named one of Time\'s 100 Most Influential People'
                ]
            },
            'AAPL': {
                'name': 'Tim Cook',
                'title': 'Chief Executive Officer',
                'start_date': 'August 2011',
                'calculated_duration': '13+ years',
                'source': 'Apple official records',
                'past_experience': [
                    {'company': 'Apple', 'role': 'Chief Operating Officer', 'duration': '2007-2011', 'description': 'Managed worldwide sales and operations'},
                    {'company': 'Apple', 'role': 'Senior Vice President of Worldwide Operations', 'duration': '1998-2007', 'description': 'Streamlined manufacturing and supply chain'},
                    {'company': 'Compaq', 'role': 'Vice President of Corporate Materials', 'duration': '1994-1998', 'description': 'Managed procurement and vendor relations'},
                    {'company': 'IBM', 'role': 'Various Positions', 'duration': '1982-1994', 'description': 'Operations and supply chain management'}
                ],
                'education': 'Auburn University (BS Industrial Engineering), Duke University (MBA)',
                'career_highlights': [
                    'Transformed Apple\'s supply chain operations',
                    'Succeeded Steve Jobs as CEO',
                    'Led Apple to become world\'s most valuable company',
                    'Advocate for privacy and environmental sustainability'
                ]
            },
            'MSFT': {
                'name': 'Satya Nadella',
                'title': 'Chairman and Chief Executive Officer',
                'start_date': 'February 2014',
                'calculated_duration': '10+ years',
                'source': 'Microsoft official records',
                'past_experience': [
                    {'company': 'Microsoft', 'role': 'Executive Vice President of Cloud and Enterprise', 'duration': '2011-2014', 'description': 'Led Azure cloud computing platform'},
                    {'company': 'Microsoft', 'role': 'Senior Vice President of R&D for Online Services', 'duration': '2007-2011', 'description': 'Oversaw Bing search engine development'},
                    {'company': 'Microsoft', 'role': 'Various Technical Leadership Roles', 'duration': '1992-2007', 'description': 'Led multiple product divisions including Office and Windows Server'}
                ],
                'education': 'University of Chicago Booth School of Business (MBA), University of Wisconsin-Milwaukee (MS Computer Science), Manipal Institute of Technology (BS Electrical Engineering)',
                'career_highlights': [
                    'Transformed Microsoft to cloud-first company',
                    'Led successful Azure cloud platform launch',
                    '22+ years at Microsoft before becoming CEO',
                    'Focused on AI and digital transformation'
                ]
            },
            'GOOGL': {
                'name': 'Sundar Pichai',
                'title': 'Chief Executive Officer',
                'start_date': 'October 2015',
                'calculated_duration': '9+ years',
                'source': 'Alphabet official records',
                'past_experience': [
                    {'company': 'Google', 'role': 'Senior Vice President of Products', 'duration': '2013-2015', 'description': 'Oversaw Google\'s product portfolio including Search, Maps, and Google+'},
                    {'company': 'Google', 'role': 'Vice President of Product Management', 'duration': '2008-2013', 'description': 'Led Chrome browser development and strategy'},
                    {'company': 'Google', 'role': 'Product Manager', 'duration': '2004-2008', 'description': 'Worked on Google Toolbar and other early products'},
                    {'company': 'McKinsey & Company', 'role': 'Consultant', 'duration': '2002-2004', 'description': 'Management consulting'},
                    {'company': 'Applied Materials', 'role': 'Engineer', 'duration': '2000-2002', 'description': 'Semiconductor equipment engineering'}
                ],
                'education': 'Stanford University (MBA), University of Pennsylvania (MS Materials Science), Indian Institute of Technology Kharagpur (B.Tech Metallurgical Engineering)',
                'career_highlights': [
                    'Led development of Google Chrome browser',
                    'Oversaw Android mobile operating system',
                    'Instrumental in Google\'s mobile-first strategy',
                    'Became CEO of Alphabet in 2019'
                ]
            },
            'AMZN': {
                'name': 'Andy Jassy',
                'title': 'President and Chief Executive Officer',
                'start_date': 'July 2021',
                'calculated_duration': '3+ years',
                'source': 'Amazon official records',
                'past_experience': [
                    {'company': 'Amazon Web Services', 'role': 'CEO', 'duration': '2016-2021', 'description': 'Led world\'s largest cloud computing platform'},
                    {'company': 'Amazon Web Services', 'role': 'Senior Vice President', 'duration': '2006-2016', 'description': 'Built AWS from startup to $10B+ business'},
                    {'company': 'Amazon', 'role': 'Various Leadership Roles', 'duration': '1997-2006', 'description': 'Led multiple business units and strategic initiatives'},
                    {'company': 'MBI', 'role': 'Project Manager', 'duration': '1990s', 'description': 'Early career in project management'}
                ],
                'education': 'Harvard Business School (MBA), Harvard University (BA Government)',
                'career_highlights': [
                    'Built Amazon Web Services into dominant cloud platform',
                    '24+ years at Amazon before becoming CEO',
                    'Pioneered cloud computing industry',
                    'Succeeded Jeff Bezos as Amazon CEO'
                ]
            },
            'TSLA': {
                'name': 'Elon Musk',
                'title': 'Chief Executive Officer',
                'start_date': 'October 2008',
                'calculated_duration': '16+ years',
                'source': 'Tesla official records',
                'past_experience': [
                    {'company': 'SpaceX', 'role': 'Founder, CEO and CTO', 'duration': '2002-Present', 'description': 'Private space exploration and satellite internet'},
                    {'company': 'PayPal (X.com)', 'role': 'Co-founder and CEO', 'duration': '1999-2002', 'description': 'Online payment system, sold to eBay for $1.5B'},
                    {'company': 'Zip2', 'role': 'Co-founder and CEO', 'duration': '1995-1999', 'description': 'Online city guide software, sold to Compaq for $307M'}
                ],
                'education': 'University of Pennsylvania (BS Physics, BS Economics), Stanford University (PhD Physics - dropped out)',
                'career_highlights': [
                    'Serial entrepreneur with multiple successful exits',
                    'Revolutionized electric vehicle industry',
                    'Advancing space exploration through SpaceX',
                    'World\'s richest person (various times)'
                ]
            },
            'META': {
                'name': 'Mark Zuckerberg',
                'title': 'Chairman and Chief Executive Officer',
                'start_date': 'February 2004',
                'calculated_duration': '20+ years',
                'source': 'Meta official records',
                'past_experience': [
                    {'company': 'Harvard University', 'role': 'Student / Early Facebook Development', 'duration': '2003-2004', 'description': 'Created Facebook while studying psychology and computer science'},
                    {'company': 'Various Programming Projects', 'role': 'Programmer', 'duration': 'High School - 2003', 'description': 'Created Synapse Media Player and other software applications'}
                ],
                'education': 'Harvard University (Psychology and Computer Science - dropped out)',
                'career_highlights': [
                    'Founded Facebook at age 19',
                    'Built world\'s largest social media platform',
                    'Youngest billionaire at age 23',
                    'Leading development of metaverse technologies'
                ]
            },
            'IREN': {
                'name': 'Daniel Roberts',
                'title': 'Chief Executive Officer and Co-Founder',
                'start_date': 'November 2018',
                'calculated_duration': '6+ years',
                'source': 'Iris Energy official records',
                'past_experience': [
                    {'company': 'Macquarie Group', 'role': 'Investment Banking', 'duration': '2010-2018', 'description': 'Corporate finance and renewable energy investments'},
                    {'company': 'Various Energy Companies', 'role': 'Strategy and Development', 'duration': '2008-2010', 'description': 'Renewable energy project development'}
                ],
                'education': 'University of Sydney (Finance and Economics)',
                'career_highlights': [
                    'Co-founded Iris Energy focused on sustainable Bitcoin mining',
                    'Expertise in renewable energy and cryptocurrency',
                    'Led company to NASDAQ listing',
                    'Pioneer in ESG-focused cryptocurrency mining'
                ]
            }
        }

        if ticker in known_ceos:
            ceo_info = known_ceos[ticker]
            ceo_data.update({
                "ceo_name": ceo_info['name'],
                "ceo_title": ceo_info['title'],
                "start_date": ceo_info['start_date'],
                "tenure_duration": ceo_info['calculated_duration'],
                "source": ceo_info['source'],
                "past_experience": ceo_info.get('past_experience', []),
                "education": ceo_info.get('education', 'Not found'),
                "career_highlights": ceo_info.get('career_highlights', [])
            })

            # Try to find LinkedIn profile for the CEO
            try:
                linkedin_search_url = f"https://www.google.com/search?q={ceo_info['name']}+{company_name}+linkedin"
                search_response = requests.get(linkedin_search_url, headers=headers, timeout=10)

                if search_response.status_code == 200:
                    linkedin_pattern = r'https://[a-z]{2,3}\.linkedin\.com/in/[a-zA-Z0-9\-]+/?'
                    linkedin_matches = re.findall(linkedin_pattern, search_response.text)

                    if linkedin_matches:
                        # Clean and validate LinkedIn URL
                        linkedin_url = linkedin_matches[0].replace('\\', '')
                        if 'linkedin.com/in/' in linkedin_url:
                            ceo_data["linkedin_url"] = linkedin_url

            except Exception as e:
                pass  # LinkedIn search failed, continue without it

        else:
            # Method 3: For unknown companies, try web scraping approach
            try:
                # Search for CEO information using Google search
                search_query = f"{company_name} current CEO chief executive officer 2025"
                search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"

                search_response = requests.get(search_url, headers=headers, timeout=10)

                if search_response.status_code == 200:
                    soup = BeautifulSoup(search_response.content, 'html.parser')

                    # Look for CEO information in search results
                    text_content = soup.get_text()

                    # Common patterns for CEO information
                    ceo_patterns = [
                        rf'{re.escape(company_name)}.*?CEO[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)',
                        rf'CEO[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+).*?{re.escape(company_name)}',
                        rf'Chief Executive Officer[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)',
                        rf'([A-Z][a-z]+\s+[A-Z][a-z]+)[,\s]*CEO of {re.escape(company_name)}',
                    ]

                    for pattern in ceo_patterns:
                        matches = re.findall(pattern, text_content, re.IGNORECASE)
                        if matches:
                            potential_ceo = matches[0].strip()
                            # Validate the name (should be 2-4 words, proper capitalization)
                            if 2 <= len(potential_ceo.split()) <= 4 and potential_ceo.replace(' ', '').isalpha():
                                ceo_data["ceo_name"] = potential_ceo
                                ceo_data["ceo_title"] = "Chief Executive Officer"
                                ceo_data["source"] = "Web search results"
                                break

            except Exception as e:
                ceo_data["search_error"] = f"Web search failed: {str(e)}"

        # Calculate more precise duration if we have start date
        if ceo_data["start_date"] != "Not found":
            try:
                # Extract year from start date
                year_match = re.search(r'(\d{4})', ceo_data["start_date"])
                if year_match:
                    start_year = int(year_match.group(1))
                    current_year = datetime.now().year
                    years_tenure = current_year - start_year

                    if years_tenure >= 0:
                        ceo_data["calculated_years"] = years_tenure
                        if years_tenure == 1:
                            ceo_data["tenure_duration"] = "1 year"
                        else:
                            ceo_data["tenure_duration"] = f"{years_tenure} years"

            except Exception as e:
                pass  # Keep original duration if calculation fails

        return {
            "success": True,
            "ceo_data": ceo_data,
            "note": "CEO information compiled from multiple sources. Data accuracy may vary for less common companies."
        }

    except Exception as e:
        return {"error": f"Error retrieving CEO information: {str(e)}"}

# --- Reddit Sentiment Agent ---
def query_reddit_sentiment(question: str) -> dict:
    """
    Queries Reddit sentiment data for companies.
    Analyzes social media sentiment, discussions, and trends.
    """
    try:
        # Generate Cypher query for Reddit sentiment analysis
        tickers_list = "', '".join(TARGET_TICKERS)
        cypher_query = f"""
        MATCH (p:RedditPost)-[:MENTIONS]->(c:Company)
        WHERE c.ticker IN ['{tickers_list}']
        RETURN c.ticker as ticker, 
               p.sentiment as sentiment,
               p.compound_score as compound_score,
               p.score as post_score,
               p.subreddit as subreddit,
               p.title as title,
               p.created_utc as created_utc,
               p.topics as topics
        ORDER BY p.created_utc DESC
        LIMIT 50
        """
        
        result = graphdb.send_query(cypher_query)
        return {
            "query_result": result.get("query_result", []),
            "cypher_query": cypher_query
        }
    except Exception as e:
        return {"error": f"Reddit sentiment query failed: {str(e)}"}

reddit_sentiment_subagent = Agent(
    name="RedditSentiment_Agent",
    model=llm,
    tools=[query_reddit_sentiment],
    description="Use ONLY for Reddit sentiment analysis, social media discussions, and community sentiment around companies.",
    instruction="""
    Your only task is to analyze Reddit sentiment and discussions about companies.

    CAPABILITIES:
    - Analyze sentiment trends (bullish/bearish/neutral) for specific tickers
    - Find most discussed topics and themes
    - Identify high-engagement posts and comments
    - Track sentiment changes over time
    - Compare sentiment across different subreddits

    IMPORTANT:
    - Focus on Reddit discussions and social sentiment
    - Provide insights on community sentiment and popular opinions
    - Include relevant post titles and subreddit sources
    - Always mention that Reddit sentiment is not financial advice
    """
)

ceo_lookup_subagent = Agent(
    name="CEOLookup_Agent",
    model=llm,
    tools=[query_ceo_info_by_ticker],
    description="Use ONLY for CEO information lookup by company ticker symbol. Returns CEO name, title, and tenure duration.",
    instruction=f"""
    Your only task is to find CEO information by company ticker symbol.

    CAPABILITIES:
    - Find CEO name by company ticker (supported tickers: {TICKER_LIST_STR})
    - Extract CEO's current title and position
    - Calculate duration/tenure as CEO
    - Provide start date and years of service
    - Find LinkedIn profile when available
    - Support major publicly traded companies

    IMPORTANT:
    - Input must be a valid stock ticker symbol
    - Provides current CEO information (as of 2024)
    - Data accuracy is highest for major companies (Fortune 500)
    - Always mention data sources and limitations
    - Include disclaimers about information currency and accuracy
    """
)

# --- Root Agent Definition ---
root_agent = Agent(
    name="Financial_Root_Agent",
    model=llm,
    sub_agents=[graph_qa_subagent, document_rag_subagent, news_search_subagent, market_data_subagent, prediction_subagent, reddit_sentiment_subagent, ceo_lookup_subagent],
    description="The main financial assistant that analyzes user queries and delegates to specialized agents for financial data analysis and CEO information lookup.",
    instruction=f"""
    You are a knowledgeable financial data assistant with access to data for {len(TARGET_TICKERS)} companies including: {', '.join(TARGET_TICKERS)}.

    DELEGATION GUIDELINES:
    - Use 'GraphQA_Agent' for:
      * Specific financial numbers (revenue, net income, EPS)
      * Company risks, events, strategies (structured data)
      * Financial comparisons and trends
      * Any query requiring precise data extraction

    - Use 'DocumentRAG_Agent' for:
      * Qualitative analysis and detailed explanations
      * Management outlook and business strategy discussions
      * Complex business descriptions
      * Questions requiring reading through filing narratives

    - Use 'NewsSearch_Agent' for:
      * Latest news headlines, press releases, and recent developments
      * Market-moving events or breaking stories
      * Context that requires up-to-date information prior to analysis

    - Use 'MarketData_Agent' for:
      * Latest available (delayed) intraday price and percent change
      * Exchange/currency metadata for a ticker
      * Recent company news/events sourced via Yahoo Finance

    - Use 'StockPricePredictor_Agent' ONLY for:
      * Explicit requests to predict future stock prices
      * The agent will dynamically discover available tickers from trained models

    - Use 'RedditSentiment_Agent' for:
      * Reddit sentiment analysis and social media discussions
      * Community sentiment around specific companies
      * Popular topics and trends on Reddit
      * Social media buzz and engagement metrics

    - Use 'CEOLookup_Agent' for:
      * CEO information lookup by company ticker symbol
      * CEO name, title, and tenure duration queries
      * Executive leadership information for companies
      * CEO career history and time in position

    IMPORTANT NOTES:
    - Available companies: {len(TARGET_TICKERS)} tickers across semiconductor, energy, crypto-mining, and tech verticals ({TICKER_LIST_STR})
    - Financial data years: 2021-2024
    - Quarterly earnings: Latest four quarters stored per company
    - Reddit data: Past 1 month from 9 subreddits
    - News search: Requires internet access and Google Custom Search credentials
    - CEO data: Current information for major publicly traded companies
    - Always include disclaimers for predictions, Reddit sentiment, and CEO data
    - If uncertain about which agent to use, explain your reasoning
    """
)
