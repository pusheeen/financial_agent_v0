# 2_populate_graph.py (Modified to embed the first 80,000 characters)
import os
import pandas as pd
import json
import math
from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import vertexai
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
vertexai.init(project=os.getenv("GOOGLE_PROJECT_ID"), location=os.getenv("GOOGLE_LOCATION"))
from langchain_neo4j import Neo4jVector
from dotenv import load_dotenv
from tqdm import tqdm
from neo4j import GraphDatabase
import re

load_dotenv()

# --- Optional ticker subset filtering ---
def _get_allowed_tickers() -> set:
    """
    Returns an optional set of allowed tickers based on environment variables:
    - TICKER_SUBSET: comma-separated list of tickers
    - TICKER_SUBSET_CSV: path to a CSV with a 'ticker' column
    If neither is provided, returns an empty set meaning "no restriction".
    """
    allowed: set = set()
    subset_env = os.getenv("TICKER_SUBSET", "").strip()
    if subset_env:
        allowed.update({t.strip().upper() for t in subset_env.split(',') if t.strip()})

    subset_csv = os.getenv("TICKER_SUBSET_CSV", "").strip()
    if subset_csv:
        try:
            df = pd.read_csv(subset_csv)
            if 'ticker' in df.columns:
                allowed.update({str(t).strip().upper() for t in df['ticker'].dropna().tolist()})
        except Exception as e:
            print(f"Warning: Failed to read TICKER_SUBSET_CSV '{subset_csv}': {e}")

    return allowed

ALLOWED_TICKERS = _get_allowed_tickers()
if ALLOWED_TICKERS:
    print(f"Ticker subset active ({len(ALLOWED_TICKERS)}): {', '.join(sorted(ALLOWED_TICKERS))}")
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
driver = GraphDatabase.driver(URI, auth=AUTH)
llm = VertexAI(model_name="gemini-2.0-flash", temperature=0)
embeddings = VertexAIEmbeddings(model_name="text-embedding-005")

# --- UNCHANGED FUNCTIONS (ingest_structured_data, extract_entities_from_filing) ---
def ingest_structured_data():
    """
    Loads company profiles from CSV and financial data from JSON files,
    then creates Company and Financials nodes in Neo4j.
    """
    print("Ingesting structured company and financial data...")
    companies_df = pd.read_csv('./companies.csv')
    if ALLOWED_TICKERS:
        companies_df = companies_df[companies_df['ticker'].str.upper().isin(ALLOWED_TICKERS)]
    company_records = companies_df.to_dict('records')
    ingest_companies_query = """
    UNWIND $records AS record
    MERGE (c:Company {ticker: record.ticker})
    SET c.name = record.company_name, c.cik = toString(record.cik)
    """
    with driver.session() as session:
        session.run(ingest_companies_query, records=company_records)

    financials_dir = './data/structured/financials'
    if os.path.exists(financials_dir):
        for filename in tqdm(os.listdir(financials_dir), desc="Ingesting Financials"):
            if filename.endswith(".json"):
                ticker = filename.split('_')[0].upper()
                if ALLOWED_TICKERS and ticker not in ALLOWED_TICKERS:
                    continue
                with open(os.path.join(financials_dir, filename), 'r') as f:
                    financials_data = json.load(f)
                records_to_ingest = []
                for item in financials_data:
                    def get_value(key):
                        val = item.get(key)
                        if val is None or (isinstance(val, float) and math.isnan(val)):
                            return None
                        return val
                    record = {
                        'ticker': ticker,
                        'year': item.get('date', '').split('-')[0],
                        'revenue': get_value('Total Revenue'),
                        'netIncome': get_value('Net Income'),
                        'eps': get_value('Basic EPS') or get_value('Diluted EPS')
                    }
                    if record['year']:
                        records_to_ingest.append(record)
                ingest_financials_query = """
                UNWIND $records AS record
                MATCH (c:Company {ticker: record.ticker})
                MERGE (f:Financials {company: c.ticker, year: record.year})
                SET f.revenue = toFloat(record.revenue), f.netIncome = toFloat(record.netIncome), f.eps = toFloat(record.eps)
                MERGE (c)-[:HAS_FINANCIALS]->(f)
                """
                with driver.session() as session:
                    session.run(ingest_financials_query, records=records_to_ingest)
    else:
        print(f"Warning: Financials directory {financials_dir} not found")
    print("Structured data ingestion complete.")

def extract_entities_from_filing(doc):
    """
    Uses an LLM to extract structured entities from the first 20,000 characters of a 10-K filing.
    """
    filename = os.path.basename(doc.metadata.get('source', ''))
    match = re.search(r"([A-Z]+)_10K_(\d{4})", filename)
    if not match:
        print(f"Warning: Could not extract ticker and year from filename: {filename}")
        return None
    ticker, year = match.groups()
    extraction_prompt = f"""
    From the SEC 10-K filing document below for ticker {ticker} and year {year}, extract the following information.
    Focus on the "Risk Factors" and "Management's Discussion and Analysis" sections if possible.
    - key_risks: A list of the 3-5 most significant risks mentioned.
    - management_outlook: A concise, one-paragraph summary of management's outlook.
    - major_events: A list of 1-3 major events from that year.
    - strategic_focus: A list of key strategic areas mentioned.
    Return the information as a valid JSON object with these exact keys. If any information is not found, use an empty list or null.
    Do not include any other text, explanation, or markdown formatting.
    DOCUMENT (first 20000 characters):
    {doc.page_content[:20000]}
    """
    try:
        response = llm.invoke(extraction_prompt)
        cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
        entities = json.loads(cleaned_response)
        entities['ticker'] = ticker
        entities['year'] = year
        return entities
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error processing document {doc.metadata.get('source', 'Unknown')}: {e}")
        print(f"LLM Response was: {response}")
        return None

def ingest_unstructured_data():
    """
    MODIFIED:
    - Extracts entities using the first 20,000 characters.
    - Chunks and creates vector embeddings for the first 80,000 characters.
    """
    print("Ingesting data from 10-K filings (2020-2025)...")
    filings_dir = './data/unstructured/10k/'
    if not os.path.exists(filings_dir):
        print(f"Warning: Filings directory {filings_dir} not found. Skipping unstructured data ingestion.")
        return
    
    loader = DirectoryLoader(
        filings_dir, glob="**/*.html", loader_cls=UnstructuredHTMLLoader,
        show_progress=True, loader_kwargs={"unstructured_kwargs": {"strategy": "fast"}}, silent_errors=True
    )
    documents = loader.load()
    if not documents:
        print("No documents found. Skipping unstructured data ingestion.")
        return
    
    target_years = [str(y) for y in range(2020, 2026)]
    docs_to_process = []
    for doc in documents:
        filename = os.path.basename(doc.metadata.get('source', ''))
        # Extract ticker prefix before first underscore if possible
        ticker_prefix = filename.split('_')[0].upper() if '_' in filename else ''
        if any(year in filename for year in target_years):
            if not ALLOWED_TICKERS or (ticker_prefix and ticker_prefix in ALLOWED_TICKERS):
                docs_to_process.append(doc)
    if not docs_to_process:
        print("No documents found for target years. Skipping unstructured data ingestion.")
        return
    
    print(f"Loaded {len(docs_to_process)} documents for years {target_years[0]}-{target_years[-1]}")
    print("Extracting and linking entities from filings...")
    with driver.session() as session:
        for doc in tqdm(docs_to_process, desc="Processing Filings"):
            entities = extract_entities_from_filing(doc)
            if entities and entities.get('ticker'):
                # Cypher query for linking entities (unchanged)
                link_query = """
                MATCH (c:Company {ticker: $ticker})
                MERGE (doc:Document {source: $source})
                ON CREATE SET doc.year = $year, doc.type = '10-K'
                MERGE (c)-[:FILED]->(doc)
                SET doc.management_outlook = $management_outlook

                WITH c, doc, $key_risks AS key_risks
                UNWIND key_risks AS risk_name
                WITH c, doc, risk_name WHERE risk_name IS NOT NULL AND risk_name <> ""
                MERGE (r:Risk {name: risk_name})
                MERGE (c)-[:HAS_RISK]->(r)
                MERGE (doc)-[:MENTIONS_RISK]->(r)

                WITH c, doc, $major_events AS major_events
                UNWIND major_events AS event_name
                WITH c, doc, event_name WHERE event_name IS NOT NULL AND event_name <> ""
                MERGE (e:Event {name: event_name})
                MERGE (c)-[:HAD_EVENT]->(e)
                MERGE (doc)-[:DESCRIBES_EVENT]->(e)

                WITH c, doc, $strategic_focus AS strategic_focus
                UNWIND strategic_focus AS strategy_name
                WITH c, doc, strategy_name WHERE strategy_name IS NOT NULL AND strategy_name <> ""
                MERGE (s:Strategy {name: strategy_name})
                MERGE (c)-[:HAS_STRATEGY]->(s)
                MERGE (doc)-[:MENTIONS_STRATEGY]->(s)
                """
                params = {
                    "source": os.path.basename(doc.metadata.get('source')), "ticker": entities.get('ticker'),
                    "year": entities.get('year'), "management_outlook": entities.get('management_outlook'),
                    "key_risks": entities.get('key_risks', []), "major_events": entities.get('major_events', []),
                    "strategic_focus": entities.get('strategic_focus', [])
                }
                try:
                    session.run(link_query, params)
                except Exception as e:
                    print(f"Error executing link query for {entities.get('ticker')}: {e}")

    print("Splitting documents and creating vector embeddings (first 80,000 chars)...")
    
    # --- FINAL MODIFICATION: Truncate documents to 80,000 characters for embedding ---
    docs_to_embed = []
    for doc in docs_to_process:
        truncated_doc = doc.copy()
        truncated_doc.page_content = doc.page_content[:80000] # Slice to 80,000
        docs_to_embed.append(truncated_doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    docs_for_vector = text_splitter.split_documents(docs_to_embed)
    
    try:
        Neo4jVector.from_documents(
            docs_for_vector, embeddings, url=URI, username=AUTH[0], password=AUTH[1],
            database="neo4j", index_name="filings", node_label="Chunk",
            text_node_property="text", embedding_node_property="embedding", create_id_index=True
        )
        print("Unstructured data ingestion and vector indexing complete.")
    except Exception as e:
        print(f"Error creating vector index: {e}")

# --- UNCHANGED MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("Clearing database...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        try:
            session.run("CALL db.index.vector.drop('filings')")
            print("Dropped existing vector index.")
        except Exception as e:
            print(f"No existing vector index to drop or error: {e}")
    ingest_structured_data()
    ingest_unstructured_data()
    print("\nDatabase population finished.")
    driver.close()