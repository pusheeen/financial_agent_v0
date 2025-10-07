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

# --- Setup ---
llm = LiteLlm(model="gemini-2.5-flash")

embeddings = VertexAIEmbeddings(model_name="text-embedding-005")

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
    - Company tickers in your data: NVDA, MSFT, AAPL, GOOGL, AMZN
    
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

# --- Sub-Agent Definitions ---
graph_qa_subagent = Agent(
    name="GraphQA_Agent",
    model=llm,
    tools=[query_graph_database],
    description="Use for questions about company financials (revenue, net income, EPS), risks, events, strategies, and any structured data queries. Works with tickers: NVDA, MSFT, AAPL, GOOGL, AMZN.",
    instruction="""
    Your task is to use the `query_graph_database` tool to answer questions about:
    - Financial metrics (revenue, net income, EPS) by company and year
    - Company risks, events, and strategic focuses
    - Comparisons between companies
    - Financial trends over time
    
    Always use the exact ticker symbols: NVDA, MSFT, AAPL, GOOGL, AMZN
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

# --- Reddit Sentiment Agent ---
def query_reddit_sentiment(question: str) -> dict:
    """
    Queries Reddit sentiment data for companies.
    Analyzes social media sentiment, discussions, and trends.
    """
    try:
        # Extract ticker from question (look for common ticker patterns)
        import re
        # Look for ticker patterns like "AVGO", "NVDA", etc. but exclude common words
        common_words = {'WHAT', 'THE', 'FOR', 'AND', 'ARE', 'IS', 'WAS', 'WERE', 'THIS', 'THAT', 'WITH', 'FROM', 'ABOUT', 'SENTIMENT', 'REDDIT'}
        ticker_matches = re.findall(r'\b([A-Z]{2,5})\b', question.upper())
        ticker = None
        for match in ticker_matches:
            if match not in common_words:
                ticker = match
                break
        
        # Generate Cypher query for Reddit sentiment analysis
        if ticker:
            cypher_query = f"""
            MATCH (p:RedditPost)-[:MENTIONS]->(c:Company)
            WHERE c.ticker = '{ticker}'
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
        else:
            # If no ticker found, return all available tickers with sentiment
            cypher_query = """
            MATCH (p:RedditPost)-[:MENTIONS]->(c:Company)
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
            "cypher_query": cypher_query,
            "searched_ticker": ticker
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

# --- Root Agent Definition ---
root_agent = Agent(
    name="Financial_Root_Agent",
    model=llm,
    sub_agents=[graph_qa_subagent, document_rag_subagent, prediction_subagent, reddit_sentiment_subagent],
    description="The main financial assistant that analyzes user queries and delegates to specialized agents for financial data analysis.",
    instruction="""
    You are a knowledgeable financial data assistant with access to data for 18 companies including: NVDA, MU, AVGO, TSM, VRT, SMCI, INOD, RR, IREN, CIFR, RIOT, OKLO, SMR, CCJ, VST, NXE, EOSE, QS, and others.
    
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
    
    - Use 'StockPricePredictor_Agent' ONLY for:
      * Explicit requests to predict future stock prices
      * The agent will dynamically discover available tickers from trained models
    
    - Use 'RedditSentiment_Agent' for:
      * ANY question mentioning "Reddit sentiment" or "Reddit"
      * Social media discussions about companies
      * Community sentiment around specific companies
      * Popular topics and trends on Reddit
      * Social media buzz and engagement metrics
      * Questions like "What is the Reddit sentiment for [TICKER]?"
    
    IMPORTANT NOTES:
    - Available companies: 18+ companies including semiconductor, energy, and tech companies
    - Financial data years: 2021-2024
    - Reddit data: Past 1 month from 9 subreddits
    - Always include disclaimers for predictions and Reddit sentiment
    - If uncertain about which agent to use, explain your reasoning
    """
)