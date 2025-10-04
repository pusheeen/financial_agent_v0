from typing import Dict, Any

from app.neo4j_client import get_neo4j_client
from app.vector_search import semantic_search


def query_graph_database(question: str) -> Dict[str, Any]:
    """
    MVP: naive routing by detecting ticker/year and example queries.
    Replace with LLM-guided Cypher generation when available.
    """
    cypher = None

    # Very simple heuristics for demo purposes
    # Examples: "revenue NVDA 2024", "risks NVDA".
    lower_q = question.lower()

    if "revenue" in lower_q:
        # Try to find ticker and year
        import re
        ticker_match = re.search(r"\b(NVDA|MSFT|AAPL|GOOGL|AMZN)\b", question, re.I)
        year_match = re.search(r"\b(20\d{2})\b", question)
        if ticker_match and year_match:
            ticker = ticker_match.group(1).upper()
            year = year_match.group(1)
            cypher = (
                "MATCH (c:Company {ticker: $ticker})-[:HAS_FINANCIALS]->(f:Financials {year: $year}) "
                "RETURN f.revenue AS revenue"
            )
            params = {"ticker": ticker, "year": year}
        else:
            return {"status": "error", "error_message": "Please specify ticker and year."}
    elif "risk" in lower_q:
        import re
        ticker_match = re.search(r"\b(NVDA|MSFT|AAPL|GOOGL|AMZN)\b", question, re.I)
        if ticker_match:
            ticker = ticker_match.group(1).upper()
            cypher = (
                "MATCH (c:Company {ticker: $ticker})-[:HAS_RISK]->(r:Risk) RETURN r.name AS risk"
            )
            params = {"ticker": ticker}
        else:
            return {"status": "error", "error_message": "Please specify ticker."}
    else:
        return {"status": "error", "error_message": "Unsupported question for MVP."}

    try:
        client = get_neo4j_client()
        result = client.run(cypher, params)
        rows = [record.data() for record in result]
        return {"status": "success", "rows": rows, "cypher": cypher, "params": params}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def retrieve_from_documents(question: str) -> Dict[str, Any]:
    return semantic_search(question)
