import os
import re
from typing import List, Dict, Any

from dotenv import load_dotenv

from app.neo4j_client import get_neo4j_client

load_dotenv()

# Placeholder for embedding/vector search to keep MVP runnable without cloud deps.
# In production, use VertexAIEmbeddings + Neo4j vector index or a local vector DB.


def embed_texts(texts: List[str]) -> List[List[float]]:
    # Minimal stub: returns zero vectors of fixed size.
    return [[0.0] * 8 for _ in texts]


def _keyword_search_in_neo4j(question: str) -> Dict[str, Any]:
    """
    Fallback keyword search over Chunk.text in Neo4j when embeddings are unavailable.
    """
    # Extract simple keywords (alphanumeric words > 3 chars)
    words = re.findall(r"[A-Za-z0-9]{4,}", question.lower())
    # Deduplicate and cap
    keywords = []
    for w in words:
        if w not in keywords:
            keywords.append(w)
    keywords = keywords[:5] or [question]

    try:
        client = get_neo4j_client()
        cypher = (
            "MATCH (ch:Chunk) "
            "WITH ch, toLower(ch.text) AS text "
            "WITH ch, text, $kws AS kws "
            "WHERE any(kw IN kws WHERE text CONTAINS toLower(kw)) "
            "RETURN ch.text AS text LIMIT 5"
        )
        result = client.run(cypher, {"kws": keywords})
        rows = [r.data() for r in result]
        hits = [{"text": r.get("text", "")} for r in rows]
        return {"status": "success", "hits": hits, "note": "Keyword search over Chunk.text"}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def semantic_search(query: str) -> Dict[str, Any]:
    # Try fallback keyword search if Neo4j is configured
    res = _keyword_search_in_neo4j(query)
    if res.get("status") == "success":
        return res
    # Minimal stub result if no DB
    return {
        "status": "success",
        "hits": [],
        "note": "Vector search not configured; integrate embeddings + Neo4j index."
    }
