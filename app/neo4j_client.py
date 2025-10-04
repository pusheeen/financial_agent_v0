import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase, Result

load_dotenv()


class Neo4jClient:
    """
    Thin wrapper around the official Neo4j Python driver.
    """

    def __init__(self) -> None:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")

        if not uri or not user or not password:
            # Allow running without Neo4j for MVP; operations will fail gracefully
            self._driver = None
            self._database = os.getenv("NEO4J_DATABASE", "neo4j")
            return

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = os.getenv("NEO4J_DATABASE", "neo4j")

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass

    def run(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> Result:
        if self._driver is None:
            raise RuntimeError(
                "Neo4j not configured. Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD to enable graph queries."
            )
        with self._driver.session(database=self._database) as session:
            return session.run(cypher, params or {})


neo4j_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    global neo4j_client
    if neo4j_client is None:
        neo4j_client = Neo4jClient()
    return neo4j_client
