# Financial MVP

Minimal FastAPI backend + UI for a financial assistant MVP.

## Run (local)

- Ensure Python 3.11+ is available on your system.
- Install deps:

```bash
pip install -r requirements.txt
```

- Set environment variables (create `.env` from `.env-example`).
- Start API:

```bash
uvicorn app.main:app --reload --port 8080
```

Open `http://127.0.0.1:8080`.

## Notes
- Neo4j is optional at startup, but required for graph queries.
- Vector search is stubbed; integrate VertexAI or OpenAI embeddings and a vector index in Neo4j for production.
