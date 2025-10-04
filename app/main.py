import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import FileResponse

from app.agents.agents import query_graph_database, retrieve_from_documents

load_dotenv()

APP_DIR = Path(__file__).resolve().parent


class QueryRequest(BaseModel):
    question: str
    mode: Optional[str] = None  # 'graph' | 'docs' | None (auto in future)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Financial MVP API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get_index():
    template_path = APP_DIR / "templates" / "index.html"
    if template_path.exists():
        return FileResponse(template_path)
    return {"status": "ok", "message": "Backend running. UI not found."}


@app.post("/chat")
async def chat(request: QueryRequest, http_request: Request):
    q = request.question or ""
    mode = (request.mode or "").lower()

    if not q or len(q.strip()) < 4:
        return {"status": "validation_error", "answer": "Question too short."}

    if mode == "graph" or ("revenue" in q.lower() or "risk" in q.lower()):
        res = query_graph_database(q)
        if res.get("status") != "success":
            return {"status": "error", "answer": res.get("error_message", "Graph query failed."), "data": res}

        rows = res.get("rows", [])
        params = res.get("params", {})

        # Format answer heuristically
        answer: str
        if rows and isinstance(rows[0], dict) and "revenue" in rows[0]:
            ticker = params.get("ticker", "?")
            year = params.get("year", "?")
            revenue = rows[0].get("revenue")
            answer = f"Revenue for {ticker} in {year}: {revenue}"
        elif rows and isinstance(rows[0], dict) and "risk" in rows[0]:
            ticker = params.get("ticker", "?")
            risks = [r.get("risk") for r in rows if isinstance(r, dict)]
            risks = [r for r in risks if r]
            if risks:
                bullets = "\n".join(f"- {r}" for r in risks[:10])
                answer = f"Key risks for {ticker}:\n{bullets}"
            else:
                answer = f"No risks found for {ticker}."
        else:
            # Fallback to echo raw rows
            import json
            answer = f"Results:\n{json.dumps(rows, indent=2)}"

        return {"status": "success", "answer": answer, "data": res}

    if mode == "docs":
        res = retrieve_from_documents(q)
        if res.get("status") != "success":
            return {"status": "error", "answer": res.get("note", res.get("error_message", "Document search unavailable.")), "data": res}
        hits = res.get("hits", [])
        if not hits:
            return {"status": "success", "answer": res.get("note", "No relevant documents found."), "data": res}
        # Combine top snippets
        snippets = [h.get("text", "") for h in hits[:3]]
        combined = "\n\n".join(snippets)
        return {"status": "success", "answer": combined, "data": res}

    # Fallback: try docs
    res = retrieve_from_documents(q)
    return {"status": res.get("status", "success"), "answer": res.get("note", ""), "data": res}
