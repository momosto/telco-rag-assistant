"""HTTP API and chat page."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .answer import answer, use_models
from .ingest import load_directory
from .retriever import BM25Retriever

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
STATIC_DIR = ROOT / "static"

# Best-effort per-IP limit. On serverless hosts each instance keeps its own window,
# so put a real limiter (gateway or Redis) in front of anything that spends money.
RATE_LIMIT = 30  # requests
RATE_WINDOW = 60  # seconds
_requests: dict[str, deque] = defaultdict(deque)

app = FastAPI(title="Telco RAG Assistant", version="1.0.0")


@lru_cache(maxsize=1)
def get_keyword_retriever() -> BM25Retriever:
    return BM25Retriever(load_directory(DOCS_DIR))


@lru_cache(maxsize=1)
def get_retriever():
    """Hybrid keyword + embedding search when the local models are installed, else keywords."""
    keyword = get_keyword_retriever()
    if not use_models():
        return keyword
    from .hybrid import HybridRetriever
    from .models import get_embedder

    return HybridRetriever(keyword, get_embedder())


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


def _check_rate(ip: str) -> None:
    now = time.monotonic()
    window = _requests[ip]
    while window and now - window[0] > RATE_WINDOW:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        raise HTTPException(429, "Too many questions. Wait a minute and try again.")
    window.append(now)


@app.post("/api/ask")
def ask(body: AskRequest, request: Request) -> dict:
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "local").split(",")[0]
    _check_rate(ip)
    return answer(body.question.strip(), get_retriever()).to_dict()


@app.get("/api/health")
def health() -> dict:
    retriever = get_retriever()
    sources = sorted({c.source for c in retriever.chunks})
    return {
        "status": "ok",
        "mode": "local-models" if use_models() else "keyword",
        "documents": len(sources),
        "chunks": len(retriever.chunks),
    }


@app.get("/api/sources")
def sources() -> list[dict]:
    seen: dict[str, str] = {}
    for c in get_retriever().chunks:
        seen.setdefault(c.source, c.title.split(" > ")[0])
    return [{"source": s, "title": t} for s, t in seen.items()]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
