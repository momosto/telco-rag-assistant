"""Turn a question and retrieved chunks into a cited answer.

Two modes:
- "claude": the chunks are sent to Claude as documents with citations enabled, so
  every claim in the answer points at the exact passage it came from.
- "extractive": no API key needed. Returns the best-matching sentences verbatim,
  each cited to its chunk. Used for the free public demo and for offline tests.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field

from .retriever import SYNONYMS, BM25Retriever, Hit, tokenize

log = logging.getLogger(__name__)

MODEL = os.environ.get("RAG_MODEL", "claude-opus-5-5")
IDK = "I don't know. I couldn't find that in Mhepo's documents."

SYSTEM_PROMPT = """You are the customer support assistant for Mhepo Telecom, a telecom provider in Zimbabwe.

Answer the customer's question using only the documents provided in this message. Be brief and specific: lead with the direct answer (price, time, code or yes/no), then add only the conditions the customer needs to know.

If the documents do not contain the answer, reply with exactly: "I don't know. I couldn't find that in Mhepo's documents." Do not guess, and do not use outside knowledge about other telecom companies.

Reply in the language the customer used. Customers may write in English or Shona."""


@dataclass
class Citation:
    source: str
    title: str
    quote: str


@dataclass
class Answer:
    text: str
    mode: str  # "claude" | "extractive" | "abstain"
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)  # chunk titles, for debugging and evals
    note: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def use_claude() -> bool:
    mode = os.environ.get("RAG_MODE", "auto")
    if mode == "extractive":
        return False
    if mode == "claude":
        return True
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def answer(question: str, retriever: BM25Retriever, k: int = 4) -> Answer:
    hits = retriever.search(question, k=k)
    retrieved = [h.chunk.title for h in hits]
    if not hits:
        return Answer(IDK, "abstain", retrieved=retrieved)
    if use_claude():
        try:
            return _answer_with_claude(question, hits)
        except Exception as exc:  # network, auth or rate-limit: degrade, don't fail the request
            log.warning("Claude call failed, falling back to extractive: %s", exc)
            result = _answer_extractive(question, hits, retriever.idf)
            result.note = "The AI model was unavailable, so this answer is quoted directly from the documents."
            return result
    return _answer_extractive(question, hits, retriever.idf)


def _answer_with_claude(question: str, hits: list[Hit]) -> Answer:
    import anthropic

    client = anthropic.Anthropic()
    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": h.chunk.text},
            "title": h.chunk.title,
            "citations": {"enabled": True},
        }
        for h in hits
    ]
    # Server-side refusal fallbacks exist for the Opus, Fable and Sonnet lines, not Haiku.
    fallback = {} if "haiku" in MODEL else {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": "low"},
        **fallback,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [*documents, {"type": "text", "text": question}]}],
    )
    retrieved = [h.chunk.title for h in hits]

    if response.stop_reason == "refusal":
        return Answer("I can't help with that request.", "abstain", retrieved=retrieved)

    text_parts: list[str] = []
    citations: list[Citation] = []
    seen = set()
    for block in response.content:
        if block.type != "text":
            continue
        text_parts.append(block.text)
        for c in getattr(block, "citations", None) or []:
            index = getattr(c, "document_index", None)
            if index is None or index >= len(hits):
                continue
            chunk = hits[index].chunk
            quote = c.cited_text.strip()
            if (chunk.id, quote) in seen:
                continue
            seen.add((chunk.id, quote))
            citations.append(Citation(chunk.source, chunk.title, quote))

    text = "".join(text_parts).strip()
    if text.lower().startswith("i don't know") or text.lower().startswith("i don’t know"):
        return Answer(IDK, "abstain", retrieved=retrieved)
    return Answer(text, "claude", citations, retrieved)


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def format_section(text: str, terms: set[str], idf: dict[str, float]) -> list[str]:
    """Render a section as readable lines: tables become "Fibre Plus: 50 / 25 Mbps, USD 85".

    When the question names a specific row (a plan or bundle), only matching rows are kept.
    """
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|"):
            table = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table.append(lines[i].strip())
                i += 1
            header = _cells(table[0])
            rows = [_cells(r) for r in table[1:] if not set(r) <= set("|-: ")]
            header_terms = set(tokenize(" ".join(header)))
            # Distinctive question words that appear in some rows but not the header.
            row_terms = {t for t in terms - header_terms if idf.get(t, 0) > 1.0}
            matching = [r for r in rows if row_terms & set(tokenize(" ".join(r)))]
            for row in matching or rows:
                details = ", ".join(f"{v}" for v in row[1:])
                out.append(f"{row[0]}: {details}" if details else row[0])
            continue
        if line:
            out.append(line)
        i += 1
    return out


def _answer_extractive(question: str, hits: list[Hit], idf: dict[str, float] | None = None) -> Answer:
    """Without a model, answer with the best-matching section, quoted in full."""
    retrieved = [h.chunk.title for h in hits]
    if not hits:
        return Answer(IDK, "abstain", retrieved=retrieved)
    terms = set(tokenize(question))
    terms |= {syn for t in terms for syn in SYNONYMS.get(t, [])}
    # Quote the best section, plus the runner-up when it scores nearly as well: near-ties
    # usually mean the answer spans two sections ("lost SIM": registration and replacement).
    chosen = [hits[0]] + [h for h in hits[1:2] if h.score >= hits[0].score * 0.8]
    parts, citations = [], []
    for hit in chosen:
        lines = format_section(hit.chunk.text, terms, idf or {})
        heading = hit.chunk.title.split(" > ")[-1]
        parts.append(f"From \u201c{heading}\u201d:\n" + "\n".join(lines))
        citations.append(Citation(hit.chunk.source, hit.chunk.title, "\n".join(lines)))
    return Answer("\n\n".join(parts), "extractive", citations, retrieved)
