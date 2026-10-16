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
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low"},
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


def _sentences(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if not line or set(line) <= set("|-: "):
            continue
        if line.startswith("|"):
            lines.append(" ".join(cell.strip() for cell in line.strip("|").split("|")))
        else:
            lines.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    return lines


def _answer_extractive(question: str, hits: list[Hit], idf: dict[str, float] | None = None) -> Answer:
    terms = set(tokenize(question))
    terms |= {syn for t in terms for syn in SYNONYMS.get(t, [])}
    idf = idf or {}
    best: list[tuple[float, int, str, Hit]] = []
    for rank, hit in enumerate(hits):
        for position, sentence in enumerate(_sentences(hit.chunk.text)):
            overlap = terms & set(tokenize(sentence))
            if overlap:
                weight = sum(idf.get(t, 1.0) for t in overlap)
                best.append((weight - rank * 0.3, position, sentence, hit))
    retrieved = [h.chunk.title for h in hits]
    if not best:
        return Answer(IDK, "abstain", retrieved=retrieved)

    best.sort(key=lambda t: t[0], reverse=True)
    top_score, top_pos, top_sentence, top_hit = best[0]
    # Quote the best sentence plus one more from the same section: the runner-up if it
    # is close, otherwise the sentence that follows (which usually carries the detail).
    section = _sentences(top_hit.chunk.text)
    runner_up = next((b for b in best[1:] if b[3] is top_hit and b[0] >= top_score * 0.6), None)
    if runner_up:
        picked = sorted([(top_pos, top_sentence), (runner_up[1], runner_up[2])])
    elif top_pos + 1 < len(section):
        picked = [(top_pos, top_sentence), (top_pos + 1, section[top_pos + 1])]
    else:
        picked = [(top_pos, top_sentence)]
    chosen = [s for _, s in picked]
    citations = [Citation(top_hit.chunk.source, top_hit.chunk.title, s) for s in chosen]
    text = " ".join(s if s.endswith((".", "!", "?")) else s + "." for s in chosen)
    return Answer(text, "extractive", citations, retrieved)
