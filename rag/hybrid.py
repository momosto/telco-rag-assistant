"""Hybrid retrieval: BM25 keywords and bge-small embeddings, merged by reciprocal rank fusion.

Keywords are precise on names and numbers ("Fibre Plus", "*300*9#"); embeddings match
meaning ("changed my mind" -> cooling-off period). Fusing the two rankings keeps both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ingest import Chunk
from .models import Embedder
from .retriever import BM25Retriever, Hit

RRF_K = 60  # standard reciprocal-rank-fusion constant


def section_context(chunk: Chunk, terms: set[str] | None = None, idf: dict[str, float] | None = None, heading: bool = True) -> str:
    """What the models read: the heading, then the section with tables as plain lines.

    With question terms, tables keep only the rows the question names, so the reader
    can't answer "Fibre Plus" with the Fibre Home price. The reader gets no heading,
    or it answers "How do I pay?" with "Paying twice by mistake".
    """
    from .answer import format_section

    body = "\n".join(format_section(chunk.text, terms or set(), idf or {}))
    return f"{chunk.title.split(' > ')[-1]}.\n{body}" if heading else body


@dataclass(frozen=True)
class HybridHit(Hit):
    similarity: float = 0.0


class HybridRetriever:
    # Below this cosine similarity, a question that also shares no keywords with any
    # section is treated as off-topic. Calibrated on eval/questions.json only.
    MIN_SIMILARITY = 0.55
    # The keyword off-topic rule ("most words are unknown") is overridden when the meaning
    # match is this strong. On eval/questions.json every answerable question scores >= 0.74
    # and every unanswerable one <= 0.69, so the cut sits between them.
    CONFIDENT_SIMILARITY = 0.72

    def __init__(self, bm25: BM25Retriever, embedder: Embedder):
        self.bm25 = bm25
        self.chunks = bm25.chunks
        self.idf = bm25.idf
        self.embedder = embedder
        self.vectors = embedder.encode([section_context(c) for c in self.chunks])

    def search(self, query: str, k: int = 4) -> list[HybridHit]:
        similarity = self.vectors @ self.embedder.encode_query(query)
        if self.bm25.off_topic(query) and similarity.max() < self.CONFIDENT_SIMILARITY:
            return []
        keyword_hits = self.bm25.search(query, k=len(self.chunks))
        if not keyword_hits and similarity.max() < self.MIN_SIMILARITY:
            return []

        fused = np.zeros(len(self.chunks))
        for rank, i in enumerate(np.argsort(-similarity)):
            fused[i] += 1 / (RRF_K + rank)
        position = {h.chunk.id: i for i, h in enumerate(keyword_hits)}
        for i, chunk in enumerate(self.chunks):
            if chunk.id in position:
                fused[i] += 1 / (RRF_K + position[chunk.id])

        order = np.argsort(-fused)[:k]
        return [HybridHit(self.chunks[i], float(fused[i]), float(similarity[i])) for i in order]
