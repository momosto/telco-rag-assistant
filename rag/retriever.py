"""BM25 keyword retrieval.

A dependency-free BM25 index is enough for a knowledge base of this size, keeps
cold starts fast on serverless hosts, and needs no second API key for embeddings.
The Retriever interface (search -> list[Hit]) is what a vector store would implement.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .ingest import Chunk

STOPWORDS = set(
    """a an and are as at be by can do does for from has have how i if in is it its
    me my of on or so that the their them there this to was what when where which
    who will with you your yours we our us much many please tell about get
    mhepo telecom""".split()
)


def stem(token: str) -> str:
    # Light suffix stripping so "monthly" matches "month" and "used" matches "use".
    for suffix in ("ly", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] if suffix != "ed" else token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    # Keep USSD codes like *300*9# and numbers like 0.10 as single tokens.
    tokens = re.findall(r"\*[\d*]+#|\d+(?:\.\d+)?|[a-z]+(?:-[a-z]+)*", text.lower())
    return [stem(t) for t in tokens if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{c.title} {c.text}")) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(term for d in self.docs for term in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query_terms: list[str], i: int) -> float:
        doc, length = self.docs[i], self.lengths[i]
        total = 0.0
        for term in query_terms:
            tf = doc.get(term)
            if not tf:
                continue
            norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            total += self.idf[term] * norm
        return total

    def search(self, query: str, k: int = 4, min_score: float = 1.0) -> list[Hit]:
        terms = tokenize(query)
        if not terms:
            return []
        scored = [Hit(c, self.score(terms, i)) for i, c in enumerate(self.chunks)]
        scored = [h for h in scored if h.score >= min_score]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]
