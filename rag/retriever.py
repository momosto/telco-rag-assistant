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
    if len(token) > 4 and token.endswith("ly"):
        return token[:-2]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if token.endswith("ed") and len(token) > 3:
        return token[:-2] if len(token) > 5 else token[:-1]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


# Query-side synonyms for words customers use that the documents phrase differently.
SYNONYMS = {
    "switch": ["activate"],
    "turn": ["activate"],
    "cost": ["price", "fee"],
    "price": ["cost"],
    "fix": ["repair"],
    "cancel": ["termination"],
}


def tokenize(text: str) -> list[str]:
    # Keep USSD codes like *300*9# and numbers like 0.10 as single tokens.
    tokens = re.findall(r"\*[\d*]+#|\d+(?:\.\d+)?|[a-z]+(?:-[a-z]+)*", text.lower())
    return [stem(t) for t in tokens if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


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
        terms += [syn for t in terms for syn in SYNONYMS.get(t, [])]
        if not terms:
            return []
        # A section must share at least two distinct words with the question (or every
        # word, for one-word questions). A single shared word like "offer" or "rate" is
        # what off-topic questions look like, so those are declined.
        needed = min(2, len(set(terms)))
        scored = [
            Hit(c, self.score(terms, i))
            for i, c in enumerate(self.chunks)
            if len(set(terms) & self.docs[i].keys()) >= needed
        ]
        scored = [h for h in scored if h.score >= min_score]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]
