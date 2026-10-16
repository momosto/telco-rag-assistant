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
    don doesn didn won isn aren t s ll ve re
    happen happens take takes need want know like would could should got make
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
    "price": ["cost", "fee", "usd"],
    "fee": ["price", "cost"],
    "pay": ["paid", "payment"],
    "cheap": ["price", "usd"],
    "cheapest": ["price", "usd"],
    "fast": ["mbps", "speed"],
    "fix": ["repair"],
    "cancel": ["termination", "cancell"],
    "package": ["plan"],
    "deal": ["plan", "bundle"],
    "internet": ["fibre", "data"],
    "wifi": ["router", "fibre"],
    "buy": ["bought", "purchase"],
    "lost": ["loss", "replacement", "replac"],
    "stolen": ["loss", "replacement", "replac"],
    "register": ["registration"],
    "refund": ["refundable"],
    "number": ["call", "contact"],
    "contact": ["call", "email"],
    "phone": ["call"],
}

# Phrases whose meaning lives in stopwords ("how much") or in word pairs ("is down").
PHRASES = [
    (r"how much|cheap|expensive|afford", ["price", "usd"]),
    (r"how long|how soon|how quickly|when will", ["day", "hour", "within"]),
    (r"how fast|\bspeed|slow", ["mbps", "download"]),
    (r"\bdown\b|not working|no internet|no connection|broken|offline", ["fault", "repair"]),
    (r"don'?t pay|not pay|didn'?t pay|\blate\b|overdue", ["unpaid", "suspend"]),
]


def tokenize(text: str) -> list[str]:
    # Keep USSD codes like *300*9# and numbers like 0.10 as single tokens.
    tokens = re.findall(r"\*[\d*]+#|\d+(?:\.\d+)?|[a-z]+(?:-[a-z]+)*", text.lower())
    return [stem(t) for t in tokens if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.5):
        self.chunks = chunks
        self.k1, self.b = k1, b
        # The section heading is counted twice (the document title once): "Plans and
        # monthly prices" says more about a section than any single sentence in it.
        self.docs = [Counter(tokenize(f"{c.title} {c.title.split(' > ')[-1]} {c.text}")) for c in chunks]
        self.headings = [set(tokenize(c.title.split(" > ")[-1])) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(term for d in self.docs for term in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def term_score(self, term: str, i: int) -> float:
        tf = self.docs[i].get(term)
        if not tf:
            return 0.0
        length = self.lengths[i]
        norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
        return self.idf[term] * norm

    def score(self, groups: list[set[str]], i: int) -> float:
        # Each idea in the question counts once, by its best-matching word, so a phrase
        # that expands to three synonyms can't outweigh the words the customer typed.
        total = sum(max(self.term_score(t, i) for t in g) for g in groups)
        # A section whose own heading names what was asked ("Installation", "Ways to pay")
        # is usually the one the customer wants.
        total += sum(0.5 * max(self.idf.get(t, 0) for t in g) for g in groups if g & self.headings[i])
        return total

    def concepts(self, query: str) -> list[set[str]]:
        """Each question word with its synonyms, plus one set per matched phrase."""
        lowered = query.lower()
        phrase_groups = []
        for pattern, terms in PHRASES:
            if re.search(pattern, lowered):
                phrase_groups.append(set(terms))
                # The phrase stands in for its words: "how long" shouldn't also match "as long as".
                lowered = re.sub(pattern, " ", lowered)
        groups = [{t, *SYNONYMS.get(t, [])} for t in dict.fromkeys(tokenize(lowered))]
        return groups + phrase_groups

    def search(self, query: str, k: int = 4, min_score: float = 1.0) -> list[Hit]:
        groups = self.concepts(query)
        if not groups:
            return []
        # A section must match at least two separate ideas in the question (or every idea,
        # for one-idea questions). Matching a single word like "offer" or "rate" is what
        # off-topic questions look like, so those are declined.
        needed = min(2, len(groups))
        # Words the knowledge base never mentions ("satellite", "iPhone", "CEO") signal an
        # off-topic question. Decline when they outnumber the words it does mention.
        vocabulary = self.idf.keys()
        known = sum(1 for g in groups if g & vocabulary)
        if len(groups) - known >= max(known, 1):
            return []
        scored = []
        for i, chunk in enumerate(self.chunks):
            keys = self.docs[i].keys()
            if sum(1 for g in groups if g & keys) >= needed:
                scored.append(Hit(chunk, self.score(groups, i)))
        scored = [h for h in scored if h.score >= min_score]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]
