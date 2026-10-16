"""Turn a question and retrieved sections into a cited answer, using only local models.

Modes, chosen per answer:
- "model": the local reading model (DistilBERT, SQuAD) found the answer span in a
  section; the reply is the sentence containing it, with the span highlighted.
- "section": the reader wasn't confident, or the models aren't installed, so the best
  section is shown in full. Still cited.
- "abstain": nothing in the documents matches the question.

RAG_MODE=keyword forces the keyword-only pipeline (no models) for tests and comparison.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

from .retriever import SYNONYMS, Hit, tokenize

IDK = "I don't know. I couldn't find that in Mhepo's documents."

# Below this span probability the reader is usually pointing at something nearby rather
# than the answer, so the whole section is shown instead. Calibrated on eval/questions.json.
MIN_CONFIDENCE = 0.3
READ_TOP = 3  # sections the reader looks at


@dataclass
class Citation:
    source: str
    title: str
    quote: str


@dataclass
class Answer:
    text: str
    mode: str  # "model" | "section" | "abstain"
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)  # section titles, for debugging and evals
    highlight: str | None = None  # the exact answer span inside `text`, when the reader found one
    confidence: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def use_models() -> bool:
    from .models import models_available

    return os.environ.get("RAG_MODE", "auto") != "keyword" and models_available()


def answer(question: str, retriever, k: int = 4) -> Answer:
    hits = retriever.search(question, k=k)
    retrieved = [h.chunk.title for h in hits]
    if not hits:
        return Answer(IDK, "abstain", retrieved=retrieved)
    if use_models():
        result = _answer_with_reader(question, hits, getattr(retriever, "idf", {}))
        if result:
            result.retrieved = retrieved
            return result
    return _answer_section(question, hits, getattr(retriever, "idf", {}))


def _answer_with_reader(question: str, hits: list[Hit], idf: dict[str, float]) -> Answer | None:
    from .hybrid import section_context
    from .models import get_reader

    terms = set(tokenize(question))
    terms |= {syn for t in terms for syn in SYNONYMS.get(t, [])}
    # Read the top sections as one text in a single pass. Reader confidence is a softmax
    # over the text it sees, so scores from separate passes aren't comparable: it can be
    # very sure of an answer inside an irrelevant section. One pass makes them compete.
    pieces, offsets, cursor = [], [], 0
    for hit in hits[:READ_TOP]:
        piece = section_context(hit.chunk, terms, idf, heading=False)
        offsets.append((cursor, cursor + len(piece), hit))
        pieces.append(piece)
        cursor += len(piece) + 2
    context = "\n\n".join(pieces)

    span = get_reader().read(question, context)
    if span is None or span.confidence < MIN_CONFIDENCE:
        return None
    # Spans can run across a line break into the next list item; keep the first line.
    text = span.text.split("\n")[0].strip()
    if not text:
        return None
    hit = next(h for begin, finish, h in offsets if begin <= span.start < finish)
    sentence = _sentence_around(context, span.start, span.start + len(text))
    # The citation carries the whole section, so the customer can check the highlighted
    # answer against everything the section says.
    section = "\n".join(format_section(hit.chunk.text, terms, idf))
    return Answer(
        text=sentence,
        mode="model",
        citations=[Citation(hit.chunk.source, hit.chunk.title, section)],
        highlight=text,
        confidence=round(span.confidence, 3),
    )


def _sentence_around(text: str, start: int, end: int) -> str:
    """The line or sentence that contains characters start..end."""
    newline, full_stop = text.rfind("\n", 0, start), text.rfind(". ", 0, start)
    begin = max(newline + 1, full_stop + 2 if full_stop != -1 else 0)
    stops = [i for i in (text.find("\n", end), text.find(". ", end)) if i != -1]
    finish = min(stops) + 1 if stops else len(text)
    return text[begin:finish].strip()


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
                details = ", ".join(row[1:])
                out.append(f"{row[0]}: {details}" if details else row[0])
            continue
        if line:
            out.append(line)
        i += 1
    return out


def _answer_section(question: str, hits: list[Hit], idf: dict[str, float]) -> Answer:
    """Show the best-matching section in full, plus the runner-up when it is a near-tie."""
    terms = set(tokenize(question))
    terms |= {syn for t in terms for syn in SYNONYMS.get(t, [])}
    chosen = [hits[0]] + [h for h in hits[1:2] if h.score >= hits[0].score * 0.8]
    parts, citations = [], []
    for hit in chosen:
        lines = format_section(hit.chunk.text, terms, idf)
        heading = hit.chunk.title.split(" > ")[-1]
        parts.append(f"From \u201c{heading}\u201d:\n" + "\n".join(lines))
        citations.append(Citation(hit.chunk.source, hit.chunk.title, "\n".join(lines)))
    return Answer("\n\n".join(parts), "section", citations)
