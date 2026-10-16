"""Retrieval check on 20 questions that were written after tuning and never tuned on.

    python -m eval.retrieval_heldout

Reports how often the expected section is ranked first, and in the top two.
"""

import json
from pathlib import Path

from rag.api import get_retriever

items = json.loads((Path(__file__).resolve().parent / "heldout.json").read_text(encoding="utf-8"))
retriever = get_retriever()
top1 = top2 = 0
for item in items:
    ranked = [h.chunk.title.split(" > ")[-1] for h in retriever.search(item["question"])]
    first, in_two = ranked[:1] == [item["section"]], item["section"] in ranked[:2]
    top1 += first
    top2 += in_two
    print(f"{'top-1' if first else 'top-2' if in_two else 'MISS '}  {item['question']}  -> {ranked[:2]}")
print(f"\nExpected section ranked first: {top1}/{len(items)}; in top two: {top2}/{len(items)}")
