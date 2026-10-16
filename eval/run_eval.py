"""Score the assistant on 20 questions with known answers.

    python -m eval.run_eval                 # local models (run scripts/download_models.py first)
    RAG_MODE=keyword python -m eval.run_eval  # keyword search only, no models

A question counts as correct when:
- answerable: every expected fact appears in the answer (any listed spelling)
  and at least one citation points at the expected document;
- unanswerable: the assistant says it doesn't know.

Results are printed and written to eval/results-<mode>.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag.answer import answer, use_models
from rag.api import get_retriever

HERE = Path(__file__).resolve().parent


def grade(item: dict, result) -> tuple[bool, str]:
    if item.get("answerable", True) is False:
        ok = result.mode == "abstain"
        return ok, "abstained" if ok else "answered when it should not have"
    if result.mode == "abstain":
        return False, "abstained on an answerable question"
    text = result.text.lower()
    missing = [group[0] for group in item["expect"] if not any(alt.lower() in text for alt in group)]
    if missing:
        return False, f"missing {', '.join(missing)}"
    if not any(c.source == item["source"] for c in result.citations):
        return False, "correct but not cited to the right document"
    return True, "correct and cited"


def main() -> int:
    items = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    retriever = get_retriever()
    mode = "local-models" if use_models() else "keyword"
    rows, passed = [], 0

    for item in items:
        result = answer(item["question"], retriever)
        ok, reason = grade(item, result)
        passed += ok
        rows.append((item["id"], item["question"], "pass" if ok else "FAIL", reason))
        print(f"{'PASS' if ok else 'FAIL'}  {item['id']:>2}. {item['question']}  ({reason})")

    answerable = [r for r, i in zip(rows, items) if i.get("answerable", True)]
    unanswerable = [r for r, i in zip(rows, items) if not i.get("answerable", True)]
    summary = (
        f"{passed}/{len(items)} correct ({passed / len(items):.0%}) in {mode} mode: "
        f"{sum(r[2] == 'pass' for r in answerable)}/{len(answerable)} answerable, "
        f"{sum(r[2] == 'pass' for r in unanswerable)}/{len(unanswerable)} correctly declined"
    )
    print("\n" + summary)

    lines = [f"# Evaluation results ({mode} mode)", "", summary, "", "| # | Question | Result | Detail |", "|---|---|---|---|"]
    lines += [f"| {i} | {q} | {res} | {why} |" for i, q, res, why in rows]
    (HERE / f"results-{mode}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
