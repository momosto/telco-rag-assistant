from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag import answer as answer_mod
from rag.ingest import chunk_html, chunk_markdown, load_directory
from rag.models import models_available
from rag.retriever import BM25Retriever, tokenize

DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"
needs_models = pytest.mark.skipif(not models_available(), reason="run scripts/download_models.py first")


@pytest.fixture(scope="module")
def keyword():
    return BM25Retriever(load_directory(DOCS))


@pytest.fixture(scope="module")
def hybrid(keyword):
    from rag.hybrid import HybridRetriever
    from rag.models import get_embedder

    return HybridRetriever(keyword, get_embedder())


# Ingestion and keyword search -------------------------------------------------------


def test_markdown_is_split_by_section_with_titles():
    chunks = chunk_markdown("plans.md", "# Plans\n\nIntro text.\n\n## Prices\n\nLite is USD 35.\n\n## Install\n\nUSD 60 once-off.")
    assert [c.title for c in chunks] == ["Plans", "Plans > Prices", "Plans > Install"]
    assert chunks[1].text == "Lite is USD 35."


def test_html_strips_scripts_and_navigation():
    chunks = chunk_html("https://x.test", "<title>FAQ</title><nav>Menu</nav><script>evil()</script><p>Pay by EcoCash.</p>")
    assert chunks[0].title == "FAQ"
    assert "Pay by EcoCash." in chunks[0].text
    assert "evil" not in chunks[0].text and "Menu" not in chunks[0].text


def test_tokenizer_keeps_ussd_codes_and_prices():
    tokens = tokenize("Dial *300*9# to block data charged at USD 0.10 per MB")
    assert "*300*9#" in tokens and "0.10" in tokens


@pytest.mark.parametrize(
    "question, section",
    [
        ("When can I use the Night bundle?", "Night bundle"),
        ("How do I switch on roaming?", "International roaming"),
        ("What merchant code do I use for EcoCash?", "Ways to pay"),
        ("How much is a replacement SIM?", "Replacing a lost or stolen SIM"),
    ],
)
def test_keyword_search_finds_the_section(keyword, question, section):
    titles = [h.chunk.title for h in keyword.search(question)]
    assert any(section in t for t in titles), titles


@pytest.mark.parametrize("question", ["Does Mhepo offer satellite TV packages?", "What is today's exchange rate for ZiG?"])
def test_single_shared_word_is_not_enough_to_answer(keyword, question):
    assert keyword.search(question) == []


def test_tables_are_rewritten_and_filtered_to_the_row_asked_about(keyword):
    chunk = next(c for c in keyword.chunks if c.title.endswith("Plans and monthly prices"))
    lines = answer_mod.format_section(chunk.text, set(tokenize("Fibre Plus price")), keyword.idf)
    assert lines[0] == "Fibre Plus: 50 / 25 Mbps, USD 85"
    assert not any("Fibre Lite" in line for line in lines)


def test_keyword_mode_shows_the_section(keyword, monkeypatch):
    monkeypatch.setenv("RAG_MODE", "keyword")
    result = answer_mod.answer("What merchant code do I use for EcoCash?", keyword)
    assert result.mode == "section"
    assert "112233" in result.text
    assert result.citations[0].source == "03-billing-and-payments.md"


def test_off_topic_question_is_declined(keyword, monkeypatch):
    monkeypatch.setenv("RAG_MODE", "keyword")
    result = answer_mod.answer("Who is the CEO of Mhepo Telecom?", keyword)
    assert result.mode == "abstain"
    assert result.text.startswith("I don't know")


# Local models ------------------------------------------------------------------------


@needs_models
def test_embeddings_match_meaning_not_just_words(hybrid):
    top = hybrid.search("I'm relocating, can I take my fibre with me?")[0]
    assert top.chunk.title.endswith("Moving house")


@needs_models
@pytest.mark.parametrize(
    "question, expected",
    [
        ("What merchant code do I use to pay with EcoCash?", "112233"),
        ("What does roaming cost without a bundle?", "USD 1 per MB"),
        ("What speed do I drop to under the fair usage policy?", "5 Mbps"),
    ],
)
def test_reader_highlights_the_answer(hybrid, monkeypatch, question, expected):
    monkeypatch.setenv("RAG_MODE", "auto")
    result = answer_mod.answer(question, hybrid)
    assert result.mode == "model"
    assert expected in result.highlight
    assert result.highlight in result.text
    assert 0 < result.confidence <= 1


@needs_models
def test_model_answer_cites_the_whole_section(hybrid, monkeypatch):
    monkeypatch.setenv("RAG_MODE", "auto")
    result = answer_mod.answer("What merchant code do I use to pay with EcoCash?", hybrid)
    assert result.citations[0].title.endswith("Ways to pay")
    assert "Debit order" in result.citations[0].quote


@needs_models
def test_hybrid_declines_off_topic_questions(hybrid):
    assert hybrid.search("Who is the CEO of Mhepo Telecom?") == []
    assert hybrid.search("Do you sell laptops?") == []


# HTTP API -----------------------------------------------------------------------------


def test_api_ask_health_and_validation():
    from rag.api import app

    client = TestClient(app)
    health = client.get("/api/health").json()
    assert health["documents"] == 8
    assert health["mode"] in {"local-models", "keyword"}
    response = client.post("/api/ask", json={"question": "How do I block out-of-bundle data?"})
    assert response.status_code == 200
    assert "*300*9#" in response.json()["text"]
    assert client.post("/api/ask", json={"question": "x" * 600}).status_code == 422
    assert client.get("/").status_code == 200


def test_sources_lists_every_document():
    from rag.api import app

    assert len(TestClient(app).get("/api/sources").json()) == 8
