from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag import answer as answer_mod
from rag.api import app, get_retriever
from rag.ingest import chunk_html, chunk_markdown, load_directory
from rag.retriever import BM25Retriever, tokenize

DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


@pytest.fixture(autouse=True)
def extractive_mode(monkeypatch):
    monkeypatch.setenv("RAG_MODE", "extractive")


@pytest.fixture(scope="module")
def retriever():
    return BM25Retriever(load_directory(DOCS))


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
def test_relevant_section_is_retrieved(retriever, question, section):
    titles = [h.chunk.title for h in retriever.search(question)]
    assert any(section in t for t in titles), titles


def test_off_topic_question_is_declined(retriever):
    result = answer_mod.answer("Who is the CEO of Mhepo Telecom?", retriever)
    assert result.mode == "abstain"
    assert result.text.startswith("I don't know")


def test_extractive_answer_quotes_and_cites(retriever):
    result = answer_mod.answer("What merchant code do I use for EcoCash?", retriever)
    assert "112233" in result.text
    assert result.citations[0].source == "03-billing-and-payments.md"


def _fake_client(content, stop_reason="end_turn", calls=None):
    def create(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return SimpleNamespace(stop_reason=stop_reason, content=content)

    return lambda: SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


def test_claude_citations_map_back_to_chunks(retriever, monkeypatch):
    import anthropic

    calls = []
    content = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(
            type="text",
            text="Fibre Plus costs USD 85 a month.",
            citations=[SimpleNamespace(document_index=0, cited_text="| Fibre Plus | 50 / 25 Mbps | USD 85 |")],
        ),
    ]
    monkeypatch.setenv("RAG_MODE", "claude")
    monkeypatch.setattr(anthropic, "Anthropic", _fake_client(content, calls=calls))

    result = answer_mod.answer("How much is Fibre Plus per month?", retriever)

    assert result.mode == "claude"
    assert result.text == "Fibre Plus costs USD 85 a month."
    assert result.citations[0].quote.startswith("| Fibre Plus")
    sent = calls[0]
    assert sent["model"] == answer_mod.MODEL
    assert all(block["citations"] == {"enabled": True} for block in sent["messages"][0]["content"][:-1])


def test_claude_saying_it_does_not_know_becomes_abstain(retriever, monkeypatch):
    import anthropic

    content = [SimpleNamespace(type="text", text="I don't know. I couldn't find that in Mhepo's documents.", citations=None)]
    monkeypatch.setenv("RAG_MODE", "claude")
    monkeypatch.setattr(anthropic, "Anthropic", _fake_client(content))
    assert answer_mod.answer("How much is Fibre Plus per month?", retriever).mode == "abstain"


def test_claude_failure_falls_back_to_quotes(retriever, monkeypatch):
    import anthropic

    def broken():
        raise RuntimeError("network down")

    monkeypatch.setenv("RAG_MODE", "claude")
    monkeypatch.setattr(anthropic, "Anthropic", broken)
    result = answer_mod.answer("What merchant code do I use for EcoCash?", retriever)
    assert result.mode == "extractive" and result.note


def test_api_ask_health_and_validation():
    client = TestClient(app)
    assert client.get("/api/health").json()["documents"] == 8
    response = client.post("/api/ask", json={"question": "How do I block out-of-bundle data?"})
    assert response.status_code == 200
    assert "*300*9#" in response.json()["text"]
    assert client.post("/api/ask", json={"question": "x" * 600}).status_code == 422
    assert client.get("/").status_code == 200


def test_sources_lists_every_document():
    assert len(get_retriever().chunks) > 20
    assert len(TestClient(app).get("/api/sources").json()) == 8


@pytest.mark.parametrize("question", ["Does Mhepo offer satellite TV packages?", "What is today's exchange rate for ZiG?"])
def test_single_shared_word_is_not_enough_to_answer(retriever, question):
    assert retriever.search(question) == []


def test_haiku_requests_skip_server_side_fallback(retriever, monkeypatch):
    import anthropic

    calls = []
    content = [SimpleNamespace(type="text", text="USD 85 a month.", citations=[])]
    monkeypatch.setenv("RAG_MODE", "claude")
    monkeypatch.setattr(answer_mod, "MODEL", "claude-haiku-5-5")
    monkeypatch.setattr(anthropic, "Anthropic", _fake_client(content, calls=calls))
    answer_mod.answer("How much is Fibre Plus per month?", retriever)
    assert "fallbacks" not in calls[0] and "betas" not in calls[0]
