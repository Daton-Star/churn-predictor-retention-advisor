"""Unit tests for the RAG assistant: chunking, cosine-similarity ranking,
and the "don't answer ungrounded" guardrail. No test here calls the real
Gemini API -- embeddings and generation are mocked/monkeypatched, the same
approach as test_genai_advisor.py.

Chunking is tested against the real, committed sql/QUERY_RESULTS.md,
reports/*.json, and README.md rather than synthetic fixtures: those files
are checked into the repo specifically so downstream consumers (this
assistant, and these tests) don't depend on the raw 45MB dataset.
"""

import sys
import types

import numpy as np
import pytest

import rag_assistant as rag


@pytest.fixture(autouse=True)
def cleanup_fake_module():
    yield
    sys.modules.pop("google.generativeai", None)


def install_fake_genai(response_text, expect_no_call=False):
    class FakeResponse:
        text = response_text

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate_content(self, *args, **kwargs):
            if expect_no_call:
                raise AssertionError("generate_content should not be reached")
            return FakeResponse()

    fake_module = types.ModuleType("google.generativeai")
    fake_module.configure = lambda **kwargs: None
    fake_module.GenerativeModel = FakeModel
    fake_module.GenerationConfig = lambda **kwargs: None
    sys.modules["google.generativeai"] = fake_module


# --- Chunking against the real committed artifacts -------------------------


def test_chunk_sql_results_one_per_query_no_sql_code():
    chunks = rag._chunk_sql_results()
    assert len(chunks) == 4
    for chunk in chunks:
        assert chunk["source"].startswith("sql/")
        assert "|" in chunk["text"]  # the markdown table survived
        assert "SELECT" not in chunk["text"].upper()  # the SQL code itself was stripped


def test_chunk_model_comparison_has_all_models_plus_final():
    chunks = rag._chunk_model_comparison()
    sources = {c["source"] for c in chunks}
    assert sources == {
        "model_comparison:logistic_regression",
        "model_comparison:random_forest",
        "model_comparison:xgboost",
        "model_comparison:final",
    }
    final = next(c for c in chunks if c["source"] == "model_comparison:final")
    assert "Test AUC-ROC" in final["text"]


def test_chunk_feature_importance_ranked_descending():
    chunks = rag._chunk_feature_importance()
    assert len(chunks) == 1
    text = chunks[0]["text"]
    # avg_days_between_purchases is the top global driver -- see reports/feature_importance.json
    assert text.index("1. avg_days_between_purchases") < text.index("2.")


def test_chunk_readme_only_whitelisted_sections():
    chunks = rag._chunk_readme()
    sources = {c["source"] for c in chunks}
    assert sources == {f"README:{title}" for title in rag.README_SECTIONS_TO_INDEX}
    # setup/deployment instructions shouldn't leak in -- this assistant
    # answers analytical questions, not "how do I run this"
    assert not any("Setup and run" in s for s in sources)


def test_build_corpus_combines_every_source_with_unique_keys():
    corpus = rag.build_corpus()
    assert len(corpus) == 4 + 4 + 1 + 3  # sql + model_comparison + feature_importance + readme
    assert len({c["source"] for c in corpus}) == len(corpus)  # no duplicate source keys


# --- Cosine similarity (pure math, no network) ------------------------------


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert rag._cosine_similarity(v, np.array([v]))[0] == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    query = np.array([1.0, 0.0])
    docs = np.array([[0.0, 1.0]])
    assert rag._cosine_similarity(query, docs)[0] == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_ranks_most_similar_first():
    query = np.array([1.0, 0.0, 0.0])
    docs = np.array(
        [
            [0.0, 1.0, 0.0],  # orthogonal -> ~0
            [0.9, 0.1, 0.0],  # close match -> high
            [-1.0, 0.0, 0.0],  # opposite -> -1
        ]
    )
    scores = rag._cosine_similarity(query, docs)
    assert np.argmax(scores) == 1


# --- retrieve() with a mocked embedder --------------------------------------


def test_retrieve_ranks_by_similarity(monkeypatch):
    fake_index = {
        "chunks": [{"source": "a", "text": "A"}, {"source": "b", "text": "B"}, {"source": "c", "text": "C"}],
        "vectors": np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]),
    }

    def fake_embed_texts(texts, task_type):
        assert task_type == "retrieval_query"
        return np.array([[1.0, 0.0]])  # matches chunk "a" exactly

    monkeypatch.setattr(rag, "embed_texts", fake_embed_texts)
    results = rag.retrieve("does not matter", fake_index, top_k=2)

    assert len(results) == 2
    assert results[0]["source"] == "a"
    assert results[0]["score"] == pytest.approx(1.0)


# --- answer_question() guardrails -------------------------------------------


def test_answer_question_no_api_key(monkeypatch):
    monkeypatch.setattr(rag, "get_api_key", lambda: None)
    result = rag.answer_question("anything")
    assert result["answer"] is None
    assert "API key" in result["error"]


def test_answer_question_below_similarity_threshold_skips_generation(monkeypatch):
    monkeypatch.setattr(rag, "get_api_key", lambda: "fake-key")
    monkeypatch.setattr(
        rag, "retrieve", lambda q, index, top_k=rag.TOP_K: [{"source": "x", "text": "irrelevant", "score": 0.1}]
    )
    install_fake_genai(response_text="should never be used", expect_no_call=True)

    result = rag.answer_question("irrelevant question", index={"chunks": [], "vectors": np.zeros((0, 2))})

    assert result["error"] is None
    assert "don't have enough information" in result["answer"]
    assert result["sources"] == []


def test_answer_question_success(monkeypatch):
    monkeypatch.setattr(rag, "get_api_key", lambda: "fake-key")
    monkeypatch.setattr(rag, "get_model_name", lambda: "gemini-2.5-flash")
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda q, index, top_k=rag.TOP_K: [
            {"source": "sql/02_churn_rate_by_country.sql", "text": "Netherlands has the highest churn rate.", "score": 0.9}
        ],
    )
    install_fake_genai(response_text="Netherlands has the highest churn rate at 72.7%. [1]")

    result = rag.answer_question("which segment churns most", index={"chunks": [], "vectors": np.zeros((0, 2))})

    assert result["error"] is None
    assert "Netherlands" in result["answer"]
    assert result["sources"] == ["sql/02_churn_rate_by_country.sql"]


def test_answer_question_generation_failure_is_caught(monkeypatch):
    monkeypatch.setattr(rag, "get_api_key", lambda: "fake-key")
    monkeypatch.setattr(rag, "get_model_name", lambda: "gemini-2.5-flash")
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda q, index, top_k=rag.TOP_K: [{"source": "x", "text": "relevant enough", "score": 0.9}],
    )

    class BoomModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate_content(self, *args, **kwargs):
            raise RuntimeError("network exploded")

    fake_module = types.ModuleType("google.generativeai")
    fake_module.configure = lambda **k: None
    fake_module.GenerativeModel = BoomModel
    fake_module.GenerationConfig = lambda **k: None
    sys.modules["google.generativeai"] = fake_module

    result = rag.answer_question("anything", index={"chunks": [], "vectors": np.zeros((0, 2))})

    assert result["answer"] is None
    assert "Gemini call failed" in result["error"]
