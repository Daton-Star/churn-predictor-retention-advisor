"""GenAI layer #2: a small retrieval-augmented Q&A assistant over this
project's own generated analysis -- SQL query results, model evaluation
metrics, SHAP feature importances, and the README's narrative findings --
so a question like "which customer segment has the highest churn risk and
why?" gets a grounded answer instead of requiring someone to go read four
different files.

Run as a script to build/rebuild the index: `python src/rag_assistant.py`
(requires GEMINI_API_KEY -- see README). app.py imports answer_question()
directly and builds the index lazily on first use if it isn't cached yet.

WHY retrieval instead of just pasting the whole corpus into the prompt
------------------------------------------------------------------------
The corpus here is small enough (~12 chunks, a few KB) to fit in a single
Gemini prompt directly -- retrieval isn't *load-bearing* at this scale.
It's still implemented as real retrieval rather than "stuff everything
in" for two reasons: (1) it's the actual RAG pattern -- indexed chunks +
semantic search -- rather than a bigger system prompt wearing a RAG
label, and (2) keeping only the chunks relevant to *this* question in the
generation prompt is what lets the same design scale to a corpus that
doesn't fit in one prompt, which "paste it all in" never does.

WHY Gemini embeddings + brute-force cosine similarity, not a vector DB
------------------------------------------------------------------------
With ~12 chunks, a vector database (Pinecone/Chroma/FAISS) is
infrastructure with no payoff -- cosine similarity against a dozen
768-dim vectors is microseconds in numpy. The embeddings come from
Gemini's free-tier `text-embedding-004` model (via the same API key as
genai_advisor.py) so retrieval is genuinely semantic: a question phrased
as "who's most likely to leave" should still match chunks about "churn
risk," which plain keyword search would miss.

WHY the index is embedded once and cached
------------------------------------------------------------------------
The corpus only changes when the pipeline (data_prep -> train_model ->
explain -> the SQL scripts) is re-run, so re-embedding it on every
question would waste free-tier quota and add latency for nothing.
build_index() embeds it once and saves to disk; answering a question only
ever embeds the (much shorter) question itself.

WHY the assistant is told to say "I don't know" rather than guess
------------------------------------------------------------------------
Same guardrail philosophy as genai_advisor.py: a fluent but ungrounded
answer is worse than no answer for a business tool. The prompt instructs
Gemini to answer only from the retrieved context, and retrieve() itself
refuses to even call the model if the best match is below MIN_SIMILARITY
-- a low top score means the corpus probably doesn't cover the question at
all, so no amount of prompting the model to "be careful" would fix it.
"""

import json
import re

import joblib
import numpy as np

from config import (
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODELS_DIR,
    RAG_INDEX_PATH,
    README_PATH,
    SQL_QUERY_RESULTS_PATH,
)
from genai_advisor import get_api_key, get_model_name

EMBEDDING_MODEL = "text-embedding-004"
TOP_K = 4
# Cosine similarity is bounded [-1, 1]; 0.5 is a starting heuristic for
# "the corpus probably doesn't cover this question," not an empirically
# tuned threshold -- recalibrate against real query/answer pairs if this
# ever produces false "I don't know" refusals or ungrounded answers.
MIN_SIMILARITY = 0.5

# Only the analytical/narrative sections of the README are indexed --
# setup and deployment instructions aren't answers to analytical
# questions this assistant is meant to field.
README_SECTIONS_TO_INDEX = {
    "1. Business problem",
    "4. Key findings — top churn drivers",
    "6. Limitations and what I'd do with more time",
}

ANSWER_SYSTEM_INSTRUCTION = (
    "You are an analytics assistant answering questions about a customer "
    "churn prediction project, using ONLY the numbered context snippets "
    "provided below (pulled from the project's SQL analysis, model "
    "evaluation metrics, SHAP feature importances, and README findings). "
    "If the context doesn't contain enough information to answer, say so "
    "explicitly rather than guessing or using outside knowledge. Cite "
    "which snippet number(s) you used. Keep the answer to 3-4 sentences, "
    "written for a business audience, not a data scientist."
)


def _chunk_sql_results(path=SQL_QUERY_RESULTS_PATH):
    """One chunk per query in sql/QUERY_RESULTS.md, table only (the SQL
    text itself isn't useful context for a business question and keeping
    it out roughly halves the chunk size)."""
    text = path.read_text()
    sections = re.split(r"\n## `", text)[1:]
    chunks = []
    for section in sections:
        header, _, body = section.partition("`\n")
        fence_parts = body.split("```")
        table = fence_parts[2].strip() if len(fence_parts) >= 3 else body.strip()
        chunks.append({"source": f"sql/{header}", "text": f"SQL query result ({header}):\n{table}"})
    return chunks


def _chunk_model_comparison(path=METRICS_PATH):
    """One chunk per model's cross-validation summary, plus one chunk for
    the final selected model's held-out test performance."""
    data = json.loads(path.read_text())
    chunks = []
    for name, cv in data["cv_comparison"].items():
        text = (
            f"5-fold cross-validation results for {name} (churn prediction model), "
            "computed on the training split only:\n"
            f"- AUC-ROC: {cv['cv_roc_auc_mean']:.3f} (+/- {cv['cv_roc_auc_std']:.3f})\n"
            f"- F1: {cv['cv_f1_mean']:.3f}\n"
            f"- Precision: {cv['cv_precision_mean']:.3f}\n"
            f"- Recall: {cv['cv_recall_mean']:.3f}\n"
        )
        chunks.append({"source": f"model_comparison:{name}", "text": text})

    tm = data["test_metrics"]
    total_customers = data["train_rows"] + data["test_rows"]
    final_text = (
        f"Final selected churn model: {data['best_model']} (chosen for highest mean "
        f"CV AUC-ROC), evaluated once on a held-out test set of {data['test_rows']} "
        "customers never used in training or cross-validation:\n"
        f"- Test AUC-ROC: {tm['test_roc_auc']:.3f}\n"
        f"- Test F1: {tm['test_f1']:.3f}\n"
        f"- Test precision: {tm['test_precision']:.3f}\n"
        f"- Test recall: {tm['test_recall']:.3f}\n"
        f"- Confusion matrix [[TN, FP], [FN, TP]]: {data['confusion_matrix']}\n"
        f"- Overall churn rate across all {total_customers} customers: {data['churn_rate']:.1%}\n"
    )
    chunks.append({"source": "model_comparison:final", "text": final_text})
    return chunks


def _chunk_feature_importance(path=FEATURE_IMPORTANCE_PATH):
    """A single chunk ranking every feature by mean absolute SHAP value."""
    data = json.loads(path.read_text())
    ranked = sorted(data.items(), key=lambda kv: kv[1], reverse=True)
    lines = "\n".join(f"{i + 1}. {feature} (mean |SHAP| = {value:.4f})" for i, (feature, value) in enumerate(ranked))
    text = (
        "Global churn driver ranking, by mean absolute SHAP value across the test "
        f"set (higher = the model relies on this feature more heavily when "
        f"predicting churn):\n{lines}"
    )
    return [{"source": "feature_importance", "text": text}]


def _chunk_readme(path=README_PATH):
    """One chunk per whitelisted top-level README section."""
    text = path.read_text()
    sections = re.split(r"\n## ", text)
    chunks = []
    for section in sections[1:]:
        title, _, body = section.partition("\n")
        title = title.strip()
        if title not in README_SECTIONS_TO_INDEX:
            continue
        chunks.append({"source": f"README:{title}", "text": f"{title}\n{body.strip()}"})
    return chunks


def build_corpus():
    """Assemble every chunk from every source into one flat list."""
    return [
        *_chunk_sql_results(),
        *_chunk_model_comparison(),
        *_chunk_feature_importance(),
        *_chunk_readme(),
    ]


def embed_texts(texts, task_type):
    """Embed a list of strings with Gemini's embedding model, in one API call.

    task_type is "RETRIEVAL_DOCUMENT" for corpus chunks or "RETRIEVAL_QUERY"
    for the user's question -- Gemini's embedding API optimizes the vector
    differently for each role, which measurably improves asymmetric
    retrieval quality (a short question matching a longer document) over
    embedding both the same way.
    """
    from google import genai

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY as an environment "
            "variable or in .streamlit/secrets.toml -- see README."
        )
    client = genai.Client(api_key=api_key)
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=genai.types.EmbedContentConfig(task_type=task_type),
    )
    return np.array([e.values for e in response.embeddings], dtype=np.float32)


def build_index():
    """Embed the full corpus and cache it to disk. Requires a Gemini API key."""
    chunks = build_corpus()
    vectors = embed_texts([c["text"] for c in chunks], task_type="RETRIEVAL_DOCUMENT")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    index = {"chunks": chunks, "vectors": vectors}
    joblib.dump(index, RAG_INDEX_PATH)
    return index


def load_or_build_index():
    """Load the cached index if it exists, else build (and cache) it."""
    if RAG_INDEX_PATH.exists():
        return joblib.load(RAG_INDEX_PATH)
    return build_index()


def _cosine_similarity(query_vec, doc_vecs):
    """Pure numpy cosine similarity -- kept separate from embed_texts() so
    the ranking logic is testable without any network call."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
    return doc_norms @ query_norm


def retrieve(question, index, top_k=TOP_K):
    """Return the top_k chunks most similar to the question, each with its
    cosine similarity score attached, ranked highest first."""
    query_vec = embed_texts([question], task_type="RETRIEVAL_QUERY")[0]
    scores = _cosine_similarity(query_vec, index["vectors"])
    ranked_idx = np.argsort(-scores)[:top_k]
    return [{**index["chunks"][i], "score": float(scores[i])} for i in ranked_idx]


def _build_context_block(retrieved):
    return "\n\n".join(f"[{i + 1}] ({c['source']})\n{c['text']}" for i, c in enumerate(retrieved))


def answer_question(question, index=None, top_k=TOP_K):
    """Answer a free-form question grounded in the project's own analysis.

    Always returns {"answer": str | None, "sources": list[str], "error":
    str | None} -- callers (app.py, the CLI below) never need to catch an
    exception, matching genai_advisor.generate_customer_explanation()'s
    contract.
    """
    api_key = get_api_key()
    if not api_key:
        return {
            "answer": None,
            "sources": [],
            "error": (
                "No Gemini API key found. Set GEMINI_API_KEY as an environment "
                "variable or in .streamlit/secrets.toml -- see README for the "
                "free aistudio.google.com setup steps."
            ),
        }

    try:
        if index is None:
            index = load_or_build_index()

        retrieved = retrieve(question, index, top_k=top_k)
        if not retrieved or retrieved[0]["score"] < MIN_SIMILARITY:
            return {
                "answer": "I don't have enough information in this project's analysis to answer that.",
                "sources": [],
                "error": None,
            }

        from google import genai

        client = genai.Client(api_key=api_key)
        prompt = f"Context snippets:\n{_build_context_block(retrieved)}\n\nQuestion: {question}"
        response = client.models.generate_content(
            model=get_model_name(),
            contents=prompt,
            config=genai.types.GenerateContentConfig(system_instruction=ANSWER_SYSTEM_INSTRUCTION, temperature=0.25),
        )

        return {"answer": response.text.strip(), "sources": [c["source"] for c in retrieved], "error": None}
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI, don't crash it
        return {"answer": None, "sources": [], "error": f"Gemini call failed: {exc}"}


def main():
    print("Building RAG index from sql/, reports/, and README.md ...")
    index = build_index()
    print(f"Indexed {len(index['chunks'])} chunks -> {RAG_INDEX_PATH}")

    example = "Which customer segment has the highest churn risk and why?"
    print(f"\nExample question: {example}")
    result = answer_question(example, index=index)
    if result["error"]:
        print(f"Error: {result['error']}")
    else:
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")


if __name__ == "__main__":
    main()
