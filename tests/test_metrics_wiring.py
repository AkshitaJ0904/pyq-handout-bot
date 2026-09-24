"""
Contract tests for the live metric panel.

These stub Ollama and the embedder, so they run without a model pulled and
without `ollama serve` -- the point is to verify the wiring (routes return
all eight metrics, ground truth unlocks the three that need it, RAG-off
correctly reports grounding as unmeasurable), not to test model quality.

Run:  python3 -m pytest tests/ -q
"""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402
from rag import live_metrics  # noqa: E402

QUALITY = {"correctness", "relevance", "retrieval_quality", "hallucination", "test_pass"}
PERFORMANCE = {"latency", "tokens", "resources"}


class FakeEmbedder:
    """Deterministic stand-in for all-MiniLM: hashes text to a fixed vector
    so cosine similarities are stable across runs."""

    def encode(self, texts, convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=32)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


FAKE_OLLAMA_RESPONSE = {
    "response": "Deadlock needs mutual exclusion, hold and wait, no preemption and circular wait.",
    "total_duration": 8_400_000_000,
    "eval_duration": 6_000_000_000,
    "prompt_eval_count": 312,
    "eval_count": 96,
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "call_ollama", lambda *a, **k: dict(FAKE_OLLAMA_RESPONSE))
    monkeypatch.setattr(app_module, "get_embedder", lambda: FakeEmbedder())
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def _by_key(metrics):
    return {m["key"]: m for m in metrics}


def test_call_ollama_keeps_token_counts():
    """The regression this whole change turns on: call_ollama used to return
    only the answer string, discarding the token counts in the same payload."""
    with patch("app.requests.post") as post:
        post.return_value.json.return_value = dict(FAKE_OLLAMA_RESPONSE)
        post.return_value.raise_for_status = lambda: None
        payload = app_module.call_ollama("hi")
    assert payload["prompt_eval_count"] == 312
    assert payload["eval_count"] == 96


def test_ask_returns_all_eight_metrics(client):
    r = client.post("/ask", json={"question": "What causes deadlock?", "rag": False})
    assert r.status_code == 200
    metrics = _by_key(r.get_json()["metrics"])
    assert set(metrics) == QUALITY | PERFORMANCE


def test_free_question_marks_ground_truth_metrics_unavailable(client):
    r = client.post("/ask", json={"question": "Anything at all?", "rag": False})
    metrics = _by_key(r.get_json()["metrics"])
    for key in ("correctness", "retrieval_quality", "test_pass"):
        assert metrics[key]["available"] is False
        assert metrics[key]["display"] == "N/A"
        assert metrics[key]["reason"], f"{key} must explain why it is N/A"


def test_performance_metrics_always_available(client):
    r = client.post("/ask", json={"question": "Anything at all?", "rag": False})
    metrics = _by_key(r.get_json()["metrics"])
    assert metrics["latency"]["available"] is True
    assert metrics["tokens"]["available"] is True
    assert metrics["tokens"]["value"] == 312 + 96


def test_labelled_question_unlocks_correctness(client):
    """Q01's expected keypoints are exactly the four the fake answer names,
    so correctness should come back fully available and scoring 1.0."""
    gt = app_module.find_eval_question("Q01")
    assert gt is not None, "Q01 must exist in eval_dataset.json"
    r = client.post("/ask", json={"question": gt["question"], "rag": False, "question_id": "Q01"})
    correctness = _by_key(r.get_json()["metrics"])["correctness"]
    assert correctness["available"] is True
    assert correctness["value"] == 1.0


def test_rag_off_cannot_measure_grounding(client):
    """With no knowledge base there is no context to ground against, so
    hallucination rate must report unmeasurable rather than 0."""
    r = client.post("/ask", json={"question": "What causes deadlock?", "rag": False})
    hall = _by_key(r.get_json()["metrics"])["hallucination"]
    assert hall["available"] is False
    assert hall["value"] is None


def test_compare_models_attaches_metrics_per_model(client):
    r = client.post("/compare_models", json={
        "question": "What causes deadlock?",
        "models": ["codellama:7b", "starcoder2:3b", "deepseek-coder:1.3b"],
    })
    results = r.get_json()["results"]
    assert len(results) == 3
    for row in results:
        assert set(_by_key(row["metrics"])) == QUALITY | PERFORMANCE


def test_eval_questions_endpoint_serves_the_dataset(client):
    qs = client.get("/eval_questions").get_json()["questions"]
    assert len(qs) == 28
    assert {"id", "category", "question", "has_code_test"} <= set(qs[0])


def test_unavailable_metric_never_reports_a_number(monkeypatch):
    """The core honesty property: an unavailable metric carries no value."""
    monkeypatch.setattr(app_module, "get_embedder", lambda: FakeEmbedder())
    metrics = live_metrics.compute(
        question="q", answer="a", context="", sources=[],
        response_json=dict(FAKE_OLLAMA_RESPONSE),
        resources={"peak_ollama_rss_mb": None, "mean_cpu_percent": None},
        wall_s=1.0, ground_truth=None,
    )
    for m in metrics:
        if not m["available"]:
            assert m["value"] is None
            assert m["display"] == "N/A"
