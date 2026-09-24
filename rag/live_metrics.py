"""
Live metric instrumentation for the web UI.

Week 4 computed the evaluation metrics offline (eval/run_eval.py) against a
fixed 28-question dataset. This module exposes those same implementations --
eval/metrics.py is imported and reused unchanged, not reimplemented -- on a
per-request basis, so the UI can show the metric panel next to every answer.

Three of the eight metrics need ground truth that a free-typed question
simply does not have:

  * Correctness      needs hand-picked expected keypoints
  * Retrieval quality needs human-labelled relevant source files
  * Test-pass rate   needs an assertion harness for the generated code

Rather than invent a number for those, they are returned with
available=False and the reason why, and the UI greys them out. Selecting one
of the 28 labelled questions from the UI dropdown supplies the ground truth
and lights all eight up. Reporting "undefined" is the honest answer here --
a confident correctness score for a question with no reference answer would
be meaningless (same reasoning as the GPU non-measurement in REPORT.md).
"""
from eval.metrics import (
    hallucination_rate,
    keypoint_coverage,
    ollama_stats,
    retrieval_precision_at_k,
    run_code_test,
    semantic_relevance,
)

# The metric panel's fixed running order: five quality metrics, then three
# performance metrics. The UI renders whatever this returns, so adding or
# removing a metric here is the only change needed on either side.
QUALITY_KEYS = ["correctness", "relevance", "retrieval_quality", "hallucination", "test_pass"]
PERFORMANCE_KEYS = ["latency", "tokens", "resources"]


def _pct(x):
    return f"{round(x * 100)}%"


def _component(key, label, group, value=None, display=None, detail=None, reason=None):
    """One metric tile. `reason` non-None means the metric could not be
    computed for this question, and the UI shows it greyed out with the
    reason as a tooltip instead of a value."""
    return {
        "key": key,
        "label": label,
        "group": group,
        "value": value,
        "display": display if reason is None else "N/A",
        "detail": detail,
        "available": reason is None,
        "reason": reason,
    }


def compute(question, answer, context, sources, response_json, resources,
            wall_s, ground_truth=None):
    """Assemble all eight metric components for a single generation.

    `ground_truth` is one entry from eval_dataset.json when the user picked a
    labelled question, or None for free-typed questions. `context` is the
    retrieved context string ("" when RAG is off), `sources` the retrieved
    chunk descriptors, `resources` a ResourceSampler.summary() dict.
    """
    from app import get_embedder  # deferred: avoids a circular import at module load

    gt = ground_truth or {}
    embedder = get_embedder()
    out = []

    # ---- Quality -----------------------------------------------------
    expected_keypoints = gt.get("expected_keypoints")
    if expected_keypoints:
        cov = keypoint_coverage(answer, expected_keypoints)
        hits = sum(1 for kp in expected_keypoints if kp.lower() in answer.lower())
        out.append(_component(
            "correctness", "Correctness", "quality", cov, _pct(cov),
            f"{hits} of {len(expected_keypoints)} expected keypoints present",
        ))
    else:
        out.append(_component(
            "correctness", "Correctness", "quality",
            reason="Needs a labelled question - correctness is measured against "
                   "hand-picked expected keypoints, which only the 28 evaluation "
                   "questions have.",
        ))

    if embedder is not None:
        rel = semantic_relevance(answer, question, embedder)
        out.append(_component(
            "relevance", "Relevance", "quality", rel, f"{rel:.2f}",
            "cosine similarity, question vs answer embedding",
        ))
    else:
        out.append(_component("relevance", "Relevance", "quality",
                              reason="Embedding model unavailable."))

    expected_files = gt.get("expected_source_files")
    retrieved_files = [s["source_file"] for s in (sources or [])]
    if expected_files and retrieved_files:
        prec = retrieval_precision_at_k(retrieved_files, expected_files)
        hits = sum(1 for f in retrieved_files if f in expected_files)
        out.append(_component(
            "retrieval_quality", "Retrieval quality", "quality", prec, _pct(prec),
            f"precision@{len(retrieved_files)} - {hits} of {len(retrieved_files)} "
            f"chunks from a labelled-relevant file",
        ))
    elif not retrieved_files:
        out.append(_component(
            "retrieval_quality", "Retrieval quality", "quality",
            reason="Nothing retrieved - the knowledge base is switched off for "
                   "this answer.",
        ))
    else:
        out.append(_component(
            "retrieval_quality", "Retrieval quality", "quality",
            reason="Needs a labelled question - precision@k is measured against "
                   "human-labelled relevant source files.",
        ))

    if context and context.strip() and embedder is not None:
        hall = hallucination_rate(answer, context, embedder)
        if hall is None:
            out.append(_component("hallucination", "Hallucination rate", "quality",
                                  reason="No usable context to ground against."))
        else:
            out.append(_component(
                "hallucination", "Hallucination rate", "quality", hall, _pct(hall),
                "share of answer sentences not grounded in retrieved context "
                "(cosine < 0.35)",
            ))
    else:
        out.append(_component(
            "hallucination", "Hallucination rate", "quality",
            reason="Grounding is measured against retrieved context, and this "
                   "answer had none - an answer with no knowledge base cannot "
                   "be checked for grounding at all.",
        ))

    code_test = gt.get("code_test")
    if code_test:
        res = run_code_test(answer, code_test)
        if res["ran"] is False:
            out.append(_component(
                "test_pass", "Test-pass", "quality", 0.0, "no code",
                "could not extract a runnable function from the answer",
            ))
        else:
            passed = bool(res["passed"])
            out.append(_component(
                "test_pass", "Test-pass", "quality", 1.0 if passed else 0.0,
                "pass" if passed else "fail",
                "generated function run against the fixed assertion harness"
                + ("" if passed else f" - {(res.get('error') or '')[:160]}"),
            ))
    else:
        out.append(_component(
            "test_pass", "Test-pass", "quality",
            reason="Only defined for code-generation questions that ship an "
                   "assertion harness.",
        ))

    # ---- Performance -------------------------------------------------
    stats = ollama_stats(response_json or {})
    reported = stats.get("total_duration_s") or 0
    out.append(_component(
        "latency", "Latency", "performance", wall_s, f"{wall_s:.1f}s",
        f"wall clock; Ollama reports {reported:.1f}s of it as generation"
        if reported else "wall clock, end to end",
    ))

    tps = stats.get("tokens_per_sec")
    total_tokens = (stats.get("prompt_tokens") or 0) + (stats.get("response_tokens") or 0)
    out.append(_component(
        "tokens", "Token usage", "performance", total_tokens, f"{total_tokens} tok",
        f"{stats.get('prompt_tokens', 0)} prompt + {stats.get('response_tokens', 0)} "
        f"response" + (f", {tps} tok/s" if tps else ""),
    ))

    rss = (resources or {}).get("peak_ollama_rss_mb")
    cpu = (resources or {}).get("mean_cpu_percent")
    if rss:
        out.append(_component(
            "resources", "Resources", "performance", rss, f"{rss / 1024:.1f} GB",
            f"peak Ollama RSS; mean CPU {cpu}%. GPU not measured - Apple Silicon "
            f"exposes no GPU counter without sudo (powermetrics needs root).",
        ))
    else:
        out.append(_component(
            "resources", "Resources", "performance",
            reason="No Ollama process sampled during this call.",
        ))

    return out
