"""
Week 4, Exercises 1+3 - run the Week 3 app's exact RAG logic (rag.retrieval
+ the same RAG_SYSTEM_PREFIX prompt template as app.py) against every
question in eval_dataset.json, once per model, swapping ONLY the `model`
field sent to Ollama. Everything else the assignment says to hold constant
(application, prompts, questions, knowledge base, top_k) is untouched.

Usage:
    venv/bin/python3 eval/run_eval.py --models codellama:7b starcoder2:3b qwen3-coder:latest --sample 5
    venv/bin/python3 eval/run_eval.py                     # full 28-question run, all default models

Writes eval/results/run_<timestamp>.json (raw, every field) and
eval/results/run_<timestamp>_summary.md (aggregated table).
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.retrieval import Retriever, build_context  # noqa: E402
from eval import metrics  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODELS = ["codellama:7b", "starcoder2:3b", "qwen3-coder:latest"]
TOP_K = 4
NUM_PREDICT = 120

RAG_SYSTEM_PREFIX = (
    "You are a study assistant for a college course. Use ONLY the context "
    "below, pulled from the course syllabus and past exam papers (PYQs), to "
    "answer the question. If the context shows a topic appears often in "
    "PYQs but is barely covered in the syllabus, point that out explicitly. "
    "If the context doesn't contain the answer, say so instead of guessing.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


def call_ollama(model: str, prompt: str, num_predict: int = NUM_PREDICT) -> dict:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict, "num_ctx": 1024},
        },
        timeout=1800,
    )
    resp.raise_for_status()
    return resp.json()


def run_one(model, q, retriever, embedder):
    question = q["question"]
    use_rag = q.get("use_rag", True)

    context = ""
    retrieved_sources = []
    if use_rag:
        chunks = retriever.search(question, top_k=TOP_K)
        retrieved_sources = [c["source_file"] for c in chunks]
        context = build_context(chunks)
        prompt = RAG_SYSTEM_PREFIX.format(context=context, question=question)
    else:
        prompt = question

    # Code-generation answers need more room than a short factual answer --
    # the default budget truncates most of them mid-function (observed in
    # the first full run: 0/2 automated code questions even produced an
    # extractable function at num_predict=120).
    num_predict = 400 if q["category"] == "code_generation" else NUM_PREDICT

    with metrics.ResourceSampler() as sampler:
        t0 = time.time()
        try:
            raw = call_ollama(model, prompt, num_predict=num_predict)
        except requests.exceptions.RequestException as e:
            return {"id": q["id"], "category": q["category"], "model": model, "error": str(e)}
        wall_clock_s = time.time() - t0

    answer = raw.get("response", "")
    perf = metrics.ollama_stats(raw)
    perf["wall_clock_s"] = round(wall_clock_s, 3)
    resource = sampler.summary()

    record = {
        "id": q["id"], "category": q["category"], "model": model,
        "question": question, "answer": answer,
        "retrieved_sources": retrieved_sources,
        "performance": {**perf, **resource},
        "quality": {},
    }

    record["quality"]["keypoint_coverage"] = metrics.keypoint_coverage(answer, q.get("expected_keypoints", []))
    record["quality"]["semantic_relevance"] = metrics.semantic_relevance(answer, question, embedder)
    if q.get("expected_source_files"):
        record["quality"]["retrieval_precision_at_k"] = metrics.retrieval_precision_at_k(
            retrieved_sources, q["expected_source_files"])
    else:
        record["quality"]["no_result_leak_rate"] = metrics.no_result_leak_rate(
            retrieved_sources, q.get("expected_source_files", []))
    if use_rag:
        record["quality"]["hallucination_rate"] = metrics.hallucination_rate(answer, context, embedder)

    if q.get("code_test") is not None:
        record["quality"]["code_test"] = metrics.run_code_test(answer, q["code_test"])
    elif q["category"] == "code_generation":
        record["quality"]["code_test"] = {"ran": None, "passed": None, "error": "manual review question"}

    return record


def summarize(records):
    by_model = defaultdict(list)
    for r in records:
        if "error" not in r:
            by_model[r["model"]].append(r)

    lines = ["# Week 4 Eval Summary", ""]
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Questions per model: {len(records) // max(len(by_model), 1)}")
    lines.append("")
    lines.append("| Model | Avg latency (s) | Avg tok/s | Avg peak RSS (MB) | Avg keypoint coverage | Avg relevance | Avg hallucination rate | Code test pass rate |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for model, recs in by_model.items():
        lat = _avg(r["performance"]["wall_clock_s"] for r in recs)
        # Exclude degenerate throughput samples: Ollama's eval_duration can
        # round to ~0 when the model emits only 1-2 tokens (e.g. an early
        # stop), which turns response_tokens/eval_duration_s into a
        # meaningless multi-million tok/s outlier that dominates a plain
        # mean. Require a handful of tokens for a sample to count.
        tps = _avg(r["performance"]["tokens_per_sec"] for r in recs
                   if r["performance"].get("tokens_per_sec") and r["performance"].get("response_tokens", 0) >= 5)
        rss = _avg(r["performance"]["peak_ollama_rss_mb"] for r in recs if r["performance"].get("peak_ollama_rss_mb"))
        kp = _avg(r["quality"]["keypoint_coverage"] for r in recs if r["quality"].get("keypoint_coverage") is not None)
        rel = _avg(r["quality"]["semantic_relevance"] for r in recs if r["quality"].get("semantic_relevance") is not None)
        hall = _avg(r["quality"]["hallucination_rate"] for r in recs if r["quality"].get("hallucination_rate") is not None)
        tests = [r["quality"]["code_test"] for r in recs if r["quality"].get("code_test", {}).get("ran")]
        pass_rate = (sum(1 for t in tests if t["passed"]) / len(tests)) if tests else None

        lines.append(
            f"| {model} | {_fmt(lat)} | {_fmt(tps)} | {_fmt(rss)} | {_fmt(kp)} | {_fmt(rel)} | {_fmt(hall)} | {_fmt(pass_rate)} |"
        )

    lines.append("")
    lines.append("## By category (keypoint coverage)")
    lines.append("")
    cats = sorted({r["category"] for r in records})
    header = "| Category | " + " | ".join(by_model.keys()) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(by_model))
    for cat in cats:
        row = [cat]
        for model, recs in by_model.items():
            vals = [r["quality"]["keypoint_coverage"] for r in recs
                    if r["category"] == cat and r["quality"].get("keypoint_coverage") is not None]
            row.append(_fmt(_avg(vals)))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _avg(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else None


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else ("-" if v is None else str(v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--sample", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--ids", nargs="+", default=None, help="only run these question ids")
    args = parser.parse_args()

    dataset = json.loads((Path(__file__).parent / "eval_dataset.json").read_text())
    questions = dataset["questions"]
    if args.ids:
        questions = [q for q in questions if q["id"] in args.ids]
    elif args.sample:
        questions = questions[: args.sample]

    print(f"Loading retriever and embedder ...")
    retriever = Retriever()
    embedder = retriever.model  # reuse the same all-MiniLM-L6-v2 instance

    results = []
    total = len(args.models) * len(questions)
    n = 0
    for model in args.models:
        for q in questions:
            n += 1
            print(f"[{n}/{total}] {model} :: {q['id']} {q['category']} ...", flush=True)
            rec = run_one(model, q, retriever, embedder)
            if "error" in rec:
                print(f"    ERROR: {rec['error']}")
            results.append(rec)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    raw_path = out_dir / f"run_{ts}.json"
    raw_path.write_text(json.dumps(results, indent=2))

    summary_md = summarize(results)
    summary_path = out_dir / f"run_{ts}_summary.md"
    summary_path.write_text(summary_md)

    print(f"\nWrote {raw_path}")
    print(f"Wrote {summary_path}")
    print("\n" + summary_md)


if __name__ == "__main__":
    main()
