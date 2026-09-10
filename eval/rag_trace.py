"""
Week 4, Exercise 5 - trace the existing RAG pipeline (rag.retrieval,
unchanged) question by question: QUESTION -> RETRIEVED CONTEXT -> LLM
RESPONSE, so retrieval quality can be inspected directly instead of only
looking at final answers.

For each traced question this records every retrieved chunk (source file,
doc_type, similarity score, and the text itself) plus the generated answer,
so a human can label each example as: relevant retrieval / irrelevant
retrieval / missing information / correct answer / hallucination despite
retrieved context. REPORT.md Exercise 5 fills in those labels and the
retrieval-quality -> context-quality -> response-quality discussion.

Usage:
    venv/bin/python3 eval/rag_trace.py --model codellama:7b
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.retrieval import Retriever, build_context  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"

RAG_SYSTEM_PREFIX = (
    "You are a study assistant for a college course. Use ONLY the context "
    "below, pulled from the course syllabus and past exam papers (PYQs), to "
    "answer the question. If the context shows a topic appears often in "
    "PYQs but is barely covered in the syllabus, point that out explicitly. "
    "If the context doesn't contain the answer, say so instead of guessing.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

# A curated subset of eval_dataset.json questions chosen to cover the
# interesting retrieval cases: exact-match, gap-analysis (needs BOTH
# syllabus + pyq chunks), cross-document synthesis, and out-of-KB probes
# that should retrieve nothing relevant.
TRACE_QUESTION_IDS = ["Q01", "Q08", "Q09", "Q14", "Q21", "Q22"]


def trace(model, question, retriever, top_k=4):
    chunks = retriever.search(question, top_k=top_k)
    context = build_context(chunks)
    prompt = RAG_SYSTEM_PREFIX.format(context=context, question=question)

    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"num_predict": 200, "num_ctx": 1024}},
        timeout=1800,
    )
    resp.raise_for_status()
    answer = resp.json().get("response", "")

    return {
        "question": question,
        "retrieved_chunks": [
            {"source_file": c["source_file"], "doc_type": c["doc_type"],
             "score": round(c["score"], 3), "text": c["text"]}
            for c in chunks
        ],
        "answer": answer,
        # left blank for the human analysis pass described in REPORT.md
        "label_retrieval": None,   # "relevant" | "irrelevant" | "missing_info"
        "label_answer": None,      # "correct" | "hallucination" | "correct_refusal"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="codellama:7b")
    parser.add_argument("--ids", nargs="+", default=TRACE_QUESTION_IDS)
    args = parser.parse_args()

    dataset = json.loads((Path(__file__).parent / "eval_dataset.json").read_text())
    questions_by_id = {q["id"]: q["question"] for q in dataset["questions"]}

    print("Loading retriever ...")
    retriever = Retriever()

    traces = []
    for qid in args.ids:
        question = questions_by_id[qid]
        print(f"Tracing {qid}: {question[:70]}...")
        t = trace(args.model, question, retriever)
        t["id"] = qid
        traces.append(t)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"rag_trace_{args.model.replace(':', '_')}_{ts}.json"
    out_path.write_text(json.dumps(traces, indent=2))
    print(f"\nWrote {out_path}")

    for t in traces:
        print(f"\n=== {t['id']}: {t['question']} ===")
        for c in t["retrieved_chunks"]:
            print(f"  [{c['score']:.3f}] {c['doc_type']:8s} {c['source_file']}")
        print(f"  ANSWER: {t['answer'][:200]}")


if __name__ == "__main__":
    main()
