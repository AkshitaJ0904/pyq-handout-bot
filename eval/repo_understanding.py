"""
Week 4, Exercise 6 - repository-level code understanding, using the SAME
LLM + RAG approach as the rest of the app (chunk -> embed -> cosine
similarity retrieve -> stuff into prompt), pointed at this repo's own
source files instead of the syllabus/PYQ knowledge base.

This deliberately reuses the existing, naive word-window chunker
(ingestion.build_index.chunk_text) rather than anything AST-aware, because
the point of the exercise is to see what a generic text-RAG system CAN'T
do for code before Sourcegraph-style repository-level tooling is
introduced next week -- not to build that tooling now.

Usage:
    venv/bin/python3 eval/repo_understanding.py --model codellama:7b
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingestion.build_index import chunk_text  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

INCLUDE_GLOBS = ["*.py", "**/*.py", "*.md", "docker/*.yml", "docker/Dockerfile.*", "scripts/*.sh"]
EXCLUDE_DIRS = {"venv", "__pycache__", "eval", ".git", "static", "templates", "ingestion/index"}

CODE_SYSTEM_PREFIX = (
    "You are a code assistant answering questions about a software "
    "repository. Use ONLY the code excerpts below, each tagged with its "
    "file path, to answer the question. If the excerpts don't contain "
    "enough information to answer confidently, say so explicitly instead "
    "of guessing.\n\n"
    "Code context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

REPO_QUESTIONS = [
    "Which files are involved in generating and storing embeddings for the knowledge base?",
    "Which service does the orchestrator call first when handling a request to /ask -- the retrieval service or the LLM service?",
    "What happens after a user uploads a PDF through the web UI, step by step?",
    "Which files would need to change if the chunk size used during ingestion was increased from 120 to 300 words?",
    "Which component performs the cosine similarity search, and is that logic duplicated anywhere in the codebase?",
    "Are there any automated test files in this repository?",
    "Which environment variable controls which Ollama model the LLM service uses?",
    "What is the difference between how app.py and services/orchestrator.py handle file uploads?",
    "Which Dockerfile builds the retrieval service, and what port does that service listen on?",
    "If the retrieval service goes down, which other service's requests will fail, and how would a user observe that failure?",
]


def collect_repo_files():
    seen = set()
    files = []
    for pattern in INCLUDE_GLOBS:
        for path in ROOT.glob(pattern):
            if path in seen or not path.is_file():
                continue
            if any(part in EXCLUDE_DIRS for part in path.relative_to(ROOT).parts):
                continue
            seen.add(path)
            files.append(path)
    return sorted(files)


def build_code_index(model):
    records = []
    for path in collect_repo_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        pieces = chunk_text(text, size=120, overlap=30)
        rel = str(path.relative_to(ROOT))
        for i, piece in enumerate(pieces):
            records.append({"file": rel, "chunk_index": i, "text": piece})
    print(f"Indexed {len(records)} chunks from {len(collect_repo_files())} repo files.")
    embeddings = model.encode([r["text"] for r in records], convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    embeddings = (embeddings / norms).astype(np.float32)
    return records, embeddings


def search(question, model, records, embeddings, top_k=4):
    qvec = model.encode([question], convert_to_numpy=True)[0]
    norm = np.linalg.norm(qvec)
    if norm > 0:
        qvec = qvec / norm
    scores = embeddings @ qvec
    top_idx = np.argsort(-scores)[:top_k]
    return [{**records[i], "score": float(scores[i])} for i in top_idx]


def build_context(chunks):
    return "\n\n".join(f"[FILE: {c['file']}]\n{c['text']}" for c in chunks)


def ask_llm(ollama_model, prompt):
    resp = requests.post(
        OLLAMA_URL,
        json={"model": ollama_model, "prompt": prompt, "stream": False,
              "options": {"num_predict": 250, "num_ctx": 2048}},
        timeout=1800,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="codellama:7b")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading embedding model '{EMBEDDING_MODEL}' ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    records, embeddings = build_code_index(embedder)

    results = []
    for i, question in enumerate(REPO_QUESTIONS, 1):
        print(f"[{i}/{len(REPO_QUESTIONS)}] {question}")
        chunks = search(question, embedder, records, embeddings, top_k=args.top_k)
        context = build_context(chunks)
        prompt = CODE_SYSTEM_PREFIX.format(context=context, question=question)
        answer = ask_llm(args.model, prompt)
        results.append({
            "question": question,
            "retrieved_files": [c["file"] for c in chunks],
            "answer": answer,
            # filled in manually for REPORT.md Exercise 6:
            "correct": None,           # True / False / "partial"
            "failure_mode": None,      # e.g. "missed cross-file link", "wrong file retrieved"
        })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"repo_understanding_{args.model.replace(':', '_')}_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    for r in results:
        print(f"\nQ: {r['question']}")
        print(f"  retrieved: {r['retrieved_files']}")
        print(f"  A: {r['answer'][:200]}")


if __name__ == "__main__":
    main()
