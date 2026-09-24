"""The chunk-similarity side of the repo-understanding comparison.

Deliberately identical in method to eval/repo_understanding.py: the same
naive word-window chunker, the same embedder, the same top-k cosine search,
and explicitly no AST or call-graph awareness. Reproducing Week 4's setup is
the point -- the comparison is only meaningful if this side is the same
system that produced the documented failures.

The index is built lazily on first use and cached on disk, because encoding
the whole repo takes long enough to ruin a live demo.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "ingestion" / "index"
CHUNKS_PATH = CACHE_DIR / "repo_chunks.json"
EMB_PATH = CACHE_DIR / "repo_embeddings.npy"

INCLUDE_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".txt", ".sh"}
EXCLUDE_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules", ".pytest_cache"}
CHUNK_SIZE, CHUNK_OVERLAP = 120, 30

_records = None
_embeddings = None


def _chunk(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    out, start = [], 0
    while start < len(words):
        out.append(" ".join(words[start:start + size]))
        start += max(1, size - overlap)
    return out


def _collect():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in INCLUDE_SUFFIXES:
            continue
        if any(part in EXCLUDE_DIRS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def build(embedder, force=False):
    global _records, _embeddings
    if not force and CHUNKS_PATH.exists() and EMB_PATH.exists():
        _records = json.loads(CHUNKS_PATH.read_text())
        _embeddings = np.load(EMB_PATH)
        return _records, _embeddings

    records = []
    for path in _collect():
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = str(path.relative_to(ROOT))
        for i, piece in enumerate(_chunk(text)):
            records.append({"file": rel, "chunk_index": i, "text": piece})

    emb = embedder.encode([r["text"] for r in records], convert_to_numpy=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    emb = (emb / norms).astype(np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(records))
    np.save(EMB_PATH, emb)
    _records, _embeddings = records, emb
    return records, emb


def search(question, embedder, top_k=4):
    records, emb = build(embedder)
    if not records:
        return []
    qv = embedder.encode([question], convert_to_numpy=True)[0]
    n = np.linalg.norm(qv)
    if n > 0:
        qv = qv / n
    scores = emb @ qv
    return [
        {**records[i], "score": round(float(scores[i]), 3)}
        for i in np.argsort(-scores)[:top_k]
    ]


def build_context(chunks):
    return "\n\n".join(f"[{c['file']}]\n{c['text']}" for c in chunks)
