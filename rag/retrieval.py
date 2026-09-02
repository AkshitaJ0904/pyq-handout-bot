"""
Exercise 3 — Retrieval

Question -> Query Embedding -> Vector Similarity -> Relevant Context

Loads the chunk index built by ingestion/build_index.py and answers
similarity queries against it using cosine similarity (a plain dot
product, since every stored embedding is already L2-normalized).
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "ingestion" / "index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self):
        chunks_path = INDEX_DIR / "chunks.json"
        emb_path = INDEX_DIR / "embeddings.npy"
        if not chunks_path.exists() or not emb_path.exists():
            raise FileNotFoundError(
                "No index found. Run: python3 ingestion/build_index.py"
            )
        self.chunks = json.loads(chunks_path.read_text())
        self.embeddings = np.load(emb_path)
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def search(self, query: str, top_k: int = 4):
        query_vec = self.model.encode([query], convert_to_numpy=True)[0]
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        scores = self.embeddings @ query_vec  # cosine similarity, both sides unit-norm
        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_idx:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results


def build_context(chunks) -> str:
    lines = []
    for c in chunks:
        tag = f"[{c['doc_type'].upper()} | {c['source_file']}]"
        lines.append(f"{tag}\n{c['text']}")
    return "\n\n".join(lines)
