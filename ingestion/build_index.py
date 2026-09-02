"""
Exercise 2 — Knowledge Base

Documents -> Chunking -> Embeddings -> Vector Representation

Reads every syllabus/handout file from data/syllabus/ and every PYQ paper
from data/pyq/ (.txt or .pdf), splits them into overlapping word-chunks,
embeds each chunk with a sentence-transformer, and writes the result to
ingestion/index/ as:
  - chunks.json      metadata + raw text for every chunk
  - embeddings.npy   one L2-normalized embedding vector per chunk (float32)

Re-run this any time the documents in data/ change:
    python3 ingestion/build_index.py
"""
import json
import re
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = Path(__file__).resolve().parent / "index"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE_WORDS = 120
CHUNK_OVERLAP_WORDS = 30


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS):
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    if not words or words == [""]:
        return []
    chunks = []
    start = 0
    step = max(size - overlap, 1)
    while start < len(words):
        chunk_words = words[start:start + size]
        chunks.append(" ".join(chunk_words))
        if start + size >= len(words):
            break
        start += step
    return chunks


def collect_documents():
    docs = []
    for doc_type, folder in (("syllabus", DATA_DIR / "syllabus"), ("pyq", DATA_DIR / "pyq")):
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in (".txt", ".pdf"):
                docs.append((doc_type, path))
    return docs


def guess_year(filename: str):
    match = re.search(r"(20\d{2})", filename)
    return match.group(1) if match else None


def build():
    print(f"Loading embedding model '{EMBEDDING_MODEL}' ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    documents = collect_documents()
    if not documents:
        print("No documents found in data/syllabus or data/pyq. Nothing to index.")
        return

    records = []
    for doc_type, path in documents:
        text = extract_text(path)
        pieces = chunk_text(text)
        for i, piece in enumerate(pieces):
            records.append({
                "id": f"{path.stem}_{i}",
                "source_file": path.name,
                "doc_type": doc_type,
                "year": guess_year(path.name),
                "chunk_index": i,
                "text": piece,
            })
        print(f"  {doc_type:9s} {path.name:30s} -> {len(pieces)} chunks")

    if not records:
        print("Documents were found but produced zero chunks (empty files?).")
        return

    print(f"Embedding {len(records)} chunks ...")
    texts = [r["text"] for r in records]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    embeddings = (embeddings / norms).astype(np.float32)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_DIR / "chunks.json", "w") as f:
        json.dump(records, f, indent=2)
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    with open(INDEX_DIR / "meta.json", "w") as f:
        json.dump({"model": EMBEDDING_MODEL, "dim": int(embeddings.shape[1]), "count": len(records)}, f, indent=2)

    n_syllabus = sum(1 for r in records if r["doc_type"] == "syllabus")
    n_pyq = sum(1 for r in records if r["doc_type"] == "pyq")
    print(f"\nDone. Indexed {len(records)} chunks ({n_syllabus} syllabus, {n_pyq} pyq).")
    print(f"Wrote {INDEX_DIR / 'chunks.json'} and {INDEX_DIR / 'embeddings.npy'}")


if __name__ == "__main__":
    build()
