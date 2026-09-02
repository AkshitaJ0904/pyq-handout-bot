"""
Exercise 4 — Data Service (port 5002)

Owns the knowledge base: the chunked syllabus/PYQ text and its embeddings,
as produced by ingestion/build_index.py. Other services fetch the index
over HTTP instead of touching the files directly.
"""
import json
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
INDEX_DIR = ROOT / "ingestion" / "index"

app = Flask(__name__)


def load_index():
    chunks_path = INDEX_DIR / "chunks.json"
    emb_path = INDEX_DIR / "embeddings.npy"
    if not chunks_path.exists() or not emb_path.exists():
        return None, None
    chunks = json.loads(chunks_path.read_text())
    embeddings = np.load(emb_path)
    return chunks, embeddings


@app.route("/index", methods=["GET"])
def get_index():
    """Full knowledge base: chunk text/metadata + their embedding vectors."""
    chunks, embeddings = load_index()
    if chunks is None:
        return jsonify({"error": "No index found. Run ingestion/build_index.py first."}), 503
    return jsonify({
        "chunks": chunks,
        "embeddings": embeddings.tolist(),
    })


@app.route("/chunks", methods=["GET"])
def get_chunks():
    """Chunk text/metadata only, no vectors (lighter, for browsing)."""
    chunks, _ = load_index()
    if chunks is None:
        return jsonify({"error": "No index found. Run ingestion/build_index.py first."}), 503
    return jsonify({"chunks": chunks, "count": len(chunks)})


@app.route("/health", methods=["GET"])
def health():
    chunks, embeddings = load_index()
    return jsonify({
        "status": "ok",
        "index_loaded": chunks is not None,
        "chunk_count": len(chunks) if chunks else 0,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True, use_reloader=False)
