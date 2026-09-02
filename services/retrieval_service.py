"""
Exercise 4 — Retrieval / RAG Service (port 5003)

Question -> Query Embedding -> Vector Similarity -> Relevant Context

Fetches the knowledge base from the Data Service, embeds the incoming
question with the same sentence-transformer used at index time, and
returns the top-k most similar chunks by cosine similarity.
"""
import os

import numpy as np
import requests
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

DATA_SERVICE_URL = os.environ.get("DATA_SERVICE_URL", "http://localhost:5002")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 4

_model = None
_chunks = None
_embeddings = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def load_index_from_data_service():
    global _chunks, _embeddings
    resp = requests.get(f"{DATA_SERVICE_URL}/index", timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    _chunks = payload["chunks"]
    _embeddings = np.array(payload["embeddings"], dtype=np.float32)


@app.route("/search", methods=["POST"])
def search():
    global _chunks, _embeddings
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    top_k = int(data.get("top_k", DEFAULT_TOP_K))

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    if _chunks is None or _embeddings is None:
        try:
            load_index_from_data_service()
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Could not reach data service: {e}"}), 502

    model = get_model()
    query_vec = model.encode([question], convert_to_numpy=True)[0]
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    scores = _embeddings @ query_vec
    top_idx = np.argsort(-scores)[:top_k]
    results = []
    for idx in top_idx:
        chunk = dict(_chunks[idx])
        chunk["score"] = float(scores[idx])
        results.append(chunk)

    return jsonify({"question": question, "results": results})


@app.route("/reload", methods=["POST"])
def reload_index():
    """Force a re-fetch of the index from the Data Service (e.g. after re-ingestion)."""
    try:
        load_index_from_data_service()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Could not reach data service: {e}"}), 502
    return jsonify({"status": "reloaded", "chunk_count": len(_chunks)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "index_loaded": _chunks is not None,
        "data_service_url": DATA_SERVICE_URL,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True, use_reloader=False)
