"""
Exercise 3 — Retrieval + RAG (builds on Exercise 1)

Question -> Query Embedding -> Vector Similarity -> Relevant Context
         -> Context + Question -> Ollama -> Code Llama -> Response

/ask now retrieves the most relevant syllabus/PYQ chunks for the question
and feeds them to Code Llama as context, instead of asking the model
cold. /compare returns both the RAG and non-RAG answers side by side so
you can see the difference retrieval makes.
"""
from flask import Flask, request, jsonify
import requests

from rag.retrieval import Retriever, build_context

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "codellama"
TOP_K = 4

RAG_SYSTEM_PREFIX = (
    "You are a study assistant for a college course. Use ONLY the context "
    "below, pulled from the course syllabus and past exam papers (PYQs), to "
    "answer the question. If the context shows a topic appears often in "
    "PYQs but is barely covered in the syllabus, point that out explicitly. "
    "If the context doesn't contain the answer, say so instead of guessing.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

_retriever = None
_retriever_error = None


def get_retriever():
    global _retriever, _retriever_error
    if _retriever is None and _retriever_error is None:
        try:
            _retriever = Retriever()
        except FileNotFoundError as e:
            _retriever_error = str(e)
    return _retriever


def call_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 150, "num_ctx": 1024},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json().get("response", "")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    use_rag = data.get("rag", True)

    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    retriever = get_retriever() if use_rag else None
    sources = []
    prompt = question

    if retriever:
        chunks = retriever.search(question, top_k=TOP_K)
        sources = [
            {"source_file": c["source_file"], "doc_type": c["doc_type"], "score": round(c["score"], 3)}
            for c in chunks
        ]
        context = build_context(chunks)
        prompt = RAG_SYSTEM_PREFIX.format(context=context, question=question)

    try:
        answer = call_ollama(prompt)
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Could not reach Ollama. Is it running? Try: ollama serve"}), 503
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Ollama request failed: {e}"}), 502

    return jsonify({
        "question": question,
        "answer": answer,
        "rag_used": retriever is not None,
        "sources": sources,
    })


@app.route("/compare", methods=["POST"])
def compare():
    """Runs the same question with and without RAG, for demoing the difference."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    retriever = get_retriever()
    if retriever is None:
        return jsonify({"error": _retriever_error or "Retriever unavailable"}), 503

    try:
        no_rag_answer = call_ollama(question)

        chunks = retriever.search(question, top_k=TOP_K)
        context = build_context(chunks)
        rag_prompt = RAG_SYSTEM_PREFIX.format(context=context, question=question)
        rag_answer = call_ollama(rag_prompt)
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Could not reach Ollama. Is it running? Try: ollama serve"}), 503
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Ollama request failed: {e}"}), 502

    return jsonify({
        "question": question,
        "without_rag": no_rag_answer,
        "with_rag": rag_answer,
        "sources_used_for_rag": [
            {"source_file": c["source_file"], "doc_type": c["doc_type"], "score": round(c["score"], 3)}
            for c in chunks
        ],
    })


@app.route("/health", methods=["GET"])
def health():
    retriever = get_retriever()
    return jsonify({
        "status": "ok",
        "rag_index_loaded": retriever is not None,
        "rag_index_error": _retriever_error,
    })


if __name__ == "__main__":
    # Port 5000 collides with macOS AirPlay Receiver on some Macs -> use 5001.
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
