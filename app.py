"""
Exercise 3 — Retrieval + RAG (builds on Exercise 1)

Question -> Query Embedding -> Vector Similarity -> Relevant Context
         -> Context + Question -> Ollama -> Code Llama -> Response

/ask now retrieves the most relevant syllabus/PYQ chunks for the question
and feeds them to Code Llama as context, instead of asking the model
cold. /compare returns both the RAG and non-RAG answers side by side so
you can see the difference retrieval makes.
"""
import re
from pathlib import Path

from flask import Flask, render_template, request, jsonify
import requests
from werkzeug.utils import secure_filename

from rag.retrieval import Retriever, build_context
from ingestion.build_index import build as rebuild_index

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB per upload request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "codellama:7b"
TOP_K = 4

DATA_DIR = Path(__file__).resolve().parent / "data"
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

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


def call_ollama(prompt: str, model: str = None) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model or MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 150, "num_ctx": 1024},
        },
        timeout=600,
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


@app.route("/compare_models", methods=["POST"])
def compare_models():
    """Runs the same question + same retrieved context through several
    Ollama models, so different LLMs can be compared side by side on
    identical input (Week 4, Exercise 1)."""
    import time

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    models = data.get("models") or [MODEL_NAME]
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400
    if not models:
        return jsonify({"error": "Missing 'models' in request body"}), 400

    retriever = get_retriever()
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

    results = []
    for model in models:
        t0 = time.time()
        try:
            answer = call_ollama(prompt, model=model)
            results.append({"model": model, "answer": answer, "latency_s": round(time.time() - t0, 1)})
        except requests.exceptions.RequestException as e:
            results.append({"model": model, "error": str(e), "latency_s": round(time.time() - t0, 1)})

    return jsonify({"question": question, "sources": sources, "results": results})


@app.route("/health", methods=["GET"])
def health():
    retriever = get_retriever()
    return jsonify({
        "status": "ok",
        "rag_index_loaded": retriever is not None,
        "rag_index_error": _retriever_error,
        "chunk_count": len(retriever.chunks) if retriever else 0,
        "model": MODEL_NAME,
    })


@app.route("/files", methods=["GET"])
def list_files():
    def files_in(sub):
        d = DATA_DIR / sub
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.suffix.lower() in ALLOWED_EXTENSIONS)

    return jsonify({"syllabus": files_in("syllabus"), "pyq": files_in("pyq")})


@app.route("/upload", methods=["POST"])
def upload():
    doc_type = request.form.get("doc_type", "").strip()
    if doc_type not in ("syllabus", "pyq"):
        return jsonify({"error": "doc_type must be 'syllabus' or 'pyq'"}), 400

    files = [f for f in request.files.getlist("file") if f and f.filename]
    if not files:
        return jsonify({"error": "No file provided"}), 400

    year = request.form.get("year", "").strip()
    if year and not re.fullmatch(r"(19|20)\d{2}", year):
        return jsonify({"error": "Year must be a 4-digit year"}), 400

    target_dir = DATA_DIR / doc_type
    target_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        filename = secure_filename(f.filename)
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type: {filename}"}), 400
        if year and not re.search(r"(19|20)\d{2}", filename):
            filename = f"{year}_{filename}"
        f.save(target_dir / filename)
        saved.append(filename)

    global _retriever, _retriever_error
    try:
        rebuild_index()
    except Exception as e:
        return jsonify({"error": f"Saved but re-indexing failed: {e}"}), 500

    _retriever = None
    _retriever_error = None
    retriever = get_retriever()

    return jsonify({
        "status": "ok",
        "saved": saved,
        "chunk_count": len(retriever.chunks) if retriever else 0,
    })


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


if __name__ == "__main__":
    # Port 5000 collides with macOS AirPlay Receiver on some Macs -> use 5001.
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
