"""
Exercise 4 — Orchestrator / Application Service (port 5001)

User -> App -> API/Orchestration -> RAG/Retrieval Service -> Context
     -> LLM Service -> Code Llama -> Response

Same /ask and /compare contract as the Exercise 3 monolith, but now the
retrieval and generation steps are separate HTTP services instead of
in-process function calls.
"""
import os
import re
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent.parent
app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)

RETRIEVAL_SERVICE_URL = os.environ.get("RETRIEVAL_SERVICE_URL", "http://localhost:5003")
LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://localhost:5004")
TOP_K = 4

DATA_DIR = ROOT / "data"
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

RAG_SYSTEM_PREFIX = (
    "You are a study assistant for a college course. Use ONLY the context "
    "below, pulled from the course syllabus and past exam papers (PYQs), to "
    "answer the question. If the context shows a topic appears often in "
    "PYQs but is barely covered in the syllabus, point that out explicitly. "
    "If the context doesn't contain the answer, say so instead of guessing.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


def build_context(chunks) -> str:
    lines = []
    for c in chunks:
        tag = f"[{c['doc_type'].upper()} | {c['source_file']}]"
        lines.append(f"{tag}\n{c['text']}")
    return "\n\n".join(lines)


def retrieve(question: str, top_k: int = TOP_K):
    resp = requests.post(
        f"{RETRIEVAL_SERVICE_URL}/search",
        json={"question": question, "top_k": top_k},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["results"]


def generate(prompt: str, model: str = None) -> str:
    resp = requests.post(
        f"{LLM_SERVICE_URL}/generate",
        json={"prompt": prompt, "model": model} if model else {"prompt": prompt},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["response"]


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    use_rag = data.get("rag", True)

    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    sources = []
    prompt = question
    try:
        if use_rag:
            chunks = retrieve(question)
            sources = [
                {"source_file": c["source_file"], "doc_type": c["doc_type"], "score": round(c["score"], 3)}
                for c in chunks
            ]
            prompt = RAG_SYSTEM_PREFIX.format(context=build_context(chunks), question=question)

        answer = generate(prompt)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Upstream service call failed: {e}"}), 502

    return jsonify({
        "question": question,
        "answer": answer,
        "rag_used": use_rag,
        "sources": sources,
    })


@app.route("/compare", methods=["POST"])
def compare():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    try:
        no_rag_answer = generate(question)

        chunks = retrieve(question)
        rag_prompt = RAG_SYSTEM_PREFIX.format(context=build_context(chunks), question=question)
        rag_answer = generate(rag_prompt)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Upstream service call failed: {e}"}), 502

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
    Ollama models via the LLM Service, so different LLMs can be compared
    side by side on identical input (Week 4, Exercise 1)."""
    import time

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    models = data.get("models") or []
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400
    if not models:
        return jsonify({"error": "Missing 'models' in request body"}), 400

    sources = []
    prompt = question
    try:
        chunks = retrieve(question)
        sources = [
            {"source_file": c["source_file"], "doc_type": c["doc_type"], "score": round(c["score"], 3)}
            for c in chunks
        ]
        prompt = RAG_SYSTEM_PREFIX.format(context=build_context(chunks), question=question)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Retrieval service call failed: {e}"}), 502

    results = []
    for model in models:
        t0 = time.time()
        try:
            answer = generate(prompt, model=model)
            results.append({"model": model, "answer": answer, "latency_s": round(time.time() - t0, 1)})
        except requests.exceptions.RequestException as e:
            results.append({"model": model, "error": str(e), "latency_s": round(time.time() - t0, 1)})

    return jsonify({"question": question, "sources": sources, "results": results})


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

    try:
        resp = requests.post(f"{RETRIEVAL_SERVICE_URL}/reindex", timeout=120)
        resp.raise_for_status()
        chunk_count = resp.json().get("chunk_count", 0)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Saved but re-indexing failed: {e}"}), 500

    return jsonify({"status": "ok", "saved": saved, "chunk_count": chunk_count})


@app.route("/health", methods=["GET"])
def health():
    services = {}
    for name, url in (("retrieval_service", RETRIEVAL_SERVICE_URL), ("llm_service", LLM_SERVICE_URL)):
        try:
            r = requests.get(f"{url}/health", timeout=5)
            services[name] = r.json()
        except requests.exceptions.RequestException as e:
            services[name] = {"status": "unreachable", "error": str(e)}
    return jsonify({"status": "ok", "services": services})


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


if __name__ == "__main__":
    # Port 5000 collides with macOS AirPlay Receiver on some Macs -> use 5001.
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
