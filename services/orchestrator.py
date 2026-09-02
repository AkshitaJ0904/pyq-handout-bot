"""
Exercise 4 — Orchestrator / Application Service (port 5001)

User -> App -> API/Orchestration -> RAG/Retrieval Service -> Context
     -> LLM Service -> Code Llama -> Response

Same /ask and /compare contract as the Exercise 3 monolith, but now the
retrieval and generation steps are separate HTTP services instead of
in-process function calls.
"""
import os

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

RETRIEVAL_SERVICE_URL = os.environ.get("RETRIEVAL_SERVICE_URL", "http://localhost:5003")
LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://localhost:5004")
TOP_K = 4

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


def generate(prompt: str) -> str:
    resp = requests.post(
        f"{LLM_SERVICE_URL}/generate",
        json={"prompt": prompt},
        timeout=300,
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


if __name__ == "__main__":
    # Port 5000 collides with macOS AirPlay Receiver on some Macs -> use 5001.
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
