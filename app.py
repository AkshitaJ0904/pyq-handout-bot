"""
Exercise 3 — Retrieval + RAG (builds on Exercise 1)

Question -> Query Embedding -> Vector Similarity -> Relevant Context
         -> Context + Question -> Ollama -> Code Llama -> Response

/ask now retrieves the most relevant syllabus/PYQ chunks for the question
and feeds them to Code Llama as context, instead of asking the model
cold. /compare returns both the RAG and non-RAG answers side by side so
you can see the difference retrieval makes.
"""
import json
import re
import time
from pathlib import Path

from flask import Flask, render_template, request, jsonify
import requests
from werkzeug.utils import secure_filename

from rag.retrieval import Retriever, build_context
from ingestion.build_index import build as rebuild_index
from eval.metrics import ResourceSampler
from rag import live_metrics
from guardrails import pipeline as guard_pipeline
from codesearch import planner, repo_rag
from codesearch.local_backend import LocalASTBackend
from codesearch.sourcegraph_backend import SourcegraphBackend

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB per upload request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "codellama:7b"
TOP_K = 4

DATA_DIR = Path(__file__).resolve().parent / "data"
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
EVAL_DATASET = Path(__file__).resolve().parent / "eval" / "eval_dataset.json"

# Week 4 found a 150-token budget truncated every code answer mid-function
# (REPORT.md, Exercise 3), so code-generation questions get a larger budget.
NUM_PREDICT_DEFAULT = 150
NUM_PREDICT_CODE = 400

# Guardrails can be turned off per-request so the UI can demonstrate
# without-guardrail vs with-guardrail behaviour on the same question.
GUARDRAILS_DEFAULT = True

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
_embedder = None
_eval_questions = None
_code_backend = None

REPO_QA_PROMPT = (
    "You are answering a question about a software repository. Use ONLY the "
    "information below. If it does not contain the answer, say so explicitly "
    "instead of guessing.\n\n{context}\n\nQuestion: {question}\nAnswer:"
)


def get_retriever():
    global _retriever, _retriever_error
    if _retriever is None and _retriever_error is None:
        try:
            _retriever = Retriever()
        except FileNotFoundError as e:
            _retriever_error = str(e)
    return _retriever


def get_embedder():
    """The sentence-transformer used for the relevance and hallucination
    metrics. Reuses the retriever's already-loaded model rather than loading
    a second copy -- this app targets an 8GB machine and the weights are not
    free."""
    retriever = get_retriever()
    if retriever is not None:
        return retriever.model
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        from rag.retrieval import EMBEDDING_MODEL
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def load_eval_questions():
    """The 28 labelled Week 4 questions, cached. These are the only questions
    with the ground truth that correctness, retrieval quality and test-pass
    need, so the UI offers them in a dropdown."""
    global _eval_questions
    if _eval_questions is None:
        try:
            _eval_questions = json.loads(EVAL_DATASET.read_text()).get("questions", [])
        except (OSError, ValueError):
            _eval_questions = []
    return _eval_questions


def find_eval_question(question_id):
    if not question_id:
        return None
    return next((q for q in load_eval_questions() if q.get("id") == question_id), None)


def call_ollama(prompt: str, model: str = None, num_predict: int = NUM_PREDICT_DEFAULT) -> dict:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model or MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict, "num_ctx": 1024},
        },
        timeout=600,
    )
    response.raise_for_status()
    # Return the whole payload, not just the text: Ollama reports
    # prompt_eval_count / eval_count / eval_duration in the same response,
    # and those are the token-usage and throughput metrics.
    return response.json()


def generate(prompt: str, model: str = None, num_predict: int = NUM_PREDICT_DEFAULT):
    """One generation plus everything the metric panel needs: the answer
    text, Ollama's raw stats payload, sampled CPU/RSS, and wall-clock
    latency."""
    t0 = time.time()
    with ResourceSampler() as sampler:
        payload = call_ollama(prompt, model=model, num_predict=num_predict)
    wall_s = time.time() - t0
    return payload.get("response", ""), payload, sampler.summary(), wall_s


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    use_rag = data.get("rag", True)
    ground_truth = find_eval_question(data.get("question_id"))

    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    retriever = get_retriever() if use_rag else None

    num_predict = NUM_PREDICT_CODE if (ground_truth or {}).get("code_test") else NUM_PREDICT_DEFAULT
    guards_on = data.get("guardrails", GUARDRAILS_DEFAULT)

    # Captured from inside the pipeline so the metrics panel still reports on
    # the real generation even when a guard replaces the answer shown.
    gen_state = {}

    def _generate(q, ctx, chunks):
        a, payload, resources, wall_s = generate(
            RAG_SYSTEM_PREFIX.format(context=ctx, question=q) if ctx else q,
            num_predict=num_predict)
        gen_state.update(answer=a, payload=payload, resources=resources, wall_s=wall_s)
        return a

    try:
        result = guard_pipeline.run(
            question,
            retrieve_fn=(lambda q: retriever.search(q, top_k=TOP_K)) if retriever else (lambda q: []),
            generate_fn=_generate,
            build_context_fn=build_context,
            embedder=get_embedder(),
            enabled=guards_on,
            # No knowledge base means nothing to ground against, so only the
            # input stage applies -- see guardrails/pipeline.py.
            stages=guard_pipeline.ALL_STAGES if retriever else ("input",),
        )
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Could not reach Ollama. Is it running? Try: ollama serve"}), 503
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Ollama request failed: {e}"}), 502

    sources = [
        {"source_file": c["source_file"], "doc_type": c["doc_type"], "score": round(c["score"], 3)}
        for c in (result.chunks or [])
    ]
    context = build_context(result.chunks) if result.chunks else ""

    body = {
        "question": question,
        "answer": result.answer,
        "rag_used": retriever is not None,
        "sources": sources,
        "guardrails": result.to_dict(),
        "guardrails_enabled": guards_on,
    }
    # Metrics only mean something when a generation actually happened.
    if gen_state:
        body["metrics"] = live_metrics.compute(
            question, gen_state["answer"], context, sources, gen_state["payload"],
            gen_state["resources"], gen_state["wall_s"], ground_truth)
        body["raw_answer"] = gen_state["answer"]
    return jsonify(body)


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

    ground_truth = find_eval_question(data.get("question_id"))

    try:
        no_rag_answer, no_rag_payload, no_rag_res, no_rag_s = generate(question)

        chunks = retriever.search(question, top_k=TOP_K)
        context = build_context(chunks)
        rag_prompt = RAG_SYSTEM_PREFIX.format(context=context, question=question)
        rag_answer, rag_payload, rag_res, rag_s = generate(rag_prompt)
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Could not reach Ollama. Is it running? Try: ollama serve"}), 503
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Ollama request failed: {e}"}), 502

    sources = [
        {"source_file": c["source_file"], "doc_type": c["doc_type"], "score": round(c["score"], 3)}
        for c in chunks
    ]

    return jsonify({
        "question": question,
        "without_rag": no_rag_answer,
        "with_rag": rag_answer,
        "sources_used_for_rag": sources,
        # Both sides get the full panel, so the with/without contrast is
        # numeric and not just two blocks of prose. Grounding is the telling
        # one: without a knowledge base it cannot be measured at all.
        "metrics_without_rag": live_metrics.compute(
            question, no_rag_answer, "", [], no_rag_payload, no_rag_res, no_rag_s, ground_truth
        ),
        "metrics_with_rag": live_metrics.compute(
            question, rag_answer, context, sources, rag_payload, rag_res, rag_s, ground_truth
        ),
    })


@app.route("/compare_models", methods=["POST"])
def compare_models():
    """Runs the same question + same retrieved context through several
    Ollama models, so different LLMs can be compared side by side on
    identical input (Week 4, Exercise 1)."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    models = data.get("models") or [MODEL_NAME]
    ground_truth = find_eval_question(data.get("question_id"))
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400
    if not models:
        return jsonify({"error": "Missing 'models' in request body"}), 400

    retriever = get_retriever()
    sources = []
    prompt = question
    context = ""
    if retriever:
        chunks = retriever.search(question, top_k=TOP_K)
        sources = [
            {"source_file": c["source_file"], "doc_type": c["doc_type"], "score": round(c["score"], 3)}
            for c in chunks
        ]
        context = build_context(chunks)
        prompt = RAG_SYSTEM_PREFIX.format(context=context, question=question)

    num_predict = NUM_PREDICT_CODE if (ground_truth or {}).get("code_test") else NUM_PREDICT_DEFAULT

    results = []
    for model in models:
        t0 = time.time()
        try:
            answer, payload, resources, wall_s = generate(
                prompt, model=model, num_predict=num_predict
            )
            results.append({
                "model": model,
                "answer": answer,
                "latency_s": round(wall_s, 1),
                "metrics": live_metrics.compute(
                    question, answer, context, sources, payload, resources, wall_s, ground_truth
                ),
            })
        except requests.exceptions.RequestException as e:
            results.append({"model": model, "error": str(e), "latency_s": round(time.time() - t0, 1)})

    return jsonify({"question": question, "sources": sources, "results": results})


def get_code_backend():
    """Prefer a configured Sourcegraph instance; fall back to the local AST
    backend so the comparison still runs when no server is reachable. Which
    one answered is reported in the response and shown in the UI -- the
    fallback is never silent."""
    global _code_backend
    if _code_backend is None:
        sg = SourcegraphBackend()
        _code_backend = sg if sg.available() else LocalASTBackend()
    return _code_backend


@app.route("/repo_qa", methods=["POST"])
def repo_qa():
    """Answers a question about THIS repository twice -- once through chunk
    similarity search (the Week 3 RAG pipeline pointed at the repo) and once
    through structural code search -- so the two can be compared directly.

    Week 4, Exercise 6 documented three question types the chunk-similarity
    side gets wrong: absence, cross-file comparison, and functional
    equivalence. This route is what closes that gap, and the UI shows both
    answers side by side."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    want_answers = data.get("answer", True)
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    # -- structural side: plan the query, run it, report the facts ----------
    backend = get_code_backend()
    queries = planner.plan(question)
    structural = [backend.run(q).to_dict() for q in queries]

    # -- chunk-similarity side: same method as eval/repo_understanding.py ---
    embedder = get_embedder()
    chunks = repo_rag.search(question, embedder) if embedder is not None else []

    payload = {
        "question": question,
        "backend": backend.name,
        "sourcegraph_configured": isinstance(backend, SourcegraphBackend),
        "structural": structural,
        "rag_chunks": chunks,
    }

    if want_answers:
        structural_context = "Structural code search results:\n" + "\n".join(
            f"- query {r['kind']}:{r['term']} -> "
            + (", ".join(f"{m['file']}:{m['line'] or ''} {m['symbol'] or ''}".strip()
                         for m in r["matches"]) if r["matches"] else "NO MATCHES FOUND")
            for r in structural
        )
        rag_context = repo_rag.build_context(chunks)
        try:
            payload["structural_answer"] = generate(
                REPO_QA_PROMPT.format(context=structural_context, question=question))[0]
            payload["rag_answer"] = generate(
                REPO_QA_PROMPT.format(context="Code excerpts:\n" + rag_context,
                                      question=question))[0]
        except requests.exceptions.RequestException as e:
            # No Ollama: the retrieved evidence is still the interesting part,
            # so return it rather than failing the whole request.
            payload["answer_error"] = f"Could not reach Ollama ({e}). Showing retrieval only."

    return jsonify(payload)


@app.route("/guardrail_demo", methods=["POST"])
def guardrail_demo():
    """Runs one question with guardrails off and on, for the
    without -> problematic / with -> controlled demonstration.

    Retrieval and generation happen ONCE and are shared by both sides, so the
    only thing that differs is whether the verdicts are acted on."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    retriever = get_retriever()
    cache = {}

    def _generate(q, ctx, chunks):
        # Generated once and shared by both runs, so the only difference
        # between them is whether the verdicts are acted on.
        if "answer" not in cache:
            try:
                cache["answer"] = generate(
                    RAG_SYSTEM_PREFIX.format(context=ctx, question=q) if ctx else q)[0]
            except requests.exceptions.RequestException as e:
                # A guard that blocks before generation is still worth showing,
                # so a missing model degrades to verdicts-only rather than 503.
                cache["answer"] = ""
                cache["error"] = f"Could not reach Ollama ({type(e).__name__}). "\
                                 f"Showing guardrail verdicts only."
        return cache["answer"]

    retrieve_fn = (lambda q: retriever.search(q, top_k=TOP_K)) if retriever else (lambda q: [])
    both = {}
    for label, enabled in (("with", True), ("without", False)):
        r = guard_pipeline.run(question, retrieve_fn, _generate, build_context,
                               embedder=get_embedder(), enabled=enabled)
        both[label] = {**r.to_dict(), "summary": guard_pipeline.summarise(r)}

    body = {"question": question, "without_guardrails": both["without"],
            "with_guardrails": both["with"]}
    if cache.get("error"):
        body["answer_error"] = cache["error"]
    return jsonify(body)


@app.route("/eval_questions", methods=["GET"])
def eval_questions():
    """The labelled Week 4 questions, for the UI dropdown. Picking one sends
    its id back with the request, which is what unlocks the three metrics
    that need ground truth."""
    return jsonify({"questions": [
        {
            "id": q["id"],
            "category": q.get("category"),
            "question": q["question"],
            "has_code_test": bool(q.get("code_test")),
        }
        for q in load_eval_questions()
    ]})


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
