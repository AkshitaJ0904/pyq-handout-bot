# PYQ + Handout Cross-Reference Bot

Reads a course syllabus/handout and cross-references it against previous
years' question papers (PYQs) to flag topics that show up often in exams
but are thin or missing in the handout.

Built progressively across 5 lab exercises. Each exercise is a git tag on
this same codebase (`git tag`) — it's one evolving app, not five separate ones.

## Architecture (final, Exercise 5)

```
User -> App -> API/Orchestration -> RAG/Retrieval Service -> Context
     -> LLM Service -> Ollama -> Code Llama -> Response
                                       ^
                           Data Service (syllabus + PYQ chunks/embeddings)
```

## Exercise progress

- **ex1-basic-app** — Flask `/ask` route forwarding straight to Code Llama via Ollama.
- **ex2-knowledge-base** — `ingestion/build_index.py` chunks syllabus/PYQ docs and embeds them.
- **ex3-rag** — `app.py` retrieves relevant chunks before asking the LLM; `/compare` shows with vs. without RAG.
- **ex4-services** — same logic split into `services/{data,retrieval,llm}_service.py` + `services/orchestrator.py`.
- **ex5-docker** — everything containerized via `docker/docker-compose.yml`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Ollama must be running locally with the model pulled:
```bash
ollama serve          # if not already running
ollama pull codellama
```

## Your real course documents

Sample placeholder docs (a generic "Operating Systems" course) live in
`data/syllabus/` and `data/pyq/` so the pipeline runs end-to-end out of the
box. **Replace them with your real syllabus PDF and PYQ PDFs**, then rebuild
the index:

```bash
python3 ingestion/build_index.py
```

Filenames just need `.pdf` or `.txt`, and PYQ filenames should contain a
4-digit year (e.g. `pyq_2023.pdf`) so results can be attributed to a year.

## Running it

**Monolith (Exercises 1-3):**
```bash
python3 app.py
curl -X POST http://localhost:5001/ask -H "Content-Type: application/json" \
  -d '{"question": "What topics come up most in past exams but are barely in the syllabus?"}'
```

**Services (Exercise 4):**
```bash
./scripts/run_services.sh
curl -X POST http://localhost:5001/ask -H "Content-Type: application/json" \
  -d '{"question": "..."}'
```

**Docker (Exercise 5):**
```bash
cd docker
docker compose up --build
```

## Notes on this machine

Code Llama 7B on an 8GB M1 with other apps open can swap heavily and get
very slow (single answers taking minutes). Keep other memory-hungry apps
closed while testing, and keep `num_predict` low for quick iteration.

## Metrics in the UI

Every answer now carries the eight Week 4 metrics, rendered beside it —
five quality (correctness, relevance, retrieval quality, hallucination rate,
test-pass) and three performance (latency, token usage, resources). The
implementations in `eval/metrics.py` are reused unchanged; `rag/live_metrics.py`
just makes them available per-request.

Three of the eight need ground truth a free-typed question doesn't have:
correctness needs expected keypoints, retrieval quality needs labelled
relevant source files, and test-pass needs an assertion harness. Those are
shown greyed with the reason on hover rather than filled with a made-up
number. The **Labelled evaluation question** dropdown loads any of the 28
questions from `eval/eval_dataset.json`, which supplies that ground truth and
lights all eight up.

The three comparison views each carry the panel:

- **Answer** — one model, full panel with per-metric detail.
- **Compare with / without** — the same question answered with and without the
  knowledge base, both sides measured. Note that grounding cannot be measured
  at all without retrieved context, which is the point.
- **Compare models** — `codellama:7b`, `starcoder2:3b` and `deepseek-coder:1.3b`
  side by side on identical input.

`deepseek-coder:1.3b` replaces `qwen3-coder:latest` as the third model. Week 4
found the 18GB qwen3-coder ran at ~7s per output token and exhausted memory on
this hardware (see `eval/REPORT.md`), which made it unusable for a live demo.
deepseek-coder:1.3b is ~776MB and is one of the models named in the course
outcomes.

```bash
ollama pull deepseek-coder:1.3b
```

## Tests

```bash
python3 -m pytest tests/ -q
```

`tests/test_metrics_wiring.py` stubs Ollama and the embedder, so the suite runs
without a model pulled and without `ollama serve`. It checks the wiring — that
every route returns all eight metrics, that a labelled question unlocks the
three needing ground truth, and that an unavailable metric never reports a
number — not model quality, which is what `eval/` is for.

## Week 4 — evaluation

Multi-model evaluation, quantitative metrics, RAG pipeline tracing, and
repository-level code-understanding experiments live under `eval/` (same
app, same knowledge base — only the Ollama model varies). See
`eval/REPORT.md` for the write-up and `eval/run_eval.py --help` /
`eval/rag_trace.py` / `eval/repo_understanding.py` to reproduce.
