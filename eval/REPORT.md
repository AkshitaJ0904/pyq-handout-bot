# Week 4 Report — Evaluating the PYQ + Handout Cross-Reference Bot

This builds on the Week 3 app (`ex3-rag` / current `main`) — no new application
was created. All Week 4 work lives under `eval/` and reuses the existing
`rag/retrieval.py` retrieval logic and `RAG_SYSTEM_PREFIX` prompt template
unchanged; only the Ollama `model` field varies across runs.

## TL;DR — conclusions

- **codellama:7b is the more accurate model** on this app's real use case:
  48.5% average keypoint coverage vs starcoder2:3b's 16.1% across all 28
  questions, and clearly better on every question category except
  `code_generation` and `refactoring_suggestion` (tied).
- **starcoder2:3b is roughly 2x faster and uses ~2x less memory**
  (4.6s vs 10.0s per answer; 2.1GB vs 4.6GB peak RAM) — but that speed isn't
  worth much when the answers are frequently wrong or empty.
- **There is a real quality/resource trade-off, and accuracy wins it here.**
  For a study assistant, a fast wrong answer is worse than a slow correct
  one, so `codellama:7b` is the better choice for this app despite being
  slower and heavier — this is a case where the more expensive model is
  also the right one, not a case of a "hidden cheaper alternative."
- **A third, much larger model (`qwen3-coder:latest`, 18GB) was tested and
  rejected**: it ran at roughly 7 seconds per output token on this hardware
  and a full evaluation attempt caused the machine to run out of free
  memory and become unresponsive. That is itself a valid finding for a
  resource-constrained deployment: bigger/newer isn't automatically usable.
- **Retrieval quality directly drives both correct answers and
  hallucinations** (Exercise 5): when the retriever pulled the right
  syllabus+PYQ chunks, the model gave correct, well-grounded answers, even
  for nuanced "is this covered?" gap questions. When the retriever pulled
  irrelevant chunks (e.g. from the mismatched Agentic AI handout, for a
  question about the Operating Systems course), the model **hallucinated
  confidently** — in one case fabricating a specific faculty name and room
  number by misattributing real facts from the wrong course document.
- **The current RAG pipeline cannot reliably answer repository-level,
  multi-file code questions** (Exercise 6): single-file lookups (which env
  var, which Dockerfile, which file) were answered correctly, but
  cross-file/negative questions failed — most strikingly, the system
  **completely fabricated a nonexistent test suite** (`tests/test_app.py`,
  etc.) when asked "are there any test files in this repo," because plain
  text-chunk similarity search has no way to represent "this doesn't
  exist." This is precisely the gap that repository-level tooling
  (Sourcegraph, covered next week) is built to close.

## Exercise 1 — Models evaluated

| Model | Size | Notes |
|---|---|---|
| `codellama:7b` | 3.8GB | The model the app originally hardcoded (`app.py:26`). |
| `starcoder2:3b` | 1.7GB | Smaller code-specialized model, different architecture/training data from Code Llama. |
| `qwen3-coder:latest` | 18GB | Larger, newer code-tuned model, already present on this machine. Tested but excluded from the full run — see Exercise 4. |

All three were run against the **same** app logic (`rag/retrieval.py`,
`RAG_SYSTEM_PREFIX`), the same prompts, the same 28-question set, and the
same knowledge base — only the Ollama `model` field varies.

## Exercise 2 — Evaluation dataset

`eval/eval_dataset.json` — 28 questions across 10 categories, grounded in
the actual indexed content (`ingestion/index/chunks.json`): the OS syllabus
placeholder, three years of OS PYQs, and the real Agentic AI course handout.

| Category | Count | What it tests |
|---|---|---|
| `factual_lookup_syllabus` | 4 | Direct recall from the syllabus |
| `factual_lookup_pyq` | 3 | Direct recall from past exam papers |
| `factual_lookup_handout` | 4 | Direct recall from the real Agentic AI handout |
| `coverage_gap_analysis` | 4 | The app's core feature — PYQ-frequent/syllabus-thin topics |
| `summarization` | 2 | Multi-sentence synthesis of one document |
| `cross_document_synthesis` | 3 | Combining PYQs across years, or syllabus+PYQ together |
| `hallucination_probe` | 3 | Questions with **no answer anywhere in the KB** — correct behavior is refusal |
| `code_generation` | 3 | Testable Python implementations of algorithms named in the PYQs |
| `refactoring_suggestion` | 1 | Code-improvement suggestion (non-RAG) |
| `bug_analysis` | 1 | Diagnosing a concurrency ordering bug (RAG-grounded) |

The same 28 questions are used for every model. Full text in
`eval/eval_dataset.json`.

## Exercise 3 — Quantitative evaluation

How each metric is computed (also documented as docstrings in `eval/metrics.py`):

**Quality**
- **Correctness (`keypoint_coverage`)** — fraction of hand-picked expected
  keywords/facts (chosen from the source document per question) present in
  the answer text.
- **Relevance (`semantic_relevance`)** — cosine similarity between the
  question's and answer's `all-MiniLM-L6-v2` embeddings (the same embedder
  the app already uses for retrieval).
- **Retrieval Quality (`retrieval_precision_at_k`)** — of the `top_k=4`
  chunks retrieved, the fraction whose source file was hand-labeled
  relevant to the question. (Identical across models by construction, since
  retrieval doesn't depend on the generation model — confirmed in the run.)
- **Hallucination Rate (`hallucination_rate`)** — fraction of answer
  sentences whose maximum embedding similarity to any retrieved context
  chunk falls below 0.35, i.e. not clearly grounded in what the model was
  actually given.
- **Test-Pass Rate (`run_code_test`)** — for `code_generation` questions
  with an automated check, the model's extracted Python function is run
  against a fixed assertion harness in an isolated subprocess.

**Performance** (read from Ollama's own response fields + `psutil`)
- **Latency** — wall-clock seconds per request.
- **Token usage** — `prompt_eval_count` / `eval_count` from Ollama, and
  derived tokens/sec.
- **Resources** — peak RSS memory (MB) of Ollama's `llama-server` runner
  process and mean system CPU% sampled every 0.2s during generation.
  **GPU utilization is not measured** — this machine is Apple Silicon and
  reading GPU counters without `sudo` isn't available through `psutil`;
  reported as such rather than fabricating a number.

### Full 28-question results (`codellama:7b` vs `starcoder2:3b`)

Source: `eval/results/run_20260910_132109.json` / `_summary.md`.

| Model | Avg latency (s) | Avg tok/s* | Avg peak RSS (MB) | Avg keypoint coverage | Avg relevance | Avg hallucination rate |
|---|---|---|---|---|---|---|
| codellama:7b | 9.96 | 18.2 | 4572.7 | **0.485** | **0.785** | **0.144** |
| starcoder2:3b | 4.55 | 39.0 | 2140.6 | 0.161 | 0.510 | 0.294 |

_* the raw run's naive mean tok/s was distorted by one degenerate
near-instant 1-token response from starcoder2:3b (reported as ~35,758 tok/s,
a divide-by-near-zero artifact of Ollama's duration counter). The harness
was patched (`run_eval.py`) to exclude samples with fewer than 5 response
tokens from the throughput average; 39.0 tok/s above is the corrected figure._

**By category (keypoint coverage — correctness proxy):**

| Category | codellama:7b | starcoder2:3b |
|---|---|---|
| factual_lookup_pyq | **0.889** | 0.222 |
| factual_lookup_syllabus | **0.750** | 0.000 |
| summarization | **0.625** | 0.375 |
| refactoring_suggestion | 0.500 | 0.500 |
| factual_lookup_handout | **0.375** | 0.125 |
| coverage_gap_analysis | **0.396** | 0.062 |
| hallucination_probe | **0.333** | 0.111 |
| code_generation | 0.333 | 0.333 |
| bug_analysis | **0.333** | 0.000 |
| cross_document_synthesis | **0.250** | 0.167 |

### Code-generation test-pass rate

The initial full run used a 120-token generation budget, which truncated
every code answer mid-function for both models (0/2 extractable). Raised to
400 tokens for `code_generation` questions specifically and re-run
(`eval/results/run_20260910_132339.json`):

| Model | Test-pass rate (of 2 auto-testable questions) | Detail |
|---|---|---|
| codellama:7b | **1/2 (50%)** | Q24 (FCFS avg waiting time): wrote a syntactically valid, cleanly extracted function, but the **logic was wrong** — `assert abs(result - 5.667) < 0.05` failed, it returned 3.33 (it didn't correctly account for each process's own arrival time when computing waiting time). Q25 (SSTF disk scheduling): correct, passed. |
| starcoder2:3b | 0/2 (0%), both unextractable | Q24: emitted a stub function body plus a stray `<|start_of_text|>` special-token artifact instead of real code. Q25: produced working-looking logic but never wrapped it in a proper fenced/complete function the harness could isolate and execute — it continued as loose script code rather than a self-contained function definition. |

This is itself a finding: `starcoder2:3b`'s code answers were harder to even
extract as runnable code, separate from correctness — a base/completion-style
model behavior that's a real integration cost if this app were to expose a
"generate code" feature.

## Exercise 4 — Analysis

- **Which model has the highest accuracy?** `codellama:7b`, by a wide
  margin (0.485 vs 0.161 average keypoint coverage across all 28
  questions; 3x higher). It also has meaningfully higher semantic
  relevance (0.785 vs 0.510).
- **Which model hallucinates less?** `codellama:7b` (0.144 vs 0.294
  average hallucination rate — roughly half). On the specific
  `hallucination_probe` category (questions with no true answer in the
  KB), `codellama:7b` also scored higher on "correct refusal" behavior
  (0.333 vs 0.111 keypoint coverage against refusal-phrase keywords).
- **Which model has better retrieval-based responses?** Retrieval itself
  is identical for both (same retriever, independent of the generation
  model), so this reduces to which model makes better use of the same
  retrieved context — `codellama:7b`, consistently, across every RAG
  category.
- **Which model generates code with a higher test-pass rate?**
  `codellama:7b` (1/2 vs 0/2) — see Exercise 3 code-generation detail above.
- **Which model has lower latency?** `starcoder2:3b`, by more than 2x
  (4.55s vs 9.96s average).
- **Which model uses fewer resources?** `starcoder2:3b`, using roughly
  53% less peak memory (2.1GB vs 4.6GB) — proportional to its smaller
  model size on disk (1.7GB vs 3.8GB).
- **Is the most accurate model also the most efficient? No.** There is a
  clear, consistent trade-off: `codellama:7b` is ~2.2x slower and uses
  ~2.1x the memory of `starcoder2:3b`, in exchange for ~3x the accuracy
  and about half the hallucination rate. **For this app's actual purpose
  (a study assistant a student trusts for exam prep), accuracy matters far
  more than a few extra seconds of latency, so the more expensive model is
  the right choice here** — this is a case where "biggest/slowest wins"
  rather than a hidden cheaper alternative being just as good.
- **`qwen3-coder:latest` (18GB) — excluded from the full run.** A single
  minimal test generation (3 output tokens) took ~112 seconds wall-clock
  (~22s of that was pure `eval_duration`, i.e. ~7.4 seconds per output
  token, versus sub-second-per-token for the two smaller models). A full
  batch-evaluation attempt drove the machine's free memory down to a few
  tens of MB and it became unresponsive; the run was killed. This is a
  legitimate, reportable data point for the quality-latency-resource
  trade-off question the assignment asks about: **a newer/larger
  code model is not automatically a better choice** if the deployment
  hardware can't hold it comfortably in memory — operational usability is
  itself part of "performance," not just tokens/sec once a model is
  loaded.

## Exercise 5 — RAG pipeline analysis

`eval/rag_trace.py` traced QUESTION → RETRIEVED CONTEXT → LLM RESPONSE
(`codellama:7b`) for six questions covering distinct retrieval scenarios.
Raw trace: `eval/results/rag_trace_codellama_7b_20260910_132240.json`.

| # | Question | Retrieved | Label |
|---|---|---|---|
| Q01 | Four deadlock conditions | 3 PYQ + 1 syllabus chunk, all on-topic | **Relevant retrieval → correct answer.** All four conditions listed correctly. |
| Q08 | Is Banker's Algorithm (asked every PYQ year) covered in lecture slides? | 1 syllabus + 3 PYQ chunks, exactly the mix needed | **Relevant retrieval → correct, nuanced answer.** Model correctly stated it is *not* covered in lecture slides per the syllabus, using both document types together — the app's core "gap analysis" feature working as intended. |
| Q09 | Same gap-analysis pattern for disk scheduling (SCAN/C-SCAN) | 1 syllabus + 3 PYQ chunks | **Relevant retrieval → correct answer**, again correctly citing self-study-only status. |
| Q14 | "Which OS topic is asked about most consistently across all 3 PYQ years?" | **2 of 4 retrieved chunks were from the unrelated Agentic AI handout**, not from any PYQ | **Irrelevant retrieval → wrong answer.** The model answered "preemptive and non-preemptive scheduling," which is not the best-supported answer (Banker's Algorithm / deadlock and disk scheduling actually recur in all three years' PYQs) — a direct consequence of the retriever surfacing the wrong document because the KB mixes two unrelated courses. |
| Q21 | "Does the syllabus cover TCP/IP networking?" (no answer in KB) | 1 syllabus + 3 Agentic AI handout chunks — none genuinely relevant | **Irrelevant retrieval → correct refusal anyway.** The model said no/not covered, correctly, despite the noisy context — the prompt's explicit "say so instead of guessing" instruction held here. |
| Q22 | "Who is the course faculty for CS301 Operating Systems, and their office room number?" (no such info anywhere in the OS-related KB) | 1 syllabus + 3 PYQ chunks — again nothing genuinely relevant to *this specific course's* faculty | **Irrelevant retrieval → confident hallucination.** The model answered "**Dr. Soharab Hossain Shaikh**, office E2 Building, 4th Floor, Cabin No. 81" — this is a **real fact from the knowledge base, but about the wrong course**: it's the Agentic AI (CSE3101) course faculty's real name and office, verified present in `Course_Handout__Agentic_AI_4.pdf`, misattributed to CS301 Operating Systems. This is the clearest hallucination example in the whole evaluation: not an invented fact, but a **cross-document misattribution** — the model didn't flag that the retrieved chunk was about a different course entirely. |

**RETRIEVAL QUALITY → CONTEXT QUALITY → LLM RESPONSE QUALITY.** The pattern
across all six traces is direct and consistent: every question where
retrieval pulled genuinely on-topic chunks (Q01, Q08, Q09) produced a
correct answer, and both misses (Q14, Q22) trace back to the retriever
surfacing Agentic-AI-handout chunks for OS-course questions — a structural
consequence of this app's knowledge base mixing two unrelated courses (the
real Agentic AI handout was added for demo purposes alongside placeholder
OS syllabus/PYQs; see README "Your real course documents"). Q21 shows RAG
is **not simply a checkbox**: even with irrelevant context retrieved, a
good prompt template can still produce a correct refusal — but Q22 shows
that guardrail is not reliable, especially when the irrelevant context
happens to contain real, specific-sounding facts (a name, an office
number) that make a wrong answer sound confidently correct. **Two concrete
fixes this analysis motivates:** (1) add a minimum cosine-similarity
threshold to `Retriever.search()` so it can return zero chunks instead of
always forcing exactly `top_k`, and (2) don't mix unrelated courses in one
knowledge base / index by `doc_type` **and course** so cross-course
retrieval leakage like Q14/Q22 can't happen.

## Exercise 6 — Repository-level code understanding

`eval/repo_understanding.py` built a **separate**, throwaway ad-hoc index
over this repo's own `.py`/config/doc files (same naive word-window
chunker, same embedder — deliberately not AST/call-graph aware) and asked
10 multi-file/component questions with `codellama:7b`. Raw output:
`eval/results/repo_understanding_codellama_7b_20260910_132750.json`.

**Answered correctly** (all single-file or single-fact lookups, where the
right chunk mostly answers the question on its own):
- Which files generate/store embeddings → correctly named `ingestion/build_index.py` and `rag/retrieval.py`.
- Which service does `/ask` call first → correctly said retrieval service, citing `services/orchestrator.py`.
- Which env var controls the LLM's model → correctly said `MODEL_NAME`.
- Which Dockerfile builds the retrieval service → correctly said `docker/Dockerfile.retrieval`.
- "If the retrieval service goes down..." → largely correct conceptually (orchestrator's requests fail, health endpoint shows it as unreachable), though it fabricated a specific-looking `curl`/JSON example that doesn't match the actual `/health` response shape in `services/orchestrator.py:125-134` — a smaller, cosmetic hallucination layered on an otherwise-correct answer.

**Failed — and instructively so:**
- **"Are there any automated test files in this repository?"** The system
  answered **"Yes"** and fabricated three specific, plausible-sounding but
  entirely nonexistent files: `tests/test_services.py`,
  `tests/test_ingestion.py`, `tests/test_app.py`, each with an invented
  description. There is **no `tests/` directory anywhere in this repo.**
  Retrieval pulled 4 chunks, all from `README.md`, none about tests — with
  no relevant chunk to ground an answer in, the model fabricated one
  instead of saying "no test files found." This is the single clearest
  illustration in the whole Week 4 evaluation of why **detecting absence
  is fundamentally hard for a chunk-similarity retriever**: there is no
  such thing as a chunk of "this doesn't exist" to retrieve.
- **"Which component performs cosine similarity search, and is that logic
  duplicated anywhere?"** The system correctly named `rag/retrieval.py`,
  but then claimed (incorrectly) that a `search()` method in `app.py`
  duplicates it. The **actual** duplication — `services/retrieval_service.py`
  independently reimplements the same cosine-similarity dot-product logic
  (confirmed by reading the two files directly) — was never mentioned. This
  is a cross-file "is X equivalent to Y" comparison, exactly the kind of
  question a symbol/call-graph-aware tool would answer immediately by
  finding all definitions of a similar operation, but which text-chunk
  similarity search has no structural way to do — it can only surface
  chunks that read similarly to the *question text*, not chunks that are
  *functionally equivalent to each other*.
- **"What is the difference between how `app.py` and
  `services/orchestrator.py` handle file uploads?"** The system invented a
  nonexistent `load_index()` function and a nonexistent `/index` GET route
  in `orchestrator.py` handling uploads. The real, correct answer — the one
  actually worth knowing — is that **`services/orchestrator.py` has no
  `/upload` route at all**; only the monolith `app.py` can ingest new
  documents (confirmed by reading both files directly). The system missed
  the real architectural gap and hallucinated a plausible-sounding fake one
  instead.
- **"Which files would need to change if chunk size went from 120 to
  300?"** Partially right (correctly named `ingestion/build_index.py`) but
  padded with an invented claim that `services/data_service.py`'s
  `get_chunks` endpoint would need updating "to return the same number of
  chunks" — that endpoint just serves whatever the index file already
  contains, unmodified; nothing there needs to change.

**Conclusion for Exercise 6:** the current LLM+RAG system reliably answers
questions that are really "which file mentions X" lookups, because that's
exactly what chunk-similarity retrieval is built for. It reliably fails or
hallucinates on anything requiring (a) noticing something is **absent**
across the whole repo, (b) comparing two files' behavior against each
other, or (c) recognizing that two pieces of code are **functionally**
related even if they don't read similarly. All three failure modes stem
from the same root cause: a flat list of independently-embedded text
chunks has no model of the repository's actual structure (call graphs,
symbol definitions/references, directory/module relationships) — which is
precisely what repository-level tools like Sourcegraph are built to
provide, and why this exercise sets up next week's material rather than
trying to solve the problem now.

## How to reproduce

```bash
source venv/bin/activate  # or use venv/bin/python3 directly
pip install -r requirements.txt -r eval/requirements-eval.txt

python3 eval/run_eval.py                                    # full 28Q x codellama:7b + starcoder2:3b
python3 eval/run_eval.py --models qwen3-coder:latest --sample 3   # only if your machine has ample free RAM
python3 eval/rag_trace.py --model codellama:7b               # Exercise 5
python3 eval/repo_understanding.py --model codellama:7b      # Exercise 6
```

Ollama must be running (`ollama serve`) with the models pulled:
`ollama pull codellama:7b && ollama pull starcoder2:3b`.
