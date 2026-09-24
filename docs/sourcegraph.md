# Sourcegraph setup and semantic code navigation

## Why the project needs it

Week 4 (`eval/REPORT.md`, Exercise 6) pointed the Week 3 RAG pipeline at this
repository and measured three failure modes. All three trace to one cause: a
flat list of independently embedded text chunks has no model of the
repository's structure.

| Question | What chunk similarity did | Why it failed |
|---|---|---|
| "Are there any automated test files?" | Invented `tests/test_app.py`, `tests/test_services.py`, `tests/test_ingestion.py` | There is no chunk of "this doesn't exist" to retrieve |
| "Is the cosine similarity logic duplicated?" | Named `app.py`, which does not duplicate it; missed the files that do | Two functions can be functionally identical without reading alike |
| "How do `app.py` and `orchestrator.py` differ on upload?" | Invented a `load_index()` and an `/index` route | Cross-file comparison needs both files' structure at once |

Structural code search closes exactly these three. The **Repo Q&A** tab runs
both paths on the same question and shows them side by side, so the
before/after is visible rather than asserted.

## Architecture

The app never depends on which backend is running:

```
question -> codesearch/planner.py   (question -> structured code query)
         -> CodeSearchBackend       (sourcegraph | local-ast)
         -> matches -> LLM, grounded in facts rather than excerpts
```

- `codesearch/planner.py` — "query formulation": maps the question to a query
  kind (`files` / `symbol` / `routes` / `text` / `pattern`). Rules are explicit
  rather than LLM-generated so they are deterministic, testable without Ollama,
  and auditable — the chosen query and its rationale are shown in the UI.
- `codesearch/sourcegraph_backend.py` — the real backend, over Sourcegraph's
  GraphQL API.
- `codesearch/local_backend.py` — an AST fallback so the demo never dies. It
  is **not** a Sourcegraph replacement: this repo's Python only, no
  cross-repository index, no commit history, and symbol *definitions* only,
  not references.

Which backend answered is reported in the response and shown as a badge in the
UI. The fallback is never silent.

## Running Sourcegraph

> **Apple Silicon will not work.** Sourcegraph's docs state that deployments on
> ARM/ARM64 are unsupported, which rules out every M-series Mac. A Sourcegraph
> license is only required above 10 users, so licensing is not a constraint for
> a student team — the architecture is.

Options, in order of least effort:

1. **An x86_64 machine** — any Intel/AMD laptop or desktop on the team.
2. **A cloud VM** — Sourcegraph publishes deployment guides for AWS, Google
   Cloud and DigitalOcean. A small VM on student credits is enough for one
   repository.
3. **Sourcegraph Cloud** — managed, no infrastructure to run.

Docker Compose is the supported single-node path:

```bash
git clone --branch release https://github.com/sourcegraph/deploy-sourcegraph-docker
cd deploy-sourcegraph-docker/docker-compose
docker compose up -d
```

Then open the host on port 80, create the admin account, and add this
repository under **Site admin → Repositories → Manage code hosts** (GitHub,
`AkshitaJ0904/pyq-handout-bot`). Wait for it to finish cloning and indexing.

## Pointing the app at it

Create an access token under **Settings → Access tokens**, then:

```bash
export SOURCEGRAPH_URL=http://<host>          # e.g. http://192.168.1.20
export SOURCEGRAPH_TOKEN=sgp_xxxxxxxxxxxx
export SOURCEGRAPH_REPO=github.com/AkshitaJ0904/pyq-handout-bot
python3 app.py
```

The badge in the Repo Q&A tab flips from "local AST fallback" to
"Sourcegraph". Nothing else changes — same questions, same UI, same
comparison.

## Verifying it

After the server is up, before touching the app:

```bash
export SOURCEGRAPH_URL=http://<ec2-public-ip>
export SOURCEGRAPH_TOKEN=sgp_...
export SOURCEGRAPH_REPO=github.com/AkshitaJ0904/pyq-handout-bot
python3 scripts/verify_sourcegraph.py
```

It checks reachability, authentication, whether the repo finished indexing,
and then the three Week 4 questions — exiting non-zero on failure, so it can
gate a CI job later. Each failure prints the likely cause (a 401 means the
token; "no results" usually means the code host is still cloning).


The three Week 4 questions are wired to the quick-chips in the Repo Q&A tab
(`absence`, `duplication`, `cross-file`). Expected results:

| Query | Expected |
|---|---|
| `files:test_*.py` | exactly `tests/test_metrics_wiring.py` |
| `pattern:topk_similarity` | `rag/retrieval.py`, `services/retrieval_service.py`, `eval/repo_understanding.py` |
| `routes:upload` | `/upload` in both `app.py` and `services/orchestrator.py` |

`tests/test_codesearch.py` asserts all three against the local backend, so a
regression fails the suite rather than the demo.

## Two findings worth reporting

1. **Structural search found a third copy of the duplicated similarity logic.**
   The Week 4 report identified two (`rag/retrieval.py` and
   `services/retrieval_service.py`) by reading the files manually. The AST
   pattern also finds `eval/repo_understanding.py` — the manual read missed a
   third of the answer.

2. **The Week 4 report is now stale on `/upload`.** It states that
   `services/orchestrator.py` has no `/upload` route; the route was added on
   17 September. A single `routes:upload` query settles it in milliseconds,
   which is the maintenance argument for structural search over a prose
   report: the query re-runs, the report does not.
