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

> **Apple Silicon will not work.** Every published `sourcegraph/server` image
> is `amd64` only, and Sourcegraph's docs state ARM/ARM64 deployments are
> unsupported. The instance needs an x86_64 machine. A licence is only
> required above 10 users, so licensing is not a constraint for a student
> team — the architecture is.

### Which deployment

Sourcegraph documents two, and the difference matters on a small machine:

| | Docker Compose | **Single container** |
|---|---|---|
| Services | ~10 (postgres, redis, zoekt, gitserver, …) | one image, everything bundled |
| Realistic RAM | 16GB+ | runs on 8GB with swap |
| Supported for production | yes | no — "quick non-production environments" |

For a course demo indexing one small repository, the single container is the
right choice. A student AWS allowance tops out around 2 vCPU / 8GB, which is
not enough for the Compose stack.

### Pin the version — this is a real trap

The docs' quickstart shows `sourcegraph/server:7.4.2513`, but **that image
does not exist**. Single-container deployment was removed in Sourcegraph
7.0.0 and the image stopped being published; the docs page simply templates
the current product version into the command regardless.

The last published tag is **`6.12.5040`** (10 February 2026, 1.17GB, amd64).
Use it explicitly. `:latest` and any 7.x tag will fail to pull.

### On AWS

Launch **Ubuntu 24.04 LTS, 64-bit (x86)** on a `t3.large` (2 vCPU, 8GB) with
30GB+ gp3 storage. Never a `t4g`/Graviton type — those are ARM64.

Security group: inbound **22** and **7080**, both scoped to your own IP.
Never `0.0.0.0/0` — the first-run wizard lets whoever reaches it first claim
the admin account.

```bash
# swap matters on 8GB: startup is the memory peak, steady state is much lower
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu && newgrp docker

docker run -d --name sourcegraph --restart unless-stopped \
  --publish 7080:7080 \
  --volume ~/.sourcegraph/config:/etc/sourcegraph \
  --volume ~/.sourcegraph/data:/var/opt/sourcegraph \
  sourcegraph/server:6.12.5040
```

Follow `docker logs -f sourcegraph` until the logo appears, then open
`http://<public-ip>:7080` and **create the admin account immediately**.

Add the repository under Site admin → Manage code hosts → GitHub, with a
GitHub token (classic, `public_repo` scope is enough — the repo is public):

```json
{ "url": "https://github.com", "token": "<token>", "repos": ["AkshitaJ0904/pyq-handout-bot"] }
```

Queries return nothing until cloning and indexing finish, which is the most
common false alarm. Watch Site admin → Repositories.

## Pointing the app at it

Create an access token under **Settings → Access tokens**, then:

```bash
export SOURCEGRAPH_URL=http://<host>:7080     # e.g. http://13.200.1.20:7080
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
export SOURCEGRAPH_URL=http://<ec2-public-ip>:7080
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
