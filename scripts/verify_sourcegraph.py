#!/usr/bin/env python3
"""Check that a Sourcegraph instance is reachable, has this repo indexed, and
answers the three Week 4 questions correctly.

Run it after setting up the server, before wiring the app to it:

    export SOURCEGRAPH_URL=http://<ec2-public-ip>
    export SOURCEGRAPH_TOKEN=sgp_...
    export SOURCEGRAPH_REPO=github.com/AkshitaJ0904/pyq-handout-bot
    python3 scripts/verify_sourcegraph.py

Exits non-zero if anything fails, so it can gate a CI job later. It reads the
token from the environment and never prints it.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from codesearch.base import CodeQuery, QueryKind  # noqa: E402
from codesearch.sourcegraph_backend import SourcegraphBackend  # noqa: E402

# The three failures from eval/REPORT.md, Exercise 6, and what a correct
# structural answer looks like for each.
CHECKS = [
    ("absence", CodeQuery(QueryKind.FILES, "test_app.py"), "expect ZERO matches", lambda r: len(r.matches) == 0),
    ("absence", CodeQuery(QueryKind.FILES, "test_*.py"), "expect the real test file", lambda r: any("test_metrics_wiring" in m.file for m in r.matches)),
    ("duplication", CodeQuery(QueryKind.SYMBOL, "search"), "expect retrieval.py and retrieval_service.py", lambda r: {"rag/retrieval.py", "services/retrieval_service.py"} <= {m.file for m in r.matches}),
    ("cross-file", CodeQuery(QueryKind.TEXT, r"@app\.route\(.\/upload"), "expect app.py and orchestrator.py", lambda r: len({m.file for m in r.matches}) >= 2),
]

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def fail(msg, hint=""):
    print(f"{RED}FAIL{RESET}  {msg}")
    if hint:
        print(f"      {DIM}{hint}{RESET}")
    return False


def main():
    url = os.environ.get("SOURCEGRAPH_URL", "").rstrip("/")
    token = os.environ.get("SOURCEGRAPH_TOKEN", "")
    repo = os.environ.get("SOURCEGRAPH_REPO", "")

    print(f"\nSourcegraph verification\n{'-' * 52}")
    print(f"  URL   {url or '(unset)'}")
    print(f"  repo  {repo or '(unset)'}")
    print(f"  token {'set (' + str(len(token)) + ' chars)' if token else '(unset)'}\n")

    if not url:
        return fail("SOURCEGRAPH_URL is not set.",
                    "export SOURCEGRAPH_URL=http://<ec2-public-ip>")

    # 1. reachable at all?
    try:
        requests.get(f"{url}/.api/graphql", timeout=10)
        print(f"{GREEN}OK{RESET}    instance reachable")
    except requests.exceptions.RequestException as e:
        return fail(f"cannot reach {url} ({type(e).__name__}).",
                    "Check the instance is running and your IP is allowed in the security group.")

    if not token:
        return fail("SOURCEGRAPH_TOKEN is not set.",
                    "Create one under Settings > Access tokens in the Sourcegraph UI.")

    backend = SourcegraphBackend(url=url, token=token, repo=repo)

    # 2. does it authenticate, and is the repo indexed?
    probe = backend.run(CodeQuery(QueryKind.TEXT, "def"))
    if probe.error:
        return fail(f"query rejected: {probe.error}",
                    "A 401 means the token is wrong; 'repo not found' means the code host "
                    "connection has not finished cloning yet.")
    if not probe.matches:
        return fail("connected and authenticated, but the repo returned no results.",
                    f"Is {repo or '(SOURCEGRAPH_REPO unset)'} added and finished indexing? "
                    "Check Site admin > Repositories.")
    print(f"{GREEN}OK{RESET}    repo indexed ({len(probe.matches)} matches on a probe query)\n")

    # 3. the three Week 4 questions
    passed = True
    for label, query, expectation, check in CHECKS:
        result = backend.run(query)
        if result.error:
            passed = fail(f"[{label}] {query.kind.value}:{query.term} errored: {result.error}")
            continue
        if check(result):
            print(f"{GREEN}OK{RESET}    [{label}] {query.kind.value}:{query.term} "
                  f"{DIM}-> {len(result.matches)} match(es){RESET}")
        else:
            passed = fail(f"[{label}] {query.kind.value}:{query.term} -- {expectation}, "
                          f"got {len(result.matches)}: "
                          f"{', '.join(sorted({m.file for m in result.matches}))[:120]}")

    print()
    if passed:
        print(f"{GREEN}All checks passed.{RESET} Point the app at it:\n"
              f"  {DIM}SOURCEGRAPH_URL={url} SOURCEGRAPH_TOKEN=... "
              f"SOURCEGRAPH_REPO={repo} python3 app.py{RESET}\n")
    else:
        print(f"{YELLOW}Some checks failed -- see the hints above.{RESET}\n")
    return passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
