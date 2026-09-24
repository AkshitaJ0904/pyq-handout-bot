"""Sourcegraph backend -- queries a Sourcegraph instance over its GraphQL API.

Configure with environment variables:

    SOURCEGRAPH_URL    e.g. http://localhost:7080   (or a teammate's host)
    SOURCEGRAPH_TOKEN  access token from Settings > Access tokens
    SOURCEGRAPH_REPO   the repo to scope queries to, as Sourcegraph names it,
                       e.g. github.com/AkshitaJ0904/pyq-handout-bot

When SOURCEGRAPH_URL is unset or the instance is unreachable, `available()`
returns False and the app falls back to the local AST backend, so the
comparison still runs. See docs/sourcegraph.md for the server setup.
"""
import os

import requests

from codesearch.base import CodeQuery, CodeSearchBackend, Match, SearchResult

SEARCH_QUERY = """
query Search($query: String!) {
  search(query: $query, version: V3, patternType: standard) {
    results {
      matchCount
      limitHit
      results {
        __typename
        ... on FileMatch {
          repository { name }
          file { path }
          lineMatches { preview lineNumber }
          symbols { name kind location { range { start { line } } } }
        }
      }
    }
  }
}
"""


class SourcegraphBackend(CodeSearchBackend):
    name = "sourcegraph"

    def __init__(self, url=None, token=None, repo=None, timeout=15):
        self.url = (url or os.environ.get("SOURCEGRAPH_URL", "")).rstrip("/")
        self.token = token or os.environ.get("SOURCEGRAPH_TOKEN", "")
        self.repo = repo or os.environ.get("SOURCEGRAPH_REPO", "")
        self.timeout = timeout
        self._available = None

    # -- availability is probed once and cached, so an unconfigured or
    #    unreachable instance costs one failed request per process, not one
    #    per question.
    def available(self) -> bool:
        if self._available is not None:
            return self._available
        if not self.url:
            self._available = False
            return False
        try:
            r = requests.get(f"{self.url}/.api/graphql", timeout=self.timeout)
            self._available = r.status_code < 500
        except requests.exceptions.RequestException:
            self._available = False
        return self._available

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"token {self.token}"
        return h

    def run(self, query: CodeQuery) -> SearchResult:
        sg_query = query.as_sourcegraph(self.repo)
        if not self.available():
            return SearchResult(query=query, backend=self.name, matches=[],
                                error="No Sourcegraph instance configured or reachable "
                                      "(set SOURCEGRAPH_URL).")
        try:
            resp = requests.post(
                f"{self.url}/.api/graphql",
                json={"query": SEARCH_QUERY, "variables": {"query": sg_query}},
                headers=self._headers(), timeout=self.timeout,
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.exceptions.RequestException as e:
            return SearchResult(query=query, backend=self.name, matches=[], error=str(e))

        if body.get("errors"):
            return SearchResult(query=query, backend=self.name, matches=[],
                                error="; ".join(e.get("message", "") for e in body["errors"]))

        results = (body.get("data") or {}).get("search", {}).get("results", {}) or {}
        matches = []
        for item in results.get("results", []) or []:
            path = (item.get("file") or {}).get("path", "")
            for sym in item.get("symbols") or []:
                line = (((sym.get("location") or {}).get("range") or {})
                        .get("start") or {}).get("line")
                matches.append(Match(file=path, line=(line + 1) if line is not None else None,
                                     symbol=sym.get("name"),
                                     kind=(sym.get("kind") or "symbol").lower(),
                                     snippet=sym.get("name", "")))
            for lm in item.get("lineMatches") or []:
                matches.append(Match(file=path, line=(lm.get("lineNumber") or 0) + 1,
                                     kind="text", snippet=(lm.get("preview") or "").strip()[:200]))
            if not (item.get("symbols") or item.get("lineMatches")):
                matches.append(Match(file=path, kind="file", snippet=path))

        return SearchResult(query=query, backend=self.name, matches=matches)
