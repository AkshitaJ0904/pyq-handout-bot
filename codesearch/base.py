"""Shared types for the code-search backends.

Week 4 (eval/REPORT.md, Exercise 6) measured three failure modes of chunk
similarity search over a repository:

  1. it cannot notice something is ABSENT -- there is no chunk of
     "this doesn't exist" to retrieve, so the model invented a test suite;
  2. it cannot compare two files' behaviour against each other;
  3. it cannot see that two pieces of code are FUNCTIONALLY equivalent when
     they don't read similarly.

All three come from the same root cause: a flat list of independently
embedded text chunks has no model of the repository's structure. These
queries are structural rather than lexical, which is what closes the gap.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class QueryKind(str, Enum):
    FILES = "files"        # does anything match this path glob? (absence)
    SYMBOL = "symbol"      # where is this function/class defined?
    ROUTES = "routes"      # which HTTP routes does this repo register?
    TEXT = "text"          # literal / regex search
    PATTERN = "pattern"    # a named structural pattern (see local_backend)


@dataclass
class CodeQuery:
    kind: QueryKind
    term: str
    rationale: str = ""     # why the planner chose this query, shown in the UI

    def as_sourcegraph(self, repo: str) -> str:
        """Render as Sourcegraph query syntax."""
        scope = f"repo:{repo}" if repo else ""
        if self.kind is QueryKind.FILES:
            return f"{scope} file:{self.term} select:file type:path".strip()
        if self.kind is QueryKind.SYMBOL:
            return f"{scope} type:symbol {self.term}".strip()
        if self.kind is QueryKind.ROUTES:
            return f"{scope} /@(app|bp)\\.route\\(/ type:regexp".strip()
        return f"{scope} /{self.term}/ type:regexp".strip()


@dataclass
class Match:
    file: str
    line: Optional[int] = None
    symbol: Optional[str] = None
    kind: str = "text"
    snippet: str = ""


@dataclass
class SearchResult:
    query: CodeQuery
    backend: str
    matches: List[Match] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def found(self) -> bool:
        return bool(self.matches)

    def to_dict(self):
        return {
            "kind": self.query.kind.value,
            "term": self.query.term,
            "rationale": self.query.rationale,
            "backend": self.backend,
            "found": self.found,
            "count": len(self.matches),
            "error": self.error,
            "matches": [
                {"file": m.file, "line": m.line, "symbol": m.symbol,
                 "kind": m.kind, "snippet": m.snippet}
                for m in self.matches
            ],
        }


class CodeSearchBackend:
    """Interface both backends implement, so the app never depends on which
    one is actually running."""

    name = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def run(self, query: CodeQuery) -> SearchResult:
        raise NotImplementedError
