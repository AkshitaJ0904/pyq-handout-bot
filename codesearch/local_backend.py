"""AST-backed local code search.

This is the fallback backend: it needs no server, no Docker and no extra
dependency, so the repo-understanding comparison is always demoable. It
parses the repository's Python with the `ast` module rather than embedding
it, which is what lets it answer the three question types chunk similarity
cannot -- absence, cross-file comparison, and functional equivalence.

It is deliberately NOT a Sourcegraph replacement: it only understands this
repo's Python, has no cross-repository index, no commit history and no
symbol *references* (only definitions). Where a real Sourcegraph instance is
configured, sourcegraph_backend.py is preferred and this stays as a fallback.
"""
import ast
import fnmatch
import re
from pathlib import Path

from codesearch.base import CodeQuery, CodeSearchBackend, Match, QueryKind, SearchResult

ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules", ".pytest_cache"}
CODE_SUFFIXES = {".py"}
TEXT_SUFFIXES = {".py", ".txt", ".md", ".yml", ".yaml", ".json", ".sh", ".html", ".css", ".js"}

# Named structural patterns: an AST shape, not a string. "topk_similarity"
# is the one Week 4 needed -- a function that both computes a similarity
# score and takes the top-k of it. Two functions match this even when they
# share almost no vocabulary, which is exactly what embedding search misses.
PATTERNS = {
    "topk_similarity": {
        "label": "top-k similarity ranking (matmul/dot + argsort)",
        "needs": ({"MatMult", "dot"}, {"argsort", "argpartition"}),
    },
}


def _iter_files(suffixes):
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(part in EXCLUDE_DIRS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def _rel(path):
    return str(path.relative_to(ROOT))


def _snippet(lines, lineno, width=0):
    i = max(0, lineno - 1)
    return lines[i].strip() if i < len(lines) else ""


class LocalASTBackend(CodeSearchBackend):
    name = "local-ast"

    def available(self) -> bool:
        return True

    def run(self, query: CodeQuery) -> SearchResult:
        handler = {
            QueryKind.FILES: self._files,
            QueryKind.SYMBOL: self._symbol,
            QueryKind.ROUTES: self._routes,
            QueryKind.TEXT: self._text,
            QueryKind.PATTERN: self._pattern,
        }[query.kind]
        try:
            return SearchResult(query=query, backend=self.name, matches=handler(query.term))
        except Exception as e:  # a broken file must not take the whole search down
            return SearchResult(query=query, backend=self.name, matches=[], error=str(e))

    # -- absence: the question chunk-similarity structurally cannot answer --
    def _files(self, glob):
        out = []
        for path in _iter_files(TEXT_SUFFIXES | {".pdf"}):
            rel = _rel(path)
            if fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(path.name, glob):
                out.append(Match(file=rel, kind="file", snippet=rel))
        return out

    def _symbol(self, name):
        exact, partial = [], []
        for path in _iter_files(CODE_SUFFIXES):
            try:
                src = path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(src)
            except SyntaxError:
                continue
            lines = src.splitlines()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name == name or name.lower() in node.name.lower():
                        kind = "class" if isinstance(node, ast.ClassDef) else "function"
                        m = Match(file=_rel(path), line=node.lineno, symbol=node.name,
                                  kind=kind, snippet=_snippet(lines, node.lineno))
                        (exact if node.name == name else partial).append(m)
        # Exact definitions first: a substring hit on an unrelated class is
        # far less useful than the symbol actually asked about.
        return exact + partial

    def _routes(self, term):
        """Every Flask route in the repo, optionally filtered by path. Answers
        'does orchestrator.py expose /upload?' definitively, in one query."""
        out = []
        pat = re.compile(r"@\w+\.route\(\s*[\"']([^\"']+)[\"']")
        for path in _iter_files(CODE_SUFFIXES):
            src = path.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(src.splitlines(), start=1):
                m = pat.search(line)
                if m and (not term or term.strip("/") in m.group(1).strip("/")):
                    out.append(Match(file=_rel(path), line=i, symbol=m.group(1),
                                     kind="route", snippet=line.strip()))
        return out

    def _text(self, pattern):
        out = []
        rx = re.compile(pattern, re.IGNORECASE)
        for path in _iter_files(TEXT_SUFFIXES):
            src = path.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(src.splitlines(), start=1):
                if rx.search(line):
                    out.append(Match(file=_rel(path), line=i, kind="text", snippet=line.strip()[:200]))
        return out

    # -- functional equivalence: matches on AST shape, not on wording --
    def _pattern(self, name):
        spec = PATTERNS.get(name)
        if spec is None:
            raise ValueError(f"unknown structural pattern: {name}")
        want_score, want_rank = spec["needs"]
        out = []
        for path in _iter_files(CODE_SUFFIXES):
            try:
                src = path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(src)
            except SyntaxError:
                continue
            lines = src.splitlines()
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                ops, calls = set(), set()
                for sub in ast.walk(node):
                    if isinstance(sub, ast.BinOp):
                        ops.add(type(sub.op).__name__)
                    elif isinstance(sub, ast.Call):
                        f = sub.func
                        calls.add(getattr(f, "attr", None) or getattr(f, "id", None))
                seen = ops | calls
                if (seen & want_score) and (seen & want_rank):
                    out.append(Match(file=_rel(path), line=node.lineno, symbol=node.name,
                                     kind="function", snippet=_snippet(lines, node.lineno)))
        return out
