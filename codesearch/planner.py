"""Question -> structured code query.

The course calls this "query formulation": the user asks in English, and the
system decides which structural query actually answers it. The rules are
deliberately explicit rather than LLM-generated, for three reasons -- they
are deterministic, they are testable without Ollama running, and the chosen
query is shown in the UI so the reasoning is auditable rather than hidden
inside a model.

A question can map to several queries; all of them run and the results are
merged, because "is this logic duplicated?" needs both the symbol lookup and
the structural-pattern lookup to be answered honestly.
"""
import re

from codesearch.base import CodeQuery, QueryKind

RULES = [
    (r"\btest(s| file| files|ing)?\b.*\b(exist|any|there|repo|repository)\b|"
     r"\b(any|are there)\b.*\btest\b",
     [(QueryKind.FILES, "test_*.py", "Absence is a path question, not a similarity question: "
                                     "glob the repo for test files and count what comes back."),
      (QueryKind.FILES, "*_test.py", "Second common test-file naming convention.")]),

    (r"\bduplicat|\bsame (logic|code)\b|\bequivalent\b|\breimplement",
     [(QueryKind.PATTERN, "topk_similarity",
       "Duplication is a structural question: match on AST shape (similarity score + "
       "top-k rank) so functionally identical code is found even when it reads differently."),
      (QueryKind.SYMBOL, "search", "Also list every definition of the symbol by name.")]),

    (r"\broute\b|\bendpoint\b|\bupload|\bapi\b",
     [(QueryKind.ROUTES, "", "Enumerate every registered HTTP route in the repo, so "
                             "'does file X expose route Y' is answered by the list, not by inference.")]),

    (r"\benvironment variable\b|\benv var\b",
     [(QueryKind.TEXT, r"os\.environ", "Environment access is a literal pattern.")]),

    (r"\bdockerfile\b|\bcontainer\b|\bdocker\b",
     [(QueryKind.FILES, "Dockerfile*", "Dockerfiles are found by path, not by prose.")]),
]

# "which/where is <name> defined" -> symbol lookup on the captured name.
# Only accept things that actually look like identifiers -- snake_case,
# CamelCase, or an explicit call -- otherwise an ordinary English word after
# "component"/"class" gets mistaken for a symbol ("component performs ...").
SYMBOL_RE = re.compile(
    r"\b(?:function|method|class|component)\s+(?:called\s+|named\s+)?[`\'\"]?(\w+)[`\'\"]?"
    r"|[`\'\"](\w+)\(\)[`\'\"]|\b(\w+)\(\)")


def _looks_like_identifier(name: str) -> bool:
    return bool(name) and len(name) > 2 and (
        "_" in name or name[:1].isupper() or any(c.isupper() for c in name[1:])
    )


ROUTE_WORDS = ("upload", "ask", "compare", "health", "files", "search", "reindex", "reload")


def _route_term(q: str) -> str:
    """If the question names a specific route, scope the route listing to it.
    Otherwise return "" and list them all."""
    for w in ROUTE_WORDS:
        if w in q:
            return w
    return ""


def plan(question: str):
    """Return the list of CodeQuery objects that answer `question`."""
    q = question.lower()
    out = []
    for pattern, specs in RULES:
        if re.search(pattern, q):
            for k, t, r in specs:
                if k is QueryKind.ROUTES and not t:
                    t = _route_term(q)
                out.append(CodeQuery(kind=k, term=t, rationale=r))

    m = SYMBOL_RE.search(question)
    if m:
        name = next((g for g in m.groups() if g), None)
        if name and _looks_like_identifier(name) and not any(
            c.kind is QueryKind.SYMBOL and c.term == name for c in out
        ):
            out.append(CodeQuery(QueryKind.SYMBOL, name,
                                 "The question names a symbol, so look up where it is defined."))

    if not out:
        # Nothing matched a rule: fall back to a literal search on the
        # question's most distinctive word rather than guessing.
        words = [w for w in re.findall(r"[a-z_]{5,}", q)
                 if w not in {"which", "where", "would", "there", "about", "repository", "codebase"}]
        term = max(words, key=len) if words else question.strip()[:40]
        out.append(CodeQuery(QueryKind.TEXT, re.escape(term),
                             "No structural rule matched, so fall back to a literal search "
                             "on the question's most distinctive term."))
    return out
