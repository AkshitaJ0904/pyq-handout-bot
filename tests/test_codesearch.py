"""Tests for structural code search.

These assert the three repository-understanding failures that Week 4
documented (eval/REPORT.md, Exercise 6) are actually closed -- absence,
functional equivalence, and cross-file comparison. They run against this
repository itself, with no server and no model, which is the point: a
structural query has a determinate answer, so it can be asserted on.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codesearch.base import CodeQuery, QueryKind  # noqa: E402
from codesearch.local_backend import LocalASTBackend  # noqa: E402
from codesearch.planner import plan  # noqa: E402
from codesearch.sourcegraph_backend import SourcegraphBackend  # noqa: E402

backend = LocalASTBackend()


def run(kind, term):
    return backend.run(CodeQuery(kind=kind, term=term))


# --- Week 4 failure 1: absence ------------------------------------------
def test_absence_is_answerable():
    """Chunk similarity invented tests/test_app.py, test_services.py and
    test_ingestion.py because there is no chunk of 'this does not exist'.
    A path query returns an empty set, which IS the answer."""
    assert run(QueryKind.FILES, "test_app.py").matches == []
    assert run(QueryKind.FILES, "test_services.py").matches == []
    assert run(QueryKind.FILES, "test_ingestion.py").matches == []


def test_the_real_test_file_is_found():
    files = {m.file for m in run(QueryKind.FILES, "test_*.py").matches}
    assert "tests/test_metrics_wiring.py" in files


# --- Week 4 failure 2: functional equivalence ---------------------------
def test_finds_duplicated_similarity_logic():
    """The report found this duplication by reading two files manually and
    missed a third. Matching on AST shape finds all of them."""
    files = {m.file for m in run(QueryKind.PATTERN, "topk_similarity").matches}
    assert {"rag/retrieval.py", "services/retrieval_service.py"} <= files
    assert "eval/repo_understanding.py" in files, "the occurrence the manual read missed"


def test_pattern_matches_on_structure_not_wording():
    """All matches are functions that both score and rank -- found by shape,
    not by sharing vocabulary with the question."""
    for m in run(QueryKind.PATTERN, "topk_similarity").matches:
        assert m.kind == "function" and m.line


# --- Week 4 failure 3: cross-file comparison ----------------------------
def test_upload_routes_across_files():
    """'How do app.py and orchestrator.py differ on upload?' -- the report
    answered by inventing a load_index() and an /index route. Enumerating
    routes answers it from fact."""
    got = {(m.file, m.symbol) for m in run(QueryKind.ROUTES, "upload").matches}
    assert ("app.py", "/upload") in got
    assert ("services/orchestrator.py", "/upload") in got


def test_symbol_exact_matches_rank_first():
    matches = run(QueryKind.SYMBOL, "search").matches
    assert matches and matches[0].symbol == "search"


# --- query planning ------------------------------------------------------
@pytest.mark.parametrize("question,expected", [
    ("Are there any automated test files in this repository?", QueryKind.FILES),
    ("Is that logic duplicated anywhere in the codebase?", QueryKind.PATTERN),
    ("How do app.py and orchestrator.py handle file uploads?", QueryKind.ROUTES),
    ("Which environment variable controls the model?", QueryKind.TEXT),
])
def test_planner_picks_the_right_query_kind(question, expected):
    assert expected in {q.kind for q in plan(question)}


def test_planner_ignores_english_words_that_follow_component():
    """'Which component performs ...' must not plan a symbol lookup for
    'performs'."""
    terms = {q.term for q in plan("Which component performs the cosine similarity search?")}
    assert "performs" not in terms


def test_every_planned_query_explains_itself():
    for q in plan("Are there any automated test files in this repository?"):
        assert q.rationale, "the UI shows the rationale, so it must not be empty"


# --- Sourcegraph backend -------------------------------------------------
def test_sourcegraph_unconfigured_is_unavailable_not_crashing():
    sg = SourcegraphBackend(url="", token="", repo="")
    assert sg.available() is False
    result = sg.run(CodeQuery(QueryKind.SYMBOL, "search"))
    assert result.matches == [] and "SOURCEGRAPH_URL" in result.error


def test_sourcegraph_query_rendering():
    repo = "github.com/AkshitaJ0904/pyq-handout-bot"
    assert "type:symbol" in CodeQuery(QueryKind.SYMBOL, "search").as_sourcegraph(repo)
    assert "file:test_*.py" in CodeQuery(QueryKind.FILES, "test_*.py").as_sourcegraph(repo)
    assert repo in CodeQuery(QueryKind.TEXT, "os.environ").as_sourcegraph(repo)
