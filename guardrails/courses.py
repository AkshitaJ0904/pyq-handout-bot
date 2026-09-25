"""Which course does a document (or a question) belong to?

The knowledge base currently mixes two unrelated courses -- 21 of its 30
chunks are the Agentic AI handout and only 3 are the Operating Systems
syllabus. eval/REPORT.md Exercise 5 traced a confident hallucination (Q22)
directly to that mixing: asked for the CS301 Operating Systems faculty, the
model returned a real name and cabin number belonging to the *Agentic AI*
course.

Course provenance is therefore a first-class property of every chunk, not a
detail. It is derived from the filename rather than stored in the index, so
this works against the existing index without a re-embed.
"""
import re

UNKNOWN = "unknown"

# (course label, filename pattern, phrases that name it in a question)
COURSES = [
    ("Agentic AI", re.compile(r"agentic", re.I),
     (r"agentic\s*ai", r"cse\s*3101")),
    ("Operating Systems", re.compile(r"\bos[_\-]|operating", re.I),
     (r"operating\s*systems?", r"\bcs\s*301\b", r"\bos\b")),
]


def course_of_file(source_file: str) -> str:
    for label, file_pat, _ in COURSES:
        if file_pat.search(source_file or ""):
            return label
    return UNKNOWN


def courses_in(chunks) -> set:
    return {course_of_file(c.get("source_file", "")) for c in (chunks or [])}


def course_named_in(question: str):
    """The course the question explicitly asks about, or None if it names none."""
    q = question or ""
    for label, _, phrases in COURSES:
        for p in phrases:
            if re.search(p, q, re.I):
                return label
    return None
