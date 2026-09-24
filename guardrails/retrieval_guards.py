"""Stage 2 -- checks on what retrieval returned, before generation.

The retriever has no notion of "nothing relevant": np.argsort always returns
exactly top_k chunks, so a question the knowledge base cannot answer still
arrives at the model wrapped in four confident-looking excerpts.
eval/REPORT.md Exercise 5 recommended a minimum-similarity threshold for
exactly this.

**Measured caveat, kept deliberately visible.** The threshold turns out to be
a weak signal on this knowledge base. Across the 28-question evaluation set,
answerable questions score 0.245-0.749 (max cosine) while the three
unanswerable probes score 0.307, 0.485 and 0.553 -- the probes sit *inside*
the answerable range. At 0.32 the threshold keeps 22/24 answerable questions
and catches only 1 of 3 probes. No threshold separates the classes, because
an unanswerable question can be highly similar to documents that simply do
not contain the answer (Q23 asks about attendance figures the handout never
records, and scores higher than most real questions).

The threshold is kept because it cheaply catches genuinely off-topic input,
but the guard that actually works on this failure mode is the course
provenance check below, and the output-stage checks after generation.
"""
from guardrails.base import Stage, allow, block, warn
from guardrails.courses import UNKNOWN, course_named_in, course_of_file

# Chosen from the measurement above: keeps 22/24 answerable, catches Q21.
MIN_RETRIEVAL_SCORE = 0.32

# Questions asking for a specific attributable fact about a specific course.
# These are the ones Q22 got catastrophically wrong: a real person's name and
# cabin number, taken from the wrong course's handout.
ATTRIBUTION_TRIGGERS = (
    "faculty", "instructor", "professor", "teacher", "lecturer", "taught by",
    "office", "room", "cabin", "email", "contact", "phone", "credits",
    "coordinator", "who is", "who teaches",
)


def check_retrieval_confidence(question, chunks):
    if not chunks:
        return block("retrieval_confidence", Stage.RETRIEVAL,
                     "Retrieval returned no chunks at all.",
                     "I couldn't find anything about that in your uploaded documents.",
                     top_score=None)
    top = max(c.get("score", 0.0) for c in chunks)
    if top < MIN_RETRIEVAL_SCORE:
        return block("retrieval_confidence", Stage.RETRIEVAL,
                     f"Best chunk scored {top:.3f}, below the "
                     f"{MIN_RETRIEVAL_SCORE} minimum.",
                     "I couldn't find anything relevant to that in your uploaded "
                     "documents, so I'd rather not guess.",
                     top_score=round(top, 3), threshold=MIN_RETRIEVAL_SCORE)
    return allow("retrieval_confidence", Stage.RETRIEVAL,
                 f"Best chunk scored {top:.3f}.")


def check_course_provenance(question, chunks):
    """The Q22 guard.

    If the question names a course and the retrieved context includes chunks
    from a *different* course, the model can answer an attribution question
    with a real-but-wrong fact. For attribution-style questions this is a
    block; otherwise it is a warning, because mixed context is often harmless
    for conceptual questions.
    """
    if not chunks:
        return allow("course_provenance", Stage.RETRIEVAL, "No chunks to check.")

    asked = course_named_in(question)
    present = {course_of_file(c.get("source_file", "")) for c in chunks}
    foreign = {c for c in present if c not in (UNKNOWN, asked)} if asked else set()
    attribution = any(t in (question or "").lower() for t in ATTRIBUTION_TRIGGERS)

    if asked and foreign and attribution:
        offending = sorted({c["source_file"] for c in chunks
                            if course_of_file(c.get("source_file", "")) in foreign})
        return block("course_provenance", Stage.RETRIEVAL,
                     f"Question asks about {asked!r} and requests an attributable "
                     f"fact, but retrieved context includes {sorted(foreign)} "
                     f"documents ({offending}). This is the Q22 failure mode.",
                     f"I can't answer that reliably — the documents I found are "
                     f"partly about a different course ({', '.join(sorted(foreign))}), "
                     f"and I won't attribute their details to {asked}.",
                     asked=asked, foreign=sorted(foreign), files=offending)

    if len(present - {UNKNOWN}) > 1:
        return warn("course_provenance", Stage.RETRIEVAL,
                    f"Context mixes {sorted(present - {UNKNOWN})}; acceptable for a "
                    f"conceptual question but noted.",
                    courses=sorted(present - {UNKNOWN}))

    return allow("course_provenance", Stage.RETRIEVAL,
                 f"All context from {sorted(present)}.")


RETRIEVAL_GUARDS = [check_retrieval_confidence, check_course_provenance]


def run_retrieval_guards(question, chunks):
    verdicts = []
    for guard in RETRIEVAL_GUARDS:
        v = guard(question, chunks)
        verdicts.append(v)
        if v.blocked:
            break
    return verdicts
