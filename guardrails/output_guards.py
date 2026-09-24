"""Stage 3 -- AI output testing: the model's answer is a candidate, not a result.

Nothing generated is returned to the user until it passes these checks. This
is the stage that catches what the earlier two cannot:

  * Q23 ("what was the average attendance in Agentic AI?") passes both input
    and retrieval guards -- it is topically close to the right course's
    handout and scores 0.553, higher than most answerable questions. The
    knowledge base simply has no attendance figure. Only inspecting the
    generated answer reveals a number that exists nowhere in the context.

  * The Week 4 repo experiment fabricated three specific filenames
    (tests/test_app.py and friends). Specific, checkable, and absent from the
    retrieved context.

The central check is therefore not "does this read as grounded" but "does
every specific, checkable claim in the answer actually appear in what the
model was given". Vague prose passes; invented specifics do not.
"""
import re

from guardrails.base import Stage, allow, block, warn

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Entity kinds that are specific enough to verify literally. Each is a claim a
# reader would act on -- a name to email, a room to walk to, a figure to quote.
ENTITY_PATTERNS = {
    "person": r"\b(?:Dr|Prof|Professor|Mr|Ms|Mrs)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}",
    "room": r"\b(?:Cabin|Room|Office)\s*(?:No\.?)?\s*\d+\b",
    "floor": r"\b\d+(?:st|nd|rd|th)\s+Floor\b",
    "email": r"\b[\w.\-]+@[\w.\-]+\.\w+\b",
    "phone": r"\b\d{10}\b",
    "percentage": r"\b\d{1,3}(?:\.\d+)?\s?%",
    "course_code": r"\b[A-Z]{2,4}\s?\d{3,4}\b",
    "file_path": r"\b[\w/]+\.(?:py|md|json|ya?ml|txt|pdf)\b",
    # Bare figures last, so a number already captured as a percentage, phone,
    # room or course code is not double-counted. Two digits or more: a
    # standalone "78 out of 100" is a checkable claim, "4 conditions" is not.
    "number": r"\b\d{2,}(?:\.\d+)?\b",
    # A capitalised word mid-sentence is usually a name, product or title --
    # i.e. an attributable claim. Sentence-initial words are excluded below,
    # since those are capitalised by grammar rather than by being names.
    "proper_noun": r"(?<![.!?]\s)(?<!^)\b[A-Z][a-zA-Z]{3,}\b",
}

# Capitalised words that are grammar or scaffolding, not claims.
PROPER_NOUN_STOPLIST = {
    "the", "this", "that", "there", "these", "those", "they", "then", "their",
    "yes", "according", "based", "however", "while", "which", "what", "when",
    "where", "course", "syllabus", "handout", "exam", "exams", "paper",
    "papers", "unit", "units", "week", "year", "years", "chapter", "chapters",
    "context", "question", "answer", "topic", "topics", "note", "please",
}

# Grounding on a very short answer is not a measurement: with one sentence the
# rate can only be 0.0 or 1.0, so a correct one-line definition that happens to
# paraphrase the source scores 100% ungrounded. Below this many sentences the
# grounding guard warns instead of blocking, and the entity check -- which does
# not care about length -- carries the load.
MIN_SENTENCES_TO_BLOCK_ON_GROUNDING = 3

REFUSAL_MARKERS = (
    "i couldn't find", "i could not find", "not covered", "does not contain",
    "doesn't contain", "no information", "not mentioned", "not available",
    "cannot answer", "can't answer", "not specified", "unable to",
    "i don't have", "i do not have", "not found",
)

MAX_HALLUCINATION_RATE = 0.60


def _normalise(s):
    return re.sub(r"\s+", " ", (s or "")).lower()


def extract_entities(text):
    """Every specific, checkable claim in the text, as (kind, surface form).

    Patterns are applied in declaration order and overlapping spans are kept
    only once, so "Cabin No. 81" is reported as a room and not also as the
    bare number 81.
    """
    out, taken = [], []
    for kind, pattern in ENTITY_PATTERNS.items():
        for m in re.finditer(pattern, text or ""):
            if any(m.start() < e and m.end() > s_ for s_, e in taken):
                continue
            if kind == "proper_noun" and m.group(0).lower() in PROPER_NOUN_STOPLIST:
                continue
            taken.append((m.start(), m.end()))
            out.append((kind, m.group(0).strip()))
    return out


def check_unsupported_entities(question, answer, context, chunks=None):
    """Every specific entity in the answer must appear in the context.

    This is what would have caught Q22: 'Dr. Soharab Hossain Shaikh' and
    'Cabin No. 81' are real strings from the knowledge base, so a pure
    is-this-invented check passes them -- but once the wrong-course chunks are
    excluded from the context, they are unsupported and the answer is refused.
    """
    ctx = _normalise(context)
    q = _normalise(question)
    unsupported = []
    for kind, surface in extract_entities(answer or ""):
        norm = _normalise(surface)
        # An entity the user themselves supplied is not the model's claim.
        if norm in ctx or norm in q:
            continue
        unsupported.append({"kind": kind, "value": surface})

    if unsupported:
        shown = ", ".join(f"{u['value']!r} ({u['kind']})" for u in unsupported[:4])
        return block("unsupported_entities", Stage.OUTPUT,
                     f"Answer states {len(unsupported)} specific claim(s) absent from "
                     f"the retrieved context: {shown}.",
                     "I couldn't verify that answer against your documents, so I'm "
                     "not going to give it. The details it relied on aren't in the "
                     "material you uploaded.",
                     unsupported=unsupported)
    return allow("unsupported_entities", Stage.OUTPUT,
                 "Every specific claim in the answer appears in the context.")


def check_grounding(question, answer, context, embedder=None, chunks=None):
    """Sentence-level grounding, reusing the Week 4 hallucination metric."""
    if embedder is None or not (context or "").strip():
        return allow("grounding", Stage.OUTPUT, "No context or embedder; skipped.")
    from eval.metrics import hallucination_rate
    rate = hallucination_rate(answer or "", context, embedder)
    if rate is None:
        return allow("grounding", Stage.OUTPUT, "Not computable.")
    n_sentences = len([x for x in _SENTENCE_SPLIT.split(answer or "") if len(x.strip()) > 8])
    if rate > MAX_HALLUCINATION_RATE and n_sentences < MIN_SENTENCES_TO_BLOCK_ON_GROUNDING:
        return warn("grounding", Stage.OUTPUT,
                    f"{rate:.0%} ungrounded, but only {n_sentences} sentence(s) -- too "
                    f"few for the rate to be meaningful, so warning rather than blocking.",
                    hallucination_rate=round(rate, 3), sentences=n_sentences)
    if rate > MAX_HALLUCINATION_RATE:
        return block("grounding", Stage.OUTPUT,
                     f"{rate:.0%} of answer sentences are not grounded in the "
                     f"retrieved context (limit {MAX_HALLUCINATION_RATE:.0%}).",
                     "That answer drifted away from your documents, so I'm not "
                     "going to give it.",
                     hallucination_rate=round(rate, 3))
    if rate > MAX_HALLUCINATION_RATE / 2:
        return warn("grounding", Stage.OUTPUT,
                    f"{rate:.0%} of sentences weakly grounded.",
                    hallucination_rate=round(rate, 3))
    return allow("grounding", Stage.OUTPUT, f"Hallucination rate {rate:.0%}.")


def looks_like_refusal(answer):
    a = _normalise(answer)
    return any(m in a for m in REFUSAL_MARKERS)


def check_expected_refusal(question, answer, context, should_refuse=False, **_):
    """When the context cannot support an answer, the answer must say so."""
    if should_refuse and not looks_like_refusal(answer):
        return block("expected_refusal", Stage.OUTPUT,
                     "Context was insufficient, but the answer asserts a result "
                     "instead of declining.",
                     "I couldn't find enough in your documents to answer that.")
    return allow("expected_refusal", Stage.OUTPUT)


OUTPUT_GUARDS = [check_unsupported_entities, check_grounding, check_expected_refusal]


def run_output_guards(question, answer, context, embedder=None, chunks=None,
                      should_refuse=False):
    verdicts = []
    for guard in OUTPUT_GUARDS:
        kwargs = {"chunks": chunks}
        if guard is check_grounding:
            kwargs["embedder"] = embedder
        if guard is check_expected_refusal:
            kwargs = {"should_refuse": should_refuse}
        v = guard(question, answer, context, **kwargs)
        verdicts.append(v)
        if v.blocked:
            break
    return verdicts
