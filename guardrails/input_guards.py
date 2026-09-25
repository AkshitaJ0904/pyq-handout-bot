"""Stage 1 -- checks on the user's input, before any retrieval or generation.

These are cheap and run first: blocking here costs nothing, whereas letting a
bad input through costs a full generation (10s and ~4.5GB of RAM on this
hardware, per eval/REPORT.md).
"""
import re

from guardrails.base import Stage, allow, block

MAX_QUESTION_CHARS = 2000
MIN_QUESTION_CHARS = 8

# Attempts to talk to the system rather than ask a question of the documents.
# The prompt template puts retrieved context and the user's text in the same
# string, so text that impersonates an instruction is a real risk.
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(the\s+)?(previous|prior|above|system)",
    r"(reveal|show|print|repeat)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(if\s+)?(you|an?)\b",
    r"forget\s+(everything|all|your)\s",
]

# A study assistant should decline these outright rather than answer from
# course documents. Deliberately narrow -- a broad denylist would block
# legitimate questions, and over-blocking is measured in the eval.
OUT_OF_BOUNDS_PATTERNS = [
    r"\b(how to (make|build|create)\s+(a\s+)?(bomb|explosive|weapon|gun))",
    r"\b(kill|harm|hurt)\s+(myself|yourself|someone|people)\b",
    r"\bsuicide\b",
    r"\b(hack|crack|bypass)\s+(into\s+)?(the\s+)?(exam|system|server|account|password)",
    r"\b(credit card|social security|aadhaar)\s+number\b",
    r"\bwrite\s+(my|the)\s+(exam|test)\s+for\s+me\b",
]


def _matches(patterns, text):
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return m.group(0)
    return None


def check_not_empty(question: str):
    if not question or not question.strip():
        return block("empty_input", Stage.INPUT,
                     "Question is empty or whitespace only.",
                     "Please type a question.")
    if len(question.strip()) < MIN_QUESTION_CHARS:
        return block("empty_input", Stage.INPUT,
                     f"Question is {len(question.strip())} chars, below the "
                     f"{MIN_QUESTION_CHARS}-char minimum.",
                     "That's too short to answer — please ask a fuller question.",
                     length=len(question.strip()))
    return allow("empty_input", Stage.INPUT)


def check_length(question: str):
    n = len(question or "")
    if n > MAX_QUESTION_CHARS:
        return block("input_length", Stage.INPUT,
                     f"Question is {n} chars, over the {MAX_QUESTION_CHARS} limit.",
                     f"That question is too long ({n:,} characters). "
                     f"Please keep it under {MAX_QUESTION_CHARS:,}.",
                     length=n, limit=MAX_QUESTION_CHARS)
    return allow("input_length", Stage.INPUT)


def check_injection(question: str):
    hit = _matches(INJECTION_PATTERNS, question or "")
    if hit:
        return block("prompt_injection", Stage.INPUT,
                     f"Input contains an instruction-style pattern: {hit!r}.",
                     "I can only answer questions about the documents you've "
                     "uploaded, not change how I work.",
                     matched=hit)
    return allow("prompt_injection", Stage.INPUT)


def check_in_bounds(question: str):
    hit = _matches(OUT_OF_BOUNDS_PATTERNS, question or "")
    if hit:
        return block("out_of_bounds", Stage.INPUT,
                     f"Input matches an out-of-bounds pattern: {hit!r}.",
                     "That's outside what this study assistant will answer.",
                     matched=hit)
    return allow("out_of_bounds", Stage.INPUT)


INPUT_GUARDS = [check_not_empty, check_length, check_injection, check_in_bounds]


def run_input_guards(question: str):
    verdicts = []
    for guard in INPUT_GUARDS:
        v = guard(question)
        verdicts.append(v)
        if v.blocked:
            break
    return verdicts
