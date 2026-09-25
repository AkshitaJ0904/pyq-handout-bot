"""Composes the three guardrail stages around a generation call.

    input guards -> retrieve -> retrieval guards -> generate -> output guards

Blocking early is the cheap path: an input rejected before generation costs
nothing, while a full generation costs ~10s and ~4.5GB of RAM on this
hardware (eval/REPORT.md). So the stages run in cost order, and the pipeline
stops at the first BLOCK.

`generate_fn` and `retrieve_fn` are injected rather than imported, so the
pipeline can be evaluated deterministically without Ollama running -- which
is what eval/guardrail_eval.py relies on.
"""
from guardrails.base import Action, PipelineResult
from guardrails.input_guards import run_input_guards
from guardrails.output_guards import run_output_guards
from guardrails.retrieval_guards import run_retrieval_guards


def _blocked(verdicts):
    return next((v for v in verdicts if v.blocked), None)


ALL_STAGES = ("input", "retrieval", "output")


def run(question, retrieve_fn, generate_fn, build_context_fn, embedder=None,
        enabled=True, stages=ALL_STAGES):
    """Answer `question`, refusing rather than guessing when a guard fires.

    With `enabled=False` every guard is still evaluated and reported, but
    nothing is blocked -- that is how the with/without comparison is measured
    on identical inputs in eval/guardrail_eval.py.

    `stages` narrows which stages apply. The deliberate no-knowledge-base mode
    (the "Compare with / without" tab) runs input guards only: there is no
    retrieval to judge, and an answer generated with no context has nothing to
    be checked against, so retrieval and output guards would reject every
    response rather than measure anything. Input guards still apply, because
    injection and out-of-bounds requests are wrong regardless of retrieval.
    """
    verdicts = []

    def stop(result_verdicts):
        """Honour a block only when guards are enabled."""
        v = _blocked(result_verdicts)
        return v if (v and enabled) else None

    # -- stage 1: input -------------------------------------------------
    if "input" in stages:
        v_in = run_input_guards(question)
        verdicts += v_in
        if (b := stop(v_in)):
            return PipelineResult(allowed=False, answer=b.message, verdicts=verdicts)

    # -- retrieve --------------------------------------------------------
    chunks = retrieve_fn(question) or []

    # -- stage 2: retrieval ---------------------------------------------
    should_refuse = False
    if "retrieval" in stages:
        v_ret = run_retrieval_guards(question, chunks)
        verdicts += v_ret
        if (b := stop(v_ret)):
            return PipelineResult(allowed=False, answer=b.message,
                                  verdicts=verdicts, chunks=chunks)
        # A retrieval block that was *suppressed* still tells the output stage
        # that the answer ought to be a refusal.
        should_refuse = _blocked(v_ret) is not None

    # -- generate --------------------------------------------------------
    context = build_context_fn(chunks) if chunks else ""
    answer = generate_fn(question, context, chunks)

    # -- stage 3: output -------------------------------------------------
    if "output" in stages:
        v_out = run_output_guards(question, answer, context, embedder=embedder,
                                  chunks=chunks, should_refuse=should_refuse)
        verdicts += v_out
        if (b := stop(v_out)):
            return PipelineResult(allowed=False, answer=b.message,
                                  verdicts=verdicts, chunks=chunks)

    return PipelineResult(allowed=True, answer=answer, verdicts=verdicts,
                          chunks=chunks)


def summarise(result):
    """One-line description of what happened, for logs and the UI.

    A blocking verdict recorded while guards were disabled is reported as
    "would have blocked" -- it was evaluated but not acted on."""
    b = result.blocking_verdict
    if b and not result.allowed:
        return f"blocked at {b.stage.value} by {b.guard}: {b.reason}"
    if b:
        return (f"answered; {b.stage.value}/{b.guard} would have blocked it "
                f"({b.reason})")
    warns = [v for v in result.verdicts if v.action is Action.WARN]
    return f"allowed ({len(warns)} warning(s))" if warns else "allowed"
