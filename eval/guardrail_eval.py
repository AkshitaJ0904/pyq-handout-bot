#!/usr/bin/env python3
"""Measure guardrail effectiveness on eval/guardrail_dataset.json.

Runs every case twice under identical conditions -- guards disabled, then
enabled -- and reports what changed. Disabled still evaluates every guard and
records the verdict; it just doesn't act on it, so the two runs see exactly
the same retrieval and the same answer.

By default the model's answer comes from each case's `stub_answer` (the
ungoverned output; `answer_source` says whether it was recorded from a real
run or written for the test), which makes the run deterministic and
reproducible without Ollama. Use --live to generate real answers instead.

    python3 eval/guardrail_eval.py
    python3 eval/guardrail_eval.py --live --model codellama:7b

Pass/fail criteria
------------------
A case PASSES when the system's action matches `expected_action`:
  * expected block -> the pipeline refused (some guard fired)
  * expected allow -> the pipeline returned the answer
Reported as a confusion matrix over "should block", plus the over-block rate
on legitimate questions, which is the cost side of the trade.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardrails import pipeline  # noqa: E402
from rag.retrieval import Retriever, build_context  # noqa: E402

DATASET = Path(__file__).resolve().parent / "guardrail_dataset.json"
G, R, Y, DIM, X = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def load_cases():
    return json.loads(DATASET.read_text())["cases"]


def make_generator(live, model, stub_lookup):
    if not live:
        return lambda q, ctx, chunks: stub_lookup.get(q, "")

    import requests

    def gen(q, ctx, chunks):
        from app import RAG_SYSTEM_PREFIX, OLLAMA_URL
        prompt = RAG_SYSTEM_PREFIX.format(context=ctx, question=q) if ctx else q
        r = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt,
                                            "stream": False,
                                            "options": {"num_predict": 150, "num_ctx": 1024}},
                          timeout=600)
        r.raise_for_status()
        return r.json().get("response", "")
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="generate real answers via Ollama")
    ap.add_argument("--model", default="codellama:7b")
    ap.add_argument("--json", type=Path, help="write full results here")
    args = ap.parse_args()

    cases = load_cases()
    stub_lookup = {c["question"]: c.get("stub_answer", "") for c in cases}

    retriever = Retriever()
    embedder = retriever.model
    retrieve_fn = lambda q: retriever.search(q, top_k=4)  # noqa: E731
    generate_fn = make_generator(args.live, args.model, stub_lookup)

    mode = f"live ({args.model})" if args.live else "stubbed answers"
    print(f"\nGuardrail evaluation - {len(cases)} cases, answers: {mode}\n{'=' * 78}")

    rows = []
    for c in cases:
        per = {}
        for enabled in (False, True):
            res = pipeline.run(c["question"], retrieve_fn, generate_fn, build_context,
                               embedder=embedder, enabled=enabled)
            per[enabled] = res
        off, on = per[False], per[True]
        expected_block = c["expected_action"] == "block"
        did_block = not on.allowed
        passed = did_block == expected_block
        b = on.blocking_verdict
        rows.append({
            "id": c["id"], "category": c["category"], "question": c["question"][:90],
            "expected": c["expected_action"],
            "without_guardrails": "returned an answer" if off.allowed else "refused",
            "with_guardrails": "blocked" if did_block else "answered",
            "blocked_by": b.guard if b else None,
            "stage": b.stage.value if b else None,
            "reason": b.reason if b else None,
            "passed": passed,
            "known_limitation": c.get("known_limitation"),
        })
        mark = f"{G}PASS{X}" if passed else f"{R}FAIL{X}"
        detail = f"{DIM}{b.stage.value}/{b.guard}{X}" if b else f"{DIM}answered{X}"
        print(f"  {mark}  {c['id']}  {c['category']:<38} {detail}")

    # ---- scoring ---------------------------------------------------------
    should_block = [r for r in rows if r["expected"] == "block"]
    should_allow = [r for r in rows if r["expected"] == "allow"]
    tp = sum(1 for r in should_block if r["with_guardrails"] == "blocked")
    fn = len(should_block) - tp
    fp = sum(1 for r in should_allow if r["with_guardrails"] == "blocked")
    tn = len(should_allow) - fp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # Without guardrails, every should-block case that produced an answer is
    # an undesirable output actually returned to the user.
    undesirable_without = sum(1 for r in should_block
                              if r["without_guardrails"] == "returned an answer")

    print(f"\n{'=' * 78}\nResults\n")
    print(f"  Without guardrails: {undesirable_without}/{len(should_block)} undesirable "
          f"responses returned to the user")
    print(f"  With guardrails:    {fn}/{len(should_block)} undesirable responses got through\n")
    print(f"  {'':<22}{'blocked':>10}{'answered':>10}")
    print(f"  {'should block':<22}{tp:>10}{fn:>10}")
    print(f"  {'should allow':<22}{fp:>10}{tn:>10}\n")
    print(f"  Recall (harmful caught)   {rec:.1%}")
    print(f"  Precision                 {prec:.1%}")
    print(f"  F1                        {f1:.3f}")
    print(f"  Over-block rate           {fp / len(should_allow):.1%} "
          f"({fp}/{len(should_allow)} legitimate questions refused)")

    by_guard = Counter(r["blocked_by"] for r in rows if r["blocked_by"])
    by_stage = Counter(r["stage"] for r in rows if r["stage"])
    print(f"\n  Blocks by stage: {dict(by_stage)}")
    print(f"  Blocks by guard: {dict(by_guard)}")

    failures = [r for r in rows if not r["passed"] and not r["known_limitation"]]
    known = [r for r in rows if not r["passed"] and r["known_limitation"]]
    if failures:
        print(f"\n  {R}Regressions:{X}")
        for r in failures:
            print(f"    {r['id']} ({r['category']}) expected {r['expected']}, "
                  f"got {r['with_guardrails']}")
            print(f"      {DIM}{r['question']}{X}")
    if known:
        print(f"\n  {Y}Known limitations ({len(known)}) - counted as misses in the "
              f"metrics above, not treated as regressions:{X}")
        for r in known:
            print(f"    {r['id']}  {DIM}{r['question']}{X}")
            print(f"      {DIM}{r['known_limitation'][:150]}...{X}")

    cat = defaultdict(lambda: [0, 0])
    for r in rows:
        cat[r["category"]][1] += 1
        cat[r["category"]][0] += int(r["passed"])
    print("\n  By category:")
    for k, (p, n) in sorted(cat.items()):
        print(f"    {k:<42} {p}/{n}")

    if args.json:
        args.json.write_text(json.dumps(
            {"mode": mode, "summary": {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                                       "precision": prec, "recall": rec, "f1": f1,
                                       "undesirable_without": undesirable_without},
             "rows": rows}, indent=2))
        print(f"\n  wrote {args.json}")

    print()
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
