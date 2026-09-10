"""
Week 4, Exercise 3 - metric implementations.

Every metric function documents exactly HOW it is calculated, per the
assignment's requirement to define the method rather than just report a
number. These are automated proxies, not a substitute for human judgment --
REPORT.md discusses where each proxy is reliable and where it isn't.
"""
import ast
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil

# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def keypoint_coverage(answer: str, expected_keypoints: list) -> float:
    """Correctness/Accuracy proxy.

    Fraction of `expected_keypoints` (case-insensitive substrings, each
    hand-picked from the source document as a fact the answer should
    contain) that appear anywhere in the model's answer text. 1.0 means
    every expected fact was mentioned; 0.0 means none were. This is a
    lexical recall proxy for correctness -- it can't catch a fluent but
    factually wrong answer that happens to repeat the right keywords, so
    REPORT.md cross-checks a sample manually.
    """
    if not expected_keypoints:
        return None
    answer_lower = answer.lower()
    hits = sum(1 for kp in expected_keypoints if kp.lower() in answer_lower)
    return hits / len(expected_keypoints)


def semantic_relevance(answer: str, question: str, embedder) -> float:
    """Relevance metric.

    Cosine similarity between the sentence-transformer embedding of the
    question and of the answer (same all-MiniLM-L6-v2 model the app
    already uses for retrieval, so no new dependency). Ranges [-1, 1];
    in practice on-topic answers score 0.3-0.7 given the asymmetry
    between a short question and a longer answer. Used as a relative
    signal across models on the same question, not an absolute bar.
    """
    vecs = embedder.encode([question, answer], convert_to_numpy=True)
    a, b = vecs[0], vecs[1]
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieval_precision_at_k(retrieved_source_files: list, expected_source_files: list) -> float:
    """Retrieval Quality metric.

    Precision@k: of the k chunks the retriever returned, what fraction
    came from a source file a human labeled as relevant to the question
    (`expected_source_files` in eval_dataset.json)? Only meaningful for
    questions where `expected_source_files` is non-empty (hallucination
    probes deliberately have no relevant source, so they're excluded and
    scored separately via `no_result_leak_rate`).
    """
    if not expected_source_files:
        return None
    if not retrieved_source_files:
        return 0.0
    hits = sum(1 for f in retrieved_source_files if f in expected_source_files)
    return hits / len(retrieved_source_files)


def no_result_leak_rate(retrieved_source_files: list, expected_source_files: list) -> float:
    """Companion to precision@k for hallucination-probe questions (where
    expected_source_files == []): fraction of retrieved chunks that come
    from ANY source file, i.e. how much irrelevant context got shoved
    into the prompt for a question the KB has no real answer to. Always
    1.0 with top_k>0 in this app since the retriever has no similarity
    threshold and always returns exactly top_k chunks -- that's the
    point of measuring it (see REPORT.md Exercise 5).
    """
    if expected_source_files:
        return None
    if not retrieved_source_files:
        return 0.0
    return 1.0


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def hallucination_rate(answer: str, context: str, embedder, threshold: float = 0.35) -> float:
    """Hallucination Rate proxy.

    Splits the answer into sentences and, for each one, computes the max
    cosine similarity (same embedder as retrieval) against every chunk
    in the retrieved context. A sentence whose best match falls below
    `threshold` is counted as "ungrounded" -- not clearly supported by
    anything the model was actually given. hallucination_rate is the
    fraction of ungrounded sentences. This is a heuristic (paraphrase or
    inference can score low even when correct, and a wrong sentence can
    coincidentally share vocabulary with the context) -- REPORT.md
    Exercise 5 manually verifies flagged examples rather than trusting
    the number in isolation.
    """
    if not context.strip():
        return None
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer) if len(s.strip()) > 8]
    if not sentences:
        return 0.0
    context_chunks = [c.strip() for c in context.split("\n\n") if c.strip()]
    if not context_chunks:
        return None
    sent_vecs = embedder.encode(sentences, convert_to_numpy=True)
    ctx_vecs = embedder.encode(context_chunks, convert_to_numpy=True)
    ctx_norms = np.linalg.norm(ctx_vecs, axis=1)
    ctx_norms[ctx_norms == 0] = 1e-8
    ungrounded = 0
    for sv in sent_vecs:
        sv_norm = np.linalg.norm(sv)
        if sv_norm == 0:
            continue
        sims = (ctx_vecs @ sv) / (ctx_norms * sv_norm)
        if float(np.max(sims)) < threshold:
            ungrounded += 1
    return ungrounded / len(sentences)


def extract_python_function(text: str, function_name: str):
    """Pulls the first ```...``` fenced code block (or, failing that, the
    whole response) out of a model's answer and returns it only if it
    parses as valid Python AND defines `function_name`. Returns None
    otherwise so the caller can record a clean test failure instead of
    crashing on unparseable model output.
    """
    fence_match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    candidate = fence_match.group(1) if fence_match else text
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return None
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if function_name not in defined:
        return None
    return candidate


def run_code_test(answer: str, code_test: dict, timeout: int = 10) -> dict:
    """Test-Pass Rate metric for code-generation questions.

    Extracts the model's Python function, appends the fixed assertion
    harness from eval_dataset.json's `code_test.harness`, and runs it in
    a fresh subprocess (isolation from the eval harness itself) with a
    timeout. Returns {"ran": bool, "passed": bool, "error": str|None}.
    "ran" is False when no valid function could even be extracted
    (syntax error / wrong function name) -- that's a distinct failure
    mode from "ran but the assertion failed", and both count as
    test-fail for the pass-rate metric but are reported separately for
    diagnosis.
    """
    if code_test is None:
        return {"ran": None, "passed": None, "error": "no automated test defined for this question"}

    code = extract_python_function(answer, code_test["function_name"])
    if code is None:
        return {"ran": False, "passed": False, "error": "could not extract a valid Python function from the answer"}

    script = code + "\n\n" + code_test["harness"] + "\nprint('__TEST_OK__')\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
        )
        passed = proc.returncode == 0 and "__TEST_OK__" in proc.stdout
        error = None if passed else (proc.stderr.strip()[-500:] or proc.stdout.strip()[-500:])
        return {"ran": True, "passed": passed, "error": error}
    except subprocess.TimeoutExpired:
        return {"ran": True, "passed": False, "error": f"timed out after {timeout}s"}
    finally:
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def ollama_stats(response_json: dict) -> dict:
    """Latency + token usage, read straight from Ollama's own response
    fields (no separate tokenizer needed):
      - total_duration_s: wall-clock time Ollama reports for the whole
        request (load + prompt eval + generation), in seconds.
      - prompt_tokens / response_tokens: prompt_eval_count / eval_count.
      - tokens_per_sec: eval_count / (eval_duration in seconds) --
        generation throughput, excludes prompt processing time.
    """
    total_duration_s = response_json.get("total_duration", 0) / 1e9
    eval_duration_s = response_json.get("eval_duration", 0) / 1e9
    prompt_tokens = response_json.get("prompt_eval_count", 0)
    response_tokens = response_json.get("eval_count", 0)
    tokens_per_sec = (response_tokens / eval_duration_s) if eval_duration_s > 0 else None
    return {
        "total_duration_s": round(total_duration_s, 3),
        "eval_duration_s": round(eval_duration_s, 3),
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec else None,
    }


class ResourceSampler:
    """Samples system-wide CPU% and the `ollama` process's RSS memory
    once per `interval` seconds on a background thread while a generate
    call is in flight, via psutil. Reports peak RSS (MB) and mean CPU%
    observed during the call.

    Limitation (documented, not hidden): on Apple Silicon there is no
    portable, no-sudo way to read discrete GPU utilization -- psutil
    doesn't expose it and `powermetrics` requires root. So "GPU
    consumption" for this run is reported as "not measured (unified
    memory / integrated GPU, no accessible counter)" rather than a made
    up number; RSS memory and CPU% stand in as the available proxies.
    """

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.cpu_samples = []
        self.rss_samples = []

    def _ollama_processes(self):
        # Ollama's own supervisor process is small; the actual weights are
        # loaded into a separate "llama-server" runner process it spawns
        # per model, which is where nearly all CPU/RAM usage actually goes.
        return [p for p in psutil.process_iter(["name", "memory_info"])
                if p.info["name"] and ("ollama" in p.info["name"].lower()
                                        or "llama-server" in p.info["name"].lower())]

    def _run(self):
        while not self._stop.is_set():
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            procs = self._ollama_processes()
            rss = sum(p.info["memory_info"].rss for p in procs if p.info["memory_info"]) / (1024 * 1024)
            self.rss_samples.append(rss)
            time.sleep(self.interval)

    def __enter__(self):
        psutil.cpu_percent(interval=None)  # prime the counter
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> dict:
        return {
            "peak_ollama_rss_mb": round(max(self.rss_samples), 1) if self.rss_samples else None,
            "mean_cpu_percent": round(sum(self.cpu_samples) / len(self.cpu_samples), 1) if self.cpu_samples else None,
            "gpu_percent": "not measured (Apple Silicon unified memory, no no-sudo counter available)",
            "samples": len(self.cpu_samples),
        }
