(function () {
  "use strict";

  const results = document.getElementById("results");
  const kbPill = document.getElementById("kb-pill");

  // ---------- helpers ----------

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str == null ? "" : String(str);
    return div.innerHTML;
  }

  async function fetchJson(url, options) {
    const resp = await fetch(url, options);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || data.error) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    return data;
  }

  // ---------- knowledge base status ----------

  async function refreshKbStatus() {
    try {
      const data = await fetchJson("/health");
      if (typeof data.chunk_count === "number") {
        kbPill.textContent = data.chunk_count > 0
          ? `${data.chunk_count} passages indexed`
          : "no documents yet";
      } else {
        kbPill.textContent = "online";
      }
    } catch (err) {
      kbPill.textContent = "server unreachable";
    }
  }

  // ---------- file lists ----------

  async function refreshFileList() {
    try {
      const data = await fetchJson("/files");
      renderFileList("list-syllabus", data.syllabus || []);
      renderFileList("list-pyq", data.pyq || []);
    } catch (err) {
      /* non-fatal */
    }
  }

  function renderFileList(id, files) {
    const el = document.getElementById(id);
    if (!files.length) {
      el.innerHTML = "";
      return;
    }
    el.innerHTML = files.map((f) => `<li>${escapeHtml(f)}</li>`).join("");
  }

  // ---------- upload ----------

  function setUploadStatus(message, isError) {
    const el = document.getElementById("upload-status");
    el.textContent = message;
    el.classList.toggle("is-error", !!isError);
  }

  async function uploadFiles(docType, fileList, year) {
    if (!fileList || !fileList.length) return;
    const form = new FormData();
    form.append("doc_type", docType);
    if (year) form.append("year", year);
    for (const f of fileList) form.append("file", f);

    setUploadStatus(`Uploading ${fileList.length} file${fileList.length > 1 ? "s" : ""} and re-indexing…`);
    try {
      const data = await fetchJson("/upload", { method: "POST", body: form });
      setUploadStatus(`Indexed. Knowledge base now has ${data.chunk_count} passages.`);
      await refreshFileList();
      await refreshKbStatus();
    } catch (err) {
      setUploadStatus(err.message, true);
    }
  }

  function wireUploadZone(zoneId, inputId, docType, yearInputId) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);

    zone.addEventListener("click", () => input.click());
    input.addEventListener("change", () => {
      const year = yearInputId ? document.getElementById(yearInputId).value.trim() : "";
      uploadFiles(docType, input.files, year);
      input.value = "";
    });

    ["dragenter", "dragover"].forEach((evt) =>
      zone.addEventListener(evt, (e) => {
        e.preventDefault();
        zone.classList.add("is-dragover");
      })
    );
    ["dragleave", "dragend", "drop"].forEach((evt) =>
      zone.addEventListener(evt, (e) => {
        e.preventDefault();
        zone.classList.remove("is-dragover");
      })
    );
    zone.addEventListener("drop", (e) => {
      const files = e.dataTransfer && e.dataTransfer.files;
      if (!files || !files.length) return;
      const year = yearInputId ? document.getElementById(yearInputId).value.trim() : "";
      uploadFiles(docType, files, year);
    });
  }

  wireUploadZone("zone-syllabus", "file-syllabus", "syllabus", null);
  wireUploadZone("zone-pyq", "file-pyq", "pyq", "pyq-year");

  // ---------- ask / compare ----------

  const form = document.getElementById("ask-form");
  const textarea = document.getElementById("question");
  const submitBtn = document.getElementById("submit-btn");
  const submitLabel = document.getElementById("submit-label");
  const tabs = document.querySelectorAll(".seg-btn");
  const modelRow = document.getElementById("model-row");

  let mode = "ask";

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => {
        t.classList.remove("is-active");
        t.setAttribute("aria-selected", "false");
      });
      tab.classList.add("is-active");
      tab.setAttribute("aria-selected", "true");
      mode = tab.dataset.mode;
      submitLabel.textContent =
        mode === "compare" ? "Compare" : mode === "models" ? "Compare models" : mode === "repo" ? "Search repo" : "Ask";
      modelRow.hidden = mode !== "models";
      const isRepo = mode === "repo";
      const repoQuick = document.getElementById("repo-quick");
      const docQuick = document.getElementById("doc-quick");
      const evalPicker = document.querySelector(".eval-picker");
      if (repoQuick) repoQuick.hidden = !isRepo;
      if (docQuick) docQuick.hidden = isRepo;
      // The labelled-question dropdown is for the course-document dataset,
      // which has no bearing on repository questions.
      if (evalPicker) evalPicker.hidden = isRepo;
      textarea.placeholder = isRepo
        ? "ask about this repository — e.g. are there any test files?"
        : "e.g. what topics come up a lot in past papers but aren't in the handout?";
    });
  });

  document.querySelectorAll(".quick").forEach((chip) => {
    chip.addEventListener("click", () => {
      textarea.value = chip.dataset.q;
      textarea.focus();
    });
  });

  const evalSelect = document.getElementById("eval-select");
  const evalHint = document.getElementById("eval-hint");

  async function loadEvalQuestions() {
    if (!evalSelect) return;
    try {
      const data = await fetchJson("/eval_questions");
      (data.questions || []).forEach((q) => {
        const opt = document.createElement("option");
        opt.value = q.id;
        const short = q.question.length > 78 ? q.question.slice(0, 75) + "\u2026" : q.question;
        opt.textContent = `${q.id} \u00b7 ${q.category} \u2014 ${short}`;
        opt.dataset.question = q.question;
        evalSelect.appendChild(opt);
      });
    } catch (err) {
      /* dropdown just stays empty; free-text questions still work */
    }
  }

  if (evalSelect) {
    evalSelect.addEventListener("change", () => {
      const opt = evalSelect.selectedOptions[0];
      if (evalSelect.value && opt && opt.dataset.question) {
        textarea.value = opt.dataset.question;
        evalHint.textContent =
          "Labelled question \u2014 ground truth available, so all eight metrics are measured.";
      } else {
        evalHint.textContent =
          "Correctness, retrieval quality and test-pass need ground truth. Pick one of the 28 labelled questions to measure all eight.";
      }
    });
  }

  function tagRowHtml(sources) {
    if (!sources || !sources.length) return "";
    const tags = sources
      .map(
        (s) =>
          `<span class="tag ${escapeHtml(s.doc_type)}">${escapeHtml(s.doc_type)} &middot; ${escapeHtml(s.source_file)}${
            s.year ? " &middot; " + escapeHtml(s.year) : ""
          }</span>`
      )
      .join("");
    return `<div class="tags-label">Sources</div><div class="tag-row">${tags}</div>`;
  }

  function coverageHtml(sources) {
    if (!sources || !sources.length) return "";
    const pyq = sources.filter((s) => s.doc_type === "pyq").length;
    const syllabus = sources.filter((s) => s.doc_type === "syllabus").length;
    const total = pyq + syllabus || 1;
    const pyqPct = Math.round((pyq / total) * 100);
    return `
      <div class="coverage">
        <div class="coverage-caption">exam pressure vs. handout depth &mdash; ${pyq} PYQ / ${syllabus} syllabus passage${syllabus === 1 && pyq === 1 ? "" : "s"} matched</div>
        <div class="coverage-bar">
          <div class="seg-pyq" style="width:${pyqPct}%"></div>
          <div class="seg-syllabus" style="width:${100 - pyqPct}%"></div>
        </div>
      </div>`;
  }

  function metricsHtml(metrics, opts) {
    if (!metrics || !metrics.length) return "";
    const compact = (opts && opts.compact) || false;
    const groups = [
      { key: "quality", label: "Quality" },
      { key: "performance", label: "Performance" },
    ];
    const blocks = groups
      .map((g) => {
        const tiles = metrics
          .filter((m) => m.group === g.key)
          .map((m) => {
            const cls = m.available ? "metric is-available" : "metric is-na";
            const tip = m.available ? m.detail || "" : m.reason || "";
            return `<div class="${cls}" title="${escapeHtml(tip)}">
              <div class="metric-label">${escapeHtml(m.label)}</div>
              <div class="metric-value">${escapeHtml(String(m.display))}</div>
              ${compact ? "" : `<div class="metric-detail">${escapeHtml(tip)}</div>`}
            </div>`;
          })
          .join("");
        if (!tiles) return "";
        return `<div class="metric-group">
          <div class="metric-group-label">${g.label}</div>
          <div class="metric-grid">${tiles}</div>
        </div>`;
      })
      .join("");
    const naCount = metrics.filter((m) => !m.available).length;
    const foot = naCount
      ? `<div class="metric-foot">${metrics.length - naCount} of ${metrics.length} measurable for this question &mdash; hover a greyed metric for why.</div>`
      : `<div class="metric-foot">All ${metrics.length} metrics measurable &mdash; this is a labelled question.</div>`;
    return `<div class="metrics-panel">${blocks}${foot}</div>`;
  }

  // ---------- metrics comparison chart ----------
  // One parameter at a time, vertical bars, paged with prev/next arrows —
  // cleaner than 8 rows stacked at once.

  const CHART_METRIC_KEYS = [
    "correctness", "relevance", "retrieval_quality", "hallucination", "test_pass",
    "latency", "tokens", "resources",
  ];
  const CHART_RATIO_KEYS = new Set(["correctness", "relevance", "retrieval_quality", "hallucination", "test_pass"]);
  const CHART_COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)"];

  function metricsChartHtml(items) {
    // items: [{label, metrics}] — metrics is the array each route already
    // returns per answer (see rag/live_metrics.py). Needs 2+ items with at
    // least one usable metric between them, otherwise there is nothing to
    // compare and the chart is skipped rather than shown empty.
    const usable = (items || []).filter((it) => it.metrics && it.metrics.length);
    if (usable.length < 2) return "";

    const byKey = usable.map((it) => {
      const m = {};
      it.metrics.forEach((x) => { m[x.key] = x; });
      return m;
    });

    const slides = CHART_METRIC_KEYS.map((key) => {
      const cells = byKey.map((m) => m[key]);
      const sample = cells.find((c) => c);
      if (!sample || !cells.some((c) => c && c.available)) return null;

      const isRatio = CHART_RATIO_KEYS.has(key);
      const maxVal = isRatio
        ? 1
        : Math.max(1e-9, ...cells.filter((c) => c && c.available && typeof c.value === "number").map((c) => c.value));

      const bars = cells
        .map((c, i) => {
          const color = CHART_COLORS[i % CHART_COLORS.length];
          const label = escapeHtml(usable[i].label);
          if (!c || !c.available || typeof c.value !== "number") {
            return `<div class="chart-vbar">
              <span class="chart-vvalue is-na">N/A</span>
              <div class="chart-vtrack"></div>
              <span class="chart-vlabel">${label}</span>
            </div>`;
          }
          const pct = Math.max(2, Math.min(100, (c.value / maxVal) * 100));
          return `<div class="chart-vbar">
            <span class="chart-vvalue">${escapeHtml(String(c.display))}</span>
            <div class="chart-vtrack"><div class="chart-vfill" style="height:${pct}%;background:${color}"></div></div>
            <span class="chart-vlabel">${label}</span>
          </div>`;
        })
        .join("");

      return { label: sample.label, html: `<div class="chart-vbars">${bars}</div>` };
    }).filter(Boolean);

    if (!slides.length) return "";

    const slidesHtml = slides
      .map((s, i) => `<div class="chart-slide" data-label="${escapeHtml(s.label)}"${i > 0 ? " hidden" : ""}>${s.html}</div>`)
      .join("");
    const dotsHtml = slides
      .map((_, i) => `<span class="chart-dot${i === 0 ? " is-active" : ""}"></span>`)
      .join("");

    return `<div class="metrics-chart">
      <div class="chart-title">Metrics comparison</div>
      <div class="chart-pager">
        <button type="button" class="chart-nav prev" aria-label="Previous parameter">&larr;</button>
        <span class="chart-param-name">${escapeHtml(slides[0].label)}</span>
        <button type="button" class="chart-nav next" aria-label="Next parameter">&rarr;</button>
      </div>
      <div class="chart-slides">${slidesHtml}</div>
      <div class="chart-dots">${dotsHtml}</div>
    </div>`;
  }

  // Delegated once for the whole results feed, since chart cards are added
  // dynamically and each keeps its own paging state via which slide/dot
  // currently lacks [hidden] / has .is-active — no per-chart id needed.
  results.addEventListener("click", (e) => {
    const nav = e.target.closest(".chart-nav");
    const dot = e.target.closest(".chart-dot");
    if (!nav && !dot) return;

    const chart = e.target.closest(".metrics-chart");
    if (!chart) return;
    const slides = Array.from(chart.querySelectorAll(".chart-slide"));
    const dots = Array.from(chart.querySelectorAll(".chart-dot"));
    const current = slides.findIndex((s) => !s.hidden);
    let next;
    if (dot) {
      next = dots.indexOf(dot);
    } else {
      next = nav.classList.contains("prev")
        ? (current - 1 + slides.length) % slides.length
        : (current + 1) % slides.length;
    }
    slides[current].hidden = true;
    dots[current].classList.remove("is-active");
    slides[next].hidden = false;
    dots[next].classList.add("is-active");
    chart.querySelector(".chart-param-name").textContent = slides[next].dataset.label;
  });

  function renderLoading(question) {
    results.insertAdjacentHTML(
      "afterbegin",
      `<div class="result-card" id="pending-card">
        <div class="result-question">${escapeHtml(question)}</div>
        <div class="loading-row"><span class="spinner"></span> thinking&hellip;</div>
      </div>`
    );
  }

  function clearLoading() {
    const el = document.getElementById("pending-card");
    if (el) el.remove();
  }

  function renderError(question, message) {
    clearLoading();
    results.insertAdjacentHTML(
      "afterbegin",
      `<div class="result-card error-card">
        <div class="result-question">${escapeHtml(question)}</div>
        <div><span class="error-tag">Error &mdash;</span> ${escapeHtml(message)}</div>
      </div>`
    );
  }

  function renderAskResult(data) {
    clearLoading();
    results.insertAdjacentHTML(
      "afterbegin",
      `<div class="result-card">
        <div class="result-question">${escapeHtml(data.question)}</div>
        <div class="answer"><p>${escapeHtml(data.answer).trim()}</p></div>
        ${tagRowHtml(data.sources)}
        ${coverageHtml(data.sources)}
        ${metricsHtml(data.metrics)}
      </div>`
    );
  }

  function renderCompareResult(data) {
    clearLoading();
    results.insertAdjacentHTML(
      "afterbegin",
      `<div class="result-card">
        <div class="result-question">${escapeHtml(data.question)}</div>
        <div class="compare-grid">
          <div class="compare-col">
            <div class="col-label"><span class="dot off"></span> Without your documents</div>
            <div class="answer"><p>${escapeHtml(data.without_rag).trim()}</p></div>
            ${metricsHtml(data.metrics_without_rag, { compact: true })}
          </div>
          <div class="compare-col">
            <div class="col-label"><span class="dot on"></span> With your documents</div>
            <div class="answer"><p>${escapeHtml(data.with_rag).trim()}</p></div>
            ${metricsHtml(data.metrics_with_rag, { compact: true })}
          </div>
        </div>
        ${metricsChartHtml([
          { label: "Without your documents", metrics: data.metrics_without_rag },
          { label: "With your documents", metrics: data.metrics_with_rag },
        ])}
        ${tagRowHtml(data.sources_used_for_rag)}
        ${coverageHtml(data.sources_used_for_rag)}
      </div>`
    );
  }

  function renderModelsResult(data) {
    clearLoading();
    const cols = data.results
      .map((r) => {
        const body = r.error
          ? `<div class="answer"><p class="error-tag">Error &mdash; ${escapeHtml(r.error)}</p></div>`
          : `<div class="answer"><p>${escapeHtml(r.answer).trim()}</p></div>`;
        return `<div class="compare-col">
          <div class="col-label"><span class="dot on"></span> ${escapeHtml(r.model)} &middot; ${r.latency_s}s</div>
          ${body}
          ${metricsHtml(r.metrics, { compact: true })}
        </div>`;
      })
      .join("");
    const wide = data.results.length >= 3 ? " is-wide" : "";
    results.insertAdjacentHTML(
      "afterbegin",
      `<div class="result-card${wide}">
        <div class="result-question">${escapeHtml(data.question)}</div>
        <div class="compare-grid models-grid">${cols}</div>
        ${metricsChartHtml(data.results.filter((r) => !r.error).map((r) => ({ label: r.model, metrics: r.metrics })))}
        ${tagRowHtml(data.sources)}
        ${coverageHtml(data.sources)}
      </div>`
    );
  }

  function structuralHtml(structural) {
    return structural
      .map((r) => {
        const rows = r.matches.length
          ? r.matches
              .map(
                (m) =>
                  `<li><code>${escapeHtml(m.file)}${m.line ? ":" + m.line : ""}</code>${
                    m.symbol ? " <span class=\"sym\">" + escapeHtml(m.symbol) + "</span>" : ""
                  }</li>`
              )
              .join("")
          : `<li class="no-match">no matches &mdash; the answer is that it does not exist</li>`;
        return `<div class="sq">
          <div class="sq-head"><code class="sq-q">${escapeHtml(r.kind)}:${escapeHtml(r.term || "*")}</code>
            <span class="sq-count">${r.count} match${r.count === 1 ? "" : "es"}</span></div>
          <div class="sq-why">${escapeHtml(r.rationale || "")}</div>
          <ul class="sq-list">${rows}</ul>
        </div>`;
      })
      .join("");
  }

  function renderRepoResult(data) {
    clearLoading();
    const chunks = (data.rag_chunks || [])
      .map((c) => `<li><code>${escapeHtml(c.file)}</code> <span class="sym">${c.score}</span></li>`)
      .join("") || `<li class="no-match">nothing retrieved</li>`;
    const ragAns = data.rag_answer
      ? `<div class="answer"><p>${escapeHtml(data.rag_answer).trim()}</p></div>`
      : "";
    const structAns = data.structural_answer
      ? `<div class="answer"><p>${escapeHtml(data.structural_answer).trim()}</p></div>`
      : "";
    const note = data.answer_error
      ? `<div class="repo-note">${escapeHtml(data.answer_error)}</div>`
      : "";
    const badge = data.sourcegraph_configured
      ? `<span class="badge on">Sourcegraph</span>`
      : `<span class="badge off">local AST fallback &mdash; no Sourcegraph instance configured</span>`;

    results.insertAdjacentHTML(
      "afterbegin",
      `<div class="result-card is-wide">
        <div class="result-question">${escapeHtml(data.question)}</div>
        ${note}
        <div class="compare-grid">
          <div class="compare-col">
            <div class="col-label"><span class="dot off"></span> Chunk similarity &mdash; Week 3 RAG on the repo</div>
            <div class="sq-sub">retrieved by embedding similarity, with no model of repository structure</div>
            <ul class="sq-list">${chunks}</ul>
            ${ragAns}
          </div>
          <div class="compare-col">
            <div class="col-label"><span class="dot on"></span> Structural code search ${badge}</div>
            <div class="sq-sub">the question is translated into a code query, then run against the repository's structure</div>
            ${structuralHtml(data.structural || [])}
            ${structAns}
          </div>
        </div>
      </div>`
    );
  }

  async function submitQuestion(question) {
    renderLoading(question);
    submitBtn.disabled = true;
    const prevLabel = submitLabel.textContent;
    submitLabel.textContent =
      mode === "compare" ? "Comparing…" : mode === "models" ? "Comparing models…" : mode === "repo" ? "Searching…" : "Asking…";

    const endpoint =
      mode === "compare" ? "/compare" : mode === "models" ? "/compare_models" : mode === "repo" ? "/repo_qa" : "/ask";
    let body;
    const questionId = evalSelect && evalSelect.value ? evalSelect.value : null;
    if (mode === "repo") {
      body = { question, answer: true };
    } else if (mode === "compare") {
      body = { question, question_id: questionId };
    } else if (mode === "models") {
      const models = Array.from(document.querySelectorAll("#model-row input:checked")).map((el) => el.value);
      if (!models.length) {
        renderError(question, "Select at least one model to compare.");
        submitBtn.disabled = false;
        submitLabel.textContent = prevLabel;
        return;
      }
      body = { question, models, question_id: questionId };
    } else {
      body = { question, rag: true, question_id: questionId };
    }

    try {
      const data = await fetchJson(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (mode === "repo") renderRepoResult(data);
      else if (mode === "compare") renderCompareResult(data);
      else if (mode === "models") renderModelsResult(data);
      else renderAskResult(data);
    } catch (err) {
      renderError(question, err.message || "Could not reach the server.");
    } finally {
      submitBtn.disabled = false;
      submitLabel.textContent = prevLabel;
    }
  }

  textarea.addEventListener("input", () => {
    if (!evalSelect || !evalSelect.value) return;
    const opt = evalSelect.selectedOptions[0];
    if (opt && opt.dataset.question !== textarea.value) {
      evalSelect.value = "";
      evalHint.textContent =
        "Edited \u2014 back to a free question, so the three ground-truth metrics are N/A.";
    }
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const question = textarea.value.trim();
    if (!question) return;
    submitQuestion(question);
  });

  refreshKbStatus();
  refreshFileList();
  loadEvalQuestions();
})();
