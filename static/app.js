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
      submitLabel.textContent = mode === "compare" ? "Compare" : mode === "models" ? "Compare models" : "Ask";
      modelRow.hidden = mode !== "models";
    });
  });

  document.querySelectorAll(".quick").forEach((chip) => {
    chip.addEventListener("click", () => {
      textarea.value = chip.dataset.q;
      textarea.focus();
    });
  });

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
          </div>
          <div class="compare-col">
            <div class="col-label"><span class="dot on"></span> With your documents</div>
            <div class="answer"><p>${escapeHtml(data.with_rag).trim()}</p></div>
          </div>
        </div>
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
        </div>`;
      })
      .join("");
    results.insertAdjacentHTML(
      "afterbegin",
      `<div class="result-card">
        <div class="result-question">${escapeHtml(data.question)}</div>
        <div class="compare-grid models-grid">${cols}</div>
        ${tagRowHtml(data.sources)}
        ${coverageHtml(data.sources)}
      </div>`
    );
  }

  async function submitQuestion(question) {
    renderLoading(question);
    submitBtn.disabled = true;
    const prevLabel = submitLabel.textContent;
    submitLabel.textContent = mode === "compare" ? "Comparing…" : mode === "models" ? "Comparing models…" : "Asking…";

    const endpoint = mode === "compare" ? "/compare" : mode === "models" ? "/compare_models" : "/ask";
    let body;
    if (mode === "compare") {
      body = { question };
    } else if (mode === "models") {
      const models = Array.from(document.querySelectorAll("#model-row input:checked")).map((el) => el.value);
      if (!models.length) {
        renderError(question, "Select at least one model to compare.");
        submitBtn.disabled = false;
        submitLabel.textContent = prevLabel;
        return;
      }
      body = { question, models };
    } else {
      body = { question, rag: true };
    }

    try {
      const data = await fetchJson(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (mode === "compare") renderCompareResult(data);
      else if (mode === "models") renderModelsResult(data);
      else renderAskResult(data);
    } catch (err) {
      renderError(question, err.message || "Could not reach the server.");
    } finally {
      submitBtn.disabled = false;
      submitLabel.textContent = prevLabel;
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const question = textarea.value.trim();
    if (!question) return;
    submitQuestion(question);
  });

  refreshKbStatus();
  refreshFileList();
})();
