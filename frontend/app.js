// Traffic Monitoring System — frontend logic
// Talks to the FastAPI backend using the async job API; no build step required.

(function () {
  "use strict";

  const DEFAULT_API = "http://127.0.0.1:8000";
  const API_KEY = "tms_api_url";
  const THEME_KEY = "tms_theme";
  const POLL_MS = 1200;

  const VEHICLES = {
    car: "🚗",
    truck: "🚚",
    bus: "🚌",
    motorcycle: "🏍️",
    bicycle: "🚲",
    total: "🧮",
  };
  const ORDER = ["car", "truck", "bus", "motorcycle", "bicycle"];
  const COLORS = {
    car: "#4cc2ff",
    truck: "#ffb02e",
    bus: "#f87171",
    motorcycle: "#a78bfa",
    bicycle: "#34d399",
  };

  const $ = (id) => document.getElementById(id);
  const els = {
    apiUrl: $("api-url"), checkBtn: $("check-btn"), statusBadge: $("status-badge"),
    themeBtn: $("theme-btn"),
    dropZone: $("drop-zone"), fileInput: $("file-input"),
    dropTitle: $("drop-title"), dropSub: $("drop-sub"),
    sampleRate: $("sample-rate"), maxFrames: $("max-frames"),
    analyzeBtn: $("analyze-btn"),
    progress: $("progress"), progressFill: $("progress-fill"),
    progressStage: $("progress-stage"), progressPct: $("progress-pct"),
    error: $("error"),
    results: $("results"), resultFile: $("result-file"),
    heroTotal: $("hero-total"), heroLabel: $("hero-label"), resultMeta: $("result-meta"),
    tabHint: $("tab-hint"), chart: $("chart"),
  };

  let selectedFile = null;
  let lastResult = null;
  let activeTab = "unique";

  // ---- helpers -------------------------------------------------------------

  const apiBase = () => (els.apiUrl.value || DEFAULT_API).trim().replace(/\/+$/, "");

  function setBadge(state, text) {
    els.statusBadge.className = "badge badge-" + state;
    els.statusBadge.textContent = text;
  }
  function showError(msg) { els.error.textContent = msg; els.error.classList.remove("hidden"); }
  function clearError() { els.error.classList.add("hidden"); els.error.textContent = ""; }

  function formatBytes(bytes) {
    if (!bytes) return "0 B";
    const u = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return (bytes / Math.pow(1024, i)).toFixed(1) + " " + u[i];
  }

  // ---- backend health ------------------------------------------------------

  async function checkHealth() {
    setBadge("unknown", "checking…");
    try {
      const res = await fetch(apiBase() + "/health");
      setBadge(res.ok ? "ok" : "bad", res.ok ? "connected" : "error " + res.status);
    } catch (e) {
      setBadge("bad", "unreachable");
    }
  }

  // ---- file selection ------------------------------------------------------

  function onFileChosen(file) {
    if (!file) return;
    selectedFile = file;
    els.dropTitle.textContent = file.name;
    els.dropSub.textContent = formatBytes(file.size) + " · ready to analyze";
    els.dropZone.classList.add("has-file");
    clearError();
    els.analyzeBtn.disabled = false;
  }

  // ---- progress ------------------------------------------------------------

  function setProgress(stage, fraction, indeterminate) {
    els.progress.classList.remove("hidden");
    els.progressStage.textContent = stage;
    if (indeterminate) {
      els.progressFill.classList.add("indeterminate");
      els.progressPct.textContent = "";
    } else {
      els.progressFill.classList.remove("indeterminate");
      const pct = Math.round((fraction || 0) * 100);
      els.progressFill.style.width = pct + "%";
      els.progressPct.textContent = pct + "%";
    }
  }
  function hideProgress() { els.progress.classList.add("hidden"); }

  // ---- analyze (async job + polling) --------------------------------------

  async function analyze() {
    if (!selectedFile) return;
    clearError();
    els.results.classList.add("hidden");
    els.analyzeBtn.disabled = true;
    setProgress("Uploading…", 0, true);

    const sampleRate = parseInt(els.sampleRate.value, 10) || 5;
    const maxFrames = parseInt(els.maxFrames.value, 10);
    const params = new URLSearchParams({ frame_sample_rate: String(sampleRate) });
    if (Number.isFinite(maxFrames) && maxFrames > 0) params.set("max_frames", String(maxFrames));

    const form = new FormData();
    form.append("file", selectedFile);

    try {
      const res = await fetch(apiBase() + "/api/video/upload-async?" + params.toString(),
        { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || "Upload failed (" + res.status + ")");

      const result = await pollJob(data.job_id);
      renderResults(result);
    } catch (e) {
      showError("Failed: " + e.message);
    } finally {
      hideProgress();
      els.analyzeBtn.disabled = false;
    }
  }

  function pollJob(jobId) {
    setProgress("Processing…", 0, false);
    return new Promise((resolve, reject) => {
      const tick = async () => {
        try {
          const res = await fetch(apiBase() + "/api/video/job/" + jobId);
          if (!res.ok) throw new Error("Job lookup failed (" + res.status + ")");
          const job = await res.json();

          if (job.status === "processing" || job.status === "queued") {
            setProgress(job.status === "queued" ? "Queued…" : "Detecting vehicles…",
                        job.progress, job.status === "queued");
            setTimeout(tick, POLL_MS);
          } else if (job.status === "done") {
            setProgress("Done", 1, false);
            resolve(job.result);
          } else {
            reject(new Error(job.error || "Processing failed"));
          }
        } catch (e) {
          reject(e);
        }
      };
      tick();
    });
  }

  // ---- rendering -----------------------------------------------------------

  function renderResults(data) {
    lastResult = data;
    els.resultFile.textContent = data.filename || "video";

    els.resultMeta.innerHTML = "";
    chip("Duration", (data.duration_sec ?? 0) + "s");
    chip("FPS", data.fps ?? "—");
    chip("Frames", data.total_frames ?? "—");
    chip("Processed", data.processed_frames ?? "—");
    chip("Sample", "1/" + (data.frame_sample_rate ?? "—"));

    renderTab(activeTab);
    els.results.classList.remove("hidden");
    els.results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function chip(label, value) {
    const c = document.createElement("span");
    c.className = "chip";
    c.innerHTML = label + " <b>" + value + "</b>";
    els.resultMeta.appendChild(c);
  }

  function renderTab(tab) {
    activeTab = tab;
    document.querySelectorAll(".tab").forEach((t) =>
      t.classList.toggle("active", t.dataset.tab === tab));

    if (!lastResult) return;
    const counts = tab === "unique" ? lastResult.unique_estimate : lastResult.peak_counts;
    els.tabHint.textContent = tab === "unique"
      ? "Distinct vehicles tracked across the whole clip."
      : "Most vehicles seen at the same time in a single frame.";
    els.heroLabel.textContent = tab === "unique"
      ? "estimated unique vehicles"
      : "peak vehicles in one frame";

    animateNumber(els.heroTotal, (counts && counts.total) || 0);
    renderChart(counts || {});
  }

  function renderChart(counts) {
    const max = Math.max(1, ...ORDER.map((k) => counts[k] || 0));
    els.chart.innerHTML = "";
    ORDER.forEach((key) => {
      const val = counts[key] || 0;
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML =
        '<div class="bar-name">' + (VEHICLES[key] || "") + " " + key + "</div>" +
        '<div class="bar-track"><div class="bar-fill" style="background:' + COLORS[key] + '"></div></div>' +
        '<div class="bar-val">' + val + "</div>";
      els.chart.appendChild(row);
      // animate width on next frame
      requestAnimationFrame(() => {
        row.querySelector(".bar-fill").style.width = ((val / max) * 100) + "%";
      });
    });
  }

  function animateNumber(el, target) {
    const start = 0, dur = 700, t0 = performance.now();
    function step(now) {
      const p = Math.min((now - t0) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(start + (target - start) * eased);
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  // ---- theme ---------------------------------------------------------------

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    els.themeBtn.textContent = theme === "light" ? "☀️" : "🌙";
    localStorage.setItem(THEME_KEY, theme);
  }

  // ---- wiring --------------------------------------------------------------

  function init() {
    els.apiUrl.value = localStorage.getItem(API_KEY) || DEFAULT_API;
    applyTheme(localStorage.getItem(THEME_KEY) || "dark");

    els.apiUrl.addEventListener("change", () => {
      localStorage.setItem(API_KEY, apiBase());
      checkHealth();
    });
    els.checkBtn.addEventListener("click", checkHealth);
    els.themeBtn.addEventListener("click", () =>
      applyTheme(document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light"));

    els.fileInput.addEventListener("change", (e) => onFileChosen(e.target.files[0]));

    ["dragenter", "dragover"].forEach((ev) =>
      els.dropZone.addEventListener(ev, (e) => { e.preventDefault(); els.dropZone.classList.add("dragover"); }));
    ["dragleave", "drop"].forEach((ev) =>
      els.dropZone.addEventListener(ev, (e) => { e.preventDefault(); els.dropZone.classList.remove("dragover"); }));
    els.dropZone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) onFileChosen(e.dataTransfer.files[0]);
    });

    els.analyzeBtn.addEventListener("click", analyze);
    document.querySelectorAll(".tab").forEach((t) =>
      t.addEventListener("click", () => renderTab(t.dataset.tab)));

    checkHealth();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
