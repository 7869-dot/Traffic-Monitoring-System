// Traffic Monitoring System — frontend logic
// Talks to the FastAPI backend; no build step required.

(function () {
  "use strict";

  const DEFAULT_API = "http://127.0.0.1:8000";
  const STORAGE_KEY = "tms_api_url";

  const els = {
    apiUrl: document.getElementById("api-url"),
    checkBtn: document.getElementById("check-btn"),
    statusBadge: document.getElementById("status-badge"),
    dropZone: document.getElementById("drop-zone"),
    fileInput: document.getElementById("file-input"),
    dropText: document.getElementById("drop-text"),
    sampleRate: document.getElementById("sample-rate"),
    maxFrames: document.getElementById("max-frames"),
    analyzeBtn: document.getElementById("analyze-btn"),
    progress: document.getElementById("progress"),
    error: document.getElementById("error"),
    results: document.getElementById("results"),
    resultMeta: document.getElementById("result-meta"),
    uniqueGrid: document.getElementById("unique-grid"),
    peakGrid: document.getElementById("peak-grid"),
  };

  let selectedFile = null;

  // ---- helpers -------------------------------------------------------------

  function apiBase() {
    return (els.apiUrl.value || DEFAULT_API).trim().replace(/\/+$/, "");
  }

  function setBadge(state, text) {
    els.statusBadge.className = "badge badge-" + state;
    els.statusBadge.textContent = text;
  }

  function showError(msg) {
    els.error.textContent = msg;
    els.error.classList.remove("hidden");
  }

  function clearError() {
    els.error.classList.add("hidden");
    els.error.textContent = "";
  }

  function updateAnalyzeState() {
    els.analyzeBtn.disabled = !selectedFile;
  }

  // ---- backend health ------------------------------------------------------

  async function checkHealth() {
    setBadge("unknown", "checking…");
    try {
      const res = await fetch(apiBase() + "/health", { method: "GET" });
      if (res.ok) {
        setBadge("ok", "connected");
      } else {
        setBadge("bad", "error " + res.status);
      }
    } catch (e) {
      setBadge("bad", "unreachable");
    }
  }

  // ---- file selection ------------------------------------------------------

  function onFileChosen(file) {
    if (!file) return;
    selectedFile = file;
    els.dropText.textContent = file.name + "  (" + formatBytes(file.size) + ")";
    clearError();
    updateAnalyzeState();
  }

  function formatBytes(bytes) {
    if (!bytes) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return (bytes / Math.pow(1024, i)).toFixed(1) + " " + units[i];
  }

  // ---- analyze -------------------------------------------------------------

  async function analyze() {
    if (!selectedFile) return;
    clearError();
    els.results.classList.add("hidden");
    els.progress.classList.remove("hidden");
    els.analyzeBtn.disabled = true;

    const sampleRate = parseInt(els.sampleRate.value, 10) || 5;
    const maxFrames = parseInt(els.maxFrames.value, 10);

    const params = new URLSearchParams({ frame_sample_rate: String(sampleRate) });
    if (Number.isFinite(maxFrames) && maxFrames > 0) {
      params.set("max_frames", String(maxFrames));
    }

    const form = new FormData();
    form.append("file", selectedFile);

    try {
      const res = await fetch(
        apiBase() + "/api/video/upload-and-process?" + params.toString(),
        { method: "POST", body: form }
      );

      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || ("Request failed (" + res.status + ")"));
      }
      renderResults(data);
    } catch (e) {
      showError("Failed to process video: " + e.message);
    } finally {
      els.progress.classList.add("hidden");
      els.analyzeBtn.disabled = false;
    }
  }

  // ---- rendering -----------------------------------------------------------

  const VEHICLE_ORDER = ["car", "truck", "bus", "motorcycle", "bicycle", "total"];

  function renderResults(data) {
    els.resultMeta.innerHTML = "";
    addMeta("File", data.filename || "—");
    addMeta("Duration", (data.duration_sec ?? 0) + " s");
    addMeta("FPS", data.fps ?? "—");
    addMeta("Total frames", data.total_frames ?? "—");
    addMeta("Processed frames", data.processed_frames ?? "—");
    addMeta("Sample rate", "every " + (data.frame_sample_rate ?? "—"));

    renderGrid(els.uniqueGrid, data.unique_estimate || {});
    renderGrid(els.peakGrid, data.peak_counts || {});

    els.results.classList.remove("hidden");
    els.results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function addMeta(label, value) {
    const span = document.createElement("span");
    span.innerHTML = label + ": <b>" + value + "</b>";
    els.resultMeta.appendChild(span);
  }

  function renderGrid(grid, counts) {
    grid.innerHTML = "";
    VEHICLE_ORDER.forEach(function (key) {
      if (!(key in counts)) return;
      const cell = document.createElement("div");
      cell.className = "stat" + (key === "total" ? " total" : "");
      cell.innerHTML =
        '<div class="num">' + (counts[key] ?? 0) + "</div>" +
        '<div class="lbl">' + key + "</div>";
      grid.appendChild(cell);
    });
  }

  // ---- wiring --------------------------------------------------------------

  function init() {
    els.apiUrl.value = localStorage.getItem(STORAGE_KEY) || DEFAULT_API;

    els.apiUrl.addEventListener("change", function () {
      localStorage.setItem(STORAGE_KEY, apiBase());
      checkHealth();
    });
    els.checkBtn.addEventListener("click", checkHealth);

    els.fileInput.addEventListener("change", function (e) {
      onFileChosen(e.target.files[0]);
    });

    // Drag & drop
    ["dragenter", "dragover"].forEach(function (ev) {
      els.dropZone.addEventListener(ev, function (e) {
        e.preventDefault();
        els.dropZone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      els.dropZone.addEventListener(ev, function (e) {
        e.preventDefault();
        els.dropZone.classList.remove("dragover");
      });
    });
    els.dropZone.addEventListener("drop", function (e) {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        onFileChosen(e.dataTransfer.files[0]);
      }
    });

    els.analyzeBtn.addEventListener("click", analyze);

    checkHealth();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
