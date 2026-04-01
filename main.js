/**
 * Tiny-TRIBE browser app entry point.
 *
 * Wires together: model loading, video upload, preprocessing,
 * inference, and 3D brain visualization.
 */

import { TinyTribeInference } from "./inference.js";
import { BrainVisualization } from "./brain-viz.js";
import { initTokenizer, tokenizeText, extractAudioMel, extractVideoFrames } from "./preprocessing.js";

// --- State ---
let inference = null;
let brainViz = null;
let currentVideoFile = null;

// --- DOM refs ---
const statusEl = document.getElementById("status");
const runBtn = document.getElementById("run-btn");
const subjectSelect = document.getElementById("subject-select");
const uploadArea = document.getElementById("upload-area");
const videoInput = document.getElementById("video-input");
const brainCanvas = document.getElementById("brain-canvas");
const metricLatency = document.getElementById("metric-latency");

function setStatus(msg, type = "loading") {
  statusEl.textContent = msg;
  statusEl.className = `status ${type}`;
}

// --- Init ---
async function init() {
  // Start brain visualization immediately
  brainViz = new BrainVisualization(brainCanvas);
  window.addEventListener("resize", () => brainViz.resize());

  // Load models in parallel
  setStatus("Loading models & tokenizer...");
  try {
    const [_, __] = await Promise.all([
      (async () => {
        inference = new TinyTribeInference();
        await inference.init();
      })(),
      initTokenizer(),
    ]);
    setStatus("Models loaded. Upload a video to begin.", "ready");
  } catch (e) {
    setStatus(`Model loading failed: ${e.message}`, "error");
    console.error(e);
    return;
  }
}

// --- Upload handling ---
uploadArea.addEventListener("click", () => videoInput.click());
uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.style.borderColor = "#6366f1";
});
uploadArea.addEventListener("dragleave", () => {
  uploadArea.style.borderColor = "#333";
});
uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.style.borderColor = "#333";
  if (e.dataTransfer.files.length) {
    handleFile(e.dataTransfer.files[0]);
  }
});
videoInput.addEventListener("change", (e) => {
  if (e.target.files.length) {
    handleFile(e.target.files[0]);
  }
});

function handleFile(file) {
  if (!file.type.startsWith("video/")) {
    setStatus("Please upload a video file.", "error");
    return;
  }
  currentVideoFile = file;
  const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
  setStatus(`Loaded: ${file.name} (${sizeMB} MB). Click "Run Prediction".`, "ready");
  runBtn.disabled = false;
}

// --- Run prediction ---
runBtn.addEventListener("click", async () => {
  if (!currentVideoFile || !inference?.ready) return;

  runBtn.disabled = true;
  const subjectId = parseInt(subjectSelect.value);

  try {
    // Step 1: Extract features from video
    setStatus("Extracting video frames...");
    const videoFrames = await extractVideoFrames(currentVideoFile);

    setStatus("Extracting audio...");
    const audioMel = await extractAudioMel(currentVideoFile);

    // Use filename as placeholder text (real app would use speech-to-text)
    setStatus("Tokenizing text...");
    const textTokens = tokenizeText(
      currentVideoFile.name.replace(/\.[^/.]+$/, "").replace(/[_-]/g, " ")
    );

    // Step 2: Run inference
    setStatus("Running inference...");
    const { predictions, latencyMs } = await inference.predict(
      textTokens,
      audioMel,
      videoFrames,
      subjectId
    );

    // Step 3: Visualize
    const vertexData = predictions.data;
    // Average over time dimension if needed: predictions is (1, 5124, T)
    const nVertices = 5124;
    const T = vertexData.length / nVertices;
    const avgActivations = new Float32Array(nVertices);
    for (let v = 0; v < nVertices; v++) {
      let sum = 0;
      for (let t = 0; t < T; t++) {
        sum += vertexData[v * T + t];
      }
      avgActivations[v] = sum / T;
    }

    brainViz.updateActivations(avgActivations);
    metricLatency.textContent = Math.round(latencyMs);
    setStatus(`Done! Inference: ${Math.round(latencyMs)}ms`, "ready");
  } catch (e) {
    setStatus(`Error: ${e.message}`, "error");
    console.error(e);
  } finally {
    runBtn.disabled = false;
  }
});

// --- Start ---
init();
