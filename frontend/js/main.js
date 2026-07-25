// main.js — the app shell wiring. End to end: load methods + images, render the chosen image
// on the canvas, gate box-drawing behind method selection (UI-01), and on release POST /search.
// This plan (04-02) adds the result + diagnostics overlays, the tiered rating widget, and the
// stats dashboard on top of the 04-01 canvas/transform foundation.

import { Viewport, finalizeBox } from "./viewport.js";
import { buildForm } from "./form.js";
import { getMethods, getImages, postSearch, imageUrl } from "./api.js";
import {
  drawResults,
  drawQueryBox,
  drawHeatmap,
  drawPointDiagnostics,
  presentDiagnosticFields,
  hitTestMatch,
} from "./overlay.js";
import { mountRating } from "./rating.js";

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById("stage"));
const methodSelect = /** @type {HTMLSelectElement} */ (document.getElementById("method"));
const imageSelect = /** @type {HTMLSelectElement} */ (document.getElementById("image"));
const configHost = document.getElementById("config");
const searchButton = /** @type {HTMLButtonElement} */ (document.getElementById("search"));
const statusEl = document.getElementById("status");
const overlayToggles = document.getElementById("overlay-toggles");
const ratingHost = document.getElementById("rating-host");
const tabRating = document.getElementById("tab-rating");
const tabStats = document.getElementById("tab-stats");
const ratingView = document.getElementById("rating-view");
const statsView = document.getElementById("stats-view");

// Swatch colours mirror overlay.js so the toggle legend reads at a glance. Kept here rather
// than exported from overlay.js — a five-line duplication is clearer than a shared constant.
const OVERLAY_SWATCH = {
  similarity_heatmap: "#ff8c42",
  keypoints: "#7cffb2",
  correspondences: "#c792ea",
  hough_peaks: "#ffb86c",
  proposals: "#5aa0ff",
};

const viewport = new Viewport(canvas);

const state = {
  /** @type {Array<{name:string, config_schema:object}>} */ methods: [],
  /** @type {string|null} */ method: null,
  /** @type {{readValues:()=>object}|null} */ form: null,
  /** @type {string|null} */ imageId: null,
  /** @type {{x0:number,y0:number,x1:number,y1:number}|null} */ box: null,
  /** @type {{id:number,start:{x:number,y:number},current:{x:number,y:number}}|null} */ drag:
    null,
  /** @type {{id:number,startX:number,startY:number}|null} */ pan: null,
  // --- result + overlay state (04-02) ---
  /** @type {number|null} */ runId: null,
  /** @type {Array<{box:{x:number,y:number,w:number,h:number},score:number,is_exemplar?:boolean}>} */
  matches: [],
  /** @type {object|null} */ diagnostics: null,
  /** @type {CanvasImageSource|null} */ heatmapImg: null,
  /** @type {Set<string>} */ overlayEnabled: new Set(),
  /** @type {Set<number>} indices a human marked wrong (per-match verdicts, Task 2) */
  wrongSet: new Set(),
  /** @type {boolean} true while the rating widget is in per-match verdict mode (Task 2) */
  verdictMode: false,
};

/** True only once a method is selected — the single gate UI-01 depends on. */
function drawingEnabled() {
  return state.method !== null && viewport.image !== null;
}

function setStatus(message) {
  if (statusEl) statusEl.textContent = message;
}

const dpr = () => window.devicePixelRatio || 1;

/** Composite one frame: image, heatmap (under boxes), results, query box, rubber-band, points. */
function renderScene() {
  const ctx = viewport.paintImage();
  if (!ctx) return;

  if (state.diagnostics && state.heatmapImg && state.overlayEnabled.has("similarity_heatmap")) {
    drawHeatmap(viewport, state.heatmapImg);
  }
  if (state.matches.length) {
    drawResults(viewport, state.matches, { wrongSet: state.wrongSet });
  }
  if (state.box) {
    drawQueryBox(viewport, state.box);
  }
  if (state.drag) {
    // Live rubber-band while dragging, in image space (transform-aware).
    const a = state.drag.start;
    const b = state.drag.current;
    ctx.setTransform(viewport.zoom, 0, 0, viewport.zoom, viewport.panX, viewport.panY);
    ctx.setLineDash([]);
    ctx.lineWidth = (1.5 * dpr()) / viewport.zoom;
    ctx.strokeStyle = "#ffd166";
    ctx.strokeRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y));
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }
  if (state.diagnostics) {
    drawPointDiagnostics(viewport, state.diagnostics, state.overlayEnabled);
  }
}

/** Rebuild the overlay-toggle checkboxes for the diagnostic fields present in this result. */
function buildOverlayToggles() {
  if (!overlayToggles) return;
  overlayToggles.replaceChildren();
  const present = presentDiagnosticFields(state.diagnostics);
  if (present.length === 0) {
    overlayToggles.hidden = true;
    return;
  }
  overlayToggles.hidden = false;
  const legend = document.createElement("span");
  legend.className = "overlay-toggles-legend";
  legend.textContent = "Overlays";
  overlayToggles.appendChild(legend);
  for (const { key, label } of present) {
    const wrap = document.createElement("label");
    wrap.className = "overlay-toggle";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = state.overlayEnabled.has(key);
    input.addEventListener("change", () => {
      if (input.checked) state.overlayEnabled.add(key);
      else state.overlayEnabled.delete(key);
      requestAnimationFrame(renderScene);
    });
    const swatch = document.createElement("span");
    swatch.className = "overlay-swatch";
    swatch.style.background = OVERLAY_SWATCH[key] || "#00e5ff";
    const text = document.createElement("span");
    text.textContent = label;
    wrap.append(input, swatch, text);
    overlayToggles.appendChild(wrap);
  }
}

/** Decode a base64 similarity-heatmap PNG into an Image, then re-render when it is ready. */
function loadHeatmap(heatmap) {
  state.heatmapImg = null;
  if (!heatmap || !heatmap.png_b64) return;
  const img = new Image();
  img.onload = () => {
    state.heatmapImg = img;
    requestAnimationFrame(renderScene);
  };
  img.src = `data:image/png;base64,${heatmap.png_b64}`;
}

/** Adopt a fresh search result: store matches + diagnostics, prime overlays, re-render. */
function adoptResult(runId, result) {
  state.runId = runId;
  state.matches = Array.isArray(result?.matches) ? result.matches : [];
  state.diagnostics = result?.diagnostics || null;
  state.wrongSet = new Set();
  state.verdictMode = false;
  // Default every present diagnostic overlay ON so a run is legible without hunting for toggles.
  state.overlayEnabled = new Set(presentDiagnosticFields(state.diagnostics).map((f) => f.key));
  loadHeatmap(state.diagnostics?.similarity_heatmap);
  buildOverlayToggles();
  if (typeof onResult === "function") onResult(runId, state.matches);
  requestAnimationFrame(renderScene);
}

/** Clear any previous result — called when a new query box begins. */
function clearResult() {
  state.runId = null;
  state.matches = [];
  state.diagnostics = null;
  state.heatmapImg = null;
  state.overlayEnabled = new Set();
  state.wrongSet = new Set();
  state.verdictMode = false;
  if (overlayToggles) overlayToggles.hidden = true;
  if (typeof onResult === "function") onResult(null, []);
}

// Hook points assigned further down, once the rating widget and stats renderer are defined.
// Declared here so the pointer handlers and result adoption can reference them before then.
/** @type {((runId:number|null, matches:Array<object>)=>void)|null} */
let onResult = null;
/** @type {((index:number)=>void)|null} */
let onVerdictToggle = null;

/** Load an image_id onto the canvas and reset the fit. */
async function loadImage(imageId) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error(`failed to load image ${imageId}`));
    img.src = imageUrl(imageId);
  });
  state.imageId = imageId;
  state.box = null;
  clearResult();
  viewport.setImage(img, img.naturalWidth, img.naturalHeight);
  renderScene();
  canvas.setAttribute("aria-disabled", String(!drawingEnabled()));
}

/** Rebuild the config form for the selected method (UI-07). */
function selectMethod(name) {
  state.method = name || null;
  const method = state.methods.find((m) => m.name === name);
  if (configHost) configHost.replaceChildren();
  if (method) {
    const { element, readValues } = buildForm(method.config_schema);
    state.form = { readValues };
    if (configHost) configHost.appendChild(element);
  } else {
    state.form = null;
  }
  searchButton.disabled = !drawingEnabled() || state.box === null;
  canvas.setAttribute("aria-disabled", String(!drawingEnabled()));
}

/** Run a search for the current box + method + config, then overlay the result. */
async function runSearch() {
  if (!state.method || !state.imageId || !state.box) return;
  const config = state.form ? state.form.readValues() : {};
  const body = {
    image_id: state.imageId,
    exemplar: {
      box: {
        x: state.box.x0,
        y: state.box.y0,
        w: state.box.x1 - state.box.x0,
        h: state.box.y1 - state.box.y0,
      },
    },
    method: state.method,
    config,
  };
  setStatus("Searching…");
  try {
    const { run_id, result } = await postSearch(body);
    adoptResult(run_id, result);
    const count = state.matches.length;
    setStatus(`Run ${run_id}: ${count} match(es). Rate it in the right panel.`);
  } catch (err) {
    setStatus(`Search failed: ${err.message}`);
    // eslint-disable-next-line no-console
    console.error(err);
  }
}

// --- Pointer interaction: draw (left), pan (middle), zoom (wheel) ---------------------

canvas.addEventListener("pointerdown", (e) => {
  if (e.button === 1) {
    // Middle button pans regardless of method selection.
    state.pan = { id: e.pointerId, startX: e.clientX, startY: e.clientY };
    canvas.setPointerCapture(e.pointerId);
    e.preventDefault();
    return;
  }
  if (e.button !== 0 || !e.isPrimary) return;
  // Per-match verdict mode (Task 2): a left click marks the box under the cursor wrong rather
  // than starting a new query, so the current run can be rated without redrawing.
  if (state.verdictMode && state.matches.length) {
    const p = viewport.screenToImage(e.clientX, e.clientY);
    const index = hitTestMatch(state.matches, p.x, p.y);
    if (index >= 0 && typeof onVerdictToggle === "function") {
      onVerdictToggle(index);
      requestAnimationFrame(renderScene);
    }
    e.preventDefault();
    return;
  }
  // UI-01 gate: no method selected (or no image) => drawing is disabled, full stop.
  if (!drawingEnabled()) {
    setStatus("Pick a method first — drawing is disabled until then.");
    return;
  }
  canvas.setPointerCapture(e.pointerId);
  const start = viewport.screenToImage(e.clientX, e.clientY);
  state.box = null;
  clearResult();
  state.drag = { id: e.pointerId, start, current: start };
  e.preventDefault();
  requestAnimationFrame(renderScene);
});

canvas.addEventListener("pointermove", (e) => {
  if (state.pan && e.pointerId === state.pan.id) {
    viewport.panBy(e.clientX - state.pan.startX, e.clientY - state.pan.startY);
    state.pan.startX = e.clientX;
    state.pan.startY = e.clientY;
    requestAnimationFrame(renderScene);
    return;
  }
  if (!state.drag || e.pointerId !== state.drag.id) return;
  state.drag.current = viewport.screenToImage(e.clientX, e.clientY);
  requestAnimationFrame(renderScene);
});

function endDrag(e) {
  if (state.pan && e.pointerId === state.pan.id) {
    state.pan = null;
    return;
  }
  if (!state.drag || e.pointerId !== state.drag.id) return;
  const box = finalizeBox(
    state.drag.start,
    state.drag.current,
    viewport.naturalW,
    viewport.naturalH,
  );
  state.drag = null;
  state.box = box;
  searchButton.disabled = !drawingEnabled() || state.box === null;
  renderScene();
  if (box) {
    void runSearch();
  } else {
    setStatus("Box too small — drag a larger region (min 8×8 image px).");
  }
}

canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("lostpointercapture", (e) => {
  if (state.drag && e.pointerId === state.drag.id) state.drag = null;
  if (state.pan && e.pointerId === state.pan.id) state.pan = null;
});

canvas.addEventListener(
  "wheel",
  (e) => {
    if (!viewport.image) return;
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    viewport.zoomAbout(factor, e.clientX, e.clientY);
    requestAnimationFrame(renderScene);
  },
  { passive: false },
);

// --- Rating widget wiring -------------------------------------------------------------
/** @type {{toggleVerdict:(index:number)=>void}|null} */
let ratingWidget = null;

/** Rebuild the rating widget for a fresh run, or reset to the placeholder when cleared. */
function showRating(runId, matches) {
  if (!ratingHost) return;
  if (runId === null || matches.length === 0) {
    ratingHost.replaceChildren();
    const msg = document.createElement("p");
    msg.className = "muted";
    msg.textContent =
      runId === null
        ? "Run a search, then rate the result here."
        : "This run returned no matches — nothing to rate.";
    ratingHost.appendChild(msg);
    ratingWidget = null;
    return;
  }
  showTab("rating");
  ratingWidget = mountRating(ratingHost, {
    runId,
    matches,
    wrongSet: state.wrongSet,
    setVerdictMode: (on) => {
      state.verdictMode = on;
      if (!on) state.wrongSet.clear();
      requestAnimationFrame(renderScene);
    },
    requestRender: () => requestAnimationFrame(renderScene),
    onSubmitted: () => {
      void refreshStats();
    },
  });
}
onResult = showRating;
onVerdictToggle = (index) => {
  if (ratingWidget) ratingWidget.toggleVerdict(index);
};

// --- Panel tab switching --------------------------------------------------------------
function showTab(which) {
  const rating = which === "rating";
  if (ratingView) ratingView.hidden = !rating;
  if (statsView) statsView.hidden = rating;
  if (tabRating) tabRating.classList.toggle("active", rating);
  if (tabStats) tabStats.classList.toggle("active", !rating);
}
if (tabRating) tabRating.addEventListener("click", () => showTab("rating"));
if (tabStats)
  tabStats.addEventListener("click", () => {
    showTab("stats");
    void refreshStats();
  });

// Stats rendering is wired in Task 3; a no-op keeps the rating path standalone until then.
/** @type {() => Promise<void>} */
let refreshStats = async () => {};

// --- Control wiring -------------------------------------------------------------------

methodSelect.addEventListener("change", () => selectMethod(methodSelect.value));
imageSelect.addEventListener("change", () => {
  if (imageSelect.value) void loadImage(imageSelect.value).catch((err) => setStatus(err.message));
});
searchButton.addEventListener("click", () => void runSearch());

// Keep the backing store in step with layout and DPR changes (PITFALLS §9.1, §9.8).
const resizeObserver = new ResizeObserver(() => {
  viewport.syncCanvasSize();
  viewport.fitContain();
  renderScene();
});
resizeObserver.observe(canvas);

// --- Bootstrap ------------------------------------------------------------------------

async function bootstrap() {
  try {
    const [methods, images] = await Promise.all([getMethods(), getImages()]);
    state.methods = methods;

    methodSelect.replaceChildren();
    const placeholder = new Option("Choose a method…", "", true, true);
    placeholder.disabled = true;
    methodSelect.add(placeholder);
    for (const m of methods) {
      methodSelect.add(new Option(m.name, m.name));
    }

    imageSelect.replaceChildren();
    if (images.length === 0) {
      imageSelect.add(new Option("No demo images found", "", true, true));
    } else {
      for (const img of images) {
        imageSelect.add(new Option(`${img.id} (${img.width}×${img.height})`, img.id));
      }
      await loadImage(images[0].id);
      imageSelect.value = images[0].id;
    }
    setStatus("Ready. Pick a method, then draw a box.");
  } catch (err) {
    setStatus(`Failed to load: ${err.message}`);
    // eslint-disable-next-line no-console
    console.error(err);
  }
}

void bootstrap();
