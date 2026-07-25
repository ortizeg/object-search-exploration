// main.js — the tracer. End to end: load methods + images, render the chosen image on the
// canvas, gate box-drawing behind method selection (UI-01), and on release convert the drag
// to image coordinates and POST /search, logging the returned match count. Rendering the
// result overlay is Plan 04-02; this file only proves the round trip is wired and correct.

import { Viewport, finalizeBox } from "./viewport.js";
import { buildForm } from "./form.js";
import { getMethods, getImages, postSearch, imageUrl } from "./api.js";

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById("stage"));
const methodSelect = /** @type {HTMLSelectElement} */ (document.getElementById("method"));
const imageSelect = /** @type {HTMLSelectElement} */ (document.getElementById("image"));
const configHost = document.getElementById("config");
const searchButton = /** @type {HTMLButtonElement} */ (document.getElementById("search"));
const statusEl = document.getElementById("status");

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
};

/** True only once a method is selected — the single gate UI-01 depends on. */
function drawingEnabled() {
  return state.method !== null && viewport.image !== null;
}

function setStatus(message) {
  if (statusEl) statusEl.textContent = message;
}

function render() {
  const boxes = [];
  if (state.box) {
    boxes.push({
      x: state.box.x0,
      y: state.box.y0,
      w: state.box.x1 - state.box.x0,
      h: state.box.y1 - state.box.y0,
      color: "#00e5ff",
    });
  } else if (state.drag) {
    // Live rubber-band while dragging (in image space; render is transform-aware).
    const a = state.drag.start;
    const b = state.drag.current;
    boxes.push({
      x: Math.min(a.x, b.x),
      y: Math.min(a.y, b.y),
      w: Math.abs(b.x - a.x),
      h: Math.abs(b.y - a.y),
      color: "#ffd166",
    });
  }
  viewport.render(boxes);
}

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
  viewport.setImage(img, img.naturalWidth, img.naturalHeight);
  render();
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

/** Run a search for the current box + method + config, and log the match count (tracer). */
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
    const count = Array.isArray(result?.matches) ? result.matches.length : 0;
    // eslint-disable-next-line no-console
    console.log("[search] run", run_id, "->", count, "matches", body.exemplar.box, result);
    setStatus(`Run ${run_id}: ${count} match(es). Overlay lands in Plan 04-02.`);
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
  // UI-01 gate: no method selected (or no image) => drawing is disabled, full stop.
  if (!drawingEnabled()) {
    setStatus("Pick a method first — drawing is disabled until then.");
    return;
  }
  canvas.setPointerCapture(e.pointerId);
  const start = viewport.screenToImage(e.clientX, e.clientY);
  state.box = null;
  state.drag = { id: e.pointerId, start, current: start };
  e.preventDefault();
  requestAnimationFrame(render);
});

canvas.addEventListener("pointermove", (e) => {
  if (state.pan && e.pointerId === state.pan.id) {
    viewport.panBy(e.clientX - state.pan.startX, e.clientY - state.pan.startY);
    state.pan.startX = e.clientX;
    state.pan.startY = e.clientY;
    requestAnimationFrame(render);
    return;
  }
  if (!state.drag || e.pointerId !== state.drag.id) return;
  state.drag.current = viewport.screenToImage(e.clientX, e.clientY);
  requestAnimationFrame(render);
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
  render();
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
    requestAnimationFrame(render);
  },
  { passive: false },
);

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
  render();
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
