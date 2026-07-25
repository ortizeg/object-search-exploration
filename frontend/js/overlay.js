// overlay.js — the result and diagnostics layers drawn on top of the scene image.
//
// Two hard rules from the phase context live here:
//
//   1. Every match box is placed through the viewport's image-space transform (the same one
//      `viewport.paintImage` set up), so a box tracks zoom and pan exactly — the server's
//      integer box is stroked verbatim, never re-derived from screen coordinates.
//
//   2. Which diagnostic overlays exist is driven by the PRESENCE of a named field in the
//      payload, NEVER by the method name. A method the UI has never seen still renders its
//      heatmap / keypoints / correspondences / Hough peaks / proposals for free, because the
//      only knowledge encoded here is the shape of `Diagnostics`, not any method's identity.
//      (Grep-enforced: no registered method name may appear in frontend/.)
//
// Coordinate spaces (see viewport.js): boxes are stroked in IMAGE space so they scale with
// zoom; labels and dots are drawn in BACKING-STORE space at a fixed size, positioned via
// `viewport.imageToBacking`, so text does not balloon or shrink with the zoom. Every helper
// resets the context transform to image space (`_imageSpace`) or backing space (`_screenSpace`)
// itself, so callers may invoke them in any order after `viewport.paintImage()`.

const COL_MATCH = "#00e5ff"; // a found instance
const COL_EXEMPLAR = "#ffd166"; // the self-match / drawn query box — distinct on purpose
const COL_WRONG = "#ff5470"; // a match a human has marked wrong (per-match verdict)
const COL_KEYPOINT = "#7cffb2";
const COL_CORR = "#c792ea";
const COL_HOUGH = "#ffb86c";
const COL_PROPOSAL = "#5aa0ff";
const COL_MARKER = "#ffd166"; // a found marker box + its pointing arrow
const COL_MARKER_LINK = "#ff8c42"; // the thin connector from a marker to its chosen proposal

// Fixed pixel length (backing space) of a drawn pointing arrow and its head, so arrows read the
// same at any zoom, exactly like labels and dots.
const MARKER_ARROW_LEN = 34;
const MARKER_ARROW_HEAD = 8;

// The named diagnostic fields the UI knows how to draw, in draw order. Presence-driven: a
// field appears as a toggle only when the payload actually carries it. No method name here.
const DIAGNOSTIC_FIELDS = [
  { key: "similarity_heatmap", label: "Similarity heatmap" },
  { key: "keypoints", label: "Keypoints" },
  { key: "correspondences", label: "Correspondences" },
  { key: "hough_peaks", label: "Hough peaks" },
  { key: "proposals", label: "Proposals" },
  { key: "markers", label: "Markers & arrows" },
];

/**
 * The diagnostic fields actually present (and non-empty) in a payload, in draw order.
 * This is what the toggle checkboxes are built from — a field the payload omits gets no
 * toggle, and a field it carries gets one regardless of which method produced it.
 * @param {object|null|undefined} diagnostics
 * @returns {Array<{key:string, label:string}>}
 */
export function presentDiagnosticFields(diagnostics) {
  if (!diagnostics) return [];
  return DIAGNOSTIC_FIELDS.filter(({ key }) => {
    const value = diagnostics[key];
    if (value === null || value === undefined) return false;
    if (Array.isArray(value)) return value.length > 0;
    return true;
  });
}

function _dpr() {
  return window.devicePixelRatio || 1;
}

function _imageSpace(ctx, viewport) {
  ctx.setTransform(viewport.zoom, 0, 0, viewport.zoom, viewport.panX, viewport.panY);
}

function _screenSpace(ctx) {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

/**
 * The index of the topmost match whose box contains an image-space point, or -1. Topmost =
 * smallest-area box wins, so a fragment sitting inside a larger match is selectable. Pure,
 * so the per-match verdict click path (main.js) can hit-test without touching the canvas.
 * @param {ReadonlyArray<{box:{x:number,y:number,w:number,h:number}}>} matches
 * @param {number} ix @param {number} iy
 * @returns {number}
 */
export function hitTestMatch(matches, ix, iy) {
  let best = -1;
  let bestArea = Infinity;
  for (let i = 0; i < matches.length; i += 1) {
    const b = matches[i].box;
    if (ix >= b.x && ix < b.x + b.w && iy >= b.y && iy < b.y + b.h) {
      const area = b.w * b.h;
      if (area < bestArea) {
        bestArea = area;
        best = i;
      }
    }
  }
  return best;
}

/** Draw a fixed-size label with a legible backing plate, at a backing-store position. */
function _label(ctx, text, bx, by, color) {
  const dpr = _dpr();
  ctx.font = `${12 * dpr}px system-ui, sans-serif`;
  ctx.textBaseline = "bottom";
  const padding = 3 * dpr;
  const width = ctx.measureText(text).width;
  const height = 14 * dpr;
  const top = Math.max(0, by - height);
  ctx.fillStyle = "rgba(9, 13, 18, 0.82)";
  ctx.fillRect(bx, top, width + padding * 2, height);
  ctx.fillStyle = color;
  ctx.fillText(text, bx + padding, top + height - padding);
}

/**
 * Draw the query/exemplar box the user drew (distinct dashed gold), so the exemplar is always
 * visible even before results return.
 * @param {import("./viewport.js").Viewport} viewport
 * @param {{x0:number,y0:number,x1:number,y1:number}} box half-open image-space corners
 */
export function drawQueryBox(viewport, box) {
  const ctx = viewport.canvas.getContext("2d");
  if (!ctx) return;
  _imageSpace(ctx, viewport);
  ctx.setLineDash([6 / viewport.zoom, 4 / viewport.zoom]);
  ctx.lineWidth = (1.5 * _dpr()) / viewport.zoom;
  ctx.strokeStyle = COL_EXEMPLAR;
  ctx.strokeRect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0);
  ctx.setLineDash([]);
  _screenSpace(ctx);
}

/**
 * Draw the result match boxes (UI-03): each with its score, the exemplar self-match drawn
 * distinctly, and any human-marked-wrong box in red. Boxes are stroked in image space so they
 * track zoom/pan; scores are drawn at a fixed size in backing space.
 * @param {import("./viewport.js").Viewport} viewport
 * @param {ReadonlyArray<{box:{x:number,y:number,w:number,h:number}, score:number, is_exemplar?:boolean}>} matches
 * @param {{wrongSet?:Set<number>}} [opts]
 */
export function drawResults(viewport, matches, opts = {}) {
  const ctx = viewport.canvas.getContext("2d");
  if (!ctx) return;
  const wrongSet = opts.wrongSet || new Set();
  const dpr = _dpr();

  // Boxes in image space.
  _imageSpace(ctx, viewport);
  for (let i = 0; i < matches.length; i += 1) {
    const m = matches[i];
    const wrong = wrongSet.has(i);
    ctx.lineWidth = ((wrong ? 2.5 : 1.5) * dpr) / viewport.zoom;
    ctx.strokeStyle = wrong ? COL_WRONG : m.is_exemplar ? COL_EXEMPLAR : COL_MATCH;
    ctx.strokeRect(m.box.x, m.box.y, m.box.w, m.box.h);
  }

  // Labels in backing space, fixed size.
  _screenSpace(ctx);
  for (let i = 0; i < matches.length; i += 1) {
    const m = matches[i];
    const wrong = wrongSet.has(i);
    const at = viewport.imageToBacking(m.box.x, m.box.y);
    const tag = m.is_exemplar ? "self" : m.score.toFixed(3);
    const text = wrong ? `✗ ${tag}` : tag;
    _label(ctx, text, at.x, at.y, wrong ? COL_WRONG : m.is_exemplar ? COL_EXEMPLAR : COL_MATCH);
  }
}

/**
 * Draw the similarity heatmap under the boxes: the decoded PNG stretched across the whole
 * scene in image space, at partial alpha. `heatmapImg` is decoded by the caller (main.js) so
 * this stays synchronous.
 * @param {import("./viewport.js").Viewport} viewport
 * @param {CanvasImageSource} heatmapImg
 * @param {number} [alpha]
 */
export function drawHeatmap(viewport, heatmapImg, alpha = 0.55) {
  const ctx = viewport.canvas.getContext("2d");
  if (!ctx || !heatmapImg) return;
  _imageSpace(ctx, viewport);
  const prevAlpha = ctx.globalAlpha;
  const prevSmoothing = ctx.imageSmoothingEnabled;
  ctx.globalAlpha = alpha;
  ctx.imageSmoothingEnabled = true; // the map is low-res and upscaled; smoothing reads better
  ctx.drawImage(heatmapImg, 0, 0, viewport.naturalW, viewport.naturalH);
  ctx.globalAlpha = prevAlpha;
  ctx.imageSmoothingEnabled = prevSmoothing;
  _screenSpace(ctx);
}

/** Filled dot at an image-space point, fixed radius in backing space. */
function _dot(ctx, viewport, ix, iy, color, radius) {
  const at = viewport.imageToBacking(ix, iy);
  ctx.beginPath();
  ctx.arc(at.x, at.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

/** Draw a fixed-length arrow (backing space) from (bx,by) along the unit vector (dx,dy). */
function _arrow(ctx, bx, by, dx, dy, length, head) {
  const dpr = _dpr();
  const tx = bx + dx * length;
  const ty = by + dy * length;
  ctx.setLineDash([]);
  ctx.lineWidth = 2 * dpr;
  ctx.strokeStyle = COL_MARKER;
  ctx.beginPath();
  ctx.moveTo(bx, by);
  ctx.lineTo(tx, ty);
  ctx.stroke();
  // Two barbs swept back from the tip so the arrow clearly points along the direction.
  const angle = Math.atan2(dy, dx);
  for (const offset of [Math.PI * 0.83, -Math.PI * 0.83]) {
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx + head * Math.cos(angle + offset), ty + head * Math.sin(angle + offset));
    ctx.stroke();
  }
}

/**
 * Draw the marker layer: each found marker's box (image space), a fixed-length pointing arrow
 * from its reference point (backing space, only where a direction exists), and a thin connector
 * to the proposal chosen for it. Presence-driven — the only knowledge used is the shape of the
 * marker diagnostics fields, never the identity of the exploration that produced them.
 * @param {import("./viewport.js").Viewport} viewport
 * @param {object} diagnostics carrying markers[], marker_reference_points[], marker_directions[]
 * @param {ReadonlyArray<{box:{x:number,y:number,w:number,h:number}}>} matches chosen proposals,
 *   one per marker in index order
 */
function drawMarkerDiagnostics(viewport, diagnostics, matches) {
  const ctx = viewport.canvas.getContext("2d");
  if (!ctx) return;
  const dpr = _dpr();
  const markers = diagnostics.markers || [];
  const refs = diagnostics.marker_reference_points || [];
  const dirs = diagnostics.marker_directions || [];

  // 1. Marker boxes in image space, so they track zoom and pan.
  _imageSpace(ctx, viewport);
  ctx.setLineDash([]);
  ctx.lineWidth = (1.5 * dpr) / viewport.zoom;
  ctx.strokeStyle = COL_MARKER;
  for (const b of markers) {
    ctx.strokeRect(b.x, b.y, b.w, b.h);
  }
  _screenSpace(ctx);

  // 2. Per marker: the connector to its chosen proposal, the reference-point dot, then the
  //    pointing arrow — all fixed-size in backing space so they read the same at any zoom.
  for (let i = 0; i < refs.length; i += 1) {
    const ref = refs[i];
    if (!ref) continue;
    const from = viewport.imageToBacking(ref.x, ref.y);

    const chosen = matches[i] && matches[i].box;
    if (chosen) {
      const to = viewport.imageToBacking(chosen.x + chosen.w / 2, chosen.y + chosen.h / 2);
      ctx.setLineDash([4 * dpr, 3 * dpr]);
      ctx.lineWidth = dpr;
      ctx.strokeStyle = COL_MARKER_LINK;
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    _dot(ctx, viewport, ref.x, ref.y, COL_MARKER, 3 * dpr);

    // A direction exists only for a pointing marker; a symmetric one draws no arrow (by design).
    const dir = dirs[i];
    if (dir) {
      _arrow(ctx, from.x, from.y, dir[0], dir[1], MARKER_ARROW_LEN * dpr, MARKER_ARROW_HEAD * dpr);
    }
  }
}

/**
 * Draw the point/vector diagnostics that sit over the boxes — keypoints, correspondences,
 * Hough peaks, proposals — each gated by its toggle and by field presence. The heatmap is
 * handled separately (`drawHeatmap`) because it draws under the boxes.
 * @param {import("./viewport.js").Viewport} viewport
 * @param {object} diagnostics
 * @param {Set<string>} enabled the field keys currently toggled on
 * @param {ReadonlyArray<{box:{x:number,y:number,w:number,h:number}}>} [matches] chosen proposals,
 *   one per marker, used to draw the marker→proposal connector when marker diagnostics are on
 */
export function drawPointDiagnostics(viewport, diagnostics, enabled, matches = []) {
  const ctx = viewport.canvas.getContext("2d");
  if (!ctx || !diagnostics) return;
  const dpr = _dpr();

  if (enabled.has("markers") && Array.isArray(diagnostics.markers)) {
    drawMarkerDiagnostics(viewport, diagnostics, matches);
  }

  if (enabled.has("proposals") && Array.isArray(diagnostics.proposals)) {
    _imageSpace(ctx, viewport);
    ctx.setLineDash([3 / viewport.zoom, 3 / viewport.zoom]);
    ctx.lineWidth = dpr / viewport.zoom;
    ctx.strokeStyle = COL_PROPOSAL;
    for (const b of diagnostics.proposals) {
      ctx.strokeRect(b.x, b.y, b.w, b.h);
    }
    ctx.setLineDash([]);
    _screenSpace(ctx);
  }

  if (enabled.has("correspondences") && Array.isArray(diagnostics.correspondences)) {
    _screenSpace(ctx);
    ctx.lineWidth = dpr;
    ctx.strokeStyle = COL_CORR;
    for (const c of diagnostics.correspondences) {
      const a = viewport.imageToBacking(c.src.x, c.src.y);
      const b = viewport.imageToBacking(c.dst.x, c.dst.y);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      _dot(ctx, viewport, c.dst.x, c.dst.y, COL_CORR, 2.5 * dpr);
    }
  }

  if (enabled.has("keypoints") && Array.isArray(diagnostics.keypoints)) {
    _screenSpace(ctx);
    for (const p of diagnostics.keypoints) {
      _dot(ctx, viewport, p.x, p.y, COL_KEYPOINT, 2.5 * dpr);
    }
  }

  if (enabled.has("hough_peaks") && Array.isArray(diagnostics.hough_peaks)) {
    // A peak is a pose vote (translation dx/dy in scene px, plus scale/rotation). It is not
    // an image location, so it is drawn as a labelled marker at (dx, dy) sized by its vote
    // weight — a legible, honest depiction of vote-space without pretending it is a box.
    _screenSpace(ctx);
    const maxVotes = Math.max(...diagnostics.hough_peaks.map((p) => p.votes || 0), 1);
    for (const peak of diagnostics.hough_peaks) {
      const at = viewport.imageToBacking(peak.dx, peak.dy);
      const radius = (3 + 7 * ((peak.votes || 0) / maxVotes)) * dpr;
      ctx.beginPath();
      ctx.arc(at.x, at.y, radius, 0, Math.PI * 2);
      ctx.strokeStyle = COL_HOUGH;
      ctx.lineWidth = 1.5 * dpr;
      ctx.stroke();
    }
  }
}
