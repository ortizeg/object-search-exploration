// viewport.js — the three-space coordinate transform, as a pair of exact inverses.
//
// Everything in the canvas UI sits on this file. A box drawn under zoom + pan on a
// high-DPI display must reach the API in *image* pixels, and the overlay the rater judges
// must be stroked from the exact integers the API returned. A sub-pixel error here is a
// whole image pixel at 2x, which silently changes the crop and pollutes every rating with
// an unattributable failure — so the transform is proven before anything is built on it
// (see frontend/dev/selfcheck.html).
//
// Three coordinate spaces exist and conflating any two is the classic bug:
//   - CSS pixels      — what the pointer reports, via getBoundingClientRect(). Fractional.
//   - backing-store px — canvas.width / canvas.height (== cssSize * devicePixelRatio on a
//                        Retina display). The space the 2-D context and zoom/pan live in.
//   - image pixels    — the ONLY space the API may see.
//
// Two measured fixes are the whole point of this module (see .planning/research/PITFALLS.md
// §9.1 and §9.3):
//
//   1. Use event.clientX / clientY minus canvas.getBoundingClientRect(), NEVER the pointer
//      event's element-relative offset coordinates. Those are rounded to an integer
//      (Chromium returns 124 where the true position is 123.5); at high zoom that half-pixel
//      is amplified by the inverse scale into a multi-pixel image-space error. Keep every
//      intermediate a float and round exactly once, in image space, in finalizeBox.
//
//   2. Compute scaleX = canvas.width / rect.width and scaleY = canvas.height / rect.height
//      SEPARATELY. For any canvas whose CSS box aspect ratio differs from its backing store
//      — which every responsive canvas does — the two ratios differ, and one shared scalar
//      skews the box: correct in the centre, progressively wrong toward an edge.
//
// The two directions are written next to each other and are exact inverses. `zoom` is in
// backing-store px per image px; `panX` / `panY` are the backing-store offset of image
// origin (0, 0). devicePixelRatio never appears in this math — it is absorbed entirely by
// scaleX / scaleY (backing store already carries it) — which is exactly what keeps the
// inverse clean.

/**
 * @typedef {Object} TransformView
 * @property {{left:number, top:number, width:number, height:number}} rect
 *   The canvas getBoundingClientRect() (CSS px, fractional). Re-measured every event.
 * @property {number} canvasWidth  Backing-store width  (canvas.width attribute).
 * @property {number} canvasHeight Backing-store height (canvas.height attribute).
 * @property {number} zoom  Backing-store px per image px.
 * @property {number} panX  Backing-store x of image origin.
 * @property {number} panY  Backing-store y of image origin.
 * @property {number} naturalW Image natural width  (for clamping).
 * @property {number} naturalH Image natural height (for clamping).
 * @property {boolean} [clamp] Clamp the image point to [0, natural]. Default true.
 */

/**
 * Pointer (client CSS px) -> image px. Returns floats; NEVER rounds.
 * @param {number} clientX event.clientX (never the element-relative offset coordinate).
 * @param {number} clientY event.clientY (never the element-relative offset coordinate).
 * @param {TransformView} view
 * @returns {{x:number, y:number}}
 */
export function screenToImage(clientX, clientY, view) {
  const { rect, canvasWidth, canvasHeight, zoom, panX, panY } = view;
  // scaleX and scaleY are computed separately — they differ for any non-square-pixel canvas.
  const scaleX = canvasWidth / rect.width;
  const scaleY = canvasHeight / rect.height;
  // clientX minus the freshly measured rect — never the integer-rounded offset coordinate.
  const backingX = (clientX - rect.left) * scaleX;
  const backingY = (clientY - rect.top) * scaleY;
  let x = (backingX - panX) / zoom;
  let y = (backingY - panY) / zoom;
  if (view.clamp !== false) {
    x = Math.min(Math.max(x, 0), view.naturalW);
    y = Math.min(Math.max(y, 0), view.naturalH);
  }
  return { x, y };
}

/**
 * Image px -> pointer (client CSS px). The exact inverse of screenToImage (modulo the
 * optional clamp, which only bites outside the image).
 * @param {number} ix image x (float).
 * @param {number} iy image y (float).
 * @param {TransformView} view
 * @returns {{x:number, y:number}}
 */
export function imageToScreen(ix, iy, view) {
  const { rect, canvasWidth, canvasHeight, zoom, panX, panY } = view;
  const scaleX = canvasWidth / rect.width;
  const scaleY = canvasHeight / rect.height;
  const backingX = ix * zoom + panX;
  const backingY = iy * zoom + panY;
  return {
    x: backingX / scaleX + rect.left,
    y: backingY / scaleY + rect.top,
  };
}

/**
 * Normalise a drag (two image-space corners in any order) into a half-open integer box,
 * rounding exactly once, at the very end. See PITFALLS §9.4:
 *   normalise -> clamp in float -> integerise (floor start, ceil end) -> re-clamp -> reject.
 * Returns null for a below-minimum box rather than silently fixing it (a stray click).
 *
 * @param {{x:number, y:number}} a first drag corner (image px, float)
 * @param {{x:number, y:number}} b second drag corner (image px, float)
 * @param {number} natW image natural width
 * @param {number} natH image natural height
 * @param {number} [minSize] minimum edge length in image px (matches the BBox validator)
 * @returns {{x0:number, y0:number, x1:number, y1:number} | null} half-open [x0,x1) x [y0,y1)
 */
export function finalizeBox(a, b, natW, natH, minSize = 8) {
  // 1) normalise — handles any drag direction.
  let x0 = Math.min(a.x, b.x);
  let x1 = Math.max(a.x, b.x);
  let y0 = Math.min(a.y, b.y);
  let y1 = Math.max(a.y, b.y);
  // 2) clamp in FLOAT space, before rounding.
  x0 = Math.max(0, Math.min(x0, natW));
  x1 = Math.max(0, Math.min(x1, natW));
  y0 = Math.max(0, Math.min(y0, natH));
  y1 = Math.max(0, Math.min(y1, natH));
  // 3) integerise: floor the inclusive start, ceil the exclusive end (grow outward).
  x0 = Math.floor(x0);
  y0 = Math.floor(y0);
  x1 = Math.ceil(x1);
  y1 = Math.ceil(y1);
  // 4) re-clamp (ceil can overshoot) and reject rather than silently fix.
  x1 = Math.min(x1, natW);
  y1 = Math.min(y1, natH);
  if (x1 - x0 < minSize || y1 - y0 < minSize) return null;
  return { x0, y0, x1, y1 };
}

/**
 * A live canvas viewport: owns the zoom/pan state and the image, and reads the canvas's
 * current backing-store size and bounding rect on every transform so a scroll, resize, or
 * DPR change mid-drag can never desynchronise the coordinates.
 */
export class Viewport {
  /** @param {HTMLCanvasElement} canvas */
  constructor(canvas) {
    this.canvas = canvas;
    /** @type {HTMLImageElement | ImageBitmap | null} */
    this.image = null;
    this.naturalW = 0;
    this.naturalH = 0;
    this.zoom = 1; // backing-store px per image px
    this.panX = 0; // backing-store x of image origin
    this.panY = 0;
  }

  /** Build a TransformView from the live canvas — re-measured every call (PITFALLS §9.6). */
  view(clamp = true) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      rect,
      canvasWidth: this.canvas.width,
      canvasHeight: this.canvas.height,
      zoom: this.zoom,
      panX: this.panX,
      panY: this.panY,
      naturalW: this.naturalW,
      naturalH: this.naturalH,
      clamp,
    };
  }

  /** Pointer event -> image px (clamped). */
  screenToImage(clientX, clientY) {
    return screenToImage(clientX, clientY, this.view(true));
  }

  /** Image px -> pointer (client CSS px). */
  imageToScreen(ix, iy) {
    return imageToScreen(ix, iy, this.view(false));
  }

  /**
   * Set the current image and reset the fit. Records natural size for clamping.
   * @param {HTMLImageElement | ImageBitmap} image
   * @param {number} naturalW
   * @param {number} naturalH
   */
  setImage(image, naturalW, naturalH) {
    this.image = image;
    this.naturalW = naturalW;
    this.naturalH = naturalH;
    this.syncCanvasSize();
    this.fitContain();
  }

  /**
   * Match the backing store to the CSS box at the current devicePixelRatio. Assigning to
   * canvas.width clears the bitmap and the context transform, so it is guarded to only fire
   * when the size actually changed (PITFALLS §9.1).
   */
  syncCanvasSize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const bw = Math.round(rect.width * dpr);
    const bh = Math.round(rect.height * dpr);
    if (bw > 0 && bh > 0 && (this.canvas.width !== bw || this.canvas.height !== bh)) {
      this.canvas.width = bw;
      this.canvas.height = bh;
    }
  }

  /** Fit the whole image into the canvas (letterbox = pan). PITFALLS §9.7. */
  fitContain() {
    if (!this.naturalW || !this.naturalH) return;
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    this.zoom = Math.min(cw / this.naturalW, ch / this.naturalH);
    this.panX = (cw - this.naturalW * this.zoom) / 2;
    this.panY = (ch - this.naturalH * this.zoom) / 2;
  }

  /**
   * Multiplicative zoom about a fixed image point, so the pixel under the cursor stays put.
   * @param {number} factor  e.g. 1.1 to zoom in
   * @param {number} clientX anchor (pointer) x
   * @param {number} clientY anchor (pointer) y
   */
  zoomAbout(factor, clientX, clientY) {
    const before = this.screenToImage(clientX, clientY);
    this.zoom *= factor;
    // Solve pan so `before` maps back to the same backing-store point.
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    const backingX = (clientX - rect.left) * scaleX;
    const backingY = (clientY - rect.top) * scaleY;
    this.panX = backingX - before.x * this.zoom;
    this.panY = backingY - before.y * this.zoom;
  }

  /** Pan by a pointer (CSS px) delta, converting to backing-store px. */
  panBy(dxCss, dyCss) {
    const rect = this.canvas.getBoundingClientRect();
    this.panX += dxCss * (this.canvas.width / rect.width);
    this.panY += dyCss * (this.canvas.height / rect.height);
  }

  /**
   * Redraw the image and any overlay boxes. Overlay boxes are given in image px and stroked
   * inside the image-space transform, so the server's integers are used verbatim — no
   * client-side arithmetic between the box the API returned and the box the rater sees.
   *
   * @param {Array<{x:number,y:number,w:number,h:number,color?:string}>} [boxes]
   */
  render(boxes = []) {
    const ctx = this.canvas.getContext("2d");
    if (!ctx) return;
    // Clear in raw backing-store space.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.image) return;
    // One absolute setTransform per frame (never accumulate). Now the context IS image space.
    ctx.setTransform(this.zoom, 0, 0, this.zoom, this.panX, this.panY);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.image, 0, 0);
    for (const box of boxes) {
      ctx.lineWidth = 1 / this.zoom; // 1 backing-store px regardless of zoom
      ctx.strokeStyle = box.color || "#00e5ff";
      ctx.strokeRect(box.x, box.y, box.w, box.h);
    }
  }
}
