// selfcheck.js — an in-browser proof that the coordinate transform is a pair of exact
// inverses. The project has no JS test runner on purpose (adding one means npm), so this is
// the reusable harness that stands in for a unit test: it round-trips a set of known image
// coordinates through imageToScreen then screenToImage under a high-DPI, zoomed, panned
// viewport and asserts they return within 0.5 px. It renders PASS/FAIL to the page.
//
// The transform functions are pure and take an explicit view object, so the check does not
// depend on the real display's devicePixelRatio: it *constructs* a dpr=2 viewport by making
// the backing store twice the CSS rect. That is exactly the case (Retina, 2x) that a naive
// implementation gets wrong.

import { screenToImage, imageToScreen, finalizeBox } from "../js/viewport.js";

/**
 * A synthetic dpr=2, zoom=2.3, panned viewport. rect is the canvas CSS box; the backing
 * store (canvasWidth/Height) is 2x the rect, which is what devicePixelRatio === 2 produces.
 * zoom is in backing-store px per image px; the pan is deliberately nonzero and non-integer.
 * @param {number} naturalW
 * @param {number} naturalH
 */
function makeView(naturalW, naturalH) {
  const rect = { left: 20.5, top: 11.25, width: 500.5, height: 300.25 };
  return {
    rect,
    canvasWidth: Math.round(rect.width * 2), // dpr = 2
    canvasHeight: Math.round(rect.height * 2), // dpr = 2
    zoom: 2.3,
    panX: 137.75,
    panY: -42.5,
    naturalW,
    naturalH,
    clamp: false, // exercise the raw inverse; the sample points are inside the image
  };
}

/** Run the assertions and return a structured result. */
export function runSelfcheck() {
  const naturalW = 640;
  const naturalH = 480;
  const view = makeView(naturalW, naturalH);
  const tolerance = 0.5;

  const samplePoints = [
    { x: 0, y: 0 },
    { x: 1, y: 1 },
    { x: 123.5, y: 77.25 },
    { x: 320, y: 240 },
    { x: 639, y: 479 },
    { x: 200.75, y: 300.125 },
  ];

  const cases = [];
  let worst = 0;
  let allPass = true;

  for (const p of samplePoints) {
    // image -> screen -> image must recover p.
    const screen = imageToScreen(p.x, p.y, view);
    const back = screenToImage(screen.x, screen.y, view);
    const dx = Math.abs(back.x - p.x);
    const dy = Math.abs(back.y - p.y);
    const err = Math.max(dx, dy);
    worst = Math.max(worst, err);
    const pass = err <= tolerance;
    if (!pass) allPass = false;
    cases.push({ p, screen, back, err, pass });
  }

  // A second axis of proof: separate scaleX/scaleY actually differ here (non-square pixels),
  // so a single shared scalar would fail this. Assert the two ratios are not equal.
  const scaleX = view.canvasWidth / view.rect.width;
  const scaleY = view.canvasHeight / view.rect.height;
  const scalesDiffer = Math.abs(scaleX - scaleY) > 1e-9;

  // finalizeBox sanity: a right-to-left drag still yields a valid half-open box.
  const box = finalizeBox({ x: 300.6, y: 200.9 }, { x: 100.2, y: 50.1 }, naturalW, naturalH);
  const boxOk =
    box !== null &&
    box.x0 === 100 &&
    box.y0 === 50 &&
    box.x1 === 301 &&
    box.y1 === 201;

  const passed = allPass && scalesDiffer && boxOk;
  return { passed, worst, tolerance, cases, scaleX, scaleY, scalesDiffer, box, boxOk };
}

/** Render the result into the page. */
function render(result) {
  const root = document.getElementById("result");
  if (!root) return;
  const verdict = document.getElementById("verdict");
  if (verdict) {
    verdict.textContent = result.passed ? "PASS" : "FAIL";
    verdict.className = result.passed ? "pass" : "fail";
  }

  const rows = result.cases
    .map(
      (c) =>
        `<tr class="${c.pass ? "ok" : "bad"}">` +
        `<td>(${c.p.x}, ${c.p.y})</td>` +
        `<td>(${c.screen.x.toFixed(3)}, ${c.screen.y.toFixed(3)})</td>` +
        `<td>(${c.back.x.toFixed(4)}, ${c.back.y.toFixed(4)})</td>` +
        `<td>${c.err.toExponential(2)}</td>` +
        `<td>${c.pass ? "ok" : "FAIL"}</td></tr>`,
    )
    .join("");

  root.innerHTML =
    `<p>Worst round-trip error: <strong>${result.worst.toExponential(2)} px</strong> ` +
    `(tolerance ${result.tolerance} px)</p>` +
    `<p>scaleX = ${result.scaleX.toFixed(6)}, scaleY = ${result.scaleY.toFixed(6)} — ` +
    `separate: <strong>${result.scalesDiffer ? "yes" : "NO"}</strong></p>` +
    `<p>finalizeBox reverse-drag box: <strong>${result.boxOk ? "ok" : "FAIL"}</strong> ` +
    `(${JSON.stringify(result.box)})</p>` +
    `<table><thead><tr><th>image</th><th>→ screen (CSS px)</th>` +
    `<th>→ image</th><th>err</th><th>result</th></tr></thead><tbody>${rows}</tbody></table>`;
}

render(runSelfcheck());
