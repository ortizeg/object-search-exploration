// stats.js — the per-method scoreboard, rendered with plain SVG (no charting dependency).
//
// The honesty rules the dashboard exists to enforce (EVAL-14 / EVAL-17), all visible at a
// glance:
//
//   * Every rate ships with the `n` it was computed over, inline — a 1/1 must not read like a
//     50/100. The thumbs-up rate is shown with its Wilson interval AND its n, and the interval's
//     meaning is labelled.
//   * Methods are RANKED BY THE WILSON LOWER BOUND, not the raw rate. That is what makes a 1/1
//     (wide interval, low floor) sort below a 50/100 (tight interval, high floor). We re-sort
//     here rather than trust the caller, so the ranking rule lives with the thing that renders it.
//   * `n = 0` renders DISTINCTLY from a genuine wide interval: "no ratings yet", never a bar
//     spanning 0–1. [0,1] means "no information", which is a different claim from "measured, and
//     the interval happens to be wide".
//   * Precision and recall each carry their OWN separate n — they are different subsets (a bare
//     thumbs-up feeds neither; a wrong_count feeds precision but not recall).
//
// The backend already sorts by the Wilson lower bound; the client re-sort is deliberate
// redundancy so this file is correct standalone.

const SVG_NS = "http://www.w3.org/2000/svg";
const BAR_W = 220;
const BAR_H = 10;

/**
 * Render the scoreboard into a host element.
 * @param {HTMLElement} host container to fill (cleared first)
 * @param {ReadonlyArray<object>} stats the /stats payload (one MethodStats per method)
 */
export function renderStats(host, stats) {
  host.replaceChildren();
  if (!stats || stats.length === 0) {
    host.appendChild(muted("No runs yet — search and rate one, then refresh."));
    return;
  }

  // Rank by the Wilson lower bound; a method with no rated runs (null) sorts last.
  const ranked = [...stats].sort((a, b) => wilsonKey(b) - wilsonKey(a));

  ranked.forEach((s, rank) => host.appendChild(methodCard(s, rank + 1)));
}

/** The sort key: the Wilson lower bound, or -1 when there is no rated run (sorts last). */
function wilsonKey(s) {
  return typeof s.thumbs_ci_lower === "number" ? s.thumbs_ci_lower : -1;
}

function methodCard(s, rank) {
  const card = el("div", "stat-card");

  const head = el("div", "stat-head");
  const name = el("span", "stat-method");
  name.textContent = `${rank}. ${s.method}`; // textContent — never innerHTML for the name
  head.appendChild(name);
  card.appendChild(head);

  // --- thumbs-up rate + Wilson interval + n -----------------------------------------
  card.appendChild(label("Thumbs-up rate"));
  if (s.thumbs_n === 0) {
    // n = 0 rendered distinctly: NOT a [0,1] bar. "No information" is not "wide interval".
    const none = el("p", "stat-empty");
    none.textContent = "no ratings yet — no information (an interval here would be the whole [0, 1])";
    card.appendChild(none);
  } else {
    card.appendChild(
      intervalBar(s.thumbs_rate, s.thumbs_ci_lower, s.thumbs_ci_upper),
    );
    const rate = pct(s.thumbs_rate);
    const lo = pct(s.thumbs_ci_lower);
    const hi = pct(s.thumbs_ci_upper);
    card.appendChild(
      caption(
        `${rate} (${s.thumbs_n_up}/${s.thumbs_n}) · 95% Wilson CI [${lo}, ${hi}]`,
      ),
    );
    card.appendChild(muted("Interval = 95% Wilson CI on the thumbs-up rate; ranking uses its lower bound."));
  }

  // --- precision and recall, each with its OWN n ------------------------------------
  card.appendChild(meanRow("Precision", s.precision_mean, s.precision_n));
  card.appendChild(meanRow("Recall", s.recall_mean, s.recall_n));

  // --- outcome + latency ------------------------------------------------------------
  const meta = el("div", "stat-meta");
  meta.appendChild(chip(`abstentions ${s.abstention_count}`));
  meta.appendChild(chip(`errors ${s.error_count}`));
  meta.appendChild(chip(`sweep-eligible ${s.threshold_sweep_eligible_count}`));
  card.appendChild(meta);

  card.appendChild(label("Latency (total ms)"));
  card.appendChild(
    caption(
      `p50 ${ms(s.latency_p50_ms)} · p90 ${ms(s.latency_p90_ms)} · p99 ${ms(s.latency_p99_ms)}`,
    ),
  );

  return card;
}

/** A mean-with-its-own-n row. n=0 renders as "not assessed", never a fabricated 0. */
function meanRow(name, mean, n) {
  const wrap = el("div", "stat-mean");
  wrap.appendChild(label(name));
  if (n === 0 || mean === null || mean === undefined) {
    const empty = el("span", "stat-empty");
    empty.textContent = "— not assessed (n = 0)";
    wrap.appendChild(empty);
  } else {
    wrap.appendChild(meanBar(mean));
    wrap.appendChild(caption(`${pct(mean)} (n = ${n})`));
  }
  return wrap;
}

// --- SVG builders ---------------------------------------------------------------------

/** A 0..1 track with the rate marked and the Wilson interval drawn as a band with end caps. */
function intervalBar(rate, lo, hi) {
  const svg = svgEl("svg", { width: BAR_W, height: BAR_H + 8, class: "stat-svg" });
  const y = 4;
  svg.appendChild(rect(0, y, BAR_W, BAR_H, "stat-track"));
  if (typeof lo === "number" && typeof hi === "number") {
    const x0 = clamp01(lo) * BAR_W;
    const x1 = clamp01(hi) * BAR_W;
    svg.appendChild(rect(x0, y, Math.max(1, x1 - x0), BAR_H, "stat-band"));
    svg.appendChild(line(x0, y - 2, x0, y + BAR_H + 2, "stat-cap"));
    svg.appendChild(line(x1, y - 2, x1, y + BAR_H + 2, "stat-cap"));
  }
  if (typeof rate === "number") {
    const x = clamp01(rate) * BAR_W;
    svg.appendChild(line(x, y - 3, x, y + BAR_H + 3, "stat-mark"));
  }
  return svg;
}

/** A simple 0..1 filled bar for a precision/recall mean. */
function meanBar(mean) {
  const svg = svgEl("svg", { width: BAR_W, height: BAR_H + 4, class: "stat-svg" });
  const y = 2;
  svg.appendChild(rect(0, y, BAR_W, BAR_H, "stat-track"));
  svg.appendChild(rect(0, y, clamp01(mean) * BAR_W, BAR_H, "stat-fill"));
  return svg;
}

function svgEl(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

function rect(x, y, w, h, cls) {
  return svgEl("rect", { x, y, width: Math.max(0, w), height: h, rx: 2, class: cls });
}

function line(x1, y1, x2, y2, cls) {
  return svgEl("line", { x1, y1, x2, y2, class: cls });
}

// --- tiny DOM helpers -----------------------------------------------------------------

function el(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function label(text) {
  const node = el("span", "stat-label");
  node.textContent = text;
  return node;
}

function caption(text) {
  const node = el("p", "stat-caption");
  node.textContent = text;
  return node;
}

function muted(text) {
  const node = el("p", "muted");
  node.textContent = text;
  return node;
}

function chip(text) {
  const node = el("span", "stat-chip");
  node.textContent = text;
  return node;
}

// --- formatting -----------------------------------------------------------------------

function clamp01(v) {
  return Math.min(1, Math.max(0, v));
}

function pct(v) {
  return typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—";
}

function ms(v) {
  return typeof v === "number" ? v.toFixed(1) : "—";
}
