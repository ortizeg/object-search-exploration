// payload.js — build the POST /ratings request body from the rating widget's state. PURE.
//
// This is the single file the whole null-vs-zero discipline (UI-08 / EVAL-17) rests on. The
// rule, restated so no one has to go find it:
//
//   `null`  means "not assessed"    -> the field is OMITTED from the body (Rating defaults it
//                                       to None, which the store writes as SQL NULL).
//   `0`     means "assessed, none"  -> the field is sent as an explicit 0.
//
// Collapsing the two is the highest-risk bug in the phase: if an untouched count is sent as 0,
// every unreviewed run silently claims perfect precision and recall and the scoreboard fills
// with fabricated 100% scores. Therefore:
//
//   * NO or-zero and NO nullish-zero coercion appears anywhere in this file — that one
//     operator is the bug, and a grep for either form is part of the phase's verification
//     (which is why this note spells the operators out in words, not symbols).
//   * An unparseable or empty count returns `null` (not assessed), never a defaulted 0.
//   * Per-match verdicts ship ONLY after an explicit confirm, and only then is
//     `verdicts_confirmed` set — an untouched grid is "not assessed", not "every box correct".
//
// Keeping this a pure function of a plain state object is deliberate: the frontend has no JS
// test runner, so the contract is proven from the Python side (tests/test_rating_contract.py)
// by POSTing the exact bodies this builds to the real /ratings endpoint.

/**
 * Parse a raw count-input value into an integer count or `null` ("not assessed").
 *
 * Empty, whitespace, non-numeric, and negative inputs all become `null` — the "not assessed"
 * signal — rather than being coerced to 0. An explicit "0" parses to the integer 0, which is
 * the distinct "assessed, none" signal. This function must never turn absence into a number.
 *
 * @param {string|number|null|undefined} raw
 * @returns {number|null}
 */
export function parseCount(raw) {
  if (raw === null || raw === undefined) return null;
  const trimmed = String(raw).trim();
  if (trimmed === "") return null; // untouched field -> not assessed
  if (!/^\d+$/.test(trimmed)) return null; // non-negative integers only; anything else = not assessed
  return Number.parseInt(trimmed, 10); // "0" -> 0 (assessed, none); never defaulted
}

/**
 * Build the /ratings body from the widget state.
 *
 * @param {{
 *   runId: number,
 *   thumbsUp: boolean|null,
 *   unratable?: boolean,
 *   wrongMode: ("per_match"|"count"|null),
 *   wrongCount?: string|number|null,
 *   missedCount?: string|number|null,
 *   matchCount?: number,
 *   wrongIndices?: Set<number>|Array<number>,
 *   verdictsConfirmed?: boolean,
 *   note?: string,
 * }} state
 * @returns {object} the request body — count fields present only when actually assessed
 */
export function buildRatingPayload(state) {
  // thumbs_up is Tier 0 and always required; the widget gates submission on it. `=== true`
  // is an explicit boolean read, not a coercion of a missing value into a count.
  const body = { run_id: state.runId, thumbs_up: state.thumbsUp === true };

  if (state.unratable === true) {
    body.unratable = true;
  }

  // Recall: a missed_count is included only when assessed (parseCount returns non-null).
  const missed = parseCount(state.missedCount);
  if (missed !== null) {
    body.missed_count = missed;
  }

  // Precision: per-match verdicts and a bare wrong_count are mutually exclusive modes.
  if (state.wrongMode === "count") {
    const wrong = parseCount(state.wrongCount);
    if (wrong !== null) {
      body.wrong_count = wrong;
    }
  } else if (state.wrongMode === "per_match") {
    // Unconfirmed verdicts are "not assessed" and ship nothing (UI-08). Only an explicit
    // confirm turns the grid into evidence and sets verdicts_confirmed.
    if (state.verdictsConfirmed === true) {
      const total = Number.isInteger(state.matchCount) ? state.matchCount : 0;
      const wrong =
        state.wrongIndices instanceof Set
          ? state.wrongIndices
          : new Set(state.wrongIndices || []);
      body.verdicts_confirmed = true;
      body.per_match_verdicts = Array.from({ length: total }, (_unused, index) => ({
        match_index: index,
        correct: !wrong.has(index),
      }));
    }
  }

  const note = typeof state.note === "string" ? state.note.trim() : "";
  if (note !== "") {
    body.note = note;
  }

  return body;
}
