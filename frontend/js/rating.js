// rating.js — the tiered rating widget (UI-05, UI-08, EVAL-16).
//
// Tiers, each optional except Tier 0:
//   Tier 0  thumbs up / down          — required, one click, the only signal that accumulates.
//   Tier 1  wrong matches (precision) — TWO mutually exclusive modes: per-match verdicts (click
//                                        the wrong boxes on the overlay) OR a bare wrong_count.
//                                        Choosing one visibly disables the other.
//   Tier 2  missed instances (recall) — a bare missed_count.
//   plus    an explicit unratable/skip, and a free-text note. No 1–5 star scale, ever.
//
// The null discipline (UI-08) is enforced here and in payload.js together:
//   * count <input>s are created with value="" — EMPTY, never "0".
//   * the "all correct" / "none missed" buttons write an explicit "0".
//   * per-match verdicts start unmarked and count as assessed only after "Confirm verdicts"
//     sets verdicts_confirmed; editing after confirming re-opens the assessment.
//
// The widget owns its own DOM state (thumbs, mode, counts, note, confirmed) but the set of
// per-match "wrong" indices lives in the shared canvas state (passed in as `wrongSet`) so the
// overlay and the widget cannot disagree about which boxes are marked.

import { buildRatingPayload } from "./payload.js";
import { postRating } from "./api.js";

/**
 * Mount the rating widget into a host element for one run.
 *
 * @param {HTMLElement} host container to fill (cleared first)
 * @param {{
 *   runId: number,
 *   matches: ReadonlyArray<{score:number, is_exemplar?:boolean}>,
 *   wrongSet: Set<number>,
 *   setVerdictMode: (on:boolean)=>void,
 *   requestRender: ()=>void,
 *   onSubmitted: (ratingId:number)=>void,
 * }} deps
 * @returns {{toggleVerdict:(index:number)=>void}}
 */
export function mountRating(host, deps) {
  const { runId, matches, wrongSet, setVerdictMode, requestRender, onSubmitted } = deps;
  const matchCount = matches.length;

  host.replaceChildren();
  wrongSet.clear();

  // Widget-owned state.
  let thumbsUp = /** @type {boolean|null} */ (null);
  let wrongMode = /** @type {"per_match"|"count"|null} */ (null);
  let verdictsConfirmed = false;

  const root = el("div", "rating");

  // --- Tier 0: thumbs -----------------------------------------------------------------
  const thumbTier = tier("Overall — required");
  const thumbsRow = el("div", "thumbs");
  const upBtn = button("👍 Useful", "thumb-btn");
  const downBtn = button("👎 Not useful", "thumb-btn");
  upBtn.addEventListener("click", () => setThumb(true));
  downBtn.addEventListener("click", () => setThumb(false));
  thumbsRow.append(upBtn, downBtn);
  thumbTier.append(thumbsRow);

  function setThumb(value) {
    thumbsUp = value;
    upBtn.classList.toggle("active", value === true);
    downBtn.classList.toggle("active", value === false);
    clearMessage();
  }

  // --- Tier 1: wrong matches (precision), two mutually exclusive modes -----------------
  const wrongTier = tier("Wrong matches — precision");
  const modeRow = el("div", "wrong-mode");
  const perMatchRadio = radio("wrong-mode", "per_match", "Mark wrong boxes on the image");
  const countRadio = radio("wrong-mode", "count", "Just a count");
  if (matchCount === 0) {
    perMatchRadio.input.disabled = true;
    perMatchRadio.label.title = "No matches to mark — use a count.";
  }
  modeRow.append(perMatchRadio.label, countRadio.label);

  // Per-match panel: the clickable match list + the explicit confirm.
  const verdictPanel = el("div", "verdict-panel");
  verdictPanel.hidden = true;
  const matchList = el("ol", "match-list");
  const rowByIndex = /** @type {HTMLLIElement[]} */ ([]);
  for (let i = 0; i < matchCount; i += 1) {
    const li = /** @type {HTMLLIElement} */ (document.createElement("li"));
    li.className = "match-row";
    const tag = matches[i].is_exemplar ? "self" : matches[i].score.toFixed(3);
    li.textContent = `#${i} · ${tag}`;
    li.addEventListener("click", () => toggleVerdict(i));
    matchList.appendChild(li);
    rowByIndex.push(li);
  }
  const confirmBtn = button("Confirm verdicts", "confirm-verdicts");
  const verdictStatus = el("small", "verdict-status");
  confirmBtn.addEventListener("click", () => {
    verdictsConfirmed = true;
    setVerdictStatus();
    clearMessage();
  });
  verdictPanel.append(matchList, confirmBtn, verdictStatus);

  // Count panel: the bare wrong_count + the explicit-zero button.
  const countPanel = el("div", "count-panel");
  countPanel.hidden = true;
  const wrongCount = numberInput("wrong-count"); // value="" — EMPTY, never "0"
  const wrongLabel = fieldLabel("Wrong count", wrongCount);
  const allCorrectBtn = button("All correct (0)", "all-correct");
  allCorrectBtn.addEventListener("click", () => {
    wrongCount.value = "0"; // an explicit, assessed zero
    clearMessage();
  });
  countPanel.append(wrongLabel, allCorrectBtn);

  perMatchRadio.input.addEventListener("change", () => setWrongMode("per_match"));
  countRadio.input.addEventListener("change", () => setWrongMode("count"));
  wrongTier.append(modeRow, verdictPanel, countPanel);

  function setWrongMode(mode) {
    wrongMode = mode;
    const perMatch = mode === "per_match";
    verdictPanel.hidden = !perMatch;
    countPanel.hidden = perMatch;
    // Mutual exclusivity: the inactive mode's inputs are disabled, not just hidden.
    wrongCount.disabled = perMatch;
    allCorrectBtn.disabled = perMatch;
    confirmBtn.disabled = !perMatch;
    if (perMatch) {
      setVerdictMode(true); // canvas left-clicks now mark boxes
      setVerdictStatus();
    } else {
      verdictsConfirmed = false;
      wrongSet.clear();
      for (const li of rowByIndex) li.classList.remove("wrong");
      setVerdictMode(false);
      requestRender();
    }
    clearMessage();
  }

  function setVerdictStatus() {
    if (verdictsConfirmed) {
      verdictStatus.textContent = `Confirmed: ${wrongSet.size} wrong of ${matchCount}.`;
      verdictStatus.classList.add("confirmed");
    } else {
      verdictStatus.textContent =
        "Not confirmed — unmarked boxes stay “not assessed”, not “correct”.";
      verdictStatus.classList.remove("confirmed");
    }
  }

  /** Toggle a match's wrong verdict (from a list-row click or a canvas click). */
  function toggleVerdict(index) {
    if (wrongMode !== "per_match" || index < 0 || index >= matchCount) return;
    if (wrongSet.has(index)) wrongSet.delete(index);
    else wrongSet.add(index);
    rowByIndex[index]?.classList.toggle("wrong", wrongSet.has(index));
    // Editing after a confirm re-opens the assessment — the confirmed set changed.
    verdictsConfirmed = false;
    setVerdictStatus();
    requestRender();
  }

  // --- Tier 2: missed instances (recall) ----------------------------------------------
  const missedTier = tier("Missed instances — recall");
  const missedCount = numberInput("missed-count"); // value="" — EMPTY, never "0"
  const missedLabel = fieldLabel("Missed count", missedCount);
  const noneMissedBtn = button("None missed (0)", "none-missed");
  noneMissedBtn.addEventListener("click", () => {
    missedCount.value = "0"; // an explicit, assessed zero
    clearMessage();
  });
  missedTier.append(missedLabel, noneMissedBtn);

  // --- Unratable / skip + note --------------------------------------------------------
  const extraTier = tier("Other");
  const unratable = checkbox("unratable-box", "Unratable / skip this one");
  const note = /** @type {HTMLTextAreaElement} */ (document.createElement("textarea"));
  note.className = "note";
  note.rows = 2;
  note.placeholder = "Optional note…";
  const noteLabel = fieldLabel("Note", note);
  extraTier.append(unratable.label, noteLabel);

  // --- EVAL-16 convention, shown next to the widget -----------------------------------
  const convention = el("p", "eval16");
  convention.textContent =
    "Convention (EVAL-16): two boxes on one instance = 1 TP + 1 FP. Count each extra box on the same object as wrong.";

  // --- Submit -------------------------------------------------------------------------
  const submitBtn = button("Submit rating", "submit-rating");
  const message = el("p", "rating-msg");
  message.setAttribute("role", "status");
  submitBtn.addEventListener("click", () => void submit());

  function clearMessage() {
    message.textContent = "";
    message.classList.remove("error", "ok");
  }

  function currentState() {
    return {
      runId,
      thumbsUp,
      unratable: unratable.input.checked,
      wrongMode,
      wrongCount: wrongCount.value,
      missedCount: missedCount.value,
      matchCount,
      wrongIndices: wrongSet,
      verdictsConfirmed,
      note: note.value,
    };
  }

  async function submit() {
    if (thumbsUp === null && !unratable.input.checked) {
      showError("Give a thumbs up/down (or mark it unratable) before submitting.");
      return;
    }
    const body = buildRatingPayload(currentState());
    submitBtn.disabled = true;
    try {
      const { rating_id } = await postRating(body);
      message.textContent = `Saved rating #${rating_id}. It now feeds /stats.`;
      message.classList.add("ok");
      setVerdictMode(false);
      onSubmitted(rating_id);
    } catch (err) {
      submitBtn.disabled = false;
      showError(`Rating rejected: ${err.message}`);
    }
  }

  function showError(text) {
    message.textContent = text;
    message.classList.remove("ok");
    message.classList.add("error");
  }

  root.append(
    thumbTier,
    wrongTier,
    missedTier,
    extraTier,
    convention,
    submitBtn,
    message,
  );
  host.appendChild(root);

  return { toggleVerdict };
}

// --- tiny DOM helpers (readability over a framework) ----------------------------------

function el(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function tier(labelText) {
  const wrap = el("div", "rating-tier");
  const label = el("span", "tier-label");
  label.textContent = labelText;
  wrap.appendChild(label);
  return wrap;
}

function button(text, className) {
  const b = /** @type {HTMLButtonElement} */ (document.createElement("button"));
  b.type = "button";
  b.className = className;
  b.textContent = text;
  return b;
}

function radio(name, value, text) {
  const label = /** @type {HTMLLabelElement} */ (el("label", "radio"));
  const input = /** @type {HTMLInputElement} */ (document.createElement("input"));
  input.type = "radio";
  input.name = name;
  input.value = value;
  const span = document.createElement("span");
  span.textContent = text;
  label.append(input, span);
  return { label, input };
}

function checkbox(className, text) {
  const label = /** @type {HTMLLabelElement} */ (el("label", "checkbox"));
  const input = /** @type {HTMLInputElement} */ (document.createElement("input"));
  input.type = "checkbox";
  input.className = className;
  const span = document.createElement("span");
  span.textContent = text;
  label.append(input, span);
  return { label, input };
}

/** A number input created EMPTY (value="") — UI-08: counts are never prepopulated with 0. */
function numberInput(className) {
  const input = /** @type {HTMLInputElement} */ (document.createElement("input"));
  input.type = "number";
  input.min = "0";
  input.step = "1";
  input.className = className;
  input.value = ""; // explicit: empty means "not assessed", not 0
  return input;
}

function fieldLabel(text, control) {
  const label = /** @type {HTMLLabelElement} */ (el("label", "field"));
  const span = document.createElement("span");
  span.textContent = text;
  label.append(span, control);
  return label;
}
