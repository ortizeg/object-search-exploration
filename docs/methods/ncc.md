# Method 1 — `ncc` (normalized cross-correlation template matching)

The zero-model baseline. It correlates the exemplar crop against the whole scene with
`cv2.matchTemplate` / `TM_CCOEFF_NORMED`, over a scale pyramid, and turns the response peaks
into boxes. No weights, no training, milliseconds per query.

The module `src/object_search/search/ncc.py` is meant to be read top to bottom; the numbered
steps below match the `# 1.` … `# 9.` comments in `search()` one-for-one (METHOD-11).

## What it is and when it wins

NCC wins when the repeated instances are **near-identical and near the exemplar's scale**: a
tray of the same part, tiles, repeated UI glyphs. There it is genuinely hard to beat and
costs milliseconds with no model. It loses when instances differ in lighting, pose, or scale
beyond the configured pyramid — that is where `dino-dense` should win.

## Algorithm

### 1. Crop the exemplar and guard against a textureless template

Compute the crop's standard deviation **first**. A flat crop makes `TM_CCOEFF_NORMED`
degenerate (`0/0`) and OpenCV returns a **constant map** — `1.0` at every pixel on 4.10,
`0.0` on the pinned 4.13 and on 5.x — never a NaN. Below `std < 1e-6` the method returns
`outcome=EMPTY` with a diagnostics note rather than emitting a wall of confident false
positives. This is mandatory, and the behaviour is pinned by a test so an OpenCV bump is
caught (see PITFALLS §1.1).

### 2. Build the scale pyramid

**Rescale the SCENE, then crop the template from the downscaled scene.** This reverses the
intuitive "resize the template" approach, and the reversal is measured: resizing the template
independently drops the exemplar's own self-match from `1.0000` to `0.3071`, and does so
**non-monotonically**, so it cannot be corrected by a per-level offset. Cropping from the
already-resized scene keeps the self-match at `1.0000` (PITFALLS §1.3). Levels whose scaled
template would be `< 8 px` on a side, or larger than the level image, are skipped.

### 3. Optionally build the rotated-template bank

Default `angles_deg=(0.0,)` — rotation is **off**, because a bank is a large constant-factor
cost (levels × angles correlations) for a case that rarely needs it. When enabled, a rotated
crop leaves fabricated constant corners (up to **half** the template at 45°) that correlate
with any uniform region. The chosen fix is a **warped mask** passed to `matchTemplate`, eroded
by one pixel to kill the interpolated fringe — chosen over inscribing an axis-aligned rectangle
because that would throw away real template pixels near the corners (PITFALLS §1.6).

### 4. Correlate over the FULL scene at every level

`cv2.matchTemplate(scene_level, template_level, cv2.TM_CCOEFF_NORMED)`. **Never** crop the
search region: restricting the extent changes ~73% of the returned floats and shifts the peak
from `0.99999994` to `1.0`, the single biggest threat to "same input ⇒ identical results"
(PITFALLS §1.8). Any NaN in the response is replaced with `-inf` before peak finding so it can
never win an argmax.

### 5. Extract peaks per level (standardised first)

Peaks come from `common.peaks.extract_peaks` with the configured strategy and the template
size at that level. The spurious noise floor of `TM_CCOEFF_NORMED` varies ~15× with template
size (`0.577` at 8×8 vs `0.039` at 128×128), so a naive argmax across levels favours the
smallest template. Each level's map is therefore **z-scored against its own median/MAD** before
peaks are compared across levels; because that transform is monotone, peak locations are
unchanged but the returned z-score is comparable across levels. The raw score is carried
alongside for the threshold and the candidate log (PITFALLS §1.4).

### 6. Map peaks to boxes

The response is `(H−h+1, W−w+1)` and each value is anchored at the template's **top-left**,
not its centre. A response index `(row, col)` maps to a box with top-left `(col, row)` and the
template size — **no centre offset, no ±1**. At level scale `s` the original-image box is
`BBox(x=round(col/s), y=round(row/s), w=round(tw/s), h=round(th/s))` (PITFALLS §1.2).

### 7. Calibrate the threshold

`common.calibration.calibrate` with the configured strategy. Default `self-similarity` cuts at
`self_score × retain_frac`, relative to the exemplar's own ~1.0 self-match — absolute
thresholds do not transfer across images. When `threshold` is set, the `"fixed"` strategy is
used. The calibrator returns its **reasoning**, which becomes an inspectable diagnostics note.

### 8. Split into matches and sub-threshold candidates

The top `max_candidates` peaks (ranked by the cross-level z-score) are kept as `Candidate`s
**with raw scores** regardless of the threshold — that is what makes an offline threshold sweep
possible later (EVAL-08). Peaks whose raw score clears the threshold become `Match`es; a final
cross-level greedy IoU NMS (prioritised by the z-score) removes duplicates. **METHOD-12: every
accepted peak survives — there is no single-best / argmax-only short-circuit.** The peak that
overlaps the exemplar box is labelled `is_exemplar=True` rather than dropped or double-counted.

### 9. Assemble diagnostics and the result

`Diagnostics` carries the level-1.0 response as a base64 PNG heatmap and `metrics` (crop std,
per-level peak counts, chosen threshold, calibration reason). `LatencyBreakdown` is timed
around preprocess / matchTemplate / postprocess (EVAL-11).

## Pre-processing (exact)

- **Colour:** the BGR scene is converted **once** to single-channel grayscale
  (`cv2.COLOR_BGR2GRAY`). Not per-channel colour correlation — a 3-channel `matchTemplate`
  merely *sums* the channel correlations, which is neither more discriminative nor documented
  (PITFALLS §1.7).
- **dtype / layout:** kept **uint8** and made C-contiguous (`np.ascontiguousarray`). It is
  deliberately **not** cast to float32: `matchTemplate` gives different numbers for uint8 vs
  float32 input (max abs diff `1.27e-06`), so the dtype is part of the method's identity
  (PITFALLS §1.8).
- **Normalization:** **none is applied by hand.** `TM_CCOEFF_NORMED` subtracts each window's
  mean and divides by its L2 norm internally, so a separate mean/std normalization would be
  double-counting. There are no ImageNet-style constants anywhere in this method.
- **Search extent:** the **full** scene, always (see step 4).

## Post-processing (exact)

- **Response shape / anchoring:** `(H−h+1, W−w+1)`, top-left anchored — box mapping is
  `x=round(col/s), y=round(row/s), w=round(tw/s), h=round(th/s)` with no centre offset (step 6).
- **Cross-level normalization:** per-level z-score against that level's own median/MAD before
  comparing or suppressing peaks across levels (step 5).
- **Calibration:** `self-similarity` by default, relative to the ~1.0 self-match (step 7).
- **Suppression:** cross-level greedy IoU NMS over the accepted matches, prioritised by the
  cross-level-comparable z-score (step 8).

## Config reference

Generated from `NCCConfig`'s JSON Schema — the same schema that drives the UI form — so it
cannot drift from the code.

| field | default | effect |
| --- | --- | --- |
| `scales` | `[0.75, 0.875, 1.0, 1.15, 1.3]` | Pyramid scale factors. The scene is resized by each factor and the template cropped from that resized scene, which keeps the self-match at 1.0. |
| `angles_deg` | `[0.0]` | Rotation bank in degrees. Default `(0.0,)` — rotation is off because it is a large constant-factor cost (levels × angles correlations) rarely needed here. |
| `threshold` | `null` | Fixed accept threshold on the raw NCC score. `null` ⇒ use the calibrator. |
| `calibration` | `"self-similarity"` | How the accept threshold is chosen when `threshold` is `null`. self-similarity cuts relative to the exemplar's own ~1.0 self-match; recommended for NCC because absolute thresholds do not transfer across images. |
| `peaks` | `"local-max"` | Peak-extraction strategy. local-max (default) separates touching instances that plain nms merges; nms is the control; watershed uses a distance transform. |
| `nms_iou` | `0.3` | IoU above which two accepted boxes are suppressed to one (cross-level NMS). |
| `suppression_radius_frac` | `0.5` | local-max footprint as a fraction of the template size (size-aware). |
| `max_candidates` | `50` | How many top peaks (with raw scores) to keep for the EVAL-08 candidate log. |
| `seed` | `0` | `random_state` for the gmm calibrator (its only genuinely stochastic step). |
| `retain_frac` | `0.7` | self-similarity accepts scores above `self_score × retain_frac`. |

## Known failure modes

- **Textureless crop.** A flat exemplar makes `TM_CCOEFF_NORMED` return a constant map — `1.0`
  everywhere on OpenCV 4.10, `0.0` on 4.13/5.x — never a NaN (PITFALLS §1.1). The step-1 guard
  abstains with `outcome=EMPTY`. The committed `lattice-touching` sample shows this: its
  instances are solid same-colour rectangles with no internal texture, so NCC honestly returns
  nothing rather than a false-positive wall.
- **Rotation / scale beyond the configured banks.** Instances rotated past `angles_deg` or
  scaled past `scales` are missed; that is where `dino-dense` should win.
- **Lighting / pose change.** NCC correlates raw intensities, so an instance under different
  lighting scores low even when a human sees the same object.
- **Cross-level noise-floor bias.** Mitigated by the per-level z-score (step 5); called out so
  a future editor does not "simplify" it away.

## ROBUSTNESS BACKLOG

Deferred deliberately (mirrored verbatim from the module docstring and
`docs/ROBUSTNESS-BACKLOG.md`); none is built in this phase:

- **FFT-based correlation for large templates.** The spatial `matchTemplate` is O(H·W·h·w); a
  single full-scene FFT cross-correlation is O(H·W·log(H·W)) and wins decisively once the
  template is large.
- **Log-polar / Fourier-Mellin registration** for joint rotation+scale invariance in one
  correlation, replacing the brute-force rotated-template × pyramid bank.
- **Discriminative correlation filters (MOSSE/KCF)** trained on the single exemplar crop, so
  the filter learns to suppress background instead of correlating raw pixels.

## Sample runs

Regenerated by `pixi run samples` and committed under [`docs/samples/ncc/`](../samples/ncc/)
(see its [`index.md`](../samples/ncc/index.md) for the per-image outcome table).

| image | panel |
| --- | --- |
| `lattice-plain` — all 12 identical instances found | ![lattice-plain](../samples/ncc/lattice-plain.png) |
| `lattice-touching` — solid rectangles, textureless ⇒ EMPTY (the step-1 guard) | ![lattice-touching](../samples/ncc/lattice-touching.png) |
| `scatter-scaled` — scale + rotation variation, NCC finds a subset | ![scatter-scaled](../samples/ncc/scatter-scaled.png) |
| `cluttered-distractors` — clutter + distractors | ![cluttered-distractors](../samples/ncc/cluttered-distractors.png) |

## Pseudocode

**Method ① NCC** — first of the four *implemented* methods. The implementation numbering is
①–④ (`ncc`, `sparse-geo`, `dino-dense`, `propose-retrieve`); the source-research numbering is
1, 2, 3, 5 — research Methods 4 and 6 were deferred. The steps below mirror the `# 1.` … `# 9.`
comments in `search()` (METHOD-11); read `src/object_search/search/ncc.py` for the ground truth.

```
1. crop <- exemplar region of scene_gray
   if std(crop) < 1e-6:                       # flat-template guard (mandatory)
       return EMPTY with a diagnostics note   # TM_CCOEFF_NORMED degenerates on a flat crop

2. for s in scales:                           # build the scale pyramid
       scene_s    <- resize(scene_gray, s)        # rescale the SCENE ...
       template_s <- crop template FROM scene_s   # ... then crop the template from it
       skip level if template_s side < 8 px or larger than scene_s

3. optionally build a rotated-template bank   # default angles_deg=[0] => rotation OFF
   (each rotated crop carries a warped mask, eroded 1 px, passed to matchTemplate)

4. for each level:
       resp <- matchTemplate(scene_s, template_s, TM_CCOEFF_NORMED)  # FULL scene, never cropped
       replace any NaN in resp with -inf      # so it can never win an argmax

5. z-score resp against its OWN median/MAD, then extract peaks
   (monotone transform: peak locations unchanged, scores comparable across levels)

6. for each peak (row, col) at level scale s:  # response is (H-h+1, W-w+1), top-left anchored
       box <- BBox(x=round(col/s), y=round(row/s), w=round(tw/s), h=round(th/s))  # no centre offset

7. calibrate the accept threshold             # self-similarity: cut at self_score * retain_frac
   (or "fixed" when config.threshold is set)

8. keep top max_candidates peaks as Candidates WITH raw scores (for the EVAL-08 sweep)
   peaks whose raw score >= threshold -> Matches
   cross-level greedy IoU NMS (nms_iou), prioritised by the z-score  # METHOD-12: no argmax short-circuit
   label the peak overlapping the exemplar box is_exemplar=True

9. assemble Diagnostics (level-1.0 heatmap + metrics) and LatencyBreakdown; return SearchResult
```

## References

- OpenCV — Template Matching tutorial: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
- OpenCV — `matchTemplate` (imgproc object detection): https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html
