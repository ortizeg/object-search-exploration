# Method — `mosse` (MOSSE/ASEF correlation-filter matching via FFT)

The fast cousin of [`ncc`](ncc.md). Where `ncc` finds rotated repeats with a **brute-force
bank** — one spatial `cv2.matchTemplate` per (scale, angle) pair, 35 correlations by default —
`mosse` **synthesizes a small bank of discriminative correlation filters** from the same
warped-exemplar set and matches each with an **FFT cross-correlation**. The rotation bank is
folded into a few sharp filters via the closed-form MOSSE/ASEF solve (Bolme et al. 2010) instead
of paid one angle at a time, so the correlation cost drops several-fold — decisively on the large
chipset scenes, where the whole method exists to stay affordable.

The module `src/object_search/search/mosse.py` is meant to be read top to bottom; the numbered
steps below match the `# 1.` … `# 9.` comments in `search()` one-for-one (METHOD-11).

## What it is and when it wins

`mosse` is the **fast fixed-scale / near-identical specialist**. On the near-identical repeats
that are `ncc`'s home turf — the chipset (EASY) and textured-plain (TEXTURED) regimes — it reaches
essentially the same F1 (EASY ≈ 0.90, TEXTURED ≈ 0.99) at **6.4× lower median latency** (244 ms vs
1553 ms over the demo set), and on the 6000×4000 chipset the correlation itself is **6× cheaper**
(8.3 s vs 49.5 s). That is the crossover this method makes visible: **the same answer on the easy
regimes, at a fraction of the correlation cost**, because the discriminative filter needs a handful
of FFT passes where the raw-template bank needs one spatial pass per rotation.

It **loses to `ncc` on the transformed regimes** (VARIED, CLUTTERED). Folding a wide rotation range
into a filter — even a small bank of them — cannot be as sharp as correlating each rotated template
separately, and the whitened filter is less discriminative against clutter than a raw normalized
template, so its recall on rotated/scaled instances is lower (VARIED ≈ 0.45, CLUTTERED ≈ 0.61 vs
`ncc`'s 0.46 / 0.77). This is the honest half of the crossover: the correlation-filter speed-up is
bought partly with transformed-instance recall, and the per-regime scoreboard shows exactly where.
See [`../reports/mosse-improvement.md`](../reports/mosse-improvement.md) for the measured iteration.

## Algorithm

### 1. Crop the exemplar and guard against a textureless template

Compute the crop's standard deviation **first**. A flat crop makes the filter degenerate — the
MOSSE denominator collapses to the regularizer and the response is noise — so below `std < 1e-6`
the method returns `outcome=EMPTY` with a diagnostics note rather than emitting confident garbage.
Same guard, same rationale as `ncc` (METHOD-04c).

### 2. Build the scale pyramid

Correlation is shift-invariant but **not** scale-invariant, and a single filter cannot span a wide
scale range, so a pyramid is still required (only the *rotation* bank folds into the filter). As in
`ncc`, the **scene** is rescaled by each factor and the exemplar region is cropped from the
downscaled scene, which keeps detection geometry aligned with the level. Levels whose scaled
template would be `< 8 px` on a side, or larger than the level image, are skipped.

### 3. Synthesize the small MOSSE/ASEF filter bank

The rotation bank is folded into a **small bank of sharp sub-filters** (`n_angle_groups`, default
3), **not** one filter averaged over all seven angles. `_angle_groups` splits the sorted bank into
that many contiguous sub-ranges (7 angles → `[-35,-23.3] / [-11.7,0] / [11.7,23.3,35]`); each
sub-filter is built only from *near* orientations and so stays sharp, while the group set still
spans the whole ±35°. For each warped patch `f_i` and the origin-peaked Gaussian target `g`, the
pooled closed form is solved once:

```
H = (Σ_i G · conj(F_i)) / (Σ_i F_i · conj(F_i) + eps)
```

Pooling numerator and denominator **separately** is what makes this **MOSSE** (numerically stable)
rather than the noise-fragile **ASEF** average-of-exact-filters or the overfit **MACE** (`eps → 0`).
The spatial filter is `k = real(IFFT2(conj(H)))`, then DC-removed (so a scene brightness offset is
rejected — the illumination robustness `ncc` lacks) and unit-normalized. `eps` is `regularization`
× the mean filter energy; larger `eps` broadens the filter toward a plain matched template.

**Why a bank and not one filter:** one filter averaged over ±35° is a blur that matches no single
orientation crisply, which loses the sharp peaks that separate near-identical repeats from their
inter-instance sidelobes (measured: TEXTURED F1 0.93 for one filter vs 0.99 for the 3-filter bank).
Three sharp sub-filters are the measured sweet spot between one blurry filter and `ncc`'s seven
separate spatial passes.

### 4. FFT-correlate the bank over the FULL scene, normalized, combined by max

Each sub-filter is cross-correlated over the **whole** scene with an FFT (`O(H·W·log(H·W))` — the
whole point), and the **scene's FFT is computed once and reused** across the bank (only the small
kernel is re-transformed), because the forward FFT of a 24-megapixel chipset scene dominates the
cost. The numerator is divided by the per-window L2 energy of the mean-subtracted scene — computed
in `O(H·W)` with box filters, **once per scale** — giving a cosine-like response in `[-1, 1]`, the
correlation-filter analogue of `TM_CCOEFF_NORMED`. An `energy_floor_frac` × median-energy floor is
added to the denominator so a flat low-energy region cannot divide a near-zero numerator up into a
spurious `~1.0` peak. The bank's responses are combined by **per-pixel max** (the sub-filters are
alternatives: an instance matches whichever sub-range covers its orientation). **Never** crop the
search region: it would change the FFT and the energy normalization, breaking reproducibility.

### 5. Extract peaks per level (standardised first)

Identical machinery to `ncc`: each level's map is **z-scored against its own median/MAD** before
peaks are compared across levels (a monotone transform, so peak locations are unchanged but the
z-score is comparable across template sizes), then `common.peaks.extract_peaks` picks peaks at
3 σ with the configured strategy. The raw normalized response is carried alongside for the
threshold and candidate log.

### 6. Map peaks to boxes

The response is `(H−h+1, W−w+1)`, top-left anchored — index `(row, col)` maps to a box with
top-left `(col, row)` and the template size, **no centre offset**; at level scale `s` the box
divides back by `s`. Identical geometry to `ncc` (PITFALLS §1.2), and verified against the exemplar
self-match localizing to its own box (the origin-peaked target + `conj(H)` kernel convention).

### 7. Calibrate the threshold

Default **`repeat-aware`**, re-anchored for the correlation filter. Unlike `ncc` (whose exemplar
self-correlates to `~1.0`), the **filter self-response is a lower, image-dependent number** — the
filter is a *whitened* exemplar, not the exemplar. The rule reads the distribution shape against
that self-response:

- Count the **distinct** locations scoring `≥ self_score × 0.85` (NMS-deduplicated — the pyramid
  detects the exemplar's own region several times).
- **≥ 2** such locations ⇒ near-identical repeats ⇒ strict cut `self_score × 0.8`, which rejects the
  diffuse-filter false peaks.
- Otherwise the instances are transformed and score lower ⇒ permissive `self_score × retain_frac`
  (0.5) tail.

The cut is tuned to the **shape** of the score distribution, **never** to the ground-truth boxes,
and the same rule runs on every dataset (the cross-dataset fairness rule). `self-similarity`,
`ratio`, and `gmm` remain as controls; `"fixed"` is used when `threshold` is set. Every branch
returns its **reasoning** as an inspectable diagnostics note.

### 8. Split into matches and sub-threshold candidates

Identical to `ncc`. Peaks whose raw (normalized) score clears the threshold are cross-level greedy
IoU NMS'd (prioritised by the z-score) into the `Match`es — **METHOD-12: every accepted peak
survives, no argmax short-circuit** — and the peak overlapping the exemplar box is labelled
`is_exemplar=True`. The `Candidate` log (EVAL-08) is the sub-threshold peaks kept **with raw
scores** and **deduplicated** (cross-level NMS, dropping any that overlap an accepted match), so
`matches + candidates` form one clean ranked detection set for the offline AP sweep.

### 9. Assemble diagnostics and the result

`Diagnostics` carries the level-1.0 response as a base64 PNG heatmap and `metrics` (crop std, filter
self-score, chosen threshold, the representative peak's **PSR** — MOSSE's native peak-to-sidelobe
confidence, computed once for the strongest peak, not per candidate — and per-level peak counts).
`LatencyBreakdown` is timed around preprocess / **the FFT correlation** / postprocess (EVAL-11), so
the breakdown shows the correlation itself is the fast part even when large-canvas post-processing
dominates the total.

## Pre-processing (exact)

- **Colour:** the BGR scene is converted **once** to single-channel grayscale
  (`cv2.COLOR_BGR2GRAY`) — never per-channel colour correlation.
- **Intensity, filter side:** each training patch is `log1p`-transformed (`log_transform`, the
  MOSSE illumination step — multiplicative lighting becomes an additive offset the DC-free filter
  rejects), then normalized to zero-mean / unit-norm, then multiplied by a 2-D **Hann window**
  (`window`). The window is mandatory because FFT correlation is **circular**: without the taper the
  patch's opposite edges wrap into a discontinuity that pollutes every frequency (Bolme et al. 2010,
  §3).
- **Intensity, scene side:** the scene is `log1p`-transformed into the **same space** as the filter
  (so the normalized response is a cosine in log-space), kept `float64` for the FFT. It is **not**
  windowed — windowing is a per-training-patch operation.
- **Search extent:** the **full** scene, always.

## Post-processing (exact)

- **Normalized response:** valid FFT cross-correlation of the DC-free unit-norm filter, divided by
  the per-window mean-subtracted L2 energy (box filters) with a median-energy floor → a cosine in
  `[-1, 1]`, top-left anchored. The scene FFT and the energy map are computed **once per scale** and
  reused across the bank.
- **Combination:** per-pixel **max** across the sub-filter bank.
- **Cross-level normalization:** per-level z-score against that level's own median/MAD (step 5).
- **Calibration:** `repeat-aware` by default, re-anchored on the filter's (sub-1.0) self-response
  (step 7).
- **Suppression / candidate log:** cross-level greedy IoU NMS over accepted matches; a deduplicated
  sub-threshold candidate log (step 8) — identical to `ncc`.

## Config reference

Generated from `MOSSEConfig`'s JSON Schema — the same schema that drives the UI form — so it cannot
drift from the code.

| field | default | effect |
| --- | --- | --- |
| `scales` | `[0.75, 0.875, 1.0, 1.15, 1.3]` | Pyramid scale factors. Correlation is not scale-invariant, so a pyramid is still needed; the scene is resized by each factor and the filter built from the exemplar cropped from that resized scene. One (bank of) FFT correlation(s) per level. |
| `train_angles_deg` | `[-35, -23.3, -11.7, 0, 11.7, 23.3, 35]` | Rotation bank folded into the filter bank (`ncc`'s angles, but paid via the closed-form solve, not one spatial pass each). |
| `train_scales` | `[1.0]` | Optional scale jitter folded into the filter alongside the rotation bank. `[1.0]` = none (the pyramid handles scale). |
| `n_angle_groups` | `3` | How many sharp sub-filters the rotation bank is split into. 1 = one blurry averaged filter; more = sharper sub-filters and more FFT correlations, but past ~3 they over-sharpen and miss tiny objects. 3 measured best. |
| `output_sigma` | `1.0` | Std (px) of the Gaussian correlation target the filter is solved to produce — the MOSSE sharpness knob. Smaller = a sharper peak (crisper, but misses off-training angles); larger = broader and more forgiving. |
| `regularization` | `0.3` | MOSSE denominator `eps` (relative to the mean filter energy) that stabilises the solve. Larger = broader, more noise-robust, less sharp (toward a plain matched filter); the numerically-stable descendant of MACE. |
| `energy_floor_frac` | `0.3` | Floor added to the local-energy denominator (fraction of the median window energy) so a flat low-energy region cannot divide a near-zero numerator up into a spurious `~1.0` response. |
| `log_transform` | `true` | Apply `log1p` to filter patches and the scene (the MOSSE illumination step). Off = raw intensities (a control). |
| `window` | `true` | Multiply each training patch by a 2-D Hann window before the FFT, so the circular FFT does not wrap an edge discontinuity into the filter. Off is a control that shows the artifact. |
| `threshold` | `null` | Fixed accept threshold on the normalized response. `null` ⇒ use the calibrator. |
| `calibration` | `"repeat-aware"` | How the accept threshold is chosen when `threshold` is `null`. repeat-aware reads the score distribution (strict cut when ≥2 distinct locations sit near the filter's self-response, else the permissive `self × retain_frac` tail), re-anchored because the filter self-response is not 1.0. self-similarity / ratio / gmm are the controls. |
| `peaks` | `"local-max"` | Peak-extraction strategy. local-max (default) separates touching instances that plain nms merges; nms is the control; watershed uses a distance transform. |
| `nms_iou` | `0.3` | IoU above which two accepted boxes are suppressed to one (cross-level NMS); also deduplicates the candidate log. |
| `suppression_radius_frac` | `0.5` | local-max footprint as a fraction of the template size (size-aware). |
| `max_candidates` | `50` | How many top (deduplicated) sub-threshold peaks to keep for the EVAL-08 candidate log. |
| `seed` | `0` | `random_state` for the gmm calibrator (its only genuinely stochastic step; the filter build is deterministic). |
| `retain_frac` | `0.5` | The permissive self-relative accept fraction: keep matches above `self_score × retain_frac`. Used by self-similarity and as the transformed-instance floor by repeat-aware. Tuned to the filter's score distribution, not to the labels. |

## Known failure modes

- **Textureless crop.** A flat exemplar makes the filter degenerate (the denominator is dominated by
  `eps`); the step-1 guard abstains with `outcome=EMPTY` rather than emit noise.
- **Scale beyond the pyramid.** Correlation is not scale-invariant and one filter cannot span a wide
  scale range, so instances scaled past `scales` are missed — the log-polar / Fourier-Mellin front
  end in the ROBUSTNESS BACKLOG is the one-shot scale+rotation alternative.
- **Transformed instances (rotation/scale in clutter).** Folding a wide rotation range into a filter
  bank cannot be as sharp as `ncc`'s per-angle spatial bank, and the whitened filter is less
  discriminative against clutter than a raw normalized template, so recall on the VARIED/CLUTTERED
  regimes is genuinely lower than `ncc`'s — the honest half of the crossover.
- **Sharpness vs generalization.** A sharper filter (smaller `output_sigma` / `regularization`)
  localizes crisply but misses off-training angles; a broader one generalizes but drops precision in
  clutter. This is the genuine knob OTSDF exists for; the defaults sit where the synthetic splits
  measured best.
- **Lighting change** is handled better than `ncc` (the `log1p` + DC-free filter), but a large
  out-of-plane **pose** change still drops the correlation.

## ROBUSTNESS BACKLOG

Deferred deliberately (mirrored verbatim from the module docstring and
`docs/ROBUSTNESS-BACKLOG.md`):

- **Log-polar / Fourier-Mellin front end** so one correlation spans rotation **and** scale, retiring
  the scale pyramid entirely (the rotation bank is already folded into the filter).
- **A dedicated DSST-style scale filter** — a separate 1-D correlation filter over a scale pyramid of
  the peak patch — for continuous scale estimation instead of the discrete pyramid.
- **OTSDF / UMACE variants** exposing an explicit sharpness-vs-noise trade-off parameter, for scenes
  where the MOSSE default is either too sharp (misses poses) or too broad (clutter FPs).
- **Kernelized correlation filters (KCF)** — a non-linear kernel in the closed-form solve, more
  discriminative against structured background than the linear MOSSE filter here.

## Sample runs

Regenerated by `pixi run samples` and committed under [`docs/samples/mosse/`](../samples/mosse/)
(see its [`index.md`](../samples/mosse/index.md) for the per-image outcome table). The renderer
iterates the registry, so `mosse` appears with no per-method code.

## Pseudocode

The steps below mirror the `# 1.` … `# 9.` comments in `search()` (METHOD-11); read
`src/object_search/search/mosse.py` for the ground truth.

```
1. crop <- exemplar region of scene_gray
   if std(crop) < 1e-6: return EMPTY            # flat-template guard (filter degenerates)

2. for s in scales:                             # build the scale pyramid (scene resized, like ncc)
       scene_s    <- resize(scene_gray, s)
       template_s <- crop template FROM scene_s
       skip level if template_s side < 8 px or larger than scene_s

3. groups <- split train_angles_deg into n_angle_groups contiguous sub-ranges
   for each group: build one MOSSE filter k = real(ifft2(conj( ΣG·conj(F_i) / (Σ|F_i|^2 + eps) )))
       (each F_i = fft2 of a log1p + zero-mean + Hann-windowed warp of the crop; g peaked at origin)
       k <- k - mean(k); k <- k / ||k||          # DC-free (brightness-invariant) + unit-norm

4. scene_fft <- fft2(log1p(scene_s))            # ONCE, reused across the bank
   energy    <- sqrt(boxsum(x^2) - boxsum(x)^2/N) + floor        # ONCE per scale
   response  <- max over group of  ifft2(scene_fft · fft2(flip(k)))[valid] / energy   # in [-1,1]

5. z-score response against its OWN median/MAD, then extract peaks at 3 sigma
6. for each peak (row,col): box <- BBox(x=round(col/s), y=round(row/s), w=round(tw/s), h=round(th/s))

7. calibrate (repeat-aware, re-anchored on the filter self-response, NOT 1.0):
   n_near <- # distinct locations with response >= self * 0.85  (NMS-deduped)
   threshold <- self * 0.8 if n_near >= 2 else self * retain_frac

8. peaks whose response >= threshold -> Matches (cross-level NMS; label the exemplar's own region)
   Candidate log = sub-threshold peaks, NMS-deduped and non-overlapping the matches, with raw scores

9. assemble Diagnostics (level-1.0 heatmap, self-score, PSR, metrics) + LatencyBreakdown; return
```

## References

- Bolme, Beveridge, Draper, Lui — *Visual Object Tracking using Adaptive Correlation Filters*
  (MOSSE), CVPR 2010: https://www.cs.colostate.edu/~vision/publications/bolme_cvpr10.pdf
- Mahalanobis, Kumar, Casasent — *Minimum Average Correlation Energy Filters* (MACE), 1987.
- Bolme, Draper, Beveridge — *Average of Synthetic Exact Filters* (ASEF), CVPR 2009.
- Henriques et al. — *High-Speed Tracking with Kernelized Correlation Filters* (KCF), TPAMI 2015.
